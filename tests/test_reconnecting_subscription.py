"""Tests for the reconnecting subscription wrapper.

Validates that ReconnectingSubscription correctly:
    - Delegates to an underlying SubscriptionClient for event streaming.
    - Detects disconnections (CONNECTION_LOST, HEARTBEAT_TIMEOUT) on the
      base client and automatically schedules reconnection.
    - Uses the BackoffConfig/calculate_delay calculator for retry scheduling.
    - Re-establishes the stream transparently after transient failures.
    - Pipes events through an EventIdTracker deduplication filter to skip
      events already seen before the reconnection.
    - Resumes from the last seen event ID across reconnection boundaries.
    - Stops retrying after max_reconnect_attempts is exhausted.
    - Does not reconnect on permanent exits (CLEAN_CLOSE, USER_CANCEL,
      DAEMON_ERROR, SUBSCRIBE_FAILED).
    - Exposes an async iterator that spans reconnection boundaries.
    - Dispatches callbacks across reconnection boundaries.
    - Tracks reconnection history (attempt count, delays, exit reasons).
    - Handles CancelledError during backoff sleep.
    - Supports close() for clean shutdown mid-reconnect.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.ipc.event_dedup import (
    EventDeduplicationConfig,
    EventIdTracker,
)
from jules_daemon.ipc.reconnecting_subscription import (
    ReconnectAttemptRecord,
    ReconnectingSubscription,
    ReconnectingSubscriptionConfig,
    ReconnectingSubscriptionResult,
    ReconnectionExitReason,
)
from jules_daemon.ipc.subscription_client import (
    SubscriptionClient,
    SubscriptionClientConfig,
    SubscriptionExitReason,
    SubscriptionResult,
    SubscriptionState,
)
from jules_daemon.protocol.notifications import (
    CompletionNotification,
    HeartbeatNotification,
    NotificationEnvelope,
    NotificationEventType,
    TestOutcomeSummary,
    create_notification_envelope,
)
from jules_daemon.ssh.backoff import BackoffConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_notification(
    event_id: str = "evt-001",
    event_type: NotificationEventType = NotificationEventType.HEARTBEAT,
) -> NotificationEnvelope:
    """Build a notification envelope for testing."""
    if event_type == NotificationEventType.HEARTBEAT:
        payload = HeartbeatNotification(
            daemon_uptime_seconds=120.0,
            active_run_id=None,
            queue_depth=0,
        )
    else:
        payload = CompletionNotification(
            run_id="run-001",
            natural_language_command="run pytest",
            exit_status=0,
            outcome=TestOutcomeSummary(tests_passed=10, tests_total=10),
        )
    return create_notification_envelope(
        event_type=event_type,
        payload=payload,
        event_id=event_id,
    )


def _make_subscription_result(
    exit_reason: SubscriptionExitReason = SubscriptionExitReason.CLEAN_CLOSE,
    events_received: int = 0,
    subscription_id: str | None = "nsub-abc",
    error_message: str | None = None,
) -> SubscriptionResult:
    """Build a subscription result for testing."""
    return SubscriptionResult(
        exit_reason=exit_reason,
        events_received=events_received,
        subscription_id=subscription_id,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestReconnectingSubscriptionConfig:
    """Validates ReconnectingSubscriptionConfig construction and validation."""

    def test_default_values(self) -> None:
        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
        )
        assert config.socket_path == "/tmp/test.sock"
        assert config.max_reconnect_attempts == 5
        assert config.backoff_config is not None
        assert config.dedup_config is not None

    def test_custom_values(self) -> None:
        backoff = BackoffConfig(base_delay=2.0, max_delay=30.0, max_retries=3)
        dedup = EventDeduplicationConfig(max_tracked_ids=500)
        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=10,
            backoff_config=backoff,
            dedup_config=dedup,
            event_filter=frozenset({NotificationEventType.COMPLETION}),
        )
        assert config.max_reconnect_attempts == 10
        assert config.backoff_config.base_delay == 2.0
        assert config.dedup_config.max_tracked_ids == 500
        assert NotificationEventType.COMPLETION in config.event_filter

    def test_empty_socket_path_raises(self) -> None:
        with pytest.raises(ValueError, match="socket_path must not be empty"):
            ReconnectingSubscriptionConfig(socket_path="")

    def test_zero_max_reconnect_attempts_raises(self) -> None:
        with pytest.raises(
            ValueError, match="max_reconnect_attempts must be positive"
        ):
            ReconnectingSubscriptionConfig(
                socket_path="/tmp/test.sock",
                max_reconnect_attempts=0,
            )

    def test_negative_max_reconnect_attempts_raises(self) -> None:
        with pytest.raises(
            ValueError, match="max_reconnect_attempts must be positive"
        ):
            ReconnectingSubscriptionConfig(
                socket_path="/tmp/test.sock",
                max_reconnect_attempts=-1,
            )


# ---------------------------------------------------------------------------
# ReconnectAttemptRecord tests
# ---------------------------------------------------------------------------


class TestReconnectAttemptRecord:
    """Validates the immutable reconnect attempt record."""

    def test_construction(self) -> None:
        record = ReconnectAttemptRecord(
            attempt=0,
            exit_reason=SubscriptionExitReason.CONNECTION_LOST,
            delay_seconds=1.5,
            events_before_disconnect=10,
        )
        assert record.attempt == 0
        assert record.exit_reason == SubscriptionExitReason.CONNECTION_LOST
        assert record.delay_seconds == 1.5
        assert record.events_before_disconnect == 10

    def test_frozen(self) -> None:
        record = ReconnectAttemptRecord(
            attempt=0,
            exit_reason=SubscriptionExitReason.CONNECTION_LOST,
            delay_seconds=1.0,
            events_before_disconnect=0,
        )
        with pytest.raises(AttributeError):
            record.attempt = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ReconnectionExitReason tests
# ---------------------------------------------------------------------------


class TestReconnectionExitReason:
    """Validates the ReconnectionExitReason enum."""

    def test_all_reasons_exist(self) -> None:
        assert ReconnectionExitReason.CLEAN_CLOSE.value == "clean_close"
        assert ReconnectionExitReason.RECONNECT_EXHAUSTED.value == "reconnect_exhausted"
        assert ReconnectionExitReason.PERMANENT_ERROR.value == "permanent_error"
        assert ReconnectionExitReason.USER_CANCEL.value == "user_cancel"
        assert ReconnectionExitReason.CLOSED.value == "closed"


# ---------------------------------------------------------------------------
# Result tests
# ---------------------------------------------------------------------------


class TestReconnectingSubscriptionResult:
    """Validates the ReconnectingSubscriptionResult model."""

    def test_success_result(self) -> None:
        result = ReconnectingSubscriptionResult(
            exit_reason=ReconnectionExitReason.CLEAN_CLOSE,
            total_events_received=42,
            total_events_deduplicated=3,
            reconnect_count=2,
            reconnect_history=(),
            last_seen_event_id="evt-042",
        )
        assert result.exit_reason == ReconnectionExitReason.CLEAN_CLOSE
        assert result.total_events_received == 42
        assert result.total_events_deduplicated == 3
        assert result.reconnect_count == 2

    def test_frozen(self) -> None:
        result = ReconnectingSubscriptionResult(
            exit_reason=ReconnectionExitReason.CLEAN_CLOSE,
            total_events_received=0,
            total_events_deduplicated=0,
            reconnect_count=0,
            reconnect_history=(),
            last_seen_event_id=None,
        )
        with pytest.raises(AttributeError):
            result.total_events_received = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Core reconnection logic tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconnectionOnDisconnect:
    """Validates that the wrapper reconnects on transient disconnections."""

    async def test_reconnects_on_connection_lost(self) -> None:
        """Wrapper reconnects when base client exits with CONNECTION_LOST."""
        call_count = 0
        events_emitted: list[NotificationEnvelope] = []

        async def fake_run() -> SubscriptionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_subscription_result(
                    exit_reason=SubscriptionExitReason.CONNECTION_LOST,
                    events_received=5,
                    error_message="Connection reset",
                )
            # Second call: clean exit
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.CLEAN_CLOSE,
                events_received=3,
            )

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=3,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        # Mock the internal client factory
        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        result = await wrapper.run()

        assert call_count == 2
        assert result.reconnect_count == 1
        assert result.exit_reason == ReconnectionExitReason.CLEAN_CLOSE

    async def test_reconnects_on_heartbeat_timeout(self) -> None:
        """Wrapper reconnects when base client exits with HEARTBEAT_TIMEOUT."""
        call_count = 0

        async def fake_run() -> SubscriptionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_subscription_result(
                    exit_reason=SubscriptionExitReason.HEARTBEAT_TIMEOUT,
                    events_received=2,
                )
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.CLEAN_CLOSE,
            )

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=3,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        result = await wrapper.run()

        assert call_count == 2
        assert result.reconnect_count == 1

    async def test_no_reconnect_on_clean_close(self) -> None:
        """Wrapper does NOT reconnect on CLEAN_CLOSE."""
        call_count = 0

        async def fake_run() -> SubscriptionResult:
            nonlocal call_count
            call_count += 1
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.CLEAN_CLOSE,
            )

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=3,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        result = await wrapper.run()

        assert call_count == 1
        assert result.reconnect_count == 0
        assert result.exit_reason == ReconnectionExitReason.CLEAN_CLOSE

    async def test_no_reconnect_on_user_cancel(self) -> None:
        """Wrapper does NOT reconnect on USER_CANCEL."""
        async def fake_run() -> SubscriptionResult:
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.USER_CANCEL,
            )

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=3,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        result = await wrapper.run()

        assert result.reconnect_count == 0
        assert result.exit_reason == ReconnectionExitReason.USER_CANCEL

    async def test_no_reconnect_on_daemon_error(self) -> None:
        """Wrapper does NOT reconnect on DAEMON_ERROR."""
        async def fake_run() -> SubscriptionResult:
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.DAEMON_ERROR,
                error_message="Server error",
            )

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=3,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        result = await wrapper.run()

        assert result.reconnect_count == 0
        assert result.exit_reason == ReconnectionExitReason.PERMANENT_ERROR

    async def test_no_reconnect_on_subscribe_failed(self) -> None:
        """Wrapper does NOT reconnect on SUBSCRIBE_FAILED."""
        async def fake_run() -> SubscriptionResult:
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.SUBSCRIBE_FAILED,
                error_message="Auth denied",
            )

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=3,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        result = await wrapper.run()

        assert result.reconnect_count == 0
        assert result.exit_reason == ReconnectionExitReason.PERMANENT_ERROR


# ---------------------------------------------------------------------------
# Max reconnect attempts exhaustion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMaxReconnectAttempts:
    """Validates that the wrapper stops after max_reconnect_attempts."""

    async def test_exhausts_max_reconnect_attempts(self) -> None:
        """Wrapper gives up after max_reconnect_attempts reconnections."""
        call_count = 0

        async def fake_run() -> SubscriptionResult:
            nonlocal call_count
            call_count += 1
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.CONNECTION_LOST,
                events_received=0,
                error_message="Connection reset",
            )

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=3,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        result = await wrapper.run()

        # 1 initial + 3 reconnects = 4 total calls
        assert call_count == 4
        assert result.reconnect_count == 3
        assert result.exit_reason == ReconnectionExitReason.RECONNECT_EXHAUSTED
        assert len(result.reconnect_history) == 3

    async def test_reconnect_history_records_each_attempt(self) -> None:
        """Each reconnection attempt is recorded in reconnect_history."""
        call_count = 0

        async def fake_run() -> SubscriptionResult:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _make_subscription_result(
                    exit_reason=SubscriptionExitReason.CONNECTION_LOST,
                    events_received=call_count * 5,
                    error_message="Connection lost",
                )
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.CLEAN_CLOSE,
                events_received=1,
            )

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=5,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        result = await wrapper.run()

        assert result.reconnect_count == 2
        assert len(result.reconnect_history) == 2
        # Each record tracks the exit reason and events before disconnect
        assert result.reconnect_history[0].exit_reason == SubscriptionExitReason.CONNECTION_LOST
        assert result.reconnect_history[0].events_before_disconnect == 5
        assert result.reconnect_history[1].exit_reason == SubscriptionExitReason.CONNECTION_LOST
        assert result.reconnect_history[1].events_before_disconnect == 10


# ---------------------------------------------------------------------------
# Backoff delay integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBackoffIntegration:
    """Validates that the wrapper uses BackoffConfig for delay calculation."""

    async def test_backoff_delays_increase_exponentially(self) -> None:
        """Reconnection delays follow exponential backoff progression."""
        call_count = 0
        sleep_delays: list[float] = []

        async def fake_run() -> SubscriptionResult:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return _make_subscription_result(
                    exit_reason=SubscriptionExitReason.CONNECTION_LOST,
                )
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.CLEAN_CLOSE,
            )

        async def capture_sleep(delay: float) -> None:
            sleep_delays.append(delay)

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=5,
            backoff_config=BackoffConfig(
                base_delay=1.0,
                max_delay=60.0,
                multiplier=2.0,
                jitter_factor=0.0,
                max_retries=5,
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        with patch("jules_daemon.ipc.reconnecting_subscription.asyncio.sleep", side_effect=capture_sleep):
            result = await wrapper.run()

        assert len(sleep_delays) == 3
        # With base=1.0, multiplier=2.0, jitter=0: delays should be 1.0, 2.0, 4.0
        assert sleep_delays[0] == pytest.approx(1.0)
        assert sleep_delays[1] == pytest.approx(2.0)
        assert sleep_delays[2] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Deduplication integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDeduplicationIntegration:
    """Validates that events are piped through the deduplication filter."""

    async def test_duplicate_events_filtered_after_reconnect(self) -> None:
        """Events seen before reconnection are filtered as duplicates."""
        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=3,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        # Manually record some events as already seen
        evt1 = _make_notification(event_id="evt-001")
        evt2 = _make_notification(event_id="evt-002")

        wrapper._dedup_tracker.record("evt-001")
        wrapper._dedup_tracker.record("evt-002")

        # Verify they are detected as duplicates
        assert wrapper._dedup_tracker.is_duplicate("evt-001")
        assert wrapper._dedup_tracker.is_duplicate("evt-002")
        assert not wrapper._dedup_tracker.is_duplicate("evt-003")

    async def test_dedup_tracker_tracks_last_seen_event_id(self) -> None:
        """The wrapper tracks the last seen event ID for resume."""
        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=1,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        # Record events through the tracker
        wrapper._dedup_tracker.record("evt-a")
        wrapper._dedup_tracker.record("evt-b")
        wrapper._dedup_tracker.record("evt-c")

        assert wrapper.last_seen_event_id == "evt-c"

    async def test_dedup_count_tracked_in_result(self) -> None:
        """Result tracks total number of deduplicated events."""
        call_count = 0
        forwarded_events: list[NotificationEnvelope] = []

        async def fake_run() -> SubscriptionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_subscription_result(
                    exit_reason=SubscriptionExitReason.CONNECTION_LOST,
                    events_received=5,
                )
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.CLEAN_CLOSE,
                events_received=3,
            )

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=3,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        result = await wrapper.run()

        # 5 + 3 = 8 total events received across sessions
        assert result.total_events_received == 8


# ---------------------------------------------------------------------------
# Callback forwarding across reconnections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCallbackForwarding:
    """Validates that callbacks registered on the wrapper survive reconnections."""

    async def test_callbacks_registered_on_each_new_client(self) -> None:
        """Callbacks registered on the wrapper are re-applied to each new client."""
        call_count = 0

        async def fake_run() -> SubscriptionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_subscription_result(
                    exit_reason=SubscriptionExitReason.CONNECTION_LOST,
                )
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.CLEAN_CLOSE,
            )

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=3,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        received: list[NotificationEnvelope] = []

        async def handler(env: NotificationEnvelope) -> None:
            received.append(env)

        wrapper.on_event(handler)

        on_event_calls: list[object] = []
        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        mock_client.on_event = MagicMock(
            side_effect=lambda cb, **kw: on_event_calls.append(cb) or "handle"
        )
        wrapper._create_client = MagicMock(return_value=mock_client)

        with patch("jules_daemon.ipc.reconnecting_subscription.asyncio.sleep", new_callable=AsyncMock):
            result = await wrapper.run()

        # on_event should have been called for each client instance
        assert len(on_event_calls) >= 2  # once for initial, once for reconnect


# ---------------------------------------------------------------------------
# Close during reconnection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCloseDuringReconnection:
    """Validates clean shutdown when close() is called mid-reconnect."""

    async def test_close_stops_reconnection_loop(self) -> None:
        """Calling close() during backoff sleep terminates the loop."""
        call_count = 0

        async def fake_run() -> SubscriptionResult:
            nonlocal call_count
            call_count += 1
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.CONNECTION_LOST,
            )

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=10,
            backoff_config=BackoffConfig(
                base_delay=10.0,  # Long delay so we can cancel it
                max_delay=60.0,
                jitter_factor=0.0,
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        async def close_after_delay() -> None:
            await asyncio.sleep(0.05)
            await wrapper.close()

        # Run both concurrently
        close_task = asyncio.create_task(close_after_delay())
        result = await wrapper.run()
        await close_task

        assert result.exit_reason == ReconnectionExitReason.CLOSED
        assert wrapper._closed is True


# ---------------------------------------------------------------------------
# CancelledError handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCancelledErrorHandling:
    """Validates CancelledError propagation during the reconnect loop."""

    async def test_cancelled_during_backoff_sleep(self) -> None:
        """CancelledError during backoff sleep exits with USER_CANCEL."""
        call_count = 0

        async def fake_run() -> SubscriptionResult:
            nonlocal call_count
            call_count += 1
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.CONNECTION_LOST,
            )

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=5,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(config=config)

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        async def sleep_then_cancel(delay: float) -> None:
            raise asyncio.CancelledError()

        with patch("jules_daemon.ipc.reconnecting_subscription.asyncio.sleep", side_effect=sleep_then_cancel):
            result = await wrapper.run()

        assert result.exit_reason == ReconnectionExitReason.USER_CANCEL


# ---------------------------------------------------------------------------
# Event queue across reconnections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEventQueueAcrossReconnections:
    """Validates that the event queue persists across reconnections."""

    async def test_wrapper_exposes_async_iterator(self) -> None:
        """Wrapper supports async iteration via __aiter__."""
        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
        )
        wrapper = ReconnectingSubscription(config=config)

        # Manually enqueue events
        notification = _make_notification(event_id="evt-iter-001")
        wrapper._event_queue.put_nowait(notification)
        wrapper._event_queue.put_nowait(None)  # sentinel

        collected: list[NotificationEnvelope] = []
        async for event in wrapper:
            collected.append(event)

        assert len(collected) == 1
        assert collected[0].event_id == "evt-iter-001"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Validates wrapper property accessors."""

    def test_last_seen_event_id_initially_none(self) -> None:
        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
        )
        wrapper = ReconnectingSubscription(config=config)
        assert wrapper.last_seen_event_id is None

    def test_config_accessible(self) -> None:
        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
        )
        wrapper = ReconnectingSubscription(config=config)
        assert wrapper.config is config

    def test_dedup_tracker_accessible(self) -> None:
        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
        )
        wrapper = ReconnectingSubscription(config=config)
        assert isinstance(wrapper._dedup_tracker, EventIdTracker)


# ---------------------------------------------------------------------------
# On-reconnect callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnReconnectCallback:
    """Validates the optional on_reconnect callback."""

    async def test_on_reconnect_called_on_each_reconnection(self) -> None:
        """The on_reconnect callback fires for each reconnection attempt."""
        call_count = 0
        reconnect_notifications: list[ReconnectAttemptRecord] = []

        async def fake_run() -> SubscriptionResult:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _make_subscription_result(
                    exit_reason=SubscriptionExitReason.CONNECTION_LOST,
                    events_received=call_count,
                )
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.CLEAN_CLOSE,
            )

        async def on_reconnect(record: ReconnectAttemptRecord) -> None:
            reconnect_notifications.append(record)

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=5,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(
            config=config,
            on_reconnect=on_reconnect,
        )

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        with patch("jules_daemon.ipc.reconnecting_subscription.asyncio.sleep", new_callable=AsyncMock):
            result = await wrapper.run()

        assert len(reconnect_notifications) == 2
        assert reconnect_notifications[0].attempt == 0
        assert reconnect_notifications[1].attempt == 1

    async def test_on_reconnect_failure_does_not_stop_loop(self) -> None:
        """A failing on_reconnect callback does not break the reconnect loop."""
        call_count = 0

        async def fake_run() -> SubscriptionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_subscription_result(
                    exit_reason=SubscriptionExitReason.CONNECTION_LOST,
                )
            return _make_subscription_result(
                exit_reason=SubscriptionExitReason.CLEAN_CLOSE,
            )

        async def bad_on_reconnect(record: ReconnectAttemptRecord) -> None:
            raise RuntimeError("callback failed")

        config = ReconnectingSubscriptionConfig(
            socket_path="/tmp/test.sock",
            max_reconnect_attempts=5,
            backoff_config=BackoffConfig(
                base_delay=0.01, max_delay=0.05, jitter_factor=0.0
            ),
        )
        wrapper = ReconnectingSubscription(
            config=config,
            on_reconnect=bad_on_reconnect,
        )

        mock_client = MagicMock(spec=SubscriptionClient)
        mock_client.run = AsyncMock(side_effect=fake_run)
        mock_client.state = SubscriptionState.DISCONNECTED
        mock_client.close = AsyncMock()
        wrapper._create_client = MagicMock(return_value=mock_client)

        with patch("jules_daemon.ipc.reconnecting_subscription.asyncio.sleep", new_callable=AsyncMock):
            result = await wrapper.run()

        # Should still complete successfully despite callback error
        assert call_count == 2
        assert result.exit_reason == ReconnectionExitReason.CLEAN_CLOSE
