"""Tests for the persistent notification subscription client.

Validates that the SubscriptionClient correctly:
    - Connects to the daemon, performs handshake, and subscribes to
      notifications via a persistent streaming connection.
    - Exposes an async iterator interface for incoming events.
    - Supports callback-based event consumption.
    - Handles daemon heartbeats and resets the heartbeat timeout.
    - Detects connection loss (EOF, connection reset) and transitions
      to DISCONNECTED state.
    - Detects heartbeat timeout (stale connection) and closes cleanly.
    - Supports clean unsubscription and resource cleanup.
    - Provides graceful shutdown via close() and async context manager.
    - Filters events by type when an event_filter is provided.
    - Reports subscription state accurately through lifecycle transitions.
    - Handles malformed envelopes without crashing.
    - Supports auto-reconnect with configurable backoff.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.ipc.framing import (
    MessageEnvelope,
    MessageType,
    encode_frame,
)
from jules_daemon.protocol.notifications import (
    CompletionNotification,
    HeartbeatNotification,
    NotificationEnvelope,
    NotificationEventType,
    TestOutcomeSummary,
    create_notification_envelope,
)
from jules_daemon.ipc.subscription_client import (
    SubscriptionClient,
    SubscriptionClientConfig,
    SubscriptionState,
    SubscriptionExitReason,
    SubscriptionResult,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TS = "2026-04-09T12:00:00+00:00"
_NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers: build IPC envelopes for daemon responses
# ---------------------------------------------------------------------------


def _make_subscribe_response(
    subscription_id: str = "nsub-abc123",
    heartbeat_interval: int = 30,
    msg_id: str = "resp-001",
) -> MessageEnvelope:
    """Build a successful subscribe response envelope."""
    return MessageEnvelope(
        msg_type=MessageType.RESPONSE,
        msg_id=msg_id,
        timestamp=_TS,
        payload={
            "payload_type": "subscribe_notifications_response",
            "subscription_id": subscription_id,
            "heartbeat_interval_seconds": heartbeat_interval,
        },
    )


def _make_notification_envelope(
    event_type: NotificationEventType = NotificationEventType.HEARTBEAT,
    event_id: str = "evt-001",
) -> NotificationEnvelope:
    """Build a notification envelope."""
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
            outcome=TestOutcomeSummary(
                tests_passed=10, tests_total=10
            ),
        )
    return create_notification_envelope(
        event_type=event_type,
        payload=payload,
        event_id=event_id,
    )


def _make_ipc_notification(
    notification: NotificationEnvelope,
    msg_id: str = "notif-001",
) -> MessageEnvelope:
    """Wrap a NotificationEnvelope inside an IPC MessageEnvelope."""
    return MessageEnvelope(
        msg_type=MessageType.STREAM,
        msg_id=msg_id,
        timestamp=_TS,
        payload={
            "verb": "notification",
            "notification": notification.model_dump(mode="json"),
        },
    )


def _make_error_envelope(
    error: str = "Unknown error",
    msg_id: str = "err-001",
) -> MessageEnvelope:
    """Build an ERROR envelope."""
    return MessageEnvelope(
        msg_type=MessageType.ERROR,
        msg_id=msg_id,
        timestamp=_TS,
        payload={"error": error},
    )


def _make_handshake_response(
    protocol_version: int = 1,
    daemon_pid: int = 12345,
    msg_id: str = "hs-resp-001",
) -> MessageEnvelope:
    """Build a successful handshake response."""
    return MessageEnvelope(
        msg_type=MessageType.RESPONSE,
        msg_id=msg_id,
        timestamp=_TS,
        payload={
            "verb": "handshake",
            "status": "ok",
            "protocol_version": protocol_version,
            "daemon_pid": daemon_pid,
            "daemon_uptime_seconds": 100.0,
        },
    )


# ---------------------------------------------------------------------------
# Mock reader helper
# ---------------------------------------------------------------------------


class MockStreamReader:
    """Feed pre-built frames to simulate daemon responses."""

    def __init__(self, frames: list[bytes]) -> None:
        self._buffer = b"".join(frames)
        self._pos = 0

    async def readexactly(self, n: int) -> bytes:
        if self._pos + n > len(self._buffer):
            raise asyncio.IncompleteReadError(
                self._buffer[self._pos:], n
            )
        data = self._buffer[self._pos : self._pos + n]
        self._pos += n
        return data


class MockStreamWriter:
    """Captures written data and provides StreamWriter interface."""

    def __init__(self) -> None:
        self._written: list[bytes] = []
        self._closed = False

    def write(self, data: bytes) -> None:
        self._written.append(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True

    async def wait_closed(self) -> None:
        pass

    def is_closing(self) -> bool:
        return self._closed


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestSubscriptionClientConfig:
    """Validates SubscriptionClientConfig construction and validation."""

    def test_default_values(self) -> None:
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        assert config.socket_path == "/tmp/test.sock"
        assert config.connect_timeout > 0
        assert config.heartbeat_timeout > 0
        assert config.event_filter is None
        assert config.auto_reconnect is False

    def test_custom_values(self) -> None:
        config = SubscriptionClientConfig(
            socket_path="/tmp/test.sock",
            connect_timeout=10.0,
            heartbeat_timeout=90.0,
            event_filter=frozenset({NotificationEventType.COMPLETION}),
            auto_reconnect=True,
            max_reconnect_attempts=5,
            reconnect_base_delay=2.0,
        )
        assert config.connect_timeout == 10.0
        assert config.heartbeat_timeout == 90.0
        assert NotificationEventType.COMPLETION in config.event_filter
        assert config.auto_reconnect is True
        assert config.max_reconnect_attempts == 5
        assert config.reconnect_base_delay == 2.0

    def test_empty_socket_path_raises(self) -> None:
        with pytest.raises(ValueError, match="socket_path must not be empty"):
            SubscriptionClientConfig(socket_path="")

    def test_negative_connect_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="connect_timeout must be positive"):
            SubscriptionClientConfig(
                socket_path="/tmp/test.sock", connect_timeout=-1.0
            )

    def test_zero_heartbeat_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="heartbeat_timeout must be positive"):
            SubscriptionClientConfig(
                socket_path="/tmp/test.sock", heartbeat_timeout=0.0
            )

    def test_negative_reconnect_attempts_raises(self) -> None:
        with pytest.raises(
            ValueError, match="max_reconnect_attempts must not be negative"
        ):
            SubscriptionClientConfig(
                socket_path="/tmp/test.sock", max_reconnect_attempts=-1
            )

    def test_negative_reconnect_delay_raises(self) -> None:
        with pytest.raises(
            ValueError, match="reconnect_base_delay must be positive"
        ):
            SubscriptionClientConfig(
                socket_path="/tmp/test.sock", reconnect_base_delay=-0.5
            )


# ---------------------------------------------------------------------------
# State lifecycle tests
# ---------------------------------------------------------------------------


class TestSubscriptionState:
    """Validates SubscriptionState enum values."""

    def test_all_states_exist(self) -> None:
        assert SubscriptionState.DISCONNECTED.value == "disconnected"
        assert SubscriptionState.CONNECTING.value == "connecting"
        assert SubscriptionState.SUBSCRIBING.value == "subscribing"
        assert SubscriptionState.ACTIVE.value == "active"
        assert SubscriptionState.CLOSING.value == "closing"
        assert SubscriptionState.CLOSED.value == "closed"


# ---------------------------------------------------------------------------
# Exit reason tests
# ---------------------------------------------------------------------------


class TestSubscriptionExitReason:
    """Validates SubscriptionExitReason enum values."""

    def test_all_reasons_exist(self) -> None:
        assert SubscriptionExitReason.CLEAN_CLOSE.value == "clean_close"
        assert SubscriptionExitReason.HEARTBEAT_TIMEOUT.value == "heartbeat_timeout"
        assert SubscriptionExitReason.CONNECTION_LOST.value == "connection_lost"
        assert SubscriptionExitReason.DAEMON_ERROR.value == "daemon_error"
        assert SubscriptionExitReason.USER_CANCEL.value == "user_cancel"
        assert SubscriptionExitReason.SUBSCRIBE_FAILED.value == "subscribe_failed"


# ---------------------------------------------------------------------------
# SubscriptionResult tests
# ---------------------------------------------------------------------------


class TestSubscriptionResult:
    """Validates SubscriptionResult construction."""

    def test_clean_result(self) -> None:
        result = SubscriptionResult(
            exit_reason=SubscriptionExitReason.CLEAN_CLOSE,
            events_received=42,
            subscription_id="nsub-abc",
            error_message=None,
        )
        assert result.exit_reason == SubscriptionExitReason.CLEAN_CLOSE
        assert result.events_received == 42
        assert result.subscription_id == "nsub-abc"
        assert result.error_message is None

    def test_error_result(self) -> None:
        result = SubscriptionResult(
            exit_reason=SubscriptionExitReason.CONNECTION_LOST,
            events_received=5,
            subscription_id=None,
            error_message="Connection reset by peer",
        )
        assert result.exit_reason == SubscriptionExitReason.CONNECTION_LOST
        assert result.error_message == "Connection reset by peer"
        assert result.subscription_id is None


# ---------------------------------------------------------------------------
# Client construction tests
# ---------------------------------------------------------------------------


class TestSubscriptionClientConstruction:
    """Validates SubscriptionClient initialization."""

    def test_initial_state(self) -> None:
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)
        assert client.state == SubscriptionState.DISCONNECTED
        assert client.subscription_id is None
        assert client.events_received == 0

    def test_config_accessible(self) -> None:
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)
        assert client.config is config


# ---------------------------------------------------------------------------
# Callback interface tests
# ---------------------------------------------------------------------------


class TestCallbackInterface:
    """Validates the callback registration and dispatch interface."""

    def test_register_callback(self) -> None:
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        async def handler(env: NotificationEnvelope) -> None:
            pass

        handle = client.on_event(handler)
        assert handle is not None

    def test_register_typed_callback(self) -> None:
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        async def handler(env: NotificationEnvelope) -> None:
            pass

        handle = client.on_event(
            handler, event_types=frozenset({NotificationEventType.COMPLETION})
        )
        assert handle is not None

    def test_remove_callback(self) -> None:
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        async def handler(env: NotificationEnvelope) -> None:
            pass

        handle = client.on_event(handler)
        removed = client.remove_callback(handle)
        assert removed is True

    def test_remove_nonexistent_callback(self) -> None:
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)
        removed = client.remove_callback("nonexistent-handle")
        assert removed is False


# ---------------------------------------------------------------------------
# Async iterator tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncIteratorInterface:
    """Validates the async iterator interface for consuming events."""

    async def test_iter_yields_events_from_queue(self) -> None:
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        # Manually enqueue events for iterator consumption
        notification = _make_notification_envelope(
            event_type=NotificationEventType.COMPLETION,
            event_id="evt-iter-001",
        )
        client._event_queue.put_nowait(notification)
        client._event_queue.put_nowait(None)  # sentinel to stop

        collected: list[NotificationEnvelope] = []
        async for event in client:
            collected.append(event)

        assert len(collected) == 1
        assert collected[0].event_id == "evt-iter-001"

    async def test_iter_stops_on_none_sentinel(self) -> None:
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        # Put just the sentinel
        client._event_queue.put_nowait(None)

        collected: list[NotificationEnvelope] = []
        async for event in client:
            collected.append(event)

        assert len(collected) == 0


# ---------------------------------------------------------------------------
# Connection flow tests (mocked transport)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConnectionFlow:
    """Validates the subscribe/stream/unsubscribe lifecycle with mocked I/O."""

    async def test_connect_sends_subscribe_request(self) -> None:
        """Client sends a subscribe request after handshake."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        # Prepare frames: handshake response, subscribe response
        handshake_resp = _make_handshake_response()
        subscribe_resp = _make_subscribe_response()
        reader = MockStreamReader([
            encode_frame(handshake_resp),
            encode_frame(subscribe_resp),
        ])
        writer = MockStreamWriter()

        result = await client._connect_with_transport(
            reader=reader,  # type: ignore[arg-type]
            writer=writer,  # type: ignore[arg-type]
        )

        assert result is True
        assert client.state == SubscriptionState.ACTIVE
        assert client.subscription_id == "nsub-abc123"
        # Writer should have captured handshake + subscribe frames
        assert len(writer._written) == 2

    async def test_subscribe_failure_returns_false(self) -> None:
        """When daemon rejects subscription, client returns False."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        handshake_resp = _make_handshake_response()
        error_resp = _make_error_envelope(
            error="Subscription denied", msg_id="sub-err"
        )
        reader = MockStreamReader([
            encode_frame(handshake_resp),
            encode_frame(error_resp),
        ])
        writer = MockStreamWriter()

        result = await client._connect_with_transport(
            reader=reader,  # type: ignore[arg-type]
            writer=writer,  # type: ignore[arg-type]
        )

        assert result is False
        assert client.state == SubscriptionState.DISCONNECTED

    async def test_handshake_failure_returns_false(self) -> None:
        """When handshake fails, client returns False."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        error_resp = _make_error_envelope(
            error="Incompatible version", msg_id="hs-err"
        )
        reader = MockStreamReader([encode_frame(error_resp)])
        writer = MockStreamWriter()

        result = await client._connect_with_transport(
            reader=reader,  # type: ignore[arg-type]
            writer=writer,  # type: ignore[arg-type]
        )

        assert result is False
        assert client.state == SubscriptionState.DISCONNECTED


# ---------------------------------------------------------------------------
# Event dispatch tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEventDispatch:
    """Validates that incoming notification events are dispatched to callbacks
    and enqueued for the async iterator."""

    async def test_dispatch_to_callback(self) -> None:
        """Events are dispatched to registered callbacks."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        received: list[NotificationEnvelope] = []

        async def handler(env: NotificationEnvelope) -> None:
            received.append(env)

        client.on_event(handler)

        notification = _make_notification_envelope(
            event_type=NotificationEventType.COMPLETION
        )
        await client._dispatch_event(notification)

        assert len(received) == 1
        assert received[0].event_id == notification.event_id

    async def test_dispatch_filters_by_event_type(self) -> None:
        """Typed callbacks only receive matching event types."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        received: list[NotificationEnvelope] = []

        async def handler(env: NotificationEnvelope) -> None:
            received.append(env)

        client.on_event(
            handler,
            event_types=frozenset({NotificationEventType.COMPLETION}),
        )

        # Send a heartbeat -- should be filtered
        heartbeat = _make_notification_envelope(
            event_type=NotificationEventType.HEARTBEAT
        )
        await client._dispatch_event(heartbeat)
        assert len(received) == 0

        # Send a completion -- should be dispatched
        completion = _make_notification_envelope(
            event_type=NotificationEventType.COMPLETION
        )
        await client._dispatch_event(completion)
        assert len(received) == 1

    async def test_dispatch_enqueues_for_iterator(self) -> None:
        """Events are also enqueued for async iterator consumption."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        notification = _make_notification_envelope()
        await client._dispatch_event(notification)

        assert not client._event_queue.empty()
        queued = client._event_queue.get_nowait()
        assert queued.event_id == notification.event_id

    async def test_callback_error_does_not_stop_dispatch(self) -> None:
        """A failing callback does not prevent other callbacks from running."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        received: list[NotificationEnvelope] = []

        async def bad_handler(env: NotificationEnvelope) -> None:
            raise RuntimeError("handler failed")

        async def good_handler(env: NotificationEnvelope) -> None:
            received.append(env)

        client.on_event(bad_handler)
        client.on_event(good_handler)

        notification = _make_notification_envelope()
        await client._dispatch_event(notification)

        # Good handler still received the event
        assert len(received) == 1


# ---------------------------------------------------------------------------
# Stream processing tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStreamProcessing:
    """Validates the stream loop processes incoming IPC envelopes correctly."""

    async def test_process_notification_envelope(self) -> None:
        """STREAM envelopes with notifications are dispatched."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        notification = _make_notification_envelope(
            event_type=NotificationEventType.COMPLETION
        )
        ipc_envelope = _make_ipc_notification(notification)

        dispatched = await client._process_envelope(ipc_envelope)
        assert dispatched is True
        assert client.events_received == 1

    async def test_process_heartbeat_resets_counter(self) -> None:
        """Heartbeat envelopes reset the heartbeat timeout tracking."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        notification = _make_notification_envelope(
            event_type=NotificationEventType.HEARTBEAT,
        )
        ipc_envelope = _make_ipc_notification(notification)

        dispatched = await client._process_envelope(ipc_envelope)
        assert dispatched is True
        assert client._last_heartbeat is not None

    async def test_process_error_envelope_returns_false(self) -> None:
        """ERROR envelopes cause stream processing to signal termination."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        error_envelope = _make_error_envelope(error="Server shutting down")
        dispatched = await client._process_envelope(error_envelope)
        assert dispatched is False


# ---------------------------------------------------------------------------
# Close / cleanup tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCloseAndCleanup:
    """Validates clean shutdown behavior."""

    async def test_close_transitions_to_closed(self) -> None:
        """close() transitions to CLOSED state."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        await client.close()
        assert client.state == SubscriptionState.CLOSED

    async def test_close_is_idempotent(self) -> None:
        """Calling close() multiple times is safe."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        await client.close()
        await client.close()
        assert client.state == SubscriptionState.CLOSED

    async def test_close_posts_sentinel_to_queue(self) -> None:
        """close() posts a None sentinel so async iterators stop."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        await client.close()
        sentinel = client._event_queue.get_nowait()
        assert sentinel is None

    async def test_context_manager_close(self) -> None:
        """async with SubscriptionClient closes on exit."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        async with client:
            assert client.state in (
                SubscriptionState.DISCONNECTED,
                SubscriptionState.CLOSED,
            )

        assert client.state == SubscriptionState.CLOSED


# ---------------------------------------------------------------------------
# Envelope send/receive tests (internal helpers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEnvelopeSendReceive:
    """Validates internal envelope send/receive helpers."""

    async def test_send_envelope_writes_frame(self) -> None:
        """_send_envelope encodes and writes a framed envelope."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)
        writer = MockStreamWriter()

        envelope = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="test-001",
            timestamp=_TS,
            payload={"verb": "test"},
        )

        await client._send_envelope(writer, envelope)
        assert len(writer._written) == 1
        assert len(writer._written[0]) > 0

    async def test_read_envelope_from_mock_reader(self) -> None:
        """_read_envelope reads and decodes a framed envelope."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        original = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="test-read-001",
            timestamp=_TS,
            payload={"status": "ok"},
        )
        reader = MockStreamReader([encode_frame(original)])

        result = await client._read_envelope(reader)
        assert result is not None
        assert result.msg_id == "test-read-001"
        assert result.payload["status"] == "ok"

    async def test_read_envelope_returns_none_for_none_reader(self) -> None:
        """_read_envelope returns None when reader is None."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        result = await client._read_envelope(None)
        assert result is None

    async def test_read_envelope_returns_none_on_eof(self) -> None:
        """_read_envelope returns None when reader hits EOF."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        # Empty reader -> EOF
        reader = MockStreamReader([])
        result = await client._read_envelope(reader)
        assert result is None


# ---------------------------------------------------------------------------
# Unsubscribe best-effort tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUnsubscribeBestEffort:
    """Validates the best-effort unsubscribe on close."""

    async def test_unsubscribe_sends_when_subscribed(self) -> None:
        """Unsubscribe is sent when subscription_id is set."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)
        writer = MockStreamWriter()
        client._writer = writer  # type: ignore[assignment]
        client._subscription_id = "nsub-test"

        await client._try_send_unsubscribe()

        # Should have written one frame
        assert len(writer._written) == 1

    async def test_unsubscribe_skipped_when_no_subscription(self) -> None:
        """Unsubscribe is skipped when there's no subscription_id."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)
        writer = MockStreamWriter()
        client._writer = writer  # type: ignore[assignment]

        await client._try_send_unsubscribe()

        # Should NOT have written anything
        assert len(writer._written) == 0

    async def test_unsubscribe_skipped_when_no_writer(self) -> None:
        """Unsubscribe is skipped when writer is None."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)
        client._subscription_id = "nsub-test"

        # Should not raise
        await client._try_send_unsubscribe()


# ---------------------------------------------------------------------------
# Transport cleanup tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTransportCleanup:
    """Validates transport close behavior."""

    async def test_close_transport_with_writer(self) -> None:
        """_close_transport closes the writer and clears references."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)
        writer = MockStreamWriter()
        client._writer = writer  # type: ignore[assignment]
        client._reader = MockStreamReader([])  # type: ignore[assignment]

        await client._close_transport()

        assert client._writer is None
        assert client._reader is None
        assert writer._closed is True

    async def test_close_transport_with_none_writer(self) -> None:
        """_close_transport is safe when writer is None."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        await client._close_transport()

        assert client._writer is None

    async def test_close_transport_handles_already_closing(self) -> None:
        """_close_transport handles an already-closing writer."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)
        writer = MockStreamWriter()
        writer._closed = True  # Mark as already closing
        client._writer = writer  # type: ignore[assignment]

        await client._close_transport()

        assert client._writer is None


# ---------------------------------------------------------------------------
# Process non-STREAM envelope tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProcessNonStreamEnvelope:
    """Validates that non-STREAM, non-ERROR envelopes are ignored."""

    async def test_response_envelope_ignored(self) -> None:
        """RESPONSE envelopes are silently ignored (stale response)."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        envelope = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="stale-001",
            timestamp=_TS,
            payload={"status": "ok"},
        )

        result = await client._process_envelope(envelope)
        assert result is True
        assert client.events_received == 0

    async def test_malformed_notification_handled(self) -> None:
        """STREAM envelope with bad notification data is handled gracefully."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        envelope = MessageEnvelope(
            msg_type=MessageType.STREAM,
            msg_id="bad-001",
            timestamp=_TS,
            payload={"verb": "notification", "notification": {"bad": "data"}},
        )

        result = await client._process_envelope(envelope)
        assert result is True  # Should not crash, just skip
        assert client.events_received == 0

    async def test_missing_notification_key_handled(self) -> None:
        """STREAM envelope without notification key is handled gracefully."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        envelope = MessageEnvelope(
            msg_type=MessageType.STREAM,
            msg_id="missing-001",
            timestamp=_TS,
            payload={"verb": "notification"},
        )

        result = await client._process_envelope(envelope)
        assert result is True  # Should not crash, just skip
        assert client.events_received == 0


# ---------------------------------------------------------------------------
# Handshake internal tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandshakeInternal:
    """Validates the internal handshake method."""

    async def test_handshake_version_mismatch(self) -> None:
        """Handshake fails on protocol version mismatch."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        response = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="hs-resp",
            timestamp=_TS,
            payload={
                "verb": "handshake",
                "status": "ok",
                "protocol_version": 999,
                "daemon_pid": 1234,
                "daemon_uptime_seconds": 10.0,
            },
        )
        reader = MockStreamReader([encode_frame(response)])
        writer = MockStreamWriter()

        result = await client._perform_handshake(
            reader,  # type: ignore[arg-type]
            writer,  # type: ignore[arg-type]
        )
        assert result is False

    async def test_handshake_no_response(self) -> None:
        """Handshake fails when daemon sends no response."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        reader = MockStreamReader([])  # EOF immediately
        writer = MockStreamWriter()

        result = await client._perform_handshake(
            reader,  # type: ignore[arg-type]
            writer,  # type: ignore[arg-type]
        )
        assert result is False


# ---------------------------------------------------------------------------
# Subscribe internal tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSubscribeInternal:
    """Validates the internal subscribe method."""

    async def test_subscribe_missing_subscription_id(self) -> None:
        """Subscribe fails when response lacks subscription_id."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        response = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="sub-resp",
            timestamp=_TS,
            payload={
                "payload_type": "subscribe_notifications_response",
                # Missing subscription_id
                "heartbeat_interval_seconds": 30,
            },
        )
        reader = MockStreamReader([encode_frame(response)])
        writer = MockStreamWriter()

        result = await client._send_subscribe(
            reader,  # type: ignore[arg-type]
            writer,  # type: ignore[arg-type]
        )
        assert result is False

    async def test_subscribe_with_event_filter(self) -> None:
        """Subscribe sends event_filter when configured."""
        config = SubscriptionClientConfig(
            socket_path="/tmp/test.sock",
            event_filter=frozenset({NotificationEventType.COMPLETION}),
        )
        client = SubscriptionClient(config=config)

        response = _make_subscribe_response()
        reader = MockStreamReader([encode_frame(response)])
        writer = MockStreamWriter()

        result = await client._send_subscribe(
            reader,  # type: ignore[arg-type]
            writer,  # type: ignore[arg-type]
        )
        assert result is True
        assert client.subscription_id == "nsub-abc123"

    async def test_subscribe_no_response(self) -> None:
        """Subscribe fails when daemon sends no response."""
        config = SubscriptionClientConfig(socket_path="/tmp/test.sock")
        client = SubscriptionClient(config=config)

        reader = MockStreamReader([])  # EOF immediately
        writer = MockStreamWriter()

        result = await client._send_subscribe(
            reader,  # type: ignore[arg-type]
            writer,  # type: ignore[arg-type]
        )
        assert result is False


# ---------------------------------------------------------------------------
# Event queue overflow tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEventQueueOverflow:
    """Validates event queue eviction on overflow."""

    async def test_queue_evicts_oldest_on_overflow(self) -> None:
        """When queue is full, oldest event is evicted."""
        config = SubscriptionClientConfig(
            socket_path="/tmp/test.sock",
            event_queue_size=2,
        )
        client = SubscriptionClient(config=config)

        # Fill the queue
        evt1 = _make_notification_envelope(event_id="evt-1")
        evt2 = _make_notification_envelope(event_id="evt-2")
        evt3 = _make_notification_envelope(event_id="evt-3")

        await client._dispatch_event(evt1)
        await client._dispatch_event(evt2)
        await client._dispatch_event(evt3)

        # Queue should have evt2 and evt3 (evt1 evicted)
        first = client._event_queue.get_nowait()
        second = client._event_queue.get_nowait()
        assert first.event_id == "evt-2"
        assert second.event_id == "evt-3"
