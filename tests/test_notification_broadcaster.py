"""Tests for the notification broadcaster.

Covers NotificationBroadcaster subscriber lifecycle (register, unregister,
thread-safe map management), event delivery with filtering, queue
management, per-subscriber error handling, fan-out broadcast results,
auto-removal of dead subscribers, and configuration validation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from jules_daemon.protocol.notifications import (
    AlertNotification,
    CompletionNotification,
    HeartbeatNotification,
    NotificationEventType,
    NotificationSeverity,
    create_notification_envelope,
)


# ---------------------------------------------------------------------------
# Helpers -- deferred imports resolved after implementation exists
# ---------------------------------------------------------------------------

from jules_daemon.ipc.notification_broadcaster import (
    BroadcastResult,
    NotificationBroadcaster,
    NotificationBroadcasterConfig,
    NotificationSubscriberHandle,
    SubscriberSendError,
)


def _make_completion_envelope(
    run_id: str = "run-001",
) -> "jules_daemon.protocol.notifications.NotificationEnvelope":
    """Build a completion NotificationEnvelope for testing."""
    payload = CompletionNotification(
        run_id=run_id,
        natural_language_command="Run pytest",
        exit_status=0,
    )
    return create_notification_envelope(
        event_type=NotificationEventType.COMPLETION,
        payload=payload,
    )


def _make_alert_envelope(
    title: str = "Test alert",
) -> "jules_daemon.protocol.notifications.NotificationEnvelope":
    """Build an alert NotificationEnvelope for testing."""
    payload = AlertNotification(
        severity=NotificationSeverity.WARNING,
        title=title,
        message="Alert detail",
    )
    return create_notification_envelope(
        event_type=NotificationEventType.ALERT,
        payload=payload,
    )


def _make_heartbeat_envelope(
    uptime: float = 100.0,
) -> "jules_daemon.protocol.notifications.NotificationEnvelope":
    """Build a heartbeat NotificationEnvelope for testing."""
    payload = HeartbeatNotification(daemon_uptime_seconds=uptime)
    return create_notification_envelope(
        event_type=NotificationEventType.HEARTBEAT,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# NotificationBroadcasterConfig
# ---------------------------------------------------------------------------


class TestNotificationBroadcasterConfig:
    """Configuration validation tests."""

    def test_defaults(self) -> None:
        config = NotificationBroadcasterConfig()
        assert config.subscriber_queue_size == 100
        assert config.heartbeat_interval_seconds == 30

    def test_custom_values(self) -> None:
        config = NotificationBroadcasterConfig(
            subscriber_queue_size=50,
            heartbeat_interval_seconds=60,
        )
        assert config.subscriber_queue_size == 50
        assert config.heartbeat_interval_seconds == 60

    def test_frozen(self) -> None:
        config = NotificationBroadcasterConfig()
        with pytest.raises(AttributeError):
            config.subscriber_queue_size = 42  # type: ignore[misc]

    def test_zero_queue_size_raises(self) -> None:
        with pytest.raises(ValueError, match="subscriber_queue_size must be positive"):
            NotificationBroadcasterConfig(subscriber_queue_size=0)

    def test_negative_queue_size_raises(self) -> None:
        with pytest.raises(ValueError, match="subscriber_queue_size must be positive"):
            NotificationBroadcasterConfig(subscriber_queue_size=-1)

    def test_zero_heartbeat_interval_raises(self) -> None:
        with pytest.raises(
            ValueError, match="heartbeat_interval_seconds must be positive"
        ):
            NotificationBroadcasterConfig(heartbeat_interval_seconds=0)

    def test_negative_heartbeat_interval_raises(self) -> None:
        with pytest.raises(
            ValueError, match="heartbeat_interval_seconds must be positive"
        ):
            NotificationBroadcasterConfig(heartbeat_interval_seconds=-1)


# ---------------------------------------------------------------------------
# NotificationSubscriberHandle
# ---------------------------------------------------------------------------


class TestNotificationSubscriberHandle:
    """Subscriber handle immutability and construction tests."""

    def test_create(self) -> None:
        handle = NotificationSubscriberHandle(
            subscription_id="sub-abc",
            event_filter=None,
        )
        assert handle.subscription_id == "sub-abc"
        assert handle.event_filter is None

    def test_with_event_filter(self) -> None:
        event_filter = frozenset(
            {NotificationEventType.COMPLETION, NotificationEventType.ALERT}
        )
        handle = NotificationSubscriberHandle(
            subscription_id="sub-xyz",
            event_filter=event_filter,
        )
        assert handle.event_filter is not None
        assert NotificationEventType.COMPLETION in handle.event_filter
        assert NotificationEventType.ALERT in handle.event_filter
        assert NotificationEventType.HEARTBEAT not in handle.event_filter

    def test_frozen(self) -> None:
        handle = NotificationSubscriberHandle(
            subscription_id="sub-abc",
            event_filter=None,
        )
        with pytest.raises(AttributeError):
            handle.subscription_id = "mutated"  # type: ignore[misc]

    def test_empty_subscription_id_raises(self) -> None:
        with pytest.raises(ValueError, match="subscription_id must not be empty"):
            NotificationSubscriberHandle(subscription_id="", event_filter=None)

    def test_whitespace_subscription_id_raises(self) -> None:
        with pytest.raises(ValueError, match="subscription_id must not be empty"):
            NotificationSubscriberHandle(subscription_id="   ", event_filter=None)


# ---------------------------------------------------------------------------
# NotificationBroadcaster -- subscribe
# ---------------------------------------------------------------------------


class TestSubscribe:
    """Subscriber registration tests."""

    @pytest.mark.asyncio
    async def test_subscribe_returns_handle(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()
        assert isinstance(handle, NotificationSubscriberHandle)
        assert handle.subscription_id.startswith("nsub-")
        assert handle.event_filter is None

    @pytest.mark.asyncio
    async def test_subscribe_with_event_filter(self) -> None:
        broadcaster = NotificationBroadcaster()
        event_filter = frozenset({NotificationEventType.COMPLETION})
        handle = await broadcaster.subscribe(event_filter=event_filter)
        assert handle.event_filter == frozenset({NotificationEventType.COMPLETION})

    @pytest.mark.asyncio
    async def test_subscribe_increments_count(self) -> None:
        broadcaster = NotificationBroadcaster()
        assert broadcaster.subscriber_count == 0
        await broadcaster.subscribe()
        assert broadcaster.subscriber_count == 1
        await broadcaster.subscribe()
        assert broadcaster.subscriber_count == 2

    @pytest.mark.asyncio
    async def test_subscribe_generates_unique_ids(self) -> None:
        broadcaster = NotificationBroadcaster()
        h1 = await broadcaster.subscribe()
        h2 = await broadcaster.subscribe()
        h3 = await broadcaster.subscribe()
        ids = {h1.subscription_id, h2.subscription_id, h3.subscription_id}
        assert len(ids) == 3

    @pytest.mark.asyncio
    async def test_has_subscriber_after_subscribe(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()
        assert broadcaster.has_subscriber(handle.subscription_id)

    @pytest.mark.asyncio
    async def test_has_subscriber_false_for_unknown(self) -> None:
        broadcaster = NotificationBroadcaster()
        assert not broadcaster.has_subscriber("nonexistent")


# ---------------------------------------------------------------------------
# NotificationBroadcaster -- unsubscribe
# ---------------------------------------------------------------------------


class TestUnsubscribe:
    """Subscriber unregistration tests."""

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_subscriber(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()
        removed = await broadcaster.unsubscribe(handle.subscription_id)
        assert removed is True
        assert not broadcaster.has_subscriber(handle.subscription_id)
        assert broadcaster.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_unknown_returns_false(self) -> None:
        broadcaster = NotificationBroadcaster()
        removed = await broadcaster.unsubscribe("nonexistent")
        assert removed is False

    @pytest.mark.asyncio
    async def test_unsubscribe_idempotent(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()
        first = await broadcaster.unsubscribe(handle.subscription_id)
        second = await broadcaster.unsubscribe(handle.subscription_id)
        assert first is True
        assert second is False

    @pytest.mark.asyncio
    async def test_unsubscribe_does_not_affect_others(self) -> None:
        broadcaster = NotificationBroadcaster()
        h1 = await broadcaster.subscribe()
        h2 = await broadcaster.subscribe()
        await broadcaster.unsubscribe(h1.subscription_id)
        assert not broadcaster.has_subscriber(h1.subscription_id)
        assert broadcaster.has_subscriber(h2.subscription_id)
        assert broadcaster.subscriber_count == 1


# ---------------------------------------------------------------------------
# NotificationBroadcaster -- broadcast
# ---------------------------------------------------------------------------


class TestBroadcast:
    """Event delivery and filtering tests."""

    @pytest.mark.asyncio
    async def test_broadcast_to_single_subscriber(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()
        envelope = _make_completion_envelope()
        result = await broadcaster.broadcast(envelope)
        assert isinstance(result, BroadcastResult)
        assert result.delivered_count == 1
        assert result.error_count == 0
        assert result.filtered_count == 0
        assert not result.has_errors
        received = await broadcaster.receive(
            handle.subscription_id, timeout=1.0
        )
        assert received is not None
        assert received.event_id == envelope.event_id

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_subscribers(self) -> None:
        broadcaster = NotificationBroadcaster()
        h1 = await broadcaster.subscribe()
        h2 = await broadcaster.subscribe()
        h3 = await broadcaster.subscribe()
        envelope = _make_alert_envelope()
        result = await broadcaster.broadcast(envelope)
        assert result.delivered_count == 3
        assert result.total_subscribers == 3

        for handle in (h1, h2, h3):
            received = await broadcaster.receive(
                handle.subscription_id, timeout=1.0
            )
            assert received is not None
            assert received.event_id == envelope.event_id

    @pytest.mark.asyncio
    async def test_broadcast_with_event_filter_match(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe(
            event_filter=frozenset({NotificationEventType.COMPLETION})
        )
        envelope = _make_completion_envelope()
        result = await broadcaster.broadcast(envelope)
        assert result.delivered_count == 1
        assert result.filtered_count == 0
        received = await broadcaster.receive(
            handle.subscription_id, timeout=1.0
        )
        assert received is not None

    @pytest.mark.asyncio
    async def test_broadcast_with_event_filter_no_match(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe(
            event_filter=frozenset({NotificationEventType.COMPLETION})
        )
        envelope = _make_alert_envelope()
        result = await broadcaster.broadcast(envelope)
        assert result.delivered_count == 0
        assert result.filtered_count == 1
        received = await broadcaster.receive(
            handle.subscription_id, timeout=0.05
        )
        assert received is None

    @pytest.mark.asyncio
    async def test_broadcast_mixed_filters(self) -> None:
        """Subscribers with different filters receive only matching events."""
        broadcaster = NotificationBroadcaster()
        h_all = await broadcaster.subscribe()  # receives everything
        h_completion = await broadcaster.subscribe(
            event_filter=frozenset({NotificationEventType.COMPLETION})
        )
        h_alert = await broadcaster.subscribe(
            event_filter=frozenset({NotificationEventType.ALERT})
        )

        completion_env = _make_completion_envelope()
        alert_env = _make_alert_envelope()

        result1 = await broadcaster.broadcast(completion_env)
        result2 = await broadcaster.broadcast(alert_env)

        # completion: h_all + h_completion delivered, h_alert filtered
        assert result1.delivered_count == 2
        assert result1.filtered_count == 1

        # alert: h_all + h_alert delivered, h_completion filtered
        assert result2.delivered_count == 2
        assert result2.filtered_count == 1

        # h_all gets both
        r1 = await broadcaster.receive(h_all.subscription_id, timeout=1.0)
        r2 = await broadcaster.receive(h_all.subscription_id, timeout=1.0)
        assert r1 is not None and r2 is not None

        # h_completion gets only completion
        r_comp = await broadcaster.receive(
            h_completion.subscription_id, timeout=1.0
        )
        assert r_comp is not None
        assert r_comp.event_type is NotificationEventType.COMPLETION
        r_none = await broadcaster.receive(
            h_completion.subscription_id, timeout=0.05
        )
        assert r_none is None

        # h_alert gets only alert
        r_alert = await broadcaster.receive(
            h_alert.subscription_id, timeout=1.0
        )
        assert r_alert is not None
        assert r_alert.event_type is NotificationEventType.ALERT
        r_none2 = await broadcaster.receive(
            h_alert.subscription_id, timeout=0.05
        )
        assert r_none2 is None

    @pytest.mark.asyncio
    async def test_broadcast_to_no_subscribers_returns_zero(self) -> None:
        broadcaster = NotificationBroadcaster()
        envelope = _make_heartbeat_envelope()
        result = await broadcaster.broadcast(envelope)
        assert result.delivered_count == 0
        assert result.total_subscribers == 0

    @pytest.mark.asyncio
    async def test_broadcast_skips_removed_subscriber(self) -> None:
        broadcaster = NotificationBroadcaster()
        h1 = await broadcaster.subscribe()
        h2 = await broadcaster.subscribe()
        await broadcaster.unsubscribe(h1.subscription_id)
        envelope = _make_completion_envelope()
        result = await broadcaster.broadcast(envelope)
        assert result.delivered_count == 1
        received = await broadcaster.receive(
            h2.subscription_id, timeout=1.0
        )
        assert received is not None


# ---------------------------------------------------------------------------
# NotificationBroadcaster -- receive
# ---------------------------------------------------------------------------


class TestReceive:
    """Queue receive behavior tests."""

    @pytest.mark.asyncio
    async def test_receive_returns_none_on_timeout(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()
        result = await broadcaster.receive(
            handle.subscription_id, timeout=0.05
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_receive_unknown_subscriber_raises(self) -> None:
        broadcaster = NotificationBroadcaster()
        with pytest.raises(ValueError, match="not found"):
            await broadcaster.receive("nonexistent", timeout=0.05)

    @pytest.mark.asyncio
    async def test_receive_preserves_fifo_order(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()

        env1 = _make_completion_envelope(run_id="run-001")
        env2 = _make_alert_envelope(title="Second")
        env3 = _make_heartbeat_envelope(uptime=200.0)

        await broadcaster.broadcast(env1)
        await broadcaster.broadcast(env2)
        await broadcaster.broadcast(env3)

        r1 = await broadcaster.receive(handle.subscription_id, timeout=1.0)
        r2 = await broadcaster.receive(handle.subscription_id, timeout=1.0)
        r3 = await broadcaster.receive(handle.subscription_id, timeout=1.0)

        assert r1 is not None and r1.event_id == env1.event_id
        assert r2 is not None and r2.event_id == env2.event_id
        assert r3 is not None and r3.event_id == env3.event_id

    @pytest.mark.asyncio
    async def test_receive_after_unsubscribe_raises(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()
        await broadcaster.unsubscribe(handle.subscription_id)
        with pytest.raises(ValueError, match="not found"):
            await broadcaster.receive(handle.subscription_id, timeout=0.05)


# ---------------------------------------------------------------------------
# NotificationBroadcaster -- backpressure
# ---------------------------------------------------------------------------


class TestBackpressure:
    """Queue overflow and backpressure handling tests."""

    @pytest.mark.asyncio
    async def test_full_queue_evicts_oldest(self) -> None:
        """When a subscriber queue is full, the oldest entry is evicted."""
        config = NotificationBroadcasterConfig(subscriber_queue_size=2)
        broadcaster = NotificationBroadcaster(config=config)
        handle = await broadcaster.subscribe()

        env1 = _make_completion_envelope(run_id="run-1")
        env2 = _make_alert_envelope(title="Alert 2")
        env3 = _make_heartbeat_envelope(uptime=300.0)

        await broadcaster.broadcast(env1)
        await broadcaster.broadcast(env2)
        await broadcaster.broadcast(env3)  # should evict env1

        r1 = await broadcaster.receive(handle.subscription_id, timeout=1.0)
        r2 = await broadcaster.receive(handle.subscription_id, timeout=1.0)

        assert r1 is not None and r1.event_id == env2.event_id
        assert r2 is not None and r2.event_id == env3.event_id

        # No more items
        r3 = await broadcaster.receive(handle.subscription_id, timeout=0.05)
        assert r3 is None


# ---------------------------------------------------------------------------
# NotificationBroadcaster -- remove_all
# ---------------------------------------------------------------------------


class TestRemoveAll:
    """Bulk subscriber removal tests."""

    @pytest.mark.asyncio
    async def test_remove_all_clears_all_subscribers(self) -> None:
        broadcaster = NotificationBroadcaster()
        await broadcaster.subscribe()
        await broadcaster.subscribe()
        await broadcaster.subscribe()
        assert broadcaster.subscriber_count == 3
        removed_count = await broadcaster.remove_all()
        assert removed_count == 3
        assert broadcaster.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_remove_all_empty_returns_zero(self) -> None:
        broadcaster = NotificationBroadcaster()
        removed_count = await broadcaster.remove_all()
        assert removed_count == 0


# ---------------------------------------------------------------------------
# NotificationBroadcaster -- get_subscriber_handle
# ---------------------------------------------------------------------------


class TestGetSubscriberHandle:
    """Handle lookup tests."""

    @pytest.mark.asyncio
    async def test_get_existing_handle(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()
        looked_up = broadcaster.get_subscriber_handle(handle.subscription_id)
        assert looked_up is not None
        assert looked_up.subscription_id == handle.subscription_id
        assert looked_up.event_filter == handle.event_filter

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self) -> None:
        broadcaster = NotificationBroadcaster()
        assert broadcaster.get_subscriber_handle("nonexistent") is None


# ---------------------------------------------------------------------------
# NotificationBroadcaster -- thread safety (concurrent operations)
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Tests for thread-safe subscriber map management."""

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_unsubscribe(self) -> None:
        """Multiple concurrent subscribe/unsubscribe calls do not corrupt state."""
        broadcaster = NotificationBroadcaster()

        async def subscribe_and_unsubscribe() -> None:
            handle = await broadcaster.subscribe()
            await asyncio.sleep(0)  # yield to event loop
            await broadcaster.unsubscribe(handle.subscription_id)

        tasks = [subscribe_and_unsubscribe() for _ in range(20)]
        await asyncio.gather(*tasks)
        assert broadcaster.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_broadcast_and_subscribe(self) -> None:
        """Broadcasting while subscribing does not raise."""
        broadcaster = NotificationBroadcaster()
        handles: list[NotificationSubscriberHandle] = []

        async def subscribe_task() -> None:
            for _ in range(5):
                handle = await broadcaster.subscribe()
                handles.append(handle)
                await asyncio.sleep(0)

        async def broadcast_task() -> None:
            for _ in range(5):
                env = _make_heartbeat_envelope()
                await broadcaster.broadcast(env)
                await asyncio.sleep(0)

        await asyncio.gather(subscribe_task(), broadcast_task())
        # No assertion on exact counts -- the point is no corruption or exception

    @pytest.mark.asyncio
    async def test_list_subscriber_ids_returns_snapshot(self) -> None:
        """list_subscriber_ids returns a frozen snapshot of current state."""
        broadcaster = NotificationBroadcaster()
        h1 = await broadcaster.subscribe()
        h2 = await broadcaster.subscribe()
        ids = broadcaster.list_subscriber_ids()
        assert isinstance(ids, frozenset)
        assert h1.subscription_id in ids
        assert h2.subscription_id in ids


# ---------------------------------------------------------------------------
# SubscriberSendError model
# ---------------------------------------------------------------------------


class TestSubscriberSendError:
    """SubscriberSendError immutability and construction tests."""

    def test_create(self) -> None:
        error = SubscriberSendError(
            subscription_id="sub-abc",
            error_type="queue_full",
            error_message="Queue is full",
            consecutive_failures=3,
        )
        assert error.subscription_id == "sub-abc"
        assert error.error_type == "queue_full"
        assert error.error_message == "Queue is full"
        assert error.consecutive_failures == 3

    def test_frozen(self) -> None:
        error = SubscriberSendError(
            subscription_id="sub-abc",
            error_type="queue_full",
            error_message="msg",
            consecutive_failures=1,
        )
        with pytest.raises(AttributeError):
            error.subscription_id = "mutated"  # type: ignore[misc]

    def test_empty_subscription_id_raises(self) -> None:
        with pytest.raises(ValueError, match="subscription_id must not be empty"):
            SubscriberSendError(
                subscription_id="",
                error_type="queue_full",
                error_message="msg",
                consecutive_failures=1,
            )

    def test_whitespace_subscription_id_raises(self) -> None:
        with pytest.raises(ValueError, match="subscription_id must not be empty"):
            SubscriberSendError(
                subscription_id="   ",
                error_type="queue_full",
                error_message="msg",
                consecutive_failures=1,
            )

    def test_empty_error_type_raises(self) -> None:
        with pytest.raises(ValueError, match="error_type must not be empty"):
            SubscriberSendError(
                subscription_id="sub-abc",
                error_type="",
                error_message="msg",
                consecutive_failures=1,
            )

    def test_negative_consecutive_failures_raises(self) -> None:
        with pytest.raises(
            ValueError, match="consecutive_failures must not be negative"
        ):
            SubscriberSendError(
                subscription_id="sub-abc",
                error_type="err",
                error_message="msg",
                consecutive_failures=-1,
            )

    def test_zero_consecutive_failures_valid(self) -> None:
        error = SubscriberSendError(
            subscription_id="sub-abc",
            error_type="err",
            error_message="msg",
            consecutive_failures=0,
        )
        assert error.consecutive_failures == 0


# ---------------------------------------------------------------------------
# BroadcastResult model
# ---------------------------------------------------------------------------


class TestBroadcastResult:
    """BroadcastResult immutability, properties, and validation tests."""

    def test_defaults(self) -> None:
        result = BroadcastResult()
        assert result.delivered_count == 0
        assert result.filtered_count == 0
        assert result.error_count == 0
        assert result.errors == ()
        assert result.auto_removed_ids == ()

    def test_create_with_values(self) -> None:
        error = SubscriberSendError(
            subscription_id="sub-1",
            error_type="queue_full",
            error_message="full",
            consecutive_failures=2,
        )
        result = BroadcastResult(
            delivered_count=5,
            filtered_count=2,
            error_count=1,
            errors=(error,),
            auto_removed_ids=("sub-dead",),
        )
        assert result.delivered_count == 5
        assert result.filtered_count == 2
        assert result.error_count == 1
        assert len(result.errors) == 1
        assert result.auto_removed_ids == ("sub-dead",)

    def test_frozen(self) -> None:
        result = BroadcastResult(delivered_count=1)
        with pytest.raises(AttributeError):
            result.delivered_count = 2  # type: ignore[misc]

    def test_total_subscribers(self) -> None:
        result = BroadcastResult(
            delivered_count=3, filtered_count=2, error_count=1
        )
        assert result.total_subscribers == 6

    def test_has_errors_true(self) -> None:
        result = BroadcastResult(error_count=1)
        assert result.has_errors is True

    def test_has_errors_false(self) -> None:
        result = BroadcastResult(error_count=0)
        assert result.has_errors is False

    def test_has_auto_removals_true(self) -> None:
        result = BroadcastResult(auto_removed_ids=("sub-1",))
        assert result.has_auto_removals is True

    def test_has_auto_removals_false(self) -> None:
        result = BroadcastResult()
        assert result.has_auto_removals is False

    def test_negative_delivered_count_raises(self) -> None:
        with pytest.raises(ValueError, match="delivered_count must not be negative"):
            BroadcastResult(delivered_count=-1)

    def test_negative_filtered_count_raises(self) -> None:
        with pytest.raises(ValueError, match="filtered_count must not be negative"):
            BroadcastResult(filtered_count=-1)

    def test_negative_error_count_raises(self) -> None:
        with pytest.raises(ValueError, match="error_count must not be negative"):
            BroadcastResult(error_count=-1)


# ---------------------------------------------------------------------------
# NotificationBroadcasterConfig -- max_consecutive_failures
# ---------------------------------------------------------------------------


class TestConfigMaxConsecutiveFailures:
    """Tests for the max_consecutive_failures config option."""

    def test_default_value(self) -> None:
        config = NotificationBroadcasterConfig()
        assert config.max_consecutive_failures == 5

    def test_custom_value(self) -> None:
        config = NotificationBroadcasterConfig(max_consecutive_failures=10)
        assert config.max_consecutive_failures == 10

    def test_zero_disables_auto_removal(self) -> None:
        config = NotificationBroadcasterConfig(max_consecutive_failures=0)
        assert config.max_consecutive_failures == 0

    def test_negative_raises(self) -> None:
        with pytest.raises(
            ValueError, match="max_consecutive_failures must not be negative"
        ):
            NotificationBroadcasterConfig(max_consecutive_failures=-1)


# ---------------------------------------------------------------------------
# NotificationBroadcaster -- fan-out per-subscriber error handling
# ---------------------------------------------------------------------------


class TestFanOutErrorHandling:
    """Tests for per-subscriber error handling during broadcast fan-out."""

    @pytest.mark.asyncio
    async def test_broadcast_returns_broadcast_result(self) -> None:
        """broadcast() returns a BroadcastResult, not an int."""
        broadcaster = NotificationBroadcaster()
        await broadcaster.subscribe()
        envelope = _make_completion_envelope()
        result = await broadcaster.broadcast(envelope)
        assert isinstance(result, BroadcastResult)
        assert result.delivered_count == 1

    @pytest.mark.asyncio
    async def test_successful_delivery_resets_failure_counter(self) -> None:
        """A successful delivery resets the consecutive failure count."""
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()
        # Manually inject a failure count
        broadcaster._failure_counts[handle.subscription_id] = 3
        envelope = _make_completion_envelope()
        result = await broadcaster.broadcast(envelope)
        assert result.delivered_count == 1
        assert broadcaster.get_failure_count(handle.subscription_id) == 0

    @pytest.mark.asyncio
    async def test_delivery_error_increments_failure_counter(self) -> None:
        """A failed delivery increments the consecutive failure count."""
        config = NotificationBroadcasterConfig(max_consecutive_failures=10)
        broadcaster = NotificationBroadcaster(config=config)
        handle = await broadcaster.subscribe()

        # Patch _deliver_to_subscriber to raise an exception
        def _raise_error(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Simulated delivery failure")

        with patch.object(
            broadcaster, "_deliver_to_subscriber", side_effect=_raise_error
        ):
            envelope = _make_completion_envelope()
            result = await broadcaster.broadcast(envelope)

        assert result.delivered_count == 0
        assert result.error_count == 1
        assert result.has_errors is True
        assert len(result.errors) == 1
        assert result.errors[0].subscription_id == handle.subscription_id
        assert result.errors[0].error_type == "unexpected_error"
        assert result.errors[0].consecutive_failures == 1
        assert broadcaster.get_failure_count(handle.subscription_id) == 1

    @pytest.mark.asyncio
    async def test_error_isolation_between_subscribers(self) -> None:
        """One subscriber's error does not block delivery to others."""
        config = NotificationBroadcasterConfig(max_consecutive_failures=10)
        broadcaster = NotificationBroadcaster(config=config)
        h1 = await broadcaster.subscribe()
        h2 = await broadcaster.subscribe()
        h3 = await broadcaster.subscribe()

        # Track call order and fail only for h2
        original_deliver = broadcaster._deliver_to_subscriber

        def _selective_fail(
            sub_id: str, queue: asyncio.Queue, envelope: object
        ) -> None:
            if sub_id == h2.subscription_id:
                raise RuntimeError("h2 failed")
            original_deliver(sub_id, queue, envelope)  # type: ignore[arg-type]

        with patch.object(
            broadcaster, "_deliver_to_subscriber", side_effect=_selective_fail
        ):
            envelope = _make_completion_envelope()
            result = await broadcaster.broadcast(envelope)

        assert result.delivered_count == 2
        assert result.error_count == 1
        assert result.errors[0].subscription_id == h2.subscription_id

        # h1 and h3 should have received the envelope
        r1 = await broadcaster.receive(h1.subscription_id, timeout=1.0)
        r3 = await broadcaster.receive(h3.subscription_id, timeout=1.0)
        assert r1 is not None
        assert r3 is not None

    @pytest.mark.asyncio
    async def test_filtered_count_in_broadcast_result(self) -> None:
        """Filtered subscribers are counted in the result."""
        broadcaster = NotificationBroadcaster()
        h_all = await broadcaster.subscribe()
        h_completion = await broadcaster.subscribe(
            event_filter=frozenset({NotificationEventType.COMPLETION})
        )

        # Send an alert -- h_completion should be filtered
        envelope = _make_alert_envelope()
        result = await broadcaster.broadcast(envelope)
        assert result.delivered_count == 1
        assert result.filtered_count == 1
        assert result.total_subscribers == 2

    @pytest.mark.asyncio
    async def test_multiple_consecutive_failures_tracked(self) -> None:
        """Multiple broadcast failures increment the counter correctly."""
        config = NotificationBroadcasterConfig(max_consecutive_failures=10)
        broadcaster = NotificationBroadcaster(config=config)
        handle = await broadcaster.subscribe()

        def _raise_error(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Delivery failure")

        with patch.object(
            broadcaster, "_deliver_to_subscriber", side_effect=_raise_error
        ):
            for i in range(3):
                result = await broadcaster.broadcast(_make_completion_envelope())
                assert result.errors[0].consecutive_failures == i + 1

        assert broadcaster.get_failure_count(handle.subscription_id) == 3

    @pytest.mark.asyncio
    async def test_failure_counter_resets_after_success(self) -> None:
        """Successful delivery after failures resets the counter."""
        config = NotificationBroadcasterConfig(max_consecutive_failures=10)
        broadcaster = NotificationBroadcaster(config=config)
        handle = await broadcaster.subscribe()

        # Simulate 2 failures
        def _raise_error(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Delivery failure")

        with patch.object(
            broadcaster, "_deliver_to_subscriber", side_effect=_raise_error
        ):
            await broadcaster.broadcast(_make_completion_envelope())
            await broadcaster.broadcast(_make_completion_envelope())

        assert broadcaster.get_failure_count(handle.subscription_id) == 2

        # Successful delivery resets
        result = await broadcaster.broadcast(_make_completion_envelope())
        assert result.delivered_count == 1
        assert result.error_count == 0
        assert broadcaster.get_failure_count(handle.subscription_id) == 0


# ---------------------------------------------------------------------------
# NotificationBroadcaster -- auto-removal of dead subscribers
# ---------------------------------------------------------------------------


class TestAutoRemoval:
    """Tests for automatic subscriber removal after max failures."""

    @pytest.mark.asyncio
    async def test_auto_remove_after_max_failures(self) -> None:
        """Subscriber is auto-removed after reaching max consecutive failures."""
        config = NotificationBroadcasterConfig(max_consecutive_failures=3)
        broadcaster = NotificationBroadcaster(config=config)
        handle = await broadcaster.subscribe()

        def _raise_error(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Dead subscriber")

        with patch.object(
            broadcaster, "_deliver_to_subscriber", side_effect=_raise_error
        ):
            # Failures 1 and 2 -- subscriber still present
            result1 = await broadcaster.broadcast(_make_completion_envelope())
            assert result1.has_auto_removals is False
            assert broadcaster.has_subscriber(handle.subscription_id)

            result2 = await broadcaster.broadcast(_make_completion_envelope())
            assert result2.has_auto_removals is False
            assert broadcaster.has_subscriber(handle.subscription_id)

            # Failure 3 -- auto-removal triggered
            result3 = await broadcaster.broadcast(_make_completion_envelope())
            assert result3.has_auto_removals is True
            assert handle.subscription_id in result3.auto_removed_ids
            assert not broadcaster.has_subscriber(handle.subscription_id)

    @pytest.mark.asyncio
    async def test_auto_removal_does_not_affect_healthy_subscribers(self) -> None:
        """Auto-removal of one subscriber leaves healthy ones intact."""
        config = NotificationBroadcasterConfig(max_consecutive_failures=2)
        broadcaster = NotificationBroadcaster(config=config)
        h_healthy = await broadcaster.subscribe()
        h_dead = await broadcaster.subscribe()

        original_deliver = broadcaster._deliver_to_subscriber

        def _fail_dead_only(
            sub_id: str, queue: asyncio.Queue, envelope: object
        ) -> None:
            if sub_id == h_dead.subscription_id:
                raise RuntimeError("Dead")
            original_deliver(sub_id, queue, envelope)  # type: ignore[arg-type]

        with patch.object(
            broadcaster, "_deliver_to_subscriber", side_effect=_fail_dead_only
        ):
            # First failure
            await broadcaster.broadcast(_make_completion_envelope())
            # Second failure triggers auto-removal
            result = await broadcaster.broadcast(_make_completion_envelope())

        assert result.has_auto_removals is True
        assert h_dead.subscription_id in result.auto_removed_ids
        assert not broadcaster.has_subscriber(h_dead.subscription_id)

        # Healthy subscriber remains
        assert broadcaster.has_subscriber(h_healthy.subscription_id)
        assert broadcaster.subscriber_count == 1

    @pytest.mark.asyncio
    async def test_auto_removal_disabled_when_max_is_zero(self) -> None:
        """Setting max_consecutive_failures=0 disables auto-removal."""
        config = NotificationBroadcasterConfig(max_consecutive_failures=0)
        broadcaster = NotificationBroadcaster(config=config)
        handle = await broadcaster.subscribe()

        def _raise_error(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Failure")

        with patch.object(
            broadcaster, "_deliver_to_subscriber", side_effect=_raise_error
        ):
            for _ in range(10):
                result = await broadcaster.broadcast(_make_completion_envelope())
                assert result.has_auto_removals is False

        # Subscriber still present despite 10 failures
        assert broadcaster.has_subscriber(handle.subscription_id)
        assert broadcaster.get_failure_count(handle.subscription_id) == 10

    @pytest.mark.asyncio
    async def test_auto_removal_cleans_up_failure_tracking(self) -> None:
        """Auto-removed subscribers have their failure counters cleaned up."""
        config = NotificationBroadcasterConfig(max_consecutive_failures=1)
        broadcaster = NotificationBroadcaster(config=config)
        handle = await broadcaster.subscribe()

        def _raise_error(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Instant death")

        with patch.object(
            broadcaster, "_deliver_to_subscriber", side_effect=_raise_error
        ):
            result = await broadcaster.broadcast(_make_completion_envelope())

        assert result.has_auto_removals is True
        assert broadcaster.get_failure_count(handle.subscription_id) == 0
        assert broadcaster.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_result_after_all_subscribers_removed(self) -> None:
        """Broadcasting after all subscribers are auto-removed works cleanly."""
        config = NotificationBroadcasterConfig(max_consecutive_failures=1)
        broadcaster = NotificationBroadcaster(config=config)
        await broadcaster.subscribe()

        def _raise_error(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Dead")

        with patch.object(
            broadcaster, "_deliver_to_subscriber", side_effect=_raise_error
        ):
            await broadcaster.broadcast(_make_completion_envelope())

        # All subscribers removed -- subsequent broadcast is clean
        result = await broadcaster.broadcast(_make_completion_envelope())
        assert result.delivered_count == 0
        assert result.error_count == 0
        assert result.filtered_count == 0
        assert result.total_subscribers == 0


# ---------------------------------------------------------------------------
# NotificationBroadcaster -- get_failure_count
# ---------------------------------------------------------------------------


class TestGetFailureCount:
    """Tests for the get_failure_count introspection method."""

    @pytest.mark.asyncio
    async def test_initial_failure_count_is_zero(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()
        assert broadcaster.get_failure_count(handle.subscription_id) == 0

    def test_unknown_subscriber_returns_zero(self) -> None:
        broadcaster = NotificationBroadcaster()
        assert broadcaster.get_failure_count("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_failure_count_after_unsubscribe(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()
        await broadcaster.unsubscribe(handle.subscription_id)
        assert broadcaster.get_failure_count(handle.subscription_id) == 0

    @pytest.mark.asyncio
    async def test_failure_count_after_remove_all(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()
        await broadcaster.remove_all()
        assert broadcaster.get_failure_count(handle.subscription_id) == 0
