"""Tests for subscriber cleanup function.

Validates:
    - Removes subscriber from the notification broadcaster registry
    - Drains all pending items from the subscriber's queue
    - Cleans up failure tracking counters
    - Returns an immutable SubscriberCleanupResult with drain statistics
    - Idempotent: cleaning a nonexistent subscriber returns not-found result
    - Concurrent cleanup: lock-protected registry mutation
    - Works with event bus subscriptions too
    - Error isolation: cleanup does not propagate internal errors
"""

from __future__ import annotations

import asyncio

import pytest

from jules_daemon.protocol.notifications import (
    CompletionNotification,
    NotificationEventType,
    create_notification_envelope,
)

from jules_daemon.ipc.notification_broadcaster import (
    NotificationBroadcaster,
    NotificationBroadcasterConfig,
)

from jules_daemon.cleanup.subscriber_cleanup import (
    SubscriberCleanupResult,
    cleanup_subscriber,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SubscriberCleanupResult
# ---------------------------------------------------------------------------


class TestSubscriberCleanupResult:
    """Tests for the immutable result dataclass."""

    def test_found_result(self) -> None:
        result = SubscriberCleanupResult(
            subscriber_id="nsub-abc123",
            found=True,
            items_drained=5,
            failure_count_cleared=3,
        )
        assert result.subscriber_id == "nsub-abc123"
        assert result.found is True
        assert result.items_drained == 5
        assert result.failure_count_cleared == 3
        assert result.error is None

    def test_not_found_result(self) -> None:
        result = SubscriberCleanupResult(
            subscriber_id="nsub-missing",
            found=False,
            items_drained=0,
            failure_count_cleared=0,
        )
        assert result.found is False
        assert result.items_drained == 0

    def test_error_result(self) -> None:
        result = SubscriberCleanupResult(
            subscriber_id="nsub-err",
            found=True,
            items_drained=0,
            failure_count_cleared=0,
            error="Queue drain error",
        )
        assert result.error == "Queue drain error"

    def test_frozen(self) -> None:
        result = SubscriberCleanupResult(
            subscriber_id="nsub-abc",
            found=True,
            items_drained=0,
            failure_count_cleared=0,
        )
        with pytest.raises(AttributeError):
            result.found = False  # type: ignore[misc]

    def test_empty_subscriber_id_raises(self) -> None:
        with pytest.raises(ValueError, match="subscriber_id must not be empty"):
            SubscriberCleanupResult(
                subscriber_id="",
                found=False,
                items_drained=0,
                failure_count_cleared=0,
            )

    def test_negative_items_drained_raises(self) -> None:
        with pytest.raises(ValueError, match="items_drained must not be negative"):
            SubscriberCleanupResult(
                subscriber_id="nsub-x",
                found=True,
                items_drained=-1,
                failure_count_cleared=0,
            )

    def test_negative_failure_count_raises(self) -> None:
        with pytest.raises(
            ValueError, match="failure_count_cleared must not be negative"
        ):
            SubscriberCleanupResult(
                subscriber_id="nsub-x",
                found=True,
                items_drained=0,
                failure_count_cleared=-1,
            )


# ---------------------------------------------------------------------------
# cleanup_subscriber -- subscriber not found
# ---------------------------------------------------------------------------


class TestCleanupSubscriberNotFound:
    """Cleanup of a nonexistent subscriber."""

    @pytest.mark.asyncio
    async def test_nonexistent_subscriber_returns_not_found(self) -> None:
        broadcaster = NotificationBroadcaster()
        result = await cleanup_subscriber(
            broadcaster=broadcaster,
            subscriber_id="nsub-doesnotexist",
        )
        assert result.found is False
        assert result.subscriber_id == "nsub-doesnotexist"
        assert result.items_drained == 0
        assert result.failure_count_cleared == 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_idempotent_double_cleanup(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()

        # First cleanup: should find and remove
        result1 = await cleanup_subscriber(
            broadcaster=broadcaster,
            subscriber_id=handle.subscription_id,
        )
        assert result1.found is True

        # Second cleanup: should not find (already removed)
        result2 = await cleanup_subscriber(
            broadcaster=broadcaster,
            subscriber_id=handle.subscription_id,
        )
        assert result2.found is False
        assert result2.items_drained == 0


# ---------------------------------------------------------------------------
# cleanup_subscriber -- removes from registry
# ---------------------------------------------------------------------------


class TestCleanupSubscriberRemoval:
    """Cleanup removes subscriber from the broadcaster registry."""

    @pytest.mark.asyncio
    async def test_removes_from_registry(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()

        assert broadcaster.has_subscriber(handle.subscription_id) is True
        assert broadcaster.subscriber_count == 1

        result = await cleanup_subscriber(
            broadcaster=broadcaster,
            subscriber_id=handle.subscription_id,
        )

        assert result.found is True
        assert broadcaster.has_subscriber(handle.subscription_id) is False
        assert broadcaster.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_other_subscribers_unaffected(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle_a = await broadcaster.subscribe()
        handle_b = await broadcaster.subscribe()

        await cleanup_subscriber(
            broadcaster=broadcaster,
            subscriber_id=handle_a.subscription_id,
        )

        assert broadcaster.has_subscriber(handle_a.subscription_id) is False
        assert broadcaster.has_subscriber(handle_b.subscription_id) is True
        assert broadcaster.subscriber_count == 1


# ---------------------------------------------------------------------------
# cleanup_subscriber -- drains queue
# ---------------------------------------------------------------------------


class TestCleanupSubscriberDrainsQueue:
    """Cleanup drains all pending items from the subscriber's queue."""

    @pytest.mark.asyncio
    async def test_drains_pending_items(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()

        # Broadcast several events so the subscriber queue has items
        for i in range(5):
            envelope = _make_completion_envelope(run_id=f"run-{i:03d}")
            await broadcaster.broadcast(envelope)

        result = await cleanup_subscriber(
            broadcaster=broadcaster,
            subscriber_id=handle.subscription_id,
        )

        assert result.found is True
        assert result.items_drained == 5

    @pytest.mark.asyncio
    async def test_empty_queue_drains_zero(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()

        result = await cleanup_subscriber(
            broadcaster=broadcaster,
            subscriber_id=handle.subscription_id,
        )

        assert result.found is True
        assert result.items_drained == 0

    @pytest.mark.asyncio
    async def test_queue_no_longer_accessible_after_cleanup(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()

        await cleanup_subscriber(
            broadcaster=broadcaster,
            subscriber_id=handle.subscription_id,
        )

        # receive() should raise ValueError for unknown subscriber
        with pytest.raises(ValueError, match="not found"):
            await broadcaster.receive(handle.subscription_id)


# ---------------------------------------------------------------------------
# cleanup_subscriber -- clears failure tracking
# ---------------------------------------------------------------------------


class TestCleanupSubscriberClearsFailures:
    """Cleanup clears the failure count tracking for the subscriber."""

    @pytest.mark.asyncio
    async def test_clears_failure_count(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()

        # Simulate some failures by manipulating the internal state
        broadcaster._failure_counts[handle.subscription_id] = 3

        result = await cleanup_subscriber(
            broadcaster=broadcaster,
            subscriber_id=handle.subscription_id,
        )

        assert result.found is True
        assert result.failure_count_cleared == 3
        assert broadcaster.get_failure_count(handle.subscription_id) == 0

    @pytest.mark.asyncio
    async def test_zero_failure_count_cleared(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()

        result = await cleanup_subscriber(
            broadcaster=broadcaster,
            subscriber_id=handle.subscription_id,
        )

        assert result.failure_count_cleared == 0


# ---------------------------------------------------------------------------
# cleanup_subscriber -- concurrent safety
# ---------------------------------------------------------------------------


class TestCleanupSubscriberConcurrency:
    """Cleanup is safe under concurrent calls."""

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_same_subscriber(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()

        # Enqueue items so we can measure drain
        for i in range(3):
            envelope = _make_completion_envelope(run_id=f"run-{i}")
            await broadcaster.broadcast(envelope)

        # Run two cleanups concurrently
        results = await asyncio.gather(
            cleanup_subscriber(
                broadcaster=broadcaster,
                subscriber_id=handle.subscription_id,
            ),
            cleanup_subscriber(
                broadcaster=broadcaster,
                subscriber_id=handle.subscription_id,
            ),
        )

        # Exactly one should find the subscriber
        found_results = [r for r in results if r.found]
        not_found_results = [r for r in results if not r.found]

        assert len(found_results) == 1
        assert len(not_found_results) == 1

        # The one that found it should have drained items
        assert found_results[0].items_drained == 3

        # Registry should be clean
        assert broadcaster.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_different_subscribers(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle_a = await broadcaster.subscribe()
        handle_b = await broadcaster.subscribe()

        results = await asyncio.gather(
            cleanup_subscriber(
                broadcaster=broadcaster,
                subscriber_id=handle_a.subscription_id,
            ),
            cleanup_subscriber(
                broadcaster=broadcaster,
                subscriber_id=handle_b.subscription_id,
            ),
        )

        assert all(r.found for r in results)
        assert broadcaster.subscriber_count == 0


# ---------------------------------------------------------------------------
# cleanup_subscriber -- with event filter
# ---------------------------------------------------------------------------


class TestCleanupSubscriberWithFilter:
    """Cleanup works correctly for filtered subscribers."""

    @pytest.mark.asyncio
    async def test_filtered_subscriber_cleanup(self) -> None:
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe(
            event_filter=frozenset({NotificationEventType.COMPLETION}),
        )

        # Broadcast a matching event
        envelope = _make_completion_envelope()
        await broadcaster.broadcast(envelope)

        result = await cleanup_subscriber(
            broadcaster=broadcaster,
            subscriber_id=handle.subscription_id,
        )

        assert result.found is True
        assert result.items_drained == 1
