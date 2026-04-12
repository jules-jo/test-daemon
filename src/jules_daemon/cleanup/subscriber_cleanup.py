"""Subscriber cleanup for notification broadcaster resources.

Provides a comprehensive cleanup function that removes a subscriber from
the notification broadcaster registry and frees all associated resources:
the subscriber's ``asyncio.Queue`` (drained of pending items), the failure
tracking counter, and the registry entry itself.

This goes beyond the broadcaster's built-in ``unsubscribe()`` method, which
only removes the registry entry. The cleanup function additionally:

- Drains all pending envelopes from the subscriber's queue so no reference
  to queued ``NotificationEnvelope`` objects is retained.
- Captures the failure count before clearing it, for audit reporting.
- Returns an immutable ``SubscriberCleanupResult`` with drain statistics.
- Acquires the broadcaster's async lock for thread-safe registry mutation.

The function is idempotent: calling it for a subscriber that has already
been removed (or never existed) returns a result with ``found=False``
and zero drain counts.

Error isolation: if draining the queue raises an unexpected exception,
the error is captured in the result's ``error`` field. The subscriber is
still removed from the registry -- cleanup errors never leave orphaned
entries.

Usage::

    from jules_daemon.cleanup.subscriber_cleanup import cleanup_subscriber

    result = await cleanup_subscriber(
        broadcaster=broadcaster,
        subscriber_id="nsub-abc123def456",
    )
    if result.found:
        print(f"Drained {result.items_drained} pending events")
    else:
        print("Subscriber not found (already cleaned or never existed)")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from jules_daemon.ipc.notification_broadcaster import (
    NotificationBroadcaster,
)

__all__ = [
    "SubscriberCleanupResult",
    "cleanup_subscriber",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Immutable result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubscriberCleanupResult:
    """Immutable result of a subscriber cleanup operation.

    Captures whether the subscriber was found, how many pending queue
    items were drained, and the failure count that was cleared.

    Attributes:
        subscriber_id:         The subscriber identifier that was cleaned up.
        found:                 True if the subscriber was in the registry.
        items_drained:         Number of pending envelopes drained from
                               the subscriber's queue.
        failure_count_cleared: The consecutive failure count that was
                               cleared (0 if no failures were tracked).
        error:                 Human-readable error description if cleanup
                               encountered an internal issue, or None on
                               clean completion.
    """

    subscriber_id: str
    found: bool
    items_drained: int
    failure_count_cleared: int
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.subscriber_id, str) or not self.subscriber_id.strip():
            raise ValueError("subscriber_id must not be empty")
        if self.items_drained < 0:
            raise ValueError("items_drained must not be negative")
        if self.failure_count_cleared < 0:
            raise ValueError("failure_count_cleared must not be negative")


# ---------------------------------------------------------------------------
# Cleanup function
# ---------------------------------------------------------------------------


async def cleanup_subscriber(
    *,
    broadcaster: NotificationBroadcaster,
    subscriber_id: str,
) -> SubscriberCleanupResult:
    """Remove a subscriber from the broadcaster and free all resources.

    Acquires the broadcaster's async lock, removes the subscriber entry,
    drains the queue, and clears failure tracking. Returns an immutable
    result describing what was cleaned up.

    Idempotent: cleaning a subscriber that does not exist returns a
    result with ``found=False`` and zero counts.

    Args:
        broadcaster:   The NotificationBroadcaster managing the subscriber.
        subscriber_id: The unique subscriber identifier to clean up.

    Returns:
        An immutable ``SubscriberCleanupResult`` with cleanup statistics.
    """
    # Acquire lock and atomically remove the subscriber entry + failure count
    queue: asyncio.Queue | None = None
    failure_count: int = 0

    async with broadcaster._lock:
        entry = broadcaster._subscribers.get(subscriber_id)
        if entry is None:
            logger.debug(
                "Subscriber %s not found in registry; nothing to clean",
                subscriber_id,
            )
            return SubscriberCleanupResult(
                subscriber_id=subscriber_id,
                found=False,
                items_drained=0,
                failure_count_cleared=0,
            )

        # Capture the queue reference before removal
        queue = entry.queue

        # Capture failure count before clearing
        failure_count = broadcaster._failure_counts.get(subscriber_id, 0)

        # Remove subscriber from registry (immutable dict pattern)
        broadcaster._subscribers = {
            sid: e
            for sid, e in broadcaster._subscribers.items()
            if sid != subscriber_id
        }

        # Remove failure tracking (immutable dict pattern)
        broadcaster._failure_counts = {
            sid: count
            for sid, count in broadcaster._failure_counts.items()
            if sid != subscriber_id
        }

    # Drain the queue outside the lock (queue is no longer shared)
    items_drained = 0
    drain_error: str | None = None

    if queue is not None:
        try:
            items_drained = _drain_queue(queue)
        except Exception as exc:
            drain_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Error draining queue for subscriber %s: %s",
                subscriber_id,
                drain_error,
            )

    logger.debug(
        "Subscriber %s cleaned: drained=%d, failures_cleared=%d",
        subscriber_id,
        items_drained,
        failure_count,
    )

    return SubscriberCleanupResult(
        subscriber_id=subscriber_id,
        found=True,
        items_drained=items_drained,
        failure_count_cleared=failure_count,
        error=drain_error,
    )


def _drain_queue(queue: asyncio.Queue) -> int:
    """Drain all items from an asyncio.Queue synchronously.

    Uses ``get_nowait()`` in a loop until the queue is empty. This is
    safe because the queue has been detached from the subscriber registry
    and no producer will enqueue new items.

    Args:
        queue: The asyncio.Queue to drain.

    Returns:
        Number of items drained.
    """
    count = 0
    while True:
        try:
            queue.get_nowait()
            count += 1
        except asyncio.QueueEmpty:
            break
    return count
