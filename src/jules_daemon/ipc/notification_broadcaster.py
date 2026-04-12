"""Notification broadcaster for daemon push events to CLI subscribers.

Manages a thread-safe registry of notification subscribers and fans out
``NotificationEnvelope`` events to matching subscribers. Each subscriber
gets an independent ``asyncio.Queue`` for async consumption, with
optional event type filtering.

This is the server-side counterpart to the notification subscription
protocol defined in ``protocol.notifications``. While the protocol
module defines the wire-format models (SubscribeRequest, etc.), this
module implements the runtime fan-out machinery inside the daemon.

Key design decisions:

- **Thread-safe subscriber map**: Uses ``asyncio.Lock`` to serialize
  subscribe and unsubscribe mutations, ensuring consistent state under
  concurrent calls from the IPC server's connection handlers.

- **Event type filtering**: Each subscriber optionally specifies a
  ``frozenset`` of ``NotificationEventType`` values. Events whose type
  is not in the filter are silently skipped for that subscriber. A
  ``None`` filter means "receive all event types" (the default).

- **Non-blocking broadcast**: If a subscriber queue is full (slow
  consumer), the oldest entry is evicted so the broadcaster never
  blocks. This trades completeness for liveness -- matching the
  ``JobOutputBroadcaster`` pattern.

- **Per-subscriber error tracking**: Each subscriber tracks consecutive
  delivery failures. When a subscriber exceeds the configured
  ``max_consecutive_failures`` threshold, it is automatically removed
  from the registry. A ``BroadcastResult`` reports per-subscriber
  delivery outcomes including error details.

- **Immutable data**: ``NotificationSubscriberHandle``,
  ``NotificationBroadcasterConfig``, ``BroadcastResult``, and
  ``SubscriberSendError`` are frozen dataclasses. The broadcaster
  itself holds mutable state (subscriber map, queues, failure counters)
  but all data flowing through it is immutable.

- **Independent of IPC transport**: The broadcaster operates on
  ``NotificationEnvelope`` objects. How those envelopes reach the
  CLI (via framed IPC, WebSocket, etc.) is the responsibility of a
  separate handler layer.

Architecture::

    IPC Handler                  NotificationBroadcaster
        |                               |
        |-- subscribe(filter) --------->|
        |<-- NotificationSubscriberHandle
        |                               |
    DaemonCore                          |
        |-- broadcast(envelope) ------->|
        |   fans out to matching queues |
        |   returns BroadcastResult     |
        |                               |
    IPC Handler                         |
        |-- receive(sub_id) ----------->|
        |<-- NotificationEnvelope       |
        |                               |
        |-- unsubscribe(sub_id) ------->|
        |<-- bool (True=removed)        |

Usage::

    from jules_daemon.ipc.notification_broadcaster import (
        NotificationBroadcaster,
        NotificationBroadcasterConfig,
        BroadcastResult,
    )

    broadcaster = NotificationBroadcaster()

    # Subscribe a client
    handle = await broadcaster.subscribe(
        event_filter=frozenset({NotificationEventType.COMPLETION}),
    )

    # Broadcast from daemon core -- returns detailed result
    result = await broadcaster.broadcast(envelope)
    print(f"Delivered: {result.delivered_count}, Errors: {result.error_count}")

    # Client receives events
    event = await broadcaster.receive(handle.subscription_id, timeout=5.0)

    # Client disconnects
    await broadcaster.unsubscribe(handle.subscription_id)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field

from jules_daemon.protocol.notifications import (
    HEARTBEAT_DEFAULT_INTERVAL_SECONDS,
    NotificationEnvelope,
    NotificationEventType,
)

__all__ = [
    "BroadcastResult",
    "NotificationBroadcaster",
    "NotificationBroadcasterConfig",
    "NotificationSubscriberHandle",
    "SubscriberSendError",
]

logger = logging.getLogger(__name__)

_DEFAULT_SUBSCRIBER_QUEUE_SIZE = 100
_DEFAULT_MAX_CONSECUTIVE_FAILURES = 5


# ---------------------------------------------------------------------------
# Per-subscriber error reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubscriberSendError:
    """Record of a failed event delivery to a specific subscriber.

    Captured during fan-out broadcasting and included in the
    ``BroadcastResult`` so callers can inspect per-subscriber errors.

    Attributes:
        subscription_id:      The subscriber that failed to receive.
        error_type:           Classification of the failure (e.g.,
                              ``"queue_full"``, ``"unexpected_error"``).
        error_message:        Human-readable description of the error.
        consecutive_failures: How many consecutive delivery failures this
                              subscriber has accumulated (including this one).
    """

    subscription_id: str
    error_type: str
    error_message: str
    consecutive_failures: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.subscription_id, str)
            or not self.subscription_id.strip()
        ):
            raise ValueError("subscription_id must not be empty")
        if not isinstance(self.error_type, str) or not self.error_type.strip():
            raise ValueError("error_type must not be empty")
        if self.consecutive_failures < 0:
            raise ValueError("consecutive_failures must not be negative")


# ---------------------------------------------------------------------------
# Broadcast result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BroadcastResult:
    """Immutable result of a fan-out broadcast operation.

    Reports per-subscriber delivery outcomes so callers can observe
    partial failures, filtered subscribers, and auto-removed dead
    subscribers without inspecting individual queues.

    Attributes:
        delivered_count:  Number of subscribers that received the event.
        filtered_count:   Number of subscribers skipped by event filter.
        error_count:      Number of subscribers with delivery errors.
        errors:           Details of each per-subscriber error.
        auto_removed_ids: Subscriber IDs that were auto-removed due to
                          exceeding the max consecutive failure threshold.
    """

    delivered_count: int = 0
    filtered_count: int = 0
    error_count: int = 0
    errors: tuple[SubscriberSendError, ...] = ()
    auto_removed_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.delivered_count < 0:
            raise ValueError("delivered_count must not be negative")
        if self.filtered_count < 0:
            raise ValueError("filtered_count must not be negative")
        if self.error_count < 0:
            raise ValueError("error_count must not be negative")

    @property
    def total_subscribers(self) -> int:
        """Total number of subscribers that were considered."""
        return self.delivered_count + self.filtered_count + self.error_count

    @property
    def has_errors(self) -> bool:
        """True if any subscriber had a delivery error."""
        return self.error_count > 0

    @property
    def has_auto_removals(self) -> bool:
        """True if any subscriber was auto-removed due to failures."""
        return len(self.auto_removed_ids) > 0


# ---------------------------------------------------------------------------
# Immutable data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NotificationBroadcasterConfig:
    """Immutable configuration for the notification broadcaster.

    Attributes:
        subscriber_queue_size:      Maximum number of envelopes that can
                                    be queued per subscriber before
                                    backpressure kicks in (oldest evicted).
        heartbeat_interval_seconds: Interval between heartbeat events.
                                    Used by the daemon's heartbeat timer,
                                    not directly by the broadcaster.
        max_consecutive_failures:   Maximum number of consecutive delivery
                                    failures allowed per subscriber before
                                    auto-removal. Set to 0 to disable
                                    auto-removal (failures are still tracked
                                    and reported).
    """

    subscriber_queue_size: int = _DEFAULT_SUBSCRIBER_QUEUE_SIZE
    heartbeat_interval_seconds: int = HEARTBEAT_DEFAULT_INTERVAL_SECONDS
    max_consecutive_failures: int = _DEFAULT_MAX_CONSECUTIVE_FAILURES

    def __post_init__(self) -> None:
        if self.subscriber_queue_size < 1:
            raise ValueError("subscriber_queue_size must be positive")
        if self.heartbeat_interval_seconds < 1:
            raise ValueError("heartbeat_interval_seconds must be positive")
        if self.max_consecutive_failures < 0:
            raise ValueError("max_consecutive_failures must not be negative")


@dataclass(frozen=True)
class NotificationSubscriberHandle:
    """Immutable handle identifying a notification subscriber.

    Returned by ``subscribe()`` and used with ``receive()``,
    ``unsubscribe()``, and ``get_subscriber_handle()``.

    Attributes:
        subscription_id: Unique identifier for this subscriber.
        event_filter:    Optional frozen set of event types to receive.
                         When None, all event types are delivered.
    """

    subscription_id: str
    event_filter: frozenset[NotificationEventType] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.subscription_id, str)
            or not self.subscription_id.strip()
        ):
            raise ValueError("subscription_id must not be empty")


# ---------------------------------------------------------------------------
# Internal subscriber entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SubscriberEntry:
    """Internal mapping between a subscriber handle and its queue.

    Not part of the public API. Managed by NotificationBroadcaster.
    """

    handle: NotificationSubscriberHandle
    queue: asyncio.Queue[NotificationEnvelope] = field(repr=False)


# ---------------------------------------------------------------------------
# NotificationBroadcaster
# ---------------------------------------------------------------------------


class NotificationBroadcaster:
    """Thread-safe broadcaster for daemon notification events.

    Manages a registry of notification subscribers and fans out
    ``NotificationEnvelope`` events to matching subscribers. Each
    subscriber gets an independent ``asyncio.Queue`` for async
    consumption.

    Fan-out broadcasting delivers events to all active subscribers
    and tracks per-subscriber send errors. When a subscriber exceeds
    the configured ``max_consecutive_failures`` threshold, it is
    automatically removed from the registry. Successful deliveries
    reset the failure counter to zero.

    Thread safety: Uses ``asyncio.Lock`` to serialize subscribe and
    unsubscribe mutations. Read-only operations (``subscriber_count``,
    ``has_subscriber``, ``list_subscriber_ids``) do not acquire the
    lock.

    Args:
        config: Optional broadcaster configuration. Uses defaults
            when None.
    """

    def __init__(
        self,
        config: NotificationBroadcasterConfig | None = None,
    ) -> None:
        self._config = config or NotificationBroadcasterConfig()
        self._subscribers: dict[str, _SubscriberEntry] = {}
        self._failure_counts: dict[str, int] = {}
        self._lock = asyncio.Lock()

    # -- Properties -----------------------------------------------------------

    @property
    def subscriber_count(self) -> int:
        """Number of currently registered subscribers."""
        return len(self._subscribers)

    # -- Subscribe ------------------------------------------------------------

    async def subscribe(
        self,
        *,
        event_filter: frozenset[NotificationEventType] | None = None,
    ) -> NotificationSubscriberHandle:
        """Register a new notification subscriber.

        Creates a subscriber with an independent queue and optional
        event type filter. The subscriber will receive all subsequent
        broadcasts that match its filter.

        Args:
            event_filter: Optional frozen set of event types to receive.
                When None, all event types are delivered (default).

        Returns:
            An immutable handle for the new subscriber. Use the
            ``subscription_id`` with ``receive()`` and ``unsubscribe()``.
        """
        subscription_id = f"nsub-{uuid.uuid4().hex[:12]}"
        handle = NotificationSubscriberHandle(
            subscription_id=subscription_id,
            event_filter=event_filter,
        )
        queue: asyncio.Queue[NotificationEnvelope] = asyncio.Queue(
            maxsize=self._config.subscriber_queue_size,
        )
        entry = _SubscriberEntry(handle=handle, queue=queue)

        async with self._lock:
            # Create new dict to follow immutable-pattern convention
            self._subscribers = {**self._subscribers, subscription_id: entry}
            self._failure_counts = {**self._failure_counts, subscription_id: 0}

        logger.debug(
            "Notification subscriber registered: %s (filter=%s)",
            subscription_id,
            event_filter,
        )
        return handle

    # -- Unsubscribe ----------------------------------------------------------

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a notification subscriber.

        Idempotent: removing a nonexistent subscriber returns False
        without raising.

        Args:
            subscription_id: The subscriber to remove.

        Returns:
            True if the subscriber was found and removed, False if
            not found.
        """
        async with self._lock:
            if subscription_id not in self._subscribers:
                return False
            # Create new dict without the removed subscriber
            self._subscribers = {
                sid: entry
                for sid, entry in self._subscribers.items()
                if sid != subscription_id
            }
            # Clean up failure tracking
            self._failure_counts = {
                sid: count
                for sid, count in self._failure_counts.items()
                if sid != subscription_id
            }

        logger.debug(
            "Notification subscriber removed: %s",
            subscription_id,
        )
        return True

    # -- Broadcast ------------------------------------------------------------

    async def broadcast(
        self,
        envelope: NotificationEnvelope,
    ) -> BroadcastResult:
        """Fan out a notification envelope to all matching subscribers.

        Enqueues the envelope into each subscriber's queue whose event
        filter matches (or is None). If a subscriber's queue is full,
        the oldest entry is evicted to make room (non-blocking).

        Per-subscriber error handling:
        - Each subscriber delivery is independent -- one failure never
          blocks delivery to remaining subscribers.
        - Consecutive failures are tracked per subscriber. A successful
          delivery resets the counter to zero.
        - When a subscriber exceeds ``max_consecutive_failures``, it is
          automatically removed from the registry (reported in the
          result's ``auto_removed_ids``). Set ``max_consecutive_failures``
          to 0 in the config to disable auto-removal.

        Args:
            envelope: The notification envelope to broadcast.

        Returns:
            A ``BroadcastResult`` with per-subscriber delivery outcomes.
        """
        # Take a snapshot of current subscribers (no lock needed for read)
        subscribers = self._subscribers
        delivered = 0
        filtered = 0
        errors: list[SubscriberSendError] = []
        auto_removed: list[str] = []

        for sub_id, entry in subscribers.items():
            # Apply event filter
            if entry.handle.event_filter is not None:
                if envelope.event_type not in entry.handle.event_filter:
                    filtered += 1
                    continue

            # Attempt delivery with per-subscriber error isolation
            try:
                self._deliver_to_subscriber(sub_id, entry.queue, envelope)
                delivered += 1
                # Reset failure counter on successful delivery
                self._failure_counts[sub_id] = 0
            except Exception as exc:
                # Track consecutive failure
                current_failures = self._failure_counts.get(sub_id, 0) + 1
                self._failure_counts[sub_id] = current_failures

                error_type = (
                    "queue_full"
                    if isinstance(exc, asyncio.QueueFull)
                    else "unexpected_error"
                )
                send_error = SubscriberSendError(
                    subscription_id=sub_id,
                    error_type=error_type,
                    error_message=str(exc),
                    consecutive_failures=current_failures,
                )
                errors.append(send_error)

                logger.warning(
                    "Delivery failed for subscriber %s "
                    "(type=%s, consecutive=%d): %s",
                    sub_id,
                    error_type,
                    current_failures,
                    exc,
                )

                # Auto-remove if threshold exceeded (0 = disabled)
                max_failures = self._config.max_consecutive_failures
                if max_failures > 0 and current_failures >= max_failures:
                    auto_removed.append(sub_id)
                    logger.warning(
                        "Auto-removing subscriber %s after %d "
                        "consecutive failures",
                        sub_id,
                        current_failures,
                    )

        # Perform auto-removals outside the iteration loop
        for sub_id in auto_removed:
            async with self._lock:
                self._subscribers = {
                    sid: entry
                    for sid, entry in self._subscribers.items()
                    if sid != sub_id
                }
                self._failure_counts = {
                    sid: count
                    for sid, count in self._failure_counts.items()
                    if sid != sub_id
                }

        result = BroadcastResult(
            delivered_count=delivered,
            filtered_count=filtered,
            error_count=len(errors),
            errors=tuple(errors),
            auto_removed_ids=tuple(auto_removed),
        )

        if delivered > 0:
            logger.debug(
                "Broadcast %s event (id=%s) to %d subscriber(s)"
                " (filtered=%d, errors=%d)",
                envelope.event_type.value,
                envelope.event_id,
                delivered,
                filtered,
                len(errors),
            )
        else:
            logger.debug(
                "Broadcast %s event (id=%s) -- no matching subscribers"
                " (filtered=%d, errors=%d)",
                envelope.event_type.value,
                envelope.event_id,
                filtered,
                len(errors),
            )

        return result

    def _deliver_to_subscriber(
        self,
        sub_id: str,
        queue: asyncio.Queue[NotificationEnvelope],
        envelope: NotificationEnvelope,
    ) -> None:
        """Deliver an envelope to a single subscriber queue.

        Handles backpressure by evicting the oldest entry when the
        queue is full. This is a synchronous operation (put_nowait).

        Args:
            sub_id:   The subscriber ID (for logging).
            queue:    The subscriber's async queue.
            envelope: The envelope to deliver.

        Raises:
            asyncio.QueueFull: If delivery fails even after eviction
                attempt (should be extremely rare).
            Exception: Any unexpected error during delivery.
        """
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass  # pragma: no cover -- race guard
            logger.debug(
                "Notification subscriber %s queue full; evicted oldest",
                sub_id,
            )
        queue.put_nowait(envelope)

    # -- Receive --------------------------------------------------------------

    async def receive(
        self,
        subscription_id: str,
        *,
        timeout: float = 5.0,
    ) -> NotificationEnvelope | None:
        """Receive the next notification from a subscriber's queue.

        Blocks (with timeout) until an envelope is available or the
        timeout expires. Returns None on timeout.

        Args:
            subscription_id: The subscriber to receive from.
            timeout: Maximum seconds to wait for an envelope.

        Returns:
            The next NotificationEnvelope, or None on timeout.

        Raises:
            ValueError: If the subscriber is not found.
        """
        entry = self._subscribers.get(subscription_id)
        if entry is None:
            raise ValueError(
                f"Notification subscriber {subscription_id!r} not found"
            )
        try:
            return await asyncio.wait_for(entry.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # -- Introspection --------------------------------------------------------

    def has_subscriber(self, subscription_id: str) -> bool:
        """Check whether a subscriber is currently registered.

        Args:
            subscription_id: The subscriber ID to check.

        Returns:
            True if the subscriber is in the registry.
        """
        return subscription_id in self._subscribers

    def get_subscriber_handle(
        self,
        subscription_id: str,
    ) -> NotificationSubscriberHandle | None:
        """Look up a subscriber handle by ID.

        Args:
            subscription_id: The subscriber ID to look up.

        Returns:
            The subscriber handle, or None if not found.
        """
        entry = self._subscribers.get(subscription_id)
        if entry is None:
            return None
        return entry.handle

    def list_subscriber_ids(self) -> frozenset[str]:
        """Return an immutable snapshot of all subscriber IDs.

        Returns:
            Frozen set of subscription_id strings.
        """
        return frozenset(self._subscribers.keys())

    # -- Bulk operations ------------------------------------------------------

    async def remove_all(self) -> int:
        """Remove all subscribers.

        Used during daemon shutdown for clean cleanup.

        Returns:
            Number of subscribers that were removed.
        """
        async with self._lock:
            count = len(self._subscribers)
            self._subscribers = {}
            self._failure_counts = {}

        if count > 0:
            logger.info(
                "Removed all %d notification subscriber(s)",
                count,
            )
        return count

    def get_failure_count(self, subscription_id: str) -> int:
        """Return the consecutive failure count for a subscriber.

        Args:
            subscription_id: The subscriber ID to query.

        Returns:
            Number of consecutive delivery failures, or 0 if the
            subscriber is not found or has no failures.
        """
        return self._failure_counts.get(subscription_id, 0)
