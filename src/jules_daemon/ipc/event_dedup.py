"""Last-seen event ID tracker and deduplication filter.

Provides client-side deduplication for the notification event stream.
Records event IDs as they arrive and determines whether each incoming
event is a duplicate (already seen) or new (should be forwarded to
consumers).

This module complements the server-side ``NotificationBroadcaster``
by giving the subscription client a way to filter duplicate events
that may arrive after reconnection, retransmission, or transport-level
retries.

The design mirrors SSE ``Last-Event-ID`` semantics: the tracker
records the most recent event ID from the stream so the client can
resume from where it left off and skip events it has already processed.

Key design decisions:

- **Bounded capacity**: The tracker maintains a bounded set of recently
  seen event IDs (configurable via ``max_tracked_ids``). When capacity
  is reached, the oldest entries are evicted in FIFO order. This
  prevents unbounded memory growth for long-lived subscriptions.

- **Insertion-order eviction**: Uses ``OrderedDict`` for O(1) lookups
  and FIFO eviction. The most recently seen IDs are always retained.

- **Thread safety**: Internal state is guarded by ``threading.Lock``.
  Safe for concurrent access from multiple threads and from async
  code (via ``asyncio.to_thread`` or inline calls).

- **Immutable data**: ``EventDeduplicationConfig`` and
  ``DeduplicationVerdict`` are frozen dataclasses. The tracker itself
  holds mutable state (the seen-ID set) but all data flowing through
  it is immutable.

- **Envelope integration**: ``filter_envelope()`` accepts a
  ``NotificationEnvelope`` directly, extracts the ``event_id``,
  and returns a verdict. This is the primary consumer-facing API.

Architecture::

    Subscription Client
        |
        v  NotificationEnvelope
    EventIdTracker.filter_envelope()
        |
        v  DeduplicationVerdict
        |-- is_duplicate=True  -> skip (do not forward)
        |-- is_duplicate=False -> forward to callbacks / queue

Usage::

    from jules_daemon.ipc.event_dedup import (
        EventDeduplicationConfig,
        EventIdTracker,
    )

    tracker = EventIdTracker()

    # Record and check
    verdict = tracker.record("evt-abc123")
    if not verdict.is_duplicate:
        print("New event!")

    # Check without recording
    if tracker.is_duplicate("evt-abc123"):
        print("Already seen")

    # Filter a NotificationEnvelope
    verdict = tracker.filter_envelope(envelope)
    if not verdict.is_duplicate:
        forward_to_consumer(envelope)

    # Track last-seen for resume
    last_id = tracker.last_seen_event_id
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

from jules_daemon.protocol.notifications import NotificationEnvelope

__all__ = [
    "DeduplicationVerdict",
    "EventDeduplicationConfig",
    "EventIdTracker",
]

logger = logging.getLogger(__name__)


_DEFAULT_MAX_TRACKED_IDS: int = 10_000
"""Default capacity for tracked event IDs.

Sized to handle extended streaming sessions without eviction under
normal event rates. At ~100 bytes per ID entry, 10k entries consume
roughly 1 MB of memory.
"""


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_event_id(event_id: str) -> str:
    """Validate and normalize an event ID string.

    Args:
        event_id: The event ID to validate.

    Returns:
        The stripped event ID string.

    Raises:
        ValueError: If the event ID is empty or whitespace-only.
    """
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must not be empty or whitespace-only")
    return event_id.strip()


# ---------------------------------------------------------------------------
# EventDeduplicationConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventDeduplicationConfig:
    """Immutable configuration for the event ID tracker.

    Controls the maximum number of event IDs retained in memory.
    When capacity is reached, the oldest entries are evicted in
    FIFO order.

    Attributes:
        max_tracked_ids: Maximum number of event IDs to track.
            Must be positive. Older entries are evicted when this
            limit is reached.
    """

    max_tracked_ids: int = _DEFAULT_MAX_TRACKED_IDS

    def __post_init__(self) -> None:
        if self.max_tracked_ids < 1:
            raise ValueError(
                f"max_tracked_ids must be positive, got {self.max_tracked_ids}"
            )


# ---------------------------------------------------------------------------
# DeduplicationVerdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeduplicationVerdict:
    """Immutable result of a deduplication check.

    Returned by ``EventIdTracker.record()`` and
    ``EventIdTracker.filter_envelope()`` to indicate whether the
    event was a duplicate or new.

    Attributes:
        event_id:     The event ID that was checked.
        is_duplicate: True if the event ID was already tracked
            (i.e., a duplicate that should be skipped).
    """

    event_id: str
    is_duplicate: bool

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must not be empty or whitespace-only")


# ---------------------------------------------------------------------------
# EventIdTracker
# ---------------------------------------------------------------------------


class EventIdTracker:
    """Thread-safe tracker for notification event IDs with dedup filtering.

    Maintains a bounded, insertion-ordered set of recently seen event IDs.
    Provides both recording (mutating) and query-only (non-mutating)
    interfaces for deduplication checks.

    The ``last_seen_event_id`` property always reflects the most recently
    recorded event ID, enabling SSE-style resume semantics.

    Thread safety: All mutable state is guarded by a ``threading.Lock``.
    Safe for concurrent access from multiple threads.

    Args:
        config: Optional deduplication configuration. Uses default
            capacity when None.
    """

    __slots__ = ("_config", "_lock", "_seen", "_last_seen")

    def __init__(
        self,
        config: EventDeduplicationConfig | None = None,
    ) -> None:
        self._config = config or EventDeduplicationConfig()
        self._lock = threading.Lock()
        # OrderedDict for O(1) lookup + FIFO eviction
        # Keys are event IDs, values are True (sentinel)
        self._seen: OrderedDict[str, bool] = OrderedDict()
        self._last_seen: str | None = None

    # -- Properties -----------------------------------------------------------

    @property
    def last_seen_event_id(self) -> str | None:
        """The most recently recorded event ID, or None if empty.

        This value is updated on every call to ``record()`` or
        ``filter_envelope()``, even for duplicates. It represents
        the latest event the tracker has processed, not the latest
        unique event.
        """
        with self._lock:
            return self._last_seen

    @property
    def tracked_count(self) -> int:
        """Number of unique event IDs currently tracked."""
        with self._lock:
            return len(self._seen)

    # -- Recording (mutating) -------------------------------------------------

    def record(self, event_id: str) -> DeduplicationVerdict:
        """Record an event ID and return whether it was a duplicate.

        If the event ID is new, it is added to the tracked set. If
        it is already tracked, it is marked as a duplicate. In both
        cases, ``last_seen_event_id`` is updated to this event ID.

        When recording a new event causes the tracked set to exceed
        ``max_tracked_ids``, the oldest entry is evicted (FIFO).

        Args:
            event_id: The event ID to record.

        Returns:
            Immutable verdict indicating whether the event was a
            duplicate.

        Raises:
            ValueError: If event_id is empty or whitespace-only.
        """
        validated_id = _validate_event_id(event_id)

        with self._lock:
            self._last_seen = validated_id

            if validated_id in self._seen:
                # Move to end to reflect recent access (still FIFO for
                # eviction based on first-seen order is simpler, but
                # we keep insertion order and just return duplicate)
                logger.debug("Duplicate event ID: %s", validated_id)
                return DeduplicationVerdict(
                    event_id=validated_id,
                    is_duplicate=True,
                )

            # New event ID -- add and enforce capacity
            self._seen[validated_id] = True
            self._enforce_capacity()

            logger.debug(
                "Recorded new event ID: %s (tracked=%d)",
                validated_id,
                len(self._seen),
            )
            return DeduplicationVerdict(
                event_id=validated_id,
                is_duplicate=False,
            )

    def filter_envelope(
        self,
        envelope: NotificationEnvelope,
    ) -> DeduplicationVerdict:
        """Record an envelope's event_id and return a dedup verdict.

        Convenience method that extracts the ``event_id`` from a
        ``NotificationEnvelope`` and delegates to ``record()``.

        Args:
            envelope: The notification envelope to check.

        Returns:
            Immutable verdict indicating whether the envelope's
            event was a duplicate.
        """
        return self.record(envelope.event_id)

    # -- Query (non-mutating) -------------------------------------------------

    def is_duplicate(self, event_id: str) -> bool:
        """Check whether an event ID has been recorded, without recording it.

        This is a read-only query. It does not modify the tracked set
        or update ``last_seen_event_id``.

        Args:
            event_id: The event ID to check.

        Returns:
            True if the event ID is currently in the tracked set.

        Raises:
            ValueError: If event_id is empty or whitespace-only.
        """
        validated_id = _validate_event_id(event_id)
        with self._lock:
            return validated_id in self._seen

    def contains(self, event_id: str) -> bool:
        """Check whether an event ID is currently tracked.

        Alias for ``is_duplicate()`` with a more descriptive name
        for membership checks.

        Args:
            event_id: The event ID to check.

        Returns:
            True if the event ID is in the tracked set.
        """
        with self._lock:
            return event_id in self._seen

    # -- State management -----------------------------------------------------

    def clear(self) -> int:
        """Remove all tracked event IDs and reset last_seen.

        Returns:
            Number of event IDs that were removed.
        """
        with self._lock:
            count = len(self._seen)
            self._seen.clear()
            self._last_seen = None

        if count > 0:
            logger.debug("Cleared %d tracked event IDs", count)
        return count

    # -- Internal helpers -----------------------------------------------------

    def _enforce_capacity(self) -> None:
        """Evict oldest entries when capacity is exceeded.

        Must be called under the lock. Uses FIFO eviction via
        ``OrderedDict.popitem(last=False)``.
        """
        limit = self._config.max_tracked_ids
        while len(self._seen) > limit:
            evicted_id, _ = self._seen.popitem(last=False)
            logger.debug(
                "Evicted oldest event ID: %s (capacity=%d)",
                evicted_id,
                limit,
            )

    # -- Dunder methods -------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._seen)
        return f"EventIdTracker(tracking {count} event IDs)"
