"""Periodic background sweep for orphaned and stale notification subscribers.

Real-time disconnect detection (via ``disconnect_event`` and the event bus)
handles the common case: when a CLI client disconnects, its subscribers are
cleaned up immediately. However, edge cases can leave orphaned subscribers:

- A disconnect event was lost or not classified correctly.
- The cleanup handler raised an exception and failed silently.
- A client crashed mid-handshake before the disconnect path could fire.
- Network timeouts that produce ambiguous disconnect signals.

This module provides a periodic sweep that runs as an ``asyncio.Task`` and
catches these stragglers. It cross-references active notification subscribers
with the ``ConnectionManager`` client registry and applies configurable
staleness heuristics.

Staleness criteria (evaluated in priority order):

1. **Orphaned (no client)**: The subscriber's associated client is no longer
   registered in the ``ConnectionManager``. This is the primary safety-net
   for missed disconnect events.

2. **Excessive failures**: The subscriber's consecutive delivery failure count
   in the broadcaster has reached or exceeded a configurable threshold.
   These subscribers are likely dead but haven't yet crossed the
   broadcaster's auto-removal limit.

3. **Idle timeout**: The subscriber has been registered for longer than a
   configurable maximum age without any recorded activity. Covers the case
   where a subscriber was created but the client never polled for events.

All data types are frozen dataclasses following the project-wide immutability
convention. The sweep itself is idempotent: running it twice in quick
succession is harmless.

Architecture::

    StaleSubscriberSweep
        |
        |-- periodic asyncio.Task (configurable interval)
        |       |
        |       v
        |   detect_stale_subscribers()
        |       |-- cross-reference broadcaster subscribers with
        |       |   ConnectionManager client registry
        |       |-- check failure counts in broadcaster
        |       |-- check registration age from metadata tracker
        |       v
        |   sweep_stale_subscribers()
        |       |-- cleanup_subscriber() for each detected subscriber
        |       v
        |   SweepResult (immutable)
        |
        |-- register_subscriber(subscription_id, client_id)
        |-- record_activity(subscription_id)
        |-- deregister_subscriber(subscription_id)

Usage::

    from jules_daemon.cleanup.subscriber_sweep import (
        StaleSubscriberSweep,
        SweepConfig,
    )

    sweep = StaleSubscriberSweep(
        broadcaster=notification_broadcaster,
        connection_manager=connection_manager,
        config=SweepConfig(sweep_interval_seconds=60.0),
    )

    # Register subscribers as they are created
    sweep.register_subscriber("nsub-abc123", client_id="client-xyz")

    # Start the background sweep task
    await sweep.start()

    # On shutdown
    await sweep.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from jules_daemon.cleanup.subscriber_cleanup import (
    SubscriberCleanupResult,
    cleanup_subscriber,
)
from jules_daemon.ipc.connection_manager import ConnectionManager
from jules_daemon.ipc.notification_broadcaster import NotificationBroadcaster

__all__ = [
    "StaleSubscriberDetection",
    "StaleSubscriberReason",
    "StaleSubscriberSweep",
    "SubscriberMetadata",
    "SweepConfig",
    "SweepResult",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SWEEP_INTERVAL = 60.0
_DEFAULT_MAX_IDLE_SECONDS = 300.0
_DEFAULT_FAILURE_THRESHOLD = 3
_SWEEP_ID_PREFIX = "sweep-"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StaleSubscriberReason(Enum):
    """Classification of why a subscriber was determined to be stale.

    Values:
        ORPHANED_NO_CLIENT: The subscriber's associated client is no longer
            registered in the ConnectionManager.
        EXCESSIVE_FAILURES: The subscriber's consecutive failure count in
            the broadcaster has met or exceeded the configured threshold.
        IDLE_TIMEOUT: The subscriber has been registered for longer than
            the configured maximum age without recorded activity.
    """

    ORPHANED_NO_CLIENT = "orphaned_no_client"
    EXCESSIVE_FAILURES = "excessive_failures"
    IDLE_TIMEOUT = "idle_timeout"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    """Immutable configuration for the stale subscriber sweep.

    Attributes:
        sweep_interval_seconds: How often the background sweep runs
            (in seconds). Must be positive.
        max_idle_seconds: Maximum time (in seconds) a subscriber can
            remain idle (no recorded activity) before being considered
            stale. Must be positive. This measures time since last
            activity, not time since registration.
        failure_count_threshold: Minimum consecutive failure count in the
            broadcaster to consider a subscriber stale. Must be positive.
            Set to a value lower than the broadcaster's auto-removal
            threshold to catch failing subscribers earlier.
        enabled: Whether the periodic sweep task is active. When False,
            ``start()`` is a no-op. Manual ``sweep_once()`` still works.
    """

    sweep_interval_seconds: float = _DEFAULT_SWEEP_INTERVAL
    max_idle_seconds: float = _DEFAULT_MAX_IDLE_SECONDS
    failure_count_threshold: int = _DEFAULT_FAILURE_THRESHOLD
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.sweep_interval_seconds <= 0:
            raise ValueError(
                f"sweep_interval_seconds must be positive, "
                f"got {self.sweep_interval_seconds}"
            )
        if self.max_idle_seconds <= 0:
            raise ValueError(
                f"max_idle_seconds must be positive, "
                f"got {self.max_idle_seconds}"
            )
        if self.failure_count_threshold < 1:
            raise ValueError(
                f"failure_count_threshold must be at least 1, "
                f"got {self.failure_count_threshold}"
            )


# ---------------------------------------------------------------------------
# Subscriber metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubscriberMetadata:
    """Immutable metadata tracking a subscriber's lifecycle timestamps.

    Maintained by the sweep's internal tracker to enable staleness
    detection based on age and activity.

    Attributes:
        subscription_id: The notification subscriber identifier.
        client_id: The associated IPC client identifier. None if the
            subscriber was created without client association (e.g.,
            internal daemon use).
        registered_at: UTC datetime when the subscriber was registered.
        last_active_at: UTC datetime of the most recent activity
            (successful delivery, receive, or explicit touch).
    """

    subscription_id: str
    client_id: str | None
    registered_at: datetime
    last_active_at: datetime

    def __post_init__(self) -> None:
        if (
            not isinstance(self.subscription_id, str)
            or not self.subscription_id.strip()
        ):
            raise ValueError("subscription_id must not be empty")
        if self.registered_at.tzinfo is None:
            raise ValueError("registered_at must be timezone-aware")
        if self.last_active_at.tzinfo is None:
            raise ValueError("last_active_at must be timezone-aware")

    def age_seconds(self, now: datetime) -> float:
        """Compute seconds elapsed since registration.

        Args:
            now: Reference time for the calculation.

        Returns:
            Non-negative age in seconds.
        """
        delta = now - self.registered_at
        return max(0.0, delta.total_seconds())

    def idle_seconds(self, now: datetime) -> float:
        """Compute seconds elapsed since last activity.

        Args:
            now: Reference time for the calculation.

        Returns:
            Non-negative idle time in seconds.
        """
        delta = now - self.last_active_at
        return max(0.0, delta.total_seconds())


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaleSubscriberDetection:
    """Immutable result of staleness detection for a single subscriber.

    Attributes:
        subscription_id: The subscriber that was evaluated.
        reason: Classification of why the subscriber is stale.
        reason_detail: Human-readable description of the condition.
        detected_at: UTC timestamp of the detection.
    """

    subscription_id: str
    reason: StaleSubscriberReason
    reason_detail: str
    detected_at: datetime

    def __post_init__(self) -> None:
        if (
            not isinstance(self.subscription_id, str)
            or not self.subscription_id.strip()
        ):
            raise ValueError("subscription_id must not be empty")
        if not isinstance(self.reason_detail, str) or not self.reason_detail.strip():
            raise ValueError("reason_detail must not be empty")


# ---------------------------------------------------------------------------
# Sweep result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepResult:
    """Immutable result of a single sweep operation.

    Reports what the sweep found, what it cleaned up, and how long
    it took. Designed for logging and audit trail purposes.

    Attributes:
        sweep_id: Unique identifier for this sweep invocation.
        swept_at: UTC datetime when the sweep started.
        subscribers_checked: Total subscribers evaluated.
        stale_detected: Number of subscribers found to be stale.
        removed_count: Number of subscribers successfully removed.
        removal_results: Per-subscriber cleanup outcomes.
        detections: Detailed detection records for stale subscribers.
        errors: Error messages from failed removal attempts.
        duration_ms: Wall-clock duration of the sweep in milliseconds.
    """

    sweep_id: str
    swept_at: datetime
    subscribers_checked: int
    stale_detected: int
    removed_count: int
    removal_results: tuple[SubscriberCleanupResult, ...] = ()
    detections: tuple[StaleSubscriberDetection, ...] = ()
    errors: tuple[str, ...] = ()
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.sweep_id, str) or not self.sweep_id.strip():
            raise ValueError("sweep_id must not be empty")
        if self.subscribers_checked < 0:
            raise ValueError("subscribers_checked must not be negative")
        if self.stale_detected < 0:
            raise ValueError("stale_detected must not be negative")
        if self.removed_count < 0:
            raise ValueError("removed_count must not be negative")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must not be negative")

    @property
    def has_errors(self) -> bool:
        """True if any removal attempt encountered an error."""
        return len(self.errors) > 0

    @property
    def clean(self) -> bool:
        """True if no stale subscribers were found."""
        return self.stale_detected == 0


# ---------------------------------------------------------------------------
# Metadata tracker
# ---------------------------------------------------------------------------


class _MetadataTracker:
    """Internal mutable tracker for subscriber lifecycle metadata.

    Not part of the public API. Used by ``StaleSubscriberSweep`` to
    maintain registration timestamps and client associations.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, SubscriberMetadata] = {}

    def register(
        self,
        subscription_id: str,
        client_id: str | None,
        now: datetime,
    ) -> SubscriberMetadata:
        """Register a new subscriber with timestamps.

        Args:
            subscription_id: The subscriber identifier.
            client_id: The associated client identifier (may be None).
            now: Registration timestamp.

        Returns:
            The created SubscriberMetadata.
        """
        metadata = SubscriberMetadata(
            subscription_id=subscription_id,
            client_id=client_id,
            registered_at=now,
            last_active_at=now,
        )
        self._entries = {**self._entries, subscription_id: metadata}
        return metadata

    def record_activity(
        self,
        subscription_id: str,
        now: datetime,
    ) -> bool:
        """Update the last_active_at timestamp for a subscriber.

        Args:
            subscription_id: The subscriber to update.
            now: Activity timestamp.

        Returns:
            True if the subscriber was found and updated, False if not found.
        """
        existing = self._entries.get(subscription_id)
        if existing is None:
            return False
        updated = SubscriberMetadata(
            subscription_id=existing.subscription_id,
            client_id=existing.client_id,
            registered_at=existing.registered_at,
            last_active_at=now,
        )
        self._entries = {**self._entries, subscription_id: updated}
        return True

    def deregister(self, subscription_id: str) -> bool:
        """Remove a subscriber from the tracker.

        Args:
            subscription_id: The subscriber to remove.

        Returns:
            True if the subscriber was found and removed.
        """
        if subscription_id not in self._entries:
            return False
        self._entries = {
            sid: meta
            for sid, meta in self._entries.items()
            if sid != subscription_id
        }
        return True

    def get(self, subscription_id: str) -> SubscriberMetadata | None:
        """Look up metadata for a subscriber."""
        return self._entries.get(subscription_id)

    def all_metadata(self) -> tuple[SubscriberMetadata, ...]:
        """Return all tracked metadata as a tuple."""
        return tuple(self._entries.values())

    @property
    def count(self) -> int:
        """Number of tracked subscribers."""
        return len(self._entries)


# ---------------------------------------------------------------------------
# Pure detection function
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def detect_stale_subscribers(
    *,
    broadcaster: NotificationBroadcaster,
    connection_manager: ConnectionManager | None,
    metadata_tracker: _MetadataTracker,
    config: SweepConfig,
    now: datetime | None = None,
) -> tuple[StaleSubscriberDetection, ...]:
    """Identify stale subscribers without removing them.

    Evaluates each subscriber in the broadcaster against three staleness
    criteria (in priority order):

    1. Orphaned: subscriber's client_id not in ConnectionManager.
    2. Excessive failures: broadcaster failure count >= threshold.
    3. Idle timeout: subscriber idle time exceeds max_idle_seconds.

    Only the first matching reason is reported per subscriber (priority
    order). Subscribers without metadata in the tracker are evaluated
    only for criteria 1 and 2 (no age data available).

    Args:
        broadcaster: The NotificationBroadcaster to inspect.
        connection_manager: Optional ConnectionManager for orphan detection.
            When None, orphan detection is skipped.
        metadata_tracker: Internal tracker with subscriber lifecycle data.
        config: Sweep configuration with thresholds.
        now: Reference time. Defaults to current UTC.

    Returns:
        Tuple of StaleSubscriberDetection for all stale subscribers found.
    """
    if now is None:
        now = _now_utc()

    subscriber_ids = broadcaster.list_subscriber_ids()
    detections: list[StaleSubscriberDetection] = []

    for sub_id in subscriber_ids:
        metadata = metadata_tracker.get(sub_id)

        # --- Check 1: Orphaned (no client) ---
        if connection_manager is not None and metadata is not None:
            if (
                metadata.client_id is not None
                and not connection_manager.has_client(metadata.client_id)
            ):
                detail = (
                    f"Client {metadata.client_id!r} no longer registered; "
                    f"subscriber {sub_id!r} is orphaned"
                )
                detections.append(
                    StaleSubscriberDetection(
                        subscription_id=sub_id,
                        reason=StaleSubscriberReason.ORPHANED_NO_CLIENT,
                        reason_detail=detail,
                        detected_at=now,
                    )
                )
                continue

        # --- Check 2: Excessive failures ---
        failure_count = broadcaster.get_failure_count(sub_id)
        if failure_count >= config.failure_count_threshold:
            detail = (
                f"Consecutive delivery failures: {failure_count} "
                f"(threshold: {config.failure_count_threshold})"
            )
            detections.append(
                StaleSubscriberDetection(
                    subscription_id=sub_id,
                    reason=StaleSubscriberReason.EXCESSIVE_FAILURES,
                    reason_detail=detail,
                    detected_at=now,
                )
            )
            continue

        # --- Check 3: Idle timeout ---
        if metadata is not None:
            idle_secs = metadata.idle_seconds(now)
            if idle_secs > config.max_idle_seconds:
                detail = (
                    f"Subscriber idle for {idle_secs:.1f}s "
                    f"(threshold: {config.max_idle_seconds:.1f}s)"
                )
                detections.append(
                    StaleSubscriberDetection(
                        subscription_id=sub_id,
                        reason=StaleSubscriberReason.IDLE_TIMEOUT,
                        reason_detail=detail,
                        detected_at=now,
                    )
                )
                continue

    return tuple(detections)


# ---------------------------------------------------------------------------
# Sweep function (detect + remove)
# ---------------------------------------------------------------------------


async def sweep_stale_subscribers(
    *,
    broadcaster: NotificationBroadcaster,
    connection_manager: ConnectionManager | None,
    metadata_tracker: _MetadataTracker,
    config: SweepConfig,
    now: datetime | None = None,
) -> SweepResult:
    """Detect and remove stale subscribers from the broadcaster.

    Runs ``detect_stale_subscribers`` and then calls ``cleanup_subscriber``
    for each stale subscriber found. Returns an immutable result with
    statistics and per-subscriber outcomes.

    Args:
        broadcaster: The NotificationBroadcaster to clean.
        connection_manager: Optional ConnectionManager for orphan detection.
        metadata_tracker: Internal tracker with subscriber lifecycle data.
        config: Sweep configuration with thresholds.
        now: Reference time. Defaults to current UTC.

    Returns:
        Immutable SweepResult with detection and cleanup outcomes.
    """
    if now is None:
        now = _now_utc()

    sweep_id = f"{_SWEEP_ID_PREFIX}{uuid.uuid4().hex[:12]}"
    start_time = time.monotonic()

    subscribers_checked = broadcaster.subscriber_count
    detections = detect_stale_subscribers(
        broadcaster=broadcaster,
        connection_manager=connection_manager,
        metadata_tracker=metadata_tracker,
        config=config,
        now=now,
    )

    removal_results: list[SubscriberCleanupResult] = []
    errors: list[str] = []
    removed_count = 0

    for detection in detections:
        try:
            cleanup_result = await cleanup_subscriber(
                broadcaster=broadcaster,
                subscriber_id=detection.subscription_id,
            )
            removal_results.append(cleanup_result)
            if cleanup_result.found:
                removed_count += 1
                # Also remove from metadata tracker
                metadata_tracker.deregister(detection.subscription_id)
                logger.info(
                    "Sweep removed stale subscriber %s (reason=%s): %s",
                    detection.subscription_id,
                    detection.reason.value,
                    detection.reason_detail,
                )
        except Exception as exc:
            error_msg = (
                f"Failed to clean subscriber {detection.subscription_id!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            errors.append(error_msg)
            logger.exception(
                "Sweep cleanup error for %s", detection.subscription_id
            )

    elapsed_ms = (time.monotonic() - start_time) * 1000.0

    sweep_result = SweepResult(
        sweep_id=sweep_id,
        swept_at=now,
        subscribers_checked=subscribers_checked,
        stale_detected=len(detections),
        removed_count=removed_count,
        removal_results=tuple(removal_results),
        detections=detections,
        errors=tuple(errors),
        duration_ms=elapsed_ms,
    )

    if sweep_result.stale_detected > 0:
        logger.info(
            "Sweep %s: checked=%d, stale=%d, removed=%d, errors=%d "
            "(%.1fms)",
            sweep_id,
            subscribers_checked,
            len(detections),
            removed_count,
            len(errors),
            elapsed_ms,
        )
    else:
        logger.debug(
            "Sweep %s: checked=%d, all clean (%.1fms)",
            sweep_id,
            subscribers_checked,
            elapsed_ms,
        )

    return sweep_result


# ---------------------------------------------------------------------------
# Background sweep task
# ---------------------------------------------------------------------------


class StaleSubscriberSweep:
    """Periodic background sweep for orphaned and stale subscribers.

    Manages a metadata tracker for subscriber lifecycle timestamps and
    runs a periodic ``asyncio.Task`` that invokes ``sweep_stale_subscribers``
    at the configured interval.

    The sweep is a safety net: it catches subscribers that slip through
    real-time disconnect detection. It is designed to be lightweight and
    idempotent.

    Args:
        broadcaster: The NotificationBroadcaster to monitor and clean.
        connection_manager: Optional ConnectionManager for orphan detection.
            When None, only failure-count and idle-timeout criteria apply.
        config: Optional sweep configuration. Uses defaults when None.
    """

    def __init__(
        self,
        *,
        broadcaster: NotificationBroadcaster,
        connection_manager: ConnectionManager | None = None,
        config: SweepConfig | None = None,
    ) -> None:
        self._broadcaster = broadcaster
        self._connection_manager = connection_manager
        self._config = config or SweepConfig()
        self._tracker = _MetadataTracker()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._sweep_count = 0
        self._last_result: SweepResult | None = None

    # -- Properties -----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the background sweep task is currently active."""
        return self._running and self._task is not None and not self._task.done()

    @property
    def sweep_count(self) -> int:
        """Number of sweep cycles completed since start."""
        return self._sweep_count

    @property
    def last_result(self) -> SweepResult | None:
        """Result of the most recent sweep, or None if no sweep has run."""
        return self._last_result

    @property
    def tracked_subscriber_count(self) -> int:
        """Number of subscribers currently tracked in the metadata store."""
        return self._tracker.count

    # -- Subscriber registration ----------------------------------------------

    def register_subscriber(
        self,
        subscription_id: str,
        *,
        client_id: str | None = None,
        now: datetime | None = None,
    ) -> SubscriberMetadata:
        """Register a subscriber for sweep tracking.

        Should be called when a new notification subscriber is created,
        passing the associated client_id for orphan detection.

        Args:
            subscription_id: The subscriber identifier.
            client_id: The associated IPC client identifier.
            now: Registration timestamp. Defaults to current UTC.

        Returns:
            The created SubscriberMetadata record.
        """
        if now is None:
            now = _now_utc()
        return self._tracker.register(subscription_id, client_id, now)

    def record_activity(
        self,
        subscription_id: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Record activity for a subscriber to prevent idle timeout.

        Should be called when a subscriber successfully receives an event
        or is otherwise confirmed active.

        Args:
            subscription_id: The subscriber identifier.
            now: Activity timestamp. Defaults to current UTC.

        Returns:
            True if the subscriber was found and updated.
        """
        if now is None:
            now = _now_utc()
        return self._tracker.record_activity(subscription_id, now)

    def deregister_subscriber(self, subscription_id: str) -> bool:
        """Remove a subscriber from sweep tracking.

        Should be called when a subscriber is explicitly unsubscribed
        (as opposed to being removed by the sweep).

        Args:
            subscription_id: The subscriber to deregister.

        Returns:
            True if the subscriber was found and removed.
        """
        return self._tracker.deregister(subscription_id)

    def get_metadata(self, subscription_id: str) -> SubscriberMetadata | None:
        """Look up metadata for a tracked subscriber.

        Args:
            subscription_id: The subscriber to look up.

        Returns:
            The subscriber's metadata, or None if not tracked.
        """
        return self._tracker.get(subscription_id)

    # -- Lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        """Start the periodic background sweep task.

        No-op if the sweep is already running or if ``config.enabled``
        is False. Idempotent.
        """
        if not self._config.enabled:
            logger.info("Stale subscriber sweep is disabled by config")
            return

        if self.is_running:
            logger.debug("Stale subscriber sweep is already running")
            return

        self._running = True
        self._task = asyncio.create_task(
            self._sweep_loop(),
            name="stale-subscriber-sweep",
        )
        logger.info(
            "Stale subscriber sweep started (interval=%.1fs)",
            self._config.sweep_interval_seconds,
        )

    async def stop(self) -> None:
        """Stop the periodic background sweep task.

        Cancels the background task and waits for it to finish. Safe
        to call even if the sweep is not running (no-op).
        """
        self._running = False

        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self._task = None
        logger.info("Stale subscriber sweep stopped")

    async def sweep_once(self) -> SweepResult:
        """Run a single sweep cycle manually.

        Does not require the periodic task to be running. Useful for
        testing and on-demand cleanup.

        Returns:
            The sweep result for this cycle.
        """
        result = await sweep_stale_subscribers(
            broadcaster=self._broadcaster,
            connection_manager=self._connection_manager,
            metadata_tracker=self._tracker,
            config=self._config,
        )
        self._sweep_count += 1
        self._last_result = result
        return result

    # -- Internal loop --------------------------------------------------------

    async def _sweep_loop(self) -> None:
        """Background loop that runs sweeps at the configured interval.

        Catches and logs all exceptions to prevent the task from dying
        on transient errors. Respects the ``_running`` flag for clean
        shutdown.
        """
        logger.debug("Sweep loop started")
        try:
            while self._running:
                await asyncio.sleep(self._config.sweep_interval_seconds)

                if not self._running:
                    break

                try:
                    await self.sweep_once()
                except Exception:
                    logger.exception("Unexpected error during subscriber sweep")
        except asyncio.CancelledError:
            logger.debug("Sweep loop cancelled")
            raise
        finally:
            logger.debug("Sweep loop exited")
