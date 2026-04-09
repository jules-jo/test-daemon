"""Stale SSH session detector.

Monitors active SSH sessions for staleness using heartbeat/keepalive
timeouts and transport-layer status checks. Marks sessions as stale
with a timestamp and reason when any of the following conditions hold:

1. **Heartbeat timeout**: No successful heartbeat within the configured
   timeout window (default 30 seconds).
2. **Transport disconnected**: The most recent transport-layer liveness
   probe returned DISCONNECTED, indicating the SSH connection is lost.
3. **Max failures exceeded**: The number of consecutive probe failures
   has reached the configured maximum (default 3).

All data structures are immutable (frozen dataclasses). State
transitions produce new instances -- nothing is mutated in place.

The detector does NOT own the probe execution loop. It receives
pre-computed ProbeResult snapshots and evaluates staleness based on
the heartbeat record and probe outcome. The polling loop
(``monitor.polling_loop``) or a dedicated keepalive scheduler is
responsible for running probes at the configured interval.

Usage:
    from jules_daemon.monitor.stale_session_detector import (
        DetectorConfig,
        HeartbeatRecord,
        HeartbeatTracker,
        detect_session_staleness,
        detect_batch_staleness,
    )

    config = DetectorConfig(heartbeat_timeout_seconds=30.0)
    record = HeartbeatRecord(session_id="run-abc", last_heartbeat_at=now)

    # After running a liveness probe:
    detection = detect_session_staleness(
        record=record,
        probe_result=probe_result,
        config=config,
        now=datetime.now(timezone.utc),
    )
    if detection.is_stale:
        logger.warning("Session %s is stale: %s", detection.session_id,
                        detection.reason_detail)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from jules_daemon.ssh.liveness import ConnectionHealth, ProbeResult

__all__ = [
    "DetectorConfig",
    "HeartbeatRecord",
    "HeartbeatTracker",
    "StalenessDetection",
    "StalenessReason",
    "detect_batch_staleness",
    "detect_session_staleness",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HEARTBEAT_TIMEOUT = 30.0
_DEFAULT_KEEPALIVE_INTERVAL = 10.0
_DEFAULT_MAX_MISSED_HEARTBEATS = 3
_DEFAULT_TRANSPORT_CHECK_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StalenessReason(Enum):
    """Classification of why a session was determined to be stale.

    Values:
        HEARTBEAT_TIMEOUT: No successful heartbeat within the timeout window.
        TRANSPORT_DISCONNECTED: SSH transport layer reported disconnected.
        MAX_FAILURES_EXCEEDED: Consecutive liveness probe failures reached
            the configured maximum.
    """

    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    TRANSPORT_DISCONNECTED = "transport_disconnected"
    MAX_FAILURES_EXCEEDED = "max_failures_exceeded"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorConfig:
    """Immutable configuration for the stale session detector.

    Attributes:
        heartbeat_timeout_seconds: Maximum elapsed time since the last
            successful heartbeat before a session is considered stale.
            Must be positive.
        keepalive_interval_seconds: How often keepalive probes should be
            sent. This is advisory metadata for the caller -- the detector
            itself does not schedule probes. Must be positive.
        max_missed_heartbeats: Number of consecutive probe failures before
            the session is declared stale via MAX_FAILURES_EXCEEDED. When
            set to 0, this check is disabled (failure count never triggers
            staleness). Must be non-negative.
        transport_check_timeout_seconds: Timeout for individual transport-
            layer liveness probes. Must be positive.
    """

    heartbeat_timeout_seconds: float = _DEFAULT_HEARTBEAT_TIMEOUT
    keepalive_interval_seconds: float = _DEFAULT_KEEPALIVE_INTERVAL
    max_missed_heartbeats: int = _DEFAULT_MAX_MISSED_HEARTBEATS
    transport_check_timeout_seconds: float = _DEFAULT_TRANSPORT_CHECK_TIMEOUT

    def __post_init__(self) -> None:
        if self.heartbeat_timeout_seconds <= 0:
            raise ValueError(
                f"heartbeat_timeout_seconds must be positive, "
                f"got {self.heartbeat_timeout_seconds}"
            )
        if self.keepalive_interval_seconds <= 0:
            raise ValueError(
                f"keepalive_interval_seconds must be positive, "
                f"got {self.keepalive_interval_seconds}"
            )
        if self.max_missed_heartbeats < 0:
            raise ValueError(
                f"max_missed_heartbeats must be non-negative, "
                f"got {self.max_missed_heartbeats}"
            )
        if self.transport_check_timeout_seconds <= 0:
            raise ValueError(
                f"transport_check_timeout_seconds must be positive, "
                f"got {self.transport_check_timeout_seconds}"
            )


# ---------------------------------------------------------------------------
# Heartbeat record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeartbeatRecord:
    """Immutable record tracking the heartbeat state of a single SSH session.

    Each probe cycle produces a new HeartbeatRecord via the ``with_heartbeat``
    or ``with_failure`` methods -- the original is never mutated.

    Attributes:
        session_id: Unique identifier for the SSH session.
        last_heartbeat_at: UTC timestamp of the last successful heartbeat
            (probe that returned CONNECTED).
        consecutive_failures: Number of consecutive probe failures since
            the last successful heartbeat. Reset to 0 on success.
        last_transport_health: Health classification from the most recent
            transport-layer probe. None if no probe has been performed.
    """

    session_id: str
    last_heartbeat_at: datetime
    consecutive_failures: int = 0
    last_transport_health: ConnectionHealth | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must not be empty")

    def heartbeat_age_seconds(self, now: datetime) -> float:
        """Compute seconds elapsed since the last successful heartbeat.

        Args:
            now: Reference time for the calculation.

        Returns:
            Non-negative age in seconds.
        """
        delta = now - self.last_heartbeat_at
        return max(0.0, delta.total_seconds())

    def with_heartbeat(self, heartbeat_at: datetime) -> HeartbeatRecord:
        """Return a new record with a successful heartbeat recorded.

        Resets consecutive_failures to 0 and sets transport health to
        CONNECTED.

        Args:
            heartbeat_at: UTC timestamp of the successful heartbeat.

        Returns:
            New HeartbeatRecord with updated fields.
        """
        return replace(
            self,
            last_heartbeat_at=heartbeat_at,
            consecutive_failures=0,
            last_transport_health=ConnectionHealth.CONNECTED,
        )

    def with_failure(
        self,
        *,
        health: ConnectionHealth,
        checked_at: datetime,
    ) -> HeartbeatRecord:
        """Return a new record with a probe failure recorded.

        Increments consecutive_failures. Does NOT advance
        last_heartbeat_at, since the heartbeat was not successful.

        Args:
            health: Transport health from the failed probe.
            checked_at: UTC timestamp of the failed probe (unused for
                heartbeat advancement but recorded for diagnostics).

        Returns:
            New HeartbeatRecord with incremented failure count.
        """
        return replace(
            self,
            consecutive_failures=self.consecutive_failures + 1,
            last_transport_health=health,
        )


# ---------------------------------------------------------------------------
# Heartbeat tracker (immutable collection)
# ---------------------------------------------------------------------------


class HeartbeatTracker:
    """Immutable container tracking heartbeat records for multiple sessions.

    Uses copy-on-write semantics: every mutation method returns a new
    HeartbeatTracker instance, leaving the original unchanged.

    This is a lightweight in-memory structure. The authoritative state
    lives in the wiki -- this tracker is a runtime cache that can be
    rebuilt from wiki data on daemon restart.
    """

    __slots__ = ("_records",)

    def __init__(
        self,
        records: Mapping[str, HeartbeatRecord] | None = None,
    ) -> None:
        if records is None:
            self._records: Mapping[str, HeartbeatRecord] = MappingProxyType({})
        elif isinstance(records, MappingProxyType):
            self._records = records
        else:
            self._records = MappingProxyType(dict(records))

    @property
    def session_count(self) -> int:
        """Number of tracked sessions."""
        return len(self._records)

    def get(self, session_id: str) -> HeartbeatRecord | None:
        """Look up a heartbeat record by session ID.

        Returns None if the session is not tracked.
        """
        return self._records.get(session_id)

    def all_records(self) -> tuple[HeartbeatRecord, ...]:
        """Return all tracked heartbeat records as a tuple."""
        return tuple(self._records.values())

    def register(self, record: HeartbeatRecord) -> HeartbeatTracker:
        """Return a new tracker with the given record added or replaced.

        Args:
            record: HeartbeatRecord to register.

        Returns:
            New HeartbeatTracker with the record included.
        """
        new_data = dict(self._records)
        new_data[record.session_id] = record
        return HeartbeatTracker(new_data)

    def update(self, record: HeartbeatRecord) -> HeartbeatTracker:
        """Return a new tracker with an existing record replaced.

        Unlike ``register``, this raises KeyError if the session is not
        already tracked. Use this when you expect the session to exist.

        Args:
            record: Updated HeartbeatRecord.

        Returns:
            New HeartbeatTracker with the record replaced.

        Raises:
            KeyError: If the session_id is not in the tracker.
        """
        if record.session_id not in self._records:
            raise KeyError(
                f"Session {record.session_id!r} not found in tracker. "
                f"Use register() for new sessions."
            )
        new_data = dict(self._records)
        new_data[record.session_id] = record
        return HeartbeatTracker(new_data)

    def remove(self, session_id: str) -> HeartbeatTracker:
        """Return a new tracker with the given session removed.

        If the session is not tracked, returns a copy unchanged.

        Args:
            session_id: Session to remove.

        Returns:
            New HeartbeatTracker without the specified session.
        """
        if session_id not in self._records:
            return HeartbeatTracker(self._records)
        new_data = dict(self._records)
        del new_data[session_id]
        return HeartbeatTracker(new_data)


# ---------------------------------------------------------------------------
# Staleness detection result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StalenessDetection:
    """Immutable result of a staleness detection check for one session.

    Produced by ``detect_session_staleness()`` and contains all the
    information needed to decide whether to mark the session as stale
    in the wiki.

    Attributes:
        session_id: Identifier of the session that was checked.
        is_stale: True if the session was determined to be stale.
        reason: Classification of why the session is stale. None when
            the session is healthy.
        reason_detail: Human-readable description of the staleness
            condition. None when the session is healthy.
        detected_at: UTC timestamp when the detection was performed.
        heartbeat_age_seconds: Seconds elapsed since the last successful
            heartbeat at detection time.
        transport_health: Health from the most recent transport probe.
            None if no probe result was available.
        consecutive_failures: Number of consecutive probe failures
            (including the current one if it failed).
    """

    session_id: str
    is_stale: bool
    reason: StalenessReason | None
    reason_detail: str | None
    detected_at: datetime
    heartbeat_age_seconds: float
    transport_health: ConnectionHealth | None
    consecutive_failures: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _compute_failures_after_probe(
    current_failures: int,
    probe_result: ProbeResult | None,
) -> tuple[int, ConnectionHealth | None]:
    """Compute updated failure count and health after a probe.

    Returns:
        Tuple of (new_consecutive_failures, transport_health).
    """
    if probe_result is None:
        return (current_failures, None)

    if probe_result.success and probe_result.health == ConnectionHealth.CONNECTED:
        return (0, ConnectionHealth.CONNECTED)

    # Probe did not fully succeed -- increment failure count
    return (current_failures + 1, probe_result.health)


# ---------------------------------------------------------------------------
# Public API: single session detection
# ---------------------------------------------------------------------------


def detect_session_staleness(
    *,
    record: HeartbeatRecord,
    probe_result: ProbeResult | None,
    config: DetectorConfig,
    now: datetime | None = None,
) -> StalenessDetection:
    """Evaluate whether a single SSH session is stale.

    Checks three staleness conditions in priority order:

    1. **Transport disconnected**: If the probe returned DISCONNECTED,
       the session is immediately stale regardless of heartbeat age.
    2. **Heartbeat timeout**: If the time since the last successful
       heartbeat exceeds ``config.heartbeat_timeout_seconds``.
    3. **Max failures exceeded**: If the consecutive failure count
       (including this probe if it failed) reaches
       ``config.max_missed_heartbeats`` (and that setting is > 0).

    If none of the conditions are met, the session is reported as healthy.

    Args:
        record: Current heartbeat state for the session.
        probe_result: Result of the most recent transport-layer liveness
            probe. None if no probe was performed this cycle.
        config: Detector configuration with thresholds.
        now: Reference time for age calculations. Defaults to current UTC.

    Returns:
        Immutable StalenessDetection with the verdict and diagnostics.
    """
    if now is None:
        now = _now_utc()

    heartbeat_age = record.heartbeat_age_seconds(now)
    new_failures, transport_health = _compute_failures_after_probe(
        record.consecutive_failures, probe_result
    )

    # --- Check 1: Transport disconnected ---
    if transport_health == ConnectionHealth.DISCONNECTED:
        detail = (
            f"Transport layer disconnected "
            f"(probe error: {probe_result.error if probe_result else 'N/A'})"
        )
        logger.warning(
            "Session %s stale: transport disconnected (age=%.1fs, failures=%d)",
            record.session_id,
            heartbeat_age,
            new_failures,
        )
        return StalenessDetection(
            session_id=record.session_id,
            is_stale=True,
            reason=StalenessReason.TRANSPORT_DISCONNECTED,
            reason_detail=detail,
            detected_at=now,
            heartbeat_age_seconds=heartbeat_age,
            transport_health=transport_health,
            consecutive_failures=new_failures,
        )

    # --- Check 2: Heartbeat timeout ---
    if heartbeat_age > config.heartbeat_timeout_seconds:
        detail = (
            f"No heartbeat for {heartbeat_age:.1f} seconds "
            f"(threshold: {config.heartbeat_timeout_seconds:.1f}s)"
        )
        logger.warning(
            "Session %s stale: heartbeat timeout (age=%.1fs > %.1fs)",
            record.session_id,
            heartbeat_age,
            config.heartbeat_timeout_seconds,
        )
        return StalenessDetection(
            session_id=record.session_id,
            is_stale=True,
            reason=StalenessReason.HEARTBEAT_TIMEOUT,
            reason_detail=detail,
            detected_at=now,
            heartbeat_age_seconds=heartbeat_age,
            transport_health=transport_health,
            consecutive_failures=new_failures,
        )

    # --- Check 3: Max consecutive failures ---
    if (
        config.max_missed_heartbeats > 0
        and new_failures >= config.max_missed_heartbeats
    ):
        detail = (
            f"Consecutive probe failures: {new_failures} "
            f"(max allowed: {config.max_missed_heartbeats})"
        )
        logger.warning(
            "Session %s stale: max failures exceeded (%d >= %d)",
            record.session_id,
            new_failures,
            config.max_missed_heartbeats,
        )
        return StalenessDetection(
            session_id=record.session_id,
            is_stale=True,
            reason=StalenessReason.MAX_FAILURES_EXCEEDED,
            reason_detail=detail,
            detected_at=now,
            heartbeat_age_seconds=heartbeat_age,
            transport_health=transport_health,
            consecutive_failures=new_failures,
        )

    # --- All checks passed: session is healthy ---
    logger.debug(
        "Session %s healthy (age=%.1fs, failures=%d, health=%s)",
        record.session_id,
        heartbeat_age,
        new_failures,
        transport_health.value if transport_health else "N/A",
    )

    return StalenessDetection(
        session_id=record.session_id,
        is_stale=False,
        reason=None,
        reason_detail=None,
        detected_at=now,
        heartbeat_age_seconds=heartbeat_age,
        transport_health=transport_health,
        consecutive_failures=new_failures,
    )


# ---------------------------------------------------------------------------
# Public API: batch detection
# ---------------------------------------------------------------------------


def detect_batch_staleness(
    *,
    records: tuple[HeartbeatRecord, ...] | list[HeartbeatRecord],
    probe_results: Mapping[str, ProbeResult],
    config: DetectorConfig,
    now: datetime | None = None,
) -> tuple[StalenessDetection, ...]:
    """Evaluate staleness for a batch of sessions.

    Iterates over the provided heartbeat records, looks up each session's
    probe result from the mapping, and runs ``detect_session_staleness``
    for each. Sessions without a probe result in the mapping are checked
    with ``probe_result=None`` (heartbeat-only evaluation).

    Args:
        records: Sequence of HeartbeatRecord instances to evaluate.
        probe_results: Mapping from session_id to its most recent
            ProbeResult. Sessions not in this mapping are evaluated
            without transport data.
        config: Detector configuration with thresholds.
        now: Reference time for age calculations. Defaults to current UTC.

    Returns:
        Tuple of StalenessDetection results in the same order as the
        input records.
    """
    if now is None:
        now = _now_utc()

    return tuple(
        detect_session_staleness(
            record=record,
            probe_result=probe_results.get(record.session_id),
            config=config,
            now=now,
        )
        for record in records
    )
