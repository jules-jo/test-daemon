"""Alert deduplication and priority-scoring logic.

Processes incoming anomaly reports before they reach the alert collector,
providing two key capabilities:

1. **Deduplication**: Suppresses duplicate alerts that share the same
   pattern_name, pattern_type, and session_id within a configurable
   time window. Prevents alert fatigue from repeated firing of the
   same detector for the same underlying issue.

2. **Priority scoring**: Assigns a numeric priority score to each alert
   based on severity level, pattern type, and occurrence frequency.
   Higher scores indicate more important alerts, enabling downstream
   consumers to sort and filter by priority.

The processor sits between the ``DetectorDispatcher`` output and the
``AlertCollector`` input in the data flow::

    DetectorDispatcher
         |
         v  DispatchResult (contains AnomalyReport instances)
    AlertProcessor        <-- THIS MODULE
         |
         v  ProcessingResult (kept vs. suppressed, with priority scores)
    AlertCollector
         |
         v  Alert (stored and queryable)

All data structures are frozen dataclasses, matching the project-wide
immutability convention. The processor itself is thread-safe: internal
state is guarded by a ``threading.Lock``.

Usage::

    from jules_daemon.monitor.alert_dedup import (
        AlertProcessor,
        AlertProcessorConfig,
    )

    config = AlertProcessorConfig(
        dedup_window_seconds=60.0,
        max_tracked_keys=500,
    )
    processor = AlertProcessor(config=config)

    result = processor.process(dispatch_result)
    for alert in result.kept:
        print(f"New alert: {alert.anomaly_report.pattern_name} "
              f"(priority={alert.priority.total:.1f})")
    for alert in result.suppressed:
        print(f"Suppressed duplicate: {alert.anomaly_report.pattern_name} "
              f"(occurrence #{alert.occurrence_count})")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final

from jules_daemon.monitor.anomaly_models import (
    AnomalyReport,
    AnomalySeverity,
    PatternType,
)
from jules_daemon.monitor.detector_dispatcher import DispatchResult

__all__ = [
    "AlertProcessor",
    "AlertProcessorConfig",
    "DeduplicationKey",
    "PriorityScore",
    "ProcessedAlert",
    "ProcessingResult",
    "compute_dedup_key",
    "compute_priority_score",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_non_empty(value: str, field_name: str) -> str:
    """Validate that a string field is non-empty after stripping whitespace.

    Args:
        value: The string value to validate.
        field_name: Name of the field for the error message.

    Returns:
        The stripped string value.

    Raises:
        ValueError: If the value is empty or whitespace-only.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


def _require_positive(value: float, field_name: str) -> None:
    """Validate that a numeric field is strictly positive.

    Args:
        value: The numeric value to validate.
        field_name: Name of the field for the error message.

    Raises:
        ValueError: If the value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")


def _require_non_negative_float(value: float, field_name: str) -> None:
    """Validate that a numeric field is not negative.

    Args:
        value: The numeric value to validate.
        field_name: Name of the field for the error message.

    Raises:
        ValueError: If the value is negative.
    """
    if value < 0:
        raise ValueError(f"{field_name} must not be negative, got {value}")


def _require_non_negative_int(value: int, field_name: str) -> None:
    """Validate that an integer field is not negative.

    Args:
        value: The integer value to validate.
        field_name: Name of the field for the error message.

    Raises:
        ValueError: If the value is negative.
    """
    if value < 0:
        raise ValueError(f"{field_name} must not be negative, got {value}")


def _require_positive_int(value: int, field_name: str) -> None:
    """Validate that an integer field is strictly positive.

    Args:
        value: The integer value to validate.
        field_name: Name of the field for the error message.

    Raises:
        ValueError: If the value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")


# ---------------------------------------------------------------------------
# Default weight tables
# ---------------------------------------------------------------------------

_DEFAULT_SEVERITY_WEIGHTS: dict[AnomalySeverity, float] = {
    AnomalySeverity.CRITICAL: 100.0,
    AnomalySeverity.WARNING: 50.0,
    AnomalySeverity.INFO: 10.0,
}

_DEFAULT_PATTERN_TYPE_WEIGHTS: dict[PatternType, float] = {
    PatternType.ERROR_KEYWORD: 10.0,
    PatternType.FAILURE_RATE: 15.0,
    PatternType.STALL_TIMEOUT: 20.0,
}


# ---------------------------------------------------------------------------
# DeduplicationKey frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeduplicationKey:
    """Immutable key identifying a unique alert source.

    Two anomaly reports with the same dedup key are considered duplicates
    if they occur within the configured time window. The key deliberately
    excludes message text, severity, and timestamp so that the same
    detector pattern firing repeatedly for the same session is correctly
    identified as a duplicate.

    Attributes:
        pattern_name: Name of the anomaly pattern.
        pattern_type: Type classification of the pattern.
        session_id: SSH session the alert belongs to.
    """

    pattern_name: str
    pattern_type: PatternType
    session_id: str

    def __post_init__(self) -> None:
        stripped_name = _require_non_empty(self.pattern_name, "pattern_name")
        if stripped_name != self.pattern_name:
            object.__setattr__(self, "pattern_name", stripped_name)

        stripped_sid = _require_non_empty(self.session_id, "session_id")
        if stripped_sid != self.session_id:
            object.__setattr__(self, "session_id", stripped_sid)


# ---------------------------------------------------------------------------
# PriorityScore frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorityScore:
    """Immutable breakdown of an alert's computed priority score.

    The total score is the sum of all component scores. Higher totals
    indicate more important alerts that should be shown first.

    Attributes:
        total: Overall priority score (sum of components).
        severity_component: Score contribution from anomaly severity.
        frequency_component: Score contribution from occurrence frequency.
        pattern_type_component: Score contribution from pattern type.
    """

    total: float
    severity_component: float
    frequency_component: float
    pattern_type_component: float

    def __post_init__(self) -> None:
        _require_non_negative_float(self.total, "total")


# ---------------------------------------------------------------------------
# ProcessedAlert frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessedAlert:
    """Immutable record of an alert after dedup and priority processing.

    Wraps the original ``AnomalyReport`` with dedup and priority metadata.
    The ``is_duplicate`` flag indicates whether this alert was suppressed
    by the dedup filter.

    Attributes:
        anomaly_report: The original anomaly report from the detector.
        priority: Computed priority score breakdown.
        dedup_key: The deduplication key for this report.
        is_duplicate: True if this report was suppressed as a duplicate.
        occurrence_count: Total times this dedup key has been seen
            within the current time window (including this occurrence).
    """

    anomaly_report: AnomalyReport
    priority: PriorityScore
    dedup_key: DeduplicationKey
    is_duplicate: bool
    occurrence_count: int

    def __post_init__(self) -> None:
        _require_non_negative_int(self.occurrence_count, "occurrence_count")


# ---------------------------------------------------------------------------
# AlertProcessorConfig frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertProcessorConfig:
    """Immutable configuration for the AlertProcessor.

    Controls dedup window size, priority scoring weights, and capacity.

    Attributes:
        dedup_window_seconds: Time window (seconds) within which
            identical dedup keys are considered duplicates. Must be
            positive.
        frequency_boost_factor: Score boost per additional occurrence
            beyond the first. Must be non-negative.
        max_tracked_keys: Maximum number of dedup keys tracked
            simultaneously. Oldest entries evicted when exceeded.
            Must be positive.
        severity_weights: Mapping from severity level to base priority
            score. Higher values for more severe levels.
        pattern_type_weights: Mapping from pattern type to score
            component. Allows emphasizing certain pattern types.
    """

    dedup_window_seconds: float = 60.0
    frequency_boost_factor: float = 5.0
    max_tracked_keys: int = 500
    severity_weights: dict[AnomalySeverity, float] = field(
        default_factory=lambda: dict(_DEFAULT_SEVERITY_WEIGHTS),
    )
    pattern_type_weights: dict[PatternType, float] = field(
        default_factory=lambda: dict(_DEFAULT_PATTERN_TYPE_WEIGHTS),
    )

    def __post_init__(self) -> None:
        _require_positive(self.dedup_window_seconds, "dedup_window_seconds")
        _require_non_negative_float(
            self.frequency_boost_factor, "frequency_boost_factor"
        )
        _require_positive_int(self.max_tracked_keys, "max_tracked_keys")

        # Defensively copy mutable dict fields so callers cannot corrupt
        # the frozen config after construction.
        object.__setattr__(
            self, "severity_weights", dict(self.severity_weights)
        )
        object.__setattr__(
            self, "pattern_type_weights", dict(self.pattern_type_weights)
        )


# ---------------------------------------------------------------------------
# ProcessingResult frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessingResult:
    """Immutable result of processing a batch of anomaly reports.

    Partitions reports into ``kept`` (passed dedup) and ``suppressed``
    (identified as duplicates). Both partitions include computed
    priority scores.

    Attributes:
        kept: Alerts that passed the dedup filter and should be stored.
        suppressed: Alerts identified as duplicates and suppressed.
        session_id: SSH session this batch belongs to.
        processed_at: UTC timestamp of when processing occurred.
    """

    kept: tuple[ProcessedAlert, ...]
    suppressed: tuple[ProcessedAlert, ...]
    session_id: str
    processed_at: datetime

    def __post_init__(self) -> None:
        stripped_sid = _require_non_empty(self.session_id, "session_id")
        if stripped_sid != self.session_id:
            object.__setattr__(self, "session_id", stripped_sid)

    @property
    def kept_count(self) -> int:
        """Number of alerts that passed the dedup filter."""
        return len(self.kept)

    @property
    def suppressed_count(self) -> int:
        """Number of alerts suppressed as duplicates."""
        return len(self.suppressed)

    @property
    def has_kept(self) -> bool:
        """True if at least one alert passed dedup."""
        return len(self.kept) > 0

    @property
    def has_suppressed(self) -> bool:
        """True if at least one alert was suppressed."""
        return len(self.suppressed) > 0

    @property
    def kept_by_priority(self) -> tuple[ProcessedAlert, ...]:
        """Return kept alerts sorted by descending priority score.

        Returns:
            Tuple of ProcessedAlert ordered from highest to lowest
            priority score.
        """
        return tuple(
            sorted(self.kept, key=lambda a: a.priority.total, reverse=True)
        )


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def compute_dedup_key(report: AnomalyReport) -> DeduplicationKey:
    """Compute the deduplication key from an anomaly report.

    The key is based on pattern_name, pattern_type, and session_id.
    Message text, severity, timestamp, and context are deliberately
    excluded so that repeated firings of the same detector are
    correctly identified as duplicates.

    Args:
        report: The anomaly report to compute a key for.

    Returns:
        Immutable DeduplicationKey for dedup comparison.
    """
    return DeduplicationKey(
        pattern_name=report.pattern_name,
        pattern_type=report.pattern_type,
        session_id=report.session_id,
    )


def compute_priority_score(
    report: AnomalyReport,
    *,
    occurrence_count: int,
    config: AlertProcessorConfig,
) -> PriorityScore:
    """Compute the priority score for an anomaly report.

    The total score is the sum of three components:
    - **Severity**: Base score from the severity_weights table.
    - **Frequency**: Boost per additional occurrence beyond the first.
    - **Pattern type**: Score from the pattern_type_weights table.

    Args:
        report: The anomaly report to score.
        occurrence_count: How many times this dedup key has been seen
            in the current window (0 = first time).
        config: Processor configuration with scoring weights.

    Returns:
        Immutable PriorityScore with component breakdown.
    """
    severity_component = config.severity_weights.get(report.severity, 0.0)

    # Frequency: only boost for occurrences beyond the first
    frequency_component = max(0, occurrence_count - 1) * config.frequency_boost_factor

    pattern_type_component = config.pattern_type_weights.get(
        report.pattern_type, 0.0
    )

    total = severity_component + frequency_component + pattern_type_component

    return PriorityScore(
        total=total,
        severity_component=severity_component,
        frequency_component=frequency_component,
        pattern_type_component=pattern_type_component,
    )


# ---------------------------------------------------------------------------
# Internal tracking entry
# ---------------------------------------------------------------------------


class _DedupEntry:
    """Mutable internal tracking entry for a dedup key.

    NOT exposed publicly. Lives only inside the AlertProcessor behind
    the lock. Tracks the most recent occurrence timestamp and the
    total occurrence count within the dedup window.

    Attributes:
        key: The deduplication key being tracked.
        last_seen: UTC timestamp of the most recent occurrence.
        occurrence_count: Total occurrences within the dedup window.
    """

    __slots__ = ("key", "last_seen", "occurrence_count")

    def __init__(
        self,
        *,
        key: DeduplicationKey,
        last_seen: datetime,
    ) -> None:
        self.key: Final[DeduplicationKey] = key
        self.last_seen: datetime = last_seen
        self.occurrence_count: int = 1


# ---------------------------------------------------------------------------
# AlertProcessor
# ---------------------------------------------------------------------------


class AlertProcessor:
    """Deduplication and priority-scoring processor for incoming alerts.

    Processes ``DispatchResult`` instances, partitioning anomaly reports
    into kept (new) and suppressed (duplicate) sets. Each report receives
    a computed priority score based on severity, pattern type, and
    occurrence frequency.

    Thread safety:
        All internal state is guarded by a ``threading.Lock``. Safe for
        concurrent access from multiple threads.

    Args:
        config: Optional configuration. Uses defaults if not provided.
    """

    __slots__ = ("_config", "_lock", "_entries")

    def __init__(
        self,
        *,
        config: AlertProcessorConfig | None = None,
    ) -> None:
        """Initialize the alert processor.

        Args:
            config: Optional dedup/priority configuration. Defaults to
                ``AlertProcessorConfig()`` with default settings.
        """
        self._config: Final[AlertProcessorConfig] = (
            config or AlertProcessorConfig()
        )
        self._lock: Final[threading.Lock] = threading.Lock()
        # DeduplicationKey -> _DedupEntry
        self._entries: dict[DeduplicationKey, _DedupEntry] = {}

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self,
        dispatch_result: DispatchResult,
        *,
        now: datetime | None = None,
    ) -> ProcessingResult:
        """Process a DispatchResult through dedup and priority scoring.

        For each anomaly report in the dispatch result:
        1. Compute the deduplication key.
        2. Check if the key has been seen within the dedup window.
        3. If seen (duplicate): mark as suppressed, increment count.
        4. If new: mark as kept, create tracking entry.
        5. Compute priority score for all reports (kept and suppressed).

        Expired entries (outside the dedup window) are cleaned up during
        processing. Capacity limits are enforced after processing.

        Args:
            dispatch_result: Result from the DetectorDispatcher.
            now: Optional override for the current timestamp. Used for
                testing determinism. Defaults to UTC now.

        Returns:
            Immutable ProcessingResult with kept and suppressed partitions.
        """
        current_time = now or datetime.now(timezone.utc)
        session_id = dispatch_result.session_id
        reports = dispatch_result.reports

        if not reports:
            with self._lock:
                self._expire_entries(current_time)
            return ProcessingResult(
                kept=(),
                suppressed=(),
                session_id=session_id,
                processed_at=current_time,
            )

        kept: list[ProcessedAlert] = []
        suppressed: list[ProcessedAlert] = []

        with self._lock:
            # Clean expired entries first
            self._expire_entries(current_time)

            for report in reports:
                key = compute_dedup_key(report)
                entry = self._entries.get(key)

                if entry is not None:
                    # Check if within window
                    elapsed = (
                        current_time - entry.last_seen
                    ).total_seconds()

                    if elapsed < self._config.dedup_window_seconds:
                        # Duplicate: increment and suppress
                        entry.occurrence_count += 1
                        entry.last_seen = current_time
                        occurrence_count = entry.occurrence_count

                        priority = compute_priority_score(
                            report,
                            occurrence_count=occurrence_count,
                            config=self._config,
                        )
                        suppressed.append(
                            ProcessedAlert(
                                anomaly_report=report,
                                priority=priority,
                                dedup_key=key,
                                is_duplicate=True,
                                occurrence_count=occurrence_count,
                            )
                        )
                        continue
                    else:
                        # Entry expired - reset it
                        entry.last_seen = current_time
                        entry.occurrence_count = 1
                        occurrence_count = 1
                else:
                    # New key - create entry
                    entry = _DedupEntry(
                        key=key,
                        last_seen=current_time,
                    )
                    self._entries[key] = entry
                    occurrence_count = 1

                # Kept: compute priority and add
                priority = compute_priority_score(
                    report,
                    occurrence_count=occurrence_count,
                    config=self._config,
                )
                kept.append(
                    ProcessedAlert(
                        anomaly_report=report,
                        priority=priority,
                        dedup_key=key,
                        is_duplicate=False,
                        occurrence_count=occurrence_count,
                    )
                )

            # Enforce capacity limit
            self._enforce_capacity()

        if suppressed:
            logger.info(
                "Suppressed %d duplicate alert(s) for session %s",
                len(suppressed),
                session_id,
            )

        return ProcessingResult(
            kept=tuple(kept),
            suppressed=tuple(suppressed),
            session_id=session_id,
            processed_at=current_time,
        )

    # ------------------------------------------------------------------
    # Clearing
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> int:
        """Remove all tracked dedup entries for a session.

        Args:
            session_id: The SSH session to clear.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            keys_to_remove = [
                key
                for key in self._entries
                if key.session_id == session_id
            ]
            for key in keys_to_remove:
                del self._entries[key]

            count = len(keys_to_remove)

        if count:
            logger.debug(
                "Cleared %d dedup entries for session %s",
                count,
                session_id,
            )
        return count

    def clear_all(self) -> int:
        """Remove all tracked dedup entries.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()

        if count:
            logger.debug("Cleared all %d dedup entries", count)
        return count

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tracked_key_count(self) -> int:
        """Number of dedup keys currently being tracked."""
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expire_entries(self, now: datetime) -> None:
        """Remove entries whose last_seen is outside the dedup window.

        Must be called under the lock.

        Args:
            now: Current UTC timestamp.
        """
        window = self._config.dedup_window_seconds
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if (now - entry.last_seen).total_seconds() > window
        ]
        for key in expired_keys:
            del self._entries[key]

        if expired_keys:
            logger.debug(
                "Expired %d dedup entries outside window", len(expired_keys)
            )

    def _enforce_capacity(self) -> None:
        """Evict oldest entries when capacity limit is exceeded.

        Must be called under the lock. Evicts entries with the oldest
        ``last_seen`` timestamp first.
        """
        limit = self._config.max_tracked_keys
        overflow = len(self._entries) - limit
        if overflow <= 0:
            return

        # Sort entries by last_seen ascending (oldest first)
        sorted_keys = sorted(
            self._entries.keys(),
            key=lambda k: self._entries[k].last_seen,
        )

        for i in range(overflow):
            evicted_key = sorted_keys[i]
            del self._entries[evicted_key]

        logger.debug(
            "Evicted %d dedup entries due to capacity limit", overflow
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._entries)
        return f"AlertProcessor(tracking {count} keys)"
