"""Alert data model for the alert collector subsystem.

Defines the immutable data structures used to represent, store, and
query alerts produced by anomaly detectors. Alerts are the bridge
between raw ``AnomalyReport`` records (produced by detectors) and the
notification system (which delivers alerts to the CLI).

The alert lifecycle is:

    ACTIVE -> ACKNOWLEDGED -> RESOLVED

Each transition produces a new ``Alert`` instance (immutability).

Data flow::

    DetectorDispatcher
         |
         v  DispatchResult (contains AnomalyReport instances)
    AlertCollector
         |
         v  Alert (wraps AnomalyReport with alert metadata)
    NotificationBroadcaster
         |
         v  AlertNotification (pushed to CLI subscribers)

All data structures are frozen dataclasses, matching the project-wide
immutability convention. Thread-safe sharing is achieved by returning
tuples and frozensets (snapshot copies) rather than mutable containers.

Usage::

    from jules_daemon.monitor.alert_models import (
        Alert,
        AlertCollectorConfig,
        AlertSnapshot,
        AlertStatus,
        CollectResult,
    )

    # Create an alert from an anomaly report
    alert = Alert(
        alert_id="alert-001",
        anomaly_report=report,
        session_id="run-42",
        status=AlertStatus.ACTIVE,
        created_at=datetime.now(timezone.utc),
    )

    # Immutable status transition
    acknowledged = alert.with_status(AlertStatus.ACKNOWLEDGED)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from jules_daemon.monitor.anomaly_models import (
    AnomalyReport,
    AnomalySeverity,
    PatternType,
)

__all__ = [
    "Alert",
    "AlertCollectorConfig",
    "AlertSnapshot",
    "AlertStatus",
    "CollectResult",
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


def _require_non_negative(value: int, field_name: str) -> None:
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
# AlertStatus enum
# ---------------------------------------------------------------------------


class AlertStatus(Enum):
    """Lifecycle state of an alert.

    Alerts progress through a linear lifecycle:
    ACTIVE -> ACKNOWLEDGED -> RESOLVED

    Values:
        ACTIVE: Alert is new and has not been seen by the user.
        ACKNOWLEDGED: User has seen or acknowledged the alert.
        RESOLVED: Alert condition is no longer active or was dismissed.
    """

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


# ---------------------------------------------------------------------------
# Alert frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Alert:
    """Immutable record representing a single alert.

    Wraps an ``AnomalyReport`` from a detector with alert-specific
    metadata: unique ID, lifecycle status, and creation timestamp.

    Attributes:
        alert_id: Unique identifier for this alert (UUID-based).
        anomaly_report: The underlying anomaly detection result.
        session_id: SSH session this alert belongs to.
        status: Current lifecycle state of the alert.
        created_at: UTC timestamp when the alert was created.
    """

    alert_id: str
    anomaly_report: AnomalyReport
    session_id: str
    status: AlertStatus
    created_at: datetime

    def __post_init__(self) -> None:
        stripped_id = _require_non_empty(self.alert_id, "alert_id")
        if stripped_id != self.alert_id:
            object.__setattr__(self, "alert_id", stripped_id)

        stripped_sid = _require_non_empty(self.session_id, "session_id")
        if stripped_sid != self.session_id:
            object.__setattr__(self, "session_id", stripped_sid)

    # ---------------------------------------------------------------
    # Convenience properties (delegate to anomaly_report)
    # ---------------------------------------------------------------

    @property
    def severity(self) -> AnomalySeverity:
        """Severity of the underlying anomaly."""
        return self.anomaly_report.severity

    @property
    def pattern_name(self) -> str:
        """Pattern name of the underlying anomaly detector."""
        return self.anomaly_report.pattern_name

    @property
    def pattern_type(self) -> PatternType:
        """Pattern type of the underlying anomaly detector."""
        return self.anomaly_report.pattern_type

    @property
    def is_active(self) -> bool:
        """True if the alert is in the ACTIVE state."""
        return self.status is AlertStatus.ACTIVE

    # ---------------------------------------------------------------
    # Immutable state transitions
    # ---------------------------------------------------------------

    def with_status(self, new_status: AlertStatus) -> Alert:
        """Return a new Alert with the given status.

        All other fields are preserved. This is the only way to
        transition alert status, enforcing immutability.

        Args:
            new_status: The new lifecycle state for the alert.

        Returns:
            A new Alert instance with the updated status.
        """
        return Alert(
            alert_id=self.alert_id,
            anomaly_report=self.anomaly_report,
            session_id=self.session_id,
            status=new_status,
            created_at=self.created_at,
        )


# ---------------------------------------------------------------------------
# AlertSnapshot frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertSnapshot:
    """Immutable summary of alerts for a single session.

    Provides aggregated counts and the highest severity level across
    all alerts in the snapshot. Useful for dashboard displays and
    status queries without iterating through individual alerts.

    Attributes:
        session_id: The SSH session these alerts belong to.
        total_alerts: Total number of alerts in the snapshot.
        active_count: Number of alerts in ACTIVE state.
        acknowledged_count: Number of alerts in ACKNOWLEDGED state.
        resolved_count: Number of alerts in RESOLVED state.
        highest_severity: The most severe alert level, or None if empty.
        alerts: Tuple of all Alert records in the snapshot.
    """

    session_id: str
    total_alerts: int
    active_count: int
    acknowledged_count: int
    resolved_count: int
    highest_severity: AnomalySeverity | None
    alerts: tuple[Alert, ...]

    def __post_init__(self) -> None:
        stripped_sid = _require_non_empty(self.session_id, "session_id")
        if stripped_sid != self.session_id:
            object.__setattr__(self, "session_id", stripped_sid)

        _require_non_negative(self.total_alerts, "total_alerts")
        _require_non_negative(self.active_count, "active_count")
        _require_non_negative(self.acknowledged_count, "acknowledged_count")
        _require_non_negative(self.resolved_count, "resolved_count")

        count_sum = (
            self.active_count
            + self.acknowledged_count
            + self.resolved_count
        )
        if count_sum != self.total_alerts:
            raise ValueError(
                f"Status counts ({count_sum}) must equal "
                f"total_alerts ({self.total_alerts})"
            )


# ---------------------------------------------------------------------------
# AlertCollectorConfig frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertCollectorConfig:
    """Immutable configuration for the AlertCollector.

    Controls capacity limits to prevent unbounded memory growth.
    When limits are reached, oldest alerts are evicted first.

    Attributes:
        max_alerts_per_session: Maximum alerts stored per SSH session.
            Oldest alerts are evicted when this limit is reached.
        max_total_alerts: Maximum alerts across all sessions.
            Oldest alerts (globally) are evicted when this limit
            is reached.
    """

    max_alerts_per_session: int = 100
    max_total_alerts: int = 1000

    def __post_init__(self) -> None:
        _require_positive_int(
            self.max_alerts_per_session, "max_alerts_per_session"
        )
        _require_positive_int(self.max_total_alerts, "max_total_alerts")


# ---------------------------------------------------------------------------
# CollectResult frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectResult:
    """Immutable result of collecting alerts from a DispatchResult.

    Produced by ``AlertCollector.collect()`` to report what happened
    during alert ingestion: which alerts were created and which were
    evicted due to capacity constraints.

    Attributes:
        new_alerts: Tuple of newly created Alert records.
        evicted_alerts: Tuple of Alert records evicted due to capacity.
        session_id: The SSH session the dispatch was for.
        collected_at: UTC timestamp of when collection occurred.
    """

    new_alerts: tuple[Alert, ...]
    evicted_alerts: tuple[Alert, ...]
    session_id: str
    collected_at: datetime

    def __post_init__(self) -> None:
        stripped_sid = _require_non_empty(self.session_id, "session_id")
        if stripped_sid != self.session_id:
            object.__setattr__(self, "session_id", stripped_sid)

    @property
    def alert_count(self) -> int:
        """Number of new alerts created."""
        return len(self.new_alerts)

    @property
    def eviction_count(self) -> int:
        """Number of alerts evicted due to capacity."""
        return len(self.evicted_alerts)

    @property
    def has_new_alerts(self) -> bool:
        """True if at least one new alert was created."""
        return len(self.new_alerts) > 0

    @property
    def has_evictions(self) -> bool:
        """True if at least one alert was evicted."""
        return len(self.evicted_alerts) > 0
