"""Query API for filtered and prioritized alert retrieval.

Provides a rich query interface that exposes alerts from the
``AlertCollector`` with multi-criteria filtering, priority-based
sorting, and result limiting. Designed specifically for the agent
loop to consume alerts efficiently during think-act cycles.

The query API sits on top of the existing alert infrastructure::

    AlertCollector (storage/indexing)
         |
         v  raw Alert records
    AlertQueryService (this module)
         |
         v  filtered + prioritized QueryableAlert records
    Agent Loop / Tools (consumers)

Key features:

- **Multi-criteria filtering**: Filter by severity threshold, alert
  status, pattern type, session ID, and time range. Filters compose
  with AND semantics.

- **Priority-based sorting**: Each alert receives a computed priority
  score based on severity, pattern type, and the weights from
  ``AlertProcessorConfig``. Results are sorted by descending priority
  by default, with alternative sort orders available.

- **Result limiting**: Configurable ``max_results`` cap prevents
  unbounded response sizes. When truncated, the highest-priority
  alerts are returned and the result metadata indicates truncation.

- **Agent-friendly output**: ``QueryableAlert.to_agent_summary()``
  and ``AlertQueryResult.to_agent_context()`` produce compact dicts
  suitable for embedding in LLM conversation context.

- **Immutable results**: All return types are frozen dataclasses,
  matching the project-wide immutability convention.

- **Thread-safe**: The service delegates all state access to the
  ``AlertCollector`` (which uses ``threading.Lock``). The service
  itself holds no mutable state.

Usage::

    from jules_daemon.monitor.alert_collector import AlertCollector
    from jules_daemon.monitor.alert_query import (
        AlertQuery,
        AlertQueryService,
        AlertSortOrder,
    )
    from jules_daemon.monitor.anomaly_models import AnomalySeverity

    collector = AlertCollector()
    service = AlertQueryService(collector=collector)

    # Get all active critical alerts for a session
    result = service.query(
        AlertQuery(
            session_id="run-42",
            min_severity=AnomalySeverity.CRITICAL,
        )
    )
    for alert in result.alerts:
        print(f"{alert.alert_id}: {alert.priority_score:.1f}")

    # Quick convenience: active critical alerts
    critical = service.active_critical_alerts(session_id="run-42")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Final

from jules_daemon.monitor.alert_collector import AlertCollector
from jules_daemon.monitor.alert_models import (
    Alert,
    AlertStatus,
)
from jules_daemon.monitor.anomaly_models import (
    AnomalySeverity,
    PatternType,
)

__all__ = [
    "AlertQuery",
    "AlertQueryResult",
    "AlertQueryService",
    "AlertSortOrder",
    "QueryableAlert",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default priority weight tables
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
# Validation helpers
# ---------------------------------------------------------------------------


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
# AlertSortOrder enum
# ---------------------------------------------------------------------------


class AlertSortOrder(Enum):
    """Sort order for alert query results.

    Values:
        PRIORITY_DESC: Sort by computed priority score, highest first.
            This is the default and recommended order for agent consumption.
        TIME_DESC: Sort by creation time, most recent first.
        SEVERITY_DESC: Sort by severity level, most severe first.
    """

    PRIORITY_DESC = "priority_desc"
    TIME_DESC = "time_desc"
    SEVERITY_DESC = "severity_desc"


# ---------------------------------------------------------------------------
# AlertQuery frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertQuery:
    """Immutable specification of alert query filters.

    All filter fields are optional. When multiple filters are specified,
    they compose with AND semantics (an alert must pass all filters to
    be included in the result).

    Attributes:
        session_id: Filter to a specific SSH session. None means all
            sessions.
        min_severity: Minimum severity threshold (inclusive). Alerts
            with severity below this level are excluded. None means
            no severity filter.
        statuses: Set of acceptable alert statuses. None means all
            statuses are included.
        pattern_types: Set of acceptable pattern types. None means
            all types are included.
        since: Only include alerts created at or after this timestamp.
            None means no lower time bound.
        until: Only include alerts created at or before this timestamp.
            None means no upper time bound.
        max_results: Maximum number of alerts to return. Defaults to
            50. Must be positive.
        sort_order: How to sort the results. Defaults to PRIORITY_DESC.
    """

    session_id: str | None = None
    min_severity: AnomalySeverity | None = None
    statuses: frozenset[AlertStatus] | None = None
    pattern_types: frozenset[PatternType] | None = None
    since: datetime | None = None
    until: datetime | None = None
    max_results: int = 50
    sort_order: AlertSortOrder = AlertSortOrder.PRIORITY_DESC

    def __post_init__(self) -> None:
        _require_positive_int(self.max_results, "max_results")


# ---------------------------------------------------------------------------
# QueryableAlert frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryableAlert:
    """Immutable alert wrapper with computed priority score.

    Enriches a raw ``Alert`` with a priority score for sorting and
    ranking. Provides convenience properties that delegate to the
    underlying alert and a compact ``to_agent_summary()`` method
    for LLM context.

    Attributes:
        alert: The underlying Alert record.
        priority_score: Computed priority score (higher = more important).
    """

    alert: Alert
    priority_score: float

    # -- Convenience delegate properties ------------------------------------

    @property
    def alert_id(self) -> str:
        """Unique alert identifier."""
        return self.alert.alert_id

    @property
    def severity(self) -> AnomalySeverity:
        """Severity of the underlying anomaly."""
        return self.alert.severity

    @property
    def pattern_type(self) -> PatternType:
        """Pattern type of the underlying anomaly detector."""
        return self.alert.pattern_type

    @property
    def pattern_name(self) -> str:
        """Pattern name of the underlying anomaly detector."""
        return self.alert.pattern_name

    @property
    def status(self) -> AlertStatus:
        """Current lifecycle state of the alert."""
        return self.alert.status

    @property
    def session_id(self) -> str:
        """SSH session this alert belongs to."""
        return self.alert.session_id

    @property
    def created_at(self) -> datetime:
        """UTC timestamp when the alert was created."""
        return self.alert.created_at

    @property
    def message(self) -> str:
        """Human-readable description from the anomaly report."""
        return self.alert.anomaly_report.message

    # -- Agent-friendly serialization ---------------------------------------

    def to_agent_summary(self) -> dict[str, Any]:
        """Serialize to a compact dict for agent loop context.

        Returns a flat dictionary suitable for embedding in LLM
        conversation messages. Includes all fields needed for the
        agent to reason about the alert and decide on actions.

        Returns:
            Dict with string keys and JSON-serializable values.
        """
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "status": self.status.value,
            "pattern_name": self.pattern_name,
            "pattern_type": self.pattern_type.value,
            "session_id": self.session_id,
            "priority_score": self.priority_score,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# AlertQueryResult frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertQueryResult:
    """Immutable result of an alert query.

    Contains the filtered and sorted alerts along with metadata
    about the query (total matches, truncation, etc.).

    Attributes:
        alerts: Tuple of QueryableAlert records matching the query,
            sorted according to the query's sort_order and truncated
            to max_results.
        total_matched: Total number of alerts that matched all filters
            before truncation. May exceed len(alerts) when truncated.
        total_available: Total number of alerts in the collector
            (before any filtering). Provides context for the agent.
        query: The AlertQuery that produced this result.
    """

    alerts: tuple[QueryableAlert, ...]
    total_matched: int
    total_available: int
    query: AlertQuery

    @property
    def result_count(self) -> int:
        """Number of alerts in this result (after truncation)."""
        return len(self.alerts)

    @property
    def is_empty(self) -> bool:
        """True if no alerts matched the query."""
        return len(self.alerts) == 0

    @property
    def was_truncated(self) -> bool:
        """True if the result was truncated by max_results."""
        return self.total_matched > self.query.max_results

    def to_agent_context(self) -> dict[str, Any]:
        """Serialize to a dict for embedding in agent loop context.

        Produces a structured summary that the LLM can parse to
        understand the current alert state without needing to see
        the raw Alert objects.

        Returns:
            Dict with result metadata and alert summaries.
        """
        return {
            "result_count": self.result_count,
            "total_matched": self.total_matched,
            "total_available": self.total_available,
            "was_truncated": self.was_truncated,
            "alerts": [a.to_agent_summary() for a in self.alerts],
        }


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------


def _compute_alert_priority(
    alert: Alert,
    *,
    severity_weights: dict[AnomalySeverity, float],
    pattern_type_weights: dict[PatternType, float],
) -> float:
    """Compute a priority score for an alert.

    The score is the sum of a severity component and a pattern-type
    component, using the provided weight tables. Higher scores
    indicate more important alerts.

    Args:
        alert: The alert to score.
        severity_weights: Mapping from severity level to score.
        pattern_type_weights: Mapping from pattern type to score.

    Returns:
        Non-negative float priority score.
    """
    severity_score = severity_weights.get(alert.severity, 0.0)
    pattern_score = pattern_type_weights.get(alert.pattern_type, 0.0)
    return severity_score + pattern_score


# ---------------------------------------------------------------------------
# Filter predicate
# ---------------------------------------------------------------------------


def _alert_matches_query(alert: Alert, query: AlertQuery) -> bool:
    """Test whether an alert passes all query filters.

    Applies each non-None filter criterion with AND semantics.
    Returns False as soon as any filter rejects the alert.

    Args:
        alert: The alert to test.
        query: The query specification with filter criteria.

    Returns:
        True if the alert passes all filters.
    """
    if query.session_id is not None and alert.session_id != query.session_id:
        return False

    if query.min_severity is not None:
        if alert.severity.numeric_level < query.min_severity.numeric_level:
            return False

    if query.statuses is not None and alert.status not in query.statuses:
        return False

    if (
        query.pattern_types is not None
        and alert.pattern_type not in query.pattern_types
    ):
        return False

    if query.since is not None and alert.created_at < query.since:
        return False

    if query.until is not None and alert.created_at > query.until:
        return False

    return True


# ---------------------------------------------------------------------------
# Sort key builders
# ---------------------------------------------------------------------------


def _sort_key_priority(qa: QueryableAlert) -> tuple[float, int, datetime]:
    """Sort key: descending priority, then severity, then time."""
    return (
        -qa.priority_score,
        -qa.severity.numeric_level,
        qa.created_at,
    )


def _sort_key_severity(qa: QueryableAlert) -> tuple[int, float, datetime]:
    """Sort key: descending severity, then priority, then time."""
    return (
        -qa.severity.numeric_level,
        -qa.priority_score,
        qa.created_at,
    )


def _sort_key_time(qa: QueryableAlert) -> tuple[datetime, float, int]:
    """Sort key: descending time (negate impossible, use reverse=True)."""
    return (
        qa.created_at,
        -qa.priority_score,
        -qa.severity.numeric_level,
    )


# ---------------------------------------------------------------------------
# AlertQueryService
# ---------------------------------------------------------------------------


class AlertQueryService:
    """Service that provides filtered, prioritized alert queries.

    Bridges the ``AlertCollector`` (raw alert storage) with the agent
    loop (which needs filtered, scored, and sorted alert data). The
    service holds no mutable state of its own; all alert state lives
    in the collector.

    Thread safety:
        The service delegates all state access to the collector, which
        uses ``threading.Lock`` internally. The service's own methods
        are stateless and safe for concurrent access.

    Args:
        collector: The AlertCollector to query alerts from.
        severity_weights: Optional custom severity-to-score mapping.
            Defaults to the standard weights (CRITICAL=100, WARNING=50,
            INFO=10).
        pattern_type_weights: Optional custom pattern-type-to-score
            mapping. Defaults to standard weights (ERROR_KEYWORD=10,
            FAILURE_RATE=15, STALL_TIMEOUT=20).
    """

    __slots__ = ("_collector", "_severity_weights", "_pattern_type_weights")

    def __init__(
        self,
        *,
        collector: AlertCollector,
        severity_weights: dict[AnomalySeverity, float] | None = None,
        pattern_type_weights: dict[PatternType, float] | None = None,
    ) -> None:
        self._collector: Final[AlertCollector] = collector
        self._severity_weights: Final[dict[AnomalySeverity, float]] = (
            dict(severity_weights)
            if severity_weights is not None
            else dict(_DEFAULT_SEVERITY_WEIGHTS)
        )
        self._pattern_type_weights: Final[dict[PatternType, float]] = (
            dict(pattern_type_weights)
            if pattern_type_weights is not None
            else dict(_DEFAULT_PATTERN_TYPE_WEIGHTS)
        )

    # ------------------------------------------------------------------
    # Core query method
    # ------------------------------------------------------------------

    def query(self, query: AlertQuery) -> AlertQueryResult:
        """Execute an alert query against the collector.

        Retrieves alerts, applies all filters, computes priority
        scores, sorts, and truncates according to the query spec.

        Args:
            query: The query specification with filters, sort order,
                and result limit.

        Returns:
            Immutable AlertQueryResult with filtered, scored, and
            sorted alerts plus metadata.
        """
        # Step 1: Gather all raw alerts from the collector
        all_alerts = self._gather_alerts(session_id=query.session_id)
        total_available = self._collector.alert_count

        # Step 2: Filter
        matched = tuple(
            alert for alert in all_alerts
            if _alert_matches_query(alert, query)
        )

        # Step 3: Score and wrap
        queryable = tuple(
            QueryableAlert(
                alert=alert,
                priority_score=_compute_alert_priority(
                    alert,
                    severity_weights=self._severity_weights,
                    pattern_type_weights=self._pattern_type_weights,
                ),
            )
            for alert in matched
        )

        # Step 4: Sort
        sorted_alerts = self._sort_alerts(queryable, query.sort_order)

        # Step 5: Truncate
        total_matched = len(sorted_alerts)
        truncated = sorted_alerts[: query.max_results]

        return AlertQueryResult(
            alerts=tuple(truncated),
            total_matched=total_matched,
            total_available=total_available,
            query=query,
        )

    # ------------------------------------------------------------------
    # Convenience query methods
    # ------------------------------------------------------------------

    def active_critical_alerts(
        self,
        *,
        session_id: str | None = None,
        max_results: int = 20,
    ) -> AlertQueryResult:
        """Convenience: query only active, critical-severity alerts.

        This is the most common agent-loop query -- "what critical
        issues need my attention right now?"

        Args:
            session_id: Optional session filter.
            max_results: Maximum alerts to return.

        Returns:
            AlertQueryResult with active CRITICAL alerts.
        """
        return self.query(
            AlertQuery(
                session_id=session_id,
                min_severity=AnomalySeverity.CRITICAL,
                statuses=frozenset({AlertStatus.ACTIVE}),
                max_results=max_results,
                sort_order=AlertSortOrder.PRIORITY_DESC,
            )
        )

    def session_alert_summary(
        self,
        session_id: str,
        *,
        max_priority_alerts: int = 5,
    ) -> dict[str, Any]:
        """Produce a compact summary of alerts for a session.

        Returns a dict suitable for embedding in agent loop context
        that gives the LLM a quick overview of the alert state for
        a session without returning every individual alert.

        Args:
            session_id: The SSH session to summarize.
            max_priority_alerts: Number of top-priority alerts to
                include in the detail section.

        Returns:
            Dict with session_id, counts by severity/status, and
            the top priority alerts.
        """
        all_result = self.query(
            AlertQuery(
                session_id=session_id,
                max_results=max(max_priority_alerts, 1),
                sort_order=AlertSortOrder.PRIORITY_DESC,
            )
        )

        # Count by severity
        by_severity: dict[str, int] = {}
        by_status: dict[str, int] = {}

        # Need all matching alerts for accurate counts
        all_alerts = self._gather_alerts(session_id=session_id)
        for alert in all_alerts:
            sev_key = alert.severity.value
            by_severity[sev_key] = by_severity.get(sev_key, 0) + 1
            status_key = alert.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

        highest_priority_alerts = [
            a.to_agent_summary()
            for a in all_result.alerts[:max_priority_alerts]
        ]

        return {
            "session_id": session_id,
            "total_alerts": len(all_alerts),
            "by_severity": by_severity,
            "by_status": by_status,
            "highest_priority_alerts": highest_priority_alerts,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_alerts(
        self,
        *,
        session_id: str | None = None,
    ) -> tuple[Alert, ...]:
        """Gather raw alerts from the collector.

        When session_id is provided, returns alerts for that session
        only. Otherwise, returns alerts across all sessions.

        Args:
            session_id: Optional session filter.

        Returns:
            Tuple of Alert records.
        """
        if session_id is not None:
            return self._collector.get_alerts(session_id)

        # Gather from all sessions
        all_alerts: list[Alert] = []
        for sid in self._collector.session_ids:
            all_alerts.extend(self._collector.get_alerts(sid))
        return tuple(all_alerts)

    @staticmethod
    def _sort_alerts(
        alerts: tuple[QueryableAlert, ...],
        sort_order: AlertSortOrder,
    ) -> list[QueryableAlert]:
        """Sort queryable alerts according to the specified order.

        Args:
            alerts: Tuple of scored alerts to sort.
            sort_order: The desired sort order.

        Returns:
            List of alerts in the specified order.
        """
        if sort_order is AlertSortOrder.TIME_DESC:
            return sorted(alerts, key=_sort_key_time, reverse=True)

        if sort_order is AlertSortOrder.SEVERITY_DESC:
            return sorted(alerts, key=_sort_key_severity)

        # Default: PRIORITY_DESC
        return sorted(alerts, key=_sort_key_priority)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AlertQueryService(collector={self._collector!r})"
        )
