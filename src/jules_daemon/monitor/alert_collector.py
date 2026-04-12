"""Alert collector core that receives and stores detector results.

Bridges the detector dispatcher output (``DispatchResult`` containing
``AnomalyReport`` records) with the alert storage and query layer.
Each anomaly report is wrapped in an immutable ``Alert`` with lifecycle
metadata (unique ID, status, timestamp) and indexed for fast retrieval.

The collector is the central authority for alert state. Other components
(notification broadcaster, CLI status handler, agent loop tools) query
the collector rather than maintaining their own alert caches.

Key properties:

- **Immutable alerts**: All ``Alert`` records are frozen dataclasses.
  Status transitions (acknowledge, resolve) replace the stored reference
  with a new instance rather than mutating in place.

- **Thread-safe**: All mutable state is guarded by a ``threading.Lock``.
  Safe for concurrent access from the async monitoring pipeline, the
  IPC handler thread, and the agent loop.

- **Capacity management**: Configurable per-session and global limits
  prevent unbounded memory growth. When limits are reached, the oldest
  alerts (by creation time) are evicted first. Evictions are reported
  in the ``CollectResult`` for audit logging.

- **Dual indexing**: Alerts are stored both by session (ordered list
  for session queries and eviction) and by alert ID (dict for O(1)
  lookup). Both indices are updated atomically under the lock.

- **Snapshot-based returns**: All query methods return immutable
  collections (tuples, frozensets) that are safe to use across async
  boundaries without synchronization.

Usage::

    from jules_daemon.monitor.alert_collector import AlertCollector
    from jules_daemon.monitor.alert_models import AlertCollectorConfig

    config = AlertCollectorConfig(
        max_alerts_per_session=100,
        max_total_alerts=1000,
    )
    collector = AlertCollector(config=config)

    # Collect results from the detector dispatcher
    result = collector.collect(dispatch_result)
    for alert in result.new_alerts:
        print(f"New alert: {alert.alert_id} ({alert.severity})")

    # Query alerts
    active = collector.active_alerts(session_id="run-42")
    snapshot = collector.session_snapshot("run-42")

    # Transition alert status
    collector.acknowledge(alert_id)
    collector.resolve(alert_id)
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Final

from jules_daemon.monitor.alert_models import (
    Alert,
    AlertCollectorConfig,
    AlertSnapshot,
    AlertStatus,
    CollectResult,
)
from jules_daemon.monitor.anomaly_models import AnomalyReport, AnomalySeverity
from jules_daemon.monitor.detector_dispatcher import DispatchResult

__all__ = ["AlertCollector"]

logger = logging.getLogger(__name__)


def _generate_alert_id() -> str:
    """Generate a unique alert ID.

    Returns:
        A string of the form ``"alert-<hex>"`` where hex is a 16-char
        random hex string derived from a UUID4.
    """
    return f"alert-{uuid.uuid4().hex[:16]}"


class AlertCollector:
    """Core alert collector that receives and stores detector results.

    Maintains a dual index of alerts: by session ID (for ordered
    per-session queries and capacity eviction) and by alert ID (for
    O(1) direct lookup). All state mutations are guarded by a
    threading.Lock for thread safety.

    Args:
        config: Optional configuration controlling capacity limits.
            Uses defaults (100 per session, 1000 total) if not provided.
    """

    __slots__ = ("_config", "_lock", "_by_session", "_by_id")

    def __init__(
        self,
        *,
        config: AlertCollectorConfig | None = None,
    ) -> None:
        """Initialize the alert collector.

        Args:
            config: Optional capacity configuration. Defaults to
                ``AlertCollectorConfig()`` with default limits.
        """
        self._config: Final[AlertCollectorConfig] = (
            config or AlertCollectorConfig()
        )
        self._lock: Final[threading.Lock] = threading.Lock()
        # Session -> ordered list of alerts (insertion order = age order)
        self._by_session: dict[str, list[Alert]] = {}
        # Alert ID -> Alert for O(1) lookup
        self._by_id: dict[str, Alert] = {}

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(self, dispatch_result: DispatchResult) -> CollectResult:
        """Process a DispatchResult and create alerts from anomaly reports.

        Each ``AnomalyReport`` in the dispatch result produces one
        ``Alert`` in ACTIVE state. Alerts are stored and indexed.
        If capacity limits are exceeded, the oldest alerts are evicted.

        Args:
            dispatch_result: Result from the ``DetectorDispatcher``
                containing anomaly reports and detector errors.

        Returns:
            Immutable ``CollectResult`` describing the new alerts
            created and any alerts evicted due to capacity.
        """
        now = datetime.now(timezone.utc)
        session_id = dispatch_result.session_id
        reports = dispatch_result.reports

        if not reports:
            return CollectResult(
                new_alerts=(),
                evicted_alerts=(),
                session_id=session_id,
                collected_at=now,
            )

        # Create alert records from reports
        new_alerts = tuple(
            self._create_alert(report=report, session_id=session_id, now=now)
            for report in reports
        )

        evicted: list[Alert] = []

        with self._lock:
            session_alerts = self._by_session.setdefault(session_id, [])

            # Add new alerts to both indices
            for alert in new_alerts:
                session_alerts.append(alert)
                self._by_id[alert.alert_id] = alert

            # Enforce per-session capacity (evict oldest first)
            per_session_evicted = self._enforce_per_session_limit(
                session_alerts
            )
            evicted.extend(per_session_evicted)

            # Enforce global capacity (evict oldest globally)
            global_evicted = self._enforce_global_limit()
            evicted.extend(global_evicted)

        if evicted:
            logger.info(
                "Evicted %d alert(s) during collection for session %s",
                len(evicted),
                session_id,
            )

        return CollectResult(
            new_alerts=new_alerts,
            evicted_alerts=tuple(evicted),
            session_id=session_id,
            collected_at=now,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_alert(self, alert_id: str) -> Alert | None:
        """Look up a single alert by its unique ID.

        Args:
            alert_id: The unique alert identifier.

        Returns:
            The Alert if found, or None if not stored.
        """
        with self._lock:
            return self._by_id.get(alert_id)

    def get_alerts(self, session_id: str) -> tuple[Alert, ...]:
        """Return all alerts for a session as an immutable tuple.

        Args:
            session_id: The SSH session identifier.

        Returns:
            Tuple of Alert records for the session, in insertion
            order. Empty tuple if no alerts exist for the session.
        """
        with self._lock:
            alerts = self._by_session.get(session_id)
            if alerts is None:
                return ()
            return tuple(alerts)

    def active_alerts(
        self,
        *,
        session_id: str | None = None,
    ) -> tuple[Alert, ...]:
        """Return all alerts in ACTIVE state.

        Args:
            session_id: Optional filter. When provided, only returns
                active alerts for that specific session. When None,
                returns active alerts across all sessions.

        Returns:
            Tuple of Alert records in ACTIVE state.
        """
        with self._lock:
            if session_id is not None:
                alerts = self._by_session.get(session_id, [])
                return tuple(a for a in alerts if a.is_active)

            return tuple(
                a
                for session_alerts in self._by_session.values()
                for a in session_alerts
                if a.is_active
            )

    def session_snapshot(self, session_id: str) -> AlertSnapshot:
        """Produce an aggregated snapshot of alerts for a session.

        Args:
            session_id: The SSH session identifier.

        Returns:
            Immutable ``AlertSnapshot`` with counts, severity info,
            and the full list of alerts.
        """
        with self._lock:
            alerts = self._by_session.get(session_id)
            if alerts is None:
                return AlertSnapshot(
                    session_id=session_id,
                    total_alerts=0,
                    active_count=0,
                    acknowledged_count=0,
                    resolved_count=0,
                    highest_severity=None,
                    alerts=(),
                )

            active = 0
            acknowledged = 0
            resolved = 0
            max_severity: AnomalySeverity | None = None

            for alert in alerts:
                if alert.status is AlertStatus.ACTIVE:
                    active += 1
                elif alert.status is AlertStatus.ACKNOWLEDGED:
                    acknowledged += 1
                elif alert.status is AlertStatus.RESOLVED:
                    resolved += 1

                severity = alert.severity
                if max_severity is None or (
                    severity.numeric_level > max_severity.numeric_level
                ):
                    max_severity = severity

            return AlertSnapshot(
                session_id=session_id,
                total_alerts=len(alerts),
                active_count=active,
                acknowledged_count=acknowledged,
                resolved_count=resolved,
                highest_severity=max_severity,
                alerts=tuple(alerts),
            )

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def acknowledge(self, alert_id: str) -> Alert:
        """Transition an alert to ACKNOWLEDGED status.

        Creates a new Alert with ACKNOWLEDGED status and replaces
        the stored reference. The original Alert is not mutated.

        Args:
            alert_id: The unique alert identifier.

        Returns:
            The new Alert instance with ACKNOWLEDGED status.

        Raises:
            KeyError: If no alert with the given ID exists.
        """
        return self._transition_status(alert_id, AlertStatus.ACKNOWLEDGED)

    def resolve(self, alert_id: str) -> Alert:
        """Transition an alert to RESOLVED status.

        Creates a new Alert with RESOLVED status and replaces
        the stored reference. The original Alert is not mutated.

        Args:
            alert_id: The unique alert identifier.

        Returns:
            The new Alert instance with RESOLVED status.

        Raises:
            KeyError: If no alert with the given ID exists.
        """
        return self._transition_status(alert_id, AlertStatus.RESOLVED)

    # ------------------------------------------------------------------
    # Clearing
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> int:
        """Remove all alerts for a session.

        Args:
            session_id: The SSH session to clear.

        Returns:
            Number of alerts removed.
        """
        with self._lock:
            alerts = self._by_session.pop(session_id, None)
            if alerts is None:
                return 0

            for alert in alerts:
                self._by_id.pop(alert.alert_id, None)

            count = len(alerts)

        logger.debug("Cleared %d alerts for session %s", count, session_id)
        return count

    def clear_all(self) -> int:
        """Remove all alerts across all sessions.

        Returns:
            Total number of alerts removed.
        """
        with self._lock:
            count = len(self._by_id)
            self._by_session.clear()
            self._by_id.clear()

        if count:
            logger.debug("Cleared all %d alerts", count)
        return count

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alert_count(self) -> int:
        """Total number of stored alerts across all sessions."""
        with self._lock:
            return len(self._by_id)

    @property
    def session_ids(self) -> frozenset[str]:
        """Frozenset of session IDs that have stored alerts.

        Returns a snapshot; safe to use without synchronization.
        """
        with self._lock:
            return frozenset(self._by_session.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_alert(
        *,
        report: AnomalyReport,
        session_id: str,
        now: datetime,
    ) -> Alert:
        """Create a new Alert record from an AnomalyReport.

        Args:
            report: The anomaly report from the detector.
            session_id: The SSH session being monitored.
            now: Creation timestamp.

        Returns:
            A new Alert in ACTIVE state.
        """
        return Alert(
            alert_id=_generate_alert_id(),
            anomaly_report=report,
            session_id=session_id,
            status=AlertStatus.ACTIVE,
            created_at=now,
        )

    def _transition_status(
        self,
        alert_id: str,
        new_status: AlertStatus,
    ) -> Alert:
        """Transition an alert to a new status.

        Replaces the stored reference in both indices with a new
        Alert instance. The original is not mutated.

        Args:
            alert_id: The unique alert identifier.
            new_status: The target lifecycle status.

        Returns:
            The new Alert instance with the updated status.

        Raises:
            KeyError: If no alert with the given ID exists.
        """
        with self._lock:
            old_alert = self._by_id.get(alert_id)
            if old_alert is None:
                raise KeyError(
                    f"No alert with ID {alert_id!r} exists"
                )

            new_alert = old_alert.with_status(new_status)

            # Update the ID index
            self._by_id[alert_id] = new_alert

            # Update the session list (replace in-place by index)
            session_alerts = self._by_session.get(old_alert.session_id)
            if session_alerts is not None:
                for i, a in enumerate(session_alerts):
                    if a.alert_id == alert_id:
                        session_alerts[i] = new_alert
                        break

        logger.debug(
            "Alert %s transitioned to %s", alert_id, new_status.value
        )
        return new_alert

    def _enforce_per_session_limit(
        self,
        session_alerts: list[Alert],
    ) -> list[Alert]:
        """Evict oldest alerts from a session to enforce per-session limit.

        Must be called under the lock.

        Args:
            session_alerts: The session's alert list (mutable, modified
                in place).

        Returns:
            List of evicted alerts (may be empty).
        """
        limit = self._config.max_alerts_per_session
        overflow = len(session_alerts) - limit
        if overflow <= 0:
            return []

        evicted = session_alerts[:overflow]
        del session_alerts[:overflow]

        for alert in evicted:
            self._by_id.pop(alert.alert_id, None)

        return evicted

    def _enforce_global_limit(self) -> list[Alert]:
        """Evict oldest alerts globally to enforce total limit.

        Must be called under the lock. Builds a sorted eviction
        queue of all alerts by creation time, then removes the
        oldest ones. O(T log T) where T = total alerts, rather
        than O(k * S) per eviction.

        Returns:
            List of globally evicted alerts (may be empty).
        """
        limit = self._config.max_total_alerts
        total = len(self._by_id)
        overflow = total - limit
        if overflow <= 0:
            return []

        # Build a sorted list of (created_at, session_id, index) for
        # all alerts, then evict the oldest `overflow` entries.
        candidates: list[tuple[datetime, str, int]] = []
        for sid, alerts in self._by_session.items():
            for idx, alert in enumerate(alerts):
                candidates.append((alert.created_at, sid, idx))

        # Sort by creation time (oldest first); stable sort preserves
        # insertion order for equal timestamps.
        candidates.sort(key=lambda c: c[0])

        evicted: list[Alert] = []
        # Track which indices to remove per session (in reverse order
        # so that removing by index doesn't shift later indices).
        removals_by_session: dict[str, list[int]] = {}

        for i in range(min(overflow, len(candidates))):
            _, sid, idx = candidates[i]
            removals_by_session.setdefault(sid, []).append(idx)

        # Remove in reverse index order within each session
        for sid, indices in removals_by_session.items():
            session_alerts = self._by_session[sid]
            for idx in sorted(indices, reverse=True):
                removed = session_alerts.pop(idx)
                self._by_id.pop(removed.alert_id, None)
                evicted.append(removed)

            # Clean up empty session lists
            if not session_alerts:
                del self._by_session[sid]

        return evicted

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._by_id)
            sessions = len(self._by_session)
        return (
            f"AlertCollector({count} alerts across {sessions} sessions)"
        )
