"""Tests for the AlertCollector core that receives and stores detector results.

Verifies that the AlertCollector:
- Receives DispatchResult objects and creates Alert records
- Stores alerts keyed by session_id with O(1) lookup by alert_id
- Returns immutable CollectResult describing what was collected
- Respects max_alerts_per_session capacity (evicts oldest first)
- Respects max_total_alerts capacity (evicts oldest globally)
- Provides query methods: get_alert, get_alerts, active_alerts
- Produces AlertSnapshot summaries with correct aggregation
- Supports acknowledge and resolve status transitions
- Clears alerts by session and globally
- Is thread-safe (uses threading.Lock)
- Skips dispatch results with no anomaly reports
- Generates unique alert IDs
- Never mutates stored Alert instances
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from jules_daemon.monitor.alert_collector import AlertCollector
from jules_daemon.monitor.alert_models import (
    AlertCollectorConfig,
    AlertStatus,
)
from jules_daemon.monitor.anomaly_models import (
    AnomalyReport,
    AnomalySeverity,
    PatternType,
)
from jules_daemon.monitor.detector_dispatcher import (
    DetectorError,
    DispatchResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)


def _make_report(
    *,
    pattern_name: str = "oom_killer",
    severity: AnomalySeverity = AnomalySeverity.WARNING,
    session_id: str = "session-1",
) -> AnomalyReport:
    """Create a minimal AnomalyReport for testing."""
    return AnomalyReport(
        pattern_name=pattern_name,
        pattern_type=PatternType.ERROR_KEYWORD,
        severity=severity,
        message=f"Detected {pattern_name}",
        detected_at=_NOW,
        session_id=session_id,
    )


def _make_dispatch_result(
    *,
    session_id: str = "session-1",
    reports: tuple[AnomalyReport, ...] = (),
    errors: tuple[DetectorError, ...] = (),
) -> DispatchResult:
    """Create a minimal DispatchResult for testing."""
    return DispatchResult(
        output_line="test output line",
        session_id=session_id,
        reports=reports,
        errors=errors,
        dispatched_at=_NOW,
    )


def _make_dispatch_with_reports(
    *,
    session_id: str = "session-1",
    count: int = 1,
    severity: AnomalySeverity = AnomalySeverity.WARNING,
) -> DispatchResult:
    """Create a DispatchResult with the specified number of anomaly reports."""
    reports = tuple(
        _make_report(
            pattern_name=f"pattern_{i}",
            severity=severity,
            session_id=session_id,
        )
        for i in range(count)
    )
    return _make_dispatch_result(session_id=session_id, reports=reports)


# ---------------------------------------------------------------------------
# Construction and configuration
# ---------------------------------------------------------------------------


class TestAlertCollectorConstruction:
    """Tests for AlertCollector initialization."""

    def test_default_construction(self) -> None:
        collector = AlertCollector()
        assert collector.alert_count == 0
        assert collector.session_ids == frozenset()

    def test_construction_with_config(self) -> None:
        config = AlertCollectorConfig(
            max_alerts_per_session=10,
            max_total_alerts=50,
        )
        collector = AlertCollector(config=config)
        assert collector.alert_count == 0

    def test_repr(self) -> None:
        collector = AlertCollector()
        result = repr(collector)
        assert "AlertCollector" in result
        assert "0 alerts" in result


# ---------------------------------------------------------------------------
# Collecting dispatch results
# ---------------------------------------------------------------------------


class TestAlertCollectorCollect:
    """Tests for collecting DispatchResult into alerts."""

    def test_collect_empty_dispatch(self) -> None:
        """Dispatch with no reports produces no alerts."""
        collector = AlertCollector()
        dispatch = _make_dispatch_result()
        result = collector.collect(dispatch)

        assert result.alert_count == 0
        assert result.has_new_alerts is False
        assert result.session_id == "session-1"
        assert collector.alert_count == 0

    def test_collect_single_report(self) -> None:
        """Single anomaly report produces one alert."""
        collector = AlertCollector()
        report = _make_report()
        dispatch = _make_dispatch_result(reports=(report,))

        result = collector.collect(dispatch)

        assert result.alert_count == 1
        assert result.has_new_alerts is True
        assert len(result.new_alerts) == 1
        assert result.new_alerts[0].status is AlertStatus.ACTIVE
        assert result.new_alerts[0].session_id == "session-1"
        assert result.new_alerts[0].anomaly_report is report
        assert collector.alert_count == 1

    def test_collect_multiple_reports(self) -> None:
        """Multiple anomaly reports produce multiple alerts."""
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(count=3)

        result = collector.collect(dispatch)

        assert result.alert_count == 3
        assert collector.alert_count == 3

    def test_collect_generates_unique_alert_ids(self) -> None:
        """Each collected alert gets a unique ID."""
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(count=5)

        result = collector.collect(dispatch)

        alert_ids = {a.alert_id for a in result.new_alerts}
        assert len(alert_ids) == 5  # All unique

    def test_collect_preserves_session_id(self) -> None:
        """Alerts inherit session_id from the dispatch result."""
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(session_id="run-42")

        result = collector.collect(dispatch)

        assert result.session_id == "run-42"
        assert all(a.session_id == "run-42" for a in result.new_alerts)

    def test_collect_from_multiple_sessions(self) -> None:
        """Alerts from different sessions are stored separately."""
        collector = AlertCollector()

        d1 = _make_dispatch_with_reports(session_id="session-A", count=2)
        d2 = _make_dispatch_with_reports(session_id="session-B", count=3)

        collector.collect(d1)
        collector.collect(d2)

        assert collector.alert_count == 5
        assert collector.session_ids == frozenset({"session-A", "session-B"})

    def test_collect_ignores_dispatch_errors(self) -> None:
        """Detector errors in the dispatch are not turned into alerts."""
        collector = AlertCollector()
        error = DetectorError(detector_name="broken", error="boom")
        dispatch = _make_dispatch_result(errors=(error,))

        result = collector.collect(dispatch)

        assert result.alert_count == 0
        assert collector.alert_count == 0

    def test_collect_returns_collected_at_timestamp(self) -> None:
        """CollectResult has a valid collected_at timestamp."""
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(count=1)

        before = datetime.now(timezone.utc)
        result = collector.collect(dispatch)
        after = datetime.now(timezone.utc)

        assert before <= result.collected_at <= after


# ---------------------------------------------------------------------------
# Querying alerts
# ---------------------------------------------------------------------------


class TestAlertCollectorQuery:
    """Tests for querying stored alerts."""

    def test_get_alerts_by_session(self) -> None:
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(session_id="session-1", count=2)
        collector.collect(dispatch)

        alerts = collector.get_alerts("session-1")
        assert len(alerts) == 2
        assert all(a.session_id == "session-1" for a in alerts)

    def test_get_alerts_empty_session(self) -> None:
        collector = AlertCollector()
        alerts = collector.get_alerts("nonexistent")
        assert alerts == ()

    def test_get_alerts_returns_tuple(self) -> None:
        """Returned collection is an immutable tuple snapshot."""
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(count=1)
        collector.collect(dispatch)

        alerts = collector.get_alerts("session-1")
        assert isinstance(alerts, tuple)

    def test_get_alert_by_id(self) -> None:
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(count=1)
        result = collector.collect(dispatch)

        alert_id = result.new_alerts[0].alert_id
        found = collector.get_alert(alert_id)
        assert found is not None
        assert found.alert_id == alert_id

    def test_get_alert_nonexistent(self) -> None:
        collector = AlertCollector()
        assert collector.get_alert("nonexistent") is None

    def test_active_alerts_all_sessions(self) -> None:
        collector = AlertCollector()
        d1 = _make_dispatch_with_reports(session_id="s1", count=2)
        d2 = _make_dispatch_with_reports(session_id="s2", count=1)
        collector.collect(d1)
        collector.collect(d2)

        active = collector.active_alerts()
        assert len(active) == 3
        assert all(a.is_active for a in active)

    def test_active_alerts_specific_session(self) -> None:
        collector = AlertCollector()
        d1 = _make_dispatch_with_reports(session_id="s1", count=2)
        d2 = _make_dispatch_with_reports(session_id="s2", count=1)
        collector.collect(d1)
        collector.collect(d2)

        active_s1 = collector.active_alerts(session_id="s1")
        assert len(active_s1) == 2
        assert all(a.session_id == "s1" for a in active_s1)

    def test_active_alerts_excludes_acknowledged(self) -> None:
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(count=2)
        result = collector.collect(dispatch)

        # Acknowledge one
        alert_id = result.new_alerts[0].alert_id
        collector.acknowledge(alert_id)

        active = collector.active_alerts()
        assert len(active) == 1

    def test_active_alerts_excludes_resolved(self) -> None:
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(count=2)
        result = collector.collect(dispatch)

        # Resolve one
        alert_id = result.new_alerts[0].alert_id
        collector.resolve(alert_id)

        active = collector.active_alerts()
        assert len(active) == 1


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


class TestAlertCollectorStatusTransitions:
    """Tests for acknowledge and resolve status transitions."""

    def test_acknowledge_alert(self) -> None:
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(count=1)
        result = collector.collect(dispatch)
        alert_id = result.new_alerts[0].alert_id

        acknowledged = collector.acknowledge(alert_id)

        assert acknowledged.status is AlertStatus.ACKNOWLEDGED
        assert acknowledged.alert_id == alert_id
        # Stored alert is updated
        stored = collector.get_alert(alert_id)
        assert stored is not None
        assert stored.status is AlertStatus.ACKNOWLEDGED

    def test_resolve_alert(self) -> None:
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(count=1)
        result = collector.collect(dispatch)
        alert_id = result.new_alerts[0].alert_id

        resolved = collector.resolve(alert_id)

        assert resolved.status is AlertStatus.RESOLVED
        stored = collector.get_alert(alert_id)
        assert stored is not None
        assert stored.status is AlertStatus.RESOLVED

    def test_acknowledge_nonexistent_raises(self) -> None:
        collector = AlertCollector()
        with pytest.raises(KeyError, match="No alert"):
            collector.acknowledge("nonexistent")

    def test_resolve_nonexistent_raises(self) -> None:
        collector = AlertCollector()
        with pytest.raises(KeyError, match="No alert"):
            collector.resolve("nonexistent")

    def test_acknowledge_preserves_other_fields(self) -> None:
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(count=1)
        result = collector.collect(dispatch)
        original = result.new_alerts[0]

        acknowledged = collector.acknowledge(original.alert_id)

        assert acknowledged.anomaly_report is original.anomaly_report
        assert acknowledged.session_id == original.session_id
        assert acknowledged.created_at == original.created_at

    def test_resolve_from_acknowledged(self) -> None:
        """Can resolve an already-acknowledged alert."""
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(count=1)
        result = collector.collect(dispatch)
        alert_id = result.new_alerts[0].alert_id

        collector.acknowledge(alert_id)
        resolved = collector.resolve(alert_id)

        assert resolved.status is AlertStatus.RESOLVED


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestAlertCollectorSnapshot:
    """Tests for session snapshot generation."""

    def test_snapshot_empty_session(self) -> None:
        collector = AlertCollector()
        snapshot = collector.session_snapshot("nonexistent")

        assert snapshot.session_id == "nonexistent"
        assert snapshot.total_alerts == 0
        assert snapshot.active_count == 0
        assert snapshot.acknowledged_count == 0
        assert snapshot.resolved_count == 0
        assert snapshot.highest_severity is None
        assert snapshot.alerts == ()

    def test_snapshot_with_alerts(self) -> None:
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(
            session_id="s1",
            count=3,
            severity=AnomalySeverity.WARNING,
        )
        collector.collect(dispatch)

        snapshot = collector.session_snapshot("s1")

        assert snapshot.session_id == "s1"
        assert snapshot.total_alerts == 3
        assert snapshot.active_count == 3
        assert snapshot.acknowledged_count == 0
        assert snapshot.resolved_count == 0
        assert snapshot.highest_severity is AnomalySeverity.WARNING

    def test_snapshot_with_mixed_statuses(self) -> None:
        collector = AlertCollector()
        dispatch = _make_dispatch_with_reports(session_id="s1", count=3)
        result = collector.collect(dispatch)

        collector.acknowledge(result.new_alerts[0].alert_id)
        collector.resolve(result.new_alerts[1].alert_id)

        snapshot = collector.session_snapshot("s1")

        assert snapshot.total_alerts == 3
        assert snapshot.active_count == 1
        assert snapshot.acknowledged_count == 1
        assert snapshot.resolved_count == 1

    def test_snapshot_highest_severity(self) -> None:
        collector = AlertCollector()
        # Add alerts with different severities
        r1 = _make_report(
            pattern_name="info_p",
            severity=AnomalySeverity.INFO,
        )
        r2 = _make_report(
            pattern_name="crit_p",
            severity=AnomalySeverity.CRITICAL,
        )
        dispatch = _make_dispatch_result(reports=(r1, r2))
        collector.collect(dispatch)

        snapshot = collector.session_snapshot("session-1")
        assert snapshot.highest_severity is AnomalySeverity.CRITICAL


# ---------------------------------------------------------------------------
# Capacity and eviction
# ---------------------------------------------------------------------------


class TestAlertCollectorCapacity:
    """Tests for capacity management and eviction."""

    def test_per_session_eviction(self) -> None:
        """Oldest alerts evicted when per-session limit reached."""
        config = AlertCollectorConfig(
            max_alerts_per_session=3,
            max_total_alerts=100,
        )
        collector = AlertCollector(config=config)

        # Add 3 alerts (at capacity)
        d1 = _make_dispatch_with_reports(count=3)
        r1 = collector.collect(d1)
        assert r1.eviction_count == 0
        first_alert_id = r1.new_alerts[0].alert_id

        # Add 1 more -> should evict oldest
        d2 = _make_dispatch_with_reports(count=1)
        r2 = collector.collect(d2)
        assert r2.eviction_count == 1
        assert r2.evicted_alerts[0].alert_id == first_alert_id
        assert collector.alert_count == 3  # Still at limit

    def test_per_session_eviction_multiple(self) -> None:
        """Adding multiple alerts at once can evict multiple oldest."""
        config = AlertCollectorConfig(
            max_alerts_per_session=2,
            max_total_alerts=100,
        )
        collector = AlertCollector(config=config)

        # Fill to capacity
        d1 = _make_dispatch_with_reports(count=2)
        collector.collect(d1)

        # Add 2 more -> should evict 2
        d2 = _make_dispatch_with_reports(count=2)
        r2 = collector.collect(d2)
        assert r2.eviction_count == 2
        assert collector.alert_count == 2

    def test_global_eviction(self) -> None:
        """Oldest alerts evicted globally when total limit reached."""
        config = AlertCollectorConfig(
            max_alerts_per_session=10,
            max_total_alerts=3,
        )
        collector = AlertCollector(config=config)

        # Add 2 to session A, 1 to session B (at global capacity)
        d_a = _make_dispatch_with_reports(session_id="sA", count=2)
        d_b = _make_dispatch_with_reports(session_id="sB", count=1)
        result_a = collector.collect(d_a)
        collector.collect(d_b)
        assert collector.alert_count == 3

        oldest_id = result_a.new_alerts[0].alert_id

        # Add 1 more to session B -> evicts oldest globally (from sA)
        d_b2 = _make_dispatch_with_reports(session_id="sB", count=1)
        r_b2 = collector.collect(d_b2)
        assert r_b2.has_evictions is True
        assert collector.alert_count == 3
        # Verify the oldest alert from sA was evicted
        assert collector.get_alert(oldest_id) is None

    def test_evicted_alert_no_longer_queryable(self) -> None:
        """Evicted alerts are removed from all indices."""
        config = AlertCollectorConfig(
            max_alerts_per_session=1,
            max_total_alerts=100,
        )
        collector = AlertCollector(config=config)

        d1 = _make_dispatch_with_reports(count=1)
        r1 = collector.collect(d1)
        evicted_id = r1.new_alerts[0].alert_id

        d2 = _make_dispatch_with_reports(count=1)
        collector.collect(d2)

        assert collector.get_alert(evicted_id) is None


# ---------------------------------------------------------------------------
# Clearing alerts
# ---------------------------------------------------------------------------


class TestAlertCollectorClear:
    """Tests for clearing alerts."""

    def test_clear_session(self) -> None:
        collector = AlertCollector()
        d1 = _make_dispatch_with_reports(session_id="s1", count=3)
        d2 = _make_dispatch_with_reports(session_id="s2", count=2)
        collector.collect(d1)
        collector.collect(d2)

        removed = collector.clear_session("s1")

        assert removed == 3
        assert collector.alert_count == 2
        assert "s1" not in collector.session_ids

    def test_clear_nonexistent_session(self) -> None:
        collector = AlertCollector()
        removed = collector.clear_session("nonexistent")
        assert removed == 0

    def test_clear_all(self) -> None:
        collector = AlertCollector()
        d1 = _make_dispatch_with_reports(session_id="s1", count=2)
        d2 = _make_dispatch_with_reports(session_id="s2", count=3)
        collector.collect(d1)
        collector.collect(d2)

        removed = collector.clear_all()

        assert removed == 5
        assert collector.alert_count == 0
        assert collector.session_ids == frozenset()

    def test_clear_all_empty_collector(self) -> None:
        collector = AlertCollector()
        removed = collector.clear_all()
        assert removed == 0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestAlertCollectorProperties:
    """Tests for collector properties."""

    def test_alert_count(self) -> None:
        collector = AlertCollector()
        assert collector.alert_count == 0

        dispatch = _make_dispatch_with_reports(count=3)
        collector.collect(dispatch)
        assert collector.alert_count == 3

    def test_session_ids(self) -> None:
        collector = AlertCollector()
        assert collector.session_ids == frozenset()

        d1 = _make_dispatch_with_reports(session_id="s1", count=1)
        d2 = _make_dispatch_with_reports(session_id="s2", count=1)
        collector.collect(d1)
        collector.collect(d2)

        assert collector.session_ids == frozenset({"s1", "s2"})

    def test_session_ids_returns_frozenset(self) -> None:
        collector = AlertCollector()
        ids = collector.session_ids
        assert isinstance(ids, frozenset)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestAlertCollectorThreadSafety:
    """Tests verifying thread-safe concurrent access."""

    def test_concurrent_collect(self) -> None:
        """Multiple threads can collect simultaneously without corruption."""
        collector = AlertCollector()
        errors: list[str] = []

        def worker(session_id: str) -> None:
            try:
                for _ in range(10):
                    dispatch = _make_dispatch_with_reports(
                        session_id=session_id, count=1
                    )
                    collector.collect(dispatch)
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=worker, args=(f"session-{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert collector.alert_count == 50  # 5 threads * 10 alerts each

    def test_concurrent_collect_with_eviction(self) -> None:
        """Concurrent collects under tight capacity maintain index consistency."""
        config = AlertCollectorConfig(
            max_alerts_per_session=3,
            max_total_alerts=15,
        )
        collector = AlertCollector(config=config)
        errors: list[str] = []

        def worker(session_id: str) -> None:
            try:
                for _ in range(10):
                    dispatch = _make_dispatch_with_reports(
                        session_id=session_id, count=1
                    )
                    collector.collect(dispatch)
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=worker, args=(f"session-{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # Per-session limit is 3, 5 sessions -> max 15
        assert collector.alert_count <= 15
        # Each session should have at most 3 alerts
        for sid in collector.session_ids:
            alerts = collector.get_alerts(sid)
            assert len(alerts) <= 3

    def test_concurrent_collect_and_query(self) -> None:
        """Collect and query can run concurrently without deadlock."""
        collector = AlertCollector()
        errors: list[str] = []

        def collect_worker() -> None:
            try:
                for _ in range(20):
                    dispatch = _make_dispatch_with_reports(count=1)
                    collector.collect(dispatch)
            except Exception as exc:
                errors.append(f"collect: {exc}")

        def query_worker() -> None:
            try:
                for _ in range(20):
                    collector.get_alerts("session-1")
                    collector.active_alerts()
                    _ = collector.alert_count
            except Exception as exc:
                errors.append(f"query: {exc}")

        t1 = threading.Thread(target=collect_worker)
        t2 = threading.Thread(target=query_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"
