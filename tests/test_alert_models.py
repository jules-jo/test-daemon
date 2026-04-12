"""Tests for alert data model used by the alert collector subsystem.

Verifies that the alert models:
- Define an AlertStatus enum with ACTIVE, ACKNOWLEDGED, RESOLVED states
- Create immutable Alert records wrapping AnomalyReport with metadata
- Validate all required fields at construction time
- Support immutable status transitions via with_status()
- Produce AlertSnapshot summaries with correct aggregations
- Enforce AlertCollectorConfig capacity constraints
- Produce immutable CollectResult records from dispatch processing
- Never mutate existing data -- all state transitions produce new instances
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jules_daemon.monitor.alert_models import (
    Alert,
    AlertCollectorConfig,
    AlertSnapshot,
    AlertStatus,
    CollectResult,
)
from jules_daemon.monitor.anomaly_models import (
    AnomalyReport,
    AnomalySeverity,
    PatternType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)
_LATER = datetime(2026, 4, 12, 12, 5, 0, tzinfo=timezone.utc)


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


def _make_alert(
    *,
    alert_id: str = "alert-001",
    session_id: str = "session-1",
    status: AlertStatus = AlertStatus.ACTIVE,
    severity: AnomalySeverity = AnomalySeverity.WARNING,
    created_at: datetime = _NOW,
) -> Alert:
    """Create a minimal Alert for testing."""
    report = _make_report(session_id=session_id, severity=severity)
    return Alert(
        alert_id=alert_id,
        anomaly_report=report,
        session_id=session_id,
        status=status,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# AlertStatus enum tests
# ---------------------------------------------------------------------------


class TestAlertStatus:
    """Verify the alert status classification enum."""

    def test_enum_values(self) -> None:
        assert AlertStatus.ACTIVE.value == "active"
        assert AlertStatus.ACKNOWLEDGED.value == "acknowledged"
        assert AlertStatus.RESOLVED.value == "resolved"

    def test_all_members_present(self) -> None:
        members = {m.value for m in AlertStatus}
        assert members == {"active", "acknowledged", "resolved"}

    def test_members_count(self) -> None:
        assert len(AlertStatus) == 3


# ---------------------------------------------------------------------------
# Alert frozen dataclass tests
# ---------------------------------------------------------------------------


class TestAlert:
    """Verify the immutable Alert record."""

    def test_construction(self) -> None:
        report = _make_report()
        alert = Alert(
            alert_id="alert-abc",
            anomaly_report=report,
            session_id="session-1",
            status=AlertStatus.ACTIVE,
            created_at=_NOW,
        )
        assert alert.alert_id == "alert-abc"
        assert alert.anomaly_report is report
        assert alert.session_id == "session-1"
        assert alert.status is AlertStatus.ACTIVE
        assert alert.created_at == _NOW

    def test_frozen(self) -> None:
        alert = _make_alert()
        with pytest.raises(AttributeError):
            alert.alert_id = "changed"  # type: ignore[misc]

    def test_frozen_status(self) -> None:
        alert = _make_alert()
        with pytest.raises(AttributeError):
            alert.status = AlertStatus.RESOLVED  # type: ignore[misc]

    def test_empty_alert_id_rejected(self) -> None:
        report = _make_report()
        with pytest.raises(ValueError, match="alert_id must not be empty"):
            Alert(
                alert_id="",
                anomaly_report=report,
                session_id="session-1",
                status=AlertStatus.ACTIVE,
                created_at=_NOW,
            )

    def test_whitespace_alert_id_rejected(self) -> None:
        report = _make_report()
        with pytest.raises(ValueError, match="alert_id must not be empty"):
            Alert(
                alert_id="   ",
                anomaly_report=report,
                session_id="session-1",
                status=AlertStatus.ACTIVE,
                created_at=_NOW,
            )

    def test_empty_session_id_rejected(self) -> None:
        report = _make_report()
        with pytest.raises(ValueError, match="session_id must not be empty"):
            Alert(
                alert_id="alert-1",
                anomaly_report=report,
                session_id="",
                status=AlertStatus.ACTIVE,
                created_at=_NOW,
            )

    def test_whitespace_session_id_stripped(self) -> None:
        report = _make_report()
        with pytest.raises(ValueError, match="session_id must not be empty"):
            Alert(
                alert_id="alert-1",
                anomaly_report=report,
                session_id="  ",
                status=AlertStatus.ACTIVE,
                created_at=_NOW,
            )

    def test_alert_id_whitespace_stripped(self) -> None:
        report = _make_report()
        alert = Alert(
            alert_id="  alert-1  ",
            anomaly_report=report,
            session_id="session-1",
            status=AlertStatus.ACTIVE,
            created_at=_NOW,
        )
        assert alert.alert_id == "alert-1"

    def test_session_id_whitespace_stripped(self) -> None:
        report = _make_report()
        alert = Alert(
            alert_id="alert-1",
            anomaly_report=report,
            session_id="  session-1  ",
            status=AlertStatus.ACTIVE,
            created_at=_NOW,
        )
        assert alert.session_id == "session-1"

    # ---------------------------------------------------------------
    # Convenience properties
    # ---------------------------------------------------------------

    def test_severity_property(self) -> None:
        alert = _make_alert(severity=AnomalySeverity.CRITICAL)
        assert alert.severity is AnomalySeverity.CRITICAL

    def test_pattern_name_property(self) -> None:
        alert = _make_alert()
        assert alert.pattern_name == "oom_killer"

    def test_pattern_type_property(self) -> None:
        alert = _make_alert()
        assert alert.pattern_type is PatternType.ERROR_KEYWORD

    def test_is_active(self) -> None:
        assert _make_alert(status=AlertStatus.ACTIVE).is_active is True
        assert _make_alert(status=AlertStatus.ACKNOWLEDGED).is_active is False
        assert _make_alert(status=AlertStatus.RESOLVED).is_active is False

    # ---------------------------------------------------------------
    # Immutable status transitions
    # ---------------------------------------------------------------

    def test_with_status_returns_new_alert(self) -> None:
        original = _make_alert(status=AlertStatus.ACTIVE)
        updated = original.with_status(AlertStatus.ACKNOWLEDGED)
        assert updated is not original
        assert updated.status is AlertStatus.ACKNOWLEDGED
        # Original is unchanged
        assert original.status is AlertStatus.ACTIVE

    def test_with_status_preserves_other_fields(self) -> None:
        original = _make_alert(
            alert_id="alert-preserve",
            session_id="session-preserve",
            status=AlertStatus.ACTIVE,
        )
        updated = original.with_status(AlertStatus.RESOLVED)
        assert updated.alert_id == original.alert_id
        assert updated.anomaly_report is original.anomaly_report
        assert updated.session_id == original.session_id
        assert updated.created_at == original.created_at

    def test_with_status_same_status_returns_new_instance(self) -> None:
        original = _make_alert(status=AlertStatus.ACTIVE)
        updated = original.with_status(AlertStatus.ACTIVE)
        assert updated is not original
        assert updated.status is AlertStatus.ACTIVE


# ---------------------------------------------------------------------------
# AlertSnapshot tests
# ---------------------------------------------------------------------------


class TestAlertSnapshot:
    """Verify the AlertSnapshot summary record."""

    def test_empty_snapshot(self) -> None:
        snapshot = AlertSnapshot(
            session_id="session-1",
            total_alerts=0,
            active_count=0,
            acknowledged_count=0,
            resolved_count=0,
            highest_severity=None,
            alerts=(),
        )
        assert snapshot.total_alerts == 0
        assert snapshot.active_count == 0
        assert snapshot.acknowledged_count == 0
        assert snapshot.resolved_count == 0
        assert snapshot.highest_severity is None
        assert snapshot.alerts == ()

    def test_snapshot_with_alerts(self) -> None:
        alert1 = _make_alert(alert_id="a1", severity=AnomalySeverity.WARNING)
        alert2 = _make_alert(alert_id="a2", severity=AnomalySeverity.CRITICAL)
        snapshot = AlertSnapshot(
            session_id="session-1",
            total_alerts=2,
            active_count=2,
            acknowledged_count=0,
            resolved_count=0,
            highest_severity=AnomalySeverity.CRITICAL,
            alerts=(alert1, alert2),
        )
        assert snapshot.total_alerts == 2
        assert snapshot.highest_severity is AnomalySeverity.CRITICAL
        assert len(snapshot.alerts) == 2

    def test_frozen(self) -> None:
        snapshot = AlertSnapshot(
            session_id="session-1",
            total_alerts=0,
            active_count=0,
            acknowledged_count=0,
            resolved_count=0,
            highest_severity=None,
            alerts=(),
        )
        with pytest.raises(AttributeError):
            snapshot.total_alerts = 5  # type: ignore[misc]

    def test_empty_session_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="session_id must not be empty"):
            AlertSnapshot(
                session_id="",
                total_alerts=0,
                active_count=0,
                acknowledged_count=0,
                resolved_count=0,
                highest_severity=None,
                alerts=(),
            )

    def test_negative_counts_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_alerts must not be negative"):
            AlertSnapshot(
                session_id="s1",
                total_alerts=-1,
                active_count=0,
                acknowledged_count=0,
                resolved_count=0,
                highest_severity=None,
                alerts=(),
            )

    def test_negative_active_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="active_count must not be negative"):
            AlertSnapshot(
                session_id="s1",
                total_alerts=0,
                active_count=-1,
                acknowledged_count=0,
                resolved_count=0,
                highest_severity=None,
                alerts=(),
            )

    def test_negative_acknowledged_count_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="acknowledged_count must not be negative"
        ):
            AlertSnapshot(
                session_id="s1",
                total_alerts=0,
                active_count=0,
                acknowledged_count=-1,
                resolved_count=0,
                highest_severity=None,
                alerts=(),
            )

    def test_negative_resolved_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="resolved_count must not be negative"):
            AlertSnapshot(
                session_id="s1",
                total_alerts=0,
                active_count=0,
                acknowledged_count=0,
                resolved_count=-1,
                highest_severity=None,
                alerts=(),
            )

    def test_counts_must_sum_to_total(self) -> None:
        with pytest.raises(ValueError, match="Status counts.*must equal"):
            AlertSnapshot(
                session_id="s1",
                total_alerts=5,
                active_count=1,
                acknowledged_count=1,
                resolved_count=0,
                highest_severity=None,
                alerts=(),
            )

    def test_counts_sum_correctly(self) -> None:
        snapshot = AlertSnapshot(
            session_id="s1",
            total_alerts=3,
            active_count=1,
            acknowledged_count=1,
            resolved_count=1,
            highest_severity=AnomalySeverity.WARNING,
            alerts=(),
        )
        assert snapshot.total_alerts == 3


# ---------------------------------------------------------------------------
# AlertCollectorConfig tests
# ---------------------------------------------------------------------------


class TestAlertCollectorConfig:
    """Verify the AlertCollectorConfig configuration schema."""

    def test_defaults(self) -> None:
        config = AlertCollectorConfig()
        assert config.max_alerts_per_session == 100
        assert config.max_total_alerts == 1000

    def test_custom_values(self) -> None:
        config = AlertCollectorConfig(
            max_alerts_per_session=50,
            max_total_alerts=500,
        )
        assert config.max_alerts_per_session == 50
        assert config.max_total_alerts == 500

    def test_frozen(self) -> None:
        config = AlertCollectorConfig()
        with pytest.raises(AttributeError):
            config.max_alerts_per_session = 200  # type: ignore[misc]

    def test_zero_max_per_session_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="max_alerts_per_session must be positive"
        ):
            AlertCollectorConfig(max_alerts_per_session=0)

    def test_negative_max_per_session_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="max_alerts_per_session must be positive"
        ):
            AlertCollectorConfig(max_alerts_per_session=-5)

    def test_zero_max_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_total_alerts must be positive"):
            AlertCollectorConfig(max_total_alerts=0)

    def test_negative_max_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_total_alerts must be positive"):
            AlertCollectorConfig(max_total_alerts=-10)


# ---------------------------------------------------------------------------
# CollectResult tests
# ---------------------------------------------------------------------------


class TestCollectResult:
    """Verify the immutable CollectResult record."""

    def test_empty_result(self) -> None:
        result = CollectResult(
            new_alerts=(),
            evicted_alerts=(),
            session_id="session-1",
            collected_at=_NOW,
        )
        assert result.new_alerts == ()
        assert result.evicted_alerts == ()
        assert result.session_id == "session-1"
        assert result.collected_at == _NOW
        assert result.alert_count == 0
        assert result.eviction_count == 0
        assert result.has_new_alerts is False
        assert result.has_evictions is False

    def test_result_with_new_alerts(self) -> None:
        alert = _make_alert()
        result = CollectResult(
            new_alerts=(alert,),
            evicted_alerts=(),
            session_id="session-1",
            collected_at=_NOW,
        )
        assert result.alert_count == 1
        assert result.has_new_alerts is True
        assert result.has_evictions is False

    def test_result_with_evictions(self) -> None:
        evicted = _make_alert(alert_id="evicted-1")
        result = CollectResult(
            new_alerts=(),
            evicted_alerts=(evicted,),
            session_id="session-1",
            collected_at=_NOW,
        )
        assert result.eviction_count == 1
        assert result.has_evictions is True

    def test_frozen(self) -> None:
        result = CollectResult(
            new_alerts=(),
            evicted_alerts=(),
            session_id="session-1",
            collected_at=_NOW,
        )
        with pytest.raises(AttributeError):
            result.session_id = "mutated"  # type: ignore[misc]

    def test_empty_session_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="session_id must not be empty"):
            CollectResult(
                new_alerts=(),
                evicted_alerts=(),
                session_id="",
                collected_at=_NOW,
            )
