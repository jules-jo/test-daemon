"""Tests for the staleness guard logic.

Covers:
- Fresh status within the 10s threshold
- Stale status exceeding the 10s threshold
- Exact boundary (10s)
- Terminal status bypass (completed runs are not checked for staleness)
- Custom threshold override
- Guard applied to MonitorStatus
- Guard applied to CurrentRun
- StaleStatusError contains diagnostic details
- validate_freshness returns a FreshnessResult without raising
"""

from datetime import datetime, timedelta, timezone

import pytest

from jules_daemon.wiki.monitor_status import (
    MonitorStatus,
    OutputPhase,
    ParsedState,
)
from jules_daemon.wiki.models import CurrentRun, RunStatus
from jules_daemon.wiki.staleness_guard import (
    DEFAULT_STALENESS_THRESHOLD_SECONDS,
    FreshnessResult,
    StaleStatusError,
    validate_monitor_freshness,
    validate_run_freshness,
    require_fresh_monitor_status,
    require_fresh_run_status,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _make_monitor_status(
    timestamp: datetime,
    session_id: str = "test-session",
    phase: OutputPhase = OutputPhase.RUNNING,
) -> MonitorStatus:
    return MonitorStatus(
        session_id=session_id,
        timestamp=timestamp,
        parsed_state=ParsedState(phase=phase),
    )


# -- Default threshold constant --


class TestDefaultThreshold:
    def test_default_threshold_is_ten_seconds(self) -> None:
        assert DEFAULT_STALENESS_THRESHOLD_SECONDS == 10.0


# -- FreshnessResult --


class TestFreshnessResult:
    def test_fresh_result(self) -> None:
        result = FreshnessResult(
            is_fresh=True,
            age_seconds=5.0,
            threshold_seconds=10.0,
        )
        assert result.is_fresh is True
        assert result.age_seconds == 5.0
        assert result.threshold_seconds == 10.0

    def test_stale_result(self) -> None:
        result = FreshnessResult(
            is_fresh=False,
            age_seconds=15.0,
            threshold_seconds=10.0,
        )
        assert result.is_fresh is False
        assert result.age_seconds == 15.0

    def test_frozen(self) -> None:
        result = FreshnessResult(
            is_fresh=True,
            age_seconds=3.0,
            threshold_seconds=10.0,
        )
        with pytest.raises(AttributeError):
            result.is_fresh = False  # type: ignore[misc]

    def test_margin_property_when_fresh(self) -> None:
        result = FreshnessResult(
            is_fresh=True,
            age_seconds=3.0,
            threshold_seconds=10.0,
        )
        assert result.margin_seconds == pytest.approx(7.0)

    def test_margin_property_when_stale(self) -> None:
        result = FreshnessResult(
            is_fresh=False,
            age_seconds=15.0,
            threshold_seconds=10.0,
        )
        assert result.margin_seconds == pytest.approx(-5.0)


# -- validate_monitor_freshness (non-raising) --


class TestValidateMonitorFreshness:
    def test_fresh_status_within_threshold(self) -> None:
        now = _utc_now()
        status = _make_monitor_status(timestamp=now - timedelta(seconds=3))
        result = validate_monitor_freshness(status, now=now)
        assert result.is_fresh is True
        assert result.age_seconds == pytest.approx(3.0, abs=0.1)

    def test_stale_status_exceeds_threshold(self) -> None:
        now = _utc_now()
        status = _make_monitor_status(timestamp=now - timedelta(seconds=15))
        result = validate_monitor_freshness(status, now=now)
        assert result.is_fresh is False
        assert result.age_seconds == pytest.approx(15.0, abs=0.1)

    def test_exact_boundary_is_fresh(self) -> None:
        """Exactly 10s old is still considered fresh (<=, not <)."""
        now = _utc_now()
        status = _make_monitor_status(timestamp=now - timedelta(seconds=10))
        result = validate_monitor_freshness(status, now=now)
        assert result.is_fresh is True

    def test_just_over_boundary_is_stale(self) -> None:
        now = _utc_now()
        status = _make_monitor_status(
            timestamp=now - timedelta(seconds=10, milliseconds=1)
        )
        result = validate_monitor_freshness(status, now=now)
        assert result.is_fresh is False

    def test_zero_age_is_fresh(self) -> None:
        now = _utc_now()
        status = _make_monitor_status(timestamp=now)
        result = validate_monitor_freshness(status, now=now)
        assert result.is_fresh is True
        assert result.age_seconds == pytest.approx(0.0, abs=0.01)

    def test_custom_threshold(self) -> None:
        now = _utc_now()
        status = _make_monitor_status(timestamp=now - timedelta(seconds=3))
        result = validate_monitor_freshness(
            status, now=now, threshold_seconds=2.0
        )
        assert result.is_fresh is False
        assert result.threshold_seconds == 2.0

    def test_terminal_status_always_fresh(self) -> None:
        """A terminal (exited) status is always considered fresh."""
        now = _utc_now()
        base = _make_monitor_status(
            timestamp=now - timedelta(seconds=60),
            phase=OutputPhase.COMPLETE,
        )
        terminal = base.with_exit(
            timestamp=now - timedelta(seconds=60),
            exit_status=0,
        )
        result = validate_monitor_freshness(terminal, now=now)
        assert result.is_fresh is True

    def test_uses_current_time_when_now_omitted(self) -> None:
        """When now is not provided, the guard uses the current clock."""
        status = _make_monitor_status(timestamp=_utc_now())
        result = validate_monitor_freshness(status)
        assert result.is_fresh is True

    def test_negative_threshold_raises_value_error(self) -> None:
        now = _utc_now()
        status = _make_monitor_status(timestamp=now)
        with pytest.raises(ValueError, match="threshold_seconds must be positive"):
            validate_monitor_freshness(status, now=now, threshold_seconds=-1.0)

    def test_zero_threshold_raises_value_error(self) -> None:
        now = _utc_now()
        status = _make_monitor_status(timestamp=now)
        with pytest.raises(ValueError, match="threshold_seconds must be positive"):
            validate_monitor_freshness(status, now=now, threshold_seconds=0.0)


# -- validate_run_freshness (non-raising) --


class TestValidateRunFreshness:
    def test_fresh_run_within_threshold(self) -> None:
        now = _utc_now()
        run = CurrentRun(
            status=RunStatus.RUNNING,
            updated_at=now - timedelta(seconds=2),
        )
        result = validate_run_freshness(run, now=now)
        assert result.is_fresh is True
        assert result.age_seconds == pytest.approx(2.0, abs=0.1)

    def test_stale_run_exceeds_threshold(self) -> None:
        now = _utc_now()
        run = CurrentRun(
            status=RunStatus.RUNNING,
            updated_at=now - timedelta(seconds=20),
        )
        result = validate_run_freshness(run, now=now)
        assert result.is_fresh is False
        assert result.age_seconds == pytest.approx(20.0, abs=0.1)

    def test_idle_run_always_fresh(self) -> None:
        """An idle run has no active monitoring -- treat as fresh."""
        now = _utc_now()
        run = CurrentRun(
            status=RunStatus.IDLE,
            updated_at=now - timedelta(seconds=999),
        )
        result = validate_run_freshness(run, now=now)
        assert result.is_fresh is True

    def test_completed_run_always_fresh(self) -> None:
        """A terminal run does not need freshness checks."""
        now = _utc_now()
        run = CurrentRun(
            status=RunStatus.COMPLETED,
            updated_at=now - timedelta(seconds=999),
        )
        result = validate_run_freshness(run, now=now)
        assert result.is_fresh is True

    def test_failed_run_always_fresh(self) -> None:
        now = _utc_now()
        run = CurrentRun(
            status=RunStatus.FAILED,
            updated_at=now - timedelta(seconds=999),
        )
        result = validate_run_freshness(run, now=now)
        assert result.is_fresh is True

    def test_cancelled_run_always_fresh(self) -> None:
        now = _utc_now()
        run = CurrentRun(
            status=RunStatus.CANCELLED,
            updated_at=now - timedelta(seconds=999),
        )
        result = validate_run_freshness(run, now=now)
        assert result.is_fresh is True

    def test_pending_approval_checked_for_staleness(self) -> None:
        """pending_approval is an active state, so staleness is checked."""
        now = _utc_now()
        run = CurrentRun(
            status=RunStatus.PENDING_APPROVAL,
            updated_at=now - timedelta(seconds=15),
        )
        result = validate_run_freshness(run, now=now)
        assert result.is_fresh is False

    def test_custom_threshold_for_run(self) -> None:
        now = _utc_now()
        run = CurrentRun(
            status=RunStatus.RUNNING,
            updated_at=now - timedelta(seconds=5),
        )
        result = validate_run_freshness(run, now=now, threshold_seconds=3.0)
        assert result.is_fresh is False


# -- require_fresh_monitor_status (raising) --


class TestRequireFreshMonitorStatus:
    def test_returns_status_when_fresh(self) -> None:
        now = _utc_now()
        status = _make_monitor_status(timestamp=now - timedelta(seconds=1))
        returned = require_fresh_monitor_status(status, now=now)
        assert returned is status

    def test_raises_stale_status_error_when_stale(self) -> None:
        now = _utc_now()
        status = _make_monitor_status(timestamp=now - timedelta(seconds=15))
        with pytest.raises(StaleStatusError) as exc_info:
            require_fresh_monitor_status(status, now=now)
        err = exc_info.value
        assert err.age_seconds == pytest.approx(15.0, abs=0.1)
        assert err.threshold_seconds == 10.0
        assert err.session_id == "test-session"

    def test_stale_error_message_includes_diagnostics(self) -> None:
        now = _utc_now()
        status = _make_monitor_status(timestamp=now - timedelta(seconds=25))
        with pytest.raises(StaleStatusError, match=r"25\.0.*seconds.*stale"):
            require_fresh_monitor_status(status, now=now)

    def test_terminal_status_never_raises(self) -> None:
        now = _utc_now()
        base = _make_monitor_status(
            timestamp=now - timedelta(seconds=60),
            phase=OutputPhase.COMPLETE,
        )
        terminal = base.with_exit(
            timestamp=now - timedelta(seconds=60),
            exit_status=1,
        )
        returned = require_fresh_monitor_status(terminal, now=now)
        assert returned is terminal

    def test_custom_threshold_raises(self) -> None:
        now = _utc_now()
        status = _make_monitor_status(timestamp=now - timedelta(seconds=4))
        with pytest.raises(StaleStatusError):
            require_fresh_monitor_status(
                status, now=now, threshold_seconds=3.0
            )


# -- require_fresh_run_status (raising) --


class TestRequireFreshRunStatus:
    def test_returns_run_when_fresh(self) -> None:
        now = _utc_now()
        run = CurrentRun(
            status=RunStatus.RUNNING,
            updated_at=now - timedelta(seconds=1),
        )
        returned = require_fresh_run_status(run, now=now)
        assert returned is run

    def test_raises_stale_status_error_when_stale(self) -> None:
        now = _utc_now()
        run = CurrentRun(
            status=RunStatus.RUNNING,
            updated_at=now - timedelta(seconds=20),
        )
        with pytest.raises(StaleStatusError) as exc_info:
            require_fresh_run_status(run, now=now)
        err = exc_info.value
        assert err.age_seconds == pytest.approx(20.0, abs=0.1)
        assert err.threshold_seconds == 10.0

    def test_idle_run_never_raises(self) -> None:
        now = _utc_now()
        run = CurrentRun(
            status=RunStatus.IDLE,
            updated_at=now - timedelta(seconds=999),
        )
        returned = require_fresh_run_status(run, now=now)
        assert returned is run

    def test_terminal_run_never_raises(self) -> None:
        now = _utc_now()
        run = CurrentRun(
            status=RunStatus.COMPLETED,
            updated_at=now - timedelta(seconds=999),
        )
        returned = require_fresh_run_status(run, now=now)
        assert returned is run


# -- StaleStatusError attributes --


class TestStaleStatusError:
    def test_is_exception(self) -> None:
        err = StaleStatusError(
            age_seconds=15.0,
            threshold_seconds=10.0,
            session_id="sess-1",
        )
        assert isinstance(err, Exception)

    def test_attributes(self) -> None:
        err = StaleStatusError(
            age_seconds=42.5,
            threshold_seconds=10.0,
            session_id="sess-abc",
        )
        assert err.age_seconds == 42.5
        assert err.threshold_seconds == 10.0
        assert err.session_id == "sess-abc"

    def test_str_representation(self) -> None:
        err = StaleStatusError(
            age_seconds=15.3,
            threshold_seconds=10.0,
            session_id="sess-x",
        )
        msg = str(err)
        assert "15.3" in msg
        assert "stale" in msg.lower() or "Stale" in msg

    def test_optional_session_id(self) -> None:
        err = StaleStatusError(
            age_seconds=20.0,
            threshold_seconds=10.0,
        )
        assert err.session_id is None
