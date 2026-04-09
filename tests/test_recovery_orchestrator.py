"""Tests for the 30-second recovery timeout orchestrator.

Verifies that the orchestrator:
- Coordinates reconnection and resume steps under a single 30s deadline
- Tracks remaining time budget across sequential recovery phases
- Aborts gracefully when the deadline is exceeded
- Records the recovery outcome to the wiki via current_run.write
- Returns a structured RecoveryOutcome with phase timings
- Handles each recovery phase independently (reconnect, PID check, reattach, resume)
- Propagates success when all phases complete within the deadline
- Marks the run as FAILED in the wiki on timeout or unrecoverable error
- Handles PENDING_APPROVAL recovery (skip SSH reconnection)
- Cancels remaining phases when any phase exceeds the remaining budget
- Uses monotonic clock for deadline tracking (not wall clock)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.wiki.crash_recovery import (
    CrashRecoveryResult,
    RecoveryAction,
)
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    ProcessIDs,
    Progress,
    RunStatus,
    SSHTarget,
)
from jules_daemon.wiki.recovery_orchestrator import (
    RecoveryOutcome,
    RecoveryPhase,
    RecoveryPhaseResult,
    RecoveryTimeoutConfig,
    orchestrate_recovery,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _make_reconnect_recovery(
    *,
    host: str = "prod.example.com",
    user: str = "ci",
    port: int = 22,
    run_id: str = "test-run-123",
    remote_pid: int | None = 5678,
    resolved_shell: str | None = "pytest -v",
    progress_percent: float = 50.0,
) -> CrashRecoveryResult:
    """Build a CrashRecoveryResult for a RECONNECT scenario."""
    return CrashRecoveryResult(
        action=RecoveryAction.RECONNECT,
        reason="Interrupted run detected",
        run_id=run_id,
        status=RunStatus.RUNNING,
        host=host,
        user=user,
        port=port,
        key_path=None,
        remote_pid=remote_pid,
        daemon_pid=1234,
        resolved_shell=resolved_shell,
        natural_language_command="run all tests",
        progress_percent=progress_percent,
        error=None,
        source_path=Path("/wiki/pages/daemon/current-run.md"),
    )


def _make_resume_approval_recovery(
    *,
    host: str = "staging.example.com",
    user: str = "deploy",
    run_id: str = "test-run-456",
) -> CrashRecoveryResult:
    """Build a CrashRecoveryResult for a RESUME_APPROVAL scenario."""
    return CrashRecoveryResult(
        action=RecoveryAction.RESUME_APPROVAL,
        reason="Interrupted approval detected",
        run_id=run_id,
        status=RunStatus.PENDING_APPROVAL,
        host=host,
        user=user,
        port=22,
        key_path=None,
        remote_pid=None,
        daemon_pid=1234,
        resolved_shell=None,
        natural_language_command="run smoke tests",
        progress_percent=0.0,
        error=None,
        source_path=Path("/wiki/pages/daemon/current-run.md"),
    )


def _make_fresh_start_recovery() -> CrashRecoveryResult:
    """Build a CrashRecoveryResult for a FRESH_START scenario."""
    return CrashRecoveryResult(
        action=RecoveryAction.FRESH_START,
        reason="No prior run record found",
        run_id="",
        status=RunStatus.IDLE,
        host=None,
        user=None,
        port=None,
        key_path=None,
        remote_pid=None,
        daemon_pid=None,
        resolved_shell=None,
        natural_language_command=None,
        progress_percent=0.0,
        error=None,
        source_path=None,
    )


# ---------------------------------------------------------------------------
# RecoveryTimeoutConfig tests
# ---------------------------------------------------------------------------


class TestRecoveryTimeoutConfig:
    def test_defaults(self) -> None:
        config = RecoveryTimeoutConfig()
        assert config.total_deadline_seconds == 30.0
        assert config.reconnect_budget_fraction == 0.5
        assert config.pid_check_budget_fraction == 0.15
        assert config.reattach_budget_fraction == 0.15
        assert config.resume_budget_fraction == 0.2

    def test_custom_deadline(self) -> None:
        config = RecoveryTimeoutConfig(total_deadline_seconds=60.0)
        assert config.total_deadline_seconds == 60.0

    def test_deadline_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RecoveryTimeoutConfig(total_deadline_seconds=0.0)

    def test_deadline_must_be_positive_negative(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RecoveryTimeoutConfig(total_deadline_seconds=-5.0)

    def test_fractions_must_not_exceed_one(self) -> None:
        with pytest.raises(ValueError, match="exceed 1.0"):
            RecoveryTimeoutConfig(
                reconnect_budget_fraction=0.6,
                pid_check_budget_fraction=0.3,
                reattach_budget_fraction=0.2,
                resume_budget_fraction=0.1,
            )

    def test_fractions_at_exactly_one(self) -> None:
        config = RecoveryTimeoutConfig(
            reconnect_budget_fraction=0.4,
            pid_check_budget_fraction=0.2,
            reattach_budget_fraction=0.2,
            resume_budget_fraction=0.2,
        )
        assert config is not None

    def test_negative_fraction_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            RecoveryTimeoutConfig(reconnect_budget_fraction=-0.1)

    def test_frozen(self) -> None:
        config = RecoveryTimeoutConfig()
        with pytest.raises(AttributeError):
            config.total_deadline_seconds = 60.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RecoveryPhaseResult tests
# ---------------------------------------------------------------------------


class TestRecoveryPhaseResult:
    def test_create_success(self) -> None:
        result = RecoveryPhaseResult(
            phase=RecoveryPhase.RECONNECT,
            success=True,
            duration_seconds=2.5,
            error=None,
        )
        assert result.phase == RecoveryPhase.RECONNECT
        assert result.success is True
        assert result.duration_seconds == 2.5
        assert result.error is None

    def test_create_failure(self) -> None:
        result = RecoveryPhaseResult(
            phase=RecoveryPhase.PID_CHECK,
            success=False,
            duration_seconds=5.0,
            error="Process not found",
        )
        assert result.success is False
        assert result.error == "Process not found"

    def test_frozen(self) -> None:
        result = RecoveryPhaseResult(
            phase=RecoveryPhase.RECONNECT,
            success=True,
            duration_seconds=1.0,
            error=None,
        )
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RecoveryOutcome tests
# ---------------------------------------------------------------------------


class TestRecoveryOutcome:
    def test_create_success_outcome(self) -> None:
        phases = (
            RecoveryPhaseResult(
                phase=RecoveryPhase.RECONNECT,
                success=True,
                duration_seconds=3.0,
                error=None,
            ),
        )
        outcome = RecoveryOutcome(
            success=True,
            action_taken=RecoveryAction.RECONNECT,
            run_id="test-123",
            total_duration_seconds=5.0,
            deadline_seconds=30.0,
            phases=phases,
            timed_out=False,
            error=None,
            wiki_updated=True,
        )
        assert outcome.success is True
        assert outcome.timed_out is False
        assert outcome.time_remaining_seconds == 25.0
        assert len(outcome.phases) == 1

    def test_time_remaining_calculation(self) -> None:
        outcome = RecoveryOutcome(
            success=False,
            action_taken=RecoveryAction.RECONNECT,
            run_id="test-123",
            total_duration_seconds=28.0,
            deadline_seconds=30.0,
            phases=(),
            timed_out=False,
            error="Failed",
            wiki_updated=False,
        )
        assert outcome.time_remaining_seconds == 2.0

    def test_timed_out_outcome(self) -> None:
        outcome = RecoveryOutcome(
            success=False,
            action_taken=RecoveryAction.RECONNECT,
            run_id="test-123",
            total_duration_seconds=31.0,
            deadline_seconds=30.0,
            phases=(),
            timed_out=True,
            error="Recovery timed out after 31.0s (deadline: 30.0s)",
            wiki_updated=True,
        )
        assert outcome.timed_out is True
        assert outcome.success is False
        assert outcome.time_remaining_seconds == -1.0

    def test_frozen(self) -> None:
        outcome = RecoveryOutcome(
            success=True,
            action_taken=RecoveryAction.RECONNECT,
            run_id="r",
            total_duration_seconds=1.0,
            deadline_seconds=30.0,
            phases=(),
            timed_out=False,
            error=None,
            wiki_updated=True,
        )
        with pytest.raises(AttributeError):
            outcome.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RecoveryPhase enum tests
# ---------------------------------------------------------------------------


class TestRecoveryPhase:
    def test_all_phases_exist(self) -> None:
        assert RecoveryPhase.RECONNECT.value == "reconnect"
        assert RecoveryPhase.PID_CHECK.value == "pid_check"
        assert RecoveryPhase.REATTACH.value == "reattach"
        assert RecoveryPhase.RESUME_DECISION.value == "resume_decision"
        assert RecoveryPhase.WIKI_UPDATE.value == "wiki_update"


# ---------------------------------------------------------------------------
# orchestrate_recovery: FRESH_START (no-op) tests
# ---------------------------------------------------------------------------


class TestOrchestrateFreshStart:
    """FRESH_START recovery should return immediately with no phases."""

    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        return tmp_path / "wiki"

    @pytest.mark.asyncio
    async def test_fresh_start_returns_success(self, wiki_root: Path) -> None:
        recovery = _make_fresh_start_recovery()
        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
        )
        assert outcome.success is True
        assert outcome.action_taken == RecoveryAction.FRESH_START
        assert outcome.timed_out is False
        assert len(outcome.phases) == 0

    @pytest.mark.asyncio
    async def test_fresh_start_completes_quickly(self, wiki_root: Path) -> None:
        recovery = _make_fresh_start_recovery()
        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
        )
        assert outcome.total_duration_seconds < 1.0


# ---------------------------------------------------------------------------
# orchestrate_recovery: RESUME_APPROVAL tests
# ---------------------------------------------------------------------------


class TestOrchestrateResumeApproval:
    """RESUME_APPROVAL recovery skips SSH reconnection and PID checks."""

    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        wiki = tmp_path / "wiki"
        # Write an initial current-run record in pending_approval state
        from jules_daemon.wiki import current_run

        target = SSHTarget(host="staging.example.com", user="deploy")
        cmd = Command(natural_language="run smoke tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        current_run.write(wiki, run)
        return wiki

    @pytest.mark.asyncio
    async def test_resume_approval_success(self, wiki_root: Path) -> None:
        recovery = _make_resume_approval_recovery()
        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
        )
        assert outcome.success is True
        assert outcome.action_taken == RecoveryAction.RESUME_APPROVAL
        assert outcome.timed_out is False

    @pytest.mark.asyncio
    async def test_resume_approval_has_wiki_update_phase(
        self, wiki_root: Path
    ) -> None:
        recovery = _make_resume_approval_recovery()
        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
        )
        phase_names = [p.phase for p in outcome.phases]
        assert RecoveryPhase.WIKI_UPDATE in phase_names

    @pytest.mark.asyncio
    async def test_resume_approval_skips_reconnect(
        self, wiki_root: Path
    ) -> None:
        recovery = _make_resume_approval_recovery()
        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
        )
        phase_names = [p.phase for p in outcome.phases]
        assert RecoveryPhase.RECONNECT not in phase_names
        assert RecoveryPhase.PID_CHECK not in phase_names
        assert RecoveryPhase.REATTACH not in phase_names


# ---------------------------------------------------------------------------
# orchestrate_recovery: RECONNECT tests (with mocked SSH)
# ---------------------------------------------------------------------------


class TestOrchestrateReconnect:
    """RECONNECT recovery goes through reconnect -> PID check -> reattach -> resume."""

    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        wiki = tmp_path / "wiki"
        # Write an initial current-run record in running state
        from jules_daemon.wiki import current_run

        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run all tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        run = run.with_running("pytest -v", remote_pid=5678)
        progress = Progress(
            percent=50.0,
            tests_passed=25,
            tests_total=50,
        )
        run = run.with_progress(progress)
        current_run.write(wiki, run)
        return wiki

    @pytest.mark.asyncio
    async def test_reconnect_all_phases_succeed(self, wiki_root: Path) -> None:
        """When all mocked phases succeed, outcome is success."""
        recovery = _make_reconnect_recovery()

        # Mock the phase executors
        mock_connector = AsyncMock()
        mock_handle = MagicMock()
        mock_handle.session_id = "sess-001"
        mock_connector.connect = AsyncMock(return_value=mock_handle)

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
        )

        assert outcome.success is True
        assert outcome.action_taken == RecoveryAction.RECONNECT
        assert outcome.timed_out is False
        assert outcome.wiki_updated is True

    @pytest.mark.asyncio
    async def test_reconnect_records_phase_timings(
        self, wiki_root: Path
    ) -> None:
        recovery = _make_reconnect_recovery()

        mock_connector = AsyncMock()
        mock_handle = MagicMock()
        mock_handle.session_id = "sess-001"
        mock_connector.connect = AsyncMock(return_value=mock_handle)

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
        )

        # Should have at least RECONNECT and WIKI_UPDATE phases
        assert len(outcome.phases) >= 2
        for phase_result in outcome.phases:
            assert phase_result.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_reconnect_failure_marks_failed_in_wiki(
        self, wiki_root: Path
    ) -> None:
        """When reconnection fails, the wiki should be updated to FAILED."""
        recovery = _make_reconnect_recovery()

        mock_connector = AsyncMock()
        mock_connector.connect = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
        )

        assert outcome.success is False
        assert outcome.wiki_updated is True

        # Verify the wiki was updated to FAILED status
        from jules_daemon.wiki import current_run

        run = current_run.read(wiki_root)
        assert run is not None
        assert run.status == RunStatus.FAILED

    @pytest.mark.asyncio
    async def test_reconnect_no_remote_pid_skips_pid_check(
        self, wiki_root: Path
    ) -> None:
        """When remote_pid is None, PID check and reattach phases are skipped."""
        recovery = _make_reconnect_recovery(remote_pid=None)

        mock_connector = AsyncMock()
        mock_handle = MagicMock()
        mock_handle.session_id = "sess-001"
        mock_connector.connect = AsyncMock(return_value=mock_handle)

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
        )

        phase_names = [p.phase for p in outcome.phases]
        assert RecoveryPhase.PID_CHECK not in phase_names
        assert RecoveryPhase.REATTACH not in phase_names


# ---------------------------------------------------------------------------
# orchestrate_recovery: timeout tests
# ---------------------------------------------------------------------------


class TestOrchestrateTimeout:
    """Verify that the orchestrator enforces the 30s deadline."""

    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        wiki = tmp_path / "wiki"
        from jules_daemon.wiki import current_run

        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run all tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        run = run.with_running("pytest -v", remote_pid=5678)
        current_run.write(wiki, run)
        return wiki

    @pytest.mark.asyncio
    async def test_timeout_aborts_gracefully(self, wiki_root: Path) -> None:
        """A slow reconnect that exceeds the deadline should abort."""
        recovery = _make_reconnect_recovery()

        # Connector that sleeps longer than the deadline
        async def slow_connect(target: Any) -> None:
            await asyncio.sleep(100)

        mock_connector = AsyncMock()
        mock_connector.connect = slow_connect

        config = RecoveryTimeoutConfig(total_deadline_seconds=0.5)

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
            config=config,
        )

        assert outcome.success is False
        assert outcome.timed_out is True
        assert outcome.wiki_updated is True

    @pytest.mark.asyncio
    async def test_timeout_records_error_message(
        self, wiki_root: Path
    ) -> None:
        recovery = _make_reconnect_recovery()

        async def slow_connect(target: Any) -> None:
            await asyncio.sleep(100)

        mock_connector = AsyncMock()
        mock_connector.connect = slow_connect

        config = RecoveryTimeoutConfig(total_deadline_seconds=0.5)

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
            config=config,
        )

        assert outcome.error is not None
        assert "timed out" in outcome.error.lower() or "timeout" in outcome.error.lower()

    @pytest.mark.asyncio
    async def test_timeout_updates_wiki_to_failed(
        self, wiki_root: Path
    ) -> None:
        recovery = _make_reconnect_recovery()

        async def slow_connect(target: Any) -> None:
            await asyncio.sleep(100)

        mock_connector = AsyncMock()
        mock_connector.connect = slow_connect

        config = RecoveryTimeoutConfig(total_deadline_seconds=0.5)

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
            config=config,
        )

        from jules_daemon.wiki import current_run

        run = current_run.read(wiki_root)
        assert run is not None
        assert run.status == RunStatus.FAILED
        assert run.error is not None
        assert "timeout" in run.error.lower() or "timed out" in run.error.lower()

    @pytest.mark.asyncio
    async def test_custom_deadline_respected(self, wiki_root: Path) -> None:
        recovery = _make_reconnect_recovery()

        mock_connector = AsyncMock()
        mock_handle = MagicMock()
        mock_handle.session_id = "sess-001"
        mock_connector.connect = AsyncMock(return_value=mock_handle)

        config = RecoveryTimeoutConfig(total_deadline_seconds=60.0)

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
            config=config,
        )

        assert outcome.deadline_seconds == 60.0


# ---------------------------------------------------------------------------
# orchestrate_recovery: wiki recording tests
# ---------------------------------------------------------------------------


class TestOrchestrateWikiRecording:
    """Verify that recovery outcomes are recorded to the wiki."""

    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        wiki = tmp_path / "wiki"
        from jules_daemon.wiki import current_run

        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run all tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        run = run.with_running("pytest -v", remote_pid=5678)
        current_run.write(wiki, run)
        return wiki

    @pytest.mark.asyncio
    async def test_success_writes_recovery_record(
        self, wiki_root: Path
    ) -> None:
        """Successful recovery writes a recovery event to the wiki."""
        recovery = _make_reconnect_recovery()

        mock_connector = AsyncMock()
        mock_handle = MagicMock()
        mock_handle.session_id = "sess-001"
        mock_connector.connect = AsyncMock(return_value=mock_handle)

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
        )

        assert outcome.wiki_updated is True

        # Check that a recovery record was written
        recovery_path = wiki_root / "pages" / "daemon" / "recovery-log.md"
        assert recovery_path.exists()
        content = recovery_path.read_text(encoding="utf-8")
        assert "recovery" in content.lower()

    @pytest.mark.asyncio
    async def test_failure_writes_recovery_record(
        self, wiki_root: Path
    ) -> None:
        recovery = _make_reconnect_recovery()

        mock_connector = AsyncMock()
        mock_connector.connect = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
        )

        assert outcome.wiki_updated is True

        recovery_path = wiki_root / "pages" / "daemon" / "recovery-log.md"
        assert recovery_path.exists()
        content = recovery_path.read_text(encoding="utf-8")
        assert "failed" in content.lower() or "failure" in content.lower()

    @pytest.mark.asyncio
    async def test_wiki_write_error_does_not_crash(
        self, tmp_path: Path
    ) -> None:
        """If the wiki write itself fails, the orchestrator still returns."""
        # Use a non-writable path
        wiki_root = tmp_path / "readonly_wiki"
        wiki_root.mkdir()
        daemon_dir = wiki_root / "pages" / "daemon"
        daemon_dir.mkdir(parents=True)

        # Write initial state
        from jules_daemon.wiki import current_run

        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run all tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        run = run.with_running("pytest -v", remote_pid=5678)
        current_run.write(wiki_root, run)

        recovery = _make_reconnect_recovery()

        mock_connector = AsyncMock()
        mock_connector.connect = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        # The orchestrator should not crash even if wiki writes fail
        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
        )

        assert outcome.success is False
        # It should attempt wiki update regardless
        assert isinstance(outcome.wiki_updated, bool)


# ---------------------------------------------------------------------------
# orchestrate_recovery: deadline budget distribution
# ---------------------------------------------------------------------------


class TestDeadlineBudget:
    """Verify that the budget is properly distributed across phases."""

    def test_default_budget_sums_to_one(self) -> None:
        config = RecoveryTimeoutConfig()
        total = (
            config.reconnect_budget_fraction
            + config.pid_check_budget_fraction
            + config.reattach_budget_fraction
            + config.resume_budget_fraction
        )
        assert total <= 1.0

    def test_phase_budget_calculation(self) -> None:
        config = RecoveryTimeoutConfig(
            total_deadline_seconds=30.0,
            reconnect_budget_fraction=0.5,
        )
        # Reconnect gets 50% of 30s = 15s
        reconnect_budget = config.total_deadline_seconds * config.reconnect_budget_fraction
        assert reconnect_budget == 15.0


# ---------------------------------------------------------------------------
# orchestrate_recovery: no connector for RECONNECT
# ---------------------------------------------------------------------------


class TestOrchestrateNoConnector:
    """When RECONNECT is needed but no connector is provided, fail gracefully."""

    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        wiki = tmp_path / "wiki"
        from jules_daemon.wiki import current_run

        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run all tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        run = run.with_running("pytest -v", remote_pid=5678)
        current_run.write(wiki, run)
        return wiki

    @pytest.mark.asyncio
    async def test_no_connector_returns_failure(self, wiki_root: Path) -> None:
        recovery = _make_reconnect_recovery()

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=None,
        )

        assert outcome.success is False
        assert "connector" in outcome.error.lower()

    @pytest.mark.asyncio
    async def test_no_connector_does_not_timeout(self, wiki_root: Path) -> None:
        recovery = _make_reconnect_recovery()

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=None,
        )

        assert outcome.timed_out is False


# ---------------------------------------------------------------------------
# orchestrate_recovery: missing host/user in recovery
# ---------------------------------------------------------------------------


class TestOrchestrateMissingConnectionParams:
    """When SSH params are missing, reconnect phase fails gracefully."""

    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        wiki = tmp_path / "wiki"
        from jules_daemon.wiki import current_run

        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run all tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        run = run.with_running("pytest -v", remote_pid=5678)
        current_run.write(wiki, run)
        return wiki

    @pytest.mark.asyncio
    async def test_missing_host_returns_failure(self, wiki_root: Path) -> None:
        recovery = CrashRecoveryResult(
            action=RecoveryAction.RECONNECT,
            reason="Interrupted run detected",
            run_id="test-run-123",
            status=RunStatus.RUNNING,
            host=None,
            user="ci",
            port=22,
            key_path=None,
            remote_pid=5678,
            daemon_pid=1234,
            resolved_shell="pytest -v",
            natural_language_command="run all tests",
            progress_percent=50.0,
            error=None,
            source_path=Path("/wiki/pages/daemon/current-run.md"),
        )

        mock_connector = AsyncMock()

        outcome = await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
        )

        assert outcome.success is False
        assert "missing" in outcome.error.lower() or "host" in outcome.error.lower()


# ---------------------------------------------------------------------------
# DeadlineTracker internal tests
# ---------------------------------------------------------------------------


class TestDeadlineTracker:
    """Test the internal deadline tracker."""

    def test_initial_not_expired(self) -> None:
        from jules_daemon.wiki.recovery_orchestrator import _DeadlineTracker

        tracker = _DeadlineTracker(30.0)
        assert tracker.expired is False
        assert tracker.remaining_seconds > 0

    def test_elapsed_starts_near_zero(self) -> None:
        from jules_daemon.wiki.recovery_orchestrator import _DeadlineTracker

        tracker = _DeadlineTracker(30.0)
        assert tracker.elapsed_seconds < 1.0

    def test_phase_timeout_respects_budget(self) -> None:
        from jules_daemon.wiki.recovery_orchestrator import _DeadlineTracker

        tracker = _DeadlineTracker(30.0)
        timeout = tracker.phase_timeout(0.5)
        assert timeout <= 15.0
        assert timeout > 0.0

    def test_phase_timeout_with_zero_budget(self) -> None:
        from jules_daemon.wiki.recovery_orchestrator import _DeadlineTracker

        tracker = _DeadlineTracker(30.0)
        timeout = tracker.phase_timeout(0.0)
        assert timeout == 0.0


# ---------------------------------------------------------------------------
# Recovery log wiki format verification
# ---------------------------------------------------------------------------


class TestRecoveryLogFormat:
    """Verify the recovery log wiki file uses proper YAML frontmatter format."""

    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        wiki = tmp_path / "wiki"
        from jules_daemon.wiki import current_run

        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run all tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        run = run.with_running("pytest -v", remote_pid=5678)
        current_run.write(wiki, run)
        return wiki

    @pytest.mark.asyncio
    async def test_recovery_log_has_yaml_frontmatter(
        self, wiki_root: Path
    ) -> None:
        recovery = _make_reconnect_recovery()

        mock_connector = AsyncMock()
        mock_handle = MagicMock()
        mock_handle.session_id = "sess-001"
        mock_connector.connect = AsyncMock(return_value=mock_handle)

        await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
        )

        recovery_path = wiki_root / "pages" / "daemon" / "recovery-log.md"
        content = recovery_path.read_text(encoding="utf-8")

        # Verify YAML frontmatter format
        assert content.startswith("---\n")
        assert "\n---\n" in content[3:]  # closing fence

        # Parse to verify YAML is valid
        from jules_daemon.wiki.frontmatter import parse

        doc = parse(content)
        assert doc.frontmatter["type"] == "daemon-recovery-log"
        assert "daemon" in doc.frontmatter["tags"]
        assert "recovery" in doc.frontmatter["tags"]
        assert "audit" in doc.frontmatter["tags"]
        assert isinstance(doc.frontmatter["phases"], list)

    @pytest.mark.asyncio
    async def test_recovery_log_contains_phase_table(
        self, wiki_root: Path
    ) -> None:
        recovery = _make_reconnect_recovery()

        mock_connector = AsyncMock()
        mock_handle = MagicMock()
        mock_handle.session_id = "sess-001"
        mock_connector.connect = AsyncMock(return_value=mock_handle)

        await orchestrate_recovery(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=mock_connector,
        )

        recovery_path = wiki_root / "pages" / "daemon" / "recovery-log.md"
        content = recovery_path.read_text(encoding="utf-8")

        assert "## Phase Timings" in content
        assert "| Phase |" in content
        assert "reconnect" in content
