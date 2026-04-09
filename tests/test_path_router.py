"""Tests for fresh-start vs recovery path router.

Verifies that the path router:
- Takes a RecoveryVerdict and routes to fresh-start or recovery path
- Fresh start: writes a clean idle CurrentRun to the wiki
- Recovery: preserves the interrupted run but updates daemon PID and timestamp
- Returns a structured BootDecision with the chosen path and resulting state
- Handles all verdict scenarios: no record, idle, terminal, running, pending_approval
- Updates the wiki file in both paths
- Is immutable (frozen BootDecision)
- Validates inputs (wiki_root, daemon_pid)
"""

import time
from pathlib import Path

import pytest

from jules_daemon.wiki.interrupted_run import RecoveryVerdict
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    Progress,
    RunStatus,
    SSHTarget,
)
from jules_daemon.wiki.path_router import (
    BootDecision,
    BootPath,
    route_boot,
)
from jules_daemon.wiki import current_run


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


# -- BootPath enum --


class TestBootPath:
    def test_fresh_start_value(self) -> None:
        assert BootPath.FRESH_START.value == "fresh_start"

    def test_recovery_value(self) -> None:
        assert BootPath.RECOVERY.value == "recovery"


# -- BootDecision dataclass --


class TestBootDecision:
    def test_create_fresh_start(self) -> None:
        run = CurrentRun(status=RunStatus.IDLE)
        decision = BootDecision(
            path=BootPath.FRESH_START,
            reason="No prior run -- starting fresh",
            run=run,
            prior_run_id="",
            wiki_file=Path("/tmp/test.md"),
        )
        assert decision.path == BootPath.FRESH_START
        assert decision.run.status == RunStatus.IDLE

    def test_create_recovery(self) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=999)
        run = run.with_running("pytest", remote_pid=888)

        decision = BootDecision(
            path=BootPath.RECOVERY,
            reason="Recovering interrupted run",
            run=run,
            prior_run_id=run.run_id,
            wiki_file=Path("/tmp/test.md"),
        )
        assert decision.path == BootPath.RECOVERY
        assert decision.prior_run_id == run.run_id

    def test_frozen(self) -> None:
        run = CurrentRun(status=RunStatus.IDLE)
        decision = BootDecision(
            path=BootPath.FRESH_START,
            reason="test",
            run=run,
            prior_run_id="",
            wiki_file=Path("/tmp/test.md"),
        )
        with pytest.raises(AttributeError):
            decision.path = BootPath.RECOVERY  # type: ignore[misc]

    def test_is_fresh_start_property(self) -> None:
        run = CurrentRun(status=RunStatus.IDLE)
        decision = BootDecision(
            path=BootPath.FRESH_START,
            reason="test",
            run=run,
            prior_run_id="",
            wiki_file=Path("/tmp/test.md"),
        )
        assert decision.is_fresh_start is True
        assert decision.is_recovery is False

    def test_is_recovery_property(self) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")

        decision = BootDecision(
            path=BootPath.RECOVERY,
            reason="test",
            run=run,
            prior_run_id="abc",
            wiki_file=Path("/tmp/test.md"),
        )
        assert decision.is_recovery is True
        assert decision.is_fresh_start is False


# -- route_boot: Fresh start cases --


class TestRouteBootFreshStartNoRecord:
    """When verdict says no recovery needed and no prior record exists."""

    def _make_no_record_verdict(self) -> RecoveryVerdict:
        return RecoveryVerdict(
            recovery_needed=False,
            reason="No prior run record found",
            interrupted_status=RunStatus.IDLE,
            run_id="",
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=None,
        )

    def test_returns_fresh_start_path(self, wiki_root: Path) -> None:
        verdict = self._make_no_record_verdict()
        decision = route_boot(verdict, None, wiki_root, daemon_pid=1000)
        assert decision.path == BootPath.FRESH_START

    def test_resulting_run_is_idle(self, wiki_root: Path) -> None:
        verdict = self._make_no_record_verdict()
        decision = route_boot(verdict, None, wiki_root, daemon_pid=1000)
        assert decision.run.status == RunStatus.IDLE

    def test_prior_run_id_is_empty(self, wiki_root: Path) -> None:
        verdict = self._make_no_record_verdict()
        decision = route_boot(verdict, None, wiki_root, daemon_pid=1000)
        assert decision.prior_run_id == ""

    def test_wiki_file_is_written(self, wiki_root: Path) -> None:
        verdict = self._make_no_record_verdict()
        decision = route_boot(verdict, None, wiki_root, daemon_pid=1000)
        assert decision.wiki_file.exists()

    def test_wiki_contains_idle_state(self, wiki_root: Path) -> None:
        verdict = self._make_no_record_verdict()
        route_boot(verdict, None, wiki_root, daemon_pid=1000)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.IDLE

    def test_reason_mentions_fresh(self, wiki_root: Path) -> None:
        verdict = self._make_no_record_verdict()
        decision = route_boot(verdict, None, wiki_root, daemon_pid=1000)
        assert "fresh" in decision.reason.lower()


class TestRouteBootFreshStartIdleRecord:
    """When verdict says no recovery needed and prior record was idle."""

    def test_returns_fresh_start(self, wiki_root: Path) -> None:
        idle_run = CurrentRun(status=RunStatus.IDLE)
        current_run.write(wiki_root, idle_run)

        verdict = RecoveryVerdict(
            recovery_needed=False,
            reason="Prior run is idle",
            interrupted_status=RunStatus.IDLE,
            run_id=idle_run.run_id,
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=None,
        )
        decision = route_boot(verdict, idle_run, wiki_root, daemon_pid=2000)
        assert decision.path == BootPath.FRESH_START

    def test_idle_run_creates_new_run_id(self, wiki_root: Path) -> None:
        """Fresh start always generates a new run ID."""
        old_run = CurrentRun(status=RunStatus.IDLE)
        current_run.write(wiki_root, old_run)

        verdict = RecoveryVerdict(
            recovery_needed=False,
            reason="Prior run is idle",
            interrupted_status=RunStatus.IDLE,
            run_id=old_run.run_id,
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=None,
        )
        decision = route_boot(verdict, old_run, wiki_root, daemon_pid=2000)
        # New run should have a different run_id from the old one
        assert decision.run.run_id != old_run.run_id


class TestRouteBootFreshStartTerminalRecord:
    """When verdict says no recovery needed because prior run was terminal."""

    def _make_completed_run(self) -> CurrentRun:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        final = Progress(percent=100.0, tests_passed=10, tests_total=10)
        return run.with_completed(final)

    def _make_failed_run(self) -> CurrentRun:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        error_progress = Progress(tests_failed=1, tests_total=1)
        return run.with_failed("Connection lost", error_progress)

    def _make_cancelled_run(self) -> CurrentRun:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        return run.with_cancelled()

    def test_completed_yields_fresh_start(self, wiki_root: Path) -> None:
        completed = self._make_completed_run()
        current_run.write(wiki_root, completed)

        verdict = RecoveryVerdict(
            recovery_needed=False,
            reason="Prior run already completed",
            interrupted_status=RunStatus.COMPLETED,
            run_id=completed.run_id,
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=1,
        )
        decision = route_boot(verdict, completed, wiki_root, daemon_pid=3000)
        assert decision.path == BootPath.FRESH_START
        assert decision.run.status == RunStatus.IDLE

    def test_failed_yields_fresh_start(self, wiki_root: Path) -> None:
        failed = self._make_failed_run()
        current_run.write(wiki_root, failed)

        verdict = RecoveryVerdict(
            recovery_needed=False,
            reason="Prior run already failed",
            interrupted_status=RunStatus.FAILED,
            run_id=failed.run_id,
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=1,
        )
        decision = route_boot(verdict, failed, wiki_root, daemon_pid=3000)
        assert decision.path == BootPath.FRESH_START

    def test_cancelled_yields_fresh_start(self, wiki_root: Path) -> None:
        cancelled = self._make_cancelled_run()
        current_run.write(wiki_root, cancelled)

        verdict = RecoveryVerdict(
            recovery_needed=False,
            reason="Prior run already cancelled",
            interrupted_status=RunStatus.CANCELLED,
            run_id=cancelled.run_id,
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=1,
        )
        decision = route_boot(verdict, cancelled, wiki_root, daemon_pid=3000)
        assert decision.path == BootPath.FRESH_START

    def test_terminal_prior_run_id_preserved_in_decision(
        self, wiki_root: Path
    ) -> None:
        completed = self._make_completed_run()
        current_run.write(wiki_root, completed)

        verdict = RecoveryVerdict(
            recovery_needed=False,
            reason="Prior run already completed",
            interrupted_status=RunStatus.COMPLETED,
            run_id=completed.run_id,
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=1,
        )
        decision = route_boot(verdict, completed, wiki_root, daemon_pid=3000)
        # Prior run ID is still tracked in the decision for audit purposes
        assert decision.prior_run_id == completed.run_id


# -- route_boot: Recovery cases --


class TestRouteBootRecoveryRunning:
    """When verdict says recovery needed for a RUNNING interrupted run."""

    def _make_running_run(
        self,
        daemon_pid: int = 1234,
        remote_pid: int | None = 5678,
    ) -> CurrentRun:
        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run full regression")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=daemon_pid)
        run = run.with_running("pytest --regression", remote_pid=remote_pid)
        progress = Progress(
            percent=50.0, tests_passed=25, tests_failed=1, tests_total=50,
            last_output_line="FAILED test_checkout",
        )
        return run.with_progress(progress)

    def _make_running_verdict(self, run: CurrentRun) -> RecoveryVerdict:
        return RecoveryVerdict(
            recovery_needed=True,
            reason="Prior run was running when daemon stopped",
            interrupted_status=RunStatus.RUNNING,
            run_id=run.run_id,
            stale_seconds=5.0,
            has_remote_process=run.pids.remote is not None,
            daemon_pid=run.pids.daemon,
        )

    def test_returns_recovery_path(self, wiki_root: Path) -> None:
        run = self._make_running_run()
        current_run.write(wiki_root, run)
        verdict = self._make_running_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)
        assert decision.path == BootPath.RECOVERY

    def test_preserves_run_id(self, wiki_root: Path) -> None:
        run = self._make_running_run()
        current_run.write(wiki_root, run)
        verdict = self._make_running_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)
        assert decision.run.run_id == run.run_id

    def test_preserves_ssh_target(self, wiki_root: Path) -> None:
        run = self._make_running_run()
        current_run.write(wiki_root, run)
        verdict = self._make_running_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)
        assert decision.run.ssh_target is not None
        assert decision.run.ssh_target.host == "prod.example.com"
        assert decision.run.ssh_target.user == "ci"

    def test_preserves_command(self, wiki_root: Path) -> None:
        run = self._make_running_run()
        current_run.write(wiki_root, run)
        verdict = self._make_running_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)
        assert decision.run.command is not None
        assert decision.run.command.natural_language == "run full regression"
        assert decision.run.command.resolved_shell == "pytest --regression"

    def test_preserves_progress(self, wiki_root: Path) -> None:
        run = self._make_running_run()
        current_run.write(wiki_root, run)
        verdict = self._make_running_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)
        assert decision.run.progress.percent == 50.0
        assert decision.run.progress.tests_passed == 25

    def test_updates_daemon_pid(self, wiki_root: Path) -> None:
        run = self._make_running_run(daemon_pid=1234)
        current_run.write(wiki_root, run)
        verdict = self._make_running_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)
        assert decision.run.pids.daemon == 9999

    def test_preserves_remote_pid(self, wiki_root: Path) -> None:
        run = self._make_running_run(remote_pid=5678)
        current_run.write(wiki_root, run)
        verdict = self._make_running_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)
        assert decision.run.pids.remote == 5678

    def test_preserves_status_as_running(self, wiki_root: Path) -> None:
        run = self._make_running_run()
        current_run.write(wiki_root, run)
        verdict = self._make_running_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)
        assert decision.run.status == RunStatus.RUNNING

    def test_updates_wiki_file(self, wiki_root: Path) -> None:
        run = self._make_running_run(daemon_pid=1234)
        current_run.write(wiki_root, run)
        verdict = self._make_running_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)

        # Wiki file should be updated with new daemon PID
        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.pids.daemon == 9999
        assert loaded.status == RunStatus.RUNNING

    def test_prior_run_id_set(self, wiki_root: Path) -> None:
        run = self._make_running_run()
        current_run.write(wiki_root, run)
        verdict = self._make_running_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)
        assert decision.prior_run_id == run.run_id

    def test_reason_mentions_recovery(self, wiki_root: Path) -> None:
        run = self._make_running_run()
        current_run.write(wiki_root, run)
        verdict = self._make_running_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)
        assert "recover" in decision.reason.lower()

    def test_updates_updated_at_timestamp(self, wiki_root: Path) -> None:
        run = self._make_running_run()
        current_run.write(wiki_root, run)
        old_updated = run.updated_at
        verdict = self._make_running_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)
        # updated_at should be refreshed
        assert decision.run.updated_at >= old_updated


class TestRouteBootRecoveryPendingApproval:
    """When verdict says recovery needed for a PENDING_APPROVAL interrupted run."""

    def _make_pending_run(self, daemon_pid: int = 1000) -> CurrentRun:
        target = SSHTarget(host="staging.example.com", user="deploy")
        cmd = Command(natural_language="run smoke tests")
        return CurrentRun().with_pending_approval(target, cmd, daemon_pid=daemon_pid)

    def _make_pending_verdict(self, run: CurrentRun) -> RecoveryVerdict:
        return RecoveryVerdict(
            recovery_needed=True,
            reason="Prior run was pending_approval when daemon stopped",
            interrupted_status=RunStatus.PENDING_APPROVAL,
            run_id=run.run_id,
            stale_seconds=10.0,
            has_remote_process=False,
            daemon_pid=run.pids.daemon,
        )

    def test_returns_recovery_path(self, wiki_root: Path) -> None:
        run = self._make_pending_run()
        current_run.write(wiki_root, run)
        verdict = self._make_pending_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=8888)
        assert decision.path == BootPath.RECOVERY

    def test_preserves_pending_status(self, wiki_root: Path) -> None:
        run = self._make_pending_run()
        current_run.write(wiki_root, run)
        verdict = self._make_pending_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=8888)
        assert decision.run.status == RunStatus.PENDING_APPROVAL

    def test_updates_daemon_pid(self, wiki_root: Path) -> None:
        run = self._make_pending_run(daemon_pid=1000)
        current_run.write(wiki_root, run)
        verdict = self._make_pending_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=8888)
        assert decision.run.pids.daemon == 8888

    def test_preserves_ssh_target(self, wiki_root: Path) -> None:
        run = self._make_pending_run()
        current_run.write(wiki_root, run)
        verdict = self._make_pending_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=8888)
        assert decision.run.ssh_target is not None
        assert decision.run.ssh_target.host == "staging.example.com"

    def test_preserves_command(self, wiki_root: Path) -> None:
        run = self._make_pending_run()
        current_run.write(wiki_root, run)
        verdict = self._make_pending_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=8888)
        assert decision.run.command is not None
        assert decision.run.command.natural_language == "run smoke tests"

    def test_no_remote_pid_for_pending(self, wiki_root: Path) -> None:
        run = self._make_pending_run()
        current_run.write(wiki_root, run)
        verdict = self._make_pending_verdict(run)

        decision = route_boot(verdict, run, wiki_root, daemon_pid=8888)
        assert decision.run.pids.remote is None

    def test_wiki_file_updated(self, wiki_root: Path) -> None:
        run = self._make_pending_run(daemon_pid=1000)
        current_run.write(wiki_root, run)
        verdict = self._make_pending_verdict(run)

        route_boot(verdict, run, wiki_root, daemon_pid=8888)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.pids.daemon == 8888
        assert loaded.status == RunStatus.PENDING_APPROVAL


# -- Input validation --


class TestRouteBootValidation:
    """Validate inputs to route_boot."""

    def test_recovery_without_run_record_raises(self, wiki_root: Path) -> None:
        """Recovery requires the original run record."""
        verdict = RecoveryVerdict(
            recovery_needed=True,
            reason="Prior run was running",
            interrupted_status=RunStatus.RUNNING,
            run_id="abc",
            stale_seconds=5.0,
            has_remote_process=True,
            daemon_pid=1234,
        )
        with pytest.raises(ValueError, match="recovery.*requires.*run record"):
            route_boot(verdict, None, wiki_root, daemon_pid=9999)

    def test_daemon_pid_must_be_positive(self, wiki_root: Path) -> None:
        verdict = RecoveryVerdict(
            recovery_needed=False,
            reason="No prior run",
            interrupted_status=RunStatus.IDLE,
            run_id="",
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=None,
        )
        with pytest.raises(ValueError, match="daemon_pid"):
            route_boot(verdict, None, wiki_root, daemon_pid=0)

    def test_negative_daemon_pid_raises(self, wiki_root: Path) -> None:
        verdict = RecoveryVerdict(
            recovery_needed=False,
            reason="No prior run",
            interrupted_status=RunStatus.IDLE,
            run_id="",
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=None,
        )
        with pytest.raises(ValueError, match="daemon_pid"):
            route_boot(verdict, None, wiki_root, daemon_pid=-1)


# -- Wiki file roundtrip integration --


class TestRouteBootWikiIntegration:
    """End-to-end: verify the wiki file reflects the routing decision."""

    def test_fresh_start_wiki_is_idle(self, wiki_root: Path) -> None:
        verdict = RecoveryVerdict(
            recovery_needed=False,
            reason="No prior run",
            interrupted_status=RunStatus.IDLE,
            run_id="",
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=None,
        )
        decision = route_boot(verdict, None, wiki_root, daemon_pid=5555)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.IDLE
        assert loaded.run_id == decision.run.run_id

    def test_recovery_wiki_preserves_full_context(self, wiki_root: Path) -> None:
        """Recovery path preserves all run context in the wiki."""
        target = SSHTarget(host="prod.example.com", user="ci", port=2222)
        cmd = Command(natural_language="run full regression suite")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1111)
        run = run.with_running("pytest -v --regression", remote_pid=2222)
        progress = Progress(
            percent=75.0,
            tests_passed=30,
            tests_failed=2,
            tests_skipped=3,
            tests_total=40,
            last_output_line="FAILED test_payment_flow",
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)

        verdict = RecoveryVerdict(
            recovery_needed=True,
            reason="Prior run was running",
            interrupted_status=RunStatus.RUNNING,
            run_id=run.run_id,
            stale_seconds=5.0,
            has_remote_process=True,
            daemon_pid=1111,
        )
        route_boot(verdict, run, wiki_root, daemon_pid=7777)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.RUNNING
        assert loaded.run_id == run.run_id
        assert loaded.ssh_target is not None
        assert loaded.ssh_target.host == "prod.example.com"
        assert loaded.ssh_target.port == 2222
        assert loaded.command is not None
        assert loaded.command.natural_language == "run full regression suite"
        assert loaded.command.resolved_shell == "pytest -v --regression"
        assert loaded.pids.daemon == 7777  # Updated
        assert loaded.pids.remote == 2222  # Preserved
        assert loaded.progress.percent == 75.0
        assert loaded.progress.tests_passed == 30
        assert loaded.progress.tests_failed == 2

    def test_fresh_start_overwrites_terminal_record(self, wiki_root: Path) -> None:
        """Fresh start clears any old terminal state."""
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        final = Progress(percent=100.0, tests_passed=10, tests_total=10)
        completed = run.with_completed(final)
        current_run.write(wiki_root, completed)

        verdict = RecoveryVerdict(
            recovery_needed=False,
            reason="Prior run already completed",
            interrupted_status=RunStatus.COMPLETED,
            run_id=completed.run_id,
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=1,
        )
        route_boot(verdict, completed, wiki_root, daemon_pid=4444)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.IDLE
        assert loaded.ssh_target is None
        assert loaded.command is None


# -- Performance --


class TestRouteBootPerformance:
    """Verify routing is fast enough for the 30s recovery SLA."""

    def test_route_completes_under_100ms(self, wiki_root: Path) -> None:
        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run regression")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest", remote_pid=2)
        progress = Progress(percent=50.0, tests_passed=25, tests_total=50)
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)

        verdict = RecoveryVerdict(
            recovery_needed=True,
            reason="Prior run was running",
            interrupted_status=RunStatus.RUNNING,
            run_id=run.run_id,
            stale_seconds=5.0,
            has_remote_process=True,
            daemon_pid=1,
        )

        start = time.monotonic()
        decision = route_boot(verdict, run, wiki_root, daemon_pid=9999)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 100.0, f"Route took {elapsed_ms:.1f}ms (>100ms)"
        assert decision.path == BootPath.RECOVERY
