"""Tests for crash recovery detection at daemon startup.

Verifies the unified crash recovery orchestrator that:
- Reads the current-run wiki record at daemon startup
- Determines if an incomplete run exists (RUNNING or PENDING_APPROVAL)
- Extracts connection details (host, user, remote PID, run metadata)
- Handles missing wiki files gracefully (no crash to recover from)
- Handles corrupted wiki files gracefully (safe degradation)
- Combines boot reading, interruption detection, and state extraction
  into a single CrashRecoveryResult
- Completes well within the 30s recovery SLA
"""

import time
from pathlib import Path

import pytest

from jules_daemon.wiki import current_run
from jules_daemon.wiki.crash_recovery import (
    CrashRecoveryResult,
    RecoveryAction,
    detect_crash_recovery,
)
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    Progress,
    RunStatus,
    SSHTarget,
)


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


# -- CrashRecoveryResult dataclass --


class TestCrashRecoveryResult:
    def test_frozen(self) -> None:
        result = CrashRecoveryResult(
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
        with pytest.raises(AttributeError):
            result.action = RecoveryAction.RECONNECT  # type: ignore[misc]

    def test_needs_recovery_when_reconnect(self) -> None:
        result = CrashRecoveryResult(
            action=RecoveryAction.RECONNECT,
            reason="Interrupted run detected",
            run_id="abc-123",
            status=RunStatus.RUNNING,
            host="prod.example.com",
            user="ci",
            port=2222,
            key_path="/home/ci/.ssh/id_ed25519",
            remote_pid=5678,
            daemon_pid=1234,
            resolved_shell="pytest -v",
            natural_language_command="run all tests",
            progress_percent=50.0,
            error=None,
            source_path=Path("/tmp/wiki/pages/daemon/current-run.md"),
        )
        assert result.needs_recovery is True

    def test_needs_recovery_when_resume_approval(self) -> None:
        result = CrashRecoveryResult(
            action=RecoveryAction.RESUME_APPROVAL,
            reason="Pending approval interrupted",
            run_id="abc-123",
            status=RunStatus.PENDING_APPROVAL,
            host="staging.example.com",
            user="deploy",
            port=22,
            key_path=None,
            remote_pid=None,
            daemon_pid=1000,
            resolved_shell=None,
            natural_language_command="run smoke tests",
            progress_percent=0.0,
            error=None,
            source_path=None,
        )
        assert result.needs_recovery is True

    def test_no_recovery_when_fresh_start(self) -> None:
        result = CrashRecoveryResult(
            action=RecoveryAction.FRESH_START,
            reason="No prior run",
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
        assert result.needs_recovery is False

    def test_has_connection_when_host_and_user_present(self) -> None:
        result = CrashRecoveryResult(
            action=RecoveryAction.RECONNECT,
            reason="Interrupted run",
            run_id="abc",
            status=RunStatus.RUNNING,
            host="prod.example.com",
            user="ci",
            port=22,
            key_path=None,
            remote_pid=5678,
            daemon_pid=1234,
            resolved_shell="pytest",
            natural_language_command="run tests",
            progress_percent=0.0,
            error=None,
            source_path=None,
        )
        assert result.has_connection is True

    def test_has_connection_false_when_no_host(self) -> None:
        result = CrashRecoveryResult(
            action=RecoveryAction.FRESH_START,
            reason="No prior run",
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
        assert result.has_connection is False


class TestRecoveryAction:
    def test_all_values(self) -> None:
        assert RecoveryAction.FRESH_START.value == "fresh_start"
        assert RecoveryAction.RECONNECT.value == "reconnect"
        assert RecoveryAction.RESUME_APPROVAL.value == "resume_approval"

    def test_every_run_status_maps_to_an_action(self) -> None:
        """Exhaustiveness guard: every RunStatus member produces a valid action."""
        from jules_daemon.wiki.crash_recovery import _determine_action

        for status in RunStatus:
            action = _determine_action(status)
            assert isinstance(action, RecoveryAction), (
                f"RunStatus.{status.name} did not map to a RecoveryAction"
            )


# -- detect_crash_recovery: No wiki file --


class TestDetectNoFile:
    """When no wiki file exists, return a fresh-start result."""

    def test_action_is_fresh_start(self, wiki_root: Path) -> None:
        result = detect_crash_recovery(wiki_root)
        assert result.action == RecoveryAction.FRESH_START

    def test_no_recovery_needed(self, wiki_root: Path) -> None:
        result = detect_crash_recovery(wiki_root)
        assert result.needs_recovery is False

    def test_status_is_idle(self, wiki_root: Path) -> None:
        result = detect_crash_recovery(wiki_root)
        assert result.status == RunStatus.IDLE

    def test_connection_fields_are_none(self, wiki_root: Path) -> None:
        result = detect_crash_recovery(wiki_root)
        assert result.host is None
        assert result.user is None
        assert result.port is None
        assert result.key_path is None

    def test_process_fields_are_none(self, wiki_root: Path) -> None:
        result = detect_crash_recovery(wiki_root)
        assert result.remote_pid is None
        assert result.daemon_pid is None

    def test_command_fields_are_none(self, wiki_root: Path) -> None:
        result = detect_crash_recovery(wiki_root)
        assert result.resolved_shell is None
        assert result.natural_language_command is None

    def test_run_id_is_empty(self, wiki_root: Path) -> None:
        result = detect_crash_recovery(wiki_root)
        assert result.run_id == ""

    def test_reason_is_set(self, wiki_root: Path) -> None:
        result = detect_crash_recovery(wiki_root)
        assert len(result.reason) > 0


# -- detect_crash_recovery: Idle wiki file --


class TestDetectIdleFile:
    """When wiki file exists with idle state, return fresh-start."""

    def test_action_is_fresh_start(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))
        result = detect_crash_recovery(wiki_root)
        assert result.action == RecoveryAction.FRESH_START

    def test_no_recovery_needed(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))
        result = detect_crash_recovery(wiki_root)
        assert result.needs_recovery is False


# -- detect_crash_recovery: Running state (crash recovery scenario) --


class TestDetectRunningState:
    """When wiki shows a running test, extract full connection details."""

    def _write_running_state(
        self,
        wiki_root: Path,
        host: str = "prod.example.com",
        user: str = "ci",
        port: int = 2222,
        key_path: str | None = "/home/ci/.ssh/id_ed25519",
        natural_language: str = "run the full regression suite",
        resolved_shell: str = "pytest -v --regression",
        daemon_pid: int = 9876,
        remote_pid: int = 5432,
    ) -> CurrentRun:
        target = SSHTarget(host=host, user=user, port=port, key_path=key_path)
        cmd = Command(natural_language=natural_language)
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=daemon_pid)
        run = run.with_running(resolved_shell, remote_pid=remote_pid)
        progress = Progress(
            percent=75.0,
            tests_passed=150,
            tests_failed=3,
            tests_total=200,
            last_output_line="FAILED test_payment_flow",
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)
        return run

    def test_action_is_reconnect(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        result = detect_crash_recovery(wiki_root)
        assert result.action == RecoveryAction.RECONNECT

    def test_needs_recovery(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        result = detect_crash_recovery(wiki_root)
        assert result.needs_recovery is True

    def test_extracts_host(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, host="prod.example.com")
        result = detect_crash_recovery(wiki_root)
        assert result.host == "prod.example.com"

    def test_extracts_user(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, user="ci")
        result = detect_crash_recovery(wiki_root)
        assert result.user == "ci"

    def test_extracts_port(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, port=2222)
        result = detect_crash_recovery(wiki_root)
        assert result.port == 2222

    def test_extracts_key_path(self, wiki_root: Path) -> None:
        self._write_running_state(
            wiki_root, key_path="/home/ci/.ssh/id_ed25519"
        )
        result = detect_crash_recovery(wiki_root)
        assert result.key_path == "/home/ci/.ssh/id_ed25519"

    def test_extracts_remote_pid(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, remote_pid=5432)
        result = detect_crash_recovery(wiki_root)
        assert result.remote_pid == 5432

    def test_extracts_daemon_pid(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, daemon_pid=9876)
        result = detect_crash_recovery(wiki_root)
        assert result.daemon_pid == 9876

    def test_extracts_resolved_shell(self, wiki_root: Path) -> None:
        self._write_running_state(
            wiki_root, resolved_shell="pytest -v --regression"
        )
        result = detect_crash_recovery(wiki_root)
        assert result.resolved_shell == "pytest -v --regression"

    def test_extracts_natural_language_command(self, wiki_root: Path) -> None:
        self._write_running_state(
            wiki_root, natural_language="run the full regression suite"
        )
        result = detect_crash_recovery(wiki_root)
        assert result.natural_language_command == "run the full regression suite"

    def test_extracts_progress_percent(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        result = detect_crash_recovery(wiki_root)
        assert result.progress_percent == 75.0

    def test_preserves_run_id(self, wiki_root: Path) -> None:
        run = self._write_running_state(wiki_root)
        result = detect_crash_recovery(wiki_root)
        assert result.run_id == run.run_id

    def test_status_is_running(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        result = detect_crash_recovery(wiki_root)
        assert result.status == RunStatus.RUNNING

    def test_has_connection(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        result = detect_crash_recovery(wiki_root)
        assert result.has_connection is True

    def test_source_path_set(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        result = detect_crash_recovery(wiki_root)
        assert result.source_path is not None
        assert result.source_path.name == "current-run.md"

    def test_no_error(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        result = detect_crash_recovery(wiki_root)
        assert result.error is None

    def test_handles_none_key_path(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, key_path=None)
        result = detect_crash_recovery(wiki_root)
        assert result.key_path is None
        assert result.has_connection is True


# -- detect_crash_recovery: Pending approval state --


class TestDetectPendingApprovalState:
    """When wiki shows pending_approval, extract connection and resume approval."""

    def test_action_is_resume_approval(self, wiki_root: Path) -> None:
        target = SSHTarget(host="staging.example.com", user="deploy")
        cmd = Command(natural_language="run smoke tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1000)
        current_run.write(wiki_root, run)

        result = detect_crash_recovery(wiki_root)
        assert result.action == RecoveryAction.RESUME_APPROVAL

    def test_needs_recovery(self, wiki_root: Path) -> None:
        target = SSHTarget(host="staging.example.com", user="deploy")
        cmd = Command(natural_language="run smoke tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1000)
        current_run.write(wiki_root, run)

        result = detect_crash_recovery(wiki_root)
        assert result.needs_recovery is True

    def test_extracts_connection_details(self, wiki_root: Path) -> None:
        target = SSHTarget(
            host="staging.example.com",
            user="deploy",
            port=22,
            key_path="/home/deploy/.ssh/id_rsa",
        )
        cmd = Command(natural_language="run smoke tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1000)
        current_run.write(wiki_root, run)

        result = detect_crash_recovery(wiki_root)
        assert result.host == "staging.example.com"
        assert result.user == "deploy"
        assert result.port == 22
        assert result.key_path == "/home/deploy/.ssh/id_rsa"

    def test_remote_pid_is_none(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        current_run.write(wiki_root, run)

        result = detect_crash_recovery(wiki_root)
        assert result.remote_pid is None

    def test_resolved_shell_is_none(self, wiki_root: Path) -> None:
        """Pending approval has not yet resolved the shell command."""
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        current_run.write(wiki_root, run)

        result = detect_crash_recovery(wiki_root)
        # Empty string in the model is converted to None in the recovery result
        assert result.resolved_shell is None

    def test_extracts_natural_language_command(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run smoke tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        current_run.write(wiki_root, run)

        result = detect_crash_recovery(wiki_root)
        assert result.natural_language_command == "run smoke tests"


# -- detect_crash_recovery: Terminal states (no recovery) --


class TestDetectTerminalStates:
    """Terminal states do not need recovery."""

    def test_completed_is_fresh_start(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        final = Progress(percent=100.0, tests_passed=10, tests_total=10)
        run = run.with_completed(final)
        current_run.write(wiki_root, run)

        result = detect_crash_recovery(wiki_root)
        assert result.action == RecoveryAction.FRESH_START
        assert result.needs_recovery is False

    def test_failed_is_fresh_start(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        error_progress = Progress(tests_failed=1, tests_total=1)
        run = run.with_failed("SSH timeout", error_progress)
        current_run.write(wiki_root, run)

        result = detect_crash_recovery(wiki_root)
        assert result.action == RecoveryAction.FRESH_START
        assert result.needs_recovery is False

    def test_cancelled_is_fresh_start(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_cancelled()
        current_run.write(wiki_root, run)

        result = detect_crash_recovery(wiki_root)
        assert result.action == RecoveryAction.FRESH_START
        assert result.needs_recovery is False


# -- detect_crash_recovery: Corrupted wiki file --


class TestDetectCorruptedFile:
    """Corrupted files fall back to fresh-start with error info."""

    def test_action_is_fresh_start(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("not valid yaml at all", encoding="utf-8")

        result = detect_crash_recovery(wiki_root)
        assert result.action == RecoveryAction.FRESH_START

    def test_no_recovery_needed(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("garbage", encoding="utf-8")

        result = detect_crash_recovery(wiki_root)
        assert result.needs_recovery is False

    def test_has_error_detail(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("bad content", encoding="utf-8")

        result = detect_crash_recovery(wiki_root)
        assert result.error is not None
        assert len(result.error) > 0

    def test_source_path_set(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("bad content", encoding="utf-8")

        result = detect_crash_recovery(wiki_root)
        assert result.source_path is not None

    def test_empty_file(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("", encoding="utf-8")

        result = detect_crash_recovery(wiki_root)
        assert result.action == RecoveryAction.FRESH_START
        assert result.needs_recovery is False

    def test_malformed_frontmatter(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "---\nstatus: not_a_valid_status\n---\nBody",
            encoding="utf-8",
        )

        result = detect_crash_recovery(wiki_root)
        assert result.action == RecoveryAction.FRESH_START
        assert result.error is not None


# -- Performance --


class TestPerformance:
    """Crash recovery detection must complete well within 30s SLA."""

    def test_completes_under_100ms(self, wiki_root: Path) -> None:
        """Full crash recovery detection should be fast."""
        target = SSHTarget(
            host="prod.example.com",
            user="ci",
            port=2222,
            key_path="/home/ci/.ssh/id_ed25519",
        )
        cmd = Command(natural_language="run full regression")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=9999)
        run = run.with_running("pytest -v --regression", remote_pid=8888)
        progress = Progress(
            percent=75.0,
            tests_passed=150,
            tests_failed=3,
            tests_total=200,
            last_output_line="FAILED test_payment_flow",
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)

        start = time.monotonic()
        result = detect_crash_recovery(wiki_root)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 100.0, (
            f"Crash recovery detection took {elapsed_ms:.1f}ms (>100ms)"
        )
        assert result.action == RecoveryAction.RECONNECT
        assert result.host == "prod.example.com"

    def test_no_file_under_10ms(self, wiki_root: Path) -> None:
        """Fresh start path should be very fast."""
        start = time.monotonic()
        result = detect_crash_recovery(wiki_root)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 10.0, (
            f"Fresh start detection took {elapsed_ms:.1f}ms (>10ms)"
        )
        assert result.action == RecoveryAction.FRESH_START


# -- End-to-end: full crash recovery scenario --


class TestEndToEndCrashRecovery:
    """Simulate a full crash-and-recover cycle using the wiki."""

    def test_daemon_crash_recovery_cycle(self, wiki_root: Path) -> None:
        """Daemon starts a run, crashes, new daemon detects and recovers."""
        # Step 1: First daemon writes running state and "crashes"
        target = SSHTarget(
            host="prod.example.com",
            user="ci",
            port=2222,
            key_path="/home/ci/.ssh/id_ed25519",
        )
        cmd = Command(natural_language="run the full regression suite")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        run = run.with_running("pytest -v --regression", remote_pid=5678)
        progress = Progress(
            percent=60.0,
            tests_passed=120,
            tests_failed=2,
            tests_skipped=5,
            tests_total=200,
            last_output_line="FAILED test_payment_flow",
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)

        # Step 2: New daemon starts, detects crash recovery
        result = detect_crash_recovery(wiki_root)

        # Step 3: Verify all fields are extracted correctly
        assert result.needs_recovery is True
        assert result.action == RecoveryAction.RECONNECT
        assert result.run_id == run.run_id
        assert result.status == RunStatus.RUNNING
        assert result.host == "prod.example.com"
        assert result.user == "ci"
        assert result.port == 2222
        assert result.key_path == "/home/ci/.ssh/id_ed25519"
        assert result.remote_pid == 5678
        assert result.daemon_pid == 1234
        assert result.resolved_shell == "pytest -v --regression"
        assert result.natural_language_command == "run the full regression suite"
        assert result.progress_percent == 60.0
        assert result.error is None
        assert result.has_connection is True

    def test_clean_start_after_completed_run(self, wiki_root: Path) -> None:
        """After a completed run, new daemon starts fresh."""
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        final = Progress(percent=100.0, tests_passed=10, tests_total=10)
        run = run.with_completed(final)
        current_run.write(wiki_root, run)

        result = detect_crash_recovery(wiki_root)
        assert result.needs_recovery is False
        assert result.action == RecoveryAction.FRESH_START
