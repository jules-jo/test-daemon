"""Integration tests simulating daemon crash (SIGKILL) and restart.

These tests exercise the full crash-and-restart pipeline end-to-end:
  1. Write wiki state representing a running daemon (pre-crash snapshot)
  2. Simulate SIGKILL (daemon process dies, wiki file persists on disk)
  3. Run the complete startup recovery flow: detection + orchestration
  4. Assert recovery completes within the 30-second SLA
  5. Verify correct structured results from the resumed run

Each test simulates the daemon lifecycle using real wiki file I/O,
real YAML frontmatter parsing, and real state transitions -- only SSH
connections are mocked. This validates that the wiki persistence layer
and recovery orchestrator work together correctly after a crash.

The 30-second SLA is enforced by measuring wall-clock time with
time.monotonic(), independent of the orchestrator's internal deadline.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from jules_daemon.wiki import current_run
from jules_daemon.wiki.checkpoint_recovery import (
    RecoverySource,
    recover_monitoring_checkpoint,
)
from jules_daemon.wiki.crash_recovery import (
    CrashRecoveryResult,
    RecoveryAction,
    detect_crash_recovery,
)
from jules_daemon.wiki.frontmatter import parse as parse_frontmatter
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
    RecoveryTimeoutConfig,
    orchestrate_recovery,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RECOVERY_SLA_SECONDS = 30.0

# Pre-crash state fixtures representing realistic mid-run progress
_DEFAULT_HOST = "ci-runner-01.internal.example.com"
_DEFAULT_USER = "testrunner"
_DEFAULT_PORT = 2222
_DEFAULT_KEY_PATH = "/home/testrunner/.ssh/id_ed25519"
_DEFAULT_COMMAND_NL = "run the full regression suite with coverage"
_DEFAULT_COMMAND_SHELL = "cd /opt/project && pytest -v --cov=src tests/"
_DEFAULT_DAEMON_PID = 48712
_DEFAULT_REMOTE_PID = 93104


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


def _make_ssh_target(
    *,
    host: str = _DEFAULT_HOST,
    user: str = _DEFAULT_USER,
    port: int = _DEFAULT_PORT,
    key_path: str | None = _DEFAULT_KEY_PATH,
) -> SSHTarget:
    return SSHTarget(host=host, user=user, port=port, key_path=key_path)


def _make_running_state(
    *,
    host: str = _DEFAULT_HOST,
    user: str = _DEFAULT_USER,
    port: int = _DEFAULT_PORT,
    key_path: str | None = _DEFAULT_KEY_PATH,
    natural_language: str = _DEFAULT_COMMAND_NL,
    resolved_shell: str = _DEFAULT_COMMAND_SHELL,
    daemon_pid: int = _DEFAULT_DAEMON_PID,
    remote_pid: int = _DEFAULT_REMOTE_PID,
    percent: float = 62.0,
    tests_passed: int = 124,
    tests_failed: int = 3,
    tests_skipped: int = 7,
    tests_total: int = 200,
    last_output_line: str = "FAILED tests/payment/test_refund.py::test_partial_refund",
) -> CurrentRun:
    """Build a realistic RUNNING state with mid-test progress."""
    target = _make_ssh_target(
        host=host, user=user, port=port, key_path=key_path,
    )
    cmd = Command(natural_language=natural_language)
    run = CurrentRun().with_pending_approval(
        target, cmd, daemon_pid=daemon_pid,
    )
    run = run.with_running(resolved_shell, remote_pid=remote_pid)
    progress = Progress(
        percent=percent,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_skipped=tests_skipped,
        tests_total=tests_total,
        last_output_line=last_output_line,
        checkpoint_at=datetime.now(timezone.utc),
    )
    return run.with_progress(progress)


def _make_pending_approval_state(
    *,
    host: str = _DEFAULT_HOST,
    user: str = _DEFAULT_USER,
    natural_language: str = _DEFAULT_COMMAND_NL,
    daemon_pid: int = _DEFAULT_DAEMON_PID,
) -> CurrentRun:
    """Build a PENDING_APPROVAL state (crash before user confirmed)."""
    target = _make_ssh_target(host=host, user=user)
    cmd = Command(natural_language=natural_language)
    return CurrentRun().with_pending_approval(
        target, cmd, daemon_pid=daemon_pid,
    )


def _make_successful_connector() -> AsyncMock:
    """Build a mock SSH connector that succeeds immediately."""
    connector = AsyncMock()
    handle = MagicMock()
    handle.session_id = "recovered-session-001"
    connector.connect = AsyncMock(return_value=handle)
    connector.close = AsyncMock(return_value=None)
    connector.is_alive = AsyncMock(return_value=True)
    return connector


def _make_slow_then_success_connector(
    fail_count: int = 2,
    delay_per_fail: float = 0.05,
) -> AsyncMock:
    """Build a connector that fails N times then succeeds."""
    connector = AsyncMock()
    handle = MagicMock()
    handle.session_id = "recovered-session-002"
    call_count = 0

    async def _connect(target: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count <= fail_count:
            await asyncio.sleep(delay_per_fail)
            raise ConnectionError(
                f"Connection refused (attempt {call_count})"
            )
        return handle

    connector.connect = _connect
    connector.close = AsyncMock(return_value=None)
    connector.is_alive = AsyncMock(return_value=True)
    return connector


# ---------------------------------------------------------------------------
# Test: Full crash-restart cycle with RUNNING state
# ---------------------------------------------------------------------------


class TestCrashRestartRunningState:
    """Simulate SIGKILL during a RUNNING test and verify full recovery."""

    def _write_crash_state(self, wiki_root: Path) -> CurrentRun:
        """Write a realistic mid-run state and return the frozen snapshot."""
        run = _make_running_state()
        current_run.write(wiki_root, run)
        return run

    @pytest.mark.asyncio
    async def test_full_recovery_within_30s_sla(
        self, wiki_root: Path,
    ) -> None:
        """End-to-end: crash detection + orchestration completes in < 30s."""
        pre_crash = self._write_crash_state(wiki_root)
        connector = _make_successful_connector()

        wall_start = time.monotonic()

        # Phase 1: detect crash recovery (what the daemon does on startup)
        detection = detect_crash_recovery(wiki_root)

        # Phase 2: orchestrate recovery under deadline
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
        )

        wall_elapsed = time.monotonic() - wall_start

        # 30-second SLA
        assert wall_elapsed < _RECOVERY_SLA_SECONDS, (
            f"Recovery took {wall_elapsed:.2f}s, exceeding "
            f"{_RECOVERY_SLA_SECONDS}s SLA"
        )

        # Structural correctness
        assert outcome.success is True
        assert outcome.action_taken == RecoveryAction.RECONNECT
        assert outcome.timed_out is False
        assert outcome.run_id == pre_crash.run_id

    @pytest.mark.asyncio
    async def test_detection_identifies_reconnect(
        self, wiki_root: Path,
    ) -> None:
        """Crash detection correctly identifies RECONNECT action."""
        self._write_crash_state(wiki_root)

        detection = detect_crash_recovery(wiki_root)

        assert detection.action == RecoveryAction.RECONNECT
        assert detection.needs_recovery is True
        assert detection.status == RunStatus.RUNNING

    @pytest.mark.asyncio
    async def test_connection_details_preserved(
        self, wiki_root: Path,
    ) -> None:
        """SSH connection details survive crash and are available for recovery."""
        self._write_crash_state(wiki_root)

        detection = detect_crash_recovery(wiki_root)

        assert detection.host == _DEFAULT_HOST
        assert detection.user == _DEFAULT_USER
        assert detection.port == _DEFAULT_PORT
        assert detection.key_path == _DEFAULT_KEY_PATH
        assert detection.has_connection is True

    @pytest.mark.asyncio
    async def test_process_ids_preserved(
        self, wiki_root: Path,
    ) -> None:
        """Daemon and remote PIDs survive crash for re-attach."""
        self._write_crash_state(wiki_root)

        detection = detect_crash_recovery(wiki_root)

        assert detection.daemon_pid == _DEFAULT_DAEMON_PID
        assert detection.remote_pid == _DEFAULT_REMOTE_PID

    @pytest.mark.asyncio
    async def test_command_details_preserved(
        self, wiki_root: Path,
    ) -> None:
        """Natural language and resolved shell commands survive crash."""
        self._write_crash_state(wiki_root)

        detection = detect_crash_recovery(wiki_root)

        assert detection.natural_language_command == _DEFAULT_COMMAND_NL
        assert detection.resolved_shell == _DEFAULT_COMMAND_SHELL

    @pytest.mark.asyncio
    async def test_progress_preserved(
        self, wiki_root: Path,
    ) -> None:
        """Mid-run progress snapshot survives crash."""
        self._write_crash_state(wiki_root)

        detection = detect_crash_recovery(wiki_root)

        assert detection.progress_percent == 62.0

    @pytest.mark.asyncio
    async def test_recovery_produces_structured_outcome(
        self, wiki_root: Path,
    ) -> None:
        """RecoveryOutcome contains all phase timings and metadata."""
        self._write_crash_state(wiki_root)
        connector = _make_successful_connector()

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
        )

        # Outcome has correct top-level structure
        assert isinstance(outcome, RecoveryOutcome)
        assert outcome.success is True
        assert outcome.deadline_seconds == 30.0
        assert outcome.total_duration_seconds > 0.0
        assert outcome.time_remaining_seconds > 0.0

        # Phases are populated with timing data
        assert len(outcome.phases) >= 2
        phase_names = [p.phase for p in outcome.phases]
        assert RecoveryPhase.RECONNECT in phase_names
        assert RecoveryPhase.WIKI_UPDATE in phase_names

        # All phases succeeded
        for phase in outcome.phases:
            assert phase.success is True
            assert phase.duration_seconds >= 0.0
            assert phase.error is None

    @pytest.mark.asyncio
    async def test_wiki_updated_after_recovery(
        self, wiki_root: Path,
    ) -> None:
        """Wiki is updated with recovery log after successful recovery."""
        self._write_crash_state(wiki_root)
        connector = _make_successful_connector()

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
        )

        assert outcome.wiki_updated is True

        # Recovery log wiki file exists and has proper frontmatter
        recovery_log = wiki_root / "pages" / "daemon" / "recovery-log.md"
        assert recovery_log.exists()

        content = recovery_log.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert doc.frontmatter["type"] == "daemon-recovery-log"
        assert doc.frontmatter["success"] is True
        assert doc.frontmatter["action"] == "reconnect"
        assert doc.frontmatter["timed_out"] is False


# ---------------------------------------------------------------------------
# Test: Crash during PENDING_APPROVAL
# ---------------------------------------------------------------------------


class TestCrashRestartPendingApproval:
    """Simulate SIGKILL during PENDING_APPROVAL and verify recovery."""

    def _write_crash_state(self, wiki_root: Path) -> CurrentRun:
        run = _make_pending_approval_state()
        current_run.write(wiki_root, run)
        return run

    @pytest.mark.asyncio
    async def test_resume_approval_within_30s_sla(
        self, wiki_root: Path,
    ) -> None:
        """PENDING_APPROVAL recovery completes within the SLA."""
        pre_crash = self._write_crash_state(wiki_root)

        wall_start = time.monotonic()

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
        )

        wall_elapsed = time.monotonic() - wall_start

        assert wall_elapsed < _RECOVERY_SLA_SECONDS
        assert outcome.success is True
        assert outcome.action_taken == RecoveryAction.RESUME_APPROVAL
        assert outcome.timed_out is False
        assert outcome.run_id == pre_crash.run_id

    @pytest.mark.asyncio
    async def test_resume_approval_skips_ssh(
        self, wiki_root: Path,
    ) -> None:
        """PENDING_APPROVAL recovery does not attempt SSH reconnection."""
        self._write_crash_state(wiki_root)

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
        )

        phase_names = [p.phase for p in outcome.phases]
        assert RecoveryPhase.RECONNECT not in phase_names
        assert RecoveryPhase.PID_CHECK not in phase_names
        assert RecoveryPhase.REATTACH not in phase_names

    @pytest.mark.asyncio
    async def test_resume_approval_preserves_command(
        self, wiki_root: Path,
    ) -> None:
        """Original NL command is preserved for re-prompting the user."""
        self._write_crash_state(wiki_root)

        detection = detect_crash_recovery(wiki_root)

        assert detection.natural_language_command == _DEFAULT_COMMAND_NL
        assert detection.host == _DEFAULT_HOST
        assert detection.user == _DEFAULT_USER


# ---------------------------------------------------------------------------
# Test: Recovery with transient SSH failures (retry scenario)
# ---------------------------------------------------------------------------


class TestCrashRestartWithRetries:
    """Simulate crash recovery where SSH reconnect requires retries."""

    def _write_crash_state(self, wiki_root: Path) -> CurrentRun:
        run = _make_running_state()
        current_run.write(wiki_root, run)
        return run

    @pytest.mark.asyncio
    async def test_retries_then_succeeds_within_sla(
        self, wiki_root: Path,
    ) -> None:
        """Recovery succeeds after transient SSH failures, within 30s."""
        pre_crash = self._write_crash_state(wiki_root)
        connector = _make_slow_then_success_connector(
            fail_count=2, delay_per_fail=0.02,
        )

        wall_start = time.monotonic()

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
        )

        wall_elapsed = time.monotonic() - wall_start

        assert wall_elapsed < _RECOVERY_SLA_SECONDS
        assert outcome.success is True
        assert outcome.action_taken == RecoveryAction.RECONNECT
        assert outcome.run_id == pre_crash.run_id

    @pytest.mark.asyncio
    async def test_reconnect_failure_marks_wiki_failed(
        self, wiki_root: Path,
    ) -> None:
        """When SSH reconnect fails permanently, wiki is marked FAILED."""
        self._write_crash_state(wiki_root)

        connector = AsyncMock()
        connector.connect = AsyncMock(
            side_effect=ConnectionError("Host unreachable"),
        )

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
        )

        assert outcome.success is False
        assert outcome.wiki_updated is True

        # Verify wiki state was updated to FAILED
        run_after = current_run.read(wiki_root)
        assert run_after is not None
        assert run_after.status == RunStatus.FAILED
        assert run_after.error is not None


# ---------------------------------------------------------------------------
# Test: Recovery deadline enforcement
# ---------------------------------------------------------------------------


class TestCrashRestartDeadlineEnforcement:
    """Verify the 30-second deadline is enforced during recovery."""

    def _write_crash_state(self, wiki_root: Path) -> CurrentRun:
        run = _make_running_state()
        current_run.write(wiki_root, run)
        return run

    @pytest.mark.asyncio
    async def test_slow_reconnect_aborts_at_deadline(
        self, wiki_root: Path,
    ) -> None:
        """Recovery that exceeds deadline is aborted and marked failed."""
        self._write_crash_state(wiki_root)

        async def slow_connect(target: Any) -> None:
            await asyncio.sleep(100)  # far exceeds any reasonable deadline

        connector = AsyncMock()
        connector.connect = slow_connect

        # Use a very short deadline so the test runs quickly
        config = RecoveryTimeoutConfig(total_deadline_seconds=0.3)

        wall_start = time.monotonic()

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
            config=config,
        )

        wall_elapsed = time.monotonic() - wall_start

        # Should abort well under the outer SLA
        assert wall_elapsed < 5.0
        assert outcome.success is False
        assert outcome.timed_out is True
        assert outcome.error is not None

        # Wiki should be updated to FAILED
        run_after = current_run.read(wiki_root)
        assert run_after is not None
        assert run_after.status == RunStatus.FAILED

    @pytest.mark.asyncio
    async def test_deadline_recorded_in_outcome(
        self, wiki_root: Path,
    ) -> None:
        """RecoveryOutcome records the configured deadline for audit."""
        self._write_crash_state(wiki_root)
        connector = _make_successful_connector()

        config = RecoveryTimeoutConfig(total_deadline_seconds=15.0)

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
            config=config,
        )

        assert outcome.deadline_seconds == 15.0
        assert outcome.total_duration_seconds < 15.0
        assert outcome.time_remaining_seconds > 0.0


# ---------------------------------------------------------------------------
# Test: Monitoring checkpoint recovery after crash
# ---------------------------------------------------------------------------


class TestCrashRestartCheckpointRecovery:
    """Verify that monitoring checkpoints are recoverable after crash."""

    @pytest.mark.asyncio
    async def test_progress_metrics_recoverable(
        self, wiki_root: Path,
    ) -> None:
        """Monitoring checkpoint recovers test counts from wiki."""
        run = _make_running_state(
            percent=62.0,
            tests_passed=124,
            tests_failed=3,
            tests_skipped=7,
            tests_total=200,
        )
        current_run.write(wiki_root, run)

        # Simulate restart: recover monitoring checkpoint
        checkpoint = recover_monitoring_checkpoint(wiki_root)

        assert checkpoint.source == RecoverySource.WIKI_STATE
        assert checkpoint.is_resumable is True
        assert checkpoint.run_id == run.run_id
        assert checkpoint.status == RunStatus.RUNNING

        # Metrics match pre-crash state
        metrics = checkpoint.extracted_metrics
        assert metrics.tests_passed == 124
        assert metrics.tests_failed == 3
        assert metrics.tests_skipped == 7
        assert metrics.tests_total == 200
        assert metrics.percent == 62.0

    @pytest.mark.asyncio
    async def test_checkpoint_not_resumable_after_completed(
        self, wiki_root: Path,
    ) -> None:
        """Completed runs are not resumable via checkpoint."""
        run = _make_running_state()
        final = Progress(
            percent=100.0,
            tests_passed=200,
            tests_failed=0,
            tests_total=200,
        )
        completed = run.with_completed(final)
        current_run.write(wiki_root, completed)

        checkpoint = recover_monitoring_checkpoint(wiki_root)
        assert checkpoint.is_resumable is False


# ---------------------------------------------------------------------------
# Test: Multiple crash-restart cycles
# ---------------------------------------------------------------------------


class TestMultipleCrashRestartCycles:
    """Verify the daemon can survive multiple crash-restart cycles."""

    @pytest.mark.asyncio
    async def test_three_consecutive_crash_restart_cycles(
        self, wiki_root: Path,
    ) -> None:
        """Daemon recovers correctly through three crash-restart cycles."""
        connector = _make_successful_connector()

        for cycle in range(3):
            # Write progressively advancing state
            percent = 20.0 + (cycle * 25.0)
            passed = 40 + (cycle * 50)
            run = _make_running_state(
                percent=percent,
                tests_passed=passed,
                tests_failed=cycle,
                tests_total=200,
            )
            current_run.write(wiki_root, run)

            # Simulate restart
            wall_start = time.monotonic()

            detection = detect_crash_recovery(wiki_root)
            outcome = await orchestrate_recovery(
                recovery=detection,
                wiki_root=wiki_root,
                connector=connector,
            )

            wall_elapsed = time.monotonic() - wall_start

            assert wall_elapsed < _RECOVERY_SLA_SECONDS, (
                f"Cycle {cycle + 1}: recovery took {wall_elapsed:.2f}s"
            )
            assert outcome.success is True
            assert outcome.action_taken == RecoveryAction.RECONNECT
            assert outcome.run_id == run.run_id

    @pytest.mark.asyncio
    async def test_recovery_after_previous_failed_recovery(
        self, wiki_root: Path,
    ) -> None:
        """Second restart recovers even if first recovery marked FAILED."""
        # First crash: write running state
        run = _make_running_state()
        current_run.write(wiki_root, run)

        # First restart: recovery fails (no connector)
        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=None,
        )
        assert outcome.success is False

        # Wiki should now be FAILED
        run_after_fail = current_run.read(wiki_root)
        assert run_after_fail is not None
        assert run_after_fail.status == RunStatus.FAILED

        # Second restart: new daemon detects FAILED state (fresh start)
        detection2 = detect_crash_recovery(wiki_root)
        assert detection2.action == RecoveryAction.FRESH_START
        assert detection2.needs_recovery is False


# ---------------------------------------------------------------------------
# Test: Wiki file integrity across crash-restart
# ---------------------------------------------------------------------------


class TestWikiIntegrityAcrossCrash:
    """Verify wiki file format integrity is maintained through crash-restart."""

    @pytest.mark.asyncio
    async def test_wiki_file_has_valid_frontmatter_after_recovery(
        self, wiki_root: Path,
    ) -> None:
        """Current-run wiki file has valid YAML frontmatter after recovery."""
        run = _make_running_state()
        current_run.write(wiki_root, run)
        connector = _make_successful_connector()

        detection = detect_crash_recovery(wiki_root)
        await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
        )

        # Read and validate the wiki file format
        wiki_file = current_run.file_path(wiki_root)
        assert wiki_file.exists()

        raw = wiki_file.read_text(encoding="utf-8")
        assert raw.startswith("---\n")

        doc = parse_frontmatter(raw)
        assert isinstance(doc.frontmatter, dict)
        assert "status" in doc.frontmatter
        assert "run_id" in doc.frontmatter
        assert "ssh_target" in doc.frontmatter
        assert "command" in doc.frontmatter
        assert "pids" in doc.frontmatter
        assert "progress" in doc.frontmatter

    @pytest.mark.asyncio
    async def test_recovery_log_has_valid_frontmatter(
        self, wiki_root: Path,
    ) -> None:
        """Recovery log wiki file has valid YAML frontmatter format."""
        run = _make_running_state()
        current_run.write(wiki_root, run)
        connector = _make_successful_connector()

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
        )

        assert outcome.wiki_updated is True

        log_path = wiki_root / "pages" / "daemon" / "recovery-log.md"
        raw = log_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(raw)

        assert "tags" in doc.frontmatter
        assert "daemon" in doc.frontmatter["tags"]
        assert "recovery" in doc.frontmatter["tags"]
        assert "audit" in doc.frontmatter["tags"]
        assert doc.frontmatter["type"] == "daemon-recovery-log"
        assert isinstance(doc.frontmatter["phases"], list)

    @pytest.mark.asyncio
    async def test_wiki_roundtrip_preserves_all_fields(
        self, wiki_root: Path,
    ) -> None:
        """Writing then reading wiki preserves all run fields across crash."""
        original = _make_running_state(
            host="custom-host.example.com",
            user="deployer",
            port=3333,
            key_path="/opt/keys/deploy_key",
            natural_language="run smoke tests quickly",
            resolved_shell="pytest -x tests/smoke/",
            daemon_pid=11111,
            remote_pid=22222,
            percent=88.5,
            tests_passed=177,
            tests_failed=1,
            tests_skipped=2,
            tests_total=200,
            last_output_line="PASSED tests/smoke/test_health.py",
        )
        current_run.write(wiki_root, original)

        # Simulate restart: read back
        restored = current_run.read(wiki_root)
        assert restored is not None

        assert restored.run_id == original.run_id
        assert restored.status == RunStatus.RUNNING

        assert restored.ssh_target is not None
        assert restored.ssh_target.host == "custom-host.example.com"
        assert restored.ssh_target.user == "deployer"
        assert restored.ssh_target.port == 3333
        assert restored.ssh_target.key_path == "/opt/keys/deploy_key"

        assert restored.command is not None
        assert restored.command.natural_language == "run smoke tests quickly"
        assert restored.command.resolved_shell == "pytest -x tests/smoke/"
        assert restored.command.approved is True

        assert restored.pids.daemon == 11111
        assert restored.pids.remote == 22222

        assert restored.progress.percent == 88.5
        assert restored.progress.tests_passed == 177
        assert restored.progress.tests_failed == 1
        assert restored.progress.tests_skipped == 2
        assert restored.progress.tests_total == 200
        assert restored.progress.last_output_line == (
            "PASSED tests/smoke/test_health.py"
        )


# ---------------------------------------------------------------------------
# Test: Simulated SIGKILL via subprocess
# ---------------------------------------------------------------------------


class TestSigkillSimulation:
    """Simulate actual SIGKILL of a child process that writes wiki state.

    Spawns a subprocess that writes RUNNING state to the wiki and then
    receives SIGKILL. The parent process then runs crash detection to
    verify the wiki state survives the kill.
    """

    @pytest.mark.asyncio
    async def test_wiki_survives_sigkill(self, wiki_root: Path) -> None:
        """Wiki state written by a killed process is readable after kill."""
        # The child script writes a running state and then blocks forever.
        # The parent sends SIGKILL, then reads the wiki.
        child_script = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "from jules_daemon.wiki import current_run\n"
            "from jules_daemon.wiki.models import (\n"
            "    Command, CurrentRun, Progress, SSHTarget,\n"
            ")\n"
            "wiki_root = Path(sys.argv[1])\n"
            "target = SSHTarget(host='sigkill-test.example.com', user='ci')\n"
            "cmd = Command(natural_language='run killed test')\n"
            "run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=99999)\n"
            "run = run.with_running('pytest -v', remote_pid=88888)\n"
            "progress = Progress(\n"
            "    percent=45.0, tests_passed=90, tests_failed=1, tests_total=200,\n"
            ")\n"
            "run = run.with_progress(progress)\n"
            "current_run.write(wiki_root, run)\n"
            "# Signal parent that write is done\n"
            "sys.stdout.write('READY\\n')\n"
            "sys.stdout.flush()\n"
            "# Block until killed\n"
            "time.sleep(3600)\n"
        )

        script_path = wiki_root.parent / "child_writer.py"
        script_path.write_text(child_script, encoding="utf-8")

        # Start child process
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(wiki_root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait for child to signal it wrote the wiki
            assert proc.stdout is not None
            line = proc.stdout.readline().strip()
            assert line == "READY", f"Child did not signal ready, got: {line!r}"

            # Send SIGKILL (no cleanup handlers run)
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=5)
            assert proc.returncode == -signal.SIGKILL

        except Exception:
            proc.kill()
            proc.wait()
            raise

        # Now verify the wiki survived the kill
        wall_start = time.monotonic()

        detection = detect_crash_recovery(wiki_root)

        wall_elapsed = time.monotonic() - wall_start

        assert wall_elapsed < _RECOVERY_SLA_SECONDS
        assert detection.action == RecoveryAction.RECONNECT
        assert detection.needs_recovery is True
        assert detection.host == "sigkill-test.example.com"
        assert detection.user == "ci"
        assert detection.remote_pid == 88888
        assert detection.daemon_pid == 99999
        assert detection.progress_percent == 45.0
        assert detection.resolved_shell == "pytest -v"
        assert detection.natural_language_command == "run killed test"

    @pytest.mark.asyncio
    async def test_full_recovery_after_sigkill(
        self, wiki_root: Path,
    ) -> None:
        """Full recovery pipeline succeeds after SIGKILL of writer process."""
        # Write state via subprocess, kill it, then run full recovery
        child_script = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "from jules_daemon.wiki import current_run\n"
            "from jules_daemon.wiki.models import (\n"
            "    Command, CurrentRun, Progress, SSHTarget,\n"
            ")\n"
            "wiki_root = Path(sys.argv[1])\n"
            "target = SSHTarget(host='recovery-test.example.com', user='ci')\n"
            "cmd = Command(natural_language='full recovery test')\n"
            "run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=77777)\n"
            "run = run.with_running('pytest --full', remote_pid=66666)\n"
            "progress = Progress(\n"
            "    percent=70.0, tests_passed=140, tests_failed=5,\n"
            "    tests_skipped=3, tests_total=200,\n"
            "    last_output_line='FAILED test_api.py::test_timeout',\n"
            ")\n"
            "run = run.with_progress(progress)\n"
            "current_run.write(wiki_root, run)\n"
            "sys.stdout.write('READY\\n')\n"
            "sys.stdout.flush()\n"
            "time.sleep(3600)\n"
        )

        script_path = wiki_root.parent / "child_writer_recovery.py"
        script_path.write_text(child_script, encoding="utf-8")

        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(wiki_root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            assert proc.stdout is not None
            line = proc.stdout.readline().strip()
            assert line == "READY"

            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=5)

        except Exception:
            proc.kill()
            proc.wait()
            raise

        # Run full recovery pipeline
        connector = _make_successful_connector()

        wall_start = time.monotonic()

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
        )

        wall_elapsed = time.monotonic() - wall_start

        # 30-second SLA
        assert wall_elapsed < _RECOVERY_SLA_SECONDS

        # Structured results are correct
        assert outcome.success is True
        assert outcome.action_taken == RecoveryAction.RECONNECT
        assert outcome.timed_out is False
        assert outcome.wiki_updated is True

        # Verify detection extracted correct data from the killed process
        assert detection.host == "recovery-test.example.com"
        assert detection.remote_pid == 66666
        assert detection.progress_percent == 70.0
        assert detection.resolved_shell == "pytest --full"


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestCrashRestartEdgeCases:
    """Edge cases in the crash-restart flow."""

    @pytest.mark.asyncio
    async def test_no_prior_state_is_fresh_start(
        self, wiki_root: Path,
    ) -> None:
        """Empty wiki directory results in FRESH_START (first boot)."""
        wall_start = time.monotonic()

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
        )

        wall_elapsed = time.monotonic() - wall_start

        assert wall_elapsed < _RECOVERY_SLA_SECONDS
        assert outcome.success is True
        assert outcome.action_taken == RecoveryAction.FRESH_START
        assert len(outcome.phases) == 0

    @pytest.mark.asyncio
    async def test_corrupted_wiki_is_fresh_start(
        self, wiki_root: Path,
    ) -> None:
        """Corrupted wiki file results in safe FRESH_START with error."""
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "THIS IS NOT VALID YAML FRONTMATTER AT ALL",
            encoding="utf-8",
        )

        wall_start = time.monotonic()

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
        )

        wall_elapsed = time.monotonic() - wall_start

        assert wall_elapsed < _RECOVERY_SLA_SECONDS
        assert detection.action == RecoveryAction.FRESH_START
        assert detection.error is not None
        assert outcome.success is True

    @pytest.mark.asyncio
    async def test_zero_progress_running_state(
        self, wiki_root: Path,
    ) -> None:
        """Recovery works when crash happened at 0% progress."""
        run = _make_running_state(
            percent=0.0,
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            tests_total=200,
            last_output_line="",
        )
        current_run.write(wiki_root, run)
        connector = _make_successful_connector()

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
        )

        assert outcome.success is True
        assert detection.progress_percent == 0.0

    @pytest.mark.asyncio
    async def test_near_complete_progress_running_state(
        self, wiki_root: Path,
    ) -> None:
        """Recovery works when crash happened at 99% progress."""
        run = _make_running_state(
            percent=99.5,
            tests_passed=199,
            tests_failed=0,
            tests_skipped=0,
            tests_total=200,
            last_output_line="PASSED tests/final/test_cleanup.py",
        )
        current_run.write(wiki_root, run)
        connector = _make_successful_connector()

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
        )

        assert outcome.success is True
        assert detection.progress_percent == 99.5


# ---------------------------------------------------------------------------
# Test: Performance - recovery detection + orchestration combined
# ---------------------------------------------------------------------------


class TestCrashRestartPerformance:
    """Verify the combined detection + orchestration performance."""

    @pytest.mark.asyncio
    async def test_full_pipeline_under_1_second(
        self, wiki_root: Path,
    ) -> None:
        """Detection + orchestration with mock SSH completes in under 1s."""
        run = _make_running_state()
        current_run.write(wiki_root, run)
        connector = _make_successful_connector()

        wall_start = time.monotonic()

        detection = detect_crash_recovery(wiki_root)
        outcome = await orchestrate_recovery(
            recovery=detection,
            wiki_root=wiki_root,
            connector=connector,
        )

        wall_elapsed = time.monotonic() - wall_start

        # With mock SSH, full pipeline should be very fast
        assert wall_elapsed < 1.0, (
            f"Full pipeline took {wall_elapsed:.3f}s (expected < 1s)"
        )
        assert outcome.success is True

    @pytest.mark.asyncio
    async def test_detection_alone_under_50ms(
        self, wiki_root: Path,
    ) -> None:
        """Crash detection (wiki read + parse) completes in under 50ms."""
        run = _make_running_state()
        current_run.write(wiki_root, run)

        wall_start = time.monotonic()
        detection = detect_crash_recovery(wiki_root)
        wall_elapsed_ms = (time.monotonic() - wall_start) * 1000

        assert wall_elapsed_ms < 50.0, (
            f"Detection took {wall_elapsed_ms:.1f}ms (expected < 50ms)"
        )
        assert detection.action == RecoveryAction.RECONNECT
