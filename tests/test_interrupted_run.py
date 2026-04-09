"""Tests for interrupted-run detection logic.

Verifies that the detection logic:
- Examines parsed run records to determine if a prior run was interrupted
- Returns a structured RecoveryVerdict with recovery-needed flag and context
- Identifies RUNNING and PENDING_APPROVAL as interrupted states
- Identifies IDLE, COMPLETED, FAILED, CANCELLED as non-interrupted states
- Handles None input (no prior record) as non-interrupted
- Computes staleness from the record's updated_at timestamp
- Extracts process IDs for daemon liveness checks
- Provides human-readable reason strings
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jules_daemon.wiki.interrupted_run import (
    RecoveryVerdict,
    detect_interrupted_run,
)
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    ProcessIDs,
    Progress,
    RunStatus,
    SSHTarget,
)


# -- RecoveryVerdict dataclass --


class TestRecoveryVerdict:
    def test_create_with_all_fields(self) -> None:
        verdict = RecoveryVerdict(
            recovery_needed=True,
            reason="Run was in progress when daemon stopped",
            interrupted_status=RunStatus.RUNNING,
            run_id="abc-123",
            stale_seconds=45.0,
            has_remote_process=True,
            daemon_pid=1234,
        )
        assert verdict.recovery_needed is True
        assert verdict.reason == "Run was in progress when daemon stopped"
        assert verdict.interrupted_status == RunStatus.RUNNING
        assert verdict.run_id == "abc-123"
        assert verdict.stale_seconds == 45.0
        assert verdict.has_remote_process is True
        assert verdict.daemon_pid == 1234

    def test_create_no_recovery(self) -> None:
        verdict = RecoveryVerdict(
            recovery_needed=False,
            reason="No active run found",
            interrupted_status=RunStatus.IDLE,
            run_id="xyz-789",
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=None,
        )
        assert verdict.recovery_needed is False
        assert verdict.stale_seconds is None
        assert verdict.daemon_pid is None

    def test_frozen(self) -> None:
        verdict = RecoveryVerdict(
            recovery_needed=False,
            reason="No active run",
            interrupted_status=RunStatus.IDLE,
            run_id="test",
            stale_seconds=None,
            has_remote_process=False,
            daemon_pid=None,
        )
        with pytest.raises(AttributeError):
            verdict.recovery_needed = True  # type: ignore[misc]


# -- detect_interrupted_run with None input --


class TestDetectNoneRecord:
    """When no prior record exists, there is nothing to recover."""

    def test_returns_no_recovery(self) -> None:
        verdict = detect_interrupted_run(None)
        assert verdict.recovery_needed is False

    def test_reason_indicates_no_record(self) -> None:
        verdict = detect_interrupted_run(None)
        assert "no" in verdict.reason.lower() or "none" in verdict.reason.lower()

    def test_status_is_idle(self) -> None:
        verdict = detect_interrupted_run(None)
        assert verdict.interrupted_status == RunStatus.IDLE

    def test_run_id_is_empty(self) -> None:
        verdict = detect_interrupted_run(None)
        assert verdict.run_id == ""

    def test_stale_seconds_is_none(self) -> None:
        verdict = detect_interrupted_run(None)
        assert verdict.stale_seconds is None

    def test_no_remote_process(self) -> None:
        verdict = detect_interrupted_run(None)
        assert verdict.has_remote_process is False

    def test_no_daemon_pid(self) -> None:
        verdict = detect_interrupted_run(None)
        assert verdict.daemon_pid is None


# -- detect_interrupted_run with IDLE status --


class TestDetectIdleRecord:
    """An idle record means no run was in progress -- no recovery needed."""

    def test_returns_no_recovery(self) -> None:
        run = CurrentRun(status=RunStatus.IDLE)
        verdict = detect_interrupted_run(run)
        assert verdict.recovery_needed is False

    def test_reason_mentions_idle(self) -> None:
        run = CurrentRun(status=RunStatus.IDLE)
        verdict = detect_interrupted_run(run)
        assert "idle" in verdict.reason.lower()

    def test_preserves_run_id(self) -> None:
        run = CurrentRun(status=RunStatus.IDLE)
        verdict = detect_interrupted_run(run)
        assert verdict.run_id == run.run_id


# -- detect_interrupted_run with terminal statuses --


class TestDetectTerminalStatuses:
    """Completed, failed, and cancelled runs do not need recovery."""

    def test_completed_no_recovery(self) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        final = Progress(percent=100.0, tests_passed=10, tests_total=10)
        run = run.with_completed(final)

        verdict = detect_interrupted_run(run)
        assert verdict.recovery_needed is False
        assert verdict.interrupted_status == RunStatus.COMPLETED

    def test_failed_no_recovery(self) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        error_progress = Progress(tests_failed=1, tests_total=1)
        run = run.with_failed("Connection lost", error_progress)

        verdict = detect_interrupted_run(run)
        assert verdict.recovery_needed is False
        assert verdict.interrupted_status == RunStatus.FAILED

    def test_cancelled_no_recovery(self) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_cancelled()

        verdict = detect_interrupted_run(run)
        assert verdict.recovery_needed is False
        assert verdict.interrupted_status == RunStatus.CANCELLED

    def test_terminal_reason_mentions_status(self) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        final = Progress(percent=100.0, tests_passed=10, tests_total=10)
        run = run.with_completed(final)

        verdict = detect_interrupted_run(run)
        assert "completed" in verdict.reason.lower()


# -- detect_interrupted_run with RUNNING status --


class TestDetectRunningRecord:
    """A RUNNING record means the daemon crashed mid-execution."""

    def _make_running_record(
        self,
        daemon_pid: int = 1234,
        remote_pid: int | None = 5678,
    ) -> CurrentRun:
        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run full regression")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=daemon_pid)
        return run.with_running("pytest --regression", remote_pid=remote_pid)

    def test_returns_recovery_needed(self) -> None:
        run = self._make_running_record()
        verdict = detect_interrupted_run(run)
        assert verdict.recovery_needed is True

    def test_interrupted_status_is_running(self) -> None:
        run = self._make_running_record()
        verdict = detect_interrupted_run(run)
        assert verdict.interrupted_status == RunStatus.RUNNING

    def test_preserves_run_id(self) -> None:
        run = self._make_running_record()
        verdict = detect_interrupted_run(run)
        assert verdict.run_id == run.run_id

    def test_extracts_daemon_pid(self) -> None:
        run = self._make_running_record(daemon_pid=9999)
        verdict = detect_interrupted_run(run)
        assert verdict.daemon_pid == 9999

    def test_has_remote_process_when_pid_set(self) -> None:
        run = self._make_running_record(remote_pid=5555)
        verdict = detect_interrupted_run(run)
        assert verdict.has_remote_process is True

    def test_no_remote_process_when_pid_none(self) -> None:
        run = self._make_running_record(remote_pid=None)
        verdict = detect_interrupted_run(run)
        assert verdict.has_remote_process is False

    def test_reason_mentions_running(self) -> None:
        run = self._make_running_record()
        verdict = detect_interrupted_run(run)
        assert "running" in verdict.reason.lower()

    def test_stale_seconds_is_nonnegative(self) -> None:
        run = self._make_running_record()
        verdict = detect_interrupted_run(run)
        assert verdict.stale_seconds is not None
        assert verdict.stale_seconds >= 0.0


# -- detect_interrupted_run with PENDING_APPROVAL status --


class TestDetectPendingApprovalRecord:
    """A PENDING_APPROVAL record means the daemon stopped before user confirmed."""

    def _make_pending_record(self, daemon_pid: int = 1000) -> CurrentRun:
        target = SSHTarget(host="staging.example.com", user="deploy")
        cmd = Command(natural_language="run smoke tests")
        return CurrentRun().with_pending_approval(target, cmd, daemon_pid=daemon_pid)

    def test_returns_recovery_needed(self) -> None:
        run = self._make_pending_record()
        verdict = detect_interrupted_run(run)
        assert verdict.recovery_needed is True

    def test_interrupted_status_is_pending(self) -> None:
        run = self._make_pending_record()
        verdict = detect_interrupted_run(run)
        assert verdict.interrupted_status == RunStatus.PENDING_APPROVAL

    def test_extracts_daemon_pid(self) -> None:
        run = self._make_pending_record(daemon_pid=4242)
        verdict = detect_interrupted_run(run)
        assert verdict.daemon_pid == 4242

    def test_no_remote_process(self) -> None:
        run = self._make_pending_record()
        verdict = detect_interrupted_run(run)
        assert verdict.has_remote_process is False

    def test_reason_mentions_pending(self) -> None:
        run = self._make_pending_record()
        verdict = detect_interrupted_run(run)
        assert "pending" in verdict.reason.lower()


# -- Staleness computation --


class TestStalenessComputation:
    """Verify that stale_seconds is computed from updated_at to detection time."""

    def test_recent_record_has_small_staleness(self) -> None:
        """A record written just now should have very low staleness."""
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")

        verdict = detect_interrupted_run(run)
        assert verdict.stale_seconds is not None
        # Should be under 2 seconds since we just created it
        assert verdict.stale_seconds < 2.0

    def test_terminal_status_stale_seconds_is_none(self) -> None:
        """Terminal statuses have no meaningful staleness."""
        run = CurrentRun(status=RunStatus.IDLE)
        verdict = detect_interrupted_run(run)
        assert verdict.stale_seconds is None


# -- Integration with wiki persistence --


class TestDetectFromWikiRoundtrip:
    """End-to-end: write a running record, read it back, detect interrupted."""

    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        return tmp_path / "wiki"

    def test_interrupted_run_detected_after_roundtrip(
        self, wiki_root: Path
    ) -> None:
        from jules_daemon.wiki import current_run

        target = SSHTarget(host="prod.example.com", user="ci", port=2222)
        cmd = Command(natural_language="run full regression")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=9876)
        run = run.with_running("pytest -v --regression", remote_pid=5432)
        progress = Progress(
            percent=50.0,
            tests_passed=25,
            tests_failed=1,
            tests_total=50,
            last_output_line="FAILED test_checkout",
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)

        # Simulate new daemon boot: read wiki, detect interrupted
        loaded = current_run.read(wiki_root)
        verdict = detect_interrupted_run(loaded)

        assert verdict.recovery_needed is True
        assert verdict.interrupted_status == RunStatus.RUNNING
        assert verdict.run_id == run.run_id
        assert verdict.has_remote_process is True
        assert verdict.daemon_pid == 9876

    def test_completed_run_not_interrupted_after_roundtrip(
        self, wiki_root: Path
    ) -> None:
        from jules_daemon.wiki import current_run

        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        final = Progress(percent=100.0, tests_passed=10, tests_total=10)
        run = run.with_completed(final)
        current_run.write(wiki_root, run)

        loaded = current_run.read(wiki_root)
        verdict = detect_interrupted_run(loaded)

        assert verdict.recovery_needed is False

    def test_no_wiki_file_not_interrupted(self, wiki_root: Path) -> None:
        from jules_daemon.wiki import current_run

        loaded = current_run.read(wiki_root)
        verdict = detect_interrupted_run(loaded)

        assert verdict.recovery_needed is False
        assert verdict.run_id == ""
