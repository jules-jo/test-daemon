"""Tests for checkpoint extraction logic.

The checkpoint extractor reads the saved run state from the wiki persistence
layer and determines the last completed checkpoint (test index, phase, or
marker). This module tests all scenarios: no file, corrupted file, idle,
pending approval, running at various progress levels, and terminal states.

The extracted checkpoint tells the daemon:
  - test_index: how many tests have been processed (0-based last-completed index)
  - phase: which execution phase the run was in
  - marker: human-readable label for the checkpoint
  - is_resumable: whether the daemon can resume from this checkpoint
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from jules_daemon.wiki import current_run
from jules_daemon.wiki.checkpoint_extractor import (
    Checkpoint,
    CheckpointPhase,
    CheckpointSource,
    _derive_phase,
    extract_checkpoint,
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


# -- Checkpoint model --


class TestCheckpointModel:
    """Verify the Checkpoint dataclass is frozen and has expected properties."""

    def test_frozen(self) -> None:
        cp = Checkpoint(
            test_index=0,
            phase=CheckpointPhase.NOT_STARTED,
            marker="",
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            tests_total=0,
            percent=0.0,
            checkpoint_at=None,
            run_id="",
            status=RunStatus.IDLE,
            source=CheckpointSource.NO_STATE,
            error=None,
        )
        with pytest.raises(AttributeError):
            cp.test_index = 5  # type: ignore[misc]

    def test_tests_completed_property(self) -> None:
        cp = Checkpoint(
            test_index=12,
            phase=CheckpointPhase.RUNNING,
            marker="test_login",
            tests_passed=8,
            tests_failed=2,
            tests_skipped=3,
            tests_total=30,
            percent=43.3,
            checkpoint_at=None,
            run_id="abc",
            status=RunStatus.RUNNING,
            source=CheckpointSource.WIKI_STATE,
            error=None,
        )
        assert cp.tests_completed == 13

    def test_is_resumable_when_running(self) -> None:
        cp = Checkpoint(
            test_index=5,
            phase=CheckpointPhase.RUNNING,
            marker="test_api",
            tests_passed=5,
            tests_failed=0,
            tests_skipped=1,
            tests_total=20,
            percent=30.0,
            checkpoint_at=datetime.now(timezone.utc),
            run_id="abc",
            status=RunStatus.RUNNING,
            source=CheckpointSource.WIKI_STATE,
            error=None,
        )
        assert cp.is_resumable is True

    def test_is_resumable_when_pending_approval(self) -> None:
        cp = Checkpoint(
            test_index=0,
            phase=CheckpointPhase.PENDING_APPROVAL,
            marker="",
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            tests_total=0,
            percent=0.0,
            checkpoint_at=None,
            run_id="abc",
            status=RunStatus.PENDING_APPROVAL,
            source=CheckpointSource.WIKI_STATE,
            error=None,
        )
        assert cp.is_resumable is True

    def test_not_resumable_when_idle(self) -> None:
        cp = Checkpoint(
            test_index=0,
            phase=CheckpointPhase.NOT_STARTED,
            marker="",
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            tests_total=0,
            percent=0.0,
            checkpoint_at=None,
            run_id="",
            status=RunStatus.IDLE,
            source=CheckpointSource.NO_STATE,
            error=None,
        )
        assert cp.is_resumable is False

    def test_not_resumable_when_completed(self) -> None:
        cp = Checkpoint(
            test_index=49,
            phase=CheckpointPhase.COMPLETE,
            marker="all tests finished",
            tests_passed=50,
            tests_failed=0,
            tests_skipped=0,
            tests_total=50,
            percent=100.0,
            checkpoint_at=datetime.now(timezone.utc),
            run_id="abc",
            status=RunStatus.COMPLETED,
            source=CheckpointSource.WIKI_STATE,
            error=None,
        )
        assert cp.is_resumable is False

    def test_not_resumable_when_failed(self) -> None:
        cp = Checkpoint(
            test_index=9,
            phase=CheckpointPhase.FAILED,
            marker="SSH timeout",
            tests_passed=5,
            tests_failed=5,
            tests_skipped=0,
            tests_total=50,
            percent=20.0,
            checkpoint_at=datetime.now(timezone.utc),
            run_id="abc",
            status=RunStatus.FAILED,
            source=CheckpointSource.WIKI_STATE,
            error="SSH connection timeout",
        )
        assert cp.is_resumable is False

    def test_not_resumable_when_cancelled(self) -> None:
        cp = Checkpoint(
            test_index=3,
            phase=CheckpointPhase.CANCELLED,
            marker="user cancelled",
            tests_passed=3,
            tests_failed=0,
            tests_skipped=0,
            tests_total=20,
            percent=15.0,
            checkpoint_at=datetime.now(timezone.utc),
            run_id="abc",
            status=RunStatus.CANCELLED,
            source=CheckpointSource.WIKI_STATE,
            error=None,
        )
        assert cp.is_resumable is False

    def test_not_resumable_when_corrupted(self) -> None:
        cp = Checkpoint(
            test_index=0,
            phase=CheckpointPhase.NOT_STARTED,
            marker="",
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            tests_total=0,
            percent=0.0,
            checkpoint_at=None,
            run_id="",
            status=RunStatus.IDLE,
            source=CheckpointSource.CORRUPTED,
            error="parse error",
        )
        assert cp.is_resumable is False


class TestCheckpointPhaseEnum:
    """Verify all expected phases exist."""

    def test_all_phases_exist(self) -> None:
        assert CheckpointPhase.NOT_STARTED.value == "not_started"
        assert CheckpointPhase.PENDING_APPROVAL.value == "pending_approval"
        assert CheckpointPhase.SETUP.value == "setup"
        assert CheckpointPhase.RUNNING.value == "running"
        assert CheckpointPhase.COMPLETE.value == "complete"
        assert CheckpointPhase.FAILED.value == "failed"
        assert CheckpointPhase.CANCELLED.value == "cancelled"


class TestCheckpointSourceEnum:
    """Verify all expected sources exist."""

    def test_all_sources_exist(self) -> None:
        assert CheckpointSource.WIKI_STATE.value == "wiki_state"
        assert CheckpointSource.NO_STATE.value == "no_state"
        assert CheckpointSource.CORRUPTED.value == "corrupted"


# -- extract_checkpoint: no file --


class TestExtractCheckpointNoFile:
    """When no wiki file exists, return a safe empty checkpoint."""

    def test_returns_not_started_phase(self, wiki_root: Path) -> None:
        cp = extract_checkpoint(wiki_root)
        assert cp.phase == CheckpointPhase.NOT_STARTED

    def test_returns_no_state_source(self, wiki_root: Path) -> None:
        cp = extract_checkpoint(wiki_root)
        assert cp.source == CheckpointSource.NO_STATE

    def test_returns_idle_status(self, wiki_root: Path) -> None:
        cp = extract_checkpoint(wiki_root)
        assert cp.status == RunStatus.IDLE

    def test_returns_zero_test_index(self, wiki_root: Path) -> None:
        cp = extract_checkpoint(wiki_root)
        assert cp.test_index == 0

    def test_returns_zero_progress(self, wiki_root: Path) -> None:
        cp = extract_checkpoint(wiki_root)
        assert cp.tests_passed == 0
        assert cp.tests_failed == 0
        assert cp.tests_skipped == 0
        assert cp.tests_total == 0
        assert cp.percent == 0.0

    def test_returns_empty_marker(self, wiki_root: Path) -> None:
        cp = extract_checkpoint(wiki_root)
        assert cp.marker == ""

    def test_checkpoint_at_is_none(self, wiki_root: Path) -> None:
        cp = extract_checkpoint(wiki_root)
        assert cp.checkpoint_at is None

    def test_error_is_none(self, wiki_root: Path) -> None:
        cp = extract_checkpoint(wiki_root)
        assert cp.error is None

    def test_not_resumable(self, wiki_root: Path) -> None:
        cp = extract_checkpoint(wiki_root)
        assert cp.is_resumable is False


# -- extract_checkpoint: corrupted file --


class TestExtractCheckpointCorrupted:
    """When wiki file is corrupted, return a safe fallback checkpoint."""

    def test_invalid_yaml_returns_corrupted(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("not valid yaml frontmatter", encoding="utf-8")

        cp = extract_checkpoint(wiki_root)
        assert cp.source == CheckpointSource.CORRUPTED

    def test_corrupted_has_error_detail(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("garbage content", encoding="utf-8")

        cp = extract_checkpoint(wiki_root)
        assert cp.error is not None
        assert len(cp.error) > 0

    def test_corrupted_returns_idle_status(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "---\nstatus: invalid_status\n---\nBody", encoding="utf-8"
        )

        cp = extract_checkpoint(wiki_root)
        assert cp.source == CheckpointSource.CORRUPTED
        assert cp.status == RunStatus.IDLE

    def test_corrupted_not_resumable(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("", encoding="utf-8")

        cp = extract_checkpoint(wiki_root)
        assert cp.is_resumable is False

    def test_empty_file_returns_corrupted(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("", encoding="utf-8")

        cp = extract_checkpoint(wiki_root)
        assert cp.source == CheckpointSource.CORRUPTED


# -- extract_checkpoint: idle state --


class TestExtractCheckpointIdle:
    """When wiki contains an idle record, return a not-started checkpoint."""

    def test_idle_returns_not_started(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        cp = extract_checkpoint(wiki_root)
        assert cp.phase == CheckpointPhase.NOT_STARTED

    def test_idle_returns_wiki_state_source(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        cp = extract_checkpoint(wiki_root)
        assert cp.source == CheckpointSource.WIKI_STATE

    def test_idle_returns_zero_index(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        cp = extract_checkpoint(wiki_root)
        assert cp.test_index == 0

    def test_idle_not_resumable(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        cp = extract_checkpoint(wiki_root)
        assert cp.is_resumable is False


# -- extract_checkpoint: pending approval --


class TestExtractCheckpointPendingApproval:
    """When wiki contains a pending-approval record."""

    def _write_pending(self, wiki_root: Path) -> CurrentRun:
        target = SSHTarget(host="staging.example.com", user="deploy")
        cmd = Command(natural_language="run the smoke tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        current_run.write(wiki_root, run)
        return run

    def test_returns_pending_approval_phase(self, wiki_root: Path) -> None:
        self._write_pending(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.phase == CheckpointPhase.PENDING_APPROVAL

    def test_returns_zero_test_index(self, wiki_root: Path) -> None:
        self._write_pending(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.test_index == 0

    def test_preserves_run_id(self, wiki_root: Path) -> None:
        run = self._write_pending(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.run_id == run.run_id

    def test_is_resumable(self, wiki_root: Path) -> None:
        self._write_pending(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.is_resumable is True

    def test_marker_is_empty_before_execution(self, wiki_root: Path) -> None:
        self._write_pending(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.marker == ""

    def test_returns_wiki_state_source(self, wiki_root: Path) -> None:
        self._write_pending(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.source == CheckpointSource.WIKI_STATE


# -- extract_checkpoint: running with progress --


class TestExtractCheckpointRunning:
    """When wiki contains a running record with test progress."""

    def _write_running_with_progress(
        self,
        wiki_root: Path,
        *,
        percent: float = 50.0,
        passed: int = 10,
        failed: int = 2,
        skipped: int = 1,
        total: int = 30,
        last_line: str = "PASSED test_checkout_flow",
        checkpoint_at: datetime | None = None,
    ) -> CurrentRun:
        target = SSHTarget(host="prod.example.com", user="ci", port=2222)
        cmd = Command(natural_language="run full regression")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=9876)
        run = run.with_running("pytest -v --regression", remote_pid=5432)
        progress = Progress(
            percent=percent,
            tests_passed=passed,
            tests_failed=failed,
            tests_skipped=skipped,
            tests_total=total,
            last_output_line=last_line,
            checkpoint_at=checkpoint_at,
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)
        return run

    def test_returns_running_phase(self, wiki_root: Path) -> None:
        self._write_running_with_progress(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.phase == CheckpointPhase.RUNNING

    def test_test_index_is_completed_count_minus_one(self, wiki_root: Path) -> None:
        """test_index is 0-based: (passed+failed+skipped) - 1."""
        self._write_running_with_progress(wiki_root, passed=10, failed=2, skipped=1)
        cp = extract_checkpoint(wiki_root)
        # 10 + 2 + 1 = 13 completed; 0-based index = 12
        assert cp.test_index == 12

    def test_test_index_zero_when_no_tests_completed(self, wiki_root: Path) -> None:
        """When no tests have completed yet, test_index is 0."""
        self._write_running_with_progress(
            wiki_root, passed=0, failed=0, skipped=0, percent=0.0, last_line=""
        )
        cp = extract_checkpoint(wiki_root)
        assert cp.test_index == 0

    def test_test_index_zero_when_exactly_one_test_completed(
        self, wiki_root: Path
    ) -> None:
        """When exactly one test completed, test_index is 0 (same as no-tests).

        Callers must check tests_completed to distinguish "no tests done"
        (tests_completed == 0) from "one test done" (tests_completed == 1).
        """
        self._write_running_with_progress(
            wiki_root, passed=1, failed=0, skipped=0, percent=5.0, total=20
        )
        cp = extract_checkpoint(wiki_root)
        assert cp.test_index == 0
        assert cp.tests_completed == 1

    def test_extracts_progress_fields(self, wiki_root: Path) -> None:
        self._write_running_with_progress(
            wiki_root, passed=15, failed=3, skipped=2, total=50, percent=40.0
        )
        cp = extract_checkpoint(wiki_root)
        assert cp.tests_passed == 15
        assert cp.tests_failed == 3
        assert cp.tests_skipped == 2
        assert cp.tests_total == 50
        assert cp.percent == 40.0

    def test_marker_from_last_output_line(self, wiki_root: Path) -> None:
        self._write_running_with_progress(
            wiki_root, last_line="PASSED test_checkout_flow"
        )
        cp = extract_checkpoint(wiki_root)
        assert cp.marker == "PASSED test_checkout_flow"

    def test_preserves_checkpoint_at_timestamp(self, wiki_root: Path) -> None:
        ts = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        self._write_running_with_progress(wiki_root, checkpoint_at=ts)
        cp = extract_checkpoint(wiki_root)
        assert cp.checkpoint_at is not None
        assert cp.checkpoint_at == ts

    def test_is_resumable(self, wiki_root: Path) -> None:
        self._write_running_with_progress(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.is_resumable is True

    def test_preserves_run_id(self, wiki_root: Path) -> None:
        run = self._write_running_with_progress(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.run_id == run.run_id

    def test_setup_phase_when_zero_percent(self, wiki_root: Path) -> None:
        """Running with 0% and no completed tests implies setup phase."""
        self._write_running_with_progress(
            wiki_root,
            percent=0.0,
            passed=0,
            failed=0,
            skipped=0,
            total=0,
            last_line="",
        )
        cp = extract_checkpoint(wiki_root)
        assert cp.phase == CheckpointPhase.SETUP


# -- extract_checkpoint: completed --


class TestExtractCheckpointCompleted:
    """When wiki contains a completed record."""

    def _write_completed(self, wiki_root: Path) -> CurrentRun:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        final = Progress(
            percent=100.0,
            tests_passed=48,
            tests_failed=0,
            tests_skipped=2,
            tests_total=50,
            last_output_line="50 passed, 2 skipped",
        )
        run = run.with_completed(final)
        current_run.write(wiki_root, run)
        return run

    def test_returns_complete_phase(self, wiki_root: Path) -> None:
        self._write_completed(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.phase == CheckpointPhase.COMPLETE

    def test_test_index_is_last_completed(self, wiki_root: Path) -> None:
        self._write_completed(wiki_root)
        cp = extract_checkpoint(wiki_root)
        # 48 passed + 0 failed + 2 skipped = 50; 0-based index = 49
        assert cp.test_index == 49

    def test_marker_from_last_output(self, wiki_root: Path) -> None:
        self._write_completed(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.marker == "50 passed, 2 skipped"

    def test_not_resumable(self, wiki_root: Path) -> None:
        self._write_completed(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.is_resumable is False

    def test_wiki_state_source(self, wiki_root: Path) -> None:
        self._write_completed(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.source == CheckpointSource.WIKI_STATE


# -- extract_checkpoint: failed --


class TestExtractCheckpointFailed:
    """When wiki contains a failed record with partial progress."""

    def _write_failed(self, wiki_root: Path) -> CurrentRun:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        error_progress = Progress(
            percent=20.0,
            tests_passed=8,
            tests_failed=2,
            tests_skipped=0,
            tests_total=50,
            last_output_line="FAILED test_payment_flow",
        )
        run = run.with_failed("SSH connection timeout after 30s", error_progress)
        current_run.write(wiki_root, run)
        return run

    def test_returns_failed_phase(self, wiki_root: Path) -> None:
        self._write_failed(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.phase == CheckpointPhase.FAILED

    def test_test_index_reflects_partial_progress(self, wiki_root: Path) -> None:
        self._write_failed(wiki_root)
        cp = extract_checkpoint(wiki_root)
        # 8 passed + 2 failed + 0 skipped = 10; 0-based index = 9
        assert cp.test_index == 9

    def test_preserves_error(self, wiki_root: Path) -> None:
        self._write_failed(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.error == "SSH connection timeout after 30s"

    def test_not_resumable(self, wiki_root: Path) -> None:
        self._write_failed(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.is_resumable is False


# -- extract_checkpoint: cancelled --


class TestExtractCheckpointCancelled:
    """When wiki contains a cancelled record."""

    def _write_cancelled(self, wiki_root: Path) -> CurrentRun:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        run = run.with_progress(
            Progress(
                percent=15.0,
                tests_passed=3,
                tests_failed=0,
                tests_skipped=0,
                tests_total=20,
            )
        )
        run = run.with_cancelled()
        current_run.write(wiki_root, run)
        return run

    def test_returns_cancelled_phase(self, wiki_root: Path) -> None:
        self._write_cancelled(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.phase == CheckpointPhase.CANCELLED

    def test_preserves_partial_progress(self, wiki_root: Path) -> None:
        self._write_cancelled(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.tests_passed == 3
        assert cp.tests_total == 20

    def test_not_resumable(self, wiki_root: Path) -> None:
        self._write_cancelled(wiki_root)
        cp = extract_checkpoint(wiki_root)
        assert cp.is_resumable is False


# -- extract_checkpoint: never raises --


class TestExtractCheckpointNeverRaises:
    """The extraction function must never raise exceptions."""

    def test_survives_directory_not_existing(self) -> None:
        cp = extract_checkpoint(Path("/nonexistent/wiki/root"))
        assert cp.source == CheckpointSource.NO_STATE

    def test_survives_corrupted_yaml(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "---\n!!python/object:os.system\n---\n", encoding="utf-8"
        )

        cp = extract_checkpoint(wiki_root)
        # Should not raise, should return corrupted
        assert cp.source in (CheckpointSource.CORRUPTED, CheckpointSource.NO_STATE)

    def test_survives_permission_error(self, wiki_root: Path) -> None:
        """PermissionError on file read returns CORRUPTED, does not raise."""
        # Write a valid state so the file exists and passes the exists() check
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        with patch.object(
            current_run,
            "read",
            side_effect=PermissionError("Permission denied: current-run.md"),
        ):
            cp = extract_checkpoint(wiki_root)

        assert cp.source == CheckpointSource.CORRUPTED
        assert cp.error is not None
        assert "Permission denied" in cp.error


# -- _derive_phase: internal edge cases --


class TestDerivePhase:
    """Test the internal _derive_phase function directly."""

    def test_unknown_status_returns_not_started(self) -> None:
        """A fabricated unknown status should default to NOT_STARTED."""
        # Monkeypatch RunStatus to simulate an unknown member. Since RunStatus
        # is an Enum, we mock the status field on the CurrentRun directly.
        run = CurrentRun(status=RunStatus.IDLE)
        # Create a mock status that is not in the known set
        mock_status = "totally_unknown"

        with patch.object(type(run), "status", new_callable=lambda: property(lambda self: mock_status)):
            result = _derive_phase(run)

        assert result == CheckpointPhase.NOT_STARTED


# -- extract_checkpoint: boundary cases --


class TestExtractCheckpointBoundary:
    """Edge cases and boundary conditions."""

    def test_single_test_passed_index_zero_with_one_completed(
        self, wiki_root: Path
    ) -> None:
        """When exactly 1 test passed, test_index=0 but tests_completed=1."""
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        progress = Progress(
            percent=10.0,
            tests_passed=1,
            tests_failed=0,
            tests_skipped=0,
            tests_total=10,
            last_output_line="PASSED test_first",
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)

        cp = extract_checkpoint(wiki_root)
        assert cp.test_index == 0
        assert cp.tests_completed == 1
        # Callers should distinguish from no-tests case by checking tests_completed
        assert cp.tests_completed > 0

    def test_all_tests_skipped(self, wiki_root: Path) -> None:
        """All tests skipped still produces a valid checkpoint."""
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        progress = Progress(
            percent=100.0,
            tests_passed=0,
            tests_failed=0,
            tests_skipped=5,
            tests_total=5,
            last_output_line="5 skipped",
        )
        run = run.with_completed(progress)
        current_run.write(wiki_root, run)

        cp = extract_checkpoint(wiki_root)
        assert cp.test_index == 4
        assert cp.tests_completed == 5
        assert cp.phase == CheckpointPhase.COMPLETE


# -- extract_checkpoint: performance --


class TestExtractCheckpointPerformance:
    """Extraction must be fast enough for the 30s crash recovery SLA."""

    def test_completes_under_100ms(self, wiki_root: Path) -> None:
        target = SSHTarget(host="prod.example.com", user="ci", port=2222)
        cmd = Command(natural_language="run full regression suite")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=9999)
        run = run.with_running("pytest -v --regression", remote_pid=8888)
        progress = Progress(
            percent=75.0,
            tests_passed=150,
            tests_failed=3,
            tests_skipped=5,
            tests_total=200,
            last_output_line="FAILED test_payment_flow",
            checkpoint_at=datetime.now(timezone.utc),
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)

        start = time.monotonic()
        cp = extract_checkpoint(wiki_root)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 100.0, f"Extraction took {elapsed_ms:.1f}ms (>100ms)"
        assert cp.test_index == 157  # 150 + 3 + 5 - 1 = 157
