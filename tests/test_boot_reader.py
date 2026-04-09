"""Tests for wiki record reader used during daemon boot.

Verifies that the boot reader:
- Loads the current-run wiki file and extracts status fields
- Handles missing files (returns idle BootRecord)
- Handles corrupted files (returns error BootRecord)
- Preserves all run status fields: status, start_time, end_time, error
- Supports crash recovery (30s SLA)
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.wiki.boot_reader import (
    BootRecord,
    LoadOutcome,
    load_boot_record,
)
from jules_daemon.wiki import current_run
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


# -- BootRecord --


class TestBootRecord:
    def test_create_with_required_fields(self) -> None:
        record = BootRecord(
            status=RunStatus.IDLE,
            run_id="test-id",
            started_at=None,
            completed_at=None,
            error=None,
            outcome=LoadOutcome.LOADED,
            source_path=Path("/tmp/test.md"),
            loaded_at=datetime.now(timezone.utc),
        )
        assert record.status == RunStatus.IDLE
        assert record.run_id == "test-id"

    def test_frozen(self) -> None:
        record = BootRecord(
            status=RunStatus.IDLE,
            run_id="test-id",
            started_at=None,
            completed_at=None,
            error=None,
            outcome=LoadOutcome.LOADED,
            source_path=Path("/tmp/test.md"),
            loaded_at=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            record.status = RunStatus.RUNNING  # type: ignore[misc]

    def test_is_active_when_running(self) -> None:
        record = BootRecord(
            status=RunStatus.RUNNING,
            run_id="abc",
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            error=None,
            outcome=LoadOutcome.LOADED,
            source_path=Path("/tmp/test.md"),
            loaded_at=datetime.now(timezone.utc),
        )
        assert record.is_active is True

    def test_is_active_when_pending_approval(self) -> None:
        record = BootRecord(
            status=RunStatus.PENDING_APPROVAL,
            run_id="abc",
            started_at=None,
            completed_at=None,
            error=None,
            outcome=LoadOutcome.LOADED,
            source_path=Path("/tmp/test.md"),
            loaded_at=datetime.now(timezone.utc),
        )
        assert record.is_active is True

    def test_is_not_active_when_idle(self) -> None:
        record = BootRecord(
            status=RunStatus.IDLE,
            run_id="abc",
            started_at=None,
            completed_at=None,
            error=None,
            outcome=LoadOutcome.LOADED,
            source_path=Path("/tmp/test.md"),
            loaded_at=datetime.now(timezone.utc),
        )
        assert record.is_active is False

    def test_is_not_active_when_completed(self) -> None:
        record = BootRecord(
            status=RunStatus.COMPLETED,
            run_id="abc",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            error=None,
            outcome=LoadOutcome.LOADED,
            source_path=Path("/tmp/test.md"),
            loaded_at=datetime.now(timezone.utc),
        )
        assert record.is_active is False

    def test_needs_recovery_when_running(self) -> None:
        record = BootRecord(
            status=RunStatus.RUNNING,
            run_id="abc",
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            error=None,
            outcome=LoadOutcome.LOADED,
            source_path=Path("/tmp/test.md"),
            loaded_at=datetime.now(timezone.utc),
        )
        assert record.needs_recovery is True

    def test_needs_recovery_when_pending_approval(self) -> None:
        record = BootRecord(
            status=RunStatus.PENDING_APPROVAL,
            run_id="abc",
            started_at=None,
            completed_at=None,
            error=None,
            outcome=LoadOutcome.LOADED,
            source_path=Path("/tmp/test.md"),
            loaded_at=datetime.now(timezone.utc),
        )
        assert record.needs_recovery is True

    def test_no_recovery_needed_when_idle(self) -> None:
        record = BootRecord(
            status=RunStatus.IDLE,
            run_id="abc",
            started_at=None,
            completed_at=None,
            error=None,
            outcome=LoadOutcome.NO_FILE,
            source_path=None,
            loaded_at=datetime.now(timezone.utc),
        )
        assert record.needs_recovery is False

    def test_no_recovery_needed_when_terminal(self) -> None:
        for terminal_status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            record = BootRecord(
                status=terminal_status,
                run_id="abc",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                error=None,
                outcome=LoadOutcome.LOADED,
                source_path=Path("/tmp/test.md"),
                loaded_at=datetime.now(timezone.utc),
            )
            assert record.needs_recovery is False


class TestLoadOutcome:
    def test_all_outcomes_exist(self) -> None:
        assert LoadOutcome.LOADED.value == "loaded"
        assert LoadOutcome.NO_FILE.value == "no_file"
        assert LoadOutcome.CORRUPTED.value == "corrupted"


# -- load_boot_record --


class TestLoadBootRecordNoFile:
    """When no wiki file exists, return a safe idle BootRecord."""

    def test_returns_idle_status(self, wiki_root: Path) -> None:
        record = load_boot_record(wiki_root)
        assert record.status == RunStatus.IDLE

    def test_outcome_is_no_file(self, wiki_root: Path) -> None:
        record = load_boot_record(wiki_root)
        assert record.outcome == LoadOutcome.NO_FILE

    def test_source_path_is_none(self, wiki_root: Path) -> None:
        record = load_boot_record(wiki_root)
        assert record.source_path is None

    def test_status_fields_are_none(self, wiki_root: Path) -> None:
        record = load_boot_record(wiki_root)
        assert record.started_at is None
        assert record.completed_at is None
        assert record.error is None

    def test_has_loaded_at_timestamp(self, wiki_root: Path) -> None:
        before = datetime.now(timezone.utc)
        record = load_boot_record(wiki_root)
        after = datetime.now(timezone.utc)
        assert before <= record.loaded_at <= after

    def test_has_run_id(self, wiki_root: Path) -> None:
        record = load_boot_record(wiki_root)
        assert record.run_id  # non-empty string


class TestLoadBootRecordIdleFile:
    """When wiki file exists with idle state."""

    def test_reads_idle_state(self, wiki_root: Path) -> None:
        idle_run = CurrentRun(status=RunStatus.IDLE)
        current_run.write(wiki_root, idle_run)

        record = load_boot_record(wiki_root)
        assert record.status == RunStatus.IDLE
        assert record.outcome == LoadOutcome.LOADED
        assert record.run_id == idle_run.run_id

    def test_source_path_set(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun())

        record = load_boot_record(wiki_root)
        assert record.source_path is not None
        assert record.source_path.name == "current-run.md"


class TestLoadBootRecordRunningState:
    """When wiki file contains a running test -- crash recovery scenario."""

    def test_extracts_running_status(self, wiki_root: Path) -> None:
        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run full regression")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        run = run.with_running("pytest --regression", remote_pid=5678)
        current_run.write(wiki_root, run)

        record = load_boot_record(wiki_root)
        assert record.status == RunStatus.RUNNING

    def test_extracts_started_at(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        current_run.write(wiki_root, run)

        record = load_boot_record(wiki_root)
        assert record.started_at is not None

    def test_completed_at_is_none(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        current_run.write(wiki_root, run)

        record = load_boot_record(wiki_root)
        assert record.completed_at is None

    def test_needs_recovery(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        current_run.write(wiki_root, run)

        record = load_boot_record(wiki_root)
        assert record.needs_recovery is True

    def test_preserves_run_id(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        current_run.write(wiki_root, run)

        record = load_boot_record(wiki_root)
        assert record.run_id == run.run_id


class TestLoadBootRecordFailedState:
    """When wiki file contains a failed run with error info."""

    def test_extracts_failed_status(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        error_progress = Progress(tests_passed=0, tests_failed=1, tests_total=1)
        run = run.with_failed("SSH connection timeout after 30s", error_progress)
        current_run.write(wiki_root, run)

        record = load_boot_record(wiki_root)
        assert record.status == RunStatus.FAILED

    def test_extracts_error_message(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        error_progress = Progress(tests_passed=0, tests_failed=1, tests_total=1)
        run = run.with_failed("SSH connection timeout after 30s", error_progress)
        current_run.write(wiki_root, run)

        record = load_boot_record(wiki_root)
        assert record.error == "SSH connection timeout after 30s"

    def test_extracts_completed_at(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        error_progress = Progress(tests_passed=0, tests_failed=1, tests_total=1)
        run = run.with_failed("SSH connection timeout after 30s", error_progress)
        current_run.write(wiki_root, run)

        record = load_boot_record(wiki_root)
        assert record.completed_at is not None


class TestLoadBootRecordCompletedState:
    """When wiki file contains a completed run."""

    def test_extracts_completed_status(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        final_progress = Progress(
            percent=100.0, tests_passed=50, tests_failed=0, tests_total=50
        )
        run = run.with_completed(final_progress)
        current_run.write(wiki_root, run)

        record = load_boot_record(wiki_root)
        assert record.status == RunStatus.COMPLETED
        assert record.completed_at is not None
        assert record.error is None
        assert record.needs_recovery is False


class TestLoadBootRecordCorruptedFile:
    """When wiki file is corrupted or has invalid content."""

    def test_invalid_yaml_returns_corrupted(self, wiki_root: Path) -> None:
        """Corrupted YAML should return a safe idle BootRecord."""
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("not valid yaml frontmatter at all", encoding="utf-8")

        record = load_boot_record(wiki_root)
        assert record.outcome == LoadOutcome.CORRUPTED
        assert record.status == RunStatus.IDLE

    def test_corrupted_has_error_detail(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("not valid yaml", encoding="utf-8")

        record = load_boot_record(wiki_root)
        assert record.error is not None
        assert len(record.error) > 0

    def test_malformed_frontmatter_returns_corrupted(self, wiki_root: Path) -> None:
        """Valid YAML but missing required fields should be handled."""
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "---\nstatus: not_a_valid_status\n---\nBody",
            encoding="utf-8",
        )

        record = load_boot_record(wiki_root)
        assert record.outcome == LoadOutcome.CORRUPTED
        assert record.status == RunStatus.IDLE
        assert record.error is not None

    def test_empty_file_returns_corrupted(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("", encoding="utf-8")

        record = load_boot_record(wiki_root)
        assert record.outcome == LoadOutcome.CORRUPTED
        assert record.status == RunStatus.IDLE

    def test_corrupted_source_path_still_set(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("garbage", encoding="utf-8")

        record = load_boot_record(wiki_root)
        assert record.source_path is not None
        assert record.source_path.exists()


class TestLoadBootRecordPerformance:
    """Verify the boot record loads fast enough for the 30s recovery SLA."""

    def test_load_completes_under_100ms(self, wiki_root: Path) -> None:
        """Boot record read should be well under the 30s recovery window."""
        import time

        target = SSHTarget(host="prod.example.com", user="ci", port=2222)
        cmd = Command(natural_language="run the full regression suite")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=9999)
        run = run.with_running("pytest -v --regression", remote_pid=8888)
        progress = Progress(
            percent=75.0,
            tests_passed=150,
            tests_failed=3,
            tests_skipped=5,
            tests_total=200,
            last_output_line="FAILED test_payment_flow",
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)

        start = time.monotonic()
        record = load_boot_record(wiki_root)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 100.0, f"Boot record load took {elapsed_ms:.1f}ms (>100ms)"
        assert record.status == RunStatus.RUNNING
