"""Tests for run promotion -- current-run record to completed history.

When a test run reaches a terminal state (completed, failed, or cancelled),
the current-run record is promoted to a timestamped history entry in the wiki,
and the current-run record is reset to idle.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.wiki import current_run
from jules_daemon.wiki.frontmatter import parse
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    ProcessIDs,
    Progress,
    RunStatus,
    SSHTarget,
)
from jules_daemon.wiki.run_promotion import (
    PromotionResult,
    list_history,
    promote_run,
    read_history_entry,
)


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


def _make_completed_run() -> CurrentRun:
    """Build a fully-populated completed run for testing."""
    target = SSHTarget(host="staging.example.com", user="deploy", port=2222)
    cmd = Command(natural_language="run the full test suite")
    run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=9876)
    run = run.with_running("pytest -v --tb=short", remote_pid=5432)
    final_progress = Progress(
        percent=100.0,
        tests_passed=25,
        tests_failed=0,
        tests_skipped=2,
        tests_total=27,
        last_output_line="27 passed, 2 skipped in 45.2s",
    )
    return run.with_completed(final_progress)


def _make_failed_run() -> CurrentRun:
    """Build a failed run for testing."""
    target = SSHTarget(host="prod.example.com", user="ci")
    cmd = Command(natural_language="run regression tests")
    run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
    run = run.with_running("pytest --regression", remote_pid=5678)
    final_progress = Progress(
        percent=80.0,
        tests_passed=16,
        tests_failed=4,
        tests_total=20,
        last_output_line="FAILED test_payment_flow",
    )
    return run.with_failed("4 tests failed", final_progress)


def _make_cancelled_run() -> CurrentRun:
    """Build a cancelled run for testing."""
    target = SSHTarget(host="dev.example.com", user="tester")
    cmd = Command(natural_language="run smoke tests")
    run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=4321)
    run = run.with_running("pytest -m smoke", remote_pid=8765)
    return run.with_cancelled()


class TestPromoteRun:
    """Tests for the promote_run function."""

    def test_completed_run_creates_history_file(self, wiki_root: Path) -> None:
        """A completed run produces a history wiki file."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        result = promote_run(wiki_root, completed)

        assert result.history_path.exists()
        assert result.history_path.suffix == ".md"
        assert "history" in str(result.history_path)

    def test_completed_run_resets_current_to_idle(self, wiki_root: Path) -> None:
        """After promotion, current-run record is reset to idle."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        promote_run(wiki_root, completed)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.IDLE
        assert loaded.ssh_target is None
        assert loaded.command is None

    def test_failed_run_promotes_to_history(self, wiki_root: Path) -> None:
        """Failed runs are also promoted to history."""
        failed = _make_failed_run()
        current_run.write(wiki_root, failed)

        result = promote_run(wiki_root, failed)

        assert result.history_path.exists()
        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.IDLE

    def test_cancelled_run_promotes_to_history(self, wiki_root: Path) -> None:
        """Cancelled runs are also promoted to history."""
        cancelled = _make_cancelled_run()
        current_run.write(wiki_root, cancelled)

        result = promote_run(wiki_root, cancelled)

        assert result.history_path.exists()
        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.IDLE

    def test_history_file_contains_valid_frontmatter(self, wiki_root: Path) -> None:
        """History file has valid YAML frontmatter with expected fields."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        result = promote_run(wiki_root, completed)

        raw = result.history_path.read_text(encoding="utf-8")
        doc = parse(raw)

        assert doc.frontmatter["type"] == "run-history"
        assert "run-history" in doc.frontmatter["tags"]
        assert doc.frontmatter["status"] == "completed"
        assert doc.frontmatter["run_id"] == completed.run_id

    def test_history_preserves_ssh_target(self, wiki_root: Path) -> None:
        """History entry preserves SSH target details."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        result = promote_run(wiki_root, completed)

        raw = result.history_path.read_text(encoding="utf-8")
        doc = parse(raw)

        ssh = doc.frontmatter["ssh_target"]
        assert ssh["host"] == "staging.example.com"
        assert ssh["user"] == "deploy"
        assert ssh["port"] == 2222

    def test_history_preserves_command(self, wiki_root: Path) -> None:
        """History entry preserves command details."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        result = promote_run(wiki_root, completed)

        raw = result.history_path.read_text(encoding="utf-8")
        doc = parse(raw)

        cmd = doc.frontmatter["command"]
        assert cmd["natural_language"] == "run the full test suite"
        assert cmd["resolved_shell"] == "pytest -v --tb=short"
        assert cmd["approved"] is True

    def test_history_preserves_final_progress(self, wiki_root: Path) -> None:
        """History entry preserves final test progress."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        result = promote_run(wiki_root, completed)

        raw = result.history_path.read_text(encoding="utf-8")
        doc = parse(raw)

        prog = doc.frontmatter["progress"]
        assert prog["tests_passed"] == 25
        assert prog["tests_failed"] == 0
        assert prog["tests_skipped"] == 2
        assert prog["tests_total"] == 27
        assert prog["percent"] == 100.0

    def test_history_preserves_error_on_failure(self, wiki_root: Path) -> None:
        """History entry preserves the error message for failed runs."""
        failed = _make_failed_run()
        current_run.write(wiki_root, failed)

        result = promote_run(wiki_root, failed)

        raw = result.history_path.read_text(encoding="utf-8")
        doc = parse(raw)

        assert doc.frontmatter["error"] == "4 tests failed"
        assert doc.frontmatter["status"] == "failed"

    def test_history_preserves_timestamps(self, wiki_root: Path) -> None:
        """History entry preserves started_at and completed_at."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        result = promote_run(wiki_root, completed)

        raw = result.history_path.read_text(encoding="utf-8")
        doc = parse(raw)

        assert doc.frontmatter["started_at"] is not None
        assert doc.frontmatter["completed_at"] is not None

    def test_history_filename_contains_run_id(self, wiki_root: Path) -> None:
        """History filename includes the run_id for uniqueness."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        result = promote_run(wiki_root, completed)

        assert completed.run_id in result.history_path.name

    def test_history_file_in_correct_directory(self, wiki_root: Path) -> None:
        """History files are stored in pages/daemon/history/."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        result = promote_run(wiki_root, completed)

        expected_dir = wiki_root / "pages" / "daemon" / "history"
        assert result.history_path.parent == expected_dir

    def test_history_body_contains_summary(self, wiki_root: Path) -> None:
        """History file markdown body contains a human-readable summary."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        result = promote_run(wiki_root, completed)

        raw = result.history_path.read_text(encoding="utf-8")
        doc = parse(raw)

        assert "# Run History" in doc.body
        assert "staging.example.com" in doc.body
        assert "completed" in doc.body

    def test_promotion_result_has_run_id(self, wiki_root: Path) -> None:
        """PromotionResult carries the run_id of the promoted run."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        result = promote_run(wiki_root, completed)

        assert result.run_id == completed.run_id

    def test_promotion_result_has_status(self, wiki_root: Path) -> None:
        """PromotionResult carries the terminal status."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        result = promote_run(wiki_root, completed)

        assert result.final_status == RunStatus.COMPLETED

    def test_reject_non_terminal_run(self, wiki_root: Path) -> None:
        """Cannot promote a run that is still active."""
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        running = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        running = running.with_running("pytest")
        current_run.write(wiki_root, running)

        with pytest.raises(ValueError, match="terminal"):
            promote_run(wiki_root, running)

    def test_reject_idle_run(self, wiki_root: Path) -> None:
        """Cannot promote an idle run."""
        idle = CurrentRun()
        current_run.write(wiki_root, idle)

        with pytest.raises(ValueError, match="terminal"):
            promote_run(wiki_root, idle)

    def test_multiple_promotions_create_separate_files(
        self, wiki_root: Path
    ) -> None:
        """Each promotion creates a unique history file."""
        run1 = _make_completed_run()
        current_run.write(wiki_root, run1)
        result1 = promote_run(wiki_root, run1)

        run2 = _make_failed_run()
        current_run.write(wiki_root, run2)
        result2 = promote_run(wiki_root, run2)

        assert result1.history_path != result2.history_path
        assert result1.history_path.exists()
        assert result2.history_path.exists()

    def test_atomic_write_no_partial_files(self, wiki_root: Path) -> None:
        """History file appears only after a complete write (no .tmp residue)."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        promote_run(wiki_root, completed)

        history_dir = wiki_root / "pages" / "daemon" / "history"
        tmp_files = list(history_dir.glob("*.tmp"))
        assert tmp_files == []

    def test_promotion_result_has_promoted_at(self, wiki_root: Path) -> None:
        """PromotionResult carries the timestamp of promotion."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)

        before = datetime.now(timezone.utc)
        result = promote_run(wiki_root, completed)

        assert result.promoted_at >= before


class TestListHistory:
    """Tests for listing history entries."""

    def test_empty_when_no_history(self, wiki_root: Path) -> None:
        """No history entries when no runs have been promoted."""
        entries = list_history(wiki_root)
        assert entries == []

    def test_returns_entries_after_promotion(self, wiki_root: Path) -> None:
        """History entries appear after promoting runs."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)
        promote_run(wiki_root, completed)

        entries = list_history(wiki_root)
        assert len(entries) == 1
        assert entries[0].run_id == completed.run_id

    def test_multiple_entries_sorted_newest_first(self, wiki_root: Path) -> None:
        """Multiple history entries are sorted with newest first."""
        run1 = _make_completed_run()
        current_run.write(wiki_root, run1)
        promote_run(wiki_root, run1)

        run2 = _make_failed_run()
        current_run.write(wiki_root, run2)
        promote_run(wiki_root, run2)

        entries = list_history(wiki_root)
        assert len(entries) == 2
        # Newest first
        assert entries[0].run_id == run2.run_id
        assert entries[1].run_id == run1.run_id

    def test_entries_have_status_and_run_id(self, wiki_root: Path) -> None:
        """Each history entry has status and run_id from frontmatter."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)
        promote_run(wiki_root, completed)

        entries = list_history(wiki_root)
        assert entries[0].status == RunStatus.COMPLETED
        assert entries[0].run_id == completed.run_id
        assert entries[0].file_path.exists()


class TestReadHistoryEntry:
    """Tests for reading a single history entry."""

    def test_read_promoted_entry(self, wiki_root: Path) -> None:
        """A promoted history entry can be read back fully."""
        completed = _make_completed_run()
        current_run.write(wiki_root, completed)
        result = promote_run(wiki_root, completed)

        entry = read_history_entry(result.history_path)

        assert entry is not None
        assert entry.status == RunStatus.COMPLETED
        assert entry.run_id == completed.run_id
        assert entry.ssh_target is not None
        assert entry.ssh_target.host == "staging.example.com"
        assert entry.command is not None
        assert entry.command.natural_language == "run the full test suite"
        assert entry.progress.tests_passed == 25

    def test_read_nonexistent_returns_none(self, wiki_root: Path) -> None:
        """Reading a nonexistent history file returns None."""
        fake = wiki_root / "pages" / "daemon" / "history" / "does-not-exist.md"
        entry = read_history_entry(fake)
        assert entry is None
