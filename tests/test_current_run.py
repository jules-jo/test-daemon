"""Tests for current-run wiki persistence functions."""

from pathlib import Path

import pytest

from jules_daemon.wiki import current_run
from jules_daemon.wiki.frontmatter import parse
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


# -- write --


class TestWrite:
    def test_creates_file_and_directories(self, wiki_root: Path) -> None:
        run = CurrentRun()
        result_path = current_run.write(wiki_root, run)

        assert result_path.exists()
        assert result_path.name == "current-run.md"
        assert "pages/daemon" in str(result_path)

    def test_idle_state_persists(self, wiki_root: Path) -> None:
        run = CurrentRun(status=RunStatus.IDLE)
        current_run.write(wiki_root, run)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.IDLE
        assert loaded.ssh_target is None

    def test_full_state_roundtrip(self, wiki_root: Path) -> None:
        target = SSHTarget(host="staging.example.com", user="deploy", port=2222)
        cmd = Command(natural_language="run the full test suite")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=9876)
        run = run.with_running("pytest -v --tb=short", remote_pid=5432)
        progress = Progress(
            percent=33.3,
            tests_passed=10,
            tests_failed=1,
            tests_skipped=2,
            tests_total=30,
            last_output_line="PASSED test_login_flow",
        )
        run = run.with_progress(progress)

        current_run.write(wiki_root, run)
        loaded = current_run.read(wiki_root)

        assert loaded is not None
        assert loaded.status == RunStatus.RUNNING
        assert loaded.run_id == run.run_id
        assert loaded.ssh_target is not None
        assert loaded.ssh_target.host == "staging.example.com"
        assert loaded.ssh_target.port == 2222
        assert loaded.command is not None
        assert loaded.command.natural_language == "run the full test suite"
        assert loaded.command.resolved_shell == "pytest -v --tb=short"
        assert loaded.command.approved is True
        assert loaded.pids.daemon == 9876
        assert loaded.pids.remote == 5432
        assert loaded.progress.percent == 33.3
        assert loaded.progress.tests_passed == 10
        assert loaded.progress.last_output_line == "PASSED test_login_flow"

    def test_file_is_valid_wiki_format(self, wiki_root: Path) -> None:
        run = CurrentRun()
        path = current_run.write(wiki_root, run)

        raw = path.read_text(encoding="utf-8")
        doc = parse(raw)
        assert "tags" in doc.frontmatter
        assert "daemon" in doc.frontmatter["tags"]
        assert doc.frontmatter["type"] == "daemon-state"
        assert "# Current Run" in doc.body

    def test_overwrites_existing(self, wiki_root: Path) -> None:
        run1 = CurrentRun()
        current_run.write(wiki_root, run1)

        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run tests")
        run2 = CurrentRun().with_pending_approval(target, cmd, daemon_pid=100)
        current_run.write(wiki_root, run2)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.PENDING_APPROVAL
        assert loaded.ssh_target is not None
        assert loaded.ssh_target.host == "prod.example.com"


# -- read --


class TestRead:
    def test_returns_none_when_no_file(self, wiki_root: Path) -> None:
        result = current_run.read(wiki_root)
        assert result is None

    def test_reads_back_written_state(self, wiki_root: Path) -> None:
        run = CurrentRun()
        current_run.write(wiki_root, run)
        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.run_id == run.run_id


# -- update --


class TestUpdate:
    def test_raises_when_no_file(self, wiki_root: Path) -> None:
        run = CurrentRun()
        with pytest.raises(FileNotFoundError, match="No current-run record"):
            current_run.update(wiki_root, run)

    def test_updates_existing_record(self, wiki_root: Path) -> None:
        # Write initial state
        run = CurrentRun()
        current_run.write(wiki_root, run)

        # Update to pending
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        pending = run.with_pending_approval(target, cmd, daemon_pid=42)
        current_run.update(wiki_root, pending)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.PENDING_APPROVAL

    def test_full_lifecycle(self, wiki_root: Path) -> None:
        """Test a complete state machine lifecycle through the wiki."""
        # IDLE
        run = CurrentRun()
        current_run.write(wiki_root, run)

        # -> PENDING_APPROVAL
        target = SSHTarget(host="staging.example.com", user="ci")
        cmd = Command(natural_language="run the smoke tests")
        pending = run.with_pending_approval(target, cmd, daemon_pid=1000)
        current_run.update(wiki_root, pending)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.PENDING_APPROVAL

        # -> RUNNING
        running = pending.with_running("pytest -m smoke", remote_pid=2000)
        current_run.update(wiki_root, running)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.RUNNING
        assert loaded.started_at is not None

        # -> Progress update
        progress = Progress(
            percent=60.0,
            tests_passed=6,
            tests_failed=0,
            tests_total=10,
            last_output_line="PASSED test_api_health",
        )
        updated = running.with_progress(progress)
        current_run.update(wiki_root, updated)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.progress.percent == 60.0

        # -> COMPLETED
        final = Progress(
            percent=100.0,
            tests_passed=10,
            tests_failed=0,
            tests_total=10,
        )
        completed = updated.with_completed(final)
        current_run.update(wiki_root, completed)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.COMPLETED
        assert loaded.completed_at is not None
        assert loaded.progress.tests_passed == 10


# -- clear --


class TestClear:
    def test_clear_resets_to_idle(self, wiki_root: Path) -> None:
        # Write an active run
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=42)
        current_run.write(wiki_root, run)

        # Clear it
        current_run.clear(wiki_root)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.IDLE
        assert loaded.ssh_target is None
        assert loaded.command is None

    def test_clear_creates_file_if_missing(self, wiki_root: Path) -> None:
        current_run.clear(wiki_root)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.IDLE

    def test_file_persists_after_clear(self, wiki_root: Path) -> None:
        current_run.clear(wiki_root)
        assert current_run.exists(wiki_root)


# -- exists --


class TestExists:
    def test_false_when_no_file(self, wiki_root: Path) -> None:
        assert current_run.exists(wiki_root) is False

    def test_true_after_write(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun())
        assert current_run.exists(wiki_root) is True


# -- file_path --


class TestFilePath:
    def test_returns_expected_path(self, wiki_root: Path) -> None:
        path = current_run.file_path(wiki_root)
        assert path == wiki_root / "pages" / "daemon" / "current-run.md"


# -- Crash recovery scenario --


class TestCrashRecovery:
    """Verify that state can be recovered from the wiki after a simulated crash."""

    def test_recovery_from_running_state(self, wiki_root: Path) -> None:
        """Daemon crashes while running; new daemon reads wiki to recover."""
        # Simulate: daemon writes running state
        target = SSHTarget(host="prod.example.com", user="ci", port=22)
        cmd = Command(natural_language="run full regression")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        run = run.with_running("pytest --regression", remote_pid=5678)
        progress = Progress(
            percent=75.0,
            tests_passed=30,
            tests_failed=2,
            tests_total=40,
            last_output_line="FAILED test_payment_flow",
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)

        # Simulate: daemon crashes, new daemon starts, reads state
        recovered = current_run.read(wiki_root)

        assert recovered is not None
        assert recovered.status == RunStatus.RUNNING
        assert recovered.run_id == run.run_id
        assert recovered.ssh_target is not None
        assert recovered.ssh_target.host == "prod.example.com"
        assert recovered.command is not None
        assert recovered.command.resolved_shell == "pytest --regression"
        assert recovered.pids.daemon == 1234
        assert recovered.pids.remote == 5678
        assert recovered.progress.percent == 75.0
        assert recovered.progress.tests_failed == 2

    def test_recovery_preserves_timestamps(self, wiki_root: Path) -> None:
        run = CurrentRun()
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="test")
        running = run.with_pending_approval(target, cmd, daemon_pid=1)
        running = running.with_running("echo test")
        current_run.write(wiki_root, running)

        recovered = current_run.read(wiki_root)
        assert recovered is not None
        assert recovered.started_at is not None
        assert recovered.created_at is not None


# -- Error state roundtrip --


class TestErrorStateRoundtrip:
    def test_error_message_preserved(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        error_progress = Progress(tests_passed=0, tests_failed=1, tests_total=1)
        failed = run.with_failed(
            "SSH connection timeout after 30s", error_progress
        )
        current_run.write(wiki_root, failed)

        loaded = current_run.read(wiki_root)
        assert loaded is not None
        assert loaded.status == RunStatus.FAILED
        assert loaded.error == "SSH connection timeout after 30s"
        assert loaded.progress.tests_failed == 1
