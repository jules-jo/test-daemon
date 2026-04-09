"""Tests for session state persistence to wiki on disconnect.

Verifies that:
- SessionSnapshot captures full session context as a frozen dataclass
- save_session_state writes a valid wiki file with YAML frontmatter
- save_session_state uses atomic writes (temp file + rename)
- load_session_state reads and deserializes the snapshot correctly
- load_session_state handles missing files gracefully
- load_session_state handles corrupted files gracefully
- discard_session_state clears the session file to an idle marker
- SessionSnapshot.from_current_run builds snapshots from run state
- Round-trip serialization preserves all fields
- Session file lives at pages/daemon/session-state.md
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    ProcessIDs,
    Progress,
    RunStatus,
    SSHTarget,
)
from jules_daemon.wiki.session_persistence import (
    LoadSessionOutcome,
    SessionLoadResult,
    SessionSnapshot,
    SessionWriteResult,
    discard_session_state,
    load_session_state,
    save_session_state,
    session_file_path,
)


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "pages" / "daemon").mkdir(parents=True)
    return wiki


def _make_ssh_target() -> SSHTarget:
    return SSHTarget(host="10.0.0.1", user="deploy", port=22, key_path="/home/deploy/.ssh/id_rsa")


def _make_command(approved: bool = True) -> Command:
    cmd = Command(natural_language="run the unit tests")
    if approved:
        return cmd.with_approval("pytest -v tests/")
    return cmd


def _make_progress() -> Progress:
    return Progress(
        percent=45.0,
        tests_passed=22,
        tests_failed=1,
        tests_skipped=2,
        tests_total=50,
        last_output_line="PASSED tests/test_foo.py::test_bar",
        checkpoint_at=datetime(2026, 4, 9, 10, 30, 0, tzinfo=timezone.utc),
    )


def _make_running_run() -> CurrentRun:
    target = _make_ssh_target()
    cmd = _make_command(approved=False)
    run = CurrentRun()
    run = run.with_pending_approval(target, cmd, daemon_pid=12345)
    run = run.with_running("pytest -v tests/", remote_pid=67890)
    return run.with_progress(_make_progress())


def _make_pending_run() -> CurrentRun:
    target = _make_ssh_target()
    cmd = _make_command(approved=False)
    run = CurrentRun()
    return run.with_pending_approval(target, cmd, daemon_pid=12345)


# -- SessionSnapshot dataclass --


class TestSessionSnapshot:
    def test_frozen(self) -> None:
        snap = SessionSnapshot(
            run_id="test-123",
            status=RunStatus.RUNNING,
            ssh_target=_make_ssh_target(),
            command=_make_command(),
            pids=ProcessIDs(daemon=123, remote=456),
            progress=_make_progress(),
            started_at=datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc),
            error=None,
            disconnect_reason="eof",
            client_name="jules-cli",
            client_pid=789,
        )
        with pytest.raises(AttributeError):
            snap.run_id = "changed"  # type: ignore[misc]

    def test_is_resumable_running(self) -> None:
        snap = SessionSnapshot(
            run_id="test-123",
            status=RunStatus.RUNNING,
            ssh_target=_make_ssh_target(),
            command=_make_command(),
            pids=ProcessIDs(daemon=123, remote=456),
            progress=_make_progress(),
            started_at=datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc),
            error=None,
            disconnect_reason="eof",
            client_name="jules-cli",
            client_pid=789,
        )
        assert snap.is_resumable is True

    def test_is_resumable_pending(self) -> None:
        snap = SessionSnapshot(
            run_id="test-123",
            status=RunStatus.PENDING_APPROVAL,
            ssh_target=_make_ssh_target(),
            command=_make_command(approved=False),
            pids=ProcessIDs(daemon=123),
            progress=Progress(),
            started_at=None,
            error=None,
            disconnect_reason="broken_pipe",
            client_name="jules-cli",
            client_pid=789,
        )
        assert snap.is_resumable is True

    def test_is_not_resumable_completed(self) -> None:
        snap = SessionSnapshot(
            run_id="test-123",
            status=RunStatus.COMPLETED,
            ssh_target=_make_ssh_target(),
            command=_make_command(),
            pids=ProcessIDs(daemon=123, remote=456),
            progress=_make_progress(),
            started_at=datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc),
            error=None,
            disconnect_reason="eof",
            client_name="jules-cli",
            client_pid=789,
        )
        assert snap.is_resumable is False

    def test_is_not_resumable_idle(self) -> None:
        snap = SessionSnapshot(
            run_id="test-123",
            status=RunStatus.IDLE,
            ssh_target=None,
            command=None,
            pids=ProcessIDs(),
            progress=Progress(),
            started_at=None,
            error=None,
            disconnect_reason="eof",
            client_name="jules-cli",
            client_pid=None,
        )
        assert snap.is_resumable is False

    def test_saved_at_default_factory_produces_unique_timestamps(self) -> None:
        """Verify that saved_at uses default_factory, not a class-level default."""
        import time

        snap1 = SessionSnapshot(
            run_id="a",
            status=RunStatus.IDLE,
            ssh_target=None,
            command=None,
            pids=ProcessIDs(),
            progress=Progress(),
            started_at=None,
            error=None,
            disconnect_reason="eof",
            client_name="cli",
            client_pid=None,
        )
        time.sleep(0.01)
        snap2 = SessionSnapshot(
            run_id="b",
            status=RunStatus.IDLE,
            ssh_target=None,
            command=None,
            pids=ProcessIDs(),
            progress=Progress(),
            started_at=None,
            error=None,
            disconnect_reason="eof",
            client_name="cli",
            client_pid=None,
        )
        # Each snapshot must get a fresh timestamp
        assert snap1.saved_at != snap2.saved_at
        assert snap2.saved_at > snap1.saved_at

    def test_from_current_run_running(self) -> None:
        run = _make_running_run()
        snap = SessionSnapshot.from_current_run(
            run=run,
            disconnect_reason="connection_reset",
            client_name="jules-cli",
            client_pid=999,
        )
        assert snap.run_id == run.run_id
        assert snap.status == RunStatus.RUNNING
        assert snap.ssh_target == run.ssh_target
        assert snap.command == run.command
        assert snap.pids == run.pids
        assert snap.progress == run.progress
        assert snap.started_at == run.started_at
        assert snap.disconnect_reason == "connection_reset"
        assert snap.client_name == "jules-cli"
        assert snap.client_pid == 999

    def test_from_current_run_pending(self) -> None:
        run = _make_pending_run()
        snap = SessionSnapshot.from_current_run(
            run=run,
            disconnect_reason="eof",
            client_name="test-client",
            client_pid=None,
        )
        assert snap.status == RunStatus.PENDING_APPROVAL
        assert snap.is_resumable is True
        assert snap.client_pid is None

    def test_from_current_run_idle(self) -> None:
        run = CurrentRun()
        snap = SessionSnapshot.from_current_run(
            run=run,
            disconnect_reason="eof",
            client_name="test-client",
            client_pid=None,
        )
        assert snap.status == RunStatus.IDLE
        assert snap.is_resumable is False


# -- save_session_state --


class TestSaveSessionState:
    def test_writes_file(self, wiki_root: Path) -> None:
        run = _make_running_run()
        snap = SessionSnapshot.from_current_run(
            run=run,
            disconnect_reason="eof",
            client_name="jules-cli",
            client_pid=999,
        )
        result = save_session_state(wiki_root, snap)
        assert result.success is True
        assert result.file_path is not None
        assert result.file_path.exists()
        assert result.file_path.name == "session-state.md"

    def test_file_contains_frontmatter(self, wiki_root: Path) -> None:
        run = _make_running_run()
        snap = SessionSnapshot.from_current_run(
            run=run,
            disconnect_reason="eof",
            client_name="jules-cli",
            client_pid=999,
        )
        save_session_state(wiki_root, snap)
        content = session_file_path(wiki_root).read_text(encoding="utf-8")
        assert content.startswith("---")
        assert "type: daemon-session-state" in content
        assert "status: running" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        wiki = tmp_path / "new_wiki"
        snap = SessionSnapshot(
            run_id="test-123",
            status=RunStatus.RUNNING,
            ssh_target=_make_ssh_target(),
            command=_make_command(),
            pids=ProcessIDs(daemon=123, remote=456),
            progress=Progress(),
            started_at=datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc),
            error=None,
            disconnect_reason="eof",
            client_name="jules-cli",
            client_pid=789,
        )
        result = save_session_state(wiki, snap)
        assert result.success is True
        assert result.file_path is not None
        assert result.file_path.exists()

    def test_result_on_success(self, wiki_root: Path) -> None:
        snap = SessionSnapshot(
            run_id="test-123",
            status=RunStatus.RUNNING,
            ssh_target=None,
            command=None,
            pids=ProcessIDs(),
            progress=Progress(),
            started_at=None,
            error=None,
            disconnect_reason="eof",
            client_name="cli",
            client_pid=None,
        )
        result = save_session_state(wiki_root, snap)
        assert result.success is True
        assert result.error is None

    def test_atomic_write_no_leftover_tmp(self, wiki_root: Path) -> None:
        snap = SessionSnapshot(
            run_id="test-123",
            status=RunStatus.RUNNING,
            ssh_target=None,
            command=None,
            pids=ProcessIDs(),
            progress=Progress(),
            started_at=None,
            error=None,
            disconnect_reason="eof",
            client_name="cli",
            client_pid=None,
        )
        save_session_state(wiki_root, snap)
        daemon_dir = wiki_root / "pages" / "daemon"
        tmp_files = list(daemon_dir.glob("*.tmp"))
        assert len(tmp_files) == 0


# -- load_session_state --


class TestLoadSessionState:
    def test_round_trip_running(self, wiki_root: Path) -> None:
        run = _make_running_run()
        original = SessionSnapshot.from_current_run(
            run=run,
            disconnect_reason="connection_reset",
            client_name="jules-cli",
            client_pid=999,
        )
        save_session_state(wiki_root, original)
        loaded = load_session_state(wiki_root)
        assert loaded.outcome == LoadSessionOutcome.LOADED
        assert loaded.snapshot is not None
        assert loaded.snapshot.run_id == original.run_id
        assert loaded.snapshot.status == RunStatus.RUNNING
        assert loaded.snapshot.disconnect_reason == "connection_reset"
        assert loaded.snapshot.client_name == "jules-cli"
        assert loaded.snapshot.client_pid == 999

    def test_round_trip_pending(self, wiki_root: Path) -> None:
        run = _make_pending_run()
        original = SessionSnapshot.from_current_run(
            run=run,
            disconnect_reason="eof",
            client_name="test-client",
            client_pid=None,
        )
        save_session_state(wiki_root, original)
        loaded = load_session_state(wiki_root)
        assert loaded.outcome == LoadSessionOutcome.LOADED
        assert loaded.snapshot is not None
        assert loaded.snapshot.status == RunStatus.PENDING_APPROVAL
        assert loaded.snapshot.client_pid is None

    def test_round_trip_preserves_ssh_target(self, wiki_root: Path) -> None:
        target = _make_ssh_target()
        snap = SessionSnapshot(
            run_id="test-ssh",
            status=RunStatus.RUNNING,
            ssh_target=target,
            command=_make_command(),
            pids=ProcessIDs(daemon=100, remote=200),
            progress=_make_progress(),
            started_at=datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc),
            error=None,
            disconnect_reason="eof",
            client_name="cli",
            client_pid=42,
        )
        save_session_state(wiki_root, snap)
        loaded = load_session_state(wiki_root)
        assert loaded.snapshot is not None
        assert loaded.snapshot.ssh_target is not None
        assert loaded.snapshot.ssh_target.host == "10.0.0.1"
        assert loaded.snapshot.ssh_target.user == "deploy"
        assert loaded.snapshot.ssh_target.port == 22
        assert loaded.snapshot.ssh_target.key_path == "/home/deploy/.ssh/id_rsa"

    def test_round_trip_preserves_progress(self, wiki_root: Path) -> None:
        snap = SessionSnapshot(
            run_id="test-prog",
            status=RunStatus.RUNNING,
            ssh_target=None,
            command=None,
            pids=ProcessIDs(),
            progress=_make_progress(),
            started_at=None,
            error=None,
            disconnect_reason="eof",
            client_name="cli",
            client_pid=None,
        )
        save_session_state(wiki_root, snap)
        loaded = load_session_state(wiki_root)
        assert loaded.snapshot is not None
        assert loaded.snapshot.progress.percent == 45.0
        assert loaded.snapshot.progress.tests_passed == 22
        assert loaded.snapshot.progress.tests_failed == 1
        assert loaded.snapshot.progress.tests_skipped == 2
        assert loaded.snapshot.progress.tests_total == 50

    def test_no_file_returns_no_file(self, wiki_root: Path) -> None:
        loaded = load_session_state(wiki_root)
        assert loaded.outcome == LoadSessionOutcome.NO_FILE
        assert loaded.snapshot is None
        assert loaded.error is None

    def test_corrupted_file_returns_corrupted(self, wiki_root: Path) -> None:
        fpath = session_file_path(wiki_root)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text("not valid yaml frontmatter", encoding="utf-8")
        loaded = load_session_state(wiki_root)
        assert loaded.outcome == LoadSessionOutcome.CORRUPTED
        assert loaded.snapshot is None
        assert loaded.error is not None

    def test_empty_file_returns_corrupted(self, wiki_root: Path) -> None:
        fpath = session_file_path(wiki_root)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text("", encoding="utf-8")
        loaded = load_session_state(wiki_root)
        assert loaded.outcome == LoadSessionOutcome.CORRUPTED
        assert loaded.snapshot is None


# -- discard_session_state --


class TestDiscardSessionState:
    def test_discard_removes_session(self, wiki_root: Path) -> None:
        run = _make_running_run()
        snap = SessionSnapshot.from_current_run(
            run=run,
            disconnect_reason="eof",
            client_name="cli",
            client_pid=None,
        )
        save_session_state(wiki_root, snap)
        assert session_file_path(wiki_root).exists()
        result = discard_session_state(wiki_root)
        assert result is True
        # After discard, loading should return idle/no-file or non-resumable
        loaded = load_session_state(wiki_root)
        if loaded.outcome == LoadSessionOutcome.LOADED:
            assert loaded.snapshot is not None
            assert loaded.snapshot.is_resumable is False
        else:
            assert loaded.outcome == LoadSessionOutcome.NO_FILE

    def test_discard_when_no_file(self, wiki_root: Path) -> None:
        result = discard_session_state(wiki_root)
        assert result is True  # No-op is still success


# -- session_file_path --


class TestSessionFilePath:
    def test_returns_expected_path(self, wiki_root: Path) -> None:
        fpath = session_file_path(wiki_root)
        assert fpath == wiki_root / "pages" / "daemon" / "session-state.md"
