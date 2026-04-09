"""Tests for wiki state reader that extracts connection parameters and run metadata.

Verifies that the state reader:
- Loads the current-run wiki file and extracts SSH connection parameters
- Extracts key_path for SSH key-based authentication
- Extracts run metadata needed for reconnection (run_id, status, command, PIDs)
- Handles missing files (returns empty ReconnectionState)
- Handles corrupted files (returns error ReconnectionState)
- Provides all fields needed for SSH reconnection after daemon restart
- Preserves all connection parameters through write/read roundtrip
"""

import time
from pathlib import Path

import pytest

from jules_daemon.wiki.state_reader import (
    ConnectionParams,
    LoadResult,
    ReconnectionState,
    load_reconnection_state,
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


# -- ConnectionParams --


class TestConnectionParams:
    def test_create_with_all_fields(self) -> None:
        params = ConnectionParams(
            host="prod.example.com",
            port=2222,
            username="ci",
            key_path="/home/ci/.ssh/id_ed25519",
        )
        assert params.host == "prod.example.com"
        assert params.port == 2222
        assert params.username == "ci"
        assert params.key_path == "/home/ci/.ssh/id_ed25519"

    def test_create_without_key_path(self) -> None:
        params = ConnectionParams(
            host="staging.example.com",
            port=22,
            username="deploy",
            key_path=None,
        )
        assert params.key_path is None

    def test_frozen(self) -> None:
        params = ConnectionParams(
            host="host",
            port=22,
            username="user",
            key_path=None,
        )
        with pytest.raises(AttributeError):
            params.host = "other"  # type: ignore[misc]


# -- ReconnectionState --


class TestReconnectionState:
    def test_create_loaded_state(self) -> None:
        conn = ConnectionParams(
            host="prod.example.com",
            port=22,
            username="ci",
            key_path="/path/to/key",
        )
        state = ReconnectionState(
            result=LoadResult.LOADED,
            connection=conn,
            run_id="abc-123",
            status=RunStatus.RUNNING,
            resolved_shell="pytest -v",
            daemon_pid=1234,
            remote_pid=5678,
            natural_language_command="run all tests",
            progress_percent=55.0,
            error=None,
            source_path=Path("/tmp/wiki/pages/daemon/current-run.md"),
        )
        assert state.result == LoadResult.LOADED
        assert state.connection is not None
        assert state.connection.host == "prod.example.com"
        assert state.run_id == "abc-123"
        assert state.status == RunStatus.RUNNING
        assert state.resolved_shell == "pytest -v"
        assert state.daemon_pid == 1234
        assert state.remote_pid == 5678

    def test_create_empty_state_no_file(self) -> None:
        state = ReconnectionState(
            result=LoadResult.NO_FILE,
            connection=None,
            run_id="",
            status=RunStatus.IDLE,
            resolved_shell="",
            daemon_pid=None,
            remote_pid=None,
            natural_language_command="",
            progress_percent=0.0,
            error=None,
            source_path=None,
        )
        assert state.result == LoadResult.NO_FILE
        assert state.connection is None
        assert state.run_id == ""

    def test_frozen(self) -> None:
        state = ReconnectionState(
            result=LoadResult.NO_FILE,
            connection=None,
            run_id="",
            status=RunStatus.IDLE,
            resolved_shell="",
            daemon_pid=None,
            remote_pid=None,
            natural_language_command="",
            progress_percent=0.0,
            error=None,
            source_path=None,
        )
        with pytest.raises(AttributeError):
            state.run_id = "changed"  # type: ignore[misc]

    def test_has_connection_property(self) -> None:
        conn = ConnectionParams(
            host="host", port=22, username="user", key_path=None,
        )
        state = ReconnectionState(
            result=LoadResult.LOADED,
            connection=conn,
            run_id="abc",
            status=RunStatus.RUNNING,
            resolved_shell="pytest",
            daemon_pid=1,
            remote_pid=2,
            natural_language_command="run tests",
            progress_percent=0.0,
            error=None,
            source_path=None,
        )
        assert state.has_connection is True

    def test_has_connection_false_when_none(self) -> None:
        state = ReconnectionState(
            result=LoadResult.NO_FILE,
            connection=None,
            run_id="",
            status=RunStatus.IDLE,
            resolved_shell="",
            daemon_pid=None,
            remote_pid=None,
            natural_language_command="",
            progress_percent=0.0,
            error=None,
            source_path=None,
        )
        assert state.has_connection is False

    def test_can_reconnect_when_running(self) -> None:
        conn = ConnectionParams(
            host="host", port=22, username="user", key_path=None,
        )
        state = ReconnectionState(
            result=LoadResult.LOADED,
            connection=conn,
            run_id="abc",
            status=RunStatus.RUNNING,
            resolved_shell="pytest",
            daemon_pid=1,
            remote_pid=2,
            natural_language_command="run tests",
            progress_percent=50.0,
            error=None,
            source_path=None,
        )
        assert state.can_reconnect is True

    def test_cannot_reconnect_when_idle(self) -> None:
        state = ReconnectionState(
            result=LoadResult.LOADED,
            connection=None,
            run_id="abc",
            status=RunStatus.IDLE,
            resolved_shell="",
            daemon_pid=None,
            remote_pid=None,
            natural_language_command="",
            progress_percent=0.0,
            error=None,
            source_path=None,
        )
        assert state.can_reconnect is False

    def test_cannot_reconnect_when_no_file(self) -> None:
        state = ReconnectionState(
            result=LoadResult.NO_FILE,
            connection=None,
            run_id="",
            status=RunStatus.IDLE,
            resolved_shell="",
            daemon_pid=None,
            remote_pid=None,
            natural_language_command="",
            progress_percent=0.0,
            error=None,
            source_path=None,
        )
        assert state.can_reconnect is False

    def test_cannot_reconnect_when_completed(self) -> None:
        conn = ConnectionParams(
            host="host", port=22, username="user", key_path=None,
        )
        state = ReconnectionState(
            result=LoadResult.LOADED,
            connection=conn,
            run_id="abc",
            status=RunStatus.COMPLETED,
            resolved_shell="pytest",
            daemon_pid=1,
            remote_pid=2,
            natural_language_command="run tests",
            progress_percent=100.0,
            error=None,
            source_path=None,
        )
        assert state.can_reconnect is False

    def test_cannot_reconnect_when_running_but_no_connection(self) -> None:
        """Edge case: active status but missing connection params."""
        state = ReconnectionState(
            result=LoadResult.LOADED,
            connection=None,
            run_id="abc",
            status=RunStatus.RUNNING,
            resolved_shell="pytest",
            daemon_pid=1,
            remote_pid=2,
            natural_language_command="run tests",
            progress_percent=50.0,
            error=None,
            source_path=None,
        )
        assert state.can_reconnect is False


# -- LoadResult enum --


class TestLoadResult:
    def test_all_values(self) -> None:
        assert LoadResult.LOADED.value == "loaded"
        assert LoadResult.NO_FILE.value == "no_file"
        assert LoadResult.CORRUPTED.value == "corrupted"


# -- load_reconnection_state: No file --


class TestLoadNoFile:
    """When no wiki file exists, return an empty ReconnectionState."""

    def test_returns_no_file_result(self, wiki_root: Path) -> None:
        state = load_reconnection_state(wiki_root)
        assert state.result == LoadResult.NO_FILE

    def test_connection_is_none(self, wiki_root: Path) -> None:
        state = load_reconnection_state(wiki_root)
        assert state.connection is None

    def test_status_is_idle(self, wiki_root: Path) -> None:
        state = load_reconnection_state(wiki_root)
        assert state.status == RunStatus.IDLE

    def test_run_id_is_empty(self, wiki_root: Path) -> None:
        state = load_reconnection_state(wiki_root)
        assert state.run_id == ""

    def test_source_path_is_none(self, wiki_root: Path) -> None:
        state = load_reconnection_state(wiki_root)
        assert state.source_path is None

    def test_cannot_reconnect(self, wiki_root: Path) -> None:
        state = load_reconnection_state(wiki_root)
        assert state.can_reconnect is False


# -- load_reconnection_state: Idle file --


class TestLoadIdleFile:
    """When wiki file exists with idle state, no connection params available."""

    def test_returns_loaded(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))
        state = load_reconnection_state(wiki_root)
        assert state.result == LoadResult.LOADED

    def test_connection_is_none(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))
        state = load_reconnection_state(wiki_root)
        assert state.connection is None

    def test_status_is_idle(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))
        state = load_reconnection_state(wiki_root)
        assert state.status == RunStatus.IDLE

    def test_cannot_reconnect(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))
        state = load_reconnection_state(wiki_root)
        assert state.can_reconnect is False


# -- load_reconnection_state: Running state with full connection params --


class TestLoadRunningState:
    """When wiki file has a running test, extract all reconnection fields."""

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
            percent=50.0,
            tests_passed=25,
            tests_failed=1,
            tests_total=50,
            last_output_line="FAILED test_checkout",
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)
        return run

    def test_extracts_host(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, host="prod.example.com")
        state = load_reconnection_state(wiki_root)
        assert state.connection is not None
        assert state.connection.host == "prod.example.com"

    def test_extracts_port(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, port=2222)
        state = load_reconnection_state(wiki_root)
        assert state.connection is not None
        assert state.connection.port == 2222

    def test_extracts_username(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, user="deploy")
        state = load_reconnection_state(wiki_root)
        assert state.connection is not None
        assert state.connection.username == "deploy"

    def test_extracts_key_path(self, wiki_root: Path) -> None:
        self._write_running_state(
            wiki_root, key_path="/home/ci/.ssh/id_ed25519"
        )
        state = load_reconnection_state(wiki_root)
        assert state.connection is not None
        assert state.connection.key_path == "/home/ci/.ssh/id_ed25519"

    def test_handles_no_key_path(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, key_path=None)
        state = load_reconnection_state(wiki_root)
        assert state.connection is not None
        assert state.connection.key_path is None

    def test_extracts_run_id(self, wiki_root: Path) -> None:
        run = self._write_running_state(wiki_root)
        state = load_reconnection_state(wiki_root)
        assert state.run_id == run.run_id

    def test_extracts_status(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        state = load_reconnection_state(wiki_root)
        assert state.status == RunStatus.RUNNING

    def test_extracts_resolved_shell(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, resolved_shell="pytest -v --regression")
        state = load_reconnection_state(wiki_root)
        assert state.resolved_shell == "pytest -v --regression"

    def test_extracts_natural_language_command(self, wiki_root: Path) -> None:
        self._write_running_state(
            wiki_root, natural_language="run the full regression suite"
        )
        state = load_reconnection_state(wiki_root)
        assert state.natural_language_command == "run the full regression suite"

    def test_extracts_daemon_pid(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, daemon_pid=9876)
        state = load_reconnection_state(wiki_root)
        assert state.daemon_pid == 9876

    def test_extracts_remote_pid(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root, remote_pid=5432)
        state = load_reconnection_state(wiki_root)
        assert state.remote_pid == 5432

    def test_extracts_progress_percent(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        state = load_reconnection_state(wiki_root)
        assert state.progress_percent == 50.0

    def test_can_reconnect(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        state = load_reconnection_state(wiki_root)
        assert state.can_reconnect is True

    def test_has_connection(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        state = load_reconnection_state(wiki_root)
        assert state.has_connection is True

    def test_source_path_set(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        state = load_reconnection_state(wiki_root)
        assert state.source_path is not None
        assert state.source_path.name == "current-run.md"

    def test_no_error(self, wiki_root: Path) -> None:
        self._write_running_state(wiki_root)
        state = load_reconnection_state(wiki_root)
        assert state.error is None


# -- load_reconnection_state: Pending approval state --


class TestLoadPendingApprovalState:
    """Pending approval state has connection but no resolved shell yet."""

    def test_extracts_connection(self, wiki_root: Path) -> None:
        target = SSHTarget(
            host="staging.example.com",
            user="deploy",
            port=22,
            key_path="/home/deploy/.ssh/id_rsa",
        )
        cmd = Command(natural_language="run smoke tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1000)
        current_run.write(wiki_root, run)

        state = load_reconnection_state(wiki_root)
        assert state.connection is not None
        assert state.connection.host == "staging.example.com"
        assert state.connection.username == "deploy"
        assert state.connection.key_path == "/home/deploy/.ssh/id_rsa"

    def test_status_is_pending(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        current_run.write(wiki_root, run)

        state = load_reconnection_state(wiki_root)
        assert state.status == RunStatus.PENDING_APPROVAL

    def test_can_reconnect_when_pending(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        current_run.write(wiki_root, run)

        state = load_reconnection_state(wiki_root)
        assert state.can_reconnect is True


# -- load_reconnection_state: Terminal states --


class TestLoadTerminalStates:
    """Terminal states still extract connection params but cannot reconnect."""

    def test_completed_extracts_connection(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user", key_path="/path/to/key")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        final = Progress(percent=100.0, tests_passed=10, tests_total=10)
        run = run.with_completed(final)
        current_run.write(wiki_root, run)

        state = load_reconnection_state(wiki_root)
        assert state.connection is not None
        assert state.connection.host == "host"
        assert state.connection.key_path == "/path/to/key"
        assert state.can_reconnect is False

    def test_failed_extracts_error(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        error_progress = Progress(tests_failed=1, tests_total=1)
        run = run.with_failed("SSH timeout", error_progress)
        current_run.write(wiki_root, run)

        state = load_reconnection_state(wiki_root)
        assert state.error == "SSH timeout"
        assert state.can_reconnect is False


# -- load_reconnection_state: Corrupted file --


class TestLoadCorruptedFile:
    """Corrupted files return a safe error state."""

    def test_invalid_content_returns_corrupted(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("not valid yaml at all", encoding="utf-8")

        state = load_reconnection_state(wiki_root)
        assert state.result == LoadResult.CORRUPTED

    def test_corrupted_has_error_detail(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("garbage", encoding="utf-8")

        state = load_reconnection_state(wiki_root)
        assert state.error is not None
        assert len(state.error) > 0

    def test_corrupted_connection_is_none(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("bad content", encoding="utf-8")

        state = load_reconnection_state(wiki_root)
        assert state.connection is None

    def test_corrupted_cannot_reconnect(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("bad content", encoding="utf-8")

        state = load_reconnection_state(wiki_root)
        assert state.can_reconnect is False

    def test_corrupted_source_path_set(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("bad content", encoding="utf-8")

        state = load_reconnection_state(wiki_root)
        assert state.source_path is not None

    def test_empty_file_returns_corrupted(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("", encoding="utf-8")

        state = load_reconnection_state(wiki_root)
        assert state.result == LoadResult.CORRUPTED


# -- key_path roundtrip through wiki persistence --


class TestKeyPathRoundtrip:
    """Verify key_path survives write -> read -> load cycle."""

    def test_key_path_persisted_and_recovered(self, wiki_root: Path) -> None:
        target = SSHTarget(
            host="prod.example.com",
            user="ci",
            port=2222,
            key_path="/home/ci/.ssh/id_ed25519",
        )
        cmd = Command(natural_language="run regression")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=999)
        run = run.with_running("pytest", remote_pid=888)
        current_run.write(wiki_root, run)

        state = load_reconnection_state(wiki_root)
        assert state.connection is not None
        assert state.connection.key_path == "/home/ci/.ssh/id_ed25519"

    def test_none_key_path_persisted(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user", key_path=None)
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        current_run.write(wiki_root, run)

        state = load_reconnection_state(wiki_root)
        assert state.connection is not None
        assert state.connection.key_path is None


# -- Performance --


class TestPerformance:
    """State reader must be fast enough for 30s crash recovery SLA."""

    def test_load_under_100ms(self, wiki_root: Path) -> None:
        target = SSHTarget(
            host="prod.example.com", user="ci", port=2222,
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
        state = load_reconnection_state(wiki_root)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 100.0, f"State reader took {elapsed_ms:.1f}ms (>100ms)"
        assert state.result == LoadResult.LOADED
        assert state.can_reconnect is True
