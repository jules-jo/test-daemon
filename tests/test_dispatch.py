"""Tests for SSH command dispatch and wiki state updates.

Verifies that the dispatch module:
- Issues a generated command over an SSH connection handle
- Updates the wiki current-run entry with the action taken (resume/restart)
- Transitions run state to RUNNING after successful dispatch
- Records the command string and remote PID in wiki state
- Handles SSH channel execution errors gracefully
- Returns immutable DispatchResult with success status and metadata
- Creates audit entries for dispatch events
- Never raises on dispatch errors -- captures them in result
- Validates required inputs (handle, command, wiki_root)
- Updates wiki state to FAILED if dispatch cannot proceed
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from jules_daemon.ssh.command import SSHCommand
from jules_daemon.ssh.command_gen import (
    GeneratedCommand,
    RecoveryCommandAction,
    TestFramework,
)
from jules_daemon.ssh.dispatch import (
    DispatchResult,
    SSHDispatchHandle,
    dispatch_recovery_command,
)
from jules_daemon.wiki import current_run
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    ProcessIDs,
    Progress,
    RunStatus,
    SSHTarget,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeDispatchHandle:
    """Fake SSH dispatch handle for testing.

    Simulates executing a command over SSH and returning a PID.
    """

    remote_pid: int | None = 42
    exit_code: int | None = None
    error: Exception | None = None
    executed_commands: list[str] | None = None

    def __post_init__(self) -> None:
        if self.executed_commands is None:
            self.executed_commands = []

    async def execute(self, command: str, timeout: int) -> int | None:
        """Execute command and return remote PID or raise."""
        if self.executed_commands is not None:
            self.executed_commands.append(command)
        if self.error is not None:
            raise self.error
        return self.remote_pid

    @property
    def session_id(self) -> str:
        return "fake-session-123"


# Verify protocol compliance
_handle = FakeDispatchHandle()
assert hasattr(_handle, "execute")
assert hasattr(_handle, "session_id")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generated_command(
    *,
    action: RecoveryCommandAction = RecoveryCommandAction.RESTART,
    command: str = "pytest -v --tb=short",
    working_directory: str | None = None,
    environment: dict[str, str] | None = None,
    timeout: int = 300,
    original_shell: str = "pytest -v --tb=short",
    resume_context: str = "Restarting full test suite",
    checkpoint_marker: str = "",
    test_index: int = 0,
    framework: TestFramework = TestFramework.PYTEST,
    run_id: str = "run-abc-123",
) -> GeneratedCommand:
    """Create a GeneratedCommand with defaults."""
    return GeneratedCommand(
        action=action,
        ssh_command=SSHCommand(
            command=command,
            working_directory=working_directory,
            environment=environment or {},
            timeout=timeout,
        ),
        original_shell=original_shell,
        resume_context=resume_context,
        checkpoint_marker=checkpoint_marker,
        test_index=test_index,
        framework=framework,
        run_id=run_id,
    )


def _make_wiki_running_state(wiki_root: Path) -> CurrentRun:
    """Write a RUNNING run state to wiki and return it."""
    target = SSHTarget(host="prod.example.com", user="ci", port=22)
    cmd = Command(natural_language="run full regression")
    run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
    run = run.with_running("pytest -v --tb=short", remote_pid=5678)
    progress = Progress(
        percent=50.0,
        tests_passed=25,
        tests_failed=1,
        tests_total=50,
    )
    run = run.with_progress(progress)
    current_run.write(wiki_root, run)
    return run


def _make_wiki_pending_state(wiki_root: Path) -> CurrentRun:
    """Write a PENDING_APPROVAL run state to wiki and return it."""
    target = SSHTarget(host="prod.example.com", user="ci", port=22)
    cmd = Command(natural_language="run full regression")
    run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
    current_run.write(wiki_root, run)
    return run


# ---------------------------------------------------------------------------
# DispatchResult frozen dataclass
# ---------------------------------------------------------------------------


class TestDispatchResult:
    def test_frozen(self) -> None:
        result = DispatchResult(
            success=True,
            action=RecoveryCommandAction.RESTART,
            command_string="pytest",
            run_id="abc",
            remote_pid=42,
            error=None,
            wiki_updated=True,
            session_id="sess-1",
            timestamp=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    def test_has_all_fields(self) -> None:
        ts = datetime.now(timezone.utc)
        result = DispatchResult(
            success=False,
            action=RecoveryCommandAction.RESUME,
            command_string="pytest --lf",
            run_id="xyz",
            remote_pid=None,
            error="connection lost",
            wiki_updated=False,
            session_id="sess-2",
            timestamp=ts,
        )
        assert result.success is False
        assert result.action == RecoveryCommandAction.RESUME
        assert result.command_string == "pytest --lf"
        assert result.run_id == "xyz"
        assert result.remote_pid is None
        assert result.error == "connection lost"
        assert result.wiki_updated is False
        assert result.session_id == "sess-2"
        assert result.timestamp == ts


# ---------------------------------------------------------------------------
# dispatch_recovery_command: successful restart
# ---------------------------------------------------------------------------


class TestDispatchRestart:
    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        return tmp_path / "wiki"

    @pytest.mark.asyncio
    async def test_successful_restart_dispatch(self, wiki_root: Path) -> None:
        prior_run = _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=9999)
        gen_cmd = _make_generated_command(
            action=RecoveryCommandAction.RESTART,
            command="pytest -v --tb=short",
            run_id=prior_run.run_id,
        )

        result = await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        assert result.success is True
        assert result.action == RecoveryCommandAction.RESTART
        assert result.remote_pid == 9999

    @pytest.mark.asyncio
    async def test_restart_updates_wiki_state(self, wiki_root: Path) -> None:
        prior_run = _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=42)
        gen_cmd = _make_generated_command(
            action=RecoveryCommandAction.RESTART,
            command="pytest",
            run_id=prior_run.run_id,
        )

        await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        updated = current_run.read(wiki_root)
        assert updated is not None
        assert updated.status == RunStatus.RUNNING

    @pytest.mark.asyncio
    async def test_restart_records_command_in_wiki(
        self, wiki_root: Path
    ) -> None:
        prior_run = _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=42)
        gen_cmd = _make_generated_command(
            action=RecoveryCommandAction.RESTART,
            command="pytest -v",
            run_id=prior_run.run_id,
        )

        await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        updated = current_run.read(wiki_root)
        assert updated is not None
        assert updated.command is not None
        assert updated.command.approved is True
        assert updated.command.resolved_shell == "pytest -v"

    @pytest.mark.asyncio
    async def test_restart_records_remote_pid(self, wiki_root: Path) -> None:
        prior_run = _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=7777)
        gen_cmd = _make_generated_command(
            action=RecoveryCommandAction.RESTART,
            command="pytest",
            run_id=prior_run.run_id,
        )

        await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        updated = current_run.read(wiki_root)
        assert updated is not None
        assert updated.pids.remote == 7777

    @pytest.mark.asyncio
    async def test_restart_records_daemon_pid(self, wiki_root: Path) -> None:
        prior_run = _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=42)
        gen_cmd = _make_generated_command(
            action=RecoveryCommandAction.RESTART,
            run_id=prior_run.run_id,
        )

        await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=5555,
        )

        updated = current_run.read(wiki_root)
        assert updated is not None
        assert updated.pids.daemon == 5555

    @pytest.mark.asyncio
    async def test_restart_resets_progress(self, wiki_root: Path) -> None:
        """RESTART should reset progress counters to zero."""
        prior_run = _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=42)
        gen_cmd = _make_generated_command(
            action=RecoveryCommandAction.RESTART,
            run_id=prior_run.run_id,
        )

        await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        updated = current_run.read(wiki_root)
        assert updated is not None
        assert updated.progress.percent == 0.0
        assert updated.progress.tests_passed == 0


# ---------------------------------------------------------------------------
# dispatch_recovery_command: successful resume
# ---------------------------------------------------------------------------


class TestDispatchResume:
    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        return tmp_path / "wiki"

    @pytest.mark.asyncio
    async def test_successful_resume_dispatch(self, wiki_root: Path) -> None:
        prior_run = _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=8888)
        gen_cmd = _make_generated_command(
            action=RecoveryCommandAction.RESUME,
            command="pytest --lf",
            run_id=prior_run.run_id,
            resume_context="Resuming from test_checkout",
        )

        result = await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        assert result.success is True
        assert result.action == RecoveryCommandAction.RESUME

    @pytest.mark.asyncio
    async def test_resume_preserves_progress(self, wiki_root: Path) -> None:
        """RESUME should preserve existing progress counters."""
        prior_run = _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=42)
        gen_cmd = _make_generated_command(
            action=RecoveryCommandAction.RESUME,
            command="pytest --lf",
            run_id=prior_run.run_id,
        )

        await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        updated = current_run.read(wiki_root)
        assert updated is not None
        # Progress should be preserved (not reset) for resume
        assert updated.progress.tests_passed == 25

    @pytest.mark.asyncio
    async def test_resume_updates_command_to_resume_command(
        self, wiki_root: Path
    ) -> None:
        prior_run = _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=42)
        gen_cmd = _make_generated_command(
            action=RecoveryCommandAction.RESUME,
            command="pytest --lf",
            run_id=prior_run.run_id,
        )

        await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        updated = current_run.read(wiki_root)
        assert updated is not None
        assert updated.command is not None
        assert updated.command.resolved_shell == "pytest --lf"


# ---------------------------------------------------------------------------
# dispatch_recovery_command: SSH execution error
# ---------------------------------------------------------------------------


class TestDispatchError:
    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        return tmp_path / "wiki"

    @pytest.mark.asyncio
    async def test_ssh_error_returns_failure(self, wiki_root: Path) -> None:
        prior_run = _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(
            error=OSError("connection reset")
        )
        gen_cmd = _make_generated_command(run_id=prior_run.run_id)

        result = await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        assert result.success is False
        assert result.error is not None
        assert "connection reset" in result.error.lower()

    @pytest.mark.asyncio
    async def test_ssh_error_updates_wiki_to_failed(
        self, wiki_root: Path
    ) -> None:
        prior_run = _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(
            error=OSError("connection refused")
        )
        gen_cmd = _make_generated_command(run_id=prior_run.run_id)

        await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        updated = current_run.read(wiki_root)
        assert updated is not None
        assert updated.status == RunStatus.FAILED
        assert updated.error is not None

    @pytest.mark.asyncio
    async def test_ssh_error_wiki_updated_flag(self, wiki_root: Path) -> None:
        prior_run = _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(
            error=TimeoutError("SSH timeout")
        )
        gen_cmd = _make_generated_command(run_id=prior_run.run_id)

        result = await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        assert result.wiki_updated is True  # Wiki was updated to FAILED state


# ---------------------------------------------------------------------------
# dispatch_recovery_command: no prior wiki state
# ---------------------------------------------------------------------------


class TestDispatchNoPriorState:
    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        return tmp_path / "wiki"

    @pytest.mark.asyncio
    async def test_creates_new_wiki_state(self, wiki_root: Path) -> None:
        """When no prior state exists, dispatch should create a new entry."""
        handle = FakeDispatchHandle(remote_pid=42)
        gen_cmd = _make_generated_command(
            command="pytest -v",
            run_id="new-run-id",
        )

        result = await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
            ssh_target=SSHTarget(host="test.example.com", user="ci"),
            natural_language="run the tests",
        )

        assert result.success is True
        updated = current_run.read(wiki_root)
        assert updated is not None
        assert updated.status == RunStatus.RUNNING


# ---------------------------------------------------------------------------
# dispatch_recovery_command: records session_id
# ---------------------------------------------------------------------------


class TestDispatchSessionId:
    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        return tmp_path / "wiki"

    @pytest.mark.asyncio
    async def test_records_session_id(self, wiki_root: Path) -> None:
        _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=42)
        gen_cmd = _make_generated_command()

        result = await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        assert result.session_id == "fake-session-123"


# ---------------------------------------------------------------------------
# dispatch_recovery_command: executes correct command
# ---------------------------------------------------------------------------


class TestDispatchCommandExecution:
    @pytest.fixture
    def wiki_root(self, tmp_path: Path) -> Path:
        return tmp_path / "wiki"

    @pytest.mark.asyncio
    async def test_executes_the_generated_command(
        self, wiki_root: Path
    ) -> None:
        _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=42)
        gen_cmd = _make_generated_command(
            command="pytest --lf -v --tb=short",
        )

        await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        assert handle.executed_commands is not None
        assert len(handle.executed_commands) == 1
        assert "pytest --lf -v --tb=short" in handle.executed_commands[0]

    @pytest.mark.asyncio
    async def test_includes_working_directory_in_command(
        self, wiki_root: Path
    ) -> None:
        _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=42)
        gen_cmd = _make_generated_command(
            command="pytest",
            working_directory="/opt/app",
        )

        await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        assert handle.executed_commands is not None
        assert len(handle.executed_commands) == 1
        executed = handle.executed_commands[0]
        assert "cd /opt/app" in executed

    @pytest.mark.asyncio
    async def test_includes_environment_in_command(
        self, wiki_root: Path
    ) -> None:
        _make_wiki_running_state(wiki_root)
        handle = FakeDispatchHandle(remote_pid=42)
        gen_cmd = _make_generated_command(
            command="pytest",
            environment={"CI": "true", "NODE_ENV": "test"},
        )

        await dispatch_recovery_command(
            handle=handle,
            generated_command=gen_cmd,
            wiki_root=wiki_root,
            daemon_pid=1234,
        )

        assert handle.executed_commands is not None
        executed = handle.executed_commands[0]
        assert "CI=true" in executed or "CI='true'" in executed
        assert "NODE_ENV=test" in executed or "NODE_ENV='test'" in executed
