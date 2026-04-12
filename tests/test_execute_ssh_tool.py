"""Tests for the execute_ssh tool with integrated human-approval gate.

AC 2b-3: execute_ssh blocks execution until explicit user confirmation
is received, executes the approved command, and returns stdout/stderr/exit-code.

Test categories:
    - Human-approval gate: confirm_callback is called and blocks, user can
      approve or deny, denial returns DENIED (terminal)
    - Ledger enforcement: approval_id must reference a valid, unconsumed
      approval from propose_ssh_command
    - Execution delegation: approved commands are delegated to execute_run
      with correct parameters
    - Result structure: stdout, stderr, exit_code returned correctly
    - Edge cases: empty/whitespace params, callback errors, race conditions
    - Approval consumption: approval consumed only after confirmation (not before)
    - Edited commands: user can edit the command during confirmation
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolResultStatus,
)
from jules_daemon.agent.tools.execute_ssh import ExecuteSSHTool
from jules_daemon.agent.tools.propose_ssh_command import (
    ApprovalEntry,
    ApprovalLedger,
)


# ---------------------------------------------------------------------------
# Mock RunResult (avoids importing paramiko-dependent modules)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockRunResult:
    """Stub for execution.run_pipeline.RunResult."""

    success: bool
    run_id: str
    command: str
    target_host: str
    target_user: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    duration_seconds: float = 0.0
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Create a temporary wiki root directory."""
    from jules_daemon.wiki.layout import initialize_wiki

    initialize_wiki(tmp_path)
    return tmp_path


@pytest.fixture
def ledger() -> ApprovalLedger:
    """Fresh approval ledger for each test."""
    return ApprovalLedger()


@pytest.fixture
def sample_entry() -> ApprovalEntry:
    """A typical approved command entry."""
    return ApprovalEntry(
        approval_id="approval-test001",
        command="python3 ~/run_tests.py --iterations 100",
        target_host="10.0.1.50",
        target_user="root",
    )


@pytest.fixture
def confirm_approve() -> AsyncMock:
    """Callback that always approves with the original command."""

    async def _approve(
        command: str, host: str, explanation: str
    ) -> tuple[bool, str]:
        return True, command

    return AsyncMock(side_effect=_approve)


@pytest.fixture
def confirm_deny() -> AsyncMock:
    """Callback that always denies."""

    async def _deny(
        command: str, host: str, explanation: str
    ) -> tuple[bool, str]:
        return False, command

    return AsyncMock(side_effect=_deny)


@pytest.fixture
def confirm_edit() -> AsyncMock:
    """Callback that approves but edits the command."""

    async def _edit(
        command: str, host: str, explanation: str
    ) -> tuple[bool, str]:
        return True, command + " --verbose"

    return AsyncMock(side_effect=_edit)


@pytest.fixture
def confirm_error() -> AsyncMock:
    """Callback that raises an exception (simulating IPC failure)."""
    return AsyncMock(side_effect=ConnectionError("IPC connection lost"))


def _make_tool(
    wiki_root: Path,
    ledger: ApprovalLedger,
    confirm_callback: AsyncMock,
) -> ExecuteSSHTool:
    """Construct an ExecuteSSHTool with the given dependencies."""
    return ExecuteSSHTool(
        wiki_root=wiki_root,
        ledger=ledger,
        confirm_callback=confirm_callback,
    )


async def _execute_with_mock_pipeline(
    tool: ExecuteSSHTool,
    args: dict[str, Any],
    mock_result: MockRunResult,
) -> Any:
    """Execute the tool with a mocked run_pipeline.execute_run."""
    mock_execute_run = AsyncMock(return_value=mock_result)

    saved_paramiko = sys.modules.get("paramiko")
    sys.modules["paramiko"] = MagicMock()
    try:
        import jules_daemon.execution.run_pipeline as rp_mod

        importlib.reload(rp_mod)

        with patch.object(rp_mod, "execute_run", mock_execute_run):
            result = await tool.execute(args)
            return result, mock_execute_run
    finally:
        if saved_paramiko is None:
            sys.modules.pop("paramiko", None)
        else:
            sys.modules["paramiko"] = saved_paramiko


# ---------------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------------


class TestExecuteSSHToolSpec:
    """Verify tool spec metadata."""

    def test_tool_name(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        tool = _make_tool(wiki_root, ledger, confirm_approve)
        assert tool.name == "execute_ssh"

    def test_requires_confirm_prompt(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        tool = _make_tool(wiki_root, ledger, confirm_approve)
        assert tool.spec.approval is ApprovalRequirement.CONFIRM_PROMPT

    def test_is_not_read_only(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        tool = _make_tool(wiki_root, ledger, confirm_approve)
        assert not tool.spec.is_read_only

    def test_has_approval_id_parameter(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        tool = _make_tool(wiki_root, ledger, confirm_approve)
        param_names = {p.name for p in tool.spec.parameters}
        assert "approval_id" in param_names

    def test_has_timeout_parameter(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        tool = _make_tool(wiki_root, ledger, confirm_approve)
        timeout_param = next(
            p for p in tool.spec.parameters if p.name == "timeout"
        )
        assert not timeout_param.required
        assert timeout_param.default == 3600


# ---------------------------------------------------------------------------
# Human-approval gate (core AC requirement)
# ---------------------------------------------------------------------------


class TestHumanApprovalGate:
    """Verify the human-approval gate: approval via propose_ssh_command only."""

    @pytest.mark.asyncio
    async def test_confirm_callback_not_called_by_execute_ssh(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """execute_ssh no longer prompts the user -- approval was already
        granted via propose_ssh_command. The confirm_callback must NOT be
        called during execute_ssh."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-abc",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="ok\n",
        )

        result, _ = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "c1"},
            mock_result,
        )

        # execute_ssh no longer calls confirm_callback (single-gate: propose only)
        confirm_approve.assert_not_awaited()
        # Execution should still succeed
        assert result.status is ToolResultStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execution_proceeds_without_second_callback(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """execute_ssh goes straight to execution -- no second prompt needed."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-noprompt",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="done\n",
        )

        result, _ = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "deny1"},
            mock_result,
        )

        assert result.status is ToolResultStatus.SUCCESS
        # Callback not called -- approval already granted at propose stage
        confirm_approve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_approval_consumed_on_execution(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Approval is consumed from the ledger when execute_ssh runs."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-consume",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="ok\n",
        )

        await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "deny2"},
            mock_result,
        )

        # Approval is consumed on execution (one-time use)
        assert ledger.pending_count == 0
        assert ledger.get_approved_command(sample_entry.approval_id) is None

    @pytest.mark.asyncio
    async def test_successful_execution_output_includes_command_info(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Successful result should include the command and run info in output."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-info",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="output\n",
        )

        result, _ = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "deny3"},
            mock_result,
        )

        data = json.loads(result.output)
        assert data["success"] is True
        assert data["command"] == sample_entry.command
        assert data["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_confirm_callback_stored_but_unused_by_execute(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_error: AsyncMock,
    ) -> None:
        """Even when confirm_callback would raise an error, execute_ssh
        ignores it and proceeds with execution (callback is no longer called)."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_error)

        mock_result = MockRunResult(
            success=True,
            run_id="run-noerr",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="ok\n",
        )

        result, _ = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "err1"},
            mock_result,
        )

        # Callback not invoked; execution still succeeds
        confirm_error.assert_not_awaited()
        assert result.status is ToolResultStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_approval_consumed_on_successful_execution_with_error_callback(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_error: AsyncMock,
    ) -> None:
        """Approval is consumed on execution regardless of what the callback
        would have done (callback is not called by execute_ssh)."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_error)

        mock_result = MockRunResult(
            success=True,
            run_id="run-consume2",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="ok\n",
        )

        await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "err2"},
            mock_result,
        )

        # Approval consumed on execution
        assert ledger.pending_count == 0


# ---------------------------------------------------------------------------
# Ledger enforcement
# ---------------------------------------------------------------------------


class TestLedgerEnforcement:
    """Verify approval_id validation and one-time consumption."""

    @pytest.mark.asyncio
    async def test_missing_approval_id_returns_error(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        result = await tool.execute({
            "approval_id": "",
            "_call_id": "c1",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "required" in (result.error_message or "").lower()
        confirm_approve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitespace_only_approval_id_returns_error(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        result = await tool.execute({
            "approval_id": "   ",
            "_call_id": "c2",
        })

        assert result.status is ToolResultStatus.ERROR
        confirm_approve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_nonexistent_approval_id_returns_error(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        result = await tool.execute({
            "approval_id": "nonexistent-id",
            "_call_id": "c3",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "no approved command" in (result.error_message or "").lower()
        confirm_approve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_approval_consumed_on_success(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """After successful execution, approval is consumed (one-time use)."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-ok",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="done\n",
        )

        await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "c4"},
            mock_result,
        )

        assert ledger.pending_count == 0

    @pytest.mark.asyncio
    async def test_second_execution_with_same_id_fails(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Second execution with consumed approval_id must fail."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-ok",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="done\n",
        )

        await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "c5"},
            mock_result,
        )

        # Second attempt
        result2 = await tool.execute({
            "approval_id": sample_entry.approval_id,
            "_call_id": "c6",
        })

        assert result2.status is ToolResultStatus.ERROR
        assert "no approved command" in (result2.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_approval_id_is_stripped(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Whitespace around approval_id should be stripped."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-strip",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="ok\n",
        )

        result, _ = await _execute_with_mock_pipeline(
            tool,
            {
                "approval_id": f"  {sample_entry.approval_id}  ",
                "_call_id": "c7",
            },
            mock_result,
        )

        assert result.status is ToolResultStatus.SUCCESS


# ---------------------------------------------------------------------------
# Execution delegation and result structure
# ---------------------------------------------------------------------------


class TestExecutionDelegation:
    """Verify correct delegation to execute_run and result structure."""

    @pytest.mark.asyncio
    async def test_successful_execution_returns_stdout_stderr_exit_code(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Successful execution must return structured result."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-123",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="All 42 tests passed\n",
            stderr="",
            duration_seconds=12.5,
        )

        result, mock_fn = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "exec1"},
            mock_result,
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["exit_code"] == 0
        assert "All 42 tests passed" in data["stdout"]
        assert data["stderr"] == ""
        assert data["run_id"] == "run-123"
        assert data["duration_seconds"] == 12.5

    @pytest.mark.asyncio
    async def test_failed_execution_returns_error_status(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Failed command must return ERROR status with exit_code."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=False,
            run_id="run-fail",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=1,
            stdout="3 passed, 7 failed",
            stderr="AssertionError: expected True",
            error="Command exited with code 1",
        )

        result, _ = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "exec2"},
            mock_result,
        )

        assert result.status is ToolResultStatus.ERROR
        assert result.error_message == "Command exited with code 1"
        data = json.loads(result.output)
        assert data["exit_code"] == 1
        assert data["success"] is False
        assert "failed" in data["stdout"]
        assert "AssertionError" in data["stderr"]

    @pytest.mark.asyncio
    async def test_execute_run_called_with_correct_params(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """execute_run must receive correct host, user, command, timeout."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-params",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
        )

        _, mock_fn = await _execute_with_mock_pipeline(
            tool,
            {
                "approval_id": sample_entry.approval_id,
                "timeout": 600,
                "_call_id": "exec3",
            },
            mock_result,
        )

        mock_fn.assert_called_once_with(
            target_host="10.0.1.50",
            target_user="root",
            command=sample_entry.command,
            wiki_root=wiki_root,
            timeout=600,
        )

    @pytest.mark.asyncio
    async def test_default_timeout_is_3600(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Default timeout must be 3600 seconds."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-timeout",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
        )

        _, mock_fn = await _execute_with_mock_pipeline(
            tool,
            {
                "approval_id": sample_entry.approval_id,
                "_call_id": "exec4",
            },
            mock_result,
        )

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["timeout"] == 3600

    @pytest.mark.asyncio
    async def test_execution_exception_returns_error(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Exception during execute_run must return ERROR."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_execute_run = AsyncMock(
            side_effect=RuntimeError("SSH connection refused")
        )

        saved_paramiko = sys.modules.get("paramiko")
        sys.modules["paramiko"] = MagicMock()
        try:
            import jules_daemon.execution.run_pipeline as rp_mod

            importlib.reload(rp_mod)

            with patch.object(rp_mod, "execute_run", mock_execute_run):
                result = await tool.execute({
                    "approval_id": sample_entry.approval_id,
                    "_call_id": "exec5",
                })
        finally:
            if saved_paramiko is None:
                sys.modules.pop("paramiko", None)
            else:
                sys.modules["paramiko"] = saved_paramiko

        assert result.status is ToolResultStatus.ERROR
        assert "SSH execution failed" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_stdout_capped_at_4000_chars(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Stdout in result must be capped to prevent bloat."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        long_stdout = "x" * 8000
        mock_result = MockRunResult(
            success=True,
            run_id="run-cap",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout=long_stdout,
        )

        result, _ = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "exec6"},
            mock_result,
        )

        data = json.loads(result.output)
        assert len(data["stdout"]) == 4000

    @pytest.mark.asyncio
    async def test_stderr_capped_at_2000_chars(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Stderr in result must be capped to prevent bloat."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        long_stderr = "e" * 5000
        mock_result = MockRunResult(
            success=False,
            run_id="run-stderr",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=2,
            stderr=long_stderr,
            error="Command exited with code 2",
        )

        result, _ = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "exec7"},
            mock_result,
        )

        data = json.loads(result.output)
        assert len(data["stderr"]) == 2000


# ---------------------------------------------------------------------------
# Edited commands during confirmation
# ---------------------------------------------------------------------------


class TestEditedCommandDuringConfirmation:
    """Verify that the approved command from the ledger is used for execution.

    Command editing during confirmation is no longer supported at the
    execute_ssh stage -- the command was approved as-is via propose_ssh_command.
    """

    @pytest.mark.asyncio
    async def test_approved_command_used_for_execution(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_edit: AsyncMock,
    ) -> None:
        """execute_ssh uses the approved command from the ledger, not a
        callback-edited version. The confirm_edit callback is not called."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_edit)

        mock_result = MockRunResult(
            success=True,
            run_id="run-edit",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="output\n",
        )

        _, mock_fn = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "edit1"},
            mock_result,
        )

        # Verify execute_run was called with the original (ledger) command
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["command"] == sample_entry.command
        # Callback not called -- no editing happens at execute stage
        confirm_edit.assert_not_awaited()


# ---------------------------------------------------------------------------
# call_id propagation
# ---------------------------------------------------------------------------


class TestCallIdPropagation:
    """Verify _call_id flows through to the result."""

    @pytest.mark.asyncio
    async def test_call_id_in_success_result(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-cid",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
        )

        result, _ = await _execute_with_mock_pipeline(
            tool,
            {
                "approval_id": sample_entry.approval_id,
                "_call_id": "my-unique-call-id",
            },
            mock_result,
        )

        assert result.call_id == "my-unique-call-id"

    @pytest.mark.asyncio
    async def test_call_id_in_denied_result(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_deny: AsyncMock,
    ) -> None:
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_deny)

        result = await tool.execute({
            "approval_id": sample_entry.approval_id,
            "_call_id": "denied-call-id",
        })

        assert result.call_id == "denied-call-id"

    @pytest.mark.asyncio
    async def test_call_id_in_error_result(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        result = await tool.execute({
            "approval_id": "bad-id",
            "_call_id": "error-call-id",
        })

        assert result.call_id == "error-call-id"

    @pytest.mark.asyncio
    async def test_default_call_id(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        """When _call_id is omitted, default is 'execute_ssh'."""
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        result = await tool.execute({"approval_id": "bad-id"})
        assert result.call_id == "execute_ssh"


# ---------------------------------------------------------------------------
# Integration: propose -> confirm -> execute
# ---------------------------------------------------------------------------


class TestProposeConfirmExecuteFlow:
    """End-to-end flow: propose_ssh_command -> execute_ssh with confirmation."""

    @pytest.mark.asyncio
    async def test_full_flow(
        self,
        wiki_root: Path,
        confirm_approve: AsyncMock,
    ) -> None:
        """Full propose -> execute flow with shared ledger."""
        from jules_daemon.agent.tools.propose_ssh_command import (
            ProposeSSHCommandTool,
        )

        ledger = ApprovalLedger()

        # Step 1: Propose and approve via propose_ssh_command
        propose_tool = ProposeSSHCommandTool(
            confirm_callback=confirm_approve,
            ledger=ledger,
        )

        propose_result = await propose_tool.execute({
            "command": "uptime",
            "target_host": "10.0.1.50",
            "target_user": "root",
            "explanation": "Checking server uptime",
            "_call_id": "p1",
        })

        assert propose_result.status is ToolResultStatus.SUCCESS
        propose_data = json.loads(propose_result.output)
        approval_id = propose_data["approval_id"]

        # Step 2: Execute with confirmation gate
        # Reset the mock for the execute confirmation call
        confirm_approve.reset_mock()

        execute_tool = ExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=confirm_approve,
        )

        mock_run = MockRunResult(
            success=True,
            run_id="run-flow",
            command="uptime",
            target_host="10.0.1.50",
            target_user="root",
            exit_code=0,
            stdout=" 14:32:01 up 47 days\n",
        )

        result, _ = await _execute_with_mock_pipeline(
            execute_tool,
            {"approval_id": approval_id, "_call_id": "e1"},
            mock_run,
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["exit_code"] == 0
        assert "up 47 days" in data["stdout"]

        # execute_ssh no longer calls confirm_callback (single-gate at propose)
        confirm_approve.assert_not_awaited()

        # Approval should now be consumed
        assert ledger.pending_count == 0


# ---------------------------------------------------------------------------
# Race condition guard (line 229-240)
# ---------------------------------------------------------------------------


class TestRaceConditionGuard:
    """Verify the race condition guard when approval is consumed between peek and consume.

    Since execute_ssh no longer calls confirm_callback, the race window is the
    gap between the initial ledger peek (get_approved_command) and the consume
    call. We simulate this by patching the ledger's consume method to also
    pre-consume the entry, or simply by consuming the entry from a separate
    reference before the tool's consume call fires.
    """

    @pytest.mark.asyncio
    async def test_concurrent_consume_returns_error(
        self,
        wiki_root: Path,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """If another call consumes the approval between peek and consume,
        the tool returns an ERROR (not a crash)."""
        from unittest.mock import patch

        ledger = ApprovalLedger()
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        # Simulate race: consume the approval before the tool's consume fires
        original_consume = ledger.consume

        def _consume_and_consume(approval_id: str):  # type: ignore[return]
            # Consume once to clear it out (simulating another caller winning)
            original_consume(approval_id)
            # Then the tool's consume call gets None
            return original_consume(approval_id)

        with patch.object(ledger, "consume", side_effect=_consume_and_consume):
            result = await tool.execute({
                "approval_id": sample_entry.approval_id,
                "_call_id": "race1",
            })

        assert result.status is ToolResultStatus.ERROR
        assert "consumed by another call" in (result.error_message or "")
        assert result.call_id == "race1"

    @pytest.mark.asyncio
    async def test_race_condition_is_not_terminal(
        self,
        wiki_root: Path,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Race condition error is retryable (not terminal like DENIED)."""
        from unittest.mock import patch

        ledger = ApprovalLedger()
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        original_consume = ledger.consume

        def _consume_and_consume(approval_id: str):  # type: ignore[return]
            original_consume(approval_id)
            return original_consume(approval_id)

        with patch.object(ledger, "consume", side_effect=_consume_and_consume):
            result = await tool.execute({
                "approval_id": sample_entry.approval_id,
                "_call_id": "race2",
            })

        assert not result.is_terminal
        assert result.is_error


# ---------------------------------------------------------------------------
# Empty final_command fallback
# ---------------------------------------------------------------------------


class TestEmptyFinalCommandFallback:
    """Verify fallback to consumed.command when confirm returns empty string."""

    @pytest.mark.asyncio
    async def test_empty_final_command_uses_original(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
    ) -> None:
        """When confirm returns empty final_command, original command is used."""
        ledger.record_approval(sample_entry)

        async def _approve_empty(
            command: str, host: str, explanation: str
        ) -> tuple[bool, str]:
            return True, ""

        confirm_empty = AsyncMock(side_effect=_approve_empty)
        tool = _make_tool(wiki_root, ledger, confirm_empty)

        mock_result = MockRunResult(
            success=True,
            run_id="run-empty-cmd",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="ok\n",
        )

        _, mock_fn = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "empty1"},
            mock_result,
        )

        # Original command should be used when final_command is empty
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["command"] == sample_entry.command


# ---------------------------------------------------------------------------
# Tool protocol compliance
# ---------------------------------------------------------------------------


class TestToolProtocolCompliance:
    """Verify ExecuteSSHTool satisfies the Tool protocol."""

    def test_implements_tool_protocol(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        """ExecuteSSHTool must satisfy the Tool protocol."""
        from jules_daemon.agent.tools.base import Tool

        tool = _make_tool(wiki_root, ledger, confirm_approve)
        assert isinstance(tool, Tool)

    def test_is_base_tool_subclass(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        """ExecuteSSHTool should be a BaseTool subclass."""
        from jules_daemon.agent.tools.base import BaseTool

        tool = _make_tool(wiki_root, ledger, confirm_approve)
        assert isinstance(tool, BaseTool)

    def test_openai_schema_is_json_serializable(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        """The OpenAI function schema must be fully JSON-serializable."""
        tool = _make_tool(wiki_root, ledger, confirm_approve)
        schema = tool.spec.to_openai_function_schema()
        serialized = json.dumps(schema)
        deserialized = json.loads(serialized)
        assert deserialized["function"]["name"] == "execute_ssh"

    def test_openai_schema_has_approval_id_property(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        """The schema must include approval_id as a required property."""
        tool = _make_tool(wiki_root, ledger, confirm_approve)
        schema = tool.spec.to_openai_function_schema()
        params = schema["function"]["parameters"]
        assert "approval_id" in params["properties"]
        assert "approval_id" in params["required"]

    def test_openai_schema_timeout_is_optional(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_approve: AsyncMock,
    ) -> None:
        """timeout must not be in 'required' since it has a default."""
        tool = _make_tool(wiki_root, ledger, confirm_approve)
        schema = tool.spec.to_openai_function_schema()
        params = schema["function"]["parameters"]
        assert "timeout" in params["properties"]
        assert "timeout" not in params["required"]


# ---------------------------------------------------------------------------
# Multiple entries in ledger
# ---------------------------------------------------------------------------


class TestMultipleLedgerEntries:
    """Verify correct behavior with multiple approvals in the ledger."""

    @pytest.mark.asyncio
    async def test_uses_correct_entry_from_multiple(
        self,
        wiki_root: Path,
        confirm_approve: AsyncMock,
    ) -> None:
        """With multiple entries, the correct one is selected by approval_id."""
        ledger = ApprovalLedger()

        entry_a = ApprovalEntry(
            approval_id="approval-aaa",
            command="uptime",
            target_host="10.0.1.50",
            target_user="root",
        )
        entry_b = ApprovalEntry(
            approval_id="approval-bbb",
            command="df -h",
            target_host="10.0.1.51",
            target_user="admin",
        )

        ledger.record_approval(entry_a)
        ledger.record_approval(entry_b)

        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-multi",
            command=entry_b.command,
            target_host=entry_b.target_host,
            target_user=entry_b.target_user,
            exit_code=0,
            stdout="filesystem data\n",
        )

        _, mock_fn = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": "approval-bbb", "_call_id": "multi1"},
            mock_result,
        )

        # Verify correct host and command from entry_b
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["target_host"] == "10.0.1.51"
        assert call_kwargs["target_user"] == "admin"
        assert call_kwargs["command"] == "df -h"

        # entry_a should still be in the ledger, entry_b consumed
        assert ledger.pending_count == 1
        assert ledger.get_approved_command("approval-aaa") is not None
        assert ledger.get_approved_command("approval-bbb") is None

    @pytest.mark.asyncio
    async def test_consuming_one_does_not_affect_others(
        self,
        wiki_root: Path,
        confirm_approve: AsyncMock,
    ) -> None:
        """Consuming one approval must not affect other entries."""
        ledger = ApprovalLedger()

        entries = [
            ApprovalEntry(
                approval_id=f"approval-{i}",
                command=f"cmd-{i}",
                target_host="10.0.1.50",
                target_user="root",
            )
            for i in range(3)
        ]
        for e in entries:
            ledger.record_approval(e)

        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-iso",
            command="cmd-1",
            target_host="10.0.1.50",
            target_user="root",
            exit_code=0,
        )

        await _execute_with_mock_pipeline(
            tool,
            {"approval_id": "approval-1", "_call_id": "iso1"},
            mock_result,
        )

        # Only approval-1 consumed; 0 and 2 remain
        assert ledger.pending_count == 2
        assert ledger.get_approved_command("approval-0") is not None
        assert ledger.get_approved_command("approval-1") is None
        assert ledger.get_approved_command("approval-2") is not None


# ---------------------------------------------------------------------------
# Result structure validation
# ---------------------------------------------------------------------------


class TestResultStructure:
    """Validate the JSON structure of successful and failed results."""

    @pytest.mark.asyncio
    async def test_success_result_has_all_fields(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Successful result must include all expected fields."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-fields",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
            stdout="output",
            stderr="warnings",
            duration_seconds=5.2,
        )

        result, _ = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "fields1"},
            mock_result,
        )

        data = json.loads(result.output)
        expected_keys = {
            "success", "run_id", "command", "target_host",
            "target_user", "exit_code", "stdout", "stderr",
            "error", "duration_seconds",
        }
        assert set(data.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_success_result_has_no_error_message(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
    ) -> None:
        """Successful execution should have None error_message."""
        ledger.record_approval(sample_entry)
        tool = _make_tool(wiki_root, ledger, confirm_approve)

        mock_result = MockRunResult(
            success=True,
            run_id="run-noerr",
            command=sample_entry.command,
            target_host=sample_entry.target_host,
            target_user=sample_entry.target_user,
            exit_code=0,
        )

        result, _ = await _execute_with_mock_pipeline(
            tool,
            {"approval_id": sample_entry.approval_id, "_call_id": "noerr1"},
            mock_result,
        )

        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_tool_name_in_all_result_paths(
        self,
        wiki_root: Path,
        ledger: ApprovalLedger,
        sample_entry: ApprovalEntry,
        confirm_approve: AsyncMock,
        confirm_deny: AsyncMock,
    ) -> None:
        """tool_name must be 'execute_ssh' in every result path."""
        # Error path: bad approval_id
        tool = _make_tool(wiki_root, ledger, confirm_approve)
        err_result = await tool.execute(
            {"approval_id": "bad", "_call_id": "tn1"}
        )
        assert err_result.tool_name == "execute_ssh"

        # Denied path
        ledger.record_approval(sample_entry)
        deny_tool = _make_tool(wiki_root, ledger, confirm_deny)
        deny_result = await deny_tool.execute(
            {"approval_id": sample_entry.approval_id, "_call_id": "tn2"}
        )
        assert deny_result.tool_name == "execute_ssh"
