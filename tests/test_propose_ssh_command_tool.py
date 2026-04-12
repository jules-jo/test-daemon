"""Tests for ProposeSSHCommandTool and ApprovalLedger (agent/tools/propose_ssh_command.py).

Comprehensive unit tests covering:
    - ApprovalLedger: record, get, has, consume, pending_count, immutability
    - ApprovalEntry: frozen dataclass, all fields populated
    - Tool specification metadata (name, description, parameters, approval)
    - Execute method: successful proposal (user approves)
    - Execute method: user denial (returns DENIED, terminal)
    - Execute method: user edits the command during approval
    - Input validation: empty/whitespace/missing command, target_host, target_user
    - call_id propagation across all result paths
    - Error handling: callback exceptions (ConnectionError, TimeoutError, etc.)
    - Ledger integration: approved entries recorded correctly
    - JSON output structure validation
    - Callback argument forwarding (command stripped, target_host stripped)
    - BaseTool conformance (spec property, name shortcut)
    - Edge cases: long commands, special characters, concurrent calls

These tests exercise the tool in isolation using mocked async callbacks,
matching the established pattern from test_ask_user_question_tool.py.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolResultStatus,
    ToolSpec,
)
from jules_daemon.agent.tools.base import BaseTool, Tool
from jules_daemon.agent.tools.propose_ssh_command import (
    ApprovalEntry,
    ApprovalLedger,
    ProposeSSHCommandTool,
)


# ---------------------------------------------------------------------------
# ApprovalEntry tests
# ---------------------------------------------------------------------------


class TestApprovalEntry:
    """Verify ApprovalEntry frozen dataclass behavior."""

    def test_creates_with_all_fields(self) -> None:
        entry = ApprovalEntry(
            approval_id="a1",
            command="ls -la",
            target_host="10.0.1.50",
            target_user="root",
        )
        assert entry.approval_id == "a1"
        assert entry.command == "ls -la"
        assert entry.target_host == "10.0.1.50"
        assert entry.target_user == "root"

    def test_is_frozen(self) -> None:
        entry = ApprovalEntry(
            approval_id="a1",
            command="ls",
            target_host="host",
            target_user="user",
        )
        with pytest.raises(AttributeError):
            entry.command = "pwd"  # type: ignore[misc]

    def test_equality(self) -> None:
        entry_a = ApprovalEntry(
            approval_id="a1",
            command="ls",
            target_host="host",
            target_user="user",
        )
        entry_b = ApprovalEntry(
            approval_id="a1",
            command="ls",
            target_host="host",
            target_user="user",
        )
        assert entry_a == entry_b

    def test_inequality_different_command(self) -> None:
        entry_a = ApprovalEntry(
            approval_id="a1",
            command="ls",
            target_host="host",
            target_user="user",
        )
        entry_b = ApprovalEntry(
            approval_id="a1",
            command="pwd",
            target_host="host",
            target_user="user",
        )
        assert entry_a != entry_b


# ---------------------------------------------------------------------------
# ApprovalLedger tests
# ---------------------------------------------------------------------------


class TestApprovalLedger:
    """Verify ApprovalLedger session-scoped approval tracking."""

    def test_empty_ledger_pending_count_is_zero(self) -> None:
        ledger = ApprovalLedger()
        assert ledger.pending_count == 0

    def test_record_approval_increments_pending(self) -> None:
        ledger = ApprovalLedger()
        entry = ApprovalEntry(
            approval_id="a1",
            command="ls",
            target_host="host",
            target_user="user",
        )
        ledger.record_approval(entry)
        assert ledger.pending_count == 1

    def test_record_multiple_approvals(self) -> None:
        ledger = ApprovalLedger()
        for i in range(3):
            entry = ApprovalEntry(
                approval_id=f"a{i}",
                command=f"cmd{i}",
                target_host="host",
                target_user="user",
            )
            ledger.record_approval(entry)
        assert ledger.pending_count == 3

    def test_get_approved_command_returns_entry(self) -> None:
        ledger = ApprovalLedger()
        entry = ApprovalEntry(
            approval_id="a1",
            command="ls -la",
            target_host="10.0.1.50",
            target_user="root",
        )
        ledger.record_approval(entry)

        result = ledger.get_approved_command("a1")
        assert result is not None
        assert result.command == "ls -la"
        assert result.target_host == "10.0.1.50"
        assert result.target_user == "root"

    def test_get_approved_command_returns_none_for_unknown(self) -> None:
        ledger = ApprovalLedger()
        assert ledger.get_approved_command("nonexistent") is None

    def test_has_approved_command_returns_approval_id(self) -> None:
        ledger = ApprovalLedger()
        entry = ApprovalEntry(
            approval_id="a1",
            command="ls",
            target_host="host",
            target_user="user",
        )
        ledger.record_approval(entry)

        result = ledger.has_approved_command("ls", "host")
        assert result == "a1"

    def test_has_approved_command_returns_none_for_mismatch(self) -> None:
        ledger = ApprovalLedger()
        entry = ApprovalEntry(
            approval_id="a1",
            command="ls",
            target_host="host1",
            target_user="user",
        )
        ledger.record_approval(entry)

        # Different host
        assert ledger.has_approved_command("ls", "host2") is None
        # Different command
        assert ledger.has_approved_command("pwd", "host1") is None

    def test_consume_returns_and_removes_entry(self) -> None:
        ledger = ApprovalLedger()
        entry = ApprovalEntry(
            approval_id="a1",
            command="ls",
            target_host="host",
            target_user="user",
        )
        ledger.record_approval(entry)

        consumed = ledger.consume("a1")
        assert consumed is not None
        assert consumed.command == "ls"
        assert ledger.pending_count == 0

    def test_consume_returns_none_for_unknown(self) -> None:
        ledger = ApprovalLedger()
        assert ledger.consume("nonexistent") is None

    def test_consume_is_one_time_use(self) -> None:
        """Once consumed, second consume returns None."""
        ledger = ApprovalLedger()
        entry = ApprovalEntry(
            approval_id="a1",
            command="ls",
            target_host="host",
            target_user="user",
        )
        ledger.record_approval(entry)

        first = ledger.consume("a1")
        second = ledger.consume("a1")
        assert first is not None
        assert second is None

    def test_get_after_consume_returns_none(self) -> None:
        """After consuming an entry, get returns None."""
        ledger = ApprovalLedger()
        entry = ApprovalEntry(
            approval_id="a1",
            command="ls",
            target_host="host",
            target_user="user",
        )
        ledger.record_approval(entry)
        ledger.consume("a1")

        assert ledger.get_approved_command("a1") is None

    def test_has_approved_after_consume_returns_none(self) -> None:
        """After consuming, has_approved_command returns None."""
        ledger = ApprovalLedger()
        entry = ApprovalEntry(
            approval_id="a1",
            command="ls",
            target_host="host",
            target_user="user",
        )
        ledger.record_approval(entry)
        ledger.consume("a1")

        assert ledger.has_approved_command("ls", "host") is None

    def test_overwrite_same_approval_id(self) -> None:
        """Recording with same approval_id overwrites the previous entry."""
        ledger = ApprovalLedger()
        entry1 = ApprovalEntry(
            approval_id="a1",
            command="ls",
            target_host="host",
            target_user="user",
        )
        entry2 = ApprovalEntry(
            approval_id="a1",
            command="pwd",
            target_host="host",
            target_user="user",
        )
        ledger.record_approval(entry1)
        ledger.record_approval(entry2)

        assert ledger.pending_count == 1
        result = ledger.get_approved_command("a1")
        assert result is not None
        assert result.command == "pwd"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ledger() -> ApprovalLedger:
    """Fresh approval ledger for each test."""
    return ApprovalLedger()


@pytest.fixture
def approve_callback() -> AsyncMock:
    """Callback that simulates user approval (approved=True, same command)."""
    return AsyncMock(return_value=(True, "ls -la"))


@pytest.fixture
def deny_callback() -> AsyncMock:
    """Callback that simulates user denial."""
    return AsyncMock(return_value=(False, ""))


@pytest.fixture
def edit_callback() -> AsyncMock:
    """Callback that simulates user editing the command during approval."""
    return AsyncMock(return_value=(True, "ls -la --color=always"))


@pytest.fixture
def error_callback() -> AsyncMock:
    """Callback that raises an exception (simulating IPC failure)."""
    return AsyncMock(side_effect=ConnectionError("IPC channel closed"))


@pytest.fixture
def tool(approve_callback: AsyncMock, ledger: ApprovalLedger) -> ProposeSSHCommandTool:
    """ProposeSSHCommandTool with standard approve callback."""
    return ProposeSSHCommandTool(
        confirm_callback=approve_callback,
        ledger=ledger,
    )


def _valid_args(
    *,
    command: str = "ls -la",
    target_host: str = "10.0.1.50",
    target_user: str = "root",
    explanation: str = "List files",
    call_id: str = "c1",
) -> dict[str, Any]:
    """Build a valid args dict for ProposeSSHCommandTool.execute()."""
    return {
        "command": command,
        "target_host": target_host,
        "target_user": target_user,
        "explanation": explanation,
        "_call_id": call_id,
    }


# ---------------------------------------------------------------------------
# Tool specification and metadata
# ---------------------------------------------------------------------------


class TestProposeSSHCommandToolSpec:
    """Verify tool spec metadata and protocol conformance."""

    def test_tool_name(self, tool: ProposeSSHCommandTool) -> None:
        assert tool.name == "propose_ssh_command"

    def test_spec_returns_tool_spec(self, tool: ProposeSSHCommandTool) -> None:
        assert isinstance(tool.spec, ToolSpec)

    def test_spec_name_matches(self, tool: ProposeSSHCommandTool) -> None:
        assert tool.spec.name == "propose_ssh_command"

    def test_spec_has_nonempty_description(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        assert isinstance(tool.spec.description, str)
        assert len(tool.spec.description) > 0

    def test_description_mentions_ssh_or_command(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        desc_lower = tool.spec.description.lower()
        assert "ssh" in desc_lower or "command" in desc_lower

    def test_description_mentions_approval(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        desc_lower = tool.spec.description.lower()
        assert "approv" in desc_lower

    def test_requires_confirm_prompt_approval(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        assert tool.spec.approval is ApprovalRequirement.CONFIRM_PROMPT

    def test_is_not_read_only(self, tool: ProposeSSHCommandTool) -> None:
        assert not tool.spec.is_read_only

    def test_has_command_parameter(self, tool: ProposeSSHCommandTool) -> None:
        param_names = {p.name for p in tool.spec.parameters}
        assert "command" in param_names

    def test_command_parameter_is_required(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        cmd_param = next(
            p for p in tool.spec.parameters if p.name == "command"
        )
        assert cmd_param.required is True

    def test_command_parameter_is_string_type(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        cmd_param = next(
            p for p in tool.spec.parameters if p.name == "command"
        )
        assert cmd_param.json_type == "string"

    def test_has_target_host_parameter(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        param_names = {p.name for p in tool.spec.parameters}
        assert "target_host" in param_names

    def test_target_host_is_required(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        param = next(
            p for p in tool.spec.parameters if p.name == "target_host"
        )
        assert param.required is True

    def test_has_target_user_parameter(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        param_names = {p.name for p in tool.spec.parameters}
        assert "target_user" in param_names

    def test_target_user_is_required(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        param = next(
            p for p in tool.spec.parameters if p.name == "target_user"
        )
        assert param.required is True

    def test_has_explanation_parameter(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        param_names = {p.name for p in tool.spec.parameters}
        assert "explanation" in param_names

    def test_explanation_is_optional(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        param = next(
            p for p in tool.spec.parameters if p.name == "explanation"
        )
        assert param.required is False

    def test_explanation_default_is_empty_string(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        param = next(
            p for p in tool.spec.parameters if p.name == "explanation"
        )
        assert param.default == ""

    def test_is_base_tool_subclass(self) -> None:
        assert issubclass(ProposeSSHCommandTool, BaseTool)

    def test_instance_satisfies_tool_protocol(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        assert isinstance(tool, Tool)

    def test_openai_schema_structure(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        schema = tool.spec.to_openai_function_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "propose_ssh_command"
        assert "description" in fn
        assert "parameters" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "command" in params["properties"]
        assert "target_host" in params["properties"]
        assert "target_user" in params["properties"]
        assert "explanation" in params["properties"]

    def test_openai_schema_required_fields(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        schema = tool.spec.to_openai_function_schema()
        required = schema["function"]["parameters"]["required"]
        assert "command" in required
        assert "target_host" in required
        assert "target_user" in required
        assert "explanation" not in required

    def test_openai_schema_is_json_serializable(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        schema = tool.spec.to_openai_function_schema()
        serialized = json.dumps(schema)
        deserialized = json.loads(serialized)
        assert deserialized["function"]["name"] == "propose_ssh_command"


# ---------------------------------------------------------------------------
# Execute: successful approval flow
# ---------------------------------------------------------------------------


class TestProposeSSHCommandApproval:
    """Verify successful proposal/approval flow."""

    @pytest.mark.asyncio
    async def test_returns_success_on_approval(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert result.status is ToolResultStatus.SUCCESS
        assert result.is_success

    @pytest.mark.asyncio
    async def test_output_contains_approved_true(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        data = json.loads(result.output)
        assert data["approved"] is True

    @pytest.mark.asyncio
    async def test_output_contains_approval_id(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        data = json.loads(result.output)
        assert "approval_id" in data
        assert data["approval_id"].startswith("approval-")

    @pytest.mark.asyncio
    async def test_output_contains_command(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command="ls -la"))

        data = json.loads(result.output)
        assert data["command"] == "ls -la"

    @pytest.mark.asyncio
    async def test_output_edited_false_when_unchanged(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        approve_callback.return_value = (True, "ls -la")
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command="ls -la"))

        data = json.loads(result.output)
        assert data["edited"] is False

    @pytest.mark.asyncio
    async def test_success_result_is_not_terminal(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        result = await tool.execute(_valid_args())
        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_success_result_has_no_error_message(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        result = await tool.execute(_valid_args())
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_output_is_valid_json(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        result = await tool.execute(_valid_args())
        data = json.loads(result.output)
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_callback_receives_stripped_command(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        await tool.execute(_valid_args(command="  ls -la  "))

        args, _ = approve_callback.call_args
        assert args[0] == "ls -la"

    @pytest.mark.asyncio
    async def test_callback_receives_stripped_host(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        await tool.execute(_valid_args(target_host="  10.0.1.50  "))

        args, _ = approve_callback.call_args
        assert args[1] == "10.0.1.50"

    @pytest.mark.asyncio
    async def test_callback_receives_explanation(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        await tool.execute(_valid_args(explanation="Running test suite"))

        args, _ = approve_callback.call_args
        assert args[2] == "Running test suite"

    @pytest.mark.asyncio
    async def test_default_explanation_is_empty(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        await tool.execute({
            "command": "ls",
            "target_host": "host",
            "target_user": "user",
            "_call_id": "c1",
        })

        args, _ = approve_callback.call_args
        assert args[2] == ""


# ---------------------------------------------------------------------------
# Execute: user edits the command
# ---------------------------------------------------------------------------


class TestProposeSSHCommandEdited:
    """Verify that user-edited commands are captured correctly."""

    @pytest.mark.asyncio
    async def test_edited_command_in_output(
        self, edit_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=edit_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command="ls -la"))

        data = json.loads(result.output)
        assert data["command"] == "ls -la --color=always"

    @pytest.mark.asyncio
    async def test_edited_flag_true_when_changed(
        self, edit_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=edit_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command="ls -la"))

        data = json.loads(result.output)
        assert data["edited"] is True

    @pytest.mark.asyncio
    async def test_ledger_records_edited_command(
        self, edit_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=edit_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command="ls -la"))

        data = json.loads(result.output)
        approval_id = data["approval_id"]
        entry = ledger.get_approved_command(approval_id)
        assert entry is not None
        assert entry.command == "ls -la --color=always"

    @pytest.mark.asyncio
    async def test_edited_command_still_succeeds(
        self, edit_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=edit_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert result.is_success


# ---------------------------------------------------------------------------
# Execute: user denial (DENIED, terminal)
# ---------------------------------------------------------------------------


class TestProposeSSHCommandDenial:
    """Verify user denial returns DENIED (terminal) status."""

    @pytest.mark.asyncio
    async def test_denied_returns_denied_status(
        self, deny_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=deny_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert result.status is ToolResultStatus.DENIED
        assert result.is_denied

    @pytest.mark.asyncio
    async def test_denied_result_is_terminal(
        self, deny_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=deny_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert result.is_terminal

    @pytest.mark.asyncio
    async def test_denied_output_contains_approved_false(
        self, deny_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=deny_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        data = json.loads(result.output)
        assert data["approved"] is False

    @pytest.mark.asyncio
    async def test_denied_has_error_message(
        self, deny_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=deny_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert result.error_message is not None
        assert "denied" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_denied_does_not_record_in_ledger(
        self, deny_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        """Denied proposals must NOT be recorded in the approval ledger."""
        tool = ProposeSSHCommandTool(
            confirm_callback=deny_callback, ledger=ledger,
        )
        await tool.execute(_valid_args())

        assert ledger.pending_count == 0

    @pytest.mark.asyncio
    async def test_denied_tool_name_correct(
        self, deny_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=deny_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert result.tool_name == "propose_ssh_command"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestProposeSSHCommandValidation:
    """Verify argument validation returns ERROR for invalid inputs."""

    @pytest.mark.asyncio
    async def test_empty_command_returns_error(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command=""))

        assert result.status is ToolResultStatus.ERROR
        assert "command" in (result.error_message or "").lower()
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitespace_command_returns_error(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command="   "))

        assert result.status is ToolResultStatus.ERROR
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_command_key_returns_error(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute({
            "target_host": "host",
            "target_user": "user",
            "_call_id": "v1",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "command" in (result.error_message or "").lower()
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_target_host_returns_error(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(target_host=""))

        assert result.status is ToolResultStatus.ERROR
        assert "target_host" in (result.error_message or "").lower()
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitespace_target_host_returns_error(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(target_host="  \t  "))

        assert result.status is ToolResultStatus.ERROR
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_target_host_key_returns_error(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute({
            "command": "ls",
            "target_user": "user",
            "_call_id": "v2",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "target_host" in (result.error_message or "").lower()
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_target_user_returns_error(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(target_user=""))

        assert result.status is ToolResultStatus.ERROR
        assert "target_user" in (result.error_message or "").lower()
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitespace_target_user_returns_error(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(target_user="  "))

        assert result.status is ToolResultStatus.ERROR
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_target_user_key_returns_error(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute({
            "command": "ls",
            "target_host": "host",
            "_call_id": "v3",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "target_user" in (result.error_message or "").lower()
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_validation_error_is_not_terminal(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        """Validation errors are ERROR, not DENIED -- agent can self-correct."""
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command=""))

        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_validation_error_output_is_empty(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command=""))

        assert result.output == ""

    @pytest.mark.asyncio
    async def test_validation_error_does_not_record_in_ledger(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        await tool.execute(_valid_args(command=""))

        assert ledger.pending_count == 0


# ---------------------------------------------------------------------------
# Ledger integration
# ---------------------------------------------------------------------------


class TestProposeSSHCommandLedgerIntegration:
    """Verify approved proposals are correctly recorded in the ledger."""

    @pytest.mark.asyncio
    async def test_approved_entry_recorded_in_ledger(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert ledger.pending_count == 1

    @pytest.mark.asyncio
    async def test_ledger_entry_has_correct_command(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        approve_callback.return_value = (True, "ls -la")
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command="ls -la"))

        data = json.loads(result.output)
        entry = ledger.get_approved_command(data["approval_id"])
        assert entry is not None
        assert entry.command == "ls -la"

    @pytest.mark.asyncio
    async def test_ledger_entry_has_correct_target_host(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(
            _valid_args(target_host="192.168.1.100")
        )

        data = json.loads(result.output)
        entry = ledger.get_approved_command(data["approval_id"])
        assert entry is not None
        assert entry.target_host == "192.168.1.100"

    @pytest.mark.asyncio
    async def test_ledger_entry_has_correct_target_user(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(
            _valid_args(target_user="admin")
        )

        data = json.loads(result.output)
        entry = ledger.get_approved_command(data["approval_id"])
        assert entry is not None
        assert entry.target_user == "admin"

    @pytest.mark.asyncio
    async def test_ledger_entry_host_is_stripped(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(
            _valid_args(target_host="  10.0.1.50  ")
        )

        data = json.loads(result.output)
        entry = ledger.get_approved_command(data["approval_id"])
        assert entry is not None
        assert entry.target_host == "10.0.1.50"

    @pytest.mark.asyncio
    async def test_ledger_entry_user_is_stripped(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(
            _valid_args(target_user="  root  ")
        )

        data = json.loads(result.output)
        entry = ledger.get_approved_command(data["approval_id"])
        assert entry is not None
        assert entry.target_user == "root"

    @pytest.mark.asyncio
    async def test_multiple_proposals_have_unique_ids(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        r1 = await tool.execute(_valid_args(call_id="c1"))
        r2 = await tool.execute(_valid_args(call_id="c2"))
        r3 = await tool.execute(_valid_args(call_id="c3"))

        id1 = json.loads(r1.output)["approval_id"]
        id2 = json.loads(r2.output)["approval_id"]
        id3 = json.loads(r3.output)["approval_id"]

        assert id1 != id2
        assert id2 != id3
        assert id1 != id3

    @pytest.mark.asyncio
    async def test_multiple_proposals_increment_ledger(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        await tool.execute(_valid_args(call_id="c1"))
        await tool.execute(_valid_args(call_id="c2"))

        assert ledger.pending_count == 2

    @pytest.mark.asyncio
    async def test_approval_id_format(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        """Approval IDs must start with 'approval-' prefix."""
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        data = json.loads(result.output)
        assert data["approval_id"].startswith("approval-")
        # The hex suffix should be 12 characters
        suffix = data["approval_id"].removeprefix("approval-")
        assert len(suffix) == 12
        # Verify it's valid hex
        int(suffix, 16)


# ---------------------------------------------------------------------------
# call_id propagation
# ---------------------------------------------------------------------------


class TestProposeSSHCommandCallId:
    """Verify _call_id flows through to results across all paths."""

    @pytest.mark.asyncio
    async def test_call_id_in_success_result(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(call_id="unique-success-id"))

        assert result.call_id == "unique-success-id"

    @pytest.mark.asyncio
    async def test_call_id_in_denied_result(
        self, deny_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=deny_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(call_id="unique-denied-id"))

        assert result.call_id == "unique-denied-id"

    @pytest.mark.asyncio
    async def test_call_id_in_validation_error_result(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute({
            "command": "",
            "target_host": "host",
            "target_user": "user",
            "_call_id": "unique-validation-id",
        })

        assert result.call_id == "unique-validation-id"

    @pytest.mark.asyncio
    async def test_call_id_in_exception_result(
        self, error_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=error_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(call_id="unique-exception-id"))

        assert result.call_id == "unique-exception-id"

    @pytest.mark.asyncio
    async def test_default_call_id_when_missing(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        """When _call_id is omitted, defaults to 'propose_ssh_command'."""
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute({
            "command": "ls",
            "target_host": "host",
            "target_user": "user",
        })

        assert result.call_id == "propose_ssh_command"

    @pytest.mark.asyncio
    async def test_tool_name_always_set(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        result = await tool.execute(_valid_args())

        assert result.tool_name == "propose_ssh_command"


# ---------------------------------------------------------------------------
# Error handling: callback exceptions
# ---------------------------------------------------------------------------


class TestProposeSSHCommandErrorHandling:
    """Verify graceful handling of callback exceptions."""

    @pytest.mark.asyncio
    async def test_callback_exception_returns_error(
        self, error_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=error_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert result.status is ToolResultStatus.ERROR
        assert result.is_error

    @pytest.mark.asyncio
    async def test_callback_exception_error_message_contains_details(
        self, error_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=error_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert result.error_message is not None
        assert "IPC channel closed" in result.error_message

    @pytest.mark.asyncio
    async def test_callback_exception_is_not_terminal(
        self, error_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        """Callback errors are ERROR (not DENIED) -- agent loop can retry."""
        tool = ProposeSSHCommandTool(
            confirm_callback=error_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_callback_exception_output_is_empty(
        self, error_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=error_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert result.output == ""

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_record_in_ledger(
        self, error_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        """Failed proposals must NOT be recorded in the ledger."""
        tool = ProposeSSHCommandTool(
            confirm_callback=error_callback, ledger=ledger,
        )
        await tool.execute(_valid_args())

        assert ledger.pending_count == 0

    @pytest.mark.asyncio
    async def test_timeout_error_returns_error_result(
        self, ledger: ApprovalLedger,
    ) -> None:
        callback = AsyncMock(side_effect=TimeoutError("IPC timeout"))
        tool = ProposeSSHCommandTool(
            confirm_callback=callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert result.status is ToolResultStatus.ERROR
        assert "timeout" in (result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_runtime_error_returns_error_result(
        self, ledger: ApprovalLedger,
    ) -> None:
        callback = AsyncMock(side_effect=RuntimeError("unexpected"))
        tool = ProposeSSHCommandTool(
            confirm_callback=callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert result.status is ToolResultStatus.ERROR
        assert "unexpected" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_value_error_from_callback(
        self, ledger: ApprovalLedger,
    ) -> None:
        callback = AsyncMock(side_effect=ValueError("bad input"))
        tool = ProposeSSHCommandTool(
            confirm_callback=callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        assert result.status is ToolResultStatus.ERROR
        assert "bad input" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestProposeSSHCommandEdgeCases:
    """Edge cases: long commands, special characters, extra args."""

    @pytest.mark.asyncio
    async def test_very_long_command(
        self, ledger: ApprovalLedger,
    ) -> None:
        long_cmd = "python3 " + "x" * 5000
        callback = AsyncMock(return_value=(True, long_cmd))
        tool = ProposeSSHCommandTool(
            confirm_callback=callback, ledger=ledger,
        )
        result = await tool.execute(
            _valid_args(command=long_cmd)
        )

        assert result.is_success
        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_command_with_special_characters(
        self, ledger: ApprovalLedger,
    ) -> None:
        cmd = 'echo "hello world" | grep -c "hello" && echo $?'
        callback = AsyncMock(return_value=(True, cmd))
        tool = ProposeSSHCommandTool(
            confirm_callback=callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command=cmd))

        assert result.is_success
        data = json.loads(result.output)
        assert data["command"] == cmd

    @pytest.mark.asyncio
    async def test_command_with_newlines(
        self, ledger: ApprovalLedger,
    ) -> None:
        cmd = "echo line1\necho line2"
        callback = AsyncMock(return_value=(True, cmd))
        tool = ProposeSSHCommandTool(
            confirm_callback=callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command=cmd))

        assert result.is_success

    @pytest.mark.asyncio
    async def test_command_with_unicode(
        self, ledger: ApprovalLedger,
    ) -> None:
        cmd = "echo 'test output'"
        callback = AsyncMock(return_value=(True, cmd))
        tool = ProposeSSHCommandTool(
            confirm_callback=callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command=cmd))

        assert result.is_success

    @pytest.mark.asyncio
    async def test_extra_args_ignored(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute({
            "command": "ls",
            "target_host": "host",
            "target_user": "user",
            "extra_param": "should be ignored",
            "another": 42,
            "_call_id": "edge1",
        })

        assert result.is_success

    @pytest.mark.asyncio
    async def test_hostname_with_port_style(
        self, ledger: ApprovalLedger,
    ) -> None:
        callback = AsyncMock(return_value=(True, "ls"))
        tool = ProposeSSHCommandTool(
            confirm_callback=callback, ledger=ledger,
        )
        result = await tool.execute(
            _valid_args(target_host="server.example.com")
        )

        assert result.is_success
        data = json.loads(result.output)
        entry = ledger.get_approved_command(data["approval_id"])
        assert entry is not None
        assert entry.target_host == "server.example.com"

    @pytest.mark.asyncio
    async def test_ipv6_target_host(
        self, ledger: ApprovalLedger,
    ) -> None:
        callback = AsyncMock(return_value=(True, "ls"))
        tool = ProposeSSHCommandTool(
            confirm_callback=callback, ledger=ledger,
        )
        result = await tool.execute(
            _valid_args(target_host="::1")
        )

        assert result.is_success


# ---------------------------------------------------------------------------
# Callback invocation ordering
# ---------------------------------------------------------------------------


class TestProposeSSHCommandCallbackInvocation:
    """Verify the callback is called exactly once with correct arguments."""

    @pytest.mark.asyncio
    async def test_callback_called_exactly_once(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        await tool.execute(_valid_args())

        assert approve_callback.await_count == 1

    @pytest.mark.asyncio
    async def test_callback_not_called_on_validation_error(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        await tool.execute(_valid_args(command=""))

        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_callback_receives_three_positional_args(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        await tool.execute(_valid_args(
            command="ls",
            target_host="host",
            explanation="test",
        ))

        args, kwargs = approve_callback.call_args
        assert len(args) == 3
        assert args[0] == "ls"       # command (stripped)
        assert args[1] == "host"     # target_host (stripped)
        assert args[2] == "test"     # explanation
        assert len(kwargs) == 0

    @pytest.mark.asyncio
    async def test_multiple_sequential_calls(
        self, ledger: ApprovalLedger,
    ) -> None:
        """Multiple sequential calls each invoke the callback once."""
        call_count = 0

        async def counting_callback(
            cmd: str, host: str, explanation: str,
        ) -> tuple[bool, str]:
            nonlocal call_count
            call_count += 1
            return (True, cmd)

        tool = ProposeSSHCommandTool(
            confirm_callback=counting_callback, ledger=ledger,
        )

        await tool.execute(_valid_args(call_id="seq1"))
        await tool.execute(_valid_args(call_id="seq2"))
        await tool.execute(_valid_args(call_id="seq3"))

        assert call_count == 3
        assert ledger.pending_count == 3


# ---------------------------------------------------------------------------
# ToolResult serialization (to_openai_tool_message, to_llm_message)
# ---------------------------------------------------------------------------


class TestProposeSSHCommandResultSerialization:
    """Verify ToolResult serialization for conversation history."""

    @pytest.mark.asyncio
    async def test_success_openai_message_format(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        result = await tool.execute(_valid_args(call_id="ser1"))

        msg = result.to_openai_tool_message()
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "ser1"
        data = json.loads(msg["content"])
        assert data["approved"] is True
        assert "approval_id" in data

    @pytest.mark.asyncio
    async def test_denied_openai_message_has_error_prefix(
        self, deny_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=deny_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(call_id="ser2"))

        msg = result.to_openai_tool_message()
        assert msg["content"].startswith("ERROR:")

    @pytest.mark.asyncio
    async def test_success_llm_message_format(
        self, tool: ProposeSSHCommandTool,
    ) -> None:
        result = await tool.execute(_valid_args(call_id="ser3"))

        text = result.to_llm_message()
        assert "[propose_ssh_command]" in text
        assert "success" in text.lower()

    @pytest.mark.asyncio
    async def test_denied_llm_message_format(
        self, deny_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=deny_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(call_id="ser4"))

        text = result.to_llm_message()
        assert "[propose_ssh_command]" in text
        assert "DENIED" in text

    @pytest.mark.asyncio
    async def test_error_llm_message_format(
        self, error_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        tool = ProposeSSHCommandTool(
            confirm_callback=error_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(call_id="ser5"))

        text = result.to_llm_message()
        assert "[propose_ssh_command]" in text
        assert "ERROR" in text


# ---------------------------------------------------------------------------
# Approval enforcement (constraint: execute_ssh can only run approved commands)
# ---------------------------------------------------------------------------


class TestProposeSSHCommandApprovalEnforcement:
    """Verify the approval_enforcement evaluation principle.

    execute_ssh can only run commands previously approved by propose_ssh_command.
    These tests verify that propose_ssh_command correctly populates the shared
    ledger that execute_ssh reads.
    """

    @pytest.mark.asyncio
    async def test_approved_command_retrievable_by_approval_id(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        """After approval, execute_ssh can look up the command by approval_id."""
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args(command="ls -la"))

        data = json.loads(result.output)
        entry = ledger.get_approved_command(data["approval_id"])
        assert entry is not None
        assert entry.command == "ls -la"

    @pytest.mark.asyncio
    async def test_approved_command_consumable(
        self, approve_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        """execute_ssh will consume the approval; verify consume works."""
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=ledger,
        )
        result = await tool.execute(_valid_args())

        data = json.loads(result.output)
        consumed = ledger.consume(data["approval_id"])
        assert consumed is not None
        # Second consume fails
        assert ledger.consume(data["approval_id"]) is None

    @pytest.mark.asyncio
    async def test_denied_command_not_in_ledger(
        self, deny_callback: AsyncMock, ledger: ApprovalLedger,
    ) -> None:
        """Denied commands must NOT appear in the ledger at all."""
        tool = ProposeSSHCommandTool(
            confirm_callback=deny_callback, ledger=ledger,
        )
        await tool.execute(_valid_args())

        assert ledger.pending_count == 0
        assert ledger.has_approved_command("ls -la", "10.0.1.50") is None

    @pytest.mark.asyncio
    async def test_shared_ledger_between_propose_and_execute(
        self, approve_callback: AsyncMock,
    ) -> None:
        """Both tools share the same ledger instance."""
        shared_ledger = ApprovalLedger()
        tool = ProposeSSHCommandTool(
            confirm_callback=approve_callback, ledger=shared_ledger,
        )
        result = await tool.execute(_valid_args())

        data = json.loads(result.output)
        # The same ledger object should have the entry
        assert shared_ledger.get_approved_command(data["approval_id"]) is not None
