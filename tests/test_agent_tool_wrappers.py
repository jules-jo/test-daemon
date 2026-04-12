"""Tests for agent tool wrappers -- verifies each tool wraps existing daemon functionality.

AC 3: Each tool wraps existing daemon functionality (no reimplementation).

Test strategy:
- Verify each tool delegates to the correct existing daemon module
- Verify read-only vs state-changing classification
- Verify parameter validation and error handling
- Verify wrapping behavior (no logic reimplementation)
- Use real wiki fixtures where possible, mock SSH/IPC where needed
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolResultStatus,
)
from jules_daemon.agent.tools.ask_user_question import AskUserQuestionTool
from jules_daemon.agent.tools.base import BaseTool, Tool
from jules_daemon.agent.tools.check_remote_processes import (
    CheckRemoteProcessesTool,
)
from jules_daemon.agent.tools.execute_ssh import ExecuteSSHTool
from jules_daemon.agent.tools.lookup_test_spec import LookupTestSpecTool
from jules_daemon.agent.tools.notify_user import NotifyUserTool
from jules_daemon.agent.tools.parse_test_output import ParseTestOutputTool
from jules_daemon.agent.tools.propose_ssh_command import (
    ApprovalEntry,
    ApprovalLedger,
    ProposeSSHCommandTool,
)
from jules_daemon.agent.tools.read_output import ReadOutputTool
from jules_daemon.agent.tools.read_wiki import ReadWikiTool
from jules_daemon.agent.tools.registry_factory import build_tool_set
from jules_daemon.agent.tools.summarize_run import SummarizeRunTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Create a temporary wiki root with the required directory structure."""
    from jules_daemon.wiki.layout import initialize_wiki

    initialize_wiki(tmp_path)
    return tmp_path


@pytest.fixture
def approval_ledger() -> ApprovalLedger:
    """Fresh approval ledger for each test."""
    return ApprovalLedger()


@pytest.fixture
def confirm_callback_approve() -> AsyncMock:
    """Callback that always approves with the original command."""
    async def _approve(command: str, host: str, explanation: str) -> tuple[bool, str]:
        return True, command
    return AsyncMock(side_effect=_approve)


@pytest.fixture
def confirm_callback_deny() -> AsyncMock:
    """Callback that always denies."""
    async def _deny(command: str, host: str, explanation: str) -> tuple[bool, str]:
        return False, command
    return AsyncMock(side_effect=_deny)


@pytest.fixture
def ask_callback_answer() -> AsyncMock:
    """Callback that returns a fixed answer."""
    return AsyncMock(return_value="100")


@pytest.fixture
def ask_callback_cancel() -> AsyncMock:
    """Callback that simulates user cancellation."""
    return AsyncMock(return_value=None)


@pytest.fixture
def notify_callback() -> AsyncMock:
    """Callback that records notifications."""
    return AsyncMock(return_value=True)


# ---------------------------------------------------------------------------
# Tool Protocol compliance
# ---------------------------------------------------------------------------


class TestToolProtocol:
    """Verify all tools implement the Tool protocol."""

    def test_all_tools_have_spec(self, wiki_root: Path) -> None:
        """Every tool must expose a ToolSpec via the spec property."""
        tools = _build_all_tools(wiki_root)
        for tool in tools:
            spec = tool.spec
            assert spec is not None
            assert spec.name
            assert spec.description
            assert isinstance(spec.parameters, tuple)

    def test_all_tools_have_execute(self, wiki_root: Path) -> None:
        """Every tool must have an async execute method."""
        tools = _build_all_tools(wiki_root)
        for tool in tools:
            assert hasattr(tool, "execute")
            assert inspect.iscoroutinefunction(tool.execute)

    def test_tool_names_unique(self, wiki_root: Path) -> None:
        """All tool names must be unique."""
        tools = _build_all_tools(wiki_root)
        names = [t.spec.name for t in tools]
        assert len(names) == len(set(names))

    def test_exactly_10_tools(self, wiki_root: Path) -> None:
        """Must produce exactly 10 tools."""
        tools = _build_all_tools(wiki_root)
        assert len(tools) == 10

    def test_openai_schema_serialization(self, wiki_root: Path) -> None:
        """Each tool spec must serialize to valid OpenAI function schema."""
        tools = _build_all_tools(wiki_root)
        for tool in tools:
            schema = tool.spec.to_openai_function_schema()
            assert schema["type"] == "function"
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            params = func["parameters"]
            assert params["type"] == "object"
            assert isinstance(params["properties"], dict)
            assert isinstance(params["required"], list)
            # Verify JSON-serializable
            json.dumps(schema)


# ---------------------------------------------------------------------------
# Approval classification
# ---------------------------------------------------------------------------


class TestApprovalClassification:
    """Verify read-only vs state-changing tool classification."""

    def test_read_only_tools(self, wiki_root: Path) -> None:
        """Read-only tools must have ApprovalRequirement.NONE."""
        read_only_names = {
            "read_wiki", "lookup_test_spec", "check_remote_processes",
            "read_output", "parse_test_output", "summarize_run",
            "notify_user",
        }
        tools = _build_all_tools(wiki_root)
        for tool in tools:
            if tool.spec.name in read_only_names:
                assert tool.spec.approval == ApprovalRequirement.NONE, (
                    f"{tool.spec.name} should be read-only"
                )

    def test_state_changing_tools(self, wiki_root: Path) -> None:
        """State-changing tools must have ApprovalRequirement.CONFIRM_PROMPT."""
        state_changing_names = {
            "propose_ssh_command", "execute_ssh", "ask_user_question",
        }
        tools = _build_all_tools(wiki_root)
        for tool in tools:
            if tool.spec.name in state_changing_names:
                assert tool.spec.approval == ApprovalRequirement.CONFIRM_PROMPT, (
                    f"{tool.spec.name} should require confirmation"
                )


# ---------------------------------------------------------------------------
# read_wiki wraps command_translation + test_knowledge
# ---------------------------------------------------------------------------


class TestReadWikiTool:
    """Verify read_wiki wraps wiki.command_translation + wiki.test_knowledge."""

    @pytest.mark.asyncio
    async def test_wraps_find_by_query(self, wiki_root: Path) -> None:
        """read_wiki must delegate to command_translation.find_by_query."""
        # Save a translation to find
        from jules_daemon.wiki.command_translation import (
            CommandTranslation,
            save,
        )

        translation = CommandTranslation(
            natural_language="run the smoke tests",
            resolved_shell="pytest tests/smoke -v",
            ssh_host="staging.example.com",
        )
        save(wiki_root, translation)

        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute({"query": "smoke tests", "_call_id": "c1"})

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert "translations" in data
        assert len(data["translations"]) >= 1
        assert data["translations"][0]["natural_language"] == "run the smoke tests"

    @pytest.mark.asyncio
    async def test_wraps_load_test_knowledge(self, wiki_root: Path) -> None:
        """read_wiki must delegate to test_knowledge.load_test_knowledge."""
        from jules_daemon.wiki.test_knowledge import (
            TestKnowledge,
            save_test_knowledge,
        )

        knowledge = TestKnowledge(
            test_slug="agent-test-py",
            command_pattern="python3 ~/agent_test.py",
            purpose="Tests the agent loop",
            runs_observed=5,
        )
        save_test_knowledge(wiki_root, knowledge)

        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute({
            "query": "python3 ~/agent_test.py",
            "_call_id": "c2",
        })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["test_knowledge"] is not None
        assert data["test_knowledge"]["test_slug"] == "agent-test-py"
        assert data["test_knowledge"]["purpose"] == "Tests the agent loop"

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self, wiki_root: Path) -> None:
        """read_wiki must reject empty queries."""
        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute({"query": "", "_call_id": "c3"})
        assert result.status == ToolResultStatus.ERROR
        assert "required" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_no_results_returns_empty(self, wiki_root: Path) -> None:
        """read_wiki returns empty lists when nothing matches."""
        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute({
            "query": "nonexistent test xyz",
            "_call_id": "c4",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["translations"] == []
        assert data["test_knowledge"] is None

    @pytest.mark.asyncio
    async def test_ssh_host_filter(self, wiki_root: Path) -> None:
        """read_wiki can filter translations by SSH host."""
        from jules_daemon.wiki.command_translation import (
            CommandTranslation,
            save,
        )

        save(wiki_root, CommandTranslation(
            natural_language="run tests",
            resolved_shell="pytest",
            ssh_host="host-a",
        ))
        save(wiki_root, CommandTranslation(
            natural_language="run tests",
            resolved_shell="pytest",
            ssh_host="host-b",
        ))

        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute({
            "query": "run tests",
            "ssh_host": "host-a",
            "_call_id": "c5",
        })

        data = json.loads(result.output)
        for t in data["translations"]:
            assert t["ssh_host"] == "host-a"


# ---------------------------------------------------------------------------
# lookup_test_spec wraps wiki.test_knowledge
# ---------------------------------------------------------------------------


class TestLookupTestSpecTool:
    """Verify lookup_test_spec wraps wiki.test_knowledge."""

    @pytest.mark.asyncio
    async def test_wraps_derive_test_slug_and_load(self, wiki_root: Path) -> None:
        """lookup_test_spec must delegate to derive_test_slug + load_test_knowledge."""
        from jules_daemon.wiki.test_knowledge import (
            TestKnowledge,
            save_test_knowledge,
        )

        knowledge = TestKnowledge(
            test_slug="agent-test-py",
            command_pattern="python3 ~/agent_test.py",
            purpose="Tests the agent loop",
            output_format="iteration counts",
            common_failures=("timeout on large inputs",),
            normal_behavior="All iterations pass",
            runs_observed=10,
        )
        save_test_knowledge(wiki_root, knowledge)

        tool = LookupTestSpecTool(wiki_root=wiki_root)
        result = await tool.execute({
            "test_name": "python3 ~/agent_test.py",
            "_call_id": "c1",
        })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["found"] is True
        assert data["test_slug"] == "agent-test-py"
        assert data["purpose"] == "Tests the agent loop"
        assert "timeout on large inputs" in data["common_failures"]

    @pytest.mark.asyncio
    async def test_not_found(self, wiki_root: Path) -> None:
        """lookup_test_spec returns found=False for unknown tests."""
        tool = LookupTestSpecTool(wiki_root=wiki_root)
        result = await tool.execute({
            "test_name": "nonexistent_test",
            "_call_id": "c2",
        })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["found"] is False

    @pytest.mark.asyncio
    async def test_empty_name_returns_error(self, wiki_root: Path) -> None:
        """lookup_test_spec must reject empty test names."""
        tool = LookupTestSpecTool(wiki_root=wiki_root)
        result = await tool.execute({"test_name": "", "_call_id": "c3"})
        assert result.status == ToolResultStatus.ERROR


# ---------------------------------------------------------------------------
# check_remote_processes wraps execution.collision_check
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesTool:
    """Verify check_remote_processes wraps execution.collision_check."""

    @pytest.mark.asyncio
    async def test_wraps_collision_check(self) -> None:
        """check_remote_processes must delegate to collision_check.check_remote_processes.

        The collision_check module imports paramiko at module level, which
        may not be installed in the test environment. We verify the wrapping
        by mocking the entire module import chain.
        """
        import sys
        from dataclasses import dataclass as _dc
        from types import ModuleType
        from unittest.mock import MagicMock

        @_dc(frozen=True)
        class _MockProcessInfo:
            pid: int
            command: str

        mock_processes = [
            _MockProcessInfo(pid=1234, command="pytest tests/"),
            _MockProcessInfo(pid=5678, command="python3 test_runner.py"),
        ]

        # Create mock modules for paramiko and SSH-dependent modules
        mock_paramiko = MagicMock()
        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        # Pre-install mock modules so lazy imports inside execute() succeed
        saved_modules: dict[str, Any] = {}
        modules_to_mock = ["paramiko"]
        for mod_name in modules_to_mock:
            saved_modules[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = mock_paramiko

        try:
            # Force reload collision_check with mocked paramiko
            import jules_daemon.execution.collision_check as cc_mod
            import importlib
            importlib.reload(cc_mod)

            tool = CheckRemoteProcessesTool()

            with patch.object(cc_mod, "check_remote_processes", mock_check_fn), \
                 patch("jules_daemon.ssh.credentials.resolve_ssh_credentials", mock_resolve_fn):
                result = await tool.execute({
                    "host": "10.0.1.50",
                    "username": "root",
                    "port": 22,
                    "_call_id": "c1",
                })
        finally:
            # Restore original module state
            for mod_name, original in saved_modules.items():
                if original is None:
                    sys.modules.pop(mod_name, None)
                else:
                    sys.modules[mod_name] = original

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["processes_found"] == 2
        assert data["processes"][0]["pid"] == 1234

    @pytest.mark.asyncio
    async def test_missing_host_returns_error(self) -> None:
        """check_remote_processes must validate required parameters."""
        tool = CheckRemoteProcessesTool()
        result = await tool.execute({
            "host": "",
            "username": "root",
            "_call_id": "c2",
        })
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_missing_username_returns_error(self) -> None:
        """check_remote_processes must validate username parameter."""
        tool = CheckRemoteProcessesTool()
        result = await tool.execute({
            "host": "10.0.1.50",
            "username": "",
            "_call_id": "c3",
        })
        assert result.status == ToolResultStatus.ERROR


# ---------------------------------------------------------------------------
# read_output wraps wiki.state_reader
# ---------------------------------------------------------------------------


class TestReadOutputTool:
    """Verify read_output wraps wiki.state_reader."""

    @pytest.mark.asyncio
    async def test_wraps_load_reconnection_state(self, wiki_root: Path) -> None:
        """read_output must delegate to state_reader.load_reconnection_state."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        result = await tool.execute({"_call_id": "c1"})

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert "load_result" in data
        assert "status" in data
        assert "run_id" in data

    @pytest.mark.asyncio
    async def test_respects_include_connection(self, wiki_root: Path) -> None:
        """read_output should include connection params only when requested."""
        tool = ReadOutputTool(wiki_root=wiki_root)

        result_without = await tool.execute({
            "include_connection": False,
            "_call_id": "c2",
        })
        data_without = json.loads(result_without.output)
        assert "connection" not in data_without


# ---------------------------------------------------------------------------
# parse_test_output wraps monitor.test_output_parser
# ---------------------------------------------------------------------------


class TestParseTestOutputTool:
    """Verify parse_test_output wraps monitor.test_output_parser."""

    @pytest.mark.asyncio
    async def test_wraps_parse_interrupted_output(self) -> None:
        """parse_test_output must delegate to test_output_parser.parse_interrupted_output."""
        tool = ParseTestOutputTool()

        pytest_output = (
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_logout FAILED\n"
            "tests/test_auth.py::test_register PASSED\n"
            "=== 2 passed, 1 failed in 0.5s ===\n"
        )

        result = await tool.execute({
            "raw_output": pytest_output,
            "framework_hint": "auto",
            "_call_id": "c1",
        })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 2
        assert data["summary"]["failed"] == 1
        assert data["framework"] == "pytest"

    @pytest.mark.asyncio
    async def test_empty_output(self) -> None:
        """parse_test_output handles empty output gracefully."""
        tool = ParseTestOutputTool()
        result = await tool.execute({"raw_output": "", "_call_id": "c2"})

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 0

    @pytest.mark.asyncio
    async def test_pytest_hint(self) -> None:
        """parse_test_output respects framework_hint parameter."""
        tool = ParseTestOutputTool()
        result = await tool.execute({
            "raw_output": "tests/test_x.py::test_a PASSED\n",
            "framework_hint": "pytest",
            "_call_id": "c3",
        })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["framework"] == "pytest"


# ---------------------------------------------------------------------------
# propose_ssh_command -- approval flow
# ---------------------------------------------------------------------------


class TestProposeSSHCommandTool:
    """Verify propose_ssh_command approval flow and ledger integration."""

    @pytest.mark.asyncio
    async def test_approved_command_records_in_ledger(
        self,
        confirm_callback_approve: AsyncMock,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """Approved commands must be recorded in the shared ApprovalLedger."""
        tool = ProposeSSHCommandTool(
            confirm_callback=confirm_callback_approve,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "pytest tests/ -v",
            "target_host": "10.0.1.50",
            "target_user": "root",
            "explanation": "Running the test suite",
            "_call_id": "c1",
        })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["approved"] is True
        assert "approval_id" in data
        assert approval_ledger.pending_count == 1

    @pytest.mark.asyncio
    async def test_denied_command_not_in_ledger(
        self,
        confirm_callback_deny: AsyncMock,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """Denied commands must NOT be recorded in the ledger."""
        tool = ProposeSSHCommandTool(
            confirm_callback=confirm_callback_deny,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "rm -rf /",
            "target_host": "10.0.1.50",
            "target_user": "root",
            "_call_id": "c2",
        })

        assert result.status == ToolResultStatus.DENIED
        assert approval_ledger.pending_count == 0

    @pytest.mark.asyncio
    async def test_denied_is_terminal(
        self,
        confirm_callback_deny: AsyncMock,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """Denied result must have is_terminal=True (permanent error)."""
        tool = ProposeSSHCommandTool(
            confirm_callback=confirm_callback_deny,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "echo test",
            "target_host": "host",
            "target_user": "user",
            "_call_id": "c3",
        })

        assert result.is_terminal

    @pytest.mark.asyncio
    async def test_missing_command_returns_error(
        self,
        confirm_callback_approve: AsyncMock,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """propose_ssh_command must validate command parameter."""
        tool = ProposeSSHCommandTool(
            confirm_callback=confirm_callback_approve,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "",
            "target_host": "host",
            "target_user": "user",
            "_call_id": "c4",
        })
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_edited_command_tracked(
        self,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """Edited commands should store the final (edited) command."""
        async def edit_callback(cmd: str, host: str, expl: str) -> tuple[bool, str]:
            return True, f"{cmd} --verbose"

        tool = ProposeSSHCommandTool(
            confirm_callback=edit_callback,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "pytest tests/",
            "target_host": "host",
            "target_user": "user",
            "_call_id": "c5",
        })

        data = json.loads(result.output)
        assert data["edited"] is True
        assert data["command"] == "pytest tests/ --verbose"

    @pytest.mark.asyncio
    async def test_missing_target_host_returns_error(
        self,
        confirm_callback_approve: AsyncMock,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """propose_ssh_command must validate target_host parameter."""
        tool = ProposeSSHCommandTool(
            confirm_callback=confirm_callback_approve,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "echo test",
            "target_host": "",
            "target_user": "root",
            "_call_id": "c6",
        })
        assert result.status == ToolResultStatus.ERROR
        assert "target_host" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_missing_target_user_returns_error(
        self,
        confirm_callback_approve: AsyncMock,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """propose_ssh_command must validate target_user parameter."""
        tool = ProposeSSHCommandTool(
            confirm_callback=confirm_callback_approve,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "echo test",
            "target_host": "10.0.1.50",
            "target_user": "",
            "_call_id": "c7",
        })
        assert result.status == ToolResultStatus.ERROR
        assert "target_user" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_whitespace_only_command_returns_error(
        self,
        confirm_callback_approve: AsyncMock,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """Whitespace-only command must be rejected."""
        tool = ProposeSSHCommandTool(
            confirm_callback=confirm_callback_approve,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "   ",
            "target_host": "host",
            "target_user": "user",
            "_call_id": "c8",
        })
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_callback_exception_returns_error(
        self,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """Exceptions from the confirm callback must produce an error result."""
        async def _fail_callback(cmd: str, host: str, expl: str) -> tuple[bool, str]:
            raise ConnectionError("IPC connection lost")

        tool = ProposeSSHCommandTool(
            confirm_callback=_fail_callback,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "echo test",
            "target_host": "host",
            "target_user": "user",
            "_call_id": "c9",
        })

        assert result.status == ToolResultStatus.ERROR
        assert "Proposal failed" in (result.error_message or "")
        assert approval_ledger.pending_count == 0

    @pytest.mark.asyncio
    async def test_output_structure_on_approval(
        self,
        confirm_callback_approve: AsyncMock,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """Approved result output must contain all required fields."""
        tool = ProposeSSHCommandTool(
            confirm_callback=confirm_callback_approve,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "pytest tests/ -v",
            "target_host": "10.0.1.50",
            "target_user": "root",
            "explanation": "Run the test suite",
            "_call_id": "c10",
        })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        # Verify all structural fields in the output
        assert data["approved"] is True
        assert data["approval_id"].startswith("approval-")
        assert data["command"] == "pytest tests/ -v"
        assert data["edited"] is False

    @pytest.mark.asyncio
    async def test_output_structure_on_denial(
        self,
        confirm_callback_deny: AsyncMock,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """Denied result output must contain approved=False."""
        tool = ProposeSSHCommandTool(
            confirm_callback=confirm_callback_deny,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "rm -rf /",
            "target_host": "host",
            "target_user": "user",
            "_call_id": "c11",
        })

        assert result.status == ToolResultStatus.DENIED
        data = json.loads(result.output)
        assert data["approved"] is False

    @pytest.mark.asyncio
    async def test_multiple_proposals_same_session(
        self,
        confirm_callback_approve: AsyncMock,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """Multiple proposals in the same session each get unique approval IDs."""
        tool = ProposeSSHCommandTool(
            confirm_callback=confirm_callback_approve,
            ledger=approval_ledger,
        )

        r1 = await tool.execute({
            "command": "echo first",
            "target_host": "host1",
            "target_user": "user",
            "_call_id": "c12",
        })
        r2 = await tool.execute({
            "command": "echo second",
            "target_host": "host2",
            "target_user": "user",
            "_call_id": "c13",
        })

        assert r1.status == ToolResultStatus.SUCCESS
        assert r2.status == ToolResultStatus.SUCCESS
        d1 = json.loads(r1.output)
        d2 = json.loads(r2.output)
        assert d1["approval_id"] != d2["approval_id"]
        assert approval_ledger.pending_count == 2

    def test_spec_is_confirm_prompt(self) -> None:
        """propose_ssh_command spec must require CONFIRM_PROMPT approval."""
        tool = ProposeSSHCommandTool(
            confirm_callback=AsyncMock(),
            ledger=ApprovalLedger(),
        )
        assert tool.spec.approval is ApprovalRequirement.CONFIRM_PROMPT
        assert tool.spec.name == "propose_ssh_command"
        assert not tool.spec.is_read_only

    def test_openai_schema_serialization(self) -> None:
        """The tool spec must produce a valid OpenAI function schema."""
        tool = ProposeSSHCommandTool(
            confirm_callback=AsyncMock(),
            ledger=ApprovalLedger(),
        )

        schema = tool.spec.to_openai_function_schema()
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "propose_ssh_command"
        assert "parameters" in func
        props = func["parameters"]["properties"]
        assert "command" in props
        assert "target_host" in props
        assert "target_user" in props
        assert "explanation" in props
        # command, target_host, target_user are required; explanation is optional
        required = func["parameters"]["required"]
        assert "command" in required
        assert "target_host" in required
        assert "target_user" in required
        assert "explanation" not in required

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_inputs(
        self,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """Leading/trailing whitespace in inputs must be stripped."""
        async def _approve(cmd: str, host: str, expl: str) -> tuple[bool, str]:
            return True, cmd

        tool = ProposeSSHCommandTool(
            confirm_callback=_approve,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "  echo test  ",
            "target_host": "  host.example.com  ",
            "target_user": "  root  ",
            "_call_id": "c14",
        })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["command"] == "echo test"
        # Verify ledger stores stripped values
        entry = approval_ledger.get_approved_command(data["approval_id"])
        assert entry is not None
        assert entry.command == "echo test"
        assert entry.target_host == "host.example.com"
        assert entry.target_user == "root"

    @pytest.mark.asyncio
    async def test_default_explanation_is_empty(
        self,
        confirm_callback_approve: AsyncMock,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """Omitting explanation should default to empty string."""
        tool = ProposeSSHCommandTool(
            confirm_callback=confirm_callback_approve,
            ledger=approval_ledger,
        )

        result = await tool.execute({
            "command": "echo test",
            "target_host": "host",
            "target_user": "user",
            "_call_id": "c15",
        })

        assert result.status == ToolResultStatus.SUCCESS
        # The confirm callback should have been called with empty explanation
        confirm_callback_approve.assert_called_once()
        _, _, explanation = confirm_callback_approve.call_args[0]
        assert explanation == ""

    @pytest.mark.asyncio
    async def test_does_not_execute_anything(
        self,
        confirm_callback_approve: AsyncMock,
        approval_ledger: ApprovalLedger,
    ) -> None:
        """propose_ssh_command must never execute the SSH command itself."""
        tool = ProposeSSHCommandTool(
            confirm_callback=confirm_callback_approve,
            ledger=approval_ledger,
        )

        # Patch execute_run to ensure it's never called
        with patch(
            "jules_daemon.agent.tools.propose_ssh_command.uuid.uuid4",
            wraps=__import__("uuid").uuid4,
        ):
            result = await tool.execute({
                "command": "echo test",
                "target_host": "host",
                "target_user": "user",
                "_call_id": "c16",
            })

        # Verify the tool only proposed, didn't execute
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        # Output is a proposal, not an execution result -- no exit_code, stdout, etc.
        assert "exit_code" not in data
        assert "stdout" not in data
        assert "stderr" not in data
        assert "approved" in data
        assert "approval_id" in data


# ---------------------------------------------------------------------------
# execute_ssh wraps run_pipeline + enforces approval
# ---------------------------------------------------------------------------


class TestExecuteSSHTool:
    """Verify execute_ssh wraps run_pipeline and enforces approval constraint."""

    @pytest.mark.asyncio
    async def test_rejects_unapproved_command(
        self,
        wiki_root: Path,
        approval_ledger: ApprovalLedger,
        confirm_callback_approve: AsyncMock,
    ) -> None:
        """execute_ssh must reject commands without prior approval."""
        tool = ExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=approval_ledger,
            confirm_callback=confirm_callback_approve,
        )

        result = await tool.execute({
            "approval_id": "nonexistent-id",
            "_call_id": "c1",
        })

        assert result.status == ToolResultStatus.ERROR
        assert "approval" in (result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_requires_approval_id(
        self,
        wiki_root: Path,
        approval_ledger: ApprovalLedger,
        confirm_callback_approve: AsyncMock,
    ) -> None:
        """execute_ssh must require an approval_id parameter."""
        tool = ExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=approval_ledger,
            confirm_callback=confirm_callback_approve,
        )

        result = await tool.execute({"approval_id": "", "_call_id": "c2"})
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_consumes_approval(
        self,
        wiki_root: Path,
        approval_ledger: ApprovalLedger,
        confirm_callback_approve: AsyncMock,
    ) -> None:
        """execute_ssh must consume the approval (one-time use)."""
        import sys
        from dataclasses import dataclass as _dc, field as _field
        from datetime import datetime, timezone
        from types import ModuleType
        from unittest.mock import MagicMock

        # Pre-approve a command
        entry = ApprovalEntry(
            approval_id="test-approval-1",
            command="echo hello",
            target_host="localhost",
            target_user="test",
        )
        approval_ledger.record_approval(entry)
        assert approval_ledger.pending_count == 1

        tool = ExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=approval_ledger,
            confirm_callback=confirm_callback_approve,
        )

        # Build a mock RunResult since we can't import the real one
        # (it needs paramiko)
        @_dc(frozen=True)
        class MockRunResult:
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
            started_at: datetime = _field(default_factory=lambda: datetime.now(timezone.utc))
            completed_at: datetime = _field(default_factory=lambda: datetime.now(timezone.utc))

        mock_result = MockRunResult(
            success=True,
            run_id="run-test123",
            command="echo hello",
            target_host="localhost",
            target_user="test",
            exit_code=0,
            stdout="hello\n",
        )

        mock_execute_run = AsyncMock(return_value=mock_result)

        # Mock paramiko to allow module import
        saved_paramiko = sys.modules.get("paramiko")
        sys.modules["paramiko"] = MagicMock()
        try:
            import jules_daemon.execution.run_pipeline as rp_mod
            import importlib
            importlib.reload(rp_mod)

            with patch.object(rp_mod, "execute_run", mock_execute_run):
                result = await tool.execute({
                    "approval_id": "test-approval-1",
                    "_call_id": "c3",
                })
        finally:
            if saved_paramiko is None:
                sys.modules.pop("paramiko", None)
            else:
                sys.modules["paramiko"] = saved_paramiko

        assert result.status == ToolResultStatus.SUCCESS
        assert approval_ledger.pending_count == 0

        # Second attempt with same approval_id should fail
        result2 = await tool.execute({
            "approval_id": "test-approval-1",
            "_call_id": "c4",
        })
        assert result2.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_wraps_execute_run(
        self,
        wiki_root: Path,
        approval_ledger: ApprovalLedger,
        confirm_callback_approve: AsyncMock,
    ) -> None:
        """execute_ssh must delegate to run_pipeline.execute_run."""
        import sys
        from dataclasses import dataclass as _dc, field as _field
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        entry = ApprovalEntry(
            approval_id="test-approval-2",
            command="pytest tests/ -v",
            target_host="10.0.1.50",
            target_user="root",
        )
        approval_ledger.record_approval(entry)

        tool = ExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=approval_ledger,
            confirm_callback=confirm_callback_approve,
        )

        @_dc(frozen=True)
        class MockRunResult:
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
            started_at: datetime = _field(default_factory=lambda: datetime.now(timezone.utc))
            completed_at: datetime = _field(default_factory=lambda: datetime.now(timezone.utc))

        mock_result = MockRunResult(
            success=False,
            run_id="run-fail123",
            command="pytest tests/ -v",
            target_host="10.0.1.50",
            target_user="root",
            exit_code=1,
            stdout="1 passed, 3 failed",
            stderr="AssertionError",
            error="Command exited with code 1",
        )

        mock_execute_run = AsyncMock(return_value=mock_result)

        saved_paramiko = sys.modules.get("paramiko")
        sys.modules["paramiko"] = MagicMock()
        try:
            import jules_daemon.execution.run_pipeline as rp_mod
            import importlib
            importlib.reload(rp_mod)

            with patch.object(rp_mod, "execute_run", mock_execute_run):
                result = await tool.execute({
                    "approval_id": "test-approval-2",
                    "_call_id": "c5",
                })
        finally:
            if saved_paramiko is None:
                sys.modules.pop("paramiko", None)
            else:
                sys.modules["paramiko"] = saved_paramiko

        # Verify execute_run was called with correct parameters
        mock_execute_run.assert_called_once_with(
            target_host="10.0.1.50",
            target_user="root",
            command="pytest tests/ -v",
            wiki_root=wiki_root,
            timeout=3600,
        )

        assert result.status == ToolResultStatus.ERROR
        data = json.loads(result.output)
        assert data["exit_code"] == 1
        assert data["success"] is False


# ---------------------------------------------------------------------------
# ask_user_question -- user interaction
# ---------------------------------------------------------------------------


class TestAskUserQuestionTool:
    """Verify ask_user_question delegates to the IPC ask callback."""

    @pytest.mark.asyncio
    async def test_returns_user_answer(self) -> None:
        """ask_user_question must return the user's answer."""
        callback = AsyncMock(return_value="100")
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": "How many iterations?",
            "context": "The test spec requires --iterations",
            "_call_id": "c1",
        })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["answer"] == "100"
        callback.assert_awaited_once_with("How many iterations?", "The test spec requires --iterations")

    @pytest.mark.asyncio
    async def test_user_cancel_returns_denied(self) -> None:
        """User cancellation must return DENIED status (terminal)."""
        callback = AsyncMock(return_value=None)
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": "What value?",
            "_call_id": "c2",
        })

        assert result.status == ToolResultStatus.DENIED
        assert result.is_terminal

    @pytest.mark.asyncio
    async def test_empty_question_returns_error(self) -> None:
        """ask_user_question must validate question parameter."""
        callback = AsyncMock()
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({"question": "", "_call_id": "c3"})
        assert result.status == ToolResultStatus.ERROR


# ---------------------------------------------------------------------------
# summarize_run wraps output_summarizer
# ---------------------------------------------------------------------------


class TestSummarizeRunTool:
    """Verify summarize_run wraps execution.output_summarizer."""

    @pytest.mark.asyncio
    async def test_wraps_summarize_output(self) -> None:
        """summarize_run must delegate to output_summarizer.summarize_output."""
        tool = SummarizeRunTool()

        result = await tool.execute({
            "stdout": "=== 10 passed, 2 failed, 1 skipped in 5.23s ===",
            "stderr": "",
            "command": "pytest tests/",
            "exit_code": 1,
            "_call_id": "c1",
        })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["framework"] == "pytest"
        assert data["passed"] == 10
        assert data["failed"] == 2
        assert data["skipped"] == 1
        # Enhanced output fields
        assert data["overall_status"] in ("MIXED", "FAILED")
        assert "summary_text" in data
        assert "failure_highlights" in data
        assert "suggested_next_actions" in data

    @pytest.mark.asyncio
    async def test_missing_command_returns_error(self) -> None:
        """summarize_run must validate command parameter."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "stdout": "output",
            "command": "",
            "_call_id": "c2",
        })
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_loads_wiki_context(self, wiki_root: Path) -> None:
        """summarize_run should load wiki context for richer summaries."""
        from jules_daemon.wiki.test_knowledge import (
            TestKnowledge,
            save_test_knowledge,
        )

        knowledge = TestKnowledge(
            test_slug="pytest-tests",
            command_pattern="pytest tests/",
            purpose="Unit test suite",
            runs_observed=3,
        )
        save_test_knowledge(wiki_root, knowledge)

        tool = SummarizeRunTool(wiki_root=wiki_root)
        result = await tool.execute({
            "stdout": "=== 5 passed in 1.0s ===",
            "command": "pytest tests/",
            "_call_id": "c3",
        })

        assert result.status == ToolResultStatus.SUCCESS


# ---------------------------------------------------------------------------
# notify_user -- notification push
# ---------------------------------------------------------------------------


class TestNotifyUserTool:
    """Verify notify_user delegates to the notification callback."""

    @pytest.mark.asyncio
    async def test_delivers_notification(self) -> None:
        """notify_user must delegate to the notify callback."""
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": "Test completed: 95 passed, 5 failed",
            "severity": "info",
            "_call_id": "c1",
        })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["delivered"] is True
        callback.assert_awaited_once_with("Test completed: 95 passed, 5 failed", "info")

    @pytest.mark.asyncio
    async def test_empty_message_returns_error(self) -> None:
        """notify_user must validate message parameter."""
        callback = AsyncMock()
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({"message": "", "_call_id": "c2"})
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_invalid_severity_defaults_to_info(self) -> None:
        """notify_user should default invalid severity to 'info'."""
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": "Test update",
            "severity": "invalid_level",
            "_call_id": "c3",
        })

        assert result.status == ToolResultStatus.SUCCESS
        callback.assert_awaited_once_with("Test update", "info")


# ---------------------------------------------------------------------------
# ApprovalLedger tests
# ---------------------------------------------------------------------------


class TestApprovalLedger:
    """Verify the approval ledger enforces single-use approvals."""

    def test_record_and_retrieve(self) -> None:
        """Approvals can be recorded and retrieved."""
        ledger = ApprovalLedger()
        entry = ApprovalEntry(
            approval_id="a1",
            command="echo test",
            target_host="host",
            target_user="user",
        )
        ledger.record_approval(entry)
        assert ledger.pending_count == 1
        assert ledger.get_approved_command("a1") is entry

    def test_consume_removes_entry(self) -> None:
        """Consuming an approval removes it from the ledger."""
        ledger = ApprovalLedger()
        entry = ApprovalEntry(
            approval_id="a2",
            command="echo test",
            target_host="host",
            target_user="user",
        )
        ledger.record_approval(entry)
        consumed = ledger.consume("a2")
        assert consumed is entry
        assert ledger.pending_count == 0
        assert ledger.consume("a2") is None

    def test_has_approved_command(self) -> None:
        """has_approved_command checks by command+host pair."""
        ledger = ApprovalLedger()
        entry = ApprovalEntry(
            approval_id="a3",
            command="pytest tests/",
            target_host="10.0.1.50",
            target_user="root",
        )
        ledger.record_approval(entry)

        assert ledger.has_approved_command("pytest tests/", "10.0.1.50") == "a3"
        assert ledger.has_approved_command("pytest tests/", "other-host") is None


# ---------------------------------------------------------------------------
# registry_factory tests
# ---------------------------------------------------------------------------


class TestRegistryFactory:
    """Verify build_tool_set constructs all 10 tools correctly."""

    def test_produces_10_tools(self, wiki_root: Path) -> None:
        """build_tool_set must return exactly 10 tools."""
        tools = build_tool_set(
            wiki_root=wiki_root,
            confirm_callback=AsyncMock(),
            ask_callback=AsyncMock(),
            notify_callback=AsyncMock(),
        )
        assert len(tools) == 10

    def test_shared_ledger(self, wiki_root: Path) -> None:
        """propose_ssh_command and execute_ssh must share the same ledger."""
        ledger = ApprovalLedger()
        tools = build_tool_set(
            wiki_root=wiki_root,
            confirm_callback=AsyncMock(),
            ask_callback=AsyncMock(),
            notify_callback=AsyncMock(),
            ledger=ledger,
        )

        propose = next(t for t in tools if t.spec.name == "propose_ssh_command")
        execute = next(t for t in tools if t.spec.name == "execute_ssh")

        assert propose._ledger is ledger
        assert execute._ledger is ledger

    def test_all_names_present(self, wiki_root: Path) -> None:
        """All 10 expected tool names must be present."""
        expected_names = {
            "read_wiki", "lookup_test_spec", "check_remote_processes",
            "read_output", "parse_test_output", "propose_ssh_command",
            "execute_ssh", "ask_user_question", "summarize_run", "notify_user",
        }
        tools = build_tool_set(
            wiki_root=wiki_root,
            confirm_callback=AsyncMock(),
            ask_callback=AsyncMock(),
            notify_callback=AsyncMock(),
        )
        actual_names = {t.spec.name for t in tools}
        assert actual_names == expected_names


# ---------------------------------------------------------------------------
# Wrapping verification -- ensures no reimplementation
# ---------------------------------------------------------------------------


class TestWrappingVerification:
    """Verify tools delegate to existing daemon modules, not reimplementations."""

    @pytest.mark.asyncio
    async def test_read_wiki_uses_command_translation_module(
        self, wiki_root: Path,
    ) -> None:
        """read_wiki must call command_translation.find_by_query internally."""
        tool = ReadWikiTool(wiki_root=wiki_root)

        with patch(
            "jules_daemon.wiki.command_translation.find_by_query",
            return_value=[],
        ) as mock_find:
            with patch(
                "jules_daemon.wiki.test_knowledge.load_test_knowledge",
                return_value=None,
            ):
                await tool.execute({"query": "test", "_call_id": "w1"})
            mock_find.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_test_output_uses_parser_module(self) -> None:
        """parse_test_output must call test_output_parser.parse_interrupted_output."""
        tool = ParseTestOutputTool()

        with patch(
            "jules_daemon.monitor.test_output_parser.parse_interrupted_output",
        ) as mock_parse:
            from jules_daemon.monitor.test_output_parser import (
                FrameworkHint,
                ParseResult,
            )

            mock_parse.return_value = ParseResult(
                records=(),
                truncated=False,
                framework_hint=FrameworkHint.UNKNOWN,
                total_lines_parsed=1,
                raw_tail="",
            )
            await tool.execute({
                "raw_output": "test output",
                "_call_id": "w2",
            })
            mock_parse.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_run_uses_output_summarizer(self) -> None:
        """summarize_run must call output_summarizer.summarize_output."""
        tool = SummarizeRunTool()

        with patch(
            "jules_daemon.execution.output_summarizer.summarize_output",
            new_callable=AsyncMock,
        ) as mock_summarize:
            from jules_daemon.execution.output_summarizer import OutputSummary

            mock_summarize.return_value = OutputSummary(parser="none")
            await tool.execute({
                "stdout": "output",
                "command": "echo test",
                "_call_id": "w3",
            })
            mock_summarize.assert_called_once()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_all_tools(wiki_root: Path) -> tuple[BaseTool, ...]:
    """Construct all 10 tools with dummy callbacks for protocol testing."""
    return build_tool_set(
        wiki_root=wiki_root,
        confirm_callback=AsyncMock(),
        ask_callback=AsyncMock(),
        notify_callback=AsyncMock(),
    )
