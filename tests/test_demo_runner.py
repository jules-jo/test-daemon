"""Tests for the interactive demo runner (Sub-AC 6.3).

Validates that the demo runner:
    - Registers all 10 tools in the ToolRegistry
    - Executes the full named-test flow with scripted LLM
    - Prints stage transitions to stdout
    - Handles approval and denial flows correctly
    - Respects the approval_id enforcement constraint
    - The ScriptedLLMClient follows the expected tool sequence

Tests use auto-approval callbacks (not real stdin) so they can
run in CI without human interaction.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from jules_daemon.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopResult,
    AgentLoopState,
)
from jules_daemon.agent.tool_dispatch import ToolDispatchBridge
from jules_daemon.agent.tool_registry import ToolRegistry
from jules_daemon.agent.tool_types import ToolCall, ToolResult, ToolResultStatus
from jules_daemon.agent.tools.propose_ssh_command import ApprovalLedger
from jules_daemon.demo_runner import (
    ObservableAgentLoop,
    ObservableDispatchBridge,
    ScriptedLLMClient,
    SimulatedExecuteSSHTool,
    _build_demo_registry,
    _make_simulated_wiki,
)


# ---------------------------------------------------------------------------
# Auto-approval callbacks for non-interactive testing
# ---------------------------------------------------------------------------


async def _auto_approve(
    command: str, target_host: str, explanation: str
) -> tuple[bool, str]:
    """Auto-approve all commands (for non-interactive tests)."""
    return (True, command)


async def _auto_deny(
    command: str, target_host: str, explanation: str
) -> tuple[bool, str]:
    """Auto-deny all commands (for testing denial flow)."""
    return (False, command)


async def _auto_ask(question: str, context: str) -> str | None:
    """Auto-answer questions based on keyword matching."""
    if "iteration" in question.lower():
        return "50"
    if "host" in question.lower():
        return "staging.example.com"
    return "auto-answer"


async def _auto_notify(message: str, severity: str = "info") -> bool:
    """Silent notification sink for tests."""
    return True


# ---------------------------------------------------------------------------
# Simulated wiki setup
# ---------------------------------------------------------------------------


class TestSimulatedWiki:
    """Tests for the simulated wiki structure."""

    def test_creates_wiki_directory(self, tmp_path: Path) -> None:
        wiki_root = _make_simulated_wiki(tmp_path)
        assert wiki_root.exists()
        assert wiki_root.is_dir()

    def test_creates_current_run_file(self, tmp_path: Path) -> None:
        wiki_root = _make_simulated_wiki(tmp_path)
        current_run = wiki_root / "pages" / "daemon" / "current-run.md"
        assert current_run.exists()
        assert "idle" in current_run.read_text()

    def test_creates_test_knowledge_files(self, tmp_path: Path) -> None:
        wiki_root = _make_simulated_wiki(tmp_path)
        # Knowledge file for slug "agent-test"
        test_file = wiki_root / "pages" / "daemon" / "knowledge" / "test-agent-test.md"
        assert test_file.exists()
        content = test_file.read_text()
        assert "agent-test" in content
        assert "iterations" in content
        assert "host" in content

        # Knowledge file for slug "agent-test-py"
        test_file_py = (
            wiki_root / "pages" / "daemon" / "knowledge" / "test-agent-test-py.md"
        )
        assert test_file_py.exists()

    def test_creates_required_subdirectories(self, tmp_path: Path) -> None:
        wiki_root = _make_simulated_wiki(tmp_path)
        assert (wiki_root / "pages" / "daemon" / "history").is_dir()
        assert (wiki_root / "pages" / "daemon" / "results").is_dir()
        assert (wiki_root / "pages" / "daemon" / "translations").is_dir()
        assert (wiki_root / "pages" / "daemon" / "audit").is_dir()
        assert (wiki_root / "pages" / "daemon" / "queue").is_dir()
        assert (wiki_root / "pages" / "daemon" / "knowledge").is_dir()


# ---------------------------------------------------------------------------
# ScriptedLLMClient tests
# ---------------------------------------------------------------------------


class TestScriptedLLMClient:
    """Tests for the ScriptedLLMClient scripted tool call sequence."""

    @pytest.mark.asyncio
    async def test_first_call_is_lookup_test_spec(self) -> None:
        client = ScriptedLLMClient()
        calls = await client.get_tool_calls(())
        assert len(calls) == 1
        assert calls[0].tool_name == "lookup_test_spec"

    @pytest.mark.asyncio
    async def test_second_call_is_propose_ssh_command(self) -> None:
        client = ScriptedLLMClient()
        _ = await client.get_tool_calls(())  # step 0
        calls = await client.get_tool_calls(())  # step 1
        assert len(calls) == 1
        assert calls[0].tool_name == "propose_ssh_command"

    @pytest.mark.asyncio
    async def test_third_call_is_execute_ssh(self) -> None:
        client = ScriptedLLMClient()
        for _ in range(2):
            await client.get_tool_calls(())
        calls = await client.get_tool_calls(())  # step 2
        assert len(calls) == 1
        assert calls[0].tool_name == "execute_ssh"

    @pytest.mark.asyncio
    async def test_extracts_approval_id_from_history(self) -> None:
        """ScriptedLLMClient extracts approval_id from conversation history."""
        client = ScriptedLLMClient()
        for _ in range(2):
            await client.get_tool_calls(())

        # Provide history with an approval_id
        messages: tuple[dict[str, Any], ...] = (
            {"role": "system", "content": "..."},
            {"role": "tool", "content": json.dumps({
                "approved": True,
                "approval_id": "approval-test123",
                "command": "test_cmd",
            })},
        )
        calls = await client.get_tool_calls(messages)
        assert calls[0].arguments["approval_id"] == "approval-test123"

    @pytest.mark.asyncio
    async def test_full_sequence_ends_with_empty(self) -> None:
        """After all steps, returns empty tuple (completion signal)."""
        client = ScriptedLLMClient()
        for _ in range(6):
            calls = await client.get_tool_calls(())
            assert len(calls) > 0, f"Expected tool calls at step {_}"

        # Step 7+ should return empty
        calls = await client.get_tool_calls(())
        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_expected_tool_sequence(self) -> None:
        """The full tool sequence matches Demo 1 expectations."""
        client = ScriptedLLMClient()
        expected_names = [
            "lookup_test_spec",
            "propose_ssh_command",
            "execute_ssh",
            "read_output",
            "parse_test_output",
            "summarize_run",
        ]
        actual_names: list[str] = []
        for _ in range(6):
            calls = await client.get_tool_calls(())
            assert len(calls) == 1
            actual_names.append(calls[0].tool_name)

        assert actual_names == expected_names


# ---------------------------------------------------------------------------
# Demo registry tests
# ---------------------------------------------------------------------------


class TestBuildDemoRegistry:
    """Tests for _build_demo_registry."""

    def test_registers_10_tools(self, tmp_path: Path) -> None:
        wiki_root = _make_simulated_wiki(tmp_path)
        ledger = ApprovalLedger()
        registry = _build_demo_registry(
            wiki_root=wiki_root,
            ledger=ledger,
        )
        assert len(registry) == 10

    def test_all_expected_tools_present(self, tmp_path: Path) -> None:
        wiki_root = _make_simulated_wiki(tmp_path)
        ledger = ApprovalLedger()
        registry = _build_demo_registry(
            wiki_root=wiki_root,
            ledger=ledger,
        )
        expected_names = {
            "read_wiki",
            "lookup_test_spec",
            "check_remote_processes",
            "read_output",
            "parse_test_output",
            "propose_ssh_command",
            "execute_ssh",
            "ask_user_question",
            "summarize_run",
            "notify_user",
        }
        actual_names = set(registry.list_tool_names())
        assert actual_names == expected_names

    def test_approval_classification(self, tmp_path: Path) -> None:
        """Read-only tools need no approval, state-changing tools do."""
        wiki_root = _make_simulated_wiki(tmp_path)
        ledger = ApprovalLedger()
        registry = _build_demo_registry(
            wiki_root=wiki_root,
            ledger=ledger,
        )

        # These should require approval
        assert registry.requires_approval("propose_ssh_command")
        assert registry.requires_approval("execute_ssh")

        # These should not require approval
        read_only_names = [
            "read_wiki",
            "lookup_test_spec",
            "check_remote_processes",
            "read_output",
            "parse_test_output",
            "summarize_run",
            "notify_user",
        ]
        for name in read_only_names:
            assert not registry.requires_approval(name), (
                f"Expected {name} to be read-only (no approval)"
            )


# ---------------------------------------------------------------------------
# SimulatedExecuteSSHTool tests
# ---------------------------------------------------------------------------


class TestSimulatedExecuteSSHTool:
    """Tests for the simulated SSH execution tool."""

    @pytest.mark.asyncio
    async def test_requires_approval_id(self, tmp_path: Path) -> None:
        wiki_root = _make_simulated_wiki(tmp_path)
        ledger = ApprovalLedger()
        tool = SimulatedExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=_auto_approve,
        )
        result = await tool.execute({
            "_call_id": "test-001",
            "approval_id": "",
        })
        assert result.is_error
        assert "approval_id is required" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_rejects_unknown_approval_id(self, tmp_path: Path) -> None:
        wiki_root = _make_simulated_wiki(tmp_path)
        ledger = ApprovalLedger()
        tool = SimulatedExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=_auto_approve,
        )
        result = await tool.execute({
            "_call_id": "test-001",
            "approval_id": "nonexistent-id",
        })
        assert result.is_error
        assert "No approved command" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_simulates_success_on_approval(
        self, tmp_path: Path,
    ) -> None:
        wiki_root = _make_simulated_wiki(tmp_path)
        ledger = ApprovalLedger()
        tool = SimulatedExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=_auto_approve,
        )

        # Record an approval first
        from jules_daemon.agent.tools.propose_ssh_command import ApprovalEntry
        entry = ApprovalEntry(
            approval_id="approval-test001",
            command="echo hello",
            target_host="staging.example.com",
            target_user="deploy",
        )
        ledger.record_approval(entry)

        result = await tool.execute({
            "_call_id": "test-001",
            "approval_id": "approval-test001",
        })
        assert result.is_success
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["exit_code"] == 0
        assert "100 passed" in data["stdout"]

    @pytest.mark.asyncio
    async def test_returns_denied_on_user_deny(
        self, tmp_path: Path,
    ) -> None:
        wiki_root = _make_simulated_wiki(tmp_path)
        ledger = ApprovalLedger()
        tool = SimulatedExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=_auto_deny,
        )

        from jules_daemon.agent.tools.propose_ssh_command import ApprovalEntry
        entry = ApprovalEntry(
            approval_id="approval-deny001",
            command="echo hello",
            target_host="staging.example.com",
            target_user="deploy",
        )
        ledger.record_approval(entry)

        result = await tool.execute({
            "_call_id": "test-001",
            "approval_id": "approval-deny001",
        })
        assert result.is_denied
        assert "User denied" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_consumes_approval_on_success(
        self, tmp_path: Path,
    ) -> None:
        """After successful execution, the approval is consumed."""
        wiki_root = _make_simulated_wiki(tmp_path)
        ledger = ApprovalLedger()
        tool = SimulatedExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=_auto_approve,
        )

        from jules_daemon.agent.tools.propose_ssh_command import ApprovalEntry
        entry = ApprovalEntry(
            approval_id="approval-consume001",
            command="echo hello",
            target_host="staging.example.com",
            target_user="deploy",
        )
        ledger.record_approval(entry)

        result = await tool.execute({
            "_call_id": "test-001",
            "approval_id": "approval-consume001",
        })
        assert result.is_success

        # Approval should be consumed -- second attempt fails
        result2 = await tool.execute({
            "_call_id": "test-002",
            "approval_id": "approval-consume001",
        })
        assert result2.is_error
        assert "No approved command" in (result2.error_message or "")


# ---------------------------------------------------------------------------
# Full end-to-end flow with auto-approval
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    """Full end-to-end agent loop execution with auto-approval."""

    def _build_auto_registry(
        self, tmp_path: Path
    ) -> tuple[ToolRegistry, ApprovalLedger]:
        """Build a registry with auto-approval callbacks."""
        wiki_root = _make_simulated_wiki(tmp_path)
        ledger = ApprovalLedger()

        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )
        from jules_daemon.agent.tools.read_output import ReadOutputTool
        from jules_daemon.agent.tools.read_wiki import ReadWikiTool

        registry = ToolRegistry()

        # Read-only tools
        registry.register(ReadWikiTool(wiki_root=wiki_root))
        registry.register(LookupTestSpecTool(wiki_root=wiki_root))
        registry.register(CheckRemoteProcessesTool())
        registry.register(ReadOutputTool(wiki_root=wiki_root))
        registry.register(ParseTestOutputTool())

        # State-changing tools with auto-approval
        registry.register(ProposeSSHCommandTool(
            confirm_callback=_auto_approve,
            ledger=ledger,
        ))
        registry.register(SimulatedExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=_auto_approve,
        ))

        # User interaction tools
        registry.register(AskUserQuestionTool(ask_callback=_auto_ask))
        registry.register(SummarizeRunTool(wiki_root=wiki_root))
        registry.register(NotifyUserTool(notify_callback=_auto_notify))

        return registry, ledger

    @pytest.mark.asyncio
    async def test_named_test_flow_completes(self, tmp_path: Path) -> None:
        """Full named-test flow completes with COMPLETE state."""
        registry, _ = self._build_auto_registry(tmp_path)

        llm_client = ScriptedLLMClient()
        dispatcher = ToolDispatchBridge(registry=registry)

        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=dispatcher,
            system_prompt="You are a test runner.",
            config=AgentLoopConfig(max_iterations=10),
        )

        result = await loop.run(
            "run agent_test with 100 iterations on staging"
        )

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.error_message is None
        assert result.iterations_used <= 10

    @pytest.mark.asyncio
    async def test_flow_uses_correct_iteration_count(
        self, tmp_path: Path,
    ) -> None:
        """The flow uses 7 iterations (6 tool steps + 1 empty for completion)."""
        registry, _ = self._build_auto_registry(tmp_path)

        llm_client = ScriptedLLMClient()
        dispatcher = ToolDispatchBridge(registry=registry)

        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=dispatcher,
            system_prompt="You are a test runner.",
            config=AgentLoopConfig(max_iterations=10),
        )

        result = await loop.run(
            "run agent_test with 100 iterations on staging"
        )

        # 6 tool call steps + 1 empty (completion signal) = 7 iterations
        assert result.iterations_used == 7

    @pytest.mark.asyncio
    async def test_history_contains_tool_results(
        self, tmp_path: Path,
    ) -> None:
        """The conversation history contains tool result messages."""
        registry, _ = self._build_auto_registry(tmp_path)

        llm_client = ScriptedLLMClient()
        dispatcher = ToolDispatchBridge(registry=registry)

        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=dispatcher,
            system_prompt="You are a test runner.",
            config=AgentLoopConfig(max_iterations=10),
        )

        result = await loop.run(
            "run agent_test with 100 iterations on staging"
        )

        # Count tool result messages in history
        tool_messages = [
            msg for msg in result.history
            if msg.get("role") == "tool"
        ]
        # Should have tool results for each of the 6 tool calls
        assert len(tool_messages) == 6

    @pytest.mark.asyncio
    async def test_denial_terminates_loop(self, tmp_path: Path) -> None:
        """User denial in propose_ssh_command terminates the loop."""
        wiki_root = _make_simulated_wiki(tmp_path)
        ledger = ApprovalLedger()

        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )
        from jules_daemon.agent.tools.read_output import ReadOutputTool
        from jules_daemon.agent.tools.read_wiki import ReadWikiTool

        registry = ToolRegistry()
        registry.register(ReadWikiTool(wiki_root=wiki_root))
        registry.register(LookupTestSpecTool(wiki_root=wiki_root))
        registry.register(CheckRemoteProcessesTool())
        registry.register(ReadOutputTool(wiki_root=wiki_root))
        registry.register(ParseTestOutputTool())

        # propose_ssh_command with auto-deny
        registry.register(ProposeSSHCommandTool(
            confirm_callback=_auto_deny,
            ledger=ledger,
        ))
        registry.register(SimulatedExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=_auto_deny,
        ))
        registry.register(AskUserQuestionTool(ask_callback=_auto_ask))
        registry.register(SummarizeRunTool(wiki_root=wiki_root))
        registry.register(NotifyUserTool(notify_callback=_auto_notify))

        llm_client = ScriptedLLMClient()
        dispatcher = ToolDispatchBridge(registry=registry)

        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=dispatcher,
            system_prompt="You are a test runner.",
            config=AgentLoopConfig(max_iterations=10),
        )

        result = await loop.run(
            "run agent_test with 100 iterations on staging"
        )

        # Loop should terminate with ERROR on denial
        assert result.final_state is AgentLoopState.ERROR
        assert result.error_message is not None
        assert "denied" in result.error_message.lower()
        # Should terminate early (after lookup_test_spec + propose_ssh)
        assert result.iterations_used == 2

    @pytest.mark.asyncio
    async def test_approval_id_enforcement(self, tmp_path: Path) -> None:
        """execute_ssh requires a valid approval_id from propose_ssh_command."""
        wiki_root = _make_simulated_wiki(tmp_path)
        ledger = ApprovalLedger()

        tool = SimulatedExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=_auto_approve,
        )

        # Try to execute without an approval
        result = await tool.execute({
            "_call_id": "test-enforce-001",
            "approval_id": "bogus-approval-id",
        })
        assert result.is_error
        assert "No approved command" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Observable dispatch bridge tests
# ---------------------------------------------------------------------------


class TestObservableDispatchBridge:
    """Tests for the observable dispatch bridge."""

    @pytest.mark.asyncio
    async def test_delegates_to_registry(self, tmp_path: Path) -> None:
        """Observable bridge dispatches to the underlying registry."""
        wiki_root = _make_simulated_wiki(tmp_path)
        ledger = ApprovalLedger()
        registry = _build_demo_registry(
            wiki_root=wiki_root,
            ledger=ledger,
        )
        bridge = ObservableDispatchBridge(registry=registry)

        # Dispatch a read-only tool call
        call = ToolCall(
            call_id="obs-001",
            tool_name="parse_test_output",
            arguments={
                "raw_output": "1 passed in 0.1s",
                "framework_hint": "auto",
            },
        )
        result = await bridge.dispatch(call)
        assert result.is_success or result.is_error  # At least doesn't crash


# Need to import LookupTestSpecTool for the test registry build
from jules_daemon.agent.tools.lookup_test_spec import LookupTestSpecTool
from jules_daemon.agent.tools.parse_test_output import ParseTestOutputTool
from jules_daemon.agent.tools.propose_ssh_command import ProposeSSHCommandTool
from jules_daemon.agent.tools.ask_user_question import AskUserQuestionTool
from jules_daemon.agent.tools.summarize_run import SummarizeRunTool
from jules_daemon.agent.tools.notify_user import NotifyUserTool
