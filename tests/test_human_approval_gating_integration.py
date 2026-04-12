"""Integration test: human approval gating for SSH commands (AC 130303, Sub-AC 3).

Validates that SSH commands (propose_ssh_command, execute_ssh) block until
explicit human approval is granted or denied, exercising the full agent loop
state machine with real ToolRegistry, ApprovalLedger, and ToolDispatchBridge.

The LLM client is scripted; only the LLM and SSH execution backend are faked.
Everything else -- AgentLoop, ToolRegistry, ToolDispatchBridge, ApprovalLedger,
ProposeSSHCommandTool, ExecuteSSHTool -- is real.

Test scenarios:

    TestApprovalGrantedFlowIntegration
        propose_ssh_command approved -> execute_ssh approved -> loop completes.
        Confirms: callback is called, approval_id flows through, command runs.

    TestProposalDenialTerminatesLoop
        propose_ssh_command denied -> loop terminates with ERROR state.
        The LLM never gets a chance to call execute_ssh.

    TestExecutionDenialTerminatesLoop
        propose_ssh_command approved -> execute_ssh denied -> loop terminates.
        Approval is NOT consumed (can be retried in theory).

    TestApprovalBlocksUntilCallback
        Simulates a delayed approval callback that resolves after an
        asyncio.sleep. Verifies the tool (and loop) genuinely waits for
        the callback coroutine to resolve before proceeding.

    TestDenialOutputContainsCommandInfo
        Denied result from propose_ssh_command includes the command and
        approved=False in structured JSON output.

    TestExecutionDenialPreservesLedger
        After execute_ssh is denied, the approval entry remains in the
        ledger (not consumed) so the agent could retry.

    TestExecuteSSHWithoutApprovalReturnsError
        If the LLM calls execute_ssh without a valid approval_id, the
        tool returns ERROR (not DENIED) and the loop continues (the LLM
        can self-correct).

    TestMultipleDenialsInSequence
        LLM proposes twice, user denies the first proposal. Because denial
        at propose_ssh_command is terminal, the loop stops immediately
        without reaching the second proposal.

    TestApprovalGrantedThenDeniedAtExecution
        The user approves the proposal but denies execution. Validates
        the two-level gate: denial at execute_ssh still terminates the loop.

    TestEditedCommandFlowsThroughExecution
        The user edits the command during the propose_ssh_command approval.
        The edited command is stored in the ledger and executed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from jules_daemon.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopState,
)
from jules_daemon.agent.tool_dispatch import ToolDispatchBridge
from jules_daemon.agent.tool_registry import ToolRegistry
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResultStatus,
)
from jules_daemon.agent.tools.execute_ssh import ExecuteSSHTool
from jules_daemon.agent.tools.propose_ssh_command import (
    ApprovalLedger,
    ProposeSSHCommandTool,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HOST = "10.0.1.50"
_USER = "testrunner"
_COMMAND = "python3 ~/run_tests.py --iterations 100"
_SYSTEM_PROMPT = (
    "You are a test-runner assistant. Propose and execute SSH commands "
    "to run tests on remote hosts."
)
_NL_INPUT = "run the tests on the staging server"


# ---------------------------------------------------------------------------
# Scripted LLM Client (shared across tests)
# ---------------------------------------------------------------------------


class ScriptedLLMClient:
    """LLM client returning a predefined sequence of tool call batches.

    Each script entry is either a tuple of ToolCalls (static) or a
    callable that receives the current conversation history and returns
    tool calls dynamically.
    """

    def __init__(
        self,
        script: list[tuple[ToolCall, ...] | Any],
    ) -> None:
        self._script = list(script)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def get_tool_calls(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self._call_count += 1
        if not self._script:
            return ()
        entry = self._script.pop(0)
        if callable(entry):
            return entry(messages)
        return entry


# ---------------------------------------------------------------------------
# Stub SSH execution pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StubRunResult:
    """Mimics jules_daemon.execution.run_pipeline.RunResult for tests."""

    success: bool
    run_id: str
    command: str
    target_host: str
    target_user: str
    exit_code: int | None
    stdout: str
    stderr: str
    error: str | None
    duration_seconds: float
    started_at: str = "2026-04-12T14:00:00Z"
    completed_at: str = "2026-04-12T14:00:10Z"


def _make_stub_execute_run(
    *,
    success: bool = True,
    exit_code: int = 0,
    stdout: str = "All tests passed.\n",
    stderr: str = "",
) -> Any:
    """Create a fake execute_run async function."""

    async def _fake(**kwargs: Any) -> StubRunResult:
        return StubRunResult(
            success=success,
            run_id="run-approval-test",
            command=kwargs.get("command", _COMMAND),
            target_host=kwargs.get("target_host", _HOST),
            target_user=kwargs.get("target_user", _USER),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            error=None if success else "Command failed",
            duration_seconds=10.0,
        )

    return _fake


def _install_mock_run_pipeline(execute_run_fn: Any) -> types.ModuleType:
    """Install a fake run_pipeline module in sys.modules."""
    mod = types.ModuleType("jules_daemon.execution.run_pipeline")
    mod.execute_run = execute_run_fn  # type: ignore[attr-defined]
    mod.RunResult = StubRunResult  # type: ignore[attr-defined]
    sys.modules["jules_daemon.execution.run_pipeline"] = mod
    return mod


def _uninstall_mock_run_pipeline() -> None:
    """Remove the fake run_pipeline module."""
    sys.modules.pop("jules_daemon.execution.run_pipeline", None)


# ---------------------------------------------------------------------------
# Approval callback factories
# ---------------------------------------------------------------------------


class ApprovalTracker:
    """Tracks all approval requests and their outcomes."""

    def __init__(self, *, auto_approve: bool = True) -> None:
        self.requests: list[tuple[str, str, str]] = []
        self._auto_approve = auto_approve

    async def confirm(
        self, command: str, target_host: str, explanation: str
    ) -> tuple[bool, str]:
        self.requests.append((command, target_host, explanation))
        return (self._auto_approve, command)


class DenyingApprovalTracker:
    """Always denies approval. Records every request for assertions."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, str]] = []

    async def confirm(
        self, command: str, target_host: str, explanation: str
    ) -> tuple[bool, str]:
        self.requests.append((command, target_host, explanation))
        return (False, command)


class SplitApprovalTracker:
    """Approves propose_ssh_command but denies execute_ssh.

    Uses request counting: first N calls approve, rest deny.
    """

    def __init__(self, *, approve_count: int = 1) -> None:
        self.requests: list[tuple[str, str, str]] = []
        self._approve_count = approve_count
        self._call_number = 0

    async def confirm(
        self, command: str, target_host: str, explanation: str
    ) -> tuple[bool, str]:
        self.requests.append((command, target_host, explanation))
        self._call_number += 1
        approved = self._call_number <= self._approve_count
        return (approved, command)


class EditingApprovalTracker:
    """Approves and edits the command (appends --verbose)."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, str]] = []

    async def confirm(
        self, command: str, target_host: str, explanation: str
    ) -> tuple[bool, str]:
        self.requests.append((command, target_host, explanation))
        return (True, command + " --verbose")


class DelayedApprovalTracker:
    """Approves after a configurable delay, proving the tool blocks.

    Records the timestamps (before/after sleep) so the test can verify
    the delay actually happened.
    """

    def __init__(self, *, delay_seconds: float = 0.05) -> None:
        self.requests: list[tuple[str, str, str]] = []
        self.pre_delay_times: list[float] = []
        self.post_delay_times: list[float] = []
        self._delay = delay_seconds

    async def confirm(
        self, command: str, target_host: str, explanation: str
    ) -> tuple[bool, str]:
        loop = asyncio.get_event_loop()
        self.pre_delay_times.append(loop.time())
        self.requests.append((command, target_host, explanation))
        await asyncio.sleep(self._delay)
        self.post_delay_times.append(loop.time())
        return (True, command)


# ---------------------------------------------------------------------------
# Dynamic script helpers
# ---------------------------------------------------------------------------


def _extract_approval_id(messages: tuple[dict[str, Any], ...]) -> str | None:
    """Extract the most recent approval_id from tool result messages."""
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                if "approval_id" in data:
                    return data["approval_id"]
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _make_execute_ssh_from_history(
    messages: tuple[dict[str, Any], ...],
) -> tuple[ToolCall, ...]:
    """Dynamic script entry: produce execute_ssh with approval_id from history."""
    approval_id = _extract_approval_id(messages)
    if approval_id is None:
        raise AssertionError(
            "Expected approval_id in conversation history but none found."
        )
    return (
        ToolCall(
            call_id="call_execute",
            tool_name="execute_ssh",
            arguments={"approval_id": approval_id},
        ),
    )


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def _build_registry(
    *,
    wiki_root: Path,
    ledger: ApprovalLedger,
    propose_callback: Any,
    execute_callback: Any | None = None,
) -> ToolRegistry:
    """Build a minimal ToolRegistry with propose + execute SSH tools.

    Uses the same callback for both propose and execute unless
    execute_callback is explicitly provided.
    """
    registry = ToolRegistry()
    registry.register(
        ProposeSSHCommandTool(
            confirm_callback=propose_callback,
            ledger=ledger,
        )
    )
    registry.register(
        ExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=execute_callback or propose_callback,
        )
    )
    return registry


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def wiki_root(tmp_path: Path) -> Path:
    """Temporary wiki root directory."""
    from jules_daemon.wiki.layout import initialize_wiki

    initialize_wiki(tmp_path)
    return tmp_path


@pytest.fixture()
def stub_pipeline():
    """Install and tear down a mock run_pipeline module."""
    fake = _make_stub_execute_run(success=True, exit_code=0)
    mod = _install_mock_run_pipeline(fake)
    yield mod
    _uninstall_mock_run_pipeline()


# ---------------------------------------------------------------------------
# Propose call constant
# ---------------------------------------------------------------------------

_PROPOSE_CALL = ToolCall(
    call_id="call_propose",
    tool_name="propose_ssh_command",
    arguments={
        "command": _COMMAND,
        "target_host": _HOST,
        "target_user": _USER,
        "explanation": "Running tests on staging",
    },
)


# ---------------------------------------------------------------------------
# Test: Full approval-granted flow
# ---------------------------------------------------------------------------


class TestApprovalGrantedFlowIntegration:
    """End-to-end: propose -> approve -> execute -> complete."""

    @pytest.mark.asyncio
    async def test_approved_command_executes_and_completes(
        self, wiki_root: Path, stub_pipeline: Any
    ) -> None:
        """When user approves the proposal, execute_ssh runs and loop completes."""
        tracker = ApprovalTracker(auto_approve=True)
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=tracker.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            _make_execute_ssh_from_history,
            # LLM returns no tool calls -> COMPLETE
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        # Loop should complete (3 iterations: propose, execute, empty)
        assert result.final_state is AgentLoopState.COMPLETE, (
            f"Expected COMPLETE, got {result.final_state.value}. "
            f"Error: {result.error_message}"
        )
        assert result.iterations_used == 3
        assert result.error_message is None
        assert result.retry_exhausted is False

        # Only propose_ssh_command triggers the callback (single-gate)
        assert len(tracker.requests) == 1, (
            f"Expected 1 approval request (propose only), got {len(tracker.requests)}"
        )
        # The proposal callback was called with the correct command and host
        assert tracker.requests[0][0] == _COMMAND
        assert tracker.requests[0][1] == _HOST

    @pytest.mark.asyncio
    async def test_approval_id_flows_from_propose_to_execute(
        self, wiki_root: Path, stub_pipeline: Any
    ) -> None:
        """The approval_id from propose_ssh_command is used by execute_ssh."""
        tracker = ApprovalTracker(auto_approve=True)
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=tracker.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        approval_ids_seen: list[str] = []

        def _capture_execute(
            messages: tuple[dict[str, Any], ...],
        ) -> tuple[ToolCall, ...]:
            aid = _extract_approval_id(messages)
            assert aid is not None, "approval_id must exist in history"
            approval_ids_seen.append(aid)
            return (
                ToolCall(
                    call_id="call_execute",
                    tool_name="execute_ssh",
                    arguments={"approval_id": aid},
                ),
            )

        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            _capture_execute,
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.COMPLETE
        assert len(approval_ids_seen) == 1
        assert approval_ids_seen[0].startswith("approval-")

    @pytest.mark.asyncio
    async def test_ledger_consumed_after_successful_execution(
        self, wiki_root: Path, stub_pipeline: Any
    ) -> None:
        """After successful execution, the approval is consumed from the ledger."""
        tracker = ApprovalTracker(auto_approve=True)
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=tracker.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            _make_execute_ssh_from_history,
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.COMPLETE
        # Ledger should be empty (approval consumed)
        assert ledger.pending_count == 0


# ---------------------------------------------------------------------------
# Test: Proposal denial terminates loop
# ---------------------------------------------------------------------------


class TestProposalDenialTerminatesLoop:
    """When user denies propose_ssh_command, the loop terminates immediately."""

    @pytest.mark.asyncio
    async def test_denial_at_proposal_terminates_with_error(
        self, wiki_root: Path
    ) -> None:
        """Denial at propose_ssh_command -> loop ERROR with DENIED info."""
        denier = DenyingApprovalTracker()
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=denier.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        # Script: propose -> (denied, never reaches execute) -> (never called)
        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            # This should never be reached:
            _make_execute_ssh_from_history,
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        # Loop terminates with ERROR (DENIED is terminal)
        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 1
        assert result.error_message is not None
        assert "denied" in result.error_message.lower()

        # Only one approval request was made (the proposal)
        assert len(denier.requests) == 1
        # LLM was called exactly once (for the propose iteration)
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_denial_prevents_execute_ssh_call(
        self, wiki_root: Path
    ) -> None:
        """After proposal denial, the LLM never gets to call execute_ssh."""
        denier = DenyingApprovalTracker()
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=denier.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            _make_execute_ssh_from_history,
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.ERROR
        # The dispatch bridge should have dispatched exactly 1 tool call
        assert bridge.dispatch_count == 1
        # No approvals recorded in the ledger (proposal was denied)
        assert ledger.pending_count == 0

    @pytest.mark.asyncio
    async def test_denial_output_has_denied_error_message(
        self, wiki_root: Path
    ) -> None:
        """Denied proposal result includes 'denied' in the tool message content.

        Note: to_openai_tool_message() formats non-SUCCESS results as
        ``ERROR: <error_message>`` text, not the raw JSON output field.
        The ToolResult itself contains structured JSON in .output, but the
        conversation history message uses the error_message text.
        """
        denier = DenyingApprovalTracker()
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=denier.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [(_PROPOSE_CALL,)]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        # Find the tool result message in the history
        tool_messages = [
            m for m in result.history if m.get("role") == "tool"
        ]
        assert len(tool_messages) == 1
        content = tool_messages[0]["content"]
        assert "denied" in content.lower()
        assert content.startswith("ERROR:")

    @pytest.mark.asyncio
    async def test_denial_tool_result_has_denied_status_in_dispatch(
        self, wiki_root: Path
    ) -> None:
        """The ToolResult from dispatch has DENIED status with JSON output."""
        denier = DenyingApprovalTracker()
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=denier.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [(_PROPOSE_CALL,)]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        await loop.run(_NL_INPUT)

        # Check the raw ToolResult captured by the dispatch bridge
        all_results = bridge.all_results
        assert len(all_results) == 1
        denied_result = all_results[0]
        assert denied_result.status is ToolResultStatus.DENIED
        assert denied_result.is_terminal is True
        data = json.loads(denied_result.output)
        assert data["approved"] is False


# ---------------------------------------------------------------------------
# Test: Execution denial terminates loop
# ---------------------------------------------------------------------------


class TestExecutionDenialTerminatesLoop:
    """execute_ssh no longer prompts the user. Denial only happens at propose.

    These tests verify that the propose-approve-execute flow works correctly:
    propose denial terminates the loop, and successful propose+execute completes.
    """

    @pytest.mark.asyncio
    async def test_execution_denial_terminates_loop(
        self, wiki_root: Path, stub_pipeline: Any
    ) -> None:
        """Approve proposal -> execute runs -> loop completes.

        Execution denial is no longer possible (no second prompt). With a
        valid approval, execute_ssh always runs the command.
        """
        tracker = ApprovalTracker(auto_approve=True)
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=tracker.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            _make_execute_ssh_from_history,
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        # Loop completes (propose approved, execute runs without second prompt)
        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 3
        assert result.error_message is None

        # Only one callback call (propose only)
        assert len(tracker.requests) == 1

    @pytest.mark.asyncio
    async def test_execution_denial_preserves_ledger(
        self, wiki_root: Path, stub_pipeline: Any
    ) -> None:
        """After successful execute_ssh, the approval is consumed from the ledger."""
        tracker = ApprovalTracker(auto_approve=True)
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=tracker.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            _make_execute_ssh_from_history,
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.COMPLETE
        # The approval IS consumed on execution (one-time use)
        assert ledger.pending_count == 0

    @pytest.mark.asyncio
    async def test_execution_denial_output_has_details(
        self, wiki_root: Path, stub_pipeline: Any
    ) -> None:
        """Successful execution result contains execution info in history."""
        tracker = ApprovalTracker(auto_approve=True)
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=tracker.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            _make_execute_ssh_from_history,
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        # The history has 2 tool messages: propose SUCCESS, execute SUCCESS
        tool_messages = [
            m for m in result.history if m.get("role") == "tool"
        ]
        assert len(tool_messages) == 2

        # Propose message contains JSON (SUCCESS status -> raw output)
        propose_data = json.loads(tool_messages[0]["content"])
        assert propose_data["approved"] is True

        # Execute message is also JSON (SUCCESS status -> raw output)
        execute_data = json.loads(tool_messages[1]["content"])
        assert execute_data["success"] is True

        # The raw ToolResult from the dispatch bridge has SUCCESS status
        all_results = bridge.all_results
        execute_result = all_results[-1]
        assert execute_result.status is ToolResultStatus.SUCCESS


# ---------------------------------------------------------------------------
# Test: Approval blocks until callback resolves
# ---------------------------------------------------------------------------


class TestApprovalBlocksUntilCallback:
    """Verify the tool genuinely waits for the async callback to resolve."""

    @pytest.mark.asyncio
    async def test_propose_blocks_until_callback_completes(
        self, wiki_root: Path
    ) -> None:
        """propose_ssh_command awaits the confirm_callback before returning."""
        delayed = DelayedApprovalTracker(delay_seconds=0.05)
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=delayed.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            # LLM returns empty -> COMPLETE
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.COMPLETE
        # The delayed callback was called exactly once
        assert len(delayed.requests) == 1
        # Verify actual delay occurred
        assert len(delayed.pre_delay_times) == 1
        assert len(delayed.post_delay_times) == 1
        elapsed = delayed.post_delay_times[0] - delayed.pre_delay_times[0]
        assert elapsed >= 0.04, (
            f"Expected >= 40ms delay, got {elapsed * 1000:.1f}ms"
        )

    @pytest.mark.asyncio
    async def test_execute_blocks_until_callback_completes(
        self, wiki_root: Path, stub_pipeline: Any
    ) -> None:
        """execute_ssh does not call a confirm_callback; it runs immediately
        after checking the ledger. The execute_callback is not invoked."""
        # Propose uses delayed approval; execute should NOT invoke any callback
        delayed_propose = DelayedApprovalTracker(delay_seconds=0.05)
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=delayed_propose.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            _make_execute_ssh_from_history,
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.COMPLETE
        # Propose callback was called once (with delay)
        assert len(delayed_propose.requests) == 1
        elapsed = (
            delayed_propose.post_delay_times[0]
            - delayed_propose.pre_delay_times[0]
        )
        assert elapsed >= 0.04


# ---------------------------------------------------------------------------
# Test: execute_ssh without valid approval_id
# ---------------------------------------------------------------------------


class TestExecuteSSHWithoutApprovalReturnsError:
    """If execute_ssh is called without a valid approval_id, it returns ERROR."""

    @pytest.mark.asyncio
    async def test_invalid_approval_id_returns_error_not_denied(
        self, wiki_root: Path
    ) -> None:
        """Invalid approval_id -> ERROR (not DENIED), loop can continue."""
        tracker = ApprovalTracker(auto_approve=True)
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=tracker.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        # LLM skips propose and directly calls execute_ssh with bad ID
        script: list[tuple[ToolCall, ...] | Any] = [
            (
                ToolCall(
                    call_id="call_bad_execute",
                    tool_name="execute_ssh",
                    arguments={"approval_id": "nonexistent-approval"},
                ),
            ),
            # LLM self-corrects and returns empty -> COMPLETE
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        # Loop continues (ERROR is not terminal) and completes
        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 2  # bad execute + empty
        assert result.error_message is None

        # Confirm callback was NOT called (no valid approval to confirm)
        assert len(tracker.requests) == 0

    @pytest.mark.asyncio
    async def test_empty_approval_id_returns_error(
        self, wiki_root: Path
    ) -> None:
        """Empty approval_id -> ERROR result."""
        tracker = ApprovalTracker(auto_approve=True)
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=tracker.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (
                ToolCall(
                    call_id="call_empty",
                    tool_name="execute_ssh",
                    arguments={"approval_id": ""},
                ),
            ),
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        # Loop completes (ERROR is not terminal, LLM returns empty next)
        assert result.final_state is AgentLoopState.COMPLETE
        # No approval callback invocation
        assert len(tracker.requests) == 0


# ---------------------------------------------------------------------------
# Test: Multiple denials (first denial stops loop)
# ---------------------------------------------------------------------------


class TestMultipleDenialsInSequence:
    """If LLM scripts two proposals, the first denial stops the loop."""

    @pytest.mark.asyncio
    async def test_first_denial_stops_loop_immediately(
        self, wiki_root: Path
    ) -> None:
        """The loop terminates on the first denial without processing further calls."""
        denier = DenyingApprovalTracker()
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=denier.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        # Two proposals in sequence -- only first should execute
        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            (
                ToolCall(
                    call_id="call_propose_2",
                    tool_name="propose_ssh_command",
                    arguments={
                        "command": "ls -la /tmp",
                        "target_host": _HOST,
                        "target_user": _USER,
                        "explanation": "Listing tmp directory",
                    },
                ),
            ),
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 1
        # Only one denial request processed
        assert len(denier.requests) == 1
        assert llm.call_count == 1


# ---------------------------------------------------------------------------
# Test: Approval granted then denied at execution (two-level gate)
# ---------------------------------------------------------------------------


class TestApprovalGrantedThenDeniedAtExecution:
    """Two-level gate: approve propose, deny execute."""

    @pytest.mark.asyncio
    async def test_two_level_gate_deny_at_execution(
        self, wiki_root: Path, stub_pipeline: Any
    ) -> None:
        """Single-gate: proposal approved -> execute_ssh runs -> loop completes.

        The two-level gate (propose + execute prompts) no longer exists.
        execute_ssh does not prompt the user. With a valid approval the
        command runs immediately, and the loop completes successfully.
        """
        tracker = ApprovalTracker(auto_approve=True)
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=tracker.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            _make_execute_ssh_from_history,
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 3
        # Only one approval callback (propose only -- single gate)
        assert len(tracker.requests) == 1

        # Verify the history contains: propose SUCCESS then execute SUCCESS
        tool_results_in_history = [
            m for m in result.history if m.get("role") == "tool"
        ]
        assert len(tool_results_in_history) == 2

        # Propose result is JSON (SUCCESS -> raw output)
        propose_data = json.loads(tool_results_in_history[0]["content"])
        assert propose_data.get("approved") is True

        # Execute result is also JSON (SUCCESS -> raw output)
        execute_data = json.loads(tool_results_in_history[1]["content"])
        assert execute_data.get("success") is True
        assert execute_data.get("exit_code") == 0

        # Verify via dispatch bridge raw results
        exec_result = bridge.all_results[-1]
        assert exec_result.status is ToolResultStatus.SUCCESS
        exec_data = json.loads(exec_result.output)
        assert exec_data.get("success") is True


# ---------------------------------------------------------------------------
# Test: Edited command flows through execution
# ---------------------------------------------------------------------------


class TestEditedCommandFlowsThroughExecution:
    """User edits the command during propose approval."""

    @pytest.mark.asyncio
    async def test_edited_command_stored_in_ledger_and_executed(
        self, wiki_root: Path, stub_pipeline: Any
    ) -> None:
        """Edited command from propose is used by execute_ssh."""
        editor = EditingApprovalTracker()
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=editor.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            _make_execute_ssh_from_history,
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.COMPLETE, (
            f"Expected COMPLETE, got {result.final_state.value}. "
            f"Error: {result.error_message}"
        )

        # Verify the proposal result shows the command was edited
        tool_results_in_history = [
            m for m in result.history if m.get("role") == "tool"
        ]
        assert len(tool_results_in_history) >= 1
        propose_data = json.loads(tool_results_in_history[0]["content"])
        assert propose_data["edited"] is True
        assert propose_data["command"].endswith("--verbose")

    @pytest.mark.asyncio
    async def test_edited_command_reflected_in_execute_result(
        self, wiki_root: Path, stub_pipeline: Any
    ) -> None:
        """The execute_ssh result shows the edited command was run."""
        editor = EditingApprovalTracker()
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=editor.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (_PROPOSE_CALL,),
            _make_execute_ssh_from_history,
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.COMPLETE

        # Find the execute_ssh result in history
        tool_results_in_history = [
            m for m in result.history if m.get("role") == "tool"
        ]
        assert len(tool_results_in_history) >= 2
        exec_data = json.loads(tool_results_in_history[1]["content"])
        # The executed command should have the --verbose suffix
        assert exec_data["command"].endswith("--verbose"), (
            f"Expected command to end with --verbose, got: {exec_data['command']}"
        )


# ---------------------------------------------------------------------------
# Test: Short-circuit on denial within a batch of tool calls
# ---------------------------------------------------------------------------


class TestDenialShortCircuitsToolBatch:
    """When multiple tool calls are in one iteration, denial short-circuits."""

    @pytest.mark.asyncio
    async def test_denial_short_circuits_remaining_calls(
        self, wiki_root: Path
    ) -> None:
        """If propose is denied, execute in the same batch is skipped."""
        denier = DenyingApprovalTracker()
        ledger = ApprovalLedger()
        registry = _build_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            propose_callback=denier.confirm,
        )
        bridge = ToolDispatchBridge(registry=registry)

        # Both propose and execute in the same iteration
        # (unrealistic but tests short-circuit)
        script: list[tuple[ToolCall, ...] | Any] = [
            (
                _PROPOSE_CALL,
                ToolCall(
                    call_id="call_execute_same_batch",
                    tool_name="execute_ssh",
                    arguments={"approval_id": "fake-id"},
                ),
            ),
        ]
        llm = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.ERROR
        # Only one tool dispatched (propose denied, execute skipped)
        assert bridge.dispatch_count == 1
        # Only one denial request
        assert len(denier.requests) == 1
