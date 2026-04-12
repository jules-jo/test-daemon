"""Tests for the tool dispatch bridge and result collection.

Validates the ToolDispatchBridge component that bridges parsed tool-call
requests from the response parser to the ToolRegistry, collects execution
results (including error/failure observations), and formats them back as
observation messages appended to conversation history for the next
iteration. This handles the ACTING and OBSERVING phases of the agent loop.

Test strategy:
    - DispatchResult immutability and classification
    - ToolDispatchBridge single call dispatch (satisfies ToolDispatcher protocol)
    - ToolDispatchBridge batch dispatch via dispatch_parsed_response
    - Terminal result detection (DENIED stops immediately)
    - Error results are non-terminal (observable by LLM)
    - History formatting: assistant message with tool_calls + tool result messages
    - Mixed success/error results in batch
    - Empty tool calls edge case
    - Protocol conformance with AgentLoop.ToolDispatcher
    - Integration with ConversationHistory (immutable append)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from jules_daemon.agent.conversation_history import (
    ConversationHistory,
    create_history,
)
from jules_daemon.agent.response_parser import (
    ToolCallsResponse,
)
from jules_daemon.agent.tool_dispatch import (
    DispatchOutcome,
    DispatchResult,
    ToolDispatchBridge,
)
from jules_daemon.agent.tool_registry import ToolRegistry
from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolCall,
    ToolParam,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)
from jules_daemon.agent.tools.base import BaseTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_call(
    name: str = "read_wiki",
    call_id: str = "call_001",
    args: dict[str, Any] | None = None,
) -> ToolCall:
    """Create a ToolCall with sensible defaults."""
    return ToolCall(
        call_id=call_id,
        tool_name=name,
        arguments=args or {"slug": "test"},
    )


def _make_success(call: ToolCall, output: str = "ok") -> ToolResult:
    """Create a success ToolResult."""
    return ToolResult.success(
        call_id=call.call_id,
        tool_name=call.tool_name,
        output=output,
    )


def _make_error(call: ToolCall, msg: str = "failed") -> ToolResult:
    """Create an error ToolResult."""
    return ToolResult.error(
        call_id=call.call_id,
        tool_name=call.tool_name,
        error_message=msg,
    )


def _make_denied(call: ToolCall) -> ToolResult:
    """Create a denied ToolResult."""
    return ToolResult.denied(
        call_id=call.call_id,
        tool_name=call.tool_name,
        error_message="User denied",
    )


class StubTool(BaseTool):
    """Minimal tool for testing registry integration."""

    def __init__(
        self,
        name: str = "read_wiki",
        result: ToolResult | None = None,
    ) -> None:
        self._spec = ToolSpec(
            name=name,
            description=f"Stub tool {name}",
            parameters=(
                ToolParam(
                    name="slug",
                    description="The page slug",
                    json_type="string",
                    required=False,
                ),
            ),
            approval=ApprovalRequirement.NONE,
        )
        self._result = result

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        call_id = args.get("_call_id", "unknown")
        if self._result is not None:
            return self._result
        return ToolResult.success(
            call_id=call_id,
            tool_name=self.name,
            output=f"stub output for {self.name}",
        )


class FailingTool(BaseTool):
    """Tool that always raises an exception."""

    def __init__(self, name: str = "broken_tool") -> None:
        self._spec = ToolSpec(
            name=name,
            description=f"Failing tool {name}",
            parameters=(),
            approval=ApprovalRequirement.NONE,
        )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("Intentional failure")


def _build_registry(*tools: BaseTool) -> ToolRegistry:
    """Build a ToolRegistry with the given tools."""
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


# ---------------------------------------------------------------------------
# DispatchOutcome enum
# ---------------------------------------------------------------------------


class TestDispatchOutcome:
    """Tests for the DispatchOutcome enum."""

    def test_all_members_exist(self) -> None:
        members = {o.name for o in DispatchOutcome}
        assert members == {"CONTINUE", "TERMINAL"}

    def test_continue_is_not_terminal(self) -> None:
        assert not DispatchOutcome.CONTINUE.is_terminal

    def test_terminal_is_terminal(self) -> None:
        assert DispatchOutcome.TERMINAL.is_terminal


# ---------------------------------------------------------------------------
# DispatchResult frozen dataclass
# ---------------------------------------------------------------------------


class TestDispatchResult:
    """Tests for the DispatchResult frozen dataclass."""

    def test_success_result(self) -> None:
        call = _make_call()
        result = _make_success(call)
        dr = DispatchResult(
            tool_calls=(call,),
            results=(result,),
            outcome=DispatchOutcome.CONTINUE,
            terminal_reason=None,
        )
        assert dr.outcome is DispatchOutcome.CONTINUE
        assert dr.terminal_reason is None
        assert len(dr.tool_calls) == 1
        assert len(dr.results) == 1

    def test_terminal_result(self) -> None:
        call = _make_call()
        result = _make_denied(call)
        dr = DispatchResult(
            tool_calls=(call,),
            results=(result,),
            outcome=DispatchOutcome.TERMINAL,
            terminal_reason="User denied",
        )
        assert dr.outcome is DispatchOutcome.TERMINAL
        assert dr.terminal_reason == "User denied"

    def test_frozen(self) -> None:
        call = _make_call()
        result = _make_success(call)
        dr = DispatchResult(
            tool_calls=(call,),
            results=(result,),
            outcome=DispatchOutcome.CONTINUE,
            terminal_reason=None,
        )
        with pytest.raises(AttributeError):
            dr.outcome = DispatchOutcome.TERMINAL  # type: ignore[misc]

    def test_has_terminal_convenience(self) -> None:
        call = _make_call()
        result = _make_denied(call)
        dr = DispatchResult(
            tool_calls=(call,),
            results=(result,),
            outcome=DispatchOutcome.TERMINAL,
            terminal_reason="denied",
        )
        assert dr.has_terminal is True

    def test_no_terminal_convenience(self) -> None:
        call = _make_call()
        result = _make_success(call)
        dr = DispatchResult(
            tool_calls=(call,),
            results=(result,),
            outcome=DispatchOutcome.CONTINUE,
            terminal_reason=None,
        )
        assert dr.has_terminal is False

    def test_all_succeeded_true(self) -> None:
        c1 = _make_call(call_id="c1")
        c2 = _make_call(call_id="c2")
        r1 = _make_success(c1)
        r2 = _make_success(c2)
        dr = DispatchResult(
            tool_calls=(c1, c2),
            results=(r1, r2),
            outcome=DispatchOutcome.CONTINUE,
            terminal_reason=None,
        )
        assert dr.all_succeeded is True

    def test_all_succeeded_false_with_error(self) -> None:
        c1 = _make_call(call_id="c1")
        c2 = _make_call(call_id="c2")
        r1 = _make_success(c1)
        r2 = _make_error(c2)
        dr = DispatchResult(
            tool_calls=(c1, c2),
            results=(r1, r2),
            outcome=DispatchOutcome.CONTINUE,
            terminal_reason=None,
        )
        assert dr.all_succeeded is False


# ---------------------------------------------------------------------------
# ToolDispatchBridge -- single dispatch
# ---------------------------------------------------------------------------


class TestToolDispatchBridgeSingleDispatch:
    """Tests for dispatching individual tool calls."""

    @pytest.mark.asyncio
    async def test_dispatch_success(self) -> None:
        """Single tool call dispatched successfully."""
        registry = _build_registry(StubTool("read_wiki"))
        bridge = ToolDispatchBridge(registry=registry)
        call = _make_call("read_wiki", call_id="c1")

        result = await bridge.dispatch(call)

        assert result.is_success
        assert result.tool_name == "read_wiki"

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self) -> None:
        """Dispatching an unknown tool returns an error result."""
        registry = _build_registry(StubTool("read_wiki"))
        bridge = ToolDispatchBridge(registry=registry)
        call = _make_call("nonexistent", call_id="c1")

        result = await bridge.dispatch(call)

        assert result.is_error
        assert "not registered" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_dispatch_satisfies_tool_dispatcher_protocol(self) -> None:
        """ToolDispatchBridge satisfies the ToolDispatcher protocol."""
        from jules_daemon.agent.agent_loop import ToolDispatcher

        registry = _build_registry(StubTool("read_wiki"))
        bridge = ToolDispatchBridge(registry=registry)
        assert isinstance(bridge, ToolDispatcher)


# ---------------------------------------------------------------------------
# ToolDispatchBridge -- batch dispatch via dispatch_parsed_response
# ---------------------------------------------------------------------------


class TestToolDispatchBridgeBatchDispatch:
    """Tests for batch dispatching from parsed responses."""

    @pytest.mark.asyncio
    async def test_dispatch_single_call(self) -> None:
        """Batch dispatch with a single tool call."""
        registry = _build_registry(StubTool("read_wiki"))
        bridge = ToolDispatchBridge(registry=registry)
        call = _make_call("read_wiki", call_id="c1")
        parsed = ToolCallsResponse(
            tool_calls=(call,),
            assistant_text=None,
        )

        dr = await bridge.dispatch_parsed_response(parsed)

        assert dr.outcome is DispatchOutcome.CONTINUE
        assert len(dr.results) == 1
        assert dr.results[0].is_success

    @pytest.mark.asyncio
    async def test_dispatch_multiple_calls(self) -> None:
        """Batch dispatch with multiple tool calls."""
        registry = _build_registry(
            StubTool("read_wiki"),
            StubTool("lookup_test_spec"),
        )
        bridge = ToolDispatchBridge(registry=registry)
        c1 = _make_call("read_wiki", call_id="c1")
        c2 = _make_call("lookup_test_spec", call_id="c2")
        parsed = ToolCallsResponse(
            tool_calls=(c1, c2),
            assistant_text=None,
        )

        dr = await bridge.dispatch_parsed_response(parsed)

        assert dr.outcome is DispatchOutcome.CONTINUE
        assert len(dr.results) == 2
        assert dr.results[0].is_success
        assert dr.results[1].is_success

    @pytest.mark.asyncio
    async def test_dispatch_preserves_call_order(self) -> None:
        """Results are in the same order as the tool calls."""
        registry = _build_registry(
            StubTool("read_wiki"),
            StubTool("lookup_test_spec"),
        )
        bridge = ToolDispatchBridge(registry=registry)
        c1 = _make_call("read_wiki", call_id="first")
        c2 = _make_call("lookup_test_spec", call_id="second")
        parsed = ToolCallsResponse(
            tool_calls=(c1, c2),
            assistant_text=None,
        )

        dr = await bridge.dispatch_parsed_response(parsed)

        assert dr.results[0].call_id == "first"
        assert dr.results[1].call_id == "second"

    @pytest.mark.asyncio
    async def test_dispatch_tool_calls_directly(self) -> None:
        """dispatch_tool_calls accepts a tuple of ToolCalls directly."""
        registry = _build_registry(StubTool("read_wiki"))
        bridge = ToolDispatchBridge(registry=registry)
        call = _make_call("read_wiki", call_id="c1")

        dr = await bridge.dispatch_tool_calls((call,))

        assert dr.outcome is DispatchOutcome.CONTINUE
        assert len(dr.results) == 1


# ---------------------------------------------------------------------------
# ToolDispatchBridge -- terminal detection
# ---------------------------------------------------------------------------


class TestToolDispatchBridgeTerminalDetection:
    """Tests for terminal result detection during dispatch."""

    @pytest.mark.asyncio
    async def test_denied_result_triggers_terminal(self) -> None:
        """A DENIED result makes the dispatch outcome TERMINAL."""
        denied_result = ToolResult.denied(
            call_id="c1",
            tool_name="propose_ssh_command",
            error_message="User denied",
        )
        tool = StubTool("propose_ssh_command", result=denied_result)
        registry = _build_registry(tool)
        bridge = ToolDispatchBridge(registry=registry)
        call = _make_call("propose_ssh_command", call_id="c1")
        parsed = ToolCallsResponse(
            tool_calls=(call,),
            assistant_text=None,
        )

        dr = await bridge.dispatch_parsed_response(parsed)

        assert dr.outcome is DispatchOutcome.TERMINAL
        assert dr.has_terminal is True
        assert dr.terminal_reason is not None
        assert "denied" in dr.terminal_reason.lower()

    @pytest.mark.asyncio
    async def test_error_result_is_not_terminal(self) -> None:
        """An ERROR result is observable, not terminal."""
        error_result = ToolResult.error(
            call_id="c1",
            tool_name="read_wiki",
            error_message="file not found",
        )
        tool = StubTool("read_wiki", result=error_result)
        registry = _build_registry(tool)
        bridge = ToolDispatchBridge(registry=registry)
        call = _make_call("read_wiki", call_id="c1")
        parsed = ToolCallsResponse(
            tool_calls=(call,),
            assistant_text=None,
        )

        dr = await bridge.dispatch_parsed_response(parsed)

        assert dr.outcome is DispatchOutcome.CONTINUE
        assert dr.has_terminal is False

    @pytest.mark.asyncio
    async def test_mixed_results_with_denial_is_terminal(self) -> None:
        """If any result is DENIED in a batch, outcome is TERMINAL."""
        success_result = ToolResult.success(
            call_id="c1",
            tool_name="read_wiki",
            output="ok",
        )
        denied_result = ToolResult.denied(
            call_id="c2",
            tool_name="propose_ssh_command",
            error_message="User denied",
        )
        tool_ok = StubTool("read_wiki", result=success_result)
        tool_deny = StubTool("propose_ssh_command", result=denied_result)
        registry = _build_registry(tool_ok, tool_deny)
        bridge = ToolDispatchBridge(registry=registry)

        c1 = _make_call("read_wiki", call_id="c1")
        c2 = _make_call("propose_ssh_command", call_id="c2")
        parsed = ToolCallsResponse(
            tool_calls=(c1, c2),
            assistant_text=None,
        )

        dr = await bridge.dispatch_parsed_response(parsed)

        assert dr.outcome is DispatchOutcome.TERMINAL
        assert len(dr.results) == 2
        # Both calls were dispatched despite the denial
        assert dr.results[0].is_success
        assert dr.results[1].is_denied


# ---------------------------------------------------------------------------
# ToolDispatchBridge -- exception handling
# ---------------------------------------------------------------------------


class TestToolDispatchBridgeExceptionHandling:
    """Tests for handling exceptions from tool execution."""

    @pytest.mark.asyncio
    async def test_tool_exception_returns_error_result(self) -> None:
        """An exception during tool execution produces an error result."""
        registry = _build_registry(FailingTool("broken_tool"))
        bridge = ToolDispatchBridge(registry=registry)
        call = _make_call("broken_tool", call_id="c1")
        parsed = ToolCallsResponse(
            tool_calls=(call,),
            assistant_text=None,
        )

        dr = await bridge.dispatch_parsed_response(parsed)

        assert dr.outcome is DispatchOutcome.CONTINUE
        assert dr.results[0].is_error
        assert "failed" in dr.results[0].error_message.lower()


# ---------------------------------------------------------------------------
# ToolDispatchBridge -- history formatting (OBSERVING phase)
# ---------------------------------------------------------------------------


class TestToolDispatchBridgeHistoryFormatting:
    """Tests for formatting dispatch results into conversation history."""

    @pytest.mark.asyncio
    async def test_format_observations_appends_to_history(self) -> None:
        """format_observations returns new history with assistant + tool msgs."""
        registry = _build_registry(StubTool("read_wiki"))
        bridge = ToolDispatchBridge(registry=registry)
        call = _make_call("read_wiki", call_id="c1")

        dr = DispatchResult(
            tool_calls=(call,),
            results=(
                ToolResult.success(
                    call_id="c1",
                    tool_name="read_wiki",
                    output="wiki content",
                ),
            ),
            outcome=DispatchOutcome.CONTINUE,
            terminal_reason=None,
        )

        history = create_history("system prompt", user_message="run tests")
        new_history = bridge.format_observations(history, dr)

        # Original history is unchanged (immutability)
        assert len(history) == 2
        # New history has: system + user + assistant(tool_calls) + tool(result)
        assert len(new_history) == 4

        msgs = new_history.to_openai_messages()
        assert msgs[2]["role"] == "assistant"
        assert "tool_calls" in msgs[2]
        assert msgs[3]["role"] == "tool"
        assert msgs[3]["tool_call_id"] == "c1"

    @pytest.mark.asyncio
    async def test_format_observations_with_error_result(self) -> None:
        """Error results include ERROR prefix in tool message content."""
        registry = _build_registry(StubTool("read_wiki"))
        bridge = ToolDispatchBridge(registry=registry)
        call = _make_call("read_wiki", call_id="c1")

        dr = DispatchResult(
            tool_calls=(call,),
            results=(
                ToolResult.error(
                    call_id="c1",
                    tool_name="read_wiki",
                    error_message="file not found",
                ),
            ),
            outcome=DispatchOutcome.CONTINUE,
            terminal_reason=None,
        )

        history = create_history("system prompt", user_message="run tests")
        new_history = bridge.format_observations(history, dr)

        msgs = new_history.to_openai_messages()
        tool_msg = msgs[3]
        assert tool_msg["role"] == "tool"
        assert "ERROR" in tool_msg["content"]
        assert "file not found" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_format_observations_multiple_results(self) -> None:
        """Multiple tool calls produce assistant + N tool result messages."""
        registry = _build_registry(
            StubTool("read_wiki"),
            StubTool("lookup_test_spec"),
        )
        bridge = ToolDispatchBridge(registry=registry)

        c1 = _make_call("read_wiki", call_id="c1")
        c2 = _make_call("lookup_test_spec", call_id="c2")

        dr = DispatchResult(
            tool_calls=(c1, c2),
            results=(
                ToolResult.success(
                    call_id="c1", tool_name="read_wiki", output="wiki data",
                ),
                ToolResult.success(
                    call_id="c2", tool_name="lookup_test_spec", output="spec data",
                ),
            ),
            outcome=DispatchOutcome.CONTINUE,
            terminal_reason=None,
        )

        history = create_history("system prompt", user_message="run tests")
        new_history = bridge.format_observations(history, dr)

        # system + user + assistant(2 tool_calls) + tool(c1) + tool(c2)
        assert len(new_history) == 5

        msgs = new_history.to_openai_messages()
        assert msgs[2]["role"] == "assistant"
        assert len(msgs[2]["tool_calls"]) == 2
        assert msgs[3]["role"] == "tool"
        assert msgs[3]["tool_call_id"] == "c1"
        assert msgs[4]["role"] == "tool"
        assert msgs[4]["tool_call_id"] == "c2"

    @pytest.mark.asyncio
    async def test_format_observations_preserves_assistant_text(self) -> None:
        """When DispatchResult has assistant_text, it is included."""
        registry = _build_registry(StubTool("read_wiki"))
        bridge = ToolDispatchBridge(registry=registry)
        call = _make_call("read_wiki", call_id="c1")

        dr = DispatchResult(
            tool_calls=(call,),
            results=(
                ToolResult.success(
                    call_id="c1", tool_name="read_wiki", output="data",
                ),
            ),
            outcome=DispatchOutcome.CONTINUE,
            terminal_reason=None,
            assistant_text="Let me look that up for you.",
        )

        history = create_history("system prompt", user_message="run tests")
        new_history = bridge.format_observations(history, dr)

        msgs = new_history.to_openai_messages()
        assistant_msg = msgs[2]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "Let me look that up for you."

    def test_format_observations_is_immutable(self) -> None:
        """format_observations returns a new history, does not mutate."""
        registry = _build_registry(StubTool("read_wiki"))
        bridge = ToolDispatchBridge(registry=registry)
        call = _make_call("read_wiki", call_id="c1")
        result = ToolResult.success(
            call_id="c1", tool_name="read_wiki", output="ok",
        )
        dr = DispatchResult(
            tool_calls=(call,),
            results=(result,),
            outcome=DispatchOutcome.CONTINUE,
            terminal_reason=None,
        )
        history = create_history("system prompt", user_message="test")

        new_history = bridge.format_observations(history, dr)

        assert len(history) == 2  # original unchanged
        assert len(new_history) > len(history)
        assert history is not new_history


# ---------------------------------------------------------------------------
# ToolDispatchBridge -- end-to-end dispatch + observe cycle
# ---------------------------------------------------------------------------


class TestToolDispatchBridgeEndToEnd:
    """Tests for the full dispatch-then-observe pipeline."""

    @pytest.mark.asyncio
    async def test_full_cycle_success(self) -> None:
        """Full cycle: parse -> dispatch -> format observations."""
        registry = _build_registry(StubTool("read_wiki"))
        bridge = ToolDispatchBridge(registry=registry)

        call = _make_call("read_wiki", call_id="c1")
        parsed = ToolCallsResponse(
            tool_calls=(call,),
            assistant_text="Looking up the wiki.",
        )

        # ACTING phase
        dr = await bridge.dispatch_parsed_response(parsed)
        assert dr.outcome is DispatchOutcome.CONTINUE

        # OBSERVING phase
        history = create_history("system prompt", user_message="help me")
        new_history = bridge.format_observations(history, dr)

        # Verify the complete observation chain
        msgs = new_history.to_openai_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "Looking up the wiki."
        assert msgs[3]["role"] == "tool"
        assert "stub output" in msgs[3]["content"]

    @pytest.mark.asyncio
    async def test_full_cycle_with_error_observation(self) -> None:
        """Full cycle: tool error is observable in conversation history."""
        error_result = ToolResult.error(
            call_id="c1",
            tool_name="read_wiki",
            error_message="page 'xyz' not found",
        )
        tool = StubTool("read_wiki", result=error_result)
        registry = _build_registry(tool)
        bridge = ToolDispatchBridge(registry=registry)

        call = _make_call("read_wiki", call_id="c1")
        parsed = ToolCallsResponse(
            tool_calls=(call,),
            assistant_text=None,
        )

        dr = await bridge.dispatch_parsed_response(parsed)

        history = create_history("system prompt", user_message="find it")
        new_history = bridge.format_observations(history, dr)

        msgs = new_history.to_openai_messages()
        tool_msg = msgs[3]
        assert tool_msg["role"] == "tool"
        # The error is formatted so the LLM can see and self-correct
        assert "ERROR" in tool_msg["content"]
        assert "page 'xyz' not found" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_self_correction_scenario(self) -> None:
        """Agent observes error, can propose correction in next iteration.

        Simulates two dispatch cycles:
        1. First call fails with error
        2. LLM sees error in history, makes corrected call that succeeds
        """
        # First cycle: error
        error_result = ToolResult.error(
            call_id="c1",
            tool_name="read_wiki",
            error_message="page not found",
        )
        tool_v1 = StubTool("read_wiki", result=error_result)
        registry_v1 = _build_registry(tool_v1)
        bridge = ToolDispatchBridge(registry=registry_v1)

        call_1 = _make_call("read_wiki", call_id="c1", args={"slug": "wrong"})
        parsed_1 = ToolCallsResponse(
            tool_calls=(call_1,),
            assistant_text=None,
        )
        dr_1 = await bridge.dispatch_parsed_response(parsed_1)

        history = create_history("system prompt", user_message="find data")
        history = bridge.format_observations(history, dr_1)

        # Error should be in history for LLM to observe
        msgs = history.to_openai_messages()
        assert any("ERROR" in str(m.get("content", "")) for m in msgs)

        # Second cycle: success with corrected registry
        success_result = ToolResult.success(
            call_id="c2",
            tool_name="read_wiki",
            output="found the data",
        )
        tool_v2 = StubTool("read_wiki", result=success_result)
        registry_v2 = _build_registry(tool_v2)
        bridge_2 = ToolDispatchBridge(registry=registry_v2)

        call_2 = _make_call("read_wiki", call_id="c2", args={"slug": "correct"})
        parsed_2 = ToolCallsResponse(
            tool_calls=(call_2,),
            assistant_text=None,
        )
        dr_2 = await bridge_2.dispatch_parsed_response(parsed_2)

        history = bridge_2.format_observations(history, dr_2)

        # Verify both cycles are in history
        all_msgs = history.to_openai_messages()
        tool_msgs = [m for m in all_msgs if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        # First was error, second was success
        assert "ERROR" in tool_msgs[0]["content"]
        assert "found the data" in tool_msgs[1]["content"]


# ---------------------------------------------------------------------------
# ToolDispatchBridge -- dispatch_count tracking
# ---------------------------------------------------------------------------


class TestToolDispatchBridgeTracking:
    """Tests for dispatch call tracking."""

    @pytest.mark.asyncio
    async def test_dispatch_count_increments(self) -> None:
        """dispatch_count tracks total calls dispatched."""
        registry = _build_registry(StubTool("read_wiki"))
        bridge = ToolDispatchBridge(registry=registry)

        assert bridge.dispatch_count == 0

        call = _make_call("read_wiki", call_id="c1")
        await bridge.dispatch(call)
        assert bridge.dispatch_count == 1

        c2 = _make_call("read_wiki", call_id="c2")
        c3 = _make_call("read_wiki", call_id="c3")
        parsed = ToolCallsResponse(
            tool_calls=(c2, c3),
            assistant_text=None,
        )
        await bridge.dispatch_parsed_response(parsed)
        assert bridge.dispatch_count == 3

    @pytest.mark.asyncio
    async def test_dispatched_results_tracked(self) -> None:
        """All dispatched results are tracked for post-loop analysis."""
        registry = _build_registry(StubTool("read_wiki"))
        bridge = ToolDispatchBridge(registry=registry)

        call = _make_call("read_wiki", call_id="c1")
        parsed = ToolCallsResponse(
            tool_calls=(call,),
            assistant_text=None,
        )
        await bridge.dispatch_parsed_response(parsed)

        results = bridge.all_results
        assert len(results) == 1
        assert results[0].call_id == "c1"
