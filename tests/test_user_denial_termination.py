"""Tests for AC 11: Permanent error (user denial) terminates loop immediately.

Validates:
    - User denial (DENIED ToolResult) terminates the agent loop with ERROR state
    - Denial during single-tool call terminates at iteration 1
    - Denial in a multi-tool batch short-circuits remaining calls
    - Denial from propose_ssh_command terminates the loop
    - Denial from execute_ssh terminates the loop
    - Error message clearly indicates denial
    - LLM is NOT called again after denial
    - History snapshot captures the denial context
    - ToolDispatchBridge short-circuits on denial during batch dispatch
    - propose_ssh_command DENIED result is terminal
    - execute_ssh DENIED result is terminal
    - Other permanent errors (SSH auth, malformed response) also terminate
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from jules_daemon.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopResult,
    AgentLoopState,
    LLMClient,
    ToolDispatcher,
)
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResult,
    ToolResultStatus,
)


# ---------------------------------------------------------------------------
# Mock helpers (self-contained to avoid cross-file conflicts)
# ---------------------------------------------------------------------------


def _make_call(
    tool_name: str,
    call_id: str = "call_0",
    **kwargs: Any,
) -> ToolCall:
    """Create a ToolCall with minimal boilerplate."""
    return ToolCall(
        call_id=call_id,
        tool_name=tool_name,
        arguments=kwargs if kwargs else {"arg": "val"},
    )


def _make_denied(call: ToolCall, message: str = "User denied the operation") -> ToolResult:
    """Create a DENIED ToolResult for the given call."""
    return ToolResult.denied(
        call_id=call.call_id,
        tool_name=call.tool_name,
        error_message=message,
    )


def _make_success(call: ToolCall, output: str = "ok") -> ToolResult:
    """Create a SUCCESS ToolResult for the given call."""
    return ToolResult.success(
        call_id=call.call_id,
        tool_name=call.tool_name,
        output=output,
    )


def _make_error(call: ToolCall, message: str = "failed") -> ToolResult:
    """Create an ERROR ToolResult for the given call."""
    return ToolResult.error(
        call_id=call.call_id,
        tool_name=call.tool_name,
        error_message=message,
    )


class _MockLLM:
    """Mock LLM that returns preconfigured tool call sequences."""

    def __init__(self, responses: list[tuple[ToolCall, ...]]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def get_tool_calls(
        self, messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self._call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return ()


class _MockDispatcher:
    """Mock dispatcher that returns configured results per call_id."""

    def __init__(self, results: dict[str, ToolResult] | None = None) -> None:
        self._results = dict(results) if results else {}
        self._dispatched: list[ToolCall] = []

    @property
    def dispatched_calls(self) -> tuple[ToolCall, ...]:
        return tuple(self._dispatched)

    @property
    def dispatched_names(self) -> tuple[str, ...]:
        return tuple(c.tool_name for c in self._dispatched)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        self._dispatched.append(call)
        if call.call_id in self._results:
            return self._results[call.call_id]
        return _make_success(call)


def _make_loop(
    llm: LLMClient,
    dispatcher: ToolDispatcher,
    max_iterations: int = 5,
) -> AgentLoop:
    """Create an AgentLoop with standard test config."""
    return AgentLoop(
        llm_client=llm,
        tool_dispatcher=dispatcher,
        system_prompt="You are a test runner assistant.",
        config=AgentLoopConfig(max_iterations=max_iterations),
    )


# ---------------------------------------------------------------------------
# AC 11 Core: User denial terminates loop immediately
# ---------------------------------------------------------------------------


class TestDenialTerminatesLoop:
    """Core tests: user denial produces ERROR state and stops the loop."""

    @pytest.mark.asyncio
    async def test_single_denied_call_terminates_with_error(self) -> None:
        """A single DENIED tool call terminates with ERROR state."""
        call = _make_call("propose_ssh_command", call_id="deny_1")
        denied = _make_denied(call)

        llm = _MockLLM([(call,)])
        dispatcher = _MockDispatcher(results={"deny_1": denied})

        result = await _make_loop(llm, dispatcher).run("run smoke tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 1
        assert result.error_message is not None
        assert "denied" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_denial_error_message_identifies_tool(self) -> None:
        """Error message names the tool that was denied."""
        call = _make_call("execute_ssh", call_id="deny_exec")
        denied = _make_denied(call, "User refused execution")

        llm = _MockLLM([(call,)])
        dispatcher = _MockDispatcher(results={"deny_exec": denied})

        result = await _make_loop(llm, dispatcher).run("execute test")

        assert result.error_message is not None
        assert "execute_ssh" in result.error_message
        assert "denied" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_denial_after_successful_iterations_terminates(self) -> None:
        """Denial at iteration 3 terminates despite earlier successes."""
        # Iterations 1 and 2: successful calls
        call_1 = _make_call("read_wiki", call_id="ok_1")
        call_2 = _make_call("lookup_test_spec", call_id="ok_2")
        # Iteration 3: denied
        call_3 = _make_call("propose_ssh_command", call_id="deny_3")
        denied = _make_denied(call_3)

        llm = _MockLLM([(call_1,), (call_2,), (call_3,)])
        dispatcher = _MockDispatcher(results={"deny_3": denied})

        result = await _make_loop(llm, dispatcher).run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 3
        assert "denied" in result.error_message.lower()


# ---------------------------------------------------------------------------
# AC 11: Short-circuit within multi-tool batch
# ---------------------------------------------------------------------------


class TestDenialShortCircuitsMultiToolBatch:
    """Denial in a multi-tool batch stops remaining dispatches."""

    @pytest.mark.asyncio
    async def test_denial_first_in_batch_skips_remaining(self) -> None:
        """When first call is denied, second call is NOT dispatched."""
        call_deny = _make_call("propose_ssh_command", call_id="deny_first")
        call_after = _make_call("read_wiki", call_id="after_deny")
        denied = _make_denied(call_deny)

        llm = _MockLLM([(call_deny, call_after)])
        dispatcher = _MockDispatcher(results={"deny_first": denied})

        result = await _make_loop(llm, dispatcher).run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        # Only the denied call was dispatched, not the second one
        assert dispatcher.dispatched_names == ("propose_ssh_command",)

    @pytest.mark.asyncio
    async def test_denial_second_in_batch_dispatches_first_only(self) -> None:
        """When second call is denied, only first two calls are dispatched."""
        call_ok = _make_call("read_wiki", call_id="ok_first")
        call_deny = _make_call("execute_ssh", call_id="deny_second")
        call_skipped = _make_call("read_output", call_id="skipped_third")
        denied = _make_denied(call_deny)

        llm = _MockLLM([(call_ok, call_deny, call_skipped)])
        dispatcher = _MockDispatcher(results={"deny_second": denied})

        result = await _make_loop(llm, dispatcher).run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        # First two dispatched, third skipped
        assert dispatcher.dispatched_names == ("read_wiki", "execute_ssh")
        assert "skipped_third" not in [
            c.call_id for c in dispatcher.dispatched_calls
        ]


# ---------------------------------------------------------------------------
# AC 11: LLM not called again after denial
# ---------------------------------------------------------------------------


class TestLLMNotCalledAfterDenial:
    """After denial, the LLM should NOT be called for another iteration."""

    @pytest.mark.asyncio
    async def test_llm_call_count_stops_at_denial(self) -> None:
        """LLM is called once (producing the denied call), then never again."""
        call = _make_call("propose_ssh_command", call_id="deny_llm")
        denied = _make_denied(call)

        # Give the LLM extra responses it should never reach
        extra_call = _make_call("read_wiki", call_id="never")
        llm = _MockLLM([(call,), (extra_call,), ()])
        dispatcher = _MockDispatcher(results={"deny_llm": denied})

        result = await _make_loop(llm, dispatcher, max_iterations=5).run("test")

        assert result.final_state is AgentLoopState.ERROR
        assert llm.call_count == 1  # Only the iteration that produced denial
        assert result.iterations_used == 1


# ---------------------------------------------------------------------------
# AC 11: History captures denial context
# ---------------------------------------------------------------------------


class TestDenialHistoryCapture:
    """History snapshot includes the denial for audit/debugging."""

    @pytest.mark.asyncio
    async def test_history_contains_denied_tool_message(self) -> None:
        """History has a tool-role message with ERROR prefix for the denial."""
        call = _make_call("propose_ssh_command", call_id="deny_hist")
        denied = _make_denied(call)

        llm = _MockLLM([(call,)])
        dispatcher = _MockDispatcher(results={"deny_hist": denied})

        result = await _make_loop(llm, dispatcher).run("test")

        # Find the tool message for the denied call
        tool_messages = [
            m for m in result.history
            if m.get("role") == "tool" and m.get("tool_call_id") == "deny_hist"
        ]
        assert len(tool_messages) == 1
        content = tool_messages[0]["content"]
        assert "ERROR" in content or "denied" in content.lower()

    @pytest.mark.asyncio
    async def test_history_starts_with_system_and_user(self) -> None:
        """Even on denial, history has the expected system and user messages."""
        call = _make_call("execute_ssh", call_id="deny_check")
        denied = _make_denied(call)

        llm = _MockLLM([(call,)])
        dispatcher = _MockDispatcher(results={"deny_check": denied})

        result = await _make_loop(llm, dispatcher).run("hello world")

        assert result.history[0]["role"] == "system"
        assert result.history[1]["role"] == "user"
        assert result.history[1]["content"] == "hello world"


# ---------------------------------------------------------------------------
# AC 11: Specific tool denial scenarios
# ---------------------------------------------------------------------------


class TestProposeSshDenialTerminates:
    """propose_ssh_command denial terminates the loop."""

    @pytest.mark.asyncio
    async def test_propose_denied_is_terminal(self) -> None:
        """DENIED from propose_ssh_command terminates with ERROR."""
        call = _make_call("propose_ssh_command", call_id="prop_deny")
        denied = _make_denied(call, "User rejected the proposed command")

        llm = _MockLLM([(call,)])
        dispatcher = _MockDispatcher(results={"prop_deny": denied})

        result = await _make_loop(llm, dispatcher).run("run command")

        assert result.final_state is AgentLoopState.ERROR
        assert "denied" in result.error_message.lower()
        assert result.iterations_used == 1


class TestExecuteSshDenialTerminates:
    """execute_ssh denial terminates the loop."""

    @pytest.mark.asyncio
    async def test_execute_denied_is_terminal(self) -> None:
        """DENIED from execute_ssh terminates with ERROR."""
        call = _make_call("execute_ssh", call_id="exec_deny")
        denied = _make_denied(call, "User denied command execution")

        llm = _MockLLM([(call,)])
        dispatcher = _MockDispatcher(results={"exec_deny": denied})

        result = await _make_loop(llm, dispatcher).run("execute")

        assert result.final_state is AgentLoopState.ERROR
        assert "denied" in result.error_message.lower()
        assert result.iterations_used == 1


# ---------------------------------------------------------------------------
# AC 11: Denial vs. error -- only denial is terminal
# ---------------------------------------------------------------------------


class TestDenialVsErrorTerminality:
    """Only DENIED is terminal; ERROR allows continued iteration."""

    @pytest.mark.asyncio
    async def test_error_does_not_terminate(self) -> None:
        """An ERROR result lets the loop continue to the next iteration."""
        call_err = _make_call("read_wiki", call_id="err_1")
        error_result = _make_error(call_err, "file not found")

        call_ok = _make_call("read_wiki", call_id="ok_1")

        llm = _MockLLM([(call_err,), (call_ok,), ()])
        dispatcher = _MockDispatcher(results={"err_1": error_result})

        result = await _make_loop(llm, dispatcher).run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 3

    @pytest.mark.asyncio
    async def test_denied_terminates_while_error_continues(self) -> None:
        """In a batch: ERROR on first call continues, DENIED on second stops."""
        call_err = _make_call("read_wiki", call_id="err_batch")
        call_deny = _make_call("propose_ssh_command", call_id="deny_batch")
        error_result = _make_error(call_err, "not found")
        denied_result = _make_denied(call_deny)

        llm = _MockLLM([(call_err, call_deny)])
        dispatcher = _MockDispatcher(results={
            "err_batch": error_result,
            "deny_batch": denied_result,
        })

        result = await _make_loop(llm, dispatcher).run("test")

        assert result.final_state is AgentLoopState.ERROR
        # Both were dispatched (error first, then denial)
        assert len(dispatcher.dispatched_calls) == 2
        assert "denied" in result.error_message.lower()


# ---------------------------------------------------------------------------
# AC 11: Other permanent errors also terminate immediately
# ---------------------------------------------------------------------------


class TestOtherPermanentErrorsTerminate:
    """Permanent errors beyond denial also terminate the loop immediately."""

    @pytest.mark.asyncio
    async def test_malformed_llm_response_terminates(self) -> None:
        """ValueError from LLM (malformed response) terminates without retry."""

        class _FailingLLM:
            call_count = 0

            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                self.call_count += 1
                raise ValueError("malformed JSON response")

        llm = _FailingLLM()
        dispatcher = _MockDispatcher()

        result = await _make_loop(llm, dispatcher, max_iterations=5).run("test")

        assert result.final_state is AgentLoopState.ERROR
        assert llm.call_count == 1  # No retries for permanent errors
        assert "malformed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_ssh_auth_failure_as_permanent_error(self) -> None:
        """RuntimeError from SSH auth is treated as permanent (not transient)."""

        class _AuthFailLLM:
            call_count = 0

            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                self.call_count += 1
                raise RuntimeError("SSH authentication failed")

        llm = _AuthFailLLM()
        dispatcher = _MockDispatcher()

        result = await _make_loop(llm, dispatcher).run("connect")

        assert result.final_state is AgentLoopState.ERROR
        assert llm.call_count == 1
        assert "ssh" in result.error_message.lower()


# ---------------------------------------------------------------------------
# AC 11: ToolResultStatus.DENIED.is_terminal contract
# ---------------------------------------------------------------------------


class TestDeniedStatusContract:
    """Verify the DENIED status is correctly classified as terminal."""

    def test_denied_is_terminal(self) -> None:
        assert ToolResultStatus.DENIED.is_terminal is True

    def test_success_is_not_terminal(self) -> None:
        assert ToolResultStatus.SUCCESS.is_terminal is False

    def test_error_is_not_terminal(self) -> None:
        assert ToolResultStatus.ERROR.is_terminal is False

    def test_timeout_is_not_terminal(self) -> None:
        assert ToolResultStatus.TIMEOUT.is_terminal is False

    def test_denied_result_is_terminal(self) -> None:
        result = ToolResult.denied(
            call_id="c1",
            tool_name="test",
            error_message="denied",
        )
        assert result.is_terminal is True
        assert result.is_denied is True

    def test_error_result_is_not_terminal(self) -> None:
        result = ToolResult.error(
            call_id="c1",
            tool_name="test",
            error_message="failed",
        )
        assert result.is_terminal is False
        assert result.is_denied is False


# ---------------------------------------------------------------------------
# AC 11: AgentLoopState after denial
# ---------------------------------------------------------------------------


class TestAgentLoopStateAfterDenial:
    """Verify the loop's internal state after denial termination."""

    @pytest.mark.asyncio
    async def test_loop_state_is_error_after_denial(self) -> None:
        """After denial, the loop.state property reads ERROR."""
        call = _make_call("propose_ssh_command", call_id="state_deny")
        denied = _make_denied(call)

        llm = _MockLLM([(call,)])
        dispatcher = _MockDispatcher(results={"state_deny": denied})

        loop = _make_loop(llm, dispatcher)
        await loop.run("test")

        assert loop.state is AgentLoopState.ERROR

    @pytest.mark.asyncio
    async def test_loop_state_transitions_through_phases(self) -> None:
        """Loop transitions THINKING -> ACTING -> ERROR on denial."""
        call = _make_call("execute_ssh", call_id="phase_deny")
        denied = _make_denied(call)

        states_seen: list[AgentLoopState] = []

        class _TrackingDispatcher:
            async def dispatch(self_, call: ToolCall) -> ToolResult:
                states_seen.append(loop.state)
                return denied

        llm = _MockLLM([(call,)])
        dispatcher = _TrackingDispatcher()

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=AgentLoopConfig(max_iterations=5),
        )
        result = await loop.run("test")

        # During dispatch, state should have been ACTING
        assert AgentLoopState.ACTING in states_seen
        # Final state should be ERROR
        assert loop.state is AgentLoopState.ERROR
        assert result.final_state is AgentLoopState.ERROR
