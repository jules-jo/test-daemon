"""Extended unit tests for core agent loop orchestration.

Supplements test_agent_loop.py with additional coverage for:
    - Short-circuit behavior during acting phase (denial stops remaining calls)
    - Sequential dispatch ordering in acting phase
    - retry_exhausted flag propagation
    - Edge cases: max_iterations=1, empty tool call batches
    - History message format verification (OpenAI-compatible structure)
    - Tool error results visible in history for LLM self-correction
    - Mixed result statuses in a single batch
    - Observing phase appends both assistant and tool messages
    - Single-use semantics (loop should not be re-run)
    - Iteration counting accuracy across various scenarios
    - Transient retry within later iterations (not just first)
    - Cancel (KeyboardInterrupt) during LLM call
    - Max retries=0 means no retries at all
    - Backoff delays are not accumulated across iterations
"""

from __future__ import annotations

from typing import Any

import pytest

from jules_daemon.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopResult,
    AgentLoopState,
    LLMClient,
    ToolDispatcher,
    _compute_backoff_delay,
)
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResult,
    ToolResultStatus,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_call(
    name: str,
    call_id: str | None = None,
    **kwargs: Any,
) -> ToolCall:
    """Create a single ToolCall with sensible defaults."""
    cid = call_id or f"call_{name}"
    return ToolCall(call_id=cid, tool_name=name, arguments=kwargs or {"k": "v"})


def _success(call: ToolCall, output: str = "") -> ToolResult:
    return ToolResult.success(
        call_id=call.call_id,
        tool_name=call.tool_name,
        output=output or f"ok:{call.tool_name}",
    )


def _error(call: ToolCall, msg: str = "failed") -> ToolResult:
    return ToolResult.error(
        call_id=call.call_id,
        tool_name=call.tool_name,
        error_message=msg,
    )


def _denied(call: ToolCall, msg: str = "User denied") -> ToolResult:
    return ToolResult.denied(
        call_id=call.call_id,
        tool_name=call.tool_name,
        error_message=msg,
    )


class _SequenceLLM:
    """Mock LLM that returns a preconfigured sequence of tool call batches.

    After the sequence is exhausted, returns empty tuple (signals completion).
    Tracks every messages snapshot it receives.
    """

    def __init__(self, responses: list[tuple[ToolCall, ...]]) -> None:
        self._responses = list(responses)
        self._captured: list[tuple[dict[str, Any], ...]] = []

    @property
    def captured_messages(self) -> list[tuple[dict[str, Any], ...]]:
        return list(self._captured)

    @property
    def call_count(self) -> int:
        return len(self._captured)

    async def get_tool_calls(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self._captured.append(messages)
        if self._responses:
            return self._responses.pop(0)
        return ()


class _OrderTrackingDispatcher:
    """Mock dispatcher that records dispatch order and returns configured results."""

    def __init__(self, results: dict[str, ToolResult] | None = None) -> None:
        self._results: dict[str, ToolResult] = dict(results or {})
        self._order: list[str] = []

    @property
    def dispatch_order(self) -> tuple[str, ...]:
        return tuple(self._order)

    @property
    def dispatched_count(self) -> int:
        return len(self._order)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        self._order.append(call.call_id)
        if call.call_id in self._results:
            return self._results[call.call_id]
        return _success(call)


class _RecordingSleep:
    """No-op sleep that records delay values."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _make_loop(
    llm: LLMClient,
    dispatcher: ToolDispatcher | None = None,
    *,
    max_iterations: int = 5,
    max_retries: int = 2,
    retry_base_delay: float = 0.0,
    system_prompt: str = "system",
    sleep_fn: _RecordingSleep | None = None,
) -> AgentLoop:
    """Convenience factory for creating an AgentLoop with test defaults."""
    config = AgentLoopConfig(
        max_iterations=max_iterations,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )
    return AgentLoop(
        llm_client=llm,
        tool_dispatcher=dispatcher or _OrderTrackingDispatcher(),
        system_prompt=system_prompt,
        config=config,
        sleep_fn=sleep_fn or _RecordingSleep(),
    )


# ---------------------------------------------------------------------------
# Short-circuit behavior during acting phase
# ---------------------------------------------------------------------------


class TestActingPhaseShortCircuit:
    """Denial in a batch stops dispatching remaining tool calls."""

    @pytest.mark.asyncio
    async def test_denial_short_circuits_remaining_calls(self) -> None:
        """When the second of three calls is denied, the third is never dispatched."""
        c1 = _make_call("read_wiki", call_id="c1")
        c2 = _make_call("execute_ssh", call_id="c2")
        c3 = _make_call("read_output", call_id="c3")

        dispatcher = _OrderTrackingDispatcher(
            results={"c2": _denied(c2)},
        )
        llm = _SequenceLLM([(c1, c2, c3)])
        loop = _make_loop(llm, dispatcher)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        # Only c1 and c2 dispatched; c3 skipped
        assert dispatcher.dispatch_order == ("c1", "c2")
        assert dispatcher.dispatched_count == 2

    @pytest.mark.asyncio
    async def test_first_call_denied_skips_all_remaining(self) -> None:
        """When the very first call is denied, no other calls are dispatched."""
        c1 = _make_call("propose_ssh_command", call_id="first")
        c2 = _make_call("read_wiki", call_id="second")

        dispatcher = _OrderTrackingDispatcher(
            results={"first": _denied(c1)},
        )
        llm = _SequenceLLM([(c1, c2)])
        loop = _make_loop(llm, dispatcher)

        result = await loop.run("do it")

        assert result.final_state is AgentLoopState.ERROR
        assert dispatcher.dispatch_order == ("first",)

    @pytest.mark.asyncio
    async def test_error_result_does_not_short_circuit(self) -> None:
        """ERROR (non-terminal) results do not short-circuit; all calls dispatched."""
        c1 = _make_call("read_wiki", call_id="err1")
        c2 = _make_call("lookup_test_spec", call_id="ok2")

        dispatcher = _OrderTrackingDispatcher(
            results={"err1": _error(c1, "not found")},
        )
        llm = _SequenceLLM([(c1, c2), ()])
        loop = _make_loop(llm, dispatcher)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert dispatcher.dispatch_order == ("err1", "ok2")


# ---------------------------------------------------------------------------
# Sequential dispatch ordering
# ---------------------------------------------------------------------------


class TestSequentialDispatchOrder:
    """Tool calls are dispatched in order they appear in the LLM response."""

    @pytest.mark.asyncio
    async def test_dispatch_preserves_order(self) -> None:
        """Five tool calls dispatched in exact order."""
        calls = tuple(
            _make_call(f"tool_{i}", call_id=f"id_{i}") for i in range(5)
        )
        dispatcher = _OrderTrackingDispatcher()
        llm = _SequenceLLM([calls, ()])
        loop = _make_loop(llm, dispatcher)

        await loop.run("go")

        expected = tuple(f"id_{i}" for i in range(5))
        assert dispatcher.dispatch_order == expected


# ---------------------------------------------------------------------------
# retry_exhausted flag
# ---------------------------------------------------------------------------


class TestRetryExhaustedFlag:
    """The retry_exhausted flag signals fallback to one-shot path."""

    @pytest.mark.asyncio
    async def test_retry_exhausted_true_when_transient_exhausted(self) -> None:
        """retry_exhausted is True when all transient retries are consumed."""

        class _AlwaysFail:
            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                raise ConnectionError("persistent")

        llm = _AlwaysFail()
        loop = _make_loop(llm, max_retries=2)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True

    @pytest.mark.asyncio
    async def test_retry_exhausted_false_on_permanent_error(self) -> None:
        """retry_exhausted is False when termination is due to permanent error."""

        class _PermanentFail:
            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                raise ValueError("bad response")

        llm = _PermanentFail()
        loop = _make_loop(llm, max_retries=2)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is False

    @pytest.mark.asyncio
    async def test_retry_exhausted_false_on_normal_completion(self) -> None:
        """retry_exhausted is False on successful completion."""
        llm = _SequenceLLM([()])
        loop = _make_loop(llm)

        result = await loop.run("hello")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False

    @pytest.mark.asyncio
    async def test_retry_exhausted_false_on_max_iterations(self) -> None:
        """retry_exhausted is False when loop hits max iterations."""
        calls = [
            (_make_call("t", call_id=f"c{i}"),) for i in range(10)
        ]
        llm = _SequenceLLM(calls)
        loop = _make_loop(llm, max_iterations=3)

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is False

    @pytest.mark.asyncio
    async def test_retry_exhausted_false_on_denial(self) -> None:
        """retry_exhausted is False when terminated by user denial."""
        c = _make_call("execute_ssh", call_id="deny1")
        dispatcher = _OrderTrackingDispatcher(
            results={"deny1": _denied(c)},
        )
        llm = _SequenceLLM([(c,)])
        loop = _make_loop(llm, dispatcher)

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is False


# ---------------------------------------------------------------------------
# Edge cases: max_iterations=1
# ---------------------------------------------------------------------------


class TestMaxIterationsEdgeCases:
    """Edge cases around the max_iterations boundary."""

    @pytest.mark.asyncio
    async def test_max_iterations_one_with_tool_calls(self) -> None:
        """With max_iterations=1, a single tool call cycle exhausts the budget."""
        c = _make_call("read_wiki", call_id="only")
        llm = _SequenceLLM([(c,)])  # returns tool calls on first iteration
        loop = _make_loop(llm, max_iterations=1)

        result = await loop.run("go")

        # The loop processes iteration 1 (tool calls + observe), then
        # tries iteration 2 but hits the guard
        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 1
        assert "max iterations" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_max_iterations_one_immediate_complete(self) -> None:
        """With max_iterations=1, empty tool calls on first iteration -> COMPLETE."""
        llm = _SequenceLLM([()])
        loop = _make_loop(llm, max_iterations=1)

        result = await loop.run("hello")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 1

    @pytest.mark.asyncio
    async def test_max_iterations_exactly_reached(self) -> None:
        """Loop uses exactly max_iterations when LLM returns calls every time."""
        n = 4
        calls = [(_make_call("t", call_id=f"c{i}"),) for i in range(n + 5)]
        llm = _SequenceLLM(calls)
        loop = _make_loop(llm, max_iterations=n)

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == n

    @pytest.mark.asyncio
    async def test_completes_just_under_max(self) -> None:
        """If LLM returns empty on the last possible iteration, it is COMPLETE."""
        n = 3
        # Returns tool calls for iterations 1..(n-1), then empty on iteration n
        calls: list[tuple[ToolCall, ...]] = [
            (_make_call("t", call_id=f"c{i}"),) for i in range(n - 1)
        ]
        calls.append(())  # empty on iteration n
        llm = _SequenceLLM(calls)
        loop = _make_loop(llm, max_iterations=n)

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == n


# ---------------------------------------------------------------------------
# History message format verification
# ---------------------------------------------------------------------------


class TestHistoryMessageFormat:
    """Verify the exact OpenAI-compatible message format in history."""

    @pytest.mark.asyncio
    async def test_system_message_format(self) -> None:
        llm = _SequenceLLM([()])
        loop = _make_loop(llm, system_prompt="You are a helpful assistant")

        result = await loop.run("hello")

        assert result.history[0] == {
            "role": "system",
            "content": "You are a helpful assistant",
        }

    @pytest.mark.asyncio
    async def test_user_message_format(self) -> None:
        llm = _SequenceLLM([()])
        loop = _make_loop(llm)

        result = await loop.run("run smoke tests on staging")

        assert result.history[1] == {
            "role": "user",
            "content": "run smoke tests on staging",
        }

    @pytest.mark.asyncio
    async def test_assistant_message_has_tool_calls_key(self) -> None:
        """After dispatching tools, the assistant message includes tool_calls."""
        c = _make_call("read_wiki", call_id="tc1")
        llm = _SequenceLLM([(c,), ()])
        loop = _make_loop(llm)

        result = await loop.run("go")

        assistant_msgs = [m for m in result.history if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        first_asst = assistant_msgs[0]
        assert "tool_calls" in first_asst
        assert first_asst["content"] is None

    @pytest.mark.asyncio
    async def test_tool_result_message_format(self) -> None:
        """Tool result messages have role='tool' and tool_call_id."""
        c = _make_call("read_wiki", call_id="tc_fmt")
        dispatcher = _OrderTrackingDispatcher(
            results={"tc_fmt": _success(c, output="wiki content")},
        )
        llm = _SequenceLLM([(c,), ()])
        loop = _make_loop(llm, dispatcher)

        result = await loop.run("go")

        tool_msgs = [m for m in result.history if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        assert tool_msgs[0]["tool_call_id"] == "tc_fmt"
        assert tool_msgs[0]["content"] == "wiki content"

    @pytest.mark.asyncio
    async def test_error_tool_result_prefixed_with_error(self) -> None:
        """Error tool results have 'ERROR:' prefix in content."""
        c = _make_call("read_wiki", call_id="tc_err")
        dispatcher = _OrderTrackingDispatcher(
            results={"tc_err": _error(c, "file not found")},
        )
        llm = _SequenceLLM([(c,), ()])
        loop = _make_loop(llm, dispatcher)

        result = await loop.run("go")

        tool_msgs = [m for m in result.history if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        assert tool_msgs[0]["content"].startswith("ERROR:")
        assert "file not found" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_history_message_ordering(self) -> None:
        """Messages appear in order: system, user, [assistant, tool]+."""
        c1 = _make_call("read_wiki", call_id="ord1")
        c2 = _make_call("lookup_test_spec", call_id="ord2")
        llm = _SequenceLLM([(c1,), (c2,), ()])
        loop = _make_loop(llm)

        result = await loop.run("go")

        roles = [m["role"] for m in result.history]
        assert roles[0] == "system"
        assert roles[1] == "user"
        # After that, alternating assistant/tool pairs
        remaining = roles[2:]
        for i in range(0, len(remaining) - 1, 2):
            assert remaining[i] == "assistant"
            assert remaining[i + 1] == "tool"


# ---------------------------------------------------------------------------
# Tool error results in history for self-correction
# ---------------------------------------------------------------------------


class TestToolErrorInHistory:
    """Error results are appended to history so the LLM can self-correct."""

    @pytest.mark.asyncio
    async def test_error_result_visible_in_next_iteration_messages(self) -> None:
        """The LLM receives error results from the prior iteration."""
        c1 = _make_call("read_wiki", call_id="err_visible")
        c2 = _make_call("read_wiki", call_id="fix_visible")
        dispatcher = _OrderTrackingDispatcher(
            results={"err_visible": _error(c1, "not found")},
        )
        llm = _SequenceLLM([(c1,), (c2,), ()])
        loop = _make_loop(llm, dispatcher)

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.COMPLETE
        # The second LLM call (iteration 2) should have seen 4 messages:
        # system + user + assistant(tool_calls) + tool(error)
        second_call_msgs = llm.captured_messages[1]
        assert len(second_call_msgs) == 4
        # The tool message should contain the error
        tool_msg = second_call_msgs[3]
        assert tool_msg["role"] == "tool"
        assert "ERROR:" in tool_msg["content"]
        assert "not found" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_multiple_errors_all_visible(self) -> None:
        """Multiple errors in one batch are all visible in history."""
        c1 = _make_call("t1", call_id="e1")
        c2 = _make_call("t2", call_id="e2")
        dispatcher = _OrderTrackingDispatcher(
            results={
                "e1": _error(c1, "err1"),
                "e2": _error(c2, "err2"),
            },
        )
        llm = _SequenceLLM([(c1, c2), ()])
        loop = _make_loop(llm, dispatcher)

        result = await loop.run("go")

        # After iteration 1, history has system + user + assistant + 2 tool msgs
        tool_msgs = [m for m in result.history if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert "err1" in tool_msgs[0]["content"]
        assert "err2" in tool_msgs[1]["content"]


# ---------------------------------------------------------------------------
# Mixed result statuses in a single batch
# ---------------------------------------------------------------------------


class TestMixedResultStatuses:
    """Batches with mixed success/error results."""

    @pytest.mark.asyncio
    async def test_success_and_error_in_same_batch_continues(self) -> None:
        """A batch with both SUCCESS and ERROR continues the loop."""
        c_ok = _make_call("read_wiki", call_id="ok")
        c_err = _make_call("lookup_test_spec", call_id="err")
        dispatcher = _OrderTrackingDispatcher(
            results={"err": _error(c_err, "spec missing")},
        )
        llm = _SequenceLLM([(c_ok, c_err), ()])
        loop = _make_loop(llm, dispatcher)

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.COMPLETE
        # Both dispatched
        assert dispatcher.dispatch_order == ("ok", "err")

    @pytest.mark.asyncio
    async def test_timeout_result_does_not_terminate(self) -> None:
        """TIMEOUT status is non-terminal; the loop continues."""
        c = _make_call("check_remote", call_id="to1")
        timeout_result = ToolResult(
            call_id="to1",
            tool_name="check_remote",
            status=ToolResultStatus.TIMEOUT,
            output="",
            error_message="timed out after 30s",
        )
        dispatcher = _OrderTrackingDispatcher(
            results={"to1": timeout_result},
        )
        llm = _SequenceLLM([(c,), ()])
        loop = _make_loop(llm, dispatcher)

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.COMPLETE


# ---------------------------------------------------------------------------
# Max retries=0 means no retries
# ---------------------------------------------------------------------------


class TestZeroRetries:
    """max_retries=0 means transient errors fail immediately."""

    @pytest.mark.asyncio
    async def test_zero_retries_transient_fails_immediately(self) -> None:
        """With max_retries=0, a transient error terminates on first attempt."""

        class _FailOnce:
            call_count = 0

            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                self.call_count += 1
                raise ConnectionError("timeout")

        llm = _FailOnce()
        loop = _make_loop(llm, max_retries=0)

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.ERROR
        assert llm.call_count == 1
        assert result.retry_exhausted is True


# ---------------------------------------------------------------------------
# Transient retry within later iterations
# ---------------------------------------------------------------------------


class TestTransientRetryLaterIterations:
    """Retry budget resets per iteration."""

    @pytest.mark.asyncio
    async def test_retry_budget_resets_each_iteration(self) -> None:
        """Each iteration gets its own retry budget."""
        sleep_rec = _RecordingSleep()
        call_count = 0

        class _FailOnSecondIteration:
            """Succeeds on iter 1, fails once on iter 2, then completes."""

            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return (_make_call("read_wiki", call_id="i1c1"),)
                if call_count == 2:
                    raise ConnectionError("blip")
                return ()

        llm = _FailOnSecondIteration()
        loop = _make_loop(
            llm,
            max_retries=2,
            retry_base_delay=1.0,
            sleep_fn=sleep_rec,
        )

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.COMPLETE
        # Call 1: iter1 success, Call 2: iter2 fail, Call 3: iter2 retry success
        assert call_count == 3
        # One sleep for the retry in iteration 2
        assert len(sleep_rec.delays) == 1
        assert sleep_rec.delays[0] == 1.0


# ---------------------------------------------------------------------------
# Backoff delays not accumulated across iterations
# ---------------------------------------------------------------------------


class TestBackoffIsolation:
    """Backoff state does not leak between iterations."""

    @pytest.mark.asyncio
    async def test_backoff_resets_between_iterations(self) -> None:
        """The backoff attempt counter starts at 0 for each new iteration."""
        sleep_rec = _RecordingSleep()
        call_count = 0

        class _FailFirstCallEachIteration:
            """Fails once per iteration, then returns calls or empty."""

            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                nonlocal call_count
                call_count += 1
                # Iteration 1: fail, then succeed with tool calls
                if call_count == 1:
                    raise ConnectionError("iter1 blip")
                if call_count == 2:
                    return (_make_call("t", call_id="i1"),)
                # Iteration 2: fail, then succeed with empty (complete)
                if call_count == 3:
                    raise ConnectionError("iter2 blip")
                return ()

        llm = _FailFirstCallEachIteration()
        loop = _make_loop(
            llm,
            max_retries=2,
            retry_base_delay=1.0,
            sleep_fn=sleep_rec,
        )

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.COMPLETE
        # Two sleeps: one per iteration, both at attempt=0 -> base_delay * 2^0 = 1.0
        assert len(sleep_rec.delays) == 2
        assert sleep_rec.delays[0] == 1.0  # iter1 retry
        assert sleep_rec.delays[1] == 1.0  # iter2 retry (reset, not 2.0)


# ---------------------------------------------------------------------------
# History immutability in result
# ---------------------------------------------------------------------------


class TestHistoryImmutability:
    """The history in AgentLoopResult is an immutable tuple."""

    @pytest.mark.asyncio
    async def test_result_history_is_tuple(self) -> None:
        llm = _SequenceLLM([()])
        loop = _make_loop(llm)

        result = await loop.run("hello")

        assert isinstance(result.history, tuple)

    @pytest.mark.asyncio
    async def test_result_is_frozen(self) -> None:
        llm = _SequenceLLM([()])
        loop = _make_loop(llm)

        result = await loop.run("hello")

        with pytest.raises(AttributeError):
            result.iterations_used = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AgentLoopResult -- retry_exhausted default
# ---------------------------------------------------------------------------


class TestAgentLoopResultRetryExhausted:
    """Tests for the retry_exhausted field default."""

    def test_retry_exhausted_defaults_false(self) -> None:
        result = AgentLoopResult(
            final_state=AgentLoopState.COMPLETE,
            iterations_used=1,
            history=(),
            error_message=None,
        )
        assert result.retry_exhausted is False

    def test_retry_exhausted_explicit_true(self) -> None:
        result = AgentLoopResult(
            final_state=AgentLoopState.ERROR,
            iterations_used=1,
            history=(),
            error_message="exhausted",
            retry_exhausted=True,
        )
        assert result.retry_exhausted is True


# ---------------------------------------------------------------------------
# State transitions during the loop
# ---------------------------------------------------------------------------


class TestStateTransitions:
    """Verify that state transitions follow the expected pattern."""

    @pytest.mark.asyncio
    async def test_state_is_complete_after_successful_run(self) -> None:
        llm = _SequenceLLM([()])
        loop = _make_loop(llm)

        await loop.run("hello")

        assert loop.state is AgentLoopState.COMPLETE

    @pytest.mark.asyncio
    async def test_state_is_error_after_max_iterations(self) -> None:
        calls = [(_make_call("t", call_id=f"c{i}"),) for i in range(10)]
        llm = _SequenceLLM(calls)
        loop = _make_loop(llm, max_iterations=2)

        await loop.run("go")

        assert loop.state is AgentLoopState.ERROR

    @pytest.mark.asyncio
    async def test_state_is_error_after_denial(self) -> None:
        c = _make_call("execute_ssh", call_id="d1")
        dispatcher = _OrderTrackingDispatcher(results={"d1": _denied(c)})
        llm = _SequenceLLM([(c,)])
        loop = _make_loop(llm, dispatcher)

        await loop.run("go")

        assert loop.state is AgentLoopState.ERROR

    @pytest.mark.asyncio
    async def test_state_is_error_after_permanent_llm_error(self) -> None:
        class _BadLLM:
            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                raise ValueError("malformed")

        loop = _make_loop(_BadLLM())

        await loop.run("go")

        assert loop.state is AgentLoopState.ERROR


# ---------------------------------------------------------------------------
# Iteration count accuracy
# ---------------------------------------------------------------------------


class TestIterationCounting:
    """Precise iteration counting across scenarios."""

    @pytest.mark.asyncio
    async def test_immediate_empty_is_one_iteration(self) -> None:
        """LLM returns empty on first call -> 1 iteration consumed."""
        llm = _SequenceLLM([()])
        loop = _make_loop(llm)

        result = await loop.run("hello")

        assert result.iterations_used == 1

    @pytest.mark.asyncio
    async def test_two_tool_rounds_plus_complete_is_three(self) -> None:
        """Two rounds of tool calls + empty completion = 3 iterations."""
        c1 = _make_call("t1", call_id="r1")
        c2 = _make_call("t2", call_id="r2")
        llm = _SequenceLLM([(c1,), (c2,), ()])
        loop = _make_loop(llm)

        result = await loop.run("go")

        assert result.iterations_used == 3

    @pytest.mark.asyncio
    async def test_denial_on_first_iteration_is_one(self) -> None:
        """Denial on first tool call -> 1 iteration."""
        c = _make_call("execute_ssh", call_id="d")
        dispatcher = _OrderTrackingDispatcher(results={"d": _denied(c)})
        llm = _SequenceLLM([(c,)])
        loop = _make_loop(llm, dispatcher)

        result = await loop.run("go")

        assert result.iterations_used == 1

    @pytest.mark.asyncio
    async def test_transient_retries_do_not_count_as_iterations(self) -> None:
        """Retries within one iteration don't increment the iteration count."""
        call_count = 0

        class _FailTwiceThenComplete:
            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise ConnectionError("blip")
                return ()

        llm = _FailTwiceThenComplete()
        loop = _make_loop(llm, max_retries=2)

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 1
        assert call_count == 3  # 2 failures + 1 success


# ---------------------------------------------------------------------------
# Multiple termination conditions are OR-composed
# ---------------------------------------------------------------------------


class TestTerminationConditionsOrComposed:
    """Any single termination condition stops the loop."""

    @pytest.mark.asyncio
    async def test_denial_terminates_before_max_iterations(self) -> None:
        """Denial terminates even if max_iterations not reached."""
        c1 = _make_call("read_wiki", call_id="ok1")
        c2 = _make_call("execute_ssh", call_id="deny2")
        dispatcher = _OrderTrackingDispatcher(results={"deny2": _denied(c2)})
        llm = _SequenceLLM([(c1,), (c2,)])
        loop = _make_loop(llm, dispatcher, max_iterations=10)

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 2
        assert "denied" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_empty_tool_calls_terminates_before_max(self) -> None:
        """Empty tool calls complete even if iterations remain."""
        llm = _SequenceLLM([()])
        loop = _make_loop(llm, max_iterations=10)

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 1

    @pytest.mark.asyncio
    async def test_permanent_error_terminates_before_max(self) -> None:
        """Permanent LLM error terminates before max_iterations."""

        class _PermFail:
            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                raise ValueError("bad")

        loop = _make_loop(_PermFail(), max_iterations=10)

        result = await loop.run("go")

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 1


# ---------------------------------------------------------------------------
# Protocol conformance checks
# ---------------------------------------------------------------------------


class TestProtocolConformanceExtended:
    """Verify custom mocks satisfy the runtime-checkable protocols."""

    def test_sequence_llm_satisfies_llm_client(self) -> None:
        llm = _SequenceLLM([])
        assert isinstance(llm, LLMClient)

    def test_order_tracking_dispatcher_satisfies_tool_dispatcher(self) -> None:
        d = _OrderTrackingDispatcher()
        assert isinstance(d, ToolDispatcher)


# ---------------------------------------------------------------------------
# AgentLoopConfig -- combined validation
# ---------------------------------------------------------------------------


class TestAgentLoopConfigCombinedValidation:
    """Validation of multiple config fields together."""

    def test_valid_minimal_config(self) -> None:
        config = AgentLoopConfig(max_iterations=1, max_retries=0, retry_base_delay=0.0)
        assert config.max_iterations == 1
        assert config.max_retries == 0
        assert config.retry_base_delay == 0.0

    def test_large_values_accepted(self) -> None:
        config = AgentLoopConfig(
            max_iterations=100, max_retries=50, retry_base_delay=60.0,
        )
        assert config.max_iterations == 100
        assert config.max_retries == 50

    @pytest.mark.parametrize("max_iter", [-1, -100, 0])
    def test_invalid_max_iterations(self, max_iter: int) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            AgentLoopConfig(max_iterations=max_iter)

    @pytest.mark.parametrize("max_ret", [-1, -100])
    def test_invalid_max_retries(self, max_ret: int) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            AgentLoopConfig(max_retries=max_ret)

    @pytest.mark.parametrize("delay", [-0.1, -1.0, -100.0])
    def test_invalid_retry_base_delay(self, delay: float) -> None:
        with pytest.raises(ValueError, match="retry_base_delay"):
            AgentLoopConfig(retry_base_delay=delay)


# ---------------------------------------------------------------------------
# Backoff delay helper
# ---------------------------------------------------------------------------


class TestComputeBackoffDelayExtended:
    """Extended tests for the backoff delay computation."""

    @pytest.mark.parametrize(
        "attempt, base, expected",
        [
            (0, 1.0, 1.0),
            (1, 1.0, 2.0),
            (2, 1.0, 4.0),
            (3, 1.0, 8.0),
            (0, 0.5, 0.5),
            (1, 0.5, 1.0),
            (2, 0.5, 2.0),
            (0, 2.0, 2.0),
            (1, 2.0, 4.0),
        ],
    )
    def test_parametrized_backoff(
        self, attempt: int, base: float, expected: float,
    ) -> None:
        assert _compute_backoff_delay(attempt, base) == expected

    def test_zero_base_always_zero(self) -> None:
        for attempt in range(10):
            assert _compute_backoff_delay(attempt, 0.0) == 0.0
