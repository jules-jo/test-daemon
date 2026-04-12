"""Integration test: failure-observation-correction cycles (AC 130302, Sub-AC 2).

Validates that the agent loop correctly handles failure-observation-correction
cycles where a mocked tool execution fails, the mocked LLM observes the error
in its conversation history, and retries with a corrected tool call.

This test suite is distinct from test_demo_2_self_correction.py (AC 7) in
that it focuses specifically on the mechanics of:

    1. Tool failure propagation through conversation history
    2. LLM inspection of prior error results to inform correction
    3. Corrected tool call dispatched after error observation
    4. Multiple sequential failures with progressive correction
    5. Mixed error/success batches with correction on next iteration
    6. Error format visibility (ERROR: prefix) in LLM context
    7. Loop state transitions during correction cycles

Test scenarios:

    TestSingleFailureCorrection
        Tool fails once, LLM observes error, retries with corrected
        arguments, second attempt succeeds.

    TestMultipleSequentialFailures
        Tool fails on two consecutive iterations, LLM progressively
        corrects, third attempt succeeds.

    TestMixedBatchFailureCorrection
        First iteration has two tools: one fails, one succeeds. The
        LLM observes both results and corrects the failed tool on the
        next iteration.

    TestErrorContentVisibleToLLM
        Verifies the exact format and content of error messages that
        the LLM receives in the conversation history.

    TestCorrectionWithDifferentToolName
        First tool fails, LLM pivots to a different tool (not just
        different arguments) as the correction strategy.

    TestCorrectionPreservesFullHistory
        After failure and correction, the full conversation history
        (including the failed attempt) is available to the LLM and
        in the final result.

    TestFailureCorrectionWithinIterationBudget
        The correction cycle stays within the max_iterations budget
        and the final result reports the correct iteration count.
"""

from __future__ import annotations

import json
from typing import Any

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
# Shared helpers
# ---------------------------------------------------------------------------


def _make_call(
    name: str,
    call_id: str | None = None,
    **kwargs: Any,
) -> ToolCall:
    """Create a ToolCall with sensible defaults."""
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


class _RecordingSleep:
    """No-op sleep that records delay values."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


# ---------------------------------------------------------------------------
# History-aware LLM mock: inspects prior errors and adjusts tool calls
# ---------------------------------------------------------------------------


class HistoryAwareLLM:
    """Mock LLM that inspects conversation history to produce corrections.

    Unlike the simple ScriptedLLMClient, this LLM examines the tool result
    messages in the history to decide which tool calls to make. This models
    the real self-correction behavior: the LLM sees the error output and
    adjusts its next action.

    The ``response_fn`` receives the full conversation history and returns
    tool calls. It can check for ERROR: prefixes, parse tool result content,
    and generate corrected tool calls accordingly.
    """

    def __init__(
        self,
        response_fn: Any,  # Callable[[tuple[dict, ...]], tuple[ToolCall, ...]]
    ) -> None:
        self._response_fn = response_fn
        self._call_count = 0
        self._received_messages: list[tuple[dict[str, Any], ...]] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def received_messages(self) -> list[tuple[dict[str, Any], ...]]:
        return list(self._received_messages)

    async def get_tool_calls(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self._call_count += 1
        self._received_messages.append(messages)
        return self._response_fn(messages, self._call_count)


class ScriptedDispatcher:
    """Mock dispatcher returning preconfigured results by call_id.

    Falls back to a generic success if no result is configured.
    Tracks all dispatched calls for sequence verification.
    """

    def __init__(
        self,
        results: dict[str, ToolResult],
    ) -> None:
        self._results = dict(results)
        self._dispatched: list[ToolCall] = []

    @property
    def dispatched_calls(self) -> tuple[ToolCall, ...]:
        return tuple(self._dispatched)

    @property
    def dispatched_tool_names(self) -> tuple[str, ...]:
        return tuple(c.tool_name for c in self._dispatched)

    @property
    def dispatched_call_ids(self) -> tuple[str, ...]:
        return tuple(c.call_id for c in self._dispatched)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        self._dispatched.append(call)
        if call.call_id in self._results:
            return self._results[call.call_id]
        return ToolResult.success(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output=f"default_ok:{call.tool_name}",
        )


def _make_loop(
    llm: LLMClient,
    dispatcher: ToolDispatcher,
    *,
    max_iterations: int = 5,
    system_prompt: str = "You are a test runner assistant.",
) -> AgentLoop:
    """Build an AgentLoop with test defaults (no sleep delay)."""
    return AgentLoop(
        llm_client=llm,
        tool_dispatcher=dispatcher,
        system_prompt=system_prompt,
        config=AgentLoopConfig(
            max_iterations=max_iterations,
            max_retries=2,
            retry_base_delay=0.0,
        ),
        sleep_fn=_RecordingSleep(),
    )


def _find_error_tool_messages(
    messages: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract tool-role messages that start with ERROR:."""
    return [
        msg
        for msg in messages
        if msg.get("role") == "tool"
        and isinstance(msg.get("content", ""), str)
        and msg["content"].startswith("ERROR:")
    ]


def _find_tool_messages(
    messages: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract all tool-role messages."""
    return [msg for msg in messages if msg.get("role") == "tool"]


# ---------------------------------------------------------------------------
# Test: Single failure -> observation -> correction -> success
# ---------------------------------------------------------------------------


class TestSingleFailureCorrection:
    """Tool fails once, LLM observes error, retries with corrected args."""

    @pytest.mark.asyncio
    async def test_llm_observes_error_and_corrects(self) -> None:
        """The LLM sees the error from iteration 1 and issues a corrected
        tool call on iteration 2 that succeeds."""

        # The read_wiki call with wrong slug fails; LLM then retries
        # with the correct slug
        wrong_call = ToolCall(
            call_id="call_wrong_slug",
            tool_name="read_wiki",
            arguments={"slug": "smok-suite"},  # typo
        )
        correct_call = ToolCall(
            call_id="call_correct_slug",
            tool_name="read_wiki",
            arguments={"slug": "smoke-suite"},  # corrected
        )

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            if call_number == 1:
                return (wrong_call,)
            if call_number == 2:
                # LLM should see the error from call_number 1
                error_msgs = _find_error_tool_messages(messages)
                assert len(error_msgs) >= 1, (
                    "LLM should see at least one ERROR tool message "
                    "from the failed read_wiki call"
                )
                assert "not found" in error_msgs[0]["content"].lower()
                return (correct_call,)
            return ()  # done

        dispatcher = ScriptedDispatcher(
            results={
                "call_wrong_slug": _error(
                    wrong_call,
                    "Wiki page not found: 'smok-suite'. "
                    "Did you mean 'smoke-suite'?",
                ),
                "call_correct_slug": _success(
                    correct_call,
                    output=json.dumps({
                        "slug": "smoke-suite",
                        "content": "Smoke suite documentation...",
                    }),
                ),
            },
        )

        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher)
        result = await loop.run("look up the smoke suite docs")

        # Loop completes successfully after correction
        assert result.final_state is AgentLoopState.COMPLETE
        assert result.error_message is None
        # 2 iterations with tools + 1 empty completion = 3
        assert result.iterations_used == 3
        # Both calls dispatched in order
        assert dispatcher.dispatched_call_ids == (
            "call_wrong_slug",
            "call_correct_slug",
        )

    @pytest.mark.asyncio
    async def test_error_result_is_non_terminal(self) -> None:
        """ERROR status from a tool is non-terminal; the loop continues."""
        failing_call = _make_call("read_wiki", call_id="fail1")
        fix_call = _make_call("read_wiki", call_id="fix1")

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            if call_number == 1:
                return (failing_call,)
            if call_number == 2:
                return (fix_call,)
            return ()

        dispatcher = ScriptedDispatcher(
            results={"fail1": _error(failing_call, "read failed")},
        )

        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher)
        result = await loop.run("go")

        # Completes normally (ERROR is not terminal)
        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 3


# ---------------------------------------------------------------------------
# Test: Multiple sequential failures with progressive correction
# ---------------------------------------------------------------------------


class TestMultipleSequentialFailures:
    """Tool fails on consecutive iterations, LLM progressively corrects."""

    @pytest.mark.asyncio
    async def test_two_failures_then_success(self) -> None:
        """Two consecutive tool failures, each observed by the LLM,
        which adjusts arguments until the third attempt succeeds."""

        call_v1 = ToolCall(
            call_id="call_v1",
            tool_name="execute_ssh",
            arguments={"approval_id": "apv-001"},
        )
        call_v2 = ToolCall(
            call_id="call_v2",
            tool_name="execute_ssh",
            arguments={"approval_id": "apv-002"},
        )
        call_v3 = ToolCall(
            call_id="call_v3",
            tool_name="execute_ssh",
            arguments={"approval_id": "apv-003"},
        )

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            if call_number == 1:
                return (call_v1,)
            if call_number == 2:
                # Should see error from v1
                errors = _find_error_tool_messages(messages)
                assert len(errors) >= 1
                assert "permission denied" in errors[-1]["content"].lower()
                return (call_v2,)
            if call_number == 3:
                # Should see errors from v1 and v2
                errors = _find_error_tool_messages(messages)
                assert len(errors) >= 2
                assert "timeout" in errors[-1]["content"].lower()
                return (call_v3,)
            return ()

        dispatcher = ScriptedDispatcher(
            results={
                "call_v1": _error(call_v1, "Permission denied: use sudo"),
                "call_v2": _error(call_v2, "Timeout: connection timed out after 30s"),
                "call_v3": _success(
                    call_v3,
                    output=json.dumps({"success": True, "exit_code": 0}),
                ),
            },
        )

        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher, max_iterations=5)
        result = await loop.run("run integration tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.error_message is None
        # 3 iterations with tool calls + 1 empty = 4
        assert result.iterations_used == 4
        assert dispatcher.dispatched_call_ids == (
            "call_v1", "call_v2", "call_v3",
        )

    @pytest.mark.asyncio
    async def test_error_count_grows_in_history(self) -> None:
        """Each iteration accumulates more error messages in the history."""
        calls = [
            ToolCall(
                call_id=f"call_attempt_{i}",
                tool_name="lookup_test_spec",
                arguments={"test_name": f"attempt_{i}"},
            )
            for i in range(3)
        ]

        error_counts_seen: list[int] = []

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            errors = _find_error_tool_messages(messages)
            error_counts_seen.append(len(errors))
            if call_number <= 3:
                return (calls[call_number - 1],)
            return ()

        dispatcher = ScriptedDispatcher(
            results={
                "call_attempt_0": _error(calls[0], "error_0"),
                "call_attempt_1": _error(calls[1], "error_1"),
                # call_attempt_2 uses default success
            },
        )

        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher, max_iterations=5)
        result = await loop.run("go")

        assert result.final_state is AgentLoopState.COMPLETE
        # Iteration 1: 0 errors seen (no prior history)
        # Iteration 2: 1 error seen (from attempt_0)
        # Iteration 3: 2 errors seen (from attempt_0 + attempt_1)
        # Iteration 4: 2 errors (no new error from attempt_2 success)
        assert error_counts_seen == [0, 1, 2, 2]


# ---------------------------------------------------------------------------
# Test: Mixed batch with error and success in same iteration
# ---------------------------------------------------------------------------


class TestMixedBatchFailureCorrection:
    """A batch with mixed error/success results; LLM corrects on next iteration."""

    @pytest.mark.asyncio
    async def test_one_fails_one_succeeds_llm_corrects_failed(self) -> None:
        """In a batch of two tools, one fails and one succeeds. The LLM
        observes both results and re-invokes only the failed tool."""

        ok_call = ToolCall(
            call_id="batch_ok",
            tool_name="read_wiki",
            arguments={"query": "smoke tests"},
        )
        fail_call = ToolCall(
            call_id="batch_fail",
            tool_name="lookup_test_spec",
            arguments={"test_name": "smke_tests"},  # typo
        )
        corrected_call = ToolCall(
            call_id="batch_corrected",
            tool_name="lookup_test_spec",
            arguments={"test_name": "smoke_tests"},  # fixed
        )

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            if call_number == 1:
                return (ok_call, fail_call)
            if call_number == 2:
                # Both results should be visible
                tool_msgs = _find_tool_messages(messages)
                assert len(tool_msgs) >= 2, (
                    "LLM should see both tool results from the batch"
                )
                error_msgs = _find_error_tool_messages(messages)
                assert len(error_msgs) >= 1, (
                    "LLM should see the ERROR from the failed tool"
                )
                # The success should also be visible
                success_msgs = [
                    m for m in tool_msgs
                    if not m.get("content", "").startswith("ERROR:")
                ]
                assert len(success_msgs) >= 1
                return (corrected_call,)
            return ()

        dispatcher = ScriptedDispatcher(
            results={
                "batch_ok": _success(ok_call, output="wiki content here"),
                "batch_fail": _error(
                    fail_call, "No test spec found for 'smke_tests'"
                ),
                "batch_corrected": _success(
                    corrected_call,
                    output=json.dumps({"found": True, "test_slug": "smoke-tests"}),
                ),
            },
        )

        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher)
        result = await loop.run("look up smoke tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 3
        # Verify dispatch sequence: both from batch, then correction
        assert dispatcher.dispatched_call_ids == (
            "batch_ok", "batch_fail", "batch_corrected",
        )


# ---------------------------------------------------------------------------
# Test: Error content format visible to LLM
# ---------------------------------------------------------------------------


class TestErrorContentVisibleToLLM:
    """Verify the exact format of error messages the LLM receives."""

    @pytest.mark.asyncio
    async def test_error_prefix_format(self) -> None:
        """Tool errors arrive as 'ERROR: <message>' in the tool message content."""
        failing_call = ToolCall(
            call_id="fmt_fail",
            tool_name="read_wiki",
            arguments={"slug": "nonexistent"},
        )
        error_message = "Wiki page 'nonexistent' does not exist in the knowledge base"

        captured_error_content: list[str] = []

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            if call_number == 1:
                return (failing_call,)
            if call_number == 2:
                error_msgs = _find_error_tool_messages(messages)
                for msg in error_msgs:
                    captured_error_content.append(msg["content"])
                return ()
            return ()

        dispatcher = ScriptedDispatcher(
            results={
                "fmt_fail": _error(failing_call, error_message),
            },
        )

        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher)
        await loop.run("read nonexistent wiki page")

        assert len(captured_error_content) == 1
        assert captured_error_content[0] == f"ERROR: {error_message}"

    @pytest.mark.asyncio
    async def test_error_message_contains_tool_call_id(self) -> None:
        """The tool result message includes the correct tool_call_id
        so the LLM can correlate it with the failed call."""
        failing_call = ToolCall(
            call_id="corr_fail_123",
            tool_name="lookup_test_spec",
            arguments={"test_name": "bad_name"},
        )

        captured_tool_call_ids: list[str] = []

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            if call_number == 1:
                return (failing_call,)
            if call_number == 2:
                tool_msgs = _find_tool_messages(messages)
                for msg in tool_msgs:
                    captured_tool_call_ids.append(msg.get("tool_call_id", ""))
                return ()
            return ()

        dispatcher = ScriptedDispatcher(
            results={
                "corr_fail_123": _error(failing_call, "not found"),
            },
        )

        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher)
        await loop.run("go")

        assert "corr_fail_123" in captured_tool_call_ids


# ---------------------------------------------------------------------------
# Test: Correction with a different tool name
# ---------------------------------------------------------------------------


class TestCorrectionWithDifferentToolName:
    """LLM pivots to a different tool as the correction strategy."""

    @pytest.mark.asyncio
    async def test_pivot_from_failed_tool_to_alternative(self) -> None:
        """When lookup_test_spec fails, the LLM falls back to read_wiki
        as an alternative approach to gather information."""
        spec_call = ToolCall(
            call_id="spec_fail",
            tool_name="lookup_test_spec",
            arguments={"test_name": "perf_benchmarks"},
        )
        wiki_call = ToolCall(
            call_id="wiki_fallback",
            tool_name="read_wiki",
            arguments={"query": "performance benchmarks"},
        )

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            if call_number == 1:
                return (spec_call,)
            if call_number == 2:
                # Verify LLM sees the spec lookup failure
                errors = _find_error_tool_messages(messages)
                assert len(errors) >= 1
                assert "no spec found" in errors[0]["content"].lower()
                # Pivot to a different tool
                return (wiki_call,)
            return ()

        dispatcher = ScriptedDispatcher(
            results={
                "spec_fail": _error(
                    spec_call,
                    "No spec found for 'perf_benchmarks'. "
                    "The test catalog does not have this entry.",
                ),
                "wiki_fallback": _success(
                    wiki_call,
                    output=json.dumps({
                        "query": "performance benchmarks",
                        "results": ["perf_bench.md"],
                    }),
                ),
            },
        )

        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher)
        result = await loop.run("look up perf benchmarks")

        assert result.final_state is AgentLoopState.COMPLETE
        # Tool sequence: lookup_test_spec -> read_wiki (different tool)
        assert dispatcher.dispatched_tool_names == (
            "lookup_test_spec", "read_wiki",
        )


# ---------------------------------------------------------------------------
# Test: Correction preserves full history
# ---------------------------------------------------------------------------


class TestCorrectionPreservesFullHistory:
    """After failure and correction, the full history is available."""

    @pytest.mark.asyncio
    async def test_history_contains_failed_and_corrected_calls(self) -> None:
        """The final result history includes both the failed call, its
        error result, the corrected call, and its success result."""
        fail_call = ToolCall(
            call_id="hist_fail",
            tool_name="read_wiki",
            arguments={"slug": "wrong"},
        )
        fix_call = ToolCall(
            call_id="hist_fix",
            tool_name="read_wiki",
            arguments={"slug": "right"},
        )

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            if call_number == 1:
                return (fail_call,)
            if call_number == 2:
                return (fix_call,)
            return ()

        dispatcher = ScriptedDispatcher(
            results={
                "hist_fail": _error(fail_call, "page not found"),
                "hist_fix": _success(fix_call, output="correct page content"),
            },
        )

        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher)
        result = await loop.run("read docs")

        # History structure:
        # [0] system, [1] user,
        # [2] assistant (fail_call), [3] tool (error),
        # [4] assistant (fix_call), [5] tool (success)
        assert len(result.history) == 6

        # System and user messages
        assert result.history[0]["role"] == "system"
        assert result.history[1]["role"] == "user"

        # Failed call: assistant message with tool_calls
        assert result.history[2]["role"] == "assistant"
        assert result.history[2]["tool_calls"] is not None
        assert result.history[2]["tool_calls"][0]["function"]["name"] == "read_wiki"
        assert result.history[2]["tool_calls"][0]["id"] == "hist_fail"

        # Failed result: tool message with ERROR prefix
        assert result.history[3]["role"] == "tool"
        assert result.history[3]["tool_call_id"] == "hist_fail"
        assert result.history[3]["content"].startswith("ERROR:")
        assert "page not found" in result.history[3]["content"]

        # Corrected call: assistant message with tool_calls
        assert result.history[4]["role"] == "assistant"
        assert result.history[4]["tool_calls"][0]["id"] == "hist_fix"

        # Corrected result: tool message with success
        assert result.history[5]["role"] == "tool"
        assert result.history[5]["tool_call_id"] == "hist_fix"
        assert result.history[5]["content"] == "correct page content"

    @pytest.mark.asyncio
    async def test_history_grows_monotonically(self) -> None:
        """Each iteration adds messages; history length never decreases."""
        calls = [
            ToolCall(
                call_id=f"grow_{i}",
                tool_name="read_wiki",
                arguments={"slug": f"page_{i}"},
            )
            for i in range(3)
        ]

        history_lengths: list[int] = []

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            history_lengths.append(len(messages))
            if call_number <= 3:
                return (calls[call_number - 1],)
            return ()

        dispatcher = ScriptedDispatcher(
            results={
                "grow_0": _error(calls[0], "err_0"),
                # grow_1 and grow_2 use default success
            },
        )

        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher)
        await loop.run("go")

        # Each call should see strictly more messages
        for i in range(1, len(history_lengths)):
            assert history_lengths[i] > history_lengths[i - 1], (
                f"History should grow: iteration {i} saw "
                f"{history_lengths[i]} msgs <= {history_lengths[i-1]} prior"
            )


# ---------------------------------------------------------------------------
# Test: Correction within iteration budget
# ---------------------------------------------------------------------------


class TestFailureCorrectionWithinIterationBudget:
    """Correction cycles stay within the max_iterations budget."""

    @pytest.mark.asyncio
    async def test_correction_within_budget_completes(self) -> None:
        """Fail on iteration 1, correct on iteration 2, complete on 3.
        All within a max_iterations=5 budget."""
        fail_call = _make_call("lookup_test_spec", call_id="bud_fail")
        fix_call = _make_call("lookup_test_spec", call_id="bud_fix")

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            if call_number == 1:
                return (fail_call,)
            if call_number == 2:
                return (fix_call,)
            return ()

        dispatcher = ScriptedDispatcher(
            results={"bud_fail": _error(fail_call, "spec missing")},
        )

        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher, max_iterations=5)
        result = await loop.run("go")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 3
        assert result.iterations_used <= 5

    @pytest.mark.asyncio
    async def test_correction_exceeds_budget_terminates(self) -> None:
        """When the LLM keeps retrying but never succeeds, the loop
        terminates at max_iterations with ERROR."""

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            # Always return a tool call (never completes voluntarily)
            return (
                ToolCall(
                    call_id=f"eternal_{call_number}",
                    tool_name="read_wiki",
                    arguments={"slug": f"attempt_{call_number}"},
                ),
            )

        # All calls fail -- the LLM can never break the cycle
        class _AlwaysFailDispatcher:
            dispatched: list[ToolCall] = []

            async def dispatch(self, call: ToolCall) -> ToolResult:
                self.dispatched.append(call)
                return _error(call, f"still failing: {call.call_id}")

        dispatcher = _AlwaysFailDispatcher()
        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher, max_iterations=3)  # type: ignore[arg-type]
        result = await loop.run("go")

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 3
        assert "max iterations" in (result.error_message or "").lower()
        assert len(dispatcher.dispatched) == 3


# ---------------------------------------------------------------------------
# Test: Full cycle -- SSH failure -> observe -> correct -> re-execute
# ---------------------------------------------------------------------------


class TestSSHFailureCorrectionCycle:
    """Integration test modeling a realistic SSH failure-correction flow.

    Iteration 1: propose_ssh_command (approved)
    Iteration 2: execute_ssh (fails -- wrong path)
    Iteration 3: propose_ssh_command (corrected path, approved)
    Iteration 4: execute_ssh (succeeds)
    Iteration 5: LLM returns empty -> COMPLETE
    """

    @pytest.mark.asyncio
    async def test_ssh_failure_observe_correct_succeed(self) -> None:
        """Full cycle: SSH command fails, agent observes the failure,
        proposes a corrected command, and successfully re-executes."""

        propose_v1 = ToolCall(
            call_id="propose_1",
            tool_name="propose_ssh_command",
            arguments={
                "command": "python3 /opt/tests/run_suite.py",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "explanation": "Run the test suite",
            },
        )
        execute_v1 = ToolCall(
            call_id="exec_1",
            tool_name="execute_ssh",
            arguments={"approval_id": "apv-001"},
        )
        propose_v2 = ToolCall(
            call_id="propose_2",
            tool_name="propose_ssh_command",
            arguments={
                "command": "python3 /home/deploy/tests/run_suite.py",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "explanation": (
                    "Corrected path: the test suite is under /home/deploy/, "
                    "not /opt/tests/"
                ),
            },
        )
        execute_v2 = ToolCall(
            call_id="exec_2",
            tool_name="execute_ssh",
            arguments={"approval_id": "apv-002"},
        )

        def _respond(
            messages: tuple[dict[str, Any], ...],
            call_number: int,
        ) -> tuple[ToolCall, ...]:
            if call_number == 1:
                return (propose_v1,)
            if call_number == 2:
                return (execute_v1,)
            if call_number == 3:
                # Verify LLM sees the SSH error from execute_v1
                errors = _find_error_tool_messages(messages)
                assert len(errors) >= 1, (
                    "LLM must see the execute_ssh error before proposing "
                    "a correction"
                )
                error_content = errors[-1]["content"]
                assert "no such file" in error_content.lower(), (
                    f"Expected 'no such file' in error, got: {error_content}"
                )
                return (propose_v2,)
            if call_number == 4:
                return (execute_v2,)
            return ()

        dispatcher = ScriptedDispatcher(
            results={
                "propose_1": _success(
                    propose_v1,
                    output=json.dumps({
                        "approved": True,
                        "approval_id": "apv-001",
                        "command": "python3 /opt/tests/run_suite.py",
                        "edited": False,
                    }),
                ),
                "exec_1": _error(
                    execute_v1,
                    "Command failed with exit code 127: "
                    "python3: can't open file '/opt/tests/run_suite.py': "
                    "[Errno 2] No such file or directory",
                ),
                "propose_2": _success(
                    propose_v2,
                    output=json.dumps({
                        "approved": True,
                        "approval_id": "apv-002",
                        "command": "python3 /home/deploy/tests/run_suite.py",
                        "edited": False,
                    }),
                ),
                "exec_2": _success(
                    execute_v2,
                    output=json.dumps({
                        "success": True,
                        "exit_code": 0,
                        "stdout": "All 42 tests passed.",
                        "stderr": "",
                    }),
                ),
            },
        )

        llm = HistoryAwareLLM(_respond)
        loop = _make_loop(llm, dispatcher, max_iterations=6)
        result = await loop.run("run the test suite on staging")

        # Verify completion
        assert result.final_state is AgentLoopState.COMPLETE
        assert result.error_message is None
        assert result.iterations_used == 5  # 4 tool + 1 empty

        # Verify tool dispatch sequence
        assert dispatcher.dispatched_tool_names == (
            "propose_ssh_command",
            "execute_ssh",
            "propose_ssh_command",  # correction
            "execute_ssh",          # re-execution
        )

        # Verify two propose->execute pairs
        names = dispatcher.dispatched_tool_names
        propose_count = names.count("propose_ssh_command")
        execute_count = names.count("execute_ssh")
        assert propose_count == 2
        assert execute_count == 2

        # Verify the LLM was called 5 times
        assert llm.call_count == 5

        # Verify error is preserved in final history
        error_msgs = _find_error_tool_messages(result.history)
        assert len(error_msgs) == 1
        assert "No such file" in error_msgs[0]["content"]

        # Verify the success result is also in history
        success_tool_msgs = [
            m for m in result.history
            if m.get("role") == "tool"
            and m.get("tool_call_id") == "exec_2"
            and not m.get("content", "").startswith("ERROR:")
        ]
        assert len(success_tool_msgs) == 1
