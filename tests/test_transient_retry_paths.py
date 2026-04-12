"""Tests for the three transient error retry paths (AC 10, Sub-AC 4).

Validates the three distinct outcomes when the agent loop encounters
transient errors (network blips, LLM timeouts, rate limits) during the
THINKING phase:

  Path 1 -- Successful retry on first transient error:
      First LLM call fails with transient error, first retry succeeds.
      Loop completes normally with retry_exhausted=False.

  Path 2 -- Successful retry on second transient error:
      First and second LLM calls fail with transient errors, second
      retry (third overall call) succeeds. Loop completes normally
      with retry_exhausted=False.

  Path 3 -- Fallback to one-shot after 2 failed retries:
      All three LLM calls (1 original + 2 retries) fail with transient
      errors. Loop terminates with ERROR state, retry_exhausted=True,
      and the error message contains a fallback-to-one-shot hint.

Constraints verified:
  - max_retries=2 allows up to 2 additional attempts (3 total).
  - Transient errors are classified correctly by error_classification.
  - Exponential backoff delays are recorded (sleep_fn injection).
  - Conversation context (messages) is preserved identically across retries.
  - retry_exhausted flag is only True when all retries are exhausted.
  - Iteration count is accurate (retries within one iteration).
  - Multiple transient error types (ConnectionError, TimeoutError,
    OSError) all follow the same retry paths.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from jules_daemon.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopState,
)
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResult,
    ToolResultStatus,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class RecordingSleepFn:
    """Records all sleep calls for backoff delay assertions.

    Does not actually sleep, keeping tests fast and deterministic.
    """

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class TransientThenSuccessLLMClient:
    """Mock LLM client that fails N times with a transient error, then succeeds.

    After ``fail_count`` transient failures, returns responses from the
    ``success_responses`` queue. When the queue is empty, returns an
    empty tuple (signaling loop completion).
    """

    def __init__(
        self,
        fail_count: int,
        error: Exception,
        success_responses: list[tuple[ToolCall, ...]],
    ) -> None:
        self._fail_count = fail_count
        self._error = error
        self._success_responses = list(success_responses)
        self._call_count = 0
        self.captured_messages: list[tuple[dict[str, Any], ...]] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    async def get_tool_calls(
        self, messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self._call_count += 1
        self.captured_messages.append(messages)
        if self._call_count <= self._fail_count:
            raise self._error
        if self._success_responses:
            return self._success_responses.pop(0)
        return ()


class AlwaysFailingLLMClient:
    """Mock LLM client that always raises a transient error.

    Used to test the exhaustion path where all retries fail.
    """

    def __init__(self, error: Exception) -> None:
        self._error = error
        self._call_count = 0
        self.captured_messages: list[tuple[dict[str, Any], ...]] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    async def get_tool_calls(
        self, messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self._call_count += 1
        self.captured_messages.append(messages)
        raise self._error


class MockToolDispatcher:
    """Mock tool dispatcher that returns success results.

    Satisfies the ToolDispatcher protocol. Always returns SUCCESS
    for any dispatched tool call.
    """

    def __init__(self) -> None:
        self._executed: list[ToolCall] = []

    @property
    def executed_calls(self) -> tuple[ToolCall, ...]:
        return tuple(self._executed)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        self._executed.append(call)
        return ToolResult.success(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output=f"result for {call.tool_name}",
        )


def _make_config(
    max_retries: int = 2,
    retry_base_delay: float = 0.0,
    max_iterations: int = 5,
) -> AgentLoopConfig:
    """Build an AgentLoopConfig with test-friendly defaults."""
    return AgentLoopConfig(
        max_iterations=max_iterations,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )


def _make_loop(
    llm_client: Any,
    *,
    config: AgentLoopConfig | None = None,
    sleep_fn: Any | None = None,
    system_prompt: str = "You are a test runner assistant.",
) -> AgentLoop:
    """Build an AgentLoop with test-friendly defaults."""
    return AgentLoop(
        llm_client=llm_client,
        tool_dispatcher=MockToolDispatcher(),
        system_prompt=system_prompt,
        config=config or _make_config(),
        sleep_fn=sleep_fn or AsyncMock(),
    )


# ---------------------------------------------------------------------------
# Path 1: Successful retry on first transient error
# ---------------------------------------------------------------------------


class TestPath1SuccessOnFirstRetry:
    """Path 1: First LLM call fails, first retry succeeds.

    The agent loop encounters a transient error on the initial call
    within an iteration. The first retry (attempt 2) succeeds, and
    the loop completes normally.

    Expected behavior:
      - 2 total LLM calls (1 failure + 1 success)
      - Final state: COMPLETE
      - retry_exhausted: False
      - iterations_used: 1
      - No error message
    """

    @pytest.mark.asyncio
    async def test_connection_error_recovers_on_first_retry(self) -> None:
        """ConnectionError on attempt 1, success on attempt 2."""
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=ConnectionError("network blip"),
            success_responses=[()],  # empty = LLM signals completion
        )
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run the smoke tests on staging")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False
        assert result.error_message is None
        assert result.iterations_used == 1
        assert llm.call_count == 2  # 1 failure + 1 success

    @pytest.mark.asyncio
    async def test_timeout_error_recovers_on_first_retry(self) -> None:
        """TimeoutError on attempt 1, success on attempt 2."""
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=TimeoutError("LLM timed out"),
            success_responses=[()],
        )
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("check test status")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False
        assert result.error_message is None
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_os_error_recovers_on_first_retry(self) -> None:
        """OSError on attempt 1, success on attempt 2."""
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=OSError("ECONNRESET"),
            success_responses=[()],
        )
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run regression suite")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_first_retry_backoff_delay_correct(self) -> None:
        """First retry uses base_delay * 2^0 = base_delay."""
        recorder = RecordingSleepFn()
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=ConnectionError("blip"),
            success_responses=[()],
        )
        config = _make_config(max_retries=2, retry_base_delay=1.5)
        loop = _make_loop(llm, config=config, sleep_fn=recorder)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        # Only 1 retry -> 1 sleep call
        assert len(recorder.delays) == 1
        # Backoff: 1.5 * 2^0 = 1.5
        assert recorder.delays[0] == 1.5

    @pytest.mark.asyncio
    async def test_first_retry_preserves_context(self) -> None:
        """Same conversation context is sent on both attempts."""
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=ConnectionError("reset"),
            success_responses=[()],
        )
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(
            llm,
            config=config,
            system_prompt="test system prompt",
        )

        result = await loop.run("run smoke tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert len(llm.captured_messages) == 2

        # Both calls receive the exact same messages
        assert llm.captured_messages[0] == llm.captured_messages[1]

        # Verify the content is the initial system + user messages
        first_call_msgs = llm.captured_messages[0]
        assert first_call_msgs[0]["role"] == "system"
        assert first_call_msgs[0]["content"] == "test system prompt"
        assert first_call_msgs[1]["role"] == "user"
        assert first_call_msgs[1]["content"] == "run smoke tests"

    @pytest.mark.asyncio
    async def test_first_retry_success_continues_to_tool_calls(self) -> None:
        """After recovering on first retry, tool calls are dispatched normally."""
        tool_call = ToolCall(
            call_id="call-001",
            tool_name="read_wiki",
            arguments={"slug": "smoke-test-spec"},
        )
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=ConnectionError("blip"),
            success_responses=[
                (tool_call,),  # iteration 1 returns a tool call
                (),            # iteration 2 signals completion
            ],
        )
        dispatcher = MockToolDispatcher()
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=config,
            sleep_fn=AsyncMock(),
        )

        result = await loop.run("run smoke tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 2
        assert len(dispatcher.executed_calls) == 1
        assert dispatcher.executed_calls[0].tool_name == "read_wiki"


# ---------------------------------------------------------------------------
# Path 2: Successful retry on second transient error
# ---------------------------------------------------------------------------


class TestPath2SuccessOnSecondRetry:
    """Path 2: First and second LLM calls fail, second retry succeeds.

    The agent loop encounters transient errors on both the initial call
    and the first retry. The second retry (attempt 3) succeeds, and the
    loop completes normally.

    Expected behavior:
      - 3 total LLM calls (2 failures + 1 success)
      - Final state: COMPLETE
      - retry_exhausted: False
      - iterations_used: 1
      - No error message
    """

    @pytest.mark.asyncio
    async def test_connection_error_recovers_on_second_retry(self) -> None:
        """ConnectionError on attempts 1-2, success on attempt 3."""
        llm = TransientThenSuccessLLMClient(
            fail_count=2,
            error=ConnectionError("persistent network issue"),
            success_responses=[()],
        )
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run all integration tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False
        assert result.error_message is None
        assert result.iterations_used == 1
        assert llm.call_count == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_timeout_error_recovers_on_second_retry(self) -> None:
        """TimeoutError on attempts 1-2, success on attempt 3."""
        llm = TransientThenSuccessLLMClient(
            fail_count=2,
            error=TimeoutError("LLM response timeout"),
            success_responses=[()],
        )
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("execute regression suite")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False
        assert result.error_message is None
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_os_error_recovers_on_second_retry(self) -> None:
        """OSError on attempts 1-2, success on attempt 3."""
        llm = TransientThenSuccessLLMClient(
            fail_count=2,
            error=OSError("ECONNREFUSED"),
            success_responses=[()],
        )
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("check test environment")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_second_retry_backoff_delays_correct(self) -> None:
        """Two retries produce exponential backoff: base*2^0, base*2^1."""
        recorder = RecordingSleepFn()
        llm = TransientThenSuccessLLMClient(
            fail_count=2,
            error=ConnectionError("down"),
            success_responses=[()],
        )
        config = _make_config(max_retries=2, retry_base_delay=1.0)
        loop = _make_loop(llm, config=config, sleep_fn=recorder)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        # 2 retries -> 2 sleep calls
        assert len(recorder.delays) == 2
        # First retry: 1.0 * 2^0 = 1.0
        assert recorder.delays[0] == 1.0
        # Second retry: 1.0 * 2^1 = 2.0
        assert recorder.delays[1] == 2.0

    @pytest.mark.asyncio
    async def test_second_retry_preserves_context(self) -> None:
        """Same conversation context is sent on all three attempts."""
        llm = TransientThenSuccessLLMClient(
            fail_count=2,
            error=TimeoutError("slow"),
            success_responses=[()],
        )
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(
            llm,
            config=config,
            system_prompt="system prompt for retry test",
        )

        result = await loop.run("deploy and test")

        assert result.final_state is AgentLoopState.COMPLETE
        assert len(llm.captured_messages) == 3

        # All three calls receive the exact same messages
        assert llm.captured_messages[0] == llm.captured_messages[1]
        assert llm.captured_messages[1] == llm.captured_messages[2]

        # Verify content
        first_call_msgs = llm.captured_messages[0]
        assert first_call_msgs[0]["role"] == "system"
        assert first_call_msgs[0]["content"] == "system prompt for retry test"
        assert first_call_msgs[1]["role"] == "user"
        assert first_call_msgs[1]["content"] == "deploy and test"

    @pytest.mark.asyncio
    async def test_second_retry_success_continues_to_tool_calls(self) -> None:
        """After recovering on second retry, tool calls are dispatched."""
        tool_call = ToolCall(
            call_id="call-002",
            tool_name="lookup_test_spec",
            arguments={"test_name": "smoke"},
        )
        llm = TransientThenSuccessLLMClient(
            fail_count=2,
            error=ConnectionError("network flap"),
            success_responses=[
                (tool_call,),  # iteration 1 returns a tool call
                (),            # iteration 2 signals completion
            ],
        )
        dispatcher = MockToolDispatcher()
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=config,
            sleep_fn=AsyncMock(),
        )

        result = await loop.run("look up smoke test spec")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 2
        assert len(dispatcher.executed_calls) == 1
        assert dispatcher.executed_calls[0].tool_name == "lookup_test_spec"

    @pytest.mark.asyncio
    async def test_exactly_at_retry_boundary(self) -> None:
        """Success on the very last allowed attempt (attempt = max_retries + 1).

        With max_retries=2, the loop makes up to 3 attempts total.
        Failing on attempts 1 and 2, succeeding on attempt 3 is the
        boundary case -- one more failure would exhaust retries.
        """
        llm = TransientThenSuccessLLMClient(
            fail_count=2,
            error=ConnectionError("boundary test"),
            success_responses=[()],
        )
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("boundary test command")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False
        assert llm.call_count == 3  # exactly at the boundary


# ---------------------------------------------------------------------------
# Path 3: Fallback to one-shot after 2 failed retries
# ---------------------------------------------------------------------------


class TestPath3FallbackAfterRetryExhaustion:
    """Path 3: All retry attempts fail, loop falls back to one-shot.

    The agent loop encounters transient errors on all attempts (1 original
    + 2 retries = 3 total calls). The loop terminates with ERROR state
    and sets retry_exhausted=True, signaling to the caller that it should
    fall back to the one-shot LLM translation path.

    Expected behavior:
      - 3 total LLM calls (1 original + 2 retries)
      - Final state: ERROR
      - retry_exhausted: True
      - error_message contains "falling back to one-shot"
      - iterations_used: 1 (failure happens within iteration 1)
    """

    @pytest.mark.asyncio
    async def test_connection_error_exhausts_retries(self) -> None:
        """Persistent ConnectionError exhausts all retries."""
        llm = AlwaysFailingLLMClient(ConnectionError("persistent failure"))
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run the smoke tests on staging")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        assert result.error_message is not None
        assert "falling back to one-shot" in result.error_message.lower()
        assert result.iterations_used == 1
        assert llm.call_count == 3  # 1 original + 2 retries

    @pytest.mark.asyncio
    async def test_timeout_error_exhausts_retries(self) -> None:
        """Persistent TimeoutError exhausts all retries."""
        llm = AlwaysFailingLLMClient(TimeoutError("LLM always times out"))
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("execute full test suite")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        assert "falling back to one-shot" in result.error_message.lower()
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_os_error_exhausts_retries(self) -> None:
        """Persistent OSError exhausts all retries."""
        llm = AlwaysFailingLLMClient(OSError("ECONNRESET persistent"))
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("verify deployment")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_exhaustion_backoff_delays_correct(self) -> None:
        """All retry backoff delays are recorded before exhaustion."""
        recorder = RecordingSleepFn()
        llm = AlwaysFailingLLMClient(ConnectionError("down"))
        config = _make_config(max_retries=2, retry_base_delay=2.0)
        loop = _make_loop(llm, config=config, sleep_fn=recorder)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        # 2 retries -> 2 sleep calls
        assert len(recorder.delays) == 2
        # First retry: 2.0 * 2^0 = 2.0
        assert recorder.delays[0] == 2.0
        # Second retry: 2.0 * 2^1 = 4.0
        assert recorder.delays[1] == 4.0

    @pytest.mark.asyncio
    async def test_exhaustion_preserves_context(self) -> None:
        """Same conversation context is sent on all three failed attempts."""
        llm = AlwaysFailingLLMClient(ConnectionError("unstable"))
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(
            llm,
            config=config,
            system_prompt="system for exhaustion test",
        )

        result = await loop.run("test that fails")

        assert result.final_state is AgentLoopState.ERROR
        assert len(llm.captured_messages) == 3

        # All three calls receive the exact same messages
        assert llm.captured_messages[0] == llm.captured_messages[1]
        assert llm.captured_messages[1] == llm.captured_messages[2]

        first_call_msgs = llm.captured_messages[0]
        assert first_call_msgs[0]["role"] == "system"
        assert first_call_msgs[0]["content"] == "system for exhaustion test"
        assert first_call_msgs[1]["role"] == "user"
        assert first_call_msgs[1]["content"] == "test that fails"

    @pytest.mark.asyncio
    async def test_exhaustion_error_message_contains_category(self) -> None:
        """Error message includes the error classification category."""
        llm = AlwaysFailingLLMClient(ConnectionError("network down"))
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run tests")

        assert result.error_message is not None
        # The error message should reference the transient error
        assert "transient" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_exhaustion_history_has_only_initial_messages(self) -> None:
        """After exhaustion, history contains only system + user messages.

        Since the loop fails in the THINKING phase of iteration 1,
        no tool calls or results are appended to history.
        """
        llm = AlwaysFailingLLMClient(ConnectionError("down"))
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run smoke tests")

        assert result.final_state is AgentLoopState.ERROR
        assert len(result.history) == 2  # system + user only
        assert result.history[0]["role"] == "system"
        assert result.history[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_exhaustion_no_tool_calls_dispatched(self) -> None:
        """No tool calls are dispatched when retries are exhausted.

        The ACTING phase is never reached because the THINKING phase
        fails on all attempts.
        """
        llm = AlwaysFailingLLMClient(TimeoutError("always times out"))
        dispatcher = MockToolDispatcher()
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=config,
            sleep_fn=AsyncMock(),
        )

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        assert len(dispatcher.executed_calls) == 0


# ---------------------------------------------------------------------------
# Cross-path comparisons and edge cases
# ---------------------------------------------------------------------------


class TestRetryPathEdgeCases:
    """Edge cases and cross-cutting concerns for all three retry paths."""

    @pytest.mark.asyncio
    async def test_zero_base_delay_skips_sleep_on_all_paths(self) -> None:
        """Zero base delay means sleep is never called (delay=0 is skipped)."""
        recorder = RecordingSleepFn()

        # Path 3: all retries fail
        llm = AlwaysFailingLLMClient(ConnectionError("down"))
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config, sleep_fn=recorder)

        await loop.run("run tests")

        # Zero delay -> no sleep calls
        assert len(recorder.delays) == 0

    @pytest.mark.asyncio
    async def test_max_retries_one_allows_single_retry(self) -> None:
        """With max_retries=1, only one retry is allowed (2 total calls).

        Path 1 analog with max_retries=1: fail once, succeed on retry.
        """
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=ConnectionError("blip"),
            success_responses=[()],
        )
        config = _make_config(max_retries=1, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_one_exhaustion_after_two_calls(self) -> None:
        """With max_retries=1, exhaustion happens after 2 total calls.

        Path 3 analog with max_retries=1: both calls fail.
        """
        llm = AlwaysFailingLLMClient(ConnectionError("persistent"))
        config = _make_config(max_retries=1, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        assert llm.call_count == 2  # 1 original + 1 retry

    @pytest.mark.asyncio
    async def test_zero_retries_exhausts_immediately(self) -> None:
        """With max_retries=0, any transient error exhausts immediately.

        No retries means the very first transient error triggers exhaustion.
        """
        llm = AlwaysFailingLLMClient(ConnectionError("instant fail"))
        config = _make_config(max_retries=0, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        assert llm.call_count == 1  # only the original call

    @pytest.mark.asyncio
    async def test_retry_in_later_iteration_path1(self) -> None:
        """Retry path 1 can occur in iteration > 1.

        First iteration succeeds, second iteration fails once then succeeds.
        """
        call_count = 0

        class SecondIterationRetryLLM:
            """Succeeds on iter 1, fails once then succeeds on iter 2."""

            def __init__(self) -> None:
                self._phase = "iter1"
                self.total_calls = 0

            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                nonlocal call_count
                call_count += 1
                self.total_calls += 1

                if self._phase == "iter1":
                    self._phase = "iter2_fail"
                    return (
                        ToolCall(
                            call_id="call-1",
                            tool_name="read_wiki",
                            arguments={"slug": "test"},
                        ),
                    )
                if self._phase == "iter2_fail":
                    self._phase = "iter2_succeed"
                    raise ConnectionError("transient in iter 2")
                # iter2_succeed: complete
                return ()

        llm = SecondIterationRetryLLM()
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False
        assert result.iterations_used == 2

    @pytest.mark.asyncio
    async def test_retry_exhaustion_in_later_iteration(self) -> None:
        """Retry exhaustion (path 3) can occur in iteration > 1.

        First iteration succeeds, second iteration exhausts all retries.
        """

        class FirstSucceedThenExhaust:
            """Succeeds on iter 1, then fails all retries on iter 2."""

            def __init__(self) -> None:
                self.total_calls = 0
                self._iter1_done = False

            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                self.total_calls += 1
                if not self._iter1_done:
                    self._iter1_done = True
                    return (
                        ToolCall(
                            call_id="call-1",
                            tool_name="read_wiki",
                            arguments={"slug": "test"},
                        ),
                    )
                raise ConnectionError("persistent failure in iter 2")

        llm = FirstSucceedThenExhaust()
        config = _make_config(
            max_retries=2,
            retry_base_delay=0.0,
            max_iterations=5,
        )
        loop = _make_loop(llm, config=config)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        assert result.iterations_used == 2  # Failed at iteration 2
        # iter1: 1 call, iter2: 1 original + 2 retries = 3 calls
        assert llm.total_calls == 4

    @pytest.mark.asyncio
    async def test_different_transient_errors_all_retry(self) -> None:
        """Mixed transient error types all trigger the same retry behavior."""

        call_count = 0

        class MixedTransientLLM:
            """Raises different transient errors then succeeds."""

            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("network issue")
                if call_count == 2:
                    raise TimeoutError("LLM timeout")
                return ()  # Third call succeeds

        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(MixedTransientLLM(), config=config)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_permanent_error_does_not_set_retry_exhausted(self) -> None:
        """Permanent errors terminate immediately without setting retry_exhausted.

        This confirms that retry_exhausted is only True when transient
        retries are exhausted -- not for permanent errors.
        """
        llm = AlwaysFailingLLMClient(ValueError("malformed JSON response"))
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is False
        assert "permanent" in result.error_message.lower()
        assert llm.call_count == 1  # no retries for permanent errors

    @pytest.mark.asyncio
    async def test_loop_state_is_error_after_exhaustion(self) -> None:
        """Agent loop state machine is in ERROR after retry exhaustion."""
        llm = AlwaysFailingLLMClient(ConnectionError("down"))
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run tests")

        assert loop.state is AgentLoopState.ERROR
        assert result.final_state is AgentLoopState.ERROR

    @pytest.mark.asyncio
    async def test_loop_state_is_complete_after_successful_retry(self) -> None:
        """Agent loop state machine is in COMPLETE after successful retry."""
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=ConnectionError("blip"),
            success_responses=[()],
        )
        config = _make_config(max_retries=2, retry_base_delay=0.0)
        loop = _make_loop(llm, config=config)

        result = await loop.run("run tests")

        assert loop.state is AgentLoopState.COMPLETE
        assert result.final_state is AgentLoopState.COMPLETE
