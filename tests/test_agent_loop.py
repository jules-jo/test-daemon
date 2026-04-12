"""Tests for the AgentLoop state machine skeleton.

Validates:
    - AgentLoopState enum members and their terminal classification
    - AgentLoopConfig immutability and validation
    - AgentLoop orchestrates think-act-observe cycles
    - Max-iteration guard terminates the loop
    - LLM returning no tool calls triggers COMPLETE
    - User denial (DENIED ToolResult) triggers immediate ERROR
    - Transient LLM errors retry up to configured count, then fall back
    - Permanent errors terminate immediately
    - Multiple tool calls in a single cycle
    - History accumulation across cycles
    - Protocol abstractions (LLMClient, ToolDispatcher) work with mocks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from jules_daemon.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopResult,
    AgentLoopState,
    AgentLoopError,
    LLMClient,
    ToolDispatcher,
)
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResult,
    ToolResultStatus,
)


# ---------------------------------------------------------------------------
# AgentLoopState enum
# ---------------------------------------------------------------------------


class TestAgentLoopState:
    """Tests for the AgentLoopState enum."""

    def test_all_members_exist(self) -> None:
        members = {s.name for s in AgentLoopState}
        assert members == {"THINKING", "ACTING", "OBSERVING", "COMPLETE", "ERROR"}

    def test_thinking_not_terminal(self) -> None:
        assert not AgentLoopState.THINKING.is_terminal

    def test_acting_not_terminal(self) -> None:
        assert not AgentLoopState.ACTING.is_terminal

    def test_observing_not_terminal(self) -> None:
        assert not AgentLoopState.OBSERVING.is_terminal

    def test_complete_is_terminal(self) -> None:
        assert AgentLoopState.COMPLETE.is_terminal

    def test_error_is_terminal(self) -> None:
        assert AgentLoopState.ERROR.is_terminal

    def test_string_values(self) -> None:
        assert AgentLoopState.THINKING.value == "thinking"
        assert AgentLoopState.ACTING.value == "acting"
        assert AgentLoopState.OBSERVING.value == "observing"
        assert AgentLoopState.COMPLETE.value == "complete"
        assert AgentLoopState.ERROR.value == "error"


# ---------------------------------------------------------------------------
# AgentLoopConfig
# ---------------------------------------------------------------------------


class TestAgentLoopConfig:
    """Tests for the AgentLoopConfig frozen dataclass."""

    def test_default_values(self) -> None:
        config = AgentLoopConfig()
        assert config.max_iterations == 5
        assert config.max_retries == 2

    def test_custom_values(self) -> None:
        config = AgentLoopConfig(max_iterations=10, max_retries=3)
        assert config.max_iterations == 10
        assert config.max_retries == 3

    def test_frozen(self) -> None:
        config = AgentLoopConfig()
        with pytest.raises(AttributeError):
            config.max_iterations = 10  # type: ignore[misc]

    def test_max_iterations_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            AgentLoopConfig(max_iterations=0)

    def test_max_retries_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            AgentLoopConfig(max_retries=-1)


# ---------------------------------------------------------------------------
# AgentLoopResult
# ---------------------------------------------------------------------------


class TestAgentLoopResult:
    """Tests for the AgentLoopResult frozen dataclass."""

    def test_success_result(self) -> None:
        result = AgentLoopResult(
            final_state=AgentLoopState.COMPLETE,
            iterations_used=3,
            history=(),
            error_message=None,
        )
        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 3
        assert result.error_message is None

    def test_error_result(self) -> None:
        result = AgentLoopResult(
            final_state=AgentLoopState.ERROR,
            iterations_used=1,
            history=(),
            error_message="User denied the operation",
        )
        assert result.final_state is AgentLoopState.ERROR
        assert result.error_message == "User denied the operation"

    def test_frozen(self) -> None:
        result = AgentLoopResult(
            final_state=AgentLoopState.COMPLETE,
            iterations_used=1,
            history=(),
            error_message=None,
        )
        with pytest.raises(AttributeError):
            result.iterations_used = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_tool_calls(
    *names: str,
    base_id: str = "call",
) -> tuple[ToolCall, ...]:
    """Create a tuple of ToolCalls with unique IDs."""
    return tuple(
        ToolCall(
            call_id=f"{base_id}_{i}",
            tool_name=name,
            arguments={"arg": f"val_{i}"},
        )
        for i, name in enumerate(names)
    )


def _make_success_result(call: ToolCall) -> ToolResult:
    """Create a success ToolResult for a given ToolCall."""
    return ToolResult.success(
        call_id=call.call_id,
        tool_name=call.tool_name,
        output=f"output for {call.tool_name}",
    )


def _make_error_result(call: ToolCall, message: str = "failed") -> ToolResult:
    """Create an error ToolResult for a given ToolCall."""
    return ToolResult.error(
        call_id=call.call_id,
        tool_name=call.tool_name,
        error_message=message,
    )


def _make_denied_result(call: ToolCall) -> ToolResult:
    """Create a denied ToolResult for a given ToolCall."""
    return ToolResult.denied(
        call_id=call.call_id,
        tool_name=call.tool_name,
        error_message="User denied the operation",
    )


class MockLLMClient:
    """Mock LLM client that returns preconfigured tool call sequences.

    Satisfies the LLMClient protocol. Each call to get_tool_calls()
    pops the next response from the queue. If the queue is empty,
    returns an empty tuple (signaling loop completion).
    """

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


class MockToolDispatcher:
    """Mock tool dispatcher that returns preconfigured results.

    Satisfies the ToolDispatcher protocol. Results are looked up by
    call_id. If no result is configured, returns a success result.
    """

    def __init__(
        self,
        results: dict[str, ToolResult] | None = None,
    ) -> None:
        self._results = dict(results) if results else {}
        self._executed: list[ToolCall] = []

    @property
    def executed_calls(self) -> tuple[ToolCall, ...]:
        return tuple(self._executed)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        self._executed.append(call)
        if call.call_id in self._results:
            return self._results[call.call_id]
        return _make_success_result(call)


class FailingLLMClient:
    """Mock LLM client that raises on every call."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def get_tool_calls(
        self, messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self._call_count += 1
        raise self._error


class TransientThenSuccessLLMClient:
    """Mock LLM client that fails N times then succeeds."""

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

    @property
    def call_count(self) -> int:
        return self._call_count

    async def get_tool_calls(
        self, messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise self._error
        if self._success_responses:
            return self._success_responses.pop(0)
        return ()


# ---------------------------------------------------------------------------
# AgentLoop -- basic lifecycle
# ---------------------------------------------------------------------------


class TestAgentLoopBasicLifecycle:
    """Tests for the basic think-act-observe loop lifecycle."""

    @pytest.mark.asyncio
    async def test_single_cycle_completes(self) -> None:
        """LLM returns tool calls once, then no calls -> COMPLETE."""
        calls = _make_tool_calls("read_wiki")
        llm = MockLLMClient([calls, ()])  # first cycle: 1 call, second: none
        dispatcher = MockToolDispatcher()

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 2
        assert result.error_message is None
        assert len(dispatcher.executed_calls) == 1

    @pytest.mark.asyncio
    async def test_immediate_complete_when_no_tool_calls(self) -> None:
        """LLM returns no tool calls on first iteration -> COMPLETE."""
        llm = MockLLMClient([()])
        dispatcher = MockToolDispatcher()

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        result = await loop.run("hello")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 1
        assert len(dispatcher.executed_calls) == 0

    @pytest.mark.asyncio
    async def test_multiple_cycles(self) -> None:
        """LLM returns tool calls over 3 cycles, then stops."""
        calls_1 = _make_tool_calls("read_wiki", base_id="c1")
        calls_2 = _make_tool_calls("lookup_test_spec", base_id="c2")
        calls_3 = _make_tool_calls("propose_ssh_command", base_id="c3")

        llm = MockLLMClient([calls_1, calls_2, calls_3, ()])
        dispatcher = MockToolDispatcher()

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 4
        assert len(dispatcher.executed_calls) == 3

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_per_cycle(self) -> None:
        """Multiple tool calls in a single LLM response are all dispatched."""
        calls = _make_tool_calls("read_wiki", "lookup_test_spec", base_id="multi")
        llm = MockLLMClient([calls, ()])
        dispatcher = MockToolDispatcher()

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert len(dispatcher.executed_calls) == 2


# ---------------------------------------------------------------------------
# AgentLoop -- max iterations guard
# ---------------------------------------------------------------------------


class TestAgentLoopMaxIterations:
    """Tests for the max-iteration termination guard."""

    @pytest.mark.asyncio
    async def test_terminates_at_max_iterations(self) -> None:
        """Loop stops at max_iterations even if LLM keeps returning calls."""
        # Create calls that would keep going forever
        infinite_calls = [
            _make_tool_calls("read_wiki", base_id=f"iter{i}")
            for i in range(20)
        ]
        llm = MockLLMClient(infinite_calls)
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_iterations=3)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 3
        assert result.error_message is not None
        assert "max iterations" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_default_max_iterations_is_five(self) -> None:
        """Default config uses max_iterations=5."""
        infinite_calls = [
            _make_tool_calls("read_wiki", base_id=f"iter{i}")
            for i in range(20)
        ]
        llm = MockLLMClient(infinite_calls)
        dispatcher = MockToolDispatcher()

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        result = await loop.run("run tests")

        assert result.iterations_used == 5
        assert result.final_state is AgentLoopState.ERROR


# ---------------------------------------------------------------------------
# AgentLoop -- termination on denial
# ---------------------------------------------------------------------------


class TestAgentLoopDenial:
    """Tests for termination on user denial."""

    @pytest.mark.asyncio
    async def test_denied_result_terminates_immediately(self) -> None:
        """A DENIED ToolResult should terminate the loop with ERROR state."""
        calls = _make_tool_calls("propose_ssh_command", base_id="deny")
        denied = _make_denied_result(calls[0])

        llm = MockLLMClient([calls])
        dispatcher = MockToolDispatcher(results={calls[0].call_id: denied})

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 1
        assert "denied" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_denial_among_multiple_calls_terminates(self) -> None:
        """If one of multiple calls is denied, loop terminates after that cycle."""
        call_ok = ToolCall(call_id="ok_1", tool_name="read_wiki", arguments={"arg": "v"})
        call_deny = ToolCall(call_id="deny_1", tool_name="execute_ssh", arguments={"arg": "v"})

        denied = _make_denied_result(call_deny)
        llm = MockLLMClient([(call_ok, call_deny)])
        dispatcher = MockToolDispatcher(results={call_deny.call_id: denied})

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert "denied" in result.error_message.lower()


# ---------------------------------------------------------------------------
# AgentLoop -- error observation (self-correction)
# ---------------------------------------------------------------------------


class TestAgentLoopSelfCorrection:
    """Tests for observing errors and continuing the loop."""

    @pytest.mark.asyncio
    async def test_tool_error_does_not_terminate(self) -> None:
        """A tool ERROR (non-terminal) should let the loop continue."""
        calls_1 = _make_tool_calls("read_wiki", base_id="err")
        error_result = _make_error_result(calls_1[0], "file not found")

        calls_2 = _make_tool_calls("read_wiki", base_id="fix")

        llm = MockLLMClient([calls_1, calls_2, ()])
        dispatcher = MockToolDispatcher(
            results={calls_1[0].call_id: error_result}
        )

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 3
        # Both calls were dispatched
        assert len(dispatcher.executed_calls) == 2


# ---------------------------------------------------------------------------
# AgentLoop -- transient LLM error retry
# ---------------------------------------------------------------------------


class TestAgentLoopTransientRetry:
    """Tests for transient LLM error retries."""

    @pytest.mark.asyncio
    async def test_transient_error_retries_succeed(self) -> None:
        """Transient error retries within limit, then succeeds."""
        llm = TransientThenSuccessLLMClient(
            fail_count=2,
            error=ConnectionError("network blip"),
            success_responses=[()],  # completes after retries
        )
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        # 2 failures + 1 success = 3 calls, but only 1 iteration counted
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_transient_error_exceeds_retries_falls_back(self) -> None:
        """Transient error exhausts retries -> ERROR with fallback hint."""
        llm = FailingLLMClient(ConnectionError("persistent failure"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.error_message is not None
        assert "retries" in result.error_message.lower() or "falling back" in result.error_message.lower()
        # 1 original + 2 retries = 3 total calls
        assert llm.call_count == 3


# ---------------------------------------------------------------------------
# AgentLoop -- permanent error termination
# ---------------------------------------------------------------------------


class TestAgentLoopPermanentError:
    """Tests for permanent error immediate termination."""

    @pytest.mark.asyncio
    async def test_permanent_error_terminates_immediately(self) -> None:
        """A permanent LLM error (e.g., ValueError) terminates without retry."""
        llm = FailingLLMClient(ValueError("malformed response"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.error_message is not None
        assert "malformed" in result.error_message.lower()
        # No retries for permanent errors
        assert llm.call_count == 1


# ---------------------------------------------------------------------------
# AgentLoop -- history accumulation
# ---------------------------------------------------------------------------


class TestAgentLoopHistory:
    """Tests for conversation history accumulation."""

    @pytest.mark.asyncio
    async def test_history_contains_system_and_user_messages(self) -> None:
        """Initial history has system prompt and user message."""
        llm = MockLLMClient([()])
        dispatcher = MockToolDispatcher()

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="You are a test runner",
        )
        result = await loop.run("run smoke tests")

        # Result history should have system + user messages
        assert len(result.history) >= 2
        assert result.history[0]["role"] == "system"
        assert result.history[0]["content"] == "You are a test runner"
        assert result.history[1]["role"] == "user"
        assert result.history[1]["content"] == "run smoke tests"

    @pytest.mark.asyncio
    async def test_history_grows_with_tool_calls_and_results(self) -> None:
        """History should include assistant tool calls and tool results."""
        calls = _make_tool_calls("read_wiki")
        llm = MockLLMClient([calls, ()])
        dispatcher = MockToolDispatcher()

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        result = await loop.run("run tests")

        # system + user + assistant(tool_calls) + tool(result) + ...
        assert len(result.history) > 2
        # Check we have assistant and tool role messages
        roles = [m["role"] for m in result.history]
        assert "assistant" in roles
        assert "tool" in roles


# ---------------------------------------------------------------------------
# AgentLoop -- state tracking
# ---------------------------------------------------------------------------


class TestAgentLoopStateTracking:
    """Tests for state transitions during the loop."""

    @pytest.mark.asyncio
    async def test_initial_state_is_thinking(self) -> None:
        """Before run(), the loop should be in THINKING state."""
        llm = MockLLMClient([()])
        dispatcher = MockToolDispatcher()

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        assert loop.state is AgentLoopState.THINKING

    @pytest.mark.asyncio
    async def test_final_state_after_complete(self) -> None:
        """After a successful run, state should be COMPLETE."""
        llm = MockLLMClient([()])
        dispatcher = MockToolDispatcher()

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        await loop.run("hello")
        assert loop.state is AgentLoopState.COMPLETE

    @pytest.mark.asyncio
    async def test_final_state_after_error(self) -> None:
        """After an error run, state should be ERROR."""
        llm = FailingLLMClient(ValueError("bad"))
        dispatcher = MockToolDispatcher()

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        await loop.run("hello")
        assert loop.state is AgentLoopState.ERROR


# ---------------------------------------------------------------------------
# AgentLoop -- protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Tests verifying that mock implementations satisfy the protocols."""

    def test_mock_llm_client_satisfies_protocol(self) -> None:
        client = MockLLMClient([])
        assert isinstance(client, LLMClient)

    def test_mock_dispatcher_satisfies_protocol(self) -> None:
        dispatcher = MockToolDispatcher()
        assert isinstance(dispatcher, ToolDispatcher)


# ---------------------------------------------------------------------------
# AgentLoopError
# ---------------------------------------------------------------------------


class TestAgentLoopError:
    """Tests for the AgentLoopError exception."""

    def test_basic_creation(self) -> None:
        err = AgentLoopError("something went wrong")
        assert str(err) == "something went wrong"

    def test_inherits_from_exception(self) -> None:
        err = AgentLoopError("test")
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# _compute_backoff_delay
# ---------------------------------------------------------------------------


class TestComputeBackoffDelay:
    """Tests for the exponential backoff delay calculation."""

    def test_first_retry_uses_base_delay(self) -> None:
        """Attempt 0 (first retry) -> base_delay * 2^0 = base_delay."""
        from jules_daemon.agent.agent_loop import _compute_backoff_delay

        assert _compute_backoff_delay(0, 1.0) == 1.0

    def test_second_retry_doubles(self) -> None:
        """Attempt 1 (second retry) -> base_delay * 2^1 = 2 * base_delay."""
        from jules_daemon.agent.agent_loop import _compute_backoff_delay

        assert _compute_backoff_delay(1, 1.0) == 2.0

    def test_third_retry_quadruples(self) -> None:
        """Attempt 2 (third retry) -> base_delay * 2^2 = 4 * base_delay."""
        from jules_daemon.agent.agent_loop import _compute_backoff_delay

        assert _compute_backoff_delay(2, 1.0) == 4.0

    def test_custom_base_delay(self) -> None:
        """Custom base delay scales proportionally."""
        from jules_daemon.agent.agent_loop import _compute_backoff_delay

        assert _compute_backoff_delay(0, 0.5) == 0.5
        assert _compute_backoff_delay(1, 0.5) == 1.0
        assert _compute_backoff_delay(2, 0.5) == 2.0

    def test_zero_base_delay_always_zero(self) -> None:
        """Zero base delay produces zero delay for all attempts."""
        from jules_daemon.agent.agent_loop import _compute_backoff_delay

        assert _compute_backoff_delay(0, 0.0) == 0.0
        assert _compute_backoff_delay(1, 0.0) == 0.0
        assert _compute_backoff_delay(5, 0.0) == 0.0


# ---------------------------------------------------------------------------
# AgentLoopConfig -- retry_base_delay
# ---------------------------------------------------------------------------


class TestAgentLoopConfigRetryBaseDelay:
    """Tests for the retry_base_delay field on AgentLoopConfig."""

    def test_default_retry_base_delay(self) -> None:
        config = AgentLoopConfig()
        assert config.retry_base_delay == 1.0

    def test_custom_retry_base_delay(self) -> None:
        config = AgentLoopConfig(retry_base_delay=2.5)
        assert config.retry_base_delay == 2.5

    def test_zero_retry_base_delay(self) -> None:
        """Zero is valid -- disables backoff."""
        config = AgentLoopConfig(retry_base_delay=0.0)
        assert config.retry_base_delay == 0.0

    def test_negative_retry_base_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="retry_base_delay"):
            AgentLoopConfig(retry_base_delay=-0.1)

    def test_frozen_retry_base_delay(self) -> None:
        config = AgentLoopConfig()
        with pytest.raises(AttributeError):
            config.retry_base_delay = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AgentLoop -- exponential backoff on transient retries
# ---------------------------------------------------------------------------


class RecordingSleepFn:
    """Records all sleep calls for assertion in tests.

    Does not actually sleep, so tests run instantly.
    """

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class TestAgentLoopExponentialBackoff:
    """Tests for exponential backoff behavior during transient error retries."""

    @pytest.mark.asyncio
    async def test_backoff_delays_are_exponential(self) -> None:
        """Sleep delays follow base_delay * 2^attempt pattern."""
        recorder = RecordingSleepFn()
        llm = FailingLLMClient(ConnectionError("network down"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=1.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
            sleep_fn=recorder,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        # 2 retries -> 2 sleep calls
        assert len(recorder.delays) == 2
        # First retry: 1.0 * 2^0 = 1.0
        assert recorder.delays[0] == 1.0
        # Second retry: 1.0 * 2^1 = 2.0
        assert recorder.delays[1] == 2.0

    @pytest.mark.asyncio
    async def test_custom_base_delay_scales_backoff(self) -> None:
        """Custom base delay is used for exponential backoff."""
        recorder = RecordingSleepFn()
        llm = FailingLLMClient(ConnectionError("timeout"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=0.5)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
            sleep_fn=recorder,
        )
        await loop.run("run tests")

        assert recorder.delays == [0.5, 1.0]

    @pytest.mark.asyncio
    async def test_zero_base_delay_skips_sleep(self) -> None:
        """Zero base delay means no sleep calls are made."""
        recorder = RecordingSleepFn()
        llm = FailingLLMClient(TimeoutError("timeout"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=0.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
            sleep_fn=recorder,
        )
        await loop.run("run tests")

        # Zero delay should not trigger sleep at all
        assert len(recorder.delays) == 0

    @pytest.mark.asyncio
    async def test_successful_retry_has_correct_backoff(self) -> None:
        """Retry that succeeds after 1 failure has a single backoff delay."""
        recorder = RecordingSleepFn()
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=ConnectionError("blip"),
            success_responses=[()],
        )
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=1.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
            sleep_fn=recorder,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        # Only 1 retry needed -> 1 sleep call
        assert len(recorder.delays) == 1
        assert recorder.delays[0] == 1.0

    @pytest.mark.asyncio
    async def test_no_backoff_on_permanent_error(self) -> None:
        """Permanent errors terminate immediately with no sleep."""
        recorder = RecordingSleepFn()
        llm = FailingLLMClient(ValueError("bad response"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=1.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
            sleep_fn=recorder,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        # No retries -> no sleep calls
        assert len(recorder.delays) == 0
        assert llm.call_count == 1


# ---------------------------------------------------------------------------
# AgentLoop -- context preservation across retries
# ---------------------------------------------------------------------------


class TestAgentLoopContextPreservation:
    """Tests verifying conversation context is preserved through retries."""

    @pytest.mark.asyncio
    async def test_same_messages_sent_on_retry(self) -> None:
        """The same message history is re-sent on each retry attempt."""
        captured_messages: list[tuple[dict[str, Any], ...]] = []

        class CapturingLLMClient:
            """Captures messages from each call for inspection."""

            def __init__(self, fail_count: int, error: Exception) -> None:
                self._fail_count = fail_count
                self._error = error
                self._call_count = 0

            @property
            def call_count(self) -> int:
                return self._call_count

            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                self._call_count += 1
                captured_messages.append(messages)
                if self._call_count <= self._fail_count:
                    raise self._error
                return ()

        llm = CapturingLLMClient(fail_count=2, error=ConnectionError("blip"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=0.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test system",
            config=config,
        )
        result = await loop.run("test command")

        assert result.final_state is AgentLoopState.COMPLETE
        # 3 calls: 2 failures + 1 success
        assert len(captured_messages) == 3

        # All three calls receive the exact same messages
        assert captured_messages[0] == captured_messages[1]
        assert captured_messages[1] == captured_messages[2]

        # Verify the content is the initial system + user messages
        assert captured_messages[0][0]["role"] == "system"
        assert captured_messages[0][0]["content"] == "test system"
        assert captured_messages[0][1]["role"] == "user"
        assert captured_messages[0][1]["content"] == "test command"

    @pytest.mark.asyncio
    async def test_history_not_corrupted_after_transient_retries(self) -> None:
        """History is clean after successful transient retry."""
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=TimeoutError("timeout"),
            success_responses=[
                _make_tool_calls("read_wiki"),
                (),  # completes
            ],
        )
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=0.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        # History should contain system, user, assistant+tool, no error artifacts
        roles = [m["role"] for m in result.history]
        assert roles[0] == "system"
        assert roles[1] == "user"
        assert "assistant" in roles
        assert "tool" in roles

    @pytest.mark.asyncio
    async def test_retry_across_iterations_preserves_full_history(self) -> None:
        """Retries in later iterations still see the full conversation history."""
        captured_messages: list[tuple[dict[str, Any], ...]] = []

        class RetryOnSecondIterationLLM:
            """Succeeds on iter 1, fails then succeeds on iter 2."""

            def __init__(self) -> None:
                self._call_count = 0

            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                self._call_count += 1
                captured_messages.append(messages)

                if self._call_count == 1:
                    # Iteration 1: return a tool call
                    return _make_tool_calls("read_wiki", base_id="iter1")
                if self._call_count == 2:
                    # Iteration 2, first attempt: fail with transient
                    raise ConnectionError("blip")
                if self._call_count == 3:
                    # Iteration 2, retry: succeed with completion
                    return ()
                return ()

        llm = RetryOnSecondIterationLLM()
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=0.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test system",
            config=config,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        # Call 1: system + user (2 messages)
        assert len(captured_messages[0]) == 2
        # Call 2 (failed): system + user + assistant + tool (4 messages)
        assert len(captured_messages[1]) == 4
        # Call 3 (retried): same messages as call 2
        assert captured_messages[1] == captured_messages[2]


# ---------------------------------------------------------------------------
# AgentLoop -- classify_error integration
# ---------------------------------------------------------------------------


class TestAgentLoopClassifyErrorIntegration:
    """Tests verifying integration with the classify_error module."""

    @pytest.mark.asyncio
    async def test_timeout_error_classified_as_transient(self) -> None:
        """TimeoutError is classified as transient by classify_error."""
        recorder = RecordingSleepFn()
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=TimeoutError("timed out"),
            success_responses=[()],
        )
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=0.5)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
            sleep_fn=recorder,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert len(recorder.delays) == 1

    @pytest.mark.asyncio
    async def test_os_error_classified_as_transient(self) -> None:
        """OSError is classified as transient by classify_error."""
        recorder = RecordingSleepFn()
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=OSError("socket error"),
            success_responses=[()],
        )
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=1.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
            sleep_fn=recorder,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_key_error_classified_as_permanent(self) -> None:
        """KeyError is classified as permanent by classify_error."""
        recorder = RecordingSleepFn()
        llm = FailingLLMClient(KeyError("missing_key"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=1.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
            sleep_fn=recorder,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert llm.call_count == 1
        assert len(recorder.delays) == 0

    @pytest.mark.asyncio
    async def test_error_message_includes_category(self) -> None:
        """Error message in result includes the error category from classification."""
        llm = FailingLLMClient(ConnectionError("network unreachable"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.error_message is not None
        # Should include the category from classify_error
        assert "network" in result.error_message.lower() or "falling back" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_exhausted_retries_error_message_mentions_one_shot(self) -> None:
        """Exhausted transient retries produce a fallback-to-one-shot hint."""
        llm = FailingLLMClient(ConnectionError("down"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=1, retry_base_delay=0.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert "falling back to one-shot" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_permanent_error_message_says_permanent(self) -> None:
        """Permanent error message clearly identifies the error as permanent."""
        llm = FailingLLMClient(ValueError("invalid JSON"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=1.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert "permanent" in result.error_message.lower()


# ---------------------------------------------------------------------------
# AgentLoop -- sleep_fn injection
# ---------------------------------------------------------------------------


class TestAgentLoopSleepFnInjection:
    """Tests for sleep function dependency injection."""

    @pytest.mark.asyncio
    async def test_default_sleep_fn_is_asyncio_sleep(self) -> None:
        """When no sleep_fn is provided, asyncio.sleep is used."""
        import asyncio

        llm = MockLLMClient([()])
        dispatcher = MockToolDispatcher()

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
        )
        # Internal _sleep should be asyncio.sleep
        assert loop._sleep is asyncio.sleep

    @pytest.mark.asyncio
    async def test_custom_sleep_fn_is_used(self) -> None:
        """Injected sleep_fn is called instead of asyncio.sleep."""
        recorder = RecordingSleepFn()
        llm = TransientThenSuccessLLMClient(
            fail_count=1,
            error=ConnectionError("blip"),
            success_responses=[()],
        )
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=1.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test prompt",
            config=config,
            sleep_fn=recorder,
        )
        await loop.run("run tests")

        # Our recording function was called
        assert len(recorder.delays) == 1
        assert recorder.delays[0] == 1.0
