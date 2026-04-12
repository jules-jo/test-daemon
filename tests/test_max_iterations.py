"""Comprehensive tests for max-iteration enforcement (AC 9).

Validates:
    - Default hard cap is 5 iterations
    - Cap is configurable via AgentLoopConfig constructor
    - Cap is configurable via environment variable JULES_AGENT_MAX_ITERATIONS
    - Loop terminates with ERROR state when cap is reached
    - iterations_used in result exactly matches the cap value
    - Error message includes the configured cap value
    - Boundary: max_iterations=1 stops after exactly 1 cycle
    - Various cap values (1, 2, 3, 5, 10) all enforce correctly
    - Validation rejects invalid values (0, negative)
    - Config is immutable (frozen dataclass)
    - Early completion (before cap) still works
    - Cap flows from RequestHandlerConfig to AgentLoopConfig
    - resolve_max_iterations priority: explicit > env > default
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from jules_daemon.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopResult,
    AgentLoopState,
)
from jules_daemon.agent.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_RETRIES,
    load_agent_config,
    load_agent_config_from_env,
    resolve_max_iterations,
)
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResult,
    ToolResultStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_calls(
    name: str = "read_wiki",
    *,
    base_id: str = "call",
) -> tuple[ToolCall, ...]:
    """Create a single-element tuple of ToolCalls."""
    return (
        ToolCall(
            call_id=f"{base_id}_0",
            tool_name=name,
            arguments={"arg": "val"},
        ),
    )


def _infinite_calls(n: int = 50) -> list[tuple[ToolCall, ...]]:
    """Generate more calls than any reasonable max_iterations cap."""
    return [
        _make_tool_calls(base_id=f"iter{i}")
        for i in range(n)
    ]


class _MockLLMClient:
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
    """Mock tool dispatcher that always returns success."""

    def __init__(self) -> None:
        self._dispatched: list[ToolCall] = []

    @property
    def dispatch_count(self) -> int:
        return len(self._dispatched)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        self._dispatched.append(call)
        return ToolResult.success(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output=f"ok: {call.tool_name}",
        )


def _make_loop(
    *,
    max_iterations: int = 5,
    responses: list[tuple[ToolCall, ...]] | None = None,
) -> tuple[AgentLoop, _MockLLMClient, _MockDispatcher]:
    """Create an AgentLoop with mocks for testing iteration cap."""
    llm = _MockLLMClient(responses or _infinite_calls())
    dispatcher = _MockDispatcher()
    config = AgentLoopConfig(max_iterations=max_iterations)
    loop = AgentLoop(
        llm_client=llm,
        tool_dispatcher=dispatcher,
        system_prompt="You are a test runner assistant.",
        config=config,
    )
    return loop, llm, dispatcher


# ---------------------------------------------------------------------------
# Default cap value
# ---------------------------------------------------------------------------


class TestDefaultCap:
    """Verify the default iteration cap is 5."""

    def test_agent_loop_config_default_is_five(self) -> None:
        """AgentLoopConfig() defaults to max_iterations=5."""
        config = AgentLoopConfig()
        assert config.max_iterations == 15

    def test_default_constant_matches(self) -> None:
        """DEFAULT_MAX_ITERATIONS constant matches AgentLoopConfig default."""
        assert DEFAULT_MAX_ITERATIONS == 15
        assert DEFAULT_MAX_ITERATIONS == AgentLoopConfig().max_iterations

    @pytest.mark.asyncio
    async def test_default_cap_enforced_at_five(self) -> None:
        """With default config, loop stops at exactly 5 iterations."""
        llm = _MockLLMClient(_infinite_calls())
        dispatcher = _MockDispatcher()
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
        )
        result = await loop.run("run tests")

        assert result.iterations_used == 15
        assert result.final_state is AgentLoopState.ERROR


# ---------------------------------------------------------------------------
# Configurable cap via constructor
# ---------------------------------------------------------------------------


class TestConfigurableCap:
    """Verify the cap is configurable via AgentLoopConfig constructor."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cap", [1, 2, 3, 5, 7, 10])
    async def test_parameterized_cap_values(self, cap: int) -> None:
        """Loop stops at exactly the configured cap for various values."""
        loop, llm, dispatcher = _make_loop(max_iterations=cap)
        result = await loop.run("run tests")

        assert result.iterations_used == cap
        assert result.final_state is AgentLoopState.ERROR
        assert result.error_message is not None
        assert str(cap) in result.error_message

    @pytest.mark.asyncio
    async def test_minimum_cap_one(self) -> None:
        """max_iterations=1 stops after exactly one cycle."""
        loop, llm, dispatcher = _make_loop(max_iterations=1)
        result = await loop.run("run tests")

        assert result.iterations_used == 1
        assert result.final_state is AgentLoopState.ERROR
        assert llm.call_count == 1
        assert dispatcher.dispatch_count == 1

    @pytest.mark.asyncio
    async def test_cap_of_two(self) -> None:
        """max_iterations=2 allows exactly two cycles."""
        loop, llm, dispatcher = _make_loop(max_iterations=2)
        result = await loop.run("run tests")

        assert result.iterations_used == 2
        assert result.final_state is AgentLoopState.ERROR
        assert llm.call_count == 2
        assert dispatcher.dispatch_count == 2

    @pytest.mark.asyncio
    async def test_large_cap(self) -> None:
        """Large cap value (e.g. 20) still enforces correctly."""
        loop, llm, dispatcher = _make_loop(
            max_iterations=20,
            responses=_infinite_calls(30),
        )
        result = await loop.run("run tests")

        assert result.iterations_used == 20
        assert result.final_state is AgentLoopState.ERROR


# ---------------------------------------------------------------------------
# Error message content
# ---------------------------------------------------------------------------


class TestErrorMessage:
    """Verify the error message is descriptive when cap is reached."""

    @pytest.mark.asyncio
    async def test_error_message_mentions_max_iterations(self) -> None:
        """Error message contains 'max iterations' phrase."""
        loop, _, _ = _make_loop(max_iterations=3)
        result = await loop.run("run tests")

        assert result.error_message is not None
        assert "max iterations" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_message_includes_cap_value(self) -> None:
        """Error message includes the actual configured cap number."""
        loop, _, _ = _make_loop(max_iterations=7)
        result = await loop.run("run tests")

        assert result.error_message is not None
        assert "7" in result.error_message

    @pytest.mark.asyncio
    async def test_error_message_for_default_cap(self) -> None:
        """Error message includes '5' for default cap."""
        llm = _MockLLMClient(_infinite_calls())
        dispatcher = _MockDispatcher()
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
        )
        result = await loop.run("run tests")

        assert result.error_message is not None
        assert "(15)" in result.error_message


# ---------------------------------------------------------------------------
# Early completion (before cap)
# ---------------------------------------------------------------------------


class TestEarlyCompletion:
    """Verify loop can complete before reaching the cap."""

    @pytest.mark.asyncio
    async def test_completes_in_two_with_cap_of_five(self) -> None:
        """Loop completes in 2 iterations even though cap is 5."""
        responses = [
            _make_tool_calls(base_id="c1"),
            (),  # signals completion
        ]
        loop, _, _ = _make_loop(
            max_iterations=5,
            responses=[responses[0], responses[1]],
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 2
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_immediate_completion_with_large_cap(self) -> None:
        """LLM returns no calls on first iteration with cap=100."""
        loop, _, _ = _make_loop(
            max_iterations=100,
            responses=[()],
        )
        result = await loop.run("hello")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 1
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_iterations_used_less_than_cap(self) -> None:
        """iterations_used < max_iterations on early completion."""
        responses = [
            _make_tool_calls(base_id="c1"),
            _make_tool_calls(base_id="c2"),
            _make_tool_calls(base_id="c3"),
            (),
        ]
        loop, _, _ = _make_loop(
            max_iterations=10,
            responses=responses,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 4
        assert result.iterations_used < 10


# ---------------------------------------------------------------------------
# Validation: invalid cap values
# ---------------------------------------------------------------------------


class TestCapValidation:
    """Verify that invalid cap values are rejected at config construction."""

    def test_zero_raises_value_error(self) -> None:
        """max_iterations=0 is rejected."""
        with pytest.raises(ValueError, match="max_iterations"):
            AgentLoopConfig(max_iterations=0)

    def test_negative_raises_value_error(self) -> None:
        """max_iterations=-1 is rejected."""
        with pytest.raises(ValueError, match="max_iterations"):
            AgentLoopConfig(max_iterations=-1)

    def test_large_negative_raises_value_error(self) -> None:
        """max_iterations=-100 is rejected."""
        with pytest.raises(ValueError, match="max_iterations"):
            AgentLoopConfig(max_iterations=-100)

    def test_one_is_valid(self) -> None:
        """max_iterations=1 is the minimum valid value."""
        config = AgentLoopConfig(max_iterations=1)
        assert config.max_iterations == 1


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestConfigImmutability:
    """Verify config cannot be mutated after construction."""

    def test_frozen_max_iterations(self) -> None:
        """Cannot change max_iterations after creation."""
        config = AgentLoopConfig(max_iterations=5)
        with pytest.raises(AttributeError):
            config.max_iterations = 10  # type: ignore[misc]

    def test_frozen_max_retries(self) -> None:
        """Cannot change max_retries after creation."""
        config = AgentLoopConfig()
        with pytest.raises(AttributeError):
            config.max_retries = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Environment variable configuration
# ---------------------------------------------------------------------------


class TestEnvVarConfiguration:
    """Verify max_iterations is configurable via environment variable."""

    def test_load_from_env_default(self) -> None:
        """Without env var, defaults are used."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_agent_config_from_env()
        assert config.max_iterations == DEFAULT_MAX_ITERATIONS
        assert config.max_retries == DEFAULT_MAX_RETRIES

    def test_load_from_env_custom_iterations(self) -> None:
        """JULES_AGENT_MAX_ITERATIONS overrides default."""
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "10"}):
            config = load_agent_config_from_env()
        assert config.max_iterations == 10
        assert config.max_retries == DEFAULT_MAX_RETRIES

    def test_load_from_env_custom_retries(self) -> None:
        """JULES_AGENT_MAX_RETRIES overrides default."""
        with patch.dict(os.environ, {"JULES_AGENT_MAX_RETRIES": "5"}):
            config = load_agent_config_from_env()
        assert config.max_retries == 5
        assert config.max_iterations == DEFAULT_MAX_ITERATIONS

    def test_load_from_env_both(self) -> None:
        """Both env vars override defaults."""
        with patch.dict(os.environ, {
            "JULES_AGENT_MAX_ITERATIONS": "8",
            "JULES_AGENT_MAX_RETRIES": "3",
        }):
            config = load_agent_config_from_env()
        assert config.max_iterations == 8
        assert config.max_retries == 3

    def test_env_invalid_iterations_raises(self) -> None:
        """Non-integer JULES_AGENT_MAX_ITERATIONS raises ValueError."""
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "abc"}):
            with pytest.raises(ValueError, match="valid integer"):
                load_agent_config_from_env()

    def test_env_zero_iterations_raises(self) -> None:
        """JULES_AGENT_MAX_ITERATIONS=0 raises ValueError."""
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "0"}):
            with pytest.raises(ValueError, match=">= 1"):
                load_agent_config_from_env()

    def test_env_negative_iterations_raises(self) -> None:
        """Negative JULES_AGENT_MAX_ITERATIONS raises ValueError."""
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "-5"}):
            with pytest.raises(ValueError, match=">= 1"):
                load_agent_config_from_env()

    def test_env_negative_retries_raises(self) -> None:
        """Negative JULES_AGENT_MAX_RETRIES raises ValueError."""
        with patch.dict(os.environ, {"JULES_AGENT_MAX_RETRIES": "-1"}):
            with pytest.raises(ValueError, match=">= 0"):
                load_agent_config_from_env()

    def test_env_invalid_retries_raises(self) -> None:
        """Non-integer JULES_AGENT_MAX_RETRIES raises ValueError."""
        with patch.dict(os.environ, {"JULES_AGENT_MAX_RETRIES": "xyz"}):
            with pytest.raises(ValueError, match="valid integer"):
                load_agent_config_from_env()


# ---------------------------------------------------------------------------
# resolve_max_iterations priority
# ---------------------------------------------------------------------------


class TestResolveMaxIterations:
    """Verify the resolution priority: explicit > env > default."""

    def test_explicit_takes_priority(self) -> None:
        """Explicit value overrides env var."""
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "10"}):
            result = resolve_max_iterations(explicit=3)
        assert result == 3

    def test_env_overrides_default(self) -> None:
        """Env var overrides default when no explicit value."""
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "8"}):
            result = resolve_max_iterations()
        assert result == 8

    def test_default_when_nothing_set(self) -> None:
        """Default is returned when no explicit or env."""
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_max_iterations()
        assert result == DEFAULT_MAX_ITERATIONS

    def test_explicit_none_falls_to_env(self) -> None:
        """explicit=None is treated as 'not provided'."""
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "7"}):
            result = resolve_max_iterations(explicit=None)
        assert result == 7

    def test_explicit_zero_raises(self) -> None:
        """explicit=0 raises ValueError (not valid)."""
        with pytest.raises(ValueError, match="max_iterations"):
            resolve_max_iterations(explicit=0)

    def test_explicit_negative_raises(self) -> None:
        """explicit=-1 raises ValueError."""
        with pytest.raises(ValueError, match="max_iterations"):
            resolve_max_iterations(explicit=-1)


# ---------------------------------------------------------------------------
# load_agent_config factory
# ---------------------------------------------------------------------------


class TestLoadAgentConfig:
    """Verify the explicit parameter factory function."""

    def test_defaults(self) -> None:
        """No args produces default config."""
        config = load_agent_config()
        assert config.max_iterations == DEFAULT_MAX_ITERATIONS
        assert config.max_retries == DEFAULT_MAX_RETRIES

    def test_custom_iterations(self) -> None:
        """Explicit max_iterations override."""
        config = load_agent_config(max_iterations=3)
        assert config.max_iterations == 3
        assert config.max_retries == DEFAULT_MAX_RETRIES

    def test_custom_retries(self) -> None:
        """Explicit max_retries override."""
        config = load_agent_config(max_retries=0)
        assert config.max_retries == 0
        assert config.max_iterations == DEFAULT_MAX_ITERATIONS

    def test_both_custom(self) -> None:
        """Both explicit overrides."""
        config = load_agent_config(max_iterations=10, max_retries=5)
        assert config.max_iterations == 10
        assert config.max_retries == 5


# ---------------------------------------------------------------------------
# Integration: env-configured cap enforced in loop
# ---------------------------------------------------------------------------


class TestEnvCapEnforcedInLoop:
    """Verify env-configured cap is enforced during loop execution."""

    @pytest.mark.asyncio
    async def test_env_cap_enforced(self) -> None:
        """Agent loop respects cap loaded from env variable."""
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "3"}):
            config = load_agent_config_from_env()

        llm = _MockLLMClient(_infinite_calls())
        dispatcher = _MockDispatcher()
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=config,
        )
        result = await loop.run("run tests")

        assert result.iterations_used == 3
        assert result.final_state is AgentLoopState.ERROR
        assert "3" in result.error_message

    @pytest.mark.asyncio
    async def test_env_cap_one_enforced(self) -> None:
        """Agent loop respects cap=1 loaded from env variable."""
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "1"}):
            config = load_agent_config_from_env()

        llm = _MockLLMClient(_infinite_calls())
        dispatcher = _MockDispatcher()
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=config,
        )
        result = await loop.run("run tests")

        assert result.iterations_used == 1
        assert result.final_state is AgentLoopState.ERROR


# ---------------------------------------------------------------------------
# Loop never exceeds cap (property-style check)
# ---------------------------------------------------------------------------


class TestIterationSafety:
    """Verify the loop NEVER exceeds the configured cap."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cap", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    async def test_iterations_used_never_exceeds_cap(self, cap: int) -> None:
        """For any cap value, iterations_used <= max_iterations."""
        loop, _, _ = _make_loop(
            max_iterations=cap,
            responses=_infinite_calls(cap + 10),
        )
        result = await loop.run("run tests")

        assert result.iterations_used <= cap
        assert result.iterations_used == cap  # Should hit exact cap

    @pytest.mark.asyncio
    async def test_dispatcher_called_exactly_cap_times(self) -> None:
        """Dispatcher is called exactly max_iterations times when capped."""
        cap = 4
        loop, llm, dispatcher = _make_loop(max_iterations=cap)
        await loop.run("run tests")

        # Each iteration dispatches one tool call
        assert dispatcher.dispatch_count == cap

    @pytest.mark.asyncio
    async def test_llm_called_exactly_cap_times(self) -> None:
        """LLM is called exactly max_iterations times when capped."""
        cap = 3
        loop, llm, _ = _make_loop(max_iterations=cap)
        await loop.run("run tests")

        assert llm.call_count == cap
