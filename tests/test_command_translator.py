"""Tests for the command translator with 5s deadline enforcement.

Covers:
    - CommandTranslator construction and configuration
    - TranslationResult data model
    - Full translation pipeline (NL input -> proposed SSH commands)
    - 5-second deadline enforcement and timeout behavior
    - System prompt caching for latency reduction
    - Error propagation from LLM client and response parser
    - Convenience function translate_command()
    - TranslationTimeout error attributes
    - Edge cases: empty input, LLM refusal, empty response

All LLM calls are mocked -- these tests verify the orchestration logic,
deadline enforcement, and data flow, not the actual LLM behavior.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from jules_daemon.llm.command_translator import (
    DEFAULT_DEADLINE_SECONDS,
    CommandTranslator,
    TranslationResult,
    TranslationTimeout,
    _SystemPromptCache,
    translate_command,
)
from jules_daemon.llm.config import LLMConfig
from jules_daemon.llm.errors import (
    LLMConnectionError,
    LLMError,
    LLMParseError,
)
from jules_daemon.llm.models import ToolCallingMode
from jules_daemon.llm.prompts import HostContext, PromptConfig
from jules_daemon.llm.response_parser import Confidence
from jules_daemon.ssh.command import SSHCommand


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> LLMConfig:
    """Create a test LLMConfig with sensible defaults."""
    defaults = {
        "base_url": "https://dss.example.com/public/api/projects/PROJ/llms/openai/v1/",
        "api_key": "dkuapi_test123",
        "default_model": "openai:my-conn:gpt-4",
    }
    return LLMConfig(**{**defaults, **overrides})


def _make_host_context(**overrides: Any) -> HostContext:
    """Create a test HostContext with minimal defaults."""
    defaults = {
        "hostname": "staging.example.com",
        "user": "deploy",
    }
    return HostContext(**{**defaults, **overrides})


def _make_llm_json_response(
    *,
    commands: list[dict[str, Any]] | None = None,
    explanation: str = "Running the test suite",
    confidence: str = "high",
    warnings: list[str] | None = None,
) -> str:
    """Build a JSON string matching the expected LLM output schema."""
    if commands is None:
        commands = [
            {
                "command": "cd /opt/app && pytest -v --tb=short",
                "description": "Run the full pytest suite with verbose output",
                "working_directory": "/opt/app",
                "timeout_seconds": 300,
            }
        ]
    return json.dumps({
        "commands": commands,
        "explanation": explanation,
        "confidence": confidence,
        "warnings": warnings or [],
    })


def _mock_completion_response(content: str) -> MagicMock:
    """Build a mock ChatCompletion response with the given content."""
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    message.role = "assistant"

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.model = "openai:my-conn:gpt-4"
    response.usage = MagicMock(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    return response


def _mock_slow_completion(delay: float, content: str) -> MagicMock:
    """Build a mock that delays before returning, simulating slow LLM."""
    mock_response = _mock_completion_response(content)

    def slow_create(**kwargs: Any) -> MagicMock:
        time.sleep(delay)
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = slow_create
    return mock_client


# ---------------------------------------------------------------------------
# TranslationResult tests
# ---------------------------------------------------------------------------


class TestTranslationResult:
    """Tests for the immutable TranslationResult dataclass."""

    def test_create_with_commands(self) -> None:
        from jules_daemon.llm.response_parser import (
            LLMCommandResponse,
            LLMCommandStep,
        )

        response = LLMCommandResponse(
            commands=[
                LLMCommandStep(
                    command="pytest -v",
                    description="Run tests",
                    working_directory="/opt/app",
                )
            ],
            explanation="Running tests",
            confidence=Confidence.HIGH,
        )
        ssh_cmd = SSHCommand(
            command="pytest -v",
            working_directory="/opt/app",
        )
        result = TranslationResult(
            response=response,
            ssh_commands=(ssh_cmd,),
            elapsed_seconds=1.5,
            natural_language="run the tests",
            deadline_seconds=5.0,
        )
        assert result.command_count == 1
        assert result.is_refusal is False
        assert result.met_deadline is True
        assert result.elapsed_seconds == 1.5
        assert result.natural_language == "run the tests"

    def test_frozen(self) -> None:
        from jules_daemon.llm.response_parser import (
            LLMCommandResponse,
            LLMCommandStep,
        )

        response = LLMCommandResponse(
            commands=[
                LLMCommandStep(command="ls", description="List files")
            ],
            explanation="Listing",
            confidence=Confidence.HIGH,
        )
        result = TranslationResult(
            response=response,
            ssh_commands=(),
            elapsed_seconds=0.5,
            natural_language="list files",
            deadline_seconds=5.0,
        )
        with pytest.raises(AttributeError):
            result.elapsed_seconds = 10.0  # type: ignore[misc]

    def test_met_deadline_false_when_exceeded(self) -> None:
        from jules_daemon.llm.response_parser import LLMCommandResponse

        response = LLMCommandResponse(
            commands=[],
            explanation="Refused",
            confidence=Confidence.LOW,
        )
        result = TranslationResult(
            response=response,
            ssh_commands=(),
            elapsed_seconds=6.0,
            natural_language="something",
            deadline_seconds=5.0,
        )
        assert result.met_deadline is False

    def test_refusal_result(self) -> None:
        from jules_daemon.llm.response_parser import LLMCommandResponse

        response = LLMCommandResponse(
            commands=[],
            explanation="Cannot fulfill request",
            confidence=Confidence.LOW,
            warnings=["Request is outside allowed actions"],
        )
        result = TranslationResult(
            response=response,
            ssh_commands=(),
            elapsed_seconds=0.8,
            natural_language="delete everything",
            deadline_seconds=5.0,
        )
        assert result.is_refusal is True
        assert result.command_count == 0

    def test_ssh_commands_is_tuple(self) -> None:
        """ssh_commands should be a tuple (immutable) not a list."""
        from jules_daemon.llm.response_parser import LLMCommandResponse

        response = LLMCommandResponse(
            commands=[],
            explanation="Empty",
            confidence=Confidence.LOW,
        )
        result = TranslationResult(
            response=response,
            ssh_commands=(),
            elapsed_seconds=0.5,
            natural_language="test",
            deadline_seconds=5.0,
        )
        assert isinstance(result.ssh_commands, tuple)


# ---------------------------------------------------------------------------
# TranslationTimeout tests
# ---------------------------------------------------------------------------


class TestTranslationTimeout:
    """Tests for the TranslationTimeout error type."""

    def test_attributes(self) -> None:
        exc = TranslationTimeout(
            message="Timed out",
            deadline_seconds=5.0,
            elapsed_seconds=6.5,
        )
        assert str(exc) == "Timed out"
        assert exc.deadline_seconds == 5.0
        assert exc.elapsed_seconds == 6.5

    def test_is_llm_error(self) -> None:
        exc = TranslationTimeout("timeout", deadline_seconds=5.0, elapsed_seconds=6.0)
        assert isinstance(exc, LLMError)


# ---------------------------------------------------------------------------
# CommandTranslator construction
# ---------------------------------------------------------------------------


class TestCommandTranslatorInit:
    """Tests for CommandTranslator initialization."""

    def test_default_deadline(self) -> None:
        config = _make_config()
        mock_client = MagicMock()
        translator = CommandTranslator(client=mock_client, config=config)
        assert translator.deadline_seconds == DEFAULT_DEADLINE_SECONDS

    def test_custom_deadline(self) -> None:
        config = _make_config()
        mock_client = MagicMock()
        translator = CommandTranslator(
            client=mock_client,
            config=config,
            deadline_seconds=3.0,
        )
        assert translator.deadline_seconds == 3.0

    def test_zero_deadline_raises(self) -> None:
        config = _make_config()
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="deadline_seconds must be positive"):
            CommandTranslator(
                client=mock_client,
                config=config,
                deadline_seconds=0,
            )

    def test_negative_deadline_raises(self) -> None:
        config = _make_config()
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="deadline_seconds must be positive"):
            CommandTranslator(
                client=mock_client,
                config=config,
                deadline_seconds=-1.0,
            )

    def test_default_is_5_seconds(self) -> None:
        """The default deadline constant must be 5 seconds per AC 4."""
        assert DEFAULT_DEADLINE_SECONDS == 5.0


# ---------------------------------------------------------------------------
# Translation pipeline tests
# ---------------------------------------------------------------------------


class TestTranslate:
    """Tests for the full translation pipeline."""

    def _make_translator(
        self,
        mock_client: MagicMock,
        deadline: float = 5.0,
    ) -> CommandTranslator:
        config = _make_config()
        return CommandTranslator(
            client=mock_client,
            config=config,
            deadline_seconds=deadline,
        )

    def test_successful_translation(self) -> None:
        """Full pipeline: NL -> LLM call -> parsed response -> SSHCommand."""
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        translator = self._make_translator(mock_client)
        result = translator.translate(
            natural_language="run the smoke tests",
            host_context=_make_host_context(),
        )

        assert isinstance(result, TranslationResult)
        assert result.command_count == 1
        assert result.ssh_commands[0].command == "cd /opt/app && pytest -v --tb=short"
        assert result.is_refusal is False
        assert result.natural_language == "run the smoke tests"

    def test_elapsed_time_recorded(self) -> None:
        """elapsed_seconds must be positive and under the deadline."""
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        translator = self._make_translator(mock_client, deadline=10.0)
        result = translator.translate(
            natural_language="run tests",
            host_context=_make_host_context(),
        )

        assert result.elapsed_seconds > 0
        assert result.elapsed_seconds < 10.0
        assert result.met_deadline is True

    def test_multiple_commands(self) -> None:
        """LLM returns multiple command steps."""
        content = _make_llm_json_response(
            commands=[
                {
                    "command": "cd /opt/app",
                    "description": "Navigate to app directory",
                },
                {
                    "command": "pytest -v --tb=short",
                    "description": "Run the test suite",
                    "working_directory": "/opt/app",
                    "timeout_seconds": 600,
                },
            ],
            explanation="Navigate and run tests",
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        translator = self._make_translator(mock_client)
        result = translator.translate(
            natural_language="run the tests",
            host_context=_make_host_context(),
        )

        assert result.command_count == 2
        assert result.ssh_commands[0].command == "cd /opt/app"
        assert result.ssh_commands[1].command == "pytest -v --tb=short"
        assert result.ssh_commands[1].timeout == 600

    def test_refusal_response(self) -> None:
        """LLM refuses a dangerous request."""
        content = _make_llm_json_response(
            commands=[],
            explanation="Cannot delete system files",
            confidence="low",
            warnings=["Request involves forbidden operations"],
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        translator = self._make_translator(mock_client)
        result = translator.translate(
            natural_language="rm -rf /",
            host_context=_make_host_context(),
        )

        assert result.is_refusal is True
        assert result.command_count == 0
        assert result.response.confidence is Confidence.LOW

    def test_empty_input_raises(self) -> None:
        mock_client = MagicMock()
        translator = self._make_translator(mock_client)

        with pytest.raises(ValueError, match="natural_language must not be empty"):
            translator.translate(
                natural_language="",
                host_context=_make_host_context(),
            )

    def test_whitespace_only_input_raises(self) -> None:
        mock_client = MagicMock()
        translator = self._make_translator(mock_client)

        with pytest.raises(ValueError, match="natural_language must not be empty"):
            translator.translate(
                natural_language="   ",
                host_context=_make_host_context(),
            )

    def test_llm_connection_error_propagated(self) -> None:
        """LLM connection errors should propagate as LLMConnectionError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = LLMConnectionError(
            "Connection refused"
        )

        translator = self._make_translator(mock_client)
        with pytest.raises(LLMConnectionError):
            translator.translate(
                natural_language="run tests",
                host_context=_make_host_context(),
            )

    def test_llm_parse_error_propagated(self) -> None:
        """Malformed LLM response should raise LLMParseError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response("This is not JSON at all.")
        )

        translator = self._make_translator(mock_client)
        with pytest.raises(LLMParseError):
            translator.translate(
                natural_language="run tests",
                host_context=_make_host_context(),
            )

    def test_empty_choices_raises(self) -> None:
        """LLM returns no choices."""
        mock_response = MagicMock()
        mock_response.choices = []

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        translator = self._make_translator(mock_client)
        with pytest.raises(LLMParseError, match="empty choices"):
            translator.translate(
                natural_language="run tests",
                host_context=_make_host_context(),
            )

    def test_empty_content_raises(self) -> None:
        """LLM returns a choice with no content."""
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        mock_response = MagicMock()
        mock_response.choices = [choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        translator = self._make_translator(mock_client)
        with pytest.raises(LLMParseError, match="empty content"):
            translator.translate(
                natural_language="run tests",
                host_context=_make_host_context(),
            )

    def test_per_call_deadline_override(self) -> None:
        """translate() can override the instance-level deadline."""
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        translator = self._make_translator(mock_client, deadline=1.0)
        result = translator.translate(
            natural_language="run tests",
            host_context=_make_host_context(),
            deadline_seconds=10.0,
        )

        assert result.deadline_seconds == 10.0

    def test_host_context_passed_to_prompt(self) -> None:
        """Host context details should flow into the LLM messages."""
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        host = _make_host_context(
            hostname="ci.internal.net",
            user="runner",
            test_framework_hint="pytest",
        )

        translator = self._make_translator(mock_client)
        translator.translate(
            natural_language="run smoke tests",
            host_context=host,
        )

        # Verify the LLM was called with messages containing host details
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        user_msg = messages[-1]["content"]
        assert "ci.internal.net" in user_msg
        assert "runner" in user_msg
        assert "pytest" in user_msg

    def test_system_prompt_present_in_messages(self) -> None:
        """The system prompt should be the first message."""
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        translator = self._make_translator(mock_client)
        translator.translate(
            natural_language="run tests",
            host_context=_make_host_context(),
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "test execution assistant" in messages[0]["content"].lower()

    def test_temperature_set_to_zero(self) -> None:
        """Temperature must be 0.0 for deterministic/fast responses."""
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        translator = self._make_translator(mock_client)
        translator.translate(
            natural_language="run tests",
            host_context=_make_host_context(),
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Deadline enforcement tests
# ---------------------------------------------------------------------------


class TestDeadlineEnforcement:
    """Tests specifically for the 5-second deadline SLA."""

    def test_fast_response_within_deadline(self) -> None:
        """A fast LLM response should complete well within 5 seconds."""
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        translator = CommandTranslator(
            client=mock_client,
            config=_make_config(),
            deadline_seconds=5.0,
        )

        start = time.monotonic()
        result = translator.translate(
            natural_language="run tests",
            host_context=_make_host_context(),
        )
        elapsed = time.monotonic() - start

        # With a mock (instant) LLM, the pipeline itself should be < 100ms
        assert elapsed < 0.5
        assert result.met_deadline is True
        assert result.elapsed_seconds < 5.0

    def test_slow_llm_raises_timeout(self) -> None:
        """An LLM call exceeding the deadline should raise TranslationTimeout."""
        content = _make_llm_json_response()
        mock_client = _mock_slow_completion(delay=0.5, content=content)

        translator = CommandTranslator(
            client=mock_client,
            config=_make_config(),
            deadline_seconds=0.2,  # Very short deadline
        )

        with pytest.raises(TranslationTimeout) as exc_info:
            translator.translate(
                natural_language="run tests",
                host_context=_make_host_context(),
            )

        assert exc_info.value.deadline_seconds == 0.2
        assert exc_info.value.elapsed_seconds > 0.2

    def test_deadline_attributes_in_result(self) -> None:
        """TranslationResult must carry the deadline that was enforced."""
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        translator = CommandTranslator(
            client=mock_client,
            config=_make_config(),
            deadline_seconds=5.0,
        )
        result = translator.translate(
            natural_language="run tests",
            host_context=_make_host_context(),
        )

        assert result.deadline_seconds == 5.0

    def test_prompt_construction_is_sub_millisecond(self) -> None:
        """Prompt construction should be negligibly fast compared to LLM call."""
        from jules_daemon.llm.prompts import build_messages

        host = _make_host_context(
            hostname="staging.example.com",
            user="deploy",
            working_directory="/opt/app",
            os_hint="Ubuntu 22.04",
            shell_hint="bash",
            test_framework_hint="pytest",
            extra_context=("Python 3.12", "Django 5.0"),
        )

        start = time.monotonic()
        for _ in range(100):
            build_messages(
                natural_language="run the complete integration test suite with coverage",
                host_context=host,
            )
        elapsed = time.monotonic() - start

        # 100 iterations should complete in well under 1 second
        # Each iteration should be sub-millisecond
        assert elapsed < 1.0, (
            f"100 prompt constructions took {elapsed:.3f}s, "
            f"avg {elapsed/100*1000:.1f}ms -- too slow for 5s deadline"
        )

    def test_response_parsing_is_sub_millisecond(self) -> None:
        """Response parsing should be negligibly fast compared to LLM call."""
        from jules_daemon.llm.response_parser import parse_llm_response

        text = _make_llm_json_response(
            commands=[
                {"command": f"pytest tests/test_{i}.py -v", "description": f"Run test {i}"}
                for i in range(5)
            ],
            explanation="Running 5 test files",
        )

        start = time.monotonic()
        for _ in range(100):
            parse_llm_response(text)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, (
            f"100 parse iterations took {elapsed:.3f}s, "
            f"avg {elapsed/100*1000:.1f}ms -- too slow for 5s deadline"
        )

    def test_pipeline_overhead_under_100ms(self) -> None:
        """Non-LLM overhead (prompt + parse + mapping) must be < 100ms."""
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        translator = CommandTranslator(
            client=mock_client,
            config=_make_config(),
            deadline_seconds=5.0,
        )

        start = time.monotonic()
        result = translator.translate(
            natural_language="run the smoke tests",
            host_context=_make_host_context(),
        )
        elapsed = time.monotonic() - start

        # With a mock (instant) LLM call, total overhead should be < 100ms
        assert elapsed < 0.1, (
            f"Pipeline overhead was {elapsed*1000:.1f}ms -- "
            f"must be under 100ms to leave 4.9s for LLM call"
        )

    def test_timeout_error_has_timing_info(self) -> None:
        """TranslationTimeout must include deadline and elapsed info."""
        content = _make_llm_json_response()
        mock_client = _mock_slow_completion(delay=1.0, content=content)

        translator = CommandTranslator(
            client=mock_client,
            config=_make_config(),
            deadline_seconds=0.3,
        )

        with pytest.raises(TranslationTimeout) as exc_info:
            translator.translate(
                natural_language="run tests",
                host_context=_make_host_context(),
            )

        timeout_err = exc_info.value
        assert timeout_err.deadline_seconds == 0.3
        assert timeout_err.elapsed_seconds >= 0.3
        assert "deadline" in str(timeout_err).lower()


# ---------------------------------------------------------------------------
# System prompt caching
# ---------------------------------------------------------------------------


class TestSystemPromptCache:
    """Tests for the system prompt cache."""

    def test_cache_returns_same_string(self) -> None:
        cache = _SystemPromptCache()
        config = PromptConfig()
        first = cache.get(config)
        second = cache.get(config)
        assert first is second  # Same object reference (cached)

    def test_cache_different_configs(self) -> None:
        """Different config objects produce different cache entries."""
        cache = _SystemPromptCache()
        config1 = PromptConfig(max_commands=3)
        config2 = PromptConfig(max_commands=7)
        prompt1 = cache.get(config1)
        prompt2 = cache.get(config2)
        assert prompt1 is not prompt2
        assert "3" in prompt1
        assert "7" in prompt2

    def test_cache_clear(self) -> None:
        cache = _SystemPromptCache()
        config = PromptConfig()
        first = cache.get(config)
        cache.clear()
        second = cache.get(config)
        # After clear, a new string is built (may or may not be same object)
        assert first == second  # Same content
        # but the cache was cleared, so it was rebuilt

    def test_cached_prompt_faster_than_uncached(self) -> None:
        """Cached prompt retrieval should be faster than fresh construction."""
        from jules_daemon.llm.prompts import build_system_prompt

        cache = _SystemPromptCache()
        config = PromptConfig()

        # Prime the cache
        cache.get(config)

        # Measure cached access
        start = time.monotonic()
        for _ in range(1000):
            cache.get(config)
        cached_time = time.monotonic() - start

        # Measure uncached construction
        start = time.monotonic()
        for _ in range(1000):
            build_system_prompt(config=config)
        uncached_time = time.monotonic() - start

        # Cached should be faster (or at least not slower)
        # Both should be fast, but cache avoids string construction
        assert cached_time <= uncached_time * 2  # Allow some margin


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestTranslateCommandFunction:
    """Tests for the translate_command() convenience function."""

    def test_basic_call(self) -> None:
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        result = translate_command(
            client=mock_client,
            config=_make_config(),
            natural_language="run the smoke tests",
            host_context=_make_host_context(),
        )

        assert isinstance(result, TranslationResult)
        assert result.command_count == 1
        assert result.met_deadline is True

    def test_custom_deadline(self) -> None:
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        result = translate_command(
            client=mock_client,
            config=_make_config(),
            natural_language="run tests",
            host_context=_make_host_context(),
            deadline_seconds=10.0,
        )

        assert result.deadline_seconds == 10.0

    def test_timeout_propagated(self) -> None:
        content = _make_llm_json_response()
        mock_client = _mock_slow_completion(delay=0.5, content=content)

        with pytest.raises(TranslationTimeout):
            translate_command(
                client=mock_client,
                config=_make_config(),
                natural_language="run tests",
                host_context=_make_host_context(),
                deadline_seconds=0.1,
            )

    def test_prompt_config_passed_through(self) -> None:
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        custom_config = PromptConfig(max_commands=2)

        translate_command(
            client=mock_client,
            config=_make_config(),
            natural_language="run tests",
            host_context=_make_host_context(),
            prompt_config=custom_config,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        system_content = messages[0]["content"]
        assert "2" in system_content

    def test_tool_calling_mode_passed_through(self) -> None:
        content = _make_llm_json_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        translate_command(
            client=mock_client,
            config=_make_config(),
            natural_language="run tests",
            host_context=_make_host_context(),
            tool_calling_mode=ToolCallingMode.PROMPT_BASED,
        )

        # Should still complete successfully with prompt-based mode
        mock_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# Integration-style tests (still using mocks, but testing the full flow)
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    """End-to-end pipeline tests verifying the full NL -> command flow."""

    def test_pytest_command_flow(self) -> None:
        """Typical pytest command generation flow."""
        content = _make_llm_json_response(
            commands=[
                {
                    "command": "cd /opt/myapp && python -m pytest tests/ -v --tb=short",
                    "description": "Run all tests with verbose output and short tracebacks",
                    "working_directory": "/opt/myapp",
                    "timeout_seconds": 300,
                }
            ],
            explanation="Running the full pytest suite in the application directory",
            confidence="high",
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        result = translate_command(
            client=mock_client,
            config=_make_config(),
            natural_language="run all the tests with verbose output",
            host_context=_make_host_context(
                working_directory="/opt/myapp",
                test_framework_hint="pytest",
            ),
        )

        assert result.command_count == 1
        cmd = result.ssh_commands[0]
        assert "pytest" in cmd.command
        assert cmd.working_directory == "/opt/myapp"
        assert cmd.timeout == 300
        assert result.response.confidence is Confidence.HIGH
        assert result.met_deadline is True
        assert result.elapsed_seconds < 5.0

    def test_multi_step_command_flow(self) -> None:
        """Multi-step command with navigation and execution."""
        content = _make_llm_json_response(
            commands=[
                {
                    "command": "cd /home/deploy/project",
                    "description": "Navigate to the project directory",
                },
                {
                    "command": "source venv/bin/activate && pytest tests/integration/ -v",
                    "description": "Activate virtualenv and run integration tests",
                    "working_directory": "/home/deploy/project",
                    "timeout_seconds": 600,
                },
            ],
            explanation="Setting up environment and running integration tests",
            confidence="medium",
            warnings=["Integration tests may take longer than expected"],
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        result = translate_command(
            client=mock_client,
            config=_make_config(),
            natural_language="run the integration tests on the deploy project",
            host_context=_make_host_context(
                hostname="prod.example.com",
                user="deploy",
            ),
        )

        assert result.command_count == 2
        assert result.response.confidence is Confidence.MEDIUM
        assert len(result.response.warnings) == 1
        assert result.met_deadline is True

    def test_code_fenced_llm_output(self) -> None:
        """LLM wraps JSON in markdown code fence."""
        json_body = _make_llm_json_response(
            commands=[
                {
                    "command": "make test",
                    "description": "Run make test target",
                }
            ],
            explanation="Using the project's Makefile to run tests",
        )
        content = f"Here is the plan:\n\n```json\n{json_body}\n```\n\nLet me know if you want changes."

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _mock_completion_response(content)
        )

        result = translate_command(
            client=mock_client,
            config=_make_config(),
            natural_language="run make test",
            host_context=_make_host_context(),
        )

        assert result.command_count == 1
        assert result.ssh_commands[0].command == "make test"

    def test_repeated_translations_with_same_translator(self) -> None:
        """Same translator instance should work for multiple requests."""
        mock_client = MagicMock()
        config = _make_config()
        translator = CommandTranslator(
            client=mock_client,
            config=config,
            deadline_seconds=5.0,
        )

        for i in range(3):
            content = _make_llm_json_response(
                commands=[
                    {
                        "command": f"pytest tests/test_{i}.py",
                        "description": f"Run test file {i}",
                    }
                ],
                explanation=f"Running test {i}",
            )
            mock_client.chat.completions.create.return_value = (
                _mock_completion_response(content)
            )

            result = translator.translate(
                natural_language=f"run test {i}",
                host_context=_make_host_context(),
            )

            assert result.command_count == 1
            assert result.met_deadline is True

        # Should have been called 3 times
        assert mock_client.chat.completions.create.call_count == 3
