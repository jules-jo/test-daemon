"""Tests for the LLM API call wrapper (call_completion method).

Validates the core async function that sends messages to the LLM API
and returns the raw response, with:
- Retry logic for transient errors (network, timeout, rate limit)
- Timeout enforcement via asyncio.wait_for
- Error classification (transient vs permanent)
- Proper error wrapping with metadata
- Backward-compatible get_tool_calls behavior
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from jules_daemon.agent.llm_adapter import (
    LLMCallError,
    LLMCallErrorKind,
    LLMCallResult,
    OpenAILLMAdapter,
    classify_sdk_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(
    content: str = "Hello!",
    tool_calls: list[dict[str, Any]] | None = None,
    model: str = "test-model",
) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    response = MagicMock()
    message = MagicMock()
    message.content = content

    if tool_calls is None:
        message.tool_calls = None
    else:
        mock_tool_calls = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc["id"]
            mock_tc.function.name = tc["function"]["name"]
            mock_tc.function.arguments = tc["function"]["arguments"]
            mock_tool_calls.append(mock_tc)
        message.tool_calls = mock_tool_calls

    choice = MagicMock(message=message, finish_reason="stop")
    response.choices = [choice]
    response.model = model
    response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    return response


def _make_adapter(
    mock_client: MagicMock | None = None,
    model: str = "test:conn:gpt-4",
    tool_schemas: tuple[dict[str, Any], ...] = (),
    default_max_retries: int = 2,
    default_timeout: float | None = None,
) -> tuple[OpenAILLMAdapter, MagicMock]:
    """Create an adapter with a mock client for testing."""
    client = mock_client or MagicMock()
    adapter = OpenAILLMAdapter(
        client=client,
        model=model,
        tool_schemas=tool_schemas,
        default_max_retries=default_max_retries,
        default_timeout=default_timeout,
    )
    return adapter, client


_SAMPLE_MESSAGES: tuple[dict[str, Any], ...] = (
    {"role": "system", "content": "You are a test runner."},
    {"role": "user", "content": "run the smoke tests"},
)


# ---------------------------------------------------------------------------
# LLMCallErrorKind tests
# ---------------------------------------------------------------------------


class TestLLMCallErrorKind:
    """Tests for the error classification enum."""

    def test_transient_value(self) -> None:
        assert LLMCallErrorKind.TRANSIENT.value == "transient"

    def test_permanent_value(self) -> None:
        assert LLMCallErrorKind.PERMANENT.value == "permanent"


# ---------------------------------------------------------------------------
# LLMCallError tests
# ---------------------------------------------------------------------------


class TestLLMCallError:
    """Tests for the classified error type."""

    def test_is_transient(self) -> None:
        err = LLMCallError("net down", LLMCallErrorKind.TRANSIENT)
        assert err.is_transient is True
        assert err.is_permanent is False

    def test_is_permanent(self) -> None:
        err = LLMCallError("auth fail", LLMCallErrorKind.PERMANENT)
        assert err.is_permanent is True
        assert err.is_transient is False

    def test_preserves_cause(self) -> None:
        cause = ConnectionError("reset")
        err = LLMCallError("wrap", LLMCallErrorKind.TRANSIENT, cause=cause)
        assert err.cause is cause

    def test_preserves_attempts(self) -> None:
        err = LLMCallError("fail", LLMCallErrorKind.TRANSIENT, attempts=3)
        assert err.attempts == 3

    def test_default_attempts_is_one(self) -> None:
        err = LLMCallError("fail", LLMCallErrorKind.TRANSIENT)
        assert err.attempts == 1

    def test_inherits_from_exception(self) -> None:
        err = LLMCallError("x", LLMCallErrorKind.TRANSIENT)
        assert isinstance(err, Exception)

    def test_str_contains_message(self) -> None:
        err = LLMCallError("something broke", LLMCallErrorKind.PERMANENT)
        assert "something broke" in str(err)


# ---------------------------------------------------------------------------
# LLMCallResult tests
# ---------------------------------------------------------------------------


class TestLLMCallResult:
    """Tests for the immutable call result."""

    def test_frozen_dataclass(self) -> None:
        result = LLMCallResult(
            response=MagicMock(),
            elapsed_seconds=1.5,
            attempts=2,
            model="test-model",
        )
        with pytest.raises(AttributeError):
            result.attempts = 5  # type: ignore[misc]

    def test_stores_all_fields(self) -> None:
        mock_resp = MagicMock()
        result = LLMCallResult(
            response=mock_resp,
            elapsed_seconds=0.5,
            attempts=1,
            model="openai:conn:gpt-4",
        )
        assert result.response is mock_resp
        assert result.elapsed_seconds == 0.5
        assert result.attempts == 1
        assert result.model == "openai:conn:gpt-4"


# ---------------------------------------------------------------------------
# classify_sdk_error tests
# ---------------------------------------------------------------------------


class TestClassifySdkError:
    """Tests for the error classification function."""

    def test_connection_error_is_transient(self) -> None:
        assert classify_sdk_error(ConnectionError("down")) is LLMCallErrorKind.TRANSIENT

    def test_timeout_error_is_transient(self) -> None:
        assert classify_sdk_error(TimeoutError("slow")) is LLMCallErrorKind.TRANSIENT

    def test_os_error_is_transient(self) -> None:
        assert classify_sdk_error(OSError("ECONNRESET")) is LLMCallErrorKind.TRANSIENT

    def test_asyncio_timeout_is_transient(self) -> None:
        assert classify_sdk_error(asyncio.TimeoutError()) is LLMCallErrorKind.TRANSIENT

    def test_value_error_is_permanent(self) -> None:
        assert classify_sdk_error(ValueError("bad")) is LLMCallErrorKind.PERMANENT

    def test_runtime_error_is_permanent(self) -> None:
        assert classify_sdk_error(RuntimeError("oops")) is LLMCallErrorKind.PERMANENT

    def test_status_code_429_is_transient(self) -> None:
        exc = MagicMock(spec=Exception)
        exc.status_code = 429
        # Need to make isinstance check work
        real_exc = Exception("rate limited")
        real_exc.status_code = 429  # type: ignore[attr-defined]
        assert classify_sdk_error(real_exc) is LLMCallErrorKind.TRANSIENT

    def test_status_code_500_is_transient(self) -> None:
        exc = Exception("server error")
        exc.status_code = 500  # type: ignore[attr-defined]
        assert classify_sdk_error(exc) is LLMCallErrorKind.TRANSIENT

    def test_status_code_502_is_transient(self) -> None:
        exc = Exception("bad gateway")
        exc.status_code = 502  # type: ignore[attr-defined]
        assert classify_sdk_error(exc) is LLMCallErrorKind.TRANSIENT

    def test_status_code_503_is_transient(self) -> None:
        exc = Exception("unavailable")
        exc.status_code = 503  # type: ignore[attr-defined]
        assert classify_sdk_error(exc) is LLMCallErrorKind.TRANSIENT

    def test_status_code_504_is_transient(self) -> None:
        exc = Exception("gateway timeout")
        exc.status_code = 504  # type: ignore[attr-defined]
        assert classify_sdk_error(exc) is LLMCallErrorKind.TRANSIENT

    def test_status_code_400_is_permanent(self) -> None:
        exc = Exception("bad request")
        exc.status_code = 400  # type: ignore[attr-defined]
        assert classify_sdk_error(exc) is LLMCallErrorKind.PERMANENT

    def test_status_code_401_is_permanent(self) -> None:
        exc = Exception("unauthorized")
        exc.status_code = 401  # type: ignore[attr-defined]
        assert classify_sdk_error(exc) is LLMCallErrorKind.PERMANENT

    def test_status_code_403_is_permanent(self) -> None:
        exc = Exception("forbidden")
        exc.status_code = 403  # type: ignore[attr-defined]
        assert classify_sdk_error(exc) is LLMCallErrorKind.PERMANENT

    def test_openai_api_connection_error_is_transient(self) -> None:
        import openai

        exc = openai.APIConnectionError(request=MagicMock())
        assert classify_sdk_error(exc) is LLMCallErrorKind.TRANSIENT

    def test_openai_rate_limit_error_is_transient(self) -> None:
        import openai

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        exc = openai.RateLimitError(
            message="rate limited",
            response=mock_response,
            body=None,
        )
        assert classify_sdk_error(exc) is LLMCallErrorKind.TRANSIENT

    def test_openai_auth_error_is_permanent(self) -> None:
        import openai

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        exc = openai.AuthenticationError(
            message="bad key",
            response=mock_response,
            body=None,
        )
        assert classify_sdk_error(exc) is LLMCallErrorKind.PERMANENT


# ---------------------------------------------------------------------------
# call_completion: success cases
# ---------------------------------------------------------------------------


class TestCallCompletionSuccess:
    """Tests for successful call_completion invocations."""

    @pytest.mark.asyncio
    async def test_returns_llm_call_result(self) -> None:
        """Successful call returns an LLMCallResult."""
        adapter, client = _make_adapter()
        mock_resp = _make_mock_response()
        client.chat.completions.create.return_value = mock_resp

        result = await adapter.call_completion(_SAMPLE_MESSAGES)

        assert isinstance(result, LLMCallResult)
        assert result.response is mock_resp
        assert result.model == "test:conn:gpt-4"

    @pytest.mark.asyncio
    async def test_attempts_one_on_first_success(self) -> None:
        """When first attempt succeeds, attempts=1."""
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_mock_response()

        result = await adapter.call_completion(_SAMPLE_MESSAGES)

        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_elapsed_seconds_populated(self) -> None:
        """Result includes positive elapsed time."""
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_mock_response()

        result = await adapter.call_completion(_SAMPLE_MESSAGES)

        assert result.elapsed_seconds >= 0

    @pytest.mark.asyncio
    async def test_messages_forwarded_to_sdk(self) -> None:
        """Messages are passed to the SDK call."""
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_mock_response()

        await adapter.call_completion(_SAMPLE_MESSAGES)

        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["messages"] == list(_SAMPLE_MESSAGES)

    @pytest.mark.asyncio
    async def test_model_forwarded_to_sdk(self) -> None:
        """Model identifier is passed to the SDK call."""
        adapter, client = _make_adapter(model="openai:conn:gpt-4o")
        client.chat.completions.create.return_value = _make_mock_response()

        await adapter.call_completion(_SAMPLE_MESSAGES)

        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "openai:conn:gpt-4o"

    @pytest.mark.asyncio
    async def test_tool_schemas_forwarded_to_sdk(self) -> None:
        """Tool schemas are included in the SDK call."""
        schemas = (
            {"type": "function", "function": {"name": "read_wiki"}},
        )
        adapter, client = _make_adapter(tool_schemas=schemas)
        client.chat.completions.create.return_value = _make_mock_response()

        await adapter.call_completion(_SAMPLE_MESSAGES)

        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["tools"] == list(schemas)

    @pytest.mark.asyncio
    async def test_empty_schemas_not_passed(self) -> None:
        """Empty tool schemas are not included in the SDK call."""
        adapter, client = _make_adapter(tool_schemas=())
        client.chat.completions.create.return_value = _make_mock_response()

        await adapter.call_completion(_SAMPLE_MESSAGES)

        call_kwargs = client.chat.completions.create.call_args
        assert "tools" not in call_kwargs.kwargs


# ---------------------------------------------------------------------------
# call_completion: retry behavior
# ---------------------------------------------------------------------------


class TestCallCompletionRetry:
    """Tests for transient error retry logic in call_completion."""

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self) -> None:
        """ConnectionError triggers retry and succeeds on second attempt."""
        adapter, client = _make_adapter(default_max_retries=2)
        mock_resp = _make_mock_response()
        client.chat.completions.create.side_effect = [
            ConnectionError("reset"),
            mock_resp,
        ]

        result = await adapter.call_completion(_SAMPLE_MESSAGES)

        assert result.attempts == 2
        assert result.response is mock_resp

    @pytest.mark.asyncio
    async def test_retries_on_timeout_error(self) -> None:
        """TimeoutError triggers retry and succeeds on third attempt."""
        adapter, client = _make_adapter(default_max_retries=2)
        mock_resp = _make_mock_response()
        client.chat.completions.create.side_effect = [
            TimeoutError("slow"),
            TimeoutError("still slow"),
            mock_resp,
        ]

        result = await adapter.call_completion(_SAMPLE_MESSAGES)

        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_retries_on_os_error(self) -> None:
        """OSError triggers retry."""
        adapter, client = _make_adapter(default_max_retries=1)
        mock_resp = _make_mock_response()
        client.chat.completions.create.side_effect = [
            OSError("ECONNRESET"),
            mock_resp,
        ]

        result = await adapter.call_completion(_SAMPLE_MESSAGES)

        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_status(self) -> None:
        """Exception with status_code=429 triggers retry."""
        adapter, client = _make_adapter(default_max_retries=1)
        mock_resp = _make_mock_response()
        rate_err = Exception("rate limited")
        rate_err.status_code = 429  # type: ignore[attr-defined]
        client.chat.completions.create.side_effect = [
            rate_err,
            mock_resp,
        ]

        result = await adapter.call_completion(_SAMPLE_MESSAGES)

        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises_transient(self) -> None:
        """All retries exhausted raises LLMCallError with TRANSIENT kind."""
        adapter, client = _make_adapter(default_max_retries=2)
        client.chat.completions.create.side_effect = ConnectionError("down")

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(_SAMPLE_MESSAGES)

        assert exc_info.value.is_transient
        assert exc_info.value.attempts == 3  # 1 initial + 2 retries
        assert exc_info.value.cause is not None

    @pytest.mark.asyncio
    async def test_zero_retries_fails_on_first_transient(self) -> None:
        """max_retries=0 means only one attempt (no retries)."""
        adapter, client = _make_adapter(default_max_retries=0)
        client.chat.completions.create.side_effect = ConnectionError("down")

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(_SAMPLE_MESSAGES)

        assert exc_info.value.attempts == 1

    @pytest.mark.asyncio
    async def test_override_max_retries(self) -> None:
        """max_retries parameter overrides the default."""
        adapter, client = _make_adapter(default_max_retries=2)
        client.chat.completions.create.side_effect = ConnectionError("down")

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(_SAMPLE_MESSAGES, max_retries=1)

        assert exc_info.value.attempts == 2  # 1 initial + 1 retry

    @pytest.mark.asyncio
    async def test_sdk_called_correct_number_of_times(self) -> None:
        """SDK is called exactly max_retries + 1 times on persistent failure."""
        adapter, client = _make_adapter(default_max_retries=2)
        client.chat.completions.create.side_effect = ConnectionError("down")

        with pytest.raises(LLMCallError):
            await adapter.call_completion(_SAMPLE_MESSAGES)

        assert client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# call_completion: permanent error behavior
# ---------------------------------------------------------------------------


class TestCallCompletionPermanentErrors:
    """Tests for permanent error handling in call_completion."""

    @pytest.mark.asyncio
    async def test_permanent_error_no_retry(self) -> None:
        """Permanent errors raise immediately without retrying."""
        adapter, client = _make_adapter(default_max_retries=2)
        client.chat.completions.create.side_effect = ValueError("bad request")

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(_SAMPLE_MESSAGES)

        assert exc_info.value.is_permanent
        assert exc_info.value.attempts == 1
        assert client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_auth_error_is_permanent(self) -> None:
        """Authentication errors are permanent and not retried."""
        import openai

        adapter, client = _make_adapter(default_max_retries=2)
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="bad key",
            response=mock_response,
            body=None,
        )

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(_SAMPLE_MESSAGES)

        assert exc_info.value.is_permanent
        assert client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_permanent_error_preserves_cause(self) -> None:
        """Permanent LLMCallError wraps the original exception."""
        adapter, client = _make_adapter()
        original = RuntimeError("bad model config")
        client.chat.completions.create.side_effect = original

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(_SAMPLE_MESSAGES)

        assert exc_info.value.cause is original

    @pytest.mark.asyncio
    async def test_permanent_error_chains_exception(self) -> None:
        """Permanent LLMCallError uses proper exception chaining."""
        adapter, client = _make_adapter()
        client.chat.completions.create.side_effect = RuntimeError("chain me")

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(_SAMPLE_MESSAGES)

        assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# call_completion: timeout enforcement
# ---------------------------------------------------------------------------


class TestCallCompletionTimeout:
    """Tests for per-attempt timeout enforcement."""

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self) -> None:
        """asyncio.TimeoutError from wait_for triggers retry."""
        adapter, client = _make_adapter(
            default_max_retries=1,
            default_timeout=0.01,
        )
        mock_resp = _make_mock_response()

        call_count = 0

        def slow_then_fast(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                import time
                time.sleep(0.1)  # Exceeds 10ms timeout
            return mock_resp

        client.chat.completions.create.side_effect = slow_then_fast

        result = await adapter.call_completion(_SAMPLE_MESSAGES)

        assert result.attempts == 2
        assert result.response is mock_resp

    @pytest.mark.asyncio
    async def test_no_timeout_when_none(self) -> None:
        """When timeout is None, no asyncio.wait_for wrapping is used."""
        adapter, client = _make_adapter(default_timeout=None)
        client.chat.completions.create.return_value = _make_mock_response()

        result = await adapter.call_completion(_SAMPLE_MESSAGES)

        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_override_timeout(self) -> None:
        """timeout parameter overrides the default."""
        adapter, client = _make_adapter(default_timeout=10.0)
        mock_resp = _make_mock_response()

        call_count = 0

        def slow_call(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                import time
                time.sleep(0.1)
            return mock_resp

        client.chat.completions.create.side_effect = slow_call

        # Override with very short timeout to trigger timeout
        result = await adapter.call_completion(
            _SAMPLE_MESSAGES, timeout=0.01, max_retries=1
        )

        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_all_timeout_retries_exhausted(self) -> None:
        """All attempts timing out raises LLMCallError."""
        adapter, client = _make_adapter(
            default_max_retries=1,
            default_timeout=0.01,
        )

        def always_slow(**kwargs: Any) -> MagicMock:
            import time
            time.sleep(0.5)
            return _make_mock_response()

        client.chat.completions.create.side_effect = always_slow

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(_SAMPLE_MESSAGES)

        assert exc_info.value.is_transient
        assert exc_info.value.attempts == 2  # 1 initial + 1 retry


# ---------------------------------------------------------------------------
# call_completion: mixed error sequences
# ---------------------------------------------------------------------------


class TestCallCompletionMixedErrors:
    """Tests for sequences mixing transient and permanent errors."""

    @pytest.mark.asyncio
    async def test_transient_then_permanent_stops(self) -> None:
        """Transient error followed by permanent error stops immediately."""
        adapter, client = _make_adapter(default_max_retries=2)
        client.chat.completions.create.side_effect = [
            ConnectionError("blip"),
            ValueError("bad config"),
        ]

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(_SAMPLE_MESSAGES)

        assert exc_info.value.is_permanent
        assert exc_info.value.attempts == 2

    @pytest.mark.asyncio
    async def test_transient_then_success(self) -> None:
        """Transient error followed by success returns result."""
        adapter, client = _make_adapter(default_max_retries=2)
        mock_resp = _make_mock_response()
        client.chat.completions.create.side_effect = [
            ConnectionError("blip"),
            mock_resp,
        ]

        result = await adapter.call_completion(_SAMPLE_MESSAGES)

        assert result.attempts == 2
        assert result.response is mock_resp


# ---------------------------------------------------------------------------
# get_tool_calls backward compatibility
# ---------------------------------------------------------------------------


class TestGetToolCallsBackwardCompat:
    """Verify get_tool_calls still works after call_completion addition."""

    @pytest.mark.asyncio
    async def test_returns_parsed_tool_calls(self) -> None:
        adapter, client = _make_adapter()
        response = _make_mock_response(
            tool_calls=[{
                "id": "call_001",
                "function": {
                    "name": "read_wiki",
                    "arguments": json.dumps({"slug": "test"}),
                },
            }]
        )
        client.chat.completions.create.return_value = response

        calls = await adapter.get_tool_calls(_SAMPLE_MESSAGES)

        assert len(calls) == 1
        assert calls[0].tool_name == "read_wiki"

    @pytest.mark.asyncio
    async def test_empty_on_no_tool_calls(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.return_value = _make_mock_response()

        calls = await adapter.get_tool_calls(_SAMPLE_MESSAGES)

        assert calls == ()

    @pytest.mark.asyncio
    async def test_connection_error_propagates(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.side_effect = ConnectionError("net")

        with pytest.raises(ConnectionError):
            await adapter.get_tool_calls(_SAMPLE_MESSAGES)

    @pytest.mark.asyncio
    async def test_timeout_error_propagates(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.side_effect = TimeoutError("slow")

        with pytest.raises(TimeoutError):
            await adapter.get_tool_calls(_SAMPLE_MESSAGES)

    @pytest.mark.asyncio
    async def test_other_error_becomes_value_error(self) -> None:
        adapter, client = _make_adapter()
        client.chat.completions.create.side_effect = RuntimeError("oops")

        with pytest.raises(ValueError, match="LLM call failed"):
            await adapter.get_tool_calls(_SAMPLE_MESSAGES)


# ---------------------------------------------------------------------------
# Constructor parameter defaults
# ---------------------------------------------------------------------------


class TestAdapterConstructorDefaults:
    """Tests for adapter constructor parameter defaults."""

    def test_default_max_retries(self) -> None:
        adapter = OpenAILLMAdapter(
            client=MagicMock(),
            model="test-model",
            tool_schemas=(),
        )
        # Default should be 2 (matching AgentLoopConfig.max_retries)
        assert adapter._default_max_retries == 2

    def test_default_timeout_is_none(self) -> None:
        adapter = OpenAILLMAdapter(
            client=MagicMock(),
            model="test-model",
            tool_schemas=(),
        )
        assert adapter._default_timeout is None

    def test_custom_max_retries(self) -> None:
        adapter = OpenAILLMAdapter(
            client=MagicMock(),
            model="test-model",
            tool_schemas=(),
            default_max_retries=5,
        )
        assert adapter._default_max_retries == 5

    def test_custom_timeout(self) -> None:
        adapter = OpenAILLMAdapter(
            client=MagicMock(),
            model="test-model",
            tool_schemas=(),
            default_timeout=30.0,
        )
        assert adapter._default_timeout == 30.0
