"""Coverage-focused tests for jules_daemon.agent.llm_adapter.

Targets uncovered lines identified by coverage analysis:
- LLMCallError.__init__ (107-110)
- LLMCallError.is_transient / is_permanent properties (115, 120)
- classify_sdk_error function (190-212)
- OpenAILLMAdapter.__init__ (260-264)
- call_completion retry loop (311-386)
- get_tool_calls high-level method (418-430)
- _execute_with_timeout with/without timeout (455-460)
- _call_llm synchronous SDK call (474-481)
- _parse_tool_calls with malformed JSON (497-528)
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
from jules_daemon.agent.tool_types import ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(
    tool_calls: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response.

    Args:
        tool_calls: List of dicts with 'id', 'function.name',
            'function.arguments' keys. None means no tool calls on
            the message.

    Returns:
        MagicMock mimicking an OpenAI ChatCompletion response.
    """
    msg = MagicMock()
    if tool_calls is None:
        msg.tool_calls = None
    else:
        mock_tcs = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc["id"]
            mock_tc.function.name = tc["function"]["name"]
            mock_tc.function.arguments = tc["function"]["arguments"]
            mock_tcs.append(mock_tc)
        msg.tool_calls = mock_tcs

    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_empty_choices_response() -> MagicMock:
    """Build a mock response with an empty choices list."""
    resp = MagicMock()
    resp.choices = []
    return resp


def _make_adapter(
    *,
    client: Any = None,
    model: str = "test-model",
    tool_schemas: tuple[dict[str, Any], ...] = (),
    default_max_retries: int = 2,
    default_timeout: float | None = None,
) -> OpenAILLMAdapter:
    """Create an OpenAILLMAdapter with sensible defaults for testing."""
    if client is None:
        client = MagicMock()
    return OpenAILLMAdapter(
        client=client,
        model=model,
        tool_schemas=tool_schemas,
        default_max_retries=default_max_retries,
        default_timeout=default_timeout,
    )


# ---------------------------------------------------------------------------
# LLMCallError (lines 107-110, 115, 120)
# ---------------------------------------------------------------------------


class TestLLMCallError:
    """Tests for LLMCallError.__init__, is_transient, is_permanent."""

    def test_init_stores_all_fields(self) -> None:
        cause = ValueError("underlying")
        err = LLMCallError(
            "something went wrong",
            LLMCallErrorKind.TRANSIENT,
            cause=cause,
            attempts=3,
        )

        assert str(err) == "something went wrong"
        assert err.kind is LLMCallErrorKind.TRANSIENT
        assert err.cause is cause
        assert err.attempts == 3

    def test_init_defaults(self) -> None:
        err = LLMCallError("fail", LLMCallErrorKind.PERMANENT)

        assert err.cause is None
        assert err.attempts == 1

    def test_is_transient_true(self) -> None:
        err = LLMCallError("net", LLMCallErrorKind.TRANSIENT)
        assert err.is_transient is True

    def test_is_transient_false_for_permanent(self) -> None:
        err = LLMCallError("auth", LLMCallErrorKind.PERMANENT)
        assert err.is_transient is False

    def test_is_permanent_true(self) -> None:
        err = LLMCallError("auth", LLMCallErrorKind.PERMANENT)
        assert err.is_permanent is True

    def test_is_permanent_false_for_transient(self) -> None:
        err = LLMCallError("net", LLMCallErrorKind.TRANSIENT)
        assert err.is_permanent is False

    def test_is_exception_subclass(self) -> None:
        err = LLMCallError("oops", LLMCallErrorKind.TRANSIENT)
        assert isinstance(err, Exception)

    def test_kind_preserved_across_raise(self) -> None:
        with pytest.raises(LLMCallError) as exc_info:
            raise LLMCallError(
                "retry me",
                LLMCallErrorKind.TRANSIENT,
                cause=ConnectionError("gone"),
                attempts=2,
            )

        caught = exc_info.value
        assert caught.is_transient is True
        assert caught.attempts == 2
        assert isinstance(caught.cause, ConnectionError)


# ---------------------------------------------------------------------------
# classify_sdk_error (lines 190-212)
# ---------------------------------------------------------------------------


class TestClassifySdkError:
    """Tests for classify_sdk_error covering all classification paths."""

    # -- Built-in transient errors (line 190-191) --------------------------

    def test_connection_error_is_transient(self) -> None:
        result = classify_sdk_error(ConnectionError("refused"))
        assert result is LLMCallErrorKind.TRANSIENT

    def test_timeout_error_is_transient(self) -> None:
        result = classify_sdk_error(TimeoutError("timed out"))
        assert result is LLMCallErrorKind.TRANSIENT

    def test_os_error_is_transient(self) -> None:
        result = classify_sdk_error(OSError("network unreachable"))
        assert result is LLMCallErrorKind.TRANSIENT

    # -- OpenAI SDK-specific errors (lines 194-204) ------------------------

    def test_openai_api_connection_error_is_transient(self) -> None:
        import openai

        exc = openai.APIConnectionError(request=MagicMock())
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.TRANSIENT

    def test_openai_api_timeout_error_is_transient(self) -> None:
        import openai

        exc = openai.APITimeoutError(request=MagicMock())
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.TRANSIENT

    def test_openai_rate_limit_error_is_transient(self) -> None:
        import openai

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.json.return_value = {
            "error": {"message": "rate limited", "type": "rate_limit"}
        }
        exc = openai.RateLimitError(
            message="rate limited",
            response=mock_response,
            body={"error": {"message": "rate limited"}},
        )
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.TRANSIENT

    # -- HTTP status code check (lines 207-209) ----------------------------

    def test_status_code_429_is_transient(self) -> None:
        exc = Exception("rate limit")
        exc.status_code = 429  # type: ignore[attr-defined]
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.TRANSIENT

    def test_status_code_500_is_transient(self) -> None:
        exc = Exception("internal server error")
        exc.status_code = 500  # type: ignore[attr-defined]
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.TRANSIENT

    def test_status_code_502_is_transient(self) -> None:
        exc = Exception("bad gateway")
        exc.status_code = 502  # type: ignore[attr-defined]
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.TRANSIENT

    def test_status_code_503_is_transient(self) -> None:
        exc = Exception("service unavailable")
        exc.status_code = 503  # type: ignore[attr-defined]
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.TRANSIENT

    def test_status_code_504_is_transient(self) -> None:
        exc = Exception("gateway timeout")
        exc.status_code = 504  # type: ignore[attr-defined]
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.TRANSIENT

    def test_status_code_408_is_transient(self) -> None:
        exc = Exception("request timeout")
        exc.status_code = 408  # type: ignore[attr-defined]
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.TRANSIENT

    # -- Non-transient status codes (line 211-212 default) -----------------

    def test_status_code_400_is_permanent(self) -> None:
        exc = Exception("bad request")
        exc.status_code = 400  # type: ignore[attr-defined]
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.PERMANENT

    def test_status_code_401_is_permanent(self) -> None:
        exc = Exception("unauthorized")
        exc.status_code = 401  # type: ignore[attr-defined]
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.PERMANENT

    def test_status_code_403_is_permanent(self) -> None:
        exc = Exception("forbidden")
        exc.status_code = 403  # type: ignore[attr-defined]
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.PERMANENT

    def test_status_code_404_is_permanent(self) -> None:
        exc = Exception("not found")
        exc.status_code = 404  # type: ignore[attr-defined]
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.PERMANENT

    # -- Default permanent (line 212) --------------------------------------

    def test_plain_exception_is_permanent(self) -> None:
        result = classify_sdk_error(Exception("unknown error"))
        assert result is LLMCallErrorKind.PERMANENT

    def test_value_error_is_permanent(self) -> None:
        result = classify_sdk_error(ValueError("bad input"))
        assert result is LLMCallErrorKind.PERMANENT

    def test_runtime_error_is_permanent(self) -> None:
        result = classify_sdk_error(RuntimeError("unexpected"))
        assert result is LLMCallErrorKind.PERMANENT

    def test_no_status_code_attribute_is_permanent(self) -> None:
        exc = TypeError("wrong type")
        assert not hasattr(exc, "status_code")
        result = classify_sdk_error(exc)
        assert result is LLMCallErrorKind.PERMANENT


# ---------------------------------------------------------------------------
# OpenAILLMAdapter.__init__ (lines 260-264)
# ---------------------------------------------------------------------------


class TestOpenAILLMAdapterInit:
    """Tests for OpenAILLMAdapter construction and stored attributes."""

    def test_stores_client_and_model(self) -> None:
        client = MagicMock()
        adapter = OpenAILLMAdapter(
            client=client,
            model="gpt-4o",
            tool_schemas=(),
        )

        assert adapter._client is client
        assert adapter._model == "gpt-4o"

    def test_stores_tool_schemas_as_list(self) -> None:
        schemas = (
            {"type": "function", "function": {"name": "read_wiki"}},
        )
        adapter = _make_adapter(tool_schemas=schemas)

        assert adapter._tool_schemas == list(schemas)
        assert isinstance(adapter._tool_schemas, list)

    def test_default_retries_and_timeout(self) -> None:
        adapter = _make_adapter()

        assert adapter._default_max_retries == 2
        assert adapter._default_timeout is None

    def test_custom_retries_and_timeout(self) -> None:
        adapter = OpenAILLMAdapter(
            client=MagicMock(),
            model="test-model",
            tool_schemas=(),
            default_max_retries=5,
            default_timeout=30.0,
        )

        assert adapter._default_max_retries == 5
        assert adapter._default_timeout == 30.0


# ---------------------------------------------------------------------------
# call_completion (lines 311-386)
# ---------------------------------------------------------------------------


class TestCallCompletion:
    """Tests for call_completion retry loop, timing, and error handling."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        mock_response = _make_mock_response()
        client = MagicMock()
        client.chat.completions.create.return_value = mock_response

        adapter = _make_adapter(client=client)
        messages = ({"role": "user", "content": "hello"},)

        result = await adapter.call_completion(messages)

        assert isinstance(result, LLMCallResult)
        assert result.response is mock_response
        assert result.attempts == 1
        assert result.model == "test-model"
        assert result.elapsed_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_uses_default_max_retries(self) -> None:
        """When max_retries is not specified, uses default_max_retries."""
        client = MagicMock()
        client.chat.completions.create.side_effect = ConnectionError("down")

        adapter = _make_adapter(client=client, default_max_retries=1)
        messages = ({"role": "user", "content": "test"},)

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(messages)

        # 1 retry + 1 initial = 2 total attempts
        assert exc_info.value.attempts == 2
        assert exc_info.value.is_transient is True

    @pytest.mark.asyncio
    async def test_explicit_max_retries_overrides_default(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = ConnectionError("down")

        adapter = _make_adapter(client=client, default_max_retries=5)
        messages = ({"role": "user", "content": "test"},)

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(messages, max_retries=0)

        # 0 retries = 1 attempt total
        assert exc_info.value.attempts == 1

    @pytest.mark.asyncio
    async def test_transient_error_retried_then_succeeds(self) -> None:
        mock_response = _make_mock_response()
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            ConnectionError("blip"),
            mock_response,
        ]

        adapter = _make_adapter(client=client, default_max_retries=2)
        messages = ({"role": "user", "content": "test"},)

        result = await adapter.call_completion(messages)

        assert result.attempts == 2
        assert result.response is mock_response

    @pytest.mark.asyncio
    async def test_transient_error_exhausts_all_retries(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = ConnectionError("down")

        adapter = _make_adapter(client=client, default_max_retries=2)
        messages = ({"role": "user", "content": "test"},)

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(messages)

        err = exc_info.value
        assert err.is_transient is True
        assert err.attempts == 3  # 2 retries + 1 initial
        assert isinstance(err.cause, ConnectionError)
        assert "failed after 3 attempts" in str(err)

    @pytest.mark.asyncio
    async def test_permanent_error_raises_immediately(self) -> None:
        """Permanent errors stop the retry loop on the first occurrence."""
        client = MagicMock()
        client.chat.completions.create.side_effect = ValueError("bad model")

        adapter = _make_adapter(client=client, default_max_retries=5)
        messages = ({"role": "user", "content": "test"},)

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(messages)

        err = exc_info.value
        assert err.is_permanent is True
        assert err.attempts == 1
        assert isinstance(err.cause, ValueError)
        assert "Permanent LLM error" in str(err)

    @pytest.mark.asyncio
    async def test_permanent_error_after_transient_retries(self) -> None:
        """A permanent error on attempt 2 stops further retries."""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            ConnectionError("blip"),
            ValueError("bad request"),
        ]

        adapter = _make_adapter(client=client, default_max_retries=5)
        messages = ({"role": "user", "content": "test"},)

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(messages)

        err = exc_info.value
        assert err.is_permanent is True
        assert err.attempts == 2

    @pytest.mark.asyncio
    async def test_timeout_error_is_retried(self) -> None:
        """asyncio.TimeoutError in the retry loop is treated as transient."""
        mock_response = _make_mock_response()
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            asyncio.TimeoutError(),
            mock_response,
        ]

        adapter = _make_adapter(client=client, default_max_retries=2)
        messages = ({"role": "user", "content": "test"},)

        result = await adapter.call_completion(messages)
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_timeout_error_exhausts_retries(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = asyncio.TimeoutError()

        adapter = _make_adapter(client=client, default_max_retries=1)
        messages = ({"role": "user", "content": "test"},)

        with pytest.raises(LLMCallError) as exc_info:
            await adapter.call_completion(messages)

        err = exc_info.value
        assert err.is_transient is True
        assert err.attempts == 2

    @pytest.mark.asyncio
    async def test_elapsed_seconds_is_positive(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _make_mock_response()

        adapter = _make_adapter(client=client)
        messages = ({"role": "user", "content": "test"},)

        result = await adapter.call_completion(messages)
        assert result.elapsed_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_explicit_timeout_parameter(self) -> None:
        """Explicit timeout=None overrides default_timeout."""
        client = MagicMock()
        client.chat.completions.create.return_value = _make_mock_response()

        adapter = _make_adapter(
            client=client, default_timeout=10.0,
        )
        messages = ({"role": "user", "content": "test"},)

        # timeout=None disables timeout enforcement
        result = await adapter.call_completion(messages, timeout=None)
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_uses_default_timeout_when_not_specified(self) -> None:
        """When timeout is not passed (sentinel ...), uses default_timeout."""
        client = MagicMock()
        client.chat.completions.create.return_value = _make_mock_response()

        adapter = _make_adapter(client=client, default_timeout=60.0)
        messages = ({"role": "user", "content": "test"},)

        with patch.object(
            adapter, "_execute_with_timeout", wraps=adapter._execute_with_timeout
        ) as mock_exec:
            await adapter.call_completion(messages)
            # The timeout passed to _execute_with_timeout should be 60.0
            call_args = mock_exec.call_args
            assert call_args[0][1] == 60.0

    @pytest.mark.asyncio
    async def test_messages_converted_to_list(self) -> None:
        """Tuple messages are converted to a list before passing to SDK."""
        client = MagicMock()
        client.chat.completions.create.return_value = _make_mock_response()

        adapter = _make_adapter(client=client)
        messages = ({"role": "user", "content": "test"},)

        await adapter.call_completion(messages)

        call_args = client.chat.completions.create.call_args
        actual_messages = call_args.kwargs.get("messages", call_args[0][0] if call_args[0] else None)
        assert isinstance(actual_messages, list)


# ---------------------------------------------------------------------------
# get_tool_calls (lines 418-430)
# ---------------------------------------------------------------------------


class TestGetToolCallsHighLevel:
    """Tests for the high-level get_tool_calls method."""

    @pytest.mark.asyncio
    async def test_returns_parsed_tool_calls(self) -> None:
        response = _make_mock_response(
            tool_calls=[{
                "id": "call_abc",
                "function": {
                    "name": "read_wiki",
                    "arguments": json.dumps({"slug": "intro"}),
                },
            }]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response

        adapter = _make_adapter(client=client)
        messages = ({"role": "user", "content": "test"},)

        calls = await adapter.get_tool_calls(messages)

        assert len(calls) == 1
        assert isinstance(calls[0], ToolCall)
        assert calls[0].call_id == "call_abc"
        assert calls[0].tool_name == "read_wiki"
        assert calls[0].arguments == {"slug": "intro"}

    @pytest.mark.asyncio
    async def test_returns_empty_tuple_for_no_tool_calls(self) -> None:
        response = _make_mock_response(tool_calls=None)
        client = MagicMock()
        client.chat.completions.create.return_value = response

        adapter = _make_adapter(client=client)
        messages = ({"role": "user", "content": "done"},)

        calls = await adapter.get_tool_calls(messages)
        assert calls == ()

    @pytest.mark.asyncio
    async def test_connection_error_propagates(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = ConnectionError("down")

        adapter = _make_adapter(client=client)

        with pytest.raises(ConnectionError):
            await adapter.get_tool_calls(
                ({"role": "user", "content": "test"},),
            )

    @pytest.mark.asyncio
    async def test_timeout_error_propagates(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = TimeoutError("slow")

        adapter = _make_adapter(client=client)

        with pytest.raises(TimeoutError):
            await adapter.get_tool_calls(
                ({"role": "user", "content": "test"},),
            )

    @pytest.mark.asyncio
    async def test_os_error_propagates(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = OSError("network")

        adapter = _make_adapter(client=client)

        with pytest.raises(OSError):
            await adapter.get_tool_calls(
                ({"role": "user", "content": "test"},),
            )

    @pytest.mark.asyncio
    async def test_non_transient_error_wrapped_as_value_error(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("bad")

        adapter = _make_adapter(client=client)

        with pytest.raises(ValueError, match="LLM call failed"):
            await adapter.get_tool_calls(
                ({"role": "user", "content": "test"},),
            )

    @pytest.mark.asyncio
    async def test_converts_tuple_messages_to_list(self) -> None:
        """get_tool_calls converts the immutable tuple to a mutable list."""
        client = MagicMock()
        client.chat.completions.create.return_value = _make_mock_response()

        adapter = _make_adapter(client=client)
        messages = (
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "test"},
        )

        await adapter.get_tool_calls(messages)

        call_args = client.chat.completions.create.call_args
        passed_messages = call_args.kwargs.get("messages")
        assert isinstance(passed_messages, list)
        assert len(passed_messages) == 2


# ---------------------------------------------------------------------------
# _execute_with_timeout (lines 455-460)
# ---------------------------------------------------------------------------


class TestExecuteWithTimeout:
    """Tests for _execute_with_timeout with and without timeout."""

    @pytest.mark.asyncio
    async def test_without_timeout(self) -> None:
        """When timeout is None, runs the coro directly without wait_for."""
        mock_response = _make_mock_response()
        client = MagicMock()
        client.chat.completions.create.return_value = mock_response

        adapter = _make_adapter(client=client)
        messages = [{"role": "user", "content": "test"}]

        result = await adapter._execute_with_timeout(messages, timeout=None)
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_with_timeout_succeeds(self) -> None:
        """When timeout is set and call completes in time, returns response."""
        mock_response = _make_mock_response()
        client = MagicMock()
        client.chat.completions.create.return_value = mock_response

        adapter = _make_adapter(client=client)
        messages = [{"role": "user", "content": "test"}]

        result = await adapter._execute_with_timeout(messages, timeout=30.0)
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_with_timeout_raises_on_expiry(self) -> None:
        """When the call exceeds the timeout, raises asyncio.TimeoutError."""
        client = MagicMock()
        adapter = _make_adapter(client=client)

        async def _slow_to_thread(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(100)

        # Replace asyncio.to_thread with an async function that sleeps
        # longer than the timeout. asyncio.wait_for will cancel it.
        with patch(
            "jules_daemon.agent.llm_adapter.asyncio.to_thread",
            new=_slow_to_thread,
        ):
            messages = [{"role": "user", "content": "test"}]

            with pytest.raises(asyncio.TimeoutError):
                await adapter._execute_with_timeout(messages, timeout=0.01)


# ---------------------------------------------------------------------------
# _call_llm (lines 474-481)
# ---------------------------------------------------------------------------


class TestCallLLM:
    """Tests for the synchronous _call_llm method."""

    def test_calls_sdk_with_model_and_messages(self) -> None:
        client = MagicMock()
        mock_response = _make_mock_response()
        client.chat.completions.create.return_value = mock_response

        adapter = _make_adapter(client=client, model="gpt-4o-mini")
        messages = [{"role": "user", "content": "hello"}]

        result = adapter._call_llm(messages)

        assert result is mock_response
        client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",
            messages=messages,
        )

    def test_passes_tools_when_schemas_present(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _make_mock_response()

        schemas = (
            {"type": "function", "function": {"name": "read_wiki"}},
            {"type": "function", "function": {"name": "run_test"}},
        )
        adapter = _make_adapter(client=client, tool_schemas=schemas)
        messages = [{"role": "user", "content": "test"}]

        adapter._call_llm(messages)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 2

    def test_omits_tools_when_schemas_empty(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _make_mock_response()

        adapter = _make_adapter(client=client, tool_schemas=())
        messages = [{"role": "user", "content": "test"}]

        adapter._call_llm(messages)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert "tools" not in call_kwargs

    def test_propagates_sdk_exception(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("SDK error")

        adapter = _make_adapter(client=client)
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(RuntimeError, match="SDK error"):
            adapter._call_llm(messages)


# ---------------------------------------------------------------------------
# _parse_tool_calls (lines 497-528)
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    """Tests for _parse_tool_calls static method -- coverage-focused."""

    def test_empty_choices_returns_empty_tuple(self) -> None:
        resp = _make_empty_choices_response()
        result = OpenAILLMAdapter._parse_tool_calls(resp)
        assert result == ()

    def test_no_tool_calls_on_message_returns_empty_tuple(self) -> None:
        resp = _make_mock_response(tool_calls=None)
        result = OpenAILLMAdapter._parse_tool_calls(resp)
        assert result == ()

    def test_single_tool_call_parsed_correctly(self) -> None:
        resp = _make_mock_response(
            tool_calls=[{
                "id": "call_001",
                "function": {
                    "name": "read_wiki",
                    "arguments": json.dumps({"slug": "intro"}),
                },
            }]
        )

        result = OpenAILLMAdapter._parse_tool_calls(resp)

        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].call_id == "call_001"
        assert result[0].tool_name == "read_wiki"
        assert result[0].arguments == {"slug": "intro"}

    def test_multiple_tool_calls_parsed(self) -> None:
        resp = _make_mock_response(
            tool_calls=[
                {
                    "id": "call_001",
                    "function": {
                        "name": "read_wiki",
                        "arguments": json.dumps({"slug": "a"}),
                    },
                },
                {
                    "id": "call_002",
                    "function": {
                        "name": "run_test",
                        "arguments": json.dumps({"name": "smoke"}),
                    },
                },
            ]
        )

        result = OpenAILLMAdapter._parse_tool_calls(resp)

        assert len(result) == 2
        assert result[0].call_id == "call_001"
        assert result[1].call_id == "call_002"
        assert result[0].tool_name == "read_wiki"
        assert result[1].tool_name == "run_test"

    def test_malformed_json_arguments_defaults_to_empty_dict(self) -> None:
        resp = _make_mock_response(
            tool_calls=[{
                "id": "call_bad",
                "function": {
                    "name": "read_wiki",
                    "arguments": "{not valid json",
                },
            }]
        )

        result = OpenAILLMAdapter._parse_tool_calls(resp)

        assert len(result) == 1
        assert result[0].arguments == {}

    def test_empty_arguments_string_defaults_to_empty_dict(self) -> None:
        resp = _make_mock_response(
            tool_calls=[{
                "id": "call_empty",
                "function": {
                    "name": "notify",
                    "arguments": "",
                },
            }]
        )

        result = OpenAILLMAdapter._parse_tool_calls(resp)

        assert len(result) == 1
        assert result[0].arguments == {}

    def test_none_arguments_defaults_to_empty_dict(self) -> None:
        resp = _make_mock_response(
            tool_calls=[{
                "id": "call_none",
                "function": {
                    "name": "notify",
                    "arguments": None,
                },
            }]
        )

        result = OpenAILLMAdapter._parse_tool_calls(resp)

        assert len(result) == 1
        assert result[0].arguments == {}

    def test_returns_tuple_not_list(self) -> None:
        resp = _make_mock_response(
            tool_calls=[{
                "id": "call_001",
                "function": {
                    "name": "read_wiki",
                    "arguments": json.dumps({"slug": "test"}),
                },
            }]
        )

        result = OpenAILLMAdapter._parse_tool_calls(resp)
        assert isinstance(result, tuple)

    def test_complex_arguments_parsed(self) -> None:
        complex_args = {
            "path": "/home/user/file.txt",
            "lines": [1, 2, 3],
            "options": {"recursive": True, "depth": 5},
        }
        resp = _make_mock_response(
            tool_calls=[{
                "id": "call_complex",
                "function": {
                    "name": "file_op",
                    "arguments": json.dumps(complex_args),
                },
            }]
        )

        result = OpenAILLMAdapter._parse_tool_calls(resp)

        assert result[0].arguments == complex_args


# ---------------------------------------------------------------------------
# LLMCallResult (dataclass validation)
# ---------------------------------------------------------------------------


class TestLLMCallResult:
    """Tests for LLMCallResult frozen dataclass."""

    def test_stores_all_fields(self) -> None:
        resp = _make_mock_response()
        result = LLMCallResult(
            response=resp,
            elapsed_seconds=1.234,
            attempts=2,
            model="gpt-4o",
        )

        assert result.response is resp
        assert result.elapsed_seconds == 1.234
        assert result.attempts == 2
        assert result.model == "gpt-4o"

    def test_is_frozen(self) -> None:
        result = LLMCallResult(
            response=MagicMock(),
            elapsed_seconds=0.5,
            attempts=1,
            model="test",
        )

        with pytest.raises(AttributeError):
            result.attempts = 99  # type: ignore[misc]
