"""Tests for the OpenAI LLM adapter (agent/llm_adapter.py).

Validates that the adapter correctly bridges between the OpenAI SDK
and the AgentLoop LLMClient protocol:
- Parses tool calls from completion responses
- Handles empty responses (signals completion)
- Classifies transient vs permanent errors
- Offloads blocking calls to thread pool
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from jules_daemon.agent.llm_adapter import OpenAILLMAdapter
from jules_daemon.agent.tool_types import ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completion_response(
    tool_calls: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a mock OpenAI completion response."""
    response = MagicMock()
    message = MagicMock()

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

    response.choices = [MagicMock(message=message)]
    return response


def _make_empty_response() -> MagicMock:
    """Build a mock response with no choices."""
    response = MagicMock()
    response.choices = []
    return response


# ---------------------------------------------------------------------------
# Parse tool calls
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    """Tests for _parse_tool_calls static method."""

    def test_parses_single_tool_call(self) -> None:
        response = _make_completion_response(
            tool_calls=[{
                "id": "call_001",
                "function": {
                    "name": "read_wiki",
                    "arguments": json.dumps({"slug": "test-spec"}),
                },
            }]
        )

        calls = OpenAILLMAdapter._parse_tool_calls(response)

        assert len(calls) == 1
        assert calls[0].call_id == "call_001"
        assert calls[0].tool_name == "read_wiki"
        assert calls[0].arguments == {"slug": "test-spec"}

    def test_parses_multiple_tool_calls(self) -> None:
        response = _make_completion_response(
            tool_calls=[
                {
                    "id": "call_001",
                    "function": {
                        "name": "read_wiki",
                        "arguments": json.dumps({"slug": "test-a"}),
                    },
                },
                {
                    "id": "call_002",
                    "function": {
                        "name": "lookup_test_spec",
                        "arguments": json.dumps({"test_name": "smoke"}),
                    },
                },
            ]
        )

        calls = OpenAILLMAdapter._parse_tool_calls(response)

        assert len(calls) == 2
        assert calls[0].tool_name == "read_wiki"
        assert calls[1].tool_name == "lookup_test_spec"

    def test_no_tool_calls_returns_empty(self) -> None:
        response = _make_completion_response(tool_calls=None)
        calls = OpenAILLMAdapter._parse_tool_calls(response)
        assert calls == ()

    def test_empty_choices_returns_empty(self) -> None:
        response = _make_empty_response()
        calls = OpenAILLMAdapter._parse_tool_calls(response)
        assert calls == ()

    def test_malformed_arguments_uses_empty_dict(self) -> None:
        """Invalid JSON in arguments defaults to empty dict."""
        response = _make_completion_response(
            tool_calls=[{
                "id": "call_001",
                "function": {
                    "name": "read_wiki",
                    "arguments": "not valid json {{{",
                },
            }]
        )

        calls = OpenAILLMAdapter._parse_tool_calls(response)

        assert len(calls) == 1
        assert calls[0].arguments == {}

    def test_empty_arguments_string(self) -> None:
        """Empty arguments string defaults to empty dict."""
        response = _make_completion_response(
            tool_calls=[{
                "id": "call_001",
                "function": {
                    "name": "notify_user",
                    "arguments": "",
                },
            }]
        )

        calls = OpenAILLMAdapter._parse_tool_calls(response)

        assert len(calls) == 1
        assert calls[0].arguments == {}


# ---------------------------------------------------------------------------
# get_tool_calls (async, with mocked SDK)
# ---------------------------------------------------------------------------


class TestGetToolCalls:
    """Tests for the async get_tool_calls method."""

    @pytest.mark.asyncio
    async def test_calls_sdk_and_parses_response(self) -> None:
        """Adapter calls SDK and returns parsed tool calls."""
        mock_client = MagicMock()
        response = _make_completion_response(
            tool_calls=[{
                "id": "call_001",
                "function": {
                    "name": "read_wiki",
                    "arguments": json.dumps({"slug": "tests"}),
                },
            }]
        )
        mock_client.chat.completions.create.return_value = response

        adapter = OpenAILLMAdapter(
            client=mock_client,
            model="test-model",
            tool_schemas=(),
        )

        messages = (
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "run the tests"},
        )

        calls = await adapter.get_tool_calls(messages)

        assert len(calls) == 1
        assert calls[0].tool_name == "read_wiki"
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_tool_schemas(self) -> None:
        """Adapter passes tool schemas to the SDK."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_completion_response(tool_calls=None)
        )

        schemas = (
            {"type": "function", "function": {"name": "read_wiki"}},
        )
        adapter = OpenAILLMAdapter(
            client=mock_client,
            model="test-model",
            tool_schemas=schemas,
        )

        await adapter.get_tool_calls(
            ({"role": "user", "content": "test"},),
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("tools") is not None

    @pytest.mark.asyncio
    async def test_connection_error_propagates(self) -> None:
        """ConnectionError from SDK propagates as transient error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = (
            ConnectionError("network down")
        )

        adapter = OpenAILLMAdapter(
            client=mock_client,
            model="test-model",
            tool_schemas=(),
        )

        with pytest.raises(ConnectionError):
            await adapter.get_tool_calls(
                ({"role": "user", "content": "test"},),
            )

    @pytest.mark.asyncio
    async def test_timeout_error_propagates(self) -> None:
        """TimeoutError from SDK propagates as transient error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = (
            TimeoutError("request timeout")
        )

        adapter = OpenAILLMAdapter(
            client=mock_client,
            model="test-model",
            tool_schemas=(),
        )

        with pytest.raises(TimeoutError):
            await adapter.get_tool_calls(
                ({"role": "user", "content": "test"},),
            )

    @pytest.mark.asyncio
    async def test_other_error_becomes_value_error(self) -> None:
        """Non-transient SDK errors are wrapped as ValueError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = (
            RuntimeError("bad request")
        )

        adapter = OpenAILLMAdapter(
            client=mock_client,
            model="test-model",
            tool_schemas=(),
        )

        with pytest.raises(ValueError, match="LLM call failed"):
            await adapter.get_tool_calls(
                ({"role": "user", "content": "test"},),
            )

    @pytest.mark.asyncio
    async def test_no_schemas_passes_none_for_tools(self) -> None:
        """When no schemas are provided, tools kwarg is None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_completion_response(tool_calls=None)
        )

        adapter = OpenAILLMAdapter(
            client=mock_client,
            model="test-model",
            tool_schemas=(),
        )

        await adapter.get_tool_calls(
            ({"role": "user", "content": "test"},),
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        # Empty schemas list -> tools is None (not passed)
        assert call_kwargs.kwargs.get("tools") is None
