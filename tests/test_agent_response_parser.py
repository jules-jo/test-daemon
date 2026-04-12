"""Tests for the agent loop LLM response parser.

Validates the discriminated union parser that classifies raw LLM responses
as either tool-call requests or final-answer text. Covers:

    - ParsedResponse discriminated union (ToolCallsResponse vs FinalAnswerResponse)
    - Native mode: parsing OpenAI-style tool_calls from ChatCompletion responses
    - Prompt-based mode: parsing JSON tool_call blocks from text content
    - Edge cases: empty responses, malformed JSON, missing fields
    - Mixed content: tool calls with accompanying text
    - Multiple tool calls in a single response
    - Immutability of all result types
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock

import pytest

from jules_daemon.agent.response_parser import (
    FinalAnswerResponse,
    ParsedResponse,
    ResponseKind,
    ToolCallsResponse,
    parse_completion_response,
    parse_prompt_based_response,
)
from jules_daemon.agent.tool_types import ToolCall


# ---------------------------------------------------------------------------
# Helpers: mock ChatCompletion objects
# ---------------------------------------------------------------------------


def _make_tool_call_obj(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> MagicMock:
    """Build a mock OpenAI tool_call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _make_completion(
    content: str | None = None,
    tool_calls: list[MagicMock] | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    """Build a mock ChatCompletion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ---------------------------------------------------------------------------
# ResponseKind enum
# ---------------------------------------------------------------------------


class TestResponseKind:
    """Tests for the ResponseKind discriminator enum."""

    def test_members(self) -> None:
        assert set(ResponseKind) == {ResponseKind.TOOL_CALLS, ResponseKind.FINAL_ANSWER}

    def test_values(self) -> None:
        assert ResponseKind.TOOL_CALLS.value == "tool_calls"
        assert ResponseKind.FINAL_ANSWER.value == "final_answer"


# ---------------------------------------------------------------------------
# FinalAnswerResponse
# ---------------------------------------------------------------------------


class TestFinalAnswerResponse:
    """Tests for the FinalAnswerResponse frozen dataclass."""

    def test_create(self) -> None:
        resp = FinalAnswerResponse(text="The test passed successfully.")
        assert resp.kind is ResponseKind.FINAL_ANSWER
        assert resp.text == "The test passed successfully."

    def test_frozen(self) -> None:
        resp = FinalAnswerResponse(text="immutable")
        with pytest.raises(FrozenInstanceError):
            resp.text = "mutated"  # type: ignore[misc]

    def test_is_tool_calls_false(self) -> None:
        resp = FinalAnswerResponse(text="done")
        assert resp.is_tool_calls is False

    def test_is_final_answer_true(self) -> None:
        resp = FinalAnswerResponse(text="done")
        assert resp.is_final_answer is True

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="text must not be empty"):
            FinalAnswerResponse(text="")

    def test_whitespace_only_text_raises(self) -> None:
        with pytest.raises(ValueError, match="text must not be empty"):
            FinalAnswerResponse(text="   ")


# ---------------------------------------------------------------------------
# ToolCallsResponse
# ---------------------------------------------------------------------------


class TestToolCallsResponse:
    """Tests for the ToolCallsResponse frozen dataclass."""

    def test_create_single(self) -> None:
        call = ToolCall(call_id="c1", tool_name="read_wiki", arguments={"slug": "smoke"})
        resp = ToolCallsResponse(
            tool_calls=(call,),
            assistant_text=None,
        )
        assert resp.kind is ResponseKind.TOOL_CALLS
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].tool_name == "read_wiki"
        assert resp.assistant_text is None

    def test_create_multiple(self) -> None:
        calls = (
            ToolCall(call_id="c1", tool_name="read_wiki", arguments={"slug": "a"}),
            ToolCall(call_id="c2", tool_name="check_remote_processes", arguments={}),
        )
        resp = ToolCallsResponse(tool_calls=calls, assistant_text="Let me check")
        assert len(resp.tool_calls) == 2
        assert resp.assistant_text == "Let me check"

    def test_frozen(self) -> None:
        call = ToolCall(call_id="c1", tool_name="read_wiki", arguments={})
        resp = ToolCallsResponse(tool_calls=(call,), assistant_text=None)
        with pytest.raises(FrozenInstanceError):
            resp.tool_calls = ()  # type: ignore[misc]

    def test_is_tool_calls_true(self) -> None:
        call = ToolCall(call_id="c1", tool_name="read_wiki", arguments={})
        resp = ToolCallsResponse(tool_calls=(call,), assistant_text=None)
        assert resp.is_tool_calls is True

    def test_is_final_answer_false(self) -> None:
        call = ToolCall(call_id="c1", tool_name="read_wiki", arguments={})
        resp = ToolCallsResponse(tool_calls=(call,), assistant_text=None)
        assert resp.is_final_answer is False

    def test_empty_tool_calls_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_calls must not be empty"):
            ToolCallsResponse(tool_calls=(), assistant_text=None)


# ---------------------------------------------------------------------------
# parse_completion_response -- native mode
# ---------------------------------------------------------------------------


class TestParseCompletionResponseNative:
    """Tests for parse_completion_response with native tool calls."""

    def test_single_tool_call(self) -> None:
        tc = _make_tool_call_obj("call_001", "read_wiki", {"slug": "smoke-test"})
        completion = _make_completion(tool_calls=[tc], finish_reason="tool_calls")

        result = parse_completion_response(completion)

        assert isinstance(result, ToolCallsResponse)
        assert result.kind is ResponseKind.TOOL_CALLS
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].call_id == "call_001"
        assert result.tool_calls[0].tool_name == "read_wiki"
        assert result.tool_calls[0].arguments == {"slug": "smoke-test"}

    def test_multiple_tool_calls(self) -> None:
        tc1 = _make_tool_call_obj("c1", "read_wiki", {"slug": "smoke"})
        tc2 = _make_tool_call_obj("c2", "check_remote_processes", {"host": "staging"})
        completion = _make_completion(
            tool_calls=[tc1, tc2],
            finish_reason="tool_calls",
        )

        result = parse_completion_response(completion)

        assert isinstance(result, ToolCallsResponse)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "read_wiki"
        assert result.tool_calls[1].tool_name == "check_remote_processes"

    def test_tool_call_with_text(self) -> None:
        """Tool calls can come with accompanying text content."""
        tc = _make_tool_call_obj("c1", "read_wiki", {"slug": "test"})
        completion = _make_completion(
            content="Let me look that up for you.",
            tool_calls=[tc],
            finish_reason="tool_calls",
        )

        result = parse_completion_response(completion)

        assert isinstance(result, ToolCallsResponse)
        assert result.assistant_text == "Let me look that up for you."
        assert len(result.tool_calls) == 1

    def test_text_only_response(self) -> None:
        completion = _make_completion(
            content="The tests all passed. No further action needed.",
            tool_calls=None,
            finish_reason="stop",
        )

        result = parse_completion_response(completion)

        assert isinstance(result, FinalAnswerResponse)
        assert result.kind is ResponseKind.FINAL_ANSWER
        assert result.text == "The tests all passed. No further action needed."

    def test_empty_tool_calls_list(self) -> None:
        """An empty tool_calls list should be treated as final answer."""
        completion = _make_completion(
            content="Done.",
            tool_calls=[],
            finish_reason="stop",
        )

        result = parse_completion_response(completion)

        assert isinstance(result, FinalAnswerResponse)
        assert result.text == "Done."

    def test_no_choices_raises(self) -> None:
        completion = MagicMock()
        completion.choices = []

        with pytest.raises(ValueError, match="No choices"):
            parse_completion_response(completion)

    def test_null_content_no_tool_calls_raises(self) -> None:
        """When content is None and no tool calls, parser should raise."""
        completion = _make_completion(
            content=None,
            tool_calls=None,
            finish_reason="stop",
        )

        with pytest.raises(ValueError, match="empty"):
            parse_completion_response(completion)

    def test_malformed_arguments_defaults_to_empty(self) -> None:
        """Malformed JSON in tool_call arguments falls back to empty dict."""
        tc = MagicMock()
        tc.id = "c1"
        tc.type = "function"
        tc.function = MagicMock()
        tc.function.name = "read_wiki"
        tc.function.arguments = "not valid json {{"

        completion = _make_completion(tool_calls=[tc], finish_reason="tool_calls")

        result = parse_completion_response(completion)

        assert isinstance(result, ToolCallsResponse)
        assert result.tool_calls[0].arguments == {}

    def test_empty_arguments_string(self) -> None:
        """Empty arguments string falls back to empty dict."""
        tc = MagicMock()
        tc.id = "c1"
        tc.type = "function"
        tc.function = MagicMock()
        tc.function.name = "read_wiki"
        tc.function.arguments = ""

        completion = _make_completion(tool_calls=[tc], finish_reason="tool_calls")

        result = parse_completion_response(completion)

        assert isinstance(result, ToolCallsResponse)
        assert result.tool_calls[0].arguments == {}

    def test_null_arguments(self) -> None:
        """Null arguments string falls back to empty dict."""
        tc = MagicMock()
        tc.id = "c1"
        tc.type = "function"
        tc.function = MagicMock()
        tc.function.name = "read_wiki"
        tc.function.arguments = None

        completion = _make_completion(tool_calls=[tc], finish_reason="tool_calls")

        result = parse_completion_response(completion)

        assert isinstance(result, ToolCallsResponse)
        assert result.tool_calls[0].arguments == {}


# ---------------------------------------------------------------------------
# parse_prompt_based_response -- prompt-based tool calling
# ---------------------------------------------------------------------------


class TestParsePromptBasedResponse:
    """Tests for parse_prompt_based_response (prompt-based tool calling)."""

    def test_single_tool_call_json(self) -> None:
        text = json.dumps({
            "tool_calls": [
                {"name": "read_wiki", "arguments": {"slug": "smoke-test"}}
            ]
        })

        result = parse_prompt_based_response(text)

        assert isinstance(result, ToolCallsResponse)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "read_wiki"
        assert result.tool_calls[0].arguments == {"slug": "smoke-test"}

    def test_multiple_tool_calls_json(self) -> None:
        text = json.dumps({
            "tool_calls": [
                {"name": "read_wiki", "arguments": {"slug": "smoke"}},
                {"name": "check_remote_processes", "arguments": {"host": "staging"}},
            ]
        })

        result = parse_prompt_based_response(text)

        assert isinstance(result, ToolCallsResponse)
        assert len(result.tool_calls) == 2

    def test_tool_call_in_code_fence(self) -> None:
        text = (
            "I will look up the test spec first.\n\n"
            "```json\n"
            + json.dumps({
                "tool_calls": [
                    {"name": "lookup_test_spec", "arguments": {"test_name": "smoke"}}
                ]
            })
            + "\n```"
        )

        result = parse_prompt_based_response(text)

        assert isinstance(result, ToolCallsResponse)
        assert result.tool_calls[0].tool_name == "lookup_test_spec"

    def test_final_answer_plain_text(self) -> None:
        text = "All tests passed successfully. The smoke suite ran in 45 seconds."

        result = parse_prompt_based_response(text)

        assert isinstance(result, FinalAnswerResponse)
        assert "All tests passed" in result.text

    def test_json_without_tool_calls_key_is_final_answer(self) -> None:
        """JSON that doesn't contain a tool_calls key is treated as final answer."""
        text = json.dumps({"status": "complete", "summary": "All tests passed"})

        result = parse_prompt_based_response(text)

        assert isinstance(result, FinalAnswerResponse)

    def test_empty_tool_calls_array_is_final_answer(self) -> None:
        """Empty tool_calls array means the LLM has no more actions."""
        text = json.dumps({"tool_calls": []})

        result = parse_prompt_based_response(text)

        assert isinstance(result, FinalAnswerResponse)

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_prompt_based_response("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_prompt_based_response("   \n  ")

    def test_tool_call_missing_name_raises(self) -> None:
        """A tool call entry without a 'name' key should raise."""
        text = json.dumps({
            "tool_calls": [
                {"arguments": {"slug": "smoke"}}
            ]
        })

        with pytest.raises(ValueError, match="name"):
            parse_prompt_based_response(text)

    def test_tool_call_missing_arguments_defaults_to_empty(self) -> None:
        """Missing arguments in a tool call entry defaults to empty dict."""
        text = json.dumps({
            "tool_calls": [
                {"name": "check_remote_processes"}
            ]
        })

        result = parse_prompt_based_response(text)

        assert isinstance(result, ToolCallsResponse)
        assert result.tool_calls[0].arguments == {}

    def test_generated_call_ids(self) -> None:
        """Prompt-based mode generates call IDs since the LLM doesn't provide them."""
        text = json.dumps({
            "tool_calls": [
                {"name": "read_wiki", "arguments": {"slug": "a"}},
                {"name": "lookup_test_spec", "arguments": {"test_name": "b"}},
            ]
        })

        result = parse_prompt_based_response(text)

        assert isinstance(result, ToolCallsResponse)
        # Each call should have a unique, non-empty ID
        ids = [c.call_id for c in result.tool_calls]
        assert len(set(ids)) == 2
        assert all(ids)

    def test_mixed_text_and_json_code_fence(self) -> None:
        """Text before and after the JSON code fence is preserved as assistant_text."""
        text = (
            "Let me check the wiki first.\n\n"
            "```json\n"
            '{"tool_calls": [{"name": "read_wiki", "arguments": {"slug": "test"}}]}'
            "\n```\n\n"
            "I will then analyze the output."
        )

        result = parse_prompt_based_response(text)

        assert isinstance(result, ToolCallsResponse)
        assert result.tool_calls[0].tool_name == "read_wiki"

    def test_malformed_json_in_code_fence_is_final_answer(self) -> None:
        """If the JSON in a code fence is malformed, treat as final answer."""
        text = (
            "Here is the result:\n"
            "```json\n"
            "{broken json{{\n"
            "```"
        )

        result = parse_prompt_based_response(text)

        assert isinstance(result, FinalAnswerResponse)

    def test_tool_call_with_empty_name_raises(self) -> None:
        text = json.dumps({
            "tool_calls": [
                {"name": "", "arguments": {}}
            ]
        })

        with pytest.raises(ValueError, match="name"):
            parse_prompt_based_response(text)


# ---------------------------------------------------------------------------
# ParsedResponse base type properties
# ---------------------------------------------------------------------------


class TestParsedResponseTypeUnion:
    """Tests that ParsedResponse acts as a proper discriminated union."""

    def test_final_answer_is_parsed_response(self) -> None:
        resp = FinalAnswerResponse(text="done")
        assert isinstance(resp, ParsedResponse)

    def test_tool_calls_is_parsed_response(self) -> None:
        call = ToolCall(call_id="c1", tool_name="read_wiki", arguments={})
        resp = ToolCallsResponse(tool_calls=(call,), assistant_text=None)
        assert isinstance(resp, ParsedResponse)

    def test_kind_discriminator_distinct(self) -> None:
        final = FinalAnswerResponse(text="done")
        call = ToolCall(call_id="c1", tool_name="read_wiki", arguments={})
        tool_resp = ToolCallsResponse(tool_calls=(call,), assistant_text=None)

        assert final.kind is not tool_resp.kind
        assert final.kind is ResponseKind.FINAL_ANSWER
        assert tool_resp.kind is ResponseKind.TOOL_CALLS
