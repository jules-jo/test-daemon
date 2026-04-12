"""Tests for the immutable conversation history accumulator.

Validates:
    - MessageRole enum members and values
    - ConversationHistory creation and immutability (frozen dataclass)
    - Factory function create_history() initializes with system prompt
    - append_system_message() returns new history with system message
    - append_user_message() returns new history with user message
    - append_assistant_message() with text content only
    - append_assistant_message() with tool calls only (content=None)
    - append_assistant_message() with both text and tool calls
    - append_tool_result() with ToolResult
    - append_tool_results() for a full cycle (assistant + tool results)
    - History ordering is preserved across appends
    - Message format matches OpenAI Chat Completions API
    - Original history is never mutated by any append operation
    - Empty history creation
    - len() support
    - to_openai_messages() returns the tuple of dicts
"""

from __future__ import annotations

from typing import Any

import pytest

from jules_daemon.agent.conversation_history import (
    ConversationHistory,
    MessageRole,
    append_assistant_message,
    append_system_message,
    append_tool_result,
    append_tool_results,
    append_user_message,
    create_history,
)
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResult,
    ToolResultStatus,
)


# ---------------------------------------------------------------------------
# MessageRole enum
# ---------------------------------------------------------------------------


class TestMessageRole:
    """Tests for the MessageRole enum."""

    def test_all_members_exist(self) -> None:
        members = {r.name for r in MessageRole}
        assert members == {"SYSTEM", "USER", "ASSISTANT", "TOOL"}

    def test_string_values(self) -> None:
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"


# ---------------------------------------------------------------------------
# ConversationHistory -- creation and immutability
# ---------------------------------------------------------------------------


class TestConversationHistoryCreation:
    """Tests for ConversationHistory construction and immutability."""

    def test_empty_history(self) -> None:
        history = ConversationHistory(messages=())
        assert len(history) == 0
        assert history.messages == ()

    def test_history_with_messages(self) -> None:
        msgs: tuple[dict[str, Any], ...] = (
            {"role": "system", "content": "You are a test runner"},
            {"role": "user", "content": "run tests"},
        )
        history = ConversationHistory(messages=msgs)
        assert len(history) == 2

    def test_frozen(self) -> None:
        history = ConversationHistory(messages=())
        with pytest.raises(AttributeError):
            history.messages = ({"role": "system", "content": "hack"},)  # type: ignore[misc]

    def test_to_openai_messages(self) -> None:
        msgs: tuple[dict[str, Any], ...] = (
            {"role": "system", "content": "prompt"},
        )
        history = ConversationHistory(messages=msgs)
        assert history.to_openai_messages() == msgs

    def test_len(self) -> None:
        msgs: tuple[dict[str, Any], ...] = (
            {"role": "system", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
        )
        history = ConversationHistory(messages=msgs)
        assert len(history) == 3

    def test_is_empty_true(self) -> None:
        history = ConversationHistory(messages=())
        assert history.is_empty is True

    def test_is_empty_false(self) -> None:
        history = ConversationHistory(
            messages=({"role": "system", "content": "x"},)
        )
        assert history.is_empty is False

    def test_last_message_returns_final(self) -> None:
        msgs: tuple[dict[str, Any], ...] = (
            {"role": "system", "content": "a"},
            {"role": "user", "content": "b"},
        )
        history = ConversationHistory(messages=msgs)
        assert history.last_message == {"role": "user", "content": "b"}

    def test_last_message_empty_returns_none(self) -> None:
        history = ConversationHistory(messages=())
        assert history.last_message is None


# ---------------------------------------------------------------------------
# create_history factory
# ---------------------------------------------------------------------------


class TestCreateHistory:
    """Tests for the create_history() factory function."""

    def test_creates_with_system_prompt(self) -> None:
        history = create_history("You are a test runner")
        assert len(history) == 1
        assert history.messages[0]["role"] == "system"
        assert history.messages[0]["content"] == "You are a test runner"

    def test_creates_with_system_and_user(self) -> None:
        history = create_history(
            "You are a test runner",
            user_message="run the smoke tests",
        )
        assert len(history) == 2
        assert history.messages[0]["role"] == "system"
        assert history.messages[1]["role"] == "user"
        assert history.messages[1]["content"] == "run the smoke tests"

    def test_empty_system_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="system_prompt must not be empty"):
            create_history("")

    def test_whitespace_system_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="system_prompt must not be empty"):
            create_history("   ")


# ---------------------------------------------------------------------------
# append_system_message
# ---------------------------------------------------------------------------


class TestAppendSystemMessage:
    """Tests for the append_system_message() helper."""

    def test_appends_system_message(self) -> None:
        original = ConversationHistory(messages=())
        result = append_system_message(original, "system prompt")

        assert len(result) == 1
        assert result.messages[0]["role"] == "system"
        assert result.messages[0]["content"] == "system prompt"

    def test_does_not_mutate_original(self) -> None:
        original = ConversationHistory(messages=())
        _ = append_system_message(original, "system prompt")
        assert len(original) == 0

    def test_preserves_existing_messages(self) -> None:
        original = ConversationHistory(
            messages=({"role": "user", "content": "hello"},)
        )
        result = append_system_message(original, "system prompt")
        assert len(result) == 2
        assert result.messages[0]["role"] == "user"
        assert result.messages[1]["role"] == "system"

    def test_empty_content_raises(self) -> None:
        history = ConversationHistory(messages=())
        with pytest.raises(ValueError, match="content must not be empty"):
            append_system_message(history, "")


# ---------------------------------------------------------------------------
# append_user_message
# ---------------------------------------------------------------------------


class TestAppendUserMessage:
    """Tests for the append_user_message() helper."""

    def test_appends_user_message(self) -> None:
        original = create_history("system prompt")
        result = append_user_message(original, "run the smoke tests")

        assert len(result) == 2
        assert result.messages[1]["role"] == "user"
        assert result.messages[1]["content"] == "run the smoke tests"

    def test_does_not_mutate_original(self) -> None:
        original = create_history("system prompt")
        _ = append_user_message(original, "run tests")
        assert len(original) == 1

    def test_empty_content_raises(self) -> None:
        history = create_history("system prompt")
        with pytest.raises(ValueError, match="content must not be empty"):
            append_user_message(history, "")


# ---------------------------------------------------------------------------
# append_assistant_message
# ---------------------------------------------------------------------------


class TestAppendAssistantMessage:
    """Tests for the append_assistant_message() helper."""

    def test_text_only(self) -> None:
        original = create_history("prompt", user_message="hello")
        result = append_assistant_message(original, content="I will help you")

        assert len(result) == 3
        msg = result.messages[2]
        assert msg["role"] == "assistant"
        assert msg["content"] == "I will help you"
        assert "tool_calls" not in msg

    def test_tool_calls_only(self) -> None:
        original = create_history("prompt", user_message="run tests")
        calls = (
            ToolCall(call_id="call_001", tool_name="read_wiki", arguments={"slug": "smoke"}),
        )
        result = append_assistant_message(original, tool_calls=calls)

        assert len(result) == 3
        msg = result.messages[2]
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "call_001"
        assert msg["tool_calls"][0]["type"] == "function"
        assert msg["tool_calls"][0]["function"]["name"] == "read_wiki"

    def test_text_and_tool_calls(self) -> None:
        original = create_history("prompt", user_message="run tests")
        calls = (
            ToolCall(call_id="call_001", tool_name="read_wiki", arguments={}),
        )
        result = append_assistant_message(
            original,
            content="Let me look that up",
            tool_calls=calls,
        )

        msg = result.messages[2]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me look that up"
        assert len(msg["tool_calls"]) == 1

    def test_multiple_tool_calls(self) -> None:
        original = create_history("prompt", user_message="run tests")
        calls = (
            ToolCall(call_id="call_001", tool_name="read_wiki", arguments={}),
            ToolCall(call_id="call_002", tool_name="lookup_test_spec", arguments={"name": "smoke"}),
        )
        result = append_assistant_message(original, tool_calls=calls)

        msg = result.messages[2]
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["function"]["name"] == "read_wiki"
        assert msg["tool_calls"][1]["function"]["name"] == "lookup_test_spec"

    def test_no_content_no_tool_calls_raises(self) -> None:
        history = create_history("prompt", user_message="hello")
        with pytest.raises(ValueError, match="must have content or tool_calls"):
            append_assistant_message(history)

    def test_does_not_mutate_original(self) -> None:
        original = create_history("prompt", user_message="hello")
        _ = append_assistant_message(original, content="reply")
        assert len(original) == 2


# ---------------------------------------------------------------------------
# append_tool_result
# ---------------------------------------------------------------------------


class TestAppendToolResult:
    """Tests for the append_tool_result() helper."""

    def test_success_result(self) -> None:
        original = create_history("prompt", user_message="run tests")
        result_obj = ToolResult.success(
            call_id="call_001",
            tool_name="read_wiki",
            output="wiki content here",
        )
        result = append_tool_result(original, result_obj)

        assert len(result) == 3
        msg = result.messages[2]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_001"
        assert msg["content"] == "wiki content here"

    def test_error_result(self) -> None:
        original = create_history("prompt", user_message="run tests")
        result_obj = ToolResult.error(
            call_id="call_002",
            tool_name="execute_ssh",
            error_message="Connection refused",
        )
        result = append_tool_result(original, result_obj)

        msg = result.messages[2]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_002"
        assert "ERROR" in msg["content"]
        assert "Connection refused" in msg["content"]

    def test_denied_result(self) -> None:
        original = create_history("prompt", user_message="run tests")
        result_obj = ToolResult.denied(
            call_id="call_003",
            tool_name="execute_ssh",
            error_message="User denied the operation",
        )
        result = append_tool_result(original, result_obj)

        msg = result.messages[2]
        assert msg["role"] == "tool"
        assert "ERROR" in msg["content"]

    def test_does_not_mutate_original(self) -> None:
        original = create_history("prompt", user_message="run tests")
        result_obj = ToolResult.success(
            call_id="call_001",
            tool_name="read_wiki",
            output="data",
        )
        _ = append_tool_result(original, result_obj)
        assert len(original) == 2


# ---------------------------------------------------------------------------
# append_tool_results
# ---------------------------------------------------------------------------


class TestAppendToolResults:
    """Tests for the append_tool_results() helper (full cycle)."""

    def test_appends_assistant_and_tool_messages(self) -> None:
        original = create_history("prompt", user_message="run tests")
        calls = (
            ToolCall(call_id="call_001", tool_name="read_wiki", arguments={"slug": "smoke"}),
        )
        results = (
            ToolResult.success(
                call_id="call_001",
                tool_name="read_wiki",
                output="wiki page content",
            ),
        )
        updated = append_tool_results(original, tool_calls=calls, results=results)

        # system + user + assistant(tool_calls) + tool(result)
        assert len(updated) == 4
        assert updated.messages[2]["role"] == "assistant"
        assert updated.messages[3]["role"] == "tool"

    def test_multiple_calls_and_results(self) -> None:
        original = create_history("prompt", user_message="run tests")
        calls = (
            ToolCall(call_id="call_001", tool_name="read_wiki", arguments={}),
            ToolCall(call_id="call_002", tool_name="lookup_test_spec", arguments={}),
        )
        results = (
            ToolResult.success(call_id="call_001", tool_name="read_wiki", output="page"),
            ToolResult.success(call_id="call_002", tool_name="lookup_test_spec", output="spec"),
        )
        updated = append_tool_results(original, tool_calls=calls, results=results)

        # system + user + assistant(2 tool_calls) + tool + tool
        assert len(updated) == 5
        assert updated.messages[2]["role"] == "assistant"
        assert len(updated.messages[2]["tool_calls"]) == 2
        assert updated.messages[3]["role"] == "tool"
        assert updated.messages[4]["role"] == "tool"

    def test_mismatched_lengths_raises(self) -> None:
        history = create_history("prompt", user_message="run tests")
        calls = (
            ToolCall(call_id="call_001", tool_name="read_wiki", arguments={}),
        )
        results = (
            ToolResult.success(call_id="call_001", tool_name="read_wiki", output="a"),
            ToolResult.success(call_id="call_002", tool_name="read_wiki", output="b"),
        )
        with pytest.raises(ValueError, match="must have the same length"):
            append_tool_results(history, tool_calls=calls, results=results)

    def test_empty_calls_raises(self) -> None:
        history = create_history("prompt", user_message="run tests")
        with pytest.raises(ValueError, match="tool_calls must not be empty"):
            append_tool_results(history, tool_calls=(), results=())

    def test_does_not_mutate_original(self) -> None:
        original = create_history("prompt", user_message="run tests")
        calls = (
            ToolCall(call_id="call_001", tool_name="read_wiki", arguments={}),
        )
        results = (
            ToolResult.success(call_id="call_001", tool_name="read_wiki", output="x"),
        )
        _ = append_tool_results(original, tool_calls=calls, results=results)
        assert len(original) == 2


# ---------------------------------------------------------------------------
# Ordering and accumulation
# ---------------------------------------------------------------------------


class TestHistoryOrdering:
    """Tests for proper message ordering across multiple appends."""

    def test_full_conversation_ordering(self) -> None:
        """Simulate a realistic 2-cycle conversation and verify ordering."""
        # Initialize
        h = create_history("You are a test runner", user_message="run smoke tests")

        # Cycle 1: LLM calls read_wiki
        calls_1 = (
            ToolCall(call_id="c1", tool_name="read_wiki", arguments={"slug": "smoke"}),
        )
        results_1 = (
            ToolResult.success(call_id="c1", tool_name="read_wiki", output="wiki data"),
        )
        h = append_tool_results(h, tool_calls=calls_1, results=results_1)

        # Cycle 2: LLM calls propose_ssh_command
        calls_2 = (
            ToolCall(call_id="c2", tool_name="propose_ssh_command", arguments={"cmd": "pytest"}),
        )
        results_2 = (
            ToolResult.success(call_id="c2", tool_name="propose_ssh_command", output="approved"),
        )
        h = append_tool_results(h, tool_calls=calls_2, results=results_2)

        # Verify full ordering
        roles = [m["role"] for m in h.messages]
        assert roles == [
            "system",       # initial
            "user",         # initial
            "assistant",    # cycle 1 tool calls
            "tool",         # cycle 1 result
            "assistant",    # cycle 2 tool calls
            "tool",         # cycle 2 result
        ]

    def test_each_append_returns_new_instance(self) -> None:
        """Every append operation returns a new ConversationHistory instance."""
        h1 = create_history("system")
        h2 = append_user_message(h1, "user message")
        h3 = append_assistant_message(h2, content="response")

        assert h1 is not h2
        assert h2 is not h3
        assert len(h1) == 1
        assert len(h2) == 2
        assert len(h3) == 3


# ---------------------------------------------------------------------------
# OpenAI format compliance
# ---------------------------------------------------------------------------


class TestOpenAIFormatCompliance:
    """Tests ensuring messages match the OpenAI Chat Completions API format."""

    def test_system_message_format(self) -> None:
        history = create_history("You are helpful")
        msg = history.messages[0]
        assert set(msg.keys()) == {"role", "content"}
        assert msg["role"] == "system"
        assert isinstance(msg["content"], str)

    def test_user_message_format(self) -> None:
        history = append_user_message(
            create_history("sys"),
            "hello",
        )
        msg = history.messages[1]
        assert set(msg.keys()) == {"role", "content"}
        assert msg["role"] == "user"

    def test_assistant_text_message_format(self) -> None:
        history = append_assistant_message(
            create_history("sys", user_message="hi"),
            content="hello back",
        )
        msg = history.messages[2]
        assert set(msg.keys()) == {"role", "content"}
        assert msg["role"] == "assistant"
        assert msg["content"] == "hello back"

    def test_assistant_tool_call_message_format(self) -> None:
        calls = (
            ToolCall(call_id="call_1", tool_name="read_wiki", arguments={"k": "v"}),
        )
        history = append_assistant_message(
            create_history("sys", user_message="hi"),
            tool_calls=calls,
        )
        msg = history.messages[2]
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        tc = msg["tool_calls"][0]
        assert "id" in tc
        assert tc["type"] == "function"
        assert "function" in tc
        assert "name" in tc["function"]
        assert "arguments" in tc["function"]

    def test_tool_result_message_format(self) -> None:
        tool_res = ToolResult.success(
            call_id="call_1",
            tool_name="read_wiki",
            output="content",
        )
        history = append_tool_result(
            create_history("sys", user_message="hi"),
            tool_res,
        )
        msg = history.messages[2]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_1"
        assert isinstance(msg["content"], str)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and defensive behavior."""

    def test_large_history_does_not_corrupt(self) -> None:
        """Building up a large history still works correctly."""
        h = create_history("system prompt", user_message="start")
        for i in range(20):
            calls = (
                ToolCall(
                    call_id=f"call_{i}",
                    tool_name="read_wiki",
                    arguments={"i": str(i)},
                ),
            )
            results = (
                ToolResult.success(
                    call_id=f"call_{i}",
                    tool_name="read_wiki",
                    output=f"output_{i}",
                ),
            )
            h = append_tool_results(h, tool_calls=calls, results=results)

        # 2 initial + 20 * (assistant + tool) = 42
        assert len(h) == 42

    def test_assistant_tool_call_arguments_serialized_as_json_string(self) -> None:
        """Tool call arguments should be JSON-serialized strings in OpenAI format."""
        calls = (
            ToolCall(
                call_id="c1",
                tool_name="read_wiki",
                arguments={"slug": "test", "depth": 3},
            ),
        )
        history = append_assistant_message(
            create_history("sys", user_message="hi"),
            tool_calls=calls,
        )
        tc = history.messages[2]["tool_calls"][0]
        # arguments should be a JSON string (OpenAI format)
        assert isinstance(tc["function"]["arguments"], str)
        import json
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed["slug"] == "test"
        assert parsed["depth"] == 3
