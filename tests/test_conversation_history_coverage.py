"""Targeted coverage tests for jules_daemon.agent.conversation_history.

Exercises every line flagged as uncovered in the coverage report:
    - Line 118: is_empty property (returns len(messages) == 0)
    - Lines 123-125: last_message property (returns last msg or None)
    - Lines 159-162: _validate_non_empty_content raises ValueError
    - Line 193: create_history raises ValueError for empty system_prompt
    - Lines 228-233: append_system_message function
    - Lines 252-257: append_user_message function
    - Line 285: append_assistant_message raises ValueError (no content, no tool_calls)
    - Line 299: append_assistant_message with text-only content (no tool_calls)
    - Lines 348-364: append_tool_results (empty tool_calls, length mismatch, happy path)
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
from jules_daemon.agent.tool_types import ToolCall, ToolResult


# ---------------------------------------------------------------------------
# is_empty property (line 118)
# ---------------------------------------------------------------------------


class TestIsEmptyProperty:
    """Cover the is_empty property on ConversationHistory."""

    def test_is_empty_on_fresh_empty_history(self) -> None:
        history = ConversationHistory(messages=())
        assert history.is_empty is True

    def test_is_empty_after_single_message(self) -> None:
        history = ConversationHistory(
            messages=({"role": "system", "content": "hello"},)
        )
        assert history.is_empty is False

    def test_is_empty_after_multiple_messages(self) -> None:
        history = ConversationHistory(
            messages=(
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
            )
        )
        assert history.is_empty is False


# ---------------------------------------------------------------------------
# last_message property (lines 123-125)
# ---------------------------------------------------------------------------


class TestLastMessageProperty:
    """Cover the last_message property on ConversationHistory."""

    def test_last_message_returns_none_for_empty_history(self) -> None:
        history = ConversationHistory(messages=())
        assert history.last_message is None

    def test_last_message_returns_sole_message(self) -> None:
        msg: dict[str, Any] = {"role": "system", "content": "only one"}
        history = ConversationHistory(messages=(msg,))
        assert history.last_message == msg

    def test_last_message_returns_final_of_many(self) -> None:
        first: dict[str, Any] = {"role": "system", "content": "first"}
        second: dict[str, Any] = {"role": "user", "content": "second"}
        third: dict[str, Any] = {"role": "assistant", "content": "third"}
        history = ConversationHistory(messages=(first, second, third))
        assert history.last_message == third
        # Verify the earlier messages are not returned.
        assert history.last_message != first


# ---------------------------------------------------------------------------
# _validate_non_empty_content (lines 159-162)
# Exercised indirectly through append_system_message and append_user_message
# ---------------------------------------------------------------------------


class TestValidateNonEmptyContent:
    """Cover _validate_non_empty_content via public API callers."""

    def test_empty_string_raises_via_system_message(self) -> None:
        history = ConversationHistory(messages=())
        with pytest.raises(ValueError, match="content must not be empty"):
            append_system_message(history, "")

    def test_whitespace_only_raises_via_system_message(self) -> None:
        history = ConversationHistory(messages=())
        with pytest.raises(ValueError, match="content must not be empty"):
            append_system_message(history, "   \t\n  ")

    def test_empty_string_raises_via_user_message(self) -> None:
        history = create_history("sys")
        with pytest.raises(ValueError, match="content must not be empty"):
            append_user_message(history, "")

    def test_whitespace_only_raises_via_user_message(self) -> None:
        history = create_history("sys")
        with pytest.raises(ValueError, match="content must not be empty"):
            append_user_message(history, "  \n\t  ")

    def test_valid_content_passes_and_is_stripped(self) -> None:
        history = ConversationHistory(messages=())
        result = append_system_message(history, "  hello  ")
        assert result.messages[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# create_history raises ValueError for empty system_prompt (line 193)
# ---------------------------------------------------------------------------


class TestCreateHistoryValidation:
    """Cover create_history ValueError path for empty system_prompt."""

    def test_empty_string_system_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="system_prompt must not be empty"):
            create_history("")

    def test_whitespace_only_system_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="system_prompt must not be empty"):
            create_history("   \n\t  ")

    def test_valid_system_prompt_strips_whitespace(self) -> None:
        history = create_history("  valid prompt  ")
        assert history.messages[0]["content"] == "valid prompt"

    def test_system_prompt_with_user_message(self) -> None:
        history = create_history("prompt", user_message="go")
        assert len(history) == 2
        assert history.messages[0]["role"] == MessageRole.SYSTEM.value
        assert history.messages[1]["role"] == MessageRole.USER.value


# ---------------------------------------------------------------------------
# append_system_message (lines 228-233)
# ---------------------------------------------------------------------------


class TestAppendSystemMessageCoverage:
    """Cover append_system_message function body."""

    def test_appends_to_empty_history(self) -> None:
        history = ConversationHistory(messages=())
        result = append_system_message(history, "injected system msg")
        assert len(result) == 1
        assert result.messages[0] == {
            "role": "system",
            "content": "injected system msg",
        }

    def test_appends_to_existing_history(self) -> None:
        history = create_history("initial system")
        result = append_system_message(history, "additional context")
        assert len(result) == 2
        assert result.messages[1]["role"] == "system"
        assert result.messages[1]["content"] == "additional context"

    def test_does_not_mutate_original(self) -> None:
        original = create_history("sys")
        original_len = len(original)
        _ = append_system_message(original, "extra")
        assert len(original) == original_len

    def test_content_is_stripped(self) -> None:
        history = ConversationHistory(messages=())
        result = append_system_message(history, "  padded  ")
        assert result.messages[0]["content"] == "padded"


# ---------------------------------------------------------------------------
# append_user_message (lines 252-257)
# ---------------------------------------------------------------------------


class TestAppendUserMessageCoverage:
    """Cover append_user_message function body."""

    def test_appends_user_message(self) -> None:
        history = create_history("sys")
        result = append_user_message(history, "run tests")
        assert len(result) == 2
        assert result.messages[1] == {
            "role": "user",
            "content": "run tests",
        }

    def test_appends_multiple_user_messages(self) -> None:
        history = create_history("sys")
        h2 = append_user_message(history, "first")
        h3 = append_user_message(h2, "second")
        assert len(h3) == 3
        assert h3.messages[1]["content"] == "first"
        assert h3.messages[2]["content"] == "second"

    def test_does_not_mutate_original(self) -> None:
        original = create_history("sys")
        _ = append_user_message(original, "user msg")
        assert len(original) == 1

    def test_content_is_stripped(self) -> None:
        history = create_history("sys")
        result = append_user_message(history, "  trimmed  ")
        assert result.messages[1]["content"] == "trimmed"


# ---------------------------------------------------------------------------
# append_assistant_message raises ValueError (line 285)
# ---------------------------------------------------------------------------


class TestAppendAssistantMessageNoContentNoToolCalls:
    """Cover ValueError when neither content nor tool_calls is provided."""

    def test_raises_when_both_missing(self) -> None:
        history = create_history("sys", user_message="hi")
        with pytest.raises(
            ValueError,
            match="Assistant message must have content or tool_calls",
        ):
            append_assistant_message(history)

    def test_raises_with_none_content_and_empty_tuple(self) -> None:
        history = create_history("sys", user_message="hi")
        with pytest.raises(
            ValueError,
            match="Assistant message must have content or tool_calls",
        ):
            append_assistant_message(history, content=None, tool_calls=())


# ---------------------------------------------------------------------------
# append_assistant_message with text-only content (line 299)
# ---------------------------------------------------------------------------


class TestAppendAssistantMessageTextOnly:
    """Cover the text-only branch (no tool_calls) in append_assistant_message."""

    def test_text_only_content(self) -> None:
        history = create_history("sys", user_message="hi")
        result = append_assistant_message(history, content="I can help")
        msg = result.messages[2]
        assert msg["role"] == "assistant"
        assert msg["content"] == "I can help"
        assert "tool_calls" not in msg

    def test_text_only_does_not_set_tool_calls_key(self) -> None:
        history = create_history("sys", user_message="hi")
        result = append_assistant_message(history, content="just text")
        msg = result.messages[2]
        assert set(msg.keys()) == {"role", "content"}

    def test_does_not_mutate_original(self) -> None:
        original = create_history("sys", user_message="hi")
        _ = append_assistant_message(original, content="reply")
        assert len(original) == 2


# ---------------------------------------------------------------------------
# append_tool_results (lines 348-364)
# ---------------------------------------------------------------------------


class TestAppendToolResultsCoverage:
    """Cover append_tool_results: empty error, mismatch error, and happy path."""

    def test_empty_tool_calls_raises(self) -> None:
        history = create_history("sys", user_message="go")
        with pytest.raises(ValueError, match="tool_calls must not be empty"):
            append_tool_results(history, tool_calls=(), results=())

    def test_length_mismatch_raises(self) -> None:
        history = create_history("sys", user_message="go")
        calls = (
            ToolCall(call_id="c1", tool_name="read_wiki", arguments={}),
            ToolCall(call_id="c2", tool_name="read_wiki", arguments={}),
        )
        results = (
            ToolResult.success(
                call_id="c1", tool_name="read_wiki", output="data"
            ),
        )
        with pytest.raises(ValueError, match="must have the same length"):
            append_tool_results(history, tool_calls=calls, results=results)

    def test_length_mismatch_error_message_includes_counts(self) -> None:
        history = create_history("sys", user_message="go")
        calls = (
            ToolCall(call_id="c1", tool_name="t", arguments={}),
        )
        results = (
            ToolResult.success(call_id="c1", tool_name="t", output="a"),
            ToolResult.success(call_id="c2", tool_name="t", output="b"),
            ToolResult.success(call_id="c3", tool_name="t", output="c"),
        )
        with pytest.raises(ValueError, match=r"got 1 and 3"):
            append_tool_results(history, tool_calls=calls, results=results)

    def test_happy_path_single_call(self) -> None:
        history = create_history("sys", user_message="go")
        calls = (
            ToolCall(
                call_id="c1",
                tool_name="read_wiki",
                arguments={"slug": "test"},
            ),
        )
        results = (
            ToolResult.success(
                call_id="c1",
                tool_name="read_wiki",
                output="page content",
            ),
        )
        updated = append_tool_results(
            history, tool_calls=calls, results=results
        )

        # system + user + assistant(tool_calls) + tool(result) = 4
        assert len(updated) == 4
        assert updated.messages[2]["role"] == "assistant"
        assert len(updated.messages[2]["tool_calls"]) == 1
        assert updated.messages[3]["role"] == "tool"
        assert updated.messages[3]["tool_call_id"] == "c1"
        assert updated.messages[3]["content"] == "page content"

    def test_happy_path_multiple_calls(self) -> None:
        history = create_history("sys", user_message="go")
        calls = (
            ToolCall(call_id="c1", tool_name="read_wiki", arguments={}),
            ToolCall(call_id="c2", tool_name="lookup_test_spec", arguments={}),
        )
        results = (
            ToolResult.success(
                call_id="c1", tool_name="read_wiki", output="wiki"
            ),
            ToolResult.success(
                call_id="c2", tool_name="lookup_test_spec", output="spec"
            ),
        )
        updated = append_tool_results(
            history, tool_calls=calls, results=results
        )

        # system + user + assistant(2 calls) + tool + tool = 5
        assert len(updated) == 5

        assistant_msg = updated.messages[2]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["tool_calls"]) == 2

        assert updated.messages[3]["role"] == "tool"
        assert updated.messages[3]["tool_call_id"] == "c1"
        assert updated.messages[4]["role"] == "tool"
        assert updated.messages[4]["tool_call_id"] == "c2"

    def test_does_not_mutate_original(self) -> None:
        original = create_history("sys", user_message="go")
        calls = (
            ToolCall(call_id="c1", tool_name="read_wiki", arguments={}),
        )
        results = (
            ToolResult.success(
                call_id="c1", tool_name="read_wiki", output="data"
            ),
        )
        _ = append_tool_results(original, tool_calls=calls, results=results)
        assert len(original) == 2

    def test_tool_results_order_matches_calls(self) -> None:
        history = create_history("sys", user_message="go")
        calls = (
            ToolCall(call_id="c1", tool_name="alpha", arguments={}),
            ToolCall(call_id="c2", tool_name="beta", arguments={}),
            ToolCall(call_id="c3", tool_name="gamma", arguments={}),
        )
        results = (
            ToolResult.success(call_id="c1", tool_name="alpha", output="a"),
            ToolResult.success(call_id="c2", tool_name="beta", output="b"),
            ToolResult.success(call_id="c3", tool_name="gamma", output="c"),
        )
        updated = append_tool_results(
            history, tool_calls=calls, results=results
        )

        # Verify tool results appear in order after the assistant message
        tool_msgs = [
            m for m in updated.messages if m.get("role") == "tool"
        ]
        assert len(tool_msgs) == 3
        assert tool_msgs[0]["tool_call_id"] == "c1"
        assert tool_msgs[1]["tool_call_id"] == "c2"
        assert tool_msgs[2]["tool_call_id"] == "c3"
