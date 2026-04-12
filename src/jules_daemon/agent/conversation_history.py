"""Immutable conversation history accumulator for the agent loop.

Provides an immutable data structure and pure helper functions for
building the LLM conversation context message by message. Each append
operation returns a **new** ConversationHistory -- the original is
never mutated.

The messages are stored as OpenAI Chat Completions API-compatible dicts
(``role``, ``content``, ``tool_calls``, ``tool_call_id``) wrapped in a
frozen dataclass.  This makes the history safe to share between the
agent loop state machine and the LLM adapter without defensive copies.

Design decisions:
    - ``tuple[dict[str, Any], ...]`` as the internal storage format
      so the data is directly passable to ``LLMClient.get_tool_calls()``
      without conversion.
    - Frozen dataclass (not Pydantic) to stay lightweight and consistent
      with the project-wide pattern for immutable value objects.
    - Standalone helper functions (``append_*``) keep the ConversationHistory
      class thin and composable -- callers can build pipelines of
      append operations.
    - ``ToolCall.to_openai_tool_call()`` and ``ToolResult.to_openai_tool_message()``
      are reused for serialization to avoid duplicating format knowledge.

Usage::

    from jules_daemon.agent.conversation_history import (
        ConversationHistory,
        append_assistant_message,
        append_tool_results,
        append_user_message,
        create_history,
    )
    from jules_daemon.agent.tool_types import ToolCall, ToolResult

    history = create_history("You are a test runner", user_message="run smoke tests")
    history = append_tool_results(
        history,
        tool_calls=(ToolCall(call_id="c1", tool_name="read_wiki", arguments={}),),
        results=(ToolResult.success(call_id="c1", tool_name="read_wiki", output="data"),),
    )
    messages = history.to_openai_messages()
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from jules_daemon.agent.tool_types import ToolCall, ToolResult

__all__ = [
    "ConversationHistory",
    "MessageRole",
    "append_assistant_message",
    "append_system_message",
    "append_tool_result",
    "append_tool_results",
    "append_user_message",
    "create_history",
]


# ---------------------------------------------------------------------------
# MessageRole enum
# ---------------------------------------------------------------------------


class MessageRole(Enum):
    """Role identifiers for OpenAI Chat Completions API messages.

    Each value maps directly to the ``role`` field in the API message dict.

    SYSTEM:    System prompt defining the agent's persona and constraints.
    USER:      User-submitted natural-language command or follow-up.
    ASSISTANT: LLM-generated response (text and/or tool calls).
    TOOL:      Execution result for a tool call issued by the assistant.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ---------------------------------------------------------------------------
# ConversationHistory frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversationHistory:
    """Immutable, ordered conversation history for the agent loop.

    Wraps a tuple of OpenAI-format message dicts.  Every mutation
    operation (via the module-level ``append_*`` helpers) returns a
    **new** ``ConversationHistory`` instance -- the original is never
    modified.

    Attributes:
        messages: Ordered tuple of message dicts.  Each dict has at
            minimum a ``role`` key and the role-specific payload
            (``content``, ``tool_calls``, ``tool_call_id``).
    """

    messages: tuple[dict[str, Any], ...]

    # -- Convenience properties ---------------------------------------------

    def __len__(self) -> int:
        """Return the number of messages in the history."""
        return len(self.messages)

    @property
    def is_empty(self) -> bool:
        """True if the history contains no messages."""
        return len(self.messages) == 0

    @property
    def last_message(self) -> dict[str, Any] | None:
        """Return the last message, or None if the history is empty."""
        if self.messages:
            return self.messages[-1]
        return None

    # -- Serialization -------------------------------------------------------

    def to_openai_messages(self) -> tuple[dict[str, Any], ...]:
        """Return the messages as a tuple of OpenAI-format dicts.

        The returned tuple is the same object stored internally -- since
        both the tuple and the ConversationHistory are immutable, sharing
        is safe.

        Returns:
            Tuple of message dicts ready for the Chat Completions API.
        """
        return self.messages


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_non_empty_content(content: str) -> str:
    """Validate that a content string is non-empty after stripping.

    Args:
        content: The string to validate.

    Returns:
        The stripped content string.

    Raises:
        ValueError: If the content is empty or whitespace-only.
    """
    stripped = content.strip()
    if not stripped:
        raise ValueError("content must not be empty")
    return stripped


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_history(
    system_prompt: str,
    *,
    user_message: str | None = None,
) -> ConversationHistory:
    """Create a new conversation history with an initial system prompt.

    Optionally includes a user message as the second entry, which is the
    typical starting point for an agent loop iteration.

    Args:
        system_prompt: The system prompt text.  Must not be empty.
        user_message: Optional initial user message to append after
            the system prompt.

    Returns:
        New ConversationHistory with one or two messages.

    Raises:
        ValueError: If system_prompt is empty or whitespace-only.
    """
    stripped = system_prompt.strip()
    if not stripped:
        raise ValueError("system_prompt must not be empty")

    messages: list[dict[str, Any]] = [
        {"role": MessageRole.SYSTEM.value, "content": stripped},
    ]

    if user_message is not None:
        messages.append(
            {"role": MessageRole.USER.value, "content": user_message}
        )

    return ConversationHistory(messages=tuple(messages))


# ---------------------------------------------------------------------------
# Append helpers -- each returns a NEW ConversationHistory
# ---------------------------------------------------------------------------


def append_system_message(
    history: ConversationHistory,
    content: str,
) -> ConversationHistory:
    """Append a system message to the history.

    Args:
        history: Existing (immutable) conversation history.
        content: System message text.  Must not be empty.

    Returns:
        New ConversationHistory with the system message appended.

    Raises:
        ValueError: If content is empty or whitespace-only.
    """
    validated = _validate_non_empty_content(content)
    new_msg: dict[str, Any] = {
        "role": MessageRole.SYSTEM.value,
        "content": validated,
    }
    return ConversationHistory(messages=history.messages + (new_msg,))


def append_user_message(
    history: ConversationHistory,
    content: str,
) -> ConversationHistory:
    """Append a user message to the history.

    Args:
        history: Existing (immutable) conversation history.
        content: User message text.  Must not be empty.

    Returns:
        New ConversationHistory with the user message appended.

    Raises:
        ValueError: If content is empty or whitespace-only.
    """
    validated = _validate_non_empty_content(content)
    new_msg: dict[str, Any] = {
        "role": MessageRole.USER.value,
        "content": validated,
    }
    return ConversationHistory(messages=history.messages + (new_msg,))


def append_assistant_message(
    history: ConversationHistory,
    content: str | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
) -> ConversationHistory:
    """Append an assistant message to the history.

    An assistant message may contain text content, tool calls, or both.
    At least one must be provided.

    When tool calls are present, they are serialized to the OpenAI
    tool_call format using ``ToolCall.to_openai_tool_call()``.

    Args:
        history: Existing (immutable) conversation history.
        content: Optional text content from the assistant.
        tool_calls: Optional tuple of ToolCall instances to include.

    Returns:
        New ConversationHistory with the assistant message appended.

    Raises:
        ValueError: If neither content nor tool_calls is provided.
    """
    if content is None and not tool_calls:
        raise ValueError(
            "Assistant message must have content or tool_calls (or both)"
        )

    new_msg: dict[str, Any] = {
        "role": MessageRole.ASSISTANT.value,
    }

    if tool_calls:
        new_msg["content"] = content  # may be None per OpenAI spec
        new_msg["tool_calls"] = [
            call.to_openai_tool_call() for call in tool_calls
        ]
    else:
        new_msg["content"] = content

    return ConversationHistory(messages=history.messages + (new_msg,))


def append_tool_result(
    history: ConversationHistory,
    result: ToolResult,
) -> ConversationHistory:
    """Append a single tool result message to the history.

    Uses ``ToolResult.to_openai_tool_message()`` for format consistency
    with the existing serialization logic.

    Args:
        history: Existing (immutable) conversation history.
        result: The ToolResult to append as a tool message.

    Returns:
        New ConversationHistory with the tool result message appended.
    """
    new_msg = result.to_openai_tool_message()
    return ConversationHistory(messages=history.messages + (new_msg,))


def append_tool_results(
    history: ConversationHistory,
    *,
    tool_calls: tuple[ToolCall, ...],
    results: tuple[ToolResult, ...],
) -> ConversationHistory:
    """Append a complete tool-calling cycle to the history.

    Adds an assistant message containing the tool calls, followed by
    one tool result message per result.  This is the typical pattern
    for a single think-act-observe cycle in the agent loop.

    Args:
        history: Existing (immutable) conversation history.
        tool_calls: The tool calls the assistant made.  Must not be empty.
        results: The results from dispatching those calls.  Must have the
            same length as ``tool_calls``.

    Returns:
        New ConversationHistory with the assistant + tool messages appended.

    Raises:
        ValueError: If tool_calls is empty, or if the lengths do not match.
    """
    if not tool_calls:
        raise ValueError("tool_calls must not be empty")

    if len(tool_calls) != len(results):
        raise ValueError(
            f"tool_calls and results must have the same length "
            f"(got {len(tool_calls)} and {len(results)})"
        )

    # Start by appending the assistant message with tool calls
    updated = append_assistant_message(history, tool_calls=tool_calls)

    # Then append each tool result message
    for result in results:
        updated = append_tool_result(updated, result)

    return updated
