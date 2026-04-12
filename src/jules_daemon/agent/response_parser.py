"""Agent loop LLM response parser with discriminated union result type.

Classifies raw LLM responses into one of two categories:

    - **ToolCallsResponse**: The LLM wants to invoke one or more tools.
      Contains parsed ToolCall instances with tool name and arguments.
    - **FinalAnswerResponse**: The LLM has finished and is returning a
      text answer. No more tool calls are needed.

Two parsing entry points are provided:

    - ``parse_completion_response(completion)``: For native mode. Takes
      a raw OpenAI ChatCompletion and inspects the message's ``tool_calls``
      field. If tool_calls are present, returns ToolCallsResponse.
      Otherwise, returns FinalAnswerResponse with the text content.

    - ``parse_prompt_based_response(text)``: For prompt-based mode.
      Takes raw text from the LLM and looks for a JSON block containing
      a ``tool_calls`` array. If found, parses tool names and arguments
      into ToolCall instances. Otherwise, returns FinalAnswerResponse.

Both modes produce the same discriminated union type (ParsedResponse)
so the agent loop can handle results uniformly regardless of the
underlying tool-calling strategy.

All result types are frozen dataclasses for project-wide immutability.

Usage::

    from jules_daemon.agent.response_parser import (
        parse_completion_response,
        parse_prompt_based_response,
        FinalAnswerResponse,
        ToolCallsResponse,
    )

    # Native mode
    result = parse_completion_response(completion)
    if result.is_tool_calls:
        for call in result.tool_calls:
            print(call.tool_name, call.arguments)
    else:
        print(result.text)

    # Prompt-based mode
    result = parse_prompt_based_response(raw_text)
    if result.is_final_answer:
        print(result.text)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from jules_daemon.agent.tool_types import ToolCall

__all__ = [
    "FinalAnswerResponse",
    "ParsedResponse",
    "ResponseKind",
    "ToolCallsResponse",
    "parse_completion_response",
    "parse_prompt_based_response",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ResponseKind discriminator enum
# ---------------------------------------------------------------------------


class ResponseKind(Enum):
    """Discriminator for the ParsedResponse union type.

    TOOL_CALLS:   The LLM wants to invoke one or more tools.
    FINAL_ANSWER: The LLM is done and returning a text answer.
    """

    TOOL_CALLS = "tool_calls"
    FINAL_ANSWER = "final_answer"


# ---------------------------------------------------------------------------
# Base class for the discriminated union
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedResponse:
    """Base type for the discriminated union of LLM response classifications.

    Subclasses override the ``kind`` property to return their specific
    discriminator value. The ``is_tool_calls`` and ``is_final_answer``
    boolean properties provide convenient type-narrowing without
    explicit isinstance checks.
    """

    @property
    def kind(self) -> ResponseKind:
        """The discriminator value identifying which variant this is."""
        raise NotImplementedError(
            "Subclasses must override the kind property"
        )

    @property
    def is_tool_calls(self) -> bool:
        """True if this response contains tool call requests."""
        return self.kind is ResponseKind.TOOL_CALLS

    @property
    def is_final_answer(self) -> bool:
        """True if this response is a final text answer."""
        return self.kind is ResponseKind.FINAL_ANSWER


# ---------------------------------------------------------------------------
# ToolCallsResponse
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallsResponse(ParsedResponse):
    """The LLM wants to invoke one or more tools.

    Attributes:
        tool_calls: Non-empty tuple of ToolCall instances extracted from
            the LLM response. Each contains the tool name, arguments,
            and a unique call_id for correlation with results.
        assistant_text: Optional text content accompanying the tool calls.
            Some LLMs emit a reasoning preamble alongside tool invocations.
            None if no text was present.
    """

    tool_calls: tuple[ToolCall, ...]
    assistant_text: str | None

    def __post_init__(self) -> None:
        if not self.tool_calls:
            raise ValueError("tool_calls must not be empty")

    @property
    def kind(self) -> ResponseKind:
        """Always TOOL_CALLS for this variant."""
        return ResponseKind.TOOL_CALLS


# ---------------------------------------------------------------------------
# FinalAnswerResponse
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalAnswerResponse(ParsedResponse):
    """The LLM has finished and is returning a text answer.

    Attributes:
        text: The final answer text. Must not be empty.
    """

    text: str

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("text must not be empty")

    @property
    def kind(self) -> ResponseKind:
        """Always FINAL_ANSWER for this variant."""
        return ResponseKind.FINAL_ANSWER


# ---------------------------------------------------------------------------
# Native mode parser
# ---------------------------------------------------------------------------


def parse_completion_response(completion: Any) -> ParsedResponse:
    """Parse a native OpenAI ChatCompletion into a discriminated union result.

    Inspects the first choice's message for tool_calls. If present,
    extracts them into ToolCall instances and returns ToolCallsResponse.
    Otherwise, extracts the text content and returns FinalAnswerResponse.

    Args:
        completion: Raw OpenAI ChatCompletion response object. Expected
            to have ``choices[0].message`` with optional ``tool_calls``
            and ``content`` fields.

    Returns:
        ToolCallsResponse if tool calls are present, FinalAnswerResponse
        if only text content is present.

    Raises:
        ValueError: If the response has no choices, or if both content
            and tool_calls are absent/empty.
    """
    if not completion.choices:
        raise ValueError("No choices in LLM response")

    message = completion.choices[0].message
    raw_tool_calls = message.tool_calls

    # Check for native tool calls
    if raw_tool_calls:
        calls = _extract_native_tool_calls(raw_tool_calls)
        assistant_text = message.content if message.content else None
        return ToolCallsResponse(
            tool_calls=tuple(calls),
            assistant_text=assistant_text,
        )

    # No tool calls -- expect text content
    content = message.content
    if not content or not content.strip():
        raise ValueError(
            "LLM response is empty: no tool calls and no text content"
        )

    return FinalAnswerResponse(text=content)


def _extract_native_tool_calls(
    raw_tool_calls: Any,
) -> list[ToolCall]:
    """Extract ToolCall instances from native OpenAI tool_calls.

    Handles malformed JSON in arguments by defaulting to an empty dict
    with a warning log. This is more resilient than raising -- the agent
    loop can observe the empty-args call and self-correct.

    Args:
        raw_tool_calls: The ``message.tool_calls`` list from an OpenAI
            ChatCompletion response.

    Returns:
        List of parsed ToolCall instances.
    """
    calls: list[ToolCall] = []

    for tc in raw_tool_calls:
        arguments = _parse_arguments_json(
            tc.function.arguments,
            tool_name=tc.function.name,
        )
        calls.append(
            ToolCall(
                call_id=tc.id,
                tool_name=tc.function.name,
                arguments=arguments,
            )
        )

    return calls


# ---------------------------------------------------------------------------
# Prompt-based mode parser
# ---------------------------------------------------------------------------

# Pattern for ```json ... ``` code fences
_CODE_FENCE_JSON_RE = re.compile(
    r"```json\s*\n(.*?)```",
    re.DOTALL,
)

# Pattern for ``` ... ``` code fences (no language tag)
_CODE_FENCE_PLAIN_RE = re.compile(
    r"```\s*\n(.*?)```",
    re.DOTALL,
)


def parse_prompt_based_response(text: str) -> ParsedResponse:
    """Parse a prompt-based LLM text response into a discriminated union result.

    Looks for a JSON block (inline or in a code fence) containing a
    ``tool_calls`` array. If found, parses tool names and arguments
    into ToolCall instances and returns ToolCallsResponse. Otherwise,
    returns FinalAnswerResponse with the raw text.

    The expected JSON format for tool calls is::

        {
            "tool_calls": [
                {"name": "tool_name", "arguments": {"key": "value"}},
                ...
            ]
        }

    This matches the format described in the prompt-based tool calling
    instructions emitted by ``client._build_tool_prompt_section()``.

    Args:
        text: Raw text content from the LLM response.

    Returns:
        ToolCallsResponse if a valid tool_calls block is found,
        FinalAnswerResponse otherwise.

    Raises:
        ValueError: If text is empty/whitespace, or if a tool call
            entry is missing the required ``name`` field.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("LLM response text is empty")

    # Try to extract JSON containing tool_calls
    json_data = _try_extract_tool_calls_json(stripped)

    if json_data is not None:
        raw_calls = json_data.get("tool_calls")
        if raw_calls and isinstance(raw_calls, list):
            calls = _parse_prompt_based_tool_calls(raw_calls)
            if calls:
                return ToolCallsResponse(
                    tool_calls=tuple(calls),
                    assistant_text=None,
                )

    # No valid tool_calls found -- treat as final answer
    return FinalAnswerResponse(text=stripped)


def _try_extract_tool_calls_json(text: str) -> dict[str, Any] | None:
    """Attempt to extract a JSON object with tool_calls from text.

    Tries strategies in order:
    1. Parse entire text as JSON
    2. Extract from ```json ... ``` code fence
    3. Extract from ``` ... ``` code fence

    Args:
        text: Stripped LLM response text.

    Returns:
        Parsed dict if a JSON object with ``tool_calls`` key is found,
        None otherwise.
    """
    # Strategy 1: Try parsing the whole text as JSON
    result = _try_parse_json(text)
    if result is not None and "tool_calls" in result:
        return result

    # Strategy 2: Extract from ```json ... ``` code fence
    match = _CODE_FENCE_JSON_RE.search(text)
    if match:
        result = _try_parse_json(match.group(1).strip())
        if result is not None and "tool_calls" in result:
            return result

    # Strategy 3: Extract from ``` ... ``` code fence
    match = _CODE_FENCE_PLAIN_RE.search(text)
    if match:
        result = _try_parse_json(match.group(1).strip())
        if result is not None and "tool_calls" in result:
            return result

    return None


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Try to parse text as a JSON object. Return None on failure.

    Args:
        text: Candidate JSON string.

    Returns:
        Parsed dict if valid JSON object, None otherwise.
    """
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _parse_prompt_based_tool_calls(
    raw_calls: list[Any],
) -> list[ToolCall]:
    """Parse tool call entries from a prompt-based JSON response.

    Each entry must have a ``name`` key. The ``arguments`` key is optional
    and defaults to an empty dict. Call IDs are generated since the LLM
    does not provide them in prompt-based mode.

    Args:
        raw_calls: List of dicts from the ``tool_calls`` JSON array.

    Returns:
        List of parsed ToolCall instances.

    Raises:
        ValueError: If any entry is missing the ``name`` field or
            the name is empty.
    """
    calls: list[ToolCall] = []

    for entry in raw_calls:
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict tool_call entry: %r", entry)
            continue

        name = entry.get("name")
        if not name or (isinstance(name, str) and not name.strip()):
            raise ValueError(
                f"Tool call entry missing or empty 'name' field: {entry!r}"
            )

        arguments = entry.get("arguments", {})
        if not isinstance(arguments, dict):
            logger.warning(
                "Tool call '%s' has non-dict arguments, defaulting to empty: %r",
                name,
                arguments,
            )
            arguments = {}

        call_id = _generate_call_id()

        calls.append(
            ToolCall(
                call_id=call_id,
                tool_name=name,
                arguments=arguments,
            )
        )

    return calls


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_arguments_json(
    raw_arguments: str | None,
    *,
    tool_name: str,
) -> dict[str, Any]:
    """Parse a JSON string of tool call arguments.

    Handles None, empty string, and malformed JSON gracefully by
    defaulting to an empty dict with a warning log.

    Args:
        raw_arguments: JSON string of arguments, or None.
        tool_name: Tool name for log context.

    Returns:
        Parsed arguments dict, or empty dict on failure.
    """
    if not raw_arguments:
        return {}

    try:
        parsed = json.loads(raw_arguments)
        if isinstance(parsed, dict):
            return parsed
        logger.warning(
            "Tool '%s' arguments parsed to non-dict type %s, defaulting to empty",
            tool_name,
            type(parsed).__name__,
        )
        return {}
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "Failed to parse tool call arguments for '%s': %s",
            tool_name,
            raw_arguments,
        )
        return {}


def _generate_call_id() -> str:
    """Generate a unique call ID for prompt-based tool calls.

    Returns:
        A string call ID in the format ``pb_<uuid4_hex_prefix>``.
        The ``pb_`` prefix distinguishes prompt-based IDs from native
        OpenAI-generated IDs.
    """
    return f"pb_{uuid.uuid4().hex[:12]}"
