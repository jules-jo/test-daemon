"""Tool protocol types for the agent loop.

Defines the immutable data structures that form the tool-calling protocol
between the LLM and the daemon. Every type is a frozen dataclass to match
the project-wide immutability convention.

Types:
    ToolParam              -- Single parameter definition with JSON schema type.
    ApprovalRequirement    -- Whether a tool requires human confirmation.
    ToolSpec               -- Complete tool specification (name, description,
                              parameters, approval). Serializes to OpenAI
                              function-calling format.
    ToolResultStatus       -- Outcome of executing a tool call.
    ToolCall               -- An LLM-issued request to invoke a tool.
    ToolResult             -- The outcome of executing a ToolCall.

OpenAI compatibility:
    ToolSpec.to_openai_function_schema() produces a dict conforming to the
    ``tools`` parameter of the Chat Completions API (``type: "function"``
    wrapper with nested ``function`` object).

    ToolCall.to_openai_tool_call() produces the assistant-side tool_call dict.

    ToolResult.to_openai_tool_message() produces a ``role: "tool"`` message
    dict for feeding execution results back into the conversation.

Usage::

    from jules_daemon.agent.tool_types import (
        ApprovalRequirement,
        ToolCall,
        ToolParam,
        ToolResult,
        ToolResultStatus,
        ToolSpec,
    )

    # Define a read-only tool spec
    spec = ToolSpec(
        name="read_wiki",
        description="Read a wiki page by slug",
        parameters=(
            ToolParam(name="slug", description="Wiki page slug",
                      json_type="string"),
        ),
        approval=ApprovalRequirement.NONE,
    )

    # Serialize for the OpenAI API
    schema = spec.to_openai_function_schema()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ApprovalRequirement",
    "ToolCall",
    "ToolParam",
    "ToolResult",
    "ToolResultStatus",
    "ToolSpec",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_JSON_TYPES: frozenset[str] = frozenset(
    {"string", "integer", "number", "boolean", "object", "array", "null"}
)
"""JSON Schema primitive types accepted by the OpenAI function-calling API."""


# ---------------------------------------------------------------------------
# ApprovalRequirement enum
# ---------------------------------------------------------------------------


class ApprovalRequirement(Enum):
    """Whether a tool invocation requires human confirmation.

    NONE:           Read-only tools that can execute freely without
                    user approval (e.g., read_wiki, lookup_test_spec).
    CONFIRM_PROMPT: State-changing tools that require explicit user
                    approval via the IPC CONFIRM_PROMPT flow before
                    execution (e.g., propose_ssh_command, execute_ssh).
    """

    NONE = "none"
    CONFIRM_PROMPT = "confirm_prompt"


# ---------------------------------------------------------------------------
# ToolResultStatus enum
# ---------------------------------------------------------------------------


class ToolResultStatus(Enum):
    """Outcome classification for a tool execution.

    SUCCESS:  Tool executed and returned valid output.
    ERROR:    Tool executed but encountered an error (retryable --
              the agent can observe the error and try a different approach).
    DENIED:   User denied the required approval. Terminal -- the agent
              loop must stop immediately.
    TIMEOUT:  Tool execution exceeded its time budget (retryable --
              the agent may retry with different parameters).
    """

    SUCCESS = "success"
    ERROR = "error"
    DENIED = "denied"
    TIMEOUT = "timeout"

    @property
    def is_terminal(self) -> bool:
        """Return True if this status should terminate the agent loop.

        Only DENIED is terminal -- errors and timeouts are observable
        by the agent and can prompt self-correction.
        """
        return self is ToolResultStatus.DENIED


# ---------------------------------------------------------------------------
# ToolParam dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolParam:
    """A single parameter definition for a tool.

    Maps directly to a property in a JSON Schema ``properties`` object
    for OpenAI function-calling compatibility.

    Attributes:
        name: Parameter identifier (must be a valid Python/JSON key).
        description: Human-readable description shown to the LLM.
        json_type: JSON Schema type string (string, integer, number,
            boolean, object, array, null).
        required: Whether the parameter is required for tool invocation.
        default: Default value when the parameter is omitted. Only
            meaningful when ``required`` is False.
        enum: Optional tuple of allowed values. Serialized as a JSON
            Schema ``enum`` array.
    """

    name: str
    description: str
    json_type: str
    required: bool = True
    default: Any = None
    enum: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.description or not self.description.strip():
            raise ValueError("description must not be empty")
        if self.json_type not in _VALID_JSON_TYPES:
            raise ValueError(
                f"json_type must be one of {sorted(_VALID_JSON_TYPES)}, "
                f"got {self.json_type!r}"
            )

    def to_json_schema(self) -> dict[str, Any]:
        """Serialize this parameter to a JSON Schema property definition.

        Returns:
            Dict suitable for inclusion in the ``properties`` object of
            an OpenAI function-calling parameter schema.
        """
        schema: dict[str, Any] = {
            "type": self.json_type,
            "description": self.description,
        }
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.default is not None:
            schema["default"] = self.default
        return schema


# ---------------------------------------------------------------------------
# ToolSpec dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """Complete specification for a tool in the agent registry.

    Describes what the tool does, what parameters it accepts, and whether
    it requires human approval. Serializes to the OpenAI function-calling
    ``tools`` format via ``to_openai_function_schema()``.

    Attributes:
        name: Unique tool name (used as the function name in API calls).
        description: Human-readable description shown to the LLM to help
            it decide when and how to invoke the tool.
        parameters: Tuple of ToolParam definitions. Order is preserved
            in serialization but has no semantic meaning.
        approval: Whether invocation requires human confirmation.
    """

    name: str
    description: str
    parameters: tuple[ToolParam, ...]
    approval: ApprovalRequirement

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.description or not self.description.strip():
            raise ValueError("description must not be empty")
        # Check for duplicate parameter names
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            seen: set[str] = set()
            dupes: list[str] = []
            for n in names:
                if n in seen:
                    dupes.append(n)
                seen.add(n)
            raise ValueError(
                f"Duplicate parameter names: {dupes}"
            )

    @property
    def is_read_only(self) -> bool:
        """True if the tool does not require human approval to execute."""
        return self.approval is ApprovalRequirement.NONE

    def to_openai_function_schema(self) -> dict[str, Any]:
        """Serialize to an OpenAI-compatible function tool schema.

        Produces a dict matching the ``tools`` array element format
        expected by the Chat Completions API::

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": { ... },
                        "required": [ ... ]
                    }
                }
            }

        Returns:
            JSON-serializable dict conforming to the OpenAI function
            tool schema.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# ---------------------------------------------------------------------------
# ToolCall dataclass
# ---------------------------------------------------------------------------


def _defensive_copy_args(args: dict[str, Any]) -> dict[str, Any]:
    """Create a shallow copy of the arguments dict for immutability."""
    return dict(args)


@dataclass(frozen=True)
class ToolCall:
    """An LLM-issued request to invoke a registered tool.

    Represents a single tool call extracted from the LLM response. The
    ``call_id`` ties the call to its corresponding ``ToolResult`` in the
    conversation history.

    Attributes:
        call_id: Unique identifier for this call (from the LLM response
            or generated by the agent loop). Used to correlate with the
            ToolResult in OpenAI message format.
        tool_name: Name of the tool to invoke (must match a registered
            tool in the ToolRegistry).
        arguments: Key-value arguments to pass to the tool. A defensive
            copy is made on construction to prevent external mutation.
    """

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.call_id or not self.call_id.strip():
            raise ValueError("call_id must not be empty")
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")
        # Defensive copy: replace mutable dict with a copy
        object.__setattr__(self, "arguments", _defensive_copy_args(self.arguments))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for persistence or logging.

        Returns:
            Dict with call_id, tool_name, and arguments keys.
        """
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        """Deserialize from a plain dict.

        Args:
            data: Dict with required keys: call_id, tool_name, arguments.

        Returns:
            New ToolCall instance.

        Raises:
            KeyError: If a required key is missing.
            ValueError: If validation fails.
        """
        return cls(
            call_id=data["call_id"],
            tool_name=data["tool_name"],
            arguments=data["arguments"],
        )

    def to_openai_tool_call(self) -> dict[str, Any]:
        """Serialize to the OpenAI assistant tool_call format.

        Produces a dict matching the tool_calls array element in an
        assistant message::

            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "read_wiki",
                    "arguments": "{\"slug\": \"test\"}"
                }
            }

        Returns:
            JSON-serializable dict conforming to the OpenAI tool_call
            format.
        """
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.tool_name,
                "arguments": json.dumps(self.arguments),
            },
        }


# ---------------------------------------------------------------------------
# ToolResult dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    """The outcome of executing a ToolCall.

    Captures the execution status, output text, and optional error
    message. The agent loop appends these to the conversation history
    so the LLM can observe results and self-correct.

    Attributes:
        call_id: The call_id of the corresponding ToolCall.
        tool_name: The name of the tool that was executed.
        status: Outcome classification (success, error, denied, timeout).
        output: The tool's output text. May be empty on error/denial.
        error_message: Human-readable error description. None on success.
    """

    call_id: str
    tool_name: str
    status: ToolResultStatus
    output: str
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.call_id or not self.call_id.strip():
            raise ValueError("call_id must not be empty")
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")

    # -- Boolean convenience properties -----------------------------------

    @property
    def is_success(self) -> bool:
        """True if the tool executed successfully."""
        return self.status is ToolResultStatus.SUCCESS

    @property
    def is_error(self) -> bool:
        """True if the tool encountered an error."""
        return self.status is ToolResultStatus.ERROR

    @property
    def is_denied(self) -> bool:
        """True if the user denied the required approval."""
        return self.status is ToolResultStatus.DENIED

    @property
    def is_terminal(self) -> bool:
        """True if this result should terminate the agent loop.

        Delegates to ``ToolResultStatus.is_terminal``.
        """
        return self.status.is_terminal

    # -- Factory classmethods ---------------------------------------------

    @classmethod
    def success(
        cls,
        *,
        call_id: str,
        tool_name: str,
        output: str,
    ) -> ToolResult:
        """Create a successful ToolResult.

        Args:
            call_id: The call_id of the corresponding ToolCall.
            tool_name: Name of the tool that was executed.
            output: The tool's output text.

        Returns:
            A new ToolResult with SUCCESS status and no error.
        """
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            status=ToolResultStatus.SUCCESS,
            output=output,
            error_message=None,
        )

    @classmethod
    def error(
        cls,
        *,
        call_id: str,
        tool_name: str,
        error_message: str,
        output: str = "",
    ) -> ToolResult:
        """Create an error ToolResult.

        Args:
            call_id: The call_id of the corresponding ToolCall.
            tool_name: Name of the tool that was executed.
            error_message: Human-readable error description.
            output: Any partial output captured before the error.

        Returns:
            A new ToolResult with ERROR status.
        """
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            status=ToolResultStatus.ERROR,
            output=output,
            error_message=error_message,
        )

    @classmethod
    def denied(
        cls,
        *,
        call_id: str,
        tool_name: str,
        error_message: str,
    ) -> ToolResult:
        """Create a denied ToolResult.

        Args:
            call_id: The call_id of the corresponding ToolCall.
            tool_name: Name of the tool that was executed.
            error_message: Human-readable denial reason.

        Returns:
            A new ToolResult with DENIED status (terminal).
        """
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            status=ToolResultStatus.DENIED,
            output="",
            error_message=error_message,
        )

    # -- Serialization -----------------------------------------------------

    def to_llm_message(self) -> str:
        """Format this result as a human-readable string for LLM context.

        Provides a concise summary that includes the tool name, status,
        and relevant content (output on success, error on failure).
        Distinct from ``to_openai_tool_message()`` which produces an
        OpenAI API message dict.

        Returns:
            Formatted string suitable for embedding in conversation context.
        """
        if self.status is ToolResultStatus.SUCCESS:
            return f"[{self.tool_name}] Result (success):\n{self.output}"
        if self.status is ToolResultStatus.DENIED:
            return (
                f"[{self.tool_name}] DENIED: "
                f"{self.error_message or 'User denied the operation'}"
            )
        # ERROR or TIMEOUT
        label = self.status.value.upper()
        return (
            f"[{self.tool_name}] {label}: "
            f"{self.error_message or 'Unknown error'}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for persistence or logging.

        Returns:
            Dict with all fields. Status is serialized as its string value.
        """
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "output": self.output,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        """Deserialize from a plain dict.

        Args:
            data: Dict with required keys. The ``status`` field is
                parsed from its string value.

        Returns:
            New ToolResult instance.

        Raises:
            KeyError: If a required key is missing.
            ValueError: If status string is invalid.
        """
        return cls(
            call_id=data["call_id"],
            tool_name=data["tool_name"],
            status=ToolResultStatus(data["status"]),
            output=data["output"],
            error_message=data.get("error_message"),
        )

    def to_openai_tool_message(self) -> dict[str, Any]:
        """Serialize to an OpenAI ``role: "tool"`` message dict.

        The content field contains the output on success, or an
        ``ERROR: <message>`` prefix on failure. This gives the LLM
        clear signal about what happened.

        Returns:
            Dict with role, tool_call_id, and content keys.
        """
        if self.status is ToolResultStatus.SUCCESS:
            content = self.output
        else:
            error_text = self.error_message or f"Tool {self.tool_name} failed"
            content = f"ERROR: {error_text}"

        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "content": content,
        }
