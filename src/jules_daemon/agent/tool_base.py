"""Tool protocol and InfoRetrievalTool base class.

Defines the structural interface that all agent tools must satisfy (the
``Tool`` protocol) and provides a shared base class for read-only /
information-retrieval tools (``InfoRetrievalTool``) with common argument
validation, error handling, and result formatting.

The Tool protocol uses Python's ``Protocol`` (structural subtyping) so
that tool implementations are not forced into an inheritance hierarchy.
``InfoRetrievalTool`` is a convenience ABC that satisfies the protocol
and handles the repetitive validation/error-catching boilerplate.

Tool classification:

    Read-only tools (InfoRetrievalTool):
        read_wiki, lookup_test_spec, check_remote_processes,
        read_output, parse_test_output
        -- execute freely without human approval.

    State-changing tools:
        propose_ssh_command, execute_ssh
        -- require explicit human approval via CONFIRM_PROMPT.

Usage::

    from jules_daemon.agent.tool_base import InfoRetrievalTool, Tool
    from jules_daemon.agent.tool_result import ToolResult

    class ReadWikiTool(InfoRetrievalTool):
        @property
        def name(self) -> str:
            return "read_wiki"
        ...
        async def _execute_impl(self, *, call_id, args):
            ...
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Protocol, runtime_checkable

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolParam,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "InfoRetrievalTool",
    "Tool",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool protocol (structural subtyping)
# ---------------------------------------------------------------------------


@runtime_checkable
class Tool(Protocol):
    """Structural interface for all agent-loop tools.

    Any object that exposes these attributes/methods satisfies the
    protocol -- no inheritance required. Use ``isinstance(obj, Tool)``
    at runtime thanks to ``@runtime_checkable``.

    Attributes:
        name: Unique tool identifier (used as function name in OpenAI API).
        description: Human-readable description shown to the LLM.
        parameters_schema: JSON Schema dict describing accepted arguments.
        requires_human_approval: Whether execution requires user confirmation.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters_schema(self) -> dict[str, Any]: ...

    @property
    def requires_human_approval(self) -> bool: ...

    async def execute(self, call_id: str, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given arguments.

        Args:
            call_id: Unique identifier tying this invocation to an LLM
                tool_call. Used to correlate the ToolResult back into the
                conversation history.
            args: Key-value arguments parsed from the LLM tool call.

        Returns:
            ToolResult capturing the outcome (success, error, or denied).
        """
        ...


# ---------------------------------------------------------------------------
# InfoRetrievalTool base class (read-only tools)
# ---------------------------------------------------------------------------


class InfoRetrievalTool(abc.ABC):
    """Shared base class for read-only information-retrieval tools.

    Provides:
    - Automatic argument validation against ``parameters_schema``
    - Exception-safe execution wrapping (``_execute_impl`` errors become
      error ToolResults instead of propagating)
    - OpenAI-compatible schema serialization (``to_openai_schema``)
    - Conversion to ``ToolSpec`` for registry integration

    Subclasses must implement:
    - ``name`` property
    - ``description`` property
    - ``parameters_schema`` property (JSON Schema dict)
    - ``_execute_impl(*, call_id, args)`` coroutine

    ``requires_human_approval`` defaults to ``False``. State-changing
    tools should NOT extend this class -- they need a separate base
    with approval flow integration.
    """

    # -- Protocol-required attributes (all read-only) ----------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique tool name (function name in OpenAI API)."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description shown to the LLM."""

    @property
    @abc.abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema dict describing accepted arguments.

        Must be an object schema with ``type: "object"`` and a
        ``properties`` dict. Optional ``required`` list specifies
        which parameters are mandatory.

        Example::

            {
                "type": "object",
                "properties": {
                    "slug": {
                        "type": "string",
                        "description": "Wiki page slug"
                    }
                },
                "required": ["slug"]
            }
        """

    @property
    def requires_human_approval(self) -> bool:
        """Read-only tools do not require human approval."""
        return False

    # -- Abstract execution hook -------------------------------------------

    @abc.abstractmethod
    async def _execute_impl(
        self, *, call_id: str, args: dict[str, Any]
    ) -> ToolResult:
        """Tool-specific execution logic.

        Subclasses implement their core functionality here. The base
        class ``execute()`` method handles validation and error catching
        before delegating to this method.

        Args:
            call_id: Unique identifier for this invocation.
            args: Validated arguments (required params guaranteed present).

        Returns:
            ToolResult with the execution outcome.

        Raises:
            Any exception -- the base class ``execute()`` will catch it
            and convert it to an error ToolResult.
        """

    # -- Public entry point ------------------------------------------------

    async def execute(self, call_id: str, args: dict[str, Any]) -> ToolResult:
        """Validate arguments, execute the tool, and catch errors.

        This is the public method called by the agent loop. It:
        1. Validates that ``call_id`` is non-empty.
        2. Normalizes ``args`` (None -> empty dict).
        3. Checks that all required parameters are present.
        4. Delegates to ``_execute_impl``.
        5. Catches any exception and wraps it in an error ToolResult.

        Args:
            call_id: Unique identifier for this invocation.
            args: Arguments from the LLM tool call.

        Returns:
            ToolResult -- never raises.
        """
        # Validate call_id
        safe_call_id = call_id if call_id and call_id.strip() else ""
        if not safe_call_id:
            # Best-effort sentinel call_id so the agent loop can still
            # associate this error result in the conversation history.
            # The literal "unknown" is used because ToolResult.__post_init__
            # rejects empty strings.
            return ToolResult.error(
                call_id="unknown",
                tool_name=self.name,
                error_message=f"Tool {self.name}: call_id must not be empty",
            )

        # Normalize args
        effective_args: dict[str, Any] = dict(args) if args else {}

        # Validate required parameters
        validation_error = self._validate_required_params(effective_args)
        if validation_error is not None:
            return ToolResult.error(
                call_id=safe_call_id,
                tool_name=self.name,
                error_message=validation_error,
            )

        # Execute with error catching
        try:
            return await self._execute_impl(
                call_id=safe_call_id, args=effective_args
            )
        except Exception as exc:
            logger.warning(
                "Tool %s raised %s: %s",
                self.name,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return ToolResult.error(
                call_id=safe_call_id,
                tool_name=self.name,
                error_message=f"{type(exc).__name__}: {exc}",
            )

    # -- Schema serialization ----------------------------------------------

    def to_openai_schema(self) -> dict[str, Any]:
        """Serialize to an OpenAI-compatible function tool schema.

        Produces a dict matching the ``tools`` array element format
        expected by the Chat Completions API::

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": { ... }
                }
            }

        Returns:
            JSON-serializable dict.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def to_tool_spec(self) -> ToolSpec:
        """Convert to a ToolSpec for ToolRegistry integration.

        Extracts parameter definitions from ``parameters_schema`` and
        maps them to ``ToolParam`` instances. The approval requirement
        is derived from ``requires_human_approval``.

        Returns:
            Immutable ToolSpec describing this tool.
        """
        schema = self.parameters_schema
        properties: dict[str, Any] = schema.get("properties", {})
        required_names: list[str] = schema.get("required", [])

        params: list[ToolParam] = []
        for param_name, param_schema in properties.items():
            params.append(
                ToolParam(
                    name=param_name,
                    description=param_schema.get("description", param_name),
                    json_type=param_schema.get("type", "string"),
                    required=param_name in required_names,
                    default=param_schema.get("default"),
                    enum=tuple(param_schema["enum"])
                    if "enum" in param_schema
                    else None,
                )
            )

        approval = (
            ApprovalRequirement.CONFIRM_PROMPT
            if self.requires_human_approval
            else ApprovalRequirement.NONE
        )

        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=tuple(params),
            approval=approval,
        )

    # -- Internal helpers --------------------------------------------------

    def _validate_required_params(self, args: dict[str, Any]) -> str | None:
        """Check that all required parameters are present in args.

        Args:
            args: The arguments dict to validate.

        Returns:
            An error message string if validation fails, or None if all
            required parameters are present.
        """
        schema = self.parameters_schema
        required: list[str] = schema.get("required", [])
        missing = [name for name in required if name not in args]
        if missing:
            return (
                f"Tool {self.name}: missing required parameter(s): "
                f"{', '.join(sorted(missing))}"
            )
        return None
