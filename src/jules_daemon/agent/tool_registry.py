"""ToolRegistry -- stores Tool instances by name with validation.

The registry is the central hub for all agent loop tools. It provides:

    - ``register(tool)``        -- add a tool by its spec name
    - ``get(name)``             -- look up a tool by name (None if missing)
    - ``list_tools()``          -- return all registered tools
    - ``list_tool_names()``     -- return sorted tuple of tool names
    - ``to_openai_schemas()``   -- serialize all specs to OpenAI format
    - ``validate_call(call)``   -- check a ToolCall against the schema
    - ``execute(call)``         -- validate then delegate to the tool

Classification helpers:

    - ``list_read_only_tools()``          -- tools with no approval required
    - ``list_approval_required_tools()``  -- tools requiring CONFIRM_PROMPT
    - ``requires_approval(name)``         -- check a specific tool

The registry is NOT frozen/immutable -- tools are registered at startup
and remain for the daemon's lifetime. However, it returns defensive
copies (tuples) from all query methods to prevent external mutation
of internal state.

Custom exceptions:

    - ``ToolRegistryError``     -- base error for registry operations
    - ``ToolValidationError``   -- raised when a ToolCall fails validation

Usage::

    from jules_daemon.agent.tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(ReadWikiTool(wiki_root=wiki_path))
    registry.register(LookupTestSpecTool(wiki_root=wiki_path))

    # Serialize for OpenAI API
    schemas = registry.to_openai_schemas()

    # Validate and execute a tool call
    result = await registry.execute(tool_call)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from jules_daemon.agent.tools.base import BaseTool

__all__ = [
    "ToolRegistry",
    "ToolRegistryError",
    "ToolValidationError",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolRegistryError(Exception):
    """Base error for ToolRegistry operations."""


class ToolValidationError(ToolRegistryError):
    """Raised when a ToolCall fails schema validation.

    Attributes:
        tool_name: The tool name from the failed call.
        missing_params: Tuple of required parameter names that were missing.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        missing_params: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.missing_params = missing_params


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Stores Tool instances by name with schema validation.

    Tools are registered once at startup and remain for the daemon's
    lifetime. The registry validates tool calls against registered
    parameter schemas before delegating to the tool's execute() method.

    Thread-safety: The registry is designed for single-threaded asyncio
    usage. Registration happens at startup; lookups and execution happen
    during the agent loop. No locking is required.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # -- Container protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self._tools)

    def __bool__(self) -> bool:
        return len(self._tools) > 0

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __repr__(self) -> str:
        count = len(self._tools)
        names = ", ".join(sorted(self._tools.keys()))
        return f"ToolRegistry({count} tools: [{names}])"

    # -- Registration --------------------------------------------------------

    def register(self, tool: BaseTool) -> ToolRegistry:
        """Register a tool by its spec name.

        Args:
            tool: A BaseTool instance with a valid ToolSpec.

        Returns:
            Self, for fluent chaining.

        Raises:
            ToolRegistryError: If a tool with the same name is already
                registered.
        """
        name = tool.spec.name
        if name in self._tools:
            raise ToolRegistryError(
                f"Tool '{name}' is already registered in the registry"
            )
        self._tools[name] = tool
        logger.debug("Registered tool: %s", name)
        return self

    # -- Lookup --------------------------------------------------------------

    def get(self, name: str) -> BaseTool | None:
        """Look up a tool by name.

        Args:
            name: The tool name to look up.

        Returns:
            The registered BaseTool instance, or None if not found.
        """
        return self._tools.get(name)

    def get_spec(self, name: str) -> ToolSpec | None:
        """Get a tool's specification by name.

        Args:
            name: The tool name to look up.

        Returns:
            The tool's ToolSpec, or None if not found.
        """
        tool = self._tools.get(name)
        if tool is None:
            return None
        return tool.spec

    # -- Listing -------------------------------------------------------------

    def list_tools(self) -> tuple[BaseTool, ...]:
        """Return all registered tools as an immutable tuple.

        Returns:
            Tuple of all registered BaseTool instances. Order is
            insertion order (Python 3.7+ dict ordering).
        """
        return tuple(self._tools.values())

    def list_tool_names(self) -> tuple[str, ...]:
        """Return sorted tuple of all registered tool names.

        Returns:
            Sorted tuple of tool name strings.
        """
        return tuple(sorted(self._tools.keys()))

    # -- Classification ------------------------------------------------------

    def list_read_only_tools(self) -> tuple[BaseTool, ...]:
        """Return tools that do not require human approval.

        Returns:
            Tuple of tools with ApprovalRequirement.NONE.
        """
        return tuple(
            tool
            for tool in self._tools.values()
            if tool.spec.approval is ApprovalRequirement.NONE
        )

    def list_approval_required_tools(self) -> tuple[BaseTool, ...]:
        """Return tools that require human approval.

        Returns:
            Tuple of tools with ApprovalRequirement.CONFIRM_PROMPT.
        """
        return tuple(
            tool
            for tool in self._tools.values()
            if tool.spec.approval is ApprovalRequirement.CONFIRM_PROMPT
        )

    def requires_approval(self, name: str) -> bool:
        """Check whether a tool requires human approval.

        Args:
            name: The tool name to check.

        Returns:
            True if the tool requires CONFIRM_PROMPT approval.

        Raises:
            ToolRegistryError: If the tool is not registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolRegistryError(
                f"Tool '{name}' is not registered in the registry"
            )
        return tool.spec.approval is ApprovalRequirement.CONFIRM_PROMPT

    # -- OpenAI schema serialization -----------------------------------------

    def to_openai_schemas(self) -> tuple[dict[str, Any], ...]:
        """Serialize all tool specs to OpenAI-compatible function schemas.

        Returns:
            Tuple of dicts conforming to the OpenAI ``tools`` array format.
            Each element has ``type: "function"`` with a nested ``function``
            object containing name, description, and parameters.
        """
        return tuple(
            tool.spec.to_openai_function_schema()
            for tool in self._tools.values()
        )

    # -- Validation ----------------------------------------------------------

    def validate_call(self, call: ToolCall) -> None:
        """Validate a ToolCall against the registered tool's schema.

        Checks:
            1. The tool name is registered.
            2. All required parameters are present in the arguments.

        Extra parameters beyond the schema are tolerated -- LLMs sometimes
        hallucinate extra fields, and strict rejection would cause unnecessary
        failures in the agent loop.

        Args:
            call: The ToolCall to validate.

        Raises:
            ToolValidationError: If the tool is not registered or required
                parameters are missing.
        """
        tool = self._tools.get(call.tool_name)
        if tool is None:
            raise ToolValidationError(
                f"Tool '{call.tool_name}' is not registered in the registry",
                tool_name=call.tool_name,
            )

        spec = tool.spec
        required_params = [p.name for p in spec.parameters if p.required]
        missing = [
            name for name in required_params if name not in call.arguments
        ]

        if missing:
            raise ToolValidationError(
                f"Tool '{call.tool_name}' missing required parameters: "
                f"{', '.join(missing)}",
                tool_name=call.tool_name,
                missing_params=tuple(missing),
            )

    # -- Execution -----------------------------------------------------------

    async def execute(self, call: ToolCall) -> ToolResult:
        """Validate a ToolCall and delegate execution to the tool.

        This method:
            1. Validates the call against the registered schema.
            2. Injects ``_call_id`` into the arguments dict.
            3. Delegates to the tool's ``execute()`` method.
            4. Catches any unhandled exceptions and wraps them
               as error ToolResults.

        Args:
            call: The ToolCall to execute.

        Returns:
            ToolResult from the tool, or an error ToolResult if
            validation or execution fails.
        """
        # Validate the call
        try:
            self.validate_call(call)
        except ToolValidationError as exc:
            logger.warning("Tool call validation failed: %s", exc)
            return ToolResult.error(
                call_id=call.call_id,
                tool_name=call.tool_name,
                error_message=str(exc),
            )

        tool = self._tools[call.tool_name]

        # Build args with injected _call_id (defensive copy)
        args = dict(call.arguments)
        args["_call_id"] = call.call_id

        try:
            return await tool.execute(args)
        except Exception as exc:
            logger.warning(
                "Tool '%s' raised during execution: %s",
                call.tool_name,
                exc,
            )
            return ToolResult.error(
                call_id=call.call_id,
                tool_name=call.tool_name,
                error_message=f"Tool execution failed: {exc}",
            )
