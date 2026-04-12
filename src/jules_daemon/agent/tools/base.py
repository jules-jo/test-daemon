"""Base Tool protocol for agent loop tool implementations.

Defines the Protocol that all tool wrappers must implement, plus a
concrete BaseTool helper that provides the ToolSpec from class-level
attributes. Each tool wrapper inherits from BaseTool and implements
the async execute() method by delegating to existing daemon functions.

The Protocol uses runtime_checkable so the ToolRegistry can validate
tool implementations at registration time.

Usage::

    from jules_daemon.agent.tools.base import BaseTool, Tool

    class MyTool(BaseTool):
        spec = ToolSpec(...)

        async def execute(self, args: dict[str, Any]) -> ToolResult:
            ...
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jules_daemon.agent.tool_types import ToolResult, ToolSpec

__all__ = [
    "BaseTool",
    "Tool",
]


@runtime_checkable
class Tool(Protocol):
    """Protocol for agent loop tools.

    Each tool exposes a ToolSpec for LLM schema serialization and an
    async execute() method that wraps existing daemon functionality.
    """

    @property
    def spec(self) -> ToolSpec:
        """Return the tool specification for LLM schema generation."""
        ...

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given arguments.

        Args:
            args: Key-value arguments matching the ToolSpec parameters.

        Returns:
            ToolResult with success/error status and output text.
        """
        ...


class BaseTool:
    """Convenience base class for tool implementations.

    Subclasses set ``_spec`` as a class attribute and implement
    ``execute()``. This base class provides the ``spec`` property
    and ``name`` shortcut.

    Not required -- tools can implement the Tool protocol directly.
    """

    _spec: ToolSpec

    @property
    def spec(self) -> ToolSpec:
        """Return the tool specification."""
        return self._spec

    @property
    def name(self) -> str:
        """Shortcut for the tool name."""
        return self._spec.name

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool. Must be overridden by subclasses."""
        raise NotImplementedError
