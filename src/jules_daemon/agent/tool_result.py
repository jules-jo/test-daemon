"""Convenience re-export for ToolResult and ToolResultStatus.

Provides a short import path for the most commonly used types in tool
implementations::

    from jules_daemon.agent.tool_result import ToolResult, ToolResultStatus

The canonical definitions live in :mod:`jules_daemon.agent.tool_types`.
"""

from jules_daemon.agent.tool_types import ToolResult, ToolResultStatus

__all__ = [
    "ToolResult",
    "ToolResultStatus",
]
