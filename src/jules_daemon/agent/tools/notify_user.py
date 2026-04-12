"""notify_user tool -- pushes notification events to CLI subscribers.

Sends a notification message to any connected CLI clients via the
persistent notification subscription channel. Used for progress updates,
completion alerts, and anomaly warnings.

Delegates to:
    - The notification callback injected at construction (wraps the
      daemon's event bus or direct IPC push mechanism)

Usage::

    tool = NotifyUserTool(notify_callback=event_bus.push)
    result = await tool.execute({
        "message": "Test completed: 95 passed, 5 failed",
        "severity": "info",
    })
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolParam,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)
from jules_daemon.agent.tools.base import BaseTool

__all__ = ["NotifyUserTool"]

logger = logging.getLogger(__name__)

# Type alias for the async notification callback.
# Takes (message, severity) and returns success boolean.
NotifyCallback = Callable[
    [str, str],
    Awaitable[bool],
]

_VALID_SEVERITIES = ("info", "warning", "error", "success")


class NotifyUserTool(BaseTool):
    """Push a notification to connected CLI subscribers.

    Sends notification events through the daemon's persistent
    subscription channel. Connected CLIs receive these as real-time
    push events.

    This is a read-only tool (ApprovalRequirement.NONE) because it
    only pushes informational messages and does not change system state.
    """

    _spec = ToolSpec(
        name="notify_user",
        description=(
            "Send a notification message to the user's CLI. "
            "Use this for progress updates, completion alerts, "
            "and anomaly warnings. Messages are pushed in real-time "
            "to any connected CLI subscribers."
        ),
        parameters=(
            ToolParam(
                name="message",
                description="Notification message text",
                json_type="string",
            ),
            ToolParam(
                name="severity",
                description="Message severity level",
                json_type="string",
                required=False,
                default="info",
                enum=("info", "warning", "error", "success"),
            ),
        ),
        approval=ApprovalRequirement.NONE,
    )

    def __init__(self, *, notify_callback: NotifyCallback) -> None:
        self._notify_callback = notify_callback

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Send a notification to connected CLI subscribers.

        Delegates to the notify_callback which handles the IPC push.
        """
        message = args.get("message", "")
        severity = args.get("severity", "info")
        call_id = args.get("_call_id", "notify_user")

        if not message or not message.strip():
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message="message parameter is required",
            )

        if severity not in _VALID_SEVERITIES:
            severity = "info"

        try:
            delivered = await self._notify_callback(message.strip(), severity)

            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.SUCCESS,
                output=json.dumps({
                    "delivered": delivered,
                    "message": message.strip(),
                    "severity": severity,
                }),
            )
        except Exception as exc:
            logger.warning("notify_user failed: %s", exc)
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=f"Notification failed: {exc}",
            )
