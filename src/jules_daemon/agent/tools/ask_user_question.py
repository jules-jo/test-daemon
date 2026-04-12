"""ask_user_question tool -- asks the user a question via IPC.

When the LLM detects missing required arguments (e.g., from a test spec),
it calls this tool to ask the user directly. The daemon sends the question
via the IPC channel and waits for the user's response.

Constraint: the agent must NEVER guess or auto-default missing arguments.
It must always ask the user via this tool.

Delegates to:
    - The IPC question callback injected at construction (wraps the
      daemon's CONFIRM_PROMPT / CONFIRM_REPLY exchange)

Usage::

    tool = AskUserQuestionTool(ask_callback=handler.ask_user)
    result = await tool.execute({
        "question": "What iteration count should I use?",
        "context": "The test spec requires --iterations but no value was provided",
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

__all__ = ["AskUserQuestionTool"]

logger = logging.getLogger(__name__)

# Type alias for the async callback that sends a question to the user
# and returns their response. Takes (question, context) -> answer.
AskCallback = Callable[
    [str, str],
    Awaitable[str | None],
]


class AskUserQuestionTool(BaseTool):
    """Ask the user a question via the IPC channel.

    Used when the LLM needs information that cannot be inferred:
    - Missing required test arguments
    - Ambiguous user intent
    - Confirmation of non-obvious choices

    The tool sends the question through the daemon's IPC CONFIRM_PROMPT
    mechanism and returns the user's response. If the user cancels,
    a DENIED result is returned (which terminates the agent loop).

    This tool uses ApprovalRequirement.CONFIRM_PROMPT because it
    interacts with the user through the confirmation channel.
    """

    _spec = ToolSpec(
        name="ask_user_question",
        description=(
            "Ask the user a question to gather missing information. "
            "Use this when required arguments are missing from a test "
            "specification, when the user's intent is ambiguous, or "
            "when you need to confirm a non-obvious choice. Never guess "
            "or auto-default missing arguments -- always ask."
        ),
        parameters=(
            ToolParam(
                name="question",
                description="The question to ask the user",
                json_type="string",
            ),
            ToolParam(
                name="context",
                description=(
                    "Brief context explaining why the question is being asked "
                    "(shown to the user alongside the question)"
                ),
                json_type="string",
                required=False,
                default="",
            ),
        ),
        approval=ApprovalRequirement.CONFIRM_PROMPT,
    )

    def __init__(self, *, ask_callback: AskCallback) -> None:
        self._ask_callback = ask_callback

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Send a question to the user and return their response.

        Delegates to the ask_callback which handles the IPC exchange.
        """
        question = args.get("question", "")
        context = args.get("context", "")
        call_id = args.get("_call_id", "ask_user_question")

        if not question or not question.strip():
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message="question parameter is required",
            )

        try:
            answer = await self._ask_callback(question.strip(), context)

            if answer is None:
                return ToolResult(
                    call_id=call_id,
                    tool_name=self.name,
                    status=ToolResultStatus.DENIED,
                    output=json.dumps({"cancelled": True}),
                    error_message="User cancelled the question",
                )

            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.SUCCESS,
                output=json.dumps({
                    "answer": answer,
                    "question": question.strip(),
                }),
            )
        except Exception as exc:
            logger.warning("ask_user_question failed: %s", exc)
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=f"Failed to ask user: {exc}",
            )
