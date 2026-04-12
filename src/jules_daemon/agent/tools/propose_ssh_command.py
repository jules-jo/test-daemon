"""propose_ssh_command tool -- proposes a command for human approval.

The LLM calls this tool when it has determined the SSH command to run.
The tool stores the proposal in a session-scoped approval ledger and
triggers the CONFIRM_PROMPT flow via IPC. The user must approve before
execute_ssh can run the command.

This is a state-changing tool (ApprovalRequirement.CONFIRM_PROMPT).

Design constraint: execute_ssh can only run commands that were previously
approved by propose_ssh_command in the same loop session.

Delegates to:
    - The IPC confirmation flow (via the confirm_callback injected at construction)

Usage::

    tool = ProposeSSHCommandTool(confirm_callback=handler.confirm)
    result = await tool.execute({
        "command": "python3 ~/agent_test.py --iterations 100",
        "target_host": "10.0.1.50",
        "target_user": "root",
        "explanation": "Running the agent test with 100 iterations",
    })
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolParam,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)
from jules_daemon.agent.tools.base import BaseTool

__all__ = ["ApprovalLedger", "ProposeSSHCommandTool"]

logger = logging.getLogger(__name__)


# Type alias for the async callback that handles user confirmation.
# Takes (command, target_host, explanation) and returns (approved, edited_command).
ConfirmCallback = Callable[
    [str, str, str],
    Awaitable[tuple[bool, str]],
]


@dataclass(frozen=True)
class ApprovalEntry:
    """Immutable record of an approved command in the session ledger.

    Attributes:
        approval_id: Unique identifier for this approval.
        command: The approved (possibly edited) command string.
        target_host: SSH target host.
        target_user: SSH target user.
    """

    approval_id: str
    command: str
    target_host: str
    target_user: str


class ApprovalLedger:
    """Session-scoped ledger of approved commands.

    Tracks which commands have been approved by propose_ssh_command so
    execute_ssh can enforce the approval constraint. Each loop session
    creates a fresh ledger.

    The ledger is append-only: approvals are never revoked. execute_ssh
    consumes an approval when it starts execution.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ApprovalEntry] = {}

    def record_approval(self, entry: ApprovalEntry) -> None:
        """Record a new approval in the ledger."""
        self._entries[entry.approval_id] = entry

    def get_approved_command(self, approval_id: str) -> ApprovalEntry | None:
        """Look up an approved command by its approval ID."""
        return self._entries.get(approval_id)

    def has_approved_command(self, command: str, target_host: str) -> str | None:
        """Check if a command+host pair has been approved.

        Returns the approval_id if found, None otherwise.
        """
        for entry in self._entries.values():
            if entry.command == command and entry.target_host == target_host:
                return entry.approval_id
        return None

    def consume(self, approval_id: str) -> ApprovalEntry | None:
        """Remove and return an approval entry (one-time use)."""
        return self._entries.pop(approval_id, None)

    @property
    def pending_count(self) -> int:
        """Number of unused approvals in the ledger."""
        return len(self._entries)


class ProposeSSHCommandTool(BaseTool):
    """Propose an SSH command for human approval.

    State-changing tool (ApprovalRequirement.CONFIRM_PROMPT). The user
    must approve the command via the IPC CONFIRM_PROMPT flow before
    execute_ssh can run it.

    The tool records approved commands in the shared ApprovalLedger
    so execute_ssh can verify the approval constraint.
    """

    _spec = ToolSpec(
        name="propose_ssh_command",
        description=(
            "Propose an SSH command for the user to approve before execution. "
            "The user will see the command and can approve, deny, or edit it. "
            "Returns the approval status and the final command (which may "
            "have been edited by the user). The approved command can then "
            "be passed to execute_ssh."
        ),
        parameters=(
            ToolParam(
                name="command",
                description="The SSH command to propose for execution",
                json_type="string",
            ),
            ToolParam(
                name="target_host",
                description="Remote hostname or IP address",
                json_type="string",
            ),
            ToolParam(
                name="target_user",
                description="SSH login username",
                json_type="string",
            ),
            ToolParam(
                name="explanation",
                description="Brief explanation of why this command is being proposed",
                json_type="string",
                required=False,
                default="",
            ),
        ),
        approval=ApprovalRequirement.CONFIRM_PROMPT,
    )

    def __init__(
        self,
        *,
        confirm_callback: ConfirmCallback,
        ledger: ApprovalLedger,
    ) -> None:
        self._confirm_callback = confirm_callback
        self._ledger = ledger

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Propose a command and wait for user approval.

        Delegates to the confirm_callback which handles the IPC
        CONFIRM_PROMPT/CONFIRM_REPLY exchange.
        """
        command = args.get("command", "")
        target_host = args.get("target_host", "")
        target_user = args.get("target_user", "")
        explanation = args.get("explanation", "")
        call_id = args.get("_call_id", "propose_ssh_command")

        if not command or not command.strip():
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message="command parameter is required",
            )
        if not target_host or not target_host.strip():
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message="target_host parameter is required",
            )
        if not target_user or not target_user.strip():
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message="target_user parameter is required",
            )

        try:
            approved, final_command = await self._confirm_callback(
                command.strip(),
                target_host.strip(),
                explanation,
            )

            if not approved:
                return ToolResult(
                    call_id=call_id,
                    tool_name=self.name,
                    status=ToolResultStatus.DENIED,
                    output=json.dumps({"approved": False}),
                    error_message="User denied the proposed command",
                )

            # Record approval in the session ledger
            approval_id = f"approval-{uuid.uuid4().hex[:12]}"
            entry = ApprovalEntry(
                approval_id=approval_id,
                command=final_command,
                target_host=target_host.strip(),
                target_user=target_user.strip(),
            )
            self._ledger.record_approval(entry)

            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.SUCCESS,
                output=json.dumps({
                    "approved": True,
                    "approval_id": approval_id,
                    "command": final_command,
                    "edited": final_command != command.strip(),
                }),
            )
        except Exception as exc:
            logger.warning("propose_ssh_command failed: %s", exc)
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=f"Proposal failed: {exc}",
            )
