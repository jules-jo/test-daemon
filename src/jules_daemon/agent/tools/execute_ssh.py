"""execute_ssh tool -- wraps execution.run_pipeline.execute_run.

Executes a previously approved SSH command on a remote host. The tool
enforces the approval constraint: it only runs commands that were
approved by propose_ssh_command in the same loop session.

Integrated human-approval gate:
    Before executing the command, execute_ssh sends a confirmation
    prompt to the user via the IPC confirm_callback and blocks until
    an explicit "yes" is received. This is a second-level gate:
    propose_ssh_command handles the initial proposal approval, and
    execute_ssh handles the final execution confirmation.

Delegates to:
    - jules_daemon.execution.run_pipeline.execute_run

Design constraints:
    - execute_ssh can ONLY be called with a command previously approved
      by propose_ssh_command in the same loop session.
    - Requires ApprovalRequirement.CONFIRM_PROMPT.
    - The approval_id from propose_ssh_command must be passed as a parameter.
    - The confirm_callback blocks until the user explicitly confirms execution.
    - User denial returns DENIED status (terminal -- agent loop stops).

Usage::

    tool = ExecuteSSHTool(
        wiki_root=Path("/data/wiki"),
        ledger=shared_approval_ledger,
        confirm_callback=my_confirm_callback,
    )
    result = await tool.execute({
        "approval_id": "approval-abc123",
    })
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolParam,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)
from jules_daemon.agent.tools.base import BaseTool
from jules_daemon.agent.tools.propose_ssh_command import (
    ApprovalLedger,
    ConfirmCallback,
)

__all__ = ["ExecuteSSHTool"]

logger = logging.getLogger(__name__)

_STDOUT_CAP: int = 4000
"""Maximum bytes of stdout to include in the result (prevents bloat)."""

_STDERR_CAP: int = 2000
"""Maximum bytes of stderr to include in the result."""


class ExecuteSSHTool(BaseTool):
    """Execute a previously approved SSH command with a human confirmation gate.

    Two-level approval enforcement:
        1. The ``approval_id`` must reference a command previously approved
           by ``propose_ssh_command`` in the same session (ledger check).
        2. Before execution, the tool sends a confirmation prompt via the
           ``confirm_callback`` and blocks until the user explicitly confirms.
           The user can deny or edit the command at this stage.

    If the user denies execution, the tool returns DENIED status (terminal).
    If the user confirms, the tool delegates to ``execute_run`` and returns
    the structured result with stdout, stderr, and exit code.

    Wraps:
        - execution.run_pipeline.execute_run (full SSH execution pipeline)

    This is a state-changing tool (ApprovalRequirement.CONFIRM_PROMPT).
    """

    _spec = ToolSpec(
        name="execute_ssh",
        description=(
            "Execute a previously approved SSH command on the remote host. "
            "You must first use propose_ssh_command to get an approval_id, "
            "then pass that approval_id here. The command and target host "
            "are retrieved from the approval record. The user will be asked "
            "to confirm execution before the command runs. Returns the "
            "execution result including exit code, stdout, and stderr."
        ),
        parameters=(
            ToolParam(
                name="approval_id",
                description=(
                    "The approval_id returned by propose_ssh_command. "
                    "This links to the approved command and target."
                ),
                json_type="string",
            ),
            ToolParam(
                name="timeout",
                description="Maximum execution time in seconds (default 3600)",
                json_type="integer",
                required=False,
                default=3600,
            ),
        ),
        approval=ApprovalRequirement.CONFIRM_PROMPT,
    )

    def __init__(
        self,
        *,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_callback: ConfirmCallback,
    ) -> None:
        self._wiki_root = wiki_root
        self._ledger = ledger
        self._confirm_callback = confirm_callback

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute the approved SSH command after human confirmation.

        Flow:
            1. Validate the approval_id parameter.
            2. Look up (without consuming) the approval entry in the ledger.
            3. Send confirmation prompt to user via confirm_callback (blocks).
            4. If denied: return DENIED result (terminal).
            5. If confirmed: consume the approval and delegate to execute_run.
            6. Return structured result with stdout, stderr, exit_code.

        The approval is only consumed (step 5) after the user confirms,
        ensuring that a denied execution does not invalidate the approval
        for a subsequent attempt.
        """
        approval_id = args.get("approval_id", "")
        timeout = args.get("timeout", 3600)
        call_id = args.get("_call_id", "execute_ssh")

        # Step 1: Validate approval_id parameter
        if not approval_id or not approval_id.strip():
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=(
                    "approval_id is required. Use propose_ssh_command first "
                    "to get an approval_id."
                ),
            )

        approval_id = approval_id.strip()

        # Step 2: Look up (peek) the approval entry without consuming
        entry = self._ledger.get_approved_command(approval_id)
        if entry is None:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=(
                    f"No approved command found for approval_id={approval_id}. "
                    "Use propose_ssh_command first to get user approval."
                ),
            )

        # Step 3: Human approval gate -- block until user confirms
        logger.info(
            "Requesting execution confirmation for approval_id=%s: "
            "%s on %s@%s",
            entry.approval_id,
            entry.command[:80],
            entry.target_user,
            entry.target_host,
        )

        try:
            explanation = (
                f"Executing approved command (approval_id={entry.approval_id})"
            )
            approved, final_command = await self._confirm_callback(
                entry.command,
                entry.target_host,
                explanation,
            )
        except Exception as exc:
            logger.warning(
                "Execution confirmation failed for approval_id=%s: %s",
                approval_id, exc,
            )
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=f"Execution confirmation failed: {exc}",
            )

        # Step 4: If denied, return DENIED (terminal -- loop stops)
        if not approved:
            logger.info(
                "User denied execution for approval_id=%s",
                approval_id,
            )
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.DENIED,
                output=json.dumps({
                    "approved": False,
                    "approval_id": approval_id,
                    "command": entry.command,
                }),
                error_message="User denied command execution",
            )

        # Step 5: Consume the approval (one-time use) now that user confirmed
        consumed = self._ledger.consume(approval_id)
        if consumed is None:
            # Race condition guard: approval was consumed between peek and consume
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=(
                    f"Approval {approval_id} was consumed by another call. "
                    "Use propose_ssh_command to get a new approval."
                ),
            )

        # Use the final command (may have been edited during confirmation)
        command_to_execute = final_command if final_command else consumed.command

        logger.info(
            "Executing approved command (approval_id=%s): %s on %s@%s",
            consumed.approval_id,
            command_to_execute[:80],
            consumed.target_user,
            consumed.target_host,
        )

        # Step 6: Delegate to the execution pipeline
        try:
            from jules_daemon.execution.run_pipeline import execute_run

            run_result = await execute_run(
                target_host=consumed.target_host,
                target_user=consumed.target_user,
                command=command_to_execute,
                wiki_root=self._wiki_root,
                timeout=int(timeout),
            )

            result_data = {
                "success": run_result.success,
                "run_id": run_result.run_id,
                "command": run_result.command,
                "target_host": run_result.target_host,
                "target_user": run_result.target_user,
                "exit_code": run_result.exit_code,
                "stdout": run_result.stdout[:_STDOUT_CAP],
                "stderr": run_result.stderr[:_STDERR_CAP],
                "error": run_result.error,
                "duration_seconds": run_result.duration_seconds,
            }

            status = (
                ToolResultStatus.SUCCESS
                if run_result.success
                else ToolResultStatus.ERROR
            )

            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=status,
                output=json.dumps(result_data, default=str),
                error_message=run_result.error if not run_result.success else None,
            )
        except Exception as exc:
            logger.error(
                "execute_ssh failed for approval_id=%s: %s",
                approval_id, exc,
            )
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=f"SSH execution failed: {exc}",
            )
