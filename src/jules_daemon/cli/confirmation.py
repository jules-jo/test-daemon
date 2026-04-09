"""Terminal-based editable confirmation prompt for SSH commands.

Displays the SSH command with its expanded context (risk level,
explanation, affected paths, target host) and allows the user to
inline-edit the command, then approve or reject it.

Security invariant: No SSH command is ever executed without explicit
human approval through this confirmation flow.

The IO layer is abstracted behind ``TerminalIO`` for testability.
Production code uses ``DefaultTerminalIO`` which reads from stdin
and writes to stdout. Tests inject ``FakeTerminalIO`` for
deterministic control.

Usage::

    from jules_daemon.cli.confirmation import (
        ConfirmationRequest,
        confirm_ssh_command,
    )

    request = ConfirmationRequest(
        ssh_command=ssh_cmd,
        context=command_context,
        target=ssh_target,
    )
    result = confirm_ssh_command(request)

    if result.is_approved:
        execute(result.final_command)
    else:
        log("User rejected command")
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TextIO

from pydantic import ValidationError

from jules_daemon.llm.command_context import CommandContext, RiskLevel
from jules_daemon.ssh.command import SSHCommand
from jules_daemon.wiki.models import SSHTarget

__all__ = [
    "ConfirmationRequest",
    "ConfirmationResult",
    "Decision",
    "DefaultTerminalIO",
    "TerminalIO",
    "confirm_ssh_command",
    "render_confirmation_display",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------


class Decision(Enum):
    """User's final decision on an SSH command.

    Values:
        APPROVE: Execute the command (possibly after editing).
        REJECT: Do not execute the command.
    """

    APPROVE = "approve"
    REJECT = "reject"


# ---------------------------------------------------------------------------
# Request / Result data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfirmationRequest:
    """Immutable input to the confirmation prompt.

    Attributes:
        ssh_command: The SSH command proposed for execution.
        context: LLM-generated analysis of the command (risk, explanation).
        target: Remote host details. None if not yet resolved.
    """

    ssh_command: SSHCommand
    context: CommandContext
    target: SSHTarget | None


@dataclass(frozen=True)
class ConfirmationResult:
    """Immutable output from the confirmation prompt.

    Attributes:
        decision: Whether the user approved or rejected.
        final_command: The command as-is or after editing.
        was_edited: True if the user modified the command text.
    """

    decision: Decision
    final_command: SSHCommand
    was_edited: bool

    @property
    def is_approved(self) -> bool:
        """Convenience: True when the decision is APPROVE."""
        return self.decision == Decision.APPROVE


# ---------------------------------------------------------------------------
# IO abstraction
# ---------------------------------------------------------------------------


class TerminalIO(ABC):
    """Abstract terminal IO for testability.

    Production uses stdin/stdout. Tests inject a fake that records
    output and replays scripted input.
    """

    @abstractmethod
    def write(self, text: str) -> None:
        """Write text to the terminal output."""

    @abstractmethod
    def read_line(self, prompt: str = "") -> str:
        """Read a single line of user input, stripping the trailing newline.

        Raises:
            EOFError: When input is exhausted (Ctrl-D / pipe closed).
        """

    @abstractmethod
    def read_editable(self, prompt: str, prefill: str) -> str:
        """Read a line with pre-filled editable text.

        The ``prefill`` value is presented as the default; the user can
        modify it inline or press Enter to accept as-is.

        Returns the final text (may equal prefill if unmodified).

        Raises:
            EOFError: When input is exhausted.
        """


class DefaultTerminalIO(TerminalIO):
    """Production terminal IO using stdin/stdout.

    Uses GNU readline when available for command editing with
    pre-filled text. Falls back to a simple prompt when readline
    is not available.
    """

    def __init__(
        self,
        *,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stdout

    def write(self, text: str) -> None:
        self._output.write(text)
        self._output.flush()

    def read_line(self, prompt: str = "") -> str:
        try:
            if self._input is sys.stdin:
                return input(prompt)
            self._output.write(prompt)
            self._output.flush()
            line = self._input.readline()
            if not line:
                raise EOFError("Input stream exhausted")
            return line.rstrip("\n")
        except KeyboardInterrupt:
            raise EOFError("User interrupted input") from None

    def read_editable(self, prompt: str, prefill: str) -> str:
        # Readline pre-fill only works with real stdin
        if self._input is sys.stdin:
            try:
                import readline

                def _prefill_hook() -> None:
                    readline.insert_text(prefill)
                    readline.redisplay()

                readline.set_startup_hook(_prefill_hook)
                try:
                    result = input(prompt)
                finally:
                    readline.set_startup_hook(None)
                return result
            except ImportError:
                pass
            except KeyboardInterrupt:
                raise EOFError("User interrupted input") from None

        # Fallback for non-stdin streams or when readline unavailable
        self.write(f"  Current: {prefill}\n")
        return self.read_line(prompt)


# ---------------------------------------------------------------------------
# Input classification
# ---------------------------------------------------------------------------

# Normalized input values that map to each action
_APPROVE_INPUTS = frozenset({"a", "y", "yes", "approve", "allow"})
_REJECT_INPUTS = frozenset({"r", "n", "d", "no", "reject", "deny"})
_EDIT_INPUTS = frozenset({"e", "edit"})


def _classify_input(raw: str) -> str:
    """Classify user input into 'approve', 'reject', 'edit', or 'unknown'.

    Args:
        raw: Raw user input string.

    Returns:
        One of 'approve', 'reject', 'edit', 'unknown'.
    """
    normalized = raw.strip().lower()
    if normalized in _APPROVE_INPUTS:
        return "approve"
    if normalized in _REJECT_INPUTS:
        return "reject"
    if normalized in _EDIT_INPUTS:
        return "edit"
    return "unknown"


# ---------------------------------------------------------------------------
# Display rendering
# ---------------------------------------------------------------------------

_SEPARATOR = "-" * 60
_RISK_LABELS: dict[RiskLevel, str] = {
    RiskLevel.LOW: "LOW",
    RiskLevel.MEDIUM: "MEDIUM",
    RiskLevel.HIGH: "HIGH",
    RiskLevel.CRITICAL: "CRITICAL",
}


def render_confirmation_display(request: ConfirmationRequest) -> str:
    """Render the confirmation display as a formatted string.

    Produces a human-readable block showing the command, its context,
    risk analysis, and instructions for the user.

    Args:
        request: The confirmation request containing command and context.

    Returns:
        Formatted multi-line string ready for terminal display.
    """
    lines: list[str] = []
    ctx = request.context
    cmd = request.ssh_command
    risk_label = _RISK_LABELS[ctx.risk_level]

    lines.append("")
    lines.append(_SEPARATOR)
    lines.append("  SSH Command Confirmation")
    lines.append(_SEPARATOR)

    # Critical risk warning
    if ctx.risk_level == RiskLevel.CRITICAL:
        lines.append("")
        lines.append("  *** WARNING: CRITICAL risk operation ***")
        lines.append("  This command may cause irreversible changes.")

    # Target info
    if request.target is not None:
        target = request.target
        port_str = f":{target.port}" if target.port != 22 else ""
        lines.append("")
        lines.append(f"  Target:    {target.user}@{target.host}{port_str}")

    # Command
    lines.append("")
    lines.append(f"  Command:   {cmd.command}")

    # Working directory
    if cmd.working_directory is not None:
        lines.append(f"  Directory: {cmd.working_directory}")

    # Timeout
    lines.append(f"  Timeout:   {cmd.timeout}s")

    # Environment variables
    if cmd.environment:
        lines.append("  Environment:")
        for key, value in sorted(cmd.environment.items()):
            lines.append(f"    {key}={value}")

    # Explanation
    lines.append("")
    lines.append(f"  Explanation: {ctx.explanation}")

    # Risk level
    lines.append(f"  Risk level:  {risk_label}")

    # Safety recommendation
    if not ctx.safe_to_execute:
        lines.append("  Safety:      NOT recommended for execution")

    # Affected paths
    if ctx.affected_paths:
        lines.append("")
        lines.append("  Affected paths:")
        for path in ctx.affected_paths:
            lines.append(f"    - {path}")

    # Risk factors
    if ctx.risk_factors:
        lines.append("")
        lines.append("  Risk factors:")
        for factor in ctx.risk_factors:
            lines.append(f"    - {factor}")

    # Action instructions
    lines.append("")
    lines.append(_SEPARATOR)
    lines.append("  [A]pprove  [R]eject  [E]dit command")
    lines.append(_SEPARATOR)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main confirmation function
# ---------------------------------------------------------------------------


def confirm_ssh_command(
    request: ConfirmationRequest,
    *,
    terminal: TerminalIO | None = None,
) -> ConfirmationResult:
    """Display an SSH command with context and prompt for user decision.

    Shows the command, its LLM-generated context (risk level, explanation,
    affected paths), and presents three options:
        [A]pprove -- execute the command as shown
        [R]eject  -- do not execute the command
        [E]dit    -- inline-edit the command, then approve or reject

    The function loops until the user provides a valid approve/reject
    decision. EOF (Ctrl-D) or KeyboardInterrupt (Ctrl-C) are treated
    as rejection for safety.

    Args:
        request: ConfirmationRequest with the command and its context.
        terminal: IO abstraction. Defaults to DefaultTerminalIO (stdin/stdout).

    Returns:
        ConfirmationResult with the user's decision and final command.
    """
    io = terminal or DefaultTerminalIO()

    # Render and display the context
    display = render_confirmation_display(request)
    io.write(display)

    current_command = request.ssh_command
    was_edited = False

    while True:
        try:
            raw = io.read_line("  Decision: ")
        except EOFError:
            logger.info("EOF received during confirmation -- treating as reject")
            return ConfirmationResult(
                decision=Decision.REJECT,
                final_command=current_command,
                was_edited=was_edited,
            )

        action = _classify_input(raw)

        if action == "approve":
            return ConfirmationResult(
                decision=Decision.APPROVE,
                final_command=current_command,
                was_edited=was_edited,
            )

        if action == "reject":
            return ConfirmationResult(
                decision=Decision.REJECT,
                final_command=current_command,
                was_edited=was_edited,
            )

        if action == "edit":
            edited_command = _handle_edit(current_command, io)
            if edited_command is not None:
                if edited_command.command != request.ssh_command.command:
                    was_edited = True
                current_command = edited_command
            # After editing, loop back to approve/reject prompt
            continue

        # Unknown input -- reprompt
        io.write("  Invalid input. Use [A]pprove, [R]eject, or [E]dit.\n")


def _handle_edit(
    current: SSHCommand,
    io: TerminalIO,
) -> SSHCommand | None:
    """Prompt the user to inline-edit the command text.

    Loops until the user provides a valid (non-empty) command string.
    Returns a new SSHCommand with the edited text, preserving all
    other fields (working_directory, timeout, environment).

    Returns None if input is exhausted (EOF).

    Args:
        current: The current SSHCommand to edit.
        io: Terminal IO for reading/writing.

    Returns:
        New SSHCommand with updated command text, or None on EOF.
    """
    while True:
        try:
            new_text = io.read_editable(
                "  Edit command: ",
                current.command,
            )
        except EOFError:
            return None

        stripped = new_text.strip()
        if not stripped:
            # Empty input in read_editable should return prefill,
            # but whitespace-only is invalid for SSHCommand
            io.write("  Command cannot be empty. Try again.\n")
            continue

        try:
            return current.with_changes(command=stripped)
        except ValidationError as exc:
            io.write(f"  Invalid command: {exc.errors()[0]['msg']}\n")
            io.write("  Try again.\n")
            continue
