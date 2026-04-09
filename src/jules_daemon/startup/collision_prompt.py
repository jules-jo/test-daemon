"""Collision warning display and interactive prompt for daemon startup.

When the collision detector discovers existing daemon processes, this
module formats a human-readable warning and presents an interactive
prompt that lets the user choose how to proceed:

    [P]roceed       -- warn-and-allow: start the new daemon alongside
                       the existing one (collision detection is
                       informational, not blocking)
    [A]bort         -- stop the new daemon startup
    [F]orce-replace -- terminate the existing daemon process(es) and
                       then proceed with startup

The warning display includes:
- PID of each detected collision
- Command line (truncated to a readable length)
- Duration since the process started (human-readable)
- Source classification (Process Table, Wiki Session, or Both)
- Wiki run ID and status when available

EOF (Ctrl-D) and KeyboardInterrupt (Ctrl-C) are treated as abort
for safety.

The IO layer uses the same ``TerminalIO`` abstraction from the CLI
confirmation module, ensuring testability via ``FakeTerminalIO``.

Usage::

    from pathlib import Path
    from jules_daemon.startup.collision_detector import detect_collisions
    from jules_daemon.startup.collision_prompt import (
        CollisionAction,
        prompt_collision_action,
    )

    report = detect_collisions(Path("wiki"))
    if report.has_collision:
        result = prompt_collision_action(report)
        if result.action == CollisionAction.ABORT:
            sys.exit(1)
        elif result.action == CollisionAction.FORCE_REPLACE:
            kill_existing_daemons(report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from jules_daemon.cli.confirmation import DefaultTerminalIO, TerminalIO
from jules_daemon.startup.collision_detector import (
    CollisionEntry,
    CollisionReport,
    CollisionSource,
)

__all__ = [
    "CollisionAction",
    "CollisionPromptResult",
    "format_collision_warning",
    "format_duration",
    "prompt_collision_action",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEPARATOR = "-" * 60
_MAX_COMMAND_DISPLAY_LEN = 80

_SOURCE_LABELS: dict[CollisionSource, str] = {
    CollisionSource.PROCESS_TABLE: "Process Table",
    CollisionSource.WIKI_SESSION: "Wiki Session",
    CollisionSource.BOTH: "Both (Process Table + Wiki)",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CollisionAction(Enum):
    """User's chosen action when a daemon collision is detected.

    Values:
        PROCEED: Warn-and-allow -- continue startup alongside existing.
        ABORT: Stop the new daemon startup.
        FORCE_REPLACE: Terminate existing daemon(s) and proceed.
    """

    PROCEED = "proceed"
    ABORT = "abort"
    FORCE_REPLACE = "force_replace"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollisionPromptResult:
    """Immutable result of the collision prompt interaction.

    Attributes:
        action: The user's chosen action (proceed, abort, force-replace).
        report: The original collision report that was displayed.
    """

    action: CollisionAction
    report: CollisionReport


# ---------------------------------------------------------------------------
# Input classification
# ---------------------------------------------------------------------------

_PROCEED_INPUTS = frozenset({"p", "proceed", "yes", "y", "continue"})
_ABORT_INPUTS = frozenset({"a", "abort", "quit", "q", "exit"})
_FORCE_INPUTS = frozenset({"f", "force", "force-replace", "replace"})


def _classify_collision_input(raw: str) -> str:
    """Classify raw user input into an action keyword.

    Args:
        raw: Raw input string from the user.

    Returns:
        One of 'proceed', 'abort', 'force', or 'unknown'.
    """
    normalized = raw.strip().lower()
    if not normalized:
        return "unknown"
    if normalized in _PROCEED_INPUTS:
        return "proceed"
    if normalized in _ABORT_INPUTS:
        return "abort"
    if normalized in _FORCE_INPUTS:
        return "force"
    return "unknown"


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------


def format_duration(seconds: Optional[float]) -> str:
    """Format a duration in seconds to a human-readable string.

    Examples:
        format_duration(None)    -> "unknown"
        format_duration(0.0)     -> "0s"
        format_duration(45.0)    -> "45s"
        format_duration(125.0)   -> "2m 5s"
        format_duration(3725.0)  -> "1h 2m 5s"

    Args:
        seconds: Duration in seconds, or None if unknown.

    Returns:
        Human-readable duration string.
    """
    if seconds is None:
        return "unknown"

    total = int(seconds)
    if total < 60:
        return f"{total}s"

    hours = total // 3600
    remaining = total % 3600
    minutes = remaining // 60
    secs = remaining % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"

    return f"{minutes}m {secs}s"


# ---------------------------------------------------------------------------
# Warning formatting
# ---------------------------------------------------------------------------


def _format_pid(pid: int) -> str:
    """Format a PID for display, handling the pid=0 case.

    Args:
        pid: Process ID. 0 indicates no PID is available.

    Returns:
        PID as string, or "N/A (no PID)" for pid=0.
    """
    if pid == 0:
        return "N/A (no PID)"
    return str(pid)


def _format_command(command_line: str) -> str:
    """Format a command line for display, truncating if needed.

    Args:
        command_line: Full command line string.

    Returns:
        Command string, truncated with ellipsis if too long.
        Returns "(none)" for empty strings.
    """
    if not command_line:
        return "(none)"
    if len(command_line) <= _MAX_COMMAND_DISPLAY_LEN:
        return command_line
    return command_line[:_MAX_COMMAND_DISPLAY_LEN - 3] + "..."


def _format_entry(entry: CollisionEntry, index: int) -> list[str]:
    """Format a single collision entry as display lines.

    Args:
        entry: The collision entry to format.
        index: 1-based entry number for display.

    Returns:
        List of formatted lines (without trailing newlines).
    """
    source_label = _SOURCE_LABELS.get(entry.source, entry.source.value)
    lines: list[str] = []

    lines.append(f"  [{index}] PID: {_format_pid(entry.pid)}")
    lines.append(f"      Source:   {source_label}")

    if entry.command_line:
        lines.append(f"      Command:  {_format_command(entry.command_line)}")

    lines.append(f"      Duration: {format_duration(entry.duration_seconds)}")

    if entry.wiki_run_id is not None:
        lines.append(f"      Run ID:   {entry.wiki_run_id}")

    if entry.wiki_status is not None:
        lines.append(f"      Status:   {entry.wiki_status}")

    return lines


def format_collision_warning(report: CollisionReport) -> str:
    """Format a collision report as a warning message for terminal display.

    Produces a human-readable block showing each detected collision with
    its PID, command, duration, source, and wiki session details. Ends
    with the three action choices.

    Args:
        report: The collision report to display.

    Returns:
        Formatted multi-line warning string ready for terminal output.
    """
    count = len(report.entries)
    plural = "es" if count != 1 else ""

    lines: list[str] = []
    lines.append("")
    lines.append(_SEPARATOR)
    lines.append("  WARNING: Daemon Collision Detected")
    lines.append(_SEPARATOR)
    lines.append("")
    lines.append(
        f"  Found {count} existing daemon process{plural}:"
    )
    lines.append("")

    for idx, entry in enumerate(report.entries, start=1):
        lines.extend(_format_entry(entry, idx))
        if idx < count:
            lines.append("")  # blank line between entries

    lines.append("")
    lines.append(_SEPARATOR)
    lines.append("  [P]roceed  [A]bort  [F]orce-replace")
    lines.append(_SEPARATOR)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive prompt
# ---------------------------------------------------------------------------


def prompt_collision_action(
    report: CollisionReport,
    *,
    terminal: TerminalIO | None = None,
) -> CollisionPromptResult:
    """Display the collision warning and prompt the user for an action.

    Shows a formatted warning with all detected collisions, then loops
    until the user provides a valid choice:

        [P]roceed       -- warn-and-allow, continue startup
        [A]bort         -- stop the new daemon
        [F]orce-replace -- kill existing daemon(s), then proceed

    EOF (Ctrl-D) and KeyboardInterrupt are treated as abort for safety.

    Args:
        report: Collision report with detected entries.
        terminal: IO abstraction. Defaults to DefaultTerminalIO.

    Returns:
        CollisionPromptResult with the user's action and original report.
    """
    io = terminal or DefaultTerminalIO()

    # Display the warning
    warning = format_collision_warning(report)
    io.write(warning)

    # Prompt loop
    while True:
        try:
            raw = io.read_line("  Choice: ")
        except EOFError:
            logger.info(
                "EOF received during collision prompt -- treating as abort"
            )
            return CollisionPromptResult(
                action=CollisionAction.ABORT,
                report=report,
            )

        action = _classify_collision_input(raw)

        if action == "proceed":
            logger.info("User chose to proceed despite collision")
            return CollisionPromptResult(
                action=CollisionAction.PROCEED,
                report=report,
            )

        if action == "abort":
            logger.info("User chose to abort due to collision")
            return CollisionPromptResult(
                action=CollisionAction.ABORT,
                report=report,
            )

        if action == "force":
            logger.info("User chose to force-replace existing daemon(s)")
            return CollisionPromptResult(
                action=CollisionAction.FORCE_REPLACE,
                report=report,
            )

        # Invalid input -- reprompt
        io.write(
            "  Invalid input. Use [P]roceed, [A]bort, or [F]orce-replace.\n"
        )
