"""Wiki persistence for recovery outcome logging.

Writes and reads recovery attempt records to the wiki in Karpathy-style
format (YAML frontmatter + markdown body). Each recovery attempt produces
a complete snapshot in {wiki_root}/pages/daemon/recovery-log.md.

This module also provides the utility to update the current-run wiki
record to FAILED status during recovery failure handling.

Usage:
    from jules_daemon.wiki.recovery_log import (
        write_recovery_log,
        update_wiki_run_to_failed,
    )

    ok = write_recovery_log(wiki_root, outcome)
    ok = update_wiki_run_to_failed(wiki_root, "Timeout after 30s")
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jules_daemon.wiki import current_run, frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument

if TYPE_CHECKING:
    from jules_daemon.wiki.recovery_orchestrator import RecoveryOutcome

__all__ = [
    "update_wiki_run_to_failed",
    "write_recovery_log",
]

logger = logging.getLogger(__name__)

_RECOVERY_LOG_FILENAME = "recovery-log.md"
_DAEMON_DIR = "pages/daemon"
_WIKI_TAGS = ["daemon", "recovery", "audit"]
_WIKI_TYPE = "daemon-recovery-log"


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _build_recovery_log_body(outcome: RecoveryOutcome) -> str:
    """Generate human-readable markdown body for the recovery log."""
    status_label = "Success" if outcome.success else "Failed"
    if outcome.timed_out:
        status_label = "Timed Out"

    lines = [
        "# Recovery Log",
        "",
        f"*Recovery attempt -- result: {status_label}*",
        "",
        "## Summary",
        "",
        f"- **Action:** {outcome.action_taken.value}",
        f"- **Run ID:** {outcome.run_id}",
        f"- **Result:** {status_label}",
        f"- **Duration:** {outcome.total_duration_seconds:.3f}s",
        f"- **Deadline:** {outcome.deadline_seconds:.1f}s",
        f"- **Time Remaining:** {outcome.time_remaining_seconds:.3f}s",
        f"- **Timed Out:** {'yes' if outcome.timed_out else 'no'}",
        "",
    ]

    if outcome.error:
        lines.extend([
            "## Error",
            "",
            "```",
            outcome.error,
            "```",
            "",
        ])

    if outcome.phases:
        lines.extend([
            "## Phase Timings",
            "",
            "| Phase | Success | Duration (s) | Error |",
            "|-------|---------|-------------|-------|",
        ])
        for phase in outcome.phases:
            error_text = phase.error or "-"
            success_text = "yes" if phase.success else "no"
            lines.append(
                f"| {phase.phase.value} | {success_text} | "
                f"{phase.duration_seconds:.3f} | {error_text} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_recovery_log(
    wiki_root: Path,
    outcome: RecoveryOutcome,
) -> bool:
    """Write the recovery outcome to the wiki recovery log.

    Creates or replaces the recovery-log.md file in the daemon wiki
    directory. Each recovery attempt produces a complete snapshot with
    YAML frontmatter and human-readable markdown body.

    Args:
        wiki_root: Path to the wiki root directory.
        outcome: The recovery outcome to record.

    Returns:
        True if the write succeeded, False on any error.
    """
    file_path = wiki_root / _DAEMON_DIR / _RECOVERY_LOG_FILENAME

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fm: dict[str, Any] = {
            "tags": list(_WIKI_TAGS),
            "type": _WIKI_TYPE,
            "created": _now_utc().isoformat(),
            "updated": _now_utc().isoformat(),
            "success": outcome.success,
            "action": outcome.action_taken.value,
            "run_id": outcome.run_id,
            "total_duration_seconds": round(outcome.total_duration_seconds, 3),
            "deadline_seconds": outcome.deadline_seconds,
            "timed_out": outcome.timed_out,
            "error": outcome.error,
            "phases": [
                {
                    "phase": p.phase.value,
                    "success": p.success,
                    "duration_seconds": round(p.duration_seconds, 3),
                    "error": p.error,
                }
                for p in outcome.phases
            ],
        }

        body = _build_recovery_log_body(outcome)
        doc = WikiDocument(frontmatter=fm, body=body)
        content = frontmatter.serialize(doc)

        # Atomic write
        tmp_path = file_path.with_suffix(".md.tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(str(tmp_path), str(file_path))

        logger.info(
            "Recovery log written to %s: success=%s timed_out=%s",
            file_path,
            outcome.success,
            outcome.timed_out,
        )
        return True

    except Exception as exc:
        logger.warning(
            "Failed to write recovery log to %s: %s",
            file_path,
            exc,
        )
        return False


def update_wiki_run_to_failed(
    wiki_root: Path,
    error_message: str,
) -> bool:
    """Update the current-run wiki record to FAILED status.

    Reads the current run, transitions it to FAILED with the given
    error message, and writes it back to the wiki.

    Args:
        wiki_root: Path to the wiki root directory.
        error_message: Error description to record.

    Returns:
        True if the update succeeded, False on any error.
    """
    try:
        run = current_run.read(wiki_root)
        if run is None:
            logger.warning("Cannot update wiki to FAILED: no current run file")
            return False

        failed_run = run.with_failed(error_message, run.progress)
        current_run.write(wiki_root, failed_run)

        logger.info(
            "Updated current-run wiki to FAILED: %s", error_message
        )
        return True

    except Exception as exc:
        logger.warning(
            "Failed to update current-run wiki to FAILED: %s", exc
        )
        return False
