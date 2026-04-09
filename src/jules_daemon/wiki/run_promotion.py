"""Promote current-run records to completed run history.

When a test run reaches a terminal state (completed, failed, or cancelled),
this module:
  1. Writes a timestamped history entry to wiki/pages/daemon/history/
  2. Resets the current-run record to idle

History entries are standalone wiki files (YAML frontmatter + markdown body)
that preserve the full run context: SSH target, command, progress, timestamps,
and error information.

Wiki layout:
  wiki/
    pages/
      daemon/
        current-run.md              # Active state (reset to idle)
        history/
          run-{run_id}.md           # One file per completed run
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jules_daemon.wiki import current_run, frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    ProcessIDs,
    Progress,
    RunStatus,
    SSHTarget,
)

logger = logging.getLogger(__name__)

_HISTORY_DIR = "pages/daemon/history"
_HISTORY_TAGS = ["daemon", "run-history"]
_HISTORY_TYPE = "run-history"


# -- Immutable result types --


@dataclass(frozen=True)
class PromotionResult:
    """Result of promoting a current-run to history.

    Carries the path to the new history file, the run_id, terminal status,
    and the timestamp when promotion occurred.
    """

    history_path: Path
    run_id: str
    final_status: RunStatus
    promoted_at: datetime


@dataclass(frozen=True)
class HistoryEntry:
    """Summary of a history entry for listing purposes.

    Provides quick access to status, run_id, and file path without
    deserializing the full run record.
    """

    run_id: str
    status: RunStatus
    completed_at: Optional[datetime]
    file_path: Path


# -- Internal serialization helpers --


def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO 8601 string, or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _iso_to_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 string to datetime, or None."""
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _ssh_target_to_dict(target: Optional[SSHTarget]) -> Optional[dict[str, Any]]:
    """Serialize SSHTarget to a plain dict for YAML."""
    if target is None:
        return None
    return {
        "host": target.host,
        "user": target.user,
        "port": target.port,
        "key_path": target.key_path,
    }


def _dict_to_ssh_target(data: Optional[dict[str, Any]]) -> Optional[SSHTarget]:
    """Deserialize SSHTarget from a plain dict."""
    if data is None:
        return None
    return SSHTarget(
        host=data["host"],
        user=data["user"],
        port=data.get("port", 22),
        key_path=data.get("key_path"),
    )


def _command_to_dict(cmd: Optional[Command]) -> Optional[dict[str, Any]]:
    """Serialize Command to a plain dict for YAML."""
    if cmd is None:
        return None
    return {
        "natural_language": cmd.natural_language,
        "resolved_shell": cmd.resolved_shell,
        "approved": cmd.approved,
        "approved_at": _datetime_to_iso(cmd.approved_at),
    }


def _dict_to_command(data: Optional[dict[str, Any]]) -> Optional[Command]:
    """Deserialize Command from a plain dict."""
    if data is None:
        return None
    return Command(
        natural_language=data["natural_language"],
        resolved_shell=data.get("resolved_shell", ""),
        approved=data.get("approved", False),
        approved_at=_iso_to_datetime(data.get("approved_at")),
    )


def _pids_to_dict(pids: ProcessIDs) -> dict[str, Optional[int]]:
    """Serialize ProcessIDs to a plain dict for YAML."""
    return {
        "daemon": pids.daemon,
        "remote": pids.remote,
    }


def _progress_to_dict(prog: Progress) -> dict[str, Any]:
    """Serialize Progress to a plain dict for YAML."""
    return {
        "percent": prog.percent,
        "tests_passed": prog.tests_passed,
        "tests_failed": prog.tests_failed,
        "tests_skipped": prog.tests_skipped,
        "tests_total": prog.tests_total,
        "last_output_line": prog.last_output_line,
        "checkpoint_at": _datetime_to_iso(prog.checkpoint_at),
    }


def _dict_to_progress(data: Optional[dict[str, Any]]) -> Progress:
    """Deserialize Progress from a plain dict."""
    if data is None:
        return Progress()
    return Progress(
        percent=float(data.get("percent", 0.0)),
        tests_passed=int(data.get("tests_passed", 0)),
        tests_failed=int(data.get("tests_failed", 0)),
        tests_skipped=int(data.get("tests_skipped", 0)),
        tests_total=int(data.get("tests_total", 0)),
        last_output_line=data.get("last_output_line", ""),
        checkpoint_at=_iso_to_datetime(data.get("checkpoint_at")),
    )


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# -- History file path --


def _history_dir(wiki_root: Path) -> Path:
    """Resolve the history directory path."""
    return wiki_root / _HISTORY_DIR


def _history_file_path(wiki_root: Path, run_id: str) -> Path:
    """Resolve the path for a specific history entry."""
    return _history_dir(wiki_root) / f"run-{run_id}.md"


def _ensure_directory(path: Path) -> None:
    """Create parent directories if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


# -- Frontmatter conversion --


def _run_to_history_frontmatter(
    run: CurrentRun,
    promoted_at: datetime,
) -> dict[str, Any]:
    """Convert a terminal CurrentRun to history frontmatter."""
    return {
        "tags": list(_HISTORY_TAGS),
        "type": _HISTORY_TYPE,
        "run_id": run.run_id,
        "status": run.status.value,
        "promoted_at": _datetime_to_iso(promoted_at),
        "created": _datetime_to_iso(run.created_at),
        "started_at": _datetime_to_iso(run.started_at),
        "completed_at": _datetime_to_iso(run.completed_at),
        "ssh_target": _ssh_target_to_dict(run.ssh_target),
        "command": _command_to_dict(run.command),
        "pids": _pids_to_dict(run.pids),
        "progress": _progress_to_dict(run.progress),
        "error": run.error,
    }


# -- Markdown body generation --


def _build_history_body(run: CurrentRun) -> str:
    """Generate the human-readable markdown body for a history entry."""
    lines = [
        "# Run History",
        "",
        f"*Completed run record -- status: {run.status.value}*",
        "",
    ]

    # SSH target
    if run.ssh_target is not None:
        lines.extend([
            "## SSH Target",
            "",
            f"- **Host:** {run.ssh_target.host}",
            f"- **User:** {run.ssh_target.user}",
            f"- **Port:** {run.ssh_target.port}",
            "",
        ])

    # Command
    if run.command is not None:
        lines.extend([
            "## Command",
            "",
            f"- **Request:** {run.command.natural_language}",
        ])
        if run.command.resolved_shell:
            lines.append(f"- **Shell:** `{run.command.resolved_shell}`")
        lines.append("")

    # Result
    lines.extend([
        f"## Result: {run.status.value}",
        "",
    ])
    if run.error:
        lines.extend([
            "### Error",
            "",
            "```",
            run.error,
            "```",
            "",
        ])
    prog = run.progress
    lines.extend([
        f"- **Passed:** {prog.tests_passed}",
        f"- **Failed:** {prog.tests_failed}",
        f"- **Skipped:** {prog.tests_skipped}",
        f"- **Total:** {prog.tests_total}",
        f"- **Percent:** {prog.percent:.1f}%",
        "",
    ])

    # Timestamps
    lines.extend([
        "## Timestamps",
        "",
        f"- **Run ID:** {run.run_id}",
    ])
    if run.started_at:
        lines.append(f"- **Started:** {_datetime_to_iso(run.started_at)}")
    if run.completed_at:
        lines.append(f"- **Completed:** {_datetime_to_iso(run.completed_at)}")
    lines.append("")

    return "\n".join(lines)


# -- History deserialization --


def _frontmatter_to_run(fm: dict[str, Any]) -> CurrentRun:
    """Reconstruct a CurrentRun from history entry frontmatter."""
    return CurrentRun(
        status=RunStatus(fm.get("status", "completed")),
        run_id=fm.get("run_id", ""),
        ssh_target=_dict_to_ssh_target(fm.get("ssh_target")),
        command=_dict_to_command(fm.get("command")),
        pids=ProcessIDs(
            daemon=fm.get("pids", {}).get("daemon"),
            remote=fm.get("pids", {}).get("remote"),
        ),
        progress=_dict_to_progress(fm.get("progress")),
        started_at=_iso_to_datetime(fm.get("started_at")),
        completed_at=_iso_to_datetime(fm.get("completed_at")),
        error=fm.get("error"),
        created_at=(
            _iso_to_datetime(fm["created"])
            if fm.get("created")
            else None
        ) or _now_utc(),
        updated_at=_now_utc(),
    )


# -- Public API --


def promote_run(wiki_root: Path, run: CurrentRun) -> PromotionResult:
    """Promote a terminal current-run record to completed history.

    This function:
      1. Validates the run is in a terminal state
      2. Writes a history entry to wiki/pages/daemon/history/
      3. Resets the current-run record to idle

    The history write uses atomic write (tmp file + rename) to prevent
    partial files. The current-run reset only happens after the history
    file is successfully written.

    Args:
        wiki_root: Path to the wiki root directory.
        run: The terminal CurrentRun to promote.

    Returns:
        PromotionResult with the history file path and metadata.

    Raises:
        ValueError: If the run is not in a terminal state.
    """
    if not run.is_terminal:
        raise ValueError(
            f"Cannot promote a run in state '{run.status.value}' -- "
            f"only terminal states (completed, failed, cancelled) can be promoted"
        )

    promoted_at = _now_utc()

    # Build the history wiki document
    history_fm = _run_to_history_frontmatter(run, promoted_at)
    history_body = _build_history_body(run)
    doc = WikiDocument(frontmatter=history_fm, body=history_body)
    content = frontmatter.serialize(doc)

    # Write history file atomically
    history_path = _history_file_path(wiki_root, run.run_id)
    _ensure_directory(history_path)

    tmp_path = history_path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(str(tmp_path), str(history_path))

    logger.info(
        "Promoted run %s (status=%s) to history: %s",
        run.run_id,
        run.status.value,
        history_path,
    )

    # Reset current-run to idle only after history is safely written
    current_run.clear(wiki_root)

    logger.info("Current-run record reset to idle after promotion")

    return PromotionResult(
        history_path=history_path,
        run_id=run.run_id,
        final_status=run.status,
        promoted_at=promoted_at,
    )


def list_history(
    wiki_root: Path,
    limit: int = 50,
) -> list[HistoryEntry]:
    """List completed run history entries, newest first.

    Scans wiki/pages/daemon/history/ for history wiki files and returns
    summary entries sorted by completed_at (newest first).

    Args:
        wiki_root: Path to the wiki root directory.
        limit: Maximum number of entries to return (default 50).

    Returns:
        List of HistoryEntry summaries, sorted newest first.
    """
    hist_dir = _history_dir(wiki_root)
    if not hist_dir.exists():
        return []

    entries: list[HistoryEntry] = []
    for md_file in sorted(hist_dir.glob("run-*.md")):
        try:
            raw = md_file.read_text(encoding="utf-8")
            doc = frontmatter.parse(raw)
            fm = doc.frontmatter

            completed_at = _iso_to_datetime(fm.get("completed_at"))
            entry = HistoryEntry(
                run_id=fm.get("run_id", ""),
                status=RunStatus(fm.get("status", "completed")),
                completed_at=completed_at,
                file_path=md_file,
            )
            entries.append(entry)
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Skipping malformed history file %s: %s", md_file, exc
            )
            continue

    # Sort newest first by completed_at (None sorts last)
    entries.sort(
        key=lambda e: e.completed_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    return entries[:limit]


def read_history_entry(file_path: Path) -> Optional[CurrentRun]:
    """Read a single history entry and return the full run record.

    Args:
        file_path: Path to the history wiki file.

    Returns:
        The deserialized CurrentRun, or None if the file does not exist.
    """
    if not file_path.exists():
        return None

    raw = file_path.read_text(encoding="utf-8")
    doc = frontmatter.parse(raw)
    return _frontmatter_to_run(doc.frontmatter)
