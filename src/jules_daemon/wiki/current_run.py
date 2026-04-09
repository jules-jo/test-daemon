"""Persist daemon current-run state to a wiki markdown file.

The current-run record is a single wiki file with YAML frontmatter containing
all structured state, and a markdown body with human-readable status and
recent output. This module provides the sole persistence API for daemon state.

Wiki file location: {wiki_root}/pages/daemon/current-run.md

State transitions:
  IDLE -> PENDING_APPROVAL -> RUNNING -> COMPLETED | FAILED | CANCELLED
  Any terminal state -> IDLE (via clear)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    ProcessIDs,
    Progress,
    RunStatus,
    SSHTarget,
)


_CURRENT_RUN_FILENAME = "current-run.md"
_DAEMON_DIR = "pages/daemon"
_WIKI_TAGS = ["daemon", "state", "current-run"]
_WIKI_TYPE = "daemon-state"


def _wiki_file_path(wiki_root: Path) -> Path:
    """Resolve the absolute path to the current-run wiki file."""
    return wiki_root / _DAEMON_DIR / _CURRENT_RUN_FILENAME


def _ensure_directory(path: Path) -> None:
    """Create parent directories if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


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


def _dict_to_pids(data: Optional[dict[str, Any]]) -> ProcessIDs:
    """Deserialize ProcessIDs from a plain dict."""
    if data is None:
        return ProcessIDs()
    return ProcessIDs(
        daemon=data.get("daemon"),
        remote=data.get("remote"),
    )


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


def _run_to_frontmatter(run: CurrentRun) -> dict[str, Any]:
    """Convert a CurrentRun to a YAML-serializable frontmatter dict."""
    return {
        "tags": list(_WIKI_TAGS),
        "type": _WIKI_TYPE,
        "created": _datetime_to_iso(run.created_at),
        "updated": _datetime_to_iso(run.updated_at),
        "status": run.status.value,
        "run_id": run.run_id,
        "ssh_target": _ssh_target_to_dict(run.ssh_target),
        "command": _command_to_dict(run.command),
        "pids": _pids_to_dict(run.pids),
        "progress": _progress_to_dict(run.progress),
        "started_at": _datetime_to_iso(run.started_at),
        "completed_at": _datetime_to_iso(run.completed_at),
        "error": run.error,
    }


def _frontmatter_to_run(fm: dict[str, Any]) -> CurrentRun:
    """Reconstruct a CurrentRun from a parsed frontmatter dict."""
    return CurrentRun(
        status=RunStatus(fm.get("status", "idle")),
        run_id=fm.get("run_id", ""),
        ssh_target=_dict_to_ssh_target(fm.get("ssh_target")),
        command=_dict_to_command(fm.get("command")),
        pids=_dict_to_pids(fm.get("pids")),
        progress=_dict_to_progress(fm.get("progress")),
        started_at=_iso_to_datetime(fm.get("started_at")),
        completed_at=_iso_to_datetime(fm.get("completed_at")),
        error=fm.get("error"),
        created_at=(_iso_to_datetime(fm["created"]) if fm.get("created") else None) or datetime.now(timezone.utc),
        updated_at=(_iso_to_datetime(fm["updated"]) if fm.get("updated") else None) or datetime.now(timezone.utc),
    )


def _build_body(run: CurrentRun) -> str:
    """Generate the human-readable markdown body for the current-run file."""
    lines = [
        "# Current Run",
        "",
        f"*Daemon state record -- status: {run.status.value}*",
        "",
    ]

    if run.status == RunStatus.IDLE:
        lines.append("No active run. The daemon is idle and ready for commands.")
        return "\n".join(lines)

    # SSH target section
    if run.ssh_target is not None:
        lines.extend([
            "## SSH Target",
            "",
            f"- **Host:** {run.ssh_target.host}",
            f"- **User:** {run.ssh_target.user}",
            f"- **Port:** {run.ssh_target.port}",
            "",
        ])

    # Command section
    if run.command is not None:
        lines.extend([
            "## Command",
            "",
            f"- **Request:** {run.command.natural_language}",
        ])
        if run.command.resolved_shell:
            lines.append(f"- **Shell:** `{run.command.resolved_shell}`")
        approval = "yes" if run.command.approved else "pending"
        lines.extend([
            f"- **Approved:** {approval}",
            "",
        ])

    # Progress section
    if run.status == RunStatus.RUNNING:
        prog = run.progress
        lines.extend([
            "## Progress",
            "",
            f"- **Percent:** {prog.percent:.1f}%",
            f"- **Passed:** {prog.tests_passed}",
            f"- **Failed:** {prog.tests_failed}",
            f"- **Skipped:** {prog.tests_skipped}",
            f"- **Total:** {prog.tests_total}",
        ])
        if prog.last_output_line:
            lines.extend([
                "",
                "### Last Output",
                "",
                "```",
                prog.last_output_line,
                "```",
            ])
        lines.append("")

    # Result section for terminal states
    if run.is_terminal:
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
    lines.append(f"- **Last Updated:** {_datetime_to_iso(run.updated_at)}")
    lines.append("")

    return "\n".join(lines)


# -- Public API --


def write(wiki_root: Path, run: CurrentRun) -> Path:
    """Write a new current-run record to the wiki file.

    Creates the file and parent directories if needed. Overwrites any
    existing content (each write is a complete snapshot).

    Args:
        wiki_root: Path to the wiki root directory.
        run: The current run state to persist.

    Returns:
        Path to the written wiki file.
    """
    file_path = _wiki_file_path(wiki_root)
    _ensure_directory(file_path)

    doc = WikiDocument(
        frontmatter=_run_to_frontmatter(run),
        body=_build_body(run),
    )
    content = frontmatter.serialize(doc)

    # Atomic write: write to temp file then rename
    tmp_path = file_path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(str(tmp_path), str(file_path))

    return file_path


def read(wiki_root: Path) -> Optional[CurrentRun]:
    """Read the current-run record from the wiki file.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        The deserialized CurrentRun, or None if the file does not exist.
    """
    file_path = _wiki_file_path(wiki_root)
    if not file_path.exists():
        return None

    raw = file_path.read_text(encoding="utf-8")
    doc = frontmatter.parse(raw)
    return _frontmatter_to_run(doc.frontmatter)


def update(wiki_root: Path, run: CurrentRun) -> Path:
    """Update the current-run record, preserving the created_at timestamp.

    This is semantically identical to write() but validates that a record
    already exists. Use this for state transitions on an active run.

    Args:
        wiki_root: Path to the wiki root directory.
        run: The updated run state.

    Returns:
        Path to the written wiki file.

    Raises:
        FileNotFoundError: If no current-run record exists yet.
    """
    file_path = _wiki_file_path(wiki_root)
    if not file_path.exists():
        raise FileNotFoundError(
            f"No current-run record at {file_path}. Use write() first."
        )
    return write(wiki_root, run)


def clear(wiki_root: Path) -> Path:
    """Reset the current-run record to idle state.

    Writes a clean idle record, clearing all run-specific fields.
    The file is kept (not deleted) so the wiki always has the record.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        Path to the written wiki file.
    """
    idle = CurrentRun(status=RunStatus.IDLE)
    return write(wiki_root, idle)


def exists(wiki_root: Path) -> bool:
    """Check whether a current-run wiki file exists.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        True if the file exists.
    """
    return _wiki_file_path(wiki_root).exists()


def file_path(wiki_root: Path) -> Path:
    """Return the expected path to the current-run wiki file.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        Path (may or may not exist yet).
    """
    return _wiki_file_path(wiki_root)
