"""Session state persistence to wiki on unexpected disconnect.

Captures a full SessionSnapshot of the daemon's current state when a CLI
client disconnects unexpectedly, writes it to a wiki markdown file with
YAML frontmatter, and provides loading and discarding operations for
reconnection handling.

Wiki file location: {wiki_root}/pages/daemon/session-state.md

The session state file is distinct from current-run.md. While current-run
tracks the active test execution lifecycle, session-state captures the
CLI-to-daemon relationship: which client was connected, why it disconnected,
and what the daemon was doing at the time. This separation allows the daemon
to continue monitoring autonomously (current-run stays active) while
recording the disconnect context for later reconnection offers.

Flow:
  1. Client disconnects unexpectedly (EOF, broken pipe, timeout, etc.)
  2. Daemon builds SessionSnapshot from current state
  3. save_session_state() writes snapshot to wiki (atomic: temp + rename)
  4. On reconnection, load_session_state() reads the snapshot
  5. Recovery logic decides whether to offer resume or discard
  6. discard_session_state() clears the file to a non-resumable marker

Usage:
    from pathlib import Path
    from jules_daemon.wiki.session_persistence import (
        SessionSnapshot,
        save_session_state,
        load_session_state,
        discard_session_state,
    )

    # On disconnect:
    snap = SessionSnapshot.from_current_run(run, "eof", "jules-cli", 999)
    save_session_state(wiki_root, snap)

    # On reconnect:
    loaded = load_session_state(wiki_root)
    if loaded.snapshot and loaded.snapshot.is_resumable:
        # Offer to resume
        ...
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

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

__all__ = [
    "LoadSessionOutcome",
    "SessionLoadResult",
    "SessionSnapshot",
    "SessionWriteResult",
    "discard_session_state",
    "load_session_state",
    "save_session_state",
    "session_file_path",
]

logger = logging.getLogger(__name__)

_SESSION_FILENAME = "session-state.md"
_DAEMON_DIR = "pages/daemon"
_WIKI_TAGS = ("daemon", "session", "state")
_WIKI_TYPE = "daemon-session-state"

# Statuses that indicate a session can be resumed
_RESUMABLE_STATUSES = frozenset({RunStatus.RUNNING, RunStatus.PENDING_APPROVAL})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


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


def _safe_int(value: Any) -> Optional[int]:
    """Coerce a value to int, or return None if not possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# SessionSnapshot model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionSnapshot:
    """Immutable snapshot of session state at the moment of disconnect.

    Captures the full context needed to offer session resumption:
    - Run lifecycle state (status, run_id, progress)
    - SSH connection details (target host/user/port)
    - Command details (natural language + resolved shell)
    - Process identifiers (daemon PID, remote PID)
    - Disconnect context (reason, client identity)

    This is a pure data transfer object. State transitions produce new
    instances via the class method from_current_run().
    """

    run_id: str
    status: RunStatus
    ssh_target: Optional[SSHTarget]
    command: Optional[Command]
    pids: ProcessIDs
    progress: Progress
    started_at: Optional[datetime]
    error: Optional[str]
    disconnect_reason: str
    client_name: str
    client_pid: Optional[int]
    saved_at: datetime = field(default_factory=_now_utc)

    @property
    def is_resumable(self) -> bool:
        """True if the session was in an active state when disconnected."""
        return self.status in _RESUMABLE_STATUSES

    @classmethod
    def from_current_run(
        cls,
        *,
        run: CurrentRun,
        disconnect_reason: str,
        client_name: str,
        client_pid: Optional[int],
    ) -> SessionSnapshot:
        """Build a snapshot from the daemon's current run state.

        Args:
            run: The current run state at the moment of disconnect.
            disconnect_reason: Classification of the disconnect type
                (e.g., "eof", "broken_pipe", "connection_reset").
            client_name: Identifier for the disconnected CLI client.
            client_pid: Process ID of the disconnected client (None if unknown).

        Returns:
            A new frozen SessionSnapshot capturing the full context.
        """
        return cls(
            run_id=run.run_id,
            status=run.status,
            ssh_target=run.ssh_target,
            command=run.command,
            pids=run.pids,
            progress=run.progress,
            started_at=run.started_at,
            error=run.error,
            disconnect_reason=disconnect_reason,
            client_name=client_name,
            client_pid=client_pid,
            saved_at=_now_utc(),
        )


# ---------------------------------------------------------------------------
# Write result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionWriteResult:
    """Outcome of writing session state to the wiki."""

    success: bool
    file_path: Optional[Path]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Load result model
# ---------------------------------------------------------------------------


class LoadSessionOutcome(Enum):
    """How the session state file load resolved."""

    LOADED = "loaded"
    NO_FILE = "no_file"
    CORRUPTED = "corrupted"


@dataclass(frozen=True)
class SessionLoadResult:
    """Outcome of loading session state from the wiki."""

    outcome: LoadSessionOutcome
    snapshot: Optional[SessionSnapshot]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Serialization: snapshot -> frontmatter dict
# ---------------------------------------------------------------------------


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


def _snapshot_to_frontmatter(snap: SessionSnapshot) -> dict[str, Any]:
    """Convert a SessionSnapshot to a YAML-serializable frontmatter dict."""
    return {
        "tags": list(_WIKI_TAGS),
        "type": _WIKI_TYPE,
        "saved_at": _datetime_to_iso(snap.saved_at),
        "run_id": snap.run_id,
        "status": snap.status.value,
        "ssh_target": _ssh_target_to_dict(snap.ssh_target),
        "command": _command_to_dict(snap.command),
        "pids": _pids_to_dict(snap.pids),
        "progress": _progress_to_dict(snap.progress),
        "started_at": _datetime_to_iso(snap.started_at),
        "error": snap.error,
        "disconnect_reason": snap.disconnect_reason,
        "client_name": snap.client_name,
        "client_pid": snap.client_pid,
    }


# ---------------------------------------------------------------------------
# Deserialization: frontmatter dict -> snapshot
# ---------------------------------------------------------------------------


def _dict_to_ssh_target(data: Optional[dict[str, Any]]) -> Optional[SSHTarget]:
    """Deserialize SSHTarget from a plain dict.

    Uses .get() with defaults so corrupted YAML produces a clear
    ValueError from SSHTarget.__post_init__ rather than a KeyError.
    """
    if data is None:
        return None
    host = data.get("host") or ""
    user = data.get("user") or ""
    return SSHTarget(
        host=host,
        user=user,
        port=data.get("port", 22),
        key_path=data.get("key_path"),
    )


def _dict_to_command(data: Optional[dict[str, Any]]) -> Optional[Command]:
    """Deserialize Command from a plain dict.

    Uses .get() with defaults so corrupted YAML produces a clear
    ValueError from Command.__post_init__ rather than a KeyError.
    """
    if data is None:
        return None
    natural_language = data.get("natural_language") or ""
    return Command(
        natural_language=natural_language,
        resolved_shell=data.get("resolved_shell", ""),
        approved=data.get("approved", False),
        approved_at=_iso_to_datetime(data.get("approved_at")),
    )


def _dict_to_pids(data: Optional[dict[str, Any]]) -> ProcessIDs:
    """Deserialize ProcessIDs from a plain dict."""
    if data is None:
        return ProcessIDs()
    return ProcessIDs(
        daemon=_safe_int(data.get("daemon")),
        remote=_safe_int(data.get("remote")),
    )


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


def _frontmatter_to_snapshot(fm: dict[str, Any]) -> SessionSnapshot:
    """Reconstruct a SessionSnapshot from a parsed frontmatter dict."""
    saved_at_raw = fm.get("saved_at")
    saved_at = _iso_to_datetime(saved_at_raw) or _now_utc()

    return SessionSnapshot(
        run_id=fm.get("run_id", ""),
        status=RunStatus(fm.get("status", "idle")),
        ssh_target=_dict_to_ssh_target(fm.get("ssh_target")),
        command=_dict_to_command(fm.get("command")),
        pids=_dict_to_pids(fm.get("pids")),
        progress=_dict_to_progress(fm.get("progress")),
        started_at=_iso_to_datetime(fm.get("started_at")),
        error=fm.get("error"),
        disconnect_reason=fm.get("disconnect_reason", "unknown"),
        client_name=fm.get("client_name", "unknown"),
        client_pid=_safe_int(fm.get("client_pid")),
        saved_at=saved_at,
    )


# ---------------------------------------------------------------------------
# Markdown body builder
# ---------------------------------------------------------------------------


def _build_body(snap: SessionSnapshot) -> str:
    """Generate human-readable markdown body for the session state file."""
    lines = [
        "# Session State",
        "",
        f"*Session snapshot -- status: {snap.status.value}*",
        "",
    ]

    if not snap.is_resumable:
        lines.append("No resumable session. The daemon is idle or the run has completed.")
        return "\n".join(lines)

    lines.extend([
        "## Disconnect Context",
        "",
        f"- **Reason:** {snap.disconnect_reason}",
        f"- **Client:** {snap.client_name}",
    ])
    if snap.client_pid is not None:
        lines.append(f"- **Client PID:** {snap.client_pid}")
    lines.extend([
        f"- **Saved At:** {_datetime_to_iso(snap.saved_at)}",
        "",
    ])

    if snap.ssh_target is not None:
        lines.extend([
            "## SSH Target",
            "",
            f"- **Host:** {snap.ssh_target.host}",
            f"- **User:** {snap.ssh_target.user}",
            f"- **Port:** {snap.ssh_target.port}",
            "",
        ])

    if snap.command is not None:
        lines.extend([
            "## Command",
            "",
            f"- **Request:** {snap.command.natural_language}",
        ])
        if snap.command.resolved_shell:
            lines.append(f"- **Shell:** `{snap.command.resolved_shell}`")
        approval = "yes" if snap.command.approved else "pending"
        lines.extend([
            f"- **Approved:** {approval}",
            "",
        ])

    if snap.status == RunStatus.RUNNING:
        prog = snap.progress
        lines.extend([
            "## Progress at Disconnect",
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

    lines.extend([
        "## Run Metadata",
        "",
        f"- **Run ID:** {snap.run_id}",
        f"- **Status:** {snap.status.value}",
    ])
    if snap.started_at:
        lines.append(f"- **Started:** {_datetime_to_iso(snap.started_at)}")
    if snap.error:
        lines.extend([
            "",
            "### Error",
            "",
            "```",
            snap.error,
            "```",
        ])
    lines.append("")

    return "\n".join(lines)


def _build_discarded_body() -> str:
    """Generate markdown body for a discarded session marker."""
    return "\n".join([
        "# Session State",
        "",
        "*Session discarded by user or cleared by daemon.*",
        "",
        "No resumable session.",
        "",
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def session_file_path(wiki_root: Path) -> Path:
    """Return the path to the session state wiki file.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        Absolute path to the session-state.md file.
    """
    return wiki_root / _DAEMON_DIR / _SESSION_FILENAME


def save_session_state(
    wiki_root: Path,
    snapshot: SessionSnapshot,
) -> SessionWriteResult:
    """Write a session state snapshot to the wiki file.

    Creates the file and parent directories if needed. Each write is a
    complete snapshot that overwrites any prior content. Uses atomic
    write (temp file + os.replace) to prevent partial writes.

    Args:
        wiki_root: Path to the wiki root directory.
        snapshot: The session state to persist.

    Returns:
        SessionWriteResult indicating success or failure.
    """
    file_path = session_file_path(wiki_root)

    tmp_path = file_path.with_suffix(".md.tmp")

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        doc = WikiDocument(
            frontmatter=_snapshot_to_frontmatter(snapshot),
            body=_build_body(snapshot),
        )
        content = frontmatter.serialize(doc)

        # Atomic write: write to temp file then rename
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, file_path)

        logger.info(
            "Session state saved to %s: run_id=%s status=%s reason=%s",
            file_path,
            snapshot.run_id,
            snapshot.status.value,
            snapshot.disconnect_reason,
        )

        return SessionWriteResult(
            success=True,
            file_path=file_path,
            error=None,
        )

    except Exception as exc:
        # Clean up temp file on failure to prevent stale leftovers
        tmp_path.unlink(missing_ok=True)
        logger.warning(
            "Failed to save session state to %s: %s",
            file_path,
            exc,
        )
        return SessionWriteResult(
            success=False,
            file_path=file_path,
            error=str(exc),
        )


def load_session_state(wiki_root: Path) -> SessionLoadResult:
    """Load the session state snapshot from the wiki file.

    Reads the session-state.md file, parses its YAML frontmatter, and
    reconstructs a SessionSnapshot. Handles missing and corrupted files
    gracefully.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        SessionLoadResult with the loaded snapshot or error details.
        Never raises.
    """
    file_path = session_file_path(wiki_root)

    # Case 1: No file exists
    if not file_path.exists():
        logger.debug("No session state file at %s", file_path)
        return SessionLoadResult(
            outcome=LoadSessionOutcome.NO_FILE,
            snapshot=None,
            error=None,
        )

    # Case 2: File exists -- try to parse
    try:
        raw = file_path.read_text(encoding="utf-8")
        if not raw.strip():
            return SessionLoadResult(
                outcome=LoadSessionOutcome.CORRUPTED,
                snapshot=None,
                error="Session state file is empty",
            )

        doc = frontmatter.parse(raw)
        snapshot = _frontmatter_to_snapshot(doc.frontmatter)

        logger.info(
            "Loaded session state from %s: run_id=%s status=%s resumable=%s",
            file_path,
            snapshot.run_id,
            snapshot.status.value,
            snapshot.is_resumable,
        )

        return SessionLoadResult(
            outcome=LoadSessionOutcome.LOADED,
            snapshot=snapshot,
            error=None,
        )

    except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        logger.warning(
            "Corrupted session state file at %s: %s",
            file_path,
            exc,
        )
        return SessionLoadResult(
            outcome=LoadSessionOutcome.CORRUPTED,
            snapshot=None,
            error=str(exc),
        )
    except Exception as exc:
        logger.warning(
            "Unexpected error reading session state from %s: %s",
            file_path,
            exc,
        )
        return SessionLoadResult(
            outcome=LoadSessionOutcome.CORRUPTED,
            snapshot=None,
            error=str(exc),
        )


def discard_session_state(wiki_root: Path) -> bool:
    """Clear the session state file, marking it as non-resumable.

    Writes a minimal idle/discarded marker to the file rather than
    deleting it. This ensures the wiki always has the session file for
    audit purposes while preventing accidental resumption.

    If no session file exists, this is a no-op (returns True).

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        True if the discard succeeded (or file did not exist).
        False on write error.
    """
    file_path = session_file_path(wiki_root)

    if not file_path.exists():
        logger.debug("No session state file to discard at %s", file_path)
        return True

    tmp_path = file_path.with_suffix(".md.tmp")

    try:
        discarded_snap = SessionSnapshot(
            run_id="",
            status=RunStatus.IDLE,
            ssh_target=None,
            command=None,
            pids=ProcessIDs(),
            progress=Progress(),
            started_at=None,
            error=None,
            disconnect_reason="discarded",
            client_name="daemon",
            client_pid=None,
            saved_at=_now_utc(),
        )

        doc = WikiDocument(
            frontmatter=_snapshot_to_frontmatter(discarded_snap),
            body=_build_discarded_body(),
        )
        content = frontmatter.serialize(doc)

        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, file_path)

        logger.info("Session state discarded at %s", file_path)
        return True

    except Exception as exc:
        # Clean up temp file on failure
        tmp_path.unlink(missing_ok=True)
        logger.warning(
            "Failed to discard session state at %s: %s",
            file_path,
            exc,
        )
        return False
