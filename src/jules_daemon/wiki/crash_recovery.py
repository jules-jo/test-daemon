"""Crash recovery detection at daemon startup.

Orchestrates the full startup recovery flow by combining:
1. Wiki file reading (current_run module)
2. Interrupted-run detection (interrupted_run module)
3. Connection and metadata extraction (state_reader patterns)

into a single CrashRecoveryResult that tells the daemon exactly what to do.

The daemon calls detect_crash_recovery(wiki_root) at startup and receives
a flat, immutable result with:
- Which action to take (FRESH_START, RECONNECT, or RESUME_APPROVAL)
- All connection details (host, user, port, key_path)
- Process IDs (remote_pid, daemon_pid)
- Run metadata (run_id, status, resolved_shell, natural_language_command)
- Progress snapshot (progress_percent)

This module is the single entry point for crash recovery. It replaces the
need to call boot_reader, interrupted_run, and state_reader separately.

Usage:
    from pathlib import Path
    from jules_daemon.wiki.crash_recovery import detect_crash_recovery

    result = detect_crash_recovery(Path("wiki"))
    if result.needs_recovery:
        if result.action == RecoveryAction.RECONNECT:
            # SSH to result.host as result.user, check result.remote_pid
            ...
        elif result.action == RecoveryAction.RESUME_APPROVAL:
            # Re-prompt user for approval of the pending command
            ...
    else:
        # Start fresh -- daemon is idle and ready for commands
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import yaml

from jules_daemon.wiki import current_run
from jules_daemon.wiki.models import CurrentRun, RunStatus

__all__ = [
    "CrashRecoveryResult",
    "RecoveryAction",
    "detect_crash_recovery",
]

logger = logging.getLogger(__name__)


# -- Recovery actions --


class RecoveryAction(Enum):
    """Action the daemon should take based on crash recovery detection.

    FRESH_START: No interrupted run found. Daemon starts idle.
    RECONNECT: A RUNNING test was interrupted. Daemon should SSH back in
        and reattach to the remote process.
    RESUME_APPROVAL: A PENDING_APPROVAL command was interrupted. Daemon
        should re-prompt the user for approval.
    """

    FRESH_START = "fresh_start"
    RECONNECT = "reconnect"
    RESUME_APPROVAL = "resume_approval"


# -- Statuses that map to recovery actions --


_RECONNECT_STATUSES = frozenset({RunStatus.RUNNING})
_RESUME_STATUSES = frozenset({RunStatus.PENDING_APPROVAL})
_TERMINAL_STATUSES = frozenset({
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
})


# -- Result model --


@dataclass(frozen=True)
class CrashRecoveryResult:
    """Immutable result of crash recovery detection at daemon startup.

    Contains everything the daemon needs to decide its startup path and
    (if recovering) reconnect to the interrupted run.

    Attributes:
        action: Which recovery action to take.
        reason: Human-readable explanation for the decision.
        run_id: Unique identifier for the interrupted run (empty if none).
        status: The run's lifecycle status when the daemon stopped.
        host: SSH target hostname (None if no connection info).
        user: SSH target username (None if no connection info).
        port: SSH target port (None if no connection info).
        key_path: Path to SSH private key (None if password auth or missing).
        remote_pid: PID of the remote test process (None if not started).
        daemon_pid: PID of the previous daemon instance (None if unknown).
        resolved_shell: The approved shell command (None if not yet resolved).
        natural_language_command: The original user request (None if absent).
        progress_percent: Last known completion percentage (0.0 to 100.0).
        error: Error description from file parsing or prior failure.
        source_path: Path to the wiki file that was read (None if no file).
    """

    action: RecoveryAction
    reason: str
    run_id: str
    status: RunStatus
    host: str | None
    user: str | None
    port: int | None
    key_path: str | None
    remote_pid: int | None
    daemon_pid: int | None
    resolved_shell: str | None
    natural_language_command: str | None
    progress_percent: float
    error: str | None
    source_path: Path | None

    @property
    def needs_recovery(self) -> bool:
        """True if the daemon must recover an interrupted run."""
        return self.action in (RecoveryAction.RECONNECT, RecoveryAction.RESUME_APPROVAL)

    @property
    def has_connection(self) -> bool:
        """True if SSH connection parameters are available."""
        return self.host is not None and self.user is not None


# -- Internal builders --


def _determine_action(status: RunStatus) -> RecoveryAction:
    """Map a run status to the appropriate recovery action.

    Explicitly handles all known status categories:
    - RUNNING -> RECONNECT
    - PENDING_APPROVAL -> RESUME_APPROVAL
    - COMPLETED/FAILED/CANCELLED -> FRESH_START
    - IDLE and any future statuses -> FRESH_START
    """
    if status in _RECONNECT_STATUSES:
        return RecoveryAction.RECONNECT
    if status in _RESUME_STATUSES:
        return RecoveryAction.RESUME_APPROVAL
    if status in _TERMINAL_STATUSES:
        return RecoveryAction.FRESH_START
    # IDLE and any future statuses default to fresh start
    return RecoveryAction.FRESH_START


def _build_from_run(
    run: CurrentRun,
    source_path: Path,
) -> CrashRecoveryResult:
    """Build a CrashRecoveryResult from a successfully parsed CurrentRun."""
    action = _determine_action(run.status)

    # Extract SSH connection fields (None if no target set)
    host: str | None = None
    user: str | None = None
    port: int | None = None
    key_path: str | None = None
    if run.ssh_target is not None:
        host = run.ssh_target.host
        user = run.ssh_target.user
        port = run.ssh_target.port
        key_path = run.ssh_target.key_path

    # Extract command fields (None if no command set)
    resolved_shell: str | None = None
    natural_language_command: str | None = None
    if run.command is not None:
        resolved_shell = run.command.resolved_shell or None
        natural_language_command = run.command.natural_language

    # Build human-readable reason
    status_label = run.status.value
    if action == RecoveryAction.RECONNECT:
        reason = (
            f"Interrupted run detected: status was {status_label}, "
            f"run_id={run.run_id}, host={host}, remote_pid={run.pids.remote}"
        )
    elif action == RecoveryAction.RESUME_APPROVAL:
        reason = (
            f"Interrupted approval detected: status was {status_label}, "
            f"run_id={run.run_id}, host={host}"
        )
    else:
        reason = f"Prior run was {status_label} -- no recovery needed"

    return CrashRecoveryResult(
        action=action,
        reason=reason,
        run_id=run.run_id,
        status=run.status,
        host=host,
        user=user,
        port=port,
        key_path=key_path,
        remote_pid=run.pids.remote,
        daemon_pid=run.pids.daemon,
        resolved_shell=resolved_shell,
        natural_language_command=natural_language_command,
        progress_percent=run.progress.percent,
        error=run.error,
        source_path=source_path,
    )


def _build_fresh_start(
    reason: str,
    error: str | None = None,
    source_path: Path | None = None,
) -> CrashRecoveryResult:
    """Build a safe fresh-start result for missing/corrupted/idle cases."""
    return CrashRecoveryResult(
        action=RecoveryAction.FRESH_START,
        reason=reason,
        run_id="",
        status=RunStatus.IDLE,
        host=None,
        user=None,
        port=None,
        key_path=None,
        remote_pid=None,
        daemon_pid=None,
        resolved_shell=None,
        natural_language_command=None,
        progress_percent=0.0,
        error=error,
        source_path=source_path,
    )


# -- Public API --


def detect_crash_recovery(wiki_root: Path) -> CrashRecoveryResult:
    """Detect if crash recovery is needed by reading the wiki state file.

    This is the single entry point for daemon startup crash recovery
    detection. It reads the current-run wiki record, determines if an
    incomplete run exists, and returns a flat CrashRecoveryResult with
    all connection details and run metadata extracted.

    Decision logic:
    1. No wiki file: FRESH_START (first ever daemon run)
    2. Corrupted wiki file: FRESH_START with error detail
    3. IDLE status: FRESH_START (no active run)
    4. Terminal status (COMPLETED/FAILED/CANCELLED): FRESH_START
    5. RUNNING status: RECONNECT (crash during test execution)
    6. PENDING_APPROVAL status: RESUME_APPROVAL (crash during confirmation)

    This function never raises. All error conditions are captured in the
    returned result's error field and action is set to FRESH_START.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        CrashRecoveryResult with action, connection details, and metadata.
    """
    file_path = current_run.file_path(wiki_root)

    # Case 1: No wiki file exists -- first ever daemon run
    if not file_path.exists():
        logger.info(
            "No current-run wiki file at %s -- fresh start",
            file_path,
        )
        return _build_fresh_start(
            reason="No prior run record found -- starting fresh",
        )

    # Case 2: Wiki file exists -- attempt to parse
    try:
        run = current_run.read(wiki_root)
    except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        logger.warning(
            "Corrupted current-run wiki file at %s: %s",
            file_path,
            exc,
        )
        return _build_fresh_start(
            reason=f"Corrupted wiki file: {exc}",
            error=str(exc),
            source_path=file_path,
        )
    except Exception as exc:
        # Catch-all for unexpected parsing errors
        logger.warning(
            "Unexpected error reading wiki file at %s: %s",
            file_path,
            exc,
        )
        return _build_fresh_start(
            reason=f"Failed to read wiki file: {exc}",
            error=str(exc),
            source_path=file_path,
        )

    # Case 3: read() returned None (defensive -- should not happen)
    if run is None:
        logger.warning(
            "current_run.read() returned None for existing file %s",
            file_path,
        )
        return _build_fresh_start(
            reason="Wiki file exists but could not be parsed",
            error="Read returned None for existing file",
            source_path=file_path,
        )

    # Case 4: Successful parse -- build result from the run record
    result = _build_from_run(run=run, source_path=file_path)

    logger.info(
        "Crash recovery detection: action=%s status=%s run_id=%s "
        "host=%s remote_pid=%s",
        result.action.value,
        result.status.value,
        result.run_id,
        result.host or "none",
        result.remote_pid,
    )

    return result
