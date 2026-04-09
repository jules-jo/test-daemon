"""Wiki state reader for extracting SSH connection parameters and run metadata.

Loads the current-run wiki file and deserializes it to extract the fields
needed for reconnection after a daemon restart or crash. This module sits
between the daemon's reconnection logic and the raw wiki persistence layer.

Unlike boot_reader (which extracts status fields for boot decisions), this
module extracts:
- SSH connection parameters: host, port, username, key_path
- Run metadata: run_id, status, resolved shell command, process IDs
- Progress snapshot: current completion percentage

On daemon restart, call load_reconnection_state(wiki_root) to get a
ReconnectionState with all fields needed to re-establish SSH and resume
monitoring.

Usage:
    from pathlib import Path
    from jules_daemon.wiki.state_reader import load_reconnection_state

    state = load_reconnection_state(Path("wiki"))
    if state.can_reconnect:
        # Use state.connection for SSH params
        # Use state.remote_pid to reattach to process
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

from jules_daemon.wiki import current_run
from jules_daemon.wiki.models import CurrentRun, RunStatus

logger = logging.getLogger(__name__)


class LoadResult(Enum):
    """Outcome of attempting to load the wiki state file."""

    LOADED = "loaded"
    NO_FILE = "no_file"
    CORRUPTED = "corrupted"


@dataclass(frozen=True)
class ConnectionParams:
    """SSH connection parameters extracted from the wiki state.

    Contains the four fields needed to establish an SSH connection:
    host, port, username, and optional key_path for key-based auth.
    """

    host: str
    port: int
    username: str
    key_path: Optional[str]


@dataclass(frozen=True)
class ReconnectionState:
    """Complete reconnection context extracted from the wiki state file.

    Provides everything the daemon needs to decide whether and how to
    reconnect to a running remote process after a restart.

    Fields:
        result: How the wiki file load resolved
        connection: SSH connection parameters (None if no target set)
        run_id: Unique identifier for the run
        status: Current run lifecycle state
        resolved_shell: The approved shell command (empty if not yet approved)
        daemon_pid: Local daemon process ID from the prior instance
        remote_pid: Remote SSH process ID (None if not yet started)
        natural_language_command: The original user request
        progress_percent: Last known completion percentage (0.0 to 100.0)
        error: Error message if the run failed (None otherwise)
        source_path: Path to the wiki file that was read (None if no file)
    """

    result: LoadResult
    connection: Optional[ConnectionParams]
    run_id: str
    status: RunStatus
    resolved_shell: str
    daemon_pid: Optional[int]
    remote_pid: Optional[int]
    natural_language_command: str
    progress_percent: float
    error: Optional[str]
    source_path: Optional[Path]

    @property
    def has_connection(self) -> bool:
        """True if SSH connection parameters are available."""
        return self.connection is not None

    @property
    def can_reconnect(self) -> bool:
        """True if the run is in an active state with connection params.

        Active states (RUNNING, PENDING_APPROVAL) indicate a run was in
        progress and reconnection may be possible. Terminal states and
        IDLE cannot be reconnected. A valid connection is also required.
        """
        if self.result != LoadResult.LOADED:
            return False
        if self.connection is None:
            return False
        return self.status in (RunStatus.RUNNING, RunStatus.PENDING_APPROVAL)


def _extract_connection(run: CurrentRun) -> Optional[ConnectionParams]:
    """Extract SSH connection parameters from a CurrentRun record.

    Args:
        run: The deserialized current run state.

    Returns:
        ConnectionParams if an SSH target is set, None otherwise.
    """
    if run.ssh_target is None:
        return None
    return ConnectionParams(
        host=run.ssh_target.host,
        port=run.ssh_target.port,
        username=run.ssh_target.user,
        key_path=run.ssh_target.key_path,
    )


def _build_from_run(
    run: CurrentRun,
    source_path: Path,
) -> ReconnectionState:
    """Build a ReconnectionState from a deserialized CurrentRun."""
    connection = _extract_connection(run)

    resolved_shell = ""
    natural_language = ""
    if run.command is not None:
        resolved_shell = run.command.resolved_shell
        natural_language = run.command.natural_language

    return ReconnectionState(
        result=LoadResult.LOADED,
        connection=connection,
        run_id=run.run_id,
        status=run.status,
        resolved_shell=resolved_shell,
        daemon_pid=run.pids.daemon,
        remote_pid=run.pids.remote,
        natural_language_command=natural_language,
        progress_percent=run.progress.percent,
        error=run.error,
        source_path=source_path,
    )


def _build_empty(
    result: LoadResult,
    error: Optional[str] = None,
    source_path: Optional[Path] = None,
) -> ReconnectionState:
    """Build a safe empty ReconnectionState for missing/corrupted file cases."""
    return ReconnectionState(
        result=result,
        connection=None,
        run_id="",
        status=RunStatus.IDLE,
        resolved_shell="",
        daemon_pid=None,
        remote_pid=None,
        natural_language_command="",
        progress_percent=0.0,
        error=error,
        source_path=source_path,
    )


def load_reconnection_state(wiki_root: Path) -> ReconnectionState:
    """Load the current-run wiki file and extract reconnection context.

    This is the primary entry point for daemon reconnection. It reads the
    wiki file at {wiki_root}/pages/daemon/current-run.md, parses the YAML
    frontmatter, and returns a ReconnectionState with all fields needed to
    re-establish SSH and resume monitoring.

    Behavior by scenario:
    - File missing: returns empty state with result=NO_FILE
    - File corrupted: returns empty state with result=CORRUPTED and
      the parse error in the error field
    - File valid: returns full state with result=LOADED and extracted
      connection parameters and run metadata

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        ReconnectionState with extracted fields. Never raises.
    """
    file_path = current_run.file_path(wiki_root)

    # Case 1: No file exists -- fresh daemon start
    if not file_path.exists():
        logger.info(
            "No current-run wiki file at %s -- no reconnection state",
            file_path,
        )
        return _build_empty(result=LoadResult.NO_FILE)

    # Case 2: File exists -- try to parse it
    # Catch only parsing-domain exceptions; let programmer errors propagate
    try:
        run = current_run.read(wiki_root)
    except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        logger.warning(
            "Corrupted current-run wiki file at %s: %s",
            file_path,
            exc,
        )
        return _build_empty(
            result=LoadResult.CORRUPTED,
            error=str(exc),
            source_path=file_path,
        )

    # Case 3: read() returned None (defensive -- should not happen for
    # existing files, but handle gracefully)
    if run is None:
        logger.warning(
            "current_run.read() returned None for existing file %s",
            file_path,
        )
        return _build_empty(
            result=LoadResult.CORRUPTED,
            error="Read returned None for existing file",
            source_path=file_path,
        )

    # Case 4: Successful load
    state = _build_from_run(run=run, source_path=file_path)
    logger.info(
        "Loaded reconnection state from %s: status=%s run_id=%s "
        "host=%s can_reconnect=%s",
        file_path,
        state.status.value,
        state.run_id,
        state.connection.host if state.connection else "none",
        state.can_reconnect,
    )
    return state
