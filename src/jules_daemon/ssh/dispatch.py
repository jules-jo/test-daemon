"""SSH command dispatch with wiki state updates.

Issues a generated recovery command (resume or restart) over a
re-established SSH connection and updates the wiki current-run entry
with the new action taken.

The dispatch flow:
1. Build the full shell invocation (cd + env + command)
2. Execute the command over the SSH handle
3. Update the wiki state to RUNNING with the new command and PID
4. On error: update the wiki state to FAILED with the error detail

This module bridges command generation (command_gen) with SSH execution
and wiki persistence. It is the single entry point for issuing recovery
commands after crash detection and resume/restart decision-making.

Security invariant: the command being dispatched must have been
previously approved by the human operator. This module does NOT
perform approval -- it trusts that the caller has already obtained
approval via the confirmation flow.

Usage:
    from jules_daemon.ssh.dispatch import dispatch_recovery_command

    result = await dispatch_recovery_command(
        handle=ssh_handle,
        generated_command=gen_cmd,
        wiki_root=wiki_root,
        daemon_pid=os.getpid(),
    )
    if result.success:
        # Monitor the running test via result.remote_pid
        ...
    else:
        # Handle dispatch failure
        log_error(result.error)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from jules_daemon.ssh.command_gen import GeneratedCommand, RecoveryCommandAction
from jules_daemon.wiki import current_run
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    ProcessIDs,
    Progress,
    RunStatus,
    SSHTarget,
)

__all__ = [
    "DispatchResult",
    "SSHDispatchHandle",
    "dispatch_recovery_command",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SSHDispatchHandle(Protocol):
    """Protocol for an SSH connection handle that can execute commands.

    Implementations wrap the SSH library's channel/session object and
    provide a simple async interface for executing commands and
    retrieving the remote process PID.
    """

    async def execute(self, command: str, timeout: int) -> int | None:
        """Execute a command on the remote host.

        Args:
            command: The full shell command string to execute.
            timeout: Maximum execution time in seconds for the initial
                dispatch (not the full test run).

        Returns:
            The remote process PID, or None if PID cannot be determined.

        Raises:
            OSError: On SSH connection or execution failure.
            TimeoutError: If the command dispatch exceeds the timeout.
        """
        ...

    @property
    def session_id(self) -> str:
        """Unique identifier for this SSH session."""
        ...


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DispatchResult:
    """Immutable result of a command dispatch operation.

    Contains the outcome of executing the recovery command over SSH
    and updating the wiki state.

    Attributes:
        success: True if the command was dispatched and wiki was updated.
        action: Whether this was a RESUME or RESTART dispatch.
        command_string: The actual command string that was executed.
        run_id: The run identifier.
        remote_pid: PID of the remote process (None on failure).
        error: Human-readable error description (None on success).
        wiki_updated: Whether the wiki state was successfully updated.
        session_id: SSH session identifier for correlation.
        timestamp: UTC datetime when this result was produced.
    """

    success: bool
    action: RecoveryCommandAction
    command_string: str
    run_id: str
    remote_pid: int | None
    error: str | None
    wiki_updated: bool
    session_id: str
    timestamp: datetime = field(default_factory=_now_utc)


# ---------------------------------------------------------------------------
# Internal: build full shell invocation
# ---------------------------------------------------------------------------


def _build_shell_invocation(generated_command: GeneratedCommand) -> str:
    """Build the full shell invocation string with cd and env vars.

    Assembles the command with optional ``cd`` prefix and environment
    variable exports. The resulting string is suitable for execution
    in a remote shell via SSH.

    Args:
        generated_command: The generated command with SSH command details.

    Returns:
        A complete shell invocation string.
    """
    ssh_cmd = generated_command.ssh_command
    parts: list[str] = []

    # Add environment variable exports
    for key, value in sorted(ssh_cmd.environment.items()):
        # Use export syntax for each variable
        safe_value = value.replace("'", "'\\''")
        parts.append(f"{key}='{safe_value}'")

    # Add working directory change
    if ssh_cmd.working_directory is not None:
        parts.append(f"cd {ssh_cmd.working_directory}")

    # Add the command itself
    parts.append(ssh_cmd.command)

    if len(parts) == 1:
        return parts[0]

    # Join with && for sequential execution with error checking
    return " && ".join(parts)


# ---------------------------------------------------------------------------
# Internal: wiki state updaters
# ---------------------------------------------------------------------------


def _update_wiki_running(
    *,
    wiki_root: Path,
    generated_command: GeneratedCommand,
    remote_pid: int | None,
    daemon_pid: int,
    ssh_target: SSHTarget | None,
    natural_language: str | None,
    prior_run: CurrentRun | None,
) -> CurrentRun:
    """Update wiki state to RUNNING after successful command dispatch.

    For RESTART: resets progress counters to zero.
    For RESUME: preserves existing progress from the prior run.

    Args:
        wiki_root: Path to the wiki root directory.
        generated_command: The generated command that was dispatched.
        remote_pid: PID of the remote process.
        daemon_pid: PID of the current daemon process.
        ssh_target: SSH connection target (used when creating new state).
        natural_language: Original NL command (used when creating new state).
        prior_run: The prior run state from wiki, if available.

    Returns:
        The updated CurrentRun instance.
    """
    resolved_shell = generated_command.ssh_command.command

    if prior_run is not None:
        # Update existing run
        if generated_command.is_restart:
            # Restart: reset progress, update command
            updated = CurrentRun(
                status=RunStatus.RUNNING,
                run_id=prior_run.run_id,
                ssh_target=prior_run.ssh_target,
                command=Command(
                    natural_language=prior_run.command.natural_language
                    if prior_run.command
                    else "recovery command",
                ).with_approval(resolved_shell),
                pids=ProcessIDs(daemon=daemon_pid, remote=remote_pid),
                progress=Progress(),
                started_at=_now_utc(),
                completed_at=None,
                error=None,
                created_at=prior_run.created_at,
                updated_at=_now_utc(),
            )
        else:
            # Resume: preserve progress, update command and PIDs
            updated = CurrentRun(
                status=RunStatus.RUNNING,
                run_id=prior_run.run_id,
                ssh_target=prior_run.ssh_target,
                command=Command(
                    natural_language=prior_run.command.natural_language
                    if prior_run.command
                    else "recovery command",
                ).with_approval(resolved_shell),
                pids=ProcessIDs(daemon=daemon_pid, remote=remote_pid),
                progress=prior_run.progress,
                started_at=prior_run.started_at or _now_utc(),
                completed_at=None,
                error=None,
                created_at=prior_run.created_at,
                updated_at=_now_utc(),
            )
    else:
        # No prior run: create new state
        nl = natural_language or "recovery command"
        updated = CurrentRun(
            status=RunStatus.RUNNING,
            ssh_target=ssh_target,
            command=Command(natural_language=nl).with_approval(resolved_shell),
            pids=ProcessIDs(daemon=daemon_pid, remote=remote_pid),
            progress=Progress(),
            started_at=_now_utc(),
        )

    current_run.write(wiki_root, updated)
    return updated


def _update_wiki_failed(
    *,
    wiki_root: Path,
    error: str,
    generated_command: GeneratedCommand,
    daemon_pid: int,
    prior_run: CurrentRun | None,
) -> None:
    """Update wiki state to FAILED after dispatch error.

    Preserves existing progress and metadata while recording the error.

    Args:
        wiki_root: Path to the wiki root directory.
        error: Human-readable error description.
        generated_command: The command that failed to dispatch.
        daemon_pid: PID of the current daemon process.
        prior_run: The prior run state from wiki, if available.
    """
    if prior_run is not None:
        failed = prior_run.with_failed(error, prior_run.progress)
    else:
        failed = CurrentRun(
            status=RunStatus.FAILED,
            error=error,
            pids=ProcessIDs(daemon=daemon_pid),
        )

    current_run.write(wiki_root, failed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def dispatch_recovery_command(
    *,
    handle: SSHDispatchHandle,
    generated_command: GeneratedCommand,
    wiki_root: Path,
    daemon_pid: int,
    ssh_target: SSHTarget | None = None,
    natural_language: str | None = None,
) -> DispatchResult:
    """Dispatch a recovery command over SSH and update wiki state.

    Executes the generated command (resume or restart) on the remote
    host via the SSH handle, then updates the wiki current-run entry
    with the new action taken.

    On success: wiki state transitions to RUNNING with the new command
    string, remote PID, and daemon PID recorded.

    On failure: wiki state transitions to FAILED with the error detail.

    This function never raises. All errors are captured in the
    DispatchResult's error field.

    Args:
        handle: SSH connection handle for executing commands.
        generated_command: The command to dispatch (from build_recovery_command).
        wiki_root: Path to the wiki root directory.
        daemon_pid: PID of the current daemon process.
        ssh_target: SSH target (needed when creating new wiki state).
        natural_language: Original NL command (needed when creating new state).

    Returns:
        DispatchResult with outcome, PID, and wiki update status.
    """
    session_id = handle.session_id
    run_id = generated_command.run_id
    action = generated_command.action

    # Build the full shell invocation
    shell_invocation = _build_shell_invocation(generated_command)

    logger.info(
        "Dispatching %s command for run %s via session %s: %s",
        action.value,
        run_id,
        session_id,
        shell_invocation[:120],
    )

    # Read prior wiki state (for preservation during update)
    prior_run = current_run.read(wiki_root)

    # Execute the command over SSH
    try:
        remote_pid = await handle.execute(
            shell_invocation,
            generated_command.ssh_command.timeout,
        )
    except Exception as exc:
        error_msg = f"SSH dispatch failed: {exc}"
        logger.warning(
            "Dispatch failed for run %s: %s",
            run_id,
            error_msg,
        )

        # Update wiki to FAILED
        wiki_updated = False
        try:
            _update_wiki_failed(
                wiki_root=wiki_root,
                error=error_msg,
                generated_command=generated_command,
                daemon_pid=daemon_pid,
                prior_run=prior_run,
            )
            wiki_updated = True
        except Exception as wiki_exc:
            logger.error(
                "Failed to update wiki after dispatch error: %s",
                wiki_exc,
            )

        return DispatchResult(
            success=False,
            action=action,
            command_string=shell_invocation,
            run_id=run_id,
            remote_pid=None,
            error=error_msg,
            wiki_updated=wiki_updated,
            session_id=session_id,
        )

    # Dispatch succeeded -- update wiki to RUNNING
    wiki_updated = False
    try:
        _update_wiki_running(
            wiki_root=wiki_root,
            generated_command=generated_command,
            remote_pid=remote_pid,
            daemon_pid=daemon_pid,
            ssh_target=ssh_target,
            natural_language=natural_language,
            prior_run=prior_run,
        )
        wiki_updated = True
    except Exception as wiki_exc:
        logger.error(
            "Failed to update wiki after successful dispatch: %s",
            wiki_exc,
        )

    logger.info(
        "Dispatch completed for run %s: action=%s remote_pid=%s "
        "wiki_updated=%s session=%s",
        run_id,
        action.value,
        remote_pid,
        wiki_updated,
        session_id,
    )

    return DispatchResult(
        success=True,
        action=action,
        command_string=shell_invocation,
        run_id=run_id,
        remote_pid=remote_pid,
        error=None,
        wiki_updated=wiki_updated,
        session_id=session_id,
    )
