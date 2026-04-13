"""End-to-end SSH execution pipeline for the ``run`` verb.

Orchestrates the full lifecycle of a remote command execution:

1. Resolve SSH credentials via the credential chain
2. Connect to the remote host via paramiko (in a thread pool)
3. Execute the command with timeout
4. Capture stdout and stderr
5. Track state in the wiki (current-run file)
6. Promote the completed run to history
7. Return a structured result

The pipeline uses ``asyncio.to_thread`` to wrap blocking paramiko
calls, keeping the event loop free for concurrent IPC handling.

Usage::

    from jules_daemon.execution.run_pipeline import execute_run, RunResult

    result = await execute_run(
        target_host="10.0.1.50",
        target_user="root",
        command="python3.8 test.py --iteration 100",
        target_port=22,
        wiki_root=Path("/data/wiki"),
    )
    if result.success:
        print(result.stdout)
    else:
        print(result.error)
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paramiko

from jules_daemon.ssh.credentials import (
    REDACTED,
    SSHCredential,
    resolve_ssh_credentials,
)
from jules_daemon.ssh.errors import SSHAuthenticationError, SSHConnectionError
from jules_daemon.wiki import current_run as current_run_io
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    Progress,
    SSHTarget,
)
from jules_daemon.wiki.run_promotion import promote_run

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "RunResult",
    "execute_run",
]

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: int = 3600
"""Default command timeout: 1 hour."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RunResult:
    """Immutable result of a remote command execution.

    Attributes:
        success: True if the command completed with exit code 0.
        run_id: Unique identifier for this run.
        command: The command string that was executed.
        target_host: The remote host.
        target_user: The SSH username.
        exit_code: Remote process exit code (None if connection failed).
        stdout: Captured standard output.
        stderr: Captured standard error.
        error: Human-readable error description (None on success).
        duration_seconds: Wall-clock execution time.
        started_at: UTC timestamp when execution began.
        completed_at: UTC timestamp when execution finished.
    """

    success: bool
    run_id: str
    command: str
    target_host: str
    target_user: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    duration_seconds: float = 0.0
    started_at: datetime = field(default_factory=_now_utc)
    completed_at: datetime = field(default_factory=_now_utc)


# ---------------------------------------------------------------------------
# Internal: blocking paramiko execution (runs in thread pool)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ParamikoResult:
    """Raw result from the blocking paramiko execution."""

    exit_code: int
    stdout: str
    stderr: str


def _execute_via_paramiko(
    *,
    host: str,
    port: int,
    username: str,
    credential: SSHCredential | None,
    command: str,
    timeout: int,
    on_output: Callable[[str], None] | None = None,
) -> _ParamikoResult:
    """Execute a command on a remote host via paramiko (blocking).

    This function runs in a thread pool via asyncio.to_thread.
    It establishes an SSH connection, executes the command, reads
    output line-by-line (streaming each line to the optional callback),
    and returns the result. The connection is always closed on exit.

    Args:
        host: Remote hostname or IP address.
        port: SSH port number.
        username: SSH login username.
        credential: Resolved SSH credential (password). None for
            key-based auth.
        command: Shell command string to execute.
        timeout: Maximum execution time in seconds.
        on_output: Optional callback invoked with each line of stdout
            as it is received. Used for streaming output to watchers.

    Returns:
        _ParamikoResult with exit code, stdout, and stderr.

    Raises:
        SSHAuthenticationError: On authentication failure.
        SSHConnectionError: On connection failure.
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs: dict[str, Any] = {
        "hostname": host,
        "port": port,
        "username": username,
        "timeout": min(timeout, 30),  # connection timeout cap
        "allow_agent": True,
        "look_for_keys": True,
    }

    if credential is not None:
        connect_kwargs["password"] = credential.password
        # When using password auth, disable key-based auth to avoid
        # paramiko trying (and failing) public key first
        connect_kwargs["allow_agent"] = False
        connect_kwargs["look_for_keys"] = False
        # If the credential provides a username override, use it
        if credential.username is not None:
            connect_kwargs["username"] = credential.username

    try:
        logger.info(
            "Connecting to %s@%s:%d (auth_source=%s)",
            connect_kwargs["username"],
            host,
            port,
            credential.source if credential else "key-based",
        )
        client.connect(**connect_kwargs)
    except paramiko.AuthenticationException as exc:
        raise SSHAuthenticationError(
            f"Authentication failed for {username}@{host}:{port}: {exc}"
        ) from exc
    except (
        paramiko.SSHException,
        OSError,
        TimeoutError,
        ConnectionRefusedError,
    ) as exc:
        raise SSHConnectionError(
            f"Connection failed to {host}:{port}: {exc}"
        ) from exc

    try:
        logger.info(
            "Executing command on %s@%s:%d: %s",
            connect_kwargs["username"],
            host,
            port,
            command[:120],
        )

        # exec_command returns (stdin, stdout, stderr) channels
        _, stdout_channel, stderr_channel = client.exec_command(
            command,
            timeout=timeout,
        )

        # Read output line by line for streaming, or all at once
        if on_output is not None:
            stdout_lines: list[str] = []
            for raw_line in iter(stdout_channel.readline, ""):
                decoded = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace")
                stdout_lines.append(decoded)
                on_output(decoded)
            stdout_text = "".join(stdout_lines)
        else:
            stdout_text = stdout_channel.read().decode("utf-8", errors="replace")

        stderr_text = stderr_channel.read().decode("utf-8", errors="replace")
        exit_code = stdout_channel.channel.recv_exit_status()

        logger.info(
            "Command completed on %s:%d with exit_code=%d "
            "(stdout=%d bytes, stderr=%d bytes)",
            host,
            port,
            exit_code,
            len(stdout_text),
            len(stderr_text),
        )

        return _ParamikoResult(
            exit_code=exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
        )
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Internal: wiki state management
# ---------------------------------------------------------------------------


def _write_running_state(
    *,
    wiki_root: Path,
    run: CurrentRun,
    command_str: str,
) -> CurrentRun:
    """Transition the run to RUNNING and persist to wiki.

    Args:
        wiki_root: Path to the wiki root directory.
        run: The current run state (in PENDING_APPROVAL).
        command_str: The resolved shell command.

    Returns:
        The updated CurrentRun in RUNNING state.
    """
    updated = run.with_running(resolved_shell=command_str)
    current_run_io.write(wiki_root, updated)
    return updated


def _write_completed_state(
    *,
    wiki_root: Path,
    run: CurrentRun,
    exit_code: int,
    last_output_line: str,
) -> CurrentRun:
    """Transition the run to COMPLETED and persist to wiki.

    Args:
        wiki_root: Path to the wiki root directory.
        run: The current run state (in RUNNING).
        exit_code: The remote process exit code.
        last_output_line: Last line of output for the progress record.

    Returns:
        The updated CurrentRun in COMPLETED state.
    """
    final_progress = Progress(
        percent=100.0,
        last_output_line=last_output_line[:200],
        checkpoint_at=_now_utc(),
    )
    updated = run.with_completed(final_progress)
    current_run_io.write(wiki_root, updated)
    return updated


def _write_failed_state(
    *,
    wiki_root: Path,
    run: CurrentRun,
    error: str,
    last_output_line: str,
) -> CurrentRun:
    """Transition the run to FAILED and persist to wiki.

    Args:
        wiki_root: Path to the wiki root directory.
        run: The current run state.
        error: Human-readable error description.
        last_output_line: Last line of output for the progress record.

    Returns:
        The updated CurrentRun in FAILED state.
    """
    final_progress = Progress(
        last_output_line=last_output_line[:200],
        checkpoint_at=_now_utc(),
    )
    updated = run.with_failed(error, final_progress)
    current_run_io.write(wiki_root, updated)
    return updated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_run(
    *,
    target_host: str,
    target_user: str,
    command: str,
    target_port: int = 22,
    wiki_root: Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    on_output: Callable[[str], None] | None = None,
    run_id: str | None = None,
) -> RunResult:
    """Execute a command on a remote host via SSH.

    Full pipeline:
    1. Resolve SSH credentials for the target host
    2. Create wiki current-run record (PENDING_APPROVAL -> RUNNING)
    3. Connect via paramiko and execute the command
    4. Capture stdout/stderr
    5. Update wiki state (COMPLETED or FAILED)
    6. Promote terminal run to history
    7. Return structured result

    The paramiko connection and command execution run in a thread pool
    to avoid blocking the asyncio event loop.

    Args:
        target_host: Remote hostname or IP address.
        target_user: SSH username.
        command: Shell command string to execute.
        target_port: SSH port (default 22).
        wiki_root: Path to the wiki root directory.
        timeout: Maximum execution time in seconds (default 1 hour).
        run_id: Optional caller-supplied run identifier. When provided,
            the wiki current-run record and returned RunResult reuse this
            identifier instead of generating a fresh one internally.

    Returns:
        RunResult with execution outcome, output, and metadata.
    """
    start_time = _now_utc()

    # Build domain objects
    ssh_target = SSHTarget(
        host=target_host,
        user=target_user,
        port=target_port,
    )
    cmd = Command(natural_language=command, resolved_shell=command)

    # Create initial run record
    daemon_pid = os.getpid()
    run = CurrentRun(run_id=run_id) if run_id is not None else CurrentRun()
    run = run.with_pending_approval(
        ssh_target=ssh_target,
        command=cmd,
        daemon_pid=daemon_pid,
    )
    run_id = run.run_id

    # Step 1: Resolve credentials
    credential = resolve_ssh_credentials(target_host)
    if credential is not None:
        logger.info(
            "Resolved credentials for %s (source=%s, password=%s)",
            target_host,
            credential.source,
            REDACTED,
        )
    else:
        logger.info(
            "No password credentials for %s; using key-based auth",
            target_host,
        )

    # Step 2: Write initial wiki state and transition to RUNNING
    try:
        current_run_io.write(wiki_root, run)
        run = _write_running_state(
            wiki_root=wiki_root,
            run=run,
            command_str=command,
        )
    except Exception as exc:
        logger.error("Failed to write wiki state: %s", exc)
        # Continue execution even if wiki write fails

    # Step 3: Execute via paramiko in thread pool
    try:
        paramiko_result = await asyncio.to_thread(
            _execute_via_paramiko,
            host=target_host,
            port=target_port,
            username=target_user,
            credential=credential,
            command=command,
            timeout=timeout,
            on_output=on_output,
        )
    except (SSHAuthenticationError, SSHConnectionError) as exc:
        end_time = _now_utc()
        error_msg = str(exc)
        logger.warning("SSH execution failed for run %s: %s", run_id, error_msg)

        # Update wiki to FAILED
        try:
            run = _write_failed_state(
                wiki_root=wiki_root,
                run=run,
                error=error_msg,
                last_output_line="",
            )
            promote_run(wiki_root, run)
        except Exception as wiki_exc:
            logger.error("Failed to update wiki after SSH failure: %s", wiki_exc)

        return RunResult(
            success=False,
            run_id=run_id,
            command=command,
            target_host=target_host,
            target_user=target_user,
            error=error_msg,
            duration_seconds=(end_time - start_time).total_seconds(),
            started_at=start_time,
            completed_at=end_time,
        )
    except Exception as exc:
        end_time = _now_utc()
        error_msg = f"Unexpected error during SSH execution: {exc}"
        logger.error("Unexpected error for run %s: %s", run_id, error_msg)

        try:
            run = _write_failed_state(
                wiki_root=wiki_root,
                run=run,
                error=error_msg,
                last_output_line="",
            )
            promote_run(wiki_root, run)
        except Exception as wiki_exc:
            logger.error("Failed to update wiki after error: %s", wiki_exc)

        return RunResult(
            success=False,
            run_id=run_id,
            command=command,
            target_host=target_host,
            target_user=target_user,
            error=error_msg,
            duration_seconds=(end_time - start_time).total_seconds(),
            started_at=start_time,
            completed_at=end_time,
        )

    # Step 4: Determine success and build result
    end_time = _now_utc()
    is_success = paramiko_result.exit_code == 0

    # Compute last output line from stdout (or stderr if stdout empty)
    output_for_last_line = paramiko_result.stdout or paramiko_result.stderr
    output_lines = output_for_last_line.strip().splitlines()
    last_output_line = output_lines[-1] if output_lines else ""

    # Step 5: Update wiki state
    try:
        if is_success:
            run = _write_completed_state(
                wiki_root=wiki_root,
                run=run,
                exit_code=paramiko_result.exit_code,
                last_output_line=last_output_line,
            )
        else:
            error_summary = (
                f"Command exited with code {paramiko_result.exit_code}"
            )
            run = _write_failed_state(
                wiki_root=wiki_root,
                run=run,
                error=error_summary,
                last_output_line=last_output_line,
            )

        # Step 6: Promote to history
        promote_run(wiki_root, run)
    except Exception as wiki_exc:
        logger.error(
            "Failed to update wiki after execution: %s", wiki_exc
        )

    # Step 7: Build and return result
    error_msg_final = None if is_success else (
        f"Command exited with code {paramiko_result.exit_code}"
    )

    return RunResult(
        success=is_success,
        run_id=run_id,
        command=command,
        target_host=target_host,
        target_user=target_user,
        exit_code=paramiko_result.exit_code,
        stdout=paramiko_result.stdout,
        stderr=paramiko_result.stderr,
        error=error_msg_final,
        duration_seconds=(end_time - start_time).total_seconds(),
        started_at=start_time,
        completed_at=end_time,
    )
