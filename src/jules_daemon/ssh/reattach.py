"""Process output re-attachment via SSH.

Given a valid SSH session and a confirmed remote PID, probes the remote
host for available output sources and builds a streaming tail command
for re-attaching to the running process's stdout/stderr stream.

This module sits in the crash recovery pipeline after:
  1. SSH re-establishment (reestablish.py)
  2. PID liveness validation (pid_liveness.py)

And before:
  3. Polling loop resumption (monitor/polling_loop.py)

Re-attachment strategy selection:

1. **Primary: /proc/<PID>/fd/1** -- Checks whether the process's stdout
   file descriptor is readable via ``test -r /proc/<PID>/fd/1``. If
   readable, builds a ``tail -f`` command targeting that fd. This works
   on Linux when the fd points to a regular file, PTY, or named pipe.

2. **Fallback: log file** -- If a known log file path is provided in
   the config and the file exists on the remote host (verified via
   ``test -f <path>``), builds a ``tail -f`` command for that file.

3. **Failure** -- If neither method is available, returns an error
   result so the caller can mark the run as unrecoverable.

The module also provides ``stream_output_lines()``, an async generator
that reads from an established SSH channel (running a tail command),
decodes bytes, buffers partial lines, and yields complete lines as
immutable ``OutputLine`` instances.

Usage:
    from jules_daemon.ssh.reattach import (
        probe_reattach_strategy,
        stream_output_lines,
    )

    strategy = await probe_reattach_strategy(executor, pid=5678)
    if strategy.success:
        # Caller opens a new SSH channel and runs strategy.command
        # Then feeds the channel to stream_output_lines()
        async for line in stream_output_lines(channel):
            process(line.text)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from jules_daemon.ssh.liveness import ProbeExecutor
from jules_daemon.ssh.reader import SSHChannelHandle, read_ssh_output

__all__ = [
    "OutputLine",
    "OutputStreamType",
    "ReattachConfig",
    "ReattachMethod",
    "ReattachStrategy",
    "build_reattach_command",
    "probe_reattach_strategy",
    "stream_output_lines",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_ENCODING = "utf-8"
_DEFAULT_POLL_INTERVAL = 0.1


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReattachMethod(Enum):
    """Method used to re-attach to a remote process's output stream.

    Values:
        PROC_FD: Tailing /proc/<PID>/fd/1 (Linux proc filesystem).
        LOG_FILE: Tailing a known log file path on the remote host.
    """

    PROC_FD = "proc_fd"
    LOG_FILE = "log_file"


class OutputStreamType(Enum):
    """Classification of an output line's source stream.

    Values:
        STDOUT: Line originated from the process's stdout.
        STDERR: Line originated from the process's stderr.
        COMBINED: Line origin is mixed or unknown (e.g., interleaved).
    """

    STDOUT = "stdout"
    STDERR = "stderr"
    COMBINED = "combined"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReattachConfig:
    """Immutable configuration for process output re-attachment.

    Attributes:
        timeout_seconds: Maximum time to wait for each probe command.
            Must be positive. Defaults to 5.0 seconds.
        log_file_path: Optional known path to the process's output log
            file on the remote host. When provided and the file exists,
            this is used as a fallback if /proc/PID/fd is not readable.
        encoding: Character encoding for decoding output bytes.
            Defaults to 'utf-8'. Must not be empty.
    """

    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    log_file_path: str | None = None
    encoding: str = _DEFAULT_ENCODING

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )
        if not self.encoding or not self.encoding.strip():
            raise ValueError("encoding must not be empty")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ReattachStrategy:
    """Immutable result of probing for a reattach method.

    Contains the selected method and the shell command to execute on
    the remote host for re-attaching to the process output stream.

    Attributes:
        success: True if a reattach method was found.
        method: The selected reattach method (None if no method found).
        pid: The remote process ID being reattached.
        command: The shell command to run for reattachment (empty string
            if no method was found).
        error: Human-readable error description (None on success).
        latency_ms: Total wall-clock time for all probes in milliseconds.
        timestamp: UTC datetime when the probing completed.
    """

    success: bool
    method: ReattachMethod | None
    pid: int
    command: str
    error: str | None
    latency_ms: float
    timestamp: datetime


@dataclass(frozen=True)
class OutputLine:
    """Immutable representation of a single output line from the stream.

    Attributes:
        text: The decoded text content of the line (without trailing
            newline). May be empty for blank lines.
        stream: Classification of which output stream produced this line.
        timestamp: UTC datetime when this line was read and decoded.
        sequence: Monotonically increasing sequence number within the
            stream, for ordering without relying on timestamp precision.
    """

    text: str
    stream: OutputStreamType
    timestamp: datetime
    sequence: int


# ---------------------------------------------------------------------------
# Internal: probe command builders
# ---------------------------------------------------------------------------


def _build_proc_fd_probe_command(pid: int) -> str:
    """Build the command to test if /proc/<PID>/fd/1 is readable."""
    return f"test -r /proc/{pid}/fd/1"


def _build_log_file_probe_command(path: str) -> str:
    """Build the command to test if a log file exists."""
    return f"test -f {path}"


# ---------------------------------------------------------------------------
# Internal: safe command execution
# ---------------------------------------------------------------------------


async def _safe_execute(
    executor: ProbeExecutor,
    command: str,
    timeout: float,
) -> tuple[str, int] | Exception:
    """Execute a probe command with timeout safety.

    Returns:
        Tuple of (output, exit_code) on success, or the caught
        Exception on failure.
    """
    try:
        raw_output, exit_code = await asyncio.wait_for(
            executor.execute_probe(command, timeout),
            timeout=timeout,
        )
        return (raw_output.strip(), exit_code)
    except Exception as exc:
        return exc


# ---------------------------------------------------------------------------
# Internal: result builders
# ---------------------------------------------------------------------------


def _build_success_strategy(
    *,
    method: ReattachMethod,
    pid: int,
    command: str,
    latency_ms: float,
) -> ReattachStrategy:
    """Build a successful reattach strategy result."""
    return ReattachStrategy(
        success=True,
        method=method,
        pid=pid,
        command=command,
        error=None,
        latency_ms=latency_ms,
        timestamp=_now_utc(),
    )


def _build_failure_strategy(
    *,
    pid: int,
    error: str,
    latency_ms: float,
) -> ReattachStrategy:
    """Build a failed reattach strategy result."""
    return ReattachStrategy(
        success=False,
        method=None,
        pid=pid,
        command="",
        error=error,
        latency_ms=latency_ms,
        timestamp=_now_utc(),
    )


# ---------------------------------------------------------------------------
# Public API: build_reattach_command
# ---------------------------------------------------------------------------


def build_reattach_command(
    method: ReattachMethod,
    pid: int,
    log_file_path: str | None = None,
) -> str:
    """Build the shell command for re-attaching to process output.

    Produces a ``tail -f`` command targeting either the process's
    stdout file descriptor via /proc or a known log file path.

    Args:
        method: The reattach method to use.
        pid: The remote process ID. Must be a positive integer.
        log_file_path: Required when method is LOG_FILE. The absolute
            path to the log file on the remote host.

    Returns:
        Shell command string ready for SSH execution.

    Raises:
        ValueError: If pid is not positive, or if LOG_FILE method is
            selected without a log_file_path.
    """
    if pid <= 0:
        raise ValueError(f"PID must be a positive integer, got {pid}")

    if method == ReattachMethod.PROC_FD:
        return f"tail -f /proc/{pid}/fd/1 2>/dev/null"

    if method == ReattachMethod.LOG_FILE:
        if not log_file_path:
            raise ValueError(
                "log_file_path is required for LOG_FILE reattach method"
            )
        return f"tail -f {log_file_path} 2>/dev/null"

    # Should not reach here, but handle defensively
    raise ValueError(f"Unknown reattach method: {method!r}")


# ---------------------------------------------------------------------------
# Public API: probe_reattach_strategy
# ---------------------------------------------------------------------------


async def probe_reattach_strategy(
    executor: ProbeExecutor,
    pid: int,
    config: ReattachConfig | None = None,
) -> ReattachStrategy:
    """Probe the remote host and select the best reattach method.

    Executes lightweight probe commands via the provided executor to
    determine which output source is available for the given PID:

    1. Primary: ``test -r /proc/<PID>/fd/1`` -- checks if the stdout
       file descriptor is readable via the proc filesystem.

    2. Fallback: ``test -f <log_file_path>`` -- checks if a configured
       log file exists on the remote host (only attempted when
       config.log_file_path is set and the /proc probe failed).

    3. Failure: If neither probe succeeds, returns an error strategy.

    Args:
        executor: ProbeExecutor that runs commands on the remote host.
        pid: The remote process ID to reattach to. Must be positive.
        config: Optional reattach configuration (timeout, log file path,
            encoding). When None, uses default ReattachConfig.

    Returns:
        Immutable ReattachStrategy with the selected method and command,
        or an error if no method is available. Never raises for probe
        failures -- all errors are captured in the result.

    Raises:
        ValueError: If pid is not a positive integer.
    """
    if pid <= 0:
        raise ValueError(f"PID must be a positive integer, got {pid}")

    effective_config = config if config is not None else ReattachConfig()
    timeout = effective_config.timeout_seconds
    start_ns = time.monotonic_ns()

    # Track probe errors for the error message
    probe_errors: list[str] = []

    # -- Step 1: Try /proc/<PID>/fd/1 --

    proc_command = _build_proc_fd_probe_command(pid)
    proc_result = await _safe_execute(executor, proc_command, timeout)

    if isinstance(proc_result, Exception):
        error_desc = f"/proc probe: {type(proc_result).__name__}: {proc_result}"
        probe_errors.append(error_desc)
        logger.warning(
            "/proc/fd probe for PID %d failed: %s", pid, error_desc
        )
    else:
        _output, exit_code = proc_result
        if exit_code == 0:
            # /proc/PID/fd/1 is readable
            elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
            reattach_cmd = build_reattach_command(ReattachMethod.PROC_FD, pid)
            logger.info(
                "Reattach via /proc/fd for PID %d: success (%.1fms)",
                pid,
                elapsed_ms,
            )
            return _build_success_strategy(
                method=ReattachMethod.PROC_FD,
                pid=pid,
                command=reattach_cmd,
                latency_ms=elapsed_ms,
            )

        probe_errors.append(
            f"/proc probe: /proc/{pid}/fd/1 not readable (exit={exit_code})"
        )
        logger.info(
            "/proc/{%d}/fd/1 not readable (exit=%d), checking fallback",
            pid,
            exit_code,
        )

    # -- Step 2: Try log file fallback --

    if effective_config.log_file_path is not None:
        log_command = _build_log_file_probe_command(
            effective_config.log_file_path
        )
        log_result = await _safe_execute(executor, log_command, timeout)

        if isinstance(log_result, Exception):
            error_desc = (
                f"log file probe: "
                f"{type(log_result).__name__}: {log_result}"
            )
            probe_errors.append(error_desc)
            logger.warning(
                "Log file probe for %s failed: %s",
                effective_config.log_file_path,
                error_desc,
            )
        else:
            _output, exit_code = log_result
            if exit_code == 0:
                # Log file exists
                elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
                reattach_cmd = build_reattach_command(
                    ReattachMethod.LOG_FILE,
                    pid,
                    log_file_path=effective_config.log_file_path,
                )
                logger.info(
                    "Reattach via log file %s for PID %d: success (%.1fms)",
                    effective_config.log_file_path,
                    pid,
                    elapsed_ms,
                )
                return _build_success_strategy(
                    method=ReattachMethod.LOG_FILE,
                    pid=pid,
                    command=reattach_cmd,
                    latency_ms=elapsed_ms,
                )

            probe_errors.append(
                f"log file probe: {effective_config.log_file_path} "
                f"not found (exit={exit_code})"
            )
            logger.info(
                "Log file %s not found (exit=%d)",
                effective_config.log_file_path,
                exit_code,
            )

    # -- Step 3: All methods exhausted --

    elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
    combined_error = (
        f"No reattach method available for PID {pid}: "
        + "; ".join(probe_errors)
    )
    logger.warning(
        "Reattach failed for PID %d: %s (%.1fms)",
        pid,
        combined_error,
        elapsed_ms,
    )
    return _build_failure_strategy(
        pid=pid,
        error=combined_error,
        latency_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Public API: stream_output_lines
# ---------------------------------------------------------------------------


async def stream_output_lines(
    channel: SSHChannelHandle,
    *,
    encoding: str = _DEFAULT_ENCODING,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> AsyncGenerator[OutputLine, None]:
    """Yield decoded output lines from an SSH channel.

    Reads from the given SSH channel (which should be running a
    ``tail -f`` or similar streaming command), decodes bytes to text,
    buffers partial lines, and yields complete ``OutputLine`` instances
    as newlines are encountered.

    When the channel reaches EOF or closes, any remaining buffered
    partial line is flushed as a final OutputLine.

    Args:
        channel: An SSH channel running a streaming command. Must
            satisfy the SSHChannelHandle protocol.
        encoding: Character encoding for decoding output bytes.
            Invalid bytes are replaced (errors='replace'). Defaults
            to 'utf-8'.
        poll_interval: Seconds between poll iterations when the
            channel has no data. Defaults to 0.1s.

    Yields:
        OutputLine instances, one per complete line (or partial line
        on EOF). Each has a monotonically increasing sequence number.
    """
    buffer = ""
    sequence = 0

    while True:
        ssh_output = await read_ssh_output(channel)

        # Decode and accumulate stdout + stderr into buffer
        if ssh_output.has_data:
            if ssh_output.stdout:
                buffer += ssh_output.stdout.decode(encoding, errors="replace")
            if ssh_output.stderr:
                buffer += ssh_output.stderr.decode(encoding, errors="replace")

        # Extract and yield complete lines from the buffer
        while "\n" in buffer:
            line_text, buffer = buffer.split("\n", 1)
            sequence += 1
            yield OutputLine(
                text=line_text,
                stream=OutputStreamType.COMBINED,
                timestamp=_now_utc(),
                sequence=sequence,
            )

        # Check for terminal conditions
        is_terminal = ssh_output.is_eof or ssh_output.channel_closed
        if is_terminal:
            # Flush any remaining partial line in the buffer
            if buffer:
                sequence += 1
                yield OutputLine(
                    text=buffer,
                    stream=OutputStreamType.COMBINED,
                    timestamp=_now_utc(),
                    sequence=sequence,
                )
            break

        # Sleep briefly if no data was available to avoid busy-waiting
        if not ssh_output.has_data:
            await asyncio.sleep(poll_interval)
