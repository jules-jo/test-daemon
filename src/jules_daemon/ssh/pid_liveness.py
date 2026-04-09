"""Remote PID liveness validation via SSH.

Checks whether a recovered remote process is still running by executing
``kill -0 <PID>`` as the primary check and falling back to a
``/proc/<PID>`` existence test when kill-0 produces inconclusive results.

This module sits in the crash recovery pipeline between connection
re-establishment and monitoring resumption. After the daemon reconnects
to the remote host, it calls ``validate_pid_liveness()`` to determine
whether the previously-running test process is still alive before
attempting to reattach to its output.

Health classification logic:

1. **kill -0** (primary):
   - Exit code 0: Process is alive and signalable -> ALIVE
   - Non-zero exit + "Operation not permitted" (EPERM): Process is alive
     but owned by a different user -> ALIVE
   - Non-zero exit + "No such process" (ESRCH): Process is dead -> DEAD
   - Timeout or transport error: Inconclusive -> fall through to /proc

2. **/proc/<PID>** (fallback):
   - ``test -d /proc/<PID>`` exit code 0: Process directory exists -> ALIVE
   - ``test -d /proc/<PID>`` exit code non-zero: No directory -> DEAD
   - Timeout or transport error: Both checks failed -> UNKNOWN

Usage:
    from jules_daemon.ssh.pid_liveness import validate_pid_liveness

    result = await validate_pid_liveness(executor, pid=5678)
    if result.alive:
        # Reattach to process output
        ...
    elif result.status == PidStatus.DEAD:
        # Mark run as failed, process exited while daemon was down
        ...
    else:
        # UNKNOWN -- both checks failed, connection may be broken
        ...
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from jules_daemon.ssh.liveness import ProbeExecutor

__all__ = [
    "PidCheckConfig",
    "PidCheckMethod",
    "PidLivenessResult",
    "PidStatus",
    "validate_pid_liveness",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_SECONDS = 5.0

# Patterns in kill -0 output for classifying the result.
# Case-insensitive matching is used.
_EPERM_PATTERN = "operation not permitted"
_ESRCH_PATTERN = "no such process"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PidStatus(Enum):
    """Liveness status of a remote process.

    Values:
        ALIVE: Process confirmed running on the remote host.
        DEAD: Process confirmed not running on the remote host.
        UNKNOWN: Could not determine status (both checks failed).
    """

    ALIVE = "alive"
    DEAD = "dead"
    UNKNOWN = "unknown"


class PidCheckMethod(Enum):
    """Which check method confirmed the PID status.

    Values:
        KILL_ZERO: Determined by ``kill -0 <PID>`` result.
        PROC_FALLBACK: Determined by ``test -d /proc/<PID>`` result.
    """

    KILL_ZERO = "kill_zero"
    PROC_FALLBACK = "proc_fallback"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PidCheckConfig:
    """Immutable configuration for a PID liveness check.

    Attributes:
        timeout_seconds: Maximum time to wait for each individual check
            command (kill -0 and /proc test each get their own timeout).
            Must be positive.
    """

    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PidLivenessResult:
    """Immutable result of a remote PID liveness check.

    Contains the combined results of the kill -0 primary check and
    the optional /proc fallback, along with diagnostic metadata.

    Attributes:
        pid: The remote process ID that was checked.
        alive: True if the process was confirmed alive by any method.
        status: Structured liveness classification (ALIVE/DEAD/UNKNOWN).
        method: Which check method confirmed the status (None if UNKNOWN).
        kill_zero_exit_code: Exit code from kill -0 (None if not executed
            or if the command errored/timed out).
        kill_zero_output: Stripped stdout/stderr from kill -0.
        proc_exit_code: Exit code from /proc test (None if not executed
            or if the command errored/timed out).
        proc_output: Stripped stdout from /proc test.
        error: Human-readable error description (None if status is
            definitively ALIVE or DEAD).
        latency_ms: Total wall-clock time for all checks in milliseconds.
        timestamp: UTC datetime when the check completed.
    """

    pid: int
    alive: bool
    status: PidStatus
    method: PidCheckMethod | None
    kill_zero_exit_code: int | None
    kill_zero_output: str
    proc_exit_code: int | None
    proc_output: str
    error: str | None
    latency_ms: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Internal: command builders
# ---------------------------------------------------------------------------


def _build_kill_zero_command(pid: int) -> str:
    """Build the kill -0 command string for the given PID.

    Redirects stderr to stdout so we can capture error messages
    like "No such process" or "Operation not permitted".
    """
    return f"kill -0 {pid} 2>&1"


def _build_proc_test_command(pid: int) -> str:
    """Build the /proc directory existence test for the given PID."""
    return f"test -d /proc/{pid}"


# ---------------------------------------------------------------------------
# Internal: result classification
# ---------------------------------------------------------------------------


def _classify_kill_zero(
    output: str,
    exit_code: int,
) -> PidStatus | None:
    """Classify the kill -0 result into a definitive PID status.

    Returns:
        PidStatus.ALIVE if exit code 0 or EPERM detected.
        PidStatus.DEAD if ESRCH ("No such process") detected.
        None if the result is ambiguous (unknown error message).
    """
    if exit_code == 0:
        return PidStatus.ALIVE

    output_lower = output.lower()

    if _EPERM_PATTERN in output_lower:
        return PidStatus.ALIVE

    if _ESRCH_PATTERN in output_lower:
        return PidStatus.DEAD

    # Ambiguous: non-zero exit with unrecognized error message
    return None


def _classify_proc_test(exit_code: int) -> PidStatus:
    """Classify the /proc test result into a PID status.

    Returns:
        PidStatus.ALIVE if exit code 0 (directory exists).
        PidStatus.DEAD if exit code non-zero (no directory).
    """
    if exit_code == 0:
        return PidStatus.ALIVE
    return PidStatus.DEAD


# ---------------------------------------------------------------------------
# Internal: safe command execution
# ---------------------------------------------------------------------------


async def _safe_execute(
    executor: ProbeExecutor,
    command: str,
    timeout: float,
) -> tuple[str, int] | Exception:
    """Execute a command via the probe executor with timeout safety.

    Wraps the executor call in asyncio.wait_for() as a safety net.

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
# Internal: helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Internal: result builders
# ---------------------------------------------------------------------------


def _build_definitive_result(
    *,
    pid: int,
    status: PidStatus,
    method: PidCheckMethod,
    kill_exit_code: int | None,
    kill_output: str,
    proc_exit_code: int | None,
    proc_output: str,
    latency_ms: float,
) -> PidLivenessResult:
    """Build a result for a definitive ALIVE or DEAD determination."""
    return PidLivenessResult(
        pid=pid,
        alive=status == PidStatus.ALIVE,
        status=status,
        method=method,
        kill_zero_exit_code=kill_exit_code,
        kill_zero_output=kill_output,
        proc_exit_code=proc_exit_code,
        proc_output=proc_output,
        error=None,
        latency_ms=latency_ms,
        timestamp=_now_utc(),
    )


def _build_unknown_result(
    *,
    pid: int,
    kill_exit_code: int | None,
    kill_output: str,
    proc_exit_code: int | None,
    proc_output: str,
    error: str,
    latency_ms: float,
) -> PidLivenessResult:
    """Build a result when liveness could not be determined."""
    return PidLivenessResult(
        pid=pid,
        alive=False,
        status=PidStatus.UNKNOWN,
        method=None,
        kill_zero_exit_code=kill_exit_code,
        kill_zero_output=kill_output,
        proc_exit_code=proc_exit_code,
        proc_output=proc_output,
        error=error,
        latency_ms=latency_ms,
        timestamp=_now_utc(),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def validate_pid_liveness(
    executor: ProbeExecutor,
    pid: int,
    config: PidCheckConfig | None = None,
) -> PidLivenessResult:
    """Check whether a remote process is still running.

    Executes ``kill -0 <PID>`` as the primary check. If the result is
    definitive (exit code 0, EPERM, or ESRCH), returns immediately.
    Otherwise falls back to ``test -d /proc/<PID>`` to determine
    liveness via the proc filesystem.

    If both checks fail (timeout or transport error), returns an
    UNKNOWN result with error details.

    Args:
        executor: ProbeExecutor implementation that runs commands on
            the remote host over an established SSH session.
        pid: The remote process ID to check. Must be a positive integer.
        config: Optional check configuration (timeout). When None,
            uses default PidCheckConfig (5s timeout).

    Returns:
        Immutable PidLivenessResult with alive/dead/unknown status and
        diagnostic details. Never raises -- all errors are captured in
        the result.

    Raises:
        ValueError: If pid is not a positive integer.
    """
    if pid <= 0:
        raise ValueError(
            f"PID must be a positive integer, got {pid}"
        )

    effective_config = config if config is not None else PidCheckConfig()
    timeout = effective_config.timeout_seconds
    start_ns = time.monotonic_ns()

    # Track results from each check
    kill_exit_code: int | None = None
    kill_output: str = ""
    kill_error: str | None = None

    proc_exit_code: int | None = None
    proc_output: str = ""
    proc_error: str | None = None

    # -- Step 1: kill -0 primary check --

    kill_command = _build_kill_zero_command(pid)
    kill_result = await _safe_execute(executor, kill_command, timeout)

    if isinstance(kill_result, Exception):
        # kill -0 failed at transport level -- record the error
        kill_error = f"{type(kill_result).__name__}: {kill_result}"
        logger.warning(
            "kill -0 check for PID %d failed: %s", pid, kill_error
        )
    else:
        kill_output, kill_exit_code = kill_result
        kill_status = _classify_kill_zero(kill_output, kill_exit_code)

        if kill_status is not None:
            # Definitive result from kill -0
            elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
            logger.info(
                "PID %d liveness via kill -0: %s (exit=%d, %.1fms)",
                pid,
                kill_status.value,
                kill_exit_code,
                elapsed_ms,
            )
            return _build_definitive_result(
                pid=pid,
                status=kill_status,
                method=PidCheckMethod.KILL_ZERO,
                kill_exit_code=kill_exit_code,
                kill_output=kill_output,
                proc_exit_code=None,
                proc_output="",
                latency_ms=elapsed_ms,
            )

        # Ambiguous kill -0 result -- fall through to /proc
        logger.info(
            "kill -0 for PID %d was ambiguous (exit=%d, output=%r) "
            "-- falling back to /proc",
            pid,
            kill_exit_code,
            kill_output,
        )

    # -- Step 2: /proc fallback --

    proc_command = _build_proc_test_command(pid)
    proc_result = await _safe_execute(executor, proc_command, timeout)

    if isinstance(proc_result, Exception):
        # /proc check also failed
        proc_error = f"{type(proc_result).__name__}: {proc_result}"
        logger.warning(
            "/proc check for PID %d failed: %s", pid, proc_error
        )

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        # Both checks failed -- UNKNOWN
        error_parts: list[str] = []
        if kill_error:
            error_parts.append(f"kill -0: {kill_error}")
        if proc_error:
            error_parts.append(f"/proc: {proc_error}")
        combined_error = "; ".join(error_parts) if error_parts else "Both checks failed"

        logger.warning(
            "PID %d liveness UNKNOWN: both checks failed (%.1fms)",
            pid,
            elapsed_ms,
        )
        return _build_unknown_result(
            pid=pid,
            kill_exit_code=kill_exit_code,
            kill_output=kill_output,
            proc_exit_code=None,
            proc_output="",
            error=combined_error,
            latency_ms=elapsed_ms,
        )

    proc_output, proc_exit_code = proc_result
    proc_status = _classify_proc_test(proc_exit_code)
    elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

    logger.info(
        "PID %d liveness via /proc fallback: %s (exit=%d, %.1fms)",
        pid,
        proc_status.value,
        proc_exit_code,
        elapsed_ms,
    )
    return _build_definitive_result(
        pid=pid,
        status=proc_status,
        method=PidCheckMethod.PROC_FALLBACK,
        kill_exit_code=kill_exit_code,
        kill_output=kill_output,
        proc_exit_code=proc_exit_code,
        proc_output=proc_output,
        latency_ms=elapsed_ms,
    )
