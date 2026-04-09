"""Local process-state checker.

Probes the OS process table to determine whether local PIDs are alive or
dead. Uses ``os.kill(pid, 0)`` which sends signal 0 (a null signal) to
test process existence without actually delivering a signal.

Signal 0 semantics:
    - No error: Process exists and caller has permission to signal it.
    - PermissionError (EPERM): Process exists but caller lacks permission.
      This is still ALIVE -- the process is running under a different user.
    - ProcessLookupError (ESRCH): No process with the given PID exists.
      This means the process is DEAD.
    - Other OSError: Unexpected error, classified as ERROR.

Race condition note:
    A process can exit between the time we check it and the time the caller
    acts on the result. This is inherent to the POSIX process model and
    callers must handle it. Each call probes the OS fresh with no caching.

Usage:
    from jules_daemon.monitor.process_state import check_pid, check_pids

    # Single PID
    result = check_pid(1234)
    if result.verdict == ProcessVerdict.ALIVE:
        ...

    # Batch check
    results = check_pids([1234, 5678, 9999])
    for pid, result in results.items():
        print(f"PID {pid}: {result.verdict.value}")
"""

from __future__ import annotations

import errno
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping

__all__ = [
    "ProcessCheckResult",
    "ProcessVerdict",
    "check_pid",
    "check_pids",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProcessVerdict(Enum):
    """Verdict for a local process liveness check.

    Values:
        ALIVE: Process confirmed running in the OS process table.
        DEAD: Process confirmed not running (ESRCH / ProcessLookupError).
        ERROR: Could not determine status due to an unexpected error.
    """

    ALIVE = "alive"
    DEAD = "dead"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessCheckResult:
    """Immutable result of a local process-state check.

    Attributes:
        pid: The local process ID that was checked.
        verdict: ALIVE, DEAD, or ERROR classification.
        error: Human-readable error description. None when the process
            is definitively alive. Contains the OS error message when
            the process is dead or when an unexpected error occurred.
        latency_ms: Time taken for the check in milliseconds.
        timestamp: UTC datetime when the check completed.
    """

    pid: int
    verdict: ProcessVerdict
    error: str | None
    latency_ms: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _classify_os_error(exc: OSError) -> tuple[ProcessVerdict, str | None]:
    """Classify an OSError from os.kill into a verdict.

    Returns:
        Tuple of (verdict, error_message). error_message is None when
        the verdict is ALIVE (EPERM case).
    """
    if exc.errno == errno.EPERM:
        # Process exists but caller lacks permission -- still alive
        return (ProcessVerdict.ALIVE, None)

    if exc.errno == errno.ESRCH:
        # No such process -- dead
        return (ProcessVerdict.DEAD, str(exc))

    # Unexpected errno
    return (ProcessVerdict.ERROR, f"OSError(errno={exc.errno}): {exc}")


# ---------------------------------------------------------------------------
# Public API: single PID check
# ---------------------------------------------------------------------------


def check_pid(pid: int) -> ProcessCheckResult:
    """Check whether a single local process is alive.

    Probes the OS process table using ``os.kill(pid, 0)``. This does not
    send any actual signal -- signal 0 is a null signal used solely to
    test for process existence.

    Args:
        pid: The local process ID to check. Must be a positive integer.

    Returns:
        Immutable ProcessCheckResult with the verdict and diagnostics.

    Raises:
        ValueError: If pid is not a positive integer.
    """
    if pid <= 0:
        raise ValueError(f"PID must be a positive integer, got {pid}")

    start_ns = time.monotonic_ns()

    try:
        os.kill(pid, 0)
    except PermissionError:
        # EPERM: process exists but we lack permission to signal it
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        logger.debug("PID %d: alive (EPERM, %.2fms)", pid, elapsed_ms)
        return ProcessCheckResult(
            pid=pid,
            verdict=ProcessVerdict.ALIVE,
            error=None,
            latency_ms=elapsed_ms,
            timestamp=_now_utc(),
        )
    except ProcessLookupError as exc:
        # ESRCH: no such process
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        logger.debug("PID %d: dead (ESRCH, %.2fms)", pid, elapsed_ms)
        return ProcessCheckResult(
            pid=pid,
            verdict=ProcessVerdict.DEAD,
            error=str(exc),
            latency_ms=elapsed_ms,
            timestamp=_now_utc(),
        )
    except OSError as exc:
        # Other OS error -- classify by errno
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        verdict, error_msg = _classify_os_error(exc)
        logger.debug(
            "PID %d: %s (OSError errno=%s, %.2fms)",
            pid,
            verdict.value,
            exc.errno,
            elapsed_ms,
        )
        return ProcessCheckResult(
            pid=pid,
            verdict=verdict,
            error=error_msg,
            latency_ms=elapsed_ms,
            timestamp=_now_utc(),
        )
    except Exception as exc:
        # Completely unexpected exception
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "PID %d: error (%s, %.2fms)", pid, error_msg, elapsed_ms
        )
        return ProcessCheckResult(
            pid=pid,
            verdict=ProcessVerdict.ERROR,
            error=error_msg,
            latency_ms=elapsed_ms,
            timestamp=_now_utc(),
        )

    # os.kill succeeded -- process is alive and signalable
    elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
    logger.debug("PID %d: alive (signal 0 ok, %.2fms)", pid, elapsed_ms)
    return ProcessCheckResult(
        pid=pid,
        verdict=ProcessVerdict.ALIVE,
        error=None,
        latency_ms=elapsed_ms,
        timestamp=_now_utc(),
    )


# ---------------------------------------------------------------------------
# Public API: batch PID check
# ---------------------------------------------------------------------------


def check_pids(pids: list[int]) -> Mapping[int, ProcessCheckResult]:
    """Check whether multiple local processes are alive.

    Deduplicates input PIDs and checks each once. Validates all PIDs
    before performing any checks (fail-fast on invalid input).

    Args:
        pids: List of local process IDs to check. Each must be a
            positive integer.

    Returns:
        Read-only mapping from PID to its ProcessCheckResult. The
        mapping is a ``types.MappingProxyType`` to prevent mutation.

    Raises:
        ValueError: If any PID is not a positive integer.
    """
    # Validate all PIDs upfront (fail-fast)
    for pid in pids:
        if pid <= 0:
            raise ValueError(f"PID must be a positive integer, got {pid}")

    # Deduplicate while preserving order
    unique_pids = dict.fromkeys(pids)

    results: dict[int, ProcessCheckResult] = {}
    for pid in unique_pids:
        results[pid] = check_pid(pid)

    return MappingProxyType(results)
