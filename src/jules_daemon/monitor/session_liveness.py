"""Per-session liveness aggregator.

Combines the local process-state verdict and SSH-connection verdict for a
given session into a unified LivenessResult. Exposes a single async entry
point ``check_session_liveness()`` that orchestrates both probes and
returns the composite verdict.

Health classification matrix:

    Process   | SSH          | Composite
    ----------|--------------|--------------------
    ALIVE     | CONNECTED    | HEALTHY
    ALIVE     | DEGRADED     | DEGRADED
    ALIVE     | DISCONNECTED | CONNECTION_LOST
    ALIVE     | (no exec)    | DEGRADED (partial)
    DEAD      | (any/skip)   | PROCESS_DEAD
    ERROR     | CONNECTED    | UNKNOWN
    ERROR     | DEGRADED     | UNKNOWN
    ERROR     | DISCONNECTED | UNKNOWN

Design choices:
    - When the local daemon process is confirmed DEAD, the SSH probe is
      skipped entirely. There is no value in checking remote connectivity
      when the local process that owns the session is already gone.
    - When no executor is provided (executor=None), only the process check
      runs. The composite verdict is DEGRADED if the process is alive
      (we cannot confirm full health without SSH), or PROCESS_DEAD if it
      is dead.
    - The ``alive`` field is True only for HEALTHY and DEGRADED states.
      CONNECTION_LOST, PROCESS_DEAD, and UNKNOWN are all considered
      non-alive because the session cannot make forward progress.

Usage:
    from jules_daemon.monitor.session_liveness import (
        SessionInfo,
        check_session_liveness,
    )

    session = SessionInfo(session_id="run-abc", daemon_pid=1234)
    result = await check_session_liveness(session, executor)
    if result.alive:
        # Session is operational
        ...
    elif result.health == SessionHealth.CONNECTION_LOST:
        # Trigger SSH reconnection
        ...
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from jules_daemon.monitor.process_state import (
    ProcessCheckResult,
    ProcessVerdict,
    check_pid,
)
from jules_daemon.ssh.liveness import (
    ConnectionHealth,
    ProbeConfig,
    ProbeExecutor,
    ProbeResult,
    validate_liveness,
)

__all__ = [
    "LivenessResult",
    "SessionHealth",
    "SessionInfo",
    "check_session_liveness",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SessionHealth(Enum):
    """Composite health classification for a session.

    Derived from combining the local process-state verdict and the
    SSH-connection verdict.

    Values:
        HEALTHY: Both process and SSH connection are operational.
        DEGRADED: Process is alive but SSH is degraded or unchecked.
        CONNECTION_LOST: Process is alive but SSH is disconnected.
        PROCESS_DEAD: Local daemon process is confirmed dead.
        UNKNOWN: Could not determine health (process check errored).
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CONNECTION_LOST = "connection_lost"
    PROCESS_DEAD = "process_dead"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionInfo:
    """Immutable input describing which session to check.

    Attributes:
        session_id: Unique identifier for the session/run (e.g., a UUID).
        daemon_pid: Local PID of the daemon process that owns this session.
    """

    session_id: str
    daemon_pid: int

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if self.daemon_pid <= 0:
            raise ValueError(
                f"daemon_pid must be a positive integer, got {self.daemon_pid}"
            )


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LivenessResult:
    """Immutable composite result of a session liveness check.

    Combines the local process-state check and the SSH-connection check
    into a single verdict with full diagnostic details.

    Attributes:
        session_id: Identifier of the session that was checked.
        health: Composite health classification.
        alive: True when the session is considered operational (HEALTHY
            or DEGRADED). False for CONNECTION_LOST, PROCESS_DEAD, and
            UNKNOWN.
        process_result: Result of the local process-state check. None
            only if the check could not be performed at all.
        ssh_result: Result of the SSH-connection check. None when the
            SSH check was skipped (process dead or no executor).
        errors: Tuple of human-readable error descriptions collected
            during the check. Empty tuple when everything passed.
        latency_ms: Total wall-clock time for all probes in milliseconds.
        timestamp: UTC datetime when the composite check completed.
    """

    session_id: str
    health: SessionHealth
    alive: bool
    process_result: ProcessCheckResult | None
    ssh_result: ProbeResult | None
    errors: tuple[str, ...]
    latency_ms: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _derive_health(
    process_verdict: ProcessVerdict,
    ssh_health: ConnectionHealth | None,
) -> tuple[SessionHealth, bool]:
    """Derive composite health and alive flag from sub-verdicts.

    Args:
        process_verdict: Verdict from the local process check.
        ssh_health: Health from the SSH connection check. None when the
            SSH check was skipped.

    Returns:
        Tuple of (SessionHealth, alive_flag).
    """
    if process_verdict == ProcessVerdict.DEAD:
        return (SessionHealth.PROCESS_DEAD, False)

    if process_verdict == ProcessVerdict.ERROR:
        return (SessionHealth.UNKNOWN, False)

    # Process is ALIVE -- classify based on SSH health
    if ssh_health is None:
        # No SSH check performed (no executor)
        return (SessionHealth.DEGRADED, True)

    if ssh_health == ConnectionHealth.CONNECTED:
        return (SessionHealth.HEALTHY, True)

    if ssh_health == ConnectionHealth.DEGRADED:
        return (SessionHealth.DEGRADED, True)

    # ssh_health == ConnectionHealth.DISCONNECTED
    return (SessionHealth.CONNECTION_LOST, False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def check_session_liveness(
    session: SessionInfo,
    executor: ProbeExecutor | None,
    *,
    probe_config: ProbeConfig | None = None,
) -> LivenessResult:
    """Check the liveness of a session by probing both local process and SSH.

    Orchestrates two probes in sequence:

    1. **Local process check**: Uses ``os.kill(pid, 0)`` via ``check_pid()``
       to determine whether the daemon process is still alive.

    2. **SSH connection check** (conditional): Uses ``validate_liveness()``
       to send a lightweight probe command over the SSH session. This check
       is skipped when the local process is confirmed dead (no point
       checking SSH for a dead session) or when no executor is provided.

    The two sub-verdicts are combined into a composite SessionHealth via
    the classification matrix documented in the module docstring.

    Args:
        session: Session identification with the daemon PID to check.
        executor: ProbeExecutor for the SSH liveness check. When None,
            only the local process check runs and the SSH check is skipped.
        probe_config: Optional ProbeConfig to customize the SSH probe
            command and timeout. When None, uses the default probe config.

    Returns:
        Immutable LivenessResult with the composite verdict, both
        sub-results, and diagnostic metadata. Never raises -- all errors
        are captured in the result.
    """
    start_ns = time.monotonic_ns()
    errors: list[str] = []

    # -- Step 1: Local process check --

    process_result = check_pid(session.daemon_pid)

    if process_result.verdict == ProcessVerdict.DEAD:
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        logger.info(
            "Session %s: daemon PID %d is dead, skipping SSH check (%.1fms)",
            session.session_id,
            session.daemon_pid,
            elapsed_ms,
        )
        return LivenessResult(
            session_id=session.session_id,
            health=SessionHealth.PROCESS_DEAD,
            alive=False,
            process_result=process_result,
            ssh_result=None,
            errors=(),
            latency_ms=elapsed_ms,
            timestamp=_now_utc(),
        )

    if process_result.verdict == ProcessVerdict.ERROR:
        errors.append(
            f"Process check error for PID {session.daemon_pid}: "
            f"{process_result.error}"
        )

    # -- Step 2: SSH connection check (if executor provided) --

    ssh_result: ProbeResult | None = None

    if executor is not None:
        ssh_result = await validate_liveness(executor, probe_config)

        if ssh_result.error is not None:
            errors.append(f"SSH probe error: {ssh_result.error}")
    else:
        errors.append("SSH check skipped: no executor provided")

    # -- Step 3: Derive composite verdict --

    ssh_health = ssh_result.health if ssh_result is not None else None
    health, alive = _derive_health(process_result.verdict, ssh_health)

    elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

    logger.info(
        "Session %s liveness: health=%s, alive=%s, "
        "process=%s, ssh=%s (%.1fms)",
        session.session_id,
        health.value,
        alive,
        process_result.verdict.value,
        ssh_health.value if ssh_health else "skipped",
        elapsed_ms,
    )

    return LivenessResult(
        session_id=session.session_id,
        health=health,
        alive=alive,
        process_result=process_result,
        ssh_result=ssh_result,
        errors=tuple(errors),
        latency_ms=elapsed_ms,
        timestamp=_now_utc(),
    )
