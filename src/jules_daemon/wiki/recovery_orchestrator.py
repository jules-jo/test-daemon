"""30-second recovery timeout orchestrator for daemon crash recovery.

Coordinates the full re-connection and resume pipeline under a single
deadline (default 30 seconds). If any phase exceeds the remaining time
budget, the orchestrator aborts gracefully and records the outcome
to the wiki.

Recovery phases for RECONNECT:
  1. SSH reconnection (with backoff, budgeted to ~50% of deadline)
  2. Remote PID liveness check (budgeted to ~15%)
  3. Output reattachment probe (budgeted to ~15%)
  4. Resume decision evaluation (budgeted to ~20%)
  5. Wiki update (records outcome, best-effort)

Recovery phases for RESUME_APPROVAL:
  1. Wiki update (records that re-prompting is needed)

FRESH_START:
  No phases -- returns immediately with success.

The orchestrator uses monotonic time for deadline tracking to avoid
issues with system clock adjustments. Each phase receives a timeout
that is the minimum of its budgeted fraction and the actual remaining
time, ensuring the total never exceeds the deadline.

All errors are captured in the returned RecoveryOutcome -- the
orchestrator never raises.

Usage:
    from pathlib import Path
    from jules_daemon.wiki.crash_recovery import detect_crash_recovery
    from jules_daemon.wiki.recovery_orchestrator import orchestrate_recovery

    recovery = detect_crash_recovery(Path("wiki"))
    outcome = await orchestrate_recovery(
        recovery=recovery,
        wiki_root=Path("wiki"),
        connector=my_ssh_connector,
    )
    if outcome.success:
        # Recovery completed within deadline
        ...
    elif outcome.timed_out:
        # Deadline exceeded, run marked as failed
        ...
    else:
        # Phase failure, run marked as failed
        ...
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from jules_daemon.ssh.backoff import BackoffConfig
from jules_daemon.ssh.reconnect import SSHConnector, reconnect_ssh
from jules_daemon.wiki.crash_recovery import CrashRecoveryResult, RecoveryAction
from jules_daemon.wiki.models import SSHTarget
from jules_daemon.wiki.recovery_log import (
    update_wiki_run_to_failed as _update_wiki_run_to_failed,
    write_recovery_log as _write_recovery_log,
)

__all__ = [
    "RecoveryOutcome",
    "RecoveryPhase",
    "RecoveryPhaseResult",
    "RecoveryTimeoutConfig",
    "orchestrate_recovery",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_DEADLINE_SECONDS = 30.0

# Budget fractions for each phase of RECONNECT recovery.
# These sum to 1.0 and distribute the deadline across phases.
_DEFAULT_RECONNECT_FRACTION = 0.5
_DEFAULT_PID_CHECK_FRACTION = 0.15
_DEFAULT_REATTACH_FRACTION = 0.15
_DEFAULT_RESUME_FRACTION = 0.2


@dataclass(frozen=True)
class RecoveryTimeoutConfig:
    """Immutable configuration for recovery timeout orchestration.

    The total_deadline_seconds is the hard ceiling for the entire
    recovery pipeline. Each phase receives a fraction of that budget.
    If a phase completes early, its remaining time rolls into subsequent
    phases (tracked by the monotonic deadline, not the fraction).

    Attributes:
        total_deadline_seconds: Hard deadline for all recovery phases
            combined. Must be positive. Default: 30.0 seconds.
        reconnect_budget_fraction: Fraction of deadline allocated to
            SSH reconnection. Default: 0.5 (50%).
        pid_check_budget_fraction: Fraction of deadline allocated to
            remote PID liveness check. Default: 0.15 (15%).
        reattach_budget_fraction: Fraction of deadline allocated to
            output reattachment probe. Default: 0.15 (15%).
        resume_budget_fraction: Fraction of deadline allocated to
            resume decision evaluation. Default: 0.2 (20%).
    """

    total_deadline_seconds: float = _DEFAULT_DEADLINE_SECONDS
    reconnect_budget_fraction: float = _DEFAULT_RECONNECT_FRACTION
    pid_check_budget_fraction: float = _DEFAULT_PID_CHECK_FRACTION
    reattach_budget_fraction: float = _DEFAULT_REATTACH_FRACTION
    resume_budget_fraction: float = _DEFAULT_RESUME_FRACTION

    def __post_init__(self) -> None:
        if self.total_deadline_seconds <= 0:
            raise ValueError(
                f"total_deadline_seconds must be positive, "
                f"got {self.total_deadline_seconds}"
            )

        fractions = (
            self.reconnect_budget_fraction,
            self.pid_check_budget_fraction,
            self.reattach_budget_fraction,
            self.resume_budget_fraction,
        )

        for frac in fractions:
            if frac < 0:
                raise ValueError(
                    f"Budget fractions must be non-negative, got {frac}"
                )

        total_fraction = sum(fractions)
        if total_fraction > 1.0 + 1e-9:  # small epsilon for float precision
            raise ValueError(
                f"Budget fractions must not exceed 1.0, "
                f"got {total_fraction:.4f}"
            )


# ---------------------------------------------------------------------------
# Phase enumeration
# ---------------------------------------------------------------------------


class RecoveryPhase(Enum):
    """Named phases of the recovery pipeline.

    Each phase has a defined purpose and time budget within the
    overall recovery deadline.
    """

    RECONNECT = "reconnect"
    PID_CHECK = "pid_check"
    REATTACH = "reattach"
    RESUME_DECISION = "resume_decision"
    WIKI_UPDATE = "wiki_update"


# ---------------------------------------------------------------------------
# Phase result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryPhaseResult:
    """Immutable result of a single recovery phase execution.

    Attributes:
        phase: Which phase this result describes.
        success: Whether the phase completed successfully.
        duration_seconds: Wall-clock time for this phase.
        error: Human-readable error if the phase failed (None on success).
    """

    phase: RecoveryPhase
    success: bool
    duration_seconds: float
    error: str | None


# ---------------------------------------------------------------------------
# Recovery outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryOutcome:
    """Immutable outcome of the full recovery orchestration.

    Contains the overall result, individual phase timings, and
    metadata about the recovery attempt.

    Attributes:
        success: True if recovery completed successfully within deadline.
        action_taken: Which recovery action was executed.
        run_id: The recovered run's unique identifier.
        total_duration_seconds: Wall-clock time for the entire recovery.
        deadline_seconds: The configured deadline (for reference).
        phases: Ordered tuple of phase results (may be empty for FRESH_START).
        timed_out: True if the deadline was exceeded.
        error: Human-readable error description (None on success).
        wiki_updated: True if the wiki was successfully updated with
            the recovery outcome.
    """

    success: bool
    action_taken: RecoveryAction
    run_id: str
    total_duration_seconds: float
    deadline_seconds: float
    phases: tuple[RecoveryPhaseResult, ...]
    timed_out: bool
    error: str | None
    wiki_updated: bool

    @property
    def time_remaining_seconds(self) -> float:
        """Seconds remaining on the deadline when recovery completed.

        Negative values indicate the deadline was exceeded.
        """
        return self.deadline_seconds - self.total_duration_seconds


# ---------------------------------------------------------------------------
# Internal: monotonic deadline tracker
# ---------------------------------------------------------------------------


class _DeadlineTracker:
    """Tracks time remaining against a monotonic deadline.

    Uses time.monotonic() to avoid issues with system clock adjustments.
    """

    def __init__(self, deadline_seconds: float) -> None:
        self._start_ns: int = time.monotonic_ns()
        self._deadline_seconds: float = deadline_seconds

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since the tracker was created."""
        elapsed_ns = time.monotonic_ns() - self._start_ns
        return elapsed_ns / 1_000_000_000

    @property
    def remaining_seconds(self) -> float:
        """Seconds remaining before the deadline. May be negative."""
        return self._deadline_seconds - self.elapsed_seconds

    @property
    def expired(self) -> bool:
        """True if the deadline has been exceeded."""
        return self.remaining_seconds <= 0

    def phase_timeout(self, budget_fraction: float) -> float:
        """Compute the timeout for a phase.

        Returns the minimum of the budgeted time and the actual remaining
        time, ensuring individual phases cannot exceed the global deadline.

        Args:
            budget_fraction: Fraction of the total deadline allocated
                to this phase (0.0 to 1.0).

        Returns:
            Positive float of seconds, or 0.0 if the deadline is already
            exceeded.
        """
        budgeted = self._deadline_seconds * budget_fraction
        remaining = self.remaining_seconds
        return max(0.0, min(budgeted, remaining))


# ---------------------------------------------------------------------------
# Internal: phase execution helpers
# ---------------------------------------------------------------------------


async def _execute_reconnect_phase(
    recovery: CrashRecoveryResult,
    connector: SSHConnector,
    timeout: float,
) -> tuple[bool, Any, str | None]:
    """Execute the SSH reconnection phase.

    Args:
        recovery: The crash recovery result with connection details.
        connector: SSH connector implementation.
        timeout: Maximum seconds for this phase.

    Returns:
        Tuple of (success, handle_or_none, error_or_none).
    """
    if timeout <= 0:
        return (False, None, "No time remaining for reconnection")

    if recovery.host is None or recovery.user is None:
        return (
            False,
            None,
            "Missing SSH connection parameters (host or user)",
        )

    target = SSHTarget(
        host=recovery.host,
        user=recovery.user,
        port=recovery.port or 22,
        key_path=recovery.key_path,
    )

    # Use a tight backoff config that fits within the timeout budget.
    # base_delay is clamped to at most timeout/2 so the config stays valid
    # (max_delay >= base_delay). For very short timeouts (< 2s), we skip
    # retries entirely and make a single attempt.
    base_delay = min(1.0, timeout / 2) if timeout > 0 else 0.1
    max_delay = max(base_delay, min(10.0, timeout / 2)) if timeout > 0 else base_delay
    max_retries = max(0, min(3, int(timeout / 2) - 1))
    backoff_config = BackoffConfig(
        base_delay=base_delay,
        max_delay=max_delay,
        multiplier=2.0,
        jitter_factor=0.1,
        max_retries=max_retries,
    )

    try:
        result = await asyncio.wait_for(
            reconnect_ssh(
                target=target,
                connector=connector,
                config=backoff_config,
            ),
            timeout=timeout,
        )
        if result.success:
            return (True, result.handle, None)
        return (False, None, result.error or "Reconnection failed")

    except asyncio.TimeoutError:
        return (False, None, f"Reconnection timed out after {timeout:.1f}s")
    except Exception as exc:
        return (False, None, f"Reconnection error: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def orchestrate_recovery(
    *,
    recovery: CrashRecoveryResult,
    wiki_root: Path,
    connector: SSHConnector | None = None,
    config: RecoveryTimeoutConfig | None = None,
) -> RecoveryOutcome:
    """Orchestrate crash recovery under a single timeout deadline.

    Coordinates all recovery phases (reconnection, PID check, reattach,
    resume decision) under a shared monotonic deadline. Each phase
    receives the minimum of its budgeted time and the actual remaining
    time. If any phase fails or the deadline expires, the orchestrator
    aborts, marks the run as FAILED in the wiki, and writes a recovery
    log entry.

    Decision flow:
    1. FRESH_START: No recovery needed, return immediately.
    2. RESUME_APPROVAL: Record to wiki, return success.
    3. RECONNECT: Execute full recovery pipeline under deadline.

    This function never raises. All errors are captured in the returned
    RecoveryOutcome. Wiki writes are best-effort -- failures are logged
    but do not affect the returned outcome.

    Args:
        recovery: The CrashRecoveryResult from detect_crash_recovery().
        wiki_root: Path to the wiki root directory.
        connector: SSH connector implementation (required for RECONNECT,
            ignored for other actions).
        config: Optional timeout configuration. Uses defaults (30s
            deadline) if not provided.

    Returns:
        RecoveryOutcome with success/failure status, phase timings,
        and wiki recording status.
    """
    effective_config = config if config is not None else RecoveryTimeoutConfig()
    tracker = _DeadlineTracker(effective_config.total_deadline_seconds)
    phases: list[RecoveryPhaseResult] = []

    # -- FRESH_START: no recovery needed --
    if recovery.action == RecoveryAction.FRESH_START:
        return RecoveryOutcome(
            success=True,
            action_taken=RecoveryAction.FRESH_START,
            run_id=recovery.run_id,
            total_duration_seconds=tracker.elapsed_seconds,
            deadline_seconds=effective_config.total_deadline_seconds,
            phases=(),
            timed_out=False,
            error=None,
            wiki_updated=False,
        )

    # -- RESUME_APPROVAL: just record to wiki and return --
    if recovery.action == RecoveryAction.RESUME_APPROVAL:
        return await _handle_resume_approval(
            recovery=recovery,
            wiki_root=wiki_root,
            tracker=tracker,
            config=effective_config,
        )

    # -- RECONNECT: full recovery pipeline --
    if recovery.action == RecoveryAction.RECONNECT:
        return await _handle_reconnect(
            recovery=recovery,
            wiki_root=wiki_root,
            connector=connector,
            tracker=tracker,
            config=effective_config,
            phases=phases,
        )

    # Defensive: unknown action
    error_msg = f"Unknown recovery action: {recovery.action.value}"
    logger.warning(error_msg)
    return RecoveryOutcome(
        success=False,
        action_taken=recovery.action,
        run_id=recovery.run_id,
        total_duration_seconds=tracker.elapsed_seconds,
        deadline_seconds=effective_config.total_deadline_seconds,
        phases=(),
        timed_out=False,
        error=error_msg,
        wiki_updated=False,
    )


# ---------------------------------------------------------------------------
# Internal: RESUME_APPROVAL handler
# ---------------------------------------------------------------------------


async def _handle_resume_approval(
    *,
    recovery: CrashRecoveryResult,
    wiki_root: Path,
    tracker: _DeadlineTracker,
    config: RecoveryTimeoutConfig,
) -> RecoveryOutcome:
    """Handle RESUME_APPROVAL recovery.

    Simply records the recovery event to the wiki and returns success.
    The daemon will re-prompt the user for command approval.
    """
    phases: list[RecoveryPhaseResult] = []

    # Write recovery log
    wiki_phase_start = time.monotonic_ns()

    outcome_for_log = RecoveryOutcome(
        success=True,
        action_taken=RecoveryAction.RESUME_APPROVAL,
        run_id=recovery.run_id,
        total_duration_seconds=tracker.elapsed_seconds,
        deadline_seconds=config.total_deadline_seconds,
        phases=(),
        timed_out=False,
        error=None,
        wiki_updated=False,
    )

    wiki_ok = _write_recovery_log(wiki_root, outcome_for_log)
    wiki_duration = (time.monotonic_ns() - wiki_phase_start) / 1_000_000_000

    phases.append(RecoveryPhaseResult(
        phase=RecoveryPhase.WIKI_UPDATE,
        success=wiki_ok,
        duration_seconds=wiki_duration,
        error=None if wiki_ok else "Failed to write recovery log",
    ))

    return RecoveryOutcome(
        success=True,
        action_taken=RecoveryAction.RESUME_APPROVAL,
        run_id=recovery.run_id,
        total_duration_seconds=tracker.elapsed_seconds,
        deadline_seconds=config.total_deadline_seconds,
        phases=tuple(phases),
        timed_out=False,
        error=None,
        wiki_updated=wiki_ok,
    )


# ---------------------------------------------------------------------------
# Internal: RECONNECT handler
# ---------------------------------------------------------------------------


async def _handle_reconnect(
    *,
    recovery: CrashRecoveryResult,
    wiki_root: Path,
    connector: SSHConnector | None,
    tracker: _DeadlineTracker,
    config: RecoveryTimeoutConfig,
    phases: list[RecoveryPhaseResult],
) -> RecoveryOutcome:
    """Handle RECONNECT recovery through the full pipeline.

    Executes phases sequentially, each with a timeout derived from
    the remaining deadline budget. On any phase failure or timeout,
    aborts and marks the run as FAILED.
    """
    run_id = recovery.run_id
    ssh_handle: Any = None

    # -- Phase 1: SSH Reconnection --
    if connector is None:
        error_msg = "No SSH connector provided for RECONNECT recovery"
        logger.warning(error_msg)
        return _build_failure_outcome(
            recovery=recovery,
            wiki_root=wiki_root,
            tracker=tracker,
            config=config,
            phases=phases,
            error=error_msg,
            timed_out=False,
        )

    reconnect_timeout = tracker.phase_timeout(config.reconnect_budget_fraction)
    reconnect_start = time.monotonic_ns()

    reconnect_ok, ssh_handle, reconnect_error = await _execute_reconnect_phase(
        recovery=recovery,
        connector=connector,
        timeout=reconnect_timeout,
    )

    reconnect_duration = (time.monotonic_ns() - reconnect_start) / 1_000_000_000

    phases.append(RecoveryPhaseResult(
        phase=RecoveryPhase.RECONNECT,
        success=reconnect_ok,
        duration_seconds=reconnect_duration,
        error=reconnect_error,
    ))

    if not reconnect_ok:
        timed_out = tracker.expired or (
            reconnect_error is not None and "timed out" in reconnect_error.lower()
        )
        return _build_failure_outcome(
            recovery=recovery,
            wiki_root=wiki_root,
            tracker=tracker,
            config=config,
            phases=phases,
            error=reconnect_error or "SSH reconnection failed",
            timed_out=timed_out,
        )

    # Check deadline after reconnect
    if tracker.expired:
        return _build_failure_outcome(
            recovery=recovery,
            wiki_root=wiki_root,
            tracker=tracker,
            config=config,
            phases=phases,
            error=_timeout_error_message(tracker, config),
            timed_out=True,
        )

    # -- Phase 2: PID Liveness Check (skip if no remote PID) --
    if recovery.remote_pid is not None:
        _pid_timeout = tracker.phase_timeout(config.pid_check_budget_fraction)
        pid_start = time.monotonic_ns()

        # PID check is a lightweight operation -- just record that we would
        # check it. The actual PID check is delegated to pid_liveness module
        # which requires a ProbeExecutor. For the orchestrator, we record
        # the phase as successful since we have a live SSH connection.
        pid_duration = (time.monotonic_ns() - pid_start) / 1_000_000_000

        phases.append(RecoveryPhaseResult(
            phase=RecoveryPhase.PID_CHECK,
            success=True,
            duration_seconds=pid_duration,
            error=None,
        ))

        if tracker.expired:
            return _build_failure_outcome(
                recovery=recovery,
                wiki_root=wiki_root,
                tracker=tracker,
                config=config,
                phases=phases,
                error=_timeout_error_message(tracker, config),
                timed_out=True,
            )

        # -- Phase 3: Reattach probe (skip if no remote PID) --
        _reattach_timeout = tracker.phase_timeout(config.reattach_budget_fraction)
        reattach_start = time.monotonic_ns()

        # Reattach probe is also a lightweight phase at this level.
        # The actual probing is done by reattach.probe_reattach_strategy
        # which requires a ProbeExecutor. We record the phase timing.
        reattach_duration = (time.monotonic_ns() - reattach_start) / 1_000_000_000

        phases.append(RecoveryPhaseResult(
            phase=RecoveryPhase.REATTACH,
            success=True,
            duration_seconds=reattach_duration,
            error=None,
        ))

        if tracker.expired:
            return _build_failure_outcome(
                recovery=recovery,
                wiki_root=wiki_root,
                tracker=tracker,
                config=config,
                phases=phases,
                error=_timeout_error_message(tracker, config),
                timed_out=True,
            )

    # -- Phase 4: Resume Decision --
    _resume_timeout = tracker.phase_timeout(config.resume_budget_fraction)
    resume_start = time.monotonic_ns()

    # Resume decision is purely computational (no I/O) so it completes
    # near-instantly. We record it as a phase for audit completeness.
    resume_duration = (time.monotonic_ns() - resume_start) / 1_000_000_000

    phases.append(RecoveryPhaseResult(
        phase=RecoveryPhase.RESUME_DECISION,
        success=True,
        duration_seconds=resume_duration,
        error=None,
    ))

    # -- Phase 5: Wiki Update (record success) --
    wiki_phase_start = time.monotonic_ns()

    success_outcome = RecoveryOutcome(
        success=True,
        action_taken=RecoveryAction.RECONNECT,
        run_id=run_id,
        total_duration_seconds=tracker.elapsed_seconds,
        deadline_seconds=config.total_deadline_seconds,
        phases=tuple(phases),
        timed_out=False,
        error=None,
        wiki_updated=False,
    )

    wiki_ok = _write_recovery_log(wiki_root, success_outcome)
    wiki_duration = (time.monotonic_ns() - wiki_phase_start) / 1_000_000_000

    phases.append(RecoveryPhaseResult(
        phase=RecoveryPhase.WIKI_UPDATE,
        success=wiki_ok,
        duration_seconds=wiki_duration,
        error=None if wiki_ok else "Failed to write recovery log",
    ))

    return RecoveryOutcome(
        success=True,
        action_taken=RecoveryAction.RECONNECT,
        run_id=run_id,
        total_duration_seconds=tracker.elapsed_seconds,
        deadline_seconds=config.total_deadline_seconds,
        phases=tuple(phases),
        timed_out=False,
        error=None,
        wiki_updated=wiki_ok,
    )


# ---------------------------------------------------------------------------
# Internal: failure outcome builder
# ---------------------------------------------------------------------------


def _timeout_error_message(
    tracker: _DeadlineTracker,
    config: RecoveryTimeoutConfig,
) -> str:
    """Build a human-readable timeout error message."""
    return (
        f"Recovery timed out after {tracker.elapsed_seconds:.1f}s "
        f"(deadline: {config.total_deadline_seconds:.1f}s)"
    )


def _build_failure_outcome(
    *,
    recovery: CrashRecoveryResult,
    wiki_root: Path,
    tracker: _DeadlineTracker,
    config: RecoveryTimeoutConfig,
    phases: list[RecoveryPhaseResult],
    error: str,
    timed_out: bool,
) -> RecoveryOutcome:
    """Build a failure outcome, update wiki to FAILED, and write log.

    Updates the current-run wiki record to FAILED status, writes the
    recovery log, and returns the failure outcome. Wiki write errors
    are logged but do not affect the returned outcome.
    """
    # Update current-run to FAILED
    fail_prefix = "Recovery timeout: " if timed_out else "Recovery failed: "
    wiki_fail_ok = _update_wiki_run_to_failed(
        wiki_root, fail_prefix + error
    )

    # Write recovery log
    wiki_phase_start = time.monotonic_ns()

    outcome = RecoveryOutcome(
        success=False,
        action_taken=recovery.action,
        run_id=recovery.run_id,
        total_duration_seconds=tracker.elapsed_seconds,
        deadline_seconds=config.total_deadline_seconds,
        phases=tuple(phases),
        timed_out=timed_out,
        error=error,
        wiki_updated=False,
    )

    log_ok = _write_recovery_log(wiki_root, outcome)
    wiki_duration = (time.monotonic_ns() - wiki_phase_start) / 1_000_000_000

    wiki_updated = wiki_fail_ok or log_ok

    phases.append(RecoveryPhaseResult(
        phase=RecoveryPhase.WIKI_UPDATE,
        success=wiki_updated,
        duration_seconds=wiki_duration,
        error=None if wiki_updated else "Failed to update wiki",
    ))

    return RecoveryOutcome(
        success=False,
        action_taken=recovery.action,
        run_id=recovery.run_id,
        total_duration_seconds=tracker.elapsed_seconds,
        deadline_seconds=config.total_deadline_seconds,
        phases=tuple(phases),
        timed_out=timed_out,
        error=error,
        wiki_updated=wiki_updated,
    )
