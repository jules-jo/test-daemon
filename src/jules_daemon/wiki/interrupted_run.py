"""Interrupted-run detection logic for daemon crash recovery.

Examines a parsed CurrentRun record to determine whether a prior run was
interrupted (record exists with status not marked complete, failed, cancelled,
or idle) and returns a structured RecoveryVerdict.

This module is a pure function layer -- it takes a CurrentRun (or None) and
returns a verdict. It does not read from disk or have side effects. The
caller (typically the daemon boot sequence) is responsible for loading the
record from the wiki first.

Terminal statuses (completed, failed, cancelled) and idle all indicate
no recovery is needed. Active statuses (running, pending_approval) indicate
the daemon stopped while a run was in progress and recovery is required.

Usage:
    from jules_daemon.wiki import current_run
    from jules_daemon.wiki.interrupted_run import detect_interrupted_run

    record = current_run.read(wiki_root)
    verdict = detect_interrupted_run(record)
    if verdict.recovery_needed:
        # handle crash recovery using verdict context
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from jules_daemon.wiki.models import CurrentRun, RunStatus


# Statuses that indicate a run reached a terminal state -- no recovery needed
_TERMINAL_STATUSES = frozenset({
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
})

# Statuses that indicate a run was in progress -- recovery needed
_ACTIVE_STATUSES = frozenset({
    RunStatus.RUNNING,
    RunStatus.PENDING_APPROVAL,
})


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _compute_stale_seconds(updated_at: datetime) -> float:
    """Compute seconds elapsed since the record was last updated.

    Args:
        updated_at: The record's last-update timestamp (must be tz-aware).

    Returns:
        Non-negative float representing elapsed seconds.
    """
    delta = _now_utc() - updated_at
    return max(0.0, delta.total_seconds())


@dataclass(frozen=True)
class RecoveryVerdict:
    """Structured result of interrupted-run detection.

    Provides all context the daemon needs to make recovery decisions:
    - Whether recovery is needed at all
    - Which status the interrupted run was in
    - How stale the record is (time since last update)
    - Whether a remote process might still be alive
    - The daemon PID that was managing the run (for liveness check)
    """

    recovery_needed: bool
    reason: str
    interrupted_status: RunStatus
    run_id: str
    stale_seconds: Optional[float]
    has_remote_process: bool
    daemon_pid: Optional[int]


def _build_no_record_verdict() -> RecoveryVerdict:
    """Build a verdict for the case where no prior record exists."""
    return RecoveryVerdict(
        recovery_needed=False,
        reason="No prior run record found",
        interrupted_status=RunStatus.IDLE,
        run_id="",
        stale_seconds=None,
        has_remote_process=False,
        daemon_pid=None,
    )


def _build_idle_verdict(run: CurrentRun) -> RecoveryVerdict:
    """Build a verdict for an idle record."""
    return RecoveryVerdict(
        recovery_needed=False,
        reason="Prior run is idle -- no recovery needed",
        interrupted_status=RunStatus.IDLE,
        run_id=run.run_id,
        stale_seconds=None,
        has_remote_process=False,
        daemon_pid=run.pids.daemon,
    )


def _build_terminal_verdict(run: CurrentRun) -> RecoveryVerdict:
    """Build a verdict for a terminal-status record."""
    status_label = run.status.value
    return RecoveryVerdict(
        recovery_needed=False,
        reason=f"Prior run already {status_label} -- no recovery needed",
        interrupted_status=run.status,
        run_id=run.run_id,
        stale_seconds=None,
        has_remote_process=False,
        daemon_pid=run.pids.daemon,
    )


def _build_interrupted_verdict(run: CurrentRun) -> RecoveryVerdict:
    """Build a verdict for an active (interrupted) record."""
    status_label = run.status.value
    stale_seconds = _compute_stale_seconds(run.updated_at)
    has_remote = run.pids.remote is not None

    return RecoveryVerdict(
        recovery_needed=True,
        reason=(
            f"Prior run was {status_label} when daemon stopped "
            f"-- recovery needed (stale {stale_seconds:.1f}s)"
        ),
        interrupted_status=run.status,
        run_id=run.run_id,
        stale_seconds=stale_seconds,
        has_remote_process=has_remote,
        daemon_pid=run.pids.daemon,
    )


def detect_interrupted_run(record: Optional[CurrentRun]) -> RecoveryVerdict:
    """Examine a parsed run record and determine if recovery is needed.

    This is the primary entry point for interrupted-run detection. It takes
    a CurrentRun (as loaded from the wiki) or None (if no wiki file exists)
    and returns a RecoveryVerdict indicating whether the daemon needs to
    perform crash recovery.

    Decision logic:
    - None (no record): no recovery needed
    - IDLE: no recovery needed
    - COMPLETED / FAILED / CANCELLED: no recovery needed (terminal states)
    - RUNNING / PENDING_APPROVAL: recovery needed (interrupted states)

    Args:
        record: The parsed CurrentRun from the wiki, or None if no file exists.

    Returns:
        RecoveryVerdict with recovery_needed flag and contextual information.
    """
    if record is None:
        return _build_no_record_verdict()

    if record.status == RunStatus.IDLE:
        return _build_idle_verdict(record)

    if record.status in _TERMINAL_STATUSES:
        return _build_terminal_verdict(record)

    if record.status in _ACTIVE_STATUSES:
        return _build_interrupted_verdict(record)

    # Defensive: unknown status -- treat as not needing recovery
    # but flag the anomaly in the reason
    return RecoveryVerdict(
        recovery_needed=False,
        reason=f"Unknown status '{record.status.value}' -- treating as non-interrupted",
        interrupted_status=record.status,
        run_id=record.run_id,
        stale_seconds=None,
        has_remote_process=False,
        daemon_pid=None,
    )
