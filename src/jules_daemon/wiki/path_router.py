"""Fresh-start vs recovery path router for daemon boot.

Takes a RecoveryVerdict (from interrupted_run.detect_interrupted_run) and
routes the daemon to either a fresh-start path or a recovery path. Each
path updates the wiki record accordingly and returns a structured
BootDecision.

Fresh-start path:
  Writes a clean idle CurrentRun to the wiki. Used when no prior run
  was interrupted (no record, idle, or terminal status).

Recovery path:
  Preserves the interrupted run's context (SSH target, command, progress)
  but updates the daemon PID to the new daemon's PID and refreshes the
  updated_at timestamp. Used when a prior run was in RUNNING or
  PENDING_APPROVAL state.

Usage:
    from jules_daemon.wiki import current_run
    from jules_daemon.wiki.interrupted_run import detect_interrupted_run
    from jules_daemon.wiki.path_router import route_boot

    record = current_run.read(wiki_root)
    verdict = detect_interrupted_run(record)
    decision = route_boot(verdict, record, wiki_root, daemon_pid=os.getpid())

    if decision.is_fresh_start:
        # daemon is ready for new commands
        ...
    elif decision.is_recovery:
        # daemon should resume monitoring the interrupted run
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from jules_daemon.wiki import current_run
from jules_daemon.wiki.interrupted_run import RecoveryVerdict
from jules_daemon.wiki.models import CurrentRun, ProcessIDs, RunStatus

logger = logging.getLogger(__name__)


class BootPath(Enum):
    """Which boot path the daemon should take."""

    FRESH_START = "fresh_start"
    RECOVERY = "recovery"


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class BootDecision:
    """Structured result of the boot path routing decision.

    Provides:
    - path: which path was chosen (fresh_start or recovery)
    - reason: human-readable explanation for the decision
    - run: the CurrentRun state resulting from the decision
    - prior_run_id: the run_id of the prior run (empty for fresh start
      with no prior record; set for terminal-state fresh starts and
      all recovery cases)
    - wiki_file: path to the written wiki file
    """

    path: BootPath
    reason: str
    run: CurrentRun
    prior_run_id: str
    wiki_file: Path

    @property
    def is_fresh_start(self) -> bool:
        """True if the daemon should start fresh."""
        return self.path == BootPath.FRESH_START

    @property
    def is_recovery(self) -> bool:
        """True if the daemon should recover an interrupted run."""
        return self.path == BootPath.RECOVERY


def _validate_inputs(
    verdict: RecoveryVerdict,
    prior_run: CurrentRun | None,
    daemon_pid: int,
) -> None:
    """Validate route_boot inputs.

    Raises:
        ValueError: If inputs are invalid.
    """
    if daemon_pid <= 0:
        raise ValueError(
            f"daemon_pid must be a positive integer, got {daemon_pid}"
        )
    if verdict.recovery_needed and prior_run is None:
        raise ValueError(
            "recovery path requires a run record but prior_run is None"
        )


def _fresh_start(
    verdict: RecoveryVerdict,
    wiki_root: Path,
) -> BootDecision:
    """Execute the fresh-start path: write a clean idle record.

    Args:
        verdict: The interruption verdict (not needing recovery).
        wiki_root: Path to the wiki root directory.

    Returns:
        BootDecision with fresh-start path and idle run.
    """
    idle_run = CurrentRun(status=RunStatus.IDLE)
    wiki_file = current_run.write(wiki_root, idle_run)

    prior_run_id = verdict.run_id if verdict.run_id else ""
    status_label = verdict.interrupted_status.value

    if not verdict.run_id:
        reason = "No prior run record -- starting fresh"
    else:
        reason = f"Prior run was {status_label} -- starting fresh"

    prior_status_label = status_label if verdict.run_id else "<none>"
    logger.info(
        "Boot path: FRESH_START (prior_run_id=%s, prior_status=%s)",
        prior_run_id or "<none>",
        prior_status_label,
    )

    return BootDecision(
        path=BootPath.FRESH_START,
        reason=reason,
        run=idle_run,
        prior_run_id=prior_run_id,
        wiki_file=wiki_file,
    )


def _recover(
    verdict: RecoveryVerdict,
    prior_run: CurrentRun,
    wiki_root: Path,
    daemon_pid: int,
) -> BootDecision:
    """Execute the recovery path: update the interrupted run with new daemon PID.

    Preserves all run context (SSH target, command, progress) but updates:
    - daemon PID to the new daemon's PID
    - updated_at timestamp to mark recovery start

    Args:
        verdict: The interruption verdict (needing recovery).
        prior_run: The interrupted run record from the wiki.
        wiki_root: Path to the wiki root directory.
        daemon_pid: PID of the new daemon process.

    Returns:
        BootDecision with recovery path and updated run.
    """
    updated_pids = ProcessIDs(
        daemon=daemon_pid,
        remote=prior_run.pids.remote,
    )
    recovered_run = replace(
        prior_run,
        pids=updated_pids,
        updated_at=_now_utc(),
    )
    wiki_file = current_run.write(wiki_root, recovered_run)

    status_label = verdict.interrupted_status.value
    reason = (
        f"Recovering interrupted run {prior_run.run_id} "
        f"(was {status_label}, stale {verdict.stale_seconds:.1f}s) "
        f"-- daemon PID updated to {daemon_pid}"
    )

    logger.info(
        "Boot path: RECOVERY (run_id=%s, status=%s, new_daemon_pid=%d)",
        prior_run.run_id,
        status_label,
        daemon_pid,
    )

    return BootDecision(
        path=BootPath.RECOVERY,
        reason=reason,
        run=recovered_run,
        prior_run_id=prior_run.run_id,
        wiki_file=wiki_file,
    )


def route_boot(
    verdict: RecoveryVerdict,
    prior_run: CurrentRun | None,
    wiki_root: Path,
    daemon_pid: int,
) -> BootDecision:
    """Route the daemon boot to fresh-start or recovery path.

    This is the primary entry point. It takes the interruption verdict
    (from detect_interrupted_run), the original run record (or None),
    and the new daemon's PID, then:

    1. If recovery_needed is False: writes a clean idle record and
       returns a fresh-start decision.
    2. If recovery_needed is True: updates the interrupted run's daemon
       PID and timestamp, writes it back to the wiki, and returns a
       recovery decision.

    Args:
        verdict: The RecoveryVerdict from detect_interrupted_run.
        prior_run: The CurrentRun from the wiki (None if no file existed).
        wiki_root: Path to the wiki root directory.
        daemon_pid: PID of the new daemon process (must be positive).

    Returns:
        BootDecision with the chosen path and resulting state.

    Raises:
        ValueError: If daemon_pid is not positive, or recovery is needed
            but prior_run is None.
    """
    _validate_inputs(verdict, prior_run, daemon_pid)

    if verdict.recovery_needed:
        # prior_run is guaranteed non-None by _validate_inputs, but guard
        # explicitly so the check survives Python -O (assert stripping)
        if prior_run is None:
            raise TypeError(
                "prior_run must not be None when recovery_needed is True"
            )
        return _recover(verdict, prior_run, wiki_root, daemon_pid)

    return _fresh_start(verdict, wiki_root)
