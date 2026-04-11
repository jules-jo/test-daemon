"""Crash recovery wiring for daemon startup.

Connects the existing crash recovery detection and orchestration
machinery (in ``jules_daemon.wiki.recovery_orchestrator``,
``jules_daemon.wiki.crash_recovery``, etc.) to the daemon's startup
sequence in ``jules_daemon.__main__``.

On daemon startup, ``try_crash_recovery()`` should be called after
``initialize_wiki()`` but before the IPC server begins accepting
connections. It reads the wiki ``current-run.md`` state file. If the
record is in an active state (RUNNING, PENDING_APPROVAL), the daemon
was killed mid-run, so recovery is required:

1. The existing ``orchestrate_recovery()`` pipeline is invoked under
   a bounded deadline (default 30 seconds).
2. Because this pre-server startup path does not yet have a live
   ``SSHConnector`` initialized, RECONNECT-style recovery will
   fail through to the orchestrator's graceful FAIL path, which marks
   the wiki run as FAILED and writes a recovery log entry.
3. The wrapper then promotes the failed run into history (resetting
   the wiki ``current-run.md`` back to idle so the daemon can accept
   new commands) and returns a ``RecoveredRunInfo`` describing the
   outcome.

The wrapper is deliberately defensive: every exception is caught
and converted to a soft-fail recovery result so that a broken wiki
file can never crash daemon startup. The recovery call is also
wrapped in ``asyncio.wait_for`` with a hard timeout on top of the
orchestrator's internal deadline -- if recovery cannot complete in
time the daemon still continues to start.

Usage::

    from pathlib import Path
    from jules_daemon.startup.crash_recovery_wire import try_crash_recovery

    recovered = await try_crash_recovery(wiki_dir)
    if recovered is not None:
        handler._last_completed_run = recovered.result
        handler._last_failure = recovered.notification
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from jules_daemon.execution.run_pipeline import RunResult
from jules_daemon.wiki import current_run as current_run_io
from jules_daemon.wiki import run_promotion
from jules_daemon.wiki.crash_recovery import (
    CrashRecoveryResult,
    RecoveryAction,
    detect_crash_recovery,
)
from jules_daemon.wiki.models import RunStatus
from jules_daemon.wiki.recovery_orchestrator import (
    RecoveryOutcome,
    RecoveryTimeoutConfig,
    orchestrate_recovery,
)

__all__ = [
    "DEFAULT_RECOVERY_DEADLINE_SECONDS",
    "RecoveredRunInfo",
    "try_crash_recovery",
]

logger = logging.getLogger(__name__)


DEFAULT_RECOVERY_DEADLINE_SECONDS: float = 30.0
"""Hard ceiling on crash-recovery wall-clock time during daemon startup.

Per the original crash-resilience SLA, the daemon must recover from a
crash in under 30 seconds. If recovery exceeds this budget, startup
continues anyway with the run marked as recovery-incomplete.
"""

_DAEMON_DOWN_ERROR_PREFIX = "daemon was down"
"""Error message prefix used when the daemon cannot re-attach to a run.

This makes it easy for downstream consumers (status, history) to
recognize runs that failed specifically because the daemon was not
running to observe them, rather than because the tests themselves
failed.
"""


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveredRunInfo:
    """Summary of a crash-recovery attempt.

    Produced by :func:`try_crash_recovery` whenever an interrupted run
    was detected and an attempt was made to reconcile it. Contains
    everything the IPC ``RequestHandler`` needs to surface the recovery
    outcome to the next CLI client:

    - ``result``: a :class:`RunResult` that can be stored in
      ``RequestHandler._last_completed_run`` so the next ``status``
      query reports the recovered run.
    - ``notification``: a multi-line human-readable message that can be
      assigned to ``RequestHandler._last_failure`` so the next handshake
      surfaces a "daemon recovered from crash" banner.
    - ``outcome``: the underlying orchestrator outcome, for tests and
      diagnostics.
    - ``recovery_complete``: ``True`` when recovery ran end-to-end
      within the deadline, ``False`` when the wall-clock timeout was
      hit (the run is marked recovery-incomplete but the daemon still
      started).
    """

    result: RunResult
    notification: str
    outcome: RecoveryOutcome | None
    recovery_complete: bool

    @property
    def run_id(self) -> str:
        """Identifier of the recovered run."""
        return self.result.run_id

    @property
    def status_label(self) -> str:
        """Human-readable status label for the recovered run."""
        return "COMPLETED" if self.result.success else "FAILED"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _is_active_status(status: RunStatus) -> bool:
    """Return True if the run status indicates an interrupted active run."""
    return status in (
        RunStatus.RUNNING,
        RunStatus.PENDING_APPROVAL,
    )


def _build_recovery_error(
    recovery: CrashRecoveryResult,
    outcome: RecoveryOutcome | None,
    wall_timeout: bool,
) -> str:
    """Build the human-readable ``error`` string for a recovered RunResult.

    The prefix ``"daemon was down"`` is used as a stable marker so
    downstream consumers can detect crash-recovered runs.
    """
    if wall_timeout:
        return (
            f"{_DAEMON_DOWN_ERROR_PREFIX}: recovery exceeded wall-clock "
            f"deadline of {DEFAULT_RECOVERY_DEADLINE_SECONDS:.0f}s -- "
            f"run marked recovery-incomplete"
        )

    if outcome is None:
        return (
            f"{_DAEMON_DOWN_ERROR_PREFIX}: recovery did not produce an "
            f"outcome -- run marked FAILED"
        )

    # RESUME_APPROVAL always succeeds in the orchestrator (it just writes
    # a log), but the daemon still cannot re-present the approval prompt
    # from a cold start, so we treat it as a FAILED recovery at this
    # level and instruct the user to re-submit the command.
    if outcome.action_taken == RecoveryAction.RESUME_APPROVAL:
        return (
            f"{_DAEMON_DOWN_ERROR_PREFIX}: command was awaiting approval "
            f"when the daemon stopped -- please re-submit the request"
        )

    if outcome.timed_out:
        return (
            f"{_DAEMON_DOWN_ERROR_PREFIX}: recovery timed out after "
            f"{outcome.total_duration_seconds:.1f}s "
            f"(deadline {outcome.deadline_seconds:.1f}s)"
        )

    if not outcome.success:
        reason = outcome.error or "recovery failed"
        return (
            f"{_DAEMON_DOWN_ERROR_PREFIX}: cannot recover "
            f"(reason: {reason})"
        )

    # Success path -- RECONNECT that reached the wiki update phase but
    # no actual re-attach happened (no SSH connector available). We
    # still report the run as partial-due-to-crash.
    return (
        f"{_DAEMON_DOWN_ERROR_PREFIX}: daemon restarted after crash -- "
        f"results may be incomplete"
    )


def _build_run_result(
    recovery: CrashRecoveryResult,
    outcome: RecoveryOutcome | None,
    wall_timeout: bool,
) -> RunResult:
    """Build a :class:`RunResult` describing a recovered run.

    The result is always marked as a failure from the perspective of
    test outcome (``success=False``) because when the daemon is down
    we cannot know whether the tests actually passed. The ``error``
    field carries the ``"daemon was down"`` prefix so downstream
    consumers can detect crash-recovered runs.
    """
    error = _build_recovery_error(
        recovery=recovery,
        outcome=outcome,
        wall_timeout=wall_timeout,
    )

    now = _now_utc()
    duration = 0.0
    if outcome is not None:
        duration = round(outcome.total_duration_seconds, 3)

    return RunResult(
        success=False,
        run_id=recovery.run_id or "unknown",
        command=recovery.resolved_shell or recovery.natural_language_command or "",
        target_host=recovery.host or "",
        target_user=recovery.user or "",
        exit_code=None,
        stdout="",
        stderr="",
        error=error,
        duration_seconds=duration,
        started_at=now,
        completed_at=now,
    )


def _build_notification(result: RunResult) -> str:
    """Build the handshake notification message for a recovered run."""
    lines = [
        "!!! DAEMON RECOVERED FROM CRASH !!!",
        f"Previous run: {result.run_id}",
        f"Status: {'COMPLETED' if result.success else 'FAILED'}",
        "Daemon was down -- results may be incomplete.",
    ]
    if result.error:
        lines.append(f"Details: {result.error}")
    if result.target_host:
        lines.append(f"Target: {result.target_host}")
    if result.command:
        lines.append(f"Command: {result.command}")
    return "\n".join(lines)


def _mark_wiki_failed_and_promote(
    wiki_dir: Path,
    error_message: str,
) -> None:
    """Write a FAILED current-run and promote to history.

    Best-effort. Never raises. On failure, logs a warning -- the
    daemon still proceeds with startup so a corrupt wiki file cannot
    keep the daemon from ever starting again.

    This resets the wiki ``current-run.md`` to idle so a subsequent
    CLI ``run`` request is not blocked by the stale record.
    """
    try:
        existing = current_run_io.read(wiki_dir)
    except Exception as exc:
        logger.warning(
            "Unable to read wiki current-run for promotion after recovery: %s",
            exc,
        )
        return

    if existing is None:
        logger.debug(
            "No wiki current-run file to promote after recovery"
        )
        return

    try:
        failed = existing.with_failed(error_message, existing.progress)
        current_run_io.write(wiki_dir, failed)
    except Exception as exc:
        logger.warning(
            "Failed to write FAILED state to wiki during recovery: %s",
            exc,
        )
        return

    try:
        run_promotion.promote_run(wiki_dir, failed)
    except Exception as exc:
        # Promotion failure is non-fatal -- the wiki still has the
        # FAILED record and the next clear() call will reset it.
        logger.warning(
            "Failed to promote recovered run to history: %s",
            exc,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def try_crash_recovery(
    wiki_dir: Path,
    *,
    deadline_seconds: float = DEFAULT_RECOVERY_DEADLINE_SECONDS,
) -> RecoveredRunInfo | None:
    """Detect and recover from an interrupted run at daemon startup.

    This is the single entry point for wiring crash recovery into the
    daemon's ``__main__._run_daemon()``. It should be called after
    ``initialize_wiki()`` but before the IPC server starts accepting
    connections.

    Behavior:

    - Reads the wiki ``current-run.md`` file via
      :func:`detect_crash_recovery`.
    - If the record is missing, corrupted, IDLE, or in a terminal
      state, returns ``None`` -- no recovery is needed.
    - Otherwise, invokes :func:`orchestrate_recovery` under a wall-
      clock deadline. Because the daemon has not yet constructed an
      ``SSHConnector`` at this point in startup, the RECONNECT path
      falls through to the orchestrator's graceful FAIL branch which
      marks the run as FAILED in the wiki.
    - Writes a FAILED state into ``current-run.md`` and promotes it
      to history so the wiki is reset to idle for the next run.
    - Returns a :class:`RecoveredRunInfo` suitable for assigning to
      ``RequestHandler._last_completed_run`` and ``_last_failure``.

    This function never raises. All exceptions are caught, logged,
    and converted into a soft-fail :class:`RecoveredRunInfo` so that
    a corrupt wiki file or broken recovery module can never prevent
    the daemon from starting.

    Args:
        wiki_dir: Path to the wiki root directory.
        deadline_seconds: Hard wall-clock deadline for the recovery
            call (wraps the orchestrator's internal deadline with an
            ``asyncio.wait_for`` safety net). Default: 30.0 seconds.

    Returns:
        :class:`RecoveredRunInfo` describing the recovery outcome, or
        ``None`` if no interrupted run was detected.
    """
    logger.debug("Crash recovery wire: checking wiki at %s", wiki_dir)

    # --- Phase 1: detect ---
    try:
        recovery = detect_crash_recovery(wiki_dir)
    except Exception as exc:
        # detect_crash_recovery should never raise, but defend against
        # a future regression with a hard try/except.
        logger.warning(
            "Crash recovery detection raised unexpectedly: %s",
            exc,
            exc_info=True,
        )
        return None

    if recovery.action == RecoveryAction.FRESH_START:
        logger.debug(
            "Crash recovery wire: no interrupted run (%s)",
            recovery.reason,
        )
        return None

    if not _is_active_status(recovery.status):
        logger.debug(
            "Crash recovery wire: status %s is not active, skipping",
            recovery.status.value,
        )
        return None

    logger.warning(
        "Crash recovery wire: interrupted run detected "
        "(run_id=%s, status=%s, host=%s) -- starting recovery",
        recovery.run_id,
        recovery.status.value,
        recovery.host or "unknown",
    )

    # --- Phase 2: orchestrate recovery under a wall-clock deadline ---
    wall_start = time.monotonic()
    outcome: RecoveryOutcome | None = None
    wall_timeout = False

    try:
        # The orchestrator has its own monotonic deadline, but we also
        # wrap it with asyncio.wait_for so that a bug in the orchestrator
        # (e.g. a hung SSH connector) can never block startup indefinitely.
        outcome = await asyncio.wait_for(
            orchestrate_recovery(
                recovery=recovery,
                wiki_root=wiki_dir,
                connector=None,  # no live connector at startup
                config=RecoveryTimeoutConfig(
                    total_deadline_seconds=deadline_seconds,
                ),
            ),
            timeout=deadline_seconds + 1.0,  # small buffer over inner deadline
        )
    except asyncio.TimeoutError:
        wall_timeout = True
        logger.warning(
            "Crash recovery wire: wall-clock deadline %.1fs exceeded "
            "-- continuing startup with run marked recovery-incomplete",
            deadline_seconds,
        )
    except Exception as exc:
        logger.warning(
            "Crash recovery wire: orchestrate_recovery raised: %s",
            exc,
            exc_info=True,
        )

    wall_elapsed = time.monotonic() - wall_start

    # --- Phase 3: build the RunResult and notification ---
    result = _build_run_result(
        recovery=recovery,
        outcome=outcome,
        wall_timeout=wall_timeout,
    )
    notification = _build_notification(result)

    # --- Phase 4: reset wiki current-run so the daemon is idle ---
    # The orchestrator's failure path writes FAILED to the wiki, but on
    # the success path (RESUME_APPROVAL, RECONNECT-with-no-connector
    # short-circuit, or any other non-FAILED outcome) we still need to
    # reset the wiki so subsequent CLI commands aren't blocked by the
    # stale record. promote_run() handles writing history + clearing.
    try:
        _mark_wiki_failed_and_promote(
            wiki_dir=wiki_dir,
            error_message=result.error or "daemon was down",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Crash recovery wire: wiki reset failed: %s",
            exc,
            exc_info=True,
        )

    logger.info(
        "Crash recovery wire: recovered run %s in %.2fs -- status=%s (%s)",
        result.run_id,
        wall_elapsed,
        "COMPLETED" if result.success else "FAILED",
        "complete" if not wall_timeout else "incomplete",
    )

    return RecoveredRunInfo(
        result=result,
        notification=notification,
        outcome=outcome,
        recovery_complete=not wall_timeout,
    )
