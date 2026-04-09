"""Monitoring loop transition logic for crash recovery resumption.

Bridges the fast-forward/reconciliation phase and the live polling loop.
Given a validated ResumptionPoint from the reconciliation outcome, this
module seamlessly switches into live autonomous progress tracking mode
without duplicating or dropping events, re-registering progress callbacks
from the reconciled state.

The transition has three stages:
  1. Build initial MonitorStatus: Seed the monitoring state from the
     recovered checkpoint metrics and the reconciled resumption point.
  2. Replay gap events: If the reconciliation detected missed lines
     (emitted during disconnect), replay each as a callback invocation
     so downstream consumers stay in sync.
  3. Handoff: Return the TransitionResult with the correct live
     sequence_number for the PollingLoop to continue from.

Key design decisions:
    - Immutable dataclasses throughout (frozen=True)
    - Never-raise contract on replay (callback errors are logged)
    - Monotonic sequence numbers across gap replay and live monitoring
    - Gap replay can be disabled via TransitionConfig
    - When disabled, the sequence number jumps past the gap count
    - TransitionOutcome distinguishes LIVE_MONITORING vs FRESH_START

Usage:
    from jules_daemon.monitor.monitor_transition import prepare_transition

    result = await prepare_transition(
        outcome=reconciliation_outcome,
        checkpoint=monitoring_checkpoint,
        session_id="run-abc",
        callback=my_status_callback,
    )

    if result.outcome == TransitionOutcome.LIVE_MONITORING:
        # Start PollingLoop from result.live_sequence_start
        ...
    elif result.outcome == TransitionOutcome.FRESH_START:
        # Start PollingLoop from sequence 0
        ...
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable

from jules_daemon.wiki.checkpoint_recovery import (
    ExtractedMetrics,
    MonitoringCheckpoint,
)
from jules_daemon.wiki.monitor_status import MonitorStatus, OutputPhase, ParsedState
from jules_daemon.wiki.models import RunStatus
from jules_daemon.wiki.resumption_reconciler import (
    GapMetadata,
    ReconciliationOutcome,
)

__all__ = [
    "TransitionConfig",
    "TransitionOutcome",
    "TransitionPhase",
    "TransitionResult",
    "build_initial_status",
    "prepare_transition",
    "replay_gap_events",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback type alias
# ---------------------------------------------------------------------------

StatusCallback = Callable[[MonitorStatus], Awaitable[None]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionConfig:
    """Immutable configuration for the monitoring loop transition.

    Attributes:
        gap_replay_batch_size: Maximum number of gap events to replay
            in a single batch. Must be positive.
        gap_replay_enabled: When False, gap events are not replayed to
            callbacks; the sequence number jumps past the gap count
            instead. Default: True.
    """

    gap_replay_batch_size: int = 50
    gap_replay_enabled: bool = True

    def __post_init__(self) -> None:
        if self.gap_replay_batch_size <= 0:
            raise ValueError(
                f"gap_replay_batch_size must be positive, "
                f"got {self.gap_replay_batch_size}"
            )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TransitionOutcome(Enum):
    """Outcome of the monitoring loop transition.

    LIVE_MONITORING: The loop should resume from the reconciled
        position, continuing where it left off.
    FRESH_START: The loop should start from the beginning because
        the reconciliation outcome was not usable.
    """

    LIVE_MONITORING = "live_monitoring"
    FRESH_START = "fresh_start"


class TransitionPhase(Enum):
    """Phase of the transition process.

    Used for tracking which stage the transition is in.
    """

    BUILDING_STATUS = "building_status"
    REPLAYING_GAP = "replaying_gap"
    COMPLETE = "complete"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionResult:
    """Immutable result of the monitoring loop transition.

    Describes the outcome of the transition process, including the
    initial MonitorStatus, how many gap events were replayed, and
    the sequence number at which live monitoring should begin.

    Attributes:
        outcome: Whether to start live monitoring or fresh start.
        phase: The current/final phase of the transition process.
        initial_status: The MonitorStatus seeded from the reconciled
            state. This is the baseline for the live monitoring loop.
        gap_events_replayed: Number of gap events replayed to the
            callback before live monitoring starts. 0 if no gap or
            gap replay is disabled.
        live_sequence_start: The sequence number at which the live
            polling loop should begin. Ensures no duplication or
            gaps in the sequence.
        error: Human-readable error description (None on success).
        reconciliation_reason: The reason string from the
            ReconciliationOutcome, for audit purposes.
    """

    outcome: TransitionOutcome
    phase: TransitionPhase
    initial_status: MonitorStatus
    gap_events_replayed: int
    live_sequence_start: int
    error: str | None
    reconciliation_reason: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


_STATUS_TO_PHASE: dict[RunStatus, OutputPhase] = {
    RunStatus.RUNNING: OutputPhase.RUNNING,
    RunStatus.PENDING_APPROVAL: OutputPhase.SETUP,
    RunStatus.COMPLETED: OutputPhase.COMPLETE,
    RunStatus.FAILED: OutputPhase.ERROR,
    RunStatus.CANCELLED: OutputPhase.COMPLETE,
    RunStatus.IDLE: OutputPhase.UNKNOWN,
}


def _metrics_to_parsed_state(
    metrics: ExtractedMetrics,
    run_status: RunStatus,
) -> ParsedState:
    """Convert checkpoint ExtractedMetrics to a ParsedState.

    Maps the recovered test counts into the monitoring ParsedState
    model so the live loop has accurate starting metrics. The phase
    is derived from the checkpoint's run status rather than hardcoded.
    """
    phase = _STATUS_TO_PHASE.get(run_status, OutputPhase.UNKNOWN)
    return ParsedState(
        phase=phase,
        tests_discovered=metrics.tests_total,
        tests_passed=metrics.tests_passed,
        tests_failed=metrics.tests_failed,
        tests_skipped=metrics.tests_skipped,
        tests_total=metrics.tests_total,
    )


# ---------------------------------------------------------------------------
# Public API: build_initial_status
# ---------------------------------------------------------------------------


def build_initial_status(
    *,
    outcome: ReconciliationOutcome,
    checkpoint: MonitoringCheckpoint,
    session_id: str,
) -> MonitorStatus:
    """Build the initial MonitorStatus from the reconciled state.

    Creates an immutable MonitorStatus snapshot that seeds the live
    monitoring loop with the correct sequence number and test metrics
    from the recovered checkpoint.

    The sequence_number is set to the resumption point's line_number,
    ensuring the live loop continues the sequence without gaps or
    duplicates.

    Args:
        outcome: The reconciliation outcome from the fast-forward phase.
        checkpoint: The monitoring checkpoint recovered from the wiki.
        session_id: The session identifier for the monitoring session.

    Returns:
        Frozen MonitorStatus seeded with the reconciled state.

    Raises:
        ValueError: If session_id is empty.
    """
    if not session_id:
        raise ValueError("session_id must not be empty")

    resume_line = outcome.resumption_point.line_number
    parsed_state = _metrics_to_parsed_state(
        checkpoint.extracted_metrics,
        checkpoint.status,
    )

    return MonitorStatus(
        session_id=session_id,
        timestamp=_now_utc(),
        raw_output_chunk="",
        parsed_state=parsed_state,
        exit_status=None,
        sequence_number=resume_line,
    )


# ---------------------------------------------------------------------------
# Public API: replay_gap_events
# ---------------------------------------------------------------------------


async def replay_gap_events(
    *,
    gap: GapMetadata,
    initial_status: MonitorStatus,
    callback: StatusCallback,
    config: TransitionConfig | None = None,
) -> MonitorStatus:
    """Replay missed gap events to the progress callback.

    When the reconciliation detected lines emitted during a disconnect,
    this function replays each missed line as a MonitorStatus callback
    invocation. This ensures downstream consumers (wiki persistence,
    IPC streaming, etc.) see every event exactly once.

    If gap replay is disabled, the sequence number is advanced past
    the gap count without invoking the callback.

    Callback errors are logged but never raised -- the replay continues
    for all missed lines.

    Args:
        gap: The gap metadata from the reconciliation outcome.
        initial_status: The MonitorStatus to build upon for gap events.
        callback: The async callback to invoke for each gap event.
        config: Optional transition configuration. Uses defaults if None.

    Returns:
        The MonitorStatus after all gap events have been replayed.
        Its sequence_number reflects the final position after the gap.
    """
    effective_config = config if config is not None else TransitionConfig()

    # No gap -- nothing to replay
    if not gap.has_gap or gap.missed_line_count == 0:
        return initial_status

    # Gap replay disabled -- advance sequence past the gap
    if not effective_config.gap_replay_enabled:
        advanced_status = initial_status.with_update(
            sequence_number=initial_status.sequence_number + gap.missed_line_count,
            timestamp=_now_utc(),
        )
        logger.info(
            "Gap replay disabled: advanced sequence by %d to %d",
            gap.missed_line_count,
            advanced_status.sequence_number,
        )
        return advanced_status

    # Replay each missed line as a callback event.
    # Yields control between batches to avoid blocking the event loop.
    current_status = initial_status
    batch_size = effective_config.gap_replay_batch_size

    for idx, missed_line in enumerate(gap.missed_lines):
        now = _now_utc()
        current_status = current_status.with_output(
            timestamp=now,
            raw_output_chunk=missed_line,
        )

        try:
            await callback(current_status)
        except asyncio.CancelledError:
            logger.warning(
                "Gap replay cancelled at event %d/%d (sequence=%d)",
                idx + 1,
                len(gap.missed_lines),
                current_status.sequence_number,
            )
            raise
        except Exception as exc:
            logger.warning(
                "Callback error during gap replay (event %d/%d): %s",
                idx + 1,
                len(gap.missed_lines),
                exc,
            )

        # Yield control between batches to keep the event loop responsive
        if (idx + 1) % batch_size == 0:
            await asyncio.sleep(0)

    logger.info(
        "Gap replay complete: replayed %d events, sequence now at %d",
        len(gap.missed_lines),
        current_status.sequence_number,
    )

    return current_status


# ---------------------------------------------------------------------------
# Public API: prepare_transition
# ---------------------------------------------------------------------------


async def prepare_transition(
    *,
    outcome: ReconciliationOutcome,
    checkpoint: MonitoringCheckpoint,
    session_id: str,
    callback: StatusCallback,
    config: TransitionConfig | None = None,
) -> TransitionResult:
    """Prepare the transition from fast-forward to live monitoring.

    Orchestrates the full transition pipeline:
    1. Determine if the outcome is usable (LIVE_MONITORING vs FRESH_START)
    2. Build the initial MonitorStatus from the reconciled state
    3. Replay any gap events to the callback
    4. Return the TransitionResult with the correct live_sequence_start

    This function never raises. All errors are captured in the returned
    TransitionResult.

    Args:
        outcome: The reconciliation outcome from the fast-forward phase.
        checkpoint: The monitoring checkpoint recovered from the wiki.
        session_id: The session identifier for the monitoring session.
        callback: The async callback for progress updates.
        config: Optional transition configuration.

    Returns:
        TransitionResult describing the transition outcome and the
        sequence number at which live monitoring should begin.
    """
    effective_config = config if config is not None else TransitionConfig()

    # -- Validate session_id up front (never-raise: return error result) --
    if not session_id:
        fallback_status = MonitorStatus(
            session_id="<invalid>",
            timestamp=_now_utc(),
            sequence_number=0,
        )
        return TransitionResult(
            outcome=TransitionOutcome.FRESH_START,
            phase=TransitionPhase.COMPLETE,
            initial_status=fallback_status,
            gap_events_replayed=0,
            live_sequence_start=0,
            error="session_id must not be empty",
            reconciliation_reason=outcome.reason,
        )

    # -- Determine transition outcome --
    if not outcome.is_usable:
        logger.info(
            "Reconciliation outcome not usable: %s -- transitioning to fresh start",
            outcome.reason,
        )

        initial_status = build_initial_status(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id=session_id,
        )

        return TransitionResult(
            outcome=TransitionOutcome.FRESH_START,
            phase=TransitionPhase.COMPLETE,
            initial_status=initial_status,
            gap_events_replayed=0,
            live_sequence_start=0,
            error=None,
            reconciliation_reason=outcome.reason,
        )

    # -- Build initial status (BUILDING_STATUS phase) --
    logger.info(
        "Building initial status for live monitoring: "
        "resumption_line=%d session=%s",
        outcome.resumption_point.line_number,
        session_id,
    )

    initial_status = build_initial_status(
        outcome=outcome,
        checkpoint=checkpoint,
        session_id=session_id,
    )

    # -- Replay gap events (REPLAYING_GAP phase) --
    gap = outcome.gap
    post_replay_status = await replay_gap_events(
        gap=gap,
        initial_status=initial_status,
        callback=callback,
        config=effective_config,
    )

    gap_events_replayed = (
        len(gap.missed_lines)
        if gap.has_gap and effective_config.gap_replay_enabled
        else 0
    )

    # -- Compute live sequence start --
    # The live polling loop should start from the sequence number
    # after the last replayed gap event (or the initial status if
    # no gap was replayed). This ensures no duplication: the PollingLoop
    # calls with_output() which auto-increments, so the first live
    # event will be at live_sequence_start + 1.
    live_sequence_start = post_replay_status.sequence_number

    logger.info(
        "Transition complete: outcome=%s gap_replayed=%d live_start=%d",
        TransitionOutcome.LIVE_MONITORING.value,
        gap_events_replayed,
        live_sequence_start,
    )

    return TransitionResult(
        outcome=TransitionOutcome.LIVE_MONITORING,
        phase=TransitionPhase.COMPLETE,
        initial_status=post_replay_status,
        gap_events_replayed=gap_events_replayed,
        live_sequence_start=live_sequence_start,
        error=None,
        reconciliation_reason=outcome.reason,
    )
