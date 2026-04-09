"""Tests for monitoring loop transition logic.

Verifies that the transition module:
- Seamlessly switches from fast-forward resumption to live monitoring
- Never duplicates or drops events across the transition boundary
- Re-registers progress callbacks from the reconciled state
- Correctly seeds initial MonitorStatus from reconciliation outcome
- Handles gap events (missed during disconnect) before starting live loop
- Handles edge cases: unusable outcomes, fresh starts, no-gap resumes
- Produces immutable TransitionResult snapshots
- Preserves sequence ordering across gap replay and live monitoring
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from jules_daemon.monitor.monitor_transition import (
    TransitionConfig,
    TransitionOutcome,
    TransitionPhase,
    build_initial_status,
    prepare_transition,
    replay_gap_events,
)
from jules_daemon.wiki.checkpoint_recovery import (
    ExtractedMetrics,
    MonitoringCheckpoint,
    RecoverySource,
)
from jules_daemon.wiki.monitor_status import (
    MonitorStatus,
    OutputPhase,
)
from jules_daemon.wiki.models import RunStatus
from jules_daemon.wiki.resumption_reconciler import (
    GapMetadata,
    GapType,
    MatchQuality,
    ReconciliationOutcome,
    ResumptionPoint,
)


# ---------------------------------------------------------------------------
# Helpers: factory functions for test data
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_resumption_point(
    *,
    line_number: int = 42,
    marker: str = "PASSED test_foo",
    is_valid: bool = True,
    match_quality: MatchQuality = MatchQuality.EXACT,
) -> ResumptionPoint:
    return ResumptionPoint(
        line_number=line_number,
        marker=marker,
        is_valid=is_valid,
        match_quality=match_quality,
    )


def _make_gap(
    *,
    gap_type: GapType = GapType.NONE,
    missed_line_count: int = 0,
    missed_lines: tuple[str, ...] = (),
    checkpoint_line_number: int = 42,
    actual_resume_line_number: int = 42,
    detail: str = "No gap",
) -> GapMetadata:
    return GapMetadata(
        gap_type=gap_type,
        missed_line_count=missed_line_count,
        missed_lines=missed_lines,
        checkpoint_line_number=checkpoint_line_number,
        actual_resume_line_number=actual_resume_line_number,
        detail=detail,
    )


def _make_outcome(
    *,
    resumption_point: ResumptionPoint | None = None,
    gap: GapMetadata | None = None,
    checkpoint_run_id: str = "run-abc-123",
    is_usable: bool = True,
    reason: str = "Marker found at expected position",
) -> ReconciliationOutcome:
    return ReconciliationOutcome(
        resumption_point=resumption_point or _make_resumption_point(),
        gap=gap or _make_gap(),
        checkpoint_run_id=checkpoint_run_id,
        is_usable=is_usable,
        reason=reason,
    )


def _make_checkpoint(
    *,
    last_parsed_line_number: int = 42,
    run_id: str = "run-abc-123",
    status: RunStatus = RunStatus.RUNNING,
    tests_passed: int = 10,
    tests_failed: int = 1,
    tests_skipped: int = 2,
    tests_total: int = 50,
    percent: float = 26.0,
) -> MonitoringCheckpoint:
    return MonitoringCheckpoint(
        last_parsed_line_number=last_parsed_line_number,
        timestamp=_now(),
        extracted_metrics=ExtractedMetrics(
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            tests_total=tests_total,
            percent=percent,
        ),
        run_id=run_id,
        status=status,
        source=RecoverySource.WIKI_STATE,
        error=None,
    )


# ---------------------------------------------------------------------------
# Test: TransitionConfig validation
# ---------------------------------------------------------------------------


class TestTransitionConfig:
    def test_defaults(self) -> None:
        config = TransitionConfig()
        assert config.gap_replay_batch_size == 50
        assert config.gap_replay_enabled is True

    def test_frozen(self) -> None:
        config = TransitionConfig()
        with pytest.raises(AttributeError):
            config.gap_replay_batch_size = 100  # type: ignore[misc]

    def test_negative_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="gap_replay_batch_size must be positive"):
            TransitionConfig(gap_replay_batch_size=-1)

    def test_zero_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="gap_replay_batch_size must be positive"):
            TransitionConfig(gap_replay_batch_size=0)

    def test_custom_values(self) -> None:
        config = TransitionConfig(gap_replay_batch_size=100, gap_replay_enabled=False)
        assert config.gap_replay_batch_size == 100
        assert config.gap_replay_enabled is False


# ---------------------------------------------------------------------------
# Test: build_initial_status from reconciled state
# ---------------------------------------------------------------------------


class TestBuildInitialStatus:
    def test_builds_status_from_outcome_and_checkpoint(self) -> None:
        outcome = _make_outcome()
        checkpoint = _make_checkpoint()

        status = build_initial_status(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="sess-1",
        )

        assert status.session_id == "sess-1"
        assert status.sequence_number == 42  # matches resumption line_number
        assert status.exit_status is None
        assert status.parsed_state.tests_passed == 10
        assert status.parsed_state.tests_failed == 1
        assert status.parsed_state.tests_skipped == 2
        assert status.parsed_state.tests_total == 50

    def test_fresh_start_produces_zero_sequence(self) -> None:
        outcome = _make_outcome(
            resumption_point=_make_resumption_point(line_number=0),
        )
        checkpoint = _make_checkpoint(
            last_parsed_line_number=0,
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            tests_total=0,
            percent=0.0,
        )

        status = build_initial_status(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="fresh-1",
        )

        assert status.sequence_number == 0
        assert status.parsed_state.tests_passed == 0

    def test_unusable_outcome_still_builds_status(self) -> None:
        """Even if the outcome is not usable, we still build a status
        to seed the loop (it may start fresh from line 0)."""
        outcome = _make_outcome(
            is_usable=False,
            resumption_point=_make_resumption_point(
                line_number=0, is_valid=False, match_quality=MatchQuality.NO_MATCH
            ),
        )
        checkpoint = _make_checkpoint(last_parsed_line_number=0)

        status = build_initial_status(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="unusable-1",
        )

        assert status.session_id == "unusable-1"
        assert status.sequence_number == 0

    def test_status_is_frozen(self) -> None:
        outcome = _make_outcome()
        checkpoint = _make_checkpoint()

        status = build_initial_status(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="frozen-1",
        )

        with pytest.raises(AttributeError):
            status.session_id = "hacked"  # type: ignore[misc]

    def test_empty_session_id_raises(self) -> None:
        outcome = _make_outcome()
        checkpoint = _make_checkpoint()

        with pytest.raises(ValueError, match="session_id must not be empty"):
            build_initial_status(
                outcome=outcome,
                checkpoint=checkpoint,
                session_id="",
            )


# ---------------------------------------------------------------------------
# Test: replay_gap_events
# ---------------------------------------------------------------------------


class TestReplayGapEvents:
    @pytest.mark.asyncio
    async def test_no_gap_produces_no_replay(self) -> None:
        gap = _make_gap(gap_type=GapType.NONE)
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        initial = MonitorStatus(
            session_id="replay-1",
            timestamp=_now(),
            sequence_number=42,
        )

        result_status = await replay_gap_events(
            gap=gap,
            initial_status=initial,
            callback=collect,
        )

        assert len(collected) == 0
        assert result_status.sequence_number == 42  # unchanged

    @pytest.mark.asyncio
    async def test_replays_missed_lines_as_events(self) -> None:
        gap = _make_gap(
            gap_type=GapType.OUTPUT_EMITTED_DURING_DISCONNECT,
            missed_line_count=3,
            missed_lines=(
                "PASSED test_alpha",
                "PASSED test_beta",
                "FAILED test_gamma",
            ),
            checkpoint_line_number=42,
            actual_resume_line_number=45,
        )
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        initial = MonitorStatus(
            session_id="replay-2",
            timestamp=_now(),
            sequence_number=42,
        )

        result_status = await replay_gap_events(
            gap=gap,
            initial_status=initial,
            callback=collect,
        )

        assert len(collected) == 3
        assert "PASSED test_alpha" in collected[0].raw_output_chunk
        assert "PASSED test_beta" in collected[1].raw_output_chunk
        assert "FAILED test_gamma" in collected[2].raw_output_chunk

        # Sequence numbers should be monotonically increasing
        seq_numbers = [s.sequence_number for s in collected]
        for i in range(1, len(seq_numbers)):
            assert seq_numbers[i] > seq_numbers[i - 1]

        # result_status should have the final sequence
        assert result_status.sequence_number == collected[-1].sequence_number

    @pytest.mark.asyncio
    async def test_callback_error_during_replay_is_logged_not_raised(self) -> None:
        gap = _make_gap(
            gap_type=GapType.OUTPUT_EMITTED_DURING_DISCONNECT,
            missed_line_count=2,
            missed_lines=("line A", "line B"),
        )
        call_count = 0

        async def failing_callback(status: MonitorStatus) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("callback exploded during replay")

        initial = MonitorStatus(
            session_id="replay-err-1",
            timestamp=_now(),
            sequence_number=10,
        )

        result_status = await replay_gap_events(
            gap=gap,
            initial_status=initial,
            callback=failing_callback,
        )

        # Both lines should have been replayed despite the error
        assert call_count == 2
        assert result_status.sequence_number > 10

    @pytest.mark.asyncio
    async def test_gap_replay_disabled_skips_replay(self) -> None:
        gap = _make_gap(
            gap_type=GapType.OUTPUT_EMITTED_DURING_DISCONNECT,
            missed_line_count=2,
            missed_lines=("line A", "line B"),
        )
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        initial = MonitorStatus(
            session_id="replay-disabled-1",
            timestamp=_now(),
            sequence_number=10,
        )

        config = TransitionConfig(gap_replay_enabled=False)
        result_status = await replay_gap_events(
            gap=gap,
            initial_status=initial,
            callback=collect,
            config=config,
        )

        assert len(collected) == 0
        # Sequence should jump past the gap
        assert result_status.sequence_number == 10 + 2

    @pytest.mark.asyncio
    async def test_partial_line_gap_produces_single_event(self) -> None:
        gap = _make_gap(
            gap_type=GapType.PARTIAL_LINE_AT_BOUNDARY,
            missed_line_count=0,
            missed_lines=(),
        )
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        initial = MonitorStatus(
            session_id="partial-1",
            timestamp=_now(),
            sequence_number=42,
        )

        result_status = await replay_gap_events(
            gap=gap,
            initial_status=initial,
            callback=collect,
        )

        # No missed lines to replay
        assert len(collected) == 0
        assert result_status.sequence_number == 42


# ---------------------------------------------------------------------------
# Test: prepare_transition (integration of build + replay + handoff)
# ---------------------------------------------------------------------------


class TestPrepareTransition:
    @pytest.mark.asyncio
    async def test_usable_outcome_no_gap(self) -> None:
        outcome = _make_outcome(
            gap=_make_gap(gap_type=GapType.NONE),
        )
        checkpoint = _make_checkpoint()
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        result = await prepare_transition(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="trans-1",
            callback=collect,
        )

        assert result.outcome == TransitionOutcome.LIVE_MONITORING
        assert result.phase == TransitionPhase.COMPLETE
        assert result.gap_events_replayed == 0
        assert result.initial_status is not None
        assert result.initial_status.session_id == "trans-1"
        assert result.live_sequence_start == 42
        assert result.error is None

    @pytest.mark.asyncio
    async def test_usable_outcome_with_gap(self) -> None:
        outcome = _make_outcome(
            gap=_make_gap(
                gap_type=GapType.OUTPUT_EMITTED_DURING_DISCONNECT,
                missed_line_count=3,
                missed_lines=(
                    "PASSED test_a",
                    "PASSED test_b",
                    "FAILED test_c",
                ),
                checkpoint_line_number=42,
                actual_resume_line_number=45,
            ),
            resumption_point=_make_resumption_point(line_number=45),
        )
        checkpoint = _make_checkpoint()
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        result = await prepare_transition(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="trans-gap-1",
            callback=collect,
        )

        assert result.outcome == TransitionOutcome.LIVE_MONITORING
        assert result.gap_events_replayed == 3
        assert len(collected) == 3  # gap events replayed
        # live_sequence_start should be after gap replay
        assert result.live_sequence_start > 42

    @pytest.mark.asyncio
    async def test_unusable_outcome_transitions_to_fresh_start(self) -> None:
        outcome = _make_outcome(
            is_usable=False,
            reason="Stream diverged completely",
            resumption_point=_make_resumption_point(
                line_number=0, is_valid=False, match_quality=MatchQuality.NO_MATCH
            ),
        )
        checkpoint = _make_checkpoint(last_parsed_line_number=0)
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        result = await prepare_transition(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="trans-fresh-1",
            callback=collect,
        )

        assert result.outcome == TransitionOutcome.FRESH_START
        assert result.phase == TransitionPhase.COMPLETE
        assert result.gap_events_replayed == 0
        assert result.live_sequence_start == 0

    @pytest.mark.asyncio
    async def test_result_is_frozen(self) -> None:
        outcome = _make_outcome()
        checkpoint = _make_checkpoint()

        async def noop(status: MonitorStatus) -> None:
            pass

        result = await prepare_transition(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="frozen-1",
            callback=noop,
        )

        with pytest.raises(AttributeError):
            result.outcome = TransitionOutcome.FRESH_START  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_diverged_stream_outcome(self) -> None:
        outcome = _make_outcome(
            gap=_make_gap(
                gap_type=GapType.STREAM_DIVERGED,
                missed_line_count=0,
                missed_lines=(),
                detail="Stream diverged: marker at 30 but expected 42",
            ),
        )
        checkpoint = _make_checkpoint()

        async def noop(status: MonitorStatus) -> None:
            pass

        result = await prepare_transition(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="diverge-1",
            callback=noop,
        )

        # Diverged stream is still usable if outcome says so
        assert result.outcome == TransitionOutcome.LIVE_MONITORING

    @pytest.mark.asyncio
    async def test_truncated_stream_unusable(self) -> None:
        outcome = _make_outcome(
            is_usable=False,
            gap=_make_gap(
                gap_type=GapType.STREAM_TRUNCATED,
                missed_line_count=0,
                missed_lines=(),
            ),
            resumption_point=_make_resumption_point(
                line_number=0, is_valid=False, match_quality=MatchQuality.NO_MATCH
            ),
        )
        checkpoint = _make_checkpoint(last_parsed_line_number=0)

        async def noop(status: MonitorStatus) -> None:
            pass

        result = await prepare_transition(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="trunc-1",
            callback=noop,
        )

        assert result.outcome == TransitionOutcome.FRESH_START


# ---------------------------------------------------------------------------
# Test: sequence continuity across transition
# ---------------------------------------------------------------------------


class TestSequenceContinuity:
    @pytest.mark.asyncio
    async def test_gap_replay_then_live_sequence_is_monotonic(self) -> None:
        """Verify that sequence numbers are monotonically increasing
        from gap replay through to the live monitoring start point.

        live_sequence_start is the current sequence position (the last
        assigned number). The PollingLoop auto-increments via with_output(),
        so the first live event will be at live_sequence_start + 1.
        This guarantees no duplication: gap events occupy sequences up to
        live_sequence_start, and live events start at live_sequence_start + 1.
        """
        outcome = _make_outcome(
            gap=_make_gap(
                gap_type=GapType.OUTPUT_EMITTED_DURING_DISCONNECT,
                missed_line_count=2,
                missed_lines=("missed_A", "missed_B"),
                checkpoint_line_number=10,
                actual_resume_line_number=12,
            ),
            resumption_point=_make_resumption_point(line_number=12),
        )
        checkpoint = _make_checkpoint(last_parsed_line_number=10)
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        result = await prepare_transition(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="seq-cont-1",
            callback=collect,
        )

        # Gap events should have been replayed
        assert len(collected) == 2

        # Gap events must have monotonically increasing sequences
        assert collected[1].sequence_number > collected[0].sequence_number

        # live_sequence_start equals the last gap event's sequence number.
        # The PollingLoop will call with_output() which auto-increments,
        # so the first live event will be at live_sequence_start + 1.
        last_gap_seq = collected[-1].sequence_number
        assert result.live_sequence_start == last_gap_seq

        # Initial status sequence must be >= resumption line number
        assert result.live_sequence_start >= 12

    @pytest.mark.asyncio
    async def test_no_gap_live_sequence_continues_from_resumption(self) -> None:
        outcome = _make_outcome(
            resumption_point=_make_resumption_point(line_number=100),
            gap=_make_gap(gap_type=GapType.NONE),
        )
        checkpoint = _make_checkpoint(last_parsed_line_number=100)

        async def noop(status: MonitorStatus) -> None:
            pass

        result = await prepare_transition(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="seq-no-gap-1",
            callback=noop,
        )

        # Live monitoring should continue from the resumption point
        assert result.live_sequence_start == 100


# ---------------------------------------------------------------------------
# Test: TransitionOutcome and TransitionPhase enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_transition_outcome_values(self) -> None:
        assert TransitionOutcome.LIVE_MONITORING.value == "live_monitoring"
        assert TransitionOutcome.FRESH_START.value == "fresh_start"

    def test_transition_phase_values(self) -> None:
        assert TransitionPhase.BUILDING_STATUS.value == "building_status"
        assert TransitionPhase.REPLAYING_GAP.value == "replaying_gap"
        assert TransitionPhase.COMPLETE.value == "complete"


# ---------------------------------------------------------------------------
# Test: never-raise contract on prepare_transition
# ---------------------------------------------------------------------------


class TestPrepareTransitionNeverRaise:
    @pytest.mark.asyncio
    async def test_empty_session_id_returns_error_result_not_exception(self) -> None:
        """prepare_transition has a never-raise contract. Empty session_id
        should return a TransitionResult with error, not raise ValueError."""
        outcome = _make_outcome()
        checkpoint = _make_checkpoint()

        async def noop(status: MonitorStatus) -> None:
            pass

        result = await prepare_transition(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="",
            callback=noop,
        )

        assert result.outcome == TransitionOutcome.FRESH_START
        assert result.error is not None
        assert "session_id" in result.error
        assert result.live_sequence_start == 0


# ---------------------------------------------------------------------------
# Test: CancelledError propagation during gap replay
# ---------------------------------------------------------------------------


class TestCancelledErrorPropagation:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_from_gap_replay(self) -> None:
        """asyncio.CancelledError should propagate through gap replay
        instead of being swallowed by the generic Exception handler."""
        gap = _make_gap(
            gap_type=GapType.OUTPUT_EMITTED_DURING_DISCONNECT,
            missed_line_count=3,
            missed_lines=("line_A", "line_B", "line_C"),
        )
        call_count = 0

        async def cancelling_callback(status: MonitorStatus) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise asyncio.CancelledError()

        initial = MonitorStatus(
            session_id="cancel-1",
            timestamp=_now(),
            sequence_number=10,
        )

        with pytest.raises(asyncio.CancelledError):
            await replay_gap_events(
                gap=gap,
                initial_status=initial,
                callback=cancelling_callback,
            )

        # Should have stopped at event 2, not continued to event 3
        assert call_count == 2


# ---------------------------------------------------------------------------
# Test: OutputPhase derived from checkpoint RunStatus
# ---------------------------------------------------------------------------


class TestOutputPhaseMapping:
    def test_running_checkpoint_produces_running_phase(self) -> None:
        outcome = _make_outcome()
        checkpoint = _make_checkpoint(status=RunStatus.RUNNING)

        status = build_initial_status(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="phase-running",
        )

        assert status.parsed_state.phase == OutputPhase.RUNNING

    def test_pending_approval_checkpoint_produces_setup_phase(self) -> None:
        outcome = _make_outcome()
        checkpoint = _make_checkpoint(status=RunStatus.PENDING_APPROVAL)

        status = build_initial_status(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="phase-pending",
        )

        assert status.parsed_state.phase == OutputPhase.SETUP

    def test_completed_checkpoint_produces_complete_phase(self) -> None:
        outcome = _make_outcome()
        checkpoint = _make_checkpoint(status=RunStatus.COMPLETED)

        status = build_initial_status(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="phase-complete",
        )

        assert status.parsed_state.phase == OutputPhase.COMPLETE

    def test_failed_checkpoint_produces_error_phase(self) -> None:
        outcome = _make_outcome()
        checkpoint = _make_checkpoint(status=RunStatus.FAILED)

        status = build_initial_status(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="phase-failed",
        )

        assert status.parsed_state.phase == OutputPhase.ERROR

    def test_idle_checkpoint_produces_unknown_phase(self) -> None:
        outcome = _make_outcome()
        checkpoint = _make_checkpoint(status=RunStatus.IDLE)

        status = build_initial_status(
            outcome=outcome,
            checkpoint=checkpoint,
            session_id="phase-idle",
        )

        assert status.parsed_state.phase == OutputPhase.UNKNOWN


# ---------------------------------------------------------------------------
# Test: batch yielding during large gap replay
# ---------------------------------------------------------------------------


class TestGapReplayBatching:
    @pytest.mark.asyncio
    async def test_large_gap_yields_between_batches(self) -> None:
        """When gap size exceeds batch_size, replay yields control
        to the event loop between batches."""
        lines = tuple(f"line_{i}" for i in range(120))
        gap = _make_gap(
            gap_type=GapType.OUTPUT_EMITTED_DURING_DISCONNECT,
            missed_line_count=120,
            missed_lines=lines,
        )
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        initial = MonitorStatus(
            session_id="batch-1",
            timestamp=_now(),
            sequence_number=0,
        )

        config = TransitionConfig(gap_replay_batch_size=50)
        result_status = await replay_gap_events(
            gap=gap,
            initial_status=initial,
            callback=collect,
            config=config,
        )

        # All 120 events should have been replayed
        assert len(collected) == 120
        assert result_status.sequence_number == 120

        # Sequences must be monotonically increasing
        for i in range(1, len(collected)):
            assert collected[i].sequence_number > collected[i - 1].sequence_number
