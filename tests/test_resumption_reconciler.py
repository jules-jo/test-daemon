"""Tests for resumption state reconciliation.

Verifies that the reconciler:
- Detects exact match between checkpoint marker and stream content
- Detects partial-line mismatches at disconnect boundary
- Detects gaps (output emitted between disconnect and reconnect)
- Handles stream shorter than checkpoint (truncated/reset)
- Handles empty streams
- Handles non-resumable checkpoints (graceful no-op)
- Produces a validated ResumptionPoint with GapMetadata
- Never raises (all errors captured in result)
- Returns immutable (frozen) results
"""

from datetime import datetime, timezone

import pytest

from jules_daemon.wiki.checkpoint_recovery import (
    ExtractedMetrics,
    MonitoringCheckpoint,
    RecoverySource,
)
from jules_daemon.wiki.models import RunStatus
from jules_daemon.wiki.resumption_reconciler import (
    GapMetadata,
    GapType,
    MatchQuality,
    ReconciliationOutcome,
    ResumptionPoint,
    reconcile_resumption_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(
    *,
    last_parsed_line_number: int = 10,
    source: RecoverySource = RecoverySource.WIKI_STATE,
    status: RunStatus = RunStatus.RUNNING,
    run_id: str = "test-run-001",
    tests_passed: int = 5,
    tests_failed: int = 0,
    tests_total: int = 20,
    error: str | None = None,
) -> MonitoringCheckpoint:
    """Build a MonitoringCheckpoint for testing."""
    return MonitoringCheckpoint(
        last_parsed_line_number=last_parsed_line_number,
        timestamp=datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc),
        extracted_metrics=ExtractedMetrics(
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_total=tests_total,
        ),
        run_id=run_id,
        status=status,
        source=source,
        error=error,
    )


def _make_stream(lines: list[str]) -> list[str]:
    """Build a list of output lines (simulating SSH output)."""
    return lines


# ---------------------------------------------------------------------------
# ResumptionPoint model tests
# ---------------------------------------------------------------------------


class TestResumptionPoint:
    def test_create_minimal(self) -> None:
        point = ResumptionPoint(
            line_number=10,
            marker="PASSED test_foo",
            is_valid=True,
            match_quality=MatchQuality.EXACT,
        )
        assert point.line_number == 10
        assert point.marker == "PASSED test_foo"
        assert point.is_valid is True
        assert point.match_quality == MatchQuality.EXACT

    def test_frozen(self) -> None:
        point = ResumptionPoint(
            line_number=10,
            marker="test",
            is_valid=True,
            match_quality=MatchQuality.EXACT,
        )
        with pytest.raises(AttributeError):
            point.line_number = 20  # type: ignore[misc]

    def test_negative_line_number_raises(self) -> None:
        with pytest.raises(ValueError, match="line_number must not be negative"):
            ResumptionPoint(
                line_number=-1,
                marker="",
                is_valid=True,
                match_quality=MatchQuality.EXACT,
            )


# ---------------------------------------------------------------------------
# GapMetadata model tests
# ---------------------------------------------------------------------------


class TestGapMetadata:
    def test_create_no_gap(self) -> None:
        gap = GapMetadata(
            gap_type=GapType.NONE,
            missed_line_count=0,
            missed_lines=(),
            checkpoint_line_number=10,
            actual_resume_line_number=10,
            detail="No gap detected",
        )
        assert gap.gap_type == GapType.NONE
        assert gap.has_gap is False

    def test_create_with_gap(self) -> None:
        gap = GapMetadata(
            gap_type=GapType.OUTPUT_EMITTED_DURING_DISCONNECT,
            missed_line_count=3,
            missed_lines=("line1", "line2", "line3"),
            checkpoint_line_number=10,
            actual_resume_line_number=13,
            detail="3 lines emitted during disconnect",
        )
        assert gap.has_gap is True
        assert gap.missed_line_count == 3

    def test_frozen(self) -> None:
        gap = GapMetadata(
            gap_type=GapType.NONE,
            missed_line_count=0,
            missed_lines=(),
            checkpoint_line_number=0,
            actual_resume_line_number=0,
            detail="",
        )
        with pytest.raises(AttributeError):
            gap.missed_line_count = 5  # type: ignore[misc]

    def test_negative_missed_line_count_raises(self) -> None:
        with pytest.raises(ValueError, match="missed_line_count must not be negative"):
            GapMetadata(
                gap_type=GapType.NONE,
                missed_line_count=-1,
                missed_lines=(),
                checkpoint_line_number=0,
                actual_resume_line_number=0,
                detail="",
            )


# ---------------------------------------------------------------------------
# ReconciliationOutcome model tests
# ---------------------------------------------------------------------------


class TestReconciliationOutcome:
    def test_create_valid(self) -> None:
        point = ResumptionPoint(
            line_number=10,
            marker="PASSED test_foo",
            is_valid=True,
            match_quality=MatchQuality.EXACT,
        )
        gap = GapMetadata(
            gap_type=GapType.NONE,
            missed_line_count=0,
            missed_lines=(),
            checkpoint_line_number=10,
            actual_resume_line_number=10,
            detail="No gap",
        )
        outcome = ReconciliationOutcome(
            resumption_point=point,
            gap=gap,
            checkpoint_run_id="test-run-001",
            is_usable=True,
            reason="Exact match at checkpoint position",
        )
        assert outcome.is_usable is True
        assert outcome.resumption_point.line_number == 10

    def test_frozen(self) -> None:
        point = ResumptionPoint(
            line_number=0,
            marker="",
            is_valid=False,
            match_quality=MatchQuality.NO_MATCH,
        )
        gap = GapMetadata(
            gap_type=GapType.NONE,
            missed_line_count=0,
            missed_lines=(),
            checkpoint_line_number=0,
            actual_resume_line_number=0,
            detail="",
        )
        outcome = ReconciliationOutcome(
            resumption_point=point,
            gap=gap,
            checkpoint_run_id="",
            is_usable=False,
            reason="test",
        )
        with pytest.raises(AttributeError):
            outcome.is_usable = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# reconcile_resumption_state: non-resumable checkpoints
# ---------------------------------------------------------------------------


class TestNonResumableCheckpoints:
    """Non-resumable checkpoints (wrong source, terminal status) return
    immediately with is_usable=False."""

    def test_no_state_checkpoint(self) -> None:
        cp = _make_checkpoint(source=RecoverySource.NO_STATE)
        stream = _make_stream(["line0", "line1"])
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="",
        )
        assert outcome.is_usable is False
        assert "not resumable" in outcome.reason.lower()

    def test_corrupted_checkpoint(self) -> None:
        cp = _make_checkpoint(
            source=RecoverySource.CORRUPTED,
            error="YAML parse error",
        )
        stream = _make_stream(["line0"])
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="",
        )
        assert outcome.is_usable is False

    def test_terminal_status_completed(self) -> None:
        cp = _make_checkpoint(status=RunStatus.COMPLETED)
        stream = _make_stream(["line0"])
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="",
        )
        assert outcome.is_usable is False

    def test_terminal_status_failed(self) -> None:
        cp = _make_checkpoint(status=RunStatus.FAILED)
        stream = _make_stream([])
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="",
        )
        assert outcome.is_usable is False

    def test_idle_status(self) -> None:
        cp = _make_checkpoint(status=RunStatus.IDLE)
        stream = _make_stream([])
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="",
        )
        assert outcome.is_usable is False

    def test_gap_type_is_none_for_non_resumable(self) -> None:
        cp = _make_checkpoint(source=RecoverySource.NO_STATE)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=[],
            checkpoint_marker="",
        )
        assert outcome.gap.gap_type == GapType.NONE


# ---------------------------------------------------------------------------
# reconcile_resumption_state: fresh start (line 0)
# ---------------------------------------------------------------------------


class TestFreshStartCheckpoint:
    """When checkpoint is at line 0, there is nothing to reconcile."""

    def test_fresh_start_with_empty_stream(self) -> None:
        cp = _make_checkpoint(last_parsed_line_number=0)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=[],
            checkpoint_marker="",
        )
        assert outcome.is_usable is True
        assert outcome.resumption_point.line_number == 0
        assert outcome.gap.gap_type == GapType.NONE

    def test_fresh_start_with_populated_stream(self) -> None:
        cp = _make_checkpoint(last_parsed_line_number=0)
        stream = _make_stream(["line0", "line1", "line2"])
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="",
        )
        assert outcome.is_usable is True
        assert outcome.resumption_point.line_number == 0
        assert outcome.resumption_point.match_quality == MatchQuality.EXACT


# ---------------------------------------------------------------------------
# reconcile_resumption_state: exact match
# ---------------------------------------------------------------------------


class TestExactMatch:
    """When the checkpoint marker matches the stream line exactly."""

    def test_exact_match_at_expected_position(self) -> None:
        stream = [
            "collecting tests...",
            "test_login PASSED",
            "test_checkout PASSED",
            "test_payment PASSED",
            "test_refund PASSED",
            "test_shipping PASSED",
        ]
        cp = _make_checkpoint(last_parsed_line_number=3)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="test_payment PASSED",
        )
        assert outcome.is_usable is True
        assert outcome.resumption_point.is_valid is True
        assert outcome.resumption_point.match_quality == MatchQuality.EXACT
        assert outcome.resumption_point.line_number == 3
        assert outcome.gap.gap_type == GapType.NONE
        assert outcome.gap.missed_line_count == 0

    def test_exact_match_preserves_run_id(self) -> None:
        stream = ["line0", "PASSED test_foo", "line2"]
        cp = _make_checkpoint(
            last_parsed_line_number=1,
            run_id="my-run-xyz",
        )
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_foo",
        )
        assert outcome.checkpoint_run_id == "my-run-xyz"

    def test_exact_match_with_trailing_whitespace(self) -> None:
        """Trailing whitespace should be stripped for comparison."""
        stream = ["line0", "PASSED test_foo  \n", "line2"]
        cp = _make_checkpoint(last_parsed_line_number=1)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_foo",
        )
        assert outcome.resumption_point.match_quality == MatchQuality.EXACT
        assert outcome.is_usable is True


# ---------------------------------------------------------------------------
# reconcile_resumption_state: partial-line at disconnect boundary
# ---------------------------------------------------------------------------


class TestPartialLineAtBoundary:
    """When the checkpoint marker is a prefix or suffix of the actual line
    (disconnect happened mid-line)."""

    def test_partial_prefix_match(self) -> None:
        """Checkpoint captured truncated line (prefix of actual line)."""
        stream = [
            "line0",
            "line1",
            "PASSED test_payment -- 0.5s",
            "line3",
        ]
        cp = _make_checkpoint(last_parsed_line_number=2)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_payment",
        )
        assert outcome.is_usable is True
        assert outcome.resumption_point.match_quality == MatchQuality.PARTIAL_LINE
        assert outcome.resumption_point.line_number == 2

    def test_partial_suffix_match(self) -> None:
        """Checkpoint captured only the end of the line."""
        stream = [
            "line0",
            "line1",
            "collecting... PASSED test_checkout",
            "line3",
        ]
        cp = _make_checkpoint(last_parsed_line_number=2)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_checkout",
        )
        assert outcome.is_usable is True
        assert outcome.resumption_point.match_quality == MatchQuality.PARTIAL_LINE

    def test_partial_match_gap_type_is_partial(self) -> None:
        stream = ["line0", "PASSED test_pay -- timing info", "line2"]
        cp = _make_checkpoint(last_parsed_line_number=1)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_pay",
        )
        assert outcome.gap.gap_type == GapType.PARTIAL_LINE_AT_BOUNDARY


# ---------------------------------------------------------------------------
# reconcile_resumption_state: gap detection (output during disconnect)
# ---------------------------------------------------------------------------


class TestGapDetection:
    """When lines were emitted between disconnect and reconnect, causing
    the marker to appear at a later position in the stream."""

    def test_marker_found_later_in_stream(self) -> None:
        """Marker is at a higher line number than checkpoint expected."""
        stream = [
            "line0",
            "line1",
            "line2",
            "missed_during_disconnect_1",
            "missed_during_disconnect_2",
            "PASSED test_payment",
            "line6",
        ]
        cp = _make_checkpoint(last_parsed_line_number=3)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_payment",
        )
        assert outcome.is_usable is True
        assert outcome.gap.gap_type == GapType.OUTPUT_EMITTED_DURING_DISCONNECT
        assert outcome.gap.missed_line_count == 2
        assert "missed_during_disconnect_1" in outcome.gap.missed_lines
        assert "missed_during_disconnect_2" in outcome.gap.missed_lines
        assert outcome.resumption_point.line_number == 5

    def test_gap_metadata_records_positions(self) -> None:
        stream = [
            "line0",
            "line1",
            "extra_line",
            "PASSED test_foo",
            "line4",
        ]
        cp = _make_checkpoint(last_parsed_line_number=2)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_foo",
        )
        assert outcome.gap.checkpoint_line_number == 2
        assert outcome.gap.actual_resume_line_number == 3


# ---------------------------------------------------------------------------
# reconcile_resumption_state: stream shorter than checkpoint
# ---------------------------------------------------------------------------


class TestStreamShorterThanCheckpoint:
    """When the stream has fewer lines than the checkpoint expected
    (e.g., process was restarted, output was truncated)."""

    def test_stream_too_short_no_marker(self) -> None:
        stream = ["line0", "line1"]  # Only 2 lines, checkpoint at 10
        cp = _make_checkpoint(last_parsed_line_number=10)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_foo",
        )
        assert outcome.is_usable is False
        assert outcome.gap.gap_type == GapType.STREAM_TRUNCATED
        assert "truncated" in outcome.reason.lower() or "shorter" in outcome.reason.lower()

    def test_stream_exactly_at_checkpoint_but_no_marker(self) -> None:
        """Stream has enough lines but marker doesn't match."""
        stream = ["line0", "line1", "line2", "WRONG CONTENT"]
        cp = _make_checkpoint(last_parsed_line_number=3)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_foo",
        )
        # Marker not found anywhere -- not usable
        assert outcome.is_usable is False
        assert outcome.resumption_point.match_quality == MatchQuality.NO_MATCH


# ---------------------------------------------------------------------------
# reconcile_resumption_state: empty marker
# ---------------------------------------------------------------------------


class TestEmptyMarker:
    """When no marker is available, reconciliation uses line count only."""

    def test_no_marker_stream_long_enough(self) -> None:
        """Without a marker, trust the line number if stream is long enough."""
        stream = ["line0", "line1", "line2", "line3", "line4"]
        cp = _make_checkpoint(last_parsed_line_number=3)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="",
        )
        assert outcome.is_usable is True
        assert outcome.resumption_point.line_number == 3
        assert outcome.resumption_point.match_quality == MatchQuality.LINE_NUMBER_ONLY

    def test_no_marker_stream_too_short(self) -> None:
        stream = ["line0", "line1"]
        cp = _make_checkpoint(last_parsed_line_number=5)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="",
        )
        assert outcome.is_usable is False
        assert outcome.gap.gap_type == GapType.STREAM_TRUNCATED


# ---------------------------------------------------------------------------
# reconcile_resumption_state: marker found earlier than expected
# ---------------------------------------------------------------------------


class TestMarkerEarlierThanExpected:
    """When the marker appears at a lower line number than the checkpoint
    expected. This can indicate the stream was reformatted or restarted."""

    def test_marker_before_checkpoint_position(self) -> None:
        stream = [
            "PASSED test_payment",
            "line1",
            "line2",
            "line3",
        ]
        cp = _make_checkpoint(last_parsed_line_number=3)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_payment",
        )
        # Found at position 0 instead of 3 -- indicates mismatch
        assert outcome.gap.gap_type == GapType.STREAM_DIVERGED
        # It's still usable because we found the marker
        assert outcome.is_usable is True
        assert outcome.resumption_point.line_number == 0

    def test_marker_at_first_line_checkpoint_at_ten(self) -> None:
        stream = [
            "PASSED test_payment",
            "more output",
        ]
        cp = _make_checkpoint(last_parsed_line_number=10)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_payment",
        )
        assert outcome.gap.gap_type == GapType.STREAM_DIVERGED


# ---------------------------------------------------------------------------
# reconcile_resumption_state: PENDING_APPROVAL status
# ---------------------------------------------------------------------------


class TestPendingApprovalStatus:
    """PENDING_APPROVAL is resumable (daemon re-prompts for confirmation)."""

    def test_pending_approval_always_resumable(self) -> None:
        cp = _make_checkpoint(
            status=RunStatus.PENDING_APPROVAL,
            last_parsed_line_number=0,
        )
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=[],
            checkpoint_marker="",
        )
        assert outcome.is_usable is True


# ---------------------------------------------------------------------------
# reconcile_resumption_state: never raises
# ---------------------------------------------------------------------------


class TestNeverRaises:
    """The reconciler should never raise exceptions."""

    def test_none_stream_lines(self) -> None:
        """Passing empty list should work fine."""
        cp = _make_checkpoint()
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=[],
            checkpoint_marker="PASSED test_foo",
        )
        # Stream is too short -- not usable
        assert outcome.is_usable is False

    def test_very_large_line_number(self) -> None:
        cp = _make_checkpoint(last_parsed_line_number=999999)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=["line0"],
            checkpoint_marker="",
        )
        assert outcome.is_usable is False


# ---------------------------------------------------------------------------
# reconcile_resumption_state: duplicate marker handling
# ---------------------------------------------------------------------------


class TestDuplicateMarkers:
    """When the marker text appears multiple times in the stream,
    prefer the occurrence closest to the checkpoint position."""

    def test_multiple_occurrences_prefers_nearest(self) -> None:
        stream = [
            "PASSED test_foo",
            "other output",
            "PASSED test_foo",
            "more output",
            "PASSED test_foo",
        ]
        cp = _make_checkpoint(last_parsed_line_number=2)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_foo",
        )
        # Should prefer the occurrence at position 2 (exact match at checkpoint)
        assert outcome.resumption_point.line_number == 2
        assert outcome.resumption_point.match_quality == MatchQuality.EXACT


# ---------------------------------------------------------------------------
# reconcile_resumption_state: integration with checkpoint_recovery types
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# reconcile_resumption_state: partial match at non-preferred position
# ---------------------------------------------------------------------------


class TestPartialMatchNotAtCheckpoint:
    """When the marker is a partial match but found at a position different
    from the checkpoint expected position."""

    def test_partial_match_later_in_stream(self) -> None:
        """Partial match at a later position -> output emitted during disconnect."""
        stream = [
            "line0",
            "line1",
            "line2",
            "extra line emitted",
            "PASSED test_foo -- timing info",
            "line5",
        ]
        cp = _make_checkpoint(last_parsed_line_number=2)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_foo",
        )
        assert outcome.is_usable is True
        assert outcome.gap.gap_type == GapType.OUTPUT_EMITTED_DURING_DISCONNECT

    def test_partial_match_earlier_in_stream(self) -> None:
        """Partial match at an earlier position -> stream diverged."""
        stream = [
            "PASSED test_foo -- extra timing",
            "line1",
            "line2",
            "line3",
            "line4",
        ]
        cp = _make_checkpoint(last_parsed_line_number=4)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_foo",
        )
        assert outcome.is_usable is True
        assert outcome.gap.gap_type == GapType.STREAM_DIVERGED


# ---------------------------------------------------------------------------
# reconcile_resumption_state: empty marker with _find_marker_in_stream
# ---------------------------------------------------------------------------


class TestFindMarkerEdgeCases:
    """Edge cases for _find_marker_in_stream internal function."""

    def test_empty_marker_returns_none_via_reconcile(self) -> None:
        """Verifies the empty-marker path through reconcile."""
        cp = _make_checkpoint(last_parsed_line_number=1)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=["line0", "line1", "line2"],
            checkpoint_marker="   ",  # whitespace-only marker
        )
        # Whitespace-only marker -> treated as empty -> line-number-only
        assert outcome.resumption_point.match_quality == MatchQuality.LINE_NUMBER_ONLY

    def test_marker_not_found_stream_has_content(self) -> None:
        """Marker not found when stream has content but different text."""
        stream = ["apple", "banana", "cherry", "date"]
        cp = _make_checkpoint(last_parsed_line_number=2)
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="PASSED test_xyz",
        )
        assert outcome.is_usable is False
        assert outcome.resumption_point.match_quality == MatchQuality.NO_MATCH


# ---------------------------------------------------------------------------
# Integration with checkpoint_recovery types
# ---------------------------------------------------------------------------


class TestIntegrationWithCheckpointRecovery:
    """Verify the reconciler works with real MonitoringCheckpoint instances."""

    def test_roundtrip_with_real_checkpoint(self) -> None:
        cp = MonitoringCheckpoint(
            last_parsed_line_number=5,
            timestamp=datetime(2026, 4, 9, 12, 30, 0, tzinfo=timezone.utc),
            extracted_metrics=ExtractedMetrics(
                tests_passed=3,
                tests_failed=0,
                tests_skipped=0,
                tests_total=10,
                percent=30.0,
            ),
            run_id="integration-test-run",
            status=RunStatus.RUNNING,
            source=RecoverySource.WIKI_STATE,
            error=None,
        )
        stream = [
            "collecting...",
            "test_a PASSED",
            "test_b PASSED",
            "test_c PASSED",
            "test_d RUNNING",
            "test_d PASSED",
            "test_e RUNNING",
        ]
        outcome = reconcile_resumption_state(
            checkpoint=cp,
            stream_lines=stream,
            checkpoint_marker="test_d PASSED",
        )
        assert outcome.is_usable is True
        assert outcome.checkpoint_run_id == "integration-test-run"
        assert outcome.resumption_point.line_number == 5
        assert outcome.resumption_point.match_quality == MatchQuality.EXACT
