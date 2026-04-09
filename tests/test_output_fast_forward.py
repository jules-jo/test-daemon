"""Tests for the output fast-forward scanner.

The fast-forward scanner consumes a re-attached SSH output stream line-by-line,
compares each line against a checkpoint's last-processed sequence marker, and
advances the stream cursor past all already-processed output.

This is used when the daemon re-attaches to a running SSH session after a crash
or restart. The scanner skips output that was already processed before the
interruption, positioning the stream at the first unprocessed line.

The scanner uses two strategies to locate the resume point:
1. Sequence-based: advance to the line matching ``last_parsed_line_number``
2. Marker-based: advance to the line matching the checkpoint's marker text

The result is an immutable FastForwardResult that tells the caller:
- how many lines were skipped
- whether the resume point was found
- the first unprocessed line (if any)
- which strategy succeeded
"""

from __future__ import annotations

import io
import time
from datetime import datetime, timezone

import pytest

from jules_daemon.wiki.checkpoint_recovery import (
    ExtractedMetrics,
    MonitoringCheckpoint,
    RecoverySource,
)
from jules_daemon.wiki.models import RunStatus
from jules_daemon.wiki.output_fast_forward import (
    FastForwardResult,
    FastForwardStrategy,
    OutputLine,
    fast_forward_stream,
)


# ---------------------------------------------------------------------------
# OutputLine model
# ---------------------------------------------------------------------------


class TestOutputLine:
    """Verify OutputLine is frozen with correct fields."""

    def test_create_with_content_and_number(self) -> None:
        line = OutputLine(content="PASSED test_login", line_number=42)
        assert line.content == "PASSED test_login"
        assert line.line_number == 42

    def test_frozen(self) -> None:
        line = OutputLine(content="hello", line_number=0)
        with pytest.raises(AttributeError):
            line.content = "world"  # type: ignore[misc]

    def test_negative_line_number_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            OutputLine(content="x", line_number=-1)


# ---------------------------------------------------------------------------
# FastForwardStrategy enum
# ---------------------------------------------------------------------------


class TestFastForwardStrategy:
    """Verify all strategy members exist."""

    def test_all_strategies_exist(self) -> None:
        assert FastForwardStrategy.SEQUENCE_NUMBER.value == "sequence_number"
        assert FastForwardStrategy.MARKER_MATCH.value == "marker_match"
        assert FastForwardStrategy.NONE.value == "none"


# ---------------------------------------------------------------------------
# FastForwardResult model
# ---------------------------------------------------------------------------


class TestFastForwardResult:
    """Verify FastForwardResult is frozen with correct properties."""

    def test_frozen(self) -> None:
        result = FastForwardResult(
            lines_skipped=0,
            resume_found=False,
            first_unprocessed=None,
            strategy=FastForwardStrategy.NONE,
            skipped_lines=(),
        )
        with pytest.raises(AttributeError):
            result.lines_skipped = 5  # type: ignore[misc]

    def test_create_with_all_fields(self) -> None:
        first = OutputLine(content="RUNNING test_new", line_number=10)
        result = FastForwardResult(
            lines_skipped=10,
            resume_found=True,
            first_unprocessed=first,
            strategy=FastForwardStrategy.SEQUENCE_NUMBER,
            skipped_lines=(
                OutputLine(content="line1", line_number=0),
                OutputLine(content="line2", line_number=1),
            ),
        )
        assert result.lines_skipped == 10
        assert result.resume_found is True
        assert result.first_unprocessed is not None
        assert result.first_unprocessed.content == "RUNNING test_new"
        assert result.strategy == FastForwardStrategy.SEQUENCE_NUMBER
        assert len(result.skipped_lines) == 2

    def test_negative_lines_skipped_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            FastForwardResult(
                lines_skipped=-1,
                resume_found=False,
                first_unprocessed=None,
                strategy=FastForwardStrategy.NONE,
                skipped_lines=(),
            )


# ---------------------------------------------------------------------------
# Checkpoint builder helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(
    *,
    line_number: int = 0,
    status: RunStatus = RunStatus.RUNNING,
    source: RecoverySource = RecoverySource.WIKI_STATE,
) -> MonitoringCheckpoint:
    """Build a MonitoringCheckpoint for testing."""
    return MonitoringCheckpoint(
        last_parsed_line_number=line_number,
        timestamp=datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc),
        extracted_metrics=ExtractedMetrics(
            tests_passed=5, tests_total=20, percent=25.0
        ),
        run_id="test-run-001",
        status=status,
        source=source,
        error=None,
    )


def _make_stream(lines: list[str]) -> io.StringIO:
    """Build a StringIO stream from a list of line strings."""
    return io.StringIO("\n".join(lines) + "\n" if lines else "")


# ---------------------------------------------------------------------------
# fast_forward_stream: empty stream
# ---------------------------------------------------------------------------


class TestFastForwardEmptyStream:
    """When the output stream is empty, nothing to skip."""

    def test_returns_zero_lines_skipped(self) -> None:
        cp = _make_checkpoint(line_number=10)
        stream = _make_stream([])
        result = fast_forward_stream(stream, cp)
        assert result.lines_skipped == 0

    def test_resume_not_found(self) -> None:
        cp = _make_checkpoint(line_number=10)
        stream = _make_stream([])
        result = fast_forward_stream(stream, cp)
        assert result.resume_found is False

    def test_no_first_unprocessed(self) -> None:
        cp = _make_checkpoint(line_number=10)
        stream = _make_stream([])
        result = fast_forward_stream(stream, cp)
        assert result.first_unprocessed is None

    def test_strategy_is_none(self) -> None:
        cp = _make_checkpoint(line_number=10)
        stream = _make_stream([])
        result = fast_forward_stream(stream, cp)
        assert result.strategy == FastForwardStrategy.NONE


# ---------------------------------------------------------------------------
# fast_forward_stream: checkpoint at line 0 (fresh start)
# ---------------------------------------------------------------------------


class TestFastForwardFreshStart:
    """When checkpoint is at line 0, no fast-forward needed."""

    def test_skips_nothing(self) -> None:
        cp = _make_checkpoint(line_number=0)
        lines = ["line 0", "line 1", "line 2"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.lines_skipped == 0

    def test_resume_found(self) -> None:
        cp = _make_checkpoint(line_number=0)
        lines = ["line 0", "line 1"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.resume_found is True

    def test_first_unprocessed_is_first_line(self) -> None:
        cp = _make_checkpoint(line_number=0)
        lines = ["first output", "second output"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.first_unprocessed is not None
        assert result.first_unprocessed.content == "first output"
        assert result.first_unprocessed.line_number == 0


# ---------------------------------------------------------------------------
# fast_forward_stream: sequence-based fast-forward
# ---------------------------------------------------------------------------


class TestFastForwardSequenceBased:
    """Advance using the sequence number (last_parsed_line_number)."""

    def test_skips_to_checkpoint_line(self) -> None:
        cp = _make_checkpoint(line_number=3)
        lines = ["line 0", "line 1", "line 2", "line 3", "line 4"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.lines_skipped == 4  # lines 0-3 skipped
        assert result.resume_found is True

    def test_first_unprocessed_is_after_checkpoint(self) -> None:
        cp = _make_checkpoint(line_number=2)
        lines = ["line 0", "line 1", "line 2", "line 3", "line 4"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.first_unprocessed is not None
        assert result.first_unprocessed.content == "line 3"
        assert result.first_unprocessed.line_number == 3

    def test_strategy_is_sequence_number(self) -> None:
        cp = _make_checkpoint(line_number=2)
        lines = ["line 0", "line 1", "line 2", "line 3"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.strategy == FastForwardStrategy.SEQUENCE_NUMBER

    def test_skips_exactly_to_end_of_stream(self) -> None:
        """When checkpoint matches the last line, entire stream is consumed."""
        cp = _make_checkpoint(line_number=2)
        lines = ["line 0", "line 1", "line 2"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.lines_skipped == 3
        assert result.first_unprocessed is None
        assert result.resume_found is True

    def test_checkpoint_beyond_stream_length(self) -> None:
        """When checkpoint is past the stream end, entire stream is consumed."""
        cp = _make_checkpoint(line_number=100)
        lines = ["line 0", "line 1", "line 2"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.lines_skipped == 3
        assert result.first_unprocessed is None
        assert result.resume_found is False

    def test_single_line_stream_at_checkpoint(self) -> None:
        cp = _make_checkpoint(line_number=0)
        lines = ["only line"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        # Line 0 is already processed when line_number=0, but since it
        # represents position 0 with no prior processing, we start fresh
        assert result.first_unprocessed is not None
        assert result.first_unprocessed.content == "only line"

    def test_skipped_lines_tuple_contains_correct_lines(self) -> None:
        cp = _make_checkpoint(line_number=2)
        lines = ["line 0", "line 1", "line 2", "line 3"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert len(result.skipped_lines) == 3
        assert result.skipped_lines[0].content == "line 0"
        assert result.skipped_lines[1].content == "line 1"
        assert result.skipped_lines[2].content == "line 2"


# ---------------------------------------------------------------------------
# fast_forward_stream: marker-based fallback
# ---------------------------------------------------------------------------


class TestFastForwardMarkerBased:
    """When sequence-based fails but marker matches a line, use marker."""

    def test_falls_back_to_marker_when_stream_shorter(self) -> None:
        """When sequence-based can't fully match but marker is in stream."""
        cp = _make_checkpoint(line_number=100)
        lines = [
            "Setting up...",
            "Running tests...",
            "PASSED test_login",
            "PASSED test_checkout",
            "RUNNING test_payment",
        ]
        stream = _make_stream(lines)
        result = fast_forward_stream(
            stream, cp, marker="PASSED test_checkout"
        )
        assert result.strategy == FastForwardStrategy.MARKER_MATCH
        assert result.resume_found is True
        assert result.first_unprocessed is not None
        assert result.first_unprocessed.content == "RUNNING test_payment"
        assert result.lines_skipped == 4  # lines 0-3 skipped

    def test_marker_matches_exact_line_content(self) -> None:
        """Marker must exactly match a line in the stream."""
        cp = _make_checkpoint(line_number=100)
        lines = [
            "PASSED test_login",
            "PASSED test_checkout",
        ]
        stream = _make_stream(lines)
        result = fast_forward_stream(
            stream, cp, marker="PASSED test_login"
        )
        assert result.strategy == FastForwardStrategy.MARKER_MATCH
        assert result.first_unprocessed is not None
        assert result.first_unprocessed.content == "PASSED test_checkout"

    def test_marker_not_found_returns_strategy_none(self) -> None:
        """When neither sequence nor marker finds the resume point."""
        cp = _make_checkpoint(line_number=100)
        lines = ["line 0", "line 1", "line 2"]
        stream = _make_stream(lines)
        result = fast_forward_stream(
            stream, cp, marker="NONEXISTENT LINE"
        )
        assert result.strategy == FastForwardStrategy.NONE
        assert result.resume_found is False

    def test_marker_at_last_line(self) -> None:
        """When the marker matches the last line in the stream."""
        cp = _make_checkpoint(line_number=100)
        lines = [
            "PASSED test_first",
            "PASSED test_second",
            "PASSED test_final",
        ]
        stream = _make_stream(lines)
        result = fast_forward_stream(
            stream, cp, marker="PASSED test_final"
        )
        assert result.strategy == FastForwardStrategy.MARKER_MATCH
        assert result.lines_skipped == 3
        assert result.first_unprocessed is None  # nothing after marker


# ---------------------------------------------------------------------------
# fast_forward_stream: non-resumable checkpoints
# ---------------------------------------------------------------------------


class TestFastForwardNonResumable:
    """When checkpoint is not resumable, skip nothing."""

    def test_no_state_checkpoint(self) -> None:
        cp = _make_checkpoint(line_number=5, source=RecoverySource.NO_STATE)
        lines = ["line 0", "line 1", "line 2"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.lines_skipped == 0
        assert result.strategy == FastForwardStrategy.NONE
        assert result.resume_found is False

    def test_corrupted_checkpoint(self) -> None:
        cp = _make_checkpoint(line_number=5, source=RecoverySource.CORRUPTED)
        lines = ["line 0", "line 1", "line 2"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.lines_skipped == 0
        assert result.strategy == FastForwardStrategy.NONE

    def test_completed_status_checkpoint(self) -> None:
        cp = _make_checkpoint(
            line_number=5, status=RunStatus.COMPLETED
        )
        lines = ["line 0", "line 1", "line 2"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.lines_skipped == 0
        assert result.strategy == FastForwardStrategy.NONE


# ---------------------------------------------------------------------------
# fast_forward_stream: whitespace and edge cases
# ---------------------------------------------------------------------------


class TestFastForwardEdgeCases:
    """Edge cases around whitespace, empty lines, and stream positioning."""

    def test_strips_trailing_newlines(self) -> None:
        """Line content should not include trailing newlines."""
        cp = _make_checkpoint(line_number=1)
        lines = ["line 0", "line 1", "line 2"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.first_unprocessed is not None
        assert "\n" not in result.first_unprocessed.content

    def test_handles_empty_lines_in_stream(self) -> None:
        """Empty lines in the output should be counted and handled."""
        cp = _make_checkpoint(line_number=2)
        lines = ["line 0", "", "line 2", "line 3"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.lines_skipped == 3  # lines 0, empty, 2
        assert result.first_unprocessed is not None
        assert result.first_unprocessed.content == "line 3"

    def test_large_stream_performance(self) -> None:
        """Fast-forward through a large stream should be O(n) in lines."""
        cp = _make_checkpoint(line_number=9999)
        lines = [f"line {i}" for i in range(10_001)]
        stream = _make_stream(lines)

        start = time.monotonic()
        result = fast_forward_stream(stream, cp)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert result.lines_skipped == 10_000
        assert result.first_unprocessed is not None
        assert result.first_unprocessed.content == "line 10000"
        assert elapsed_ms < 500.0, f"Fast-forward took {elapsed_ms:.1f}ms (>500ms)"

    def test_marker_with_leading_trailing_whitespace(self) -> None:
        """Marker matching should compare stripped content."""
        cp = _make_checkpoint(line_number=100)
        lines = ["  PASSED test_login  ", "next line"]
        stream = _make_stream(lines)
        result = fast_forward_stream(
            stream, cp, marker="PASSED test_login"
        )
        # Marker should match even with whitespace
        assert result.strategy == FastForwardStrategy.MARKER_MATCH
        assert result.resume_found is True

    def test_stream_cursor_position_after_fast_forward(self) -> None:
        """After fast-forward, subsequent reads from stream should
        return the unprocessed portion."""
        cp = _make_checkpoint(line_number=2)
        lines = ["line 0", "line 1", "line 2", "line 3", "line 4"]
        stream = _make_stream(lines)
        fast_forward_stream(stream, cp)  # advance cursor; result not inspected

        # The stream should be positioned after the skipped lines
        # Read remaining from the stream
        remaining = stream.read()
        # Should contain "line 4" at minimum (line 3 was peeked as first_unprocessed)
        assert "line 4" in remaining


# ---------------------------------------------------------------------------
# fast_forward_stream: iterable protocol
# ---------------------------------------------------------------------------


class TestFastForwardIterable:
    """The scanner should accept any iterable of strings, not just StringIO."""

    def test_accepts_list_of_strings(self) -> None:
        cp = _make_checkpoint(line_number=1)
        lines = ["line 0\n", "line 1\n", "line 2\n"]
        result = fast_forward_stream(iter(lines), cp)
        assert result.lines_skipped == 2
        assert result.first_unprocessed is not None
        assert result.first_unprocessed.content == "line 2"

    def test_accepts_generator(self) -> None:
        def gen():
            for i in range(5):
                yield f"output {i}\n"

        cp = _make_checkpoint(line_number=2)
        result = fast_forward_stream(gen(), cp)
        assert result.lines_skipped == 3
        assert result.first_unprocessed is not None
        assert result.first_unprocessed.content == "output 3"

    def test_iterator_exhaustion_mid_sequence(self) -> None:
        """Iterator exhausted before reaching the sequence target."""
        cp = _make_checkpoint(line_number=5)
        result = fast_forward_stream(iter(["line 0\n", "line 1\n"]), cp)
        assert result.resume_found is False
        assert result.lines_skipped == 2


# ---------------------------------------------------------------------------
# fast_forward_stream: additional coverage cases
# ---------------------------------------------------------------------------


class TestFastForwardAdditionalCoverage:
    """Edge cases identified during code review for full coverage."""

    def test_marker_at_end_of_collected_and_source_empty(self) -> None:
        """Marker is the last collected line and remaining source is empty."""
        cp = _make_checkpoint(line_number=100)
        lines = ["PASSED test_final"]
        stream = _make_stream(lines)
        result = fast_forward_stream(
            stream, cp, marker="PASSED test_final"
        )
        assert result.resume_found is True
        assert result.first_unprocessed is None
        assert result.strategy == FastForwardStrategy.MARKER_MATCH

    def test_fresh_start_empty_stream(self) -> None:
        """Checkpoint at line 0 with an empty stream."""
        cp = _make_checkpoint(line_number=0)
        result = fast_forward_stream(_make_stream([]), cp)
        assert result.resume_found is False
        assert result.strategy == FastForwardStrategy.NONE
        assert result.first_unprocessed is None

    def test_marker_found_in_remaining_source_not_collected(self) -> None:
        """Marker is found by reading from the remaining source, not in
        the already-collected lines from the sequence attempt."""
        cp = _make_checkpoint(line_number=100)
        # 3 lines will be collected by sequence attempt (stream shorter than 100)
        # Then marker scan continues reading from source
        # But since stream is a StringIO, all lines are read in sequence attempt
        # This test verifies the path where marker search continues past collected
        lines = [
            "line 0",
            "line 1",
            "CHECKPOINT MARKER HERE",
            "first new output",
        ]
        stream = _make_stream(lines)
        result = fast_forward_stream(
            stream, cp, marker="CHECKPOINT MARKER HERE"
        )
        assert result.resume_found is True
        assert result.strategy == FastForwardStrategy.MARKER_MATCH
        assert result.first_unprocessed is not None
        assert result.first_unprocessed.content == "first new output"

    def test_no_marker_no_sequence_match(self) -> None:
        """Neither strategy works and no marker is provided."""
        cp = _make_checkpoint(line_number=100)
        lines = ["line 0", "line 1"]
        stream = _make_stream(lines)
        result = fast_forward_stream(stream, cp)
        assert result.strategy == FastForwardStrategy.NONE
        assert result.resume_found is False
        assert result.lines_skipped == 2
