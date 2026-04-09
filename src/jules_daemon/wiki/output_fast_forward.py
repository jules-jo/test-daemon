"""Output fast-forward scanner for SSH stream re-attachment.

When the daemon re-attaches to a running SSH session after a crash or restart,
the output stream contains all output from the beginning of the remote process.
The fast-forward scanner consumes lines that have already been processed
(according to the monitoring checkpoint), advancing the stream cursor to the
first unprocessed line.

Two strategies are tried in order:
1. **Sequence-based**: Advance past ``last_parsed_line_number`` lines. This is
   the primary strategy when the checkpoint has a valid line number > 0.
2. **Marker-based**: If sequence-based fails (stream is shorter than expected),
   scan for the checkpoint's marker text in the stream. This handles cases
   where the output was truncated or reformatted.

If neither strategy finds the resume point, the result indicates no match and
the caller decides whether to start fresh or skip the entire stream.

The scanner accepts any line-iterable (file-like objects, generators, lists).
Output lines are stripped of trailing whitespace before comparison.

Usage:
    from jules_daemon.wiki.checkpoint_recovery import recover_monitoring_checkpoint
    from jules_daemon.wiki.output_fast_forward import fast_forward_stream

    checkpoint = recover_monitoring_checkpoint(wiki_root)
    stream = ssh_channel.makefile("r")
    result = fast_forward_stream(stream, checkpoint)

    if result.resume_found:
        # Continue processing from result.first_unprocessed
        ...
    else:
        # Handle: resume point not found
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import IO, Iterator, Union

from jules_daemon.wiki.checkpoint_recovery import (
    MonitoringCheckpoint,
    RecoverySource,
)
from jules_daemon.wiki.models import RunStatus

__all__ = [
    "FastForwardResult",
    "FastForwardStrategy",
    "OutputLine",
    "fast_forward_stream",
]

logger = logging.getLogger(__name__)

# Statuses that indicate the run is active and can be resumed
_RESUMABLE_STATUSES = frozenset({RunStatus.RUNNING, RunStatus.PENDING_APPROVAL})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FastForwardStrategy(Enum):
    """Which strategy located the resume point.

    SEQUENCE_NUMBER: Advanced past the checkpoint's last_parsed_line_number.
    MARKER_MATCH: Found the checkpoint's marker text in the stream.
    NONE: Neither strategy found the resume point.
    """

    SEQUENCE_NUMBER = "sequence_number"
    MARKER_MATCH = "marker_match"
    NONE = "none"


@dataclass(frozen=True)
class OutputLine:
    """A single line from the SSH output stream.

    Immutable record of a line's content and its 0-based position in the
    stream. Content is stripped of trailing whitespace.

    Attributes:
        content: The line text, stripped of trailing whitespace.
        line_number: 0-based position in the stream.
    """

    content: str
    line_number: int

    def __post_init__(self) -> None:
        if self.line_number < 0:
            raise ValueError("line_number must not be negative")


@dataclass(frozen=True)
class FastForwardResult:
    """Immutable result of the fast-forward scan.

    Tells the caller how many lines were skipped, whether the resume point
    was found, and which line to process next.

    Attributes:
        lines_skipped: Number of lines consumed and skipped.
        resume_found: True if the resume point was located in the stream.
        first_unprocessed: The first line after the resume point, or None
            if the stream was fully consumed.
        strategy: Which strategy located the resume point.
        skipped_lines: Tuple of all skipped OutputLine records. Useful for
            audit or debugging but may be large for long streams.
    """

    lines_skipped: int
    resume_found: bool
    first_unprocessed: OutputLine | None
    strategy: FastForwardStrategy
    skipped_lines: tuple[OutputLine, ...]

    def __post_init__(self) -> None:
        if self.lines_skipped < 0:
            raise ValueError("lines_skipped must not be negative")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Type alias for anything that yields lines (file-like or iterator)
LineSource = Union[IO[str], Iterator[str]]


def _is_checkpoint_resumable(checkpoint: MonitoringCheckpoint) -> bool:
    """Check if the checkpoint represents a resumable state.

    A checkpoint is only usable for fast-forward when:
    1. It was loaded from a valid wiki state (not corrupted or missing)
    2. The run status is active (RUNNING or PENDING_APPROVAL)
    """
    if checkpoint.source != RecoverySource.WIKI_STATE:
        return False
    return checkpoint.status in _RESUMABLE_STATUSES


def _read_line(source: LineSource) -> str | None:
    """Read a single line from the source.

    Returns None when the source is exhausted. Handles both
    file-like objects (with readline()) and iterators (with next()).
    """
    if hasattr(source, "readline"):
        line = source.readline()  # type: ignore[union-attr]
        if line == "":
            return None
        return line
    else:
        try:
            return next(source)  # type: ignore[arg-type]
        except StopIteration:
            return None


def _strip_line(raw: str) -> str:
    """Strip trailing whitespace (including newlines) from a line.

    Note: only trailing whitespace is stripped (rstrip). Leading whitespace
    is preserved in OutputLine.content because indentation may be meaningful
    in test output. Marker comparison uses full .strip() separately to
    handle lines with leading whitespace that still match the marker text.
    """
    return raw.rstrip()


def _build_no_skip_result() -> FastForwardResult:
    """Build a result for when no fast-forward is needed or possible."""
    return FastForwardResult(
        lines_skipped=0,
        resume_found=False,
        first_unprocessed=None,
        strategy=FastForwardStrategy.NONE,
        skipped_lines=(),
    )


def _build_fresh_start_result(
    first_line: str,
) -> FastForwardResult:
    """Build a result for a fresh start (checkpoint at line 0)."""
    return FastForwardResult(
        lines_skipped=0,
        resume_found=True,
        first_unprocessed=OutputLine(content=first_line, line_number=0),
        strategy=FastForwardStrategy.SEQUENCE_NUMBER,
        skipped_lines=(),
    )


# ---------------------------------------------------------------------------
# Sequence-based fast-forward
# ---------------------------------------------------------------------------


def _fast_forward_by_sequence(
    source: LineSource,
    target_line_number: int,
) -> tuple[list[OutputLine], OutputLine | None, bool]:
    """Advance the stream past ``target_line_number`` lines.

    Reads lines numbered 0 through ``target_line_number`` (inclusive),
    collecting them as skipped. Then reads one more line as the
    first unprocessed line.

    Args:
        source: The line source to read from.
        target_line_number: The 0-based line number of the last processed
            line. Lines 0..target_line_number are skipped.

    Returns:
        Tuple of (skipped_lines, first_unprocessed, fully_matched).
        - skipped_lines: List of OutputLine for lines 0..target_line_number.
        - first_unprocessed: OutputLine for the line after the target, or
          None if the stream ended at or before the target.
        - fully_matched: True if we successfully advanced past all
          target_line_number + 1 lines (stream had enough lines).
    """
    skipped: list[OutputLine] = []
    current_number = 0

    # Skip lines 0 through target_line_number
    while current_number <= target_line_number:
        raw = _read_line(source)
        if raw is None:
            # Stream exhausted before reaching the target
            return skipped, None, False

        content = _strip_line(raw)
        skipped.append(OutputLine(content=content, line_number=current_number))
        current_number += 1

    # Read the first unprocessed line
    raw = _read_line(source)
    if raw is None:
        # Stream ended exactly at the target -- no unprocessed lines
        return skipped, None, True

    content = _strip_line(raw)
    first_unprocessed = OutputLine(content=content, line_number=current_number)
    return skipped, first_unprocessed, True


# ---------------------------------------------------------------------------
# Marker-based fallback
# ---------------------------------------------------------------------------


def _fast_forward_by_marker(
    collected_lines: list[OutputLine],
    remaining_source: LineSource,
    marker: str,
    start_line_number: int,
) -> tuple[list[OutputLine], OutputLine | None, bool]:
    """Scan collected and remaining lines for the marker text.

    Searches through already-read lines first, then continues reading
    from the source. When the marker is found, the next line is returned
    as the first unprocessed line.

    Args:
        collected_lines: Lines already read during sequence-based attempt.
        remaining_source: The source to continue reading from.
        marker: The marker text to search for (stripped comparison).
        start_line_number: The next line number to assign when reading
            more lines from the source.

    Returns:
        Tuple of (skipped_lines, first_unprocessed, found).
    """
    stripped_marker = marker.strip()

    # Search through already-collected lines
    for idx, line in enumerate(collected_lines):
        if line.content.strip() == stripped_marker:
            # Found the marker in collected lines
            skipped = collected_lines[: idx + 1]

            # The first unprocessed is the next collected line, or
            # the first line from the remaining source
            if idx + 1 < len(collected_lines):
                first_unprocessed = collected_lines[idx + 1]
                return skipped, first_unprocessed, True
            else:
                # Need to read one more line from the source
                raw = _read_line(remaining_source)
                if raw is None:
                    return skipped, None, True

                content = _strip_line(raw)
                first_unprocessed = OutputLine(
                    content=content, line_number=start_line_number
                )
                return skipped, first_unprocessed, True

    # Marker not in collected lines -- continue reading from source
    all_skipped = list(collected_lines)
    current_number = start_line_number

    while True:
        raw = _read_line(remaining_source)
        if raw is None:
            # Stream exhausted, marker not found
            return all_skipped, None, False

        content = _strip_line(raw)
        line = OutputLine(content=content, line_number=current_number)
        all_skipped.append(line)
        current_number += 1

        if content.strip() == stripped_marker:
            # Found the marker -- read the next line
            raw = _read_line(remaining_source)
            if raw is None:
                return all_skipped, None, True

            next_content = _strip_line(raw)
            first_unprocessed = OutputLine(
                content=next_content, line_number=current_number
            )
            return all_skipped, first_unprocessed, True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fast_forward_stream(
    source: LineSource,
    checkpoint: MonitoringCheckpoint,
    *,
    marker: str = "",
) -> FastForwardResult:
    """Fast-forward an output stream past already-processed lines.

    Consumes the re-attached SSH output stream line-by-line, compares
    each line against the checkpoint's last-processed sequence marker,
    and advances the stream cursor past all already-processed output.

    Strategy order:
    1. If checkpoint is not resumable (wrong source or terminal status),
       return immediately with no skipping.
    2. If checkpoint's last_parsed_line_number is 0, peek the first line
       and return it as first_unprocessed (fresh start).
    3. Sequence-based: skip lines 0..last_parsed_line_number.
    4. If sequence-based consumed the entire stream without finding enough
       lines, fall back to marker-based search using the provided marker
       text (typically progress.last_output_line from the wiki state).
    5. If neither strategy finds the resume point, return with
       strategy=NONE and resume_found=False.

    This function never raises. All error paths return a safe result.

    Args:
        source: Line-iterable source (file-like object, iterator, or
            generator). Each iteration yields one line of output.
        checkpoint: The monitoring checkpoint recovered from the wiki.
        marker: Optional marker text to use for marker-based fallback.
            Typically the progress.last_output_line from the wiki state.
            Empty string disables marker-based fallback.

    Returns:
        FastForwardResult with the scan outcome. Never raises.
    """
    # Gate: non-resumable checkpoints get no fast-forward
    if not _is_checkpoint_resumable(checkpoint):
        logger.info(
            "Checkpoint not resumable (source=%s, status=%s) -- no fast-forward",
            checkpoint.source.value,
            checkpoint.status.value,
        )
        return _build_no_skip_result()

    target = checkpoint.last_parsed_line_number

    # Gate: fresh start (line 0 = no lines processed yet)
    if target == 0:
        raw = _read_line(source)
        if raw is None:
            # Empty stream
            return FastForwardResult(
                lines_skipped=0,
                resume_found=False,
                first_unprocessed=None,
                strategy=FastForwardStrategy.NONE,
                skipped_lines=(),
            )

        content = _strip_line(raw)
        logger.info(
            "Checkpoint at line 0 -- starting fresh with first line: %s",
            content[:80],
        )
        return _build_fresh_start_result(content)

    # Strategy 1: Sequence-based fast-forward
    skipped, first_unprocessed, fully_matched = _fast_forward_by_sequence(
        source, target
    )

    if fully_matched:
        logger.info(
            "Sequence-based fast-forward: skipped %d lines to line %d",
            len(skipped),
            target,
        )
        return FastForwardResult(
            lines_skipped=len(skipped),
            resume_found=True,
            first_unprocessed=first_unprocessed,
            strategy=FastForwardStrategy.SEQUENCE_NUMBER,
            skipped_lines=tuple(skipped),
        )

    # Strategy 2: Marker-based fallback
    # The marker is typically the progress.last_output_line from the wiki
    # state, provided by the caller. If no marker was given, skip this.
    if marker:
        logger.info(
            "Sequence-based fast-forward failed at line %d/%d -- "
            "trying marker-based with %r",
            len(skipped),
            target,
            marker[:80],
        )

        # Use already-collected lines + continue reading from source
        start_line_number = len(skipped)
        (
            marker_skipped,
            marker_first,
            marker_found,
        ) = _fast_forward_by_marker(
            skipped, source, marker, start_line_number
        )

        if marker_found:
            logger.info(
                "Marker-based fast-forward: found marker after %d lines",
                len(marker_skipped),
            )
            return FastForwardResult(
                lines_skipped=len(marker_skipped),
                resume_found=True,
                first_unprocessed=marker_first,
                strategy=FastForwardStrategy.MARKER_MATCH,
                skipped_lines=tuple(marker_skipped),
            )

    # Neither strategy found the resume point
    logger.warning(
        "Fast-forward failed: sequence target=%d, stream had %d lines, "
        "marker=%r not found",
        target,
        len(skipped),
        marker[:80] if marker else "",
    )

    return FastForwardResult(
        lines_skipped=len(skipped),
        resume_found=False,
        first_unprocessed=None,
        strategy=FastForwardStrategy.NONE,
        skipped_lines=tuple(skipped),
    )
