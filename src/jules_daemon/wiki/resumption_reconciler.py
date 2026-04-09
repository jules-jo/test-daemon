"""Resumption state reconciliation for crash recovery and CLI reconnection.

Compares a recovered MonitoringCheckpoint (from the wiki persistence layer)
against the actual SSH output stream content to detect gaps, partial-line
mismatches, and stream divergence. Produces a validated ResumptionPoint with
gap metadata that the monitoring loop uses to decide where to resume.

Edge cases handled:
- Exact match: checkpoint marker matches stream line at expected position
- Partial line at disconnect boundary: marker is a prefix or suffix of
  the actual line (disconnect happened mid-line)
- Output emitted during disconnect: marker found at a later position
  than expected, indicating lines were emitted while the CLI was gone
- Stream shorter than checkpoint: process was restarted or output truncated
- Stream diverged: marker found at an earlier position than expected
- No marker available: line-number-only validation
- Empty stream: handled as fresh start or truncated depending on checkpoint
- Duplicate markers: prefer the occurrence closest to checkpoint position

Algorithm:
1. Gate: if checkpoint is not resumable, return immediately
2. Gate: if checkpoint is at line 0, return fresh start
3. If no marker: validate stream length >= checkpoint position
4. If marker provided:
   a. Check exact match at checkpoint position
   b. Check partial match at checkpoint position
   c. Search entire stream for marker (gap/divergence detection)
5. Build ResumptionPoint and GapMetadata from findings

This module never raises. All error conditions are captured in the
returned ReconciliationOutcome's reason and gap fields.

Usage:
    from jules_daemon.wiki.checkpoint_recovery import recover_monitoring_checkpoint
    from jules_daemon.wiki.resumption_reconciler import reconcile_resumption_state

    checkpoint = recover_monitoring_checkpoint(wiki_root)
    stream_lines = list(ssh_output_buffer)
    outcome = reconcile_resumption_state(
        checkpoint=checkpoint,
        stream_lines=stream_lines,
        checkpoint_marker=checkpoint_marker,
    )
    if outcome.is_usable:
        # Resume from outcome.resumption_point.line_number
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from jules_daemon.wiki.checkpoint_recovery import (
    MonitoringCheckpoint,
    RecoverySource,
)
from jules_daemon.wiki.models import RunStatus

__all__ = [
    "GapMetadata",
    "GapType",
    "MatchQuality",
    "ReconciliationOutcome",
    "ResumptionPoint",
    "reconcile_resumption_state",
]

logger = logging.getLogger(__name__)

# Statuses that indicate the run is active and can be resumed
_RESUMABLE_STATUSES = frozenset({RunStatus.RUNNING, RunStatus.PENDING_APPROVAL})


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MatchQuality(Enum):
    """How well the checkpoint marker matched the stream content.

    EXACT: Marker matches the line at the checkpoint position exactly.
    PARTIAL_LINE: Marker is a prefix or suffix of the actual line
        (disconnect happened mid-line output).
    LINE_NUMBER_ONLY: No marker was available; the stream was long enough
        to reach the checkpoint position, so line number is trusted.
    NO_MATCH: Marker was not found at or near the expected position.
    """

    EXACT = "exact"
    PARTIAL_LINE = "partial_line"
    LINE_NUMBER_ONLY = "line_number_only"
    NO_MATCH = "no_match"


class GapType(Enum):
    """Classification of the gap between checkpoint and actual stream.

    NONE: No gap detected; checkpoint and stream are in sync.
    PARTIAL_LINE_AT_BOUNDARY: The disconnect happened mid-line. The
        checkpoint's marker is a substring of the actual line content.
    OUTPUT_EMITTED_DURING_DISCONNECT: New output was emitted between
        disconnect and reconnect. The marker appears at a later
        position than the checkpoint expected.
    STREAM_TRUNCATED: The stream has fewer lines than the checkpoint
        expected. The process may have been restarted.
    STREAM_DIVERGED: The marker appears at an earlier position than
        expected, indicating the stream content has changed (e.g.,
        the process was restarted and produced similar output).
    """

    NONE = "none"
    PARTIAL_LINE_AT_BOUNDARY = "partial_line_at_boundary"
    OUTPUT_EMITTED_DURING_DISCONNECT = "output_emitted_during_disconnect"
    STREAM_TRUNCATED = "stream_truncated"
    STREAM_DIVERGED = "stream_diverged"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResumptionPoint:
    """Validated position in the stream where processing should resume.

    Attributes:
        line_number: 0-based line number to resume from. This is the
            position of the last processed line (the monitoring loop
            should process from line_number + 1 onward).
        marker: The matched marker text at the resumption point.
        is_valid: True if the resumption point is trustworthy.
        match_quality: How the resumption point was determined.
    """

    line_number: int
    marker: str
    is_valid: bool
    match_quality: MatchQuality

    def __post_init__(self) -> None:
        if self.line_number < 0:
            raise ValueError("line_number must not be negative")


@dataclass(frozen=True)
class GapMetadata:
    """Description of any gap between the checkpoint and actual stream.

    Attributes:
        gap_type: Classification of the detected gap.
        missed_line_count: Number of lines that were missed (emitted
            during disconnect but not processed). 0 when no gap.
        missed_lines: Tuple of the actual missed line contents.
            May be empty even when missed_line_count > 0 if lines
            could not be captured.
        checkpoint_line_number: The line number from the checkpoint.
        actual_resume_line_number: The actual line number where the
            marker was found (or where resumption should happen).
        detail: Human-readable description of the gap.
    """

    gap_type: GapType
    missed_line_count: int
    missed_lines: tuple[str, ...]
    checkpoint_line_number: int
    actual_resume_line_number: int
    detail: str

    def __post_init__(self) -> None:
        if self.missed_line_count < 0:
            raise ValueError("missed_line_count must not be negative")

    @property
    def has_gap(self) -> bool:
        """True if a gap was detected between checkpoint and stream."""
        return self.gap_type != GapType.NONE


@dataclass(frozen=True)
class ReconciliationOutcome:
    """Complete result of the resumption state reconciliation.

    Attributes:
        resumption_point: Validated position to resume from.
        gap: Metadata about any detected gap.
        checkpoint_run_id: The run ID from the checkpoint, for audit.
        is_usable: True if the daemon can safely resume from this point.
        reason: Human-readable explanation of the reconciliation outcome.
    """

    resumption_point: ResumptionPoint
    gap: GapMetadata
    checkpoint_run_id: str
    is_usable: bool
    reason: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_line(raw: str) -> str:
    """Strip trailing whitespace for comparison.

    Leading whitespace is preserved because indentation may be meaningful
    in test output.
    """
    return raw.rstrip()


def _is_checkpoint_resumable(checkpoint: MonitoringCheckpoint) -> bool:
    """Check if the checkpoint represents a resumable state."""
    if checkpoint.source != RecoverySource.WIKI_STATE:
        return False
    return checkpoint.status in _RESUMABLE_STATUSES


def _build_no_resume_outcome(
    checkpoint: MonitoringCheckpoint,
    reason: str,
) -> ReconciliationOutcome:
    """Build an outcome for non-resumable checkpoints."""
    return ReconciliationOutcome(
        resumption_point=ResumptionPoint(
            line_number=0,
            marker="",
            is_valid=False,
            match_quality=MatchQuality.NO_MATCH,
        ),
        gap=GapMetadata(
            gap_type=GapType.NONE,
            missed_line_count=0,
            missed_lines=(),
            checkpoint_line_number=checkpoint.last_parsed_line_number,
            actual_resume_line_number=0,
            detail=reason,
        ),
        checkpoint_run_id=checkpoint.run_id,
        is_usable=False,
        reason=reason,
    )


def _check_exact_match(
    stream_line: str,
    marker: str,
) -> bool:
    """Check if the stream line exactly matches the marker after normalization."""
    return _normalize_line(stream_line) == _normalize_line(marker)


def _check_partial_match(
    stream_line: str,
    marker: str,
) -> bool:
    """Check if the marker is a substring of the stream line (partial-line match).

    Handles both prefix and suffix matches:
    - Prefix: marker is the start of the actual line (disconnect cut the end)
    - Suffix: marker is the end of the actual line (disconnect cut the start)
    """
    normalized_stream = _normalize_line(stream_line)
    normalized_marker = _normalize_line(marker)

    if not normalized_marker:
        return False

    return (
        normalized_marker in normalized_stream
        and normalized_marker != normalized_stream
    )


def _find_marker_in_stream(
    stream_lines: list[str],
    marker: str,
    preferred_position: int,
) -> int | None:
    """Search the stream for the marker, preferring the position closest
    to the checkpoint's expected position.

    If the marker appears at the preferred_position, return that immediately.
    Otherwise, find all occurrences and return the one closest to preferred.

    Args:
        stream_lines: The stream content as a list of lines.
        marker: The marker text to search for.
        preferred_position: The checkpoint's expected line number.

    Returns:
        The 0-based line number of the best match, or None if not found.
    """
    normalized_marker = _normalize_line(marker)
    if not normalized_marker:
        return None

    # Check preferred position first (fast path)
    if preferred_position < len(stream_lines):
        if _normalize_line(stream_lines[preferred_position]) == normalized_marker:
            return preferred_position

    # Check for partial match at preferred position
    if preferred_position < len(stream_lines):
        if _check_partial_match(stream_lines[preferred_position], marker):
            return preferred_position

    # Collect all exact match positions
    exact_positions: list[int] = []
    for idx, line in enumerate(stream_lines):
        if _normalize_line(line) == normalized_marker:
            exact_positions.append(idx)

    if exact_positions:
        # Return the position closest to preferred
        return min(exact_positions, key=lambda p: abs(p - preferred_position))

    # Collect all partial match positions
    partial_positions: list[int] = []
    for idx, line in enumerate(stream_lines):
        if _check_partial_match(line, marker):
            partial_positions.append(idx)

    if partial_positions:
        return min(partial_positions, key=lambda p: abs(p - preferred_position))

    return None


def _determine_match_quality(
    stream_lines: list[str],
    found_position: int,
    marker: str,
) -> MatchQuality:
    """Determine the match quality between the marker and the stream line."""
    if found_position >= len(stream_lines):
        return MatchQuality.NO_MATCH

    stream_line = stream_lines[found_position]

    if _check_exact_match(stream_line, marker):
        return MatchQuality.EXACT

    if _check_partial_match(stream_line, marker):
        return MatchQuality.PARTIAL_LINE

    return MatchQuality.NO_MATCH


def _classify_gap(
    checkpoint_position: int,
    found_position: int,
    match_quality: MatchQuality,
) -> GapType:
    """Classify the type of gap between checkpoint and found position.

    Args:
        checkpoint_position: Where the checkpoint expected the marker.
        found_position: Where the marker was actually found.
        match_quality: How the marker matched the stream line.
    """
    if match_quality == MatchQuality.PARTIAL_LINE:
        if found_position == checkpoint_position:
            return GapType.PARTIAL_LINE_AT_BOUNDARY
        # Partial match at a different position implies both divergence
        # and partial line -- classify based on position relationship
        if found_position > checkpoint_position:
            return GapType.OUTPUT_EMITTED_DURING_DISCONNECT
        return GapType.STREAM_DIVERGED

    if found_position == checkpoint_position:
        return GapType.NONE

    if found_position > checkpoint_position:
        return GapType.OUTPUT_EMITTED_DURING_DISCONNECT

    # found_position < checkpoint_position
    return GapType.STREAM_DIVERGED


def _extract_missed_lines(
    stream_lines: list[str],
    checkpoint_position: int,
    found_position: int,
) -> tuple[str, ...]:
    """Extract lines that were missed between checkpoint and found position.

    Only applicable when found_position > checkpoint_position (gap exists).
    Returns the lines between the checkpoint position and the found position.
    """
    if found_position <= checkpoint_position:
        return ()

    start = checkpoint_position
    end = found_position
    return tuple(
        _normalize_line(stream_lines[i])
        for i in range(start, end)
        if i < len(stream_lines)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reconcile_resumption_state(
    *,
    checkpoint: MonitoringCheckpoint,
    stream_lines: list[str],
    checkpoint_marker: str,
) -> ReconciliationOutcome:
    """Reconcile the recovered checkpoint against the actual stream content.

    Compares the checkpoint's last known position and marker against the
    SSH output stream to detect gaps, partial-line mismatches, and stream
    divergence. Returns a validated ResumptionPoint with gap metadata.

    This function never raises. All error conditions are captured in the
    returned ReconciliationOutcome.

    Args:
        checkpoint: The monitoring checkpoint recovered from the wiki.
        stream_lines: The actual SSH output stream content as a list of
            lines. Each element is one line of output (may include
            trailing whitespace/newlines).
        checkpoint_marker: The marker text from the checkpoint (typically
            the last output line recorded in the wiki). Empty string
            means no marker is available.

    Returns:
        ReconciliationOutcome with validated resumption point and gap
        metadata. Never raises.
    """
    # -- Gate 1: Non-resumable checkpoints --
    if not _is_checkpoint_resumable(checkpoint):
        reason = (
            f"Checkpoint is not resumable "
            f"(source={checkpoint.source.value}, "
            f"status={checkpoint.status.value})"
        )
        logger.info("Reconciliation: %s", reason)
        return _build_no_resume_outcome(checkpoint, reason)

    cp_line = checkpoint.last_parsed_line_number
    stream_length = len(stream_lines)
    marker = checkpoint_marker.strip()

    # -- Gate 2: Fresh start (line 0) --
    if cp_line == 0:
        logger.info(
            "Reconciliation: checkpoint at line 0 -- fresh start "
            "(stream has %d lines)",
            stream_length,
        )
        return ReconciliationOutcome(
            resumption_point=ResumptionPoint(
                line_number=0,
                marker="",
                is_valid=True,
                match_quality=MatchQuality.EXACT,
            ),
            gap=GapMetadata(
                gap_type=GapType.NONE,
                missed_line_count=0,
                missed_lines=(),
                checkpoint_line_number=0,
                actual_resume_line_number=0,
                detail="Fresh start -- no reconciliation needed",
            ),
            checkpoint_run_id=checkpoint.run_id,
            is_usable=True,
            reason="Checkpoint at line 0 -- starting fresh",
        )

    # -- Path A: No marker available --
    if not marker:
        return _reconcile_without_marker(checkpoint, stream_lines)

    # -- Path B: Marker available -- full reconciliation --
    return _reconcile_with_marker(checkpoint, stream_lines, marker)


# ---------------------------------------------------------------------------
# Path A: No marker
# ---------------------------------------------------------------------------


def _reconcile_without_marker(
    checkpoint: MonitoringCheckpoint,
    stream_lines: list[str],
) -> ReconciliationOutcome:
    """Reconcile using line count only (no marker available).

    Trusts the checkpoint line number if the stream is long enough.
    """
    cp_line = checkpoint.last_parsed_line_number
    stream_length = len(stream_lines)

    if stream_length > cp_line:
        # Stream is long enough -- trust the line number
        logger.info(
            "Reconciliation: no marker, stream long enough (%d > %d) "
            "-- trusting line number",
            stream_length,
            cp_line,
        )
        return ReconciliationOutcome(
            resumption_point=ResumptionPoint(
                line_number=cp_line,
                marker="",
                is_valid=True,
                match_quality=MatchQuality.LINE_NUMBER_ONLY,
            ),
            gap=GapMetadata(
                gap_type=GapType.NONE,
                missed_line_count=0,
                missed_lines=(),
                checkpoint_line_number=cp_line,
                actual_resume_line_number=cp_line,
                detail=(
                    f"No marker available; stream length {stream_length} "
                    f"exceeds checkpoint position {cp_line}"
                ),
            ),
            checkpoint_run_id=checkpoint.run_id,
            is_usable=True,
            reason=(
                f"No marker; trusting line number {cp_line} "
                f"(stream has {stream_length} lines)"
            ),
        )

    # Stream too short -- cannot validate
    reason = (
        f"Stream is shorter than checkpoint position "
        f"({stream_length} lines vs checkpoint at line {cp_line})"
    )
    logger.warning("Reconciliation: %s", reason)

    return ReconciliationOutcome(
        resumption_point=ResumptionPoint(
            line_number=cp_line,
            marker="",
            is_valid=False,
            match_quality=MatchQuality.NO_MATCH,
        ),
        gap=GapMetadata(
            gap_type=GapType.STREAM_TRUNCATED,
            missed_line_count=0,
            missed_lines=(),
            checkpoint_line_number=cp_line,
            actual_resume_line_number=stream_length,
            detail=reason,
        ),
        checkpoint_run_id=checkpoint.run_id,
        is_usable=False,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Path B: Marker-based reconciliation
# ---------------------------------------------------------------------------


def _reconcile_with_marker(
    checkpoint: MonitoringCheckpoint,
    stream_lines: list[str],
    marker: str,
) -> ReconciliationOutcome:
    """Reconcile using marker text against stream content.

    Searches for the marker at the expected position first, then falls back
    to a full stream scan. Classifies the relationship between expected
    and actual positions to determine gap type.
    """
    cp_line = checkpoint.last_parsed_line_number
    stream_length = len(stream_lines)

    # Find the marker in the stream
    found_position = _find_marker_in_stream(stream_lines, marker, cp_line)

    # -- Marker not found anywhere --
    if found_position is None:
        # Check if stream is also too short
        if stream_length <= cp_line:
            reason = (
                f"Stream is shorter than checkpoint and marker not found "
                f"({stream_length} lines vs checkpoint at line {cp_line})"
            )
            gap_type = GapType.STREAM_TRUNCATED
        else:
            reason = (
                f"Marker not found in stream "
                f"(stream has {stream_length} lines, "
                f"checkpoint at line {cp_line})"
            )
            gap_type = GapType.STREAM_TRUNCATED

        logger.warning("Reconciliation: %s", reason)

        return ReconciliationOutcome(
            resumption_point=ResumptionPoint(
                line_number=cp_line,
                marker=marker,
                is_valid=False,
                match_quality=MatchQuality.NO_MATCH,
            ),
            gap=GapMetadata(
                gap_type=gap_type,
                missed_line_count=0,
                missed_lines=(),
                checkpoint_line_number=cp_line,
                actual_resume_line_number=min(stream_length, cp_line),
                detail=reason,
            ),
            checkpoint_run_id=checkpoint.run_id,
            is_usable=False,
            reason=reason,
        )

    # -- Marker found -- classify the match --
    match_quality = _determine_match_quality(stream_lines, found_position, marker)
    gap_type = _classify_gap(cp_line, found_position, match_quality)
    missed_lines = _extract_missed_lines(stream_lines, cp_line, found_position)
    missed_count = len(missed_lines)

    # Build detail description
    if gap_type == GapType.NONE:
        detail = f"Exact match at checkpoint position {cp_line}"
    elif gap_type == GapType.PARTIAL_LINE_AT_BOUNDARY:
        actual_line = _normalize_line(stream_lines[found_position])
        detail = (
            f"Partial-line match at position {found_position}: "
            f"marker {marker!r} is a substring of {actual_line!r}"
        )
    elif gap_type == GapType.OUTPUT_EMITTED_DURING_DISCONNECT:
        detail = (
            f"{missed_count} lines emitted during disconnect "
            f"(checkpoint at {cp_line}, marker found at {found_position})"
        )
    elif gap_type == GapType.STREAM_DIVERGED:
        detail = (
            f"Stream diverged: marker found at position {found_position} "
            f"but checkpoint expected position {cp_line}"
        )
    else:
        detail = f"Marker found at position {found_position}"

    # The outcome is usable if we found the marker
    reason = (
        f"Marker found at line {found_position} "
        f"(checkpoint expected {cp_line}, "
        f"quality={match_quality.value}, gap={gap_type.value})"
    )

    logger.info("Reconciliation: %s", reason)

    return ReconciliationOutcome(
        resumption_point=ResumptionPoint(
            line_number=found_position,
            marker=marker,
            is_valid=True,
            match_quality=match_quality,
        ),
        gap=GapMetadata(
            gap_type=gap_type,
            missed_line_count=missed_count,
            missed_lines=missed_lines,
            checkpoint_line_number=cp_line,
            actual_resume_line_number=found_position,
            detail=detail,
        ),
        checkpoint_run_id=checkpoint.run_id,
        is_usable=True,
        reason=reason,
    )
