"""Checkpoint extraction logic for crash recovery and run resumption.

Reads the saved run state from the wiki persistence layer and determines
the last completed checkpoint: test index, execution phase, and marker.

The checkpoint tells the daemon exactly where it left off, enabling:
- Crash recovery: resume monitoring from the last known progress point
- Status display: show the user where their test run stands
- Decision logic: determine if a run can be resumed or must restart

The test_index is 0-based: it is the index of the last fully processed
test (passed + failed + skipped - 1). When no tests have been processed,
test_index is 0.

The phase is derived from the RunStatus and progress state:
- IDLE -> NOT_STARTED
- PENDING_APPROVAL -> PENDING_APPROVAL
- RUNNING + no progress -> SETUP
- RUNNING + progress > 0 -> RUNNING
- COMPLETED -> COMPLETE
- FAILED -> FAILED
- CANCELLED -> CANCELLED

This module is a read-only extraction layer. It reads from the wiki but
never writes. It never raises -- all error conditions are captured in
the returned Checkpoint's source and error fields.

Usage:
    from pathlib import Path
    from jules_daemon.wiki.checkpoint_extractor import extract_checkpoint

    cp = extract_checkpoint(Path("wiki"))
    if cp.is_resumable:
        # Resume from cp.test_index in cp.phase
        ...
    else:
        # Start fresh or report final status
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import yaml

from jules_daemon.wiki import current_run
from jules_daemon.wiki.models import CurrentRun, RunStatus

__all__ = [
    "Checkpoint",
    "CheckpointPhase",
    "CheckpointSource",
    "extract_checkpoint",
]

logger = logging.getLogger(__name__)


# -- Enums --


class CheckpointPhase(Enum):
    """Execution phase at the time of the checkpoint.

    Maps from RunStatus + progress analysis to a phase that describes
    what the daemon was doing when the checkpoint was captured.
    """

    NOT_STARTED = "not_started"
    PENDING_APPROVAL = "pending_approval"
    SETUP = "setup"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CheckpointSource(Enum):
    """How the checkpoint was resolved.

    WIKI_STATE: Successfully extracted from the current-run wiki file.
    NO_STATE: No wiki state file found (fresh daemon start).
    CORRUPTED: Wiki state file exists but could not be parsed.
    """

    WIKI_STATE = "wiki_state"
    NO_STATE = "no_state"
    CORRUPTED = "corrupted"


# -- Status groupings --


_RESUMABLE_STATUSES = frozenset({RunStatus.RUNNING, RunStatus.PENDING_APPROVAL})


# -- Checkpoint model --


@dataclass(frozen=True)
class Checkpoint:
    """Extracted checkpoint from the wiki persistence layer.

    Immutable snapshot of the last completed checkpoint. The daemon uses
    this to decide whether and where to resume a test run.

    Attributes:
        test_index: 0-based index of the last processed test. Computed as
            (tests_passed + tests_failed + tests_skipped - 1), clamped to
            a minimum of 0. When no tests have been processed, this is 0.
            Note: test_index == 0 is ambiguous -- it means either "no tests
            done" or "exactly one test done". Callers must check
            tests_completed to distinguish the two cases.
        phase: Execution phase at the time of the checkpoint.
        marker: Human-readable label for the checkpoint. Typically the
            last output line from the test runner (e.g., "PASSED test_login").
        tests_passed: Count of passed tests at checkpoint time.
        tests_failed: Count of failed tests at checkpoint time.
        tests_skipped: Count of skipped tests at checkpoint time.
        tests_total: Total number of tests in the suite (0 if unknown).
        percent: Completion percentage (0.0 to 100.0).
        checkpoint_at: UTC timestamp of the last progress update (None if
            no checkpoint was recorded).
        run_id: Unique identifier for the run (empty if no run).
        status: The RunStatus from the wiki record.
        source: How this checkpoint was resolved (wiki, no_state, corrupted).
        error: Error description if the run failed or the file was corrupted.
    """

    test_index: int
    phase: CheckpointPhase
    marker: str
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    tests_total: int
    percent: float
    checkpoint_at: datetime | None
    run_id: str
    status: RunStatus
    source: CheckpointSource
    error: str | None

    @property
    def tests_completed(self) -> int:
        """Total number of tests that have been fully processed."""
        return self.tests_passed + self.tests_failed + self.tests_skipped

    @property
    def is_resumable(self) -> bool:
        """True if the daemon can resume from this checkpoint.

        A checkpoint is resumable when:
        1. It was loaded from a valid wiki state (not corrupted/missing)
        2. The run is in an active state (RUNNING or PENDING_APPROVAL)
        """
        if self.source != CheckpointSource.WIKI_STATE:
            return False
        return self.status in _RESUMABLE_STATUSES


# -- Phase derivation --


def _derive_phase(run: CurrentRun) -> CheckpointPhase:
    """Map a CurrentRun to the appropriate CheckpointPhase.

    Decision logic:
    - IDLE -> NOT_STARTED
    - PENDING_APPROVAL -> PENDING_APPROVAL
    - RUNNING with no completed tests and 0% -> SETUP
    - RUNNING with any completed tests or progress > 0 -> RUNNING
    - COMPLETED -> COMPLETE
    - FAILED -> FAILED
    - CANCELLED -> CANCELLED

    Args:
        run: The deserialized current run state.

    Returns:
        The checkpoint phase corresponding to the run state.
    """
    if run.status == RunStatus.IDLE:
        return CheckpointPhase.NOT_STARTED

    if run.status == RunStatus.PENDING_APPROVAL:
        return CheckpointPhase.PENDING_APPROVAL

    if run.status == RunStatus.RUNNING:
        prog = run.progress
        completed = prog.tests_passed + prog.tests_failed + prog.tests_skipped
        if completed == 0 and prog.percent == 0.0:
            return CheckpointPhase.SETUP
        return CheckpointPhase.RUNNING

    if run.status == RunStatus.COMPLETED:
        return CheckpointPhase.COMPLETE

    if run.status == RunStatus.FAILED:
        return CheckpointPhase.FAILED

    if run.status == RunStatus.CANCELLED:
        return CheckpointPhase.CANCELLED

    # Defensive: unknown status -> NOT_STARTED. This branch fires if a
    # new RunStatus member is added without updating this function.
    logger.error(
        "Unknown RunStatus %r in _derive_phase -- defaulting to NOT_STARTED",
        run.status,
    )
    return CheckpointPhase.NOT_STARTED


# -- Test index computation --


def _compute_test_index(run: CurrentRun) -> int:
    """Compute the 0-based index of the last completed test.

    The index is (tests_passed + tests_failed + tests_skipped - 1),
    clamped to a minimum of 0. When no tests have been processed, the
    index is 0 (meaning "start from the beginning").

    Args:
        run: The deserialized current run state.

    Returns:
        Non-negative integer representing the last completed test index.
    """
    prog = run.progress
    completed = prog.tests_passed + prog.tests_failed + prog.tests_skipped
    return max(0, completed - 1)


# -- Marker extraction --


def _extract_marker(run: CurrentRun) -> str:
    """Extract a human-readable marker from the run state.

    Uses the last_output_line from progress as the checkpoint marker.
    This is typically something like "PASSED test_checkout_flow" or
    "FAILED test_payment_flow".

    Args:
        run: The deserialized current run state.

    Returns:
        The marker string, or empty string if no marker is available.
    """
    return run.progress.last_output_line


# -- Checkpoint builders --


def _build_from_run(run: CurrentRun) -> Checkpoint:
    """Build a Checkpoint from a successfully parsed CurrentRun.

    Args:
        run: The deserialized current run state.

    Returns:
        Checkpoint with all fields populated from the run.
    """
    return Checkpoint(
        test_index=_compute_test_index(run),
        phase=_derive_phase(run),
        marker=_extract_marker(run),
        tests_passed=run.progress.tests_passed,
        tests_failed=run.progress.tests_failed,
        tests_skipped=run.progress.tests_skipped,
        tests_total=run.progress.tests_total,
        percent=run.progress.percent,
        checkpoint_at=run.progress.checkpoint_at,
        run_id=run.run_id,
        status=run.status,
        source=CheckpointSource.WIKI_STATE,
        error=run.error,
    )


def _build_empty(
    source: CheckpointSource,
    error: str | None = None,
) -> Checkpoint:
    """Build a safe empty Checkpoint for missing/corrupted file cases.

    Args:
        source: How the checkpoint was resolved (NO_STATE or CORRUPTED).
        error: Error description if the file was corrupted.

    Returns:
        Checkpoint with all fields at their zero/empty defaults.
    """
    return Checkpoint(
        test_index=0,
        phase=CheckpointPhase.NOT_STARTED,
        marker="",
        tests_passed=0,
        tests_failed=0,
        tests_skipped=0,
        tests_total=0,
        percent=0.0,
        checkpoint_at=None,
        run_id="",
        status=RunStatus.IDLE,
        source=source,
        error=error,
    )


# -- Public API --


def extract_checkpoint(wiki_root: Path) -> Checkpoint:
    """Extract the last completed checkpoint from the wiki state.

    Reads the current-run wiki file, parses the YAML frontmatter, and
    returns a Checkpoint with the extracted test index, phase, marker,
    and all progress fields.

    This function never raises. All error conditions are captured in the
    returned Checkpoint's source and error fields:
    - No wiki file: source=NO_STATE, all fields at zero/empty defaults
    - Corrupted file: source=CORRUPTED, error contains the parse message
    - Valid file: source=WIKI_STATE, all fields populated from the record

    The checkpoint includes:
    - test_index: 0-based index of the last completed test
    - phase: execution phase derived from RunStatus + progress analysis
    - marker: human-readable label (last output line from test runner)
    - is_resumable: whether the daemon can resume from this checkpoint

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        Checkpoint with extracted state. Never raises.
    """
    file_path = current_run.file_path(wiki_root)

    # Case 1: No wiki file exists -- fresh daemon start
    if not file_path.exists():
        logger.info(
            "No current-run wiki file at %s -- no checkpoint",
            file_path,
        )
        return _build_empty(source=CheckpointSource.NO_STATE)

    # Case 2: Wiki file exists -- attempt to parse
    try:
        run = current_run.read(wiki_root)
    except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        logger.warning(
            "Corrupted current-run wiki file at %s: %s",
            file_path,
            exc,
        )
        return _build_empty(
            source=CheckpointSource.CORRUPTED,
            error=str(exc),
        )
    except Exception as exc:
        # Catch-all for OS-level errors (PermissionError, IsADirectoryError)
        # and any other unexpected failures. Preserves the never-raise contract.
        logger.warning(
            "Unexpected error reading wiki file at %s: %s",
            file_path,
            exc,
        )
        return _build_empty(
            source=CheckpointSource.CORRUPTED,
            error=str(exc),
        )

    # Case 3: read() returned None. The file_path.exists() check above
    # means this should not happen unless the file is deleted concurrently.
    # Guard defensively since current_run.read() returns Optional[CurrentRun].
    if run is None:
        logger.warning(
            "current_run.read() returned None for existing file %s",
            file_path,
        )
        return _build_empty(
            source=CheckpointSource.CORRUPTED,
            error="Read returned None for existing file",
        )

    # Case 4: Successful parse -- extract checkpoint from run
    checkpoint = _build_from_run(run)

    logger.info(
        "Extracted checkpoint from %s: phase=%s test_index=%d "
        "marker=%r resumable=%s run_id=%s",
        file_path,
        checkpoint.phase.value,
        checkpoint.test_index,
        checkpoint.marker[:50] if checkpoint.marker else "",
        checkpoint.is_resumable,
        checkpoint.run_id,
    )

    return checkpoint
