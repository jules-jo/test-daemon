"""Wiki record reader for daemon boot.

Loads the current-run wiki file on daemon startup and extracts the
run status fields needed for boot decisions and crash recovery.

This module sits between the daemon's boot logic and the raw wiki
persistence layer (current_run module). It provides:
- Graceful handling of missing files (fresh start)
- Graceful handling of corrupted files (safe fallback)
- A focused BootRecord with just the status fields the daemon needs
- LoadOutcome to indicate how the load went

On daemon boot, call load_boot_record(wiki_root) to get the BootRecord.
Check needs_recovery to decide if the daemon should attempt crash recovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from jules_daemon.wiki import current_run
from jules_daemon.wiki.models import CurrentRun, RunStatus

logger = logging.getLogger(__name__)


class LoadOutcome(Enum):
    """How the wiki file load resolved."""

    LOADED = "loaded"
    NO_FILE = "no_file"
    CORRUPTED = "corrupted"


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class BootRecord:
    """Focused view of run state for daemon boot decisions.

    Contains the status fields extracted from the wiki frontmatter:
    - status: current run lifecycle state
    - run_id: unique identifier for the run
    - started_at: when execution began (None if not yet started)
    - completed_at: when the run ended (None if not yet complete)
    - error: error message if the run failed (None otherwise)

    Also includes metadata about the load itself:
    - outcome: how the file load resolved (loaded, no_file, corrupted)
    - source_path: path to the wiki file that was read (None if no file)
    - loaded_at: timestamp of when this record was loaded
    """

    status: RunStatus
    run_id: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[str]
    outcome: LoadOutcome
    source_path: Optional[Path]
    loaded_at: datetime = field(default_factory=_now_utc)

    @property
    def is_active(self) -> bool:
        """True if the run is in a non-terminal, non-idle state."""
        return self.status in (RunStatus.PENDING_APPROVAL, RunStatus.RUNNING)

    @property
    def needs_recovery(self) -> bool:
        """True if the daemon should attempt crash recovery.

        A run that was active (pending_approval or running) when the previous
        daemon instance stopped requires recovery handling.
        """
        return self.is_active


def _build_from_run(
    run: CurrentRun,
    outcome: LoadOutcome,
    source_path: Optional[Path],
) -> BootRecord:
    """Build a BootRecord from a deserialized CurrentRun."""
    return BootRecord(
        status=run.status,
        run_id=run.run_id,
        started_at=run.started_at,
        completed_at=run.completed_at,
        error=run.error,
        outcome=outcome,
        source_path=source_path,
    )


def _build_idle(
    outcome: LoadOutcome,
    source_path: Optional[Path] = None,
    error: Optional[str] = None,
) -> BootRecord:
    """Build a safe idle BootRecord for missing/corrupted file cases."""
    idle_run = CurrentRun(status=RunStatus.IDLE)
    return BootRecord(
        status=RunStatus.IDLE,
        run_id=idle_run.run_id,
        started_at=None,
        completed_at=None,
        error=error,
        outcome=outcome,
        source_path=source_path,
    )


def load_boot_record(wiki_root: Path) -> BootRecord:
    """Load the current-run wiki file and extract run status fields.

    This is the primary entry point for daemon boot. It reads the wiki
    file at {wiki_root}/pages/daemon/current-run.md, parses the YAML
    frontmatter, and returns a BootRecord with the extracted status fields.

    Behavior by scenario:
    - File missing: returns idle BootRecord with outcome=NO_FILE
    - File corrupted: returns idle BootRecord with outcome=CORRUPTED and
      the parse error in the error field
    - File valid: returns BootRecord with extracted status fields and
      outcome=LOADED

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        BootRecord with the extracted status fields. Never raises.
    """
    file_path = current_run.file_path(wiki_root)

    # Case 1: No file exists -- fresh daemon start
    if not file_path.exists():
        logger.info("No current-run wiki file at %s -- starting fresh", file_path)
        return _build_idle(outcome=LoadOutcome.NO_FILE)

    # Case 2: File exists -- try to parse it
    try:
        run = current_run.read(wiki_root)
    except Exception as exc:
        logger.warning(
            "Corrupted current-run wiki file at %s: %s",
            file_path,
            exc,
        )
        return _build_idle(
            outcome=LoadOutcome.CORRUPTED,
            source_path=file_path,
            error=str(exc),
        )

    # Case 3: read() returned None (should not happen if file exists, but
    # handle defensively)
    if run is None:
        logger.warning(
            "current_run.read() returned None for existing file %s",
            file_path,
        )
        return _build_idle(
            outcome=LoadOutcome.CORRUPTED,
            source_path=file_path,
            error="Read returned None for existing file",
        )

    # Case 4: Successful load
    logger.info(
        "Loaded boot record from %s: status=%s run_id=%s",
        file_path,
        run.status.value,
        run.run_id,
    )
    return _build_from_run(
        run=run,
        outcome=LoadOutcome.LOADED,
        source_path=file_path,
    )
