"""Monitoring checkpoint recovery from the wiki persistence layer.

Reads the last-known monitoring progress state from the current-run wiki file
so the monitoring loop knows where it left off after a daemon crash or restart.

The three key fields recovered are:
  - last_parsed_line_number: position in the SSH output stream that was last
    processed by the monitoring loop
  - timestamp: when the checkpoint was captured
  - extracted_metrics: test counts and progress percentage at checkpoint time

The monitoring checkpoint is stored as an overlay in the current-run wiki
file's YAML frontmatter under the ``monitoring`` key:

    monitoring:
      last_parsed_line_number: 142
      checkpoint_ts: "2026-04-09T12:30:00+00:00"

The extracted metrics come from the ``progress`` section which is already
maintained by the current_run module.

This module is a focused read/write layer for monitoring checkpoint data.
It reads from the wiki but never modifies the run lifecycle state. It never
raises from the recovery path -- all error conditions are captured in the
returned MonitoringCheckpoint's source and error fields.

Usage:
    from pathlib import Path
    from jules_daemon.wiki.checkpoint_recovery import recover_monitoring_checkpoint

    cp = recover_monitoring_checkpoint(Path("wiki"))
    if cp.is_resumable:
        # Resume from cp.last_parsed_line_number
        # Use cp.extracted_metrics for progress display
        ...
    else:
        # Start fresh or report final status
        ...
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

from jules_daemon.wiki import current_run, frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument
from jules_daemon.wiki.models import CurrentRun, RunStatus

__all__ = [
    "ExtractedMetrics",
    "MonitoringCheckpoint",
    "RecoverySource",
    "persist_monitoring_checkpoint",
    "recover_monitoring_checkpoint",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RecoverySource(Enum):
    """How the monitoring checkpoint recovery resolved.

    WIKI_STATE: Successfully extracted from the current-run wiki file.
    NO_STATE: No wiki state file found (fresh daemon start).
    CORRUPTED: Wiki state file exists but could not be parsed.
    """

    WIKI_STATE = "wiki_state"
    NO_STATE = "no_state"
    CORRUPTED = "corrupted"


# ---------------------------------------------------------------------------
# Resumable status groupings
# ---------------------------------------------------------------------------


_RESUMABLE_STATUSES = frozenset({RunStatus.RUNNING, RunStatus.PENDING_APPROVAL})


# ---------------------------------------------------------------------------
# ExtractedMetrics model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedMetrics:
    """Test metrics extracted from the wiki progress section.

    Immutable snapshot of the test counts and completion percentage at
    the time of the monitoring checkpoint.
    """

    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    tests_total: int = 0
    percent: float = 0.0

    def __post_init__(self) -> None:
        if self.tests_passed < 0:
            raise ValueError("tests_passed must not be negative")
        if self.tests_failed < 0:
            raise ValueError("tests_failed must not be negative")
        if self.tests_skipped < 0:
            raise ValueError("tests_skipped must not be negative")
        if self.tests_total < 0:
            raise ValueError("tests_total must not be negative")
        if not (0.0 <= self.percent <= 100.0):
            raise ValueError(
                f"percent must be 0-100, got {self.percent}"
            )

    @property
    def tests_completed(self) -> int:
        """Total number of tests that have been fully processed."""
        return self.tests_passed + self.tests_failed + self.tests_skipped


# ---------------------------------------------------------------------------
# MonitoringCheckpoint model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringCheckpoint:
    """Recovered monitoring checkpoint from the wiki persistence layer.

    Immutable snapshot that tells the monitoring loop exactly where it left
    off. The daemon uses this after a crash or restart to resume monitoring
    from the correct position in the SSH output stream.

    Attributes:
        last_parsed_line_number: 0-based line number in the SSH output
            stream that was last processed. When no monitoring data exists,
            this is 0 (meaning "start from the beginning").
        timestamp: UTC datetime when this checkpoint was captured. None if
            no checkpoint has been recorded.
        extracted_metrics: Test counts and percentage at checkpoint time.
        run_id: Unique identifier for the run (empty if no run).
        status: The RunStatus from the wiki record.
        source: How this checkpoint was resolved (wiki, no_state, corrupted).
        error: Error description if the run failed or the file was corrupted.
    """

    last_parsed_line_number: int
    timestamp: Optional[datetime]
    extracted_metrics: ExtractedMetrics
    run_id: str
    status: RunStatus
    source: RecoverySource
    error: Optional[str]

    def __post_init__(self) -> None:
        if self.last_parsed_line_number < 0:
            raise ValueError("last_parsed_line_number must not be negative")

    @property
    def has_progress(self) -> bool:
        """True if any test metrics have been recorded."""
        return self.extracted_metrics.tests_completed > 0

    @property
    def is_resumable(self) -> bool:
        """True if the monitoring loop can resume from this checkpoint.

        A checkpoint is resumable when:
        1. It was loaded from a valid wiki state (not corrupted/missing)
        2. The run is in an active state (RUNNING or PENDING_APPROVAL)
        """
        if self.source != RecoverySource.WIKI_STATE:
            return False
        return self.status in _RESUMABLE_STATUSES


# ---------------------------------------------------------------------------
# Internal helpers: extract from frontmatter
# ---------------------------------------------------------------------------


def _extract_metrics_from_progress(
    progress_data: Optional[dict[str, Any]],
) -> ExtractedMetrics:
    """Build ExtractedMetrics from the progress section of frontmatter.

    Handles missing or malformed fields gracefully with zero defaults.
    """
    if progress_data is None:
        return ExtractedMetrics()

    try:
        return ExtractedMetrics(
            tests_passed=int(progress_data.get("tests_passed", 0)),
            tests_failed=int(progress_data.get("tests_failed", 0)),
            tests_skipped=int(progress_data.get("tests_skipped", 0)),
            tests_total=int(progress_data.get("tests_total", 0)),
            percent=float(progress_data.get("percent", 0.0)),
        )
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Malformed progress data in frontmatter: %s", exc
        )
        return ExtractedMetrics()


def _extract_monitoring_line_number(
    monitoring_data: Optional[dict[str, Any]],
) -> int:
    """Extract last_parsed_line_number from the monitoring overlay.

    Returns 0 if the monitoring section is missing or has invalid data.
    """
    if monitoring_data is None:
        return 0

    raw_value = monitoring_data.get("last_parsed_line_number", 0)
    try:
        line_number = int(raw_value)
        return max(0, line_number)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid last_parsed_line_number in monitoring overlay: %r",
            raw_value,
        )
        return 0


def _extract_monitoring_timestamp(
    monitoring_data: Optional[dict[str, Any]],
    progress_data: Optional[dict[str, Any]],
) -> Optional[datetime]:
    """Extract the checkpoint timestamp.

    Prefers the monitoring overlay's checkpoint_ts. Falls back to the
    progress section's checkpoint_at if the monitoring overlay is missing.
    Returns None if neither is available.
    """
    # Try monitoring overlay first
    if monitoring_data is not None:
        ts_value = monitoring_data.get("checkpoint_ts")
        if ts_value is not None:
            try:
                return datetime.fromisoformat(str(ts_value))
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Invalid checkpoint_ts in monitoring overlay: %r (%s)",
                    ts_value,
                    exc,
                )

    # Fall back to progress checkpoint_at
    if progress_data is not None:
        checkpoint_at = progress_data.get("checkpoint_at")
        if checkpoint_at is not None:
            try:
                return datetime.fromisoformat(str(checkpoint_at))
            except (ValueError, TypeError):
                pass

    return None


# ---------------------------------------------------------------------------
# Checkpoint builders
# ---------------------------------------------------------------------------


def _build_from_frontmatter(
    fm: dict[str, Any],
    run: CurrentRun,
) -> MonitoringCheckpoint:
    """Build a MonitoringCheckpoint from parsed frontmatter and CurrentRun.

    Reads both the standard progress fields and the monitoring overlay
    to construct the complete checkpoint.
    """
    progress_data = fm.get("progress")
    monitoring_data = fm.get("monitoring")

    return MonitoringCheckpoint(
        last_parsed_line_number=_extract_monitoring_line_number(
            monitoring_data
        ),
        timestamp=_extract_monitoring_timestamp(
            monitoring_data, progress_data
        ),
        extracted_metrics=_extract_metrics_from_progress(progress_data),
        run_id=run.run_id,
        status=run.status,
        source=RecoverySource.WIKI_STATE,
        error=run.error,
    )


def _build_empty(
    source: RecoverySource,
    error: Optional[str] = None,
) -> MonitoringCheckpoint:
    """Build a safe empty MonitoringCheckpoint for missing/corrupted cases."""
    return MonitoringCheckpoint(
        last_parsed_line_number=0,
        timestamp=None,
        extracted_metrics=ExtractedMetrics(),
        run_id="",
        status=RunStatus.IDLE,
        source=source,
        error=error,
    )


# ---------------------------------------------------------------------------
# Public API: recover
# ---------------------------------------------------------------------------


def recover_monitoring_checkpoint(wiki_root: Path) -> MonitoringCheckpoint:
    """Recover the last-known monitoring checkpoint from the wiki state.

    Reads the current-run wiki file, parses the YAML frontmatter, and
    returns a MonitoringCheckpoint with the extracted monitoring position,
    timestamp, and metrics. The monitoring loop uses this to resume from
    where it left off after a daemon crash or restart.

    This function never raises. All error conditions are captured in the
    returned MonitoringCheckpoint's source and error fields:
    - No wiki file: source=NO_STATE, all fields at zero/empty defaults
    - Corrupted file: source=CORRUPTED, error contains the parse message
    - Valid file: source=WIKI_STATE, all fields populated from the record

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        MonitoringCheckpoint with recovered state. Never raises.
    """
    file_path = current_run.file_path(wiki_root)

    # Case 1: No wiki file exists -- fresh daemon start
    if not file_path.exists():
        logger.info(
            "No current-run wiki file at %s -- no monitoring checkpoint",
            file_path,
        )
        return _build_empty(source=RecoverySource.NO_STATE)

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
            source=RecoverySource.CORRUPTED,
            error=str(exc),
        )
    except Exception as exc:
        # Catch-all for OS-level errors and unexpected failures.
        # Preserves the never-raise contract.
        logger.warning(
            "Unexpected error reading wiki file at %s: %s",
            file_path,
            exc,
        )
        return _build_empty(
            source=RecoverySource.CORRUPTED,
            error=str(exc),
        )

    # Case 3: read() returned None (defensive guard)
    if run is None:
        logger.warning(
            "current_run.read() returned None for existing file %s",
            file_path,
        )
        return _build_empty(
            source=RecoverySource.CORRUPTED,
            error="Read returned None for existing file",
        )

    # Case 4: Successful run parse -- now read the raw frontmatter for
    # the monitoring overlay (which is not part of the CurrentRun model)
    try:
        raw = file_path.read_text(encoding="utf-8")
        doc = frontmatter.parse(raw)
        fm = doc.frontmatter
    except Exception as exc:
        # If raw frontmatter read fails, still return what we have from
        # the CurrentRun model, just without the monitoring overlay.
        logger.warning(
            "Could not read raw frontmatter from %s: %s", file_path, exc
        )
        fm = {
            "progress": {
                "tests_passed": run.progress.tests_passed,
                "tests_failed": run.progress.tests_failed,
                "tests_skipped": run.progress.tests_skipped,
                "tests_total": run.progress.tests_total,
                "percent": run.progress.percent,
                "checkpoint_at": (
                    run.progress.checkpoint_at.isoformat()
                    if run.progress.checkpoint_at
                    else None
                ),
            },
        }

    checkpoint = _build_from_frontmatter(fm, run)

    logger.info(
        "Recovered monitoring checkpoint from %s: "
        "line=%d ts=%s metrics=(p=%d f=%d s=%d t=%d %.1f%%) "
        "resumable=%s run_id=%s",
        file_path,
        checkpoint.last_parsed_line_number,
        checkpoint.timestamp,
        checkpoint.extracted_metrics.tests_passed,
        checkpoint.extracted_metrics.tests_failed,
        checkpoint.extracted_metrics.tests_skipped,
        checkpoint.extracted_metrics.tests_total,
        checkpoint.extracted_metrics.percent,
        checkpoint.is_resumable,
        checkpoint.run_id,
    )

    return checkpoint


# ---------------------------------------------------------------------------
# Public API: persist
# ---------------------------------------------------------------------------


def persist_monitoring_checkpoint(
    wiki_root: Path,
    checkpoint: MonitoringCheckpoint,
) -> Path:
    """Persist monitoring checkpoint data to the wiki file.

    Updates the ``monitoring`` section of the current-run wiki file's
    frontmatter without modifying the run lifecycle state. The existing
    run data (status, SSH target, command, progress, etc.) is preserved.

    This is used by the monitoring loop to periodically save its position
    in the SSH output stream so it can recover after a crash.

    Args:
        wiki_root: Path to the wiki root directory.
        checkpoint: The monitoring checkpoint to persist.

    Returns:
        Path to the written wiki file.

    Raises:
        FileNotFoundError: If no current-run wiki file exists.
    """
    file_path = current_run.file_path(wiki_root)
    if not file_path.exists():
        raise FileNotFoundError(
            f"No current-run record at {file_path}. "
            "Cannot persist monitoring checkpoint without an active run."
        )

    # Read the existing wiki file
    raw = file_path.read_text(encoding="utf-8")
    doc = frontmatter.parse(raw)

    # Build the monitoring overlay (new dict, never mutate the original)
    monitoring_data: dict[str, Any] = {
        "last_parsed_line_number": checkpoint.last_parsed_line_number,
        "checkpoint_ts": (
            checkpoint.timestamp.isoformat()
            if checkpoint.timestamp is not None
            else None
        ),
    }

    # Create updated frontmatter with the monitoring overlay
    updated_fm = dict(doc.frontmatter)
    updated_fm["monitoring"] = monitoring_data

    updated_doc = WikiDocument(frontmatter=updated_fm, body=doc.body)
    content = frontmatter.serialize(updated_doc)

    # Atomic write: write to temp file then rename
    tmp_path = file_path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(str(tmp_path), str(file_path))

    logger.info(
        "Persisted monitoring checkpoint to %s: line=%d",
        file_path,
        checkpoint.last_parsed_line_number,
    )

    return file_path
