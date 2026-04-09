"""Stale session marker -- marks non-live sessions as stale in the wiki.

Given liveness verdicts from the session liveness aggregator, this module
updates each non-live session's wiki entry by writing a NEW file with the
YAML frontmatter set to ``status: stale``, a human-readable staleness
reason, and a detection timestamp. The original file is never mutated.

Immutable write-new-file semantics:
    Instead of overwriting the existing wiki file (which would destroy the
    last known good state), this module writes a new timestamped copy with
    the stale status applied. The original file remains intact for audit
    and crash-recovery purposes.

The new file is placed in the same directory as the original with a
filename pattern: ``{stem}.stale.{timestamp}.md``

Usage:
    from jules_daemon.wiki.stale_session_marker import (
        StaleMarkerInput,
        mark_stale_sessions,
    )

    inputs = [
        StaleMarkerInput(
            liveness_result=verdict,
            source_path=session_entry.source_path,
        )
        for session_entry, verdict in zip(entries, verdicts)
    ]
    results = mark_stale_sessions(inputs, wiki_root)
    for r in results:
        if r.outcome == MarkOutcome.MARKED_STALE:
            logger.info("Marked %s stale at %s", r.session_id, r.stale_path)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from jules_daemon.monitor.session_liveness import (
    LivenessResult,
    SessionHealth,
)
from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument

__all__ = [
    "MarkOutcome",
    "MarkResult",
    "StaleMarkerInput",
    "build_staleness_reason",
    "mark_single_session_stale",
    "mark_stale_sessions",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STALE_STATUS = "stale"


# ---------------------------------------------------------------------------
# Staleness reason mapping
# ---------------------------------------------------------------------------

_HEALTH_REASON_MAP: dict[SessionHealth, str] = {
    SessionHealth.PROCESS_DEAD: "Daemon process dead -- PID no longer exists",
    SessionHealth.CONNECTION_LOST: (
        "SSH connection lost -- remote host unreachable"
    ),
    SessionHealth.UNKNOWN: (
        "Health undetermined -- could not verify process or connection state"
    ),
    SessionHealth.DEGRADED: (
        "Session degraded -- partial connectivity detected"
    ),
    SessionHealth.HEALTHY: (
        "Session was live at detection time (defensive mark)"
    ),
}


def build_staleness_reason(health: SessionHealth) -> str:
    """Generate a human-readable staleness reason from a SessionHealth value.

    Args:
        health: The SessionHealth that triggered the staleness detection.

    Returns:
        A human-readable reason string describing why the session is stale.
    """
    return _HEALTH_REASON_MAP.get(
        health,
        f"Session health: {health.value}",
    )


# ---------------------------------------------------------------------------
# Enums and models
# ---------------------------------------------------------------------------


class MarkOutcome(Enum):
    """Outcome of a single stale-marking operation."""

    MARKED_STALE = "marked_stale"
    SKIPPED_ALIVE = "skipped_alive"
    SOURCE_MISSING = "source_missing"
    ERROR = "error"


@dataclass(frozen=True)
class StaleMarkerInput:
    """Immutable input for a single stale-marking operation.

    Attributes:
        liveness_result: The LivenessResult verdict for this session.
        source_path: Path to the wiki file to mark stale.
    """

    liveness_result: LivenessResult
    source_path: Path


@dataclass(frozen=True)
class MarkResult:
    """Immutable result of a single stale-marking operation.

    Attributes:
        session_id: Identifier of the session that was processed.
        outcome: What happened (marked, skipped, missing, error).
        source_path: Path to the original wiki file.
        stale_path: Path to the newly written stale file (None if not
            written).
        reason: Human-readable staleness reason (None if not marked).
        detected_at: Timestamp when staleness was detected (None if not
            marked).
        error: Error description if outcome is ERROR.
    """

    session_id: str
    outcome: MarkOutcome
    source_path: Path
    stale_path: Optional[Path]
    reason: Optional[str]
    detected_at: Optional[datetime]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _generate_stale_filename(source_path: Path, detected_at: datetime) -> str:
    """Generate a unique filename for the stale copy.

    Pattern: {original_stem}.stale.{compact_timestamp}.md

    The compact timestamp uses only digits and hyphens to avoid filesystem
    issues with colons.

    Args:
        source_path: The original wiki file path.
        detected_at: The detection timestamp.

    Returns:
        The generated filename string.
    """
    stem = source_path.stem
    ts_compact = detected_at.strftime("%Y%m%d-%H%M%S-%f")
    return f"{stem}.stale.{ts_compact}.md"


def _build_stale_frontmatter(
    original_fm: dict,
    *,
    health: SessionHealth,
    reason: str,
    detected_at: datetime,
) -> dict:
    """Build a new frontmatter dict with stale status applied.

    Creates a new dict (never mutates the original) that copies all
    existing fields and overrides/adds the staleness-specific ones.

    Args:
        original_fm: The original frontmatter dict to copy from.
        health: The SessionHealth that triggered staleness.
        reason: The human-readable staleness reason.
        detected_at: The detection timestamp.

    Returns:
        A new frontmatter dict with stale status applied.
    """
    # Copy original fields into a new dict (immutable semantics)
    new_fm = dict(original_fm)

    # Record previous status for audit trail
    previous_status = original_fm.get("status", "unknown")
    new_fm["previous_status"] = previous_status

    # Apply stale status
    new_fm["status"] = _STALE_STATUS
    new_fm["staleness_reason"] = reason
    new_fm["staleness_health"] = health.value
    new_fm["staleness_detected_at"] = detected_at.isoformat()
    new_fm["updated"] = detected_at.isoformat()

    return new_fm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mark_single_session_stale(
    inp: StaleMarkerInput,
    wiki_root: Path,
) -> MarkResult:
    """Mark a single session as stale by writing a new wiki file.

    Uses immutable write-new-file semantics: the original file at
    ``inp.source_path`` is read but never modified. A new file is written
    alongside it with the stale status applied.

    Args:
        inp: The input containing the liveness verdict and source path.
        wiki_root: Path to the wiki root directory (used for logging).

    Returns:
        MarkResult describing what happened. Never raises.
    """
    session_id = inp.liveness_result.session_id
    source = inp.source_path

    # Guard: source file must exist
    if not source.exists():
        logger.warning(
            "Cannot mark session %s stale: source file missing at %s",
            session_id,
            source,
        )
        return MarkResult(
            session_id=session_id,
            outcome=MarkOutcome.SOURCE_MISSING,
            source_path=source,
            stale_path=None,
            reason=None,
            detected_at=None,
            error=f"Source file not found: {source}",
        )

    try:
        # Read the original file
        raw = source.read_text(encoding="utf-8")
        doc = frontmatter.parse(raw)

        # Build the stale version
        detected_at = _now_utc()
        health = inp.liveness_result.health
        reason = build_staleness_reason(health)

        stale_fm = _build_stale_frontmatter(
            doc.frontmatter,
            health=health,
            reason=reason,
            detected_at=detected_at,
        )

        # Assemble the new document (preserving the original body)
        stale_doc = WikiDocument(
            frontmatter=stale_fm,
            body=doc.body,
        )
        content = frontmatter.serialize(stale_doc)

        # Generate new filename and write atomically
        stale_filename = _generate_stale_filename(source, detected_at)
        stale_path = source.parent / stale_filename

        # Ensure parent directory exists
        stale_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: temp file then rename
        tmp_path = stale_path.with_suffix(".md.tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(str(tmp_path), str(stale_path))

        logger.info(
            "Marked session %s stale: health=%s, reason=%s, path=%s",
            session_id,
            health.value,
            reason,
            stale_path,
        )

        return MarkResult(
            session_id=session_id,
            outcome=MarkOutcome.MARKED_STALE,
            source_path=source,
            stale_path=stale_path,
            reason=reason,
            detected_at=detected_at,
            error=None,
        )

    except Exception as exc:
        logger.error(
            "Error marking session %s stale: %s",
            session_id,
            exc,
            exc_info=True,
        )
        return MarkResult(
            session_id=session_id,
            outcome=MarkOutcome.ERROR,
            source_path=source,
            stale_path=None,
            reason=None,
            detected_at=None,
            error=str(exc),
        )


def mark_stale_sessions(
    inputs: list[StaleMarkerInput] | tuple[StaleMarkerInput, ...],
    wiki_root: Path,
) -> tuple[MarkResult, ...]:
    """Process a batch of liveness verdicts and mark non-live sessions stale.

    For each input:
    - If the session is alive (``liveness_result.alive is True``), skip it
      with outcome SKIPPED_ALIVE.
    - If the session is not alive, write a new stale wiki file alongside
      the original.

    Args:
        inputs: Sequence of StaleMarkerInput records to process.
        wiki_root: Path to the wiki root directory.

    Returns:
        Tuple of MarkResult records, one per input, in the same order.
        Never raises.
    """
    results: list[MarkResult] = []

    for inp in inputs:
        session_id = inp.liveness_result.session_id

        if inp.liveness_result.alive:
            logger.debug(
                "Session %s is alive (health=%s) -- skipping stale mark",
                session_id,
                inp.liveness_result.health.value,
            )
            results.append(
                MarkResult(
                    session_id=session_id,
                    outcome=MarkOutcome.SKIPPED_ALIVE,
                    source_path=inp.source_path,
                    stale_path=None,
                    reason=None,
                    detected_at=None,
                    error=None,
                )
            )
            continue

        result = mark_single_session_stale(inp, wiki_root)
        results.append(result)

    return tuple(results)
