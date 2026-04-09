"""Structured logging for stale sessions detected during startup scan.

Emits structured log records via Python's ``logging`` module for each
stale session discovered during the scan-probe-mark pipeline. Each log
record carries a defined schema:

    session_id              -- unique identifier for the session/run
    host                    -- SSH host that was being monitored
    last_activity_timestamp -- UTC ISO 8601 timestamp of last known activity
    staleness_reason        -- human-readable explanation of why stale

The structured data is attached to each log record as a ``stale_session``
extra attribute (a JSON-serializable dict), making it available to any
log handler (JSON formatters, log aggregators, etc.).

This module does NOT perform wiki writes or file I/O -- it only emits
log records. The actual stale marking is handled by
``stale_session_marker``.

Usage:
    from jules_daemon.startup.stale_session_logger import (
        log_stale_sessions_from_verdicts,
    )

    entries = log_stale_sessions_from_verdicts(
        verdicts=pipeline_result.verdicts,
        mark_results=pipeline_result.mark_results,
    )
    for entry in entries:
        print(f"Logged stale: {entry.session_id} on {entry.host}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from jules_daemon.startup.scan_probe_mark import SessionVerdict
from jules_daemon.wiki.stale_session_marker import MarkOutcome, MarkResult

__all__ = [
    "StaleSessionLogEntry",
    "build_stale_log_entry",
    "log_stale_sessions_from_verdicts",
]

logger = logging.getLogger(__name__)

# Mark outcomes that indicate a session was detected as stale
# (MARKED_STALE = successfully written, ERROR = write failed but still stale,
#  SOURCE_MISSING = file gone but session was stale)
_STALE_OUTCOMES = frozenset({
    MarkOutcome.MARKED_STALE,
    MarkOutcome.ERROR,
    MarkOutcome.SOURCE_MISSING,
})

_UNKNOWN_HOST = "unknown"


# ---------------------------------------------------------------------------
# Log entry model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaleSessionLogEntry:
    """Immutable structured log entry for a stale session detection.

    This is the defined log schema for stale sessions. Every stale
    session detected during the startup scan produces exactly one
    of these records.

    Attributes:
        session_id: Unique identifier for the session/run (e.g., UUID).
        host: SSH host that was being monitored. ``"unknown"`` when
            no SSH host was recorded in the session entry.
        last_activity_timestamp: UTC datetime of the last known activity
            for this session (from the wiki ``updated_at`` field).
        staleness_reason: Human-readable explanation of why the session
            was classified as stale (e.g., "Daemon process dead",
            "SSH connection lost").
    """

    session_id: str
    host: str
    last_activity_timestamp: datetime
    staleness_reason: str

    def to_log_dict(self) -> dict[str, str]:
        """Return a JSON-serializable dict of the log schema fields.

        All values are strings to ensure universal JSON compatibility.
        The ``last_activity_timestamp`` is formatted as ISO 8601.

        Returns:
            Dict with keys: session_id, host, last_activity_timestamp,
            staleness_reason.
        """
        return {
            "session_id": self.session_id,
            "host": self.host,
            "last_activity_timestamp": self.last_activity_timestamp.isoformat(),
            "staleness_reason": self.staleness_reason,
        }


# ---------------------------------------------------------------------------
# Entry builder
# ---------------------------------------------------------------------------


def build_stale_log_entry(
    verdict: SessionVerdict,
    mark_result: MarkResult,
) -> StaleSessionLogEntry:
    """Build a StaleSessionLogEntry from a verdict and mark result.

    Extracts the four schema fields from the verdict's session entry
    and the mark result's staleness reason.

    Args:
        verdict: The session verdict from the probe phase.
        mark_result: The mark result from the stale marking phase.

    Returns:
        An immutable StaleSessionLogEntry with all schema fields populated.
    """
    session_entry = verdict.session_entry

    host = session_entry.ssh_host if session_entry.ssh_host is not None else _UNKNOWN_HOST

    # Prefer the mark result's reason; fall back to health-based description
    reason: str
    if mark_result.reason:
        reason = mark_result.reason
    else:
        reason = f"Session health: {verdict.health.value}"

    return StaleSessionLogEntry(
        session_id=session_entry.run_id,
        host=host,
        last_activity_timestamp=session_entry.updated_at,
        staleness_reason=reason,
    )


# ---------------------------------------------------------------------------
# Batch logging
# ---------------------------------------------------------------------------


def log_stale_sessions_from_verdicts(
    *,
    verdicts: tuple[SessionVerdict, ...],
    mark_results: tuple[MarkResult, ...],
) -> tuple[StaleSessionLogEntry, ...]:
    """Log structured records for each stale session from the startup scan.

    Pairs each verdict with its corresponding mark result (by index),
    filters to only stale sessions (those with a mark outcome indicating
    staleness), and emits a WARNING-level log record for each one.

    Each log record includes a ``stale_session`` extra attribute containing
    the structured log schema dict, making it accessible to JSON log
    formatters and log aggregation systems.

    Args:
        verdicts: Tuple of SessionVerdict objects from the probe phase.
            Must have the same length as ``mark_results``.
        mark_results: Tuple of MarkResult objects from the mark phase.
            Must have the same length as ``verdicts``.

    Returns:
        Tuple of StaleSessionLogEntry records for all stale sessions.
        Empty tuple if no stale sessions were detected.

    Raises:
        ValueError: If ``verdicts`` and ``mark_results`` have different
            lengths.
    """
    if len(verdicts) != len(mark_results):
        raise ValueError(
            f"verdicts and mark_results must have the same length, "
            f"got {len(verdicts)} and {len(mark_results)}"
        )

    entries: list[StaleSessionLogEntry] = []

    for verdict, mark in zip(verdicts, mark_results, strict=True):
        # Only log sessions that were actually stale
        if mark.outcome not in _STALE_OUTCOMES:
            continue

        log_entry = build_stale_log_entry(verdict, mark)
        log_dict = log_entry.to_log_dict()

        logger.warning(
            "Stale session detected during startup scan: "
            "session_id=%s host=%s last_activity=%s reason=%s",
            log_entry.session_id,
            log_entry.host,
            log_entry.last_activity_timestamp.isoformat(),
            log_entry.staleness_reason,
            extra={"stale_session": log_dict},
        )

        entries.append(log_entry)

    return tuple(entries)
