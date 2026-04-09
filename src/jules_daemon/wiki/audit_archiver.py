"""Wiki audit-log archiver -- moves approved entries to the archive directory.

Implements the archival operation for audit-log wiki entries:
- Reads each source file and parses its YAML frontmatter
- Adds archival metadata (archived_at, archived_from, archived_age_days)
- Changes the type field from 'audit-log' to 'audit-log-archived'
- Writes the updated file to pages/daemon/audit/archive/
- Removes the original source file only after successful archive write
- Returns immutable structured results for each operation

Constraints satisfied:
- Audit log archival requires explicit user approval before calling this module
  (the caller is responsible for obtaining approval)
- One audit file per command execution event (preserved)
- Collision detection is warn-and-allow, not hard block
- Wiki is the sole persistence layer (no external state stores)

Usage:
    from pathlib import Path
    from jules_daemon.wiki.audit_archiver import archive_audit_entries
    from jules_daemon.wiki.audit_age_scanner import scan_aged_audit_entries

    scan_result = scan_aged_audit_entries(Path("wiki"))
    # ... present aged_entries to user for approval ...
    archival_result = archive_audit_entries(Path("wiki"), scan_result.aged_entries)
    for r in archival_result.results:
        print(f"{r.event_id}: {r.outcome.value}")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Sequence

from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.audit_age_scanner import AgedAuditEntry
from jules_daemon.wiki.frontmatter import WikiDocument

__all__ = [
    "ArchivalOutcome",
    "ArchivalResult",
    "BatchArchivalResult",
    "archive_audit_entries",
    "archive_single_entry",
]

logger = logging.getLogger(__name__)

# Relative path from wiki root to the archive directory
_ARCHIVE_DIR = "pages/daemon/audit/archive"

# The type value for archived audit-log entries
_ARCHIVED_TYPE = "audit-log-archived"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _archive_file_path(wiki_root: Path, source_path: Path) -> Path:
    """Compute the archive destination path for a given source file."""
    return wiki_root / _ARCHIVE_DIR / source_path.name


def _build_archived_frontmatter(
    original_fm: dict[str, object],
    *,
    archived_at: datetime,
    archived_from: str,
    archived_age_days: int,
) -> dict[str, object]:
    """Build a new frontmatter dict with archival metadata added.

    Returns a new dict -- the original is not mutated. The 'type' field
    is changed to 'audit-log-archived' to distinguish archived entries.

    Args:
        original_fm: The parsed frontmatter from the source file.
        archived_at: UTC timestamp of the archival operation.
        archived_from: Relative path to the original source file.
        archived_age_days: Age of the entry at time of archival.

    Returns:
        New frontmatter dict with archival metadata added.
    """
    updated: dict[str, object] = {**original_fm}
    updated["type"] = _ARCHIVED_TYPE
    updated["archived_at"] = archived_at.isoformat()
    updated["archived_from"] = archived_from
    updated["archived_age_days"] = archived_age_days
    return updated


# ---------------------------------------------------------------------------
# Enums and data models
# ---------------------------------------------------------------------------


class ArchivalOutcome(Enum):
    """Outcome of a single audit entry archival operation."""

    ARCHIVED = "archived"
    SOURCE_MISSING = "source_missing"
    PARSE_ERROR = "parse_error"
    WRITE_ERROR = "write_error"


@dataclass(frozen=True)
class ArchivalResult:
    """Immutable result of archiving a single audit-log entry.

    Attributes:
        event_id: The audit event identifier.
        source_path: Path to the original audit file.
        archive_path: Path to the archived file (None on failure).
        outcome: How the archival resolved.
        archived_at: UTC timestamp of the archival attempt.
        error: Human-readable error message (None on success).
    """

    event_id: str
    source_path: Path
    archive_path: Path | None
    outcome: ArchivalOutcome
    archived_at: datetime
    error: str | None

    @property
    def is_success(self) -> bool:
        """True if the entry was successfully archived."""
        return self.outcome == ArchivalOutcome.ARCHIVED


@dataclass(frozen=True)
class BatchArchivalResult:
    """Immutable result of a batch archival operation.

    Attributes:
        results: Individual results for each entry.
        archived_at: UTC timestamp when the batch operation started.
    """

    results: tuple[ArchivalResult, ...]
    archived_at: datetime

    @property
    def total_count(self) -> int:
        """Total number of entries processed."""
        return len(self.results)

    @property
    def succeeded_count(self) -> int:
        """Number of entries successfully archived."""
        return sum(1 for r in self.results if r.is_success)

    @property
    def failed_count(self) -> int:
        """Number of entries that failed to archive."""
        return self.total_count - self.succeeded_count


# ---------------------------------------------------------------------------
# Single entry archival
# ---------------------------------------------------------------------------


def archive_single_entry(
    wiki_root: Path,
    entry: AgedAuditEntry,
    *,
    archived_at: datetime | None = None,
) -> ArchivalResult:
    """Archive a single approved audit-log entry.

    Reads the source file, adds archival metadata to its frontmatter,
    writes the updated content to the archive directory, and removes the
    original source file. The source file is only removed after a
    successful write to the archive.

    Collision policy: warn-and-allow. If an archived file already exists
    at the destination, it is overwritten with a log warning.

    Args:
        wiki_root: Path to the wiki root directory.
        entry: The aged audit entry to archive (from the age scanner).
        archived_at: Override the archival timestamp (for testing).

    Returns:
        ArchivalResult describing the outcome.
    """
    now = archived_at if archived_at is not None else _now_utc()
    source_path = entry.source_path
    archive_path = _archive_file_path(wiki_root, source_path)

    # Step 1: Verify source file exists
    if not source_path.exists():
        logger.warning(
            "Audit archival: source file missing: %s", source_path
        )
        return ArchivalResult(
            event_id=entry.event_id,
            source_path=source_path,
            archive_path=None,
            outcome=ArchivalOutcome.SOURCE_MISSING,
            archived_at=now,
            error=f"Source file not found: {source_path.name}",
        )

    # Step 2: Read and parse the source file
    try:
        raw = source_path.read_text(encoding="utf-8")
        doc = frontmatter.parse(raw)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Audit archival: parse error for %s: %s",
            source_path,
            exc,
        )
        return ArchivalResult(
            event_id=entry.event_id,
            source_path=source_path,
            archive_path=None,
            outcome=ArchivalOutcome.PARSE_ERROR,
            archived_at=now,
            error=f"Parse error: {exc}",
        )

    # Step 3: Build updated frontmatter with archival metadata
    # Compute a relative archived_from path for portability
    archived_from = str(source_path.relative_to(wiki_root))
    updated_fm = _build_archived_frontmatter(
        doc.frontmatter,
        archived_at=now,
        archived_from=archived_from,
        archived_age_days=entry.age_days,
    )

    # Step 4: Serialize the archived document
    archived_doc = WikiDocument(frontmatter=updated_fm, body=doc.body)
    archived_content = frontmatter.serialize(archived_doc)

    # Step 5: Write to archive directory (atomic: temp + rename)
    try:
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        # Warn on collision but allow overwrite
        if archive_path.exists():
            logger.warning(
                "Audit archival: archive file already exists, overwriting: %s",
                archive_path,
            )

        tmp_path = archive_path.with_suffix(".md.tmp")
        tmp_path.write_text(archived_content, encoding="utf-8")
        os.replace(str(tmp_path), str(archive_path))
    except OSError as exc:
        logger.error(
            "Audit archival: write error for %s: %s",
            archive_path,
            exc,
        )
        return ArchivalResult(
            event_id=entry.event_id,
            source_path=source_path,
            archive_path=None,
            outcome=ArchivalOutcome.WRITE_ERROR,
            archived_at=now,
            error=f"Write error: {exc}",
        )

    # Step 6: Remove the original source file (only after successful write)
    try:
        source_path.unlink()
    except OSError as exc:
        # Archive was written successfully, but source cleanup failed.
        # Log the warning but still report success -- the archived copy
        # is the authoritative record.
        logger.warning(
            "Audit archival: could not remove source %s after archiving: %s",
            source_path,
            exc,
        )

    logger.info(
        "Archived audit entry: event_id=%s from=%s to=%s age=%d days",
        entry.event_id,
        source_path,
        archive_path,
        entry.age_days,
    )

    return ArchivalResult(
        event_id=entry.event_id,
        source_path=source_path,
        archive_path=archive_path,
        outcome=ArchivalOutcome.ARCHIVED,
        archived_at=now,
        error=None,
    )


# ---------------------------------------------------------------------------
# Batch archival
# ---------------------------------------------------------------------------


def archive_audit_entries(
    wiki_root: Path,
    entries: Sequence[AgedAuditEntry],
    *,
    archived_at: datetime | None = None,
) -> BatchArchivalResult:
    """Archive a batch of approved audit-log entries.

    Iterates through the provided entries and archives each one
    individually. Failures on individual entries do not stop the batch.

    The caller is responsible for obtaining explicit user approval
    before calling this function (per the security constraint).

    Args:
        wiki_root: Path to the wiki root directory.
        entries: Sequence of aged audit entries to archive.
        archived_at: Override the archival timestamp (for testing).

    Returns:
        BatchArchivalResult with individual results for each entry.
    """
    now = archived_at if archived_at is not None else _now_utc()

    results: list[ArchivalResult] = []
    for entry in entries:
        result = archive_single_entry(wiki_root, entry, archived_at=now)
        results.append(result)

    batch_result = BatchArchivalResult(
        results=tuple(results),
        archived_at=now,
    )

    logger.info(
        "Batch archival complete: %d total, %d succeeded, %d failed",
        batch_result.total_count,
        batch_result.succeeded_count,
        batch_result.failed_count,
    )

    return batch_result
