"""Wiki audit-log age scanner -- identifies aged audit entries for archival.

Scans the wiki audit directory (pages/daemon/audit/) for audit-log wiki
entries, parses their YAML frontmatter timestamps, and identifies entries
older than a configurable threshold (default: 90 days).

This module supports the audit log archival workflow described in the
constraints:
- Audit log archival requires explicit user approval before moving to archive
- One audit file per command execution event
- The scanner identifies candidates; the archival step is separate

Each audit-log wiki entry is expected to have YAML frontmatter with:
  - type: audit-log
  - created: ISO 8601 timestamp (primary age source)
  - executed_at: ISO 8601 timestamp (fallback age source)
  - event_id: unique event identifier

The scanner:
- Only reads files directly in pages/daemon/audit/ (not archive/)
- Skips README.md files
- Skips non-audit-log type entries
- Returns results sorted oldest-first for deterministic ordering
- Never modifies any files (read-only operation)

Usage:
    from pathlib import Path
    from jules_daemon.wiki.audit_age_scanner import scan_aged_audit_entries

    result = scan_aged_audit_entries(Path("wiki"))
    for entry in result.aged_entries:
        print(f"Aged: {entry.event_id} ({entry.age_days} days old)")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from jules_daemon.wiki import frontmatter

__all__ = [
    "AgedAuditEntry",
    "AuditAgeScanResult",
    "ScanOutcome",
    "scan_aged_audit_entries",
]

logger = logging.getLogger(__name__)

# The frontmatter type value that identifies audit-log entries
_AUDIT_TYPE = "audit-log"

# Default age threshold in days
_DEFAULT_THRESHOLD_DAYS = 90

# Relative path from wiki root to the audit directory
_AUDIT_DIR = "pages/daemon/audit"

# Files to skip during scanning
_SKIP_FILENAMES = frozenset({"README.md"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _iso_to_datetime(value: str | None) -> datetime | None:
    """Parse ISO 8601 string to datetime, or None.

    If the parsed datetime is naive (no timezone info), UTC is assumed.
    """
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _compute_age_days(created_at: datetime, now: datetime) -> int:
    """Compute the age in whole days between created_at and now."""
    delta = now - created_at
    return max(0, delta.days)


# ---------------------------------------------------------------------------
# Enums and data models
# ---------------------------------------------------------------------------


class ScanOutcome(Enum):
    """How the audit directory scan resolved."""

    SCANNED = "scanned"
    NO_DIRECTORY = "no_directory"


@dataclass(frozen=True)
class AgedAuditEntry:
    """Immutable record of an audit-log entry with age metadata.

    Attributes:
        source_path: Path to the wiki file this entry was parsed from.
        event_id: Unique event identifier from frontmatter.
        created_at: The timestamp used for age calculation.
        age_days: Age of the entry in whole days.
    """

    source_path: Path
    event_id: str
    created_at: datetime
    age_days: int

    def is_over_threshold(self, threshold_days: int) -> bool:
        """True if this entry is strictly older than the given threshold."""
        return self.age_days > threshold_days


@dataclass(frozen=True)
class AuditAgeScanResult:
    """Immutable result of scanning the audit directory for aged entries.

    Attributes:
        outcome: How the scan resolved (scanned vs no_directory).
        entries: All discovered audit-log entries (sorted oldest first).
        errors: Human-readable error descriptions for unparseable files.
        scanned_count: Number of .md files examined.
        threshold_days: The age threshold used for this scan.
    """

    outcome: ScanOutcome
    entries: tuple[AgedAuditEntry, ...]
    errors: tuple[str, ...]
    scanned_count: int
    threshold_days: int

    @property
    def total_count(self) -> int:
        """Total number of successfully parsed audit entries."""
        return len(self.entries)

    @property
    def aged_entries(self) -> tuple[AgedAuditEntry, ...]:
        """Only entries that exceed the threshold age."""
        return tuple(
            e for e in self.entries if e.is_over_threshold(self.threshold_days)
        )

    @property
    def aged_count(self) -> int:
        """Number of entries exceeding the threshold age."""
        return len(self.aged_entries)

    @property
    def error_count(self) -> int:
        """Number of files that could not be parsed."""
        return len(self.errors)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _is_audit_entry(fm: dict[str, Any]) -> bool:
    """Check if a frontmatter dict represents an audit-log entry."""
    return fm.get("type") == _AUDIT_TYPE


def _extract_timestamp(fm: dict[str, Any]) -> datetime | None:
    """Extract the primary timestamp for age calculation.

    Priority order:
    1. 'created' field
    2. 'executed_at' field (fallback)

    Returns None if neither field is present or parseable.
    """
    for field_name in ("created", "executed_at"):
        raw_value = fm.get(field_name)
        if raw_value is not None:
            parsed = _iso_to_datetime(str(raw_value))
            if parsed is not None:
                return parsed
    return None


def _extract_event_id(fm: dict[str, Any], file_path: Path) -> str:
    """Extract the event_id from frontmatter, falling back to filename."""
    event_id = fm.get("event_id")
    if event_id is not None:
        return str(event_id)
    # Derive from filename: audit-{event_id}.md -> {event_id}
    stem = file_path.stem
    if stem.startswith("audit-"):
        return stem[len("audit-"):]
    return stem


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _discover_audit_files(audit_dir: Path) -> list[Path]:
    """Discover audit-log .md files in the audit directory.

    Only reads files directly in the audit directory (not subdirectories
    like archive/). Skips README.md and other non-audit files.

    Returns files sorted by name for deterministic ordering.
    """
    if not audit_dir.is_dir():
        return []

    files: list[Path] = []
    for child in sorted(audit_dir.iterdir()):
        if child.is_file() and child.suffix == ".md":
            if child.name not in _SKIP_FILENAMES:
                files.append(child)

    return files


# ---------------------------------------------------------------------------
# Single-file parsing
# ---------------------------------------------------------------------------


def _try_parse_audit_entry(
    file_path: Path,
    now: datetime,
) -> tuple[AgedAuditEntry | None, str | None]:
    """Attempt to parse a single wiki file as an audit-log entry.

    Returns:
        (entry, None) on success for audit-log files.
        (None, None) for non-audit wiki files (silently skipped).
        (None, error_description) for parse failures.
    """
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"{file_path.name}: read error: {exc}"

    if not raw.strip():
        return None, f"{file_path.name}: empty file"

    try:
        doc = frontmatter.parse(raw)
    except ValueError as exc:
        return None, f"{file_path.name}: parse error: {exc}"

    fm = doc.frontmatter

    # Only process audit-log type entries
    if not _is_audit_entry(fm):
        return None, None

    # Extract timestamp
    created_at = _extract_timestamp(fm)
    if created_at is None:
        return None, f"{file_path.name}: missing timestamp (created/executed_at)"

    # Compute age
    age_days = _compute_age_days(created_at, now)

    # Extract event_id
    event_id = _extract_event_id(fm, file_path)

    entry = AgedAuditEntry(
        source_path=file_path,
        event_id=event_id,
        created_at=created_at,
        age_days=age_days,
    )

    return entry, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_aged_audit_entries(
    wiki_root: Path,
    *,
    threshold_days: int = _DEFAULT_THRESHOLD_DAYS,
) -> AuditAgeScanResult:
    """Scan the wiki audit directory for audit-log entries and compute ages.

    Discovers all .md files in {wiki_root}/pages/daemon/audit/ (excluding
    the archive/ subdirectory), parses their YAML frontmatter, and returns
    structured results with age metadata.

    The returned entries are sorted oldest-first for deterministic ordering.
    Entries in the archive/ subdirectory are excluded from results.

    Args:
        wiki_root: Path to the wiki root directory.
        threshold_days: Age threshold in days for identifying aged entries.
            Must be positive. Defaults to 90.

    Returns:
        AuditAgeScanResult with all discovered entries and age metadata.

    Raises:
        ValueError: If threshold_days is not positive.
    """
    if threshold_days <= 0:
        raise ValueError(
            f"threshold_days must be positive, got {threshold_days}"
        )

    audit_dir = wiki_root / _AUDIT_DIR

    if not audit_dir.is_dir():
        logger.info(
            "Audit directory does not exist at %s -- no entries to scan",
            audit_dir,
        )
        return AuditAgeScanResult(
            outcome=ScanOutcome.NO_DIRECTORY,
            entries=(),
            errors=(),
            scanned_count=0,
            threshold_days=threshold_days,
        )

    now = _now_utc()
    files = _discover_audit_files(audit_dir)
    entries: list[AgedAuditEntry] = []
    errors: list[str] = []
    scanned = 0

    for file_path in files:
        scanned += 1
        entry, error = _try_parse_audit_entry(file_path, now)

        if error is not None:
            errors.append(error)
            logger.warning("Audit age scan: %s", error)
            continue

        if entry is not None:
            entries.append(entry)
            logger.debug(
                "Found audit entry: event_id=%s age=%d days path=%s",
                entry.event_id,
                entry.age_days,
                file_path,
            )

    # Sort oldest first (highest age_days first)
    sorted_entries = sorted(entries, key=lambda e: e.age_days, reverse=True)

    result = AuditAgeScanResult(
        outcome=ScanOutcome.SCANNED,
        entries=tuple(sorted_entries),
        errors=tuple(errors),
        scanned_count=scanned,
        threshold_days=threshold_days,
    )

    logger.info(
        "Audit age scan complete: %d files scanned, %d entries found "
        "(%d aged > %d days), %d errors",
        scanned,
        result.total_count,
        result.aged_count,
        threshold_days,
        result.error_count,
    )

    return result
