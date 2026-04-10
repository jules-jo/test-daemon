"""Wiki session scanner -- discovers and parses all session entries.

Scans the wiki directory for all markdown files with ``type: daemon-state``
frontmatter, parses their YAML metadata, and returns structured SessionEntry
records suitable for liveness evaluation.

Unlike boot_reader or state_reader (which read a single known file), this
module discovers and loads ALL session-type entries across the wiki directory.
This is needed for:
- Daemon boot: scan for any active sessions that might need recovery
- Liveness checks: enumerate sessions and evaluate PID freshness
- Collision detection: warn if multiple active sessions exist

Each SessionEntry is an immutable frozen dataclass containing:
- Source file path
- Run metadata: run_id, status
- Process IDs: daemon PID, remote SSH PID
- SSH target: host, user, port
- Timestamps: started_at, updated_at (for age/staleness calculations)

Usage:
    from pathlib import Path
    from jules_daemon.wiki.session_scanner import (
        scan_all_sessions,
        scan_active_sessions,
    )

    result = scan_all_sessions(Path("wiki"))
    for entry in result.active_entries:
        print(f"Active session {entry.run_id} on {entry.ssh_host}")

    # Or just the active ones:
    active = scan_active_sessions(Path("wiki"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from jules_daemon.wiki import frontmatter

__all__ = [
    "ScanOutcome",
    "ScanResult",
    "SessionEntry",
    "scan_active_sessions",
    "scan_all_sessions",
]

logger = logging.getLogger(__name__)

# The frontmatter type value that identifies daemon-state session files
_SESSION_TYPE = "daemon-state"

# Statuses that indicate an active (non-terminal, non-idle) session
_ACTIVE_STATUSES = frozenset({"running", "pending_approval"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_to_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 string to datetime, or None.

    If the parsed datetime is naive (no timezone info), UTC is assumed.
    """
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _safe_int(value: Any) -> Optional[int]:
    """Coerce a value to int, or return None if not possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_str(value: Any) -> Optional[str]:
    """Coerce a value to str, or return None if not possible."""
    if value is None:
        return None
    s = str(value)
    return s if s else None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ScanOutcome(Enum):
    """How the wiki directory scan resolved."""

    SCANNED = "scanned"
    NO_DIRECTORY = "no_directory"


@dataclass(frozen=True)
class SessionEntry:
    """Immutable record of a session extracted from a wiki file.

    Contains the metadata needed for liveness evaluation:
    - source_path: which wiki file this came from
    - run_id: unique run identifier
    - status: lifecycle status (as RunStatus enum)
    - daemon_pid / remote_pid: process identifiers
    - ssh_host / ssh_user / ssh_port: SSH connection target
    - started_at / updated_at: timestamps for age calculations
    """

    source_path: Path
    run_id: str
    status: "RunStatus"
    daemon_pid: Optional[int]
    remote_pid: Optional[int]
    ssh_host: Optional[str]
    ssh_user: Optional[str]
    ssh_port: Optional[int]
    started_at: Optional[datetime]
    updated_at: datetime

    @property
    def is_active(self) -> bool:
        """True if the session is in a non-terminal, non-idle state."""
        return self.status.value in _ACTIVE_STATUSES

    @property
    def has_ssh_target(self) -> bool:
        """True if SSH connection parameters are present."""
        return self.ssh_host is not None and self.ssh_user is not None

    @property
    def age_seconds(self) -> float:
        """Seconds elapsed since the last update timestamp."""
        delta = _now_utc() - self.updated_at
        return max(0.0, delta.total_seconds())


@dataclass(frozen=True)
class ScanResult:
    """Immutable result of scanning the wiki directory for sessions.

    Contains:
    - outcome: how the scan resolved
    - entries: tuple of all discovered SessionEntry records
    - errors: tuple of human-readable error descriptions for files
      that could not be parsed
    - scanned_count: how many .md files were examined
    """

    outcome: ScanOutcome
    entries: tuple[SessionEntry, ...]
    errors: tuple[str, ...]
    scanned_count: int

    @property
    def total_count(self) -> int:
        """Total number of successfully parsed session entries."""
        return len(self.entries)

    @property
    def active_entries(self) -> tuple[SessionEntry, ...]:
        """Only entries in an active (non-terminal, non-idle) state."""
        return tuple(e for e in self.entries if e.is_active)

    @property
    def active_count(self) -> int:
        """Number of active session entries."""
        return len(self.active_entries)

    @property
    def error_count(self) -> int:
        """Number of files that could not be parsed."""
        return len(self.errors)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

# Import RunStatus here to avoid circular import issues at module level.
# The import is deferred so the module can be loaded independently of models
# if needed, but in practice this is always available.
from jules_daemon.wiki.models import RunStatus  # noqa: E402


def _is_session_file(fm: dict[str, Any]) -> bool:
    """Check if a frontmatter dict represents a daemon-state session."""
    return fm.get("type") == _SESSION_TYPE


def _resolve_status(raw_status: Any) -> RunStatus:
    """Convert a raw status string to a RunStatus enum member.

    Falls back to IDLE for unrecognized values.
    """
    if raw_status is None:
        return RunStatus.IDLE
    try:
        return RunStatus(str(raw_status))
    except ValueError:
        return RunStatus.IDLE


def _extract_session_entry(
    fm: dict[str, Any],
    source_path: Path,
) -> SessionEntry:
    """Extract a SessionEntry from parsed frontmatter.

    Handles missing or malformed fields gracefully with None/defaults.
    """
    # Run metadata
    run_id = str(fm.get("run_id", ""))
    status = _resolve_status(fm.get("status"))

    # Process IDs
    pids = fm.get("pids") or {}
    daemon_pid = _safe_int(pids.get("daemon"))
    remote_pid = _safe_int(pids.get("remote"))

    # SSH target
    ssh_target = fm.get("ssh_target") or {}
    ssh_host = _safe_str(ssh_target.get("host")) if isinstance(ssh_target, dict) else None
    ssh_user = _safe_str(ssh_target.get("user")) if isinstance(ssh_target, dict) else None
    ssh_port = _safe_int(ssh_target.get("port")) if isinstance(ssh_target, dict) else None

    # Timestamps
    started_at = _iso_to_datetime(fm.get("started_at"))
    updated_raw = fm.get("updated")
    updated_at = (_iso_to_datetime(updated_raw) if updated_raw else None) or _now_utc()

    return SessionEntry(
        source_path=source_path,
        run_id=run_id,
        status=status,
        daemon_pid=daemon_pid,
        remote_pid=remote_pid,
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
        started_at=started_at,
        updated_at=updated_at,
    )


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------


def _discover_wiki_files(wiki_root: Path) -> list[Path]:
    """Recursively discover session .md files under the wiki pages directory.

    Only searches under pages/ to avoid scanning raw/, schema/, etc.
    Skips subdirectories that don't contain session files (audit,
    translations, history, runs, knowledge) to avoid parsing files
    that are intentionally large or have malformed YAML.

    Returns files sorted by path for deterministic ordering.
    """
    pages_dir = wiki_root / "pages"
    if not pages_dir.is_dir():
        return []

    # Directories to skip: these contain non-session wiki files
    _SKIP_SUBDIRS = frozenset({
        "audit",
        "translations",
        "history",
        "runs",
        "knowledge",
    })

    files: list[Path] = []
    for md_file in pages_dir.rglob("*.md"):
        # Skip if any parent directory name is in the skip list
        if any(part in _SKIP_SUBDIRS for part in md_file.relative_to(pages_dir).parts):
            continue
        files.append(md_file)

    return sorted(files)


def _try_parse_session(
    file_path: Path,
) -> tuple[Optional[SessionEntry], Optional[str]]:
    """Attempt to parse a single wiki file as a session entry.

    Returns:
        (entry, None) on success for session-type files.
        (None, None) for non-session wiki files (silently skipped).
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

    # Only process daemon-state session files
    if not _is_session_file(doc.frontmatter):
        return None, None

    try:
        entry = _extract_session_entry(doc.frontmatter, file_path)
    except Exception as exc:
        return None, f"{file_path.name}: extraction error: {exc}"

    return entry, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_all_sessions(wiki_root: Path) -> ScanResult:
    """Scan the wiki directory for all daemon-state session entries.

    Discovers all .md files under {wiki_root}/pages/, parses their YAML
    frontmatter, and extracts SessionEntry records for files with
    ``type: daemon-state``.

    Files that are not daemon-state type are silently skipped.
    Files that cannot be parsed are recorded in the errors tuple.

    Behavior by scenario:
    - Wiki directory missing: returns ScanResult with NO_DIRECTORY outcome
    - Empty pages directory: returns ScanResult with SCANNED and empty entries
    - Mixed valid/invalid files: returns valid entries + error descriptions

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        ScanResult with all discovered session entries. Never raises.
    """
    if not wiki_root.is_dir():
        logger.info(
            "Wiki directory does not exist at %s -- no sessions to scan",
            wiki_root,
        )
        return ScanResult(
            outcome=ScanOutcome.NO_DIRECTORY,
            entries=(),
            errors=(),
            scanned_count=0,
        )

    files = _discover_wiki_files(wiki_root)
    entries: list[SessionEntry] = []
    errors: list[str] = []
    scanned = 0

    for file_path in files:
        scanned += 1
        entry, error = _try_parse_session(file_path)

        if error is not None:
            errors.append(error)
            logger.warning("Session scan: %s", error)
            continue

        if entry is not None:
            entries.append(entry)
            logger.debug(
                "Found session entry: run_id=%s status=%s path=%s",
                entry.run_id,
                entry.status.value,
                file_path,
            )

    logger.info(
        "Wiki session scan complete: %d files scanned, %d sessions found "
        "(%d active), %d errors",
        scanned,
        len(entries),
        sum(1 for e in entries if e.is_active),
        len(errors),
    )

    return ScanResult(
        outcome=ScanOutcome.SCANNED,
        entries=tuple(entries),
        errors=tuple(errors),
        scanned_count=scanned,
    )


def scan_active_sessions(wiki_root: Path) -> tuple[SessionEntry, ...]:
    """Convenience function: scan and return only active session entries.

    Active sessions are those with status RUNNING or PENDING_APPROVAL.
    These are the sessions that need liveness evaluation.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        Tuple of active SessionEntry records (may be empty). Never raises.
    """
    result = scan_all_sessions(wiki_root)
    return result.active_entries
