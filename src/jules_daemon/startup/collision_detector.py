"""Daemon collision detection via OS process table and wiki session scanning.

Discovers existing jules-daemon processes through two complementary methods:

1. **Process table scanning**: Runs ``ps`` to find processes whose command
   line matches the daemon process name. Extracts PID, command line, start
   time, and computed duration for each match.

2. **Wiki session scanning**: Uses the existing session_scanner to find
   active (RUNNING or PENDING_APPROVAL) sessions in the wiki. Extracts
   the daemon PID, run ID, and status from each active session.

The two sources are cross-referenced by PID:
- If a PID appears in both sources, the collision is classified as BOTH.
- If a PID appears only in the process table, it is PROCESS_TABLE.
- If a PID appears only in a wiki session, it is WIKI_SESSION.

The current process (our_pid) is always excluded from collision results.

Collision detection is **warn-and-allow**: the has_collision flag is
purely informational and does NOT block daemon startup. Callers should
log a warning and proceed.

Race condition note:
    Processes can start or exit between the time we scan and the time the
    caller acts on the result. Each call probes the OS fresh with no caching.

Usage:
    from pathlib import Path
    from jules_daemon.startup.collision_detector import detect_collisions

    report = detect_collisions(Path("wiki"))
    if report.has_collision:
        for entry in report.entries:
            print(f"Collision: PID {entry.pid} ({entry.source.value})")
        # warn-and-allow: proceed anyway
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

__all__ = [
    "CollisionEntry",
    "CollisionReport",
    "CollisionSource",
    "DetectedProcess",
    "detect_collisions",
    "parse_ps_output",
    "scan_process_table",
]

logger = logging.getLogger(__name__)

# Default process name pattern to search for in the process table.
_DEFAULT_PROCESS_NAME = "jules_daemon"

# ps LSTART format: "Wed Apr  9 10:00:00 2026"
# This is locale-dependent but covers the common POSIX case.
_LSTART_FORMATS = (
    "%a %b %d %H:%M:%S %Y",   # Wed Apr  9 10:00:00 2026
    "%a %b  %d %H:%M:%S %Y",  # Wed Apr  9 10:00:00 2026 (double space)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _parse_lstart(raw: str) -> Optional[datetime]:
    """Parse a ps LSTART timestamp into a UTC datetime.

    Tries multiple format variants to handle locale differences in
    day-of-month padding (single digit with space vs zero-padded).

    Args:
        raw: Raw LSTART string from ps output.

    Returns:
        Parsed datetime with UTC timezone, or None if unparseable.
    """
    cleaned = " ".join(raw.split())  # normalize whitespace
    for fmt in _LSTART_FORMATS:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _compute_duration(start_time: Optional[datetime]) -> Optional[float]:
    """Compute duration in seconds from start_time to now.

    Args:
        start_time: Process start time. None if unknown.

    Returns:
        Duration in seconds (non-negative), or None if start_time is None.
    """
    if start_time is None:
        return None
    delta = _now_utc() - start_time
    return max(0.0, delta.total_seconds())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CollisionSource(Enum):
    """Source of a collision detection entry.

    Values:
        PROCESS_TABLE: Discovered via OS process table scanning.
        WIKI_SESSION: Discovered via wiki active session scanning.
        BOTH: Confirmed by both process table and wiki session.
    """

    PROCESS_TABLE = "process_table"
    WIKI_SESSION = "wiki_session"
    BOTH = "both"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectedProcess:
    """A single daemon process discovered in the OS process table.

    Attributes:
        pid: Process ID from the OS process table.
        command_line: Full command line of the process.
        start_time: UTC datetime when the process started, or None
            if the start time could not be parsed.
        duration_seconds: Seconds since the process started, or None
            if start_time is unavailable.
    """

    pid: int
    command_line: str
    start_time: Optional[datetime]
    duration_seconds: Optional[float]


@dataclass(frozen=True)
class CollisionEntry:
    """A single collision entry combining process table and wiki info.

    Attributes:
        pid: Process ID of the colliding daemon. 0 if no PID is
            available (e.g., wiki session with missing daemon PID).
        command_line: Full command line from the process table, or
            empty string if only discovered via wiki.
        start_time: UTC datetime when the process started, or None.
        duration_seconds: Seconds since the process started, or None.
        source: How the collision was discovered.
        wiki_run_id: Run ID from the wiki session, or None if the
            collision was only found in the process table.
        wiki_status: Status string from the wiki session, or None.
    """

    pid: int
    command_line: str
    start_time: Optional[datetime]
    duration_seconds: Optional[float]
    source: CollisionSource
    wiki_run_id: Optional[str]
    wiki_status: Optional[str]


@dataclass(frozen=True)
class CollisionReport:
    """Complete collision detection report.

    Attributes:
        entries: Tuple of CollisionEntry records, one per detected
            collision. Excludes the current process.
        has_collision: True if at least one collision was detected.
            This is informational only (warn-and-allow).
        our_pid: The PID of the current process, used to exclude
            ourselves from collision results.
        checked_at: UTC datetime when the detection was performed.
    """

    entries: tuple[CollisionEntry, ...]
    has_collision: bool
    our_pid: int
    checked_at: datetime


# ---------------------------------------------------------------------------
# Internal: subprocess wrapper for ps
# ---------------------------------------------------------------------------


def _run_ps_command(process_name: str) -> str:
    """Run ``ps`` to find processes matching the given name.

    Uses ``ps -eo pid,lstart,command`` to get all processes with their
    start times, then greps for the process name. The grep itself is
    excluded by wrapping the first character in brackets (classic trick).

    Args:
        process_name: Substring to search for in command lines.

    Returns:
        Raw stdout from the ps | grep pipeline.

    Raises:
        OSError: If the subprocess cannot be started.
    """
    # The bracket trick: grep "[j]ules_daemon" matches the process but
    # not the grep command itself. We build this dynamically.
    if not process_name:
        return ""

    # Build bracket pattern: "jules_daemon" -> "[j]ules_daemon"
    _bracket_pattern = f"[{process_name[0]}]{process_name[1:]}"

    # We use shell=False with a pipe via subprocess for safety.
    # First get all processes, then filter in Python to avoid shell=True.
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,lstart,command"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        logger.warning("ps command not found -- cannot scan process table")
        raise OSError("ps command not found")
    except subprocess.TimeoutExpired:
        logger.warning("ps command timed out")
        raise OSError("ps command timed out")

    if result.returncode != 0:
        logger.warning(
            "ps command failed with return code %d: %s",
            result.returncode,
            result.stderr.strip(),
        )
        return ""

    # Filter lines containing the process name (case-sensitive)
    lines = result.stdout.splitlines()
    if not lines:
        return ""

    # Keep the header line + matching lines
    header = lines[0] if lines else ""
    matching = [
        line for line in lines[1:]
        if process_name in line
    ]

    if not matching:
        return header + "\n"

    return header + "\n" + "\n".join(matching) + "\n"


# ---------------------------------------------------------------------------
# Public: parse ps output
# ---------------------------------------------------------------------------


def parse_ps_output(raw_output: str) -> tuple[DetectedProcess, ...]:
    """Parse raw ``ps -eo pid,lstart,command`` output into DetectedProcess records.

    Expected format per line (after header):
        <PID> <LSTART (5 fields: DayOfWeek Month Day HH:MM:SS Year)> <COMMAND...>

    Lines that cannot be parsed are silently skipped.

    Args:
        raw_output: Raw stdout from ps command.

    Returns:
        Tuple of DetectedProcess records. Never raises.
    """
    if not raw_output or not raw_output.strip():
        return ()

    lines = raw_output.strip().splitlines()

    # Skip header line (first line typically starts with PID or whitespace+PID)
    if len(lines) <= 1:
        return ()

    data_lines = lines[1:]
    results: list[DetectedProcess] = []

    for line in data_lines:
        stripped = line.strip()
        if not stripped:
            continue

        parsed = _parse_ps_line(stripped)
        if parsed is not None:
            results.append(parsed)

    return tuple(results)


def _parse_ps_line(line: str) -> Optional[DetectedProcess]:
    """Parse a single ps output line into a DetectedProcess.

    Expected format:
        <PID> <DayOfWeek> <Month> <Day> <HH:MM:SS> <Year> <COMMAND...>

    The LSTART field occupies positions 1-5 (5 tokens).

    Args:
        line: A single stripped line from ps output.

    Returns:
        DetectedProcess or None if the line cannot be parsed.
    """
    parts = line.split()

    # Minimum tokens: PID(1) + LSTART(5) + COMMAND(1) = 7
    if len(parts) < 7:
        # Try simpler format: PID + some timestamp + command
        # If we can at least extract a PID and command, do so
        if len(parts) >= 2:
            try:
                pid = int(parts[0])
            except ValueError:
                return None
            command = " ".join(parts[1:])
            return DetectedProcess(
                pid=pid,
                command_line=command,
                start_time=None,
                duration_seconds=None,
            )
        return None

    # Extract PID
    try:
        pid = int(parts[0])
    except ValueError:
        return None

    # Extract LSTART: 5 tokens after PID
    lstart_raw = " ".join(parts[1:6])
    start_time = _parse_lstart(lstart_raw)
    duration = _compute_duration(start_time)

    # Extract command: everything after the LSTART fields
    command = " ".join(parts[6:])

    return DetectedProcess(
        pid=pid,
        command_line=command,
        start_time=start_time,
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Public: scan process table
# ---------------------------------------------------------------------------


def scan_process_table(
    process_name: str = _DEFAULT_PROCESS_NAME,
) -> tuple[DetectedProcess, ...]:
    """Scan the OS process table for daemon processes.

    Runs ``ps -eo pid,lstart,command`` and filters for lines containing
    the given process name. Each matching line is parsed into a
    DetectedProcess record.

    This function never raises. If the ps command fails or produces
    no output, an empty tuple is returned.

    Args:
        process_name: Substring to search for in process command lines.
            Default: "jules_daemon".

    Returns:
        Tuple of DetectedProcess records for matching processes.
    """
    try:
        raw_output = _run_ps_command(process_name)
    except OSError as exc:
        logger.warning("Process table scan failed: %s", exc)
        return ()

    processes = parse_ps_output(raw_output)

    logger.info(
        "Process table scan: found %d matching processes for '%s'",
        len(processes),
        process_name,
    )

    return processes


# ---------------------------------------------------------------------------
# Public: full collision detection
# ---------------------------------------------------------------------------


def detect_collisions(
    wiki_root: Path,
    our_pid: Optional[int] = None,
) -> CollisionReport:
    """Detect existing daemon processes via process table and wiki sessions.

    Combines two detection methods:

    1. **Process table**: Scans the OS process table for processes whose
       command line contains "jules_daemon".

    2. **Wiki sessions**: Scans the wiki for active (RUNNING or
       PENDING_APPROVAL) sessions with daemon PIDs.

    Results from both sources are cross-referenced by PID:
    - Matching PIDs produce a BOTH entry.
    - Process-table-only PIDs produce a PROCESS_TABLE entry.
    - Wiki-only PIDs produce a WIKI_SESSION entry.
    - Active wiki sessions with no daemon PID produce a WIKI_SESSION
      entry with pid=0.

    The current process (our_pid) is excluded from all results.

    This function never raises. All errors are captured and logged.

    Args:
        wiki_root: Path to the wiki root directory.
        our_pid: PID of the current process. Default: os.getpid().

    Returns:
        CollisionReport with all detected collision entries.
    """
    effective_pid = our_pid if our_pid is not None else os.getpid()

    # -- Step 1: Scan process table --
    ps_processes = scan_process_table()

    # Build a map of PID -> DetectedProcess (excluding our PID)
    ps_by_pid: dict[int, DetectedProcess] = {}
    for proc in ps_processes:
        if proc.pid != effective_pid:
            ps_by_pid[proc.pid] = proc

    # -- Step 2: Scan wiki sessions --
    wiki_entries = _scan_wiki_active_sessions(wiki_root)

    # Build a map of daemon_pid -> wiki session info (excluding our PID)
    # Also track sessions with no daemon PID separately
    wiki_by_pid: dict[int, _WikiSessionInfo] = {}
    wiki_no_pid: list[_WikiSessionInfo] = []

    for info in wiki_entries:
        if info.daemon_pid is not None and info.daemon_pid == effective_pid:
            continue  # exclude our own PID
        if info.daemon_pid is not None:
            wiki_by_pid[info.daemon_pid] = info
        else:
            wiki_no_pid.append(info)

    # -- Step 3: Cross-reference and build entries --
    entries: list[CollisionEntry] = []
    seen_pids: set[int] = set()

    # PIDs present in both sources -> BOTH
    common_pids = set(ps_by_pid.keys()) & set(wiki_by_pid.keys())
    for pid in sorted(common_pids):
        proc = ps_by_pid[pid]
        wiki = wiki_by_pid[pid]
        entries.append(CollisionEntry(
            pid=pid,
            command_line=proc.command_line,
            start_time=proc.start_time,
            duration_seconds=proc.duration_seconds,
            source=CollisionSource.BOTH,
            wiki_run_id=wiki.run_id,
            wiki_status=wiki.status,
        ))
        seen_pids.add(pid)

    # PIDs only in process table -> PROCESS_TABLE
    for pid in sorted(ps_by_pid.keys()):
        if pid in seen_pids:
            continue
        proc = ps_by_pid[pid]
        entries.append(CollisionEntry(
            pid=pid,
            command_line=proc.command_line,
            start_time=proc.start_time,
            duration_seconds=proc.duration_seconds,
            source=CollisionSource.PROCESS_TABLE,
            wiki_run_id=None,
            wiki_status=None,
        ))

    # PIDs only in wiki sessions -> WIKI_SESSION
    for pid in sorted(wiki_by_pid.keys()):
        if pid in seen_pids:
            continue
        wiki = wiki_by_pid[pid]
        entries.append(CollisionEntry(
            pid=pid,
            command_line="",
            start_time=wiki.started_at,
            duration_seconds=_compute_duration(wiki.started_at),
            source=CollisionSource.WIKI_SESSION,
            wiki_run_id=wiki.run_id,
            wiki_status=wiki.status,
        ))

    # Wiki sessions with no daemon PID -> WIKI_SESSION with pid=0
    for wiki in wiki_no_pid:
        entries.append(CollisionEntry(
            pid=0,
            command_line="",
            start_time=wiki.started_at,
            duration_seconds=_compute_duration(wiki.started_at),
            source=CollisionSource.WIKI_SESSION,
            wiki_run_id=wiki.run_id,
            wiki_status=wiki.status,
        ))

    collision_entries = tuple(entries)
    has_collision = len(collision_entries) > 0

    if has_collision:
        logger.warning(
            "Collision detected: %d existing daemon process(es) found",
            len(collision_entries),
        )
        for entry in collision_entries:
            logger.warning(
                "  PID=%d source=%s cmd=%s wiki_run=%s",
                entry.pid,
                entry.source.value,
                entry.command_line[:80] if entry.command_line else "(none)",
                entry.wiki_run_id or "(none)",
            )
    else:
        logger.info("No daemon collision detected")

    return CollisionReport(
        entries=collision_entries,
        has_collision=has_collision,
        our_pid=effective_pid,
        checked_at=_now_utc(),
    )


# ---------------------------------------------------------------------------
# Internal: wiki session scanning adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WikiSessionInfo:
    """Lightweight projection of a wiki session for collision detection.

    Avoids coupling the collision detector to the full SessionEntry model.
    """

    run_id: str
    status: str
    daemon_pid: Optional[int]
    started_at: Optional[datetime]


def _scan_wiki_active_sessions(
    wiki_root: Path,
) -> tuple[_WikiSessionInfo, ...]:
    """Scan the wiki for active sessions and project to lightweight info.

    This wraps the session_scanner to extract only the fields needed
    for collision detection.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        Tuple of _WikiSessionInfo for active sessions. Never raises.
    """
    try:
        from jules_daemon.wiki.session_scanner import scan_active_sessions

        active_entries = scan_active_sessions(wiki_root)
    except Exception as exc:
        logger.warning("Wiki session scan failed: %s", exc)
        return ()

    results: list[_WikiSessionInfo] = []
    for entry in active_entries:
        results.append(_WikiSessionInfo(
            run_id=entry.run_id,
            status=entry.status.value,
            daemon_pid=entry.daemon_pid,
            started_at=entry.started_at,
        ))

    return tuple(results)
