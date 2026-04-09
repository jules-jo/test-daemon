"""Tests for daemon collision detection.

Verifies that the collision detector:
- Discovers existing daemon processes by scanning the OS process table
- Parses process table output into structured DetectedProcess records
- Returns structured info: PID, command line, start time, duration
- Cross-references process table findings with wiki active sessions
- Produces a CollisionReport with entries from both sources
- Excludes the current process from collision results
- Classifies collision source: PROCESS_TABLE, WIKI_SESSION, or BOTH
- Handles missing wiki directory gracefully
- Handles empty process table output gracefully
- Handles malformed process table lines gracefully
- Returns immutable (frozen dataclass) results
- Collision detection is warn-and-allow (has_collision is informational)
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from jules_daemon.startup.collision_detector import (
    CollisionEntry,
    CollisionReport,
    CollisionSource,
    DetectedProcess,
    detect_collisions,
    parse_ps_output,
    scan_process_table,
)


# ---------------------------------------------------------------------------
# DetectedProcess model
# ---------------------------------------------------------------------------


class TestDetectedProcess:
    """Tests for the immutable DetectedProcess dataclass."""

    def test_frozen(self) -> None:
        proc = DetectedProcess(
            pid=1234,
            command_line="python -m jules_daemon",
            start_time=datetime.now(timezone.utc),
            duration_seconds=120.0,
        )
        with pytest.raises(AttributeError):
            proc.pid = 9999  # type: ignore[misc]

    def test_has_all_fields(self) -> None:
        ts = datetime.now(timezone.utc)
        proc = DetectedProcess(
            pid=42,
            command_line="python run_daemon.py",
            start_time=ts,
            duration_seconds=60.0,
        )
        assert proc.pid == 42
        assert proc.command_line == "python run_daemon.py"
        assert proc.start_time == ts
        assert proc.duration_seconds == 60.0

    def test_none_start_time_allowed(self) -> None:
        proc = DetectedProcess(
            pid=100,
            command_line="jules_daemon",
            start_time=None,
            duration_seconds=None,
        )
        assert proc.start_time is None
        assert proc.duration_seconds is None


# ---------------------------------------------------------------------------
# CollisionEntry model
# ---------------------------------------------------------------------------


class TestCollisionEntry:
    """Tests for the immutable CollisionEntry dataclass."""

    def test_frozen(self) -> None:
        entry = CollisionEntry(
            pid=1234,
            command_line="python -m jules_daemon",
            start_time=None,
            duration_seconds=None,
            source=CollisionSource.PROCESS_TABLE,
            wiki_run_id=None,
            wiki_status=None,
        )
        with pytest.raises(AttributeError):
            entry.source = CollisionSource.WIKI_SESSION  # type: ignore[misc]

    def test_process_table_source(self) -> None:
        entry = CollisionEntry(
            pid=1234,
            command_line="python -m jules_daemon",
            start_time=None,
            duration_seconds=None,
            source=CollisionSource.PROCESS_TABLE,
            wiki_run_id=None,
            wiki_status=None,
        )
        assert entry.source == CollisionSource.PROCESS_TABLE
        assert entry.wiki_run_id is None

    def test_wiki_session_source(self) -> None:
        entry = CollisionEntry(
            pid=5678,
            command_line="",
            start_time=None,
            duration_seconds=60.0,
            source=CollisionSource.WIKI_SESSION,
            wiki_run_id="abc-123",
            wiki_status="running",
        )
        assert entry.source == CollisionSource.WIKI_SESSION
        assert entry.wiki_run_id == "abc-123"
        assert entry.wiki_status == "running"

    def test_both_source(self) -> None:
        entry = CollisionEntry(
            pid=1234,
            command_line="python -m jules_daemon",
            start_time=None,
            duration_seconds=None,
            source=CollisionSource.BOTH,
            wiki_run_id="run-001",
            wiki_status="running",
        )
        assert entry.source == CollisionSource.BOTH


# ---------------------------------------------------------------------------
# CollisionReport model
# ---------------------------------------------------------------------------


class TestCollisionReport:
    """Tests for the immutable CollisionReport dataclass."""

    def test_frozen(self) -> None:
        report = CollisionReport(
            entries=(),
            has_collision=False,
            our_pid=os.getpid(),
            checked_at=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            report.has_collision = True  # type: ignore[misc]

    def test_no_collision_when_empty(self) -> None:
        report = CollisionReport(
            entries=(),
            has_collision=False,
            our_pid=os.getpid(),
            checked_at=datetime.now(timezone.utc),
        )
        assert report.has_collision is False
        assert len(report.entries) == 0

    def test_has_collision_when_entries_present(self) -> None:
        entry = CollisionEntry(
            pid=1234,
            command_line="python -m jules_daemon",
            start_time=None,
            duration_seconds=None,
            source=CollisionSource.PROCESS_TABLE,
            wiki_run_id=None,
            wiki_status=None,
        )
        report = CollisionReport(
            entries=(entry,),
            has_collision=True,
            our_pid=os.getpid(),
            checked_at=datetime.now(timezone.utc),
        )
        assert report.has_collision is True
        assert len(report.entries) == 1

    def test_our_pid_recorded(self) -> None:
        my_pid = os.getpid()
        report = CollisionReport(
            entries=(),
            has_collision=False,
            our_pid=my_pid,
            checked_at=datetime.now(timezone.utc),
        )
        assert report.our_pid == my_pid


# ---------------------------------------------------------------------------
# CollisionSource enum
# ---------------------------------------------------------------------------


class TestCollisionSource:
    def test_all_values_exist(self) -> None:
        assert CollisionSource.PROCESS_TABLE.value == "process_table"
        assert CollisionSource.WIKI_SESSION.value == "wiki_session"
        assert CollisionSource.BOTH.value == "both"


# ---------------------------------------------------------------------------
# parse_ps_output: parsing raw ps output
# ---------------------------------------------------------------------------


class TestParsePsOutput:
    """Tests for parsing raw `ps` command output into DetectedProcess records."""

    def test_empty_output(self) -> None:
        result = parse_ps_output("")
        assert result == ()

    def test_header_only_output(self) -> None:
        output = "  PID LSTART                     COMMAND\n"
        result = parse_ps_output(output)
        assert result == ()

    def test_single_process_line(self) -> None:
        output = textwrap.dedent("""\
            PID LSTART                         COMMAND
            1234 Wed Apr  9 10:00:00 2026       python -m jules_daemon
        """)
        result = parse_ps_output(output)
        assert len(result) == 1
        assert result[0].pid == 1234
        assert "jules_daemon" in result[0].command_line

    def test_multiple_process_lines(self) -> None:
        output = textwrap.dedent("""\
            PID LSTART                         COMMAND
            1234 Wed Apr  9 10:00:00 2026       python -m jules_daemon serve
            5678 Wed Apr  9 09:30:00 2026       python -m jules_daemon --port 8080
        """)
        result = parse_ps_output(output)
        assert len(result) == 2
        pids = {p.pid for p in result}
        assert pids == {1234, 5678}

    def test_malformed_line_skipped(self) -> None:
        output = textwrap.dedent("""\
            PID LSTART                         COMMAND
            not_a_valid_line
            1234 Wed Apr  9 10:00:00 2026       python -m jules_daemon
        """)
        result = parse_ps_output(output)
        assert len(result) == 1
        assert result[0].pid == 1234

    def test_whitespace_only_lines_skipped(self) -> None:
        output = "\n  \n\t\n"
        result = parse_ps_output(output)
        assert result == ()

    def test_start_time_parsed(self) -> None:
        output = textwrap.dedent("""\
            PID LSTART                         COMMAND
            1234 Wed Apr  9 10:00:00 2026       python -m jules_daemon
        """)
        result = parse_ps_output(output)
        assert len(result) == 1
        assert result[0].start_time is not None
        assert result[0].start_time.year == 2026
        assert result[0].start_time.month == 4
        assert result[0].start_time.day == 9

    def test_unparseable_start_time_returns_none(self) -> None:
        """If start time cannot be parsed, we still return the process."""
        output = textwrap.dedent("""\
            PID LSTART                         COMMAND
            1234 GARBAGE_TIMESTAMP              python -m jules_daemon
        """)
        result = parse_ps_output(output)
        assert len(result) == 1
        assert result[0].pid == 1234
        assert result[0].start_time is None

    def test_duration_computed_from_start_time(self) -> None:
        """Duration should be non-negative when start time is available."""
        output = textwrap.dedent("""\
            PID LSTART                         COMMAND
            1234 Wed Apr  9 10:00:00 2026       python -m jules_daemon
        """)
        # Mock time so duration is deterministic
        fixed_now = datetime(2026, 4, 9, 10, 2, 0, tzinfo=timezone.utc)
        with patch(
            "jules_daemon.startup.collision_detector._now_utc",
            return_value=fixed_now,
        ):
            result = parse_ps_output(output)

        assert len(result) == 1
        assert result[0].duration_seconds is not None
        assert result[0].duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# scan_process_table: OS process discovery
# ---------------------------------------------------------------------------


class TestScanProcessTable:
    """Tests for scanning the OS process table for jules-daemon processes."""

    def test_returns_tuple(self) -> None:
        """Return type must be a tuple of DetectedProcess."""
        with patch(
            "jules_daemon.startup.collision_detector._run_ps_command",
            return_value="",
        ):
            result = scan_process_table()
        assert isinstance(result, tuple)

    def test_empty_when_no_matching_processes(self) -> None:
        with patch(
            "jules_daemon.startup.collision_detector._run_ps_command",
            return_value="  PID LSTART                     COMMAND\n",
        ):
            result = scan_process_table()
        assert result == ()

    def test_finds_jules_daemon_processes(self) -> None:
        ps_output = textwrap.dedent("""\
            PID LSTART                         COMMAND
            1234 Wed Apr  9 10:00:00 2026       python -m jules_daemon
        """)
        with patch(
            "jules_daemon.startup.collision_detector._run_ps_command",
            return_value=ps_output,
        ):
            result = scan_process_table()
        assert len(result) == 1
        assert result[0].pid == 1234

    def test_handles_subprocess_failure(self) -> None:
        """If the ps command fails, return empty tuple (never raise)."""
        with patch(
            "jules_daemon.startup.collision_detector._run_ps_command",
            side_effect=OSError("ps command not found"),
        ):
            result = scan_process_table()
        assert result == ()

    def test_custom_process_name_filter(self) -> None:
        ps_output = textwrap.dedent("""\
            PID LSTART                         COMMAND
            1234 Wed Apr  9 10:00:00 2026       custom_daemon_process
        """)
        with patch(
            "jules_daemon.startup.collision_detector._run_ps_command",
            return_value=ps_output,
        ):
            result = scan_process_table(process_name="custom_daemon")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# detect_collisions: full collision detection
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


class TestDetectCollisionsNoWikiNoProcesses:
    """When there is no wiki directory and no OS processes."""

    def test_no_collision(self, wiki_root: Path) -> None:
        with patch(
            "jules_daemon.startup.collision_detector.scan_process_table",
            return_value=(),
        ):
            report = detect_collisions(wiki_root)
        assert report.has_collision is False
        assert report.entries == ()

    def test_records_our_pid(self, wiki_root: Path) -> None:
        with patch(
            "jules_daemon.startup.collision_detector.scan_process_table",
            return_value=(),
        ):
            report = detect_collisions(wiki_root)
        assert report.our_pid == os.getpid()

    def test_records_timestamp(self, wiki_root: Path) -> None:
        before = datetime.now(timezone.utc)
        with patch(
            "jules_daemon.startup.collision_detector.scan_process_table",
            return_value=(),
        ):
            report = detect_collisions(wiki_root)
        after = datetime.now(timezone.utc)
        assert before <= report.checked_at <= after


class TestDetectCollisionsProcessTableOnly:
    """When collision is detected only from the OS process table."""

    def test_process_table_collision(self, wiki_root: Path) -> None:
        detected = DetectedProcess(
            pid=9999,
            command_line="python -m jules_daemon",
            start_time=datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc),
            duration_seconds=120.0,
        )
        with patch(
            "jules_daemon.startup.collision_detector.scan_process_table",
            return_value=(detected,),
        ):
            report = detect_collisions(wiki_root, our_pid=os.getpid())

        assert report.has_collision is True
        assert len(report.entries) == 1
        assert report.entries[0].source == CollisionSource.PROCESS_TABLE
        assert report.entries[0].pid == 9999
        assert report.entries[0].command_line == "python -m jules_daemon"
        assert report.entries[0].duration_seconds == 120.0

    def test_our_own_pid_excluded(self, wiki_root: Path) -> None:
        """Our own process should not appear as a collision."""
        my_pid = os.getpid()
        detected = DetectedProcess(
            pid=my_pid,
            command_line="python -m jules_daemon",
            start_time=None,
            duration_seconds=None,
        )
        with patch(
            "jules_daemon.startup.collision_detector.scan_process_table",
            return_value=(detected,),
        ):
            report = detect_collisions(wiki_root, our_pid=my_pid)

        assert report.has_collision is False
        assert len(report.entries) == 0


class TestDetectCollisionsWikiOnly:
    """When collision is detected only from wiki active sessions."""

    def test_wiki_session_collision(self, wiki_root: Path) -> None:
        from jules_daemon.wiki import current_run
        from jules_daemon.wiki.models import (
            Command,
            CurrentRun,
            SSHTarget,
        )

        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=7777)
        run = run.with_running("pytest -v", remote_pid=8888)
        current_run.write(wiki_root, run)

        with patch(
            "jules_daemon.startup.collision_detector.scan_process_table",
            return_value=(),
        ):
            report = detect_collisions(wiki_root, our_pid=os.getpid())

        assert report.has_collision is True
        assert len(report.entries) == 1
        assert report.entries[0].source == CollisionSource.WIKI_SESSION
        assert report.entries[0].pid == 7777
        assert report.entries[0].wiki_run_id == run.run_id
        assert report.entries[0].wiki_status == "running"


class TestDetectCollisionsBothSources:
    """When collision detected from both process table and wiki sessions."""

    def test_both_source_when_pid_matches(self, wiki_root: Path) -> None:
        from jules_daemon.wiki import current_run
        from jules_daemon.wiki.models import (
            Command,
            CurrentRun,
            SSHTarget,
        )

        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=9999)
        run = run.with_running("pytest -v", remote_pid=8888)
        current_run.write(wiki_root, run)

        detected = DetectedProcess(
            pid=9999,
            command_line="python -m jules_daemon",
            start_time=datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc),
            duration_seconds=120.0,
        )
        with patch(
            "jules_daemon.startup.collision_detector.scan_process_table",
            return_value=(detected,),
        ):
            report = detect_collisions(wiki_root, our_pid=os.getpid())

        assert report.has_collision is True
        # Both sources should be merged into a single BOTH entry
        assert len(report.entries) == 1
        assert report.entries[0].source == CollisionSource.BOTH
        assert report.entries[0].pid == 9999
        assert report.entries[0].wiki_run_id == run.run_id
        assert report.entries[0].command_line == "python -m jules_daemon"


class TestDetectCollisionsExcludesOurPidFromWiki:
    """Wiki sessions with our own daemon PID are not collisions."""

    def test_wiki_session_with_our_pid_excluded(self, wiki_root: Path) -> None:
        from jules_daemon.wiki import current_run
        from jules_daemon.wiki.models import (
            Command,
            CurrentRun,
            SSHTarget,
        )

        my_pid = os.getpid()
        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=my_pid)
        run = run.with_running("pytest -v", remote_pid=8888)
        current_run.write(wiki_root, run)

        with patch(
            "jules_daemon.startup.collision_detector.scan_process_table",
            return_value=(),
        ):
            report = detect_collisions(wiki_root, our_pid=my_pid)

        assert report.has_collision is False
        assert len(report.entries) == 0


class TestDetectCollisionsWikiNoPid:
    """Wiki session with no daemon PID still shows as collision."""

    def test_wiki_session_no_pid_is_collision(self, wiki_root: Path) -> None:
        """Active wiki session with no daemon_pid is still a collision entry."""
        daemon_dir = wiki_root / "pages" / "daemon"
        daemon_dir.mkdir(parents=True, exist_ok=True)

        session_file = daemon_dir / "orphaned-session.md"
        session_file.write_text(
            "---\n"
            "tags: [daemon, state, current-run]\n"
            "type: daemon-state\n"
            "status: running\n"
            "run_id: orphan-001\n"
            "created: '2026-04-09T08:00:00+00:00'\n"
            "updated: '2026-04-09T08:00:00+00:00'\n"
            "started_at: '2026-04-09T08:00:00+00:00'\n"
            "---\n\n# Orphaned Session\n",
            encoding="utf-8",
        )

        with patch(
            "jules_daemon.startup.collision_detector.scan_process_table",
            return_value=(),
        ):
            report = detect_collisions(wiki_root, our_pid=os.getpid())

        assert report.has_collision is True
        assert len(report.entries) == 1
        entry = report.entries[0]
        assert entry.source == CollisionSource.WIKI_SESSION
        assert entry.wiki_run_id == "orphan-001"
        assert entry.pid == 0  # no PID available


class TestDetectCollisionsMultiple:
    """When multiple collisions are detected."""

    def test_multiple_process_table_entries(self, wiki_root: Path) -> None:
        detected = (
            DetectedProcess(
                pid=1111,
                command_line="python -m jules_daemon serve",
                start_time=None,
                duration_seconds=None,
            ),
            DetectedProcess(
                pid=2222,
                command_line="python -m jules_daemon --port 9090",
                start_time=None,
                duration_seconds=None,
            ),
        )
        with patch(
            "jules_daemon.startup.collision_detector.scan_process_table",
            return_value=detected,
        ):
            report = detect_collisions(wiki_root, our_pid=os.getpid())

        assert report.has_collision is True
        assert len(report.entries) == 2
        pids = {e.pid for e in report.entries}
        assert pids == {1111, 2222}


class TestDetectCollisionsDefaultOurPid:
    """When our_pid is not passed, os.getpid() is used by default."""

    def test_uses_current_pid_by_default(self, wiki_root: Path) -> None:
        my_pid = os.getpid()
        detected = DetectedProcess(
            pid=my_pid,
            command_line="python -m jules_daemon",
            start_time=None,
            duration_seconds=None,
        )
        with patch(
            "jules_daemon.startup.collision_detector.scan_process_table",
            return_value=(detected,),
        ):
            report = detect_collisions(wiki_root)  # no our_pid passed

        assert report.our_pid == my_pid
        assert report.has_collision is False  # our own PID excluded


class TestDetectCollisionsPerformance:
    """Collision detection should complete quickly for crash recovery SLA."""

    def test_completes_under_500ms(self, wiki_root: Path) -> None:
        import time

        with patch(
            "jules_daemon.startup.collision_detector.scan_process_table",
            return_value=(),
        ):
            start = time.monotonic()
            report = detect_collisions(wiki_root)
            elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 500.0, f"Collision detection took {elapsed_ms:.1f}ms"
