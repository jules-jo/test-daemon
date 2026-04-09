"""Tests for wiki session scanner.

Verifies that the session scanner:
- Discovers all wiki files with daemon-state frontmatter type
- Parses YAML frontmatter and extracts session metadata
- Returns structured SessionEntry records with PIDs, SSH info, timestamps
- Handles missing wiki directory gracefully (empty result)
- Handles corrupted files gracefully (skips with warning, does not crash)
- Handles non-session wiki files (skips them)
- Identifies active sessions vs terminal/idle ones
- Supports liveness evaluation via is_active and age_seconds properties
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from jules_daemon.wiki.session_scanner import (
    ScanOutcome,
    ScanResult,
    SessionEntry,
    scan_active_sessions,
    scan_all_sessions,
)
from jules_daemon.wiki import current_run
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    ProcessIDs,
    Progress,
    RunStatus,
    SSHTarget,
)


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


# -- SessionEntry model --


class TestSessionEntry:
    """Tests for the immutable SessionEntry dataclass."""

    def test_frozen(self) -> None:
        entry = SessionEntry(
            source_path=Path("/tmp/test.md"),
            run_id="abc-123",
            status=RunStatus.IDLE,
            daemon_pid=None,
            remote_pid=None,
            ssh_host=None,
            ssh_user=None,
            ssh_port=None,
            started_at=None,
            updated_at=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            entry.status = RunStatus.RUNNING  # type: ignore[misc]

    def test_is_active_when_running(self) -> None:
        entry = SessionEntry(
            source_path=Path("/tmp/test.md"),
            run_id="abc-123",
            status=RunStatus.RUNNING,
            daemon_pid=1234,
            remote_pid=5678,
            ssh_host="staging.example.com",
            ssh_user="deploy",
            ssh_port=22,
            started_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert entry.is_active is True

    def test_is_active_when_pending_approval(self) -> None:
        entry = SessionEntry(
            source_path=Path("/tmp/test.md"),
            run_id="abc-123",
            status=RunStatus.PENDING_APPROVAL,
            daemon_pid=1234,
            remote_pid=None,
            ssh_host="staging.example.com",
            ssh_user="deploy",
            ssh_port=22,
            started_at=None,
            updated_at=datetime.now(timezone.utc),
        )
        assert entry.is_active is True

    def test_not_active_when_idle(self) -> None:
        entry = SessionEntry(
            source_path=Path("/tmp/test.md"),
            run_id="abc-123",
            status=RunStatus.IDLE,
            daemon_pid=None,
            remote_pid=None,
            ssh_host=None,
            ssh_user=None,
            ssh_port=None,
            started_at=None,
            updated_at=datetime.now(timezone.utc),
        )
        assert entry.is_active is False

    def test_not_active_when_terminal(self) -> None:
        for status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            entry = SessionEntry(
                source_path=Path("/tmp/test.md"),
                run_id="abc-123",
                status=status,
                daemon_pid=1234,
                remote_pid=None,
                ssh_host="host",
                ssh_user="user",
                ssh_port=22,
                started_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            assert entry.is_active is False, f"Expected inactive for {status}"

    def test_has_ssh_target_when_populated(self) -> None:
        entry = SessionEntry(
            source_path=Path("/tmp/test.md"),
            run_id="abc-123",
            status=RunStatus.RUNNING,
            daemon_pid=1234,
            remote_pid=5678,
            ssh_host="staging.example.com",
            ssh_user="deploy",
            ssh_port=22,
            started_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert entry.has_ssh_target is True

    def test_no_ssh_target_when_none(self) -> None:
        entry = SessionEntry(
            source_path=Path("/tmp/test.md"),
            run_id="abc-123",
            status=RunStatus.IDLE,
            daemon_pid=None,
            remote_pid=None,
            ssh_host=None,
            ssh_user=None,
            ssh_port=None,
            started_at=None,
            updated_at=datetime.now(timezone.utc),
        )
        assert entry.has_ssh_target is False

    def test_age_seconds_computed(self) -> None:
        past = datetime.now(timezone.utc) - timedelta(seconds=30)
        entry = SessionEntry(
            source_path=Path("/tmp/test.md"),
            run_id="abc-123",
            status=RunStatus.RUNNING,
            daemon_pid=1234,
            remote_pid=5678,
            ssh_host="host",
            ssh_user="user",
            ssh_port=22,
            started_at=past,
            updated_at=past,
        )
        age = entry.age_seconds
        assert age >= 30.0
        assert age < 35.0  # generous upper bound


# -- ScanResult model --


class TestScanResult:
    """Tests for the ScanResult container."""

    def test_empty_result(self) -> None:
        result = ScanResult(
            outcome=ScanOutcome.NO_DIRECTORY,
            entries=(),
            errors=(),
            scanned_count=0,
        )
        assert result.active_entries == ()
        assert result.total_count == 0
        assert result.active_count == 0
        assert result.error_count == 0

    def test_active_entries_filtered(self) -> None:
        now = datetime.now(timezone.utc)
        active = SessionEntry(
            source_path=Path("/tmp/a.md"),
            run_id="a",
            status=RunStatus.RUNNING,
            daemon_pid=1,
            remote_pid=2,
            ssh_host="host",
            ssh_user="user",
            ssh_port=22,
            started_at=now,
            updated_at=now,
        )
        idle = SessionEntry(
            source_path=Path("/tmp/b.md"),
            run_id="b",
            status=RunStatus.IDLE,
            daemon_pid=None,
            remote_pid=None,
            ssh_host=None,
            ssh_user=None,
            ssh_port=None,
            started_at=None,
            updated_at=now,
        )
        result = ScanResult(
            outcome=ScanOutcome.SCANNED,
            entries=(active, idle),
            errors=(),
            scanned_count=2,
        )
        assert result.active_count == 1
        assert result.total_count == 2
        assert result.active_entries == (active,)

    def test_error_count(self) -> None:
        result = ScanResult(
            outcome=ScanOutcome.SCANNED,
            entries=(),
            errors=("file1.md: parse error", "file2.md: missing field"),
            scanned_count=3,
        )
        assert result.error_count == 2


# -- scan_all_sessions (full directory scan) --


class TestScanAllSessionsNoDirectory:
    """When wiki directory does not exist."""

    def test_returns_no_directory_outcome(self, wiki_root: Path) -> None:
        result = scan_all_sessions(wiki_root)
        assert result.outcome == ScanOutcome.NO_DIRECTORY

    def test_returns_empty_entries(self, wiki_root: Path) -> None:
        result = scan_all_sessions(wiki_root)
        assert result.entries == ()
        assert result.scanned_count == 0


class TestScanAllSessionsEmptyDirectory:
    """When wiki directory exists but has no files."""

    def test_returns_empty_scanned(self, wiki_root: Path) -> None:
        (wiki_root / "pages" / "daemon").mkdir(parents=True, exist_ok=True)
        result = scan_all_sessions(wiki_root)
        assert result.outcome == ScanOutcome.SCANNED
        assert result.entries == ()
        assert result.scanned_count == 0


class TestScanAllSessionsIdleRun:
    """When a single idle current-run record exists."""

    def test_finds_idle_session(self, wiki_root: Path) -> None:
        idle_run = CurrentRun(status=RunStatus.IDLE)
        current_run.write(wiki_root, idle_run)

        result = scan_all_sessions(wiki_root)
        assert result.outcome == ScanOutcome.SCANNED
        assert result.total_count == 1
        assert result.entries[0].status == RunStatus.IDLE
        assert result.entries[0].run_id == idle_run.run_id

    def test_idle_session_not_active(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        result = scan_all_sessions(wiki_root)
        assert result.active_count == 0


class TestScanAllSessionsRunningRun:
    """When a running session exists with full SSH metadata."""

    def test_extracts_ssh_metadata(self, wiki_root: Path) -> None:
        target = SSHTarget(host="prod.example.com", user="ci", port=2222)
        cmd = Command(natural_language="run regression suite")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=9999)
        run = run.with_running("pytest --regression", remote_pid=8888)
        current_run.write(wiki_root, run)

        result = scan_all_sessions(wiki_root)
        assert result.total_count == 1

        entry = result.entries[0]
        assert entry.status == RunStatus.RUNNING
        assert entry.ssh_host == "prod.example.com"
        assert entry.ssh_user == "ci"
        assert entry.ssh_port == 2222
        assert entry.daemon_pid == 9999
        assert entry.remote_pid == 8888
        assert entry.started_at is not None

    def test_running_session_is_active(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        current_run.write(wiki_root, run)

        result = scan_all_sessions(wiki_root)
        assert result.active_count == 1


class TestScanAllSessionsPendingApproval:
    """When a pending-approval session exists."""

    def test_extracts_pending_approval_metadata(self, wiki_root: Path) -> None:
        target = SSHTarget(host="staging.example.com", user="deploy", port=22)
        cmd = Command(natural_language="run smoke tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=4321)
        current_run.write(wiki_root, run)

        result = scan_all_sessions(wiki_root)
        entry = result.entries[0]
        assert entry.status == RunStatus.PENDING_APPROVAL
        assert entry.ssh_host == "staging.example.com"
        assert entry.daemon_pid == 4321
        assert entry.remote_pid is None
        assert entry.is_active is True


class TestScanAllSessionsCorruptedFiles:
    """When some wiki files are corrupted."""

    def test_skips_corrupted_file(self, wiki_root: Path) -> None:
        # Write a valid run
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        # Write a corrupted file in the same directory
        corrupted = wiki_root / "pages" / "daemon" / "bad-session.md"
        corrupted.write_text("not valid wiki content", encoding="utf-8")

        result = scan_all_sessions(wiki_root)
        # Should have at least the valid entry
        assert result.total_count >= 1
        assert result.error_count >= 1

    def test_empty_file_is_error(self, wiki_root: Path) -> None:
        daemon_dir = wiki_root / "pages" / "daemon"
        daemon_dir.mkdir(parents=True, exist_ok=True)
        (daemon_dir / "empty.md").write_text("", encoding="utf-8")

        result = scan_all_sessions(wiki_root)
        assert result.error_count == 1

    def test_non_session_file_skipped(self, wiki_root: Path) -> None:
        # Write a valid non-session wiki page
        pages_dir = wiki_root / "pages" / "concepts"
        pages_dir.mkdir(parents=True, exist_ok=True)
        (pages_dir / "llm-wiki.md").write_text(
            "---\ntags: [concept]\ncreated: 2026-04-09\n---\n# LLM Wiki\nContent.",
            encoding="utf-8",
        )

        result = scan_all_sessions(wiki_root)
        assert result.total_count == 0  # non-session entries not included


class TestScanAllSessionsMultipleEntries:
    """When multiple session-type entries exist."""

    def test_scans_multiple_daemon_state_files(self, wiki_root: Path) -> None:
        # Write the primary current-run
        target = SSHTarget(host="host1", user="user1")
        cmd = Command(natural_language="run tests on host1")
        run1 = CurrentRun().with_pending_approval(target, cmd, daemon_pid=100)
        run1 = run1.with_running("pytest", remote_pid=200)
        current_run.write(wiki_root, run1)

        # Write an additional session-type file (e.g., archived or alternate)
        daemon_dir = wiki_root / "pages" / "daemon"
        additional = daemon_dir / "session-archive-001.md"
        additional.write_text(
            "---\n"
            "tags: [daemon, state, current-run]\n"
            "type: daemon-state\n"
            "status: completed\n"
            "run_id: old-run-001\n"
            "created: '2026-04-08T12:00:00+00:00'\n"
            "updated: '2026-04-08T13:00:00+00:00'\n"
            "ssh_target:\n"
            "  host: host2\n"
            "  user: user2\n"
            "  port: 22\n"
            "pids:\n"
            "  daemon: 300\n"
            "  remote: 400\n"
            "started_at: '2026-04-08T12:01:00+00:00'\n"
            "completed_at: '2026-04-08T13:00:00+00:00'\n"
            "error: null\n"
            "command: null\n"
            "progress:\n"
            "  percent: 100.0\n"
            "  tests_passed: 50\n"
            "  tests_failed: 0\n"
            "  tests_skipped: 0\n"
            "  tests_total: 50\n"
            "  last_output_line: ''\n"
            "  checkpoint_at: null\n"
            "---\n\n# Session Archive\n\nCompleted run.\n",
            encoding="utf-8",
        )

        result = scan_all_sessions(wiki_root)
        assert result.total_count == 2
        # Only the running one should be active
        assert result.active_count == 1

    def test_preserves_source_paths(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        result = scan_all_sessions(wiki_root)
        for entry in result.entries:
            assert entry.source_path is not None
            assert entry.source_path.exists()


# -- scan_active_sessions (convenience filter) --


class TestScanActiveSessions:
    """Tests for the convenience function that returns only active entries."""

    def test_returns_only_active(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest", remote_pid=2)
        current_run.write(wiki_root, run)

        entries = scan_active_sessions(wiki_root)
        assert len(entries) == 1
        assert entries[0].is_active is True

    def test_returns_empty_when_idle(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        entries = scan_active_sessions(wiki_root)
        assert entries == ()

    def test_returns_empty_when_no_directory(self, wiki_root: Path) -> None:
        entries = scan_active_sessions(wiki_root)
        assert entries == ()


class TestScanPerformance:
    """Verify scan completes well within the 30s crash recovery SLA."""

    def test_scan_completes_under_200ms(self, wiki_root: Path) -> None:
        import time

        # Write a session with full metadata
        target = SSHTarget(host="prod.example.com", user="ci", port=2222)
        cmd = Command(natural_language="run the full regression suite")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=9999)
        run = run.with_running("pytest -v --regression", remote_pid=8888)
        progress = Progress(
            percent=75.0,
            tests_passed=150,
            tests_failed=3,
            tests_skipped=5,
            tests_total=200,
            last_output_line="FAILED test_payment_flow",
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)

        start = time.monotonic()
        result = scan_all_sessions(wiki_root)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 200.0, f"Scan took {elapsed_ms:.1f}ms (>200ms)"
        assert result.active_count == 1
