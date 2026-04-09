"""Tests for monitoring checkpoint recovery.

The monitoring checkpoint recovery module reads the last-known progress state
from the wiki persistence layer so the monitoring loop knows where it left off
after a daemon crash or restart.

The three key fields recovered are:
  - last_parsed_line_number: position in the SSH output stream
  - timestamp: when the checkpoint was captured
  - extracted_metrics: test counts and progress percentage

This module tests all recovery scenarios: no file, corrupted file, file with
no monitoring section, file with full monitoring data, and boundary cases.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from jules_daemon.wiki import current_run, frontmatter
from jules_daemon.wiki.checkpoint_recovery import (
    ExtractedMetrics,
    MonitoringCheckpoint,
    RecoverySource,
    recover_monitoring_checkpoint,
    persist_monitoring_checkpoint,
)
from jules_daemon.wiki.frontmatter import WikiDocument
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    Progress,
    RunStatus,
    SSHTarget,
)


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


# ---------------------------------------------------------------------------
# ExtractedMetrics model
# ---------------------------------------------------------------------------


class TestExtractedMetrics:
    """Verify ExtractedMetrics is frozen with correct defaults."""

    def test_defaults(self) -> None:
        metrics = ExtractedMetrics()
        assert metrics.tests_passed == 0
        assert metrics.tests_failed == 0
        assert metrics.tests_skipped == 0
        assert metrics.tests_total == 0
        assert metrics.percent == 0.0

    def test_create_with_values(self) -> None:
        metrics = ExtractedMetrics(
            tests_passed=15,
            tests_failed=3,
            tests_skipped=2,
            tests_total=50,
            percent=40.0,
        )
        assert metrics.tests_passed == 15
        assert metrics.tests_failed == 3
        assert metrics.tests_skipped == 2
        assert metrics.tests_total == 50
        assert metrics.percent == 40.0

    def test_frozen(self) -> None:
        metrics = ExtractedMetrics()
        with pytest.raises(AttributeError):
            metrics.tests_passed = 5  # type: ignore[misc]

    def test_tests_completed_property(self) -> None:
        metrics = ExtractedMetrics(
            tests_passed=10, tests_failed=2, tests_skipped=3
        )
        assert metrics.tests_completed == 15

    def test_negative_counts_raise(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            ExtractedMetrics(tests_passed=-1)
        with pytest.raises(ValueError, match="must not be negative"):
            ExtractedMetrics(tests_failed=-1)
        with pytest.raises(ValueError, match="must not be negative"):
            ExtractedMetrics(tests_skipped=-1)
        with pytest.raises(ValueError, match="must not be negative"):
            ExtractedMetrics(tests_total=-1)

    def test_percent_bounds(self) -> None:
        with pytest.raises(ValueError, match="percent must be 0-100"):
            ExtractedMetrics(percent=-1.0)
        with pytest.raises(ValueError, match="percent must be 0-100"):
            ExtractedMetrics(percent=100.1)


# ---------------------------------------------------------------------------
# MonitoringCheckpoint model
# ---------------------------------------------------------------------------


class TestMonitoringCheckpoint:
    """Verify MonitoringCheckpoint is frozen with correct properties."""

    def test_frozen(self) -> None:
        cp = MonitoringCheckpoint(
            last_parsed_line_number=0,
            timestamp=None,
            extracted_metrics=ExtractedMetrics(),
            run_id="",
            status=RunStatus.IDLE,
            source=RecoverySource.NO_STATE,
            error=None,
        )
        with pytest.raises(AttributeError):
            cp.last_parsed_line_number = 10  # type: ignore[misc]

    def test_create_with_all_fields(self) -> None:
        ts = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        metrics = ExtractedMetrics(
            tests_passed=15,
            tests_failed=2,
            tests_total=50,
            percent=34.0,
        )
        cp = MonitoringCheckpoint(
            last_parsed_line_number=142,
            timestamp=ts,
            extracted_metrics=metrics,
            run_id="run-abc-123",
            status=RunStatus.RUNNING,
            source=RecoverySource.WIKI_STATE,
            error=None,
        )
        assert cp.last_parsed_line_number == 142
        assert cp.timestamp == ts
        assert cp.extracted_metrics.tests_passed == 15
        assert cp.run_id == "run-abc-123"
        assert cp.status == RunStatus.RUNNING
        assert cp.source == RecoverySource.WIKI_STATE
        assert cp.error is None

    def test_has_progress_when_metrics_nonzero(self) -> None:
        cp = MonitoringCheckpoint(
            last_parsed_line_number=10,
            timestamp=datetime.now(timezone.utc),
            extracted_metrics=ExtractedMetrics(tests_passed=5, tests_total=20),
            run_id="abc",
            status=RunStatus.RUNNING,
            source=RecoverySource.WIKI_STATE,
            error=None,
        )
        assert cp.has_progress is True

    def test_no_progress_when_metrics_zero(self) -> None:
        cp = MonitoringCheckpoint(
            last_parsed_line_number=0,
            timestamp=None,
            extracted_metrics=ExtractedMetrics(),
            run_id="",
            status=RunStatus.IDLE,
            source=RecoverySource.NO_STATE,
            error=None,
        )
        assert cp.has_progress is False

    def test_is_resumable_when_running_with_progress(self) -> None:
        cp = MonitoringCheckpoint(
            last_parsed_line_number=42,
            timestamp=datetime.now(timezone.utc),
            extracted_metrics=ExtractedMetrics(tests_passed=5, tests_total=20),
            run_id="abc",
            status=RunStatus.RUNNING,
            source=RecoverySource.WIKI_STATE,
            error=None,
        )
        assert cp.is_resumable is True

    def test_is_resumable_when_pending_approval(self) -> None:
        cp = MonitoringCheckpoint(
            last_parsed_line_number=0,
            timestamp=None,
            extracted_metrics=ExtractedMetrics(),
            run_id="abc",
            status=RunStatus.PENDING_APPROVAL,
            source=RecoverySource.WIKI_STATE,
            error=None,
        )
        assert cp.is_resumable is True

    def test_not_resumable_when_idle(self) -> None:
        cp = MonitoringCheckpoint(
            last_parsed_line_number=0,
            timestamp=None,
            extracted_metrics=ExtractedMetrics(),
            run_id="",
            status=RunStatus.IDLE,
            source=RecoverySource.NO_STATE,
            error=None,
        )
        assert cp.is_resumable is False

    def test_not_resumable_when_completed(self) -> None:
        cp = MonitoringCheckpoint(
            last_parsed_line_number=100,
            timestamp=datetime.now(timezone.utc),
            extracted_metrics=ExtractedMetrics(
                tests_passed=50, tests_total=50, percent=100.0
            ),
            run_id="abc",
            status=RunStatus.COMPLETED,
            source=RecoverySource.WIKI_STATE,
            error=None,
        )
        assert cp.is_resumable is False

    def test_not_resumable_when_failed(self) -> None:
        cp = MonitoringCheckpoint(
            last_parsed_line_number=50,
            timestamp=datetime.now(timezone.utc),
            extracted_metrics=ExtractedMetrics(tests_passed=10, tests_total=50),
            run_id="abc",
            status=RunStatus.FAILED,
            source=RecoverySource.WIKI_STATE,
            error="connection lost",
        )
        assert cp.is_resumable is False

    def test_not_resumable_when_corrupted_source(self) -> None:
        cp = MonitoringCheckpoint(
            last_parsed_line_number=0,
            timestamp=None,
            extracted_metrics=ExtractedMetrics(),
            run_id="",
            status=RunStatus.IDLE,
            source=RecoverySource.CORRUPTED,
            error="parse error",
        )
        assert cp.is_resumable is False

    def test_negative_line_number_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            MonitoringCheckpoint(
                last_parsed_line_number=-1,
                timestamp=None,
                extracted_metrics=ExtractedMetrics(),
                run_id="",
                status=RunStatus.IDLE,
                source=RecoverySource.NO_STATE,
                error=None,
            )


# ---------------------------------------------------------------------------
# RecoverySource enum
# ---------------------------------------------------------------------------


class TestRecoverySource:
    def test_all_sources_exist(self) -> None:
        assert RecoverySource.WIKI_STATE.value == "wiki_state"
        assert RecoverySource.NO_STATE.value == "no_state"
        assert RecoverySource.CORRUPTED.value == "corrupted"


# ---------------------------------------------------------------------------
# recover_monitoring_checkpoint: no file
# ---------------------------------------------------------------------------


class TestRecoverNoFile:
    """When no wiki file exists, return a safe empty checkpoint."""

    def test_returns_no_state_source(self, wiki_root: Path) -> None:
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.source == RecoverySource.NO_STATE

    def test_returns_zero_line_number(self, wiki_root: Path) -> None:
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.last_parsed_line_number == 0

    def test_returns_none_timestamp(self, wiki_root: Path) -> None:
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.timestamp is None

    def test_returns_zero_metrics(self, wiki_root: Path) -> None:
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.extracted_metrics.tests_passed == 0
        assert cp.extracted_metrics.tests_failed == 0
        assert cp.extracted_metrics.tests_skipped == 0
        assert cp.extracted_metrics.tests_total == 0
        assert cp.extracted_metrics.percent == 0.0

    def test_returns_idle_status(self, wiki_root: Path) -> None:
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.status == RunStatus.IDLE

    def test_returns_empty_run_id(self, wiki_root: Path) -> None:
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.run_id == ""

    def test_error_is_none(self, wiki_root: Path) -> None:
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.error is None

    def test_not_resumable(self, wiki_root: Path) -> None:
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.is_resumable is False

    def test_no_progress(self, wiki_root: Path) -> None:
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.has_progress is False


# ---------------------------------------------------------------------------
# recover_monitoring_checkpoint: corrupted file
# ---------------------------------------------------------------------------


class TestRecoverCorruptedFile:
    """When the wiki file is corrupted, return a safe fallback."""

    def test_invalid_yaml_returns_corrupted(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("not valid yaml frontmatter", encoding="utf-8")

        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.source == RecoverySource.CORRUPTED

    def test_corrupted_has_error_detail(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("garbage content", encoding="utf-8")

        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.error is not None
        assert len(cp.error) > 0

    def test_corrupted_returns_idle_status(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "---\nstatus: invalid_status\n---\nBody", encoding="utf-8"
        )

        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.source == RecoverySource.CORRUPTED
        assert cp.status == RunStatus.IDLE

    def test_empty_file_returns_corrupted(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("", encoding="utf-8")

        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.source == RecoverySource.CORRUPTED

    def test_corrupted_not_resumable(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("garbage", encoding="utf-8")

        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.is_resumable is False


# ---------------------------------------------------------------------------
# recover_monitoring_checkpoint: idle state (no monitoring section)
# ---------------------------------------------------------------------------


class TestRecoverIdle:
    """When wiki has an idle record, monitoring checkpoint is empty."""

    def test_returns_wiki_state_source(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.source == RecoverySource.WIKI_STATE

    def test_returns_zero_line_number(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.last_parsed_line_number == 0

    def test_returns_idle_status(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.status == RunStatus.IDLE

    def test_not_resumable(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.is_resumable is False


# ---------------------------------------------------------------------------
# recover_monitoring_checkpoint: running with progress (no monitoring overlay)
# ---------------------------------------------------------------------------


class TestRecoverRunningFromProgress:
    """When wiki has a running record with progress but no monitoring overlay,
    metrics are extracted from the progress section and line number is 0."""

    def _write_running(
        self,
        wiki_root: Path,
        *,
        passed: int = 10,
        failed: int = 2,
        skipped: int = 1,
        total: int = 30,
        percent: float = 43.3,
        checkpoint_at: datetime | None = None,
    ) -> CurrentRun:
        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run regression tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        run = run.with_running("pytest -v", remote_pid=5678)
        progress = Progress(
            percent=percent,
            tests_passed=passed,
            tests_failed=failed,
            tests_skipped=skipped,
            tests_total=total,
            last_output_line="PASSED test_checkout",
            checkpoint_at=checkpoint_at,
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)
        return run

    def test_returns_wiki_state_source(self, wiki_root: Path) -> None:
        self._write_running(wiki_root)
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.source == RecoverySource.WIKI_STATE

    def test_extracts_metrics_from_progress(self, wiki_root: Path) -> None:
        self._write_running(
            wiki_root, passed=15, failed=3, skipped=2, total=50, percent=40.0
        )
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.extracted_metrics.tests_passed == 15
        assert cp.extracted_metrics.tests_failed == 3
        assert cp.extracted_metrics.tests_skipped == 2
        assert cp.extracted_metrics.tests_total == 50
        assert cp.extracted_metrics.percent == 40.0

    def test_line_number_zero_without_monitoring_overlay(
        self, wiki_root: Path
    ) -> None:
        self._write_running(wiki_root)
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.last_parsed_line_number == 0

    def test_timestamp_from_checkpoint_at(self, wiki_root: Path) -> None:
        ts = datetime(2026, 4, 9, 12, 30, 0, tzinfo=timezone.utc)
        self._write_running(wiki_root, checkpoint_at=ts)
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.timestamp == ts

    def test_timestamp_none_when_no_checkpoint_at(
        self, wiki_root: Path
    ) -> None:
        self._write_running(wiki_root, checkpoint_at=None)
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.timestamp is None

    def test_preserves_run_id(self, wiki_root: Path) -> None:
        run = self._write_running(wiki_root)
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.run_id == run.run_id

    def test_is_resumable(self, wiki_root: Path) -> None:
        self._write_running(wiki_root, passed=5, total=20)
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.is_resumable is True

    def test_has_progress(self, wiki_root: Path) -> None:
        self._write_running(wiki_root, passed=5, total=20)
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.has_progress is True


# ---------------------------------------------------------------------------
# recover_monitoring_checkpoint: running WITH monitoring overlay
# ---------------------------------------------------------------------------


class TestRecoverRunningWithMonitoring:
    """When the wiki has a monitoring overlay (persisted by the daemon),
    the recovery reads the last_parsed_line_number from it."""

    def _write_with_monitoring(
        self,
        wiki_root: Path,
        *,
        line_number: int = 142,
        monitoring_ts: str = "2026-04-09T12:30:00+00:00",
        passed: int = 15,
        failed: int = 2,
        skipped: int = 1,
        total: int = 50,
        percent: float = 36.0,
    ) -> str:
        """Write a current-run wiki file with monitoring overlay."""
        # First, write a normal running state
        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run regression tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
        run = run.with_running("pytest -v", remote_pid=5678)
        progress = Progress(
            percent=percent,
            tests_passed=passed,
            tests_failed=failed,
            tests_skipped=skipped,
            tests_total=total,
            last_output_line="PASSED test_checkout",
            checkpoint_at=datetime.fromisoformat(monitoring_ts),
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)

        # Now overlay monitoring data into the frontmatter
        file_path = current_run.file_path(wiki_root)
        raw = file_path.read_text(encoding="utf-8")
        doc = frontmatter.parse(raw)

        updated_fm = dict(doc.frontmatter)
        updated_fm["monitoring"] = {
            "last_parsed_line_number": line_number,
            "checkpoint_ts": monitoring_ts,
        }
        updated_doc = WikiDocument(frontmatter=updated_fm, body=doc.body)
        content = frontmatter.serialize(updated_doc)
        file_path.write_text(content, encoding="utf-8")

        return run.run_id

    def test_recovers_line_number(self, wiki_root: Path) -> None:
        self._write_with_monitoring(wiki_root, line_number=142)
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.last_parsed_line_number == 142

    def test_recovers_timestamp_from_monitoring(self, wiki_root: Path) -> None:
        ts_str = "2026-04-09T14:00:00+00:00"
        self._write_with_monitoring(wiki_root, monitoring_ts=ts_str)
        cp = recover_monitoring_checkpoint(wiki_root)
        expected = datetime.fromisoformat(ts_str)
        assert cp.timestamp == expected

    def test_recovers_metrics(self, wiki_root: Path) -> None:
        self._write_with_monitoring(
            wiki_root, passed=20, failed=5, skipped=3, total=60, percent=46.7
        )
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.extracted_metrics.tests_passed == 20
        assert cp.extracted_metrics.tests_failed == 5
        assert cp.extracted_metrics.tests_skipped == 3
        assert cp.extracted_metrics.tests_total == 60
        assert cp.extracted_metrics.percent == 46.7

    def test_preserves_run_id(self, wiki_root: Path) -> None:
        run_id = self._write_with_monitoring(wiki_root)
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.run_id == run_id

    def test_is_resumable(self, wiki_root: Path) -> None:
        self._write_with_monitoring(wiki_root)
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.is_resumable is True

    def test_large_line_number(self, wiki_root: Path) -> None:
        self._write_with_monitoring(wiki_root, line_number=999999)
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.last_parsed_line_number == 999999


# ---------------------------------------------------------------------------
# persist_monitoring_checkpoint (write side, needed for round-trip)
# ---------------------------------------------------------------------------


class TestPersistMonitoringCheckpoint:
    """Verify that persist_monitoring_checkpoint writes monitoring data
    to the wiki file, and recover can read it back."""

    def _create_running_state(self, wiki_root: Path) -> CurrentRun:
        target = SSHTarget(host="staging.example.com", user="deploy")
        cmd = Command(natural_language="run smoke tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=100)
        run = run.with_running("pytest --smoke", remote_pid=200)
        current_run.write(wiki_root, run)
        return run

    def test_round_trip_line_number(self, wiki_root: Path) -> None:
        self._create_running_state(wiki_root)

        ts = datetime(2026, 4, 9, 15, 0, 0, tzinfo=timezone.utc)
        metrics = ExtractedMetrics(
            tests_passed=8, tests_failed=1, tests_total=20, percent=45.0
        )
        checkpoint = MonitoringCheckpoint(
            last_parsed_line_number=77,
            timestamp=ts,
            extracted_metrics=metrics,
            run_id="will-be-ignored",  # persist uses the existing run_id
            status=RunStatus.RUNNING,
            source=RecoverySource.WIKI_STATE,
            error=None,
        )
        persist_monitoring_checkpoint(wiki_root, checkpoint)

        recovered = recover_monitoring_checkpoint(wiki_root)
        assert recovered.last_parsed_line_number == 77

    def test_round_trip_timestamp(self, wiki_root: Path) -> None:
        self._create_running_state(wiki_root)

        ts = datetime(2026, 4, 9, 15, 30, 0, tzinfo=timezone.utc)
        metrics = ExtractedMetrics()
        checkpoint = MonitoringCheckpoint(
            last_parsed_line_number=10,
            timestamp=ts,
            extracted_metrics=metrics,
            run_id="",
            status=RunStatus.RUNNING,
            source=RecoverySource.WIKI_STATE,
            error=None,
        )
        persist_monitoring_checkpoint(wiki_root, checkpoint)

        recovered = recover_monitoring_checkpoint(wiki_root)
        assert recovered.timestamp == ts

    def test_round_trip_preserves_existing_run_data(
        self, wiki_root: Path
    ) -> None:
        """Persisting monitoring data does not overwrite existing run state."""
        run = self._create_running_state(wiki_root)

        ts = datetime(2026, 4, 9, 15, 0, 0, tzinfo=timezone.utc)
        checkpoint = MonitoringCheckpoint(
            last_parsed_line_number=50,
            timestamp=ts,
            extracted_metrics=ExtractedMetrics(),
            run_id="",
            status=RunStatus.RUNNING,
            source=RecoverySource.WIKI_STATE,
            error=None,
        )
        persist_monitoring_checkpoint(wiki_root, checkpoint)

        # Verify the original run data is preserved
        reloaded = current_run.read(wiki_root)
        assert reloaded is not None
        assert reloaded.run_id == run.run_id
        assert reloaded.status == RunStatus.RUNNING
        assert reloaded.ssh_target is not None
        assert reloaded.ssh_target.host == "staging.example.com"

    def test_persist_requires_existing_file(self, wiki_root: Path) -> None:
        """Cannot persist monitoring data without an existing run file."""
        checkpoint = MonitoringCheckpoint(
            last_parsed_line_number=10,
            timestamp=datetime.now(timezone.utc),
            extracted_metrics=ExtractedMetrics(),
            run_id="",
            status=RunStatus.RUNNING,
            source=RecoverySource.WIKI_STATE,
            error=None,
        )
        with pytest.raises(FileNotFoundError):
            persist_monitoring_checkpoint(wiki_root, checkpoint)

    def test_multiple_persists_update_correctly(self, wiki_root: Path) -> None:
        """Multiple persist calls update the monitoring data correctly."""
        self._create_running_state(wiki_root)

        ts1 = datetime(2026, 4, 9, 15, 0, 0, tzinfo=timezone.utc)
        cp1 = MonitoringCheckpoint(
            last_parsed_line_number=10,
            timestamp=ts1,
            extracted_metrics=ExtractedMetrics(tests_passed=2, tests_total=20),
            run_id="",
            status=RunStatus.RUNNING,
            source=RecoverySource.WIKI_STATE,
            error=None,
        )
        persist_monitoring_checkpoint(wiki_root, cp1)

        ts2 = datetime(2026, 4, 9, 15, 5, 0, tzinfo=timezone.utc)
        cp2 = MonitoringCheckpoint(
            last_parsed_line_number=85,
            timestamp=ts2,
            extracted_metrics=ExtractedMetrics(tests_passed=12, tests_total=20),
            run_id="",
            status=RunStatus.RUNNING,
            source=RecoverySource.WIKI_STATE,
            error=None,
        )
        persist_monitoring_checkpoint(wiki_root, cp2)

        recovered = recover_monitoring_checkpoint(wiki_root)
        assert recovered.last_parsed_line_number == 85
        assert recovered.timestamp == ts2


# ---------------------------------------------------------------------------
# recover_monitoring_checkpoint: terminal states
# ---------------------------------------------------------------------------


class TestRecoverTerminalStates:
    """Recovery from terminal states (completed, failed, cancelled)."""

    def test_completed_state(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        final = Progress(
            percent=100.0,
            tests_passed=50,
            tests_failed=0,
            tests_total=50,
        )
        run = run.with_completed(final)
        current_run.write(wiki_root, run)

        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.status == RunStatus.COMPLETED
        assert cp.extracted_metrics.tests_passed == 50
        assert cp.extracted_metrics.percent == 100.0
        assert cp.is_resumable is False

    def test_failed_state_preserves_error(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        error_progress = Progress(
            tests_passed=5, tests_failed=3, tests_total=50
        )
        run = run.with_failed("SSH timeout after 30s", error_progress)
        current_run.write(wiki_root, run)

        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.status == RunStatus.FAILED
        assert cp.error == "SSH timeout after 30s"
        assert cp.is_resumable is False

    def test_cancelled_state(self, wiki_root: Path) -> None:
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        run = run.with_cancelled()
        current_run.write(wiki_root, run)

        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.status == RunStatus.CANCELLED
        assert cp.is_resumable is False


# ---------------------------------------------------------------------------
# recover_monitoring_checkpoint: never raises
# ---------------------------------------------------------------------------


class TestRecoverNeverRaises:
    """The recovery function must never raise exceptions."""

    def test_survives_nonexistent_directory(self) -> None:
        cp = recover_monitoring_checkpoint(Path("/nonexistent/wiki/root"))
        assert cp.source == RecoverySource.NO_STATE

    def test_survives_corrupted_yaml(self, wiki_root: Path) -> None:
        file_path = wiki_root / "pages" / "daemon" / "current-run.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "---\n!!python/object:os.system\n---\n", encoding="utf-8"
        )
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.source in (RecoverySource.CORRUPTED, RecoverySource.NO_STATE)

    def test_survives_permission_error(self, wiki_root: Path) -> None:
        current_run.write(wiki_root, CurrentRun(status=RunStatus.IDLE))

        with patch.object(
            current_run,
            "read",
            side_effect=PermissionError("Permission denied"),
        ):
            cp = recover_monitoring_checkpoint(wiki_root)

        assert cp.source == RecoverySource.CORRUPTED
        assert cp.error is not None
        assert "Permission denied" in cp.error

    def test_survives_monitoring_with_bad_line_number(
        self, wiki_root: Path
    ) -> None:
        """If monitoring section has invalid data, fall back gracefully."""
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        run = run.with_running("pytest")
        current_run.write(wiki_root, run)

        # Write invalid monitoring data
        file_path = current_run.file_path(wiki_root)
        raw = file_path.read_text(encoding="utf-8")
        doc = frontmatter.parse(raw)
        updated_fm = dict(doc.frontmatter)
        updated_fm["monitoring"] = {
            "last_parsed_line_number": "not_a_number",
            "checkpoint_ts": "invalid",
        }
        updated_doc = WikiDocument(frontmatter=updated_fm, body=doc.body)
        file_path.write_text(
            frontmatter.serialize(updated_doc), encoding="utf-8"
        )

        # Should still recover with defaults for bad monitoring fields
        cp = recover_monitoring_checkpoint(wiki_root)
        assert cp.source == RecoverySource.WIKI_STATE
        assert cp.last_parsed_line_number == 0
        assert cp.status == RunStatus.RUNNING


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestRecoverPerformance:
    """Recovery must complete fast enough for the 30s crash recovery SLA."""

    def test_completes_under_100ms(self, wiki_root: Path) -> None:
        target = SSHTarget(host="prod.example.com", user="ci", port=2222)
        cmd = Command(natural_language="run full regression suite")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=9999)
        run = run.with_running("pytest -v --regression", remote_pid=8888)
        progress = Progress(
            percent=75.0,
            tests_passed=150,
            tests_failed=3,
            tests_skipped=5,
            tests_total=200,
            last_output_line="FAILED test_payment_flow",
            checkpoint_at=datetime.now(timezone.utc),
        )
        run = run.with_progress(progress)
        current_run.write(wiki_root, run)

        # Add monitoring overlay
        file_path = current_run.file_path(wiki_root)
        raw = file_path.read_text(encoding="utf-8")
        doc = frontmatter.parse(raw)
        updated_fm = dict(doc.frontmatter)
        updated_fm["monitoring"] = {
            "last_parsed_line_number": 4567,
            "checkpoint_ts": datetime.now(timezone.utc).isoformat(),
        }
        updated_doc = WikiDocument(frontmatter=updated_fm, body=doc.body)
        file_path.write_text(
            frontmatter.serialize(updated_doc), encoding="utf-8"
        )

        start = time.monotonic()
        cp = recover_monitoring_checkpoint(wiki_root)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 100.0, f"Recovery took {elapsed_ms:.1f}ms (>100ms)"
        assert cp.last_parsed_line_number == 4567
        assert cp.extracted_metrics.tests_passed == 150
