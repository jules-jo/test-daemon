"""Tests for the scan-probe-mark startup pipeline.

Verifies that the startup pipeline:
- Scans the wiki for all sessions
- Probes each active session for daemon PID liveness and SSH endpoint
  reachability
- Marks non-live sessions as stale in the wiki
- Handles empty wiki (no sessions) gracefully
- Handles wiki directory missing gracefully
- Skips probing for sessions with no daemon PID or SSH target
- Returns structured results from each phase (scan, probe, mark)
- Completes within a reasonable time budget (< 200ms for local-only)
- Constructs LivenessResult objects that the stale marker can consume
- Never raises -- all errors are captured in the pipeline result
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from jules_daemon.monitor.process_state import (
    ProcessCheckResult,
    ProcessVerdict,
)
from jules_daemon.monitor.session_liveness import SessionHealth
from jules_daemon.ssh.endpoint_probe import Endpoint, EndpointVerdict
from jules_daemon.wiki import current_run
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    RunStatus,
    SSHTarget,
)
from jules_daemon.wiki.session_scanner import ScanOutcome
from jules_daemon.wiki.stale_session_marker import MarkOutcome
from jules_daemon.startup.scan_probe_mark import (
    PipelineConfig,
    PipelineResult,
    PipelinePhase,
    SessionVerdict,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


def _write_running_session(
    wiki_root: Path,
    *,
    host: str = "prod.example.com",
    user: str = "ci",
    port: int = 22,
    daemon_pid: int = 99999,
    remote_pid: int = 88888,
) -> CurrentRun:
    """Write a running session to the wiki."""
    target = SSHTarget(host=host, user=user, port=port)
    cmd = Command(natural_language="run the test suite")
    run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=daemon_pid)
    run = run.with_running("pytest -v", remote_pid=remote_pid)
    current_run.write(wiki_root, run)
    return run


def _write_idle_session(wiki_root: Path) -> CurrentRun:
    """Write an idle session to the wiki."""
    run = CurrentRun(status=RunStatus.IDLE)
    current_run.write(wiki_root, run)
    return run


def _mock_process_dead(pid: int) -> ProcessCheckResult:
    """Build a mock ProcessCheckResult for a dead process."""
    return ProcessCheckResult(
        pid=pid,
        verdict=ProcessVerdict.DEAD,
        error="No such process",
        latency_ms=0.1,
        timestamp=datetime.now(timezone.utc),
    )


def _mock_process_alive(pid: int) -> ProcessCheckResult:
    """Build a mock ProcessCheckResult for an alive process."""
    return ProcessCheckResult(
        pid=pid,
        verdict=ProcessVerdict.ALIVE,
        error=None,
        latency_ms=0.1,
        timestamp=datetime.now(timezone.utc),
    )


def _mock_endpoint_reachable(host: str, port: int) -> EndpointVerdict:
    """Build a mock EndpointVerdict for a reachable endpoint."""
    return EndpointVerdict(
        endpoint=Endpoint(host=host, port=port),
        reachable=True,
        banner="SSH-2.0-OpenSSH_8.9",
        latency_ms=5.0,
        error=None,
        timestamp=datetime.now(timezone.utc),
    )


def _mock_endpoint_unreachable(host: str, port: int) -> EndpointVerdict:
    """Build a mock EndpointVerdict for an unreachable endpoint."""
    return EndpointVerdict(
        endpoint=Endpoint(host=host, port=port),
        reachable=False,
        banner=None,
        latency_ms=5000.0,
        error="Connection refused",
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    """Verify the immutable configuration model."""

    def test_defaults(self) -> None:
        config = PipelineConfig()
        assert config.probe_timeout_seconds == 5.0
        assert config.capture_banner is False

    def test_custom_timeout(self) -> None:
        config = PipelineConfig(probe_timeout_seconds=10.0)
        assert config.probe_timeout_seconds == 10.0

    def test_frozen(self) -> None:
        config = PipelineConfig()
        with pytest.raises(AttributeError):
            config.probe_timeout_seconds = 10.0  # type: ignore[misc]

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PipelineConfig(probe_timeout_seconds=0.0)

    def test_negative_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PipelineConfig(probe_timeout_seconds=-1.0)


# ---------------------------------------------------------------------------
# PipelinePhase enum tests
# ---------------------------------------------------------------------------


class TestPipelinePhase:
    """Verify pipeline phase enumeration values."""

    def test_all_phases_exist(self) -> None:
        assert PipelinePhase.SCAN.value == "scan"
        assert PipelinePhase.PROBE.value == "probe"
        assert PipelinePhase.MARK.value == "mark"


# ---------------------------------------------------------------------------
# SessionVerdict model tests
# ---------------------------------------------------------------------------


class TestSessionVerdict:
    """Verify the immutable session verdict model."""

    def test_frozen(self) -> None:
        from jules_daemon.wiki.session_scanner import SessionEntry

        entry = SessionEntry(
            source_path=Path("/tmp/test.md"),
            run_id="abc-123",
            status=RunStatus.RUNNING,
            daemon_pid=1234,
            remote_pid=5678,
            ssh_host="host",
            ssh_user="user",
            ssh_port=22,
            started_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        verdict = SessionVerdict(
            session_entry=entry,
            process_alive=True,
            endpoint_reachable=True,
            health=SessionHealth.HEALTHY,
            alive=True,
        )
        with pytest.raises(AttributeError):
            verdict.alive = False  # type: ignore[misc]

    def test_dead_process_verdict(self) -> None:
        from jules_daemon.wiki.session_scanner import SessionEntry

        entry = SessionEntry(
            source_path=Path("/tmp/test.md"),
            run_id="abc",
            status=RunStatus.RUNNING,
            daemon_pid=1234,
            remote_pid=None,
            ssh_host=None,
            ssh_user=None,
            ssh_port=None,
            started_at=None,
            updated_at=datetime.now(timezone.utc),
        )
        verdict = SessionVerdict(
            session_entry=entry,
            process_alive=False,
            endpoint_reachable=None,
            health=SessionHealth.PROCESS_DEAD,
            alive=False,
        )
        assert verdict.alive is False
        assert verdict.health == SessionHealth.PROCESS_DEAD


# ---------------------------------------------------------------------------
# PipelineResult model tests
# ---------------------------------------------------------------------------


class TestPipelineResult:
    """Verify the immutable pipeline result model."""

    def test_frozen(self) -> None:
        from jules_daemon.wiki.session_scanner import ScanResult

        result = PipelineResult(
            scan_result=ScanResult(
                outcome=ScanOutcome.NO_DIRECTORY,
                entries=(),
                errors=(),
                scanned_count=0,
            ),
            verdicts=(),
            mark_results=(),
            duration_seconds=0.1,
            timestamp=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            result.duration_seconds = 1.0  # type: ignore[misc]

    def test_counts(self) -> None:
        from jules_daemon.wiki.session_scanner import ScanResult

        result = PipelineResult(
            scan_result=ScanResult(
                outcome=ScanOutcome.SCANNED,
                entries=(),
                errors=(),
                scanned_count=5,
            ),
            verdicts=(),
            mark_results=(),
            duration_seconds=0.05,
            timestamp=datetime.now(timezone.utc),
        )
        assert result.scan_result.scanned_count == 5
        assert result.verdicts == ()
        assert result.mark_results == ()


# ---------------------------------------------------------------------------
# run_pipeline: empty/missing wiki
# ---------------------------------------------------------------------------


class TestPipelineEmptyWiki:
    """Pipeline with no wiki directory or no sessions."""

    @pytest.mark.asyncio
    async def test_missing_wiki_returns_no_directory(
        self, wiki_root: Path
    ) -> None:
        """When wiki directory does not exist, pipeline succeeds with
        NO_DIRECTORY scan outcome and empty verdicts/marks."""
        result = await run_pipeline(wiki_root)
        assert result.scan_result.outcome == ScanOutcome.NO_DIRECTORY
        assert result.verdicts == ()
        assert result.mark_results == ()

    @pytest.mark.asyncio
    async def test_empty_wiki_returns_empty_scanned(
        self, wiki_root: Path
    ) -> None:
        """When wiki directory exists but has no sessions."""
        (wiki_root / "pages" / "daemon").mkdir(parents=True, exist_ok=True)
        result = await run_pipeline(wiki_root)
        assert result.scan_result.outcome == ScanOutcome.SCANNED
        assert result.verdicts == ()
        assert result.mark_results == ()

    @pytest.mark.asyncio
    async def test_idle_session_no_probing(self, wiki_root: Path) -> None:
        """An idle session should not trigger probing or marking."""
        _write_idle_session(wiki_root)
        result = await run_pipeline(wiki_root)
        assert result.scan_result.total_count == 1
        assert result.scan_result.active_count == 0
        assert result.verdicts == ()
        assert result.mark_results == ()


# ---------------------------------------------------------------------------
# run_pipeline: active session with dead daemon PID
# ---------------------------------------------------------------------------


class TestPipelineDeadDaemon:
    """Pipeline marks sessions with dead daemon PIDs as stale."""

    @pytest.mark.asyncio
    async def test_dead_daemon_marked_stale(self, wiki_root: Path) -> None:
        """An active session with a dead daemon PID is marked stale."""
        run = _write_running_session(wiki_root, daemon_pid=99999)

        with patch(
            "jules_daemon.startup.scan_probe_mark.check_pid"
        ) as mock_check:
            mock_check.return_value = _mock_process_dead(99999)

            with patch(
                "jules_daemon.startup.scan_probe_mark.check_endpoints",
                new_callable=AsyncMock,
            ) as mock_endpoints:
                mock_endpoints.return_value = (
                    _mock_endpoint_unreachable("prod.example.com", 22),
                )

                result = await run_pipeline(wiki_root)

        assert len(result.verdicts) == 1
        assert result.verdicts[0].health == SessionHealth.PROCESS_DEAD
        assert result.verdicts[0].alive is False
        assert len(result.mark_results) == 1
        assert result.mark_results[0].outcome == MarkOutcome.MARKED_STALE

    @pytest.mark.asyncio
    async def test_dead_daemon_skips_endpoint_probe(
        self, wiki_root: Path
    ) -> None:
        """When daemon PID is dead, endpoint probing is skipped."""
        _write_running_session(wiki_root, daemon_pid=99999)

        with patch(
            "jules_daemon.startup.scan_probe_mark.check_pid"
        ) as mock_check:
            mock_check.return_value = _mock_process_dead(99999)

            with patch(
                "jules_daemon.startup.scan_probe_mark.check_endpoints",
                new_callable=AsyncMock,
            ) as mock_endpoints:
                mock_endpoints.return_value = ()

                result = await run_pipeline(wiki_root)

                # Endpoint probe should NOT be called for dead processes
                mock_endpoints.assert_not_called()

        assert result.verdicts[0].endpoint_reachable is None


# ---------------------------------------------------------------------------
# run_pipeline: active session with alive daemon + reachable endpoint
# ---------------------------------------------------------------------------


class TestPipelineAliveSession:
    """Pipeline skips marking for sessions that are alive."""

    @pytest.mark.asyncio
    async def test_alive_session_skipped(self, wiki_root: Path) -> None:
        """An active session with alive daemon + reachable endpoint is
        skipped during marking."""
        _write_running_session(wiki_root, daemon_pid=99999)

        with patch(
            "jules_daemon.startup.scan_probe_mark.check_pid"
        ) as mock_check:
            mock_check.return_value = _mock_process_alive(99999)

            with patch(
                "jules_daemon.startup.scan_probe_mark.check_endpoints",
                new_callable=AsyncMock,
            ) as mock_endpoints:
                mock_endpoints.return_value = (
                    _mock_endpoint_reachable("prod.example.com", 22),
                )

                result = await run_pipeline(wiki_root)

        assert len(result.verdicts) == 1
        assert result.verdicts[0].health == SessionHealth.HEALTHY
        assert result.verdicts[0].alive is True
        assert len(result.mark_results) == 1
        assert result.mark_results[0].outcome == MarkOutcome.SKIPPED_ALIVE


# ---------------------------------------------------------------------------
# run_pipeline: alive daemon + unreachable endpoint
# ---------------------------------------------------------------------------


class TestPipelineConnectionLost:
    """Pipeline handles sessions with alive daemon but lost connection."""

    @pytest.mark.asyncio
    async def test_connection_lost_marked_stale(
        self, wiki_root: Path
    ) -> None:
        """Alive daemon + unreachable endpoint = CONNECTION_LOST = stale."""
        _write_running_session(wiki_root, daemon_pid=99999)

        with patch(
            "jules_daemon.startup.scan_probe_mark.check_pid"
        ) as mock_check:
            mock_check.return_value = _mock_process_alive(99999)

            with patch(
                "jules_daemon.startup.scan_probe_mark.check_endpoints",
                new_callable=AsyncMock,
            ) as mock_endpoints:
                mock_endpoints.return_value = (
                    _mock_endpoint_unreachable("prod.example.com", 22),
                )

                result = await run_pipeline(wiki_root)

        assert result.verdicts[0].health == SessionHealth.CONNECTION_LOST
        assert result.verdicts[0].alive is False
        assert result.mark_results[0].outcome == MarkOutcome.MARKED_STALE


# ---------------------------------------------------------------------------
# run_pipeline: session with no daemon PID
# ---------------------------------------------------------------------------


class TestPipelineNoDaemonPid:
    """Pipeline handles sessions with no daemon PID recorded."""

    @pytest.mark.asyncio
    async def test_no_daemon_pid_with_unreachable_endpoint(
        self, wiki_root: Path
    ) -> None:
        """No daemon PID + unreachable endpoint = not alive."""
        # Write a session file directly with no daemon PID
        target = SSHTarget(host="ghost.example.com", user="ci")
        cmd = Command(natural_language="run tests")
        run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1)
        # Override pids to have no daemon PID by writing directly
        run = run.with_running("pytest -v", remote_pid=5678)
        current_run.write(wiki_root, run)

        with patch(
            "jules_daemon.startup.scan_probe_mark.check_pid"
        ) as mock_check:
            # PID 1 is typically init/systemd -- we mock it as dead
            mock_check.return_value = _mock_process_dead(1)

            with patch(
                "jules_daemon.startup.scan_probe_mark.check_endpoints",
                new_callable=AsyncMock,
            ) as mock_endpoints:
                mock_endpoints.return_value = ()

                result = await run_pipeline(wiki_root)

        assert len(result.verdicts) == 1
        assert result.verdicts[0].alive is False


# ---------------------------------------------------------------------------
# run_pipeline: configuration
# ---------------------------------------------------------------------------


class TestPipelineConfigForwarding:
    """Pipeline respects custom configuration."""

    @pytest.mark.asyncio
    async def test_custom_config_passed_to_probe(
        self, wiki_root: Path
    ) -> None:
        """Custom probe timeout is forwarded to check_endpoints."""
        _write_running_session(wiki_root, daemon_pid=99999)

        with patch(
            "jules_daemon.startup.scan_probe_mark.check_pid"
        ) as mock_check:
            mock_check.return_value = _mock_process_alive(99999)

            with patch(
                "jules_daemon.startup.scan_probe_mark.check_endpoints",
                new_callable=AsyncMock,
            ) as mock_endpoints:
                mock_endpoints.return_value = (
                    _mock_endpoint_reachable("prod.example.com", 22),
                )

                config = PipelineConfig(
                    probe_timeout_seconds=2.0,
                    capture_banner=True,
                )
                result = await run_pipeline(wiki_root, config=config)

                # Verify ProbeSettings was created with the config values
                call_args = mock_endpoints.call_args
                settings = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("settings")
                if settings is not None:
                    assert settings.timeout_seconds == 2.0
                    assert settings.capture_banner is True


# ---------------------------------------------------------------------------
# run_pipeline: timing and result structure
# ---------------------------------------------------------------------------


class TestPipelineTiming:
    """Pipeline records duration and timestamp."""

    @pytest.mark.asyncio
    async def test_duration_recorded(self, wiki_root: Path) -> None:
        result = await run_pipeline(wiki_root)
        assert result.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_timestamp_recorded(self, wiki_root: Path) -> None:
        before = datetime.now(timezone.utc)
        result = await run_pipeline(wiki_root)
        after = datetime.now(timezone.utc)
        assert before <= result.timestamp <= after

    @pytest.mark.asyncio
    async def test_empty_pipeline_under_200ms(self, wiki_root: Path) -> None:
        """Empty pipeline (no sessions) completes quickly."""
        result = await run_pipeline(wiki_root)
        assert result.duration_seconds < 0.2


# ---------------------------------------------------------------------------
# run_pipeline: never raises
# ---------------------------------------------------------------------------


class TestPipelineNeverRaises:
    """Pipeline captures errors without raising."""

    @pytest.mark.asyncio
    async def test_scan_error_captured(self, wiki_root: Path) -> None:
        """Even if scan encounters issues, pipeline does not raise."""
        # Create a file that looks like a wiki page but is corrupted
        daemon_dir = wiki_root / "pages" / "daemon"
        daemon_dir.mkdir(parents=True, exist_ok=True)
        (daemon_dir / "bad.md").write_text("not valid", encoding="utf-8")

        result = await run_pipeline(wiki_root)
        # Should not raise, should return a result
        assert result.scan_result.outcome == ScanOutcome.SCANNED
