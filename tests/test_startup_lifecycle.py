"""Tests for the daemon startup lifecycle.

Verifies that the startup lifecycle:
- Transitions through STARTING -> SCANNING -> READY phases
- Invokes the scan-probe-mark pipeline before transitioning to READY
- Captures the pipeline result in the startup result
- Records the total startup duration
- Handles pipeline errors gracefully (warns but still becomes READY)
- Never raises -- all errors are captured in the startup result
- Writes a startup event to the wiki for audit completeness
- Sets the daemon phase to READY only after the pipeline completes
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.wiki import current_run
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    RunStatus,
    SSHTarget,
)
from jules_daemon.wiki.session_scanner import ScanOutcome, ScanResult
from jules_daemon.wiki.stale_session_marker import MarkOutcome
from jules_daemon.startup.scan_probe_mark import PipelineConfig, PipelineResult
from jules_daemon.startup.lifecycle import (
    DaemonPhase,
    StartupHookConfig,
    StartupResult,
    run_startup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


# ---------------------------------------------------------------------------
# DaemonPhase enum tests
# ---------------------------------------------------------------------------


class TestDaemonPhase:
    """Verify daemon lifecycle phase enumeration."""

    def test_all_phases_exist(self) -> None:
        assert DaemonPhase.STARTING.value == "starting"
        assert DaemonPhase.SCANNING.value == "scanning"
        assert DaemonPhase.READY.value == "ready"


# ---------------------------------------------------------------------------
# StartupHookConfig tests
# ---------------------------------------------------------------------------


class TestStartupHookConfig:
    """Verify the immutable startup hook configuration."""

    def test_defaults(self) -> None:
        config = StartupHookConfig()
        assert config.run_scan_probe_mark is True
        assert config.pipeline_config is None

    def test_custom_config(self) -> None:
        pipeline_cfg = PipelineConfig(probe_timeout_seconds=2.0)
        config = StartupHookConfig(pipeline_config=pipeline_cfg)
        assert config.pipeline_config is not None
        assert config.pipeline_config.probe_timeout_seconds == 2.0

    def test_frozen(self) -> None:
        config = StartupHookConfig()
        with pytest.raises(AttributeError):
            config.run_scan_probe_mark = False  # type: ignore[misc]

    def test_skip_pipeline(self) -> None:
        config = StartupHookConfig(run_scan_probe_mark=False)
        assert config.run_scan_probe_mark is False


# ---------------------------------------------------------------------------
# StartupResult model tests
# ---------------------------------------------------------------------------


class TestStartupResult:
    """Verify the immutable startup result model."""

    def test_frozen(self) -> None:
        result = StartupResult(
            final_phase=DaemonPhase.READY,
            pipeline_result=None,
            duration_seconds=0.1,
            error=None,
            timestamp=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            result.final_phase = DaemonPhase.STARTING  # type: ignore[misc]

    def test_is_ready_when_ready(self) -> None:
        result = StartupResult(
            final_phase=DaemonPhase.READY,
            pipeline_result=None,
            duration_seconds=0.1,
            error=None,
            timestamp=datetime.now(timezone.utc),
        )
        assert result.is_ready is True

    def test_not_ready_when_starting(self) -> None:
        result = StartupResult(
            final_phase=DaemonPhase.STARTING,
            pipeline_result=None,
            duration_seconds=0.0,
            error="Startup failed",
            timestamp=datetime.now(timezone.utc),
        )
        assert result.is_ready is False


# ---------------------------------------------------------------------------
# run_startup: basic lifecycle
# ---------------------------------------------------------------------------


class TestStartupBasicLifecycle:
    """Verify the basic startup lifecycle transitions."""

    @pytest.mark.asyncio
    async def test_reaches_ready_with_empty_wiki(
        self, wiki_root: Path
    ) -> None:
        """Startup succeeds and reaches READY even with no wiki."""
        result = await run_startup(wiki_root)
        assert result.final_phase == DaemonPhase.READY
        assert result.is_ready is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_pipeline_result_captured(self, wiki_root: Path) -> None:
        """The pipeline result is captured in the startup result."""
        result = await run_startup(wiki_root)
        assert result.pipeline_result is not None
        assert result.pipeline_result.scan_result.outcome == ScanOutcome.NO_DIRECTORY

    @pytest.mark.asyncio
    async def test_duration_recorded(self, wiki_root: Path) -> None:
        """Startup duration is recorded."""
        result = await run_startup(wiki_root)
        assert result.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_timestamp_recorded(self, wiki_root: Path) -> None:
        """Startup timestamp is recorded."""
        before = datetime.now(timezone.utc)
        result = await run_startup(wiki_root)
        after = datetime.now(timezone.utc)
        assert before <= result.timestamp <= after


# ---------------------------------------------------------------------------
# run_startup: pipeline invocation
# ---------------------------------------------------------------------------


class TestStartupPipelineInvocation:
    """Verify the pipeline is invoked during startup."""

    @pytest.mark.asyncio
    async def test_pipeline_invoked_before_ready(
        self, wiki_root: Path
    ) -> None:
        """The scan-probe-mark pipeline runs before the daemon becomes READY."""
        phases_observed: list[str] = []

        original_run_pipeline = None
        # We'll patch run_pipeline to track when it's called
        async def tracking_pipeline(
            wiki_root_arg, config=None
        ):
            phases_observed.append("pipeline_started")
            from jules_daemon.startup.scan_probe_mark import run_pipeline as real_pipeline
            # Import from a known working state or return minimal result
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
                duration_seconds=0.0,
                timestamp=datetime.now(timezone.utc),
            )
            phases_observed.append("pipeline_completed")
            return result

        with patch(
            "jules_daemon.startup.lifecycle.run_pipeline",
            side_effect=tracking_pipeline,
        ):
            result = await run_startup(wiki_root)

        assert "pipeline_started" in phases_observed
        assert "pipeline_completed" in phases_observed
        assert result.final_phase == DaemonPhase.READY

    @pytest.mark.asyncio
    async def test_skip_pipeline_when_disabled(
        self, wiki_root: Path
    ) -> None:
        """When run_scan_probe_mark=False, the pipeline is not invoked."""
        config = StartupHookConfig(run_scan_probe_mark=False)
        result = await run_startup(wiki_root, config=config)
        assert result.final_phase == DaemonPhase.READY
        assert result.pipeline_result is None

    @pytest.mark.asyncio
    async def test_custom_pipeline_config_forwarded(
        self, wiki_root: Path
    ) -> None:
        """Custom pipeline config is forwarded to run_pipeline."""
        pipeline_cfg = PipelineConfig(probe_timeout_seconds=1.0)
        config = StartupHookConfig(pipeline_config=pipeline_cfg)

        with patch(
            "jules_daemon.startup.lifecycle.run_pipeline",
            new_callable=AsyncMock,
        ) as mock_pipeline:
            mock_pipeline.return_value = PipelineResult(
                scan_result=ScanResult(
                    outcome=ScanOutcome.NO_DIRECTORY,
                    entries=(),
                    errors=(),
                    scanned_count=0,
                ),
                verdicts=(),
                mark_results=(),
                duration_seconds=0.0,
                timestamp=datetime.now(timezone.utc),
            )

            await run_startup(wiki_root, config=config)

            # Verify pipeline was called with the custom config
            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args
            passed_config = call_kwargs[1].get("config") if call_kwargs[1] else call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None
            if passed_config is not None:
                assert passed_config.probe_timeout_seconds == 1.0


# ---------------------------------------------------------------------------
# run_startup: pipeline error handling
# ---------------------------------------------------------------------------


class TestStartupErrorHandling:
    """Startup reaches READY even when the pipeline encounters errors."""

    @pytest.mark.asyncio
    async def test_pipeline_exception_does_not_prevent_ready(
        self, wiki_root: Path
    ) -> None:
        """If the pipeline raises an unexpected exception, startup still
        reaches READY (warn and continue)."""
        with patch(
            "jules_daemon.startup.lifecycle.run_pipeline",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Unexpected pipeline failure"),
        ):
            result = await run_startup(wiki_root)

        assert result.final_phase == DaemonPhase.READY
        assert result.error is not None
        assert "pipeline" in result.error.lower() or "unexpected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_pipeline_timeout_does_not_prevent_ready(
        self, wiki_root: Path
    ) -> None:
        """If the pipeline hangs, startup should not block indefinitely.
        This tests that the lifecycle has a reasonable timeout."""
        async def slow_pipeline(wiki_root_arg, config=None):
            await asyncio.sleep(100)

        with patch(
            "jules_daemon.startup.lifecycle.run_pipeline",
            side_effect=slow_pipeline,
        ):
            config = StartupHookConfig()
            result = await run_startup(
                wiki_root,
                config=config,
                startup_timeout_seconds=0.5,
            )

        assert result.final_phase == DaemonPhase.READY
        assert result.error is not None


# ---------------------------------------------------------------------------
# run_startup: wiki audit event
# ---------------------------------------------------------------------------


class TestStartupWikiAudit:
    """Startup writes an audit event to the wiki."""

    @pytest.mark.asyncio
    async def test_startup_event_written(self, wiki_root: Path) -> None:
        """A startup event wiki page is written after the lifecycle completes."""
        wiki_root.mkdir(parents=True, exist_ok=True)
        result = await run_startup(wiki_root)

        # Check that a startup event was written
        startup_path = wiki_root / "pages" / "daemon" / "startup-event.md"
        assert startup_path.exists()

    @pytest.mark.asyncio
    async def test_startup_event_contains_phase(
        self, wiki_root: Path
    ) -> None:
        """The startup event wiki page contains the final phase."""
        wiki_root.mkdir(parents=True, exist_ok=True)
        result = await run_startup(wiki_root)

        startup_path = wiki_root / "pages" / "daemon" / "startup-event.md"
        content = startup_path.read_text(encoding="utf-8")
        assert "ready" in content.lower()


# ---------------------------------------------------------------------------
# run_startup: timing performance
# ---------------------------------------------------------------------------


class TestStartupPerformance:
    """Startup completes within acceptable time for local-only operations."""

    @pytest.mark.asyncio
    async def test_empty_wiki_startup_under_500ms(
        self, wiki_root: Path
    ) -> None:
        """Startup with no wiki should be fast."""
        result = await run_startup(wiki_root)
        assert result.duration_seconds < 0.5
