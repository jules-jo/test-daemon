"""Tests for the OutputMonitor real-time analysis service.

Verifies that the OutputMonitor:
- Subscribes to a JobOutputBroadcaster and reads lines in the background
- Maintains a line buffer of observed output lines
- Has a clean start/stop/async-context-manager lifecycle
- Reports state transitions correctly (IDLE -> RUNNING -> STOPPED)
- Stops gracefully when the output stream ends (is_end sentinel)
- Provides a snapshot of accumulated lines and line count
- Respects max_buffer_lines configuration (ring-buffer semantics)
- Can be tapped from multiple consumers via the snapshot interface
- Handles errors during line reading without crashing
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from jules_daemon.monitor.output_broadcaster import (
    JobOutputBroadcaster,
    OutputLine,
    SubscriberHandle,
)
from jules_daemon.monitor.output_monitor import (
    OutputMonitor,
    OutputMonitorConfig,
    OutputMonitorSnapshot,
    OutputMonitorState,
)


# ---------------------------------------------------------------------------
# OutputMonitorConfig
# ---------------------------------------------------------------------------


class TestOutputMonitorConfig:
    """Tests for the immutable monitor configuration."""

    def test_defaults(self) -> None:
        config = OutputMonitorConfig()
        assert config.max_buffer_lines == 5000
        assert config.stop_timeout_seconds == 5.0

    def test_custom_values(self) -> None:
        config = OutputMonitorConfig(
            max_buffer_lines=100,
            stop_timeout_seconds=2.0,
        )
        assert config.max_buffer_lines == 100
        assert config.stop_timeout_seconds == 2.0

    def test_frozen(self) -> None:
        config = OutputMonitorConfig()
        with pytest.raises(AttributeError):
            config.max_buffer_lines = 99  # type: ignore[misc]

    def test_max_buffer_lines_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_buffer_lines must be positive"):
            OutputMonitorConfig(max_buffer_lines=0)

    def test_stop_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="stop_timeout_seconds must be positive"):
            OutputMonitorConfig(stop_timeout_seconds=0.0)


# ---------------------------------------------------------------------------
# OutputMonitorSnapshot
# ---------------------------------------------------------------------------


class TestOutputMonitorSnapshot:
    """Tests for the immutable monitor snapshot."""

    def test_empty_snapshot(self) -> None:
        snap = OutputMonitorSnapshot(
            job_id="job-1",
            state=OutputMonitorState.IDLE,
            lines=(),
            total_lines_observed=0,
        )
        assert snap.job_id == "job-1"
        assert snap.state is OutputMonitorState.IDLE
        assert snap.lines == ()
        assert snap.total_lines_observed == 0
        assert snap.line_count == 0

    def test_snapshot_with_lines(self) -> None:
        snap = OutputMonitorSnapshot(
            job_id="job-1",
            state=OutputMonitorState.RUNNING,
            lines=("PASSED test_a", "FAILED test_b"),
            total_lines_observed=2,
        )
        assert snap.line_count == 2
        assert snap.lines == ("PASSED test_a", "FAILED test_b")
        assert snap.total_lines_observed == 2

    def test_frozen(self) -> None:
        snap = OutputMonitorSnapshot(
            job_id="job-1",
            state=OutputMonitorState.IDLE,
            lines=(),
            total_lines_observed=0,
        )
        with pytest.raises(AttributeError):
            snap.job_id = "mutated"  # type: ignore[misc]

    def test_total_lines_observed_must_not_be_negative(self) -> None:
        with pytest.raises(
            ValueError, match="total_lines_observed must not be negative"
        ):
            OutputMonitorSnapshot(
                job_id="job-1",
                state=OutputMonitorState.IDLE,
                lines=(),
                total_lines_observed=-1,
            )


# ---------------------------------------------------------------------------
# OutputMonitorState
# ---------------------------------------------------------------------------


class TestOutputMonitorState:
    """Tests for the lifecycle state enum."""

    def test_values(self) -> None:
        assert OutputMonitorState.IDLE.value == "idle"
        assert OutputMonitorState.RUNNING.value == "running"
        assert OutputMonitorState.STOPPING.value == "stopping"
        assert OutputMonitorState.STOPPED.value == "stopped"
        assert OutputMonitorState.ERROR.value == "error"


# ---------------------------------------------------------------------------
# OutputMonitor lifecycle
# ---------------------------------------------------------------------------


class TestOutputMonitorLifecycle:
    """Tests for start/stop and async context manager lifecycle."""

    @pytest.mark.asyncio
    async def test_initial_state_is_idle(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        monitor = OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        )
        assert monitor.state is OutputMonitorState.IDLE
        assert monitor.job_id == "job-1"

    @pytest.mark.asyncio
    async def test_start_transitions_to_running(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        monitor = OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        )
        await monitor.start()
        assert monitor.state is OutputMonitorState.RUNNING

        await monitor.stop()
        assert monitor.state is OutputMonitorState.STOPPED

    @pytest.mark.asyncio
    async def test_start_when_already_running_raises(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        monitor = OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        )
        await monitor.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await monitor.start()
        finally:
            await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_when_idle_is_noop(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        monitor = OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        )
        # Should not raise
        await monitor.stop()
        assert monitor.state is OutputMonitorState.IDLE

    @pytest.mark.asyncio
    async def test_stop_when_already_stopped_is_noop(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        monitor = OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        )
        await monitor.start()
        await monitor.stop()
        assert monitor.state is OutputMonitorState.STOPPED

        # Second stop is a no-op
        await monitor.stop()
        assert monitor.state is OutputMonitorState.STOPPED

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")

        async with OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        ) as monitor:
            assert monitor.state is OutputMonitorState.RUNNING

        assert monitor.state is OutputMonitorState.STOPPED

    @pytest.mark.asyncio
    async def test_unregistered_job_raises_on_start(self) -> None:
        broadcaster = JobOutputBroadcaster()
        monitor = OutputMonitor(
            broadcaster=broadcaster,
            job_id="nonexistent",
        )
        with pytest.raises(ValueError, match="not registered"):
            await monitor.start()


# ---------------------------------------------------------------------------
# Line-buffering loop
# ---------------------------------------------------------------------------


class TestOutputMonitorLineBuffering:
    """Tests for the core line-buffering loop that reads from the stream."""

    @pytest.mark.asyncio
    async def test_buffers_published_lines(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")

        async with OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        ) as monitor:
            # Publish lines to the broadcaster
            broadcaster.publish("job-1", "PASSED test_login")
            broadcaster.publish("job-1", "PASSED test_logout")
            broadcaster.publish("job-1", "FAILED test_signup")

            # Allow the monitor's background task to process
            await asyncio.sleep(0.05)

            snap = monitor.snapshot()
            assert snap.line_count == 3
            assert snap.total_lines_observed == 3
            assert snap.lines == (
                "PASSED test_login",
                "PASSED test_logout",
                "FAILED test_signup",
            )

    @pytest.mark.asyncio
    async def test_stops_on_end_of_stream(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")

        monitor = OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        )
        await monitor.start()

        # Publish some lines then signal end
        broadcaster.publish("job-1", "PASSED test_a")
        broadcaster.unregister_job("job-1")  # sends end sentinel

        # Allow processing
        await asyncio.sleep(0.05)

        assert monitor.state is OutputMonitorState.STOPPED
        snap = monitor.snapshot()
        assert snap.total_lines_observed == 1
        assert snap.lines == ("PASSED test_a",)

    @pytest.mark.asyncio
    async def test_ring_buffer_eviction(self) -> None:
        """When max_buffer_lines is exceeded, oldest lines are evicted."""
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")

        config = OutputMonitorConfig(max_buffer_lines=3)
        async with OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
            config=config,
        ) as monitor:
            # Publish 5 lines, only the last 3 should be retained
            for i in range(5):
                broadcaster.publish("job-1", f"line-{i}")

            await asyncio.sleep(0.05)

            snap = monitor.snapshot()
            assert snap.line_count == 3
            assert snap.total_lines_observed == 5
            # Only the last 3 lines retained
            assert snap.lines == ("line-2", "line-3", "line-4")

    @pytest.mark.asyncio
    async def test_empty_lines_are_skipped(self) -> None:
        """Lines with empty text content are not buffered."""
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")

        async with OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        ) as monitor:
            broadcaster.publish("job-1", "real content")
            broadcaster.publish("job-1", "")
            broadcaster.publish("job-1", "more content")

            await asyncio.sleep(0.05)

            snap = monitor.snapshot()
            assert snap.line_count == 2
            assert snap.lines == ("real content", "more content")


# ---------------------------------------------------------------------------
# Snapshot access (tapping)
# ---------------------------------------------------------------------------


class TestOutputMonitorSnapshot_Access:
    """Tests for tapping into the monitor's state via snapshot()."""

    @pytest.mark.asyncio
    async def test_snapshot_before_start(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        monitor = OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        )
        snap = monitor.snapshot()
        assert snap.state is OutputMonitorState.IDLE
        assert snap.lines == ()
        assert snap.total_lines_observed == 0

    @pytest.mark.asyncio
    async def test_snapshot_reflects_running_state(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")

        async with OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        ) as monitor:
            broadcaster.publish("job-1", "some output")
            await asyncio.sleep(0.05)

            snap = monitor.snapshot()
            assert snap.state is OutputMonitorState.RUNNING

    @pytest.mark.asyncio
    async def test_snapshot_after_stop(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")

        monitor = OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        )
        await monitor.start()
        broadcaster.publish("job-1", "captured line")
        await asyncio.sleep(0.05)
        await monitor.stop()

        snap = monitor.snapshot()
        assert snap.state is OutputMonitorState.STOPPED
        assert snap.lines == ("captured line",)
        assert snap.total_lines_observed == 1

    @pytest.mark.asyncio
    async def test_multiple_snapshots_are_independent(self) -> None:
        """Each snapshot call returns a new immutable object."""
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")

        async with OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        ) as monitor:
            broadcaster.publish("job-1", "line A")
            await asyncio.sleep(0.05)
            snap1 = monitor.snapshot()

            broadcaster.publish("job-1", "line B")
            await asyncio.sleep(0.05)
            snap2 = monitor.snapshot()

            # snap1 should not have been mutated
            assert snap1.line_count == 1
            assert snap2.line_count == 2


# ---------------------------------------------------------------------------
# get_lines convenience
# ---------------------------------------------------------------------------


class TestOutputMonitorGetLines:
    """Tests for the get_lines convenience method."""

    @pytest.mark.asyncio
    async def test_get_lines_returns_all(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")

        async with OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        ) as monitor:
            broadcaster.publish("job-1", "line-0")
            broadcaster.publish("job-1", "line-1")
            broadcaster.publish("job-1", "line-2")
            await asyncio.sleep(0.05)

            lines = monitor.get_lines()
            assert lines == ("line-0", "line-1", "line-2")

    @pytest.mark.asyncio
    async def test_get_lines_with_last_n(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")

        async with OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        ) as monitor:
            for i in range(5):
                broadcaster.publish("job-1", f"line-{i}")
            await asyncio.sleep(0.05)

            lines = monitor.get_lines(last_n=2)
            assert lines == ("line-3", "line-4")

    @pytest.mark.asyncio
    async def test_get_lines_with_last_n_exceeding_buffer(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")

        async with OutputMonitor(
            broadcaster=broadcaster,
            job_id="job-1",
        ) as monitor:
            broadcaster.publish("job-1", "line-0")
            await asyncio.sleep(0.05)

            lines = monitor.get_lines(last_n=100)
            assert lines == ("line-0",)
