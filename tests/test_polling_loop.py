"""Tests for the async SSH polling loop.

Verifies that the polling loop:
- Polls the SSH output reader at a configurable interval (default <=10s)
- Feeds each output chunk to the MonitorStatus update method
- Exposes start/stop/reconfigure controls
- Handles reader errors gracefully without crashing the loop
- Stops automatically when the SSH session reaches a terminal state
- Maintains sequence ordering across iterations
- Never lets status be more than 10s stale while running
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from jules_daemon.monitor.polling_loop import (
    PollingConfig,
    PollingLoop,
    PollingState,
    StatusCallback,
)
from jules_daemon.ssh.reader import SSHChannelHandle, SSHOutput
from jules_daemon.wiki.monitor_status import (
    MonitorStatus,
    OutputPhase,
    ParsedState,
)


# ---------------------------------------------------------------------------
# Test fixtures: fake SSH channel handle
# ---------------------------------------------------------------------------


class FakeChannel:
    """Fake SSH channel for testing. Satisfies SSHChannelHandle protocol."""

    def __init__(self) -> None:
        self._stdout_queue: list[bytes] = []
        self._stderr_queue: list[bytes] = []
        self._is_closed: bool = False
        self._eof: bool = False
        self._exit_ready: bool = False
        self._exit_status: int = 0

    def enqueue_stdout(self, data: bytes) -> None:
        self._stdout_queue.append(data)

    def enqueue_stderr(self, data: bytes) -> None:
        self._stderr_queue.append(data)

    def set_exit(self, code: int) -> None:
        self._exit_ready = True
        self._exit_status = code
        self._eof = True

    def close(self) -> None:
        self._is_closed = True

    # -- SSHChannelHandle protocol --

    def recv_ready(self) -> bool:
        return len(self._stdout_queue) > 0

    def recv(self, nbytes: int) -> bytes:
        if not self._stdout_queue:
            return b""
        return self._stdout_queue.pop(0)

    def recv_stderr_ready(self) -> bool:
        return len(self._stderr_queue) > 0

    def recv_stderr(self, nbytes: int) -> bytes:
        if not self._stderr_queue:
            return b""
        return self._stderr_queue.pop(0)

    @property
    def closed(self) -> bool:
        return self._is_closed

    def eof_received(self) -> bool:
        return self._eof

    def exit_status_is_ready(self) -> bool:
        return self._exit_ready

    def get_exit_status(self) -> int:
        return self._exit_status


# ---------------------------------------------------------------------------
# PollingConfig tests
# ---------------------------------------------------------------------------


class TestPollingConfig:
    def test_defaults(self) -> None:
        config = PollingConfig()
        assert config.interval_seconds == 10.0
        assert config.max_consecutive_errors == 5

    def test_default_interval_at_most_10s(self) -> None:
        """Status freshness SLA: default interval must be <=10s."""
        config = PollingConfig()
        assert config.interval_seconds <= 10.0

    def test_custom_interval(self) -> None:
        config = PollingConfig(interval_seconds=5.0)
        assert config.interval_seconds == 5.0

    def test_custom_max_errors(self) -> None:
        config = PollingConfig(max_consecutive_errors=10)
        assert config.max_consecutive_errors == 10

    def test_frozen(self) -> None:
        config = PollingConfig()
        with pytest.raises(AttributeError):
            config.interval_seconds = 5.0  # type: ignore[misc]

    def test_negative_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="interval_seconds must be positive"):
            PollingConfig(interval_seconds=-1.0)

    def test_zero_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="interval_seconds must be positive"):
            PollingConfig(interval_seconds=0.0)

    def test_negative_max_errors_raises(self) -> None:
        with pytest.raises(ValueError, match="max_consecutive_errors must be non-negative"):
            PollingConfig(max_consecutive_errors=-1)


# ---------------------------------------------------------------------------
# PollingState enum
# ---------------------------------------------------------------------------


class TestPollingState:
    def test_all_states(self) -> None:
        assert PollingState.IDLE.value == "idle"
        assert PollingState.RUNNING.value == "running"
        assert PollingState.STOPPING.value == "stopping"
        assert PollingState.STOPPED.value == "stopped"
        assert PollingState.ERROR.value == "error"


# ---------------------------------------------------------------------------
# PollingLoop construction
# ---------------------------------------------------------------------------


class TestPollingLoopConstruction:
    def test_create_with_defaults(self) -> None:
        channel = FakeChannel()
        callback = AsyncMock()
        loop = PollingLoop(
            channel=channel,
            session_id="test-session",
            on_status_update=callback,
        )
        assert loop.state == PollingState.IDLE
        assert loop.session_id == "test-session"
        assert loop.latest_status is None

    def test_create_with_custom_config(self) -> None:
        channel = FakeChannel()
        callback = AsyncMock()
        config = PollingConfig(interval_seconds=3.0)
        loop = PollingLoop(
            channel=channel,
            session_id="test-session",
            on_status_update=callback,
            config=config,
        )
        assert loop.config.interval_seconds == 3.0

    def test_empty_session_id_raises(self) -> None:
        channel = FakeChannel()
        callback = AsyncMock()
        with pytest.raises(ValueError, match="session_id must not be empty"):
            PollingLoop(
                channel=channel,
                session_id="",
                on_status_update=callback,
            )


# ---------------------------------------------------------------------------
# Start / stop lifecycle
# ---------------------------------------------------------------------------


class TestPollingLoopLifecycle:
    @pytest.mark.asyncio
    async def test_start_transitions_to_running(self) -> None:
        channel = FakeChannel()
        callback = AsyncMock()
        loop = PollingLoop(
            channel=channel,
            session_id="lifecycle-1",
            on_status_update=callback,
            config=PollingConfig(interval_seconds=0.05),
        )
        await loop.start()
        assert loop.state == PollingState.RUNNING
        await loop.stop()

    @pytest.mark.asyncio
    async def test_stop_transitions_to_stopped(self) -> None:
        channel = FakeChannel()
        callback = AsyncMock()
        loop = PollingLoop(
            channel=channel,
            session_id="lifecycle-2",
            on_status_update=callback,
            config=PollingConfig(interval_seconds=0.05),
        )
        await loop.start()
        await loop.stop()
        assert loop.state == PollingState.STOPPED

    @pytest.mark.asyncio
    async def test_double_start_raises(self) -> None:
        channel = FakeChannel()
        callback = AsyncMock()
        loop = PollingLoop(
            channel=channel,
            session_id="lifecycle-3",
            on_status_update=callback,
            config=PollingConfig(interval_seconds=0.05),
        )
        await loop.start()
        with pytest.raises(RuntimeError, match="already running"):
            await loop.start()
        await loop.stop()

    @pytest.mark.asyncio
    async def test_stop_before_start_is_noop(self) -> None:
        channel = FakeChannel()
        callback = AsyncMock()
        loop = PollingLoop(
            channel=channel,
            session_id="lifecycle-4",
            on_status_update=callback,
        )
        await loop.stop()
        assert loop.state == PollingState.IDLE


# ---------------------------------------------------------------------------
# Polling reads output and invokes callback
# ---------------------------------------------------------------------------


class TestPollingReadsOutput:
    @pytest.mark.asyncio
    async def test_reads_stdout_and_calls_callback(self) -> None:
        channel = FakeChannel()
        channel.enqueue_stdout(b"PASSED test_one\n")
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        loop = PollingLoop(
            channel=channel,
            session_id="read-1",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02),
        )
        await loop.start()
        # Give the loop time for at least one iteration
        await asyncio.sleep(0.1)
        await loop.stop()

        assert len(collected) >= 1
        first = collected[0]
        assert first.session_id == "read-1"
        assert "PASSED test_one" in first.raw_output_chunk

    @pytest.mark.asyncio
    async def test_reads_stderr_included_in_output(self) -> None:
        channel = FakeChannel()
        channel.enqueue_stderr(b"WARNING: deprecated API\n")
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        loop = PollingLoop(
            channel=channel,
            session_id="read-2",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02),
        )
        await loop.start()
        await asyncio.sleep(0.1)
        await loop.stop()

        assert len(collected) >= 1
        first = collected[0]
        assert "WARNING: deprecated API" in first.raw_output_chunk

    @pytest.mark.asyncio
    async def test_multiple_reads_produce_increasing_sequence(self) -> None:
        channel = FakeChannel()
        collected: list[MonitorStatus] = []
        iteration = 0

        async def collect(status: MonitorStatus) -> None:
            nonlocal iteration
            collected.append(status)
            iteration += 1
            if iteration < 3:
                channel.enqueue_stdout(f"line {iteration + 1}\n".encode())

        channel.enqueue_stdout(b"line 1\n")

        loop = PollingLoop(
            channel=channel,
            session_id="seq-1",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02),
        )
        await loop.start()
        await asyncio.sleep(0.2)
        await loop.stop()

        # Verify monotonically increasing sequence numbers
        seq_numbers = [s.sequence_number for s in collected]
        for i in range(1, len(seq_numbers)):
            assert seq_numbers[i] > seq_numbers[i - 1], (
                f"Sequence not increasing: {seq_numbers}"
            )


# ---------------------------------------------------------------------------
# Automatic stop on terminal state (exit code received)
# ---------------------------------------------------------------------------


class TestPollingAutoStop:
    @pytest.mark.asyncio
    async def test_stops_when_exit_code_received(self) -> None:
        channel = FakeChannel()
        channel.enqueue_stdout(b"all tests passed\n")
        channel.set_exit(0)
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        loop = PollingLoop(
            channel=channel,
            session_id="exit-1",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02),
        )
        await loop.start()
        await asyncio.sleep(0.15)

        # Loop should auto-stop on terminal state
        assert loop.state == PollingState.STOPPED

        # Should have received at least one callback with terminal status
        terminal = [s for s in collected if s.is_terminal]
        assert len(terminal) >= 1
        assert terminal[-1].exit_status == 0

    @pytest.mark.asyncio
    async def test_stops_when_nonzero_exit_code(self) -> None:
        channel = FakeChannel()
        channel.enqueue_stdout(b"3 failed\n")
        channel.set_exit(1)
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        loop = PollingLoop(
            channel=channel,
            session_id="exit-2",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02),
        )
        await loop.start()
        await asyncio.sleep(0.15)

        assert loop.state == PollingState.STOPPED
        terminal = [s for s in collected if s.is_terminal]
        assert len(terminal) >= 1
        assert terminal[-1].exit_status == 1

    @pytest.mark.asyncio
    async def test_stops_when_channel_closed(self) -> None:
        channel = FakeChannel()
        channel.set_exit(0)
        channel.close()
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        loop = PollingLoop(
            channel=channel,
            session_id="exit-3",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02),
        )
        await loop.start()
        await asyncio.sleep(0.15)

        assert loop.state == PollingState.STOPPED


# ---------------------------------------------------------------------------
# Reconfigure controls
# ---------------------------------------------------------------------------


class TestPollingReconfigure:
    @pytest.mark.asyncio
    async def test_reconfigure_changes_interval(self) -> None:
        channel = FakeChannel()
        callback = AsyncMock()
        loop = PollingLoop(
            channel=channel,
            session_id="reconf-1",
            on_status_update=callback,
            config=PollingConfig(interval_seconds=0.05),
        )
        await loop.start()

        new_config = PollingConfig(interval_seconds=0.02)
        loop.reconfigure(new_config)
        assert loop.config.interval_seconds == 0.02

        await loop.stop()

    @pytest.mark.asyncio
    async def test_reconfigure_while_idle(self) -> None:
        channel = FakeChannel()
        callback = AsyncMock()
        loop = PollingLoop(
            channel=channel,
            session_id="reconf-2",
            on_status_update=callback,
        )
        new_config = PollingConfig(interval_seconds=5.0)
        loop.reconfigure(new_config)
        assert loop.config.interval_seconds == 5.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class ErrorChannel:
    """Channel that raises on recv."""

    def __init__(self) -> None:
        self._call_count = 0
        self._error_count = 2
        self._is_closed = False

    def recv_ready(self) -> bool:
        return True

    def recv(self, nbytes: int) -> bytes:
        self._call_count += 1
        if self._call_count <= self._error_count:
            raise OSError("Connection reset")
        return b"recovered\n"

    def recv_stderr_ready(self) -> bool:
        return False

    def recv_stderr(self, nbytes: int) -> bytes:
        return b""

    @property
    def closed(self) -> bool:
        return self._is_closed

    def eof_received(self) -> bool:
        return False

    def exit_status_is_ready(self) -> bool:
        return False

    def get_exit_status(self) -> int:
        return -1


class TestPollingErrorHandling:
    @pytest.mark.asyncio
    async def test_survives_transient_reader_errors(self) -> None:
        """Loop should not crash on individual read errors."""
        channel = ErrorChannel()
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        loop = PollingLoop(
            channel=channel,
            session_id="err-1",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02, max_consecutive_errors=5),
        )
        await loop.start()
        await asyncio.sleep(0.2)
        await loop.stop()

        # Loop survived and eventually read successfully
        assert loop.state == PollingState.STOPPED
        assert loop.consecutive_errors == 0 or len(collected) > 0

    @pytest.mark.asyncio
    async def test_stops_after_max_consecutive_errors(self) -> None:
        """Loop transitions to ERROR state after too many consecutive errors."""

        class AlwaysErrorChannel:
            def recv_ready(self) -> bool:
                return True

            def recv(self, nbytes: int) -> bytes:
                raise OSError("Persistent failure")

            def recv_stderr_ready(self) -> bool:
                return False

            def recv_stderr(self, nbytes: int) -> bytes:
                return b""

            @property
            def closed(self) -> bool:
                return False

            def eof_received(self) -> bool:
                return False

            def exit_status_is_ready(self) -> bool:
                return False

            def get_exit_status(self) -> int:
                return -1

        channel = AlwaysErrorChannel()
        callback = AsyncMock()
        loop = PollingLoop(
            channel=channel,
            session_id="err-2",
            on_status_update=callback,
            config=PollingConfig(interval_seconds=0.02, max_consecutive_errors=3),
        )
        await loop.start()
        await asyncio.sleep(0.2)

        assert loop.state == PollingState.ERROR


# ---------------------------------------------------------------------------
# latest_status property
# ---------------------------------------------------------------------------


class TestPollingLatestStatus:
    @pytest.mark.asyncio
    async def test_latest_status_updated_after_read(self) -> None:
        channel = FakeChannel()
        channel.enqueue_stdout(b"test output\n")
        callback = AsyncMock()

        loop = PollingLoop(
            channel=channel,
            session_id="latest-1",
            on_status_update=callback,
            config=PollingConfig(interval_seconds=0.02),
        )
        assert loop.latest_status is None

        await loop.start()
        await asyncio.sleep(0.1)
        await loop.stop()

        assert loop.latest_status is not None
        assert loop.latest_status.session_id == "latest-1"

    @pytest.mark.asyncio
    async def test_latest_status_is_most_recent(self) -> None:
        channel = FakeChannel()
        collected: list[MonitorStatus] = []
        iteration = 0

        async def collect(status: MonitorStatus) -> None:
            nonlocal iteration
            collected.append(status)
            iteration += 1
            if iteration < 3:
                channel.enqueue_stdout(f"output {iteration + 1}\n".encode())

        channel.enqueue_stdout(b"output 1\n")

        loop = PollingLoop(
            channel=channel,
            session_id="latest-2",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02),
        )
        await loop.start()
        await asyncio.sleep(0.2)
        await loop.stop()

        if collected:
            assert loop.latest_status == collected[-1]


# ---------------------------------------------------------------------------
# Polling interval timing
# ---------------------------------------------------------------------------


class TestPollingInterval:
    @pytest.mark.asyncio
    async def test_no_data_polls_do_not_fire_callback(self) -> None:
        """Empty reads (no stdout/stderr) should not fire callback."""
        channel = FakeChannel()
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        loop = PollingLoop(
            channel=channel,
            session_id="interval-1",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02),
        )
        await loop.start()
        await asyncio.sleep(0.1)
        await loop.stop()

        # With no data, callback should not have been called
        # (empty reads are skipped to avoid noisy status updates)
        assert len(collected) == 0

    @pytest.mark.asyncio
    async def test_callback_receives_immutable_status(self) -> None:
        """Each callback invocation receives a frozen MonitorStatus."""
        channel = FakeChannel()
        channel.enqueue_stdout(b"test data\n")
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)
            # Verify frozen
            with pytest.raises(AttributeError):
                status.session_id = "hacked"  # type: ignore[misc]

        loop = PollingLoop(
            channel=channel,
            session_id="immutable-1",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02),
        )
        await loop.start()
        await asyncio.sleep(0.1)
        await loop.stop()

        assert len(collected) >= 1


# ---------------------------------------------------------------------------
# Context manager support
# ---------------------------------------------------------------------------


class TestPollingContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        channel = FakeChannel()
        channel.enqueue_stdout(b"context test\n")
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        loop = PollingLoop(
            channel=channel,
            session_id="ctx-1",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02),
        )
        async with loop:
            await asyncio.sleep(0.1)

        assert loop.state == PollingState.STOPPED


# ---------------------------------------------------------------------------
# Edge cases for coverage
# ---------------------------------------------------------------------------


class TestPollingEdgeCases:
    @pytest.mark.asyncio
    async def test_stdout_and_stderr_combined(self) -> None:
        """When both stdout and stderr have data, stderr gets [stderr] prefix."""
        channel = FakeChannel()
        channel.enqueue_stdout(b"output line\n")
        channel.enqueue_stderr(b"error line\n")
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)

        loop = PollingLoop(
            channel=channel,
            session_id="combo-1",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02),
        )
        await loop.start()
        await asyncio.sleep(0.1)
        await loop.stop()

        assert len(collected) >= 1
        text = collected[0].raw_output_chunk
        assert "output line" in text
        assert "[stderr]" in text
        assert "error line" in text

    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash_loop(self) -> None:
        """If the callback raises, the loop should continue."""
        channel = FakeChannel()
        call_count = 0

        async def failing_callback(status: MonitorStatus) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                channel.enqueue_stdout(b"second chunk\n")
                raise RuntimeError("callback exploded")

        channel.enqueue_stdout(b"first chunk\n")

        loop = PollingLoop(
            channel=channel,
            session_id="cb-err-1",
            on_status_update=failing_callback,
            config=PollingConfig(interval_seconds=0.02),
        )
        await loop.start()
        await asyncio.sleep(0.15)
        await loop.stop()

        # Loop should have survived the callback error and processed more
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_terminal_with_prior_status_uses_with_exit(self) -> None:
        """When exit code comes after prior updates, with_exit path is used."""
        channel = FakeChannel()
        collected: list[MonitorStatus] = []

        async def collect(status: MonitorStatus) -> None:
            collected.append(status)
            if len(collected) == 1:
                # After first non-terminal read, trigger exit
                channel.enqueue_stdout(b"final output\n")
                channel.set_exit(0)

        channel.enqueue_stdout(b"initial output\n")

        loop = PollingLoop(
            channel=channel,
            session_id="term-prior-1",
            on_status_update=collect,
            config=PollingConfig(interval_seconds=0.02),
        )
        await loop.start()
        await asyncio.sleep(0.2)

        assert loop.state == PollingState.STOPPED
        assert len(collected) >= 2
        # Last status should be terminal with exit code
        terminal = [s for s in collected if s.is_terminal]
        assert len(terminal) >= 1
        assert terminal[-1].exit_status == 0
        # Sequence should be > 1 since we had prior updates
        assert terminal[-1].sequence_number > 1
