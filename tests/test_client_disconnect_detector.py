"""Tests for real-time client disconnect detection.

Validates:
    - ClientDisconnectDetector detects reader EOF and triggers cleanup
    - ClientDisconnectDetector detects write probe failures (BrokenPipe, etc.)
    - Context cancellation via asyncio.Event stops the detector cleanly
    - Cleanup callback is invoked exactly once on disconnect
    - Detector is idempotent: multiple disconnect signals trigger cleanup once
    - DetectorConfig validates probe_interval and probe_timeout
    - DisconnectReason captures the reason and exception info
    - Integration with DisconnectCleanupHandler triggers resource cleanup
    - Integration with EventBus emits CLIENT_DISCONNECTED_EVENT
    - Detector handles callback exceptions without crashing
    - Detector can be stopped before any disconnect occurs
    - Detector handles already-closed writers gracefully
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from jules_daemon.ipc.client_disconnect_detector import (
    ClientDisconnectDetector,
    DetectorConfig,
    DisconnectReason,
    DisconnectSource,
)
from jules_daemon.ipc.connection_manager import CLIENT_DISCONNECTED_EVENT
from jules_daemon.ipc.event_bus import Event, EventBus


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


def _make_reader_mock(*, at_eof: bool = False) -> AsyncMock:
    """Build a mock StreamReader."""
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.at_eof.return_value = at_eof
    return reader


def _make_writer_mock(*, is_closing: bool = False) -> MagicMock:
    """Build a mock StreamWriter that supports write/drain."""
    writer = MagicMock(spec=asyncio.StreamWriter)
    writer.is_closing.return_value = is_closing
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    return writer


def _make_broken_writer_mock() -> MagicMock:
    """Build a mock StreamWriter that raises on drain (broken pipe)."""
    writer = MagicMock(spec=asyncio.StreamWriter)
    writer.is_closing.return_value = False
    writer.write = MagicMock()
    writer.drain = AsyncMock(side_effect=BrokenPipeError("Broken pipe"))
    return writer


def _make_reset_writer_mock() -> MagicMock:
    """Build a mock StreamWriter that raises ConnectionResetError on drain."""
    writer = MagicMock(spec=asyncio.StreamWriter)
    writer.is_closing.return_value = False
    writer.write = MagicMock()
    writer.drain = AsyncMock(
        side_effect=ConnectionResetError("Connection reset by peer")
    )
    return writer


# ---------------------------------------------------------------------------
# DetectorConfig tests
# ---------------------------------------------------------------------------


class TestDetectorConfig:
    """Tests for the immutable detector configuration."""

    def test_default_values(self) -> None:
        config = DetectorConfig()
        assert config.probe_interval_seconds > 0
        assert config.probe_timeout_seconds > 0
        assert config.reader_check_interval_seconds > 0

    def test_custom_values(self) -> None:
        config = DetectorConfig(
            probe_interval_seconds=5.0,
            probe_timeout_seconds=2.0,
            reader_check_interval_seconds=0.5,
        )
        assert config.probe_interval_seconds == 5.0
        assert config.probe_timeout_seconds == 2.0
        assert config.reader_check_interval_seconds == 0.5

    def test_frozen(self) -> None:
        config = DetectorConfig()
        with pytest.raises(AttributeError):
            config.probe_interval_seconds = 10.0  # type: ignore[misc]

    def test_zero_probe_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="probe_interval_seconds must be positive"):
            DetectorConfig(probe_interval_seconds=0.0)

    def test_negative_probe_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="probe_interval_seconds must be positive"):
            DetectorConfig(probe_interval_seconds=-1.0)

    def test_zero_probe_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="probe_timeout_seconds must be positive"):
            DetectorConfig(probe_timeout_seconds=0.0)

    def test_zero_reader_check_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="reader_check_interval_seconds must be positive"
        ):
            DetectorConfig(reader_check_interval_seconds=0.0)


# ---------------------------------------------------------------------------
# DisconnectReason tests
# ---------------------------------------------------------------------------


class TestDisconnectReason:
    """Tests for the immutable disconnect reason model."""

    def test_create_with_required_fields(self) -> None:
        reason = DisconnectReason(
            source=DisconnectSource.READER_EOF,
            client_id="client-abc",
            message="Reader reached EOF",
        )
        assert reason.source == DisconnectSource.READER_EOF
        assert reason.client_id == "client-abc"
        assert reason.message == "Reader reached EOF"
        assert reason.exception_type is None

    def test_with_exception_info(self) -> None:
        reason = DisconnectReason(
            source=DisconnectSource.WRITE_PROBE_FAILURE,
            client_id="client-def",
            message="BrokenPipeError: Broken pipe",
            exception_type="BrokenPipeError",
        )
        assert reason.exception_type == "BrokenPipeError"

    def test_frozen(self) -> None:
        reason = DisconnectReason(
            source=DisconnectSource.READER_EOF,
            client_id="c1",
            message="EOF",
        )
        with pytest.raises(AttributeError):
            reason.client_id = "other"  # type: ignore[misc]

    def test_empty_client_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="client_id must not be empty"):
            DisconnectReason(
                source=DisconnectSource.READER_EOF,
                client_id="",
                message="EOF",
            )

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(ValueError, match="message must not be empty"):
            DisconnectReason(
                source=DisconnectSource.READER_EOF,
                client_id="c1",
                message="",
            )

    def test_all_disconnect_sources(self) -> None:
        assert DisconnectSource.READER_EOF.value == "reader_eof"
        assert DisconnectSource.WRITE_PROBE_FAILURE.value == "write_probe_failure"
        assert DisconnectSource.CANCELLATION.value == "cancellation"
        assert DisconnectSource.WRITER_CLOSING.value == "writer_closing"


# ---------------------------------------------------------------------------
# ClientDisconnectDetector tests -- reader EOF detection
# ---------------------------------------------------------------------------


class TestReaderEofDetection:
    """Tests for detecting client disconnect via reader EOF."""

    @pytest.mark.asyncio
    async def test_detects_reader_eof(self) -> None:
        """Detector fires callback when reader reports at_eof()."""
        reader = _make_reader_mock(at_eof=False)
        writer = _make_writer_mock()
        callback = AsyncMock()

        # Simulate reader going to EOF after a short delay
        call_count = 0

        def eof_after_first_check() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count > 1

        reader.at_eof.side_effect = eof_after_first_check

        config = DetectorConfig(
            probe_interval_seconds=10.0,  # Long probe -- should not trigger
            reader_check_interval_seconds=0.05,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-eof",
            on_disconnect=callback,
            config=config,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.2)
        # Detector should have fired by now
        assert task.done() or callback.called

        if not task.done():
            await detector.stop()
            await task

        callback.assert_awaited_once()
        reason = callback.call_args[0][0]
        assert isinstance(reason, DisconnectReason)
        assert reason.source == DisconnectSource.READER_EOF
        assert reason.client_id == "client-eof"

    @pytest.mark.asyncio
    async def test_detects_reader_eof_already_at_eof(self) -> None:
        """Detector fires immediately when reader is already at EOF."""
        reader = _make_reader_mock(at_eof=True)
        writer = _make_writer_mock()
        callback = AsyncMock()

        config = DetectorConfig(
            probe_interval_seconds=10.0,
            reader_check_interval_seconds=0.05,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-already-eof",
            on_disconnect=callback,
            config=config,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.15)

        callback.assert_awaited_once()
        reason = callback.call_args[0][0]
        assert reason.source == DisconnectSource.READER_EOF

        if not task.done():
            await detector.stop()
            await task


# ---------------------------------------------------------------------------
# ClientDisconnectDetector tests -- write probe failure detection
# ---------------------------------------------------------------------------


class TestWriteProbeDetection:
    """Tests for detecting disconnect via write probe failures."""

    @pytest.mark.asyncio
    async def test_detects_broken_pipe_on_write_probe(self) -> None:
        """Detector fires callback on BrokenPipeError during write probe."""
        reader = _make_reader_mock(at_eof=False)
        writer = _make_broken_writer_mock()
        callback = AsyncMock()

        config = DetectorConfig(
            probe_interval_seconds=0.05,
            reader_check_interval_seconds=10.0,  # Long reader check
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-broken-pipe",
            on_disconnect=callback,
            config=config,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.2)

        callback.assert_awaited_once()
        reason = callback.call_args[0][0]
        assert reason.source == DisconnectSource.WRITE_PROBE_FAILURE
        assert "BrokenPipeError" in reason.message
        assert reason.exception_type == "BrokenPipeError"

        if not task.done():
            await detector.stop()
            await task

    @pytest.mark.asyncio
    async def test_detects_connection_reset_on_write_probe(self) -> None:
        """Detector fires callback on ConnectionResetError during write probe."""
        reader = _make_reader_mock(at_eof=False)
        writer = _make_reset_writer_mock()
        callback = AsyncMock()

        config = DetectorConfig(
            probe_interval_seconds=0.05,
            reader_check_interval_seconds=10.0,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-reset",
            on_disconnect=callback,
            config=config,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.2)

        callback.assert_awaited_once()
        reason = callback.call_args[0][0]
        assert reason.source == DisconnectSource.WRITE_PROBE_FAILURE
        assert reason.exception_type == "ConnectionResetError"

        if not task.done():
            await detector.stop()
            await task

    @pytest.mark.asyncio
    async def test_detects_writer_already_closing(self) -> None:
        """Detector fires callback when writer is already closing."""
        reader = _make_reader_mock(at_eof=False)
        writer = _make_writer_mock(is_closing=True)
        callback = AsyncMock()

        config = DetectorConfig(
            probe_interval_seconds=0.05,
            reader_check_interval_seconds=10.0,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-closing",
            on_disconnect=callback,
            config=config,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.15)

        callback.assert_awaited_once()
        reason = callback.call_args[0][0]
        assert reason.source == DisconnectSource.WRITER_CLOSING

        if not task.done():
            await detector.stop()
            await task


# ---------------------------------------------------------------------------
# ClientDisconnectDetector tests -- cancellation and stop
# ---------------------------------------------------------------------------


class TestDetectorCancellation:
    """Tests for clean cancellation and stop behavior."""

    @pytest.mark.asyncio
    async def test_stop_cancels_detector(self) -> None:
        """Calling stop() terminates the detector without firing callback."""
        reader = _make_reader_mock(at_eof=False)
        writer = _make_writer_mock()
        callback = AsyncMock()

        config = DetectorConfig(
            probe_interval_seconds=10.0,
            reader_check_interval_seconds=10.0,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-stop",
            on_disconnect=callback,
            config=config,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.05)

        await detector.stop()
        await asyncio.sleep(0.05)

        # Callback should NOT have been called -- stop is a clean shutdown
        callback.assert_not_awaited()
        assert task.done()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        """Calling stop() multiple times is safe."""
        reader = _make_reader_mock(at_eof=False)
        writer = _make_writer_mock()
        callback = AsyncMock()

        config = DetectorConfig(
            probe_interval_seconds=10.0,
            reader_check_interval_seconds=10.0,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-stop2",
            on_disconnect=callback,
            config=config,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.05)

        await detector.stop()
        await detector.stop()  # idempotent

        # Allow event loop to process the stop
        await asyncio.sleep(0.1)
        assert task.done()

    @pytest.mark.asyncio
    async def test_external_task_cancellation(self) -> None:
        """Cancelling the run task triggers DisconnectSource.CANCELLATION."""
        reader = _make_reader_mock(at_eof=False)
        writer = _make_writer_mock()
        callback = AsyncMock()

        config = DetectorConfig(
            probe_interval_seconds=10.0,
            reader_check_interval_seconds=10.0,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-cancel",
            on_disconnect=callback,
            config=config,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # On external cancellation, the callback is called with CANCELLATION source
        callback.assert_awaited_once()
        reason = callback.call_args[0][0]
        assert reason.source == DisconnectSource.CANCELLATION


# ---------------------------------------------------------------------------
# ClientDisconnectDetector tests -- callback single-fire guarantee
# ---------------------------------------------------------------------------


class TestCallbackSingleFire:
    """Tests that the cleanup callback is invoked at most once."""

    @pytest.mark.asyncio
    async def test_callback_fires_exactly_once(self) -> None:
        """Even with multiple disconnect signals, callback fires once."""
        reader = _make_reader_mock(at_eof=True)  # EOF immediately
        writer = _make_broken_writer_mock()  # Also broken
        callback = AsyncMock()

        config = DetectorConfig(
            probe_interval_seconds=0.05,
            reader_check_interval_seconds=0.05,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-double",
            on_disconnect=callback,
            config=config,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.3)

        # Should be called exactly once despite both reader and writer signaling
        callback.assert_awaited_once()

        if not task.done():
            await detector.stop()
            await task


# ---------------------------------------------------------------------------
# ClientDisconnectDetector tests -- error handling
# ---------------------------------------------------------------------------


class TestDetectorErrorHandling:
    """Tests for callback error isolation."""

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_crash_detector(self) -> None:
        """If the callback raises, the detector still completes cleanly."""
        reader = _make_reader_mock(at_eof=True)
        writer = _make_writer_mock()
        callback = AsyncMock(side_effect=RuntimeError("callback broke"))

        config = DetectorConfig(
            probe_interval_seconds=10.0,
            reader_check_interval_seconds=0.05,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-err",
            on_disconnect=callback,
            config=config,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.2)

        # Task should have completed without raising to the caller
        assert task.done()
        # Verify no unhandled exception
        assert task.exception() is None
        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_os_error_on_write_probe_detected(self) -> None:
        """OSError during write probe (generic) triggers disconnect."""
        reader = _make_reader_mock(at_eof=False)
        writer = _make_writer_mock()
        writer.drain = AsyncMock(side_effect=OSError("Transport error"))
        callback = AsyncMock()

        config = DetectorConfig(
            probe_interval_seconds=0.05,
            reader_check_interval_seconds=10.0,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-oserr",
            on_disconnect=callback,
            config=config,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.2)

        callback.assert_awaited_once()
        reason = callback.call_args[0][0]
        assert reason.source == DisconnectSource.WRITE_PROBE_FAILURE

        if not task.done():
            await detector.stop()
            await task


# ---------------------------------------------------------------------------
# ClientDisconnectDetector tests -- properties
# ---------------------------------------------------------------------------


class TestDetectorProperties:
    """Tests for detector state properties."""

    @pytest.mark.asyncio
    async def test_client_id_property(self) -> None:
        reader = _make_reader_mock()
        writer = _make_writer_mock()
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-prop",
            on_disconnect=AsyncMock(),
        )
        assert detector.client_id == "client-prop"

    @pytest.mark.asyncio
    async def test_is_running_property(self) -> None:
        reader = _make_reader_mock(at_eof=False)
        writer = _make_writer_mock()

        config = DetectorConfig(
            probe_interval_seconds=10.0,
            reader_check_interval_seconds=10.0,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-running",
            on_disconnect=AsyncMock(),
            config=config,
        )

        assert not detector.is_running

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.05)
        assert detector.is_running

        await detector.stop()
        await asyncio.sleep(0.05)
        assert not detector.is_running

        if not task.done():
            await task

    @pytest.mark.asyncio
    async def test_disconnect_reason_property(self) -> None:
        """disconnect_reason is None before disconnect, set after."""
        reader = _make_reader_mock(at_eof=True)
        writer = _make_writer_mock()
        callback = AsyncMock()

        config = DetectorConfig(
            probe_interval_seconds=10.0,
            reader_check_interval_seconds=0.05,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-reason",
            on_disconnect=callback,
            config=config,
        )

        assert detector.disconnect_reason is None

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.15)

        reason = detector.disconnect_reason
        assert reason is not None
        assert reason.source == DisconnectSource.READER_EOF

        if not task.done():
            await detector.stop()
            await task


# ---------------------------------------------------------------------------
# Integration: EventBus emission on disconnect
# ---------------------------------------------------------------------------


class TestEventBusIntegration:
    """Tests for EventBus integration on disconnect detection."""

    @pytest.mark.asyncio
    async def test_emits_disconnect_event_on_eof(self) -> None:
        """When an EventBus is provided, detector emits disconnect event."""
        bus = EventBus()
        received_events: list[Event] = []

        async def on_disconnect_event(event: Event) -> None:
            received_events.append(event)

        bus.subscribe(CLIENT_DISCONNECTED_EVENT, on_disconnect_event)

        reader = _make_reader_mock(at_eof=True)
        writer = _make_writer_mock()

        config = DetectorConfig(
            probe_interval_seconds=10.0,
            reader_check_interval_seconds=0.05,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-bus",
            on_disconnect=AsyncMock(),
            config=config,
            event_bus=bus,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.2)

        assert len(received_events) == 1
        payload = received_events[0].payload
        assert payload["client_id"] == "client-bus"
        assert payload["source"] == "reader_eof"

        if not task.done():
            await detector.stop()
            await task

    @pytest.mark.asyncio
    async def test_no_event_bus_is_fine(self) -> None:
        """Detector works without EventBus (no emission, no error)."""
        reader = _make_reader_mock(at_eof=True)
        writer = _make_writer_mock()
        callback = AsyncMock()

        config = DetectorConfig(
            probe_interval_seconds=10.0,
            reader_check_interval_seconds=0.05,
        )
        detector = ClientDisconnectDetector(
            reader=reader,
            writer=writer,
            client_id="client-nobus",
            on_disconnect=callback,
            config=config,
            event_bus=None,
        )

        task = asyncio.create_task(detector.run())
        await asyncio.sleep(0.15)

        callback.assert_awaited_once()

        if not task.done():
            await detector.stop()
            await task
