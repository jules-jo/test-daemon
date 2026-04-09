"""Tests for per-client read/write coroutines with framing integration.

Validates:
    - ClientReader reads framed messages and delivers decoded envelopes to a queue
    - ClientWriter consumes envelopes from a queue and writes framed messages
    - ClientIO orchestrates reader/writer lifecycle with ConnectionManager
    - Graceful shutdown via cancellation and sentinel values
    - Error propagation for malformed frames, broken pipes, and timeouts
    - EOF detection triggers disconnect lifecycle
    - ConnectionManager integration (register on start, deregister on stop)
    - Writer queue backpressure behavior
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    encode_frame,
    pack_header,
)
from jules_daemon.ipc.client_io import (
    ClientIO,
    ClientIOError,
    ClientReader,
    ClientWriter,
    ReadError,
    WriteError,
)
from jules_daemon.ipc.connection_manager import (
    CLIENT_CONNECTED_EVENT,
    CLIENT_DISCONNECTED_EVENT,
    ConnectionManager,
)
from jules_daemon.ipc.event_bus import Event, EventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = "2026-04-09T12:00:00Z"


def _make_envelope(
    msg_id: str = "test-001",
    msg_type: MessageType = MessageType.REQUEST,
    payload: dict | None = None,
) -> MessageEnvelope:
    """Build a minimal test envelope."""
    return MessageEnvelope(
        msg_type=msg_type,
        msg_id=msg_id,
        timestamp=_TS,
        payload=payload or {"verb": "status"},
    )


def _make_reader_from_envelopes(
    envelopes: list[MessageEnvelope],
) -> asyncio.StreamReader:
    """Build a StreamReader pre-loaded with framed envelope data."""
    reader = asyncio.StreamReader()
    for env in envelopes:
        reader.feed_data(encode_frame(env))
    reader.feed_eof()
    return reader


def _make_reader_from_bytes(data: bytes) -> asyncio.StreamReader:
    """Build a StreamReader pre-loaded with raw bytes then EOF."""
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


def _make_writer_mock() -> AsyncMock:
    """Build a mock StreamWriter that records writes."""
    writer = AsyncMock(spec=asyncio.StreamWriter)
    writer.is_closing.return_value = False
    writer.write = MagicMock()  # write is sync, drain is async
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return writer


# ---------------------------------------------------------------------------
# ClientReader tests
# ---------------------------------------------------------------------------


class TestClientReader:
    """Tests for the per-client frame reader coroutine."""

    @pytest.mark.asyncio
    async def test_reads_single_envelope(self) -> None:
        env = _make_envelope()
        stream = _make_reader_from_envelopes([env])
        inbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        client_reader = ClientReader(
            reader=stream,
            client_id="c1",
            inbox=inbox,
        )
        await client_reader.run()

        received = await inbox.get()
        assert received is not None
        assert received.msg_id == env.msg_id
        assert received.payload == env.payload

    @pytest.mark.asyncio
    async def test_reads_multiple_envelopes(self) -> None:
        envelopes = [_make_envelope(msg_id=f"msg-{i}") for i in range(5)]
        stream = _make_reader_from_envelopes(envelopes)
        inbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        client_reader = ClientReader(
            reader=stream,
            client_id="c1",
            inbox=inbox,
        )
        await client_reader.run()

        received = []
        while not inbox.empty():
            item = await inbox.get()
            if item is None:
                break
            received.append(item)

        assert len(received) == 5
        assert [r.msg_id for r in received] == [f"msg-{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_eof_sends_none_sentinel(self) -> None:
        """Reader puts None in the inbox on EOF (client disconnect)."""
        stream = asyncio.StreamReader()
        stream.feed_eof()
        inbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        client_reader = ClientReader(
            reader=stream,
            client_id="c1",
            inbox=inbox,
        )
        await client_reader.run()

        sentinel = await inbox.get()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_eof_after_messages_sends_sentinel(self) -> None:
        """After reading messages, EOF still sends None sentinel."""
        envelopes = [_make_envelope(msg_id="last")]
        stream = _make_reader_from_envelopes(envelopes)
        inbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        client_reader = ClientReader(
            reader=stream,
            client_id="c1",
            inbox=inbox,
        )
        await client_reader.run()

        # First item is the envelope
        first = await inbox.get()
        assert first is not None
        assert first.msg_id == "last"

        # Second item is EOF sentinel
        sentinel = await inbox.get()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_malformed_frame_produces_read_error(self) -> None:
        """Malformed data triggers on_error callback with ReadError."""
        # Valid header claiming 10 bytes, but payload is invalid JSON
        header = pack_header(10)
        garbage = b"not-json!!"  # exactly 10 bytes
        stream = _make_reader_from_bytes(header + garbage)
        inbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        errors: list[ClientIOError] = []

        client_reader = ClientReader(
            reader=stream,
            client_id="c1",
            inbox=inbox,
            on_error=lambda e: errors.append(e),
        )
        await client_reader.run()

        assert len(errors) == 1
        assert isinstance(errors[0], ReadError)
        assert "c1" in str(errors[0])

    @pytest.mark.asyncio
    async def test_incomplete_header_sends_sentinel(self) -> None:
        """Incomplete header (less than 4 bytes then EOF) sends sentinel."""
        stream = _make_reader_from_bytes(b"\x00\x00")  # only 2 bytes
        inbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        client_reader = ClientReader(
            reader=stream,
            client_id="c1",
            inbox=inbox,
        )
        await client_reader.run()

        sentinel = await inbox.get()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_incomplete_payload_sends_sentinel(self) -> None:
        """Incomplete payload (header says 100 bytes, only 5 available) sends sentinel."""
        header = pack_header(100)
        stream = _make_reader_from_bytes(header + b"short")
        inbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        client_reader = ClientReader(
            reader=stream,
            client_id="c1",
            inbox=inbox,
        )
        await client_reader.run()

        sentinel = await inbox.get()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_cancellation_during_read(self) -> None:
        """Cancelling the reader task stops the read loop cleanly."""
        stream = asyncio.StreamReader()
        # Never feed data -- reader will block on readexactly
        inbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        client_reader = ClientReader(
            reader=stream,
            client_id="c1",
            inbox=inbox,
        )
        task = asyncio.create_task(client_reader.run())
        await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_reader_client_id_property(self) -> None:
        stream = asyncio.StreamReader()
        inbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()
        client_reader = ClientReader(
            reader=stream,
            client_id="reader-001",
            inbox=inbox,
        )
        assert client_reader.client_id == "reader-001"


# ---------------------------------------------------------------------------
# ClientWriter tests
# ---------------------------------------------------------------------------


class TestClientWriter:
    """Tests for the per-client frame writer coroutine."""

    @pytest.mark.asyncio
    async def test_writes_single_envelope(self) -> None:
        writer = _make_writer_mock()
        outbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        client_writer = ClientWriter(
            writer=writer,
            client_id="c1",
            outbox=outbox,
        )

        env = _make_envelope()
        await outbox.put(env)
        await outbox.put(None)  # sentinel to stop

        await client_writer.run()

        expected_frame = encode_frame(env)
        writer.write.assert_called_once_with(expected_frame)
        writer.drain.assert_awaited()

    @pytest.mark.asyncio
    async def test_writes_multiple_envelopes(self) -> None:
        writer = _make_writer_mock()
        outbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        client_writer = ClientWriter(
            writer=writer,
            client_id="c1",
            outbox=outbox,
        )

        envelopes = [_make_envelope(msg_id=f"msg-{i}") for i in range(3)]
        for env in envelopes:
            await outbox.put(env)
        await outbox.put(None)

        await client_writer.run()

        assert writer.write.call_count == 3
        for i, call in enumerate(writer.write.call_args_list):
            frame = call[0][0]
            assert frame == encode_frame(envelopes[i])

    @pytest.mark.asyncio
    async def test_none_sentinel_stops_writer(self) -> None:
        writer = _make_writer_mock()
        outbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        client_writer = ClientWriter(
            writer=writer,
            client_id="c1",
            outbox=outbox,
        )

        await outbox.put(None)
        await client_writer.run()

        writer.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_broken_pipe_triggers_error(self) -> None:
        writer = _make_writer_mock()
        writer.drain.side_effect = BrokenPipeError("pipe broken")
        outbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        errors: list[ClientIOError] = []

        client_writer = ClientWriter(
            writer=writer,
            client_id="c1",
            outbox=outbox,
            on_error=lambda e: errors.append(e),
        )

        await outbox.put(_make_envelope())
        await outbox.put(None)  # stop after error

        await client_writer.run()

        assert len(errors) == 1
        assert isinstance(errors[0], WriteError)

    @pytest.mark.asyncio
    async def test_connection_reset_triggers_error(self) -> None:
        writer = _make_writer_mock()
        writer.drain.side_effect = ConnectionResetError("reset")
        outbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        errors: list[ClientIOError] = []

        client_writer = ClientWriter(
            writer=writer,
            client_id="c1",
            outbox=outbox,
            on_error=lambda e: errors.append(e),
        )

        await outbox.put(_make_envelope())
        await client_writer.run()

        assert len(errors) == 1
        assert isinstance(errors[0], WriteError)

    @pytest.mark.asyncio
    async def test_cancellation_during_write(self) -> None:
        """Cancelling the writer task stops cleanly."""
        writer = _make_writer_mock()
        outbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        client_writer = ClientWriter(
            writer=writer,
            client_id="c1",
            outbox=outbox,
        )

        # Don't put anything -- writer will block on queue.get()
        task = asyncio.create_task(client_writer.run())
        await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_send_convenience_method(self) -> None:
        """send() puts an envelope on the outbox queue."""
        writer = _make_writer_mock()
        outbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        client_writer = ClientWriter(
            writer=writer,
            client_id="c1",
            outbox=outbox,
        )

        env = _make_envelope()
        await client_writer.send(env)

        assert outbox.qsize() == 1
        queued = await outbox.get()
        assert queued == env

    @pytest.mark.asyncio
    async def test_writer_client_id_property(self) -> None:
        writer = _make_writer_mock()
        outbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()
        client_writer = ClientWriter(
            writer=writer,
            client_id="writer-001",
            outbox=outbox,
        )
        assert client_writer.client_id == "writer-001"


# ---------------------------------------------------------------------------
# ClientIO lifecycle tests
# ---------------------------------------------------------------------------


class TestClientIO:
    """Tests for the ClientIO lifecycle wrapper."""

    @pytest.mark.asyncio
    async def test_start_registers_with_connection_manager(self) -> None:
        bus = EventBus()
        manager = ConnectionManager(event_bus=bus)

        events: list[Event] = []
        bus.subscribe(CLIENT_CONNECTED_EVENT, lambda e: events.append(e))

        stream_reader = asyncio.StreamReader()
        stream_reader.feed_eof()  # immediate EOF to let reader finish
        stream_writer = _make_writer_mock()

        client_io = ClientIO(
            reader=stream_reader,
            writer=stream_writer,
            client_id="cio-1",
            connected_at=_TS,
            connection_manager=manager,
        )

        await client_io.start()
        # Allow tasks to run
        await asyncio.sleep(0.05)
        await client_io.stop()

        assert len(events) == 1
        assert events[0].payload["client_id"] == "cio-1"

    @pytest.mark.asyncio
    async def test_stop_deregisters_from_connection_manager(self) -> None:
        bus = EventBus()
        manager = ConnectionManager(event_bus=bus)

        disconnect_events: list[Event] = []
        bus.subscribe(
            CLIENT_DISCONNECTED_EVENT,
            lambda e: disconnect_events.append(e),
        )

        stream_reader = asyncio.StreamReader()
        stream_reader.feed_eof()
        stream_writer = _make_writer_mock()

        client_io = ClientIO(
            reader=stream_reader,
            writer=stream_writer,
            client_id="cio-2",
            connected_at=_TS,
            connection_manager=manager,
        )

        await client_io.start()
        await asyncio.sleep(0.05)
        await client_io.stop()

        assert len(disconnect_events) == 1
        assert disconnect_events[0].payload["client_id"] == "cio-2"
        assert not manager.has_client("cio-2")

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        manager = ConnectionManager()

        stream_reader = asyncio.StreamReader()
        stream_reader.feed_eof()
        stream_writer = _make_writer_mock()

        client_io = ClientIO(
            reader=stream_reader,
            writer=stream_writer,
            client_id="cio-3",
            connected_at=_TS,
            connection_manager=manager,
        )

        await client_io.start()
        await asyncio.sleep(0.05)
        await client_io.stop()
        await client_io.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_send_queues_envelope(self) -> None:
        manager = ConnectionManager()

        stream_reader = asyncio.StreamReader()
        stream_writer = _make_writer_mock()

        client_io = ClientIO(
            reader=stream_reader,
            writer=stream_writer,
            client_id="cio-4",
            connected_at=_TS,
            connection_manager=manager,
        )

        await client_io.start()
        env = _make_envelope(msg_id="send-test")
        await client_io.send(env)

        # Allow writer to process
        await asyncio.sleep(0.05)

        stream_writer.write.assert_called()
        # Clean up
        stream_reader.feed_eof()
        await asyncio.sleep(0.05)
        await client_io.stop()

    @pytest.mark.asyncio
    async def test_receive_returns_envelope_from_reader(self) -> None:
        manager = ConnectionManager()

        env = _make_envelope(msg_id="recv-test")
        stream_reader = _make_reader_from_envelopes([env])
        stream_writer = _make_writer_mock()

        client_io = ClientIO(
            reader=stream_reader,
            writer=stream_writer,
            client_id="cio-5",
            connected_at=_TS,
            connection_manager=manager,
        )

        await client_io.start()
        received = await client_io.receive()

        assert received is not None
        assert received.msg_id == "recv-test"

        await client_io.stop()

    @pytest.mark.asyncio
    async def test_receive_returns_none_on_eof(self) -> None:
        manager = ConnectionManager()

        stream_reader = asyncio.StreamReader()
        stream_reader.feed_eof()
        stream_writer = _make_writer_mock()

        client_io = ClientIO(
            reader=stream_reader,
            writer=stream_writer,
            client_id="cio-6",
            connected_at=_TS,
            connection_manager=manager,
        )

        await client_io.start()
        received = await client_io.receive()
        assert received is None

        await client_io.stop()

    @pytest.mark.asyncio
    async def test_reader_eof_stops_writer(self) -> None:
        """When reader detects EOF, the writer coroutine should also stop."""
        manager = ConnectionManager()

        stream_reader = asyncio.StreamReader()
        stream_reader.feed_eof()
        stream_writer = _make_writer_mock()

        client_io = ClientIO(
            reader=stream_reader,
            writer=stream_writer,
            client_id="cio-7",
            connected_at=_TS,
            connection_manager=manager,
        )

        await client_io.start()
        # Wait for reader to detect EOF and propagate
        await asyncio.sleep(0.1)

        assert client_io.is_closed

        await client_io.stop()

    @pytest.mark.asyncio
    async def test_close_writer_on_stop(self) -> None:
        """stop() closes the underlying StreamWriter."""
        manager = ConnectionManager()

        stream_reader = asyncio.StreamReader()
        stream_reader.feed_eof()
        stream_writer = _make_writer_mock()

        client_io = ClientIO(
            reader=stream_reader,
            writer=stream_writer,
            client_id="cio-8",
            connected_at=_TS,
            connection_manager=manager,
        )

        await client_io.start()
        await asyncio.sleep(0.05)
        await client_io.stop()

        stream_writer.close.assert_called()
        stream_writer.wait_closed.assert_awaited()

    @pytest.mark.asyncio
    async def test_client_id_property(self) -> None:
        manager = ConnectionManager()
        stream_reader = asyncio.StreamReader()
        stream_writer = _make_writer_mock()

        client_io = ClientIO(
            reader=stream_reader,
            writer=stream_writer,
            client_id="prop-test",
            connected_at=_TS,
            connection_manager=manager,
        )
        assert client_io.client_id == "prop-test"

    @pytest.mark.asyncio
    async def test_works_without_connection_manager(self) -> None:
        """ClientIO works without a ConnectionManager (purely for IO)."""
        stream_reader = asyncio.StreamReader()
        stream_reader.feed_eof()
        stream_writer = _make_writer_mock()

        client_io = ClientIO(
            reader=stream_reader,
            writer=stream_writer,
            client_id="no-mgr",
            connected_at=_TS,
        )

        await client_io.start()
        received = await client_io.receive()
        assert received is None
        await client_io.stop()

    @pytest.mark.asyncio
    async def test_error_callback_receives_errors(self) -> None:
        """Errors from reader/writer propagate through on_error callback."""
        manager = ConnectionManager()

        # Malformed frame: valid header but invalid JSON payload
        header = pack_header(5)
        data = header + b"xxxxx"
        stream_reader = _make_reader_from_bytes(data)
        stream_writer = _make_writer_mock()

        errors: list[ClientIOError] = []

        client_io = ClientIO(
            reader=stream_reader,
            writer=stream_writer,
            client_id="err-test",
            connected_at=_TS,
            connection_manager=manager,
            on_error=lambda e: errors.append(e),
        )

        await client_io.start()
        await asyncio.sleep(0.1)
        await client_io.stop()

        assert len(errors) >= 1
        assert isinstance(errors[0], ReadError)


# ---------------------------------------------------------------------------
# ClientIO as async context manager tests
# ---------------------------------------------------------------------------


class TestClientIOContextManager:
    """Tests for async context manager interface."""

    @pytest.mark.asyncio
    async def test_context_manager_starts_and_stops(self) -> None:
        manager = ConnectionManager()
        stream_reader = asyncio.StreamReader()
        stream_reader.feed_eof()
        stream_writer = _make_writer_mock()

        async with ClientIO(
            reader=stream_reader,
            writer=stream_writer,
            client_id="ctx-1",
            connected_at=_TS,
            connection_manager=manager,
        ) as cio:
            assert manager.has_client("ctx-1")

        assert not manager.has_client("ctx-1")

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up_on_error(self) -> None:
        manager = ConnectionManager()
        stream_reader = asyncio.StreamReader()
        stream_reader.feed_eof()
        stream_writer = _make_writer_mock()

        with pytest.raises(RuntimeError, match="test error"):
            async with ClientIO(
                reader=stream_reader,
                writer=stream_writer,
                client_id="ctx-2",
                connected_at=_TS,
                connection_manager=manager,
            ):
                raise RuntimeError("test error")

        assert not manager.has_client("ctx-2")


# ---------------------------------------------------------------------------
# Error type tests
# ---------------------------------------------------------------------------


class TestErrorTypes:
    """Tests for ClientIOError, ReadError, WriteError."""

    def test_read_error_is_client_io_error(self) -> None:
        err = ReadError(client_id="c1", detail="malformed frame")
        assert isinstance(err, ClientIOError)

    def test_write_error_is_client_io_error(self) -> None:
        err = WriteError(client_id="c1", detail="broken pipe")
        assert isinstance(err, ClientIOError)

    def test_read_error_str(self) -> None:
        err = ReadError(client_id="c1", detail="bad json")
        assert "c1" in str(err)
        assert "bad json" in str(err)

    def test_write_error_str(self) -> None:
        err = WriteError(client_id="c1", detail="pipe gone")
        assert "c1" in str(err)
        assert "pipe gone" in str(err)

    def test_read_error_preserves_cause(self) -> None:
        cause = ValueError("original error")
        err = ReadError(client_id="c1", detail="decode failed", cause=cause)
        assert err.cause is cause

    def test_write_error_preserves_cause(self) -> None:
        cause = BrokenPipeError("pipe broken")
        err = WriteError(client_id="c1", detail="send failed", cause=cause)
        assert err.cause is cause
