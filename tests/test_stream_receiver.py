"""Tests for the IPC stream chunk receiver.

Validates that the StreamChunkReceiver correctly:
    - Reads incremental STREAM envelopes from a socket connection.
    - Buffers partial lines that do not end with a newline.
    - Enqueues complete lines/chunks for rendering via an async queue.
    - Handles end-of-stream sentinels.
    - Handles ERROR envelopes from the daemon.
    - Handles connection loss (EOF) gracefully.
    - Flushes any remaining buffered text on stream termination.
    - Produces correct sequence numbers for each chunk.
    - Ignores non-STREAM, non-ERROR envelope types.
"""

from __future__ import annotations

import asyncio

import pytest

from jules_daemon.ipc.framing import (
    MessageEnvelope,
    MessageType,
    encode_frame,
)
from jules_daemon.ipc.stream_receiver import (
    ChunkType,
    LineBuffer,
    StreamChunk,
    StreamChunkReceiver,
    StreamReceiverConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TS = "2026-04-09T12:00:00Z"


# ---------------------------------------------------------------------------
# Helpers: build envelopes
# ---------------------------------------------------------------------------


def _make_stream_envelope(
    line: str,
    sequence: int = 0,
    is_end: bool = False,
    msg_id: str = "stream-001",
) -> MessageEnvelope:
    """Build a STREAM envelope carrying an output chunk."""
    return MessageEnvelope(
        msg_type=MessageType.STREAM,
        msg_id=msg_id,
        timestamp=_TS,
        payload={
            "line": line,
            "sequence": sequence,
            "is_end": is_end,
        },
    )


def _make_error_envelope(
    error: str,
    msg_id: str = "err-001",
) -> MessageEnvelope:
    """Build an ERROR envelope."""
    return MessageEnvelope(
        msg_type=MessageType.ERROR,
        msg_id=msg_id,
        timestamp=_TS,
        payload={"error": error},
    )


def _build_stream_reader(
    envelopes: list[MessageEnvelope],
    *,
    feed_eof: bool = True,
) -> asyncio.StreamReader:
    """Build a StreamReader pre-loaded with framed envelope data."""
    reader = asyncio.StreamReader()
    for env in envelopes:
        reader.feed_data(encode_frame(env))
    if feed_eof:
        reader.feed_eof()
    return reader


async def _drain_queue(
    queue: asyncio.Queue[StreamChunk | None],
    *,
    max_items: int = 100,
) -> list[StreamChunk]:
    """Drain all non-None items from the queue."""
    items: list[StreamChunk] = []
    for _ in range(max_items):
        item = await asyncio.wait_for(queue.get(), timeout=1.0)
        if item is None:
            break
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# ChunkType tests
# ---------------------------------------------------------------------------


class TestChunkType:
    """Tests for the ChunkType enum."""

    def test_all_values(self) -> None:
        assert ChunkType.OUTPUT.value == "output"
        assert ChunkType.END_OF_STREAM.value == "end_of_stream"
        assert ChunkType.ERROR.value == "error"
        assert ChunkType.CONNECTION_LOST.value == "connection_lost"

    def test_count(self) -> None:
        assert len(ChunkType) == 4


# ---------------------------------------------------------------------------
# StreamChunk tests
# ---------------------------------------------------------------------------


class TestStreamChunk:
    """Tests for the immutable StreamChunk dataclass."""

    def test_create_output_chunk(self) -> None:
        chunk = StreamChunk(
            chunk_type=ChunkType.OUTPUT,
            text="PASSED test_login",
            sequence=0,
            timestamp=_TS,
        )
        assert chunk.chunk_type == ChunkType.OUTPUT
        assert chunk.text == "PASSED test_login"
        assert chunk.sequence == 0
        assert chunk.timestamp == _TS

    def test_frozen(self) -> None:
        chunk = StreamChunk(
            chunk_type=ChunkType.OUTPUT,
            text="line",
            sequence=0,
            timestamp=_TS,
        )
        with pytest.raises(AttributeError):
            chunk.text = "mutated"  # type: ignore[misc]

    def test_error_chunk(self) -> None:
        chunk = StreamChunk(
            chunk_type=ChunkType.ERROR,
            text="Connection refused",
            sequence=0,
            timestamp=_TS,
        )
        assert chunk.chunk_type == ChunkType.ERROR

    def test_is_terminal_for_end_of_stream(self) -> None:
        chunk = StreamChunk(
            chunk_type=ChunkType.END_OF_STREAM,
            text="",
            sequence=5,
            timestamp=_TS,
        )
        assert chunk.is_terminal is True

    def test_is_terminal_for_error(self) -> None:
        chunk = StreamChunk(
            chunk_type=ChunkType.ERROR,
            text="fail",
            sequence=0,
            timestamp=_TS,
        )
        assert chunk.is_terminal is True

    def test_is_terminal_for_connection_lost(self) -> None:
        chunk = StreamChunk(
            chunk_type=ChunkType.CONNECTION_LOST,
            text="",
            sequence=0,
            timestamp=_TS,
        )
        assert chunk.is_terminal is True

    def test_is_not_terminal_for_output(self) -> None:
        chunk = StreamChunk(
            chunk_type=ChunkType.OUTPUT,
            text="line",
            sequence=0,
            timestamp=_TS,
        )
        assert chunk.is_terminal is False

    def test_negative_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="sequence must not be negative"):
            StreamChunk(
                chunk_type=ChunkType.OUTPUT,
                text="line",
                sequence=-1,
                timestamp=_TS,
            )


# ---------------------------------------------------------------------------
# LineBuffer tests
# ---------------------------------------------------------------------------


class TestLineBuffer:
    """Tests for the immutable LineBuffer partial-line accumulator."""

    def test_empty_buffer(self) -> None:
        buf = LineBuffer()
        assert buf.pending == ""
        assert buf.is_empty is True

    def test_frozen(self) -> None:
        buf = LineBuffer()
        with pytest.raises(AttributeError):
            buf.pending = "mutated"  # type: ignore[misc]

    def test_feed_complete_single_line(self) -> None:
        buf = LineBuffer()
        new_buf, lines = buf.feed("hello world\n")
        assert lines == ("hello world",)
        assert new_buf.pending == ""
        assert new_buf.is_empty is True

    def test_feed_multiple_complete_lines(self) -> None:
        buf = LineBuffer()
        new_buf, lines = buf.feed("line1\nline2\nline3\n")
        assert lines == ("line1", "line2", "line3")
        assert new_buf.pending == ""

    def test_feed_partial_line_buffered(self) -> None:
        buf = LineBuffer()
        new_buf, lines = buf.feed("partial")
        assert lines == ()
        assert new_buf.pending == "partial"
        assert new_buf.is_empty is False

    def test_feed_completes_buffered_partial(self) -> None:
        buf = LineBuffer(pending="hello ")
        new_buf, lines = buf.feed("world\n")
        assert lines == ("hello world",)
        assert new_buf.pending == ""

    def test_feed_partial_then_more_partial(self) -> None:
        buf = LineBuffer()
        buf2, lines1 = buf.feed("hel")
        assert lines1 == ()
        assert buf2.pending == "hel"

        buf3, lines2 = buf2.feed("lo wo")
        assert lines2 == ()
        assert buf3.pending == "hello wo"

        buf4, lines3 = buf3.feed("rld\n")
        assert lines3 == ("hello world",)
        assert buf4.pending == ""

    def test_feed_mixed_complete_and_partial(self) -> None:
        buf = LineBuffer()
        new_buf, lines = buf.feed("line1\nline2\npartial")
        assert lines == ("line1", "line2")
        assert new_buf.pending == "partial"

    def test_feed_empty_string(self) -> None:
        buf = LineBuffer(pending="existing")
        new_buf, lines = buf.feed("")
        assert lines == ()
        assert new_buf.pending == "existing"

    def test_flush_returns_pending(self) -> None:
        buf = LineBuffer(pending="leftover text")
        new_buf, flushed = buf.flush()
        assert flushed == "leftover text"
        assert new_buf.pending == ""
        assert new_buf.is_empty is True

    def test_flush_empty_returns_empty(self) -> None:
        buf = LineBuffer()
        new_buf, flushed = buf.flush()
        assert flushed == ""
        assert new_buf.is_empty is True

    def test_feed_preserves_empty_lines(self) -> None:
        buf = LineBuffer()
        new_buf, lines = buf.feed("a\n\nb\n")
        assert lines == ("a", "", "b")

    def test_carriage_return_newline(self) -> None:
        """Lines ending in \\r\\n should have the \\r stripped."""
        buf = LineBuffer()
        new_buf, lines = buf.feed("line1\r\nline2\r\n")
        assert lines == ("line1", "line2")
        assert new_buf.pending == ""


# ---------------------------------------------------------------------------
# StreamReceiverConfig tests
# ---------------------------------------------------------------------------


class TestStreamReceiverConfig:
    """Tests for the immutable StreamReceiverConfig."""

    def test_defaults(self) -> None:
        config = StreamReceiverConfig()
        assert config.read_timeout == 10.0
        assert config.max_queue_size == 1024

    def test_frozen(self) -> None:
        config = StreamReceiverConfig()
        with pytest.raises(AttributeError):
            config.read_timeout = 99.0  # type: ignore[misc]

    def test_negative_read_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="read_timeout must be positive"):
            StreamReceiverConfig(read_timeout=-1.0)

    def test_zero_read_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="read_timeout must be positive"):
            StreamReceiverConfig(read_timeout=0.0)

    def test_zero_max_queue_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_queue_size must be positive"):
            StreamReceiverConfig(max_queue_size=0)

    def test_custom_values(self) -> None:
        config = StreamReceiverConfig(read_timeout=30.0, max_queue_size=500)
        assert config.read_timeout == 30.0
        assert config.max_queue_size == 500


# ---------------------------------------------------------------------------
# StreamChunkReceiver -- complete lines
# ---------------------------------------------------------------------------


class TestStreamChunkReceiverCompleteLines:
    """Tests for receiving complete lines from STREAM envelopes."""

    @pytest.mark.asyncio
    async def test_single_complete_line(self) -> None:
        """A STREAM envelope with a newline-terminated line produces one chunk."""
        envelopes = [
            _make_stream_envelope("PASSED test_foo\n", sequence=0),
            _make_stream_envelope("", sequence=1, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        config = StreamReceiverConfig(read_timeout=1.0)
        receiver = StreamChunkReceiver(reader=reader, config=config)

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        assert len(output_chunks) == 1
        assert output_chunks[0].text == "PASSED test_foo"

    @pytest.mark.asyncio
    async def test_multiple_lines_in_one_envelope(self) -> None:
        """A single STREAM envelope with multiple lines produces multiple chunks."""
        envelopes = [
            _make_stream_envelope("line1\nline2\nline3\n", sequence=0),
            _make_stream_envelope("", sequence=1, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        assert len(output_chunks) == 3
        assert output_chunks[0].text == "line1"
        assert output_chunks[1].text == "line2"
        assert output_chunks[2].text == "line3"

    @pytest.mark.asyncio
    async def test_sequences_are_monotonic(self) -> None:
        """Output chunks have monotonically increasing sequence numbers."""
        envelopes = [
            _make_stream_envelope("a\n", sequence=0),
            _make_stream_envelope("b\n", sequence=1),
            _make_stream_envelope("c\n", sequence=2),
            _make_stream_envelope("", sequence=3, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        sequences = [c.sequence for c in output_chunks]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)


# ---------------------------------------------------------------------------
# StreamChunkReceiver -- partial line buffering
# ---------------------------------------------------------------------------


class TestStreamChunkReceiverPartialLines:
    """Tests for partial line buffering across envelopes."""

    @pytest.mark.asyncio
    async def test_partial_line_buffered_until_complete(self) -> None:
        """Partial lines are held until a newline arrives."""
        envelopes = [
            _make_stream_envelope("PASS", sequence=0),
            _make_stream_envelope("ED test_foo\n", sequence=1),
            _make_stream_envelope("", sequence=2, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        assert len(output_chunks) == 1
        assert output_chunks[0].text == "PASSED test_foo"

    @pytest.mark.asyncio
    async def test_multiple_partials_assemble_one_line(self) -> None:
        """Multiple partial chunks assemble into a single complete line."""
        envelopes = [
            _make_stream_envelope("hel", sequence=0),
            _make_stream_envelope("lo ", sequence=1),
            _make_stream_envelope("world\n", sequence=2),
            _make_stream_envelope("", sequence=3, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        assert len(output_chunks) == 1
        assert output_chunks[0].text == "hello world"

    @pytest.mark.asyncio
    async def test_mixed_complete_and_partial(self) -> None:
        """An envelope with complete lines followed by a partial."""
        envelopes = [
            _make_stream_envelope("line1\nline2\npart", sequence=0),
            _make_stream_envelope("ial\n", sequence=1),
            _make_stream_envelope("", sequence=2, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        assert len(output_chunks) == 3
        assert output_chunks[0].text == "line1"
        assert output_chunks[1].text == "line2"
        assert output_chunks[2].text == "partial"


# ---------------------------------------------------------------------------
# StreamChunkReceiver -- flush on termination
# ---------------------------------------------------------------------------


class TestStreamChunkReceiverFlush:
    """Tests for flushing buffered text on stream termination."""

    @pytest.mark.asyncio
    async def test_flush_on_end_of_stream(self) -> None:
        """Buffered partial text is flushed as a chunk before end sentinel."""
        envelopes = [
            _make_stream_envelope("no trailing newline", sequence=0),
            _make_stream_envelope("", sequence=1, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        assert len(output_chunks) == 1
        assert output_chunks[0].text == "no trailing newline"

    @pytest.mark.asyncio
    async def test_flush_on_eof(self) -> None:
        """Buffered partial text is flushed on connection loss (EOF)."""
        envelopes = [
            _make_stream_envelope("partial data", sequence=0),
            # No end sentinel -- just EOF
        ]
        reader = _build_stream_reader(envelopes, feed_eof=True)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        assert len(output_chunks) == 1
        assert output_chunks[0].text == "partial data"

        # Should also have a CONNECTION_LOST terminal chunk
        terminal = [c for c in chunks if c.chunk_type == ChunkType.CONNECTION_LOST]
        assert len(terminal) == 1

    @pytest.mark.asyncio
    async def test_no_flush_when_buffer_empty(self) -> None:
        """When buffer is empty at termination, no spurious OUTPUT chunk."""
        envelopes = [
            _make_stream_envelope("complete line\n", sequence=0),
            _make_stream_envelope("", sequence=1, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        assert len(output_chunks) == 1
        assert output_chunks[0].text == "complete line"


# ---------------------------------------------------------------------------
# StreamChunkReceiver -- end-of-stream
# ---------------------------------------------------------------------------


class TestStreamChunkReceiverEndOfStream:
    """Tests for end-of-stream sentinel handling."""

    @pytest.mark.asyncio
    async def test_end_sentinel_produces_terminal_chunk(self) -> None:
        """The is_end=True envelope produces an END_OF_STREAM chunk."""
        envelopes = [
            _make_stream_envelope("line\n", sequence=0),
            _make_stream_envelope("", sequence=1, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        terminal = [c for c in chunks if c.chunk_type == ChunkType.END_OF_STREAM]
        assert len(terminal) == 1

    @pytest.mark.asyncio
    async def test_end_sentinel_is_last_chunk(self) -> None:
        """The END_OF_STREAM chunk is the last item before the None sentinel."""
        envelopes = [
            _make_stream_envelope("data\n", sequence=0),
            _make_stream_envelope("", sequence=1, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        assert len(chunks) >= 2
        assert chunks[-1].chunk_type == ChunkType.END_OF_STREAM


# ---------------------------------------------------------------------------
# StreamChunkReceiver -- error handling
# ---------------------------------------------------------------------------


class TestStreamChunkReceiverErrors:
    """Tests for error envelope and connection loss handling."""

    @pytest.mark.asyncio
    async def test_error_envelope_produces_error_chunk(self) -> None:
        """An ERROR envelope produces an ERROR chunk with the error message."""
        envelopes = [
            _make_stream_envelope("line\n", sequence=0),
            _make_error_envelope("something broke"),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        error_chunks = [c for c in chunks if c.chunk_type == ChunkType.ERROR]
        assert len(error_chunks) == 1
        assert "something broke" in error_chunks[0].text

    @pytest.mark.asyncio
    async def test_connection_lost_on_eof(self) -> None:
        """EOF without end sentinel produces a CONNECTION_LOST chunk."""
        envelopes = [
            _make_stream_envelope("line\n", sequence=0),
            # No end sentinel, just EOF
        ]
        reader = _build_stream_reader(envelopes, feed_eof=True)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        terminal = [c for c in chunks if c.chunk_type == ChunkType.CONNECTION_LOST]
        assert len(terminal) == 1

    @pytest.mark.asyncio
    async def test_empty_stream_only_connection_lost(self) -> None:
        """Immediate EOF produces only a CONNECTION_LOST chunk."""
        reader = asyncio.StreamReader()
        reader.feed_eof()
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.CONNECTION_LOST


# ---------------------------------------------------------------------------
# StreamChunkReceiver -- non-STREAM envelopes
# ---------------------------------------------------------------------------


class TestStreamChunkReceiverIgnoreNonStream:
    """Tests for skipping non-STREAM, non-ERROR envelopes."""

    @pytest.mark.asyncio
    async def test_response_envelopes_skipped(self) -> None:
        """RESPONSE envelopes during streaming are silently ignored."""
        envelopes = [
            _make_stream_envelope("line1\n", sequence=0),
            MessageEnvelope(
                msg_type=MessageType.RESPONSE,
                msg_id="stray-resp",
                timestamp=_TS,
                payload={"verb": "unwatch", "status": "ok"},
            ),
            _make_stream_envelope("line2\n", sequence=1),
            _make_stream_envelope("", sequence=2, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        assert len(output_chunks) == 2
        assert output_chunks[0].text == "line1"
        assert output_chunks[1].text == "line2"


# ---------------------------------------------------------------------------
# StreamChunkReceiver -- cancellation
# ---------------------------------------------------------------------------


class TestStreamChunkReceiverCancellation:
    """Tests for cancellation (Ctrl+C / task.cancel()) handling."""

    @pytest.mark.asyncio
    async def test_cancel_flushes_buffer_and_terminates(self) -> None:
        """Cancellation flushes any buffered text, enqueues it, then stops."""
        reader = asyncio.StreamReader()
        # Feed a partial line, then leave the reader open (no EOF)
        reader.feed_data(encode_frame(_make_stream_envelope("partial", sequence=0)))

        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=60.0),
        )

        task = asyncio.create_task(receiver.run())
        # Let the receiver consume the partial
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # The queue should still have the flushed partial
        chunks = []
        while not receiver.queue.empty():
            item = receiver.queue.get_nowait()
            if item is not None:
                chunks.append(item)

        # At minimum, the partial text should have been flushed
        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        assert len(output_chunks) == 1
        assert output_chunks[0].text == "partial"


# ---------------------------------------------------------------------------
# StreamChunkReceiver -- queue property
# ---------------------------------------------------------------------------


class TestStreamChunkReceiverQueue:
    """Tests for the receiver's queue interface."""

    @pytest.mark.asyncio
    async def test_queue_is_accessible(self) -> None:
        """The queue property returns an asyncio.Queue."""
        reader = asyncio.StreamReader()
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(),
        )
        assert isinstance(receiver.queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_queue_respects_max_size(self) -> None:
        """The queue's maxsize matches the config's max_queue_size."""
        reader = asyncio.StreamReader()
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(max_queue_size=42),
        )
        assert receiver.queue.maxsize == 42


# ---------------------------------------------------------------------------
# StreamChunkReceiver -- timestamps
# ---------------------------------------------------------------------------


class TestStreamChunkReceiverTimestamps:
    """Tests for timestamp propagation from envelopes to chunks."""

    @pytest.mark.asyncio
    async def test_chunks_carry_envelope_timestamp(self) -> None:
        """Output chunks carry the timestamp from the source envelope."""
        envelopes = [
            MessageEnvelope(
                msg_type=MessageType.STREAM,
                msg_id="s1",
                timestamp="2026-04-09T14:30:00Z",
                payload={"line": "time test\n", "sequence": 0, "is_end": False},
            ),
            _make_stream_envelope("", sequence=1, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        assert len(output_chunks) == 1
        assert output_chunks[0].timestamp == "2026-04-09T14:30:00Z"


# ---------------------------------------------------------------------------
# StreamChunkReceiver -- carriage return handling
# ---------------------------------------------------------------------------


class TestStreamChunkReceiverCRLF:
    """Tests for \\r\\n line ending normalization."""

    @pytest.mark.asyncio
    async def test_crlf_stripped(self) -> None:
        """Lines with \\r\\n are delivered without the \\r."""
        envelopes = [
            _make_stream_envelope("windows line\r\n", sequence=0),
            _make_stream_envelope("", sequence=1, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        receiver = StreamChunkReceiver(
            reader=reader,
            config=StreamReceiverConfig(read_timeout=1.0),
        )

        await receiver.run()
        chunks = await _drain_queue(receiver.queue)

        output_chunks = [c for c in chunks if c.chunk_type == ChunkType.OUTPUT]
        assert len(output_chunks) == 1
        assert output_chunks[0].text == "windows line"
