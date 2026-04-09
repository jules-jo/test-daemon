"""IPC stream chunk receiver for incremental output rendering.

Reads STREAM envelopes from a socket connection, buffers partial lines
that do not end with a newline delimiter, and enqueues complete chunks
into an async queue for the rendering layer to consume.

SSH output from the daemon arrives as arbitrary byte chunks that may
split in the middle of a line. The STREAM envelopes carry these raw
text chunks. This receiver:

    1. Decodes framed STREAM envelopes from the asyncio.StreamReader.
    2. Feeds each text chunk into a LineBuffer that accumulates partial
       lines and yields complete ones.
    3. Wraps each complete line in an immutable StreamChunk and puts it
       on the output queue.
    4. On stream termination (end sentinel, error, EOF), flushes any
       remaining buffered text as a final OUTPUT chunk, then enqueues
       the appropriate terminal chunk (END_OF_STREAM, ERROR, or
       CONNECTION_LOST).
    5. Places a None sentinel on the queue to signal the consumer that
       no more chunks will arrive.

Architecture::

    asyncio.StreamReader
        |
        v
    [framing decode] --> MessageEnvelope (STREAM)
        |
        v
    LineBuffer.feed(text) --> complete lines
        |
        v
    asyncio.Queue[StreamChunk | None]
        |
        v
    Rendering layer (consumer)

The receiver is designed for single-use: create a new instance for each
streaming session.

Usage::

    from jules_daemon.ipc.stream_receiver import (
        StreamChunkReceiver,
        StreamReceiverConfig,
    )

    receiver = StreamChunkReceiver(
        reader=stream_reader,
        config=StreamReceiverConfig(read_timeout=10.0),
    )

    # Start the receiver (runs until stream ends)
    task = asyncio.create_task(receiver.run())

    # Consume chunks from the queue
    while True:
        chunk = await receiver.queue.get()
        if chunk is None:
            break
        render(chunk)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    unpack_header,
)

__all__ = [
    "ChunkType",
    "LineBuffer",
    "StreamChunk",
    "StreamChunkReceiver",
    "StreamReceiverConfig",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChunkType enum
# ---------------------------------------------------------------------------


class ChunkType(Enum):
    """Type of stream chunk delivered to the rendering queue.

    Values:
        OUTPUT:          A complete line of SSH output ready for display.
        END_OF_STREAM:   The remote job completed normally. Terminal.
        ERROR:           The daemon reported an error. Terminal.
        CONNECTION_LOST: The socket connection was lost. Terminal.
    """

    OUTPUT = "output"
    END_OF_STREAM = "end_of_stream"
    ERROR = "error"
    CONNECTION_LOST = "connection_lost"


_TERMINAL_TYPES = frozenset(
    {
        ChunkType.END_OF_STREAM,
        ChunkType.ERROR,
        ChunkType.CONNECTION_LOST,
    }
)


# ---------------------------------------------------------------------------
# StreamChunk dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamChunk:
    """Immutable chunk of stream output ready for rendering.

    Each chunk represents either a complete line of SSH output, or a
    terminal event (end-of-stream, error, connection lost).

    Attributes:
        chunk_type: Category of this chunk (output, terminal, etc.).
        text:       The text content. For OUTPUT chunks this is a single
                    complete line (without trailing newline). For ERROR
                    chunks this is the error message. For terminal chunks
                    this may be empty.
        sequence:   Monotonically increasing counter assigned by the
                    receiver. Enables ordering without relying solely
                    on timestamp precision.
        timestamp:  ISO 8601 timestamp from the source STREAM envelope.
    """

    chunk_type: ChunkType
    text: str
    sequence: int
    timestamp: str

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError(f"sequence must not be negative, got {self.sequence}")

    @property
    def is_terminal(self) -> bool:
        """True if this chunk signals the end of the stream."""
        return self.chunk_type in _TERMINAL_TYPES


# ---------------------------------------------------------------------------
# LineBuffer dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineBuffer:
    """Immutable partial-line accumulator for SSH output buffering.

    Accumulates text that does not end with a newline delimiter. When
    ``feed()`` is called with text containing newlines, the buffer
    yields all complete lines and retains any trailing partial text.

    Immutable: every operation returns a new LineBuffer instance.

    Attributes:
        pending: Text accumulated so far that has not been terminated
                 by a newline. Empty string when no partial data exists.
    """

    pending: str = ""

    @property
    def is_empty(self) -> bool:
        """True if no partial text is buffered."""
        return self.pending == ""

    def feed(self, text: str) -> tuple[LineBuffer, tuple[str, ...]]:
        """Feed new text into the buffer and extract complete lines.

        Splits the combined (pending + text) on newline boundaries.
        Complete lines (terminated by ``\\n``) are returned as a tuple.
        Any trailing text without a newline is retained in the new buffer.

        Lines ending in ``\\r\\n`` have the ``\\r`` stripped.

        Args:
            text: New text chunk from a STREAM envelope.

        Returns:
            Tuple of (new_buffer, complete_lines).
            ``new_buffer`` holds any remaining partial text.
            ``complete_lines`` is a tuple of strings without trailing
            newlines, in order of appearance.
        """
        if not text:
            return (LineBuffer(pending=self.pending), ())

        combined = self.pending + text

        # Split on newline. If the text ends with \n, the last element
        # of split will be an empty string (the part after the final \n).
        parts = combined.split("\n")

        # The last element is the new pending (may be empty if text
        # ended with a newline).
        new_pending = parts[-1]
        raw_lines = parts[:-1]

        # Strip trailing \r from each line (handles \r\n line endings)
        complete_lines = tuple(line.rstrip("\r") for line in raw_lines)

        return (LineBuffer(pending=new_pending), complete_lines)

    def flush(self) -> tuple[LineBuffer, str]:
        """Flush any remaining buffered text.

        Returns the pending text and resets the buffer. Used at stream
        termination to emit any incomplete final line.

        Returns:
            Tuple of (empty_buffer, flushed_text).
            ``flushed_text`` is the pending text (may be empty).
        """
        return (LineBuffer(pending=""), self.pending)


# ---------------------------------------------------------------------------
# StreamReceiverConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamReceiverConfig:
    """Immutable configuration for the stream chunk receiver.

    Attributes:
        read_timeout:   Maximum seconds to wait for the next envelope
                        from the daemon before treating the connection
                        as lost.
        max_queue_size: Maximum number of chunks that can be buffered
                        in the output queue before the receiver blocks.
    """

    read_timeout: float = 10.0
    max_queue_size: int = 1024

    def __post_init__(self) -> None:
        if self.read_timeout <= 0:
            raise ValueError(f"read_timeout must be positive, got {self.read_timeout}")
        if self.max_queue_size <= 0:
            raise ValueError(
                f"max_queue_size must be positive, got {self.max_queue_size}"
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# StreamChunkReceiver
# ---------------------------------------------------------------------------


class StreamChunkReceiver:
    """Async receiver that reads STREAM envelopes and enqueues rendered chunks.

    Reads length-prefixed framed envelopes from an ``asyncio.StreamReader``,
    buffers partial lines using ``LineBuffer``, and enqueues complete
    ``StreamChunk`` objects into an async queue for the rendering layer.

    On stream termination (end sentinel, error, EOF, or cancellation),
    any remaining buffered text is flushed as a final OUTPUT chunk, the
    appropriate terminal chunk is enqueued, and a ``None`` sentinel is
    placed on the queue to signal the consumer.

    The receiver is designed for single-use: create a new instance for
    each streaming session.

    Note on mutability: Unlike the frozen dataclass value objects in this
    module (StreamChunk, LineBuffer, etc.), this class intentionally holds
    mutable internal state (_buffer, _sequence, _last_timestamp) because
    it is an active coroutine, not a value object. The immutable LineBuffer
    is replaced (not mutated) on each feed/flush call.

    Args:
        reader: The asyncio.StreamReader to read framed envelopes from.
        config: Receiver configuration (timeouts, queue size).
    """

    def __init__(
        self,
        *,
        reader: asyncio.StreamReader,
        config: StreamReceiverConfig | None = None,
    ) -> None:
        effective_config = config or StreamReceiverConfig()
        self._reader = reader
        self._config = effective_config
        self._queue: asyncio.Queue[StreamChunk | None] = asyncio.Queue(
            maxsize=effective_config.max_queue_size
        )
        # Mutable per-session state (see class docstring note on mutability)
        self._buffer = LineBuffer()
        self._sequence = 0
        self._last_timestamp = _now_iso()

    @property
    def queue(self) -> asyncio.Queue[StreamChunk | None]:
        """The output queue where complete chunks are placed.

        Consumers should read from this queue. A ``None`` value
        signals that no more chunks will arrive.
        """
        return self._queue

    async def run(self) -> None:
        """Read envelopes and enqueue chunks until the stream terminates.

        This is the main entry point. It enters a read loop that:
        1. Reads one framed envelope from the stream.
        2. Routes the envelope by type (STREAM, ERROR, other).
        3. For STREAM envelopes, feeds the text into the LineBuffer
           and enqueues complete lines.
        4. On termination, flushes the buffer and enqueues a terminal
           chunk.
        5. Places a None sentinel on the queue.

        The method handles ``asyncio.CancelledError`` by flushing
        any buffered text before re-raising.

        Raises:
            asyncio.CancelledError: If the coroutine is cancelled
                externally. Buffered text is flushed before re-raise.
        """
        try:
            await self._receive_loop()
        except asyncio.CancelledError:
            # Flush any remaining buffered text before propagating
            self._flush_pending()
            raise
        finally:
            # Signal the consumer that the stream is done
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                logger.warning(
                    "Queue full when placing None sentinel -- "
                    "consumer may not detect stream end"
                )

    async def _receive_loop(self) -> None:
        """Internal receive loop that processes envelopes."""
        while True:
            envelope = await self._read_envelope()

            if envelope is None:
                # EOF, timeout, or connection lost
                self._flush_pending()
                self._enqueue_chunk(
                    chunk_type=ChunkType.CONNECTION_LOST,
                    text="",
                    timestamp=_now_iso(),
                )
                return

            if envelope.msg_type == MessageType.ERROR:
                error_msg = envelope.payload.get("error", "Unknown error")
                self._flush_pending()
                self._enqueue_chunk(
                    chunk_type=ChunkType.ERROR,
                    text=error_msg,
                    timestamp=envelope.timestamp,
                )
                return

            if envelope.msg_type == MessageType.STREAM:
                payload = envelope.payload
                is_end = payload.get("is_end", False)

                if is_end:
                    # End of stream -- flush buffer, enqueue terminal
                    self._flush_pending()
                    self._enqueue_chunk(
                        chunk_type=ChunkType.END_OF_STREAM,
                        text="",
                        timestamp=envelope.timestamp,
                    )
                    return

                # Normal output chunk -- feed into line buffer
                text = payload.get("line", "")
                self._last_timestamp = envelope.timestamp
                self._process_text(text, timestamp=envelope.timestamp)

            # else: skip non-STREAM, non-ERROR envelopes silently

    def _process_text(self, text: str, *, timestamp: str) -> None:
        """Feed text into the line buffer and enqueue complete lines.

        Args:
            text: Raw text from a STREAM envelope.
            timestamp: The envelope's ISO 8601 timestamp.
        """
        new_buffer, lines = self._buffer.feed(text)
        self._buffer = new_buffer

        for line in lines:
            self._enqueue_chunk(
                chunk_type=ChunkType.OUTPUT,
                text=line,
                timestamp=timestamp,
            )

    def _flush_pending(self) -> None:
        """Flush any remaining text in the line buffer as an OUTPUT chunk.

        Called before enqueuing a terminal chunk to ensure no data is lost.
        """
        new_buffer, flushed = self._buffer.flush()
        self._buffer = new_buffer

        if flushed:
            self._enqueue_chunk(
                chunk_type=ChunkType.OUTPUT,
                text=flushed,
                timestamp=self._last_timestamp,
            )

    def _enqueue_chunk(
        self,
        *,
        chunk_type: ChunkType,
        text: str,
        timestamp: str,
    ) -> None:
        """Create a StreamChunk and place it on the output queue.

        Uses put_nowait to avoid blocking the receive loop. If the
        queue is full, the chunk is dropped with a warning log.

        Args:
            chunk_type: The type of chunk to create.
            text: The text content for the chunk.
            timestamp: The ISO 8601 timestamp for the chunk.
        """
        chunk = StreamChunk(
            chunk_type=chunk_type,
            text=text,
            sequence=self._sequence,
            timestamp=timestamp,
        )
        self._sequence += 1

        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            logger.warning(
                "Stream chunk queue full -- dropping %s chunk (seq=%d)",
                chunk_type.value,
                chunk.sequence,
            )

    async def _read_envelope(self) -> MessageEnvelope | None:
        """Read one framed envelope from the stream with timeout.

        Returns None on EOF, incomplete data, timeout, or decode error.

        Returns:
            Decoded MessageEnvelope, or None on any read failure.
        """
        try:
            header_bytes = await asyncio.wait_for(
                self._reader.readexactly(HEADER_SIZE),
                timeout=self._config.read_timeout,
            )
        except (
            asyncio.IncompleteReadError,
            ConnectionResetError,
            asyncio.TimeoutError,
        ):
            return None

        try:
            payload_length = unpack_header(header_bytes)
            payload_bytes = await asyncio.wait_for(
                self._reader.readexactly(payload_length),
                timeout=self._config.read_timeout,
            )
        except (
            asyncio.IncompleteReadError,
            ConnectionResetError,
            asyncio.TimeoutError,
        ):
            return None

        try:
            return decode_envelope(payload_bytes)
        except (ValueError, KeyError) as exc:
            logger.warning("Malformed stream envelope: %s", exc)
            return None
