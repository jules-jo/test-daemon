"""Per-client read/write coroutines with framing integration.

Provides dedicated reader and writer coroutines for each connected CLI
client, wiring the framing encode/decode layer into the connection
lifecycle manager. This module decouples read and write concerns into
independent async coroutines communicating through queues, enabling:

- Non-blocking bidirectional IO (daemon can push streaming output while
  reading client requests)
- Clean EOF detection and disconnect propagation
- Graceful shutdown via sentinel values and task cancellation
- Error isolation (write failures do not block reads and vice versa)
- Integration with ConnectionManager for lifecycle tracking

Architecture::

    StreamReader --> ClientReader --> inbox Queue --> consumer (dispatcher)
                                                        |
                                                        v
    StreamWriter <-- ClientWriter <-- outbox Queue <-- producer (handler)
                                                        |
                        ConnectionManager <-------------|
                        (register/deregister)

    ClientIO wraps both reader and writer into a single lifecycle unit
    and ties them to the ConnectionManager for event emission.

Usage::

    from jules_daemon.ipc.client_io import ClientIO
    from jules_daemon.ipc.connection_manager import ConnectionManager

    manager = ConnectionManager(event_bus=bus)

    async with ClientIO(
        reader=stream_reader,
        writer=stream_writer,
        client_id="client-abc",
        connected_at="2026-04-09T12:00:00Z",
        connection_manager=manager,
    ) as cio:
        while True:
            envelope = await cio.receive()
            if envelope is None:
                break  # client disconnected
            response = process(envelope)
            await cio.send(response)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from jules_daemon.ipc.connection_manager import ClientInfo, ConnectionManager
from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    decode_envelope,
    encode_frame,
    unpack_header,
)

__all__ = [
    "ClientIO",
    "ClientIOError",
    "ClientReader",
    "ClientWriter",
    "ReadError",
    "WriteError",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientIOError:
    """Base error descriptor for client IO operations.

    Attributes:
        client_id: The client that experienced the error.
        detail:    Human-readable description of what went wrong.
        cause:     The underlying exception, if any.
    """

    client_id: str
    detail: str
    cause: Exception | None = None

    def __str__(self) -> str:
        base = f"[{self.client_id}] {self.detail}"
        if self.cause is not None:
            return f"{base} (caused by {self.cause!r})"
        return base


@dataclass(frozen=True)
class ReadError(ClientIOError):
    """Error during frame reading or decoding."""


@dataclass(frozen=True)
class WriteError(ClientIOError):
    """Error during frame encoding or writing."""


# ---------------------------------------------------------------------------
# Type alias for error callback
# ---------------------------------------------------------------------------

ErrorCallback = Callable[[ClientIOError], Any]
"""Sync callable invoked when a read or write error occurs."""


def _noop_error_handler(_error: ClientIOError) -> None:
    """Default no-op error handler."""


# ---------------------------------------------------------------------------
# ClientReader
# ---------------------------------------------------------------------------


class ClientReader:
    """Per-client read coroutine that decodes framed messages from a stream.

    Reads length-prefixed frames from an ``asyncio.StreamReader``, decodes
    them into ``MessageEnvelope`` objects using the framing module, and
    delivers them to an inbox queue. On EOF (client disconnect), a ``None``
    sentinel is placed in the inbox.

    Handles:
        - Clean EOF detection (IncompleteReadError, ConnectionResetError)
        - Malformed frame errors (reported via on_error callback)
        - Cancellation for graceful shutdown

    The reader does NOT close the stream. Stream ownership belongs to
    the caller (typically ``ClientIO``).

    Args:
        reader:    The asyncio StreamReader to read frames from.
        client_id: Identifier for this client (used in logs and errors).
        inbox:     Queue where decoded envelopes (or None sentinel) are placed.
        on_error:  Optional callback invoked on malformed frame errors.
    """

    def __init__(
        self,
        *,
        reader: asyncio.StreamReader,
        client_id: str,
        inbox: asyncio.Queue[MessageEnvelope | None],
        on_error: ErrorCallback | None = None,
    ) -> None:
        self._reader = reader
        self._client_id = client_id
        self._inbox = inbox
        self._on_error = on_error or _noop_error_handler

    @property
    def client_id(self) -> str:
        """The client identifier for this reader."""
        return self._client_id

    async def run(self) -> None:
        """Read framed messages until EOF or cancellation.

        Each frame consists of a 4-byte length header followed by a JSON
        payload. Decoded envelopes are placed in the inbox queue. On EOF,
        a ``None`` sentinel is placed in the inbox to signal disconnect.

        Malformed frames trigger the ``on_error`` callback and terminate
        the read loop (the malformed frame breaks the stream framing
        alignment, so recovery is not possible).

        Raises:
            asyncio.CancelledError: If the coroutine is cancelled externally.
        """
        try:
            await self._read_loop()
        finally:
            await self._inbox.put(None)
            logger.debug("Reader stopped for client %s", self._client_id)

    async def _read_loop(self) -> None:
        """Internal read loop that processes frames."""
        while True:
            # Read the 4-byte length header
            try:
                header_bytes = await self._reader.readexactly(HEADER_SIZE)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                # EOF or connection lost -- clean exit
                return

            # Decode header and read payload
            try:
                payload_length = unpack_header(header_bytes)
                payload_bytes = await self._reader.readexactly(payload_length)
            except asyncio.IncompleteReadError:
                # Client disconnected mid-message
                return

            # Decode the envelope
            try:
                envelope = decode_envelope(payload_bytes)
            except (ValueError, KeyError) as exc:
                error = ReadError(
                    client_id=self._client_id,
                    detail=f"Malformed frame: {exc}",
                    cause=exc,
                )
                logger.warning("%s", error)
                self._on_error(error)
                return

            await self._inbox.put(envelope)


# ---------------------------------------------------------------------------
# ClientWriter
# ---------------------------------------------------------------------------


class ClientWriter:
    """Per-client write coroutine that encodes and sends framed messages.

    Consumes ``MessageEnvelope`` objects from an outbox queue, encodes them
    as length-prefixed frames using the framing module, and writes them to
    an ``asyncio.StreamWriter``. A ``None`` sentinel in the queue signals
    the writer to stop.

    Handles:
        - Queue consumption with blocking get
        - Write errors (BrokenPipeError, ConnectionResetError)
        - Cancellation for graceful shutdown

    The writer does NOT close the stream. Stream ownership belongs to
    the caller (typically ``ClientIO``).

    Args:
        writer:    The asyncio StreamWriter to write frames to.
        client_id: Identifier for this client (used in logs and errors).
        outbox:    Queue from which envelopes (or None sentinel) are consumed.
        on_error:  Optional callback invoked on write errors.
    """

    def __init__(
        self,
        *,
        writer: asyncio.StreamWriter,
        client_id: str,
        outbox: asyncio.Queue[MessageEnvelope | None],
        on_error: ErrorCallback | None = None,
    ) -> None:
        self._writer = writer
        self._client_id = client_id
        self._outbox = outbox
        self._on_error = on_error or _noop_error_handler

    @property
    def client_id(self) -> str:
        """The client identifier for this writer."""
        return self._client_id

    async def send(self, envelope: MessageEnvelope) -> None:
        """Queue an envelope for sending.

        This is a convenience method that puts the envelope on the outbox
        queue. The actual write happens in the ``run()`` coroutine.

        Args:
            envelope: The message envelope to send.
        """
        await self._outbox.put(envelope)

    async def run(self) -> None:
        """Consume envelopes from the outbox and write framed messages.

        Runs until a ``None`` sentinel is received from the queue, or
        until the coroutine is cancelled. Write errors (BrokenPipeError,
        ConnectionResetError) trigger the ``on_error`` callback and
        terminate the write loop.

        Raises:
            asyncio.CancelledError: If the coroutine is cancelled externally.
        """
        try:
            await self._write_loop()
        finally:
            logger.debug("Writer stopped for client %s", self._client_id)

    async def _write_loop(self) -> None:
        """Internal write loop that processes the outbox queue."""
        while True:
            envelope = await self._outbox.get()

            # None sentinel means stop
            if envelope is None:
                return

            try:
                frame = encode_frame(envelope)
                self._writer.write(frame)
                await self._writer.drain()
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                error = WriteError(
                    client_id=self._client_id,
                    detail=f"Write failed: {exc}",
                    cause=exc,
                )
                logger.warning("%s", error)
                self._on_error(error)
                return


# ---------------------------------------------------------------------------
# ClientIO lifecycle wrapper
# ---------------------------------------------------------------------------


class ClientIO:
    """Lifecycle wrapper that orchestrates per-client reader and writer.

    Creates and manages ``ClientReader`` and ``ClientWriter`` coroutines
    as asyncio tasks, tying their lifecycle to the ``ConnectionManager``
    for event emission (connect on start, disconnect on stop).

    The reader and writer communicate through separate queues:
        - ``inbox``:  Reader puts decoded envelopes here; consumer calls
          ``receive()`` to get them.
        - ``outbox``: Producer calls ``send()`` to queue envelopes;
          writer consumes and sends them.

    When the reader detects EOF (client disconnect), it sends a ``None``
    sentinel to the inbox and also signals the writer to stop. This
    ensures both coroutines shut down cleanly when the client disconnects.

    Supports both explicit ``start()``/``stop()`` and async context
    manager usage.

    Args:
        reader:             asyncio StreamReader for the client.
        writer:             asyncio StreamWriter for the client.
        client_id:          Unique identifier for this client.
        connected_at:       ISO 8601 timestamp of connection time.
        connection_manager: Optional ConnectionManager for lifecycle events.
        on_error:           Optional callback for reader/writer errors.
    """

    def __init__(
        self,
        *,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        client_id: str,
        connected_at: str,
        connection_manager: ConnectionManager | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        self._stream_reader = reader
        self._stream_writer = writer
        self._client_id = client_id
        self._connected_at = connected_at
        self._connection_manager = connection_manager
        self._on_error = on_error or _noop_error_handler

        self._inbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()
        self._outbox: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue()

        self._client_reader = ClientReader(
            reader=self._stream_reader,
            client_id=self._client_id,
            inbox=self._inbox,
            on_error=self._on_error,
        )
        self._client_writer = ClientWriter(
            writer=self._stream_writer,
            client_id=self._client_id,
            outbox=self._outbox,
            on_error=self._on_error,
        )

        self._reader_task: asyncio.Task[None] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._closed = False
        self._stopped = False
        self._started = False

    # -- Properties --

    @property
    def client_id(self) -> str:
        """The client identifier."""
        return self._client_id

    @property
    def is_closed(self) -> bool:
        """Whether the IO session has been closed (EOF or explicit stop)."""
        return self._closed

    # -- Async context manager --

    async def __aenter__(self) -> ClientIO:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.stop()

    # -- Lifecycle --

    async def start(self) -> None:
        """Start reader and writer coroutines, register with manager.

        Registers the client with the ConnectionManager (if provided),
        then spawns the reader, writer, and monitor tasks.

        Raises:
            RuntimeError: If start() has already been called.
        """
        if self._started:
            raise RuntimeError(
                f"ClientIO for {self._client_id} is already started"
            )
        self._started = True

        # Register with connection manager
        if self._connection_manager is not None:
            client_info = ClientInfo(
                client_id=self._client_id,
                connected_at=self._connected_at,
            )
            await self._connection_manager.add_client(client_info)

        # Spawn reader and writer tasks
        self._reader_task = asyncio.create_task(
            self._client_reader.run(),
            name=f"reader-{self._client_id}",
        )
        self._writer_task = asyncio.create_task(
            self._client_writer.run(),
            name=f"writer-{self._client_id}",
        )

        # Monitor task watches for reader completion to signal writer
        self._monitor_task = asyncio.create_task(
            self._monitor_reader(),
            name=f"monitor-{self._client_id}",
        )

        logger.info("ClientIO started for %s", self._client_id)

    async def stop(self) -> None:
        """Gracefully stop reader and writer, deregister from manager.

        Sends a stop sentinel to the writer, cancels the reader if still
        running, waits for all tasks to complete, closes the stream
        writer, and deregisters from the ConnectionManager.

        This method is idempotent: calling it on an already-stopped
        ClientIO is a safe no-op.
        """
        if self._stopped:
            return

        self._stopped = True
        self._closed = True

        # Signal the writer to stop
        try:
            self._outbox.put_nowait(None)
        except asyncio.QueueFull:
            pass

        # Cancel reader if still running
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()

        # Cancel monitor if still running
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()

        # Wait for all tasks to finish
        tasks = [
            t
            for t in (self._reader_task, self._writer_task, self._monitor_task)
            if t is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Close the stream writer
        await self._close_writer()

        # Deregister from connection manager
        if self._connection_manager is not None:
            await self._connection_manager.remove_client(self._client_id)

        logger.info("ClientIO stopped for %s", self._client_id)

    # -- Public IO methods --

    async def send(self, envelope: MessageEnvelope) -> None:
        """Queue an envelope for sending to the client.

        Args:
            envelope: The message envelope to send.
        """
        await self._outbox.put(envelope)

    async def receive(self) -> MessageEnvelope | None:
        """Wait for the next envelope from the client.

        Returns:
            The next MessageEnvelope, or None if the client disconnected.
        """
        return await self._inbox.get()

    # -- Internal helpers --

    async def _monitor_reader(self) -> None:
        """Watch the reader task and propagate shutdown when it ends.

        When the reader finishes (due to EOF, error, or cancellation),
        this monitor signals the writer to stop by putting a None
        sentinel on the outbox, and marks the IO session as closed.
        """
        if self._reader_task is None:
            return

        try:
            await self._reader_task
        except asyncio.CancelledError:
            pass

        # Reader is done -- signal writer to stop
        self._closed = True
        try:
            self._outbox.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def _close_writer(self) -> None:
        """Safely close the underlying StreamWriter."""
        try:
            if not self._stream_writer.is_closing():
                self._stream_writer.close()
                await self._stream_writer.wait_closed()
        except Exception as exc:
            logger.debug(
                "Error closing writer for %s: %s", self._client_id, exc
            )
