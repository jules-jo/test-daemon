"""IPC client-side watch command for real-time SSH output streaming.

Connects to the daemon via a Unix domain socket, sends a watch
subscription request for a specific job (or the current active run),
enters a streaming-read loop that prints incoming SSH output lines
in real time, and cleanly detaches on user interrupt or job completion.

Key responsibilities:

- **Subscribe**: Send a framed watch REQUEST to the daemon and parse the
  subscription RESPONSE (containing subscriber_id and buffered line count).

- **Stream loop**: Read STREAM envelopes from the daemon in a loop, format
  each output line (optionally with timestamps and sequence numbers), and
  print to the output stream. The loop terminates on:
    - End-of-stream sentinel (``is_end=True``) -- job completed.
    - EOF / ConnectionReset -- daemon connection lost.
    - ERROR envelope -- daemon-side error during streaming.
    - asyncio.CancelledError -- user interrupt (Ctrl+C).

- **Clean detach**: On exit (whether from completion, error, or interrupt),
  send an unwatch REQUEST so the daemon cleans up the subscriber queue.
  Close the transport. The daemon continues monitoring autonomously --
  CLI disconnect does not affect running tests.

Architecture::

    CLI Process                         Daemon Process
        |                                    |
        |-- connect (Unix socket) ---------->|
        |-- REQUEST {watch, job_id} -------->|
        |<-- RESPONSE {subscribed, sub_id} --|
        |                                    |
        |<-- STREAM {line, seq, is_end} -----|  (repeated)
        |    print(line)                     |
        |                                    |
        |  (on completion / Ctrl+C / error)  |
        |-- REQUEST {unwatch, sub_id} ------>|
        |<-- RESPONSE {unsubscribed} --------|
        |-- close connection ---------------->|

Usage::

    from jules_daemon.ipc.watch_client import WatchClient, WatchClientConfig

    config = WatchClientConfig(socket_path="/run/jules/daemon.sock")
    client = WatchClient(config=config)

    # Blocking call: streams output until job completes or user interrupts
    result = await client.run(job_id="job-abc")
    print(f"Exit: {result.exit_reason.value}, lines: {result.lines_received}")
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, TextIO, runtime_checkable

from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
    unpack_header,
)

__all__ = [
    "StreamWriterLike",
    "WatchClient",
    "WatchClientConfig",
    "WatchClientResult",
    "WatchExitReason",
    "format_output_line",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERB_WATCH = "watch"
_VERB_UNWATCH = "unwatch"
_DEFAULT_CONNECT_TIMEOUT = 5.0
_DEFAULT_READ_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Writer protocol (structural typing for testability)
# ---------------------------------------------------------------------------


@runtime_checkable
class StreamWriterLike(Protocol):
    """Protocol for stream writers (real or mock).

    Defines the minimal interface needed by the watch client to send
    framed messages and close the connection. Using a protocol instead
    of ``asyncio.StreamWriter | object`` gives type-safe injection
    without coupling to the concrete asyncio implementation.
    """

    def write(self, data: bytes) -> None: ...  # pragma: no cover

    async def drain(self) -> None: ...  # pragma: no cover

    def close(self) -> None: ...  # pragma: no cover

    async def wait_closed(self) -> None: ...  # pragma: no cover

    def is_closing(self) -> bool: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _generate_msg_id() -> str:
    """Generate a unique message ID for request-response correlation."""
    return f"cli-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# WatchExitReason enum
# ---------------------------------------------------------------------------


class WatchExitReason(Enum):
    """Why the watch stream ended.

    Values:
        JOB_COMPLETED:   The job finished and sent an end-of-stream sentinel.
        USER_INTERRUPT:   The user pressed Ctrl+C (or the task was cancelled).
        DAEMON_ERROR:     The daemon returned an ERROR envelope.
        CONNECTION_LOST:  The socket connection was dropped unexpectedly.
    """

    JOB_COMPLETED = "job_completed"
    USER_INTERRUPT = "user_interrupt"
    DAEMON_ERROR = "daemon_error"
    CONNECTION_LOST = "connection_lost"


# ---------------------------------------------------------------------------
# WatchClientConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchClientConfig:
    """Immutable configuration for the watch client.

    Attributes:
        socket_path:      Path to the daemon's Unix domain socket.
        connect_timeout:  Maximum seconds to wait for socket connection.
        read_timeout:     Maximum seconds to wait between stream messages
                          before considering the connection stale. The stream
                          loop uses this as the per-read timeout; exceeding
                          it triggers a reconnect or exit (depending on
                          daemon state).
        show_timestamps:  When True, prefix each output line with its
                          ISO 8601 timestamp.
        show_sequence:    When True, prefix each output line with its
                          sequence number.
    """

    socket_path: str
    connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT
    read_timeout: float = _DEFAULT_READ_TIMEOUT
    show_timestamps: bool = False
    show_sequence: bool = False

    def __post_init__(self) -> None:
        if not self.socket_path or not self.socket_path.strip():
            raise ValueError("socket_path must not be empty")
        if self.connect_timeout <= 0:
            raise ValueError(
                f"connect_timeout must be positive, got {self.connect_timeout}"
            )
        if self.read_timeout <= 0:
            raise ValueError(
                f"read_timeout must be positive, got {self.read_timeout}"
            )


# ---------------------------------------------------------------------------
# WatchClientResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchClientResult:
    """Immutable result of a watch session.

    Attributes:
        exit_reason:    Why the watch stream ended.
        lines_received: Total number of output lines printed during the session.
        job_id:         The job ID that was watched (may differ from the
                        requested ID if the daemon resolved "current run").
        error_message:  Human-readable error description, or None if clean exit.
    """

    exit_reason: WatchExitReason
    lines_received: int
    job_id: str | None
    error_message: str | None


# ---------------------------------------------------------------------------
# Line formatting
# ---------------------------------------------------------------------------


def format_output_line(
    *,
    line: str,
    sequence: int,
    timestamp: str,
    show_timestamps: bool,
    show_sequence: bool,
) -> str:
    """Format an output line with optional metadata prefixes.

    Constructs the display string by prepending timestamp and/or
    sequence number as configured. The base line text is always
    included.

    Args:
        line:            The raw output text from the SSH session.
        sequence:        Monotonically increasing line counter.
        timestamp:       ISO 8601 timestamp of when the line was captured.
        show_timestamps: Whether to include the timestamp prefix.
        show_sequence:   Whether to include the sequence number prefix.

    Returns:
        Formatted string ready for terminal display (no trailing newline).
    """
    parts: list[str] = []

    if show_timestamps:
        parts.append(f"[{timestamp}]")

    if show_sequence:
        parts.append(f"[#{sequence}]")

    parts.append(line)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# WatchClient
# ---------------------------------------------------------------------------


class WatchClient:
    """IPC client that streams SSH output from the daemon in real time.

    Connects to the daemon's Unix domain socket, sends a watch request,
    and enters a streaming-read loop that prints output lines as they
    arrive. Cleanly detaches on job completion, user interrupt, daemon
    error, or connection loss.

    The client is designed for single-use: create a new instance for each
    watch session.

    Args:
        config: Connection and display configuration.
        output: Text IO stream for printing output lines. Defaults to
            ``sys.stdout``.
    """

    def __init__(
        self,
        *,
        config: WatchClientConfig,
        output: TextIO | None = None,
    ) -> None:
        self._config = config
        self._output = output or sys.stdout

    # -- Public API ----------------------------------------------------------

    async def run(self, *, job_id: str | None = None) -> WatchClientResult:
        """Connect to the daemon and stream output for a job.

        This is the main entry point. It:
        1. Opens a Unix socket connection to the daemon.
        2. Sends a watch subscription request.
        3. Reads and prints STREAM envelopes in real time.
        4. Sends an unwatch request on exit for clean cleanup.
        5. Closes the connection.

        Args:
            job_id: The job to watch. When None, the daemon resolves
                to the current active run.

        Returns:
            WatchClientResult describing how the session ended.
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self._config.socket_path),
                timeout=self._config.connect_timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            logger.error("Failed to connect to daemon: %s", exc)
            return WatchClientResult(
                exit_reason=WatchExitReason.CONNECTION_LOST,
                lines_received=0,
                job_id=job_id,
                error_message=f"Connection failed: {exc}",
            )

        try:
            return await self._run_with_transport(
                reader=reader,
                writer=writer,
                job_id=job_id,
            )
        finally:
            await self._close_transport(writer)

    # -- Internal: full run with pre-existing transport ----------------------

    async def _run_with_transport(
        self,
        *,
        reader: asyncio.StreamReader,
        writer: StreamWriterLike,
        job_id: str | None,
    ) -> WatchClientResult:
        """Execute the watch session on an existing reader/writer pair.

        Separated from ``run()`` for testability -- tests can inject
        mock StreamReader/StreamWriter without a real socket.

        Args:
            reader: Stream to read framed messages from.
            writer: Stream to write framed messages to.
            job_id: The job to watch (None for current run).

        Returns:
            WatchClientResult.
        """
        # 1. Send watch request
        try:
            await self._send_watch_request(writer, job_id=job_id)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            logger.error("Failed to send watch request: %s", exc)
            return WatchClientResult(
                exit_reason=WatchExitReason.CONNECTION_LOST,
                lines_received=0,
                job_id=job_id,
                error_message=f"Send failed: {exc}",
            )

        # 2. Read subscription response
        response = await self._read_envelope(reader)
        if response is None:
            return WatchClientResult(
                exit_reason=WatchExitReason.CONNECTION_LOST,
                lines_received=0,
                job_id=job_id,
                error_message="Connection lost before subscription response",
            )

        try:
            resolved_job_id, subscriber_id, buffered = (
                self._parse_subscription_response(response)
            )
        except ValueError as exc:
            return WatchClientResult(
                exit_reason=WatchExitReason.DAEMON_ERROR,
                lines_received=0,
                job_id=job_id,
                error_message=str(exc),
            )

        # Print subscription confirmation
        self._write_info(
            f"Watching job {resolved_job_id} "
            f"(subscriber={subscriber_id}, buffered={buffered})"
        )

        # 3. Enter streaming loop (handles CancelledError internally)
        exit_reason, lines_count = await self._stream_loop(reader)

        # 4. Send unwatch for clean cleanup (best-effort)
        await self._try_send_unwatch(
            writer, job_id=resolved_job_id, subscriber_id=subscriber_id
        )

        # Build result with appropriate error message
        clean_exits = (
            WatchExitReason.JOB_COMPLETED,
            WatchExitReason.USER_INTERRUPT,
        )
        error_msg = (
            None if exit_reason in clean_exits
            else f"Watch ended: {exit_reason.value}"
        )

        return WatchClientResult(
            exit_reason=exit_reason,
            lines_received=lines_count,
            job_id=resolved_job_id,
            error_message=error_msg,
        )

    # -- Internal: send watch request ----------------------------------------

    async def _send_watch_request(
        self,
        writer: StreamWriterLike,
        *,
        job_id: str | None,
    ) -> None:
        """Encode and send a framed watch request envelope.

        Args:
            writer: The StreamWriter (or mock) to write to.
            job_id: Job to watch, or None for current run.
        """
        payload: dict[str, object] = {"verb": _VERB_WATCH}
        if job_id is not None:
            payload["job_id"] = job_id

        envelope = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id=_generate_msg_id(),
            timestamp=_now_iso(),
            payload=payload,
        )

        frame = encode_frame(envelope)
        writer.write(frame)
        await writer.drain()

        logger.debug(
            "Sent watch request (msg_id=%s, job_id=%s)",
            envelope.msg_id,
            job_id,
        )

    # -- Internal: send unwatch request --------------------------------------

    async def _send_unwatch_request(
        self,
        writer: StreamWriterLike,
        *,
        job_id: str,
        subscriber_id: str,
    ) -> None:
        """Encode and send a framed unwatch request envelope.

        Args:
            writer: The StreamWriter (or mock) to write to.
            job_id: The job to unsubscribe from.
            subscriber_id: The subscriber handle ID to remove.
        """
        envelope = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id=_generate_msg_id(),
            timestamp=_now_iso(),
            payload={
                "verb": _VERB_UNWATCH,
                "job_id": job_id,
                "subscriber_id": subscriber_id,
            },
        )

        frame = encode_frame(envelope)
        writer.write(frame)
        await writer.drain()

        logger.debug(
            "Sent unwatch request (job_id=%s, subscriber_id=%s)",
            job_id,
            subscriber_id,
        )

    async def _try_send_unwatch(
        self,
        writer: StreamWriterLike,
        *,
        job_id: str,
        subscriber_id: str,
    ) -> None:
        """Best-effort unwatch: swallow errors silently.

        The daemon will eventually clean up orphaned subscriptions
        on disconnect, so this is a courtesy cleanup.

        Args:
            writer: The StreamWriter (or mock) to write to.
            job_id: The job to unsubscribe from.
            subscriber_id: The subscriber handle ID to remove.
        """
        try:
            await self._send_unwatch_request(
                writer, job_id=job_id, subscriber_id=subscriber_id
            )
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            logger.debug(
                "Unwatch send failed (expected during disconnect): %s", exc
            )

    # -- Internal: read a single envelope ------------------------------------

    async def _read_envelope(
        self,
        reader: asyncio.StreamReader,
        *,
        timeout: float | None = None,
    ) -> MessageEnvelope | None:
        """Read one framed envelope from the stream.

        Returns None on EOF, incomplete data, or timeout (connection
        lost or stale).

        Args:
            reader:  The StreamReader to read from.
            timeout: Maximum seconds to wait for the next envelope.
                When None, uses the config's ``read_timeout``.

        Returns:
            Decoded MessageEnvelope, or None on connection loss or timeout.
        """
        effective_timeout = timeout if timeout is not None else self._config.read_timeout

        try:
            header_bytes = await asyncio.wait_for(
                reader.readexactly(HEADER_SIZE),
                timeout=effective_timeout,
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
                reader.readexactly(payload_length),
                timeout=effective_timeout,
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
            logger.warning("Malformed envelope from daemon: %s", exc)
            return None

    # -- Internal: parse subscription response -------------------------------

    def _parse_subscription_response(
        self,
        envelope: MessageEnvelope,
    ) -> tuple[str, str, int]:
        """Parse the daemon's watch subscription response.

        Validates the response envelope and extracts the subscription
        metadata.

        Args:
            envelope: The response envelope from the daemon.

        Returns:
            Tuple of (job_id, subscriber_id, buffered_lines).

        Raises:
            ValueError: If the response is an error or missing fields.
        """
        if envelope.msg_type == MessageType.ERROR:
            error_msg = envelope.payload.get("error", "Unknown daemon error")
            raise ValueError(error_msg)

        payload = envelope.payload
        job_id = payload.get("job_id")
        subscriber_id = payload.get("subscriber_id")
        buffered_lines = payload.get("buffered_lines", 0)

        if not job_id:
            raise ValueError(
                "Subscription response missing job_id"
            )
        if not subscriber_id:
            raise ValueError(
                "Subscription response missing subscriber_id"
            )

        return (str(job_id), str(subscriber_id), int(buffered_lines))

    # -- Internal: streaming loop --------------------------------------------

    async def _stream_loop(
        self,
        reader: asyncio.StreamReader,
    ) -> tuple[WatchExitReason, int]:
        """Read and print STREAM envelopes until termination.

        Processes envelopes in a loop:
        - STREAM with ``is_end=False``: format and print the line.
        - STREAM with ``is_end=True``: job completed, exit loop.
        - ERROR: daemon error, exit loop.
        - Other types: skip silently (stale responses, etc.).
        - EOF / timeout: connection lost, exit loop.
        - CancelledError: user interrupt, exit loop (preserving count).

        Args:
            reader: The StreamReader to read envelopes from.

        Returns:
            Tuple of (exit_reason, lines_printed). The count reflects
            lines actually printed, even when interrupted mid-stream.
        """
        lines_count = 0

        try:
            while True:
                envelope = await self._read_envelope(reader)

                if envelope is None:
                    # EOF, timeout, or connection lost
                    return (WatchExitReason.CONNECTION_LOST, lines_count)

                if envelope.msg_type == MessageType.ERROR:
                    error_msg = envelope.payload.get("error", "Unknown error")
                    self._write_info(f"Daemon error: {error_msg}")
                    return (WatchExitReason.DAEMON_ERROR, lines_count)

                if envelope.msg_type == MessageType.STREAM:
                    payload = envelope.payload
                    is_end = payload.get("is_end", False)

                    if is_end:
                        self._write_info("Job completed.")
                        return (WatchExitReason.JOB_COMPLETED, lines_count)

                    # Format and print the output line
                    line_text = payload.get("line", "")
                    sequence = payload.get("sequence", 0)
                    timestamp = envelope.timestamp

                    formatted = format_output_line(
                        line=line_text,
                        sequence=sequence,
                        timestamp=timestamp,
                        show_timestamps=self._config.show_timestamps,
                        show_sequence=self._config.show_sequence,
                    )
                    self._write_line(formatted)
                    lines_count += 1

                # else: skip non-STREAM, non-ERROR envelopes silently

        except asyncio.CancelledError:
            self._write_info("Detaching from stream (user interrupt).")
            return (WatchExitReason.USER_INTERRUPT, lines_count)

    # -- Internal: output helpers --------------------------------------------

    def _write_line(self, text: str) -> None:
        """Write a line of output followed by a newline.

        Args:
            text: The text to write.
        """
        print(text, file=self._output, flush=True)

    def _write_info(self, message: str) -> None:
        """Write an informational message (prefixed with --).

        Args:
            message: The info message to write.
        """
        self._output.write(f"-- {message}\n")
        self._output.flush()

    # -- Internal: transport cleanup -----------------------------------------

    async def _close_transport(
        self,
        writer: StreamWriterLike,
    ) -> None:
        """Safely close the underlying StreamWriter.

        Handles the case where the writer is already closing or the
        transport has been lost.

        Args:
            writer: The stream writer to close.
        """
        try:
            if writer.is_closing():
                return
            writer.close()
            await writer.wait_closed()
        except (OSError, ConnectionResetError) as exc:
            # Expected if transport was already dropped by the daemon
            logger.debug("Error closing transport: %s", exc)
