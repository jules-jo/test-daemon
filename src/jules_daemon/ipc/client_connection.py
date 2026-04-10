"""IPC client connection module for CLI-daemon communication.

Provides the base client-side connection that all CLI verbs use to
communicate with the daemon. Handles three responsibilities:

    1. **Socket discovery**: Resolves the daemon socket path via the
       ``socket_discovery`` module (env var, XDG, tmpdir fallback).

    2. **Connection establishment**: Opens an async Unix domain socket
       connection with configurable timeout.

    3. **Protocol handshake**: Exchanges a versioned handshake with the
       daemon to verify compatibility and obtain daemon metadata (PID,
       uptime). The handshake uses the same framed MessageEnvelope
       protocol as all other IPC messages.

After a successful ``connect()``, the connection provides ``send()``
and ``receive()`` methods for exchanging framed MessageEnvelopes with
the daemon.

Handshake protocol::

    CLI                                     Daemon
     |                                        |
     |-- REQUEST {verb: "handshake",          |
     |     protocol_version: N,               |
     |     client_pid: PID}              ---->|
     |                                        |
     |<-- RESPONSE {verb: "handshake",        |
     |     status: "ok",                      |
     |     protocol_version: N,               |
     |     daemon_pid: PID,                   |
     |     daemon_uptime_seconds: X}     -----|
     |                                        |

If the daemon does not support the client's protocol version, it
returns an ERROR envelope instead of a RESPONSE.

Usage::

    from pathlib import Path
    from jules_daemon.ipc.client_connection import (
        ClientConnection,
        ConnectionConfig,
    )

    # Explicit socket path
    config = ConnectionConfig(socket_path=Path("/run/user/1000/jules/daemon.sock"))
    async with ClientConnection(config=config) as conn:
        await conn.send(request_envelope)
        response = await conn.receive()

    # Auto-discover socket path
    config = ConnectionConfig(socket_path=None)
    async with ClientConnection(config=config) as conn:
        ...
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
    unpack_header,
)
from jules_daemon.ipc.socket_discovery import default_socket_path

__all__ = [
    "HANDSHAKE_VERB",
    "PROTOCOL_VERSION",
    "ClientConnection",
    "ConnectionConfig",
    "ConnectionError",
    "ConnectionState",
    "HandshakeError",
    "HandshakeResult",
    "StreamWriterLike",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION: int = 1
"""Current IPC protocol version. Incremented on breaking changes."""

HANDSHAKE_VERB: str = "handshake"
"""Verb used in the initial handshake REQUEST envelope."""

_DEFAULT_CONNECT_TIMEOUT: float = 5.0
"""Default maximum seconds to wait for socket connection."""

_DEFAULT_HANDSHAKE_TIMEOUT: float = 5.0
"""Default maximum seconds to wait for handshake response."""

_DEFAULT_RECEIVE_TIMEOUT: float = 30.0
"""Default maximum seconds to wait for a response envelope."""


# ---------------------------------------------------------------------------
# StreamWriter protocol (structural typing for testability)
# ---------------------------------------------------------------------------


@runtime_checkable
class StreamWriterLike(Protocol):
    """Protocol for stream writers (real or mock).

    Defines the minimal interface needed by the client connection to
    send framed messages and close the transport. Using a protocol
    instead of ``asyncio.StreamWriter`` directly gives type-safe
    injection without coupling to the concrete asyncio implementation.
    """

    def write(self, data: bytes) -> None: ...  # pragma: no cover

    async def drain(self) -> None: ...  # pragma: no cover

    def close(self) -> None: ...  # pragma: no cover

    async def wait_closed(self) -> None: ...  # pragma: no cover

    def is_closing(self) -> bool: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------


class ConnectionError(Exception):
    """Raised when the IPC connection to the daemon fails.

    This is raised by the context manager when connect() returns a
    failed HandshakeResult, allowing ``async with`` usage to fail
    fast with a meaningful error.
    """


class HandshakeError(ConnectionError):
    """Raised when the handshake with the daemon fails.

    A specialization of ConnectionError that indicates the TCP/Unix
    socket connected but the protocol handshake was rejected or timed out.
    """


# ---------------------------------------------------------------------------
# ConnectionState enum
# ---------------------------------------------------------------------------


class ConnectionState(Enum):
    """Lifecycle state of a client connection.

    Values:
        DISCONNECTED: Not yet connected. Initial state.
        CONNECTING:   Socket connection in progress.
        HANDSHAKING:  Socket connected, handshake in progress.
        CONNECTED:    Handshake complete, ready for send/receive.
        CLOSED:       Connection has been closed (terminal state).
    """

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    HANDSHAKING = "handshaking"
    CONNECTED = "connected"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# ConnectionConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectionConfig:
    """Immutable configuration for a client connection.

    Attributes:
        socket_path:       Path to the daemon's Unix domain socket.
                           When None, the socket path is auto-discovered
                           using the standard search order.
        connect_timeout:   Maximum seconds to wait for socket connection.
        handshake_timeout: Maximum seconds to wait for the handshake
                           response from the daemon.
        protocol_version:  IPC protocol version to request in the
                           handshake. Must be positive.
    """

    socket_path: Path | None = None
    connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT
    handshake_timeout: float = _DEFAULT_HANDSHAKE_TIMEOUT
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.connect_timeout <= 0:
            raise ValueError(
                f"connect_timeout must be positive, got {self.connect_timeout}"
            )
        if self.handshake_timeout <= 0:
            raise ValueError(
                f"handshake_timeout must be positive, got {self.handshake_timeout}"
            )
        if self.protocol_version <= 0:
            raise ValueError(
                f"protocol_version must be positive, got {self.protocol_version}"
            )


# ---------------------------------------------------------------------------
# HandshakeResult dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandshakeResult:
    """Immutable result of the protocol handshake with the daemon.

    Captures the outcome of the handshake exchange. On success, contains
    daemon metadata (PID, uptime, protocol version). On failure, contains
    a human-readable error description.

    Attributes:
        success:               True if handshake completed successfully.
        protocol_version:      Protocol version the daemon supports.
                               0 on failure.
        daemon_pid:            Daemon process ID. None on failure.
        daemon_uptime_seconds: Daemon uptime in seconds. None on failure.
        error:                 Error description on failure. None on success.
        pending_failure:       Unreported failure from a background run,
                               present when the daemon has a stored
                               failure that the client has not yet seen.
    """

    success: bool
    protocol_version: int
    daemon_pid: int | None
    daemon_uptime_seconds: float | None
    error: str | None
    pending_failure: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _generate_msg_id() -> str:
    """Generate a unique message ID for request-response correlation."""
    return f"cli-{uuid.uuid4().hex[:12]}"


def _build_handshake_request(protocol_version: int) -> MessageEnvelope:
    """Build the handshake REQUEST envelope sent by the client.

    Args:
        protocol_version: The protocol version the client supports.

    Returns:
        MessageEnvelope with the handshake verb and version metadata.
    """
    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=_generate_msg_id(),
        timestamp=_now_iso(),
        payload={
            "verb": HANDSHAKE_VERB,
            "protocol_version": protocol_version,
            "client_pid": os.getpid(),
        },
    )


def _parse_handshake_response(
    envelope: MessageEnvelope,
    expected_version: int,
) -> HandshakeResult:
    """Parse the daemon's handshake response into a HandshakeResult.

    Validates the response envelope and extracts daemon metadata.
    Returns a failure result if:
    - The envelope is an ERROR type.
    - Required fields are missing from the payload.
    - The daemon's protocol version does not match the expected version.

    Args:
        envelope: The response envelope from the daemon.
        expected_version: The protocol version the client requested.

    Returns:
        HandshakeResult reflecting the handshake outcome.
    """
    # Handle ERROR envelope
    if envelope.msg_type == MessageType.ERROR:
        error_msg = envelope.payload.get("error", "Unknown daemon error")
        return HandshakeResult(
            success=False,
            protocol_version=0,
            daemon_pid=None,
            daemon_uptime_seconds=None,
            error=f"Daemon rejected handshake: {error_msg}",
        )

    payload = envelope.payload
    daemon_version = payload.get("protocol_version", 0)
    daemon_pid = payload.get("daemon_pid")
    daemon_uptime = payload.get("daemon_uptime_seconds")

    # Version mismatch check
    if daemon_version != expected_version:
        return HandshakeResult(
            success=False,
            protocol_version=int(daemon_version),
            daemon_pid=int(daemon_pid) if daemon_pid is not None else None,
            daemon_uptime_seconds=(
                float(daemon_uptime) if daemon_uptime is not None else None
            ),
            error=(
                f"Protocol version mismatch: client={expected_version}, "
                f"daemon={daemon_version}"
            ),
        )

    pending_failure = payload.get("pending_failure")
    return HandshakeResult(
        success=True,
        protocol_version=int(daemon_version),
        daemon_pid=int(daemon_pid) if daemon_pid is not None else None,
        daemon_uptime_seconds=(
            float(daemon_uptime) if daemon_uptime is not None else None
        ),
        error=None,
        pending_failure=pending_failure if isinstance(pending_failure, str) else None,
    )


def _failure_result(error: str) -> HandshakeResult:
    """Build a failure HandshakeResult with the given error message.

    Args:
        error: Human-readable error description.

    Returns:
        HandshakeResult with success=False and the error set.
    """
    return HandshakeResult(
        success=False,
        protocol_version=0,
        daemon_pid=None,
        daemon_uptime_seconds=None,
        error=error,
    )


# ---------------------------------------------------------------------------
# ClientConnection
# ---------------------------------------------------------------------------


class ClientConnection:
    """IPC client connection for CLI-daemon communication.

    Handles socket discovery, connection establishment, protocol
    handshake, and framed message exchange. After ``connect()``
    completes successfully, use ``send()`` and ``receive()`` to
    exchange MessageEnvelopes with the daemon.

    Supports two usage patterns:

    **Explicit lifecycle:**

        conn = ClientConnection(config=config)
        result = await conn.connect()
        if result.success:
            await conn.send(request)
            response = await conn.receive()
        await conn.close()

    **Context manager:**

        async with ClientConnection(config=config) as conn:
            await conn.send(request)
            response = await conn.receive()

    The context manager raises ``ConnectionError`` if connect fails,
    making it safe for "fail fast" usage.

    Args:
        config: Connection configuration (socket path, timeouts, version).
    """

    def __init__(self, *, config: ConnectionConfig) -> None:
        self._config = config
        self._state = ConnectionState.DISCONNECTED
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._resolved_path: Path | None = None

    # -- Properties --

    @property
    def config(self) -> ConnectionConfig:
        """The connection configuration."""
        return self._config

    @property
    def state(self) -> ConnectionState:
        """Current connection lifecycle state."""
        return self._state

    @property
    def resolved_path(self) -> Path | None:
        """The actual socket path used (after discovery). None if not yet resolved."""
        return self._resolved_path

    # -- Async context manager --

    async def __aenter__(self) -> ClientConnection:
        """Connect and handshake on context entry.

        Raises:
            ConnectionError: If connect or handshake fails.
        """
        result = await self.connect()
        if not result.success:
            error_msg = result.error or "Unknown connection error"
            raise ConnectionError(
                f"Failed to connect to daemon: {error_msg}"
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Close the connection on context exit."""
        await self.close()

    # -- Public API: connect --

    async def connect(self) -> HandshakeResult:
        """Establish a connection to the daemon and perform the handshake.

        This is the primary entry point. It:
        1. Resolves the socket path (explicit or auto-discovered).
        2. Opens a Unix domain socket connection with timeout.
        3. Sends a handshake REQUEST and reads the RESPONSE.
        4. Validates the handshake and returns the result.

        On success, transitions to CONNECTED state. On failure, remains
        in DISCONNECTED state.

        Returns:
            HandshakeResult describing the handshake outcome. Check
            ``result.success`` to determine if the connection is usable.
        """
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            return _failure_result(
                f"Connection already in state {self._state.value}"
            )

        # Step 1: Resolve socket path
        socket_path = self._resolve_socket_path()
        if socket_path is None:
            return _failure_result("Could not determine daemon socket path")

        self._resolved_path = socket_path

        # Step 2: Establish connection (Unix socket or TCP on Windows)
        self._state = ConnectionState.CONNECTING
        logger.debug("Connecting to daemon at %s", socket_path)

        try:
            import sys

            if sys.platform == "win32":
                # Windows: read TCP port from the socket path file
                try:
                    port = int(socket_path.read_text().strip())
                except (FileNotFoundError, ValueError) as read_exc:
                    self._state = ConnectionState.DISCONNECTED
                    return _failure_result(
                        f"Daemon not running (no port file at {socket_path}): {read_exc}"
                    )
                connect_coro = asyncio.open_connection("127.0.0.1", port)
            else:
                connect_coro = asyncio.open_unix_connection(str(socket_path))

            reader, writer = await asyncio.wait_for(
                connect_coro,
                timeout=self._config.connect_timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            logger.error("Failed to connect to daemon at %s: %s", socket_path, exc)
            self._state = ConnectionState.DISCONNECTED
            return _failure_result(f"Connection failed: {exc}")

        self._reader = reader
        self._writer = writer

        # Step 3: Perform handshake
        self._state = ConnectionState.HANDSHAKING
        result = await self._perform_handshake(reader, writer)

        if result.success:
            self._state = ConnectionState.CONNECTED
            logger.info(
                "Connected to daemon (pid=%s, uptime=%.1fs, proto=v%d) "
                "via %s",
                result.daemon_pid,
                result.daemon_uptime_seconds or 0.0,
                result.protocol_version,
                socket_path,
            )
        else:
            self._state = ConnectionState.DISCONNECTED
            # Clean up the socket on handshake failure
            await self._close_writer()
            logger.warning(
                "Handshake failed with daemon at %s: %s",
                socket_path,
                result.error,
            )

        return result

    # -- Public API: send / receive --

    async def send(self, envelope: MessageEnvelope) -> None:
        """Send a framed MessageEnvelope to the daemon.

        Args:
            envelope: The message to send.

        Raises:
            ConnectionError: If not connected or the write fails.
        """
        if self._writer is None:
            raise ConnectionError("Not connected -- call connect() first")

        frame = encode_frame(envelope)
        self._writer.write(frame)
        await self._writer.drain()

    async def receive(
        self,
        *,
        timeout: float | None = None,
    ) -> MessageEnvelope | None:
        """Read and decode one framed MessageEnvelope from the daemon.

        Args:
            timeout: Maximum seconds to wait for the next envelope.
                When None, uses the default receive timeout (30s).

        Returns:
            Decoded MessageEnvelope, or None on EOF, timeout, or error.
        """
        if self._reader is None:
            return None

        effective_timeout = (
            timeout if timeout is not None else _DEFAULT_RECEIVE_TIMEOUT
        )

        try:
            header_bytes = await asyncio.wait_for(
                self._reader.readexactly(HEADER_SIZE),
                timeout=effective_timeout,
            )
        except asyncio.IncompleteReadError as exc:
            logger.debug("Receive: EOF during header read: %s", exc)
            return None
        except ConnectionResetError as exc:
            logger.debug("Receive: connection reset during header read: %s", exc)
            return None
        except asyncio.TimeoutError:
            logger.debug("Receive: timed out after %.1fs waiting for header", effective_timeout)
            return None

        try:
            payload_length = unpack_header(header_bytes)
            payload_bytes = await asyncio.wait_for(
                self._reader.readexactly(payload_length),
                timeout=effective_timeout,
            )
        except asyncio.IncompleteReadError as exc:
            logger.debug("Receive: EOF during payload read: %s", exc)
            return None
        except ConnectionResetError as exc:
            logger.debug("Receive: connection reset during payload read: %s", exc)
            return None
        except asyncio.TimeoutError:
            logger.debug("Receive: timed out after %.1fs waiting for payload", effective_timeout)
            return None

        try:
            return decode_envelope(payload_bytes)
        except (ValueError, KeyError) as exc:
            logger.warning("Malformed envelope from daemon: %s", exc)
            return None

    # -- Public API: close --

    async def close(self) -> None:
        """Close the connection to the daemon.

        Safely closes the underlying StreamWriter and transitions to
        CLOSED state. Idempotent: calling on an already-closed connection
        is a safe no-op.
        """
        if self._state == ConnectionState.CLOSED:
            return

        await self._close_writer()
        self._reader = None
        self._writer = None
        self._state = ConnectionState.CLOSED
        logger.debug("Connection closed")

    # -- Internal: socket path resolution --

    def _resolve_socket_path(self) -> Path | None:
        """Resolve the socket path from config or auto-discovery.

        Returns:
            The resolved socket path, or None if resolution failed.
        """
        if self._config.socket_path is not None:
            return self._config.socket_path

        try:
            return default_socket_path()
        except RuntimeError as exc:
            logger.error("Socket path discovery failed: %s", exc)
            return None

    # -- Internal: handshake --

    async def _perform_handshake(
        self,
        reader: asyncio.StreamReader,
        writer: StreamWriterLike,
    ) -> HandshakeResult:
        """Execute the protocol handshake with the daemon.

        Sends a handshake REQUEST with the client's protocol version
        and PID, then reads the daemon's RESPONSE. Validates the
        response and returns a structured result.

        Args:
            reader: StreamReader for reading the daemon's response.
            writer: StreamWriter for sending the request. Accepts any
                object satisfying the ``StreamWriterLike`` protocol.

        Returns:
            HandshakeResult describing the handshake outcome.
        """
        # Send handshake request
        request = _build_handshake_request(self._config.protocol_version)
        try:
            frame = encode_frame(request)
            writer.write(frame)
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            return _failure_result(f"Failed to send handshake: {exc}")

        logger.debug(
            "Handshake request sent (msg_id=%s, proto=v%d)",
            request.msg_id,
            self._config.protocol_version,
        )

        # Read handshake response
        response = await self._read_envelope_with_timeout(
            reader,
            timeout=self._config.handshake_timeout,
        )

        if response is None:
            return _failure_result(
                "Handshake timed out or connection lost before "
                "daemon responded"
            )

        # Parse and validate the response
        return _parse_handshake_response(
            response,
            expected_version=self._config.protocol_version,
        )

    async def _read_envelope_with_timeout(
        self,
        reader: asyncio.StreamReader,
        *,
        timeout: float,
    ) -> MessageEnvelope | None:
        """Read one framed envelope from the stream with a timeout.

        Returns None on EOF, incomplete data, decode error, or timeout.

        Args:
            reader: The StreamReader to read from.
            timeout: Maximum seconds to wait for the envelope.

        Returns:
            Decoded MessageEnvelope, or None on any failure.
        """
        try:
            header_bytes = await asyncio.wait_for(
                reader.readexactly(HEADER_SIZE),
                timeout=timeout,
            )
        except asyncio.IncompleteReadError as exc:
            logger.debug("Handshake read: EOF during header: %s", exc)
            return None
        except ConnectionResetError as exc:
            logger.debug("Handshake read: connection reset during header: %s", exc)
            return None
        except asyncio.TimeoutError:
            logger.debug("Handshake read: timed out after %.1fs", timeout)
            return None

        try:
            payload_length = unpack_header(header_bytes)
            payload_bytes = await asyncio.wait_for(
                reader.readexactly(payload_length),
                timeout=timeout,
            )
        except asyncio.IncompleteReadError as exc:
            logger.debug("Handshake read: EOF during payload: %s", exc)
            return None
        except ConnectionResetError as exc:
            logger.debug("Handshake read: connection reset during payload: %s", exc)
            return None
        except asyncio.TimeoutError:
            logger.debug("Handshake read: timed out after %.1fs during payload", timeout)
            return None

        try:
            return decode_envelope(payload_bytes)
        except (ValueError, KeyError) as exc:
            logger.warning("Malformed handshake response: %s", exc)
            return None

    # -- Internal: writer cleanup --

    async def _close_writer(self) -> None:
        """Safely close the underlying StreamWriter.

        Handles the case where the writer is already closing or the
        underlying transport has been lost.
        """
        if self._writer is None:
            return

        try:
            if not self._writer.is_closing():
                self._writer.close()
                await self._writer.wait_closed()
        except (OSError, ConnectionResetError) as exc:
            logger.debug("Error closing writer: %s", exc)
