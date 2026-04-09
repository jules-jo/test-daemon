"""Async Unix domain socket server for CLI-daemon IPC.

Provides a single-user IPC server that listens on a Unix domain socket
and exchanges length-prefixed JSON messages (using the framing module)
with CLI clients. The server supports:

- Async bind/listen/accept loop via ``asyncio.start_unix_server``
- Concurrent client handling via ``ConnectionDispatcher``
- Graceful shutdown with timeout for in-flight requests
- Socket file cleanup on both normal and abnormal exits
- Stale socket file removal on startup (crash recovery)
- Restrictive socket file permissions (owner-only)
- Handler errors propagated as ERROR envelopes (never crash the server)

Architecture:

    CLI Client  <--- Unix domain socket --->  SocketServer
                     (framed messages)          |
                                                v
                                          ConnectionDispatcher
                                                |
                                                v (explicit asyncio.Task per client)
                                          ClientHandler.handle_message()
                                                |
                                                v
                                          MessageEnvelope (response)

The server does NOT own business logic. It delegates to a ``ClientHandler``
protocol implementor that processes each ``MessageEnvelope`` and returns a
response envelope. The ``ConnectionDispatcher`` manages explicit task
spawning, concurrency limits, and ConnectionManager integration.

Usage::

    from pathlib import Path
    from jules_daemon.ipc.server import ServerConfig, SocketServer

    class MyHandler:
        async def handle_message(self, envelope, client):
            # process the request and return a response envelope
            ...

    config = ServerConfig(socket_path=Path("/tmp/jules.sock"))
    async with SocketServer(config=config, handler=MyHandler()) as server:
        # server is now accepting connections
        await some_shutdown_signal.wait()
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from jules_daemon.ipc.framing import MessageEnvelope

if TYPE_CHECKING:
    from jules_daemon.ipc.connection_dispatcher import ConnectionDispatcher

__all__ = [
    "DEFAULT_BACKLOG",
    "DEFAULT_SHUTDOWN_TIMEOUT_SECONDS",
    "MAX_CONCURRENT_CLIENTS",
    "ClientConnection",
    "ClientHandler",
    "ServerConfig",
    "ServerState",
    "SocketServer",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BACKLOG: int = 5
"""Default socket listen backlog (pending connection queue size)."""

DEFAULT_SHUTDOWN_TIMEOUT_SECONDS: float = 10.0
"""Default maximum seconds to wait for in-flight requests during shutdown."""

MAX_CONCURRENT_CLIENTS: int = 10
"""Default maximum number of concurrent client connections."""

_SOCKET_PERMISSIONS: int = 0o600
"""File permissions for the socket (owner read/write only)."""


# ---------------------------------------------------------------------------
# ServerState enum
# ---------------------------------------------------------------------------


class ServerState(Enum):
    """Lifecycle state of the socket server.

    Values:
        STOPPED:       Server is not running. Initial and final state.
        STARTING:      Server is binding the socket and preparing to accept.
        RUNNING:       Server is accepting client connections.
        SHUTTING_DOWN: Graceful shutdown in progress. No new connections
                       accepted; in-flight requests are being drained.
    """

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"


# ---------------------------------------------------------------------------
# ClientConnection dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientConnection:
    """Immutable descriptor for a connected CLI client.

    Attributes:
        client_id:    Unique identifier assigned on connection.
        reader:       asyncio StreamReader for this client.
        writer:       asyncio StreamWriter for this client.
        connected_at: ISO 8601 timestamp of connection time.
    """

    client_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    connected_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.client_id, str) or not self.client_id.strip():
            raise ValueError("client_id must not be empty")


# ---------------------------------------------------------------------------
# ClientHandler protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ClientHandler(Protocol):
    """Protocol for message handlers that process client requests.

    Implementors receive a deserialized ``MessageEnvelope`` and the
    ``ClientConnection`` context, and return a response envelope. The
    server catches any exceptions and converts them to ERROR envelopes.
    """

    async def handle_message(
        self,
        envelope: MessageEnvelope,
        client: ClientConnection,
    ) -> MessageEnvelope:
        """Process a client request and return a response.

        Args:
            envelope: The deserialized request envelope.
            client: Connection context for the requesting client.

        Returns:
            A response MessageEnvelope to send back to the client.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# ServerConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServerConfig:
    """Immutable configuration for the Unix domain socket server.

    Attributes:
        socket_path:             Path to the Unix domain socket file.
        backlog:                 Listen backlog (pending connection queue).
        shutdown_timeout_seconds: Max seconds to wait during graceful shutdown.
        max_concurrent_clients:  Maximum simultaneous client connections.
    """

    socket_path: Path
    backlog: int = DEFAULT_BACKLOG
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
    max_concurrent_clients: int = MAX_CONCURRENT_CLIENTS

    def __post_init__(self) -> None:
        if self.backlog <= 0:
            raise ValueError(
                f"backlog must be positive, got {self.backlog}"
            )
        if self.shutdown_timeout_seconds <= 0:
            raise ValueError(
                f"shutdown_timeout_seconds must be positive, "
                f"got {self.shutdown_timeout_seconds}"
            )
        if self.max_concurrent_clients <= 0:
            raise ValueError(
                f"max_concurrent_clients must be positive, "
                f"got {self.max_concurrent_clients}"
            )


# ---------------------------------------------------------------------------
# SocketServer
# ---------------------------------------------------------------------------


class SocketServer:
    """Async Unix domain socket server for CLI-daemon IPC.

    Manages the full socket lifecycle: bind, listen, accept, and
    graceful shutdown with socket file cleanup. Connection handling
    is delegated to a ``ConnectionDispatcher`` that explicitly spawns
    an asyncio task per client, enforces concurrency limits, and
    integrates with ConnectionManager for lifecycle events.

    The server delegates message handling to a ``ClientHandler`` protocol
    implementor, keeping transport concerns separate from business logic.

    Usage::

        config = ServerConfig(socket_path=Path("/tmp/jules.sock"))
        handler = MyHandler()

        # Option 1: explicit start/shutdown
        server = SocketServer(config=config, handler=handler)
        await server.start()
        # ... serve ...
        await server.shutdown()

        # Option 2: async context manager
        async with SocketServer(config=config, handler=handler) as srv:
            # ... serve ...
            pass
    """

    def __init__(
        self,
        *,
        config: ServerConfig,
        handler: ClientHandler,
    ) -> None:
        # Import here to avoid circular import at module level
        from jules_daemon.ipc.connection_dispatcher import (
            ConnectionDispatcher,
            DispatcherConfig,
        )

        self._config = config
        self._handler = handler
        self._state = ServerState.STOPPED
        self._server: asyncio.Server | None = None

        # Create the connection dispatcher
        dispatcher_config = DispatcherConfig(
            handler=handler,
            max_concurrent_clients=config.max_concurrent_clients,
        )
        self._dispatcher = ConnectionDispatcher(config=dispatcher_config)

    # -- Properties --

    @property
    def state(self) -> ServerState:
        """Current lifecycle state of the server."""
        return self._state

    @property
    def active_client_count(self) -> int:
        """Number of currently active client connection tasks."""
        return self._dispatcher.active_connection_count

    @property
    def dispatcher(self) -> ConnectionDispatcher:
        """The ConnectionDispatcher managing client connections.

        Returns the dispatcher instance for introspection and testing.
        The type annotation uses a forward reference resolved via
        ``TYPE_CHECKING`` to avoid circular imports at runtime.
        """
        return self._dispatcher

    # -- Async context manager --

    async def __aenter__(self) -> SocketServer:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.shutdown()

    # -- Lifecycle --

    async def start(self) -> None:
        """Bind to the Unix domain socket and begin accepting connections.

        Creates parent directories and removes stale socket files if
        necessary. Sets restrictive permissions on the socket file.

        Raises:
            RuntimeError: If the server is already running.
            OSError: If the socket cannot be bound.
        """
        if self._state in (ServerState.RUNNING, ServerState.STARTING):
            raise RuntimeError(
                f"Server is already running (state={self._state.value})"
            )

        self._state = ServerState.STARTING
        socket_path = self._config.socket_path

        try:
            # Ensure parent directory exists
            socket_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove stale socket file from previous crash
            if socket_path.exists():
                logger.warning(
                    "Removing stale socket file: %s", socket_path
                )
                socket_path.unlink()

            # Start the async server -- Unix socket on Linux/macOS,
            # TCP localhost on Windows (which lacks Unix sockets)
            import sys

            if sys.platform == "win32":
                # Windows: use TCP on localhost with a port derived from
                # the socket path hash to avoid collisions
                port = 49152 + (hash(str(socket_path)) % 16383)
                self._server = await asyncio.start_server(
                    self._on_client_connected,
                    host="127.0.0.1",
                    port=port,
                    backlog=self._config.backlog,
                )
                # Write the port to the socket_path file so clients
                # can discover it
                socket_path.parent.mkdir(parents=True, exist_ok=True)
                socket_path.write_text(str(port))
                self._win_port = port
            else:
                self._server = await asyncio.start_unix_server(
                    self._on_client_connected,
                    path=str(socket_path),
                    backlog=self._config.backlog,
                )
                # Set restrictive permissions (owner read/write only)
                os.chmod(socket_path, _SOCKET_PERMISSIONS)

            self._state = ServerState.RUNNING
            logger.info(
                "IPC server listening on %s (backlog=%d, max_clients=%d)",
                socket_path,
                self._config.backlog,
                self._config.max_concurrent_clients,
            )

        except Exception:
            self._state = ServerState.STOPPED
            # Clean up socket file if it was partially created
            if socket_path.exists():
                try:
                    socket_path.unlink()
                except OSError:
                    pass
            raise

    async def shutdown(self) -> None:
        """Gracefully shut down the server.

        1. Stop accepting new connections.
        2. Drain the ConnectionDispatcher (wait for in-flight tasks).
        3. Remove the socket file.

        This method is idempotent: calling it on an already-stopped
        server is a safe no-op.
        """
        if self._state == ServerState.STOPPED:
            return

        self._state = ServerState.SHUTTING_DOWN
        logger.info("IPC server shutting down...")

        # 1. Stop accepting new connections
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # 2. Drain the dispatcher (waits for in-flight, cancels stragglers)
        await self._dispatcher.drain(
            timeout=self._config.shutdown_timeout_seconds
        )

        # 3. Remove the socket file
        socket_path = self._config.socket_path
        if socket_path.exists():
            try:
                socket_path.unlink()
                logger.info("Removed socket file: %s", socket_path)
            except OSError as exc:
                logger.warning(
                    "Failed to remove socket file %s: %s",
                    socket_path,
                    exc,
                )

        self._state = ServerState.STOPPED
        logger.info("IPC server stopped")

    # -- Client connection callback --

    async def _on_client_connected(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Callback invoked by asyncio when a new client connects.

        Delegates immediately to the ConnectionDispatcher, which spawns
        an explicit task for the connection. This keeps the accept loop
        lightweight and non-blocking.
        """
        accepted = await self._dispatcher.dispatch(reader, writer)
        if not accepted:
            logger.debug("Connection rejected (server shutting down)")
