"""Concurrent connection dispatcher for the IPC server.

Manages the lifecycle of multiple concurrent client connections, replacing
the implicit task tracking previously embedded in the server's accept
callback. This module provides explicit task spawning, concurrency
limiting, and ConnectionManager integration.

Key responsibilities:

- **Explicit task spawning**: Each accepted connection gets its own
  ``asyncio.create_task`` call with a descriptive name, replacing the
  fragile ``asyncio.current_task()`` pattern.
- **Concurrency limiting**: An asyncio.Semaphore enforces the maximum
  number of concurrent client connections. Excess connections wait for
  a slot rather than being rejected.
- **Task tracking**: Maintains a set of active tasks for graceful drain
  on shutdown. The ``drain()`` method waits for in-flight tasks up to
  a timeout, then cancels stragglers.
- **ConnectionManager integration**: Optionally registers/deregisters
  clients with a ConnectionManager for lifecycle event emission
  (CLIENT_CONNECTED / CLIENT_DISCONNECTED events).
- **Error isolation**: Handler errors are caught per-connection and
  returned as ERROR envelopes. One bad client never crashes others.

Architecture::

    asyncio.start_unix_server callback
        |
        v
    ConnectionDispatcher.dispatch(reader, writer)
        |
        +-- creates ClientConnection
        +-- spawns asyncio.Task via asyncio.create_task
        +-- registers with ConnectionManager (optional)
        |
        v
    Task: _handle_connection_lifecycle()
        |
        +-- acquires semaphore (concurrency gate)
        +-- reads framed messages in a loop
        +-- dispatches each to ClientHandler
        +-- on EOF/error: releases semaphore, deregisters, closes writer

Usage::

    from jules_daemon.ipc.connection_dispatcher import (
        ConnectionDispatcher,
        DispatcherConfig,
    )
    from jules_daemon.ipc.connection_manager import ConnectionManager

    manager = ConnectionManager(event_bus=bus)
    config = DispatcherConfig(
        handler=my_handler,
        max_concurrent_clients=10,
        connection_manager=manager,
    )
    dispatcher = ConnectionDispatcher(config=config)

    # In the server's accept callback:
    await dispatcher.dispatch(reader, writer)

    # On shutdown:
    await dispatcher.drain(timeout=10.0)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from jules_daemon.ipc.connection_manager import ClientInfo, ConnectionManager
from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
    unpack_header,
)
from jules_daemon.ipc.server import ClientConnection, ClientHandler

__all__ = [
    "ConnectionDispatcher",
    "DispatcherConfig",
    "DispatcherState",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_CONCURRENT_CLIENTS: int = 10
"""Default maximum number of concurrent client connections."""


# ---------------------------------------------------------------------------
# DispatcherState enum
# ---------------------------------------------------------------------------


class DispatcherState(Enum):
    """Lifecycle state of the connection dispatcher.

    Values:
        READY:    Dispatcher is created but has not dispatched any connections.
        ACTIVE:   Dispatcher has at least one active or completed connection.
        DRAINING: Drain has been called; no new connections accepted.
    """

    READY = "ready"
    ACTIVE = "active"
    DRAINING = "draining"


# ---------------------------------------------------------------------------
# DispatcherConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatcherConfig:
    """Immutable configuration for the connection dispatcher.

    Attributes:
        handler:                The ClientHandler that processes messages.
        max_concurrent_clients: Maximum simultaneous client connections.
        connection_manager:     Optional ConnectionManager for lifecycle
                                event emission.
    """

    handler: ClientHandler
    max_concurrent_clients: int = _DEFAULT_MAX_CONCURRENT_CLIENTS
    connection_manager: ConnectionManager | None = None

    def __post_init__(self) -> None:
        if self.max_concurrent_clients <= 0:
            raise ValueError(
                f"max_concurrent_clients must be positive, "
                f"got {self.max_concurrent_clients}"
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _generate_client_id() -> str:
    """Generate a unique client connection identifier."""
    return f"client-{uuid.uuid4().hex[:12]}"


def _build_error_envelope(
    msg_id: str,
    error_message: str,
) -> MessageEnvelope:
    """Build an ERROR envelope for returning errors to clients.

    Args:
        msg_id: The original request's msg_id for correlation.
        error_message: Human-readable error description.

    Returns:
        A MessageEnvelope with MessageType.ERROR.
    """
    return MessageEnvelope(
        msg_type=MessageType.ERROR,
        msg_id=msg_id,
        timestamp=_now_iso(),
        payload={"error": error_message},
    )


# ---------------------------------------------------------------------------
# ConnectionDispatcher
# ---------------------------------------------------------------------------


class ConnectionDispatcher:
    """Concurrent connection dispatcher for the IPC server.

    Accepts ``(reader, writer)`` pairs from the server's accept loop and
    spawns an explicit asyncio task for each one. Tasks are tracked for
    graceful shutdown via ``drain()``. Concurrency is limited by an
    asyncio.Semaphore.

    The dispatcher delegates message handling to a ``ClientHandler``
    protocol implementor, keeping connection lifecycle management
    separate from business logic.

    Args:
        config: Dispatcher configuration (handler, limits, manager).
    """

    def __init__(self, *, config: DispatcherConfig) -> None:
        self._config = config
        self._state = DispatcherState.READY
        self._tasks: set[asyncio.Task[None]] = set()
        self._semaphore = asyncio.Semaphore(config.max_concurrent_clients)

    # -- Properties --

    @property
    def state(self) -> DispatcherState:
        """Current lifecycle state of the dispatcher."""
        return self._state

    @property
    def active_connection_count(self) -> int:
        """Number of currently active client connection tasks."""
        return len(self._tasks)

    # -- Public API: dispatch --

    async def dispatch(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> bool:
        """Accept a new client connection and spawn a handler task.

        Creates a ``ClientConnection`` descriptor, spawns an asyncio task
        to handle the connection lifecycle, and tracks the task for
        graceful shutdown.

        Args:
            reader: asyncio StreamReader for the new client.
            writer: asyncio StreamWriter for the new client.

        Returns:
            True if the connection was accepted and dispatched.
            False if the dispatcher is draining (no new connections).
        """
        if self._state == DispatcherState.DRAINING:
            logger.debug("Rejecting connection: dispatcher is draining")
            await self._close_writer_safe(writer, "rejected")
            return False

        client_id = _generate_client_id()
        connected_at = _now_iso()

        client = ClientConnection(
            client_id=client_id,
            reader=reader,
            writer=writer,
            connected_at=connected_at,
        )

        logger.debug("Dispatching client: %s", client_id)

        # Spawn an explicit task for this connection
        task = asyncio.create_task(
            self._handle_connection_lifecycle(client),
            name=f"dispatch-{client_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)

        # Transition to ACTIVE on first dispatch
        if self._state == DispatcherState.READY:
            self._state = DispatcherState.ACTIVE

        return True

    # -- Public API: drain --

    async def drain(self, *, timeout: float = 10.0) -> None:
        """Gracefully drain all active connections.

        1. Marks the dispatcher as DRAINING (no new connections accepted).
        2. Waits for all in-flight tasks up to the given timeout.
        3. Cancels any tasks that did not complete in time.

        This method is idempotent: calling it multiple times is safe.

        Args:
            timeout: Maximum seconds to wait for in-flight tasks.
        """
        self._state = DispatcherState.DRAINING

        if not self._tasks:
            return

        logger.info(
            "Draining %d active connection(s) (timeout=%.1fs)...",
            len(self._tasks),
            timeout,
        )

        # Snapshot the task set to avoid mutation during iteration
        _, pending = await asyncio.wait(
            set(self._tasks),
            timeout=timeout,
        )

        # Cancel tasks that did not complete in time
        for task in pending:
            task.cancel()

        if pending:
            logger.warning(
                "Cancelled %d connection(s) that exceeded drain timeout",
                len(pending),
            )
            # Wait for cancellations to propagate
            await asyncio.gather(*pending, return_exceptions=True)

        self._tasks.clear()

    # -- Internal: task done callback --

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        """Remove a completed task from the tracking set.

        This is registered as a done callback on each spawned task.
        It runs synchronously in the event loop thread.
        """
        self._tasks.discard(task)

        # Log unhandled exceptions (should not happen due to our try/except)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.error(
                    "Unhandled exception in connection task %s: %s",
                    task.get_name(),
                    exc,
                )

    # -- Internal: connection lifecycle --

    async def _handle_connection_lifecycle(
        self,
        client: ClientConnection,
    ) -> None:
        """Full lifecycle for a single client connection.

        1. Registers with ConnectionManager (if configured).
        2. Acquires the concurrency semaphore.
        3. Reads framed messages in a loop, dispatching each to the handler.
        4. On EOF/error: releases semaphore, deregisters, closes writer.

        This method catches all exceptions to ensure cleanup always runs.
        """
        client_id = client.client_id

        # Register with ConnectionManager
        await self._register_client(client)

        try:
            await self._semaphore.acquire()
            try:
                await self._handle_client_messages(client)
            finally:
                self._semaphore.release()
        except asyncio.CancelledError:
            logger.debug("Connection task cancelled: %s", client_id)
            raise
        except Exception as exc:
            logger.warning(
                "Unhandled error in connection %s: %s", client_id, exc
            )
        finally:
            # Shield cleanup from secondary cancellation to avoid
            # leaking file descriptors or orphaned ConnectionManager entries.
            try:
                await asyncio.shield(self._deregister_client(client_id))
            except asyncio.CancelledError:
                pass
            try:
                await asyncio.shield(
                    self._close_writer_safe(client.writer, client_id)
                )
            except asyncio.CancelledError:
                pass
            logger.debug("Connection lifecycle ended: %s", client_id)

    async def _handle_client_messages(
        self,
        client: ClientConnection,
    ) -> None:
        """Read and process framed messages from a client.

        Reads one framed message at a time in a loop. Each message is
        deserialized, passed to the handler, and the response is sent
        back. The loop exits on EOF (client disconnect) or malformed data.
        """
        while True:
            # Read the 4-byte length header
            try:
                header_bytes = await client.reader.readexactly(HEADER_SIZE)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                # Client disconnected cleanly or abruptly
                break

            # Decode header and read payload
            try:
                payload_length = unpack_header(header_bytes)
                payload_bytes = await client.reader.readexactly(
                    payload_length
                )
                envelope = decode_envelope(payload_bytes)
            except asyncio.IncompleteReadError:
                # Client disconnected mid-message
                break
            except (ValueError, KeyError) as exc:
                # Malformed frame: send error and close
                logger.warning(
                    "Malformed message from %s: %s",
                    client.client_id,
                    exc,
                )
                error_envelope = _build_error_envelope(
                    msg_id="unknown",
                    error_message=f"Malformed message: {exc}",
                )
                await self._send_envelope_safe(
                    client.writer, error_envelope, client.client_id
                )
                break

            # Dispatch to handler
            response = await self._dispatch_to_handler(envelope, client)

            # Send response
            sent = await self._send_envelope_safe(
                client.writer, response, client.client_id
            )
            if not sent:
                break

    async def _dispatch_to_handler(
        self,
        envelope: MessageEnvelope,
        client: ClientConnection,
    ) -> MessageEnvelope:
        """Call the handler and catch any exceptions.

        Returns the handler's response envelope on success, or an
        ERROR envelope if the handler raises.
        """
        try:
            return await self._config.handler.handle_message(envelope, client)
        except Exception as exc:
            logger.warning(
                "Handler error for msg_id=%s from %s: %s",
                envelope.msg_id,
                client.client_id,
                exc,
            )
            return _build_error_envelope(
                msg_id=envelope.msg_id,
                error_message=str(exc),
            )

    # -- Internal: ConnectionManager integration --

    async def _register_client(self, client: ClientConnection) -> None:
        """Register a client with the ConnectionManager if configured."""
        manager = self._config.connection_manager
        if manager is None:
            return

        try:
            client_info = ClientInfo(
                client_id=client.client_id,
                connected_at=client.connected_at,
            )
            await manager.add_client(client_info)
            logger.debug(
                "Registered client %s with ConnectionManager",
                client.client_id,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "Failed to register client %s: %s",
                client.client_id,
                exc,
                exc_info=True,
            )

    async def _deregister_client(self, client_id: str) -> None:
        """Deregister a client from the ConnectionManager if configured."""
        manager = self._config.connection_manager
        if manager is None:
            return

        try:
            await manager.remove_client(client_id)
            logger.debug(
                "Deregistered client %s from ConnectionManager",
                client_id,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "Failed to deregister client %s: %s",
                client_id,
                exc,
                exc_info=True,
            )

    # -- Internal: safe IO helpers --

    @staticmethod
    async def _send_envelope_safe(
        writer: asyncio.StreamWriter,
        envelope: MessageEnvelope,
        client_id: str,
    ) -> bool:
        """Encode and send a framed envelope, returning success status.

        Args:
            writer: The client's StreamWriter.
            envelope: The response envelope to send.
            client_id: Client identifier for logging.

        Returns:
            True if the send succeeded, False on connection error.
        """
        try:
            frame = encode_frame(envelope)
            writer.write(frame)
            await writer.drain()
            return True
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            logger.debug(
                "Client %s disconnected before response sent: %s",
                client_id,
                exc,
            )
            return False

    @staticmethod
    async def _close_writer_safe(
        writer: asyncio.StreamWriter,
        client_id: str,
    ) -> None:
        """Safely close a client's StreamWriter.

        Handles the case where the writer is already closed or the
        underlying transport has been lost.
        """
        try:
            if not writer.is_closing():
                writer.close()
                await writer.wait_closed()
        except (OSError, ConnectionResetError, BrokenPipeError) as exc:
            logger.debug(
                "Error closing writer for %s: %s", client_id, exc
            )
