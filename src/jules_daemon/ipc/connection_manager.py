"""Async connection lifecycle manager with client registry and event emission.

Manages the registry of connected CLI clients and emits lifecycle events
(connect/disconnect) via an optional async event bus. This module sits
between the low-level socket server and the higher-level command dispatcher,
providing a clean abstraction for tracking who is connected.

Key responsibilities:

- **Client registry**: Add, remove, and lookup connected clients by ID.
  Returns immutable ``ClientInfo`` snapshots.
- **Event emission**: On add, emits ``CLIENT_CONNECTED_EVENT``. On remove,
  emits ``CLIENT_DISCONNECTED_EVENT``. Event payloads include the client
  metadata for downstream consumers (e.g., wiki persistence, audit logging).
- **Concurrency safety**: Uses an asyncio Lock to serialize mutations,
  ensuring consistent state under concurrent ``add_client`` / ``remove_client``
  calls from the socket server's accept loop.

The event bus is optional. When no bus is provided, the manager operates
as a pure registry without event emission.

Architecture::

    SocketServer._on_client_connected()
        |
        v
    ConnectionManager.add_client(info)
        |---> registry update
        |---> bus.emit(CLIENT_CONNECTED_EVENT)
        v
    ConnectionManager.remove_client(client_id)
        |---> registry update
        |---> bus.emit(CLIENT_DISCONNECTED_EVENT)

Usage::

    from jules_daemon.ipc.connection_manager import ConnectionManager, ClientInfo
    from jules_daemon.ipc.event_bus import EventBus

    bus = EventBus()
    manager = ConnectionManager(event_bus=bus)

    info = ClientInfo(client_id="client-abc123", connected_at="2026-04-09T12:00:00Z")
    await manager.add_client(info)

    client = manager.lookup("client-abc123")
    all_clients = manager.list_clients()

    await manager.remove_client("client-abc123")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from jules_daemon.ipc.event_bus import Event, EventBus

__all__ = [
    "CLIENT_CONNECTED_EVENT",
    "CLIENT_DISCONNECTED_EVENT",
    "ClientInfo",
    "ConnectionManager",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

CLIENT_CONNECTED_EVENT: str = "client_connected"
"""Event type emitted when a new client is registered."""

CLIENT_DISCONNECTED_EVENT: str = "client_disconnected"
"""Event type emitted when a client is deregistered."""


# ---------------------------------------------------------------------------
# ClientInfo dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientInfo:
    """Immutable descriptor for a connected CLI client.

    This is a transport-agnostic representation of a client connection.
    It does NOT hold stream reader/writer references -- those belong to
    the socket server layer. This class captures only the metadata needed
    for registry tracking and event payloads.

    Attributes:
        client_id:    Unique identifier assigned on connection.
        connected_at: ISO 8601 timestamp of when the client connected.
    """

    client_id: str
    connected_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.client_id, str) or not self.client_id.strip():
            raise ValueError("client_id must not be empty")
        if not isinstance(self.connected_at, str) or not self.connected_at.strip():
            raise ValueError("connected_at must not be empty")

    def to_event_payload(self) -> dict[str, Any]:
        """Serialize to a dict suitable for event bus payloads.

        Returns:
            Dict with client_id and connected_at fields.
        """
        return {
            "client_id": self.client_id,
            "connected_at": self.connected_at,
        }


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Async connection lifecycle manager with client registry.

    Maintains an in-memory registry of connected clients, keyed by
    ``client_id``. Emits lifecycle events through an optional event bus
    when clients connect or disconnect.

    Concurrency: Uses an asyncio.Lock to serialize add/remove operations.
    Read-only operations (lookup, list_clients, has_client, client_count)
    do not acquire the lock and are safe to call from any coroutine.

    Args:
        event_bus: Optional EventBus for emitting connect/disconnect events.
            When None, the manager operates as a pure registry.
    """

    def __init__(self, *, event_bus: EventBus | None = None) -> None:
        self._event_bus = event_bus
        self._clients: dict[str, ClientInfo] = {}
        self._lock = asyncio.Lock()

    # -- Properties --

    @property
    def client_count(self) -> int:
        """Number of currently registered clients."""
        return len(self._clients)

    # -- Registry operations --

    async def add_client(self, info: ClientInfo) -> None:
        """Register a new client connection.

        Adds the client to the registry and emits a ``CLIENT_CONNECTED_EVENT``
        through the event bus (if configured).

        Args:
            info: Immutable client descriptor to register.

        Raises:
            ValueError: If a client with the same ``client_id`` is already
                registered.
        """
        async with self._lock:
            if info.client_id in self._clients:
                raise ValueError(
                    f"Client {info.client_id!r} is already registered"
                )
            self._clients = {**self._clients, info.client_id: info}

        logger.info("Client registered: %s", info.client_id)

        if self._event_bus is not None:
            await self._event_bus.emit(
                Event(
                    event_type=CLIENT_CONNECTED_EVENT,
                    payload=info.to_event_payload(),
                )
            )

    async def remove_client(self, client_id: str) -> ClientInfo | None:
        """Deregister a client connection.

        Removes the client from the registry and emits a
        ``CLIENT_DISCONNECTED_EVENT`` through the event bus (if configured).

        If the client is not found, this is a safe no-op that returns None
        and does not emit an event.

        Args:
            client_id: The unique identifier of the client to remove.

        Returns:
            The removed ClientInfo, or None if not found.
        """
        removed: ClientInfo | None = None

        async with self._lock:
            if client_id in self._clients:
                removed = self._clients[client_id]
                # Create new dict without the removed client (immutable pattern)
                self._clients = {
                    cid: info
                    for cid, info in self._clients.items()
                    if cid != client_id
                }

        if removed is None:
            logger.debug(
                "Attempted to remove nonexistent client: %s", client_id
            )
            return None

        logger.info("Client deregistered: %s", client_id)

        if self._event_bus is not None:
            await self._event_bus.emit(
                Event(
                    event_type=CLIENT_DISCONNECTED_EVENT,
                    payload=removed.to_event_payload(),
                )
            )

        return removed

    def lookup(self, client_id: str) -> ClientInfo | None:
        """Look up a client by its unique identifier.

        Args:
            client_id: The client ID to search for.

        Returns:
            The matching ClientInfo, or None if not found.
        """
        return self._clients.get(client_id)

    def has_client(self, client_id: str) -> bool:
        """Check whether a client is currently registered.

        Args:
            client_id: The client ID to check.

        Returns:
            True if the client is in the registry.
        """
        return client_id in self._clients

    def list_clients(self) -> tuple[ClientInfo, ...]:
        """Return an immutable snapshot of all registered clients.

        Returns:
            Tuple of ClientInfo objects. Empty tuple if no clients
            are registered.
        """
        return tuple(self._clients.values())
