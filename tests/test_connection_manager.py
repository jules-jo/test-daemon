"""Tests for the async connection lifecycle manager.

Covers:
    - Client registration (add) and deregistration (remove)
    - Client lookup by ID
    - Listing all connected clients
    - Connect event emission on client add
    - Disconnect event emission on client remove
    - Duplicate client_id rejection
    - Remove nonexistent client is safe no-op
    - Lookup nonexistent client returns None
    - Client count tracking
    - Immutability of returned client info
    - Event payloads contain correct client metadata
    - Manager is usable without event bus (events optional)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from jules_daemon.ipc.connection_manager import (
    CLIENT_CONNECTED_EVENT,
    CLIENT_DISCONNECTED_EVENT,
    ClientInfo,
    ConnectionManager,
)
from jules_daemon.ipc.event_bus import Event, EventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client_info(
    client_id: str = "client-001",
    connected_at: str = "2026-04-09T12:00:00Z",
) -> ClientInfo:
    """Build a ClientInfo for testing."""
    return ClientInfo(
        client_id=client_id,
        connected_at=connected_at,
    )


# ---------------------------------------------------------------------------
# ClientInfo model tests
# ---------------------------------------------------------------------------


class TestClientInfo:
    """Tests for the immutable ClientInfo dataclass."""

    def test_create(self) -> None:
        info = _make_client_info()
        assert info.client_id == "client-001"
        assert info.connected_at == "2026-04-09T12:00:00Z"

    def test_frozen(self) -> None:
        info = _make_client_info()
        with pytest.raises(AttributeError):
            info.client_id = "mutated"  # type: ignore[misc]

    def test_empty_client_id_raises(self) -> None:
        with pytest.raises(ValueError, match="client_id must not be empty"):
            ClientInfo(client_id="", connected_at="2026-04-09T12:00:00Z")

    def test_whitespace_client_id_raises(self) -> None:
        with pytest.raises(ValueError, match="client_id must not be empty"):
            ClientInfo(client_id="   ", connected_at="2026-04-09T12:00:00Z")

    def test_empty_connected_at_raises(self) -> None:
        with pytest.raises(ValueError, match="connected_at must not be empty"):
            ClientInfo(client_id="client-001", connected_at="")


# ---------------------------------------------------------------------------
# ConnectionManager registration tests
# ---------------------------------------------------------------------------


class TestConnectionManagerRegistration:
    """Tests for client add/remove/lookup operations."""

    @pytest.mark.asyncio
    async def test_add_client(self) -> None:
        manager = ConnectionManager()
        info = _make_client_info()
        await manager.add_client(info)
        assert manager.client_count == 1

    @pytest.mark.asyncio
    async def test_add_and_lookup_client(self) -> None:
        manager = ConnectionManager()
        info = _make_client_info(client_id="c1")
        await manager.add_client(info)

        found = manager.lookup("c1")
        assert found is not None
        assert found.client_id == "c1"

    @pytest.mark.asyncio
    async def test_lookup_nonexistent_returns_none(self) -> None:
        manager = ConnectionManager()
        assert manager.lookup("no-such-id") is None

    @pytest.mark.asyncio
    async def test_remove_client(self) -> None:
        manager = ConnectionManager()
        info = _make_client_info(client_id="c1")
        await manager.add_client(info)
        assert manager.client_count == 1

        removed = await manager.remove_client("c1")
        assert removed is not None
        assert removed.client_id == "c1"
        assert manager.client_count == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_returns_none(self) -> None:
        manager = ConnectionManager()
        removed = await manager.remove_client("no-such-id")
        assert removed is None

    @pytest.mark.asyncio
    async def test_list_clients_empty(self) -> None:
        manager = ConnectionManager()
        assert manager.list_clients() == ()

    @pytest.mark.asyncio
    async def test_list_clients_returns_all(self) -> None:
        manager = ConnectionManager()
        await manager.add_client(_make_client_info(client_id="c1"))
        await manager.add_client(_make_client_info(client_id="c2"))
        await manager.add_client(_make_client_info(client_id="c3"))

        clients = manager.list_clients()
        assert len(clients) == 3
        ids = {c.client_id for c in clients}
        assert ids == {"c1", "c2", "c3"}

    @pytest.mark.asyncio
    async def test_list_clients_returns_tuple(self) -> None:
        """list_clients returns a tuple (immutable snapshot)."""
        manager = ConnectionManager()
        await manager.add_client(_make_client_info(client_id="c1"))
        result = manager.list_clients()
        assert isinstance(result, tuple)

    @pytest.mark.asyncio
    async def test_duplicate_client_id_raises(self) -> None:
        manager = ConnectionManager()
        info = _make_client_info(client_id="c1")
        await manager.add_client(info)

        with pytest.raises(ValueError, match="already registered"):
            await manager.add_client(info)

    @pytest.mark.asyncio
    async def test_client_count_tracks_adds_and_removes(self) -> None:
        manager = ConnectionManager()
        assert manager.client_count == 0

        await manager.add_client(_make_client_info(client_id="c1"))
        assert manager.client_count == 1

        await manager.add_client(_make_client_info(client_id="c2"))
        assert manager.client_count == 2

        await manager.remove_client("c1")
        assert manager.client_count == 1

        await manager.remove_client("c2")
        assert manager.client_count == 0


# ---------------------------------------------------------------------------
# ConnectionManager event emission tests
# ---------------------------------------------------------------------------


class TestConnectionManagerEvents:
    """Tests for connect/disconnect event emission."""

    @pytest.mark.asyncio
    async def test_add_client_emits_connected_event(self) -> None:
        bus = EventBus()
        manager = ConnectionManager(event_bus=bus)

        received: list[Event] = []

        async def on_connect(event: Event) -> None:
            received.append(event)

        bus.subscribe(CLIENT_CONNECTED_EVENT, on_connect)

        info = _make_client_info(client_id="c1")
        await manager.add_client(info)

        assert len(received) == 1
        assert received[0].event_type == CLIENT_CONNECTED_EVENT
        assert received[0].payload["client_id"] == "c1"
        assert received[0].payload["connected_at"] == info.connected_at

    @pytest.mark.asyncio
    async def test_remove_client_emits_disconnected_event(self) -> None:
        bus = EventBus()
        manager = ConnectionManager(event_bus=bus)

        received: list[Event] = []

        async def on_disconnect(event: Event) -> None:
            received.append(event)

        bus.subscribe(CLIENT_DISCONNECTED_EVENT, on_disconnect)

        info = _make_client_info(client_id="c1")
        await manager.add_client(info)
        await manager.remove_client("c1")

        assert len(received) == 1
        assert received[0].event_type == CLIENT_DISCONNECTED_EVENT
        assert received[0].payload["client_id"] == "c1"

    @pytest.mark.asyncio
    async def test_remove_nonexistent_does_not_emit_event(self) -> None:
        bus = EventBus()
        manager = ConnectionManager(event_bus=bus)

        received: list[Event] = []

        async def on_disconnect(event: Event) -> None:
            received.append(event)

        bus.subscribe(CLIENT_DISCONNECTED_EVENT, on_disconnect)

        await manager.remove_client("no-such-id")
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_events_contain_timestamp(self) -> None:
        bus = EventBus()
        manager = ConnectionManager(event_bus=bus)

        received: list[Event] = []

        async def on_connect(event: Event) -> None:
            received.append(event)

        bus.subscribe(CLIENT_CONNECTED_EVENT, on_connect)

        await manager.add_client(_make_client_info(client_id="c1"))
        assert received[0].timestamp != ""

    @pytest.mark.asyncio
    async def test_no_event_bus_still_works(self) -> None:
        """Manager works without an event bus (events just not emitted)."""
        manager = ConnectionManager()
        info = _make_client_info(client_id="c1")
        await manager.add_client(info)
        assert manager.client_count == 1

        removed = await manager.remove_client("c1")
        assert removed is not None
        assert manager.client_count == 0

    @pytest.mark.asyncio
    async def test_connect_disconnect_event_sequence(self) -> None:
        """Events arrive in correct connect-then-disconnect order."""
        bus = EventBus()
        manager = ConnectionManager(event_bus=bus)

        events: list[str] = []

        async def on_connect(event: Event) -> None:
            events.append("connected")

        async def on_disconnect(event: Event) -> None:
            events.append("disconnected")

        bus.subscribe(CLIENT_CONNECTED_EVENT, on_connect)
        bus.subscribe(CLIENT_DISCONNECTED_EVENT, on_disconnect)

        await manager.add_client(_make_client_info(client_id="c1"))
        await manager.remove_client("c1")

        assert events == ["connected", "disconnected"]


# ---------------------------------------------------------------------------
# ConnectionManager concurrent access tests
# ---------------------------------------------------------------------------


class TestConnectionManagerConcurrency:
    """Tests for concurrent client registration."""

    @pytest.mark.asyncio
    async def test_add_multiple_clients_concurrently(self) -> None:
        manager = ConnectionManager()

        async def add_client(idx: int) -> None:
            info = _make_client_info(client_id=f"c{idx}")
            await manager.add_client(info)

        await asyncio.gather(*(add_client(i) for i in range(10)))
        assert manager.client_count == 10

    @pytest.mark.asyncio
    async def test_add_and_remove_concurrently(self) -> None:
        manager = ConnectionManager()

        # Pre-populate
        for i in range(5):
            await manager.add_client(_make_client_info(client_id=f"c{i}"))

        async def remove_client(idx: int) -> None:
            await manager.remove_client(f"c{idx}")

        async def add_client(idx: int) -> None:
            await manager.add_client(_make_client_info(client_id=f"new-{idx}"))

        # Remove and add concurrently
        tasks = [remove_client(i) for i in range(5)]
        tasks += [add_client(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # All old removed, all new added
        assert manager.client_count == 5
        for i in range(5):
            assert manager.lookup(f"c{i}") is None
            assert manager.lookup(f"new-{i}") is not None


# ---------------------------------------------------------------------------
# ConnectionManager has_client tests
# ---------------------------------------------------------------------------


class TestConnectionManagerHasClient:
    """Tests for the has_client convenience method."""

    @pytest.mark.asyncio
    async def test_has_client_true(self) -> None:
        manager = ConnectionManager()
        await manager.add_client(_make_client_info(client_id="c1"))
        assert manager.has_client("c1")

    @pytest.mark.asyncio
    async def test_has_client_false(self) -> None:
        manager = ConnectionManager()
        assert not manager.has_client("no-such-id")

    @pytest.mark.asyncio
    async def test_has_client_after_remove(self) -> None:
        manager = ConnectionManager()
        await manager.add_client(_make_client_info(client_id="c1"))
        await manager.remove_client("c1")
        assert not manager.has_client("c1")
