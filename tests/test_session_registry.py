"""Tests for the per-client session registry.

Covers:
    - Session creation with unique IDs and metadata
    - Session lookup by session ID
    - Session lookup by client ID
    - Enumerate active sessions
    - List all sessions (including disconnected)
    - Session state lifecycle transitions
    - Disconnect session
    - Event emission on session lifecycle changes
    - Duplicate client_id handling
    - Concurrent session creation
    - Session count tracking
    - Registry operation without event bus (optional)
"""

from __future__ import annotations

import asyncio

import pytest

from jules_daemon.ipc.event_bus import Event, EventBus
from jules_daemon.ipc.session_models import (
    ClientMetadata,
    SessionState,
)
from jules_daemon.ipc.session_registry import (
    SESSION_CREATED_EVENT,
    SESSION_DISCONNECTED_EVENT,
    SESSION_STATE_CHANGED_EVENT,
    SessionRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(
    client_name: str = "jules-cli",
    client_version: str = "0.1.0",
    client_pid: int | None = 12345,
    hostname: str = "workstation.local",
) -> ClientMetadata:
    """Build a ClientMetadata for testing."""
    return ClientMetadata(
        client_name=client_name,
        client_version=client_version,
        client_pid=client_pid,
        hostname=hostname,
        extra={},
    )


# ---------------------------------------------------------------------------
# Session creation tests
# ---------------------------------------------------------------------------


class TestSessionCreation:
    """Tests for creating sessions in the registry."""

    @pytest.mark.asyncio
    async def test_create_session_returns_record(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        record = await registry.create_session(client_id="c1", metadata=meta)
        assert record.client_id == "c1"
        assert record.state == SessionState.CREATED
        assert record.metadata == meta
        assert record.session_id  # non-empty

    @pytest.mark.asyncio
    async def test_create_session_generates_unique_ids(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        r1 = await registry.create_session(client_id="c1", metadata=meta)
        r2 = await registry.create_session(client_id="c2", metadata=meta)
        assert r1.session_id != r2.session_id

    @pytest.mark.asyncio
    async def test_create_session_populates_timestamps(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        record = await registry.create_session(client_id="c1", metadata=meta)
        assert record.created_at  # non-empty
        assert record.updated_at  # non-empty
        assert record.disconnected_at is None

    @pytest.mark.asyncio
    async def test_create_session_increments_count(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        assert registry.total_count == 0
        await registry.create_session(client_id="c1", metadata=meta)
        assert registry.total_count == 1
        await registry.create_session(client_id="c2", metadata=meta)
        assert registry.total_count == 2

    @pytest.mark.asyncio
    async def test_create_duplicate_client_id_raises(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        await registry.create_session(client_id="c1", metadata=meta)
        with pytest.raises(ValueError, match="already has an active session"):
            await registry.create_session(client_id="c1", metadata=meta)


# ---------------------------------------------------------------------------
# Session lookup tests
# ---------------------------------------------------------------------------


class TestSessionLookup:
    """Tests for looking up sessions by ID."""

    @pytest.mark.asyncio
    async def test_lookup_by_session_id(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        created = await registry.create_session(client_id="c1", metadata=meta)
        found = registry.lookup(created.session_id)
        assert found is not None
        assert found.session_id == created.session_id

    @pytest.mark.asyncio
    async def test_lookup_nonexistent_returns_none(self) -> None:
        registry = SessionRegistry()
        assert registry.lookup("no-such-id") is None

    @pytest.mark.asyncio
    async def test_lookup_by_client_id(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        created = await registry.create_session(client_id="c1", metadata=meta)
        found = registry.lookup_by_client_id("c1")
        assert found is not None
        assert found.session_id == created.session_id

    @pytest.mark.asyncio
    async def test_lookup_by_client_id_nonexistent(self) -> None:
        registry = SessionRegistry()
        assert registry.lookup_by_client_id("no-such") is None

    @pytest.mark.asyncio
    async def test_has_session_true(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        created = await registry.create_session(client_id="c1", metadata=meta)
        assert registry.has_session(created.session_id)

    @pytest.mark.asyncio
    async def test_has_session_false(self) -> None:
        registry = SessionRegistry()
        assert not registry.has_session("no-such")


# ---------------------------------------------------------------------------
# Session enumeration tests
# ---------------------------------------------------------------------------


class TestSessionEnumeration:
    """Tests for listing and enumerating sessions."""

    @pytest.mark.asyncio
    async def test_list_active_returns_non_disconnected(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        await registry.create_session(client_id="c1", metadata=meta)
        await registry.create_session(client_id="c2", metadata=meta)

        active = registry.list_active()
        assert len(active) == 2

    @pytest.mark.asyncio
    async def test_list_active_excludes_disconnected(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        r1 = await registry.create_session(client_id="c1", metadata=meta)
        await registry.create_session(client_id="c2", metadata=meta)
        await registry.disconnect_session(r1.session_id)

        active = registry.list_active()
        assert len(active) == 1
        assert active[0].client_id == "c2"

    @pytest.mark.asyncio
    async def test_list_all_includes_disconnected(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        r1 = await registry.create_session(client_id="c1", metadata=meta)
        await registry.create_session(client_id="c2", metadata=meta)
        await registry.disconnect_session(r1.session_id)

        all_sessions = registry.list_all()
        assert len(all_sessions) == 2

    @pytest.mark.asyncio
    async def test_list_active_returns_tuple(self) -> None:
        registry = SessionRegistry()
        result = registry.list_active()
        assert isinstance(result, tuple)

    @pytest.mark.asyncio
    async def test_list_all_returns_tuple(self) -> None:
        registry = SessionRegistry()
        result = registry.list_all()
        assert isinstance(result, tuple)

    @pytest.mark.asyncio
    async def test_active_count_property(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        assert registry.active_count == 0

        r1 = await registry.create_session(client_id="c1", metadata=meta)
        assert registry.active_count == 1

        await registry.create_session(client_id="c2", metadata=meta)
        assert registry.active_count == 2

        await registry.disconnect_session(r1.session_id)
        assert registry.active_count == 1


# ---------------------------------------------------------------------------
# Session state transition tests
# ---------------------------------------------------------------------------


class TestSessionStateTransitions:
    """Tests for session lifecycle state transitions."""

    @pytest.mark.asyncio
    async def test_transition_created_to_active(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        created = await registry.create_session(client_id="c1", metadata=meta)

        updated = await registry.transition_state(
            created.session_id, SessionState.ACTIVE
        )
        assert updated is not None
        assert updated.state == SessionState.ACTIVE
        assert updated.session_id == created.session_id

    @pytest.mark.asyncio
    async def test_transition_active_to_idle(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        created = await registry.create_session(client_id="c1", metadata=meta)
        await registry.transition_state(
            created.session_id, SessionState.ACTIVE
        )
        updated = await registry.transition_state(
            created.session_id, SessionState.IDLE
        )
        assert updated is not None
        assert updated.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_transition_updates_registry(self) -> None:
        """After transition, lookup reflects the new state."""
        registry = SessionRegistry()
        meta = _make_metadata()
        created = await registry.create_session(client_id="c1", metadata=meta)
        await registry.transition_state(
            created.session_id, SessionState.ACTIVE
        )

        found = registry.lookup(created.session_id)
        assert found is not None
        assert found.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_transition_nonexistent_returns_none(self) -> None:
        registry = SessionRegistry()
        result = await registry.transition_state(
            "no-such", SessionState.ACTIVE
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_transition_disconnected_session_returns_none(self) -> None:
        """Cannot transition out of DISCONNECTED terminal state."""
        registry = SessionRegistry()
        meta = _make_metadata()
        created = await registry.create_session(client_id="c1", metadata=meta)
        await registry.disconnect_session(created.session_id)

        result = await registry.transition_state(
            created.session_id, SessionState.ACTIVE
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_transition_updates_timestamp(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        created = await registry.create_session(client_id="c1", metadata=meta)
        original_ts = created.updated_at

        updated = await registry.transition_state(
            created.session_id, SessionState.ACTIVE
        )
        assert updated is not None
        # Timestamps should be different (or at least not regress)
        assert updated.updated_at >= original_ts


# ---------------------------------------------------------------------------
# Disconnect session tests
# ---------------------------------------------------------------------------


class TestSessionDisconnect:
    """Tests for disconnecting sessions."""

    @pytest.mark.asyncio
    async def test_disconnect_session(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        created = await registry.create_session(client_id="c1", metadata=meta)

        disconnected = await registry.disconnect_session(created.session_id)
        assert disconnected is not None
        assert disconnected.state == SessionState.DISCONNECTED
        assert disconnected.disconnected_at is not None

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_returns_none(self) -> None:
        registry = SessionRegistry()
        result = await registry.disconnect_session("no-such")
        assert result is None

    @pytest.mark.asyncio
    async def test_disconnect_already_disconnected_returns_none(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()
        created = await registry.create_session(client_id="c1", metadata=meta)
        await registry.disconnect_session(created.session_id)

        # Second disconnect is a no-op
        result = await registry.disconnect_session(created.session_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_disconnect_frees_client_id(self) -> None:
        """After disconnect, the same client_id can create a new session."""
        registry = SessionRegistry()
        meta = _make_metadata()
        r1 = await registry.create_session(client_id="c1", metadata=meta)
        await registry.disconnect_session(r1.session_id)

        # Should not raise -- client_id is freed
        r2 = await registry.create_session(client_id="c1", metadata=meta)
        assert r2.session_id != r1.session_id


# ---------------------------------------------------------------------------
# Event emission tests
# ---------------------------------------------------------------------------


class TestSessionRegistryEvents:
    """Tests for lifecycle event emission via EventBus."""

    @pytest.mark.asyncio
    async def test_create_emits_session_created_event(self) -> None:
        bus = EventBus()
        registry = SessionRegistry(event_bus=bus)

        received: list[Event] = []

        async def on_created(event: Event) -> None:
            received.append(event)

        bus.subscribe(SESSION_CREATED_EVENT, on_created)

        meta = _make_metadata()
        record = await registry.create_session(client_id="c1", metadata=meta)

        assert len(received) == 1
        assert received[0].event_type == SESSION_CREATED_EVENT
        assert received[0].payload["session_id"] == record.session_id
        assert received[0].payload["client_id"] == "c1"

    @pytest.mark.asyncio
    async def test_transition_emits_state_changed_event(self) -> None:
        bus = EventBus()
        registry = SessionRegistry(event_bus=bus)

        received: list[Event] = []

        async def on_changed(event: Event) -> None:
            received.append(event)

        bus.subscribe(SESSION_STATE_CHANGED_EVENT, on_changed)

        meta = _make_metadata()
        created = await registry.create_session(client_id="c1", metadata=meta)
        await registry.transition_state(
            created.session_id, SessionState.ACTIVE
        )

        assert len(received) == 1
        assert received[0].payload["new_state"] == "active"
        assert received[0].payload["previous_state"] == "created"

    @pytest.mark.asyncio
    async def test_disconnect_emits_disconnected_event(self) -> None:
        bus = EventBus()
        registry = SessionRegistry(event_bus=bus)

        received: list[Event] = []

        async def on_disconnected(event: Event) -> None:
            received.append(event)

        bus.subscribe(SESSION_DISCONNECTED_EVENT, on_disconnected)

        meta = _make_metadata()
        created = await registry.create_session(client_id="c1", metadata=meta)
        await registry.disconnect_session(created.session_id)

        assert len(received) == 1
        assert received[0].payload["session_id"] == created.session_id

    @pytest.mark.asyncio
    async def test_no_event_bus_still_works(self) -> None:
        """Registry operates without an event bus."""
        registry = SessionRegistry()
        meta = _make_metadata()
        record = await registry.create_session(client_id="c1", metadata=meta)
        assert record is not None

        await registry.disconnect_session(record.session_id)
        assert registry.active_count == 0


# ---------------------------------------------------------------------------
# Concurrent operation tests
# ---------------------------------------------------------------------------


class TestSessionRegistryConcurrency:
    """Tests for concurrent session operations."""

    @pytest.mark.asyncio
    async def test_concurrent_creates(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()

        async def create_session(idx: int) -> None:
            await registry.create_session(client_id=f"c{idx}", metadata=meta)

        await asyncio.gather(*(create_session(i) for i in range(10)))
        assert registry.total_count == 10
        assert registry.active_count == 10

    @pytest.mark.asyncio
    async def test_concurrent_create_and_disconnect(self) -> None:
        registry = SessionRegistry()
        meta = _make_metadata()

        # Pre-create 5 sessions
        records = []
        for i in range(5):
            r = await registry.create_session(client_id=f"c{i}", metadata=meta)
            records.append(r)

        async def disconnect(idx: int) -> None:
            await registry.disconnect_session(records[idx].session_id)

        async def create(idx: int) -> None:
            await registry.create_session(
                client_id=f"new-{idx}", metadata=meta
            )

        tasks = [disconnect(i) for i in range(5)]
        tasks += [create(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # 5 old disconnected, 5 new active
        assert registry.active_count == 5
        assert registry.total_count == 10
