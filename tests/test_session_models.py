"""Tests for per-client session data models.

Covers:
    - SessionState enum values and transitions
    - ClientMetadata construction, validation, and immutability
    - SessionRecord construction, validation, and immutability
    - SessionRecord state transition helpers
    - SessionRecord computed properties (is_active, is_disconnected)
    - SessionRecord serialization to event payload format
    - Edge cases: empty strings, negative PIDs, blank hostnames
"""

from __future__ import annotations

import pytest

from jules_daemon.ipc.session_models import (
    ClientMetadata,
    SessionRecord,
    SessionState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(
    client_name: str = "jules-cli",
    client_version: str = "0.1.0",
    client_pid: int | None = 12345,
    hostname: str = "workstation.local",
    extra: dict[str, str] | None = None,
) -> ClientMetadata:
    """Build a ClientMetadata for testing."""
    return ClientMetadata(
        client_name=client_name,
        client_version=client_version,
        client_pid=client_pid,
        hostname=hostname,
        extra=extra if extra is not None else {},
    )


def _make_session(
    session_id: str = "sess-abc123",
    client_id: str = "client-001",
    state: SessionState = SessionState.CREATED,
    metadata: ClientMetadata | None = None,
    created_at: str = "2026-04-09T12:00:00Z",
    updated_at: str = "2026-04-09T12:00:00Z",
    disconnected_at: str | None = None,
) -> SessionRecord:
    """Build a SessionRecord for testing."""
    return SessionRecord(
        session_id=session_id,
        client_id=client_id,
        state=state,
        metadata=metadata or _make_metadata(),
        created_at=created_at,
        updated_at=updated_at,
        disconnected_at=disconnected_at,
    )


# ---------------------------------------------------------------------------
# SessionState enum tests
# ---------------------------------------------------------------------------


class TestSessionState:
    """Tests for the SessionState enum."""

    def test_values_exist(self) -> None:
        assert SessionState.CREATED.value == "created"
        assert SessionState.ACTIVE.value == "active"
        assert SessionState.IDLE.value == "idle"
        assert SessionState.DISCONNECTED.value == "disconnected"

    def test_all_members(self) -> None:
        names = {s.name for s in SessionState}
        assert names == {"CREATED", "ACTIVE", "IDLE", "DISCONNECTED"}

    def test_value_uniqueness(self) -> None:
        values = [s.value for s in SessionState]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# ClientMetadata tests
# ---------------------------------------------------------------------------


class TestClientMetadata:
    """Tests for the immutable ClientMetadata dataclass."""

    def test_create(self) -> None:
        meta = _make_metadata()
        assert meta.client_name == "jules-cli"
        assert meta.client_version == "0.1.0"
        assert meta.client_pid == 12345
        assert meta.hostname == "workstation.local"
        assert meta.extra == {}

    def test_frozen(self) -> None:
        meta = _make_metadata()
        with pytest.raises(AttributeError):
            meta.client_name = "mutated"  # type: ignore[misc]

    def test_empty_client_name_raises(self) -> None:
        with pytest.raises(ValueError, match="client_name"):
            _make_metadata(client_name="")

    def test_whitespace_client_name_raises(self) -> None:
        with pytest.raises(ValueError, match="client_name"):
            _make_metadata(client_name="   ")

    def test_empty_client_version_raises(self) -> None:
        with pytest.raises(ValueError, match="client_version"):
            _make_metadata(client_version="")

    def test_empty_hostname_raises(self) -> None:
        with pytest.raises(ValueError, match="hostname"):
            _make_metadata(hostname="")

    def test_client_pid_none_allowed(self) -> None:
        meta = _make_metadata(client_pid=None)
        assert meta.client_pid is None

    def test_client_pid_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="client_pid"):
            _make_metadata(client_pid=0)

    def test_client_pid_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="client_pid"):
            _make_metadata(client_pid=-1)

    def test_extra_metadata(self) -> None:
        meta = _make_metadata(extra={"os": "linux", "arch": "x86_64"})
        assert meta.extra == {"os": "linux", "arch": "x86_64"}

    def test_to_dict(self) -> None:
        meta = _make_metadata(extra={"os": "linux"})
        d = meta.to_dict()
        assert d["client_name"] == "jules-cli"
        assert d["client_version"] == "0.1.0"
        assert d["client_pid"] == 12345
        assert d["hostname"] == "workstation.local"
        assert d["extra"] == {"os": "linux"}

    def test_to_dict_is_new_dict(self) -> None:
        """to_dict returns a new dict, not a shared reference."""
        meta = _make_metadata()
        d1 = meta.to_dict()
        d2 = meta.to_dict()
        assert d1 is not d2

    def test_extra_is_immutable_after_construction(self) -> None:
        """Caller cannot mutate extra through the original reference."""
        original = {"os": "linux"}
        meta = _make_metadata(extra=original)
        original["injected"] = "value"
        assert "injected" not in meta.extra

    def test_extra_cannot_be_mutated_directly(self) -> None:
        """The extra mapping is read-only after construction."""
        meta = _make_metadata(extra={"os": "linux"})
        with pytest.raises(TypeError):
            meta.extra["new_key"] = "value"  # type: ignore[index]


# ---------------------------------------------------------------------------
# SessionRecord construction tests
# ---------------------------------------------------------------------------


class TestSessionRecordConstruction:
    """Tests for SessionRecord creation and validation."""

    def test_create(self) -> None:
        record = _make_session()
        assert record.session_id == "sess-abc123"
        assert record.client_id == "client-001"
        assert record.state == SessionState.CREATED
        assert record.metadata.client_name == "jules-cli"
        assert record.created_at == "2026-04-09T12:00:00Z"
        assert record.updated_at == "2026-04-09T12:00:00Z"
        assert record.disconnected_at is None

    def test_frozen(self) -> None:
        record = _make_session()
        with pytest.raises(AttributeError):
            record.state = SessionState.ACTIVE  # type: ignore[misc]

    def test_empty_session_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            _make_session(session_id="")

    def test_whitespace_session_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            _make_session(session_id="   ")

    def test_empty_client_id_raises(self) -> None:
        with pytest.raises(ValueError, match="client_id"):
            _make_session(client_id="")

    def test_empty_created_at_raises(self) -> None:
        with pytest.raises(ValueError, match="created_at"):
            _make_session(created_at="")

    def test_empty_updated_at_raises(self) -> None:
        with pytest.raises(ValueError, match="updated_at"):
            _make_session(updated_at="")


# ---------------------------------------------------------------------------
# SessionRecord computed properties tests
# ---------------------------------------------------------------------------


class TestSessionRecordProperties:
    """Tests for SessionRecord computed properties."""

    def test_is_active_created(self) -> None:
        record = _make_session(state=SessionState.CREATED)
        assert record.is_active is True

    def test_is_active_active(self) -> None:
        record = _make_session(state=SessionState.ACTIVE)
        assert record.is_active is True

    def test_is_active_idle(self) -> None:
        record = _make_session(state=SessionState.IDLE)
        assert record.is_active is True

    def test_is_active_disconnected(self) -> None:
        record = _make_session(state=SessionState.DISCONNECTED)
        assert record.is_active is False

    def test_is_disconnected_true(self) -> None:
        record = _make_session(state=SessionState.DISCONNECTED)
        assert record.is_disconnected is True

    def test_is_disconnected_false(self) -> None:
        record = _make_session(state=SessionState.ACTIVE)
        assert record.is_disconnected is False


# ---------------------------------------------------------------------------
# SessionRecord state transition tests
# ---------------------------------------------------------------------------


class TestSessionRecordTransitions:
    """Tests for immutable state transition methods."""

    def test_with_state_returns_new_instance(self) -> None:
        original = _make_session(state=SessionState.CREATED)
        updated = original.with_state(
            SessionState.ACTIVE, updated_at="2026-04-09T12:01:00Z"
        )
        assert updated is not original
        assert updated.state == SessionState.ACTIVE
        assert updated.updated_at == "2026-04-09T12:01:00Z"

    def test_with_state_preserves_other_fields(self) -> None:
        original = _make_session(state=SessionState.CREATED)
        updated = original.with_state(
            SessionState.ACTIVE, updated_at="2026-04-09T12:01:00Z"
        )
        assert updated.session_id == original.session_id
        assert updated.client_id == original.client_id
        assert updated.metadata == original.metadata
        assert updated.created_at == original.created_at

    def test_with_state_does_not_mutate_original(self) -> None:
        original = _make_session(state=SessionState.CREATED)
        _ = original.with_state(
            SessionState.ACTIVE, updated_at="2026-04-09T12:01:00Z"
        )
        assert original.state == SessionState.CREATED

    def test_with_disconnected_sets_disconnected_at(self) -> None:
        record = _make_session(state=SessionState.ACTIVE)
        disconnected = record.with_disconnected(
            disconnected_at="2026-04-09T13:00:00Z"
        )
        assert disconnected.state == SessionState.DISCONNECTED
        assert disconnected.disconnected_at == "2026-04-09T13:00:00Z"
        assert disconnected.updated_at == "2026-04-09T13:00:00Z"

    def test_with_disconnected_preserves_other_fields(self) -> None:
        record = _make_session(state=SessionState.ACTIVE)
        disconnected = record.with_disconnected(
            disconnected_at="2026-04-09T13:00:00Z"
        )
        assert disconnected.session_id == record.session_id
        assert disconnected.client_id == record.client_id
        assert disconnected.metadata == record.metadata
        assert disconnected.created_at == record.created_at


# ---------------------------------------------------------------------------
# SessionRecord serialization tests
# ---------------------------------------------------------------------------


class TestSessionRecordSerialization:
    """Tests for to_event_payload serialization."""

    def test_to_event_payload_contains_all_fields(self) -> None:
        record = _make_session()
        payload = record.to_event_payload()
        assert payload["session_id"] == "sess-abc123"
        assert payload["client_id"] == "client-001"
        assert payload["state"] == "created"
        assert payload["created_at"] == "2026-04-09T12:00:00Z"
        assert payload["updated_at"] == "2026-04-09T12:00:00Z"
        assert payload["disconnected_at"] is None

    def test_to_event_payload_includes_metadata(self) -> None:
        record = _make_session()
        payload = record.to_event_payload()
        assert "metadata" in payload
        assert payload["metadata"]["client_name"] == "jules-cli"

    def test_to_event_payload_is_new_dict(self) -> None:
        record = _make_session()
        p1 = record.to_event_payload()
        p2 = record.to_event_payload()
        assert p1 is not p2
