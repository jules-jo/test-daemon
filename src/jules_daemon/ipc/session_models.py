"""Immutable data models for per-client session tracking.

Provides the core data types used by the session registry to track
client sessions with unique IDs, rich metadata, and state lifecycle.

A session represents a client's logical interaction period with the
daemon. It is created on connect, transitions through states as the
client interacts, and terminates on disconnect.

Models:
    SessionState    -- Lifecycle states for a client session.
    ClientMetadata  -- Immutable descriptor of the connecting client.
    SessionRecord   -- Complete snapshot of a session with state and timestamps.

All models are frozen dataclasses -- state transitions produce new
instances, never mutate existing ones.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "ClientMetadata",
    "SessionRecord",
    "SessionState",
]


# ---------------------------------------------------------------------------
# SessionState enum
# ---------------------------------------------------------------------------


class SessionState(Enum):
    """Lifecycle states for a client session.

    Values:
        CREATED:      Session just established, not yet actively communicating.
        ACTIVE:       Client is actively sending/receiving messages.
        IDLE:         Client connected but no recent activity.
        DISCONNECTED: Client has disconnected (terminal state).
    """

    CREATED = "created"
    ACTIVE = "active"
    IDLE = "idle"
    DISCONNECTED = "disconnected"


# ---------------------------------------------------------------------------
# ClientMetadata dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientMetadata:
    """Immutable descriptor for metadata about the connecting client.

    Captures identity and environment information provided by the client
    during session establishment. Used for audit logging and debugging.

    Attributes:
        client_name:    Human-readable client identifier (e.g., "jules-cli").
        client_version: Semantic version of the client software.
        client_pid:     Process ID of the client (None if unknown).
        hostname:       Hostname of the machine running the client.
        extra:          Additional key-value metadata from the client.
    """

    client_name: str
    client_version: str
    client_pid: int | None
    hostname: str
    extra: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.client_name, str) or not self.client_name.strip():
            raise ValueError("client_name must not be empty")
        if not isinstance(self.client_version, str) or not self.client_version.strip():
            raise ValueError("client_version must not be empty")
        if not isinstance(self.hostname, str) or not self.hostname.strip():
            raise ValueError("hostname must not be empty")
        if self.client_pid is not None and self.client_pid <= 0:
            raise ValueError(
                f"client_pid must be a positive integer or None, "
                f"got {self.client_pid}"
            )
        # Defensive copy to prevent caller mutation via shared reference.
        # MappingProxyType wraps a fresh dict copy as a read-only view.
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for event payloads and logging.

        Returns:
            New dict with all metadata fields. The ``extra`` dict is
            shallow-copied to prevent shared references.
        """
        return {
            "client_name": self.client_name,
            "client_version": self.client_version,
            "client_pid": self.client_pid,
            "hostname": self.hostname,
            "extra": dict(self.extra),
        }


# ---------------------------------------------------------------------------
# SessionRecord dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionRecord:
    """Complete snapshot of a client session with state and timestamps.

    This is the primary record type stored in the session registry.
    State transitions produce new instances via the ``with_*`` methods,
    preserving immutability.

    Attributes:
        session_id:      Unique identifier for this session (UUID-based).
        client_id:       Connection-level client identifier.
        state:           Current lifecycle state.
        metadata:        Rich client metadata captured on connect.
        created_at:      ISO 8601 UTC timestamp of session creation.
        updated_at:      ISO 8601 UTC timestamp of last state change.
        disconnected_at: ISO 8601 UTC timestamp of disconnect (None while active).
    """

    session_id: str
    client_id: str
    state: SessionState
    metadata: ClientMetadata
    created_at: str
    updated_at: str
    disconnected_at: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if not isinstance(self.client_id, str) or not self.client_id.strip():
            raise ValueError("client_id must not be empty")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must not be empty")
        if not isinstance(self.updated_at, str) or not self.updated_at.strip():
            raise ValueError("updated_at must not be empty")

    # -- Computed properties --

    @property
    def is_active(self) -> bool:
        """True if the session is not disconnected.

        Sessions in CREATED, ACTIVE, and IDLE states are considered active
        (i.e., they are not in a terminal state).
        """
        return self.state != SessionState.DISCONNECTED

    @property
    def is_disconnected(self) -> bool:
        """True if the session has been disconnected (terminal state)."""
        return self.state == SessionState.DISCONNECTED

    # -- Immutable state transition methods --

    def with_state(
        self,
        new_state: SessionState,
        updated_at: str,
    ) -> SessionRecord:
        """Return a new SessionRecord with the specified state.

        Preserves all other fields from the current instance.

        Args:
            new_state: The target lifecycle state.
            updated_at: ISO 8601 UTC timestamp for the transition.

        Returns:
            New SessionRecord with updated state and timestamp.
        """
        return replace(
            self,
            state=new_state,
            updated_at=updated_at,
        )

    def with_disconnected(
        self,
        disconnected_at: str,
    ) -> SessionRecord:
        """Return a new SessionRecord in the DISCONNECTED terminal state.

        Sets both the state and the disconnected_at timestamp.

        Args:
            disconnected_at: ISO 8601 UTC timestamp of the disconnect.

        Returns:
            New SessionRecord in DISCONNECTED state.
        """
        return replace(
            self,
            state=SessionState.DISCONNECTED,
            updated_at=disconnected_at,
            disconnected_at=disconnected_at,
        )

    # -- Serialization --

    def to_event_payload(self) -> dict[str, Any]:
        """Serialize to a dict suitable for event bus payloads.

        Returns:
            New dict with all session fields. Metadata is nested as a
            sub-dict via ``ClientMetadata.to_dict()``.
        """
        return {
            "session_id": self.session_id,
            "client_id": self.client_id,
            "state": self.state.value,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "disconnected_at": self.disconnected_at,
        }
