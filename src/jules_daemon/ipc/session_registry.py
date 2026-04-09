"""Per-client session registry with lifecycle management.

Manages the lifecycle of client sessions -- from creation on connect
through state transitions to disconnect. Each session is uniquely
identified and tracks rich client metadata.

This module sits above the ConnectionManager (which tracks raw socket
connections) and provides a higher-level abstraction for tracking
logical client interaction sessions with state management.

Key responsibilities:

- **Session creation**: Generate unique session IDs and register new
  sessions with client metadata on connect.
- **Session lookup**: Find sessions by session ID or client ID.
- **Enumeration**: List active (non-disconnected) sessions or all sessions.
- **State transitions**: Move sessions through lifecycle states
  (CREATED -> ACTIVE -> IDLE -> DISCONNECTED).
- **Disconnect handling**: Cleanly terminate sessions and free resources.
- **Event emission**: Emit lifecycle events through an optional event bus
  for downstream consumers (wiki persistence, audit logging).
- **Concurrency safety**: Uses an asyncio Lock to serialize mutations.

The event bus is optional. When no bus is provided, the registry
operates as a pure in-memory session store.

Architecture::

    ConnectionDispatcher._handle_connection_lifecycle()
        |
        v
    SessionRegistry.create_session(client_id, metadata)
        |---> generates session_id
        |---> stores SessionRecord
        |---> bus.emit(SESSION_CREATED_EVENT)
        v
    SessionRegistry.transition_state(session_id, new_state)
        |---> updates SessionRecord (immutably)
        |---> bus.emit(SESSION_STATE_CHANGED_EVENT)
        v
    SessionRegistry.disconnect_session(session_id)
        |---> transitions to DISCONNECTED
        |---> frees client_id mapping
        |---> bus.emit(SESSION_DISCONNECTED_EVENT)

Usage::

    from jules_daemon.ipc.session_registry import SessionRegistry
    from jules_daemon.ipc.session_models import ClientMetadata
    from jules_daemon.ipc.event_bus import EventBus

    bus = EventBus()
    registry = SessionRegistry(event_bus=bus)

    metadata = ClientMetadata(
        client_name="jules-cli",
        client_version="0.1.0",
        client_pid=12345,
        hostname="workstation.local",
        extra={},
    )
    record = await registry.create_session(client_id="c1", metadata=metadata)
    active = registry.list_active()
    await registry.disconnect_session(record.session_id)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from jules_daemon.ipc.event_bus import Event, EventBus
from jules_daemon.ipc.session_models import (
    ClientMetadata,
    SessionRecord,
    SessionState,
)

__all__ = [
    "SESSION_CREATED_EVENT",
    "SESSION_DISCONNECTED_EVENT",
    "SESSION_STATE_CHANGED_EVENT",
    "SessionRegistry",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

SESSION_CREATED_EVENT: str = "session_created"
"""Event type emitted when a new session is created."""

SESSION_STATE_CHANGED_EVENT: str = "session_state_changed"
"""Event type emitted when a session transitions to a new state."""

SESSION_DISCONNECTED_EVENT: str = "session_disconnected"
"""Event type emitted when a session is disconnected."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_session_id() -> str:
    """Generate a unique session identifier.

    Returns:
        String in ``sess-{uuid_hex_prefix}`` format.
    """
    return f"sess-{uuid.uuid4().hex}"


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# SessionRegistry
# ---------------------------------------------------------------------------


class SessionRegistry:
    """Per-client session registry with lifecycle management.

    Maintains an in-memory registry of client sessions, keyed by both
    ``session_id`` (primary) and ``client_id`` (secondary index for active
    sessions only). Emits lifecycle events through an optional event bus.

    Concurrency: Uses an asyncio.Lock to serialize create/transition/disconnect
    operations. Read-only operations (lookup, list_active, list_all, has_session,
    active_count, total_count) do not acquire the lock and are safe to call
    from any coroutine.

    Args:
        event_bus: Optional EventBus for emitting session lifecycle events.
            When None, the registry operates without event emission.
    """

    def __init__(self, *, event_bus: EventBus | None = None) -> None:
        self._event_bus = event_bus
        # Primary index: session_id -> SessionRecord
        self._sessions: dict[str, SessionRecord] = {}
        # Secondary index: client_id -> session_id (active sessions only)
        self._client_index: dict[str, str] = {}
        self._lock = asyncio.Lock()

    # -- Properties --
    # Note: Read-only properties do not acquire self._lock. This is safe
    # because (a) mutations replace self._sessions / self._client_index
    # atomically via dict comprehension, and (b) no await exists in any
    # read path, so cooperative scheduling cannot interrupt mid-read.

    @property
    def total_count(self) -> int:
        """Total number of sessions (including disconnected)."""
        return len(self._sessions)

    @property
    def active_count(self) -> int:
        """Number of currently active (non-disconnected) sessions."""
        return len(self._client_index)

    # -- Session creation --

    async def create_session(
        self,
        *,
        client_id: str,
        metadata: ClientMetadata,
    ) -> SessionRecord:
        """Create a new session for a connecting client.

        Generates a unique session ID, creates a SessionRecord in CREATED
        state, and emits a SESSION_CREATED_EVENT if an event bus is configured.

        Args:
            client_id: The connection-level client identifier.
            metadata: Rich metadata about the connecting client.

        Returns:
            The newly created SessionRecord.

        Raises:
            ValueError: If the client_id already has an active session.
        """
        now = _now_iso()
        session_id = _generate_session_id()

        async with self._lock:
            if client_id in self._client_index:
                raise ValueError(
                    f"Client {client_id!r} already has an active session "
                    f"({self._client_index[client_id]!r})"
                )

            record = SessionRecord(
                session_id=session_id,
                client_id=client_id,
                state=SessionState.CREATED,
                metadata=metadata,
                created_at=now,
                updated_at=now,
            )

            # Update both indices (immutable pattern for the dicts)
            self._sessions = {**self._sessions, session_id: record}
            self._client_index = {**self._client_index, client_id: session_id}

        logger.info(
            "Session created: session_id=%s client_id=%s client_name=%s",
            session_id,
            client_id,
            metadata.client_name,
        )

        await self._emit(
            SESSION_CREATED_EVENT,
            record.to_event_payload(),
        )

        return record

    # -- Session lookup --

    def lookup(self, session_id: str) -> SessionRecord | None:
        """Look up a session by its unique session ID.

        Args:
            session_id: The session identifier to search for.

        Returns:
            The matching SessionRecord, or None if not found.
        """
        return self._sessions.get(session_id)

    def lookup_by_client_id(self, client_id: str) -> SessionRecord | None:
        """Look up the active session for a given client ID.

        Only returns sessions that are not disconnected. If the client
        has disconnected, returns None.

        Args:
            client_id: The connection-level client identifier.

        Returns:
            The active SessionRecord for that client, or None.
        """
        session_id = self._client_index.get(client_id)
        if session_id is None:
            return None
        return self._sessions.get(session_id)

    def has_session(self, session_id: str) -> bool:
        """Check whether a session exists in the registry.

        Args:
            session_id: The session identifier to check.

        Returns:
            True if the session exists (active or disconnected).
        """
        return session_id in self._sessions

    # -- Session enumeration --

    def list_active(self) -> tuple[SessionRecord, ...]:
        """Return an immutable snapshot of all active (non-disconnected) sessions.

        Returns:
            Tuple of SessionRecord objects for sessions that are not in
            DISCONNECTED state. Empty tuple if no active sessions.
        """
        return tuple(
            record
            for record in self._sessions.values()
            if record.is_active
        )

    def list_all(self) -> tuple[SessionRecord, ...]:
        """Return an immutable snapshot of all sessions.

        Includes both active and disconnected sessions.

        Returns:
            Tuple of all SessionRecord objects. Empty tuple if empty.
        """
        return tuple(self._sessions.values())

    # -- State transitions --

    async def transition_state(
        self,
        session_id: str,
        new_state: SessionState,
    ) -> SessionRecord | None:
        """Transition a session to a new lifecycle state.

        Creates a new SessionRecord with the updated state and timestamp.
        Emits a SESSION_STATE_CHANGED_EVENT if an event bus is configured.

        For disconnecting, prefer ``disconnect_session()`` which handles
        the secondary index cleanup and disconnect-specific event emission.

        Args:
            session_id: The session to transition.
            new_state: The target lifecycle state.

        Returns:
            The updated SessionRecord, or None if the session was not found
            or is already disconnected (terminal state).
        """
        previous_state: SessionState | None = None

        async with self._lock:
            current = self._sessions.get(session_id)
            if current is None:
                return None

            # DISCONNECTED is a terminal state -- no transitions allowed.
            if current.is_disconnected:
                logger.warning(
                    "Attempted to transition disconnected session %s to %s",
                    session_id,
                    new_state.value,
                )
                return None

            previous_state = current.state
            now = _now_iso()
            updated = current.with_state(new_state, updated_at=now)

            # Replace in the primary index (immutable pattern)
            self._sessions = {
                sid: (updated if sid == session_id else rec)
                for sid, rec in self._sessions.items()
            }

        logger.info(
            "Session state changed: session_id=%s %s -> %s",
            session_id,
            previous_state.value if previous_state else "unknown",
            new_state.value,
        )

        await self._emit(
            SESSION_STATE_CHANGED_EVENT,
            {
                "session_id": session_id,
                "previous_state": previous_state.value if previous_state else None,
                "new_state": new_state.value,
            },
        )

        return updated

    async def disconnect_session(
        self,
        session_id: str,
    ) -> SessionRecord | None:
        """Disconnect a session and free its client_id mapping.

        Transitions the session to DISCONNECTED state, removes it from
        the client_id secondary index, and emits SESSION_DISCONNECTED_EVENT.

        This is idempotent for already-disconnected sessions: returns None
        if the session is not found or already disconnected.

        Args:
            session_id: The session to disconnect.

        Returns:
            The disconnected SessionRecord, or None if not found or
            already disconnected.
        """
        async with self._lock:
            current = self._sessions.get(session_id)
            if current is None:
                logger.debug(
                    "Attempted to disconnect nonexistent session: %s",
                    session_id,
                )
                return None

            if current.is_disconnected:
                logger.debug(
                    "Session %s is already disconnected", session_id
                )
                return None

            now = _now_iso()
            disconnected = current.with_disconnected(disconnected_at=now)

            # Update primary index
            self._sessions = {
                sid: (disconnected if sid == session_id else rec)
                for sid, rec in self._sessions.items()
            }

            # Remove from client_id secondary index
            client_id = current.client_id
            self._client_index = {
                cid: sid
                for cid, sid in self._client_index.items()
                if cid != client_id
            }

        logger.info(
            "Session disconnected: session_id=%s client_id=%s",
            session_id,
            client_id,
        )

        await self._emit(
            SESSION_DISCONNECTED_EVENT,
            disconnected.to_event_payload(),
        )

        return disconnected

    # -- Internal: event emission --

    async def _emit(
        self,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        """Emit an event through the event bus if configured.

        Silently does nothing when no event bus is set.

        Args:
            event_type: The event type string.
            payload: The event payload dict.
        """
        if self._event_bus is None:
            return

        await self._event_bus.emit(
            Event(event_type=event_type, payload=payload)
        )
