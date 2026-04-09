"""Disconnect event detection and classification for IPC client connections.

Catches and classifies unexpected client disconnection signals into a
unified ``DisconnectEvent`` model. The daemon's IPC layer encounters
several distinct failure modes when a CLI client disconnects:

    - **EOF**: The client closed the connection cleanly or the stream
      ended mid-read (``asyncio.IncompleteReadError``, ``EOFError``).
    - **Broken pipe**: A write was attempted on a pipe whose read end
      has been closed (``BrokenPipeError``).
    - **Connection reset**: The remote end forcibly closed the connection
      (``ConnectionResetError``).
    - **Socket timeout**: A read or write timed out waiting for the peer
      (``asyncio.TimeoutError``).
    - **OS error**: A generic transport-level failure (``OSError`` and
      its non-specific subclasses).
    - **Unknown**: Any other exception not covered above.

The ``classify_disconnect`` function is the primary entry point. It
inspects the exception type (checking subclass specificity order to
avoid masking, e.g., ``BrokenPipeError`` before ``OSError``) and
returns an immutable ``DisconnectEvent`` with structured metadata.

Downstream consumers -- such as the ``ConnectionManager``, audit log,
and wiki persistence layer -- can subscribe to ``DISCONNECT_EVENT_TYPE``
events on the ``EventBus`` to react to disconnections.

Architecture::

    ClientReader / ClientWriter
        |
        v  (exception caught)
    classify_disconnect(exc, client_id)
        |
        v
    DisconnectEvent (immutable model)
        |
        v
    EventBus.emit(Event(DISCONNECT_EVENT_TYPE, payload))
        |
        v
    ConnectionManager / AuditLog / WikiPersistence

Usage::

    from jules_daemon.ipc.disconnect_event import (
        DisconnectEvent,
        DisconnectType,
        classify_disconnect,
    )

    try:
        data = await reader.readexactly(4)
    except Exception as exc:
        event = classify_disconnect(exception=exc, client_id="client-abc")
        await bus.emit(Event(
            event_type=DISCONNECT_EVENT_TYPE,
            payload=event.to_event_payload(),
        ))
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "DISCONNECT_EVENT_TYPE",
    "DisconnectEvent",
    "DisconnectType",
    "classify_disconnect",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCONNECT_EVENT_TYPE: str = "client_disconnect"
"""Event type string for emitting disconnect events on the EventBus."""


# ---------------------------------------------------------------------------
# DisconnectType enum
# ---------------------------------------------------------------------------


class DisconnectType(Enum):
    """Classification of client disconnection signal types.

    Each variant maps to a category of exception that can occur during
    IPC read/write operations. The classification is used for logging,
    metrics, and downstream decision-making (e.g., should the daemon
    attempt to wait for reconnection vs. immediately clean up).

    Values:
        EOF:              Stream ended -- clean close or mid-read EOF.
        BROKEN_PIPE:      Write to a closed read-end pipe.
        CONNECTION_RESET: Remote forcibly closed the connection.
        SOCKET_TIMEOUT:   Read or write operation timed out.
        OS_ERROR:         Generic transport-level OS error.
        UNKNOWN:          Unrecognized exception type.
    """

    EOF = "eof"
    BROKEN_PIPE = "broken_pipe"
    CONNECTION_RESET = "connection_reset"
    SOCKET_TIMEOUT = "socket_timeout"
    OS_ERROR = "os_error"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# DisconnectEvent model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisconnectEvent:
    """Immutable model representing a classified client disconnection.

    Captures the disconnect type, the affected client, a human-readable
    reason, a timestamp, and an extensible metadata dict for exception-
    specific details (errno, bytes received, etc.).

    Attributes:
        disconnect_type: The classified category of the disconnect signal.
        client_id:       Unique identifier of the disconnected client.
        timestamp:       ISO 8601 UTC timestamp of when the disconnect
                         was detected.
        reason:          Human-readable description of the disconnect cause.
        metadata:        Additional exception-specific data. Defaults to
                         an empty dict. Must be JSON-serializable.
    """

    disconnect_type: DisconnectType
    client_id: str
    timestamp: str
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.disconnect_type, DisconnectType):
            raise TypeError(
                "disconnect_type must be a DisconnectType instance, "
                f"got {type(self.disconnect_type).__name__}"
            )
        if not isinstance(self.client_id, str) or not self.client_id.strip():
            raise ValueError("client_id must not be empty")
        if not isinstance(self.timestamp, str) or not self.timestamp.strip():
            raise ValueError("timestamp must not be empty")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must not be empty")
        # Defensive copy into an immutable mapping to prevent external mutation
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )

    def to_event_payload(self) -> dict[str, Any]:
        """Serialize to a dict suitable for EventBus emission.

        Returns a new dict each call (no shared mutable state).

        Returns:
            Dict with disconnect_type (as string value), client_id,
            timestamp, reason, and metadata fields.
        """
        return {
            "disconnect_type": self.disconnect_type.value,
            "client_id": self.client_id,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _exc_class_name(exc: Exception) -> str:
    """Return the unqualified class name of an exception."""
    return type(exc).__name__


def _build_reason(exc: Exception) -> str:
    """Build a human-readable reason string from an exception.

    Format: ``"ExceptionClassName: message"`` or just
    ``"ExceptionClassName"`` when the message is empty.

    Args:
        exc: The exception to describe.

    Returns:
        Non-empty reason string.
    """
    class_name = _exc_class_name(exc)
    msg = str(exc).strip()
    if msg:
        return f"{class_name}: {msg}"
    return class_name


def _extract_errno(exc: OSError) -> dict[str, Any]:
    """Extract errno from an OSError into a metadata dict fragment.

    Args:
        exc: An OSError (or subclass) instance.

    Returns:
        Dict with ``errno`` key if the errno attribute is set,
        otherwise an empty dict.
    """
    if exc.errno is not None:
        return {"errno": exc.errno}
    return {}


# ---------------------------------------------------------------------------
# Classification: exception type -> DisconnectType mapping
# ---------------------------------------------------------------------------

# Classification order matters: more-specific types must be checked
# before their parent classes.
#
#   asyncio.IncompleteReadError  (EOF)         -- before EOFError
#   EOFError                     (EOF)
#   BrokenPipeError              (BROKEN_PIPE) -- before OSError
#   ConnectionResetError         (CONN_RESET)  -- before OSError
#   asyncio.TimeoutError         (TIMEOUT)     -- standalone
#   OSError                      (OS_ERROR)    -- catch-all for socket errors
#   *                            (UNKNOWN)     -- everything else


def _classify_incomplete_read(
    exc: asyncio.IncompleteReadError,
    client_id: str,
    timestamp: str,
) -> DisconnectEvent:
    """Classify an asyncio.IncompleteReadError as an EOF disconnect."""
    return DisconnectEvent(
        disconnect_type=DisconnectType.EOF,
        client_id=client_id,
        timestamp=timestamp,
        reason=_build_reason(exc),
        metadata={
            "exception_class": _exc_class_name(exc),
            "bytes_received": len(exc.partial),
            "bytes_expected": exc.expected,
        },
    )


def _classify_eof_error(
    exc: EOFError,
    client_id: str,
    timestamp: str,
) -> DisconnectEvent:
    """Classify an EOFError as an EOF disconnect."""
    return DisconnectEvent(
        disconnect_type=DisconnectType.EOF,
        client_id=client_id,
        timestamp=timestamp,
        reason=_build_reason(exc),
        metadata={
            "exception_class": _exc_class_name(exc),
        },
    )


def _classify_broken_pipe(
    exc: BrokenPipeError,
    client_id: str,
    timestamp: str,
) -> DisconnectEvent:
    """Classify a BrokenPipeError as a BROKEN_PIPE disconnect."""
    meta: dict[str, Any] = {"exception_class": _exc_class_name(exc)}
    meta.update(_extract_errno(exc))
    return DisconnectEvent(
        disconnect_type=DisconnectType.BROKEN_PIPE,
        client_id=client_id,
        timestamp=timestamp,
        reason=_build_reason(exc),
        metadata=meta,
    )


def _classify_connection_reset(
    exc: ConnectionResetError,
    client_id: str,
    timestamp: str,
) -> DisconnectEvent:
    """Classify a ConnectionResetError as a CONNECTION_RESET disconnect."""
    meta: dict[str, Any] = {"exception_class": _exc_class_name(exc)}
    meta.update(_extract_errno(exc))
    return DisconnectEvent(
        disconnect_type=DisconnectType.CONNECTION_RESET,
        client_id=client_id,
        timestamp=timestamp,
        reason=_build_reason(exc),
        metadata=meta,
    )


def _classify_timeout(
    exc: TimeoutError,
    client_id: str,
    timestamp: str,
) -> DisconnectEvent:
    """Classify a TimeoutError as a SOCKET_TIMEOUT disconnect."""
    return DisconnectEvent(
        disconnect_type=DisconnectType.SOCKET_TIMEOUT,
        client_id=client_id,
        timestamp=timestamp,
        reason=_build_reason(exc),
        metadata={
            "exception_class": _exc_class_name(exc),
        },
    )


def _classify_os_error(
    exc: OSError,
    client_id: str,
    timestamp: str,
) -> DisconnectEvent:
    """Classify a generic OSError as an OS_ERROR disconnect."""
    meta: dict[str, Any] = {"exception_class": _exc_class_name(exc)}
    meta.update(_extract_errno(exc))
    return DisconnectEvent(
        disconnect_type=DisconnectType.OS_ERROR,
        client_id=client_id,
        timestamp=timestamp,
        reason=_build_reason(exc),
        metadata=meta,
    )


def _classify_unknown(
    exc: Exception,
    client_id: str,
    timestamp: str,
) -> DisconnectEvent:
    """Classify an unknown exception as an UNKNOWN disconnect."""
    return DisconnectEvent(
        disconnect_type=DisconnectType.UNKNOWN,
        client_id=client_id,
        timestamp=timestamp,
        reason=_build_reason(exc),
        metadata={
            "exception_class": _exc_class_name(exc),
        },
    )


# ---------------------------------------------------------------------------
# Public API: classify_disconnect
# ---------------------------------------------------------------------------


def classify_disconnect(
    *,
    exception: Exception,
    client_id: str,
    timestamp: str | None = None,
) -> DisconnectEvent:
    """Classify an exception into a structured DisconnectEvent.

    Inspects the exception type in subclass-specificity order and
    delegates to the appropriate classifier. If no specific match is
    found, falls back to ``UNKNOWN``.

    The ordering is critical:

    1. ``asyncio.IncompleteReadError`` (subclass of ``EOFError``)
    2. ``EOFError``
    3. ``BrokenPipeError`` (subclass of ``OSError``)
    4. ``ConnectionResetError`` (subclass of ``OSError``)
    5. ``TimeoutError`` / ``asyncio.TimeoutError``
    6. ``OSError`` (catch-all for socket errors)
    7. Everything else -> ``UNKNOWN``

    Args:
        exception: The exception that triggered the disconnect.
        client_id: Unique identifier of the affected client.
        timestamp: Optional ISO 8601 timestamp. When None, the
            current UTC time is used.

    Returns:
        Immutable DisconnectEvent with classified type and metadata.
    """
    ts = timestamp if timestamp is not None else _now_iso()

    # Order matters: most-specific subclass first
    if isinstance(exception, asyncio.IncompleteReadError):
        return _classify_incomplete_read(exception, client_id, ts)

    if isinstance(exception, EOFError):
        return _classify_eof_error(exception, client_id, ts)

    if isinstance(exception, BrokenPipeError):
        return _classify_broken_pipe(exception, client_id, ts)

    if isinstance(exception, ConnectionResetError):
        return _classify_connection_reset(exception, client_id, ts)

    # On Python 3.11+ asyncio.TimeoutError is TimeoutError.
    # On Python 3.10 they differ, so check both for portability.
    if isinstance(exception, (asyncio.TimeoutError, TimeoutError)):
        return _classify_timeout(exception, client_id, ts)

    if isinstance(exception, OSError):
        return _classify_os_error(exception, client_id, ts)

    return _classify_unknown(exception, client_id, ts)
