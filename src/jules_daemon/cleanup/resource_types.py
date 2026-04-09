"""Protocols and immutable data types for resource cleanup operations.

Defines the type contracts and value objects used throughout the cleanup
subsystem. All result types are frozen dataclasses for immutability and
safe sharing across async boundaries.

ResourceType classifies which kind of resource was cleaned up (SSH channel,
socket writer, or I/O buffer). CleanupResult captures the outcome of a
single resource cleanup operation. CleanupSummary aggregates multiple
CleanupResult instances for a single disconnect event.

The CleanableResource protocol defines the minimal interface that any
resource must satisfy to be registered with the DisconnectCleanupHandler.

Usage::

    result = CleanupResult(
        resource_id="ssh-chan-1",
        resource_type=ResourceType.SSH_CHANNEL,
        success=True,
        error=None,
        bytes_flushed=1024,
    )

    summary = CleanupSummary(
        event_id="disconnect-abc",
        results=(result,),
    )
    assert summary.all_succeeded
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable

__all__ = [
    "CleanableResource",
    "CleanupResult",
    "CleanupSummary",
    "MAX_FLUSH_BYTES",
    "ResourceType",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FLUSH_BYTES: int = 65536
"""Maximum bytes to drain from a single stream during cleanup."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# ResourceType enum
# ---------------------------------------------------------------------------


class ResourceType(Enum):
    """Classification of a resource managed by the cleanup handler.

    Values:
        SSH_CHANNEL: An SSH channel (paramiko Channel or equivalent).
            Cleanup involves flushing pending I/O and closing the channel.
        SOCKET_WRITER: An asyncio.StreamWriter for IPC sockets.
            Cleanup involves draining pending writes and closing.
        IO_BUFFER: A generic I/O buffer that holds pending data.
            Cleanup involves flushing the buffer contents.
    """

    SSH_CHANNEL = "ssh_channel"
    SOCKET_WRITER = "socket_writer"
    IO_BUFFER = "io_buffer"


# ---------------------------------------------------------------------------
# Protocol: CleanableResource
# ---------------------------------------------------------------------------


@runtime_checkable
class CleanableResource(Protocol):
    """Protocol for resources that support deterministic cleanup.

    Any object registered with the DisconnectCleanupHandler must satisfy
    this protocol (or be wrapped in an adapter that does). The handler
    calls ``close()`` during cleanup, and checks ``is_closed`` to skip
    already-released resources.
    """

    def close(self) -> None:
        """Release the resource.

        Must be idempotent -- calling close() on an already-closed
        resource must not raise.
        """
        ...

    @property
    def is_closed(self) -> bool:
        """Return True if the resource has already been released."""
        ...


# ---------------------------------------------------------------------------
# CleanupResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanupResult:
    """Immutable result of a single resource cleanup operation.

    Captures whether the cleanup succeeded, how many bytes of pending
    I/O were flushed, and any error that occurred.

    Attributes:
        resource_id:   Unique identifier for the cleaned resource.
        resource_type: Classification of the resource (SSH, socket, buffer).
        success:       True if cleanup completed without errors.
        error:         Human-readable error description (None on success).
        bytes_flushed: Number of pending I/O bytes drained during cleanup.
        timestamp:     UTC datetime when cleanup completed.
    """

    resource_id: str
    resource_type: ResourceType
    success: bool
    error: str | None
    bytes_flushed: int
    timestamp: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource_id must not be empty")
        if self.bytes_flushed < 0:
            raise ValueError(
                f"bytes_flushed must not be negative, got {self.bytes_flushed}"
            )


# ---------------------------------------------------------------------------
# CleanupSummary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanupSummary:
    """Immutable aggregate of cleanup results for one disconnect event.

    Provides computed properties for quick status checks (total, succeeded,
    failed counts) and the total bytes flushed across all resources.

    Attributes:
        event_id: Identifier for the disconnect event that triggered cleanup.
        results:  Ordered tuple of individual CleanupResult instances.
        timestamp: UTC datetime when the summary was assembled.
    """

    event_id: str
    results: tuple[CleanupResult, ...]
    timestamp: datetime = field(default_factory=_now_utc)

    @property
    def total_resources(self) -> int:
        """Number of resources that were cleaned up."""
        return len(self.results)

    @property
    def successful(self) -> int:
        """Number of resources that cleaned up successfully."""
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        """Number of resources where cleanup encountered errors."""
        return sum(1 for r in self.results if not r.success)

    @property
    def all_succeeded(self) -> bool:
        """True if all cleanups succeeded (or there were no resources)."""
        return all(r.success for r in self.results)

    @property
    def total_bytes_flushed(self) -> int:
        """Total bytes flushed across all resources during cleanup."""
        return sum(r.bytes_flushed for r in self.results)
