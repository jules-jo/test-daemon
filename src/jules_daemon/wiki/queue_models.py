"""Immutable data models for the command queue.

All models are frozen dataclasses -- state transitions produce new instances,
never mutate existing ones. Each queued command becomes a wiki file in
pages/daemon/queue/ with YAML frontmatter and a markdown body.

Ordering: commands are sorted by (priority descending, sequence ascending).
Priority tiers allow urgent commands to jump ahead in the queue while
maintaining FIFO order within the same priority level.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class QueueStatus(Enum):
    """Lifecycle states for a queued command."""

    QUEUED = "queued"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True if this status represents a terminal state."""
        return self in (
            QueueStatus.COMPLETED,
            QueueStatus.FAILED,
            QueueStatus.CANCELLED,
        )

    @property
    def is_pending(self) -> bool:
        """True if this command is waiting in the queue."""
        return self == QueueStatus.QUEUED


class QueuePriority(Enum):
    """Priority levels for queued commands.

    Higher numeric values indicate higher priority. Commands are dequeued
    in order of (priority descending, sequence ascending).
    """

    NORMAL = 10
    HIGH = 20
    URGENT = 30


def _generate_queue_id() -> str:
    """Generate a new unique queue entry identifier."""
    return str(uuid.uuid4())


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _next_sequence() -> int:
    """Generate a default sequence number based on timestamp microseconds.

    This provides a monotonically increasing default when no explicit
    sequence is given. The CommandQueue class overrides this with its
    own atomic counter for proper ordering.
    """
    now = datetime.now(timezone.utc)
    return int(now.timestamp() * 1_000_000) % 10_000_000


@dataclass(frozen=True)
class QueuedCommand:
    """A single command entry in the queue.

    Each instance maps to one wiki file in pages/daemon/queue/.
    File naming: {sequence:06d}-{queue_id}.md

    Attributes:
        queue_id: Unique identifier for this queue entry.
        sequence: Monotonic ordering number (lower = earlier in queue).
        natural_language: The user's natural-language command text.
        status: Current lifecycle state.
        priority: Priority tier for ordering.
        ssh_host: Optional target SSH hostname.
        ssh_user: Optional target SSH username.
        ssh_port: Optional target SSH port (default 22).
        queued_at: Timestamp when the command was enqueued.
        started_at: Timestamp when the command became active (None if not yet).
        completed_at: Timestamp when the command reached a terminal state.
        error: Error message if the command failed.
    """

    natural_language: str
    queue_id: str = field(default_factory=_generate_queue_id)
    sequence: int = field(default_factory=_next_sequence)
    status: QueueStatus = QueueStatus.QUEUED
    priority: QueuePriority = QueuePriority.NORMAL
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: int = 22
    queued_at: datetime = field(default_factory=_now_utc)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.natural_language.strip():
            raise ValueError("natural_language must not be empty")
        if self.sequence < 1:
            raise ValueError(
                f"sequence must be positive, got {self.sequence}"
            )

    # -- State transition methods (each returns a new frozen instance) --

    def with_activated(self) -> QueuedCommand:
        """Return a new command marked as active (being executed)."""
        return replace(
            self,
            status=QueueStatus.ACTIVE,
            started_at=_now_utc(),
        )

    def with_completed(self) -> QueuedCommand:
        """Return a new command marked as completed."""
        return replace(
            self,
            status=QueueStatus.COMPLETED,
            completed_at=_now_utc(),
        )

    def with_failed(self, error: str) -> QueuedCommand:
        """Return a new command marked as failed with an error message."""
        return replace(
            self,
            status=QueueStatus.FAILED,
            error=error,
            completed_at=_now_utc(),
        )

    def with_cancelled(self) -> QueuedCommand:
        """Return a new command marked as cancelled."""
        return replace(
            self,
            status=QueueStatus.CANCELLED,
            completed_at=_now_utc(),
        )

    # -- Derived properties --

    @property
    def file_stem(self) -> str:
        """Wiki filename stem: {sequence:06d}-{queue_id} (no extension)."""
        return f"{self.sequence:06d}-{self.queue_id}"

    @property
    def sort_key(self) -> tuple[int, int]:
        """Sort key for queue ordering: (negative priority, sequence).

        Negating the priority value puts higher-priority items first when
        sorting in ascending order. Within the same priority, lower
        sequence numbers come first (FIFO).
        """
        return (-self.priority.value, self.sequence)
