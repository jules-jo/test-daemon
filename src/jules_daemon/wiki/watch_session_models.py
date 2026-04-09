"""Immutable data models for watch-mode session metadata.

Tracks active watchers (CLI clients subscribed to SSH output streaming)
and stream state (per-job broadcasting status). All models are frozen
dataclasses -- state transitions produce new instances, never mutate
existing ones.

Models:
    WatcherStatus   -- Lifecycle states for a watcher subscription
    StreamStatus    -- Lifecycle states for a job output stream
    WatcherRecord   -- Per-watcher metadata (client, job, progress)
    StreamRecord    -- Per-job stream metadata (buffer, publishing stats)
    WatchSessionSnapshot -- Top-level snapshot of all watchers and streams
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WatcherStatus(Enum):
    """Lifecycle states for a watcher subscription.

    Values:
        ACTIVE:        Client is connected and receiving output.
        DISCONNECTED:  Client disconnected but stream may still be live.
        COMPLETED:     Watcher finished (job ended or client unsubscribed).
    """

    ACTIVE = "active"
    DISCONNECTED = "disconnected"
    COMPLETED = "completed"


class StreamStatus(Enum):
    """Lifecycle states for a job output stream.

    Values:
        IDLE:  Job registered but no output yet.
        LIVE:  Job actively producing output.
        ENDED: Job completed; stream terminated.
    """

    IDLE = "idle"
    LIVE = "live"
    ENDED = "ended"


# ---------------------------------------------------------------------------
# WatcherRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatcherRecord:
    """Per-watcher metadata for a CLI client subscribed to SSH output.

    Tracks the subscription identity, progress (sequence numbers and
    line counts), and connection state. Immutable -- transitions produce
    new instances via the with_* methods.

    Attributes:
        watcher_id:    Unique identifier for this watcher subscription.
        client_id:     IPC client that created the subscription.
        job_id:        Job being watched.
        subscriber_id: Broadcaster subscriber handle ID.
        connected_at:  UTC timestamp when the watcher connected.
        status:        Current watcher lifecycle state.
        last_sequence: Last sequence number received by this watcher.
        lines_received: Total number of output lines received.
    """

    watcher_id: str
    client_id: str
    job_id: str
    subscriber_id: str
    connected_at: datetime
    status: WatcherStatus = WatcherStatus.ACTIVE
    last_sequence: int = 0
    lines_received: int = 0

    def __post_init__(self) -> None:
        if not self.watcher_id:
            raise ValueError("watcher_id must not be empty")
        if not self.client_id:
            raise ValueError("client_id must not be empty")
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if not self.subscriber_id:
            raise ValueError("subscriber_id must not be empty")
        if self.last_sequence < 0:
            raise ValueError("last_sequence must not be negative")
        if self.lines_received < 0:
            raise ValueError("lines_received must not be negative")

    def with_status(self, status: WatcherStatus) -> WatcherRecord:
        """Return a new record with the given status."""
        return replace(self, status=status)

    def with_progress(
        self,
        *,
        last_sequence: int,
        lines_received: int,
    ) -> WatcherRecord:
        """Return a new record with updated progress counters."""
        return replace(
            self,
            last_sequence=last_sequence,
            lines_received=lines_received,
        )


# ---------------------------------------------------------------------------
# StreamRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamRecord:
    """Per-job stream metadata for output broadcasting.

    Tracks the broadcasting status, buffer state, and publishing
    statistics for a single job. Immutable -- transitions produce
    new instances via the with_* methods.

    Attributes:
        job_id:               Job identifier being streamed.
        status:               Current stream lifecycle state.
        buffer_size:          Number of lines in the ring buffer.
        total_lines_published: Total lines published since stream start.
        subscriber_count:     Number of active subscribers.
        last_publish_at:      UTC timestamp of the most recent publish.
    """

    job_id: str
    status: StreamStatus = StreamStatus.IDLE
    buffer_size: int = 0
    total_lines_published: int = 0
    subscriber_count: int = 0
    last_publish_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if self.buffer_size < 0:
            raise ValueError("buffer_size must not be negative")
        if self.total_lines_published < 0:
            raise ValueError("total_lines_published must not be negative")
        if self.subscriber_count < 0:
            raise ValueError("subscriber_count must not be negative")

    def with_status(self, status: StreamStatus) -> StreamRecord:
        """Return a new record with the given status."""
        return replace(self, status=status)

    def with_publish_update(
        self,
        *,
        total_lines: int,
        buffer_size: int,
        last_publish_at: datetime,
    ) -> StreamRecord:
        """Return a new record with updated publishing stats."""
        return replace(
            self,
            total_lines_published=total_lines,
            buffer_size=buffer_size,
            last_publish_at=last_publish_at,
        )

    def with_subscriber_count(self, count: int) -> StreamRecord:
        """Return a new record with the given subscriber count."""
        return replace(self, subscriber_count=count)


# ---------------------------------------------------------------------------
# WatchSessionSnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchSessionSnapshot:
    """Top-level snapshot of all watch-mode session metadata.

    Contains the complete set of active watchers and stream states.
    This is the record that gets serialized to the wiki file.
    Immutable -- transitions produce new instances via the with_* methods.

    Attributes:
        watchers:    Tuple of all watcher records (active and recent).
        streams:     Tuple of all stream records (live and recent).
        snapshot_at: UTC timestamp when this snapshot was created.
        daemon_pid:  PID of the daemon managing these sessions.
    """

    watchers: tuple[WatcherRecord, ...] = ()
    streams: tuple[StreamRecord, ...] = ()
    snapshot_at: datetime = field(default_factory=_now_utc)
    daemon_pid: int | None = None

    # -- Watcher transitions ---------------------------------------------------

    def with_watcher_added(self, watcher: WatcherRecord) -> WatchSessionSnapshot:
        """Return a new snapshot with the watcher appended."""
        return replace(
            self,
            watchers=self.watchers + (watcher,),
            snapshot_at=_now_utc(),
        )

    def with_watcher_removed(self, watcher_id: str) -> WatchSessionSnapshot:
        """Return a new snapshot with the watcher removed by ID.

        If no watcher matches the given ID, returns a new snapshot
        with identical watchers (safe no-op).
        """
        filtered = tuple(w for w in self.watchers if w.watcher_id != watcher_id)
        return replace(
            self,
            watchers=filtered,
            snapshot_at=_now_utc(),
        )

    # -- Stream transitions ----------------------------------------------------

    def with_stream_added(self, stream: StreamRecord) -> WatchSessionSnapshot:
        """Return a new snapshot with the stream appended."""
        return replace(
            self,
            streams=self.streams + (stream,),
            snapshot_at=_now_utc(),
        )

    def with_stream_removed(self, job_id: str) -> WatchSessionSnapshot:
        """Return a new snapshot with the stream removed by job ID."""
        filtered = tuple(s for s in self.streams if s.job_id != job_id)
        return replace(
            self,
            streams=filtered,
            snapshot_at=_now_utc(),
        )

    def with_stream_updated(self, stream: StreamRecord) -> WatchSessionSnapshot:
        """Return a new snapshot with the stream replaced by job ID.

        If no existing stream matches, the stream is appended.
        """
        found = False
        updated_streams: list[StreamRecord] = []
        for existing in self.streams:
            if existing.job_id == stream.job_id:
                updated_streams.append(stream)
                found = True
            else:
                updated_streams.append(existing)

        if not found:
            updated_streams.append(stream)

        return replace(
            self,
            streams=tuple(updated_streams),
            snapshot_at=_now_utc(),
        )

    # -- Computed properties ---------------------------------------------------

    @property
    def active_watcher_count(self) -> int:
        """Number of watchers with ACTIVE status."""
        return sum(1 for w in self.watchers if w.status == WatcherStatus.ACTIVE)

    @property
    def live_stream_count(self) -> int:
        """Number of streams with LIVE status."""
        return sum(1 for s in self.streams if s.status == StreamStatus.LIVE)

    # -- Lookup methods --------------------------------------------------------

    def find_watcher(self, watcher_id: str) -> WatcherRecord | None:
        """Find a watcher by ID, or None if not found."""
        for w in self.watchers:
            if w.watcher_id == watcher_id:
                return w
        return None

    def find_stream(self, job_id: str) -> StreamRecord | None:
        """Find a stream by job ID, or None if not found."""
        for s in self.streams:
            if s.job_id == job_id:
                return s
        return None

    def watchers_for_job(self, job_id: str) -> tuple[WatcherRecord, ...]:
        """Return all watchers subscribed to a given job."""
        return tuple(w for w in self.watchers if w.job_id == job_id)
