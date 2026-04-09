"""Server-side output broadcaster for SSH job output streaming.

Buffers recent output lines per running job and fans out to multiple
async subscriber queues. Designed for the daemon's monitoring layer
where SSH output must be streamed to zero or more connected CLI clients
without blocking the producer (the SSH polling loop).

Key design decisions:

- **Per-job ring buffer**: Each registered job maintains a bounded deque
  of recent OutputLine objects. Late-joining subscribers can request a
  replay of the buffer to catch up.

- **Fan-out via asyncio.Queue**: Each subscriber gets an independent
  asyncio.Queue. Publishing to a job enqueues the OutputLine into every
  active subscriber queue for that job.

- **Non-blocking publish**: If a subscriber queue is full (slow consumer),
  the oldest entry is evicted so the producer never blocks. This trades
  completeness for liveness -- the ring buffer still holds the full
  recent history for replay.

- **End-of-stream sentinel**: When a job is unregistered, an OutputLine
  with ``is_end=True`` is pushed to all active subscriber queues so
  consumers know the stream has ended.

- **Immutable data**: OutputLine, BroadcasterConfig, and SubscriberHandle
  are frozen dataclasses. The broadcaster itself is mutable (it manages
  live state), but all data flowing through it is immutable.

Usage::

    broadcaster = JobOutputBroadcaster()

    broadcaster.register_job("job-123")
    handle = broadcaster.subscribe("job-123")

    broadcaster.publish("job-123", "PASSED test_foo")
    line = await broadcaster.receive(handle, timeout=1.0)

    broadcaster.unsubscribe(handle)
    broadcaster.unregister_job("job-123")
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator

__all__ = [
    "BroadcasterConfig",
    "JobOutputBroadcaster",
    "OutputLine",
    "SubscriberHandle",
]

logger = logging.getLogger(__name__)

_DEFAULT_BUFFER_SIZE = 1000
_DEFAULT_SUBSCRIBER_QUEUE_SIZE = 500


# ---------------------------------------------------------------------------
# Immutable data types
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class OutputLine:
    """A single line of output from an SSH job.

    Attributes:
        job_id:    Identifier of the job that produced this line.
        line:      The text content of the output line.
        sequence:  Monotonically increasing counter within the job.
        timestamp: ISO 8601 UTC timestamp of when the line was captured.
        is_end:    True if this is an end-of-stream sentinel.
    """

    job_id: str
    line: str
    sequence: int
    timestamp: str = field(default="")
    is_end: bool = False

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if self.sequence < 0:
            raise ValueError("sequence must not be negative")
        if not self.timestamp:
            object.__setattr__(self, "timestamp", _now_iso())


@dataclass(frozen=True)
class BroadcasterConfig:
    """Configuration for the output broadcaster.

    Attributes:
        buffer_size:          Maximum number of recent lines to retain
                              per job in the ring buffer.
        subscriber_queue_size: Maximum number of lines that can be queued
                               per subscriber before backpressure kicks in.
    """

    buffer_size: int = _DEFAULT_BUFFER_SIZE
    subscriber_queue_size: int = _DEFAULT_SUBSCRIBER_QUEUE_SIZE

    def __post_init__(self) -> None:
        if self.buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        if self.subscriber_queue_size < 1:
            raise ValueError("subscriber_queue_size must be positive")


@dataclass(frozen=True)
class SubscriberHandle:
    """Immutable handle identifying a subscriber queue.

    Returned by ``subscribe()`` and used with ``receive()``,
    ``unsubscribe()``, ``replay_buffer()``, and ``iter_lines()``.

    Attributes:
        subscriber_id: Unique identifier for this subscriber.
        job_id:        The job this subscriber is attached to.
    """

    subscriber_id: str
    job_id: str


# ---------------------------------------------------------------------------
# Internal per-job state
# ---------------------------------------------------------------------------


class _JobState:
    """Internal per-job state: ring buffer + subscriber registry.

    Not part of the public API. Managed exclusively by JobOutputBroadcaster.
    """

    __slots__ = ("_buffer", "_subscribers", "_sequence_counter")

    def __init__(self, buffer_size: int) -> None:
        self._buffer: deque[OutputLine] = deque(maxlen=buffer_size)
        self._subscribers: dict[str, asyncio.Queue[OutputLine]] = {}
        self._sequence_counter: int = 0

    @property
    def buffer(self) -> tuple[OutputLine, ...]:
        """Snapshot of the current ring buffer contents."""
        return tuple(self._buffer)

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers for this job."""
        return len(self._subscribers)

    @property
    def next_sequence(self) -> int:
        """Return the next sequence number and advance the counter."""
        seq = self._sequence_counter
        self._sequence_counter += 1
        return seq

    def add_subscriber(
        self,
        subscriber_id: str,
        queue_size: int,
    ) -> asyncio.Queue[OutputLine]:
        """Add a subscriber queue and return it."""
        queue: asyncio.Queue[OutputLine] = asyncio.Queue(maxsize=queue_size)
        self._subscribers[subscriber_id] = queue
        return queue

    def remove_subscriber(self, subscriber_id: str) -> None:
        """Remove a subscriber queue by ID. Safe to call with unknown IDs."""
        self._subscribers.pop(subscriber_id, None)

    def get_queue(self, subscriber_id: str) -> asyncio.Queue[OutputLine] | None:
        """Look up a subscriber queue. Returns None if not found."""
        return self._subscribers.get(subscriber_id)

    def append_to_buffer(self, output_line: OutputLine) -> None:
        """Append a line to the ring buffer (oldest evicted if full)."""
        self._buffer.append(output_line)

    def fan_out(self, output_line: OutputLine) -> None:
        """Push an output line to every subscriber queue.

        If a queue is full, the oldest item is evicted to make room
        so the producer never blocks.
        """
        for sub_id, queue in self._subscribers.items():
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass  # pragma: no cover -- race guard
                logger.debug(
                    "Subscriber %s queue full; evicted oldest entry",
                    sub_id,
                )
            try:
                queue.put_nowait(output_line)
            except asyncio.QueueFull:
                pass  # pragma: no cover -- should not happen after eviction

    def drain_all_subscribers(self) -> list[str]:
        """Return all subscriber IDs and clear the registry."""
        ids = list(self._subscribers.keys())
        self._subscribers.clear()
        return ids


# ---------------------------------------------------------------------------
# JobOutputBroadcaster
# ---------------------------------------------------------------------------


class JobOutputBroadcaster:
    """Broadcasts SSH output lines to multiple subscriber queues per job.

    The broadcaster is the bridge between the SSH monitoring loop (producer)
    and connected CLI clients (consumers). It maintains per-job ring buffers
    and per-subscriber async queues.

    Thread safety: designed for single-threaded async use within one event
    loop. Does not use locks.

    Usage::

        broadcaster = JobOutputBroadcaster()

        # Register a job when SSH monitoring starts
        broadcaster.register_job("job-123")

        # CLI client subscribes
        handle = broadcaster.subscribe("job-123")

        # SSH monitoring loop publishes lines
        broadcaster.publish("job-123", "PASSED test_foo")

        # CLI client receives lines
        line = await broadcaster.receive(handle, timeout=1.0)

        # CLI client disconnects
        broadcaster.unsubscribe(handle)

        # SSH job finishes
        broadcaster.unregister_job("job-123")
    """

    def __init__(self, config: BroadcasterConfig | None = None) -> None:
        self._config = config or BroadcasterConfig()
        self._jobs: dict[str, _JobState] = {}
        # Queues detached by unregister_job but not yet consumed by receive.
        # Keyed by (job_id, subscriber_id).
        self._detached_queues: dict[
            tuple[str, str], asyncio.Queue[OutputLine]
        ] = {}

    # -- Job lifecycle -------------------------------------------------------

    def register_job(self, job_id: str) -> None:
        """Register a job for output broadcasting.

        Idempotent: re-registering an existing job is a safe no-op.

        Args:
            job_id: Unique identifier for the job.
        """
        if job_id in self._jobs:
            logger.debug("Job %s already registered; ignoring", job_id)
            return
        self._jobs[job_id] = _JobState(buffer_size=self._config.buffer_size)
        logger.debug("Registered job %s for broadcasting", job_id)

    def unregister_job(self, job_id: str) -> None:
        """Unregister a job and signal end-of-stream to all subscribers.

        Sends an OutputLine with ``is_end=True`` to every active subscriber
        so they know the stream has ended, then removes the job state.

        Idempotent: unregistering an unknown job is a safe no-op.

        Args:
            job_id: The job to unregister.
        """
        state = self._jobs.pop(job_id, None)
        if state is None:
            return

        # Send end sentinel to all active subscribers
        end_line = OutputLine(
            job_id=job_id,
            line="",
            sequence=state.next_sequence,
            is_end=True,
        )
        state.fan_out(end_line)

        # Preserve subscriber queues so receive() can still drain them.
        # Each queue now contains the end sentinel as the last item.
        for sub_id in list(state._subscribers):
            queue = state.get_queue(sub_id)
            if queue is not None:
                self._detached_queues[(job_id, sub_id)] = queue

        sub_ids = state.drain_all_subscribers()
        logger.debug(
            "Unregistered job %s; notified %d subscriber(s)",
            job_id,
            len(sub_ids),
        )

    def is_registered(self, job_id: str) -> bool:
        """Check whether a job is currently registered.

        Args:
            job_id: The job to check.

        Returns:
            True if the job is registered.
        """
        return job_id in self._jobs

    def registered_job_ids(self) -> frozenset[str]:
        """Return the set of currently registered job IDs.

        Returns:
            Frozen set of job IDs.
        """
        return frozenset(self._jobs.keys())

    # -- Publishing ----------------------------------------------------------

    def publish(self, job_id: str, line_text: str) -> OutputLine:
        """Publish a line of output for a job.

        Appends the line to the job's ring buffer and fans it out to
        all active subscriber queues. If a subscriber queue is full,
        the oldest entry is evicted (non-blocking).

        Args:
            job_id:    The job producing the output.
            line_text: The text content of the output line.

        Returns:
            The OutputLine that was created and published.

        Raises:
            ValueError: If the job is not registered.
        """
        state = self._jobs.get(job_id)
        if state is None:
            raise ValueError(f"Job {job_id!r} is not registered")

        output_line = OutputLine(
            job_id=job_id,
            line=line_text,
            sequence=state.next_sequence,
        )
        state.append_to_buffer(output_line)
        state.fan_out(output_line)

        logger.debug(
            "Published line seq=%d to job %s (%d subscribers)",
            output_line.sequence,
            job_id,
            state.subscriber_count,
        )
        return output_line

    # -- Buffer access -------------------------------------------------------

    def get_buffer(self, job_id: str) -> tuple[OutputLine, ...]:
        """Return a snapshot of the ring buffer for a job.

        Args:
            job_id: The job whose buffer to retrieve.

        Returns:
            Tuple of OutputLine objects (oldest first), or empty tuple
            if the job is not registered.
        """
        state = self._jobs.get(job_id)
        if state is None:
            return ()
        return state.buffer

    # -- Subscription management ---------------------------------------------

    def subscribe(self, job_id: str) -> SubscriberHandle:
        """Create a new subscriber queue for a job.

        The subscriber will receive all lines published after this call.
        To also receive previously buffered lines, call ``replay_buffer()``
        after subscribing.

        Args:
            job_id: The job to subscribe to.

        Returns:
            A SubscriberHandle to use with ``receive()`` and ``unsubscribe()``.

        Raises:
            ValueError: If the job is not registered.
        """
        state = self._jobs.get(job_id)
        if state is None:
            raise ValueError(f"Job {job_id!r} is not registered")

        subscriber_id = f"sub-{uuid.uuid4().hex[:12]}"
        state.add_subscriber(
            subscriber_id=subscriber_id,
            queue_size=self._config.subscriber_queue_size,
        )

        handle = SubscriberHandle(
            subscriber_id=subscriber_id,
            job_id=job_id,
        )
        logger.debug(
            "Subscriber %s attached to job %s (total: %d)",
            subscriber_id,
            job_id,
            state.subscriber_count,
        )
        return handle

    def unsubscribe(self, handle: SubscriberHandle) -> None:
        """Remove a subscriber from a job.

        Idempotent: unsubscribing an already-removed handle is a safe no-op.

        Args:
            handle: The subscriber handle returned by ``subscribe()``.
        """
        state = self._jobs.get(handle.job_id)
        if state is None:
            return
        state.remove_subscriber(handle.subscriber_id)
        logger.debug(
            "Subscriber %s detached from job %s (remaining: %d)",
            handle.subscriber_id,
            handle.job_id,
            state.subscriber_count,
        )

    def subscriber_count(self, job_id: str) -> int:
        """Return the number of active subscribers for a job.

        Args:
            job_id: The job to query.

        Returns:
            Number of active subscribers, or 0 if the job is not registered.
        """
        state = self._jobs.get(job_id)
        if state is None:
            return 0
        return state.subscriber_count

    # -- Receiving -----------------------------------------------------------

    async def receive(
        self,
        handle: SubscriberHandle,
        *,
        timeout: float = 5.0,
    ) -> OutputLine | None:
        """Receive the next output line from a subscriber queue.

        Blocks (with timeout) until a line is available or the timeout
        expires. Returns None if the timeout expires with no data.

        Args:
            handle:  The subscriber handle returned by ``subscribe()``.
            timeout: Maximum seconds to wait for a line.

        Returns:
            The next OutputLine, or None on timeout.

        Raises:
            ValueError: If the subscriber handle is not found.
        """
        queue = self._resolve_queue(handle)
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # -- Buffer replay -------------------------------------------------------

    async def replay_buffer(
        self,
        handle: SubscriberHandle,
    ) -> tuple[OutputLine, ...]:
        """Replay the current ring buffer contents into a subscriber queue.

        This is used for catch-up when a CLI client connects to an
        already-running job. The buffered lines are enqueued into the
        subscriber's queue in order.

        Args:
            handle: The subscriber handle to replay into.

        Returns:
            Tuple of the replayed OutputLine objects.

        Raises:
            ValueError: If the subscriber handle is not found.
        """
        queue = self._resolve_queue(handle)
        state = self._jobs.get(handle.job_id)
        if state is None:
            return ()

        buffered = state.buffer
        for output_line in buffered:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass  # pragma: no cover
            try:
                queue.put_nowait(output_line)
            except asyncio.QueueFull:
                pass  # pragma: no cover
        return buffered

    # -- Async iteration -----------------------------------------------------

    async def iter_lines(
        self,
        handle: SubscriberHandle,
    ) -> AsyncIterator[OutputLine]:
        """Async iterator that yields output lines until end-of-stream.

        Yields OutputLine objects as they arrive. Terminates when an
        end-of-stream sentinel (``is_end=True``) is received.

        Args:
            handle: The subscriber handle to iterate over.

        Yields:
            OutputLine objects (never the end sentinel itself).

        Raises:
            ValueError: If the subscriber handle is not found at start.
        """
        queue = self._resolve_queue(handle)
        while True:
            line = await queue.get()
            if line.is_end:
                return
            yield line

    # -- Internal helpers ----------------------------------------------------

    def _resolve_queue(
        self,
        handle: SubscriberHandle,
    ) -> asyncio.Queue[OutputLine]:
        """Look up the queue for a subscriber handle.

        Checks both the job state and the subscriber within it.

        Args:
            handle: The subscriber handle.

        Returns:
            The subscriber's asyncio.Queue.

        Raises:
            ValueError: If the job or subscriber is not found.
        """
        # Check active job state first
        state = self._jobs.get(handle.job_id)
        if state is not None:
            queue = state.get_queue(handle.subscriber_id)
            if queue is not None:
                return queue

        # Check detached queues (from unregistered jobs)
        detached_key = (handle.job_id, handle.subscriber_id)
        detached_queue = self._detached_queues.get(detached_key)
        if detached_queue is not None:
            return detached_queue

        raise ValueError(
            f"Subscriber {handle.subscriber_id!r} for job "
            f"{handle.job_id!r} not found"
        )
