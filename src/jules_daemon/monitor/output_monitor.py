"""OutputMonitor -- real-time line-buffering service for test output streams.

Subscribes to a JobOutputBroadcaster for a given job and runs a background
asyncio task that consumes OutputLine events as they arrive. Lines are
accumulated in a bounded deque (ring buffer) with configurable capacity,
providing O(1) append and automatic eviction of oldest lines when full.

The monitor is the foundation for real-time analysis: downstream consumers
(pattern matchers, anomaly detectors, the agent loop's read_output tool)
tap into the monitor via the ``snapshot()`` method which returns an
immutable ``OutputMonitorSnapshot`` -- a frozen dataclass capturing the
current buffer state, total line count, and lifecycle state.

Lifecycle states mirror the existing PollingLoop pattern::

    IDLE -> RUNNING -> (STOPPING ->) STOPPED
                  \-> ERROR (on unrecoverable failure)

Key design decisions:

- **Subscriber-based consumption**: The monitor subscribes to the
  broadcaster rather than polling. This means zero CPU when idle and
  immediate processing when lines arrive.

- **Ring-buffer semantics**: The internal deque has a configurable maxlen
  (``max_buffer_lines``). When full, oldest lines are silently evicted.
  The ``total_lines_observed`` counter continues incrementing so consumers
  know how much output was produced even if older lines are gone.

- **End-of-stream aware**: When the broadcaster sends an ``is_end=True``
  sentinel (job unregistered), the monitor transitions to STOPPED
  automatically. No explicit stop() call needed for normal completion.

- **Immutable snapshots**: ``snapshot()`` returns a frozen dataclass that
  is safe to pass across async boundaries without synchronization concerns.
  Each call produces a new independent snapshot.

- **Async context manager**: Supports ``async with`` for scoped lifecycle
  management. Ensures stop() is always called even on exception paths.

- **Empty lines skipped**: Lines with empty text content are not buffered
  (they carry no analysis value).

Usage::

    broadcaster = JobOutputBroadcaster()
    broadcaster.register_job("job-123")

    async with OutputMonitor(
        broadcaster=broadcaster,
        job_id="job-123",
    ) as monitor:
        # Lines published to the broadcaster are automatically buffered
        broadcaster.publish("job-123", "PASSED test_login")

        # Tap into the buffer at any time
        snap = monitor.snapshot()
        print(snap.lines, snap.total_lines_observed)

    # After exit, monitor is STOPPED
    final = monitor.snapshot()
    print(final.state)  # OutputMonitorState.STOPPED
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from types import TracebackType

from jules_daemon.monitor.output_broadcaster import (
    JobOutputBroadcaster,
    SubscriberHandle,
)

__all__ = [
    "OutputMonitor",
    "OutputMonitorConfig",
    "OutputMonitorSnapshot",
    "OutputMonitorState",
]

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BUFFER_LINES = 5000
_DEFAULT_STOP_TIMEOUT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputMonitorConfig:
    """Immutable configuration for the OutputMonitor.

    Attributes:
        max_buffer_lines: Maximum number of output lines to retain in the
            ring buffer. When full, oldest lines are silently evicted.
            Must be positive. Default: 5000.
        stop_timeout_seconds: Maximum seconds to wait for the background
            task to complete during stop(). Must be positive. Default: 5.0.
    """

    max_buffer_lines: int = _DEFAULT_MAX_BUFFER_LINES
    stop_timeout_seconds: float = _DEFAULT_STOP_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.max_buffer_lines < 1:
            raise ValueError(
                f"max_buffer_lines must be positive, got {self.max_buffer_lines}"
            )
        if self.stop_timeout_seconds <= 0:
            raise ValueError(
                f"stop_timeout_seconds must be positive, "
                f"got {self.stop_timeout_seconds}"
            )


# ---------------------------------------------------------------------------
# Lifecycle state
# ---------------------------------------------------------------------------


class OutputMonitorState(Enum):
    """Lifecycle states for the OutputMonitor.

    Values:
        IDLE: Monitor created but not yet started.
        RUNNING: Background task is actively consuming output lines.
        STOPPING: Stop requested, waiting for background task to finish.
        STOPPED: Background task has completed (normal or end-of-stream).
        ERROR: An unrecoverable error occurred during monitoring.
    """

    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Snapshot (immutable tap into monitor state)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputMonitorSnapshot:
    """Immutable snapshot of the OutputMonitor's current state.

    Returned by ``OutputMonitor.snapshot()`` for safe, lock-free access
    to the monitor's accumulated output and lifecycle state.

    Attributes:
        job_id: The job being monitored.
        state: Current lifecycle state of the monitor.
        lines: Tuple of buffered output line texts (oldest first).
            May be shorter than total_lines_observed when the ring
            buffer has evicted older entries.
        total_lines_observed: Total number of non-empty lines the
            monitor has seen since start, including evicted ones.
    """

    job_id: str
    state: OutputMonitorState
    lines: tuple[str, ...]
    total_lines_observed: int

    def __post_init__(self) -> None:
        if self.total_lines_observed < 0:
            raise ValueError(
                f"total_lines_observed must not be negative, "
                f"got {self.total_lines_observed}"
            )

    @property
    def line_count(self) -> int:
        """Number of lines currently in the buffer."""
        return len(self.lines)


# ---------------------------------------------------------------------------
# OutputMonitor service
# ---------------------------------------------------------------------------


class OutputMonitor:
    """Real-time line-buffering service for test output streams.

    Subscribes to a ``JobOutputBroadcaster`` for a given job and runs a
    background asyncio task that consumes ``OutputLine`` events. Lines are
    accumulated in a bounded deque and accessible via ``snapshot()``.

    Thread safety:
        This class is NOT thread-safe. All methods must be called from the
        same asyncio event loop.

    Args:
        broadcaster: The job output broadcaster to subscribe to.
        job_id: The job to monitor. Must be registered in the broadcaster
            before ``start()`` is called.
        config: Optional monitor configuration. Uses defaults if None.
    """

    __slots__ = (
        "_broadcaster",
        "_job_id",
        "_config",
        "_state",
        "_task",
        "_handle",
        "_buffer",
        "_total_lines_observed",
    )

    def __init__(
        self,
        *,
        broadcaster: JobOutputBroadcaster,
        job_id: str,
        config: OutputMonitorConfig | None = None,
    ) -> None:
        self._broadcaster = broadcaster
        self._job_id = job_id
        self._config = config or OutputMonitorConfig()
        self._state = OutputMonitorState.IDLE
        self._task: asyncio.Task[None] | None = None
        self._handle: SubscriberHandle | None = None
        self._buffer: deque[str] = deque(maxlen=self._config.max_buffer_lines)
        self._total_lines_observed: int = 0

    # -- Public properties ---------------------------------------------------

    @property
    def state(self) -> OutputMonitorState:
        """Current lifecycle state of the monitor."""
        return self._state

    @property
    def job_id(self) -> str:
        """The job being monitored."""
        return self._job_id

    # -- Snapshot (tap) ------------------------------------------------------

    def snapshot(self) -> OutputMonitorSnapshot:
        """Return an immutable snapshot of the monitor's current state.

        Safe to call at any point in the lifecycle (before start, during
        monitoring, after stop). Each call returns a new independent
        frozen dataclass instance.

        Returns:
            OutputMonitorSnapshot with the current buffer contents,
            total line count, and lifecycle state.
        """
        return OutputMonitorSnapshot(
            job_id=self._job_id,
            state=self._state,
            lines=tuple(self._buffer),
            total_lines_observed=self._total_lines_observed,
        )

    def get_lines(self, *, last_n: int | None = None) -> tuple[str, ...]:
        """Return buffered output lines, optionally limited to the last N.

        Convenience method that avoids constructing a full snapshot when
        only the line text is needed.

        Args:
            last_n: If provided, return only the most recent N lines.
                If None or greater than buffer size, returns all lines.

        Returns:
            Tuple of line text strings (oldest first).
        """
        all_lines = tuple(self._buffer)
        if last_n is not None and last_n < len(all_lines):
            return all_lines[-last_n:]
        return all_lines

    # -- Lifecycle controls --------------------------------------------------

    async def start(self) -> None:
        """Start the background line-buffering task.

        Subscribes to the broadcaster for the configured job and spawns
        an asyncio task that consumes output lines.

        Raises:
            RuntimeError: If the monitor is already running.
            ValueError: If the job is not registered in the broadcaster.
        """
        if self._state == OutputMonitorState.RUNNING:
            raise RuntimeError(
                f"OutputMonitor for job {self._job_id!r} is already running"
            )

        # Subscribe to the broadcaster (raises ValueError if not registered)
        self._handle = self._broadcaster.subscribe(self._job_id)
        self._state = OutputMonitorState.RUNNING

        self._task = asyncio.create_task(
            self._line_buffer_loop(),
            name=f"output-monitor-{self._job_id}",
        )

        logger.debug(
            "OutputMonitor started for job %s (subscriber=%s)",
            self._job_id,
            self._handle.subscriber_id,
        )

    async def stop(self) -> None:
        """Stop the background task gracefully.

        If not running, this is a no-op. Unsubscribes from the broadcaster
        and waits for the background task to complete within the configured
        timeout. If the task does not finish in time, it is cancelled.
        """
        if self._state in (OutputMonitorState.IDLE, OutputMonitorState.STOPPED):
            return

        self._state = OutputMonitorState.STOPPING

        # Unsubscribe to trigger end-of-stream in the background task
        if self._handle is not None:
            self._broadcaster.unsubscribe(self._handle)

        if self._task is not None:
            try:
                await asyncio.wait_for(
                    self._task,
                    timeout=self._config.stop_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "OutputMonitor for job %s did not stop within %.1fs, "
                    "cancelling",
                    self._job_id,
                    self._config.stop_timeout_seconds,
                )
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None

        self._state = OutputMonitorState.STOPPED
        logger.debug(
            "OutputMonitor stopped for job %s "
            "(total_lines_observed=%d, buffer_size=%d)",
            self._job_id,
            self._total_lines_observed,
            len(self._buffer),
        )

    # -- Async context manager -----------------------------------------------

    async def __aenter__(self) -> OutputMonitor:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()

    # -- Internal line-buffering loop ----------------------------------------

    async def _line_buffer_loop(self) -> None:
        """Core background loop: read lines from the subscriber queue.

        Consumes OutputLine events from the broadcaster subscription and
        appends non-empty line text to the bounded deque. Terminates when:

        - An end-of-stream sentinel (``is_end=True``) is received.
        - The subscriber queue raises (unsubscribed / cancelled).
        - The task is cancelled externally.

        Empty lines (``line.line == ""``) are skipped to avoid buffering
        noise. The ``total_lines_observed`` counter only increments for
        non-empty, non-sentinel lines.
        """
        if self._handle is None:
            logger.error(
                "OutputMonitor._line_buffer_loop called without a subscriber handle"
            )
            self._state = OutputMonitorState.ERROR
            return

        logger.debug(
            "OutputMonitor line-buffer loop started for job %s",
            self._job_id,
        )

        try:
            async for output_line in self._broadcaster.iter_lines(self._handle):
                # Skip empty content lines
                if not output_line.line:
                    continue

                self._buffer.append(output_line.line)
                self._total_lines_observed += 1

        except asyncio.CancelledError:
            logger.debug(
                "OutputMonitor line-buffer loop cancelled for job %s",
                self._job_id,
            )
            raise
        except ValueError:
            # Subscriber was removed (unsubscribe called during stop)
            logger.debug(
                "OutputMonitor subscriber removed for job %s (normal stop)",
                self._job_id,
            )
        except Exception as exc:
            logger.error(
                "Unexpected error in OutputMonitor line-buffer loop "
                "for job %s: %s",
                self._job_id,
                exc,
            )
            self._state = OutputMonitorState.ERROR
            return
        finally:
            if self._state not in (
                OutputMonitorState.STOPPING,
                OutputMonitorState.ERROR,
            ):
                self._state = OutputMonitorState.STOPPED
            logger.debug(
                "OutputMonitor line-buffer loop ended for job %s "
                "(total_lines_observed=%d)",
                self._job_id,
                self._total_lines_observed,
            )
