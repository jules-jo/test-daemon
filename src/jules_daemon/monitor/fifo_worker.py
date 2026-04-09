"""Sequential FIFO worker loop that processes queued commands one at a time.

Dequeues commands from the AsyncCommandQueue in submission order and processes
each through the full lifecycle, updating wiki state at every transition:

  QUEUED -> ACTIVE -> COMPLETED | FAILED

Architecture::

    AsyncCommandQueue (wiki-backed)
         |
         v
    FifoWorkerLoop._process_loop
         |
         +-- wait for next command (get with timeout)
         +-- activate (wiki updated to ACTIVE)
         +-- dispatch to handler
         +-- mark completed or failed (wiki updated)
         +-- repeat

Key design decisions:
  - Sequential processing: only one command is ACTIVE at any point.
  - Wiki state is the source of truth: each transition is persisted
    before the next step begins.
  - Handler errors are caught and converted to FAILED state -- the
    loop never crashes from a handler exception.
  - Crash recovery: on startup, scan for ACTIVE commands left over
    from a previous crash and mark them as FAILED.
  - Graceful shutdown: stop event interrupts waiting, and any
    in-flight command completes before the loop exits.

Usage::

    worker = FifoWorkerLoop(
        queue=async_queue,
        wiki_root=wiki_root,
        handler=my_handler,
    )

    async with worker:
        # worker processes commands in the background
        await some_other_work()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType

from jules_daemon.wiki.async_queue import AsyncCommandQueue
from jules_daemon.wiki.queue_models import QueuedCommand

__all__ = [
    "CommandResult",
    "FifoWorkerLoop",
    "ProcessingHandler",
    "WorkerConfig",
    "WorkerState",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerConfig:
    """Immutable configuration for the FIFO worker loop.

    Attributes:
        poll_interval_seconds: Seconds to sleep between loop iterations
            when the queue is empty. Must be positive. Default: 5.0.
        dequeue_timeout_seconds: Maximum seconds to wait for a queue item
            in a single dequeue attempt. Must be positive. Default: 10.0.
        stop_timeout_seconds: Maximum seconds to wait for the loop to
            complete during stop(). Must be positive. Default: 15.0.
    """

    poll_interval_seconds: float = 5.0
    dequeue_timeout_seconds: float = 10.0
    stop_timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError(
                f"poll_interval_seconds must be positive, "
                f"got {self.poll_interval_seconds}"
            )
        if self.dequeue_timeout_seconds <= 0:
            raise ValueError(
                f"dequeue_timeout_seconds must be positive, "
                f"got {self.dequeue_timeout_seconds}"
            )
        if self.stop_timeout_seconds <= 0:
            raise ValueError(
                f"stop_timeout_seconds must be positive, "
                f"got {self.stop_timeout_seconds}"
            )


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class WorkerState(Enum):
    """Lifecycle states for the FIFO worker loop."""

    IDLE = "idle"
    WAITING = "waiting"
    PROCESSING = "processing"
    STOPPED = "stopped"


# ---------------------------------------------------------------------------
# Command result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    """Immutable result from processing a command.

    Attributes:
        success: True if the command completed successfully.
        error: Human-readable error description, or None on success.
    """

    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Handler type alias
# ---------------------------------------------------------------------------

ProcessingHandler = Callable[[QueuedCommand], Awaitable[CommandResult]]
"""Async callback invoked to process each dequeued command.

The handler receives the command after it has been activated (wiki state
is ACTIVE). The worker marks the command as COMPLETED or FAILED based on
the handler's return value or any raised exception.
"""


# ---------------------------------------------------------------------------
# FIFO worker loop
# ---------------------------------------------------------------------------


class FifoWorkerLoop:
    """Sequential FIFO worker that processes queued commands one at a time.

    Dequeues commands in submission order from the AsyncCommandQueue,
    updates wiki state at each transition point, and processes them
    through a handler callback.

    Thread safety:
        This class is NOT thread-safe. All methods must be called from
        the same asyncio event loop.
    """

    __slots__ = (
        "_queue",
        "_wiki_root",
        "_handler",
        "_config",
        "_stop_event",
        "_task",
        "_state",
        "_total_processed",
        "_total_errors",
    )

    def __init__(
        self,
        *,
        queue: AsyncCommandQueue,
        wiki_root: Path,
        handler: ProcessingHandler,
        config: WorkerConfig | None = None,
    ) -> None:
        """Initialize the FIFO worker loop.

        The worker starts in IDLE state and does NOT automatically
        begin processing. Call start() or use the async context manager.

        Args:
            queue: The async command queue to dequeue from.
            wiki_root: Path to the wiki root directory.
            handler: Async callback invoked to process each command.
            config: Optional worker configuration. Uses defaults if None.
        """
        self._queue = queue
        self._wiki_root = wiki_root
        self._handler = handler
        self._config = config or WorkerConfig()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._state = WorkerState.IDLE
        self._total_processed: int = 0
        self._total_errors: int = 0

    # -- Public properties --

    @property
    def state(self) -> WorkerState:
        """Current lifecycle state of the worker."""
        return self._state

    @property
    def total_processed(self) -> int:
        """Total number of commands successfully processed."""
        return self._total_processed

    @property
    def total_errors(self) -> int:
        """Total number of commands that failed."""
        return self._total_errors

    # -- Lifecycle controls --

    async def start(self) -> None:
        """Start the worker as a background asyncio task.

        Raises:
            RuntimeError: If the worker is already running.
        """
        if self._state in (WorkerState.WAITING, WorkerState.PROCESSING):
            raise RuntimeError("FIFO worker is already running")

        self._stop_event.clear()
        self._state = WorkerState.WAITING
        self._task = asyncio.create_task(
            self._process_loop(),
            name="fifo-worker-loop",
        )

    async def stop(self) -> None:
        """Stop the worker gracefully.

        If not running, this is a no-op. Sets the stop event and waits
        for the background task to complete. Any in-flight command
        finishes before the loop exits.
        """
        if self._state == WorkerState.IDLE:
            return
        if self._state == WorkerState.STOPPED:
            return

        self._stop_event.set()

        if self._task is not None:
            try:
                await asyncio.wait_for(
                    self._task,
                    timeout=self._config.stop_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "FIFO worker did not stop within %.1fs, cancelling",
                    self._config.stop_timeout_seconds,
                )
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None

        if self._state != WorkerState.STOPPED:
            self._state = WorkerState.STOPPED

    # -- Crash recovery --

    async def recover_active_commands(self) -> list[QueuedCommand]:
        """Recover commands left in ACTIVE state from a previous crash.

        Scans wiki files for ACTIVE commands and marks them as FAILED
        with a crash recovery error message. Call this before start()
        to clean up after an unclean shutdown.

        Returns:
            List of commands that were marked as FAILED.
        """
        active = await self._queue.scan_active()
        recovered: list[QueuedCommand] = []

        for cmd in active:
            failed = await self._queue.mark_failed(
                cmd.queue_id,
                error="Crash recovery: command was active when daemon stopped",
            )
            if failed is not None:
                recovered.append(failed)
                logger.info(
                    "Crash recovery: marked active command as failed: "
                    "queue_id=%s cmd=%s",
                    cmd.queue_id,
                    cmd.natural_language[:80],
                )

        return recovered

    # -- Async context manager --

    async def __aenter__(self) -> FifoWorkerLoop:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()

    # -- Internal processing loop --

    async def _process_loop(self) -> None:
        """Main processing loop: dequeue, activate, process, complete/fail.

        This coroutine runs as a background task until the stop event
        is set. Each iteration follows a four-phase cycle:

        1. **Wait**: Get the next command from the queue (with timeout).
        2. **Activate**: Transition to ACTIVE, updating the wiki file.
        3. **Process**: Invoke the handler callback.
        4. **Resolve**: Mark as COMPLETED or FAILED in the wiki file.

        The loop processes one command at a time. If the queue is empty,
        it sleeps for poll_interval_seconds before retrying.
        """
        logger.info("FIFO worker loop started")

        try:
            while not self._stop_event.is_set():
                # Phase 1: Dequeue
                self._state = WorkerState.WAITING
                cmd = await self._try_dequeue()
                if cmd is None:
                    await self._interruptible_sleep(
                        self._config.poll_interval_seconds,
                    )
                    continue

                # Phase 2: Activate (update wiki to ACTIVE)
                self._state = WorkerState.PROCESSING
                activated = await self._queue.activate(cmd.queue_id)
                if activated is None:
                    logger.warning(
                        "Failed to activate command (may have been "
                        "cancelled): queue_id=%s",
                        cmd.queue_id,
                    )
                    continue

                # Phase 3+4: Process and resolve
                await self._process_and_resolve(activated)

        except asyncio.CancelledError:
            logger.info("FIFO worker loop cancelled")
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error in FIFO worker loop: %s", exc
            )
        finally:
            self._state = WorkerState.STOPPED
            logger.info(
                "FIFO worker loop ended: processed=%d errors=%d",
                self._total_processed,
                self._total_errors,
            )

    async def _try_dequeue(self) -> QueuedCommand | None:
        """Peek at the next pending command without removing it.

        The command stays in QUEUED state in the wiki file. The caller
        is responsible for activating it (transitioning to ACTIVE)
        before processing.

        Uses peek() instead of dequeue() because we want the wiki file
        to remain and be updated through state transitions, not deleted.

        First tries an immediate peek. If empty, waits efficiently
        using the queue's condition variable for up to
        ``dequeue_timeout_seconds``.

        Returns:
            The next QueuedCommand in QUEUED state, or None if empty.
        """
        # Immediate check
        peeked = await self._queue.peek()
        if peeked is not None:
            return peeked

        # Wait for a new item, then peek again
        has_pending = await self._queue.wait_for_pending(
            timeout=self._config.dequeue_timeout_seconds,
        )
        if has_pending:
            return await self._queue.peek()
        return None

    async def _process_and_resolve(self, cmd: QueuedCommand) -> None:
        """Process an activated command and resolve to terminal state.

        Invokes the handler and marks the command as COMPLETED or FAILED
        based on the result or any raised exception.

        Args:
            cmd: The activated command (must be in ACTIVE state).
        """
        logger.info(
            "FIFO worker processing: queue_id=%s cmd=%s",
            cmd.queue_id,
            cmd.natural_language[:80],
        )

        try:
            result = await self._handler(cmd)

            if result.success:
                await self._queue.mark_completed(cmd.queue_id)
                self._total_processed += 1
                logger.info(
                    "FIFO worker completed: queue_id=%s",
                    cmd.queue_id,
                )
            else:
                error = result.error or "Handler returned failure"
                await self._queue.mark_failed(cmd.queue_id, error)
                self._total_errors += 1
                logger.warning(
                    "FIFO worker command failed: queue_id=%s error=%s",
                    cmd.queue_id,
                    error[:80],
                )

        except Exception as exc:
            error_msg = f"Handler exception: {exc}"
            await self._queue.mark_failed(cmd.queue_id, error_msg)
            self._total_errors += 1
            logger.error(
                "FIFO worker handler exception: queue_id=%s error=%s",
                cmd.queue_id,
                exc,
            )

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep for the given duration, interruptible by stop event.

        Returns immediately if the stop event is already set or becomes
        set during the sleep.
        """
        if self._stop_event.is_set():
            return
        try:
            await asyncio.wait_for(
                self._stop_event.wait(),
                timeout=seconds,
            )
        except TimeoutError:
            pass
