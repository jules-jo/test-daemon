"""Queue consumer that drains enqueued commands after the current run completes.

Integrates with the daemon's run loop by watching for idle-state signals and
draining the next command from the ``AsyncCommandQueue``. The consumer runs
as a background asyncio task and never interrupts or blocks active runs.

Architecture::

    PollingLoop (SSH monitoring)
         |
         | (run completes -> terminal state)
         v
    notify_run_idle()
         |
         v
    QueueConsumer._drain_loop
         |
         +-- wait for idle signal (asyncio.Event)
         +-- dequeue from AsyncCommandQueue
         +-- dispatch via on_command callback
         +-- repeat
         |
         v
    on_command callback (starts next execution)

Signaling protocol:
  - ``notify_run_idle()``: Sets the idle event. The consumer wakes and
    attempts to dequeue. Called when the daemon transitions to a terminal
    or idle state.
  - ``notify_run_started()``: Clears the idle event. The consumer blocks
    on the next iteration until idle is signaled again. Called when a new
    run begins (from queue drain or direct ``run`` command).

Key design decisions:
  - The consumer does NOT own run lifecycle -- it only dequeues and
    dispatches via the callback. The callback decides what to do.
  - ``asyncio.Event`` for idle signaling: zero-cost when set, instant
    wake when signaled.
  - Uses ``AsyncCommandQueue.get(timeout=...)`` for efficient wait-on-
    empty (condition variable underneath).
  - Callback errors are logged and counted but never crash the loop.
  - Graceful shutdown via a separate stop event.
  - Dequeued commands are always dispatched even during shutdown to
    prevent silent data loss.

Usage::

    consumer = QueueConsumer(
        queue=async_queue,
        on_command=my_handler,
    )
    consumer.notify_run_idle()

    async with consumer:
        # consumer drains commands in the background
        await some_other_work()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from types import TracebackType

from jules_daemon.wiki.async_queue import AsyncCommandQueue
from jules_daemon.wiki.queue_models import QueuedCommand

__all__ = [
    "AsyncCommandHandler",
    "ConsumerConfig",
    "ConsumerState",
    "DrainResult",
    "QueueConsumer",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumerConfig:
    """Immutable configuration for the queue consumer.

    Attributes:
        poll_interval_seconds: Seconds to sleep between drain loop
            iterations when the queue is empty. Prevents busy-spinning
            while keeping the consumer responsive. Must be positive.
            Default: 5.0.
        drain_timeout_seconds: Maximum seconds to wait for a queue item
            in a single drain attempt. Must be positive. Default: 10.0.
        stop_timeout_seconds: Maximum seconds to wait for the drain loop
            to complete during ``stop()``. Should exceed
            ``drain_timeout_seconds`` to allow a pending drain to finish.
            Must be positive. Default: 15.0.
    """

    poll_interval_seconds: float = 5.0
    drain_timeout_seconds: float = 10.0
    stop_timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError(
                f"poll_interval_seconds must be positive, "
                f"got {self.poll_interval_seconds}"
            )
        if self.drain_timeout_seconds <= 0:
            raise ValueError(
                f"drain_timeout_seconds must be positive, "
                f"got {self.drain_timeout_seconds}"
            )
        if self.stop_timeout_seconds <= 0:
            raise ValueError(
                f"stop_timeout_seconds must be positive, "
                f"got {self.stop_timeout_seconds}"
            )


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class ConsumerState(Enum):
    """Lifecycle states for the queue consumer."""

    IDLE = "idle"
    WAITING_FOR_IDLE = "waiting_for_idle"
    DRAINING = "draining"
    DISPATCHING = "dispatching"
    STOPPED = "stopped"


# ---------------------------------------------------------------------------
# Drain result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrainResult:
    """Immutable result from a command dispatch callback.

    Attributes:
        success: True if the command was successfully accepted.
        error: Human-readable error description, or None on success.
    """

    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Callback type alias
# ---------------------------------------------------------------------------

AsyncCommandHandler = Callable[[QueuedCommand], Awaitable[DrainResult]]
"""Async callback invoked with each dequeued command.

The handler is responsible for initiating command execution (e.g.,
transitioning the daemon to PENDING_APPROVAL state). The consumer
waits for the handler to return before proceeding to the next command.
"""


# ---------------------------------------------------------------------------
# Queue consumer
# ---------------------------------------------------------------------------


class QueueConsumer:
    """Background asyncio task that drains enqueued commands.

    Monitors the daemon's idle state and dequeues commands from the
    ``AsyncCommandQueue`` when no run is active. Each dequeued command
    is dispatched to the ``on_command`` callback. The consumer never
    interrupts active runs.

    Thread safety:
        This class is NOT thread-safe. All methods must be called from
        the same asyncio event loop.
    """

    __slots__ = (
        "_queue",
        "_on_command",
        "_config",
        "_idle_event",
        "_stop_event",
        "_task",
        "_state",
        "_total_drained",
        "_total_errors",
    )

    def __init__(
        self,
        *,
        queue: AsyncCommandQueue,
        on_command: AsyncCommandHandler,
        config: ConsumerConfig | None = None,
    ) -> None:
        """Initialize the queue consumer.

        The consumer starts in IDLE state and does NOT automatically
        begin draining. Call ``start()`` or use the async context
        manager to begin.

        By default, the consumer assumes a run is active (idle event
        is cleared). Call ``notify_run_idle()`` when the daemon is
        ready to accept queued commands.

        Args:
            queue: The async command queue to drain from.
            on_command: Async callback invoked with each dequeued command.
            config: Optional consumer configuration. Uses defaults if None.
        """
        self._queue = queue
        self._on_command = on_command
        self._config = config or ConsumerConfig()
        self._idle_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._state = ConsumerState.IDLE
        self._total_drained: int = 0
        self._total_errors: int = 0

    # -- Public properties --

    @property
    def state(self) -> ConsumerState:
        """Current lifecycle state of the consumer."""
        return self._state

    @property
    def total_drained(self) -> int:
        """Total number of commands successfully dispatched."""
        return self._total_drained

    @property
    def total_errors(self) -> int:
        """Total number of dispatch errors."""
        return self._total_errors

    # -- Signaling --

    def notify_run_idle(self) -> None:
        """Signal that no run is active -- consumer may drain.

        Call this when the daemon transitions to IDLE, COMPLETED,
        FAILED, or CANCELLED state. The consumer will attempt to
        dequeue the next command on its next loop iteration.
        """
        self._idle_event.set()
        logger.debug("Queue consumer: idle signal received")

    def notify_run_started(self) -> None:
        """Signal that a new run has started -- consumer must wait.

        Call this when a new run begins (from queue drain or direct
        ``run`` command). The consumer will block until the next
        ``notify_run_idle()`` call.
        """
        self._idle_event.clear()
        logger.debug("Queue consumer: run-started signal received")

    # -- Lifecycle controls --

    async def start(self) -> None:
        """Start the consumer as a background asyncio task.

        Raises:
            RuntimeError: If the consumer is already running.
        """
        if self._state in (
            ConsumerState.WAITING_FOR_IDLE,
            ConsumerState.DRAINING,
            ConsumerState.DISPATCHING,
        ):
            raise RuntimeError("Queue consumer is already running")

        self._stop_event.clear()
        self._state = ConsumerState.WAITING_FOR_IDLE
        self._task = asyncio.create_task(
            self._drain_loop(),
            name="queue-consumer-drain",
        )

    async def stop(self) -> None:
        """Stop the consumer gracefully.

        If not running, this is a no-op. Sets the stop event and waits
        for the background task to complete. The drain loop owns the
        terminal state transition to STOPPED.
        """
        if self._state == ConsumerState.IDLE:
            return
        if self._state == ConsumerState.STOPPED:
            return

        self._stop_event.set()
        # Also set idle to unblock any wait
        self._idle_event.set()

        if self._task is not None:
            try:
                await asyncio.wait_for(
                    self._task,
                    timeout=self._config.stop_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Queue consumer did not stop within %.1fs, cancelling",
                    self._config.stop_timeout_seconds,
                )
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None

        # Defensive: ensure stopped state if the loop exited abnormally
        # without reaching its finally block (e.g., task was never started)
        if self._state != ConsumerState.STOPPED:
            self._state = ConsumerState.STOPPED

    # -- Async context manager --

    async def __aenter__(self) -> QueueConsumer:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()

    # -- Internal drain loop --

    async def _drain_loop(self) -> None:
        """Main drain loop: wait for idle, dequeue, dispatch, repeat.

        This coroutine runs as a background task until the stop event
        is set. It follows a three-phase cycle:

        1. **Wait for idle**: Block on ``_idle_event`` until the daemon
           signals no active run.
        2. **Dequeue**: Attempt to get the next command from the queue.
           If the queue is empty, sleep for ``poll_interval_seconds``
           before retrying.
        3. **Dispatch**: Invoke ``_on_command`` with the dequeued command.
           Always completes dispatch even if stop is signaled mid-drain,
           to prevent silent data loss.

        State ownership: This method is the sole authority for the
        STOPPED state transition. ``stop()`` defers to the finally
        block here.
        """
        logger.info("Queue consumer drain loop started")

        try:
            while not self._stop_event.is_set():
                # Phase 1: Wait for idle
                self._state = ConsumerState.WAITING_FOR_IDLE
                await self._wait_for_idle_or_stop()
                if self._stop_event.is_set():
                    break

                # Phase 2: Dequeue
                self._state = ConsumerState.DRAINING
                cmd = await self._try_dequeue()
                if cmd is None:
                    # Queue empty or timed out -- sleep before retrying
                    await self._interruptible_sleep(
                        self._config.poll_interval_seconds,
                    )
                    continue

                # Phase 3: Dispatch (always complete, even on stop signal)
                self._state = ConsumerState.DISPATCHING
                await self._dispatch(cmd)

        except asyncio.CancelledError:
            logger.info("Queue consumer drain loop cancelled")
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error in queue consumer drain loop: %s",
                exc,
            )
        finally:
            self._state = ConsumerState.STOPPED
            logger.info(
                "Queue consumer drain loop ended: drained=%d errors=%d",
                self._total_drained,
                self._total_errors,
            )

    async def _wait_for_idle_or_stop(self) -> None:
        """Wait until the idle event or stop event is set.

        Uses a dual-wait pattern: whichever event fires first wins.
        The idle event signals the daemon is ready for the next
        command; the stop event signals graceful shutdown.
        """
        if self._idle_event.is_set():
            return

        idle_wait = asyncio.ensure_future(self._idle_event.wait())
        stop_wait = asyncio.ensure_future(self._stop_event.wait())

        try:
            _done, _pending = await asyncio.wait(
                {idle_wait, stop_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for fut in (idle_wait, stop_wait):
                if not fut.done():
                    fut.cancel()
                    try:
                        await fut
                    except asyncio.CancelledError:
                        pass

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

    async def _try_dequeue(self) -> QueuedCommand | None:
        """Attempt to dequeue the next command from the queue.

        First tries an immediate dequeue. If the queue is empty,
        waits up to ``drain_timeout_seconds`` for a new item.

        Returns:
            The dequeued QueuedCommand, or None if the queue is empty
            and the timeout elapsed.
        """
        # Check idle is still set (may have been cleared between phases)
        if not self._idle_event.is_set():
            return None

        # Immediate attempt
        cmd = await self._queue.get_nowait()
        if cmd is not None:
            return cmd

        # Wait for a new item with timeout
        return await self._queue.get(
            timeout=self._config.drain_timeout_seconds,
        )

    async def _dispatch(self, cmd: QueuedCommand) -> None:
        """Dispatch a dequeued command to the on_command callback.

        On success, increments the drained counter. On failure (either
        via returned DrainResult or raised exception), increments the
        error counter and re-signals idle so the next command can be
        attempted.

        Args:
            cmd: The dequeued command to dispatch.
        """
        logger.info(
            "Queue consumer dispatching: queue_id=%s cmd=%s",
            cmd.queue_id,
            cmd.natural_language[:80],
        )

        try:
            result = await self._on_command(cmd)

            if result.success:
                self._total_drained += 1
                logger.info(
                    "Queue consumer dispatch success: queue_id=%s",
                    cmd.queue_id,
                )
            else:
                self._total_errors += 1
                logger.warning(
                    "Queue consumer dispatch returned failure: "
                    "queue_id=%s error=%s",
                    cmd.queue_id,
                    result.error,
                )
                # Re-signal idle since the command was not accepted
                self._idle_event.set()

        except Exception as exc:
            self._total_errors += 1
            logger.error(
                "Queue consumer dispatch error: queue_id=%s error=%s",
                cmd.queue_id,
                exc,
            )
            # Re-signal idle so we can try the next command
            self._idle_event.set()
