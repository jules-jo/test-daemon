"""Async command queue wrapper with non-blocking put/get operations.

Wraps the synchronous, thread-safe ``CommandQueue`` to provide an
asyncio-native interface for the daemon's event loop. Wiki file I/O
(which is blocking) runs off-thread via ``asyncio.to_thread`` so the
event loop is never blocked.

Non-blocking semantics:
  - ``put()`` enqueues a command and notifies any waiting consumers.
  - ``get_nowait()`` returns immediately (None if empty).
  - ``get(timeout=...)`` waits up to *timeout* seconds for an item,
    returning None on timeout.

Signaling: An ``asyncio.Condition`` bundles the async lock and the
wait/notify in a single primitive, eliminating lost-wakeup races that
can occur with a separate Event + Lock pattern. Producers call
``notify_all()`` after enqueue; consumers call ``wait_for()`` with a
predicate that checks the underlying queue size.

Usage::

    queue = await AsyncCommandQueue.create(wiki_root=Path("./wiki"))

    # Producer
    cmd = await queue.put("run the full test suite")

    # Consumer (blocking wait with timeout)
    next_cmd = await queue.get(timeout=5.0)

    # Consumer (immediate, non-blocking)
    next_cmd = await queue.get_nowait()
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from jules_daemon.wiki.command_queue import CommandQueue
from jules_daemon.wiki.queue_models import (
    QueuedCommand,
    QueuePriority,
)

__all__ = ["AsyncCommandQueue"]

logger = logging.getLogger(__name__)


class AsyncCommandQueue:
    """Async wrapper around the synchronous CommandQueue.

    Provides non-blocking ``put`` and ``get`` operations for the daemon's
    asyncio event loop. All blocking wiki I/O is delegated to a thread
    via ``asyncio.to_thread``.

    Thread safety: The underlying ``CommandQueue`` serializes all mutations
    via its own ``threading.Lock``. This wrapper uses an ``asyncio.Condition``
    that bundles an async lock with wait/notify signaling to coordinate
    async callers without lost-wakeup races.

    Create instances via the ``create`` classmethod (not ``__init__``),
    because wiki recovery involves blocking I/O that must run off-thread.
    """

    __slots__ = ("_inner", "_condition")

    def __init__(self, inner: CommandQueue) -> None:
        """Initialize with a pre-built synchronous CommandQueue.

        Prefer ``AsyncCommandQueue.create()`` which handles off-thread
        construction.
        """
        self._inner = inner
        self._condition = asyncio.Condition()

    @classmethod
    async def create(cls, *, wiki_root: Path) -> AsyncCommandQueue:
        """Create a new AsyncCommandQueue, recovering wiki state off-thread.

        Args:
            wiki_root: Path to the wiki root directory.

        Returns:
            A fully initialized AsyncCommandQueue.
        """
        inner = await asyncio.to_thread(CommandQueue, wiki_root)
        return cls(inner)

    def _has_pending(self) -> bool:
        """Check if the inner queue has pending items (sync, no I/O)."""
        return self._inner.size() > 0

    async def put(
        self,
        natural_language: str,
        *,
        ssh_host: str | None = None,
        ssh_user: str | None = None,
        ssh_port: int = 22,
        priority: QueuePriority = QueuePriority.NORMAL,
    ) -> QueuedCommand:
        """Enqueue a command (non-blocking for the event loop).

        Runs the wiki file write off-thread. After enqueueing, notifies
        any consumers waiting in ``get()``.

        Args:
            natural_language: The user's command text.
            ssh_host: Optional target SSH hostname.
            ssh_user: Optional target SSH username.
            ssh_port: Target SSH port (default 22).
            priority: Priority tier for ordering.

        Returns:
            The newly created QueuedCommand.

        Raises:
            ValueError: If natural_language is empty.
        """
        async with self._condition:
            cmd = await asyncio.to_thread(
                self._inner.enqueue,
                natural_language,
                ssh_host=ssh_host,
                ssh_user=ssh_user,
                ssh_port=ssh_port,
                priority=priority,
            )
            self._condition.notify_all()
            logger.debug(
                "Async put: seq=%d id=%s", cmd.sequence, cmd.queue_id
            )
            return cmd

    async def get_nowait(self) -> QueuedCommand | None:
        """Dequeue the next command without waiting.

        Returns:
            The activated QueuedCommand, or None if the queue is empty.
        """
        async with self._condition:
            return await asyncio.to_thread(self._inner.dequeue)

    async def get(self, *, timeout: float = 0.0) -> QueuedCommand | None:
        """Dequeue the next command, waiting up to *timeout* seconds.

        If *timeout* is 0 or negative, behaves like ``get_nowait()``.

        Uses ``asyncio.Condition.wait_for()`` with a predicate that checks
        the underlying queue size. This eliminates lost-wakeup races: the
        condition variable atomically releases the lock while waiting and
        re-acquires it before the predicate check.

        Args:
            timeout: Maximum seconds to wait for an item.

        Returns:
            The activated QueuedCommand, or None on timeout.
        """
        if timeout <= 0:
            return await self.get_nowait()

        async with self._condition:
            # Wait until the predicate is true (items available) or timeout
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(self._has_pending),
                    timeout=timeout,
                )
            except TimeoutError:
                return None

            # Predicate was satisfied -- dequeue under the held lock
            return await asyncio.to_thread(self._inner.dequeue)

    async def peek(self) -> QueuedCommand | None:
        """Inspect the next command without removing it.

        Returns:
            The next QueuedCommand that would be dequeued, or None.
        """
        async with self._condition:
            return await asyncio.to_thread(self._inner.peek)

    async def list_pending(self) -> tuple[QueuedCommand, ...]:
        """Return all pending commands sorted by priority and sequence.

        Returns:
            Tuple of QueuedCommand instances in dequeue order.
        """
        async with self._condition:
            return await asyncio.to_thread(self._inner.list_pending)

    async def cancel(self, queue_id: str) -> bool:
        """Cancel a queued command and remove its wiki file.

        Args:
            queue_id: The queue_id of the command to cancel.

        Returns:
            True if the command was found and cancelled, False otherwise.
        """
        async with self._condition:
            return await asyncio.to_thread(self._inner.cancel, queue_id)

    async def size(self) -> int:
        """Return the number of pending commands in the queue.

        Returns:
            Count of QUEUED-status entries.
        """
        return await asyncio.to_thread(self._inner.size)

    async def wait_for_pending(self, *, timeout: float = 0.0) -> bool:
        """Wait until at least one pending command is available.

        Does NOT dequeue or modify any command. This is a read-only wait
        that uses the condition variable for efficient signaling.

        Args:
            timeout: Maximum seconds to wait. 0 or negative returns
                immediately.

        Returns:
            True if at least one pending command is available, False
            on timeout or empty queue.
        """
        if timeout <= 0:
            return self._has_pending()

        async with self._condition:
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(self._has_pending),
                    timeout=timeout,
                )
                return True
            except TimeoutError:
                return False

    async def get_by_id(self, queue_id: str) -> QueuedCommand | None:
        """Look up a specific queue entry by ID.

        Args:
            queue_id: The queue_id to look up.

        Returns:
            The QueuedCommand if found, or None.
        """
        return await asyncio.to_thread(self._inner.get, queue_id)

    # -- Lifecycle transition methods --

    async def activate(self, queue_id: str) -> QueuedCommand | None:
        """Transition a QUEUED command to ACTIVE, updating the wiki file.

        The wiki file is updated in-place. Only QUEUED commands can be
        activated.

        Args:
            queue_id: The queue_id of the command to activate.

        Returns:
            The activated QueuedCommand, or None if not found or not QUEUED.
        """
        return await asyncio.to_thread(self._inner.activate, queue_id)

    async def mark_completed(self, queue_id: str) -> QueuedCommand | None:
        """Transition an ACTIVE command to COMPLETED, updating the wiki file.

        Args:
            queue_id: The queue_id of the command to complete.

        Returns:
            The completed QueuedCommand, or None if not found or not ACTIVE.
        """
        return await asyncio.to_thread(self._inner.mark_completed, queue_id)

    async def mark_failed(
        self, queue_id: str, error: str
    ) -> QueuedCommand | None:
        """Transition an ACTIVE command to FAILED, updating the wiki file.

        Args:
            queue_id: The queue_id of the command to fail.
            error: Human-readable error description.

        Returns:
            The failed QueuedCommand, or None if not found or not ACTIVE.
        """
        return await asyncio.to_thread(
            self._inner.mark_failed, queue_id, error
        )

    async def scan_active(self) -> list[QueuedCommand]:
        """Scan wiki files for commands left in ACTIVE state.

        Used for crash recovery.

        Returns:
            List of QueuedCommand instances with ACTIVE status.
        """
        return await asyncio.to_thread(self._inner.scan_active)
