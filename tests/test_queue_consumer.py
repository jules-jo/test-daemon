"""Tests for queue consumer integration with the daemon run loop.

The QueueConsumer is a background asyncio task that drains enqueued commands
from the AsyncCommandQueue after the current execution completes. It must
not interrupt or block active runs.

Test categories:
  - Lifecycle: start, stop, state transitions
  - Draining: dequeue after run completes, respects priority
  - Non-interference: does not dequeue during active runs
  - Signaling: wakes on run_complete and on new enqueue
  - Error handling: callback failures, graceful degradation
  - Configuration: poll interval, drain behavior
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import pytest_asyncio

from jules_daemon.monitor.queue_consumer import (
    AsyncCommandHandler,
    ConsumerConfig,
    ConsumerState,
    DrainResult,
    QueueConsumer,
)
from jules_daemon.wiki.async_queue import AsyncCommandQueue
from jules_daemon.wiki.queue_models import QueuedCommand, QueuePriority


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


@pytest_asyncio.fixture
async def async_queue(wiki_root: Path) -> AsyncCommandQueue:
    """Provide a fresh async command queue backed by a temp wiki."""
    return await AsyncCommandQueue.create(wiki_root=wiki_root)


@pytest.fixture
def fast_config() -> ConsumerConfig:
    """Provide a fast-polling config for tests."""
    return ConsumerConfig(
        poll_interval_seconds=0.05,
        drain_timeout_seconds=0.1,
        stop_timeout_seconds=2.0,
    )


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initial_state_is_idle(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        assert consumer.state == ConsumerState.IDLE

    @pytest.mark.asyncio
    async def test_start_transitions_to_waiting(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        await consumer.start()
        try:
            # Give the task a moment to enter the loop
            await asyncio.sleep(0.02)
            assert consumer.state in (
                ConsumerState.WAITING_FOR_IDLE,
                ConsumerState.DRAINING,
            )
        finally:
            await consumer.stop()

    @pytest.mark.asyncio
    async def test_stop_transitions_to_stopped(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        await consumer.start()
        await consumer.stop()
        assert consumer.state == ConsumerState.STOPPED

    @pytest.mark.asyncio
    async def test_double_start_raises(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        await consumer.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await consumer.start()
        finally:
            await consumer.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_started_is_noop(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        await consumer.stop()
        assert consumer.state == ConsumerState.IDLE

    @pytest.mark.asyncio
    async def test_async_context_manager(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        dispatched: list[QueuedCommand] = []
        async with QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        ) as consumer:
            assert consumer.state in (
                ConsumerState.WAITING_FOR_IDLE,
                ConsumerState.DRAINING,
            )
        assert consumer.state == ConsumerState.STOPPED


# ---------------------------------------------------------------------------
# Draining tests
# ---------------------------------------------------------------------------


class TestDraining:
    @pytest.mark.asyncio
    async def test_drains_command_when_idle(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """Consumer dequeues a command when the daemon is idle."""
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        # Mark as idle so consumer can drain
        consumer.notify_run_idle()

        await async_queue.put("run the full test suite")
        await consumer.start()

        # Wait for the consumer to drain
        await _wait_for(lambda: len(dispatched) >= 1, timeout=1.0)
        await consumer.stop()

        assert len(dispatched) == 1
        assert dispatched[0].natural_language == "run the full test suite"

    @pytest.mark.asyncio
    async def test_drains_multiple_commands_sequentially(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """Consumer drains multiple commands one at a time."""
        dispatched: list[QueuedCommand] = []

        async def handler(cmd: QueuedCommand) -> DrainResult:
            dispatched.append(cmd)
            return DrainResult(success=True, error=None)

        consumer = QueueConsumer(
            queue=async_queue,
            on_command=handler,
            config=fast_config,
        )
        consumer.notify_run_idle()

        await async_queue.put("first")
        await async_queue.put("second")
        await async_queue.put("third")

        await consumer.start()
        await _wait_for(lambda: len(dispatched) >= 3, timeout=2.0)
        await consumer.stop()

        assert len(dispatched) == 3
        assert dispatched[0].natural_language == "first"
        assert dispatched[1].natural_language == "second"
        assert dispatched[2].natural_language == "third"

    @pytest.mark.asyncio
    async def test_drains_in_priority_order(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """Higher-priority commands are drained first."""
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        consumer.notify_run_idle()

        await async_queue.put("normal cmd", priority=QueuePriority.NORMAL)
        await async_queue.put("urgent cmd", priority=QueuePriority.URGENT)

        await consumer.start()
        await _wait_for(lambda: len(dispatched) >= 2, timeout=1.0)
        await consumer.stop()

        assert dispatched[0].natural_language == "urgent cmd"
        assert dispatched[1].natural_language == "normal cmd"

    @pytest.mark.asyncio
    async def test_drain_returns_result(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """Consumer passes drain result from callback."""
        results: list[DrainResult] = []

        async def handler(cmd: QueuedCommand) -> DrainResult:
            result = DrainResult(success=True, error=None)
            results.append(result)
            return result

        consumer = QueueConsumer(
            queue=async_queue,
            on_command=handler,
            config=fast_config,
        )
        consumer.notify_run_idle()

        await async_queue.put("test command")
        await consumer.start()
        await _wait_for(lambda: len(results) >= 1, timeout=1.0)
        await consumer.stop()

        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_waits_for_new_items_when_queue_empty(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """Consumer waits without busy-spinning when queue is empty."""
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        consumer.notify_run_idle()

        await consumer.start()
        # Let it spin for a bit with empty queue
        await asyncio.sleep(0.1)
        assert len(dispatched) == 0

        # Now add an item
        await async_queue.put("late arrival")
        await _wait_for(lambda: len(dispatched) >= 1, timeout=1.0)
        await consumer.stop()

        assert len(dispatched) == 1
        assert dispatched[0].natural_language == "late arrival"


# ---------------------------------------------------------------------------
# Non-interference tests (active run protection)
# ---------------------------------------------------------------------------


class TestNonInterference:
    @pytest.mark.asyncio
    async def test_does_not_drain_during_active_run(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """Consumer does not dequeue while a run is active."""
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        # Explicitly mark run as active (default state)
        consumer.notify_run_started()

        await async_queue.put("should not be drained yet")
        await consumer.start()

        # Wait a bit -- should NOT drain
        await asyncio.sleep(0.2)
        assert len(dispatched) == 0

        await consumer.stop()

    @pytest.mark.asyncio
    async def test_drains_after_run_completes(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """Consumer starts draining only after run_complete notification."""
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        consumer.notify_run_started()

        await async_queue.put("queued during active run")
        await consumer.start()

        # Should not drain yet
        await asyncio.sleep(0.1)
        assert len(dispatched) == 0

        # Signal run complete
        consumer.notify_run_idle()
        await _wait_for(lambda: len(dispatched) >= 1, timeout=1.0)
        await consumer.stop()

        assert len(dispatched) == 1
        assert dispatched[0].natural_language == "queued during active run"

    @pytest.mark.asyncio
    async def test_pauses_drain_when_new_run_starts(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """Consumer pauses draining when a new run is signaled."""
        dispatched: list[QueuedCommand] = []
        first_dispatched = asyncio.Event()
        proceed = asyncio.Event()

        async def gated_handler(cmd: QueuedCommand) -> DrainResult:
            dispatched.append(cmd)
            if len(dispatched) == 1:
                # Signal the test that the first command was dispatched
                first_dispatched.set()
                # Block until the test says to proceed
                await asyncio.wait_for(proceed.wait(), timeout=5.0)
            return DrainResult(success=True, error=None)

        consumer = QueueConsumer(
            queue=async_queue,
            on_command=gated_handler,
            config=fast_config,
        )
        consumer.notify_run_idle()

        await async_queue.put("first")
        await async_queue.put("second")

        await consumer.start()
        # Wait for first item to be dispatched (handler is now blocked)
        await asyncio.wait_for(first_dispatched.wait(), timeout=1.0)

        # Mark run as active BEFORE releasing the handler
        consumer.notify_run_started()
        # Release the handler so the dispatch completes
        proceed.set()

        count_after_pause = 1  # only first was dispatched
        await asyncio.sleep(0.15)
        # Second item should NOT have been drained
        assert len(dispatched) == count_after_pause

        # Release: mark idle again
        consumer.notify_run_idle()
        await _wait_for(lambda: len(dispatched) >= 2, timeout=1.0)
        await consumer.stop()

        assert len(dispatched) == 2


# ---------------------------------------------------------------------------
# Signaling tests
# ---------------------------------------------------------------------------


class TestSignaling:
    @pytest.mark.asyncio
    async def test_notify_run_idle_wakes_consumer(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """notify_run_idle wakes the consumer immediately."""
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        consumer.notify_run_started()

        await async_queue.put("waiting command")
        await consumer.start()
        await asyncio.sleep(0.05)
        assert len(dispatched) == 0

        consumer.notify_run_idle()
        await _wait_for(lambda: len(dispatched) >= 1, timeout=1.0)
        await consumer.stop()

        assert len(dispatched) == 1

    @pytest.mark.asyncio
    async def test_new_enqueue_during_idle_wakes_consumer(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """Enqueuing a command while idle wakes the consumer."""
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        consumer.notify_run_idle()

        await consumer.start()
        await asyncio.sleep(0.05)
        assert len(dispatched) == 0

        # Enqueue triggers the condition variable in AsyncCommandQueue
        await async_queue.put("new command")
        await _wait_for(lambda: len(dispatched) >= 1, timeout=1.0)
        await consumer.stop()

        assert len(dispatched) == 1

    @pytest.mark.asyncio
    async def test_total_commands_drained_counter(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """Consumer tracks total number of commands drained."""
        dispatched: list[QueuedCommand] = []
        consumer = QueueConsumer(
            queue=async_queue,
            on_command=_make_handler(dispatched),
            config=fast_config,
        )
        consumer.notify_run_idle()

        await async_queue.put("a")
        await async_queue.put("b")

        await consumer.start()
        await _wait_for(lambda: len(dispatched) >= 2, timeout=1.0)
        await consumer.stop()

        assert consumer.total_drained == 2


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash_consumer(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """Consumer continues after a callback raises an exception."""
        call_count = 0
        dispatched: list[QueuedCommand] = []

        async def failing_then_ok(cmd: QueuedCommand) -> DrainResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated failure")
            dispatched.append(cmd)
            return DrainResult(success=True, error=None)

        consumer = QueueConsumer(
            queue=async_queue,
            on_command=failing_then_ok,
            config=fast_config,
        )
        consumer.notify_run_idle()

        await async_queue.put("will fail")
        await async_queue.put("will succeed")

        await consumer.start()
        await _wait_for(lambda: len(dispatched) >= 1, timeout=1.0)
        await consumer.stop()

        assert len(dispatched) >= 1
        assert dispatched[0].natural_language == "will succeed"

    @pytest.mark.asyncio
    async def test_failed_drain_increments_error_counter(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """Consumer tracks number of failed drain attempts."""

        async def always_fail(cmd: QueuedCommand) -> DrainResult:
            raise RuntimeError("boom")

        consumer = QueueConsumer(
            queue=async_queue,
            on_command=always_fail,
            config=fast_config,
        )
        consumer.notify_run_idle()

        await async_queue.put("fail1")
        await async_queue.put("fail2")

        await consumer.start()
        await _wait_for(lambda: consumer.total_errors >= 2, timeout=1.0)
        await consumer.stop()

        assert consumer.total_errors >= 2

    @pytest.mark.asyncio
    async def test_failure_result_increments_error_counter(
        self, async_queue: AsyncCommandQueue, fast_config: ConsumerConfig
    ) -> None:
        """DrainResult(success=False) increments total_errors, not total_drained."""

        async def reject_handler(cmd: QueuedCommand) -> DrainResult:
            return DrainResult(success=False, error="rejected")

        consumer = QueueConsumer(
            queue=async_queue,
            on_command=reject_handler,
            config=fast_config,
        )
        consumer.notify_run_idle()

        await async_queue.put("will be rejected")
        await consumer.start()
        await _wait_for(lambda: consumer.total_errors >= 1, timeout=1.0)
        await consumer.stop()

        assert consumer.total_errors >= 1
        assert consumer.total_drained == 0


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_default_config(self) -> None:
        config = ConsumerConfig()
        assert config.poll_interval_seconds > 0
        assert config.drain_timeout_seconds > 0
        assert config.stop_timeout_seconds > 0

    def test_invalid_poll_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="poll_interval_seconds"):
            ConsumerConfig(poll_interval_seconds=0)

    def test_invalid_drain_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="drain_timeout_seconds"):
            ConsumerConfig(drain_timeout_seconds=-1)

    def test_invalid_stop_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="stop_timeout_seconds"):
            ConsumerConfig(stop_timeout_seconds=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(
    dispatched: list[QueuedCommand],
) -> AsyncCommandHandler:
    """Create a simple handler that records dispatched commands."""

    async def handler(cmd: QueuedCommand) -> DrainResult:
        dispatched.append(cmd)
        return DrainResult(success=True, error=None)

    return handler


async def _wait_for(
    predicate: Callable[[], bool],
    *,
    timeout: float = 1.0,
    poll: float = 0.01,
) -> None:
    """Poll a predicate until it returns True or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(poll)
    if not predicate():
        pytest.fail(f"Predicate not satisfied within {timeout}s")
