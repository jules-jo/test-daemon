"""Tests for the sequential FIFO worker loop.

The FifoWorkerLoop dequeues commands from the AsyncCommandQueue and processes
them sequentially, updating wiki state at each transition point:
  QUEUED -> ACTIVE -> COMPLETED/FAILED

Test categories:
  - Lifecycle: start, stop, state transitions
  - Sequential processing: one command at a time, FIFO order
  - Wiki state: each transition visible in wiki files
  - Error handling: handler failures, graceful degradation
  - Crash recovery: resume or fail active commands on startup
  - Configuration: timeouts and intervals
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import pytest_asyncio
import yaml

from jules_daemon.monitor.fifo_worker import (
    CommandResult,
    FifoWorkerLoop,
    ProcessingHandler,
    WorkerConfig,
    WorkerState,
)
from jules_daemon.wiki.async_queue import AsyncCommandQueue
from jules_daemon.wiki.command_queue import CommandQueue
from jules_daemon.wiki.queue_models import QueuedCommand, QueuePriority, QueueStatus


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
def fast_config() -> WorkerConfig:
    """Provide a fast-polling config for tests."""
    return WorkerConfig(
        poll_interval_seconds=0.05,
        dequeue_timeout_seconds=0.1,
        stop_timeout_seconds=2.0,
    )


def _read_wiki_status(wiki_root: Path, cmd: QueuedCommand) -> str:
    """Read the status field from a queue wiki file's frontmatter."""
    queue_dir = wiki_root / "pages" / "daemon" / "queue"
    file_path = queue_dir / f"{cmd.file_stem}.md"
    raw = file_path.read_text(encoding="utf-8")
    parts = raw.split("---", 2)
    fm = yaml.safe_load(parts[1])
    return fm["status"]


def _make_handler(
    results: list[QueuedCommand],
    *,
    delay: float = 0.0,
) -> ProcessingHandler:
    """Create a handler that records processed commands."""

    async def handler(cmd: QueuedCommand) -> CommandResult:
        if delay > 0:
            await asyncio.sleep(delay)
        results.append(cmd)
        return CommandResult(success=True)

    return handler


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initial_state_is_idle(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        processed: list[QueuedCommand] = []
        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=_make_handler(processed),
            config=fast_config,
        )
        assert worker.state == WorkerState.IDLE

    @pytest.mark.asyncio
    async def test_start_transitions_to_waiting(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        processed: list[QueuedCommand] = []
        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=_make_handler(processed),
            config=fast_config,
        )
        await worker.start()
        try:
            await asyncio.sleep(0.02)
            assert worker.state in (
                WorkerState.WAITING,
                WorkerState.PROCESSING,
            )
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_stop_transitions_to_stopped(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        processed: list[QueuedCommand] = []
        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=_make_handler(processed),
            config=fast_config,
        )
        await worker.start()
        await worker.stop()
        assert worker.state == WorkerState.STOPPED

    @pytest.mark.asyncio
    async def test_context_manager(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        processed: list[QueuedCommand] = []
        async with FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=_make_handler(processed),
            config=fast_config,
        ) as worker:
            assert worker.state in (
                WorkerState.WAITING,
                WorkerState.PROCESSING,
            )
        assert worker.state == WorkerState.STOPPED

    @pytest.mark.asyncio
    async def test_double_start_raises(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        processed: list[QueuedCommand] = []
        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=_make_handler(processed),
            config=fast_config,
        )
        await worker.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await worker.start()
        finally:
            await worker.stop()


# ---------------------------------------------------------------------------
# Sequential processing tests
# ---------------------------------------------------------------------------


class TestSequentialProcessing:
    @pytest.mark.asyncio
    async def test_processes_single_command(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        """Worker processes a single enqueued command."""
        processed: list[QueuedCommand] = []
        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=_make_handler(processed),
            config=fast_config,
        )

        await async_queue.put("run unit tests")
        await worker.start()

        await _wait_for(lambda: len(processed) >= 1, timeout=1.0)
        await worker.stop()

        assert len(processed) == 1
        assert processed[0].natural_language == "run unit tests"

    @pytest.mark.asyncio
    async def test_processes_in_fifo_order(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        """Commands are processed in submission (FIFO) order."""
        processed: list[QueuedCommand] = []
        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=_make_handler(processed),
            config=fast_config,
        )

        await async_queue.put("first")
        await async_queue.put("second")
        await async_queue.put("third")

        await worker.start()
        await _wait_for(lambda: len(processed) >= 3, timeout=2.0)
        await worker.stop()

        assert [cmd.natural_language for cmd in processed] == [
            "first",
            "second",
            "third",
        ]

    @pytest.mark.asyncio
    async def test_processes_one_at_a_time(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        """Only one command is in ACTIVE state at any point."""
        active_count_peak = 0
        active_count = 0
        processed: list[QueuedCommand] = []

        async def tracking_handler(cmd: QueuedCommand) -> CommandResult:
            nonlocal active_count, active_count_peak
            active_count += 1
            active_count_peak = max(active_count_peak, active_count)
            await asyncio.sleep(0.02)
            processed.append(cmd)
            active_count -= 1
            return CommandResult(success=True)

        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=tracking_handler,
            config=fast_config,
        )

        await async_queue.put("a")
        await async_queue.put("b")
        await async_queue.put("c")

        await worker.start()
        await _wait_for(lambda: len(processed) >= 3, timeout=2.0)
        await worker.stop()

        assert active_count_peak == 1

    @pytest.mark.asyncio
    async def test_respects_priority_within_fifo(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        """Higher-priority commands are processed first."""
        processed: list[QueuedCommand] = []
        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=_make_handler(processed),
            config=fast_config,
        )

        await async_queue.put("normal cmd", priority=QueuePriority.NORMAL)
        await async_queue.put("urgent cmd", priority=QueuePriority.URGENT)

        await worker.start()
        await _wait_for(lambda: len(processed) >= 2, timeout=1.0)
        await worker.stop()

        assert processed[0].natural_language == "urgent cmd"
        assert processed[1].natural_language == "normal cmd"

    @pytest.mark.asyncio
    async def test_waits_for_new_items(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        """Worker waits for new items without busy-spinning."""
        processed: list[QueuedCommand] = []
        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=_make_handler(processed),
            config=fast_config,
        )

        await worker.start()
        await asyncio.sleep(0.1)
        assert len(processed) == 0

        await async_queue.put("late arrival")
        await _wait_for(lambda: len(processed) >= 1, timeout=1.0)
        await worker.stop()

        assert len(processed) == 1


# ---------------------------------------------------------------------------
# Wiki state tracking tests
# ---------------------------------------------------------------------------


class TestWikiStateTracking:
    @pytest.mark.asyncio
    async def test_command_transitions_to_active_in_wiki(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        """While processing, the wiki file shows ACTIVE status."""
        active_seen = asyncio.Event()
        wiki_status_during_processing: list[str] = []

        async def checking_handler(cmd: QueuedCommand) -> CommandResult:
            status = _read_wiki_status(wiki_root, cmd)
            wiki_status_during_processing.append(status)
            active_seen.set()
            return CommandResult(success=True)

        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=checking_handler,
            config=fast_config,
        )

        cmd = await async_queue.put("check wiki state")
        await worker.start()
        await asyncio.wait_for(active_seen.wait(), timeout=1.0)
        await worker.stop()

        assert wiki_status_during_processing[0] == "active"

    @pytest.mark.asyncio
    async def test_successful_command_shows_completed_in_wiki(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        """After successful processing, wiki file shows COMPLETED."""
        processed: list[QueuedCommand] = []
        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=_make_handler(processed),
            config=fast_config,
        )

        cmd = await async_queue.put("run tests")
        await worker.start()
        await _wait_for(lambda: len(processed) >= 1, timeout=1.0)
        await worker.stop()

        status = _read_wiki_status(wiki_root, cmd)
        assert status == "completed"

    @pytest.mark.asyncio
    async def test_failed_command_shows_failed_in_wiki(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        """After handler failure, wiki file shows FAILED with error."""
        async def failing_handler(cmd: QueuedCommand) -> CommandResult:
            return CommandResult(success=False, error="test timeout")

        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=failing_handler,
            config=fast_config,
        )

        cmd = await async_queue.put("run failing tests")
        await worker.start()
        await _wait_for(lambda: worker.total_errors >= 1, timeout=1.0)
        await worker.stop()

        queue_dir = wiki_root / "pages" / "daemon" / "queue"
        file_path = queue_dir / f"{cmd.file_stem}.md"
        raw = file_path.read_text(encoding="utf-8")
        parts = raw.split("---", 2)
        fm = yaml.safe_load(parts[1])

        assert fm["status"] == "failed"
        assert fm["error"] == "test timeout"

    @pytest.mark.asyncio
    async def test_handler_exception_marks_failed_in_wiki(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        """Handler raising an exception marks command as FAILED."""
        async def exploding_handler(cmd: QueuedCommand) -> CommandResult:
            raise RuntimeError("kaboom")

        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=exploding_handler,
            config=fast_config,
        )

        cmd = await async_queue.put("run explosive test")
        await worker.start()
        await _wait_for(lambda: worker.total_errors >= 1, timeout=1.0)
        await worker.stop()

        status = _read_wiki_status(wiki_root, cmd)
        assert status == "failed"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_handler_exception_does_not_crash_worker(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        """Worker continues after a handler raises an exception."""
        call_count = 0
        processed: list[QueuedCommand] = []

        async def failing_then_ok(cmd: QueuedCommand) -> CommandResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated failure")
            processed.append(cmd)
            return CommandResult(success=True)

        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=failing_then_ok,
            config=fast_config,
        )

        await async_queue.put("will fail")
        await async_queue.put("will succeed")

        await worker.start()
        await _wait_for(lambda: len(processed) >= 1, timeout=1.0)
        await worker.stop()

        assert len(processed) >= 1
        assert processed[0].natural_language == "will succeed"

    @pytest.mark.asyncio
    async def test_total_processed_counter(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        """Worker tracks total processed commands."""
        processed: list[QueuedCommand] = []
        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=_make_handler(processed),
            config=fast_config,
        )

        await async_queue.put("a")
        await async_queue.put("b")

        await worker.start()
        await _wait_for(lambda: len(processed) >= 2, timeout=1.0)
        await worker.stop()

        assert worker.total_processed == 2

    @pytest.mark.asyncio
    async def test_total_errors_counter(
        self,
        async_queue: AsyncCommandQueue,
        wiki_root: Path,
        fast_config: WorkerConfig,
    ) -> None:
        """Worker tracks total error count."""
        async def always_fail(cmd: QueuedCommand) -> CommandResult:
            return CommandResult(success=False, error="nope")

        worker = FifoWorkerLoop(
            queue=async_queue,
            wiki_root=wiki_root,
            handler=always_fail,
            config=fast_config,
        )

        await async_queue.put("fail1")
        await async_queue.put("fail2")

        await worker.start()
        await _wait_for(lambda: worker.total_errors >= 2, timeout=1.0)
        await worker.stop()

        assert worker.total_errors >= 2


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_default_config(self) -> None:
        config = WorkerConfig()
        assert config.poll_interval_seconds > 0
        assert config.dequeue_timeout_seconds > 0
        assert config.stop_timeout_seconds > 0

    def test_invalid_poll_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="poll_interval_seconds"):
            WorkerConfig(poll_interval_seconds=0)

    def test_invalid_dequeue_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="dequeue_timeout_seconds"):
            WorkerConfig(dequeue_timeout_seconds=-1)

    def test_invalid_stop_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="stop_timeout_seconds"):
            WorkerConfig(stop_timeout_seconds=0)


# ---------------------------------------------------------------------------
# Crash recovery tests
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_recover_marks_active_as_failed(
        self, wiki_root: Path, fast_config: WorkerConfig
    ) -> None:
        """On startup, commands left ACTIVE (from a crash) are marked FAILED."""
        # Simulate a crash: enqueue, activate, then "crash" (abandon)
        sync_queue = CommandQueue(wiki_root)
        cmd = sync_queue.enqueue("interrupted by crash")
        sync_queue.activate(cmd.queue_id)

        # New worker starts and recovers
        new_async_queue = await AsyncCommandQueue.create(wiki_root=wiki_root)
        processed: list[QueuedCommand] = []
        worker = FifoWorkerLoop(
            queue=new_async_queue,
            wiki_root=wiki_root,
            handler=_make_handler(processed),
            config=fast_config,
        )

        recovered = await worker.recover_active_commands()
        assert len(recovered) == 1
        assert recovered[0].status == QueueStatus.FAILED
        assert "crash recovery" in (recovered[0].error or "").lower()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
