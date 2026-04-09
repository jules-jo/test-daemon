"""Tests for async command queue wrapper.

The AsyncCommandQueue wraps the synchronous CommandQueue to provide
non-blocking async put/get operations suitable for the daemon's asyncio
event loop. Wiki I/O runs off-thread via asyncio.to_thread, and an
asyncio.Event signals when items become available for non-blocking get.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from jules_daemon.wiki.async_queue import AsyncCommandQueue
from jules_daemon.wiki.queue_models import (
    QueuedCommand,
    QueuePriority,
    QueueStatus,
)


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


@pytest_asyncio.fixture
async def async_queue(wiki_root: Path) -> AsyncCommandQueue:
    """Provide a fresh async command queue backed by a temp wiki."""
    return await AsyncCommandQueue.create(wiki_root=wiki_root)


# -- put (enqueue) --


class TestPut:
    @pytest.mark.asyncio
    async def test_put_returns_queued_command(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        result = await async_queue.put("run the tests")
        assert isinstance(result, QueuedCommand)
        assert result.natural_language == "run the tests"
        assert result.status == QueueStatus.QUEUED

    @pytest.mark.asyncio
    async def test_put_creates_wiki_file(
        self, async_queue: AsyncCommandQueue, wiki_root: Path
    ) -> None:
        result = await async_queue.put("run the tests")
        queue_dir = wiki_root / "pages" / "daemon" / "queue"
        files = list(queue_dir.glob("*.md"))
        assert len(files) == 1
        assert result.queue_id in files[0].name

    @pytest.mark.asyncio
    async def test_put_increments_size(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        await async_queue.put("first")
        await async_queue.put("second")
        size = await async_queue.size()
        assert size == 2

    @pytest.mark.asyncio
    async def test_put_with_ssh_target(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        result = await async_queue.put(
            "run tests",
            ssh_host="staging.example.com",
            ssh_user="ci",
            ssh_port=2222,
        )
        assert result.ssh_host == "staging.example.com"
        assert result.ssh_user == "ci"
        assert result.ssh_port == 2222

    @pytest.mark.asyncio
    async def test_put_with_priority(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        result = await async_queue.put(
            "urgent fix", priority=QueuePriority.URGENT
        )
        assert result.priority == QueuePriority.URGENT

    @pytest.mark.asyncio
    async def test_put_empty_raises(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        with pytest.raises(ValueError, match="natural_language must not be empty"):
            await async_queue.put("")

    @pytest.mark.asyncio
    async def test_put_is_non_blocking(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        """put() should not block the event loop for other coroutines."""
        completed_order: list[str] = []

        async def put_task() -> None:
            await async_queue.put("some command")
            completed_order.append("put")

        async def other_task() -> None:
            await asyncio.sleep(0)
            completed_order.append("other")

        await asyncio.gather(put_task(), other_task())
        # Both should complete -- ordering is non-deterministic but both present
        assert "put" in completed_order
        assert "other" in completed_order


# -- get (dequeue) --


class TestGet:
    @pytest.mark.asyncio
    async def test_get_returns_none_when_empty_nowait(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        result = await async_queue.get_nowait()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_oldest_queued(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        await async_queue.put("first")
        await async_queue.put("second")
        result = await async_queue.get_nowait()
        assert result is not None
        assert result.natural_language == "first"
        assert result.status == QueueStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_get_respects_priority(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        await async_queue.put("normal cmd", priority=QueuePriority.NORMAL)
        await async_queue.put("urgent cmd", priority=QueuePriority.URGENT)
        result = await async_queue.get_nowait()
        assert result is not None
        assert result.natural_language == "urgent cmd"

    @pytest.mark.asyncio
    async def test_get_removes_wiki_file(
        self, async_queue: AsyncCommandQueue, wiki_root: Path
    ) -> None:
        await async_queue.put("to dequeue")
        await async_queue.get_nowait()
        queue_dir = wiki_root / "pages" / "daemon" / "queue"
        files = list(queue_dir.glob("*.md"))
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_get_decrements_size(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        await async_queue.put("a")
        await async_queue.put("b")
        await async_queue.get_nowait()
        size = await async_queue.size()
        assert size == 1

    @pytest.mark.asyncio
    async def test_get_wait_blocks_until_item(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        """get() with timeout should wait for an item and then return it."""
        result_holder: list[QueuedCommand | None] = []

        async def delayed_put() -> None:
            await asyncio.sleep(0.05)
            await async_queue.put("delayed item")

        async def wait_get() -> None:
            result = await async_queue.get(timeout=2.0)
            result_holder.append(result)

        await asyncio.gather(delayed_put(), wait_get())
        assert len(result_holder) == 1
        assert result_holder[0] is not None
        assert result_holder[0].natural_language == "delayed item"

    @pytest.mark.asyncio
    async def test_get_wait_times_out(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        """get() with a short timeout returns None when no item arrives."""
        result = await async_queue.get(timeout=0.05)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_sequential_fifo(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        await async_queue.put("first")
        await async_queue.put("second")
        await async_queue.put("third")

        r1 = await async_queue.get_nowait()
        r2 = await async_queue.get_nowait()
        r3 = await async_queue.get_nowait()

        assert r1 is not None and r1.natural_language == "first"
        assert r2 is not None and r2.natural_language == "second"
        assert r3 is not None and r3.natural_language == "third"


# -- peek --


class TestPeek:
    @pytest.mark.asyncio
    async def test_peek_empty_returns_none(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        result = await async_queue.peek()
        assert result is None

    @pytest.mark.asyncio
    async def test_peek_does_not_remove(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        await async_queue.put("command")
        first = await async_queue.peek()
        second = await async_queue.peek()
        assert first is not None
        assert second is not None
        assert first.queue_id == second.queue_id

    @pytest.mark.asyncio
    async def test_peek_returns_next_in_order(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        await async_queue.put("first")
        await async_queue.put("second")
        result = await async_queue.peek()
        assert result is not None
        assert result.natural_language == "first"


# -- list_pending --


class TestListPending:
    @pytest.mark.asyncio
    async def test_list_empty(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        items = await async_queue.list_pending()
        assert items == ()

    @pytest.mark.asyncio
    async def test_list_returns_all_queued(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        await async_queue.put("a")
        await async_queue.put("b")
        await async_queue.put("c")
        items = await async_queue.list_pending()
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_list_sorted_by_priority(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        await async_queue.put("normal1", priority=QueuePriority.NORMAL)
        await async_queue.put("urgent1", priority=QueuePriority.URGENT)
        await async_queue.put("normal2", priority=QueuePriority.NORMAL)
        items = await async_queue.list_pending()
        assert items[0].natural_language == "urgent1"


# -- cancel --


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_existing(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        cmd = await async_queue.put("to cancel")
        result = await async_queue.cancel(cmd.queue_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_false(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        result = await async_queue.cancel("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_removes_from_pending(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        cmd = await async_queue.put("to cancel")
        await async_queue.cancel(cmd.queue_id)
        items = await async_queue.list_pending()
        assert len(items) == 0


# -- Wiki recovery --


class TestAsyncWikiRecovery:
    @pytest.mark.asyncio
    async def test_recovers_existing_entries(self, wiki_root: Path) -> None:
        q1 = await AsyncCommandQueue.create(wiki_root=wiki_root)
        await q1.put("persisted command 1")
        await q1.put("persisted command 2")

        q2 = await AsyncCommandQueue.create(wiki_root=wiki_root)
        size = await q2.size()
        assert size == 2

    @pytest.mark.asyncio
    async def test_recovered_entries_maintain_order(
        self, wiki_root: Path
    ) -> None:
        q1 = await AsyncCommandQueue.create(wiki_root=wiki_root)
        await q1.put("first")
        await q1.put("second")

        q2 = await AsyncCommandQueue.create(wiki_root=wiki_root)
        result = await q2.get_nowait()
        assert result is not None
        assert result.natural_language == "first"

    @pytest.mark.asyncio
    async def test_sequence_counter_survives_restart(
        self, wiki_root: Path
    ) -> None:
        q1 = await AsyncCommandQueue.create(wiki_root=wiki_root)
        await q1.put("cmd1")
        await q1.put("cmd2")
        items = await q1.list_pending()
        last_seq = items[-1].sequence

        q2 = await AsyncCommandQueue.create(wiki_root=wiki_root)
        new_cmd = await q2.put("cmd3")
        assert new_cmd.sequence > last_seq


# -- Concurrent async operations --


class TestConcurrentAsync:
    @pytest.mark.asyncio
    async def test_concurrent_puts(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        """Multiple concurrent puts should not lose entries."""
        count = 30
        tasks = [
            async_queue.put(f"command-{i}")
            for i in range(count)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == count
        size = await async_queue.size()
        assert size == count

    @pytest.mark.asyncio
    async def test_concurrent_puts_unique_ids(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        """All concurrently enqueued commands get unique queue IDs."""
        count = 20
        tasks = [
            async_queue.put(f"command-{i}")
            for i in range(count)
        ]
        results = await asyncio.gather(*tasks)
        ids = [r.queue_id for r in results]
        assert len(ids) == len(set(ids)), "Duplicate queue IDs found"

    @pytest.mark.asyncio
    async def test_concurrent_puts_unique_sequences(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        """All concurrently enqueued commands get unique sequence numbers."""
        count = 20
        tasks = [
            async_queue.put(f"command-{i}")
            for i in range(count)
        ]
        results = await asyncio.gather(*tasks)
        sequences = [r.sequence for r in results]
        assert len(sequences) == len(set(sequences)), "Duplicate sequences found"

    @pytest.mark.asyncio
    async def test_put_signals_waiting_get(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        """A waiting get() should be woken when put() adds an item."""
        received: list[QueuedCommand] = []

        async def consumer() -> None:
            result = await async_queue.get(timeout=2.0)
            if result is not None:
                received.append(result)

        async def producer() -> None:
            await asyncio.sleep(0.02)
            await async_queue.put("signal test")

        await asyncio.gather(consumer(), producer())
        assert len(received) == 1
        assert received[0].natural_language == "signal test"

    @pytest.mark.asyncio
    async def test_multiple_consumers_single_item(
        self, async_queue: AsyncCommandQueue
    ) -> None:
        """Only one of multiple waiting consumers should get the item."""
        received: list[QueuedCommand | None] = []

        async def consumer() -> None:
            result = await async_queue.get(timeout=0.3)
            received.append(result)

        async def producer() -> None:
            await asyncio.sleep(0.05)
            await async_queue.put("single item")

        await asyncio.gather(
            consumer(), consumer(), consumer(), producer()
        )
        non_none = [r for r in received if r is not None]
        assert len(non_none) == 1
        assert non_none[0].natural_language == "single item"
