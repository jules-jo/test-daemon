"""Tests for thread-safe command queue with wiki-backed persistence."""

import threading
from pathlib import Path

import pytest

from jules_daemon.wiki.command_queue import CommandQueue
from jules_daemon.wiki.queue_models import (
    QueuedCommand,
    QueuePriority,
    QueueStatus,
)


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


@pytest.fixture
def queue(wiki_root: Path) -> CommandQueue:
    """Provide a fresh command queue backed by a temp wiki."""
    return CommandQueue(wiki_root=wiki_root)


# -- Enqueue --


class TestEnqueue:
    def test_enqueue_returns_queued_command(self, queue: CommandQueue) -> None:
        result = queue.enqueue("run the tests")
        assert result.natural_language == "run the tests"
        assert result.status == QueueStatus.QUEUED
        assert result.sequence > 0

    def test_enqueue_creates_wiki_file(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        result = queue.enqueue("run the tests")
        queue_dir = wiki_root / "pages" / "daemon" / "queue"
        files = list(queue_dir.glob("*.md"))
        assert len(files) == 1
        assert result.queue_id in files[0].name

    def test_enqueue_assigns_incrementing_sequences(
        self, queue: CommandQueue
    ) -> None:
        first = queue.enqueue("first command")
        second = queue.enqueue("second command")
        third = queue.enqueue("third command")
        assert first.sequence < second.sequence < third.sequence

    def test_enqueue_with_ssh_target(self, queue: CommandQueue) -> None:
        result = queue.enqueue(
            "run tests",
            ssh_host="staging.example.com",
            ssh_user="ci",
            ssh_port=2222,
        )
        assert result.ssh_host == "staging.example.com"
        assert result.ssh_user == "ci"
        assert result.ssh_port == 2222

    def test_enqueue_with_priority(self, queue: CommandQueue) -> None:
        result = queue.enqueue("urgent fix", priority=QueuePriority.URGENT)
        assert result.priority == QueuePriority.URGENT

    def test_enqueue_empty_command_raises(self, queue: CommandQueue) -> None:
        with pytest.raises(ValueError, match="natural_language must not be empty"):
            queue.enqueue("")


# -- Dequeue --


class TestDequeue:
    def test_dequeue_empty_returns_none(self, queue: CommandQueue) -> None:
        result = queue.dequeue()
        assert result is None

    def test_dequeue_returns_oldest_queued(self, queue: CommandQueue) -> None:
        queue.enqueue("first")
        queue.enqueue("second")
        result = queue.dequeue()
        assert result is not None
        assert result.natural_language == "first"
        assert result.status == QueueStatus.ACTIVE

    def test_dequeue_removes_file_from_queue_dir(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        queue.enqueue("command a")
        queue.dequeue()
        queue_dir = wiki_root / "pages" / "daemon" / "queue"
        files = list(queue_dir.glob("*.md"))
        assert len(files) == 0

    def test_dequeue_respects_priority(self, queue: CommandQueue) -> None:
        queue.enqueue("low priority", priority=QueuePriority.NORMAL)
        queue.enqueue("high priority", priority=QueuePriority.HIGH)
        queue.enqueue("medium normal", priority=QueuePriority.NORMAL)

        first = queue.dequeue()
        assert first is not None
        assert first.natural_language == "high priority"

    def test_dequeue_skips_non_queued_entries(
        self, queue: CommandQueue
    ) -> None:
        cmd = queue.enqueue("will be cancelled")
        queue.cancel(cmd.queue_id)
        queue.enqueue("still queued")

        result = queue.dequeue()
        assert result is not None
        assert result.natural_language == "still queued"

    def test_sequential_dequeue_order(self, queue: CommandQueue) -> None:
        queue.enqueue("first")
        queue.enqueue("second")
        queue.enqueue("third")

        r1 = queue.dequeue()
        r2 = queue.dequeue()
        r3 = queue.dequeue()

        assert r1 is not None and r1.natural_language == "first"
        assert r2 is not None and r2.natural_language == "second"
        assert r3 is not None and r3.natural_language == "third"


# -- Peek --


class TestPeek:
    def test_peek_empty_returns_none(self, queue: CommandQueue) -> None:
        result = queue.peek()
        assert result is None

    def test_peek_does_not_remove(self, queue: CommandQueue) -> None:
        queue.enqueue("command")
        first = queue.peek()
        second = queue.peek()
        assert first is not None
        assert second is not None
        assert first.queue_id == second.queue_id

    def test_peek_returns_next_in_order(self, queue: CommandQueue) -> None:
        queue.enqueue("first")
        queue.enqueue("second")
        result = queue.peek()
        assert result is not None
        assert result.natural_language == "first"


# -- List --


class TestList:
    def test_list_empty(self, queue: CommandQueue) -> None:
        items = queue.list_pending()
        assert items == ()

    def test_list_returns_all_queued(self, queue: CommandQueue) -> None:
        queue.enqueue("a")
        queue.enqueue("b")
        queue.enqueue("c")
        items = queue.list_pending()
        assert len(items) == 3

    def test_list_sorted_by_priority_then_sequence(
        self, queue: CommandQueue
    ) -> None:
        queue.enqueue("normal1", priority=QueuePriority.NORMAL)
        queue.enqueue("urgent1", priority=QueuePriority.URGENT)
        queue.enqueue("normal2", priority=QueuePriority.NORMAL)

        items = queue.list_pending()
        assert items[0].natural_language == "urgent1"
        assert items[1].natural_language == "normal1"
        assert items[2].natural_language == "normal2"

    def test_list_excludes_cancelled(self, queue: CommandQueue) -> None:
        cmd = queue.enqueue("will cancel")
        queue.enqueue("will stay")
        queue.cancel(cmd.queue_id)

        items = queue.list_pending()
        assert len(items) == 1
        assert items[0].natural_language == "will stay"


# -- Cancel --


class TestCancel:
    def test_cancel_existing(self, queue: CommandQueue) -> None:
        cmd = queue.enqueue("to cancel")
        result = queue.cancel(cmd.queue_id)
        assert result is True

    def test_cancel_nonexistent_returns_false(
        self, queue: CommandQueue
    ) -> None:
        result = queue.cancel("nonexistent-id")
        assert result is False

    def test_cancel_removes_from_pending(self, queue: CommandQueue) -> None:
        cmd = queue.enqueue("to cancel")
        queue.cancel(cmd.queue_id)
        items = queue.list_pending()
        assert len(items) == 0

    def test_cancel_removes_wiki_file(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        cmd = queue.enqueue("to cancel")
        queue.cancel(cmd.queue_id)
        queue_dir = wiki_root / "pages" / "daemon" / "queue"
        files = list(queue_dir.glob("*.md"))
        assert len(files) == 0


# -- Size --


class TestSize:
    def test_size_empty(self, queue: CommandQueue) -> None:
        assert queue.size() == 0

    def test_size_after_enqueue(self, queue: CommandQueue) -> None:
        queue.enqueue("a")
        queue.enqueue("b")
        assert queue.size() == 2

    def test_size_after_dequeue(self, queue: CommandQueue) -> None:
        queue.enqueue("a")
        queue.enqueue("b")
        queue.dequeue()
        assert queue.size() == 1

    def test_size_after_cancel(self, queue: CommandQueue) -> None:
        cmd = queue.enqueue("a")
        queue.enqueue("b")
        queue.cancel(cmd.queue_id)
        assert queue.size() == 1


# -- Wiki persistence recovery --


class TestWikiRecovery:
    def test_new_queue_loads_existing_entries(
        self, wiki_root: Path
    ) -> None:
        # First queue creates entries
        q1 = CommandQueue(wiki_root=wiki_root)
        q1.enqueue("persisted command 1")
        q1.enqueue("persisted command 2")

        # Second queue (simulating daemon restart) loads from wiki
        q2 = CommandQueue(wiki_root=wiki_root)
        assert q2.size() == 2

    def test_recovered_entries_maintain_order(
        self, wiki_root: Path
    ) -> None:
        q1 = CommandQueue(wiki_root=wiki_root)
        q1.enqueue("first")
        q1.enqueue("second")

        q2 = CommandQueue(wiki_root=wiki_root)
        result = q2.dequeue()
        assert result is not None
        assert result.natural_language == "first"

    def test_recovered_entries_preserve_metadata(
        self, wiki_root: Path
    ) -> None:
        q1 = CommandQueue(wiki_root=wiki_root)
        q1.enqueue(
            "test command",
            ssh_host="host.example.com",
            ssh_user="root",
            priority=QueuePriority.HIGH,
        )

        q2 = CommandQueue(wiki_root=wiki_root)
        items = q2.list_pending()
        assert len(items) == 1
        assert items[0].ssh_host == "host.example.com"
        assert items[0].ssh_user == "root"
        assert items[0].priority == QueuePriority.HIGH

    def test_sequence_counter_survives_restart(
        self, wiki_root: Path
    ) -> None:
        q1 = CommandQueue(wiki_root=wiki_root)
        q1.enqueue("cmd1")
        q1.enqueue("cmd2")
        last_seq = q1.list_pending()[-1].sequence

        q2 = CommandQueue(wiki_root=wiki_root)
        new_cmd = q2.enqueue("cmd3")
        assert new_cmd.sequence > last_seq


# -- Wiki file format --


class TestWikiFileFormat:
    def test_queue_file_has_valid_frontmatter(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        from jules_daemon.wiki.frontmatter import parse

        queue.enqueue("run integration tests")
        queue_dir = wiki_root / "pages" / "daemon" / "queue"
        files = list(queue_dir.glob("*.md"))
        assert len(files) == 1

        raw = files[0].read_text(encoding="utf-8")
        doc = parse(raw)
        assert "tags" in doc.frontmatter
        assert "daemon" in doc.frontmatter["tags"]
        assert "queue" in doc.frontmatter["tags"]
        assert doc.frontmatter["type"] == "queued-command"
        assert doc.frontmatter["status"] == "queued"
        assert "# Queued Command" in doc.body

    def test_queue_file_name_format(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        cmd = queue.enqueue("run tests")
        queue_dir = wiki_root / "pages" / "daemon" / "queue"
        files = list(queue_dir.glob("*.md"))
        expected_name = f"{cmd.file_stem}.md"
        assert files[0].name == expected_name


# -- Thread safety --


class TestThreadSafety:
    def test_concurrent_enqueue(self, queue: CommandQueue) -> None:
        """Multiple threads enqueuing simultaneously should not lose entries."""
        errors: list[str] = []
        count = 50

        def enqueue_one(idx: int) -> None:
            try:
                queue.enqueue(f"command-{idx}")
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=enqueue_one, args=(i,))
            for i in range(count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent enqueue: {errors}"
        assert queue.size() == count

    def test_concurrent_enqueue_unique_sequences(
        self, queue: CommandQueue
    ) -> None:
        """All concurrently enqueued commands get unique sequence numbers."""
        results: list[QueuedCommand] = []
        lock = threading.Lock()

        def enqueue_one(idx: int) -> None:
            cmd = queue.enqueue(f"command-{idx}")
            with lock:
                results.append(cmd)

        threads = [
            threading.Thread(target=enqueue_one, args=(i,))
            for i in range(30)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        sequences = [r.sequence for r in results]
        assert len(sequences) == len(set(sequences)), "Duplicate sequences found"

    def test_concurrent_enqueue_and_dequeue(
        self, queue: CommandQueue
    ) -> None:
        """Enqueue and dequeue running concurrently should not corrupt state."""
        errors: list[str] = []
        dequeued: list[QueuedCommand] = []
        lock = threading.Lock()

        # Pre-load some entries
        for i in range(20):
            queue.enqueue(f"preloaded-{i}")

        def enqueue_batch() -> None:
            for i in range(10):
                try:
                    queue.enqueue(f"concurrent-{i}")
                except Exception as exc:
                    with lock:
                        errors.append(str(exc))

        def dequeue_batch() -> None:
            for _ in range(15):
                try:
                    result = queue.dequeue()
                    if result is not None:
                        with lock:
                            dequeued.append(result)
                except Exception as exc:
                    with lock:
                        errors.append(str(exc))

        t1 = threading.Thread(target=enqueue_batch)
        t2 = threading.Thread(target=dequeue_batch)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # All dequeued should have unique IDs
        ids = [d.queue_id for d in dequeued]
        assert len(ids) == len(set(ids)), "Duplicate dequeued IDs"

    def test_concurrent_cancel(self, queue: CommandQueue) -> None:
        """Cancelling the same entry from two threads should be safe."""
        cmd = queue.enqueue("to cancel")
        results: list[bool] = []
        lock = threading.Lock()

        def cancel_one() -> None:
            result = queue.cancel(cmd.queue_id)
            with lock:
                results.append(result)

        t1 = threading.Thread(target=cancel_one)
        t2 = threading.Thread(target=cancel_one)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one should succeed
        assert results.count(True) == 1
        assert results.count(False) == 1
