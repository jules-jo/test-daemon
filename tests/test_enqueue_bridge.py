"""Tests for the enqueue bridge function.

The enqueue bridge wires validated commands into the thread-safe
wiki-backed queue. It accepts a CommandRequest, checks backpressure
(queue capacity), enqueues the command, generates an immutable
confirmation receipt, and returns it to the socket handler.

Tests cover:
- Successful enqueue with receipt generation
- Receipt field correctness (receipt_id, queue_id, position, etc.)
- Backpressure: QueueFullError when queue is at capacity
- Queue position tracking
- Priority forwarding
- Thread safety under concurrent enqueue calls
- Immutability of receipt objects
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pytest

from jules_daemon.ipc.enqueue_bridge import (
    DEFAULT_MAX_QUEUE_SIZE,
    EnqueueReceipt,
    QueueFullError,
    enqueue_command,
)
from jules_daemon.models.command_request import CommandRequest
from jules_daemon.wiki.command_queue import CommandQueue
from jules_daemon.wiki.queue_models import QueuePriority, QueueStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


@pytest.fixture
def queue(wiki_root: Path) -> CommandQueue:
    """Provide a fresh command queue backed by a temp wiki."""
    return CommandQueue(wiki_root=wiki_root)


@pytest.fixture
def valid_command() -> CommandRequest:
    """Provide a valid CommandRequest for testing."""
    return CommandRequest(
        natural_language_command="run the full test suite",
        target_host="staging.example.com",
        target_user="deploy",
        target_port=22,
    )


# ---------------------------------------------------------------------------
# EnqueueReceipt model tests
# ---------------------------------------------------------------------------


class TestEnqueueReceipt:
    """Tests for the immutable EnqueueReceipt dataclass."""

    def test_receipt_is_frozen(self) -> None:
        receipt = EnqueueReceipt(
            receipt_id="r-001",
            queue_id="q-001",
            command_id="c-001",
            sequence=1,
            position=1,
            queue_size=1,
            natural_language_preview="run the tests",
            target_host="staging.example.com",
            enqueued_at="2026-04-09T12:00:00Z",
        )
        with pytest.raises(AttributeError):
            receipt.receipt_id = "mutated"  # type: ignore[misc]

    def test_receipt_to_dict(self) -> None:
        receipt = EnqueueReceipt(
            receipt_id="r-001",
            queue_id="q-001",
            command_id="c-001",
            sequence=1,
            position=1,
            queue_size=1,
            natural_language_preview="run the tests",
            target_host="staging.example.com",
            enqueued_at="2026-04-09T12:00:00Z",
        )
        d = receipt.to_dict()
        assert d["receipt_id"] == "r-001"
        assert d["queue_id"] == "q-001"
        assert d["command_id"] == "c-001"
        assert d["sequence"] == 1
        assert d["position"] == 1
        assert d["queue_size"] == 1
        assert d["natural_language_preview"] == "run the tests"
        assert d["target_host"] == "staging.example.com"
        assert d["enqueued_at"] == "2026-04-09T12:00:00Z"

    def test_receipt_fields_are_accessible(self) -> None:
        receipt = EnqueueReceipt(
            receipt_id="r-001",
            queue_id="q-001",
            command_id="c-001",
            sequence=5,
            position=3,
            queue_size=3,
            natural_language_preview="run tests",
            target_host="host.example.com",
            enqueued_at="2026-04-09T12:00:00Z",
        )
        assert receipt.receipt_id == "r-001"
        assert receipt.queue_id == "q-001"
        assert receipt.command_id == "c-001"
        assert receipt.sequence == 5
        assert receipt.position == 3
        assert receipt.queue_size == 3
        assert receipt.natural_language_preview == "run tests"
        assert receipt.target_host == "host.example.com"

    def test_receipt_empty_receipt_id_raises(self) -> None:
        with pytest.raises(ValueError, match="receipt_id must not be empty"):
            EnqueueReceipt(
                receipt_id="",
                queue_id="q-001",
                command_id="c-001",
                sequence=1,
                position=1,
                queue_size=1,
                natural_language_preview="run tests",
                target_host="host.example.com",
                enqueued_at="2026-04-09T12:00:00Z",
            )

    def test_receipt_empty_queue_id_raises(self) -> None:
        with pytest.raises(ValueError, match="queue_id must not be empty"):
            EnqueueReceipt(
                receipt_id="r-001",
                queue_id="",
                command_id="c-001",
                sequence=1,
                position=1,
                queue_size=1,
                natural_language_preview="run tests",
                target_host="host.example.com",
                enqueued_at="2026-04-09T12:00:00Z",
            )


# ---------------------------------------------------------------------------
# QueueFullError tests
# ---------------------------------------------------------------------------


class TestQueueFullError:
    """Tests for the QueueFullError exception."""

    def test_is_exception(self) -> None:
        error = QueueFullError(current_size=10, max_size=10)
        assert isinstance(error, Exception)

    def test_attributes(self) -> None:
        error = QueueFullError(current_size=10, max_size=10)
        assert error.current_size == 10
        assert error.max_size == 10

    def test_message_contains_sizes(self) -> None:
        error = QueueFullError(current_size=10, max_size=10)
        msg = str(error)
        assert "10" in msg

    def test_to_dict(self) -> None:
        error = QueueFullError(current_size=5, max_size=5)
        d = error.to_dict()
        assert d["error"] == "queue_full"
        assert d["current_size"] == 5
        assert d["max_size"] == 5


# ---------------------------------------------------------------------------
# enqueue_command: successful enqueue
# ---------------------------------------------------------------------------


class TestEnqueueCommandSuccess:
    """Tests for successful command enqueue through the bridge."""

    def test_returns_receipt(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(valid_command, queue)
        assert isinstance(receipt, EnqueueReceipt)

    def test_receipt_has_unique_receipt_id(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(valid_command, queue)
        assert receipt.receipt_id
        # Should be a valid UUID
        uuid.UUID(receipt.receipt_id)

    def test_receipt_has_queue_id(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(valid_command, queue)
        assert receipt.queue_id
        # Should be a valid UUID
        uuid.UUID(receipt.queue_id)

    def test_receipt_has_command_id(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(valid_command, queue)
        assert receipt.command_id == valid_command.command_id

    def test_receipt_has_sequence(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(valid_command, queue)
        assert receipt.sequence > 0

    def test_receipt_position_is_one_for_first_command(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(valid_command, queue)
        assert receipt.position == 1

    def test_receipt_queue_size_matches(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(valid_command, queue)
        assert receipt.queue_size == 1

    def test_receipt_natural_language_preview(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(valid_command, queue)
        assert receipt.natural_language_preview == "run the full test suite"

    def test_receipt_target_host(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(valid_command, queue)
        assert receipt.target_host == "staging.example.com"

    def test_receipt_has_enqueued_at_timestamp(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(valid_command, queue)
        assert receipt.enqueued_at

    def test_command_is_in_queue_after_enqueue(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(valid_command, queue)
        queued = queue.get(receipt.queue_id)
        assert queued is not None
        assert queued.status == QueueStatus.QUEUED

    def test_wiki_file_created_after_enqueue(
        self, queue: CommandQueue, valid_command: CommandRequest,
        wiki_root: Path
    ) -> None:
        enqueue_command(valid_command, queue)
        queue_dir = wiki_root / "pages" / "daemon" / "queue"
        files = list(queue_dir.glob("*.md"))
        assert len(files) == 1

    def test_ssh_target_forwarded_to_queue(
        self, queue: CommandQueue
    ) -> None:
        cmd = CommandRequest(
            natural_language_command="run smoke tests",
            target_host="prod.example.com",
            target_user="ci",
            target_port=2222,
        )
        receipt = enqueue_command(cmd, queue)
        queued = queue.get(receipt.queue_id)
        assert queued is not None
        assert queued.ssh_host == "prod.example.com"
        assert queued.ssh_user == "ci"
        assert queued.ssh_port == 2222


# ---------------------------------------------------------------------------
# enqueue_command: queue position tracking
# ---------------------------------------------------------------------------


class TestEnqueueCommandPosition:
    """Tests for position tracking across multiple enqueues."""

    def test_sequential_positions(self, queue: CommandQueue) -> None:
        receipts = []
        for i in range(3):
            cmd = CommandRequest(
                natural_language_command=f"run test suite {i}",
                target_host=f"host-{i}.example.com",
            )
            receipt = enqueue_command(cmd, queue)
            receipts.append(receipt)

        assert receipts[0].position == 1
        assert receipts[1].position == 2
        assert receipts[2].position == 3

    def test_queue_size_increments(self, queue: CommandQueue) -> None:
        receipts = []
        for i in range(3):
            cmd = CommandRequest(
                natural_language_command=f"run test suite {i}",
                target_host=f"host-{i}.example.com",
            )
            receipt = enqueue_command(cmd, queue)
            receipts.append(receipt)

        assert receipts[0].queue_size == 1
        assert receipts[1].queue_size == 2
        assert receipts[2].queue_size == 3

    def test_unique_receipt_ids(self, queue: CommandQueue) -> None:
        receipt_ids = set()
        for i in range(10):
            cmd = CommandRequest(
                natural_language_command=f"run test suite {i}",
                target_host=f"host-{i}.example.com",
            )
            receipt = enqueue_command(cmd, queue)
            receipt_ids.add(receipt.receipt_id)

        assert len(receipt_ids) == 10

    def test_unique_queue_ids(self, queue: CommandQueue) -> None:
        queue_ids = set()
        for i in range(10):
            cmd = CommandRequest(
                natural_language_command=f"run test suite {i}",
                target_host=f"host-{i}.example.com",
            )
            receipt = enqueue_command(cmd, queue)
            queue_ids.add(receipt.queue_id)

        assert len(queue_ids) == 10


# ---------------------------------------------------------------------------
# enqueue_command: priority forwarding
# ---------------------------------------------------------------------------


class TestEnqueueCommandPriority:
    """Tests for priority forwarding through the bridge."""

    def test_default_priority_is_normal(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(valid_command, queue)
        queued = queue.get(receipt.queue_id)
        assert queued is not None
        assert queued.priority == QueuePriority.NORMAL

    def test_high_priority_forwarded(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(
            valid_command, queue, priority=QueuePriority.HIGH
        )
        queued = queue.get(receipt.queue_id)
        assert queued is not None
        assert queued.priority == QueuePriority.HIGH

    def test_urgent_priority_forwarded(
        self, queue: CommandQueue, valid_command: CommandRequest
    ) -> None:
        receipt = enqueue_command(
            valid_command, queue, priority=QueuePriority.URGENT
        )
        queued = queue.get(receipt.queue_id)
        assert queued is not None
        assert queued.priority == QueuePriority.URGENT


# ---------------------------------------------------------------------------
# enqueue_command: backpressure (queue full)
# ---------------------------------------------------------------------------


class TestEnqueueCommandBackpressure:
    """Tests for backpressure when queue reaches capacity."""

    def test_queue_full_raises_error(self, queue: CommandQueue) -> None:
        max_size = 3
        for i in range(max_size):
            cmd = CommandRequest(
                natural_language_command=f"command {i}",
                target_host="host.example.com",
            )
            enqueue_command(cmd, queue, max_queue_size=max_size)

        overflow_cmd = CommandRequest(
            natural_language_command="overflow command",
            target_host="host.example.com",
        )
        with pytest.raises(QueueFullError) as exc_info:
            enqueue_command(overflow_cmd, queue, max_queue_size=max_size)

        assert exc_info.value.current_size == max_size
        assert exc_info.value.max_size == max_size

    def test_queue_not_full_allows_enqueue(
        self, queue: CommandQueue
    ) -> None:
        max_size = 5
        cmd = CommandRequest(
            natural_language_command="first command",
            target_host="host.example.com",
        )
        receipt = enqueue_command(cmd, queue, max_queue_size=max_size)
        assert receipt.position == 1

    def test_queue_at_limit_minus_one_allows_enqueue(
        self, queue: CommandQueue
    ) -> None:
        max_size = 3
        for i in range(max_size - 1):
            cmd = CommandRequest(
                natural_language_command=f"command {i}",
                target_host="host.example.com",
            )
            enqueue_command(cmd, queue, max_queue_size=max_size)

        last_cmd = CommandRequest(
            natural_language_command="last allowed command",
            target_host="host.example.com",
        )
        receipt = enqueue_command(last_cmd, queue, max_queue_size=max_size)
        assert receipt.position == max_size

    def test_default_max_queue_size(self) -> None:
        assert DEFAULT_MAX_QUEUE_SIZE > 0

    def test_queue_full_error_has_dict_representation(
        self, queue: CommandQueue
    ) -> None:
        max_size = 1
        cmd = CommandRequest(
            natural_language_command="first",
            target_host="host.example.com",
        )
        enqueue_command(cmd, queue, max_queue_size=max_size)

        overflow_cmd = CommandRequest(
            natural_language_command="second",
            target_host="host.example.com",
        )
        with pytest.raises(QueueFullError) as exc_info:
            enqueue_command(overflow_cmd, queue, max_queue_size=max_size)

        d = exc_info.value.to_dict()
        assert d["error"] == "queue_full"
        assert d["current_size"] == 1
        assert d["max_size"] == 1

    def test_queue_full_does_not_modify_queue(
        self, queue: CommandQueue
    ) -> None:
        max_size = 2
        for i in range(max_size):
            cmd = CommandRequest(
                natural_language_command=f"command {i}",
                target_host="host.example.com",
            )
            enqueue_command(cmd, queue, max_queue_size=max_size)

        assert queue.size() == max_size

        overflow_cmd = CommandRequest(
            natural_language_command="overflow",
            target_host="host.example.com",
        )
        with pytest.raises(QueueFullError):
            enqueue_command(overflow_cmd, queue, max_queue_size=max_size)

        # Queue size should be unchanged
        assert queue.size() == max_size


# ---------------------------------------------------------------------------
# enqueue_command: natural language preview truncation
# ---------------------------------------------------------------------------


class TestEnqueueCommandPreview:
    """Tests for natural language preview in receipts."""

    def test_short_command_not_truncated(
        self, queue: CommandQueue
    ) -> None:
        cmd = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
        )
        receipt = enqueue_command(cmd, queue)
        assert receipt.natural_language_preview == "run tests"

    def test_long_command_truncated(
        self, queue: CommandQueue
    ) -> None:
        long_text = "x" * 200
        cmd = CommandRequest(
            natural_language_command=long_text,
            target_host="host.example.com",
        )
        receipt = enqueue_command(cmd, queue)
        assert len(receipt.natural_language_preview) <= 120
        assert receipt.natural_language_preview.endswith("...")


# ---------------------------------------------------------------------------
# enqueue_command: thread safety
# ---------------------------------------------------------------------------


class TestEnqueueCommandThreadSafety:
    """Tests for thread safety of concurrent enqueue operations."""

    def test_concurrent_enqueue_no_lost_commands(
        self, queue: CommandQueue
    ) -> None:
        errors: list[str] = []
        receipts: list[EnqueueReceipt] = []
        lock = threading.Lock()
        count = 30

        def enqueue_one(idx: int) -> None:
            try:
                cmd = CommandRequest(
                    natural_language_command=f"concurrent command {idx}",
                    target_host=f"host-{idx}.example.com",
                )
                receipt = enqueue_command(cmd, queue)
                with lock:
                    receipts.append(receipt)
            except Exception as exc:
                with lock:
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
        assert len(receipts) == count
        assert queue.size() == count

    def test_concurrent_enqueue_unique_receipt_ids(
        self, queue: CommandQueue
    ) -> None:
        receipts: list[EnqueueReceipt] = []
        lock = threading.Lock()
        count = 20

        def enqueue_one(idx: int) -> None:
            cmd = CommandRequest(
                natural_language_command=f"concurrent command {idx}",
                target_host=f"host-{idx}.example.com",
            )
            receipt = enqueue_command(cmd, queue)
            with lock:
                receipts.append(receipt)

        threads = [
            threading.Thread(target=enqueue_one, args=(i,))
            for i in range(count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        receipt_ids = [r.receipt_id for r in receipts]
        assert len(receipt_ids) == len(set(receipt_ids)), "Duplicate receipt IDs"

    def test_concurrent_enqueue_respects_backpressure(
        self, queue: CommandQueue
    ) -> None:
        max_size = 10
        errors: list[QueueFullError] = []
        receipts: list[EnqueueReceipt] = []
        lock = threading.Lock()
        count = 20  # more than max_size

        def enqueue_one(idx: int) -> None:
            try:
                cmd = CommandRequest(
                    natural_language_command=f"concurrent command {idx}",
                    target_host=f"host-{idx}.example.com",
                )
                receipt = enqueue_command(
                    cmd, queue, max_queue_size=max_size
                )
                with lock:
                    receipts.append(receipt)
            except QueueFullError as exc:
                with lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=enqueue_one, args=(i,))
            for i in range(count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have either succeeded or gotten QueueFullError
        assert len(receipts) + len(errors) == count
        # Should never exceed max_size
        assert queue.size() <= max_size
        # At least some should have been rejected
        assert len(errors) > 0
