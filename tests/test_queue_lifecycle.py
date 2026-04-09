"""Tests for queue command lifecycle transitions with wiki persistence.

Verifies that commands transition through QUEUED -> ACTIVE -> COMPLETED/FAILED
with each state change persisted to wiki files. The wiki file is updated
in-place (not deleted) so the full lifecycle is visible.

Test categories:
  - Activate: QUEUED -> ACTIVE transition, wiki file updated
  - Complete: ACTIVE -> COMPLETED transition, wiki file updated
  - Fail: ACTIVE -> FAILED transition, wiki file updated
  - Recovery: scan active entries on restart
  - Invalid transitions: wrong state, missing entry
  - Wiki file content: frontmatter reflects each state
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jules_daemon.wiki.command_queue import CommandQueue
from jules_daemon.wiki.queue_models import QueuedCommand, QueuePriority, QueueStatus


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
    return CommandQueue(wiki_root)


def _read_wiki_frontmatter(wiki_root: Path, cmd: QueuedCommand) -> dict:
    """Read and parse frontmatter from a queue wiki file."""
    queue_dir = wiki_root / "pages" / "daemon" / "queue"
    file_path = queue_dir / f"{cmd.file_stem}.md"
    assert file_path.exists(), f"Wiki file not found: {file_path}"
    raw = file_path.read_text(encoding="utf-8")
    # Parse frontmatter between --- delimiters
    parts = raw.split("---", 2)
    assert len(parts) >= 3, "Missing frontmatter delimiters"
    return yaml.safe_load(parts[1])


def _wiki_file_exists(wiki_root: Path, cmd: QueuedCommand) -> bool:
    """Check if a queue wiki file exists."""
    queue_dir = wiki_root / "pages" / "daemon" / "queue"
    file_path = queue_dir / f"{cmd.file_stem}.md"
    return file_path.exists()


# ---------------------------------------------------------------------------
# Activate (QUEUED -> ACTIVE) tests
# ---------------------------------------------------------------------------


class TestActivate:
    def test_activate_returns_active_command(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """activate() returns the command with ACTIVE status."""
        cmd = queue.enqueue("run unit tests")
        activated = queue.activate(cmd.queue_id)

        assert activated is not None
        assert activated.status == QueueStatus.ACTIVE
        assert activated.natural_language == "run unit tests"

    def test_activate_sets_started_at(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Activated command has a started_at timestamp."""
        cmd = queue.enqueue("run integration tests")
        activated = queue.activate(cmd.queue_id)

        assert activated is not None
        assert activated.started_at is not None

    def test_activate_updates_wiki_file(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Wiki file is updated (not deleted) with ACTIVE status."""
        cmd = queue.enqueue("run linting")
        queue.activate(cmd.queue_id)

        fm = _read_wiki_frontmatter(wiki_root, cmd)
        assert fm["status"] == "active"
        assert fm["started_at"] is not None

    def test_activate_preserves_original_metadata(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Activation preserves queue_id, sequence, and other metadata."""
        cmd = queue.enqueue(
            "run tests",
            ssh_host="prod-server",
            ssh_user="deploy",
        )
        activated = queue.activate(cmd.queue_id)

        assert activated is not None
        assert activated.queue_id == cmd.queue_id
        assert activated.sequence == cmd.sequence
        assert activated.ssh_host == "prod-server"
        assert activated.ssh_user == "deploy"

    def test_activate_removes_from_pending(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Activated command is no longer in pending list."""
        cmd = queue.enqueue("run tests")
        queue.activate(cmd.queue_id)

        pending = queue.list_pending()
        assert len(pending) == 0

    def test_activate_does_not_affect_pending_count(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Pending count decreases after activation."""
        queue.enqueue("first")
        cmd2 = queue.enqueue("second")
        queue.activate(cmd2.queue_id)

        assert queue.size() == 1

    def test_activate_nonexistent_returns_none(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Activating a nonexistent queue_id returns None."""
        result = queue.activate("nonexistent-id")
        assert result is None

    def test_activate_already_active_returns_none(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Activating an already-active command returns None."""
        cmd = queue.enqueue("run tests")
        queue.activate(cmd.queue_id)
        result = queue.activate(cmd.queue_id)
        assert result is None

    def test_activate_respects_fifo_order(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """First-enqueued command is the first to be activated."""
        cmd1 = queue.enqueue("first command")
        cmd2 = queue.enqueue("second command")

        # Activate the first one
        activated = queue.activate(cmd1.queue_id)
        assert activated is not None
        assert activated.natural_language == "first command"

        # Second is still pending
        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0].queue_id == cmd2.queue_id


# ---------------------------------------------------------------------------
# Complete (ACTIVE -> COMPLETED) tests
# ---------------------------------------------------------------------------


class TestMarkCompleted:
    def test_mark_completed_returns_completed_command(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """mark_completed() returns the command with COMPLETED status."""
        cmd = queue.enqueue("run tests")
        queue.activate(cmd.queue_id)
        completed = queue.mark_completed(cmd.queue_id)

        assert completed is not None
        assert completed.status == QueueStatus.COMPLETED

    def test_mark_completed_sets_completed_at(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Completed command has a completed_at timestamp."""
        cmd = queue.enqueue("run tests")
        queue.activate(cmd.queue_id)
        completed = queue.mark_completed(cmd.queue_id)

        assert completed is not None
        assert completed.completed_at is not None

    def test_mark_completed_updates_wiki_file(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Wiki file is updated with COMPLETED status."""
        cmd = queue.enqueue("run tests")
        queue.activate(cmd.queue_id)
        queue.mark_completed(cmd.queue_id)

        fm = _read_wiki_frontmatter(wiki_root, cmd)
        assert fm["status"] == "completed"
        assert fm["completed_at"] is not None

    def test_mark_completed_removes_from_index(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Completed command is removed from in-memory index."""
        cmd = queue.enqueue("run tests")
        queue.activate(cmd.queue_id)
        queue.mark_completed(cmd.queue_id)

        assert queue.get(cmd.queue_id) is None

    def test_mark_completed_on_queued_returns_none(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Cannot complete a QUEUED (non-active) command."""
        cmd = queue.enqueue("run tests")
        result = queue.mark_completed(cmd.queue_id)
        assert result is None

    def test_mark_completed_nonexistent_returns_none(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Completing a nonexistent queue_id returns None."""
        result = queue.mark_completed("nonexistent-id")
        assert result is None


# ---------------------------------------------------------------------------
# Fail (ACTIVE -> FAILED) tests
# ---------------------------------------------------------------------------


class TestMarkFailed:
    def test_mark_failed_returns_failed_command(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """mark_failed() returns the command with FAILED status."""
        cmd = queue.enqueue("run tests")
        queue.activate(cmd.queue_id)
        failed = queue.mark_failed(cmd.queue_id, error="connection lost")

        assert failed is not None
        assert failed.status == QueueStatus.FAILED
        assert failed.error == "connection lost"

    def test_mark_failed_sets_completed_at(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Failed command has a completed_at timestamp."""
        cmd = queue.enqueue("run tests")
        queue.activate(cmd.queue_id)
        failed = queue.mark_failed(cmd.queue_id, error="timeout")

        assert failed is not None
        assert failed.completed_at is not None

    def test_mark_failed_updates_wiki_file(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Wiki file is updated with FAILED status and error."""
        cmd = queue.enqueue("run tests")
        queue.activate(cmd.queue_id)
        queue.mark_failed(cmd.queue_id, error="SSH disconnect")

        fm = _read_wiki_frontmatter(wiki_root, cmd)
        assert fm["status"] == "failed"
        assert fm["error"] == "SSH disconnect"
        assert fm["completed_at"] is not None

    def test_mark_failed_removes_from_index(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Failed command is removed from in-memory index."""
        cmd = queue.enqueue("run tests")
        queue.activate(cmd.queue_id)
        queue.mark_failed(cmd.queue_id, error="boom")

        assert queue.get(cmd.queue_id) is None

    def test_mark_failed_on_queued_returns_none(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Cannot fail a QUEUED (non-active) command."""
        cmd = queue.enqueue("run tests")
        result = queue.mark_failed(cmd.queue_id, error="nope")
        assert result is None

    def test_mark_failed_nonexistent_returns_none(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Failing a nonexistent queue_id returns None."""
        result = queue.mark_failed("nonexistent-id", error="nope")
        assert result is None


# ---------------------------------------------------------------------------
# Full lifecycle tests
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_full_lifecycle_queued_to_completed(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Command transitions through all states with wiki updates."""
        cmd = queue.enqueue("run full test suite")

        # State 1: QUEUED
        fm = _read_wiki_frontmatter(wiki_root, cmd)
        assert fm["status"] == "queued"

        # State 2: ACTIVE
        activated = queue.activate(cmd.queue_id)
        assert activated is not None
        fm = _read_wiki_frontmatter(wiki_root, cmd)
        assert fm["status"] == "active"

        # State 3: COMPLETED
        completed = queue.mark_completed(cmd.queue_id)
        assert completed is not None
        fm = _read_wiki_frontmatter(wiki_root, cmd)
        assert fm["status"] == "completed"

    def test_full_lifecycle_queued_to_failed(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Failed command lifecycle persists error to wiki."""
        cmd = queue.enqueue("run smoke tests")

        # QUEUED -> ACTIVE -> FAILED
        queue.activate(cmd.queue_id)
        failed = queue.mark_failed(cmd.queue_id, error="test failure")

        assert failed is not None
        fm = _read_wiki_frontmatter(wiki_root, cmd)
        assert fm["status"] == "failed"
        assert fm["error"] == "test failure"

    def test_multiple_commands_sequential_lifecycle(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Multiple commands process sequentially in FIFO order."""
        cmd1 = queue.enqueue("first")
        cmd2 = queue.enqueue("second")
        cmd3 = queue.enqueue("third")

        # Process first: activate -> complete
        queue.activate(cmd1.queue_id)
        queue.mark_completed(cmd1.queue_id)

        # Process second: activate -> fail
        queue.activate(cmd2.queue_id)
        queue.mark_failed(cmd2.queue_id, error="oops")

        # Process third: activate -> complete
        queue.activate(cmd3.queue_id)
        queue.mark_completed(cmd3.queue_id)

        # All wiki files should exist with terminal states
        fm1 = _read_wiki_frontmatter(wiki_root, cmd1)
        fm2 = _read_wiki_frontmatter(wiki_root, cmd2)
        fm3 = _read_wiki_frontmatter(wiki_root, cmd3)
        assert fm1["status"] == "completed"
        assert fm2["status"] == "failed"
        assert fm3["status"] == "completed"

    def test_wiki_file_persists_after_terminal_state(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """Wiki file still exists after command reaches terminal state."""
        cmd = queue.enqueue("run tests")
        queue.activate(cmd.queue_id)
        queue.mark_completed(cmd.queue_id)

        assert _wiki_file_exists(wiki_root, cmd)


# ---------------------------------------------------------------------------
# Active entry recovery tests
# ---------------------------------------------------------------------------


class TestActiveRecovery:
    def test_scan_active_returns_active_entries(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """scan_active() finds commands left in ACTIVE state."""
        cmd1 = queue.enqueue("active command")
        cmd2 = queue.enqueue("still queued")
        queue.activate(cmd1.queue_id)

        active = queue.scan_active()
        assert len(active) == 1
        assert active[0].queue_id == cmd1.queue_id
        assert active[0].status == QueueStatus.ACTIVE

    def test_scan_active_empty_when_all_queued(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """scan_active() returns empty when all commands are QUEUED."""
        queue.enqueue("queued 1")
        queue.enqueue("queued 2")

        active = queue.scan_active()
        assert len(active) == 0

    def test_scan_active_empty_when_all_completed(
        self, queue: CommandQueue, wiki_root: Path
    ) -> None:
        """scan_active() returns empty when all commands are terminal."""
        cmd = queue.enqueue("will complete")
        queue.activate(cmd.queue_id)
        queue.mark_completed(cmd.queue_id)

        active = queue.scan_active()
        assert len(active) == 0

    def test_recovery_after_restart_finds_active(
        self, wiki_root: Path
    ) -> None:
        """After restart (new CommandQueue), ACTIVE entries are found."""
        # First queue: enqueue and activate
        q1 = CommandQueue(wiki_root)
        cmd = q1.enqueue("interrupted command")
        q1.activate(cmd.queue_id)

        # Simulated restart: new queue instance
        q2 = CommandQueue(wiki_root)
        active = q2.scan_active()

        assert len(active) == 1
        assert active[0].queue_id == cmd.queue_id
        assert active[0].status == QueueStatus.ACTIVE
