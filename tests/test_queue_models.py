"""Tests for command queue data models."""

from datetime import datetime, timezone

import pytest

from jules_daemon.wiki.queue_models import (
    QueuedCommand,
    QueuePriority,
    QueueStatus,
)


class TestQueueStatus:
    def test_all_statuses_have_values(self) -> None:
        assert QueueStatus.QUEUED.value == "queued"
        assert QueueStatus.ACTIVE.value == "active"
        assert QueueStatus.COMPLETED.value == "completed"
        assert QueueStatus.FAILED.value == "failed"
        assert QueueStatus.CANCELLED.value == "cancelled"

    def test_terminal_statuses(self) -> None:
        assert QueueStatus.COMPLETED.is_terminal
        assert QueueStatus.FAILED.is_terminal
        assert QueueStatus.CANCELLED.is_terminal
        assert not QueueStatus.QUEUED.is_terminal
        assert not QueueStatus.ACTIVE.is_terminal

    def test_pending_statuses(self) -> None:
        assert QueueStatus.QUEUED.is_pending
        assert not QueueStatus.ACTIVE.is_pending
        assert not QueueStatus.COMPLETED.is_pending


class TestQueuePriority:
    def test_ordering(self) -> None:
        assert QueuePriority.NORMAL.value < QueuePriority.HIGH.value
        assert QueuePriority.HIGH.value < QueuePriority.URGENT.value

    def test_default_is_normal(self) -> None:
        cmd = QueuedCommand(
            natural_language="run tests",
        )
        assert cmd.priority == QueuePriority.NORMAL


class TestQueuedCommand:
    def test_creation_with_defaults(self) -> None:
        cmd = QueuedCommand(natural_language="run the tests")
        assert cmd.natural_language == "run the tests"
        assert cmd.status == QueueStatus.QUEUED
        assert cmd.priority == QueuePriority.NORMAL
        assert cmd.sequence > 0
        assert cmd.queue_id != ""
        assert cmd.queued_at is not None
        assert cmd.started_at is None
        assert cmd.completed_at is None
        assert cmd.error is None
        assert cmd.ssh_host is None
        assert cmd.ssh_user is None

    def test_frozen_immutability(self) -> None:
        cmd = QueuedCommand(natural_language="run tests")
        with pytest.raises(AttributeError):
            cmd.status = QueueStatus.ACTIVE  # type: ignore[misc]

    def test_empty_natural_language_raises(self) -> None:
        with pytest.raises(ValueError, match="natural_language must not be empty"):
            QueuedCommand(natural_language="")

    def test_whitespace_only_natural_language_raises(self) -> None:
        with pytest.raises(ValueError, match="natural_language must not be empty"):
            QueuedCommand(natural_language="   ")

    def test_negative_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="sequence must be positive"):
            QueuedCommand(natural_language="run tests", sequence=0)

    def test_negative_sequence_number_raises(self) -> None:
        with pytest.raises(ValueError, match="sequence must be positive"):
            QueuedCommand(natural_language="run tests", sequence=-1)

    def test_with_ssh_target(self) -> None:
        cmd = QueuedCommand(
            natural_language="run tests",
            ssh_host="staging.example.com",
            ssh_user="deploy",
            ssh_port=2222,
        )
        assert cmd.ssh_host == "staging.example.com"
        assert cmd.ssh_user == "deploy"
        assert cmd.ssh_port == 2222

    def test_with_activated(self) -> None:
        cmd = QueuedCommand(natural_language="run tests")
        activated = cmd.with_activated()
        assert activated.status == QueueStatus.ACTIVE
        assert activated.started_at is not None
        assert activated.queue_id == cmd.queue_id
        assert activated.sequence == cmd.sequence

    def test_with_completed(self) -> None:
        cmd = QueuedCommand(natural_language="run tests").with_activated()
        completed = cmd.with_completed()
        assert completed.status == QueueStatus.COMPLETED
        assert completed.completed_at is not None

    def test_with_failed(self) -> None:
        cmd = QueuedCommand(natural_language="run tests").with_activated()
        failed = cmd.with_failed("SSH connection refused")
        assert failed.status == QueueStatus.FAILED
        assert failed.error == "SSH connection refused"
        assert failed.completed_at is not None

    def test_with_cancelled(self) -> None:
        cmd = QueuedCommand(natural_language="run tests")
        cancelled = cmd.with_cancelled()
        assert cancelled.status == QueueStatus.CANCELLED
        assert cancelled.completed_at is not None

    def test_file_stem(self) -> None:
        cmd = QueuedCommand(
            natural_language="run tests",
            sequence=42,
            queue_id="abc-123",
        )
        assert cmd.file_stem == "000042-abc-123"

    def test_file_stem_zero_pads(self) -> None:
        cmd = QueuedCommand(
            natural_language="run tests",
            sequence=7,
            queue_id="x",
        )
        assert cmd.file_stem.startswith("000007-")

    def test_sort_key_orders_by_priority_then_sequence(self) -> None:
        normal = QueuedCommand(
            natural_language="a",
            sequence=1,
            priority=QueuePriority.NORMAL,
        )
        high = QueuedCommand(
            natural_language="b",
            sequence=2,
            priority=QueuePriority.HIGH,
        )
        urgent = QueuedCommand(
            natural_language="c",
            sequence=3,
            priority=QueuePriority.URGENT,
        )
        normal_later = QueuedCommand(
            natural_language="d",
            sequence=10,
            priority=QueuePriority.NORMAL,
        )
        # Urgent first (highest priority value sorts earliest = reversed),
        # then high, then normal by sequence
        items = sorted([normal_later, normal, high, urgent], key=lambda c: c.sort_key)
        assert items[0].priority == QueuePriority.URGENT
        assert items[1].priority == QueuePriority.HIGH
        assert items[2].sequence == 1  # normal, earlier
        assert items[3].sequence == 10  # normal, later
