"""Tests for CLI verb grammar data models.

Covers the six CLI verbs (status, watch, run, queue, cancel, history),
their argument schemas, validation rules, and the Verb enum.
"""

from __future__ import annotations

import pytest

from jules_daemon.cli.verbs import (
    CancelArgs,
    HistoryArgs,
    ParsedCommand,
    QueueArgs,
    RunArgs,
    StatusArgs,
    Verb,
    WatchArgs,
    parse_verb,
)


# ---------------------------------------------------------------------------
# Verb enum
# ---------------------------------------------------------------------------


class TestVerb:
    """Tests for the Verb enum."""

    def test_has_six_members(self) -> None:
        assert len(Verb) == 6

    def test_status_value(self) -> None:
        assert Verb.STATUS.value == "status"

    def test_watch_value(self) -> None:
        assert Verb.WATCH.value == "watch"

    def test_run_value(self) -> None:
        assert Verb.RUN.value == "run"

    def test_queue_value(self) -> None:
        assert Verb.QUEUE.value == "queue"

    def test_cancel_value(self) -> None:
        assert Verb.CANCEL.value == "cancel"

    def test_history_value(self) -> None:
        assert Verb.HISTORY.value == "history"


class TestParseVerb:
    """Tests for the parse_verb factory function."""

    def test_parse_status(self) -> None:
        assert parse_verb("status") == Verb.STATUS

    def test_parse_watch(self) -> None:
        assert parse_verb("watch") == Verb.WATCH

    def test_parse_run(self) -> None:
        assert parse_verb("run") == Verb.RUN

    def test_parse_queue(self) -> None:
        assert parse_verb("queue") == Verb.QUEUE

    def test_parse_cancel(self) -> None:
        assert parse_verb("cancel") == Verb.CANCEL

    def test_parse_history(self) -> None:
        assert parse_verb("history") == Verb.HISTORY

    def test_parse_case_insensitive(self) -> None:
        assert parse_verb("STATUS") == Verb.STATUS
        assert parse_verb("Run") == Verb.RUN
        assert parse_verb("HISTORY") == Verb.HISTORY

    def test_parse_strips_whitespace(self) -> None:
        assert parse_verb("  status  ") == Verb.STATUS

    def test_parse_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Verb must not be empty"):
            parse_verb("")

    def test_parse_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Verb must not be empty"):
            parse_verb("   ")

    def test_parse_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown verb 'start'"):
            parse_verb("start")


# ---------------------------------------------------------------------------
# StatusArgs
# ---------------------------------------------------------------------------


class TestStatusArgs:
    """Tests for the status verb argument schema."""

    def test_defaults(self) -> None:
        args = StatusArgs()
        assert args.verbose is False

    def test_verbose_flag(self) -> None:
        args = StatusArgs(verbose=True)
        assert args.verbose is True

    def test_frozen(self) -> None:
        args = StatusArgs()
        with pytest.raises(AttributeError):
            args.verbose = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# WatchArgs
# ---------------------------------------------------------------------------


class TestWatchArgs:
    """Tests for the watch verb argument schema."""

    def test_defaults(self) -> None:
        args = WatchArgs()
        assert args.run_id is None
        assert args.tail_lines == 50

    def test_custom_run_id(self) -> None:
        args = WatchArgs(run_id="abc-123")
        assert args.run_id == "abc-123"

    def test_custom_tail_lines(self) -> None:
        args = WatchArgs(tail_lines=100)
        assert args.tail_lines == 100

    def test_frozen(self) -> None:
        args = WatchArgs()
        with pytest.raises(AttributeError):
            args.run_id = "new"  # type: ignore[misc]

    def test_tail_lines_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="tail_lines must be positive"):
            WatchArgs(tail_lines=0)

    def test_tail_lines_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="tail_lines must be positive"):
            WatchArgs(tail_lines=-1)

    def test_empty_run_id_raises(self) -> None:
        with pytest.raises(ValueError, match="run_id must not be empty"):
            WatchArgs(run_id="")

    def test_whitespace_run_id_raises(self) -> None:
        with pytest.raises(ValueError, match="run_id must not be empty"):
            WatchArgs(run_id="   ")


# ---------------------------------------------------------------------------
# RunArgs
# ---------------------------------------------------------------------------


class TestRunArgs:
    """Tests for the run verb argument schema."""

    def test_minimal(self) -> None:
        args = RunArgs(
            target_host="staging.example.com",
            target_user="deploy",
            natural_language="run the test suite",
        )
        assert args.target_host == "staging.example.com"
        assert args.target_user == "deploy"
        assert args.natural_language == "run the test suite"
        assert args.target_port == 22
        assert args.key_path is None

    def test_full_args(self) -> None:
        args = RunArgs(
            target_host="prod.example.com",
            target_user="ci",
            target_port=2222,
            key_path="/home/ci/.ssh/id_rsa",
            natural_language="run the regression suite for payments",
        )
        assert args.target_port == 2222
        assert args.key_path == "/home/ci/.ssh/id_rsa"

    def test_frozen(self) -> None:
        args = RunArgs(
            target_host="host",
            target_user="user",
            natural_language="run tests",
        )
        with pytest.raises(AttributeError):
            args.target_host = "other"  # type: ignore[misc]

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError, match="target_host must not be empty"):
            RunArgs(
                target_host="",
                target_user="user",
                natural_language="run tests",
            )

    def test_whitespace_host_raises(self) -> None:
        with pytest.raises(ValueError, match="target_host must not be empty"):
            RunArgs(
                target_host="   ",
                target_user="user",
                natural_language="run tests",
            )

    def test_empty_user_raises(self) -> None:
        with pytest.raises(ValueError, match="target_user must not be empty"):
            RunArgs(
                target_host="host",
                target_user="",
                natural_language="run tests",
            )

    def test_empty_natural_language_raises(self) -> None:
        with pytest.raises(
            ValueError, match="natural_language must not be empty"
        ):
            RunArgs(
                target_host="host",
                target_user="user",
                natural_language="",
            )

    def test_port_below_range_raises(self) -> None:
        with pytest.raises(ValueError, match="target_port must be 1-65535"):
            RunArgs(
                target_host="host",
                target_user="user",
                natural_language="run tests",
                target_port=0,
            )

    def test_port_above_range_raises(self) -> None:
        with pytest.raises(ValueError, match="target_port must be 1-65535"):
            RunArgs(
                target_host="host",
                target_user="user",
                natural_language="run tests",
                target_port=70000,
            )

    def test_relative_key_path_raises(self) -> None:
        with pytest.raises(ValueError, match="key_path must be an absolute"):
            RunArgs(
                target_host="host",
                target_user="user",
                natural_language="run tests",
                key_path="relative/path/key",
            )

    def test_to_ssh_target(self) -> None:
        args = RunArgs(
            target_host="staging.example.com",
            target_user="deploy",
            target_port=2222,
            key_path="/home/deploy/.ssh/id_rsa",
            natural_language="run tests",
        )
        target = args.to_ssh_target()
        assert target.host == "staging.example.com"
        assert target.user == "deploy"
        assert target.port == 2222
        assert target.key_path == "/home/deploy/.ssh/id_rsa"


# ---------------------------------------------------------------------------
# QueueArgs
# ---------------------------------------------------------------------------


class TestQueueArgs:
    """Tests for the queue verb argument schema."""

    def test_minimal(self) -> None:
        args = QueueArgs(
            target_host="staging.example.com",
            target_user="deploy",
            natural_language="run smoke tests",
        )
        assert args.target_host == "staging.example.com"
        assert args.priority == 0

    def test_with_priority(self) -> None:
        args = QueueArgs(
            target_host="host",
            target_user="user",
            natural_language="run tests",
            priority=5,
        )
        assert args.priority == 5

    def test_frozen(self) -> None:
        args = QueueArgs(
            target_host="host",
            target_user="user",
            natural_language="run tests",
        )
        with pytest.raises(AttributeError):
            args.priority = 10  # type: ignore[misc]

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError, match="target_host must not be empty"):
            QueueArgs(
                target_host="",
                target_user="user",
                natural_language="run tests",
            )

    def test_empty_user_raises(self) -> None:
        with pytest.raises(ValueError, match="target_user must not be empty"):
            QueueArgs(
                target_host="host",
                target_user="",
                natural_language="run tests",
            )

    def test_empty_natural_language_raises(self) -> None:
        with pytest.raises(
            ValueError, match="natural_language must not be empty"
        ):
            QueueArgs(
                target_host="host",
                target_user="user",
                natural_language="",
            )

    def test_negative_priority_raises(self) -> None:
        with pytest.raises(
            ValueError, match="priority must not be negative"
        ):
            QueueArgs(
                target_host="host",
                target_user="user",
                natural_language="run tests",
                priority=-1,
            )

    def test_port_below_range_raises(self) -> None:
        with pytest.raises(ValueError, match="target_port must be 1-65535"):
            QueueArgs(
                target_host="host",
                target_user="user",
                natural_language="run tests",
                target_port=0,
            )

    def test_to_ssh_target(self) -> None:
        args = QueueArgs(
            target_host="host",
            target_user="user",
            target_port=2222,
            natural_language="run tests",
        )
        target = args.to_ssh_target()
        assert target.host == "host"
        assert target.user == "user"
        assert target.port == 2222


# ---------------------------------------------------------------------------
# CancelArgs
# ---------------------------------------------------------------------------


class TestCancelArgs:
    """Tests for the cancel verb argument schema."""

    def test_defaults(self) -> None:
        args = CancelArgs()
        assert args.run_id is None
        assert args.force is False
        assert args.reason is None

    def test_with_run_id(self) -> None:
        args = CancelArgs(run_id="abc-123")
        assert args.run_id == "abc-123"

    def test_with_force(self) -> None:
        args = CancelArgs(force=True)
        assert args.force is True

    def test_with_reason(self) -> None:
        args = CancelArgs(reason="Tests are flaky today")
        assert args.reason == "Tests are flaky today"

    def test_frozen(self) -> None:
        args = CancelArgs()
        with pytest.raises(AttributeError):
            args.force = True  # type: ignore[misc]

    def test_empty_run_id_raises(self) -> None:
        with pytest.raises(ValueError, match="run_id must not be empty"):
            CancelArgs(run_id="")

    def test_whitespace_run_id_raises(self) -> None:
        with pytest.raises(ValueError, match="run_id must not be empty"):
            CancelArgs(run_id="   ")

    def test_empty_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="reason must not be empty"):
            CancelArgs(reason="")


# ---------------------------------------------------------------------------
# HistoryArgs
# ---------------------------------------------------------------------------


class TestHistoryArgs:
    """Tests for the history verb argument schema."""

    def test_defaults(self) -> None:
        args = HistoryArgs()
        assert args.limit == 20
        assert args.status_filter is None
        assert args.host_filter is None
        assert args.verbose is False

    def test_custom_limit(self) -> None:
        args = HistoryArgs(limit=50)
        assert args.limit == 50

    def test_status_filter(self) -> None:
        args = HistoryArgs(status_filter="completed")
        assert args.status_filter == "completed"

    def test_host_filter(self) -> None:
        args = HistoryArgs(host_filter="staging.example.com")
        assert args.host_filter == "staging.example.com"

    def test_frozen(self) -> None:
        args = HistoryArgs()
        with pytest.raises(AttributeError):
            args.limit = 100  # type: ignore[misc]

    def test_limit_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="limit must be positive"):
            HistoryArgs(limit=0)

    def test_limit_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="limit must be positive"):
            HistoryArgs(limit=-5)

    def test_limit_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError, match="limit must not exceed 1000"):
            HistoryArgs(limit=1001)

    def test_empty_status_filter_raises(self) -> None:
        with pytest.raises(
            ValueError, match="status_filter must not be empty"
        ):
            HistoryArgs(status_filter="")

    def test_invalid_status_filter_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid status_filter"):
            HistoryArgs(status_filter="unknown_state")

    def test_empty_host_filter_raises(self) -> None:
        with pytest.raises(ValueError, match="host_filter must not be empty"):
            HistoryArgs(host_filter="")

    def test_valid_status_filters(self) -> None:
        """All RunStatus values should be accepted."""
        for status in ("idle", "pending_approval", "running", "completed",
                       "failed", "cancelled"):
            args = HistoryArgs(status_filter=status)
            assert args.status_filter == status


# ---------------------------------------------------------------------------
# ParsedCommand
# ---------------------------------------------------------------------------


class TestParsedCommand:
    """Tests for the ParsedCommand composite type."""

    def test_status_command(self) -> None:
        cmd = ParsedCommand(verb=Verb.STATUS, args=StatusArgs(verbose=True))
        assert cmd.verb == Verb.STATUS
        assert isinstance(cmd.args, StatusArgs)
        assert cmd.args.verbose is True

    def test_run_command(self) -> None:
        run_args = RunArgs(
            target_host="host",
            target_user="user",
            natural_language="run tests",
        )
        cmd = ParsedCommand(verb=Verb.RUN, args=run_args)
        assert cmd.verb == Verb.RUN
        assert isinstance(cmd.args, RunArgs)
        assert cmd.args.natural_language == "run tests"

    def test_frozen(self) -> None:
        cmd = ParsedCommand(verb=Verb.STATUS, args=StatusArgs())
        with pytest.raises(AttributeError):
            cmd.verb = Verb.RUN  # type: ignore[misc]

    def test_verb_args_type_mismatch_raises(self) -> None:
        """ParsedCommand validates that args type matches verb."""
        with pytest.raises(ValueError, match="Expected StatusArgs.*got RunArgs"):
            ParsedCommand(
                verb=Verb.STATUS,
                args=RunArgs(
                    target_host="host",
                    target_user="user",
                    natural_language="run tests",
                ),
            )

    def test_watch_with_watch_args(self) -> None:
        cmd = ParsedCommand(
            verb=Verb.WATCH,
            args=WatchArgs(tail_lines=100),
        )
        assert cmd.verb == Verb.WATCH
        assert cmd.args.tail_lines == 100

    def test_queue_with_queue_args(self) -> None:
        queue_args = QueueArgs(
            target_host="host",
            target_user="user",
            natural_language="run tests",
            priority=3,
        )
        cmd = ParsedCommand(verb=Verb.QUEUE, args=queue_args)
        assert cmd.verb == Verb.QUEUE
        assert cmd.args.priority == 3

    def test_cancel_with_cancel_args(self) -> None:
        cmd = ParsedCommand(
            verb=Verb.CANCEL,
            args=CancelArgs(force=True, reason="flaky"),
        )
        assert cmd.verb == Verb.CANCEL
        assert cmd.args.force is True

    def test_history_with_history_args(self) -> None:
        cmd = ParsedCommand(
            verb=Verb.HISTORY,
            args=HistoryArgs(limit=50, status_filter="completed"),
        )
        assert cmd.verb == Verb.HISTORY
        assert cmd.args.limit == 50
