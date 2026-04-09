"""Tests for the structured command parser with verb registry normalization.

Validates that the structured parser:
1. Pattern-matches verb-style input strings (both exact and aliased verbs)
2. Extracts the verb and positional/keyword arguments
3. Normalizes the verb via the canonical verb registry
4. Returns structured results (ParsedCommand or ParseError)
5. Produces ClassificationResult for pre-LLM deterministic classification
"""

from __future__ import annotations

import pytest

from jules_daemon.cli.parser import ParseError, parse_command
from jules_daemon.cli.verbs import (
    CancelArgs,
    HistoryArgs,
    ParsedCommand,
    QueueArgs,
    RunArgs,
    StatusArgs,
    Verb,
    WatchArgs,
)
from jules_daemon.classifier.models import ClassificationResult, InputType


# ---------------------------------------------------------------------------
# Verb alias normalization through parse_command
# ---------------------------------------------------------------------------


class TestParseCommandVerbAliases:
    """Verify verb aliases resolve correctly through parse_command."""

    # -- run aliases --

    def test_execute_normalizes_to_run(self) -> None:
        result = parse_command("execute deploy@host run the tests")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN
        assert isinstance(result.args, RunArgs)
        assert result.args.target_host == "host"
        assert result.args.target_user == "deploy"
        assert result.args.natural_language == "run the tests"

    def test_exec_normalizes_to_run(self) -> None:
        result = parse_command("exec deploy@staging run smoke tests")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN
        assert result.args.target_host == "staging"

    def test_start_normalizes_to_run(self) -> None:
        result = parse_command("start ci@prod run full regression")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN
        assert result.args.natural_language == "run full regression"

    def test_launch_normalizes_to_run(self) -> None:
        result = parse_command("launch deploy@host run tests --port 2222")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN
        assert result.args.target_port == 2222

    def test_begin_normalizes_to_run(self) -> None:
        result = parse_command("begin deploy@host run the unit tests")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN

    def test_test_normalizes_to_run(self) -> None:
        result = parse_command("test deploy@host run the unit tests")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN

    def test_trigger_normalizes_to_run(self) -> None:
        result = parse_command("trigger deploy@host run smoke tests")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN

    def test_kick_normalizes_to_run(self) -> None:
        result = parse_command("kick deploy@host run the suite")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN

    # -- status aliases --

    def test_check_normalizes_to_status(self) -> None:
        result = parse_command("check")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.STATUS
        assert isinstance(result.args, StatusArgs)

    def test_state_normalizes_to_status(self) -> None:
        result = parse_command("state --verbose")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.STATUS
        assert result.args.verbose is True

    def test_info_normalizes_to_status(self) -> None:
        result = parse_command("info")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.STATUS

    def test_progress_normalizes_to_status(self) -> None:
        result = parse_command("progress -v")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.STATUS
        assert result.args.verbose is True

    def test_ping_normalizes_to_status(self) -> None:
        result = parse_command("ping")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.STATUS

    # -- cancel aliases --

    def test_stop_normalizes_to_cancel(self) -> None:
        result = parse_command("stop --force")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.CANCEL
        assert isinstance(result.args, CancelArgs)
        assert result.args.force is True

    def test_abort_normalizes_to_cancel(self) -> None:
        result = parse_command("abort --run-id abc-123")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.CANCEL
        assert result.args.run_id == "abc-123"

    def test_kill_normalizes_to_cancel(self) -> None:
        result = parse_command("kill")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.CANCEL

    def test_terminate_normalizes_to_cancel(self) -> None:
        result = parse_command('terminate --reason "flaky test"')
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.CANCEL
        assert result.args.reason == "flaky test"

    def test_halt_normalizes_to_cancel(self) -> None:
        result = parse_command("halt -f")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.CANCEL
        assert result.args.force is True

    # -- watch aliases --

    def test_tail_normalizes_to_watch(self) -> None:
        result = parse_command("tail --tail 100")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.WATCH
        assert isinstance(result.args, WatchArgs)
        assert result.args.tail_lines == 100

    def test_follow_normalizes_to_watch(self) -> None:
        result = parse_command("follow --run-id run-42")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.WATCH
        assert result.args.run_id == "run-42"

    def test_stream_normalizes_to_watch(self) -> None:
        result = parse_command("stream")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.WATCH

    def test_monitor_normalizes_to_watch(self) -> None:
        result = parse_command("monitor")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.WATCH

    def test_logs_normalizes_to_watch(self) -> None:
        result = parse_command("logs --tail 200")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.WATCH
        assert result.args.tail_lines == 200

    def test_output_normalizes_to_watch(self) -> None:
        result = parse_command("output")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.WATCH

    def test_attach_normalizes_to_watch(self) -> None:
        result = parse_command("attach --run-id xyz")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.WATCH
        assert result.args.run_id == "xyz"

    # -- queue aliases --

    def test_enqueue_normalizes_to_queue(self) -> None:
        result = parse_command("enqueue deploy@host run tests")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.QUEUE
        assert isinstance(result.args, QueueArgs)

    def test_schedule_normalizes_to_queue(self) -> None:
        result = parse_command("schedule deploy@host run tests --priority 5")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.QUEUE
        assert result.args.priority == 5

    def test_defer_normalizes_to_queue(self) -> None:
        result = parse_command("defer deploy@host run tests")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.QUEUE

    def test_later_normalizes_to_queue(self) -> None:
        result = parse_command("later deploy@host run tests")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.QUEUE

    # -- history aliases --

    def test_past_normalizes_to_history(self) -> None:
        result = parse_command("past --limit 50")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.HISTORY
        assert isinstance(result.args, HistoryArgs)
        assert result.args.limit == 50

    def test_results_normalizes_to_history(self) -> None:
        result = parse_command("results --status completed")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.HISTORY
        assert result.args.status_filter == "completed"

    def test_previous_normalizes_to_history(self) -> None:
        result = parse_command("previous --verbose")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.HISTORY
        assert result.args.verbose is True

    def test_log_normalizes_to_history(self) -> None:
        result = parse_command("log --host prod.example.com")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.HISTORY
        assert result.args.host_filter == "prod.example.com"

    def test_report_normalizes_to_history(self) -> None:
        result = parse_command("report")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.HISTORY

    def test_reports_normalizes_to_history(self) -> None:
        result = parse_command("reports --limit 10 -v")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.HISTORY
        assert result.args.limit == 10
        assert result.args.verbose is True


class TestParseCommandAliasCaseHandling:
    """Verify case-insensitive alias normalization."""

    def test_uppercase_alias(self) -> None:
        result = parse_command("EXECUTE deploy@host run tests")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN

    def test_mixed_case_alias(self) -> None:
        result = parse_command("Execute deploy@host run tests")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN

    def test_whitespace_padded_alias(self) -> None:
        result = parse_command("  check  ")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.STATUS

    def test_uppercase_stop_alias(self) -> None:
        result = parse_command("STOP --force")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.CANCEL
        assert result.args.force is True


class TestParseCommandAliasErrors:
    """Verify error handling still works for truly unknown verbs."""

    def test_unrecognized_word_returns_error(self) -> None:
        result = parse_command("obliterate deploy@host")
        assert isinstance(result, ParseError)
        assert result.verb is None
        assert "Unknown verb" in result.message

    def test_error_message_lists_valid_verbs(self) -> None:
        result = parse_command("deploy staging")
        assert isinstance(result, ParseError)
        assert "Valid verbs" in result.message or "valid verbs" in result.message.lower()

    def test_empty_input_still_returns_error(self) -> None:
        result = parse_command("")
        assert isinstance(result, ParseError)
        assert "empty" in result.message.lower()

    def test_alias_with_invalid_args_returns_error_with_verb(self) -> None:
        """When an alias resolves but args are invalid, verb should be set."""
        result = parse_command("execute")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN

    def test_stop_with_unknown_flag_returns_error_with_cancel_verb(self) -> None:
        result = parse_command("stop --unknown-flag")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.CANCEL
        assert "Unknown flag" in result.message


class TestParseCommandAliasedRunFull:
    """Full argument extraction tests for aliased run verbs."""

    def test_execute_with_port_and_key(self) -> None:
        result = parse_command(
            "execute deploy@host:2222 run tests --key /home/ci/.ssh/id_rsa"
        )
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN
        assert result.args.target_port == 2222
        assert result.args.key_path == "/home/ci/.ssh/id_rsa"

    def test_launch_with_quoted_natural_language(self) -> None:
        result = parse_command(
            'launch deploy@host "run the full regression suite for payments"'
        )
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN
        assert "regression suite for payments" in result.args.natural_language

    def test_start_with_port_override_flag(self) -> None:
        result = parse_command("start deploy@host:22 run tests --port 3333")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN
        assert result.args.target_port == 3333


class TestParseCommandAliasedQueueFull:
    """Full argument extraction tests for aliased queue verbs."""

    def test_schedule_with_priority_and_key(self) -> None:
        result = parse_command(
            "schedule deploy@host:2222 run tests "
            "--key /home/ci/.ssh/id_rsa --priority 10"
        )
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.QUEUE
        assert result.args.priority == 10
        assert result.args.key_path == "/home/ci/.ssh/id_rsa"

    def test_enqueue_missing_nl_returns_error(self) -> None:
        result = parse_command("enqueue deploy@host")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.QUEUE


class TestParseCommandAliasedCancelFull:
    """Full argument extraction tests for aliased cancel verbs."""

    def test_abort_all_flags(self) -> None:
        result = parse_command(
            'abort --run-id abc-123 --force --reason "too slow"'
        )
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.CANCEL
        assert result.args.run_id == "abc-123"
        assert result.args.force is True
        assert result.args.reason == "too slow"

    def test_kill_with_short_force(self) -> None:
        result = parse_command("kill -f")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.CANCEL
        assert result.args.force is True


# ---------------------------------------------------------------------------
# Structured classification result tests
# ---------------------------------------------------------------------------


class TestClassifyStructuredCommand:
    """Tests for the deterministic classify_structured_command function."""

    def test_exact_verb_high_confidence(self) -> None:
        from jules_daemon.cli.parser import classify_structured_command

        result = classify_structured_command("run deploy@host run tests")
        assert isinstance(result, ClassificationResult)
        assert result.canonical_verb == "run"
        assert result.confidence_score == 1.0
        assert result.input_type == InputType.COMMAND
        assert result.is_confident is True

    def test_alias_verb_high_confidence(self) -> None:
        from jules_daemon.cli.parser import classify_structured_command

        result = classify_structured_command("execute deploy@host run tests")
        assert isinstance(result, ClassificationResult)
        assert result.canonical_verb == "run"
        assert result.confidence_score == 0.9
        assert result.input_type == InputType.COMMAND
        assert result.is_confident is True

    def test_exact_status_high_confidence(self) -> None:
        from jules_daemon.cli.parser import classify_structured_command

        result = classify_structured_command("status --verbose")
        assert isinstance(result, ClassificationResult)
        assert result.canonical_verb == "status"
        assert result.confidence_score == 1.0

    def test_alias_check_produces_status(self) -> None:
        from jules_daemon.cli.parser import classify_structured_command

        result = classify_structured_command("check")
        assert isinstance(result, ClassificationResult)
        assert result.canonical_verb == "status"
        assert result.confidence_score == 0.9

    def test_unknown_verb_returns_none(self) -> None:
        from jules_daemon.cli.parser import classify_structured_command

        result = classify_structured_command("obliterate everything")
        assert result is None

    def test_empty_input_returns_none(self) -> None:
        from jules_daemon.cli.parser import classify_structured_command

        result = classify_structured_command("")
        assert result is None

    def test_extracted_args_contain_positionals(self) -> None:
        from jules_daemon.cli.parser import classify_structured_command

        result = classify_structured_command("stop --run-id abc-123 --force")
        assert result is not None
        assert result.canonical_verb == "cancel"
        assert "positional_args" in result.extracted_args
        assert "keyword_args" in result.extracted_args

    def test_keyword_args_extracted(self) -> None:
        from jules_daemon.cli.parser import classify_structured_command

        result = classify_structured_command("tail --tail 100")
        assert result is not None
        assert result.canonical_verb == "watch"
        assert result.extracted_args["keyword_args"]["--tail"] == "100"

    def test_whitespace_input_returns_none(self) -> None:
        from jules_daemon.cli.parser import classify_structured_command

        result = classify_structured_command("   ")
        assert result is None

    def test_unterminated_quote_returns_none(self) -> None:
        from jules_daemon.cli.parser import classify_structured_command

        result = classify_structured_command('run deploy@host "unterminated')
        assert result is None


class TestNormalizeVerb:
    """Tests for the normalize_verb function that bridges registry to Verb enum."""

    def test_exact_verb_returns_verb_enum(self) -> None:
        from jules_daemon.cli.parser import normalize_verb

        assert normalize_verb("run") == Verb.RUN
        assert normalize_verb("status") == Verb.STATUS
        assert normalize_verb("cancel") == Verb.CANCEL
        assert normalize_verb("watch") == Verb.WATCH
        assert normalize_verb("queue") == Verb.QUEUE
        assert normalize_verb("history") == Verb.HISTORY

    def test_alias_returns_canonical_verb_enum(self) -> None:
        from jules_daemon.cli.parser import normalize_verb

        assert normalize_verb("execute") == Verb.RUN
        assert normalize_verb("stop") == Verb.CANCEL
        assert normalize_verb("tail") == Verb.WATCH
        assert normalize_verb("enqueue") == Verb.QUEUE
        assert normalize_verb("check") == Verb.STATUS
        assert normalize_verb("past") == Verb.HISTORY

    def test_case_insensitive(self) -> None:
        from jules_daemon.cli.parser import normalize_verb

        assert normalize_verb("EXECUTE") == Verb.RUN
        assert normalize_verb("Stop") == Verb.CANCEL
        assert normalize_verb("TAIL") == Verb.WATCH

    def test_unknown_returns_none(self) -> None:
        from jules_daemon.cli.parser import normalize_verb

        assert normalize_verb("obliterate") is None
        assert normalize_verb("destroy") is None

    def test_empty_returns_none(self) -> None:
        from jules_daemon.cli.parser import normalize_verb

        assert normalize_verb("") is None
        assert normalize_verb("   ") is None
