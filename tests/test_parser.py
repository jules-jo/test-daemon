"""Tests for the CLI verb parser.

Covers tokenization of raw input strings, verb matching, argument extraction
and validation, and structured error results for all six verbs.
"""

from __future__ import annotations

import pytest

from jules_daemon.cli.parser import (
    ParseError,
    parse_command,
    tokenize,
)
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


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenize:
    """Tests for the tokenize function."""

    def test_simple_words(self) -> None:
        assert tokenize("status --verbose") == ["status", "--verbose"]

    def test_single_quoted_string(self) -> None:
        tokens = tokenize("run user@host 'run the tests'")
        assert tokens == ["run", "user@host", "run the tests"]

    def test_double_quoted_string(self) -> None:
        tokens = tokenize('cancel --reason "tests are flaky"')
        assert tokens == ["cancel", "--reason", "tests are flaky"]

    def test_mixed_quotes(self) -> None:
        tokens = tokenize("""run user@host "run the 'full' suite" """)
        assert tokens == ["run", "user@host", "run the 'full' suite"]

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_whitespace_only(self) -> None:
        assert tokenize("   ") == []

    def test_preserves_inner_whitespace_in_quotes(self) -> None:
        tokens = tokenize('run user@host "run   regression   tests"')
        assert tokens == ["run", "user@host", "run   regression   tests"]

    def test_extra_surrounding_whitespace(self) -> None:
        tokens = tokenize("  status  --verbose  ")
        assert tokens == ["status", "--verbose"]

    def test_escaped_quote_in_double_quoted_string(self) -> None:
        tokens = tokenize(r'cancel --reason "it\"s broken"')
        assert len(tokens) == 3
        assert "broken" in tokens[2]

    def test_apostrophe_in_double_quotes(self) -> None:
        tokens = tokenize('cancel --reason "it\'s broken"')
        assert len(tokens) == 3
        assert "it's broken" == tokens[2]

    def test_tab_separated(self) -> None:
        tokens = tokenize("status\t--verbose")
        assert tokens == ["status", "--verbose"]


# ---------------------------------------------------------------------------
# Empty / malformed input
# ---------------------------------------------------------------------------


class TestParseCommandErrors:
    """Tests for error handling in parse_command."""

    def test_empty_input_returns_error(self) -> None:
        result = parse_command("")
        assert isinstance(result, ParseError)
        assert "empty" in result.message.lower()
        assert result.raw_input == ""
        assert result.verb is None

    def test_whitespace_only_returns_error(self) -> None:
        result = parse_command("   ")
        assert isinstance(result, ParseError)
        assert "empty" in result.message.lower()
        assert result.raw_input == "   "

    def test_unknown_verb_returns_error(self) -> None:
        result = parse_command("deploy staging")
        assert isinstance(result, ParseError)
        assert "Unknown verb" in result.message
        assert "'deploy'" in result.message
        assert result.verb is None

    def test_error_preserves_raw_input(self) -> None:
        raw = "  unknown_verb arg1 arg2  "
        result = parse_command(raw)
        assert isinstance(result, ParseError)
        assert result.raw_input == raw

    def test_unterminated_quote_returns_error(self) -> None:
        result = parse_command('run user@host "unterminated quote')
        assert isinstance(result, ParseError)
        assert "Tokenization error" in result.message
        assert result.verb is None


# ---------------------------------------------------------------------------
# status verb
# ---------------------------------------------------------------------------


class TestParseStatus:
    """Tests for parsing the status verb."""

    def test_bare_status(self) -> None:
        result = parse_command("status")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.STATUS
        assert isinstance(result.args, StatusArgs)
        assert result.args.verbose is False

    def test_status_verbose(self) -> None:
        result = parse_command("status --verbose")
        assert isinstance(result, ParsedCommand)
        assert result.args.verbose is True

    def test_status_short_flag(self) -> None:
        result = parse_command("status -v")
        assert isinstance(result, ParsedCommand)
        assert result.args.verbose is True

    def test_status_case_insensitive(self) -> None:
        result = parse_command("STATUS")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.STATUS

    def test_status_unknown_flag_returns_error(self) -> None:
        result = parse_command("status --unknown")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.STATUS
        assert "Unknown flag" in result.message
        assert "--unknown" in result.message


# ---------------------------------------------------------------------------
# watch verb
# ---------------------------------------------------------------------------


class TestParseWatch:
    """Tests for parsing the watch verb."""

    def test_bare_watch(self) -> None:
        result = parse_command("watch")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.WATCH
        assert isinstance(result.args, WatchArgs)
        assert result.args.run_id is None
        assert result.args.tail_lines == 50

    def test_watch_with_run_id(self) -> None:
        result = parse_command("watch --run-id abc-123")
        assert isinstance(result, ParsedCommand)
        assert result.args.run_id == "abc-123"

    def test_watch_with_tail(self) -> None:
        result = parse_command("watch --tail 100")
        assert isinstance(result, ParsedCommand)
        assert result.args.tail_lines == 100

    def test_watch_with_both_flags(self) -> None:
        result = parse_command("watch --run-id abc-123 --tail 200")
        assert isinstance(result, ParsedCommand)
        assert result.args.run_id == "abc-123"
        assert result.args.tail_lines == 200

    def test_watch_invalid_tail_returns_error(self) -> None:
        result = parse_command("watch --tail notanumber")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.WATCH
        assert "invalid int value" in result.message or "--tail" in result.message

    def test_watch_zero_tail_returns_error(self) -> None:
        result = parse_command("watch --tail 0")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.WATCH
        assert "tail_lines must be positive" in result.message

    def test_watch_negative_tail_via_equals_returns_error(self) -> None:
        """Negative values via = syntax are caught as invalid."""
        result = parse_command("watch --tail=-5")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.WATCH
        assert "tail_lines must be positive" in result.message

    def test_watch_missing_run_id_value_returns_error(self) -> None:
        result = parse_command("watch --run-id")
        assert isinstance(result, ParseError)
        assert "--run-id" in result.message

    def test_watch_unknown_flag_returns_error(self) -> None:
        result = parse_command("watch --color")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.WATCH
        assert "unrecognized arguments" in result.message or "--color" in result.message

    def test_watch_empty_tail_via_equals_returns_error(self) -> None:
        result = parse_command("watch --tail=")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.WATCH
        assert "invalid int value" in result.message or "--tail" in result.message

    def test_watch_follow_mode(self) -> None:
        result = parse_command("watch --follow")
        assert isinstance(result, ParsedCommand)
        assert result.args.follow is True

    def test_watch_follow_short_flag(self) -> None:
        result = parse_command("watch -f")
        assert isinstance(result, ParsedCommand)
        assert result.args.follow is True

    def test_watch_follow_default_false(self) -> None:
        result = parse_command("watch")
        assert isinstance(result, ParsedCommand)
        assert result.args.follow is False

    def test_watch_format_json(self) -> None:
        result = parse_command("watch --format json")
        assert isinstance(result, ParsedCommand)
        assert result.args.output_format == "json"

    def test_watch_format_summary(self) -> None:
        result = parse_command("watch --format summary")
        assert isinstance(result, ParsedCommand)
        assert result.args.output_format == "summary"

    def test_watch_format_text(self) -> None:
        result = parse_command("watch --format text")
        assert isinstance(result, ParsedCommand)
        assert result.args.output_format == "text"

    def test_watch_format_default_text(self) -> None:
        result = parse_command("watch")
        assert isinstance(result, ParsedCommand)
        assert result.args.output_format == "text"

    def test_watch_invalid_format_returns_error(self) -> None:
        result = parse_command("watch --format xml")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.WATCH

    def test_watch_all_flags_combined(self) -> None:
        result = parse_command("watch --run-id job-1 --follow --format json --tail 75")
        assert isinstance(result, ParsedCommand)
        assert result.args.run_id == "job-1"
        assert result.args.follow is True
        assert result.args.output_format == "json"
        assert result.args.tail_lines == 75


# ---------------------------------------------------------------------------
# run verb
# ---------------------------------------------------------------------------


class TestParseRun:
    """Tests for parsing the run verb."""

    def test_minimal_run(self) -> None:
        result = parse_command("run deploy@staging.example.com run the test suite")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN
        assert isinstance(result.args, RunArgs)
        assert result.args.target_host == "staging.example.com"
        assert result.args.target_user == "deploy"
        assert result.args.natural_language == "run the test suite"
        assert result.args.target_port == 22
        assert result.args.key_path is None

    def test_run_with_port(self) -> None:
        result = parse_command("run deploy@host:2222 run the tests")
        assert isinstance(result, ParsedCommand)
        assert result.args.target_host == "host"
        assert result.args.target_port == 2222

    def test_run_with_key(self) -> None:
        result = parse_command(
            "run deploy@host run tests --key /home/deploy/.ssh/id_rsa"
        )
        assert isinstance(result, ParsedCommand)
        assert result.args.key_path == "/home/deploy/.ssh/id_rsa"

    def test_run_with_port_flag(self) -> None:
        result = parse_command("run deploy@host run tests --port 2222")
        assert isinstance(result, ParsedCommand)
        assert result.args.target_port == 2222

    def test_run_quoted_natural_language(self) -> None:
        result = parse_command(
            'run deploy@host "run the full regression suite for payments"'
        )
        assert isinstance(result, ParsedCommand)
        assert result.args.natural_language == "run the full regression suite for payments"

    def test_run_missing_target_returns_error(self) -> None:
        result = parse_command("run")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN
        assert "requires a target" in result.message

    def test_run_missing_natural_language_returns_error(self) -> None:
        result = parse_command("run deploy@host")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN
        assert "natural-language command" in result.message

    def test_run_invalid_target_format_returns_error(self) -> None:
        result = parse_command("run invalid-target run the tests")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN
        assert "Invalid SSH target" in result.message
        assert "user@host" in result.message

    def test_run_invalid_port_returns_error(self) -> None:
        result = parse_command("run deploy@host:notaport run tests")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN
        assert "not a number" in result.message

    def test_run_port_out_of_range_high_returns_error(self) -> None:
        result = parse_command("run deploy@host:99999 run tests")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN
        assert "1-65535" in result.message

    def test_run_port_out_of_range_zero_returns_error(self) -> None:
        result = parse_command("run deploy@host:0 run tests")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN
        assert "1-65535" in result.message

    def test_run_relative_key_path_returns_error(self) -> None:
        result = parse_command("run deploy@host run tests --key relative/key")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN
        assert "absolute path" in result.message

    def test_run_natural_language_with_multiple_words(self) -> None:
        result = parse_command(
            "run ci@prod run the full regression suite for the payments module"
        )
        assert isinstance(result, ParsedCommand)
        assert "payments module" in result.args.natural_language

    def test_run_port_inline_overridden_by_flag(self) -> None:
        """Flag port takes precedence over inline port."""
        result = parse_command("run deploy@host:2222 run tests --port 3333")
        assert isinstance(result, ParsedCommand)
        assert result.args.target_port == 3333

    def test_run_case_insensitive_verb(self) -> None:
        result = parse_command("RUN deploy@host run tests")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.RUN

    def test_run_empty_user_returns_error(self) -> None:
        result = parse_command("run @host run tests")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN
        assert "user must not be empty" in result.message

    def test_run_empty_host_returns_error(self) -> None:
        result = parse_command("run user@ run tests")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN
        assert "host must not be empty" in result.message

    def test_run_port_flag_missing_value_returns_error(self) -> None:
        result = parse_command("run deploy@host run tests --port")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN
        assert "--port requires a value" in result.message

    def test_run_unknown_flag_returns_error(self) -> None:
        result = parse_command("run deploy@host run tests --timeout 30")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN
        assert "Unknown flag" in result.message


# ---------------------------------------------------------------------------
# queue verb
# ---------------------------------------------------------------------------


class TestParseQueue:
    """Tests for parsing the queue verb."""

    def test_minimal_queue(self) -> None:
        result = parse_command("queue deploy@staging run smoke tests")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.QUEUE
        assert isinstance(result.args, QueueArgs)
        assert result.args.target_host == "staging"
        assert result.args.target_user == "deploy"
        assert result.args.natural_language == "run smoke tests"
        assert result.args.priority == 0

    def test_queue_with_priority(self) -> None:
        result = parse_command("queue deploy@host run tests --priority 5")
        assert isinstance(result, ParsedCommand)
        assert result.args.priority == 5

    def test_queue_with_port_and_key(self) -> None:
        result = parse_command(
            "queue deploy@host:2222 run tests --key /home/ci/.ssh/id_rsa"
        )
        assert isinstance(result, ParsedCommand)
        assert result.args.target_port == 2222
        assert result.args.key_path == "/home/ci/.ssh/id_rsa"

    def test_queue_missing_target_returns_error(self) -> None:
        result = parse_command("queue")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.QUEUE
        assert "requires a target" in result.message

    def test_queue_missing_natural_language_returns_error(self) -> None:
        result = parse_command("queue deploy@host")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.QUEUE
        assert "natural-language command" in result.message

    def test_queue_negative_priority_returns_error(self) -> None:
        result = parse_command("queue deploy@host run tests --priority=-1")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.QUEUE
        assert "priority must not be negative" in result.message

    def test_queue_invalid_priority_returns_error(self) -> None:
        result = parse_command("queue deploy@host run tests --priority abc")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.QUEUE
        assert "--priority must be a number" in result.message

    def test_queue_empty_user_returns_error(self) -> None:
        result = parse_command("queue @host run tests")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.QUEUE
        assert "user must not be empty" in result.message

    def test_queue_port_flag_missing_value_returns_error(self) -> None:
        result = parse_command("queue deploy@host run tests --port")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.QUEUE
        assert "--port requires a value" in result.message


# ---------------------------------------------------------------------------
# cancel verb
# ---------------------------------------------------------------------------


class TestParseCancel:
    """Tests for parsing the cancel verb."""

    def test_bare_cancel(self) -> None:
        result = parse_command("cancel")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.CANCEL
        assert isinstance(result.args, CancelArgs)
        assert result.args.run_id is None
        assert result.args.force is False
        assert result.args.reason is None

    def test_cancel_with_run_id(self) -> None:
        result = parse_command("cancel --run-id abc-123")
        assert isinstance(result, ParsedCommand)
        assert result.args.run_id == "abc-123"

    def test_cancel_with_force(self) -> None:
        result = parse_command("cancel --force")
        assert isinstance(result, ParsedCommand)
        assert result.args.force is True

    def test_cancel_with_reason(self) -> None:
        result = parse_command('cancel --reason "tests are flaky"')
        assert isinstance(result, ParsedCommand)
        assert result.args.reason == "tests are flaky"

    def test_cancel_all_flags(self) -> None:
        result = parse_command(
            'cancel --run-id abc-123 --force --reason "too slow"'
        )
        assert isinstance(result, ParsedCommand)
        assert result.args.run_id == "abc-123"
        assert result.args.force is True
        assert result.args.reason == "too slow"

    def test_cancel_short_force_flag(self) -> None:
        result = parse_command("cancel -f")
        assert isinstance(result, ParsedCommand)
        assert result.args.force is True

    def test_cancel_unknown_flag_returns_error(self) -> None:
        result = parse_command("cancel --unknown")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.CANCEL
        assert "Unknown flag" in result.message

    def test_cancel_missing_reason_value_returns_error(self) -> None:
        result = parse_command("cancel --reason")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.CANCEL
        assert "--reason requires a value" in result.message

    def test_cancel_missing_run_id_value_returns_error(self) -> None:
        result = parse_command("cancel --run-id")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.CANCEL
        assert "--run-id requires a value" in result.message


# ---------------------------------------------------------------------------
# history verb
# ---------------------------------------------------------------------------


class TestParseHistory:
    """Tests for parsing the history verb."""

    def test_bare_history(self) -> None:
        result = parse_command("history")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.HISTORY
        assert isinstance(result.args, HistoryArgs)
        assert result.args.limit == 20
        assert result.args.status_filter is None
        assert result.args.host_filter is None
        assert result.args.verbose is False

    def test_history_with_limit(self) -> None:
        result = parse_command("history --limit 50")
        assert isinstance(result, ParsedCommand)
        assert result.args.limit == 50

    def test_history_with_status(self) -> None:
        result = parse_command("history --status completed")
        assert isinstance(result, ParsedCommand)
        assert result.args.status_filter == "completed"

    def test_history_with_host(self) -> None:
        result = parse_command("history --host staging.example.com")
        assert isinstance(result, ParsedCommand)
        assert result.args.host_filter == "staging.example.com"

    def test_history_verbose(self) -> None:
        result = parse_command("history --verbose")
        assert isinstance(result, ParsedCommand)
        assert result.args.verbose is True

    def test_history_short_verbose(self) -> None:
        result = parse_command("history -v")
        assert isinstance(result, ParsedCommand)
        assert result.args.verbose is True

    def test_history_all_flags(self) -> None:
        result = parse_command(
            "history --limit 100 --status failed --host prod.example.com --verbose"
        )
        assert isinstance(result, ParsedCommand)
        assert result.args.limit == 100
        assert result.args.status_filter == "failed"
        assert result.args.host_filter == "prod.example.com"
        assert result.args.verbose is True

    def test_history_invalid_limit_returns_error(self) -> None:
        result = parse_command("history --limit abc")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.HISTORY
        assert "--limit must be a number" in result.message

    def test_history_zero_limit_returns_error(self) -> None:
        result = parse_command("history --limit 0")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.HISTORY
        assert "limit must be positive" in result.message

    def test_history_limit_exceeds_max_returns_error(self) -> None:
        result = parse_command("history --limit 5000")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.HISTORY
        assert "must not exceed 1000" in result.message

    def test_history_invalid_status_returns_error(self) -> None:
        result = parse_command("history --status bogus")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.HISTORY
        assert "Invalid status_filter" in result.message

    def test_history_unknown_flag_returns_error(self) -> None:
        result = parse_command("history --color")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.HISTORY
        assert "Unknown flag" in result.message

    def test_history_limit_missing_value_returns_error(self) -> None:
        result = parse_command("history --limit")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.HISTORY
        assert "--limit requires a value" in result.message

    def test_history_status_missing_value_returns_error(self) -> None:
        result = parse_command("history --status")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.HISTORY
        assert "--status requires a value" in result.message

    def test_history_host_missing_value_returns_error(self) -> None:
        result = parse_command("history --host")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.HISTORY
        assert "--host requires a value" in result.message


# ---------------------------------------------------------------------------
# ParseError properties
# ---------------------------------------------------------------------------


class TestParseError:
    """Tests for the ParseError dataclass."""

    def test_is_frozen(self) -> None:
        err = ParseError(message="test", raw_input="bad")
        with pytest.raises(AttributeError):
            err.message = "changed"  # type: ignore[misc]

    def test_verb_none_for_bad_verb(self) -> None:
        err = ParseError(message="unknown", raw_input="x", verb=None)
        assert err.verb is None

    def test_verb_set_for_arg_error(self) -> None:
        err = ParseError(
            message="bad args", raw_input="watch --tail x", verb=Verb.WATCH
        )
        assert err.verb == Verb.WATCH

    def test_str_representation(self) -> None:
        err = ParseError(message="Empty input", raw_input="")
        assert "Empty input" in str(err)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestParseEdgeCases:
    """Edge-case tests for the parser."""

    def test_verb_with_trailing_whitespace(self) -> None:
        result = parse_command("status   ")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.STATUS

    def test_verb_with_leading_whitespace(self) -> None:
        result = parse_command("   status")
        assert isinstance(result, ParsedCommand)
        assert result.verb == Verb.STATUS

    def test_mixed_case_flags(self) -> None:
        """Flags should be case-insensitive."""
        result = parse_command("status --VERBOSE")
        assert isinstance(result, ParsedCommand)
        assert result.args.verbose is True

    def test_equals_sign_flag_syntax(self) -> None:
        """Support --flag=value syntax."""
        result = parse_command("watch --tail=100")
        assert isinstance(result, ParsedCommand)
        assert result.args.tail_lines == 100

    def test_run_with_flags_between_nl_words(self) -> None:
        """Flags in run/queue are extracted, rest is natural language."""
        result = parse_command(
            "run deploy@host run regression tests --key /home/ci/.ssh/id_rsa"
        )
        assert isinstance(result, ParsedCommand)
        assert result.args.key_path == "/home/ci/.ssh/id_rsa"
        assert "run regression tests" == result.args.natural_language

    def test_run_empty_host_with_port_returns_error(self) -> None:
        """user@:2222 has empty host."""
        result = parse_command("run user@:2222 run tests")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.RUN
        assert "host must not be empty" in result.message

    def test_negative_number_as_flag_name(self) -> None:
        """Tokens starting with - are flags, not negative values.
        This documents the intentional behavior: use --flag=-N syntax
        for negative values.
        """
        result = parse_command("watch --tail -5")
        assert isinstance(result, ParseError)
        assert result.verb == Verb.WATCH
        # -5 is treated as an unknown flag, not as the value of --tail
        assert "-5" in result.message

    def test_queue_port_via_equals_syntax(self) -> None:
        result = parse_command("queue deploy@host run tests --port=8080")
        assert isinstance(result, ParsedCommand)
        assert result.args.target_port == 8080

    def test_history_limit_via_equals_syntax(self) -> None:
        result = parse_command("history --limit=42")
        assert isinstance(result, ParsedCommand)
        assert result.args.limit == 42

    def test_cancel_reason_via_equals_syntax(self) -> None:
        result = parse_command("cancel --reason=flaky")
        assert isinstance(result, ParsedCommand)
        assert result.args.reason == "flaky"
