"""Tests for the argparse-based watch command parser.

Verifies that the watch command parser:
- Builds an argparse.ArgumentParser with correct structure
- Parses job ID (--run-id / positional) correctly
- Parses follow mode (--follow / -f) as a boolean flag
- Parses output format (--format) with restricted choices
- Applies correct defaults for all optional arguments
- Returns WatchArgs on success and error strings on failure
- Handles edge cases (unknown flags, invalid format, etc.)
- Integrates with the existing CLI parser pipeline
"""

from __future__ import annotations

from jules_daemon.cli.watch_parser import (
    VALID_OUTPUT_FORMATS,
    build_watch_argparser,
    parse_watch_tokens,
)
from jules_daemon.cli.verbs import WatchArgs


# ---------------------------------------------------------------------------
# build_watch_argparser: structural tests
# ---------------------------------------------------------------------------


class TestBuildWatchArgparser:
    """Tests for the argparse.ArgumentParser factory."""

    def test_returns_argument_parser(self) -> None:
        parser = build_watch_argparser()
        # argparse.ArgumentParser has parse_args method
        assert callable(getattr(parser, "parse_args", None))

    def test_prog_name_is_watch(self) -> None:
        parser = build_watch_argparser()
        assert parser.prog == "watch"

    def test_parser_does_not_exit_on_error(self) -> None:
        """Parser should be configured to not call sys.exit."""
        parser = build_watch_argparser()
        assert parser.exit_on_error is False


# ---------------------------------------------------------------------------
# parse_watch_tokens: defaults
# ---------------------------------------------------------------------------


class TestParseWatchTokensDefaults:
    """Tests for default values when no flags are provided."""

    def test_bare_watch_returns_defaults(self) -> None:
        result = parse_watch_tokens([])
        assert isinstance(result, WatchArgs)
        assert result.run_id is None
        assert result.tail_lines == 50
        assert result.follow is False
        assert result.output_format == "text"

    def test_empty_tokens_list(self) -> None:
        result = parse_watch_tokens([])
        assert isinstance(result, WatchArgs)


# ---------------------------------------------------------------------------
# parse_watch_tokens: job ID (--run-id)
# ---------------------------------------------------------------------------


class TestParseWatchTokensRunId:
    """Tests for the --run-id flag."""

    def test_run_id_long_flag(self) -> None:
        result = parse_watch_tokens(["--run-id", "abc-123"])
        assert isinstance(result, WatchArgs)
        assert result.run_id == "abc-123"

    def test_run_id_equals_syntax(self) -> None:
        result = parse_watch_tokens(["--run-id=job-456"])
        assert isinstance(result, WatchArgs)
        assert result.run_id == "job-456"

    def test_run_id_with_uuid_format(self) -> None:
        result = parse_watch_tokens(
            ["--run-id", "550e8400-e29b-41d4-a716-446655440000"]
        )
        assert isinstance(result, WatchArgs)
        assert result.run_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_run_id_none_when_not_provided(self) -> None:
        result = parse_watch_tokens([])
        assert isinstance(result, WatchArgs)
        assert result.run_id is None

    def test_run_id_missing_value_returns_error(self) -> None:
        result = parse_watch_tokens(["--run-id"])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# parse_watch_tokens: follow mode (--follow / -f)
# ---------------------------------------------------------------------------


class TestParseWatchTokensFollow:
    """Tests for the --follow / -f flag."""

    def test_follow_long_flag(self) -> None:
        result = parse_watch_tokens(["--follow"])
        assert isinstance(result, WatchArgs)
        assert result.follow is True

    def test_follow_short_flag(self) -> None:
        result = parse_watch_tokens(["-f"])
        assert isinstance(result, WatchArgs)
        assert result.follow is True

    def test_follow_default_is_false(self) -> None:
        result = parse_watch_tokens([])
        assert isinstance(result, WatchArgs)
        assert result.follow is False

    def test_follow_combined_with_run_id(self) -> None:
        result = parse_watch_tokens(["--run-id", "abc", "--follow"])
        assert isinstance(result, WatchArgs)
        assert result.run_id == "abc"
        assert result.follow is True

    def test_follow_combined_with_tail(self) -> None:
        result = parse_watch_tokens(["-f", "--tail", "100"])
        assert isinstance(result, WatchArgs)
        assert result.follow is True
        assert result.tail_lines == 100


# ---------------------------------------------------------------------------
# parse_watch_tokens: output format (--format)
# ---------------------------------------------------------------------------


class TestParseWatchTokensFormat:
    """Tests for the --format flag."""

    def test_format_text(self) -> None:
        result = parse_watch_tokens(["--format", "text"])
        assert isinstance(result, WatchArgs)
        assert result.output_format == "text"

    def test_format_json(self) -> None:
        result = parse_watch_tokens(["--format", "json"])
        assert isinstance(result, WatchArgs)
        assert result.output_format == "json"

    def test_format_summary(self) -> None:
        result = parse_watch_tokens(["--format", "summary"])
        assert isinstance(result, WatchArgs)
        assert result.output_format == "summary"

    def test_format_default_is_text(self) -> None:
        result = parse_watch_tokens([])
        assert isinstance(result, WatchArgs)
        assert result.output_format == "text"

    def test_format_equals_syntax(self) -> None:
        result = parse_watch_tokens(["--format=json"])
        assert isinstance(result, WatchArgs)
        assert result.output_format == "json"

    def test_invalid_format_returns_error(self) -> None:
        result = parse_watch_tokens(["--format", "xml"])
        assert isinstance(result, str)
        assert "xml" in result.lower() or "format" in result.lower()

    def test_format_missing_value_returns_error(self) -> None:
        result = parse_watch_tokens(["--format"])
        assert isinstance(result, str)

    def test_valid_output_formats_constant(self) -> None:
        """Verify the exported constant contains expected formats."""
        assert "text" in VALID_OUTPUT_FORMATS
        assert "json" in VALID_OUTPUT_FORMATS
        assert "summary" in VALID_OUTPUT_FORMATS


# ---------------------------------------------------------------------------
# parse_watch_tokens: tail lines (--tail)
# ---------------------------------------------------------------------------


class TestParseWatchTokensTail:
    """Tests for the --tail flag."""

    def test_tail_long_flag(self) -> None:
        result = parse_watch_tokens(["--tail", "100"])
        assert isinstance(result, WatchArgs)
        assert result.tail_lines == 100

    def test_tail_equals_syntax(self) -> None:
        result = parse_watch_tokens(["--tail=200"])
        assert isinstance(result, WatchArgs)
        assert result.tail_lines == 200

    def test_tail_default(self) -> None:
        result = parse_watch_tokens([])
        assert isinstance(result, WatchArgs)
        assert result.tail_lines == 50

    def test_tail_invalid_returns_error(self) -> None:
        result = parse_watch_tokens(["--tail", "notanumber"])
        assert isinstance(result, str)

    def test_tail_zero_returns_error(self) -> None:
        result = parse_watch_tokens(["--tail", "0"])
        assert isinstance(result, str)
        assert "positive" in result.lower() or "tail" in result.lower()

    def test_tail_negative_returns_error(self) -> None:
        result = parse_watch_tokens(["--tail", "-5"])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# parse_watch_tokens: combination tests
# ---------------------------------------------------------------------------


class TestParseWatchTokensCombinations:
    """Tests for multiple flags combined."""

    def test_all_flags(self) -> None:
        result = parse_watch_tokens([
            "--run-id", "job-789",
            "--follow",
            "--format", "json",
            "--tail", "200",
        ])
        assert isinstance(result, WatchArgs)
        assert result.run_id == "job-789"
        assert result.follow is True
        assert result.output_format == "json"
        assert result.tail_lines == 200

    def test_short_follow_with_all_others(self) -> None:
        result = parse_watch_tokens([
            "-f",
            "--run-id", "abc",
            "--format=summary",
            "--tail=75",
        ])
        assert isinstance(result, WatchArgs)
        assert result.follow is True
        assert result.run_id == "abc"
        assert result.output_format == "summary"
        assert result.tail_lines == 75

    def test_flags_in_any_order(self) -> None:
        result = parse_watch_tokens([
            "--format", "json",
            "--tail", "10",
            "--run-id", "xyz",
            "-f",
        ])
        assert isinstance(result, WatchArgs)
        assert result.run_id == "xyz"
        assert result.follow is True
        assert result.output_format == "json"
        assert result.tail_lines == 10


# ---------------------------------------------------------------------------
# parse_watch_tokens: error cases
# ---------------------------------------------------------------------------


class TestParseWatchTokensErrors:
    """Tests for error handling."""

    def test_unknown_flag_returns_error(self) -> None:
        result = parse_watch_tokens(["--color"])
        assert isinstance(result, str)

    def test_unknown_short_flag_returns_error(self) -> None:
        result = parse_watch_tokens(["-x"])
        assert isinstance(result, str)

    def test_unexpected_positional_returns_error(self) -> None:
        """Positional arguments are not accepted."""
        result = parse_watch_tokens(["some-job-id"])
        assert isinstance(result, str)

    def test_never_raises_returns_error_string(self) -> None:
        """Parse errors are returned as strings, never raised."""
        result = parse_watch_tokens(["--tail", "abc", "--format", "bad"])
        assert isinstance(result, str)
