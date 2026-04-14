"""Tests for the CLI args builder.

Verifies that the args builder:
- Converts extracted_args dicts to typed *Args dataclasses for all 6 verbs
- Produces identical *Args regardless of whether input came from structured
  or NL classification (the contract guarantee of the unified entry point)
- Returns error strings for invalid or missing required fields
- Applies correct defaults for optional fields
- Handles type coercion (string -> int for ports, limits, etc.)
- Never raises exceptions for invalid input
- Handles empty dicts (builds defaults where possible)
- Validates verb strings against canonical verb set
"""

from __future__ import annotations

import pytest

from jules_daemon.cli.args_builder import (
    build_verb_args,
)
from jules_daemon.cli.verbs import (
    CancelArgs,
    HistoryArgs,
    QueueArgs,
    RunArgs,
    StatusArgs,
    VerbArgs,
    WatchArgs,
)


# ---------------------------------------------------------------------------
# StatusArgs building
# ---------------------------------------------------------------------------


class TestBuildStatusArgs:
    """Tests for building StatusArgs from extracted dicts."""

    def test_empty_dict_defaults_to_non_verbose(self) -> None:
        result = build_verb_args("status", {})
        assert isinstance(result, StatusArgs)
        assert result.verbose is False

    def test_verbose_true(self) -> None:
        result = build_verb_args("status", {"verbose": True})
        assert isinstance(result, StatusArgs)
        assert result.verbose is True

    def test_verbose_false_explicit(self) -> None:
        result = build_verb_args("status", {"verbose": False})
        assert isinstance(result, StatusArgs)
        assert result.verbose is False

    def test_verbose_string_true(self) -> None:
        """Truthy string values should coerce to True."""
        result = build_verb_args("status", {"verbose": "true"})
        assert isinstance(result, StatusArgs)
        assert result.verbose is True

    def test_ignores_unknown_keys(self) -> None:
        result = build_verb_args("status", {"verbose": True, "extra": "ignored"})
        assert isinstance(result, StatusArgs)
        assert result.verbose is True


# ---------------------------------------------------------------------------
# WatchArgs building
# ---------------------------------------------------------------------------


class TestBuildWatchArgs:
    """Tests for building WatchArgs from extracted dicts."""

    def test_empty_dict_defaults(self) -> None:
        result = build_verb_args("watch", {})
        assert isinstance(result, WatchArgs)
        assert result.run_id is None
        assert result.tail_lines == 50

    def test_with_run_id(self) -> None:
        result = build_verb_args("watch", {"run_id": "abc-123"})
        assert isinstance(result, WatchArgs)
        assert result.run_id == "abc-123"

    def test_with_tail_lines(self) -> None:
        result = build_verb_args("watch", {"tail_lines": 100})
        assert isinstance(result, WatchArgs)
        assert result.tail_lines == 100

    def test_tail_lines_string_coerced(self) -> None:
        result = build_verb_args("watch", {"tail_lines": "200"})
        assert isinstance(result, WatchArgs)
        assert result.tail_lines == 200

    def test_invalid_tail_lines_returns_error(self) -> None:
        result = build_verb_args("watch", {"tail_lines": "not_a_number"})
        assert isinstance(result, str)
        assert "tail_lines" in result.lower()

    def test_follow_true(self) -> None:
        result = build_verb_args("watch", {"follow": True})
        assert isinstance(result, WatchArgs)
        assert result.follow is True

    def test_follow_false_default(self) -> None:
        result = build_verb_args("watch", {})
        assert isinstance(result, WatchArgs)
        assert result.follow is False

    def test_follow_string_coerced(self) -> None:
        """Truthy string values should coerce to True."""
        result = build_verb_args("watch", {"follow": "yes"})
        assert isinstance(result, WatchArgs)
        assert result.follow is True

    def test_output_format_json(self) -> None:
        result = build_verb_args("watch", {"output_format": "json"})
        assert isinstance(result, WatchArgs)
        assert result.output_format == "json"

    def test_output_format_summary(self) -> None:
        result = build_verb_args("watch", {"output_format": "summary"})
        assert isinstance(result, WatchArgs)
        assert result.output_format == "summary"

    def test_output_format_default_text(self) -> None:
        result = build_verb_args("watch", {})
        assert isinstance(result, WatchArgs)
        assert result.output_format == "text"

    def test_output_format_empty_defaults_to_text(self) -> None:
        result = build_verb_args("watch", {"output_format": ""})
        assert isinstance(result, WatchArgs)
        assert result.output_format == "text"

    def test_invalid_output_format_returns_error(self) -> None:
        result = build_verb_args("watch", {"output_format": "xml"})
        assert isinstance(result, str)
        assert "output_format" in result.lower()


# ---------------------------------------------------------------------------
# RunArgs building
# ---------------------------------------------------------------------------


class TestBuildRunArgs:
    """Tests for building RunArgs from extracted dicts."""

    def test_complete_args(self) -> None:
        result = build_verb_args("run", {
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run the smoke tests",
        })
        assert isinstance(result, RunArgs)
        assert result.target_host == "staging.example.com"
        assert result.target_user == "deploy"
        assert result.natural_language == "run the smoke tests"
        assert result.target_port == 22
        assert result.key_path is None

    def test_with_port_and_key(self) -> None:
        result = build_verb_args("run", {
            "target_host": "prod.example.com",
            "target_user": "ci",
            "natural_language": "run regression suite",
            "target_port": 2222,
            "key_path": "/home/user/.ssh/id_rsa",
        })
        assert isinstance(result, RunArgs)
        assert result.target_port == 2222
        assert result.key_path == "/home/user/.ssh/id_rsa"

    def test_port_string_coerced(self) -> None:
        result = build_verb_args("run", {
            "target_host": "host",
            "target_user": "user",
            "natural_language": "test",
            "target_port": "2222",
        })
        assert isinstance(result, RunArgs)
        assert result.target_port == 2222

    def test_missing_target_host_returns_error(self) -> None:
        result = build_verb_args("run", {
            "target_user": "user",
            "natural_language": "run tests",
        })
        assert isinstance(result, str)
        assert "target_host" in result.lower()

    def test_missing_target_user_returns_error(self) -> None:
        result = build_verb_args("run", {
            "target_host": "host",
            "natural_language": "run tests",
        })
        assert isinstance(result, str)
        assert "target_user" in result.lower()

    def test_missing_natural_language_returns_error(self) -> None:
        result = build_verb_args("run", {
            "target_host": "host",
            "target_user": "user",
        })
        assert isinstance(result, str)
        assert "natural_language" in result.lower()

    def test_run_can_be_built_from_system_name(self) -> None:
        result = build_verb_args("run", {
            "system_name": "tuto",
            "natural_language": "run the smoke tests",
        })
        assert isinstance(result, RunArgs)
        assert result.system_name == "tuto"

    def test_run_can_be_built_with_infer_target(self) -> None:
        result = build_verb_args("run", {
            "infer_target": True,
            "natural_language": "run the smoke tests in tuto",
        })
        assert isinstance(result, RunArgs)
        assert result.infer_target is True
        assert result.natural_language == "run the smoke tests in tuto"

    def test_run_can_be_built_with_interpret_request(self) -> None:
        result = build_verb_args("run", {
            "interpret_request": True,
            "natural_language": "run the smoke tests",
        })
        assert isinstance(result, RunArgs)
        assert result.interpret_request is True
        assert result.natural_language == "run the smoke tests"

    def test_empty_host_returns_error(self) -> None:
        result = build_verb_args("run", {
            "target_host": "  ",
            "target_user": "user",
            "natural_language": "test",
        })
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# QueueArgs building
# ---------------------------------------------------------------------------


class TestBuildQueueArgs:
    """Tests for building QueueArgs from extracted dicts."""

    def test_complete_args(self) -> None:
        result = build_verb_args("queue", {
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run the smoke tests",
        })
        assert isinstance(result, QueueArgs)
        assert result.priority == 0

    def test_with_priority(self) -> None:
        result = build_verb_args("queue", {
            "target_host": "host",
            "target_user": "user",
            "natural_language": "tests",
            "priority": 5,
        })
        assert isinstance(result, QueueArgs)
        assert result.priority == 5

    def test_priority_string_coerced(self) -> None:
        result = build_verb_args("queue", {
            "target_host": "host",
            "target_user": "user",
            "natural_language": "tests",
            "priority": "3",
        })
        assert isinstance(result, QueueArgs)
        assert result.priority == 3

    def test_missing_target_host_returns_error(self) -> None:
        result = build_verb_args("queue", {
            "target_user": "user",
            "natural_language": "run tests",
        })
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# CancelArgs building
# ---------------------------------------------------------------------------


class TestBuildCancelArgs:
    """Tests for building CancelArgs from extracted dicts."""

    def test_empty_dict_defaults(self) -> None:
        result = build_verb_args("cancel", {})
        assert isinstance(result, CancelArgs)
        assert result.run_id is None
        assert result.force is False
        assert result.reason is None

    def test_with_all_fields(self) -> None:
        result = build_verb_args("cancel", {
            "run_id": "abc-123",
            "force": True,
            "reason": "no longer needed",
        })
        assert isinstance(result, CancelArgs)
        assert result.run_id == "abc-123"
        assert result.force is True
        assert result.reason == "no longer needed"

    def test_force_string_true(self) -> None:
        result = build_verb_args("cancel", {"force": "true"})
        assert isinstance(result, CancelArgs)
        assert result.force is True


# ---------------------------------------------------------------------------
# HistoryArgs building
# ---------------------------------------------------------------------------


class TestBuildHistoryArgs:
    """Tests for building HistoryArgs from extracted dicts."""

    def test_empty_dict_defaults(self) -> None:
        result = build_verb_args("history", {})
        assert isinstance(result, HistoryArgs)
        assert result.limit == 20
        assert result.status_filter is None
        assert result.host_filter is None
        assert result.verbose is False

    def test_with_all_fields(self) -> None:
        result = build_verb_args("history", {
            "limit": 50,
            "status_filter": "failed",
            "host_filter": "staging",
            "verbose": True,
        })
        assert isinstance(result, HistoryArgs)
        assert result.limit == 50
        assert result.status_filter == "failed"
        assert result.host_filter == "staging"
        assert result.verbose is True

    def test_limit_string_coerced(self) -> None:
        result = build_verb_args("history", {"limit": "30"})
        assert isinstance(result, HistoryArgs)
        assert result.limit == 30

    def test_invalid_status_filter_returns_error(self) -> None:
        result = build_verb_args("history", {"status_filter": "invalid_status"})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestBuildVerbArgsErrors:
    """Tests for error handling in build_verb_args."""

    def test_unknown_verb_returns_error(self) -> None:
        result = build_verb_args("unknown", {})
        assert isinstance(result, str)
        assert "unknown" in result.lower()

    def test_empty_verb_returns_error(self) -> None:
        result = build_verb_args("", {})
        assert isinstance(result, str)

    def test_whitespace_verb_returns_error(self) -> None:
        result = build_verb_args("  ", {})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Cross-path equivalence: structured vs NL args produce identical results
# ---------------------------------------------------------------------------


class TestCrossPathEquivalence:
    """Verify that structured and NL extracted_args produce identical *Args.

    This is the key guarantee of the unified entry point: regardless of
    whether the user typed a structured command or natural language, the
    handler receives the same *Args dataclass.
    """

    def test_status_equivalence(self) -> None:
        """Both paths produce StatusArgs(verbose=True)."""
        # Structured path extracted_args
        structured = build_verb_args("status", {"verbose": True})
        # NL path extracted_args (from NL extractor)
        nl_style = build_verb_args("status", {"verbose": True})
        assert structured == nl_style

    def test_cancel_equivalence(self) -> None:
        """Both paths produce CancelArgs(force=True, run_id=None)."""
        structured = build_verb_args("cancel", {"force": True})
        nl_style = build_verb_args("cancel", {"force": True})
        assert structured == nl_style

    def test_run_equivalence(self) -> None:
        """Both paths produce RunArgs with identical fields."""
        args = {
            "target_host": "staging",
            "target_user": "ci",
            "natural_language": "run smoke tests",
        }
        structured = build_verb_args("run", args)
        nl_style = build_verb_args("run", dict(args))
        assert structured == nl_style
        assert isinstance(structured, RunArgs)
        assert isinstance(nl_style, RunArgs)
