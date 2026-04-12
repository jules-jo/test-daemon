"""Tests for argument gap detection logic.

Covers:
    - Detecting missing required arguments when user provides partial args
    - Full coverage: all args provided, no args provided, partial overlap
    - Immutable result model (frozen dataclass)
    - Empty required_args means no gaps ever
    - Case-sensitive argument name matching
    - Integration with TestKnowledge required_args field
    - Edge cases: whitespace names, empty names, duplicate detection
"""

from __future__ import annotations

import pytest

from jules_daemon.agent.argument_gap import (
    ArgumentGapResult,
    detect_argument_gaps,
)


# ---------------------------------------------------------------------------
# Basic detection logic
# ---------------------------------------------------------------------------


class TestDetectArgumentGaps:
    """Verify core gap detection between required args and provided args."""

    def test_no_required_args_means_no_gaps(self) -> None:
        """When the spec has no required args, result should have zero gaps."""
        result = detect_argument_gaps(
            required_args=(),
            provided_args={},
        )
        assert result.missing_args == ()
        assert result.is_complete is True
        assert result.gap_count == 0

    def test_all_required_args_provided(self) -> None:
        """When all required args are provided, no gaps detected."""
        result = detect_argument_gaps(
            required_args=("iterations", "host"),
            provided_args={"iterations": "100", "host": "staging.example.com"},
        )
        assert result.missing_args == ()
        assert result.is_complete is True
        assert result.gap_count == 0

    def test_some_required_args_missing(self) -> None:
        """When only some required args are provided, detect the missing ones."""
        result = detect_argument_gaps(
            required_args=("iterations", "host", "concurrency"),
            provided_args={"iterations": "100"},
        )
        assert set(result.missing_args) == {"host", "concurrency"}
        assert result.is_complete is False
        assert result.gap_count == 2

    def test_all_required_args_missing(self) -> None:
        """When no required args are provided, all are reported missing."""
        result = detect_argument_gaps(
            required_args=("iterations", "host"),
            provided_args={},
        )
        assert set(result.missing_args) == {"iterations", "host"}
        assert result.is_complete is False
        assert result.gap_count == 2

    def test_extra_provided_args_ignored(self) -> None:
        """Extra args beyond required should not affect gap detection."""
        result = detect_argument_gaps(
            required_args=("iterations",),
            provided_args={"iterations": "100", "verbose": "true", "env": "staging"},
        )
        assert result.missing_args == ()
        assert result.is_complete is True

    def test_single_missing_arg(self) -> None:
        """Detect a single missing required argument."""
        result = detect_argument_gaps(
            required_args=("iterations", "host"),
            provided_args={"iterations": "100"},
        )
        assert result.missing_args == ("host",)
        assert result.is_complete is False
        assert result.gap_count == 1


# ---------------------------------------------------------------------------
# Argument name matching semantics
# ---------------------------------------------------------------------------


class TestArgumentNameMatching:
    """Verify argument name matching rules."""

    def test_case_sensitive_matching(self) -> None:
        """Argument names must match exactly (case-sensitive)."""
        result = detect_argument_gaps(
            required_args=("Host",),
            provided_args={"host": "example.com"},
        )
        assert result.missing_args == ("Host",)
        assert result.is_complete is False

    def test_whitespace_in_provided_key_not_stripped(self) -> None:
        """Provided arg keys with whitespace do not match stripped required args."""
        result = detect_argument_gaps(
            required_args=("host",),
            provided_args={" host ": "example.com"},
        )
        assert result.missing_args == ("host",)
        assert result.is_complete is False


# ---------------------------------------------------------------------------
# Ordering guarantees
# ---------------------------------------------------------------------------


class TestOrdering:
    """Verify that missing args preserve the order from required_args."""

    def test_missing_args_preserve_spec_order(self) -> None:
        """Missing args should appear in the same order as required_args."""
        result = detect_argument_gaps(
            required_args=("alpha", "beta", "gamma", "delta"),
            provided_args={"beta": "b"},
        )
        assert result.missing_args == ("alpha", "gamma", "delta")

    def test_provided_args_listed(self) -> None:
        """provided_arg_names should list what was actually provided."""
        result = detect_argument_gaps(
            required_args=("iterations", "host"),
            provided_args={"iterations": "100", "extra": "val"},
        )
        assert "iterations" in result.provided_arg_names
        assert "extra" in result.provided_arg_names


# ---------------------------------------------------------------------------
# ArgumentGapResult model
# ---------------------------------------------------------------------------


class TestArgumentGapResult:
    """Verify the immutable result model properties."""

    def test_frozen_dataclass(self) -> None:
        """ArgumentGapResult must be frozen (immutable)."""
        result = detect_argument_gaps(
            required_args=("host",),
            provided_args={},
        )
        with pytest.raises(AttributeError):
            result.missing_args = ("something",)  # type: ignore[misc]

    def test_gap_count_matches_missing_args_length(self) -> None:
        """gap_count must equal len(missing_args)."""
        result = detect_argument_gaps(
            required_args=("a", "b", "c"),
            provided_args={"b": "val"},
        )
        assert result.gap_count == len(result.missing_args)

    def test_required_args_stored(self) -> None:
        """The result should record the original required_args."""
        result = detect_argument_gaps(
            required_args=("iterations", "host"),
            provided_args={"iterations": "100"},
        )
        assert result.required_args == ("iterations", "host")

    def test_provided_arg_names_stored(self) -> None:
        """The result should record the provided argument names."""
        result = detect_argument_gaps(
            required_args=("host",),
            provided_args={"host": "val", "port": "22"},
        )
        assert set(result.provided_arg_names) == {"host", "port"}

    def test_to_prompt_message_complete(self) -> None:
        """to_prompt_message for complete args returns empty string."""
        result = detect_argument_gaps(
            required_args=("host",),
            provided_args={"host": "val"},
        )
        assert result.to_prompt_message() == ""

    def test_to_prompt_message_single_gap(self) -> None:
        """to_prompt_message for one missing arg returns singular message."""
        result = detect_argument_gaps(
            required_args=("iterations", "host"),
            provided_args={"iterations": "100"},
        )
        msg = result.to_prompt_message()
        assert "host" in msg
        assert "missing" in msg.lower()
        # Singular form: should not say "arguments" (plural)
        assert "Missing required argument:" in msg

    def test_to_prompt_message_multiple_gaps(self) -> None:
        """to_prompt_message for multiple missing args returns plural message."""
        result = detect_argument_gaps(
            required_args=("iterations", "host"),
            provided_args={},
        )
        msg = result.to_prompt_message()
        assert "iterations" in msg
        assert "host" in msg
        assert "missing" in msg.lower()
        assert "2 required arguments" in msg.lower()


# ---------------------------------------------------------------------------
# Integration with TestKnowledge required_args
# ---------------------------------------------------------------------------


class TestTestKnowledgeIntegration:
    """Verify gap detection works with real TestKnowledge data shapes."""

    def test_from_test_knowledge_required_args(self) -> None:
        """Detect gaps using required_args from a TestKnowledge instance."""
        from jules_daemon.wiki.test_knowledge import TestKnowledge

        knowledge = TestKnowledge(
            test_slug="agent-test-py",
            command_pattern="python3 ~/agent_test.py",
            required_args=("iterations", "host"),
        )

        result = detect_argument_gaps(
            required_args=knowledge.required_args,
            provided_args={"iterations": "100"},
        )
        assert result.missing_args == ("host",)
        assert result.is_complete is False

    def test_from_test_knowledge_no_required_args(self) -> None:
        """When a test has no required_args, gap detection finds nothing."""
        from jules_daemon.wiki.test_knowledge import TestKnowledge

        knowledge = TestKnowledge(
            test_slug="smoke-test-sh",
            command_pattern="./smoke_test.sh",
            required_args=(),
        )

        result = detect_argument_gaps(
            required_args=knowledge.required_args,
            provided_args={},
        )
        assert result.is_complete is True
        assert result.gap_count == 0

    def test_detect_from_spec_convenience(self) -> None:
        """detect_argument_gaps_from_spec uses TestKnowledge directly."""
        from jules_daemon.agent.argument_gap import (
            detect_argument_gaps_from_spec,
        )
        from jules_daemon.wiki.test_knowledge import TestKnowledge

        knowledge = TestKnowledge(
            test_slug="agent-test-py",
            command_pattern="python3 ~/agent_test.py",
            required_args=("iterations", "host", "concurrency"),
        )

        result = detect_argument_gaps_from_spec(
            test_knowledge=knowledge,
            provided_args={"host": "staging.example.com"},
        )
        assert set(result.missing_args) == {"iterations", "concurrency"}
        assert result.required_args == ("iterations", "host", "concurrency")

    def test_detect_from_spec_none_knowledge(self) -> None:
        """When test_knowledge is None, treat as no required args (complete)."""
        from jules_daemon.agent.argument_gap import (
            detect_argument_gaps_from_spec,
        )

        result = detect_argument_gaps_from_spec(
            test_knowledge=None,
            provided_args={"anything": "val"},
        )
        assert result.is_complete is True
        assert result.gap_count == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions for argument gap detection."""

    def test_none_value_counts_as_provided(self) -> None:
        """A provided arg with None value still counts as provided."""
        result = detect_argument_gaps(
            required_args=("host",),
            provided_args={"host": None},
        )
        assert result.is_complete is True
        assert result.gap_count == 0

    def test_empty_string_value_counts_as_provided(self) -> None:
        """A provided arg with empty string still counts as provided."""
        result = detect_argument_gaps(
            required_args=("host",),
            provided_args={"host": ""},
        )
        assert result.is_complete is True
        assert result.gap_count == 0

    def test_list_required_args_coerced(self) -> None:
        """required_args as a list should work (coerced internally)."""
        result = detect_argument_gaps(
            required_args=["iterations", "host"],  # type: ignore[arg-type]
            provided_args={"iterations": "100"},
        )
        assert result.missing_args == ("host",)

    def test_large_number_of_args(self) -> None:
        """Gap detection handles many arguments efficiently."""
        required = tuple(f"arg_{i}" for i in range(50))
        provided = {f"arg_{i}": str(i) for i in range(25)}

        result = detect_argument_gaps(
            required_args=required,
            provided_args=provided,
        )
        assert result.gap_count == 25
        assert result.is_complete is False

    def test_provided_args_defensive_copy(self) -> None:
        """Modifying provided_args after call should not affect the result."""
        provided = {"host": "example.com"}
        result = detect_argument_gaps(
            required_args=("host", "port"),
            provided_args=provided,
        )
        provided["port"] = "22"  # mutate after call
        assert result.missing_args == ("port",)  # result unchanged
