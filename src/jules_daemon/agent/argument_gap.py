"""Argument gap detection for the agent loop.

Compares user-provided partial arguments against the wiki test catalog
spec (``TestKnowledge.required_args``) to identify which required
arguments are missing. The agent loop uses this to decide when to invoke
``ask_user_question`` before proposing an SSH command.

Design:
    - Pure function: no I/O, no side effects, no LLM calls.
    - Deterministic: same inputs always produce the same output.
    - Immutable result: ``ArgumentGapResult`` is a frozen dataclass.
    - Order-preserving: ``missing_args`` preserves the order from the spec's
      ``required_args``, so the user sees them in the same order the spec
      author intended.
    - Case-sensitive matching: argument names are compared exactly as they
      appear in the spec and provided dict. No normalization is applied.
    - Key-presence only: a provided arg with any value (including ``None``
      or empty string) counts as "provided". The semantic validity of
      argument *values* is outside this module's scope.

Usage::

    from jules_daemon.agent.argument_gap import (
        ArgumentGapResult,
        detect_argument_gaps,
        detect_argument_gaps_from_spec,
    )

    result = detect_argument_gaps(
        required_args=("iterations", "host"),
        provided_args={"iterations": "100"},
    )
    if not result.is_complete:
        print(f"Missing: {result.missing_args}")  # ("host",)
        print(result.to_prompt_message())

    # Or from a TestKnowledge directly:
    result = detect_argument_gaps_from_spec(
        test_knowledge=knowledge,
        provided_args=user_args,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from jules_daemon.wiki.test_knowledge import TestKnowledge

__all__ = [
    "ArgumentGapResult",
    "detect_argument_gaps",
    "detect_argument_gaps_from_spec",
]


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArgumentGapResult:
    """Immutable result of comparing provided args against required args.

    Attributes:
        required_args: The full tuple of required argument names from the
            test spec, in their original order.
        provided_arg_names: Tuple of argument names the user actually
            provided (from the ``provided_args`` dict keys).
        missing_args: Tuple of required argument names that were not
            found in the provided args. Preserves the original ordering
            from ``required_args``.
    """

    required_args: tuple[str, ...]
    provided_arg_names: tuple[str, ...]
    missing_args: tuple[str, ...]

    # -- Computed properties ------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True if all required arguments have been provided."""
        return len(self.missing_args) == 0

    @property
    def gap_count(self) -> int:
        """Number of required arguments that are missing."""
        return len(self.missing_args)

    # -- Formatting ---------------------------------------------------------

    def to_prompt_message(self) -> str:
        """Format the gap as a message suitable for the LLM or user prompt.

        Returns an empty string when no arguments are missing. Otherwise
        returns a concise message listing the missing argument names.

        Returns:
            Human-readable gap description, or empty string if complete.
        """
        if self.is_complete:
            return ""

        if self.gap_count == 1:
            return (
                f"Missing required argument: {self.missing_args[0]}. "
                f"Please provide a value before proceeding."
            )

        names = ", ".join(self.missing_args)
        return (
            f"Missing {self.gap_count} required arguments: {names}. "
            f"Please provide values for each before proceeding."
        )


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------


def detect_argument_gaps(
    *,
    required_args: Sequence[str],
    provided_args: dict[str, Any],
) -> ArgumentGapResult:
    """Compare provided arguments against required arguments to find gaps.

    This is the core detection function. It checks which entries from
    ``required_args`` are absent from ``provided_args`` keys. Matching is
    case-sensitive and checks key presence only (not value validity).

    Args:
        required_args: Sequence of argument names that the test spec
            declares as required. Typically from
            ``TestKnowledge.required_args``.
        provided_args: Dict of argument names to values that the user
            has provided so far. Only keys are inspected; values are
            not validated.

    Returns:
        An immutable ``ArgumentGapResult`` with the missing argument names.
    """
    required_tuple = tuple(required_args)
    provided_keys = frozenset(provided_args.keys())
    provided_names = tuple(provided_args.keys())

    missing = tuple(
        arg for arg in required_tuple if arg not in provided_keys
    )

    return ArgumentGapResult(
        required_args=required_tuple,
        provided_arg_names=provided_names,
        missing_args=missing,
    )


def detect_argument_gaps_from_spec(
    *,
    test_knowledge: TestKnowledge | None,
    provided_args: dict[str, Any],
) -> ArgumentGapResult:
    """Detect argument gaps using a TestKnowledge instance directly.

    Convenience wrapper around ``detect_argument_gaps`` that extracts
    ``required_args`` from the wiki test knowledge record.

    When ``test_knowledge`` is ``None`` (test not found in catalog),
    the result indicates no required args and therefore no gaps -- the
    caller should proceed without blocking on missing arguments.

    Args:
        test_knowledge: Wiki test knowledge record, or ``None`` if the
            test was not found in the catalog.
        provided_args: Dict of argument names to values that the user
            has provided so far.

    Returns:
        An immutable ``ArgumentGapResult``.
    """
    if test_knowledge is None:
        return detect_argument_gaps(
            required_args=(),
            provided_args=provided_args,
        )

    return detect_argument_gaps(
        required_args=test_knowledge.required_args,
        provided_args=provided_args,
    )
