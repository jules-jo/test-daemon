"""Iterative argument collection via ask_user_question prompting.

Bridges the gap between ``argument_gap.detect_argument_gaps`` (which
detects missing required arguments) and the ``ask_user_question`` tool
callback (which sends a question to the user via IPC). This module
provides the iterative loop that walks over each missing argument,
asks the user for a value, and accumulates responses until all
required arguments are collected (or the user cancels / an error occurs).

Design:
    - Pure async function: no internal state, no mutation of inputs.
    - Immutable result: ``ArgumentCollectionResult`` is a frozen dataclass.
    - Early termination on user cancellation (returns partial results).
    - Error isolation: callback exceptions are caught and returned as
      ERROR status results with the error message preserved.
    - Defensive copy: the ``collected_args`` dict in the result is a
      new dict on every access to prevent external mutation.
    - Questions include position indicators (1/3, 2/3, 3/3) when
      multiple arguments are missing.

Usage::

    from jules_daemon.agent.argument_collector import (
        ArgumentCollectionResult,
        ArgumentCollectionStatus,
        collect_missing_arguments,
    )
    from jules_daemon.agent.argument_gap import detect_argument_gaps

    gap = detect_argument_gaps(
        required_args=("iterations", "host"),
        provided_args={"iterations": "100"},
    )

    if not gap.is_complete:
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=my_ask_callback,
            test_name="agent_test",
        )
        if result.is_complete:
            merged = {**existing_args, **result.collected_args}
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

from jules_daemon.agent.argument_gap import ArgumentGapResult

__all__ = [
    "ArgumentCollectionResult",
    "ArgumentCollectionStatus",
    "collect_missing_arguments",
]

logger = logging.getLogger(__name__)

# Type alias matching AskUserQuestionTool's callback signature.
# Takes (question, context) -> answer_or_none.
AskCallback = Callable[
    [str, str],
    Awaitable[str | None],
]


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class ArgumentCollectionStatus(Enum):
    """Outcome of an argument collection attempt.

    COMPLETE:  All missing arguments were collected successfully.
    CANCELLED: The user cancelled a question (returned None from
               the ask callback). Partial results are available.
    ERROR:     The ask callback raised an exception. Partial results
               are available up to the point of failure.
    """

    COMPLETE = "complete"
    CANCELLED = "cancelled"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArgumentCollectionResult:
    """Immutable result of iterative argument collection.

    Attributes:
        status: Outcome classification (COMPLETE, CANCELLED, ERROR).
        collected_args: Dict mapping argument names to user-provided
            values. May be partial if the user cancelled or an error
            occurred. Returns a defensive copy on each access.
        questions_asked: Total number of questions sent to the user
            (including the one that was cancelled, if applicable).
        cancelled_arg: Name of the argument whose question was
            cancelled by the user. None when status is not CANCELLED.
        error_message: Human-readable error description. None unless
            status is ERROR.
        missing_args: The original tuple of missing argument names
            from the gap result (for reference).
    """

    status: ArgumentCollectionStatus
    _collected_args: dict[str, str]
    questions_asked: int
    cancelled_arg: str | None = None
    error_message: str | None = None
    missing_args: tuple[str, ...] = ()

    @property
    def collected_args(self) -> dict[str, str]:
        """Return a defensive copy of the collected arguments."""
        return dict(self._collected_args)

    @property
    def is_complete(self) -> bool:
        """True if all missing arguments were collected."""
        return self.status is ArgumentCollectionStatus.COMPLETE


# ---------------------------------------------------------------------------
# Question formatting
# ---------------------------------------------------------------------------


def _format_question(
    *,
    arg_name: str,
    position: int,
    total: int,
) -> str:
    """Format the question text for a single missing argument.

    Includes a position indicator when multiple args are missing,
    e.g., "(1/3) What value should be used for 'iterations'?"

    Args:
        arg_name: The name of the missing argument.
        position: 1-based position in the missing args list.
        total: Total number of missing arguments.

    Returns:
        Formatted question string.
    """
    if total > 1:
        return (
            f"({position}/{total}) What value should be used "
            f"for '{arg_name}'?"
        )
    return f"What value should be used for '{arg_name}'?"


def _format_context(
    *,
    arg_name: str,
    test_name: str | None,
) -> str:
    """Format the context string explaining why the question is asked.

    Args:
        arg_name: The name of the missing argument.
        test_name: Optional test name for additional context.

    Returns:
        Context string.
    """
    if test_name:
        return (
            f"The argument '{arg_name}' is required by the test "
            f"specification for '{test_name}'. Please provide a value."
        )
    return (
        f"The argument '{arg_name}' is required by the test "
        f"specification. Please provide a value."
    )


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------


async def collect_missing_arguments(
    *,
    gap_result: ArgumentGapResult,
    ask_callback: AskCallback,
    test_name: str | None = None,
) -> ArgumentCollectionResult:
    """Iteratively ask the user for each missing required argument.

    Walks through ``gap_result.missing_args`` in order, asking the
    user for each value via ``ask_callback``. Accumulates responses
    into a dict and returns an immutable result.

    Termination conditions:
        - All missing args collected (COMPLETE).
        - User cancels a question by returning None (CANCELLED).
        - ask_callback raises an exception (ERROR).

    The function never guesses or auto-defaults missing arguments --
    it always asks the user, per the project constraint.

    Args:
        gap_result: The result of ``detect_argument_gaps`` indicating
            which arguments are missing.
        ask_callback: Async callable matching the AskUserQuestionTool
            callback signature: ``(question, context) -> str | None``.
        test_name: Optional test name for richer context in questions.

    Returns:
        Immutable ``ArgumentCollectionResult`` with the collected
        arguments and outcome.
    """
    missing = gap_result.missing_args

    # Fast path: no missing arguments
    if not missing:
        return ArgumentCollectionResult(
            status=ArgumentCollectionStatus.COMPLETE,
            _collected_args={},
            questions_asked=0,
            missing_args=missing,
        )

    collected: dict[str, str] = {}
    questions_asked = 0
    total = len(missing)

    for position_zero_based, arg_name in enumerate(missing):
        position = position_zero_based + 1

        question = _format_question(
            arg_name=arg_name,
            position=position,
            total=total,
        )
        context = _format_context(
            arg_name=arg_name,
            test_name=test_name,
        )

        logger.debug(
            "Asking user for missing argument '%s' (%d/%d)",
            arg_name,
            position,
            total,
        )

        try:
            questions_asked += 1
            answer = await ask_callback(question, context)
        except Exception as exc:
            logger.warning(
                "ask_callback error for argument '%s': %s",
                arg_name,
                exc,
            )
            return ArgumentCollectionResult(
                status=ArgumentCollectionStatus.ERROR,
                _collected_args=dict(collected),
                questions_asked=questions_asked,
                error_message=f"Failed to ask for '{arg_name}': {exc}",
                missing_args=missing,
            )

        # User cancelled
        if answer is None:
            logger.info(
                "User cancelled argument collection at '%s' (%d/%d)",
                arg_name,
                position,
                total,
            )
            return ArgumentCollectionResult(
                status=ArgumentCollectionStatus.CANCELLED,
                _collected_args=dict(collected),
                questions_asked=questions_asked,
                cancelled_arg=arg_name,
                missing_args=missing,
            )

        # Accumulate the answer (accept any string, including empty)
        collected[arg_name] = answer

    logger.info(
        "All %d missing arguments collected successfully",
        total,
    )
    return ArgumentCollectionResult(
        status=ArgumentCollectionStatus.COMPLETE,
        _collected_args=dict(collected),
        questions_asked=questions_asked,
        missing_args=missing,
    )
