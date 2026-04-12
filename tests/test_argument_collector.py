"""Tests for iterative argument collection flow.

Covers Sub-AC 2 of AC 8: iterative ask_user_question prompting flow
that loops over each missing required arg, asks the user, and
accumulates responses until all required args are collected.

Covers:
    - Collecting all missing arguments one at a time
    - User cancellation terminates early with partial results
    - Empty missing args returns immediately with no questions
    - Single missing arg: one question, one answer
    - Multiple missing args: iterated questions, accumulated answers
    - Error from ask callback propagated as error result
    - Frozen (immutable) result dataclass
    - Integration with ArgumentGapResult from detect_argument_gaps
    - Question formatting includes arg name and context
    - Collected args dict is a defensive copy (not mutated externally)
"""

from __future__ import annotations

import pytest

from jules_daemon.agent.argument_collector import (
    ArgumentCollectionResult,
    ArgumentCollectionStatus,
    collect_missing_arguments,
)
from jules_daemon.agent.argument_gap import (
    ArgumentGapResult,
    detect_argument_gaps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gap_result(
    *,
    required: tuple[str, ...],
    provided: dict[str, object] | None = None,
) -> ArgumentGapResult:
    """Helper to create an ArgumentGapResult from required + provided."""
    return detect_argument_gaps(
        required_args=required,
        provided_args=provided or {},
    )


async def _mock_ask_success(question: str, context: str) -> str | None:
    """Always returns the arg name extracted from the question as the value."""
    # Convention: the question will contain the arg name. For testing, just
    # return a canned value per arg name detected in the question.
    if "iterations" in question.lower():
        return "100"
    if "host" in question.lower():
        return "staging.example.com"
    if "concurrency" in question.lower():
        return "4"
    if "port" in question.lower():
        return "22"
    return "user_value"


async def _mock_ask_cancel(question: str, context: str) -> str | None:
    """Simulates user cancelling (returns None)."""
    return None


async def _mock_ask_cancel_on_second(question: str, context: str) -> str | None:
    """Returns a value for the first question, cancels on the second."""
    if not hasattr(_mock_ask_cancel_on_second, "_call_count"):
        _mock_ask_cancel_on_second._call_count = 0  # type: ignore[attr-defined]
    _mock_ask_cancel_on_second._call_count += 1  # type: ignore[attr-defined]
    if _mock_ask_cancel_on_second._call_count <= 1:  # type: ignore[attr-defined]
        return "first_value"
    return None


async def _mock_ask_error(question: str, context: str) -> str | None:
    """Simulates an error in the ask callback."""
    raise ConnectionError("IPC connection lost")


# ---------------------------------------------------------------------------
# No missing args: immediate return
# ---------------------------------------------------------------------------


class TestNoMissingArgs:
    """When there are no missing args, collect should return immediately."""

    @pytest.mark.asyncio
    async def test_empty_missing_args_returns_complete(self) -> None:
        """No missing args -> COMPLETE status with empty collected dict."""
        gap = _make_gap_result(
            required=("host",),
            provided={"host": "example.com"},
        )
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_success,
        )
        assert result.status is ArgumentCollectionStatus.COMPLETE
        assert result.collected_args == {}
        assert result.questions_asked == 0

    @pytest.mark.asyncio
    async def test_no_required_args_returns_complete(self) -> None:
        """Empty required args -> COMPLETE status, no questions asked."""
        gap = _make_gap_result(required=())
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_success,
        )
        assert result.status is ArgumentCollectionStatus.COMPLETE
        assert result.collected_args == {}
        assert result.questions_asked == 0


# ---------------------------------------------------------------------------
# Single missing arg
# ---------------------------------------------------------------------------


class TestSingleMissingArg:
    """Collect a single missing required argument."""

    @pytest.mark.asyncio
    async def test_single_missing_arg_collected(self) -> None:
        """One missing arg -> one question -> collected in result."""
        gap = _make_gap_result(
            required=("iterations", "host"),
            provided={"iterations": "100"},
        )
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_success,
        )
        assert result.status is ArgumentCollectionStatus.COMPLETE
        assert "host" in result.collected_args
        assert result.collected_args["host"] == "staging.example.com"
        assert result.questions_asked == 1

    @pytest.mark.asyncio
    async def test_single_missing_arg_cancelled(self) -> None:
        """User cancels on the single missing arg -> CANCELLED status."""
        gap = _make_gap_result(
            required=("host",),
            provided={},
        )
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_cancel,
        )
        assert result.status is ArgumentCollectionStatus.CANCELLED
        assert result.collected_args == {}
        assert result.questions_asked == 1
        assert result.cancelled_arg == "host"


# ---------------------------------------------------------------------------
# Multiple missing args
# ---------------------------------------------------------------------------


class TestMultipleMissingArgs:
    """Collect multiple missing required arguments iteratively."""

    @pytest.mark.asyncio
    async def test_all_missing_args_collected(self) -> None:
        """All missing args collected one by one."""
        gap = _make_gap_result(
            required=("iterations", "host", "concurrency"),
            provided={},
        )
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_success,
        )
        assert result.status is ArgumentCollectionStatus.COMPLETE
        assert result.collected_args == {
            "iterations": "100",
            "host": "staging.example.com",
            "concurrency": "4",
        }
        assert result.questions_asked == 3

    @pytest.mark.asyncio
    async def test_cancel_on_second_arg_preserves_first(self) -> None:
        """Cancel on second arg -> CANCELLED with first arg collected."""
        # Reset the call counter
        if hasattr(_mock_ask_cancel_on_second, "_call_count"):
            _mock_ask_cancel_on_second._call_count = 0  # type: ignore[attr-defined]

        gap = _make_gap_result(
            required=("iterations", "host", "concurrency"),
            provided={},
        )
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_cancel_on_second,
        )
        assert result.status is ArgumentCollectionStatus.CANCELLED
        # First arg was collected before cancellation
        assert "iterations" in result.collected_args
        assert result.collected_args["iterations"] == "first_value"
        # Cancelled on second arg
        assert result.cancelled_arg == "host"
        assert result.questions_asked == 2

    @pytest.mark.asyncio
    async def test_order_preserved_in_collection(self) -> None:
        """Missing args are asked in the order from the spec."""
        asked_order: list[str] = []

        async def _tracking_ask(question: str, context: str) -> str | None:
            asked_order.append(question)
            return "val"

        gap = _make_gap_result(
            required=("alpha", "beta", "gamma"),
            provided={},
        )
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_tracking_ask,
        )
        assert result.status is ArgumentCollectionStatus.COMPLETE
        # Verify the order: alpha, beta, gamma
        assert "alpha" in asked_order[0]
        assert "beta" in asked_order[1]
        assert "gamma" in asked_order[2]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify error propagation from the ask callback."""

    @pytest.mark.asyncio
    async def test_callback_error_returns_error_status(self) -> None:
        """When ask callback raises, result has ERROR status."""
        gap = _make_gap_result(
            required=("host",),
            provided={},
        )
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_error,
        )
        assert result.status is ArgumentCollectionStatus.ERROR
        assert result.error_message is not None
        assert "IPC connection lost" in result.error_message
        assert result.questions_asked <= 1

    @pytest.mark.asyncio
    async def test_error_on_second_preserves_first(self) -> None:
        """Error on second arg -> ERROR with first arg collected."""
        call_count = 0

        async def _ask_error_on_second(q: str, ctx: str) -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "first_answer"
            raise ConnectionError("lost connection")

        gap = _make_gap_result(
            required=("iterations", "host"),
            provided={},
        )
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_ask_error_on_second,
        )
        assert result.status is ArgumentCollectionStatus.ERROR
        assert "iterations" in result.collected_args
        assert result.error_message is not None


# ---------------------------------------------------------------------------
# Result model properties
# ---------------------------------------------------------------------------


class TestResultModel:
    """Verify ArgumentCollectionResult immutability and properties."""

    @pytest.mark.asyncio
    async def test_result_is_frozen(self) -> None:
        """ArgumentCollectionResult should be a frozen dataclass."""
        gap = _make_gap_result(required=("host",), provided={})
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_success,
        )
        with pytest.raises(AttributeError):
            result.status = ArgumentCollectionStatus.CANCELLED  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_collected_args_defensive_copy(self) -> None:
        """Modifying returned collected_args should not affect internal state."""
        gap = _make_gap_result(
            required=("host",),
            provided={},
        )
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_success,
        )
        # Access the collected_args -- it should be a dict
        args = result.collected_args
        original_host = args.get("host")
        # Mutate the returned dict
        args["host"] = "mutated"
        # Original result should be unaffected (defensive copy)
        assert result.collected_args.get("host") == original_host

    @pytest.mark.asyncio
    async def test_is_complete_property(self) -> None:
        """is_complete should be True when status is COMPLETE."""
        gap = _make_gap_result(required=("host",), provided={})
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_success,
        )
        assert result.is_complete is True

    @pytest.mark.asyncio
    async def test_is_complete_false_on_cancel(self) -> None:
        """is_complete should be False when status is CANCELLED."""
        gap = _make_gap_result(required=("host",), provided={})
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_cancel,
        )
        assert result.is_complete is False


# ---------------------------------------------------------------------------
# Question formatting
# ---------------------------------------------------------------------------


class TestQuestionFormatting:
    """Verify that questions include the arg name and context."""

    @pytest.mark.asyncio
    async def test_question_contains_arg_name(self) -> None:
        """The question sent to the user should contain the argument name."""
        captured_questions: list[str] = []

        async def _capturing_ask(question: str, context: str) -> str | None:
            captured_questions.append(question)
            return "answer"

        gap = _make_gap_result(
            required=("iterations",),
            provided={},
        )
        await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_capturing_ask,
        )
        assert len(captured_questions) == 1
        assert "iterations" in captured_questions[0]

    @pytest.mark.asyncio
    async def test_context_contains_test_info(self) -> None:
        """The context should mention that the arg is required."""
        captured_contexts: list[str] = []

        async def _capturing_ask(question: str, context: str) -> str | None:
            captured_contexts.append(context)
            return "answer"

        gap = _make_gap_result(
            required=("host",),
            provided={},
        )
        await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_capturing_ask,
        )
        assert len(captured_contexts) == 1
        assert "required" in captured_contexts[0].lower()

    @pytest.mark.asyncio
    async def test_custom_test_name_in_context(self) -> None:
        """When test_name is provided, context should include it."""
        captured_contexts: list[str] = []

        async def _capturing_ask(question: str, context: str) -> str | None:
            captured_contexts.append(context)
            return "answer"

        gap = _make_gap_result(
            required=("host",),
            provided={},
        )
        await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_capturing_ask,
            test_name="agent_test",
        )
        assert len(captured_contexts) == 1
        assert "agent_test" in captured_contexts[0]

    @pytest.mark.asyncio
    async def test_position_indicator_in_question(self) -> None:
        """When multiple args are missing, question shows position (1/3, 2/3)."""
        captured_questions: list[str] = []

        async def _capturing_ask(question: str, context: str) -> str | None:
            captured_questions.append(question)
            return "answer"

        gap = _make_gap_result(
            required=("alpha", "beta", "gamma"),
            provided={},
        )
        await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_capturing_ask,
        )
        assert len(captured_questions) == 3
        assert "1" in captured_questions[0] and "3" in captured_questions[0]
        assert "2" in captured_questions[1] and "3" in captured_questions[1]
        assert "3" in captured_questions[2] and "3" in captured_questions[2]


# ---------------------------------------------------------------------------
# Integration with detect_argument_gaps
# ---------------------------------------------------------------------------


class TestGapIntegration:
    """End-to-end integration: detect gaps then collect."""

    @pytest.mark.asyncio
    async def test_detect_then_collect(self) -> None:
        """Full flow: detect gaps from a TestKnowledge, collect missing args."""
        from jules_daemon.wiki.test_knowledge import TestKnowledge

        knowledge = TestKnowledge(
            test_slug="agent-test-py",
            command_pattern="python3 ~/agent_test.py",
            required_args=("iterations", "host"),
        )

        gap = detect_argument_gaps(
            required_args=knowledge.required_args,
            provided_args={"iterations": "100"},
        )
        assert not gap.is_complete

        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_success,
            test_name="agent_test",
        )
        assert result.is_complete
        assert result.collected_args["host"] == "staging.example.com"

    @pytest.mark.asyncio
    async def test_detect_complete_no_collection_needed(self) -> None:
        """When all args provided, collection returns immediately."""
        from jules_daemon.wiki.test_knowledge import TestKnowledge

        knowledge = TestKnowledge(
            test_slug="smoke-sh",
            command_pattern="./smoke.sh",
            required_args=("host",),
        )

        gap = detect_argument_gaps(
            required_args=knowledge.required_args,
            provided_args={"host": "staging"},
        )
        assert gap.is_complete

        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_mock_ask_success,
        )
        assert result.is_complete
        assert result.collected_args == {}
        assert result.questions_asked == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for argument collection."""

    @pytest.mark.asyncio
    async def test_empty_string_answer_still_collected(self) -> None:
        """An empty string answer is still accepted (not treated as cancel)."""
        async def _ask_empty(question: str, context: str) -> str | None:
            return ""

        gap = _make_gap_result(required=("host",), provided={})
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_ask_empty,
        )
        assert result.is_complete
        assert result.collected_args["host"] == ""

    @pytest.mark.asyncio
    async def test_whitespace_answer_collected(self) -> None:
        """A whitespace-only answer is collected as-is (no stripping)."""
        async def _ask_whitespace(question: str, context: str) -> str | None:
            return "   "

        gap = _make_gap_result(required=("host",), provided={})
        result = await collect_missing_arguments(
            gap_result=gap,
            ask_callback=_ask_whitespace,
        )
        assert result.is_complete
        assert result.collected_args["host"] == "   "
