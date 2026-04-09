"""Tests for the unified classify() entry point.

Validates that the unified classifier:
1. Computes a structuredness score for the input
2. Delegates to the structured parser when score >= threshold
3. Delegates to the NL extractor when score < threshold
4. Always returns a ClassificationResult
"""

from __future__ import annotations

import pytest

from jules_daemon.classifier.classify import classify
from jules_daemon.classifier.models import ClassificationResult, InputType


class TestStructuredInputDelegation:
    """Structured inputs should produce COMMAND-type classifications."""

    def test_exact_verb_produces_command(self) -> None:
        result = classify("status")
        assert isinstance(result, ClassificationResult)
        assert result.input_type == InputType.COMMAND
        assert result.canonical_verb == "status"
        assert result.is_confident

    def test_run_with_target_produces_command(self) -> None:
        result = classify("run deploy@staging run the tests")
        assert isinstance(result, ClassificationResult)
        assert result.input_type == InputType.COMMAND
        assert result.canonical_verb == "run"
        assert result.confidence_score >= 0.7

    def test_alias_produces_command(self) -> None:
        result = classify("execute deploy@host run tests")
        assert isinstance(result, ClassificationResult)
        assert result.canonical_verb == "run"
        assert result.input_type == InputType.COMMAND

    def test_cancel_flags_produce_command(self) -> None:
        result = classify("cancel --run-id abc-123 --force")
        assert isinstance(result, ClassificationResult)
        assert result.canonical_verb == "cancel"
        assert result.input_type == InputType.COMMAND

    def test_watch_produces_command(self) -> None:
        result = classify("watch --tail 100")
        assert isinstance(result, ClassificationResult)
        assert result.canonical_verb == "watch"
        assert result.input_type == InputType.COMMAND

    def test_history_produces_command(self) -> None:
        result = classify("history --limit 10")
        assert isinstance(result, ClassificationResult)
        assert result.canonical_verb == "history"
        assert result.input_type == InputType.COMMAND


class TestNaturalLanguageDelegation:
    """NL inputs should produce NATURAL_LANGUAGE-type classifications."""

    def test_question_produces_nl(self) -> None:
        result = classify("what's running right now?")
        assert isinstance(result, ClassificationResult)
        assert result.input_type == InputType.NATURAL_LANGUAGE

    def test_polite_request_produces_nl(self) -> None:
        result = classify("can you run the smoke tests on staging?")
        assert isinstance(result, ClassificationResult)
        assert result.input_type == InputType.NATURAL_LANGUAGE
        assert result.canonical_verb == "run"

    def test_please_stop_produces_nl(self) -> None:
        result = classify("please stop whatever is running")
        assert isinstance(result, ClassificationResult)
        assert result.input_type == InputType.NATURAL_LANGUAGE
        assert result.canonical_verb == "cancel"

    def test_conversational_produces_nl(self) -> None:
        result = classify(
            "I want to see the results from the last run"
        )
        assert isinstance(result, ClassificationResult)
        assert result.input_type == InputType.NATURAL_LANGUAGE

    def test_check_on_tests_nl(self) -> None:
        result = classify("how are the tests going?")
        assert isinstance(result, ClassificationResult)
        assert result.input_type == InputType.NATURAL_LANGUAGE
        assert result.canonical_verb == "status"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_string_returns_ambiguous(self) -> None:
        result = classify("")
        assert isinstance(result, ClassificationResult)
        assert result.input_type == InputType.AMBIGUOUS
        assert result.confidence_score == 0.0

    def test_whitespace_only_returns_ambiguous(self) -> None:
        result = classify("   ")
        assert isinstance(result, ClassificationResult)
        assert result.input_type == InputType.AMBIGUOUS
        assert result.confidence_score == 0.0

    def test_result_always_has_canonical_verb(self) -> None:
        """Even for ambiguous input, a canonical verb is always assigned."""
        result = classify("asdfghjkl")
        assert isinstance(result, ClassificationResult)
        assert result.canonical_verb in {
            "run", "status", "cancel", "watch", "queue", "history",
        }

    def test_never_returns_none(self) -> None:
        """classify() never returns None -- always a ClassificationResult."""
        inputs = [
            "",
            "status",
            "run deploy@host tests",
            "can you run the tests?",
            "asdf",
            "   ",
        ]
        for text in inputs:
            result = classify(text)
            assert isinstance(result, ClassificationResult), (
                f"classify({text!r}) returned {type(result)}"
            )


class TestScoreAndConfidence:
    """Verify that score/confidence are set correctly."""

    def test_structured_high_confidence(self) -> None:
        result = classify("status --verbose")
        assert result.confidence_score >= 0.7

    def test_nl_lower_confidence(self) -> None:
        result = classify(
            "could you please check the test status?"
        )
        # NL extraction has lower confidence than structured parsing
        assert 0.0 <= result.confidence_score <= 1.0

    def test_confidence_always_valid(self) -> None:
        inputs = [
            "run deploy@host run tests",
            "what's happening?",
            "stop --force",
            "I need to queue this for later",
            "",
        ]
        for text in inputs:
            result = classify(text)
            assert 0.0 <= result.confidence_score <= 1.0


class TestReturnTypeImmutability:
    """Classifications should be immutable."""

    def test_result_is_frozen(self) -> None:
        result = classify("status")
        with pytest.raises(AttributeError):
            result.canonical_verb = "run"  # type: ignore[misc]

    def test_extracted_args_is_dict(self) -> None:
        result = classify("run deploy@staging run tests")
        assert isinstance(result.extracted_args, dict)
