"""Tests for input classifier data models.

Validates the ClassificationResult dataclass and InputType enum
used by the deterministic input classification layer.
"""

from __future__ import annotations

import pytest

from jules_daemon.classifier.models import (
    ClassificationResult,
    InputType,
)


class TestInputType:
    """Tests for the InputType enum."""

    def test_command_value(self) -> None:
        assert InputType.COMMAND.value == "command"

    def test_natural_language_value(self) -> None:
        assert InputType.NATURAL_LANGUAGE.value == "natural_language"

    def test_query_value(self) -> None:
        assert InputType.QUERY.value == "query"

    def test_ambiguous_value(self) -> None:
        assert InputType.AMBIGUOUS.value == "ambiguous"

    def test_all_members_present(self) -> None:
        expected = {"COMMAND", "NATURAL_LANGUAGE", "QUERY", "AMBIGUOUS"}
        actual = {m.name for m in InputType}
        assert actual == expected


class TestClassificationResult:
    """Tests for the frozen ClassificationResult dataclass."""

    def test_create_valid_result(self) -> None:
        result = ClassificationResult(
            canonical_verb="run",
            extracted_args={"target_host": "staging.example.com"},
            confidence_score=0.95,
            input_type=InputType.COMMAND,
        )
        assert result.canonical_verb == "run"
        assert result.extracted_args == {"target_host": "staging.example.com"}
        assert result.confidence_score == 0.95
        assert result.input_type == InputType.COMMAND

    def test_frozen_immutability(self) -> None:
        result = ClassificationResult(
            canonical_verb="status",
            extracted_args={},
            confidence_score=0.8,
            input_type=InputType.QUERY,
        )
        with pytest.raises(AttributeError):
            result.canonical_verb = "run"  # type: ignore[misc]

    def test_empty_extracted_args(self) -> None:
        result = ClassificationResult(
            canonical_verb="status",
            extracted_args={},
            confidence_score=1.0,
            input_type=InputType.COMMAND,
        )
        assert result.extracted_args == {}

    def test_confidence_score_zero(self) -> None:
        result = ClassificationResult(
            canonical_verb="status",
            extracted_args={},
            confidence_score=0.0,
            input_type=InputType.AMBIGUOUS,
        )
        assert result.confidence_score == 0.0

    def test_confidence_score_one(self) -> None:
        result = ClassificationResult(
            canonical_verb="run",
            extracted_args={},
            confidence_score=1.0,
            input_type=InputType.COMMAND,
        )
        assert result.confidence_score == 1.0

    def test_empty_canonical_verb_raises(self) -> None:
        with pytest.raises(ValueError, match="canonical_verb must not be empty"):
            ClassificationResult(
                canonical_verb="",
                extracted_args={},
                confidence_score=0.5,
                input_type=InputType.COMMAND,
            )

    def test_whitespace_only_canonical_verb_raises(self) -> None:
        with pytest.raises(ValueError, match="canonical_verb must not be empty"):
            ClassificationResult(
                canonical_verb="   ",
                extracted_args={},
                confidence_score=0.5,
                input_type=InputType.COMMAND,
            )

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_score must be between 0.0 and 1.0"):
            ClassificationResult(
                canonical_verb="run",
                extracted_args={},
                confidence_score=-0.1,
                input_type=InputType.COMMAND,
            )

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_score must be between 0.0 and 1.0"):
            ClassificationResult(
                canonical_verb="run",
                extracted_args={},
                confidence_score=1.01,
                input_type=InputType.COMMAND,
            )

    def test_invalid_canonical_verb_not_in_registry_raises(self) -> None:
        with pytest.raises(ValueError, match="canonical_verb .* is not a recognized canonical verb"):
            ClassificationResult(
                canonical_verb="destroy",
                extracted_args={},
                confidence_score=0.5,
                input_type=InputType.COMMAND,
            )

    def test_complex_extracted_args(self) -> None:
        args = {
            "target_host": "prod.example.com",
            "target_user": "deploy",
            "natural_language": "run the smoke tests",
            "target_port": 2222,
        }
        result = ClassificationResult(
            canonical_verb="run",
            extracted_args=args,
            confidence_score=0.9,
            input_type=InputType.NATURAL_LANGUAGE,
        )
        assert result.extracted_args["target_host"] == "prod.example.com"
        assert result.extracted_args["target_port"] == 2222

    def test_equality(self) -> None:
        a = ClassificationResult(
            canonical_verb="run",
            extracted_args={"host": "a"},
            confidence_score=0.9,
            input_type=InputType.COMMAND,
        )
        b = ClassificationResult(
            canonical_verb="run",
            extracted_args={"host": "a"},
            confidence_score=0.9,
            input_type=InputType.COMMAND,
        )
        assert a == b

    def test_inequality_different_verb(self) -> None:
        a = ClassificationResult(
            canonical_verb="run",
            extracted_args={},
            confidence_score=0.9,
            input_type=InputType.COMMAND,
        )
        b = ClassificationResult(
            canonical_verb="status",
            extracted_args={},
            confidence_score=0.9,
            input_type=InputType.COMMAND,
        )
        assert a != b

    def test_to_dict_serialization(self) -> None:
        result = ClassificationResult(
            canonical_verb="cancel",
            extracted_args={"run_id": "abc-123", "force": True},
            confidence_score=0.85,
            input_type=InputType.COMMAND,
        )
        serialized = result.to_dict()
        assert serialized == {
            "canonical_verb": "cancel",
            "extracted_args": {"run_id": "abc-123", "force": True},
            "confidence_score": 0.85,
            "input_type": "command",
        }

    def test_to_dict_returns_new_dict(self) -> None:
        result = ClassificationResult(
            canonical_verb="run",
            extracted_args={"host": "x"},
            confidence_score=0.5,
            input_type=InputType.NATURAL_LANGUAGE,
        )
        d1 = result.to_dict()
        d2 = result.to_dict()
        assert d1 is not d2
        assert d1["extracted_args"] is not d2["extracted_args"]

    def test_is_confident_property(self) -> None:
        high = ClassificationResult(
            canonical_verb="run",
            extracted_args={},
            confidence_score=0.8,
            input_type=InputType.COMMAND,
        )
        assert high.is_confident is True

        low = ClassificationResult(
            canonical_verb="run",
            extracted_args={},
            confidence_score=0.3,
            input_type=InputType.AMBIGUOUS,
        )
        assert low.is_confident is False

    def test_is_confident_at_threshold(self) -> None:
        """Confidence threshold is 0.7 -- exactly 0.7 should be confident."""
        at_threshold = ClassificationResult(
            canonical_verb="status",
            extracted_args={},
            confidence_score=0.7,
            input_type=InputType.QUERY,
        )
        assert at_threshold.is_confident is True

    def test_is_confident_below_threshold(self) -> None:
        below = ClassificationResult(
            canonical_verb="status",
            extracted_args={},
            confidence_score=0.69,
            input_type=InputType.QUERY,
        )
        assert below.is_confident is False
