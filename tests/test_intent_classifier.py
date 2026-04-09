"""Tests for LLM-powered natural-language intent classifier.

Verifies that free-text user input is correctly mapped to one of the six
canonical CLI verbs (status, watch, run, queue, cancel, history) plus
extracted parameters.

All LLM calls are mocked -- these are unit tests for the classifier
logic, prompt construction, and response parsing, not integration tests
for the LLM backend.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from jules_daemon.cli.verbs import Verb
from jules_daemon.llm.errors import LLMParseError
from jules_daemon.llm.intent_classifier import (
    ClassifiedIntent,
    IntentClassifier,
    IntentConfidence,
    build_intent_system_prompt,
    build_intent_user_prompt,
    classify_intent,
    parse_intent_response,
)


# ---------------------------------------------------------------------------
# ClassifiedIntent model tests
# ---------------------------------------------------------------------------


class TestClassifiedIntent:
    """Tests for the immutable ClassifiedIntent data model."""

    def test_create_basic_intent(self) -> None:
        intent = ClassifiedIntent(
            verb=Verb.STATUS,
            confidence=IntentConfidence.HIGH,
            parameters={},
            raw_input="what's the current status?",
            reasoning="User is asking about the current state.",
        )
        assert intent.verb == Verb.STATUS
        assert intent.confidence == IntentConfidence.HIGH
        assert intent.parameters == {}
        assert intent.raw_input == "what's the current status?"
        assert intent.reasoning == "User is asking about the current state."

    def test_create_intent_with_parameters(self) -> None:
        intent = ClassifiedIntent(
            verb=Verb.RUN,
            confidence=IntentConfidence.HIGH,
            parameters={
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run the smoke tests",
            },
            raw_input="run the smoke tests on deploy@staging.example.com",
            reasoning="User wants to execute smoke tests on staging.",
        )
        assert intent.verb == Verb.RUN
        assert intent.parameters["target_host"] == "staging.example.com"
        assert intent.parameters["target_user"] == "deploy"

    def test_frozen_immutability(self) -> None:
        intent = ClassifiedIntent(
            verb=Verb.STATUS,
            confidence=IntentConfidence.HIGH,
            parameters={},
            raw_input="status",
            reasoning="Checking status.",
        )
        with pytest.raises(AttributeError):
            intent.verb = Verb.CANCEL  # type: ignore[misc]

    def test_empty_raw_input_raises(self) -> None:
        with pytest.raises(ValueError, match="raw_input must not be empty"):
            ClassifiedIntent(
                verb=Verb.STATUS,
                confidence=IntentConfidence.HIGH,
                parameters={},
                raw_input="",
                reasoning="Checking status.",
            )

    def test_whitespace_raw_input_raises(self) -> None:
        with pytest.raises(ValueError, match="raw_input must not be empty"):
            ClassifiedIntent(
                verb=Verb.STATUS,
                confidence=IntentConfidence.HIGH,
                parameters={},
                raw_input="   ",
                reasoning="Checking status.",
            )

    def test_empty_reasoning_raises(self) -> None:
        with pytest.raises(ValueError, match="reasoning must not be empty"):
            ClassifiedIntent(
                verb=Verb.STATUS,
                confidence=IntentConfidence.HIGH,
                parameters={},
                raw_input="status",
                reasoning="",
            )

    def test_is_ambiguous_property(self) -> None:
        low = ClassifiedIntent(
            verb=Verb.STATUS,
            confidence=IntentConfidence.LOW,
            parameters={},
            raw_input="do something",
            reasoning="Unclear what the user wants.",
        )
        assert low.is_ambiguous is True

        high = ClassifiedIntent(
            verb=Verb.STATUS,
            confidence=IntentConfidence.HIGH,
            parameters={},
            raw_input="check status",
            reasoning="Clear request.",
        )
        assert high.is_ambiguous is False

        medium = ClassifiedIntent(
            verb=Verb.STATUS,
            confidence=IntentConfidence.MEDIUM,
            parameters={},
            raw_input="check stuff",
            reasoning="Probably status.",
        )
        assert medium.is_ambiguous is False

    def test_to_dict_serialization(self) -> None:
        intent = ClassifiedIntent(
            verb=Verb.RUN,
            confidence=IntentConfidence.HIGH,
            parameters={"target_host": "prod"},
            raw_input="run tests on prod",
            reasoning="User wants to run tests.",
        )
        data = intent.to_dict()
        assert data["verb"] == "run"
        assert data["confidence"] == "high"
        assert data["parameters"]["target_host"] == "prod"
        assert data["raw_input"] == "run tests on prod"
        assert data["reasoning"] == "User wants to run tests."


class TestIntentConfidence:
    """Tests for the IntentConfidence enum."""

    def test_values(self) -> None:
        assert IntentConfidence.HIGH.value == "high"
        assert IntentConfidence.MEDIUM.value == "medium"
        assert IntentConfidence.LOW.value == "low"


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------


class TestBuildIntentSystemPrompt:
    """Tests for system prompt construction."""

    def test_prompt_contains_all_verbs(self) -> None:
        prompt = build_intent_system_prompt()
        for verb in Verb:
            assert verb.value in prompt

    def test_prompt_contains_json_schema(self) -> None:
        prompt = build_intent_system_prompt()
        assert '"verb"' in prompt
        assert '"confidence"' in prompt
        assert '"parameters"' in prompt
        assert '"reasoning"' in prompt

    def test_prompt_contains_role_definition(self) -> None:
        prompt = build_intent_system_prompt()
        assert "intent classifier" in prompt.lower() or "classify" in prompt.lower()

    def test_prompt_describes_each_verb(self) -> None:
        prompt = build_intent_system_prompt()
        assert "status" in prompt.lower()
        assert "watch" in prompt.lower()
        assert "run" in prompt.lower()
        assert "queue" in prompt.lower()
        assert "cancel" in prompt.lower()
        assert "history" in prompt.lower()

    def test_prompt_describes_parameter_extraction(self) -> None:
        prompt = build_intent_system_prompt()
        # Should instruct the LLM to extract SSH target params for run/queue
        assert "target_host" in prompt or "host" in prompt.lower()
        assert "target_user" in prompt or "user" in prompt.lower()


class TestBuildIntentUserPrompt:
    """Tests for user prompt construction."""

    def test_contains_user_input(self) -> None:
        prompt = build_intent_user_prompt(
            user_input="run the smoke tests on deploy@staging",
        )
        assert "run the smoke tests on deploy@staging" in prompt

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            build_intent_user_prompt(user_input="")

    def test_whitespace_input_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            build_intent_user_prompt(user_input="   ")

    def test_with_conversation_context(self) -> None:
        prompt = build_intent_user_prompt(
            user_input="check on those tests",
            conversation_context="Previously ran tests on staging.example.com",
        )
        assert "check on those tests" in prompt
        assert "Previously ran tests on staging.example.com" in prompt


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


class TestParseIntentResponse:
    """Tests for parsing raw LLM text into ClassifiedIntent."""

    def test_parse_valid_status_response(self) -> None:
        response = json.dumps({
            "verb": "status",
            "confidence": "high",
            "parameters": {"verbose": False},
            "reasoning": "User is asking about current test status.",
        })
        intent = parse_intent_response(response, raw_input="what's happening?")
        assert intent.verb == Verb.STATUS
        assert intent.confidence == IntentConfidence.HIGH
        assert intent.parameters == {"verbose": False}
        assert intent.raw_input == "what's happening?"

    def test_parse_valid_run_response(self) -> None:
        response = json.dumps({
            "verb": "run",
            "confidence": "high",
            "parameters": {
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run the smoke tests",
            },
            "reasoning": "User wants to execute smoke tests on the staging server.",
        })
        intent = parse_intent_response(
            response,
            raw_input="run the smoke tests on deploy@staging.example.com",
        )
        assert intent.verb == Verb.RUN
        assert intent.parameters["target_host"] == "staging.example.com"
        assert intent.parameters["target_user"] == "deploy"
        assert intent.parameters["natural_language"] == "run the smoke tests"

    def test_parse_valid_cancel_response(self) -> None:
        response = json.dumps({
            "verb": "cancel",
            "confidence": "high",
            "parameters": {"reason": "No longer needed"},
            "reasoning": "User wants to stop the current test run.",
        })
        intent = parse_intent_response(response, raw_input="stop the tests")
        assert intent.verb == Verb.CANCEL
        assert intent.parameters["reason"] == "No longer needed"

    def test_parse_valid_watch_response(self) -> None:
        response = json.dumps({
            "verb": "watch",
            "confidence": "medium",
            "parameters": {"tail_lines": 100},
            "reasoning": "User wants to see live test output.",
        })
        intent = parse_intent_response(
            response, raw_input="show me the test output",
        )
        assert intent.verb == Verb.WATCH
        assert intent.parameters["tail_lines"] == 100

    def test_parse_valid_queue_response(self) -> None:
        response = json.dumps({
            "verb": "queue",
            "confidence": "high",
            "parameters": {
                "target_host": "prod.example.com",
                "target_user": "ci",
                "natural_language": "run integration tests",
            },
            "reasoning": "User wants to queue tests for later.",
        })
        intent = parse_intent_response(
            response, raw_input="queue integration tests on ci@prod.example.com",
        )
        assert intent.verb == Verb.QUEUE
        assert intent.parameters["target_host"] == "prod.example.com"

    def test_parse_valid_history_response(self) -> None:
        response = json.dumps({
            "verb": "history",
            "confidence": "high",
            "parameters": {"limit": 10},
            "reasoning": "User wants to see past test results.",
        })
        intent = parse_intent_response(
            response, raw_input="show me the last 10 test runs",
        )
        assert intent.verb == Verb.HISTORY
        assert intent.parameters["limit"] == 10

    def test_parse_json_in_code_fence(self) -> None:
        response = (
            "I'll classify this intent:\n\n```json\n"
            + json.dumps({
                "verb": "status",
                "confidence": "high",
                "parameters": {},
                "reasoning": "User is checking status.",
            })
            + "\n```"
        )
        intent = parse_intent_response(
            response, raw_input="what's going on?",
        )
        assert intent.verb == Verb.STATUS

    def test_parse_missing_verb_raises(self) -> None:
        response = json.dumps({
            "confidence": "high",
            "parameters": {},
            "reasoning": "Missing verb.",
        })
        with pytest.raises(LLMParseError, match="verb"):
            parse_intent_response(response, raw_input="test")

    def test_parse_invalid_verb_raises(self) -> None:
        response = json.dumps({
            "verb": "destroy",
            "confidence": "high",
            "parameters": {},
            "reasoning": "Invalid verb.",
        })
        with pytest.raises(LLMParseError, match="verb"):
            parse_intent_response(response, raw_input="test")

    def test_parse_missing_confidence_raises(self) -> None:
        response = json.dumps({
            "verb": "status",
            "parameters": {},
            "reasoning": "No confidence.",
        })
        with pytest.raises(LLMParseError, match="confidence"):
            parse_intent_response(response, raw_input="test")

    def test_parse_invalid_confidence_raises(self) -> None:
        response = json.dumps({
            "verb": "status",
            "confidence": "ultra",
            "parameters": {},
            "reasoning": "Bad confidence.",
        })
        with pytest.raises(LLMParseError, match="confidence"):
            parse_intent_response(response, raw_input="test")

    def test_parse_missing_reasoning_raises(self) -> None:
        response = json.dumps({
            "verb": "status",
            "confidence": "high",
            "parameters": {},
        })
        with pytest.raises(LLMParseError, match="reasoning"):
            parse_intent_response(response, raw_input="test")

    def test_parse_empty_response_raises(self) -> None:
        with pytest.raises(LLMParseError):
            parse_intent_response("", raw_input="test")

    def test_parse_non_json_raises(self) -> None:
        with pytest.raises(LLMParseError):
            parse_intent_response(
                "I think you want to check status",
                raw_input="test",
            )

    def test_parse_missing_parameters_defaults_to_empty(self) -> None:
        response = json.dumps({
            "verb": "status",
            "confidence": "high",
            "reasoning": "User wants status.",
        })
        intent = parse_intent_response(response, raw_input="check status")
        assert intent.parameters == {}

    def test_parse_low_confidence_intent(self) -> None:
        response = json.dumps({
            "verb": "status",
            "confidence": "low",
            "parameters": {},
            "reasoning": "The user's request is ambiguous.",
        })
        intent = parse_intent_response(
            response, raw_input="hmm do something",
        )
        assert intent.confidence == IntentConfidence.LOW
        assert intent.is_ambiguous is True


# ---------------------------------------------------------------------------
# IntentClassifier class tests (LLM mocked)
# ---------------------------------------------------------------------------


def _make_mock_completion(content: str) -> MagicMock:
    """Create a mock ChatCompletion with the given content."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    return mock


class TestIntentClassifier:
    """Tests for the IntentClassifier class with mocked LLM."""

    def _make_classifier(self) -> tuple[IntentClassifier, MagicMock]:
        """Create an IntentClassifier with a mocked OpenAI client."""
        mock_client = MagicMock()
        config = MagicMock()
        config.default_model = "openai:conn:gpt-4"
        config.base_url = "https://example.com/api"
        config.api_key = "test-key"
        config.timeout = 120.0
        config.max_retries = 2
        config.verify_ssl = True

        classifier = IntentClassifier(
            client=mock_client,
            config=config,
        )
        return classifier, mock_client

    @patch("jules_daemon.llm.intent_classifier.create_completion")
    def test_classify_status_intent(
        self, mock_create: MagicMock,
    ) -> None:
        classifier, _ = self._make_classifier()

        mock_create.return_value = _make_mock_completion(
            json.dumps({
                "verb": "status",
                "confidence": "high",
                "parameters": {"verbose": False},
                "reasoning": "User is asking about test status.",
            })
        )

        intent = classifier.classify(user_input="what's happening with my tests?")
        assert intent.verb == Verb.STATUS
        assert intent.confidence == IntentConfidence.HIGH

    @patch("jules_daemon.llm.intent_classifier.create_completion")
    def test_classify_run_intent(
        self, mock_create: MagicMock,
    ) -> None:
        classifier, _ = self._make_classifier()

        mock_create.return_value = _make_mock_completion(
            json.dumps({
                "verb": "run",
                "confidence": "high",
                "parameters": {
                    "target_host": "staging.example.com",
                    "target_user": "deploy",
                    "natural_language": "run the regression suite",
                },
                "reasoning": "User wants to execute the regression suite on staging.",
            })
        )

        intent = classifier.classify(
            user_input="run the regression suite on deploy@staging.example.com",
        )
        assert intent.verb == Verb.RUN
        assert intent.parameters["target_host"] == "staging.example.com"
        assert intent.parameters["target_user"] == "deploy"

    @patch("jules_daemon.llm.intent_classifier.create_completion")
    def test_classify_cancel_intent(
        self, mock_create: MagicMock,
    ) -> None:
        classifier, _ = self._make_classifier()

        mock_create.return_value = _make_mock_completion(
            json.dumps({
                "verb": "cancel",
                "confidence": "high",
                "parameters": {"reason": "Taking too long"},
                "reasoning": "User wants to abort the current run.",
            })
        )

        intent = classifier.classify(user_input="stop the tests, they're taking too long")
        assert intent.verb == Verb.CANCEL
        assert intent.parameters["reason"] == "Taking too long"

    @patch("jules_daemon.llm.intent_classifier.create_completion")
    def test_classify_watch_intent(
        self, mock_create: MagicMock,
    ) -> None:
        classifier, _ = self._make_classifier()

        mock_create.return_value = _make_mock_completion(
            json.dumps({
                "verb": "watch",
                "confidence": "medium",
                "parameters": {"tail_lines": 50},
                "reasoning": "User wants to see live output.",
            })
        )

        intent = classifier.classify(user_input="show me the test output")
        assert intent.verb == Verb.WATCH

    @patch("jules_daemon.llm.intent_classifier.create_completion")
    def test_classify_history_intent(
        self, mock_create: MagicMock,
    ) -> None:
        classifier, _ = self._make_classifier()

        mock_create.return_value = _make_mock_completion(
            json.dumps({
                "verb": "history",
                "confidence": "high",
                "parameters": {"limit": 5},
                "reasoning": "User wants recent test results.",
            })
        )

        intent = classifier.classify(user_input="show me the last 5 test runs")
        assert intent.verb == Verb.HISTORY
        assert intent.parameters["limit"] == 5

    @patch("jules_daemon.llm.intent_classifier.create_completion")
    def test_classify_queue_intent(
        self, mock_create: MagicMock,
    ) -> None:
        classifier, _ = self._make_classifier()

        mock_create.return_value = _make_mock_completion(
            json.dumps({
                "verb": "queue",
                "confidence": "high",
                "parameters": {
                    "target_host": "prod.example.com",
                    "target_user": "ci",
                    "natural_language": "run integration tests",
                },
                "reasoning": "User wants to queue tests for later execution.",
            })
        )

        intent = classifier.classify(
            user_input="queue integration tests on ci@prod.example.com",
        )
        assert intent.verb == Verb.QUEUE

    @patch("jules_daemon.llm.intent_classifier.create_completion")
    def test_classify_empty_input_raises(
        self, mock_create: MagicMock,
    ) -> None:
        classifier, _ = self._make_classifier()
        with pytest.raises(ValueError, match="must not be empty"):
            classifier.classify(user_input="")

    @patch("jules_daemon.llm.intent_classifier.create_completion")
    def test_classify_empty_llm_response_raises(
        self, mock_create: MagicMock,
    ) -> None:
        classifier, _ = self._make_classifier()

        mock_response = MagicMock()
        mock_response.choices = []
        mock_create.return_value = mock_response

        with pytest.raises(LLMParseError, match="empty"):
            classifier.classify(user_input="check status")

    @patch("jules_daemon.llm.intent_classifier.create_completion")
    def test_classify_none_content_raises(
        self, mock_create: MagicMock,
    ) -> None:
        classifier, _ = self._make_classifier()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_create.return_value = mock_response

        with pytest.raises(LLMParseError, match="empty"):
            classifier.classify(user_input="check status")

    @patch("jules_daemon.llm.intent_classifier.create_completion")
    def test_classify_preserves_raw_input(
        self, mock_create: MagicMock,
    ) -> None:
        classifier, _ = self._make_classifier()

        original_input = "  run the smoke tests on staging  "
        mock_create.return_value = _make_mock_completion(
            json.dumps({
                "verb": "run",
                "confidence": "high",
                "parameters": {"natural_language": "run the smoke tests on staging"},
                "reasoning": "Test request.",
            })
        )

        intent = classifier.classify(user_input=original_input)
        assert intent.raw_input == original_input.strip()

    @patch("jules_daemon.llm.intent_classifier.create_completion")
    def test_classify_with_conversation_context(
        self, mock_create: MagicMock,
    ) -> None:
        classifier, _ = self._make_classifier()

        mock_create.return_value = _make_mock_completion(
            json.dumps({
                "verb": "watch",
                "confidence": "high",
                "parameters": {},
                "reasoning": "User wants to watch previously mentioned tests.",
            })
        )

        intent = classifier.classify(
            user_input="show me how they're doing",
            conversation_context="Previously ran tests on staging.",
        )
        assert intent.verb == Verb.WATCH

    def test_classifier_validates_temperature_range(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            IntentClassifier(
                client=MagicMock(),
                config=MagicMock(),
                temperature=-0.1,
            )
        with pytest.raises(ValueError, match="temperature"):
            IntentClassifier(
                client=MagicMock(),
                config=MagicMock(),
                temperature=2.1,
            )


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------


class TestClassifyIntentFunction:
    """Tests for the classify_intent convenience function."""

    @patch("jules_daemon.llm.intent_classifier.create_completion")
    def test_one_shot_classification(
        self, mock_create: MagicMock,
    ) -> None:
        mock_create.return_value = _make_mock_completion(
            json.dumps({
                "verb": "status",
                "confidence": "high",
                "parameters": {},
                "reasoning": "Checking status.",
            })
        )

        intent = classify_intent(
            user_input="what's the status?",
            client=MagicMock(),
            config=MagicMock(),
        )
        assert intent.verb == Verb.STATUS
        assert intent.confidence == IntentConfidence.HIGH
