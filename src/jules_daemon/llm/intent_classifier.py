"""LLM-powered natural-language intent classifier.

Maps free-text user input to one of the six canonical CLI verbs
(status, watch, run, queue, cancel, history) plus extracted parameters.

This is the daemon's natural-language front-end: instead of requiring
structured CLI syntax like ``run deploy@staging run the tests``, the
classifier lets users type free-form English like "run the smoke tests
on staging" and maps it to the correct verb with extracted SSH target
parameters.

Design:
    - Single LLM call per classification (no multi-turn)
    - Temperature 0.0 for deterministic classification
    - System prompt describes all six verbs and their parameter schemas
    - Response is structured JSON parsed into a ClassifiedIntent
    - Missing parameters default to empty dict (caller validates)
    - Immutable data models throughout

The classifier does not validate parameters against the verb-specific
Args schema -- that is the caller's responsibility. The classifier
extracts what the LLM identifies and the caller constructs the
appropriate ``*Args`` dataclass with validation.

Usage::

    from jules_daemon.llm.intent_classifier import (
        IntentClassifier,
        classify_intent,
    )

    # Reusable classifier (preferred for daemon)
    classifier = IntentClassifier(client=client, config=config)
    intent = classifier.classify(user_input="run the smoke tests on staging")

    # One-shot convenience
    intent = classify_intent(
        user_input="check on my tests",
        client=client,
        config=config,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openai import OpenAI

from jules_daemon.cli.verbs import Verb
from jules_daemon.llm.client import create_completion
from jules_daemon.llm.config import LLMConfig
from jules_daemon.llm.errors import LLMParseError
from jules_daemon.llm.models import ToolCallingMode
from jules_daemon.llm.response_parser import extract_json_from_text

logger = logging.getLogger(__name__)

__all__ = [
    "ClassifiedIntent",
    "IntentClassifier",
    "IntentConfidence",
    "build_intent_system_prompt",
    "build_intent_user_prompt",
    "classify_intent",
    "parse_intent_response",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TEMPERATURE: float = 0.0
"""Deterministic output for consistent classification."""

_MIN_TEMPERATURE: float = 0.0
_MAX_TEMPERATURE: float = 2.0

# Valid verb string values for validation
_VALID_VERBS: frozenset[str] = frozenset(v.value for v in Verb)

# Mapping from string to Verb enum for fast lookup
_VERB_LOOKUP: dict[str, Verb] = {v.value: v for v in Verb}


# ---------------------------------------------------------------------------
# IntentConfidence
# ---------------------------------------------------------------------------


class IntentConfidence(Enum):
    """Classifier's confidence in the verb classification.

    Values:
        HIGH: Clear, unambiguous match to a single verb.
        MEDIUM: Likely match, but some ambiguity. Parameters may
            be incomplete.
        LOW: Ambiguous input. The verb is the classifier's best
            guess but the user should confirm.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_CONFIDENCE_LOOKUP: dict[str, IntentConfidence] = {
    c.value: c for c in IntentConfidence
}


# ---------------------------------------------------------------------------
# ClassifiedIntent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassifiedIntent:
    """Immutable result of intent classification.

    Binds the classified verb to the extracted parameters, confidence
    level, and the LLM's reasoning for traceability.

    Attributes:
        verb: The canonical CLI verb the input was classified as.
        confidence: How confident the classifier is in this mapping.
        parameters: Extracted parameters as a plain dict. Keys depend
            on the verb (e.g., target_host, target_user for run/queue).
            Empty dict when no parameters were extracted.
        raw_input: The original user input string (stripped).
        reasoning: The LLM's explanation of why it chose this verb.
    """

    verb: Verb
    confidence: IntentConfidence
    parameters: dict[str, Any]
    raw_input: str
    reasoning: str

    def __post_init__(self) -> None:
        if not self.raw_input or not self.raw_input.strip():
            raise ValueError("raw_input must not be empty or whitespace-only")
        if not self.reasoning or not self.reasoning.strip():
            raise ValueError("reasoning must not be empty or whitespace-only")

    @property
    def is_ambiguous(self) -> bool:
        """True if the classification confidence is LOW.

        When ambiguous, the CLI should present the classification to the
        user for confirmation before proceeding.
        """
        return self.confidence == IntentConfidence.LOW

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for wiki YAML or IPC.

        The verb and confidence are serialized as their string values.
        """
        return {
            "verb": self.verb.value,
            "confidence": self.confidence.value,
            "parameters": dict(self.parameters),
            "raw_input": self.raw_input,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Verb definitions for the system prompt
# ---------------------------------------------------------------------------

_VERB_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    (
        "status",
        "Query the current run state -- is a test running, idle, or failed?",
        '{"verbose": false}',
    ),
    (
        "watch",
        "Live-stream output from a running test session. Use when the user wants to see output, logs, or monitor progress.",
        '{"run_id": null, "tail_lines": 50}',
    ),
    (
        "run",
        "Start a new test execution on a remote host via SSH. Use when the user wants to run, execute, or start tests.",
        '{"target_host": "hostname", "target_user": "username", "natural_language": "what to run", "target_port": 22, "key_path": null}',
    ),
    (
        "queue",
        "Queue a test command for later execution when the daemon is busy. Use when the user explicitly says queue, schedule, or run later.",
        '{"target_host": "hostname", "target_user": "username", "natural_language": "what to run", "target_port": 22, "key_path": null, "priority": 0}',
    ),
    (
        "cancel",
        "Cancel the current or a queued test run. Use when the user wants to stop, abort, cancel, or kill a test.",
        '{"run_id": null, "force": false, "reason": null}',
    ),
    (
        "history",
        "View past test run results. Use when the user asks about previous runs, results, or test history.",
        '{"limit": 20, "status_filter": null, "host_filter": null, "verbose": false}',
    ),
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_intent_system_prompt() -> str:
    """Build the system prompt for intent classification.

    The system prompt instructs the LLM to:
    1. Classify the user's input into exactly one of six verbs
    2. Extract relevant parameters for that verb
    3. Provide a confidence level and reasoning
    4. Return structured JSON

    Returns:
        Complete system prompt string.
    """
    sections: list[str] = [
        _section_role(),
        _section_verb_definitions(),
        _section_parameter_extraction(),
        _section_output_schema(),
        _section_classification_rules(),
    ]
    return "\n\n".join(sections)


def _section_role() -> str:
    """Role definition for the intent classifier."""
    return (
        "You are a natural-language intent classifier for a test execution daemon. "
        "Your job is to analyze free-text user input and classify it into exactly "
        "one of six canonical verb actions, extract relevant parameters, and report "
        "your confidence level.\n"
        "\n"
        "You do NOT execute commands or interact with any systems. You only classify "
        "intent and extract parameters from the user's text."
    )


def _section_verb_definitions() -> str:
    """Verb definitions with descriptions and parameter schemas."""
    lines: list[str] = [
        "## Verb Definitions",
        "",
        "Classify the user's input into exactly one of these six verbs:",
        "",
    ]

    for verb_name, description, param_example in _VERB_DEFINITIONS:
        lines.append(f"### {verb_name}")
        lines.append(f"Description: {description}")
        lines.append(f"Parameters: {param_example}")
        lines.append("")

    return "\n".join(lines)


def _section_parameter_extraction() -> str:
    """Instructions for extracting parameters from user input."""
    return (
        "## Parameter Extraction Rules\n"
        "\n"
        "For `run` and `queue` verbs:\n"
        "- Extract `target_host` and `target_user` from SSH target patterns "
        "like `user@host`, `user@host:port`, or from context clues.\n"
        "- Extract `natural_language` as the description of what tests to run, "
        "excluding the SSH target information.\n"
        "- If port is mentioned, include `target_port`.\n"
        "- If key path is mentioned, include `key_path`.\n"
        "\n"
        "For `cancel` verb:\n"
        "- Extract `reason` if the user provides one.\n"
        "- Extract `run_id` if the user references a specific run.\n"
        "- Set `force` to true if the user says force, kill, or similar.\n"
        "\n"
        "For `watch` verb:\n"
        "- Extract `run_id` if the user references a specific run.\n"
        "- Extract `tail_lines` if the user specifies how many lines.\n"
        "\n"
        "For `history` verb:\n"
        "- Extract `limit` if the user specifies how many results.\n"
        "- Extract `status_filter` if the user filters by status (e.g., 'failed runs').\n"
        "- Extract `host_filter` if the user filters by host.\n"
        "\n"
        "For `status` verb:\n"
        "- Set `verbose` to true if the user asks for details or extended info.\n"
        "\n"
        "Only include parameters that are explicitly or clearly implied in "
        "the user's input. Omit parameters you cannot determine -- do not guess."
    )


def _section_output_schema() -> str:
    """JSON output schema with example."""
    example = json.dumps(
        {
            "verb": "run",
            "confidence": "high",
            "parameters": {
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run the smoke tests",
            },
            "reasoning": "The user explicitly requests running smoke tests on a specific host.",
        },
        indent=2,
    )

    return (
        "## Output Schema\n"
        "\n"
        "You MUST respond with a JSON object in exactly this format:\n"
        "\n"
        "```json\n"
        f"{example}\n"
        "```\n"
        "\n"
        "Field definitions:\n"
        '  - "verb": One of "status", "watch", "run", "queue", "cancel", "history"\n'
        '  - "confidence": One of "high", "medium", "low"\n'
        '  - "parameters": Object with extracted parameters for the verb (empty {} if none)\n'
        '  - "reasoning": Brief explanation of why you chose this verb and parameters'
    )


def _section_classification_rules() -> str:
    """Behavioral rules for classification."""
    return (
        "## Classification Rules\n"
        "\n"
        '1. If the input is ambiguous between two verbs, choose the most likely one and set confidence to "low"\n'
        '2. If the user mentions running/executing tests, classify as "run" unless they explicitly say "queue" or "schedule"\n'
        '3. If the user asks "what\'s happening" or checks on status, classify as "status"\n'
        '4. If the user wants to see output or logs, classify as "watch"\n'
        '5. If the user says stop/cancel/abort/kill, classify as "cancel"\n'
        '6. If the user asks about past results or history, classify as "history"\n'
        '7. Default to "status" with confidence "low" if the intent is completely unclear\n'
        "8. Never invent parameters that are not present or strongly implied in the input\n"
        "9. When the user provides SSH-style targets (user@host), always extract them"
    )


def build_intent_user_prompt(
    *,
    user_input: str,
    conversation_context: str | None = None,
) -> str:
    """Build the user prompt for intent classification.

    Args:
        user_input: The user's free-text input to classify.
        conversation_context: Optional prior conversation context that
            may help disambiguate the user's intent.

    Returns:
        Formatted user prompt string.

    Raises:
        ValueError: If user_input is empty or whitespace-only.
    """
    stripped = user_input.strip()
    if not stripped:
        raise ValueError("user_input must not be empty or whitespace-only")

    sections: list[str] = [
        "## User Input",
        "",
        stripped,
    ]

    if conversation_context is not None:
        context_stripped = conversation_context.strip()
        if context_stripped:
            sections.extend([
                "",
                "## Conversation Context",
                "",
                context_stripped,
            ])

    sections.extend([
        "",
        "Classify this input and respond with the JSON intent classification.",
    ])

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_intent_response(
    text: str,
    *,
    raw_input: str,
) -> ClassifiedIntent:
    """Parse raw LLM text into a validated ClassifiedIntent.

    Pipeline:
    1. Extract JSON from the text (handles code fences, mixed text)
    2. Validate required fields (verb, confidence, reasoning)
    3. Resolve verb string to Verb enum
    4. Resolve confidence string to IntentConfidence enum
    5. Construct immutable ClassifiedIntent

    Args:
        text: Raw LLM response text containing JSON.
        raw_input: The original user input string for inclusion
            in the ClassifiedIntent.

    Returns:
        Validated, immutable ClassifiedIntent.

    Raises:
        LLMParseError: If JSON extraction or validation fails.
    """
    data = extract_json_from_text(text)

    # Validate and resolve verb
    verb_str = data.get("verb")
    if verb_str is None:
        raise LLMParseError(
            "Missing required field 'verb' in intent classification response",
            raw_content=text,
        )

    verb_lower = str(verb_str).strip().lower()
    verb = _VERB_LOOKUP.get(verb_lower)
    if verb is None:
        raise LLMParseError(
            f"Invalid verb {verb_str!r} in intent classification response. "
            f"Valid verbs: {', '.join(sorted(_VALID_VERBS))}",
            raw_content=text,
        )

    # Validate and resolve confidence
    confidence_str = data.get("confidence")
    if confidence_str is None:
        raise LLMParseError(
            "Missing required field 'confidence' in intent classification response",
            raw_content=text,
        )

    confidence_lower = str(confidence_str).strip().lower()
    confidence = _CONFIDENCE_LOOKUP.get(confidence_lower)
    if confidence is None:
        valid_conf = ", ".join(sorted(c.value for c in IntentConfidence))
        raise LLMParseError(
            f"Invalid confidence {confidence_str!r} in intent classification response. "
            f"Valid values: {valid_conf}",
            raw_content=text,
        )

    # Validate reasoning
    reasoning = data.get("reasoning")
    if reasoning is None or not str(reasoning).strip():
        raise LLMParseError(
            "Missing or empty required field 'reasoning' in intent classification response",
            raw_content=text,
        )

    # Parameters default to empty dict if missing
    parameters = data.get("parameters")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise LLMParseError(
            f"'parameters' must be a JSON object, got {type(parameters).__name__}",
            raw_content=text,
        )

    return ClassifiedIntent(
        verb=verb,
        confidence=confidence,
        parameters=dict(parameters),
        raw_input=raw_input.strip(),
        reasoning=str(reasoning).strip(),
    )


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------


class IntentClassifier:
    """LLM-powered natural-language intent classifier.

    Maps free-text user input to one of the six canonical CLI verbs
    plus extracted parameters. The classifier is stateless and reusable
    across multiple classification calls.

    For the daemon's single-user model, one instance per daemon lifetime.

    Args:
        client: OpenAI client configured for Dataiku Mesh.
        config: LLM configuration.
        tool_calling_mode: How tools are passed to the LLM.
        temperature: LLM temperature (0.0 for deterministic).
    """

    def __init__(
        self,
        *,
        client: OpenAI,
        config: LLMConfig,
        tool_calling_mode: ToolCallingMode = ToolCallingMode.NATIVE,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> None:
        if temperature < _MIN_TEMPERATURE or temperature > _MAX_TEMPERATURE:
            raise ValueError(
                f"temperature must be between {_MIN_TEMPERATURE} and "
                f"{_MAX_TEMPERATURE}, got {temperature}"
            )

        self._client = client
        self._config = config
        self._tool_calling_mode = tool_calling_mode
        self._temperature = temperature

        # Cache the system prompt (deterministic, same per classifier)
        self._system_prompt = build_intent_system_prompt()

    def classify(
        self,
        *,
        user_input: str,
        conversation_context: str | None = None,
    ) -> ClassifiedIntent:
        """Classify free-text user input into a verb + parameters.

        Pipeline:
        1. Validate input
        2. Build system + user prompt messages
        3. Call LLM via Dataiku Mesh
        4. Parse and validate JSON response
        5. Return immutable ClassifiedIntent

        Args:
            user_input: The user's free-text input to classify.
            conversation_context: Optional prior conversation context
                for disambiguation.

        Returns:
            Validated ClassifiedIntent with verb, parameters, and reasoning.

        Raises:
            ValueError: If user_input is empty.
            LLMParseError: If the LLM response cannot be parsed.
            LLMError: For LLM client errors (auth, connection, etc.).
        """
        stripped_input = user_input.strip()
        if not stripped_input:
            raise ValueError("user_input must not be empty or whitespace-only")

        logger.info(
            "Classifying intent: %.100s",
            stripped_input,
        )

        messages = self._build_messages(
            user_input=stripped_input,
            conversation_context=conversation_context,
        )

        raw_content = self._call_llm(messages=messages)

        intent = parse_intent_response(
            raw_content,
            raw_input=stripped_input,
        )

        logger.info(
            "Intent classified: verb=%s, confidence=%s, params=%d",
            intent.verb.value,
            intent.confidence.value,
            len(intent.parameters),
        )

        return intent

    def _build_messages(
        self,
        *,
        user_input: str,
        conversation_context: str | None,
    ) -> list[dict[str, str]]:
        """Build the LLM message list for classification."""
        user_content = build_intent_user_prompt(
            user_input=user_input,
            conversation_context=conversation_context,
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _call_llm(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> str:
        """Call the LLM and extract the response content.

        Args:
            messages: Message list for the LLM.

        Returns:
            Raw content string from the LLM response.

        Raises:
            LLMError: On any LLM client error.
            LLMParseError: If the response has no content.
        """
        response = create_completion(
            client=self._client,
            config=self._config,
            messages=messages,
            tool_calling_mode=self._tool_calling_mode,
            temperature=self._temperature,
        )

        if not response.choices:
            raise LLMParseError(
                "LLM returned empty choices list",
                raw_content="",
            )

        content = response.choices[0].message.content
        if not content:
            raise LLMParseError(
                "LLM returned empty content",
                raw_content="",
            )

        return content


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def classify_intent(
    *,
    user_input: str,
    client: OpenAI,
    config: LLMConfig,
    conversation_context: str | None = None,
    tool_calling_mode: ToolCallingMode = ToolCallingMode.NATIVE,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> ClassifiedIntent:
    """One-shot convenience function for intent classification.

    Creates a temporary IntentClassifier and calls classify().
    For repeated classifications, prefer creating an IntentClassifier
    instance directly to benefit from system prompt caching.

    Args:
        user_input: Free-text user input to classify.
        client: OpenAI client configured for Dataiku Mesh.
        config: LLM configuration.
        conversation_context: Optional prior conversation context.
        tool_calling_mode: How tools are passed to the LLM.
        temperature: LLM temperature.

    Returns:
        Validated ClassifiedIntent with verb, parameters, and reasoning.

    Raises:
        ValueError: If user_input is empty.
        LLMParseError: If the LLM response cannot be parsed.
        LLMError: For LLM client errors.
    """
    classifier = IntentClassifier(
        client=client,
        config=config,
        tool_calling_mode=tool_calling_mode,
        temperature=temperature,
    )
    return classifier.classify(
        user_input=user_input,
        conversation_context=conversation_context,
    )
