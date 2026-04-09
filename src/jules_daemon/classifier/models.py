"""Input classifier data models.

Defines the immutable ClassificationResult dataclass and InputType enum
used by the deterministic input classification layer. This is a fast,
pre-LLM layer that resolves common verb aliases and extracts basic
parameters without requiring a network call.

The ClassificationResult binds:
    - A canonical verb (normalized from the verb registry)
    - Extracted arguments (verb-specific key-value pairs)
    - A confidence score (0.0 to 1.0)
    - The structural input type (command, query, NL, ambiguous)

All models follow the project immutability convention: frozen dataclasses
with ``__post_init__`` validation. State changes require creating new
instances.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any

from jules_daemon.classifier.verb_registry import CANONICAL_VERBS

__all__ = [
    "ClassificationResult",
    "InputType",
]


# Confidence threshold: scores at or above this value are considered
# "confident" classifications that do not require LLM fallback.
_CONFIDENCE_THRESHOLD: float = 0.7


class InputType(Enum):
    """Structural classification of the user's input.

    Determines how the daemon processes the classified input:

    Values:
        COMMAND: Direct structured command with clear verb and arguments.
            Example: ``run deploy@staging run the tests``
        NATURAL_LANGUAGE: Free-form text requiring LLM interpretation
            to extract the verb and parameters.
            Example: ``can you kick off the smoke suite on staging?``
        QUERY: Informational request about system state or history.
            Example: ``what's running right now?``
        AMBIGUOUS: Input that cannot be classified with confidence.
            The daemon should request clarification or fall through
            to the LLM-powered classifier.
    """

    COMMAND = "command"
    NATURAL_LANGUAGE = "natural_language"
    QUERY = "query"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ClassificationResult:
    """Immutable result of deterministic input classification.

    Produced by the fast classifier (no LLM call). When the confidence
    score is below the threshold (0.7), the daemon should fall through
    to the LLM-powered IntentClassifier for a more nuanced analysis.

    Attributes:
        canonical_verb: Normalized verb from the canonical registry.
            Must be one of the six recognized verbs: run, status,
            cancel, watch, queue, history.
        extracted_args: Extracted arguments as a plain dict. Keys
            depend on the verb (e.g., target_host, target_user for
            run/queue). Empty dict when no arguments were extracted.
        confidence_score: Float between 0.0 and 1.0 (inclusive)
            indicating how confident the classifier is in this
            classification. Scores >= 0.7 are considered confident.
        input_type: Structural type of the input, indicating whether
            it is a direct command, a query, natural language, or
            ambiguous.
    """

    canonical_verb: str
    extracted_args: dict[str, Any]
    confidence_score: float
    input_type: InputType

    def __post_init__(self) -> None:
        # Validate canonical_verb is non-empty
        if not self.canonical_verb or not self.canonical_verb.strip():
            raise ValueError("canonical_verb must not be empty")

        # Validate canonical_verb is recognized
        if self.canonical_verb not in CANONICAL_VERBS:
            raise ValueError(
                f"canonical_verb {self.canonical_verb!r} is not a recognized "
                f"canonical verb. Valid verbs: {', '.join(sorted(CANONICAL_VERBS))}"
            )

        # Validate confidence_score range
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(
                f"confidence_score must be between 0.0 and 1.0 (inclusive), "
                f"got {self.confidence_score}"
            )

    @property
    def is_confident(self) -> bool:
        """True if the confidence score meets the threshold (>= 0.7).

        When confident, the classification can be used directly without
        falling through to the LLM-powered classifier.
        """
        return self.confidence_score >= _CONFIDENCE_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for wiki YAML or IPC.

        Returns a new dict each call with deep-copied extracted_args
        to preserve immutability.

        Returns:
            Dict with string-serialized enum values and copied args.
        """
        return {
            "canonical_verb": self.canonical_verb,
            "extracted_args": copy.deepcopy(self.extracted_args),
            "confidence_score": self.confidence_score,
            "input_type": self.input_type.value,
        }
