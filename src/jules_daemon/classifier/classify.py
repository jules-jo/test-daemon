"""Unified input classification entry point.

Provides a single ``classify()`` function that:
1. Computes a structuredness score for the raw input
2. If the score meets the structured threshold, delegates to the
   deterministic structured parser (``classify_structured_command``)
3. Otherwise, delegates to the heuristic NL extractor
4. Always returns a ``ClassificationResult``

This is the primary entry point for the CLI-to-daemon classification
pipeline. It resolves the structured-vs-NL question without any LLM
calls. When the returned ``ClassificationResult.is_confident`` is
False, the caller should optionally fall through to the LLM-powered
IntentClassifier for a more nuanced analysis.

Usage::

    from jules_daemon.classifier.classify import classify

    result = classify("run deploy@staging run the tests")
    # result.input_type == InputType.COMMAND
    # result.canonical_verb == "run"
    # result.is_confident == True

    result = classify("can you run the smoke tests on staging?")
    # result.input_type == InputType.NATURAL_LANGUAGE
    # result.canonical_verb == "run"
"""

from __future__ import annotations

import logging
from typing import Any

from jules_daemon.classifier.models import ClassificationResult, InputType
from jules_daemon.classifier.nl_extractor import extract_from_natural_language
from jules_daemon.classifier.scorer import (
    STRUCTURED_THRESHOLD,
    compute_structuredness_score,
)

logger = logging.getLogger(__name__)

__all__ = [
    "classify",
]


def classify(raw: str) -> ClassificationResult:
    """Classify raw user input as structured or natural language.

    Pipeline:
    1. Compute structuredness score
    2. If score >= threshold, try structured parser
    3. If structured parser succeeds, return its result
    4. Otherwise, use NL extractor and return its result
    5. Never returns None -- always a valid ClassificationResult

    The function never raises exceptions for any input. Empty,
    whitespace, or gibberish input produces an AMBIGUOUS result
    with zero confidence.

    Args:
        raw: Raw user input string.

    Returns:
        Immutable ClassificationResult with canonical verb, extracted
        arguments, confidence score, and input type.
    """
    stripped = raw.strip()

    # Edge case: empty/whitespace input
    if not stripped:
        logger.debug("Empty input -> AMBIGUOUS with zero confidence")
        return ClassificationResult(
            canonical_verb="status",
            extracted_args={},
            confidence_score=0.0,
            input_type=InputType.AMBIGUOUS,
        )

    # Step 1: Compute structuredness score
    score = compute_structuredness_score(stripped)
    logger.debug(
        "Structuredness score for %.80s: %.3f (threshold=%.3f)",
        stripped,
        score,
        STRUCTURED_THRESHOLD,
    )

    # Step 2: Try structured path if score meets threshold
    if score >= STRUCTURED_THRESHOLD:
        structured_result = _try_structured_path(stripped)
        if structured_result is not None:
            logger.debug(
                "Structured classification: verb=%s, confidence=%.2f",
                structured_result.canonical_verb,
                structured_result.confidence_score,
            )
            return structured_result

    # Step 3: Fall through to NL extraction
    nl_result = _extract_nl_path(stripped)
    logger.debug(
        "NL classification: verb=%s, confidence=%.2f",
        nl_result.canonical_verb,
        nl_result.confidence_score,
    )
    return nl_result


def _try_structured_path(stripped: str) -> ClassificationResult | None:
    """Attempt deterministic structured classification.

    Delegates to ``classify_structured_command()`` from the CLI parser.
    Returns None if the input cannot be classified structurally (e.g.,
    unrecognized verb), in which case the caller should fall through
    to NL extraction.

    Uses a deferred import to break the circular dependency between
    the ``classifier`` and ``cli`` packages.

    Args:
        stripped: Whitespace-stripped input string.

    Returns:
        ClassificationResult on success, None on failure.
    """
    # Deferred import to break cli <-> classifier circular dependency.
    # cli.parser imports classifier.models; classifier.classify imports
    # cli.parser. By deferring here, both packages can finish
    # initialization before this function is called at runtime.
    from jules_daemon.cli.parser import classify_structured_command  # noqa: PLC0415

    return classify_structured_command(stripped)


def _extract_nl_path(stripped: str) -> ClassificationResult:
    """Extract intent via heuristic NL analysis.

    Delegates to the NL extractor and wraps the result in a
    ClassificationResult with the appropriate InputType.

    Args:
        stripped: Whitespace-stripped input string.

    Returns:
        ClassificationResult with NATURAL_LANGUAGE or AMBIGUOUS type.
    """
    extraction = extract_from_natural_language(stripped)

    # Determine input type based on extraction confidence
    input_type = _determine_nl_input_type(extraction.confidence)

    # Build the extracted args dict
    extracted_args: dict[str, Any] = dict(extraction.extracted_args)

    return ClassificationResult(
        canonical_verb=extraction.canonical_verb,
        extracted_args=extracted_args,
        confidence_score=extraction.confidence,
        input_type=input_type,
    )


def _determine_nl_input_type(confidence: float) -> InputType:
    """Map NL extraction confidence to an InputType.

    Args:
        confidence: NL extraction confidence [0.0, 1.0].

    Returns:
        NATURAL_LANGUAGE for confident extractions,
        AMBIGUOUS for low-confidence or zero-confidence.
    """
    if confidence <= 0.0:
        return InputType.AMBIGUOUS
    return InputType.NATURAL_LANGUAGE
