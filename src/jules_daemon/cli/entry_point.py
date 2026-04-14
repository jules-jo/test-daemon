"""Unified CLI entry point: classify, resolve, build, dispatch.

The single function that raw user input enters and a structured
``DispatchResponse`` exits. Regardless of whether the user typed a
structured CLI command, a verb alias, or free-form natural language,
the pipeline resolves to the same canonical verb and produces the same
typed ``*Args`` dataclass for the handler.

Pipeline:
    1. **Classify** -- pass raw input through the deterministic classifier
       (structuredness scorer -> structured parser or NL extractor)
    2. **Resolve** -- map the ``ClassificationResult.canonical_verb`` to
       a ``Verb`` enum via the handler registry
    3. **Build** -- construct the verb-specific ``*Args`` dataclass from
       either the structured parser output (high-fidelity) or the
       ``build_verb_args`` bridge (NL path)
    4. **Dispatch** -- route the ``ParsedCommand`` through the
       ``CommandDispatcher`` to the registered handler

The entry point never raises exceptions. All errors are captured in the
``InputProcessingResult`` return value. This makes it safe for both
IPC handlers and direct CLI invocation.

Usage::

    from jules_daemon.cli.entry_point import process_input

    result = await process_input(
        "run deploy@staging run the smoke tests",
        registry=registry,
        dispatcher=dispatcher,
    )
    if result.success:
        print(result.dispatch_response.payload)
    else:
        print(f"Error: {result.error}")

    # Natural language works identically:
    result = await process_input(
        "execute deploy@staging run the smoke tests",
        registry=registry,
        dispatcher=dispatcher,
    )
    # Same handler, same args, same result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from jules_daemon.classifier.classify import classify
from jules_daemon.classifier.models import ClassificationResult, InputType
from jules_daemon.classifier.nl_extractor import extract_from_natural_language
from jules_daemon.cli.args_builder import build_verb_args
from jules_daemon.cli.dispatcher import CommandDispatcher, DispatchResponse
from jules_daemon.cli.parser import ParseError, parse_command
from jules_daemon.cli.registry import CommandHandlerRegistry
from jules_daemon.cli.verbs import ParsedCommand, Verb

__all__ = [
    "InputProcessingResult",
    "process_input",
]

logger = logging.getLogger(__name__)

# Lookup table: canonical verb string -> Verb enum
_CANONICAL_TO_VERB: dict[str, Verb] = {v.value: v for v in Verb}


# ---------------------------------------------------------------------------
# InputProcessingResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputProcessingResult:
    """Immutable result of processing raw input through the unified pipeline.

    Captures the full trace from classification through dispatch so
    callers can inspect any stage for diagnostics, logging, or error
    reporting.

    Attributes:
        success: True if the input was classified, built, and dispatched
            successfully. False if any stage failed.
        dispatch_response: The ``DispatchResponse`` from the handler
            dispatch. None if the pipeline did not reach dispatch
            (classification or build failure).
        classification: The ``ClassificationResult`` from the input
            classifier. None only for empty/invalid input that could
            not be classified at all.
        parsed_command: The ``ParsedCommand`` that was dispatched. None
            if the pipeline did not reach the build stage.
        error: Human-readable error description when ``success`` is
            False. None when ``success`` is True.
        input_style: How the input was classified (COMMAND, NL, etc.).
    """

    success: bool
    dispatch_response: DispatchResponse | None
    classification: ClassificationResult | None
    parsed_command: ParsedCommand | None
    error: str | None
    input_style: InputType

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for IPC or wiki persistence.

        Returns a new dict each call to preserve immutability.

        Returns:
            Dict with string-serialized enum values and nested dicts.
        """
        result: dict[str, Any] = {
            "success": self.success,
            "input_style": self.input_style.value,
            "error": self.error,
        }

        if self.classification is not None:
            result["classification"] = self.classification.to_dict()
        else:
            result["classification"] = None

        if self.dispatch_response is not None:
            result["dispatch_response"] = {
                "success": self.dispatch_response.success,
                "verb": self.dispatch_response.verb.value,
                "payload": self.dispatch_response.payload,
                "error": self.dispatch_response.error,
            }
        else:
            result["dispatch_response"] = None

        if self.parsed_command is not None:
            result["parsed_command"] = {
                "verb": self.parsed_command.verb.value,
            }
        else:
            result["parsed_command"] = None

        return result


# ---------------------------------------------------------------------------
# Pipeline step: classify
# ---------------------------------------------------------------------------


def _classify_input(raw: str) -> ClassificationResult:
    """Classify raw input via the deterministic classifier.

    Wraps ``classify()`` with error handling and logging. Returns a
    safe AMBIGUOUS fallback if the classifier raises unexpectedly.

    Args:
        raw: Raw user input string.

    Returns:
        ClassificationResult (always non-None, never raises).
    """
    try:
        result = classify(raw)
    except Exception:
        logger.exception("Unexpected error in classify(%r), returning AMBIGUOUS", raw)
        return ClassificationResult(
            canonical_verb="status",
            extracted_args={},
            confidence_score=0.0,
            input_type=InputType.AMBIGUOUS,
        )
    logger.debug(
        "Classification: verb=%s, type=%s, confidence=%.2f",
        result.canonical_verb,
        result.input_type.value,
        result.confidence_score,
    )
    return result


# ---------------------------------------------------------------------------
# Pipeline step: resolve verb and build ParsedCommand
# ---------------------------------------------------------------------------


def _resolve_and_build(
    raw: str,
    classification: ClassificationResult,
    registry: CommandHandlerRegistry,
) -> ParsedCommand | str:
    """Resolve canonical verb and build a ParsedCommand.

    Strategy:
    1. For COMMAND input (structured path): use ``parse_command()`` which
       handles tokenization, verb normalization, and per-verb argument
       parsing with full fidelity. This is the fast, high-confidence path.
    2. For NATURAL_LANGUAGE / AMBIGUOUS input: use the classification's
       ``extracted_args`` dict with ``build_verb_args()`` to construct
       the typed ``*Args`` dataclass.

    In both cases, the handler receives an identical ``*Args`` dataclass.

    Args:
        raw: The original raw input string.
        classification: The classification result from the classifier.
        registry: The handler registry for verb validation.

    Returns:
        ParsedCommand on success, or an error string on failure.
    """
    canonical_verb = classification.canonical_verb

    # Validate that the verb has a registered handler
    verb = _CANONICAL_TO_VERB.get(canonical_verb)
    if verb is None:
        return f"Unknown canonical verb: {canonical_verb!r}"

    if not registry.has_handler(verb):
        logger.warning("No handler registered for verb %r", canonical_verb)
        # Allow dispatch to proceed -- the dispatcher will return a
        # meaningful error. This separates classification from registration.

    # Strategy selection: structured vs NL path
    if classification.input_type == InputType.COMMAND and classification.is_confident:
        return _build_from_structured_path(raw, verb)

    return _build_from_classification(classification, verb)


def _build_from_structured_path(
    raw: str,
    expected_verb: Verb,
) -> ParsedCommand | str:
    """Build ParsedCommand via the full structured parser.

    The structured parser handles tokenization, flag parsing, SSH target
    extraction, and per-verb validation with maximum fidelity.

    Args:
        raw: Original raw input string.
        expected_verb: The verb the classifier identified (for cross-check).

    Returns:
        ParsedCommand on success, or error string on failure.
    """
    result = parse_command(raw)

    if isinstance(result, ParseError):
        # Structured parse failed. Fall back to classification-based build.
        logger.debug(
            "Structured parse failed for %r: %s. "
            "Falling back to classification-based build.",
            raw,
            result.message,
        )
        return result.message

    # Cross-check: the parser's verb should match the classifier's verb.
    # If they disagree, trust the parser (higher fidelity).
    if result.verb != expected_verb:
        logger.debug(
            "Verb cross-check: classifier=%s, parser=%s. Using parser.",
            expected_verb.value,
            result.verb.value,
        )

    return result


def _build_from_classification(
    classification: ClassificationResult,
    verb: Verb,
) -> ParsedCommand | str:
    """Build ParsedCommand from classification extracted_args.

    Uses ``build_verb_args()`` to convert the untyped dict into a
    typed ``*Args`` dataclass, then wraps it in a ``ParsedCommand``.

    Args:
        classification: The classification result with extracted args.
        verb: The resolved Verb enum.

    Returns:
        ParsedCommand on success, or error string on failure.
    """
    args_result = build_verb_args(
        classification.canonical_verb,
        classification.extracted_args,
    )

    if isinstance(args_result, str):
        return args_result

    try:
        return ParsedCommand(verb=verb, args=args_result)
    except ValueError as exc:
        return str(exc)


def _reclassify_as_natural_language(raw: str) -> ClassificationResult:
    """Force the NL extractor path for structured-looking conversational input.

    This is used when the structured parser recognizes a verb token but the
    arguments do not actually fit the structured grammar, for example:

        run the smoke tests on deploy@staging
    """
    extraction = extract_from_natural_language(raw)
    input_type = (
        InputType.NATURAL_LANGUAGE
        if extraction.confidence > 0.0
        else InputType.AMBIGUOUS
    )
    return ClassificationResult(
        canonical_verb=extraction.canonical_verb,
        extracted_args=dict(extraction.extracted_args),
        confidence_score=extraction.confidence,
        input_type=input_type,
    )


# ---------------------------------------------------------------------------
# Pipeline step: dispatch
# ---------------------------------------------------------------------------


async def _dispatch_command(
    command: ParsedCommand,
    dispatcher: CommandDispatcher,
) -> DispatchResponse:
    """Dispatch a ParsedCommand to its handler via the dispatcher.

    Wraps ``dispatcher.dispatch()`` with logging. Never raises.

    Args:
        command: The validated ParsedCommand.
        dispatcher: The command dispatcher with registered handlers.

    Returns:
        DispatchResponse from the handler.
    """
    logger.debug("Dispatching: verb=%s", command.verb.value)
    response = await dispatcher.dispatch(command)
    logger.debug(
        "Dispatch result: verb=%s, success=%s",
        response.verb.value,
        response.success,
    )
    return response


# ---------------------------------------------------------------------------
# Public API: unified entry point
# ---------------------------------------------------------------------------


async def process_input(
    raw: str,
    *,
    registry: CommandHandlerRegistry,
    dispatcher: CommandDispatcher,
) -> InputProcessingResult:
    """Process raw user input through the full classify-resolve-dispatch pipeline.

    This is the unified CLI entry point. It accepts any form of user
    input (structured CLI command, verb alias, or natural language),
    classifies it, resolves the canonical verb, builds typed arguments,
    and dispatches to the registered handler.

    The function never raises exceptions. All errors at any stage are
    captured in the returned ``InputProcessingResult``.

    Pipeline:
        1. Classify the raw input (structured vs NL)
        2. Resolve the canonical verb and build ``ParsedCommand``
        3. Dispatch to the handler via ``CommandDispatcher``

    Args:
        raw: Raw user input string. Can be:
            - Structured: ``"run deploy@host run tests"``
            - Alias: ``"execute deploy@host run tests"``
            - Natural language: ``"can you run the smoke tests?"``
        registry: The command handler registry for verb validation
            and handler lookup metadata.
        dispatcher: The command dispatcher with registered handlers.

    Returns:
        Immutable ``InputProcessingResult`` with the full pipeline trace.
    """
    # Step 1: Classify
    classification = _classify_input(raw)

    # Early exit for empty/ambiguous input with zero confidence
    if (
        classification.input_type == InputType.AMBIGUOUS
        and classification.confidence_score <= 0.0
    ):
        return InputProcessingResult(
            success=False,
            dispatch_response=None,
            classification=classification,
            parsed_command=None,
            error="Could not classify input: empty or unrecognizable",
            input_style=classification.input_type,
        )

    # Step 2: Resolve verb and build ParsedCommand
    build_result = _resolve_and_build(raw, classification, registry)

    if isinstance(build_result, str):
        # Build failed. If this was a structured path failure, try NL fallback.
        if classification.input_type == InputType.COMMAND:
            structured_error = build_result
            logger.debug(
                "Structured parse failed (%s), attempting NL reclassification",
                structured_error,
            )
            nl_classification = _reclassify_as_natural_language(raw)
            verb = _CANONICAL_TO_VERB.get(nl_classification.canonical_verb)
            if verb is not None:
                nl_result = _build_from_classification(nl_classification, verb)
                if not isinstance(nl_result, str):
                    classification = nl_classification
                    build_result = nl_result

    if isinstance(build_result, str):
        return InputProcessingResult(
            success=False,
            dispatch_response=None,
            classification=classification,
            parsed_command=None,
            error=build_result,
            input_style=classification.input_type,
        )

    parsed_command = build_result

    # Step 3: Dispatch
    dispatch_response = await _dispatch_command(parsed_command, dispatcher)

    return InputProcessingResult(
        success=dispatch_response.success,
        dispatch_response=dispatch_response,
        classification=classification,
        parsed_command=parsed_command,
        error=dispatch_response.error,
        input_style=classification.input_type,
    )
