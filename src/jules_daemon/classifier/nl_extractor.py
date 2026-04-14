"""Heuristic natural-language intent and argument extractor.

Extracts a canonical verb (intent) and arguments from free-form English
input using keyword matching and pattern recognition -- no LLM calls.

This is the fast heuristic path for common NL patterns. When the
heuristic confidence is too low, the caller should fall through to
the LLM-powered IntentClassifier for a more nuanced analysis.

Extraction strategy:
    1. Scan the input for verb-associated keywords (weighted by
       specificity and position)
    2. Extract SSH targets (user@host[:port] patterns)
    3. Extract NL description (remaining text after removing
       structural tokens)
    4. Return the highest-scoring verb with extracted arguments

The extractor never raises exceptions for invalid input -- it always
returns an NLExtraction with a fallback verb and low confidence.

Usage::

    from jules_daemon.classifier.nl_extractor import (
        NLExtraction,
        extract_from_natural_language,
    )

    result = extract_from_natural_language(
        "can you run the smoke tests on staging?"
    )
    assert result.canonical_verb == "run"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "NLExtraction",
    "extract_from_natural_language",
]


# ---------------------------------------------------------------------------
# NLExtraction dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NLExtraction:
    """Immutable result of heuristic NL intent extraction.

    Attributes:
        canonical_verb: The detected canonical verb (one of 6).
        confidence: Confidence in the extraction [0.0, 1.0].
        extracted_args: Extracted arguments (SSH target, NL text, etc.).
        raw_input: The original user input string.
    """

    canonical_verb: str
    confidence: float
    extracted_args: dict[str, Any]
    raw_input: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, "
                f"got {self.confidence}"
            )
        if not self.raw_input and self.raw_input != "":
            raise ValueError("raw_input must not be None")
        # Allow empty string for edge case (empty input)
        if self.raw_input is None:
            raise ValueError("raw_input must not be None")
        if isinstance(self.raw_input, str) and not self.raw_input.strip() and self.confidence > 0.0:
            raise ValueError("raw_input must not be empty for non-zero confidence")


# ---------------------------------------------------------------------------
# Keyword registries
# ---------------------------------------------------------------------------

# Each entry: (keyword, canonical_verb, weight)
# Higher weight = stronger signal. Keywords are checked in order; the
# first match within each verb group contributes to that verb's score.
_VERB_KEYWORDS: tuple[tuple[str, str, float], ...] = (
    # --- run ---
    ("run", "run", 0.6),
    ("execute", "run", 0.6),
    ("exec", "run", 0.5),
    ("start", "run", 0.5),
    ("launch", "run", 0.5),
    ("begin", "run", 0.4),
    ("kick", "run", 0.5),
    ("trigger", "run", 0.5),
    ("fire", "run", 0.4),

    # --- status ---
    ("status", "status", 0.6),
    ("happening", "status", 0.5),
    ("running", "status", 0.5),
    ("progress", "status", 0.5),
    ("going", "status", 0.3),
    ("check", "status", 0.4),
    ("state", "status", 0.4),
    ("info", "status", 0.3),

    # --- cancel ---
    ("stop", "cancel", 0.6),
    ("cancel", "cancel", 0.7),
    ("abort", "cancel", 0.7),
    ("kill", "cancel", 0.7),
    ("terminate", "cancel", 0.6),
    ("halt", "cancel", 0.5),

    # --- watch ---
    ("watch", "watch", 0.6),
    ("tail", "watch", 0.6),
    ("follow", "watch", 0.5),
    ("stream", "watch", 0.5),
    ("monitor", "watch", 0.4),
    ("logs", "watch", 0.6),
    ("log", "watch", 0.4),
    ("output", "watch", 0.5),
    ("attach", "watch", 0.4),

    # --- queue ---
    ("queue", "queue", 0.7),
    ("enqueue", "queue", 0.7),
    ("schedule", "queue", 0.6),
    ("defer", "queue", 0.6),
    ("later", "queue", 0.5),

    # --- history ---
    ("history", "history", 0.7),
    ("past", "history", 0.5),
    ("results", "history", 0.4),
    ("previous", "history", 0.5),
    ("recent", "history", 0.4),
    ("reports", "history", 0.5),
    ("report", "history", 0.4),
    ("last", "history", 0.3),
    ("happened", "history", 0.4),
)

# NL phrases that signal specific verbs (checked before keywords)
_PHRASE_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("kick off", "run", 0.7),
    ("run later", "queue", 0.8),
    ("run .* later", "queue", 0.8),
    ("what's happening", "status", 0.7),
    ("what's running", "status", 0.7),
    ("whats happening", "status", 0.7),
    ("whats running", "status", 0.7),
    ("how are .* going", "status", 0.6),
    ("is .* running", "status", 0.6),
    ("is there .* running", "status", 0.6),
    ("show me .* output", "watch", 0.6),
    ("see the logs", "watch", 0.6),
    ("see the output", "watch", 0.6),
    ("show .* results", "history", 0.5),
    ("show .* reports", "history", 0.5),
    ("past results", "history", 0.6),
    ("previous .* results", "history", 0.6),
    ("last run", "history", 0.7),
    ("recent runs", "history", 0.7),
    ("what happened", "history", 0.7),
)

# SSH target pattern
_SSH_TARGET_RE: re.Pattern[str] = re.compile(
    r"\b([a-zA-Z0-9_.-]+)@([a-zA-Z0-9_.-]+)(?::(\d+))?\b"
)

_SYSTEM_NAME_RE: re.Pattern[str] = re.compile(
    r"\b(?:in|on)\s+system\s+([a-zA-Z0-9_.-]+)\b",
    re.IGNORECASE,
)
_IMPLICIT_SYSTEM_NAME_RE: re.Pattern[str] = re.compile(
    r"\b(?:in|on|at)\s+([a-zA-Z0-9_.-]+)\b",
    re.IGNORECASE,
)

# Default verb when nothing matches
_DEFAULT_VERB: str = "status"
_DEFAULT_CONFIDENCE: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _score_phrases(text_lower: str) -> dict[str, float]:
    """Score verbs based on phrase pattern matches.

    Returns a dict mapping canonical verb -> highest phrase score.
    """
    scores: dict[str, float] = {}
    for pattern, verb, weight in _PHRASE_PATTERNS:
        if re.search(pattern, text_lower):
            scores[verb] = max(scores.get(verb, 0.0), weight)
    return scores


def _score_keywords(text_lower: str) -> dict[str, float]:
    """Score verbs based on keyword matches in the text.

    Each keyword contributes its weight to the corresponding verb.
    Only the highest-weight match per verb is counted to avoid
    double-counting synonyms.

    Returns a dict mapping canonical verb -> highest keyword score.
    """
    scores: dict[str, float] = {}
    for keyword, verb, weight in _VERB_KEYWORDS:
        # Word boundary match to avoid partial matches
        if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
            scores[verb] = max(scores.get(verb, 0.0), weight)
    return scores


def _extract_ssh_target(
    raw: str,
) -> dict[str, Any]:
    """Extract SSH target (user@host[:port]) from input.

    Returns a dict with target_user, target_host, and optionally
    target_port. Returns empty dict if no SSH target found.
    """
    match = _SSH_TARGET_RE.search(raw)
    if match is None:
        return {}

    result: dict[str, Any] = {
        "target_user": match.group(1),
        "target_host": match.group(2),
    }

    port_str = match.group(3)
    if port_str is not None:
        try:
            result["target_port"] = int(port_str)
        except ValueError:
            pass

    return result


def _extract_system_name(raw: str) -> dict[str, Any]:
    """Extract ``system_name`` from phrases like ``in system tuto``."""
    match = _SYSTEM_NAME_RE.search(raw)
    if match is None:
        return {}
    return {"system_name": match.group(1)}


def _extract_infer_target_hint(raw: str) -> dict[str, Any]:
    """Mark NL requests that likely contain a named-system reference.

    This does not guess the system locally. It simply tells the daemon to
    try resolving a system mention like ``in tuto`` against its live wiki.
    The actual resolution remains daemon-side against known system aliases.
    """
    if _SYSTEM_NAME_RE.search(raw):
        return {}
    match = _IMPLICIT_SYSTEM_NAME_RE.search(raw)
    if match is None:
        return {}
    return {"infer_target": True}


def _select_best_verb(
    phrase_scores: dict[str, float],
    keyword_scores: dict[str, float],
) -> tuple[str, float]:
    """Select the best verb from combined phrase and keyword scores.

    Phrase scores take priority (they are more specific). When phrases
    and keywords both match, use the maximum score.

    Returns:
        Tuple of (canonical_verb, confidence).
    """
    # Merge scores: phrase matches get a bonus because they are more
    # specific (multi-word patterns) than single-keyword matches.
    _PHRASE_BONUS: float = 0.1
    combined: dict[str, float] = dict(keyword_scores)
    for verb, score in phrase_scores.items():
        boosted = score + _PHRASE_BONUS
        combined[verb] = max(combined.get(verb, 0.0), boosted)

    if not combined:
        return _DEFAULT_VERB, _DEFAULT_CONFIDENCE

    # Select verb with highest score
    best_verb = max(combined, key=lambda v: combined[v])
    best_score = combined[best_verb]

    return best_verb, best_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_from_natural_language(raw: str) -> NLExtraction:
    """Extract intent (canonical verb) and arguments from NL input.

    Heuristic extraction pipeline:
    1. Check phrase patterns (most specific)
    2. Check keyword matches (broader)
    3. Extract SSH target if present
    4. Return best-scoring verb with extracted arguments

    This function never raises for invalid input. Empty or gibberish
    input returns a default ``status`` verb with zero confidence.

    Args:
        raw: Raw user input string.

    Returns:
        Immutable NLExtraction with canonical verb, confidence, and
        extracted arguments.
    """
    stripped = raw.strip()
    if not stripped:
        return NLExtraction(
            canonical_verb=_DEFAULT_VERB,
            confidence=_DEFAULT_CONFIDENCE,
            extracted_args={},
            raw_input=raw,
        )

    text_lower = stripped.lower()

    # Score via phrases (highest specificity)
    phrase_scores = _score_phrases(text_lower)

    # Score via keywords
    keyword_scores = _score_keywords(text_lower)

    # Select best verb
    best_verb, confidence = _select_best_verb(phrase_scores, keyword_scores)

    # Extract SSH target or named system reference
    ssh_args = _extract_ssh_target(stripped)
    system_args = _extract_system_name(stripped)
    infer_args = _extract_infer_target_hint(stripped)

    # Build extracted_args. For run/queue intents we keep the original
    # free-form text so downstream builders can reuse it as the NL command.
    extracted_args: dict[str, Any] = dict(ssh_args)
    extracted_args.update(system_args)
    extracted_args.update(infer_args)
    if best_verb in {"run", "queue"}:
        extracted_args["natural_language"] = stripped

    return NLExtraction(
        canonical_verb=best_verb,
        confidence=confidence,
        extracted_args=extracted_args,
        raw_input=stripped,
    )
