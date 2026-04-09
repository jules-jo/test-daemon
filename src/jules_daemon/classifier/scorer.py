"""Structuredness scorer for input classification.

Computes a normalized score in [0.0, 1.0] indicating how "structured"
a user input string is:

    1.0 = clearly a CLI-style command (verb + flags + SSH target)
    0.0 = clearly natural language (question, conversational text)

The scorer uses lightweight heuristic signals -- no LLM calls, no
network I/O. It is designed to be the first gate in the classification
pipeline, determining whether to delegate to the structured parser
(fast, deterministic) or the NL extractor (heuristic) / LLM classifier.

Signals that increase the score (structured indicators):
    - First token matches a known verb or alias
    - Presence of ``--flag`` patterns
    - Presence of ``user@host`` SSH target patterns
    - Short token count (commands are terse)

Signals that decrease the score (NL indicators):
    - Question marks
    - Question words at start (what, how, is, are, can, could, etc.)
    - Conversational markers (please, I, you, hey, hi)
    - Articles in leading position (the, a, an)
    - High token count relative to structure signals

The score is clamped to [0.0, 1.0] after weighted summation.

Usage::

    from jules_daemon.classifier.scorer import (
        STRUCTURED_THRESHOLD,
        compute_structuredness_score,
    )

    score = compute_structuredness_score("run deploy@host run tests")
    if score >= STRUCTURED_THRESHOLD:
        # Delegate to structured parser
        ...
    else:
        # Delegate to NL extractor
        ...
"""

from __future__ import annotations

import re
import shlex

from jules_daemon.classifier.verb_registry import VERB_ALIASES

__all__ = [
    "STRUCTURED_THRESHOLD",
    "compute_structuredness_score",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRUCTURED_THRESHOLD: float = 0.6
"""Score at or above this threshold indicates structured input."""

# Regex patterns (compiled once at module load)
_SSH_TARGET_PATTERN: re.Pattern[str] = re.compile(
    r"\b[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+(:\d+)?\b"
)
_FLAG_PATTERN: re.Pattern[str] = re.compile(r"(?:^|\s)--?[a-zA-Z]")
_QUESTION_MARK_PATTERN: re.Pattern[str] = re.compile(r"\?")

# Words that strongly signal natural language when at the start
_NL_LEADING_WORDS: frozenset[str] = frozenset({
    "what", "whats", "what's",
    "how", "why", "when", "where", "who",
    "is", "are", "was", "were", "will", "would", "could", "should",
    "can", "do", "does", "did",
    "i", "you", "we", "they", "he", "she", "it",
    "please", "hey", "hi", "hello",
    "the", "a", "an",
    "my", "your", "our",
    "let", "show",
    "any",
})

# Words anywhere in the input that signal conversational NL
_NL_MARKER_WORDS: frozenset[str] = frozenset({
    "please", "thanks", "thank",
    "want", "need", "like", "would",
    "could", "should",
    "going", "ahead",
    "me", "myself",
    "right", "now",
})

# Weight configuration for score signals
_WEIGHT_VERB_MATCH: float = 0.45
_WEIGHT_SSH_TARGET: float = 0.15
_WEIGHT_FLAGS: float = 0.15
_WEIGHT_SHORT_INPUT: float = 0.15
_WEIGHT_NL_LEADING: float = -0.30
_WEIGHT_QUESTION_MARK: float = -0.20
_WEIGHT_NL_MARKERS: float = -0.10
_WEIGHT_HIGH_TOKEN_COUNT: float = -0.10

# Threshold for "high token count" penalty
_HIGH_TOKEN_THRESHOLD: int = 8


# ---------------------------------------------------------------------------
# Tokenization (safe, never raises)
# ---------------------------------------------------------------------------


def _safe_tokenize(raw: str) -> list[str]:
    """Tokenize input, falling back to whitespace split on shlex errors.

    shlex.split raises ValueError for unterminated quotes. We fall back
    to simple whitespace splitting in that case, which is good enough
    for scoring purposes.

    Args:
        raw: Raw user input.

    Returns:
        List of tokens (may be empty).
    """
    stripped = raw.strip()
    if not stripped:
        return []
    try:
        return shlex.split(stripped)
    except ValueError:
        return stripped.split()


# ---------------------------------------------------------------------------
# Signal detectors
# ---------------------------------------------------------------------------


def _has_verb_match(tokens: list[str]) -> bool:
    """Check if the first token matches a known verb or alias."""
    if not tokens:
        return False
    first = tokens[0].strip().lower()
    return first in VERB_ALIASES


def _has_ssh_target(raw: str) -> bool:
    """Check if the input contains a user@host SSH target pattern."""
    return bool(_SSH_TARGET_PATTERN.search(raw))


def _has_flags(raw: str) -> bool:
    """Check if the input contains CLI-style flags (--flag or -f)."""
    return bool(_FLAG_PATTERN.search(raw))


def _has_question_mark(raw: str) -> bool:
    """Check if the input contains a question mark."""
    return bool(_QUESTION_MARK_PATTERN.search(raw))


def _has_nl_leading_word(tokens: list[str]) -> bool:
    """Check if the first token is a natural-language leading word."""
    if not tokens:
        return False
    first = tokens[0].strip().lower().rstrip("'s")
    # Also strip common contractions
    base = first.replace("'", "")
    return first in _NL_LEADING_WORDS or base in _NL_LEADING_WORDS


def _count_nl_markers(tokens: list[str]) -> int:
    """Count the number of NL marker words in the token list."""
    return sum(
        1
        for token in tokens
        if token.strip().lower() in _NL_MARKER_WORDS
    )


def _is_short_input(tokens: list[str]) -> bool:
    """Check if the input is short (typical for structured commands)."""
    return 0 < len(tokens) <= 4


def _is_high_token_count(tokens: list[str]) -> bool:
    """Check if the input has a high token count (typical for NL)."""
    return len(tokens) > _HIGH_TOKEN_THRESHOLD


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


def compute_structuredness_score(raw: str) -> float:
    """Compute a normalized [0.0, 1.0] structuredness score for input.

    Higher scores indicate the input looks like a structured CLI command.
    Lower scores indicate natural-language free-text.

    The score is computed by summing weighted heuristic signals and
    clamping to [0.0, 1.0]. This function never raises exceptions
    and never performs I/O.

    Args:
        raw: Raw user input string.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 for empty/whitespace input.
    """
    stripped = raw.strip()
    if not stripped:
        return 0.0

    tokens = _safe_tokenize(stripped)
    if not tokens:
        return 0.0

    # Accumulate weighted signals
    score: float = 0.0

    # Positive signals (structured indicators)
    if _has_verb_match(tokens):
        score += _WEIGHT_VERB_MATCH

    if _has_ssh_target(stripped):
        score += _WEIGHT_SSH_TARGET

    if _has_flags(stripped):
        score += _WEIGHT_FLAGS

    if _is_short_input(tokens):
        score += _WEIGHT_SHORT_INPUT

    # Negative signals (NL indicators)
    if _has_nl_leading_word(tokens):
        score += _WEIGHT_NL_LEADING

    if _has_question_mark(stripped):
        score += _WEIGHT_QUESTION_MARK

    nl_marker_count = _count_nl_markers(tokens)
    if nl_marker_count > 0:
        # Scale the penalty by number of markers, but cap at 2x
        multiplier = min(nl_marker_count, 2)
        score += _WEIGHT_NL_MARKERS * multiplier

    if _is_high_token_count(tokens):
        score += _WEIGHT_HIGH_TOKEN_COUNT

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, score))
