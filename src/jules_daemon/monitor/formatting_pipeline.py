"""Output formatting pipeline for SSH output chunks.

Processes raw output chunks from remote SSH sessions by:
1. Preserving, normalizing, or stripping ANSI color codes
2. Prepending configurable timestamps to each line

The pipeline is composed of pure functions that transform text,
plus a top-level ``format_chunk`` that chains them according to
a ``FormatterConfig``. All data types are immutable.

ANSI handling modes:
- PRESERVE: Keep all ANSI escape sequences as-is
- STRIP: Remove all ANSI escape sequences
- NORMALIZE: Keep SGR (Select Graphic Rendition) codes for colors and
  text styling, but strip non-visual sequences like cursor movement,
  erase-line, and OSC (Operating System Command) sequences

Usage::

    from jules_daemon.monitor.formatting_pipeline import (
        FormatterConfig,
        AnsiMode,
        format_chunk,
    )

    config = FormatterConfig(
        ansi_mode=AnsiMode.NORMALIZE,
        timestamp_format="%H:%M:%S",
        timestamp_enabled=True,
    )
    result = format_chunk(raw_output, config=config)
    print(result.formatted)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

__all__ = [
    "AnsiMode",
    "FormattedChunk",
    "FormatterConfig",
    "format_chunk",
    "normalize_ansi",
    "prepend_timestamps",
    "strip_ansi",
]


# ---------------------------------------------------------------------------
# ANSI regex patterns
# ---------------------------------------------------------------------------

# Matches all ANSI escape sequences (CSI, OSC, and simple ESC codes).
# CSI: ESC [ ... final_byte
# OSC: ESC ] ... ST (where ST is BEL or ESC \)
# Simple: ESC followed by a single character
_ANSI_ALL_RE = re.compile(
    r"\033"           # ESC character
    r"(?:"
    r"\[[0-9;]*[A-Za-z]"   # CSI: ESC [ params final_byte
    r"|"
    r"\][^\007\033]*(?:\007|\033\\)"  # OSC: ESC ] ... BEL or ESC \
    r"|"
    r"[^[\]]"              # Simple ESC + single char
    r")"
)

# Matches only SGR (Select Graphic Rendition) sequences.
# SGR is CSI with 'm' as the final byte: ESC [ params m
# This covers colors, bold, dim, underline, inverse, etc.
_ANSI_SGR_RE = re.compile(r"\033\[[0-9;]*m")

# Matches non-SGR CSI sequences and OSC sequences -- the "noise" to strip
# in normalize mode. We strip everything matched by _ANSI_ALL_RE that is
# NOT matched by _ANSI_SGR_RE.
_ANSI_NON_SGR_RE = re.compile(
    r"\033"
    r"(?:"
    r"\[[0-9;]*[A-La-lN-Zn-z]"  # CSI with non-'m' final byte (excludes 'M'/'m')
    r"|"
    r"\][^\007\033]*(?:\007|\033\\)"  # OSC sequences
    r"|"
    r"[^[\]]"                         # Simple ESC + single char
    r")"
)


# ---------------------------------------------------------------------------
# ANSI modes
# ---------------------------------------------------------------------------


class AnsiMode(Enum):
    """Strategy for handling ANSI escape codes in output.

    PRESERVE:  Keep all ANSI sequences unchanged.
    STRIP:     Remove all ANSI sequences.
    NORMALIZE: Keep SGR (color/style) codes, strip everything else
               (cursor movement, erase, OSC, etc.).
    """

    PRESERVE = "preserve"
    STRIP = "strip"
    NORMALIZE = "normalize"


# ---------------------------------------------------------------------------
# Pure ANSI functions
# ---------------------------------------------------------------------------


def strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from text.

    Handles CSI sequences (colors, cursor movement), OSC sequences
    (window titles, hyperlinks), and simple ESC codes.

    Args:
        text: Raw text potentially containing ANSI codes.

    Returns:
        Text with all ANSI escape sequences removed.
    """
    if not text:
        return text
    return _ANSI_ALL_RE.sub("", text)


def normalize_ansi(text: str) -> str:
    """Keep SGR color/style codes but strip non-visual ANSI sequences.

    Preserves:
    - Standard foreground/background colors (e.g., ESC[31m for red)
    - 8-bit colors (e.g., ESC[38;5;196m)
    - 24-bit/truecolor (e.g., ESC[38;2;255;100;0m)
    - Text attributes (bold, dim, underline, inverse, reset)

    Strips:
    - Cursor movement (ESC[2A, ESC[3B, etc.)
    - Erase sequences (ESC[K, ESC[2J, etc.)
    - OSC sequences (ESC]0;title BEL, etc.)

    Args:
        text: Raw text potentially containing ANSI codes.

    Returns:
        Text with only SGR sequences preserved.
    """
    if not text:
        return text
    return _ANSI_NON_SGR_RE.sub("", text)


# ---------------------------------------------------------------------------
# Timestamp prepending
# ---------------------------------------------------------------------------


def _default_clock() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def prepend_timestamps(
    text: str,
    *,
    timestamp_format: str = "%H:%M:%S",
    clock: Callable[[], datetime] | None = None,
) -> str:
    """Prepend a bracketed timestamp to each line of text.

    Each line receives a ``[timestamp] `` prefix. Empty trailing lines
    (from a trailing newline) are preserved without a timestamp prefix
    to maintain the original line-ending behavior.

    Args:
        text: Multi-line text to process.
        timestamp_format: strftime format string for the timestamp.
        clock: Optional callable returning a datetime. Defaults to UTC now.
            Useful for testing with deterministic timestamps.

    Returns:
        Text with timestamps prepended to each non-trailing-empty line.
    """
    if not text:
        return text

    now = (clock or _default_clock)()
    stamp = f"[{now.strftime(timestamp_format)}]"

    # Split preserving the trailing newline structure
    has_trailing_newline = text.endswith("\n")
    lines = text.split("\n")

    # If there's a trailing newline, the last element is an empty string
    # that represents the newline itself -- don't timestamp it
    if has_trailing_newline and lines and lines[-1] == "":
        content_lines = lines[:-1]
        stamped = [f"{stamp} {line}" for line in content_lines]
        stamped.append("")  # restore trailing newline element
    else:
        stamped = [f"{stamp} {line}" for line in lines]

    return "\n".join(stamped)


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormatterConfig:
    """Immutable configuration for the output formatting pipeline.

    Attributes:
        ansi_mode:        How to handle ANSI escape codes.
        timestamp_format: strftime format string for line timestamps.
        timestamp_enabled: Whether to prepend timestamps to output lines.
    """

    ansi_mode: AnsiMode = AnsiMode.PRESERVE
    timestamp_format: str = "%H:%M:%S"
    timestamp_enabled: bool = True


@dataclass(frozen=True)
class FormattedChunk:
    """Result of processing a raw output chunk through the pipeline.

    Attributes:
        raw:           The original unprocessed text.
        formatted:     The processed text after ANSI handling and timestamps.
        line_count:    Number of content lines in the chunk.
        ansi_stripped: Whether ANSI codes were fully removed.
    """

    raw: str
    formatted: str
    line_count: int
    ansi_stripped: bool

    def __post_init__(self) -> None:
        if self.line_count < 0:
            raise ValueError("line_count must not be negative")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def _count_content_lines(text: str) -> int:
    """Count the number of content lines in text.

    A trailing newline does not add an extra line. An empty string
    has zero lines.
    """
    if not text:
        return 0
    # Normalize \r\n to \n for counting
    normalized = text.replace("\r\n", "\n")
    if normalized.endswith("\n"):
        normalized = normalized[:-1]
    if not normalized:
        return 0
    return normalized.count("\n") + 1


def format_chunk(
    raw: str,
    *,
    config: FormatterConfig | None = None,
    clock: Callable[[], datetime] | None = None,
) -> FormattedChunk:
    """Process a raw output chunk through the formatting pipeline.

    Pipeline stages (in order):
    1. Normalize line endings (\\r\\n -> \\n)
    2. Apply ANSI handling (preserve / strip / normalize)
    3. Prepend timestamps (if enabled)

    Args:
        raw: Raw output text from SSH session.
        config: Formatting configuration. Defaults to FormatterConfig().
        clock: Optional clock function for timestamps. Defaults to UTC now.

    Returns:
        FormattedChunk with the processed text and metadata.
    """
    if config is None:
        config = FormatterConfig()

    if not raw:
        return FormattedChunk(
            raw=raw,
            formatted="",
            line_count=0,
            ansi_stripped=False,
        )

    # Stage 1: Normalize line endings
    text = raw.replace("\r\n", "\n")

    # Stage 2: ANSI handling
    ansi_stripped = False
    if config.ansi_mode == AnsiMode.STRIP:
        text = strip_ansi(text)
        ansi_stripped = True
    elif config.ansi_mode == AnsiMode.NORMALIZE:
        text = normalize_ansi(text)
    # PRESERVE: no-op

    # Stage 3: Timestamps
    if config.timestamp_enabled:
        text = prepend_timestamps(
            text,
            timestamp_format=config.timestamp_format,
            clock=clock,
        )

    line_count = _count_content_lines(raw)

    return FormattedChunk(
        raw=raw,
        formatted=text,
        line_count=line_count,
        ansi_stripped=ansi_stripped,
    )
