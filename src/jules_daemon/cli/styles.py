"""Terminal styling utilities for consistent CLI output.

Provides a centralized color palette, text-based icon constants, and
layout helper functions used by all event renderers in the agent loop
notification system.

Design decisions:

- **Color palette**: Standard ANSI SGR codes only (no 8-bit or truecolor)
  for maximum terminal compatibility across Linux and Windows.
- **Icon constants**: Plain ASCII characters (no Unicode emoji) per project
  coding style. Each icon is a short, distinct string.
- **Layout helpers**: Pure functions for indentation, padding, truncation,
  horizontal rules, and box drawing. All functions are stateless and
  return new strings (no mutation).
- **StyleConfig**: Frozen dataclass controlling color enable/disable and
  terminal width. Passed through ``RenderContext`` to all renderers.

Usage::

    from jules_daemon.cli.styles import (
        Color,
        ICON_RUNNING,
        StyleConfig,
        indent,
        styled,
    )

    config = StyleConfig(color_enabled=True, terminal_width=80)
    text = styled("PASSED", Color.GREEN, Color.BOLD, config=config)
    print(indent(text, level=1))
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Color",
    "ICON_APPROVE",
    "ICON_ARROW",
    "ICON_BULLET",
    "ICON_CANCEL",
    "ICON_DONE",
    "ICON_ERROR",
    "ICON_INFO",
    "ICON_PENDING",
    "ICON_REJECT",
    "ICON_RUNNING",
    "ICON_TOOL",
    "ICON_WARNING",
    "StyleConfig",
    "box_bottom",
    "box_line",
    "box_top",
    "horizontal_rule",
    "indent",
    "pad_right",
    "styled",
    "truncate",
]


# ---------------------------------------------------------------------------
# Color palette (ANSI SGR codes)
# ---------------------------------------------------------------------------


class Color(Enum):
    """ANSI SGR color codes for terminal output.

    Uses standard 16-color codes for maximum terminal compatibility.
    All values are complete SGR sequences (ESC[ ... m) ready for
    direct string concatenation.

    Attributes -- modifiers:
        RESET:  Reset all attributes to default.
        BOLD:   Bold / increased intensity.
        DIM:    Dim / decreased intensity.

    Attributes -- standard foreground:
        RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE

    Attributes -- bright/high-intensity foreground:
        BRIGHT_RED, BRIGHT_GREEN, BRIGHT_YELLOW, BRIGHT_CYAN
    """

    # Modifiers
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Standard foreground
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright / high-intensity foreground
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"


# ---------------------------------------------------------------------------
# Icon constants (ASCII only, no emoji)
# ---------------------------------------------------------------------------

ICON_RUNNING: str = "[~]"
"""Displayed next to actively running operations."""

ICON_DONE: str = "[+]"
"""Displayed next to successfully completed operations."""

ICON_ERROR: str = "[!]"
"""Displayed next to failed operations."""

ICON_WARNING: str = "[*]"
"""Displayed next to warnings or caution items."""

ICON_INFO: str = "[i]"
"""Displayed next to informational messages."""

ICON_PENDING: str = "[.]"
"""Displayed next to queued or waiting operations."""

ICON_APPROVE: str = "[Y]"
"""Displayed next to approved actions."""

ICON_REJECT: str = "[N]"
"""Displayed next to rejected/denied actions."""

ICON_TOOL: str = "[T]"
"""Displayed next to tool invocation events."""

ICON_ARROW: str = "-->"
"""Arrow indicator for flow/transition lines."""

ICON_BULLET: str = " - "
"""Bullet point prefix for list items."""

ICON_CANCEL: str = "[X]"
"""Displayed next to cancelled operations."""


# ---------------------------------------------------------------------------
# StyleConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StyleConfig:
    """Immutable configuration for terminal styling behavior.

    Controls whether ANSI color codes are emitted and the assumed
    terminal width for layout calculations.

    Attributes:
        color_enabled:  When False, all ``styled()`` calls return plain
                        text without ANSI codes. Default True.
        terminal_width: Assumed character width of the terminal for
                        layout calculations. Default 80.
        indent_width:   Number of spaces per indent level. Default 2.
    """

    color_enabled: bool = True
    terminal_width: int = 80
    indent_width: int = 2


# ---------------------------------------------------------------------------
# Text composition
# ---------------------------------------------------------------------------


def styled(text: str, *colors: Color, config: StyleConfig | None = None) -> str:
    """Wrap text with ANSI color codes.

    Applies the given color/modifier codes before the text and appends
    a RESET code after it. When no colors are provided or when color
    is disabled via config, returns the text unchanged.

    Args:
        text:    The text to style.
        *colors: One or more Color enum values to apply.
        config:  Optional StyleConfig. When ``color_enabled`` is False,
                 ANSI codes are suppressed.

    Returns:
        Styled text string with ANSI codes, or plain text if color
        is disabled or no colors are provided.
    """
    if not colors:
        return text

    if config is not None and not config.color_enabled:
        return text

    prefix = "".join(c.value for c in colors)
    return f"{prefix}{text}{Color.RESET.value}"


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def indent(text: str, *, level: int = 1, width: int = 2) -> str:
    """Indent text by a given number of levels.

    Each level adds ``width`` space characters. Multi-line text has
    each line indented independently.

    Args:
        text:  The text to indent.
        level: Number of indent levels (0 = no indent).
        width: Number of spaces per indent level.

    Returns:
        Indented text string.
    """
    if level <= 0:
        return text

    prefix = " " * (level * width)
    lines = text.split("\n")
    return "\n".join(f"{prefix}{line}" for line in lines)


def pad_right(text: str, width: int) -> str:
    """Pad text with trailing spaces to reach the specified width.

    If the text is already at least ``width`` characters, it is
    returned unchanged (never truncated).

    Args:
        text:  The text to pad.
        width: Minimum total width.

    Returns:
        Right-padded text string.
    """
    if len(text) >= width:
        return text
    return text + " " * (width - len(text))


def truncate(text: str, max_width: int, *, suffix: str = "...") -> str:
    """Truncate text to a maximum width, appending a suffix if cut.

    If the text fits within ``max_width``, it is returned unchanged.
    Otherwise, it is cut and the suffix is appended so that the total
    length equals ``max_width``.

    Edge cases:
    - Empty text is returned as-is regardless of max_width.
    - If max_width is 0, returns empty string.
    - If max_width is less than the suffix length, the suffix itself
      is truncated to fit.

    Args:
        text:      The text to truncate.
        max_width: Maximum character width of the result.
        suffix:    String appended when truncation occurs.

    Returns:
        Truncated text string.
    """
    if not text or max_width <= 0:
        return "" if max_width <= 0 and text else text if not text else ""

    if len(text) <= max_width:
        return text

    if max_width <= len(suffix):
        return suffix[:max_width]

    return text[: max_width - len(suffix)] + suffix


def horizontal_rule(*, width: int = 60, char: str = "-") -> str:
    """Generate a horizontal rule of the given width.

    Args:
        width: Number of characters. Default 60.
        char:  The character to repeat. Default ``-``.

    Returns:
        A string of ``char`` repeated ``width`` times.
    """
    if width <= 0:
        return ""
    return char * width


# ---------------------------------------------------------------------------
# Box drawing
# ---------------------------------------------------------------------------


def box_top(*, width: int = 60) -> str:
    """Generate the top border of a text box.

    Produces a line like ``+----...----+`` where the inner dash area
    is ``width`` characters wide.

    Args:
        width: Inner content width. Default 60.

    Returns:
        Top border string.
    """
    return f"+{'-' * width}+"


def box_line(content: str, *, width: int = 60) -> str:
    """Generate a single content line within a text box.

    The content is left-aligned and padded (or truncated) to fill
    the inner width. The line is framed with ``|`` on both sides.

    Args:
        content: Text to display within the box line.
        width:   Inner content width. Default 60.

    Returns:
        Framed content line string.
    """
    # Reserve 2 chars for padding spaces around content
    inner_width = width - 2
    if inner_width < 0:
        inner_width = 0

    if len(content) > inner_width:
        display = truncate(content, inner_width)
    else:
        display = pad_right(content, inner_width)

    return f"| {display} |"


def box_bottom(*, width: int = 60) -> str:
    """Generate the bottom border of a text box.

    Identical to ``box_top()`` -- produces ``+----...----+``.

    Args:
        width: Inner content width. Default 60.

    Returns:
        Bottom border string.
    """
    return f"+{'-' * width}+"
