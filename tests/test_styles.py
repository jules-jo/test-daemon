"""Tests for terminal styling utilities.

Covers:
- Color palette ANSI code correctness
- Color disable/enable behavior
- Icon constant integrity
- Layout helpers: indent, pad_right, truncate, horizontal rule
- Box drawing helpers
- Styled text composition
"""

from __future__ import annotations

import pytest

from jules_daemon.cli.styles import (
    ICON_APPROVE,
    ICON_ARROW,
    ICON_BULLET,
    ICON_CANCEL,
    ICON_DONE,
    ICON_ERROR,
    ICON_INFO,
    ICON_PENDING,
    ICON_REJECT,
    ICON_RUNNING,
    ICON_TOOL,
    ICON_WARNING,
    Color,
    StyleConfig,
    box_bottom,
    box_line,
    box_top,
    horizontal_rule,
    indent,
    pad_right,
    styled,
    truncate,
)


# ---------------------------------------------------------------------------
# Color enum
# ---------------------------------------------------------------------------


class TestColor:
    """Tests for the Color enum values."""

    def test_reset_is_sgr_zero(self) -> None:
        assert Color.RESET.value == "\033[0m"

    def test_bold_is_sgr_one(self) -> None:
        assert Color.BOLD.value == "\033[1m"

    def test_dim_is_sgr_two(self) -> None:
        assert Color.DIM.value == "\033[2m"

    def test_standard_foreground_colors_are_valid_sgr(self) -> None:
        """All standard foreground colors should be valid SGR sequences."""
        expected_prefixes = {
            Color.RED: "\033[31m",
            Color.GREEN: "\033[32m",
            Color.YELLOW: "\033[33m",
            Color.BLUE: "\033[34m",
            Color.MAGENTA: "\033[35m",
            Color.CYAN: "\033[36m",
            Color.WHITE: "\033[37m",
        }
        for color, expected in expected_prefixes.items():
            assert color.value == expected, f"{color.name} mismatch"

    def test_bright_colors_use_high_intensity_codes(self) -> None:
        assert Color.BRIGHT_RED.value == "\033[91m"
        assert Color.BRIGHT_GREEN.value == "\033[92m"
        assert Color.BRIGHT_YELLOW.value == "\033[93m"
        assert Color.BRIGHT_CYAN.value == "\033[96m"

    def test_all_values_start_with_esc(self) -> None:
        for color in Color:
            assert color.value.startswith("\033["), (
                f"{color.name} does not start with ESC["
            )

    def test_all_values_end_with_m(self) -> None:
        for color in Color:
            assert color.value.endswith("m"), (
                f"{color.name} does not end with 'm'"
            )


# ---------------------------------------------------------------------------
# StyleConfig
# ---------------------------------------------------------------------------


class TestStyleConfig:
    """Tests for the StyleConfig frozen dataclass."""

    def test_default_config_enables_color(self) -> None:
        config = StyleConfig()
        assert config.color_enabled is True

    def test_default_config_width(self) -> None:
        config = StyleConfig()
        assert config.terminal_width == 80

    def test_custom_width(self) -> None:
        config = StyleConfig(terminal_width=120)
        assert config.terminal_width == 120

    def test_disabled_color(self) -> None:
        config = StyleConfig(color_enabled=False)
        assert config.color_enabled is False

    def test_is_frozen(self) -> None:
        config = StyleConfig()
        with pytest.raises(AttributeError):
            config.color_enabled = False  # type: ignore[misc]

    def test_custom_indent_width(self) -> None:
        config = StyleConfig(indent_width=4)
        assert config.indent_width == 4

    def test_default_indent_width(self) -> None:
        config = StyleConfig()
        assert config.indent_width == 2


# ---------------------------------------------------------------------------
# Icon constants
# ---------------------------------------------------------------------------


class TestIcons:
    """Tests for text-based icon constants."""

    def test_icons_are_non_empty_strings(self) -> None:
        icons = [
            ICON_RUNNING,
            ICON_DONE,
            ICON_ERROR,
            ICON_WARNING,
            ICON_INFO,
            ICON_PENDING,
            ICON_APPROVE,
            ICON_REJECT,
            ICON_TOOL,
            ICON_ARROW,
            ICON_BULLET,
            ICON_CANCEL,
        ]
        for icon in icons:
            assert isinstance(icon, str)
            assert len(icon) > 0

    def test_icons_are_distinct(self) -> None:
        """Each icon should be unique to avoid visual ambiguity."""
        icons = [
            ICON_RUNNING,
            ICON_DONE,
            ICON_ERROR,
            ICON_WARNING,
            ICON_INFO,
            ICON_PENDING,
            ICON_APPROVE,
            ICON_REJECT,
            ICON_TOOL,
            ICON_CANCEL,
        ]
        assert len(set(icons)) == len(icons)


# ---------------------------------------------------------------------------
# styled()
# ---------------------------------------------------------------------------


class TestStyled:
    """Tests for the styled() text composition function."""

    def test_single_color(self) -> None:
        result = styled("hello", Color.GREEN)
        assert result == f"{Color.GREEN.value}hello{Color.RESET.value}"

    def test_multiple_colors(self) -> None:
        result = styled("hello", Color.BOLD, Color.RED)
        expected = f"{Color.BOLD.value}{Color.RED.value}hello{Color.RESET.value}"
        assert result == expected

    def test_no_colors_returns_plain_text(self) -> None:
        result = styled("hello")
        assert result == "hello"

    def test_empty_text(self) -> None:
        result = styled("", Color.GREEN)
        assert result == f"{Color.GREEN.value}{Color.RESET.value}"

    def test_config_disables_color(self) -> None:
        config = StyleConfig(color_enabled=False)
        result = styled("hello", Color.GREEN, config=config)
        assert result == "hello"

    def test_config_disables_color_with_multiple(self) -> None:
        config = StyleConfig(color_enabled=False)
        result = styled("hello", Color.BOLD, Color.RED, config=config)
        assert result == "hello"


# ---------------------------------------------------------------------------
# indent()
# ---------------------------------------------------------------------------


class TestIndent:
    """Tests for the indent() layout helper."""

    def test_default_indent(self) -> None:
        result = indent("hello")
        assert result == "  hello"

    def test_custom_level(self) -> None:
        result = indent("hello", level=2)
        assert result == "    hello"

    def test_zero_level(self) -> None:
        result = indent("hello", level=0)
        assert result == "hello"

    def test_custom_width(self) -> None:
        result = indent("hello", width=4)
        assert result == "    hello"

    def test_multiline(self) -> None:
        result = indent("line1\nline2")
        assert result == "  line1\n  line2"

    def test_empty_string(self) -> None:
        result = indent("")
        assert result == "  "

    def test_level_three_width_three(self) -> None:
        result = indent("x", level=3, width=3)
        assert result == "         x"


# ---------------------------------------------------------------------------
# pad_right()
# ---------------------------------------------------------------------------


class TestPadRight:
    """Tests for the pad_right() layout helper."""

    def test_shorter_than_width(self) -> None:
        result = pad_right("hi", 10)
        assert result == "hi        "
        assert len(result) == 10

    def test_exact_width(self) -> None:
        result = pad_right("hello", 5)
        assert result == "hello"

    def test_longer_than_width(self) -> None:
        result = pad_right("hello world", 5)
        assert result == "hello world"

    def test_empty_string(self) -> None:
        result = pad_right("", 5)
        assert result == "     "
        assert len(result) == 5

    def test_zero_width(self) -> None:
        result = pad_right("hi", 0)
        assert result == "hi"


# ---------------------------------------------------------------------------
# truncate()
# ---------------------------------------------------------------------------


class TestTruncate:
    """Tests for the truncate() layout helper."""

    def test_shorter_than_max(self) -> None:
        result = truncate("hi", 10)
        assert result == "hi"

    def test_exact_max(self) -> None:
        result = truncate("hello", 5)
        assert result == "hello"

    def test_longer_than_max(self) -> None:
        result = truncate("hello world", 8)
        assert result == "hello..."

    def test_custom_suffix(self) -> None:
        result = truncate("hello world", 8, suffix="~")
        assert result == "hello w~"

    def test_max_less_than_suffix_length(self) -> None:
        result = truncate("hello", 2)
        assert result == ".."

    def test_empty_string(self) -> None:
        result = truncate("", 10)
        assert result == ""

    def test_max_zero(self) -> None:
        result = truncate("hello", 0)
        assert result == ""


# ---------------------------------------------------------------------------
# horizontal_rule()
# ---------------------------------------------------------------------------


class TestHorizontalRule:
    """Tests for the horizontal_rule() layout helper."""

    def test_default_width(self) -> None:
        result = horizontal_rule()
        assert result == "-" * 60

    def test_custom_width(self) -> None:
        result = horizontal_rule(width=40)
        assert result == "-" * 40

    def test_custom_char(self) -> None:
        result = horizontal_rule(char="=", width=10)
        assert result == "=" * 10

    def test_zero_width(self) -> None:
        result = horizontal_rule(width=0)
        assert result == ""

    def test_single_width(self) -> None:
        result = horizontal_rule(width=1)
        assert result == "-"


# ---------------------------------------------------------------------------
# Box drawing
# ---------------------------------------------------------------------------


class TestBoxDrawing:
    """Tests for box_top, box_line, box_bottom helpers."""

    def test_box_top_default(self) -> None:
        result = box_top()
        assert result.startswith("+")
        assert result.endswith("+")
        assert len(result) == 62  # +--...--+

    def test_box_top_custom_width(self) -> None:
        result = box_top(width=20)
        assert len(result) == 22  # +--...--+

    def test_box_line_content(self) -> None:
        result = box_line("hello", width=20)
        assert result.startswith("| ")
        assert result.endswith(" |")
        assert "hello" in result

    def test_box_line_pads_short_content(self) -> None:
        result = box_line("hi", width=20)
        # | hi               |
        assert len(result) == 22

    def test_box_line_truncates_long_content(self) -> None:
        long_text = "a" * 100
        result = box_line(long_text, width=20)
        assert len(result) == 22
        assert "..." in result

    def test_box_bottom_default(self) -> None:
        result = box_bottom()
        assert result.startswith("+")
        assert result.endswith("+")
        assert len(result) == 62

    def test_box_bottom_custom_width(self) -> None:
        result = box_bottom(width=20)
        assert len(result) == 22
