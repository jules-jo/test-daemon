"""Tests for the output formatting pipeline.

Covers:
- ANSI color code preservation and stripping
- Configurable timestamp prepending
- Multi-line chunk processing
- Empty and edge-case input handling
- Pipeline composition
- Immutable configuration
- Custom timestamp formats
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import pytest

from jules_daemon.monitor.formatting_pipeline import (
    AnsiMode,
    FormattedChunk,
    FormatterConfig,
    format_chunk,
    normalize_ansi,
    prepend_timestamps,
    strip_ansi,
)


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

# Common ANSI sequences used in test output
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"
# 8-bit color
COLOR_256 = "\033[38;5;196m"
# 24-bit (truecolor)
COLOR_RGB = "\033[38;2;255;100;0m"


# ---------------------------------------------------------------------------
# strip_ansi
# ---------------------------------------------------------------------------


class TestStripAnsi:
    def test_no_ansi(self) -> None:
        assert strip_ansi("plain text") == "plain text"

    def test_empty_string(self) -> None:
        assert strip_ansi("") == ""

    def test_single_code(self) -> None:
        assert strip_ansi(f"{GREEN}PASSED{RESET}") == "PASSED"

    def test_multiple_codes(self) -> None:
        text = f"{BOLD}{RED}FAILED{RESET} test_login"
        assert strip_ansi(text) == "FAILED test_login"

    def test_8bit_color(self) -> None:
        text = f"{COLOR_256}bright red{RESET}"
        assert strip_ansi(text) == "bright red"

    def test_24bit_color(self) -> None:
        text = f"{COLOR_RGB}truecolor{RESET}"
        assert strip_ansi(text) == "truecolor"

    def test_partial_line_codes(self) -> None:
        text = f"prefix {GREEN}middle{RESET} suffix"
        assert strip_ansi(text) == "prefix middle suffix"

    def test_nested_codes(self) -> None:
        text = f"{BOLD}{UNDERLINE}{RED}nested{RESET}"
        assert strip_ansi(text) == "nested"

    def test_multiline(self) -> None:
        text = f"{GREEN}line1{RESET}\n{RED}line2{RESET}"
        assert strip_ansi(text) == "line1\nline2"

    def test_cursor_movement_codes(self) -> None:
        # CSI sequences like cursor up/down
        text = "before\033[2Aafter"
        assert strip_ansi(text) == "beforeafter"

    def test_osc_sequences(self) -> None:
        # Operating System Command (title bar, hyperlinks)
        text = "\033]0;window title\007rest"
        assert strip_ansi(text) == "rest"


# ---------------------------------------------------------------------------
# normalize_ansi
# ---------------------------------------------------------------------------


class TestNormalizeAnsi:
    def test_preserves_standard_sgr(self) -> None:
        text = f"{GREEN}PASSED{RESET}"
        result = normalize_ansi(text)
        assert GREEN in result
        assert RESET in result
        assert "PASSED" in result

    def test_preserves_bold(self) -> None:
        text = f"{BOLD}important{RESET}"
        result = normalize_ansi(text)
        assert BOLD in result

    def test_strips_cursor_movement(self) -> None:
        text = f"\033[2A{GREEN}PASSED{RESET}"
        result = normalize_ansi(text)
        assert GREEN in result
        assert "\033[2A" not in result

    def test_strips_osc(self) -> None:
        text = f"\033]0;title\007{RED}error{RESET}"
        result = normalize_ansi(text)
        assert RED in result
        assert "\033]0;title\007" not in result

    def test_preserves_8bit_color(self) -> None:
        text = f"{COLOR_256}colored{RESET}"
        result = normalize_ansi(text)
        assert COLOR_256 in result

    def test_preserves_24bit_color(self) -> None:
        text = f"{COLOR_RGB}truecolor{RESET}"
        result = normalize_ansi(text)
        assert COLOR_RGB in result

    def test_empty_string(self) -> None:
        assert normalize_ansi("") == ""

    def test_no_ansi(self) -> None:
        assert normalize_ansi("plain text") == "plain text"

    def test_erase_line_stripped(self) -> None:
        # CSI K (erase in line) is non-SGR, should be stripped
        text = f"\033[K{GREEN}ok{RESET}"
        result = normalize_ansi(text)
        assert "\033[K" not in result
        assert GREEN in result


# ---------------------------------------------------------------------------
# FormatterConfig
# ---------------------------------------------------------------------------


class TestFormatterConfig:
    def test_defaults(self) -> None:
        config = FormatterConfig()
        assert config.ansi_mode == AnsiMode.PRESERVE
        assert config.timestamp_format == "%H:%M:%S"
        assert config.timestamp_enabled is True

    def test_frozen(self) -> None:
        config = FormatterConfig()
        with pytest.raises(AttributeError):
            config.ansi_mode = AnsiMode.STRIP  # type: ignore[misc]

    def test_custom_values(self) -> None:
        config = FormatterConfig(
            ansi_mode=AnsiMode.STRIP,
            timestamp_format="%Y-%m-%d %H:%M:%S",
            timestamp_enabled=False,
        )
        assert config.ansi_mode == AnsiMode.STRIP
        assert config.timestamp_format == "%Y-%m-%d %H:%M:%S"
        assert config.timestamp_enabled is False

    def test_normalize_mode(self) -> None:
        config = FormatterConfig(ansi_mode=AnsiMode.NORMALIZE)
        assert config.ansi_mode == AnsiMode.NORMALIZE


# ---------------------------------------------------------------------------
# prepend_timestamps
# ---------------------------------------------------------------------------


class TestPrependTimestamps:
    def _fixed_clock(self) -> Callable[[], datetime]:
        """Return a clock function that returns a fixed datetime."""
        fixed = datetime(2026, 4, 9, 14, 30, 45, tzinfo=timezone.utc)
        return lambda: fixed

    def test_single_line(self) -> None:
        clock = self._fixed_clock()
        result = prepend_timestamps(
            "PASSED test_foo",
            timestamp_format="%H:%M:%S",
            clock=clock,
        )
        assert result == "[14:30:45] PASSED test_foo"

    def test_multi_line(self) -> None:
        clock = self._fixed_clock()
        text = "line1\nline2\nline3"
        result = prepend_timestamps(
            text,
            timestamp_format="%H:%M:%S",
            clock=clock,
        )
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "[14:30:45] line1"
        assert lines[1] == "[14:30:45] line2"
        assert lines[2] == "[14:30:45] line3"

    def test_empty_string(self) -> None:
        clock = self._fixed_clock()
        result = prepend_timestamps("", timestamp_format="%H:%M:%S", clock=clock)
        assert result == ""

    def test_custom_format(self) -> None:
        clock = self._fixed_clock()
        result = prepend_timestamps(
            "test output",
            timestamp_format="%Y-%m-%d %H:%M:%S",
            clock=clock,
        )
        assert result == "[2026-04-09 14:30:45] test output"

    def test_preserves_trailing_newline(self) -> None:
        clock = self._fixed_clock()
        result = prepend_timestamps(
            "line1\nline2\n",
            timestamp_format="%H:%M:%S",
            clock=clock,
        )
        # Trailing empty string after split should not get a timestamp
        assert result.endswith("\n")
        lines = result.split("\n")
        # "line1", "line2", "" (trailing)
        assert lines[-1] == ""
        assert lines[0] == "[14:30:45] line1"
        assert lines[1] == "[14:30:45] line2"

    def test_blank_lines_get_timestamps(self) -> None:
        clock = self._fixed_clock()
        result = prepend_timestamps(
            "line1\n\nline3",
            timestamp_format="%H:%M:%S",
            clock=clock,
        )
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "[14:30:45] line1"
        assert lines[1] == "[14:30:45] "
        assert lines[2] == "[14:30:45] line3"

    def test_ansi_codes_not_disrupted(self) -> None:
        clock = self._fixed_clock()
        text = f"{GREEN}PASSED{RESET} test_foo"
        result = prepend_timestamps(
            text,
            timestamp_format="%H:%M:%S",
            clock=clock,
        )
        assert result == f"[14:30:45] {GREEN}PASSED{RESET} test_foo"


# ---------------------------------------------------------------------------
# FormattedChunk
# ---------------------------------------------------------------------------


class TestFormattedChunk:
    def test_creation(self) -> None:
        chunk = FormattedChunk(
            raw="raw text",
            formatted="[14:30:45] raw text",
            line_count=1,
            ansi_stripped=False,
        )
        assert chunk.raw == "raw text"
        assert chunk.formatted == "[14:30:45] raw text"
        assert chunk.line_count == 1
        assert chunk.ansi_stripped is False

    def test_frozen(self) -> None:
        chunk = FormattedChunk(
            raw="raw",
            formatted="formatted",
            line_count=1,
            ansi_stripped=False,
        )
        with pytest.raises(AttributeError):
            chunk.raw = "mutated"  # type: ignore[misc]

    def test_negative_line_count_raises(self) -> None:
        with pytest.raises(ValueError, match="line_count must not be negative"):
            FormattedChunk(
                raw="x",
                formatted="x",
                line_count=-1,
                ansi_stripped=False,
            )


# ---------------------------------------------------------------------------
# format_chunk (full pipeline)
# ---------------------------------------------------------------------------


class TestFormatChunk:
    def _fixed_clock(self) -> Callable[[], datetime]:
        fixed = datetime(2026, 4, 9, 14, 30, 45, tzinfo=timezone.utc)
        return lambda: fixed

    def test_default_config_preserves_ansi_adds_timestamps(self) -> None:
        clock = self._fixed_clock()
        config = FormatterConfig()
        raw = f"{GREEN}PASSED{RESET} test_foo"
        result = format_chunk(raw, config=config, clock=clock)

        assert isinstance(result, FormattedChunk)
        assert result.raw == raw
        assert GREEN in result.formatted
        assert "[14:30:45]" in result.formatted
        assert result.ansi_stripped is False
        assert result.line_count == 1

    def test_strip_mode_removes_ansi(self) -> None:
        clock = self._fixed_clock()
        config = FormatterConfig(ansi_mode=AnsiMode.STRIP)
        raw = f"{GREEN}PASSED{RESET} test_foo"
        result = format_chunk(raw, config=config, clock=clock)

        assert GREEN not in result.formatted
        assert RESET not in result.formatted
        assert "PASSED test_foo" in result.formatted
        assert result.ansi_stripped is True

    def test_normalize_mode_keeps_sgr_strips_cursor(self) -> None:
        clock = self._fixed_clock()
        config = FormatterConfig(ansi_mode=AnsiMode.NORMALIZE)
        raw = f"\033[2A{GREEN}PASSED{RESET}"
        result = format_chunk(raw, config=config, clock=clock)

        assert GREEN in result.formatted
        assert "\033[2A" not in result.formatted
        assert result.ansi_stripped is False

    def test_timestamps_disabled(self) -> None:
        clock = self._fixed_clock()
        config = FormatterConfig(timestamp_enabled=False)
        raw = "plain text"
        result = format_chunk(raw, config=config, clock=clock)

        assert result.formatted == "plain text"
        assert "[14:30:45]" not in result.formatted

    def test_multiline_chunk(self) -> None:
        clock = self._fixed_clock()
        config = FormatterConfig()
        raw = "line1\nline2\nline3"
        result = format_chunk(raw, config=config, clock=clock)

        assert result.line_count == 3
        lines = result.formatted.split("\n")
        assert len(lines) == 3
        for line in lines:
            assert line.startswith("[14:30:45]")

    def test_empty_input(self) -> None:
        clock = self._fixed_clock()
        config = FormatterConfig()
        result = format_chunk("", config=config, clock=clock)

        assert result.raw == ""
        assert result.formatted == ""
        assert result.line_count == 0

    def test_strip_plus_no_timestamps(self) -> None:
        clock = self._fixed_clock()
        config = FormatterConfig(
            ansi_mode=AnsiMode.STRIP,
            timestamp_enabled=False,
        )
        raw = f"{RED}FAILED{RESET} test_bar"
        result = format_chunk(raw, config=config, clock=clock)

        assert result.formatted == "FAILED test_bar"
        assert result.ansi_stripped is True

    def test_default_clock_produces_timestamp(self) -> None:
        """Without an explicit clock, format_chunk uses UTC now."""
        config = FormatterConfig()
        result = format_chunk("test line", config=config)

        # Should have a bracketed timestamp at the start
        assert result.formatted.startswith("[")
        assert "]" in result.formatted

    def test_custom_timestamp_format(self) -> None:
        clock = self._fixed_clock()
        config = FormatterConfig(timestamp_format="%Y-%m-%dT%H:%M:%S")
        result = format_chunk("line", config=config, clock=clock)

        assert "[2026-04-09T14:30:45]" in result.formatted

    def test_trailing_newline_preserved(self) -> None:
        clock = self._fixed_clock()
        config = FormatterConfig()
        raw = "line1\nline2\n"
        result = format_chunk(raw, config=config, clock=clock)

        assert result.formatted.endswith("\n")
        assert result.line_count == 2  # trailing empty not counted as a line

    def test_carriage_return_lines(self) -> None:
        """Handle \\r\\n line endings from remote systems."""
        clock = self._fixed_clock()
        config = FormatterConfig()
        raw = "line1\r\nline2\r\n"
        result = format_chunk(raw, config=config, clock=clock)

        assert "[14:30:45] line1" in result.formatted
        assert "[14:30:45] line2" in result.formatted
        assert result.line_count == 2

    def test_none_config_uses_defaults(self) -> None:
        """Passing config=None uses default FormatterConfig."""
        clock = self._fixed_clock()
        result = format_chunk("test line", config=None, clock=clock)

        assert "[14:30:45]" in result.formatted
        assert result.ansi_stripped is False

    def test_single_newline_only(self) -> None:
        """A string that is just a newline has zero content lines."""
        clock = self._fixed_clock()
        config = FormatterConfig()
        result = format_chunk("\n", config=config, clock=clock)

        assert result.line_count == 0
