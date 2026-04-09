"""Tests for the terminal renderer.

Covers:
- Progress pattern detection (percentage, counter, ratio, spinner)
- Progress bar string formatting
- Spinner frame generation
- RendererConfig immutability and defaults
- TerminalRenderer queue consumption and output writing
- In-place progress indicator rendering
- Terminal chunk handling (END_OF_STREAM, ERROR, CONNECTION_LOST)
- None sentinel handling (stream end)
- Non-TTY fallback (no in-place updates)
- RenderResult reporting
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime, timezone

import pytest

from jules_daemon.ipc.stream_receiver import ChunkType, StreamChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    text: str,
    *,
    chunk_type: ChunkType = ChunkType.OUTPUT,
    sequence: int = 0,
    timestamp: str = "2026-04-09T14:30:45+00:00",
) -> StreamChunk:
    """Create a StreamChunk for testing."""
    return StreamChunk(
        chunk_type=chunk_type,
        text=text,
        sequence=sequence,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# ProgressType enum
# ---------------------------------------------------------------------------


class TestProgressType:
    def test_values(self) -> None:
        from jules_daemon.cli.terminal_renderer import ProgressType

        assert ProgressType.NONE.value == "none"
        assert ProgressType.PERCENTAGE.value == "percentage"
        assert ProgressType.COUNTER.value == "counter"
        assert ProgressType.RATIO.value == "ratio"


# ---------------------------------------------------------------------------
# ProgressMatch frozen dataclass
# ---------------------------------------------------------------------------


class TestProgressMatch:
    def test_creation(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressMatch,
            ProgressType,
        )

        match = ProgressMatch(
            progress_type=ProgressType.PERCENTAGE,
            percentage=45.0,
            label="Running tests",
            raw_text="Running tests [45%]",
        )
        assert match.progress_type == ProgressType.PERCENTAGE
        assert match.percentage == 45.0
        assert match.label == "Running tests"
        assert match.raw_text == "Running tests [45%]"

    def test_frozen(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressMatch,
            ProgressType,
        )

        match = ProgressMatch(
            progress_type=ProgressType.NONE,
            percentage=None,
            label="",
            raw_text="plain text",
        )
        with pytest.raises(AttributeError):
            match.percentage = 50.0  # type: ignore[misc]

    def test_is_progress_true_for_percentage(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressMatch,
            ProgressType,
        )

        match = ProgressMatch(
            progress_type=ProgressType.PERCENTAGE,
            percentage=50.0,
            label="",
            raw_text="[50%]",
        )
        assert match.is_progress is True

    def test_is_progress_false_for_none(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressMatch,
            ProgressType,
        )

        match = ProgressMatch(
            progress_type=ProgressType.NONE,
            percentage=None,
            label="",
            raw_text="no progress here",
        )
        assert match.is_progress is False


# ---------------------------------------------------------------------------
# detect_progress_pattern
# ---------------------------------------------------------------------------


class TestDetectProgressPattern:
    def test_pytest_bracket_percentage(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressType,
            detect_progress_pattern,
        )

        result = detect_progress_pattern(
            "tests/test_auth.py::test_login PASSED  [ 45%]"
        )
        assert result.progress_type == ProgressType.PERCENTAGE
        assert result.percentage == pytest.approx(45.0)

    def test_bare_percentage(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressType,
            detect_progress_pattern,
        )

        result = detect_progress_pattern("Progress: 78%")
        assert result.progress_type == ProgressType.PERCENTAGE
        assert result.percentage == pytest.approx(78.0)

    def test_100_percent(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressType,
            detect_progress_pattern,
        )

        result = detect_progress_pattern("[100%]")
        assert result.progress_type == ProgressType.PERCENTAGE
        assert result.percentage == pytest.approx(100.0)

    def test_zero_percent(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressType,
            detect_progress_pattern,
        )

        result = detect_progress_pattern("[  0%]")
        assert result.progress_type == ProgressType.PERCENTAGE
        assert result.percentage == pytest.approx(0.0)

    def test_counter_pattern(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressType,
            detect_progress_pattern,
        )

        result = detect_progress_pattern("Running test 3 of 10")
        assert result.progress_type == ProgressType.COUNTER
        assert result.percentage == pytest.approx(30.0)

    def test_ratio_pattern(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressType,
            detect_progress_pattern,
        )

        result = detect_progress_pattern("3/10 tests passed")
        assert result.progress_type == ProgressType.RATIO
        assert result.percentage == pytest.approx(30.0)

    def test_no_progress(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressType,
            detect_progress_pattern,
        )

        result = detect_progress_pattern("PASSED test_login")
        assert result.progress_type == ProgressType.NONE
        assert result.percentage is None

    def test_empty_string(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressType,
            detect_progress_pattern,
        )

        result = detect_progress_pattern("")
        assert result.progress_type == ProgressType.NONE

    def test_percentage_in_parentheses(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            ProgressType,
            detect_progress_pattern,
        )

        result = detect_progress_pattern("Test suite (67%)")
        assert result.progress_type == ProgressType.PERCENTAGE
        assert result.percentage == pytest.approx(67.0)

    def test_counter_zero_total_no_crash(self) -> None:
        """Counter with 0 total should not cause division by zero."""
        from jules_daemon.cli.terminal_renderer import (
            ProgressType,
            detect_progress_pattern,
        )

        result = detect_progress_pattern("0 of 0 complete")
        # Could be NONE or COUNTER with 0%; either is valid
        if result.progress_type == ProgressType.COUNTER:
            assert result.percentage == pytest.approx(0.0)

    def test_ratio_zero_denominator(self) -> None:
        """Ratio with 0 denominator should not cause division by zero."""
        from jules_daemon.cli.terminal_renderer import (
            ProgressType,
            detect_progress_pattern,
        )

        result = detect_progress_pattern("0/0 tests")
        if result.progress_type == ProgressType.RATIO:
            assert result.percentage == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# format_progress_bar
# ---------------------------------------------------------------------------


class TestFormatProgressBar:
    def test_zero_percent(self) -> None:
        from jules_daemon.cli.terminal_renderer import format_progress_bar

        bar = format_progress_bar(0.0, width=20)
        assert "[" in bar
        assert "]" in bar
        assert "0%" in bar

    def test_fifty_percent(self) -> None:
        from jules_daemon.cli.terminal_renderer import format_progress_bar

        bar = format_progress_bar(50.0, width=20)
        assert "50.0%" in bar

    def test_hundred_percent(self) -> None:
        from jules_daemon.cli.terminal_renderer import format_progress_bar

        bar = format_progress_bar(100.0, width=20)
        assert "100.0%" in bar

    def test_clamps_above_100(self) -> None:
        from jules_daemon.cli.terminal_renderer import format_progress_bar

        bar = format_progress_bar(150.0, width=20)
        assert "100.0%" in bar

    def test_clamps_below_0(self) -> None:
        from jules_daemon.cli.terminal_renderer import format_progress_bar

        bar = format_progress_bar(-10.0, width=20)
        assert "0%" in bar

    def test_custom_width(self) -> None:
        from jules_daemon.cli.terminal_renderer import format_progress_bar

        bar_narrow = format_progress_bar(50.0, width=10)
        bar_wide = format_progress_bar(50.0, width=40)
        assert len(bar_narrow) < len(bar_wide)

    def test_default_width(self) -> None:
        from jules_daemon.cli.terminal_renderer import format_progress_bar

        bar = format_progress_bar(50.0)
        assert "50.0%" in bar


# ---------------------------------------------------------------------------
# format_spinner_frame
# ---------------------------------------------------------------------------


class TestFormatSpinnerFrame:
    def test_frames_cycle(self) -> None:
        from jules_daemon.cli.terminal_renderer import format_spinner_frame

        frames: list[str] = []
        for i in range(8):
            frames.append(format_spinner_frame(i))
        # Should cycle back
        assert format_spinner_frame(0) == format_spinner_frame(len(frames))

    def test_different_frames(self) -> None:
        from jules_daemon.cli.terminal_renderer import format_spinner_frame

        # Not all frames are identical (at least 2 distinct frames)
        unique = {format_spinner_frame(i) for i in range(4)}
        assert len(unique) >= 2

    def test_negative_index_wraps(self) -> None:
        from jules_daemon.cli.terminal_renderer import format_spinner_frame

        # Should not raise
        frame = format_spinner_frame(-1)
        assert isinstance(frame, str)


# ---------------------------------------------------------------------------
# RendererConfig
# ---------------------------------------------------------------------------


class TestRendererConfig:
    def test_defaults(self) -> None:
        from jules_daemon.cli.terminal_renderer import RendererConfig

        config = RendererConfig()
        assert config.show_progress_bar is True
        assert config.show_spinner is True
        assert config.progress_bar_width == 30
        assert config.is_tty is True

    def test_frozen(self) -> None:
        from jules_daemon.cli.terminal_renderer import RendererConfig

        config = RendererConfig()
        with pytest.raises(AttributeError):
            config.show_progress_bar = False  # type: ignore[misc]

    def test_custom_values(self) -> None:
        from jules_daemon.cli.terminal_renderer import RendererConfig

        config = RendererConfig(
            show_progress_bar=False,
            show_spinner=False,
            progress_bar_width=50,
            is_tty=False,
        )
        assert config.show_progress_bar is False
        assert config.show_spinner is False
        assert config.progress_bar_width == 50
        assert config.is_tty is False

    def test_negative_width_raises(self) -> None:
        from jules_daemon.cli.terminal_renderer import RendererConfig

        with pytest.raises(ValueError, match="progress_bar_width"):
            RendererConfig(progress_bar_width=0)


# ---------------------------------------------------------------------------
# RenderResult
# ---------------------------------------------------------------------------


class TestRenderResult:
    def test_creation(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            RenderResult,
            TerminalExitReason,
        )

        result = RenderResult(
            lines_rendered=42,
            progress_updates=10,
            exit_reason=TerminalExitReason.STREAM_END,
        )
        assert result.lines_rendered == 42
        assert result.progress_updates == 10
        assert result.exit_reason == TerminalExitReason.STREAM_END

    def test_frozen(self) -> None:
        from jules_daemon.cli.terminal_renderer import (
            RenderResult,
            TerminalExitReason,
        )

        result = RenderResult(
            lines_rendered=0,
            progress_updates=0,
            exit_reason=TerminalExitReason.STREAM_END,
        )
        with pytest.raises(AttributeError):
            result.lines_rendered = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TerminalExitReason
# ---------------------------------------------------------------------------


class TestTerminalExitReason:
    def test_values(self) -> None:
        from jules_daemon.cli.terminal_renderer import TerminalExitReason

        assert TerminalExitReason.STREAM_END.value == "stream_end"
        assert TerminalExitReason.ERROR.value == "error"
        assert TerminalExitReason.CONNECTION_LOST.value == "connection_lost"


# ---------------------------------------------------------------------------
# TerminalRenderer -- async tests
# ---------------------------------------------------------------------------


class TestTerminalRenderer:
    """Integration tests for the TerminalRenderer queue consumer."""

    @pytest.fixture
    def output_buf(self) -> io.StringIO:
        """In-memory output buffer for capturing rendered output."""
        return io.StringIO()

    @pytest.fixture
    def queue(self) -> asyncio.Queue[StreamChunk | None]:
        """Pre-built async queue for feeding chunks to the renderer."""
        return asyncio.Queue()

    @pytest.mark.asyncio
    async def test_renders_output_lines(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalExitReason,
            TerminalRenderer,
        )

        config = RendererConfig(is_tty=False)
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        # Enqueue output lines + None sentinel
        await queue.put(_make_chunk("line 1", sequence=0))
        await queue.put(_make_chunk("line 2", sequence=1))
        await queue.put(None)

        result = await renderer.run()

        assert result.lines_rendered == 2
        assert result.exit_reason == TerminalExitReason.STREAM_END
        output = output_buf.getvalue()
        assert "line 1" in output
        assert "line 2" in output

    @pytest.mark.asyncio
    async def test_handles_end_of_stream(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalExitReason,
            TerminalRenderer,
        )

        config = RendererConfig(is_tty=False)
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        await queue.put(
            _make_chunk("", chunk_type=ChunkType.END_OF_STREAM, sequence=0)
        )
        await queue.put(None)

        result = await renderer.run()
        assert result.exit_reason == TerminalExitReason.STREAM_END

    @pytest.mark.asyncio
    async def test_handles_error_chunk(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalExitReason,
            TerminalRenderer,
        )

        config = RendererConfig(is_tty=False)
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        await queue.put(
            _make_chunk(
                "Something went wrong",
                chunk_type=ChunkType.ERROR,
                sequence=0,
            )
        )
        await queue.put(None)

        result = await renderer.run()
        assert result.exit_reason == TerminalExitReason.ERROR

    @pytest.mark.asyncio
    async def test_handles_connection_lost_chunk(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalExitReason,
            TerminalRenderer,
        )

        config = RendererConfig(is_tty=False)
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        await queue.put(
            _make_chunk("", chunk_type=ChunkType.CONNECTION_LOST, sequence=0)
        )
        await queue.put(None)

        result = await renderer.run()
        assert result.exit_reason == TerminalExitReason.CONNECTION_LOST

    @pytest.mark.asyncio
    async def test_none_sentinel_alone(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalExitReason,
            TerminalRenderer,
        )

        config = RendererConfig(is_tty=False)
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        await queue.put(None)

        result = await renderer.run()
        assert result.lines_rendered == 0
        assert result.exit_reason == TerminalExitReason.STREAM_END

    @pytest.mark.asyncio
    async def test_progress_line_counted_as_update_tty(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalExitReason,
            TerminalRenderer,
        )

        config = RendererConfig(is_tty=True, show_progress_bar=True)
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        await queue.put(
            _make_chunk(
                "tests/test_auth.py::test_login PASSED  [ 45%]",
                sequence=0,
            )
        )
        await queue.put(None)

        result = await renderer.run()
        assert result.progress_updates >= 1

    @pytest.mark.asyncio
    async def test_non_tty_no_inline_progress(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        """In non-TTY mode, progress lines are printed normally (no \\r)."""
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalRenderer,
        )

        config = RendererConfig(is_tty=False, show_progress_bar=True)
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        await queue.put(
            _make_chunk(
                "tests/test_auth.py::test_login PASSED  [ 45%]",
                sequence=0,
            )
        )
        await queue.put(None)

        await renderer.run()
        output = output_buf.getvalue()
        # Non-TTY mode should not use \r carriage returns
        assert "\r" not in output

    @pytest.mark.asyncio
    async def test_tty_progress_uses_carriage_return(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        """In TTY mode, progress indicators use \\r for in-place updates."""
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalRenderer,
        )

        config = RendererConfig(is_tty=True, show_progress_bar=True)
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        await queue.put(
            _make_chunk(
                "tests/test_auth.py::test_login PASSED  [ 45%]",
                sequence=0,
            )
        )
        await queue.put(
            _make_chunk(
                "tests/test_auth.py::test_signup PASSED  [ 90%]",
                sequence=1,
            )
        )
        await queue.put(None)

        await renderer.run()
        output = output_buf.getvalue()
        # TTY mode should use \r for progress updates
        assert "\r" in output

    @pytest.mark.asyncio
    async def test_mixed_progress_and_regular_lines(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalRenderer,
        )

        config = RendererConfig(is_tty=True, show_progress_bar=True)
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        await queue.put(_make_chunk("=== test session starts ===", sequence=0))
        await queue.put(
            _make_chunk(
                "tests/test_foo.py::test_a PASSED  [ 50%]", sequence=1
            )
        )
        await queue.put(
            _make_chunk(
                "tests/test_foo.py::test_b PASSED  [100%]", sequence=2
            )
        )
        await queue.put(
            _make_chunk("=== 2 passed in 0.5s ===", sequence=3)
        )
        await queue.put(None)

        result = await renderer.run()
        assert result.lines_rendered >= 2  # at least the non-progress lines
        output = output_buf.getvalue()
        assert "test session starts" in output
        assert "2 passed" in output

    @pytest.mark.asyncio
    async def test_progress_bar_disabled(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalRenderer,
        )

        config = RendererConfig(
            is_tty=True, show_progress_bar=False, show_spinner=False
        )
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        await queue.put(
            _make_chunk(
                "tests/test_auth.py::test_login PASSED  [ 45%]",
                sequence=0,
            )
        )
        await queue.put(None)

        result = await renderer.run()
        # With progress indicators disabled, output is printed normally
        assert result.progress_updates == 0
        output = output_buf.getvalue()
        assert "\r" not in output

    @pytest.mark.asyncio
    async def test_spinner_shows_for_progress_line_bar_disabled(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        """When show_spinner is True and show_progress_bar is False, use spinner."""
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalRenderer,
        )

        config = RendererConfig(
            is_tty=True, show_spinner=True, show_progress_bar=False
        )
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        # This line has a progress pattern but bar is disabled, so spinner kicks in
        await queue.put(
            _make_chunk("tests/test_auth.py::test_login PASSED  [ 45%]", sequence=0)
        )
        await queue.put(None)

        result = await renderer.run()
        assert result.progress_updates >= 1
        output = output_buf.getvalue()
        # Spinner uses \r for in-place update
        assert "\r" in output

    @pytest.mark.asyncio
    async def test_no_progress_line_prints_normally(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        """Non-progress lines are printed normally even when spinner is enabled."""
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalRenderer,
        )

        config = RendererConfig(
            is_tty=True, show_spinner=True, show_progress_bar=False
        )
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        await queue.put(_make_chunk("collecting tests...", sequence=0))
        await queue.put(None)

        result = await renderer.run()
        assert result.lines_rendered >= 1

    @pytest.mark.asyncio
    async def test_clears_progress_line_before_final_output(
        self,
        output_buf: io.StringIO,
        queue: asyncio.Queue[StreamChunk | None],
    ) -> None:
        """After progress updates, final non-progress output clears the line."""
        from jules_daemon.cli.terminal_renderer import (
            RendererConfig,
            TerminalRenderer,
        )

        config = RendererConfig(is_tty=True, show_progress_bar=True)
        renderer = TerminalRenderer(
            queue=queue,
            config=config,
            output=output_buf,
        )

        await queue.put(
            _make_chunk("test_a PASSED  [ 50%]", sequence=0)
        )
        await queue.put(
            _make_chunk("=== 1 passed ===", sequence=1)
        )
        await queue.put(None)

        await renderer.run()
        output = output_buf.getvalue()
        # The final line should appear after clearing the progress bar
        assert "1 passed" in output
