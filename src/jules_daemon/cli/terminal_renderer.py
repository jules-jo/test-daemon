"""Terminal renderer for real-time SSH output with progress indicators.

Consumes formatted StreamChunk objects from an async queue (produced by
StreamChunkReceiver), detects progress patterns in the output text, and
renders in-place progress indicators (spinners, percentage bars) when
connected to a TTY. Non-progress output lines are written normally.

Progress pattern detection:

- **Percentage**: ``[45%]``, ``(67%)``, ``78%`` -- extracted as float.
- **Counter**: ``3 of 10``, ``test 5 of 20`` -- computed as percentage.
- **Ratio**: ``3/10 tests``, ``7/15`` -- computed as percentage.

When running on a TTY with progress indicators enabled, the renderer
uses carriage return (``\\r``) to overwrite the current line with an
updated progress bar. When the stream switches from progress to
non-progress output, the progress line is cleared before writing the
new output normally.

When running on a non-TTY (pipe, file redirect), progress indicators
are disabled and all lines are printed sequentially without ``\\r``.

Architecture::

    asyncio.Queue[StreamChunk | None]
        |
        v
    TerminalRenderer.run()
        |
        +--> detect_progress_pattern(text)
        |        |
        |        +--> ProgressMatch
        |
        +--> format_progress_bar(percentage, width)
        |
        +--> write to TextIO output
        |
        v
    RenderResult

The renderer is designed for single-use: create a new instance for
each streaming session.

Usage::

    from jules_daemon.cli.terminal_renderer import (
        RendererConfig,
        TerminalRenderer,
    )

    renderer = TerminalRenderer(
        queue=receiver.queue,
        config=RendererConfig(is_tty=True),
    )
    result = await renderer.run()
    print(f"Rendered {result.lines_rendered} lines")
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TextIO

from jules_daemon.ipc.stream_receiver import ChunkType, StreamChunk

__all__ = [
    "ProgressMatch",
    "ProgressType",
    "RenderResult",
    "RendererConfig",
    "TerminalExitReason",
    "TerminalRenderer",
    "detect_progress_pattern",
    "format_progress_bar",
    "format_spinner_frame",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Spinner frames for in-place animation
_SPINNER_FRAMES: tuple[str, ...] = (
    "|", "/", "-", "\\", "|", "/", "-", "\\",
)

# Bar fill and empty characters
_BAR_FILL = "="
_BAR_HEAD = ">"
_BAR_EMPTY = " "


# ---------------------------------------------------------------------------
# Progress pattern regexes
# ---------------------------------------------------------------------------

# Matches: [45%], [ 45%], [100%], (67%), etc.
_BRACKET_PERCENT_RE = re.compile(r"[\[\(]\s*(\d{1,3})%\s*[\]\)]")

# Matches: 78%, Progress: 78%
_BARE_PERCENT_RE = re.compile(r"(?<!\d)(\d{1,3})%(?!\d)")

# Matches: 3 of 10, test 5 of 20
_COUNTER_RE = re.compile(r"(\d+)\s+of\s+(\d+)", re.IGNORECASE)

# Matches: 3/10, 7/15 (optionally followed by text)
_RATIO_RE = re.compile(r"(?<!\w)(\d+)/(\d+)(?!\d)")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProgressType(Enum):
    """Type of progress pattern detected in output text.

    Values:
        NONE:       No progress pattern detected.
        PERCENTAGE: Explicit percentage value (e.g., ``[45%]``).
        COUNTER:    Counter pattern (e.g., ``3 of 10``).
        RATIO:      Ratio pattern (e.g., ``3/10``).
    """

    NONE = "none"
    PERCENTAGE = "percentage"
    COUNTER = "counter"
    RATIO = "ratio"


class TerminalExitReason(Enum):
    """Why the terminal renderer stopped.

    Values:
        STREAM_END:      The stream ended normally (end-of-stream or None sentinel).
        ERROR:           The daemon reported an error.
        CONNECTION_LOST: The socket connection was lost.
    """

    STREAM_END = "stream_end"
    ERROR = "error"
    CONNECTION_LOST = "connection_lost"


# ---------------------------------------------------------------------------
# Immutable data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressMatch:
    """Result of detecting a progress pattern in output text.

    Attributes:
        progress_type: The type of pattern detected.
        percentage:    Computed progress as a float (0.0 - 100.0), or None
                       when no progress was detected.
        label:         Descriptive label extracted from the text (may be empty).
        raw_text:      The original text that was analyzed.
    """

    progress_type: ProgressType
    percentage: float | None
    label: str
    raw_text: str

    @property
    def is_progress(self) -> bool:
        """True if a progress pattern was detected."""
        return self.progress_type != ProgressType.NONE


@dataclass(frozen=True)
class RendererConfig:
    """Immutable configuration for the terminal renderer.

    Attributes:
        show_progress_bar: Whether to render progress bars for percentage
                           patterns. Only effective when is_tty is True.
        show_spinner:      Whether to render spinner animations for active
                           processing. Only effective when is_tty is True.
        progress_bar_width: Character width of the progress bar (fill area).
        is_tty:            Whether the output is a TTY. When False, in-place
                           updates are disabled and all lines are printed
                           sequentially.
    """

    show_progress_bar: bool = True
    show_spinner: bool = True
    progress_bar_width: int = 30
    is_tty: bool = True

    def __post_init__(self) -> None:
        if self.progress_bar_width < 1:
            raise ValueError(
                f"progress_bar_width must be positive, got {self.progress_bar_width}"
            )


@dataclass(frozen=True)
class RenderResult:
    """Immutable result of a terminal rendering session.

    Attributes:
        lines_rendered:   Total number of non-progress output lines written.
        progress_updates: Number of in-place progress bar updates performed.
        exit_reason:      Why the renderer stopped.
    """

    lines_rendered: int
    progress_updates: int
    exit_reason: TerminalExitReason


# ---------------------------------------------------------------------------
# Pure functions: progress detection
# ---------------------------------------------------------------------------


def _safe_percentage(numerator: int, denominator: int) -> float:
    """Compute percentage safely, returning 0.0 when denominator is zero.

    Args:
        numerator:   The current count.
        denominator: The total count.

    Returns:
        Percentage as a float clamped to [0.0, 100.0].
    """
    if denominator <= 0:
        return 0.0
    return min(100.0, max(0.0, (numerator / denominator) * 100.0))


def detect_progress_pattern(text: str) -> ProgressMatch:
    """Detect progress patterns in output text.

    Checks for the following patterns in priority order:
    1. Bracketed percentages: ``[45%]``, ``(67%)``
    2. Bare percentages: ``78%``
    3. Counter patterns: ``3 of 10``
    4. Ratio patterns: ``3/10``

    Args:
        text: A single line of output text.

    Returns:
        ProgressMatch with the detected pattern type and computed
        percentage. Returns ProgressType.NONE when no pattern is found.
    """
    if not text:
        return ProgressMatch(
            progress_type=ProgressType.NONE,
            percentage=None,
            label="",
            raw_text=text,
        )

    # Priority 1: Bracketed percentage [45%], (67%)
    match = _BRACKET_PERCENT_RE.search(text)
    if match is not None:
        pct = float(match.group(1))
        return ProgressMatch(
            progress_type=ProgressType.PERCENTAGE,
            percentage=min(100.0, max(0.0, pct)),
            label=text[:match.start()].strip(),
            raw_text=text,
        )

    # Priority 2: Bare percentage 78%
    match = _BARE_PERCENT_RE.search(text)
    if match is not None:
        pct = float(match.group(1))
        return ProgressMatch(
            progress_type=ProgressType.PERCENTAGE,
            percentage=min(100.0, max(0.0, pct)),
            label=text[:match.start()].strip(),
            raw_text=text,
        )

    # Priority 3: Counter pattern (3 of 10)
    match = _COUNTER_RE.search(text)
    if match is not None:
        current = int(match.group(1))
        total = int(match.group(2))
        return ProgressMatch(
            progress_type=ProgressType.COUNTER,
            percentage=_safe_percentage(current, total),
            label=text[:match.start()].strip(),
            raw_text=text,
        )

    # Priority 4: Ratio pattern (3/10)
    match = _RATIO_RE.search(text)
    if match is not None:
        current = int(match.group(1))
        total = int(match.group(2))
        return ProgressMatch(
            progress_type=ProgressType.RATIO,
            percentage=_safe_percentage(current, total),
            label=text[:match.start()].strip(),
            raw_text=text,
        )

    return ProgressMatch(
        progress_type=ProgressType.NONE,
        percentage=None,
        label="",
        raw_text=text,
    )


# ---------------------------------------------------------------------------
# Pure functions: formatting
# ---------------------------------------------------------------------------


def format_progress_bar(percentage: float, width: int = 30) -> str:
    """Format a text-based progress bar string.

    Produces output like: ``[========>           ]  45%``

    The bar is clamped to [0%, 100%] regardless of input.

    Args:
        percentage: Progress value (0.0 to 100.0).
        width:      Character width of the bar fill area.

    Returns:
        Formatted progress bar string.
    """
    clamped = min(100.0, max(0.0, percentage))
    filled = int(width * clamped / 100.0)

    if filled > 0 and filled < width:
        bar = _BAR_FILL * (filled - 1) + _BAR_HEAD + _BAR_EMPTY * (width - filled)
    elif filled >= width:
        bar = _BAR_FILL * width
    else:
        bar = _BAR_EMPTY * width

    return f"[{bar}] {clamped:5.1f}%"


def format_spinner_frame(frame_index: int) -> str:
    """Return a single spinner frame character for the given index.

    The spinner cycles through a fixed set of frames. Any integer
    index is accepted (negative indices wrap naturally via modulo).

    Args:
        frame_index: The frame index (modulo the number of frames).

    Returns:
        A single-character spinner frame.
    """
    return _SPINNER_FRAMES[frame_index % len(_SPINNER_FRAMES)]


# ---------------------------------------------------------------------------
# TerminalRenderer
# ---------------------------------------------------------------------------


class TerminalRenderer:
    """Async renderer that consumes stream chunks and writes to the terminal.

    Reads StreamChunk objects from an async queue, detects progress patterns,
    and renders in-place progress indicators when running on a TTY. Falls
    back to sequential line output when not on a TTY.

    The renderer tracks two metrics:
    - ``lines_rendered``: Non-progress lines written normally (with newline).
    - ``progress_updates``: In-place progress bar updates performed.

    The renderer is designed for single-use: create a new instance for
    each streaming session.

    Note on mutability: Unlike the frozen dataclass value objects in this
    module, this class intentionally holds mutable internal state
    (_lines_rendered, _progress_updates, _spinner_index, _in_progress)
    because it is an active coroutine, not a value object.

    Args:
        queue:  The async queue to consume StreamChunk | None from.
        config: Renderer configuration.
        output: Text IO stream for writing output. Defaults to sys.stdout.
    """

    def __init__(
        self,
        *,
        queue: asyncio.Queue[StreamChunk | None],
        config: RendererConfig | None = None,
        output: TextIO | None = None,
    ) -> None:
        self._queue = queue
        self._config = config or RendererConfig()
        self._output = output or sys.stdout

        # Mutable per-session state
        self._lines_rendered = 0
        self._progress_updates = 0
        self._spinner_index = 0
        self._in_progress = False

    async def run(self) -> RenderResult:
        """Consume chunks from the queue and render to the terminal.

        Processes chunks in a loop until:
        - A None sentinel is received (stream exhausted).
        - A terminal chunk (END_OF_STREAM, ERROR, CONNECTION_LOST) is received.

        For OUTPUT chunks:
        - Detects progress patterns in the text.
        - When a progress pattern is found and TTY mode is enabled,
          renders an in-place progress bar using carriage return.
        - When no progress pattern is found, writes the line normally
          with a trailing newline.

        Returns:
            RenderResult summarizing the session.
        """
        exit_reason = TerminalExitReason.STREAM_END

        while True:
            chunk = await self._queue.get()

            if chunk is None:
                self._clear_progress_line()
                break

            if chunk.is_terminal:
                exit_reason = self._map_terminal_reason(chunk.chunk_type)
                self._clear_progress_line()
                if chunk.chunk_type == ChunkType.ERROR and chunk.text:
                    self._write_info(f"Error: {chunk.text}")
                elif chunk.chunk_type == ChunkType.CONNECTION_LOST:
                    self._write_info("Connection lost.")
                break

            # OUTPUT chunk -- detect progress and render
            self._render_output_chunk(chunk)

        return RenderResult(
            lines_rendered=self._lines_rendered,
            progress_updates=self._progress_updates,
            exit_reason=exit_reason,
        )

    # -- Internal rendering logic -------------------------------------------

    def _render_output_chunk(self, chunk: StreamChunk) -> None:
        """Render a single OUTPUT chunk to the terminal.

        Detects progress patterns and decides whether to render an
        in-place progress bar or a normal output line.

        Args:
            chunk: The OUTPUT StreamChunk to render.
        """
        progress = detect_progress_pattern(chunk.text)

        use_inline = (
            self._config.is_tty
            and progress.is_progress
            and (self._config.show_progress_bar or self._config.show_spinner)
        )

        if use_inline and self._config.show_progress_bar and progress.percentage is not None:
            self._render_progress_bar(progress)
        elif use_inline and self._config.show_spinner:
            self._render_spinner(chunk.text)
        else:
            # Normal line output
            self._clear_progress_line()
            self._write_line(chunk.text)
            self._lines_rendered += 1

    def _render_progress_bar(self, progress: ProgressMatch) -> None:
        """Render an in-place progress bar on the current terminal line.

        Uses carriage return to overwrite the current line. The progress
        bar is formatted with the detected percentage.

        Args:
            progress: The detected progress match with percentage.
        """
        if progress.percentage is None:
            return

        bar = format_progress_bar(
            progress.percentage,
            width=self._config.progress_bar_width,
        )

        label = progress.label
        if label:
            display = f"{bar} {label}"
        else:
            display = bar

        self._write_inline(display)
        self._in_progress = True
        self._progress_updates += 1

    def _render_spinner(self, text: str) -> None:
        """Render a spinner with the given text on the current line.

        Args:
            text: The text to display alongside the spinner.
        """
        frame = format_spinner_frame(self._spinner_index)
        self._spinner_index += 1
        display = f"{frame} {text}"
        self._write_inline(display)
        self._in_progress = True
        self._progress_updates += 1

    def _clear_progress_line(self) -> None:
        """Clear the current progress line if one is active.

        Writes spaces over the current line content and returns the
        cursor to the beginning. Only effective in TTY mode.
        """
        if not self._in_progress:
            return
        if not self._config.is_tty:
            return

        # Overwrite with spaces and return to start
        clear_width = self._config.progress_bar_width + 40
        self._output.write(f"\r{' ' * clear_width}\r")
        self._output.flush()
        self._in_progress = False

    # -- Output helpers -----------------------------------------------------

    def _write_line(self, text: str) -> None:
        """Write a line of text followed by a newline.

        Args:
            text: The text to write.
        """
        self._output.write(f"{text}\n")
        self._output.flush()

    def _write_inline(self, text: str) -> None:
        """Write text with carriage return for in-place update (TTY only).

        Args:
            text: The text to write on the current line.
        """
        self._output.write(f"\r{text}")
        self._output.flush()

    def _write_info(self, message: str) -> None:
        """Write an informational message prefixed with --.

        Args:
            message: The info message.
        """
        self._output.write(f"-- {message}\n")
        self._output.flush()

    @staticmethod
    def _map_terminal_reason(chunk_type: ChunkType) -> TerminalExitReason:
        """Map a terminal ChunkType to a TerminalExitReason.

        Args:
            chunk_type: The terminal chunk type.

        Returns:
            The corresponding exit reason.
        """
        mapping = {
            ChunkType.END_OF_STREAM: TerminalExitReason.STREAM_END,
            ChunkType.ERROR: TerminalExitReason.ERROR,
            ChunkType.CONNECTION_LOST: TerminalExitReason.CONNECTION_LOST,
        }
        return mapping.get(chunk_type, TerminalExitReason.STREAM_END)
