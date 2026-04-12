"""Base renderer types and EventRenderer protocol for agent loop events.

Defines the foundational types that all event formatters implement:

- **EventRenderer**: A ``typing.Protocol`` that every concrete event
  renderer must satisfy. Provides a ``render()`` method and an
  ``event_type`` property for registry-based dispatch.
- **RenderContext**: Frozen dataclass carrying style configuration and
  contextual metadata (iteration number, verbosity) through the
  rendering pipeline.
- **RenderedOutput**: Frozen dataclass representing the final text
  output of a render operation, with factory methods for common
  construction patterns.
- **EventSeverity**: Enum classifying the importance level of rendered
  events (debug through error).
- **render_header / render_footer**: Convenience helpers that produce
  visually consistent section boundaries.

Architecture::

    EventBus emits Event
        |
        v
    EventRenderer.render(event, context)
        |
        v
    RenderedOutput (text + metadata)
        |
        v
    CLI writes to terminal

All concrete renderers (tool call, approval prompt, iteration summary,
etc.) implement the ``EventRenderer`` protocol and are registered in
a renderer registry for dispatch by event type.

Usage::

    from jules_daemon.cli.event_renderer import (
        EventRenderer,
        EventSeverity,
        RenderContext,
        RenderedOutput,
    )

    class ToolCallRenderer:
        @property
        def event_type(self) -> str:
            return "tool_call"

        def render(
            self, event: object, context: RenderContext
        ) -> RenderedOutput:
            return RenderedOutput.from_text(f"Calling tool: {event}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from jules_daemon.cli.styles import (
    Color,
    StyleConfig,
    horizontal_rule,
    styled,
)

__all__ = [
    "EventRenderer",
    "EventSeverity",
    "RenderContext",
    "RenderedOutput",
    "render_footer",
    "render_header",
]


# ---------------------------------------------------------------------------
# EventSeverity
# ---------------------------------------------------------------------------


class EventSeverity(Enum):
    """Severity level for rendered event output.

    Used to classify the importance of an event for filtering and
    visual emphasis (e.g., errors get red styling, debug is dimmed).

    Values:
        DEBUG:   Verbose diagnostic information.
        INFO:    Normal operational messages.
        SUCCESS: Positive outcome notifications.
        WARNING: Caution or non-fatal issue.
        ERROR:   Failure or critical problem.
    """

    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


# Mapping from severity to color for styled output
_SEVERITY_COLORS: dict[EventSeverity, Color] = {
    EventSeverity.DEBUG: Color.DIM,
    EventSeverity.INFO: Color.CYAN,
    EventSeverity.SUCCESS: Color.GREEN,
    EventSeverity.WARNING: Color.YELLOW,
    EventSeverity.ERROR: Color.RED,
}


# ---------------------------------------------------------------------------
# RenderContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderContext:
    """Immutable context passed to every EventRenderer.render() call.

    Carries style configuration and contextual metadata about the
    current agent loop state. Renderers use this to adapt their output
    (e.g., suppress color codes, show iteration numbers, adjust width).

    Attributes:
        style:             Terminal styling configuration.
        verbose:           When True, renderers include extra detail
                           (e.g., full tool arguments, raw LLM response).
        current_iteration: The current think-act cycle number (1-based),
                           or None when not in an agent loop.
        max_iterations:    The configured iteration cap, or None when
                           not in an agent loop.
    """

    style: StyleConfig = field(default_factory=StyleConfig)
    verbose: bool = False
    current_iteration: int | None = None
    max_iterations: int | None = None


# ---------------------------------------------------------------------------
# RenderedOutput
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedOutput:
    """Immutable result of rendering an event to terminal text.

    Contains the formatted text and metadata about the output.

    Attributes:
        text:       The rendered text, ready to write to the terminal.
        line_count: Number of newline-separated lines in the text.
        severity:   The severity level of the rendered event.
    """

    text: str
    line_count: int
    severity: EventSeverity = EventSeverity.INFO

    @staticmethod
    def empty() -> RenderedOutput:
        """Create an empty RenderedOutput with no text.

        Returns:
            RenderedOutput with empty text and zero line count.
        """
        return RenderedOutput(text="", line_count=0)

    @staticmethod
    def from_text(text: str) -> RenderedOutput:
        """Create a RenderedOutput from a text string.

        Automatically computes the line count from the text content.
        Empty text produces a zero line count.

        Args:
            text: The rendered text.

        Returns:
            RenderedOutput with computed line count.
        """
        if not text:
            return RenderedOutput(text="", line_count=0)
        count = text.count("\n") + 1
        return RenderedOutput(text=text, line_count=count)

    @staticmethod
    def from_lines(lines: list[str]) -> RenderedOutput:
        """Create a RenderedOutput from a list of text lines.

        Joins lines with newline characters and sets the line count
        to the number of non-empty input lines.

        Args:
            lines: List of text lines (without trailing newlines).

        Returns:
            RenderedOutput with joined text and line count.
        """
        if not lines:
            return RenderedOutput(text="", line_count=0)
        return RenderedOutput(
            text="\n".join(lines),
            line_count=len(lines),
        )


# ---------------------------------------------------------------------------
# EventRenderer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EventRenderer(Protocol):
    """Protocol for event rendering implementations.

    Every concrete renderer must provide:

    - ``event_type``: A string property identifying which event type
      this renderer handles (used for registry dispatch).
    - ``render(event, context)``: Formats an event into terminal-ready
      text wrapped in a ``RenderedOutput``.

    The ``event`` parameter is typed as ``object`` to allow renderers
    to handle different event dataclass types. Concrete implementations
    should document which event type they expect and cast accordingly.

    Example::

        class ToolCallRenderer:
            @property
            def event_type(self) -> str:
                return "tool_call"

            def render(
                self, event: object, context: RenderContext
            ) -> RenderedOutput:
                # Cast event to expected type
                tool_event = cast(ToolCallEvent, event)
                return RenderedOutput.from_text(
                    f"Calling: {tool_event.tool_name}"
                )
    """

    @property
    def event_type(self) -> str:
        """The event type string this renderer handles."""
        ...

    def render(self, event: object, context: RenderContext) -> RenderedOutput:
        """Render an event to terminal-ready text.

        Args:
            event:   The event object to render.
            context: Rendering context with style and metadata.

        Returns:
            RenderedOutput containing the formatted text.
        """
        ...


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def render_header(title: str, context: RenderContext) -> RenderedOutput:
    """Render a section header with title and separator.

    Produces a two-line block: a horizontal rule followed by the title.
    When color is enabled, the title is styled with bold + cyan.

    Args:
        title:   The section title text.
        context: Rendering context for style configuration.

    Returns:
        RenderedOutput with the header text.
    """
    width = context.style.terminal_width
    rule = horizontal_rule(width=width)

    styled_title = styled(
        title,
        Color.BOLD,
        Color.CYAN,
        config=context.style,
    )

    lines = [rule, styled_title]
    return RenderedOutput.from_lines(lines)


def render_footer(context: RenderContext) -> RenderedOutput:
    """Render a section footer (horizontal rule).

    Produces a single horizontal rule line matching the terminal width.

    Args:
        context: Rendering context for style configuration.

    Returns:
        RenderedOutput with the footer text.
    """
    width = context.style.terminal_width
    rule = horizontal_rule(width=width)
    return RenderedOutput.from_text(rule)
