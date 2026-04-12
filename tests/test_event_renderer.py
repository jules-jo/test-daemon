"""Tests for the EventRenderer protocol and base types.

Covers:
- EventRenderer protocol compliance
- RenderContext frozen dataclass
- RenderedOutput frozen dataclass
- EventSeverity enum values
- render_header / render_footer helpers
- Concrete renderer implementations satisfy the protocol
"""

from __future__ import annotations

from typing import runtime_checkable

import pytest

from jules_daemon.cli.event_renderer import (
    EventRenderer,
    EventSeverity,
    RenderContext,
    RenderedOutput,
    render_footer,
    render_header,
)
from jules_daemon.cli.styles import StyleConfig


# ---------------------------------------------------------------------------
# EventSeverity
# ---------------------------------------------------------------------------


class TestEventSeverity:
    """Tests for the EventSeverity enum."""

    def test_has_expected_members(self) -> None:
        members = {s.value for s in EventSeverity}
        expected = {"debug", "info", "success", "warning", "error"}
        assert members == expected

    def test_ordering_by_value(self) -> None:
        """Verify that severity names are distinct strings."""
        values = [s.value for s in EventSeverity]
        assert len(set(values)) == len(values)


# ---------------------------------------------------------------------------
# RenderContext
# ---------------------------------------------------------------------------


class TestRenderContext:
    """Tests for the RenderContext frozen dataclass."""

    def test_default_creation(self) -> None:
        ctx = RenderContext()
        assert ctx.style is not None
        assert isinstance(ctx.style, StyleConfig)
        assert ctx.verbose is False

    def test_custom_style(self) -> None:
        style = StyleConfig(color_enabled=False, terminal_width=120)
        ctx = RenderContext(style=style)
        assert ctx.style.color_enabled is False
        assert ctx.style.terminal_width == 120

    def test_verbose_flag(self) -> None:
        ctx = RenderContext(verbose=True)
        assert ctx.verbose is True

    def test_is_frozen(self) -> None:
        ctx = RenderContext()
        with pytest.raises(AttributeError):
            ctx.verbose = True  # type: ignore[misc]

    def test_iteration_context(self) -> None:
        ctx = RenderContext(current_iteration=3, max_iterations=5)
        assert ctx.current_iteration == 3
        assert ctx.max_iterations == 5

    def test_default_iteration_values(self) -> None:
        ctx = RenderContext()
        assert ctx.current_iteration is None
        assert ctx.max_iterations is None


# ---------------------------------------------------------------------------
# RenderedOutput
# ---------------------------------------------------------------------------


class TestRenderedOutput:
    """Tests for the RenderedOutput frozen dataclass."""

    def test_creation(self) -> None:
        output = RenderedOutput(text="hello", line_count=1)
        assert output.text == "hello"
        assert output.line_count == 1

    def test_is_frozen(self) -> None:
        output = RenderedOutput(text="hello", line_count=1)
        with pytest.raises(AttributeError):
            output.text = "world"  # type: ignore[misc]

    def test_empty_output(self) -> None:
        output = RenderedOutput.empty()
        assert output.text == ""
        assert output.line_count == 0

    def test_from_text_single_line(self) -> None:
        output = RenderedOutput.from_text("hello")
        assert output.text == "hello"
        assert output.line_count == 1

    def test_from_text_multi_line(self) -> None:
        output = RenderedOutput.from_text("line1\nline2\nline3")
        assert output.text == "line1\nline2\nline3"
        assert output.line_count == 3

    def test_from_text_empty(self) -> None:
        output = RenderedOutput.from_text("")
        assert output.text == ""
        assert output.line_count == 0

    def test_from_lines(self) -> None:
        output = RenderedOutput.from_lines(["line1", "line2"])
        assert output.text == "line1\nline2"
        assert output.line_count == 2

    def test_from_lines_empty(self) -> None:
        output = RenderedOutput.from_lines([])
        assert output.text == ""
        assert output.line_count == 0

    def test_severity_default_is_info(self) -> None:
        output = RenderedOutput(text="hello", line_count=1)
        assert output.severity == EventSeverity.INFO

    def test_severity_custom(self) -> None:
        output = RenderedOutput(
            text="error!",
            line_count=1,
            severity=EventSeverity.ERROR,
        )
        assert output.severity == EventSeverity.ERROR


# ---------------------------------------------------------------------------
# EventRenderer protocol
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Minimal fake event for testing renderers."""

    def __init__(self, message: str = "test") -> None:
        self.message = message


class _ValidRenderer:
    """A concrete class that satisfies the EventRenderer protocol."""

    def render(self, event: object, context: RenderContext) -> RenderedOutput:
        return RenderedOutput.from_text(f"rendered: {event}")

    @property
    def event_type(self) -> str:
        return "fake_event"


class _InvalidRenderer:
    """A class that does NOT satisfy the EventRenderer protocol."""

    def draw(self, event: object) -> str:
        return "drawn"


class TestEventRendererProtocol:
    """Tests for EventRenderer protocol compliance."""

    def test_valid_renderer_satisfies_protocol(self) -> None:
        renderer = _ValidRenderer()
        assert isinstance(renderer, EventRenderer)

    def test_invalid_renderer_does_not_satisfy(self) -> None:
        renderer = _InvalidRenderer()
        assert not isinstance(renderer, EventRenderer)

    def test_renderer_render_returns_rendered_output(self) -> None:
        renderer = _ValidRenderer()
        ctx = RenderContext()
        result = renderer.render(_FakeEvent(), ctx)
        assert isinstance(result, RenderedOutput)
        assert "rendered" in result.text

    def test_renderer_event_type_is_string(self) -> None:
        renderer = _ValidRenderer()
        assert isinstance(renderer.event_type, str)
        assert renderer.event_type == "fake_event"


# ---------------------------------------------------------------------------
# render_header / render_footer helpers
# ---------------------------------------------------------------------------


class TestRenderHeader:
    """Tests for the render_header helper."""

    def test_returns_rendered_output(self) -> None:
        ctx = RenderContext()
        result = render_header("Test Section", ctx)
        assert isinstance(result, RenderedOutput)

    def test_contains_title(self) -> None:
        ctx = RenderContext()
        result = render_header("My Title", ctx)
        assert "My Title" in result.text

    def test_has_separator(self) -> None:
        ctx = RenderContext()
        result = render_header("Title", ctx)
        # Header should contain some kind of visual separator
        assert "-" in result.text or "=" in result.text

    def test_line_count_positive(self) -> None:
        ctx = RenderContext()
        result = render_header("Title", ctx)
        assert result.line_count > 0

    def test_no_color_mode(self) -> None:
        style = StyleConfig(color_enabled=False)
        ctx = RenderContext(style=style)
        result = render_header("Title", ctx)
        assert "\033[" not in result.text


class TestRenderFooter:
    """Tests for the render_footer helper."""

    def test_returns_rendered_output(self) -> None:
        ctx = RenderContext()
        result = render_footer(ctx)
        assert isinstance(result, RenderedOutput)

    def test_has_separator(self) -> None:
        ctx = RenderContext()
        result = render_footer(ctx)
        assert "-" in result.text or "=" in result.text

    def test_line_count_positive(self) -> None:
        ctx = RenderContext()
        result = render_footer(ctx)
        assert result.line_count > 0

    def test_no_color_mode(self) -> None:
        style = StyleConfig(color_enabled=False)
        ctx = RenderContext(style=style)
        result = render_footer(ctx)
        assert "\033[" not in result.text
