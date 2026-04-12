"""Per-event-type format functions for agent loop CLI rendering.

Provides frozen dataclass event models and format functions for each
event kind produced by the agent loop. Each format function accepts an
event object and a ``RenderContext``, and returns a ``RenderedOutput``
with styled terminal text.

Event kinds and their format functions:

    ``ToolCallEvent``       -> ``format_tool_call()``
        Displays which tool the LLM is invoking and its arguments.

    ``ApprovalPromptEvent`` -> ``format_approval_prompt()``
        Displays a visually framed approval prompt for state-changing
        tools (propose_ssh_command, execute_ssh).

    ``ObservationEvent``    -> ``format_observation()``
        Displays the result of a tool execution (success, error,
        denied, timeout) with appropriate icons and severity.

    ``ErrorEvent``          -> ``format_error()``
        Displays loop-level or LLM-level errors (transient, permanent,
        loop exhausted).

    ``StatusChangeEvent``   -> ``format_status_change()``
        Displays agent loop state transitions (THINKING -> ACTING, etc.)
        with iteration counters.

Each event kind also has a concrete ``EventRenderer`` implementation
that satisfies the ``EventRenderer`` protocol and can be registered in
a renderer registry for dispatch by event type.

All event models are frozen dataclasses following the project-wide
immutability convention.

Usage::

    from jules_daemon.cli.event_formats import (
        ToolCallEvent,
        format_tool_call,
    )
    from jules_daemon.cli.event_renderer import RenderContext

    event = ToolCallEvent(
        tool_name="read_wiki",
        arguments={"slug": "auth-tests"},
        call_id="call_001",
    )
    ctx = RenderContext()
    output = format_tool_call(event, ctx)
    print(output.text)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from jules_daemon.agent.agent_loop import AgentLoopState
from jules_daemon.agent.tool_types import ToolResultStatus
from jules_daemon.cli.event_renderer import (
    EventSeverity,
    RenderContext,
    RenderedOutput,
)
from jules_daemon.cli.styles import (
    ICON_APPROVE,
    ICON_ARROW,
    ICON_DONE,
    ICON_ERROR,
    ICON_REJECT,
    ICON_RUNNING,
    ICON_TOOL,
    ICON_WARNING,
    Color,
    StyleConfig,
    box_bottom,
    box_line,
    box_top,
    indent,
    styled,
    truncate,
)

__all__ = [
    "ApprovalPromptEvent",
    "ApprovalPromptRenderer",
    "ErrorEvent",
    "ErrorRenderer",
    "ErrorType",
    "ObservationEvent",
    "ObservationRenderer",
    "StatusChangeEvent",
    "StatusChangeRenderer",
    "ToolCallEvent",
    "ToolCallRenderer",
    "format_approval_prompt",
    "format_error",
    "format_observation",
    "format_status_change",
    "format_tool_call",
]


# ---------------------------------------------------------------------------
# ErrorType enum
# ---------------------------------------------------------------------------


class ErrorType(Enum):
    """Classification of agent loop errors for display purposes.

    Values:
        TRANSIENT:      Retryable error (LLM timeout, network blip).
        PERMANENT:      Non-retryable error (malformed response, auth failure).
        LOOP_EXHAUSTED: Max iterations reached without completion.
    """

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    LOOP_EXHAUSTED = "loop_exhausted"


# ---------------------------------------------------------------------------
# Severity mapping helpers
# ---------------------------------------------------------------------------

_STATUS_SEVERITY: dict[ToolResultStatus, EventSeverity] = {
    ToolResultStatus.SUCCESS: EventSeverity.SUCCESS,
    ToolResultStatus.ERROR: EventSeverity.ERROR,
    ToolResultStatus.DENIED: EventSeverity.ERROR,
    ToolResultStatus.TIMEOUT: EventSeverity.WARNING,
}

_STATUS_ICONS: dict[ToolResultStatus, str] = {
    ToolResultStatus.SUCCESS: ICON_DONE,
    ToolResultStatus.ERROR: ICON_ERROR,
    ToolResultStatus.DENIED: ICON_REJECT,
    ToolResultStatus.TIMEOUT: ICON_WARNING,
}

_STATUS_COLORS: dict[ToolResultStatus, Color] = {
    ToolResultStatus.SUCCESS: Color.GREEN,
    ToolResultStatus.ERROR: Color.RED,
    ToolResultStatus.DENIED: Color.RED,
    ToolResultStatus.TIMEOUT: Color.YELLOW,
}

_STATUS_LABELS: dict[ToolResultStatus, str] = {
    ToolResultStatus.SUCCESS: "success",
    ToolResultStatus.ERROR: "error",
    ToolResultStatus.DENIED: "denied",
    ToolResultStatus.TIMEOUT: "timeout",
}

_ERROR_TYPE_LABELS: dict[ErrorType, str] = {
    ErrorType.TRANSIENT: "Transient",
    ErrorType.PERMANENT: "Permanent",
    ErrorType.LOOP_EXHAUSTED: "Loop exhausted",
}


# ---------------------------------------------------------------------------
# Output preview helper
# ---------------------------------------------------------------------------

_MAX_PREVIEW_LINES = 5
_MAX_PREVIEW_WIDTH = 120


def _format_output_preview(
    output: str,
    *,
    config: StyleConfig,
    indent_level: int = 2,
) -> list[str]:
    """Format an output string as a truncated, indented preview.

    Returns a list of indented lines suitable for joining into the
    rendered output. Limits the preview to the first few lines and
    truncates each line to the configured width.

    Args:
        output: Raw tool output text.
        config: StyleConfig for layout parameters.
        indent_level: Number of indent levels to apply.

    Returns:
        List of indented, truncated preview lines.
    """
    if not output or not output.strip():
        return []

    raw_lines = output.strip().split("\n")
    preview_lines: list[str] = []
    indent_width = config.indent_width

    for line in raw_lines[:_MAX_PREVIEW_LINES]:
        truncated = truncate(line, _MAX_PREVIEW_WIDTH)
        preview_lines.append(
            indent(truncated, level=indent_level, width=indent_width)
        )

    remaining = len(raw_lines) - _MAX_PREVIEW_LINES
    if remaining > 0:
        suffix = indent(
            f"... ({remaining} more line{'s' if remaining != 1 else ''})",
            level=indent_level,
            width=indent_width,
        )
        preview_lines.append(suffix)

    return preview_lines


# ---------------------------------------------------------------------------
# ToolCallEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallEvent:
    """Event representing an LLM-issued tool invocation.

    Attributes:
        tool_name: Name of the tool being called.
        arguments: Key-value arguments passed to the tool.
        call_id:   Unique identifier for this tool call.
        iteration: Current agent loop iteration (optional).
    """

    tool_name: str
    arguments: dict[str, Any]
    call_id: str
    iteration: int | None = None

    def __post_init__(self) -> None:
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")
        if not self.call_id or not self.call_id.strip():
            raise ValueError("call_id must not be empty")
        # Defensive copy to prevent external mutation of the frozen dataclass
        object.__setattr__(self, "arguments", dict(self.arguments))


def format_tool_call(event: ToolCallEvent, context: RenderContext) -> RenderedOutput:
    """Format a tool call event for terminal display.

    Displays the tool name with a tool icon. In verbose mode, includes
    the full argument key-value pairs.

    Args:
        event:   The tool call event to render.
        context: Rendering context with style configuration.

    Returns:
        RenderedOutput with styled text and INFO severity.
    """
    config = context.style
    lines: list[str] = []

    # Header line: [T] Calling: tool_name
    icon = styled(ICON_TOOL, Color.CYAN, config=config)
    label = styled("Calling:", Color.BOLD, config=config)
    name = styled(event.tool_name, Color.CYAN, Color.BOLD, config=config)
    header = f"{icon} {label} {name}"

    # Append argument summary
    if event.arguments:
        arg_keys = ", ".join(sorted(event.arguments.keys()))
        args_display = styled(f"({arg_keys})", Color.DIM, config=config)
        header = f"{header} {args_display}"
    else:
        empty_args = styled("(no arguments)", Color.DIM, config=config)
        header = f"{header} {empty_args}"

    lines.append(header)

    # Verbose: show full arguments as indented JSON
    if context.verbose and event.arguments:
        indent_width = config.indent_width
        for key, value in sorted(event.arguments.items()):
            value_str = json.dumps(value) if not isinstance(value, str) else value
            arg_line = indent(
                f"{key}: {value_str}",
                level=2,
                width=indent_width,
            )
            lines.append(styled(arg_line, Color.DIM, config=config))

    return RenderedOutput(
        text="\n".join(lines),
        line_count=len(lines),
        severity=EventSeverity.INFO,
    )


# ---------------------------------------------------------------------------
# ApprovalPromptEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovalPromptEvent:
    """Event representing an approval prompt for a state-changing tool.

    Attributes:
        tool_name: Name of the tool requiring approval.
        command:   The SSH command or action requiring approval.
        call_id:   Unique identifier for the associated tool call.
    """

    tool_name: str
    command: str
    call_id: str

    def __post_init__(self) -> None:
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")
        if not self.command or not self.command.strip():
            raise ValueError("command must not be empty")
        if not self.call_id or not self.call_id.strip():
            raise ValueError("call_id must not be empty")


def format_approval_prompt(
    event: ApprovalPromptEvent,
    context: RenderContext,
) -> RenderedOutput:
    """Format an approval prompt for terminal display.

    Renders the command in a visually distinct box with a warning-level
    styling to draw the user's attention. Approval prompts are always
    prominent -- they require human confirmation before execution.

    Args:
        event:   The approval prompt event to render.
        context: Rendering context with style configuration.

    Returns:
        RenderedOutput with styled text and WARNING severity.
    """
    config = context.style
    box_width = min(config.terminal_width - 4, 72)
    lines: list[str] = []

    # Top border
    lines.append(styled(box_top(width=box_width), Color.YELLOW, config=config))

    # Tool name line
    tool_label = f"  {ICON_APPROVE} {event.tool_name}"
    lines.append(
        styled(
            box_line(tool_label, width=box_width),
            Color.YELLOW,
            config=config,
        )
    )

    # Separator
    lines.append(
        styled(
            box_line("-" * (box_width - 2), width=box_width),
            Color.YELLOW,
            config=config,
        )
    )

    # Command line(s) -- may wrap for long commands
    cmd_display = f"  Command: {event.command}"
    lines.append(
        styled(
            box_line(cmd_display, width=box_width),
            Color.YELLOW,
            Color.BOLD,
            config=config,
        )
    )

    # Bottom border
    lines.append(styled(box_bottom(width=box_width), Color.YELLOW, config=config))

    return RenderedOutput(
        text="\n".join(lines),
        line_count=len(lines),
        severity=EventSeverity.WARNING,
    )


# ---------------------------------------------------------------------------
# ObservationEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationEvent:
    """Event representing a tool execution result (observation).

    Attributes:
        tool_name:        Name of the tool that was executed.
        status:           Outcome classification (success, error, denied, timeout).
        output:           The tool's output text.
        call_id:          Unique identifier for the tool call.
        error_message:    Human-readable error description (on failure).
        duration_seconds: Wall-clock execution time (optional).
    """

    tool_name: str
    status: ToolResultStatus
    output: str
    call_id: str
    error_message: str | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")
        if not self.call_id or not self.call_id.strip():
            raise ValueError("call_id must not be empty")


def format_observation(
    event: ObservationEvent,
    context: RenderContext,
) -> RenderedOutput:
    """Format a tool observation (result) for terminal display.

    Selects icon, color, and severity based on the result status.
    On success, shows a concise completion line. On failure, shows
    the error message. In verbose mode, includes an output preview.

    Args:
        event:   The observation event to render.
        context: Rendering context with style configuration.

    Returns:
        RenderedOutput with styled text and status-appropriate severity.
    """
    config = context.style
    lines: list[str] = []

    icon = _STATUS_ICONS.get(event.status, ICON_ERROR)
    color = _STATUS_COLORS.get(event.status, Color.RED)
    severity = _STATUS_SEVERITY.get(event.status, EventSeverity.ERROR)
    status_label = _STATUS_LABELS.get(event.status, "unknown")

    # Header line: [icon] tool_name: status_label
    styled_icon = styled(icon, color, config=config)
    styled_name = styled(event.tool_name, Color.BOLD, config=config)
    styled_status = styled(status_label, color, config=config)
    header = f"{styled_icon} {styled_name}: {styled_status}"

    # Append duration if available
    if event.duration_seconds is not None:
        duration_str = styled(
            f"({event.duration_seconds:.1f}s)",
            Color.DIM,
            config=config,
        )
        header = f"{header} {duration_str}"

    lines.append(header)

    # Error message for non-success statuses
    if event.status is not ToolResultStatus.SUCCESS and event.error_message:
        indent_width = config.indent_width
        error_line = indent(
            styled(event.error_message, color, config=config),
            level=2,
            width=indent_width,
        )
        lines.append(error_line)

    # Verbose: output preview for all statuses
    if context.verbose and event.output:
        preview_lines = _format_output_preview(
            event.output,
            config=config,
        )
        lines.extend(preview_lines)

    return RenderedOutput(
        text="\n".join(lines),
        line_count=len(lines),
        severity=severity,
    )


# ---------------------------------------------------------------------------
# ErrorEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorEvent:
    """Event representing an agent loop or LLM error.

    Attributes:
        error_message: Human-readable error description.
        error_type:    Classification of the error (transient, permanent,
                       loop exhausted).
        iteration:     Current agent loop iteration (optional).
    """

    error_message: str
    error_type: ErrorType
    iteration: int | None = None

    def __post_init__(self) -> None:
        if not self.error_message or not self.error_message.strip():
            raise ValueError("error_message must not be empty")


def format_error(event: ErrorEvent, context: RenderContext) -> RenderedOutput:
    """Format an error event for terminal display.

    Transient errors are displayed as warnings (retriable). Permanent
    errors and loop exhaustion are displayed as errors.

    Args:
        event:   The error event to render.
        context: Rendering context with style configuration.

    Returns:
        RenderedOutput with styled text and appropriate severity.
    """
    config = context.style
    lines: list[str] = []

    is_transient = event.error_type is ErrorType.TRANSIENT
    color = Color.YELLOW if is_transient else Color.RED
    severity = EventSeverity.WARNING if is_transient else EventSeverity.ERROR
    type_label = _ERROR_TYPE_LABELS.get(event.error_type, "Error")

    # Header line: [!] Type: message
    styled_icon = styled(ICON_ERROR, color, config=config)
    styled_type = styled(type_label, color, Color.BOLD, config=config)
    header = f"{styled_icon} {styled_type}: {event.error_message}"

    # Append iteration context if available
    if event.iteration is not None:
        iter_str = styled(
            f"(iteration {event.iteration})",
            Color.DIM,
            config=config,
        )
        header = f"{header} {iter_str}"

    lines.append(header)

    return RenderedOutput(
        text="\n".join(lines),
        line_count=len(lines),
        severity=severity,
    )


# ---------------------------------------------------------------------------
# StatusChangeEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatusChangeEvent:
    """Event representing an agent loop state transition.

    Attributes:
        from_state:     The state being transitioned from.
        to_state:       The state being transitioned to.
        iteration:      Current agent loop iteration (1-based).
        max_iterations: Configured maximum iteration cap.
    """

    from_state: AgentLoopState
    to_state: AgentLoopState
    iteration: int
    max_iterations: int

    def __post_init__(self) -> None:
        if self.iteration < 1:
            raise ValueError(
                f"iteration must be >= 1, got {self.iteration}"
            )
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {self.max_iterations}"
            )


def format_status_change(
    event: StatusChangeEvent,
    context: RenderContext,
) -> RenderedOutput:
    """Format a state transition event for terminal display.

    Shows the iteration counter and the state transition with
    appropriate icons: running icon for non-terminal states, done
    icon for COMPLETE, error icon for ERROR.

    Args:
        event:   The status change event to render.
        context: Rendering context with style configuration.

    Returns:
        RenderedOutput with styled text and state-appropriate severity.
    """
    config = context.style
    is_complete = event.to_state is AgentLoopState.COMPLETE
    is_error = event.to_state is AgentLoopState.ERROR

    # Choose icon and severity based on target state
    if is_complete:
        icon = ICON_DONE
        color = Color.GREEN
        severity = EventSeverity.SUCCESS
    elif is_error:
        icon = ICON_ERROR
        color = Color.RED
        severity = EventSeverity.ERROR
    else:
        icon = ICON_RUNNING
        color = Color.CYAN
        severity = EventSeverity.INFO

    # Format: [icon] Iteration N/M: FROM --> TO
    styled_icon = styled(icon, color, config=config)
    iter_label = styled(
        f"Iteration {event.iteration}/{event.max_iterations}:",
        Color.BOLD,
        config=config,
    )
    from_label = styled(
        event.from_state.name,
        Color.DIM,
        config=config,
    )
    arrow = styled(ICON_ARROW, Color.DIM, config=config)
    to_label = styled(
        event.to_state.name,
        color,
        Color.BOLD,
        config=config,
    )

    line = f"{styled_icon} {iter_label} {from_label} {arrow} {to_label}"

    return RenderedOutput(
        text=line,
        line_count=1,
        severity=severity,
    )


# ---------------------------------------------------------------------------
# Concrete EventRenderer implementations
# ---------------------------------------------------------------------------


class ToolCallRenderer:
    """Concrete EventRenderer for tool call events.

    Delegates to ``format_tool_call()`` for formatting.
    """

    @property
    def event_type(self) -> str:
        """The event type string: ``"tool_call"``."""
        return "tool_call"

    def render(self, event: object, context: RenderContext) -> RenderedOutput:
        """Render a ToolCallEvent.

        Args:
            event:   A ToolCallEvent instance.
            context: Rendering context.

        Returns:
            RenderedOutput from format_tool_call().

        Raises:
            TypeError: If event is not a ToolCallEvent.
        """
        if not isinstance(event, ToolCallEvent):
            raise TypeError(
                f"ToolCallRenderer expects ToolCallEvent, "
                f"got {type(event).__name__}"
            )
        return format_tool_call(event, context)


class ApprovalPromptRenderer:
    """Concrete EventRenderer for approval prompt events.

    Delegates to ``format_approval_prompt()`` for formatting.
    """

    @property
    def event_type(self) -> str:
        """The event type string: ``"approval_prompt"``."""
        return "approval_prompt"

    def render(self, event: object, context: RenderContext) -> RenderedOutput:
        """Render an ApprovalPromptEvent.

        Args:
            event:   An ApprovalPromptEvent instance.
            context: Rendering context.

        Returns:
            RenderedOutput from format_approval_prompt().

        Raises:
            TypeError: If event is not an ApprovalPromptEvent.
        """
        if not isinstance(event, ApprovalPromptEvent):
            raise TypeError(
                f"ApprovalPromptRenderer expects ApprovalPromptEvent, "
                f"got {type(event).__name__}"
            )
        return format_approval_prompt(event, context)


class ObservationRenderer:
    """Concrete EventRenderer for observation (tool result) events.

    Delegates to ``format_observation()`` for formatting.
    """

    @property
    def event_type(self) -> str:
        """The event type string: ``"observation"``."""
        return "observation"

    def render(self, event: object, context: RenderContext) -> RenderedOutput:
        """Render an ObservationEvent.

        Args:
            event:   An ObservationEvent instance.
            context: Rendering context.

        Returns:
            RenderedOutput from format_observation().

        Raises:
            TypeError: If event is not an ObservationEvent.
        """
        if not isinstance(event, ObservationEvent):
            raise TypeError(
                f"ObservationRenderer expects ObservationEvent, "
                f"got {type(event).__name__}"
            )
        return format_observation(event, context)


class ErrorRenderer:
    """Concrete EventRenderer for error events.

    Delegates to ``format_error()`` for formatting.
    """

    @property
    def event_type(self) -> str:
        """The event type string: ``"error"``."""
        return "error"

    def render(self, event: object, context: RenderContext) -> RenderedOutput:
        """Render an ErrorEvent.

        Args:
            event:   An ErrorEvent instance.
            context: Rendering context.

        Returns:
            RenderedOutput from format_error().

        Raises:
            TypeError: If event is not an ErrorEvent.
        """
        if not isinstance(event, ErrorEvent):
            raise TypeError(
                f"ErrorRenderer expects ErrorEvent, "
                f"got {type(event).__name__}"
            )
        return format_error(event, context)


class StatusChangeRenderer:
    """Concrete EventRenderer for status change events.

    Delegates to ``format_status_change()`` for formatting.
    """

    @property
    def event_type(self) -> str:
        """The event type string: ``"status_change"``."""
        return "status_change"

    def render(self, event: object, context: RenderContext) -> RenderedOutput:
        """Render a StatusChangeEvent.

        Args:
            event:   A StatusChangeEvent instance.
            context: Rendering context.

        Returns:
            RenderedOutput from format_status_change().

        Raises:
            TypeError: If event is not a StatusChangeEvent.
        """
        if not isinstance(event, StatusChangeEvent):
            raise TypeError(
                f"StatusChangeRenderer expects StatusChangeEvent, "
                f"got {type(event).__name__}"
            )
        return format_status_change(event, context)
