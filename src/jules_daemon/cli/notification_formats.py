"""Format functions for daemon notification events.

Provides format functions and concrete ``EventRenderer`` implementations
for the three notification event types pushed by the daemon:

    ``CompletionNotification`` -> ``format_completion()``
        Displays a test run completion summary with exit status, duration,
        and parsed test outcome counts.

    ``AlertNotification``      -> ``format_alert()``
        Displays a severity-classified alert with title and message.

    ``HeartbeatNotification``  -> ``format_heartbeat()``
        Displays daemon liveness status with uptime and active run info.

These complement the agent loop event formatters in ``event_formats.py``.
Together they cover all event types that flow through the daemon's
notification subscription stream.

All format functions accept a notification payload object and a
``RenderContext``, and return a ``RenderedOutput`` with styled terminal
text. Concrete ``EventRenderer`` implementations wrap the format
functions for registry-based dispatch.

Usage::

    from jules_daemon.cli.notification_formats import (
        CompletionRenderer,
        format_completion,
    )
    from jules_daemon.cli.event_renderer import RenderContext
    from jules_daemon.protocol.notifications import CompletionNotification

    notification = CompletionNotification(
        run_id="run-abc",
        natural_language_command="Run pytest on auth",
        exit_status=0,
    )
    ctx = RenderContext()
    output = format_completion(notification, ctx)
    print(output.text)
"""

from __future__ import annotations

from jules_daemon.cli.event_renderer import (
    EventSeverity,
    RenderContext,
    RenderedOutput,
)
from jules_daemon.cli.styles import (
    ICON_DONE,
    ICON_ERROR,
    ICON_INFO,
    ICON_WARNING,
    Color,
    indent,
    styled,
)
from jules_daemon.protocol.notifications import (
    AlertNotification,
    CompletionNotification,
    HeartbeatNotification,
    NotificationSeverity,
)

__all__ = [
    "AlertRenderer",
    "CompletionRenderer",
    "HeartbeatRenderer",
    "format_alert",
    "format_completion",
    "format_heartbeat",
]


# ---------------------------------------------------------------------------
# Severity mapping helpers
# ---------------------------------------------------------------------------

_NOTIFICATION_SEVERITY_MAP: dict[NotificationSeverity, EventSeverity] = {
    NotificationSeverity.INFO: EventSeverity.INFO,
    NotificationSeverity.WARNING: EventSeverity.WARNING,
    NotificationSeverity.ERROR: EventSeverity.ERROR,
    NotificationSeverity.SUCCESS: EventSeverity.SUCCESS,
}

_NOTIFICATION_SEVERITY_ICONS: dict[NotificationSeverity, str] = {
    NotificationSeverity.INFO: ICON_INFO,
    NotificationSeverity.WARNING: ICON_WARNING,
    NotificationSeverity.ERROR: ICON_ERROR,
    NotificationSeverity.SUCCESS: ICON_DONE,
}

_NOTIFICATION_SEVERITY_COLORS: dict[NotificationSeverity, Color] = {
    NotificationSeverity.INFO: Color.CYAN,
    NotificationSeverity.WARNING: Color.YELLOW,
    NotificationSeverity.ERROR: Color.RED,
    NotificationSeverity.SUCCESS: Color.GREEN,
}


# ---------------------------------------------------------------------------
# format_completion
# ---------------------------------------------------------------------------


def format_completion(
    notification: CompletionNotification,
    context: RenderContext,
) -> RenderedOutput:
    """Format a completion notification for terminal display.

    Shows the run ID, original command, exit status, duration (if
    available), and parsed test outcome counts (if available).
    Success (exit_status == 0) uses green styling; failure uses red.

    Args:
        notification: The completion notification payload.
        context:      Rendering context with style configuration.

    Returns:
        RenderedOutput with styled text and appropriate severity.
    """
    config = context.style
    lines: list[str] = []

    is_success = notification.exit_status == 0
    color = Color.GREEN if is_success else Color.RED
    icon = ICON_DONE if is_success else ICON_ERROR
    severity = EventSeverity.SUCCESS if is_success else EventSeverity.ERROR
    status_word = "PASSED" if is_success else "FAILED"

    # Header: [+] Run run-abc: PASSED (exit 0)
    styled_icon = styled(icon, color, config=config)
    styled_label = styled("Run", Color.BOLD, config=config)
    styled_run_id = styled(notification.run_id, Color.BOLD, color, config=config)
    styled_status = styled(status_word, color, Color.BOLD, config=config)
    header = (
        f"{styled_icon} {styled_label} {styled_run_id}: "
        f"{styled_status} (exit {notification.exit_status})"
    )

    # Append duration if available
    if notification.duration_seconds is not None:
        duration_str = styled(
            f"in {notification.duration_seconds:.1f}s",
            Color.DIM,
            config=config,
        )
        header = f"{header} {duration_str}"

    lines.append(header)

    # Command line
    indent_width = config.indent_width
    cmd_line = indent(
        styled(
            f"Command: {notification.natural_language_command}",
            Color.DIM,
            config=config,
        ),
        level=2,
        width=indent_width,
    )
    lines.append(cmd_line)

    # Resolved shell command (verbose mode)
    if context.verbose and notification.resolved_shell:
        shell_line = indent(
            styled(
                f"Shell: {notification.resolved_shell}",
                Color.DIM,
                config=config,
            ),
            level=2,
            width=indent_width,
        )
        lines.append(shell_line)

    # Test outcome summary
    if notification.outcome is not None:
        outcome = notification.outcome
        passed = styled(str(outcome.tests_passed), Color.GREEN, config=config)
        failed_str = str(outcome.tests_failed)
        failed_color = Color.RED if outcome.tests_failed > 0 else Color.DIM
        failed = styled(failed_str, failed_color, config=config)
        skipped = styled(str(outcome.tests_skipped), Color.YELLOW, config=config)
        total = styled(str(outcome.tests_total), Color.BOLD, config=config)

        outcome_text = (
            f"Tests: {passed} passed, {failed} failed, "
            f"{skipped} skipped / {total} total"
        )
        outcome_line = indent(outcome_text, level=2, width=indent_width)
        lines.append(outcome_line)

    # Error message (if present)
    if notification.error_message:
        error_line = indent(
            styled(
                f"Error: {notification.error_message}",
                Color.RED,
                config=config,
            ),
            level=2,
            width=indent_width,
        )
        lines.append(error_line)

    return RenderedOutput(
        text="\n".join(lines),
        line_count=len(lines),
        severity=severity,
    )


# ---------------------------------------------------------------------------
# format_alert
# ---------------------------------------------------------------------------


def format_alert(
    notification: AlertNotification,
    context: RenderContext,
) -> RenderedOutput:
    """Format an alert notification for terminal display.

    Shows the alert title and message with severity-appropriate styling.
    In verbose mode, includes the run ID and details dictionary.

    Args:
        notification: The alert notification payload.
        context:      Rendering context with style configuration.

    Returns:
        RenderedOutput with styled text and severity-appropriate level.
    """
    config = context.style
    lines: list[str] = []

    severity = _NOTIFICATION_SEVERITY_MAP.get(
        notification.severity, EventSeverity.INFO
    )
    icon = _NOTIFICATION_SEVERITY_ICONS.get(
        notification.severity, ICON_INFO
    )
    color = _NOTIFICATION_SEVERITY_COLORS.get(
        notification.severity, Color.CYAN
    )

    # Header: [!] Alert: High failure rate detected
    styled_icon = styled(icon, color, config=config)
    styled_label = styled("Alert:", color, Color.BOLD, config=config)
    styled_title = styled(notification.title, Color.BOLD, config=config)
    header = f"{styled_icon} {styled_label} {styled_title}"

    # Append run ID if present
    if notification.run_id is not None:
        run_label = styled(
            f"(run: {notification.run_id})",
            Color.DIM,
            config=config,
        )
        header = f"{header} {run_label}"

    lines.append(header)

    # Message body
    indent_width = config.indent_width
    msg_line = indent(
        notification.message,
        level=2,
        width=indent_width,
    )
    lines.append(msg_line)

    # Verbose: show details dict
    if context.verbose and notification.details:
        for key, value in sorted(notification.details.items()):
            detail_text = f"{key}: {value}"
            detail_line = indent(
                styled(detail_text, Color.DIM, config=config),
                level=3,
                width=indent_width,
            )
            lines.append(detail_line)

    return RenderedOutput(
        text="\n".join(lines),
        line_count=len(lines),
        severity=severity,
    )


# ---------------------------------------------------------------------------
# format_heartbeat
# ---------------------------------------------------------------------------


def _format_uptime(seconds: float) -> str:
    """Format an uptime duration as a human-readable string.

    Args:
        seconds: Uptime in seconds.

    Returns:
        Formatted string like "2h 15m" or "45s".
    """
    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes = int(seconds // 60)
    if minutes < 60:
        remaining_secs = int(seconds % 60)
        return f"{minutes}m {remaining_secs}s"

    hours = minutes // 60
    remaining_mins = minutes % 60
    return f"{hours}h {remaining_mins}m"


def format_heartbeat(
    notification: HeartbeatNotification,
    context: RenderContext,
) -> RenderedOutput:
    """Format a heartbeat notification for terminal display.

    Shows daemon uptime, active run status, and queue depth.
    Heartbeats use DEBUG severity since they are routine liveness
    signals.

    Args:
        notification: The heartbeat notification payload.
        context:      Rendering context with style configuration.

    Returns:
        RenderedOutput with styled text and DEBUG severity.
    """
    config = context.style
    uptime_str = _format_uptime(notification.daemon_uptime_seconds)

    # Build status components
    parts: list[str] = [f"uptime {uptime_str}"]

    if notification.active_run_id is not None:
        parts.append(f"active: {notification.active_run_id}")
    else:
        parts.append("idle")

    if notification.queue_depth > 0:
        parts.append(f"queue: {notification.queue_depth}")

    status_text = ", ".join(parts)

    # Format: [i] Heartbeat: uptime 2h 15m, idle, queue: 0
    styled_icon = styled(ICON_INFO, Color.DIM, config=config)
    styled_label = styled("Heartbeat:", Color.DIM, config=config)
    styled_status = styled(status_text, Color.DIM, config=config)
    line = f"{styled_icon} {styled_label} {styled_status}"

    return RenderedOutput(
        text=line,
        line_count=1,
        severity=EventSeverity.DEBUG,
    )


# ---------------------------------------------------------------------------
# Concrete EventRenderer implementations
# ---------------------------------------------------------------------------


class CompletionRenderer:
    """Concrete EventRenderer for completion notification events.

    Delegates to ``format_completion()`` for formatting.
    """

    @property
    def event_type(self) -> str:
        """The event type string: ``"completion"``."""
        return "completion"

    def render(self, event: object, context: RenderContext) -> RenderedOutput:
        """Render a CompletionNotification.

        Args:
            event:   A CompletionNotification instance.
            context: Rendering context.

        Returns:
            RenderedOutput from format_completion().

        Raises:
            TypeError: If event is not a CompletionNotification.
        """
        if not isinstance(event, CompletionNotification):
            raise TypeError(
                f"CompletionRenderer expects CompletionNotification, "
                f"got {type(event).__name__}"
            )
        return format_completion(event, context)


class AlertRenderer:
    """Concrete EventRenderer for alert notification events.

    Delegates to ``format_alert()`` for formatting.
    """

    @property
    def event_type(self) -> str:
        """The event type string: ``"alert"``."""
        return "alert"

    def render(self, event: object, context: RenderContext) -> RenderedOutput:
        """Render an AlertNotification.

        Args:
            event:   An AlertNotification instance.
            context: Rendering context.

        Returns:
            RenderedOutput from format_alert().

        Raises:
            TypeError: If event is not an AlertNotification.
        """
        if not isinstance(event, AlertNotification):
            raise TypeError(
                f"AlertRenderer expects AlertNotification, "
                f"got {type(event).__name__}"
            )
        return format_alert(event, context)


class HeartbeatRenderer:
    """Concrete EventRenderer for heartbeat notification events.

    Delegates to ``format_heartbeat()`` for formatting.
    """

    @property
    def event_type(self) -> str:
        """The event type string: ``"heartbeat"``."""
        return "heartbeat"

    def render(self, event: object, context: RenderContext) -> RenderedOutput:
        """Render a HeartbeatNotification.

        Args:
            event:   A HeartbeatNotification instance.
            context: Rendering context.

        Returns:
            RenderedOutput from format_heartbeat().

        Raises:
            TypeError: If event is not a HeartbeatNotification.
        """
        if not isinstance(event, HeartbeatNotification):
            raise TypeError(
                f"HeartbeatRenderer expects HeartbeatNotification, "
                f"got {type(event).__name__}"
            )
        return format_heartbeat(event, context)
