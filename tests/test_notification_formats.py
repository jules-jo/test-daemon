"""Tests for notification event format functions and renderers.

Covers:
- format_completion: completion notification display
- format_alert: alert notification display with severity mapping
- format_heartbeat: heartbeat liveness signal display
- Concrete EventRenderer protocol compliance for each renderer
- Color enable/disable behavior for all formatters
- Verbose mode behavior for all formatters
- Uptime formatting helper
"""

from __future__ import annotations

import pytest

from jules_daemon.cli.event_renderer import (
    EventRenderer,
    EventSeverity,
    RenderContext,
    RenderedOutput,
)
from jules_daemon.cli.notification_formats import (
    AlertRenderer,
    CompletionRenderer,
    HeartbeatRenderer,
    _format_uptime,
    format_alert,
    format_completion,
    format_heartbeat,
)
from jules_daemon.cli.styles import StyleConfig
from jules_daemon.protocol.notifications import (
    AlertNotification,
    CompletionNotification,
    HeartbeatNotification,
    NotificationSeverity,
    TestOutcomeSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _no_color_ctx(*, verbose: bool = False) -> RenderContext:
    """Create a RenderContext with color disabled for assertion-friendly output."""
    return RenderContext(
        style=StyleConfig(color_enabled=False),
        verbose=verbose,
    )


def _color_ctx(*, verbose: bool = False) -> RenderContext:
    """Create a RenderContext with color enabled."""
    return RenderContext(
        style=StyleConfig(color_enabled=True),
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# format_completion tests
# ---------------------------------------------------------------------------


class TestFormatCompletion:
    """Tests for the format_completion function."""

    def test_success_basic(self) -> None:
        """Successful run shows PASSED and green severity."""
        notification = CompletionNotification(
            run_id="run-001",
            natural_language_command="Run pytest on auth",
            exit_status=0,
        )
        ctx = _no_color_ctx()
        output = format_completion(notification, ctx)

        assert "PASSED" in output.text
        assert "run-001" in output.text
        assert "exit 0" in output.text
        assert output.severity == EventSeverity.SUCCESS

    def test_failure_basic(self) -> None:
        """Failed run shows FAILED and error severity."""
        notification = CompletionNotification(
            run_id="run-002",
            natural_language_command="Run make test",
            exit_status=1,
        )
        ctx = _no_color_ctx()
        output = format_completion(notification, ctx)

        assert "FAILED" in output.text
        assert "run-002" in output.text
        assert "exit 1" in output.text
        assert output.severity == EventSeverity.ERROR

    def test_with_duration(self) -> None:
        """Duration is shown when available."""
        notification = CompletionNotification(
            run_id="run-003",
            natural_language_command="Run tests",
            exit_status=0,
            duration_seconds=42.5,
        )
        ctx = _no_color_ctx()
        output = format_completion(notification, ctx)

        assert "42.5s" in output.text

    def test_without_duration(self) -> None:
        """No duration info when duration is None."""
        notification = CompletionNotification(
            run_id="run-004",
            natural_language_command="Run tests",
            exit_status=0,
        )
        ctx = _no_color_ctx()
        output = format_completion(notification, ctx)

        assert "in " not in output.text.split("\n")[0] or "in" not in output.text

    def test_with_test_outcome(self) -> None:
        """Test outcome counts are displayed when available."""
        notification = CompletionNotification(
            run_id="run-005",
            natural_language_command="Run pytest",
            exit_status=0,
            outcome=TestOutcomeSummary(
                tests_passed=95,
                tests_failed=3,
                tests_skipped=2,
                tests_total=100,
            ),
        )
        ctx = _no_color_ctx()
        output = format_completion(notification, ctx)

        assert "95" in output.text
        assert "3" in output.text
        assert "2" in output.text
        assert "100" in output.text
        assert "passed" in output.text
        assert "failed" in output.text

    def test_with_error_message(self) -> None:
        """Error message is shown when present."""
        notification = CompletionNotification(
            run_id="run-006",
            natural_language_command="Run tests",
            exit_status=127,
            error_message="Command not found: pytest",
        )
        ctx = _no_color_ctx()
        output = format_completion(notification, ctx)

        assert "Command not found: pytest" in output.text

    def test_command_shown(self) -> None:
        """The original command is shown."""
        notification = CompletionNotification(
            run_id="run-007",
            natural_language_command="Run the auth tests please",
            exit_status=0,
        )
        ctx = _no_color_ctx()
        output = format_completion(notification, ctx)

        assert "Run the auth tests please" in output.text

    def test_resolved_shell_verbose(self) -> None:
        """Resolved shell command is shown in verbose mode."""
        notification = CompletionNotification(
            run_id="run-008",
            natural_language_command="Run auth tests",
            resolved_shell="cd /app && pytest tests/auth/",
            exit_status=0,
        )
        ctx = _no_color_ctx(verbose=True)
        output = format_completion(notification, ctx)

        assert "cd /app && pytest tests/auth/" in output.text

    def test_resolved_shell_hidden_non_verbose(self) -> None:
        """Resolved shell command is hidden in non-verbose mode."""
        notification = CompletionNotification(
            run_id="run-009",
            natural_language_command="Run auth tests",
            resolved_shell="cd /app && pytest tests/auth/",
            exit_status=0,
        )
        ctx = _no_color_ctx(verbose=False)
        output = format_completion(notification, ctx)

        assert "cd /app && pytest tests/auth/" not in output.text

    def test_line_count(self) -> None:
        """Line count matches actual lines in output."""
        notification = CompletionNotification(
            run_id="run-010",
            natural_language_command="Run tests",
            exit_status=0,
            outcome=TestOutcomeSummary(tests_total=10),
        )
        ctx = _no_color_ctx()
        output = format_completion(notification, ctx)

        actual_lines = output.text.count("\n") + 1
        assert output.line_count == actual_lines

    def test_returns_rendered_output(self) -> None:
        """Returns a RenderedOutput instance."""
        notification = CompletionNotification(
            run_id="run-011",
            natural_language_command="Run tests",
            exit_status=0,
        )
        ctx = _no_color_ctx()
        output = format_completion(notification, ctx)

        assert isinstance(output, RenderedOutput)
        assert isinstance(output.text, str)


# ---------------------------------------------------------------------------
# format_alert tests
# ---------------------------------------------------------------------------


class TestFormatAlert:
    """Tests for the format_alert function."""

    def test_info_alert(self) -> None:
        """Info-level alert shows info icon."""
        notification = AlertNotification(
            severity=NotificationSeverity.INFO,
            title="Connection established",
            message="SSH connection to remote host is up",
        )
        ctx = _no_color_ctx()
        output = format_alert(notification, ctx)

        assert "Connection established" in output.text
        assert "SSH connection" in output.text
        assert output.severity == EventSeverity.INFO

    def test_warning_alert(self) -> None:
        """Warning-level alert uses warning severity."""
        notification = AlertNotification(
            severity=NotificationSeverity.WARNING,
            title="High failure rate",
            message="50% of tests are failing",
        )
        ctx = _no_color_ctx()
        output = format_alert(notification, ctx)

        assert "High failure rate" in output.text
        assert "50% of tests" in output.text
        assert output.severity == EventSeverity.WARNING

    def test_error_alert(self) -> None:
        """Error-level alert uses error severity."""
        notification = AlertNotification(
            severity=NotificationSeverity.ERROR,
            title="SSH disconnect",
            message="Connection to remote host lost",
        )
        ctx = _no_color_ctx()
        output = format_alert(notification, ctx)

        assert "SSH disconnect" in output.text
        assert output.severity == EventSeverity.ERROR

    def test_success_alert(self) -> None:
        """Success-level alert uses success severity."""
        notification = AlertNotification(
            severity=NotificationSeverity.SUCCESS,
            title="All tests passed",
            message="100% pass rate achieved",
        )
        ctx = _no_color_ctx()
        output = format_alert(notification, ctx)

        assert "All tests passed" in output.text
        assert output.severity == EventSeverity.SUCCESS

    def test_with_run_id(self) -> None:
        """Run ID is shown when present."""
        notification = AlertNotification(
            severity=NotificationSeverity.WARNING,
            title="Slow test detected",
            message="test_login took 30s",
            run_id="run-abc",
        )
        ctx = _no_color_ctx()
        output = format_alert(notification, ctx)

        assert "run-abc" in output.text

    def test_without_run_id(self) -> None:
        """No run ID reference when not present."""
        notification = AlertNotification(
            severity=NotificationSeverity.INFO,
            title="System notice",
            message="Daemon memory usage normal",
        )
        ctx = _no_color_ctx()
        output = format_alert(notification, ctx)

        assert "run:" not in output.text

    def test_verbose_with_details(self) -> None:
        """Details dict is shown in verbose mode."""
        notification = AlertNotification(
            severity=NotificationSeverity.WARNING,
            title="Pattern match",
            message="SIGSEGV detected in output",
            details={"pattern": "SIGSEGV", "line_number": 42},
        )
        ctx = _no_color_ctx(verbose=True)
        output = format_alert(notification, ctx)

        assert "pattern: SIGSEGV" in output.text
        assert "line_number: 42" in output.text

    def test_non_verbose_hides_details(self) -> None:
        """Details dict is hidden in non-verbose mode."""
        notification = AlertNotification(
            severity=NotificationSeverity.WARNING,
            title="Pattern match",
            message="SIGSEGV detected",
            details={"pattern": "SIGSEGV"},
        )
        ctx = _no_color_ctx(verbose=False)
        output = format_alert(notification, ctx)

        assert "pattern: SIGSEGV" not in output.text

    def test_line_count(self) -> None:
        """Line count matches actual lines."""
        notification = AlertNotification(
            severity=NotificationSeverity.INFO,
            title="Notice",
            message="Something happened",
        )
        ctx = _no_color_ctx()
        output = format_alert(notification, ctx)

        actual_lines = output.text.count("\n") + 1
        assert output.line_count == actual_lines


# ---------------------------------------------------------------------------
# format_heartbeat tests
# ---------------------------------------------------------------------------


class TestFormatHeartbeat:
    """Tests for the format_heartbeat function."""

    def test_idle_heartbeat(self) -> None:
        """Idle heartbeat shows uptime and idle status."""
        notification = HeartbeatNotification(
            daemon_uptime_seconds=120.0,
        )
        ctx = _no_color_ctx()
        output = format_heartbeat(notification, ctx)

        assert "Heartbeat" in output.text
        assert "idle" in output.text
        assert output.severity == EventSeverity.DEBUG

    def test_active_run(self) -> None:
        """Active run is shown when present."""
        notification = HeartbeatNotification(
            daemon_uptime_seconds=3600.0,
            active_run_id="run-xyz",
        )
        ctx = _no_color_ctx()
        output = format_heartbeat(notification, ctx)

        assert "run-xyz" in output.text
        assert "idle" not in output.text

    def test_queue_depth(self) -> None:
        """Queue depth is shown when > 0."""
        notification = HeartbeatNotification(
            daemon_uptime_seconds=60.0,
            queue_depth=3,
        )
        ctx = _no_color_ctx()
        output = format_heartbeat(notification, ctx)

        assert "queue: 3" in output.text

    def test_zero_queue_hidden(self) -> None:
        """Queue depth is hidden when 0."""
        notification = HeartbeatNotification(
            daemon_uptime_seconds=60.0,
            queue_depth=0,
        )
        ctx = _no_color_ctx()
        output = format_heartbeat(notification, ctx)

        assert "queue:" not in output.text

    def test_single_line(self) -> None:
        """Heartbeats are always a single line."""
        notification = HeartbeatNotification(
            daemon_uptime_seconds=500.0,
        )
        ctx = _no_color_ctx()
        output = format_heartbeat(notification, ctx)

        assert output.line_count == 1


# ---------------------------------------------------------------------------
# _format_uptime tests
# ---------------------------------------------------------------------------


class TestFormatUptime:
    """Tests for the uptime formatting helper."""

    def test_seconds_only(self) -> None:
        assert _format_uptime(45.0) == "45s"

    def test_zero_seconds(self) -> None:
        assert _format_uptime(0.0) == "0s"

    def test_minutes_and_seconds(self) -> None:
        assert _format_uptime(130.0) == "2m 10s"

    def test_hours_and_minutes(self) -> None:
        assert _format_uptime(7500.0) == "2h 5m"

    def test_exactly_one_minute(self) -> None:
        assert _format_uptime(60.0) == "1m 0s"

    def test_exactly_one_hour(self) -> None:
        assert _format_uptime(3600.0) == "1h 0m"


# ---------------------------------------------------------------------------
# CompletionRenderer tests
# ---------------------------------------------------------------------------


class TestCompletionRenderer:
    """Tests for the CompletionRenderer EventRenderer implementation."""

    def test_satisfies_protocol(self) -> None:
        """CompletionRenderer satisfies the EventRenderer protocol."""
        renderer = CompletionRenderer()
        assert isinstance(renderer, EventRenderer)

    def test_event_type(self) -> None:
        renderer = CompletionRenderer()
        assert renderer.event_type == "completion"

    def test_render_valid_event(self) -> None:
        renderer = CompletionRenderer()
        notification = CompletionNotification(
            run_id="run-001",
            natural_language_command="Run tests",
            exit_status=0,
        )
        output = renderer.render(notification, _no_color_ctx())
        assert isinstance(output, RenderedOutput)
        assert "PASSED" in output.text

    def test_render_wrong_type_raises(self) -> None:
        renderer = CompletionRenderer()
        with pytest.raises(TypeError, match="CompletionRenderer expects"):
            renderer.render("not a notification", _no_color_ctx())


# ---------------------------------------------------------------------------
# AlertRenderer tests
# ---------------------------------------------------------------------------


class TestAlertRenderer:
    """Tests for the AlertRenderer EventRenderer implementation."""

    def test_satisfies_protocol(self) -> None:
        renderer = AlertRenderer()
        assert isinstance(renderer, EventRenderer)

    def test_event_type(self) -> None:
        renderer = AlertRenderer()
        assert renderer.event_type == "alert"

    def test_render_valid_event(self) -> None:
        renderer = AlertRenderer()
        notification = AlertNotification(
            severity=NotificationSeverity.WARNING,
            title="Test warning",
            message="Something is off",
        )
        output = renderer.render(notification, _no_color_ctx())
        assert isinstance(output, RenderedOutput)
        assert "Test warning" in output.text

    def test_render_wrong_type_raises(self) -> None:
        renderer = AlertRenderer()
        with pytest.raises(TypeError, match="AlertRenderer expects"):
            renderer.render("not a notification", _no_color_ctx())


# ---------------------------------------------------------------------------
# HeartbeatRenderer tests
# ---------------------------------------------------------------------------


class TestHeartbeatRenderer:
    """Tests for the HeartbeatRenderer EventRenderer implementation."""

    def test_satisfies_protocol(self) -> None:
        renderer = HeartbeatRenderer()
        assert isinstance(renderer, EventRenderer)

    def test_event_type(self) -> None:
        renderer = HeartbeatRenderer()
        assert renderer.event_type == "heartbeat"

    def test_render_valid_event(self) -> None:
        renderer = HeartbeatRenderer()
        notification = HeartbeatNotification(
            daemon_uptime_seconds=60.0,
        )
        output = renderer.render(notification, _no_color_ctx())
        assert isinstance(output, RenderedOutput)
        assert "Heartbeat" in output.text

    def test_render_wrong_type_raises(self) -> None:
        renderer = HeartbeatRenderer()
        with pytest.raises(TypeError, match="HeartbeatRenderer expects"):
            renderer.render("not a notification", _no_color_ctx())


# ---------------------------------------------------------------------------
# Color behavior tests
# ---------------------------------------------------------------------------


class TestColorBehavior:
    """Tests verifying ANSI codes presence/absence based on color config."""

    _ANSI_ESC = "\033["

    def test_completion_has_ansi_when_enabled(self) -> None:
        notification = CompletionNotification(
            run_id="r", natural_language_command="x", exit_status=0,
        )
        output = format_completion(notification, _color_ctx())
        assert self._ANSI_ESC in output.text

    def test_completion_no_ansi_when_disabled(self) -> None:
        notification = CompletionNotification(
            run_id="r", natural_language_command="x", exit_status=0,
        )
        output = format_completion(notification, _no_color_ctx())
        assert self._ANSI_ESC not in output.text

    def test_alert_has_ansi_when_enabled(self) -> None:
        notification = AlertNotification(
            severity=NotificationSeverity.WARNING,
            title="t", message="m",
        )
        output = format_alert(notification, _color_ctx())
        assert self._ANSI_ESC in output.text

    def test_alert_no_ansi_when_disabled(self) -> None:
        notification = AlertNotification(
            severity=NotificationSeverity.WARNING,
            title="t", message="m",
        )
        output = format_alert(notification, _no_color_ctx())
        assert self._ANSI_ESC not in output.text

    def test_heartbeat_has_ansi_when_enabled(self) -> None:
        notification = HeartbeatNotification(
            daemon_uptime_seconds=60.0,
        )
        output = format_heartbeat(notification, _color_ctx())
        assert self._ANSI_ESC in output.text

    def test_heartbeat_no_ansi_when_disabled(self) -> None:
        notification = HeartbeatNotification(
            daemon_uptime_seconds=60.0,
        )
        output = format_heartbeat(notification, _no_color_ctx())
        assert self._ANSI_ESC not in output.text
