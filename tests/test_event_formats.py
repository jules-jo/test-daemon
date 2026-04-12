"""Tests for per-event-type format functions.

Covers:
- Event data models (frozen dataclasses) for each event kind
- format_tool_call: tool invocation display
- format_approval_prompt: approval-required action prompt
- format_observation: tool result (success, error, denied, timeout)
- format_error: loop/LLM error display
- format_status_change: agent loop state transition display
- Concrete EventRenderer protocol compliance for each renderer
- Color enable/disable behavior for all formatters
- Verbose mode behavior for all formatters
- Iteration context display
"""

from __future__ import annotations

import pytest

from jules_daemon.agent.agent_loop import AgentLoopState
from jules_daemon.agent.tool_types import ToolResultStatus
from jules_daemon.cli.event_formats import (
    ApprovalPromptEvent,
    ApprovalPromptRenderer,
    ErrorEvent,
    ErrorRenderer,
    ErrorType,
    ObservationEvent,
    ObservationRenderer,
    StatusChangeEvent,
    StatusChangeRenderer,
    ToolCallEvent,
    ToolCallRenderer,
    format_approval_prompt,
    format_error,
    format_observation,
    format_status_change,
    format_tool_call,
)
from jules_daemon.cli.event_renderer import (
    EventRenderer,
    EventSeverity,
    RenderContext,
    RenderedOutput,
)
from jules_daemon.cli.styles import StyleConfig


# ---------------------------------------------------------------------------
# ToolCallEvent model
# ---------------------------------------------------------------------------


class TestToolCallEvent:
    """Tests for the ToolCallEvent frozen dataclass."""

    def test_creation(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={"slug": "auth-tests"},
            call_id="call_001",
        )
        assert event.tool_name == "read_wiki"
        assert event.arguments == {"slug": "auth-tests"}
        assert event.call_id == "call_001"

    def test_is_frozen(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={},
            call_id="call_001",
        )
        with pytest.raises(AttributeError):
            event.tool_name = "other"  # type: ignore[misc]

    def test_default_iteration(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={},
            call_id="call_001",
        )
        assert event.iteration is None

    def test_with_iteration(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={},
            call_id="call_001",
            iteration=3,
        )
        assert event.iteration == 3

    def test_empty_tool_name_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_name"):
            ToolCallEvent(
                tool_name="",
                arguments={},
                call_id="call_001",
            )

    def test_empty_call_id_raises(self) -> None:
        with pytest.raises(ValueError, match="call_id"):
            ToolCallEvent(
                tool_name="read_wiki",
                arguments={},
                call_id="",
            )


# ---------------------------------------------------------------------------
# ApprovalPromptEvent model
# ---------------------------------------------------------------------------


class TestApprovalPromptEvent:
    """Tests for the ApprovalPromptEvent frozen dataclass."""

    def test_creation(self) -> None:
        event = ApprovalPromptEvent(
            tool_name="propose_ssh_command",
            command="cd /opt/tests && pytest -x",
            call_id="call_002",
        )
        assert event.tool_name == "propose_ssh_command"
        assert event.command == "cd /opt/tests && pytest -x"
        assert event.call_id == "call_002"

    def test_is_frozen(self) -> None:
        event = ApprovalPromptEvent(
            tool_name="execute_ssh",
            command="ls",
            call_id="call_002",
        )
        with pytest.raises(AttributeError):
            event.command = "rm -rf /"  # type: ignore[misc]

    def test_empty_tool_name_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_name"):
            ApprovalPromptEvent(
                tool_name="",
                command="ls",
                call_id="call_002",
            )

    def test_empty_command_raises(self) -> None:
        with pytest.raises(ValueError, match="command"):
            ApprovalPromptEvent(
                tool_name="execute_ssh",
                command="",
                call_id="call_002",
            )


# ---------------------------------------------------------------------------
# ObservationEvent model
# ---------------------------------------------------------------------------


class TestObservationEvent:
    """Tests for the ObservationEvent frozen dataclass."""

    def test_creation_success(self) -> None:
        event = ObservationEvent(
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="Page content here",
            call_id="call_003",
        )
        assert event.tool_name == "read_wiki"
        assert event.status == ToolResultStatus.SUCCESS
        assert event.output == "Page content here"

    def test_creation_error(self) -> None:
        event = ObservationEvent(
            tool_name="execute_ssh",
            status=ToolResultStatus.ERROR,
            output="",
            call_id="call_003",
            error_message="Command failed with exit code 1",
        )
        assert event.status == ToolResultStatus.ERROR
        assert event.error_message == "Command failed with exit code 1"

    def test_is_frozen(self) -> None:
        event = ObservationEvent(
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="content",
            call_id="call_003",
        )
        with pytest.raises(AttributeError):
            event.output = "changed"  # type: ignore[misc]

    def test_optional_duration(self) -> None:
        event = ObservationEvent(
            tool_name="execute_ssh",
            status=ToolResultStatus.SUCCESS,
            output="ok",
            call_id="call_003",
            duration_seconds=12.5,
        )
        assert event.duration_seconds == 12.5

    def test_default_duration_is_none(self) -> None:
        event = ObservationEvent(
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="content",
            call_id="call_003",
        )
        assert event.duration_seconds is None


# ---------------------------------------------------------------------------
# ErrorEvent model
# ---------------------------------------------------------------------------


class TestErrorEvent:
    """Tests for the ErrorEvent frozen dataclass."""

    def test_creation(self) -> None:
        event = ErrorEvent(
            error_message="LLM timeout after 30s",
            error_type=ErrorType.TRANSIENT,
        )
        assert event.error_message == "LLM timeout after 30s"
        assert event.error_type == ErrorType.TRANSIENT

    def test_permanent_error(self) -> None:
        event = ErrorEvent(
            error_message="Malformed LLM response",
            error_type=ErrorType.PERMANENT,
        )
        assert event.error_type == ErrorType.PERMANENT

    def test_is_frozen(self) -> None:
        event = ErrorEvent(
            error_message="test",
            error_type=ErrorType.TRANSIENT,
        )
        with pytest.raises(AttributeError):
            event.error_message = "changed"  # type: ignore[misc]

    def test_empty_error_message_raises(self) -> None:
        with pytest.raises(ValueError, match="error_message"):
            ErrorEvent(
                error_message="",
                error_type=ErrorType.TRANSIENT,
            )

    def test_with_iteration(self) -> None:
        event = ErrorEvent(
            error_message="timeout",
            error_type=ErrorType.TRANSIENT,
            iteration=2,
        )
        assert event.iteration == 2

    def test_loop_exhausted_error(self) -> None:
        event = ErrorEvent(
            error_message="Max iterations reached",
            error_type=ErrorType.LOOP_EXHAUSTED,
        )
        assert event.error_type == ErrorType.LOOP_EXHAUSTED


# ---------------------------------------------------------------------------
# ErrorType enum
# ---------------------------------------------------------------------------


class TestErrorType:
    """Tests for the ErrorType enum."""

    def test_has_expected_members(self) -> None:
        values = {e.value for e in ErrorType}
        expected = {"transient", "permanent", "loop_exhausted"}
        assert values == expected


# ---------------------------------------------------------------------------
# StatusChangeEvent model
# ---------------------------------------------------------------------------


class TestStatusChangeEvent:
    """Tests for the StatusChangeEvent frozen dataclass."""

    def test_creation(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.THINKING,
            to_state=AgentLoopState.ACTING,
            iteration=2,
            max_iterations=5,
        )
        assert event.from_state == AgentLoopState.THINKING
        assert event.to_state == AgentLoopState.ACTING
        assert event.iteration == 2
        assert event.max_iterations == 5

    def test_is_frozen(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.THINKING,
            to_state=AgentLoopState.ACTING,
            iteration=1,
            max_iterations=5,
        )
        with pytest.raises(AttributeError):
            event.iteration = 3  # type: ignore[misc]

    def test_iteration_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="iteration"):
            StatusChangeEvent(
                from_state=AgentLoopState.THINKING,
                to_state=AgentLoopState.ACTING,
                iteration=0,
                max_iterations=5,
            )

    def test_max_iterations_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            StatusChangeEvent(
                from_state=AgentLoopState.THINKING,
                to_state=AgentLoopState.ACTING,
                iteration=1,
                max_iterations=0,
            )


# ---------------------------------------------------------------------------
# format_tool_call
# ---------------------------------------------------------------------------


class TestFormatToolCall:
    """Tests for the format_tool_call formatting function."""

    def test_returns_rendered_output(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={"slug": "auth-tests"},
            call_id="call_001",
        )
        ctx = RenderContext()
        result = format_tool_call(event, ctx)
        assert isinstance(result, RenderedOutput)

    def test_contains_tool_name(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={"slug": "auth-tests"},
            call_id="call_001",
        )
        ctx = RenderContext()
        result = format_tool_call(event, ctx)
        assert "read_wiki" in result.text

    def test_contains_tool_icon(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={},
            call_id="call_001",
        )
        ctx = RenderContext()
        result = format_tool_call(event, ctx)
        assert "[T]" in result.text

    def test_verbose_includes_arguments(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={"slug": "auth-tests", "section": "setup"},
            call_id="call_001",
        )
        ctx = RenderContext(verbose=True)
        result = format_tool_call(event, ctx)
        assert "slug" in result.text
        assert "auth-tests" in result.text

    def test_non_verbose_omits_argument_values(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={"slug": "auth-tests"},
            call_id="call_001",
        )
        ctx = RenderContext(verbose=False)
        result = format_tool_call(event, ctx)
        # Should show parameter names but may omit full values
        assert "read_wiki" in result.text

    def test_empty_arguments(self) -> None:
        event = ToolCallEvent(
            tool_name="check_remote_processes",
            arguments={},
            call_id="call_001",
        )
        ctx = RenderContext()
        result = format_tool_call(event, ctx)
        assert "check_remote_processes" in result.text

    def test_no_color_mode(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={},
            call_id="call_001",
        )
        style = StyleConfig(color_enabled=False)
        ctx = RenderContext(style=style)
        result = format_tool_call(event, ctx)
        assert "\033[" not in result.text

    def test_with_color_mode(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={},
            call_id="call_001",
        )
        style = StyleConfig(color_enabled=True)
        ctx = RenderContext(style=style)
        result = format_tool_call(event, ctx)
        assert "\033[" in result.text

    def test_with_iteration_context(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={},
            call_id="call_001",
            iteration=3,
        )
        ctx = RenderContext(current_iteration=3, max_iterations=5)
        result = format_tool_call(event, ctx)
        assert result.line_count > 0

    def test_severity_is_info(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={},
            call_id="call_001",
        )
        ctx = RenderContext()
        result = format_tool_call(event, ctx)
        assert result.severity == EventSeverity.INFO


# ---------------------------------------------------------------------------
# format_approval_prompt
# ---------------------------------------------------------------------------


class TestFormatApprovalPrompt:
    """Tests for the format_approval_prompt formatting function."""

    def test_returns_rendered_output(self) -> None:
        event = ApprovalPromptEvent(
            tool_name="propose_ssh_command",
            command="pytest -x",
            call_id="call_002",
        )
        ctx = RenderContext()
        result = format_approval_prompt(event, ctx)
        assert isinstance(result, RenderedOutput)

    def test_contains_command(self) -> None:
        event = ApprovalPromptEvent(
            tool_name="propose_ssh_command",
            command="cd /opt && pytest -x",
            call_id="call_002",
        )
        ctx = RenderContext()
        result = format_approval_prompt(event, ctx)
        assert "cd /opt && pytest -x" in result.text

    def test_contains_tool_name(self) -> None:
        event = ApprovalPromptEvent(
            tool_name="propose_ssh_command",
            command="pytest",
            call_id="call_002",
        )
        ctx = RenderContext()
        result = format_approval_prompt(event, ctx)
        assert "propose_ssh_command" in result.text

    def test_has_visual_framing(self) -> None:
        """Approval prompts should be visually distinct with box/border."""
        event = ApprovalPromptEvent(
            tool_name="execute_ssh",
            command="pytest -x",
            call_id="call_002",
        )
        ctx = RenderContext()
        result = format_approval_prompt(event, ctx)
        # Should have some box-like framing characters
        assert "+" in result.text or "|" in result.text

    def test_severity_is_warning(self) -> None:
        event = ApprovalPromptEvent(
            tool_name="propose_ssh_command",
            command="pytest",
            call_id="call_002",
        )
        ctx = RenderContext()
        result = format_approval_prompt(event, ctx)
        assert result.severity == EventSeverity.WARNING

    def test_no_color_mode(self) -> None:
        event = ApprovalPromptEvent(
            tool_name="propose_ssh_command",
            command="pytest",
            call_id="call_002",
        )
        style = StyleConfig(color_enabled=False)
        ctx = RenderContext(style=style)
        result = format_approval_prompt(event, ctx)
        assert "\033[" not in result.text

    def test_multiline_output(self) -> None:
        event = ApprovalPromptEvent(
            tool_name="propose_ssh_command",
            command="pytest -x",
            call_id="call_002",
        )
        ctx = RenderContext()
        result = format_approval_prompt(event, ctx)
        assert result.line_count >= 3  # at least top + content + bottom


# ---------------------------------------------------------------------------
# format_observation
# ---------------------------------------------------------------------------


class TestFormatObservation:
    """Tests for the format_observation formatting function."""

    def test_returns_rendered_output(self) -> None:
        event = ObservationEvent(
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="Page content",
            call_id="call_003",
        )
        ctx = RenderContext()
        result = format_observation(event, ctx)
        assert isinstance(result, RenderedOutput)

    def test_success_contains_tool_name(self) -> None:
        event = ObservationEvent(
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="Page content",
            call_id="call_003",
        )
        ctx = RenderContext()
        result = format_observation(event, ctx)
        assert "read_wiki" in result.text

    def test_success_contains_done_icon(self) -> None:
        event = ObservationEvent(
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="content",
            call_id="call_003",
        )
        ctx = RenderContext()
        result = format_observation(event, ctx)
        assert "[+]" in result.text

    def test_error_contains_error_icon(self) -> None:
        event = ObservationEvent(
            tool_name="execute_ssh",
            status=ToolResultStatus.ERROR,
            output="",
            call_id="call_003",
            error_message="Command failed",
        )
        ctx = RenderContext()
        result = format_observation(event, ctx)
        assert "[!]" in result.text

    def test_error_contains_error_message(self) -> None:
        event = ObservationEvent(
            tool_name="execute_ssh",
            status=ToolResultStatus.ERROR,
            output="",
            call_id="call_003",
            error_message="Command failed with exit code 1",
        )
        ctx = RenderContext()
        result = format_observation(event, ctx)
        assert "Command failed" in result.text

    def test_denied_contains_reject_icon(self) -> None:
        event = ObservationEvent(
            tool_name="execute_ssh",
            status=ToolResultStatus.DENIED,
            output="",
            call_id="call_003",
            error_message="User denied",
        )
        ctx = RenderContext()
        result = format_observation(event, ctx)
        assert "[N]" in result.text

    def test_timeout_contains_warning_icon(self) -> None:
        event = ObservationEvent(
            tool_name="execute_ssh",
            status=ToolResultStatus.TIMEOUT,
            output="",
            call_id="call_003",
            error_message="Timed out",
        )
        ctx = RenderContext()
        result = format_observation(event, ctx)
        assert "[*]" in result.text

    def test_verbose_includes_output_preview(self) -> None:
        event = ObservationEvent(
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="Line 1\nLine 2\nLine 3",
            call_id="call_003",
        )
        ctx = RenderContext(verbose=True)
        result = format_observation(event, ctx)
        assert "Line 1" in result.text

    def test_duration_displayed_when_present(self) -> None:
        event = ObservationEvent(
            tool_name="execute_ssh",
            status=ToolResultStatus.SUCCESS,
            output="ok",
            call_id="call_003",
            duration_seconds=12.5,
        )
        ctx = RenderContext()
        result = format_observation(event, ctx)
        assert "12.5" in result.text

    def test_severity_success(self) -> None:
        event = ObservationEvent(
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="content",
            call_id="call_003",
        )
        ctx = RenderContext()
        result = format_observation(event, ctx)
        assert result.severity == EventSeverity.SUCCESS

    def test_severity_error(self) -> None:
        event = ObservationEvent(
            tool_name="execute_ssh",
            status=ToolResultStatus.ERROR,
            output="",
            call_id="call_003",
            error_message="fail",
        )
        ctx = RenderContext()
        result = format_observation(event, ctx)
        assert result.severity == EventSeverity.ERROR

    def test_severity_denied(self) -> None:
        event = ObservationEvent(
            tool_name="execute_ssh",
            status=ToolResultStatus.DENIED,
            output="",
            call_id="call_003",
            error_message="denied",
        )
        ctx = RenderContext()
        result = format_observation(event, ctx)
        assert result.severity == EventSeverity.ERROR

    def test_severity_timeout(self) -> None:
        event = ObservationEvent(
            tool_name="execute_ssh",
            status=ToolResultStatus.TIMEOUT,
            output="",
            call_id="call_003",
            error_message="timeout",
        )
        ctx = RenderContext()
        result = format_observation(event, ctx)
        assert result.severity == EventSeverity.WARNING

    def test_no_color_mode(self) -> None:
        event = ObservationEvent(
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="content",
            call_id="call_003",
        )
        style = StyleConfig(color_enabled=False)
        ctx = RenderContext(style=style)
        result = format_observation(event, ctx)
        assert "\033[" not in result.text


# ---------------------------------------------------------------------------
# format_error
# ---------------------------------------------------------------------------


class TestFormatError:
    """Tests for the format_error formatting function."""

    def test_returns_rendered_output(self) -> None:
        event = ErrorEvent(
            error_message="LLM timeout",
            error_type=ErrorType.TRANSIENT,
        )
        ctx = RenderContext()
        result = format_error(event, ctx)
        assert isinstance(result, RenderedOutput)

    def test_contains_error_message(self) -> None:
        event = ErrorEvent(
            error_message="LLM timeout after 30s",
            error_type=ErrorType.TRANSIENT,
        )
        ctx = RenderContext()
        result = format_error(event, ctx)
        assert "LLM timeout after 30s" in result.text

    def test_contains_error_icon(self) -> None:
        event = ErrorEvent(
            error_message="something failed",
            error_type=ErrorType.PERMANENT,
        )
        ctx = RenderContext()
        result = format_error(event, ctx)
        assert "[!]" in result.text

    def test_transient_shows_type(self) -> None:
        event = ErrorEvent(
            error_message="timeout",
            error_type=ErrorType.TRANSIENT,
        )
        ctx = RenderContext()
        result = format_error(event, ctx)
        text_lower = result.text.lower()
        assert "transient" in text_lower

    def test_permanent_shows_type(self) -> None:
        event = ErrorEvent(
            error_message="malformed",
            error_type=ErrorType.PERMANENT,
        )
        ctx = RenderContext()
        result = format_error(event, ctx)
        text_lower = result.text.lower()
        assert "permanent" in text_lower

    def test_loop_exhausted_shows_type(self) -> None:
        event = ErrorEvent(
            error_message="Max iterations reached",
            error_type=ErrorType.LOOP_EXHAUSTED,
        )
        ctx = RenderContext()
        result = format_error(event, ctx)
        text_lower = result.text.lower()
        assert "loop" in text_lower or "exhausted" in text_lower

    def test_severity_is_error(self) -> None:
        event = ErrorEvent(
            error_message="test error",
            error_type=ErrorType.PERMANENT,
        )
        ctx = RenderContext()
        result = format_error(event, ctx)
        assert result.severity == EventSeverity.ERROR

    def test_transient_severity_is_warning(self) -> None:
        event = ErrorEvent(
            error_message="timeout",
            error_type=ErrorType.TRANSIENT,
        )
        ctx = RenderContext()
        result = format_error(event, ctx)
        assert result.severity == EventSeverity.WARNING

    def test_no_color_mode(self) -> None:
        event = ErrorEvent(
            error_message="test error",
            error_type=ErrorType.PERMANENT,
        )
        style = StyleConfig(color_enabled=False)
        ctx = RenderContext(style=style)
        result = format_error(event, ctx)
        assert "\033[" not in result.text

    def test_with_iteration(self) -> None:
        event = ErrorEvent(
            error_message="timeout",
            error_type=ErrorType.TRANSIENT,
            iteration=3,
        )
        ctx = RenderContext(current_iteration=3, max_iterations=5)
        result = format_error(event, ctx)
        assert "3" in result.text


# ---------------------------------------------------------------------------
# format_status_change
# ---------------------------------------------------------------------------


class TestFormatStatusChange:
    """Tests for the format_status_change formatting function."""

    def test_returns_rendered_output(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.THINKING,
            to_state=AgentLoopState.ACTING,
            iteration=1,
            max_iterations=5,
        )
        ctx = RenderContext()
        result = format_status_change(event, ctx)
        assert isinstance(result, RenderedOutput)

    def test_contains_state_names(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.THINKING,
            to_state=AgentLoopState.ACTING,
            iteration=1,
            max_iterations=5,
        )
        ctx = RenderContext()
        result = format_status_change(event, ctx)
        text_upper = result.text.upper()
        assert "THINKING" in text_upper
        assert "ACTING" in text_upper

    def test_contains_iteration_counter(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.THINKING,
            to_state=AgentLoopState.ACTING,
            iteration=2,
            max_iterations=5,
        )
        ctx = RenderContext()
        result = format_status_change(event, ctx)
        assert "2" in result.text
        assert "5" in result.text

    def test_contains_arrow_indicator(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.THINKING,
            to_state=AgentLoopState.ACTING,
            iteration=1,
            max_iterations=5,
        )
        ctx = RenderContext()
        result = format_status_change(event, ctx)
        assert "-->" in result.text

    def test_terminal_state_complete(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.OBSERVING,
            to_state=AgentLoopState.COMPLETE,
            iteration=3,
            max_iterations=5,
        )
        ctx = RenderContext()
        result = format_status_change(event, ctx)
        text_upper = result.text.upper()
        assert "COMPLETE" in text_upper

    def test_terminal_state_error(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.THINKING,
            to_state=AgentLoopState.ERROR,
            iteration=2,
            max_iterations=5,
        )
        ctx = RenderContext()
        result = format_status_change(event, ctx)
        text_upper = result.text.upper()
        assert "ERROR" in text_upper

    def test_severity_info_for_normal_transition(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.THINKING,
            to_state=AgentLoopState.ACTING,
            iteration=1,
            max_iterations=5,
        )
        ctx = RenderContext()
        result = format_status_change(event, ctx)
        assert result.severity == EventSeverity.INFO

    def test_severity_success_for_complete(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.OBSERVING,
            to_state=AgentLoopState.COMPLETE,
            iteration=3,
            max_iterations=5,
        )
        ctx = RenderContext()
        result = format_status_change(event, ctx)
        assert result.severity == EventSeverity.SUCCESS

    def test_severity_error_for_error_state(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.THINKING,
            to_state=AgentLoopState.ERROR,
            iteration=2,
            max_iterations=5,
        )
        ctx = RenderContext()
        result = format_status_change(event, ctx)
        assert result.severity == EventSeverity.ERROR

    def test_no_color_mode(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.THINKING,
            to_state=AgentLoopState.ACTING,
            iteration=1,
            max_iterations=5,
        )
        style = StyleConfig(color_enabled=False)
        ctx = RenderContext(style=style)
        result = format_status_change(event, ctx)
        assert "\033[" not in result.text

    def test_running_icon_for_non_terminal(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.THINKING,
            to_state=AgentLoopState.ACTING,
            iteration=1,
            max_iterations=5,
        )
        ctx = RenderContext()
        result = format_status_change(event, ctx)
        assert "[~]" in result.text

    def test_done_icon_for_complete(self) -> None:
        event = StatusChangeEvent(
            from_state=AgentLoopState.OBSERVING,
            to_state=AgentLoopState.COMPLETE,
            iteration=3,
            max_iterations=5,
        )
        ctx = RenderContext()
        result = format_status_change(event, ctx)
        assert "[+]" in result.text


# ---------------------------------------------------------------------------
# EventRenderer protocol compliance
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Edge cases: defensive copy, whitespace output, renderer type guards
# ---------------------------------------------------------------------------


class TestToolCallEventDefensiveCopy:
    """Tests that ToolCallEvent makes a defensive copy of arguments."""

    def test_external_mutation_does_not_affect_event(self) -> None:
        original = {"slug": "auth"}
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments=original,
            call_id="call_001",
        )
        original["injected"] = "payload"
        assert "injected" not in event.arguments

    def test_returned_arguments_match_input(self) -> None:
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={"slug": "auth", "section": "setup"},
            call_id="call_001",
        )
        assert event.arguments == {"slug": "auth", "section": "setup"}


class TestObservationWhitespaceOutput:
    """Tests for whitespace-only output in verbose mode."""

    def test_verbose_whitespace_only_output_no_preview(self) -> None:
        event = ObservationEvent(
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="   ",
            call_id="call_003",
        )
        ctx = RenderContext(verbose=True)
        result = format_observation(event, ctx)
        assert result.line_count == 1  # only the header line

    def test_verbose_empty_output_no_preview(self) -> None:
        event = ObservationEvent(
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="",
            call_id="call_003",
        )
        ctx = RenderContext(verbose=True)
        result = format_observation(event, ctx)
        assert result.line_count == 1


class TestRendererTypeGuards:
    """Tests that renderer type guards raise on wrong event type."""

    def test_tool_call_renderer_rejects_wrong_type(self) -> None:
        renderer = ToolCallRenderer()
        ctx = RenderContext()
        with pytest.raises(TypeError, match="ToolCallRenderer"):
            renderer.render("not an event", ctx)

    def test_approval_prompt_renderer_rejects_wrong_type(self) -> None:
        renderer = ApprovalPromptRenderer()
        ctx = RenderContext()
        with pytest.raises(TypeError, match="ApprovalPromptRenderer"):
            renderer.render("not an event", ctx)

    def test_observation_renderer_rejects_wrong_type(self) -> None:
        renderer = ObservationRenderer()
        ctx = RenderContext()
        with pytest.raises(TypeError, match="ObservationRenderer"):
            renderer.render("not an event", ctx)

    def test_error_renderer_rejects_wrong_type(self) -> None:
        renderer = ErrorRenderer()
        ctx = RenderContext()
        with pytest.raises(TypeError, match="ErrorRenderer"):
            renderer.render("not an event", ctx)

    def test_status_change_renderer_rejects_wrong_type(self) -> None:
        renderer = StatusChangeRenderer()
        ctx = RenderContext()
        with pytest.raises(TypeError, match="StatusChangeRenderer"):
            renderer.render("not an event", ctx)


# ---------------------------------------------------------------------------
# EventRenderer protocol compliance
# ---------------------------------------------------------------------------


class TestToolCallRendererProtocol:
    """Tests that ToolCallRenderer satisfies the EventRenderer protocol."""

    def test_satisfies_protocol(self) -> None:
        renderer = ToolCallRenderer()
        assert isinstance(renderer, EventRenderer)

    def test_event_type(self) -> None:
        renderer = ToolCallRenderer()
        assert renderer.event_type == "tool_call"

    def test_render_returns_rendered_output(self) -> None:
        renderer = ToolCallRenderer()
        event = ToolCallEvent(
            tool_name="read_wiki",
            arguments={},
            call_id="call_001",
        )
        ctx = RenderContext()
        result = renderer.render(event, ctx)
        assert isinstance(result, RenderedOutput)


class TestApprovalPromptRendererProtocol:
    """Tests that ApprovalPromptRenderer satisfies the EventRenderer protocol."""

    def test_satisfies_protocol(self) -> None:
        renderer = ApprovalPromptRenderer()
        assert isinstance(renderer, EventRenderer)

    def test_event_type(self) -> None:
        renderer = ApprovalPromptRenderer()
        assert renderer.event_type == "approval_prompt"

    def test_render_returns_rendered_output(self) -> None:
        renderer = ApprovalPromptRenderer()
        event = ApprovalPromptEvent(
            tool_name="propose_ssh_command",
            command="pytest",
            call_id="call_002",
        )
        ctx = RenderContext()
        result = renderer.render(event, ctx)
        assert isinstance(result, RenderedOutput)


class TestObservationRendererProtocol:
    """Tests that ObservationRenderer satisfies the EventRenderer protocol."""

    def test_satisfies_protocol(self) -> None:
        renderer = ObservationRenderer()
        assert isinstance(renderer, EventRenderer)

    def test_event_type(self) -> None:
        renderer = ObservationRenderer()
        assert renderer.event_type == "observation"

    def test_render_returns_rendered_output(self) -> None:
        renderer = ObservationRenderer()
        event = ObservationEvent(
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="content",
            call_id="call_003",
        )
        ctx = RenderContext()
        result = renderer.render(event, ctx)
        assert isinstance(result, RenderedOutput)


class TestErrorRendererProtocol:
    """Tests that ErrorRenderer satisfies the EventRenderer protocol."""

    def test_satisfies_protocol(self) -> None:
        renderer = ErrorRenderer()
        assert isinstance(renderer, EventRenderer)

    def test_event_type(self) -> None:
        renderer = ErrorRenderer()
        assert renderer.event_type == "error"

    def test_render_returns_rendered_output(self) -> None:
        renderer = ErrorRenderer()
        event = ErrorEvent(
            error_message="test error",
            error_type=ErrorType.PERMANENT,
        )
        ctx = RenderContext()
        result = renderer.render(event, ctx)
        assert isinstance(result, RenderedOutput)


class TestStatusChangeRendererProtocol:
    """Tests that StatusChangeRenderer satisfies the EventRenderer protocol."""

    def test_satisfies_protocol(self) -> None:
        renderer = StatusChangeRenderer()
        assert isinstance(renderer, EventRenderer)

    def test_event_type(self) -> None:
        renderer = StatusChangeRenderer()
        assert renderer.event_type == "status_change"

    def test_render_returns_rendered_output(self) -> None:
        renderer = StatusChangeRenderer()
        event = StatusChangeEvent(
            from_state=AgentLoopState.THINKING,
            to_state=AgentLoopState.ACTING,
            iteration=1,
            max_iterations=5,
        )
        ctx = RenderContext()
        result = renderer.render(event, ctx)
        assert isinstance(result, RenderedOutput)
