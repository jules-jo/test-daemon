"""Tests for the terminal-based editable SSH command confirmation prompt.

Covers data models (ConfirmationRequest, ConfirmationResult, Decision),
rendering logic, user interaction flows (approve, reject, edit-then-approve),
input validation, edge cases, IO abstraction for testability, and
DefaultTerminalIO with injected streams.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import pytest

from jules_daemon.cli.confirmation import (
    ConfirmationRequest,
    ConfirmationResult,
    Decision,
    DefaultTerminalIO,
    TerminalIO,
    confirm_ssh_command,
    render_confirmation_display,
)
from jules_daemon.llm.command_context import CommandContext, RiskLevel
from jules_daemon.ssh.command import SSHCommand
from jules_daemon.wiki.models import SSHTarget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_context(
    *,
    command: str = "pytest -v",
    explanation: str = "Run the test suite with verbose output",
    risk_level: RiskLevel = RiskLevel.LOW,
    affected_paths: tuple[str, ...] = ("/opt/app/tests",),
    risk_factors: tuple[str, ...] = (),
    safe_to_execute: bool = True,
) -> CommandContext:
    return CommandContext(
        command=command,
        explanation=explanation,
        risk_level=risk_level,
        affected_paths=affected_paths,
        risk_factors=risk_factors,
        safe_to_execute=safe_to_execute,
    )


def _make_ssh_command(
    *,
    command: str = "pytest -v",
    working_directory: str | None = "/opt/app",
    timeout: int = 300,
) -> SSHCommand:
    return SSHCommand(
        command=command,
        working_directory=working_directory,
        timeout=timeout,
    )


def _make_target(
    *,
    host: str = "test.example.com",
    user: str = "deploy",
    port: int = 22,
) -> SSHTarget:
    return SSHTarget(host=host, user=user, port=port)


_SENTINEL = object()


def _make_request(
    *,
    ssh_command: SSHCommand | None = None,
    context: CommandContext | None = None,
    target: SSHTarget | None | object = _SENTINEL,
) -> ConfirmationRequest:
    resolved_target: SSHTarget | None
    if target is _SENTINEL:
        resolved_target = _make_target()
    else:
        resolved_target = target  # type: ignore[assignment]
    return ConfirmationRequest(
        ssh_command=ssh_command or _make_ssh_command(),
        context=context or _make_context(),
        target=resolved_target,
    )


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------


class TestDecision:
    """Decision enum covers all user choices."""

    def test_approve_value(self) -> None:
        assert Decision.APPROVE.value == "approve"

    def test_reject_value(self) -> None:
        assert Decision.REJECT.value == "reject"

    def test_all_decisions(self) -> None:
        values = {d.value for d in Decision}
        assert values == {"approve", "reject"}


# ---------------------------------------------------------------------------
# ConfirmationRequest immutability
# ---------------------------------------------------------------------------


class TestConfirmationRequest:
    """ConfirmationRequest is a frozen data container."""

    def test_construction(self) -> None:
        req = _make_request()
        assert req.ssh_command.command == "pytest -v"
        assert req.context.explanation == "Run the test suite with verbose output"
        assert req.target.host == "test.example.com"

    def test_immutability(self) -> None:
        req = _make_request()
        with pytest.raises(AttributeError):
            req.ssh_command = _make_ssh_command(command="other")  # type: ignore[misc]

    def test_without_target(self) -> None:
        req = ConfirmationRequest(
            ssh_command=_make_ssh_command(),
            context=_make_context(),
            target=None,
        )
        assert req.target is None


# ---------------------------------------------------------------------------
# ConfirmationResult
# ---------------------------------------------------------------------------


class TestConfirmationResult:
    """ConfirmationResult captures the user's final decision."""

    def test_approved_with_original_command(self) -> None:
        result = ConfirmationResult(
            decision=Decision.APPROVE,
            final_command=_make_ssh_command(),
            was_edited=False,
        )
        assert result.decision == Decision.APPROVE
        assert result.final_command.command == "pytest -v"
        assert result.was_edited is False

    def test_approved_with_edited_command(self) -> None:
        edited = _make_ssh_command(command="pytest -v --tb=short")
        result = ConfirmationResult(
            decision=Decision.APPROVE,
            final_command=edited,
            was_edited=True,
        )
        assert result.decision == Decision.APPROVE
        assert result.final_command.command == "pytest -v --tb=short"
        assert result.was_edited is True

    def test_rejected(self) -> None:
        result = ConfirmationResult(
            decision=Decision.REJECT,
            final_command=_make_ssh_command(),
            was_edited=False,
        )
        assert result.decision == Decision.REJECT

    def test_immutability(self) -> None:
        result = ConfirmationResult(
            decision=Decision.APPROVE,
            final_command=_make_ssh_command(),
            was_edited=False,
        )
        with pytest.raises(AttributeError):
            result.decision = Decision.REJECT  # type: ignore[misc]

    def test_is_approved_property(self) -> None:
        approved = ConfirmationResult(
            decision=Decision.APPROVE,
            final_command=_make_ssh_command(),
            was_edited=False,
        )
        rejected = ConfirmationResult(
            decision=Decision.REJECT,
            final_command=_make_ssh_command(),
            was_edited=False,
        )
        assert approved.is_approved is True
        assert rejected.is_approved is False


# ---------------------------------------------------------------------------
# Render confirmation display
# ---------------------------------------------------------------------------


class TestRenderConfirmationDisplay:
    """The render function produces a human-readable command context display."""

    def test_contains_command(self) -> None:
        req = _make_request()
        output = render_confirmation_display(req)
        assert "pytest -v" in output

    def test_contains_explanation(self) -> None:
        req = _make_request()
        output = render_confirmation_display(req)
        assert "Run the test suite with verbose output" in output

    def test_contains_risk_level(self) -> None:
        req = _make_request(
            context=_make_context(risk_level=RiskLevel.HIGH),
        )
        output = render_confirmation_display(req)
        assert "HIGH" in output

    def test_contains_target_info(self) -> None:
        req = _make_request()
        output = render_confirmation_display(req)
        assert "test.example.com" in output
        assert "deploy" in output

    def test_contains_affected_paths(self) -> None:
        req = _make_request(
            context=_make_context(
                affected_paths=("/opt/app/tests", "/opt/app/src"),
            ),
        )
        output = render_confirmation_display(req)
        assert "/opt/app/tests" in output
        assert "/opt/app/src" in output

    def test_contains_working_directory(self) -> None:
        req = _make_request(
            ssh_command=_make_ssh_command(working_directory="/home/deploy/project"),
        )
        output = render_confirmation_display(req)
        assert "/home/deploy/project" in output

    def test_contains_timeout(self) -> None:
        req = _make_request(
            ssh_command=_make_ssh_command(timeout=600),
        )
        output = render_confirmation_display(req)
        assert "600" in output

    def test_contains_risk_factors(self) -> None:
        req = _make_request(
            context=_make_context(
                risk_level=RiskLevel.HIGH,
                risk_factors=("Modifies configuration files", "Writes to disk"),
            ),
        )
        output = render_confirmation_display(req)
        assert "Modifies configuration files" in output
        assert "Writes to disk" in output

    def test_no_risk_factors_omits_section(self) -> None:
        req = _make_request(
            context=_make_context(risk_factors=()),
        )
        output = render_confirmation_display(req)
        assert "Risk factors" not in output

    def test_no_target_omits_target_section(self) -> None:
        req = _make_request(target=None)
        output = render_confirmation_display(req)
        assert "Target" not in output

    def test_critical_risk_contains_warning(self) -> None:
        req = _make_request(
            context=_make_context(risk_level=RiskLevel.CRITICAL),
        )
        output = render_confirmation_display(req)
        assert "CRITICAL" in output
        assert "WARNING" in output or "CAUTION" in output

    def test_no_affected_paths_omits_section(self) -> None:
        req = _make_request(
            context=_make_context(affected_paths=()),
        )
        output = render_confirmation_display(req)
        assert "Affected paths" not in output

    def test_contains_action_instructions(self) -> None:
        req = _make_request()
        output = render_confirmation_display(req)
        # Must show the user how to approve, reject, or edit
        assert "[A]" in output or "approve" in output.lower()
        assert "[R]" in output or "reject" in output.lower()
        assert "[E]" in output or "edit" in output.lower()


# ---------------------------------------------------------------------------
# TerminalIO abstraction
# ---------------------------------------------------------------------------


class FakeTerminalIO(TerminalIO):
    """Fake terminal IO for testing user interaction flows."""

    def __init__(self, inputs: list[str]) -> None:
        self._inputs = list(inputs)
        self._input_index = 0
        self._output = StringIO()

    def write(self, text: str) -> None:
        self._output.write(text)

    def read_line(self, prompt: str = "") -> str:
        if self._input_index >= len(self._inputs):
            raise EOFError("No more test input available")
        self._output.write(prompt)
        value = self._inputs[self._input_index]
        self._input_index += 1
        return value

    def read_editable(self, prompt: str, prefill: str) -> str:
        if self._input_index >= len(self._inputs):
            raise EOFError("No more test input available")
        self._output.write(prompt)
        value = self._inputs[self._input_index]
        self._input_index += 1
        # Empty input means keep the prefill value
        if value == "":
            return prefill
        return value

    @property
    def output_text(self) -> str:
        return self._output.getvalue()


# ---------------------------------------------------------------------------
# confirm_ssh_command: approve flow
# ---------------------------------------------------------------------------


class TestConfirmApproveFlow:
    """User types 'a' or 'y' to approve the command as-is."""

    def test_approve_with_a(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE
        assert result.final_command.command == "pytest -v"
        assert result.was_edited is False

    def test_approve_with_y(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=["y"])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE
        assert result.was_edited is False

    def test_approve_case_insensitive(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=["A"])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE

    def test_approve_with_yes(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=["yes"])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE


# ---------------------------------------------------------------------------
# confirm_ssh_command: reject flow
# ---------------------------------------------------------------------------


class TestConfirmRejectFlow:
    """User types 'r', 'n', or 'd' to reject the command."""

    def test_reject_with_r(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=["r"])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.REJECT
        assert result.was_edited is False

    def test_reject_with_n(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=["n"])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.REJECT

    def test_reject_with_d(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=["d"])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.REJECT

    def test_reject_with_no(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=["no"])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.REJECT

    def test_reject_with_deny(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=["deny"])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.REJECT


# ---------------------------------------------------------------------------
# confirm_ssh_command: edit flow
# ---------------------------------------------------------------------------


class TestConfirmEditFlow:
    """User types 'e' to edit the command, then approves or rejects."""

    def test_edit_then_approve(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=[
            "e",                        # choose edit
            "pytest -v --tb=short",     # new command
            "a",                        # approve the edited version
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE
        assert result.final_command.command == "pytest -v --tb=short"
        assert result.was_edited is True

    def test_edit_then_reject(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=[
            "e",                        # choose edit
            "pytest -v --tb=short",     # new command
            "r",                        # reject after editing
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.REJECT
        assert result.final_command.command == "pytest -v --tb=short"
        assert result.was_edited is True

    def test_edit_keep_original(self) -> None:
        """User enters edit mode but submits empty (keeps original)."""
        req = _make_request()
        io = FakeTerminalIO(inputs=[
            "e",    # choose edit
            "",     # keep original (empty = prefill)
            "a",    # approve
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE
        assert result.final_command.command == "pytest -v"
        assert result.was_edited is False

    def test_edit_multiple_times(self) -> None:
        """User can edit multiple times before deciding."""
        req = _make_request()
        io = FakeTerminalIO(inputs=[
            "e",                        # first edit
            "pytest -v --tb=short",     # change command
            "e",                        # edit again
            "pytest -v --tb=line",      # change again
            "a",                        # approve final version
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE
        assert result.final_command.command == "pytest -v --tb=line"
        assert result.was_edited is True


# ---------------------------------------------------------------------------
# confirm_ssh_command: invalid input handling
# ---------------------------------------------------------------------------


class TestConfirmInvalidInput:
    """Invalid input is rejected with a reprompt."""

    def test_invalid_then_approve(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=[
            "x",   # invalid
            "a",   # approve
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE

    def test_multiple_invalid_then_reject(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=[
            "foo",  # invalid
            "bar",  # invalid
            "n",    # reject
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.REJECT

    def test_whitespace_input_reprompts(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=[
            "  ",   # whitespace
            "y",    # approve
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE


# ---------------------------------------------------------------------------
# confirm_ssh_command: EOF / interrupt handling
# ---------------------------------------------------------------------------


class TestConfirmEOFHandling:
    """EOF or keyboard interrupt results in rejection (safe default)."""

    def test_eof_results_in_reject(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=[])  # No input = immediate EOF
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.REJECT
        assert result.was_edited is False


# ---------------------------------------------------------------------------
# confirm_ssh_command: display is shown
# ---------------------------------------------------------------------------


class TestConfirmDisplaysContext:
    """The confirmation prompt displays the full context before prompting."""

    def test_displays_command_in_output(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=["a"])
        confirm_ssh_command(req, terminal=io)
        assert "pytest -v" in io.output_text

    def test_displays_risk_level(self) -> None:
        req = _make_request(
            context=_make_context(risk_level=RiskLevel.HIGH),
        )
        io = FakeTerminalIO(inputs=["a"])
        confirm_ssh_command(req, terminal=io)
        assert "HIGH" in io.output_text

    def test_displays_explanation(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=["a"])
        confirm_ssh_command(req, terminal=io)
        assert "Run the test suite with verbose output" in io.output_text

    def test_displays_target(self) -> None:
        req = _make_request()
        io = FakeTerminalIO(inputs=["a"])
        confirm_ssh_command(req, terminal=io)
        assert "test.example.com" in io.output_text


# ---------------------------------------------------------------------------
# confirm_ssh_command: edited command validation
# ---------------------------------------------------------------------------


class TestConfirmEditValidation:
    """Edited commands must pass SSHCommand validation."""

    def test_edit_empty_command_reprompts(self) -> None:
        """If user clears the command entirely, reprompt for edit."""
        req = _make_request()
        io = FakeTerminalIO(inputs=[
            "e",          # enter edit mode
            "   ",        # whitespace-only command (invalid)
            "pytest -x",  # valid command on retry
            "a",          # approve
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE
        assert result.final_command.command == "pytest -x"
        assert result.was_edited is True

    def test_edit_preserves_working_directory(self) -> None:
        """Editing command text does not affect working_directory."""
        req = _make_request(
            ssh_command=_make_ssh_command(
                command="pytest -v",
                working_directory="/opt/app",
            ),
        )
        io = FakeTerminalIO(inputs=[
            "e",
            "pytest -x",
            "a",
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.final_command.working_directory == "/opt/app"

    def test_edit_preserves_timeout(self) -> None:
        """Editing command text does not affect timeout."""
        req = _make_request(
            ssh_command=_make_ssh_command(timeout=600),
        )
        io = FakeTerminalIO(inputs=[
            "e",
            "pytest -x",
            "a",
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.final_command.timeout == 600

    def test_edit_preserves_environment(self) -> None:
        """Editing command text does not affect environment variables."""
        cmd = SSHCommand(
            command="pytest -v",
            working_directory="/opt/app",
            environment={"CI": "true"},
        )
        req = ConfirmationRequest(
            ssh_command=cmd,
            context=_make_context(),
            target=_make_target(),
        )
        io = FakeTerminalIO(inputs=[
            "e",
            "pytest -x",
            "a",
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.final_command.environment == {"CI": "true"}

    def test_edit_too_long_command_reprompts(self) -> None:
        """If edited command exceeds max length, reprompt."""
        req = _make_request()
        io = FakeTerminalIO(inputs=[
            "e",
            "x" * 8193,   # exceeds MAX_COMMAND_LENGTH
            "pytest -x",   # valid retry
            "a",
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE
        assert result.final_command.command == "pytest -x"

    def test_edit_eof_during_editing_rejects(self) -> None:
        """EOF during editing results in rejection."""
        req = _make_request()
        io = FakeTerminalIO(inputs=[
            "e",
            # EOF when trying to read edit input
        ])
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.REJECT


# ---------------------------------------------------------------------------
# Render: safety recommendation and environment display
# ---------------------------------------------------------------------------


class TestRenderEdgeCases:
    """Edge cases in display rendering."""

    def test_unsafe_command_shows_safety_warning(self) -> None:
        req = _make_request(
            context=_make_context(safe_to_execute=False),
        )
        output = render_confirmation_display(req)
        assert "NOT recommended" in output

    def test_environment_variables_displayed(self) -> None:
        cmd = SSHCommand(
            command="pytest -v",
            working_directory="/opt/app",
            environment={"CI": "true", "DEBUG": "1"},
        )
        req = ConfirmationRequest(
            ssh_command=cmd,
            context=_make_context(),
            target=_make_target(),
        )
        output = render_confirmation_display(req)
        assert "CI=true" in output
        assert "DEBUG=1" in output

    def test_non_default_port_shown(self) -> None:
        req = _make_request(
            target=SSHTarget(host="example.com", user="admin", port=2222),
        )
        output = render_confirmation_display(req)
        assert ":2222" in output

    def test_default_port_not_shown(self) -> None:
        req = _make_request(
            target=SSHTarget(host="example.com", user="admin", port=22),
        )
        output = render_confirmation_display(req)
        assert ":22" not in output

    def test_medium_risk_level(self) -> None:
        req = _make_request(
            context=_make_context(risk_level=RiskLevel.MEDIUM),
        )
        output = render_confirmation_display(req)
        assert "MEDIUM" in output


# ---------------------------------------------------------------------------
# DefaultTerminalIO with injected streams
# ---------------------------------------------------------------------------


class TestDefaultTerminalIO:
    """DefaultTerminalIO works with injected input/output streams."""

    def test_write_to_output_stream(self) -> None:
        output = StringIO()
        io = DefaultTerminalIO(output_stream=output)
        io.write("hello world")
        assert output.getvalue() == "hello world"

    def test_read_line_from_input_stream(self) -> None:
        input_stream = StringIO("approve\n")
        output = StringIO()
        io = DefaultTerminalIO(input_stream=input_stream, output_stream=output)
        result = io.read_line("prompt: ")
        assert result == "approve"

    def test_read_line_eof_raises(self) -> None:
        input_stream = StringIO("")  # empty = EOF
        output = StringIO()
        io = DefaultTerminalIO(input_stream=input_stream, output_stream=output)
        with pytest.raises(EOFError):
            io.read_line("prompt: ")

    def test_read_editable_fallback_from_stream(self) -> None:
        """When using non-stdin streams, read_editable falls back to read_line."""
        input_stream = StringIO("new command\n")
        output = StringIO()
        io = DefaultTerminalIO(input_stream=input_stream, output_stream=output)
        result = io.read_editable("edit: ", "original command")
        assert result == "new command"

    def test_full_confirmation_with_default_io(self) -> None:
        """End-to-end confirmation using DefaultTerminalIO with stream injection."""
        input_stream = StringIO("a\n")
        output = StringIO()
        io = DefaultTerminalIO(input_stream=input_stream, output_stream=output)
        req = _make_request()
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE
        assert "pytest -v" in output.getvalue()

    def test_edit_flow_with_default_io(self) -> None:
        """Edit flow using DefaultTerminalIO with stream injection."""
        input_stream = StringIO("e\npytest -x\na\n")
        output = StringIO()
        io = DefaultTerminalIO(input_stream=input_stream, output_stream=output)
        req = _make_request()
        result = confirm_ssh_command(req, terminal=io)
        assert result.decision == Decision.APPROVE
        assert result.final_command.command == "pytest -x"
        assert result.was_edited is True
