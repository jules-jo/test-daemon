"""Tests for collision warning display and interactive prompt.

Verifies that the collision prompt:
- Formats collision details (PID, command, duration) into a readable warning
- Shows source classification (process_table, wiki_session, both)
- Includes wiki run ID and status when available
- Handles unknown/None duration gracefully
- Handles pid=0 (wiki sessions with no daemon PID) gracefully
- Presents three choices: Proceed, Abort, Force-replace
- Returns the correct CollisionAction for each user choice
- Treats EOF/Ctrl-C as abort for safety
- Reprompts on invalid input
- Formats duration in human-readable form (seconds, minutes, hours)
- Handles single and multiple collision entries
- Uses TerminalIO abstraction for testability
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import pytest

from jules_daemon.startup.collision_detector import (
    CollisionEntry,
    CollisionReport,
    CollisionSource,
)
from jules_daemon.startup.collision_prompt import (
    CollisionAction,
    CollisionPromptResult,
    format_collision_warning,
    format_duration,
    prompt_collision_action,
)
from jules_daemon.cli.confirmation import TerminalIO


# ---------------------------------------------------------------------------
# Fake TerminalIO for testing
# ---------------------------------------------------------------------------


class FakeTerminalIO(TerminalIO):
    """Test double that replays scripted input and captures output."""

    def __init__(self, inputs: list[str]) -> None:
        self._inputs = list(inputs)
        self._input_index = 0
        self.output: list[str] = []

    def write(self, text: str) -> None:
        self.output.append(text)

    def read_line(self, prompt: str = "") -> str:
        if self._input_index >= len(self._inputs):
            raise EOFError("No more scripted input")
        value = self._inputs[self._input_index]
        self._input_index += 1
        return value

    def read_editable(self, prompt: str, prefill: str) -> str:
        return self.read_line(prompt)

    @property
    def full_output(self) -> str:
        return "".join(self.output)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    *,
    pid: int = 1234,
    command_line: str = "python -m jules_daemon",
    start_time: Optional[datetime] = None,
    duration_seconds: Optional[float] = 120.0,
    source: CollisionSource = CollisionSource.PROCESS_TABLE,
    wiki_run_id: Optional[str] = None,
    wiki_status: Optional[str] = None,
) -> CollisionEntry:
    return CollisionEntry(
        pid=pid,
        command_line=command_line,
        start_time=start_time,
        duration_seconds=duration_seconds,
        source=source,
        wiki_run_id=wiki_run_id,
        wiki_status=wiki_status,
    )


def _make_report(
    entries: tuple[CollisionEntry, ...] = (),
    our_pid: int = 0,
) -> CollisionReport:
    effective_pid = our_pid or os.getpid()
    return CollisionReport(
        entries=entries,
        has_collision=len(entries) > 0,
        our_pid=effective_pid,
        checked_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# CollisionAction enum
# ---------------------------------------------------------------------------


class TestCollisionAction:
    """Tests for the CollisionAction enum."""

    def test_all_values_exist(self) -> None:
        assert CollisionAction.PROCEED.value == "proceed"
        assert CollisionAction.ABORT.value == "abort"
        assert CollisionAction.FORCE_REPLACE.value == "force_replace"

    def test_three_members(self) -> None:
        assert len(CollisionAction) == 3


# ---------------------------------------------------------------------------
# CollisionPromptResult model
# ---------------------------------------------------------------------------


class TestCollisionPromptResult:
    """Tests for the immutable CollisionPromptResult dataclass."""

    def test_frozen(self) -> None:
        report = _make_report()
        result = CollisionPromptResult(
            action=CollisionAction.ABORT,
            report=report,
        )
        with pytest.raises(AttributeError):
            result.action = CollisionAction.PROCEED  # type: ignore[misc]

    def test_fields(self) -> None:
        report = _make_report()
        result = CollisionPromptResult(
            action=CollisionAction.PROCEED,
            report=report,
        )
        assert result.action == CollisionAction.PROCEED
        assert result.report is report


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------


class TestFormatDuration:
    """Tests for human-readable duration formatting."""

    def test_none_returns_unknown(self) -> None:
        assert format_duration(None) == "unknown"

    def test_seconds_only(self) -> None:
        assert format_duration(45.0) == "45s"

    def test_minutes_and_seconds(self) -> None:
        result = format_duration(125.0)
        assert result == "2m 5s"

    def test_hours_minutes_seconds(self) -> None:
        result = format_duration(3725.0)
        assert result == "1h 2m 5s"

    def test_exact_minutes(self) -> None:
        assert format_duration(120.0) == "2m 0s"

    def test_exact_hours(self) -> None:
        assert format_duration(3600.0) == "1h 0m 0s"

    def test_zero_duration(self) -> None:
        assert format_duration(0.0) == "0s"

    def test_fractional_seconds_truncated(self) -> None:
        assert format_duration(45.7) == "45s"

    def test_large_duration(self) -> None:
        # 25 hours, 30 minutes, 15 seconds
        result = format_duration(91815.0)
        assert result == "25h 30m 15s"


# ---------------------------------------------------------------------------
# format_collision_warning
# ---------------------------------------------------------------------------


class TestFormatCollisionWarning:
    """Tests for formatting the collision warning display."""

    def test_single_process_table_entry(self) -> None:
        entry = _make_entry(
            pid=1234,
            command_line="python -m jules_daemon serve",
            duration_seconds=120.0,
            source=CollisionSource.PROCESS_TABLE,
        )
        report = _make_report(entries=(entry,))
        warning = format_collision_warning(report)

        assert "WARNING" in warning or "Warning" in warning
        assert "1234" in warning
        assert "jules_daemon" in warning
        assert "2m 0s" in warning
        assert "process_table" in warning or "Process Table" in warning

    def test_wiki_session_entry(self) -> None:
        entry = _make_entry(
            pid=5678,
            command_line="",
            duration_seconds=3600.0,
            source=CollisionSource.WIKI_SESSION,
            wiki_run_id="run-abc-123",
            wiki_status="running",
        )
        report = _make_report(entries=(entry,))
        warning = format_collision_warning(report)

        assert "5678" in warning
        assert "run-abc-123" in warning
        assert "running" in warning
        assert "wiki" in warning.lower()

    def test_both_source_entry(self) -> None:
        entry = _make_entry(
            pid=9999,
            command_line="python -m jules_daemon --port 8080",
            duration_seconds=60.0,
            source=CollisionSource.BOTH,
            wiki_run_id="run-xyz-789",
            wiki_status="running",
        )
        report = _make_report(entries=(entry,))
        warning = format_collision_warning(report)

        assert "9999" in warning
        assert "both" in warning.lower() or "Both" in warning

    def test_unknown_duration(self) -> None:
        entry = _make_entry(duration_seconds=None)
        report = _make_report(entries=(entry,))
        warning = format_collision_warning(report)

        assert "unknown" in warning.lower()

    def test_pid_zero_shows_no_pid_label(self) -> None:
        entry = _make_entry(
            pid=0,
            command_line="",
            source=CollisionSource.WIKI_SESSION,
            wiki_run_id="orphan-001",
        )
        report = _make_report(entries=(entry,))
        warning = format_collision_warning(report)

        # Should not just show "0" as the PID -- show something meaningful
        assert "no PID" in warning or "N/A" in warning or "unknown" in warning.lower()

    def test_multiple_entries(self) -> None:
        entry1 = _make_entry(pid=1111, duration_seconds=60.0)
        entry2 = _make_entry(pid=2222, duration_seconds=300.0)
        report = _make_report(entries=(entry1, entry2))
        warning = format_collision_warning(report)

        assert "1111" in warning
        assert "2222" in warning

    def test_contains_action_choices(self) -> None:
        entry = _make_entry()
        report = _make_report(entries=(entry,))
        warning = format_collision_warning(report)

        # The warning should mention the available choices
        assert "[P]roceed" in warning
        assert "[A]bort" in warning
        assert "[F]orce-replace" in warning

    def test_long_command_line_truncated(self) -> None:
        long_cmd = "python -m jules_daemon " + "x" * 200
        entry = _make_entry(command_line=long_cmd)
        report = _make_report(entries=(entry,))
        warning = format_collision_warning(report)

        # The full 200+ char command should be truncated
        assert len(warning) < len(long_cmd) + 500  # reasonable bound

    def test_empty_command_line(self) -> None:
        entry = _make_entry(command_line="", source=CollisionSource.WIKI_SESSION)
        report = _make_report(entries=(entry,))
        warning = format_collision_warning(report)

        # Should handle gracefully, not crash
        assert "1234" in warning


# ---------------------------------------------------------------------------
# prompt_collision_action: interactive prompt
# ---------------------------------------------------------------------------


class TestPromptCollisionActionProceed:
    """Tests for the proceed action in the collision prompt."""

    @pytest.mark.parametrize("input_val", ["p", "P", "proceed", "PROCEED", "Proceed"])
    def test_proceed_inputs(self, input_val: str) -> None:
        entry = _make_entry()
        report = _make_report(entries=(entry,))
        terminal = FakeTerminalIO([input_val])

        result = prompt_collision_action(report, terminal=terminal)

        assert result.action == CollisionAction.PROCEED
        assert result.report is report


class TestPromptCollisionActionAbort:
    """Tests for the abort action in the collision prompt."""

    @pytest.mark.parametrize("input_val", ["a", "A", "abort", "ABORT", "Abort"])
    def test_abort_inputs(self, input_val: str) -> None:
        entry = _make_entry()
        report = _make_report(entries=(entry,))
        terminal = FakeTerminalIO([input_val])

        result = prompt_collision_action(report, terminal=terminal)

        assert result.action == CollisionAction.ABORT
        assert result.report is report


class TestPromptCollisionActionForceReplace:
    """Tests for the force-replace action in the collision prompt."""

    @pytest.mark.parametrize(
        "input_val", ["f", "F", "force", "FORCE", "Force", "force-replace"]
    )
    def test_force_replace_inputs(self, input_val: str) -> None:
        entry = _make_entry()
        report = _make_report(entries=(entry,))
        terminal = FakeTerminalIO([input_val])

        result = prompt_collision_action(report, terminal=terminal)

        assert result.action == CollisionAction.FORCE_REPLACE
        assert result.report is report


class TestPromptCollisionActionEOF:
    """Tests for EOF/Ctrl-C handling (treated as abort for safety)."""

    def test_eof_treated_as_abort(self) -> None:
        entry = _make_entry()
        report = _make_report(entries=(entry,))
        terminal = FakeTerminalIO([])  # empty -> EOFError

        result = prompt_collision_action(report, terminal=terminal)

        assert result.action == CollisionAction.ABORT

    def test_eof_after_invalid_input_treated_as_abort(self) -> None:
        entry = _make_entry()
        report = _make_report(entries=(entry,))
        terminal = FakeTerminalIO(["xyz"])  # invalid, then EOF

        result = prompt_collision_action(report, terminal=terminal)

        assert result.action == CollisionAction.ABORT


class TestPromptCollisionActionInvalidInput:
    """Tests for reprompting on invalid input."""

    def test_reprompts_on_invalid_then_accepts(self) -> None:
        entry = _make_entry()
        report = _make_report(entries=(entry,))
        terminal = FakeTerminalIO(["xyz", "what?", "p"])

        result = prompt_collision_action(report, terminal=terminal)

        assert result.action == CollisionAction.PROCEED
        # Should have shown an error message for invalid input
        output = terminal.full_output
        assert "Invalid" in output or "invalid" in output

    def test_empty_input_reprompts(self) -> None:
        entry = _make_entry()
        report = _make_report(entries=(entry,))
        terminal = FakeTerminalIO(["", "  ", "a"])

        result = prompt_collision_action(report, terminal=terminal)

        assert result.action == CollisionAction.ABORT


class TestPromptCollisionActionDisplay:
    """Tests that the warning is displayed before prompting."""

    def test_warning_displayed(self) -> None:
        entry = _make_entry(pid=4567, duration_seconds=90.0)
        report = _make_report(entries=(entry,))
        terminal = FakeTerminalIO(["p"])

        prompt_collision_action(report, terminal=terminal)

        output = terminal.full_output
        assert "4567" in output
        assert "1m 30s" in output

    def test_multiple_entries_all_displayed(self) -> None:
        entry1 = _make_entry(pid=1111, duration_seconds=60.0)
        entry2 = _make_entry(pid=2222, duration_seconds=300.0)
        report = _make_report(entries=(entry1, entry2))
        terminal = FakeTerminalIO(["p"])

        prompt_collision_action(report, terminal=terminal)

        output = terminal.full_output
        assert "1111" in output
        assert "2222" in output
