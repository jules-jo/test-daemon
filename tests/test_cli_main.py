"""Tests for the legacy CLI entry point's active input resolution path."""

from __future__ import annotations

from unittest.mock import patch

from jules_daemon.cli_main import _resolve_input


class TestResolveInput:
    """The active CLI flow should route user text to daemon interpretation."""

    def test_empty_input_returns_error(self) -> None:
        result = _resolve_input("", allow_prompt=False)
        assert result.parts is None
        assert result.error == "Empty input."

    def test_health_remains_local_escape_hatch(self) -> None:
        result = _resolve_input("health", allow_prompt=False)
        assert result.parts == ("health",)
        assert result.hint is None
        assert result.error is None

    def test_interpret_remains_explicit_debug_escape_hatch(self) -> None:
        result = _resolve_input(
            "interpret give me the current status",
            allow_prompt=False,
        )
        assert result.parts == ("interpret", "give me the current status")
        assert result.hint is None
        assert result.error is None

    def test_structured_status_is_forwarded_to_daemon_interpretation(self) -> None:
        raw = "status --verbose"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("interpret", raw)
        assert result.hint == "(sent to daemon for interpretation)"
        assert result.error is None

    def test_structured_run_is_forwarded_to_daemon_interpretation(self) -> None:
        raw = "run deploy@staging run the smoke tests"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("interpret", raw)
        assert result.hint == "(sent to daemon for interpretation)"
        assert result.error is None

    def test_conversational_status_is_forwarded_to_daemon_interpretation(self) -> None:
        raw = "give me the current status"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("interpret", raw)
        assert result.hint == "(sent to daemon for interpretation)"
        assert result.error is None

    def test_conversational_run_is_forwarded_to_daemon_interpretation(self) -> None:
        raw = "run the smoke tests in tuto. 1 iteration"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("interpret", raw)
        assert result.hint == "(sent to daemon for interpretation)"
        assert result.error is None

    def test_no_local_target_prompt_for_conversational_run(self) -> None:
        raw = "run the smoke tests there"
        with patch("builtins.input") as mock_input:
            result = _resolve_input(raw, allow_prompt=True)
        assert result.parts == ("interpret", raw)
        assert result.hint == "(sent to daemon for interpretation)"
        assert result.error is None
        mock_input.assert_not_called()
