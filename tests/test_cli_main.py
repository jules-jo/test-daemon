"""Tests for natural-language resolution in the legacy CLI entry point."""

from __future__ import annotations

from unittest.mock import patch

from jules_daemon.cli_main import _resolve_input


class TestResolveInput:
    """Natural-language run requests should resolve to explicit run argv."""

    def test_structured_run_is_preserved(self) -> None:
        result = _resolve_input(
            "run deploy@staging run the smoke tests",
            allow_prompt=False,
        )
        assert result.parts == (
            "run",
            "deploy@staging",
            "run",
            "the",
            "smoke",
            "tests",
        )
        assert result.hint is None
        assert result.error is None

    def test_structured_run_with_system_flag_is_preserved(self) -> None:
        result = _resolve_input(
            "run --system tuto run the smoke tests",
            allow_prompt=False,
        )
        assert result.parts == (
            "run",
            "--system",
            "tuto",
            "run",
            "the",
            "smoke",
            "tests",
        )
        assert result.hint is None
        assert result.error is None

    def test_bare_run_sentence_with_target_is_rewritten(self) -> None:
        raw = "run the smoke tests on deploy@staging"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("run", "deploy@staging", raw)
        assert result.hint == "(interpreted as 'run')"
        assert result.error is None

    def test_polite_run_sentence_with_target_is_rewritten(self) -> None:
        raw = "can you run the smoke tests on ci@staging?"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("run", "ci@staging", raw)
        assert result.hint == "(interpreted as 'run')"
        assert result.error is None

    def test_run_sentence_without_target_uses_interpret_request(self) -> None:
        raw = "run the smoke tests"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("run", "--interpret-request", raw)
        assert result.hint == "(interpreted as 'run')"
        assert result.error is None

    def test_run_sentence_without_target_does_not_prompt_locally(self) -> None:
        raw = "run the smoke tests"
        with patch("builtins.input") as mock_input:
            result = _resolve_input(raw, allow_prompt=True)
        assert result.parts == ("run", "--interpret-request", raw)
        assert result.hint == "(interpreted as 'run')"
        assert result.error is None
        mock_input.assert_not_called()

    def test_run_sentence_with_system_alias_is_rewritten(self) -> None:
        raw = "run the smoke tests in system tuto"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("run", "--system", "tuto", raw)
        assert result.hint == "(interpreted as 'run')"
        assert result.error is None

    def test_run_sentence_with_implicit_system_alias_uses_infer_target(self) -> None:
        raw = "run the smoke tests in tuto"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("run", "--infer-target", raw)
        assert result.hint == "(interpreted as 'run')"
        assert result.error is None

    def test_run_sentence_with_implicit_system_alias_and_following_args_uses_infer_target(
        self,
    ) -> None:
        raw = "run the smoke tests in tuto. 1 iteration"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("run", "--infer-target", raw)
        assert result.hint == "(interpreted as 'run')"
        assert result.error is None
