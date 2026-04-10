"""Tests for the hybrid test output summarizer.

Covers the regex layer (pytest, unittest, jest, iteration counter),
the no-match fallback, the ``llm_client=None`` short-circuit, and
the defensive LLM failure paths.

No real network calls are made -- the LLM layer is stubbed with
lightweight fake clients that return canned responses.
"""

from __future__ import annotations

from typing import Any

import pytest

from jules_daemon.execution.output_summarizer import (
    LLM_SUMMARIZER_TIMEOUT_SECONDS,
    OutputSummary,
    _regex_summary,
    summarize_output,
)


# ---------------------------------------------------------------------------
# Regex layer unit tests
# ---------------------------------------------------------------------------


class TestPytestRegex:
    """The pytest parser handles the standard ``= N passed ... in Xs =`` trailer."""

    def test_passed_and_failed(self) -> None:
        text = "======== 95 passed, 5 failed in 3.45s ========"
        summary = _regex_summary(text)
        assert summary is not None
        assert summary.parser == "pytest"
        assert summary.passed == 95
        assert summary.failed == 5
        assert summary.skipped == 0
        assert summary.total == 100
        assert summary.duration_seconds == pytest.approx(3.45)

    def test_all_passed(self) -> None:
        text = "======== 42 passed in 0.50s ========"
        summary = _regex_summary(text)
        assert summary is not None
        assert summary.parser == "pytest"
        assert summary.passed == 42
        assert summary.failed == 0
        assert summary.total == 42

    def test_errors_are_rolled_into_failed(self) -> None:
        """pytest errors should be counted as failures."""
        text = "======== 3 passed, 1 failed, 2 errors in 0.12s ========"
        summary = _regex_summary(text)
        assert summary is not None
        assert summary.parser == "pytest"
        assert summary.passed == 3
        assert summary.failed == 3  # 1 failed + 2 errors


class TestUnittestRegex:
    """The unittest parser handles ``Ran N tests in Xs`` trailers."""

    def test_basic_ran(self) -> None:
        text = "Ran 42 tests in 0.500s\n\nOK"
        summary = _regex_summary(text)
        assert summary is not None
        assert summary.parser == "unittest"
        assert summary.total == 42
        assert summary.passed == 42
        assert summary.failed == 0
        assert summary.duration_seconds == pytest.approx(0.5)

    def test_failed_with_failures_line(self) -> None:
        text = (
            "Ran 10 tests in 1.230s\n\n"
            "FAILED (failures=2, errors=1)"
        )
        summary = _regex_summary(text)
        assert summary is not None
        assert summary.parser == "unittest"
        assert summary.total == 10
        assert summary.failed == 3
        assert summary.passed == 7

    def test_skipped_tests(self) -> None:
        text = "Ran 5 tests in 0.100s\n\nOK (skipped=2)"
        summary = _regex_summary(text)
        assert summary is not None
        assert summary.parser == "unittest"
        assert summary.total == 5
        assert summary.skipped == 2
        assert summary.passed == 3


class TestJestRegex:
    """The jest parser handles ``Tests: ... total`` summary lines."""

    def test_passed_and_failed(self) -> None:
        text = "Tests:       5 failed, 95 passed, 100 total"
        summary = _regex_summary(text)
        assert summary is not None
        assert summary.parser == "jest"
        assert summary.passed == 95
        assert summary.failed == 5
        assert summary.total == 100

    def test_only_passed(self) -> None:
        text = "Tests:       42 passed, 42 total"
        summary = _regex_summary(text)
        assert summary is not None
        assert summary.parser == "jest"
        assert summary.passed == 42
        assert summary.failed == 0
        assert summary.total == 42


class TestIterationCounter:
    """The iteration parser counts PASSED/FAILED lines from custom loops."""

    def test_counts_passed_and_failed(self) -> None:
        text = (
            "Iteration 1/100: PASSED\n"
            "Iteration 2/100: FAILED\n"
            "Iteration 3/100: PASSED\n"
            "Iteration 4/100: PASSED\n"
        )
        summary = _regex_summary(text)
        assert summary is not None
        assert summary.parser == "iteration"
        assert summary.passed == 3
        assert summary.failed == 1
        assert summary.total == 4

    def test_lowercase_iteration(self) -> None:
        text = "iteration 1: PASSED\niteration 2: FAILED"
        summary = _regex_summary(text)
        assert summary is not None
        assert summary.parser == "iteration"
        assert summary.passed == 1
        assert summary.failed == 1

    def test_no_slash_total(self) -> None:
        text = "Iteration 1: PASSED\nIteration 2: PASSED"
        summary = _regex_summary(text)
        assert summary is not None
        assert summary.parser == "iteration"
        assert summary.total == 2


class TestNoMatchFallback:
    """Inputs that match none of the patterns yield ``None``."""

    def test_random_text_returns_none(self) -> None:
        assert _regex_summary("nothing useful here") is None

    def test_empty_string_returns_none(self) -> None:
        assert _regex_summary("") is None


# ---------------------------------------------------------------------------
# summarize_output integration tests (LLM disabled)
# ---------------------------------------------------------------------------


class TestSummarizeOutputWithoutLLM:
    """When no LLM client is provided, the regex result is returned verbatim."""

    @pytest.mark.asyncio
    async def test_pytest_output_no_llm(self) -> None:
        result = await summarize_output(
            stdout="======== 10 passed, 2 failed in 1.00s ========",
            stderr="",
            command="pytest",
            exit_code=1,
            llm_client=None,
            llm_model=None,
        )
        assert result.parser == "pytest"
        assert result.passed == 10
        assert result.failed == 2
        assert result.duration_seconds == pytest.approx(1.00)
        assert result.narrative == ""  # no LLM -> no narrative
        assert "10 passed" in result.raw_excerpt

    @pytest.mark.asyncio
    async def test_unknown_output_no_llm(self) -> None:
        result = await summarize_output(
            stdout="some random output",
            stderr="",
            command="./run.sh",
            exit_code=0,
            llm_client=None,
        )
        assert result.parser == "none"
        assert result.passed == 0
        assert result.failed == 0
        assert result.narrative == ""
        assert result.raw_excerpt == "some random output"

    @pytest.mark.asyncio
    async def test_llm_model_missing_treated_as_disabled(self) -> None:
        """A client without a model is treated the same as ``llm_client=None``."""
        sentinel = object()  # any non-None client is OK
        result = await summarize_output(
            stdout="Ran 3 tests in 0.01s",
            stderr="",
            command="python -m unittest",
            exit_code=0,
            llm_client=sentinel,
            llm_model=None,
        )
        assert result.parser == "unittest"
        assert result.total == 3
        assert result.narrative == ""  # LLM skipped due to missing model


# ---------------------------------------------------------------------------
# summarize_output with stubbed LLM clients
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for an OpenAI ChatCompletion response."""

    def __init__(self, content: str) -> None:
        self.choices = [
            type("_Choice", (), {
                "message": type("_Message", (), {"content": content})()
            })()
        ]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeClient:
    """Lightweight fake OpenAI-style client for testing."""

    def __init__(self, content: str) -> None:
        self.chat = _FakeChat(content)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.calls


class TestSummarizeOutputWithLLM:
    """Verify narrative synthesis and LLM fallback paths."""

    @pytest.mark.asyncio
    async def test_narrative_attached_when_regex_matches(self) -> None:
        llm_response = (
            '{"passed": 10, "failed": 2, "skipped": 0, "total": 12, '
            '"key_failures": ["test_login failed"], '
            '"narrative": "10 passed, 2 failed, mostly stable."}'
        )
        client = _FakeClient(llm_response)
        result = await summarize_output(
            stdout="======== 10 passed, 2 failed in 1.00s ========",
            stderr="",
            command="pytest",
            exit_code=1,
            llm_client=client,
            llm_model="openai:default:gpt-4o",
        )
        # Regex provides the counts, LLM provides narrative + failures
        assert result.parser == "pytest"
        assert result.passed == 10
        assert result.failed == 2
        assert result.narrative == "10 passed, 2 failed, mostly stable."
        assert result.key_failures == ("test_login failed",)
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_llm_fallback_when_regex_fails(self) -> None:
        """LLM counts are used when no regex pattern matched."""
        llm_response = (
            '{"passed": 3, "failed": 1, "skipped": 0, "total": 4, '
            '"key_failures": ["assertion failure"], '
            '"narrative": "3 out of 4 custom checks succeeded."}'
        )
        client = _FakeClient(llm_response)
        result = await summarize_output(
            stdout="custom check harness output: 3 ok, 1 bad",
            stderr="",
            command="./harness",
            exit_code=1,
            llm_client=client,
            llm_model="openai:default:gpt-4o",
        )
        assert result.parser == "llm"
        assert result.passed == 3
        assert result.failed == 1
        assert result.total == 4
        assert result.narrative == "3 out of 4 custom checks succeeded."
        assert result.key_failures == ("assertion failure",)

    @pytest.mark.asyncio
    async def test_llm_json_inside_code_fence(self) -> None:
        """Fenced JSON blocks in the LLM reply are parsed."""
        llm_response = (
            "Here is the analysis:\n"
            "```json\n"
            '{"passed": 2, "failed": 0, "skipped": 0, "total": 2, '
            '"key_failures": [], "narrative": "Both tests passed."}'
            "\n```\n"
        )
        client = _FakeClient(llm_response)
        result = await summarize_output(
            stdout="========= 2 passed in 0.10s =========",
            stderr="",
            command="pytest",
            exit_code=0,
            llm_client=client,
            llm_model="openai:default:gpt-4o",
        )
        assert result.parser == "pytest"
        assert result.narrative == "Both tests passed."

    @pytest.mark.asyncio
    async def test_llm_error_returns_regex_result(self) -> None:
        """If the LLM raises, the regex result is returned with empty narrative."""

        class _BrokenClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**_kwargs: Any) -> None:
                        raise RuntimeError("network unavailable")

        result = await summarize_output(
            stdout="========= 5 passed in 0.05s =========",
            stderr="",
            command="pytest",
            exit_code=0,
            llm_client=_BrokenClient(),
            llm_model="openai:default:gpt-4o",
        )
        assert result.parser == "pytest"
        assert result.passed == 5
        assert result.narrative == ""

    @pytest.mark.asyncio
    async def test_llm_malformed_json_returns_regex_result(self) -> None:
        """Garbage LLM responses leave counts intact but blank narrative."""
        client = _FakeClient("this is not JSON at all")
        result = await summarize_output(
            stdout="========= 5 passed in 0.05s =========",
            stderr="",
            command="pytest",
            exit_code=0,
            llm_client=client,
            llm_model="openai:default:gpt-4o",
        )
        assert result.parser == "pytest"
        assert result.passed == 5
        assert result.narrative == ""

    @pytest.mark.asyncio
    async def test_llm_timeout_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A slow LLM call is replaced by a quick timeout."""

        class _SlowClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**_kwargs: Any) -> None:
                        # Sleep longer than the wait_for timeout so the
                        # asyncio wrapper raises TimeoutError.
                        import time

                        time.sleep(0.5)

        # Monkeypatch the module-level timeout to a very short value so
        # this test runs quickly and deterministically.
        monkeypatch.setattr(
            "jules_daemon.execution.output_summarizer."
            "LLM_SUMMARIZER_TIMEOUT_SECONDS",
            0.05,
        )
        result = await summarize_output(
            stdout="========= 1 passed in 0.01s =========",
            stderr="",
            command="pytest",
            exit_code=0,
            llm_client=_SlowClient(),
            llm_model="openai:default:gpt-4o",
        )
        assert result.parser == "pytest"
        assert result.narrative == ""


# ---------------------------------------------------------------------------
# OutputSummary dataclass sanity
# ---------------------------------------------------------------------------


class TestOutputSummaryDataclass:
    def test_defaults(self) -> None:
        summary = OutputSummary(parser="none")
        assert summary.passed == 0
        assert summary.failed == 0
        assert summary.skipped == 0
        assert summary.total == 0
        assert summary.duration_seconds is None
        assert summary.key_failures == ()
        assert summary.narrative == ""
        assert summary.raw_excerpt == ""

    def test_is_frozen(self) -> None:
        summary = OutputSummary(parser="none")
        with pytest.raises(Exception):
            summary.parser = "pytest"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Module-level constant sanity
# ---------------------------------------------------------------------------


def test_timeout_constant_is_positive() -> None:
    assert LLM_SUMMARIZER_TIMEOUT_SECONDS > 0
