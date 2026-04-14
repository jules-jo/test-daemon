"""Tests for the heuristic natural-language intent extractor.

Validates that the NL extractor can identify intent (canonical verb)
and extract arguments from free-form English input without requiring
an LLM call. This is the fast heuristic path for common NL patterns.
"""

from __future__ import annotations

import pytest

from jules_daemon.classifier.nl_extractor import (
    NLExtraction,
    extract_from_natural_language,
)


class TestRunIntent:
    """NL inputs that should extract a 'run' intent."""

    def test_run_smoke_tests_on_staging(self) -> None:
        result = extract_from_natural_language(
            "run the smoke tests on staging"
        )
        assert result.canonical_verb == "run"
        assert result.confidence > 0.0

    def test_can_you_run_tests(self) -> None:
        result = extract_from_natural_language(
            "can you run the tests on staging?"
        )
        assert result.canonical_verb == "run"

    def test_kick_off_regression(self) -> None:
        result = extract_from_natural_language(
            "kick off the regression suite on staging"
        )
        assert result.canonical_verb == "run"

    def test_execute_test_suite(self) -> None:
        result = extract_from_natural_language(
            "please execute the test suite on production"
        )
        assert result.canonical_verb == "run"

    def test_start_the_tests(self) -> None:
        result = extract_from_natural_language(
            "start the integration tests"
        )
        assert result.canonical_verb == "run"

    def test_trigger_tests(self) -> None:
        result = extract_from_natural_language(
            "I need to trigger the smoke tests"
        )
        assert result.canonical_verb == "run"

    def test_run_intent_keeps_original_natural_language(self) -> None:
        raw = "can you run the smoke tests on ci@staging?"
        result = extract_from_natural_language(raw)
        assert result.extracted_args.get("natural_language") == raw


class TestStatusIntent:
    """NL inputs that should extract a 'status' intent."""

    def test_whats_happening(self) -> None:
        result = extract_from_natural_language("what's happening?")
        assert result.canonical_verb == "status"

    def test_whats_running(self) -> None:
        result = extract_from_natural_language("what's running right now?")
        assert result.canonical_verb == "status"

    def test_check_on_tests(self) -> None:
        result = extract_from_natural_language("check on the tests for me")
        assert result.canonical_verb == "status"

    def test_how_are_tests_going(self) -> None:
        result = extract_from_natural_language(
            "how are the tests going?"
        )
        assert result.canonical_verb == "status"

    def test_is_anything_running(self) -> None:
        result = extract_from_natural_language(
            "is there anything running?"
        )
        assert result.canonical_verb == "status"

    def test_any_progress(self) -> None:
        result = extract_from_natural_language("any progress?")
        assert result.canonical_verb == "status"


class TestCancelIntent:
    """NL inputs that should extract a 'cancel' intent."""

    def test_stop_the_tests(self) -> None:
        result = extract_from_natural_language("stop the tests")
        assert result.canonical_verb == "cancel"

    def test_cancel_current_run(self) -> None:
        result = extract_from_natural_language("cancel the current run")
        assert result.canonical_verb == "cancel"

    def test_abort_everything(self) -> None:
        result = extract_from_natural_language("abort everything")
        assert result.canonical_verb == "cancel"

    def test_kill_the_run(self) -> None:
        result = extract_from_natural_language("kill the run please")
        assert result.canonical_verb == "cancel"

    def test_please_stop(self) -> None:
        result = extract_from_natural_language(
            "please stop whatever is running"
        )
        assert result.canonical_verb == "cancel"


class TestWatchIntent:
    """NL inputs that should extract a 'watch' intent."""

    def test_show_me_output(self) -> None:
        result = extract_from_natural_language("show me the output")
        assert result.canonical_verb == "watch"

    def test_see_the_logs(self) -> None:
        result = extract_from_natural_language("let me see the logs")
        assert result.canonical_verb == "watch"

    def test_follow_the_output(self) -> None:
        result = extract_from_natural_language("follow the test output")
        assert result.canonical_verb == "watch"

    def test_tail_the_logs(self) -> None:
        result = extract_from_natural_language(
            "can I tail the logs?"
        )
        assert result.canonical_verb == "watch"

    def test_stream_results(self) -> None:
        result = extract_from_natural_language("stream the results")
        assert result.canonical_verb == "watch"


class TestHistoryIntent:
    """NL inputs that should extract a 'history' intent."""

    def test_show_past_results(self) -> None:
        result = extract_from_natural_language("show me past results")
        assert result.canonical_verb == "history"

    def test_what_happened_last_time(self) -> None:
        result = extract_from_natural_language(
            "what happened in the last run?"
        )
        assert result.canonical_verb == "history"

    def test_previous_test_results(self) -> None:
        result = extract_from_natural_language("show previous test results")
        assert result.canonical_verb == "history"

    def test_show_reports(self) -> None:
        result = extract_from_natural_language("show me the reports")
        assert result.canonical_verb == "history"

    def test_recent_runs(self) -> None:
        result = extract_from_natural_language(
            "what were the recent runs?"
        )
        assert result.canonical_verb == "history"


class TestQueueIntent:
    """NL inputs that should extract a 'queue' intent."""

    def test_queue_for_later(self) -> None:
        result = extract_from_natural_language(
            "queue the tests for later"
        )
        assert result.canonical_verb == "queue"

    def test_schedule_tests(self) -> None:
        result = extract_from_natural_language(
            "schedule the regression tests"
        )
        assert result.canonical_verb == "queue"

    def test_run_later(self) -> None:
        result = extract_from_natural_language(
            "run the tests later when the daemon is free"
        )
        assert result.canonical_verb == "queue"

    def test_defer_execution(self) -> None:
        result = extract_from_natural_language(
            "defer the test execution"
        )
        assert result.canonical_verb == "queue"

    def test_queue_intent_keeps_original_natural_language(self) -> None:
        raw = "queue the smoke tests for later on ci@staging"
        result = extract_from_natural_language(raw)
        assert result.extracted_args.get("natural_language") == raw


class TestSSHTargetExtraction:
    """NL inputs containing SSH targets should extract them."""

    def test_ssh_target_in_nl(self) -> None:
        result = extract_from_natural_language(
            "run the tests on deploy@staging"
        )
        assert result.extracted_args.get("target_user") == "deploy"
        assert result.extracted_args.get("target_host") == "staging"

    def test_ssh_target_with_port(self) -> None:
        result = extract_from_natural_language(
            "execute the suite on ci@prod:2222"
        )
        assert result.extracted_args.get("target_user") == "ci"
        assert result.extracted_args.get("target_host") == "prod"
        assert result.extracted_args.get("target_port") == 2222

    def test_system_alias_reference_is_extracted(self) -> None:
        result = extract_from_natural_language(
            "run the smoke tests in system tuto"
        )
        assert result.extracted_args.get("system_name") == "tuto"

    def test_implicit_system_alias_sets_infer_target_hint(self) -> None:
        raw = "run the smoke tests in tuto"
        result = extract_from_natural_language(raw)
        assert result.extracted_args.get("infer_target") is True
        assert result.extracted_args.get("natural_language") == raw
        assert "system_name" not in result.extracted_args

    def test_implicit_system_alias_with_following_arguments_sets_infer_target_hint(
        self,
    ) -> None:
        raw = "run the smoke tests in tuto. 1 iteration"
        result = extract_from_natural_language(raw)
        assert result.extracted_args.get("infer_target") is True
        assert result.extracted_args.get("natural_language") == raw
        assert "system_name" not in result.extracted_args


class TestNLExtractionDataclass:
    """Tests for the NLExtraction dataclass itself."""

    def test_frozen(self) -> None:
        result = NLExtraction(
            canonical_verb="run",
            confidence=0.5,
            extracted_args={},
            raw_input="test input",
        )
        with pytest.raises(AttributeError):
            result.canonical_verb = "status"  # type: ignore[misc]

    def test_confidence_bounds(self) -> None:
        result = NLExtraction(
            canonical_verb="status",
            confidence=0.0,
            extracted_args={},
            raw_input="test",
        )
        assert result.confidence == 0.0

        result_high = NLExtraction(
            canonical_verb="status",
            confidence=1.0,
            extracted_args={},
            raw_input="test",
        )
        assert result_high.confidence == 1.0

    def test_invalid_confidence_below_zero(self) -> None:
        with pytest.raises(ValueError, match="confidence must be between"):
            NLExtraction(
                canonical_verb="status",
                confidence=-0.1,
                extracted_args={},
                raw_input="test",
            )

    def test_invalid_confidence_above_one(self) -> None:
        with pytest.raises(ValueError, match="confidence must be between"):
            NLExtraction(
                canonical_verb="status",
                confidence=1.1,
                extracted_args={},
                raw_input="test",
            )

    def test_empty_raw_input_raises(self) -> None:
        with pytest.raises(ValueError, match="raw_input must not be empty"):
            NLExtraction(
                canonical_verb="status",
                confidence=0.5,
                extracted_args={},
                raw_input="",
            )


class TestAmbiguousInput:
    """Inputs that are genuinely ambiguous should still produce a result."""

    def test_empty_returns_status_with_low_confidence(self) -> None:
        """Empty input defaults to status with zero confidence."""
        result = extract_from_natural_language("")
        assert result.canonical_verb == "status"
        assert result.confidence == 0.0

    def test_gibberish_has_low_confidence(self) -> None:
        result = extract_from_natural_language("asdfghjkl zxcvbnm")
        assert result.confidence < 0.5

    def test_single_word_non_verb(self) -> None:
        result = extract_from_natural_language("hello")
        assert result.confidence < 0.5
