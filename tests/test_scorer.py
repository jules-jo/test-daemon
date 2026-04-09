"""Tests for the structuredness scorer.

Validates that the scorer correctly distinguishes structured CLI-style
commands from natural-language free-text input, returning a normalized
score in [0.0, 1.0].

Score semantics:
    1.0 = purely structured command (verb + flags + SSH target)
    0.0 = purely natural language (question, conversational text)
"""

from __future__ import annotations

import pytest

from jules_daemon.classifier.scorer import (
    STRUCTURED_THRESHOLD,
    compute_structuredness_score,
)


class TestStructuredCommandsHighScore:
    """Inputs that should score high (>= threshold) as structured."""

    def test_exact_verb_only(self) -> None:
        score = compute_structuredness_score("status")
        assert score >= STRUCTURED_THRESHOLD

    def test_exact_verb_with_flag(self) -> None:
        score = compute_structuredness_score("status --verbose")
        assert score >= STRUCTURED_THRESHOLD

    def test_run_with_ssh_target(self) -> None:
        score = compute_structuredness_score("run deploy@staging run the tests")
        assert score >= STRUCTURED_THRESHOLD

    def test_alias_verb(self) -> None:
        score = compute_structuredness_score("execute deploy@host run tests")
        assert score >= STRUCTURED_THRESHOLD

    def test_cancel_with_flags(self) -> None:
        score = compute_structuredness_score("cancel --run-id abc-123 --force")
        assert score >= STRUCTURED_THRESHOLD

    def test_watch_with_tail(self) -> None:
        score = compute_structuredness_score("watch --tail 100")
        assert score >= STRUCTURED_THRESHOLD

    def test_queue_with_priority(self) -> None:
        score = compute_structuredness_score(
            "queue deploy@host run tests --priority 5"
        )
        assert score >= STRUCTURED_THRESHOLD

    def test_history_verb(self) -> None:
        score = compute_structuredness_score("history --limit 10")
        assert score >= STRUCTURED_THRESHOLD

    def test_stop_alias(self) -> None:
        score = compute_structuredness_score("stop")
        assert score >= STRUCTURED_THRESHOLD

    def test_tail_alias(self) -> None:
        score = compute_structuredness_score("tail --tail 50")
        assert score >= STRUCTURED_THRESHOLD


class TestNaturalLanguageLowScore:
    """Inputs that should score low (< threshold) as natural language."""

    def test_conversational_question(self) -> None:
        score = compute_structuredness_score(
            "can you run the smoke tests on staging?"
        )
        assert score < STRUCTURED_THRESHOLD

    def test_whats_happening(self) -> None:
        score = compute_structuredness_score("what's happening right now?")
        assert score < STRUCTURED_THRESHOLD

    def test_please_run_tests(self) -> None:
        score = compute_structuredness_score(
            "please run the regression suite on production"
        )
        assert score < STRUCTURED_THRESHOLD

    def test_question_with_how(self) -> None:
        score = compute_structuredness_score(
            "how are the tests going on staging?"
        )
        assert score < STRUCTURED_THRESHOLD

    def test_want_to_see_results(self) -> None:
        score = compute_structuredness_score(
            "I want to see the results from the last run"
        )
        assert score < STRUCTURED_THRESHOLD

    def test_polite_request(self) -> None:
        score = compute_structuredness_score(
            "could you please check the test status?"
        )
        assert score < STRUCTURED_THRESHOLD

    def test_long_conversational_sentence(self) -> None:
        score = compute_structuredness_score(
            "hey, I need you to go ahead and kick off the "
            "integration tests on the staging server for me"
        )
        assert score < STRUCTURED_THRESHOLD

    def test_natural_language_with_question_mark(self) -> None:
        score = compute_structuredness_score("is there anything running?")
        assert score < STRUCTURED_THRESHOLD


class TestScoreNormalization:
    """Score is always in [0.0, 1.0] range."""

    def test_empty_string_returns_zero(self) -> None:
        score = compute_structuredness_score("")
        assert score == 0.0

    def test_whitespace_only_returns_zero(self) -> None:
        score = compute_structuredness_score("   ")
        assert score == 0.0

    def test_score_minimum_bound(self) -> None:
        """Long gibberish NL text should still produce >= 0.0."""
        score = compute_structuredness_score(
            "the quick brown fox jumps over the lazy dog and "
            "then the cat went to sleep on the couch"
        )
        assert 0.0 <= score <= 1.0

    def test_score_maximum_bound(self) -> None:
        score = compute_structuredness_score("status --verbose")
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize(
        "text",
        [
            "run deploy@host run tests",
            "check on the tests for me please",
            "status",
            "what is running?",
            "queue deploy@staging run smoke tests --priority 1",
            "I'd like to cancel the current run",
            "a",
            "------",
        ],
    )
    def test_all_scores_within_bounds(self, text: str) -> None:
        score = compute_structuredness_score(text)
        assert 0.0 <= score <= 1.0


class TestEdgeCases:
    """Boundary and edge-case inputs."""

    def test_single_unknown_word(self) -> None:
        score = compute_structuredness_score("hello")
        assert 0.0 <= score <= 1.0
        # Unknown word, no structure signals -> should be relatively low
        assert score < STRUCTURED_THRESHOLD

    def test_ssh_target_without_verb(self) -> None:
        """An SSH target alone is structural but missing a verb."""
        score = compute_structuredness_score("deploy@staging")
        assert 0.0 <= score <= 1.0

    def test_flags_without_verb(self) -> None:
        """Flags alone look somewhat structured."""
        score = compute_structuredness_score("--verbose --force")
        assert 0.0 <= score <= 1.0

    def test_mixed_nl_with_ssh_target(self) -> None:
        """NL containing an SSH target -- ambiguous."""
        score = compute_structuredness_score(
            "can you please run tests on deploy@staging?"
        )
        assert 0.0 <= score <= 1.0

    def test_verb_inside_natural_sentence(self) -> None:
        """A canonical verb embedded in a natural sentence."""
        score = compute_structuredness_score(
            "I would like to check the status of the current run"
        )
        # The verb 'check' is not at position 0 in structured sense
        assert 0.0 <= score <= 1.0

    def test_unterminated_quote(self) -> None:
        """Unterminated quotes should not crash the scorer."""
        score = compute_structuredness_score('run "unterminated')
        assert 0.0 <= score <= 1.0


class TestStructuredHigherThanNL:
    """Structured inputs should score higher than equivalent NL inputs."""

    def test_structured_run_vs_nl_run(self) -> None:
        structured = compute_structuredness_score(
            "run deploy@staging run the tests"
        )
        natural = compute_structuredness_score(
            "can you run the tests on staging?"
        )
        assert structured > natural

    def test_structured_status_vs_nl_status(self) -> None:
        structured = compute_structuredness_score("status --verbose")
        natural = compute_structuredness_score("what's the status?")
        assert structured > natural

    def test_structured_cancel_vs_nl_cancel(self) -> None:
        structured = compute_structuredness_score("cancel --force")
        natural = compute_structuredness_score(
            "please stop whatever is running"
        )
        assert structured > natural
