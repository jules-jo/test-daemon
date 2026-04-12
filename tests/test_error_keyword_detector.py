"""Tests for the error keyword regex matcher anomaly detector.

Verifies that the ErrorKeywordDetector:
- Conforms to the AnomalyDetector protocol (structural subtyping)
- Matches output lines against configurable regex patterns
- Supports multiple patterns with independent severity levels
- Produces immutable AnomalyReport records with correct metadata
- Handles case-sensitive and case-insensitive matching
- Returns the first matching pattern when multiple patterns match
- Returns no match when no patterns match
- Reports pattern name, type, severity, and context correctly
- Rejects construction with empty pattern tuples
- Is safe for concurrent read access (immutable state)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jules_daemon.monitor.anomaly_models import (
    AnomalyDetector,
    AnomalyReport,
    AnomalySeverity,
    ErrorKeywordPattern,
    PatternType,
)
from jules_daemon.monitor.error_keyword_detector import ErrorKeywordDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)

_OOM_PATTERN = ErrorKeywordPattern(
    name="oom_killer",
    regex=r"Out of memory|OOM",
    severity=AnomalySeverity.CRITICAL,
    description="Detects OOM killer events",
)

_SEGFAULT_PATTERN = ErrorKeywordPattern(
    name="segfault",
    regex=r"SIGSEGV|Segmentation fault",
    severity=AnomalySeverity.CRITICAL,
    case_sensitive=True,
    description="Detects segfaults",
)

_ASSERTION_PATTERN = ErrorKeywordPattern(
    name="assertion_failure",
    regex=r"AssertionError|assert.*failed",
    severity=AnomalySeverity.WARNING,
    description="Detects assertion failures",
)

_STACK_TRACE_PATTERN = ErrorKeywordPattern(
    name="stack_trace",
    regex=r"Traceback \(most recent call last\)",
    severity=AnomalySeverity.INFO,
    description="Detects Python stack traces",
)


def _make_detector(
    *patterns: ErrorKeywordPattern,
) -> ErrorKeywordDetector:
    """Helper to build a detector with given patterns."""
    return ErrorKeywordDetector(patterns=patterns)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify ErrorKeywordDetector satisfies AnomalyDetector protocol."""

    def test_isinstance_check(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        assert isinstance(detector, AnomalyDetector)

    def test_has_pattern_name_property(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        assert isinstance(detector.pattern_name, str)

    def test_has_pattern_type_property(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        assert detector.pattern_type is PatternType.ERROR_KEYWORD

    def test_has_match_method(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        assert callable(detector.match)

    def test_has_report_method(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        assert callable(detector.report)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Verify detector construction and validation."""

    def test_single_pattern(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        assert detector.pattern_name == "error_keyword_detector"
        assert detector.pattern_type is PatternType.ERROR_KEYWORD
        assert len(detector.patterns) == 1

    def test_multiple_patterns(self) -> None:
        detector = _make_detector(
            _OOM_PATTERN,
            _SEGFAULT_PATTERN,
            _ASSERTION_PATTERN,
        )
        assert len(detector.patterns) == 3

    def test_patterns_are_immutable_tuple(self) -> None:
        detector = _make_detector(_OOM_PATTERN, _SEGFAULT_PATTERN)
        assert isinstance(detector.patterns, tuple)

    def test_empty_patterns_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one ErrorKeywordPattern"):
            ErrorKeywordDetector(patterns=())

    def test_custom_name(self) -> None:
        detector = ErrorKeywordDetector(
            patterns=(_OOM_PATTERN,),
            name="custom_detector",
        )
        assert detector.pattern_name == "custom_detector"

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            ErrorKeywordDetector(
                patterns=(_OOM_PATTERN,),
                name="",
            )

    def test_whitespace_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            ErrorKeywordDetector(
                patterns=(_OOM_PATTERN,),
                name="   ",
            )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


class TestMatching:
    """Verify regex matching against output lines."""

    def test_match_single_pattern_positive(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        assert detector.match("Out of memory: Kill process 12345") is True

    def test_match_single_pattern_negative(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        assert detector.match("All tests passed") is False

    def test_match_case_insensitive_default(self) -> None:
        """Default case_sensitive=False should match regardless of case."""
        detector = _make_detector(_OOM_PATTERN)
        assert detector.match("out of memory") is True
        assert detector.match("OUT OF MEMORY") is True
        assert detector.match("OOM") is True
        assert detector.match("oom") is True

    def test_match_case_sensitive(self) -> None:
        """case_sensitive=True should respect exact case."""
        detector = _make_detector(_SEGFAULT_PATTERN)
        assert detector.match("SIGSEGV received") is True
        assert detector.match("Segmentation fault") is True
        assert detector.match("sigsegv received") is False
        assert detector.match("segmentation fault") is False

    def test_match_multiple_patterns_first_wins(self) -> None:
        """With multiple patterns, any match should return True."""
        detector = _make_detector(
            _OOM_PATTERN,
            _SEGFAULT_PATTERN,
            _ASSERTION_PATTERN,
        )
        # Matches first pattern
        assert detector.match("OOM detected") is True
        # Matches second pattern
        assert detector.match("SIGSEGV at 0x0") is True
        # Matches third pattern
        assert detector.match("AssertionError: expected True") is True
        # No match
        assert detector.match("Test completed successfully") is False

    def test_match_empty_line(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        assert detector.match("") is False

    def test_match_whitespace_only_line(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        assert detector.match("   \t  ") is False

    def test_match_partial_line(self) -> None:
        """Pattern should match anywhere in the line (search, not fullmatch)."""
        detector = _make_detector(_OOM_PATTERN)
        assert detector.match("[2026-04-12 12:00:00] ERROR: Out of memory") is True

    def test_match_regex_special_chars(self) -> None:
        """Regex special characters in patterns should work correctly."""
        detector = _make_detector(_STACK_TRACE_PATTERN)
        assert detector.match("Traceback (most recent call last):") is True
        assert detector.match("Traceback most recent call last") is False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    """Verify anomaly report generation."""

    def test_report_basic_fields(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        report = detector.report(
            "Out of memory: Kill process 12345",
            session_id="run-abc",
            detected_at=_NOW,
        )
        assert isinstance(report, AnomalyReport)
        assert report.pattern_type is PatternType.ERROR_KEYWORD
        assert report.severity == AnomalySeverity.CRITICAL
        assert report.session_id == "run-abc"
        assert report.detected_at == _NOW

    def test_report_pattern_name_is_matching_pattern(self) -> None:
        """Report should contain the name of the specific pattern that matched."""
        detector = _make_detector(_OOM_PATTERN, _SEGFAULT_PATTERN)
        report = detector.report(
            "SIGSEGV at 0x0",
            session_id="run-abc",
            detected_at=_NOW,
        )
        assert report.pattern_name == "segfault"

    def test_report_uses_first_matching_pattern(self) -> None:
        """When multiple patterns match, report uses the first matching one."""
        # Create a pattern that matches "error"
        broad = ErrorKeywordPattern(
            name="broad_error",
            regex=r"error",
            severity=AnomalySeverity.INFO,
        )
        specific = ErrorKeywordPattern(
            name="specific_error",
            regex=r"OutOfMemoryError",
            severity=AnomalySeverity.CRITICAL,
        )
        detector = _make_detector(broad, specific)
        # "OutOfMemoryError" matches both -- first pattern wins
        report = detector.report(
            "OutOfMemoryError: heap space",
            session_id="run-1",
            detected_at=_NOW,
        )
        assert report.pattern_name == "broad_error"
        assert report.severity == AnomalySeverity.INFO

    def test_report_message_contains_matched_text(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        report = detector.report(
            "Out of memory: Kill process 12345",
            session_id="run-abc",
            detected_at=_NOW,
        )
        assert "Out of memory" in report.message or "oom_killer" in report.message

    def test_report_context_contains_matched_text(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        report = detector.report(
            "Out of memory: Kill process 12345",
            session_id="run-abc",
            detected_at=_NOW,
        )
        assert report.context is not None
        assert "matched_text" in report.context

    def test_report_context_contains_pattern_name(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        report = detector.report(
            "Out of memory: Kill process 12345",
            session_id="run-abc",
            detected_at=_NOW,
        )
        assert report.context is not None
        assert report.context.get("pattern_name") == "oom_killer"

    def test_report_is_frozen(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        report = detector.report(
            "OOM",
            session_id="run-abc",
            detected_at=_NOW,
        )
        with pytest.raises(AttributeError):
            report.message = "changed"  # type: ignore[misc]

    def test_report_when_no_match_returns_fallback(self) -> None:
        """Report called on non-matching line should still produce a valid report.

        While callers should only call report() after match() returns True,
        the implementation should handle the edge case gracefully.
        """
        detector = _make_detector(_OOM_PATTERN)
        report = detector.report(
            "All tests passed",
            session_id="run-abc",
            detected_at=_NOW,
        )
        # Should still return a valid report (using the detector's name)
        assert isinstance(report, AnomalyReport)
        assert report.pattern_type is PatternType.ERROR_KEYWORD


# ---------------------------------------------------------------------------
# find_match helper
# ---------------------------------------------------------------------------


class TestFindMatch:
    """Verify the find_match method that returns the matching pattern."""

    def test_find_match_returns_pattern(self) -> None:
        detector = _make_detector(_OOM_PATTERN, _SEGFAULT_PATTERN)
        result = detector.find_match("OOM in process 123")
        assert result is not None
        assert result.name == "oom_killer"

    def test_find_match_returns_none_on_no_match(self) -> None:
        detector = _make_detector(_OOM_PATTERN, _SEGFAULT_PATTERN)
        result = detector.find_match("All tests passed")
        assert result is None

    def test_find_match_returns_first_match(self) -> None:
        """When multiple patterns match, returns the first one."""
        broad = ErrorKeywordPattern(name="broad", regex=r"error", severity=AnomalySeverity.INFO)
        narrow = ErrorKeywordPattern(name="narrow", regex=r"error.*fatal", severity=AnomalySeverity.CRITICAL)
        detector = _make_detector(broad, narrow)
        result = detector.find_match("error: fatal condition")
        assert result is not None
        assert result.name == "broad"


# ---------------------------------------------------------------------------
# match_all helper
# ---------------------------------------------------------------------------


class TestMatchAll:
    """Verify the match_all method that returns all matching patterns."""

    def test_match_all_returns_all_matches(self) -> None:
        broad = ErrorKeywordPattern(name="broad", regex=r"error", severity=AnomalySeverity.INFO)
        narrow = ErrorKeywordPattern(name="narrow", regex=r"fatal", severity=AnomalySeverity.CRITICAL)
        detector = _make_detector(broad, narrow)
        results = detector.match_all("error: fatal condition")
        assert len(results) == 2
        names = [p.name for p in results]
        assert "broad" in names
        assert "narrow" in names

    def test_match_all_returns_empty_on_no_match(self) -> None:
        detector = _make_detector(_OOM_PATTERN, _SEGFAULT_PATTERN)
        results = detector.match_all("All tests passed")
        assert results == ()

    def test_match_all_preserves_pattern_order(self) -> None:
        p1 = ErrorKeywordPattern(name="first", regex=r"error", severity=AnomalySeverity.INFO)
        p2 = ErrorKeywordPattern(name="second", regex=r"err", severity=AnomalySeverity.WARNING)
        detector = _make_detector(p1, p2)
        results = detector.match_all("error occurred")
        assert len(results) == 2
        assert results[0].name == "first"
        assert results[1].name == "second"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Verify edge case handling."""

    def test_very_long_line(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        long_line = "x" * 10_000 + " OOM " + "y" * 10_000
        assert detector.match(long_line) is True

    def test_multiline_input_treated_as_single_line(self) -> None:
        """Detector works on individual lines -- multiline strings still match."""
        detector = _make_detector(_OOM_PATTERN)
        # Single string with embedded newline
        assert detector.match("first line\nOOM on second line") is True

    def test_unicode_in_output(self) -> None:
        pattern = ErrorKeywordPattern(
            name="unicode_err",
            regex=r"erreur",
            severity=AnomalySeverity.WARNING,
        )
        detector = _make_detector(pattern)
        assert detector.match("erreur: fichier non trouv\u00e9") is True

    def test_repr_is_readable(self) -> None:
        detector = _make_detector(_OOM_PATTERN)
        r = repr(detector)
        assert "ErrorKeywordDetector" in r
        assert "oom_killer" in r or "1 pattern" in r
