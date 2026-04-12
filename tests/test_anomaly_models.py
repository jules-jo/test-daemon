"""Tests for anomaly pattern configuration schema and detector interface.

Verifies that the anomaly detection models:
- Define immutable configuration schemas for three pattern types:
  (a) error keyword patterns (regex-based string matching)
  (b) failure-rate threshold patterns (count/window-based)
  (c) stall timeout patterns (elapsed-time-based)
- Validate configuration fields at construction time (frozen dataclasses)
- Provide a common AnomalyDetector protocol with match() and report() methods
- Produce immutable AnomalyReport records containing match metadata
- Support severity levels (INFO, WARNING, CRITICAL) for anomaly classification
- Never mutate existing data -- all state transitions produce new instances
- Support composite pattern configuration via AnomalyPatternConfig
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from jules_daemon.monitor.anomaly_models import (
    AnomalyDetector,
    AnomalyPatternConfig,
    AnomalyReport,
    AnomalySeverity,
    ErrorKeywordPattern,
    FailureRatePattern,
    PatternType,
    StallTimeoutPattern,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# PatternType enum tests
# ---------------------------------------------------------------------------


class TestPatternType:
    """Verify the pattern type classification enum."""

    def test_enum_values(self) -> None:
        assert PatternType.ERROR_KEYWORD.value == "error_keyword"
        assert PatternType.FAILURE_RATE.value == "failure_rate"
        assert PatternType.STALL_TIMEOUT.value == "stall_timeout"

    def test_all_members_present(self) -> None:
        members = {m.value for m in PatternType}
        assert members == {"error_keyword", "failure_rate", "stall_timeout"}


# ---------------------------------------------------------------------------
# AnomalySeverity enum tests
# ---------------------------------------------------------------------------


class TestAnomalySeverity:
    """Verify the anomaly severity classification enum."""

    def test_enum_values(self) -> None:
        assert AnomalySeverity.INFO.value == "info"
        assert AnomalySeverity.WARNING.value == "warning"
        assert AnomalySeverity.CRITICAL.value == "critical"

    def test_ordering_by_value(self) -> None:
        # Severity should be comparable for threshold logic
        severities = sorted(AnomalySeverity, key=lambda s: s.numeric_level)
        assert severities == [
            AnomalySeverity.INFO,
            AnomalySeverity.WARNING,
            AnomalySeverity.CRITICAL,
        ]


# ---------------------------------------------------------------------------
# ErrorKeywordPattern tests
# ---------------------------------------------------------------------------


class TestErrorKeywordPattern:
    """Verify the error keyword pattern configuration schema."""

    def test_construction_with_defaults(self) -> None:
        pattern = ErrorKeywordPattern(
            name="oom_killer",
            regex=r"Out of memory|OOM",
        )
        assert pattern.name == "oom_killer"
        assert pattern.regex == r"Out of memory|OOM"
        assert pattern.severity == AnomalySeverity.WARNING
        assert pattern.pattern_type is PatternType.ERROR_KEYWORD
        assert pattern.case_sensitive is False
        assert pattern.description == ""

    def test_construction_with_all_fields(self) -> None:
        pattern = ErrorKeywordPattern(
            name="segfault",
            regex=r"Segmentation fault",
            severity=AnomalySeverity.CRITICAL,
            case_sensitive=True,
            description="Detects segfault in SSH output",
        )
        assert pattern.severity == AnomalySeverity.CRITICAL
        assert pattern.case_sensitive is True
        assert pattern.description == "Detects segfault in SSH output"

    def test_frozen(self) -> None:
        pattern = ErrorKeywordPattern(name="test", regex=r"error")
        with pytest.raises(AttributeError):
            pattern.name = "changed"  # type: ignore[misc]

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            ErrorKeywordPattern(name="", regex=r"error")

    def test_whitespace_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            ErrorKeywordPattern(name="   ", regex=r"error")

    def test_empty_regex_rejected(self) -> None:
        with pytest.raises(ValueError, match="regex must not be empty"):
            ErrorKeywordPattern(name="test", regex="")

    def test_invalid_regex_rejected(self) -> None:
        with pytest.raises(ValueError, match="regex is not a valid regular expression"):
            ErrorKeywordPattern(name="test", regex=r"[invalid")

    def test_compiled_pattern_available(self) -> None:
        pattern = ErrorKeywordPattern(
            name="test",
            regex=r"ERROR:\s+\d+",
            case_sensitive=True,
        )
        compiled = pattern.compiled_regex
        assert compiled.pattern == r"ERROR:\s+\d+"
        # case_sensitive=True means no IGNORECASE flag
        assert compiled.match("ERROR: 42") is not None
        assert compiled.match("error: 42") is None

    def test_compiled_pattern_case_insensitive(self) -> None:
        pattern = ErrorKeywordPattern(
            name="test",
            regex=r"FATAL",
            case_sensitive=False,
        )
        compiled = pattern.compiled_regex
        assert compiled.match("fatal") is not None
        assert compiled.match("FATAL") is not None


# ---------------------------------------------------------------------------
# FailureRatePattern tests
# ---------------------------------------------------------------------------


class TestFailureRatePattern:
    """Verify the failure-rate threshold pattern configuration schema."""

    def test_construction_with_defaults(self) -> None:
        pattern = FailureRatePattern(
            name="high_failure_rate",
            threshold_count=5,
            window_seconds=60.0,
        )
        assert pattern.name == "high_failure_rate"
        assert pattern.threshold_count == 5
        assert pattern.window_seconds == 60.0
        assert pattern.severity == AnomalySeverity.WARNING
        assert pattern.pattern_type is PatternType.FAILURE_RATE
        assert pattern.description == ""

    def test_frozen(self) -> None:
        pattern = FailureRatePattern(
            name="test",
            threshold_count=3,
            window_seconds=30.0,
        )
        with pytest.raises(AttributeError):
            pattern.threshold_count = 10  # type: ignore[misc]

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            FailureRatePattern(name="", threshold_count=5, window_seconds=60.0)

    def test_zero_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="threshold_count must be positive"):
            FailureRatePattern(
                name="test", threshold_count=0, window_seconds=60.0
            )

    def test_negative_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="threshold_count must be positive"):
            FailureRatePattern(
                name="test", threshold_count=-1, window_seconds=60.0
            )

    def test_zero_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="window_seconds must be positive"):
            FailureRatePattern(
                name="test", threshold_count=5, window_seconds=0.0
            )

    def test_negative_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="window_seconds must be positive"):
            FailureRatePattern(
                name="test", threshold_count=5, window_seconds=-10.0
            )


# ---------------------------------------------------------------------------
# StallTimeoutPattern tests
# ---------------------------------------------------------------------------


class TestStallTimeoutPattern:
    """Verify the stall timeout pattern configuration schema."""

    def test_construction_with_defaults(self) -> None:
        pattern = StallTimeoutPattern(
            name="output_stall",
            timeout_seconds=300.0,
        )
        assert pattern.name == "output_stall"
        assert pattern.timeout_seconds == 300.0
        assert pattern.severity == AnomalySeverity.WARNING
        assert pattern.pattern_type is PatternType.STALL_TIMEOUT
        assert pattern.description == ""

    def test_custom_severity(self) -> None:
        pattern = StallTimeoutPattern(
            name="long_stall",
            timeout_seconds=600.0,
            severity=AnomalySeverity.CRITICAL,
            description="No output for 10 minutes",
        )
        assert pattern.severity == AnomalySeverity.CRITICAL
        assert pattern.description == "No output for 10 minutes"

    def test_frozen(self) -> None:
        pattern = StallTimeoutPattern(name="test", timeout_seconds=60.0)
        with pytest.raises(AttributeError):
            pattern.timeout_seconds = 120.0  # type: ignore[misc]

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            StallTimeoutPattern(name="", timeout_seconds=60.0)

    def test_zero_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            StallTimeoutPattern(name="test", timeout_seconds=0.0)

    def test_negative_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            StallTimeoutPattern(name="test", timeout_seconds=-5.0)


# ---------------------------------------------------------------------------
# AnomalyReport tests
# ---------------------------------------------------------------------------


class TestAnomalyReport:
    """Verify the immutable anomaly report record."""

    def test_construction(self) -> None:
        report = AnomalyReport(
            pattern_name="oom_killer",
            pattern_type=PatternType.ERROR_KEYWORD,
            severity=AnomalySeverity.WARNING,
            message="Matched 'Out of memory' in output at line 42",
            detected_at=_NOW,
            session_id="run-abc",
        )
        assert report.pattern_name == "oom_killer"
        assert report.pattern_type is PatternType.ERROR_KEYWORD
        assert report.severity == AnomalySeverity.WARNING
        assert "Out of memory" in report.message
        assert report.detected_at == _NOW
        assert report.session_id == "run-abc"
        assert report.context is None

    def test_construction_with_context(self) -> None:
        ctx = {"line_number": 42, "matched_text": "Out of memory: Kill process"}
        report = AnomalyReport(
            pattern_name="oom",
            pattern_type=PatternType.ERROR_KEYWORD,
            severity=AnomalySeverity.CRITICAL,
            message="OOM detected",
            detected_at=_NOW,
            session_id="run-xyz",
            context=ctx,
        )
        assert report.context == ctx

    def test_frozen(self) -> None:
        report = AnomalyReport(
            pattern_name="test",
            pattern_type=PatternType.STALL_TIMEOUT,
            severity=AnomalySeverity.INFO,
            message="test",
            detected_at=_NOW,
            session_id="run-1",
        )
        with pytest.raises(AttributeError):
            report.message = "changed"  # type: ignore[misc]

    def test_empty_pattern_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="pattern_name must not be empty"):
            AnomalyReport(
                pattern_name="",
                pattern_type=PatternType.ERROR_KEYWORD,
                severity=AnomalySeverity.WARNING,
                message="test",
                detected_at=_NOW,
                session_id="run-1",
            )

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(ValueError, match="message must not be empty"):
            AnomalyReport(
                pattern_name="test",
                pattern_type=PatternType.ERROR_KEYWORD,
                severity=AnomalySeverity.WARNING,
                message="",
                detected_at=_NOW,
                session_id="run-1",
            )

    def test_empty_session_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="session_id must not be empty"):
            AnomalyReport(
                pattern_name="test",
                pattern_type=PatternType.ERROR_KEYWORD,
                severity=AnomalySeverity.WARNING,
                message="test msg",
                detected_at=_NOW,
                session_id="",
            )


# ---------------------------------------------------------------------------
# AnomalyPatternConfig tests
# ---------------------------------------------------------------------------


class TestAnomalyPatternConfig:
    """Verify the composite pattern configuration container."""

    def test_empty_construction(self) -> None:
        config = AnomalyPatternConfig()
        assert config.error_keywords == ()
        assert config.failure_rates == ()
        assert config.stall_timeouts == ()
        assert config.total_patterns == 0

    def test_construction_with_patterns(self) -> None:
        kw = ErrorKeywordPattern(name="oom", regex=r"OOM")
        fr = FailureRatePattern(name="rate", threshold_count=5, window_seconds=60.0)
        st = StallTimeoutPattern(name="stall", timeout_seconds=300.0)

        config = AnomalyPatternConfig(
            error_keywords=(kw,),
            failure_rates=(fr,),
            stall_timeouts=(st,),
        )
        assert len(config.error_keywords) == 1
        assert len(config.failure_rates) == 1
        assert len(config.stall_timeouts) == 1
        assert config.total_patterns == 3

    def test_frozen(self) -> None:
        config = AnomalyPatternConfig()
        with pytest.raises(AttributeError):
            config.error_keywords = ()  # type: ignore[misc]

    def test_all_patterns_returns_union(self) -> None:
        kw1 = ErrorKeywordPattern(name="oom", regex=r"OOM")
        kw2 = ErrorKeywordPattern(name="segfault", regex=r"SIGSEGV")
        fr = FailureRatePattern(name="rate", threshold_count=5, window_seconds=60.0)
        st = StallTimeoutPattern(name="stall", timeout_seconds=300.0)

        config = AnomalyPatternConfig(
            error_keywords=(kw1, kw2),
            failure_rates=(fr,),
            stall_timeouts=(st,),
        )
        all_patterns = config.all_patterns
        assert len(all_patterns) == 4
        # Verify ordering: keywords first, then failure_rates, then stall_timeouts
        assert all_patterns[0].name == "oom"
        assert all_patterns[1].name == "segfault"
        assert all_patterns[2].name == "rate"
        assert all_patterns[3].name == "stall"

    def test_duplicate_names_rejected(self) -> None:
        kw = ErrorKeywordPattern(name="dup", regex=r"error")
        fr = FailureRatePattern(name="dup", threshold_count=5, window_seconds=60.0)
        with pytest.raises(ValueError, match="Duplicate pattern name"):
            AnomalyPatternConfig(
                error_keywords=(kw,),
                failure_rates=(fr,),
            )

    def test_get_pattern_by_name(self) -> None:
        kw = ErrorKeywordPattern(name="oom", regex=r"OOM")
        fr = FailureRatePattern(name="rate", threshold_count=5, window_seconds=60.0)
        config = AnomalyPatternConfig(
            error_keywords=(kw,),
            failure_rates=(fr,),
        )
        assert config.get_pattern("oom") is kw
        assert config.get_pattern("rate") is fr
        assert config.get_pattern("nonexistent") is None


# ---------------------------------------------------------------------------
# AnomalyDetector protocol tests
# ---------------------------------------------------------------------------


class TestAnomalyDetectorProtocol:
    """Verify the AnomalyDetector protocol interface contract."""

    def test_protocol_has_match_method(self) -> None:
        """AnomalyDetector defines a match method that returns bool."""
        import inspect
        hints = {
            name: member
            for name, member in inspect.getmembers(AnomalyDetector)
            if not name.startswith("_")
        }
        assert "match" in hints
        assert "report" in hints

    def test_concrete_implementation_satisfies_protocol(self) -> None:
        """A concrete class implementing match/report satisfies AnomalyDetector."""

        class ConcreteDetector:
            """Minimal implementation satisfying AnomalyDetector."""

            @property
            def pattern_name(self) -> str:
                return "test_detector"

            @property
            def pattern_type(self) -> PatternType:
                return PatternType.ERROR_KEYWORD

            def match(self, output_line: str) -> bool:
                return "error" in output_line.lower()

            def report(
                self,
                output_line: str,
                *,
                session_id: str,
                detected_at: datetime,
            ) -> AnomalyReport:
                return AnomalyReport(
                    pattern_name=self.pattern_name,
                    pattern_type=self.pattern_type,
                    severity=AnomalySeverity.WARNING,
                    message=f"Matched error in: {output_line}",
                    detected_at=detected_at,
                    session_id=session_id,
                )

        detector = ConcreteDetector()

        # Structural subtyping: verify it satisfies the protocol
        assert isinstance(detector, AnomalyDetector)

        # Verify match works
        assert detector.match("ERROR: something failed") is True
        assert detector.match("All tests passed") is False

        # Verify report works
        report = detector.report(
            "ERROR: something failed",
            session_id="run-abc",
            detected_at=_NOW,
        )
        assert isinstance(report, AnomalyReport)
        assert report.pattern_name == "test_detector"
        assert report.session_id == "run-abc"

    def test_non_conforming_class_fails_isinstance(self) -> None:
        """A class without match/report does NOT satisfy AnomalyDetector."""

        class NotADetector:
            pass

        assert not isinstance(NotADetector(), AnomalyDetector)

    def test_partial_implementation_fails_isinstance(self) -> None:
        """A class with only match (no report) does NOT satisfy AnomalyDetector."""

        class PartialDetector:
            @property
            def pattern_name(self) -> str:
                return "partial"

            @property
            def pattern_type(self) -> PatternType:
                return PatternType.ERROR_KEYWORD

            def match(self, output_line: str) -> bool:
                return False

        assert not isinstance(PartialDetector(), AnomalyDetector)
