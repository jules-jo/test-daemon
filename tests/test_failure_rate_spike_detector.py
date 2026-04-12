"""Tests for the failure rate spike tracker anomaly detector.

Verifies that the FailureRateSpikeDetector:
- Conforms to the AnomalyDetector protocol (structural subtyping)
- Uses a sliding window counter to track failure events over time
- Detects spikes when failure count in the window exceeds the threshold
- Evicts timestamps older than the configured window_seconds
- Matches output lines against a configurable failure regex
- Produces immutable AnomalyReport records with correct metadata
- Supports explicit failure recording via record_failure()
- Provides current_count, threshold, and window_seconds properties
- Resets state cleanly via reset()
- Rejects construction with invalid parameters
- Uses injectable time function for deterministic testing
- Is safe for concurrent access (uses threading lock)
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from jules_daemon.monitor.anomaly_models import (
    AnomalyDetector,
    AnomalyReport,
    AnomalySeverity,
    FailureRatePattern,
    PatternType,
)
from jules_daemon.monitor.failure_rate_spike_detector import (
    FailureRateSpikeDetector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW: Final[datetime] = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)

_DEFAULT_PATTERN: Final[FailureRatePattern] = FailureRatePattern(
    name="high_fail",
    threshold_count=3,
    window_seconds=60.0,
    severity=AnomalySeverity.WARNING,
    description="Detects high failure rate spikes",
)

_CRITICAL_PATTERN: Final[FailureRatePattern] = FailureRatePattern(
    name="critical_fail",
    threshold_count=5,
    window_seconds=120.0,
    severity=AnomalySeverity.CRITICAL,
    description="Critical failure rate spike",
)

_TIGHT_PATTERN: Final[FailureRatePattern] = FailureRatePattern(
    name="tight_window",
    threshold_count=2,
    window_seconds=10.0,
    severity=AnomalySeverity.WARNING,
)


def _make_detector(
    pattern: FailureRatePattern = _DEFAULT_PATTERN,
    *,
    failure_regex: str = r"(?i)FAIL|ERROR|FAILED",
    name: str = "failure_rate_spike_detector",
    time_func: Callable[[], datetime] | None = None,
) -> FailureRateSpikeDetector:
    """Helper to build a detector with given config."""
    kwargs: dict[str, object] = {
        "pattern": pattern,
        "failure_regex": failure_regex,
        "name": name,
    }
    if time_func is not None:
        kwargs["time_func"] = time_func
    return FailureRateSpikeDetector(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify FailureRateSpikeDetector satisfies AnomalyDetector protocol."""

    def test_isinstance_check(self) -> None:
        detector = _make_detector()
        assert isinstance(detector, AnomalyDetector)

    def test_has_pattern_name_property(self) -> None:
        detector = _make_detector()
        assert isinstance(detector.pattern_name, str)

    def test_has_pattern_type_property(self) -> None:
        detector = _make_detector()
        assert detector.pattern_type is PatternType.FAILURE_RATE

    def test_has_match_method(self) -> None:
        detector = _make_detector()
        assert callable(detector.match)

    def test_has_report_method(self) -> None:
        detector = _make_detector()
        assert callable(detector.report)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Verify detector construction and validation."""

    def test_default_construction(self) -> None:
        detector = _make_detector()
        assert detector.pattern_name == "failure_rate_spike_detector"
        assert detector.pattern_type is PatternType.FAILURE_RATE
        assert detector.threshold == 3
        assert detector.window_seconds == 60.0
        assert detector.current_count == 0

    def test_custom_name(self) -> None:
        detector = _make_detector(name="my_detector")
        assert detector.pattern_name == "my_detector"

    def test_custom_pattern(self) -> None:
        detector = _make_detector(_CRITICAL_PATTERN)
        assert detector.threshold == 5
        assert detector.window_seconds == 120.0

    def test_custom_failure_regex(self) -> None:
        detector = _make_detector(failure_regex=r"BROKEN|CRASHED")
        assert detector.failure_regex == r"BROKEN|CRASHED"

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            _make_detector(name="")

    def test_whitespace_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            _make_detector(name="   ")

    def test_empty_failure_regex_rejected(self) -> None:
        with pytest.raises(ValueError, match="failure_regex must not be empty"):
            _make_detector(failure_regex="")

    def test_whitespace_failure_regex_rejected(self) -> None:
        with pytest.raises(ValueError, match="failure_regex must not be empty"):
            _make_detector(failure_regex="   ")

    def test_invalid_failure_regex_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="failure_regex is not a valid regular expression",
        ):
            _make_detector(failure_regex=r"[invalid")

    def test_compiled_regex_available(self) -> None:
        detector = _make_detector(failure_regex=r"FAIL")
        assert isinstance(detector.compiled_failure_regex, re.Pattern)
        assert detector.compiled_failure_regex.search("test FAIL") is not None


# ---------------------------------------------------------------------------
# Sliding window -- match()
# ---------------------------------------------------------------------------


class TestSlidingWindowMatch:
    """Verify sliding window counting via match() calls."""

    def test_no_match_on_non_failure_line(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        assert detector.match("All tests passed") is False
        assert detector.current_count == 0

    def test_no_spike_below_threshold(self) -> None:
        """Failure lines below threshold should not trigger a spike."""
        detector = _make_detector(time_func=lambda: _NOW)
        # threshold_count=3, so 2 failures should not trigger
        assert detector.match("test FAIL") is False
        assert detector.current_count == 1
        assert detector.match("another ERROR") is False
        assert detector.current_count == 2

    def test_spike_at_threshold(self) -> None:
        """Reaching the threshold should trigger the spike."""
        detector = _make_detector(time_func=lambda: _NOW)
        assert detector.match("FAIL 1") is False
        assert detector.match("FAIL 2") is False
        assert detector.match("FAIL 3") is True  # threshold=3
        assert detector.current_count == 3

    def test_spike_above_threshold(self) -> None:
        """Exceeding the threshold should continue to trigger."""
        detector = _make_detector(time_func=lambda: _NOW)
        detector.match("FAIL 1")
        detector.match("FAIL 2")
        detector.match("FAIL 3")
        assert detector.match("FAIL 4") is True
        assert detector.current_count == 4

    def test_non_failure_lines_do_not_count(self) -> None:
        """Lines not matching failure_regex should not be counted."""
        detector = _make_detector(time_func=lambda: _NOW)
        detector.match("FAIL 1")
        detector.match("FAIL 2")
        detector.match("ok line")
        detector.match("success")
        detector.match("passed")
        assert detector.current_count == 2
        assert detector.match("FAIL 3") is True

    def test_case_insensitive_default_regex(self) -> None:
        """Default regex includes (?i) flag for case-insensitive matching."""
        detector = _make_detector(time_func=lambda: _NOW)
        detector.match("fail 1")
        detector.match("error 2")
        assert detector.match("FAILED 3") is True


# ---------------------------------------------------------------------------
# Sliding window -- expiration
# ---------------------------------------------------------------------------


class TestSlidingWindowExpiration:
    """Verify that timestamps are evicted when they fall outside the window."""

    def test_old_failures_evicted(self) -> None:
        """Failures older than window_seconds should be evicted."""
        current_time = _NOW
        detector = _make_detector(
            _TIGHT_PATTERN,  # window=10s, threshold=2
            time_func=lambda: current_time,
        )
        # Record 2 failures at _NOW
        detector.match("FAIL 1")
        detector.match("FAIL 2")
        assert detector.current_count == 2

        # Advance time past the window
        current_time = _NOW + timedelta(seconds=11)
        # The old failures should be evicted on the next check
        assert detector.match("ok line") is False
        assert detector.current_count == 0

    def test_partial_eviction(self) -> None:
        """Only expired failures are evicted; recent ones remain."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = _make_detector(
            _TIGHT_PATTERN,  # window=10s, threshold=2
            time_func=time_func,
        )
        # Record failure at t=0
        detector.match("FAIL 1")
        assert detector.current_count == 1

        # Advance to t=6s, record another failure
        times[0] = _NOW + timedelta(seconds=6)
        detector.match("FAIL 2")
        assert detector.current_count == 2

        # Advance to t=11s: first failure expired, second still in window
        times[0] = _NOW + timedelta(seconds=11)
        assert detector.match("ok line") is False
        assert detector.current_count == 1

    def test_all_evicted_resets_spike(self) -> None:
        """After all failures expire, spike detection resets."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = _make_detector(
            _TIGHT_PATTERN,  # window=10s, threshold=2
            time_func=time_func,
        )
        # Trigger a spike
        detector.match("FAIL 1")
        assert detector.match("FAIL 2") is True

        # Advance time past window -- all evicted
        times[0] = _NOW + timedelta(seconds=11)
        detector.match("ok line")
        assert detector.current_count == 0

        # New failures should start from scratch
        assert detector.match("FAIL 3") is False
        assert detector.current_count == 1

    def test_exact_boundary_not_evicted(self) -> None:
        """A failure exactly at the window boundary is NOT evicted."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = _make_detector(
            _TIGHT_PATTERN,  # window=10s, threshold=2
            time_func=time_func,
        )
        detector.match("FAIL 1")

        # Advance to exactly 10s (the boundary)
        times[0] = _NOW + timedelta(seconds=10)
        detector.match("ok line")
        # The failure at t=0 should still be within the window at t=10
        # because window is [now - 10, now], inclusive
        assert detector.current_count == 1


# ---------------------------------------------------------------------------
# record_failure() method
# ---------------------------------------------------------------------------


class TestRecordFailure:
    """Verify explicit failure recording via record_failure()."""

    def test_record_failure_increments_count(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        assert detector.current_count == 0
        detector.record_failure(timestamp=_NOW)
        assert detector.current_count == 1

    def test_record_failure_returns_spike_status(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        assert detector.record_failure(timestamp=_NOW) is False
        assert detector.record_failure(timestamp=_NOW) is False
        assert detector.record_failure(timestamp=_NOW) is True

    def test_record_failure_uses_time_func_when_no_timestamp(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        detector.record_failure()
        assert detector.current_count == 1

    def test_record_failure_with_explicit_timestamps(self) -> None:
        """Explicit timestamps are used instead of time_func."""
        detector = _make_detector(
            _TIGHT_PATTERN,  # window=10s, threshold=2
            time_func=lambda: _NOW + timedelta(seconds=20),
        )
        # Record failures at old timestamps -- they should be evicted
        detector.record_failure(timestamp=_NOW)
        # Now check -- the time_func returns _NOW+20s, so the failure at _NOW
        # (20s ago) is outside the 10s window
        assert detector.current_count == 0

    def test_record_failure_respects_window(self) -> None:
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = _make_detector(
            _TIGHT_PATTERN,  # window=10s, threshold=2
            time_func=time_func,
        )
        detector.record_failure(timestamp=_NOW)
        detector.record_failure(timestamp=_NOW + timedelta(seconds=5))
        assert detector.current_count == 2

        # Advance time past first failure's window
        times[0] = _NOW + timedelta(seconds=11)
        assert detector.current_count == 1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    """Verify anomaly report generation."""

    def test_report_basic_fields(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        # Trigger a spike
        detector.match("FAIL 1")
        detector.match("FAIL 2")
        detector.match("FAIL 3")

        report = detector.report(
            "FAIL 3",
            session_id="run-abc",
            detected_at=_NOW,
        )
        assert isinstance(report, AnomalyReport)
        assert report.pattern_name == "high_fail"
        assert report.pattern_type is PatternType.FAILURE_RATE
        assert report.severity == AnomalySeverity.WARNING
        assert report.session_id == "run-abc"
        assert report.detected_at == _NOW

    def test_report_message_describes_spike(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        detector.match("FAIL 1")
        detector.match("FAIL 2")
        detector.match("FAIL 3")

        report = detector.report(
            "FAIL 3",
            session_id="run-abc",
            detected_at=_NOW,
        )
        # Message should contain useful information about the spike
        assert "high_fail" in report.message or "3" in report.message

    def test_report_context_contains_metadata(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        detector.match("FAIL 1")
        detector.match("FAIL 2")
        detector.match("FAIL 3")

        report = detector.report(
            "FAIL 3",
            session_id="run-abc",
            detected_at=_NOW,
        )
        assert report.context is not None
        assert "failure_count" in report.context
        assert "threshold" in report.context
        assert "window_seconds" in report.context
        assert report.context["failure_count"] == 3
        assert report.context["threshold"] == 3
        assert report.context["window_seconds"] == 60.0

    def test_report_context_contains_output_line(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        detector.match("FAIL 1")
        detector.match("FAIL 2")
        detector.match("FAIL 3")

        report = detector.report(
            "FAIL 3",
            session_id="run-abc",
            detected_at=_NOW,
        )
        assert report.context is not None
        assert report.context.get("output_line") == "FAIL 3"

    def test_report_uses_pattern_severity(self) -> None:
        """Report severity comes from the FailureRatePattern config."""
        detector = _make_detector(_CRITICAL_PATTERN, time_func=lambda: _NOW)
        report = detector.report(
            "FAIL line",
            session_id="run-1",
            detected_at=_NOW,
        )
        assert report.severity == AnomalySeverity.CRITICAL

    def test_report_is_frozen(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        report = detector.report(
            "FAIL line",
            session_id="run-abc",
            detected_at=_NOW,
        )
        with pytest.raises(AttributeError):
            report.message = "changed"  # type: ignore[misc]

    def test_report_when_no_spike(self) -> None:
        """Report called below threshold still produces a valid report."""
        detector = _make_detector(time_func=lambda: _NOW)
        detector.match("FAIL 1")
        report = detector.report(
            "FAIL 1",
            session_id="run-abc",
            detected_at=_NOW,
        )
        assert isinstance(report, AnomalyReport)
        assert report.pattern_type is PatternType.FAILURE_RATE


# ---------------------------------------------------------------------------
# reset() method
# ---------------------------------------------------------------------------


class TestReset:
    """Verify state reset functionality."""

    def test_reset_clears_all_failures(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        detector.match("FAIL 1")
        detector.match("FAIL 2")
        assert detector.current_count == 2

        detector.reset()
        assert detector.current_count == 0

    def test_reset_allows_fresh_counting(self) -> None:
        """After reset, spike detection starts from scratch."""
        detector = _make_detector(time_func=lambda: _NOW)
        detector.match("FAIL 1")
        detector.match("FAIL 2")
        assert detector.match("FAIL 3") is True

        detector.reset()
        assert detector.match("FAIL 4") is False
        assert detector.current_count == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Verify thread-safe access to mutable state."""

    def test_concurrent_match_calls(self) -> None:
        """Multiple threads can call match() without data corruption."""
        pattern = FailureRatePattern(
            name="concurrent_test",
            threshold_count=100,
            window_seconds=60.0,
        )
        detector = _make_detector(pattern, time_func=lambda: _NOW)

        barrier = threading.Barrier(4)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=5.0)
                for i in range(50):
                    detector.match(f"FAIL {i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors
        # 4 threads x 50 matches = 200 failures recorded
        assert detector.current_count == 200

    def test_concurrent_record_and_reset(self) -> None:
        """Concurrent record_failure and reset should not raise."""
        detector = _make_detector(time_func=lambda: _NOW)

        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def recorder() -> None:
            try:
                barrier.wait(timeout=5.0)
                for _ in range(100):
                    detector.record_failure(timestamp=_NOW)
            except Exception as exc:
                errors.append(exc)

        def resetter() -> None:
            try:
                barrier.wait(timeout=5.0)
                for _ in range(100):
                    detector.reset()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=recorder)
        t2 = threading.Thread(target=resetter)
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert not errors


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Verify edge case handling."""

    def test_threshold_of_one(self) -> None:
        """With threshold=1, the first failure immediately triggers."""
        pattern = FailureRatePattern(
            name="instant",
            threshold_count=1,
            window_seconds=60.0,
        )
        detector = _make_detector(pattern, time_func=lambda: _NOW)
        assert detector.match("FAIL") is True
        assert detector.current_count == 1

    def test_very_large_window(self) -> None:
        """Failures within a very large window are all counted."""
        pattern = FailureRatePattern(
            name="large_window",
            threshold_count=3,
            window_seconds=86400.0,  # 24 hours
        )
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = _make_detector(pattern, time_func=time_func)
        detector.match("FAIL 1")

        times[0] = _NOW + timedelta(hours=12)
        detector.match("FAIL 2")

        times[0] = _NOW + timedelta(hours=23)
        assert detector.match("FAIL 3") is True
        assert detector.current_count == 3

    def test_empty_output_line(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        assert detector.match("") is False
        assert detector.current_count == 0

    def test_whitespace_only_line(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        assert detector.match("   \t  ") is False
        assert detector.current_count == 0

    def test_very_long_line(self) -> None:
        detector = _make_detector(time_func=lambda: _NOW)
        long_line = "x" * 10_000 + " FAIL " + "y" * 10_000
        assert detector.match(long_line) is False  # below threshold
        assert detector.current_count == 1

    def test_unicode_in_output(self) -> None:
        detector = _make_detector(
            failure_regex=r"echec",
            time_func=lambda: _NOW,
        )
        detector.match("echec: fichier non trouve")
        assert detector.current_count == 1

    def test_repr_is_readable(self) -> None:
        detector = _make_detector()
        r = repr(detector)
        assert "FailureRateSpikeDetector" in r
        assert "high_fail" in r or "failure_rate_spike_detector" in r
