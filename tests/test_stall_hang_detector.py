"""Tests for the stall/hang detector anomaly detector.

Verifies that the StallHangDetector:
- Conforms to the AnomalyDetector protocol (structural subtyping)
- Tracks elapsed time since last output line was received
- Detects stalls when elapsed time exceeds the configurable timeout threshold
- Updates the last-output timestamp on every match() call (output received)
- Provides a separate check_stall() method for periodic polling
- Supports explicit output recording via record_output()
- Produces immutable AnomalyReport records with correct metadata
- Resets state cleanly via reset()
- Rejects construction with invalid parameters
- Uses injectable time function for deterministic testing
- Is safe for concurrent access (uses threading lock)
- Reports elapsed time and timeout threshold in context
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from jules_daemon.monitor.anomaly_models import (
    AnomalyDetector,
    AnomalyReport,
    AnomalySeverity,
    PatternType,
    StallTimeoutPattern,
)
from jules_daemon.monitor.stall_hang_detector import StallHangDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW: Final[datetime] = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)

_DEFAULT_PATTERN: Final[StallTimeoutPattern] = StallTimeoutPattern(
    name="output_stall",
    timeout_seconds=300.0,
    severity=AnomalySeverity.WARNING,
    description="No output for 5 minutes",
)

_CRITICAL_PATTERN: Final[StallTimeoutPattern] = StallTimeoutPattern(
    name="long_stall",
    timeout_seconds=600.0,
    severity=AnomalySeverity.CRITICAL,
    description="No output for 10 minutes",
)

_SHORT_PATTERN: Final[StallTimeoutPattern] = StallTimeoutPattern(
    name="short_stall",
    timeout_seconds=10.0,
    severity=AnomalySeverity.WARNING,
)


def _make_detector(
    pattern: StallTimeoutPattern = _DEFAULT_PATTERN,
    *,
    name: str = "stall_hang_detector",
    time_func: type | None = None,
) -> StallHangDetector:
    """Helper to build a detector with given config."""
    kwargs: dict[str, object] = {
        "pattern": pattern,
        "name": name,
    }
    if time_func is not None:
        kwargs["time_func"] = time_func
    return StallHangDetector(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify StallHangDetector satisfies AnomalyDetector protocol."""

    def test_isinstance_check(self) -> None:
        detector = StallHangDetector(pattern=_DEFAULT_PATTERN, time_func=lambda: _NOW)
        assert isinstance(detector, AnomalyDetector)

    def test_has_pattern_name_property(self) -> None:
        detector = StallHangDetector(pattern=_DEFAULT_PATTERN, time_func=lambda: _NOW)
        assert isinstance(detector.pattern_name, str)

    def test_has_pattern_type_property(self) -> None:
        detector = StallHangDetector(pattern=_DEFAULT_PATTERN, time_func=lambda: _NOW)
        assert detector.pattern_type is PatternType.STALL_TIMEOUT

    def test_has_match_method(self) -> None:
        detector = StallHangDetector(pattern=_DEFAULT_PATTERN, time_func=lambda: _NOW)
        assert callable(detector.match)

    def test_has_report_method(self) -> None:
        detector = StallHangDetector(pattern=_DEFAULT_PATTERN, time_func=lambda: _NOW)
        assert callable(detector.report)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Verify detector construction and validation."""

    def test_default_construction(self) -> None:
        detector = StallHangDetector(pattern=_DEFAULT_PATTERN, time_func=lambda: _NOW)
        assert detector.pattern_name == "stall_hang_detector"
        assert detector.pattern_type is PatternType.STALL_TIMEOUT
        assert detector.timeout_seconds == 300.0

    def test_custom_name(self) -> None:
        detector = StallHangDetector(
            pattern=_DEFAULT_PATTERN,
            name="my_stall_detector",
            time_func=lambda: _NOW,
        )
        assert detector.pattern_name == "my_stall_detector"

    def test_custom_pattern(self) -> None:
        detector = StallHangDetector(
            pattern=_CRITICAL_PATTERN,
            time_func=lambda: _NOW,
        )
        assert detector.timeout_seconds == 600.0

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            StallHangDetector(pattern=_DEFAULT_PATTERN, name="", time_func=lambda: _NOW)

    def test_whitespace_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            StallHangDetector(
                pattern=_DEFAULT_PATTERN, name="   ", time_func=lambda: _NOW
            )

    def test_last_output_time_initialized_to_construction_time(self) -> None:
        """The detector should initialize last_output_time to construction time."""
        detector = StallHangDetector(pattern=_DEFAULT_PATTERN, time_func=lambda: _NOW)
        assert detector.last_output_time == _NOW

    def test_severity_from_pattern(self) -> None:
        detector = StallHangDetector(
            pattern=_CRITICAL_PATTERN, time_func=lambda: _NOW
        )
        assert detector.severity == AnomalySeverity.CRITICAL


# ---------------------------------------------------------------------------
# match() -- output recording and stall detection
# ---------------------------------------------------------------------------


class TestMatch:
    """Verify match() behavior: records output and detects prior stalls."""

    def test_match_returns_false_when_no_stall(self) -> None:
        """When output arrives within timeout, match returns False."""
        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=lambda: _NOW)
        assert detector.match("test output") is False

    def test_match_returns_true_when_stall_detected(self) -> None:
        """When output gap exceeds timeout, match returns True."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)
        # Initial output at _NOW
        detector.match("first line")

        # Advance past timeout (10s pattern)
        times[0] = _NOW + timedelta(seconds=11)
        # This line arrives after a stall
        assert detector.match("late line") is True

    def test_match_updates_last_output_time(self) -> None:
        """match() should update the last output timestamp."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)
        detector.match("first line")

        times[0] = _NOW + timedelta(seconds=5)
        detector.match("second line")
        assert detector.last_output_time == _NOW + timedelta(seconds=5)

    def test_match_after_stall_resets_timestamp(self) -> None:
        """After detecting a stall, the timestamp is updated so the next
        match does not re-trigger unless another stall occurs."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)
        detector.match("first line")

        # Stall detected
        times[0] = _NOW + timedelta(seconds=11)
        assert detector.match("after stall") is True

        # Immediate next line should not be a stall
        times[0] = _NOW + timedelta(seconds=12)
        assert detector.match("quick follow-up") is False

    def test_match_exact_boundary_not_stall(self) -> None:
        """Elapsed time exactly at timeout should NOT trigger a stall.
        The condition is strictly greater than timeout."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)
        detector.match("first line")

        # Advance exactly to the timeout boundary
        times[0] = _NOW + timedelta(seconds=10)
        assert detector.match("boundary line") is False

    def test_match_just_over_boundary_is_stall(self) -> None:
        """Elapsed time just over timeout triggers a stall."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)
        detector.match("first line")

        # Just over the threshold
        times[0] = _NOW + timedelta(seconds=10.001)
        assert detector.match("late line") is True

    def test_match_empty_line_still_records(self) -> None:
        """Even empty lines count as output (process is producing output)."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)
        detector.match("first line")

        times[0] = _NOW + timedelta(seconds=5)
        detector.match("")
        assert detector.last_output_time == _NOW + timedelta(seconds=5)

    def test_multiple_stall_cycles(self) -> None:
        """Detector can detect multiple stall events in a single session."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)
        detector.match("line 1")

        # First stall
        times[0] = _NOW + timedelta(seconds=15)
        assert detector.match("after first stall") is True

        # Normal output
        times[0] = _NOW + timedelta(seconds=16)
        assert detector.match("normal line") is False

        # Second stall
        times[0] = _NOW + timedelta(seconds=30)
        assert detector.match("after second stall") is True


# ---------------------------------------------------------------------------
# check_stall() -- periodic polling
# ---------------------------------------------------------------------------


class TestCheckStall:
    """Verify the check_stall() method for periodic stall detection."""

    def test_check_stall_returns_false_initially(self) -> None:
        """Immediately after construction, there is no stall."""
        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=lambda: _NOW)
        assert detector.check_stall() is False

    def test_check_stall_returns_true_after_timeout(self) -> None:
        """After timeout has elapsed with no output, check_stall returns True."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)

        # Advance past timeout
        times[0] = _NOW + timedelta(seconds=11)
        assert detector.check_stall() is True

    def test_check_stall_returns_false_within_timeout(self) -> None:
        """Within the timeout window, check_stall returns False."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)

        times[0] = _NOW + timedelta(seconds=9)
        assert detector.check_stall() is False

    def test_check_stall_exact_boundary(self) -> None:
        """At exactly the timeout boundary, check_stall returns False."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)

        times[0] = _NOW + timedelta(seconds=10)
        assert detector.check_stall() is False

    def test_check_stall_after_output_resets(self) -> None:
        """After receiving output, check_stall timer resets."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)

        # Record output at _NOW + 5s
        times[0] = _NOW + timedelta(seconds=5)
        detector.match("some output")

        # Check at _NOW + 12s: 7s since last output, within 10s timeout
        times[0] = _NOW + timedelta(seconds=12)
        assert detector.check_stall() is False

        # Check at _NOW + 16s: 11s since last output, past 10s timeout
        times[0] = _NOW + timedelta(seconds=16)
        assert detector.check_stall() is True

    def test_check_stall_does_not_update_timestamp(self) -> None:
        """check_stall() is read-only; it does not update last_output_time."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)
        initial_time = detector.last_output_time

        times[0] = _NOW + timedelta(seconds=15)
        detector.check_stall()
        assert detector.last_output_time == initial_time

    def test_elapsed_seconds_property(self) -> None:
        """elapsed_seconds reports time since last output."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)

        times[0] = _NOW + timedelta(seconds=7)
        assert abs(detector.elapsed_seconds - 7.0) < 0.01


# ---------------------------------------------------------------------------
# record_output() -- explicit output recording
# ---------------------------------------------------------------------------


class TestRecordOutput:
    """Verify explicit output recording via record_output()."""

    def test_record_output_updates_timestamp(self) -> None:
        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=lambda: _NOW)
        new_time = _NOW + timedelta(seconds=5)
        detector.record_output(timestamp=new_time)
        assert detector.last_output_time == new_time

    def test_record_output_without_timestamp_uses_time_func(self) -> None:
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)

        times[0] = _NOW + timedelta(seconds=42)
        detector.record_output()
        assert detector.last_output_time == _NOW + timedelta(seconds=42)

    def test_record_output_prevents_stall(self) -> None:
        """Recording output resets the stall timer."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)

        # Advance close to timeout and record output
        times[0] = _NOW + timedelta(seconds=9)
        detector.record_output()

        # Check slightly after: 2s since last output, not stalled
        times[0] = _NOW + timedelta(seconds=11)
        assert detector.check_stall() is False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    """Verify anomaly report generation."""

    def test_report_basic_fields(self) -> None:
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_DEFAULT_PATTERN, time_func=time_func)

        times[0] = _NOW + timedelta(seconds=301)
        report = detector.report(
            "",
            session_id="run-abc",
            detected_at=times[0],
        )
        assert isinstance(report, AnomalyReport)
        assert report.pattern_name == "output_stall"
        assert report.pattern_type is PatternType.STALL_TIMEOUT
        assert report.severity == AnomalySeverity.WARNING
        assert report.session_id == "run-abc"
        assert report.detected_at == times[0]

    def test_report_message_describes_stall(self) -> None:
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_DEFAULT_PATTERN, time_func=time_func)

        times[0] = _NOW + timedelta(seconds=350)
        report = detector.report(
            "",
            session_id="run-abc",
            detected_at=times[0],
        )
        assert "output_stall" in report.message or "stall" in report.message.lower()
        assert "300" in report.message or "350" in report.message

    def test_report_context_contains_metadata(self) -> None:
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)

        times[0] = _NOW + timedelta(seconds=15)
        report = detector.report(
            "",
            session_id="run-abc",
            detected_at=times[0],
        )
        assert report.context is not None
        assert "elapsed_seconds" in report.context
        assert "timeout_seconds" in report.context
        assert "stall_active" in report.context
        assert abs(report.context["elapsed_seconds"] - 15.0) < 0.01
        assert report.context["timeout_seconds"] == 10.0
        assert report.context["stall_active"] is True

    def test_report_context_when_not_stalled(self) -> None:
        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=lambda: _NOW)
        report = detector.report(
            "some output",
            session_id="run-abc",
            detected_at=_NOW,
        )
        assert report.context is not None
        assert report.context["stall_active"] is False

    def test_report_uses_pattern_severity(self) -> None:
        """Report severity comes from the StallTimeoutPattern config."""
        detector = StallHangDetector(
            pattern=_CRITICAL_PATTERN, time_func=lambda: _NOW
        )
        report = detector.report(
            "",
            session_id="run-1",
            detected_at=_NOW,
        )
        assert report.severity == AnomalySeverity.CRITICAL

    def test_report_is_frozen(self) -> None:
        detector = StallHangDetector(pattern=_DEFAULT_PATTERN, time_func=lambda: _NOW)
        report = detector.report(
            "",
            session_id="run-abc",
            detected_at=_NOW,
        )
        with pytest.raises(AttributeError):
            report.message = "changed"  # type: ignore[misc]

    def test_report_message_for_active_stall(self) -> None:
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)

        times[0] = _NOW + timedelta(seconds=25)
        report = detector.report(
            "",
            session_id="run-abc",
            detected_at=times[0],
        )
        # Message should indicate stall is active
        assert "stall" in report.message.lower()

    def test_report_message_for_no_stall(self) -> None:
        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=lambda: _NOW)
        report = detector.report(
            "normal output",
            session_id="run-abc",
            detected_at=_NOW,
        )
        # Message should indicate no stall
        assert report.context is not None
        assert report.context["stall_active"] is False


# ---------------------------------------------------------------------------
# reset() method
# ---------------------------------------------------------------------------


class TestReset:
    """Verify state reset functionality."""

    def test_reset_sets_last_output_to_current_time(self) -> None:
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)

        # Advance time
        times[0] = _NOW + timedelta(seconds=20)
        # Was stalling
        assert detector.check_stall() is True

        # Reset should use current time from time_func
        detector.reset()
        assert detector.last_output_time == _NOW + timedelta(seconds=20)
        assert detector.check_stall() is False

    def test_reset_clears_stall(self) -> None:
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)

        times[0] = _NOW + timedelta(seconds=15)
        assert detector.check_stall() is True

        detector.reset()
        assert detector.check_stall() is False


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Verify thread-safe access to mutable state."""

    def test_concurrent_match_calls(self) -> None:
        """Multiple threads can call match() without data corruption."""
        detector = StallHangDetector(
            pattern=_SHORT_PATTERN, time_func=lambda: _NOW
        )

        barrier = threading.Barrier(4)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=5.0)
                for i in range(50):
                    detector.match(f"line {i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors

    def test_concurrent_check_and_record(self) -> None:
        """Concurrent check_stall and record_output should not raise."""
        detector = StallHangDetector(
            pattern=_SHORT_PATTERN, time_func=lambda: _NOW
        )

        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def checker() -> None:
            try:
                barrier.wait(timeout=5.0)
                for _ in range(100):
                    detector.check_stall()
            except Exception as exc:
                errors.append(exc)

        def recorder() -> None:
            try:
                barrier.wait(timeout=5.0)
                for _ in range(100):
                    detector.record_output()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=checker)
        t2 = threading.Thread(target=recorder)
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert not errors

    def test_concurrent_match_and_reset(self) -> None:
        """Concurrent match and reset should not raise."""
        detector = StallHangDetector(
            pattern=_SHORT_PATTERN, time_func=lambda: _NOW
        )

        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def matcher() -> None:
            try:
                barrier.wait(timeout=5.0)
                for i in range(100):
                    detector.match(f"line {i}")
            except Exception as exc:
                errors.append(exc)

        def resetter() -> None:
            try:
                barrier.wait(timeout=5.0)
                for _ in range(100):
                    detector.reset()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=matcher)
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

    def test_very_long_timeout(self) -> None:
        """A very long timeout should not cause issues."""
        pattern = StallTimeoutPattern(
            name="long_timeout",
            timeout_seconds=86400.0,  # 24 hours
        )
        detector = StallHangDetector(pattern=pattern, time_func=lambda: _NOW)
        assert detector.check_stall() is False
        assert detector.timeout_seconds == 86400.0

    def test_very_short_timeout(self) -> None:
        """A very short (but positive) timeout should work correctly."""
        pattern = StallTimeoutPattern(
            name="micro_timeout",
            timeout_seconds=0.001,  # 1ms
        )
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=pattern, time_func=time_func)

        times[0] = _NOW + timedelta(seconds=0.002)
        assert detector.check_stall() is True

    def test_very_long_output_line(self) -> None:
        detector = StallHangDetector(
            pattern=_SHORT_PATTERN, time_func=lambda: _NOW
        )
        long_line = "x" * 100_000
        assert detector.match(long_line) is False

    def test_unicode_in_output(self) -> None:
        detector = StallHangDetector(
            pattern=_SHORT_PATTERN, time_func=lambda: _NOW
        )
        assert detector.match("sortie: fichier non trouve") is False

    def test_repr_is_readable(self) -> None:
        detector = StallHangDetector(
            pattern=_DEFAULT_PATTERN, time_func=lambda: _NOW
        )
        r = repr(detector)
        assert "StallHangDetector" in r
        assert "output_stall" in r or "stall_hang_detector" in r

    def test_no_output_at_all(self) -> None:
        """When no output is ever received, stall detected after timeout."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)
        # Never call match() -- just check periodically
        times[0] = _NOW + timedelta(seconds=11)
        assert detector.check_stall() is True

    def test_rapid_output_never_stalls(self) -> None:
        """Continuous output within timeout never triggers a stall."""
        times = [_NOW]

        def time_func() -> datetime:
            return times[0]

        detector = StallHangDetector(pattern=_SHORT_PATTERN, time_func=time_func)

        for i in range(100):
            times[0] = _NOW + timedelta(seconds=i * 0.1)
            assert detector.match(f"line {i}") is False
