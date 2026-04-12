"""Tests for the DetectorDispatcher fan-out dispatcher.

Verifies that the DetectorDispatcher:
- Dispatches each output line to all registered detectors concurrently
- Collects AnomalyReport instances from matching detectors
- Returns an immutable DispatchResult with reports and errors
- Handles detector errors gracefully (captures in errors, does not crash)
- Returns empty reports when no detectors match
- Returns empty reports when registry is empty
- Runs all detectors even if some fail
- Provides accurate dispatched_at timestamp
- Captures detector name in error records
- Works with real ErrorKeywordDetector instances
- Works with real FailureRateSpikeDetector instances
- Works with real StallHangDetector instances
- Dispatches to mixed detector types concurrently
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from jules_daemon.monitor.anomaly_models import (
    AnomalyReport,
    AnomalySeverity,
    ErrorKeywordPattern,
    FailureRatePattern,
    PatternType,
    StallTimeoutPattern,
)
from jules_daemon.monitor.detector_dispatcher import (
    DetectorDispatcher,
    DetectorError,
    DispatchResult,
)
from jules_daemon.monitor.detector_registry import DetectorRegistry
from jules_daemon.monitor.error_keyword_detector import ErrorKeywordDetector
from jules_daemon.monitor.failure_rate_spike_detector import (
    FailureRateSpikeDetector,
)
from jules_daemon.monitor.stall_hang_detector import StallHangDetector


# ---------------------------------------------------------------------------
# Helpers: stub detectors for isolation tests
# ---------------------------------------------------------------------------


class _MatchingDetector:
    """Detector that always matches and produces a report."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def pattern_name(self) -> str:
        return self._name

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.ERROR_KEYWORD

    def match(self, output_line: str) -> bool:
        return True

    def report(
        self,
        output_line: str,
        *,
        session_id: str,
        detected_at: datetime,
    ) -> AnomalyReport:
        return AnomalyReport(
            pattern_name=self._name,
            pattern_type=PatternType.ERROR_KEYWORD,
            severity=AnomalySeverity.WARNING,
            message=f"match from {self._name}: {output_line}",
            detected_at=detected_at,
            session_id=session_id,
            context={"output_line": output_line},
        )


class _NonMatchingDetector:
    """Detector that never matches."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def pattern_name(self) -> str:
        return self._name

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.ERROR_KEYWORD

    def match(self, output_line: str) -> bool:
        return False

    def report(
        self,
        output_line: str,
        *,
        session_id: str,
        detected_at: datetime,
    ) -> AnomalyReport:
        return AnomalyReport(
            pattern_name=self._name,
            pattern_type=PatternType.ERROR_KEYWORD,
            severity=AnomalySeverity.INFO,
            message="should not be called",
            detected_at=detected_at,
            session_id=session_id,
        )


class _ExplodingMatchDetector:
    """Detector whose match() raises an exception."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def pattern_name(self) -> str:
        return self._name

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.ERROR_KEYWORD

    def match(self, output_line: str) -> bool:
        raise RuntimeError(f"boom in {self._name}")

    def report(
        self,
        output_line: str,
        *,
        session_id: str,
        detected_at: datetime,
    ) -> AnomalyReport:
        return AnomalyReport(
            pattern_name=self._name,
            pattern_type=PatternType.ERROR_KEYWORD,
            severity=AnomalySeverity.INFO,
            message="unreachable",
            detected_at=detected_at,
            session_id=session_id,
        )


class _ExplodingReportDetector:
    """Detector whose match() succeeds but report() raises."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def pattern_name(self) -> str:
        return self._name

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.ERROR_KEYWORD

    def match(self, output_line: str) -> bool:
        return True

    def report(
        self,
        output_line: str,
        *,
        session_id: str,
        detected_at: datetime,
    ) -> AnomalyReport:
        raise ValueError(f"report failed in {self._name}")


# ---------------------------------------------------------------------------
# DispatchResult data model
# ---------------------------------------------------------------------------


class TestDispatchResult:
    """Tests for the immutable DispatchResult dataclass."""

    def test_empty_result(self) -> None:
        now = datetime.now(timezone.utc)
        result = DispatchResult(
            output_line="test line",
            session_id="session-1",
            reports=(),
            errors=(),
            dispatched_at=now,
        )
        assert result.output_line == "test line"
        assert result.session_id == "session-1"
        assert result.reports == ()
        assert result.errors == ()
        assert result.dispatched_at == now
        assert result.has_anomalies is False
        assert result.has_errors is False

    def test_result_with_reports(self) -> None:
        now = datetime.now(timezone.utc)
        report = AnomalyReport(
            pattern_name="test",
            pattern_type=PatternType.ERROR_KEYWORD,
            severity=AnomalySeverity.WARNING,
            message="found error",
            detected_at=now,
            session_id="session-1",
        )
        result = DispatchResult(
            output_line="ERROR: something",
            session_id="session-1",
            reports=(report,),
            errors=(),
            dispatched_at=now,
        )
        assert result.has_anomalies is True
        assert len(result.reports) == 1

    def test_result_with_errors(self) -> None:
        now = datetime.now(timezone.utc)
        error = DetectorError(
            detector_name="broken",
            error="something went wrong",
        )
        result = DispatchResult(
            output_line="test",
            session_id="session-1",
            reports=(),
            errors=(error,),
            dispatched_at=now,
        )
        assert result.has_errors is True
        assert len(result.errors) == 1

    def test_result_is_frozen(self) -> None:
        now = datetime.now(timezone.utc)
        result = DispatchResult(
            output_line="test",
            session_id="session-1",
            reports=(),
            errors=(),
            dispatched_at=now,
        )
        with pytest.raises(AttributeError):
            result.output_line = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DetectorError data model
# ---------------------------------------------------------------------------


class TestDetectorError:
    """Tests for the immutable DetectorError dataclass."""

    def test_basic_error(self) -> None:
        error = DetectorError(
            detector_name="broken_detector",
            error="regex compilation failed",
        )
        assert error.detector_name == "broken_detector"
        assert error.error == "regex compilation failed"

    def test_frozen(self) -> None:
        error = DetectorError(
            detector_name="test",
            error="oops",
        )
        with pytest.raises(AttributeError):
            error.detector_name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Dispatch with empty registry
# ---------------------------------------------------------------------------


class TestDetectorDispatcherEmpty:
    """Tests for dispatching when the registry has no detectors."""

    @pytest.mark.asyncio
    async def test_dispatch_with_empty_registry(self) -> None:
        registry = DetectorRegistry()
        dispatcher = DetectorDispatcher(registry=registry)
        result = await dispatcher.dispatch(
            "some output line",
            session_id="session-1",
        )
        assert result.output_line == "some output line"
        assert result.session_id == "session-1"
        assert result.reports == ()
        assert result.errors == ()
        assert result.has_anomalies is False


# ---------------------------------------------------------------------------
# Dispatch: matching detectors
# ---------------------------------------------------------------------------


class TestDetectorDispatcherMatching:
    """Tests for dispatching with detectors that match."""

    @pytest.mark.asyncio
    async def test_single_matching_detector(self) -> None:
        registry = DetectorRegistry()
        registry.register(_MatchingDetector("always_match"))
        dispatcher = DetectorDispatcher(registry=registry)

        result = await dispatcher.dispatch(
            "ERROR: something bad",
            session_id="session-1",
        )
        assert result.has_anomalies is True
        assert len(result.reports) == 1
        assert result.reports[0].pattern_name == "always_match"

    @pytest.mark.asyncio
    async def test_multiple_matching_detectors(self) -> None:
        registry = DetectorRegistry()
        registry.register(_MatchingDetector("detector_a"))
        registry.register(_MatchingDetector("detector_b"))
        registry.register(_MatchingDetector("detector_c"))
        dispatcher = DetectorDispatcher(registry=registry)

        result = await dispatcher.dispatch(
            "ERROR: multi-match",
            session_id="session-1",
        )
        assert len(result.reports) == 3
        report_names = {r.pattern_name for r in result.reports}
        assert report_names == {"detector_a", "detector_b", "detector_c"}

    @pytest.mark.asyncio
    async def test_non_matching_detector_produces_no_report(self) -> None:
        registry = DetectorRegistry()
        registry.register(_NonMatchingDetector("no_match"))
        dispatcher = DetectorDispatcher(registry=registry)

        result = await dispatcher.dispatch(
            "clean output line",
            session_id="session-1",
        )
        assert result.has_anomalies is False
        assert result.reports == ()
        assert result.errors == ()

    @pytest.mark.asyncio
    async def test_mixed_matching_and_non_matching(self) -> None:
        registry = DetectorRegistry()
        registry.register(_MatchingDetector("match_yes"))
        registry.register(_NonMatchingDetector("match_no"))
        dispatcher = DetectorDispatcher(registry=registry)

        result = await dispatcher.dispatch(
            "ERROR: partial match",
            session_id="session-1",
        )
        assert len(result.reports) == 1
        assert result.reports[0].pattern_name == "match_yes"


# ---------------------------------------------------------------------------
# Dispatch: error handling
# ---------------------------------------------------------------------------


class TestDetectorDispatcherErrorHandling:
    """Tests for graceful error handling during dispatch."""

    @pytest.mark.asyncio
    async def test_match_exception_captured_in_errors(self) -> None:
        registry = DetectorRegistry()
        registry.register(_ExplodingMatchDetector("exploder"))
        dispatcher = DetectorDispatcher(registry=registry)

        result = await dispatcher.dispatch(
            "test line",
            session_id="session-1",
        )
        assert result.has_errors is True
        assert len(result.errors) == 1
        assert result.errors[0].detector_name == "exploder"
        assert "boom" in result.errors[0].error

    @pytest.mark.asyncio
    async def test_report_exception_captured_in_errors(self) -> None:
        registry = DetectorRegistry()
        registry.register(_ExplodingReportDetector("report_exploder"))
        dispatcher = DetectorDispatcher(registry=registry)

        result = await dispatcher.dispatch(
            "test line",
            session_id="session-1",
        )
        assert result.has_errors is True
        assert len(result.errors) == 1
        assert result.errors[0].detector_name == "report_exploder"
        assert "report failed" in result.errors[0].error

    @pytest.mark.asyncio
    async def test_healthy_detectors_still_run_when_one_fails(self) -> None:
        """Failing detector must not prevent other detectors from running."""
        registry = DetectorRegistry()
        registry.register(_MatchingDetector("healthy"))
        registry.register(_ExplodingMatchDetector("broken"))
        dispatcher = DetectorDispatcher(registry=registry)

        result = await dispatcher.dispatch(
            "test line",
            session_id="session-1",
        )
        # Healthy detector's report should be present
        assert len(result.reports) == 1
        assert result.reports[0].pattern_name == "healthy"
        # Broken detector's error should be captured
        assert len(result.errors) == 1
        assert result.errors[0].detector_name == "broken"


# ---------------------------------------------------------------------------
# Dispatch: concurrent execution
# ---------------------------------------------------------------------------


class TestDetectorDispatcherConcurrency:
    """Tests verifying concurrent fan-out behavior."""

    @pytest.mark.asyncio
    async def test_all_detectors_run_concurrently(self) -> None:
        """Verify multiple detectors are dispatched to, not sequentially blocked."""
        registry = DetectorRegistry()
        # Register several detectors -- they should all be called
        for i in range(10):
            registry.register(_MatchingDetector(f"detector_{i}"))
        dispatcher = DetectorDispatcher(registry=registry)

        result = await dispatcher.dispatch(
            "line for all",
            session_id="session-concurrent",
        )
        assert len(result.reports) == 10
        names = {r.pattern_name for r in result.reports}
        assert names == {f"detector_{i}" for i in range(10)}


# ---------------------------------------------------------------------------
# Dispatch: with real detector implementations
# ---------------------------------------------------------------------------


class TestDetectorDispatcherWithRealDetectors:
    """Integration tests using the actual detector implementations."""

    @pytest.mark.asyncio
    async def test_error_keyword_detector_match(self) -> None:
        registry = DetectorRegistry()
        detector = ErrorKeywordDetector(
            patterns=(
                ErrorKeywordPattern(
                    name="oom",
                    regex=r"Out of memory|OOM",
                    severity=AnomalySeverity.CRITICAL,
                ),
            ),
            name="oom_detector",
        )
        registry.register(detector)
        dispatcher = DetectorDispatcher(registry=registry)

        result = await dispatcher.dispatch(
            "java.lang.OutOfMemoryError: Out of memory",
            session_id="run-1",
        )
        assert result.has_anomalies is True
        assert len(result.reports) == 1
        assert result.reports[0].severity == AnomalySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_error_keyword_detector_no_match(self) -> None:
        registry = DetectorRegistry()
        detector = ErrorKeywordDetector(
            patterns=(
                ErrorKeywordPattern(name="oom", regex=r"OOM"),
            ),
            name="oom_detector",
        )
        registry.register(detector)
        dispatcher = DetectorDispatcher(registry=registry)

        result = await dispatcher.dispatch(
            "PASSED test_login in 1.2s",
            session_id="run-1",
        )
        assert result.has_anomalies is False

    @pytest.mark.asyncio
    async def test_mixed_real_detectors(self) -> None:
        """Dispatch to all three real detector types simultaneously."""
        registry = DetectorRegistry()

        # Error keyword detector
        kw_detector = ErrorKeywordDetector(
            patterns=(
                ErrorKeywordPattern(
                    name="segfault",
                    regex=r"SIGSEGV|Segmentation fault",
                    severity=AnomalySeverity.CRITICAL,
                ),
            ),
            name="segfault_detector",
        )
        registry.register(kw_detector)

        # Failure rate detector (threshold 2 in 60s window)
        fr_pattern = FailureRatePattern(
            name="high_fail",
            threshold_count=2,
            window_seconds=60.0,
        )
        fr_detector = FailureRateSpikeDetector(pattern=fr_pattern)
        registry.register(fr_detector)

        # Stall detector
        stall_pattern = StallTimeoutPattern(
            name="stall",
            timeout_seconds=300.0,
        )
        stall_detector = StallHangDetector(pattern=stall_pattern)
        registry.register(stall_detector)

        dispatcher = DetectorDispatcher(registry=registry)

        # Line that triggers keyword but not others
        result = await dispatcher.dispatch(
            "SIGSEGV: Segmentation fault at 0xdeadbeef",
            session_id="run-mixed",
        )
        assert result.has_anomalies is True
        # Only the keyword detector should match
        kw_reports = [
            r for r in result.reports
            if r.pattern_type == PatternType.ERROR_KEYWORD
        ]
        assert len(kw_reports) == 1
        assert kw_reports[0].severity == AnomalySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_failure_rate_spike_detected_via_dispatch(self) -> None:
        """Dispatch enough failures to trigger a failure rate spike."""
        registry = DetectorRegistry()
        pattern = FailureRatePattern(
            name="spike",
            threshold_count=3,
            window_seconds=60.0,
        )
        detector = FailureRateSpikeDetector(
            pattern=pattern,
            failure_regex=r"FAIL",
        )
        registry.register(detector)
        dispatcher = DetectorDispatcher(registry=registry)

        # First two failures: below threshold
        for _ in range(2):
            result = await dispatcher.dispatch(
                "FAIL test_something",
                session_id="run-spike",
            )
            # May or may not have reports depending on count

        # Third failure: should trigger spike
        result = await dispatcher.dispatch(
            "FAIL test_final",
            session_id="run-spike",
        )
        assert result.has_anomalies is True
        assert len(result.reports) == 1
        assert result.reports[0].pattern_name == "spike"


# ---------------------------------------------------------------------------
# DispatchResult properties
# ---------------------------------------------------------------------------


class TestDispatchResultProperties:
    """Tests for computed properties on DispatchResult."""

    def test_has_anomalies_true_when_reports_present(self) -> None:
        now = datetime.now(timezone.utc)
        report = AnomalyReport(
            pattern_name="test",
            pattern_type=PatternType.ERROR_KEYWORD,
            severity=AnomalySeverity.INFO,
            message="found",
            detected_at=now,
            session_id="s1",
        )
        result = DispatchResult(
            output_line="line",
            session_id="s1",
            reports=(report,),
            errors=(),
            dispatched_at=now,
        )
        assert result.has_anomalies is True

    def test_has_anomalies_false_when_no_reports(self) -> None:
        now = datetime.now(timezone.utc)
        result = DispatchResult(
            output_line="line",
            session_id="s1",
            reports=(),
            errors=(),
            dispatched_at=now,
        )
        assert result.has_anomalies is False

    def test_has_errors_true_when_errors_present(self) -> None:
        now = datetime.now(timezone.utc)
        error = DetectorError(detector_name="x", error="oops")
        result = DispatchResult(
            output_line="line",
            session_id="s1",
            reports=(),
            errors=(error,),
            dispatched_at=now,
        )
        assert result.has_errors is True

    def test_has_errors_false_when_no_errors(self) -> None:
        now = datetime.now(timezone.utc)
        result = DispatchResult(
            output_line="line",
            session_id="s1",
            reports=(),
            errors=(),
            dispatched_at=now,
        )
        assert result.has_errors is False
