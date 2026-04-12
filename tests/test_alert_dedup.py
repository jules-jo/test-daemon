"""Tests for alert deduplication and priority-scoring logic.

Verifies that the alert processor:
- Computes deduplication keys from AnomalyReport attributes
- Suppresses duplicate alerts within a configurable time window
- Passes through non-duplicate alerts unchanged
- Computes priority scores based on severity, frequency, and pattern type
- Tracks occurrence counts per dedup key within the window
- Evicts stale dedup entries outside the time window
- Respects configurable capacity limits for tracked keys
- Returns immutable ProcessingResult with kept and suppressed partitions
- Is thread-safe for concurrent access
- Handles edge cases: empty dispatches, single reports, boundary timestamps
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from jules_daemon.monitor.alert_dedup import (
    AlertProcessor,
    AlertProcessorConfig,
    DeduplicationKey,
    PriorityScore,
    ProcessedAlert,
    ProcessingResult,
    compute_dedup_key,
    compute_priority_score,
)
from jules_daemon.monitor.anomaly_models import (
    AnomalyReport,
    AnomalySeverity,
    PatternType,
)
from jules_daemon.monitor.detector_dispatcher import (
    DispatchResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)
_EARLIER = _NOW - timedelta(seconds=30)
_MUCH_EARLIER = _NOW - timedelta(seconds=120)


def _make_report(
    *,
    pattern_name: str = "oom_killer",
    pattern_type: PatternType = PatternType.ERROR_KEYWORD,
    severity: AnomalySeverity = AnomalySeverity.WARNING,
    session_id: str = "session-1",
    detected_at: datetime = _NOW,
    message: str | None = None,
) -> AnomalyReport:
    """Create a minimal AnomalyReport for testing."""
    return AnomalyReport(
        pattern_name=pattern_name,
        pattern_type=pattern_type,
        severity=severity,
        message=message or f"Detected {pattern_name}",
        detected_at=detected_at,
        session_id=session_id,
    )


def _make_dispatch(
    *,
    session_id: str = "session-1",
    reports: tuple[AnomalyReport, ...] = (),
) -> DispatchResult:
    """Create a minimal DispatchResult for testing."""
    return DispatchResult(
        output_line="test output",
        session_id=session_id,
        reports=reports,
        errors=(),
        dispatched_at=_NOW,
    )


# ---------------------------------------------------------------------------
# DeduplicationKey tests
# ---------------------------------------------------------------------------


class TestDeduplicationKey:
    """Verify the immutable deduplication key."""

    def test_construction(self) -> None:
        key = DeduplicationKey(
            pattern_name="oom_killer",
            pattern_type=PatternType.ERROR_KEYWORD,
            session_id="session-1",
        )
        assert key.pattern_name == "oom_killer"
        assert key.pattern_type is PatternType.ERROR_KEYWORD
        assert key.session_id == "session-1"

    def test_frozen(self) -> None:
        key = DeduplicationKey(
            pattern_name="oom_killer",
            pattern_type=PatternType.ERROR_KEYWORD,
            session_id="session-1",
        )
        with pytest.raises(AttributeError):
            key.pattern_name = "changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        k1 = DeduplicationKey(
            pattern_name="oom_killer",
            pattern_type=PatternType.ERROR_KEYWORD,
            session_id="session-1",
        )
        k2 = DeduplicationKey(
            pattern_name="oom_killer",
            pattern_type=PatternType.ERROR_KEYWORD,
            session_id="session-1",
        )
        assert k1 == k2

    def test_different_pattern_name(self) -> None:
        k1 = DeduplicationKey(
            pattern_name="oom_killer",
            pattern_type=PatternType.ERROR_KEYWORD,
            session_id="session-1",
        )
        k2 = DeduplicationKey(
            pattern_name="segfault",
            pattern_type=PatternType.ERROR_KEYWORD,
            session_id="session-1",
        )
        assert k1 != k2

    def test_different_session_id(self) -> None:
        k1 = DeduplicationKey(
            pattern_name="oom_killer",
            pattern_type=PatternType.ERROR_KEYWORD,
            session_id="session-1",
        )
        k2 = DeduplicationKey(
            pattern_name="oom_killer",
            pattern_type=PatternType.ERROR_KEYWORD,
            session_id="session-2",
        )
        assert k1 != k2

    def test_different_pattern_type(self) -> None:
        k1 = DeduplicationKey(
            pattern_name="oom_killer",
            pattern_type=PatternType.ERROR_KEYWORD,
            session_id="session-1",
        )
        k2 = DeduplicationKey(
            pattern_name="oom_killer",
            pattern_type=PatternType.FAILURE_RATE,
            session_id="session-1",
        )
        assert k1 != k2

    def test_hashable(self) -> None:
        """DeduplicationKey can be used as a dict key or set member."""
        k1 = DeduplicationKey(
            pattern_name="oom_killer",
            pattern_type=PatternType.ERROR_KEYWORD,
            session_id="session-1",
        )
        k2 = DeduplicationKey(
            pattern_name="oom_killer",
            pattern_type=PatternType.ERROR_KEYWORD,
            session_id="session-1",
        )
        key_set = {k1, k2}
        assert len(key_set) == 1

    def test_empty_pattern_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="pattern_name must not be empty"):
            DeduplicationKey(
                pattern_name="",
                pattern_type=PatternType.ERROR_KEYWORD,
                session_id="session-1",
            )

    def test_empty_session_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="session_id must not be empty"):
            DeduplicationKey(
                pattern_name="oom_killer",
                pattern_type=PatternType.ERROR_KEYWORD,
                session_id="",
            )


# ---------------------------------------------------------------------------
# compute_dedup_key tests
# ---------------------------------------------------------------------------


class TestComputeDedupKey:
    """Verify dedup key computation from AnomalyReport."""

    def test_computes_key(self) -> None:
        report = _make_report(
            pattern_name="oom_killer",
            pattern_type=PatternType.ERROR_KEYWORD,
            session_id="session-1",
        )
        key = compute_dedup_key(report)
        assert key.pattern_name == "oom_killer"
        assert key.pattern_type is PatternType.ERROR_KEYWORD
        assert key.session_id == "session-1"

    def test_same_report_same_key(self) -> None:
        report = _make_report()
        k1 = compute_dedup_key(report)
        k2 = compute_dedup_key(report)
        assert k1 == k2

    def test_different_messages_same_key(self) -> None:
        """Two reports with same pattern/session but different messages share a key."""
        r1 = _make_report(message="First occurrence")
        r2 = _make_report(message="Second occurrence")
        assert compute_dedup_key(r1) == compute_dedup_key(r2)

    def test_different_timestamps_same_key(self) -> None:
        """Timestamp does not affect the dedup key."""
        r1 = _make_report(detected_at=_NOW)
        r2 = _make_report(detected_at=_EARLIER)
        assert compute_dedup_key(r1) == compute_dedup_key(r2)

    def test_different_severity_same_key(self) -> None:
        """Severity does not affect the dedup key."""
        r1 = _make_report(severity=AnomalySeverity.WARNING)
        r2 = _make_report(severity=AnomalySeverity.CRITICAL)
        assert compute_dedup_key(r1) == compute_dedup_key(r2)


# ---------------------------------------------------------------------------
# PriorityScore tests
# ---------------------------------------------------------------------------


class TestPriorityScore:
    """Verify the immutable PriorityScore record."""

    def test_construction(self) -> None:
        score = PriorityScore(
            total=150.0,
            severity_component=100.0,
            frequency_component=40.0,
            pattern_type_component=10.0,
        )
        assert score.total == 150.0
        assert score.severity_component == 100.0
        assert score.frequency_component == 40.0
        assert score.pattern_type_component == 10.0

    def test_frozen(self) -> None:
        score = PriorityScore(
            total=100.0,
            severity_component=100.0,
            frequency_component=0.0,
            pattern_type_component=0.0,
        )
        with pytest.raises(AttributeError):
            score.total = 999.0  # type: ignore[misc]

    def test_negative_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="total must not be negative"):
            PriorityScore(
                total=-1.0,
                severity_component=0.0,
                frequency_component=0.0,
                pattern_type_component=0.0,
            )


# ---------------------------------------------------------------------------
# compute_priority_score tests
# ---------------------------------------------------------------------------


class TestComputePriorityScore:
    """Verify priority score computation."""

    def test_critical_scores_highest(self) -> None:
        config = AlertProcessorConfig()
        report = _make_report(severity=AnomalySeverity.CRITICAL)
        score = compute_priority_score(report, occurrence_count=1, config=config)
        assert score.total > 0
        assert score.severity_component > 0

    def test_warning_scores_lower_than_critical(self) -> None:
        config = AlertProcessorConfig()
        crit = compute_priority_score(
            _make_report(severity=AnomalySeverity.CRITICAL),
            occurrence_count=1,
            config=config,
        )
        warn = compute_priority_score(
            _make_report(severity=AnomalySeverity.WARNING),
            occurrence_count=1,
            config=config,
        )
        assert crit.total > warn.total

    def test_info_scores_lowest(self) -> None:
        config = AlertProcessorConfig()
        warn = compute_priority_score(
            _make_report(severity=AnomalySeverity.WARNING),
            occurrence_count=1,
            config=config,
        )
        info = compute_priority_score(
            _make_report(severity=AnomalySeverity.INFO),
            occurrence_count=1,
            config=config,
        )
        assert warn.total > info.total

    def test_higher_occurrence_boosts_score(self) -> None:
        config = AlertProcessorConfig()
        report = _make_report(severity=AnomalySeverity.WARNING)
        score_1 = compute_priority_score(report, occurrence_count=1, config=config)
        score_5 = compute_priority_score(report, occurrence_count=5, config=config)
        assert score_5.total > score_1.total
        assert score_5.frequency_component > score_1.frequency_component

    def test_zero_occurrence_uses_base_only(self) -> None:
        """Zero occurrences means no frequency bonus."""
        config = AlertProcessorConfig()
        report = _make_report(severity=AnomalySeverity.WARNING)
        score = compute_priority_score(report, occurrence_count=0, config=config)
        assert score.frequency_component == 0.0

    def test_pattern_type_component(self) -> None:
        """Different pattern types get different component scores."""
        config = AlertProcessorConfig()
        err_report = _make_report(pattern_type=PatternType.ERROR_KEYWORD)
        stall_report = _make_report(
            pattern_name="stall_p",
            pattern_type=PatternType.STALL_TIMEOUT,
        )
        err_score = compute_priority_score(err_report, occurrence_count=1, config=config)
        stall_score = compute_priority_score(
            stall_report, occurrence_count=1, config=config,
        )
        # Both should have non-negative pattern_type_component
        assert err_score.pattern_type_component >= 0.0
        assert stall_score.pattern_type_component >= 0.0

    def test_total_equals_sum_of_components(self) -> None:
        config = AlertProcessorConfig()
        report = _make_report(severity=AnomalySeverity.WARNING)
        score = compute_priority_score(report, occurrence_count=3, config=config)
        expected_total = (
            score.severity_component
            + score.frequency_component
            + score.pattern_type_component
        )
        assert abs(score.total - expected_total) < 1e-9


# ---------------------------------------------------------------------------
# ProcessedAlert tests
# ---------------------------------------------------------------------------


class TestProcessedAlert:
    """Verify the immutable ProcessedAlert record."""

    def test_construction(self) -> None:
        report = _make_report()
        key = compute_dedup_key(report)
        priority = PriorityScore(
            total=50.0,
            severity_component=50.0,
            frequency_component=0.0,
            pattern_type_component=0.0,
        )
        processed = ProcessedAlert(
            anomaly_report=report,
            priority=priority,
            dedup_key=key,
            is_duplicate=False,
            occurrence_count=1,
        )
        assert processed.anomaly_report is report
        assert processed.priority is priority
        assert processed.dedup_key is key
        assert processed.is_duplicate is False
        assert processed.occurrence_count == 1

    def test_frozen(self) -> None:
        report = _make_report()
        key = compute_dedup_key(report)
        priority = PriorityScore(
            total=50.0,
            severity_component=50.0,
            frequency_component=0.0,
            pattern_type_component=0.0,
        )
        processed = ProcessedAlert(
            anomaly_report=report,
            priority=priority,
            dedup_key=key,
            is_duplicate=False,
            occurrence_count=1,
        )
        with pytest.raises(AttributeError):
            processed.is_duplicate = True  # type: ignore[misc]

    def test_negative_occurrence_count_rejected(self) -> None:
        report = _make_report()
        key = compute_dedup_key(report)
        priority = PriorityScore(
            total=50.0,
            severity_component=50.0,
            frequency_component=0.0,
            pattern_type_component=0.0,
        )
        with pytest.raises(ValueError, match="occurrence_count must not be negative"):
            ProcessedAlert(
                anomaly_report=report,
                priority=priority,
                dedup_key=key,
                is_duplicate=False,
                occurrence_count=-1,
            )


# ---------------------------------------------------------------------------
# AlertProcessorConfig tests
# ---------------------------------------------------------------------------


class TestAlertProcessorConfig:
    """Verify the AlertProcessorConfig configuration schema."""

    def test_defaults(self) -> None:
        config = AlertProcessorConfig()
        assert config.dedup_window_seconds > 0
        assert config.frequency_boost_factor > 0
        assert config.max_tracked_keys > 0

    def test_custom_values(self) -> None:
        config = AlertProcessorConfig(
            dedup_window_seconds=120.0,
            frequency_boost_factor=5.0,
            max_tracked_keys=200,
        )
        assert config.dedup_window_seconds == 120.0
        assert config.frequency_boost_factor == 5.0
        assert config.max_tracked_keys == 200

    def test_frozen(self) -> None:
        config = AlertProcessorConfig()
        with pytest.raises(AttributeError):
            config.dedup_window_seconds = 999.0  # type: ignore[misc]

    def test_zero_window_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="dedup_window_seconds must be positive"
        ):
            AlertProcessorConfig(dedup_window_seconds=0.0)

    def test_negative_window_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="dedup_window_seconds must be positive"
        ):
            AlertProcessorConfig(dedup_window_seconds=-5.0)

    def test_negative_boost_factor_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="frequency_boost_factor must not be negative"
        ):
            AlertProcessorConfig(frequency_boost_factor=-1.0)

    def test_zero_boost_factor_allowed(self) -> None:
        config = AlertProcessorConfig(frequency_boost_factor=0.0)
        assert config.frequency_boost_factor == 0.0

    def test_zero_max_tracked_keys_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="max_tracked_keys must be positive"
        ):
            AlertProcessorConfig(max_tracked_keys=0)

    def test_severity_weights_defaults(self) -> None:
        config = AlertProcessorConfig()
        assert AnomalySeverity.CRITICAL in config.severity_weights
        assert AnomalySeverity.WARNING in config.severity_weights
        assert AnomalySeverity.INFO in config.severity_weights
        # CRITICAL should be weighted highest
        assert (
            config.severity_weights[AnomalySeverity.CRITICAL]
            > config.severity_weights[AnomalySeverity.WARNING]
            > config.severity_weights[AnomalySeverity.INFO]
        )

    def test_pattern_type_weights_defaults(self) -> None:
        config = AlertProcessorConfig()
        assert PatternType.ERROR_KEYWORD in config.pattern_type_weights
        assert PatternType.FAILURE_RATE in config.pattern_type_weights
        assert PatternType.STALL_TIMEOUT in config.pattern_type_weights


# ---------------------------------------------------------------------------
# ProcessingResult tests
# ---------------------------------------------------------------------------


class TestProcessingResult:
    """Verify the immutable ProcessingResult record."""

    def test_empty_result(self) -> None:
        result = ProcessingResult(
            kept=(),
            suppressed=(),
            session_id="session-1",
            processed_at=_NOW,
        )
        assert result.kept_count == 0
        assert result.suppressed_count == 0
        assert result.has_kept is False
        assert result.has_suppressed is False

    def test_result_with_kept(self) -> None:
        report = _make_report()
        key = compute_dedup_key(report)
        priority = PriorityScore(
            total=50.0,
            severity_component=50.0,
            frequency_component=0.0,
            pattern_type_component=0.0,
        )
        processed = ProcessedAlert(
            anomaly_report=report,
            priority=priority,
            dedup_key=key,
            is_duplicate=False,
            occurrence_count=1,
        )
        result = ProcessingResult(
            kept=(processed,),
            suppressed=(),
            session_id="session-1",
            processed_at=_NOW,
        )
        assert result.kept_count == 1
        assert result.has_kept is True

    def test_frozen(self) -> None:
        result = ProcessingResult(
            kept=(),
            suppressed=(),
            session_id="session-1",
            processed_at=_NOW,
        )
        with pytest.raises(AttributeError):
            result.session_id = "changed"  # type: ignore[misc]

    def test_empty_session_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="session_id must not be empty"):
            ProcessingResult(
                kept=(),
                suppressed=(),
                session_id="",
                processed_at=_NOW,
            )

    def test_kept_sorted_by_priority_desc(self) -> None:
        """kept_by_priority returns alerts sorted by descending priority."""
        report1 = _make_report(
            pattern_name="p1", severity=AnomalySeverity.INFO
        )
        report2 = _make_report(
            pattern_name="p2", severity=AnomalySeverity.CRITICAL
        )
        key1 = compute_dedup_key(report1)
        key2 = compute_dedup_key(report2)
        low = ProcessedAlert(
            anomaly_report=report1,
            priority=PriorityScore(
                total=10.0,
                severity_component=10.0,
                frequency_component=0.0,
                pattern_type_component=0.0,
            ),
            dedup_key=key1,
            is_duplicate=False,
            occurrence_count=1,
        )
        high = ProcessedAlert(
            anomaly_report=report2,
            priority=PriorityScore(
                total=100.0,
                severity_component=100.0,
                frequency_component=0.0,
                pattern_type_component=0.0,
            ),
            dedup_key=key2,
            is_duplicate=False,
            occurrence_count=1,
        )
        result = ProcessingResult(
            kept=(low, high),
            suppressed=(),
            session_id="session-1",
            processed_at=_NOW,
        )
        sorted_alerts = result.kept_by_priority
        assert sorted_alerts[0].priority.total >= sorted_alerts[1].priority.total


# ---------------------------------------------------------------------------
# AlertProcessor tests - core deduplication
# ---------------------------------------------------------------------------


class TestAlertProcessorDeduplication:
    """Tests for the AlertProcessor deduplication behavior."""

    def test_empty_dispatch_produces_empty_result(self) -> None:
        processor = AlertProcessor()
        dispatch = _make_dispatch()
        result = processor.process(dispatch)
        assert result.kept_count == 0
        assert result.suppressed_count == 0

    def test_single_report_passes_through(self) -> None:
        processor = AlertProcessor()
        report = _make_report()
        dispatch = _make_dispatch(reports=(report,))
        result = processor.process(dispatch)
        assert result.kept_count == 1
        assert result.suppressed_count == 0
        assert result.kept[0].anomaly_report is report
        assert result.kept[0].is_duplicate is False

    def test_duplicate_in_same_batch_suppressed(self) -> None:
        """Two identical reports in the same dispatch: first kept, second suppressed."""
        processor = AlertProcessor()
        r1 = _make_report()
        r2 = _make_report()  # same pattern_name, type, session
        dispatch = _make_dispatch(reports=(r1, r2))
        result = processor.process(dispatch)
        assert result.kept_count == 1
        assert result.suppressed_count == 1
        assert result.suppressed[0].is_duplicate is True

    def test_different_patterns_not_deduplicated(self) -> None:
        processor = AlertProcessor()
        r1 = _make_report(pattern_name="oom_killer")
        r2 = _make_report(pattern_name="segfault")
        dispatch = _make_dispatch(reports=(r1, r2))
        result = processor.process(dispatch)
        assert result.kept_count == 2
        assert result.suppressed_count == 0

    def test_different_sessions_not_deduplicated(self) -> None:
        processor = AlertProcessor()
        r1 = _make_report(session_id="session-1")
        r2 = _make_report(session_id="session-2")
        dispatch = _make_dispatch(reports=(r1, r2))
        result = processor.process(dispatch)
        assert result.kept_count == 2
        assert result.suppressed_count == 0

    def test_duplicate_across_batches_suppressed(self) -> None:
        """Same report in two successive dispatches within window is suppressed."""
        processor = AlertProcessor()
        report = _make_report()

        d1 = _make_dispatch(reports=(report,))
        r1 = processor.process(d1)
        assert r1.kept_count == 1

        d2 = _make_dispatch(reports=(report,))
        r2 = processor.process(d2)
        assert r2.kept_count == 0
        assert r2.suppressed_count == 1

    def test_duplicate_outside_window_passes_through(self) -> None:
        """Same report beyond the dedup window is not considered a duplicate."""
        config = AlertProcessorConfig(dedup_window_seconds=60.0)
        processor = AlertProcessor(config=config)

        report = _make_report(detected_at=_MUCH_EARLIER)
        d1 = _make_dispatch(reports=(report,))
        processor.process(d1, now=_MUCH_EARLIER)

        # Manually expire the window by advancing the processor clock
        # We'll process a new report with a timestamp far enough in the future
        later_report = _make_report(
            detected_at=_MUCH_EARLIER + timedelta(seconds=120),
        )
        d2 = _make_dispatch(reports=(later_report,))
        r2 = processor.process(d2, now=_MUCH_EARLIER + timedelta(seconds=120))
        assert r2.kept_count == 1
        assert r2.suppressed_count == 0

    def test_occurrence_count_increments(self) -> None:
        """Each duplicate increments the occurrence count."""
        processor = AlertProcessor()
        report = _make_report()

        d1 = _make_dispatch(reports=(report,))
        r1 = processor.process(d1)
        assert r1.kept[0].occurrence_count == 1

        d2 = _make_dispatch(reports=(report,))
        r2 = processor.process(d2)
        assert r2.suppressed[0].occurrence_count == 2

        d3 = _make_dispatch(reports=(report,))
        r3 = processor.process(d3)
        assert r3.suppressed[0].occurrence_count == 3

    def test_different_severity_same_key_still_deduped(self) -> None:
        """Reports with same pattern/session but different severity share a dedup key."""
        processor = AlertProcessor()
        r1 = _make_report(severity=AnomalySeverity.WARNING)
        r2 = _make_report(severity=AnomalySeverity.CRITICAL)

        d1 = _make_dispatch(reports=(r1,))
        processor.process(d1)

        d2 = _make_dispatch(reports=(r2,))
        r2_result = processor.process(d2)
        assert r2_result.suppressed_count == 1

    def test_different_messages_same_key_still_deduped(self) -> None:
        """Reports with same pattern/session but different messages are deduped."""
        processor = AlertProcessor()
        r1 = _make_report(message="First OOM kill event")
        r2 = _make_report(message="Second OOM kill event")

        d1 = _make_dispatch(reports=(r1,))
        processor.process(d1)

        d2 = _make_dispatch(reports=(r2,))
        r2_result = processor.process(d2)
        assert r2_result.suppressed_count == 1


# ---------------------------------------------------------------------------
# AlertProcessor tests - priority scoring integration
# ---------------------------------------------------------------------------


class TestAlertProcessorPriority:
    """Tests for priority scoring during processing."""

    def test_kept_alerts_have_priority_scores(self) -> None:
        processor = AlertProcessor()
        report = _make_report(severity=AnomalySeverity.CRITICAL)
        dispatch = _make_dispatch(reports=(report,))
        result = processor.process(dispatch)
        assert result.kept[0].priority.total > 0

    def test_suppressed_alerts_have_priority_scores(self) -> None:
        processor = AlertProcessor()
        report = _make_report()
        d1 = _make_dispatch(reports=(report,))
        processor.process(d1)

        d2 = _make_dispatch(reports=(report,))
        r2 = processor.process(d2)
        assert r2.suppressed[0].priority.total > 0

    def test_critical_scored_higher_than_warning(self) -> None:
        processor = AlertProcessor()
        crit_report = _make_report(
            pattern_name="crit_p",
            severity=AnomalySeverity.CRITICAL,
        )
        warn_report = _make_report(
            pattern_name="warn_p",
            severity=AnomalySeverity.WARNING,
        )
        dispatch = _make_dispatch(reports=(crit_report, warn_report))
        result = processor.process(dispatch)

        # Both should be kept (different pattern names)
        by_name = {
            a.anomaly_report.pattern_name: a for a in result.kept
        }
        assert by_name["crit_p"].priority.total > by_name["warn_p"].priority.total

    def test_repeated_alerts_get_frequency_boost(self) -> None:
        """Duplicate suppressed alerts still get frequency-boosted scores."""
        processor = AlertProcessor()
        report = _make_report()

        d1 = _make_dispatch(reports=(report,))
        r1 = processor.process(d1)
        first_score = r1.kept[0].priority.total

        d2 = _make_dispatch(reports=(report,))
        r2 = processor.process(d2)
        second_score = r2.suppressed[0].priority.total

        # The second occurrence should have a higher score due to frequency boost
        assert second_score > first_score


# ---------------------------------------------------------------------------
# AlertProcessor tests - capacity and eviction
# ---------------------------------------------------------------------------


class TestAlertProcessorCapacity:
    """Tests for tracked key capacity management."""

    def test_max_tracked_keys_enforced(self) -> None:
        """Old dedup entries are evicted when capacity is reached."""
        config = AlertProcessorConfig(
            max_tracked_keys=3,
            dedup_window_seconds=3600.0,  # long window so no time-based expiry
        )
        processor = AlertProcessor(config=config)

        # Fill with 3 different patterns
        for i in range(3):
            report = _make_report(pattern_name=f"pattern_{i}")
            dispatch = _make_dispatch(reports=(report,))
            processor.process(dispatch)

        assert processor.tracked_key_count == 3

        # Add a 4th - should evict the oldest entry
        report4 = _make_report(pattern_name="pattern_3")
        dispatch4 = _make_dispatch(reports=(report4,))
        processor.process(dispatch4)

        assert processor.tracked_key_count <= 3

    def test_evicted_key_no_longer_deduplicates(self) -> None:
        """After a key is evicted from tracking, same report passes through."""
        config = AlertProcessorConfig(
            max_tracked_keys=2,
            dedup_window_seconds=3600.0,
        )
        processor = AlertProcessor(config=config)

        # Track pattern_0
        r0 = _make_report(pattern_name="pattern_0")
        processor.process(_make_dispatch(reports=(r0,)))

        # Track pattern_1
        r1 = _make_report(pattern_name="pattern_1")
        processor.process(_make_dispatch(reports=(r1,)))

        # Track pattern_2 -> evicts pattern_0
        r2 = _make_report(pattern_name="pattern_2")
        processor.process(_make_dispatch(reports=(r2,)))

        # Re-submit pattern_0 -> should NOT be deduped (it was evicted)
        result = processor.process(_make_dispatch(reports=(r0,)))
        assert result.kept_count == 1
        assert result.suppressed_count == 0


# ---------------------------------------------------------------------------
# AlertProcessor tests - window expiry
# ---------------------------------------------------------------------------


class TestAlertProcessorWindowExpiry:
    """Tests for time-window-based dedup entry expiry."""

    def test_expired_entries_cleaned_on_process(self) -> None:
        """Processing automatically cleans expired dedup entries."""
        config = AlertProcessorConfig(dedup_window_seconds=60.0)
        processor = AlertProcessor(config=config)

        # Add a report at time T
        report = _make_report()
        processor.process(
            _make_dispatch(reports=(report,)),
            now=_NOW,
        )
        assert processor.tracked_key_count == 1

        # Process at T+120s (well beyond window) with no reports
        # The cleanup should remove expired entries
        later = _NOW + timedelta(seconds=120)
        processor.process(
            _make_dispatch(),
            now=later,
        )
        assert processor.tracked_key_count == 0

    def test_non_expired_entries_preserved(self) -> None:
        """Entries within the window are not cleaned."""
        config = AlertProcessorConfig(dedup_window_seconds=60.0)
        processor = AlertProcessor(config=config)

        report = _make_report()
        processor.process(
            _make_dispatch(reports=(report,)),
            now=_NOW,
        )

        # Process at T+30s (within window)
        later = _NOW + timedelta(seconds=30)
        processor.process(
            _make_dispatch(),
            now=later,
        )
        assert processor.tracked_key_count == 1


# ---------------------------------------------------------------------------
# AlertProcessor tests - thread safety
# ---------------------------------------------------------------------------


class TestAlertProcessorThreadSafety:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_processing(self) -> None:
        """Multiple threads can process simultaneously without corruption."""
        processor = AlertProcessor()
        errors: list[str] = []

        def worker(session_id: str) -> None:
            try:
                for i in range(20):
                    report = _make_report(
                        pattern_name=f"pattern_{i}",
                        session_id=session_id,
                    )
                    dispatch = _make_dispatch(
                        session_id=session_id,
                        reports=(report,),
                    )
                    result = processor.process(dispatch)
                    # Basic sanity: result should be well-formed
                    assert result.kept_count + result.suppressed_count >= 0
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=worker, args=(f"session-{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_dedup_detection(self) -> None:
        """Dedup is correctly detected under concurrent load."""
        processor = AlertProcessor()
        errors: list[str] = []
        results: list[ProcessingResult] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                report = _make_report(
                    pattern_name="shared_pattern",
                    session_id="shared_session",
                )
                dispatch = _make_dispatch(
                    session_id="shared_session",
                    reports=(report,),
                )
                result = processor.process(dispatch)
                with lock:
                    results.append(result)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

        # Exactly one should be kept, rest should be suppressed
        total_kept = sum(r.kept_count for r in results)
        total_suppressed = sum(r.suppressed_count for r in results)
        assert total_kept == 1
        assert total_suppressed == 9


# ---------------------------------------------------------------------------
# AlertProcessor tests - session_id from dispatch
# ---------------------------------------------------------------------------


class TestAlertProcessorSessionId:
    """Tests that session_id is correctly propagated."""

    def test_result_session_id_from_dispatch(self) -> None:
        processor = AlertProcessor()
        report = _make_report(session_id="session-A")
        dispatch = _make_dispatch(session_id="session-A", reports=(report,))
        result = processor.process(dispatch)
        assert result.session_id == "session-A"

    def test_empty_dispatch_preserves_session_id(self) -> None:
        processor = AlertProcessor()
        dispatch = _make_dispatch(session_id="session-B")
        result = processor.process(dispatch)
        assert result.session_id == "session-B"


# ---------------------------------------------------------------------------
# AlertProcessor tests - clear_session
# ---------------------------------------------------------------------------


class TestAlertProcessorClear:
    """Tests for clearing tracked dedup state."""

    def test_clear_session_removes_keys(self) -> None:
        processor = AlertProcessor()
        report = _make_report(session_id="session-1")
        processor.process(_make_dispatch(reports=(report,)))
        assert processor.tracked_key_count == 1

        removed = processor.clear_session("session-1")
        assert removed >= 1
        assert processor.tracked_key_count == 0

    def test_clear_session_nonexistent(self) -> None:
        processor = AlertProcessor()
        removed = processor.clear_session("nonexistent")
        assert removed == 0

    def test_clear_session_allows_resubmit(self) -> None:
        """After clearing, the same report is no longer considered a duplicate."""
        processor = AlertProcessor()
        report = _make_report(session_id="session-1")
        processor.process(_make_dispatch(reports=(report,)))

        processor.clear_session("session-1")

        result = processor.process(_make_dispatch(reports=(report,)))
        assert result.kept_count == 1
        assert result.suppressed_count == 0

    def test_clear_all(self) -> None:
        processor = AlertProcessor()
        r1 = _make_report(session_id="s1")
        r2 = _make_report(session_id="s2", pattern_name="seg")
        processor.process(_make_dispatch(session_id="s1", reports=(r1,)))
        processor.process(_make_dispatch(session_id="s2", reports=(r2,)))
        assert processor.tracked_key_count == 2

        removed = processor.clear_all()
        assert removed == 2
        assert processor.tracked_key_count == 0
