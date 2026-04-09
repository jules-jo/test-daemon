"""Tests for the AssembledTestResult structured result dataclass.

Verifies:
- All dataclasses are frozen (immutable)
- Input validation rejects invalid data
- Computed properties derive correct values
- Default factory values are sensible
- Tuples are used for immutable collections
- Serialization-friendly structure (no mutation)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jules_daemon.wiki.assembled_result import (
    AssembledTestResult,
    CompletenessRatio,
    CoverageGap,
    DaemonDowntime,
    GapSeverity,
    InterruptionPoint,
    TestOutcome,
    TestRecord,
)


# -- TestOutcome enum --


class TestTestOutcome:
    """Tests for the TestOutcome enum values."""

    def test_all_outcomes_exist(self) -> None:
        assert TestOutcome.PASSED.value == "passed"
        assert TestOutcome.FAILED.value == "failed"
        assert TestOutcome.SKIPPED.value == "skipped"
        assert TestOutcome.ERROR.value == "error"

    def test_from_string(self) -> None:
        assert TestOutcome("passed") == TestOutcome.PASSED
        assert TestOutcome("error") == TestOutcome.ERROR


# -- GapSeverity enum --


class TestGapSeverity:
    """Tests for the GapSeverity enum values."""

    def test_all_severities_exist(self) -> None:
        assert GapSeverity.LOW.value == "low"
        assert GapSeverity.MEDIUM.value == "medium"
        assert GapSeverity.HIGH.value == "high"
        assert GapSeverity.CRITICAL.value == "critical"

    def test_from_string(self) -> None:
        assert GapSeverity("high") == GapSeverity.HIGH


# -- TestRecord --


class TestTestRecord:
    """Tests for individual test result records."""

    def test_create_minimal(self) -> None:
        record = TestRecord(
            test_name="test_login_flow",
            outcome=TestOutcome.PASSED,
        )
        assert record.test_name == "test_login_flow"
        assert record.outcome == TestOutcome.PASSED
        assert record.duration_seconds is None
        assert record.error_message == ""
        assert record.module == ""
        assert record.line_number is None

    def test_create_full(self) -> None:
        record = TestRecord(
            test_name="test_checkout",
            outcome=TestOutcome.FAILED,
            duration_seconds=1.234,
            error_message="AssertionError: expected 200",
            module="tests/test_api.py",
            line_number=42,
        )
        assert record.test_name == "test_checkout"
        assert record.outcome == TestOutcome.FAILED
        assert record.duration_seconds == 1.234
        assert record.error_message == "AssertionError: expected 200"
        assert record.module == "tests/test_api.py"
        assert record.line_number == 42

    def test_frozen(self) -> None:
        record = TestRecord(
            test_name="test_x",
            outcome=TestOutcome.PASSED,
        )
        with pytest.raises(AttributeError):
            record.test_name = "test_y"  # type: ignore[misc]

    def test_empty_test_name_raises(self) -> None:
        with pytest.raises(ValueError, match="test_name must not be empty"):
            TestRecord(test_name="", outcome=TestOutcome.PASSED)

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration_seconds must not be negative"):
            TestRecord(
                test_name="test_x",
                outcome=TestOutcome.PASSED,
                duration_seconds=-0.5,
            )

    def test_zero_duration_is_valid(self) -> None:
        record = TestRecord(
            test_name="test_instant",
            outcome=TestOutcome.SKIPPED,
            duration_seconds=0.0,
        )
        assert record.duration_seconds == 0.0

    def test_negative_line_number_raises(self) -> None:
        with pytest.raises(ValueError, match="line_number must be >= 1"):
            TestRecord(
                test_name="test_x",
                outcome=TestOutcome.PASSED,
                line_number=-1,
            )

    def test_zero_line_number_raises(self) -> None:
        with pytest.raises(ValueError, match="line_number must be >= 1"):
            TestRecord(
                test_name="test_x",
                outcome=TestOutcome.PASSED,
                line_number=0,
            )

    def test_line_number_one_is_valid(self) -> None:
        record = TestRecord(
            test_name="test_x",
            outcome=TestOutcome.PASSED,
            line_number=1,
        )
        assert record.line_number == 1

    def test_is_failure_property(self) -> None:
        passed = TestRecord(test_name="t", outcome=TestOutcome.PASSED)
        failed = TestRecord(test_name="t", outcome=TestOutcome.FAILED)
        error = TestRecord(test_name="t", outcome=TestOutcome.ERROR)
        skipped = TestRecord(test_name="t", outcome=TestOutcome.SKIPPED)

        assert passed.is_failure is False
        assert failed.is_failure is True
        assert error.is_failure is True
        assert skipped.is_failure is False


# -- CompletenessRatio --


class TestCompletenessRatio:
    """Tests for the completeness ratio calculation."""

    def test_create_with_defaults(self) -> None:
        ratio = CompletenessRatio()
        assert ratio.executed == 0
        assert ratio.expected == 0

    def test_create_with_values(self) -> None:
        ratio = CompletenessRatio(executed=80, expected=100)
        assert ratio.executed == 80
        assert ratio.expected == 100

    def test_frozen(self) -> None:
        ratio = CompletenessRatio(executed=10, expected=20)
        with pytest.raises(AttributeError):
            ratio.executed = 15  # type: ignore[misc]

    def test_ratio_property_normal(self) -> None:
        ratio = CompletenessRatio(executed=75, expected=100)
        assert ratio.ratio == 0.75

    def test_ratio_property_full(self) -> None:
        ratio = CompletenessRatio(executed=100, expected=100)
        assert ratio.ratio == 1.0

    def test_ratio_property_empty(self) -> None:
        ratio = CompletenessRatio(executed=0, expected=100)
        assert ratio.ratio == 0.0

    def test_ratio_property_zero_expected(self) -> None:
        """When expected is zero, ratio should be 0.0 to avoid division by zero."""
        ratio = CompletenessRatio(executed=0, expected=0)
        assert ratio.ratio == 0.0

    def test_negative_executed_raises(self) -> None:
        with pytest.raises(ValueError, match="executed must not be negative"):
            CompletenessRatio(executed=-1, expected=10)

    def test_negative_expected_raises(self) -> None:
        with pytest.raises(ValueError, match="expected must not be negative"):
            CompletenessRatio(executed=0, expected=-5)

    def test_executed_exceeds_expected_raises(self) -> None:
        with pytest.raises(ValueError, match="executed must not exceed expected"):
            CompletenessRatio(executed=15, expected=10)

    def test_is_complete_property(self) -> None:
        complete = CompletenessRatio(executed=10, expected=10)
        incomplete = CompletenessRatio(executed=5, expected=10)
        empty = CompletenessRatio(executed=0, expected=0)

        assert complete.is_complete is True
        assert incomplete.is_complete is False
        assert empty.is_complete is True


# -- CoverageGap --


class TestCoverageGap:
    """Tests for coverage gap metadata."""

    def test_create_minimal(self) -> None:
        gap = CoverageGap(
            module="tests/test_auth.py",
            reason="No test coverage for OAuth refresh flow",
        )
        assert gap.module == "tests/test_auth.py"
        assert gap.reason == "No test coverage for OAuth refresh flow"
        assert gap.severity == GapSeverity.MEDIUM

    def test_create_full(self) -> None:
        gap = CoverageGap(
            module="tests/test_payment.py",
            reason="Payment processing branch not covered",
            severity=GapSeverity.CRITICAL,
            expected_tests=5,
            actual_tests=1,
        )
        assert gap.severity == GapSeverity.CRITICAL
        assert gap.expected_tests == 5
        assert gap.actual_tests == 1

    def test_frozen(self) -> None:
        gap = CoverageGap(module="m", reason="r")
        with pytest.raises(AttributeError):
            gap.module = "other"  # type: ignore[misc]

    def test_empty_module_raises(self) -> None:
        with pytest.raises(ValueError, match="module must not be empty"):
            CoverageGap(module="", reason="missing tests")

    def test_empty_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="reason must not be empty"):
            CoverageGap(module="tests/test_x.py", reason="")

    def test_negative_expected_tests_raises(self) -> None:
        with pytest.raises(ValueError, match="expected_tests must not be negative"):
            CoverageGap(module="m", reason="r", expected_tests=-1)

    def test_negative_actual_tests_raises(self) -> None:
        with pytest.raises(ValueError, match="actual_tests must not be negative"):
            CoverageGap(module="m", reason="r", actual_tests=-1)

    def test_actual_exceeds_expected_raises(self) -> None:
        with pytest.raises(ValueError, match="actual_tests must not exceed expected_tests"):
            CoverageGap(module="m", reason="r", expected_tests=3, actual_tests=5)


# -- InterruptionPoint --


class TestInterruptionPoint:
    """Tests for interruption point metadata."""

    def test_create_not_interrupted(self) -> None:
        point = InterruptionPoint()
        assert point.interrupted is False
        assert point.at_test == ""
        assert point.at_timestamp is None
        assert point.reason == ""
        assert point.exit_code is None

    def test_create_interrupted(self) -> None:
        ts = datetime(2026, 4, 9, 14, 30, 0, tzinfo=timezone.utc)
        point = InterruptionPoint(
            interrupted=True,
            at_test="test_complex_workflow",
            at_timestamp=ts,
            reason="SSH connection dropped",
            exit_code=137,
        )
        assert point.interrupted is True
        assert point.at_test == "test_complex_workflow"
        assert point.at_timestamp == ts
        assert point.reason == "SSH connection dropped"
        assert point.exit_code == 137

    def test_frozen(self) -> None:
        point = InterruptionPoint()
        with pytest.raises(AttributeError):
            point.interrupted = True  # type: ignore[misc]

    def test_interrupted_without_reason_raises(self) -> None:
        """If interrupted is True, reason must be provided."""
        with pytest.raises(ValueError, match="reason must not be empty when interrupted"):
            InterruptionPoint(interrupted=True, reason="")

    def test_naive_timestamp_raises(self) -> None:
        """Naive datetimes are rejected to prevent timezone confusion."""
        naive_ts = datetime(2026, 4, 9, 14, 30, 0)  # no tzinfo
        with pytest.raises(ValueError, match="at_timestamp must be timezone-aware"):
            InterruptionPoint(at_timestamp=naive_ts)


# -- DaemonDowntime --


class TestDaemonDowntime:
    """Tests for daemon downtime metadata used to distinguish crash vs timeout."""

    def test_create_defaults(self) -> None:
        """Default DaemonDowntime indicates no downtime."""
        dt = DaemonDowntime()
        assert dt.daemon_was_down is False
        assert dt.down_started_at is None
        assert dt.down_ended_at is None
        assert dt.estimated_down_seconds is None
        assert dt.recovery_method == ""

    def test_create_with_downtime(self) -> None:
        """DaemonDowntime with full downtime metadata."""
        start = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 9, 12, 0, 25, tzinfo=timezone.utc)
        dt = DaemonDowntime(
            daemon_was_down=True,
            down_started_at=start,
            down_ended_at=end,
            estimated_down_seconds=25.0,
            recovery_method="reconnect",
        )
        assert dt.daemon_was_down is True
        assert dt.down_started_at == start
        assert dt.down_ended_at == end
        assert dt.estimated_down_seconds == 25.0
        assert dt.recovery_method == "reconnect"

    def test_frozen(self) -> None:
        """DaemonDowntime is immutable."""
        dt = DaemonDowntime()
        with pytest.raises(AttributeError):
            dt.daemon_was_down = True  # type: ignore[misc]

    def test_daemon_was_down_true_requires_estimated_seconds(self) -> None:
        """When daemon_was_down is True, estimated_down_seconds must be provided."""
        with pytest.raises(ValueError, match="estimated_down_seconds must be provided"):
            DaemonDowntime(daemon_was_down=True, estimated_down_seconds=None)

    def test_negative_estimated_down_seconds_raises(self) -> None:
        """Negative estimated_down_seconds is invalid."""
        with pytest.raises(ValueError, match="estimated_down_seconds must not be negative"):
            DaemonDowntime(
                daemon_was_down=True,
                estimated_down_seconds=-1.0,
            )

    def test_zero_estimated_down_seconds_valid(self) -> None:
        """Zero estimated_down_seconds is valid (instant restart)."""
        dt = DaemonDowntime(
            daemon_was_down=True,
            estimated_down_seconds=0.0,
        )
        assert dt.estimated_down_seconds == 0.0

    def test_naive_down_started_at_raises(self) -> None:
        """Naive datetime for down_started_at is rejected."""
        naive_ts = datetime(2026, 4, 9, 12, 0, 0)
        with pytest.raises(ValueError, match="down_started_at must be timezone-aware"):
            DaemonDowntime(
                daemon_was_down=True,
                estimated_down_seconds=10.0,
                down_started_at=naive_ts,
            )

    def test_naive_down_ended_at_raises(self) -> None:
        """Naive datetime for down_ended_at is rejected."""
        naive_ts = datetime(2026, 4, 9, 12, 0, 30)
        with pytest.raises(ValueError, match="down_ended_at must be timezone-aware"):
            DaemonDowntime(
                daemon_was_down=True,
                estimated_down_seconds=30.0,
                down_ended_at=naive_ts,
            )

    def test_not_down_with_timestamps_allowed(self) -> None:
        """When daemon_was_down is False, timestamps are ignored (no validation)."""
        dt = DaemonDowntime(daemon_was_down=False)
        assert dt.down_started_at is None
        assert dt.down_ended_at is None

    def test_equality(self) -> None:
        """Two identical DaemonDowntime instances are equal."""
        a = DaemonDowntime(daemon_was_down=True, estimated_down_seconds=10.0)
        b = DaemonDowntime(daemon_was_down=True, estimated_down_seconds=10.0)
        assert a == b

    def test_hashable(self) -> None:
        """DaemonDowntime can be used in sets."""
        dt = DaemonDowntime()
        assert hash(dt) is not None
        s = {dt}
        assert dt in s


# -- AssembledTestResult --


class TestAssembledTestResult:
    """Tests for the top-level structured result."""

    def _make_records(self) -> tuple[TestRecord, ...]:
        return (
            TestRecord(
                test_name="test_login",
                outcome=TestOutcome.PASSED,
                duration_seconds=0.5,
                module="tests/test_auth.py",
            ),
            TestRecord(
                test_name="test_checkout",
                outcome=TestOutcome.FAILED,
                duration_seconds=1.2,
                error_message="AssertionError",
                module="tests/test_cart.py",
            ),
            TestRecord(
                test_name="test_refund",
                outcome=TestOutcome.SKIPPED,
                module="tests/test_payment.py",
            ),
        )

    def _make_gaps(self) -> tuple[CoverageGap, ...]:
        return (
            CoverageGap(
                module="tests/test_payment.py",
                reason="Refund flow not tested",
                severity=GapSeverity.HIGH,
            ),
        )

    def test_create_minimal(self) -> None:
        result = AssembledTestResult(
            run_id="abc-123",
            session_id="session-456",
            host="prod.example.com",
        )
        assert result.run_id == "abc-123"
        assert result.session_id == "session-456"
        assert result.host == "prod.example.com"
        assert result.records == ()
        assert result.completeness == CompletenessRatio()
        assert result.coverage_gaps == ()
        assert result.interruption == InterruptionPoint()
        assert result.daemon_downtime == DaemonDowntime()
        assert result.assembled_at is not None

    def test_create_full(self) -> None:
        ts = datetime(2026, 4, 9, 15, 0, 0, tzinfo=timezone.utc)
        records = self._make_records()
        gaps = self._make_gaps()
        completeness = CompletenessRatio(executed=2, expected=3)
        interruption = InterruptionPoint()

        result = AssembledTestResult(
            run_id="run-789",
            session_id="sess-001",
            host="staging.example.com",
            records=records,
            completeness=completeness,
            coverage_gaps=gaps,
            interruption=interruption,
            assembled_at=ts,
        )

        assert result.run_id == "run-789"
        assert len(result.records) == 3
        assert result.records[0].test_name == "test_login"
        assert result.completeness.ratio == pytest.approx(2 / 3)
        assert len(result.coverage_gaps) == 1
        assert result.coverage_gaps[0].severity == GapSeverity.HIGH
        assert result.assembled_at == ts

    def test_frozen(self) -> None:
        result = AssembledTestResult(
            run_id="x",
            session_id="s",
            host="h",
        )
        with pytest.raises(AttributeError):
            result.run_id = "y"  # type: ignore[misc]

    def test_empty_run_id_raises(self) -> None:
        with pytest.raises(ValueError, match="run_id must not be empty"):
            AssembledTestResult(run_id="", session_id="s", host="h")

    def test_empty_session_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id must not be empty"):
            AssembledTestResult(run_id="r", session_id="", host="h")

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError, match="host must not be empty"):
            AssembledTestResult(run_id="r", session_id="s", host="")

    def test_records_is_tuple(self) -> None:
        """Verify records are stored as a tuple (immutable sequence)."""
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            records=self._make_records(),
        )
        assert isinstance(result.records, tuple)

    def test_coverage_gaps_is_tuple(self) -> None:
        """Verify coverage_gaps are stored as a tuple (immutable sequence)."""
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            coverage_gaps=self._make_gaps(),
        )
        assert isinstance(result.coverage_gaps, tuple)

    # -- Computed properties --

    def test_total_tests(self) -> None:
        records = self._make_records()
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            records=records,
        )
        assert result.total_tests == 3

    def test_passed_count(self) -> None:
        records = self._make_records()
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            records=records,
        )
        assert result.passed_count == 1

    def test_failed_count(self) -> None:
        records = self._make_records()
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            records=records,
        )
        assert result.failed_count == 1

    def test_skipped_count(self) -> None:
        records = self._make_records()
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            records=records,
        )
        assert result.skipped_count == 1

    def test_error_count(self) -> None:
        records = (
            TestRecord(test_name="t1", outcome=TestOutcome.ERROR),
            TestRecord(test_name="t2", outcome=TestOutcome.ERROR),
            TestRecord(test_name="t3", outcome=TestOutcome.PASSED),
        )
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            records=records,
        )
        assert result.error_count == 2

    def test_pass_rate_normal(self) -> None:
        records = self._make_records()  # 1 passed, 1 failed, 1 skipped
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            records=records,
        )
        assert result.pass_rate == pytest.approx(1 / 3)

    def test_pass_rate_no_records(self) -> None:
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.pass_rate == 0.0

    def test_has_failures(self) -> None:
        records = self._make_records()
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            records=records,
        )
        assert result.has_failures is True

    def test_has_failures_all_passed(self) -> None:
        records = (
            TestRecord(test_name="t1", outcome=TestOutcome.PASSED),
            TestRecord(test_name="t2", outcome=TestOutcome.PASSED),
        )
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            records=records,
        )
        assert result.has_failures is False

    def test_was_interrupted(self) -> None:
        result_not_interrupted = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result_not_interrupted.was_interrupted is False

        result_interrupted = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            interruption=InterruptionPoint(
                interrupted=True,
                reason="SSH dropped",
            ),
        )
        assert result_interrupted.was_interrupted is True

    def test_failed_records(self) -> None:
        records = self._make_records()
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            records=records,
        )
        failures = result.failed_records
        assert isinstance(failures, tuple)
        assert len(failures) == 1
        assert failures[0].test_name == "test_checkout"

    def test_total_duration(self) -> None:
        records = (
            TestRecord(
                test_name="t1",
                outcome=TestOutcome.PASSED,
                duration_seconds=1.5,
            ),
            TestRecord(
                test_name="t2",
                outcome=TestOutcome.PASSED,
                duration_seconds=2.5,
            ),
            TestRecord(
                test_name="t3",
                outcome=TestOutcome.SKIPPED,
                duration_seconds=None,
            ),
        )
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            records=records,
        )
        assert result.total_duration_seconds == pytest.approx(4.0)

    def test_total_duration_all_none(self) -> None:
        records = (
            TestRecord(test_name="t1", outcome=TestOutcome.SKIPPED),
            TestRecord(test_name="t2", outcome=TestOutcome.SKIPPED),
        )
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            records=records,
        )
        assert result.total_duration_seconds == 0.0

    def test_total_duration_empty(self) -> None:
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.total_duration_seconds == 0.0

    def test_daemon_was_down_property_false(self) -> None:
        """daemon_was_down is False by default."""
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.daemon_was_down is False

    def test_daemon_was_down_property_true(self) -> None:
        """daemon_was_down returns True when daemon_downtime says so."""
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            daemon_downtime=DaemonDowntime(
                daemon_was_down=True,
                estimated_down_seconds=15.0,
                recovery_method="reconnect",
            ),
        )
        assert result.daemon_was_down is True

    def test_daemon_downtime_field(self) -> None:
        """daemon_downtime field carries full downtime metadata."""
        start = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 9, 12, 0, 25, tzinfo=timezone.utc)
        downtime = DaemonDowntime(
            daemon_was_down=True,
            down_started_at=start,
            down_ended_at=end,
            estimated_down_seconds=25.0,
            recovery_method="reconnect",
        )
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            daemon_downtime=downtime,
        )
        assert result.daemon_downtime.daemon_was_down is True
        assert result.daemon_downtime.down_started_at == start
        assert result.daemon_downtime.down_ended_at == end
        assert result.daemon_downtime.estimated_down_seconds == 25.0
        assert result.daemon_downtime.recovery_method == "reconnect"

    def test_distinguish_crash_from_timeout(self) -> None:
        """Downstream consumers can distinguish partial-due-to-crash
        from partial-due-to-timeout using daemon_was_down + interruption."""
        # Partial due to crash (daemon was down)
        crash_result = AssembledTestResult(
            run_id="crash-1",
            session_id="s",
            host="h",
            interruption=InterruptionPoint(
                interrupted=True,
                reason="Daemon crashed during execution",
            ),
            daemon_downtime=DaemonDowntime(
                daemon_was_down=True,
                estimated_down_seconds=20.0,
                recovery_method="reconnect",
            ),
        )
        # Partial due to timeout (daemon was up the whole time)
        timeout_result = AssembledTestResult(
            run_id="timeout-1",
            session_id="s",
            host="h",
            interruption=InterruptionPoint(
                interrupted=True,
                reason="Test execution timed out after 3600s",
            ),
            daemon_downtime=DaemonDowntime(daemon_was_down=False),
        )

        # Both are interrupted
        assert crash_result.was_interrupted is True
        assert timeout_result.was_interrupted is True

        # But only one had daemon downtime
        assert crash_result.daemon_was_down is True
        assert timeout_result.daemon_was_down is False


# -- Equality and hashing --


class TestEquality:
    """Frozen dataclasses support equality and hashing by default."""

    def test_equal_records(self) -> None:
        r1 = TestRecord(test_name="t", outcome=TestOutcome.PASSED)
        r2 = TestRecord(test_name="t", outcome=TestOutcome.PASSED)
        assert r1 == r2

    def test_unequal_records(self) -> None:
        r1 = TestRecord(test_name="t1", outcome=TestOutcome.PASSED)
        r2 = TestRecord(test_name="t2", outcome=TestOutcome.PASSED)
        assert r1 != r2

    def test_hashable_record(self) -> None:
        r = TestRecord(test_name="t", outcome=TestOutcome.PASSED)
        assert hash(r) is not None
        # Can be used in sets
        s = {r}
        assert r in s

    def test_hashable_result(self) -> None:
        ts = datetime(2026, 4, 9, 15, 0, 0, tzinfo=timezone.utc)
        result = AssembledTestResult(
            run_id="r",
            session_id="s",
            host="h",
            assembled_at=ts,
        )
        assert hash(result) is not None
        # Can be used in sets
        s = {result}
        assert result in s
