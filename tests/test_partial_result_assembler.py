"""Tests for the partial result assembler function.

Verifies that the assembler:
- Accepts a sequence of parsed partial test records (ParseResult)
- Merges them by deduplication (same test_name + module -> last wins)
- Maintains ordering (by module, then by original appearance)
- Computes completeness ratio (executed vs expected)
- Identifies coverage gaps (modules with fewer tests than expected)
- Records the interruption point when output was truncated
- Returns an immutable AssembledTestResult
- Handles empty input gracefully
- Handles multiple ParseResults from different output chunks
- Maps TestStatus -> TestOutcome correctly
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jules_daemon.monitor.test_output_parser import (
    FrameworkHint,
    ParseResult,
    TestRecord as ParsedTestRecord,
    TestStatus,
)
from jules_daemon.wiki.assembled_result import (
    AssembledTestResult,
    CompletenessRatio,
    CoverageGap,
    GapSeverity,
    InterruptionPoint,
    TestOutcome,
    TestRecord as AssembledTestRecord,
)
from jules_daemon.wiki.partial_result_assembler import (
    assemble_partial_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parse_result(
    records: tuple[ParsedTestRecord, ...] = (),
    truncated: bool = False,
    framework_hint: FrameworkHint = FrameworkHint.PYTEST,
    total_lines_parsed: int = 10,
    raw_tail: str = "",
) -> ParseResult:
    """Create a ParseResult for testing."""
    return ParseResult(
        records=records,
        truncated=truncated,
        framework_hint=framework_hint,
        total_lines_parsed=total_lines_parsed,
        raw_tail=raw_tail,
    )


def _make_parsed_record(
    name: str = "test_example",
    status: TestStatus = TestStatus.PASSED,
    module: str = "tests/test_example.py",
    duration_seconds: float | None = None,
    line_number: int | None = None,
) -> ParsedTestRecord:
    """Create a ParsedTestRecord for testing."""
    return ParsedTestRecord(
        name=name,
        status=status,
        module=module,
        duration_seconds=duration_seconds,
        line_number=line_number,
    )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Assembler handles empty or no-record input gracefully."""

    def test_empty_sequence_returns_empty_result(self) -> None:
        result = assemble_partial_results(
            partials=[],
            run_id="run-1",
            session_id="sess-1",
            host="example.com",
        )
        assert isinstance(result, AssembledTestResult)
        assert result.records == ()
        assert result.total_tests == 0

    def test_single_empty_parse_result(self) -> None:
        empty = _make_parse_result()
        result = assemble_partial_results(
            partials=[empty],
            run_id="run-1",
            session_id="sess-1",
            host="example.com",
        )
        assert result.records == ()
        assert result.total_tests == 0

    def test_preserves_run_metadata(self) -> None:
        result = assemble_partial_results(
            partials=[],
            run_id="abc-123",
            session_id="sess-456",
            host="staging.example.com",
        )
        assert result.run_id == "abc-123"
        assert result.session_id == "sess-456"
        assert result.host == "staging.example.com"

    def test_empty_completeness_with_expected(self) -> None:
        result = assemble_partial_results(
            partials=[],
            run_id="r",
            session_id="s",
            host="h",
            expected_test_count=10,
        )
        assert result.completeness.executed == 0
        assert result.completeness.expected == 10
        assert result.completeness.ratio == 0.0

    def test_empty_completeness_without_expected(self) -> None:
        result = assemble_partial_results(
            partials=[],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.completeness.executed == 0
        assert result.completeness.expected == 0


# ---------------------------------------------------------------------------
# Single ParseResult with records
# ---------------------------------------------------------------------------


class TestSingleParseResult:
    """Assembler processes a single ParseResult with test records."""

    def test_converts_passed_records(self) -> None:
        records = (
            _make_parsed_record("test_login", TestStatus.PASSED, "tests/test_auth.py"),
            _make_parsed_record("test_logout", TestStatus.PASSED, "tests/test_auth.py"),
        )
        pr = _make_parse_result(records=records)

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.total_tests == 2
        assert result.passed_count == 2

    def test_converts_failed_records(self) -> None:
        records = (
            _make_parsed_record("test_bad", TestStatus.FAILED, "tests/test_x.py"),
        )
        pr = _make_parse_result(records=records)

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.failed_count == 1
        assert result.records[0].outcome == TestOutcome.FAILED

    def test_converts_error_records(self) -> None:
        records = (
            _make_parsed_record("test_err", TestStatus.ERROR, "tests/test_x.py"),
        )
        pr = _make_parse_result(records=records)

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.error_count == 1
        assert result.records[0].outcome == TestOutcome.ERROR

    def test_converts_skipped_records(self) -> None:
        records = (
            _make_parsed_record("test_skip", TestStatus.SKIPPED, "tests/test_x.py"),
        )
        pr = _make_parse_result(records=records)

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.skipped_count == 1
        assert result.records[0].outcome == TestOutcome.SKIPPED

    def test_incomplete_maps_to_error(self) -> None:
        """INCOMPLETE tests have no direct outcome -- map to ERROR."""
        records = (
            _make_parsed_record("test_inc", TestStatus.INCOMPLETE, "tests/test_x.py"),
        )
        pr = _make_parse_result(records=records, truncated=True)

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.records[0].outcome == TestOutcome.ERROR
        assert "incomplete" in result.records[0].error_message.lower()

    def test_preserves_module(self) -> None:
        records = (
            _make_parsed_record("test_a", TestStatus.PASSED, "tests/test_auth.py"),
        )
        pr = _make_parse_result(records=records)

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.records[0].module == "tests/test_auth.py"

    def test_preserves_duration(self) -> None:
        records = (
            _make_parsed_record(
                "test_a", TestStatus.PASSED, "tests/test_x.py",
                duration_seconds=1.5,
            ),
        )
        pr = _make_parse_result(records=records)

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.records[0].duration_seconds == 1.5


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """When the same test appears in multiple partials, last occurrence wins."""

    def test_duplicate_test_name_and_module_deduped(self) -> None:
        """Second parse result's record overrides first for same test+module."""
        pr1 = _make_parse_result(records=(
            _make_parsed_record("test_login", TestStatus.PASSED, "tests/test_auth.py"),
        ))
        pr2 = _make_parse_result(records=(
            _make_parsed_record("test_login", TestStatus.FAILED, "tests/test_auth.py"),
        ))

        result = assemble_partial_results(
            partials=[pr1, pr2],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.total_tests == 1
        assert result.records[0].outcome == TestOutcome.FAILED

    def test_same_name_different_module_not_deduped(self) -> None:
        """Tests with the same name but different modules are distinct."""
        pr = _make_parse_result(records=(
            _make_parsed_record("test_create", TestStatus.PASSED, "tests/test_user.py"),
            _make_parsed_record("test_create", TestStatus.FAILED, "tests/test_order.py"),
        ))

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.total_tests == 2

    def test_dedup_across_multiple_partials(self) -> None:
        """Deduplication works across any number of ParseResults."""
        pr1 = _make_parse_result(records=(
            _make_parsed_record("test_a", TestStatus.PASSED, "m1.py"),
            _make_parsed_record("test_b", TestStatus.PASSED, "m1.py"),
        ))
        pr2 = _make_parse_result(records=(
            _make_parsed_record("test_a", TestStatus.FAILED, "m1.py"),
            _make_parsed_record("test_c", TestStatus.PASSED, "m1.py"),
        ))
        pr3 = _make_parse_result(records=(
            _make_parsed_record("test_a", TestStatus.ERROR, "m1.py"),
        ))

        result = assemble_partial_results(
            partials=[pr1, pr2, pr3],
            run_id="r",
            session_id="s",
            host="h",
        )
        # test_a: last wins (ERROR), test_b: PASSED, test_c: PASSED
        assert result.total_tests == 3
        test_a = next(r for r in result.records if r.test_name == "test_a")
        assert test_a.outcome == TestOutcome.ERROR

    def test_incomplete_overridden_by_terminal(self) -> None:
        """If a test was INCOMPLETE first but later completes, terminal wins."""
        pr1 = _make_parse_result(
            records=(_make_parsed_record("test_x", TestStatus.INCOMPLETE, "m.py"),),
            truncated=True,
        )
        pr2 = _make_parse_result(
            records=(_make_parsed_record("test_x", TestStatus.PASSED, "m.py"),),
        )

        result = assemble_partial_results(
            partials=[pr1, pr2],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.total_tests == 1
        assert result.records[0].outcome == TestOutcome.PASSED


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    """Records are ordered by module then by first appearance."""

    def test_records_ordered_by_module(self) -> None:
        pr = _make_parse_result(records=(
            _make_parsed_record("test_z", TestStatus.PASSED, "tests/z_module.py"),
            _make_parsed_record("test_a", TestStatus.PASSED, "tests/a_module.py"),
            _make_parsed_record("test_m", TestStatus.PASSED, "tests/m_module.py"),
        ))

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        modules = [r.module for r in result.records]
        assert modules == sorted(modules)

    def test_records_within_same_module_preserve_first_appearance_order(self) -> None:
        pr = _make_parse_result(records=(
            _make_parsed_record("test_first", TestStatus.PASSED, "tests/test_x.py"),
            _make_parsed_record("test_second", TestStatus.PASSED, "tests/test_x.py"),
            _make_parsed_record("test_third", TestStatus.PASSED, "tests/test_x.py"),
        ))

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        names = [r.test_name for r in result.records]
        assert names == ["test_first", "test_second", "test_third"]

    def test_ordering_across_multiple_partials(self) -> None:
        pr1 = _make_parse_result(records=(
            _make_parsed_record("test_b1", TestStatus.PASSED, "tests/b.py"),
        ))
        pr2 = _make_parse_result(records=(
            _make_parsed_record("test_a1", TestStatus.PASSED, "tests/a.py"),
        ))

        result = assemble_partial_results(
            partials=[pr1, pr2],
            run_id="r",
            session_id="s",
            host="h",
        )
        modules = [r.module for r in result.records]
        assert modules == ["tests/a.py", "tests/b.py"]


# ---------------------------------------------------------------------------
# Completeness ratio
# ---------------------------------------------------------------------------


class TestCompletenessComputation:
    """Assembler computes the completeness ratio correctly."""

    def test_completeness_with_explicit_expected(self) -> None:
        records = (
            _make_parsed_record("test_1", TestStatus.PASSED, "m.py"),
            _make_parsed_record("test_2", TestStatus.PASSED, "m.py"),
        )
        pr = _make_parse_result(records=records)

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
            expected_test_count=5,
        )
        assert result.completeness.executed == 2
        assert result.completeness.expected == 5
        assert result.completeness.ratio == pytest.approx(0.4)

    def test_completeness_defaults_to_executed_count(self) -> None:
        """When no expected count is given, expected = executed (100% of known)."""
        records = (
            _make_parsed_record("test_1", TestStatus.PASSED, "m.py"),
            _make_parsed_record("test_2", TestStatus.FAILED, "m.py"),
        )
        pr = _make_parse_result(records=records)

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.completeness.executed == 2
        assert result.completeness.expected == 2
        assert result.completeness.is_complete is True

    def test_completeness_excludes_incomplete_from_executed(self) -> None:
        """INCOMPLETE tests don't count as executed."""
        records = (
            _make_parsed_record("test_1", TestStatus.PASSED, "m.py"),
            _make_parsed_record("test_2", TestStatus.INCOMPLETE, "m.py"),
        )
        pr = _make_parse_result(records=records, truncated=True)

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
            expected_test_count=5,
        )
        assert result.completeness.executed == 1
        assert result.completeness.expected == 5

    def test_completeness_full_coverage(self) -> None:
        records = tuple(
            _make_parsed_record(f"test_{i}", TestStatus.PASSED, "m.py")
            for i in range(10)
        )
        pr = _make_parse_result(records=records)

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
            expected_test_count=10,
        )
        assert result.completeness.ratio == 1.0
        assert result.completeness.is_complete is True


# ---------------------------------------------------------------------------
# Coverage gaps
# ---------------------------------------------------------------------------


class TestCoverageGaps:
    """Assembler identifies modules with missing test coverage."""

    def test_gap_when_expected_module_has_no_tests(self) -> None:
        pr = _make_parse_result(records=(
            _make_parsed_record("test_a", TestStatus.PASSED, "tests/test_auth.py"),
        ))

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
            expected_modules=("tests/test_auth.py", "tests/test_payment.py"),
        )
        gap_modules = [g.module for g in result.coverage_gaps]
        assert "tests/test_payment.py" in gap_modules
        assert "tests/test_auth.py" not in gap_modules

    def test_no_gaps_when_all_modules_covered(self) -> None:
        pr = _make_parse_result(records=(
            _make_parsed_record("test_a", TestStatus.PASSED, "tests/test_auth.py"),
            _make_parsed_record("test_b", TestStatus.PASSED, "tests/test_payment.py"),
        ))

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
            expected_modules=("tests/test_auth.py", "tests/test_payment.py"),
        )
        assert result.coverage_gaps == ()

    def test_no_gaps_when_no_expected_modules(self) -> None:
        pr = _make_parse_result(records=(
            _make_parsed_record("test_a", TestStatus.PASSED, "m.py"),
        ))

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.coverage_gaps == ()

    def test_gap_severity_based_on_missing_count(self) -> None:
        """Gap for an entirely missing module should be HIGH or CRITICAL."""
        pr = _make_parse_result(records=())

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
            expected_modules=("tests/test_critical.py",),
        )
        assert len(result.coverage_gaps) == 1
        gap = result.coverage_gaps[0]
        assert gap.module == "tests/test_critical.py"
        assert gap.severity in (GapSeverity.HIGH, GapSeverity.CRITICAL)

    def test_gap_for_module_with_only_incomplete_tests(self) -> None:
        """Module with only INCOMPLETE tests still generates a gap."""
        pr = _make_parse_result(
            records=(
                _make_parsed_record("test_x", TestStatus.INCOMPLETE, "tests/test_flaky.py"),
            ),
            truncated=True,
        )

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
            expected_modules=("tests/test_flaky.py",),
        )
        gap_modules = [g.module for g in result.coverage_gaps]
        assert "tests/test_flaky.py" in gap_modules

    def test_coverage_gaps_are_immutable_tuple(self) -> None:
        pr = _make_parse_result(records=())

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
            expected_modules=("tests/test_a.py",),
        )
        assert isinstance(result.coverage_gaps, tuple)


# ---------------------------------------------------------------------------
# Interruption point
# ---------------------------------------------------------------------------


class TestInterruptionPoint:
    """Assembler records interruption metadata when output was truncated."""

    def test_no_interruption_for_complete_output(self) -> None:
        pr = _make_parse_result(
            records=(
                _make_parsed_record("test_a", TestStatus.PASSED, "m.py"),
            ),
            truncated=False,
        )

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.interruption.interrupted is False

    def test_interruption_when_truncated(self) -> None:
        pr = _make_parse_result(
            records=(
                _make_parsed_record("test_a", TestStatus.PASSED, "m.py"),
                _make_parsed_record("test_b", TestStatus.INCOMPLETE, "m.py"),
            ),
            truncated=True,
        )

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.interruption.interrupted is True
        assert result.interruption.reason != ""

    def test_interruption_records_last_test(self) -> None:
        """at_test should reference the last INCOMPLETE test."""
        pr = _make_parse_result(
            records=(
                _make_parsed_record("test_a", TestStatus.PASSED, "m.py"),
                _make_parsed_record("test_b", TestStatus.INCOMPLETE, "m.py"),
            ),
            truncated=True,
        )

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.interruption.at_test == "test_b"

    def test_interruption_with_truncated_but_no_incomplete(self) -> None:
        """Truncated output without INCOMPLETE tests still records interruption."""
        pr = _make_parse_result(
            records=(
                _make_parsed_record("test_a", TestStatus.PASSED, "m.py"),
            ),
            truncated=True,
        )

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.interruption.interrupted is True
        assert result.interruption.at_test == "test_a"

    def test_interruption_from_any_partial(self) -> None:
        """If any partial was truncated, the result is marked interrupted."""
        pr1 = _make_parse_result(
            records=(
                _make_parsed_record("test_a", TestStatus.PASSED, "m.py"),
            ),
            truncated=True,
        )
        pr2 = _make_parse_result(
            records=(
                _make_parsed_record("test_b", TestStatus.PASSED, "m.py"),
            ),
            truncated=False,
        )

        result = assemble_partial_results(
            partials=[pr1, pr2],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.interruption.interrupted is True

    def test_interruption_timestamp_is_timezone_aware(self) -> None:
        pr = _make_parse_result(
            records=(
                _make_parsed_record("test_a", TestStatus.INCOMPLETE, "m.py"),
            ),
            truncated=True,
        )

        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        if result.interruption.at_timestamp is not None:
            assert result.interruption.at_timestamp.tzinfo is not None

    def test_no_interruption_for_empty_partials(self) -> None:
        result = assemble_partial_results(
            partials=[],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.interruption.interrupted is False


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------


class TestStatusMapping:
    """TestStatus from the parser maps to the correct TestOutcome."""

    def test_all_status_mappings(self) -> None:
        status_to_outcome = {
            TestStatus.PASSED: TestOutcome.PASSED,
            TestStatus.FAILED: TestOutcome.FAILED,
            TestStatus.ERROR: TestOutcome.ERROR,
            TestStatus.SKIPPED: TestOutcome.SKIPPED,
            TestStatus.INCOMPLETE: TestOutcome.ERROR,
        }
        for status, expected_outcome in status_to_outcome.items():
            records = (_make_parsed_record(f"test_{status.value}", status, "m.py"),)
            pr = _make_parse_result(records=records, truncated=(status == TestStatus.INCOMPLETE))
            result = assemble_partial_results(
                partials=[pr],
                run_id="r",
                session_id="s",
                host="h",
            )
            assert result.records[0].outcome == expected_outcome, (
                f"Expected {status} -> {expected_outcome}, "
                f"got {result.records[0].outcome}"
            )


# ---------------------------------------------------------------------------
# Multiple partial results (multi-chunk assembly)
# ---------------------------------------------------------------------------


class TestMultiChunkAssembly:
    """Assembler correctly merges records from multiple partial chunks."""

    def test_records_from_multiple_partials_are_merged(self) -> None:
        pr1 = _make_parse_result(records=(
            _make_parsed_record("test_a", TestStatus.PASSED, "tests/test_auth.py"),
            _make_parsed_record("test_b", TestStatus.PASSED, "tests/test_auth.py"),
        ))
        pr2 = _make_parse_result(records=(
            _make_parsed_record("test_c", TestStatus.FAILED, "tests/test_cart.py"),
        ))

        result = assemble_partial_results(
            partials=[pr1, pr2],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.total_tests == 3
        assert result.passed_count == 2
        assert result.failed_count == 1

    def test_completeness_across_chunks(self) -> None:
        pr1 = _make_parse_result(records=(
            _make_parsed_record("test_1", TestStatus.PASSED, "m.py"),
        ))
        pr2 = _make_parse_result(records=(
            _make_parsed_record("test_2", TestStatus.PASSED, "m.py"),
        ))

        result = assemble_partial_results(
            partials=[pr1, pr2],
            run_id="r",
            session_id="s",
            host="h",
            expected_test_count=10,
        )
        assert result.completeness.executed == 2
        assert result.completeness.expected == 10

    def test_coverage_gaps_across_chunks(self) -> None:
        pr1 = _make_parse_result(records=(
            _make_parsed_record("test_a", TestStatus.PASSED, "tests/test_auth.py"),
        ))
        pr2 = _make_parse_result(records=(
            _make_parsed_record("test_b", TestStatus.PASSED, "tests/test_cart.py"),
        ))

        result = assemble_partial_results(
            partials=[pr1, pr2],
            run_id="r",
            session_id="s",
            host="h",
            expected_modules=(
                "tests/test_auth.py",
                "tests/test_cart.py",
                "tests/test_payment.py",
            ),
        )
        gap_modules = [g.module for g in result.coverage_gaps]
        assert "tests/test_payment.py" in gap_modules
        assert "tests/test_auth.py" not in gap_modules
        assert "tests/test_cart.py" not in gap_modules


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    """Assembler always returns a valid, immutable AssembledTestResult."""

    def test_returns_assembled_test_result(self) -> None:
        result = assemble_partial_results(
            partials=[],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert isinstance(result, AssembledTestResult)

    def test_result_is_frozen(self) -> None:
        result = assemble_partial_results(
            partials=[],
            run_id="r",
            session_id="s",
            host="h",
        )
        with pytest.raises(AttributeError):
            result.run_id = "changed"  # type: ignore[misc]

    def test_records_are_tuple(self) -> None:
        pr = _make_parse_result(records=(
            _make_parsed_record("test_a", TestStatus.PASSED, "m.py"),
        ))
        result = assemble_partial_results(
            partials=[pr],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert isinstance(result.records, tuple)

    def test_assembled_at_is_set(self) -> None:
        result = assemble_partial_results(
            partials=[],
            run_id="r",
            session_id="s",
            host="h",
        )
        assert result.assembled_at is not None
        assert result.assembled_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Assembler validates required parameters."""

    def test_empty_run_id_raises(self) -> None:
        with pytest.raises(ValueError):
            assemble_partial_results(
                partials=[],
                run_id="",
                session_id="s",
                host="h",
            )

    def test_empty_session_id_raises(self) -> None:
        with pytest.raises(ValueError):
            assemble_partial_results(
                partials=[],
                run_id="r",
                session_id="",
                host="h",
            )

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError):
            assemble_partial_results(
                partials=[],
                run_id="r",
                session_id="s",
                host="",
            )

    def test_negative_expected_test_count_raises(self) -> None:
        with pytest.raises(ValueError, match="expected_test_count must not be negative"):
            assemble_partial_results(
                partials=[],
                run_id="r",
                session_id="s",
                host="h",
                expected_test_count=-1,
            )
