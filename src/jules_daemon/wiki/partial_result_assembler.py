"""Partial result assembler for interrupted or chunked test output.

Accepts a sequence of ``ParseResult`` objects (from the test output parser),
merges their records by deduplication and ordering, computes a completeness
ratio, identifies coverage gaps, and records the interruption point.

Returns a single immutable ``AssembledTestResult``.

Design:
    - Pure function: no side effects, no disk I/O, no mutation
    - Deduplication key: (test_name, module) -- last occurrence wins
    - Ordering: records sorted by module (ascending), then by first-seen order
    - Completeness: counts tests with terminal status as "executed"
    - Coverage gaps: compares observed modules against expected modules
    - Interruption: detected from truncated flag or INCOMPLETE test status
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from typing import Sequence

from jules_daemon.monitor.test_output_parser import (
    ParseResult,
    TestRecord as ParsedTestRecord,
    TestStatus,
)
from jules_daemon.wiki.assembled_result import (
    AssembledTestResult,
    CompletenessRatio,
    CoverageGap,
    DaemonDowntime,
    GapSeverity,
    InterruptionPoint,
    TestOutcome,
    TestRecord as AssembledTestRecord,
)

__all__ = ["assemble_partial_results"]


# ---------------------------------------------------------------------------
# Status mapping: parser TestStatus -> assembled TestOutcome
# ---------------------------------------------------------------------------

_STATUS_TO_OUTCOME: dict[TestStatus, TestOutcome] = {
    TestStatus.PASSED: TestOutcome.PASSED,
    TestStatus.FAILED: TestOutcome.FAILED,
    TestStatus.ERROR: TestOutcome.ERROR,
    TestStatus.SKIPPED: TestOutcome.SKIPPED,
    TestStatus.INCOMPLETE: TestOutcome.ERROR,
}


# ---------------------------------------------------------------------------
# Internal helpers (pure functions, no mutation of inputs)
# ---------------------------------------------------------------------------


def _map_status(status: TestStatus) -> TestOutcome:
    """Map a parser TestStatus to an assembled TestOutcome."""
    return _STATUS_TO_OUTCOME.get(status, TestOutcome.ERROR)


def _convert_record(parsed: ParsedTestRecord) -> AssembledTestRecord:
    """Convert a parsed test record to an assembled test record.

    INCOMPLETE tests get an error_message indicating they were interrupted.
    """
    outcome = _map_status(parsed.status)
    error_message = ""
    if parsed.status == TestStatus.INCOMPLETE:
        error_message = "Test incomplete: output was interrupted before result"

    return AssembledTestRecord(
        test_name=parsed.name,
        outcome=outcome,
        duration_seconds=parsed.duration_seconds,
        error_message=error_message,
        module=parsed.module,
        line_number=(
            parsed.line_number + 1
            if parsed.line_number is not None and parsed.line_number >= 0
            else None
        ),
    )


def _dedup_key(record: ParsedTestRecord) -> tuple[str, str]:
    """Deduplication key for a parsed test record: (name, module)."""
    return (record.name, record.module)


def _deduplicate_records(
    partials: Sequence[ParseResult],
) -> tuple[AssembledTestRecord, ...]:
    """Merge records from all partials, deduplicating by (name, module).

    Last occurrence wins. Results are ordered by module (ascending),
    then by first-seen insertion order within each module.

    Returns an immutable tuple of AssembledTestRecord.
    """
    # Use OrderedDict to preserve first-seen insertion order for each key.
    # When a duplicate is found, the value is replaced but insertion order
    # of the *first* appearance is retained via move_to_end=False semantics.
    seen: OrderedDict[tuple[str, str], ParsedTestRecord] = OrderedDict()

    for partial in partials:
        for record in partial.records:
            key = _dedup_key(record)
            seen[key] = record  # last write wins

    # Convert to assembled records
    converted = [_convert_record(rec) for rec in seen.values()]

    # Sort by module (ascending), stable sort preserves insertion order within module
    converted.sort(key=lambda r: r.module)

    return tuple(converted)


def _compute_completeness(
    records: tuple[AssembledTestRecord, ...],
    expected_test_count: int | None,
    original_partials: Sequence[ParseResult],
) -> CompletenessRatio:
    """Compute the completeness ratio.

    'Executed' counts only tests with a terminal outcome (not INCOMPLETE/ERROR
    from interruption). If expected_test_count is None, expected defaults to
    the number of executed tests (i.e., 100% of what was observed).
    """
    # Count tests that actually completed (terminal parser status).
    # We need to check the original parsed status, not just the assembled outcome.
    # Build a set of keys that had INCOMPLETE status as their final form.
    incomplete_keys: set[tuple[str, str]] = set()
    for partial in original_partials:
        for record in partial.records:
            key = _dedup_key(record)
            if record.status == TestStatus.INCOMPLETE:
                incomplete_keys.add(key)
            elif record.status.is_terminal:
                incomplete_keys.discard(key)

    executed = sum(
        1 for r in records
        if (r.test_name, r.module) not in incomplete_keys
    )

    if expected_test_count is None:
        expected = executed
    else:
        expected = expected_test_count

    return CompletenessRatio(executed=executed, expected=expected)


def _identify_coverage_gaps(
    records: tuple[AssembledTestRecord, ...],
    expected_modules: tuple[str, ...],
    incomplete_keys: frozenset[tuple[str, str]],
) -> tuple[CoverageGap, ...]:
    """Identify modules that have missing or insufficient test coverage.

    A gap is reported when:
    - An expected module has zero tests in the results
    - An expected module has only incomplete/interrupted tests

    Args:
        records: Deduplicated, assembled test records.
        expected_modules: Modules that should have test coverage.
        incomplete_keys: Set of (name, module) keys that remained INCOMPLETE.

    Returns:
        Immutable tuple of CoverageGap instances.
    """
    if not expected_modules:
        return ()

    # Count completed (non-incomplete) tests per module
    completed_per_module: dict[str, int] = {}
    total_per_module: dict[str, int] = {}

    for record in records:
        total_per_module[record.module] = total_per_module.get(record.module, 0) + 1
        if (record.test_name, record.module) not in incomplete_keys:
            completed_per_module[record.module] = (
                completed_per_module.get(record.module, 0) + 1
            )

    gaps: list[CoverageGap] = []
    for module in expected_modules:
        completed = completed_per_module.get(module, 0)
        total = total_per_module.get(module, 0)

        if completed > 0:
            # Module has at least some completed tests -- no gap
            continue

        if total == 0:
            # Module entirely missing
            gaps.append(CoverageGap(
                module=module,
                reason=f"No tests executed from module {module}",
                severity=GapSeverity.HIGH,
                expected_tests=0,
                actual_tests=0,
            ))
        else:
            # Module has tests but all are incomplete
            gaps.append(CoverageGap(
                module=module,
                reason=f"All tests in {module} were interrupted before completion",
                severity=GapSeverity.HIGH,
                expected_tests=total,
                actual_tests=0,
            ))

    return tuple(gaps)


def _build_interruption_point(
    partials: Sequence[ParseResult],
) -> InterruptionPoint:
    """Build an InterruptionPoint from the partial results.

    The run is marked as interrupted if any partial has truncated=True
    or contains INCOMPLETE tests. The at_test field records the last
    test that was in progress at interruption time.
    """
    any_truncated = any(p.truncated for p in partials)
    incomplete_tests: list[ParsedTestRecord] = []

    for partial in partials:
        for record in partial.records:
            if record.status == TestStatus.INCOMPLETE:
                incomplete_tests.append(record)

    if not any_truncated and not incomplete_tests:
        return InterruptionPoint()

    # Determine the test that was running at interruption
    if incomplete_tests:
        # Last incomplete test is the interruption point
        at_test = incomplete_tests[-1].name
        reason = (
            f"Output interrupted during test execution; "
            f"{len(incomplete_tests)} test(s) did not complete"
        )
    else:
        # Truncated but no incomplete tests -- use the last test from the
        # last truncated partial
        at_test = ""
        for partial in reversed(partials):
            if partial.truncated and partial.records:
                at_test = partial.records[-1].name
                break
        reason = "Output was truncated before test suite completion"

    return InterruptionPoint(
        interrupted=True,
        at_test=at_test,
        at_timestamp=datetime.now(timezone.utc),
        reason=reason,
    )


def _compute_incomplete_keys(
    partials: Sequence[ParseResult],
) -> frozenset[tuple[str, str]]:
    """Compute the set of (name, module) keys that ended as INCOMPLETE.

    A key is considered incomplete if its last occurrence across all
    partials had INCOMPLETE status.
    """
    status_map: dict[tuple[str, str], TestStatus] = {}
    for partial in partials:
        for record in partial.records:
            status_map[_dedup_key(record)] = record.status

    return frozenset(
        key for key, status in status_map.items()
        if status == TestStatus.INCOMPLETE
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_partial_results(
    partials: Sequence[ParseResult],
    *,
    run_id: str,
    session_id: str,
    host: str,
    expected_test_count: int | None = None,
    expected_modules: tuple[str, ...] = (),
    daemon_downtime: DaemonDowntime | None = None,
) -> AssembledTestResult:
    """Assemble partial test results into a single structured result.

    Takes a sequence of ``ParseResult`` objects (typically from parsing
    chunked or interrupted test output), merges them by deduplication,
    computes completeness, identifies coverage gaps, and records any
    interruption point.

    Args:
        partials: Sequence of parsed partial test results.
        run_id: Identifier of the daemon run.
        session_id: SSH monitoring session identifier.
        host: Remote host where tests executed.
        expected_test_count: Number of tests that were expected to run.
            Defaults to the number of actually-executed tests.
        expected_modules: Modules that should have test coverage.
            Used to identify coverage gaps.
        daemon_downtime: Optional daemon downtime metadata. When the
            daemon was down during execution, this allows downstream
            consumers to distinguish partial-due-to-crash from
            partial-due-to-timeout. Defaults to no-downtime.

    Returns:
        An immutable ``AssembledTestResult``.

    Raises:
        ValueError: If run_id, session_id, or host is empty, or if
            expected_test_count is negative.
    """
    # Validate inputs eagerly (fail fast before any work)
    if expected_test_count is not None and expected_test_count < 0:
        raise ValueError("expected_test_count must not be negative")

    # Deduplicate and order records
    records = _deduplicate_records(partials)

    # Compute which keys remained INCOMPLETE
    incomplete_keys = _compute_incomplete_keys(partials)

    # Compute completeness
    completeness = _compute_completeness(records, expected_test_count, partials)

    # Identify coverage gaps
    coverage_gaps = _identify_coverage_gaps(records, expected_modules, incomplete_keys)

    # Build interruption point
    interruption = _build_interruption_point(partials)

    # Use provided daemon downtime or default to no-downtime
    resolved_downtime = daemon_downtime if daemon_downtime is not None else DaemonDowntime()

    # Assemble the final result (AssembledTestResult validates run_id etc.)
    return AssembledTestResult(
        run_id=run_id,
        session_id=session_id,
        host=host,
        records=records,
        completeness=completeness,
        coverage_gaps=coverage_gaps,
        interruption=interruption,
        daemon_downtime=resolved_downtime,
    )
