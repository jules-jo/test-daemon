"""Structured result dataclass for assembled test execution outcomes.

After a test suite completes (or is interrupted), the daemon assembles
the raw monitoring data into an ``AssembledTestResult`` -- a single
immutable snapshot containing:

- Aggregated per-test records with outcome and timing
- A completeness ratio (executed vs expected tests)
- A list of coverage gaps (modules/areas with missing coverage)
- Interruption point metadata (if the run was cut short)
- Daemon downtime metadata (to distinguish crash-partial from timeout-partial)

All types are frozen dataclasses with validation in ``__post_init__``.
Collections use tuples for full immutability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

__all__ = [
    "AssembledTestResult",
    "CompletenessRatio",
    "CoverageGap",
    "DaemonDowntime",
    "GapSeverity",
    "InterruptionPoint",
    "TestOutcome",
    "TestRecord",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestOutcome(Enum):
    """Result of an individual test case execution."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class GapSeverity(Enum):
    """Severity level for a coverage gap."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestRecord:
    """Result record for a single test case.

    Immutable -- each record is created once during result assembly.

    Fields:
        test_name        -- fully-qualified test name (e.g. ``test_login_flow``)
        outcome          -- pass/fail/skip/error
        duration_seconds -- wall-clock time for this test (None if unknown)
        error_message    -- failure or error detail (empty if passed/skipped)
        module           -- source module or file path
        line_number      -- 1-based source line where the test is defined (None if unknown)
    """

    test_name: str
    outcome: TestOutcome
    duration_seconds: float | None = None
    error_message: str = ""
    module: str = ""
    line_number: int | None = None

    def __post_init__(self) -> None:
        if not self.test_name:
            raise ValueError("test_name must not be empty")
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("duration_seconds must not be negative")
        if self.line_number is not None and self.line_number < 1:
            raise ValueError("line_number must be >= 1")

    @property
    def is_failure(self) -> bool:
        """True if this test failed or errored."""
        return self.outcome in (TestOutcome.FAILED, TestOutcome.ERROR)


@dataclass(frozen=True)
class CompletenessRatio:
    """Ratio of tests executed vs tests expected.

    Tracks how much of the intended test suite actually ran. A ratio below
    1.0 indicates tests were skipped, filtered out, or lost to interruption.

    Fields:
        executed -- number of tests that actually ran
        expected -- number of tests that were expected to run
    """

    executed: int = 0
    expected: int = 0

    def __post_init__(self) -> None:
        if self.executed < 0:
            raise ValueError("executed must not be negative")
        if self.expected < 0:
            raise ValueError("expected must not be negative")
        if self.executed > self.expected:
            raise ValueError(
                f"executed must not exceed expected "
                f"({self.executed} > {self.expected})"
            )

    @property
    def ratio(self) -> float:
        """Completion ratio as a float in [0.0, 1.0].

        Returns 0.0 when expected is zero (no tests expected).
        """
        if self.expected == 0:
            return 0.0
        return self.executed / self.expected

    @property
    def is_complete(self) -> bool:
        """True if all expected tests were executed."""
        return self.executed == self.expected


@dataclass(frozen=True)
class CoverageGap:
    """A specific gap in test coverage identified during result assembly.

    Fields:
        module         -- module or area where coverage is missing
        reason         -- human-readable description of the gap
        severity       -- how critical this gap is
        expected_tests -- number of tests expected in this module
        actual_tests   -- number of tests that actually ran
    """

    module: str
    reason: str
    severity: GapSeverity = GapSeverity.MEDIUM
    expected_tests: int = 0
    actual_tests: int = 0

    def __post_init__(self) -> None:
        if not self.module:
            raise ValueError("module must not be empty")
        if not self.reason:
            raise ValueError("reason must not be empty")
        if self.expected_tests < 0:
            raise ValueError("expected_tests must not be negative")
        if self.actual_tests < 0:
            raise ValueError("actual_tests must not be negative")
        if (
            self.expected_tests > 0
            and self.actual_tests > self.expected_tests
        ):
            raise ValueError(
                f"actual_tests must not exceed expected_tests "
                f"({self.actual_tests} > {self.expected_tests})"
            )


@dataclass(frozen=True)
class InterruptionPoint:
    """Metadata about where and why test execution was interrupted.

    When ``interrupted`` is False, all other fields carry their defaults.
    When ``interrupted`` is True, ``reason`` must be provided.

    Fields:
        interrupted  -- whether the run was interrupted before completion
        at_test      -- name of the test that was running at interruption
        at_timestamp -- UTC time when interruption occurred
        reason       -- human-readable interruption reason
        exit_code    -- process exit code at interruption (None if unknown)
    """

    interrupted: bool = False
    at_test: str = ""
    at_timestamp: datetime | None = None
    reason: str = ""
    exit_code: int | None = None

    def __post_init__(self) -> None:
        if self.interrupted and not self.reason:
            raise ValueError("reason must not be empty when interrupted")
        if self.at_timestamp is not None and self.at_timestamp.tzinfo is None:
            raise ValueError("at_timestamp must be timezone-aware")


@dataclass(frozen=True)
class DaemonDowntime:
    """Metadata about daemon downtime during a test run.

    Enables downstream consumers to distinguish results that are partial
    because the daemon crashed (daemon_was_down=True) from results that
    are partial because of a timeout or other reason while the daemon was
    running the entire time (daemon_was_down=False).

    When ``daemon_was_down`` is True, ``estimated_down_seconds`` must be
    provided (may be 0.0 for an instant restart). Timestamps are optional
    but recommended for audit trail completeness.

    Fields:
        daemon_was_down       -- whether the daemon was down during this run
        down_started_at       -- estimated start of downtime (None if not down)
        down_ended_at         -- estimated end of downtime (None if not down)
        estimated_down_seconds -- estimated duration of downtime in seconds
        recovery_method       -- how the daemon recovered (e.g. 'reconnect',
                                 'resume_approval', 'fresh_start')
    """

    daemon_was_down: bool = False
    down_started_at: datetime | None = None
    down_ended_at: datetime | None = None
    estimated_down_seconds: float | None = None
    recovery_method: str = ""

    def __post_init__(self) -> None:
        if self.daemon_was_down and self.estimated_down_seconds is None:
            raise ValueError(
                "estimated_down_seconds must be provided when daemon_was_down is True"
            )
        if self.estimated_down_seconds is not None and self.estimated_down_seconds < 0:
            raise ValueError("estimated_down_seconds must not be negative")
        if self.down_started_at is not None and self.down_started_at.tzinfo is None:
            raise ValueError("down_started_at must be timezone-aware")
        if self.down_ended_at is not None and self.down_ended_at.tzinfo is None:
            raise ValueError("down_ended_at must be timezone-aware")


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AssembledTestResult:
    """Top-level structured result assembled from a completed test run.

    This is the immutable output of the result-assembly phase. It
    aggregates per-test records, a completeness ratio, identified
    coverage gaps, interruption metadata, and daemon downtime info
    into a single frozen snapshot suitable for wiki persistence and
    reporting.

    The ``daemon_downtime`` field enables downstream consumers to
    distinguish results that are partial because the daemon crashed
    (partial-due-to-crash) from results that are partial because of
    a timeout (partial-due-to-timeout). Both cases have
    ``was_interrupted=True``, but only crash cases have
    ``daemon_was_down=True``.

    Fields:
        run_id           -- identifier of the daemon run that produced this result
        session_id       -- SSH monitoring session identifier
        host             -- remote host where tests were executed
        records          -- per-test result records (immutable tuple)
        completeness     -- ratio of executed to expected tests
        coverage_gaps    -- identified coverage gaps (immutable tuple)
        interruption     -- interruption point metadata
        daemon_downtime  -- daemon downtime metadata for crash vs timeout
        assembled_at     -- UTC timestamp when this result was assembled
    """

    run_id: str
    session_id: str
    host: str
    records: tuple[TestRecord, ...] = ()
    completeness: CompletenessRatio = field(default_factory=CompletenessRatio)
    coverage_gaps: tuple[CoverageGap, ...] = ()
    interruption: InterruptionPoint = field(default_factory=InterruptionPoint)
    daemon_downtime: DaemonDowntime = field(default_factory=DaemonDowntime)
    assembled_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.host:
            raise ValueError("host must not be empty")

    # -- Computed properties --

    @property
    def total_tests(self) -> int:
        """Total number of test records."""
        return len(self.records)

    @property
    def passed_count(self) -> int:
        """Number of tests that passed."""
        return sum(1 for r in self.records if r.outcome == TestOutcome.PASSED)

    @property
    def failed_count(self) -> int:
        """Number of tests that failed."""
        return sum(1 for r in self.records if r.outcome == TestOutcome.FAILED)

    @property
    def skipped_count(self) -> int:
        """Number of tests that were skipped."""
        return sum(1 for r in self.records if r.outcome == TestOutcome.SKIPPED)

    @property
    def error_count(self) -> int:
        """Number of tests that errored."""
        return sum(1 for r in self.records if r.outcome == TestOutcome.ERROR)

    @property
    def pass_rate(self) -> float:
        """Ratio of passed tests to total tests.

        Returns 0.0 when there are no records.
        """
        if not self.records:
            return 0.0
        return self.passed_count / len(self.records)

    @property
    def has_failures(self) -> bool:
        """True if any test failed or errored."""
        return any(r.is_failure for r in self.records)

    @property
    def was_interrupted(self) -> bool:
        """True if the run was interrupted before completion."""
        return self.interruption.interrupted

    @property
    def daemon_was_down(self) -> bool:
        """True if the daemon was down during this run.

        Allows downstream consumers to distinguish partial results caused
        by a daemon crash from partial results caused by timeouts or other
        reasons where the daemon was running the entire time.
        """
        return self.daemon_downtime.daemon_was_down

    @property
    def failed_records(self) -> tuple[TestRecord, ...]:
        """Subset of records that failed or errored."""
        return tuple(r for r in self.records if r.is_failure)

    @property
    def total_duration_seconds(self) -> float:
        """Sum of all test durations (ignoring None values).

        Returns 0.0 when no durations are available.
        """
        return sum(
            r.duration_seconds
            for r in self.records
            if r.duration_seconds is not None
        )
