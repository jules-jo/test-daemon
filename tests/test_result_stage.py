"""Tests for audit instrumentation wired into the result structuring stage.

Verifies that ``audit_result_structuring`` wraps the result structuring
step with a ``StageAudit`` context manager, producing an audit entry that
captures the structured output summary (test counts, pass rate, outcome).

Covers:
- Returns a StageResult with the structured summary as value
- Appends a single entry to the audit chain
- Entry stage name is "result_structuring"
- Entry status is "success" for normal operation
- Output snapshot contains the structured summary dict
- Summary includes pass/fail/skip/error counts, pass_rate, outcome
- Summary includes run_id, host, and total_duration_seconds
- Chains correctly when appended to an existing chain
- Duration is non-negative
- Handles empty test results correctly
- Handles results with failures
- Handles interrupted results
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jules_daemon.audit.instrumentation import StageResult
from jules_daemon.audit.result_stage import (
    audit_result_structuring,
    build_result_summary,
)
from jules_daemon.audit_models import AuditChain, AuditEntry
from jules_daemon.wiki.assembled_result import (
    AssembledTestResult,
    CompletenessRatio,
    InterruptionPoint,
    TestOutcome,
    TestRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


def _empty_chain() -> AuditChain:
    return AuditChain.empty()


def _make_passing_result() -> AssembledTestResult:
    """Build a fully-passing test result with 3 tests."""
    records = (
        TestRecord(
            test_name="test_login",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.5,
            module="auth/test_login.py",
        ),
        TestRecord(
            test_name="test_logout",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.3,
            module="auth/test_login.py",
        ),
        TestRecord(
            test_name="test_order",
            outcome=TestOutcome.PASSED,
            duration_seconds=1.2,
            module="orders/test_flow.py",
        ),
    )
    return AssembledTestResult(
        run_id="run-001",
        session_id="sess-001",
        host="staging.example.com",
        records=records,
        completeness=CompletenessRatio(executed=3, expected=3),
    )


def _make_failing_result() -> AssembledTestResult:
    """Build a result with mixed outcomes."""
    records = (
        TestRecord(
            test_name="test_login",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.5,
        ),
        TestRecord(
            test_name="test_payment",
            outcome=TestOutcome.FAILED,
            duration_seconds=2.3,
            error_message="AssertionError: expected 200",
        ),
        TestRecord(
            test_name="test_webhook",
            outcome=TestOutcome.ERROR,
            duration_seconds=0.1,
            error_message="ConnectionRefusedError",
        ),
        TestRecord(
            test_name="test_legacy",
            outcome=TestOutcome.SKIPPED,
        ),
    )
    return AssembledTestResult(
        run_id="run-002",
        session_id="sess-002",
        host="prod.example.com",
        records=records,
        completeness=CompletenessRatio(executed=3, expected=5),
    )


def _make_empty_result() -> AssembledTestResult:
    """Build a result with no test records."""
    return AssembledTestResult(
        run_id="run-003",
        session_id="sess-003",
        host="empty.example.com",
    )


def _make_interrupted_result() -> AssembledTestResult:
    """Build a result that was interrupted mid-run."""
    records = (
        TestRecord(
            test_name="test_login",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.5,
        ),
    )
    return AssembledTestResult(
        run_id="run-004",
        session_id="sess-004",
        host="dev.example.com",
        records=records,
        completeness=CompletenessRatio(executed=1, expected=10),
        interruption=InterruptionPoint(
            interrupted=True,
            reason="SSH connection lost",
        ),
    )


# ---------------------------------------------------------------------------
# build_result_summary -- pure function tests
# ---------------------------------------------------------------------------


class TestBuildResultSummary:
    """Tests for the pure summary builder."""

    def test_passing_result_summary(self) -> None:
        result = _make_passing_result()
        summary = build_result_summary(result)
        assert summary["outcome"] == "passed"
        assert summary["tests_passed"] == 3
        assert summary["tests_failed"] == 0
        assert summary["tests_skipped"] == 0
        assert summary["tests_errored"] == 0
        assert summary["tests_total"] == 3
        assert summary["pass_rate"] == pytest.approx(1.0)

    def test_failing_result_summary(self) -> None:
        result = _make_failing_result()
        summary = build_result_summary(result)
        assert summary["outcome"] == "failed"
        assert summary["tests_passed"] == 1
        assert summary["tests_failed"] == 1
        assert summary["tests_errored"] == 1
        assert summary["tests_skipped"] == 1
        assert summary["tests_total"] == 4
        assert summary["pass_rate"] == pytest.approx(0.25)

    def test_empty_result_summary(self) -> None:
        result = _make_empty_result()
        summary = build_result_summary(result)
        assert summary["outcome"] == "empty"
        assert summary["tests_total"] == 0
        assert summary["pass_rate"] == 0.0

    def test_interrupted_result_summary(self) -> None:
        result = _make_interrupted_result()
        summary = build_result_summary(result)
        assert summary["outcome"] == "interrupted"
        assert summary["was_interrupted"] is True

    def test_summary_includes_run_id(self) -> None:
        result = _make_passing_result()
        summary = build_result_summary(result)
        assert summary["run_id"] == "run-001"

    def test_summary_includes_host(self) -> None:
        result = _make_passing_result()
        summary = build_result_summary(result)
        assert summary["host"] == "staging.example.com"

    def test_summary_includes_total_duration(self) -> None:
        result = _make_passing_result()
        summary = build_result_summary(result)
        assert summary["total_duration_seconds"] == pytest.approx(2.0)

    def test_summary_includes_completeness(self) -> None:
        result = _make_failing_result()
        summary = build_result_summary(result)
        assert summary["completeness_executed"] == 3
        assert summary["completeness_expected"] == 5

    def test_summary_is_plain_dict(self) -> None:
        """Summary must be a plain dict (JSON-serializable for audit snapshot)."""
        result = _make_passing_result()
        summary = build_result_summary(result)
        assert isinstance(summary, dict)
        # All values should be JSON-primitive types
        for value in summary.values():
            assert isinstance(value, (str, int, float, bool, type(None)))


# ---------------------------------------------------------------------------
# audit_result_structuring -- instrumented wrapper tests
# ---------------------------------------------------------------------------


class TestAuditResultStructuring:
    """Tests for the audit-instrumented result structuring."""

    def test_returns_stage_result(self) -> None:
        result = _make_passing_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        assert isinstance(stage_result, StageResult)

    def test_value_is_summary_dict(self) -> None:
        result = _make_passing_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        assert isinstance(stage_result.value, dict)
        assert stage_result.value["outcome"] == "passed"
        assert stage_result.value["tests_total"] == 3

    def test_chain_has_one_entry(self) -> None:
        result = _make_passing_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        assert len(stage_result.chain) == 1

    def test_entry_stage_name(self) -> None:
        result = _make_passing_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        assert stage_result.entry.stage == "result_structuring"

    def test_entry_status_success(self) -> None:
        result = _make_passing_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        assert stage_result.entry.status == "success"

    def test_entry_has_no_error(self) -> None:
        result = _make_passing_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        assert stage_result.entry.error is None

    def test_entry_duration_non_negative(self) -> None:
        result = _make_passing_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        assert stage_result.entry.duration is not None
        assert stage_result.entry.duration >= 0.0

    def test_after_snapshot_contains_summary(self) -> None:
        """The after-snapshot should contain the structured output summary."""
        result = _make_failing_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        after = stage_result.entry.after_snapshot
        assert isinstance(after, dict)
        outputs = after["partial_outputs"]
        assert "summary" in outputs
        summary = outputs["summary"]
        assert summary["outcome"] == "failed"
        assert summary["tests_failed"] == 1
        assert summary["tests_errored"] == 1

    def test_before_snapshot_contains_run_id(self) -> None:
        """The before-snapshot should capture the input run_id."""
        result = _make_passing_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        before = stage_result.entry.before_snapshot
        assert isinstance(before, dict)
        inputs = before["inputs"]
        assert inputs["run_id"] == "run-001"

    def test_before_snapshot_contains_host(self) -> None:
        """The before-snapshot should capture the input host."""
        result = _make_passing_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        before = stage_result.entry.before_snapshot
        assert isinstance(before, dict)
        inputs = before["inputs"]
        assert inputs["host"] == "staging.example.com"

    def test_appends_to_existing_chain(self) -> None:
        """When given a pre-populated chain, appends rather than replaces."""
        prior_entry = AuditEntry(
            stage="ssh_dispatch",
            timestamp=_T0,
            before_snapshot=None,
            after_snapshot=None,
            duration=1.0,
            status="success",
            error=None,
        )
        chain = _empty_chain().append(prior_entry)
        result = _make_passing_result()
        stage_result = audit_result_structuring(result, chain)
        assert len(stage_result.chain) == 2
        assert stage_result.chain.stages == ("ssh_dispatch", "result_structuring")

    def test_chain_is_new_instance(self) -> None:
        """Original chain is not mutated."""
        chain = _empty_chain()
        result = _make_passing_result()
        stage_result = audit_result_structuring(result, chain)
        assert len(chain) == 0  # original unchanged
        assert len(stage_result.chain) == 1

    def test_empty_result_produces_valid_entry(self) -> None:
        result = _make_empty_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        assert stage_result.entry.stage == "result_structuring"
        assert stage_result.entry.status == "success"
        assert stage_result.value["outcome"] == "empty"

    def test_interrupted_result_produces_valid_entry(self) -> None:
        result = _make_interrupted_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        assert stage_result.entry.stage == "result_structuring"
        assert stage_result.value["outcome"] == "interrupted"
        assert stage_result.value["was_interrupted"] is True

    def test_chain_serializes(self) -> None:
        """The resulting chain can be serialized to a list."""
        result = _make_passing_result()
        stage_result = audit_result_structuring(result, _empty_chain())
        serialized = stage_result.chain.to_list()
        assert len(serialized) == 1
        assert serialized[0]["stage"] == "result_structuring"
        assert serialized[0]["status"] == "success"
