"""Audit instrumentation for the result structuring pipeline stage.

Wires the ``StageAudit`` context manager into the result structuring step,
producing an audit entry that captures the structured output summary
(test counts, pass rate, outcome, completeness) for every command
execution that reaches the EXECUTION_COMPLETE stage.

The structured summary is a plain dict of JSON-primitive values, suitable
for embedding in the audit snapshot and for wiki YAML persistence.

Two public functions:

``build_result_summary``
    Pure function that extracts a flat summary dict from an
    ``AssembledTestResult``. Contains test counts, pass rate, outcome,
    run identifiers, timing, and completeness data. All values are
    JSON-primitive types (str, int, float, bool, None).

``audit_result_structuring``
    Wraps ``build_result_summary`` in a ``StageAudit`` context manager.
    Records the assembled result's key identifiers as inputs in the
    before-snapshot and the structured summary as output in the
    after-snapshot. Returns a ``StageResult`` with the summary dict
    as value and the updated audit chain.

Design principles:
    - Pure summary builder: no I/O, no side effects
    - Immutable outputs: ``StageResult`` is frozen
    - Defensive copies: summary dict is a fresh dict on every call
    - Composable: chain multiple stages by threading the chain through

Usage::

    from jules_daemon.audit.result_stage import audit_result_structuring
    from jules_daemon.audit_models import AuditChain

    chain = AuditChain.empty()
    # ... earlier stages append entries to chain ...
    stage_result = audit_result_structuring(assembled_result, chain)
    stage_result.value   # summary dict
    stage_result.chain   # chain with result_structuring entry appended
    stage_result.entry   # the AuditEntry for this stage
"""

from __future__ import annotations

from typing import Any

from jules_daemon.audit.instrumentation import StageAudit, StageResult
from jules_daemon.audit_models import AuditChain
from jules_daemon.wiki.assembled_result import AssembledTestResult

__all__ = [
    "audit_result_structuring",
    "build_result_summary",
]

_STAGE_NAME = "result_structuring"


# ---------------------------------------------------------------------------
# Outcome classification (mirrors test_result_writer logic)
# ---------------------------------------------------------------------------


def _classify_outcome(result: AssembledTestResult) -> str:
    """Classify the overall result into a human-readable outcome string.

    Returns one of: 'passed', 'failed', 'interrupted', 'empty'.
    """
    if result.total_tests == 0:
        return "empty"
    if result.was_interrupted:
        return "interrupted"
    if result.has_failures:
        return "failed"
    return "passed"


# ---------------------------------------------------------------------------
# Pure summary builder
# ---------------------------------------------------------------------------


def build_result_summary(result: AssembledTestResult) -> dict[str, Any]:
    """Extract a flat summary dict from an AssembledTestResult.

    The summary contains only JSON-primitive values (str, int, float,
    bool, None) so it can be embedded directly in audit snapshots and
    serialized to YAML without any special handling.

    Args:
        result: The assembled test result to summarize.

    Returns:
        A plain dict with the structured output summary. Keys:
            - run_id, session_id, host
            - outcome: 'passed' | 'failed' | 'interrupted' | 'empty'
            - tests_passed, tests_failed, tests_skipped, tests_errored, tests_total
            - pass_rate: float in [0.0, 1.0]
            - total_duration_seconds: float
            - was_interrupted: bool
            - completeness_executed, completeness_expected: int
    """
    return {
        "run_id": result.run_id,
        "session_id": result.session_id,
        "host": result.host,
        "outcome": _classify_outcome(result),
        "tests_passed": result.passed_count,
        "tests_failed": result.failed_count,
        "tests_skipped": result.skipped_count,
        "tests_errored": result.error_count,
        "tests_total": result.total_tests,
        "pass_rate": result.pass_rate,
        "total_duration_seconds": result.total_duration_seconds,
        "was_interrupted": result.was_interrupted,
        "completeness_executed": result.completeness.executed,
        "completeness_expected": result.completeness.expected,
    }


# ---------------------------------------------------------------------------
# Audit-instrumented result structuring
# ---------------------------------------------------------------------------


def audit_result_structuring(
    result: AssembledTestResult,
    chain: AuditChain,
) -> StageResult:
    """Structure a test result and record an audit entry with the summary.

    Wraps ``build_result_summary`` in a ``StageAudit`` context manager,
    capturing the assembled result's key identifiers as before-snapshot
    inputs and the structured summary as after-snapshot output.

    Args:
        result: The assembled test result to structure and audit.
        chain: The current audit chain to append to.

    Returns:
        A ``StageResult`` whose:
            - ``value``: the structured summary dict
            - ``chain``: the chain with a new "result_structuring" entry
            - ``entry``: the ``AuditEntry`` for this stage
    """
    audit = StageAudit(
        _STAGE_NAME,
        chain,
        inputs={
            "run_id": result.run_id,
            "session_id": result.session_id,
            "host": result.host,
            "tests_total": result.total_tests,
        },
    )
    summary: dict[str, Any] | None = None
    with audit:
        summary = build_result_summary(result)
        audit.record_output({"summary": summary})

    audit_entry = audit.entry
    assert audit_entry is not None, "StageAudit.entry must be set after context block"
    return StageResult(
        value=summary,
        chain=audit.chain,
        entry=audit_entry,
    )
