"""Generic workflow step output interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jules_daemon.monitor.test_output_parser import (
    FrameworkHint,
    OutputContext,
    TestStatus,
    parse_interrupted_output,
)

_SUMMARY_FIELD_ORDER: tuple[str, ...] = (
    "passed",
    "failed",
    "skipped",
    "error",
    "incomplete",
)


def _last_meaningful_lines(raw_output: str, *, max_lines: int = 3) -> list[str]:
    """Return the last few non-empty lines from the raw output."""
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    return lines[-max_lines:]


def _format_summary_counts(summary: dict[str, int], *, active: bool) -> str | None:
    """Format parser summary counts into a short status message."""
    parts = [
        f"{int(summary.get(field, 0) or 0)} {field}"
        for field in _SUMMARY_FIELD_ORDER
        if int(summary.get(field, 0) or 0) > 0
    ]
    if not parts:
        return None
    prefix = "test output so far" if active else "test output summary"
    return prefix + ": " + ", ".join(parts)


@dataclass(frozen=True)
class ParsedStepStatus:
    """Structured interpretation of one workflow step's output."""

    state: str
    progress_message: str
    summary_fields: dict[str, Any]
    success_detected: bool
    failure_detected: bool
    raw_evidence_lines: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""
        return {
            "state": self.state,
            "progress_message": self.progress_message,
            "summary_fields": dict(self.summary_fields),
            "success_detected": self.success_detected,
            "failure_detected": self.failure_detected,
            "raw_evidence_lines": list(self.raw_evidence_lines),
        }


def interpret_generic_step_output(
    *,
    raw_output: str,
    command: str,
    success: bool | None,
    active: bool,
) -> dict[str, Any] | None:
    """Interpret one step's output using generic parser-first heuristics."""
    del command  # reserved for future family-specific branching
    normalized_output = raw_output.strip()
    if not normalized_output:
        return None

    summary_fields: dict[str, Any] = {}
    progress_message: str | None = None
    success_detected = bool(success) if success is not None else False
    failure_detected = False

    try:
        parse_result = parse_interrupted_output(
            normalized_output,
            context=OutputContext(framework_hint=FrameworkHint.AUTO),
        )
    except Exception:
        parse_result = None

    if parse_result is not None and (
        parse_result.records
        or parse_result.framework_hint is not FrameworkHint.UNKNOWN
    ):
        summary_fields = {
            "passed": parse_result.passed_count,
            "failed": parse_result.failed_count,
            "skipped": parse_result.skipped_count,
            "error": parse_result.error_count,
            "incomplete": parse_result.incomplete_count,
            "framework": parse_result.framework_hint.value,
        }
        failing_tests = [
            record.name
            for record in parse_result.records
            if record.status in (TestStatus.FAILED, TestStatus.ERROR)
        ][:3]
        if failing_tests:
            summary_fields["failing_tests"] = failing_tests
        incomplete_tests = [
            record.name
            for record in parse_result.records
            if record.status is TestStatus.INCOMPLETE
        ][:3]
        if incomplete_tests:
            summary_fields["incomplete_tests"] = incomplete_tests

        progress_message = _format_summary_counts(
            {
                "passed": parse_result.passed_count,
                "failed": parse_result.failed_count,
                "skipped": parse_result.skipped_count,
                "error": parse_result.error_count,
                "incomplete": parse_result.incomplete_count,
            },
            active=active,
        )
        failure_detected = (
            parse_result.failed_count > 0 or parse_result.error_count > 0
        )
        success_detected = success_detected or (
            not active
            and not failure_detected
            and parse_result.incomplete_count == 0
            and parse_result.passed_count > 0
        )

    raw_evidence_lines = tuple(_last_meaningful_lines(normalized_output))
    if progress_message is None:
        progress_message = raw_evidence_lines[-1] if raw_evidence_lines else ""
    if success is False:
        failure_detected = True
    state = (
        "running"
        if active
        else (
            "completed_success"
            if success_detected and not failure_detected
            else "completed_failure" if failure_detected else "completed"
        )
    )

    return ParsedStepStatus(
        state=state,
        progress_message=progress_message,
        summary_fields=summary_fields,
        success_detected=success_detected,
        failure_detected=failure_detected,
        raw_evidence_lines=raw_evidence_lines,
    ).to_payload()
