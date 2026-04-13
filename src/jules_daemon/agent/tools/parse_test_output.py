"""parse_test_output tool -- wraps monitor.test_output_parser.

Parses raw test output (stdout/stderr) into structured per-test records.
Detects test framework (pytest, jest, go test), extracts pass/fail/skip
counts, failure messages, and identifies incomplete tests from truncated
output.

Supported frameworks:
    - pytest (verbose and short output)
    - Jest (JavaScript/TypeScript)
    - go test (Go standard testing)

Delegates to:
    - jules_daemon.monitor.test_output_parser.parse_interrupted_output

Usage::

    tool = ParseTestOutputTool()
    result = await tool.execute({
        "raw_output": "tests/test_auth.py::test_login PASSED\\n...",
        "framework_hint": "auto",
    })
"""

from __future__ import annotations

import json
import logging
from typing import Any

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolParam,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)
from jules_daemon.agent.tools.base import BaseTool

__all__ = ["ParseTestOutputTool"]

logger = logging.getLogger(__name__)

_SUMMARY_FIELD_ORDER: tuple[str, ...] = (
    "passed",
    "failed",
    "skipped",
    "error",
    "incomplete",
)


def _empty_summary() -> dict[str, int]:
    """Return the canonical empty test summary shape."""
    return {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "error": 0,
        "incomplete": 0,
    }


def _coerce_summary_fields(raw_value: Any) -> tuple[str, ...]:
    """Validate and normalize optional summary_fields input."""
    if raw_value is None:
        return ()
    if isinstance(raw_value, str) or not isinstance(raw_value, (list, tuple)):
        raise ValueError("summary_fields must be an array of non-empty strings")

    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_value:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "summary_fields must be an array of non-empty strings"
            )
        field = value.strip()
        if field in seen:
            continue
        seen.add(field)
        normalized.append(field)
    return tuple(normalized)


def _build_result_data(
    *,
    records_data: list[dict[str, Any]],
    truncated: bool,
    framework: str,
    total_lines_parsed: int,
    summary: dict[str, int],
    summary_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Build the JSON-serializable tool result payload."""
    result_data: dict[str, Any] = {
        "records": records_data,
        "truncated": truncated,
        "framework": framework,
        "total_lines_parsed": total_lines_parsed,
        "summary": summary,
    }

    if not summary_fields:
        return result_data

    focused_summary: dict[str, int] = {}
    unmapped_summary_fields: list[str] = []
    for field in summary_fields:
        if field in _SUMMARY_FIELD_ORDER:
            focused_summary[field] = int(summary.get(field, 0) or 0)
        else:
            unmapped_summary_fields.append(field)

    result_data["summary_fields"] = list(summary_fields)
    if focused_summary:
        result_data["focused_summary"] = focused_summary
    if unmapped_summary_fields:
        result_data["unmapped_summary_fields"] = unmapped_summary_fields
    return result_data


class ParseTestOutputTool(BaseTool):
    """Parse raw test output into structured records.

    Wraps:
        - monitor.test_output_parser.parse_interrupted_output

    This is a read-only tool (ApprovalRequirement.NONE).
    Pure function -- no I/O, no side effects.

    Output JSON structure::

        {
            "records": [
                {
                    "name": "test_login",
                    "status": "passed",
                    "module": "tests/test_auth.py",
                    "failure_message": null,
                    "duration_seconds": null
                }
            ],
            "truncated": false,
            "framework": "pytest",
            "total_lines_parsed": 10,
            "summary": {
                "passed": 5,
                "failed": 1,
                "skipped": 0,
                "error": 0,
                "incomplete": 0
            },
            "focused_summary": {
                "failed": 1,
                "passed": 5
            },
            "unmapped_summary_fields": ["iterations_done"]
        }
    """

    _spec = ToolSpec(
        name="parse_test_output",
        description=(
            "Parse raw test output (stdout/stderr) into structured per-test "
            "records. Detects the test framework (pytest, jest, go test), "
            "extracts pass/fail/skip/error counts, individual test names, "
            "failure messages, and error details. "
            "Use this to analyze test results after execution."
        ),
        parameters=(
            ToolParam(
                name="raw_output",
                description="Raw test output text (stdout and/or stderr)",
                json_type="string",
            ),
            ToolParam(
                name="framework_hint",
                description=(
                    "Test framework hint: 'auto' (detect from output), "
                    "'pytest', 'jest', 'go_test', or 'unknown'"
                ),
                json_type="string",
                required=False,
                default="auto",
                enum=("auto", "pytest", "jest", "go_test", "unknown"),
            ),
            ToolParam(
                name="summary_fields",
                description=(
                    "Optional ordered list of summary fields from "
                    "lookup_test_spec. Known fields are returned in "
                    "focused_summary; unknown fields are reported in "
                    "unmapped_summary_fields."
                ),
                json_type="array",
                required=False,
            ),
        ),
        approval=ApprovalRequirement.NONE,
    )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Parse test output into structured records.

        This is a pure function with no I/O, so it runs synchronously
        in the event loop (no thread pool needed).

        Args:
            args: Dict with 'raw_output' (required), 'framework_hint'
                (optional, default 'auto'), and '_call_id' (injected
                by ToolRegistry).

        Returns:
            ToolResult with JSON output containing records and summary.
        """
        raw_output = args.get("raw_output", "")
        framework_hint = args.get("framework_hint", "auto")
        try:
            summary_fields = _coerce_summary_fields(args.get("summary_fields"))
        except ValueError as exc:
            return ToolResult(
                call_id=args.get("_call_id", "parse_test_output"),
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=str(exc),
            )
        call_id = args.get("_call_id", "parse_test_output")

        if not raw_output:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.SUCCESS,
                output=json.dumps(_build_result_data(
                    records_data=[],
                    truncated=False,
                    framework="unknown",
                    total_lines_parsed=0,
                    summary=_empty_summary(),
                    summary_fields=summary_fields,
                )),
            )

        try:
            from jules_daemon.monitor.test_output_parser import (
                FrameworkHint,
                OutputContext,
                parse_interrupted_output,
            )

            hint_map = {
                "auto": FrameworkHint.AUTO,
                "pytest": FrameworkHint.PYTEST,
                "jest": FrameworkHint.JEST,
                "go_test": FrameworkHint.GO_TEST,
                "unknown": FrameworkHint.UNKNOWN,
            }
            hint = hint_map.get(framework_hint, FrameworkHint.AUTO)
            context = OutputContext(framework_hint=hint)

            parse_result = parse_interrupted_output(raw_output, context=context)

            records_data = [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "module": r.module,
                    "failure_message": r.failure_message,
                    "duration_seconds": r.duration_seconds,
                }
                for r in parse_result.records
            ]

            result_data = _build_result_data(
                records_data=records_data,
                truncated=parse_result.truncated,
                framework=parse_result.framework_hint.value,
                total_lines_parsed=parse_result.total_lines_parsed,
                summary={
                    "passed": parse_result.passed_count,
                    "failed": parse_result.failed_count,
                    "skipped": parse_result.skipped_count,
                    "error": parse_result.error_count,
                    "incomplete": parse_result.incomplete_count,
                },
                summary_fields=summary_fields,
            )

            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.SUCCESS,
                output=json.dumps(result_data),
            )
        except Exception as exc:
            logger.warning("parse_test_output failed: %s", exc)
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=f"Failed to parse test output: {exc}",
            )
