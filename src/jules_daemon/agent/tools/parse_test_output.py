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
            }
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
        call_id = args.get("_call_id", "parse_test_output")

        if not raw_output:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.SUCCESS,
                output=json.dumps({
                    "records": [],
                    "truncated": False,
                    "framework": "unknown",
                    "total_lines_parsed": 0,
                    "summary": {
                        "passed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "error": 0,
                        "incomplete": 0,
                    },
                }),
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

            result_data = {
                "records": records_data,
                "truncated": parse_result.truncated,
                "framework": parse_result.framework_hint.value,
                "total_lines_parsed": parse_result.total_lines_parsed,
                "summary": {
                    "passed": parse_result.passed_count,
                    "failed": parse_result.failed_count,
                    "skipped": parse_result.skipped_count,
                    "error": parse_result.error_count,
                    "incomplete": parse_result.incomplete_count,
                },
            }

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
