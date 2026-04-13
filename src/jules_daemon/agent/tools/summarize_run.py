"""summarize_run tool -- produces human-readable test run summaries.

Accepts input in three modes:

1. **Structured test results** (from ``parse_test_output``): a JSON string
   with per-test records, counts, and framework info. This is the preferred
   path when the agent has already parsed raw output.

2. **Tool call history**: a JSON array of prior tool call result dicts from
   the current agent loop session. The tool scans the history for
   ``parse_test_output`` and ``execute_ssh`` results to synthesize a summary.

3. **Raw output** (backward-compatible): ``stdout``, ``stderr``, ``command``,
   ``exit_code`` -- delegates to the existing
   ``execution.output_summarizer.summarize_output`` regex+LLM pipeline.

All modes produce a unified JSON output with:
    - ``overall_status``: PASSED, FAILED, MIXED, ERROR, or NO_TESTS
    - ``summary_text``: Human-readable text summary
    - ``passed`` / ``failed`` / ``skipped`` / ``total``: counts
    - ``failure_highlights``: Up to 5 failing test details
    - ``suggested_next_actions``: Actionable follow-up recommendations

Delegates to:
    - jules_daemon.execution.output_summarizer.summarize_output (raw mode)

Usage::

    tool = SummarizeRunTool(llm_client=client, llm_model="gpt-4")

    # Mode 1: Structured test results
    result = await tool.execute({
        "test_results": '{"summary":{"passed":5,"failed":2,...},...}',
        "command": "pytest tests/",
    })

    # Mode 2: Tool call history
    result = await tool.execute({
        "tool_history": '[{"tool_name":"parse_test_output",...}]',
        "command": "pytest tests/",
    })

    # Mode 3: Raw output (backward-compatible)
    result = await tool.execute({
        "stdout": "...",
        "stderr": "...",
        "command": "pytest tests/",
        "exit_code": 0,
    })
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolParam,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)
from jules_daemon.agent.tools.base import BaseTool

__all__ = ["SummarizeRunTool"]

logger = logging.getLogger(__name__)

# Maximum number of failure highlights to include in the summary.
_MAX_FAILURE_HIGHLIGHTS: int = 5
_SUMMARY_FIELD_ORDER: tuple[str, ...] = (
    "passed",
    "failed",
    "skipped",
    "error",
    "incomplete",
)


# ---------------------------------------------------------------------------
# Overall status enum values
# ---------------------------------------------------------------------------

_STATUS_PASSED = "PASSED"
_STATUS_FAILED = "FAILED"
_STATUS_MIXED = "MIXED"
_STATUS_ERROR = "ERROR"
_STATUS_NO_TESTS = "NO_TESTS"


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _determine_overall_status(
    *,
    passed: int,
    failed: int,
    skipped: int,
    error: int,
    exit_code: int | None,
    has_ssh_error: bool,
) -> str:
    """Derive the overall run status from test counts and context.

    Rules (evaluated in order):
        1. SSH-level failure with no test results -> ERROR
        2. Errors present with no passes -> ERROR
        3. Failures present with passes -> MIXED
        4. Failures present with no passes -> FAILED
        5. Passes present (or only skipped) -> PASSED
        6. Non-zero exit code with no tests -> ERROR
        7. No tests detected at all -> NO_TESTS

    Args:
        passed: Number of passing tests.
        failed: Number of failing tests.
        skipped: Number of skipped tests.
        error: Number of errored tests.
        exit_code: Process exit code (None if unavailable).
        has_ssh_error: Whether an SSH-level error was encountered.

    Returns:
        One of the status string constants.
    """
    total_executed = passed + failed + error

    if has_ssh_error and total_executed == 0:
        return _STATUS_ERROR

    if error > 0 and passed == 0 and failed == 0:
        return _STATUS_ERROR

    if failed > 0 and passed > 0:
        return _STATUS_MIXED

    if failed > 0:
        return _STATUS_FAILED

    if error > 0 and passed > 0:
        return _STATUS_MIXED

    if passed > 0 or skipped > 0:
        return _STATUS_PASSED

    if exit_code is not None and exit_code != 0:
        return _STATUS_ERROR

    return _STATUS_NO_TESTS


def _extract_failure_highlights(
    records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Extract failure/error highlights from individual test records.

    Filters for records with status "failed" or "error", extracts the
    test name, module, and failure message, and caps at
    ``_MAX_FAILURE_HIGHLIGHTS``.

    Args:
        records: List of per-test record dicts from parse_test_output.

    Returns:
        List of highlight dicts, each with test_name, module, message keys.
    """
    highlights: list[dict[str, str]] = []
    for record in records:
        status = str(record.get("status", "")).lower()
        if status not in ("failed", "error"):
            continue
        test_name = str(record.get("name", "unknown"))
        module = str(record.get("module", ""))
        message = str(record.get("failure_message") or "No details available")
        highlights.append({
            "test_name": test_name,
            "module": module,
            "message": message,
        })
        if len(highlights) >= _MAX_FAILURE_HIGHLIGHTS:
            break
    return highlights


def _generate_next_actions(
    *,
    overall_status: str,
    failed: int,
    error: int,
    truncated: bool,
    has_ssh_error: bool,
    ssh_error_message: str,
    failure_highlights: list[dict[str, str]],
    command: str,
) -> list[str]:
    """Generate actionable follow-up recommendations.

    The recommendations are deterministic and based purely on the summary
    data -- no LLM is needed. Actions are ordered by priority.

    Args:
        overall_status: The derived overall status string.
        failed: Number of failing tests.
        error: Number of errored tests.
        truncated: Whether the output was truncated.
        has_ssh_error: Whether an SSH-level error occurred.
        ssh_error_message: SSH error message (empty string if none).
        failure_highlights: List of failure highlight dicts.
        command: The command that was executed.

    Returns:
        List of recommendation strings.
    """
    actions: list[str] = []

    if has_ssh_error:
        detail = f": {ssh_error_message}" if ssh_error_message else ""
        actions.append(f"Investigate SSH connection issue{detail}")

    if failed > 0:
        # Suggest rerunning failed tests with verbose output
        if failure_highlights:
            test_names = [h["test_name"] for h in failure_highlights[:3]]
            names_str = ", ".join(test_names)
            actions.append(
                f"Rerun the {failed} failed test(s) with verbose output "
                f"(failing: {names_str})"
            )
        else:
            actions.append(
                f"Rerun the {failed} failed test(s) with verbose output"
            )

    if error > 0:
        actions.append(
            f"Investigate the {error} test error(s) -- "
            "these may indicate setup/import issues rather than test logic failures"
        )

    if truncated:
        actions.append(
            "Output was truncated -- use read_output to view the full "
            "output for complete error context"
        )

    if overall_status == _STATUS_PASSED:
        actions.append("All tests passed -- no further action required")

    if overall_status == _STATUS_NO_TESTS:
        actions.append(
            "No test results detected -- verify the command produced "
            "test output or check the output format"
        )

    return actions


def _coerce_summary_fields(raw_value: Any) -> tuple[str, ...]:
    """Normalize optional summary field lists from structured inputs."""
    if raw_value is None:
        return ()
    if isinstance(raw_value, str) or not isinstance(raw_value, (list, tuple)):
        return ()

    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_value:
        if not isinstance(value, str) or not value.strip():
            continue
        field = value.strip()
        if field in seen:
            continue
        seen.add(field)
        normalized.append(field)
    return tuple(normalized)


def _coerce_named_counts(raw_value: Any) -> dict[str, int]:
    """Normalize an optional mapping of named summary counts."""
    if not isinstance(raw_value, dict):
        return {}

    counts: dict[str, int] = {}
    for key, value in raw_value.items():
        if not isinstance(key, str) or not key.strip():
            continue
        try:
            counts[key.strip()] = int(value)
        except (TypeError, ValueError):
            continue
    return counts


def _format_count_label(field: str) -> str:
    """Render count field names for the human-readable summary."""
    if field == "error":
        return "errors"
    return field.replace("_", " ")


def _build_summary_text(
    *,
    overall_status: str,
    passed: int,
    failed: int,
    skipped: int,
    error: int,
    total: int,
    framework: str,
    duration_seconds: float | None,
    failure_highlights: list[dict[str, str]],
    suggested_next_actions: list[str],
    command: str,
    narrative: str,
    focused_summary: dict[str, int] | None = None,
) -> str:
    """Build a human-readable summary text.

    Produces a structured text summary with sections for status, counts,
    failures, and next actions.

    Args:
        overall_status: The derived overall status.
        passed: Passing test count.
        failed: Failing test count.
        skipped: Skipped test count.
        error: Error test count.
        total: Total test count.
        framework: Test framework name.
        duration_seconds: Run duration (None if unknown).
        failure_highlights: Failure details.
        suggested_next_actions: Recommended actions.
        command: The executed command.
        narrative: Optional LLM-generated narrative.
        focused_summary: Optional ordered summary subset to prefer for the
            human-readable count line.

    Returns:
        Multi-line human-readable summary string.
    """
    lines: list[str] = []

    # Header
    lines.append(f"Run Summary: {overall_status}")
    lines.append(f"Command: {command}")
    if framework and framework != "unknown":
        lines.append(f"Framework: {framework}")

    # Counts
    parts: list[str] = []
    requested_counts = focused_summary or {}
    include_requested_zero_counts = any(
        value > 0 for value in requested_counts.values()
    )
    for field, value in requested_counts.items():
        if value > 0 or include_requested_zero_counts:
            parts.append(f"{value} {_format_count_label(field)}")

    for field, value in (
        ("passed", passed),
        ("failed", failed),
        ("skipped", skipped),
        ("error", error),
    ):
        if field in requested_counts:
            continue
        if value > 0:
            parts.append(f"{value} {_format_count_label(field)}")
    if parts:
        count_line = f"Tests: {', '.join(parts)} ({total} total)"
        lines.append(count_line)
    else:
        lines.append("Tests: none detected")

    if duration_seconds is not None:
        lines.append(f"Duration: {duration_seconds:.1f}s")

    # Narrative (from LLM or regex layer)
    if narrative:
        lines.append("")
        lines.append(narrative)

    # Failure highlights
    if failure_highlights:
        lines.append("")
        lines.append("Failure Highlights:")
        for i, highlight in enumerate(failure_highlights, 1):
            module_prefix = f"{highlight['module']}::" if highlight["module"] else ""
            lines.append(
                f"  {i}. {module_prefix}{highlight['test_name']} - "
                f"{highlight['message']}"
            )

    # Next actions
    if suggested_next_actions:
        lines.append("")
        lines.append("Suggested Next Actions:")
        for action in suggested_next_actions:
            lines.append(f"  - {action}")

    return "\n".join(lines)


def _parse_structured_results(
    test_results_json: str,
) -> dict[str, Any]:
    """Parse and validate a test_results JSON string.

    Expects the format produced by parse_test_output tool.

    Args:
        test_results_json: JSON string from parse_test_output.

    Returns:
        Parsed dict with normalized fields.

    Raises:
        ValueError: If the JSON is malformed or missing required structure.
    """
    try:
        data = json.loads(test_results_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid test_results JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("test_results must be a JSON object")

    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError("test_results.summary must be a JSON object")

    summary_fields = _coerce_summary_fields(data.get("summary_fields"))
    raw_focused_summary = _coerce_named_counts(data.get("focused_summary"))
    focused_summary: dict[str, int] = {}
    seen_focused_fields: set[str] = set()
    for field in summary_fields:
        if field in raw_focused_summary:
            focused_summary[field] = raw_focused_summary[field]
            seen_focused_fields.add(field)
        elif field in summary:
            focused_summary[field] = int(summary[field])
            seen_focused_fields.add(field)
    for field, value in raw_focused_summary.items():
        if field in seen_focused_fields:
            continue
        focused_summary[field] = value

    raw_unmapped_summary_fields = _coerce_summary_fields(
        data.get("unmapped_summary_fields"),
    )
    if raw_unmapped_summary_fields:
        unmapped_summary_fields = raw_unmapped_summary_fields
    else:
        unmapped_summary_fields = tuple(
            field
            for field in summary_fields
            if field not in focused_summary
            and field not in _SUMMARY_FIELD_ORDER
        )

    return {
        "records": data.get("records", []),
        "truncated": bool(data.get("truncated", False)),
        "framework": str(data.get("framework", "unknown")),
        "summary": {
            "passed": int(summary.get("passed", 0)),
            "failed": int(summary.get("failed", 0)),
            "skipped": int(summary.get("skipped", 0)),
            "error": int(summary.get("error", 0)),
            "incomplete": int(summary.get("incomplete", 0)),
        },
        "summary_fields": summary_fields,
        "focused_summary": focused_summary,
        "unmapped_summary_fields": unmapped_summary_fields,
    }


def _extract_from_tool_history(
    tool_history_json: str,
) -> dict[str, Any]:
    """Extract test results from accumulated tool call history.

    Scans the history for parse_test_output and execute_ssh results.
    If a parse_test_output result is found, its output is used as the
    primary data source. If only execute_ssh results are found, the
    tool checks for SSH errors.

    Args:
        tool_history_json: JSON array of tool call result dicts.

    Returns:
        Dict with extracted data, including ssh_error_message and
        optionally parsed test results.

    Raises:
        ValueError: If the JSON is malformed.
    """
    try:
        entries = json.loads(tool_history_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid tool_history JSON: {exc}") from exc

    if not isinstance(entries, list):
        raise ValueError("tool_history must be a JSON array")

    result: dict[str, Any] = {
        "has_ssh_error": False,
        "ssh_error_message": "",
        "parsed_results": None,
        "exit_code": None,
    }

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        tool_name = entry.get("tool_name", "")
        status = entry.get("status", "")
        output = entry.get("output", "")
        error_msg = entry.get("error_message")

        if tool_name == "execute_ssh":
            if status == "error":
                result["has_ssh_error"] = True
                result["ssh_error_message"] = error_msg or "SSH execution failed"
            elif output:
                try:
                    ssh_data = json.loads(output)
                    result["exit_code"] = ssh_data.get("exit_code")
                except (json.JSONDecodeError, TypeError):
                    pass

        if tool_name == "parse_test_output" and status == "success" and output:
            try:
                result["parsed_results"] = _parse_structured_results(output)
            except ValueError:
                logger.debug("Failed to parse parse_test_output from history")

    return result


# ---------------------------------------------------------------------------
# SummarizeRunTool
# ---------------------------------------------------------------------------


class SummarizeRunTool(BaseTool):
    """Produce a structured, human-readable summary of a test run.

    Accepts structured test results, tool call history, or raw output.
    Always produces a unified JSON output with overall status, failure
    highlights, and suggested next actions.

    Wraps:
        - execution.output_summarizer.summarize_output (raw mode fallback)

    This is a read-only tool (ApprovalRequirement.NONE).
    """

    _spec = ToolSpec(
        name="summarize_run",
        description=(
            "Summarize the output of a completed test run. Accepts structured "
            "test results (from parse_test_output), accumulated tool call "
            "history, or raw stdout/stderr. Returns overall status (PASSED/"
            "FAILED/MIXED/ERROR/NO_TESTS), failure highlights with test names "
            "and error messages, and suggested next actions. "
            "Use this after execute_ssh and parse_test_output to produce a "
            "human-readable report."
        ),
        parameters=(
            ToolParam(
                name="test_results",
                description=(
                    "JSON string from parse_test_output tool containing "
                    "structured per-test records and summary counts"
                ),
                json_type="string",
                required=False,
            ),
            ToolParam(
                name="tool_history",
                description=(
                    "JSON array of prior tool call result dicts from this "
                    "session. The tool scans for parse_test_output and "
                    "execute_ssh results to synthesize a summary"
                ),
                json_type="string",
                required=False,
            ),
            ToolParam(
                name="stdout",
                description="Standard output from the command (raw mode)",
                json_type="string",
                required=False,
                default="",
            ),
            ToolParam(
                name="stderr",
                description="Standard error from the command (raw mode)",
                json_type="string",
                required=False,
                default="",
            ),
            ToolParam(
                name="command",
                description="The shell command that produced the output",
                json_type="string",
            ),
            ToolParam(
                name="exit_code",
                description="Process exit code (0 = success)",
                json_type="integer",
                required=False,
            ),
        ),
        approval=ApprovalRequirement.NONE,
    )

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        llm_model: str | None = None,
        wiki_root: Path | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._llm_model = llm_model
        self._wiki_root = wiki_root

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Summarize a test run using the best available input mode.

        Priority order:
        1. ``test_results`` (structured JSON from parse_test_output)
        2. ``tool_history`` (accumulated tool call history)
        3. ``stdout``/``stderr``/``exit_code`` (raw output, backward-compat)

        Returns:
            ToolResult with JSON output containing overall_status,
            summary_text, counts, failure_highlights, and
            suggested_next_actions. Never raises.
        """
        command = args.get("command", "")
        call_id = args.get("_call_id", "summarize_run")

        if not command:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message="command parameter is required",
            )

        test_results_json = args.get("test_results")
        tool_history_json = args.get("tool_history")

        # Mode 1: Structured test results from parse_test_output
        if test_results_json:
            return await self._summarize_structured(
                test_results_json=test_results_json,
                command=command,
                call_id=call_id,
                exit_code=args.get("exit_code"),
            )

        # Mode 2: Tool call history
        if tool_history_json:
            return await self._summarize_from_history(
                tool_history_json=tool_history_json,
                command=command,
                call_id=call_id,
                exit_code=args.get("exit_code"),
            )

        # Mode 3: Raw output (backward-compatible)
        return await self._summarize_raw(
            stdout=args.get("stdout", ""),
            stderr=args.get("stderr", ""),
            command=command,
            exit_code=args.get("exit_code"),
            call_id=call_id,
        )

    # -- Mode 1: Structured test results ------------------------------------

    async def _summarize_structured(
        self,
        *,
        test_results_json: str,
        command: str,
        call_id: str,
        exit_code: int | None,
    ) -> ToolResult:
        """Build summary from structured parse_test_output results."""
        try:
            parsed = _parse_structured_results(test_results_json)
        except ValueError as exc:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=str(exc),
            )

        summary = parsed["summary"]
        records = parsed.get("records", [])
        framework = parsed.get("framework", "unknown")
        truncated = parsed.get("truncated", False)

        return self._build_output(
            call_id=call_id,
            command=command,
            passed=summary["passed"],
            failed=summary["failed"],
            skipped=summary["skipped"],
            error=summary["error"],
            exit_code=exit_code,
            framework=framework,
            duration_seconds=None,
            records=records,
            truncated=truncated,
            has_ssh_error=False,
            ssh_error_message="",
            narrative="",
            summary_fields=parsed.get("summary_fields", ()),
            focused_summary=parsed.get("focused_summary", {}),
            unmapped_summary_fields=parsed.get(
                "unmapped_summary_fields",
                (),
            ),
        )

    # -- Mode 2: Tool call history ------------------------------------------

    async def _summarize_from_history(
        self,
        *,
        tool_history_json: str,
        command: str,
        call_id: str,
        exit_code: int | None,
    ) -> ToolResult:
        """Build summary by scanning accumulated tool call history."""
        try:
            extracted = _extract_from_tool_history(tool_history_json)
        except ValueError as exc:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=str(exc),
            )

        has_ssh_error = extracted["has_ssh_error"]
        ssh_error_message = extracted["ssh_error_message"]
        parsed_results = extracted["parsed_results"]
        history_exit_code = extracted.get("exit_code")
        effective_exit_code = exit_code if exit_code is not None else history_exit_code

        if parsed_results is not None:
            summary = parsed_results["summary"]
            return self._build_output(
                call_id=call_id,
                command=command,
                passed=summary["passed"],
                failed=summary["failed"],
                skipped=summary["skipped"],
                error=summary["error"],
                exit_code=effective_exit_code,
                framework=parsed_results.get("framework", "unknown"),
                duration_seconds=None,
                records=parsed_results.get("records", []),
                truncated=parsed_results.get("truncated", False),
                has_ssh_error=has_ssh_error,
                ssh_error_message=ssh_error_message,
                narrative="",
                summary_fields=parsed_results.get("summary_fields", ()),
                focused_summary=parsed_results.get("focused_summary", {}),
                unmapped_summary_fields=parsed_results.get(
                    "unmapped_summary_fields",
                    (),
                ),
            )

        # No parse_test_output in history -- produce a minimal summary
        return self._build_output(
            call_id=call_id,
            command=command,
            passed=0,
            failed=0,
            skipped=0,
            error=0,
            exit_code=effective_exit_code,
            framework="unknown",
            duration_seconds=None,
            records=[],
            truncated=False,
            has_ssh_error=has_ssh_error,
            ssh_error_message=ssh_error_message,
            narrative="",
        )

    # -- Mode 3: Raw output (backward-compatible) ---------------------------

    async def _summarize_raw(
        self,
        *,
        stdout: str,
        stderr: str,
        command: str,
        exit_code: int | None,
        call_id: str,
    ) -> ToolResult:
        """Build summary from raw stdout/stderr via output_summarizer."""
        from jules_daemon.execution.output_summarizer import (
            summarize_output,
        )

        wiki_context = ""
        if self._wiki_root is not None:
            wiki_context = self._load_wiki_context(command)

        try:
            summary = await summarize_output(
                stdout=stdout,
                stderr=stderr,
                command=command,
                exit_code=exit_code,
                llm_client=self._llm_client,
                llm_model=self._llm_model,
                wiki_context=wiki_context,
            )
        except (OSError, ValueError, TimeoutError) as exc:
            logger.warning("summarize_run raw mode failed: %s", exc)
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                status=ToolResultStatus.ERROR,
                output="",
                error_message=f"Summarization failed: {exc}",
            )

        # Convert OutputSummary into our enhanced format
        return self._build_output(
            call_id=call_id,
            command=command,
            passed=summary.passed,
            failed=summary.failed,
            skipped=summary.skipped,
            error=0,
            exit_code=exit_code,
            framework=summary.parser,
            duration_seconds=summary.duration_seconds,
            records=[],
            truncated=False,
            has_ssh_error=False,
            ssh_error_message="",
            narrative=summary.narrative,
            key_failures=list(summary.key_failures),
            raw_excerpt=summary.raw_excerpt,
        )

    # -- Shared output builder -----------------------------------------------

    def _build_output(
        self,
        *,
        call_id: str,
        command: str,
        passed: int,
        failed: int,
        skipped: int,
        error: int,
        exit_code: int | None,
        framework: str,
        duration_seconds: float | None,
        records: list[dict[str, Any]],
        truncated: bool,
        has_ssh_error: bool,
        ssh_error_message: str,
        narrative: str,
        key_failures: list[str] | None = None,
        raw_excerpt: str = "",
        summary_fields: tuple[str, ...] = (),
        focused_summary: dict[str, int] | None = None,
        unmapped_summary_fields: tuple[str, ...] = (),
    ) -> ToolResult:
        """Build the unified output JSON from normalized inputs.

        Computes overall_status, extracts failure_highlights, generates
        suggested_next_actions, and builds the human-readable summary_text.

        Returns:
            ToolResult with SUCCESS status and JSON output.
        """
        total = passed + failed + skipped + error

        overall_status = _determine_overall_status(
            passed=passed,
            failed=failed,
            skipped=skipped,
            error=error,
            exit_code=exit_code,
            has_ssh_error=has_ssh_error,
        )

        failure_highlights = _extract_failure_highlights(records)

        # If no per-test records but we have key_failures from output_summarizer,
        # convert those into highlight entries
        if not failure_highlights and key_failures:
            failure_highlights = [
                {
                    "test_name": f"failure_{i + 1}",
                    "module": "",
                    "message": msg,
                }
                for i, msg in enumerate(key_failures[:_MAX_FAILURE_HIGHLIGHTS])
            ]

        suggested_next_actions = _generate_next_actions(
            overall_status=overall_status,
            failed=failed,
            error=error,
            truncated=truncated,
            has_ssh_error=has_ssh_error,
            ssh_error_message=ssh_error_message,
            failure_highlights=failure_highlights,
            command=command,
        )

        summary_text = _build_summary_text(
            overall_status=overall_status,
            passed=passed,
            failed=failed,
            skipped=skipped,
            error=error,
            total=total,
            focused_summary=focused_summary,
            framework=framework,
            duration_seconds=duration_seconds,
            failure_highlights=failure_highlights,
            suggested_next_actions=suggested_next_actions,
            command=command,
            narrative=narrative,
        )

        result_data: dict[str, Any] = {
            "overall_status": overall_status,
            "summary_text": summary_text,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": total,
            "framework": framework,
            "duration_seconds": duration_seconds,
            "failure_highlights": failure_highlights,
            "suggested_next_actions": suggested_next_actions,
            "narrative": narrative,
            "raw_excerpt": raw_excerpt,
        }
        if summary_fields:
            result_data["summary_fields"] = list(summary_fields)
        if focused_summary:
            result_data["focused_summary"] = focused_summary
        if unmapped_summary_fields:
            result_data["unmapped_summary_fields"] = list(
                unmapped_summary_fields
            )

        return ToolResult(
            call_id=call_id,
            tool_name=self.name,
            status=ToolResultStatus.SUCCESS,
            output=json.dumps(result_data),
        )

    # -- Wiki context loader (unchanged from v1.2) ---------------------------

    def _load_wiki_context(self, command: str) -> str:
        """Load test knowledge context from wiki for richer summaries.

        Delegates to wiki.test_knowledge -- fails silently so the
        summarizer can still produce regex-only results.
        """
        try:
            from jules_daemon.wiki.test_knowledge import (
                derive_test_slug,
                load_test_knowledge,
            )

            if self._wiki_root is None:
                return ""
            slug = derive_test_slug(command)
            knowledge = load_test_knowledge(self._wiki_root, slug)
            if knowledge is not None:
                return knowledge.to_prompt_context()
        except Exception as exc:
            logger.debug("Failed to load wiki context for summarizer: %s", exc)
        return ""
