"""Tests for the enhanced summarize_run tool.

Verifies the summarize_run tool produces human-readable summaries with:
- Overall status (PASSED, FAILED, MIXED, ERROR, NO_TESTS)
- Failure highlights from structured test results
- Suggested next actions based on observed outcomes
- Support for structured test_results input (from parse_test_output)
- Support for accumulated tool_history input
- Backward-compatible raw stdout/stderr/command mode

Coverage includes:
- Pure helper functions tested directly
- Aggregation logic (count totals, combined metrics)
- Status determination (all branches of priority rules)
- Edge cases (empty input, missing keys, malformed JSON, boundary values)
- Integration paths through SummarizeRunTool.execute()
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from jules_daemon.agent.tool_types import ToolResultStatus
from jules_daemon.agent.tools.summarize_run import (
    SummarizeRunTool,
    _build_summary_text,
    _determine_overall_status,
    _extract_failure_highlights,
    _extract_from_tool_history,
    _generate_next_actions,
    _parse_structured_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_test_results(
    *,
    passed: int = 0,
    failed: int = 0,
    skipped: int = 0,
    error: int = 0,
    incomplete: int = 0,
    framework: str = "pytest",
    records: list[dict[str, Any]] | None = None,
    truncated: bool = False,
) -> str:
    """Build a parse_test_output JSON string for testing."""
    if records is None:
        records = []
    return json.dumps({
        "records": records,
        "truncated": truncated,
        "framework": framework,
        "total_lines_parsed": 10,
        "summary": {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "error": error,
            "incomplete": incomplete,
        },
    })


def _tool_history_entry(
    *,
    tool_name: str,
    status: str = "success",
    output: str = "",
    error_message: str | None = None,
) -> dict[str, Any]:
    """Build a single tool history entry for testing."""
    return {
        "call_id": f"call_{tool_name}",
        "tool_name": tool_name,
        "status": status,
        "output": output,
        "error_message": error_message,
    }


# ---------------------------------------------------------------------------
# _determine_overall_status -- direct unit tests for all branches
# ---------------------------------------------------------------------------


class TestDetermineOverallStatus:
    """Direct unit tests for the _determine_overall_status helper.

    Tests every documented rule branch and edge case.
    """

    def test_ssh_error_no_tests_returns_error(self) -> None:
        """Rule 1: SSH error with zero executed tests -> ERROR."""
        result = _determine_overall_status(
            passed=0, failed=0, skipped=0, error=0,
            exit_code=None, has_ssh_error=True,
        )
        assert result == "ERROR"

    def test_ssh_error_with_tests_still_uses_test_logic(self) -> None:
        """SSH error present but tests ran -> should use test-based logic."""
        result = _determine_overall_status(
            passed=5, failed=0, skipped=0, error=0,
            exit_code=0, has_ssh_error=True,
        )
        # Tests ran and passed, so despite SSH error flag, test logic applies
        assert result == "PASSED"

    def test_errors_only_returns_error(self) -> None:
        """Rule 2: Only errors, no passes or fails -> ERROR."""
        result = _determine_overall_status(
            passed=0, failed=0, skipped=0, error=3,
            exit_code=1, has_ssh_error=False,
        )
        assert result == "ERROR"

    def test_failed_and_passed_returns_mixed(self) -> None:
        """Rule 3: Some fail + some pass -> MIXED."""
        result = _determine_overall_status(
            passed=5, failed=2, skipped=0, error=0,
            exit_code=1, has_ssh_error=False,
        )
        assert result == "MIXED"

    def test_failed_only_returns_failed(self) -> None:
        """Rule 4: Failures but no passes -> FAILED."""
        result = _determine_overall_status(
            passed=0, failed=5, skipped=0, error=0,
            exit_code=1, has_ssh_error=False,
        )
        assert result == "FAILED"

    def test_errors_and_passes_returns_mixed(self) -> None:
        """Rule (line 137-138): Errors + passes with no fails -> MIXED."""
        result = _determine_overall_status(
            passed=3, failed=0, skipped=0, error=2,
            exit_code=1, has_ssh_error=False,
        )
        assert result == "MIXED"

    def test_passed_only_returns_passed(self) -> None:
        """Rule 5: Only passes -> PASSED."""
        result = _determine_overall_status(
            passed=10, failed=0, skipped=0, error=0,
            exit_code=0, has_ssh_error=False,
        )
        assert result == "PASSED"

    def test_skipped_only_returns_passed(self) -> None:
        """Rule 5: Only skipped -> PASSED."""
        result = _determine_overall_status(
            passed=0, failed=0, skipped=5, error=0,
            exit_code=0, has_ssh_error=False,
        )
        assert result == "PASSED"

    def test_passed_and_skipped_returns_passed(self) -> None:
        """Rule 5: Passes + skipped -> PASSED."""
        result = _determine_overall_status(
            passed=7, failed=0, skipped=3, error=0,
            exit_code=0, has_ssh_error=False,
        )
        assert result == "PASSED"

    def test_nonzero_exit_no_tests_returns_error(self) -> None:
        """Rule 6: Non-zero exit with no test results -> ERROR."""
        result = _determine_overall_status(
            passed=0, failed=0, skipped=0, error=0,
            exit_code=2, has_ssh_error=False,
        )
        assert result == "ERROR"

    def test_zero_exit_no_tests_returns_no_tests(self) -> None:
        """Rule 7: Zero exit with no tests -> NO_TESTS."""
        result = _determine_overall_status(
            passed=0, failed=0, skipped=0, error=0,
            exit_code=0, has_ssh_error=False,
        )
        assert result == "NO_TESTS"

    def test_none_exit_no_tests_returns_no_tests(self) -> None:
        """Rule 7: None exit code with no tests -> NO_TESTS."""
        result = _determine_overall_status(
            passed=0, failed=0, skipped=0, error=0,
            exit_code=None, has_ssh_error=False,
        )
        assert result == "NO_TESTS"

    def test_failed_and_errors_and_passed_returns_mixed(self) -> None:
        """All three: fails + errors + passes -> MIXED (fails+passes rule)."""
        result = _determine_overall_status(
            passed=3, failed=2, skipped=0, error=1,
            exit_code=1, has_ssh_error=False,
        )
        assert result == "MIXED"

    def test_failed_and_errors_no_passed_returns_failed(self) -> None:
        """Fails + errors but no passes -> FAILED (fails without passes)."""
        result = _determine_overall_status(
            passed=0, failed=5, skipped=0, error=2,
            exit_code=1, has_ssh_error=False,
        )
        assert result == "FAILED"

    def test_failed_with_skipped_no_passed_returns_failed(self) -> None:
        """Fails + skipped but no passes -> FAILED."""
        result = _determine_overall_status(
            passed=0, failed=3, skipped=5, error=0,
            exit_code=1, has_ssh_error=False,
        )
        assert result == "FAILED"


# ---------------------------------------------------------------------------
# _extract_failure_highlights -- direct unit tests
# ---------------------------------------------------------------------------


class TestExtractFailureHighlights:
    """Direct unit tests for _extract_failure_highlights helper."""

    def test_empty_records_returns_empty(self) -> None:
        result = _extract_failure_highlights([])
        assert result == []

    def test_no_failures_returns_empty(self) -> None:
        records = [
            {"name": "test_ok", "status": "passed", "module": "m.py"},
        ]
        result = _extract_failure_highlights(records)
        assert result == []

    def test_extracts_failed_records(self) -> None:
        records = [
            {
                "name": "test_a",
                "status": "failed",
                "module": "m.py",
                "failure_message": "assert False",
            },
        ]
        result = _extract_failure_highlights(records)
        assert len(result) == 1
        assert result[0]["test_name"] == "test_a"
        assert result[0]["module"] == "m.py"
        assert result[0]["message"] == "assert False"

    def test_extracts_error_records(self) -> None:
        records = [
            {
                "name": "test_err",
                "status": "error",
                "module": "m.py",
                "failure_message": "ImportError",
            },
        ]
        result = _extract_failure_highlights(records)
        assert len(result) == 1
        assert result[0]["test_name"] == "test_err"

    def test_caps_at_five(self) -> None:
        records = [
            {
                "name": f"test_{i}",
                "status": "failed",
                "module": "m.py",
                "failure_message": f"Error {i}",
            }
            for i in range(10)
        ]
        result = _extract_failure_highlights(records)
        assert len(result) == 5

    def test_exactly_five_records_all_included(self) -> None:
        records = [
            {
                "name": f"test_{i}",
                "status": "failed",
                "module": "m.py",
                "failure_message": f"Error {i}",
            }
            for i in range(5)
        ]
        result = _extract_failure_highlights(records)
        assert len(result) == 5

    def test_missing_name_defaults_to_unknown(self) -> None:
        records = [
            {"status": "failed", "module": "m.py", "failure_message": "oops"},
        ]
        result = _extract_failure_highlights(records)
        assert result[0]["test_name"] == "unknown"

    def test_missing_failure_message_defaults(self) -> None:
        records = [
            {"name": "test_x", "status": "failed", "module": "m.py"},
        ]
        result = _extract_failure_highlights(records)
        assert result[0]["message"] == "No details available"

    def test_none_failure_message_defaults(self) -> None:
        records = [
            {
                "name": "test_x",
                "status": "failed",
                "module": "m.py",
                "failure_message": None,
            },
        ]
        result = _extract_failure_highlights(records)
        assert result[0]["message"] == "No details available"

    def test_missing_module_defaults_to_empty(self) -> None:
        records = [
            {"name": "test_x", "status": "failed", "failure_message": "err"},
        ]
        result = _extract_failure_highlights(records)
        assert result[0]["module"] == ""

    def test_mixed_statuses_only_extracts_failures(self) -> None:
        records = [
            {"name": "pass_1", "status": "passed", "module": "m.py"},
            {"name": "fail_1", "status": "failed", "module": "m.py",
             "failure_message": "x"},
            {"name": "skip_1", "status": "skipped", "module": "m.py"},
            {"name": "err_1", "status": "error", "module": "m.py",
             "failure_message": "y"},
        ]
        result = _extract_failure_highlights(records)
        assert len(result) == 2
        names = [h["test_name"] for h in result]
        assert "fail_1" in names
        assert "err_1" in names

    def test_case_insensitive_status_matching(self) -> None:
        records = [
            {"name": "test_x", "status": "FAILED", "module": "m.py",
             "failure_message": "err"},
        ]
        result = _extract_failure_highlights(records)
        assert len(result) == 1

    def test_missing_status_skips_record(self) -> None:
        records = [{"name": "test_x", "module": "m.py"}]
        result = _extract_failure_highlights(records)
        assert result == []


# ---------------------------------------------------------------------------
# _generate_next_actions -- direct unit tests
# ---------------------------------------------------------------------------


class TestGenerateNextActions:
    """Direct unit tests for _generate_next_actions helper."""

    def test_ssh_error_action(self) -> None:
        actions = _generate_next_actions(
            overall_status="ERROR",
            failed=0, error=0, truncated=False,
            has_ssh_error=True,
            ssh_error_message="Connection refused",
            failure_highlights=[], command="pytest",
        )
        assert any("SSH" in a and "Connection refused" in a for a in actions)

    def test_ssh_error_without_message(self) -> None:
        actions = _generate_next_actions(
            overall_status="ERROR",
            failed=0, error=0, truncated=False,
            has_ssh_error=True,
            ssh_error_message="",
            failure_highlights=[], command="pytest",
        )
        assert any("SSH" in a for a in actions)

    def test_failed_with_highlights_includes_names(self) -> None:
        highlights = [
            {"test_name": "test_login", "module": "m.py", "message": "err"},
            {"test_name": "test_logout", "module": "m.py", "message": "err"},
        ]
        actions = _generate_next_actions(
            overall_status="FAILED",
            failed=2, error=0, truncated=False,
            has_ssh_error=False, ssh_error_message="",
            failure_highlights=highlights, command="pytest",
        )
        rerun = [a for a in actions if "erun" in a.lower()]
        assert len(rerun) >= 1
        assert "test_login" in rerun[0]

    def test_failed_without_highlights(self) -> None:
        actions = _generate_next_actions(
            overall_status="FAILED",
            failed=3, error=0, truncated=False,
            has_ssh_error=False, ssh_error_message="",
            failure_highlights=[], command="pytest",
        )
        rerun = [a for a in actions if "erun" in a.lower()]
        assert len(rerun) >= 1
        assert "3" in rerun[0]

    def test_error_action_mentions_setup(self) -> None:
        actions = _generate_next_actions(
            overall_status="ERROR",
            failed=0, error=2, truncated=False,
            has_ssh_error=False, ssh_error_message="",
            failure_highlights=[], command="pytest",
        )
        assert any("setup" in a.lower() or "import" in a.lower() for a in actions)

    def test_truncated_action(self) -> None:
        actions = _generate_next_actions(
            overall_status="MIXED",
            failed=1, error=0, truncated=True,
            has_ssh_error=False, ssh_error_message="",
            failure_highlights=[], command="pytest",
        )
        assert any("truncat" in a.lower() for a in actions)

    def test_passed_suggests_no_action(self) -> None:
        actions = _generate_next_actions(
            overall_status="PASSED",
            failed=0, error=0, truncated=False,
            has_ssh_error=False, ssh_error_message="",
            failure_highlights=[], command="pytest",
        )
        assert any("no further action" in a.lower() for a in actions)

    def test_no_tests_suggests_verification(self) -> None:
        actions = _generate_next_actions(
            overall_status="NO_TESTS",
            failed=0, error=0, truncated=False,
            has_ssh_error=False, ssh_error_message="",
            failure_highlights=[], command="pytest",
        )
        assert any("no test results" in a.lower() for a in actions)

    def test_multiple_conditions_combined(self) -> None:
        """When SSH error + failed + truncated, all relevant actions appear."""
        highlights = [
            {"test_name": "test_x", "module": "", "message": "e"},
        ]
        actions = _generate_next_actions(
            overall_status="ERROR",
            failed=2, error=1, truncated=True,
            has_ssh_error=True, ssh_error_message="timeout",
            failure_highlights=highlights, command="pytest",
        )
        assert any("SSH" in a for a in actions)
        assert any("erun" in a.lower() for a in actions)
        assert any("truncat" in a.lower() for a in actions)
        assert any("error" in a.lower() for a in actions)

    def test_failure_highlights_limited_to_three_names(self) -> None:
        """Rerun message shows at most 3 test names from highlights."""
        highlights = [
            {"test_name": f"test_{i}", "module": "", "message": "e"}
            for i in range(5)
        ]
        actions = _generate_next_actions(
            overall_status="FAILED",
            failed=5, error=0, truncated=False,
            has_ssh_error=False, ssh_error_message="",
            failure_highlights=highlights, command="pytest",
        )
        rerun = [a for a in actions if "erun" in a.lower()]
        assert len(rerun) >= 1
        # Only first 3 names appear
        assert "test_0" in rerun[0]
        assert "test_2" in rerun[0]
        # 4th should not appear in the names list
        assert "test_3" not in rerun[0]


# ---------------------------------------------------------------------------
# _build_summary_text -- direct unit tests
# ---------------------------------------------------------------------------


class TestBuildSummaryText:
    """Direct unit tests for _build_summary_text helper."""

    def test_header_includes_status_and_command(self) -> None:
        text = _build_summary_text(
            overall_status="PASSED", passed=5, failed=0, skipped=0,
            error=0, total=5, framework="pytest", duration_seconds=None,
            failure_highlights=[], suggested_next_actions=[],
            command="pytest tests/", narrative="",
        )
        assert "Run Summary: PASSED" in text
        assert "Command: pytest tests/" in text

    def test_framework_shown_when_known(self) -> None:
        text = _build_summary_text(
            overall_status="PASSED", passed=5, failed=0, skipped=0,
            error=0, total=5, framework="jest", duration_seconds=None,
            failure_highlights=[], suggested_next_actions=[],
            command="npm test", narrative="",
        )
        assert "Framework: jest" in text

    def test_framework_hidden_when_unknown(self) -> None:
        text = _build_summary_text(
            overall_status="PASSED", passed=5, failed=0, skipped=0,
            error=0, total=5, framework="unknown", duration_seconds=None,
            failure_highlights=[], suggested_next_actions=[],
            command="pytest tests/", narrative="",
        )
        assert "Framework:" not in text

    def test_duration_shown_when_present(self) -> None:
        text = _build_summary_text(
            overall_status="PASSED", passed=5, failed=0, skipped=0,
            error=0, total=5, framework="pytest", duration_seconds=12.345,
            failure_highlights=[], suggested_next_actions=[],
            command="pytest tests/", narrative="",
        )
        assert "Duration: 12.3s" in text

    def test_duration_hidden_when_none(self) -> None:
        text = _build_summary_text(
            overall_status="PASSED", passed=5, failed=0, skipped=0,
            error=0, total=5, framework="pytest", duration_seconds=None,
            failure_highlights=[], suggested_next_actions=[],
            command="pytest tests/", narrative="",
        )
        assert "Duration:" not in text

    def test_narrative_included(self) -> None:
        text = _build_summary_text(
            overall_status="PASSED", passed=5, failed=0, skipped=0,
            error=0, total=5, framework="pytest", duration_seconds=None,
            failure_highlights=[], suggested_next_actions=[],
            command="pytest tests/",
            narrative="All 5 tests passed successfully.",
        )
        assert "All 5 tests passed successfully." in text

    def test_narrative_omitted_when_empty(self) -> None:
        text = _build_summary_text(
            overall_status="PASSED", passed=5, failed=0, skipped=0,
            error=0, total=5, framework="pytest", duration_seconds=None,
            failure_highlights=[], suggested_next_actions=[],
            command="pytest tests/", narrative="",
        )
        # No empty sections (double blank lines would indicate blank section)
        assert "\n\n\n" not in text

    def test_counts_line_shows_all_categories(self) -> None:
        text = _build_summary_text(
            overall_status="MIXED", passed=5, failed=2, skipped=1,
            error=1, total=9, framework="pytest", duration_seconds=None,
            failure_highlights=[], suggested_next_actions=[],
            command="pytest tests/", narrative="",
        )
        assert "5 passed" in text
        assert "2 failed" in text
        assert "1 skipped" in text
        assert "1 errors" in text
        assert "9 total" in text

    def test_counts_line_omits_zero_categories(self) -> None:
        text = _build_summary_text(
            overall_status="PASSED", passed=5, failed=0, skipped=0,
            error=0, total=5, framework="pytest", duration_seconds=None,
            failure_highlights=[], suggested_next_actions=[],
            command="pytest tests/", narrative="",
        )
        assert "5 passed" in text
        assert "failed" not in text.lower().replace("run summary", "")

    def test_no_counts_shows_none_detected(self) -> None:
        text = _build_summary_text(
            overall_status="NO_TESTS", passed=0, failed=0, skipped=0,
            error=0, total=0, framework="unknown", duration_seconds=None,
            failure_highlights=[], suggested_next_actions=[],
            command="pytest tests/", narrative="",
        )
        assert "none detected" in text.lower()

    def test_failure_highlights_section(self) -> None:
        highlights = [
            {"test_name": "test_a", "module": "m.py", "message": "boom"},
        ]
        text = _build_summary_text(
            overall_status="FAILED", passed=0, failed=1, skipped=0,
            error=0, total=1, framework="pytest", duration_seconds=None,
            failure_highlights=highlights, suggested_next_actions=[],
            command="pytest tests/", narrative="",
        )
        assert "Failure Highlights:" in text
        assert "m.py::test_a" in text
        assert "boom" in text

    def test_failure_highlight_without_module(self) -> None:
        highlights = [
            {"test_name": "test_a", "module": "", "message": "boom"},
        ]
        text = _build_summary_text(
            overall_status="FAILED", passed=0, failed=1, skipped=0,
            error=0, total=1, framework="pytest", duration_seconds=None,
            failure_highlights=highlights, suggested_next_actions=[],
            command="pytest tests/", narrative="",
        )
        assert "Failure Highlights:" in text
        # No "::" prefix when module is empty
        assert "::test_a" not in text
        assert "test_a" in text

    def test_suggested_next_actions_section(self) -> None:
        actions = ["Do this first", "Then do that"]
        text = _build_summary_text(
            overall_status="FAILED", passed=0, failed=1, skipped=0,
            error=0, total=1, framework="pytest", duration_seconds=None,
            failure_highlights=[], suggested_next_actions=actions,
            command="pytest tests/", narrative="",
        )
        assert "Suggested Next Actions:" in text
        assert "Do this first" in text
        assert "Then do that" in text


# ---------------------------------------------------------------------------
# _parse_structured_results -- direct unit tests
# ---------------------------------------------------------------------------


class TestParseStructuredResults:
    """Direct unit tests for _parse_structured_results helper."""

    def test_valid_json(self) -> None:
        data = _parse_test_results(passed=3, failed=1, framework="jest")
        result = _parse_structured_results(data)
        assert result["summary"]["passed"] == 3
        assert result["summary"]["failed"] == 1
        assert result["framework"] == "jest"

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid test_results JSON"):
            _parse_structured_results("{bad json")

    def test_non_dict_json_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a JSON object"):
            _parse_structured_results(json.dumps([1, 2, 3]))

    def test_non_dict_summary_raises(self) -> None:
        with pytest.raises(ValueError, match="summary must be a JSON object"):
            _parse_structured_results(json.dumps({"summary": "not a dict"}))

    def test_missing_summary_uses_defaults(self) -> None:
        result = _parse_structured_results(json.dumps({}))
        assert result["summary"]["passed"] == 0
        assert result["summary"]["failed"] == 0
        assert result["summary"]["skipped"] == 0
        assert result["summary"]["error"] == 0
        assert result["framework"] == "unknown"
        assert result["records"] == []
        assert result["truncated"] is False

    def test_partial_summary_fills_defaults(self) -> None:
        data = json.dumps({"summary": {"passed": 5}})
        result = _parse_structured_results(data)
        assert result["summary"]["passed"] == 5
        assert result["summary"]["failed"] == 0
        assert result["summary"]["skipped"] == 0

    def test_none_input_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid test_results JSON"):
            _parse_structured_results(None)  # type: ignore[arg-type]

    def test_truncated_field_coerced_to_bool(self) -> None:
        data = json.dumps({"summary": {}, "truncated": 1})
        result = _parse_structured_results(data)
        assert result["truncated"] is True

    def test_records_preserved(self) -> None:
        records = [{"name": "t1", "status": "passed"}]
        data = json.dumps({"summary": {}, "records": records})
        result = _parse_structured_results(data)
        assert len(result["records"]) == 1
        assert result["records"][0]["name"] == "t1"


# ---------------------------------------------------------------------------
# _extract_from_tool_history -- direct unit tests
# ---------------------------------------------------------------------------


class TestExtractFromToolHistory:
    """Direct unit tests for _extract_from_tool_history helper."""

    def test_empty_history(self) -> None:
        result = _extract_from_tool_history(json.dumps([]))
        assert result["has_ssh_error"] is False
        assert result["parsed_results"] is None
        assert result["exit_code"] is None

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid tool_history JSON"):
            _extract_from_tool_history("not json")

    def test_non_list_json_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a JSON array"):
            _extract_from_tool_history(json.dumps({"key": "val"}))

    def test_non_dict_entries_skipped(self) -> None:
        history = json.dumps(["string_entry", 42, None])
        result = _extract_from_tool_history(history)
        assert result["has_ssh_error"] is False
        assert result["parsed_results"] is None

    def test_ssh_error_detected(self) -> None:
        history = json.dumps([
            _tool_history_entry(
                tool_name="execute_ssh",
                status="error",
                error_message="Auth failed",
            ),
        ])
        result = _extract_from_tool_history(history)
        assert result["has_ssh_error"] is True
        assert result["ssh_error_message"] == "Auth failed"

    def test_ssh_error_without_message(self) -> None:
        history = json.dumps([
            _tool_history_entry(
                tool_name="execute_ssh",
                status="error",
                error_message=None,
            ),
        ])
        result = _extract_from_tool_history(history)
        assert result["has_ssh_error"] is True
        assert result["ssh_error_message"] == "SSH execution failed"

    def test_ssh_success_extracts_exit_code(self) -> None:
        history = json.dumps([
            _tool_history_entry(
                tool_name="execute_ssh",
                status="success",
                output=json.dumps({"exit_code": 42}),
            ),
        ])
        result = _extract_from_tool_history(history)
        assert result["exit_code"] == 42
        assert result["has_ssh_error"] is False

    def test_ssh_success_with_malformed_output(self) -> None:
        """SSH output that isn't valid JSON is silently ignored."""
        history = json.dumps([
            _tool_history_entry(
                tool_name="execute_ssh",
                status="success",
                output="not json",
            ),
        ])
        result = _extract_from_tool_history(history)
        assert result["exit_code"] is None

    def test_parse_test_output_extracted(self) -> None:
        parse_output = _parse_test_results(passed=8, failed=1)
        history = json.dumps([
            _tool_history_entry(
                tool_name="parse_test_output",
                status="success",
                output=parse_output,
            ),
        ])
        result = _extract_from_tool_history(history)
        assert result["parsed_results"] is not None
        assert result["parsed_results"]["summary"]["passed"] == 8

    def test_parse_test_output_error_status_ignored(self) -> None:
        """parse_test_output with error status should not be extracted."""
        history = json.dumps([
            _tool_history_entry(
                tool_name="parse_test_output",
                status="error",
                output=_parse_test_results(passed=5),
            ),
        ])
        result = _extract_from_tool_history(history)
        assert result["parsed_results"] is None

    def test_parse_test_output_malformed_output_skipped(self) -> None:
        """parse_test_output with unparseable output should be silently skipped."""
        history = json.dumps([
            _tool_history_entry(
                tool_name="parse_test_output",
                status="success",
                output="[1,2,3]",  # valid JSON but not a dict
            ),
        ])
        result = _extract_from_tool_history(history)
        assert result["parsed_results"] is None

    def test_multiple_entries_last_ssh_wins(self) -> None:
        """When multiple execute_ssh entries exist, later ones override."""
        history = json.dumps([
            _tool_history_entry(
                tool_name="execute_ssh",
                status="success",
                output=json.dumps({"exit_code": 0}),
            ),
            _tool_history_entry(
                tool_name="execute_ssh",
                status="success",
                output=json.dumps({"exit_code": 1}),
            ),
        ])
        result = _extract_from_tool_history(history)
        assert result["exit_code"] == 1

    def test_ssh_error_and_parse_results_both_present(self) -> None:
        """SSH error + valid parse results should both be extracted."""
        parse_output = _parse_test_results(passed=5, failed=2)
        history = json.dumps([
            _tool_history_entry(
                tool_name="execute_ssh",
                status="error",
                error_message="Partial SSH failure",
            ),
            _tool_history_entry(
                tool_name="parse_test_output",
                status="success",
                output=parse_output,
            ),
        ])
        result = _extract_from_tool_history(history)
        assert result["has_ssh_error"] is True
        assert result["parsed_results"] is not None
        assert result["parsed_results"]["summary"]["passed"] == 5


# ---------------------------------------------------------------------------
# Overall status determination (integration through execute)
# ---------------------------------------------------------------------------


class TestOverallStatus:
    """Verify overall_status is derived correctly from test counts."""

    @pytest.mark.asyncio
    async def test_all_passed_status(self) -> None:
        """When all tests pass, overall_status should be PASSED."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=10, failed=0, skipped=0)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c1",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["overall_status"] == "PASSED"

    @pytest.mark.asyncio
    async def test_all_failed_status(self) -> None:
        """When all tests fail, overall_status should be FAILED."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=0, failed=5, skipped=0)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c2",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["overall_status"] == "FAILED"

    @pytest.mark.asyncio
    async def test_mixed_status(self) -> None:
        """When some tests pass and some fail, overall_status should be MIXED."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=8, failed=2, skipped=1)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c3",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["overall_status"] == "MIXED"

    @pytest.mark.asyncio
    async def test_error_status(self) -> None:
        """When errors are present with no passes, overall_status should be ERROR."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=0, failed=0, error=3)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c4",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["overall_status"] == "ERROR"

    @pytest.mark.asyncio
    async def test_no_tests_status(self) -> None:
        """When no tests were detected, overall_status should be NO_TESTS."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=0, failed=0, skipped=0)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c5",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["overall_status"] == "NO_TESTS"

    @pytest.mark.asyncio
    async def test_skipped_only_status(self) -> None:
        """When only skipped tests exist, overall_status should be PASSED."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=0, failed=0, skipped=5)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c6",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["overall_status"] == "PASSED"

    @pytest.mark.asyncio
    async def test_exit_code_overrides_to_error(self) -> None:
        """Non-zero exit code with no failed tests should result in ERROR."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "stdout": "no test output here",
            "command": "pytest tests/",
            "exit_code": 2,
            "_call_id": "c7",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["overall_status"] in ("ERROR", "NO_TESTS")

    @pytest.mark.asyncio
    async def test_errors_with_passes_returns_mixed(self) -> None:
        """Errors + passes (no fails) -> MIXED via structured input."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=3, failed=0, error=2)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c8",
        })
        data = json.loads(result.output)
        assert data["overall_status"] == "MIXED"


# ---------------------------------------------------------------------------
# Failure highlights (integration through execute)
# ---------------------------------------------------------------------------


class TestFailureHighlights:
    """Verify failure_highlights are extracted from structured results."""

    @pytest.mark.asyncio
    async def test_extracts_failure_records(self) -> None:
        """Failure highlights should include test name and failure message."""
        tool = SummarizeRunTool()
        records = [
            {
                "name": "test_login",
                "status": "failed",
                "module": "tests/test_auth.py",
                "failure_message": "AssertionError: expected 401 got 200",
                "duration_seconds": 0.5,
            },
            {
                "name": "test_signup",
                "status": "passed",
                "module": "tests/test_auth.py",
                "failure_message": None,
                "duration_seconds": 0.1,
            },
            {
                "name": "test_logout",
                "status": "failed",
                "module": "tests/test_auth.py",
                "failure_message": "ConnectionError: timeout",
                "duration_seconds": 5.0,
            },
        ]
        test_results = _parse_test_results(
            passed=1, failed=2, records=records, framework="pytest",
        )
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/test_auth.py",
            "_call_id": "c1",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        highlights = data["failure_highlights"]
        assert len(highlights) == 2
        assert highlights[0]["test_name"] == "test_login"
        assert "AssertionError" in highlights[0]["message"]
        assert highlights[1]["test_name"] == "test_logout"

    @pytest.mark.asyncio
    async def test_caps_failure_highlights_at_five(self) -> None:
        """At most 5 failure highlights should be included."""
        tool = SummarizeRunTool()
        records = [
            {
                "name": f"test_case_{i}",
                "status": "failed",
                "module": "tests/test_bulk.py",
                "failure_message": f"Error in case {i}",
                "duration_seconds": None,
            }
            for i in range(10)
        ]
        test_results = _parse_test_results(
            passed=0, failed=10, records=records,
        )
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/test_bulk.py",
            "_call_id": "c2",
        })
        data = json.loads(result.output)
        assert len(data["failure_highlights"]) <= 5

    @pytest.mark.asyncio
    async def test_error_records_in_highlights(self) -> None:
        """Error status records should also appear in failure highlights."""
        tool = SummarizeRunTool()
        records = [
            {
                "name": "test_broken",
                "status": "error",
                "module": "tests/test_setup.py",
                "failure_message": "ImportError: cannot import X",
                "duration_seconds": None,
            },
        ]
        test_results = _parse_test_results(
            passed=0, failed=0, error=1, records=records,
        )
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/test_setup.py",
            "_call_id": "c3",
        })
        data = json.loads(result.output)
        assert len(data["failure_highlights"]) == 1
        assert data["failure_highlights"][0]["test_name"] == "test_broken"


# ---------------------------------------------------------------------------
# Suggested next actions (integration through execute)
# ---------------------------------------------------------------------------


class TestSuggestedNextActions:
    """Verify suggested_next_actions are generated based on outcomes."""

    @pytest.mark.asyncio
    async def test_all_passed_suggests_nothing_critical(self) -> None:
        """When all tests pass, no critical rerun action needed."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=10, failed=0)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c1",
        })
        data = json.loads(result.output)
        actions = data["suggested_next_actions"]
        assert isinstance(actions, list)
        # No rerun suggestions when everything passes
        rerun_actions = [a for a in actions if "rerun" in a.lower()]
        assert len(rerun_actions) == 0

    @pytest.mark.asyncio
    async def test_failures_suggest_rerun(self) -> None:
        """When tests fail, suggest rerunning the failed tests."""
        tool = SummarizeRunTool()
        records = [
            {
                "name": "test_login",
                "status": "failed",
                "module": "tests/test_auth.py",
                "failure_message": "AssertionError",
                "duration_seconds": None,
            },
        ]
        test_results = _parse_test_results(
            passed=9, failed=1, records=records,
        )
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c2",
        })
        data = json.loads(result.output)
        actions = data["suggested_next_actions"]
        assert len(actions) >= 1
        # At least one action should mention rerunning
        assert any("rerun" in a.lower() or "re-run" in a.lower() for a in actions)

    @pytest.mark.asyncio
    async def test_errors_suggest_investigation(self) -> None:
        """When errors occur, suggest investigating the errors."""
        tool = SummarizeRunTool()
        records = [
            {
                "name": "test_broken",
                "status": "error",
                "module": "tests/test_setup.py",
                "failure_message": "ImportError: cannot import module_x",
                "duration_seconds": None,
            },
        ]
        test_results = _parse_test_results(
            passed=0, failed=0, error=1, records=records,
        )
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c3",
        })
        data = json.loads(result.output)
        actions = data["suggested_next_actions"]
        assert len(actions) >= 1

    @pytest.mark.asyncio
    async def test_truncated_output_suggests_full_output(self) -> None:
        """When output was truncated, suggest viewing the full output."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(
            passed=5, failed=1, truncated=True,
        )
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c4",
        })
        data = json.loads(result.output)
        actions = data["suggested_next_actions"]
        assert any(
            "truncat" in a.lower() or "full" in a.lower() or "output" in a.lower()
            for a in actions
        )


# ---------------------------------------------------------------------------
# Summary text (integration through execute)
# ---------------------------------------------------------------------------


class TestSummaryText:
    """Verify the human-readable summary_text is well-formed."""

    @pytest.mark.asyncio
    async def test_summary_text_present(self) -> None:
        """A summary_text field should be present in the output."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=5, failed=2, skipped=1)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c1",
        })
        data = json.loads(result.output)
        assert "summary_text" in data
        assert isinstance(data["summary_text"], str)
        assert len(data["summary_text"]) > 0

    @pytest.mark.asyncio
    async def test_summary_text_contains_counts(self) -> None:
        """Summary text should contain the test counts."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=10, failed=3, skipped=2)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c2",
        })
        data = json.loads(result.output)
        text = data["summary_text"]
        assert "10" in text  # passed count
        assert "3" in text   # failed count

    @pytest.mark.asyncio
    async def test_summary_text_contains_status(self) -> None:
        """Summary text should contain the overall status."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=0, failed=5)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c3",
        })
        data = json.loads(result.output)
        text = data["summary_text"]
        assert "FAILED" in text


# ---------------------------------------------------------------------------
# Structured test_results input (from parse_test_output)
# ---------------------------------------------------------------------------


class TestStructuredTestResultsInput:
    """Verify summarize_run accepts parse_test_output JSON directly."""

    @pytest.mark.asyncio
    async def test_accepts_test_results_json(self) -> None:
        """Tool should accept test_results as JSON string from parse_test_output."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(
            passed=8, failed=2, skipped=1, framework="pytest",
        )
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c1",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["passed"] == 8
        assert data["failed"] == 2
        assert data["skipped"] == 1
        assert data["total"] == 11  # 8 + 2 + 1

    @pytest.mark.asyncio
    async def test_malformed_test_results_returns_error(self) -> None:
        """Malformed test_results JSON should produce an error result."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "test_results": "this is {not valid json",
            "command": "pytest tests/",
            "_call_id": "c2",
        })
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_preserves_framework_field(self) -> None:
        """Framework detected by parse_test_output should be preserved."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(
            passed=5, failed=0, framework="jest",
        )
        result = await tool.execute({
            "test_results": test_results,
            "command": "npm test",
            "_call_id": "c3",
        })
        data = json.loads(result.output)
        assert data["framework"] == "jest"

    @pytest.mark.asyncio
    async def test_non_dict_test_results_returns_error(self) -> None:
        """Non-dict test_results JSON should produce an error result."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "test_results": json.dumps([1, 2, 3]),
            "command": "pytest tests/",
            "_call_id": "c4",
        })
        assert result.status == ToolResultStatus.ERROR
        assert "JSON object" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_exit_code_passed_through_structured(self) -> None:
        """exit_code provided alongside test_results should affect status."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=0, failed=0, skipped=0)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "exit_code": 2,
            "_call_id": "c5",
        })
        data = json.loads(result.output)
        # No tests detected + non-zero exit -> ERROR
        assert data["overall_status"] == "ERROR"


# ---------------------------------------------------------------------------
# Tool call history input
# ---------------------------------------------------------------------------


class TestToolHistoryInput:
    """Verify summarize_run can synthesize from tool call history."""

    @pytest.mark.asyncio
    async def test_extracts_from_tool_history(self) -> None:
        """Tool should extract results from accumulated tool call history."""
        tool = SummarizeRunTool()
        parse_output = _parse_test_results(passed=5, failed=2, framework="pytest")
        history = json.dumps([
            _tool_history_entry(
                tool_name="execute_ssh",
                output=json.dumps({"command": "pytest tests/", "exit_code": 1}),
            ),
            _tool_history_entry(
                tool_name="parse_test_output",
                output=parse_output,
            ),
        ])
        result = await tool.execute({
            "tool_history": history,
            "command": "pytest tests/",
            "_call_id": "c1",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["passed"] == 5
        assert data["failed"] == 2

    @pytest.mark.asyncio
    async def test_tool_history_with_ssh_failure(self) -> None:
        """History with SSH failure should result in ERROR status."""
        tool = SummarizeRunTool()
        history = json.dumps([
            _tool_history_entry(
                tool_name="execute_ssh",
                status="error",
                error_message="SSH connection refused",
            ),
        ])
        result = await tool.execute({
            "tool_history": history,
            "command": "pytest tests/",
            "_call_id": "c2",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["overall_status"] == "ERROR"

    @pytest.mark.asyncio
    async def test_empty_tool_history(self) -> None:
        """Empty tool history should produce a NO_TESTS summary."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "tool_history": json.dumps([]),
            "command": "pytest tests/",
            "_call_id": "c3",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["overall_status"] == "NO_TESTS"

    @pytest.mark.asyncio
    async def test_malformed_tool_history_returns_error(self) -> None:
        """Malformed tool_history JSON should produce an error result."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "tool_history": "invalid{json",
            "command": "pytest tests/",
            "_call_id": "c4",
        })
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_non_list_tool_history_returns_error(self) -> None:
        """Non-list tool_history should produce an error result."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "tool_history": json.dumps({"not": "a list"}),
            "command": "pytest tests/",
            "_call_id": "c5",
        })
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_tool_history_ssh_error_plus_test_results(self) -> None:
        """SSH error + test results both extracted from history."""
        tool = SummarizeRunTool()
        parse_output = _parse_test_results(passed=5, failed=2)
        history = json.dumps([
            _tool_history_entry(
                tool_name="execute_ssh",
                status="error",
                error_message="Partial failure",
            ),
            _tool_history_entry(
                tool_name="parse_test_output",
                status="success",
                output=parse_output,
            ),
        ])
        result = await tool.execute({
            "tool_history": history,
            "command": "pytest tests/",
            "_call_id": "c6",
        })
        data = json.loads(result.output)
        assert data["overall_status"] == "MIXED"
        assert data["passed"] == 5
        assert data["failed"] == 2
        # SSH error should appear in next actions
        actions = data["suggested_next_actions"]
        assert any("SSH" in a for a in actions)

    @pytest.mark.asyncio
    async def test_tool_history_exit_code_from_ssh(self) -> None:
        """Exit code extracted from execute_ssh output in history."""
        tool = SummarizeRunTool()
        history = json.dumps([
            _tool_history_entry(
                tool_name="execute_ssh",
                status="success",
                output=json.dumps({"exit_code": 2}),
            ),
        ])
        result = await tool.execute({
            "tool_history": history,
            "command": "unknown_cmd",
            "_call_id": "c7",
        })
        data = json.loads(result.output)
        # No tests + non-zero exit_code -> ERROR
        assert data["overall_status"] == "ERROR"

    @pytest.mark.asyncio
    async def test_tool_history_exit_code_param_overrides_history(self) -> None:
        """Explicit exit_code param overrides one from history."""
        tool = SummarizeRunTool()
        history = json.dumps([
            _tool_history_entry(
                tool_name="execute_ssh",
                status="success",
                output=json.dumps({"exit_code": 2}),
            ),
        ])
        result = await tool.execute({
            "tool_history": history,
            "command": "pytest tests/",
            "exit_code": 0,
            "_call_id": "c8",
        })
        data = json.loads(result.output)
        # No tests + exit_code=0 -> NO_TESTS
        assert data["overall_status"] == "NO_TESTS"

    @pytest.mark.asyncio
    async def test_tool_history_with_non_dict_entries(self) -> None:
        """Non-dict entries in history array should be silently skipped."""
        tool = SummarizeRunTool()
        history = json.dumps([
            "a string entry",
            42,
            None,
            _tool_history_entry(
                tool_name="parse_test_output",
                status="success",
                output=_parse_test_results(passed=3),
            ),
        ])
        result = await tool.execute({
            "tool_history": history,
            "command": "pytest tests/",
            "_call_id": "c9",
        })
        data = json.loads(result.output)
        assert data["passed"] == 3


# ---------------------------------------------------------------------------
# Backward compatibility with raw stdout/stderr/command mode
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Verify the raw stdout/stderr/command mode still works."""

    @pytest.mark.asyncio
    async def test_raw_mode_with_pytest_output(self) -> None:
        """Raw mode with pytest output should still delegate to output_summarizer."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "stdout": "=== 10 passed, 2 failed, 1 skipped in 5.23s ===",
            "stderr": "",
            "command": "pytest tests/",
            "exit_code": 1,
            "_call_id": "c1",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["passed"] == 10
        assert data["failed"] == 2
        assert data["overall_status"] in ("MIXED", "FAILED")

    @pytest.mark.asyncio
    async def test_raw_mode_missing_command_returns_error(self) -> None:
        """Missing command parameter should still return error."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "stdout": "output",
            "command": "",
            "_call_id": "c2",
        })
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_raw_mode_all_passed(self) -> None:
        """Raw mode with all-pass output should have PASSED status."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "stdout": "=== 20 passed in 3.0s ===",
            "command": "pytest tests/",
            "exit_code": 0,
            "_call_id": "c3",
        })
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["overall_status"] == "PASSED"

    @pytest.mark.asyncio
    async def test_raw_mode_summarizer_exception_returns_error(self) -> None:
        """If output_summarizer raises, tool should return error cleanly."""
        tool = SummarizeRunTool()
        with patch(
            "jules_daemon.execution.output_summarizer.summarize_output",
            side_effect=TimeoutError("LLM timeout"),
        ):
            result = await tool.execute({
                "stdout": "some output",
                "stderr": "",
                "command": "pytest tests/",
                "exit_code": 0,
                "_call_id": "c4",
            })
        assert result.status == ToolResultStatus.ERROR
        assert "Summarization failed" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_raw_mode_with_duration(self) -> None:
        """Raw mode preserves duration from output_summarizer."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "stdout": "=== 5 passed in 12.34s ===",
            "command": "pytest tests/",
            "exit_code": 0,
            "_call_id": "c5",
        })
        data = json.loads(result.output)
        assert data["duration_seconds"] is not None
        assert data["duration_seconds"] > 0


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Verify the output JSON has the required structure."""

    @pytest.mark.asyncio
    async def test_output_has_required_fields(self) -> None:
        """Output must have overall_status, failure_highlights, suggested_next_actions."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=5, failed=2)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c1",
        })
        data = json.loads(result.output)
        required_fields = {
            "overall_status",
            "summary_text",
            "passed",
            "failed",
            "skipped",
            "total",
            "failure_highlights",
            "suggested_next_actions",
        }
        assert required_fields.issubset(set(data.keys()))

    @pytest.mark.asyncio
    async def test_failure_highlights_structure(self) -> None:
        """Each failure highlight must have test_name and message."""
        tool = SummarizeRunTool()
        records = [
            {
                "name": "test_example",
                "status": "failed",
                "module": "tests/test_ex.py",
                "failure_message": "ValueError: bad input",
                "duration_seconds": None,
            },
        ]
        test_results = _parse_test_results(
            passed=0, failed=1, records=records,
        )
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c2",
        })
        data = json.loads(result.output)
        for highlight in data["failure_highlights"]:
            assert "test_name" in highlight
            assert "message" in highlight

    @pytest.mark.asyncio
    async def test_spec_metadata(self) -> None:
        """Tool spec should have correct name and approval requirement."""
        tool = SummarizeRunTool()
        assert tool.spec.name == "summarize_run"
        assert tool.spec.is_read_only is True

    @pytest.mark.asyncio
    async def test_output_includes_framework_and_narrative(self) -> None:
        """Output should include framework and narrative fields."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=5, framework="mocha")
        result = await tool.execute({
            "test_results": test_results,
            "command": "npm test",
            "_call_id": "c3",
        })
        data = json.loads(result.output)
        assert "framework" in data
        assert data["framework"] == "mocha"
        assert "narrative" in data

    @pytest.mark.asyncio
    async def test_output_total_aggregation(self) -> None:
        """Total should equal sum of passed + failed + skipped + error."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=3, failed=2, skipped=1, error=1)
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c4",
        })
        data = json.loads(result.output)
        # total = passed + failed + skipped + error (error is included in
        # the total but not exposed as a separate JSON field)
        assert data["total"] == 3 + 2 + 1 + 1  # 7
        assert data["passed"] == 3
        assert data["failed"] == 2
        assert data["skipped"] == 1


# ---------------------------------------------------------------------------
# Mode selection priority
# ---------------------------------------------------------------------------


class TestModeSelectionPriority:
    """Verify test_results takes priority over tool_history over raw."""

    @pytest.mark.asyncio
    async def test_test_results_takes_priority_over_history(self) -> None:
        """When both test_results and tool_history given, test_results wins."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(passed=10, failed=0)
        history = json.dumps([
            _tool_history_entry(
                tool_name="parse_test_output",
                status="success",
                output=_parse_test_results(passed=0, failed=5),
            ),
        ])
        result = await tool.execute({
            "test_results": test_results,
            "tool_history": history,
            "command": "pytest tests/",
            "_call_id": "c1",
        })
        data = json.loads(result.output)
        # test_results (10 passed) should take priority
        assert data["passed"] == 10
        assert data["failed"] == 0

    @pytest.mark.asyncio
    async def test_history_takes_priority_over_raw(self) -> None:
        """When both tool_history and stdout given, tool_history wins."""
        tool = SummarizeRunTool()
        history = json.dumps([
            _tool_history_entry(
                tool_name="parse_test_output",
                status="success",
                output=_parse_test_results(passed=7, failed=1),
            ),
        ])
        result = await tool.execute({
            "tool_history": history,
            "stdout": "=== 20 passed in 1.0s ===",
            "command": "pytest tests/",
            "_call_id": "c2",
        })
        data = json.loads(result.output)
        # tool_history (7 passed) should take priority over raw (20 passed)
        assert data["passed"] == 7
        assert data["failed"] == 1

    @pytest.mark.asyncio
    async def test_falls_through_to_raw_when_no_structured_input(self) -> None:
        """When neither test_results nor tool_history, falls to raw mode."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "stdout": "=== 15 passed in 2.0s ===",
            "command": "pytest tests/",
            "exit_code": 0,
            "_call_id": "c3",
        })
        data = json.loads(result.output)
        assert data["passed"] == 15


# ---------------------------------------------------------------------------
# Key failures fallback (from output_summarizer when no records)
# ---------------------------------------------------------------------------


class TestKeyFailuresFallback:
    """Verify key_failures from output_summarizer populate highlights."""

    @pytest.mark.asyncio
    async def test_key_failures_become_highlights(self) -> None:
        """When no per-test records, key_failures from summarizer should become highlights."""
        tool = SummarizeRunTool()

        # Mock the output_summarizer to return key_failures
        mock_summary = AsyncMock()
        mock_summary.passed = 3
        mock_summary.failed = 2
        mock_summary.skipped = 0
        mock_summary.parser = "pytest"
        mock_summary.duration_seconds = 5.0
        mock_summary.key_failures = ("test_a failed: boom", "test_b failed: crash")
        mock_summary.narrative = ""
        mock_summary.raw_excerpt = ""

        with patch(
            "jules_daemon.execution.output_summarizer.summarize_output",
            return_value=mock_summary,
        ):
            result = await tool.execute({
                "stdout": "test output",
                "command": "pytest tests/",
                "exit_code": 1,
                "_call_id": "c1",
            })

        data = json.loads(result.output)
        assert len(data["failure_highlights"]) == 2
        assert data["failure_highlights"][0]["test_name"] == "failure_1"
        assert "boom" in data["failure_highlights"][0]["message"]
        assert data["failure_highlights"][1]["test_name"] == "failure_2"

    @pytest.mark.asyncio
    async def test_key_failures_capped_at_five(self) -> None:
        """key_failures fallback should also respect the 5-highlight cap."""
        tool = SummarizeRunTool()

        mock_summary = AsyncMock()
        mock_summary.passed = 0
        mock_summary.failed = 8
        mock_summary.skipped = 0
        mock_summary.parser = "pytest"
        mock_summary.duration_seconds = None
        mock_summary.key_failures = tuple(f"failure_{i}" for i in range(8))
        mock_summary.narrative = ""
        mock_summary.raw_excerpt = ""

        with patch(
            "jules_daemon.execution.output_summarizer.summarize_output",
            return_value=mock_summary,
        ):
            result = await tool.execute({
                "stdout": "test output",
                "command": "pytest tests/",
                "exit_code": 1,
                "_call_id": "c2",
            })

        data = json.loads(result.output)
        assert len(data["failure_highlights"]) <= 5


# ---------------------------------------------------------------------------
# Edge cases and error resilience
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_missing_command_returns_error(self) -> None:
        """Missing command in any mode should return error."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "test_results": _parse_test_results(passed=5),
            "_call_id": "c1",
        })
        assert result.status == ToolResultStatus.ERROR
        assert "command" in (result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_default_call_id(self) -> None:
        """When _call_id is not provided, a default is used."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "test_results": _parse_test_results(passed=5),
            "command": "pytest tests/",
        })
        assert result.status == ToolResultStatus.SUCCESS
        assert result.call_id == "summarize_run"

    @pytest.mark.asyncio
    async def test_large_counts(self) -> None:
        """Very large test counts should be handled correctly."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(
            passed=50000, failed=0, skipped=100,
        )
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c2",
        })
        data = json.loads(result.output)
        assert data["passed"] == 50000
        assert data["total"] == 50100
        assert data["overall_status"] == "PASSED"

    @pytest.mark.asyncio
    async def test_all_zero_counts(self) -> None:
        """All zero counts with zero exit code -> NO_TESTS."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(
            passed=0, failed=0, skipped=0, error=0,
        )
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "exit_code": 0,
            "_call_id": "c3",
        })
        data = json.loads(result.output)
        assert data["overall_status"] == "NO_TESTS"
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_tool_result_always_has_tool_name(self) -> None:
        """ToolResult from execute should always have tool_name set."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "test_results": _parse_test_results(passed=1),
            "command": "pytest",
            "_call_id": "c4",
        })
        assert result.tool_name == "summarize_run"

    @pytest.mark.asyncio
    async def test_error_result_has_tool_name(self) -> None:
        """Even error results should have tool_name set."""
        tool = SummarizeRunTool()
        result = await tool.execute({
            "test_results": "bad json",
            "command": "pytest",
            "_call_id": "c5",
        })
        assert result.tool_name == "summarize_run"
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_incomplete_field_ignored_in_total(self) -> None:
        """The 'incomplete' summary field should not affect total."""
        tool = SummarizeRunTool()
        test_results = _parse_test_results(
            passed=5, failed=0, skipped=0, error=0, incomplete=3,
        )
        result = await tool.execute({
            "test_results": test_results,
            "command": "pytest tests/",
            "_call_id": "c6",
        })
        data = json.loads(result.output)
        # total = passed + failed + skipped + error (not incomplete)
        assert data["total"] == 5

    @pytest.mark.asyncio
    async def test_tool_history_no_parse_results_uses_exit_code(self) -> None:
        """History with SSH only (no parse_test_output) uses exit_code for status."""
        tool = SummarizeRunTool()
        history = json.dumps([
            _tool_history_entry(
                tool_name="execute_ssh",
                status="success",
                output=json.dumps({"exit_code": 0}),
            ),
        ])
        result = await tool.execute({
            "tool_history": history,
            "command": "custom_script.sh",
            "_call_id": "c7",
        })
        data = json.loads(result.output)
        # No tests + exit_code=0 -> NO_TESTS
        assert data["overall_status"] == "NO_TESTS"

    @pytest.mark.asyncio
    async def test_summary_text_is_parseable_string(self) -> None:
        """Summary text should always be a non-empty string for all statuses."""
        tool = SummarizeRunTool()
        for status_config in [
            {"passed": 10, "failed": 0},
            {"passed": 0, "failed": 5},
            {"passed": 5, "failed": 3},
            {"passed": 0, "failed": 0, "error": 2},
            {"passed": 0, "failed": 0},
        ]:
            test_results = _parse_test_results(**status_config)
            result = await tool.execute({
                "test_results": test_results,
                "command": "pytest tests/",
                "_call_id": "status_check",
            })
            data = json.loads(result.output)
            assert isinstance(data["summary_text"], str)
            assert len(data["summary_text"]) > 0
