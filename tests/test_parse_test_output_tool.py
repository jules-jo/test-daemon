"""Tests for the parse_test_output agent tool.

Verifies that the tool:
- Parses pytest output into structured records with failure messages
- Parses jest output into structured records with failure messages
- Parses go test output into structured records with failure messages
- Returns proper pass/fail/skip/error counts in summary
- Handles empty input gracefully
- Handles unknown framework output gracefully
- Returns proper ToolResult status codes
- Includes error_details for failed tests
- Works with the BaseTool calling convention (args dict with _call_id)
"""

from __future__ import annotations

import json

import pytest

from jules_daemon.agent.tool_types import ToolResultStatus
from jules_daemon.agent.tools.parse_test_output import ParseTestOutputTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(raw_output: str, framework_hint: str = "auto") -> dict:
    """Run the tool and return the parsed JSON output."""
    tool = ParseTestOutputTool()
    result = pytest.helpers.run_async(
        tool.execute({
            "raw_output": raw_output,
            "framework_hint": framework_hint,
            "_call_id": "test-call-1",
        })
    )
    assert result.status == ToolResultStatus.SUCCESS
    return json.loads(result.output)


@pytest.fixture
def tool() -> ParseTestOutputTool:
    return ParseTestOutputTool()


# ---------------------------------------------------------------------------
# conftest helper: run_async
# ---------------------------------------------------------------------------


def run_async(coro):
    """Run an async coroutine synchronously for test execution."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_tool_name(self, tool: ParseTestOutputTool) -> None:
        assert tool.name == "parse_test_output"

    def test_tool_spec_name(self, tool: ParseTestOutputTool) -> None:
        assert tool.spec.name == "parse_test_output"

    def test_tool_is_read_only(self, tool: ParseTestOutputTool) -> None:
        from jules_daemon.agent.tool_types import ApprovalRequirement
        assert tool.spec.approval is ApprovalRequirement.NONE

    def test_tool_has_raw_output_param(self, tool: ParseTestOutputTool) -> None:
        param_names = [p.name for p in tool.spec.parameters]
        assert "raw_output" in param_names

    def test_tool_has_framework_hint_param(self, tool: ParseTestOutputTool) -> None:
        param_names = [p.name for p in tool.spec.parameters]
        assert "framework_hint" in param_names

    def test_tool_has_summary_fields_param(
        self, tool: ParseTestOutputTool
    ) -> None:
        param_names = [p.name for p in tool.spec.parameters]
        assert "summary_fields" in param_names

    def test_framework_hint_includes_jest(self, tool: ParseTestOutputTool) -> None:
        for p in tool.spec.parameters:
            if p.name == "framework_hint":
                assert p.enum is not None
                assert "jest" in p.enum

    def test_framework_hint_includes_go_test(self, tool: ParseTestOutputTool) -> None:
        for p in tool.spec.parameters:
            if p.name == "framework_hint":
                assert p.enum is not None
                assert "go_test" in p.enum


# ---------------------------------------------------------------------------
# Empty / trivial input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_string_returns_success(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": "",
            "_call_id": "c1",
        }))
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["records"] == []
        assert data["summary"]["passed"] == 0
        assert data["summary"]["failed"] == 0

    def test_whitespace_only_returns_empty(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": "   \n\n  ",
            "_call_id": "c2",
        }))
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["records"] == []


# ---------------------------------------------------------------------------
# Pytest output parsing
# ---------------------------------------------------------------------------


class TestPytestOutput:
    """Tool-level tests for pytest output parsing."""

    def test_simple_pass(self, tool: ParseTestOutputTool) -> None:
        output = (
            "============================= test session starts =============================\n"
            "collected 1 item\n\n"
            "tests/test_auth.py::test_login PASSED\n\n"
            "============================== 1 passed in 0.12s ==============================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "c1",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 0
        assert len(data["records"]) == 1
        assert data["records"][0]["name"] == "test_login"
        assert data["records"][0]["status"] == "passed"

    def test_mixed_results(self, tool: ParseTestOutputTool) -> None:
        output = (
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_register FAILED\n"
            "tests/test_auth.py::test_legacy SKIPPED\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "c2",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1
        assert data["summary"]["skipped"] == 1

    def test_failure_message_extracted(self, tool: ParseTestOutputTool) -> None:
        output = (
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_register FAILED\n"
            "\n"
            "=================================== FAILURES ===================================\n"
            "_________________________________ test_register _________________________________\n"
            "\n"
            "    def test_register():\n"
            ">       assert create_user('test') is not None\n"
            "E       AssertionError: assert None is not None\n"
            "\n"
            "tests/test_auth.py:42: AssertionError\n"
            "=========================== short test summary info ============================\n"
            "FAILED tests/test_auth.py::test_register - AssertionError\n"
            "========================= 1 failed, 1 passed in 2.34s =========================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "c3",
        }))
        data = json.loads(result.output)
        failed_records = [r for r in data["records"] if r["status"] == "failed"]
        assert len(failed_records) == 1
        # The failure message should be captured
        assert "failure_message" in failed_records[0]
        assert failed_records[0]["failure_message"] is not None
        assert "AssertionError" in failed_records[0]["failure_message"]

    def test_multiple_failure_messages(self, tool: ParseTestOutputTool) -> None:
        output = (
            "tests/test_math.py::test_add FAILED\n"
            "tests/test_math.py::test_sub FAILED\n"
            "tests/test_math.py::test_mul PASSED\n"
            "\n"
            "=================================== FAILURES ===================================\n"
            "_________________________________ test_add _____________________________________\n"
            "\n"
            "    def test_add():\n"
            ">       assert 1 + 1 == 3\n"
            "E       assert 2 == 3\n"
            "\n"
            "tests/test_math.py:5: AssertionError\n"
            "_________________________________ test_sub _____________________________________\n"
            "\n"
            "    def test_sub():\n"
            ">       assert 5 - 3 == 1\n"
            "E       assert 2 == 1\n"
            "\n"
            "tests/test_math.py:10: AssertionError\n"
            "========================= 2 failed, 1 passed in 0.03s =========================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "c4",
        }))
        data = json.loads(result.output)
        failed_records = [r for r in data["records"] if r["status"] == "failed"]
        assert len(failed_records) == 2
        # Each failure should have its own message
        assert "assert 2 == 3" in failed_records[0]["failure_message"]
        assert "assert 2 == 1" in failed_records[1]["failure_message"]

    def test_error_record(self, tool: ParseTestOutputTool) -> None:
        output = (
            "tests/test_auth.py::test_setup ERROR\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "c5",
        }))
        data = json.loads(result.output)
        assert data["summary"]["error"] == 1
        assert data["records"][0]["status"] == "error"


# ---------------------------------------------------------------------------
# Jest output parsing
# ---------------------------------------------------------------------------


JEST_SIMPLE_PASS = """\
 PASS  src/utils/math.test.ts
  Math utils
    \u2713 adds numbers (3 ms)
    \u2713 subtracts numbers (1 ms)

Test Suites: 1 passed, 1 total
Tests:       2 passed, 2 total
Snapshots:   0 total
Time:        1.234 s
"""

JEST_MIXED = """\
 PASS  src/utils/math.test.ts
  Math utils
    \u2713 adds numbers (3 ms)
    \u2713 subtracts numbers (1 ms)

 FAIL  src/components/Form.test.tsx
  Form validation
    \u2717 validates email field (5 ms)
    \u2713 validates name field (2 ms)

  \u25cf Form validation > validates email field

    expect(received).toBe(expected)

    Expected: true
    Received: false

      41 |   const result = validateEmail('invalid');
      42 |   expect(result).toBe(true);
         |                  ^
      43 | });

      at Object.<anonymous> (src/components/Form.test.tsx:42:18)

Test Suites: 1 failed, 1 passed, 2 total
Tests:       1 failed, 3 passed, 4 total
Snapshots:   0 total
Time:        2.456 s
"""

JEST_ALL_FAIL = """\
 FAIL  src/api/client.test.ts
  API Client
    \u2717 fetches data (10 ms)
    \u2717 handles errors (3 ms)

  \u25cf API Client > fetches data

    TypeError: fetch is not a function

      5 |   it('fetches data', async () => {
      6 |     const data = await fetchData('/api/test');
        |                       ^
      7 |     expect(data).toBeDefined();

      at fetchData (src/api/client.ts:12:10)
      at Object.<anonymous> (src/api/client.test.ts:6:23)

  \u25cf API Client > handles errors

    TypeError: fetch is not a function

      10 |   it('handles errors', async () => {
      11 |     await expect(fetchData('/bad')).rejects.toThrow();
         |           ^
      12 |   });

      at fetchData (src/api/client.ts:12:10)
      at Object.<anonymous> (src/api/client.test.ts:11:11)

Test Suites: 1 failed, 1 total
Tests:       2 failed, 2 total
Snapshots:   0 total
Time:        0.789 s
"""

JEST_SKIPPED = """\
 PASS  src/utils/skip.test.ts
  Skip tests
    \u2713 runs this test (2 ms)
    \u25cb skips this test
    \u25cb also skipped

Test Suites: 1 passed, 1 total
Tests:       2 skipped, 1 passed, 3 total
Snapshots:   0 total
Time:        0.5 s
"""


class TestJestOutput:
    """Tool-level tests for jest output parsing."""

    def test_jest_simple_pass(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": JEST_SIMPLE_PASS,
            "_call_id": "j1",
        }))
        data = json.loads(result.output)
        assert data["framework"] == "jest"
        assert data["summary"]["passed"] == 2
        assert data["summary"]["failed"] == 0
        assert len(data["records"]) == 2

    def test_jest_mixed_results(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": JEST_MIXED,
            "_call_id": "j2",
        }))
        data = json.loads(result.output)
        assert data["framework"] == "jest"
        assert data["summary"]["passed"] == 3
        assert data["summary"]["failed"] == 1

    def test_jest_failure_message(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": JEST_MIXED,
            "_call_id": "j3",
        }))
        data = json.loads(result.output)
        failed = [r for r in data["records"] if r["status"] == "failed"]
        assert len(failed) == 1
        assert "failure_message" in failed[0]
        assert "expect(received).toBe(expected)" in failed[0]["failure_message"]

    def test_jest_all_fail(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": JEST_ALL_FAIL,
            "_call_id": "j4",
        }))
        data = json.loads(result.output)
        assert data["summary"]["failed"] == 2
        assert data["summary"]["passed"] == 0
        failed = [r for r in data["records"] if r["status"] == "failed"]
        assert len(failed) == 2
        for r in failed:
            assert "TypeError: fetch is not a function" in r["failure_message"]

    def test_jest_skipped_tests(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": JEST_SKIPPED,
            "_call_id": "j5",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 1
        assert data["summary"]["skipped"] == 2

    def test_jest_with_explicit_hint(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": JEST_SIMPLE_PASS,
            "framework_hint": "jest",
            "_call_id": "j6",
        }))
        data = json.loads(result.output)
        assert data["framework"] == "jest"

    def test_jest_test_names_extracted(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": JEST_SIMPLE_PASS,
            "_call_id": "j7",
        }))
        data = json.loads(result.output)
        names = [r["name"] for r in data["records"]]
        assert "adds numbers" in names
        assert "subtracts numbers" in names


# ---------------------------------------------------------------------------
# Go test output parsing
# ---------------------------------------------------------------------------


GO_TEST_PASS = """\
=== RUN   TestAdd
--- PASS: TestAdd (0.00s)
=== RUN   TestMultiply
--- PASS: TestMultiply (0.00s)
PASS
ok      github.com/user/calc    0.003s
"""

GO_TEST_MIXED = """\
=== RUN   TestAdd
--- PASS: TestAdd (0.00s)
=== RUN   TestSubtract
    calc_test.go:15: expected 2, got 3
--- FAIL: TestSubtract (0.00s)
=== RUN   TestMultiply
--- PASS: TestMultiply (0.00s)
FAIL
exit status 1
FAIL    github.com/user/calc    0.015s
"""

GO_TEST_VERBOSE_FAIL = """\
=== RUN   TestParseConfig
    config_test.go:22: failed to parse config: invalid JSON
    config_test.go:23: expected error to be nil
--- FAIL: TestParseConfig (0.01s)
=== RUN   TestLoadConfig
--- PASS: TestLoadConfig (0.00s)
FAIL
exit status 1
FAIL    github.com/user/app/config    0.025s
"""

GO_TEST_SUBTESTS = """\
=== RUN   TestMath
=== RUN   TestMath/add
--- PASS: TestMath/add (0.00s)
=== RUN   TestMath/subtract
    math_test.go:20: expected 2, got 3
--- FAIL: TestMath/subtract (0.00s)
--- FAIL: TestMath (0.00s)
FAIL
exit status 1
FAIL    github.com/user/math    0.005s
"""

GO_TEST_SKIP = """\
=== RUN   TestFeature
    feature_test.go:10: skipping: requires network
--- SKIP: TestFeature (0.00s)
=== RUN   TestOther
--- PASS: TestOther (0.00s)
PASS
ok      github.com/user/app    0.002s
"""


class TestGoTestOutput:
    """Tool-level tests for go test output parsing."""

    def test_go_test_all_pass(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": GO_TEST_PASS,
            "_call_id": "g1",
        }))
        data = json.loads(result.output)
        assert data["framework"] == "go_test"
        assert data["summary"]["passed"] == 2
        assert data["summary"]["failed"] == 0

    def test_go_test_mixed(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": GO_TEST_MIXED,
            "_call_id": "g2",
        }))
        data = json.loads(result.output)
        assert data["framework"] == "go_test"
        assert data["summary"]["passed"] == 2
        assert data["summary"]["failed"] == 1

    def test_go_test_failure_message(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": GO_TEST_VERBOSE_FAIL,
            "_call_id": "g3",
        }))
        data = json.loads(result.output)
        failed = [r for r in data["records"] if r["status"] == "failed"]
        assert len(failed) == 1
        assert failed[0]["name"] == "TestParseConfig"
        assert "failure_message" in failed[0]
        assert "failed to parse config" in failed[0]["failure_message"]

    def test_go_test_names_extracted(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": GO_TEST_PASS,
            "_call_id": "g4",
        }))
        data = json.loads(result.output)
        names = [r["name"] for r in data["records"]]
        assert "TestAdd" in names
        assert "TestMultiply" in names

    def test_go_test_subtests(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": GO_TEST_SUBTESTS,
            "_call_id": "g5",
        }))
        data = json.loads(result.output)
        names = [r["name"] for r in data["records"]]
        assert "TestMath/add" in names
        assert "TestMath/subtract" in names

    def test_go_test_skip(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": GO_TEST_SKIP,
            "_call_id": "g6",
        }))
        data = json.loads(result.output)
        assert data["summary"]["skipped"] == 1
        assert data["summary"]["passed"] == 1

    def test_go_test_with_explicit_hint(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": GO_TEST_PASS,
            "framework_hint": "go_test",
            "_call_id": "g7",
        }))
        data = json.loads(result.output)
        assert data["framework"] == "go_test"

    def test_go_test_duration_extracted(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": GO_TEST_VERBOSE_FAIL,
            "_call_id": "g8",
        }))
        data = json.loads(result.output)
        # At least one record should have a duration
        records_with_duration = [
            r for r in data["records"] if r.get("duration_seconds") is not None
        ]
        assert len(records_with_duration) > 0


# ---------------------------------------------------------------------------
# Framework auto-detection
# ---------------------------------------------------------------------------


class TestFrameworkAutoDetect:
    def test_detects_pytest(self, tool: ParseTestOutputTool) -> None:
        output = (
            "============================= test session starts =============================\n"
            "tests/test_auth.py::test_login PASSED\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "d1",
        }))
        data = json.loads(result.output)
        assert data["framework"] == "pytest"

    def test_detects_jest(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": JEST_SIMPLE_PASS,
            "_call_id": "d2",
        }))
        data = json.loads(result.output)
        assert data["framework"] == "jest"

    def test_detects_go_test(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": GO_TEST_PASS,
            "_call_id": "d3",
        }))
        data = json.loads(result.output)
        assert data["framework"] == "go_test"

    def test_unknown_framework(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": "some random log output\nno test markers here\n",
            "_call_id": "d4",
        }))
        data = json.loads(result.output)
        assert data["framework"] == "unknown"


# ---------------------------------------------------------------------------
# Output structure validation
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Verify the JSON output structure matches the documented contract."""

    def test_output_has_required_fields(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": "tests/test_x.py::test_a PASSED\n",
            "_call_id": "s1",
        }))
        data = json.loads(result.output)
        assert "records" in data
        assert "truncated" in data
        assert "framework" in data
        assert "summary" in data

    def test_focused_summary_added_when_summary_fields_requested(
        self, tool: ParseTestOutputTool
    ) -> None:
        result = run_async(tool.execute({
            "raw_output": "tests/test_x.py::test_a PASSED\n",
            "summary_fields": ["failed", "passed", "iterations_done"],
            "_call_id": "s1b",
        }))
        data = json.loads(result.output)
        assert data["summary_fields"] == [
            "failed",
            "passed",
            "iterations_done",
        ]
        assert data["focused_summary"] == {
            "failed": 0,
            "passed": 1,
        }
        assert data["unmapped_summary_fields"] == ["iterations_done"]

    def test_summary_has_all_counts(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": "tests/test_x.py::test_a PASSED\n",
            "_call_id": "s2",
        }))
        data = json.loads(result.output)
        summary = data["summary"]
        assert "passed" in summary
        assert "failed" in summary
        assert "skipped" in summary
        assert "error" in summary
        assert "incomplete" in summary

    def test_record_has_required_fields(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": "tests/test_x.py::test_a PASSED\n",
            "_call_id": "s3",
        }))
        data = json.loads(result.output)
        assert len(data["records"]) == 1
        record = data["records"][0]
        assert "name" in record
        assert "status" in record
        assert "module" in record
        assert "failure_message" in record

    def test_passed_record_has_null_failure_message(
        self, tool: ParseTestOutputTool
    ) -> None:
        result = run_async(tool.execute({
            "raw_output": "tests/test_x.py::test_a PASSED\n",
            "_call_id": "s4",
        }))
        data = json.loads(result.output)
        assert data["records"][0]["failure_message"] is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_raw_output_defaults_empty(
        self, tool: ParseTestOutputTool
    ) -> None:
        result = run_async(tool.execute({
            "_call_id": "e1",
        }))
        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["records"] == []

    def test_call_id_extracted(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": "",
            "_call_id": "my-call-42",
        }))
        assert result.call_id == "my-call-42"

    def test_tool_name_in_result(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": "",
            "_call_id": "e3",
        }))
        assert result.tool_name == "parse_test_output"

    def test_invalid_summary_fields_returns_error(
        self, tool: ParseTestOutputTool
    ) -> None:
        result = run_async(tool.execute({
            "raw_output": "tests/test_x.py::test_a PASSED\n",
            "summary_fields": "passed,failed",
            "_call_id": "e4",
        }))
        assert result.status == ToolResultStatus.ERROR
        assert "summary_fields must be an array" in (result.error_message or "")


# ---------------------------------------------------------------------------
# OpenAI schema serialization
# ---------------------------------------------------------------------------


class TestOpenAISchema:
    def test_openai_schema_structure(self, tool: ParseTestOutputTool) -> None:
        schema = tool.spec.to_openai_function_schema()
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "parse_test_output"
        assert "description" in func
        assert "parameters" in func
        params = func["parameters"]
        assert params["type"] == "object"
        assert "raw_output" in params["properties"]
        assert "framework_hint" in params["properties"]
        assert "summary_fields" in params["properties"]


# ---------------------------------------------------------------------------
# Pytest: additional pass/fail/error/skip edge cases
# ---------------------------------------------------------------------------


class TestPytestPassVariants:
    """Pytest output: PASSED variants including XPASS, parametrized, class."""

    def test_xpass_maps_to_passed(self, tool: ParseTestOutputTool) -> None:
        """XPASS (unexpected pass) should report as 'passed' status."""
        output = (
            "tests/test_auth.py::test_known_bug XPASS\n"
            "============================== 1 passed in 0.01s ==============================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "pp1",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 1
        assert data["records"][0]["status"] == "passed"
        assert data["records"][0]["name"] == "test_known_bug"

    def test_parametrized_passed(self, tool: ParseTestOutputTool) -> None:
        """Parametrized test names with brackets should be preserved."""
        output = (
            "tests/test_math.py::test_add[1-2-3] PASSED\n"
            "tests/test_math.py::test_add[0-0-0] PASSED\n"
            "tests/test_math.py::test_add[-1-1-0] PASSED\n"
            "============================== 3 passed in 0.05s ==============================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "pp2",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 3
        names = [r["name"] for r in data["records"]]
        assert "test_add[1-2-3]" in names
        assert "test_add[0-0-0]" in names
        assert "test_add[-1-1-0]" in names

    def test_class_method_passed(self, tool: ParseTestOutputTool) -> None:
        """Class::method format test names should be fully preserved."""
        output = (
            "tests/test_auth.py::TestLogin::test_valid_credentials PASSED\n"
            "tests/test_auth.py::TestLogin::test_invalid_password FAILED\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "pp3",
        }))
        data = json.loads(result.output)
        assert len(data["records"]) == 2
        assert data["records"][0]["name"] == "TestLogin::test_valid_credentials"
        assert data["records"][0]["status"] == "passed"
        assert data["records"][1]["name"] == "TestLogin::test_invalid_password"
        assert data["records"][1]["status"] == "failed"

    def test_progress_percentages_ignored(self, tool: ParseTestOutputTool) -> None:
        """Progress indicators like [100%] should not affect parsing."""
        output = (
            "tests/test_a.py::test_one PASSED                               [  33%]\n"
            "tests/test_a.py::test_two PASSED                               [  66%]\n"
            "tests/test_a.py::test_three PASSED                             [ 100%]\n"
            "============================== 3 passed in 0.02s ==============================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "pp4",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 3
        assert len(data["records"]) == 3


class TestPytestFailVariants:
    """Pytest output: FAILED variants with different failure patterns."""

    def test_assertion_error_failure(self, tool: ParseTestOutputTool) -> None:
        """Failure with AssertionError traceback should capture message."""
        output = (
            "tests/test_calc.py::test_division FAILED\n"
            "\n"
            "=================================== FAILURES ===================================\n"
            "_________________________________ test_division ________________________________\n"
            "\n"
            "    def test_division():\n"
            ">       assert 10 / 3 == 3\n"
            "E       assert 3.3333333333333335 == 3\n"
            "\n"
            "tests/test_calc.py:5: AssertionError\n"
            "=========================== short test summary info ============================\n"
            "========================= 1 failed in 0.01s =========================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "pf1",
        }))
        data = json.loads(result.output)
        assert data["summary"]["failed"] == 1
        failed = data["records"][0]
        assert failed["status"] == "failed"
        assert failed["failure_message"] is not None
        assert "3.333" in failed["failure_message"]

    def test_multiple_modules_with_failures(
        self, tool: ParseTestOutputTool
    ) -> None:
        """Failures across different modules should each be tracked."""
        output = (
            "tests/test_auth.py::test_login FAILED\n"
            "tests/test_db.py::test_connect FAILED\n"
            "tests/test_api.py::test_health PASSED\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "pf2",
        }))
        data = json.loads(result.output)
        assert data["summary"]["failed"] == 2
        assert data["summary"]["passed"] == 1
        # Modules should be distinct
        modules = {r["module"] for r in data["records"]}
        assert "tests/test_auth.py" in modules
        assert "tests/test_db.py" in modules
        assert "tests/test_api.py" in modules

    def test_failed_record_without_failures_section(
        self, tool: ParseTestOutputTool
    ) -> None:
        """FAILED without FAILURES section: failure_message should be None."""
        output = "tests/test_x.py::test_broken FAILED\n"
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "pf3",
        }))
        data = json.loads(result.output)
        assert data["summary"]["failed"] == 1
        assert data["records"][0]["failure_message"] is None


class TestPytestErrorVariants:
    """Pytest output: ERROR status cases."""

    def test_single_error(self, tool: ParseTestOutputTool) -> None:
        """A single test with ERROR status."""
        output = (
            "tests/test_setup.py::test_db_init ERROR\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "pe1",
        }))
        data = json.loads(result.output)
        assert data["summary"]["error"] == 1
        assert data["records"][0]["status"] == "error"
        assert data["records"][0]["name"] == "test_db_init"

    def test_mixed_pass_and_error(self, tool: ParseTestOutputTool) -> None:
        """Tests with mixed PASSED and ERROR statuses."""
        output = (
            "tests/test_setup.py::test_db_init PASSED\n"
            "tests/test_setup.py::test_cache_init ERROR\n"
            "tests/test_setup.py::test_redis_init PASSED\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "pe2",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 2
        assert data["summary"]["error"] == 1
        assert data["records"][1]["status"] == "error"


class TestPytestSkipVariants:
    """Pytest output: SKIPPED and XFAIL cases."""

    def test_single_skip(self, tool: ParseTestOutputTool) -> None:
        """A single SKIPPED test."""
        output = (
            "tests/test_feature.py::test_new_flag SKIPPED\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "ps1",
        }))
        data = json.loads(result.output)
        assert data["summary"]["skipped"] == 1
        assert data["records"][0]["status"] == "skipped"

    def test_xfail_maps_to_skipped(self, tool: ParseTestOutputTool) -> None:
        """XFAIL (expected failure) should map to 'skipped' status."""
        output = (
            "tests/test_known.py::test_known_bug XFAIL\n"
            "============================== 1 passed in 0.01s ==============================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "ps2",
        }))
        data = json.loads(result.output)
        assert data["summary"]["skipped"] == 1
        assert data["records"][0]["status"] == "skipped"

    def test_all_skipped(self, tool: ParseTestOutputTool) -> None:
        """All tests skipped."""
        output = (
            "tests/test_feature.py::test_a SKIPPED\n"
            "tests/test_feature.py::test_b SKIPPED\n"
            "tests/test_feature.py::test_c SKIPPED\n"
            "============================== 3 skipped in 0.01s ==============================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "ps3",
        }))
        data = json.loads(result.output)
        assert data["summary"]["skipped"] == 3
        assert data["summary"]["passed"] == 0
        assert data["summary"]["failed"] == 0

    def test_mixed_skip_and_pass(self, tool: ParseTestOutputTool) -> None:
        """Mixture of PASSED, SKIPPED, and XFAIL results."""
        output = (
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_legacy SKIPPED\n"
            "tests/test_auth.py::test_known XFAIL\n"
            "tests/test_auth.py::test_logout PASSED\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "ps4",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 2
        assert data["summary"]["skipped"] == 2  # SKIPPED + XFAIL


class TestPytestShortFormat:
    """Pytest short-format output (dots) parsing through the tool."""

    def test_short_format_all_pass(self, tool: ParseTestOutputTool) -> None:
        """All-dot short output should produce all-passed records."""
        output = (
            "============================= test session starts =============================\n"
            "collected 3 items\n\n"
            "tests/test_auth.py ...                                                   [100%]\n\n"
            "============================== 3 passed in 0.05s ==============================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "sh1",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 3
        assert data["summary"]["failed"] == 0

    def test_short_format_mixed(self, tool: ParseTestOutputTool) -> None:
        """Mixed short format (.Fs.E) should produce correct status counts."""
        output = (
            "============================= test session starts =============================\n"
            "collected 5 items\n\n"
            "tests/test_auth.py .Fs.E                                                 [100%]\n\n"
            "========================= 1 failed, 1 error in 0.05s =========================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "sh2",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 2
        assert data["summary"]["failed"] == 1
        assert data["summary"]["skipped"] == 1
        assert data["summary"]["error"] == 1

    def test_short_format_record_names(self, tool: ParseTestOutputTool) -> None:
        """Short format records should use module#N naming convention."""
        output = (
            "tests/test_auth.py .F                                                    [100%]\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "sh3",
        }))
        data = json.loads(result.output)
        assert len(data["records"]) == 2
        # Short format generates module#N names
        names = [r["name"] for r in data["records"]]
        assert "tests/test_auth.py#1" in names
        assert "tests/test_auth.py#2" in names


class TestPytestTruncatedOutput:
    """Pytest output truncated mid-stream, tested through the tool."""

    def test_truncated_incomplete_detection(
        self, tool: ParseTestOutputTool
    ) -> None:
        """Output cut off mid-test should report truncated=True and incomplete."""
        output = (
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_slow"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "pt1",
        }))
        data = json.loads(result.output)
        assert data["truncated"] is True
        assert data["summary"]["incomplete"] >= 1
        incomplete = [r for r in data["records"] if r["status"] == "incomplete"]
        assert len(incomplete) >= 1

    def test_complete_run_not_truncated(
        self, tool: ParseTestOutputTool
    ) -> None:
        """Complete output with summary should report truncated=False."""
        output = (
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_logout PASSED\n\n"
            "============================== 2 passed in 0.01s ==============================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "pt2",
        }))
        data = json.loads(result.output)
        assert data["truncated"] is False
        assert data["summary"]["incomplete"] == 0


class TestPytestAnsiEscapes:
    """ANSI escape codes in pytest output, tested through the tool."""

    def test_ansi_colored_pass(self, tool: ParseTestOutputTool) -> None:
        """ANSI color codes around PASSED should be stripped cleanly."""
        output = (
            "tests/test_auth.py::test_login \x1b[32mPASSED\x1b[0m\n"
            "tests/test_auth.py::test_logout \x1b[31mFAILED\x1b[0m\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "an1",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1

    def test_ansi_in_session_header(self, tool: ParseTestOutputTool) -> None:
        """ANSI in session header should not prevent framework detection."""
        output = (
            "\x1b[1m============================= test session starts =============================\x1b[0m\n"
            "tests/test_x.py::test_a PASSED\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "an2",
        }))
        data = json.loads(result.output)
        assert data["framework"] == "pytest"
        assert data["summary"]["passed"] == 1


# ---------------------------------------------------------------------------
# Jest: additional pass/fail/skip edge cases
# ---------------------------------------------------------------------------


JEST_TRUNCATED = """\
 PASS  src/utils/math.test.ts
  Math utils
    \u2713 adds numbers (3 ms)
    \u2713 subtracts numbers (1 ms)

 FAIL  src/components/Form.test.tsx
  Form validation
    \u2717 validates email field (5 ms)
"""


class TestJestPassFailSkip:
    """Jest output additional pass/fail/skip coverage."""

    def test_jest_pass_only_summary_counts(
        self, tool: ParseTestOutputTool
    ) -> None:
        """Jest pass-only output should have zero fails and skips."""
        result = run_async(tool.execute({
            "raw_output": JEST_SIMPLE_PASS,
            "_call_id": "jp1",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 2
        assert data["summary"]["failed"] == 0
        assert data["summary"]["skipped"] == 0
        assert data["summary"]["error"] == 0

    def test_jest_mixed_counts(self, tool: ParseTestOutputTool) -> None:
        """Jest mixed output: 3 passed + 1 failed."""
        result = run_async(tool.execute({
            "raw_output": JEST_MIXED,
            "_call_id": "jp2",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 3
        assert data["summary"]["failed"] == 1
        assert data["summary"]["skipped"] == 0

    def test_jest_skipped_counts(self, tool: ParseTestOutputTool) -> None:
        """Jest skipped output: 1 passed + 2 skipped."""
        result = run_async(tool.execute({
            "raw_output": JEST_SKIPPED,
            "_call_id": "jp3",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 1
        assert data["summary"]["skipped"] == 2
        skipped = [r for r in data["records"] if r["status"] == "skipped"]
        assert len(skipped) == 2
        names = [r["name"] for r in skipped]
        assert "skips this test" in names
        assert "also skipped" in names

    def test_jest_all_fail_counts(self, tool: ParseTestOutputTool) -> None:
        """Jest all-fail output: 0 passed + 2 failed."""
        result = run_async(tool.execute({
            "raw_output": JEST_ALL_FAIL,
            "_call_id": "jp4",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 0
        assert data["summary"]["failed"] == 2

    def test_jest_truncated_output(self, tool: ParseTestOutputTool) -> None:
        """Jest output truncated before summary should report truncated."""
        result = run_async(tool.execute({
            "raw_output": JEST_TRUNCATED,
            "_call_id": "jp5",
        }))
        data = json.loads(result.output)
        assert data["truncated"] is True
        assert data["framework"] == "jest"

    def test_jest_module_in_records(self, tool: ParseTestOutputTool) -> None:
        """Jest records should include the suite module path."""
        result = run_async(tool.execute({
            "raw_output": JEST_SIMPLE_PASS,
            "_call_id": "jp6",
        }))
        data = json.loads(result.output)
        for record in data["records"]:
            assert record["module"] == "src/utils/math.test.ts"


# ---------------------------------------------------------------------------
# Go test: additional pass/fail/skip edge cases
# ---------------------------------------------------------------------------


GO_TEST_TRUNCATED = """\
=== RUN   TestAdd
--- PASS: TestAdd (0.00s)
=== RUN   TestSubtract
    calc_test.go:15: expected 2, got 3
"""

GO_TEST_ALL_SKIP = """\
=== RUN   TestFeatureA
    feature_test.go:10: skipping: requires network
--- SKIP: TestFeatureA (0.00s)
=== RUN   TestFeatureB
    feature_test.go:15: skipping: CI only
--- SKIP: TestFeatureB (0.00s)
PASS
ok      github.com/user/app    0.001s
"""


class TestGoTestPassFailSkip:
    """Go test output additional pass/fail/skip coverage."""

    def test_go_test_all_pass_counts(
        self, tool: ParseTestOutputTool
    ) -> None:
        """Go test all-pass should have zero fail/skip counts."""
        result = run_async(tool.execute({
            "raw_output": GO_TEST_PASS,
            "_call_id": "gp1",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 2
        assert data["summary"]["failed"] == 0
        assert data["summary"]["skipped"] == 0
        assert data["summary"]["error"] == 0

    def test_go_test_mixed_counts(self, tool: ParseTestOutputTool) -> None:
        """Go test mixed output: 2 passed + 1 failed."""
        result = run_async(tool.execute({
            "raw_output": GO_TEST_MIXED,
            "_call_id": "gp2",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 2
        assert data["summary"]["failed"] == 1

    def test_go_test_all_skip(self, tool: ParseTestOutputTool) -> None:
        """Go test all-skip output: 0 passed + 2 skipped."""
        result = run_async(tool.execute({
            "raw_output": GO_TEST_ALL_SKIP,
            "_call_id": "gp3",
        }))
        data = json.loads(result.output)
        assert data["summary"]["skipped"] == 2
        assert data["summary"]["passed"] == 0

    def test_go_test_truncated_output(
        self, tool: ParseTestOutputTool
    ) -> None:
        """Go test output truncated mid-test should report truncated."""
        result = run_async(tool.execute({
            "raw_output": GO_TEST_TRUNCATED,
            "_call_id": "gp4",
        }))
        data = json.loads(result.output)
        assert data["truncated"] is True
        assert data["framework"] == "go_test"

    def test_go_test_failure_message_multiline(
        self, tool: ParseTestOutputTool
    ) -> None:
        """Go test failure with multiple log lines should concatenate them."""
        result = run_async(tool.execute({
            "raw_output": GO_TEST_VERBOSE_FAIL,
            "_call_id": "gp5",
        }))
        data = json.loads(result.output)
        failed = [r for r in data["records"] if r["status"] == "failed"]
        assert len(failed) == 1
        msg = failed[0]["failure_message"]
        assert "failed to parse config" in msg
        assert "expected error to be nil" in msg

    def test_go_test_subtest_pass_fail(
        self, tool: ParseTestOutputTool
    ) -> None:
        """Go test subtests should report individual subtest results."""
        result = run_async(tool.execute({
            "raw_output": GO_TEST_SUBTESTS,
            "_call_id": "gp6",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] >= 1
        assert data["summary"]["failed"] >= 1
        names = [r["name"] for r in data["records"]]
        assert "TestMath/add" in names
        assert "TestMath/subtract" in names


# ---------------------------------------------------------------------------
# total_lines_parsed field validation
# ---------------------------------------------------------------------------


class TestTotalLinesParsed:
    """Verify total_lines_parsed is tracked correctly."""

    def test_single_line_output(self, tool: ParseTestOutputTool) -> None:
        output = "tests/test_x.py::test_a PASSED\n"
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "tlp1",
        }))
        data = json.loads(result.output)
        assert data["total_lines_parsed"] >= 1

    def test_multiline_output(self, tool: ParseTestOutputTool) -> None:
        output = (
            "tests/test_a.py::test_one PASSED\n"
            "tests/test_a.py::test_two FAILED\n"
            "tests/test_a.py::test_three SKIPPED\n"
            "tests/test_a.py::test_four ERROR\n"
            "\n"
            "============================== 1 failed in 0.05s ==============================\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "tlp2",
        }))
        data = json.loads(result.output)
        # Should count all lines (including blanks)
        assert data["total_lines_parsed"] >= 6

    def test_empty_output_zero_lines(self, tool: ParseTestOutputTool) -> None:
        result = run_async(tool.execute({
            "raw_output": "",
            "_call_id": "tlp3",
        }))
        data = json.loads(result.output)
        assert data["total_lines_parsed"] == 0


# ---------------------------------------------------------------------------
# Default call_id and framework_hint edge cases
# ---------------------------------------------------------------------------


class TestDefaultCallId:
    """Verify behavior when optional args are missing."""

    def test_default_call_id(self, tool: ParseTestOutputTool) -> None:
        """When _call_id is not provided, should use default."""
        result = run_async(tool.execute({
            "raw_output": "tests/test_x.py::test_a PASSED\n",
        }))
        assert result.call_id == "parse_test_output"
        assert result.status == ToolResultStatus.SUCCESS

    def test_default_framework_hint(self, tool: ParseTestOutputTool) -> None:
        """When framework_hint is not provided, should default to 'auto'."""
        output = (
            "============================= test session starts =============================\n"
            "tests/test_x.py::test_a PASSED\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "dh1",
        }))
        data = json.loads(result.output)
        assert data["framework"] == "pytest"

    def test_invalid_framework_hint_falls_to_auto(
        self, tool: ParseTestOutputTool
    ) -> None:
        """Unrecognized framework_hint should fall through to AUTO detection."""
        output = (
            "============================= test session starts =============================\n"
            "tests/test_x.py::test_a PASSED\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "framework_hint": "nonexistent_framework",
            "_call_id": "dh2",
        }))
        data = json.loads(result.output)
        # Should fall back to auto-detect (pytest markers present)
        assert data["framework"] == "pytest"


# ---------------------------------------------------------------------------
# Error branch: parse failure
# ---------------------------------------------------------------------------


class TestParseErrorBranch:
    """Cover the exception-handling branch in execute()."""

    def test_import_error_returns_error_result(
        self, tool: ParseTestOutputTool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the underlying parser raises, tool returns ERROR status."""
        import jules_daemon.agent.tools.parse_test_output as mod

        original_execute = mod.ParseTestOutputTool.execute

        async def patched_execute(self, args):
            # Force an exception by making parse_interrupted_output raise
            raw = args.get("raw_output", "")
            if raw == "__FORCE_ERROR__":
                raise RuntimeError("Simulated parser crash")
            return await original_execute(self, args)

        # Instead of patching execute, patch the imported parser
        import jules_daemon.monitor.test_output_parser as parser_mod

        original_parse = parser_mod.parse_interrupted_output

        def broken_parse(*a, **kw):
            raise RuntimeError("Simulated parser crash")

        monkeypatch.setattr(parser_mod, "parse_interrupted_output", broken_parse)

        result = run_async(tool.execute({
            "raw_output": "tests/test_x.py::test_a PASSED\n",
            "_call_id": "err1",
        }))
        assert result.status == ToolResultStatus.ERROR
        assert result.error_message is not None
        assert "Failed to parse test output" in result.error_message
        assert "Simulated parser crash" in result.error_message
        assert result.output == ""

    def test_value_error_returns_error_result(
        self, tool: ParseTestOutputTool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ValueError from parser also returns ERROR status."""
        import jules_daemon.monitor.test_output_parser as parser_mod

        def broken_parse(*a, **kw):
            raise ValueError("Bad output format")

        monkeypatch.setattr(parser_mod, "parse_interrupted_output", broken_parse)

        result = run_async(tool.execute({
            "raw_output": "some test output",
            "_call_id": "err2",
        }))
        assert result.status == ToolResultStatus.ERROR
        assert "Bad output format" in result.error_message


# ---------------------------------------------------------------------------
# All four statuses in one run (integration)
# ---------------------------------------------------------------------------


class TestAllStatusesInOneRun:
    """Verify all four terminal statuses in a single parse call."""

    def test_pytest_all_four_statuses(
        self, tool: ParseTestOutputTool
    ) -> None:
        """A run with PASSED, FAILED, ERROR, and SKIPPED tests."""
        output = (
            "tests/test_all.py::test_pass PASSED\n"
            "tests/test_all.py::test_fail FAILED\n"
            "tests/test_all.py::test_err ERROR\n"
            "tests/test_all.py::test_skip SKIPPED\n"
            "\n"
            "========== 1 failed, 1 error, 1 skipped, 1 passed in 0.05s ==========\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "all1",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1
        assert data["summary"]["error"] == 1
        assert data["summary"]["skipped"] == 1
        assert data["summary"]["incomplete"] == 0
        assert data["truncated"] is False
        # Verify status values are lowercase strings
        statuses = [r["status"] for r in data["records"]]
        assert statuses == ["passed", "failed", "error", "skipped"]

    def test_go_test_pass_fail_skip_in_one(
        self, tool: ParseTestOutputTool
    ) -> None:
        """Go test with PASS, FAIL, SKIP results in one run."""
        output = (
            "=== RUN   TestA\n"
            "--- PASS: TestA (0.00s)\n"
            "=== RUN   TestB\n"
            "    b_test.go:10: expected 1, got 2\n"
            "--- FAIL: TestB (0.01s)\n"
            "=== RUN   TestC\n"
            "    c_test.go:5: skipping: not supported\n"
            "--- SKIP: TestC (0.00s)\n"
            "FAIL\n"
            "exit status 1\n"
            "FAIL    github.com/user/pkg    0.015s\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "all2",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1
        assert data["summary"]["skipped"] == 1
        assert data["framework"] == "go_test"

    def test_jest_pass_fail_skip_in_one(
        self, tool: ParseTestOutputTool
    ) -> None:
        """Jest output with pass, fail, and skip in one suite."""
        output = (
            " FAIL  src/components/Form.test.tsx\n"
            "  Form tests\n"
            "    \u2713 renders form (2 ms)\n"
            "    \u2717 validates input (5 ms)\n"
            "    \u25cb skipped pending test\n"
            "\n"
            "  \u25cf Form tests > validates input\n"
            "\n"
            "    expect(received).toBe(expected)\n"
            "    Expected: true\n"
            "    Received: false\n"
            "\n"
            "Test Suites: 1 failed, 1 total\n"
            "Tests:       1 failed, 1 skipped, 1 passed, 3 total\n"
        )
        result = run_async(tool.execute({
            "raw_output": output,
            "_call_id": "all3",
        }))
        data = json.loads(result.output)
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1
        assert data["summary"]["skipped"] == 1
        assert data["framework"] == "jest"
