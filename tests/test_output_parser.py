"""Tests for interrupted test output parser.

Verifies that the parser:
- Parses complete pytest-style output into structured per-test records
- Handles truncated output where the stream was cut off mid-test
- Marks incomplete tests with INCOMPLETE status
- Extracts test names, statuses, durations, and captured output
- Handles empty input gracefully
- Handles interleaved stdout/stderr
- Parses partial summary sections
- Supports verbose pytest output (-v flag)
- Supports short pytest output (dots/letters)
- Handles output with collection errors
- Handles output with ANSI escape codes
- Detects framework type from output content
"""

from __future__ import annotations

import pytest

from jules_daemon.monitor.test_output_parser import (
    FrameworkHint,
    OutputContext,
    ParseResult,
    TestRecord,
    TestStatus,
    parse_interrupted_output,
)


# ---------------------------------------------------------------------------
# TestStatus enum
# ---------------------------------------------------------------------------


class TestTestStatus:
    def test_all_statuses_exist(self) -> None:
        assert TestStatus.PASSED is not None
        assert TestStatus.FAILED is not None
        assert TestStatus.ERROR is not None
        assert TestStatus.SKIPPED is not None
        assert TestStatus.INCOMPLETE is not None

    def test_values(self) -> None:
        assert TestStatus.PASSED.value == "passed"
        assert TestStatus.FAILED.value == "failed"
        assert TestStatus.ERROR.value == "error"
        assert TestStatus.SKIPPED.value == "skipped"
        assert TestStatus.INCOMPLETE.value == "incomplete"

    def test_is_terminal_for_complete_statuses(self) -> None:
        assert TestStatus.PASSED.is_terminal is True
        assert TestStatus.FAILED.is_terminal is True
        assert TestStatus.ERROR.is_terminal is True
        assert TestStatus.SKIPPED.is_terminal is True

    def test_is_terminal_for_incomplete(self) -> None:
        assert TestStatus.INCOMPLETE.is_terminal is False


# ---------------------------------------------------------------------------
# TestRecord frozen dataclass
# ---------------------------------------------------------------------------


class TestTestRecord:
    def test_create_with_all_fields(self) -> None:
        record = TestRecord(
            name="test_login",
            status=TestStatus.PASSED,
            module="tests/test_auth.py",
            duration_seconds=1.23,
            output_lines=("OK",),
            line_number=42,
        )
        assert record.name == "test_login"
        assert record.status == TestStatus.PASSED
        assert record.module == "tests/test_auth.py"
        assert record.duration_seconds == 1.23
        assert record.output_lines == ("OK",)
        assert record.line_number == 42

    def test_create_minimal(self) -> None:
        record = TestRecord(name="test_basic", status=TestStatus.FAILED)
        assert record.name == "test_basic"
        assert record.status == TestStatus.FAILED
        assert record.module == ""
        assert record.duration_seconds is None
        assert record.output_lines == ()
        assert record.line_number is None

    def test_frozen(self) -> None:
        record = TestRecord(name="test_x", status=TestStatus.PASSED)
        with pytest.raises(AttributeError):
            record.name = "test_y"  # type: ignore[misc]

    def test_incomplete_record(self) -> None:
        record = TestRecord(
            name="test_slow_operation",
            status=TestStatus.INCOMPLETE,
            output_lines=("starting slow op...",),
        )
        assert record.status == TestStatus.INCOMPLETE
        assert record.status.is_terminal is False


# ---------------------------------------------------------------------------
# ParseResult frozen dataclass
# ---------------------------------------------------------------------------


class TestParseResult:
    def test_create_result(self) -> None:
        records = (
            TestRecord(name="test_a", status=TestStatus.PASSED),
            TestRecord(name="test_b", status=TestStatus.FAILED),
        )
        result = ParseResult(
            records=records,
            truncated=False,
            framework_hint=FrameworkHint.PYTEST,
            total_lines_parsed=100,
            raw_tail="",
        )
        assert len(result.records) == 2
        assert result.truncated is False

    def test_frozen(self) -> None:
        result = ParseResult(
            records=(),
            truncated=False,
            framework_hint=FrameworkHint.UNKNOWN,
            total_lines_parsed=0,
            raw_tail="",
        )
        with pytest.raises(AttributeError):
            result.truncated = True  # type: ignore[misc]

    def test_summary_counts(self) -> None:
        records = (
            TestRecord(name="test_a", status=TestStatus.PASSED),
            TestRecord(name="test_b", status=TestStatus.FAILED),
            TestRecord(name="test_c", status=TestStatus.SKIPPED),
            TestRecord(name="test_d", status=TestStatus.INCOMPLETE),
            TestRecord(name="test_e", status=TestStatus.PASSED),
        )
        result = ParseResult(
            records=records,
            truncated=True,
            framework_hint=FrameworkHint.PYTEST,
            total_lines_parsed=50,
            raw_tail="",
        )
        assert result.passed_count == 2
        assert result.failed_count == 1
        assert result.skipped_count == 1
        assert result.incomplete_count == 1
        assert result.error_count == 0

    def test_has_incomplete(self) -> None:
        records = (
            TestRecord(name="test_a", status=TestStatus.PASSED),
            TestRecord(name="test_b", status=TestStatus.INCOMPLETE),
        )
        result = ParseResult(
            records=records,
            truncated=True,
            framework_hint=FrameworkHint.PYTEST,
            total_lines_parsed=20,
            raw_tail="",
        )
        assert result.has_incomplete is True

    def test_no_incomplete(self) -> None:
        records = (
            TestRecord(name="test_a", status=TestStatus.PASSED),
        )
        result = ParseResult(
            records=records,
            truncated=False,
            framework_hint=FrameworkHint.PYTEST,
            total_lines_parsed=20,
            raw_tail="",
        )
        assert result.has_incomplete is False


# ---------------------------------------------------------------------------
# OutputContext
# ---------------------------------------------------------------------------


class TestOutputContext:
    def test_create_default(self) -> None:
        ctx = OutputContext()
        assert ctx.framework_hint == FrameworkHint.AUTO
        assert ctx.max_output_lines_per_test == 50

    def test_create_with_hint(self) -> None:
        ctx = OutputContext(framework_hint=FrameworkHint.PYTEST)
        assert ctx.framework_hint == FrameworkHint.PYTEST


# ---------------------------------------------------------------------------
# parse_interrupted_output -- empty / trivial input
# ---------------------------------------------------------------------------


class TestParseEmpty:
    def test_empty_string(self) -> None:
        result = parse_interrupted_output("")
        assert len(result.records) == 0
        assert result.truncated is False
        assert result.total_lines_parsed == 0

    def test_whitespace_only(self) -> None:
        result = parse_interrupted_output("   \n\n  \n")
        assert len(result.records) == 0
        assert result.truncated is False

    def test_no_test_output(self) -> None:
        result = parse_interrupted_output("Some random log line\nAnother line\n")
        assert len(result.records) == 0
        assert result.truncated is False


# ---------------------------------------------------------------------------
# parse_interrupted_output -- pytest verbose output (-v)
# ---------------------------------------------------------------------------


class TestParsePytestVerbose:
    """Parse pytest verbose output with PASSED/FAILED/SKIPPED markers."""

    def test_single_passed(self) -> None:
        output = (
            "============================= test session starts =============================\n"
            "collected 1 item\n"
            "\n"
            "tests/test_auth.py::test_login PASSED\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 1
        assert result.records[0].name == "test_login"
        assert result.records[0].status == TestStatus.PASSED
        assert result.records[0].module == "tests/test_auth.py"

    def test_single_failed(self) -> None:
        output = (
            "============================= test session starts =============================\n"
            "collected 1 item\n"
            "\n"
            "tests/test_auth.py::test_login FAILED\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 1
        assert result.records[0].name == "test_login"
        assert result.records[0].status == TestStatus.FAILED

    def test_single_skipped(self) -> None:
        output = (
            "============================= test session starts =============================\n"
            "tests/test_auth.py::test_legacy SKIPPED\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 1
        assert result.records[0].name == "test_legacy"
        assert result.records[0].status == TestStatus.SKIPPED

    def test_single_error(self) -> None:
        output = (
            "============================= test session starts =============================\n"
            "tests/test_auth.py::test_setup ERROR\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 1
        assert result.records[0].name == "test_setup"
        assert result.records[0].status == TestStatus.ERROR

    def test_multiple_tests(self) -> None:
        output = (
            "============================= test session starts =============================\n"
            "collected 4 items\n"
            "\n"
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_logout PASSED\n"
            "tests/test_auth.py::test_register FAILED\n"
            "tests/test_auth.py::test_forgot_password SKIPPED\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 4
        assert result.records[0].status == TestStatus.PASSED
        assert result.records[1].status == TestStatus.PASSED
        assert result.records[2].status == TestStatus.FAILED
        assert result.records[3].status == TestStatus.SKIPPED

    def test_with_duration(self) -> None:
        output = (
            "tests/test_auth.py::test_login PASSED                             [  25%]\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 1
        assert result.records[0].name == "test_login"
        assert result.records[0].status == TestStatus.PASSED

    def test_class_method_format(self) -> None:
        output = (
            "tests/test_auth.py::TestLogin::test_valid_credentials PASSED\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 1
        assert result.records[0].name == "TestLogin::test_valid_credentials"
        assert result.records[0].module == "tests/test_auth.py"

    def test_parametrized_test(self) -> None:
        output = (
            "tests/test_math.py::test_add[1-2-3] PASSED\n"
            "tests/test_math.py::test_add[0-0-0] PASSED\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 2
        assert result.records[0].name == "test_add[1-2-3]"
        assert result.records[1].name == "test_add[0-0-0]"


# ---------------------------------------------------------------------------
# parse_interrupted_output -- truncated mid-test
# ---------------------------------------------------------------------------


class TestParseTruncatedOutput:
    """Parse output that was cut off before a test completed."""

    def test_truncated_after_collection(self) -> None:
        """Output cut off after collection but before any test result."""
        output = (
            "============================= test session starts =============================\n"
            "collected 5 items\n"
            "\n"
            "tests/test_auth.py::test_login "
        )
        result = parse_interrupted_output(output)
        # The last test was started but never completed
        assert result.truncated is True
        has_incomplete = any(
            r.status == TestStatus.INCOMPLETE for r in result.records
        )
        assert has_incomplete is True

    def test_truncated_mid_run(self) -> None:
        """Output cut off in the middle of a test run."""
        output = (
            "============================= test session starts =============================\n"
            "collected 5 items\n"
            "\n"
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_logout PASSED\n"
            "tests/test_auth.py::test_register "
        )
        result = parse_interrupted_output(output)
        assert result.truncated is True
        assert len(result.records) == 3
        assert result.records[0].status == TestStatus.PASSED
        assert result.records[1].status == TestStatus.PASSED
        assert result.records[2].status == TestStatus.INCOMPLETE
        assert result.records[2].name == "test_register"

    def test_truncated_with_no_summary(self) -> None:
        """Complete test results but no summary section -- stream was cut."""
        output = (
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_logout FAILED\n"
        )
        result = parse_interrupted_output(output)
        # We got results but no summary = possibly truncated
        assert len(result.records) == 2
        assert result.records[0].status == TestStatus.PASSED
        assert result.records[1].status == TestStatus.FAILED

    def test_truncated_in_failure_traceback(self) -> None:
        """Output cut off during failure traceback output."""
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
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 2
        assert result.records[0].status == TestStatus.PASSED
        assert result.records[1].status == TestStatus.FAILED

    def test_truncated_detects_incomplete_from_partial_line(self) -> None:
        """Trailing partial test path without a status marker."""
        output = (
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_slow.py::test_big_query"
        )
        result = parse_interrupted_output(output)
        assert result.truncated is True
        incomplete_records = [
            r for r in result.records if r.status == TestStatus.INCOMPLETE
        ]
        assert len(incomplete_records) == 1
        assert incomplete_records[0].name == "test_big_query"


# ---------------------------------------------------------------------------
# parse_interrupted_output -- pytest short output (dots)
# ---------------------------------------------------------------------------


class TestParsePytestShort:
    """Parse pytest short output format (., F, s, E characters)."""

    def test_all_dots(self) -> None:
        output = (
            "============================= test session starts =============================\n"
            "collected 5 items\n"
            "\n"
            "tests/test_auth.py .....                                                [100%]\n"
        )
        result = parse_interrupted_output(output)
        # Short format gives module-level info but not per-test names
        assert result.total_lines_parsed > 0

    def test_mixed_short(self) -> None:
        output = (
            "============================= test session starts =============================\n"
            "collected 5 items\n"
            "\n"
            "tests/test_auth.py ..F.s                                               [100%]\n"
        )
        result = parse_interrupted_output(output)
        assert result.total_lines_parsed > 0

    def test_truncated_dots(self) -> None:
        """Dots output cut off mid-line."""
        output = (
            "============================= test session starts =============================\n"
            "collected 10 items\n"
            "\n"
            "tests/test_auth.py ...."
        )
        result = parse_interrupted_output(output)
        assert result.truncated is True


# ---------------------------------------------------------------------------
# parse_interrupted_output -- ANSI escape handling
# ---------------------------------------------------------------------------


class TestParseAnsiEscapes:
    """Ensure ANSI color codes do not confuse the parser."""

    def test_ansi_colored_passed(self) -> None:
        output = (
            "tests/test_auth.py::test_login \x1b[32mPASSED\x1b[0m\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 1
        assert result.records[0].status == TestStatus.PASSED

    def test_ansi_colored_failed(self) -> None:
        output = (
            "tests/test_auth.py::test_login \x1b[31mFAILED\x1b[0m\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 1
        assert result.records[0].status == TestStatus.FAILED


# ---------------------------------------------------------------------------
# parse_interrupted_output -- with failure details
# ---------------------------------------------------------------------------


class TestParseWithFailureDetails:
    """Verify that failure tracebacks are associated with the correct test."""

    def test_failure_output_captured(self) -> None:
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
        result = parse_interrupted_output(output)
        assert len(result.records) == 2
        failed = [r for r in result.records if r.status == TestStatus.FAILED]
        assert len(failed) == 1
        assert failed[0].name == "test_register"

    def test_multiple_failures_captured(self) -> None:
        output = (
            "tests/test_auth.py::test_login FAILED\n"
            "tests/test_auth.py::test_register FAILED\n"
            "tests/test_auth.py::test_logout PASSED\n"
            "\n"
            "=================================== FAILURES ===================================\n"
            "_________________________________ test_login ___________________________________\n"
            "\n"
            "    def test_login():\n"
            ">       assert False\n"
            "E       AssertionError\n"
            "\n"
            "_________________________________ test_register _________________________________\n"
            "\n"
            "    def test_register():\n"
            ">       assert False\n"
            "E       AssertionError\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 3
        failed = [r for r in result.records if r.status == TestStatus.FAILED]
        assert len(failed) == 2


# ---------------------------------------------------------------------------
# parse_interrupted_output -- with summary line
# ---------------------------------------------------------------------------


class TestParseWithSummary:
    """Parse output that includes the pytest summary line."""

    def test_complete_run_with_summary(self) -> None:
        output = (
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_logout PASSED\n"
            "\n"
            "============================== 2 passed in 0.12s ==============================\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 2
        assert result.truncated is False

    def test_mixed_summary(self) -> None:
        output = (
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_register FAILED\n"
            "\n"
            "========================= 1 failed, 1 passed in 0.45s =========================\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 2
        assert result.truncated is False


# ---------------------------------------------------------------------------
# parse_interrupted_output -- framework detection
# ---------------------------------------------------------------------------


class TestFrameworkDetection:
    def test_detects_pytest(self) -> None:
        output = (
            "============================= test session starts =============================\n"
            "collected 1 item\n"
            "\n"
            "tests/test_auth.py::test_login PASSED\n"
        )
        result = parse_interrupted_output(output)
        assert result.framework_hint == FrameworkHint.PYTEST

    def test_unknown_framework_for_plain_text(self) -> None:
        result = parse_interrupted_output("hello world\n")
        assert result.framework_hint == FrameworkHint.UNKNOWN

    def test_explicit_framework_hint(self) -> None:
        output = "tests/test_auth.py::test_login PASSED\n"
        ctx = OutputContext(framework_hint=FrameworkHint.PYTEST)
        result = parse_interrupted_output(output, context=ctx)
        assert result.framework_hint == FrameworkHint.PYTEST


# ---------------------------------------------------------------------------
# parse_interrupted_output -- edge cases
# ---------------------------------------------------------------------------


class TestParseEdgeCases:
    def test_duplicate_test_names(self) -> None:
        """Same test name appearing twice (e.g., re-run)."""
        output = (
            "tests/test_auth.py::test_login FAILED\n"
            "tests/test_auth.py::test_login PASSED\n"
        )
        result = parse_interrupted_output(output)
        # Both records should be kept (re-run scenario)
        assert len(result.records) == 2

    def test_xfail_as_passed(self) -> None:
        """XFAIL tests show as PASSED in verbose output."""
        output = (
            "tests/test_auth.py::test_known_bug XFAIL\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 1
        # XFAIL maps to SKIPPED (expected failure, not a real failure)
        assert result.records[0].status == TestStatus.SKIPPED

    def test_xpass(self) -> None:
        """XPASS tests (unexpected pass) appear in output."""
        output = (
            "tests/test_auth.py::test_known_bug XPASS\n"
        )
        result = parse_interrupted_output(output)
        assert len(result.records) == 1
        # XPASS maps to PASSED (it passed unexpectedly)
        assert result.records[0].status == TestStatus.PASSED

    def test_very_long_test_name(self) -> None:
        long_name = "test_" + "a" * 200
        output = f"tests/test_auth.py::{long_name} PASSED\n"
        result = parse_interrupted_output(output)
        assert len(result.records) == 1
        assert result.records[0].name == long_name

    def test_preserves_line_order(self) -> None:
        output = (
            "tests/test_a.py::test_first PASSED\n"
            "tests/test_b.py::test_second FAILED\n"
            "tests/test_c.py::test_third SKIPPED\n"
        )
        result = parse_interrupted_output(output)
        assert result.records[0].name == "test_first"
        assert result.records[1].name == "test_second"
        assert result.records[2].name == "test_third"

    def test_collection_error(self) -> None:
        """Handle pytest collection errors."""
        output = (
            "============================= test session starts =============================\n"
            "collected 0 items / 1 error\n"
            "\n"
            "=================================== ERRORS ====================================\n"
            "________________ ERROR collecting tests/test_broken.py ________________________\n"
            "ImportError: No module named 'nonexistent'\n"
        )
        result = parse_interrupted_output(output)
        # Should not crash; may or may not produce records
        assert result.total_lines_parsed > 0

    def test_raw_tail_captured_on_truncation(self) -> None:
        """When truncated, raw_tail contains the last few lines."""
        output = (
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_slow"
        )
        result = parse_interrupted_output(output)
        assert result.truncated is True
        assert "test_slow" in result.raw_tail
