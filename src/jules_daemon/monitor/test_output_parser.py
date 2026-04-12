"""Interrupted test output parser for truncated or incomplete test streams.

Parses incomplete or truncated test output (stdout/stderr captured before
interruption) into structured per-test records, marking incomplete tests
explicitly with INCOMPLETE status.

Supported frameworks:
- pytest verbose output (-v flag): ``module::test_name STATUS``
- pytest short output (dots): ``.FsE`` characters per module
- Auto-detection from output content

The parser handles:
- Complete test results (PASSED, FAILED, ERROR, SKIPPED, XFAIL, XPASS)
- Truncated output where the stream was cut off mid-test
- ANSI escape codes (stripped before parsing)
- Parametrized test names (e.g., ``test_add[1-2-3]``)
- Class-method format (e.g., ``TestLogin::test_valid_credentials``)
- Failure traceback sections
- Summary lines

This module is a pure function layer -- it takes raw output text and
returns structured results. No side effects, no disk I/O.

Usage:
    from jules_daemon.monitor.test_output_parser import parse_interrupted_output

    result = parse_interrupted_output(raw_output)
    for record in result.records:
        if record.status == TestStatus.INCOMPLETE:
            print(f"Test {record.name} was interrupted")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = [
    "FrameworkHint",
    "OutputContext",
    "ParseResult",
    "TestRecord",
    "TestStatus",
    "parse_interrupted_output",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestStatus(Enum):
    """Status of a single test record extracted from output.

    PASSED, FAILED, ERROR, SKIPPED are terminal statuses indicating the
    test completed and produced a definitive result.

    INCOMPLETE means the test was started but the output stream was
    interrupted before a result marker appeared.
    """

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    INCOMPLETE = "incomplete"

    @property
    def is_terminal(self) -> bool:
        """True if this status represents a completed test."""
        return self != TestStatus.INCOMPLETE


class FrameworkHint(Enum):
    """Detected or specified test framework.

    AUTO: Let the parser auto-detect from output content.
    PYTEST: Python pytest framework.
    JEST: JavaScript/TypeScript Jest framework.
    GO_TEST: Go testing framework (go test).
    UNKNOWN: Could not detect the framework.
    """

    AUTO = "auto"
    PYTEST = "pytest"
    JEST = "jest"
    GO_TEST = "go_test"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestRecord:
    """Structured record for a single test extracted from output.

    Attributes:
        name: Test function/method name (e.g., "test_login").
        status: Terminal status or INCOMPLETE if the test was interrupted.
        module: Source file path (e.g., "tests/test_auth.py").
        duration_seconds: Execution duration if available.
        output_lines: Captured output lines associated with this test.
        line_number: 0-based line number in the output where this test appeared.
        failure_message: Extracted failure/error message for failed tests.
            None for passing or skipped tests.
    """

    name: str
    status: TestStatus
    module: str = ""
    duration_seconds: Optional[float] = None
    output_lines: tuple[str, ...] = ()
    line_number: Optional[int] = None
    failure_message: Optional[str] = None


@dataclass(frozen=True)
class OutputContext:
    """Configuration for the output parser.

    Attributes:
        framework_hint: Which framework to assume, or AUTO to detect.
        max_output_lines_per_test: Cap on captured output per test.
    """

    framework_hint: FrameworkHint = FrameworkHint.AUTO
    max_output_lines_per_test: int = 50


@dataclass(frozen=True)
class ParseResult:
    """Immutable result of parsing interrupted test output.

    Attributes:
        records: Tuple of TestRecord instances in order of appearance.
        truncated: True if the output appears to have been cut off.
        framework_hint: Detected or specified framework.
        total_lines_parsed: Number of lines processed from the output.
        raw_tail: Last few lines of the raw output (for debugging).
    """

    records: tuple[TestRecord, ...]
    truncated: bool
    framework_hint: FrameworkHint
    total_lines_parsed: int
    raw_tail: str

    @property
    def passed_count(self) -> int:
        """Number of tests that passed."""
        return sum(1 for r in self.records if r.status == TestStatus.PASSED)

    @property
    def failed_count(self) -> int:
        """Number of tests that failed."""
        return sum(1 for r in self.records if r.status == TestStatus.FAILED)

    @property
    def error_count(self) -> int:
        """Number of tests with errors."""
        return sum(1 for r in self.records if r.status == TestStatus.ERROR)

    @property
    def skipped_count(self) -> int:
        """Number of tests that were skipped."""
        return sum(1 for r in self.records if r.status == TestStatus.SKIPPED)

    @property
    def incomplete_count(self) -> int:
        """Number of tests that were interrupted (incomplete)."""
        return sum(1 for r in self.records if r.status == TestStatus.INCOMPLETE)

    @property
    def has_incomplete(self) -> bool:
        """True if any test has INCOMPLETE status."""
        return any(r.status == TestStatus.INCOMPLETE for r in self.records)


# ---------------------------------------------------------------------------
# ANSI escape code stripping
# ---------------------------------------------------------------------------

# Matches ANSI escape sequences (CSI sequences and OSC sequences)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return _ANSI_ESCAPE_RE.sub("", text)


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------

_PYTEST_SESSION_START_RE = re.compile(r"=+ test session starts =+")
_PYTEST_SUMMARY_RE = re.compile(
    r"=+ .*(passed|failed|error|skipped|warning).* in [\d.]+s =+"
)


_JEST_SUITE_RE = re.compile(r"^\s*(PASS|FAIL)\s+\S+\.test\.\w+")
"""Matches Jest suite-level result lines like: PASS  src/utils/math.test.ts"""

_JEST_SUMMARY_RE = re.compile(r"^Test Suites:\s+")
"""Matches Jest summary: Test Suites: 1 failed, 1 passed, 2 total"""

_GO_TEST_RUN_RE = re.compile(r"^=== RUN\s+\S+")
"""Matches go test verbose run lines: === RUN   TestAdd"""

_GO_TEST_RESULT_RE = re.compile(
    r"^--- (PASS|FAIL|SKIP): (\S+) \((\d+\.\d+)s\)"
)
"""Matches go test result lines: --- PASS: TestAdd (0.00s)"""


def _detect_framework(lines: tuple[str, ...]) -> FrameworkHint:
    """Detect the test framework from output content.

    Checks for framework-specific markers in the output lines.
    Detection order: pytest, jest, go test, unknown.

    Args:
        lines: Stripped output lines.

    Returns:
        Detected FrameworkHint, or UNKNOWN if no markers found.
    """
    for line in lines:
        if _PYTEST_SESSION_START_RE.search(line):
            return FrameworkHint.PYTEST
        if "::" in line and _is_pytest_result_line(line):
            return FrameworkHint.PYTEST

    for line in lines:
        if _JEST_SUITE_RE.match(line):
            return FrameworkHint.JEST
        if _JEST_SUMMARY_RE.match(line):
            return FrameworkHint.JEST

    for line in lines:
        if _GO_TEST_RUN_RE.match(line):
            return FrameworkHint.GO_TEST
        if _GO_TEST_RESULT_RE.match(line):
            return FrameworkHint.GO_TEST

    return FrameworkHint.UNKNOWN


# ---------------------------------------------------------------------------
# Pytest verbose output parsing
# ---------------------------------------------------------------------------

# Matches: tests/test_auth.py::test_login PASSED [optional progress %]
# Also handles: tests/test_auth.py::TestClass::test_method PASSED
_PYTEST_VERBOSE_RESULT_RE = re.compile(
    r"^(\S+?\.py)::([\w\[\]:\-\.]+?)\s+"
    r"(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)"
    r"(?:\s+\[.*\])?"
    r"\s*$"
)

# Matches a partial test path line (test started but no result yet)
# e.g., "tests/test_auth.py::test_login " or "tests/test_auth.py::test_login"
_PYTEST_PARTIAL_TEST_RE = re.compile(
    r"^(\S+?\.py)::([\w\[\]:\-\.]+?)\s*$"
)

# Matches pytest short output: tests/test_auth.py .F..s [progress%]
_PYTEST_SHORT_RE = re.compile(
    r"^(\S+?\.py)\s+([.FsExX]+)\s*(?:\[.*\])?\s*$"
)

# Matches summary line: === N passed, M failed in X.XXs ===
_PYTEST_FINAL_SUMMARY_RE = re.compile(
    r"=+\s+.*(?:passed|failed|error|skipped|warning).*\s+in\s+[\d.]+s\s+=+"
)

# Status mapping from pytest output markers to TestStatus
_STATUS_MAP: dict[str, TestStatus] = {
    "PASSED": TestStatus.PASSED,
    "FAILED": TestStatus.FAILED,
    "ERROR": TestStatus.ERROR,
    "SKIPPED": TestStatus.SKIPPED,
    "XFAIL": TestStatus.SKIPPED,  # Expected failure = skip-like
    "XPASS": TestStatus.PASSED,   # Unexpected pass = passed
}


def _is_pytest_result_line(line: str) -> bool:
    """Check if a line contains a pytest verbose result."""
    return _PYTEST_VERBOSE_RESULT_RE.match(line) is not None


def _parse_pytest_verbose_line(
    line: str,
    line_number: int,
) -> Optional[TestRecord]:
    """Parse a single pytest verbose result line.

    Args:
        line: Stripped line content.
        line_number: 0-based line number in the output.

    Returns:
        TestRecord if the line matches, else None.
    """
    match = _PYTEST_VERBOSE_RESULT_RE.match(line)
    if match is None:
        return None

    module = match.group(1)
    test_path = match.group(2)
    status_str = match.group(3)

    status = _STATUS_MAP.get(status_str, TestStatus.ERROR)

    return TestRecord(
        name=test_path,
        status=status,
        module=module,
        line_number=line_number,
    )


def _parse_partial_test_line(
    line: str,
    line_number: int,
) -> Optional[TestRecord]:
    """Parse a partial test path line (test started, no result).

    This indicates a test that was in progress when the output was
    interrupted. The test is marked as INCOMPLETE.

    Args:
        line: Stripped line content.
        line_number: 0-based line number in the output.

    Returns:
        TestRecord with INCOMPLETE status if matched, else None.
    """
    match = _PYTEST_PARTIAL_TEST_RE.match(line)
    if match is None:
        return None

    module = match.group(1)
    test_path = match.group(2)

    return TestRecord(
        name=test_path,
        status=TestStatus.INCOMPLETE,
        module=module,
        line_number=line_number,
    )


def _parse_pytest_short_line(
    line: str,
    line_number: int,
) -> tuple[TestRecord, ...]:
    """Parse a pytest short-format output line (dots).

    Short format: ``tests/test_auth.py ..F.s [100%]``

    Each character maps to a test outcome:
    - ``.`` -> PASSED
    - ``F`` -> FAILED
    - ``s`` -> SKIPPED
    - ``E`` -> ERROR
    - ``x`` -> SKIPPED (xfail)
    - ``X`` -> PASSED (xpass)

    Since individual test names are not available in short format,
    names are generated as ``{module}#N`` where N is the 1-based index
    within the module.

    Args:
        line: Stripped line content.
        line_number: 0-based line number in the output.

    Returns:
        Tuple of TestRecord instances (may be empty).
    """
    match = _PYTEST_SHORT_RE.match(line)
    if match is None:
        return ()

    module = match.group(1)
    chars = match.group(2)

    _short_status_map: dict[str, TestStatus] = {
        ".": TestStatus.PASSED,
        "F": TestStatus.FAILED,
        "s": TestStatus.SKIPPED,
        "E": TestStatus.ERROR,
        "x": TestStatus.SKIPPED,
        "X": TestStatus.PASSED,
    }

    records: list[TestRecord] = []
    for idx, char in enumerate(chars):
        status = _short_status_map.get(char, TestStatus.INCOMPLETE)
        records.append(TestRecord(
            name=f"{module}#{idx + 1}",
            status=status,
            module=module,
            line_number=line_number,
        ))

    return tuple(records)


# ---------------------------------------------------------------------------
# Pytest failure message extraction
# ---------------------------------------------------------------------------

_PYTEST_FAILURES_HEADER_RE = re.compile(r"^=+ FAILURES =+$")
_PYTEST_FAILURE_NAME_RE = re.compile(r"^_+ (\S+) _+$")
_PYTEST_SHORT_SUMMARY_RE = re.compile(r"^=+ short test summary info =+$")
_PYTEST_ERRORS_HEADER_RE = re.compile(r"^=+ ERRORS =+$")


def _extract_pytest_failure_messages(
    lines: tuple[str, ...],
) -> dict[str, str]:
    """Extract per-test failure messages from the FAILURES section.

    Scans for the ``=== FAILURES ===`` header, then parses each failure
    block delimited by ``___ test_name ___`` underlines.

    Args:
        lines: Stripped, ANSI-free output lines.

    Returns:
        Dict mapping test names to their failure message text.
    """
    failures: dict[str, str] = {}
    in_failures_section = False
    current_test_name: str | None = None
    current_lines: list[str] = []

    for line in lines:
        # Detect start of FAILURES section
        if _PYTEST_FAILURES_HEADER_RE.match(line):
            in_failures_section = True
            continue

        # Detect end of FAILURES section
        if in_failures_section and (
            _PYTEST_SHORT_SUMMARY_RE.match(line)
            or _PYTEST_FINAL_SUMMARY_RE.match(line)
            or _PYTEST_ERRORS_HEADER_RE.match(line)
        ):
            if current_test_name is not None and current_lines:
                failures[current_test_name] = "\n".join(current_lines).strip()
            break

        if not in_failures_section:
            continue

        # Detect individual test failure block header
        name_match = _PYTEST_FAILURE_NAME_RE.match(line)
        if name_match is not None:
            # Save previous block
            if current_test_name is not None and current_lines:
                failures[current_test_name] = "\n".join(current_lines).strip()
            current_test_name = name_match.group(1)
            current_lines = []
            continue

        if current_test_name is not None:
            current_lines.append(line)

    # Save final block if we ran off the end of input
    if current_test_name is not None and current_lines:
        failures[current_test_name] = "\n".join(current_lines).strip()

    return failures


# ---------------------------------------------------------------------------
# Jest output parsing
# ---------------------------------------------------------------------------

# Matches Jest individual test results:
#   check mark (pass): \u2713
#   cross mark (fail): \u2717 or \u2715
#   circle (skip/todo): \u25cb
_JEST_PASS_RE = re.compile(r"^\s+\u2713\s+(.+?)(?:\s+\(\d+\s*m?s\))?\s*$")
_JEST_FAIL_RE = re.compile(r"^\s+\u2717\s+(.+?)(?:\s+\(\d+\s*m?s\))?\s*$")
_JEST_SKIP_RE = re.compile(r"^\s+\u25cb\s+(.+?)$")

# Matches Jest failure detail header:
#   bullet (fail detail): \u25cf Description > test name
_JEST_FAILURE_DETAIL_RE = re.compile(r"^\s+\u25cf\s+(.+)$")

_JEST_TEST_SUITES_RE = re.compile(
    r"^Test Suites:\s+(.+)$"
)


def _parse_jest_output(
    lines: tuple[str, ...],
    raw_output: str,
) -> ParseResult:
    """Parse Jest-style output into structured records.

    Extracts test results from:
    1. Check/cross/circle markers for individual tests
    2. Failure detail blocks for error messages

    Args:
        lines: Stripped, ANSI-free output lines.
        raw_output: Original raw output for truncation detection.

    Returns:
        ParseResult with all extracted records.
    """
    records: list[TestRecord] = []
    current_suite = ""

    # First pass: extract test results
    for line_number, line in enumerate(lines):
        # Track current suite
        suite_match = _JEST_SUITE_RE.match(line)
        if suite_match is not None:
            # Extract module path from "PASS  src/utils/math.test.ts"
            parts = line.strip().split(None, 1)
            current_suite = parts[1].strip() if len(parts) > 1 else ""
            continue

        pass_match = _JEST_PASS_RE.match(line)
        if pass_match is not None:
            records.append(TestRecord(
                name=pass_match.group(1).strip(),
                status=TestStatus.PASSED,
                module=current_suite,
                line_number=line_number,
            ))
            continue

        fail_match = _JEST_FAIL_RE.match(line)
        if fail_match is not None:
            records.append(TestRecord(
                name=fail_match.group(1).strip(),
                status=TestStatus.FAILED,
                module=current_suite,
                line_number=line_number,
            ))
            continue

        skip_match = _JEST_SKIP_RE.match(line)
        if skip_match is not None:
            records.append(TestRecord(
                name=skip_match.group(1).strip(),
                status=TestStatus.SKIPPED,
                module=current_suite,
                line_number=line_number,
            ))
            continue

    # Second pass: extract failure messages
    failure_messages = _extract_jest_failure_messages(lines)

    # Attach failure messages to failed records
    records_with_messages = _attach_jest_failure_messages(records, failure_messages)

    has_summary = any(_JEST_SUMMARY_RE.match(line) for line in lines)
    records_tuple = tuple(records_with_messages)
    truncated = not has_summary and len(records_tuple) > 0

    return ParseResult(
        records=records_tuple,
        truncated=truncated,
        framework_hint=FrameworkHint.JEST,
        total_lines_parsed=len(lines),
        raw_tail=_extract_raw_tail(lines),
    )


def _extract_jest_failure_messages(
    lines: tuple[str, ...],
) -> dict[str, str]:
    """Extract per-test failure messages from Jest failure detail blocks.

    Jest failure blocks start with a bullet marker line:
        ``  * Description > test name``
    followed by the error message and stack trace, and end when the
    next bullet or summary section starts.

    Args:
        lines: Stripped output lines.

    Returns:
        Dict mapping test names (last segment after >) to failure text.
    """
    failures: dict[str, str] = {}
    current_test_name: str | None = None
    current_lines: list[str] = []

    for line in lines:
        detail_match = _JEST_FAILURE_DETAIL_RE.match(line)
        if detail_match is not None:
            # Save previous block
            if current_test_name is not None and current_lines:
                failures[current_test_name] = "\n".join(current_lines).strip()
            # Extract just the test name (last part after >)
            full_path = detail_match.group(1).strip()
            parts = full_path.split(">")
            current_test_name = parts[-1].strip()
            current_lines = []
            continue

        # Detect end of failure blocks
        if current_test_name is not None:
            if _JEST_TEST_SUITES_RE.match(line):
                failures[current_test_name] = "\n".join(current_lines).strip()
                current_test_name = None
                current_lines = []
                continue
            current_lines.append(line)

    # Save final block
    if current_test_name is not None and current_lines:
        failures[current_test_name] = "\n".join(current_lines).strip()

    return failures


def _attach_jest_failure_messages(
    records: list[TestRecord],
    failure_messages: dict[str, str],
) -> list[TestRecord]:
    """Attach failure messages to failed Jest test records.

    Creates new TestRecord instances with failure_message set for tests
    that have matching entries in the failure_messages dict.

    Args:
        records: Original test records.
        failure_messages: Dict mapping test names to failure text.

    Returns:
        New list of TestRecord instances with failure messages attached.
    """
    result: list[TestRecord] = []
    for record in records:
        if record.status == TestStatus.FAILED and record.name in failure_messages:
            result.append(TestRecord(
                name=record.name,
                status=record.status,
                module=record.module,
                duration_seconds=record.duration_seconds,
                output_lines=record.output_lines,
                line_number=record.line_number,
                failure_message=failure_messages[record.name],
            ))
        else:
            result.append(record)
    return result


# ---------------------------------------------------------------------------
# Go test output parsing
# ---------------------------------------------------------------------------


def _parse_go_test_output(
    lines: tuple[str, ...],
    raw_output: str,
) -> ParseResult:
    """Parse go test output into structured records.

    Extracts test results from:
    1. ``=== RUN`` lines to track test names
    2. ``--- PASS/FAIL/SKIP`` lines for results with duration
    3. Log lines between RUN and result for failure messages

    Handles subtests (``TestParent/subtest``) by tracking the full
    test path.

    Args:
        lines: Stripped, ANSI-free output lines.
        raw_output: Original raw output for truncation detection.

    Returns:
        ParseResult with all extracted records.
    """
    records: list[TestRecord] = []
    # Track log lines per test for failure messages
    test_log_lines: dict[str, list[str]] = {}
    current_test: str | None = None

    _go_run_re = re.compile(r"^=== RUN\s+(\S+)")
    _go_result_re = re.compile(
        r"^--- (PASS|FAIL|SKIP): (\S+) \((\d+\.\d+)s\)"
    )
    _go_log_re = re.compile(r"^\s+\S+\.go:\d+: (.+)")
    _go_pkg_result_re = re.compile(
        r"^(ok|FAIL)\s+\S+\s+\d+\.\d+s"
    )

    _go_status_map = {
        "PASS": TestStatus.PASSED,
        "FAIL": TestStatus.FAILED,
        "SKIP": TestStatus.SKIPPED,
    }

    for line_number, line in enumerate(lines):
        # Track current test from RUN lines
        run_match = _go_run_re.match(line)
        if run_match is not None:
            test_name = run_match.group(1)
            current_test = test_name
            if test_name not in test_log_lines:
                test_log_lines[test_name] = []
            continue

        # Capture log lines for the current test
        log_match = _go_log_re.match(line)
        if log_match is not None and current_test is not None:
            test_log_lines[current_test].append(log_match.group(1))
            continue

        # Parse result lines
        result_match = _go_result_re.match(line)
        if result_match is not None:
            status_str = result_match.group(1)
            test_name = result_match.group(2)
            duration = float(result_match.group(3))
            status = _go_status_map.get(status_str, TestStatus.ERROR)

            # Build failure message from collected log lines
            failure_message: str | None = None
            if status == TestStatus.FAILED:
                log_entries = test_log_lines.get(test_name, [])
                if log_entries:
                    failure_message = "\n".join(log_entries)

            # Skip parent test results that just aggregate subtests
            # (parent appears after subtests with same prefix)
            is_parent = any(
                r.name.startswith(test_name + "/") for r in records
            )
            if is_parent:
                continue

            records.append(TestRecord(
                name=test_name,
                status=status,
                module="",
                duration_seconds=duration,
                line_number=line_number,
                failure_message=failure_message,
            ))
            current_test = None
            continue

    has_pkg_result = any(
        re.match(r"^(ok|FAIL)\s+\S+\s+\d+\.\d+s", line) for line in lines
    )
    records_tuple = tuple(records)
    truncated = not has_pkg_result and len(records_tuple) > 0

    return ParseResult(
        records=records_tuple,
        truncated=truncated,
        framework_hint=FrameworkHint.GO_TEST,
        total_lines_parsed=len(lines),
        raw_tail=_extract_raw_tail(lines),
    )


# ---------------------------------------------------------------------------
# Truncation detection
# ---------------------------------------------------------------------------


def _detect_truncation(
    lines: tuple[str, ...],
    records: tuple[TestRecord, ...],
    has_summary: bool,
    raw_output: str,
) -> bool:
    """Determine if the output was truncated.

    Truncation indicators:
    - Any test with INCOMPLETE status
    - Raw output does not end with a newline (cut mid-line)
    - Short output line with incomplete character sequence
    - No summary line found and at least one test was parsed

    Args:
        lines: Stripped output lines.
        records: Parsed test records.
        has_summary: Whether a pytest summary line was found.
        raw_output: The original raw output string.

    Returns:
        True if the output appears truncated.
    """
    # Explicit incomplete tests
    if any(r.status == TestStatus.INCOMPLETE for r in records):
        return True

    # Output doesn't end with newline and has content
    if raw_output and not raw_output.endswith("\n"):
        return True

    return False


# ---------------------------------------------------------------------------
# Raw tail extraction
# ---------------------------------------------------------------------------

_TAIL_LINES = 5


def _extract_raw_tail(lines: tuple[str, ...]) -> str:
    """Extract the last few lines of output for debugging.

    Args:
        lines: All output lines.

    Returns:
        Last N lines joined by newline.
    """
    tail = lines[-_TAIL_LINES:] if len(lines) > _TAIL_LINES else lines
    return "\n".join(tail)


# ---------------------------------------------------------------------------
# Main parsing pipeline
# ---------------------------------------------------------------------------


def _parse_pytest_output(
    lines: tuple[str, ...],
    raw_output: str,
) -> ParseResult:
    """Parse pytest-style output into structured records.

    Processes lines in order, extracting test results from:
    1. Verbose result lines (module::test STATUS)
    2. Short result lines (module .F.s)
    3. Partial test lines (module::test without status -- INCOMPLETE)

    After extracting test records, scans the FAILURES section to attach
    failure messages to the corresponding failed test records.

    Args:
        lines: Stripped, ANSI-free output lines.
        raw_output: Original raw output for truncation detection.

    Returns:
        ParseResult with all extracted records.
    """
    records: list[TestRecord] = []
    has_summary = False

    for line_number, line in enumerate(lines):
        # Check for summary line
        if _PYTEST_FINAL_SUMMARY_RE.match(line):
            has_summary = True
            continue

        # Try verbose result
        record = _parse_pytest_verbose_line(line, line_number)
        if record is not None:
            records.append(record)
            continue

        # Try short result
        short_records = _parse_pytest_short_line(line, line_number)
        if short_records:
            records.extend(short_records)
            continue

    # Check for truncated partial test on the last non-empty line
    last_meaningful = _find_last_meaningful_line(lines)
    if last_meaningful is not None:
        line_idx, last_line = last_meaningful
        # Only check if we haven't already parsed this line as a result
        already_parsed = any(
            r.line_number == line_idx for r in records
        )
        if not already_parsed:
            partial = _parse_partial_test_line(last_line, line_idx)
            if partial is not None:
                records.append(partial)

    # Also check if raw output ends without newline and last part
    # looks like a test path
    if raw_output and not raw_output.endswith("\n"):
        trailing = raw_output.rstrip()
        last_part = trailing.rsplit("\n", 1)[-1] if "\n" in trailing else trailing
        last_part_stripped = last_part.strip()
        # Check if this trailing content looks like a partial test
        partial = _parse_partial_test_line(last_part_stripped, len(lines) - 1)
        if partial is not None:
            # Avoid duplicating if already found
            already_has = any(
                r.name == partial.name and r.status == TestStatus.INCOMPLETE
                for r in records
            )
            if not already_has:
                records.append(partial)

    # Extract failure messages and attach to failed records
    failure_messages = _extract_pytest_failure_messages(lines)
    records = _attach_pytest_failure_messages(records, failure_messages)

    records_tuple = tuple(records)
    truncated = _detect_truncation(lines, records_tuple, has_summary, raw_output)

    return ParseResult(
        records=records_tuple,
        truncated=truncated,
        framework_hint=FrameworkHint.PYTEST,
        total_lines_parsed=len(lines),
        raw_tail=_extract_raw_tail(lines),
    )


def _attach_pytest_failure_messages(
    records: list[TestRecord],
    failure_messages: dict[str, str],
) -> list[TestRecord]:
    """Attach failure messages to failed pytest test records.

    Matches failure block names to test record names. Pytest failure
    block names use the short test function name (e.g., ``test_register``),
    while test record names may include the class prefix
    (e.g., ``TestAuth::test_register``).

    Args:
        records: Original test records.
        failure_messages: Dict mapping short test names to failure text.

    Returns:
        New list of TestRecord instances with failure messages attached.
    """
    result: list[TestRecord] = []
    for record in records:
        if record.status == TestStatus.FAILED:
            # Try exact match first, then match by last part of name
            message = failure_messages.get(record.name)
            if message is None:
                # Try short name (last part after ::)
                short_name = record.name.rsplit("::", 1)[-1]
                message = failure_messages.get(short_name)
            if message is not None:
                result.append(TestRecord(
                    name=record.name,
                    status=record.status,
                    module=record.module,
                    duration_seconds=record.duration_seconds,
                    output_lines=record.output_lines,
                    line_number=record.line_number,
                    failure_message=message,
                ))
                continue
        result.append(record)
    return result


def _find_last_meaningful_line(
    lines: tuple[str, ...],
) -> Optional[tuple[int, str]]:
    """Find the last non-empty line and its index.

    Args:
        lines: Output lines.

    Returns:
        Tuple of (index, line_content) or None if all empty.
    """
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip():
            return idx, lines[idx]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_interrupted_output(
    raw_output: str,
    *,
    context: Optional[OutputContext] = None,
) -> ParseResult:
    """Parse interrupted or truncated test output into structured records.

    Takes raw test output (stdout/stderr captured before interruption)
    and produces structured per-test records. Tests that were in progress
    when the output was interrupted are marked with INCOMPLETE status.

    The parser:
    1. Strips ANSI escape codes
    2. Detects the test framework (or uses the provided hint)
    3. Extracts per-test records with name, status, module
    4. Marks incomplete tests explicitly
    5. Detects output truncation

    Args:
        raw_output: Raw test output text (may be incomplete/truncated).
        context: Optional parser configuration. Defaults to auto-detect.

    Returns:
        ParseResult with structured test records. Never raises.
    """
    effective_context = context if context is not None else OutputContext()

    # Handle empty input
    if not raw_output or not raw_output.strip():
        return ParseResult(
            records=(),
            truncated=False,
            framework_hint=FrameworkHint.UNKNOWN,
            total_lines_parsed=0,
            raw_tail="",
        )

    # Strip ANSI escape codes
    cleaned = _strip_ansi(raw_output)

    # Split into lines and strip trailing whitespace per line
    lines = tuple(line.rstrip() for line in cleaned.split("\n"))

    # Determine framework
    if effective_context.framework_hint == FrameworkHint.AUTO:
        detected = _detect_framework(lines)
    else:
        detected = effective_context.framework_hint

    # Parse based on framework
    if detected == FrameworkHint.PYTEST:
        return _parse_pytest_output(lines, raw_output)

    if detected == FrameworkHint.JEST:
        return _parse_jest_output(lines, raw_output)

    if detected == FrameworkHint.GO_TEST:
        return _parse_go_test_output(lines, raw_output)

    # For unknown frameworks, still try pytest patterns (common fallback)
    # Then fall back to returning no records
    pytest_result = _parse_pytest_output(lines, raw_output)
    if pytest_result.records:
        return pytest_result

    # No recognizable test output
    return ParseResult(
        records=(),
        truncated=False,
        framework_hint=FrameworkHint.UNKNOWN,
        total_lines_parsed=len(lines),
        raw_tail=_extract_raw_tail(lines),
    )
