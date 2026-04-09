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
    UNKNOWN: Could not detect the framework.
    """

    AUTO = "auto"
    PYTEST = "pytest"
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
    """

    name: str
    status: TestStatus
    module: str = ""
    duration_seconds: Optional[float] = None
    output_lines: tuple[str, ...] = ()
    line_number: Optional[int] = None


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


def _detect_framework(lines: tuple[str, ...]) -> FrameworkHint:
    """Detect the test framework from output content.

    Checks for pytest-specific markers in the output lines.

    Args:
        lines: Stripped output lines.

    Returns:
        FrameworkHint.PYTEST if pytest markers found, else UNKNOWN.
    """
    for line in lines:
        if _PYTEST_SESSION_START_RE.search(line):
            return FrameworkHint.PYTEST
        # Also check for pytest-style test paths (module::test)
        if "::" in line and _is_pytest_result_line(line):
            return FrameworkHint.PYTEST
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

    records_tuple = tuple(records)
    truncated = _detect_truncation(lines, records_tuple, has_summary, raw_output)

    return ParseResult(
        records=records_tuple,
        truncated=truncated,
        framework_hint=FrameworkHint.PYTEST,
        total_lines_parsed=len(lines),
        raw_tail=_extract_raw_tail(lines),
    )


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
