"""Tests for wiki persistence serializer -- AssembledTestResult to wiki entry.

Converts an AssembledTestResult into a Karpathy-style wiki entry
(YAML frontmatter with metadata + markdown body with formatted results)
and writes it to the wiki directory.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.wiki.assembled_result import (
    AssembledTestResult,
    CompletenessRatio,
    CoverageGap,
    DaemonDowntime,
    GapSeverity,
    InterruptionPoint,
    TestOutcome,
    TestRecord,
)
from jules_daemon.wiki.frontmatter import parse
from jules_daemon.wiki.test_result_writer import (
    read_result,
    result_to_document,
    write_result,
)


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _make_passing_result() -> AssembledTestResult:
    """Build a fully-passing test result."""
    records = (
        TestRecord(
            test_name="test_login",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.5,
            module="auth/test_login.py",
            line_number=10,
        ),
        TestRecord(
            test_name="test_logout",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.3,
            module="auth/test_login.py",
            line_number=25,
        ),
        TestRecord(
            test_name="test_create_order",
            outcome=TestOutcome.PASSED,
            duration_seconds=1.2,
            module="orders/test_flow.py",
            line_number=5,
        ),
    )
    return AssembledTestResult(
        run_id="abc-123",
        session_id="sess-456",
        host="staging.example.com",
        records=records,
        completeness=CompletenessRatio(executed=3, expected=3),
        coverage_gaps=(),
        interruption=InterruptionPoint(),
    )


def _make_failing_result() -> AssembledTestResult:
    """Build a result with failures and errors."""
    records = (
        TestRecord(
            test_name="test_login",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.5,
            module="auth/test_login.py",
        ),
        TestRecord(
            test_name="test_payment_flow",
            outcome=TestOutcome.FAILED,
            duration_seconds=2.3,
            error_message="AssertionError: expected 200 got 500",
            module="payments/test_flow.py",
            line_number=42,
        ),
        TestRecord(
            test_name="test_webhook",
            outcome=TestOutcome.ERROR,
            duration_seconds=0.1,
            error_message="ConnectionRefusedError: [Errno 111]",
            module="webhooks/test_handler.py",
            line_number=18,
        ),
        TestRecord(
            test_name="test_skip_legacy",
            outcome=TestOutcome.SKIPPED,
            module="legacy/test_old.py",
        ),
    )
    return AssembledTestResult(
        run_id="def-789",
        session_id="sess-012",
        host="prod.example.com",
        records=records,
        completeness=CompletenessRatio(executed=3, expected=5),
        coverage_gaps=(
            CoverageGap(
                module="billing/test_invoices.py",
                reason="No tests executed from module billing/test_invoices.py",
                severity=GapSeverity.HIGH,
                expected_tests=0,
                actual_tests=0,
            ),
        ),
        interruption=InterruptionPoint(),
    )


def _make_interrupted_result() -> AssembledTestResult:
    """Build a result with interruption metadata."""
    records = (
        TestRecord(
            test_name="test_login",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.5,
            module="auth/test_login.py",
        ),
        TestRecord(
            test_name="test_long_running",
            outcome=TestOutcome.ERROR,
            error_message="Test incomplete: output was interrupted",
            module="integration/test_suite.py",
        ),
    )
    return AssembledTestResult(
        run_id="ghi-345",
        session_id="sess-678",
        host="dev.example.com",
        records=records,
        completeness=CompletenessRatio(executed=1, expected=10),
        coverage_gaps=(
            CoverageGap(
                module="integration/test_suite.py",
                reason="All tests in integration/test_suite.py were interrupted",
                severity=GapSeverity.HIGH,
                expected_tests=8,
                actual_tests=0,
            ),
            CoverageGap(
                module="e2e/test_smoke.py",
                reason="No tests executed from module e2e/test_smoke.py",
                severity=GapSeverity.CRITICAL,
                expected_tests=0,
                actual_tests=0,
            ),
        ),
        interruption=InterruptionPoint(
            interrupted=True,
            at_test="test_long_running",
            at_timestamp=_now_utc(),
            reason="SSH connection lost during test execution",
            exit_code=137,
        ),
    )


def _make_empty_result() -> AssembledTestResult:
    """Build a result with no test records."""
    return AssembledTestResult(
        run_id="empty-001",
        session_id="sess-empty",
        host="empty.example.com",
        records=(),
        completeness=CompletenessRatio(executed=0, expected=0),
        coverage_gaps=(),
        interruption=InterruptionPoint(),
    )


def _make_crash_partial_result() -> AssembledTestResult:
    """Build a result that is partial due to daemon crash (daemon was down)."""
    records = (
        TestRecord(
            test_name="test_login",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.5,
            module="auth/test_login.py",
        ),
        TestRecord(
            test_name="test_payment",
            outcome=TestOutcome.ERROR,
            error_message="Test incomplete: daemon crashed during execution",
            module="payments/test_flow.py",
        ),
    )
    return AssembledTestResult(
        run_id="crash-001",
        session_id="sess-crash",
        host="prod.example.com",
        records=records,
        completeness=CompletenessRatio(executed=1, expected=10),
        coverage_gaps=(),
        interruption=InterruptionPoint(
            interrupted=True,
            at_test="test_payment",
            at_timestamp=_now_utc(),
            reason="Daemon crashed during test execution",
            exit_code=None,
        ),
        daemon_downtime=DaemonDowntime(
            daemon_was_down=True,
            down_started_at=_now_utc(),
            down_ended_at=_now_utc(),
            estimated_down_seconds=25.3,
            recovery_method="reconnect",
        ),
    )


def _make_timeout_partial_result() -> AssembledTestResult:
    """Build a result that is partial due to timeout (daemon was NOT down)."""
    records = (
        TestRecord(
            test_name="test_login",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.5,
            module="auth/test_login.py",
        ),
    )
    return AssembledTestResult(
        run_id="timeout-001",
        session_id="sess-timeout",
        host="staging.example.com",
        records=records,
        completeness=CompletenessRatio(executed=1, expected=20),
        coverage_gaps=(),
        interruption=InterruptionPoint(
            interrupted=True,
            at_test="test_long_running",
            at_timestamp=_now_utc(),
            reason="Test execution timed out after 3600s",
        ),
        daemon_downtime=DaemonDowntime(daemon_was_down=False),
    )


# ---------------------------------------------------------------------------
# result_to_document tests
# ---------------------------------------------------------------------------


class TestResultToDocument:
    """Tests for converting AssembledTestResult to a WikiDocument."""

    def test_returns_wiki_document(self) -> None:
        """result_to_document returns a WikiDocument."""
        result = _make_passing_result()
        doc = result_to_document(result)
        assert doc.frontmatter is not None
        assert doc.body is not None

    def test_frontmatter_has_required_tags(self) -> None:
        """Frontmatter includes daemon and test-result tags."""
        doc = result_to_document(_make_passing_result())
        assert "daemon" in doc.frontmatter["tags"]
        assert "test-result" in doc.frontmatter["tags"]

    def test_frontmatter_type(self) -> None:
        """Frontmatter type is test-result."""
        doc = result_to_document(_make_passing_result())
        assert doc.frontmatter["type"] == "test-result"

    def test_frontmatter_contains_run_id(self) -> None:
        """Frontmatter includes the run_id."""
        doc = result_to_document(_make_passing_result())
        assert doc.frontmatter["run_id"] == "abc-123"

    def test_frontmatter_contains_session_id(self) -> None:
        """Frontmatter includes the session_id."""
        doc = result_to_document(_make_passing_result())
        assert doc.frontmatter["session_id"] == "sess-456"

    def test_frontmatter_contains_host(self) -> None:
        """Frontmatter includes the remote host."""
        doc = result_to_document(_make_passing_result())
        assert doc.frontmatter["host"] == "staging.example.com"

    def test_frontmatter_contains_assembled_at(self) -> None:
        """Frontmatter includes the assembled_at timestamp as ISO string."""
        doc = result_to_document(_make_passing_result())
        assert doc.frontmatter["assembled_at"] is not None
        # Should be parseable as ISO 8601
        datetime.fromisoformat(doc.frontmatter["assembled_at"])

    def test_frontmatter_outcome_passed(self) -> None:
        """Outcome is 'passed' when all tests pass."""
        doc = result_to_document(_make_passing_result())
        assert doc.frontmatter["outcome"] == "passed"

    def test_frontmatter_outcome_failed(self) -> None:
        """Outcome is 'failed' when there are failures."""
        doc = result_to_document(_make_failing_result())
        assert doc.frontmatter["outcome"] == "failed"

    def test_frontmatter_outcome_interrupted(self) -> None:
        """Outcome is 'interrupted' when the run was interrupted."""
        doc = result_to_document(_make_interrupted_result())
        assert doc.frontmatter["outcome"] == "interrupted"

    def test_frontmatter_outcome_empty(self) -> None:
        """Outcome is 'empty' when there are no records."""
        doc = result_to_document(_make_empty_result())
        assert doc.frontmatter["outcome"] == "empty"

    def test_frontmatter_counts(self) -> None:
        """Frontmatter includes test count breakdown."""
        doc = result_to_document(_make_failing_result())
        counts = doc.frontmatter["counts"]
        assert counts["total"] == 4
        assert counts["passed"] == 1
        assert counts["failed"] == 1
        assert counts["skipped"] == 1
        assert counts["errors"] == 1

    def test_frontmatter_pass_rate(self) -> None:
        """Frontmatter includes pass rate as a float."""
        doc = result_to_document(_make_passing_result())
        assert doc.frontmatter["pass_rate"] == 1.0

    def test_frontmatter_total_duration(self) -> None:
        """Frontmatter includes total_duration_seconds."""
        doc = result_to_document(_make_passing_result())
        assert doc.frontmatter["total_duration_seconds"] == pytest.approx(2.0)

    def test_frontmatter_completeness(self) -> None:
        """Frontmatter includes completeness ratio."""
        doc = result_to_document(_make_failing_result())
        comp = doc.frontmatter["completeness"]
        assert comp["executed"] == 3
        assert comp["expected"] == 5
        assert comp["ratio"] == pytest.approx(0.6)
        assert comp["is_complete"] is False

    def test_frontmatter_completeness_complete(self) -> None:
        """Completeness shows is_complete=True when all tests ran."""
        doc = result_to_document(_make_passing_result())
        comp = doc.frontmatter["completeness"]
        assert comp["is_complete"] is True

    def test_frontmatter_interruption_not_interrupted(self) -> None:
        """Interruption section shows interrupted=False for normal runs."""
        doc = result_to_document(_make_passing_result())
        assert doc.frontmatter["interruption"]["interrupted"] is False

    def test_frontmatter_interruption_data(self) -> None:
        """Interruption section has full data for interrupted runs."""
        doc = result_to_document(_make_interrupted_result())
        intr = doc.frontmatter["interruption"]
        assert intr["interrupted"] is True
        assert intr["at_test"] == "test_long_running"
        assert intr["reason"] == "SSH connection lost during test execution"
        assert intr["exit_code"] == 137
        assert intr["at_timestamp"] is not None

    def test_frontmatter_coverage_gaps(self) -> None:
        """Frontmatter includes coverage gap summaries."""
        doc = result_to_document(_make_interrupted_result())
        gaps = doc.frontmatter["coverage_gaps"]
        assert len(gaps) == 2
        assert gaps[0]["module"] == "integration/test_suite.py"
        assert gaps[0]["severity"] == "high"

    def test_frontmatter_no_coverage_gaps_when_none(self) -> None:
        """Coverage gaps list is empty when no gaps exist."""
        doc = result_to_document(_make_passing_result())
        assert doc.frontmatter["coverage_gaps"] == []

    def test_frontmatter_records(self) -> None:
        """Frontmatter includes per-test records for round-trip fidelity."""
        doc = result_to_document(_make_passing_result())
        records = doc.frontmatter["records"]
        assert len(records) == 3
        assert records[0]["test_name"] == "test_login"
        assert records[0]["outcome"] == "passed"
        assert records[0]["duration_seconds"] == 0.5

    def test_frontmatter_records_with_errors(self) -> None:
        """Records include error_message for failed tests."""
        doc = result_to_document(_make_failing_result())
        records = doc.frontmatter["records"]
        failed = [r for r in records if r["outcome"] == "failed"]
        assert len(failed) == 1
        assert "AssertionError" in failed[0]["error_message"]


class TestMarkdownBody:
    """Tests for the human-readable markdown body."""

    def test_body_has_title(self) -> None:
        """Body starts with a test results title."""
        doc = result_to_document(_make_passing_result())
        assert "# Test Results" in doc.body

    def test_body_has_host(self) -> None:
        """Body includes the remote host."""
        doc = result_to_document(_make_passing_result())
        assert "staging.example.com" in doc.body

    def test_body_has_summary_section(self) -> None:
        """Body includes a summary section."""
        doc = result_to_document(_make_passing_result())
        assert "## Summary" in doc.body

    def test_body_has_passed_count(self) -> None:
        """Body includes the passed count."""
        doc = result_to_document(_make_passing_result())
        assert "3" in doc.body

    def test_body_has_failed_tests_section(self) -> None:
        """Body includes a failed tests section when there are failures."""
        doc = result_to_document(_make_failing_result())
        assert "## Failed Tests" in doc.body

    def test_body_failed_section_includes_test_name(self) -> None:
        """Failed tests section lists the test name."""
        doc = result_to_document(_make_failing_result())
        assert "test_payment_flow" in doc.body

    def test_body_failed_section_includes_error(self) -> None:
        """Failed tests section includes the error message."""
        doc = result_to_document(_make_failing_result())
        assert "AssertionError" in doc.body

    def test_body_no_failed_section_when_all_pass(self) -> None:
        """No failed tests section when all tests pass."""
        doc = result_to_document(_make_passing_result())
        assert "## Failed Tests" not in doc.body

    def test_body_has_coverage_gaps_section(self) -> None:
        """Body includes coverage gaps when they exist."""
        doc = result_to_document(_make_failing_result())
        assert "## Coverage Gaps" in doc.body

    def test_body_no_coverage_section_when_none(self) -> None:
        """No coverage gaps section when no gaps exist."""
        doc = result_to_document(_make_passing_result())
        assert "## Coverage Gaps" not in doc.body

    def test_body_has_interruption_section(self) -> None:
        """Body includes interruption section when interrupted."""
        doc = result_to_document(_make_interrupted_result())
        assert "## Interruption" in doc.body

    def test_body_no_interruption_section_when_normal(self) -> None:
        """No interruption section for non-interrupted runs."""
        doc = result_to_document(_make_passing_result())
        assert "## Interruption" not in doc.body

    def test_body_has_completeness_section(self) -> None:
        """Body includes completeness information."""
        doc = result_to_document(_make_failing_result())
        assert "## Completeness" in doc.body

    def test_body_has_run_metadata(self) -> None:
        """Body includes run metadata (run_id, session_id)."""
        doc = result_to_document(_make_passing_result())
        assert "abc-123" in doc.body
        assert "sess-456" in doc.body

    def test_body_empty_result(self) -> None:
        """Body handles empty results gracefully."""
        doc = result_to_document(_make_empty_result())
        assert "# Test Results" in doc.body
        assert "No test records" in doc.body


# ---------------------------------------------------------------------------
# write / read round-trip tests
# ---------------------------------------------------------------------------


class TestWriteResult:
    """Tests for writing AssembledTestResult to wiki."""

    def test_write_creates_file(self, wiki_root: Path) -> None:
        """write_result creates a wiki file."""
        result = _make_passing_result()
        outcome = write_result(wiki_root, result)
        assert outcome.file_path.exists()

    def test_write_file_is_markdown(self, wiki_root: Path) -> None:
        """Written file has .md extension."""
        result = _make_passing_result()
        outcome = write_result(wiki_root, result)
        assert outcome.file_path.suffix == ".md"

    def test_write_filename_contains_run_id(self, wiki_root: Path) -> None:
        """Filename includes the run_id."""
        result = _make_passing_result()
        outcome = write_result(wiki_root, result)
        assert "abc-123" in outcome.file_path.name

    def test_write_in_results_directory(self, wiki_root: Path) -> None:
        """File is stored in pages/daemon/results/."""
        result = _make_passing_result()
        outcome = write_result(wiki_root, result)
        expected_dir = wiki_root / "pages" / "daemon" / "results"
        assert outcome.file_path.parent == expected_dir

    def test_write_contains_valid_frontmatter(self, wiki_root: Path) -> None:
        """Written file has valid YAML frontmatter."""
        result = _make_passing_result()
        outcome = write_result(wiki_root, result)
        raw = outcome.file_path.read_text(encoding="utf-8")
        doc = parse(raw)
        assert doc.frontmatter["type"] == "test-result"
        assert doc.frontmatter["run_id"] == "abc-123"

    def test_write_contains_markdown_body(self, wiki_root: Path) -> None:
        """Written file has a markdown body."""
        result = _make_passing_result()
        outcome = write_result(wiki_root, result)
        raw = outcome.file_path.read_text(encoding="utf-8")
        doc = parse(raw)
        assert "# Test Results" in doc.body

    def test_write_outcome_has_run_id(self, wiki_root: Path) -> None:
        """ResultWriteOutcome carries the run_id."""
        result = _make_passing_result()
        outcome = write_result(wiki_root, result)
        assert outcome.run_id == "abc-123"

    def test_write_outcome_has_written_at(self, wiki_root: Path) -> None:
        """ResultWriteOutcome carries a written_at timestamp."""
        before = _now_utc()
        result = _make_passing_result()
        outcome = write_result(wiki_root, result)
        assert outcome.written_at >= before

    def test_write_no_tmp_residue(self, wiki_root: Path) -> None:
        """No .tmp files remain after writing (atomic write)."""
        result = _make_passing_result()
        write_result(wiki_root, result)
        results_dir = wiki_root / "pages" / "daemon" / "results"
        tmp_files = list(results_dir.glob("*.tmp"))
        assert tmp_files == []

    def test_write_creates_directories(self, wiki_root: Path) -> None:
        """write_result creates the directory tree if it does not exist."""
        result = _make_passing_result()
        outcome = write_result(wiki_root, result)
        assert outcome.file_path.parent.is_dir()

    def test_write_overwrite_existing(self, wiki_root: Path) -> None:
        """Writing the same run_id overwrites the previous file."""
        result = _make_passing_result()
        outcome1 = write_result(wiki_root, result)
        outcome2 = write_result(wiki_root, result)
        assert outcome1.file_path == outcome2.file_path
        assert outcome1.file_path.exists()

    def test_write_multiple_results(self, wiki_root: Path) -> None:
        """Multiple results produce separate files."""
        result1 = _make_passing_result()
        result2 = _make_failing_result()
        outcome1 = write_result(wiki_root, result1)
        outcome2 = write_result(wiki_root, result2)
        assert outcome1.file_path != outcome2.file_path
        assert outcome1.file_path.exists()
        assert outcome2.file_path.exists()


class TestReadResult:
    """Tests for reading AssembledTestResult back from wiki."""

    def test_round_trip_passing(self, wiki_root: Path) -> None:
        """A passing result survives write + read round-trip."""
        original = _make_passing_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        assert loaded.run_id == original.run_id
        assert loaded.session_id == original.session_id
        assert loaded.host == original.host
        assert loaded.total_tests == original.total_tests
        assert loaded.passed_count == original.passed_count

    def test_round_trip_failing(self, wiki_root: Path) -> None:
        """A failing result survives write + read round-trip."""
        original = _make_failing_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        assert loaded.run_id == original.run_id
        assert loaded.failed_count == original.failed_count
        assert loaded.error_count == original.error_count
        assert loaded.skipped_count == original.skipped_count

    def test_round_trip_interrupted(self, wiki_root: Path) -> None:
        """An interrupted result survives write + read round-trip."""
        original = _make_interrupted_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        assert loaded.was_interrupted is True
        assert loaded.interruption.at_test == "test_long_running"
        assert loaded.interruption.exit_code == 137

    def test_round_trip_empty(self, wiki_root: Path) -> None:
        """An empty result survives write + read round-trip."""
        original = _make_empty_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        assert loaded.total_tests == 0

    def test_round_trip_records(self, wiki_root: Path) -> None:
        """Individual test records survive round-trip."""
        original = _make_failing_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        assert len(loaded.records) == len(original.records)
        for orig_rec, loaded_rec in zip(original.records, loaded.records):
            assert loaded_rec.test_name == orig_rec.test_name
            assert loaded_rec.outcome == orig_rec.outcome
            assert loaded_rec.error_message == orig_rec.error_message

    def test_round_trip_coverage_gaps(self, wiki_root: Path) -> None:
        """Coverage gaps survive round-trip."""
        original = _make_interrupted_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        assert len(loaded.coverage_gaps) == len(original.coverage_gaps)
        for orig_gap, loaded_gap in zip(
            original.coverage_gaps, loaded.coverage_gaps
        ):
            assert loaded_gap.module == orig_gap.module
            assert loaded_gap.severity == orig_gap.severity

    def test_round_trip_completeness(self, wiki_root: Path) -> None:
        """Completeness ratio survives round-trip."""
        original = _make_failing_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        assert loaded.completeness.executed == original.completeness.executed
        assert loaded.completeness.expected == original.completeness.expected

    def test_read_nonexistent_returns_none(self, wiki_root: Path) -> None:
        """Reading a nonexistent file returns None."""
        fake = wiki_root / "pages" / "daemon" / "results" / "does-not-exist.md"
        loaded = read_result(fake)
        assert loaded is None

    def test_read_malformed_frontmatter_raises(self, wiki_root: Path) -> None:
        """Reading a file with missing required keys raises KeyError."""
        results_dir = wiki_root / "pages" / "daemon" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        bad_file = results_dir / "result-bad.md"
        bad_file.write_text(
            "---\ntype: test-result\nrecords:\n- outcome: passed\n---\n\n# Bad\n",
            encoding="utf-8",
        )
        with pytest.raises(KeyError):
            read_result(bad_file)

    def test_read_invalid_yaml_raises(self, wiki_root: Path) -> None:
        """Reading a file with invalid YAML delimiters raises ValueError."""
        results_dir = wiki_root / "pages" / "daemon" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        bad_file = results_dir / "result-invalid.md"
        bad_file.write_text("not yaml at all", encoding="utf-8")
        with pytest.raises(ValueError):
            read_result(bad_file)

    def test_round_trip_record_durations(self, wiki_root: Path) -> None:
        """Test record durations survive round-trip."""
        original = _make_passing_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        assert loaded.total_duration_seconds == pytest.approx(
            original.total_duration_seconds
        )

    def test_round_trip_record_line_numbers(self, wiki_root: Path) -> None:
        """Test record line numbers survive round-trip."""
        original = _make_passing_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        for orig_rec, loaded_rec in zip(original.records, loaded.records):
            assert loaded_rec.line_number == orig_rec.line_number

    def test_round_trip_record_modules(self, wiki_root: Path) -> None:
        """Test record modules survive round-trip."""
        original = _make_passing_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        for orig_rec, loaded_rec in zip(original.records, loaded.records):
            assert loaded_rec.module == orig_rec.module


# ---------------------------------------------------------------------------
# Daemon downtime frontmatter tests
# ---------------------------------------------------------------------------


class TestDaemonDowntimeFrontmatter:
    """Tests for daemon_downtime in wiki frontmatter."""

    def test_daemon_was_down_false_by_default(self) -> None:
        """Default result has daemon_was_down=False in frontmatter."""
        doc = result_to_document(_make_passing_result())
        dd = doc.frontmatter["daemon_downtime"]
        assert dd["daemon_was_down"] is False

    def test_daemon_was_down_true_in_frontmatter(self) -> None:
        """Crash-partial result has daemon_was_down=True in frontmatter."""
        doc = result_to_document(_make_crash_partial_result())
        dd = doc.frontmatter["daemon_downtime"]
        assert dd["daemon_was_down"] is True

    def test_estimated_down_seconds_in_frontmatter(self) -> None:
        """Frontmatter includes estimated_down_seconds."""
        doc = result_to_document(_make_crash_partial_result())
        dd = doc.frontmatter["daemon_downtime"]
        assert dd["estimated_down_seconds"] == pytest.approx(25.3)

    def test_recovery_method_in_frontmatter(self) -> None:
        """Frontmatter includes recovery_method."""
        doc = result_to_document(_make_crash_partial_result())
        dd = doc.frontmatter["daemon_downtime"]
        assert dd["recovery_method"] == "reconnect"

    def test_down_timestamps_in_frontmatter(self) -> None:
        """Frontmatter includes down_started_at and down_ended_at as ISO strings."""
        doc = result_to_document(_make_crash_partial_result())
        dd = doc.frontmatter["daemon_downtime"]
        assert dd["down_started_at"] is not None
        assert dd["down_ended_at"] is not None
        # Should be parseable as ISO 8601
        datetime.fromisoformat(dd["down_started_at"])
        datetime.fromisoformat(dd["down_ended_at"])

    def test_timeout_partial_not_daemon_down(self) -> None:
        """Timeout-partial result has daemon_was_down=False."""
        doc = result_to_document(_make_timeout_partial_result())
        dd = doc.frontmatter["daemon_downtime"]
        assert dd["daemon_was_down"] is False
        assert dd["estimated_down_seconds"] is None
        assert dd["down_started_at"] is None
        assert dd["down_ended_at"] is None

    def test_distinguish_crash_vs_timeout_in_frontmatter(self) -> None:
        """Downstream consumers can use frontmatter to distinguish crash from timeout."""
        crash_doc = result_to_document(_make_crash_partial_result())
        timeout_doc = result_to_document(_make_timeout_partial_result())

        # Both are interrupted
        assert crash_doc.frontmatter["interruption"]["interrupted"] is True
        assert timeout_doc.frontmatter["interruption"]["interrupted"] is True

        # Only crash has daemon_was_down=True
        assert crash_doc.frontmatter["daemon_downtime"]["daemon_was_down"] is True
        assert timeout_doc.frontmatter["daemon_downtime"]["daemon_was_down"] is False


class TestDaemonDowntimeMarkdownBody:
    """Tests for daemon downtime in the markdown body."""

    def test_body_has_downtime_section_when_down(self) -> None:
        """Markdown body includes daemon downtime section when daemon was down."""
        doc = result_to_document(_make_crash_partial_result())
        assert "## Daemon Downtime" in doc.body

    def test_body_no_downtime_section_when_not_down(self) -> None:
        """No daemon downtime section when daemon was not down."""
        doc = result_to_document(_make_passing_result())
        assert "## Daemon Downtime" not in doc.body

    def test_body_downtime_section_has_estimated_seconds(self) -> None:
        """Downtime section includes estimated downtime duration."""
        doc = result_to_document(_make_crash_partial_result())
        assert "25.3" in doc.body

    def test_body_downtime_section_has_recovery_method(self) -> None:
        """Downtime section includes recovery method."""
        doc = result_to_document(_make_crash_partial_result())
        assert "reconnect" in doc.body


class TestDaemonDowntimeRoundTrip:
    """Tests for daemon_downtime surviving write + read round-trip."""

    def test_round_trip_crash_partial(self, wiki_root: Path) -> None:
        """Crash-partial daemon downtime survives wiki round-trip."""
        original = _make_crash_partial_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        assert loaded.daemon_was_down is True
        assert loaded.daemon_downtime.daemon_was_down is True
        assert loaded.daemon_downtime.estimated_down_seconds == pytest.approx(25.3)
        assert loaded.daemon_downtime.recovery_method == "reconnect"

    def test_round_trip_timeout_partial(self, wiki_root: Path) -> None:
        """Timeout-partial result survives with daemon_was_down=False."""
        original = _make_timeout_partial_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        assert loaded.daemon_was_down is False
        assert loaded.daemon_downtime.estimated_down_seconds is None

    def test_round_trip_no_downtime(self, wiki_root: Path) -> None:
        """Normal result has default DaemonDowntime after round-trip."""
        original = _make_passing_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        assert loaded.daemon_was_down is False
        assert loaded.daemon_downtime.daemon_was_down is False

    def test_round_trip_down_timestamps(self, wiki_root: Path) -> None:
        """Downtime timestamps survive round-trip."""
        original = _make_crash_partial_result()
        outcome = write_result(wiki_root, original)
        loaded = read_result(outcome.file_path)
        assert loaded is not None
        assert loaded.daemon_downtime.down_started_at is not None
        assert loaded.daemon_downtime.down_ended_at is not None
