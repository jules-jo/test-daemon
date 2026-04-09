"""Tests for the audit wiki persistence writer.

Validates that completed audit chains (AuditRecord at EXECUTION_COMPLETE
or DENIED stages) are correctly persisted as Karpathy-style wiki pages
(YAML frontmatter + markdown body) with full traceability from NL input
to execution result.

Test categories:
    - Frontmatter correctness (structured data round-trips)
    - Markdown body traceability (all stages present)
    - Atomic write safety (temp file + rename)
    - Read/write round-trips (write then read produces equivalent record)
    - Partial records (denied at confirmation, missing stages)
    - Chain inclusion (optional AuditChain stored in frontmatter)
    - Edge cases (missing fields, overwrite behavior)
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.audit.models import (
    AuditRecord,
    ConfirmationDecision,
    ConfirmationRecord,
    NLInputRecord,
    ParsedCommandRecord,
    PipelineStage,
    SSHExecutionRecord,
    StructuredResultRecord,
)
from jules_daemon.audit_models import AuditChain, AuditEntry
from jules_daemon.wiki.audit_writer import (
    AuditWriteOutcome,
    audit_file_path,
    audit_to_document,
    list_audit_files,
    read_audit,
    write_audit,
)
from jules_daemon.wiki.frontmatter import WikiDocument, parse
from jules_daemon.wiki.layout import initialize_wiki


# ---------------------------------------------------------------------------
# Test helpers -- factory functions for building test data
# ---------------------------------------------------------------------------


def _ts() -> datetime:
    """Return a deterministic UTC timestamp for testing."""
    return datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


def _make_nl_input(
    raw: str = "run the full test suite on staging",
) -> NLInputRecord:
    return NLInputRecord(
        raw_input=raw,
        timestamp=_ts(),
        source="cli",
    )


def _make_parsed_command(
    command: str = "pytest -v --tb=short",
) -> ParsedCommandRecord:
    return ParsedCommandRecord(
        natural_language="run the full test suite on staging",
        resolved_shell=command,
        model_id="openai:mesh-conn:gpt-4",
        risk_level="low",
        explanation="Run the test suite with verbose output",
        affected_paths=("/opt/app/tests",),
        timestamp=_ts(),
    )


def _make_confirmation(
    decision: ConfirmationDecision = ConfirmationDecision.APPROVED,
    command: str = "pytest -v --tb=short",
) -> ConfirmationRecord:
    return ConfirmationRecord(
        decision=decision,
        original_command=command,
        final_command=command,
        decided_by="deploy@staging.example.com",
        timestamp=_ts(),
    )


def _make_ssh_execution(
    command: str = "pytest -v --tb=short",
) -> SSHExecutionRecord:
    return SSHExecutionRecord(
        host="staging.example.com",
        user="deploy",
        port=22,
        command=command,
        session_id="sess-abc123",
        started_at=_ts(),
        remote_pid=12345,
        completed_at=_ts(),
        exit_code=0,
        duration_seconds=45.3,
    )


def _make_structured_result() -> StructuredResultRecord:
    return StructuredResultRecord(
        tests_passed=42,
        tests_failed=3,
        tests_skipped=2,
        tests_total=47,
        exit_code=1,
        success=False,
        error_message="3 tests failed",
        summary="42 passed, 3 failed, 2 skipped out of 47 total",
        timestamp=_ts(),
    )


def _make_complete_record(
    run_id: str = "run-001",
    correlation_id: str | None = None,
) -> AuditRecord:
    """Build a fully populated AuditRecord at EXECUTION_COMPLETE."""
    nl = _make_nl_input()
    record = AuditRecord.create(run_id=run_id, nl_input=nl)
    if correlation_id is not None:
        # Replace correlation_id for deterministic tests
        record = replace(record, correlation_id=correlation_id)
    record = record.with_parsed_command(_make_parsed_command())
    record = record.with_confirmation(_make_confirmation())
    record = record.with_ssh_execution(_make_ssh_execution())
    record = record.with_structured_result(_make_structured_result())
    return record


def _make_denied_record(run_id: str = "run-002") -> AuditRecord:
    """Build an AuditRecord denied at the CONFIRMATION stage."""
    nl = _make_nl_input("drop all tables")
    record = AuditRecord.create(run_id=run_id, nl_input=nl)
    record = record.with_parsed_command(
        ParsedCommandRecord(
            natural_language="drop all tables",
            resolved_shell="psql -c 'DROP TABLE *'",
            model_id="openai:mesh-conn:gpt-4",
            risk_level="critical",
            explanation="Drop all database tables",
            affected_paths=("/var/lib/postgresql",),
            timestamp=_ts(),
        )
    )
    record = record.with_confirmation(
        ConfirmationRecord(
            decision=ConfirmationDecision.DENIED,
            original_command="psql -c 'DROP TABLE *'",
            final_command="",
            decided_by="admin@prod.example.com",
            timestamp=_ts(),
        )
    )
    return record


def _make_audit_chain() -> AuditChain:
    """Build an AuditChain with four stage entries."""
    entry1 = AuditEntry(
        stage="nl_input",
        timestamp=_ts(),
        before_snapshot={"raw_input": "run the full test suite"},
        after_snapshot={"canonical_verb": "run"},
        duration=0.1,
        status="success",
        error=None,
    )
    entry2 = AuditEntry(
        stage="confirmation",
        timestamp=_ts(),
        before_snapshot={"command": "pytest -v"},
        after_snapshot={"decision": "approve"},
        duration=2.5,
        status="success",
        error=None,
    )
    entry3 = AuditEntry(
        stage="ssh_execution",
        timestamp=_ts(),
        before_snapshot={"host": "staging.example.com"},
        after_snapshot={"exit_code": 0},
        duration=45.0,
        status="success",
        error=None,
    )
    entry4 = AuditEntry(
        stage="result_structuring",
        timestamp=_ts(),
        before_snapshot={"tests_total": 47},
        after_snapshot={"outcome": "failed"},
        duration=0.05,
        status="success",
        error=None,
    )
    chain = AuditChain.empty()
    chain = chain.append(entry1)
    chain = chain.append(entry2)
    chain = chain.append(entry3)
    chain = chain.append(entry4)
    return chain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def wiki_root(tmp_path: Path) -> Path:
    """Create an initialized wiki directory in a temp path."""
    root = tmp_path / "wiki"
    initialize_wiki(root)
    return root


# ---------------------------------------------------------------------------
# Tests: audit_to_document -- pure conversion
# ---------------------------------------------------------------------------


class TestAuditToDocument:
    """Tests for the pure conversion from AuditRecord to WikiDocument."""

    def test_returns_wiki_document(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert isinstance(doc, WikiDocument)

    def test_frontmatter_has_type(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert doc.frontmatter["type"] == "audit-log"

    def test_frontmatter_has_tags(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        tags = doc.frontmatter["tags"]
        assert "daemon" in tags
        assert "audit-log" in tags

    def test_frontmatter_has_correlation_id(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert doc.frontmatter["correlation_id"] == record.correlation_id

    def test_frontmatter_has_run_id(self) -> None:
        record = _make_complete_record(run_id="run-xyz")
        doc = audit_to_document(record)
        assert doc.frontmatter["run_id"] == "run-xyz"

    def test_frontmatter_has_pipeline_stage(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert doc.frontmatter["pipeline_stage"] == "execution_complete"

    def test_frontmatter_has_all_stage_data(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        fm = doc.frontmatter
        assert fm["nl_input"] is not None
        assert fm["parsed_command"] is not None
        assert fm["confirmation"] is not None
        assert fm["ssh_execution"] is not None
        assert fm["structured_result"] is not None

    def test_frontmatter_nl_input_has_raw_input(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert doc.frontmatter["nl_input"]["raw_input"] == (
            "run the full test suite on staging"
        )

    def test_frontmatter_confirmation_has_decision(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert doc.frontmatter["confirmation"]["decision"] == "approved"

    def test_frontmatter_ssh_execution_has_host(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert doc.frontmatter["ssh_execution"]["host"] == (
            "staging.example.com"
        )

    def test_frontmatter_structured_result_has_counts(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        result = doc.frontmatter["structured_result"]
        assert result["tests_passed"] == 42
        assert result["tests_failed"] == 3
        assert result["tests_total"] == 47

    def test_frontmatter_has_timestamps(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert doc.frontmatter["created_at"] is not None
        assert doc.frontmatter["completed_at"] is not None

    def test_denied_record_has_correct_stage(self) -> None:
        record = _make_denied_record()
        doc = audit_to_document(record)
        assert doc.frontmatter["pipeline_stage"] == "confirmation"

    def test_denied_record_has_no_ssh_execution(self) -> None:
        record = _make_denied_record()
        doc = audit_to_document(record)
        assert doc.frontmatter["ssh_execution"] is None

    def test_denied_record_has_no_structured_result(self) -> None:
        record = _make_denied_record()
        doc = audit_to_document(record)
        assert doc.frontmatter["structured_result"] is None

    def test_chain_entries_included_when_provided(self) -> None:
        record = _make_complete_record()
        chain = _make_audit_chain()
        doc = audit_to_document(record, chain=chain)
        assert "chain_entries" in doc.frontmatter
        assert len(doc.frontmatter["chain_entries"]) == 4

    def test_chain_entries_absent_when_not_provided(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert "chain_entries" not in doc.frontmatter


# ---------------------------------------------------------------------------
# Tests: markdown body traceability
# ---------------------------------------------------------------------------


class TestAuditBodyTraceability:
    """Tests that the markdown body provides full pipeline traceability."""

    def test_body_contains_title(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert "# Audit Record" in doc.body

    def test_body_contains_nl_input_section(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert "## Natural Language Input" in doc.body

    def test_body_contains_raw_input_text(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert "run the full test suite on staging" in doc.body

    def test_body_contains_parsed_command_section(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert "## Parsed Command" in doc.body

    def test_body_contains_resolved_shell_command(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert "pytest -v --tb=short" in doc.body

    def test_body_contains_confirmation_section(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert "## Confirmation" in doc.body

    def test_body_contains_ssh_execution_section(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert "## SSH Execution" in doc.body

    def test_body_contains_host(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert "staging.example.com" in doc.body

    def test_body_contains_result_section(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert "## Structured Result" in doc.body

    def test_body_contains_test_counts(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert "42" in doc.body  # tests_passed
        assert "47" in doc.body  # tests_total

    def test_body_shows_denied_decision(self) -> None:
        record = _make_denied_record()
        doc = audit_to_document(record)
        assert "denied" in doc.body.lower()

    def test_body_omits_ssh_for_denied(self) -> None:
        record = _make_denied_record()
        doc = audit_to_document(record)
        assert "## SSH Execution" not in doc.body

    def test_body_omits_result_for_denied(self) -> None:
        record = _make_denied_record()
        doc = audit_to_document(record)
        assert "## Structured Result" not in doc.body

    def test_body_contains_metadata_section(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert "## Metadata" in doc.body

    def test_body_contains_correlation_id(self) -> None:
        record = _make_complete_record()
        doc = audit_to_document(record)
        assert record.correlation_id in doc.body

    def test_body_contains_chain_summary_when_provided(self) -> None:
        record = _make_complete_record()
        chain = _make_audit_chain()
        doc = audit_to_document(record, chain=chain)
        assert "## Audit Chain" in doc.body


# ---------------------------------------------------------------------------
# Tests: write_audit -- filesystem persistence
# ---------------------------------------------------------------------------


class TestWriteAudit:
    """Tests for writing audit records to the wiki filesystem."""

    def test_write_returns_outcome(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        assert isinstance(outcome, AuditWriteOutcome)

    def test_write_creates_file(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        assert outcome.file_path.exists()

    def test_write_file_in_audit_directory(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        assert "pages/daemon/audit" in str(outcome.file_path)

    def test_write_file_has_correlation_id_in_name(
        self, wiki_root: Path
    ) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        assert record.correlation_id in outcome.file_path.name

    def test_write_file_is_markdown(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        assert outcome.file_path.suffix == ".md"

    def test_write_outcome_has_correlation_id(
        self, wiki_root: Path
    ) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        assert outcome.correlation_id == record.correlation_id

    def test_write_outcome_has_written_at(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        assert outcome.written_at is not None
        assert isinstance(outcome.written_at, datetime)

    def test_write_file_has_valid_frontmatter(
        self, wiki_root: Path
    ) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        content = outcome.file_path.read_text(encoding="utf-8")
        doc = parse(content)
        assert doc.frontmatter["type"] == "audit-log"
        assert doc.frontmatter["correlation_id"] == record.correlation_id

    def test_write_no_tmp_file_left(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        tmp = outcome.file_path.with_suffix(".md.tmp")
        assert not tmp.exists()

    def test_write_overwrite_existing(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome1 = write_audit(wiki_root, record)
        outcome2 = write_audit(wiki_root, record)
        assert outcome1.file_path == outcome2.file_path
        assert outcome2.file_path.exists()

    def test_write_with_chain(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        chain = _make_audit_chain()
        outcome = write_audit(wiki_root, record, chain=chain)
        content = outcome.file_path.read_text(encoding="utf-8")
        doc = parse(content)
        assert "chain_entries" in doc.frontmatter
        assert len(doc.frontmatter["chain_entries"]) == 4

    def test_write_denied_record(self, wiki_root: Path) -> None:
        record = _make_denied_record()
        outcome = write_audit(wiki_root, record)
        assert outcome.file_path.exists()
        content = outcome.file_path.read_text(encoding="utf-8")
        doc = parse(content)
        assert doc.frontmatter["pipeline_stage"] == "confirmation"

    def test_write_creates_parent_directories(
        self, tmp_path: Path
    ) -> None:
        """Write creates audit dir even if wiki is not initialized."""
        wiki_root = tmp_path / "fresh-wiki"
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        assert outcome.file_path.exists()


# ---------------------------------------------------------------------------
# Tests: read_audit -- deserialization
# ---------------------------------------------------------------------------


class TestReadAudit:
    """Tests for reading audit records back from the wiki."""

    def test_read_returns_record(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        restored = read_audit(outcome.file_path)
        assert restored is not None

    def test_read_returns_none_for_missing(self, tmp_path: Path) -> None:
        result = read_audit(tmp_path / "nonexistent.md")
        assert result is None

    def test_read_preserves_correlation_id(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        restored = read_audit(outcome.file_path)
        assert restored is not None
        assert restored.correlation_id == record.correlation_id

    def test_read_preserves_run_id(self, wiki_root: Path) -> None:
        record = _make_complete_record(run_id="run-roundtrip")
        outcome = write_audit(wiki_root, record)
        restored = read_audit(outcome.file_path)
        assert restored is not None
        assert restored.run_id == "run-roundtrip"

    def test_read_preserves_pipeline_stage(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        restored = read_audit(outcome.file_path)
        assert restored is not None
        assert restored.pipeline_stage == PipelineStage.EXECUTION_COMPLETE

    def test_read_preserves_nl_input(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        restored = read_audit(outcome.file_path)
        assert restored is not None
        assert restored.nl_input.raw_input == (
            "run the full test suite on staging"
        )
        assert restored.nl_input.source == "cli"

    def test_read_preserves_parsed_command(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        restored = read_audit(outcome.file_path)
        assert restored is not None
        assert restored.parsed_command is not None
        assert restored.parsed_command.resolved_shell == "pytest -v --tb=short"
        assert restored.parsed_command.risk_level == "low"
        assert restored.parsed_command.model_id == "openai:mesh-conn:gpt-4"

    def test_read_preserves_confirmation(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        restored = read_audit(outcome.file_path)
        assert restored is not None
        assert restored.confirmation is not None
        assert restored.confirmation.decision == ConfirmationDecision.APPROVED

    def test_read_preserves_ssh_execution(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        restored = read_audit(outcome.file_path)
        assert restored is not None
        assert restored.ssh_execution is not None
        assert restored.ssh_execution.host == "staging.example.com"
        assert restored.ssh_execution.exit_code == 0
        assert restored.ssh_execution.duration_seconds == 45.3

    def test_read_preserves_structured_result(
        self, wiki_root: Path
    ) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        restored = read_audit(outcome.file_path)
        assert restored is not None
        assert restored.structured_result is not None
        assert restored.structured_result.tests_passed == 42
        assert restored.structured_result.tests_failed == 3
        assert restored.structured_result.tests_total == 47
        assert restored.structured_result.success is False

    def test_read_preserves_denied_record(self, wiki_root: Path) -> None:
        record = _make_denied_record()
        outcome = write_audit(wiki_root, record)
        restored = read_audit(outcome.file_path)
        assert restored is not None
        assert restored.is_denied is True
        assert restored.pipeline_stage == PipelineStage.CONFIRMATION
        assert restored.ssh_execution is None
        assert restored.structured_result is None

    def test_read_preserves_completed_at(self, wiki_root: Path) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        restored = read_audit(outcome.file_path)
        assert restored is not None
        assert restored.completed_at is not None


# ---------------------------------------------------------------------------
# Tests: audit_file_path -- path resolution
# ---------------------------------------------------------------------------


class TestAuditFilePath:
    """Tests for the audit file path resolver."""

    def test_path_includes_audit_directory(self) -> None:
        path = audit_file_path(Path("/wiki"), "abc-123")
        assert "pages/daemon/audit" in str(path)

    def test_path_includes_correlation_id(self) -> None:
        cid = str(uuid.uuid4())
        path = audit_file_path(Path("/wiki"), cid)
        assert cid in path.name

    def test_path_has_md_extension(self) -> None:
        path = audit_file_path(Path("/wiki"), "test-id")
        assert path.suffix == ".md"

    def test_path_has_audit_prefix(self) -> None:
        path = audit_file_path(Path("/wiki"), "test-id")
        assert path.name.startswith("audit-")


# ---------------------------------------------------------------------------
# Tests: list_audit_files
# ---------------------------------------------------------------------------


class TestListAuditFiles:
    """Tests for listing audit files in the wiki."""

    def test_empty_wiki_returns_empty_list(self, wiki_root: Path) -> None:
        files = list_audit_files(wiki_root)
        assert files == []

    def test_lists_written_audit_files(self, wiki_root: Path) -> None:
        record1 = _make_complete_record(run_id="run-1")
        record2 = _make_denied_record(run_id="run-2")
        write_audit(wiki_root, record1)
        write_audit(wiki_root, record2)
        files = list_audit_files(wiki_root)
        assert len(files) == 2

    def test_ignores_non_audit_files(self, wiki_root: Path) -> None:
        # Write a non-audit file in the audit directory
        audit_dir = wiki_root / "pages" / "daemon" / "audit"
        readme = audit_dir / "README.md"
        assert readme.exists()  # from initialize_wiki
        files = list_audit_files(wiki_root)
        assert len(files) == 0

    def test_returned_paths_are_sorted(self, wiki_root: Path) -> None:
        records = [
            _make_complete_record(run_id=f"run-{i}")
            for i in range(3)
        ]
        for r in records:
            write_audit(wiki_root, r)
        files = list_audit_files(wiki_root)
        names = [f.name for f in files]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Tests: full round-trip
# ---------------------------------------------------------------------------


class TestFullRoundTrip:
    """Integration tests verifying write -> read round-trips."""

    def test_complete_record_round_trips(self, wiki_root: Path) -> None:
        original = _make_complete_record(run_id="round-trip-001")
        outcome = write_audit(wiki_root, original)
        restored = read_audit(outcome.file_path)
        assert restored is not None
        # All fields match
        assert restored.correlation_id == original.correlation_id
        assert restored.run_id == original.run_id
        assert restored.pipeline_stage == original.pipeline_stage
        assert restored.nl_input.raw_input == original.nl_input.raw_input
        assert (
            restored.parsed_command.resolved_shell
            == original.parsed_command.resolved_shell
        )
        assert (
            restored.confirmation.decision
            == original.confirmation.decision
        )
        assert (
            restored.ssh_execution.host
            == original.ssh_execution.host
        )
        assert (
            restored.structured_result.tests_passed
            == original.structured_result.tests_passed
        )

    def test_denied_record_round_trips(self, wiki_root: Path) -> None:
        original = _make_denied_record(run_id="denied-rt-001")
        outcome = write_audit(wiki_root, original)
        restored = read_audit(outcome.file_path)
        assert restored is not None
        assert restored.is_denied is True
        assert restored.correlation_id == original.correlation_id
        assert restored.run_id == original.run_id

    def test_serialized_content_is_valid_yaml_frontmatter(
        self, wiki_root: Path
    ) -> None:
        record = _make_complete_record()
        outcome = write_audit(wiki_root, record)
        content = outcome.file_path.read_text(encoding="utf-8")
        # Should start with ---
        assert content.startswith("---")
        # Should parse without error
        doc = parse(content)
        assert isinstance(doc.frontmatter, dict)
        assert len(doc.body) > 0

    def test_multiple_records_coexist(self, wiki_root: Path) -> None:
        records = [
            _make_complete_record(run_id=f"multi-{i}")
            for i in range(5)
        ]
        outcomes = [write_audit(wiki_root, r) for r in records]
        for outcome, original in zip(outcomes, records):
            restored = read_audit(outcome.file_path)
            assert restored is not None
            assert restored.run_id == original.run_id
