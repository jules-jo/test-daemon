"""Tests for NL input parsing audit instrumentation.

Verifies that the classify_with_audit() function:
1. Records an AuditEntry on each invocation
2. Writes the audit entry to the wiki filesystem
3. Captures the correct stage, input snapshots, and output snapshots
4. Handles empty/edge-case inputs without crashing
5. Appends to an existing AuditChain when provided
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.audit_models import AuditChain, AuditEntry
from jules_daemon.classifier.nl_audit import NLAuditResult, classify_with_audit
from jules_daemon.wiki.frontmatter import parse as parse_frontmatter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def wiki_root(tmp_path: Path) -> Path:
    """Create a minimal wiki directory structure for audit tests."""
    audit_dir = tmp_path / "wiki" / "pages" / "daemon" / "audit"
    audit_dir.mkdir(parents=True)
    return tmp_path / "wiki"


# ---------------------------------------------------------------------------
# Core: audit entry is written on each invocation
# ---------------------------------------------------------------------------


class TestAuditEntryWritten:
    """Confirm that classify_with_audit writes an audit file."""

    def test_writes_audit_file_on_invocation(self, wiki_root: Path) -> None:
        """A single invocation must produce exactly one audit file."""
        audit_dir = wiki_root / "pages" / "daemon" / "audit"

        classify_with_audit("run the tests on staging", wiki_root)

        audit_files = list(audit_dir.glob("audit-*.md"))
        assert len(audit_files) == 1, (
            f"Expected exactly 1 audit file, found {len(audit_files)}"
        )

    def test_second_invocation_writes_second_file(self, wiki_root: Path) -> None:
        """Each invocation produces a separate audit file."""
        audit_dir = wiki_root / "pages" / "daemon" / "audit"

        classify_with_audit("run the tests", wiki_root)
        classify_with_audit("check status", wiki_root)

        audit_files = list(audit_dir.glob("audit-*.md"))
        assert len(audit_files) == 2

    def test_returns_nl_audit_result(self, wiki_root: Path) -> None:
        """Return type is NLAuditResult with all required fields."""
        result = classify_with_audit("run the smoke tests", wiki_root)

        assert isinstance(result, NLAuditResult)
        assert result.classification is not None
        assert result.chain is not None
        assert result.entry is not None
        assert result.audit_path is not None


# ---------------------------------------------------------------------------
# Audit entry content correctness
# ---------------------------------------------------------------------------


class TestAuditEntryContent:
    """Verify the AuditEntry has correct stage and snapshot data."""

    def test_entry_stage_is_nl_input(self, wiki_root: Path) -> None:
        result = classify_with_audit("run the tests", wiki_root)
        assert result.entry.stage == "nl_input"

    def test_entry_status_is_success(self, wiki_root: Path) -> None:
        result = classify_with_audit("run the tests", wiki_root)
        assert result.entry.status == "success"

    def test_entry_before_snapshot_contains_raw_input(
        self, wiki_root: Path
    ) -> None:
        raw = "run the smoke tests on staging"
        result = classify_with_audit(raw, wiki_root)

        before = result.entry.before_snapshot
        assert isinstance(before, dict)
        assert before["raw_input"] == raw

    def test_entry_after_snapshot_contains_classification(
        self, wiki_root: Path
    ) -> None:
        result = classify_with_audit("run the tests", wiki_root)

        after = result.entry.after_snapshot
        assert isinstance(after, dict)
        assert "canonical_verb" in after
        assert "confidence_score" in after
        assert "input_type" in after

    def test_entry_has_non_negative_duration(self, wiki_root: Path) -> None:
        result = classify_with_audit("cancel the run", wiki_root)
        assert result.entry.duration is not None
        assert result.entry.duration >= 0.0

    def test_entry_has_timestamp(self, wiki_root: Path) -> None:
        result = classify_with_audit("watch the output", wiki_root)
        assert result.entry.timestamp is not None


# ---------------------------------------------------------------------------
# Wiki file content correctness
# ---------------------------------------------------------------------------


class TestWikiFileContent:
    """Verify the written markdown file has correct YAML frontmatter."""

    def test_file_has_yaml_frontmatter(self, wiki_root: Path) -> None:
        result = classify_with_audit("run the tests", wiki_root)

        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert doc.frontmatter is not None
        assert isinstance(doc.frontmatter, dict)

    def test_frontmatter_has_type_audit_log(self, wiki_root: Path) -> None:
        result = classify_with_audit("run the tests", wiki_root)

        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert doc.frontmatter["type"] == "audit-log"

    def test_frontmatter_has_stage_nl_input(self, wiki_root: Path) -> None:
        result = classify_with_audit("run the tests", wiki_root)

        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert doc.frontmatter["stage"] == "nl_input"

    def test_frontmatter_has_status(self, wiki_root: Path) -> None:
        result = classify_with_audit("run the tests", wiki_root)

        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert doc.frontmatter["status"] == "success"

    def test_frontmatter_has_event_id(self, wiki_root: Path) -> None:
        result = classify_with_audit("run the tests", wiki_root)

        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert "event_id" in doc.frontmatter
        assert len(doc.frontmatter["event_id"]) > 0

    def test_frontmatter_has_raw_input(self, wiki_root: Path) -> None:
        raw = "run the smoke tests"
        result = classify_with_audit(raw, wiki_root)

        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert doc.frontmatter["raw_input"] == raw

    def test_file_body_contains_audit_heading(self, wiki_root: Path) -> None:
        result = classify_with_audit("run the tests", wiki_root)

        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert "# NL Input Audit" in doc.body


# ---------------------------------------------------------------------------
# Chain threading
# ---------------------------------------------------------------------------


class TestChainThreading:
    """Verify that an existing AuditChain is correctly extended."""

    def test_default_chain_has_one_entry(self, wiki_root: Path) -> None:
        result = classify_with_audit("run the tests", wiki_root)
        assert len(result.chain) == 1

    def test_appends_to_provided_chain(self, wiki_root: Path) -> None:
        existing_entry = AuditEntry(
            stage="prior_stage",
            timestamp=result_ts(),
            before_snapshot=None,
            after_snapshot=None,
            duration=0.1,
            status="success",
            error=None,
        )
        existing_chain = AuditChain.empty().append(existing_entry)

        result = classify_with_audit(
            "run the tests", wiki_root, chain=existing_chain
        )
        assert len(result.chain) == 2
        assert result.chain.entries[0].stage == "prior_stage"
        assert result.chain.entries[1].stage == "nl_input"

    def test_original_chain_not_mutated(self, wiki_root: Path) -> None:
        original = AuditChain.empty()

        classify_with_audit("run the tests", wiki_root, chain=original)
        assert len(original) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Verify instrumentation handles edge inputs gracefully."""

    def test_empty_input_still_writes_audit(self, wiki_root: Path) -> None:
        audit_dir = wiki_root / "pages" / "daemon" / "audit"

        result = classify_with_audit("", wiki_root)

        audit_files = list(audit_dir.glob("audit-*.md"))
        assert len(audit_files) == 1
        assert result.entry.stage == "nl_input"

    def test_whitespace_input_still_writes_audit(self, wiki_root: Path) -> None:
        result = classify_with_audit("   ", wiki_root)
        assert result.entry.stage == "nl_input"
        assert result.audit_path.exists()

    def test_long_input_writes_audit(self, wiki_root: Path) -> None:
        long_input = "run " * 500
        result = classify_with_audit(long_input, wiki_root)
        assert result.audit_path.exists()

    def test_missing_audit_dir_raises(self, tmp_path: Path) -> None:
        """File write must fail clearly when audit directory does not exist."""
        missing_root = tmp_path / "nonexistent_wiki"
        with pytest.raises((FileNotFoundError, OSError)):
            classify_with_audit("run the tests", missing_root)

    def test_newline_in_input_does_not_break_markdown(
        self, wiki_root: Path
    ) -> None:
        """Newlines in raw input are sanitized in the markdown body."""
        raw = "run the tests\n# Injected heading"
        result = classify_with_audit(raw, wiki_root)

        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        # The injected heading must not appear as a standalone markdown
        # heading (i.e. at the start of a line). It may still appear as
        # text inside a blockquote.
        body_lines = doc.body.split("\n")
        standalone_headings = [
            line for line in body_lines
            if line.strip().startswith("# Injected")
        ]
        assert standalone_headings == [], (
            f"Injected heading appeared as standalone: {standalone_headings}"
        )
        # The body should still have the blockquote with sanitized text
        assert "> " in doc.body


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def result_ts() -> datetime:
    """Return a fixed UTC timestamp for test entries."""
    return datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
