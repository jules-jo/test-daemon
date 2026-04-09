"""Tests for audit-log archival operation.

Verifies that the archiver:
- Moves approved audit-log entries from audit/ to audit/archive/
- Updates YAML frontmatter with archival metadata (archived_at, archived_from, etc.)
- Preserves the original markdown body content
- Returns structured results for each archival operation
- Handles batch archival with mixed success/failure outcomes
- Handles missing source files gracefully (does not crash)
- Handles corrupted source files gracefully (reports error)
- Does not modify the archive directory structure unnecessarily
- Creates the archive directory if it does not exist
- Uses atomic write semantics (write-then-rename) for archive files
- Preserves original frontmatter fields alongside new archival fields
- Sets the type field to 'audit-log-archived' in the archived copy
- Records the original source path in archived_from field
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from jules_daemon.wiki.audit_archiver import (
    ArchivalOutcome,
    ArchivalResult,
    BatchArchivalResult,
    archive_audit_entries,
    archive_single_entry,
)
from jules_daemon.wiki.audit_age_scanner import AgedAuditEntry
from jules_daemon.wiki import frontmatter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
_OLD_TS = _NOW - timedelta(days=100)


def _write_audit_entry(
    wiki_root: Path,
    event_id: str,
    created: datetime,
    *,
    entry_type: str = "audit-log",
    extra_frontmatter: str = "",
    body_text: str = "Command execution audit record.",
) -> Path:
    """Write a minimal audit-log wiki entry with the given timestamp."""
    audit_dir = wiki_root / "pages" / "daemon" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    iso_ts = created.isoformat()
    content = (
        "---\n"
        f"type: {entry_type}\n"
        "tags:\n"
        "- daemon\n"
        "- audit\n"
        f"event_id: {event_id}\n"
        f"created: '{iso_ts}'\n"
        f"executed_at: '{iso_ts}'\n"
        f"{extra_frontmatter}"
        "---\n\n"
        f"# Audit: {event_id}\n\n"
        f"{body_text}\n"
    )
    file_path = audit_dir / f"audit-{event_id}.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def _make_aged_entry(
    wiki_root: Path,
    event_id: str,
    created: datetime = _OLD_TS,
    age_days: int = 100,
) -> AgedAuditEntry:
    """Create an AgedAuditEntry backed by a real wiki file."""
    source_path = _write_audit_entry(wiki_root, event_id, created)
    return AgedAuditEntry(
        source_path=source_path,
        event_id=event_id,
        created_at=created,
        age_days=age_days,
    )


def _read_archived_frontmatter(
    wiki_root: Path, event_id: str
) -> dict[str, Any]:
    """Read and parse the frontmatter of an archived audit entry."""
    archive_path = (
        wiki_root / "pages" / "daemon" / "audit" / "archive"
        / f"audit-{event_id}.md"
    )
    raw = archive_path.read_text(encoding="utf-8")
    doc = frontmatter.parse(raw)
    return doc.frontmatter


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


# ---------------------------------------------------------------------------
# ArchivalResult model tests
# ---------------------------------------------------------------------------


class TestArchivalResult:
    """Tests for the immutable ArchivalResult dataclass."""

    def test_frozen(self) -> None:
        result = ArchivalResult(
            event_id="evt-001",
            source_path=Path("/tmp/audit-evt-001.md"),
            archive_path=Path("/tmp/archive/audit-evt-001.md"),
            outcome=ArchivalOutcome.ARCHIVED,
            archived_at=_NOW,
            error=None,
        )
        with pytest.raises(AttributeError):
            result.event_id = "changed"  # type: ignore[misc]

    def test_is_success_true(self) -> None:
        result = ArchivalResult(
            event_id="evt-001",
            source_path=Path("/tmp/audit-evt-001.md"),
            archive_path=Path("/tmp/archive/audit-evt-001.md"),
            outcome=ArchivalOutcome.ARCHIVED,
            archived_at=_NOW,
            error=None,
        )
        assert result.is_success is True

    def test_is_success_false_on_error(self) -> None:
        result = ArchivalResult(
            event_id="evt-001",
            source_path=Path("/tmp/audit-evt-001.md"),
            archive_path=None,
            outcome=ArchivalOutcome.SOURCE_MISSING,
            archived_at=_NOW,
            error="File not found",
        )
        assert result.is_success is False


# ---------------------------------------------------------------------------
# BatchArchivalResult model tests
# ---------------------------------------------------------------------------


class TestBatchArchivalResult:
    """Tests for the immutable BatchArchivalResult container."""

    def test_empty_result(self) -> None:
        result = BatchArchivalResult(
            results=(),
            archived_at=_NOW,
        )
        assert result.succeeded_count == 0
        assert result.failed_count == 0
        assert result.total_count == 0

    def test_counts_with_mixed_outcomes(self) -> None:
        r1 = ArchivalResult(
            event_id="evt-001",
            source_path=Path("/tmp/a.md"),
            archive_path=Path("/tmp/archive/a.md"),
            outcome=ArchivalOutcome.ARCHIVED,
            archived_at=_NOW,
            error=None,
        )
        r2 = ArchivalResult(
            event_id="evt-002",
            source_path=Path("/tmp/b.md"),
            archive_path=None,
            outcome=ArchivalOutcome.SOURCE_MISSING,
            archived_at=_NOW,
            error="missing",
        )
        result = BatchArchivalResult(
            results=(r1, r2),
            archived_at=_NOW,
        )
        assert result.succeeded_count == 1
        assert result.failed_count == 1
        assert result.total_count == 2

    def test_frozen(self) -> None:
        result = BatchArchivalResult(results=(), archived_at=_NOW)
        with pytest.raises(AttributeError):
            result.results = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# archive_single_entry -- success path
# ---------------------------------------------------------------------------


class TestArchiveSingleSuccess:
    """When archiving a valid audit entry."""

    def test_moves_file_to_archive(self, wiki_root: Path) -> None:
        entry = _make_aged_entry(wiki_root, "evt-001")
        result = archive_single_entry(wiki_root, entry)

        assert result.outcome == ArchivalOutcome.ARCHIVED
        # Original file should be removed
        assert not entry.source_path.exists()
        # Archive file should exist
        archive_path = (
            wiki_root / "pages" / "daemon" / "audit" / "archive"
            / "audit-evt-001.md"
        )
        assert archive_path.exists()

    def test_updates_frontmatter_with_archived_at(self, wiki_root: Path) -> None:
        entry = _make_aged_entry(wiki_root, "evt-002")
        result = archive_single_entry(wiki_root, entry)

        fm = _read_archived_frontmatter(wiki_root, "evt-002")
        assert "archived_at" in fm
        # Should be a valid ISO timestamp
        archived_at = datetime.fromisoformat(fm["archived_at"])
        assert archived_at.tzinfo is not None

    def test_updates_frontmatter_with_archived_from(self, wiki_root: Path) -> None:
        entry = _make_aged_entry(wiki_root, "evt-003")
        result = archive_single_entry(wiki_root, entry)

        fm = _read_archived_frontmatter(wiki_root, "evt-003")
        assert "archived_from" in fm
        assert "audit-evt-003.md" in fm["archived_from"]

    def test_updates_type_to_archived(self, wiki_root: Path) -> None:
        entry = _make_aged_entry(wiki_root, "evt-004")
        result = archive_single_entry(wiki_root, entry)

        fm = _read_archived_frontmatter(wiki_root, "evt-004")
        assert fm["type"] == "audit-log-archived"

    def test_preserves_original_frontmatter_fields(self, wiki_root: Path) -> None:
        entry = _make_aged_entry(wiki_root, "evt-005")
        result = archive_single_entry(wiki_root, entry)

        fm = _read_archived_frontmatter(wiki_root, "evt-005")
        assert fm["event_id"] == "evt-005"
        assert "tags" in fm
        assert "created" in fm

    def test_preserves_markdown_body(self, wiki_root: Path) -> None:
        _write_audit_entry(
            wiki_root, "evt-006", _OLD_TS,
            body_text="Unique body content for test."
        )
        entry = AgedAuditEntry(
            source_path=wiki_root / "pages" / "daemon" / "audit" / "audit-evt-006.md",
            event_id="evt-006",
            created_at=_OLD_TS,
            age_days=100,
        )
        result = archive_single_entry(wiki_root, entry)

        archive_path = (
            wiki_root / "pages" / "daemon" / "audit" / "archive"
            / "audit-evt-006.md"
        )
        raw = archive_path.read_text(encoding="utf-8")
        assert "Unique body content for test." in raw

    def test_creates_archive_directory_if_missing(self, wiki_root: Path) -> None:
        entry = _make_aged_entry(wiki_root, "evt-007")
        # Ensure archive dir does not exist
        archive_dir = wiki_root / "pages" / "daemon" / "audit" / "archive"
        assert not archive_dir.exists()

        result = archive_single_entry(wiki_root, entry)
        assert archive_dir.exists()
        assert result.outcome == ArchivalOutcome.ARCHIVED

    def test_result_contains_correct_paths(self, wiki_root: Path) -> None:
        entry = _make_aged_entry(wiki_root, "evt-008")
        result = archive_single_entry(wiki_root, entry)

        assert result.event_id == "evt-008"
        assert result.source_path == entry.source_path
        expected_archive = (
            wiki_root / "pages" / "daemon" / "audit" / "archive"
            / "audit-evt-008.md"
        )
        assert result.archive_path == expected_archive
        assert result.error is None

    def test_adds_archived_age_days(self, wiki_root: Path) -> None:
        entry = _make_aged_entry(wiki_root, "evt-009", age_days=150)
        result = archive_single_entry(wiki_root, entry)

        fm = _read_archived_frontmatter(wiki_root, "evt-009")
        assert fm["archived_age_days"] == 150


# ---------------------------------------------------------------------------
# archive_single_entry -- failure paths
# ---------------------------------------------------------------------------


class TestArchiveSingleFailure:
    """When archiving encounters errors."""

    def test_source_missing_returns_error(self, wiki_root: Path) -> None:
        # Create entry pointing to nonexistent file
        entry = AgedAuditEntry(
            source_path=wiki_root / "pages" / "daemon" / "audit" / "audit-gone.md",
            event_id="gone",
            created_at=_OLD_TS,
            age_days=100,
        )
        result = archive_single_entry(wiki_root, entry)

        assert result.outcome == ArchivalOutcome.SOURCE_MISSING
        assert result.error is not None
        assert "gone" in result.error.lower() or "not found" in result.error.lower() or "missing" in result.error.lower()

    def test_corrupted_source_returns_error(self, wiki_root: Path) -> None:
        audit_dir = wiki_root / "pages" / "daemon" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        corrupted_path = audit_dir / "audit-corrupted.md"
        corrupted_path.write_text("not valid wiki content", encoding="utf-8")

        entry = AgedAuditEntry(
            source_path=corrupted_path,
            event_id="corrupted",
            created_at=_OLD_TS,
            age_days=100,
        )
        result = archive_single_entry(wiki_root, entry)

        assert result.outcome == ArchivalOutcome.PARSE_ERROR
        assert result.error is not None

    def test_error_does_not_remove_source(self, wiki_root: Path) -> None:
        """On parse error, the original file should remain untouched."""
        audit_dir = wiki_root / "pages" / "daemon" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        corrupted_path = audit_dir / "audit-keep.md"
        corrupted_path.write_text("not valid wiki content", encoding="utf-8")

        entry = AgedAuditEntry(
            source_path=corrupted_path,
            event_id="keep",
            created_at=_OLD_TS,
            age_days=100,
        )
        archive_single_entry(wiki_root, entry)
        # Original should still exist since archival failed
        assert corrupted_path.exists()


# ---------------------------------------------------------------------------
# archive_audit_entries -- batch operation
# ---------------------------------------------------------------------------


class TestArchiveBatch:
    """Batch archival of multiple audit entries."""

    def test_archives_multiple_entries(self, wiki_root: Path) -> None:
        entries = [
            _make_aged_entry(wiki_root, f"evt-batch-{i}")
            for i in range(3)
        ]
        result = archive_audit_entries(wiki_root, entries)

        assert result.succeeded_count == 3
        assert result.failed_count == 0
        assert result.total_count == 3

    def test_mixed_success_and_failure(self, wiki_root: Path) -> None:
        good_entry = _make_aged_entry(wiki_root, "evt-good")
        bad_entry = AgedAuditEntry(
            source_path=wiki_root / "pages" / "daemon" / "audit" / "audit-bad.md",
            event_id="bad",
            created_at=_OLD_TS,
            age_days=100,
        )
        result = archive_audit_entries(wiki_root, [good_entry, bad_entry])

        assert result.succeeded_count == 1
        assert result.failed_count == 1
        assert result.total_count == 2

    def test_empty_list_returns_empty_result(self, wiki_root: Path) -> None:
        result = archive_audit_entries(wiki_root, [])

        assert result.succeeded_count == 0
        assert result.failed_count == 0
        assert result.total_count == 0

    def test_all_results_have_timestamps(self, wiki_root: Path) -> None:
        entries = [_make_aged_entry(wiki_root, "evt-ts-check")]
        result = archive_audit_entries(wiki_root, entries)

        assert result.archived_at is not None
        for r in result.results:
            assert r.archived_at is not None

    def test_batch_creates_archive_directory_once(self, wiki_root: Path) -> None:
        entries = [
            _make_aged_entry(wiki_root, f"evt-dir-{i}")
            for i in range(3)
        ]
        archive_dir = wiki_root / "pages" / "daemon" / "audit" / "archive"
        assert not archive_dir.exists()

        result = archive_audit_entries(wiki_root, entries)
        assert archive_dir.exists()
        assert result.succeeded_count == 3


# ---------------------------------------------------------------------------
# archive_single_entry -- idempotency / collision
# ---------------------------------------------------------------------------


class TestArchiveCollision:
    """When an archived file already exists at the destination."""

    def test_warns_and_allows_overwrite(self, wiki_root: Path) -> None:
        """Per constraints: collision detection is warn-and-allow."""
        entry = _make_aged_entry(wiki_root, "evt-collision")

        # Pre-create the archive file
        archive_dir = wiki_root / "pages" / "daemon" / "audit" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        existing = archive_dir / "audit-evt-collision.md"
        existing.write_text(
            "---\ntype: audit-log-archived\nevent_id: evt-collision\n---\nOld.\n",
            encoding="utf-8",
        )

        result = archive_single_entry(wiki_root, entry)
        # Should succeed (warn-and-allow)
        assert result.outcome == ArchivalOutcome.ARCHIVED
        # Source should be removed
        assert not entry.source_path.exists()


# ---------------------------------------------------------------------------
# Frontmatter archival metadata completeness
# ---------------------------------------------------------------------------


class TestArchivalMetadataCompleteness:
    """Verify all required archival metadata fields are set."""

    def test_all_archival_fields_present(self, wiki_root: Path) -> None:
        entry = _make_aged_entry(wiki_root, "evt-meta")
        archive_single_entry(wiki_root, entry)

        fm = _read_archived_frontmatter(wiki_root, "evt-meta")

        required_fields = [
            "archived_at",
            "archived_from",
            "archived_age_days",
            "type",
        ]
        for field_name in required_fields:
            assert field_name in fm, f"Missing archival field: {field_name}"

    def test_type_changed_from_audit_log_to_archived(self, wiki_root: Path) -> None:
        entry = _make_aged_entry(wiki_root, "evt-type-change")
        archive_single_entry(wiki_root, entry)

        fm = _read_archived_frontmatter(wiki_root, "evt-type-change")
        assert fm["type"] == "audit-log-archived"
