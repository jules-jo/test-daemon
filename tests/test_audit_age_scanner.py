"""Tests for wiki audit-log age scanner.

Verifies that the age scanner:
- Discovers all audit-log wiki files in pages/daemon/audit/
- Parses YAML frontmatter timestamps to determine entry age
- Identifies entries older than the configurable threshold (default 90 days)
- Returns structured AgedAuditEntry records with age metadata
- Handles missing audit directory gracefully (empty result)
- Handles corrupted files gracefully (skips with error, does not crash)
- Skips non-audit wiki files and archived entries
- Excludes README.md files from scan results
- Supports custom age thresholds
- Is deterministic and sorted by age (oldest first)
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jules_daemon.wiki.audit_age_scanner import (
    AgedAuditEntry,
    AuditAgeScanResult,
    ScanOutcome,
    scan_aged_audit_entries,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_audit_entry(
    wiki_root: Path,
    event_id: str,
    created: datetime,
    *,
    entry_type: str = "audit-log",
    extra_frontmatter: str = "",
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
        "Command execution audit record.\n"
    )
    file_path = audit_dir / f"audit-{event_id}.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


# ---------------------------------------------------------------------------
# AgedAuditEntry model tests
# ---------------------------------------------------------------------------


class TestAgedAuditEntry:
    """Tests for the immutable AgedAuditEntry dataclass."""

    def test_frozen(self, wiki_root: Path) -> None:
        entry = AgedAuditEntry(
            source_path=Path("/tmp/audit-001.md"),
            event_id="001",
            created_at=datetime.now(timezone.utc),
            age_days=100,
        )
        with pytest.raises(AttributeError):
            entry.age_days = 200  # type: ignore[misc]

    def test_is_over_threshold_true(self) -> None:
        entry = AgedAuditEntry(
            source_path=Path("/tmp/audit-001.md"),
            event_id="001",
            created_at=datetime.now(timezone.utc) - timedelta(days=91),
            age_days=91,
        )
        assert entry.is_over_threshold(90) is True

    def test_is_over_threshold_false(self) -> None:
        entry = AgedAuditEntry(
            source_path=Path("/tmp/audit-001.md"),
            event_id="001",
            created_at=datetime.now(timezone.utc) - timedelta(days=30),
            age_days=30,
        )
        assert entry.is_over_threshold(90) is False

    def test_is_over_threshold_exact_boundary(self) -> None:
        entry = AgedAuditEntry(
            source_path=Path("/tmp/audit-001.md"),
            event_id="001",
            created_at=datetime.now(timezone.utc) - timedelta(days=90),
            age_days=90,
        )
        # Exactly 90 days is NOT over threshold (strictly greater than)
        assert entry.is_over_threshold(90) is False


# ---------------------------------------------------------------------------
# AuditAgeScanResult model tests
# ---------------------------------------------------------------------------


class TestAuditAgeScanResult:
    """Tests for the immutable AuditAgeScanResult container."""

    def test_empty_result(self) -> None:
        result = AuditAgeScanResult(
            outcome=ScanOutcome.NO_DIRECTORY,
            entries=(),
            errors=(),
            scanned_count=0,
            threshold_days=90,
        )
        assert result.aged_entries == ()
        assert result.total_count == 0
        assert result.aged_count == 0
        assert result.error_count == 0

    def test_aged_entries_filtered(self) -> None:
        now = datetime.now(timezone.utc)
        old_entry = AgedAuditEntry(
            source_path=Path("/tmp/audit-old.md"),
            event_id="old-001",
            created_at=now - timedelta(days=120),
            age_days=120,
        )
        recent_entry = AgedAuditEntry(
            source_path=Path("/tmp/audit-new.md"),
            event_id="new-001",
            created_at=now - timedelta(days=10),
            age_days=10,
        )
        result = AuditAgeScanResult(
            outcome=ScanOutcome.SCANNED,
            entries=(old_entry, recent_entry),
            errors=(),
            scanned_count=2,
            threshold_days=90,
        )
        assert result.aged_count == 1
        assert result.aged_entries == (old_entry,)
        assert result.total_count == 2


# ---------------------------------------------------------------------------
# scan_aged_audit_entries -- no directory
# ---------------------------------------------------------------------------


class TestScanNoDirectory:
    """When the audit directory does not exist."""

    def test_returns_no_directory_outcome(self, wiki_root: Path) -> None:
        result = scan_aged_audit_entries(wiki_root)
        assert result.outcome == ScanOutcome.NO_DIRECTORY

    def test_returns_empty_entries(self, wiki_root: Path) -> None:
        result = scan_aged_audit_entries(wiki_root)
        assert result.entries == ()
        assert result.scanned_count == 0


# ---------------------------------------------------------------------------
# scan_aged_audit_entries -- empty directory
# ---------------------------------------------------------------------------


class TestScanEmptyDirectory:
    """When the audit directory exists but has no audit files."""

    def test_returns_empty_scanned(self, wiki_root: Path) -> None:
        audit_dir = wiki_root / "pages" / "daemon" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        result = scan_aged_audit_entries(wiki_root)
        assert result.outcome == ScanOutcome.SCANNED
        assert result.entries == ()
        assert result.scanned_count == 0

    def test_skips_readme_file(self, wiki_root: Path) -> None:
        audit_dir = wiki_root / "pages" / "daemon" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "README.md").write_text(
            "---\ntags: [wiki-structure]\ntype: wiki-directory\n---\n# Audit\n",
            encoding="utf-8",
        )
        result = scan_aged_audit_entries(wiki_root)
        assert result.scanned_count == 0
        assert result.total_count == 0


# ---------------------------------------------------------------------------
# scan_aged_audit_entries -- old entries
# ---------------------------------------------------------------------------


class TestScanOldEntries:
    """When audit entries older than 90 days exist."""

    def test_identifies_old_entry(self, wiki_root: Path) -> None:
        old_ts = datetime.now(timezone.utc) - timedelta(days=100)
        _write_audit_entry(wiki_root, "evt-old-001", old_ts)

        result = scan_aged_audit_entries(wiki_root)
        assert result.outcome == ScanOutcome.SCANNED
        assert result.total_count == 1
        assert result.aged_count == 1
        assert result.entries[0].event_id == "evt-old-001"
        assert result.entries[0].age_days >= 100

    def test_mixed_old_and_recent(self, wiki_root: Path) -> None:
        now = datetime.now(timezone.utc)
        old_ts = now - timedelta(days=120)
        recent_ts = now - timedelta(days=5)

        _write_audit_entry(wiki_root, "evt-old", old_ts)
        _write_audit_entry(wiki_root, "evt-recent", recent_ts)

        result = scan_aged_audit_entries(wiki_root)
        assert result.total_count == 2
        assert result.aged_count == 1
        assert result.aged_entries[0].event_id == "evt-old"

    def test_sorted_oldest_first(self, wiki_root: Path) -> None:
        now = datetime.now(timezone.utc)
        _write_audit_entry(wiki_root, "evt-91", now - timedelta(days=91))
        _write_audit_entry(wiki_root, "evt-200", now - timedelta(days=200))
        _write_audit_entry(wiki_root, "evt-100", now - timedelta(days=100))

        result = scan_aged_audit_entries(wiki_root)
        # All entries sorted oldest first
        ages = [e.age_days for e in result.entries]
        assert ages == sorted(ages, reverse=True)

    def test_multiple_old_entries(self, wiki_root: Path) -> None:
        now = datetime.now(timezone.utc)
        _write_audit_entry(wiki_root, "evt-a", now - timedelta(days=91))
        _write_audit_entry(wiki_root, "evt-b", now - timedelta(days=180))
        _write_audit_entry(wiki_root, "evt-c", now - timedelta(days=365))

        result = scan_aged_audit_entries(wiki_root)
        assert result.aged_count == 3


# ---------------------------------------------------------------------------
# scan_aged_audit_entries -- custom threshold
# ---------------------------------------------------------------------------


class TestScanCustomThreshold:
    """When a custom age threshold is provided."""

    def test_custom_threshold_30_days(self, wiki_root: Path) -> None:
        now = datetime.now(timezone.utc)
        _write_audit_entry(wiki_root, "evt-40", now - timedelta(days=40))
        _write_audit_entry(wiki_root, "evt-10", now - timedelta(days=10))

        result = scan_aged_audit_entries(wiki_root, threshold_days=30)
        assert result.threshold_days == 30
        assert result.aged_count == 1
        assert result.aged_entries[0].event_id == "evt-40"

    def test_custom_threshold_365_days(self, wiki_root: Path) -> None:
        now = datetime.now(timezone.utc)
        _write_audit_entry(wiki_root, "evt-100", now - timedelta(days=100))
        _write_audit_entry(wiki_root, "evt-400", now - timedelta(days=400))

        result = scan_aged_audit_entries(wiki_root, threshold_days=365)
        assert result.aged_count == 1
        assert result.aged_entries[0].event_id == "evt-400"

    def test_invalid_threshold_raises(self, wiki_root: Path) -> None:
        with pytest.raises(ValueError, match="threshold_days must be positive"):
            scan_aged_audit_entries(wiki_root, threshold_days=0)

        with pytest.raises(ValueError, match="threshold_days must be positive"):
            scan_aged_audit_entries(wiki_root, threshold_days=-10)


# ---------------------------------------------------------------------------
# scan_aged_audit_entries -- corrupted/malformed files
# ---------------------------------------------------------------------------


class TestScanCorruptedFiles:
    """When some audit files are corrupted or malformed."""

    def test_skips_corrupted_file(self, wiki_root: Path) -> None:
        old_ts = datetime.now(timezone.utc) - timedelta(days=100)
        _write_audit_entry(wiki_root, "evt-good", old_ts)

        corrupted = wiki_root / "pages" / "daemon" / "audit" / "audit-bad.md"
        corrupted.write_text("not valid wiki content", encoding="utf-8")

        result = scan_aged_audit_entries(wiki_root)
        assert result.total_count == 1
        assert result.error_count >= 1

    def test_empty_file_is_error(self, wiki_root: Path) -> None:
        audit_dir = wiki_root / "pages" / "daemon" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "audit-empty.md").write_text("", encoding="utf-8")

        result = scan_aged_audit_entries(wiki_root)
        assert result.error_count == 1

    def test_missing_timestamp_is_error(self, wiki_root: Path) -> None:
        audit_dir = wiki_root / "pages" / "daemon" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        no_ts = audit_dir / "audit-nots.md"
        no_ts.write_text(
            "---\ntype: audit-log\ntags: [daemon, audit]\nevent_id: nots\n---\n# No timestamp\n",
            encoding="utf-8",
        )

        result = scan_aged_audit_entries(wiki_root)
        assert result.error_count == 1
        assert result.total_count == 0

    def test_non_audit_type_skipped(self, wiki_root: Path) -> None:
        old_ts = datetime.now(timezone.utc) - timedelta(days=100)
        _write_audit_entry(
            wiki_root, "evt-non-audit", old_ts, entry_type="daemon-state"
        )

        result = scan_aged_audit_entries(wiki_root)
        assert result.total_count == 0


# ---------------------------------------------------------------------------
# scan_aged_audit_entries -- archive exclusion
# ---------------------------------------------------------------------------


class TestScanArchiveExclusion:
    """Archived audit entries should not be included in scan results."""

    def test_excludes_archive_directory(self, wiki_root: Path) -> None:
        old_ts = datetime.now(timezone.utc) - timedelta(days=100)

        # Write an entry in the main audit directory
        _write_audit_entry(wiki_root, "evt-main", old_ts)

        # Write an entry in the archive subdirectory
        archive_dir = wiki_root / "pages" / "daemon" / "audit" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_content = (
            "---\n"
            "type: audit-log\n"
            "tags:\n- daemon\n- audit\n"
            f"event_id: evt-archived\n"
            f"created: '{old_ts.isoformat()}'\n"
            f"executed_at: '{old_ts.isoformat()}'\n"
            "---\n\n# Audit: evt-archived\n\nArchived.\n"
        )
        (archive_dir / "audit-evt-archived.md").write_text(
            archived_content, encoding="utf-8"
        )

        result = scan_aged_audit_entries(wiki_root)
        assert result.total_count == 1
        assert result.entries[0].event_id == "evt-main"


# ---------------------------------------------------------------------------
# scan_aged_audit_entries -- timestamp parsing
# ---------------------------------------------------------------------------


class TestScanTimestampParsing:
    """Verify different ISO 8601 timestamp formats are handled."""

    def test_timezone_aware_timestamp(self, wiki_root: Path) -> None:
        old_ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        _write_audit_entry(wiki_root, "evt-tz", old_ts)

        result = scan_aged_audit_entries(wiki_root)
        assert result.total_count == 1
        assert result.entries[0].created_at.tzinfo is not None

    def test_naive_timestamp_assumes_utc(self, wiki_root: Path) -> None:
        audit_dir = wiki_root / "pages" / "daemon" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        # Write a naive timestamp (no timezone info)
        naive_ts = "2025-01-01T12:00:00"
        content = (
            "---\n"
            "type: audit-log\n"
            "tags:\n- daemon\n- audit\n"
            "event_id: evt-naive\n"
            f"created: '{naive_ts}'\n"
            f"executed_at: '{naive_ts}'\n"
            "---\n\n# Audit: evt-naive\n\nNaive timestamp.\n"
        )
        (audit_dir / "audit-evt-naive.md").write_text(content, encoding="utf-8")

        result = scan_aged_audit_entries(wiki_root)
        assert result.total_count == 1
        assert result.entries[0].created_at.tzinfo is not None

    def test_uses_executed_at_when_created_missing(self, wiki_root: Path) -> None:
        """Falls back to executed_at if created is missing."""
        audit_dir = wiki_root / "pages" / "daemon" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        content = (
            "---\n"
            "type: audit-log\n"
            "tags:\n- daemon\n- audit\n"
            "event_id: evt-fallback\n"
            f"executed_at: '{old_ts}'\n"
            "---\n\n# Audit: evt-fallback\n\nFallback timestamp.\n"
        )
        (audit_dir / "audit-evt-fallback.md").write_text(content, encoding="utf-8")

        result = scan_aged_audit_entries(wiki_root)
        assert result.total_count == 1
        assert result.entries[0].age_days >= 100


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestScanPerformance:
    """Verify scan completes efficiently for reasonable audit volumes."""

    def test_scan_50_entries_under_500ms(self, wiki_root: Path) -> None:
        import time

        now = datetime.now(timezone.utc)
        for i in range(50):
            ts = now - timedelta(days=i * 3)
            _write_audit_entry(wiki_root, f"evt-perf-{i:03d}", ts)

        start = time.monotonic()
        result = scan_aged_audit_entries(wiki_root)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 500.0, f"Scan took {elapsed_ms:.1f}ms (>500ms)"
        assert result.total_count == 50
