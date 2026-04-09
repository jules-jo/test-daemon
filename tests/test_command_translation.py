"""Tests for command translation wiki persistence.

Each NL-to-command mapping is stored as a separate wiki page with
YAML frontmatter + markdown body, enabling the system to learn
from and reference past translations.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.wiki.command_translation import (
    CommandTranslation,
    TranslationOutcome,
    find_by_query,
    list_all,
    load,
    save,
)
from jules_daemon.wiki.frontmatter import parse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


def _make_translation(
    *,
    natural_language: str = "run the full test suite",
    resolved_shell: str = "cd /opt/app && pytest -v --tb=short",
    ssh_host: str = "staging.example.com",
    outcome: TranslationOutcome = TranslationOutcome.APPROVED,
    model_id: str = "dataiku-mesh-gpt4",
) -> CommandTranslation:
    """Create a CommandTranslation with sensible defaults for tests."""
    return CommandTranslation(
        natural_language=natural_language,
        resolved_shell=resolved_shell,
        ssh_host=ssh_host,
        outcome=outcome,
        model_id=model_id,
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestCommandTranslation:
    def test_frozen_immutability(self) -> None:
        t = _make_translation()
        with pytest.raises(AttributeError):
            t.natural_language = "something else"  # type: ignore[misc]

    def test_empty_natural_language_raises(self) -> None:
        with pytest.raises(ValueError, match="natural_language must not be empty"):
            CommandTranslation(
                natural_language="",
                resolved_shell="echo test",
                ssh_host="host.example.com",
            )

    def test_empty_resolved_shell_raises(self) -> None:
        with pytest.raises(ValueError, match="resolved_shell must not be empty"):
            CommandTranslation(
                natural_language="run tests",
                resolved_shell="",
                ssh_host="host.example.com",
            )

    def test_empty_ssh_host_raises(self) -> None:
        with pytest.raises(ValueError, match="ssh_host must not be empty"):
            CommandTranslation(
                natural_language="run tests",
                resolved_shell="pytest -v",
                ssh_host="",
            )

    def test_auto_generated_id(self) -> None:
        t = _make_translation()
        assert t.translation_id
        assert len(t.translation_id) > 0

    def test_unique_ids(self) -> None:
        t1 = _make_translation()
        t2 = _make_translation()
        assert t1.translation_id != t2.translation_id

    def test_default_outcome_is_approved(self) -> None:
        t = _make_translation()
        assert t.outcome == TranslationOutcome.APPROVED

    def test_default_timestamps(self) -> None:
        t = _make_translation()
        assert t.created_at is not None
        assert t.created_at.tzinfo is not None

    def test_default_model_id(self) -> None:
        t = CommandTranslation(
            natural_language="run tests",
            resolved_shell="pytest",
            ssh_host="host.example.com",
        )
        assert t.model_id == ""

    def test_all_outcomes(self) -> None:
        for outcome in TranslationOutcome:
            t = _make_translation(outcome=outcome)
            assert t.outcome == outcome


# ---------------------------------------------------------------------------
# Slug generation (deterministic filenames)
# ---------------------------------------------------------------------------


class TestSlugGeneration:
    def test_slug_from_natural_language(self) -> None:
        t = _make_translation(natural_language="run the full test suite")
        path = save(wiki_root=Path("/tmp/test-wiki"), translation=t)
        assert path.suffix == ".md"
        # Slug should be derived from NL + id, be filesystem-safe
        assert " " not in path.stem

    def test_slug_handles_special_characters(self, wiki_root: Path) -> None:
        t = _make_translation(
            natural_language="run pytest -v --tb=short && echo 'done'"
        )
        path = save(wiki_root=wiki_root, translation=t)
        # No special characters in filename
        stem = path.stem
        for ch in "&|;'\"<>(){}[]!@#$%^*=+":
            assert ch not in stem

    def test_slug_truncation(self, wiki_root: Path) -> None:
        long_nl = "a" * 300
        t = _make_translation(natural_language=long_nl)
        path = save(wiki_root=wiki_root, translation=t)
        # Filename should be reasonably short
        assert len(path.stem) <= 120


# ---------------------------------------------------------------------------
# Save / Load roundtrip
# ---------------------------------------------------------------------------


class TestSave:
    def test_creates_file_and_directories(self, wiki_root: Path) -> None:
        t = _make_translation()
        result_path = save(wiki_root, t)

        assert result_path.exists()
        assert result_path.suffix == ".md"
        assert "pages/daemon/translations" in str(result_path)

    def test_file_is_valid_wiki_format(self, wiki_root: Path) -> None:
        t = _make_translation()
        path = save(wiki_root, t)

        raw = path.read_text(encoding="utf-8")
        doc = parse(raw)

        assert "tags" in doc.frontmatter
        assert "command-translation" in doc.frontmatter["tags"]
        assert doc.frontmatter["type"] == "command-translation"
        assert "# Command Translation" in doc.body

    def test_frontmatter_contains_all_fields(self, wiki_root: Path) -> None:
        t = _make_translation()
        path = save(wiki_root, t)

        raw = path.read_text(encoding="utf-8")
        doc = parse(raw)
        fm = doc.frontmatter

        assert fm["translation_id"] == t.translation_id
        assert fm["natural_language"] == t.natural_language
        assert fm["resolved_shell"] == t.resolved_shell
        assert fm["ssh_host"] == t.ssh_host
        assert fm["outcome"] == t.outcome.value
        assert fm["model_id"] == t.model_id
        assert fm["created"] is not None

    def test_body_contains_command_details(self, wiki_root: Path) -> None:
        t = _make_translation()
        path = save(wiki_root, t)

        raw = path.read_text(encoding="utf-8")
        doc = parse(raw)

        assert t.natural_language in doc.body
        assert t.resolved_shell in doc.body
        assert t.ssh_host in doc.body


class TestLoad:
    def test_returns_none_for_missing_file(self, wiki_root: Path) -> None:
        result = load(wiki_root, "nonexistent-id")
        assert result is None

    def test_roundtrip_preserves_all_fields(self, wiki_root: Path) -> None:
        original = _make_translation()
        save(wiki_root, original)

        loaded = load(wiki_root, original.translation_id)

        assert loaded is not None
        assert loaded.translation_id == original.translation_id
        assert loaded.natural_language == original.natural_language
        assert loaded.resolved_shell == original.resolved_shell
        assert loaded.ssh_host == original.ssh_host
        assert loaded.outcome == original.outcome
        assert loaded.model_id == original.model_id

    def test_roundtrip_with_denied_outcome(self, wiki_root: Path) -> None:
        original = _make_translation(outcome=TranslationOutcome.DENIED)
        save(wiki_root, original)

        loaded = load(wiki_root, original.translation_id)
        assert loaded is not None
        assert loaded.outcome == TranslationOutcome.DENIED

    def test_roundtrip_with_edited_outcome(self, wiki_root: Path) -> None:
        original = _make_translation(outcome=TranslationOutcome.EDITED)
        save(wiki_root, original)

        loaded = load(wiki_root, original.translation_id)
        assert loaded is not None
        assert loaded.outcome == TranslationOutcome.EDITED


# ---------------------------------------------------------------------------
# List all translations
# ---------------------------------------------------------------------------


class TestListAll:
    def test_empty_when_no_files(self, wiki_root: Path) -> None:
        result = list_all(wiki_root)
        assert result == []

    def test_returns_all_saved_translations(self, wiki_root: Path) -> None:
        t1 = _make_translation(natural_language="run unit tests")
        t2 = _make_translation(natural_language="run integration tests")
        t3 = _make_translation(natural_language="run smoke tests")

        save(wiki_root, t1)
        save(wiki_root, t2)
        save(wiki_root, t3)

        results = list_all(wiki_root)
        assert len(results) == 3

        ids = {t.translation_id for t in results}
        assert t1.translation_id in ids
        assert t2.translation_id in ids
        assert t3.translation_id in ids

    def test_results_ordered_by_created_at_desc(self, wiki_root: Path) -> None:
        """Most recent translations should come first."""
        t1 = CommandTranslation(
            natural_language="first command",
            resolved_shell="echo first",
            ssh_host="host.example.com",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        t2 = CommandTranslation(
            natural_language="second command",
            resolved_shell="echo second",
            ssh_host="host.example.com",
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        t3 = CommandTranslation(
            natural_language="third command",
            resolved_shell="echo third",
            ssh_host="host.example.com",
            created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )

        save(wiki_root, t1)
        save(wiki_root, t2)
        save(wiki_root, t3)

        results = list_all(wiki_root)
        assert len(results) == 3
        # Most recent first
        assert results[0].translation_id == t2.translation_id
        assert results[1].translation_id == t3.translation_id
        assert results[2].translation_id == t1.translation_id

    def test_skips_malformed_files(self, wiki_root: Path) -> None:
        """Malformed wiki files should be skipped, not crash listing."""
        t = _make_translation()
        save(wiki_root, t)

        # Write a malformed file
        translations_dir = wiki_root / "pages" / "daemon" / "translations"
        bad_file = translations_dir / "bad-translation.md"
        bad_file.write_text("not a valid wiki file", encoding="utf-8")

        results = list_all(wiki_root)
        assert len(results) == 1
        assert results[0].translation_id == t.translation_id


# ---------------------------------------------------------------------------
# Find by query (substring match for learning from past translations)
# ---------------------------------------------------------------------------


class TestFindByQuery:
    def test_empty_when_no_files(self, wiki_root: Path) -> None:
        result = find_by_query(wiki_root, "pytest")
        assert result == []

    def test_finds_matching_translations(self, wiki_root: Path) -> None:
        t1 = _make_translation(
            natural_language="run the pytest suite",
            resolved_shell="pytest -v",
        )
        t2 = _make_translation(
            natural_language="run jest tests",
            resolved_shell="npx jest --verbose",
        )
        t3 = _make_translation(
            natural_language="run pytest with coverage",
            resolved_shell="pytest --cov",
        )

        save(wiki_root, t1)
        save(wiki_root, t2)
        save(wiki_root, t3)

        # Search for "pytest" should match t1 and t3
        results = find_by_query(wiki_root, "pytest")
        ids = {t.translation_id for t in results}
        assert t1.translation_id in ids
        assert t3.translation_id in ids
        assert t2.translation_id not in ids

    def test_case_insensitive_search(self, wiki_root: Path) -> None:
        t = _make_translation(
            natural_language="Run the PyTest suite",
            resolved_shell="pytest -v",
        )
        save(wiki_root, t)

        results = find_by_query(wiki_root, "pytest")
        assert len(results) == 1

    def test_searches_both_nl_and_shell(self, wiki_root: Path) -> None:
        t = _make_translation(
            natural_language="run all tests",
            resolved_shell="cd /app && pytest -v --tb=short",
        )
        save(wiki_root, t)

        # Search by shell command content
        results = find_by_query(wiki_root, "tb=short")
        assert len(results) == 1
        assert results[0].translation_id == t.translation_id

    def test_filters_by_ssh_host(self, wiki_root: Path) -> None:
        t1 = _make_translation(
            natural_language="run tests",
            resolved_shell="pytest",
            ssh_host="staging.example.com",
        )
        t2 = _make_translation(
            natural_language="run tests on prod",
            resolved_shell="pytest",
            ssh_host="prod.example.com",
        )
        save(wiki_root, t1)
        save(wiki_root, t2)

        results = find_by_query(
            wiki_root, "run tests", ssh_host="staging.example.com"
        )
        assert len(results) == 1
        assert results[0].ssh_host == "staging.example.com"

    def test_max_results_limit(self, wiki_root: Path) -> None:
        for i in range(10):
            t = _make_translation(
                natural_language=f"run test suite variant {i}",
                resolved_shell=f"pytest -k variant_{i}",
            )
            save(wiki_root, t)

        results = find_by_query(wiki_root, "test suite", max_results=3)
        assert len(results) == 3

    def test_empty_query_returns_empty(self, wiki_root: Path) -> None:
        t = _make_translation()
        save(wiki_root, t)

        results = find_by_query(wiki_root, "")
        assert results == []
