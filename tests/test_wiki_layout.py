"""Tests for wiki directory layout management.

Verifies that the wiki layout module:
- Defines organized directory structure for daemon-managed and user-managed data
- Separates daemon-managed directories (auto-managed) from user-managed ones
- Provides a central registry of all wiki paths
- Initializes directory structure with index files on demand
- Validates wiki structure integrity
- All data types are frozen/immutable
- All directory constants are consistent with actual module paths
"""

from pathlib import Path

import pytest

from jules_daemon.wiki.layout import (
    DAEMON_MANAGED_DIRS,
    USER_MANAGED_DIRS,
    DirectoryKind,
    WikiDirectory,
    WikiLayout,
    WikiValidationResult,
    get_layout,
    initialize_wiki,
    resolve_path,
    validate_wiki,
)


# -- WikiDirectory dataclass --


class TestWikiDirectory:
    def test_frozen(self) -> None:
        d = WikiDirectory(
            relative_path="pages/daemon",
            kind=DirectoryKind.DAEMON_MANAGED,
            description="Daemon state files",
        )
        with pytest.raises(AttributeError):
            d.relative_path = "other"  # type: ignore[misc]

    def test_daemon_managed_kind(self) -> None:
        d = WikiDirectory(
            relative_path="pages/daemon/history",
            kind=DirectoryKind.DAEMON_MANAGED,
            description="Completed run history",
        )
        assert d.kind == DirectoryKind.DAEMON_MANAGED
        assert d.is_daemon_managed is True
        assert d.is_user_managed is False

    def test_user_managed_kind(self) -> None:
        d = WikiDirectory(
            relative_path="pages/concepts",
            kind=DirectoryKind.USER_MANAGED,
            description="User knowledge base",
        )
        assert d.kind == DirectoryKind.USER_MANAGED
        assert d.is_user_managed is True
        assert d.is_daemon_managed is False

    def test_resolve_from_wiki_root(self) -> None:
        d = WikiDirectory(
            relative_path="pages/daemon/results",
            kind=DirectoryKind.DAEMON_MANAGED,
            description="Test results",
        )
        wiki_root = Path("/tmp/wiki")
        assert d.resolve(wiki_root) == Path("/tmp/wiki/pages/daemon/results")


# -- DirectoryKind enum --


class TestDirectoryKind:
    def test_daemon_managed_value(self) -> None:
        assert DirectoryKind.DAEMON_MANAGED.value == "daemon_managed"

    def test_user_managed_value(self) -> None:
        assert DirectoryKind.USER_MANAGED.value == "user_managed"


# -- WikiLayout constants --


class TestWikiLayoutConstants:
    def test_daemon_managed_dirs_not_empty(self) -> None:
        assert len(DAEMON_MANAGED_DIRS) > 0

    def test_user_managed_dirs_not_empty(self) -> None:
        assert len(USER_MANAGED_DIRS) > 0

    def test_daemon_managed_all_have_daemon_kind(self) -> None:
        for d in DAEMON_MANAGED_DIRS:
            assert d.kind == DirectoryKind.DAEMON_MANAGED, (
                f"{d.relative_path} should be DAEMON_MANAGED"
            )

    def test_user_managed_all_have_user_kind(self) -> None:
        for d in USER_MANAGED_DIRS:
            assert d.kind == DirectoryKind.USER_MANAGED, (
                f"{d.relative_path} should be USER_MANAGED"
            )

    def test_no_duplicate_paths(self) -> None:
        all_paths = [d.relative_path for d in DAEMON_MANAGED_DIRS + USER_MANAGED_DIRS]
        assert len(all_paths) == len(set(all_paths)), (
            "Duplicate directory paths found"
        )

    def test_daemon_dirs_include_current_run_parent(self) -> None:
        """current-run.md lives in pages/daemon."""
        paths = {d.relative_path for d in DAEMON_MANAGED_DIRS}
        assert "pages/daemon" in paths

    def test_daemon_dirs_include_history(self) -> None:
        paths = {d.relative_path for d in DAEMON_MANAGED_DIRS}
        assert "pages/daemon/history" in paths

    def test_daemon_dirs_include_results(self) -> None:
        paths = {d.relative_path for d in DAEMON_MANAGED_DIRS}
        assert "pages/daemon/results" in paths

    def test_daemon_dirs_include_translations(self) -> None:
        paths = {d.relative_path for d in DAEMON_MANAGED_DIRS}
        assert "pages/daemon/translations" in paths

    def test_daemon_dirs_include_audit(self) -> None:
        paths = {d.relative_path for d in DAEMON_MANAGED_DIRS}
        assert "pages/daemon/audit" in paths

    def test_daemon_dirs_include_audit_archive(self) -> None:
        paths = {d.relative_path for d in DAEMON_MANAGED_DIRS}
        assert "pages/daemon/audit/archive" in paths

    def test_daemon_dirs_include_queue(self) -> None:
        paths = {d.relative_path for d in DAEMON_MANAGED_DIRS}
        assert "pages/daemon/queue" in paths

    def test_user_dirs_include_concepts(self) -> None:
        paths = {d.relative_path for d in USER_MANAGED_DIRS}
        assert "pages/concepts" in paths

    def test_user_dirs_include_agents(self) -> None:
        paths = {d.relative_path for d in USER_MANAGED_DIRS}
        assert "pages/agents" in paths

    def test_user_dirs_include_systems(self) -> None:
        paths = {d.relative_path for d in USER_MANAGED_DIRS}
        assert "pages/systems" in paths

    def test_all_paths_are_relative(self) -> None:
        all_dirs = DAEMON_MANAGED_DIRS + USER_MANAGED_DIRS
        for d in all_dirs:
            assert not d.relative_path.startswith("/"), (
                f"{d.relative_path} should be relative"
            )

    def test_all_descriptions_nonempty(self) -> None:
        all_dirs = DAEMON_MANAGED_DIRS + USER_MANAGED_DIRS
        for d in all_dirs:
            assert d.description.strip(), (
                f"{d.relative_path} has empty description"
            )


# -- get_layout --


class TestGetLayout:
    def test_returns_wiki_layout(self) -> None:
        layout = get_layout()
        assert isinstance(layout, WikiLayout)

    def test_layout_is_frozen(self) -> None:
        layout = get_layout()
        with pytest.raises(AttributeError):
            layout.daemon_dirs = ()  # type: ignore[misc]

    def test_all_dirs_includes_both_kinds(self) -> None:
        layout = get_layout()
        kinds = {d.kind for d in layout.all_dirs}
        assert DirectoryKind.DAEMON_MANAGED in kinds
        assert DirectoryKind.USER_MANAGED in kinds

    def test_daemon_dirs_returns_only_daemon_managed(self) -> None:
        layout = get_layout()
        for d in layout.daemon_dirs:
            assert d.kind == DirectoryKind.DAEMON_MANAGED

    def test_user_dirs_returns_only_user_managed(self) -> None:
        layout = get_layout()
        for d in layout.user_dirs:
            assert d.kind == DirectoryKind.USER_MANAGED

    def test_all_dirs_is_union(self) -> None:
        layout = get_layout()
        assert len(layout.all_dirs) == len(layout.daemon_dirs) + len(layout.user_dirs)

    def test_find_by_path_returns_match(self) -> None:
        layout = get_layout()
        found = layout.find_by_path("pages/daemon/history")
        assert found is not None
        assert found.relative_path == "pages/daemon/history"

    def test_find_by_path_returns_none_for_missing(self) -> None:
        layout = get_layout()
        assert layout.find_by_path("pages/nonexistent") is None

    def test_find_by_kind_daemon(self) -> None:
        layout = get_layout()
        results = layout.find_by_kind(DirectoryKind.DAEMON_MANAGED)
        assert len(results) > 0
        for d in results:
            assert d.kind == DirectoryKind.DAEMON_MANAGED

    def test_find_by_kind_user(self) -> None:
        layout = get_layout()
        results = layout.find_by_kind(DirectoryKind.USER_MANAGED)
        assert len(results) > 0
        for d in results:
            assert d.kind == DirectoryKind.USER_MANAGED


# -- resolve_path --


class TestResolvePath:
    def test_resolve_daemon_directory(self) -> None:
        wiki_root = Path("/tmp/wiki")
        result = resolve_path(wiki_root, "pages/daemon/history")
        assert result == Path("/tmp/wiki/pages/daemon/history")

    def test_resolve_user_directory(self) -> None:
        wiki_root = Path("/tmp/wiki")
        result = resolve_path(wiki_root, "pages/concepts")
        assert result == Path("/tmp/wiki/pages/concepts")

    def test_resolve_unknown_path_raises(self) -> None:
        wiki_root = Path("/tmp/wiki")
        with pytest.raises(KeyError, match="not a registered"):
            resolve_path(wiki_root, "pages/nonexistent")


# -- initialize_wiki --


class TestInitializeWiki:
    def test_creates_all_directories(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)

        layout = get_layout()
        for d in layout.all_dirs:
            full_path = wiki_root / d.relative_path
            assert full_path.is_dir(), f"Missing directory: {d.relative_path}"

    def test_creates_wiki_root(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        assert not wiki_root.exists()
        initialize_wiki(wiki_root)
        assert wiki_root.is_dir()

    def test_creates_index_at_root(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)
        index_path = wiki_root / "index.md"
        assert index_path.exists()

    def test_index_contains_daemon_section(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)
        index_content = (wiki_root / "index.md").read_text()
        assert "Daemon" in index_content

    def test_index_contains_user_section(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)
        index_content = (wiki_root / "index.md").read_text()
        # User-managed sections should appear in the index
        assert "User" in index_content or "Knowledge" in index_content

    def test_daemon_dirs_have_readme(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)

        for d in DAEMON_MANAGED_DIRS:
            readme = wiki_root / d.relative_path / "README.md"
            assert readme.exists(), f"Missing README: {d.relative_path}"

    def test_user_dirs_have_readme(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)

        for d in USER_MANAGED_DIRS:
            readme = wiki_root / d.relative_path / "README.md"
            assert readme.exists(), f"Missing README: {d.relative_path}"

    def test_readme_has_frontmatter(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)

        from jules_daemon.wiki import frontmatter

        for d in DAEMON_MANAGED_DIRS:
            readme = wiki_root / d.relative_path / "README.md"
            content = readme.read_text()
            doc = frontmatter.parse(content)
            assert "tags" in doc.frontmatter
            assert doc.frontmatter.get("kind") == "daemon_managed"

    def test_user_readme_has_user_kind(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)

        from jules_daemon.wiki import frontmatter

        for d in USER_MANAGED_DIRS:
            readme = wiki_root / d.relative_path / "README.md"
            content = readme.read_text()
            doc = frontmatter.parse(content)
            assert doc.frontmatter.get("kind") == "user_managed"

    def test_idempotent(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)

        # Write a file in a user dir
        user_file = wiki_root / "pages" / "concepts" / "my-note.md"
        user_file.write_text("# My Note\nSome content")

        # Re-initialize should not destroy user files
        initialize_wiki(wiki_root)
        assert user_file.exists()
        assert user_file.read_text() == "# My Note\nSome content"

    def test_does_not_overwrite_existing_readme(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)

        # Modify a README
        readme = wiki_root / "pages" / "daemon" / "README.md"
        original = readme.read_text()
        readme.write_text("Custom README content")

        # Re-initialize should NOT overwrite
        initialize_wiki(wiki_root)
        assert readme.read_text() == "Custom README content"

    def test_returns_list_of_created_paths(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        result = initialize_wiki(wiki_root)
        assert isinstance(result, list)
        assert len(result) > 0
        for p in result:
            assert isinstance(p, Path)

    def test_second_init_returns_empty_list(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)
        result = initialize_wiki(wiki_root)
        # No new files should be created on second init
        assert result == []


# -- validate_wiki --


class TestValidateWiki:
    def test_valid_after_init(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)
        result = validate_wiki(wiki_root)
        assert isinstance(result, WikiValidationResult)
        assert result.is_valid is True
        assert len(result.missing_dirs) == 0
        assert len(result.missing_readmes) == 0

    def test_missing_wiki_root(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "no-such-wiki"
        result = validate_wiki(wiki_root)
        assert result.is_valid is False
        assert len(result.missing_dirs) > 0

    def test_missing_single_directory(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)

        # Remove one daemon dir
        import shutil
        shutil.rmtree(wiki_root / "pages" / "daemon" / "history")

        result = validate_wiki(wiki_root)
        assert result.is_valid is False
        assert "pages/daemon/history" in result.missing_dirs

    def test_missing_readme(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)

        # Remove a README
        readme = wiki_root / "pages" / "daemon" / "history" / "README.md"
        readme.unlink()

        result = validate_wiki(wiki_root)
        assert result.is_valid is False
        assert "pages/daemon/history" in result.missing_readmes

    def test_validation_result_frozen(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        initialize_wiki(wiki_root)
        result = validate_wiki(wiki_root)
        with pytest.raises(AttributeError):
            result.is_valid = False  # type: ignore[misc]


# -- Consistency with existing module paths --


class TestLayoutConsistencyWithModules:
    """Verify that the layout constants match the paths hardcoded in existing modules."""

    def test_current_run_dir_matches(self) -> None:
        """current_run.py uses 'pages/daemon' as its directory."""
        layout = get_layout()
        d = layout.find_by_path("pages/daemon")
        assert d is not None
        assert d.kind == DirectoryKind.DAEMON_MANAGED

    def test_history_dir_matches(self) -> None:
        """run_promotion.py uses 'pages/daemon/history'."""
        layout = get_layout()
        d = layout.find_by_path("pages/daemon/history")
        assert d is not None
        assert d.kind == DirectoryKind.DAEMON_MANAGED

    def test_results_dir_matches(self) -> None:
        """test_result_writer.py uses 'pages/daemon/results'."""
        layout = get_layout()
        d = layout.find_by_path("pages/daemon/results")
        assert d is not None
        assert d.kind == DirectoryKind.DAEMON_MANAGED

    def test_translations_dir_matches(self) -> None:
        """command_translation.py uses 'pages/daemon/translations'."""
        layout = get_layout()
        d = layout.find_by_path("pages/daemon/translations")
        assert d is not None
        assert d.kind == DirectoryKind.DAEMON_MANAGED

    def test_audit_dir_exists(self) -> None:
        """Audit records need their own directory (AC: audit_completeness)."""
        layout = get_layout()
        d = layout.find_by_path("pages/daemon/audit")
        assert d is not None
        assert d.kind == DirectoryKind.DAEMON_MANAGED

    def test_queue_dir_exists(self) -> None:
        """Queue needs its own directory (AC: queue accepts commands)."""
        layout = get_layout()
        d = layout.find_by_path("pages/daemon/queue")
        assert d is not None
        assert d.kind == DirectoryKind.DAEMON_MANAGED
