"""Tests for markdown-backed system alias loading."""

from __future__ import annotations

from pathlib import Path

from jules_daemon.wiki.system_info import SYSTEMS_DIR, find_system, list_systems


def _write_system(
    wiki_root: Path,
    name: str,
    *,
    frontmatter_text: str,
    body: str = "# System\n\nTest system.\n",
) -> None:
    systems_dir = wiki_root / SYSTEMS_DIR
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / f"{name}.md").write_text(
        f"---\n{frontmatter_text}---\n\n{body}",
        encoding="utf-8",
    )


class TestSystemInfo:
    def test_find_system_by_primary_name(self, tmp_path: Path) -> None:
        _write_system(
            tmp_path,
            "tuto",
            frontmatter_text=(
                "type: system-info\n"
                "system_name: tuto\n"
                "host: 10.0.0.8\n"
                "user: root\n"
                "port: 22\n"
            ),
        )

        system = find_system(tmp_path, "tuto")
        assert system is not None
        assert system.host == "10.0.0.8"
        assert system.user == "root"

    def test_find_system_by_alias(self, tmp_path: Path) -> None:
        _write_system(
            tmp_path,
            "tutorial-box",
            frontmatter_text=(
                "type: system-info\n"
                "system_name: tutorial-box\n"
                "aliases:\n"
                "  - tuto\n"
                "  - tutorial\n"
                "host: 10.0.0.9\n"
                "user: root\n"
            ),
        )

        system = find_system(tmp_path, "tuto")
        assert system is not None
        assert system.system_name == "tutorial-box"

    def test_invalid_system_page_is_skipped(self, tmp_path: Path) -> None:
        _write_system(
            tmp_path,
            "broken",
            frontmatter_text=(
                "type: system-info\n"
                "system_name: broken\n"
                "host: \n"
                "user: root\n"
            ),
        )

        assert list_systems(tmp_path) == ()

    def test_optional_hostname_and_ip_are_loaded(self, tmp_path: Path) -> None:
        _write_system(
            tmp_path,
            "tuto",
            frontmatter_text=(
                "type: system-info\n"
                "system_name: tuto\n"
                "host: 10.0.0.10\n"
                "hostname: tuto.internal.example\n"
                "ip_address: 10.0.0.10\n"
                "user: root\n"
            ),
        )

        system = find_system(tmp_path, "tuto")
        assert system is not None
        assert system.display_hostname == "tuto.internal.example"
        assert system.display_ip_address == "10.0.0.10"
