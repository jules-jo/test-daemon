"""Tests for YAML frontmatter parser/serializer."""

import pytest

from jules_daemon.wiki.frontmatter import WikiDocument, parse, serialize


class TestParse:
    def test_basic_document(self) -> None:
        raw = (
            "---\n"
            "title: Test Page\n"
            "tags: [a, b]\n"
            "---\n"
            "\n"
            "# Test Page\n"
            "\n"
            "Body content here.\n"
        )
        doc = parse(raw)
        assert doc.frontmatter["title"] == "Test Page"
        assert doc.frontmatter["tags"] == ["a", "b"]
        assert "# Test Page" in doc.body
        assert "Body content here." in doc.body

    def test_empty_body(self) -> None:
        raw = "---\nstatus: idle\n---\n"
        doc = parse(raw)
        assert doc.frontmatter["status"] == "idle"
        assert doc.body == ""

    def test_nested_frontmatter(self) -> None:
        raw = (
            "---\n"
            "ssh_target:\n"
            "  host: example.com\n"
            "  user: deploy\n"
            "  port: 22\n"
            "---\n"
            "\n"
            "# Page\n"
        )
        doc = parse(raw)
        assert doc.frontmatter["ssh_target"]["host"] == "example.com"
        assert doc.frontmatter["ssh_target"]["port"] == 22

    def test_missing_opening_fence_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            parse("title: no fence\n---\nBody")

    def test_missing_closing_fence_raises(self) -> None:
        with pytest.raises(ValueError):
            parse("---\ntitle: Test\n")

    def test_non_dict_frontmatter_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            parse("---\n- item1\n- item2\n---\nBody")


class TestSerialize:
    def test_roundtrip(self) -> None:
        original = WikiDocument(
            frontmatter={"status": "idle", "tags": ["daemon"]},
            body="# Title\n\nSome body.",
        )
        serialized = serialize(original)
        restored = parse(serialized)
        assert restored.frontmatter == original.frontmatter
        assert restored.body == original.body

    def test_output_format(self) -> None:
        doc = WikiDocument(
            frontmatter={"key": "value"},
            body="# Hello",
        )
        result = serialize(doc)
        assert result.startswith("---\n")
        assert "\n---\n" in result
        assert result.strip().endswith("# Hello")

    def test_nested_dict_roundtrip(self) -> None:
        fm = {
            "ssh_target": {
                "host": "prod.example.com",
                "user": "ci",
                "port": 2222,
            },
            "progress": {
                "percent": 42.5,
                "tests_passed": 10,
            },
        }
        doc = WikiDocument(frontmatter=fm, body="# Run")
        restored = parse(serialize(doc))
        assert restored.frontmatter["ssh_target"]["host"] == "prod.example.com"
        assert restored.frontmatter["progress"]["percent"] == 42.5

    def test_none_values_preserved(self) -> None:
        doc = WikiDocument(
            frontmatter={"error": None, "started_at": None},
            body="# Page",
        )
        restored = parse(serialize(doc))
        assert restored.frontmatter["error"] is None
        assert restored.frontmatter["started_at"] is None
