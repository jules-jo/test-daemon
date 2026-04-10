"""YAML frontmatter parser and serializer for wiki markdown files.

Handles the Karpathy-style wiki format:
  ---
  key: value
  ---
  # Title
  markdown body...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


_FENCE = "---"


@dataclass(frozen=True)
class WikiDocument:
    """Parsed wiki document with separated frontmatter and body."""

    frontmatter: dict[str, Any]
    body: str


def parse(raw: str) -> WikiDocument:
    """Parse a markdown document with YAML frontmatter.

    Args:
        raw: Full file content, expected to start with '---'.

    Returns:
        WikiDocument with parsed frontmatter dict and markdown body.

    Raises:
        ValueError: If the frontmatter delimiters are missing or malformed.
    """
    stripped = raw.strip()
    if not stripped.startswith(_FENCE):
        raise ValueError(
            "Document must start with YAML frontmatter delimiter '---'"
        )

    # Find the closing fence -- must be '---' on its own line, not an
    # occurrence of '---' inside a YAML string value. Split lines and
    # find the first line after the opening fence that is exactly '---'.
    lines = stripped.split("\n")
    if len(lines) < 2 or lines[0].strip() != _FENCE:
        raise ValueError(
            "Document must start with YAML frontmatter delimiter '---'"
        )

    end_line_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == _FENCE:
            end_line_idx = idx
            break

    if end_line_idx is None:
        raise ValueError("Missing closing YAML frontmatter delimiter '---'")

    yaml_block = "\n".join(lines[1:end_line_idx]).strip()
    body = "\n".join(lines[end_line_idx + 1 :]).strip()

    try:
        frontmatter = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML frontmatter: {exc}") from exc

    if not isinstance(frontmatter, dict):
        raise ValueError(
            f"Frontmatter must be a YAML mapping, got {type(frontmatter).__name__}"
        )

    return WikiDocument(frontmatter=frontmatter, body=body)


def serialize(doc: WikiDocument) -> str:
    """Serialize a WikiDocument back to markdown with YAML frontmatter.

    Args:
        doc: The document to serialize.

    Returns:
        String with YAML frontmatter and markdown body.
    """
    yaml_str = yaml.dump(
        doc.frontmatter,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    ).rstrip()

    return f"{_FENCE}\n{yaml_str}\n{_FENCE}\n\n{doc.body}\n"
