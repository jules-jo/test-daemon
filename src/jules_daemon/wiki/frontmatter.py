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

    # Find the closing fence (index() raises ValueError if not found)
    try:
        end_idx = stripped.index(_FENCE, len(_FENCE))
    except ValueError:
        raise ValueError("Missing closing YAML frontmatter delimiter '---'")

    yaml_block = stripped[len(_FENCE) : end_idx].strip()
    body = stripped[end_idx + len(_FENCE) :].strip()

    frontmatter = yaml.safe_load(yaml_block) or {}
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
