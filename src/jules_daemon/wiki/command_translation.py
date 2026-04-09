"""Wiki persistence for NL-to-command translation mappings.

Each natural-language-to-shell-command translation is stored as a separate
wiki page with YAML frontmatter + markdown body. This enables the system
to learn from and reference past translations when processing new requests.

Wiki file location: {wiki_root}/pages/daemon/translations/{slug}.md

The translation store is append-only -- translations are never modified
after creation. Each file is a self-contained wiki page following the
Karpathy-style format (YAML frontmatter + markdown body).

Usage:
    from jules_daemon.wiki.command_translation import (
        CommandTranslation,
        TranslationOutcome,
        save,
        load,
        list_all,
        find_by_query,
    )

    translation = CommandTranslation(
        natural_language="run the full test suite",
        resolved_shell="cd /opt/app && pytest -v --tb=short",
        ssh_host="staging.example.com",
        outcome=TranslationOutcome.APPROVED,
        model_id="dataiku-mesh-gpt4",
    )
    path = save(wiki_root, translation)

    # Look up past translations for context
    similar = find_by_query(wiki_root, "test suite")
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument

__all__ = [
    "CommandTranslation",
    "TranslationOutcome",
    "find_by_query",
    "list_all",
    "load",
    "save",
]

logger = logging.getLogger(__name__)

_TRANSLATIONS_DIR = "pages/daemon/translations"
_WIKI_TAGS = ["daemon", "command-translation"]
_WIKI_TYPE = "command-translation"
_MAX_SLUG_LENGTH = 60


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TranslationOutcome(Enum):
    """Outcome of the human approval step for a translation."""

    APPROVED = "approved"
    DENIED = "denied"
    EDITED = "edited"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _generate_translation_id() -> str:
    """Generate a unique translation identifier."""
    return str(uuid.uuid4())


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CommandTranslation:
    """Immutable record of a natural-language to shell-command translation.

    Each instance represents one NL-to-command mapping that was presented
    to the human for approval. Stores the original request, the resolved
    shell command, the SSH host context, and the approval outcome.

    Attributes:
        translation_id: Unique identifier for this translation.
        natural_language: The original natural-language command request.
        resolved_shell: The shell command the LLM generated.
        ssh_host: The SSH host this translation was targeted at.
        outcome: Whether the human approved, denied, or edited.
        model_id: Which LLM model produced the translation.
        created_at: UTC timestamp when the translation was created.
    """

    natural_language: str
    resolved_shell: str
    ssh_host: str
    outcome: TranslationOutcome = TranslationOutcome.APPROVED
    model_id: str = ""
    translation_id: str = field(default_factory=_generate_translation_id)
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        if not self.natural_language.strip():
            raise ValueError("natural_language must not be empty")
        if not self.resolved_shell.strip():
            raise ValueError("resolved_shell must not be empty")
        if not self.ssh_host.strip():
            raise ValueError("ssh_host must not be empty")


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug.

    Lowercase, replace non-alphanumeric characters with hyphens,
    collapse multiple hyphens, strip leading/trailing hyphens,
    and truncate to _MAX_SLUG_LENGTH characters.
    """
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    if len(slug) > _MAX_SLUG_LENGTH:
        slug = slug[:_MAX_SLUG_LENGTH].rstrip("-")
    return slug or "translation"


def _build_filename(translation: CommandTranslation) -> str:
    """Build a unique filename from a translation.

    Format: {nl-slug}--{short-id}.md
    The short ID ensures uniqueness even for identical NL strings.
    """
    nl_slug = _slugify(translation.natural_language)
    short_id = translation.translation_id[:8]
    return f"{nl_slug}--{short_id}.md"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO 8601 string, or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _iso_to_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 string to timezone-aware datetime, or None.

    If the parsed datetime is naive, UTC is assumed.
    """
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _translation_to_frontmatter(
    translation: CommandTranslation,
) -> dict[str, Any]:
    """Convert a CommandTranslation to a YAML-serializable frontmatter dict."""
    return {
        "tags": list(_WIKI_TAGS),
        "type": _WIKI_TYPE,
        "translation_id": translation.translation_id,
        "natural_language": translation.natural_language,
        "resolved_shell": translation.resolved_shell,
        "ssh_host": translation.ssh_host,
        "outcome": translation.outcome.value,
        "model_id": translation.model_id,
        "created": _datetime_to_iso(translation.created_at),
    }


def _frontmatter_to_translation(
    fm: dict[str, Any],
) -> CommandTranslation:
    """Reconstruct a CommandTranslation from parsed frontmatter."""
    return CommandTranslation(
        translation_id=fm["translation_id"],
        natural_language=fm["natural_language"],
        resolved_shell=fm["resolved_shell"],
        ssh_host=fm["ssh_host"],
        outcome=TranslationOutcome(fm.get("outcome", "approved")),
        model_id=fm.get("model_id", ""),
        created_at=_iso_to_datetime(fm.get("created")) or _now_utc(),
    )


def _build_body(translation: CommandTranslation) -> str:
    """Generate the human-readable markdown body for a translation page."""
    outcome_label = translation.outcome.value.capitalize()
    lines = [
        "# Command Translation",
        "",
        f"*NL-to-command mapping -- outcome: {outcome_label}*",
        "",
        "## Natural Language Request",
        "",
        f"> {translation.natural_language}",
        "",
        "## Resolved Shell Command",
        "",
        "```bash",
        translation.resolved_shell,
        "```",
        "",
        "## Context",
        "",
        f"- **SSH Host:** {translation.ssh_host}",
        f"- **Outcome:** {outcome_label}",
        f"- **Model:** {translation.model_id or 'unknown'}",
        f"- **Created:** {_datetime_to_iso(translation.created_at)}",
        f"- **Translation ID:** {translation.translation_id}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------


def _translations_dir(wiki_root: Path) -> Path:
    """Resolve the translations directory path."""
    return wiki_root / _TRANSLATIONS_DIR


def _ensure_directory(path: Path) -> None:
    """Create parent directories if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _find_file_by_id(
    wiki_root: Path,
    translation_id: str,
) -> Optional[Path]:
    """Find the wiki file for a given translation ID.

    Scans the translations directory for a file whose frontmatter
    contains the matching translation_id.
    """
    translations_path = _translations_dir(wiki_root)
    if not translations_path.exists():
        return None

    short_id = translation_id[:8]

    # Fast path: check files ending with the short ID
    for md_file in translations_path.glob(f"*--{short_id}.md"):
        if md_file.is_file():
            return md_file

    # Slow path: scan all files and check frontmatter
    for md_file in sorted(translations_path.glob("*.md")):
        if not md_file.is_file():
            continue
        try:
            raw = md_file.read_text(encoding="utf-8")
            doc = frontmatter.parse(raw)
            if doc.frontmatter.get("translation_id") == translation_id:
                return md_file
        except (ValueError, OSError):
            continue

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save(wiki_root: Path, translation: CommandTranslation) -> Path:
    """Save a command translation as a wiki page.

    Creates the file and parent directories if needed. Each translation
    gets its own file in the translations directory.

    Args:
        wiki_root: Path to the wiki root directory.
        translation: The translation record to persist.

    Returns:
        Path to the written wiki file.
    """
    filename = _build_filename(translation)
    file_path = _translations_dir(wiki_root) / filename
    _ensure_directory(file_path)

    doc = WikiDocument(
        frontmatter=_translation_to_frontmatter(translation),
        body=_build_body(translation),
    )
    content = frontmatter.serialize(doc)

    # Atomic write: write to temp file then rename
    tmp_path = file_path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(str(tmp_path), str(file_path))

    logger.info(
        "Saved command translation %s to %s",
        translation.translation_id,
        file_path,
    )
    return file_path


def load(
    wiki_root: Path,
    translation_id: str,
) -> Optional[CommandTranslation]:
    """Load a specific command translation by its ID.

    Args:
        wiki_root: Path to the wiki root directory.
        translation_id: The unique translation identifier.

    Returns:
        The deserialized CommandTranslation, or None if not found.
    """
    file_path = _find_file_by_id(wiki_root, translation_id)
    if file_path is None:
        return None

    try:
        raw = file_path.read_text(encoding="utf-8")
        doc = frontmatter.parse(raw)
        return _frontmatter_to_translation(doc.frontmatter)
    except (ValueError, KeyError, OSError) as exc:
        logger.warning(
            "Failed to load translation from %s: %s",
            file_path,
            exc,
        )
        return None


def list_all(wiki_root: Path) -> list[CommandTranslation]:
    """List all stored command translations, most recent first.

    Scans the translations directory for valid wiki pages and returns
    them sorted by created_at descending (newest first).

    Malformed files are logged and skipped rather than raising.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        List of CommandTranslation records, newest first.
    """
    translations_path = _translations_dir(wiki_root)
    if not translations_path.exists():
        return []

    results: list[CommandTranslation] = []

    for md_file in sorted(translations_path.glob("*.md")):
        if not md_file.is_file():
            continue
        try:
            raw = md_file.read_text(encoding="utf-8")
            doc = frontmatter.parse(raw)
            if doc.frontmatter.get("type") != _WIKI_TYPE:
                continue
            translation = _frontmatter_to_translation(doc.frontmatter)
            results.append(translation)
        except (ValueError, KeyError, OSError) as exc:
            logger.warning(
                "Skipping malformed translation file %s: %s",
                md_file,
                exc,
            )
            continue

    # Sort by created_at descending (most recent first)
    results.sort(key=lambda t: t.created_at, reverse=True)
    return results


def find_by_query(
    wiki_root: Path,
    query: str,
    *,
    ssh_host: Optional[str] = None,
    max_results: int = 10,
) -> list[CommandTranslation]:
    """Find translations matching a search query.

    Performs case-insensitive substring matching against both the
    natural_language and resolved_shell fields. Optionally filters
    by SSH host.

    Args:
        wiki_root: Path to the wiki root directory.
        query: Search string to match against NL and shell fields.
        ssh_host: If provided, only return translations for this host.
        max_results: Maximum number of results to return.

    Returns:
        Matching translations, most recent first, up to max_results.
    """
    if not query.strip():
        return []

    all_translations = list_all(wiki_root)
    query_lower = query.lower()

    matches: list[CommandTranslation] = []
    for translation in all_translations:
        # Filter by SSH host if specified
        if ssh_host is not None and translation.ssh_host != ssh_host:
            continue

        # Case-insensitive substring match on NL and shell
        nl_lower = translation.natural_language.lower()
        shell_lower = translation.resolved_shell.lower()
        if query_lower in nl_lower or query_lower in shell_lower:
            matches.append(translation)

        if len(matches) >= max_results:
            break

    return matches
