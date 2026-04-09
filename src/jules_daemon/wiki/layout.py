"""Wiki directory layout manager -- central registry for all wiki paths.

Defines the complete directory structure for the wiki, separating
daemon-managed directories (auto-managed by the daemon for state,
results, audit, and queue data) from user-managed directories (where
users create and curate knowledge).

The wiki serves as the sole persistence backbone. This module ensures
consistent directory organization across all components.

Directory structure:
  wiki/
    index.md                         # Wiki index (auto-generated scaffold)
    pages/
      daemon/                        # Daemon-managed: run state
        current-run.md               # Active state record
        recovery-log.md              # Recovery attempt log
        history/                     # Daemon-managed: completed run archives
          run-{run_id}.md
        results/                     # Daemon-managed: test result entries
          result-{run_id}.md
        translations/                # Daemon-managed: NL-to-command mappings
          {slug}--{id}.md
        audit/                       # Daemon-managed: per-command audit trail
          audit-{event_id}.md
          archive/                   # Daemon-managed: archived audit logs
            audit-{event_id}.md
        queue/                       # Daemon-managed: pending command queue
          {seq}-{run_id}.md
      agents/                        # User-managed: agent documentation
      architecture/                  # User-managed: architecture notes
      concepts/                      # User-managed: general concepts
      security/                      # User-managed: security notes
      tools-and-sdks/                # User-managed: tool documentation
    raw/                             # User-managed: unprocessed research notes
    schema/                          # User-managed: schema documentation

Usage:
    from jules_daemon.wiki.layout import (
        get_layout,
        initialize_wiki,
        resolve_path,
        validate_wiki,
    )

    layout = get_layout()
    initialize_wiki(wiki_root)
    history_dir = resolve_path(wiki_root, "pages/daemon/history")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument

__all__ = [
    "DAEMON_MANAGED_DIRS",
    "USER_MANAGED_DIRS",
    "DirectoryKind",
    "WikiDirectory",
    "WikiLayout",
    "WikiValidationResult",
    "get_layout",
    "initialize_wiki",
    "resolve_path",
    "validate_wiki",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data models
# ---------------------------------------------------------------------------


class DirectoryKind(Enum):
    """Classification of wiki directory ownership."""

    DAEMON_MANAGED = "daemon_managed"
    USER_MANAGED = "user_managed"


@dataclass(frozen=True)
class WikiDirectory:
    """Immutable descriptor for a single wiki directory.

    Attributes:
        relative_path: Path relative to wiki root (e.g. "pages/daemon/history").
        kind: Whether daemon-managed or user-managed.
        description: Human-readable purpose of this directory.
    """

    relative_path: str
    kind: DirectoryKind
    description: str

    @property
    def is_daemon_managed(self) -> bool:
        """True if this directory is daemon-managed."""
        return self.kind == DirectoryKind.DAEMON_MANAGED

    @property
    def is_user_managed(self) -> bool:
        """True if this directory is user-managed."""
        return self.kind == DirectoryKind.USER_MANAGED

    def resolve(self, wiki_root: Path) -> Path:
        """Resolve this directory to an absolute path given the wiki root."""
        return wiki_root / self.relative_path


@dataclass(frozen=True)
class WikiValidationResult:
    """Result of validating the wiki directory structure.

    Attributes:
        is_valid: True if all expected directories and READMEs exist.
        missing_dirs: Relative paths of missing directories.
        missing_readmes: Relative paths of directories missing their README.
    """

    is_valid: bool
    missing_dirs: tuple[str, ...]
    missing_readmes: tuple[str, ...]


@dataclass(frozen=True)
class WikiLayout:
    """Complete wiki directory layout with lookup methods.

    Provides typed access to all registered directories and convenience
    methods for filtering by kind and looking up by path.
    """

    daemon_dirs: tuple[WikiDirectory, ...]
    user_dirs: tuple[WikiDirectory, ...]

    @property
    def all_dirs(self) -> tuple[WikiDirectory, ...]:
        """All registered directories (daemon + user)."""
        return self.daemon_dirs + self.user_dirs

    def find_by_path(self, relative_path: str) -> WikiDirectory | None:
        """Find a directory by its relative path, or None if not registered."""
        for d in self.all_dirs:
            if d.relative_path == relative_path:
                return d
        return None

    def find_by_kind(self, kind: DirectoryKind) -> tuple[WikiDirectory, ...]:
        """Return all directories matching the given kind."""
        return tuple(d for d in self.all_dirs if d.kind == kind)


# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------


DAEMON_MANAGED_DIRS: tuple[WikiDirectory, ...] = (
    WikiDirectory(
        relative_path="pages/daemon",
        kind=DirectoryKind.DAEMON_MANAGED,
        description="Daemon state files (current-run, recovery-log)",
    ),
    WikiDirectory(
        relative_path="pages/daemon/history",
        kind=DirectoryKind.DAEMON_MANAGED,
        description="Completed run history (one file per terminal run)",
    ),
    WikiDirectory(
        relative_path="pages/daemon/results",
        kind=DirectoryKind.DAEMON_MANAGED,
        description="Assembled test results (one file per run)",
    ),
    WikiDirectory(
        relative_path="pages/daemon/translations",
        kind=DirectoryKind.DAEMON_MANAGED,
        description="NL-to-command translation mappings for learning",
    ),
    WikiDirectory(
        relative_path="pages/daemon/audit",
        kind=DirectoryKind.DAEMON_MANAGED,
        description="Per-command audit trail (one file per execution event)",
    ),
    WikiDirectory(
        relative_path="pages/daemon/audit/archive",
        kind=DirectoryKind.DAEMON_MANAGED,
        description="Archived audit logs (moved with explicit user approval)",
    ),
    WikiDirectory(
        relative_path="pages/daemon/queue",
        kind=DirectoryKind.DAEMON_MANAGED,
        description="Pending command queue (one file per queued command)",
    ),
)

USER_MANAGED_DIRS: tuple[WikiDirectory, ...] = (
    WikiDirectory(
        relative_path="pages/agents",
        kind=DirectoryKind.USER_MANAGED,
        description="Agent documentation and research",
    ),
    WikiDirectory(
        relative_path="pages/architecture",
        kind=DirectoryKind.USER_MANAGED,
        description="Architecture notes and design decisions",
    ),
    WikiDirectory(
        relative_path="pages/concepts",
        kind=DirectoryKind.USER_MANAGED,
        description="General concepts and knowledge base",
    ),
    WikiDirectory(
        relative_path="pages/security",
        kind=DirectoryKind.USER_MANAGED,
        description="Security notes, audits, and patterns",
    ),
    WikiDirectory(
        relative_path="pages/tools-and-sdks",
        kind=DirectoryKind.USER_MANAGED,
        description="Tool and SDK documentation",
    ),
    WikiDirectory(
        relative_path="raw",
        kind=DirectoryKind.USER_MANAGED,
        description="Unprocessed research notes and raw material",
    ),
    WikiDirectory(
        relative_path="schema",
        kind=DirectoryKind.USER_MANAGED,
        description="Schema documentation and reference",
    ),
)


# ---------------------------------------------------------------------------
# Internal: path lookup index
# ---------------------------------------------------------------------------


def _build_path_index() -> dict[str, WikiDirectory]:
    """Build a dict mapping relative_path -> WikiDirectory for fast lookup."""
    index: dict[str, WikiDirectory] = {}
    for d in DAEMON_MANAGED_DIRS:
        index[d.relative_path] = d
    for d in USER_MANAGED_DIRS:
        index[d.relative_path] = d
    return index


_PATH_INDEX: dict[str, WikiDirectory] = _build_path_index()


# ---------------------------------------------------------------------------
# Internal: README generation
# ---------------------------------------------------------------------------


_README_TAGS_DAEMON = ["daemon", "wiki-structure"]
_README_TAGS_USER = ["wiki-structure"]
_README_TYPE = "wiki-directory"


def _build_readme(directory: WikiDirectory) -> str:
    """Generate a Karpathy-style wiki README for a directory.

    Produces YAML frontmatter with directory metadata and a markdown
    body describing the directory's purpose and ownership.
    """
    tags = (
        list(_README_TAGS_DAEMON)
        if directory.is_daemon_managed
        else list(_README_TAGS_USER)
    )
    fm = {
        "tags": tags,
        "type": _README_TYPE,
        "kind": directory.kind.value,
        "path": directory.relative_path,
    }
    kind_label = (
        "Daemon-Managed" if directory.is_daemon_managed else "User-Managed"
    )

    body_lines = [
        f"# {directory.relative_path.split('/')[-1].replace('-', ' ').title()}",
        "",
        f"*{kind_label} directory*",
        "",
        f"{directory.description}",
        "",
        "## Ownership",
        "",
    ]

    if directory.is_daemon_managed:
        body_lines.extend([
            "This directory is managed by the daemon. Files here are",
            "created, updated, and archived automatically. Do not",
            "manually edit files in this directory.",
        ])
    else:
        body_lines.extend([
            "This directory is user-managed. Create, edit, and organize",
            "files here as needed. The daemon will not modify files",
            "in this directory.",
        ])

    body_lines.append("")
    body = "\n".join(body_lines)

    doc = WikiDocument(frontmatter=fm, body=body)
    return frontmatter.serialize(doc)


def _build_index(layout: WikiLayout) -> str:
    """Generate the wiki root index.md with frontmatter.

    Lists all directories organized by kind (daemon vs user).
    """
    fm = {
        "tags": ["wiki-structure", "index"],
        "type": "wiki-index",
    }

    body_lines = [
        "# Wiki Index",
        "",
        "Central knowledge base and daemon persistence backbone.",
        "",
        "## Daemon-Managed Directories",
        "",
        "These directories are automatically managed by the daemon.",
        "Do not manually edit files in daemon-managed directories.",
        "",
    ]

    for d in layout.daemon_dirs:
        dirname = d.relative_path.split("/")[-1]
        body_lines.append(f"- **{dirname}** (`{d.relative_path}/`) -- {d.description}")

    body_lines.extend([
        "",
        "## User-Managed Directories",
        "",
        "These directories are for user-curated knowledge and notes.",
        "",
    ])

    for d in layout.user_dirs:
        dirname = d.relative_path.split("/")[-1]
        body_lines.append(f"- **{dirname}** (`{d.relative_path}/`) -- {d.description}")

    body_lines.append("")
    body = "\n".join(body_lines)

    doc = WikiDocument(frontmatter=fm, body=body)
    return frontmatter.serialize(doc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_layout() -> WikiLayout:
    """Return the complete wiki layout descriptor.

    Returns:
        WikiLayout with all daemon-managed and user-managed directories.
    """
    return WikiLayout(
        daemon_dirs=DAEMON_MANAGED_DIRS,
        user_dirs=USER_MANAGED_DIRS,
    )


def resolve_path(wiki_root: Path, relative_path: str) -> Path:
    """Resolve a registered wiki directory to an absolute path.

    Args:
        wiki_root: Path to the wiki root directory.
        relative_path: Relative path to a registered directory.

    Returns:
        Absolute path to the directory.

    Raises:
        KeyError: If the relative_path is not a registered directory.
    """
    if relative_path not in _PATH_INDEX:
        raise KeyError(
            f"'{relative_path}' is not a registered wiki directory"
        )
    return wiki_root / relative_path


def initialize_wiki(wiki_root: Path) -> list[Path]:
    """Initialize the wiki directory structure.

    Creates all registered directories and their README files if they
    do not already exist. Existing files are never overwritten -- this
    operation is idempotent.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        List of paths that were newly created (directories and files).
    """
    layout = get_layout()
    created: list[Path] = []

    # Ensure wiki root exists
    if not wiki_root.exists():
        wiki_root.mkdir(parents=True, exist_ok=True)
        created.append(wiki_root)

    # Create all directories and their READMEs
    for d in layout.all_dirs:
        dir_path = wiki_root / d.relative_path
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created.append(dir_path)

        readme_path = dir_path / "README.md"
        if not readme_path.exists():
            content = _build_readme(d)
            readme_path.write_text(content, encoding="utf-8")
            created.append(readme_path)

    # Create root index if it does not exist
    index_path = wiki_root / "index.md"
    if not index_path.exists():
        content = _build_index(layout)
        index_path.write_text(content, encoding="utf-8")
        created.append(index_path)

    if created:
        logger.info(
            "Initialized wiki at %s: created %d items",
            wiki_root,
            len(created),
        )
    else:
        logger.debug("Wiki at %s is already initialized", wiki_root)

    return created


def validate_wiki(wiki_root: Path) -> WikiValidationResult:
    """Validate the wiki directory structure.

    Checks that all registered directories exist and contain their
    README files. Returns a structured result describing any gaps.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        WikiValidationResult with validity status and any missing items.
    """
    layout = get_layout()
    missing_dirs: list[str] = []
    missing_readmes: list[str] = []

    for d in layout.all_dirs:
        dir_path = wiki_root / d.relative_path
        if not dir_path.is_dir():
            missing_dirs.append(d.relative_path)
        else:
            readme_path = dir_path / "README.md"
            if not readme_path.exists():
                missing_readmes.append(d.relative_path)

    is_valid = len(missing_dirs) == 0 and len(missing_readmes) == 0

    if not is_valid:
        logger.warning(
            "Wiki validation failed: %d missing dirs, %d missing READMEs",
            len(missing_dirs),
            len(missing_readmes),
        )

    return WikiValidationResult(
        is_valid=is_valid,
        missing_dirs=tuple(missing_dirs),
        missing_readmes=tuple(missing_readmes),
    )
