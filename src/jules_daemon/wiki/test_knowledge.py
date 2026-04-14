"""Wiki persistence for accumulated per-test knowledge.

Each known test command has one wiki page that accumulates durable
observations across runs (purpose, output format, normal behavior,
common failures). The summarizer reads this knowledge before each run
so the LLM can produce richer narratives, and the knowledge extractor
appends fresh observations after each run.

Wiki file location: ``{wiki_root}/pages/daemon/knowledge/test-{slug}.md``

The slug is derived from the command string by stripping arguments and
keeping the first meaningful path/script tokens. Identical test
commands collapse to the same slug so knowledge accumulates correctly
across runs.

Design notes:

- All I/O failures are caught and logged at warning level. The audit
  flow must never break because knowledge can not be loaded or saved.
- Each ``TestKnowledge`` is a frozen dataclass; merges return a new
  instance instead of mutating in place.
- The wiki file uses the standard Karpathy YAML-frontmatter format so
  humans can edit the body and the daemon will not clobber their
  changes (the merge strategy preserves existing values).

Usage::

    from jules_daemon.wiki.test_knowledge import (
        derive_test_slug,
        load_test_knowledge,
        merge_knowledge,
        save_test_knowledge,
    )

    slug = derive_test_slug("python3 ~/agent_test.py --iterations 100")
    existing = load_test_knowledge(wiki_root, slug)
    merged = merge_knowledge(existing, new_observations)
    save_test_knowledge(wiki_root, merged)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument

__all__ = [
    "TestKnowledge",
    "KNOWLEDGE_DIR",
    "derive_test_slug",
    "knowledge_file_path",
    "load_test_knowledge",
    "merge_knowledge",
    "save_test_knowledge",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


KNOWLEDGE_DIR: str = "pages/daemon/knowledge"
"""Wiki-relative directory holding per-test knowledge pages."""

_WIKI_TAGS: tuple[str, ...] = ("daemon", "test-knowledge", "learning")
_WIKI_TYPE: str = "test-knowledge"

# Maximum length of the slug component appended to ``test-`` filenames.
# Long enough to disambiguate the typical command, short enough that
# filesystem path limits stay safe on every platform.
_MAX_SLUG_LENGTH: int = 60

# Maximum number of common failure patterns we keep on a knowledge page.
# Beyond this the merge strategy drops the oldest entries to bound the
# wiki file size and keep summarization prompts compact.
_MAX_COMMON_FAILURES: int = 10

# Tokens that frequently appear at the start of a command but do not
# tell us anything about the test itself. They are stripped before slug
# derivation so ``python3 ~/agent_test.py`` and ``./agent_test.py``
# produce the same slug.
_INTERPRETER_PREFIXES: frozenset[str] = frozenset({
    "python",
    "python2",
    "python3",
    "python3.8",
    "python3.9",
    "python3.10",
    "python3.11",
    "python3.12",
    "python3.13",
    "python3.14",
    "py",
    "py3",
    "node",
    "ruby",
    "perl",
    "php",
    "java",
    "bash",
    "sh",
    "zsh",
    "fish",
    "exec",
})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TestKnowledge:
    """Accumulated knowledge about a specific test command.

    Each instance summarizes everything the daemon has learned about
    one test by observing its runs over time. The fields are:

    Attributes:
        test_slug: Filesystem-safe slug derived from the command, used
            both as the wiki filename suffix and as a stable identifier.
        command_pattern: Canonical command string (typically the first
            command this slug saw) -- helpful for debugging the slug
            derivation when several commands collide.
        purpose: One sentence describing what the test does.
        output_format: Short description of how to read the test output
            (counts, log lines, custom format).
        summary_fields: Ordered tuple of fields that matter most when
            summarizing status or completion for this test (for example
            ``("passed", "failed", "iterations_done")``).
        common_failures: Tuple of short failure patterns the test has
            exhibited in past runs. Capped at :data:`_MAX_COMMON_FAILURES`.
        normal_behavior: Description of what a healthy run looks like.
        required_args: Tuple of argument names that this test requires
            (e.g., ``("iterations", "host")``). Populated from the user's
            starter spec in the wiki. The agent loop uses this to detect
            missing arguments and prompt the user via ask_user_question.
        runs_observed: Number of completed runs the daemon has learned
            from. Incremented on each merge.
        last_updated: UTC timestamp of the most recent merge.
    """

    # Tell pytest not to try collecting this dataclass as a test class
    # (the ``Test`` prefix is semantic, not pytest-specific).
    __test__ = False

    test_slug: str
    command_pattern: str
    purpose: str = ""
    output_format: str = ""
    summary_fields: tuple[str, ...] = field(default_factory=tuple)
    common_failures: tuple[str, ...] = field(default_factory=tuple)
    normal_behavior: str = ""
    required_args: tuple[str, ...] = field(default_factory=tuple)
    runs_observed: int = 0
    last_updated: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        if not self.test_slug.strip():
            raise ValueError("test_slug must not be empty")
        if not self.command_pattern.strip():
            raise ValueError("command_pattern must not be empty")
        if self.runs_observed < 0:
            raise ValueError("runs_observed must be non-negative")

    def to_prompt_context(self) -> str:
        """Format this knowledge as context for the summarizer LLM prompt.

        Returns a multi-line string with only the populated sections.
        Empty fields are omitted so the prompt stays scannable. The
        rendered text is intended to be embedded inside a system or
        user message above the actual test output.
        """
        lines: list[str] = [
            "Prior knowledge about this test (from past runs):",
        ]
        if self.purpose:
            lines.append(f"- Purpose: {self.purpose}")
        if self.output_format:
            lines.append(f"- Output format: {self.output_format}")
        if self.summary_fields:
            lines.append(
                f"- Summary fields: {', '.join(self.summary_fields)}"
            )
        if self.normal_behavior:
            lines.append(f"- Normal behavior: {self.normal_behavior}")
        if self.required_args:
            lines.append(
                f"- Required arguments: {', '.join(self.required_args)}"
            )
        if self.common_failures:
            lines.append("- Common failure patterns:")
            for failure in self.common_failures:
                lines.append(f"  - {failure}")
        if self.runs_observed:
            lines.append(
                f"- Observed across {self.runs_observed} prior run(s)."
            )
        # If only the header line is present, no prior knowledge has
        # been captured yet -- return an empty string so callers can
        # avoid sending an empty section to the LLM.
        if len(lines) == 1:
            return ""
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Slug derivation
# ---------------------------------------------------------------------------


def _strip_quotes(token: str) -> str:
    """Remove leading/trailing quotes from a command token."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


def _looks_like_flag(token: str) -> bool:
    """Return True if the token is a CLI flag/option."""
    return token.startswith("-")


def _slugify_segment(segment: str) -> str:
    """Convert a path/command segment into a hyphen-separated slug.

    Lowercases, replaces non-alphanumeric runs with hyphens, collapses
    duplicate hyphens, and trims leading/trailing hyphens.
    """
    lowered = segment.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered)
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned.strip("-")


def _meaningful_token(token: str, *, keep_path: bool = False) -> Optional[str]:
    """Return the slug fragment for a token, or ``None`` to skip it.

    Strips interpreter prefixes (``python``, ``node``, ...) and leading
    path noise. When ``keep_path`` is False (the default) only the
    basename of the token is used so that ``python3 ~/agent_test.py``
    collapses to ``agent-test-py``. When ``keep_path`` is True the
    intermediate directories are preserved so ``tests/integration``
    becomes ``tests-integration`` -- this is what we want for runner
    arguments where the path itself is the test identity.
    """
    if not token:
        return None
    token = _strip_quotes(token)
    if not token:
        return None
    if _looks_like_flag(token):
        return None

    # Drop redirections, env-var assignments, etc.
    if "=" in token and not ("/" in token or "." in token):
        return None

    # If the token is an interpreter, skip it.
    base_only = os.path.basename(token.rstrip("/"))
    if base_only.lower() in _INTERPRETER_PREFIXES:
        return None

    if keep_path:
        # Keep intermediate directories so ``tests/integration`` and
        # ``tests/unit`` produce different slugs. Strip leading ``./``
        # or ``~/`` so they do not pollute the slug.
        cleaned = token.lstrip()
        if cleaned.startswith("./"):
            cleaned = cleaned[2:]
        elif cleaned.startswith("~/"):
            cleaned = cleaned[2:]
        cleaned = cleaned.rstrip("/")
        if not cleaned:
            return None
        slug = _slugify_segment(cleaned)
        return slug or None

    # Default: only the basename. ``agent_test.py`` becomes
    # ``agent-test-py`` so the slug contains the runtime hint without
    # the directory noise.
    base = base_only.lstrip("./").lstrip("~/")
    if not base:
        return None
    slug = _slugify_segment(base)
    return slug or None


def derive_test_slug(command: str) -> str:
    """Extract a filesystem-safe slug from a command string.

    The strategy is intentionally conservative: it picks the first
    "meaningful" token (skipping interpreters and flags) and slugifies
    it. The slug is also augmented with the next non-flag positional
    when the first meaningful token is a known runner like ``pytest``,
    ``npm``, or ``cargo`` so that two pytest invocations against
    different paths do not collapse to the same slug.

    Args:
        command: The full shell command string.

    Returns:
        A slug string suitable for use in filenames. Always non-empty;
        falls back to ``"unknown-test"`` for empty inputs.

    Examples:
        ``python3.8 ~/agent_test.py --name JJ`` -> ``agent-test-py``
        ``pytest tests/integration/ -v`` -> ``pytest-tests-integration``
        ``npm test`` -> ``npm-test``
        ``./run.sh`` -> ``run-sh``
    """
    if not command or not command.strip():
        return "unknown-test"

    tokens = command.strip().split()

    # Walk tokens until we find a meaningful one (skipping interpreters
    # and flags).
    primary: Optional[str] = None
    primary_index: int = -1
    for idx, raw_token in enumerate(tokens):
        candidate = _meaningful_token(raw_token)
        if candidate:
            primary = candidate
            primary_index = idx
            break

    if primary is None:
        # Everything was a flag or interpreter -- fall back to slugifying
        # the entire command. This still produces something stable for
        # weird inputs like ``--version``.
        fallback = _slugify_segment(command)
        slug = (fallback or "unknown-test")[:_MAX_SLUG_LENGTH]
        return slug.rstrip("-") or "unknown-test"

    # Detect runners like ``pytest`` / ``npm`` / ``cargo`` and capture
    # the next positional to disambiguate sibling commands.
    runner_like = primary in {
        "pytest",
        "npm",
        "npx",
        "yarn",
        "cargo",
        "go",
        "make",
        "mvn",
        "gradle",
    }

    parts: list[str] = [primary]
    if runner_like:
        for raw_token in tokens[primary_index + 1 :]:
            candidate = _meaningful_token(raw_token, keep_path=True)
            if not candidate:
                continue
            parts.append(candidate)
            break

    slug = "-".join(parts)
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        return "unknown-test"
    if len(slug) > _MAX_SLUG_LENGTH:
        slug = slug[:_MAX_SLUG_LENGTH].rstrip("-")
    return slug or "unknown-test"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _knowledge_dir(wiki_root: Path) -> Path:
    """Resolve the knowledge directory under the wiki root."""
    return wiki_root / KNOWLEDGE_DIR


def knowledge_file_path(wiki_root: Path, test_slug: str) -> Path:
    """Return the absolute wiki file path for a given test slug.

    Args:
        wiki_root: Path to the wiki root directory.
        test_slug: Slug previously produced by :func:`derive_test_slug`.

    Returns:
        Absolute path to the markdown file (which may not yet exist).
    """
    safe_slug = test_slug.strip() or "unknown-test"
    return _knowledge_dir(wiki_root) / f"test-{safe_slug}.md"


def _ensure_parent_directory(path: Path) -> None:
    """Create the parent directory for *path* if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _datetime_to_iso(dt: datetime) -> str:
    """Convert a datetime to an ISO 8601 string."""
    return dt.isoformat()


def _iso_to_datetime(value: Optional[str]) -> datetime:
    """Parse an ISO 8601 string to a timezone-aware datetime.

    Falls back to the current UTC time if the value is missing or
    malformed -- callers must never crash because of a stale wiki file.
    """
    if not value:
        return _now_utc()
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return _now_utc()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _knowledge_to_frontmatter(knowledge: TestKnowledge) -> dict[str, Any]:
    """Convert a :class:`TestKnowledge` to a YAML-friendly dict."""
    return {
        "tags": list(_WIKI_TAGS),
        "type": _WIKI_TYPE,
        "test_slug": knowledge.test_slug,
        "command_pattern": knowledge.command_pattern,
        "purpose": knowledge.purpose,
        "output_format": knowledge.output_format,
        "summary_fields": list(knowledge.summary_fields),
        "normal_behavior": knowledge.normal_behavior,
        "required_args": list(knowledge.required_args),
        "common_failures": list(knowledge.common_failures),
        "runs_observed": knowledge.runs_observed,
        "last_updated": _datetime_to_iso(knowledge.last_updated),
    }


def _coerce_string(value: Any) -> str:
    """Coerce a YAML scalar to a stripped string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_failures(value: Any) -> tuple[str, ...]:
    """Coerce a YAML list (or scalar) into a bounded tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    cleaned: list[str] = []
    for item in items:
        text = _coerce_string(item)
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= _MAX_COMMON_FAILURES:
            break
    return tuple(cleaned)


def _coerce_required_args(value: Any) -> tuple[str, ...]:
    """Coerce a YAML list (or scalar) into a tuple of argument name strings."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    cleaned: list[str] = []
    for item in items:
        text = _coerce_string(item)
        if text and text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned)


def _coerce_runs_observed(value: Any) -> int:
    """Coerce the runs_observed field, defaulting to 0 on bad input."""
    if value is None:
        return 0
    try:
        runs = int(value)
    except (TypeError, ValueError):
        return 0
    return max(runs, 0)


def _frontmatter_to_knowledge(fm: dict[str, Any]) -> TestKnowledge:
    """Reconstruct a :class:`TestKnowledge` from parsed frontmatter.

    Missing fields fall back to safe defaults so a partially curated
    file (one that a human edited by hand) still loads cleanly.
    """
    test_slug = _coerce_string(fm.get("test_slug")) or "unknown-test"
    command_pattern = (
        _coerce_string(fm.get("command_pattern"))
        or _coerce_string(fm.get("command_template"))
        or test_slug
    )
    return TestKnowledge(
        test_slug=test_slug,
        command_pattern=command_pattern,
        purpose=_coerce_string(fm.get("purpose")),
        output_format=_coerce_string(fm.get("output_format")),
        summary_fields=_coerce_required_args(fm.get("summary_fields")),
        normal_behavior=_coerce_string(fm.get("normal_behavior")),
        required_args=_coerce_required_args(fm.get("required_args")),
        common_failures=_coerce_failures(fm.get("common_failures")),
        runs_observed=_coerce_runs_observed(fm.get("runs_observed")),
        last_updated=_iso_to_datetime(fm.get("last_updated")),
    )


def _build_body(knowledge: TestKnowledge) -> str:
    """Generate the human-readable markdown body for the wiki page."""
    lines: list[str] = [
        f"# Test Knowledge: {knowledge.test_slug}",
        "",
        "*Auto-curated knowledge accumulated across runs of this test.*",
        "*You may edit this file by hand -- the daemon will preserve "
        "non-empty fields when merging new observations.*",
        "",
        "## Command Pattern",
        "",
        "```bash",
        knowledge.command_pattern,
        "```",
        "",
    ]
    if knowledge.purpose:
        lines.extend([
            "## Purpose",
            "",
            knowledge.purpose,
            "",
        ])
    if knowledge.output_format:
        lines.extend([
            "## Output Format",
            "",
            knowledge.output_format,
            "",
        ])
    if knowledge.summary_fields:
        lines.extend([
            "## Summary Fields",
            "",
        ])
        for field_name in knowledge.summary_fields:
            lines.append(f"- `{field_name}`")
        lines.append("")
    if knowledge.normal_behavior:
        lines.extend([
            "## Normal Behavior",
            "",
            knowledge.normal_behavior,
            "",
        ])
    if knowledge.required_args:
        lines.extend([
            "## Required Arguments",
            "",
        ])
        for arg in knowledge.required_args:
            lines.append(f"- `{arg}`")
        lines.append("")
    if knowledge.common_failures:
        lines.extend([
            "## Common Failures",
            "",
        ])
        for failure in knowledge.common_failures:
            lines.append(f"- {failure}")
        lines.append("")
    lines.extend([
        "## Statistics",
        "",
        f"- Runs observed: {knowledge.runs_observed}",
        f"- Last updated: {_datetime_to_iso(knowledge.last_updated)}",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_test_knowledge(
    wiki_root: Path,
    test_slug: str,
) -> Optional[TestKnowledge]:
    """Load existing knowledge for a test from the wiki.

    Args:
        wiki_root: Path to the wiki root directory.
        test_slug: Slug previously produced by :func:`derive_test_slug`.

    Returns:
        The parsed :class:`TestKnowledge`, or ``None`` if the file does
        not exist or cannot be parsed (errors are logged at warning
        level rather than raised).
    """
    if not test_slug:
        return None
    file_path = knowledge_file_path(wiki_root, test_slug)
    if not file_path.is_file():
        return None
    try:
        raw = file_path.read_text(encoding="utf-8")
        doc = frontmatter.parse(raw)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Failed to read test knowledge from %s: %s", file_path, exc
        )
        return None
    try:
        return _frontmatter_to_knowledge(doc.frontmatter)
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning(
            "Failed to deserialize test knowledge at %s: %s",
            file_path,
            exc,
        )
        return None


def save_test_knowledge(
    wiki_root: Path,
    knowledge: TestKnowledge,
) -> Path:
    """Persist a knowledge record to the wiki via an atomic write.

    The write is performed by writing to a temp file and then renaming
    it on top of the destination, so concurrent readers always observe
    a complete file.

    Args:
        wiki_root: Path to the wiki root directory.
        knowledge: The record to persist.

    Returns:
        The absolute path that was written.
    """
    file_path = knowledge_file_path(wiki_root, knowledge.test_slug)
    _ensure_parent_directory(file_path)

    doc = WikiDocument(
        frontmatter=_knowledge_to_frontmatter(knowledge),
        body=_build_body(knowledge),
    )
    content = frontmatter.serialize(doc)

    tmp_path = file_path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(str(tmp_path), str(file_path))
    logger.debug("Saved test knowledge for %s -> %s", knowledge.test_slug, file_path)
    return file_path


def merge_knowledge(
    existing: Optional[TestKnowledge],
    new_observations: dict[str, Any],
    *,
    test_slug: Optional[str] = None,
    command_pattern: Optional[str] = None,
) -> TestKnowledge:
    """Merge fresh LLM observations into existing knowledge.

    Strategy:

    - ``common_failures``: union of existing and new (deduped, capped at
      :data:`_MAX_COMMON_FAILURES`). New entries are appended to the end
      so the order reflects discovery time.
    - ``purpose``, ``output_format``, ``normal_behavior``: prefer the
      existing value if it is non-empty (so human-curated text is never
      overwritten); otherwise adopt the new observation.
    - ``runs_observed``: incremented by one.
    - ``last_updated``: set to the current UTC time.

    When ``existing`` is ``None`` the function constructs a fresh record
    using ``test_slug`` and ``command_pattern`` (which become required
    in that case).

    Args:
        existing: The previously stored record, or ``None`` for a first
            observation.
        new_observations: A dict from
            :func:`jules_daemon.execution.knowledge_extractor.extract_knowledge`
            with the keys ``purpose``, ``output_format``,
            ``common_failures``, ``normal_behavior`` (all optional).
        test_slug: Required when *existing* is ``None``.
        command_pattern: Required when *existing* is ``None``.

    Returns:
        A new :class:`TestKnowledge` instance with the merged values.
    """
    new_purpose = _coerce_string(new_observations.get("purpose"))
    new_format = _coerce_string(new_observations.get("output_format"))
    new_summary_fields = _coerce_required_args(
        new_observations.get("summary_fields")
    )
    new_normal = _coerce_string(new_observations.get("normal_behavior"))
    new_failures = _coerce_failures(new_observations.get("common_failures"))

    if existing is None:
        if not test_slug or not command_pattern:
            raise ValueError(
                "merge_knowledge requires test_slug and command_pattern "
                "when existing is None"
            )
        return TestKnowledge(
            test_slug=test_slug,
            command_pattern=command_pattern,
            purpose=new_purpose,
            output_format=new_format,
            summary_fields=new_summary_fields,
            normal_behavior=new_normal,
            common_failures=new_failures,
            runs_observed=1,
            last_updated=_now_utc(),
        )

    merged_failures: list[str] = list(existing.common_failures)
    for failure in new_failures:
        if failure and failure not in merged_failures:
            merged_failures.append(failure)
    if len(merged_failures) > _MAX_COMMON_FAILURES:
        merged_failures = merged_failures[-_MAX_COMMON_FAILURES:]

    return replace(
        existing,
        purpose=existing.purpose or new_purpose,
        output_format=existing.output_format or new_format,
        summary_fields=existing.summary_fields or new_summary_fields,
        normal_behavior=existing.normal_behavior or new_normal,
        common_failures=tuple(merged_failures),
        runs_observed=existing.runs_observed + 1,
        last_updated=_now_utc(),
    )
