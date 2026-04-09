"""Audit-instrumented NL input classification.

Wraps the deterministic ``classify()`` pipeline with audit instrumentation
so that every NL input parsing invocation produces a persisted audit entry
in the wiki filesystem.

The audit entry captures:
    - The raw user input (before-snapshot)
    - The classification result: verb, confidence, input type (after-snapshot)
    - Wall-clock duration of the classification
    - Success/error status

Each invocation writes exactly one audit markdown file to
``pages/daemon/audit/audit-{event_id}.md`` with YAML frontmatter and a
human-readable body. The audit file is written atomically -- if the
classification succeeds, the audit file is guaranteed to exist.

This is the first pipeline stage instrumented for audit completeness:
every command must start with an NL_INPUT audit entry.

Usage::

    from jules_daemon.classifier.nl_audit import classify_with_audit

    result = classify_with_audit("run the smoke tests", wiki_root)
    result.classification  # ClassificationResult
    result.entry           # AuditEntry
    result.chain           # AuditChain with the new entry
    result.audit_path      # Path to the written audit file
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jules_daemon.audit_models import AuditChain, AuditEntry
from jules_daemon.classifier.classify import classify
from jules_daemon.classifier.models import ClassificationResult
from jules_daemon.wiki.frontmatter import WikiDocument, serialize

__all__ = [
    "NLAuditResult",
    "classify_with_audit",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAGE_NAME: str = "nl_input"
_AUDIT_FILE_PREFIX: str = "audit-"
_AUDIT_FILE_SUFFIX: str = ".md"
_AUDIT_DIR_RELATIVE: str = "pages/daemon/audit"
_AUDIT_TYPE: str = "audit-log"


# ---------------------------------------------------------------------------
# NLAuditResult -- immutable result of instrumented classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NLAuditResult:
    """Immutable result of an audit-instrumented NL classification.

    Bundles the classification result with the audit trail artifacts:
    the AuditEntry, the updated AuditChain, and the filesystem path
    to the written audit file.

    Attributes:
        classification: The deterministic classification result.
        entry: The AuditEntry recording this invocation.
        chain: The AuditChain with the new entry appended.
        audit_path: Path to the written wiki audit markdown file.
    """

    classification: ClassificationResult
    entry: AuditEntry
    chain: AuditChain
    audit_path: Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_event_id() -> str:
    """Generate a unique event identifier for the audit file."""
    return str(uuid.uuid4())


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _build_before_snapshot(raw: str) -> dict[str, Any]:
    """Build the before-snapshot dict from raw NL input."""
    return {
        "raw_input": raw,
        "source": "cli",
    }


def _build_after_snapshot(result: ClassificationResult) -> dict[str, Any]:
    """Build the after-snapshot dict from the classification result."""
    return {
        "canonical_verb": result.canonical_verb,
        "confidence_score": result.confidence_score,
        "input_type": result.input_type.value,
        "extracted_args": dict(result.extracted_args),
    }


def _build_audit_entry(
    *,
    raw: str,
    result: ClassificationResult,
    duration: float,
    timestamp: datetime,
) -> AuditEntry:
    """Build an immutable AuditEntry for the NL input stage."""
    return AuditEntry(
        stage=_STAGE_NAME,
        timestamp=timestamp,
        before_snapshot=_build_before_snapshot(raw),
        after_snapshot=_build_after_snapshot(result),
        duration=duration,
        status="success",
        error=None,
    )


def _build_frontmatter(
    *,
    event_id: str,
    raw: str,
    result: ClassificationResult,
    entry: AuditEntry,
) -> dict[str, Any]:
    """Build YAML frontmatter dict for the audit wiki file."""
    return {
        "type": _AUDIT_TYPE,
        "event_id": event_id,
        "stage": _STAGE_NAME,
        "status": entry.status,
        "timestamp": entry.timestamp.isoformat(),
        "raw_input": raw,
        "canonical_verb": result.canonical_verb,
        "confidence_score": result.confidence_score,
        "input_type": result.input_type.value,
        "duration_seconds": entry.duration,
        "tags": ["audit", "nl-input", "daemon"],
    }


def _build_body(
    *,
    raw: str,
    result: ClassificationResult,
    event_id: str,
) -> str:
    """Build the markdown body for the audit wiki file."""
    # Sanitize raw input for safe markdown embedding: collapse
    # newlines and carriage returns so the blockquote stays intact.
    safe_raw = raw.replace("\n", " ").replace("\r", " ")

    lines = [
        "# NL Input Audit",
        "",
        f"Event: `{event_id}`",
        "",
        "## Input",
        "",
        f"> {safe_raw}" if raw.strip() else "> (empty input)",
        "",
        "## Classification",
        "",
        f"- **Verb**: {result.canonical_verb}",
        f"- **Confidence**: {result.confidence_score:.2f}",
        f"- **Input Type**: {result.input_type.value}",
        "",
    ]
    if result.extracted_args:
        lines.append("## Extracted Arguments")
        lines.append("")
        for key, value in result.extracted_args.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    return "\n".join(lines)


def _write_audit_file(
    *,
    wiki_root: Path,
    event_id: str,
    frontmatter_dict: dict[str, Any],
    body: str,
) -> Path:
    """Write the audit entry as a wiki markdown file.

    Creates the file at ``pages/daemon/audit/audit-{event_id}.md``.
    The parent directory must already exist.

    Returns:
        Path to the written file.
    """
    audit_dir = wiki_root / _AUDIT_DIR_RELATIVE
    filename = f"{_AUDIT_FILE_PREFIX}{event_id}{_AUDIT_FILE_SUFFIX}"
    audit_path = audit_dir / filename

    doc = WikiDocument(frontmatter=frontmatter_dict, body=body)
    content = serialize(doc)
    audit_path.write_text(content, encoding="utf-8")

    return audit_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_with_audit(
    raw: str,
    wiki_root: Path,
    *,
    chain: AuditChain | None = None,
) -> NLAuditResult:
    """Classify NL input with audit instrumentation.

    Wraps the deterministic ``classify()`` function with before/after
    snapshot capture, timing, and wiki persistence. Produces one audit
    file per invocation at ``pages/daemon/audit/audit-{event_id}.md``.

    The function never raises for classification failures -- it always
    returns a valid NLAuditResult. File I/O errors during audit writing
    will propagate to the caller.

    Args:
        raw: Raw user input string (may be empty).
        wiki_root: Path to the wiki root directory. The audit directory
            (``pages/daemon/audit/``) must exist.
        chain: Optional existing AuditChain to append to. If None, a
            fresh empty chain is used.

    Returns:
        Immutable NLAuditResult with classification, audit entry,
        updated chain, and path to the written audit file.
    """
    if chain is None:
        chain = AuditChain.empty()

    event_id = _generate_event_id()
    timestamp = _now_utc()

    # Time the classification
    start = time.monotonic()
    result = classify(raw)
    duration = time.monotonic() - start

    # Build the audit entry
    entry = _build_audit_entry(
        raw=raw,
        result=result,
        duration=duration,
        timestamp=timestamp,
    )

    # Append to chain (immutable)
    updated_chain = chain.append(entry)

    # Build and write wiki file
    fm = _build_frontmatter(
        event_id=event_id,
        raw=raw,
        result=result,
        entry=entry,
    )
    body = _build_body(
        raw=raw,
        result=result,
        event_id=event_id,
    )
    audit_path = _write_audit_file(
        wiki_root=wiki_root,
        event_id=event_id,
        frontmatter_dict=fm,
        body=body,
    )

    return NLAuditResult(
        classification=result,
        entry=entry,
        chain=updated_chain,
        audit_path=audit_path,
    )
