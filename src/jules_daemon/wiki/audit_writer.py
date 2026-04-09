"""Persist completed audit chains as Karpathy-style wiki pages.

Converts a completed AuditRecord (at EXECUTION_COMPLETE or DENIED stage)
into a wiki document with:
  - YAML frontmatter containing the full structured audit record for
    machine consumption (all pipeline stages serialized)
  - Markdown body with human-readable traceability from NL input through
    to execution result

Wiki file location: {wiki_root}/pages/daemon/audit/audit-{correlation_id}.md

Each completed audit chain is a standalone wiki file that preserves the
complete execution trail:
  - Correlation ID linking all stages
  - NL input (what the user asked for)
  - Parsed command (what the LLM translated it to)
  - Confirmation decision (what the human approved/denied/edited)
  - SSH execution details (where and how the command ran)
  - Structured result (what happened -- test counts, exit code, summary)

Optional: an AuditChain (generic stage-by-stage ledger) can be included
in the frontmatter for fine-grained stage timing and snapshot data.

Constraints satisfied:
  - Wiki is the sole persistence backbone (no SQLite, PID files)
  - One audit file per command execution event
  - Atomic writes (temp file + rename) prevent partial files
  - Immutable data models throughout (frozen dataclasses, no mutation)

Usage::

    from pathlib import Path
    from jules_daemon.wiki.audit_writer import write_audit, read_audit

    outcome = write_audit(wiki_root, audit_record, chain=audit_chain)
    outcome.file_path    # Path to the written wiki file
    outcome.correlation_id
    outcome.written_at

    restored = read_audit(outcome.file_path)
    restored.correlation_id == audit_record.correlation_id  # True
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jules_daemon.audit.models import (
    AuditRecord,
    ConfirmationRecord,
    NLInputRecord,
    ParsedCommandRecord,
    SSHExecutionRecord,
    StructuredResultRecord,
)
from jules_daemon.audit_models import AuditChain
from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument

__all__ = [
    "AuditWriteOutcome",
    "audit_file_path",
    "audit_to_document",
    "list_audit_files",
    "read_audit",
    "write_audit",
]

logger = logging.getLogger(__name__)

_AUDIT_DIR = "pages/daemon/audit"
_AUDIT_PREFIX = "audit-"
_AUDIT_SUFFIX = ".md"
_WIKI_TAGS: tuple[str, ...] = ("daemon", "audit-log")
_WIKI_TYPE = "audit-log"


# ---------------------------------------------------------------------------
# Immutable result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditWriteOutcome:
    """Outcome of writing an AuditRecord to the wiki.

    Carries the path to the written file, the correlation ID that
    links all pipeline stages, and the timestamp of the write.

    Attributes:
        file_path: Absolute path to the written wiki file.
        correlation_id: The audit record's correlation identifier.
        written_at: UTC timestamp when the file was written.
    """

    file_path: Path
    correlation_id: str
    written_at: datetime


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _audit_dir(wiki_root: Path) -> Path:
    """Resolve the audit directory path."""
    return wiki_root / _AUDIT_DIR


def audit_file_path(wiki_root: Path, correlation_id: str) -> Path:
    """Resolve the path for a specific audit record.

    Args:
        wiki_root: Path to the wiki root directory.
        correlation_id: The unique correlation identifier for the record.

    Returns:
        Path to the audit wiki file (may or may not exist yet).
    """
    return _audit_dir(wiki_root) / f"{_AUDIT_PREFIX}{correlation_id}{_AUDIT_SUFFIX}"


def _ensure_directory(path: Path) -> None:
    """Create parent directories if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def _datetime_to_iso(dt: datetime | None) -> str | None:
    """Convert datetime to ISO 8601 string, or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Serialization: AuditRecord -> frontmatter dict
# ---------------------------------------------------------------------------


def _record_to_frontmatter(
    record: AuditRecord,
    chain: AuditChain | None = None,
) -> dict[str, Any]:
    """Convert an AuditRecord to a YAML-serializable frontmatter dict.

    The record's ``to_dict()`` method provides the core serialization.
    This function adds wiki-specific metadata (tags, type) and optionally
    includes the AuditChain entries for fine-grained stage tracking.

    Args:
        record: The audit record to serialize.
        chain: Optional AuditChain to include as chain_entries.

    Returns:
        Dict suitable for YAML serialization as frontmatter.
    """
    record_data = record.to_dict()

    fm: dict[str, Any] = {
        "tags": list(_WIKI_TAGS),
        "type": _WIKI_TYPE,
    }
    # Merge the record data into the frontmatter
    fm.update(record_data)

    # Optionally include the audit chain entries
    if chain is not None and len(chain) > 0:
        fm["chain_entries"] = chain.to_list()

    return fm


# ---------------------------------------------------------------------------
# Deserialization: frontmatter dict -> AuditRecord
# ---------------------------------------------------------------------------


def _frontmatter_to_record(fm: dict[str, Any]) -> AuditRecord:
    """Reconstruct an AuditRecord from parsed frontmatter.

    Strips wiki-specific metadata (tags, type, chain_entries) before
    passing to ``AuditRecord.from_dict()``.

    Args:
        fm: Parsed YAML frontmatter dict.

    Returns:
        The deserialized AuditRecord.
    """
    # Build a clean dict for AuditRecord.from_dict, excluding wiki-specific keys
    record_keys = {
        "correlation_id",
        "run_id",
        "pipeline_stage",
        "nl_input",
        "parsed_command",
        "confirmation",
        "ssh_execution",
        "structured_result",
        "created_at",
        "completed_at",
    }
    record_data = {k: v for k, v in fm.items() if k in record_keys}
    return AuditRecord.from_dict(record_data)


# ---------------------------------------------------------------------------
# Markdown body generation
# ---------------------------------------------------------------------------


def _format_decision(decision: str) -> str:
    """Format a confirmation decision for display."""
    labels = {
        "approved": "Approved",
        "denied": "Denied",
        "edited": "Edited",
    }
    return labels.get(decision, decision.title())


def _build_nl_section(nl: NLInputRecord) -> list[str]:
    """Build the NL input section lines."""
    return [
        "## Natural Language Input",
        "",
        f"> {nl.raw_input}",
        "",
        f"- **Source:** {nl.source}",
        f"- **Timestamp:** {_datetime_to_iso(nl.timestamp)}",
        "",
    ]


def _build_parsed_command_section(pc: ParsedCommandRecord) -> list[str]:
    """Build the parsed command section lines."""
    lines = [
        "## Parsed Command",
        "",
        f"- **Shell Command:** `{pc.resolved_shell}`",
        f"- **Risk Level:** {pc.risk_level}",
        f"- **Model:** {pc.model_id}",
        f"- **Explanation:** {pc.explanation}",
    ]
    if pc.affected_paths:
        lines.append(
            f"- **Affected Paths:** {', '.join(pc.affected_paths)}"
        )
    lines.append("")
    return lines


def _build_confirmation_section(conf: ConfirmationRecord) -> list[str]:
    """Build the confirmation section lines."""
    decision_display = _format_decision(conf.decision.value)
    lines = [
        "## Confirmation",
        "",
        f"- **Decision:** {decision_display}",
        f"- **Decided By:** {conf.decided_by}",
        f"- **Original Command:** `{conf.original_command}`",
    ]
    if conf.final_command and conf.final_command != conf.original_command:
        lines.append(f"- **Final Command:** `{conf.final_command}`")
    lines.extend([
        f"- **Timestamp:** {_datetime_to_iso(conf.timestamp)}",
        "",
    ])
    return lines


def _build_ssh_section(ssh: SSHExecutionRecord) -> list[str]:
    """Build the SSH execution section lines."""
    lines = [
        "## SSH Execution",
        "",
        f"- **Host:** {ssh.host}",
        f"- **User:** {ssh.user}",
        f"- **Port:** {ssh.port}",
        f"- **Command:** `{ssh.command}`",
        f"- **Session ID:** {ssh.session_id}",
        f"- **Started At:** {_datetime_to_iso(ssh.started_at)}",
    ]
    if ssh.remote_pid is not None:
        lines.append(f"- **Remote PID:** {ssh.remote_pid}")
    if ssh.completed_at is not None:
        lines.append(
            f"- **Completed At:** {_datetime_to_iso(ssh.completed_at)}"
        )
    if ssh.exit_code is not None:
        lines.append(f"- **Exit Code:** {ssh.exit_code}")
    if ssh.duration_seconds is not None:
        lines.append(f"- **Duration:** {ssh.duration_seconds:.2f}s")
    lines.append("")
    return lines


def _build_result_section(sr: StructuredResultRecord) -> list[str]:
    """Build the structured result section lines."""
    success_label = "Yes" if sr.success else "No"
    lines = [
        "## Structured Result",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Tests Passed | {sr.tests_passed} |",
        f"| Tests Failed | {sr.tests_failed} |",
        f"| Tests Skipped | {sr.tests_skipped} |",
        f"| Tests Total | {sr.tests_total} |",
        f"| Exit Code | {sr.exit_code} |",
        f"| Success | {success_label} |",
        "",
    ]
    if sr.error_message:
        lines.extend(["### Error", "", "```", sr.error_message, "```", ""])
    if sr.summary:
        lines.extend(["### Summary", "", sr.summary, ""])
    return lines


def _build_chain_section(chain: AuditChain) -> list[str]:
    """Build the audit chain summary section lines."""
    lines = [
        "## Audit Chain",
        "",
        f"*{len(chain)} stage entries recorded*",
        "",
        "| Stage | Status | Duration |",
        "|-------|--------|----------|",
    ]
    for entry in chain.entries:
        duration_str = (
            f"{entry.duration:.3f}s"
            if entry.duration is not None
            else "N/A"
        )
        lines.append(
            f"| {entry.stage} | {entry.status} | {duration_str} |"
        )
    lines.append("")
    return lines


def _build_metadata_section(
    record: AuditRecord,
    stage_label: str,
) -> list[str]:
    """Build the metadata section lines."""
    lines = [
        "## Metadata",
        "",
        f"- **Correlation ID:** {record.correlation_id}",
        f"- **Run ID:** {record.run_id}",
        f"- **Pipeline Stage:** {stage_label}",
        f"- **Created At:** {_datetime_to_iso(record.created_at)}",
    ]
    if record.completed_at is not None:
        lines.append(
            f"- **Completed At:** {_datetime_to_iso(record.completed_at)}"
        )
    lines.append("")
    return lines


def _build_body(
    record: AuditRecord,
    chain: AuditChain | None = None,
) -> str:
    """Generate the human-readable markdown body for an audit wiki entry.

    Composes section-building helpers to produce full traceability
    from NL input to execution result. Sections are conditionally
    included based on how far the pipeline progressed.
    """
    stage_label = record.pipeline_stage.value.replace("_", " ").title()
    outcome_label = "Denied" if record.is_denied else (
        "Complete" if record.is_complete else "In Progress"
    )

    lines: list[str] = [
        f"# Audit Record: {record.correlation_id[:8]}",
        "",
        f"*Full-chain audit trail -- pipeline stage: {stage_label}, outcome: {outcome_label}*",
        "",
    ]

    lines.extend(_build_nl_section(record.nl_input))

    if record.parsed_command is not None:
        lines.extend(_build_parsed_command_section(record.parsed_command))

    if record.confirmation is not None:
        lines.extend(_build_confirmation_section(record.confirmation))

    if record.ssh_execution is not None:
        lines.extend(_build_ssh_section(record.ssh_execution))

    if record.structured_result is not None:
        lines.extend(_build_result_section(record.structured_result))

    if chain is not None and len(chain) > 0:
        lines.extend(_build_chain_section(chain))

    lines.extend(_build_metadata_section(record, stage_label))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_to_document(
    record: AuditRecord,
    chain: AuditChain | None = None,
) -> WikiDocument:
    """Convert an AuditRecord to a WikiDocument.

    Produces a Karpathy-style wiki document with YAML frontmatter
    containing the full structured audit record and a markdown body
    with human-readable pipeline traceability.

    This is the pure-conversion step. Use ``write_audit`` to persist
    the document to disk.

    Args:
        record: The audit record to convert.
        chain: Optional AuditChain with fine-grained stage entries
            to include in the frontmatter and body.

    Returns:
        WikiDocument with frontmatter and body.
    """
    return WikiDocument(
        frontmatter=_record_to_frontmatter(record, chain),
        body=_build_body(record, chain),
    )


def write_audit(
    wiki_root: Path,
    record: AuditRecord,
    chain: AuditChain | None = None,
) -> AuditWriteOutcome:
    """Write an AuditRecord as a wiki entry.

    Creates the file and parent directories if needed. Uses atomic
    write (tmp file + rename) to prevent partial files. Overwrites
    any existing file for the same correlation_id.

    Wiki file location:
        {wiki_root}/pages/daemon/audit/audit-{correlation_id}.md

    Args:
        wiki_root: Path to the wiki root directory.
        record: The audit record to persist.
        chain: Optional AuditChain with fine-grained stage entries.

    Returns:
        AuditWriteOutcome with the file path and metadata.
    """
    file_path = audit_file_path(wiki_root, record.correlation_id)
    _ensure_directory(file_path)

    doc = audit_to_document(record, chain)
    content = frontmatter.serialize(doc)

    written_at = _now_utc()

    # Atomic write: write to temp file then rename (os.replace is atomic on POSIX)
    tmp_path = file_path.with_suffix(".md.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(str(tmp_path), str(file_path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info(
        "Wrote audit record %s (stage=%s) to %s",
        record.correlation_id,
        record.pipeline_stage.value,
        file_path,
    )

    return AuditWriteOutcome(
        file_path=file_path,
        correlation_id=record.correlation_id,
        written_at=written_at,
    )


def read_audit(file_path: Path) -> AuditRecord | None:
    """Read an AuditRecord from a wiki entry.

    Args:
        file_path: Path to the audit wiki file.

    Returns:
        The deserialized AuditRecord, or None if the file does not exist.

    Raises:
        ValueError: If the file exists but contains invalid frontmatter.
        KeyError: If the file is missing required fields.
    """
    if not file_path.exists():
        return None

    raw = file_path.read_text(encoding="utf-8")
    try:
        doc = frontmatter.parse(raw)
        return _frontmatter_to_record(doc.frontmatter)
    except (ValueError, KeyError, TypeError) as exc:
        logger.error(
            "Failed to parse audit record from %s: %s",
            file_path,
            exc,
        )
        raise


def list_audit_files(wiki_root: Path) -> list[Path]:
    """List all audit wiki files in the audit directory.

    Returns only files matching the ``audit-*.md`` naming pattern,
    excluding README.md and archived files. Results are sorted
    alphabetically by filename.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        Sorted list of paths to audit wiki files.
    """
    audit_directory = _audit_dir(wiki_root)
    if not audit_directory.is_dir():
        return []

    files = [
        p
        for p in sorted(audit_directory.iterdir())
        if p.is_file()
        and p.name.startswith(_AUDIT_PREFIX)
        and p.name.endswith(_AUDIT_SUFFIX)
    ]
    return files
