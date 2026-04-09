"""Audit-instrumented SSH command confirmation.

Wraps the interactive ``confirm_ssh_command()`` flow with audit
instrumentation so that every confirmation decision (approve, reject,
or edit-then-approve) produces a persisted audit entry in the wiki
filesystem.

The audit entry captures:
    - The original command and risk context (before-snapshot)
    - The user's decision, whether the command was edited, and the
      final command text (after-snapshot)
    - Wall-clock duration of the confirmation interaction
    - Success status

Each invocation writes exactly one audit markdown file to
``pages/daemon/audit/audit-{event_id}.md`` with YAML frontmatter
and a human-readable body.

Security invariant: This module only *records* the confirmation
decision. The actual execution gate is in ``confirm_ssh_command``;
this module does not bypass it.

Usage::

    from jules_daemon.cli.confirmation_audit import confirm_with_audit

    result = confirm_with_audit(request, wiki_root, terminal=io)
    result.confirmation  # ConfirmationResult
    result.entry         # AuditEntry
    result.chain         # AuditChain with the new entry
    result.audit_path    # Path to the written audit file
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jules_daemon.audit_models import AuditChain, AuditEntry
from jules_daemon.cli.confirmation import (
    ConfirmationRequest,
    ConfirmationResult,
    Decision,
    TerminalIO,
    confirm_ssh_command,
)
from jules_daemon.wiki.frontmatter import WikiDocument, serialize

__all__ = [
    "ConfirmationAuditResult",
    "confirm_with_audit",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAGE_NAME: str = "confirmation"
_AUDIT_FILE_PREFIX: str = "audit-"
_AUDIT_FILE_SUFFIX: str = ".md"
_AUDIT_DIR_RELATIVE: str = "pages/daemon/audit"
_AUDIT_TYPE: str = "audit-log"


# ---------------------------------------------------------------------------
# ConfirmationAuditResult -- immutable result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfirmationAuditResult:
    """Immutable result of an audit-instrumented confirmation.

    Bundles the confirmation result with the audit trail artifacts:
    the AuditEntry, the updated AuditChain, and the filesystem path
    to the written audit file.

    Attributes:
        confirmation: The confirmation result (decision + final command).
        entry: The AuditEntry recording this confirmation event.
        chain: The AuditChain with the new entry appended.
        audit_path: Path to the written wiki audit markdown file.
    """

    confirmation: ConfirmationResult
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


def _classify_decision(result: ConfirmationResult) -> str:
    """Classify the user's confirmation action for audit recording.

    Returns one of: "approve", "reject", "edited".

    The "edited" label is used when the user approved after editing
    the command. If the user edited but ultimately rejected, the
    decision is "reject" (the rejection takes precedence).
    """
    if result.decision == Decision.REJECT:
        return "reject"
    if result.was_edited:
        return "edited"
    return "approve"


def _build_before_snapshot(request: ConfirmationRequest) -> dict[str, Any]:
    """Build the before-snapshot dict from the confirmation request."""
    snapshot: dict[str, Any] = {
        "original_command": request.ssh_command.command,
        "risk_level": request.context.risk_level.value,
        "explanation": request.context.explanation,
    }
    if request.target is not None:
        snapshot["target_host"] = request.target.host
        snapshot["target_user"] = request.target.user
    if request.ssh_command.working_directory is not None:
        snapshot["working_directory"] = request.ssh_command.working_directory
    return snapshot


def _build_after_snapshot(
    result: ConfirmationResult,
    decision_label: str,
) -> dict[str, Any]:
    """Build the after-snapshot dict from the confirmation result."""
    return {
        "decision": decision_label,
        "final_command": result.final_command.command,
        "was_edited": result.was_edited,
    }


def _build_audit_entry(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    duration: float,
    timestamp: datetime,
) -> AuditEntry:
    """Build an immutable AuditEntry for the confirmation stage."""
    return AuditEntry(
        stage=_STAGE_NAME,
        timestamp=timestamp,
        before_snapshot=before,
        after_snapshot=after,
        duration=duration,
        status="success",
        error=None,
    )


def _build_frontmatter(
    *,
    event_id: str,
    decision_label: str,
    entry: AuditEntry,
    before: dict[str, Any],
) -> dict[str, Any]:
    """Build YAML frontmatter dict for the audit wiki file."""
    return {
        "type": _AUDIT_TYPE,
        "event_id": event_id,
        "stage": _STAGE_NAME,
        "status": entry.status,
        "decision": decision_label,
        "timestamp": entry.timestamp.isoformat(),
        "original_command": before["original_command"],
        "risk_level": before["risk_level"],
        "duration_seconds": entry.duration,
        "tags": ["audit", "confirmation", "daemon"],
    }


def _build_body(
    *,
    event_id: str,
    decision_label: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> str:
    """Build the markdown body for the audit wiki file."""
    lines = [
        "# Confirmation Audit",
        "",
        f"Event: `{event_id}`",
        "",
        "## Command",
        "",
        "```",
        before["original_command"],
        "```",
        "",
        f"- **Risk Level**: {before['risk_level']}",
        f"- **Explanation**: {before['explanation']}",
        "",
        "## Decision",
        "",
        f"- **Action**: {decision_label}",
        f"- **Was Edited**: {after['was_edited']}",
        f"- **Final Command**: `{after['final_command']}`",
        "",
    ]
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


def confirm_with_audit(
    request: ConfirmationRequest,
    wiki_root: Path,
    *,
    terminal: TerminalIO | None = None,
    chain: AuditChain | None = None,
) -> ConfirmationAuditResult:
    """Run the SSH command confirmation prompt with audit instrumentation.

    Wraps ``confirm_ssh_command()`` with before/after snapshot capture,
    timing, and wiki persistence. Produces one audit file per invocation
    at ``pages/daemon/audit/audit-{event_id}.md``.

    The audit entry records the user's action:
        - ``"approve"`` -- command accepted as-is
        - ``"reject"`` -- command rejected (or EOF/interrupt)
        - ``"edited"`` -- command modified then approved

    Args:
        request: The confirmation request with command and context.
        wiki_root: Path to the wiki root directory. The audit directory
            (``pages/daemon/audit/``) must exist.
        terminal: IO abstraction. Defaults to DefaultTerminalIO.
        chain: Optional existing AuditChain to append to. If None,
            a fresh empty chain is used.

    Returns:
        Immutable ConfirmationAuditResult with the confirmation result,
        audit entry, updated chain, and path to the written audit file.
    """
    if chain is None:
        chain = AuditChain.empty()

    event_id = _generate_event_id()
    timestamp = _now_utc()

    # Build before-snapshot from the request
    before = _build_before_snapshot(request)

    # Time the interactive confirmation
    start = time.monotonic()
    confirmation_result = confirm_ssh_command(request, terminal=terminal)
    duration = time.monotonic() - start

    # Classify the decision for audit labeling
    decision_label = _classify_decision(confirmation_result)

    # Build the after-snapshot from the result
    after = _build_after_snapshot(confirmation_result, decision_label)

    # Build the audit entry
    entry = _build_audit_entry(
        before=before,
        after=after,
        duration=duration,
        timestamp=timestamp,
    )

    # Append to chain (immutable)
    updated_chain = chain.append(entry)

    # Build and write wiki file
    fm = _build_frontmatter(
        event_id=event_id,
        decision_label=decision_label,
        entry=entry,
        before=before,
    )
    body = _build_body(
        event_id=event_id,
        decision_label=decision_label,
        before=before,
        after=after,
    )
    audit_path = _write_audit_file(
        wiki_root=wiki_root,
        event_id=event_id,
        frontmatter_dict=fm,
        body=body,
    )

    return ConfirmationAuditResult(
        confirmation=confirmation_result,
        entry=entry,
        chain=updated_chain,
        audit_path=audit_path,
    )
