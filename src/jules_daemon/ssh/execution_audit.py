"""SSH execution audit instrumentation -- records audit entries for dispatch.

Bridges the SSH dispatch stage with the audit pipeline by constructing
``SSHExecutionRecord`` instances from dispatch results and advancing
``AuditRecord`` to the ``SSH_DISPATCHED`` pipeline stage. Also appends
a stage entry to the generic ``AuditChain`` for chain-based audit trails.

This module is the single integration point where SSH execution outcomes
(command, host, success/failure, PID, session_id) are captured as
structured audit data. It does not perform SSH execution itself -- it
wraps the result of ``dispatch_recovery_command`` with audit recording.

Design principles:
    - Immutable outputs: ``AuditedDispatchResult`` is a frozen dataclass
    - No side effects: builds data structures only, no I/O or wiki writes
    - Composable: accepts and returns audit chain for pipeline threading
    - Defensive: never raises on valid inputs

Usage::

    from jules_daemon.ssh.execution_audit import record_ssh_execution_audit

    audited = record_ssh_execution_audit(
        audit_record=record,
        target=ssh_target,
        dispatch_result=dispatch_result,
        audit_chain=chain,
    )
    # audited.audit_record is at SSH_DISPATCHED stage
    # audited.audit_chain has an "ssh_execution" entry appended
    # audited.ssh_execution_record has host, command, outcome
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from jules_daemon.audit.models import (
    AuditRecord,
    SSHExecutionRecord,
)
from jules_daemon.audit_models import AuditChain, AuditEntry
from jules_daemon.ssh.dispatch import DispatchResult
from jules_daemon.wiki.models import SSHTarget

__all__ = [
    "AuditedDispatchResult",
    "build_ssh_execution_record",
    "record_ssh_execution_audit",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditedDispatchResult:
    """Immutable result of an audited SSH dispatch operation.

    Bundles the dispatch outcome with the updated audit record and
    chain so callers can thread both through the pipeline.

    Attributes:
        dispatch_result: The original DispatchResult from SSH dispatch.
        audit_record: AuditRecord advanced to SSH_DISPATCHED stage.
        audit_chain: AuditChain with "ssh_execution" entry appended.
        ssh_execution_record: The SSHExecutionRecord that was created.
    """

    dispatch_result: DispatchResult
    audit_record: AuditRecord
    audit_chain: AuditChain
    ssh_execution_record: SSHExecutionRecord


# ---------------------------------------------------------------------------
# SSHExecutionRecord builder
# ---------------------------------------------------------------------------


def build_ssh_execution_record(
    *,
    target: SSHTarget,
    dispatch_result: DispatchResult,
    started_at: datetime | None = None,
) -> SSHExecutionRecord:
    """Build an SSHExecutionRecord from a dispatch result and SSH target.

    Captures the command, host, user, port, session ID, and remote PID
    from the dispatch outcome. The exit_code and duration are not known
    at dispatch time (the command is still running), so they are set to
    None.

    Args:
        target: SSH connection target with host, user, port.
        dispatch_result: Result from ``dispatch_recovery_command``.
        started_at: Override the start timestamp (for testing).
            Defaults to the dispatch result's timestamp.

    Returns:
        A frozen SSHExecutionRecord capturing the dispatch details.
    """
    effective_started_at = (
        started_at
        if started_at is not None
        else dispatch_result.timestamp
    )

    return SSHExecutionRecord(
        host=target.host,
        user=target.user,
        port=target.port,
        command=dispatch_result.command_string,
        session_id=dispatch_result.session_id,
        started_at=effective_started_at,
        remote_pid=dispatch_result.remote_pid,
        completed_at=None,
        exit_code=None,
        duration_seconds=None,
    )


# ---------------------------------------------------------------------------
# Audit chain entry builder
# ---------------------------------------------------------------------------


def _build_audit_entry(
    *,
    ssh_record: SSHExecutionRecord,
    dispatch_result: DispatchResult,
) -> AuditEntry:
    """Build an AuditEntry for the ssh_execution stage.

    Captures the dispatch inputs (command, host, session) as the
    before_snapshot and the outcome (success, PID, error) as the
    after_snapshot. Uses the ``audit_models.AuditEntry`` format with
    before/after snapshots, duration, status, and error fields.

    Args:
        ssh_record: The SSHExecutionRecord for this dispatch.
        dispatch_result: The raw dispatch outcome.

    Returns:
        A frozen AuditEntry with stage "ssh_execution".
    """
    before_snapshot: dict[str, Any] = {
        "host": ssh_record.host,
        "user": ssh_record.user,
        "port": ssh_record.port,
        "command": ssh_record.command,
        "session_id": ssh_record.session_id,
    }

    after_snapshot: dict[str, Any] = {
        "success": dispatch_result.success,
        "remote_pid": dispatch_result.remote_pid,
        "action": dispatch_result.action.value,
        "wiki_updated": dispatch_result.wiki_updated,
    }
    if dispatch_result.error is not None:
        after_snapshot["error"] = dispatch_result.error

    status = "success" if dispatch_result.success else "error"
    error_msg = dispatch_result.error if not dispatch_result.success else None

    return AuditEntry(
        stage="ssh_execution",
        timestamp=ssh_record.started_at,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        duration=None,
        status=status,
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_ssh_execution_audit(
    *,
    audit_record: AuditRecord,
    target: SSHTarget,
    dispatch_result: DispatchResult,
    audit_chain: AuditChain | None = None,
    started_at: datetime | None = None,
) -> AuditedDispatchResult:
    """Record an audit entry for the SSH execution stage.

    Builds an ``SSHExecutionRecord`` from the dispatch result, advances
    the ``AuditRecord`` to ``SSH_DISPATCHED``, and appends an
    ``ssh_execution`` entry to the ``AuditChain``.

    This function is pure: it builds data structures only and performs
    no I/O. Wiki persistence of the audit record is the caller's
    responsibility.

    Args:
        audit_record: The current AuditRecord (at CONFIRMATION stage).
        target: SSH connection target with host, user, port.
        dispatch_result: Result from ``dispatch_recovery_command``.
        audit_chain: Optional existing chain to append to. If None,
            a fresh empty chain is created.
        started_at: Override the start timestamp (for testing).

    Returns:
        AuditedDispatchResult with the updated audit record, chain,
        SSH execution record, and original dispatch result.
    """
    effective_chain = (
        audit_chain if audit_chain is not None else AuditChain.empty()
    )

    # Build the SSH execution record
    ssh_record = build_ssh_execution_record(
        target=target,
        dispatch_result=dispatch_result,
        started_at=started_at,
    )

    # Advance the audit record to SSH_DISPATCHED
    updated_record = audit_record.with_ssh_execution(ssh_record)

    # Build and append audit chain entry
    chain_entry = _build_audit_entry(
        ssh_record=ssh_record,
        dispatch_result=dispatch_result,
    )
    updated_chain = effective_chain.append(chain_entry)

    logger.info(
        "SSH execution audit recorded: host=%s command=%s success=%s "
        "correlation_id=%s",
        ssh_record.host,
        ssh_record.command[:80],
        dispatch_result.success,
        updated_record.correlation_id,
    )

    return AuditedDispatchResult(
        dispatch_result=dispatch_result,
        audit_record=updated_record,
        audit_chain=updated_chain,
        ssh_execution_record=ssh_record,
    )
