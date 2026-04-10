"""Full-chain audit record builder for the run pipeline.

Provides convenience builders and a safe writer that assembles an
``AuditRecord`` progressively as a run moves through its lifecycle:

    NL input -> parsed command -> confirmation -> SSH exec -> result

The helpers in this module adapt the runtime values produced by the
daemon's ``_handle_run`` / ``_background_execute`` methods onto the
immutable record shapes defined in :mod:`jules_daemon.audit.models`.

Key constraints satisfied:

- **No model mutation**: the ``AuditRecord`` and its sub-records are
  frozen dataclasses and are created via their factory/``with_*``
  transitions only.
- **No credential leakage**: callers pass only the host/user/port and
  command strings -- passwords stay inside the credential resolver
  and are never referenced here.
- **Safe to fail**: :func:`safe_write_audit` swallows any exception
  raised by the wiki writer so that audit persistence can never
  block or crash the run pipeline.
- **Cross-platform**: no Unix-only syscalls; uses ``os.getenv`` with
  a sensible default for the approver identity.

The builder functions are pure: they take primitive values and return
immutable records. Persistence is handled by :func:`safe_write_audit`,
which wraps the synchronous wiki writer in a ``try/except`` and logs
on failure.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jules_daemon.audit.models import (
    AuditRecord,
    ConfirmationDecision,
    ConfirmationRecord,
    NLInputRecord,
    ParsedCommandRecord,
    SSHExecutionRecord,
    StructuredResultRecord,
)

__all__ = [
    "AUDIT_OUTPUT_LIMIT",
    "build_confirmation_record",
    "build_nl_input_record",
    "build_parsed_command_record",
    "build_ssh_execution_record",
    "build_structured_result_record",
    "create_initial_audit",
    "safe_write_audit",
    "safe_write_audit_async",
    "truncate_text",
]

logger = logging.getLogger(__name__)

# Maximum bytes of stdout/stderr (or any command output) that will be
# stored inside a wiki audit record. The audit file is a human-readable
# markdown document -- we cap output to keep it manageable and to
# prevent pathologically large commands from bloating the wiki.
AUDIT_OUTPUT_LIMIT: int = 50_000

# Default approver identity when no username is available from the
# environment. Kept as a constant so tests and logs can reference it.
_DEFAULT_APPROVER: str = "unknown"

# Default model identifier when the LLM is not configured and a command
# is dispatched verbatim (e.g. direct shell commands).
_DIRECT_MODEL_ID: str = "direct"

# Default risk classification for commands that have not been scored
# by the LLM. The audit model requires one of the canonical risk
# levels -- "low" is the most conservative default.
_DEFAULT_RISK_LEVEL: str = "low"


def _now_utc() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _resolve_approver() -> str:
    """Determine the approver identity for a confirmation record.

    Cross-platform: prefers ``USER`` (POSIX) then ``USERNAME`` (Windows)
    then falls back to :data:`_DEFAULT_APPROVER` if neither is set.
    ``ConfirmationRecord`` rejects empty strings, so this function
    guarantees a non-empty return value.
    """
    candidate = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    candidate = candidate.strip()
    if candidate:
        return candidate
    return _DEFAULT_APPROVER


def truncate_text(text: str | None, limit: int = AUDIT_OUTPUT_LIMIT) -> str:
    """Return *text* truncated to at most *limit* characters.

    ``None`` and empty strings are normalized to an empty string. If
    the text exceeds the limit, a clear truncation marker is appended
    so readers can distinguish a naturally short output from a clipped
    one. The limit is applied after the marker is added so the final
    string length remains bounded.

    Args:
        text: The content to truncate. May be ``None``.
        limit: Maximum number of characters to keep before truncation.

    Returns:
        A bounded string safe to embed in audit records.
    """
    if not text:
        return ""
    if len(text) <= limit:
        return text
    marker = f"\n... [truncated, original length {len(text)} chars]"
    return text[:limit] + marker


def build_nl_input_record(
    *,
    raw_input: str,
    source: str = "ipc",
    timestamp: datetime | None = None,
) -> NLInputRecord:
    """Build an :class:`NLInputRecord` for the start of the pipeline.

    The caller is responsible for passing a non-empty ``raw_input`` --
    the daemon's request validator already enforces this. If somehow
    an empty string reaches this function, the record's own validation
    will raise ``ValueError``.

    Args:
        raw_input: The raw natural-language (or direct command) text
            the user submitted.
        source: Identifier of the input channel. The IPC request
            handler uses ``"ipc"``.
        timestamp: Override for the record's timestamp. Defaults to
            the current UTC time when ``None``.

    Returns:
        A frozen :class:`NLInputRecord`.
    """
    return NLInputRecord(
        raw_input=raw_input,
        timestamp=timestamp or _now_utc(),
        source=source,
    )


def create_initial_audit(
    *,
    run_id: str,
    nl_input: NLInputRecord,
) -> AuditRecord:
    """Create a brand-new :class:`AuditRecord` at the NL_INPUT stage.

    Thin wrapper around :meth:`AuditRecord.create` kept here so the
    pipeline code only needs to import a single helper module.
    """
    return AuditRecord.create(run_id=run_id, nl_input=nl_input)


def build_parsed_command_record(
    *,
    natural_language: str,
    resolved_shell: str,
    is_direct_command: bool,
    model_id: str | None,
    timestamp: datetime | None = None,
    risk_level: str = _DEFAULT_RISK_LEVEL,
    explanation: str = "",
    affected_paths: tuple[str, ...] = (),
) -> ParsedCommandRecord:
    """Build a :class:`ParsedCommandRecord` after command translation.

    Normalizes the many input shapes the handler may see:

    - For direct commands (``is_direct_command=True``) we record
      ``model_id=_DIRECT_MODEL_ID`` because :class:`ParsedCommandRecord`
      requires a non-empty model identifier.
    - ``explanation`` defaults to a short human-readable note when
      callers do not have richer LLM output available.
    - ``affected_paths`` is coerced to a tuple so the frozen dataclass
      invariant holds.

    Args:
        natural_language: The raw user input (before translation).
        resolved_shell: The final shell command that will be proposed
            to the user.
        is_direct_command: ``True`` if the user typed a verbatim
            command that bypasses the LLM.
        model_id: The LLM model identifier that produced the
            translation. ``None`` is converted to a fallback default.
        timestamp: Override for the record's timestamp.
        risk_level: Risk classification. Must be one of the canonical
            levels (``low``, ``medium``, ``high``, ``critical``).
        explanation: Optional human-readable note about the command.
        affected_paths: Optional tuple of paths the command touches.

    Returns:
        A frozen :class:`ParsedCommandRecord`.
    """
    if is_direct_command:
        resolved_model = _DIRECT_MODEL_ID
        default_explanation = "Direct shell command; LLM translation bypassed."
    else:
        resolved_model = (model_id or "llm").strip() or "llm"
        default_explanation = "Translated from natural language via LLM."

    return ParsedCommandRecord(
        natural_language=natural_language,
        resolved_shell=resolved_shell,
        model_id=resolved_model,
        risk_level=risk_level,
        explanation=explanation or default_explanation,
        affected_paths=tuple(affected_paths),
        timestamp=timestamp or _now_utc(),
    )


def build_confirmation_record(
    *,
    original_command: str,
    final_command: str,
    approved: bool,
    edited: bool,
    decided_by: str | None = None,
    timestamp: datetime | None = None,
) -> ConfirmationRecord:
    """Build a :class:`ConfirmationRecord` after the user decides.

    Args:
        original_command: The command that was presented to the user.
        final_command: The command the user approved (equals
            *original_command* for plain approvals, differs for
            edits, and is an empty string for denials -- the model
            accepts empty ``final_command`` as long as
            ``original_command`` is non-empty).
        approved: ``True`` if the user approved the run.
        edited: ``True`` if the user modified the command before
            approving. Ignored when ``approved=False``.
        decided_by: Override for the approver identity. Defaults to
            :func:`_resolve_approver` (the local user) when ``None``.
        timestamp: Override for the decision timestamp.

    Returns:
        A frozen :class:`ConfirmationRecord`.
    """
    if not approved:
        decision = ConfirmationDecision.DENIED
    elif edited:
        decision = ConfirmationDecision.EDITED
    else:
        decision = ConfirmationDecision.APPROVED

    approver = (decided_by or _resolve_approver()).strip()
    if not approver:
        approver = _DEFAULT_APPROVER

    return ConfirmationRecord(
        decision=decision,
        original_command=original_command,
        final_command=final_command,
        decided_by=approver,
        timestamp=timestamp or _now_utc(),
    )


def build_ssh_execution_record(
    *,
    host: str,
    user: str,
    port: int,
    command: str,
    session_id: str,
    started_at: datetime,
    completed_at: datetime,
    exit_code: int | None,
    duration_seconds: float | None,
) -> SSHExecutionRecord:
    """Build an :class:`SSHExecutionRecord` after the SSH run completes.

    Duration is clamped to a non-negative value because the record's
    own validator rejects negatives -- clock skew or monotonic
    mismatches should never crash audit persistence.

    Args:
        host: Remote hostname or IP address.
        user: SSH username used to connect.
        port: SSH port number (1-65535).
        command: The final shell command that was dispatched.
        session_id: Stable identifier for this SSH session. The daemon
            reuses the ``run_id`` here so the audit record links back
            to the wiki history entry.
        started_at: When the SSH command started executing.
        completed_at: When the SSH command finished (or failed).
        exit_code: Remote exit code. ``None`` indicates the connection
            failed before a code could be captured.
        duration_seconds: Wall-clock duration. Negative values are
            clamped to ``0.0``.

    Returns:
        A frozen :class:`SSHExecutionRecord`.
    """
    safe_duration: float | None
    if duration_seconds is None:
        safe_duration = None
    else:
        safe_duration = duration_seconds if duration_seconds >= 0 else 0.0

    return SSHExecutionRecord(
        host=host,
        user=user,
        port=port,
        command=command,
        session_id=session_id,
        started_at=started_at,
        remote_pid=None,
        completed_at=completed_at,
        exit_code=exit_code,
        duration_seconds=safe_duration,
    )


def build_structured_result_record(
    *,
    success: bool,
    exit_code: int | None,
    summary: str,
    error_message: str | None = None,
    timestamp: datetime | None = None,
    tests_passed: int = 0,
    tests_failed: int = 0,
    tests_skipped: int = 0,
    tests_total: int = 0,
) -> StructuredResultRecord:
    """Build a :class:`StructuredResultRecord` for the final stage.

    The existing record model is biased toward test results, so this
    helper defaults the test counts to zero. Callers that run a test
    suite and parse its output may override the counts; callers that
    run arbitrary shell commands should leave them at zero and rely
    on ``summary`` / ``error_message`` for context.

    Args:
        success: ``True`` if the command completed with exit code 0.
        exit_code: The observed exit code. ``None`` (e.g. connection
            failure) is mapped to ``-1`` so the record's integer
            field stays valid.
        summary: Short human-readable summary of the outcome.
        error_message: Optional long-form error details. ``None`` is
            preserved so consumers can distinguish "no error" from
            "empty error".
        timestamp: Override for the record timestamp.
        tests_passed: Number of tests that passed (defaults to ``0``).
        tests_failed: Number of tests that failed (defaults to ``0``).
        tests_skipped: Number of tests that were skipped.
        tests_total: Total test count.

    Returns:
        A frozen :class:`StructuredResultRecord`.
    """
    safe_exit_code = exit_code if exit_code is not None else -1

    return StructuredResultRecord(
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_skipped=tests_skipped,
        tests_total=tests_total,
        exit_code=safe_exit_code,
        success=success,
        error_message=error_message if error_message else None,
        summary=summary,
        timestamp=timestamp or _now_utc(),
    )


# ---------------------------------------------------------------------------
# Safe persistence wrappers
# ---------------------------------------------------------------------------


def safe_write_audit(
    *,
    wiki_root: Path,
    record: AuditRecord,
) -> Any:
    """Persist an :class:`AuditRecord` without crashing the caller.

    Wraps :func:`jules_daemon.wiki.audit_writer.write_audit` in a broad
    ``try/except`` so that disk full, permission errors, or malformed
    data never block the SSH run pipeline. Failures are logged at
    warning level (never raised) and ``None`` is returned.

    The audit writer is imported lazily so that the audit module has
    no hard dependency on the wiki writer at import time -- this keeps
    the audit layer independently testable.

    Args:
        wiki_root: Path to the wiki root directory.
        record: The fully- or partially-populated audit record.

    Returns:
        The :class:`AuditWriteOutcome` on success, or ``None`` on
        failure.
    """
    try:
        from jules_daemon.wiki.audit_writer import write_audit

        return write_audit(wiki_root, record)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to write audit record %s: %s",
            record.correlation_id,
            exc,
        )
        return None


async def safe_write_audit_async(
    *,
    wiki_root: Path,
    record: AuditRecord,
) -> Any:
    """Asynchronous wrapper around :func:`safe_write_audit`.

    Offloads the synchronous wiki write to a thread so that the
    asyncio event loop is never blocked by disk I/O. Like the sync
    variant, all exceptions are caught and logged.

    Args:
        wiki_root: Path to the wiki root directory.
        record: The fully- or partially-populated audit record.

    Returns:
        The :class:`AuditWriteOutcome` on success, or ``None`` on
        failure.
    """
    try:
        return await asyncio.to_thread(
            safe_write_audit, wiki_root=wiki_root, record=record,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to write audit record %s (async): %s",
            record.correlation_id,
            exc,
        )
        return None
