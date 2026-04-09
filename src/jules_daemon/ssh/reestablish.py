"""SSH re-establishment from recovered run records.

Bridges the crash recovery detection layer with the SSH reconnection
layer. Reads host, port, and credentials from a CrashRecoveryResult
(the wiki-based recovered run record) and opens a new SSH connection
with exponential backoff retry and timeout handling.

This module is the primary entry point for re-establishing SSH sessions
after a daemon crash. It:

1. Validates that the recovery result is actionable (RECONNECT action)
2. Extracts and reconstructs an SSHTarget from the flattened fields
3. Delegates connection to the existing reconnect_ssh() orchestrator
4. Returns a structured ReestablishmentResult with connection handle,
   retry history, and recovery metadata

The module is library-agnostic: actual SSH connection is handled by the
SSHConnector protocol implementation passed by the caller.

Usage:
    from jules_daemon.wiki.crash_recovery import detect_crash_recovery
    from jules_daemon.ssh.reestablish import reestablish_ssh

    recovery = detect_crash_recovery(wiki_root)
    if recovery.needs_recovery and recovery.action == RecoveryAction.RECONNECT:
        result = await reestablish_ssh(recovery, connector)
        if result.success:
            channel = result.handle
        else:
            handle_failure(result.error, result.retry_history)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable

from jules_daemon.ssh.backoff import BackoffConfig
from jules_daemon.ssh.reconnect import (
    ReconnectionResult,
    RetryRecord,
    SSHConnectionHandle,
    SSHConnector,
    reconnect_ssh,
)
from jules_daemon.wiki.crash_recovery import CrashRecoveryResult, RecoveryAction
from jules_daemon.wiki.models import SSHTarget

__all__ = [
    "ReestablishmentResult",
    "extract_ssh_target",
    "reestablish_ssh",
]

logger = logging.getLogger(__name__)

_DEFAULT_SSH_PORT = 22


# ---------------------------------------------------------------------------
# Default backoff config for re-establishment
# ---------------------------------------------------------------------------

# Re-establishment uses a more conservative backoff than the generic
# reconnect_ssh() default (5 retries, 60s max). Crash recovery should
# fail faster so the daemon can mark the run as failed and accept new
# commands, rather than blocking for minutes on a likely-dead host.
_DEFAULT_REESTABLISH_CONFIG = BackoffConfig(
    base_delay=1.0,
    max_delay=30.0,
    multiplier=2.0,
    jitter_factor=0.1,
    max_retries=3,
)


# ---------------------------------------------------------------------------
# SSHTarget extraction
# ---------------------------------------------------------------------------


def extract_ssh_target(recovery: CrashRecoveryResult) -> SSHTarget:
    """Extract and validate an SSHTarget from a CrashRecoveryResult.

    Reconstructs an SSHTarget from the flattened connection fields in
    the recovery result. Validates that required fields (host, user)
    are present.

    Args:
        recovery: The crash recovery result containing connection params.

    Returns:
        A validated SSHTarget ready for connection.

    Raises:
        ValueError: If required connection parameters are missing.
    """
    if recovery.host is None:
        raise ValueError(
            "Cannot extract SSH target: host is missing from recovery record "
            f"(run_id={recovery.run_id})"
        )
    if recovery.user is None:
        raise ValueError(
            "Cannot extract SSH target: user is missing from recovery record "
            f"(run_id={recovery.run_id})"
        )

    port = recovery.port if recovery.port is not None else _DEFAULT_SSH_PORT

    return SSHTarget(
        host=recovery.host,
        user=recovery.user,
        port=port,
        key_path=recovery.key_path,
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ReestablishmentResult:
    """Immutable outcome of an SSH re-establishment attempt.

    Contains both the connection result and the recovery metadata
    from the original run record, giving the caller everything needed
    to resume monitoring the interrupted test.

    Attributes:
        success: True if the SSH connection was re-established.
        handle: The SSH connection handle (None if failed).
        attempts: Total number of connection attempts made.
        total_duration_seconds: Wall-clock time from start to finish.
        retry_history: Ordered tuple of retry records (empty on first-try
            success or pre-validation failure).
        error: Human-readable error description (None on success).
        target: The SSHTarget used for connection (None if extraction failed).
        run_id: The recovered run's unique identifier.
        remote_pid: PID of the remote test process (None if unknown).
        resolved_shell: The shell command that was running (None if absent).
        timestamp: UTC datetime when this result was produced.
    """

    success: bool
    handle: SSHConnectionHandle | None
    attempts: int
    total_duration_seconds: float
    retry_history: tuple[RetryRecord, ...]
    error: str | None
    target: SSHTarget | None
    run_id: str
    remote_pid: int | None
    resolved_shell: str | None
    timestamp: datetime = field(default_factory=_now_utc)


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _build_validation_failure(
    *,
    error: str,
    run_id: str,
    remote_pid: int | None,
    resolved_shell: str | None,
) -> ReestablishmentResult:
    """Build a failed result for pre-connection validation failures."""
    return ReestablishmentResult(
        success=False,
        handle=None,
        attempts=0,
        total_duration_seconds=0.0,
        retry_history=(),
        error=error,
        target=None,
        run_id=run_id,
        remote_pid=remote_pid,
        resolved_shell=resolved_shell,
    )


def _build_from_reconnection(
    *,
    reconnection: ReconnectionResult,
    run_id: str,
    remote_pid: int | None,
    resolved_shell: str | None,
) -> ReestablishmentResult:
    """Build a ReestablishmentResult from a ReconnectionResult."""
    return ReestablishmentResult(
        success=reconnection.success,
        handle=reconnection.handle,
        attempts=reconnection.attempts,
        total_duration_seconds=reconnection.total_duration_seconds,
        retry_history=reconnection.retry_history,
        error=reconnection.error,
        target=reconnection.target,
        run_id=run_id,
        remote_pid=remote_pid,
        resolved_shell=resolved_shell,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def reestablish_ssh(
    recovery: CrashRecoveryResult,
    connector: SSHConnector,
    config: BackoffConfig | None = None,
    *,
    on_progress: Callable[[str], Awaitable[None]] | None = None,
) -> ReestablishmentResult:
    """Re-establish an SSH connection from a recovered run record.

    Reads host, port, and credentials from the CrashRecoveryResult,
    validates that the recovery action is RECONNECT, reconstructs the
    SSHTarget, and attempts to open a new SSH connection with
    exponential backoff retry.

    Pre-connection validation:
        - Action must be RECONNECT (not FRESH_START or RESUME_APPROVAL)
        - Host and user must be present in the recovery record
        - Port defaults to 22 if not specified

    Connection behavior:
        - Transient errors (timeout, connection refused) trigger retries
        - Permanent errors (auth failure, host key mismatch) fail immediately
        - Configurable backoff via the config parameter

    Args:
        recovery: The CrashRecoveryResult from detect_crash_recovery().
        connector: An SSHConnector implementation for the SSH backend.
        config: Optional backoff config. Defaults to 3 retries with 1s
            base delay and 2x multiplier.
        on_progress: Optional async callback invoked with human-readable
            progress messages during re-establishment. Errors in the
            callback are logged and swallowed.

    Returns:
        ReestablishmentResult with connection handle and recovery metadata.
        Never raises -- all error conditions are captured in the result.
    """
    effective_config = config if config is not None else _DEFAULT_REESTABLISH_CONFIG

    run_id = recovery.run_id
    remote_pid = recovery.remote_pid
    resolved_shell = recovery.resolved_shell

    # Validation: action must be RECONNECT
    if recovery.action != RecoveryAction.RECONNECT:
        error_msg = (
            f"SSH re-establishment requires action=RECONNECT, "
            f"got action={recovery.action.value}. "
            f"Only interrupted RUNNING tests can be reconnected."
        )
        logger.warning(
            "Re-establishment rejected: %s (run_id=%s)",
            error_msg,
            run_id,
        )
        return _build_validation_failure(
            error=error_msg,
            run_id=run_id,
            remote_pid=remote_pid,
            resolved_shell=resolved_shell,
        )

    # Validation: extract SSH target (host + user required)
    try:
        target = extract_ssh_target(recovery)
    except ValueError as exc:
        error_msg = str(exc)
        logger.warning(
            "Re-establishment rejected: %s (run_id=%s)",
            error_msg,
            run_id,
        )
        return _build_validation_failure(
            error=error_msg,
            run_id=run_id,
            remote_pid=remote_pid,
            resolved_shell=resolved_shell,
        )

    # Progress notification: starting
    await _notify_progress(
        on_progress,
        f"Re-establishing SSH to {target.user}@{target.host}:{target.port} "
        f"for run {run_id}",
    )

    # Build on_retry adapter for progress notifications.
    # The closure explicitly captures the checked callback reference
    # to avoid future refactoring from inadvertently rebinding it.
    on_retry_callback = None
    if on_progress is not None:
        _progress_cb = on_progress  # captured explicitly

        async def _retry_adapter(record: RetryRecord) -> None:
            msg = (
                f"SSH attempt {record.attempt + 1} failed "
                f"({record.error_type}: {record.error_message}). "
                f"Retrying in {record.delay_seconds:.1f}s..."
            )
            await _notify_progress(_progress_cb, msg)

        on_retry_callback = _retry_adapter

    # Attempt reconnection
    logger.info(
        "Re-establishing SSH connection to %s@%s:%d for run %s "
        "(max_retries=%d, base_delay=%.1fs)",
        target.user,
        target.host,
        target.port,
        run_id,
        effective_config.max_retries,
        effective_config.base_delay,
    )

    reconnection = await reconnect_ssh(
        target=target,
        connector=connector,
        config=effective_config,
        on_retry=on_retry_callback,
    )

    # Build result
    result = _build_from_reconnection(
        reconnection=reconnection,
        run_id=run_id,
        remote_pid=remote_pid,
        resolved_shell=resolved_shell,
    )

    if result.success:
        logger.info(
            "SSH re-established to %s@%s:%d for run %s "
            "after %d attempt(s) (%.2fs elapsed)",
            target.user,
            target.host,
            target.port,
            run_id,
            result.attempts,
            result.total_duration_seconds,
        )
        await _notify_progress(
            on_progress,
            f"SSH connection re-established after {result.attempts} attempt(s)",
        )
    else:
        logger.warning(
            "SSH re-establishment failed for run %s: %s "
            "(%d attempts, %.2fs elapsed)",
            run_id,
            result.error,
            result.attempts,
            result.total_duration_seconds,
        )
        await _notify_progress(
            on_progress,
            f"SSH re-establishment failed: {result.error}",
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _notify_progress(
    callback: Callable[[str], Awaitable[None]] | None,
    message: str,
) -> None:
    """Invoke the progress callback, swallowing any errors.

    Args:
        callback: The async callback, or None.
        message: The progress message to send.
    """
    if callback is None:
        return
    try:
        await callback(message)
    except Exception:
        logger.debug(
            "on_progress callback raised (ignored)", exc_info=True
        )
