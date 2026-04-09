"""SSH command executor with 3-retry exponential backoff.

Provides the daemon's default SSH connection policy: 1 initial attempt
plus exactly 3 retries with exponential backoff on transient failures.
Permanent errors (authentication, host key) fail immediately without
consuming retries.

This module wraps the lower-level ``reconnect_ssh()`` orchestrator with
a fixed 3-retry policy and provides structured ``ExecutionOutcome``
results for audit logging.

Backoff schedule (default, no jitter):
    Retry 1: ~1s   (base_delay * 2^0)
    Retry 2: ~2s   (base_delay * 2^1)
    Retry 3: ~4s   (base_delay * 2^2)
    Total max wait: ~7s before final attempt

Usage:
    connector = ParamikoConnector()
    target = SSHTarget(host="prod.example.com", user="deploy")

    outcome = await execute_ssh_command(target, connector)
    if outcome.success:
        channel = outcome.handle
    else:
        log(outcome.error, outcome.attempts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from jules_daemon.ssh.backoff import BackoffConfig
from jules_daemon.ssh.reconnect import (
    ReconnectionResult,
    RetryRecord,
    reconnect_ssh,
)
from jules_daemon.wiki.models import SSHTarget

__all__ = [
    "DEFAULT_SSH_BACKOFF",
    "ExecutionAttempt",
    "ExecutionOutcome",
    "create_ssh_backoff",
    "execute_ssh_command",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 3
"""Fixed retry count for SSH connections. Not configurable -- this is a
project-wide safety constraint ensuring bounded reconnection attempts."""


# ---------------------------------------------------------------------------
# Factory and default config
# ---------------------------------------------------------------------------


def create_ssh_backoff(
    *,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    multiplier: float = 2.0,
    jitter_factor: float = 0.1,
) -> BackoffConfig:
    """Create a BackoffConfig locked to exactly 3 retries.

    All backoff parameters are configurable except ``max_retries``,
    which is always fixed at 3 per the project requirement.

    Args:
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Upper bound on any single delay (seconds).
        multiplier: Exponential growth factor per retry.
        jitter_factor: Random jitter fraction (0.0 = none, 1.0 = full).

    Returns:
        A BackoffConfig with max_retries=3.
    """
    return BackoffConfig(
        base_delay=base_delay,
        max_delay=max_delay,
        multiplier=multiplier,
        jitter_factor=jitter_factor,
        max_retries=_MAX_RETRIES,
    )


DEFAULT_SSH_BACKOFF: BackoffConfig = create_ssh_backoff()
"""Project-wide default SSH backoff config: 3 retries, 1s base, 2x growth."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ExecutionAttempt:
    """Immutable record of a single SSH connection attempt.

    Attributes:
        attempt_number: One-indexed attempt number (1 = first try).
        success: Whether this attempt succeeded.
        error_type: Fully qualified exception class name (None on success).
        error_message: String representation of the error (empty on success).
        delay_before_seconds: Backoff delay applied before this attempt
            (0.0 for the initial attempt).
        is_transient: Whether the error was classified as transient
            (None on success).
        timestamp: UTC datetime when this attempt completed.
    """

    attempt_number: int
    success: bool
    error_type: str | None = None
    error_message: str = ""
    delay_before_seconds: float = 0.0
    is_transient: bool | None = None
    timestamp: datetime = field(default_factory=_now_utc)


@dataclass(frozen=True)
class ExecutionOutcome:
    """Immutable result of an SSH command execution attempt sequence.

    Attributes:
        success: True if connection was established.
        handle: The SSH connection handle (None if all attempts failed).
        total_attempts: Number of connection attempts made.
        total_duration_seconds: Wall-clock time from start to finish.
        attempts: Ordered tuple of ExecutionAttempt records.
        error: Final error message (None on success).
        target: The SSHTarget used.
        config: The BackoffConfig used (always has max_retries=3).
    """

    success: bool
    handle: Any  # SSHConnectionHandle | None
    total_attempts: int
    total_duration_seconds: float
    attempts: tuple[ExecutionAttempt, ...]
    error: str | None
    target: SSHTarget
    config: BackoffConfig


# ---------------------------------------------------------------------------
# Internal: convert reconnection result to execution outcome
# ---------------------------------------------------------------------------


def _retry_record_to_attempt(record: RetryRecord) -> ExecutionAttempt:
    """Convert a RetryRecord (failed attempt) to an ExecutionAttempt."""
    return ExecutionAttempt(
        attempt_number=record.attempt + 1,  # Convert 0-indexed to 1-indexed
        success=False,
        error_type=record.error_type,
        error_message=record.error_message,
        delay_before_seconds=record.delay_seconds,
        is_transient=record.is_transient,
        timestamp=record.timestamp,
    )


def _build_outcome(
    result: ReconnectionResult,
    config: BackoffConfig,
) -> ExecutionOutcome:
    """Build an ExecutionOutcome from a ReconnectionResult.

    Converts retry history to ExecutionAttempt records and appends
    a success record if the connection was established.
    """
    attempts: list[ExecutionAttempt] = [
        _retry_record_to_attempt(record)
        for record in result.retry_history
    ]

    # Append successful attempt if connection was established
    if result.success:
        attempts.append(
            ExecutionAttempt(
                attempt_number=result.attempts,
                success=True,
                delay_before_seconds=0.0,
            )
        )

    return ExecutionOutcome(
        success=result.success,
        handle=result.handle,
        total_attempts=result.attempts,
        total_duration_seconds=result.total_duration_seconds,
        attempts=tuple(attempts),
        error=result.error,
        target=result.target,
        config=config,
    )


# ---------------------------------------------------------------------------
# Public executor
# ---------------------------------------------------------------------------


async def execute_ssh_command(
    target: SSHTarget,
    connector: Any,
    config: BackoffConfig | None = None,
    *,
    on_attempt: Callable[[ExecutionAttempt], Awaitable[None]] | None = None,
) -> ExecutionOutcome:
    """Execute an SSH connection with 3-retry exponential backoff.

    Makes up to 4 connection attempts (1 initial + 3 retries). Transient
    errors (network timeout, connection refused, etc.) trigger retries
    with exponential backoff. Permanent errors (auth failure, host key
    mismatch) fail immediately.

    Args:
        target: SSH connection parameters.
        connector: SSHConnector implementation for the SSH backend.
        config: Optional backoff config. Defaults to DEFAULT_SSH_BACKOFF
            (3 retries, 1s base, 2x multiplier). The max_retries field
            is always overridden to 3 for safety.
        on_attempt: Optional async callback invoked after each attempt
            (including the successful one). Errors in the callback are
            logged and swallowed.

    Returns:
        ExecutionOutcome with structured result and attempt history.
    """
    effective_config = config if config is not None else DEFAULT_SSH_BACKOFF

    # Safety: enforce 3 retries regardless of what config was passed.
    # This is the project-wide constraint from AC 5.
    if effective_config.max_retries != _MAX_RETRIES:
        logger.warning(
            "SSH backoff config had max_retries=%d, overriding to %d",
            effective_config.max_retries,
            _MAX_RETRIES,
        )
        effective_config = BackoffConfig(
            base_delay=effective_config.base_delay,
            max_delay=effective_config.max_delay,
            multiplier=effective_config.multiplier,
            jitter_factor=effective_config.jitter_factor,
            max_retries=_MAX_RETRIES,
        )

    # Wrap on_attempt callback into an on_retry callback for reconnect_ssh
    on_retry_callback = None
    if on_attempt is not None:
        async def _on_retry_adapter(record: RetryRecord) -> None:
            attempt = _retry_record_to_attempt(record)
            try:
                await on_attempt(attempt)  # type: ignore[misc]
            except Exception:
                logger.debug(
                    "on_attempt callback raised (ignored)", exc_info=True
                )

        on_retry_callback = _on_retry_adapter

    # Execute with reconnection logic
    result = await reconnect_ssh(
        target=target,
        connector=connector,
        config=effective_config,
        on_retry=on_retry_callback,
    )

    # Build structured outcome
    outcome = _build_outcome(result, effective_config)

    # Fire callback for the final attempt (success or last failure)
    if on_attempt is not None and result.success:
        final_attempt = ExecutionAttempt(
            attempt_number=result.attempts,
            success=True,
            delay_before_seconds=0.0,
        )
        try:
            await on_attempt(final_attempt)
        except Exception:
            logger.debug(
                "on_attempt callback raised on success (ignored)",
                exc_info=True,
            )

    logger.info(
        "SSH execution to %s@%s:%d completed: success=%s, attempts=%d, "
        "duration=%.2fs",
        target.user,
        target.host,
        target.port,
        outcome.success,
        outcome.total_attempts,
        outcome.total_duration_seconds,
    )

    return outcome
