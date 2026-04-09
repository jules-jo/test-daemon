"""SSH reconnection with exponential backoff.

Orchestrates re-establishing an SSH session using stored SSHTarget
connection parameters, applying configurable exponential backoff
between retry attempts. Classifies errors as transient (retriable)
or permanent (fail-fast) to avoid wasting time on unrecoverable
failures.

The module is library-agnostic: actual SSH connection establishment
is delegated to an SSHConnector protocol implementation. This allows
testing with fakes and swapping SSH backends (paramiko, asyncssh, etc.).

Usage:
    connector = ParamikoConnector()
    target = SSHTarget(host="prod.example.com", user="deploy")
    config = BackoffConfig(base_delay=1.0, max_retries=5)

    result = await reconnect_ssh(target, connector, config)
    if result.success:
        channel = result.handle
    else:
        log_failure(result.error, result.retry_history)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from jules_daemon.ssh.backoff import BackoffConfig, BackoffDelay, calculate_delay
from jules_daemon.ssh.errors import (
    SSHReconnectionExhaustedError,
    is_permanent,
    is_transient,
)
from jules_daemon.wiki.models import SSHTarget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SSHConnectionHandle(Protocol):
    """Protocol for an established SSH connection.

    The handle is an opaque reference that the SSHConnector implementation
    creates. The reconnection logic does not inspect its internals --
    it only passes it back to the connector for health checks and cleanup.
    """

    @property
    def session_id(self) -> str:
        """Unique identifier for this SSH session."""
        ...


@runtime_checkable
class SSHConnector(Protocol):
    """Protocol for establishing and managing SSH connections.

    Implementations wrap a specific SSH library (paramiko, asyncssh, etc.)
    and translate its connection/auth exceptions into the project's
    SSH error hierarchy (SSHConnectionError, SSHAuthenticationError, etc.).
    """

    async def connect(self, target: SSHTarget) -> SSHConnectionHandle:
        """Establish a new SSH connection to the target host.

        Args:
            target: Connection parameters (host, user, port, key_path).

        Returns:
            An opaque connection handle.

        Raises:
            SSHConnectionError: Transient network failure.
            SSHAuthenticationError: Invalid credentials.
            SSHHostKeyError: Host key mismatch.
        """
        ...

    async def close(self, handle: SSHConnectionHandle) -> None:
        """Close an established SSH connection.

        Must not raise if the connection is already closed.
        """
        ...

    async def is_alive(self, handle: SSHConnectionHandle) -> bool:
        """Check if the SSH connection is still responsive.

        Returns:
            True if the connection is alive and usable.
        """
        ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RetryRecord:
    """Immutable record of a single retry attempt.

    Attributes:
        attempt: Zero-indexed attempt number.
        error_type: Fully qualified name of the exception class.
        error_message: String representation of the error.
        delay_seconds: How long the backoff waited before this attempt.
        is_transient: Whether the error was classified as transient.
        timestamp: UTC datetime when this attempt completed.
    """

    attempt: int
    error_type: str
    error_message: str
    delay_seconds: float
    is_transient: bool
    timestamp: datetime = field(default_factory=_now_utc)


@dataclass(frozen=True)
class ReconnectionResult:
    """Immutable outcome of a reconnection attempt sequence.

    Attributes:
        success: True if a connection was established.
        handle: The SSH connection handle (None if failed).
        attempts: Total number of connection attempts made.
        total_duration_seconds: Wall-clock time from start to finish.
        retry_history: Ordered tuple of retry records (empty on first-try success).
        error: Final error message if all attempts failed (None on success).
        target: The SSHTarget that was used for reconnection.
    """

    success: bool
    handle: Any  # SSHConnectionHandle | None (Any to avoid Protocol in frozen)
    attempts: int
    total_duration_seconds: float
    retry_history: tuple[RetryRecord, ...]
    error: str | None
    target: SSHTarget


# ---------------------------------------------------------------------------
# Core reconnection logic
# ---------------------------------------------------------------------------


async def reconnect_ssh(
    target: SSHTarget,
    connector: SSHConnector,
    config: BackoffConfig = BackoffConfig(),
    *,
    on_retry: Any | None = None,
) -> ReconnectionResult:
    """Attempt to re-establish an SSH connection with exponential backoff.

    Makes up to (1 + config.max_retries) connection attempts: one initial
    attempt plus up to max_retries retries. Between retries, sleeps for the
    computed backoff delay.

    Error classification:
        - Transient errors (SSHConnectionError, OSError, TimeoutError, etc.)
          trigger a retry with backoff.
        - Permanent errors (SSHAuthenticationError, SSHHostKeyError) cause
          an immediate failure without further retries.

    Args:
        target: SSH connection parameters from the wiki.
        connector: Implementation of the SSHConnector protocol.
        config: Exponential backoff configuration.
        on_retry: Optional async callback(retry_record: RetryRecord) called
            after each failed attempt. Useful for logging or wiki updates.

    Returns:
        ReconnectionResult with the outcome.

    Raises:
        SSHReconnectionExhaustedError: If raise_on_exhaustion is desired,
            callers can check result.success and raise manually.
    """
    retry_history: list[RetryRecord] = []
    start_time = _now_utc()
    max_attempts = 1 + config.max_retries

    for attempt in range(max_attempts):
        try:
            logger.info(
                "SSH connection attempt %d/%d to %s@%s:%d",
                attempt + 1,
                max_attempts,
                target.user,
                target.host,
                target.port,
            )
            handle = await connector.connect(target)

            # Success
            elapsed = (_now_utc() - start_time).total_seconds()
            logger.info(
                "SSH connection established to %s@%s:%d after %d attempt(s) "
                "(%.2fs elapsed)",
                target.user,
                target.host,
                target.port,
                attempt + 1,
                elapsed,
            )
            return ReconnectionResult(
                success=True,
                handle=handle,
                attempts=attempt + 1,
                total_duration_seconds=elapsed,
                retry_history=tuple(retry_history),
                error=None,
                target=target,
            )

        except Exception as exc:
            # Compute delay for this attempt (0 for the initial attempt's
            # record, actual delay for subsequent retries)
            delay_seconds = 0.0
            if attempt < config.max_retries:
                backoff_delay: BackoffDelay = calculate_delay(config, attempt)
                delay_seconds = backoff_delay.total

            error_is_transient = is_transient(exc)
            record = RetryRecord(
                attempt=attempt,
                error_type=type(exc).__qualname__,
                error_message=str(exc),
                delay_seconds=delay_seconds,
                is_transient=error_is_transient,
            )
            retry_history.append(record)

            # Invoke optional callback
            if on_retry is not None:
                try:
                    await on_retry(record)
                except Exception:
                    logger.debug(
                        "on_retry callback raised (ignored)", exc_info=True
                    )

            # Permanent error: fail immediately
            if is_permanent(exc):
                elapsed = (_now_utc() - start_time).total_seconds()
                logger.warning(
                    "SSH connection to %s@%s:%d failed with permanent error: %s",
                    target.user,
                    target.host,
                    target.port,
                    exc,
                )
                return ReconnectionResult(
                    success=False,
                    handle=None,
                    attempts=attempt + 1,
                    total_duration_seconds=elapsed,
                    retry_history=tuple(retry_history),
                    error=f"Permanent error: {exc}",
                    target=target,
                )

            # Transient error: retry if attempts remain
            if attempt < config.max_retries:
                logger.info(
                    "SSH attempt %d/%d failed (transient): %s. "
                    "Retrying in %.2fs...",
                    attempt + 1,
                    max_attempts,
                    exc,
                    delay_seconds,
                )
                await asyncio.sleep(delay_seconds)
            else:
                # Last attempt exhausted
                elapsed = (_now_utc() - start_time).total_seconds()
                logger.warning(
                    "SSH reconnection to %s@%s:%d exhausted after %d attempts "
                    "(%.2fs elapsed). Last error: %s",
                    target.user,
                    target.host,
                    target.port,
                    attempt + 1,
                    elapsed,
                    exc,
                )
                return ReconnectionResult(
                    success=False,
                    handle=None,
                    attempts=attempt + 1,
                    total_duration_seconds=elapsed,
                    retry_history=tuple(retry_history),
                    error=f"Reconnection exhausted after {attempt + 1} "
                    f"attempts. Last error: {exc}",
                    target=target,
                )

    # Should be unreachable, but satisfy the type checker
    elapsed = (_now_utc() - start_time).total_seconds()
    return ReconnectionResult(
        success=False,
        handle=None,
        attempts=max_attempts,
        total_duration_seconds=elapsed,
        retry_history=tuple(retry_history),
        error="Unexpected: reconnection loop exited without result",
        target=target,
    )


def raise_on_failure(result: ReconnectionResult) -> None:
    """Raise SSHReconnectionExhaustedError if the result indicates failure.

    Convenience function for callers who prefer exception-based flow
    over checking result.success.

    Args:
        result: The reconnection result to check.

    Raises:
        SSHReconnectionExhaustedError: If result.success is False.
    """
    if not result.success:
        raise SSHReconnectionExhaustedError(
            message=result.error or "Reconnection failed",
            attempts=result.attempts,
            last_error=result.retry_history[-1].error_message
            if result.retry_history
            else None,
        )
