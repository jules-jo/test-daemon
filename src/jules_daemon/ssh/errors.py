"""SSH error hierarchy for connection and reconnection failures.

All errors inherit from SSHError, enabling callers to catch the base
class for any SSH-related failure or catch specific subclasses for
targeted handling.

Classification:
    - SSHConnectionError: Transient network failures (retriable)
    - SSHAuthenticationError: Permanent auth failures (not retriable)
    - SSHHostKeyError: Permanent host key mismatch (not retriable)
    - SSHReconnectionExhaustedError: All retry attempts consumed
"""

from __future__ import annotations


class SSHError(Exception):
    """Base error for all SSH operations."""


class SSHConnectionError(SSHError):
    """Transient connection failure (network timeout, refused, reset).

    These errors are retriable -- the reconnection logic should attempt
    again with exponential backoff.
    """


class SSHAuthenticationError(SSHError):
    """Permanent authentication failure (invalid key, wrong password).

    These errors are NOT retriable -- retrying with the same credentials
    will produce the same result.
    """


class SSHHostKeyError(SSHError):
    """Permanent host key verification failure.

    The remote host presented an unexpected key. This requires human
    intervention and is NOT retriable.
    """


class SSHReconnectionExhaustedError(SSHError):
    """All reconnection retry attempts have been exhausted.

    Contains the history of failures for diagnosis.
    """

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        last_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

# Exception types that indicate transient, retriable failures.
_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    SSHConnectionError,
    OSError,
    ConnectionError,
    ConnectionRefusedError,
    ConnectionResetError,
    ConnectionAbortedError,
    TimeoutError,
    BrokenPipeError,
    EOFError,
)

# Exception types that indicate permanent, non-retriable failures.
_PERMANENT_TYPES: tuple[type[BaseException], ...] = (
    SSHAuthenticationError,
    SSHHostKeyError,
    PermissionError,
)


def is_transient(error: BaseException) -> bool:
    """Return True if the error represents a transient, retriable failure.

    Transient errors include network timeouts, connection resets, refused
    connections, and similar I/O problems that may resolve on retry.

    Args:
        error: The exception to classify.

    Returns:
        True if the error is retriable.
    """
    return isinstance(error, _TRANSIENT_TYPES)


def is_permanent(error: BaseException) -> bool:
    """Return True if the error represents a permanent, non-retriable failure.

    Permanent errors include authentication failures and host key
    mismatches that require human intervention.

    Args:
        error: The exception to classify.

    Returns:
        True if the error should not be retried.
    """
    return isinstance(error, _PERMANENT_TYPES)
