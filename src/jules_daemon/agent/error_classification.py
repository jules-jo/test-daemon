"""Dedicated error classification for the agent loop.

Classifies errors from LLM calls, tool executions, SSH operations, and
user actions into two categories:

- **TRANSIENT**: Safe to retry. The underlying issue is likely temporary
  and a subsequent attempt may succeed. Examples: network timeouts, rate
  limits, connection resets, server 5xx errors.

- **PERMANENT**: Not retryable. Retrying with the same input will produce
  the same failure. Examples: authentication failures, user denial, user
  cancel, malformed LLM responses, validation errors.

This module is the single canonical classification point for the agent
loop. All retry/terminate decisions should flow through ``classify_error()``
rather than ad-hoc ``isinstance()`` checks scattered across modules.

Classification priority (first match wins):
    1. Agent-specific error types (UserDenialError, UserCancelError, etc.)
    2. LLM error hierarchy (LLMAuthenticationError, LLMConnectionError, etc.)
    3. SSH error hierarchy (SSHAuthenticationError, SSHConnectionError, etc.)
    4. OpenAI SDK error types (APIConnectionError, RateLimitError, etc.)
    5. HTTP status code on the exception object (429, 5xx = transient)
    6. Python built-in exception types (ConnectionError, TimeoutError, etc.)
    7. Default: PERMANENT (unknown errors are not safe to retry)

Usage::

    from jules_daemon.agent.error_classification import (
        classify_error,
        is_transient,
        is_permanent,
    )

    try:
        result = await llm_client.get_tool_calls(messages)
    except Exception as exc:
        classified = classify_error(exc)
        if classified.is_retryable:
            # retry the operation
            ...
        else:
            # terminate the loop
            ...
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ClassifiedError",
    "ErrorCategory",
    "ErrorKind",
    "RetryExhaustedError",
    "ToolNotFoundError",
    "ToolValidationError",
    "UserCancelError",
    "UserDenialError",
    "classify_error",
    "is_permanent",
    "is_transient",
]


# ---------------------------------------------------------------------------
# ErrorKind enum
# ---------------------------------------------------------------------------


class ErrorKind(Enum):
    """Binary classification of an error for retry decisions.

    TRANSIENT: The error is likely temporary. The agent loop should
        retry the operation (up to its retry budget). Network blips,
        timeouts, and rate limits fall here.

    PERMANENT: The error will not resolve on retry. The agent loop
        should terminate immediately or fall back to the one-shot path.
        Auth failures, user denial, and validation errors fall here.
    """

    TRANSIENT = "transient"
    PERMANENT = "permanent"


# ---------------------------------------------------------------------------
# ErrorCategory enum
# ---------------------------------------------------------------------------


class ErrorCategory(Enum):
    """Fine-grained error category for logging and telemetry.

    Each category maps to exactly one ErrorKind. The category provides
    human-readable context about what went wrong, while the kind drives
    the retry/terminate decision.
    """

    # -- Transient categories ------------------------------------------------
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_CONNECTION = "llm_connection"
    LLM_SERVER_ERROR = "llm_server_error"
    SSH_CONNECTION = "ssh_connection"
    NETWORK = "network"
    TOOL_TIMEOUT = "tool_timeout"

    # -- Permanent categories ------------------------------------------------
    LLM_AUTH = "llm_auth"
    LLM_MALFORMED_RESPONSE = "llm_malformed_response"
    LLM_UNSUPPORTED = "llm_unsupported"
    SSH_AUTH = "ssh_auth"
    USER_DENIAL = "user_denial"
    USER_CANCEL = "user_cancel"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_VALIDATION = "tool_validation"
    UNKNOWN = "unknown"

    @property
    def kind(self) -> ErrorKind:
        """Return the ErrorKind this category maps to."""
        return _CATEGORY_TO_KIND[self]


# Pre-computed mapping from category to kind.
_CATEGORY_TO_KIND: dict[ErrorCategory, ErrorKind] = {
    # Transient
    ErrorCategory.LLM_TIMEOUT: ErrorKind.TRANSIENT,
    ErrorCategory.LLM_RATE_LIMIT: ErrorKind.TRANSIENT,
    ErrorCategory.LLM_CONNECTION: ErrorKind.TRANSIENT,
    ErrorCategory.LLM_SERVER_ERROR: ErrorKind.TRANSIENT,
    ErrorCategory.SSH_CONNECTION: ErrorKind.TRANSIENT,
    ErrorCategory.NETWORK: ErrorKind.TRANSIENT,
    ErrorCategory.TOOL_TIMEOUT: ErrorKind.TRANSIENT,
    # Permanent
    ErrorCategory.LLM_AUTH: ErrorKind.PERMANENT,
    ErrorCategory.LLM_MALFORMED_RESPONSE: ErrorKind.PERMANENT,
    ErrorCategory.LLM_UNSUPPORTED: ErrorKind.PERMANENT,
    ErrorCategory.SSH_AUTH: ErrorKind.PERMANENT,
    ErrorCategory.USER_DENIAL: ErrorKind.PERMANENT,
    ErrorCategory.USER_CANCEL: ErrorKind.PERMANENT,
    ErrorCategory.TOOL_NOT_FOUND: ErrorKind.PERMANENT,
    ErrorCategory.TOOL_VALIDATION: ErrorKind.PERMANENT,
    ErrorCategory.UNKNOWN: ErrorKind.PERMANENT,
}


# ---------------------------------------------------------------------------
# Agent-specific error types
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Base error for agent-loop-specific failures."""


class UserDenialError(AgentError):
    """Raised when the user denies a proposed SSH command or tool call.

    This is a permanent error -- the agent loop must terminate when the
    user explicitly refuses an operation.
    """


class UserCancelError(AgentError):
    """Raised when the user cancels the agent loop (e.g., Ctrl-C, cancel button).

    This is a permanent error -- the user has explicitly requested termination.
    """


class ToolNotFoundError(AgentError):
    """Raised when the LLM requests a tool that is not in the registry.

    This is a permanent error -- the tool name is invalid and retrying
    with the same name will not help.
    """


class RetryExhaustedError(AgentError):
    """Raised when the agent loop exhausts all transient error retries.

    This signals that the agent loop failed due to persistent transient
    errors (network blips, LLM timeouts) and the caller should fall back
    to the one-shot LLM translation path. The original error message is
    preserved for logging.

    Attributes:
        iterations_used: Number of agent loop iterations consumed before
            retries were exhausted.
    """

    def __init__(
        self,
        message: str,
        *,
        iterations_used: int = 0,
    ) -> None:
        super().__init__(message)
        self.iterations_used = iterations_used


class ToolValidationError(AgentError):
    """Raised when tool arguments fail schema validation.

    This is a permanent error within a single dispatch -- the arguments
    are malformed. The agent may observe this error and propose corrected
    arguments in the next iteration, but the current dispatch is not
    retryable.
    """


# ---------------------------------------------------------------------------
# ClassifiedError frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassifiedError:
    """Immutable classification result for an exception.

    Wraps the original exception with its classification (kind + category)
    and a human-readable message. The ``is_retryable`` property provides a
    convenient boolean for retry/terminate decisions.

    Attributes:
        kind: Binary classification (TRANSIENT or PERMANENT).
        category: Fine-grained error category for logging/telemetry.
        message: Human-readable description of the classified error.
        original: The original exception that was classified. None if
            the error was constructed synthetically.
    """

    kind: ErrorKind
    category: ErrorCategory
    message: str
    original: BaseException | None = None

    @property
    def is_retryable(self) -> bool:
        """True if this error is transient and safe to retry."""
        return self.kind is ErrorKind.TRANSIENT

    def __str__(self) -> str:
        return f"[{self.kind.value.upper()}:{self.category.value}] {self.message}"


# ---------------------------------------------------------------------------
# Built-in exception type tables
# ---------------------------------------------------------------------------

# Python built-in exceptions that indicate transient network/IO failures.
# Note: TimeoutError is intentionally absent -- it is handled separately in
# _classify_builtin_error() to assign the LLM_TIMEOUT category rather than NETWORK.
# ConnectionRefusedError, ConnectionResetError, ConnectionAbortedError, and
# BrokenPipeError are all subclasses of ConnectionError/OSError but are listed
# explicitly for clarity in case the hierarchy is checked against directly.
_TRANSIENT_BUILTIN_TYPES: tuple[type[BaseException], ...] = (
    ConnectionError,
    OSError,
    BrokenPipeError,
    EOFError,
)

# Python built-in exceptions that indicate permanent failures.
# Note: PermissionError is handled separately in _classify_builtin_error()
# because it is a subclass of OSError (which is transient). The explicit
# check ensures PermissionError is classified as permanent before the
# OSError check catches it as transient.
_PERMANENT_BUILTIN_TYPES: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)

# HTTP status codes that indicate transient server-side issues.
_TRANSIENT_HTTP_STATUS_CODES: frozenset[int] = frozenset({
    408,  # Request Timeout
    429,  # Too Many Requests (rate limit)
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
})


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------


def _classify_http_status(status_code: int) -> tuple[ErrorKind, ErrorCategory]:
    """Classify an HTTP status code into kind and category.

    Args:
        status_code: The HTTP status code.

    Returns:
        Tuple of (ErrorKind, ErrorCategory).
    """
    if status_code == 429:
        return ErrorKind.TRANSIENT, ErrorCategory.LLM_RATE_LIMIT
    if status_code == 408:
        return ErrorKind.TRANSIENT, ErrorCategory.LLM_TIMEOUT
    if 500 <= status_code < 600:
        return ErrorKind.TRANSIENT, ErrorCategory.LLM_SERVER_ERROR
    return ErrorKind.PERMANENT, ErrorCategory.UNKNOWN


def _classify_agent_error(exc: BaseException) -> ClassifiedError | None:
    """Classify agent-specific error types.

    Returns None if the exception is not an agent-specific type.
    """
    if isinstance(exc, UserDenialError):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.USER_DENIAL,
            message=str(exc) if exc.args else "User denied the operation",
            original=exc,
        )
    if isinstance(exc, UserCancelError):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.USER_CANCEL,
            message=str(exc) if exc.args else "User cancelled the operation",
            original=exc,
        )
    if isinstance(exc, ToolNotFoundError):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.TOOL_NOT_FOUND,
            message=str(exc) if exc.args else "Tool not found in registry",
            original=exc,
        )
    if isinstance(exc, ToolValidationError):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.TOOL_VALIDATION,
            message=str(exc) if exc.args else "Tool arguments failed validation",
            original=exc,
        )
    return None


def _classify_llm_error(exc: BaseException) -> ClassifiedError | None:
    """Classify errors from the jules_daemon.llm.errors hierarchy.

    Returns None if the exception is not from the LLM error hierarchy.
    Handles LLMResponseError specially by inspecting its status_code.
    """
    try:
        from jules_daemon.llm.errors import (
            LLMAuthenticationError,
            LLMConnectionError,
            LLMError,
            LLMParseError,
            LLMResponseError,
            LLMToolCallingUnsupportedError,
        )
    except ImportError:
        return None

    if isinstance(exc, LLMAuthenticationError):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.LLM_AUTH,
            message=str(exc),
            original=exc,
        )

    if isinstance(exc, LLMConnectionError):
        return ClassifiedError(
            kind=ErrorKind.TRANSIENT,
            category=ErrorCategory.LLM_CONNECTION,
            message=str(exc),
            original=exc,
        )

    if isinstance(exc, LLMParseError):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.LLM_MALFORMED_RESPONSE,
            message=str(exc),
            original=exc,
        )

    if isinstance(exc, LLMToolCallingUnsupportedError):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.LLM_UNSUPPORTED,
            message=str(exc),
            original=exc,
        )

    if isinstance(exc, LLMResponseError):
        # LLMResponseError has an optional status_code that determines
        # whether this is transient (5xx, 429) or permanent (4xx).
        status_code = exc.status_code
        if status_code is not None:
            kind, category = _classify_http_status(status_code)
            return ClassifiedError(
                kind=kind,
                category=category,
                message=str(exc),
                original=exc,
            )
        # No status code -- treat as permanent (we cannot know if retry helps)
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.UNKNOWN,
            message=str(exc),
            original=exc,
        )

    # Generic LLMError -- permanent by default
    if isinstance(exc, LLMError):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.UNKNOWN,
            message=str(exc),
            original=exc,
        )

    return None


def _classify_ssh_error(exc: BaseException) -> ClassifiedError | None:
    """Classify errors from the jules_daemon.ssh.errors hierarchy.

    Returns None if the exception is not from the SSH error hierarchy.
    """
    try:
        from jules_daemon.ssh.errors import (
            SSHAuthenticationError,
            SSHConnectionError,
            SSHHostKeyError,
        )
    except ImportError:
        return None

    if isinstance(exc, SSHAuthenticationError):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.SSH_AUTH,
            message=str(exc),
            original=exc,
        )

    if isinstance(exc, SSHHostKeyError):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.SSH_AUTH,
            message=str(exc),
            original=exc,
        )

    if isinstance(exc, SSHConnectionError):
        return ClassifiedError(
            kind=ErrorKind.TRANSIENT,
            category=ErrorCategory.SSH_CONNECTION,
            message=str(exc),
            original=exc,
        )

    return None


def _classify_openai_sdk_error(exc: BaseException) -> ClassifiedError | None:
    """Classify OpenAI SDK-specific exception types.

    Returns None if openai is not installed or the exception is not
    from the OpenAI SDK.
    """
    try:
        import openai
    except ImportError:
        return None

    # APITimeoutError is a subclass of APIConnectionError -- check it first
    # to avoid the more general check masking the specific timeout category.
    if isinstance(exc, openai.APITimeoutError):
        return ClassifiedError(
            kind=ErrorKind.TRANSIENT,
            category=ErrorCategory.LLM_TIMEOUT,
            message=str(exc),
            original=exc,
        )

    if isinstance(exc, openai.APIConnectionError):
        return ClassifiedError(
            kind=ErrorKind.TRANSIENT,
            category=ErrorCategory.LLM_CONNECTION,
            message=str(exc),
            original=exc,
        )

    if isinstance(exc, openai.RateLimitError):
        return ClassifiedError(
            kind=ErrorKind.TRANSIENT,
            category=ErrorCategory.LLM_RATE_LIMIT,
            message=str(exc),
            original=exc,
        )

    if isinstance(exc, openai.AuthenticationError):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.LLM_AUTH,
            message=str(exc),
            original=exc,
        )

    # Check for status_code on other OpenAI SDK errors
    if isinstance(exc, openai.APIStatusError):
        kind, category = _classify_http_status(exc.status_code)
        return ClassifiedError(
            kind=kind,
            category=category,
            message=str(exc),
            original=exc,
        )

    return None


def _classify_builtin_error(exc: BaseException) -> ClassifiedError | None:
    """Classify Python built-in exception types.

    Returns None if the exception does not match any known built-in type.
    Checks more specific types before general types to avoid masking
    (e.g., PermissionError before OSError, since PermissionError is a
    subclass of OSError).
    """
    # Check permanent types first (some are subclasses of transient types,
    # e.g., PermissionError is a subclass of OSError)
    if isinstance(exc, PermissionError):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.UNKNOWN,
            message=str(exc),
            original=exc,
        )

    if isinstance(exc, _PERMANENT_BUILTIN_TYPES):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.UNKNOWN,
            message=str(exc),
            original=exc,
        )

    # Timeout errors get LLM_TIMEOUT category.
    # On Python 3.11+, asyncio.TimeoutError is TimeoutError (same class).
    if isinstance(exc, TimeoutError):
        return ClassifiedError(
            kind=ErrorKind.TRANSIENT,
            category=ErrorCategory.LLM_TIMEOUT,
            message=str(exc) if exc.args else "Operation timed out",
            original=exc,
        )

    # Network/IO errors get NETWORK category
    if isinstance(exc, _TRANSIENT_BUILTIN_TYPES):
        return ClassifiedError(
            kind=ErrorKind.TRANSIENT,
            category=ErrorCategory.NETWORK,
            message=str(exc),
            original=exc,
        )

    return None


def _classify_by_status_code(exc: BaseException) -> ClassifiedError | None:
    """Classify by HTTP status_code attribute on the exception, if present.

    Many HTTP client libraries attach a ``status_code`` attribute to
    their exception instances. This function reads that attribute and
    classifies based on standard HTTP semantics.

    Returns None if the exception has no status_code attribute.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is not None and isinstance(status_code, int):
        kind, category = _classify_http_status(status_code)
        return ClassifiedError(
            kind=kind,
            category=category,
            message=str(exc),
            original=exc,
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_error(exc: BaseException) -> ClassifiedError:
    """Classify an exception as transient (retryable) or permanent.

    Checks multiple classification layers in priority order, returning
    the first match. Unknown exceptions default to PERMANENT for safety
    (it is better to terminate than to retry indefinitely).

    Classification priority:
        1. KeyboardInterrupt -> USER_CANCEL (permanent)
        2. Agent-specific types (UserDenialError, ToolNotFoundError, etc.)
        3. LLM error hierarchy (LLMAuthenticationError, LLMConnectionError, etc.)
        4. SSH error hierarchy (SSHAuthenticationError, SSHConnectionError, etc.)
        5. OpenAI SDK types (APIConnectionError, RateLimitError, etc.)
        6. HTTP status_code attribute on the exception
        7. Python built-in types (ConnectionError, TimeoutError, etc.)
        8. Default: PERMANENT / UNKNOWN

    Args:
        exc: The exception to classify.

    Returns:
        ClassifiedError with kind, category, message, and original.
    """
    # 1. KeyboardInterrupt is always user cancel
    if isinstance(exc, KeyboardInterrupt):
        return ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.USER_CANCEL,
            message="User cancelled via keyboard interrupt",
            original=exc,
        )

    # 2-7: Check each classifier in priority order
    classifiers = (
        _classify_agent_error,
        _classify_llm_error,
        _classify_ssh_error,
        _classify_openai_sdk_error,
        _classify_by_status_code,
        _classify_builtin_error,
    )

    for classifier in classifiers:
        result = classifier(exc)
        if result is not None:
            return result

    # 8. Default: permanent / unknown
    return ClassifiedError(
        kind=ErrorKind.PERMANENT,
        category=ErrorCategory.UNKNOWN,
        message=str(exc) or f"Unclassified error: {type(exc).__name__}",
        original=exc,
    )


def is_transient(exc: BaseException) -> bool:
    """Return True if the exception is transient (retryable).

    Convenience wrapper around ``classify_error()`` for simple boolean
    checks in retry loops.

    Args:
        exc: The exception to check.

    Returns:
        True if the error is transient and safe to retry.
    """
    return classify_error(exc).is_retryable


def is_permanent(exc: BaseException) -> bool:
    """Return True if the exception is permanent (not retryable).

    Convenience wrapper around ``classify_error()`` for simple boolean
    checks in termination logic.

    Args:
        exc: The exception to check.

    Returns:
        True if the error is permanent and should not be retried.
    """
    return not classify_error(exc).is_retryable
