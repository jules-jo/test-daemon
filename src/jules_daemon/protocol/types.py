"""IPC protocol version constants, message-type enums, and status codes.

This module is the canonical source of truth for the Jules daemon IPC
protocol. Every component that participates in CLI-daemon communication
imports its protocol-level constants and type definitions from here.

Contents:

    Protocol identity:
        ``PROTOCOL_NAME``           -- human-readable protocol identifier
        ``PROTOCOL_VERSION``        -- semver version string
        ``PROTOCOL_VERSION_MAJOR``  -- major version component
        ``PROTOCOL_VERSION_MINOR``  -- minor version component
        ``PROTOCOL_VERSION_PATCH``  -- patch version component

    Message kinds (``MessageKind``):
        REQUEST         -- CLI sends a command to the daemon
        RESPONSE        -- daemon returns a result to the CLI
        NOTIFICATION    -- daemon pushes an unsolicited notification
        ERROR           -- error from either side
        STREAM          -- daemon pushes streaming output (watch verb)
        CONFIRM_PROMPT  -- daemon asks CLI to display a confirmation dialog
        CONFIRM_REPLY   -- CLI sends the user's approval or denial

    Status codes (``StatusCode``):
        Numeric codes modelled after HTTP semantics (2xx success,
        4xx client error, 5xx server error) for structured IPC responses.

    Helpers:
        ``parse_message_kind``   -- case-insensitive string -> MessageKind
        ``parse_status_code``    -- int -> StatusCode
        ``is_terminal_message``  -- True for kinds that end a request cycle
        ``is_success``           -- True for 2xx status codes
        ``is_client_error``      -- True for 4xx status codes
        ``is_server_error``      -- True for 5xx status codes
        ``status_code_to_reason``-- StatusCode -> human-readable reason string

Usage::

    from jules_daemon.protocol.types import (
        PROTOCOL_VERSION,
        MessageKind,
        StatusCode,
        is_success,
        parse_message_kind,
    )

    kind = parse_message_kind("request")
    assert kind is MessageKind.REQUEST
    assert is_success(StatusCode.OK)
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "PROTOCOL_VERSION_MAJOR",
    "PROTOCOL_VERSION_MINOR",
    "PROTOCOL_VERSION_PATCH",
    "MessageKind",
    "StatusCode",
    "is_client_error",
    "is_server_error",
    "is_success",
    "is_terminal_message",
    "parse_message_kind",
    "parse_status_code",
    "status_code_to_reason",
]


# ---------------------------------------------------------------------------
# Protocol identity
# ---------------------------------------------------------------------------

PROTOCOL_NAME: str = "jules-ipc"
"""Human-readable identifier for the IPC protocol."""

PROTOCOL_VERSION_MAJOR: int = 1
"""Major version -- incremented on breaking changes."""

PROTOCOL_VERSION_MINOR: int = 0
"""Minor version -- incremented on backward-compatible additions."""

PROTOCOL_VERSION_PATCH: int = 0
"""Patch version -- incremented on backward-compatible fixes."""

PROTOCOL_VERSION: str = (
    f"{PROTOCOL_VERSION_MAJOR}.{PROTOCOL_VERSION_MINOR}.{PROTOCOL_VERSION_PATCH}"
)
"""Full semver version string (``MAJOR.MINOR.PATCH``)."""


# ---------------------------------------------------------------------------
# MessageKind enum
# ---------------------------------------------------------------------------


class MessageKind(Enum):
    """Categories for every IPC message exchanged between CLI and daemon.

    The protocol supports seven distinct message kinds that cover
    request-response cycles, streaming output, asynchronous notifications,
    and the security confirmation flow.

    Values:
        REQUEST:        CLI sends a command to the daemon.
        RESPONSE:       Daemon sends a result back to the CLI. Terminal.
        NOTIFICATION:   Daemon pushes an unsolicited event to the CLI
                        (e.g., state change, warning). Not terminal.
        ERROR:          Error from either side. Terminal.
        STREAM:         Daemon pushes streaming output for the watch verb.
                        Not terminal (stream ends with RESPONSE or ERROR).
        CONFIRM_PROMPT: Daemon asks the CLI to display a confirmation
                        dialog for an SSH command (security approval).
        CONFIRM_REPLY:  CLI sends the user's approval or denial back
                        to the daemon.
    """

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    STREAM = "stream"
    CONFIRM_PROMPT = "confirm_prompt"
    CONFIRM_REPLY = "confirm_reply"


# Lookup table: lowered value -> MessageKind (for deserialization)
_MESSAGE_KIND_LOOKUP: dict[str, MessageKind] = {
    mk.value: mk for mk in MessageKind
}

# Terminal kinds end a request-response cycle
_TERMINAL_KINDS: frozenset[MessageKind] = frozenset({
    MessageKind.RESPONSE,
    MessageKind.ERROR,
})


def parse_message_kind(raw: str) -> MessageKind:
    """Parse a raw string into a MessageKind enum member.

    Matching is case-insensitive with leading/trailing whitespace stripped.

    Args:
        raw: Wire-format message kind string (e.g., ``"request"``).

    Returns:
        The matching ``MessageKind`` member.

    Raises:
        ValueError: If the string is empty or does not match any kind.
    """
    normalized = raw.strip().lower()
    if not normalized:
        raise ValueError("Message kind must not be empty")

    kind = _MESSAGE_KIND_LOOKUP.get(normalized)
    if kind is None:
        valid = ", ".join(sorted(_MESSAGE_KIND_LOOKUP))
        raise ValueError(
            f"Unknown message kind {raw.strip()!r}. Valid kinds: {valid}"
        )
    return kind


def is_terminal_message(kind: MessageKind) -> bool:
    """Return True if this message kind ends a request-response cycle.

    Terminal kinds are RESPONSE and ERROR. Non-terminal kinds (STREAM,
    NOTIFICATION, CONFIRM_PROMPT, CONFIRM_REPLY, REQUEST) may be followed
    by additional messages in the same exchange.

    Args:
        kind: The message kind to classify.

    Returns:
        True if the kind is terminal, False otherwise.
    """
    return kind in _TERMINAL_KINDS


# ---------------------------------------------------------------------------
# StatusCode enum
# ---------------------------------------------------------------------------


class StatusCode(Enum):
    """Numeric status codes for IPC responses.

    Modelled after HTTP status code semantics for familiarity:

    2xx -- Success:
        OK (200)                  Request succeeded.
        ACCEPTED (202)            Request accepted for async processing.
        NO_CONTENT (204)          Success with no response body.

    4xx -- Client errors:
        BAD_REQUEST (400)         Malformed request or invalid arguments.
        UNAUTHORIZED (401)        Missing or invalid authentication.
        FORBIDDEN (403)           Authenticated but not permitted.
        NOT_FOUND (404)           Requested resource does not exist.
        CONFLICT (409)            Request conflicts with current state
                                  (e.g., collision detection).
        UNPROCESSABLE (422)       Syntactically valid but semantically
                                  invalid request.

    5xx -- Server errors:
        INTERNAL_ERROR (500)      Unexpected daemon failure.
        NOT_IMPLEMENTED (501)     Verb or feature not yet implemented.
        SERVICE_UNAVAILABLE (502) Upstream service not reachable
                                  (e.g., SSH host, LLM backend).
        BUSY (503)                Daemon is busy with another run;
                                  command has been queued.
        TIMEOUT (504)             Operation timed out (e.g., SSH, LLM).
    """

    # -- Success (2xx) --
    OK = 200
    ACCEPTED = 202
    NO_CONTENT = 204

    # -- Client errors (4xx) --
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    UNPROCESSABLE = 422

    # -- Server errors (5xx) --
    INTERNAL_ERROR = 500
    NOT_IMPLEMENTED = 501
    SERVICE_UNAVAILABLE = 502
    BUSY = 503
    TIMEOUT = 504


# Lookup table: int value -> StatusCode (for deserialization)
_STATUS_CODE_LOOKUP: dict[int, StatusCode] = {
    sc.value: sc for sc in StatusCode
}

# Human-readable reason strings, keyed by StatusCode member
_STATUS_REASON: dict[StatusCode, str] = {
    StatusCode.OK: "OK",
    StatusCode.ACCEPTED: "Accepted",
    StatusCode.NO_CONTENT: "No Content",
    StatusCode.BAD_REQUEST: "Bad Request",
    StatusCode.UNAUTHORIZED: "Unauthorized",
    StatusCode.FORBIDDEN: "Forbidden",
    StatusCode.NOT_FOUND: "Not Found",
    StatusCode.CONFLICT: "Conflict",
    StatusCode.UNPROCESSABLE: "Unprocessable",
    StatusCode.INTERNAL_ERROR: "Internal Error",
    StatusCode.NOT_IMPLEMENTED: "Not Implemented",
    StatusCode.SERVICE_UNAVAILABLE: "Service Unavailable",
    StatusCode.BUSY: "Busy",
    StatusCode.TIMEOUT: "Timeout",
}


def parse_status_code(code: int) -> StatusCode:
    """Parse a numeric code into a StatusCode enum member.

    Args:
        code: Integer status code (e.g., ``200``).

    Returns:
        The matching ``StatusCode`` member.

    Raises:
        ValueError: If the code does not match any known status.
    """
    status = _STATUS_CODE_LOOKUP.get(code)
    if status is None:
        valid = ", ".join(str(v) for v in sorted(_STATUS_CODE_LOOKUP))
        raise ValueError(
            f"Unknown status code {code}. Valid codes: {valid}"
        )
    return status


def is_success(code: StatusCode) -> bool:
    """Return True if the status code indicates success (2xx range).

    Args:
        code: The status code to classify.

    Returns:
        True if the code is in the 200-299 range.
    """
    return 200 <= code.value < 300


def is_client_error(code: StatusCode) -> bool:
    """Return True if the status code indicates a client error (4xx range).

    Args:
        code: The status code to classify.

    Returns:
        True if the code is in the 400-499 range.
    """
    return 400 <= code.value < 500


def is_server_error(code: StatusCode) -> bool:
    """Return True if the status code indicates a server error (5xx range).

    Args:
        code: The status code to classify.

    Returns:
        True if the code is in the 500-599 range.
    """
    return 500 <= code.value < 600


def status_code_to_reason(code: StatusCode) -> str:
    """Return the human-readable reason string for a status code.

    Every ``StatusCode`` member has a corresponding reason string.

    Args:
        code: The status code to look up.

    Returns:
        Short human-readable reason (e.g., ``"OK"``, ``"Bad Request"``).
    """
    return _STATUS_REASON[code]
