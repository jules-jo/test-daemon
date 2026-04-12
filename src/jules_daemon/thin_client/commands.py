"""Verb-specific envelope factories for the thin client.

Each factory builds a MessageEnvelope for a specific daemon verb.
These are intentionally simple -- just verb + required fields, no
NL classification or structured parsing. This proves that any client
can construct valid IPC messages from minimal inputs.

All factories return immutable MessageEnvelope instances. None of
them perform IO or mutation.

Usage::

    from jules_daemon.thin_client.commands import build_status_request

    envelope = build_status_request(verbose=True)
    await connection.send(envelope)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from jules_daemon.ipc.framing import MessageEnvelope, MessageType

__all__ = [
    "build_cancel_request",
    "build_confirm_reply",
    "build_discover_request",
    "build_health_request",
    "build_history_request",
    "build_run_request",
    "build_status_request",
    "build_watch_request",
    "SSHTargetParams",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _generate_msg_id() -> str:
    """Generate a unique message ID for request-response correlation."""
    return f"thin-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# SSH target parameter container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SSHTargetParams:
    """Immutable SSH target parameters for run/queue commands.

    Attributes:
        host: Remote hostname or IP address.
        user: SSH username on the remote host.
        port: SSH port. Default 22.
        key_path: Absolute path to SSH private key. None for default.
    """

    host: str
    user: str
    port: int = 22
    key_path: str | None = None

    def __post_init__(self) -> None:
        if not self.host or not self.host.strip():
            raise ValueError("host must not be empty")
        if not self.user or not self.user.strip():
            raise ValueError("user must not be empty")
        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be 1-65535, got {self.port}")
        if self.key_path is not None and not self.key_path.startswith("/"):
            raise ValueError(
                f"key_path must be an absolute path, got {self.key_path!r}"
            )

    def to_payload_dict(self) -> dict[str, object]:
        """Convert to a dict suitable for inclusion in a message payload.

        Returns:
            Dict with host, user, port, and optionally key_path.
        """
        result: dict[str, object] = {
            "target_host": self.host,
            "target_user": self.user,
            "target_port": self.port,
        }
        if self.key_path is not None:
            result["key_path"] = self.key_path
        return result


# ---------------------------------------------------------------------------
# Health / handshake
# ---------------------------------------------------------------------------


def build_health_request() -> MessageEnvelope:
    """Build a health-check REQUEST envelope.

    The health verb is the simplest possible daemon interaction --
    it verifies the daemon is responsive without requiring any
    arguments.

    Returns:
        MessageEnvelope for a health check.
    """
    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=_generate_msg_id(),
        timestamp=_now_iso(),
        payload={"verb": "health"},
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def build_status_request(*, verbose: bool = False) -> MessageEnvelope:
    """Build a status REQUEST envelope.

    Args:
        verbose: When True, request extended details.

    Returns:
        MessageEnvelope for a status query.
    """
    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=_generate_msg_id(),
        timestamp=_now_iso(),
        payload={"verb": "status", "verbose": verbose},
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def build_history_request(
    *,
    limit: int = 20,
    status_filter: str | None = None,
    host_filter: str | None = None,
) -> MessageEnvelope:
    """Build a history REQUEST envelope.

    Args:
        limit: Maximum number of records to return.
        status_filter: Optional status filter (idle, running, etc.).
        host_filter: Optional hostname filter.

    Returns:
        MessageEnvelope for a history query.

    Raises:
        ValueError: If limit is not positive or exceeds 1000.
    """
    if limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    if limit > 1000:
        raise ValueError(f"limit must not exceed 1000, got {limit}")

    payload: dict[str, object] = {"verb": "history", "limit": limit}
    if status_filter is not None:
        payload["status_filter"] = status_filter
    if host_filter is not None:
        payload["host_filter"] = host_filter

    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=_generate_msg_id(),
        timestamp=_now_iso(),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def build_cancel_request(
    *,
    run_id: str | None = None,
    force: bool = False,
    reason: str | None = None,
) -> MessageEnvelope:
    """Build a cancel REQUEST envelope.

    Args:
        run_id: Target a specific run. None cancels the current run.
        force: When True, send SIGKILL instead of SIGTERM.
        reason: Optional human-readable cancellation reason.

    Returns:
        MessageEnvelope for a cancel command.
    """
    payload: dict[str, object] = {"verb": "cancel", "force": force}
    if run_id is not None:
        payload["run_id"] = run_id
    if reason is not None:
        payload["reason"] = reason

    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=_generate_msg_id(),
        timestamp=_now_iso(),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def build_run_request(
    *,
    target: SSHTargetParams,
    natural_language: str,
) -> MessageEnvelope:
    """Build a run REQUEST envelope.

    Args:
        target: SSH target connection parameters.
        natural_language: Free-form description of what tests to run.

    Returns:
        MessageEnvelope for a run command.

    Raises:
        ValueError: If natural_language is empty.
    """
    if not natural_language or not natural_language.strip():
        raise ValueError("natural_language must not be empty")

    payload: dict[str, object] = {
        "verb": "run",
        "natural_language": natural_language,
        **target.to_payload_dict(),
    }

    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=_generate_msg_id(),
        timestamp=_now_iso(),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Watch
# ---------------------------------------------------------------------------


def build_watch_request(
    *,
    run_id: str | None = None,
    tail_lines: int = 50,
    follow: bool = True,
) -> MessageEnvelope:
    """Build a watch REQUEST envelope.

    Args:
        run_id: Target a specific run. None watches the current run.
        tail_lines: Number of recent output lines on initial attach.
        follow: When True, continuously stream new output.

    Returns:
        MessageEnvelope for a watch subscription.

    Raises:
        ValueError: If tail_lines is not positive.
    """
    if tail_lines < 1:
        raise ValueError(f"tail_lines must be positive, got {tail_lines}")

    payload: dict[str, object] = {
        "verb": "watch",
        "tail_lines": tail_lines,
        "follow": follow,
    }
    if run_id is not None:
        payload["run_id"] = run_id

    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=_generate_msg_id(),
        timestamp=_now_iso(),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


def build_discover_request(
    *,
    target: SSHTargetParams,
    command: str,
) -> MessageEnvelope:
    """Build a discover REQUEST envelope.

    Args:
        target: SSH target connection parameters.
        command: The command to discover (will be run with -h).

    Returns:
        MessageEnvelope for a discover command.

    Raises:
        ValueError: If command is empty.
    """
    if not command or not command.strip():
        raise ValueError("command must not be empty")

    payload: dict[str, object] = {
        "verb": "discover",
        "command": command,
        **target.to_payload_dict(),
    }

    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=_generate_msg_id(),
        timestamp=_now_iso(),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Confirm reply
# ---------------------------------------------------------------------------


def build_confirm_reply(
    *,
    approved: bool,
    original_msg_id: str,
    edited_command: str | None = None,
) -> MessageEnvelope:
    """Build a CONFIRM_REPLY envelope for the security approval flow.

    This is the response the client sends when the daemon asks for
    SSH command approval. Every SSH command requires explicit human
    confirmation -- this envelope carries that decision.

    Args:
        approved: True if the user approved the command.
        original_msg_id: The msg_id from the CONFIRM_PROMPT envelope
            this reply corresponds to.
        edited_command: If the user edited the command, the new text.
            None means the original command was accepted as-is.

    Returns:
        MessageEnvelope with the user's confirmation decision.
    """
    payload: dict[str, object] = {
        "verb": "confirm",
        "approved": approved,
        "original_msg_id": original_msg_id,
    }
    if edited_command is not None:
        payload["edited_command"] = edited_command
        # Also set as "answer" and "text" so the ask_user_question IPC
        # bridge can read the user's response from the same reply envelope
        payload["answer"] = edited_command
        payload["text"] = edited_command

    return MessageEnvelope(
        msg_type=MessageType.CONFIRM_REPLY,
        msg_id=_generate_msg_id(),
        timestamp=_now_iso(),
        payload=payload,
    )
