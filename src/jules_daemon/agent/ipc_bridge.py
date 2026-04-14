"""IPC bridge callbacks for agent loop tools.

Provides factory functions that create the confirm_callback,
ask_callback, and notify_callback closures required by the agent
tool set. These callbacks bridge between the agent tool layer
and the IPC CONFIRM_PROMPT/CONFIRM_REPLY protocol.

Each factory captures a ``ClientConnection`` reference and returns
an async callable that the tool can invoke without knowing about
IPC framing details.

Usage::

    from jules_daemon.agent.ipc_bridge import (
        make_ask_callback,
        make_confirm_callback,
        make_notify_callback,
    )

    confirm_cb = make_confirm_callback(client)
    ask_cb = make_ask_callback(client)
    notify_cb = make_notify_callback(client)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
    unpack_header,
)

if TYPE_CHECKING:
    from jules_daemon.ipc.notification_broadcaster import NotificationBroadcaster
    from jules_daemon.ipc.server import ClientConnection

__all__ = [
    "make_ask_callback",
    "make_confirm_callback",
    "make_notify_callback",
]

logger = logging.getLogger(__name__)

_CONFIRM_TIMEOUT: float = 120.0


# ---------------------------------------------------------------------------
# Internal I/O helpers (duplicated from RequestHandler static methods to
# avoid a dependency on the handler class -- keeps the bridge self-contained)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


async def _send_envelope(
    client: ClientConnection,
    envelope: MessageEnvelope,
) -> None:
    """Encode and send a framed envelope to the client.

    Args:
        client: The client connection with writer.
        envelope: The envelope to send.

    Raises:
        OSError: On connection failure.
    """
    frame = encode_frame(envelope)
    client.writer.write(frame)
    await client.writer.drain()


async def _read_envelope(
    client: ClientConnection,
    *,
    timeout: float = _CONFIRM_TIMEOUT,
) -> MessageEnvelope | None:
    """Read a single framed envelope from the client.

    Args:
        client: The client connection with reader.
        timeout: Maximum seconds to wait for a response.

    Returns:
        The decoded MessageEnvelope, or None on EOF/disconnect.

    Raises:
        asyncio.TimeoutError: If no message arrives within timeout.
    """
    try:
        header_bytes = await asyncio.wait_for(
            client.reader.readexactly(HEADER_SIZE),
            timeout=timeout,
        )
    except asyncio.IncompleteReadError:
        return None

    payload_length = unpack_header(header_bytes)
    payload_bytes = await asyncio.wait_for(
        client.reader.readexactly(payload_length),
        timeout=timeout,
    )
    return decode_envelope(payload_bytes)


# ---------------------------------------------------------------------------
# Callback factories
# ---------------------------------------------------------------------------


def make_confirm_callback(
    client: ClientConnection,
    *,
    target_context: Mapping[str, Any] | None = None,
) -> object:
    """Create an async confirm callback bound to a client connection.

    The returned callable has the signature expected by
    ``ProposeSSHCommandTool``::

        async (command: str, target_host: str, explanation: str)
            -> tuple[bool, str]

    The callback sends a CONFIRM_PROMPT to the CLI, waits for a
    CONFIRM_REPLY, and returns ``(approved, final_command)`` where
    ``final_command`` may differ from the original if the user edited it.

    Args:
        client: The IPC client connection for the active session.

    Returns:
        An async callable satisfying the ConfirmCallback protocol.
    """

    def _prompt_target_payload(target_host: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"target_host": target_host}
        context = dict(target_context or {})
        context_host = context.get("target_host")
        if isinstance(context_host, str) and context_host.strip():
            if context_host.strip() != target_host.strip():
                return payload
        target_user = context.get("target_user")
        if isinstance(target_user, str) and target_user.strip():
            payload["target_user"] = target_user.strip()
        target_port = context.get("target_port")
        if isinstance(target_port, int):
            payload["target_port"] = target_port
        for source_key, payload_key in (
            ("resolved_system_name", "system_name"),
            ("resolved_system_hostname", "system_hostname"),
            ("resolved_system_ip_address", "system_ip_address"),
            ("resolved_system_description", "system_description"),
            ("auth_mode", "auth_mode"),
            ("credential_source", "credential_source"),
            ("credential_guidance", "credential_guidance"),
        ):
            value = context.get(source_key)
            if isinstance(value, str) and value.strip():
                payload[payload_key] = value.strip()
        return payload

    async def _confirm(
        command: str,
        target_host: str,
        explanation: str,
    ) -> tuple[bool, str]:
        confirm_msg_id = f"confirm-{uuid.uuid4().hex[:12]}"
        prompt_payload = _prompt_target_payload(target_host)
        target_user = prompt_payload.get("target_user", "")
        target_port = prompt_payload.get("target_port")
        target_label = target_host
        if isinstance(target_user, str) and target_user:
            target_label = f"{target_user}@{target_label}"
        if isinstance(target_port, int):
            target_label = f"{target_label}:{target_port}"
        message_text = (
            f"{explanation}\n"
            f"Execute on {target_label}?\n"
            f"  $ {command}"
        ) if explanation else (
            f"Execute on {target_label}?\n"
            f"  $ {command}"
        )

        prompt = MessageEnvelope(
            msg_type=MessageType.CONFIRM_PROMPT,
            msg_id=confirm_msg_id,
            timestamp=_now_iso(),
            payload={
                "proposed_command": command,
                "message": message_text,
                **prompt_payload,
            },
        )

        await _send_envelope(client, prompt)

        reply = await _read_envelope(client, timeout=_CONFIRM_TIMEOUT)
        if reply is None:
            return (False, command)

        approved = reply.payload.get("approved", False)
        if not approved:
            return (False, command)

        edited_command = reply.payload.get("edited_command")
        final_command = edited_command if edited_command else command
        return (True, final_command)

    return _confirm


def make_ask_callback(
    client: ClientConnection,
) -> object:
    """Create an async ask callback bound to a client connection.

    The returned callable has the signature expected by
    ``AskUserQuestionTool``::

        async (question: str, context: str) -> str | None

    The callback sends a question via CONFIRM_PROMPT and waits for
    the user's text reply. Returns ``None`` if the user cancels.

    Args:
        client: The IPC client connection for the active session.

    Returns:
        An async callable satisfying the AskCallback protocol.
    """

    async def _ask(question: str, context: str) -> str | None:
        ask_msg_id = f"ask-{uuid.uuid4().hex[:12]}"

        full_message = question
        if context:
            full_message = f"{context}\n\n{question}"

        prompt = MessageEnvelope(
            msg_type=MessageType.CONFIRM_PROMPT,
            msg_id=ask_msg_id,
            timestamp=_now_iso(),
            payload={
                "prompt_title": "Question",
                "question": question,
                "context": context,
                "message": full_message,
                "type": "question",
            },
        )

        await _send_envelope(client, prompt)

        reply = await _read_envelope(client, timeout=_CONFIRM_TIMEOUT)
        if reply is None:
            return None

        # Check for cancellation
        if reply.payload.get("cancelled", False):
            return None
        if not reply.payload.get("approved", True):
            return None

        return reply.payload.get(
            "answer",
            reply.payload.get("text", ""),
        )

    return _ask


def make_notify_callback(
    client: ClientConnection,
    *,
    notification_broadcaster: NotificationBroadcaster | None = None,
) -> object:
    """Create an async notify callback bound to a client connection.

    The returned callable has the signature expected by
    ``NotifyUserTool``::

        async (message: str, severity: str) -> bool

    The callback prefers the daemon's notification broadcaster when one
    is available so subscribed CLIs receive a persistent push event.
    When no broadcaster-backed subscribers exist, it falls back to a
    best-effort direct STREAM message to the active client.

    Args:
        client: The IPC client connection for the active session.

    Returns:
        An async callable satisfying the NotifyCallback protocol.
    """

    async def _notify(message: str, severity: str = "info") -> bool:
        if (
            notification_broadcaster is not None
            and notification_broadcaster.subscriber_count > 0
        ):
            from jules_daemon.protocol.notifications import (
                AlertNotification,
                NotificationEventType,
                NotificationSeverity,
                create_notification_envelope,
            )

            try:
                severity_enum = NotificationSeverity(severity)
            except ValueError:
                severity_enum = NotificationSeverity.INFO

            notification = create_notification_envelope(
                event_type=NotificationEventType.ALERT,
                payload=AlertNotification(
                    severity=severity_enum,
                    title="Agent notification",
                    message=message,
                    details={"source": "notify_user"},
                ),
            )
            try:
                result = await notification_broadcaster.broadcast(notification)
                if result.delivered_count > 0:
                    return True
            except Exception as exc:
                logger.debug("Failed to broadcast notification: %s", exc)

        notify_msg = MessageEnvelope(
            msg_type=MessageType.STREAM,
            msg_id=f"notify-{uuid.uuid4().hex[:12]}",
            timestamp=_now_iso(),
            payload={
                "line": message + "\n",
                "is_end": False,
                "level": severity,
            },
        )
        try:
            await _send_envelope(client, notify_msg)
            return True
        except Exception as exc:
            logger.debug("Failed to send notification: %s", exc)
            return False

    return _notify
