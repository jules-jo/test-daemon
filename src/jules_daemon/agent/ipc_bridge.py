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
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
    unpack_header,
)

if TYPE_CHECKING:
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

    async def _confirm(
        command: str,
        target_host: str,
        explanation: str,
    ) -> tuple[bool, str]:
        confirm_msg_id = f"confirm-{uuid.uuid4().hex[:12]}"
        message_text = (
            f"{explanation}\n"
            f"Execute on {target_host}?\n"
            f"  $ {command}"
        ) if explanation else (
            f"Execute on {target_host}?\n"
            f"  $ {command}"
        )

        prompt = MessageEnvelope(
            msg_type=MessageType.CONFIRM_PROMPT,
            msg_id=confirm_msg_id,
            timestamp=_now_iso(),
            payload={
                "proposed_command": command,
                "target_host": target_host,
                "message": message_text,
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
) -> object:
    """Create an async notify callback bound to a client connection.

    The returned callable has the signature expected by
    ``NotifyUserTool``::

        async (message: str, severity: str) -> bool

    The callback sends a STREAM message to the CLI. Failures are
    silently caught (best-effort delivery).

    Args:
        client: The IPC client connection for the active session.

    Returns:
        An async callable satisfying the NotifyCallback protocol.
    """

    async def _notify(message: str, severity: str = "info") -> bool:
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
