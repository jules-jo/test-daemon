"""Tests for the IPC bridge callbacks (agent/ipc_bridge.py).

Validates that the bridge callbacks correctly translate between the
agent tool layer and the IPC CONFIRM_PROMPT/CONFIRM_REPLY protocol:
- confirm_callback sends CONFIRM_PROMPT, reads CONFIRM_REPLY
- ask_callback sends question prompt, reads text reply
- notify_callback prefers broadcaster delivery and falls back to STREAM
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from jules_daemon.agent.ipc_bridge import (
    make_ask_callback,
    make_confirm_callback,
    make_notify_callback,
)
from jules_daemon.ipc.framing import (
    unpack_header,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
)
from jules_daemon.ipc.notification_broadcaster import NotificationBroadcaster
from jules_daemon.ipc.server import ClientConnection
from jules_daemon.protocol.notifications import NotificationEventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> ClientConnection:
    """Build a stub ClientConnection for testing."""
    return ClientConnection(
        client_id="test-client-001",
        reader=AsyncMock(spec=asyncio.StreamReader),
        writer=AsyncMock(spec=asyncio.StreamWriter),
        connected_at="2026-04-12T12:00:00Z",
    )


def _setup_reply(
    client: ClientConnection,
    payload: dict[str, Any],
    msg_type: MessageType = MessageType.CONFIRM_REPLY,
) -> None:
    """Configure the mock reader to return a specific reply."""
    reply = MessageEnvelope(
        msg_type=msg_type,
        msg_id="reply-001",
        timestamp="2026-04-12T12:00:01Z",
        payload=payload,
    )
    frame = encode_frame(reply)
    header_bytes = frame[:4]
    payload_bytes = frame[4:]
    client.reader.readexactly = AsyncMock(
        side_effect=[header_bytes, payload_bytes]
    )


# ---------------------------------------------------------------------------
# make_confirm_callback
# ---------------------------------------------------------------------------


class TestMakeConfirmCallback:
    """Tests for the confirm callback factory."""

    @pytest.mark.asyncio
    async def test_approved_returns_true_and_command(self) -> None:
        client = _make_client()
        _setup_reply(client, {"approved": True})

        callback = make_confirm_callback(client)
        approved, final_cmd = await callback(
            "pytest -v", "staging.example.com", "Running tests",
        )

        assert approved is True
        assert final_cmd == "pytest -v"

    @pytest.mark.asyncio
    async def test_denied_returns_false(self) -> None:
        client = _make_client()
        _setup_reply(client, {"approved": False})

        callback = make_confirm_callback(client)
        approved, final_cmd = await callback(
            "pytest -v", "staging.example.com", "",
        )

        assert approved is False

    @pytest.mark.asyncio
    async def test_edited_command_returned(self) -> None:
        client = _make_client()
        _setup_reply(client, {
            "approved": True,
            "edited_command": "pytest -v --timeout=60",
        })

        callback = make_confirm_callback(client)
        approved, final_cmd = await callback(
            "pytest -v", "staging.example.com", "",
        )

        assert approved is True
        assert final_cmd == "pytest -v --timeout=60"

    @pytest.mark.asyncio
    async def test_disconnect_returns_false(self) -> None:
        """If the client disconnects, returns (False, original_command)."""
        client = _make_client()
        client.reader.readexactly = AsyncMock(
            side_effect=asyncio.IncompleteReadError(b"", 4)
        )

        callback = make_confirm_callback(client)
        approved, final_cmd = await callback(
            "pytest -v", "staging.example.com", "",
        )

        assert approved is False
        assert final_cmd == "pytest -v"

    @pytest.mark.asyncio
    async def test_sends_confirm_prompt(self) -> None:
        """Verify the callback sends a CONFIRM_PROMPT envelope."""
        client = _make_client()
        _setup_reply(client, {"approved": True})

        callback = make_confirm_callback(client)
        await callback("pytest -v", "staging.example.com", "Test run")

        # Verify write was called (the frame was sent)
        assert client.writer.write.called
        assert client.writer.drain.called

    @pytest.mark.asyncio
    async def test_includes_target_context_metadata(self) -> None:
        client = _make_client()
        _setup_reply(client, {"approved": True})

        callback = make_confirm_callback(
            client,
            target_context={
                "target_host": "10.0.0.10",
                "target_user": "root",
                "target_port": 22,
                "resolved_system_name": "tuto",
                "resolved_system_hostname": "tuto.internal.example",
                "resolved_system_ip_address": "10.0.0.10",
                "resolved_system_description": "Tutorial box",
            },
        )
        await callback("pytest -v", "10.0.0.10", "Test run")

        written = b"".join(call.args[0] for call in client.writer.write.call_args_list)
        payload_length = unpack_header(written[:4])
        prompt = decode_envelope(written[4:4 + payload_length])

        assert prompt.msg_type == MessageType.CONFIRM_PROMPT
        assert prompt.payload["system_name"] == "tuto"
        assert prompt.payload["system_hostname"] == "tuto.internal.example"
        assert prompt.payload["system_ip_address"] == "10.0.0.10"
        assert prompt.payload["target_user"] == "root"
        assert prompt.payload["target_port"] == 22


# ---------------------------------------------------------------------------
# make_ask_callback
# ---------------------------------------------------------------------------


class TestMakeAskCallback:
    """Tests for the ask callback factory."""

    @pytest.mark.asyncio
    async def test_returns_answer(self) -> None:
        client = _make_client()
        _setup_reply(client, {"answer": "100"})

        callback = make_ask_callback(client)
        answer = await callback("How many iterations?", "Test needs count")

        assert answer == "100"

    @pytest.mark.asyncio
    async def test_returns_text_fallback(self) -> None:
        """Falls back to 'text' key when 'answer' is missing."""
        client = _make_client()
        _setup_reply(client, {"text": "use defaults"})

        callback = make_ask_callback(client)
        answer = await callback("What config?", "")

        assert answer == "use defaults"

    @pytest.mark.asyncio
    async def test_cancelled_returns_none(self) -> None:
        client = _make_client()
        _setup_reply(client, {"cancelled": True})

        callback = make_ask_callback(client)
        answer = await callback("Question?", "")

        assert answer is None

    @pytest.mark.asyncio
    async def test_not_approved_returns_none(self) -> None:
        """When approved=False, returns None (cancellation)."""
        client = _make_client()
        _setup_reply(client, {"approved": False})

        callback = make_ask_callback(client)
        answer = await callback("Question?", "")

        assert answer is None

    @pytest.mark.asyncio
    async def test_disconnect_returns_none(self) -> None:
        client = _make_client()
        client.reader.readexactly = AsyncMock(
            side_effect=asyncio.IncompleteReadError(b"", 4)
        )

        callback = make_ask_callback(client)
        answer = await callback("Question?", "")

        assert answer is None


# ---------------------------------------------------------------------------
# make_notify_callback
# ---------------------------------------------------------------------------


class TestMakeNotifyCallback:
    """Tests for the notify callback factory."""

    @pytest.mark.asyncio
    async def test_prefers_broadcaster_delivery(self) -> None:
        client = _make_client()
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe(
            event_filter=frozenset({NotificationEventType.ALERT}),
        )

        callback = make_notify_callback(
            client,
            notification_broadcaster=broadcaster,
        )
        result = await callback("Test completed", "success")

        assert result is True
        assert client.writer.write.call_count == 0

        notification = await broadcaster.receive(
            handle.subscription_id,
            timeout=0.1,
        )
        assert notification is not None
        assert notification.event_type is NotificationEventType.ALERT
        assert notification.payload.message == "Test completed"

        await broadcaster.unsubscribe(handle.subscription_id)

    @pytest.mark.asyncio
    async def test_sends_stream_message(self) -> None:
        client = _make_client()

        callback = make_notify_callback(client)
        result = await callback("Test completed", "info")

        assert result is True
        assert client.writer.write.called

    @pytest.mark.asyncio
    async def test_failure_returns_false(self) -> None:
        """On send failure, returns False without raising."""
        client = _make_client()
        client.writer.drain = AsyncMock(
            side_effect=OSError("broken pipe")
        )

        callback = make_notify_callback(client)
        result = await callback("Test completed", "info")

        assert result is False

    @pytest.mark.asyncio
    async def test_default_severity_is_info(self) -> None:
        client = _make_client()

        callback = make_notify_callback(client)
        result = await callback("Progress update")

        assert result is True
