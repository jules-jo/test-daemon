"""Integration tests for the thin client with a mock IPC server.

Spins up a real asyncio Unix domain socket server that speaks the
Jules daemon IPC protocol (length-prefixed JSON framing), then
exercises the thin client against it. This proves:

    1. The thin client can connect via a real Unix socket
    2. The framing protocol (4-byte header + JSON payload) works end-to-end
    3. The handshake exchange completes successfully
    4. Request-response cycles work for all verbs
    5. The confirmation flow (CONFIRM_PROMPT -> CONFIRM_REPLY) works
    6. Streaming output (STREAM envelopes) works
    7. The client handles server disconnects gracefully

These tests use temporary socket files and short timeouts to avoid
interfering with a real daemon or causing test hangs.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jules_daemon.ipc.client_connection import PROTOCOL_VERSION
from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
    unpack_header,
)
from jules_daemon.thin_client.client import ThinClient, ThinClientConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = "2026-04-09T12:00:00Z"


def _envelope(
    msg_type: MessageType,
    msg_id: str,
    payload: dict,
) -> MessageEnvelope:
    """Build a test envelope."""
    return MessageEnvelope(
        msg_type=msg_type,
        msg_id=msg_id,
        timestamp=_TS,
        payload=payload,
    )


async def _read_one_envelope(reader: asyncio.StreamReader) -> MessageEnvelope:
    """Read one framed envelope from a stream."""
    header = await reader.readexactly(HEADER_SIZE)
    length = unpack_header(header)
    payload_bytes = await reader.readexactly(length)
    return decode_envelope(payload_bytes)


async def _send_envelope(
    writer: asyncio.StreamWriter,
    envelope: MessageEnvelope,
) -> None:
    """Write one framed envelope to a stream."""
    writer.write(encode_frame(envelope))
    await writer.drain()


# ---------------------------------------------------------------------------
# Mock daemon server
# ---------------------------------------------------------------------------


class MockDaemonServer:
    """A minimal mock daemon that speaks the Jules IPC protocol.

    Handles:
        - Handshake (validates protocol version, returns daemon metadata)
        - Verb-specific responses via a configurable handler

    Args:
        socket_path: Path for the Unix domain socket.
        verb_handler: Async callable that receives (verb, request_envelope)
            and returns a list of response envelopes to send back.
    """

    def __init__(
        self,
        socket_path: Path,
        verb_handler=None,
    ) -> None:
        self._socket_path = socket_path
        self._verb_handler = verb_handler or self._default_handler
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        """Start listening on the Unix socket."""
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._socket_path),
        )

    async def stop(self) -> None:
        """Stop the server and clean up."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        # Clean up socket file
        try:
            self._socket_path.unlink(missing_ok=True)
        except OSError:
            pass

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle one client connection."""
        try:
            # Step 1: Handshake
            request = await _read_one_envelope(reader)

            if request.payload.get("verb") == "handshake":
                handshake_response = _envelope(
                    MessageType.RESPONSE,
                    request.msg_id,
                    {
                        "verb": "handshake",
                        "status": "ok",
                        "protocol_version": request.payload.get(
                            "protocol_version", PROTOCOL_VERSION
                        ),
                        "daemon_pid": os.getpid(),
                        "daemon_uptime_seconds": 42.0,
                    },
                )
                await _send_envelope(writer, handshake_response)

                # Step 2: Read verb request
                try:
                    verb_request = await asyncio.wait_for(
                        _read_one_envelope(reader),
                        timeout=5.0,
                    )
                except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                    return

                verb = verb_request.payload.get("verb", "unknown")
                responses = await self._verb_handler(verb, verb_request)

                for resp in responses:
                    await _send_envelope(writer, resp)

                # Check if there's a confirm reply coming back
                if any(r.msg_type == MessageType.CONFIRM_PROMPT for r in responses):
                    try:
                        reply = await asyncio.wait_for(
                            _read_one_envelope(reader),
                            timeout=5.0,
                        )
                        # Send final response after confirm reply
                        if reply.msg_type == MessageType.CONFIRM_REPLY:
                            approved = reply.payload.get("approved", False)
                            if approved:
                                final = _envelope(
                                    MessageType.RESPONSE,
                                    "daemon-final",
                                    {"verb": "run", "status": "ok", "run_id": "run-confirmed"},
                                )
                            else:
                                final = _envelope(
                                    MessageType.ERROR,
                                    "daemon-denied",
                                    {"verb": "run", "error": "Denied by user", "status_code": 403},
                                )
                            await _send_envelope(writer, final)
                    except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                        pass

        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, ConnectionResetError):
                pass

    @staticmethod
    async def _default_handler(
        verb: str,
        request: MessageEnvelope,
    ) -> list[MessageEnvelope]:
        """Default handler: echo back the verb with status ok."""
        return [
            _envelope(
                MessageType.RESPONSE,
                f"resp-{request.msg_id}",
                {"verb": verb, "status": "ok"},
            )
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def socket_path(tmp_path):
    """Provide a temporary socket path."""
    return tmp_path / "test-daemon.sock"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestThinClientIntegrationHealth:
    """Integration: health check with real Unix socket."""

    @pytest.mark.asyncio
    async def test_health_check(self, socket_path):
        server = MockDaemonServer(socket_path)
        await server.start()

        try:
            config = ThinClientConfig(
                socket_path=socket_path,
                connect_timeout=2.0,
                receive_timeout=5.0,
            )
            client = ThinClient(config=config)
            result = await client.health()

            assert result.success is True
            assert result.verb == "health"
            assert result.response is not None
            assert result.response.payload["verb"] == "health"
        finally:
            await server.stop()


class TestThinClientIntegrationStatus:
    """Integration: status query with real Unix socket."""

    @pytest.mark.asyncio
    async def test_status_query(self, socket_path):
        async def handler(verb, req):
            return [
                _envelope(
                    MessageType.RESPONSE,
                    f"resp-{req.msg_id}",
                    {
                        "verb": "status",
                        "status": "ok",
                        "run_state": "running",
                        "progress": 65.0,
                    },
                )
            ]

        server = MockDaemonServer(socket_path, verb_handler=handler)
        await server.start()

        try:
            config = ThinClientConfig(
                socket_path=socket_path,
                connect_timeout=2.0,
                receive_timeout=5.0,
            )
            client = ThinClient(config=config)
            result = await client.status()

            assert result.success is True
            assert result.response is not None
            assert result.response.payload["run_state"] == "running"
            assert result.response.payload["progress"] == 65.0
        finally:
            await server.stop()


class TestThinClientIntegrationHistory:
    """Integration: history query with real Unix socket."""

    @pytest.mark.asyncio
    async def test_history_query(self, socket_path):
        async def handler(verb, req):
            limit = req.payload.get("limit", 20)
            return [
                _envelope(
                    MessageType.RESPONSE,
                    f"resp-{req.msg_id}",
                    {
                        "verb": "history",
                        "status": "ok",
                        "runs": [
                            {"run_id": f"run-{i}", "status": "completed"}
                            for i in range(min(limit, 3))
                        ],
                    },
                )
            ]

        server = MockDaemonServer(socket_path, verb_handler=handler)
        await server.start()

        try:
            config = ThinClientConfig(
                socket_path=socket_path,
                connect_timeout=2.0,
                receive_timeout=5.0,
            )
            client = ThinClient(config=config)
            result = await client.history(limit=3)

            assert result.success is True
            assert result.response is not None
            runs = result.response.payload["runs"]
            assert len(runs) == 3
        finally:
            await server.stop()


class TestThinClientIntegrationCancel:
    """Integration: cancel command with real Unix socket."""

    @pytest.mark.asyncio
    async def test_cancel_command(self, socket_path):
        async def handler(verb, req):
            return [
                _envelope(
                    MessageType.RESPONSE,
                    f"resp-{req.msg_id}",
                    {
                        "verb": "cancel",
                        "status": "ok",
                        "cancelled_run": "run-active-001",
                    },
                )
            ]

        server = MockDaemonServer(socket_path, verb_handler=handler)
        await server.start()

        try:
            config = ThinClientConfig(
                socket_path=socket_path,
                connect_timeout=2.0,
                receive_timeout=5.0,
            )
            client = ThinClient(config=config)
            result = await client.cancel(reason="Tests are stuck")

            assert result.success is True
            assert result.response is not None
            assert result.response.payload["cancelled_run"] == "run-active-001"
        finally:
            await server.stop()


class TestThinClientIntegrationRun:
    """Integration: run command with confirmation flow."""

    @pytest.mark.asyncio
    async def test_run_with_approval(self, socket_path):
        """Full confirmation cycle: request -> prompt -> reply -> response."""

        async def handler(verb, req):
            # Return a confirmation prompt
            return [
                _envelope(
                    MessageType.CONFIRM_PROMPT,
                    "daemon-prompt-001",
                    {
                        "verb": "confirm",
                        "command": "cd /app && pytest -v",
                        "target_host": req.payload.get("target_host", "unknown"),
                        "target_user": req.payload.get("target_user", "unknown"),
                        "risk_level": "LOW",
                        "explanation": "Runs unit tests",
                    },
                )
            ]

        server = MockDaemonServer(socket_path, verb_handler=handler)
        await server.start()

        try:
            config = ThinClientConfig(
                socket_path=socket_path,
                connect_timeout=2.0,
                receive_timeout=5.0,
            )

            approval_called = MagicMock(return_value=True)
            client = ThinClient(config=config, on_confirm=approval_called)

            result = await client.run(
                target_host="ci.example.com",
                target_user="deploy",
                natural_language="run unit tests",
            )

            assert result.success is True
            approval_called.assert_called_once()
            # Verify the prompt envelope was passed to callback
            prompt_env = approval_called.call_args[0][0]
            assert prompt_env.msg_type == MessageType.CONFIRM_PROMPT
            assert prompt_env.payload["command"] == "cd /app && pytest -v"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_run_with_denial(self, socket_path):
        """Denial flow: request -> prompt -> deny reply -> error response."""

        async def handler(verb, req):
            return [
                _envelope(
                    MessageType.CONFIRM_PROMPT,
                    "daemon-prompt-002",
                    {
                        "verb": "confirm",
                        "command": "rm -rf /",
                        "target_host": "prod",
                        "target_user": "root",
                        "risk_level": "CRITICAL",
                    },
                )
            ]

        server = MockDaemonServer(socket_path, verb_handler=handler)
        await server.start()

        try:
            config = ThinClientConfig(
                socket_path=socket_path,
                connect_timeout=2.0,
                receive_timeout=5.0,
            )

            deny_callback = MagicMock(return_value=False)
            client = ThinClient(config=config, on_confirm=deny_callback)

            result = await client.run(
                target_host="prod",
                target_user="root",
                natural_language="clean up disk",
            )

            assert result.success is False
            deny_callback.assert_called_once()
        finally:
            await server.stop()


class TestThinClientIntegrationWatch:
    """Integration: watch command with streaming output."""

    @pytest.mark.asyncio
    async def test_watch_streaming(self, socket_path):
        """Receive multiple stream lines followed by end-of-stream."""

        async def handler(verb, req):
            responses = [
                # Subscription response
                _envelope(
                    MessageType.RESPONSE,
                    "sub-resp-001",
                    {
                        "verb": "watch",
                        "status": "ok",
                        "subscriber_id": "sub-001",
                        "job_id": "run-active",
                    },
                ),
                # Stream lines
                _envelope(
                    MessageType.STREAM,
                    "stream-001",
                    {"line": "test_auth.py::test_login PASSED", "sequence": 1, "is_end": False},
                ),
                _envelope(
                    MessageType.STREAM,
                    "stream-002",
                    {"line": "test_auth.py::test_logout PASSED", "sequence": 2, "is_end": False},
                ),
                _envelope(
                    MessageType.STREAM,
                    "stream-003",
                    {"line": "2 passed in 1.5s", "sequence": 3, "is_end": False},
                ),
                # End of stream
                _envelope(
                    MessageType.STREAM,
                    "stream-end",
                    {"line": "", "sequence": 4, "is_end": True},
                ),
            ]
            return responses

        server = MockDaemonServer(socket_path, verb_handler=handler)
        await server.start()

        try:
            config = ThinClientConfig(
                socket_path=socket_path,
                connect_timeout=2.0,
                receive_timeout=5.0,
                stream_timeout=5.0,
            )

            received_lines: list[str] = []
            client = ThinClient(config=config)
            result = await client.watch(on_line=received_lines.append)

            assert result.success is True
            assert len(received_lines) == 3
            assert "test_login PASSED" in received_lines[0]
            assert "test_logout PASSED" in received_lines[1]
            assert "2 passed" in received_lines[2]
        finally:
            await server.stop()


class TestThinClientIntegrationServerDown:
    """Integration: behavior when daemon is not running."""

    @pytest.mark.asyncio
    async def test_connection_to_nonexistent_socket(self, socket_path):
        """Graceful failure when no daemon is listening."""
        config = ThinClientConfig(
            socket_path=socket_path,
            connect_timeout=1.0,
            receive_timeout=2.0,
        )
        client = ThinClient(config=config)
        result = await client.health()

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_all_verbs_fail_gracefully(self, socket_path):
        """Every verb produces a clean error when daemon is down."""
        config = ThinClientConfig(
            socket_path=socket_path,
            connect_timeout=1.0,
            receive_timeout=2.0,
        )
        client = ThinClient(config=config)

        # Test all verbs
        results = [
            await client.health(),
            await client.status(),
            await client.history(),
            await client.cancel(),
            await client.run(
                target_host="ci.example.com",
                target_user="deploy",
                natural_language="run tests",
            ),
            await client.watch(),
        ]

        for result in results:
            assert result.success is False
            assert result.error is not None
            # No exceptions should propagate
