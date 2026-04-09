"""End-to-end integration tests for the Unix socket server with request handler.

Validates the complete IPC flow: CLI client connects to the Unix socket,
sends a JSON request, the server deserializes it, calls the validation
layer via the RequestHandler, and returns a JSON response with either
an enqueue confirmation or validation errors.

This exercises the full chain:
    Client -> Unix socket -> SocketServer -> RequestHandler
                                              -> validate_request()
                                              -> CommandQueue (for queue verb)
                                            -> framed JSON response -> Client
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
    unpack_header,
)
from jules_daemon.ipc.request_handler import (
    RequestHandler,
    RequestHandlerConfig,
)
from jules_daemon.ipc.server import ServerConfig, ServerState, SocketServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_socket_path(tmp_path: Path) -> Path:
    """Return a unique socket path inside the given temp directory."""
    return tmp_path / "integration.sock"


def _build_request(
    verb: str,
    msg_id: str = "int-001",
    **extra: object,
) -> MessageEnvelope:
    """Build a REQUEST envelope with the given verb and extra payload fields."""
    payload = {"verb": verb, **extra}
    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=msg_id,
        timestamp="2026-04-09T12:00:00Z",
        payload=payload,
    )


async def _send_request(
    socket_path: Path,
    envelope: MessageEnvelope,
) -> MessageEnvelope:
    """Connect to the server, send a request, read back the response."""
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        frame = encode_frame(envelope)
        writer.write(frame)
        await writer.drain()

        header_bytes = await reader.readexactly(HEADER_SIZE)
        payload_length = unpack_header(header_bytes)
        payload_bytes = await reader.readexactly(payload_length)
        return decode_envelope(payload_bytes)
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Integration: status verb end-to-end
# ---------------------------------------------------------------------------


class TestIntegrationStatus:
    """End-to-end tests for the status verb through the full IPC stack."""

    @pytest.mark.asyncio
    async def test_status_request_returns_response(
        self, tmp_path: Path
    ) -> None:
        handler_config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=handler_config)
        sock_path = _make_socket_path(tmp_path)
        server_config = ServerConfig(socket_path=sock_path)

        async with SocketServer(config=server_config, handler=handler):
            request = _build_request("status")
            response = await _send_request(sock_path, request)

            assert response.msg_type == MessageType.RESPONSE
            assert response.msg_id == "int-001"
            assert response.payload["verb"] == "status"
            assert "state" in response.payload


# ---------------------------------------------------------------------------
# Integration: queue verb end-to-end with enqueue confirmation
# ---------------------------------------------------------------------------


class TestIntegrationQueue:
    """End-to-end tests for the queue verb with wiki-backed enqueuing."""

    @pytest.mark.asyncio
    async def test_queue_returns_enqueue_confirmation(
        self, tmp_path: Path
    ) -> None:
        handler_config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=handler_config)
        sock_path = _make_socket_path(tmp_path)
        server_config = ServerConfig(socket_path=sock_path)

        async with SocketServer(config=server_config, handler=handler):
            request = _build_request(
                "queue",
                target_host="staging.example.com",
                target_user="deploy",
                natural_language="run the smoke tests",
            )
            response = await _send_request(sock_path, request)

            assert response.msg_type == MessageType.RESPONSE
            assert response.payload["verb"] == "queue"
            assert response.payload["status"] == "enqueued"
            assert "queue_id" in response.payload
            assert response.payload["position"] >= 1

    @pytest.mark.asyncio
    async def test_queue_persists_to_wiki(self, tmp_path: Path) -> None:
        handler_config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=handler_config)
        sock_path = _make_socket_path(tmp_path)
        server_config = ServerConfig(socket_path=sock_path)

        async with SocketServer(config=server_config, handler=handler):
            request = _build_request(
                "queue",
                target_host="staging.example.com",
                target_user="deploy",
                natural_language="run integration tests",
            )
            await _send_request(sock_path, request)

        # After server shutdown, verify wiki file was created
        queue_dir = tmp_path / "pages" / "daemon" / "queue"
        assert queue_dir.exists()
        queue_files = list(queue_dir.glob("*.md"))
        assert len(queue_files) == 1

        # Verify the wiki file has proper YAML frontmatter
        content = queue_files[0].read_text(encoding="utf-8")
        assert "---" in content
        assert "natural_language:" in content
        assert "run integration tests" in content

    @pytest.mark.asyncio
    async def test_multiple_queued_commands(self, tmp_path: Path) -> None:
        handler_config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=handler_config)
        sock_path = _make_socket_path(tmp_path)
        server_config = ServerConfig(socket_path=sock_path)

        async with SocketServer(config=server_config, handler=handler):
            positions = []
            for i in range(3):
                request = _build_request(
                    "queue",
                    msg_id=f"q-{i}",
                    target_host=f"host-{i}.example.com",
                    target_user="deploy",
                    natural_language=f"run test suite {i}",
                )
                response = await _send_request(sock_path, request)
                positions.append(response.payload["position"])

            # Positions should increment
            assert positions == [1, 2, 3]


# ---------------------------------------------------------------------------
# Integration: validation error end-to-end
# ---------------------------------------------------------------------------


class TestIntegrationValidationErrors:
    """End-to-end tests for validation error handling through the IPC stack."""

    @pytest.mark.asyncio
    async def test_missing_verb_returns_structured_error(
        self, tmp_path: Path
    ) -> None:
        handler_config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=handler_config)
        sock_path = _make_socket_path(tmp_path)
        server_config = ServerConfig(socket_path=sock_path)

        async with SocketServer(config=server_config, handler=handler):
            request = MessageEnvelope(
                msg_type=MessageType.REQUEST,
                msg_id="err-001",
                timestamp="2026-04-09T12:00:00Z",
                payload={},  # missing verb
            )
            response = await _send_request(sock_path, request)

            assert response.msg_type == MessageType.ERROR
            assert response.msg_id == "err-001"
            assert "validation_errors" in response.payload
            errors = response.payload["validation_errors"]
            assert len(errors) >= 1
            assert any(e["field"] == "verb" for e in errors)

    @pytest.mark.asyncio
    async def test_unknown_verb_returns_error(self, tmp_path: Path) -> None:
        handler_config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=handler_config)
        sock_path = _make_socket_path(tmp_path)
        server_config = ServerConfig(socket_path=sock_path)

        async with SocketServer(config=server_config, handler=handler):
            request = _build_request("teleport", msg_id="err-002")
            response = await _send_request(sock_path, request)

            assert response.msg_type == MessageType.ERROR
            errors = response.payload["validation_errors"]
            assert any(e["code"] == "unknown_verb" for e in errors)

    @pytest.mark.asyncio
    async def test_run_missing_fields_returns_all_errors(
        self, tmp_path: Path
    ) -> None:
        handler_config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=handler_config)
        sock_path = _make_socket_path(tmp_path)
        server_config = ServerConfig(socket_path=sock_path)

        async with SocketServer(config=server_config, handler=handler):
            request = _build_request("run", msg_id="err-003")
            response = await _send_request(sock_path, request)

            assert response.msg_type == MessageType.ERROR
            errors = response.payload["validation_errors"]
            field_names = {e["field"] for e in errors}
            assert "target_host" in field_names
            assert "target_user" in field_names
            assert "natural_language" in field_names


# ---------------------------------------------------------------------------
# Integration: run verb end-to-end
# ---------------------------------------------------------------------------


class TestIntegrationRun:
    """End-to-end tests for the run verb through the full IPC stack."""

    @pytest.mark.asyncio
    async def test_run_accepted(self, tmp_path: Path) -> None:
        handler_config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=handler_config)
        sock_path = _make_socket_path(tmp_path)
        server_config = ServerConfig(socket_path=sock_path)

        async with SocketServer(config=server_config, handler=handler):
            request = _build_request(
                "run",
                target_host="staging.example.com",
                target_user="deploy",
                natural_language="run the full regression suite",
            )
            response = await _send_request(sock_path, request)

            assert response.msg_type == MessageType.RESPONSE
            assert response.payload["verb"] == "run"
            assert response.payload["status"] == "accepted"


# ---------------------------------------------------------------------------
# Integration: concurrent requests
# ---------------------------------------------------------------------------


class TestIntegrationConcurrent:
    """End-to-end tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_mixed_requests(self, tmp_path: Path) -> None:
        handler_config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=handler_config)
        sock_path = _make_socket_path(tmp_path)
        server_config = ServerConfig(socket_path=sock_path)

        async with SocketServer(config=server_config, handler=handler):
            # Mix of valid and invalid requests
            tasks = [
                _send_request(
                    sock_path,
                    _build_request("status", msg_id="c-0"),
                ),
                _send_request(
                    sock_path,
                    _build_request("teleport", msg_id="c-1"),
                ),
                _send_request(
                    sock_path,
                    _build_request(
                        "queue",
                        msg_id="c-2",
                        target_host="host.example.com",
                        target_user="deploy",
                        natural_language="run tests",
                    ),
                ),
            ]
            responses = await asyncio.gather(*tasks)

            # status: success
            assert responses[0].msg_type == MessageType.RESPONSE
            assert responses[0].payload["verb"] == "status"

            # teleport: error
            assert responses[1].msg_type == MessageType.ERROR

            # queue: success
            assert responses[2].msg_type == MessageType.RESPONSE
            assert responses[2].payload["status"] == "enqueued"


# ---------------------------------------------------------------------------
# Integration: server resilience after errors
# ---------------------------------------------------------------------------


class TestIntegrationResilience:
    """Tests that the server remains healthy after validation errors."""

    @pytest.mark.asyncio
    async def test_error_then_success(self, tmp_path: Path) -> None:
        """Server handles error request, then processes valid request."""
        handler_config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=handler_config)
        sock_path = _make_socket_path(tmp_path)
        server_config = ServerConfig(socket_path=sock_path)

        async with SocketServer(
            config=server_config, handler=handler
        ) as server:
            # First: invalid request
            r1 = await _send_request(
                sock_path,
                _build_request("explode", msg_id="r-1"),
            )
            assert r1.msg_type == MessageType.ERROR

            # Server still running
            assert server.state == ServerState.RUNNING

            # Second: valid request
            r2 = await _send_request(
                sock_path,
                _build_request("status", msg_id="r-2"),
            )
            assert r2.msg_type == MessageType.RESPONSE
            assert r2.payload["verb"] == "status"
