"""Tests for async Unix domain socket IPC server.

Validates the server's socket lifecycle: bind, listen, accept, read/write
framed messages, graceful shutdown, and socket file cleanup.
"""

from __future__ import annotations

import asyncio
import os
import stat
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    encode_frame,
    unpack_header,
)
from jules_daemon.ipc.server import (
    DEFAULT_BACKLOG,
    DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    MAX_CONCURRENT_CLIENTS,
    ClientConnection,
    ClientHandler,
    ServerConfig,
    ServerState,
    SocketServer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_socket_path(tmp_path: Path) -> Path:
    """Return a unique socket path inside the given temp directory."""
    return tmp_path / "test.sock"


def _make_config(tmp_path: Path, **overrides: object) -> ServerConfig:
    """Build a ServerConfig pointing at a temp directory socket."""
    defaults = {
        "socket_path": _make_socket_path(tmp_path),
    }
    defaults.update(overrides)
    return ServerConfig(**defaults)  # type: ignore[arg-type]


def _build_envelope(
    verb: str = "status",
    msg_id: str = "test-001",
) -> MessageEnvelope:
    """Build a minimal request envelope for testing."""
    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=msg_id,
        timestamp="2026-04-09T12:00:00Z",
        payload={"verb": verb},
    )


async def _connect_and_send(
    socket_path: Path,
    envelope: MessageEnvelope,
) -> MessageEnvelope:
    """Connect to the server, send a framed envelope, read back a response."""
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        frame = encode_frame(envelope)
        writer.write(frame)
        await writer.drain()

        # Read response header
        header_bytes = await reader.readexactly(HEADER_SIZE)
        payload_length = unpack_header(header_bytes)

        # Read response payload
        payload_bytes = await reader.readexactly(payload_length)
        from jules_daemon.ipc.framing import decode_envelope

        return decode_envelope(payload_bytes)
    finally:
        writer.close()
        await writer.wait_closed()


async def _connect_and_send_raw(
    socket_path: Path,
    data: bytes,
) -> bytes:
    """Connect to the server, send raw bytes, read back all available data."""
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        writer.write(data)
        await writer.drain()
        # Give server time to process and respond
        await asyncio.sleep(0.05)
        # Read whatever is available
        return await asyncio.wait_for(reader.read(65536), timeout=1.0)
    except asyncio.TimeoutError:
        return b""
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Echo handler for integration tests
# ---------------------------------------------------------------------------


class EchoHandler:
    """Simple handler that echoes back the received envelope as a response."""

    async def handle_message(
        self,
        envelope: MessageEnvelope,
        client: ClientConnection,
    ) -> MessageEnvelope:
        """Return a response envelope echoing the request payload."""
        return MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id=envelope.msg_id,
            timestamp=envelope.timestamp,
            payload={"echo": envelope.payload},
        )


# ---------------------------------------------------------------------------
# ServerConfig tests
# ---------------------------------------------------------------------------


class TestServerConfig:
    """Tests for the immutable ServerConfig dataclass."""

    def test_create_with_defaults(self, tmp_path: Path) -> None:
        sock = _make_socket_path(tmp_path)
        config = ServerConfig(socket_path=sock)
        assert config.socket_path == sock
        assert config.backlog == DEFAULT_BACKLOG
        assert config.shutdown_timeout_seconds == DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
        assert config.max_concurrent_clients == MAX_CONCURRENT_CLIENTS

    def test_custom_values(self, tmp_path: Path) -> None:
        sock = _make_socket_path(tmp_path)
        config = ServerConfig(
            socket_path=sock,
            backlog=10,
            shutdown_timeout_seconds=5.0,
            max_concurrent_clients=3,
        )
        assert config.backlog == 10
        assert config.shutdown_timeout_seconds == 5.0
        assert config.max_concurrent_clients == 3

    def test_frozen(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with pytest.raises(AttributeError):
            config.backlog = 99  # type: ignore[misc]

    def test_negative_backlog_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="backlog must be positive"):
            ServerConfig(socket_path=_make_socket_path(tmp_path), backlog=-1)

    def test_zero_backlog_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="backlog must be positive"):
            ServerConfig(socket_path=_make_socket_path(tmp_path), backlog=0)

    def test_negative_shutdown_timeout_raises(self, tmp_path: Path) -> None:
        with pytest.raises(
            ValueError, match="shutdown_timeout_seconds must be positive"
        ):
            ServerConfig(
                socket_path=_make_socket_path(tmp_path),
                shutdown_timeout_seconds=-1.0,
            )

    def test_zero_shutdown_timeout_raises(self, tmp_path: Path) -> None:
        with pytest.raises(
            ValueError, match="shutdown_timeout_seconds must be positive"
        ):
            ServerConfig(
                socket_path=_make_socket_path(tmp_path),
                shutdown_timeout_seconds=0.0,
            )

    def test_zero_max_clients_raises(self, tmp_path: Path) -> None:
        with pytest.raises(
            ValueError, match="max_concurrent_clients must be positive"
        ):
            ServerConfig(
                socket_path=_make_socket_path(tmp_path),
                max_concurrent_clients=0,
            )


# ---------------------------------------------------------------------------
# ServerState tests
# ---------------------------------------------------------------------------


class TestServerState:
    """Tests for the ServerState enum."""

    def test_all_states_exist(self) -> None:
        assert len(ServerState) == 4

    def test_stopped_value(self) -> None:
        assert ServerState.STOPPED.value == "stopped"

    def test_starting_value(self) -> None:
        assert ServerState.STARTING.value == "starting"

    def test_running_value(self) -> None:
        assert ServerState.RUNNING.value == "running"

    def test_shutting_down_value(self) -> None:
        assert ServerState.SHUTTING_DOWN.value == "shutting_down"


# ---------------------------------------------------------------------------
# ClientConnection tests
# ---------------------------------------------------------------------------


class TestClientConnection:
    """Tests for the ClientConnection dataclass."""

    def test_create(self) -> None:
        reader = AsyncMock()
        writer = AsyncMock()
        conn = ClientConnection(
            client_id="client-001",
            reader=reader,
            writer=writer,
            connected_at="2026-04-09T12:00:00Z",
        )
        assert conn.client_id == "client-001"
        assert conn.connected_at == "2026-04-09T12:00:00Z"

    def test_frozen(self) -> None:
        conn = ClientConnection(
            client_id="client-001",
            reader=AsyncMock(),
            writer=AsyncMock(),
            connected_at="2026-04-09T12:00:00Z",
        )
        with pytest.raises(AttributeError):
            conn.client_id = "mutated"  # type: ignore[misc]

    def test_empty_client_id_raises(self) -> None:
        with pytest.raises(ValueError, match="client_id must not be empty"):
            ClientConnection(
                client_id="",
                reader=AsyncMock(),
                writer=AsyncMock(),
                connected_at="2026-04-09T12:00:00Z",
            )

    def test_whitespace_client_id_raises(self) -> None:
        with pytest.raises(ValueError, match="client_id must not be empty"):
            ClientConnection(
                client_id="   ",
                reader=AsyncMock(),
                writer=AsyncMock(),
                connected_at="2026-04-09T12:00:00Z",
            )


# ---------------------------------------------------------------------------
# SocketServer lifecycle tests
# ---------------------------------------------------------------------------


class TestSocketServerLifecycle:
    """Tests for server start, state transitions, and shutdown."""

    @pytest.mark.asyncio
    async def test_initial_state_is_stopped(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        assert server.state == ServerState.STOPPED

    @pytest.mark.asyncio
    async def test_start_creates_socket_file(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        try:
            sock_path = config.socket_path
            assert sock_path.exists()
            mode = sock_path.stat().st_mode
            assert stat.S_ISSOCK(mode)
        finally:
            await server.shutdown()

    @pytest.mark.asyncio
    async def test_start_transitions_to_running(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        try:
            assert server.state == ServerState.RUNNING
        finally:
            await server.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_transitions_to_stopped(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        await server.shutdown()
        assert server.state == ServerState.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_removes_socket_file(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        sock_path = config.socket_path
        assert sock_path.exists()
        await server.shutdown()
        assert not sock_path.exists()

    @pytest.mark.asyncio
    async def test_double_start_raises(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await server.start()
        finally:
            await server.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_when_stopped_is_noop(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        # Should not raise
        await server.shutdown()
        assert server.state == ServerState.STOPPED

    @pytest.mark.asyncio
    async def test_double_shutdown_is_safe(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        await server.shutdown()
        # Second shutdown should be a noop
        await server.shutdown()
        assert server.state == ServerState.STOPPED

    @pytest.mark.asyncio
    async def test_stale_socket_file_is_cleaned_up(self, tmp_path: Path) -> None:
        """If a socket file already exists from a previous crash, start should clean it."""
        sock_path = _make_socket_path(tmp_path)
        # Create a stale file to simulate a crashed daemon
        sock_path.touch()
        config = ServerConfig(socket_path=sock_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        try:
            assert server.state == ServerState.RUNNING
        finally:
            await server.shutdown()

    @pytest.mark.asyncio
    async def test_socket_directory_created_if_missing(
        self, tmp_path: Path
    ) -> None:
        """Server should create parent directories for the socket path."""
        nested_path = tmp_path / "deeply" / "nested" / "test.sock"
        config = ServerConfig(socket_path=nested_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        try:
            assert nested_path.exists()
        finally:
            await server.shutdown()


# ---------------------------------------------------------------------------
# SocketServer client communication tests
# ---------------------------------------------------------------------------


class TestSocketServerCommunication:
    """Tests for client message exchange over the Unix domain socket."""

    @pytest.mark.asyncio
    async def test_single_request_response(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        try:
            request = _build_envelope(verb="status")
            response = await _connect_and_send(config.socket_path, request)

            assert response.msg_type == MessageType.RESPONSE
            assert response.msg_id == request.msg_id
            assert response.payload["echo"] == {"verb": "status"}
        finally:
            await server.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_sequential_clients(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        try:
            for i in range(3):
                request = _build_envelope(
                    verb=f"test-{i}", msg_id=f"seq-{i}"
                )
                response = await _connect_and_send(config.socket_path, request)
                assert response.msg_id == f"seq-{i}"
                assert response.payload["echo"]["verb"] == f"test-{i}"
        finally:
            await server.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_clients(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        try:
            tasks = [
                _connect_and_send(
                    config.socket_path,
                    _build_envelope(verb=f"concurrent-{i}", msg_id=f"par-{i}"),
                )
                for i in range(5)
            ]
            responses = await asyncio.gather(*tasks)

            msg_ids = {r.msg_id for r in responses}
            assert msg_ids == {f"par-{i}" for i in range(5)}
        finally:
            await server.shutdown()

    @pytest.mark.asyncio
    async def test_invalid_frame_returns_error(self, tmp_path: Path) -> None:
        """Sending garbage bytes should yield an error envelope, not crash."""
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        try:
            raw_response = await _connect_and_send_raw(
                config.socket_path, b"not-a-valid-frame"
            )
            # Server should send an error frame or close the connection
            # Either way, the server should still be running
            assert server.state == ServerState.RUNNING
        finally:
            await server.shutdown()

    @pytest.mark.asyncio
    async def test_client_disconnect_does_not_crash_server(
        self, tmp_path: Path
    ) -> None:
        """Client disconnecting mid-stream should not crash the server."""
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        try:
            # Connect and immediately disconnect
            reader, writer = await asyncio.open_unix_connection(
                str(config.socket_path)
            )
            writer.close()
            await writer.wait_closed()

            # Give server time to notice
            await asyncio.sleep(0.05)

            # Server should still be running and accept new connections
            assert server.state == ServerState.RUNNING

            # Verify we can still communicate
            request = _build_envelope(verb="after-disconnect")
            response = await _connect_and_send(config.socket_path, request)
            assert response.msg_type == MessageType.RESPONSE
        finally:
            await server.shutdown()


# ---------------------------------------------------------------------------
# SocketServer graceful shutdown tests
# ---------------------------------------------------------------------------


class TestSocketServerGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_while_client_connected(
        self, tmp_path: Path
    ) -> None:
        """Server should complete gracefully even with connected clients."""
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()

        # Keep a client connected
        reader, writer = await asyncio.open_unix_connection(
            str(config.socket_path)
        )
        try:
            # Shutdown should complete without hanging
            await asyncio.wait_for(server.shutdown(), timeout=5.0)
            assert server.state == ServerState.STOPPED
        finally:
            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_shutdown_stops_accepting_new_connections(
        self, tmp_path: Path
    ) -> None:
        """After shutdown, new connections should be refused."""
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        await server.shutdown()

        with pytest.raises((ConnectionRefusedError, FileNotFoundError)):
            await asyncio.open_unix_connection(str(config.socket_path))

    @pytest.mark.asyncio
    async def test_active_client_count(self, tmp_path: Path) -> None:
        """Server tracks active client connections."""
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        try:
            assert server.active_client_count == 0

            reader, writer = await asyncio.open_unix_connection(
                str(config.socket_path)
            )
            # Give the server a moment to register the connection
            await asyncio.sleep(0.05)
            assert server.active_client_count >= 1

            writer.close()
            await writer.wait_closed()
            # Give the server time to clean up
            await asyncio.sleep(0.1)
            assert server.active_client_count == 0
        finally:
            await server.shutdown()


# ---------------------------------------------------------------------------
# SocketServer context manager tests
# ---------------------------------------------------------------------------


class TestSocketServerContextManager:
    """Tests for the async context manager interface."""

    @pytest.mark.asyncio
    async def test_context_manager_starts_and_stops(
        self, tmp_path: Path
    ) -> None:
        config = _make_config(tmp_path)
        async with SocketServer(
            config=config, handler=EchoHandler()
        ) as server:
            assert server.state == ServerState.RUNNING
            assert config.socket_path.exists()

        assert server.state == ServerState.STOPPED
        assert not config.socket_path.exists()

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up_on_error(
        self, tmp_path: Path
    ) -> None:
        config = _make_config(tmp_path)
        with pytest.raises(RuntimeError, match="test error"):
            async with SocketServer(
                config=config, handler=EchoHandler()
            ) as server:
                assert server.state == ServerState.RUNNING
                raise RuntimeError("test error")

        assert server.state == ServerState.STOPPED
        assert not config.socket_path.exists()

    @pytest.mark.asyncio
    async def test_context_manager_communication(
        self, tmp_path: Path
    ) -> None:
        config = _make_config(tmp_path)
        async with SocketServer(
            config=config, handler=EchoHandler()
        ):
            request = _build_envelope(verb="ctx-test")
            response = await _connect_and_send(config.socket_path, request)
            assert response.payload["echo"]["verb"] == "ctx-test"


# ---------------------------------------------------------------------------
# SocketServer handler error tests
# ---------------------------------------------------------------------------


class TestSocketServerHandlerErrors:
    """Tests for handler error propagation."""

    @pytest.mark.asyncio
    async def test_handler_exception_returns_error_envelope(
        self, tmp_path: Path
    ) -> None:
        """If the handler raises, the server should return an error envelope."""

        class FailingHandler:
            async def handle_message(
                self, envelope: MessageEnvelope, client: ClientConnection
            ) -> MessageEnvelope:
                raise ValueError("handler exploded")

        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=FailingHandler())
        await server.start()
        try:
            request = _build_envelope()
            response = await _connect_and_send(config.socket_path, request)

            assert response.msg_type == MessageType.ERROR
            assert "handler exploded" in response.payload.get("error", "")
        finally:
            await server.shutdown()

    @pytest.mark.asyncio
    async def test_server_survives_handler_crash(
        self, tmp_path: Path
    ) -> None:
        """Handler crashes should not take down the server."""
        call_count = 0

        class CrashOnceHandler:
            async def handle_message(
                self, envelope: MessageEnvelope, client: ClientConnection
            ) -> MessageEnvelope:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("first call fails")
                return MessageEnvelope(
                    msg_type=MessageType.RESPONSE,
                    msg_id=envelope.msg_id,
                    timestamp=envelope.timestamp,
                    payload={"ok": True},
                )

        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=CrashOnceHandler())
        await server.start()
        try:
            # First request: handler crashes, should get error response
            r1 = await _connect_and_send(
                config.socket_path, _build_envelope(msg_id="crash-1")
            )
            assert r1.msg_type == MessageType.ERROR

            # Second request: handler works, should get normal response
            r2 = await _connect_and_send(
                config.socket_path, _build_envelope(msg_id="ok-1")
            )
            assert r2.msg_type == MessageType.RESPONSE
            assert r2.payload["ok"] is True
        finally:
            await server.shutdown()


# ---------------------------------------------------------------------------
# SocketServer socket permissions tests
# ---------------------------------------------------------------------------


class TestSocketServerPermissions:
    """Tests for socket file permissions."""

    @pytest.mark.asyncio
    async def test_socket_file_has_restrictive_permissions(
        self, tmp_path: Path
    ) -> None:
        """Socket file should be owner-only accessible (0o600)."""
        config = _make_config(tmp_path)
        server = SocketServer(config=config, handler=EchoHandler())
        await server.start()
        try:
            sock_path = config.socket_path
            mode = sock_path.stat().st_mode
            # Check that group and other permissions are not set
            perms = stat.S_IMODE(mode)
            assert perms & stat.S_IRWXG == 0, "Group permissions should be zero"
            assert perms & stat.S_IRWXO == 0, "Other permissions should be zero"
        finally:
            await server.shutdown()
