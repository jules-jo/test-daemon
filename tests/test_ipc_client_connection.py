"""Tests for the IPC client connection module.

Validates:
    - ConnectionConfig dataclass immutability and validation
    - HandshakeResult dataclass immutability and fields
    - ClientConnection connects to a Unix domain socket
    - ClientConnection performs protocol handshake with the daemon
    - ClientConnection sends and receives framed MessageEnvelopes
    - Timeout handling for connect and handshake phases
    - Graceful error handling for connection refused, daemon not running
    - Handshake protocol version mismatch detection
    - Async context manager interface (connect on enter, close on exit)
    - Auto-discovery of socket path when not explicitly configured
    - Connection state tracking (connected, handshake_complete, closed)
"""

from __future__ import annotations

import asyncio
import os
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    encode_frame,
    unpack_header,
)
from jules_daemon.ipc.client_connection import (
    HANDSHAKE_VERB,
    PROTOCOL_VERSION,
    ClientConnection,
    ConnectionConfig,
    ConnectionError as IpcConnectionError,
    ConnectionState,
    HandshakeError,
    HandshakeResult,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TS = "2026-04-09T12:00:00Z"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handshake_response(
    *,
    protocol_version: int = PROTOCOL_VERSION,
    daemon_pid: int = 12345,
    daemon_uptime: float = 42.0,
    msg_id: str = "daemon-hs-001",
    status: str = "ok",
) -> MessageEnvelope:
    """Build a daemon handshake response envelope."""
    return MessageEnvelope(
        msg_type=MessageType.RESPONSE,
        msg_id=msg_id,
        timestamp=_TS,
        payload={
            "verb": HANDSHAKE_VERB,
            "status": status,
            "protocol_version": protocol_version,
            "daemon_pid": daemon_pid,
            "daemon_uptime_seconds": daemon_uptime,
        },
    )


def _make_handshake_error(
    *,
    error_msg: str = "Unsupported protocol version",
    msg_id: str = "daemon-err-001",
) -> MessageEnvelope:
    """Build a daemon handshake error envelope."""
    return MessageEnvelope(
        msg_type=MessageType.ERROR,
        msg_id=msg_id,
        timestamp=_TS,
        payload={"error": error_msg},
    )


def _build_stream_reader(envelopes: list[MessageEnvelope]) -> asyncio.StreamReader:
    """Build a StreamReader pre-loaded with framed envelope data."""
    reader = asyncio.StreamReader()
    for env in envelopes:
        reader.feed_data(encode_frame(env))
    reader.feed_eof()
    return reader


def _make_writer_mock() -> AsyncMock:
    """Build a mock StreamWriter that records writes."""
    writer = AsyncMock(spec=asyncio.StreamWriter)
    writer.is_closing.return_value = False
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return writer


# ---------------------------------------------------------------------------
# ConnectionConfig tests
# ---------------------------------------------------------------------------


class TestConnectionConfig:
    """Tests for the immutable ConnectionConfig dataclass."""

    def test_defaults(self) -> None:
        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        assert config.socket_path == Path("/tmp/test.sock")
        assert config.connect_timeout == 5.0
        assert config.handshake_timeout == 5.0
        assert config.protocol_version == PROTOCOL_VERSION

    def test_custom_values(self) -> None:
        config = ConnectionConfig(
            socket_path=Path("/custom/path.sock"),
            connect_timeout=10.0,
            handshake_timeout=3.0,
            protocol_version=2,
        )
        assert config.connect_timeout == 10.0
        assert config.handshake_timeout == 3.0
        assert config.protocol_version == 2

    def test_frozen(self) -> None:
        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        with pytest.raises(AttributeError):
            config.socket_path = Path("/mutated")  # type: ignore[misc]

    def test_negative_connect_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="connect_timeout must be positive"):
            ConnectionConfig(
                socket_path=Path("/tmp/test.sock"),
                connect_timeout=-1.0,
            )

    def test_zero_handshake_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="handshake_timeout must be positive"):
            ConnectionConfig(
                socket_path=Path("/tmp/test.sock"),
                handshake_timeout=0.0,
            )

    def test_zero_protocol_version_raises(self) -> None:
        with pytest.raises(ValueError, match="protocol_version must be positive"):
            ConnectionConfig(
                socket_path=Path("/tmp/test.sock"),
                protocol_version=0,
            )

    def test_none_socket_path_allowed(self) -> None:
        """None socket_path triggers auto-discovery."""
        config = ConnectionConfig(socket_path=None)
        assert config.socket_path is None


# ---------------------------------------------------------------------------
# HandshakeResult tests
# ---------------------------------------------------------------------------


class TestHandshakeResult:
    """Tests for the immutable HandshakeResult dataclass."""

    def test_success_result(self) -> None:
        result = HandshakeResult(
            success=True,
            protocol_version=1,
            daemon_pid=12345,
            daemon_uptime_seconds=42.0,
            error=None,
        )
        assert result.success is True
        assert result.protocol_version == 1
        assert result.daemon_pid == 12345
        assert result.daemon_uptime_seconds == 42.0
        assert result.error is None

    def test_failure_result(self) -> None:
        result = HandshakeResult(
            success=False,
            protocol_version=0,
            daemon_pid=None,
            daemon_uptime_seconds=None,
            error="Connection refused",
        )
        assert result.success is False
        assert result.error == "Connection refused"

    def test_frozen(self) -> None:
        result = HandshakeResult(
            success=True,
            protocol_version=1,
            daemon_pid=1,
            daemon_uptime_seconds=0.0,
            error=None,
        )
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ConnectionState tests
# ---------------------------------------------------------------------------


class TestConnectionState:
    """Tests for the ConnectionState enum."""

    def test_all_values(self) -> None:
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.HANDSHAKING.value == "handshaking"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.CLOSED.value == "closed"


# ---------------------------------------------------------------------------
# ClientConnection -- creation and state
# ---------------------------------------------------------------------------


class TestClientConnectionCreation:
    """Tests for ClientConnection initialization."""

    def test_initial_state_is_disconnected(self) -> None:
        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)
        assert conn.state == ConnectionState.DISCONNECTED

    def test_config_is_accessible(self) -> None:
        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)
        assert conn.config == config

    def test_auto_discover_socket_path_when_none(self) -> None:
        """When socket_path is None, it should be resolved during connect."""
        config = ConnectionConfig(socket_path=None)
        conn = ClientConnection(config=config)
        assert conn.config.socket_path is None


# ---------------------------------------------------------------------------
# ClientConnection -- handshake protocol
# ---------------------------------------------------------------------------


class TestClientConnectionHandshake:
    """Tests for the handshake protocol exchange."""

    @pytest.mark.asyncio
    async def test_successful_handshake(self) -> None:
        """Client sends handshake request, daemon responds with success."""
        response = _make_handshake_response(daemon_pid=9999, daemon_uptime=100.0)
        reader = _build_stream_reader([response])
        writer = _make_writer_mock()

        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)

        result = await conn._perform_handshake(reader, writer)

        assert result.success is True
        assert result.protocol_version == PROTOCOL_VERSION
        assert result.daemon_pid == 9999
        assert result.daemon_uptime_seconds == 100.0
        assert result.error is None

        # Verify a handshake request frame was sent
        writer.write.assert_called_once()
        writer.drain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handshake_sends_correct_verb(self) -> None:
        """The handshake request contains the handshake verb and version."""
        response = _make_handshake_response()
        reader = _build_stream_reader([response])
        writer = _make_writer_mock()

        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)

        await conn._perform_handshake(reader, writer)

        frame_bytes = writer.write.call_args[0][0]
        assert HANDSHAKE_VERB.encode() in frame_bytes
        assert b"protocol_version" in frame_bytes

    @pytest.mark.asyncio
    async def test_handshake_error_response(self) -> None:
        """Daemon returns ERROR envelope during handshake."""
        error_env = _make_handshake_error(error_msg="Protocol mismatch")
        reader = _build_stream_reader([error_env])
        writer = _make_writer_mock()

        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)

        result = await conn._perform_handshake(reader, writer)

        assert result.success is False
        assert "Protocol mismatch" in (result.error or "")

    @pytest.mark.asyncio
    async def test_handshake_connection_lost(self) -> None:
        """Connection drops before handshake response is received."""
        reader = asyncio.StreamReader()
        reader.feed_eof()
        writer = _make_writer_mock()

        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)

        result = await conn._perform_handshake(reader, writer)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_handshake_timeout(self) -> None:
        """Handshake times out when daemon does not respond."""
        reader = asyncio.StreamReader()
        # Never feed data -- reader will block
        writer = _make_writer_mock()

        config = ConnectionConfig(
            socket_path=Path("/tmp/test.sock"),
            handshake_timeout=0.05,
        )
        conn = ClientConnection(config=config)

        result = await conn._perform_handshake(reader, writer)

        assert result.success is False
        assert result.error is not None
        assert "timeout" in result.error.lower() or "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handshake_version_mismatch(self) -> None:
        """Daemon reports a different protocol version."""
        response = _make_handshake_response(protocol_version=999)
        reader = _build_stream_reader([response])
        writer = _make_writer_mock()

        config = ConnectionConfig(
            socket_path=Path("/tmp/test.sock"),
            protocol_version=1,
        )
        conn = ClientConnection(config=config)

        result = await conn._perform_handshake(reader, writer)

        assert result.success is False
        assert "version" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# ClientConnection -- send and receive
# ---------------------------------------------------------------------------


class TestClientConnectionIO:
    """Tests for send/receive on an established connection."""

    @pytest.mark.asyncio
    async def test_send_envelope(self) -> None:
        """send() writes a framed envelope to the stream."""
        writer = _make_writer_mock()

        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)
        # Inject writer for testing
        conn._writer = writer

        env = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="test-001",
            timestamp=_TS,
            payload={"verb": "status"},
        )
        await conn.send(env)

        writer.write.assert_called_once()
        writer.drain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_receive_envelope(self) -> None:
        """receive() reads and decodes a framed envelope from the stream."""
        env = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="resp-001",
            timestamp=_TS,
            payload={"status": "ok"},
        )
        reader = _build_stream_reader([env])

        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)
        # Inject reader for testing
        conn._reader = reader

        result = await conn.receive()

        assert result is not None
        assert result.msg_id == "resp-001"
        assert result.payload["status"] == "ok"

    @pytest.mark.asyncio
    async def test_receive_returns_none_on_eof(self) -> None:
        """receive() returns None when the stream reaches EOF."""
        reader = asyncio.StreamReader()
        reader.feed_eof()

        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)
        conn._reader = reader

        result = await conn.receive()
        assert result is None

    @pytest.mark.asyncio
    async def test_receive_with_timeout(self) -> None:
        """receive() returns None on timeout."""
        reader = asyncio.StreamReader()
        # Never feed data

        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)
        conn._reader = reader

        result = await conn.receive(timeout=0.05)
        assert result is None


# ---------------------------------------------------------------------------
# ClientConnection -- close
# ---------------------------------------------------------------------------


class TestClientConnectionClose:
    """Tests for connection teardown."""

    @pytest.mark.asyncio
    async def test_close_closes_writer(self) -> None:
        writer = _make_writer_mock()

        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)
        conn._writer = writer
        conn._state = ConnectionState.CONNECTED

        await conn.close()

        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()
        assert conn.state == ConnectionState.CLOSED

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        """Calling close() multiple times is safe."""
        writer = _make_writer_mock()

        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)
        conn._writer = writer
        conn._state = ConnectionState.CONNECTED

        await conn.close()
        await conn.close()  # Should not raise

        assert conn.state == ConnectionState.CLOSED

    @pytest.mark.asyncio
    async def test_close_on_disconnected_is_safe(self) -> None:
        """Closing a never-connected connection is a safe no-op."""
        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)

        await conn.close()
        assert conn.state == ConnectionState.CLOSED

    @pytest.mark.asyncio
    async def test_close_handles_writer_error(self) -> None:
        """close() swallows OSError from the writer."""
        writer = _make_writer_mock()
        writer.close.side_effect = OSError("already closed")

        config = ConnectionConfig(socket_path=Path("/tmp/test.sock"))
        conn = ClientConnection(config=config)
        conn._writer = writer
        conn._state = ConnectionState.CONNECTED

        await conn.close()  # Should not raise
        assert conn.state == ConnectionState.CLOSED


# ---------------------------------------------------------------------------
# ClientConnection -- full connect flow (integration)
# ---------------------------------------------------------------------------


class TestClientConnectionConnect:
    """Integration tests for the full connect + handshake flow."""

    @pytest.mark.asyncio
    async def test_connect_to_real_socket(self, tmp_path: Path) -> None:
        """Full flow: start a mock server, connect, handshake, close."""
        sock_path = tmp_path / "test-daemon.sock"

        # Set up a minimal server that responds with a handshake
        handshake_resp = _make_handshake_response(
            daemon_pid=os.getpid(),
            daemon_uptime=1.0,
        )
        response_frame = encode_frame(handshake_resp)

        async def handle_client(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            # Read the handshake request
            header = await reader.readexactly(HEADER_SIZE)
            length = unpack_header(header)
            await reader.readexactly(length)
            # Send the handshake response
            writer.write(response_frame)
            await writer.drain()
            # Keep connection open briefly then close
            await asyncio.sleep(0.1)
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(
            handle_client,
            path=str(sock_path),
        )

        try:
            config = ConnectionConfig(
                socket_path=sock_path,
                connect_timeout=2.0,
                handshake_timeout=2.0,
            )
            conn = ClientConnection(config=config)

            result = await conn.connect()

            assert result.success is True
            assert result.daemon_pid == os.getpid()
            assert conn.state == ConnectionState.CONNECTED

            await conn.close()
            assert conn.state == ConnectionState.CLOSED
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_connect_refuses_when_no_server(self, tmp_path: Path) -> None:
        """Connection fails gracefully when no daemon is running."""
        sock_path = tmp_path / "nonexistent.sock"
        config = ConnectionConfig(
            socket_path=sock_path,
            connect_timeout=0.5,
        )
        conn = ClientConnection(config=config)

        result = await conn.connect()

        assert result.success is False
        assert result.error is not None
        assert conn.state == ConnectionState.DISCONNECTED


# ---------------------------------------------------------------------------
# ClientConnection -- async context manager
# ---------------------------------------------------------------------------


class TestClientConnectionContextManager:
    """Tests for the async context manager interface."""

    @pytest.mark.asyncio
    async def test_context_manager_connects_and_closes(
        self, tmp_path: Path
    ) -> None:
        """async with ClientConnection connects on entry, closes on exit."""
        sock_path = tmp_path / "ctx-daemon.sock"

        handshake_resp = _make_handshake_response(daemon_pid=1)
        response_frame = encode_frame(handshake_resp)

        async def handle_client(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            header = await reader.readexactly(HEADER_SIZE)
            length = unpack_header(header)
            await reader.readexactly(length)
            writer.write(response_frame)
            await writer.drain()
            await asyncio.sleep(0.1)
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(
            handle_client,
            path=str(sock_path),
        )

        try:
            config = ConnectionConfig(
                socket_path=sock_path,
                connect_timeout=2.0,
                handshake_timeout=2.0,
            )
            async with ClientConnection(config=config) as conn:
                assert conn.state == ConnectionState.CONNECTED
            assert conn.state == ConnectionState.CLOSED
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_context_manager_raises_on_failed_connect(
        self, tmp_path: Path
    ) -> None:
        """async with raises IpcConnectionError when connection fails."""
        sock_path = tmp_path / "no-daemon.sock"
        config = ConnectionConfig(
            socket_path=sock_path,
            connect_timeout=0.5,
        )
        with pytest.raises(IpcConnectionError):
            async with ClientConnection(config=config):
                pass  # pragma: no cover
