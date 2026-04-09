"""Tests for the concurrent connection dispatcher.

Validates:
    - Explicit task spawning for each client connection
    - Concurrency limiting via semaphore
    - Task tracking for graceful shutdown
    - ConnectionManager integration for lifecycle events
    - Error isolation between client connections
    - Reject-during-shutdown behavior
    - Drain behavior with timeout
    - Active connection count accuracy
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from jules_daemon.ipc.connection_manager import (
    CLIENT_CONNECTED_EVENT,
    CLIENT_DISCONNECTED_EVENT,
    ConnectionManager,
)
from jules_daemon.ipc.event_bus import Event, EventBus
from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
    unpack_header,
)
from jules_daemon.ipc.server import (
    ClientConnection,
    ServerConfig,
    SocketServer,
)

# Import will exist after implementation
from jules_daemon.ipc.connection_dispatcher import (
    ConnectionDispatcher,
    DispatcherConfig,
    DispatcherState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_reader() -> AsyncMock:
    """Create a mock StreamReader."""
    return AsyncMock(spec=asyncio.StreamReader)


def _make_mock_writer() -> AsyncMock:
    """Create a mock StreamWriter that tracks close state."""
    writer = AsyncMock(spec=asyncio.StreamWriter)
    writer.is_closing.return_value = False
    return writer


class _EchoHandler:
    """Simple handler that echoes back the received envelope."""

    async def handle_message(
        self,
        envelope: MessageEnvelope,
        client: ClientConnection,
    ) -> MessageEnvelope:
        return MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id=envelope.msg_id,
            timestamp=envelope.timestamp,
            payload={"echo": envelope.payload},
        )


class _SlowHandler:
    """Handler that takes time to process, for concurrency testing."""

    def __init__(self, delay: float = 0.5) -> None:
        self._delay = delay

    async def handle_message(
        self,
        envelope: MessageEnvelope,
        client: ClientConnection,
    ) -> MessageEnvelope:
        await asyncio.sleep(self._delay)
        return MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id=envelope.msg_id,
            timestamp=envelope.timestamp,
            payload={"slow": True},
        )


class _FailingHandler:
    """Handler that raises an exception."""

    async def handle_message(
        self,
        envelope: MessageEnvelope,
        client: ClientConnection,
    ) -> MessageEnvelope:
        raise ValueError("handler boom")


# ---------------------------------------------------------------------------
# DispatcherConfig tests
# ---------------------------------------------------------------------------


class TestDispatcherConfig:
    """Tests for the immutable DispatcherConfig dataclass."""

    def test_create_with_defaults(self) -> None:
        handler = _EchoHandler()
        config = DispatcherConfig(handler=handler)
        assert config.max_concurrent_clients == 10
        assert config.handler is handler
        assert config.connection_manager is None

    def test_custom_values(self) -> None:
        handler = _EchoHandler()
        manager = ConnectionManager()
        config = DispatcherConfig(
            handler=handler,
            max_concurrent_clients=5,
            connection_manager=manager,
        )
        assert config.max_concurrent_clients == 5
        assert config.connection_manager is manager

    def test_zero_max_clients_raises(self) -> None:
        with pytest.raises(ValueError, match="max_concurrent_clients must be positive"):
            DispatcherConfig(handler=_EchoHandler(), max_concurrent_clients=0)

    def test_negative_max_clients_raises(self) -> None:
        with pytest.raises(ValueError, match="max_concurrent_clients must be positive"):
            DispatcherConfig(handler=_EchoHandler(), max_concurrent_clients=-1)


# ---------------------------------------------------------------------------
# DispatcherState tests
# ---------------------------------------------------------------------------


class TestDispatcherState:
    """Tests for the DispatcherState enum."""

    def test_all_states_exist(self) -> None:
        assert len(DispatcherState) == 3

    def test_ready_value(self) -> None:
        assert DispatcherState.READY.value == "ready"

    def test_active_value(self) -> None:
        assert DispatcherState.ACTIVE.value == "active"

    def test_draining_value(self) -> None:
        assert DispatcherState.DRAINING.value == "draining"


# ---------------------------------------------------------------------------
# ConnectionDispatcher lifecycle tests
# ---------------------------------------------------------------------------


class TestDispatcherLifecycle:
    """Tests for dispatcher creation and state transitions."""

    def test_initial_state_is_ready(self) -> None:
        config = DispatcherConfig(handler=_EchoHandler())
        dispatcher = ConnectionDispatcher(config=config)
        assert dispatcher.state == DispatcherState.READY

    def test_active_connection_count_starts_at_zero(self) -> None:
        config = DispatcherConfig(handler=_EchoHandler())
        dispatcher = ConnectionDispatcher(config=config)
        assert dispatcher.active_connection_count == 0

    @pytest.mark.asyncio
    async def test_drain_on_empty_dispatcher(self) -> None:
        """Draining with no active connections completes immediately."""
        config = DispatcherConfig(handler=_EchoHandler())
        dispatcher = ConnectionDispatcher(config=config)
        await dispatcher.drain(timeout=1.0)
        assert dispatcher.state == DispatcherState.DRAINING

    @pytest.mark.asyncio
    async def test_drain_is_idempotent(self) -> None:
        """Calling drain multiple times is safe."""
        config = DispatcherConfig(handler=_EchoHandler())
        dispatcher = ConnectionDispatcher(config=config)
        await dispatcher.drain(timeout=1.0)
        await dispatcher.drain(timeout=1.0)
        assert dispatcher.state == DispatcherState.DRAINING


# ---------------------------------------------------------------------------
# ConnectionDispatcher dispatch tests
# ---------------------------------------------------------------------------


class TestDispatcherDispatch:
    """Tests for connection dispatching and task spawning."""

    @pytest.mark.asyncio
    async def test_dispatch_creates_task(self) -> None:
        """Dispatching a connection should create an explicit asyncio task."""
        config = DispatcherConfig(handler=_EchoHandler())
        dispatcher = ConnectionDispatcher(config=config)

        reader = _make_mock_reader()
        writer = _make_mock_writer()

        # Simulate immediate EOF on read
        reader.readexactly.side_effect = asyncio.IncompleteReadError(b"", HEADER_SIZE)

        await dispatcher.dispatch(reader, writer)

        # Give the task time to run and complete
        await asyncio.sleep(0.05)

        # After EOF, the connection should have been cleaned up
        assert dispatcher.active_connection_count == 0

    @pytest.mark.asyncio
    async def test_dispatch_rejected_during_drain(self) -> None:
        """New connections are rejected after drain has been called."""
        config = DispatcherConfig(handler=_EchoHandler())
        dispatcher = ConnectionDispatcher(config=config)
        await dispatcher.drain(timeout=1.0)

        reader = _make_mock_reader()
        writer = _make_mock_writer()

        result = await dispatcher.dispatch(reader, writer)
        assert result is False

    @pytest.mark.asyncio
    async def test_dispatch_returns_true_on_accept(self) -> None:
        """Dispatch returns True when the connection is accepted."""
        config = DispatcherConfig(handler=_EchoHandler())
        dispatcher = ConnectionDispatcher(config=config)

        reader = _make_mock_reader()
        writer = _make_mock_writer()
        reader.readexactly.side_effect = asyncio.IncompleteReadError(b"", HEADER_SIZE)

        result = await dispatcher.dispatch(reader, writer)
        assert result is True

        # Cleanup
        await asyncio.sleep(0.05)
        await dispatcher.drain(timeout=1.0)


# ---------------------------------------------------------------------------
# Concurrency limiting tests
# ---------------------------------------------------------------------------


class TestDispatcherConcurrency:
    """Tests for concurrency limiting via semaphore."""

    @pytest.mark.asyncio
    async def test_concurrency_limit_enforced(self) -> None:
        """Connections beyond max_concurrent_clients wait for semaphore."""
        handler_entered = asyncio.Event()
        handler_entered_count = 0
        barrier = asyncio.Event()

        class HoldHandler:
            async def handle_message(
                self, envelope: MessageEnvelope, client: ClientConnection
            ) -> MessageEnvelope:
                nonlocal handler_entered_count
                handler_entered_count += 1
                handler_entered.set()
                await barrier.wait()
                return MessageEnvelope(
                    msg_type=MessageType.RESPONSE,
                    msg_id=envelope.msg_id,
                    timestamp=envelope.timestamp,
                    payload={},
                )

        config = DispatcherConfig(
            handler=HoldHandler(),
            max_concurrent_clients=2,
        )
        dispatcher = ConnectionDispatcher(config=config)

        def _make_blocking_reader() -> AsyncMock:
            """Create a reader that sends one message then blocks forever."""
            reader = AsyncMock(spec=asyncio.StreamReader)
            request = MessageEnvelope(
                msg_type=MessageType.REQUEST,
                msg_id="conc-001",
                timestamp="2026-04-09T12:00:00Z",
                payload={"verb": "status"},
            )
            frame = encode_frame(request)
            header = frame[:HEADER_SIZE]
            payload = frame[HEADER_SIZE:]

            call_idx_holder = [0]

            async def mock_readexactly(n: int) -> bytes:
                idx = call_idx_holder[0]
                call_idx_holder[0] += 1
                if idx == 0:
                    return header
                if idx == 1:
                    return payload
                # Block forever after first message (simulating long connection)
                await asyncio.sleep(100)
                raise asyncio.IncompleteReadError(b"", n)

            reader.readexactly = mock_readexactly
            return reader

        # Dispatch 3 connections with max_concurrent_clients=2
        for _ in range(3):
            reader = _make_blocking_reader()
            writer = _make_mock_writer()
            await dispatcher.dispatch(reader, writer)

        # Give tasks time to start
        await asyncio.sleep(0.15)

        # All 3 dispatched, but only 2 should enter the handler
        # (the 3rd waits on semaphore)
        assert dispatcher.active_connection_count == 3
        assert handler_entered_count == 2

        # Cleanup
        barrier.set()
        await dispatcher.drain(timeout=2.0)


# ---------------------------------------------------------------------------
# ConnectionManager integration tests
# ---------------------------------------------------------------------------


class TestDispatcherConnectionManager:
    """Tests for ConnectionManager integration."""

    @pytest.mark.asyncio
    async def test_dispatch_registers_with_manager(self) -> None:
        """Dispatched connection should be registered with ConnectionManager."""
        bus = EventBus()
        manager = ConnectionManager(event_bus=bus)

        events: list[Event] = []

        async def on_connect(event: Event) -> None:
            events.append(event)

        bus.subscribe(CLIENT_CONNECTED_EVENT, on_connect)

        config = DispatcherConfig(
            handler=_EchoHandler(),
            connection_manager=manager,
        )
        dispatcher = ConnectionDispatcher(config=config)

        reader = _make_mock_reader()
        writer = _make_mock_writer()
        reader.readexactly.side_effect = asyncio.IncompleteReadError(b"", HEADER_SIZE)

        await dispatcher.dispatch(reader, writer)
        await asyncio.sleep(0.05)

        # Connection should have been registered
        assert len(events) >= 1
        assert events[0].event_type == CLIENT_CONNECTED_EVENT

        await dispatcher.drain(timeout=1.0)

    @pytest.mark.asyncio
    async def test_disconnect_deregisters_from_manager(self) -> None:
        """When client disconnects, it should be deregistered from manager."""
        bus = EventBus()
        manager = ConnectionManager(event_bus=bus)

        disconnect_events: list[Event] = []

        async def on_disconnect(event: Event) -> None:
            disconnect_events.append(event)

        bus.subscribe(CLIENT_DISCONNECTED_EVENT, on_disconnect)

        config = DispatcherConfig(
            handler=_EchoHandler(),
            connection_manager=manager,
        )
        dispatcher = ConnectionDispatcher(config=config)

        reader = _make_mock_reader()
        writer = _make_mock_writer()
        # Simulate immediate EOF
        reader.readexactly.side_effect = asyncio.IncompleteReadError(b"", HEADER_SIZE)

        await dispatcher.dispatch(reader, writer)
        await asyncio.sleep(0.1)

        # After disconnect, should have been deregistered
        assert len(disconnect_events) >= 1
        assert manager.client_count == 0

        await dispatcher.drain(timeout=1.0)

    @pytest.mark.asyncio
    async def test_works_without_connection_manager(self) -> None:
        """Dispatcher works fine without a ConnectionManager."""
        config = DispatcherConfig(handler=_EchoHandler())
        dispatcher = ConnectionDispatcher(config=config)

        reader = _make_mock_reader()
        writer = _make_mock_writer()
        reader.readexactly.side_effect = asyncio.IncompleteReadError(b"", HEADER_SIZE)

        # Should not raise
        await dispatcher.dispatch(reader, writer)
        await asyncio.sleep(0.05)
        await dispatcher.drain(timeout=1.0)


# ---------------------------------------------------------------------------
# Error isolation tests
# ---------------------------------------------------------------------------


class TestDispatcherErrorIsolation:
    """Tests that handler errors do not crash the dispatcher."""

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash_dispatcher(self) -> None:
        """A crashing handler should not take down the dispatcher."""
        config = DispatcherConfig(handler=_FailingHandler())
        dispatcher = ConnectionDispatcher(config=config)

        reader = _make_mock_reader()
        writer = _make_mock_writer()

        # Simulate: one valid message, then EOF
        request = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="fail-001",
            timestamp="2026-04-09T12:00:00Z",
            payload={"verb": "status"},
        )
        frame = encode_frame(request)
        header = frame[:HEADER_SIZE]
        payload = frame[HEADER_SIZE:]

        read_values = [header, payload]
        call_idx = 0

        async def mock_readexactly(n: int) -> bytes:
            nonlocal call_idx
            if call_idx < len(read_values):
                val = read_values[call_idx]
                call_idx += 1
                return val
            raise asyncio.IncompleteReadError(b"", n)

        reader.readexactly = mock_readexactly

        await dispatcher.dispatch(reader, writer)
        await asyncio.sleep(0.1)

        # Dispatcher should still be accepting connections (not draining)
        assert dispatcher.state != DispatcherState.DRAINING
        await dispatcher.drain(timeout=1.0)


# ---------------------------------------------------------------------------
# Drain / shutdown tests
# ---------------------------------------------------------------------------


class TestDispatcherDrain:
    """Tests for graceful drain behavior."""

    @pytest.mark.asyncio
    async def test_drain_waits_for_active_tasks(self) -> None:
        """Drain should wait for in-flight connections to complete."""
        completed = asyncio.Event()

        class SignalHandler:
            async def handle_message(
                self, envelope: MessageEnvelope, client: ClientConnection
            ) -> MessageEnvelope:
                completed.set()
                return MessageEnvelope(
                    msg_type=MessageType.RESPONSE,
                    msg_id=envelope.msg_id,
                    timestamp=envelope.timestamp,
                    payload={},
                )

        config = DispatcherConfig(handler=SignalHandler())
        dispatcher = ConnectionDispatcher(config=config)

        reader = _make_mock_reader()
        writer = _make_mock_writer()

        # Simulate: one message then EOF
        request = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="drain-001",
            timestamp="2026-04-09T12:00:00Z",
            payload={"verb": "status"},
        )
        frame = encode_frame(request)
        header = frame[:HEADER_SIZE]
        payload = frame[HEADER_SIZE:]

        read_values = [header, payload]
        call_idx = 0

        async def mock_readexactly(n: int) -> bytes:
            nonlocal call_idx
            if call_idx < len(read_values):
                val = read_values[call_idx]
                call_idx += 1
                return val
            raise asyncio.IncompleteReadError(b"", n)

        reader.readexactly = mock_readexactly

        await dispatcher.dispatch(reader, writer)
        await asyncio.sleep(0.1)
        await dispatcher.drain(timeout=2.0)

        # Handler should have completed
        assert completed.is_set()
        assert dispatcher.active_connection_count == 0

    @pytest.mark.asyncio
    async def test_drain_cancels_on_timeout(self) -> None:
        """Drain should cancel tasks that exceed the timeout."""
        config = DispatcherConfig(handler=_SlowHandler(delay=5.0))
        dispatcher = ConnectionDispatcher(config=config)

        reader = _make_mock_reader()
        writer = _make_mock_writer()

        # Simulate: one message then block forever (slow handler)
        request = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="timeout-001",
            timestamp="2026-04-09T12:00:00Z",
            payload={"verb": "status"},
        )
        frame = encode_frame(request)
        header = frame[:HEADER_SIZE]
        payload = frame[HEADER_SIZE:]

        call_idx = 0

        async def mock_readexactly(n: int) -> bytes:
            nonlocal call_idx
            if call_idx == 0:
                call_idx += 1
                return header
            if call_idx == 1:
                call_idx += 1
                return payload
            # Block to simulate long connection
            await asyncio.sleep(100)
            raise asyncio.IncompleteReadError(b"", n)

        reader.readexactly = mock_readexactly

        await dispatcher.dispatch(reader, writer)
        await asyncio.sleep(0.05)

        # Drain with short timeout should cancel the slow task
        await dispatcher.drain(timeout=0.2)
        assert dispatcher.active_connection_count == 0


# ---------------------------------------------------------------------------
# Integration tests with SocketServer
# ---------------------------------------------------------------------------


class TestDispatcherServerIntegration:
    """Tests verifying SocketServer uses ConnectionDispatcher correctly."""

    @pytest.mark.asyncio
    async def test_server_uses_dispatcher_for_concurrent_clients(
        self, tmp_path: Path
    ) -> None:
        """Server should handle multiple concurrent clients via dispatcher."""
        sock_path = tmp_path / "test.sock"
        config = ServerConfig(socket_path=sock_path, max_concurrent_clients=5)
        server = SocketServer(config=config, handler=_EchoHandler())
        await server.start()
        try:
            # Send multiple concurrent requests
            tasks = []
            for i in range(5):
                request = MessageEnvelope(
                    msg_type=MessageType.REQUEST,
                    msg_id=f"par-{i}",
                    timestamp="2026-04-09T12:00:00Z",
                    payload={"verb": f"test-{i}"},
                )
                tasks.append(_connect_and_send(sock_path, request))

            responses = await asyncio.gather(*tasks)
            msg_ids = {r.msg_id for r in responses}
            assert msg_ids == {f"par-{i}" for i in range(5)}
        finally:
            await server.shutdown()

    @pytest.mark.asyncio
    async def test_server_exposes_dispatcher(self, tmp_path: Path) -> None:
        """Server should expose its dispatcher for introspection."""
        sock_path = tmp_path / "test.sock"
        config = ServerConfig(socket_path=sock_path)
        server = SocketServer(config=config, handler=_EchoHandler())
        assert server.dispatcher is not None
        assert isinstance(server.dispatcher, ConnectionDispatcher)

    @pytest.mark.asyncio
    async def test_server_dispatcher_state_follows_server(
        self, tmp_path: Path
    ) -> None:
        """Dispatcher state should reflect server lifecycle."""
        sock_path = tmp_path / "test.sock"
        config = ServerConfig(socket_path=sock_path)
        server = SocketServer(config=config, handler=_EchoHandler())

        # Before start
        assert server.dispatcher.state == DispatcherState.READY

        await server.start()
        try:
            # A dispatch in READY or ACTIVE is valid
            assert server.dispatcher.state in (
                DispatcherState.READY,
                DispatcherState.ACTIVE,
            )
        finally:
            await server.shutdown()

        # After shutdown
        assert server.dispatcher.state == DispatcherState.DRAINING


# ---------------------------------------------------------------------------
# Helper for integration tests
# ---------------------------------------------------------------------------


async def _connect_and_send(
    socket_path: Path,
    envelope: MessageEnvelope,
) -> MessageEnvelope:
    """Connect, send a framed envelope, read back a response."""
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
