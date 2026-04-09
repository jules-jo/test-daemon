"""Tests for resource cleanup handlers on disconnect events.

Validates:
    - SSH channel cleanup: close channel, flush pending I/O buffers
    - Socket connection cleanup: close StreamWriter, drain pending writes
    - I/O buffer flushing before resource release
    - DisconnectCleanupHandler orchestrates all cleanup on disconnect events
    - Context manager protocol for deterministic teardown
    - atexit registration for process-level shutdown safety
    - Cleanup order: flush buffers -> close channels -> close sockets
    - Idempotent cleanup (double-close is safe)
    - Error isolation (one failing cleanup does not block others)
    - EventBus integration for disconnect event detection
    - CleanupSummary captures results for audit
    - SSHChannelGuard context manager for scoped channel lifecycle
"""

from __future__ import annotations

import asyncio
import atexit
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.cleanup.resource_types import (
    CleanupResult,
    CleanupSummary,
    ResourceType,
)
from jules_daemon.cleanup.disconnect_handler import (
    DisconnectCleanupHandler,
)
from jules_daemon.cleanup.channel_guard import (
    SSHChannelGuard,
)
from jules_daemon.ipc.connection_manager import (
    CLIENT_DISCONNECTED_EVENT,
)
from jules_daemon.ipc.event_bus import Event, EventBus


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

_TS = "2026-04-09T12:00:00Z"


class FakeSSHChannel:
    """Fake SSH channel that tracks close/flush operations."""

    def __init__(
        self,
        *,
        has_stdout: bool = False,
        has_stderr: bool = False,
        stdout_data: bytes = b"",
        stderr_data: bytes = b"",
        is_closed: bool = False,
    ) -> None:
        self._has_stdout = has_stdout
        self._has_stderr = has_stderr
        self._stdout_data = stdout_data
        self._stderr_data = stderr_data
        self._closed = is_closed
        self.close_called = False
        self.shutdown_called = False
        self.shutdown_how: int | None = None

    def recv_ready(self) -> bool:
        return self._has_stdout and not self._closed

    def recv(self, nbytes: int) -> bytes:
        data = self._stdout_data[:nbytes]
        self._stdout_data = self._stdout_data[nbytes:]
        self._has_stdout = len(self._stdout_data) > 0
        return data

    def recv_stderr_ready(self) -> bool:
        return self._has_stderr and not self._closed

    def recv_stderr(self, nbytes: int) -> bytes:
        data = self._stderr_data[:nbytes]
        self._stderr_data = self._stderr_data[nbytes:]
        self._has_stderr = len(self._stderr_data) > 0
        return data

    @property
    def closed(self) -> bool:
        return self._closed

    def eof_received(self) -> bool:
        return self._closed

    def exit_status_is_ready(self) -> bool:
        return self._closed

    def get_exit_status(self) -> int:
        return 0

    def close(self) -> None:
        self.close_called = True
        self._closed = True

    def shutdown(self, how: int) -> None:
        self.shutdown_called = True
        self.shutdown_how = how


class FakeSSHChannelRaising:
    """Fake SSH channel that raises on close."""

    def __init__(self) -> None:
        self._closed = False

    def recv_ready(self) -> bool:
        return False

    def recv(self, nbytes: int) -> bytes:
        return b""

    def recv_stderr_ready(self) -> bool:
        return False

    def recv_stderr(self, nbytes: int) -> bytes:
        return b""

    @property
    def closed(self) -> bool:
        return self._closed

    def eof_received(self) -> bool:
        return False

    def exit_status_is_ready(self) -> bool:
        return False

    def get_exit_status(self) -> int:
        return -1

    def close(self) -> None:
        raise OSError("channel close failed")

    def shutdown(self, how: int) -> None:
        raise OSError("channel shutdown failed")


def _make_writer_mock() -> AsyncMock:
    """Build a mock StreamWriter that records writes."""
    writer = AsyncMock(spec=asyncio.StreamWriter)
    writer.is_closing.return_value = False
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return writer


def _make_broken_writer_mock() -> AsyncMock:
    """Build a mock StreamWriter that raises on close."""
    writer = AsyncMock(spec=asyncio.StreamWriter)
    writer.is_closing.return_value = False
    writer.write = MagicMock()
    writer.drain = AsyncMock(side_effect=BrokenPipeError("pipe broken"))
    writer.close = MagicMock(side_effect=OSError("close failed"))
    writer.wait_closed = AsyncMock()
    return writer


# ---------------------------------------------------------------------------
# CleanupResult and CleanupSummary tests
# ---------------------------------------------------------------------------


class TestCleanupResult:
    """Tests for immutable CleanupResult data model."""

    def test_successful_result(self) -> None:
        result = CleanupResult(
            resource_id="ssh-chan-1",
            resource_type=ResourceType.SSH_CHANNEL,
            success=True,
            error=None,
            bytes_flushed=1024,
        )
        assert result.success is True
        assert result.error is None
        assert result.bytes_flushed == 1024
        assert result.resource_type == ResourceType.SSH_CHANNEL

    def test_failed_result(self) -> None:
        result = CleanupResult(
            resource_id="sock-1",
            resource_type=ResourceType.SOCKET_WRITER,
            success=False,
            error="Connection reset",
            bytes_flushed=0,
        )
        assert result.success is False
        assert result.error == "Connection reset"

    def test_result_is_frozen(self) -> None:
        result = CleanupResult(
            resource_id="x",
            resource_type=ResourceType.SSH_CHANNEL,
            success=True,
            error=None,
            bytes_flushed=0,
        )
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    def test_result_has_timestamp(self) -> None:
        result = CleanupResult(
            resource_id="x",
            resource_type=ResourceType.SSH_CHANNEL,
            success=True,
            error=None,
            bytes_flushed=0,
        )
        assert isinstance(result.timestamp, datetime)


class TestCleanupSummary:
    """Tests for immutable CleanupSummary aggregate."""

    def test_summary_counts(self) -> None:
        results = (
            CleanupResult(
                resource_id="a",
                resource_type=ResourceType.SSH_CHANNEL,
                success=True,
                error=None,
                bytes_flushed=100,
            ),
            CleanupResult(
                resource_id="b",
                resource_type=ResourceType.SOCKET_WRITER,
                success=False,
                error="timeout",
                bytes_flushed=0,
            ),
            CleanupResult(
                resource_id="c",
                resource_type=ResourceType.IO_BUFFER,
                success=True,
                error=None,
                bytes_flushed=50,
            ),
        )
        summary = CleanupSummary(
            event_id="disconnect-1",
            results=results,
        )
        assert summary.total_resources == 3
        assert summary.successful == 2
        assert summary.failed == 1
        assert summary.total_bytes_flushed == 150

    def test_empty_summary(self) -> None:
        summary = CleanupSummary(
            event_id="disconnect-2",
            results=(),
        )
        assert summary.total_resources == 0
        assert summary.successful == 0
        assert summary.failed == 0
        assert summary.total_bytes_flushed == 0

    def test_summary_is_frozen(self) -> None:
        summary = CleanupSummary(event_id="x", results=())
        with pytest.raises(AttributeError):
            summary.event_id = "y"  # type: ignore[misc]

    def test_all_succeeded(self) -> None:
        results = (
            CleanupResult(
                resource_id="a",
                resource_type=ResourceType.SSH_CHANNEL,
                success=True,
                error=None,
                bytes_flushed=0,
            ),
        )
        summary = CleanupSummary(event_id="ok", results=results)
        assert summary.all_succeeded is True

    def test_not_all_succeeded(self) -> None:
        results = (
            CleanupResult(
                resource_id="a",
                resource_type=ResourceType.SSH_CHANNEL,
                success=False,
                error="err",
                bytes_flushed=0,
            ),
        )
        summary = CleanupSummary(event_id="fail", results=results)
        assert summary.all_succeeded is False


# ---------------------------------------------------------------------------
# ResourceType tests
# ---------------------------------------------------------------------------


class TestResourceType:
    """Tests for ResourceType enum."""

    def test_all_types_exist(self) -> None:
        assert ResourceType.SSH_CHANNEL.value == "ssh_channel"
        assert ResourceType.SOCKET_WRITER.value == "socket_writer"
        assert ResourceType.IO_BUFFER.value == "io_buffer"


# ---------------------------------------------------------------------------
# DisconnectCleanupHandler tests
# ---------------------------------------------------------------------------


class TestDisconnectCleanupHandler:
    """Tests for the disconnect event cleanup coordinator."""

    @pytest.mark.asyncio
    async def test_register_and_cleanup_ssh_channel(self) -> None:
        """Registering an SSH channel and cleaning up closes it."""
        handler = DisconnectCleanupHandler()
        channel = FakeSSHChannel()

        handler.register_ssh_channel("chan-1", channel)
        summary = await handler.cleanup_all()

        assert channel.close_called
        assert summary.total_resources == 1
        assert summary.successful == 1

    @pytest.mark.asyncio
    async def test_register_and_cleanup_socket_writer(self) -> None:
        """Registering a socket writer and cleaning up closes it."""
        handler = DisconnectCleanupHandler()
        writer = _make_writer_mock()

        handler.register_socket_writer("sock-1", writer)
        summary = await handler.cleanup_all()

        writer.close.assert_called()
        writer.wait_closed.assert_awaited()
        assert summary.total_resources == 1
        assert summary.successful == 1

    @pytest.mark.asyncio
    async def test_flush_ssh_channel_buffers_on_cleanup(self) -> None:
        """Cleanup flushes pending stdout/stderr from SSH channel."""
        handler = DisconnectCleanupHandler()
        channel = FakeSSHChannel(
            has_stdout=True,
            stdout_data=b"pending output\n",
            has_stderr=True,
            stderr_data=b"pending error\n",
        )

        handler.register_ssh_channel("chan-2", channel)
        summary = await handler.cleanup_all()

        assert summary.results[0].bytes_flushed > 0
        assert channel.close_called

    @pytest.mark.asyncio
    async def test_flush_socket_writer_on_cleanup(self) -> None:
        """Cleanup drains pending writes from socket writer."""
        handler = DisconnectCleanupHandler()
        writer = _make_writer_mock()

        handler.register_socket_writer("sock-2", writer)
        summary = await handler.cleanup_all()

        writer.drain.assert_awaited()

    @pytest.mark.asyncio
    async def test_cleanup_order_buffers_then_channels_then_sockets(
        self,
    ) -> None:
        """Cleanup follows deterministic order: buffers, channels, sockets."""
        handler = DisconnectCleanupHandler()
        order: list[str] = []

        channel = FakeSSHChannel()
        original_close = channel.close

        def tracked_channel_close() -> None:
            order.append("ssh_channel")
            original_close()

        channel.close = tracked_channel_close  # type: ignore[assignment]

        writer = _make_writer_mock()

        async def tracked_writer_close() -> None:
            order.append("socket_writer")

        writer.wait_closed = tracked_writer_close

        handler.register_ssh_channel("chan", channel)
        handler.register_socket_writer("sock", writer)

        await handler.cleanup_all()

        # SSH channels should be cleaned before socket writers
        assert order.index("ssh_channel") < order.index("socket_writer")

    @pytest.mark.asyncio
    async def test_error_isolation_one_failure_does_not_block_others(
        self,
    ) -> None:
        """A failing cleanup does not prevent other resources from cleaning."""
        handler = DisconnectCleanupHandler()

        bad_channel = FakeSSHChannelRaising()
        good_channel = FakeSSHChannel()

        handler.register_ssh_channel("bad", bad_channel)
        handler.register_ssh_channel("good", good_channel)

        summary = await handler.cleanup_all()

        assert good_channel.close_called
        assert summary.total_resources == 2
        assert summary.failed >= 1
        assert summary.successful >= 1

    @pytest.mark.asyncio
    async def test_cleanup_all_is_idempotent(self) -> None:
        """Calling cleanup_all() twice does not re-close resources."""
        handler = DisconnectCleanupHandler()
        channel = FakeSSHChannel()

        handler.register_ssh_channel("chan", channel)
        summary1 = await handler.cleanup_all()
        summary2 = await handler.cleanup_all()

        assert summary1.total_resources == 1
        assert summary2.total_resources == 0  # already cleaned

    @pytest.mark.asyncio
    async def test_unregister_prevents_cleanup(self) -> None:
        """Unregistered resources are not cleaned up."""
        handler = DisconnectCleanupHandler()
        channel = FakeSSHChannel()

        handler.register_ssh_channel("chan", channel)
        handler.unregister("chan")
        summary = await handler.cleanup_all()

        assert not channel.close_called
        assert summary.total_resources == 0

    @pytest.mark.asyncio
    async def test_cleanup_for_client_targets_specific_client(self) -> None:
        """cleanup_for_client only cleans resources tagged to that client."""
        handler = DisconnectCleanupHandler()
        chan_a = FakeSSHChannel()
        chan_b = FakeSSHChannel()

        handler.register_ssh_channel("chan-a", chan_a, client_id="client-1")
        handler.register_ssh_channel("chan-b", chan_b, client_id="client-2")

        summary = await handler.cleanup_for_client("client-1")

        assert chan_a.close_called
        assert not chan_b.close_called
        assert summary.total_resources == 1

    @pytest.mark.asyncio
    async def test_event_bus_integration_triggers_cleanup_on_disconnect(
        self,
    ) -> None:
        """CLIENT_DISCONNECTED_EVENT triggers cleanup for that client."""
        bus = EventBus()
        handler = DisconnectCleanupHandler(event_bus=bus)
        channel = FakeSSHChannel()

        handler.register_ssh_channel("chan-1", channel, client_id="cli-abc")
        await handler.start()

        # Emit disconnect event
        await bus.emit(
            Event(
                event_type=CLIENT_DISCONNECTED_EVENT,
                payload={"client_id": "cli-abc"},
            )
        )

        # Allow async handlers to run
        await asyncio.sleep(0.05)

        assert channel.close_called

        await handler.stop()

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up_on_exit(self) -> None:
        """Async context manager calls cleanup_all on exit."""
        channel = FakeSSHChannel()

        async with DisconnectCleanupHandler() as handler:
            handler.register_ssh_channel("chan", channel)

        assert channel.close_called

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up_on_exception(self) -> None:
        """Async context manager cleans up even when exception occurs."""
        channel = FakeSSHChannel()

        with pytest.raises(RuntimeError, match="test error"):
            async with DisconnectCleanupHandler() as handler:
                handler.register_ssh_channel("chan", channel)
                raise RuntimeError("test error")

        assert channel.close_called

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self) -> None:
        """start() subscribes to events, stop() unsubscribes and cleans."""
        bus = EventBus()
        handler = DisconnectCleanupHandler(event_bus=bus)
        channel = FakeSSHChannel()
        handler.register_ssh_channel("chan", channel, client_id="c1")

        await handler.start()
        assert bus.has_subscribers(CLIENT_DISCONNECTED_EVENT)

        await handler.stop()
        assert not bus.has_subscribers(CLIENT_DISCONNECTED_EVENT)
        assert channel.close_called

    @pytest.mark.asyncio
    async def test_double_register_replaces(self) -> None:
        """Re-registering the same resource_id replaces the old one."""
        handler = DisconnectCleanupHandler()
        old_channel = FakeSSHChannel()
        new_channel = FakeSSHChannel()

        handler.register_ssh_channel("chan", old_channel)
        handler.register_ssh_channel("chan", new_channel)

        summary = await handler.cleanup_all()

        # Only the new channel should be cleaned
        assert new_channel.close_called
        assert not old_channel.close_called
        assert summary.total_resources == 1

    @pytest.mark.asyncio
    async def test_resource_count_property(self) -> None:
        """resource_count reflects current registered resources."""
        handler = DisconnectCleanupHandler()
        assert handler.resource_count == 0

        handler.register_ssh_channel("a", FakeSSHChannel())
        assert handler.resource_count == 1

        handler.register_socket_writer("b", _make_writer_mock())
        assert handler.resource_count == 2

        handler.unregister("a")
        assert handler.resource_count == 1

    @pytest.mark.asyncio
    async def test_broken_writer_cleanup_captures_error(self) -> None:
        """A broken writer that raises on close has its error captured."""
        handler = DisconnectCleanupHandler()
        writer = _make_broken_writer_mock()

        handler.register_socket_writer("broken", writer)
        summary = await handler.cleanup_all()

        assert summary.total_resources == 1
        assert summary.failed == 1
        assert summary.results[0].error is not None


# ---------------------------------------------------------------------------
# SSHChannelGuard tests
# ---------------------------------------------------------------------------


class TestSSHChannelGuard:
    """Tests for the SSH channel context manager."""

    @pytest.mark.asyncio
    async def test_guard_closes_channel_on_exit(self) -> None:
        """Exiting the guard closes the SSH channel."""
        channel = FakeSSHChannel()

        async with SSHChannelGuard(channel=channel, resource_id="g1"):
            assert not channel.close_called

        assert channel.close_called

    @pytest.mark.asyncio
    async def test_guard_flushes_buffers_before_close(self) -> None:
        """Guard drains pending stdout/stderr before closing."""
        channel = FakeSSHChannel(
            has_stdout=True,
            stdout_data=b"remaining output\n",
        )

        async with SSHChannelGuard(
            channel=channel, resource_id="g2"
        ) as guard:
            pass

        # Channel should be closed and buffers should have been read
        assert channel.close_called

    @pytest.mark.asyncio
    async def test_guard_returns_channel_on_enter(self) -> None:
        """Entering the guard returns the underlying channel."""
        channel = FakeSSHChannel()

        async with SSHChannelGuard(
            channel=channel, resource_id="g3"
        ) as guarded:
            assert guarded is channel

    @pytest.mark.asyncio
    async def test_guard_cleans_up_on_exception(self) -> None:
        """Guard cleans up even when an exception occurs."""
        channel = FakeSSHChannel()

        with pytest.raises(ValueError, match="oops"):
            async with SSHChannelGuard(
                channel=channel, resource_id="g4"
            ):
                raise ValueError("oops")

        assert channel.close_called

    @pytest.mark.asyncio
    async def test_guard_with_cleanup_handler_registration(self) -> None:
        """Guard registers with DisconnectCleanupHandler and unregisters on exit."""
        handler = DisconnectCleanupHandler()
        channel = FakeSSHChannel()

        async with SSHChannelGuard(
            channel=channel,
            resource_id="g5",
            cleanup_handler=handler,
        ):
            assert handler.resource_count == 1

        assert handler.resource_count == 0
        assert channel.close_called

    @pytest.mark.asyncio
    async def test_guard_safe_on_already_closed_channel(self) -> None:
        """Guard handles already-closed channels without error."""
        channel = FakeSSHChannel(is_closed=True)

        async with SSHChannelGuard(
            channel=channel, resource_id="g6"
        ):
            pass

        # Should not raise, and close_called is tracked
        # (close may still be called for consistency, even if already closed)

    @pytest.mark.asyncio
    async def test_guard_captures_close_error(self) -> None:
        """Guard captures errors from channel.close() without propagating."""
        channel = FakeSSHChannelRaising()

        # Should not raise
        async with SSHChannelGuard(
            channel=channel, resource_id="g7"
        ):
            pass

    @pytest.mark.asyncio
    async def test_guard_cleanup_result(self) -> None:
        """Guard exposes cleanup_result after exit."""
        channel = FakeSSHChannel(
            has_stdout=True,
            stdout_data=b"data",
        )

        guard = SSHChannelGuard(channel=channel, resource_id="g8")
        async with guard:
            pass

        result = guard.cleanup_result
        assert result is not None
        assert result.success is True
        assert result.resource_type == ResourceType.SSH_CHANNEL


# ---------------------------------------------------------------------------
# atexit integration tests
# ---------------------------------------------------------------------------


class TestAtexitRegistration:
    """Tests for process-level atexit cleanup registration."""

    @pytest.mark.asyncio
    async def test_register_atexit_handler(self) -> None:
        """DisconnectCleanupHandler can register an atexit handler."""
        handler = DisconnectCleanupHandler()
        channel = FakeSSHChannel()
        handler.register_ssh_channel("chan", channel)

        with patch.object(atexit, "register") as mock_register:
            handler.register_atexit()
            mock_register.assert_called_once()

    @pytest.mark.asyncio
    async def test_atexit_handler_cleans_remaining_resources(self) -> None:
        """The atexit handler runs synchronous cleanup of remaining resources."""
        handler = DisconnectCleanupHandler()
        channel = FakeSSHChannel()
        handler.register_ssh_channel("chan", channel)

        # Simulate what the atexit handler does
        handler.sync_cleanup_all()

        assert channel.close_called

    @pytest.mark.asyncio
    async def test_atexit_idempotent_registration(self) -> None:
        """register_atexit is idempotent -- second call is no-op."""
        handler = DisconnectCleanupHandler()
        with patch.object(atexit, "register") as mock_register:
            handler.register_atexit()
            handler.register_atexit()
            mock_register.assert_called_once()


# ---------------------------------------------------------------------------
# Edge case and coverage tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional tests for edge cases and branch coverage."""

    @pytest.mark.asyncio
    async def test_cleanup_already_closed_ssh_channel(self) -> None:
        """Cleaning an already-closed SSH channel is a success with 0 bytes."""
        handler = DisconnectCleanupHandler()
        channel = FakeSSHChannel(is_closed=True)

        handler.register_ssh_channel("closed-chan", channel)
        summary = await handler.cleanup_all()

        assert summary.total_resources == 1
        assert summary.successful == 1
        assert summary.results[0].bytes_flushed == 0

    @pytest.mark.asyncio
    async def test_cleanup_already_closing_socket(self) -> None:
        """Cleaning a socket that is_closing returns early success."""
        handler = DisconnectCleanupHandler()
        writer = _make_writer_mock()
        writer.is_closing.return_value = True

        handler.register_socket_writer("closing-sock", writer)
        summary = await handler.cleanup_all()

        assert summary.total_resources == 1
        assert summary.successful == 1
        writer.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_io_buffer(self) -> None:
        """I/O buffer cleanup calls flush() and close()."""
        handler = DisconnectCleanupHandler()
        buf = MagicMock()

        handler.register_io_buffer("buf-1", buf)
        summary = await handler.cleanup_all()

        buf.flush.assert_called_once()
        buf.close.assert_called_once()
        assert summary.total_resources == 1
        assert summary.successful == 1

    @pytest.mark.asyncio
    async def test_cleanup_io_buffer_error(self) -> None:
        """I/O buffer that raises on flush has error captured."""
        handler = DisconnectCleanupHandler()
        buf = MagicMock()
        buf.flush.side_effect = IOError("flush failed")

        handler.register_io_buffer("bad-buf", buf)
        summary = await handler.cleanup_all()

        assert summary.total_resources == 1
        assert summary.failed == 1
        assert "flush failed" in (summary.results[0].error or "")

    @pytest.mark.asyncio
    async def test_cleanup_for_nonexistent_client(self) -> None:
        """Cleanup for a nonexistent client returns empty summary."""
        handler = DisconnectCleanupHandler()
        handler.register_ssh_channel("chan", FakeSSHChannel(), client_id="a")

        summary = await handler.cleanup_for_client("nonexistent")

        assert summary.total_resources == 0

    @pytest.mark.asyncio
    async def test_start_without_event_bus(self) -> None:
        """start() without event_bus is a no-op but sets started flag."""
        handler = DisconnectCleanupHandler()
        await handler.start()
        # Should not raise; started flag is set internally

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self) -> None:
        """Calling start() twice does not double-subscribe."""
        bus = EventBus()
        handler = DisconnectCleanupHandler(event_bus=bus)
        await handler.start()
        await handler.start()  # should not raise
        assert bus.subscriber_count(CLIENT_DISCONNECTED_EVENT) == 1
        await handler.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        """Calling stop() twice does not error."""
        handler = DisconnectCleanupHandler()
        await handler.stop()
        await handler.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_disconnect_event_missing_client_id(self) -> None:
        """Disconnect event without client_id logs warning but does not crash."""
        bus = EventBus()
        handler = DisconnectCleanupHandler(event_bus=bus)
        await handler.start()

        # Emit event with empty payload
        await bus.emit(
            Event(
                event_type=CLIENT_DISCONNECTED_EVENT,
                payload={},
            )
        )
        await asyncio.sleep(0.05)

        await handler.stop()

    @pytest.mark.asyncio
    async def test_sync_cleanup_all_with_all_resource_types(self) -> None:
        """sync_cleanup_all handles SSH channels, sockets, and buffers."""
        handler = DisconnectCleanupHandler()
        channel = FakeSSHChannel(
            has_stdout=True,
            stdout_data=b"data",
        )
        writer = _make_writer_mock()
        buf = MagicMock()

        handler.register_ssh_channel("chan", channel)
        handler.register_socket_writer("sock", writer)
        handler.register_io_buffer("buf", buf)

        handler.sync_cleanup_all()

        assert channel.close_called
        buf.flush.assert_called_once()
        buf.close.assert_called_once()
        assert handler.resource_count == 0

    @pytest.mark.asyncio
    async def test_sync_cleanup_with_raising_resource(self) -> None:
        """sync_cleanup_all handles errors without propagating."""
        handler = DisconnectCleanupHandler()
        channel = FakeSSHChannelRaising()

        handler.register_ssh_channel("bad", channel)
        # Should not raise
        handler.sync_cleanup_all()
        assert handler.resource_count == 0

    @pytest.mark.asyncio
    async def test_sync_cleanup_empty(self) -> None:
        """sync_cleanup_all with no resources is a no-op."""
        handler = DisconnectCleanupHandler()
        handler.sync_cleanup_all()  # should not raise

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self) -> None:
        """Unregistering a nonexistent resource is a safe no-op."""
        handler = DisconnectCleanupHandler()
        handler.unregister("doesnt-exist")  # should not raise
        assert handler.resource_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_with_event_bus(self) -> None:
        """Context manager with event_bus calls start and cleans on exit."""
        bus = EventBus()
        channel = FakeSSHChannel()

        async with DisconnectCleanupHandler(event_bus=bus) as handler:
            handler.register_ssh_channel("chan", channel, client_id="c1")
            assert bus.has_subscribers(CLIENT_DISCONNECTED_EVENT)

        assert channel.close_called
        assert not bus.has_subscribers(CLIENT_DISCONNECTED_EVENT)

    @pytest.mark.asyncio
    async def test_validation_empty_resource_id(self) -> None:
        """CleanupResult rejects empty resource_id."""
        with pytest.raises(ValueError, match="resource_id must not be empty"):
            CleanupResult(
                resource_id="",
                resource_type=ResourceType.SSH_CHANNEL,
                success=True,
                error=None,
                bytes_flushed=0,
            )

    @pytest.mark.asyncio
    async def test_validation_negative_bytes(self) -> None:
        """CleanupResult rejects negative bytes_flushed."""
        with pytest.raises(ValueError, match="bytes_flushed must not be negative"):
            CleanupResult(
                resource_id="x",
                resource_type=ResourceType.SSH_CHANNEL,
                success=True,
                error=None,
                bytes_flushed=-1,
            )
