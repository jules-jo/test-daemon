"""Tests for async SSH output reader.

Covers:
    - SSHOutput frozen dataclass behavior and defaults
    - SSHChannelHandle Protocol compliance
    - read_ssh_output() non-blocking reads for stdout, stderr, both, and neither
    - read_ssh_output() handling of closed channels and exit status
    - read_ssh_output() max_bytes capping
    - read_ssh_output() error propagation from broken channels
    - read_ssh_output() with EOF detection
    - Concurrency: reader does not block the event loop
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import pytest

from jules_daemon.ssh.reader import (
    SSHChannelHandle,
    SSHOutput,
    read_ssh_output,
)


# ---------------------------------------------------------------------------
# Fake SSH channel for testing (implements SSHChannelHandle Protocol)
# ---------------------------------------------------------------------------


class FakeSSHChannel:
    """Fake SSH channel that implements the SSHChannelHandle protocol."""

    def __init__(
        self,
        *,
        stdout_buffer: bytes = b"",
        stderr_buffer: bytes = b"",
        is_closed: bool = False,
        exit_status: int | None = None,
        exit_status_ready: bool = False,
        is_eof: bool = False,
        raise_on_recv: Exception | None = None,
    ) -> None:
        self._stdout_buffer = stdout_buffer
        self._stderr_buffer = stderr_buffer
        self._closed = is_closed
        self._exit_status = exit_status
        self._exit_status_ready = exit_status_ready
        self._is_eof = is_eof
        self._raise_on_recv = raise_on_recv

    def recv_ready(self) -> bool:
        return len(self._stdout_buffer) > 0

    def recv(self, nbytes: int) -> bytes:
        if self._raise_on_recv is not None:
            raise self._raise_on_recv
        chunk = self._stdout_buffer[:nbytes]
        self._stdout_buffer = self._stdout_buffer[nbytes:]
        return chunk

    def recv_stderr_ready(self) -> bool:
        return len(self._stderr_buffer) > 0

    def recv_stderr(self, nbytes: int) -> bytes:
        if self._raise_on_recv is not None:
            raise self._raise_on_recv
        chunk = self._stderr_buffer[:nbytes]
        self._stderr_buffer = self._stderr_buffer[nbytes:]
        return chunk

    @property
    def closed(self) -> bool:
        return self._closed

    def eof_received(self) -> bool:
        return self._is_eof

    def exit_status_is_ready(self) -> bool:
        return self._exit_status_ready

    def get_exit_status(self) -> int:
        if self._exit_status is None:
            raise RuntimeError(
                "get_exit_status called before exit status is ready"
            )
        return self._exit_status


# Verify the fake satisfies the protocol at runtime
assert isinstance(FakeSSHChannel(), SSHChannelHandle)


# ---------------------------------------------------------------------------
# SSHOutput dataclass tests
# ---------------------------------------------------------------------------


class TestSSHOutput:
    def test_create_with_defaults(self) -> None:
        output = SSHOutput(stdout=b"", stderr=b"")
        assert output.stdout == b""
        assert output.stderr == b""
        assert output.is_eof is False
        assert output.exit_code is None
        assert output.channel_closed is False
        assert isinstance(output.timestamp, datetime)

    def test_create_with_data(self) -> None:
        now = datetime.now(timezone.utc)
        output = SSHOutput(
            stdout=b"PASSED test_foo\n",
            stderr=b"warning: deprecated\n",
            is_eof=True,
            exit_code=0,
            channel_closed=False,
            timestamp=now,
        )
        assert output.stdout == b"PASSED test_foo\n"
        assert output.stderr == b"warning: deprecated\n"
        assert output.is_eof is True
        assert output.exit_code == 0
        assert output.timestamp == now

    def test_frozen(self) -> None:
        output = SSHOutput(stdout=b"data", stderr=b"")
        with pytest.raises(FrozenInstanceError):
            output.stdout = b"other"  # type: ignore[misc]

    def test_has_data_property_true(self) -> None:
        output = SSHOutput(stdout=b"something", stderr=b"")
        assert output.has_data is True

    def test_has_data_property_false(self) -> None:
        output = SSHOutput(stdout=b"", stderr=b"")
        assert output.has_data is False

    def test_has_data_stderr_only(self) -> None:
        output = SSHOutput(stdout=b"", stderr=b"err")
        assert output.has_data is True

    def test_total_bytes(self) -> None:
        output = SSHOutput(stdout=b"abc", stderr=b"de")
        assert output.total_bytes == 5

    def test_total_bytes_empty(self) -> None:
        output = SSHOutput(stdout=b"", stderr=b"")
        assert output.total_bytes == 0


# ---------------------------------------------------------------------------
# read_ssh_output() async tests
# ---------------------------------------------------------------------------


class TestReadSSHOutputStdout:
    @pytest.mark.asyncio
    async def test_reads_available_stdout(self) -> None:
        channel = FakeSSHChannel(stdout_buffer=b"hello world")
        result = await read_ssh_output(channel)  # type: ignore[arg-type]
        assert result.stdout == b"hello world"
        assert result.stderr == b""
        assert result.has_data is True

    @pytest.mark.asyncio
    async def test_respects_max_bytes(self) -> None:
        channel = FakeSSHChannel(stdout_buffer=b"0123456789")
        result = await read_ssh_output(channel, max_bytes=5)  # type: ignore[arg-type]
        assert result.stdout == b"01234"
        assert len(result.stdout) <= 5


class TestReadSSHOutputStderr:
    @pytest.mark.asyncio
    async def test_reads_available_stderr(self) -> None:
        channel = FakeSSHChannel(stderr_buffer=b"error output")
        result = await read_ssh_output(channel)  # type: ignore[arg-type]
        assert result.stdout == b""
        assert result.stderr == b"error output"
        assert result.has_data is True

    @pytest.mark.asyncio
    async def test_stderr_respects_max_bytes(self) -> None:
        channel = FakeSSHChannel(stderr_buffer=b"0123456789")
        result = await read_ssh_output(channel, max_bytes=4)  # type: ignore[arg-type]
        assert result.stderr == b"0123"
        assert len(result.stderr) <= 4


class TestReadSSHOutputBoth:
    @pytest.mark.asyncio
    async def test_reads_both_streams(self) -> None:
        channel = FakeSSHChannel(
            stdout_buffer=b"out",
            stderr_buffer=b"err",
        )
        result = await read_ssh_output(channel)  # type: ignore[arg-type]
        assert result.stdout == b"out"
        assert result.stderr == b"err"
        assert result.has_data is True


class TestReadSSHOutputEmpty:
    @pytest.mark.asyncio
    async def test_returns_empty_when_nothing_available(self) -> None:
        channel = FakeSSHChannel()
        result = await read_ssh_output(channel)  # type: ignore[arg-type]
        assert result.stdout == b""
        assert result.stderr == b""
        assert result.has_data is False


class TestReadSSHOutputClosedChannel:
    @pytest.mark.asyncio
    async def test_closed_channel_returns_channel_closed(self) -> None:
        channel = FakeSSHChannel(is_closed=True)
        result = await read_ssh_output(channel)  # type: ignore[arg-type]
        assert result.channel_closed is True
        assert result.has_data is False

    @pytest.mark.asyncio
    async def test_closed_channel_still_reads_buffered_data(self) -> None:
        channel = FakeSSHChannel(
            stdout_buffer=b"final output",
            is_closed=True,
        )
        result = await read_ssh_output(channel)  # type: ignore[arg-type]
        assert result.stdout == b"final output"
        assert result.channel_closed is True
        assert result.has_data is True


class TestReadSSHOutputExitStatus:
    @pytest.mark.asyncio
    async def test_captures_exit_status_when_ready(self) -> None:
        channel = FakeSSHChannel(
            exit_status_ready=True,
            exit_status=0,
        )
        result = await read_ssh_output(channel)  # type: ignore[arg-type]
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_exit_code_none_when_not_ready(self) -> None:
        channel = FakeSSHChannel(
            exit_status_ready=False,
            exit_status=42,
        )
        result = await read_ssh_output(channel)  # type: ignore[arg-type]
        assert result.exit_code is None

    @pytest.mark.asyncio
    async def test_nonzero_exit_captured(self) -> None:
        channel = FakeSSHChannel(
            exit_status_ready=True,
            exit_status=1,
        )
        result = await read_ssh_output(channel)  # type: ignore[arg-type]
        assert result.exit_code == 1


class TestReadSSHOutputEOF:
    @pytest.mark.asyncio
    async def test_eof_detected(self) -> None:
        channel = FakeSSHChannel(is_eof=True)
        result = await read_ssh_output(channel)  # type: ignore[arg-type]
        assert result.is_eof is True

    @pytest.mark.asyncio
    async def test_no_eof_when_stream_open(self) -> None:
        channel = FakeSSHChannel(is_eof=False)
        result = await read_ssh_output(channel)  # type: ignore[arg-type]
        assert result.is_eof is False


class TestReadSSHOutputErrors:
    @pytest.mark.asyncio
    async def test_propagates_oserror(self) -> None:
        channel = FakeSSHChannel(
            stdout_buffer=b"data",
            raise_on_recv=OSError("Connection reset"),
        )
        with pytest.raises(OSError, match="Connection reset"):
            await read_ssh_output(channel)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_propagates_socket_timeout(self) -> None:
        channel = FakeSSHChannel(
            stdout_buffer=b"data",
            raise_on_recv=TimeoutError("timed out"),
        )
        with pytest.raises(TimeoutError, match="timed out"):
            await read_ssh_output(channel)  # type: ignore[arg-type]


class TestReadSSHOutputTimestamp:
    @pytest.mark.asyncio
    async def test_result_has_utc_timestamp(self) -> None:
        channel = FakeSSHChannel()
        before = datetime.now(timezone.utc)
        result = await read_ssh_output(channel)  # type: ignore[arg-type]
        after = datetime.now(timezone.utc)
        assert before <= result.timestamp <= after
        assert result.timestamp.tzinfo is not None


class TestReadSSHOutputNonBlocking:
    @pytest.mark.asyncio
    async def test_does_not_block_event_loop(self) -> None:
        """Verify that read_ssh_output returns promptly even when no data."""
        channel = FakeSSHChannel()
        result = await asyncio.wait_for(
            read_ssh_output(channel),  # type: ignore[arg-type]
            timeout=1.0,
        )
        assert result.has_data is False

    @pytest.mark.asyncio
    async def test_concurrent_reads_work(self) -> None:
        """Multiple concurrent reads should not deadlock."""
        channels = [
            FakeSSHChannel(stdout_buffer=f"data-{i}".encode())
            for i in range(5)
        ]
        results = await asyncio.gather(
            *(read_ssh_output(ch) for ch in channels)  # type: ignore[arg-type]
        )
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.stdout == f"data-{i}".encode()


class TestReadSSHOutputMaxBytesValidation:
    @pytest.mark.asyncio
    async def test_max_bytes_must_be_positive(self) -> None:
        channel = FakeSSHChannel()
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            await read_ssh_output(channel, max_bytes=0)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_max_bytes_negative_raises(self) -> None:
        channel = FakeSSHChannel()
        with pytest.raises(ValueError, match="max_bytes must be positive"):
            await read_ssh_output(channel, max_bytes=-1)  # type: ignore[arg-type]
