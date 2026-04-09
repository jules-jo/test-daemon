"""Non-blocking async SSH output reader.

Reads available stdout/stderr bytes from an SSH channel without blocking
the asyncio event loop. Uses a Protocol-based channel handle so the reader
is decoupled from any specific SSH library (paramiko, asyncssh, etc.).

Usage:
    output = await read_ssh_output(channel)
    if output.has_data:
        process(output.stdout, output.stderr)

Design decisions:
    - Protocol-based channel handle for library independence
    - Frozen dataclass result for immutability
    - asyncio.to_thread() for offloading potentially blocking recv() calls
    - Readiness checks before recv to avoid unnecessary thread dispatch
    - max_bytes cap to prevent unbounded memory allocation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

__all__ = ["SSHChannelHandle", "SSHOutput", "read_ssh_output"]


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_MAX_BYTES = 65536


# ---------------------------------------------------------------------------
# Protocol: SSH channel handle
# ---------------------------------------------------------------------------


@runtime_checkable
class SSHChannelHandle(Protocol):
    """Protocol defining the SSH channel interface for output reading.

    This Protocol is compatible with paramiko's Channel and can be adapted
    for asyncssh or other SSH libraries via a thin wrapper.

    Methods mirror the subset of paramiko.Channel needed for non-blocking
    output collection.

    Thread-safety contract:
        - recv() MUST return immediately (without blocking) when
          recv_ready() has returned True. Likewise for recv_stderr()
          and recv_stderr_ready().
        - closed, eof_received(), exit_status_is_ready(), and
          get_exit_status() MUST be O(1) in-memory reads that do not
          perform I/O. They may be called from any thread.
    """

    def recv_ready(self) -> bool:
        """Return True if stdout data is available to read."""
        ...

    def recv(self, nbytes: int) -> bytes:
        """Read up to nbytes of stdout data.

        Must return immediately when recv_ready() is True.
        """
        ...

    def recv_stderr_ready(self) -> bool:
        """Return True if stderr data is available to read."""
        ...

    def recv_stderr(self, nbytes: int) -> bytes:
        """Read up to nbytes of stderr data.

        Must return immediately when recv_stderr_ready() is True.
        """
        ...

    @property
    def closed(self) -> bool:
        """Return True if the channel is closed. Must be O(1)."""
        ...

    def eof_received(self) -> bool:
        """Return True if the remote side sent EOF. Must be O(1)."""
        ...

    def exit_status_is_ready(self) -> bool:
        """Return True if the remote process exit status is available. Must be O(1)."""
        ...

    def get_exit_status(self) -> int:
        """Return the remote process exit status. Must be O(1)."""
        ...


# ---------------------------------------------------------------------------
# Immutable result type
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SSHOutput:
    """Immutable snapshot of SSH channel output at a point in time.

    Attributes:
        stdout: Raw bytes read from the channel's stdout stream.
        stderr: Raw bytes read from the channel's stderr stream.
        is_eof: True if the remote side has sent EOF.
        exit_code: Remote process exit code, or None if not yet exited.
        channel_closed: True if the SSH channel is closed.
        timestamp: UTC datetime when this snapshot was captured.
    """

    stdout: bytes
    stderr: bytes
    is_eof: bool = False
    exit_code: int | None = None
    channel_closed: bool = False
    # Default factory provides a timestamp for direct construction (e.g. tests).
    # read_ssh_output() always provides an explicit timestamp post-gather.
    timestamp: datetime = field(default_factory=_now_utc)

    @property
    def has_data(self) -> bool:
        """True if any stdout or stderr bytes were read."""
        return len(self.stdout) > 0 or len(self.stderr) > 0

    @property
    def total_bytes(self) -> int:
        """Total number of bytes across both streams."""
        return len(self.stdout) + len(self.stderr)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_stdout_sync(channel: SSHChannelHandle, max_bytes: int) -> bytes:
    """Synchronously read available stdout bytes from the channel.

    Only calls recv() if data is ready, ensuring no blocking wait.
    """
    if not channel.recv_ready():
        return b""
    return channel.recv(max_bytes)


def _read_stderr_sync(channel: SSHChannelHandle, max_bytes: int) -> bytes:
    """Synchronously read available stderr bytes from the channel.

    Only calls recv_stderr() if data is ready, ensuring no blocking wait.
    """
    if not channel.recv_stderr_ready():
        return b""
    return channel.recv_stderr(max_bytes)


@dataclass(frozen=True)
class _ChannelMetadata:
    """Internal snapshot of channel metadata captured in thread pool."""

    is_closed: bool
    is_eof: bool
    exit_code: int | None


def _read_metadata_sync(channel: SSHChannelHandle) -> _ChannelMetadata:
    """Synchronously capture channel metadata.

    Called in thread pool alongside recv reads so the event loop
    is never blocked, even if a future backend does minor I/O in
    these accessors.
    """
    is_closed = channel.closed
    is_eof = channel.eof_received()

    exit_code: int | None = None
    if channel.exit_status_is_ready():
        exit_code = channel.get_exit_status()

    return _ChannelMetadata(
        is_closed=is_closed,
        is_eof=is_eof,
        exit_code=exit_code,
    )


# ---------------------------------------------------------------------------
# Public async reader
# ---------------------------------------------------------------------------


async def read_ssh_output(
    channel: SSHChannelHandle,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> SSHOutput:
    """Read available SSH output without blocking the event loop.

    Checks channel readiness, then offloads the actual recv() calls to a
    thread pool via asyncio.to_thread() so the asyncio event loop is never
    blocked -- even if the underlying SSH library does brief I/O during recv().

    Args:
        channel: An object satisfying the SSHChannelHandle protocol.
        max_bytes: Maximum bytes to read per stream (stdout and stderr
            each get up to max_bytes). Must be positive.

    Returns:
        An immutable SSHOutput snapshot with the latest available bytes.

    Raises:
        ValueError: If max_bytes is not positive.
        OSError: If the channel raises an OS-level error during recv.
        TimeoutError: If the channel raises a timeout during recv.
    """
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")

    # Offload all channel I/O to thread pool concurrently.
    stdout_bytes, stderr_bytes, metadata = await asyncio.gather(
        asyncio.to_thread(_read_stdout_sync, channel, max_bytes),
        asyncio.to_thread(_read_stderr_sync, channel, max_bytes),
        asyncio.to_thread(_read_metadata_sync, channel),
    )

    return SSHOutput(
        stdout=stdout_bytes,
        stderr=stderr_bytes,
        is_eof=metadata.is_eof,
        exit_code=metadata.exit_code,
        channel_closed=metadata.is_closed,
        timestamp=_now_utc(),
    )
