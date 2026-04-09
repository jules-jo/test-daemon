"""Async context manager for SSH channel lifecycle with buffer flushing.

Wraps an SSH channel handle in a context manager that guarantees:

1. On enter: optionally registers the channel with a
   DisconnectCleanupHandler for event-driven cleanup.
2. On exit: flushes any pending stdout/stderr data from the channel,
   closes the channel, and unregisters from the cleanup handler.

This provides deterministic, scope-based cleanup for SSH channels,
ensuring that no buffered output is lost and that transport resources
are released promptly when the channel is no longer needed.

The guard is designed for use cases where a channel has a clear
ownership scope (e.g., a single command execution). For longer-lived
channels managed by the daemon, use the DisconnectCleanupHandler
directly.

Usage::

    from jules_daemon.cleanup.channel_guard import SSHChannelGuard

    async with SSHChannelGuard(
        channel=paramiko_channel,
        resource_id="run-abc-channel",
        cleanup_handler=handler,
    ) as channel:
        output = await read_ssh_output(channel)
        process(output)
    # Channel is flushed and closed here, even on exception.

    # Access the cleanup result after exit:
    result = guard.cleanup_result
    if result and result.success:
        log_audit(result)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from jules_daemon.cleanup.resource_types import (
    MAX_FLUSH_BYTES,
    CleanupResult,
    ResourceType,
)

__all__ = [
    "SSHChannelGuard",
]

logger = logging.getLogger(__name__)

_MAX_FLUSH_BYTES = MAX_FLUSH_BYTES


class SSHChannelGuard:
    """Async context manager for SSH channel lifecycle management.

    Ensures that the SSH channel is flushed and closed when the context
    exits, regardless of whether the exit is normal or due to an
    exception. Optionally integrates with a DisconnectCleanupHandler
    for additional event-driven cleanup protection.

    The guard does NOT suppress exceptions: if an exception occurs
    inside the ``async with`` block, it propagates after cleanup.

    Channel close errors are logged but not propagated (cleanup should
    not mask the original exception).

    Args:
        channel:         The SSH channel object to guard. Must support
                         recv_ready(), recv(), recv_stderr_ready(),
                         recv_stderr(), close(), and a ``closed`` property.
        resource_id:     Unique identifier for this channel (used in logs
                         and cleanup handler registration).
        cleanup_handler: Optional DisconnectCleanupHandler to register
                         with on enter and unregister on exit.
        client_id:       Optional client ID for cleanup handler targeting.
    """

    def __init__(
        self,
        *,
        channel: Any,
        resource_id: str,
        cleanup_handler: Any | None = None,
        client_id: str | None = None,
    ) -> None:
        self._channel = channel
        self._resource_id = resource_id
        self._cleanup_handler = cleanup_handler
        self._client_id = client_id
        self._cleanup_result: CleanupResult | None = None

    @property
    def cleanup_result(self) -> CleanupResult | None:
        """The cleanup result after context exit, or None if not yet exited."""
        return self._cleanup_result

    async def __aenter__(self) -> Any:
        """Enter the guard: register with cleanup handler (if provided).

        Returns:
            The underlying SSH channel object, for direct use.
        """
        if self._cleanup_handler is not None:
            self._cleanup_handler.register_ssh_channel(
                self._resource_id,
                self._channel,
                client_id=self._client_id,
            )

        return self._channel

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the guard: flush buffers, close channel, unregister.

        Cleanup errors are logged but never propagated. The original
        exception (if any) always propagates unchanged.
        """
        self._cleanup_result = await self._perform_cleanup()

        # Unregister from cleanup handler (already cleaned up locally)
        if self._cleanup_handler is not None:
            self._cleanup_handler.unregister(self._resource_id)

    async def _perform_cleanup(self) -> CleanupResult:
        """Flush pending I/O and close the channel.

        Returns:
            Immutable CleanupResult capturing the outcome.
        """
        bytes_flushed = 0

        try:
            # Skip cleanup if already closed
            if hasattr(self._channel, "closed") and self._channel.closed:
                return CleanupResult(
                    resource_id=self._resource_id,
                    resource_type=ResourceType.SSH_CHANNEL,
                    success=True,
                    error=None,
                    bytes_flushed=0,
                )

            # Flush stdout buffer
            bytes_flushed += await self._flush_stream("stdout")

            # Flush stderr buffer
            bytes_flushed += await self._flush_stream("stderr")

            # Close the channel
            if hasattr(self._channel, "close"):
                try:
                    await asyncio.to_thread(self._channel.close)
                except Exception as close_exc:
                    logger.debug(
                        "Channel close error for %s: %s",
                        self._resource_id,
                        close_exc,
                    )

            logger.info(
                "SSHChannelGuard %s: flushed %d bytes, channel closed",
                self._resource_id,
                bytes_flushed,
            )
            return CleanupResult(
                resource_id=self._resource_id,
                resource_type=ResourceType.SSH_CHANNEL,
                success=True,
                error=None,
                bytes_flushed=bytes_flushed,
            )

        except Exception as exc:
            logger.warning(
                "SSHChannelGuard %s cleanup error: %s: %s",
                self._resource_id,
                type(exc).__name__,
                exc,
            )
            return CleanupResult(
                resource_id=self._resource_id,
                resource_type=ResourceType.SSH_CHANNEL,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                bytes_flushed=bytes_flushed,
            )

    async def _flush_stream(self, stream: str) -> int:
        """Flush pending data from one channel stream.

        Args:
            stream: Either "stdout" or "stderr".

        Returns:
            Number of bytes flushed.
        """
        try:
            if stream == "stdout":
                ready_fn = getattr(self._channel, "recv_ready", None)
                recv_fn = getattr(self._channel, "recv", None)
            else:
                ready_fn = getattr(
                    self._channel, "recv_stderr_ready", None
                )
                recv_fn = getattr(
                    self._channel, "recv_stderr", None
                )

            if ready_fn is None or recv_fn is None:
                return 0

            def _drain() -> int:
                total = 0
                while total < _MAX_FLUSH_BYTES:
                    try:
                        if not ready_fn():
                            break
                        chunk = recv_fn(_MAX_FLUSH_BYTES - total)
                        if not chunk:
                            break
                        total += len(chunk)
                    except Exception as drain_exc:
                        logger.debug(
                            "SSHChannelGuard %s: drain recv error on %s: %s",
                            self._resource_id,
                            stream,
                            drain_exc,
                        )
                        break
                return total

            return await asyncio.to_thread(_drain)

        except Exception as exc:
            logger.debug(
                "SSHChannelGuard %s: error flushing %s: %s",
                self._resource_id,
                stream,
                exc,
            )
            return 0
