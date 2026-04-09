"""Event-driven resource cleanup coordinator for disconnect events.

Orchestrates deterministic teardown of SSH channels, socket connections,
and I/O buffers when a CLI client disconnects or when the daemon shuts
down. Integrates with the EventBus to react to CLIENT_DISCONNECTED_EVENT
automatically.

Cleanup order (deterministic):
    1. Flush and close SSH channels (release remote resources first)
    2. Drain and close socket writers (release IPC connections)
    3. Flush I/O buffers (release any remaining pending data)

Each individual cleanup is error-isolated: a failure in one resource
does not prevent cleanup of the remaining resources. All results are
captured in an immutable CleanupSummary for audit logging.

Supports three usage patterns:

1. **Event-driven**: Subscribe to EventBus for automatic cleanup::

    handler = DisconnectCleanupHandler(event_bus=bus)
    await handler.start()   # subscribes to CLIENT_DISCONNECTED_EVENT
    ...
    await handler.stop()    # unsubscribes, cleans remaining resources

2. **Context manager**: Deterministic scope-based cleanup::

    async with DisconnectCleanupHandler() as handler:
        handler.register_ssh_channel("chan", channel)
        ...
    # All resources cleaned on exit

3. **atexit**: Process-level shutdown safety net::

    handler = DisconnectCleanupHandler()
    handler.register_atexit()  # registers synchronous cleanup at exit
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from jules_daemon.cleanup.resource_types import (
    MAX_FLUSH_BYTES,
    CleanupResult,
    CleanupSummary,
    ResourceType,
)
from jules_daemon.ipc.connection_manager import CLIENT_DISCONNECTED_EVENT
from jules_daemon.ipc.event_bus import Event, EventBus, Subscription

__all__ = [
    "DisconnectCleanupHandler",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: registered resource descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResourceEntry:
    """Internal descriptor for a registered resource.

    Attributes:
        resource_id:   Unique identifier for the resource.
        resource_type: Classification (SSH channel, socket, buffer).
        resource:      The actual resource object (channel, writer, etc.).
        client_id:     Optional client ID for per-client cleanup targeting.
    """

    resource_id: str
    resource_type: ResourceType
    resource: Any
    client_id: str | None = None


# ---------------------------------------------------------------------------
# Internal: cleanup executors (one per resource type)
# ---------------------------------------------------------------------------

_MAX_FLUSH_BYTES = MAX_FLUSH_BYTES


async def _cleanup_ssh_channel(
    entry: _ResourceEntry,
) -> CleanupResult:
    """Flush pending I/O and close an SSH channel.

    Reads any remaining stdout/stderr data from the channel before
    closing it, to prevent data loss. The flush is best-effort: if
    the channel is already closed or raises during read, the close
    is still attempted.

    Args:
        entry: The resource entry containing the SSH channel.

    Returns:
        Immutable CleanupResult capturing outcome and bytes flushed.
    """
    channel = entry.resource
    bytes_flushed = 0

    try:
        # Check if already closed
        if hasattr(channel, "closed") and channel.closed:
            return CleanupResult(
                resource_id=entry.resource_id,
                resource_type=ResourceType.SSH_CHANNEL,
                success=True,
                error=None,
                bytes_flushed=0,
            )

        # Flush stdout buffer
        bytes_flushed += await _flush_channel_stream(
            channel, "stdout"
        )

        # Flush stderr buffer
        bytes_flushed += await _flush_channel_stream(
            channel, "stderr"
        )

        # Close the channel
        if hasattr(channel, "close"):
            await asyncio.to_thread(channel.close)

        logger.info(
            "SSH channel %s cleaned: %d bytes flushed",
            entry.resource_id,
            bytes_flushed,
        )
        return CleanupResult(
            resource_id=entry.resource_id,
            resource_type=ResourceType.SSH_CHANNEL,
            success=True,
            error=None,
            bytes_flushed=bytes_flushed,
        )

    except Exception as exc:
        logger.warning(
            "Error cleaning SSH channel %s: %s: %s",
            entry.resource_id,
            type(exc).__name__,
            exc,
        )
        return CleanupResult(
            resource_id=entry.resource_id,
            resource_type=ResourceType.SSH_CHANNEL,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            bytes_flushed=bytes_flushed,
        )


async def _flush_channel_stream(
    channel: Any,
    stream: str,
) -> int:
    """Flush pending data from one SSH channel stream.

    Reads available bytes without blocking until the stream is drained
    or the max flush limit is reached.

    Args:
        channel: The SSH channel object.
        stream: Either "stdout" or "stderr".

    Returns:
        Number of bytes flushed from this stream.
    """
    total = 0

    try:
        if stream == "stdout":
            ready_fn = getattr(channel, "recv_ready", None)
            recv_fn = getattr(channel, "recv", None)
        else:
            ready_fn = getattr(channel, "recv_stderr_ready", None)
            recv_fn = getattr(channel, "recv_stderr", None)

        if ready_fn is None or recv_fn is None:
            return 0

        # Drain in a thread to avoid blocking the event loop
        def _drain() -> int:
            drained = 0
            while drained < _MAX_FLUSH_BYTES:
                try:
                    if not ready_fn():
                        break
                    chunk = recv_fn(_MAX_FLUSH_BYTES - drained)
                    if not chunk:
                        break
                    drained += len(chunk)
                except Exception as drain_exc:
                    # Log at module level since logger is module-global
                    logger.debug(
                        "Drain recv error on %s: %s", stream, drain_exc
                    )
                    break
            return drained

        total = await asyncio.to_thread(_drain)

    except Exception as exc:
        logger.debug(
            "Error flushing %s stream: %s", stream, exc
        )

    return total


async def _cleanup_socket_writer(
    entry: _ResourceEntry,
) -> CleanupResult:
    """Drain pending writes and close a socket writer.

    Attempts to drain() any buffered data in the StreamWriter before
    closing the connection. Both drain and close errors are captured
    rather than propagated.

    Args:
        entry: The resource entry containing the asyncio.StreamWriter.

    Returns:
        Immutable CleanupResult capturing outcome.
    """
    writer = entry.resource

    try:
        # Check if already closing
        if hasattr(writer, "is_closing") and writer.is_closing():
            return CleanupResult(
                resource_id=entry.resource_id,
                resource_type=ResourceType.SOCKET_WRITER,
                success=True,
                error=None,
                bytes_flushed=0,
            )

        # Drain pending writes (best-effort)
        try:
            if hasattr(writer, "drain"):
                await writer.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            logger.debug(
                "Drain failed for %s (expected during disconnect): %s",
                entry.resource_id,
                exc,
            )

        # Close the writer
        if hasattr(writer, "close"):
            writer.close()

        if hasattr(writer, "wait_closed"):
            await writer.wait_closed()

        logger.info(
            "Socket writer %s cleaned", entry.resource_id
        )
        return CleanupResult(
            resource_id=entry.resource_id,
            resource_type=ResourceType.SOCKET_WRITER,
            success=True,
            error=None,
            bytes_flushed=0,
        )

    except Exception as exc:
        logger.warning(
            "Error cleaning socket writer %s: %s: %s",
            entry.resource_id,
            type(exc).__name__,
            exc,
        )
        return CleanupResult(
            resource_id=entry.resource_id,
            resource_type=ResourceType.SOCKET_WRITER,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            bytes_flushed=0,
        )


async def _cleanup_io_buffer(
    entry: _ResourceEntry,
) -> CleanupResult:
    """Flush an I/O buffer resource.

    Calls flush() or close() on the buffer object if those methods
    are available.

    Args:
        entry: The resource entry containing the I/O buffer.

    Returns:
        Immutable CleanupResult capturing outcome.
    """
    buffer_obj = entry.resource

    try:
        if hasattr(buffer_obj, "flush"):
            buffer_obj.flush()

        if hasattr(buffer_obj, "close"):
            buffer_obj.close()

        logger.info("I/O buffer %s cleaned", entry.resource_id)
        return CleanupResult(
            resource_id=entry.resource_id,
            resource_type=ResourceType.IO_BUFFER,
            success=True,
            error=None,
            bytes_flushed=0,
        )

    except Exception as exc:
        logger.warning(
            "Error cleaning I/O buffer %s: %s: %s",
            entry.resource_id,
            type(exc).__name__,
            exc,
        )
        return CleanupResult(
            resource_id=entry.resource_id,
            resource_type=ResourceType.IO_BUFFER,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            bytes_flushed=0,
        )


# Type alias for cleanup functions
CleanupFn = Callable[[_ResourceEntry], Awaitable[CleanupResult]]

# Dispatch table for resource type -> cleanup function
_CLEANUP_DISPATCH: dict[ResourceType, CleanupFn] = {
    ResourceType.SSH_CHANNEL: _cleanup_ssh_channel,
    ResourceType.SOCKET_WRITER: _cleanup_socket_writer,
    ResourceType.IO_BUFFER: _cleanup_io_buffer,
}

# Cleanup order: SSH channels first, then sockets, then buffers.
# Remote resources are released before local ones.
_CLEANUP_ORDER: tuple[ResourceType, ...] = (
    ResourceType.SSH_CHANNEL,
    ResourceType.SOCKET_WRITER,
    ResourceType.IO_BUFFER,
)


# ---------------------------------------------------------------------------
# DisconnectCleanupHandler
# ---------------------------------------------------------------------------


class DisconnectCleanupHandler:
    """Coordinates resource cleanup when disconnect events are detected.

    Maintains a registry of resources (SSH channels, socket writers,
    I/O buffers) tagged by resource_id and optionally by client_id.
    On disconnect, cleans up all resources for the disconnected client
    in deterministic order.

    Thread safety: designed for single-threaded async use within one
    event loop. Does not use locks -- all mutations happen in the
    event loop thread.

    Args:
        event_bus: Optional EventBus for automatic disconnect detection.
            When provided, ``start()`` subscribes to CLIENT_DISCONNECTED_EVENT.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._resources: dict[str, _ResourceEntry] = {}
        self._subscription: Subscription | None = None
        self._started = False
        self._atexit_registered = False

    # -- Properties --

    @property
    def resource_count(self) -> int:
        """Number of currently registered resources."""
        return len(self._resources)

    # -- Registration --

    def register_ssh_channel(
        self,
        resource_id: str,
        channel: Any,
        client_id: str | None = None,
    ) -> None:
        """Register an SSH channel for cleanup on disconnect.

        If a resource with the same ``resource_id`` is already registered,
        it is silently replaced (the old resource is NOT cleaned up --
        the caller is responsible for prior cleanup if needed).

        Args:
            resource_id: Unique identifier for this resource.
            channel: The SSH channel object (must support close(), recv_ready(),
                recv(), recv_stderr_ready(), recv_stderr()).
            client_id: Optional client ID for per-client cleanup targeting.
        """
        entry = _ResourceEntry(
            resource_id=resource_id,
            resource_type=ResourceType.SSH_CHANNEL,
            resource=channel,
            client_id=client_id,
        )
        self._resources = {
            **{k: v for k, v in self._resources.items() if k != resource_id},
            resource_id: entry,
        }
        logger.debug(
            "Registered SSH channel: %s (client=%s)",
            resource_id,
            client_id,
        )

    def register_socket_writer(
        self,
        resource_id: str,
        writer: Any,
        client_id: str | None = None,
    ) -> None:
        """Register a socket writer for cleanup on disconnect.

        If a resource with the same ``resource_id`` is already registered,
        it is silently replaced.

        Args:
            resource_id: Unique identifier for this resource.
            writer: The asyncio.StreamWriter to close on disconnect.
            client_id: Optional client ID for per-client cleanup targeting.
        """
        entry = _ResourceEntry(
            resource_id=resource_id,
            resource_type=ResourceType.SOCKET_WRITER,
            resource=writer,
            client_id=client_id,
        )
        self._resources = {
            **{k: v for k, v in self._resources.items() if k != resource_id},
            resource_id: entry,
        }
        logger.debug(
            "Registered socket writer: %s (client=%s)",
            resource_id,
            client_id,
        )

    def register_io_buffer(
        self,
        resource_id: str,
        buffer: Any,
        client_id: str | None = None,
    ) -> None:
        """Register an I/O buffer for cleanup on disconnect.

        If a resource with the same ``resource_id`` is already registered,
        it is silently replaced.

        Args:
            resource_id: Unique identifier for this resource.
            buffer: The buffer object (should support flush() and/or close()).
            client_id: Optional client ID for per-client cleanup targeting.
        """
        entry = _ResourceEntry(
            resource_id=resource_id,
            resource_type=ResourceType.IO_BUFFER,
            resource=buffer,
            client_id=client_id,
        )
        self._resources = {
            **{k: v for k, v in self._resources.items() if k != resource_id},
            resource_id: entry,
        }
        logger.debug(
            "Registered I/O buffer: %s (client=%s)",
            resource_id,
            client_id,
        )

    def unregister(self, resource_id: str) -> None:
        """Remove a resource from the cleanup registry.

        If the resource_id is not found, this is a safe no-op.

        Args:
            resource_id: The unique identifier of the resource to remove.
        """
        if resource_id in self._resources:
            self._resources = {
                k: v
                for k, v in self._resources.items()
                if k != resource_id
            }
            logger.debug("Unregistered resource: %s", resource_id)

    # -- Cleanup operations --

    async def cleanup_all(self) -> CleanupSummary:
        """Clean up all registered resources in deterministic order.

        Processes resources in type order (SSH channels, then sockets,
        then I/O buffers). Each cleanup is error-isolated: a failure in
        one resource does not prevent cleanup of others.

        After cleanup, the registry is emptied.

        Returns:
            Immutable CleanupSummary with individual results for each
            resource.
        """
        event_id = f"cleanup-{uuid.uuid4().hex[:12]}"
        entries = self._collect_all_entries()

        if not entries:
            return CleanupSummary(event_id=event_id, results=())

        results = await self._execute_cleanup(entries)

        # Clear the registry (immutable pattern)
        self._resources = {}

        logger.info(
            "Cleanup complete (%s): %d resources, %d succeeded, %d failed",
            event_id,
            len(results),
            sum(1 for r in results if r.success),
            sum(1 for r in results if not r.success),
        )

        return CleanupSummary(event_id=event_id, results=tuple(results))

    async def cleanup_for_client(
        self,
        client_id: str,
    ) -> CleanupSummary:
        """Clean up resources tagged to a specific client.

        Only resources registered with the matching ``client_id`` are
        cleaned. Other resources remain in the registry.

        Args:
            client_id: The client ID whose resources should be cleaned.

        Returns:
            Immutable CleanupSummary with results for the client's resources.
        """
        event_id = f"client-cleanup-{client_id}-{uuid.uuid4().hex[:8]}"
        client_entries = [
            entry
            for entry in self._resources.values()
            if entry.client_id == client_id
        ]

        if not client_entries:
            return CleanupSummary(event_id=event_id, results=())

        # Sort by cleanup order
        ordered = self._sort_by_type(client_entries)
        results = await self._execute_cleanup(ordered)

        # Remove cleaned resources from registry (immutable pattern)
        cleaned_ids = frozenset(e.resource_id for e in client_entries)
        self._resources = {
            k: v
            for k, v in self._resources.items()
            if k not in cleaned_ids
        }

        logger.info(
            "Client cleanup complete (%s): %d resources for client %s",
            event_id,
            len(results),
            client_id,
        )

        return CleanupSummary(event_id=event_id, results=tuple(results))

    def sync_cleanup_all(self) -> None:
        """Synchronous cleanup for atexit and signal handlers.

        Creates a new event loop (or reuses existing) to run the async
        cleanup. This is the fallback for process shutdown scenarios
        where the main event loop may have been stopped.

        This method swallows all exceptions to ensure atexit handlers
        never propagate errors.
        """
        entries = self._collect_all_entries()
        if not entries:
            return

        for entry in entries:
            try:
                self._sync_cleanup_entry(entry)
            except Exception as exc:
                logger.debug(
                    "Sync cleanup error for %s: %s",
                    entry.resource_id,
                    exc,
                )

        self._resources = {}
        logger.info(
            "Sync cleanup complete: %d resources", len(entries)
        )

    # -- Event bus integration --

    async def start(self) -> None:
        """Subscribe to CLIENT_DISCONNECTED_EVENT on the event bus.

        Must be called after construction if an event_bus was provided.
        Idempotent: calling start() on an already-started handler is a
        safe no-op.

        Raises:
            RuntimeError: If no event_bus was provided at construction.
        """
        if self._started:
            return

        if self._event_bus is not None:
            self._subscription = self._event_bus.subscribe(
                CLIENT_DISCONNECTED_EVENT,
                self._on_client_disconnected,
            )
            logger.info(
                "DisconnectCleanupHandler subscribed to %s",
                CLIENT_DISCONNECTED_EVENT,
            )

        self._started = True

    async def stop(self) -> None:
        """Unsubscribe from events and clean up all remaining resources.

        Idempotent: calling stop() on an already-stopped handler is a
        safe no-op.
        """
        if self._subscription is not None and self._event_bus is not None:
            self._event_bus.unsubscribe(self._subscription)
            self._subscription = None

        await self.cleanup_all()
        self._started = False

        logger.info("DisconnectCleanupHandler stopped")

    # -- atexit integration --

    def register_atexit(self) -> None:
        """Register a synchronous cleanup handler with atexit.

        Ensures that any remaining registered resources are cleaned up
        when the Python interpreter exits, even if the async event loop
        has already been shut down.

        Idempotent: calling this multiple times registers only once.
        """
        if self._atexit_registered:
            return

        atexit.register(self.sync_cleanup_all)
        self._atexit_registered = True
        logger.info("atexit cleanup handler registered")

    # -- Async context manager --

    async def __aenter__(self) -> DisconnectCleanupHandler:
        """Enter the cleanup context. Calls start() if event_bus exists."""
        if self._event_bus is not None:
            await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the cleanup context. Cleans all remaining resources."""
        await self.cleanup_all()
        if self._subscription is not None and self._event_bus is not None:
            self._event_bus.unsubscribe(self._subscription)
            self._subscription = None

    # -- Internal helpers --

    def _collect_all_entries(self) -> list[_ResourceEntry]:
        """Collect all entries sorted by cleanup order."""
        return self._sort_by_type(list(self._resources.values()))

    def _sort_by_type(
        self, entries: list[_ResourceEntry]
    ) -> list[_ResourceEntry]:
        """Sort entries by the deterministic cleanup order."""
        order_map = {rt: idx for idx, rt in enumerate(_CLEANUP_ORDER)}
        return sorted(
            entries,
            key=lambda e: order_map.get(e.resource_type, len(_CLEANUP_ORDER)),
        )

    async def _execute_cleanup(
        self,
        entries: list[_ResourceEntry],
    ) -> list[CleanupResult]:
        """Execute cleanup for a list of resource entries.

        Each entry is cleaned using the appropriate dispatch function.
        Errors are captured per-resource (error isolation).

        Args:
            entries: Resource entries to clean, already sorted by type.

        Returns:
            List of CleanupResult instances, one per entry.
        """
        results: list[CleanupResult] = []

        for entry in entries:
            cleanup_fn = _CLEANUP_DISPATCH.get(entry.resource_type)
            if cleanup_fn is None:
                logger.warning(
                    "No cleanup function for resource type %s",
                    entry.resource_type,
                )
                results.append(
                    CleanupResult(
                        resource_id=entry.resource_id,
                        resource_type=entry.resource_type,
                        success=False,
                        error=f"Unknown resource type: {entry.resource_type}",
                        bytes_flushed=0,
                    )
                )
                continue

            try:
                result = await cleanup_fn(entry)
                results.append(result)
            except Exception as exc:
                logger.warning(
                    "Cleanup dispatch error for %s: %s",
                    entry.resource_id,
                    exc,
                )
                results.append(
                    CleanupResult(
                        resource_id=entry.resource_id,
                        resource_type=entry.resource_type,
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                        bytes_flushed=0,
                    )
                )

        return results

    async def _on_client_disconnected(self, event: Event) -> None:
        """Event handler for CLIENT_DISCONNECTED_EVENT.

        Extracts the client_id from the event payload and triggers
        cleanup for all resources tagged to that client.

        Args:
            event: The disconnect event from the EventBus.
        """
        client_id = event.payload.get("client_id")
        if not client_id:
            logger.warning(
                "Disconnect event missing client_id in payload"
            )
            return

        logger.info(
            "Client disconnect detected: %s -- starting cleanup",
            client_id,
        )
        summary = await self.cleanup_for_client(client_id)
        logger.info(
            "Client %s cleanup: %d resources, %d succeeded, %d failed",
            client_id,
            summary.total_resources,
            summary.successful,
            summary.failed,
        )

    @staticmethod
    def _sync_cleanup_entry(entry: _ResourceEntry) -> None:
        """Synchronously clean a single resource entry.

        Used by sync_cleanup_all() for atexit scenarios where
        the async event loop may not be available.

        Does best-effort cleanup: flushes buffers, then closes.
        All errors are logged at DEBUG level for observability.
        """
        resource = entry.resource

        if entry.resource_type == ResourceType.SSH_CHANNEL:
            # Flush stdout
            try:
                ready_fn = getattr(resource, "recv_ready", None)
                recv_fn = getattr(resource, "recv", None)
                if ready_fn and recv_fn:
                    while ready_fn():
                        chunk = recv_fn(_MAX_FLUSH_BYTES)
                        if not chunk:
                            break
            except Exception as exc:
                logger.debug(
                    "Sync stdout flush error for %s: %s",
                    entry.resource_id,
                    exc,
                )

            # Flush stderr
            try:
                ready_fn = getattr(resource, "recv_stderr_ready", None)
                recv_fn = getattr(resource, "recv_stderr", None)
                if ready_fn and recv_fn:
                    while ready_fn():
                        chunk = recv_fn(_MAX_FLUSH_BYTES)
                        if not chunk:
                            break
            except Exception as exc:
                logger.debug(
                    "Sync stderr flush error for %s: %s",
                    entry.resource_id,
                    exc,
                )

            # Close
            try:
                if hasattr(resource, "close"):
                    resource.close()
            except Exception as exc:
                logger.debug(
                    "Sync channel close error for %s: %s",
                    entry.resource_id,
                    exc,
                )

        elif entry.resource_type == ResourceType.SOCKET_WRITER:
            try:
                if hasattr(resource, "close"):
                    resource.close()
            except Exception as exc:
                logger.debug(
                    "Sync socket close error for %s: %s",
                    entry.resource_id,
                    exc,
                )

        elif entry.resource_type == ResourceType.IO_BUFFER:
            try:
                if hasattr(resource, "flush"):
                    resource.flush()
                if hasattr(resource, "close"):
                    resource.close()
            except Exception as exc:
                logger.debug(
                    "Sync buffer cleanup error for %s: %s",
                    entry.resource_id,
                    exc,
                )
