"""Real-time client disconnect detection for persistent IPC connections.

Monitors a client's asyncio reader/writer pair for disconnect signals and
triggers a cleanup callback when the connection drops. Two complementary
detection strategies run concurrently:

1. **Reader EOF check**: Periodically polls ``reader.at_eof()`` to detect
   when the client has closed its end of the connection (clean disconnect,
   client crash, or network partition where the OS has already torn down
   the socket).

2. **Write probe**: Periodically performs a zero-byte write + drain on the
   writer to detect broken-pipe, connection-reset, and other transport
   errors that only manifest when the daemon attempts to send data.

These two strategies are complementary: reader EOF catches clean disconnects
quickly (even without pending writes), while write probes catch scenarios
where the OS has not yet closed the read end (e.g., kernel TCP keepalive
has not expired) but the write path is already broken.

The detector guarantees:

- **Single-fire callback**: The ``on_disconnect`` callback is invoked at
  most once, even if both strategies detect the disconnect simultaneously.
- **Error isolation**: If the callback raises, the error is logged but
  does not propagate -- the detector terminates cleanly.
- **Clean cancellation**: Calling ``stop()`` sets an internal asyncio.Event
  that terminates both strategies. The callback is NOT invoked on a clean
  stop (only on genuine disconnects and external task cancellation).
- **Reason capture**: The ``disconnect_reason`` property exposes the
  classified reason after detection.

The detector integrates with the daemon's EventBus (optional) to emit
``CLIENT_DISCONNECTED_EVENT`` so that other subsystems (cleanup handlers,
audit logging, subscription managers) can react.

Architecture::

    asyncio.StreamReader        asyncio.StreamWriter
        |                            |
        v                            v
    [reader EOF check]         [write probe]
        |                            |
        +--- first to detect --------+
                    |
                    v
            DisconnectReason (immutable)
                    |
                    +---> on_disconnect callback
                    +---> EventBus.emit (optional)
                    +---> detector.disconnect_reason property

Usage::

    from jules_daemon.ipc.client_disconnect_detector import (
        ClientDisconnectDetector,
        DetectorConfig,
    )

    async def on_disconnect(reason: DisconnectReason) -> None:
        await cleanup_handler.cleanup_for_client(reason.client_id)

    detector = ClientDisconnectDetector(
        reader=reader,
        writer=writer,
        client_id="client-abc",
        on_disconnect=on_disconnect,
        config=DetectorConfig(probe_interval_seconds=5.0),
        event_bus=bus,
    )

    task = asyncio.create_task(detector.run())
    # ... later, when shutting down cleanly:
    await detector.stop()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from jules_daemon.ipc.connection_manager import CLIENT_DISCONNECTED_EVENT
from jules_daemon.ipc.event_bus import Event, EventBus

__all__ = [
    "ClientDisconnectDetector",
    "DetectorConfig",
    "DisconnectReason",
    "DisconnectSource",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PROBE_INTERVAL_SECONDS: float = 10.0
"""Default interval between write probes (in seconds)."""

_DEFAULT_PROBE_TIMEOUT_SECONDS: float = 3.0
"""Default timeout for a single write probe drain operation."""

_DEFAULT_READER_CHECK_INTERVAL_SECONDS: float = 1.0
"""Default interval between reader EOF checks (in seconds)."""


# ---------------------------------------------------------------------------
# DisconnectSource enum
# ---------------------------------------------------------------------------


class DisconnectSource(Enum):
    """Classification of how the disconnect was detected.

    Values:
        READER_EOF:          The reader reported at_eof().
        WRITE_PROBE_FAILURE: A write probe raised a transport error.
        CANCELLATION:        The detector task was cancelled externally.
        WRITER_CLOSING:      The writer was already in a closing state.
    """

    READER_EOF = "reader_eof"
    WRITE_PROBE_FAILURE = "write_probe_failure"
    CANCELLATION = "cancellation"
    WRITER_CLOSING = "writer_closing"


# ---------------------------------------------------------------------------
# DisconnectReason model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisconnectReason:
    """Immutable model describing why a client disconnect was detected.

    Attributes:
        source:         How the disconnect was detected.
        client_id:      The affected client identifier.
        message:        Human-readable description of the disconnect.
        exception_type: The exception class name if the disconnect was
                        caused by an exception (e.g., "BrokenPipeError").
                        None for non-exception disconnects (EOF, closing).
        timestamp:      ISO 8601 UTC timestamp of detection.
    """

    source: DisconnectSource
    client_id: str
    message: str
    exception_type: str | None = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.client_id, str) or not self.client_id.strip():
            raise ValueError("client_id must not be empty")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must not be empty")
        if not self.timestamp:
            object.__setattr__(
                self, "timestamp", datetime.now(timezone.utc).isoformat()
            )


# ---------------------------------------------------------------------------
# DetectorConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorConfig:
    """Immutable configuration for the client disconnect detector.

    Attributes:
        probe_interval_seconds:        Interval between write probes.
        probe_timeout_seconds:         Timeout for each write probe drain.
        reader_check_interval_seconds: Interval between reader EOF checks.
    """

    probe_interval_seconds: float = _DEFAULT_PROBE_INTERVAL_SECONDS
    probe_timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS
    reader_check_interval_seconds: float = _DEFAULT_READER_CHECK_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        if self.probe_interval_seconds <= 0:
            raise ValueError(
                f"probe_interval_seconds must be positive, "
                f"got {self.probe_interval_seconds}"
            )
        if self.probe_timeout_seconds <= 0:
            raise ValueError(
                f"probe_timeout_seconds must be positive, "
                f"got {self.probe_timeout_seconds}"
            )
        if self.reader_check_interval_seconds <= 0:
            raise ValueError(
                f"reader_check_interval_seconds must be positive, "
                f"got {self.reader_check_interval_seconds}"
            )


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

DisconnectCallback = Callable[["DisconnectReason"], Awaitable[None]]
"""Async callback invoked when a disconnect is detected."""


# ---------------------------------------------------------------------------
# ClientDisconnectDetector
# ---------------------------------------------------------------------------


class ClientDisconnectDetector:
    """Real-time client disconnect detector for persistent IPC connections.

    Runs two concurrent monitoring strategies (reader EOF check and write
    probe) and triggers a callback when the first one detects a disconnect.
    The callback is guaranteed to fire at most once.

    Thread safety: designed for single-threaded async use within one
    event loop. Does not use locks -- the single-fire guarantee is
    enforced by an asyncio.Event.

    Args:
        reader:        The asyncio.StreamReader for the client connection.
        writer:        The asyncio.StreamWriter for the client connection.
        client_id:     Unique identifier for the monitored client.
        on_disconnect: Async callback invoked when disconnect is detected.
        config:        Optional detector configuration.
        event_bus:     Optional EventBus for emitting disconnect events.
    """

    def __init__(
        self,
        *,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        client_id: str,
        on_disconnect: DisconnectCallback,
        config: DetectorConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._client_id = client_id
        self._on_disconnect = on_disconnect
        self._config = config or DetectorConfig()
        self._event_bus = event_bus

        # Internal state
        self._stop_event = asyncio.Event()
        self._fired = asyncio.Event()
        self._running = False
        self._disconnect_reason: DisconnectReason | None = None

    # -- Properties --

    @property
    def client_id(self) -> str:
        """The client ID being monitored."""
        return self._client_id

    @property
    def is_running(self) -> bool:
        """True if the detector is currently monitoring."""
        return self._running

    @property
    def disconnect_reason(self) -> DisconnectReason | None:
        """The disconnect reason, or None if not yet disconnected."""
        return self._disconnect_reason

    # -- Public API --

    async def run(self) -> None:
        """Start monitoring the client connection for disconnects.

        Runs until one of:
        - A disconnect is detected (reader EOF or write probe failure)
        - ``stop()`` is called
        - The task is cancelled externally

        On external cancellation, the callback IS invoked with
        ``DisconnectSource.CANCELLATION`` (since the client connection
        may be in an indeterminate state and resources should be cleaned).

        Raises:
            asyncio.CancelledError: If the task is cancelled externally.
                The callback is invoked before re-raising.
        """
        self._running = True

        try:
            await self._monitor_loop()
        except asyncio.CancelledError:
            # External cancellation -- trigger cleanup
            if not self._fired.is_set():
                reason = DisconnectReason(
                    source=DisconnectSource.CANCELLATION,
                    client_id=self._client_id,
                    message="Detector task cancelled externally",
                )
                await self._fire_callback(reason)
            raise
        finally:
            self._running = False

    async def stop(self) -> None:
        """Request the detector to stop monitoring.

        Sets the internal stop event, which causes the monitoring loop
        to exit cleanly without invoking the callback.

        Idempotent: calling stop() multiple times is safe.
        """
        self._stop_event.set()

    # -- Internal: monitoring loop --

    async def _monitor_loop(self) -> None:
        """Run both detection strategies concurrently.

        Uses asyncio.wait with FIRST_COMPLETED to react to whichever
        strategy detects a disconnect first. The stop event is also
        included as a candidate for clean shutdown.
        """
        reader_task = asyncio.create_task(
            self._reader_eof_loop(),
            name=f"disconnect-reader-{self._client_id}",
        )
        probe_task = asyncio.create_task(
            self._write_probe_loop(),
            name=f"disconnect-probe-{self._client_id}",
        )
        stop_task = asyncio.create_task(
            self._stop_event.wait(),
            name=f"disconnect-stop-{self._client_id}",
        )

        try:
            done, pending = await asyncio.wait(
                {reader_task, probe_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Suppress CancelledError from cancelled subtasks
            for task in pending:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            # Parent was cancelled -- clean up subtasks
            for task in (reader_task, probe_task, stop_task):
                task.cancel()
            for task in (reader_task, probe_task, stop_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            raise

    # -- Internal: reader EOF detection --

    async def _reader_eof_loop(self) -> None:
        """Periodically check if the reader has reached EOF.

        When EOF is detected, fires the callback with READER_EOF source.
        """
        interval = self._config.reader_check_interval_seconds

        while not self._stop_event.is_set() and not self._fired.is_set():
            if self._reader.at_eof():
                reason = DisconnectReason(
                    source=DisconnectSource.READER_EOF,
                    client_id=self._client_id,
                    message="Client connection reader reached EOF",
                )
                await self._fire_callback(reason)
                return

            # Sleep for the check interval, but wake up early if stopped
            try:
                await asyncio.wait_for(
                    self._wait_for_stop_or_fired(),
                    timeout=interval,
                )
                # If we get here, stop or fired was set
                return
            except asyncio.TimeoutError:
                # Interval elapsed -- loop around and check again
                continue

    # -- Internal: write probe detection --

    async def _write_probe_loop(self) -> None:
        """Periodically send a write probe to detect broken connections.

        A write probe consists of writing zero bytes and draining.
        If the drain raises a transport error, the connection is broken.
        """
        interval = self._config.probe_interval_seconds
        timeout = self._config.probe_timeout_seconds

        while not self._stop_event.is_set() and not self._fired.is_set():
            # Sleep first, then probe (avoids probing immediately on start)
            try:
                await asyncio.wait_for(
                    self._wait_for_stop_or_fired(),
                    timeout=interval,
                )
                # If we get here, stop or fired was set
                return
            except asyncio.TimeoutError:
                # Interval elapsed -- proceed with probe
                pass

            # Check if writer is already closing
            if self._writer.is_closing():
                reason = DisconnectReason(
                    source=DisconnectSource.WRITER_CLOSING,
                    client_id=self._client_id,
                    message="Writer is already in closing state",
                )
                await self._fire_callback(reason)
                return

            # Attempt write probe
            disconnect_reason = await self._execute_write_probe(timeout)
            if disconnect_reason is not None:
                await self._fire_callback(disconnect_reason)
                return

    async def _execute_write_probe(
        self,
        timeout: float,
    ) -> DisconnectReason | None:
        """Execute a single write probe and return a reason on failure.

        Writes zero bytes and drains with a timeout. Returns None if the
        probe succeeds (connection is healthy), or a DisconnectReason if
        a transport error is detected.

        Args:
            timeout: Maximum seconds to wait for the drain to complete.

        Returns:
            DisconnectReason on failure, None on success.
        """
        try:
            self._writer.write(b"")
            await asyncio.wait_for(
                self._writer.drain(),
                timeout=timeout,
            )
            return None

        except asyncio.TimeoutError:
            # Must be caught before OSError: on Python 3.11+,
            # asyncio.TimeoutError is a subclass of OSError.
            return DisconnectReason(
                source=DisconnectSource.WRITE_PROBE_FAILURE,
                client_id=self._client_id,
                message="Write probe drain timed out",
                exception_type="TimeoutError",
            )

        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            exc_name = type(exc).__name__
            return DisconnectReason(
                source=DisconnectSource.WRITE_PROBE_FAILURE,
                client_id=self._client_id,
                message=f"{exc_name}: {exc}",
                exception_type=exc_name,
            )

    # -- Internal: callback firing --

    async def _fire_callback(self, reason: DisconnectReason) -> None:
        """Invoke the disconnect callback exactly once.

        Uses the ``_fired`` event as a single-fire gate. If the event
        is already set, this method is a no-op.

        The callback is wrapped in a try/except for error isolation --
        a failing callback is logged but does not propagate.

        Args:
            reason: The disconnect reason to pass to the callback.
        """
        # Single-fire gate: safe because all callers run in the same
        # event loop thread (no context switch between check and set).
        if self._fired.is_set():
            return

        self._fired.set()
        self._disconnect_reason = reason

        logger.info(
            "Client %s disconnect detected (source=%s): %s",
            reason.client_id,
            reason.source.value,
            reason.message,
        )

        # Invoke the user callback with error isolation
        try:
            await self._on_disconnect(reason)
        except Exception as exc:
            logger.warning(
                "Disconnect callback error for client %s: %s: %s",
                reason.client_id,
                type(exc).__name__,
                exc,
                exc_info=True,
            )

        # Emit event on the EventBus (if configured)
        if self._event_bus is not None:
            try:
                await self._event_bus.emit(
                    Event(
                        event_type=CLIENT_DISCONNECTED_EVENT,
                        payload={
                            "client_id": reason.client_id,
                            "source": reason.source.value,
                            "message": reason.message,
                            "exception_type": reason.exception_type,
                            "timestamp": reason.timestamp,
                        },
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Failed to emit disconnect event for client %s: %s",
                    reason.client_id,
                    exc,
                    exc_info=True,
                )

    # -- Internal: helper --

    async def _wait_for_stop_or_fired(self) -> None:
        """Wait until either the stop event or fired event is set.

        This is used as a cancellable sleep: if neither event is set,
        the caller should use ``asyncio.wait_for`` with a timeout to
        bound the wait.
        """
        stop_waiter = asyncio.create_task(self._stop_event.wait())
        fired_waiter = asyncio.create_task(self._fired.wait())

        try:
            done, pending = await asyncio.wait(
                {stop_waiter, fired_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        except asyncio.CancelledError:
            stop_waiter.cancel()
            fired_waiter.cancel()
            try:
                await stop_waiter
            except asyncio.CancelledError:
                pass
            try:
                await fired_waiter
            except asyncio.CancelledError:
                pass
            raise
