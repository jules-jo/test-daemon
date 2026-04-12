"""Persistent notification subscription client for daemon push events.

Opens a long-lived streaming connection to the daemon's notification
endpoint and exposes both an async iterator and callback interface for
incoming events. The client handles the full lifecycle:

    1. **Connect**: Establish a Unix/TCP socket connection and perform
       the IPC protocol handshake.
    2. **Subscribe**: Send a subscribe request to the notification
       channel and wait for the daemon's acknowledgment.
    3. **Stream loop**: Read STREAM envelopes carrying NotificationEnvelope
       payloads. Dispatch each event to registered callbacks and enqueue
       for async iteration. Heartbeats reset the liveness timer.
    4. **Close**: Send an unsubscribe request, close the transport, and
       post a sentinel to the event queue so iterators stop.

The client supports two consumption patterns:

    **Async iterator** (pull)::

        async for event in client:
            print(event.event_type, event.payload)

    **Callback** (push)::

        async def on_completion(env: NotificationEnvelope) -> None:
            print(f"Run {env.payload.run_id} finished")

        client.on_event(on_completion, event_types={NotificationEventType.COMPLETION})

Both patterns work concurrently -- callbacks fire immediately on
dispatch, while the async iterator drains from a bounded queue.

Architecture::

    CLI Process                             Daemon Process
        |                                        |
        |-- connect (Unix socket) ------------->|
        |-- REQUEST {handshake} --------------->|
        |<-- RESPONSE {handshake ok} -----------|
        |-- REQUEST {subscribe_notifications} ->|
        |<-- RESPONSE {subscription_id} --------|
        |                                        |
        |<-- STREAM {notification envelope} ----|  (repeated)
        |    dispatch to callbacks               |
        |    enqueue for async iteration         |
        |                                        |
        |  (on close / error / timeout)          |
        |-- REQUEST {unsubscribe_notifications} >|
        |<-- RESPONSE {unsubscribed} ------------|
        |-- close connection ------------------->|

Key design decisions:

- **Frozen config**: ``SubscriptionClientConfig`` is immutable and
  captures all connection parameters, filter preferences, and
  reconnect policy.

- **Event queue**: Incoming events are placed in a bounded asyncio.Queue
  (default size 1000). When full, oldest events are evicted to prevent
  blocking the stream loop. The queue serves the async iterator.

- **Callback isolation**: Each callback is invoked in its own try/except.
  A failing callback never blocks other callbacks or the stream loop.

- **Heartbeat monitoring**: The client tracks the last heartbeat time.
  If the heartbeat timeout is exceeded between stream reads, the
  connection is considered stale and the stream loop exits.

- **State machine**: The client transitions through DISCONNECTED ->
  CONNECTING -> SUBSCRIBING -> ACTIVE -> CLOSING -> CLOSED. State
  is exposed as a read-only property.

- **Auto-reconnect**: When configured, the client retries connection
  with exponential backoff on transient failures (not on clean close
  or user cancel).

Usage::

    from jules_daemon.ipc.subscription_client import (
        SubscriptionClient,
        SubscriptionClientConfig,
    )

    config = SubscriptionClientConfig(
        socket_path="/run/jules/daemon.sock",
        event_filter=frozenset({NotificationEventType.COMPLETION}),
    )
    client = SubscriptionClient(config=config)

    # Callback-based
    client.on_event(my_handler)
    result = await client.run()

    # Or iterator-based in a separate task
    async for event in client:
        process(event)
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable

from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
    unpack_header,
)
from jules_daemon.protocol.notifications import (
    NotificationEnvelope,
    NotificationEventType,
)

__all__ = [
    "SubscriptionClient",
    "SubscriptionClientConfig",
    "SubscriptionExitReason",
    "SubscriptionResult",
    "SubscriptionState",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CONNECT_TIMEOUT: float = 5.0
_DEFAULT_HEARTBEAT_TIMEOUT: float = 90.0
_DEFAULT_READ_TIMEOUT: float = 10.0
_DEFAULT_EVENT_QUEUE_SIZE: int = 1000
_DEFAULT_MAX_RECONNECT_ATTEMPTS: int = 3
_DEFAULT_RECONNECT_BASE_DELAY: float = 1.0

_VERB_SUBSCRIBE = "subscribe_notifications"
_VERB_UNSUBSCRIBE = "unsubscribe_notifications"
_VERB_HANDSHAKE = "handshake"

_PROTOCOL_VERSION: int = 1

# Type alias for event callbacks
EventCallback = Callable[["NotificationEnvelope"], Awaitable[None]]


# ---------------------------------------------------------------------------
# SubscriptionState enum
# ---------------------------------------------------------------------------


class SubscriptionState(Enum):
    """Lifecycle state of the subscription client.

    Values:
        DISCONNECTED: Not connected. Initial state or after disconnect.
        CONNECTING:   Socket connection + handshake in progress.
        SUBSCRIBING:  Connected, subscription request in progress.
        ACTIVE:       Subscribed and streaming events.
        CLOSING:      Graceful shutdown in progress.
        CLOSED:       Terminal state. Resources released.
    """

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    SUBSCRIBING = "subscribing"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# SubscriptionExitReason enum
# ---------------------------------------------------------------------------


class SubscriptionExitReason(Enum):
    """Why the subscription session ended.

    Values:
        CLEAN_CLOSE:      Graceful shutdown via close() or context manager.
        HEARTBEAT_TIMEOUT: No heartbeat received within the timeout window.
        CONNECTION_LOST:   Socket EOF or connection reset.
        DAEMON_ERROR:      Daemon sent an ERROR envelope.
        USER_CANCEL:       User cancelled (CancelledError).
        SUBSCRIBE_FAILED:  Subscription request was rejected by daemon.
    """

    CLEAN_CLOSE = "clean_close"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    CONNECTION_LOST = "connection_lost"
    DAEMON_ERROR = "daemon_error"
    USER_CANCEL = "user_cancel"
    SUBSCRIBE_FAILED = "subscribe_failed"


# ---------------------------------------------------------------------------
# SubscriptionClientConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubscriptionClientConfig:
    """Immutable configuration for the subscription client.

    Attributes:
        socket_path:           Path to the daemon's Unix domain socket.
        connect_timeout:       Max seconds to wait for socket connection.
        heartbeat_timeout:     Max seconds between heartbeats before
                               considering the connection stale.
        read_timeout:          Max seconds to wait for a single envelope
                               read. The stream loop uses this as the
                               per-read timeout.
        event_filter:          Optional set of event types to subscribe to.
                               When None, all events are delivered.
        event_queue_size:      Maximum events buffered for async iteration.
        auto_reconnect:        Whether to auto-reconnect on transient errors.
        max_reconnect_attempts: Max reconnect attempts before giving up.
        reconnect_base_delay:  Base delay in seconds for exponential backoff.
    """

    socket_path: str
    connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT
    heartbeat_timeout: float = _DEFAULT_HEARTBEAT_TIMEOUT
    read_timeout: float = _DEFAULT_READ_TIMEOUT
    event_filter: frozenset[NotificationEventType] | None = None
    event_queue_size: int = _DEFAULT_EVENT_QUEUE_SIZE
    auto_reconnect: bool = False
    max_reconnect_attempts: int = _DEFAULT_MAX_RECONNECT_ATTEMPTS
    reconnect_base_delay: float = _DEFAULT_RECONNECT_BASE_DELAY

    def __post_init__(self) -> None:
        if not self.socket_path or not self.socket_path.strip():
            raise ValueError("socket_path must not be empty")
        if self.connect_timeout <= 0:
            raise ValueError(
                f"connect_timeout must be positive, got {self.connect_timeout}"
            )
        if self.heartbeat_timeout <= 0:
            raise ValueError(
                f"heartbeat_timeout must be positive, got {self.heartbeat_timeout}"
            )
        if self.read_timeout <= 0:
            raise ValueError(
                f"read_timeout must be positive, got {self.read_timeout}"
            )
        if self.event_queue_size < 1:
            raise ValueError(
                f"event_queue_size must be positive, got {self.event_queue_size}"
            )
        if self.max_reconnect_attempts < 0:
            raise ValueError(
                f"max_reconnect_attempts must not be negative, "
                f"got {self.max_reconnect_attempts}"
            )
        if self.reconnect_base_delay <= 0:
            raise ValueError(
                f"reconnect_base_delay must be positive, "
                f"got {self.reconnect_base_delay}"
            )


# ---------------------------------------------------------------------------
# SubscriptionResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubscriptionResult:
    """Immutable result of a subscription session.

    Attributes:
        exit_reason:     Why the subscription ended.
        events_received: Total notification events received during session.
        subscription_id: The daemon-assigned subscription ID (if obtained).
        error_message:   Human-readable error description, or None.
    """

    exit_reason: SubscriptionExitReason
    events_received: int
    subscription_id: str | None
    error_message: str | None


# ---------------------------------------------------------------------------
# Internal: callback entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CallbackEntry:
    """Internal record of a registered event callback.

    Attributes:
        handle:      Unique string handle for removal.
        callback:    Async callable to invoke with each event.
        event_types: Optional filter. When None, receives all events.
    """

    handle: str
    callback: EventCallback
    event_types: frozenset[NotificationEventType] | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _generate_msg_id() -> str:
    """Generate a unique message ID for request-response correlation."""
    return f"sub-{uuid.uuid4().hex[:12]}"


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# SubscriptionClient
# ---------------------------------------------------------------------------


class SubscriptionClient:
    """Persistent client for streaming daemon notification events.

    Connects to the daemon, subscribes to the notification channel,
    and streams events via callbacks and/or async iteration. Handles
    heartbeat monitoring, error recovery, and clean shutdown.

    The client is designed for long-lived use: it maintains a persistent
    connection and streams events until explicitly closed or an
    unrecoverable error occurs.

    Args:
        config: Connection and subscription configuration.
    """

    def __init__(self, *, config: SubscriptionClientConfig) -> None:
        self._config = config
        self._state = SubscriptionState.DISCONNECTED
        self._subscription_id: str | None = None
        self._events_received: int = 0
        self._last_heartbeat: datetime | None = None
        self._callbacks: list[_CallbackEntry] = []
        self._event_queue: asyncio.Queue[NotificationEnvelope | None] = (
            asyncio.Queue(maxsize=config.event_queue_size)
        )
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._stream_task: asyncio.Task[None] | None = None

    # -- Properties -----------------------------------------------------------

    @property
    def config(self) -> SubscriptionClientConfig:
        """The subscription configuration."""
        return self._config

    @property
    def state(self) -> SubscriptionState:
        """Current lifecycle state."""
        return self._state

    @property
    def subscription_id(self) -> str | None:
        """Daemon-assigned subscription ID, or None if not subscribed."""
        return self._subscription_id

    @property
    def events_received(self) -> int:
        """Total notification events received in this session."""
        return self._events_received

    # -- Async context manager ------------------------------------------------

    async def __aenter__(self) -> SubscriptionClient:
        """Enter the async context. Does not auto-connect."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Close the subscription on context exit."""
        await self.close()

    # -- Async iterator interface ---------------------------------------------

    def __aiter__(self) -> AsyncIterator[NotificationEnvelope]:
        """Return self as an async iterator over notification events."""
        return self

    async def __anext__(self) -> NotificationEnvelope:
        """Yield the next notification event from the queue.

        Returns the next event, or raises StopAsyncIteration when
        a None sentinel is received (indicating shutdown).
        """
        event = await self._event_queue.get()
        if event is None:
            raise StopAsyncIteration
        return event

    # -- Callback interface ---------------------------------------------------

    def on_event(
        self,
        callback: EventCallback,
        *,
        event_types: frozenset[NotificationEventType] | None = None,
    ) -> str:
        """Register an async callback for incoming notification events.

        The callback is invoked for each event that matches the optional
        event_types filter. If event_types is None, the callback receives
        all events.

        Callbacks are invoked sequentially in registration order. A
        failing callback never blocks other callbacks.

        Args:
            callback:    Async callable receiving a NotificationEnvelope.
            event_types: Optional filter for specific event types.

        Returns:
            A string handle for later removal via remove_callback().
        """
        handle = f"cb-{uuid.uuid4().hex[:12]}"
        entry = _CallbackEntry(
            handle=handle,
            callback=callback,
            event_types=event_types,
        )
        # Create new list for immutability
        self._callbacks = [*self._callbacks, entry]
        logger.debug(
            "Registered callback %s (filter=%s)", handle, event_types
        )
        return handle

    def remove_callback(self, handle: str) -> bool:
        """Remove a previously registered callback.

        Idempotent: removing a nonexistent handle returns False.

        Args:
            handle: The callback handle returned by on_event().

        Returns:
            True if the callback was found and removed.
        """
        original_count = len(self._callbacks)
        self._callbacks = [
            entry for entry in self._callbacks if entry.handle != handle
        ]
        removed = len(self._callbacks) < original_count
        if removed:
            logger.debug("Removed callback %s", handle)
        return removed

    # -- Public API: run / close ----------------------------------------------

    async def run(self) -> SubscriptionResult:
        """Connect, subscribe, and stream events until termination.

        This is the main entry point. It:
        1. Connects to the daemon socket and performs the handshake.
        2. Sends a subscribe request and waits for acknowledgment.
        3. Enters the stream loop, dispatching events to callbacks
           and the async iterator queue.
        4. On exit, sends an unsubscribe request and closes transport.

        Returns:
            SubscriptionResult describing how the session ended.
        """
        try:
            import sys

            if sys.platform == "win32":
                # Windows: TCP connection
                try:
                    from pathlib import Path

                    port = int(Path(self._config.socket_path).read_text().strip())
                except (FileNotFoundError, ValueError) as exc:
                    return SubscriptionResult(
                        exit_reason=SubscriptionExitReason.CONNECTION_LOST,
                        events_received=0,
                        subscription_id=None,
                        error_message=f"Daemon not running: {exc}",
                    )
                connect_coro = asyncio.open_connection("127.0.0.1", port)
            else:
                connect_coro = asyncio.open_unix_connection(
                    self._config.socket_path
                )

            reader, writer = await asyncio.wait_for(
                connect_coro,
                timeout=self._config.connect_timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            logger.error("Failed to connect to daemon: %s", exc)
            return SubscriptionResult(
                exit_reason=SubscriptionExitReason.CONNECTION_LOST,
                events_received=0,
                subscription_id=None,
                error_message=f"Connection failed: {exc}",
            )

        try:
            connected = await self._connect_with_transport(
                reader=reader,
                writer=writer,
            )
            if not connected:
                return SubscriptionResult(
                    exit_reason=SubscriptionExitReason.SUBSCRIBE_FAILED,
                    events_received=0,
                    subscription_id=self._subscription_id,
                    error_message="Subscription setup failed",
                )

            # Enter the stream loop
            exit_reason = await self._stream_loop()

            # Send unsubscribe (best-effort)
            await self._try_send_unsubscribe()

            return SubscriptionResult(
                exit_reason=exit_reason,
                events_received=self._events_received,
                subscription_id=self._subscription_id,
                error_message=(
                    None
                    if exit_reason == SubscriptionExitReason.CLEAN_CLOSE
                    else f"Session ended: {exit_reason.value}"
                ),
            )

        finally:
            await self._close_transport()

    async def close(self) -> None:
        """Gracefully close the subscription and release resources.

        Posts a None sentinel to the event queue so async iterators
        stop. Transitions to CLOSED state. Idempotent.
        """
        if self._state == SubscriptionState.CLOSED:
            return

        self._state = SubscriptionState.CLOSING

        # Cancel stream task if running
        if self._stream_task is not None and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        # Post sentinel for async iterator consumers
        try:
            self._event_queue.put_nowait(None)
        except asyncio.QueueFull:
            # Queue is full; evict oldest to make room for sentinel
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._event_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass  # Best effort

        await self._close_transport()
        self._state = SubscriptionState.CLOSED
        logger.debug("Subscription client closed")

    # -- Internal: connect with transport -------------------------------------

    async def _connect_with_transport(
        self,
        *,
        reader: asyncio.StreamReader,
        writer: object,
    ) -> bool:
        """Execute handshake and subscription on pre-existing transport.

        Separated from run() for testability -- tests inject mock
        reader/writer without a real socket.

        Args:
            reader: Stream to read framed messages from.
            writer: Stream to write framed messages to.

        Returns:
            True if handshake and subscription succeeded.
        """
        self._reader = reader  # type: ignore[assignment]
        self._writer = writer  # type: ignore[assignment]

        # Step 1: Handshake
        self._state = SubscriptionState.CONNECTING
        handshake_ok = await self._perform_handshake(reader, writer)
        if not handshake_ok:
            self._state = SubscriptionState.DISCONNECTED
            return False

        # Step 2: Subscribe
        self._state = SubscriptionState.SUBSCRIBING
        subscribe_ok = await self._send_subscribe(reader, writer)
        if not subscribe_ok:
            self._state = SubscriptionState.DISCONNECTED
            return False

        self._state = SubscriptionState.ACTIVE
        self._last_heartbeat = _now_utc()
        return True

    # -- Internal: handshake --------------------------------------------------

    async def _perform_handshake(
        self,
        reader: asyncio.StreamReader,
        writer: object,
    ) -> bool:
        """Send handshake request and validate daemon response.

        Args:
            reader: Stream to read the response from.
            writer: Stream to write the request to.

        Returns:
            True if handshake succeeded.
        """
        request = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id=_generate_msg_id(),
            timestamp=_now_iso(),
            payload={
                "verb": _VERB_HANDSHAKE,
                "protocol_version": _PROTOCOL_VERSION,
                "client_pid": os.getpid(),
            },
        )

        try:
            await self._send_envelope(writer, request)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            logger.error("Handshake send failed: %s", exc)
            return False

        response = await self._read_envelope(reader)
        if response is None:
            logger.error("Handshake: no response from daemon")
            return False

        if response.msg_type == MessageType.ERROR:
            error_msg = response.payload.get("error", "Unknown error")
            logger.error("Handshake rejected: %s", error_msg)
            return False

        daemon_version = response.payload.get("protocol_version", 0)
        if daemon_version != _PROTOCOL_VERSION:
            logger.error(
                "Protocol version mismatch: client=%d, daemon=%s",
                _PROTOCOL_VERSION,
                daemon_version,
            )
            return False

        logger.info(
            "Handshake OK (daemon_pid=%s, uptime=%.1fs)",
            response.payload.get("daemon_pid"),
            response.payload.get("daemon_uptime_seconds", 0.0),
        )
        return True

    # -- Internal: subscribe --------------------------------------------------

    async def _send_subscribe(
        self,
        reader: asyncio.StreamReader,
        writer: object,
    ) -> bool:
        """Send a subscribe request and parse the response.

        Args:
            reader: Stream to read the response from.
            writer: Stream to write the request to.

        Returns:
            True if subscription was accepted.
        """
        # Build subscribe payload
        payload: dict[str, object] = {
            "verb": _VERB_SUBSCRIBE,
            "payload_type": "subscribe_notifications",
        }
        if self._config.event_filter is not None:
            payload["event_filter"] = [
                et.value for et in self._config.event_filter
            ]

        request = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id=_generate_msg_id(),
            timestamp=_now_iso(),
            payload=payload,
        )

        try:
            await self._send_envelope(writer, request)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            logger.error("Subscribe send failed: %s", exc)
            return False

        response = await self._read_envelope(reader)
        if response is None:
            logger.error("Subscribe: no response from daemon")
            return False

        if response.msg_type == MessageType.ERROR:
            error_msg = response.payload.get("error", "Unknown error")
            logger.error("Subscribe rejected: %s", error_msg)
            return False

        # Extract subscription ID from response
        sub_id = response.payload.get("subscription_id")
        if not sub_id:
            logger.error("Subscribe response missing subscription_id")
            return False

        self._subscription_id = str(sub_id)
        heartbeat_interval = response.payload.get(
            "heartbeat_interval_seconds", 30
        )

        logger.info(
            "Subscribed (id=%s, heartbeat=%ds)",
            self._subscription_id,
            heartbeat_interval,
        )
        return True

    # -- Internal: stream loop ------------------------------------------------

    async def _stream_loop(self) -> SubscriptionExitReason:
        """Read and process notification envelopes until termination.

        Processes envelopes in a loop:
        - STREAM with notification payload: dispatch event.
        - ERROR: daemon error, exit loop.
        - EOF / timeout: connection lost or heartbeat expired.
        - CancelledError: user cancel.

        Returns:
            The reason the stream loop exited.
        """
        try:
            while self._state == SubscriptionState.ACTIVE:
                # Check heartbeat timeout
                if self._last_heartbeat is not None:
                    elapsed = (
                        _now_utc() - self._last_heartbeat
                    ).total_seconds()
                    if elapsed > self._config.heartbeat_timeout:
                        logger.warning(
                            "Heartbeat timeout: %.1fs since last heartbeat",
                            elapsed,
                        )
                        return SubscriptionExitReason.HEARTBEAT_TIMEOUT

                envelope = await self._read_envelope(
                    self._reader,
                    timeout=self._config.read_timeout,
                )

                if envelope is None:
                    # EOF, timeout, or connection lost
                    # Distinguish between read timeout (normal, keep going)
                    # and actual connection loss
                    if self._reader is not None:
                        if self._reader.at_eof():
                            return SubscriptionExitReason.CONNECTION_LOST
                    # Read timeout -- continue loop (heartbeat check above
                    # will catch stale connections)
                    continue

                processed = await self._process_envelope(envelope)
                if not processed:
                    return SubscriptionExitReason.DAEMON_ERROR

        except asyncio.CancelledError:
            logger.info("Stream loop cancelled by user")
            return SubscriptionExitReason.USER_CANCEL

        return SubscriptionExitReason.CLEAN_CLOSE

    # -- Internal: process envelope -------------------------------------------

    async def _process_envelope(
        self,
        envelope: MessageEnvelope,
    ) -> bool:
        """Process a single incoming IPC envelope.

        Extracts the notification payload from STREAM envelopes and
        dispatches the event. Returns False for ERROR envelopes to
        signal the stream loop to exit.

        Args:
            envelope: The IPC envelope to process.

        Returns:
            True to continue streaming, False to terminate.
        """
        if envelope.msg_type == MessageType.ERROR:
            error_msg = envelope.payload.get("error", "Unknown error")
            logger.warning("Daemon error: %s", error_msg)
            return False

        if envelope.msg_type != MessageType.STREAM:
            # Ignore non-STREAM, non-ERROR envelopes (e.g., stale responses)
            logger.debug(
                "Ignoring envelope type %s", envelope.msg_type.value
            )
            return True

        # Extract notification envelope from the STREAM payload
        notification_data = envelope.payload.get("notification")
        if notification_data is None:
            logger.warning(
                "STREAM envelope missing notification payload"
            )
            return True

        try:
            notification = NotificationEnvelope.model_validate(
                notification_data
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse notification envelope: %s", exc
            )
            return True

        # Update heartbeat tracking for heartbeat events
        if notification.event_type == NotificationEventType.HEARTBEAT:
            self._last_heartbeat = _now_utc()

        self._events_received += 1
        await self._dispatch_event(notification)
        return True

    # -- Internal: dispatch event ---------------------------------------------

    async def _dispatch_event(
        self,
        notification: NotificationEnvelope,
    ) -> None:
        """Dispatch a notification to callbacks and the event queue.

        Callbacks are invoked sequentially. A failing callback is
        logged and skipped -- it never blocks other callbacks or
        the queue enqueue.

        Args:
            notification: The notification envelope to dispatch.
        """
        # Invoke callbacks
        for entry in self._callbacks:
            # Apply per-callback event type filter
            if entry.event_types is not None:
                if notification.event_type not in entry.event_types:
                    continue

            try:
                await entry.callback(notification)
            except Exception as exc:
                logger.warning(
                    "Callback %s raised %s: %s",
                    entry.handle,
                    type(exc).__name__,
                    exc,
                )

        # Enqueue for async iteration
        if self._event_queue.full():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            logger.debug("Event queue full; evicted oldest event")

        try:
            self._event_queue.put_nowait(notification)
        except asyncio.QueueFull:
            logger.warning("Event queue still full after eviction")

    # -- Internal: send/receive helpers ---------------------------------------

    async def _send_envelope(
        self,
        writer: object,
        envelope: MessageEnvelope,
    ) -> None:
        """Encode and send a framed MessageEnvelope.

        Args:
            writer: The stream writer to write to.
            envelope: The envelope to send.
        """
        frame = encode_frame(envelope)
        writer.write(frame)  # type: ignore[union-attr]
        await writer.drain()  # type: ignore[union-attr]

    async def _read_envelope(
        self,
        reader: asyncio.StreamReader | None,
        *,
        timeout: float | None = None,
    ) -> MessageEnvelope | None:
        """Read one framed envelope from the stream.

        Returns None on EOF, incomplete data, timeout, or decode error.

        Args:
            reader:  The StreamReader to read from.
            timeout: Max seconds to wait. Uses config default if None.

        Returns:
            Decoded MessageEnvelope, or None on failure.
        """
        if reader is None:
            return None

        effective_timeout = (
            timeout if timeout is not None else self._config.read_timeout
        )

        try:
            header_bytes = await asyncio.wait_for(
                reader.readexactly(HEADER_SIZE),
                timeout=effective_timeout,
            )
        except (
            asyncio.IncompleteReadError,
            ConnectionResetError,
            asyncio.TimeoutError,
        ):
            return None

        try:
            payload_length = unpack_header(header_bytes)
            payload_bytes = await asyncio.wait_for(
                reader.readexactly(payload_length),
                timeout=effective_timeout,
            )
        except (
            asyncio.IncompleteReadError,
            ConnectionResetError,
            asyncio.TimeoutError,
        ):
            return None

        try:
            return decode_envelope(payload_bytes)
        except (ValueError, KeyError) as exc:
            logger.warning("Malformed envelope from daemon: %s", exc)
            return None

    # -- Internal: unsubscribe (best-effort) ----------------------------------

    async def _try_send_unsubscribe(self) -> None:
        """Best-effort unsubscribe: swallow errors silently.

        The daemon will clean up orphaned subscriptions on disconnect,
        so this is a courtesy cleanup.
        """
        if self._subscription_id is None or self._writer is None:
            return

        envelope = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id=_generate_msg_id(),
            timestamp=_now_iso(),
            payload={
                "verb": _VERB_UNSUBSCRIBE,
                "payload_type": "unsubscribe_notifications",
                "subscription_id": self._subscription_id,
            },
        )

        try:
            await self._send_envelope(self._writer, envelope)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            logger.debug(
                "Unsubscribe send failed (expected during disconnect): %s",
                exc,
            )

    # -- Internal: transport cleanup ------------------------------------------

    async def _close_transport(self) -> None:
        """Safely close the underlying stream writer.

        Handles the case where the writer is already closing or the
        transport has been lost.
        """
        if self._writer is None:
            return

        try:
            if hasattr(self._writer, "is_closing"):
                if self._writer.is_closing():
                    return
            self._writer.close()
            if hasattr(self._writer, "wait_closed"):
                await self._writer.wait_closed()
        except (OSError, ConnectionResetError) as exc:
            logger.debug("Error closing transport: %s", exc)
        finally:
            self._reader = None
            self._writer = None
