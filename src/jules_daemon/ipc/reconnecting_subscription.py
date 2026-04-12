"""Reconnecting subscription wrapper with deduplication and backoff.

Wraps the base ``SubscriptionClient`` with transparent reconnection
on transient disconnections. When the underlying subscription session
ends due to a retriable condition (CONNECTION_LOST or HEARTBEAT_TIMEOUT),
the wrapper:

    1. Records the disconnection as a ``ReconnectAttemptRecord``.
    2. Calculates an exponential backoff delay using ``BackoffConfig``.
    3. Sleeps for the computed delay.
    4. Creates a fresh ``SubscriptionClient`` and re-establishes the stream.
    5. Pipes incoming events through an ``EventIdTracker`` deduplication
       filter so events already seen before the reconnection are skipped.

The wrapper maintains a single ``EventIdTracker`` across all reconnection
sessions, enabling SSE-style resume-from-last-event-ID semantics without
requiring daemon-side replay support.

Permanent exit reasons (CLEAN_CLOSE, USER_CANCEL, DAEMON_ERROR,
SUBSCRIBE_FAILED) terminate the wrapper immediately without retries.

Architecture::

    CLI Process
        |
    ReconnectingSubscription.run()
        |
        +-- loop:
        |   |
        |   +-- _create_client() -> SubscriptionClient
        |   +-- _apply_callbacks(client)
        |   +-- client.run() -> SubscriptionResult
        |   |
        |   +-- if retriable exit:
        |   |       record attempt
        |   |       calculate_delay(backoff_config, attempt)
        |   |       await asyncio.sleep(delay)
        |   |       continue loop
        |   |
        |   +-- if permanent exit:
        |           break loop
        |
        +-- return ReconnectingSubscriptionResult

Key design decisions:

- **Frozen config**: ``ReconnectingSubscriptionConfig`` is immutable and
  captures all reconnection policy parameters, backoff config, dedup
  config, and event filter preferences.

- **Deduplication**: A single ``EventIdTracker`` instance persists across
  reconnection sessions. Events seen before a disconnect are recorded
  in the tracker; after reconnection, replayed events are detected as
  duplicates and filtered out before reaching consumers.

- **Callback persistence**: Callbacks registered via ``on_event()`` are
  stored on the wrapper and re-applied to each fresh ``SubscriptionClient``
  instance. Consumers register once and receive events transparently
  across reconnections.

- **Event queue persistence**: The wrapper maintains its own event queue
  that spans reconnection boundaries. Consumers iterating the wrapper
  see a continuous stream of events.

- **Backoff delegation**: Delay calculation is delegated to the existing
  ``ssh.backoff.calculate_delay()`` pure function. The wrapper only
  calls ``asyncio.sleep()`` with the computed delay.

- **Close signal**: The wrapper checks a ``_closed`` flag after each
  base client run and after each backoff sleep. Calling ``close()``
  sets the flag, causing the loop to terminate cleanly.

Usage::

    from jules_daemon.ipc.reconnecting_subscription import (
        ReconnectingSubscription,
        ReconnectingSubscriptionConfig,
    )

    config = ReconnectingSubscriptionConfig(
        socket_path="/run/jules/daemon.sock",
        max_reconnect_attempts=5,
    )
    wrapper = ReconnectingSubscription(config=config)
    wrapper.on_event(my_handler)

    result = await wrapper.run()
    print(f"Exited: {result.exit_reason.value}")
    print(f"Reconnections: {result.reconnect_count}")
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable

from jules_daemon.ipc.event_dedup import (
    EventDeduplicationConfig,
    EventIdTracker,
)
from jules_daemon.ipc.subscription_client import (
    EventCallback,
    SubscriptionClient,
    SubscriptionClientConfig,
    SubscriptionExitReason,
)
from jules_daemon.protocol.notifications import (
    NotificationEnvelope,
    NotificationEventType,
)
from jules_daemon.ssh.backoff import BackoffConfig, calculate_delay

__all__ = [
    "ReconnectAttemptRecord",
    "ReconnectingSubscription",
    "ReconnectingSubscriptionConfig",
    "ReconnectingSubscriptionResult",
    "ReconnectionExitReason",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RECONNECT_ATTEMPTS: int = 5
_DEFAULT_EVENT_QUEUE_SIZE: int = 1000

# Exit reasons that should trigger a reconnection attempt
_RETRIABLE_EXIT_REASONS: frozenset[SubscriptionExitReason] = frozenset(
    {
        SubscriptionExitReason.CONNECTION_LOST,
        SubscriptionExitReason.HEARTBEAT_TIMEOUT,
    }
)


# ---------------------------------------------------------------------------
# ReconnectionExitReason enum
# ---------------------------------------------------------------------------


class ReconnectionExitReason(Enum):
    """Why the reconnecting subscription wrapper terminated.

    Values:
        CLEAN_CLOSE:        Base client exited cleanly (no error).
        RECONNECT_EXHAUSTED: Max reconnect attempts exceeded.
        PERMANENT_ERROR:     Base client hit a non-retriable error
                             (DAEMON_ERROR, SUBSCRIBE_FAILED).
        USER_CANCEL:         User cancelled or CancelledError received.
        CLOSED:              close() was called on the wrapper.
    """

    CLEAN_CLOSE = "clean_close"
    RECONNECT_EXHAUSTED = "reconnect_exhausted"
    PERMANENT_ERROR = "permanent_error"
    USER_CANCEL = "user_cancel"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# ReconnectAttemptRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconnectAttemptRecord:
    """Immutable record of a single reconnection attempt.

    Captures the context of why the previous session ended and how
    long the wrapper waited before retrying.

    Attributes:
        attempt:                  Zero-indexed reconnection attempt number.
        exit_reason:              Why the previous session ended.
        delay_seconds:            Backoff delay applied before this reconnect.
        events_before_disconnect: Events received in the session that ended.
    """

    attempt: int
    exit_reason: SubscriptionExitReason
    delay_seconds: float
    events_before_disconnect: int


# ---------------------------------------------------------------------------
# ReconnectingSubscriptionConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconnectingSubscriptionConfig:
    """Immutable configuration for the reconnecting subscription wrapper.

    Attributes:
        socket_path:             Path to the daemon's Unix domain socket.
        max_reconnect_attempts:  Max reconnection attempts before giving up.
                                 Must be positive (>= 1).
        backoff_config:          Exponential backoff parameters for retry
                                 delays. Uses sensible defaults when None.
        dedup_config:            Configuration for the event ID tracker.
                                 Uses sensible defaults when None.
        event_filter:            Optional set of event types to subscribe to.
                                 Passed through to each SubscriptionClient.
        connect_timeout:         Per-connection timeout in seconds.
        heartbeat_timeout:       Heartbeat timeout in seconds.
        read_timeout:            Per-read timeout in seconds.
        event_queue_size:        Max events buffered for async iteration.
    """

    socket_path: str
    max_reconnect_attempts: int = _DEFAULT_MAX_RECONNECT_ATTEMPTS
    backoff_config: BackoffConfig = field(default_factory=BackoffConfig)
    dedup_config: EventDeduplicationConfig = field(
        default_factory=EventDeduplicationConfig
    )
    event_filter: frozenset[NotificationEventType] | None = None
    connect_timeout: float = 5.0
    heartbeat_timeout: float = 90.0
    read_timeout: float = 10.0
    event_queue_size: int = _DEFAULT_EVENT_QUEUE_SIZE

    def __post_init__(self) -> None:
        if not self.socket_path or not self.socket_path.strip():
            raise ValueError("socket_path must not be empty")
        if self.max_reconnect_attempts < 1:
            raise ValueError(
                f"max_reconnect_attempts must be positive, "
                f"got {self.max_reconnect_attempts}"
            )


# ---------------------------------------------------------------------------
# ReconnectingSubscriptionResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconnectingSubscriptionResult:
    """Immutable result of a reconnecting subscription session.

    Aggregates metrics across all reconnection sessions.

    Attributes:
        exit_reason:              Why the wrapper terminated.
        total_events_received:    Total events received across all sessions.
        total_events_deduplicated: Events that were filtered as duplicates.
        reconnect_count:          Number of reconnection attempts made.
        reconnect_history:        Ordered tuple of reconnection attempt records.
        last_seen_event_id:       Most recent event ID seen (for resume).
    """

    exit_reason: ReconnectionExitReason
    total_events_received: int
    total_events_deduplicated: int
    reconnect_count: int
    reconnect_history: tuple[ReconnectAttemptRecord, ...]
    last_seen_event_id: str | None


# ---------------------------------------------------------------------------
# Internal: callback entry for persistence across reconnections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PersistentCallback:
    """Internal record of a callback registered on the wrapper.

    Stored on the wrapper and re-applied to each fresh SubscriptionClient.

    Attributes:
        callback:    Async callable to invoke with each event.
        event_types: Optional filter for specific event types.
    """

    callback: EventCallback
    event_types: frozenset[NotificationEventType] | None = None


# ---------------------------------------------------------------------------
# ReconnectingSubscription
# ---------------------------------------------------------------------------


# Type alias for the optional on_reconnect callback
OnReconnectCallback = Callable[
    [ReconnectAttemptRecord], Awaitable[None]
]


class ReconnectingSubscription:
    """Subscription wrapper with transparent reconnection and deduplication.

    Wraps the base ``SubscriptionClient`` and automatically reconnects
    on transient disconnections. Events are deduplicated across
    reconnection boundaries using an ``EventIdTracker``.

    Supports the same async iterator and callback interfaces as the
    base ``SubscriptionClient``, but callbacks and the event queue
    persist across reconnections.

    Args:
        config:       Reconnection and subscription configuration.
        on_reconnect: Optional async callback invoked for each
                      reconnection attempt. Receives a
                      ``ReconnectAttemptRecord``. Errors are logged
                      and swallowed.
    """

    def __init__(
        self,
        *,
        config: ReconnectingSubscriptionConfig,
        on_reconnect: OnReconnectCallback | None = None,
    ) -> None:
        self._config = config
        self._on_reconnect = on_reconnect
        self._dedup_tracker = EventIdTracker(config=config.dedup_config)
        self._callbacks: list[_PersistentCallback] = []
        self._event_queue: asyncio.Queue[NotificationEnvelope | None] = (
            asyncio.Queue(maxsize=config.event_queue_size)
        )
        self._closed = False
        self._total_events_received: int = 0
        self._total_events_deduplicated: int = 0

    # -- Properties -----------------------------------------------------------

    @property
    def config(self) -> ReconnectingSubscriptionConfig:
        """The reconnection configuration."""
        return self._config

    @property
    def last_seen_event_id(self) -> str | None:
        """Most recently recorded event ID from the dedup tracker."""
        return self._dedup_tracker.last_seen_event_id

    # -- Async iterator interface ---------------------------------------------

    def __aiter__(self) -> AsyncIterator[NotificationEnvelope]:
        """Return self as an async iterator over notification events."""
        return self

    async def __anext__(self) -> NotificationEnvelope:
        """Yield the next notification event from the persistent queue.

        Returns the next event, or raises StopAsyncIteration when a
        None sentinel is received (indicating shutdown).
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
    ) -> None:
        """Register a persistent callback for notification events.

        Callbacks registered on the wrapper are re-applied to each
        fresh ``SubscriptionClient`` instance across reconnections.
        They also receive events that pass deduplication.

        Args:
            callback:    Async callable receiving a NotificationEnvelope.
            event_types: Optional filter for specific event types.
        """
        entry = _PersistentCallback(
            callback=callback,
            event_types=event_types,
        )
        # Create new list for immutability
        self._callbacks = [*self._callbacks, entry]

    # -- Public API: run / close ----------------------------------------------

    async def run(self) -> ReconnectingSubscriptionResult:
        """Connect, subscribe, stream, and auto-reconnect on transient errors.

        This is the main entry point. It creates a SubscriptionClient,
        runs it, and if the session ends with a retriable exit reason,
        calculates a backoff delay, sleeps, and retries. Events are
        deduplicated across reconnection boundaries.

        Returns:
            ReconnectingSubscriptionResult with aggregated metrics.
        """
        reconnect_history: list[ReconnectAttemptRecord] = []
        reconnect_count = 0

        try:
            while not self._closed:
                # Create a fresh subscription client for this session
                client = self._create_client()
                self._apply_callbacks(client)

                # Run the base subscription session
                result = await client.run()

                # Accumulate metrics
                self._total_events_received += result.events_received

                # Check if we should reconnect
                if result.exit_reason not in _RETRIABLE_EXIT_REASONS:
                    # Non-retriable exit -- determine the wrapper exit reason
                    wrapper_reason = _map_exit_reason(result.exit_reason)
                    return self._build_result(
                        exit_reason=wrapper_reason,
                        reconnect_history=reconnect_history,
                    )

                # Retriable exit -- check reconnection budget
                if reconnect_count >= self._config.max_reconnect_attempts:
                    logger.warning(
                        "Reconnection attempts exhausted (%d/%d). "
                        "Last exit: %s",
                        reconnect_count,
                        self._config.max_reconnect_attempts,
                        result.exit_reason.value,
                    )
                    return self._build_result(
                        exit_reason=ReconnectionExitReason.RECONNECT_EXHAUSTED,
                        reconnect_history=reconnect_history,
                    )

                # Calculate backoff delay
                delay = calculate_delay(
                    self._config.backoff_config,
                    reconnect_count,
                )

                # Record this reconnection attempt
                record = ReconnectAttemptRecord(
                    attempt=reconnect_count,
                    exit_reason=result.exit_reason,
                    delay_seconds=delay.total,
                    events_before_disconnect=result.events_received,
                )
                reconnect_history.append(record)
                reconnect_count += 1

                logger.info(
                    "Reconnecting (attempt %d/%d) after %s. "
                    "Backoff delay: %.2fs. Events before disconnect: %d",
                    reconnect_count,
                    self._config.max_reconnect_attempts,
                    result.exit_reason.value,
                    delay.total,
                    result.events_received,
                )

                # Invoke optional on_reconnect callback
                await self._invoke_on_reconnect(record)

                # Sleep for backoff delay (can be cancelled by close())
                await asyncio.sleep(delay.total)

                # Check if closed during sleep
                if self._closed:
                    return self._build_result(
                        exit_reason=ReconnectionExitReason.CLOSED,
                        reconnect_history=reconnect_history,
                    )

        except asyncio.CancelledError:
            logger.info("Reconnecting subscription cancelled by user")
            return self._build_result(
                exit_reason=ReconnectionExitReason.USER_CANCEL,
                reconnect_history=reconnect_history,
            )

        # Reached when self._closed is True at loop start
        return self._build_result(
            exit_reason=ReconnectionExitReason.CLOSED,
            reconnect_history=reconnect_history,
        )

    async def close(self) -> None:
        """Signal the reconnection loop to stop.

        Sets the closed flag so the loop terminates after the current
        iteration. Posts a None sentinel to the event queue so async
        iterators stop. Idempotent.
        """
        if self._closed:
            return

        self._closed = True

        # Post sentinel for async iterator consumers
        try:
            self._event_queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._event_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass  # Best effort

        logger.debug("Reconnecting subscription closed")

    # -- Internal: client factory ---------------------------------------------

    def _create_client(self) -> SubscriptionClient:
        """Create a fresh SubscriptionClient for a new session.

        Separated as a method for testability -- tests can override this
        to inject mocked clients.

        Returns:
            A new SubscriptionClient configured from the wrapper's config.
        """
        client_config = SubscriptionClientConfig(
            socket_path=self._config.socket_path,
            connect_timeout=self._config.connect_timeout,
            heartbeat_timeout=self._config.heartbeat_timeout,
            read_timeout=self._config.read_timeout,
            event_filter=self._config.event_filter,
            event_queue_size=self._config.event_queue_size,
            auto_reconnect=False,  # Wrapper handles reconnection
        )
        return SubscriptionClient(config=client_config)

    # -- Internal: apply callbacks to a fresh client --------------------------

    def _apply_callbacks(self, client: SubscriptionClient) -> None:
        """Re-apply all persistent callbacks to a fresh client.

        Each callback is wrapped in a dedup-filtering adapter that
        checks the event against the EventIdTracker before forwarding
        to the original callback. This ensures events seen before
        reconnection are not delivered again.

        Args:
            client: The SubscriptionClient to register callbacks on.
        """
        for entry in self._callbacks:
            # Create a dedup-filtering wrapper for this callback
            adapted = self._make_dedup_adapter(entry.callback)
            client.on_event(adapted, event_types=entry.event_types)

    def _make_dedup_adapter(
        self,
        callback: EventCallback,
    ) -> EventCallback:
        """Create a callback adapter that filters duplicates.

        The adapter checks each event against the dedup tracker.
        New events are recorded and forwarded; duplicates are skipped.

        Args:
            callback: The original callback to wrap.

        Returns:
            An async callback that filters duplicates before forwarding.
        """

        async def adapter(envelope: NotificationEnvelope) -> None:
            verdict = self._dedup_tracker.filter_envelope(envelope)
            if verdict.is_duplicate:
                self._total_events_deduplicated += 1
                logger.debug(
                    "Dedup: skipping duplicate event %s",
                    envelope.event_id,
                )
                return

            # Forward non-duplicate events to the original callback
            await callback(envelope)

            # Also enqueue for the wrapper's async iterator
            self._enqueue_event(envelope)

        return adapter

    # -- Internal: event queue management -------------------------------------

    def _enqueue_event(self, notification: NotificationEnvelope) -> None:
        """Enqueue an event for the wrapper's async iterator.

        Uses the same eviction strategy as the base SubscriptionClient:
        when the queue is full, the oldest event is evicted.

        Args:
            notification: The event to enqueue.
        """
        if self._event_queue.full():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            logger.debug("Wrapper event queue full; evicted oldest event")

        try:
            self._event_queue.put_nowait(notification)
        except asyncio.QueueFull:
            logger.warning("Wrapper event queue still full after eviction")

    # -- Internal: build result -----------------------------------------------

    def _build_result(
        self,
        *,
        exit_reason: ReconnectionExitReason,
        reconnect_history: list[ReconnectAttemptRecord],
    ) -> ReconnectingSubscriptionResult:
        """Build the final result from accumulated metrics.

        Args:
            exit_reason:       Why the wrapper terminated.
            reconnect_history: Ordered list of reconnection attempt records.

        Returns:
            Immutable result with aggregated metrics.
        """
        return ReconnectingSubscriptionResult(
            exit_reason=exit_reason,
            total_events_received=self._total_events_received,
            total_events_deduplicated=self._total_events_deduplicated,
            reconnect_count=len(reconnect_history),
            reconnect_history=tuple(reconnect_history),
            last_seen_event_id=self._dedup_tracker.last_seen_event_id,
        )

    # -- Internal: invoke on_reconnect callback --------------------------------

    async def _invoke_on_reconnect(
        self,
        record: ReconnectAttemptRecord,
    ) -> None:
        """Invoke the optional on_reconnect callback with error isolation.

        A failing callback is logged and swallowed -- it never breaks
        the reconnection loop.

        Args:
            record: The reconnection attempt record.
        """
        if self._on_reconnect is None:
            return

        try:
            await self._on_reconnect(record)
        except Exception as exc:
            logger.warning(
                "on_reconnect callback raised %s: %s",
                type(exc).__name__,
                exc,
            )


# ---------------------------------------------------------------------------
# Internal helper: map base exit reason to wrapper exit reason
# ---------------------------------------------------------------------------


def _map_exit_reason(
    base_reason: SubscriptionExitReason,
) -> ReconnectionExitReason:
    """Map a base subscription exit reason to a wrapper exit reason.

    Called only for non-retriable exits. Retriable exits (CONNECTION_LOST,
    HEARTBEAT_TIMEOUT) are handled by the reconnection loop.

    Args:
        base_reason: The exit reason from the base SubscriptionClient.

    Returns:
        The corresponding ReconnectionExitReason.
    """
    mapping: dict[SubscriptionExitReason, ReconnectionExitReason] = {
        SubscriptionExitReason.CLEAN_CLOSE: ReconnectionExitReason.CLEAN_CLOSE,
        SubscriptionExitReason.USER_CANCEL: ReconnectionExitReason.USER_CANCEL,
        SubscriptionExitReason.DAEMON_ERROR: ReconnectionExitReason.PERMANENT_ERROR,
        SubscriptionExitReason.SUBSCRIBE_FAILED: ReconnectionExitReason.PERMANENT_ERROR,
    }
    return mapping.get(base_reason, ReconnectionExitReason.PERMANENT_ERROR)
