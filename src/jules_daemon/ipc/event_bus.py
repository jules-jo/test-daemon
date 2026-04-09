"""Async event bus for internal daemon event routing.

Provides a lightweight, typed publish/subscribe event bus for decoupling
daemon components. Events are immutable dataclasses routed by string
event type. Subscribers are async callables invoked sequentially in
subscription order.

Key design decisions:

- **Sequential dispatch**: Subscribers are called in subscription order,
  not concurrently, to preserve causal ordering (e.g., connect before
  disconnect events).
- **Error isolation**: A failing subscriber never blocks or crashes other
  subscribers or the emitter. Errors are logged and swallowed.
- **Subscription handles**: ``subscribe()`` returns a ``Subscription``
  object that can be passed to ``unsubscribe()`` for clean teardown.
- **No persistence**: The event bus is purely in-memory. For durable
  events, the caller must persist to the wiki layer separately.

Usage::

    from jules_daemon.ipc.event_bus import Event, EventBus

    bus = EventBus()

    async def on_connect(event: Event) -> None:
        print(f"Client connected: {event.payload}")

    sub = bus.subscribe("client_connected", on_connect)
    await bus.emit(Event(event_type="client_connected", payload={"id": "c1"}))

    bus.unsubscribe(sub)
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

__all__ = [
    "Event",
    "EventBus",
    "EventHandler",
    "Subscription",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

EventHandler = Callable[["Event"], Awaitable[None]]
"""Async callable that receives an Event. Must not raise to callers."""


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Event:
    """Immutable event emitted through the event bus.

    Attributes:
        event_type: String identifier for the event category.
            Used for routing to subscribers.
        payload:    JSON-serializable dict with event-specific data.
            Defaults to an empty dict.
        timestamp:  ISO 8601 UTC timestamp of event creation.
            Auto-populated if not provided.
    """

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default="")

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, str) or not self.event_type.strip():
            raise ValueError("event_type must not be empty")
        # Auto-populate timestamp if not provided
        if not self.timestamp:
            object.__setattr__(self, "timestamp", _now_iso())


# ---------------------------------------------------------------------------
# Subscription dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Subscription:
    """Immutable handle for a registered event subscription.

    Used to unsubscribe a specific handler from the event bus.

    Attributes:
        subscription_id: Unique identifier for this subscription.
        event_type:      The event type this subscription listens for.
    """

    subscription_id: str
    event_type: str


# ---------------------------------------------------------------------------
# Internal subscriber entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SubscriberEntry:
    """Internal mapping between a subscription handle and its handler."""

    subscription: Subscription
    handler: EventHandler


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Async event bus for typed publish/subscribe within the daemon.

    Maintains a registry of subscribers keyed by event type. Events are
    dispatched sequentially in subscription order. Subscriber exceptions
    are caught and logged without propagating.

    Thread safety note: This class is designed for single-threaded async
    use within one event loop. It does not use locks.

    Usage::

        bus = EventBus()

        async def handler(event: Event) -> None:
            ...

        sub = bus.subscribe("my_event", handler)
        await bus.emit(Event(event_type="my_event", payload={"key": "val"}))
        bus.unsubscribe(sub)
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[_SubscriberEntry]] = defaultdict(list)

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> Subscription:
        """Register an async handler for a specific event type.

        Args:
            event_type: The event type to listen for.
            handler:    Async callable invoked when a matching event is emitted.

        Returns:
            A Subscription handle that can be passed to ``unsubscribe()``.
        """
        subscription = Subscription(
            subscription_id=f"sub-{uuid.uuid4().hex[:12]}",
            event_type=event_type,
        )
        entry = _SubscriberEntry(subscription=subscription, handler=handler)
        self._subscribers[event_type].append(entry)

        logger.debug(
            "Subscribed %s to event_type=%s",
            subscription.subscription_id,
            event_type,
        )
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a previously registered subscription.

        Idempotent: calling with an already-removed subscription is a
        safe no-op.

        Args:
            subscription: The subscription handle returned by ``subscribe()``.
        """
        entries = self._subscribers.get(subscription.event_type)
        if entries is None:
            return

        original_count = len(entries)
        self._subscribers[subscription.event_type] = [
            entry
            for entry in entries
            if entry.subscription.subscription_id != subscription.subscription_id
        ]

        new_count = len(self._subscribers[subscription.event_type])
        if new_count < original_count:
            logger.debug(
                "Unsubscribed %s from event_type=%s",
                subscription.subscription_id,
                subscription.event_type,
            )

        # Clean up empty lists
        if not self._subscribers[subscription.event_type]:
            del self._subscribers[subscription.event_type]

    async def emit(self, event: Event) -> None:
        """Dispatch an event to all subscribers of its type.

        Subscribers are invoked sequentially in subscription order.
        If a subscriber raises an exception, it is logged and the
        remaining subscribers are still invoked.

        Args:
            event: The event to dispatch.
        """
        entries = self._subscribers.get(event.event_type)
        if not entries:
            logger.debug(
                "No subscribers for event_type=%s", event.event_type
            )
            return

        for entry in entries:
            try:
                await entry.handler(event)
            except Exception as exc:
                logger.warning(
                    "Subscriber %s raised %s for event_type=%s: %s",
                    entry.subscription.subscription_id,
                    type(exc).__name__,
                    event.event_type,
                    exc,
                )

    def subscriber_count(self, event_type: str) -> int:
        """Return the number of subscribers for a given event type.

        Args:
            event_type: The event type to query.

        Returns:
            Number of active subscriptions for that type.
        """
        entries = self._subscribers.get(event_type)
        if entries is None:
            return 0
        return len(entries)

    def has_subscribers(self, event_type: str) -> bool:
        """Check whether any subscribers exist for a given event type.

        Args:
            event_type: The event type to query.

        Returns:
            True if at least one subscriber is registered.
        """
        return self.subscriber_count(event_type) > 0
