"""Tests for the async event bus.

Covers:
    - Subscribing and receiving events
    - Multiple subscribers for the same event type
    - Unsubscribing stops delivery
    - Events with different types are routed to correct subscribers
    - Subscriber exceptions do not crash the bus or block other subscribers
    - Event dataclass immutability
    - Subscribe returns an unsubscribe handle
    - Emitting to no subscribers is a safe no-op
    - Event ordering (subscribers receive in subscription order)
    - Type safety on event names
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from jules_daemon.ipc.event_bus import (
    Event,
    EventBus,
    Subscription,
)


# ---------------------------------------------------------------------------
# Event model tests
# ---------------------------------------------------------------------------


class TestEvent:
    """Tests for the immutable Event dataclass."""

    def test_create_event(self) -> None:
        event = Event(event_type="client_connected", payload={"id": "c1"})
        assert event.event_type == "client_connected"
        assert event.payload == {"id": "c1"}

    def test_frozen(self) -> None:
        event = Event(event_type="test", payload={})
        with pytest.raises(AttributeError):
            event.event_type = "mutated"  # type: ignore[misc]

    def test_default_payload_is_empty_dict(self) -> None:
        event = Event(event_type="test")
        assert event.payload == {}

    def test_empty_event_type_raises(self) -> None:
        with pytest.raises(ValueError, match="event_type must not be empty"):
            Event(event_type="", payload={})

    def test_whitespace_event_type_raises(self) -> None:
        with pytest.raises(ValueError, match="event_type must not be empty"):
            Event(event_type="   ", payload={})

    def test_timestamp_is_populated(self) -> None:
        event = Event(event_type="test")
        assert event.timestamp != ""
        assert "T" in event.timestamp  # ISO 8601 format


# ---------------------------------------------------------------------------
# Subscription model tests
# ---------------------------------------------------------------------------


class TestSubscription:
    """Tests for the Subscription dataclass."""

    def test_create_subscription(self) -> None:
        sub = Subscription(
            subscription_id="sub-001",
            event_type="client_connected",
        )
        assert sub.subscription_id == "sub-001"
        assert sub.event_type == "client_connected"

    def test_frozen(self) -> None:
        sub = Subscription(
            subscription_id="sub-001",
            event_type="test",
        )
        with pytest.raises(AttributeError):
            sub.subscription_id = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EventBus subscribe/emit tests
# ---------------------------------------------------------------------------


class TestEventBusSubscription:
    """Tests for event subscription and emission."""

    @pytest.mark.asyncio
    async def test_subscribe_and_receive_event(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("client_connected", handler)
        await bus.emit(Event(event_type="client_connected", payload={"id": "c1"}))

        assert len(received) == 1
        assert received[0].payload == {"id": "c1"}

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_event(self) -> None:
        bus = EventBus()
        received_a: list[Event] = []
        received_b: list[Event] = []

        async def handler_a(event: Event) -> None:
            received_a.append(event)

        async def handler_b(event: Event) -> None:
            received_b.append(event)

        bus.subscribe("test_event", handler_a)
        bus.subscribe("test_event", handler_b)
        await bus.emit(Event(event_type="test_event"))

        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_events_routed_to_correct_type(self) -> None:
        bus = EventBus()
        received_connect: list[Event] = []
        received_disconnect: list[Event] = []

        async def on_connect(event: Event) -> None:
            received_connect.append(event)

        async def on_disconnect(event: Event) -> None:
            received_disconnect.append(event)

        bus.subscribe("connected", on_connect)
        bus.subscribe("disconnected", on_disconnect)

        await bus.emit(Event(event_type="connected", payload={"id": "c1"}))
        await bus.emit(Event(event_type="disconnected", payload={"id": "c2"}))

        assert len(received_connect) == 1
        assert received_connect[0].payload == {"id": "c1"}
        assert len(received_disconnect) == 1
        assert received_disconnect[0].payload == {"id": "c2"}

    @pytest.mark.asyncio
    async def test_emit_with_no_subscribers_is_safe(self) -> None:
        bus = EventBus()
        # Should not raise
        await bus.emit(Event(event_type="no_listeners"))

    @pytest.mark.asyncio
    async def test_subscriber_ordering_preserved(self) -> None:
        bus = EventBus()
        order: list[int] = []

        async def handler_1(event: Event) -> None:
            order.append(1)

        async def handler_2(event: Event) -> None:
            order.append(2)

        async def handler_3(event: Event) -> None:
            order.append(3)

        bus.subscribe("ordered", handler_1)
        bus.subscribe("ordered", handler_2)
        bus.subscribe("ordered", handler_3)

        await bus.emit(Event(event_type="ordered"))
        assert order == [1, 2, 3]


# ---------------------------------------------------------------------------
# EventBus unsubscribe tests
# ---------------------------------------------------------------------------


class TestEventBusUnsubscribe:
    """Tests for unsubscription behavior."""

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        subscription = bus.subscribe("test_event", handler)

        await bus.emit(Event(event_type="test_event"))
        assert len(received) == 1

        bus.unsubscribe(subscription)

        await bus.emit(Event(event_type="test_event"))
        assert len(received) == 1  # No new delivery

    @pytest.mark.asyncio
    async def test_unsubscribe_only_affects_target(self) -> None:
        bus = EventBus()
        received_a: list[Event] = []
        received_b: list[Event] = []

        async def handler_a(event: Event) -> None:
            received_a.append(event)

        async def handler_b(event: Event) -> None:
            received_b.append(event)

        sub_a = bus.subscribe("test_event", handler_a)
        bus.subscribe("test_event", handler_b)

        bus.unsubscribe(sub_a)

        await bus.emit(Event(event_type="test_event"))
        assert len(received_a) == 0
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_double_unsubscribe_is_safe(self) -> None:
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        subscription = bus.subscribe("test_event", handler)
        bus.unsubscribe(subscription)
        # Should not raise
        bus.unsubscribe(subscription)

    @pytest.mark.asyncio
    async def test_subscribe_returns_subscription(self) -> None:
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        subscription = bus.subscribe("client_connected", handler)
        assert isinstance(subscription, Subscription)
        assert subscription.event_type == "client_connected"
        assert subscription.subscription_id != ""


# ---------------------------------------------------------------------------
# EventBus error isolation tests
# ---------------------------------------------------------------------------


class TestEventBusErrorIsolation:
    """Tests for subscriber error isolation."""

    @pytest.mark.asyncio
    async def test_subscriber_exception_does_not_crash_bus(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def failing_handler(event: Event) -> None:
            raise RuntimeError("subscriber exploded")

        async def good_handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test_event", failing_handler)
        bus.subscribe("test_event", good_handler)

        # Should not raise; good_handler should still receive the event
        await bus.emit(Event(event_type="test_event"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_multiple_subscriber_failures_isolated(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def fail_1(event: Event) -> None:
            raise ValueError("fail 1")

        async def fail_2(event: Event) -> None:
            raise TypeError("fail 2")

        async def success(event: Event) -> None:
            received.append(event)

        bus.subscribe("test_event", fail_1)
        bus.subscribe("test_event", success)
        bus.subscribe("test_event", fail_2)

        await bus.emit(Event(event_type="test_event"))
        assert len(received) == 1


# ---------------------------------------------------------------------------
# EventBus subscriber count tests
# ---------------------------------------------------------------------------


class TestEventBusIntrospection:
    """Tests for event bus introspection methods."""

    def test_subscriber_count_empty(self) -> None:
        bus = EventBus()
        assert bus.subscriber_count("test_event") == 0

    def test_subscriber_count_after_subscribe(self) -> None:
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        bus.subscribe("test_event", handler)
        assert bus.subscriber_count("test_event") == 1

    def test_subscriber_count_after_unsubscribe(self) -> None:
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        sub = bus.subscribe("test_event", handler)
        bus.unsubscribe(sub)
        assert bus.subscriber_count("test_event") == 0

    def test_has_subscribers_false_when_empty(self) -> None:
        bus = EventBus()
        assert not bus.has_subscribers("test_event")

    def test_has_subscribers_true_when_subscribed(self) -> None:
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        bus.subscribe("test_event", handler)
        assert bus.has_subscribers("test_event")
