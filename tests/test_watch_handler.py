"""Tests for the IPC watch/unwatch subscription handler.

Validates that the WatchSubscriptionManager correctly bridges IPC
request-response messages to the JobOutputBroadcaster's subscribe
and unsubscribe API.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from jules_daemon.ipc.framing import MessageEnvelope, MessageType
from jules_daemon.ipc.server import ClientConnection
from jules_daemon.ipc.watch_handler import (
    UnwatchRequest,
    UnwatchResponse,
    WatchRequest,
    WatchResponse,
    WatchSubscriptionManager,
    parse_unwatch_request,
    parse_watch_request,
)
from jules_daemon.monitor.output_broadcaster import (
    JobOutputBroadcaster,
    SubscriberHandle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(client_id: str = "client-001") -> ClientConnection:
    """Build a minimal ClientConnection for testing."""
    return ClientConnection(
        client_id=client_id,
        reader=AsyncMock(),
        writer=AsyncMock(),
        connected_at="2026-04-09T12:00:00Z",
    )


def _make_watch_envelope(
    job_id: str = "job-abc",
    msg_id: str = "req-001",
) -> MessageEnvelope:
    """Build a watch request envelope."""
    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=msg_id,
        timestamp="2026-04-09T12:00:00Z",
        payload={"verb": "watch", "job_id": job_id},
    )


def _make_unwatch_envelope(
    job_id: str = "job-abc",
    subscriber_id: str = "sub-abc",
    msg_id: str = "req-002",
) -> MessageEnvelope:
    """Build an unwatch request envelope."""
    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=msg_id,
        timestamp="2026-04-09T12:00:00Z",
        payload={
            "verb": "unwatch",
            "job_id": job_id,
            "subscriber_id": subscriber_id,
        },
    )


# ---------------------------------------------------------------------------
# WatchRequest
# ---------------------------------------------------------------------------


class TestWatchRequest:
    """Tests for the immutable WatchRequest dataclass."""

    def test_create_with_job_id(self) -> None:
        req = WatchRequest(job_id="job-1")
        assert req.job_id == "job-1"

    def test_frozen(self) -> None:
        req = WatchRequest(job_id="job-1")
        with pytest.raises(AttributeError):
            req.job_id = "mutated"  # type: ignore[misc]

    def test_empty_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="job_id must not be empty"):
            WatchRequest(job_id="")

    def test_whitespace_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="job_id must not be empty"):
            WatchRequest(job_id="   ")


class TestUnwatchRequest:
    """Tests for the immutable UnwatchRequest dataclass."""

    def test_create(self) -> None:
        req = UnwatchRequest(job_id="job-1", subscriber_id="sub-abc")
        assert req.job_id == "job-1"
        assert req.subscriber_id == "sub-abc"

    def test_frozen(self) -> None:
        req = UnwatchRequest(job_id="job-1", subscriber_id="sub-abc")
        with pytest.raises(AttributeError):
            req.job_id = "mutated"  # type: ignore[misc]

    def test_empty_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="job_id must not be empty"):
            UnwatchRequest(job_id="", subscriber_id="sub-abc")

    def test_empty_subscriber_id_raises(self) -> None:
        with pytest.raises(ValueError, match="subscriber_id must not be empty"):
            UnwatchRequest(job_id="job-1", subscriber_id="")

    def test_whitespace_subscriber_id_raises(self) -> None:
        with pytest.raises(ValueError, match="subscriber_id must not be empty"):
            UnwatchRequest(job_id="job-1", subscriber_id="   ")


# ---------------------------------------------------------------------------
# WatchResponse
# ---------------------------------------------------------------------------


class TestWatchResponse:
    """Tests for the immutable WatchResponse dataclass."""

    def test_create(self) -> None:
        resp = WatchResponse(
            job_id="job-1",
            subscriber_id="sub-abc",
            buffered_lines=50,
        )
        assert resp.job_id == "job-1"
        assert resp.subscriber_id == "sub-abc"
        assert resp.buffered_lines == 50

    def test_frozen(self) -> None:
        resp = WatchResponse(
            job_id="job-1",
            subscriber_id="sub-abc",
            buffered_lines=0,
        )
        with pytest.raises(AttributeError):
            resp.job_id = "mutated"  # type: ignore[misc]

    def test_negative_buffered_lines_raises(self) -> None:
        with pytest.raises(ValueError, match="buffered_lines must not be negative"):
            WatchResponse(
                job_id="job-1",
                subscriber_id="sub-abc",
                buffered_lines=-1,
            )

    def test_to_payload(self) -> None:
        resp = WatchResponse(
            job_id="job-1",
            subscriber_id="sub-abc",
            buffered_lines=10,
        )
        payload = resp.to_payload()
        assert payload == {
            "verb": "watch",
            "status": "subscribed",
            "job_id": "job-1",
            "subscriber_id": "sub-abc",
            "buffered_lines": 10,
        }


class TestUnwatchResponse:
    """Tests for the immutable UnwatchResponse dataclass."""

    def test_create(self) -> None:
        resp = UnwatchResponse(job_id="job-1", subscriber_id="sub-abc")
        assert resp.job_id == "job-1"
        assert resp.subscriber_id == "sub-abc"

    def test_frozen(self) -> None:
        resp = UnwatchResponse(job_id="job-1", subscriber_id="sub-abc")
        with pytest.raises(AttributeError):
            resp.subscriber_id = "mutated"  # type: ignore[misc]

    def test_to_payload(self) -> None:
        resp = UnwatchResponse(job_id="job-1", subscriber_id="sub-abc")
        payload = resp.to_payload()
        assert payload == {
            "verb": "unwatch",
            "status": "unsubscribed",
            "job_id": "job-1",
            "subscriber_id": "sub-abc",
        }


# ---------------------------------------------------------------------------
# parse_watch_request / parse_unwatch_request
# ---------------------------------------------------------------------------


class TestParseWatchRequest:
    """Tests for parsing watch requests from envelope payloads."""

    def test_parse_valid_payload(self) -> None:
        payload = {"verb": "watch", "job_id": "job-123"}
        req = parse_watch_request(payload)
        assert req.job_id == "job-123"

    def test_missing_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="job_id"):
            parse_watch_request({"verb": "watch"})

    def test_empty_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="job_id must not be empty"):
            parse_watch_request({"verb": "watch", "job_id": ""})


class TestParseUnwatchRequest:
    """Tests for parsing unwatch requests from envelope payloads."""

    def test_parse_valid_payload(self) -> None:
        payload = {
            "verb": "unwatch",
            "job_id": "job-123",
            "subscriber_id": "sub-abc",
        }
        req = parse_unwatch_request(payload)
        assert req.job_id == "job-123"
        assert req.subscriber_id == "sub-abc"

    def test_missing_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="job_id"):
            parse_unwatch_request({"verb": "unwatch", "subscriber_id": "s"})

    def test_missing_subscriber_id_raises(self) -> None:
        with pytest.raises(ValueError, match="subscriber_id"):
            parse_unwatch_request({"verb": "unwatch", "job_id": "j"})


# ---------------------------------------------------------------------------
# WatchSubscriptionManager -- handle_watch
# ---------------------------------------------------------------------------


class TestHandleWatch:
    """Tests for the watch request handler."""

    @pytest.mark.asyncio
    async def test_watch_subscribes_to_registered_job(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-abc")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()
        envelope = _make_watch_envelope(job_id="job-abc")

        response = await manager.handle_watch(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.msg_id == envelope.msg_id
        assert response.payload["status"] == "subscribed"
        assert response.payload["job_id"] == "job-abc"
        assert "subscriber_id" in response.payload
        assert broadcaster.subscriber_count("job-abc") == 1

    @pytest.mark.asyncio
    async def test_watch_returns_buffered_line_count(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-abc")
        broadcaster.publish("job-abc", "line 1")
        broadcaster.publish("job-abc", "line 2")
        broadcaster.publish("job-abc", "line 3")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()
        envelope = _make_watch_envelope(job_id="job-abc")

        response = await manager.handle_watch(envelope, client)

        assert response.payload["buffered_lines"] == 3

    @pytest.mark.asyncio
    async def test_watch_unregistered_job_returns_error(self) -> None:
        broadcaster = JobOutputBroadcaster()
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()
        envelope = _make_watch_envelope(job_id="nonexistent")

        response = await manager.handle_watch(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert "not registered" in response.payload.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_watch_invalid_payload_returns_error(self) -> None:
        broadcaster = JobOutputBroadcaster()
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()
        envelope = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="req-bad",
            timestamp="2026-04-09T12:00:00Z",
            payload={"verb": "watch"},  # missing job_id
        )

        response = await manager.handle_watch(envelope, client)

        assert response.msg_type == MessageType.ERROR

    @pytest.mark.asyncio
    async def test_watch_same_job_twice_replaces_subscription(self) -> None:
        """Watching the same job from the same client replaces the prior
        subscription rather than creating a duplicate."""
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-abc")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()

        resp1 = await manager.handle_watch(
            _make_watch_envelope(job_id="job-abc", msg_id="r1"), client
        )
        resp2 = await manager.handle_watch(
            _make_watch_envelope(job_id="job-abc", msg_id="r2"), client
        )

        # The second watch should have replaced the first
        sub_id_1 = resp1.payload["subscriber_id"]
        sub_id_2 = resp2.payload["subscriber_id"]
        assert sub_id_1 != sub_id_2
        # Only one active subscription on the broadcaster
        assert broadcaster.subscriber_count("job-abc") == 1

    @pytest.mark.asyncio
    async def test_watch_multiple_jobs_from_same_client(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-a")
        broadcaster.register_job("job-b")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()

        await manager.handle_watch(
            _make_watch_envelope(job_id="job-a", msg_id="r1"), client
        )
        await manager.handle_watch(
            _make_watch_envelope(job_id="job-b", msg_id="r2"), client
        )

        assert broadcaster.subscriber_count("job-a") == 1
        assert broadcaster.subscriber_count("job-b") == 1

    @pytest.mark.asyncio
    async def test_watch_tracks_subscription_per_client(self) -> None:
        """Different clients watching the same job produce separate
        subscriptions."""
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-abc")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client_a = _make_client("client-a")
        client_b = _make_client("client-b")

        await manager.handle_watch(
            _make_watch_envelope(job_id="job-abc", msg_id="r1"), client_a
        )
        await manager.handle_watch(
            _make_watch_envelope(job_id="job-abc", msg_id="r2"), client_b
        )

        assert broadcaster.subscriber_count("job-abc") == 2


# ---------------------------------------------------------------------------
# WatchSubscriptionManager -- handle_unwatch
# ---------------------------------------------------------------------------


class TestHandleUnwatch:
    """Tests for the unwatch request handler."""

    @pytest.mark.asyncio
    async def test_unwatch_removes_subscription(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-abc")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()

        # First watch
        watch_resp = await manager.handle_watch(
            _make_watch_envelope(job_id="job-abc"), client
        )
        subscriber_id = watch_resp.payload["subscriber_id"]
        assert broadcaster.subscriber_count("job-abc") == 1

        # Then unwatch
        unwatch_envelope = _make_unwatch_envelope(
            job_id="job-abc",
            subscriber_id=subscriber_id,
        )
        response = await manager.handle_unwatch(unwatch_envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.msg_id == unwatch_envelope.msg_id
        assert response.payload["status"] == "unsubscribed"
        assert response.payload["job_id"] == "job-abc"
        assert response.payload["subscriber_id"] == subscriber_id
        assert broadcaster.subscriber_count("job-abc") == 0

    @pytest.mark.asyncio
    async def test_unwatch_unknown_subscriber_returns_error(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-abc")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()

        envelope = _make_unwatch_envelope(
            job_id="job-abc",
            subscriber_id="sub-nonexistent",
        )
        response = await manager.handle_unwatch(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert "not found" in response.payload.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_unwatch_mismatched_subscriber_id_returns_error(self) -> None:
        """Client has a subscription but sends the wrong subscriber_id."""
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-abc")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()

        await manager.handle_watch(
            _make_watch_envelope(job_id="job-abc"), client
        )

        envelope = _make_unwatch_envelope(
            job_id="job-abc",
            subscriber_id="sub-wrong-id",
        )
        response = await manager.handle_unwatch(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert "not found" in response.payload.get("error", "").lower()
        # Subscription should still be active
        assert broadcaster.subscriber_count("job-abc") == 1

    @pytest.mark.asyncio
    async def test_unwatch_wrong_client_returns_error(self) -> None:
        """A client cannot unwatch a subscription owned by another client."""
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-abc")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client_a = _make_client("client-a")
        client_b = _make_client("client-b")

        watch_resp = await manager.handle_watch(
            _make_watch_envelope(job_id="job-abc"), client_a
        )
        subscriber_id = watch_resp.payload["subscriber_id"]

        # client_b tries to unwatch client_a's subscription
        envelope = _make_unwatch_envelope(
            job_id="job-abc",
            subscriber_id=subscriber_id,
        )
        response = await manager.handle_unwatch(envelope, client_b)

        assert response.msg_type == MessageType.ERROR
        assert "not found" in response.payload.get("error", "").lower()
        # Subscription should still be active
        assert broadcaster.subscriber_count("job-abc") == 1

    @pytest.mark.asyncio
    async def test_unwatch_invalid_payload_returns_error(self) -> None:
        broadcaster = JobOutputBroadcaster()
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()
        envelope = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="req-bad",
            timestamp="2026-04-09T12:00:00Z",
            payload={"verb": "unwatch", "job_id": "j"},  # missing subscriber_id
        )

        response = await manager.handle_unwatch(envelope, client)

        assert response.msg_type == MessageType.ERROR

    @pytest.mark.asyncio
    async def test_unwatch_idempotent_after_first_removal(self) -> None:
        """Unwatching an already-unwatched subscription returns error
        (not a crash)."""
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-abc")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()

        watch_resp = await manager.handle_watch(
            _make_watch_envelope(job_id="job-abc"), client
        )
        subscriber_id = watch_resp.payload["subscriber_id"]

        envelope = _make_unwatch_envelope(
            job_id="job-abc",
            subscriber_id=subscriber_id,
        )
        # First unwatch succeeds
        resp1 = await manager.handle_unwatch(envelope, client)
        assert resp1.msg_type == MessageType.RESPONSE

        # Second unwatch returns error (already removed)
        resp2 = await manager.handle_unwatch(envelope, client)
        assert resp2.msg_type == MessageType.ERROR


# ---------------------------------------------------------------------------
# WatchSubscriptionManager -- client disconnect cleanup
# ---------------------------------------------------------------------------


class TestClientDisconnectCleanup:
    """Tests for subscription cleanup when a client disconnects."""

    @pytest.mark.asyncio
    async def test_remove_client_subscriptions_cleans_all(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-a")
        broadcaster.register_job("job-b")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()

        await manager.handle_watch(
            _make_watch_envelope(job_id="job-a", msg_id="r1"), client
        )
        await manager.handle_watch(
            _make_watch_envelope(job_id="job-b", msg_id="r2"), client
        )
        assert broadcaster.subscriber_count("job-a") == 1
        assert broadcaster.subscriber_count("job-b") == 1

        removed = manager.remove_client_subscriptions(client.client_id)

        assert len(removed) == 2
        assert broadcaster.subscriber_count("job-a") == 0
        assert broadcaster.subscriber_count("job-b") == 0

    @pytest.mark.asyncio
    async def test_remove_unknown_client_returns_empty(self) -> None:
        broadcaster = JobOutputBroadcaster()
        manager = WatchSubscriptionManager(broadcaster=broadcaster)

        removed = manager.remove_client_subscriptions("nonexistent-client")
        assert removed == ()

    @pytest.mark.asyncio
    async def test_remove_client_does_not_affect_others(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-abc")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client_a = _make_client("client-a")
        client_b = _make_client("client-b")

        await manager.handle_watch(
            _make_watch_envelope(job_id="job-abc", msg_id="r1"), client_a
        )
        await manager.handle_watch(
            _make_watch_envelope(job_id="job-abc", msg_id="r2"), client_b
        )
        assert broadcaster.subscriber_count("job-abc") == 2

        manager.remove_client_subscriptions("client-a")

        assert broadcaster.subscriber_count("job-abc") == 1


# ---------------------------------------------------------------------------
# WatchSubscriptionManager -- introspection
# ---------------------------------------------------------------------------


class TestSubscriptionIntrospection:
    """Tests for querying active subscriptions."""

    @pytest.mark.asyncio
    async def test_get_client_subscriptions(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-a")
        broadcaster.register_job("job-b")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()

        await manager.handle_watch(
            _make_watch_envelope(job_id="job-a", msg_id="r1"), client
        )
        await manager.handle_watch(
            _make_watch_envelope(job_id="job-b", msg_id="r2"), client
        )

        subs = manager.get_client_subscriptions(client.client_id)
        assert len(subs) == 2
        job_ids = {s.job_id for s in subs}
        assert job_ids == {"job-a", "job-b"}

    @pytest.mark.asyncio
    async def test_get_client_subscriptions_empty(self) -> None:
        broadcaster = JobOutputBroadcaster()
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        subs = manager.get_client_subscriptions("nonexistent")
        assert subs == ()

    @pytest.mark.asyncio
    async def test_has_subscription(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-abc")
        manager = WatchSubscriptionManager(broadcaster=broadcaster)
        client = _make_client()

        assert not manager.has_subscription(client.client_id, "job-abc")

        await manager.handle_watch(
            _make_watch_envelope(job_id="job-abc"), client
        )

        assert manager.has_subscription(client.client_id, "job-abc")
