"""IPC watch/unwatch subscription handler for job output streaming.

Bridges the IPC request-response protocol with the JobOutputBroadcaster's
subscribe/unsubscribe API. When a CLI client sends a ``watch`` request,
this handler registers a subscription on the broadcaster for the given
job ID. An ``unwatch`` request removes that subscription.

Key responsibilities:

- **Typed payloads**: WatchRequest, UnwatchRequest, WatchResponse, and
  UnwatchResponse are frozen dataclasses that define the contract between
  the CLI and daemon for watch-related IPC messages.

- **Subscription lifecycle**: The WatchSubscriptionManager tracks active
  subscriptions per (client_id, job_id) so it can:
  - Replace an existing subscription when the same client re-watches
    the same job (no duplicate subscribers on the broadcaster).
  - Clean up all subscriptions when a client disconnects.
  - Prevent cross-client unwatch (a client cannot remove another
    client's subscription).

- **Error isolation**: Invalid payloads and broadcaster errors are
  caught and returned as ERROR envelopes -- never crash the server.

Architecture::

    CLI                    IPC Server                WatchSubscriptionManager
     |                        |                              |
     |-- watch {job_id} ----->|-- handle_watch() ----------->|
     |                        |     parse payload            |
     |                        |     broadcaster.subscribe()  |
     |                        |     track (client, job)      |
     |<-- RESPONSE {sub_id} --|<-- WatchResponse ------------|
     |                        |                              |
     |-- unwatch {sub_id} --->|-- handle_unwatch() --------->|
     |                        |     validate ownership       |
     |                        |     broadcaster.unsubscribe()|
     |<-- RESPONSE ---------- |<-- UnwatchResponse ----------|

Usage::

    from jules_daemon.ipc.watch_handler import WatchSubscriptionManager
    from jules_daemon.monitor.output_broadcaster import JobOutputBroadcaster

    broadcaster = JobOutputBroadcaster()
    manager = WatchSubscriptionManager(broadcaster=broadcaster)

    # In the server's message handler:
    if verb == "watch":
        response = await manager.handle_watch(envelope, client)
    elif verb == "unwatch":
        response = await manager.handle_unwatch(envelope, client)

    # On client disconnect:
    manager.remove_client_subscriptions(client.client_id)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from jules_daemon.ipc.framing import MessageEnvelope, MessageType
from jules_daemon.ipc.server import ClientConnection
from jules_daemon.monitor.output_broadcaster import (
    JobOutputBroadcaster,
    SubscriberHandle,
)

__all__ = [
    "UnwatchRequest",
    "UnwatchResponse",
    "WatchRequest",
    "WatchResponse",
    "WatchSubscriptionManager",
    "parse_unwatch_request",
    "parse_watch_request",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERB_WATCH = "watch"
_VERB_UNWATCH = "unwatch"
_STATUS_SUBSCRIBED = "subscribed"
_STATUS_UNSUBSCRIBED = "unsubscribed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _build_error_envelope(
    msg_id: str,
    error_message: str,
) -> MessageEnvelope:
    """Build an ERROR envelope for watch/unwatch errors.

    Args:
        msg_id: The original request's msg_id for correlation.
        error_message: Human-readable error description.

    Returns:
        A MessageEnvelope with MessageType.ERROR.
    """
    return MessageEnvelope(
        msg_type=MessageType.ERROR,
        msg_id=msg_id,
        timestamp=_now_iso(),
        payload={"error": error_message},
    )


def _build_response_envelope(
    msg_id: str,
    payload: dict[str, Any],
) -> MessageEnvelope:
    """Build a RESPONSE envelope with the given payload.

    Args:
        msg_id: The original request's msg_id for correlation.
        payload: The response payload dict.

    Returns:
        A MessageEnvelope with MessageType.RESPONSE.
    """
    return MessageEnvelope(
        msg_type=MessageType.RESPONSE,
        msg_id=msg_id,
        timestamp=_now_iso(),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Typed request dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchRequest:
    """Typed payload for a watch subscription request.

    Attributes:
        job_id: The job ID to subscribe to for output streaming.
    """

    job_id: str

    def __post_init__(self) -> None:
        if not self.job_id or not self.job_id.strip():
            raise ValueError("job_id must not be empty")


@dataclass(frozen=True)
class UnwatchRequest:
    """Typed payload for an unwatch (unsubscribe) request.

    Attributes:
        job_id: The job ID to unsubscribe from.
        subscriber_id: The subscriber handle ID to remove.
    """

    job_id: str
    subscriber_id: str

    def __post_init__(self) -> None:
        if not self.job_id or not self.job_id.strip():
            raise ValueError("job_id must not be empty")
        if not self.subscriber_id or not self.subscriber_id.strip():
            raise ValueError("subscriber_id must not be empty")


# ---------------------------------------------------------------------------
# Typed response dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchResponse:
    """Typed payload for a successful watch subscription response.

    Attributes:
        job_id: The job ID that was subscribed to.
        subscriber_id: The subscriber handle ID for the new subscription.
        buffered_lines: Number of lines currently in the job's ring buffer
            (available for replay).
    """

    job_id: str
    subscriber_id: str
    buffered_lines: int

    def __post_init__(self) -> None:
        if self.buffered_lines < 0:
            raise ValueError("buffered_lines must not be negative")

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a dict suitable for MessageEnvelope payload.

        Returns:
            Dict with verb, status, job_id, subscriber_id, buffered_lines.
        """
        return {
            "verb": _VERB_WATCH,
            "status": _STATUS_SUBSCRIBED,
            "job_id": self.job_id,
            "subscriber_id": self.subscriber_id,
            "buffered_lines": self.buffered_lines,
        }


@dataclass(frozen=True)
class UnwatchResponse:
    """Typed payload for a successful unwatch response.

    Attributes:
        job_id: The job ID that was unsubscribed from.
        subscriber_id: The subscriber handle ID that was removed.
    """

    job_id: str
    subscriber_id: str

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a dict suitable for MessageEnvelope payload.

        Returns:
            Dict with verb, status, job_id, subscriber_id.
        """
        return {
            "verb": _VERB_UNWATCH,
            "status": _STATUS_UNSUBSCRIBED,
            "job_id": self.job_id,
            "subscriber_id": self.subscriber_id,
        }


# ---------------------------------------------------------------------------
# Payload parsers
# ---------------------------------------------------------------------------


def parse_watch_request(payload: dict[str, Any]) -> WatchRequest:
    """Parse a watch request from a raw envelope payload dict.

    Extracts and validates the ``job_id`` field from the payload.

    Args:
        payload: The ``MessageEnvelope.payload`` dict.

    Returns:
        A validated WatchRequest.

    Raises:
        ValueError: If ``job_id`` is missing or empty.
    """
    job_id = payload.get("job_id")
    if job_id is None:
        raise ValueError("Missing required field: job_id")
    return WatchRequest(job_id=job_id)


def parse_unwatch_request(payload: dict[str, Any]) -> UnwatchRequest:
    """Parse an unwatch request from a raw envelope payload dict.

    Extracts and validates both ``job_id`` and ``subscriber_id`` fields.

    Args:
        payload: The ``MessageEnvelope.payload`` dict.

    Returns:
        A validated UnwatchRequest.

    Raises:
        ValueError: If ``job_id`` or ``subscriber_id`` is missing or empty.
    """
    job_id = payload.get("job_id")
    if job_id is None:
        raise ValueError("Missing required field: job_id")
    subscriber_id = payload.get("subscriber_id")
    if subscriber_id is None:
        raise ValueError("Missing required field: subscriber_id")
    return UnwatchRequest(job_id=job_id, subscriber_id=subscriber_id)


# ---------------------------------------------------------------------------
# WatchSubscriptionManager
# ---------------------------------------------------------------------------


class WatchSubscriptionManager:
    """Manages watch/unwatch subscriptions for the IPC server.

    Bridges the IPC layer and the JobOutputBroadcaster. Tracks active
    subscriptions per (client_id, job_id) pair and handles cleanup when
    clients disconnect.

    Thread safety: designed for single-threaded async use within one
    event loop. Does not use locks (matches JobOutputBroadcaster design).

    Args:
        broadcaster: The JobOutputBroadcaster to manage subscriptions on.
    """

    def __init__(self, *, broadcaster: JobOutputBroadcaster) -> None:
        self._broadcaster = broadcaster
        # Nested mapping: client_id -> {job_id -> SubscriberHandle}
        self._subscriptions: dict[str, dict[str, SubscriberHandle]] = {}

    # -- Watch handler -------------------------------------------------------

    async def handle_watch(
        self,
        envelope: MessageEnvelope,
        client: ClientConnection,
    ) -> MessageEnvelope:
        """Process a watch request and return a response envelope.

        Parses the watch payload, subscribes the client to the
        broadcaster for the given job ID, and returns a response
        with the subscriber ID and buffered line count.

        If the client already has a subscription for the same job,
        the old subscription is replaced (unsubscribed then re-subscribed).

        Args:
            envelope: The incoming request envelope.
            client: The client connection that sent the request.

        Returns:
            A RESPONSE envelope on success, or an ERROR envelope on failure.
        """
        try:
            request = parse_watch_request(envelope.payload)
        except ValueError as exc:
            logger.warning(
                "Invalid watch payload from %s: %s",
                client.client_id,
                exc,
            )
            return _build_error_envelope(envelope.msg_id, str(exc))

        # Unsubscribe existing subscription for this (client, job) pair
        self._remove_single_subscription(client.client_id, request.job_id)

        # Subscribe on the broadcaster
        try:
            handle = self._broadcaster.subscribe(request.job_id)
        except ValueError as exc:
            logger.warning(
                "Watch failed for client %s, job %s: %s",
                client.client_id,
                request.job_id,
                exc,
            )
            return _build_error_envelope(
                envelope.msg_id,
                f"Job {request.job_id!r} is not registered",
            )

        # Track the subscription
        client_subs = self._subscriptions.setdefault(client.client_id, {})
        client_subs[request.job_id] = handle

        # Build response with buffer info
        buffer = self._broadcaster.get_buffer(request.job_id)
        response = WatchResponse(
            job_id=request.job_id,
            subscriber_id=handle.subscriber_id,
            buffered_lines=len(buffer),
        )

        logger.info(
            "Client %s subscribed to job %s (subscriber=%s, buffered=%d)",
            client.client_id,
            request.job_id,
            handle.subscriber_id,
            len(buffer),
        )

        return _build_response_envelope(envelope.msg_id, response.to_payload())

    # -- Unwatch handler -----------------------------------------------------

    async def handle_unwatch(
        self,
        envelope: MessageEnvelope,
        client: ClientConnection,
    ) -> MessageEnvelope:
        """Process an unwatch request and return a response envelope.

        Validates that the subscription belongs to the requesting client,
        unsubscribes from the broadcaster, and removes the tracking entry.

        Args:
            envelope: The incoming request envelope.
            client: The client connection that sent the request.

        Returns:
            A RESPONSE envelope on success, or an ERROR envelope on failure.
        """
        try:
            request = parse_unwatch_request(envelope.payload)
        except ValueError as exc:
            logger.warning(
                "Invalid unwatch payload from %s: %s",
                client.client_id,
                exc,
            )
            return _build_error_envelope(envelope.msg_id, str(exc))

        # Verify the subscription belongs to this client
        client_subs = self._subscriptions.get(client.client_id, {})
        tracked_handle = client_subs.get(request.job_id)

        if tracked_handle is None:
            return _build_error_envelope(
                envelope.msg_id,
                f"Subscription not found for client {client.client_id!r} "
                f"on job {request.job_id!r}",
            )

        if tracked_handle.subscriber_id != request.subscriber_id:
            return _build_error_envelope(
                envelope.msg_id,
                f"Subscriber {request.subscriber_id!r} not found for "
                f"client {client.client_id!r} on job {request.job_id!r}",
            )

        # Unsubscribe from the broadcaster
        self._broadcaster.unsubscribe(tracked_handle)

        # Remove from tracking
        del client_subs[request.job_id]
        if not client_subs:
            del self._subscriptions[client.client_id]

        response = UnwatchResponse(
            job_id=request.job_id,
            subscriber_id=request.subscriber_id,
        )

        logger.info(
            "Client %s unsubscribed from job %s (subscriber=%s)",
            client.client_id,
            request.job_id,
            request.subscriber_id,
        )

        return _build_response_envelope(envelope.msg_id, response.to_payload())

    # -- Client disconnect cleanup -------------------------------------------

    def remove_client_subscriptions(
        self,
        client_id: str,
    ) -> tuple[SubscriberHandle, ...]:
        """Remove all subscriptions for a disconnecting client.

        Unsubscribes all active handles for the given client from the
        broadcaster and removes them from the tracking registry.

        This should be called when a client disconnects to prevent
        orphaned subscriber queues on the broadcaster.

        Args:
            client_id: The ID of the disconnecting client.

        Returns:
            Tuple of SubscriberHandle objects that were removed.
        """
        client_subs = self._subscriptions.pop(client_id, None)
        if client_subs is None:
            return ()

        removed_handles: list[SubscriberHandle] = []
        for job_id, handle in client_subs.items():
            self._broadcaster.unsubscribe(handle)
            removed_handles.append(handle)
            logger.debug(
                "Cleaned up subscription for disconnected client %s "
                "on job %s (subscriber=%s)",
                client_id,
                job_id,
                handle.subscriber_id,
            )

        logger.info(
            "Removed %d subscription(s) for disconnected client %s",
            len(removed_handles),
            client_id,
        )

        return tuple(removed_handles)

    # -- Introspection -------------------------------------------------------

    def get_client_subscriptions(
        self,
        client_id: str,
    ) -> tuple[SubscriberHandle, ...]:
        """Return all active subscriptions for a client.

        Args:
            client_id: The client ID to query.

        Returns:
            Tuple of SubscriberHandle objects for the client.
            Empty tuple if no subscriptions exist.
        """
        client_subs = self._subscriptions.get(client_id, {})
        return tuple(client_subs.values())

    def has_subscription(
        self,
        client_id: str,
        job_id: str,
    ) -> bool:
        """Check whether a client has an active subscription for a job.

        Args:
            client_id: The client ID to check.
            job_id: The job ID to check.

        Returns:
            True if the client has an active subscription for the job.
        """
        client_subs = self._subscriptions.get(client_id, {})
        return job_id in client_subs

    # -- Internal helpers ----------------------------------------------------

    def _remove_single_subscription(
        self,
        client_id: str,
        job_id: str,
    ) -> None:
        """Remove a single subscription for a (client, job) pair.

        Used internally to clean up before re-subscribing (replace pattern).
        Does nothing if no subscription exists.
        """
        client_subs = self._subscriptions.get(client_id)
        if client_subs is None:
            return

        handle = client_subs.pop(job_id, None)
        if handle is not None:
            self._broadcaster.unsubscribe(handle)
            logger.debug(
                "Replaced subscription for client %s on job %s",
                client_id,
                job_id,
            )

        if not client_subs:
            del self._subscriptions[client_id]
