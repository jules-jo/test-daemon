"""Tests for the IPC request handler.

Validates that the RequestHandler (concrete ClientHandler implementation)
correctly bridges the socket server to the validation + dispatch pipeline:
- Validates incoming requests via the validation layer
- Returns validation error envelopes for invalid requests
- Dispatches valid requests to registered handlers
- Returns response envelopes with handler results
- Returns enqueue confirmation for queue verb
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from jules_daemon.ipc.framing import MessageEnvelope, MessageType
from jules_daemon.ipc.request_handler import (
    RequestHandler,
    RequestHandlerConfig,
)
from jules_daemon.ipc.server import ClientConnection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> ClientConnection:
    """Build a stub ClientConnection for testing."""
    return ClientConnection(
        client_id="test-client-001",
        reader=AsyncMock(spec=asyncio.StreamReader),
        writer=AsyncMock(spec=asyncio.StreamWriter),
        connected_at="2026-04-09T12:00:00Z",
    )


def _make_request(
    payload: dict[str, Any],
    msg_id: str = "req-001",
) -> MessageEnvelope:
    """Build a REQUEST-type envelope."""
    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=msg_id,
        timestamp="2026-04-09T12:00:00Z",
        payload=payload,
    )


def _make_non_request(
    payload: dict[str, Any],
    msg_type: MessageType = MessageType.RESPONSE,
    msg_id: str = "req-001",
) -> MessageEnvelope:
    """Build a non-REQUEST envelope."""
    return MessageEnvelope(
        msg_type=msg_type,
        msg_id=msg_id,
        timestamp="2026-04-09T12:00:00Z",
        payload=payload,
    )


# ---------------------------------------------------------------------------
# RequestHandlerConfig tests
# ---------------------------------------------------------------------------


class TestRequestHandlerConfig:
    """Tests for the immutable handler configuration."""

    def test_create_with_wiki_root(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        assert config.wiki_root == tmp_path

    def test_frozen(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        with pytest.raises(AttributeError):
            config.wiki_root = Path("/other")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RequestHandler: protocol conformance
# ---------------------------------------------------------------------------


class TestRequestHandlerProtocol:
    """Tests that RequestHandler implements the ClientHandler protocol."""

    def test_has_handle_message_method(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        assert hasattr(handler, "handle_message")
        assert callable(handler.handle_message)

    def test_implements_client_handler(self, tmp_path: Path) -> None:
        from jules_daemon.ipc.server import ClientHandler

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        assert isinstance(handler, ClientHandler)


# ---------------------------------------------------------------------------
# RequestHandler: invalid message type
# ---------------------------------------------------------------------------


class TestRequestHandlerInvalidType:
    """Tests for non-REQUEST message type handling."""

    @pytest.mark.asyncio
    async def test_non_request_returns_error(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_non_request(payload={"verb": "status"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert response.msg_id == envelope.msg_id
        assert "error" in response.payload
        assert "validation_errors" in response.payload

    @pytest.mark.asyncio
    async def test_stream_type_returns_error(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_non_request(
            payload={"verb": "status"},
            msg_type=MessageType.STREAM,
        )

        response = await handler.handle_message(envelope, client)
        assert response.msg_type == MessageType.ERROR


# ---------------------------------------------------------------------------
# RequestHandler: missing / invalid verb
# ---------------------------------------------------------------------------


class TestRequestHandlerVerbErrors:
    """Tests for verb-level validation errors."""

    @pytest.mark.asyncio
    async def test_missing_verb_returns_error(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert "validation_errors" in response.payload
        errors = response.payload["validation_errors"]
        assert any(e["field"] == "verb" for e in errors)

    @pytest.mark.asyncio
    async def test_unknown_verb_returns_error(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "teleport"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        errors = response.payload["validation_errors"]
        assert any(e["code"] == "unknown_verb" for e in errors)


# ---------------------------------------------------------------------------
# RequestHandler: field-level validation errors
# ---------------------------------------------------------------------------


class TestRequestHandlerFieldErrors:
    """Tests for verb-specific field validation errors."""

    @pytest.mark.asyncio
    async def test_run_missing_fields_returns_all_errors(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "run"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        errors = response.payload["validation_errors"]
        fields = {e["field"] for e in errors}
        assert "target_host" in fields
        assert "target_user" in fields
        assert "natural_language" in fields

    @pytest.mark.asyncio
    async def test_queue_missing_fields_returns_errors(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "queue"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        errors = response.payload["validation_errors"]
        fields = {e["field"] for e in errors}
        assert "target_host" in fields


# ---------------------------------------------------------------------------
# RequestHandler: valid status request
# ---------------------------------------------------------------------------


class TestRequestHandlerStatusVerb:
    """Tests for valid status verb handling."""

    @pytest.mark.asyncio
    async def test_status_returns_response(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "status"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.msg_id == envelope.msg_id
        assert "verb" in response.payload
        assert response.payload["verb"] == "status"

    @pytest.mark.asyncio
    async def test_status_verbose(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={
            "verb": "status",
            "verbose": True,
        })

        response = await handler.handle_message(envelope, client)
        assert response.msg_type == MessageType.RESPONSE


# ---------------------------------------------------------------------------
# RequestHandler: valid queue request with enqueue confirmation
# ---------------------------------------------------------------------------


class TestRequestHandlerQueueVerb:
    """Tests for queue verb with enqueue confirmation."""

    @pytest.mark.asyncio
    async def test_queue_returns_enqueue_confirmation(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={
            "verb": "queue",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run the smoke tests",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.msg_id == envelope.msg_id
        assert response.payload["verb"] == "queue"
        assert response.payload["status"] == "enqueued"
        assert "queue_id" in response.payload
        assert "position" in response.payload

    @pytest.mark.asyncio
    async def test_queue_creates_wiki_file(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={
            "verb": "queue",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run integration tests",
        })

        response = await handler.handle_message(envelope, client)

        # Verify the wiki queue directory was created
        queue_dir = tmp_path / "pages" / "daemon" / "queue"
        assert queue_dir.exists()
        queue_files = list(queue_dir.glob("*.md"))
        assert len(queue_files) == 1

    @pytest.mark.asyncio
    async def test_multiple_queues_increment_position(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        for i in range(3):
            envelope = _make_request(
                payload={
                    "verb": "queue",
                    "target_host": f"host-{i}.example.com",
                    "target_user": "deploy",
                    "natural_language": f"run test suite {i}",
                },
                msg_id=f"req-{i}",
            )
            response = await handler.handle_message(envelope, client)
            assert response.payload["status"] == "enqueued"
            assert response.payload["position"] == i + 1


# ---------------------------------------------------------------------------
# RequestHandler: valid cancel request
# ---------------------------------------------------------------------------


class TestRequestHandlerCancelVerb:
    """Tests for cancel verb handling."""

    @pytest.mark.asyncio
    async def test_cancel_returns_response(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "cancel"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "cancel"


# ---------------------------------------------------------------------------
# RequestHandler: valid history request
# ---------------------------------------------------------------------------


class TestRequestHandlerHistoryVerb:
    """Tests for history verb handling."""

    @pytest.mark.asyncio
    async def test_history_returns_response(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "history"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "history"


# ---------------------------------------------------------------------------
# RequestHandler: valid watch request
# ---------------------------------------------------------------------------


class TestRequestHandlerWatchVerb:
    """Tests for watch verb handling."""

    @pytest.mark.asyncio
    async def test_watch_returns_response(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "watch"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "watch"


# ---------------------------------------------------------------------------
# RequestHandler: valid run request
# ---------------------------------------------------------------------------


class TestRequestHandlerRunVerb:
    """Tests for run verb handling."""

    @pytest.mark.asyncio
    async def test_run_returns_response(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={
            "verb": "run",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run the full regression suite",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "run"
        assert response.payload["status"] == "accepted"


# ---------------------------------------------------------------------------
# RequestHandler: response correlation
# ---------------------------------------------------------------------------


class TestRequestHandlerCorrelation:
    """Tests for request/response ID correlation."""

    @pytest.mark.asyncio
    async def test_response_preserves_msg_id(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(
            payload={"verb": "status"},
            msg_id="unique-correlation-id",
        )

        response = await handler.handle_message(envelope, client)
        assert response.msg_id == "unique-correlation-id"

    @pytest.mark.asyncio
    async def test_error_response_preserves_msg_id(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(
            payload={"verb": "teleport"},
            msg_id="err-correlation-id",
        )

        response = await handler.handle_message(envelope, client)
        assert response.msg_id == "err-correlation-id"


# ---------------------------------------------------------------------------
# RequestHandler: response envelope structure
# ---------------------------------------------------------------------------


class TestRequestHandlerResponseStructure:
    """Tests for response envelope structure consistency."""

    @pytest.mark.asyncio
    async def test_error_has_validation_errors_list(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert "error" in response.payload
        assert "validation_errors" in response.payload
        assert isinstance(response.payload["validation_errors"], list)
        for err in response.payload["validation_errors"]:
            assert "field" in err
            assert "message" in err
            assert "code" in err

    @pytest.mark.asyncio
    async def test_success_has_verb_and_timestamp(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "status"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert "verb" in response.payload
        assert response.timestamp  # non-empty
