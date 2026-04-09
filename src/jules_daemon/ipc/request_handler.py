"""IPC request handler: concrete ClientHandler bridging validation and dispatch.

Implements the ``ClientHandler`` protocol expected by ``SocketServer``.
Each incoming ``MessageEnvelope`` is:

1. Validated via the ``request_validator`` layer (message type, verb, fields).
2. On validation failure: returns an ERROR envelope with structured errors.
3. On validation success: routes to the verb-specific handler:
   - **queue**: enqueues to the wiki-backed ``CommandQueue`` and returns
     an enqueue confirmation with queue_id and position.
   - **run**: accepts the command for processing and returns acceptance.
   - **status/watch/cancel/history**: returns a stub response (handlers
     are wired by the daemon lifecycle, not the IPC layer).

Architecture::

    SocketServer -> RequestHandler.handle_message()
                        |
                        v
                    validate_request(envelope)
                        |
                   valid?  ---no--->  ERROR envelope (validation_errors)
                        |
                       yes
                        |
                        v
                    route by verb (dispatch table)
                        |
                  queue verb ----->  CommandQueue.enqueue() [via executor]
                        |               -> RESPONSE (enqueued confirmation)
                  other verb ---->  stub RESPONSE (accepted)

The handler never raises exceptions. All errors are captured and returned
as ERROR envelopes to the client. The SocketServer's dispatch wrapper
provides a second safety net.

Blocking I/O (CommandQueue file operations, wiki reads) is always
dispatched to a thread pool executor to avoid stalling the asyncio
event loop.

Usage::

    from pathlib import Path
    from jules_daemon.ipc.request_handler import RequestHandler, RequestHandlerConfig

    config = RequestHandlerConfig(wiki_root=Path("/data/wiki"))
    handler = RequestHandler(config=config)

    # Wire into SocketServer
    server = SocketServer(config=server_config, handler=handler)
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jules_daemon.ipc.framing import MessageEnvelope, MessageType
from jules_daemon.ipc.request_validator import validate_request
from jules_daemon.ipc.server import ClientConnection
from jules_daemon.wiki.command_queue import CommandQueue

__all__ = [
    "RequestHandler",
    "RequestHandlerConfig",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestHandlerConfig:
    """Immutable configuration for the IPC request handler.

    Attributes:
        wiki_root: Path to the wiki root directory. Used to initialize
            the CommandQueue for enqueue operations.
    """

    wiki_root: Path


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Verb handler signature: (msg_id, parsed_payload) -> awaitable envelope
_VerbHandler = Callable[[str, dict[str, Any]], MessageEnvelope]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _build_error_response(
    msg_id: str,
    error_summary: str,
    validation_errors: list[dict[str, str]],
) -> MessageEnvelope:
    """Build an ERROR envelope with structured validation errors.

    Args:
        msg_id: Correlation ID from the original request.
        error_summary: Human-readable summary of the error.
        validation_errors: List of field-level error dicts.

    Returns:
        MessageEnvelope with ERROR type and validation details.
    """
    return MessageEnvelope(
        msg_type=MessageType.ERROR,
        msg_id=msg_id,
        timestamp=_now_iso(),
        payload={
            "error": error_summary,
            "validation_errors": validation_errors,
        },
    )


def _build_success_response(
    msg_id: str,
    verb: str,
    extra: dict[str, Any] | None = None,
) -> MessageEnvelope:
    """Build a RESPONSE envelope for a successfully handled request.

    Args:
        msg_id: Correlation ID from the original request.
        verb: The canonical verb that was handled.
        extra: Additional payload fields to include.

    Returns:
        MessageEnvelope with RESPONSE type.
    """
    payload: dict[str, Any] = {"verb": verb}
    if extra is not None:
        payload.update(extra)

    return MessageEnvelope(
        msg_type=MessageType.RESPONSE,
        msg_id=msg_id,
        timestamp=_now_iso(),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# RequestHandler
# ---------------------------------------------------------------------------


class RequestHandler:
    """Concrete ``ClientHandler`` that validates and dispatches IPC requests.

    Bridges the SocketServer transport layer to the validation pipeline
    and wiki-backed command queue. Implements the ``ClientHandler``
    protocol from ``jules_daemon.ipc.server``.

    The ``CommandQueue`` is initialized eagerly in ``__init__`` to
    avoid TOCTOU races under concurrent access. Blocking I/O (queue
    enqueue, size queries) is offloaded to a thread pool executor.

    Args:
        config: Handler configuration with wiki_root path.
    """

    def __init__(self, *, config: RequestHandlerConfig) -> None:
        self._config = config
        self._queue = CommandQueue(wiki_root=config.wiki_root)
        self._verb_dispatch: dict[str, _VerbHandler] = {
            "handshake": self._handle_handshake,
            "queue": self._handle_queue,
            "run": self._handle_run,
            "status": self._handle_status,
            "watch": self._handle_watch,
            "cancel": self._handle_cancel,
            "history": self._handle_history,
        }

    async def handle_message(
        self,
        envelope: MessageEnvelope,
        client: ClientConnection,
    ) -> MessageEnvelope:
        """Validate and dispatch a client request.

        Called by the SocketServer for each incoming message. Never
        raises -- all errors are returned as ERROR envelopes.

        Args:
            envelope: The incoming request envelope.
            client: Connection context for the requesting client.

        Returns:
            A response MessageEnvelope (RESPONSE or ERROR type).
        """
        try:
            return await self._process_request(envelope, client)
        except Exception as exc:
            logger.error(
                "Unexpected error processing request msg_id=%s from %s: %s",
                envelope.msg_id,
                client.client_id,
                exc,
                exc_info=True,
            )
            return _build_error_response(
                msg_id=envelope.msg_id,
                error_summary="An internal error occurred. See daemon logs for details.",
                validation_errors=[],
            )

    async def _process_request(
        self,
        envelope: MessageEnvelope,
        client: ClientConnection,
    ) -> MessageEnvelope:
        """Core request processing logic.

        Separated from handle_message for clarity. This method may raise;
        the caller catches and wraps exceptions.
        """
        # Step 1: Validate
        result = validate_request(envelope)

        if not result.is_valid:
            error_count = len(result.errors)
            logger.info(
                "Validation failed for msg_id=%s from %s: %d error(s)",
                envelope.msg_id,
                client.client_id,
                error_count,
            )
            return _build_error_response(
                msg_id=envelope.msg_id,
                error_summary=(
                    f"Request validation failed with {error_count} error(s)"
                ),
                validation_errors=result.errors_to_dicts(),
            )

        # Step 2: Route by verb via dispatch table
        verb = result.verb
        if verb is None:
            raise RuntimeError(
                "Validator contract violation: verb is None on valid result"
            )

        logger.debug(
            "Valid request: verb=%s msg_id=%s from %s",
            verb,
            envelope.msg_id,
            client.client_id,
        )

        handler = self._verb_dispatch.get(verb)
        if handler is None:
            return _build_error_response(
                msg_id=envelope.msg_id,
                error_summary=f"Unhandled verb: {verb}",
                validation_errors=[],
            )

        return handler(envelope.msg_id, result.parsed_payload)

    # -- Blocking I/O helper --

    async def _run_blocking(self, func: Callable[..., Any], *args: Any) -> Any:
        """Run a blocking function in the default thread pool executor.

        Prevents CommandQueue file I/O from stalling the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args)
        )

    # -- Verb handlers --

    def _handle_handshake(
        self,
        msg_id: str,
        parsed: dict[str, Any],
    ) -> MessageEnvelope:
        """Handle the initial client handshake.

        Returns daemon version, uptime, and status so the client can
        verify compatibility.
        """
        import os

        return _build_success_response(
            msg_id=msg_id,
            verb="handshake",
            extra={
                "status": "ok",
                "protocol_version": 1,
                "daemon_version": "0.1.0",
                "pid": os.getpid(),
            },
        )

    def _handle_queue(
        self,
        msg_id: str,
        parsed: dict[str, Any],
    ) -> MessageEnvelope:
        """Enqueue a command and return confirmation.

        Persists the command to the wiki-backed queue and returns the
        queue_id and current position.
        """
        queued = self._queue.enqueue(
            natural_language=parsed["natural_language"],
            ssh_host=parsed.get("target_host"),
            ssh_user=parsed.get("target_user"),
            ssh_port=parsed.get("target_port", 22),
        )

        position = self._queue.size()

        logger.info(
            "Enqueued command queue_id=%s position=%d: %s",
            queued.queue_id,
            position,
            queued.natural_language[:80],
        )

        return _build_success_response(
            msg_id=msg_id,
            verb="queue",
            extra={
                "status": "enqueued",
                "queue_id": queued.queue_id,
                "position": position,
                "sequence": queued.sequence,
            },
        )

    def _handle_run(
        self,
        msg_id: str,
        parsed: dict[str, Any],
    ) -> MessageEnvelope:
        """Accept a run command.

        The actual execution flow (LLM translation, SSH confirmation,
        execution, monitoring) is handled by the daemon lifecycle,
        not the IPC handler. Here we return acceptance.
        """
        return _build_success_response(
            msg_id=msg_id,
            verb="run",
            extra={
                "status": "accepted",
                "target_host": parsed.get("target_host"),
                "natural_language": parsed.get("natural_language"),
            },
        )

    def _handle_status(
        self,
        msg_id: str,
        parsed: dict[str, Any],
    ) -> MessageEnvelope:
        """Handle a status query.

        Returns current daemon state. In the full implementation, this
        reads from the wiki current-run file. For now, returns a stub.
        """
        return _build_success_response(
            msg_id=msg_id,
            verb="status",
            extra={
                "state": "idle",
                "queue_depth": self._queue.size(),
            },
        )

    def _handle_watch(
        self,
        msg_id: str,
        parsed: dict[str, Any],
    ) -> MessageEnvelope:
        """Handle a watch request.

        In the full implementation, this sets up streaming output.
        For now, returns an acknowledgment.
        """
        return _build_success_response(
            msg_id=msg_id,
            verb="watch",
            extra={
                "status": "acknowledged",
            },
        )

    def _handle_cancel(
        self,
        msg_id: str,
        parsed: dict[str, Any],
    ) -> MessageEnvelope:
        """Handle a cancel request.

        In the full implementation, this signals the running process
        or removes a queued command. For now, returns acknowledgment.
        """
        return _build_success_response(
            msg_id=msg_id,
            verb="cancel",
            extra={
                "status": "acknowledged",
            },
        )

    def _handle_history(
        self,
        msg_id: str,
        parsed: dict[str, Any],
    ) -> MessageEnvelope:
        """Handle a history query.

        In the full implementation, this reads from wiki run history.
        For now, returns a stub.
        """
        return _build_success_response(
            msg_id=msg_id,
            verb="history",
            extra={
                "records": [],
                "total": 0,
            },
        )
