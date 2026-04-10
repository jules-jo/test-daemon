"""IPC request handler: concrete ClientHandler bridging validation and dispatch.

Implements the ``ClientHandler`` protocol expected by ``SocketServer``.
Each incoming ``MessageEnvelope`` is:

1. Validated via the ``request_validator`` layer (message type, verb, fields).
2. On validation failure: returns an ERROR envelope with structured errors.
3. On validation success: routes to the verb-specific handler:
   - **queue**: enqueues to the wiki-backed ``CommandQueue`` and returns
     an enqueue confirmation with queue_id and position.
   - **run**: sends a CONFIRM_PROMPT to the CLI, waits for approval,
     executes the command via SSH, and returns the result.
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
                  run verb ------> confirmation prompt -> SSH execution
                        |               -> RESPONSE (result)
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
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jules_daemon.execution.collision_check import (
    check_remote_processes,
    format_collision_warning,
)
from jules_daemon.execution.run_pipeline import RunResult, execute_run
from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
    unpack_header,
)
from jules_daemon.ipc.request_validator import validate_request
from jules_daemon.ipc.server import ClientConnection
from jules_daemon.ssh.credentials import resolve_ssh_credentials
from jules_daemon.wiki import current_run as current_run_io
from jules_daemon.wiki.command_queue import CommandQueue
from jules_daemon.wiki.run_promotion import list_history, read_history_entry

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

# Verb handler signature: (msg_id, parsed_payload) -> envelope
_VerbHandler = Callable[[str, dict[str, Any]], MessageEnvelope]

# Async verb handler that needs client access for multi-message flows
# (confirmation prompts, streaming, etc.)
_AsyncClientVerbHandler = Callable[
    [str, dict[str, Any], ClientConnection],
    Awaitable[MessageEnvelope],
]


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
        self._current_task: asyncio.Task[RunResult] | None = None
        self._current_run_id: str | None = None
        self._output_buffer: list[str] = []
        self._output_lock = asyncio.Lock()
        self._output_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._verb_dispatch: dict[str, _VerbHandler] = {
            "handshake": self._handle_handshake,
            "queue": self._handle_queue,
            "status": self._handle_status,
            "cancel": self._handle_cancel,
            "history": self._handle_history,
        }
        # Verbs that require async client access (multi-message flows)
        self._async_client_dispatch: dict[str, _AsyncClientVerbHandler] = {
            "run": self._handle_run,
            "watch": self._handle_watch,
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

        # Check for async client handlers first (multi-message flows)
        async_handler = self._async_client_dispatch.get(verb)
        if async_handler is not None:
            return await async_handler(
                envelope.msg_id, result.parsed_payload, client,
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

    async def _handle_run(
        self,
        msg_id: str,
        parsed: dict[str, Any],
        client: ClientConnection,
    ) -> MessageEnvelope:
        """Execute a run command with confirmation, collision check, and background execution.

        Full flow:
        1. Extract target host, user, port, and command from parsed payload
        2. Send a CONFIRM_PROMPT to the CLI with the proposed command
        3. Wait for the CLI to send a CONFIRM_REPLY (approve/deny)
        4. If denied, return a denial response
        5. If approved, check for running test processes on the remote host
        6. If processes found, warn the user and ask for a second confirmation
        7. Spawn execute_run() as a background asyncio task
        8. Return a "started" response immediately with the run_id

        Args:
            msg_id: Correlation ID from the original request.
            parsed: Validated payload with target_host, target_user,
                natural_language, and optional target_port.
            client: Connection context with reader/writer for the
                confirmation exchange.

        Returns:
            RESPONSE envelope with started status or denial.
        """
        target_host = parsed.get("target_host", "")
        target_user = parsed.get("target_user", "")
        natural_language = parsed.get("natural_language", "")
        target_port = parsed.get("target_port", 22)

        # Use the natural_language input directly as the SSH command
        # (explicit command path -- no LLM translation needed)
        proposed_command = natural_language

        # Step 1: Send CONFIRM_PROMPT to the CLI
        confirm_msg_id = f"confirm-{uuid.uuid4().hex[:12]}"
        confirm_prompt = MessageEnvelope(
            msg_type=MessageType.CONFIRM_PROMPT,
            msg_id=confirm_msg_id,
            timestamp=_now_iso(),
            payload={
                "proposed_command": proposed_command,
                "target_host": target_host,
                "target_user": target_user,
                "target_port": target_port,
                "original_msg_id": msg_id,
                "message": (
                    f"Execute on {target_user}@{target_host}:{target_port}?\n"
                    f"  $ {proposed_command}"
                ),
            },
        )

        try:
            await self._send_envelope(client, confirm_prompt)
        except Exception as exc:
            logger.warning(
                "Failed to send confirmation prompt for msg_id=%s: %s",
                msg_id,
                exc,
            )
            return _build_error_response(
                msg_id=msg_id,
                error_summary="Failed to send confirmation prompt to CLI",
                validation_errors=[],
            )

        # Step 2: Wait for CONFIRM_REPLY from the CLI
        try:
            reply = await self._read_envelope(
                client, timeout=120.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Confirmation timeout for msg_id=%s from %s",
                msg_id,
                client.client_id,
            )
            return _build_error_response(
                msg_id=msg_id,
                error_summary="Confirmation timed out (120s)",
                validation_errors=[],
            )
        except Exception as exc:
            logger.warning(
                "Failed to read confirmation reply for msg_id=%s: %s",
                msg_id,
                exc,
            )
            return _build_error_response(
                msg_id=msg_id,
                error_summary="Failed to read confirmation reply from CLI",
                validation_errors=[],
            )

        if reply is None:
            return _build_error_response(
                msg_id=msg_id,
                error_summary="CLI disconnected before confirming",
                validation_errors=[],
            )

        # Step 3: Check approval
        if reply.msg_type != MessageType.CONFIRM_REPLY:
            logger.warning(
                "Expected CONFIRM_REPLY but got %s for msg_id=%s",
                reply.msg_type.value,
                msg_id,
            )
            return _build_error_response(
                msg_id=msg_id,
                error_summary=(
                    f"Expected confirm_reply, got {reply.msg_type.value}"
                ),
                validation_errors=[],
            )

        approved = reply.payload.get("approved", False)
        if not approved:
            logger.info(
                "Run denied by user for msg_id=%s (host=%s)",
                msg_id,
                target_host,
            )
            return _build_success_response(
                msg_id=msg_id,
                verb="run",
                extra={
                    "status": "denied",
                    "target_host": target_host,
                    "message": "Command execution denied by user",
                },
            )

        logger.info(
            "Run approved for msg_id=%s: '%s' on %s@%s:%d",
            msg_id,
            proposed_command[:80],
            target_user,
            target_host,
            target_port,
        )

        # Step 4: Collision detection -- check for running test processes
        credential = resolve_ssh_credentials(target_host)
        remote_processes = await check_remote_processes(
            host=target_host,
            port=target_port,
            username=target_user,
            credential=credential,
        )

        if remote_processes:
            warning_text = format_collision_warning(remote_processes)
            collision_msg = MessageEnvelope(
                msg_type=MessageType.STREAM,
                msg_id=f"collision-{uuid.uuid4().hex[:12]}",
                timestamp=_now_iso(),
                payload={
                    "line": (
                        warning_text
                        + "Do you want to proceed anyway? "
                        "Waiting for confirmation...\n"
                    ),
                    "is_end": False,
                },
            )
            try:
                await self._send_envelope(client, collision_msg)
            except Exception:
                pass  # Best effort

            # Ask for a second confirmation
            collision_confirm = MessageEnvelope(
                msg_type=MessageType.CONFIRM_PROMPT,
                msg_id=f"collision-confirm-{uuid.uuid4().hex[:12]}",
                timestamp=_now_iso(),
                payload={
                    "proposed_command": proposed_command,
                    "target_host": target_host,
                    "original_msg_id": msg_id,
                    "message": (
                        "Test processes detected on remote host. "
                        "Proceed anyway?"
                    ),
                },
            )
            try:
                await self._send_envelope(client, collision_confirm)
            except Exception as exc:
                logger.warning(
                    "Failed to send collision confirmation for msg_id=%s: %s",
                    msg_id,
                    exc,
                )
                return _build_error_response(
                    msg_id=msg_id,
                    error_summary="Failed to send collision confirmation",
                    validation_errors=[],
                )

            try:
                collision_reply = await self._read_envelope(
                    client, timeout=120.0,
                )
            except asyncio.TimeoutError:
                return _build_error_response(
                    msg_id=msg_id,
                    error_summary="Collision confirmation timed out (120s)",
                    validation_errors=[],
                )
            except Exception:
                return _build_error_response(
                    msg_id=msg_id,
                    error_summary="Failed to read collision confirmation",
                    validation_errors=[],
                )

            if collision_reply is None:
                return _build_error_response(
                    msg_id=msg_id,
                    error_summary="CLI disconnected during collision check",
                    validation_errors=[],
                )

            collision_approved = collision_reply.payload.get("approved", False)
            if not collision_approved:
                logger.info(
                    "Run denied after collision warning for msg_id=%s",
                    msg_id,
                )
                return _build_success_response(
                    msg_id=msg_id,
                    verb="run",
                    extra={
                        "status": "denied",
                        "target_host": target_host,
                        "message": (
                            "Command execution denied after collision warning"
                        ),
                    },
                )

        # Step 5: Generate run_id and spawn background task
        run_id = f"run-{uuid.uuid4().hex[:12]}"

        # Send "connecting" status to CLI
        connecting_msg = MessageEnvelope(
            msg_type=MessageType.STREAM,
            msg_id=f"status-{uuid.uuid4().hex[:12]}",
            timestamp=_now_iso(),
            payload={
                "line": (
                    f"\nConnecting to {target_user}@{target_host}:{target_port}...\n"
                    f"Executing: {proposed_command}\n"
                    f"Test started. Use 'status' to check progress.\n"
                ),
                "is_end": False,
            },
        )
        try:
            await self._send_envelope(client, connecting_msg)
        except Exception:
            pass  # Best effort -- don't fail the run if status send fails

        # Spawn execute_run as a background task
        self._current_run_id = run_id
        self._current_task = asyncio.create_task(
            self._background_execute(
                target_host=target_host,
                target_user=target_user,
                command=proposed_command,
                target_port=target_port,
            ),
            name=f"run-{run_id}",
        )

        # Step 6: Return "started" response immediately
        return _build_success_response(
            msg_id=msg_id,
            verb="run",
            extra={
                "status": "started",
                "run_id": run_id,
                "target_host": target_host,
                "command": proposed_command,
                "message": "Test started. Use 'status' to check progress.",
            },
        )

    async def _background_execute(
        self,
        *,
        target_host: str,
        target_user: str,
        command: str,
        target_port: int,
    ) -> RunResult:
        """Run execute_run in the background, handling exceptions gracefully.

        On failure, writes FAILED state to the wiki so the status command
        can report it. Never allows exceptions to crash the daemon.

        After execution completes (success or failure), checks the queue
        and auto-starts the next queued command if one is available.

        Args:
            target_host: Remote hostname or IP address.
            target_user: SSH username.
            command: Shell command string to execute.
            target_port: SSH port number.

        Returns:
            RunResult from the execution pipeline.
        """
        # Clear output buffer for the new run
        async with self._output_lock:
            self._output_buffer = []
        # Drain any stale items from the output queue
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        def _on_output(line: str) -> None:
            """Thread-safe callback for streaming output lines.

            Appends to the shared buffer and puts into the asyncio queue
            for any active watch subscribers.
            """
            self._output_buffer.append(line)
            try:
                self._output_queue.put_nowait(line)
            except asyncio.QueueFull:
                pass  # Best effort for watch consumers

        try:
            result = await execute_run(
                target_host=target_host,
                target_user=target_user,
                command=command,
                target_port=target_port,
                wiki_root=self._config.wiki_root,
                on_output=_on_output,
            )
            logger.info(
                "Background run completed: run_id=%s success=%s",
                result.run_id,
                result.success,
            )
        except Exception as exc:
            logger.error(
                "Background run failed with unexpected error: %s",
                exc,
                exc_info=True,
            )
            # Write FAILED state to wiki so status can report it
            try:
                from jules_daemon.wiki.models import (
                    Command as WikiCommand,
                    CurrentRun,
                    ProcessIDs,
                    Progress,
                    RunStatus,
                    SSHTarget,
                )
                import os

                failed_run = CurrentRun(
                    status=RunStatus.FAILED,
                    ssh_target=SSHTarget(
                        host=target_host,
                        user=target_user,
                        port=target_port,
                    ),
                    command=WikiCommand(
                        natural_language=command,
                        resolved_shell=command,
                    ),
                    pids=ProcessIDs(daemon=os.getpid()),
                    error=str(exc),
                )
                current_run_io.write(self._config.wiki_root, failed_run)
            except Exception as wiki_exc:
                logger.error(
                    "Failed to write FAILED state to wiki: %s",
                    wiki_exc,
                )

            # Build a failed result rather than propagating the exception
            result = RunResult(
                success=False,
                run_id=self._current_run_id or "",
                command=command,
                target_host=target_host,
                target_user=target_user,
                error=str(exc),
            )

        # Signal end-of-stream to any watch subscribers
        try:
            self._output_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        # Check the queue for the next command to auto-execute
        self._try_start_next_queued()

        return result

    def _try_start_next_queued(self) -> None:
        """Check the queue and spawn the next command if available.

        Dequeues the next pending command and spawns a new background
        task for it. This creates a chain where each run checks the
        queue on completion.
        """
        remaining = self._queue.size()
        if remaining == 0:
            return

        next_cmd = self._queue.dequeue()
        if next_cmd is None:
            return

        remaining_after = self._queue.size()
        logger.info(
            "Starting queued command (queue_id=%s, remaining=%d)",
            next_cmd.queue_id,
            remaining_after,
        )

        run_id = f"run-{uuid.uuid4().hex[:12]}"
        self._current_run_id = run_id
        self._current_task = asyncio.create_task(
            self._background_execute(
                target_host=next_cmd.ssh_host or "",
                target_user=next_cmd.ssh_user or "",
                command=next_cmd.natural_language,
                target_port=next_cmd.ssh_port,
            ),
            name=f"run-{run_id}",
        )

    # -- Client I/O helpers for multi-message flows --

    @staticmethod
    async def _send_envelope(
        client: ClientConnection,
        envelope: MessageEnvelope,
    ) -> None:
        """Encode and send a framed envelope to the client.

        Args:
            client: The client connection with writer.
            envelope: The envelope to send.

        Raises:
            OSError: On connection failure.
        """
        frame = encode_frame(envelope)
        client.writer.write(frame)
        await client.writer.drain()

    @staticmethod
    async def _read_envelope(
        client: ClientConnection,
        *,
        timeout: float = 30.0,
    ) -> MessageEnvelope | None:
        """Read a single framed envelope from the client.

        Args:
            client: The client connection with reader.
            timeout: Maximum seconds to wait.

        Returns:
            The decoded MessageEnvelope, or None on EOF.

        Raises:
            asyncio.TimeoutError: If no message arrives within timeout.
        """
        try:
            header_bytes = await asyncio.wait_for(
                client.reader.readexactly(HEADER_SIZE),
                timeout=timeout,
            )
        except asyncio.IncompleteReadError:
            return None

        payload_length = unpack_header(header_bytes)
        payload_bytes = await asyncio.wait_for(
            client.reader.readexactly(payload_length),
            timeout=timeout,
        )
        return decode_envelope(payload_bytes)

    def _handle_status(
        self,
        msg_id: str,
        parsed: dict[str, Any],
    ) -> MessageEnvelope:
        """Handle a status query.

        Reads the wiki current-run file and checks the background task
        to determine the current daemon state. Returns run details if
        a run is active, or idle state with queue depth otherwise.
        """
        queue_depth = self._queue.size()

        # Check the background task state
        task_running = (
            self._current_task is not None
            and not self._current_task.done()
        )

        # Read wiki current-run state
        try:
            wiki_run = current_run_io.read(self._config.wiki_root)
        except Exception as exc:
            logger.warning("Failed to read current-run wiki: %s", exc)
            wiki_run = None

        if wiki_run is not None and wiki_run.is_active:
            # Compute duration so far
            duration_seconds: float | None = None
            if wiki_run.started_at is not None:
                delta = datetime.now(timezone.utc) - wiki_run.started_at
                duration_seconds = max(0.0, delta.total_seconds())

            # Map wiki RunStatus to a simple string for the response
            if task_running:
                status_str = "RUNNING"
            elif wiki_run.status.value == "running":
                # Task finished but wiki still says running -- check result
                status_str = "RUNNING"
            else:
                status_str = wiki_run.status.value.upper()

            extra: dict[str, Any] = {
                "state": "active",
                "run_id": wiki_run.run_id,
                "host": (
                    wiki_run.ssh_target.host
                    if wiki_run.ssh_target is not None
                    else ""
                ),
                "command": (
                    wiki_run.command.resolved_shell
                    if wiki_run.command is not None
                    else ""
                ),
                "status": status_str,
                "queue_depth": queue_depth,
            }
            if duration_seconds is not None:
                extra["duration_seconds"] = round(duration_seconds, 2)

            return _build_success_response(
                msg_id=msg_id,
                verb="status",
                extra=extra,
            )

        # Check if the background task completed with a terminal state
        if wiki_run is not None and wiki_run.is_terminal:
            duration_seconds = None
            if (
                wiki_run.started_at is not None
                and wiki_run.completed_at is not None
            ):
                delta = wiki_run.completed_at - wiki_run.started_at
                duration_seconds = max(0.0, delta.total_seconds())

            extra = {
                "state": "completed",
                "run_id": wiki_run.run_id,
                "host": (
                    wiki_run.ssh_target.host
                    if wiki_run.ssh_target is not None
                    else ""
                ),
                "command": (
                    wiki_run.command.resolved_shell
                    if wiki_run.command is not None
                    else ""
                ),
                "status": wiki_run.status.value.upper(),
                "queue_depth": queue_depth,
            }
            if duration_seconds is not None:
                extra["duration_seconds"] = round(duration_seconds, 2)
            if wiki_run.error is not None:
                extra["error"] = wiki_run.error

            return _build_success_response(
                msg_id=msg_id,
                verb="status",
                extra=extra,
            )

        # No active run
        return _build_success_response(
            msg_id=msg_id,
            verb="status",
            extra={
                "state": "idle",
                "queue_depth": queue_depth,
            },
        )

    async def _handle_watch(
        self,
        msg_id: str,
        parsed: dict[str, Any],
        client: ClientConnection,
    ) -> MessageEnvelope:
        """Handle a watch request with streaming output.

        Sends buffered output lines as STREAM messages, then polls for
        new lines every 1 second until the run completes. Sends an
        end-of-stream STREAM message when done.

        Args:
            msg_id: Correlation ID from the original request.
            parsed: Validated payload (currently unused).
            client: Connection context for streaming.

        Returns:
            RESPONSE envelope after streaming completes.
        """
        task_running = (
            self._current_task is not None
            and not self._current_task.done()
        )

        if not task_running and not self._output_buffer:
            return _build_success_response(
                msg_id=msg_id,
                verb="watch",
                extra={
                    "status": "no_active_run",
                    "message": "No test is currently running and no output is buffered",
                },
            )

        # Step 1: Send existing buffer lines as STREAM messages
        async with self._output_lock:
            snapshot = list(self._output_buffer)
        sent_count = len(snapshot)

        for line in snapshot:
            stream_msg = MessageEnvelope(
                msg_type=MessageType.STREAM,
                msg_id=f"watch-{uuid.uuid4().hex[:12]}",
                timestamp=_now_iso(),
                payload={"line": line, "is_end": False},
            )
            try:
                await self._send_envelope(client, stream_msg)
            except Exception:
                break  # Client disconnected

        # Step 2: Poll for new lines until run completes
        while (
            self._current_task is not None
            and not self._current_task.done()
        ):
            try:
                line = await asyncio.wait_for(
                    self._output_queue.get(), timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            if line is None:
                # End sentinel
                break

            sent_count += 1
            stream_msg = MessageEnvelope(
                msg_type=MessageType.STREAM,
                msg_id=f"watch-{uuid.uuid4().hex[:12]}",
                timestamp=_now_iso(),
                payload={"line": line, "is_end": False},
            )
            try:
                await self._send_envelope(client, stream_msg)
            except Exception:
                break

        # Drain any remaining items in the queue
        while not self._output_queue.empty():
            try:
                line = self._output_queue.get_nowait()
                if line is None:
                    break
                sent_count += 1
                stream_msg = MessageEnvelope(
                    msg_type=MessageType.STREAM,
                    msg_id=f"watch-{uuid.uuid4().hex[:12]}",
                    timestamp=_now_iso(),
                    payload={"line": line, "is_end": False},
                )
                await self._send_envelope(client, stream_msg)
            except (asyncio.QueueEmpty, Exception):
                break

        # Step 3: Send end-of-stream
        end_msg = MessageEnvelope(
            msg_type=MessageType.STREAM,
            msg_id=f"watch-end-{uuid.uuid4().hex[:12]}",
            timestamp=_now_iso(),
            payload={"line": "", "is_end": True},
        )
        try:
            await self._send_envelope(client, end_msg)
        except Exception:
            pass

        return _build_success_response(
            msg_id=msg_id,
            verb="watch",
            extra={
                "status": "completed",
                "lines_sent": sent_count,
            },
        )

    def _handle_cancel(
        self,
        msg_id: str,
        parsed: dict[str, Any],
    ) -> MessageEnvelope:
        """Handle a cancel request.

        If a background task is running, cancels the asyncio task,
        updates the wiki current-run state to CANCELLED, and clears
        the internal task references. Returns success with the
        cancelled run_id.

        If no task is running, returns an error response.
        """
        if self._current_task is None or self._current_task.done():
            return _build_error_response(
                msg_id=msg_id,
                error_summary="No test is currently running",
                validation_errors=[],
            )

        cancelled_run_id = self._current_run_id or ""

        # Cancel the asyncio task
        self._current_task.cancel()

        # Update wiki current-run state to CANCELLED
        try:
            wiki_run = current_run_io.read(self._config.wiki_root)
            if wiki_run is not None and wiki_run.is_active:
                cancelled_run = wiki_run.with_cancelled()
                current_run_io.write(self._config.wiki_root, cancelled_run)
                logger.info(
                    "Updated wiki state to CANCELLED for run_id=%s",
                    cancelled_run_id,
                )
        except Exception as exc:
            logger.warning(
                "Failed to update wiki state to CANCELLED: %s", exc,
            )

        # Clear internal references
        self._current_task = None
        self._current_run_id = None

        logger.info("Cancelled run run_id=%s", cancelled_run_id)

        return _build_success_response(
            msg_id=msg_id,
            verb="cancel",
            extra={
                "status": "cancelled",
                "run_id": cancelled_run_id,
            },
        )

    def _handle_history(
        self,
        msg_id: str,
        parsed: dict[str, Any],
    ) -> MessageEnvelope:
        """Handle a history query.

        Scans the wiki history directory for completed run files,
        parses YAML frontmatter from each, and returns run summaries
        sorted by created_at descending (newest first).

        Supports optional filters in the parsed payload:
        - limit: Maximum records to return (default 20).
        - status_filter: Only return runs matching this status string.
        - host_filter: Only return runs matching this host string.
        """
        limit = int(parsed.get("limit", 20))
        status_filter: str | None = parsed.get("status_filter")
        host_filter: str | None = parsed.get("host_filter")

        history_dir = self._config.wiki_root / "pages" / "daemon" / "history"
        if not history_dir.exists():
            return _build_success_response(
                msg_id=msg_id,
                verb="history",
                extra={"records": [], "total": 0},
            )

        records: list[dict[str, Any]] = []
        for md_file in sorted(history_dir.glob("run-*.md")):
            try:
                raw = md_file.read_text(encoding="utf-8")
                from jules_daemon.wiki.frontmatter import parse as parse_fm

                doc = parse_fm(raw)
                fm = doc.frontmatter

                run_id = fm.get("run_id", "")
                status = fm.get("status", "")
                created_at = fm.get("created") or fm.get("completed_at") or ""
                host = ""
                ssh_target = fm.get("ssh_target")
                if isinstance(ssh_target, dict):
                    host = ssh_target.get("host", "")

                command_text = ""
                command_data = fm.get("command")
                if isinstance(command_data, dict):
                    command_text = command_data.get("resolved_shell", "")

                exit_code = None
                error = fm.get("error")
                if status == "completed":
                    exit_code = 0
                elif status == "failed" and isinstance(error, str):
                    # Try to extract exit code from error message
                    import re
                    code_match = re.search(r"code\s+(\d+)", error)
                    if code_match:
                        exit_code = int(code_match.group(1))

                duration: float | None = None
                started_at_str = fm.get("started_at")
                completed_at_str = fm.get("completed_at")
                if started_at_str and completed_at_str:
                    try:
                        started_dt = datetime.fromisoformat(str(started_at_str))
                        completed_dt = datetime.fromisoformat(str(completed_at_str))
                        duration = round(
                            (completed_dt - started_dt).total_seconds(), 2,
                        )
                    except (ValueError, TypeError):
                        pass

                # Apply filters
                if status_filter and status != status_filter:
                    continue
                if host_filter and host != host_filter:
                    continue

                records.append({
                    "run_id": run_id,
                    "host": host,
                    "command": command_text,
                    "status": status,
                    "exit_code": exit_code,
                    "duration": duration,
                    "created_at": str(created_at) if created_at else None,
                })
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "Skipping malformed history file %s: %s", md_file, exc,
                )
                continue

        # Sort by created_at descending (newest first)
        records.sort(
            key=lambda r: r.get("created_at") or "",
            reverse=True,
        )

        total = len(records)
        records = records[:limit]

        return _build_success_response(
            msg_id=msg_id,
            verb="history",
            extra={"records": records, "total": total},
        )
