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

from jules_daemon.audit.models import AuditRecord
from jules_daemon.audit.run_audit_builder import (
    build_confirmation_record,
    build_nl_input_record,
    build_parsed_command_record,
    build_ssh_execution_record,
    build_structured_result_record,
    create_initial_audit,
    safe_write_audit_async,
    truncate_text,
)
from jules_daemon.execution.collision_check import (
    check_remote_processes,
    format_collision_warning,
)
from jules_daemon.execution.knowledge_extractor import extract_knowledge
from jules_daemon.execution.output_summarizer import (
    OutputSummary,
    summarize_output,
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
from jules_daemon.classifier.direct_command import (
    DirectCommandDetection,
    detect_direct_command,
)
from jules_daemon.ipc.notification_broadcaster import NotificationBroadcaster
from jules_daemon.ipc.notification_emitter import (
    emit_agent_loop_completion,
    emit_run_completion,
)
from jules_daemon.ipc.request_validator import validate_request
from jules_daemon.ipc.server import ClientConnection
from jules_daemon.ssh.credentials import resolve_ssh_credentials
from jules_daemon.wiki import current_run as current_run_io
from jules_daemon.wiki.command_queue import CommandQueue
from jules_daemon.execution.test_discovery import (
    DiscoveredTestSpec,
    discover_test,
    format_spec_preview,
    save_discovered_spec,
)
from jules_daemon.wiki.test_knowledge import (
    TestKnowledge,
    derive_test_slug,
    load_test_knowledge,
    merge_knowledge,
    save_test_knowledge,
)

# Conditional LLM imports -- these are Optional and never crash the daemon.
# The TYPE_CHECKING guard keeps the type annotations available to type
# checkers while avoiding import-time failures if LLM dependencies are
# somehow not installed (defensive, since they are listed in project deps).
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

    from jules_daemon.llm.command_translator import CommandTranslator
    from jules_daemon.llm.config import LLMConfig

__all__ = [
    "RequestHandler",
    "RequestHandlerConfig",
    "detect_direct_command",
    "is_direct_command",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit summary formatting
# ---------------------------------------------------------------------------

# Maximum number of characters of the raw stdout/stderr that the audit
# writer embeds in the ``summary`` / ``error_message`` fields. Smaller
# than :data:`AUDIT_OUTPUT_LIMIT` because the new structured summary
# (counts + narrative) carries the high-level information. The raw
# tail is kept as a fallback for debugging when a parser label is
# ``"none"``.
_AUDIT_RAW_TAIL_LIMIT: int = 2_000


def _format_output_summary(*, summary_obj: OutputSummary) -> str:
    """Render an :class:`OutputSummary` as human-readable multi-line text.

    The shape mirrors what audit-file readers expect:

        <narrative>

        Parser: <parser>
        Passed: N | Failed: N | Skipped: N
        Duration: N.NNs            # only when available
        Key failures:              # only when non-empty
          - ...
          - ...

    Empty values are omitted so short summaries stay scannable.
    """
    lines: list[str] = []
    if summary_obj.narrative:
        lines.append(summary_obj.narrative)
        lines.append("")
    lines.append(f"Parser: {summary_obj.parser}")
    lines.append(
        f"Passed: {summary_obj.passed} | "
        f"Failed: {summary_obj.failed} | "
        f"Skipped: {summary_obj.skipped}"
    )
    if summary_obj.duration_seconds is not None:
        lines.append(f"Duration: {summary_obj.duration_seconds:.2f}s")
    if summary_obj.key_failures:
        lines.append("")
        lines.append("Key failures:")
        for failure in summary_obj.key_failures:
            lines.append(f"  - {failure}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Direct-command detection
# ---------------------------------------------------------------------------

# Known executable prefixes that indicate a direct shell command.
# If the user input starts with one of these tokens (case-sensitive),
# it is treated as a verbatim command and the LLM is bypassed.
_DIRECT_COMMAND_PREFIXES: tuple[str, ...] = (
    "python",
    "python3",
    "pytest",
    "npm",
    "npx",
    "node",
    "go",
    "make",
    "bash",
    "sh",
    "zsh",
    "./",
    "docker",
    "docker-compose",
    "cargo",
    "mvn",
    "gradle",
    "java",
    "javac",
    "ruby",
    "bundle",
    "perl",
    "php",
    "dotnet",
    "cmake",
    "gcc",
    "g++",
    "clang",
    "rustc",
    "ls",
    "cat",
    "cd",
    "grep",
    "find",
    "which",
    "echo",
    "env",
    "export",
    "pip",
    "pip3",
    "uv",
    "poetry",
)


def is_direct_command(text: str) -> bool:
    """Determine whether *text* looks like a direct shell command.

    A direct command starts with a known executable name (e.g. ``python``,
    ``pytest``, ``./``) or a path separator, meaning the user typed an
    exact command rather than natural language that needs LLM translation.

    The check is intentionally conservative: if in doubt, return ``False``
    so the LLM can translate the ambiguous input.

    Args:
        text: The raw user input string.

    Returns:
        ``True`` if *text* appears to be a verbatim shell command.
    """
    stripped = text.strip()
    if not stripped:
        return False

    first_token = stripped.split()[0]

    # Absolute or explicit relative path (e.g. /usr/bin/pytest, ./run.sh)
    if first_token.startswith("/") or first_token.startswith("./"):
        return True

    # Known executable prefix (exact match on the first whitespace-delimited token)
    return first_token in _DIRECT_COMMAND_PREFIXES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestHandlerConfig:
    """Immutable configuration for the IPC request handler.

    Attributes:
        wiki_root: Path to the wiki root directory. Used to initialize
            the CommandQueue for enqueue operations.
        llm_client: Optional OpenAI client pre-configured for Dataiku Mesh.
            When ``None``, LLM translation is disabled and natural-language
            input is used as-is (backward-compatible behavior).
        llm_config: Optional LLMConfig required when *llm_client* is set.
        one_shot: When ``True``, forces the one-shot LLM translation
            path even when the agent loop is available. Defaults to
            ``False`` (agent loop is the default for NL commands).
        max_agent_iterations: Hard cap on think-act-observe cycles per
            agent loop invocation. Defaults to 5.
        notification_broadcaster: Optional broadcaster for pushing
            completion and alert events to subscribed CLI clients.
            When ``None``, no notification events are emitted (the
            notification channel is effectively disabled).
    """

    wiki_root: Path
    llm_client: OpenAI | None = None
    llm_config: LLMConfig | None = None
    one_shot: bool = False
    max_agent_iterations: int = 5
    notification_broadcaster: NotificationBroadcaster | None = None


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


class _RegistryDispatcher:
    """Thin adapter bridging ToolRegistry to the ToolDispatcher protocol.

    The AgentLoop expects a ``ToolDispatcher`` with a ``dispatch(call)``
    method, while ToolRegistry exposes ``execute(call)``. This adapter
    satisfies the protocol contract without modifying either class.
    """

    __slots__ = ("_registry",)

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    async def dispatch(self, call: Any) -> Any:
        """Delegate to ToolRegistry.execute().

        Args:
            call: A ToolCall instance.

        Returns:
            ToolResult from the registry.
        """
        return await self._registry.execute(call)


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
        # Last failure message, shown to the next CLI that connects
        # after a background run fails. Cleared when displayed.
        self._last_failure: str | None = None
        # Last completed run (success or failure) kept in memory so
        # `status` can report it after promote_run() clears the wiki
        # current-run state. Cleared when a new run starts.
        self._last_completed_run: RunResult | None = None

        # LLM command translator -- optional, None when env vars are not set.
        self._command_translator: CommandTranslator | None = None
        if config.llm_client is not None and config.llm_config is not None:
            from jules_daemon.llm.command_translator import (
                CommandTranslator as _CT,
            )

            self._command_translator = _CT(
                client=config.llm_client,
                config=config.llm_config,
            )
            logger.info("LLM command translator initialized")
        else:
            logger.info(
                "LLM command translator not configured -- "
                "natural-language input will be used as-is"
            )

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
            "discover": self._handle_discover,
        }

    # -- Agent loop availability -------------------------------------------------

    @property
    def _can_use_agent_loop(self) -> bool:
        """Determine whether the agent loop is available for NL commands.

        The agent loop requires:
        1. LLM client is configured (llm_client and llm_config are set)
        2. one_shot mode is not forced via config flag

        When any condition fails, the handler falls back to the v1.2-mvp
        one-shot LLM translation path.

        Returns:
            True if the agent loop can be initialized for this request.
        """
        if self._config.one_shot:
            return False
        if self._config.llm_client is None:
            return False
        if self._config.llm_config is None:
            return False
        return True

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

    # -- LLM translation helper --

    async def _translate_via_llm(
        self,
        *,
        natural_language: str,
        target_host: str,
        target_user: str,
        target_port: int,
    ) -> str:
        """Translate natural-language input to a shell command via the LLM.

        If the LLM translator is not configured (env vars missing) or the
        LLM call fails for any reason, the original *natural_language* text
        is returned as-is with a warning log. This guarantees the daemon
        never crashes due to an LLM issue.

        Wiki translation history for the target host is included as extra
        context so the LLM can make better guesses based on past commands.

        Args:
            natural_language: The user's free-text request.
            target_host: Remote hostname or IP.
            target_user: SSH username.
            target_port: SSH port number.

        Returns:
            A concrete shell command string (either LLM-translated or the
            original input as fallback).
        """
        if self._command_translator is None:
            logger.debug(
                "LLM not configured, using input as-is: %s",
                natural_language[:80],
            )
            return natural_language

        # Build host context with wiki history for better translation
        extra_context = self._build_wiki_context(
            target_host=target_host,
        )

        from jules_daemon.llm.prompts import HostContext

        host_context = HostContext(
            hostname=target_host,
            user=target_user,
            port=target_port,
            extra_context=tuple(extra_context),
        )

        try:
            # The translator is synchronous (blocking LLM HTTP call).
            # Run it in the thread pool to avoid stalling the event loop.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(
                    self._command_translator.translate,
                    natural_language=natural_language,
                    host_context=host_context,
                ),
            )

            if result.is_refusal:
                logger.warning(
                    "LLM refused to translate '%s': %s",
                    natural_language[:80],
                    result.response.explanation,
                )
                return natural_language

            if result.command_count == 0:
                logger.warning(
                    "LLM returned 0 commands for '%s', using input as-is",
                    natural_language[:80],
                )
                return natural_language

            # Use the first command from the LLM response.
            # Multiple commands are joined with && for sequential execution.
            commands = [cmd.command for cmd in result.ssh_commands]
            translated = " && ".join(commands)

            logger.info(
                "LLM translated '%s' -> '%s' (confidence=%s, %.2fs)",
                natural_language[:60],
                translated[:80],
                result.response.confidence.value,
                result.elapsed_seconds,
            )
            return translated

        except Exception as exc:
            # Never let LLM failures block command execution.
            # Fall back to using the raw input.
            logger.warning(
                "LLM translation failed for '%s', using input as-is: %s",
                natural_language[:80],
                exc,
            )
            return natural_language

    def _build_wiki_context(
        self,
        *,
        target_host: str,
    ) -> list[str]:
        """Gather wiki translation history for the target host.

        Returns a list of context strings suitable for inclusion in the
        LLM prompt's ``extra_context`` field. Reads from the wiki
        translation store; returns an empty list if the wiki is empty
        or an error occurs.

        Filters and ordering:
          - DENIED entries are skipped (the user rejected those, so they
            shouldn't influence future proposals).
          - EDITED entries are listed FIRST -- they represent the user's
            corrections of LLM proposals and should be treated as ground
            truth. The prompt explicitly tells the LLM to mirror their
            format.
          - Within each group, entries are sorted by created_at descending
            (newest first) so the LLM weights the most recent corrections.
          - Up to 8 entries total (5 edited + 3 approved as headroom).

        Args:
            target_host: The SSH host to look up history for.

        Returns:
            List of human-readable context strings.
        """
        context_lines: list[str] = []
        try:
            from jules_daemon.wiki.command_translation import (
                TranslationOutcome,
                list_all,
            )

            all_translations = list_all(self._config.wiki_root)
            host_translations = [
                t for t in all_translations
                if t.ssh_host == target_host
            ]

            # Sort newest first
            host_translations.sort(
                key=lambda t: t.created_at,
                reverse=True,
            )

            # Partition by outcome (skip DENIED)
            edited = [
                t for t in host_translations
                if t.outcome == TranslationOutcome.EDITED
            ][:5]
            approved = [
                t for t in host_translations
                if t.outcome == TranslationOutcome.APPROVED
            ][:3]

            if edited:
                context_lines.append(
                    "### USER-CORRECTED COMMANDS (treat as ground truth)"
                )
                context_lines.append(
                    "These are corrections the user made to previous LLM proposals. "
                    "Mirror this exact format -- including quoting, flags, and "
                    "argument style -- when generating new commands."
                )
                for t in edited:
                    context_lines.append(
                        f"- '{t.natural_language}' -> `{t.resolved_shell}`"
                    )

            if approved:
                context_lines.append("")
                context_lines.append("### Previously approved commands (for reference)")
                for t in approved:
                    context_lines.append(
                        f"- '{t.natural_language}' -> `{t.resolved_shell}`"
                    )
        except Exception as exc:
            logger.debug(
                "Failed to read wiki translation history for %s: %s",
                target_host,
                exc,
            )

        return context_lines

    # -- Verb handlers --

    def _handle_handshake(
        self,
        msg_id: str,
        parsed: dict[str, Any],
    ) -> MessageEnvelope:
        """Handle the initial client handshake.

        Returns daemon version, uptime, and status so the client can
        verify compatibility. Also includes any unreported failure from
        a background run (cleared after being delivered).
        """
        import os

        extra: dict[str, Any] = {
            "status": "ok",
            "protocol_version": 1,
            "daemon_version": "0.1.0",
            "pid": os.getpid(),
        }

        # Deliver any pending failure notification to the next client
        if self._last_failure:
            extra["pending_failure"] = self._last_failure
            self._last_failure = None

        return _build_success_response(
            msg_id=msg_id,
            verb="handshake",
            extra=extra,
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

    async def _handle_discover(
        self,
        msg_id: str,
        parsed: dict[str, Any],
        client: ClientConnection,
    ) -> MessageEnvelope:
        """Discover a test spec by running command -h on the remote host.

        Flow:
        1. SSH in and run ``command -h`` (falling back to ``--help``).
        2. Send captured help text to the LLM for structured parsing.
        3. Send a CONFIRM_PROMPT with the draft spec for user review.
        4. If approved, write the wiki file and return the path.

        Args:
            msg_id: Correlation ID from the original request.
            parsed: Validated payload with target_host, target_user,
                command, and optional target_port.
            client: Connection context for the confirmation exchange.

        Returns:
            RESPONSE envelope with the discovery result.
        """
        target_host = parsed.get("target_host", "")
        target_user = parsed.get("target_user", "")
        command = parsed.get("command", "")
        target_port = parsed.get("target_port", 22)

        logger.info(
            "Discover request: %s@%s:%d %s (msg_id=%s)",
            target_user,
            target_host,
            target_port,
            command[:80],
            msg_id,
        )

        # Step 1: Ask permission to run -h on the remote host
        help_command = f"{command} -h"
        pre_confirm_id = f"discover-pre-{uuid.uuid4().hex[:12]}"
        pre_confirm = MessageEnvelope(
            msg_type=MessageType.CONFIRM_PROMPT,
            msg_id=pre_confirm_id,
            timestamp=_now_iso(),
            payload={
                "proposed_command": help_command,
                "target_host": target_host,
                "target_user": target_user,
                "message": (
                    f"Will run '{help_command}' on "
                    f"{target_user}@{target_host}:{target_port} "
                    f"to discover test arguments."
                ),
            },
        )
        try:
            await self._send_envelope(client, pre_confirm)
        except Exception as exc:
            return _build_error_response(
                msg_id=msg_id,
                error_summary="Failed to send discovery confirmation",
                validation_errors=[],
            )

        try:
            pre_reply = await self._read_envelope(client, timeout=120.0)
        except (asyncio.TimeoutError, Exception):
            return _build_error_response(
                msg_id=msg_id,
                error_summary="Discovery confirmation timed out",
                validation_errors=[],
            )

        if pre_reply is None or not pre_reply.payload.get("approved", False):
            return _build_success_response(
                msg_id=msg_id,
                verb="discover",
                extra={"status": "denied", "message": "Discovery cancelled."},
            )

        # Step 2: SSH in, run -h, parse with LLM
        status_msg = MessageEnvelope(
            msg_type=MessageType.STREAM,
            msg_id=f"discover-status-{uuid.uuid4().hex[:12]}",
            timestamp=_now_iso(),
            payload={
                "line": (
                    f"\nRunning '{help_command}' on "
                    f"{target_user}@{target_host}...\n"
                ),
                "is_end": False,
            },
        )
        try:
            await self._send_envelope(client, status_msg)
        except Exception:
            pass

        try:
            spec = await discover_test(
                host=target_host,
                user=target_user,
                command=command,
                port=target_port,
                llm_client=self._config.llm_client,
                llm_config=self._config.llm_config,
            )
        except Exception as exc:
            logger.error("Discovery failed for msg_id=%s: %s", msg_id, exc)
            return _build_error_response(
                msg_id=msg_id,
                error_summary=f"Discovery failed: {exc}",
                validation_errors=[],
            )

        if spec is None:
            return _build_error_response(
                msg_id=msg_id,
                error_summary=(
                    f"Could not fetch help output for '{command}' "
                    f"on {target_user}@{target_host}:{target_port}"
                ),
                validation_errors=[],
            )

        # Step 3: Show discovered spec as a STREAM message, then ask to save
        preview = format_spec_preview(spec)
        preview_msg = MessageEnvelope(
            msg_type=MessageType.STREAM,
            msg_id=f"discover-preview-{uuid.uuid4().hex[:12]}",
            timestamp=_now_iso(),
            payload={
                "line": f"\n{preview}\n",
                "is_end": False,
            },
        )
        try:
            await self._send_envelope(client, preview_msg)
        except Exception:
            pass

        confirm_msg_id = f"discover-confirm-{uuid.uuid4().hex[:12]}"
        confirm_prompt = MessageEnvelope(
            msg_type=MessageType.CONFIRM_PROMPT,
            msg_id=confirm_msg_id,
            timestamp=_now_iso(),
            payload={
                "prompt_title": "Save Test Spec to Wiki",
                "message": "Save this test spec to your local wiki?",
            },
        )

        try:
            await self._send_envelope(client, confirm_prompt)
        except Exception:
            return _build_error_response(
                msg_id=msg_id,
                error_summary="Failed to send discovery confirmation prompt",
                validation_errors=[],
            )

        # Step 3: Wait for user approval
        try:
            reply = await self._read_envelope(client, timeout=120.0)
        except (asyncio.TimeoutError, Exception):
            return _build_error_response(
                msg_id=msg_id,
                error_summary="Discovery confirmation timed out",
                validation_errors=[],
            )

        if reply is None:
            return _build_error_response(
                msg_id=msg_id,
                error_summary="CLI disconnected during discovery confirmation",
                validation_errors=[],
            )

        approved = reply.payload.get("approved", False)
        if not approved:
            logger.info("User declined to save discovered spec for msg_id=%s", msg_id)
            return _build_success_response(
                msg_id=msg_id,
                verb="discover",
                extra={
                    "status": "declined",
                    "message": "Discovery result not saved.",
                    "preview": preview,
                },
            )

        # Step 4: Write the wiki file
        try:
            wiki_path = await self._run_blocking(
                save_discovered_spec,
                self._config.wiki_root,
                spec,
                command,
                target_host,
            )
        except Exception as exc:
            logger.error("Failed to save discovered spec: %s", exc)
            return _build_error_response(
                msg_id=msg_id,
                error_summary=f"Failed to write wiki file: {exc}",
                validation_errors=[],
            )

        logger.info(
            "Saved discovered spec to %s for msg_id=%s", wiki_path, msg_id,
        )
        return _build_success_response(
            msg_id=msg_id,
            verb="discover",
            extra={
                "status": "saved",
                "wiki_path": str(wiki_path),
                "preview": preview,
                "message": f"Test spec saved to {wiki_path}",
            },
        )

    async def _handle_run(
        self,
        msg_id: str,
        parsed: dict[str, Any],
        client: ClientConnection,
    ) -> MessageEnvelope:
        """Route run commands to the appropriate execution path.

        Direct commands (starting with known executables) always use the
        one-shot path. Natural language commands use the agent loop by
        default, with one-shot as fallback when:
          (a) LLM is not configured
          (b) Agent loop initialization fails
          (c) Explicit ``--one-shot`` flag is set in config

        Args:
            msg_id: Correlation ID from the original request.
            parsed: Validated payload with target_host, target_user,
                natural_language, and optional target_port.
            client: Connection context with reader/writer for the
                confirmation exchange.

        Returns:
            RESPONSE envelope with started/completed/denied status.
        """
        natural_language = parsed.get("natural_language", "")

        # Fast, sub-millisecond direct-command detection. When the input
        # starts with a known executable (pytest, python3, ./script, etc.)
        # the agent loop is bypassed entirely and the command goes straight
        # to the SSH approval flow -- preserving v1.2-mvp latency.
        detection = detect_direct_command(natural_language)

        if detection.bypass_agent_loop:
            logger.debug(
                "Direct command detected (executable=%s, confidence=%.1f), "
                "bypassing agent loop: %s",
                detection.executable,
                detection.confidence,
                natural_language[:80],
            )
            return await self._handle_run_oneshot(
                msg_id, parsed, client, detection=detection,
            )

        # NL commands: try agent loop first if available
        if self._can_use_agent_loop:
            try:
                return await self._handle_run_agent_loop(
                    msg_id, parsed, client,
                )
            except Exception as exc:
                # Import here to avoid top-level coupling to agent module
                try:
                    from jules_daemon.agent.error_classification import (
                        RetryExhaustedError,
                    )
                    is_retry_exhausted = isinstance(exc, RetryExhaustedError)
                except ImportError:
                    is_retry_exhausted = False

                if is_retry_exhausted:
                    logger.warning(
                        "Agent loop transient retries exhausted for "
                        "msg_id=%s (iterations=%d), falling back to "
                        "one-shot LLM translation: %s",
                        msg_id,
                        getattr(exc, "iterations_used", 0),
                        exc,
                    )
                else:
                    logger.warning(
                        "Agent loop failed for msg_id=%s, "
                        "falling back to one-shot: %s",
                        msg_id,
                        exc,
                    )
                # Fall through to one-shot path

        # Fallback: one-shot LLM translation (v1.2-mvp behavior)
        return await self._handle_run_oneshot(msg_id, parsed, client)

    async def _handle_run_agent_loop(
        self,
        msg_id: str,
        parsed: dict[str, Any],
        client: ClientConnection,
    ) -> MessageEnvelope:
        """Execute a run command via the iterative agent loop.

        The agent loop replaces the one-shot LLM translation path for
        natural language commands. The LLM receives a ToolRegistry of
        tools it can call iteratively (think-act cycles), observe results
        including failures, and propose corrections.

        Flow:
        1. Build the ToolRegistry with all 10 tools
        2. Create IPC callback bridges (confirm, ask, notify)
        3. Create the LLM adapter and agent loop
        4. Run the agent loop with the user's NL input
        5. Interpret the result and return a response envelope

        Every SSH command requires explicit human approval via the
        propose_ssh_command tool. execute_ssh can only run commands
        previously approved in the same loop session.

        Args:
            msg_id: Correlation ID from the original request.
            parsed: Validated payload with target_host, target_user,
                natural_language, and optional target_port.
            client: Connection context for IPC callbacks.

        Returns:
            RESPONSE envelope with the agent loop result.

        Raises:
            Exception: If the agent loop cannot be initialized (triggers
                fallback to one-shot in the caller).
        """
        from jules_daemon.agent.agent_loop import (
            AgentLoop,
            AgentLoopConfig,
            AgentLoopState,
        )
        from jules_daemon.agent.ipc_bridge import (
            make_ask_callback,
            make_confirm_callback,
            make_notify_callback,
        )
        from jules_daemon.agent.llm_adapter import OpenAILLMAdapter
        from jules_daemon.agent.tool_registry import ToolRegistry
        from jules_daemon.agent.tools.registry_factory import build_tool_set

        natural_language = parsed.get("natural_language", "")
        target_host = parsed.get("target_host", "")
        target_user = parsed.get("target_user", "")
        target_port = parsed.get("target_port", 22)

        logger.info(
            "Starting agent loop for msg_id=%s: '%s' on %s@%s:%d",
            msg_id,
            natural_language[:80],
            target_user,
            target_host,
            target_port,
        )

        # Build IPC callback bridges
        confirm_cb = make_confirm_callback(client)
        ask_cb = make_ask_callback(client)
        notify_cb = make_notify_callback(client)

        # Build the tool registry with all 10 tools
        llm_model = (
            self._config.llm_config.default_model
            if self._config.llm_config is not None
            else None
        )

        tools = build_tool_set(
            wiki_root=self._config.wiki_root,
            confirm_callback=confirm_cb,
            ask_callback=ask_cb,
            notify_callback=notify_cb,
            llm_client=self._config.llm_client,
            llm_model=llm_model,
        )

        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)

        logger.debug(
            "Agent tool registry initialized: %s",
            registry.list_tool_names(),
        )

        # Build the LLM adapter
        tool_schemas = registry.to_openai_schemas()
        llm_adapter = OpenAILLMAdapter(
            client=self._config.llm_client,
            model=self._config.llm_config.default_model,
            tool_schemas=tool_schemas,
        )

        # Build the system prompt with host context
        system_prompt = self._build_agent_system_prompt(
            target_host=target_host,
            target_user=target_user,
            target_port=target_port,
        )

        # Wrap registry in a dispatcher adapter that satisfies the
        # ToolDispatcher protocol (dispatch method delegates to execute).
        dispatcher = _RegistryDispatcher(registry)

        # Create and run the agent loop
        loop_config = AgentLoopConfig(
            max_iterations=self._config.max_agent_iterations,
        )
        agent_loop = AgentLoop(
            llm_client=llm_adapter,
            tool_dispatcher=dispatcher,
            system_prompt=system_prompt,
            config=loop_config,
        )

        result = await agent_loop.run(natural_language)

        logger.info(
            "Agent loop finished for msg_id=%s: state=%s, "
            "iterations=%d, error=%s, retry_exhausted=%s",
            msg_id,
            result.final_state.value,
            result.iterations_used,
            result.error_message,
            result.retry_exhausted,
        )

        # Emit notification event for the agent loop result. This is
        # fire-and-forget: notification delivery failures never affect
        # the response to the requesting client.
        try:
            await emit_agent_loop_completion(
                broadcaster=self._config.notification_broadcaster,
                loop_result=result,
                natural_language_command=natural_language,
                run_id=msg_id,
            )
        except Exception as notify_exc:
            logger.warning(
                "Failed to emit agent loop notification for msg_id=%s: %s",
                msg_id,
                notify_exc,
            )

        # If retries were exhausted (transient errors persisted through
        # all retry attempts), raise RetryExhaustedError to signal the
        # caller (_handle_run) to fall back to the one-shot path.
        if result.retry_exhausted:
            from jules_daemon.agent.error_classification import (
                RetryExhaustedError,
            )

            raise RetryExhaustedError(
                result.error_message or "Transient error retries exhausted",
                iterations_used=result.iterations_used,
            )

        # Translate the agent loop result into a response envelope
        if result.final_state is AgentLoopState.COMPLETE:
            return _build_success_response(
                msg_id=msg_id,
                verb="run",
                extra={
                    "status": "completed",
                    "mode": "agent_loop",
                    "iterations_used": result.iterations_used,
                    "message": (
                        f"Agent loop completed in {result.iterations_used} "
                        f"iteration(s)."
                    ),
                },
            )
        elif result.final_state is AgentLoopState.ERROR:
            error_msg = result.error_message or "Agent loop terminated"

            # Check if this was a user denial (should return denied status)
            if "denied" in error_msg.lower():
                return _build_success_response(
                    msg_id=msg_id,
                    verb="run",
                    extra={
                        "status": "denied",
                        "mode": "agent_loop",
                        "iterations_used": result.iterations_used,
                        "message": error_msg,
                    },
                )

            return _build_success_response(
                msg_id=msg_id,
                verb="run",
                extra={
                    "status": "agent_error",
                    "mode": "agent_loop",
                    "iterations_used": result.iterations_used,
                    "error": error_msg,
                    "message": error_msg,
                },
            )
        else:
            # Unexpected terminal state -- should not happen
            return _build_error_response(
                msg_id=msg_id,
                error_summary=(
                    f"Agent loop ended in unexpected state: "
                    f"{result.final_state.value}"
                ),
                validation_errors=[],
            )

    def _build_agent_system_prompt(
        self,
        *,
        target_host: str,
        target_user: str,
        target_port: int,
    ) -> str:
        """Build the system prompt for the agent loop.

        Includes:
        - Role definition and behavioral constraints
        - SSH target context (host, user, port)
        - Wiki translation history (if available)
        - Test catalog awareness
        - Security constraints (approval requirements)

        Args:
            target_host: Remote hostname or IP address.
            target_user: SSH username.
            target_port: SSH port number.

        Returns:
            The system prompt string.
        """
        wiki_context_lines = self._build_wiki_context(
            target_host=target_host,
        )
        wiki_block = "\n".join(wiki_context_lines) if wiki_context_lines else ""

        prompt_parts: list[str] = [
            "You are Jules, an SSH test runner assistant. You help users "
            "run tests on remote servers by translating natural language "
            "requests into SSH commands.",
            "",
            "## SSH Target",
            f"- Host: {target_host}",
            f"- User: {target_user}",
            f"- Port: {target_port}",
            "",
            "## Rules",
            "1. ALWAYS use propose_ssh_command to propose a command before "
            "executing it. NEVER skip the approval step.",
            "2. After getting approval, use execute_ssh with the approval_id "
            "to run the command.",
            "3. If a command fails, analyze the error and propose a corrected "
            "command. Do NOT repeat the same failing command.",
            "4. If required arguments are missing from a test specification, "
            "use ask_user_question to ask the user. NEVER guess or "
            "auto-default missing required arguments.",
            "5. Use read_wiki and lookup_test_spec to find information "
            "about available tests and their configurations.",
            "6. Use check_remote_processes before proposing a command if "
            "you suspect other tests may be running.",
            "7. When done, stop calling tools to signal completion.",
            "",
            "## Available Information",
            "- Wiki pages contain test documentation and specifications.",
            "- Test specs define command templates and required arguments.",
            "- Past command history helps with command format.",
        ]

        if wiki_block:
            prompt_parts.extend([
                "",
                "## Past Commands on This Host",
                wiki_block,
            ])

        return "\n".join(prompt_parts)

    async def _handle_run_oneshot(
        self,
        msg_id: str,
        parsed: dict[str, Any],
        client: ClientConnection,
        *,
        detection: DirectCommandDetection | None = None,
    ) -> MessageEnvelope:
        """Execute a run command with the v1.2-mvp one-shot path.

        One-shot confirmation, collision check, and background execution.

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
            detection: Optional pre-computed DirectCommandDetection from
                the caller (_handle_run). When provided, the one-shot path
                reuses the result instead of re-running detection. This
                avoids double-detection and keeps the audit chain
                consistent with the routing decision.

        Returns:
            RESPONSE envelope with started status or denial.
        """
        target_host = parsed.get("target_host", "")
        target_user = parsed.get("target_user", "")
        natural_language = parsed.get("natural_language", "")
        target_port = parsed.get("target_port", 22)

        # Pre-generate the run_id so it can be referenced from the audit
        # record before the background task is spawned. The _current_run_id
        # is only assigned after approval; audit records need the link
        # from the very first stage (NL input).
        provisional_run_id = f"run-{uuid.uuid4().hex[:12]}"

        # Build the NL input audit record and seed the full-chain
        # AuditRecord. Failures in audit construction must never crash
        # the run -- we fall back to ``audit = None`` and continue.
        audit: AuditRecord | None = None
        try:
            nl_input = build_nl_input_record(
                raw_input=natural_language,
                source="ipc",
            )
            audit = create_initial_audit(
                run_id=provisional_run_id,
                nl_input=nl_input,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to build NL input audit record for msg_id=%s: %s",
                msg_id,
                exc,
            )
            audit = None

        # Determine the proposed command: if the input is a direct shell
        # command (starts with a known executable), use it as-is.
        # Otherwise, attempt LLM translation via the Dataiku Mesh endpoint.
        #
        # When a pre-computed detection is passed from _handle_run(), we
        # reuse it directly instead of re-running the classifier. This
        # preserves the single-detection-per-request invariant and avoids
        # redundant work on the bypass path.
        if detection is None:
            detection = detect_direct_command(natural_language)

        direct_command = detection.is_direct_command
        if direct_command:
            proposed_command = natural_language
            logger.debug(
                "Input recognized as direct command (executable=%s), "
                "skipping LLM: %s",
                detection.executable,
                natural_language[:80],
            )
        else:
            proposed_command = await self._translate_via_llm(
                natural_language=natural_language,
                target_host=target_host,
                target_user=target_user,
                target_port=target_port,
            )

        # Append the parsed-command stage to the audit chain. Empty
        # proposed commands (which would violate the record's own
        # validation) are skipped -- audit remains partial rather
        # than crashing the run.
        if audit is not None and proposed_command.strip():
            try:
                parsed_record = build_parsed_command_record(
                    natural_language=natural_language,
                    resolved_shell=proposed_command,
                    is_direct_command=direct_command,
                    model_id=(
                        self._config.llm_config.default_model
                        if self._config.llm_config is not None
                        else None
                    ),
                )
                audit = audit.with_parsed_command(parsed_record)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to build parsed command audit record: %s", exc,
                )

        # Step 0: Check if a run is already active -- ask user to queue or cancel
        if (
            self._current_task is not None
            and not self._current_task.done()
        ):
            # Notify user that a run is active
            busy_msg = MessageEnvelope(
                msg_type=MessageType.STREAM,
                msg_id=f"busy-{uuid.uuid4().hex[:12]}",
                timestamp=_now_iso(),
                payload={
                    "line": (
                        f"\nA test is already running (run_id={self._current_run_id}).\n"
                        f"Command: {proposed_command}\n"
                        f"Target: {target_user}@{target_host}\n"
                    ),
                    "is_end": False,
                },
            )
            try:
                await self._send_envelope(client, busy_msg)
            except Exception:
                pass

            # Ask: queue it or cancel?
            queue_prompt = MessageEnvelope(
                msg_type=MessageType.CONFIRM_PROMPT,
                msg_id=f"queue-confirm-{uuid.uuid4().hex[:12]}",
                timestamp=_now_iso(),
                payload={
                    "proposed_command": proposed_command,
                    "target_host": target_host,
                    "message": "Queue this command to run after the current test finishes?",
                },
            )
            try:
                await self._send_envelope(client, queue_prompt)
            except Exception:
                return _build_error_response(
                    msg_id=msg_id,
                    error_summary="Failed to send queue prompt",
                    validation_errors=[],
                )

            try:
                queue_reply = await self._read_envelope(client, timeout=120.0)
            except (asyncio.TimeoutError, Exception):
                return _build_error_response(
                    msg_id=msg_id,
                    error_summary="Queue confirmation timed out",
                    validation_errors=[],
                )

            if queue_reply is None:
                return _build_error_response(
                    msg_id=msg_id,
                    error_summary="CLI disconnected during queue prompt",
                    validation_errors=[],
                )

            queue_approved = queue_reply.payload.get("approved", False)
            if not queue_approved:
                logger.info("User chose not to queue command for msg_id=%s", msg_id)
                return _build_success_response(
                    msg_id=msg_id,
                    verb="run",
                    extra={
                        "status": "cancelled",
                        "message": "Command not queued. No action taken.",
                    },
                )

            # User approved queuing
            queued = self._queue.enqueue(
                natural_language=proposed_command,
                ssh_host=target_host,
                ssh_user=target_user,
                ssh_port=target_port,
            )
            position = self._queue.size()
            logger.info(
                "User queued command (queue_id=%s, position=%d): %s",
                queued.queue_id,
                position,
                proposed_command[:80],
            )
            return _build_success_response(
                msg_id=msg_id,
                verb="run",
                extra={
                    "status": "queued",
                    "queue_id": queued.queue_id,
                    "position": position,
                    "current_run_id": self._current_run_id,
                    "message": (
                        f"Command queued at position {position}. "
                        f"It will start automatically when the current run finishes."
                    ),
                },
            )

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
            # Append the denial to the audit chain and persist a
            # partial record. Denials are audit-worthy events even
            # though no SSH execution takes place.
            if audit is not None:
                try:
                    denied_confirmation = build_confirmation_record(
                        original_command=(
                            proposed_command if proposed_command.strip()
                            else natural_language
                        ),
                        final_command="",
                        approved=False,
                        edited=False,
                    )
                    audit = audit.with_confirmation(denied_confirmation)
                    await safe_write_audit_async(
                        wiki_root=self._config.wiki_root,
                        record=audit,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Failed to persist denied audit for msg_id=%s: %s",
                        msg_id,
                        exc,
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

        # If the user edited the command, use the edited version
        edited_command = reply.payload.get("edited_command")
        command_was_edited = False
        if edited_command:
            logger.info(
                "User edited command for msg_id=%s: '%s' -> '%s'",
                msg_id,
                proposed_command[:80],
                edited_command[:80],
            )
            command_was_edited = True
            proposed_command = edited_command

        # Append the confirmation stage to the audit chain now that
        # we know the user approved (possibly with edits).
        if audit is not None and proposed_command.strip():
            try:
                approved_confirmation = build_confirmation_record(
                    original_command=(
                        audit.parsed_command.resolved_shell
                        if audit.parsed_command is not None
                        else proposed_command
                    ),
                    final_command=proposed_command,
                    approved=True,
                    edited=command_was_edited,
                )
                audit = audit.with_confirmation(approved_confirmation)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to build approved confirmation audit: %s", exc,
                )

        logger.info(
            "Run approved for msg_id=%s: '%s' on %s@%s:%d",
            msg_id,
            proposed_command[:80],
            target_user,
            target_host,
            target_port,
        )

        # Save the approved translation to the wiki so future NL requests
        # on this host can use it as context. Only save NL-translated commands
        # (skip direct commands since there's nothing to learn there).
        if not direct_command:
            try:
                from jules_daemon.wiki.command_translation import (
                    CommandTranslation,
                    TranslationOutcome,
                    save as save_translation,
                )
                was_edited = proposed_command != natural_language
                translation = CommandTranslation(
                    natural_language=natural_language,
                    resolved_shell=proposed_command,
                    ssh_host=target_host,
                    outcome=(
                        TranslationOutcome.EDITED if was_edited
                        else TranslationOutcome.APPROVED
                    ),
                    model_id=(
                        self._config.llm_config.default_model
                        if self._config.llm_config else "direct"
                    ),
                )
                save_translation(self._config.wiki_root, translation)
                logger.debug(
                    "Saved translation to wiki: '%s' -> '%s' on %s",
                    natural_language[:50],
                    proposed_command[:80],
                    target_host,
                )
            except Exception as exc:
                logger.warning("Failed to save translation to wiki: %s", exc)

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
                # Replace the prior APPROVED confirmation with a
                # DENIED one -- the user approved the command but
                # then backed out after seeing the collision warning.
                if audit is not None:
                    try:
                        denied_confirmation = build_confirmation_record(
                            original_command=(
                                audit.parsed_command.resolved_shell
                                if audit.parsed_command is not None
                                else proposed_command
                            ),
                            final_command="",
                            approved=False,
                            edited=False,
                        )
                        audit = audit.with_confirmation(denied_confirmation)
                        await safe_write_audit_async(
                            wiki_root=self._config.wiki_root,
                            record=audit,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning(
                            "Failed to persist collision-denied audit: %s",
                            exc,
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

        # Step 5: Reuse the provisional run_id generated at the start of
        # the handler so the audit record (seeded in the NL input stage)
        # stays correlated with the actual background run.
        run_id = provisional_run_id

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

        # Clear the previous completed run -- a new run is starting
        self._last_completed_run = None

        # Spawn execute_run as a background task. The audit record
        # carries the NL/parsed/confirmation stages forward into
        # the background task, which will append the SSH execution
        # and structured result stages and persist the full chain.
        self._current_run_id = run_id
        self._current_task = asyncio.create_task(
            self._background_execute(
                target_host=target_host,
                target_user=target_user,
                command=proposed_command,
                target_port=target_port,
                audit=audit,
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
        audit: AuditRecord | None = None,
    ) -> RunResult:
        """Run execute_run in the background, handling exceptions gracefully.

        On failure, writes FAILED state to the wiki so the status command
        can report it. Never allows exceptions to crash the daemon.

        After execution completes (success or failure), checks the queue
        and auto-starts the next queued command if one is available.

        If an ``audit`` record is supplied, the SSH execution and
        structured result stages are appended and the full chain is
        persisted to ``pages/daemon/audit``. Audit persistence failures
        are logged but never block execution.

        Args:
            target_host: Remote hostname or IP address.
            target_user: SSH username.
            command: Shell command string to execute.
            target_port: SSH port number.
            audit: Optional partial :class:`AuditRecord` carrying the
                NL input, parsed command, and confirmation stages.
                When provided, the SSH execution and structured
                result stages are appended after ``execute_run``
                completes (regardless of success or failure) and
                the full chain is written to the wiki.

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
            # Set _last_completed_run IMMEDIATELY after execute_run
            # returns to avoid a race window where status sees no active
            # run state. The audit/summary work below can take a few
            # hundred ms (LLM calls), so users typing 'status' during
            # that window would otherwise see "[ok]" instead of [COMPLETED].
            self._last_completed_run = result
            logger.info(
                "Background run completed: run_id=%s success=%s",
                result.run_id,
                result.success,
            )

            # If the run failed, push an error notification into the
            # output queue so any active watchers see it. Also store
            # the failure summary so the next CLI session can display it.
            if not result.success:
                error_summary = (
                    f"\n!!! RUN FAILED !!!\n"
                    f"Run ID: {result.run_id}\n"
                    f"Command: {command}\n"
                    f"Exit code: {result.exit_code}\n"
                )
                if result.error:
                    error_summary += f"Error: {result.error}\n"
                if result.stderr:
                    error_summary += f"Stderr:\n{result.stderr[:1000]}\n"
                try:
                    self._output_queue.put_nowait(error_summary)
                except asyncio.QueueFull:
                    pass
                # Persist the last failure so the CLI sees it on reconnect
                self._last_failure = error_summary
                logger.warning(
                    "Run %s failed with exit_code=%s",
                    result.run_id,
                    result.exit_code,
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

        # Finalize the full-chain audit record with the SSH execution
        # and structured result stages, then persist to the wiki.
        # All exceptions in this block are swallowed so audit failures
        # can never affect the run outcome or crash the daemon.
        if audit is not None:
            try:
                await self._finalize_and_write_audit(
                    audit=audit,
                    result=result,
                    command=command,
                    target_host=target_host,
                    target_user=target_user,
                    target_port=target_port,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to finalize audit for run_id=%s: %s",
                    result.run_id,
                    exc,
                )

        # Store the last completed run so `status` can report it even
        # after promote_run() clears the wiki current-run state.
        self._last_completed_run = result
        logger.info(
            "_background_execute: stored _last_completed_run "
            "(run_id=%s, success=%s, exit_code=%s)",
            result.run_id,
            result.success,
            result.exit_code,
        )

        # Emit completion notification to any subscribed CLI clients.
        # Fire-and-forget: notification delivery failures never affect
        # the run outcome or the watch stream.
        try:
            await emit_run_completion(
                broadcaster=self._config.notification_broadcaster,
                run_result=result,
                natural_language_command=command,
            )
        except Exception as notify_exc:
            logger.warning(
                "Failed to emit run completion notification for "
                "run_id=%s: %s",
                result.run_id,
                notify_exc,
            )

        # Signal end-of-stream to any watch subscribers
        try:
            self._output_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        # Check the queue for the next command to auto-execute
        self._try_start_next_queued()

        return result

    async def _finalize_and_write_audit(
        self,
        *,
        audit: AuditRecord,
        result: RunResult,
        command: str,
        target_host: str,
        target_user: str,
        target_port: int,
    ) -> None:
        """Append the SSH execution and result stages and persist the audit.

        Builds the :class:`SSHExecutionRecord` from the run result's
        connection details and the :class:`StructuredResultRecord` from
        the success flag, exit code, and any error message. The caller's
        ``audit`` argument is treated as immutable -- each stage
        transition returns a new record. The final full-chain record
        is aligned to ``result.run_id`` so the audit file correlates
        with the wiki history entry.

        The structured summary is produced by
        :func:`jules_daemon.execution.output_summarizer.summarize_output`,
        which yields a high-level description (counts, narrative, key
        failures) instead of the raw stdout/stderr. A bounded tail of
        the raw output is still embedded in the summary text (success
        runs) or in ``error_message`` (failed runs) so the audit file
        retains enough context for debugging. Passwords are never
        referenced in this path -- only the host, user, port, and
        command strings are consumed.

        Args:
            audit: The partial audit record carrying NL input, parsed
                command, and confirmation stages.
            result: The final :class:`RunResult` from ``execute_run``.
            command: The final shell command that was dispatched.
            target_host: Remote hostname or IP address.
            target_user: SSH username used to connect.
            target_port: SSH port number.
        """
        from dataclasses import replace as dataclass_replace

        # Align the audit record's run_id with the wiki history so
        # downstream queries can correlate by a single identifier.
        if result.run_id and result.run_id != audit.run_id:
            audit = dataclass_replace(audit, run_id=result.run_id)

        ssh_record = build_ssh_execution_record(
            host=target_host or "unknown",
            user=target_user or "unknown",
            port=target_port,
            command=command or "(empty)",
            session_id=result.run_id or audit.run_id or "unknown",
            started_at=result.started_at,
            completed_at=result.completed_at,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
        )
        audit = audit.with_ssh_execution(ssh_record)

        # Load any prior knowledge for this test from the wiki and
        # convert it into a prompt context block. The wiki I/O is
        # wrapped in try/except so a corrupt or missing file cannot
        # break the audit flow.
        test_slug, existing_knowledge = self._safe_load_test_knowledge(
            command=command,
        )
        wiki_context = ""
        if existing_knowledge is not None:
            try:
                wiki_context = existing_knowledge.to_prompt_context()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to format prior knowledge for slug=%s: %s",
                    test_slug,
                    exc,
                )
                wiki_context = ""

        # Produce a high-level summary of the command output. The
        # summarizer tries regex parsers first (pytest, unittest, jest,
        # generic iteration loops) and then optionally asks the LLM for
        # a narrative description. Any LLM error is swallowed inside
        # the summarizer so this call never crashes the audit flow.
        # When prior knowledge exists, it is forwarded to the LLM as a
        # context block so the narrative can leverage past observations.
        summary_obj = await self._safe_summarize_output(
            result=result,
            wiki_context=wiki_context,
        )

        # After the summary is produced, ask the LLM to extract durable
        # observations about this test and persist them to the wiki so
        # subsequent runs benefit from the accumulated knowledge.
        await self._safe_extract_and_save_knowledge(
            command=command,
            result=result,
            test_slug=test_slug,
            existing_knowledge=existing_knowledge,
        )

        output_summary_text = _format_output_summary(summary_obj=summary_obj)

        # Bound the raw stdout/stderr tail we embed for debugging. The
        # summarizer already truncates to ~500 chars, but the audit
        # writer used to embed up to AUDIT_OUTPUT_LIMIT characters --
        # we keep a larger 2000-character tail for the audit file to
        # retain meaningful context without exploding file size.
        stdout_tail = truncate_text(result.stdout, _AUDIT_RAW_TAIL_LIMIT)
        stderr_tail = truncate_text(result.stderr, _AUDIT_RAW_TAIL_LIMIT)

        if result.success:
            summary_pieces: list[str] = [
                output_summary_text,
                f"Exit code: {result.exit_code}"
                if result.exit_code is not None
                else "Exit code: (none)",
            ]
            if stdout_tail:
                summary_pieces.append(f"=== stdout (tail) ===\n{stdout_tail}")
            elif stderr_tail:
                summary_pieces.append(f"=== stderr (tail) ===\n{stderr_tail}")
            summary = "\n\n".join(piece for piece in summary_pieces if piece)
            error_message: str | None = None
        else:
            exit_fragment = (
                f"exit code {result.exit_code}"
                if result.exit_code is not None
                else "no exit code (connection failure)"
            )
            summary_pieces = [
                output_summary_text,
                f"Command failed ({exit_fragment})",
            ]
            summary = "\n\n".join(piece for piece in summary_pieces if piece)
            error_parts: list[str] = []
            if result.error:
                error_parts.append(result.error)
            if stderr_tail:
                error_parts.append(f"=== stderr (tail) ===\n{stderr_tail}")
            elif stdout_tail:
                error_parts.append(f"=== stdout (tail) ===\n{stdout_tail}")
            error_message = "\n\n".join(error_parts) if error_parts else None

        structured = build_structured_result_record(
            success=result.success,
            exit_code=result.exit_code,
            summary=summary,
            error_message=error_message,
            tests_passed=summary_obj.passed,
            tests_failed=summary_obj.failed,
            tests_skipped=summary_obj.skipped,
            tests_total=summary_obj.total,
        )
        audit = audit.with_structured_result(structured)

        await safe_write_audit_async(
            wiki_root=self._config.wiki_root,
            record=audit,
        )

    async def _safe_summarize_output(
        self,
        *,
        result: RunResult,
        wiki_context: str = "",
    ) -> OutputSummary:
        """Call the output summarizer without crashing the audit flow.

        Any exception raised by the summarizer is caught and converted
        into an empty :class:`OutputSummary`. This guarantees that the
        audit record can always be written, even if the LLM client is
        misconfigured or the regex layer hits a pathological input.

        Args:
            result: The run result whose output should be summarized.
            wiki_context: Optional formatted prior knowledge for this
                test. Passed through to
                :func:`summarize_output` so the LLM can leverage past
                observations when crafting the narrative.
        """
        try:
            llm_model: str | None = None
            if self._config.llm_config is not None:
                llm_model = self._config.llm_config.default_model
            return await summarize_output(
                stdout=result.stdout,
                stderr=result.stderr,
                command=result.command,
                exit_code=result.exit_code,
                llm_client=self._config.llm_client,
                llm_model=llm_model,
                wiki_context=wiki_context,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Output summarizer failed for run_id=%s: %s",
                result.run_id,
                exc,
            )
            return OutputSummary(parser="none")

    def _safe_load_test_knowledge(
        self,
        *,
        command: str,
    ) -> tuple[str, TestKnowledge | None]:
        """Best-effort load of accumulated knowledge for *command*.

        Returns a ``(slug, knowledge)`` tuple. The slug is always
        derived (so the caller can save fresh knowledge later); the
        knowledge value is ``None`` when no prior file exists or the
        load failed. Wiki I/O failures never raise from this method.
        """
        try:
            slug = derive_test_slug(command)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to derive test slug for command %r: %s",
                command,
                exc,
            )
            return ("unknown-test", None)
        try:
            existing = load_test_knowledge(self._config.wiki_root, slug)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to load test knowledge for slug=%s: %s",
                slug,
                exc,
            )
            existing = None
        return (slug, existing)

    async def _safe_extract_and_save_knowledge(
        self,
        *,
        command: str,
        result: RunResult,
        test_slug: str,
        existing_knowledge: TestKnowledge | None,
    ) -> None:
        """Best-effort knowledge extraction and persistence after a run.

        Calls :func:`extract_knowledge` to ask the LLM for fresh
        observations, merges them with the existing knowledge (if any),
        and saves the result to the wiki. Every step is wrapped in
        try/except so a wiki I/O or LLM failure cannot break the audit
        flow.
        """
        if not command or not command.strip():
            return
        llm_model: str | None = None
        if self._config.llm_config is not None:
            llm_model = self._config.llm_config.default_model
        try:
            new_observations = await extract_knowledge(
                command=command,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                existing_knowledge=existing_knowledge,
                llm_client=self._config.llm_client,
                llm_model=llm_model,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Knowledge extractor failed for slug=%s: %s",
                test_slug,
                exc,
            )
            return

        # When the LLM is disabled and there is no existing record,
        # there is nothing useful to persist -- skip the save entirely.
        if new_observations is None and existing_knowledge is None:
            return

        try:
            merged = merge_knowledge(
                existing_knowledge,
                new_observations or {},
                test_slug=test_slug,
                command_pattern=command,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to merge knowledge for slug=%s: %s",
                test_slug,
                exc,
            )
            return

        try:
            save_test_knowledge(self._config.wiki_root, merged)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to save test knowledge for slug=%s: %s",
                test_slug,
                exc,
            )

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

        # Clear the previous completed run -- a new run is starting
        self._last_completed_run = None

        run_id = f"run-{uuid.uuid4().hex[:12]}"
        self._current_run_id = run_id

        # Seed a partial audit record for the queued run. The user
        # already gave explicit consent when they queued the command,
        # so we record an APPROVED confirmation with a synthetic
        # approver identity so readers can distinguish queued auto-
        # starts from interactive confirmations.
        queued_audit: AuditRecord | None = None
        try:
            queued_audit = self._build_queued_audit(
                run_id=run_id,
                command=next_cmd.natural_language,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to build audit for queued run %s: %s",
                run_id,
                exc,
            )

        self._current_task = asyncio.create_task(
            self._background_execute(
                target_host=next_cmd.ssh_host or "",
                target_user=next_cmd.ssh_user or "",
                command=next_cmd.natural_language,
                target_port=next_cmd.ssh_port,
                audit=queued_audit,
            ),
            name=f"run-{run_id}",
        )

    def _build_queued_audit(
        self,
        *,
        run_id: str,
        command: str,
    ) -> AuditRecord | None:
        """Construct a partial audit record for an auto-started queued run.

        Queued runs bypass the interactive confirmation loop because
        the user already agreed to execute them when they opted into
        queueing. To keep audit coverage uniform, we still record:
        NL input, a parsed-command entry (treated as a direct command),
        and an APPROVED confirmation attributed to ``"queue-autostart"``.

        Returns ``None`` on any error so the queued run is not blocked.
        """
        if not command.strip():
            return None

        nl_record = build_nl_input_record(
            raw_input=command,
            source="queue",
        )
        audit = create_initial_audit(run_id=run_id, nl_input=nl_record)
        parsed = build_parsed_command_record(
            natural_language=command,
            resolved_shell=command,
            is_direct_command=True,
            model_id=None,
        )
        audit = audit.with_parsed_command(parsed)
        confirmation = build_confirmation_record(
            original_command=command,
            final_command=command,
            approved=True,
            edited=False,
            decided_by="queue-autostart",
        )
        audit = audit.with_confirmation(confirmation)
        return audit

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

        Priority order:
        1. Active background task -> show RUNNING with live wiki data
        2. Recently completed run in memory -> show final result
        3. Wiki current-run file -> show whatever state is there
        4. Otherwise -> idle
        """
        queue_depth = self._queue.size()

        # If we have a stored "last completed run", report it FIRST.
        # This takes priority even if the task technically hasn't returned
        # yet -- the task may still be writing audit files or extracting
        # knowledge after execute_run() returned. From the user's perspective
        # the run is done; only the bookkeeping is still in flight.
        if self._last_completed_run is not None:
            last = self._last_completed_run
            final_status = "COMPLETED" if last.success else "FAILED"
            duration = last.duration_seconds
            extra: dict[str, Any] = {
                "state": "completed",
                "run_id": last.run_id,
                "host": last.target_host,
                "command": last.command,
                "status": final_status,
                "exit_code": last.exit_code,
                "duration_seconds": round(duration, 2),
                "queue_depth": queue_depth,
            }
            if last.error:
                extra["error"] = last.error
            if last.stderr:
                extra["stderr"] = last.stderr[:2000]
            return _build_success_response(
                msg_id=msg_id,
                verb="status",
                extra=extra,
            )

        # Check the background task state
        task_running = (
            self._current_task is not None
            and not self._current_task.done()
        )

        logger.debug(
            "_handle_status: task_running=%s, _current_task=%s, "
            "_last_completed_run=%s, queue_depth=%d",
            task_running,
            "set" if self._current_task else "None",
            "set" if self._last_completed_run else "None",
            queue_depth,
        )

        # Legacy branch -- kept for safety but should never be reached
        # because the priority block above already handled it.
        if not task_running and self._last_completed_run is not None:
            last = self._last_completed_run
            final_status = "COMPLETED" if last.success else "FAILED"
            duration = last.duration_seconds
            extra: dict[str, Any] = {
                "state": "completed",
                "run_id": last.run_id,
                "host": last.target_host,
                "command": last.command,
                "status": final_status,
                "exit_code": last.exit_code,
                "duration_seconds": round(duration, 2),
                "queue_depth": queue_depth,
            }
            if last.error:
                extra["error"] = last.error
            if last.stderr:
                extra["stderr"] = last.stderr[:2000]
            return _build_success_response(
                msg_id=msg_id,
                verb="status",
                extra=extra,
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

        # Step 2: Poll for new lines until end sentinel (None) is received.
        # Don't rely on task.done() -- the queue may still have lines
        # after the task finishes. Wait for the explicit None sentinel.
        while True:
            try:
                line = await asyncio.wait_for(
                    self._output_queue.get(), timeout=5.0,
                )
            except asyncio.TimeoutError:
                # No new output for 5 seconds -- check if task is still alive
                if (
                    self._current_task is None
                    or self._current_task.done()
                ):
                    # Task finished and queue is drained -- exit
                    break
                # Task still running, just no output yet -- keep waiting
                continue

            if line is None:
                # End sentinel from background task -- all output received
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
