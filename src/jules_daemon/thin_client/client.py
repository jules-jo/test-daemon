"""Minimal example thin client for the Jules daemon.

A standalone client that connects to the daemon via the same IPC
interface as the primary CLI, proving that any process can participate
in the protocol. Exercises all core commands:

    - health  (handshake / liveness check)
    - status  (current run state)
    - history (past run results)
    - cancel  (stop the current or a queued run)
    - run     (submit a new test execution)
    - watch   (subscribe to streaming output)
    - confirm (approve or deny SSH commands)

Design decisions:

    1. **Separate from primary CLI**: No imports from ``jules_daemon.cli``
       beyond shared IPC infrastructure. Proves the protocol is the
       contract, not the CLI implementation.

    2. **No classifier**: Commands are built directly from typed parameters.
       No NL processing, no structured parser -- just verb-level envelope
       construction.

    3. **Confirmation flow support**: The client can handle CONFIRM_PROMPT
       envelopes from the daemon and send CONFIRM_REPLY responses,
       completing the security approval cycle.

    4. **Streaming support**: The client can subscribe to a watch stream,
       receive STREAM envelopes, and render output lines.

    5. **Immutable configuration**: ``ThinClientConfig`` is a frozen
       dataclass following the project convention.

Usage::

    from jules_daemon.thin_client.client import ThinClient, ThinClientConfig

    config = ThinClientConfig(socket_path="/run/jules/daemon.sock")
    client = ThinClient(config=config)

    # One-shot status check
    result = await client.status()

    # Full run with confirmation handling
    result = await client.run(
        target_host="ci-server.example.com",
        target_user="deploy",
        natural_language="run the unit tests for the auth module",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from jules_daemon.ipc.client_connection import (
    ClientConnection,
    ConnectionConfig,
)
from jules_daemon.ipc.framing import MessageEnvelope, MessageType
from jules_daemon.thin_client.commands import (
    SSHTargetParams,
    build_cancel_request,
    build_confirm_reply,
    build_discover_request,
    build_health_request,
    build_history_request,
    build_run_request,
    build_status_request,
    build_watch_request,
)
from jules_daemon.thin_client.renderer import render_response

__all__ = [
    "CommandResult",
    "ThinClient",
    "ThinClientConfig",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RECEIVE_TIMEOUT: float = 7200.0  # 2 hours: must outlast the longest test run
"""Maximum seconds to wait for a daemon response."""

_STREAM_READ_TIMEOUT: float = 10.0
"""Maximum seconds between stream messages before treating as lost."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThinClientConfig:
    """Immutable configuration for the thin client.

    Attributes:
        socket_path: Path to the daemon's Unix domain socket.
            When None, auto-discovers via the standard search order.
        connect_timeout: Maximum seconds to wait for connection.
        receive_timeout: Maximum seconds to wait for each response.
        stream_timeout: Maximum seconds between stream messages.
    """

    socket_path: Path | None = None
    connect_timeout: float = 5.0
    receive_timeout: float = _DEFAULT_RECEIVE_TIMEOUT
    stream_timeout: float = _STREAM_READ_TIMEOUT

    def __post_init__(self) -> None:
        if self.connect_timeout <= 0:
            raise ValueError(
                f"connect_timeout must be positive, got {self.connect_timeout}"
            )
        if self.receive_timeout <= 0:
            raise ValueError(
                f"receive_timeout must be positive, got {self.receive_timeout}"
            )
        if self.stream_timeout <= 0:
            raise ValueError(
                f"stream_timeout must be positive, got {self.stream_timeout}"
            )


# ---------------------------------------------------------------------------
# Command result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    """Immutable result of a thin client command.

    Attributes:
        success: True if the daemon returned a successful response.
        verb: The verb that was executed.
        response: The raw daemon response envelope. None if the
            command failed before receiving a response (e.g., timeout).
        rendered: Human-readable rendering of the response.
        error: Error description on failure. None on success.
    """

    success: bool
    verb: str
    response: MessageEnvelope | None
    rendered: str
    error: str | None


# ---------------------------------------------------------------------------
# Confirm callback type
# ---------------------------------------------------------------------------

ConfirmCallback = Callable[[MessageEnvelope], "tuple[bool, str | None]"]
"""Callback invoked when the daemon sends a CONFIRM_PROMPT.

Receives the prompt envelope and returns (approved, edited_command).
If approved is True and edited_command is None, the original is used.
If approved is True and edited_command is provided, that replaces the
original. If approved is False, the command is denied.
"""


def _default_confirm_callback(envelope: MessageEnvelope) -> tuple[bool, str | None]:
    """Default confirmation callback -- always denies for safety.

    In a real integration, this would prompt the user. The thin client
    defaults to deny to satisfy the security-first constraint.

    Args:
        envelope: The CONFIRM_PROMPT envelope from the daemon.

    Returns:
        (False, None) -- always deny.
    """
    logger.warning(
        "Default confirm callback: denying command (msg_id=%s)",
        envelope.msg_id,
    )
    return False, None


# ---------------------------------------------------------------------------
# Thin client
# ---------------------------------------------------------------------------


class ThinClient:
    """Minimal IPC client that exercises all daemon verbs.

    Connects to the daemon via the shared Unix domain socket protocol,
    sends verb-specific REQUEST envelopes, and processes responses.
    Handles the full confirmation flow (CONFIRM_PROMPT -> CONFIRM_REPLY)
    and streaming output (STREAM envelopes for the watch verb).

    The client manages one connection at a time. Each command method
    opens a connection, performs the exchange, and closes the connection.
    For streaming operations (watch), the connection remains open until
    the stream ends.

    Args:
        config: Client configuration (socket path, timeouts).
        on_confirm: Callback for SSH command approval prompts.
            Defaults to always-deny for security.
    """

    def __init__(
        self,
        *,
        config: ThinClientConfig | None = None,
        on_confirm: ConfirmCallback | None = None,
    ) -> None:
        self._config = config or ThinClientConfig()
        self._on_confirm = on_confirm or _default_confirm_callback

    @property
    def config(self) -> ThinClientConfig:
        """The client configuration."""
        return self._config

    # -- One-shot command methods -------------------------------------------

    async def health(self) -> CommandResult:
        """Check daemon liveness.

        Sends a health-check request and waits for the response.
        This is the simplest possible daemon interaction.

        Returns:
            CommandResult with the daemon's health response.
        """
        envelope = build_health_request()
        return await self._execute_command("health", envelope)

    async def status(self, *, verbose: bool = False) -> CommandResult:
        """Query the current run state.

        Args:
            verbose: When True, request extended details.

        Returns:
            CommandResult with the daemon's status response.
        """
        envelope = build_status_request(verbose=verbose)
        return await self._execute_command("status", envelope)

    async def history(
        self,
        *,
        limit: int = 20,
        status_filter: str | None = None,
        host_filter: str | None = None,
    ) -> CommandResult:
        """Query past test run results.

        Args:
            limit: Maximum number of records to return.
            status_filter: Optional status filter.
            host_filter: Optional hostname filter.

        Returns:
            CommandResult with the daemon's history response.
        """
        try:
            envelope = build_history_request(
                limit=limit,
                status_filter=status_filter,
                host_filter=host_filter,
            )
        except ValueError as exc:
            return CommandResult(
                success=False,
                verb="history",
                response=None,
                rendered=f"Invalid parameters: {exc}",
                error=str(exc),
            )
        return await self._execute_command("history", envelope)

    async def cancel(
        self,
        *,
        run_id: str | None = None,
        force: bool = False,
        reason: str | None = None,
    ) -> CommandResult:
        """Cancel the current or a queued run.

        Args:
            run_id: Target a specific run. None cancels the current.
            force: When True, send SIGKILL instead of SIGTERM.
            reason: Optional cancellation reason.

        Returns:
            CommandResult with the daemon's cancel response.
        """
        envelope = build_cancel_request(
            run_id=run_id,
            force=force,
            reason=reason,
        )
        return await self._execute_command("cancel", envelope)

    async def run(
        self,
        *,
        natural_language: str,
        target_host: str | None = None,
        target_user: str | None = None,
        target_port: int = 22,
        key_path: str | None = None,
        system_name: str | None = None,
        infer_target: bool = False,
        interpret_request: bool = False,
    ) -> CommandResult:
        """Submit a new test execution.

        Sends a run request and handles the full confirmation flow:
        if the daemon sends a CONFIRM_PROMPT, the ``on_confirm``
        callback is invoked and the reply is sent back.

        Args:
            natural_language: Description of what tests to run.
            target_host: Remote hostname or IP.
            target_user: SSH username.
            target_port: SSH port. Default 22.
            key_path: Absolute path to SSH private key.
            system_name: Named system alias defined in the wiki.
            infer_target: Ask the daemon to infer a named system from
                the natural-language request using its live wiki.
            interpret_request: Ask the daemon to use its LLM-assisted
                interpretation fallback for unresolved conversational
                run requests.

        Returns:
            CommandResult with the daemon's run response (which may
            include the confirmation cycle results).
        """
        try:
            if system_name is not None:
                if (
                    target_host is not None
                    or target_user is not None
                    or target_port != 22
                    or key_path is not None
                    or infer_target
                    or interpret_request
                ):
                    raise ValueError(
                        "system_name cannot be combined with explicit target fields, infer_target, or interpret_request"
                    )
                envelope = build_run_request(
                    natural_language=natural_language,
                    system_name=system_name,
                )
            elif infer_target:
                if interpret_request:
                    raise ValueError(
                        "infer_target cannot be combined with interpret_request"
                    )
                if (
                    target_host is not None
                    or target_user is not None
                    or target_port != 22
                    or key_path is not None
                ):
                    raise ValueError(
                        "infer_target cannot be combined with explicit target fields"
                    )
                envelope = build_run_request(
                    natural_language=natural_language,
                    infer_target=True,
                )
            elif interpret_request:
                if (
                    target_host is not None
                    or target_user is not None
                    or target_port != 22
                    or key_path is not None
                ):
                    raise ValueError(
                        "interpret_request cannot be combined with explicit target fields"
                    )
                envelope = build_run_request(
                    natural_language=natural_language,
                    interpret_request=True,
                )
            else:
                target = SSHTargetParams(
                    host=target_host or "",
                    user=target_user or "",
                    port=target_port,
                    key_path=key_path,
                )
                envelope = build_run_request(
                    target=target,
                    natural_language=natural_language,
                )
        except ValueError as exc:
            return CommandResult(
                success=False,
                verb="run",
                response=None,
                rendered=f"Invalid parameters: {exc}",
                error=str(exc),
            )

        return await self._execute_with_confirmation("run", envelope)

    async def discover(
        self,
        *,
        target_host: str,
        target_user: str,
        command: str,
        target_port: int = 22,
    ) -> CommandResult:
        """Discover a test spec by running command -h on the remote host.

        Sends a discover request and handles the confirmation flow:
        the daemon sends back a draft spec as a CONFIRM_PROMPT, the
        user approves or denies, and the daemon writes the wiki file.

        Args:
            target_host: Remote hostname or IP.
            target_user: SSH username.
            command: The command to discover (without -h flag).
            target_port: SSH port. Default 22.

        Returns:
            CommandResult with the discovery outcome.
        """
        try:
            target = SSHTargetParams(
                host=target_host,
                user=target_user,
                port=target_port,
            )
            envelope = build_discover_request(
                target=target,
                command=command,
            )
        except ValueError as exc:
            return CommandResult(
                success=False,
                verb="discover",
                response=None,
                rendered=f"Invalid parameters: {exc}",
                error=str(exc),
            )

        return await self._execute_with_confirmation("discover", envelope)

    async def watch(
        self,
        *,
        run_id: str | None = None,
        tail_lines: int = 50,
        on_line: Callable[[str], None] | None = None,
        max_lines: int | None = None,
    ) -> CommandResult:
        """Subscribe to streaming output from a running test.

        Opens a connection, sends a watch request, and reads STREAM
        envelopes until the stream ends, an error occurs, or the
        optional ``max_lines`` limit is reached.

        Args:
            run_id: Target a specific run. None watches the current.
            tail_lines: Number of recent lines on initial attach.
            on_line: Optional callback invoked for each output line.
            max_lines: Stop after receiving this many lines. None
                for unlimited.

        Returns:
            CommandResult summarizing the watch session.
        """
        try:
            request = build_watch_request(
                run_id=run_id,
                tail_lines=tail_lines,
            )
        except ValueError as exc:
            return CommandResult(
                success=False,
                verb="watch",
                response=None,
                rendered=f"Invalid parameters: {exc}",
                error=str(exc),
            )

        return await self._execute_stream(
            request,
            on_line=on_line,
            max_lines=max_lines,
        )

    # -- Internal: single request-response ----------------------------------

    async def _execute_command(
        self,
        verb: str,
        request: MessageEnvelope,
    ) -> CommandResult:
        """Send a request and wait for one response.

        Opens a connection, sends the envelope, reads one response,
        and closes the connection.

        Args:
            verb: The verb name (for result metadata).
            request: The request envelope to send.

        Returns:
            CommandResult with the daemon's response.
        """
        conn = self._create_connection()

        try:
            result = await conn.connect()
        except Exception as exc:
            return CommandResult(
                success=False,
                verb=verb,
                response=None,
                rendered=f"Connection failed: {exc}",
                error=str(exc),
            )

        if not result.success:
            return CommandResult(
                success=False,
                verb=verb,
                response=None,
                rendered=f"Handshake failed: {result.error}",
                error=result.error,
            )

        # Print any pending failure notification from a prior background
        # run or crash recovery (delivered via the handshake response)
        if result.pending_failure:
            print(result.pending_failure, flush=True)

        try:
            await conn.send(request)
            response = await conn.receive(
                timeout=self._config.receive_timeout,
            )
        except Exception as exc:
            return CommandResult(
                success=False,
                verb=verb,
                response=None,
                rendered=f"Communication error: {exc}",
                error=str(exc),
            )
        finally:
            await conn.close()

        if response is None:
            return CommandResult(
                success=False,
                verb=verb,
                response=None,
                rendered="No response from daemon (timeout or disconnect)",
                error="No response received",
            )

        is_success = response.msg_type != MessageType.ERROR
        rendered = render_response(response)
        error_msg = (
            response.payload.get("error") if not is_success else None
        )

        return CommandResult(
            success=is_success,
            verb=verb,
            response=response,
            rendered=rendered,
            error=error_msg,
        )

    # -- Internal: request with confirmation --------------------------------

    async def _execute_with_confirmation(
        self,
        verb: str,
        request: MessageEnvelope,
    ) -> CommandResult:
        """Send a request that may trigger a confirmation prompt.

        After sending the request, reads responses in a loop:
        - CONFIRM_PROMPT: invokes the on_confirm callback, sends reply
        - RESPONSE: returns as success
        - ERROR: returns as failure
        - None: timeout or disconnect

        Args:
            verb: The verb name (for result metadata).
            request: The request envelope to send.

        Returns:
            CommandResult with the final daemon response.
        """
        conn = self._create_connection()

        try:
            handshake = await conn.connect()
        except Exception as exc:
            return CommandResult(
                success=False,
                verb=verb,
                response=None,
                rendered=f"Connection failed: {exc}",
                error=str(exc),
            )

        if not handshake.success:
            return CommandResult(
                success=False,
                verb=verb,
                response=None,
                rendered=f"Handshake failed: {handshake.error}",
                error=handshake.error,
            )

        # Print any pending failure notification from a prior background run
        if handshake.pending_failure:
            print(handshake.pending_failure, flush=True)

        try:
            await conn.send(request)
            return await self._confirmation_loop(conn, verb)
        except Exception as exc:
            return CommandResult(
                success=False,
                verb=verb,
                response=None,
                rendered=f"Communication error: {exc}",
                error=str(exc),
            )
        finally:
            await conn.close()

    async def _confirmation_loop(
        self,
        conn: ClientConnection,
        verb: str,
    ) -> CommandResult:
        """Read responses until a terminal message or timeout.

        Handles the CONFIRM_PROMPT -> CONFIRM_REPLY cycle inline.

        Args:
            conn: Active client connection.
            verb: The verb name (for result metadata).

        Returns:
            CommandResult with the final response.
        """
        while True:
            response = await conn.receive(
                timeout=self._config.receive_timeout,
            )

            if response is None:
                return CommandResult(
                    success=False,
                    verb=verb,
                    response=None,
                    rendered="No response from daemon (timeout or disconnect)",
                    error="No response received",
                )

            # Handle confirmation prompt
            if response.msg_type == MessageType.CONFIRM_PROMPT:
                approved, edited = self._on_confirm(response)
                reply = build_confirm_reply(
                    approved=approved,
                    original_msg_id=response.msg_id,
                    edited_command=edited,
                )
                await conn.send(reply)
                # Continue reading for the final response
                continue

            # Handle inline status/stream messages (print and continue)
            if response.msg_type == MessageType.STREAM:
                rendered_line = render_response(response)
                print(rendered_line, flush=True)
                continue

            # Terminal response
            is_success = response.msg_type != MessageType.ERROR
            rendered = render_response(response)
            error_msg = (
                response.payload.get("error") if not is_success else None
            )

            return CommandResult(
                success=is_success,
                verb=verb,
                response=response,
                rendered=rendered,
                error=error_msg,
            )

    # -- Internal: streaming ------------------------------------------------

    async def _execute_stream(
        self,
        request: MessageEnvelope,
        *,
        on_line: Callable[[str], None] | None = None,
        max_lines: int | None = None,
    ) -> CommandResult:
        """Send a watch request and consume the stream.

        Opens a connection, sends the watch envelope, reads the
        subscription response, then enters a streaming loop.

        Args:
            request: The watch request envelope.
            on_line: Optional callback for each output line.
            max_lines: Stop after this many lines.

        Returns:
            CommandResult summarizing the watch session.
        """
        conn = self._create_connection()

        try:
            handshake = await conn.connect()
        except Exception as exc:
            return CommandResult(
                success=False,
                verb="watch",
                response=None,
                rendered=f"Connection failed: {exc}",
                error=str(exc),
            )

        if not handshake.success:
            return CommandResult(
                success=False,
                verb="watch",
                response=None,
                rendered=f"Handshake failed: {handshake.error}",
                error=handshake.error,
            )

        try:
            await conn.send(request)

            # Read subscription response
            sub_response = await conn.receive(
                timeout=self._config.receive_timeout,
            )
            if sub_response is None:
                return CommandResult(
                    success=False,
                    verb="watch",
                    response=None,
                    rendered="No subscription response",
                    error="No subscription response received",
                )

            if sub_response.msg_type == MessageType.ERROR:
                return CommandResult(
                    success=False,
                    verb="watch",
                    response=sub_response,
                    rendered=render_response(sub_response),
                    error=sub_response.payload.get("error"),
                )

            # Stream loop
            lines_received = 0
            last_response = sub_response

            while True:
                envelope = await conn.receive(
                    timeout=self._config.stream_timeout,
                )

                if envelope is None:
                    break

                last_response = envelope

                if envelope.msg_type == MessageType.ERROR:
                    return CommandResult(
                        success=False,
                        verb="watch",
                        response=envelope,
                        rendered=render_response(envelope),
                        error=envelope.payload.get("error"),
                    )

                if envelope.msg_type == MessageType.STREAM:
                    is_end = envelope.payload.get("is_end", False)
                    if is_end:
                        break

                    line = envelope.payload.get("line", "")
                    lines_received += 1
                    if on_line is not None:
                        on_line(line)

                    if max_lines is not None and lines_received >= max_lines:
                        break

            return CommandResult(
                success=True,
                verb="watch",
                response=last_response,
                rendered=f"Watch complete: {lines_received} lines received",
                error=None,
            )

        except Exception as exc:
            return CommandResult(
                success=False,
                verb="watch",
                response=None,
                rendered=f"Stream error: {exc}",
                error=str(exc),
            )
        finally:
            await conn.close()

    # -- Internal: connection factory ---------------------------------------

    def _create_connection(self) -> ClientConnection:
        """Create a new ClientConnection from the current config.

        Returns:
            A fresh, unconnected ClientConnection instance.
        """
        return ClientConnection(
            config=ConnectionConfig(
                socket_path=self._config.socket_path,
                connect_timeout=self._config.connect_timeout,
            ),
        )
