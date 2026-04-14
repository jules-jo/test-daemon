"""Tests for the ThinClient class and its configuration.

Validates:
    - ThinClientConfig validation (timeouts must be positive)
    - ThinClient.health() performs handshake and health check
    - ThinClient.status() sends status requests
    - ThinClient.history() sends history requests with validation
    - ThinClient.cancel() sends cancel requests
    - ThinClient.run() handles the confirmation flow
    - ThinClient.watch() handles streaming output
    - Connection failures produce graceful error results
    - Timeout handling for all commands
    - Confirm callback integration for security approval flow
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.ipc.client_connection import (
    HANDSHAKE_VERB,
    PROTOCOL_VERSION,
    ConnectionConfig,
    HandshakeResult,
)
from jules_daemon.ipc.framing import (
    MessageEnvelope,
    MessageType,
    encode_frame,
)
from jules_daemon.thin_client.client import (
    CommandResult,
    ThinClient,
    ThinClientConfig,
    _default_confirm_callback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = "2026-04-09T12:00:00Z"


def _make_response(
    verb: str,
    status: str = "ok",
    extra: dict | None = None,
) -> MessageEnvelope:
    """Build a RESPONSE envelope."""
    payload = {"verb": verb, "status": status}
    if extra:
        payload.update(extra)
    return MessageEnvelope(
        msg_type=MessageType.RESPONSE,
        msg_id="daemon-resp-001",
        timestamp=_TS,
        payload=payload,
    )


def _make_error(
    verb: str,
    error: str,
    status_code: int = 500,
) -> MessageEnvelope:
    """Build an ERROR envelope."""
    return MessageEnvelope(
        msg_type=MessageType.ERROR,
        msg_id="daemon-err-001",
        timestamp=_TS,
        payload={
            "verb": verb,
            "error": error,
            "status_code": status_code,
        },
    )


def _make_confirm_prompt(
    command: str = "pytest -v",
    target_host: str = "ci.example.com",
) -> MessageEnvelope:
    """Build a CONFIRM_PROMPT envelope."""
    return MessageEnvelope(
        msg_type=MessageType.CONFIRM_PROMPT,
        msg_id="daemon-prompt-001",
        timestamp=_TS,
        payload={
            "verb": "confirm",
            "command": command,
            "target_host": target_host,
            "target_user": "deploy",
            "risk_level": "LOW",
        },
    )


def _make_stream(
    line: str,
    sequence: int = 0,
    is_end: bool = False,
) -> MessageEnvelope:
    """Build a STREAM envelope."""
    return MessageEnvelope(
        msg_type=MessageType.STREAM,
        msg_id=f"daemon-stream-{sequence}",
        timestamp=_TS,
        payload={
            "line": line,
            "sequence": sequence,
            "is_end": is_end,
        },
    )


# ---------------------------------------------------------------------------
# ThinClientConfig
# ---------------------------------------------------------------------------


class TestThinClientConfig:
    """Tests for ThinClientConfig validation."""

    def test_defaults(self):
        config = ThinClientConfig()
        assert config.socket_path is None
        assert config.connect_timeout == 5.0
        assert config.receive_timeout == 7200.0
        assert config.stream_timeout == 10.0

    def test_custom_values(self):
        config = ThinClientConfig(
            socket_path=Path("/tmp/test.sock"),
            connect_timeout=2.0,
            receive_timeout=15.0,
            stream_timeout=5.0,
        )
        assert config.socket_path == Path("/tmp/test.sock")
        assert config.connect_timeout == 2.0

    def test_zero_connect_timeout_rejected(self):
        with pytest.raises(ValueError, match="connect_timeout must be positive"):
            ThinClientConfig(connect_timeout=0)

    def test_negative_connect_timeout_rejected(self):
        with pytest.raises(ValueError, match="connect_timeout must be positive"):
            ThinClientConfig(connect_timeout=-1.0)

    def test_zero_receive_timeout_rejected(self):
        with pytest.raises(ValueError, match="receive_timeout must be positive"):
            ThinClientConfig(receive_timeout=0)

    def test_zero_stream_timeout_rejected(self):
        with pytest.raises(ValueError, match="stream_timeout must be positive"):
            ThinClientConfig(stream_timeout=0)

    def test_frozen(self):
        config = ThinClientConfig()
        with pytest.raises(AttributeError):
            config.connect_timeout = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CommandResult
# ---------------------------------------------------------------------------


class TestCommandResult:
    """Tests for the CommandResult frozen dataclass."""

    def test_success_result(self):
        response = _make_response("status")
        result = CommandResult(
            success=True,
            verb="status",
            response=response,
            rendered="RESPONSE (status) [ok]",
            error=None,
        )
        assert result.success is True
        assert result.verb == "status"
        assert result.response is response
        assert result.error is None

    def test_failure_result(self):
        result = CommandResult(
            success=False,
            verb="run",
            response=None,
            rendered="Connection failed",
            error="Connection refused",
        )
        assert result.success is False
        assert result.error == "Connection refused"

    def test_frozen(self):
        result = CommandResult(
            success=True,
            verb="health",
            response=None,
            rendered="ok",
            error=None,
        )
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Default confirm callback
# ---------------------------------------------------------------------------


class TestDefaultConfirmCallback:
    """Tests for the default (always-deny) confirm callback."""

    def test_always_denies(self):
        prompt = _make_confirm_prompt()
        assert _default_confirm_callback(prompt) == (False, None)


# ---------------------------------------------------------------------------
# ThinClient with mocked connection
# ---------------------------------------------------------------------------


def _make_mock_connection(
    handshake_success: bool = True,
    responses: list[MessageEnvelope | None] | None = None,
):
    """Create a mock ClientConnection.

    Args:
        handshake_success: Whether connect() should succeed.
        responses: Sequence of responses that receive() will return.

    Returns:
        Mock ClientConnection instance.
    """
    conn = AsyncMock()
    conn.connect = AsyncMock(
        return_value=HandshakeResult(
            success=handshake_success,
            protocol_version=PROTOCOL_VERSION,
            daemon_pid=12345,
            daemon_uptime_seconds=60.0,
            error=None if handshake_success else "Connection refused",
        )
    )

    if responses is not None:
        conn.receive = AsyncMock(side_effect=responses)
    else:
        conn.receive = AsyncMock(return_value=None)

    conn.send = AsyncMock()
    conn.close = AsyncMock()
    return conn


class TestThinClientHealth:
    """Tests for ThinClient.health()."""

    @pytest.mark.asyncio
    async def test_successful_health_check(self):
        response = _make_response("health")
        conn = _make_mock_connection(responses=[response])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.health()

        assert result.success is True
        assert result.verb == "health"
        assert result.response is response
        conn.send.assert_called_once()
        conn.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_handshake_failure(self):
        conn = _make_mock_connection(handshake_success=False)

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.health()

        assert result.success is False
        assert "Handshake failed" in result.rendered

    @pytest.mark.asyncio
    async def test_no_response_timeout(self):
        conn = _make_mock_connection(responses=[None])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.health()

        assert result.success is False
        assert "No response" in result.rendered

    @pytest.mark.asyncio
    async def test_connection_exception(self):
        conn = _make_mock_connection()
        conn.connect = AsyncMock(side_effect=OSError("Connection refused"))

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.health()

        assert result.success is False
        assert "Connection failed" in result.rendered


class TestThinClientStatus:
    """Tests for ThinClient.status()."""

    @pytest.mark.asyncio
    async def test_status_default(self):
        response = _make_response("status", extra={"run_state": "idle"})
        conn = _make_mock_connection(responses=[response])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.status()

        assert result.success is True
        assert "idle" in result.rendered

    @pytest.mark.asyncio
    async def test_status_verbose(self):
        response = _make_response("status", extra={"run_state": "running", "pid": 4567})
        conn = _make_mock_connection(responses=[response])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.status(verbose=True)

        assert result.success is True
        # Verify the request was sent with verbose=True
        sent_envelope = conn.send.call_args[0][0]
        assert sent_envelope.payload["verbose"] is True

    @pytest.mark.asyncio
    async def test_status_error_response(self):
        error = _make_error("status", "Internal error", 500)
        conn = _make_mock_connection(responses=[error])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.status()

        assert result.success is False
        assert result.error == "Internal error"


class TestThinClientHistory:
    """Tests for ThinClient.history()."""

    @pytest.mark.asyncio
    async def test_history_default(self):
        response = _make_response("history", extra={"runs": []})
        conn = _make_mock_connection(responses=[response])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.history()

        assert result.success is True
        sent_envelope = conn.send.call_args[0][0]
        assert sent_envelope.payload["limit"] == 20

    @pytest.mark.asyncio
    async def test_history_with_filters(self):
        response = _make_response("history", extra={"runs": [{"id": "run-1"}]})
        conn = _make_mock_connection(responses=[response])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.history(
                limit=5,
                status_filter="completed",
                host_filter="ci.example.com",
            )

        assert result.success is True
        sent = conn.send.call_args[0][0]
        assert sent.payload["limit"] == 5
        assert sent.payload["status_filter"] == "completed"
        assert sent.payload["host_filter"] == "ci.example.com"

    @pytest.mark.asyncio
    async def test_history_invalid_limit(self):
        client = ThinClient()
        result = await client.history(limit=0)

        assert result.success is False
        assert "Invalid parameters" in result.rendered


class TestThinClientCancel:
    """Tests for ThinClient.cancel()."""

    @pytest.mark.asyncio
    async def test_cancel_current_run(self):
        response = _make_response("cancel", extra={"cancelled_run": "run-abc"})
        conn = _make_mock_connection(responses=[response])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.cancel()

        assert result.success is True

    @pytest.mark.asyncio
    async def test_cancel_specific_run(self):
        response = _make_response("cancel")
        conn = _make_mock_connection(responses=[response])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.cancel(run_id="run-xyz", force=True, reason="Hanging")

        sent = conn.send.call_args[0][0]
        assert sent.payload["run_id"] == "run-xyz"
        assert sent.payload["force"] is True
        assert sent.payload["reason"] == "Hanging"


class TestThinClientRun:
    """Tests for ThinClient.run() with confirmation flow."""

    @pytest.mark.asyncio
    async def test_run_without_confirmation(self):
        """Daemon accepts run immediately (no confirmation needed)."""
        response = _make_response("run", extra={"run_id": "run-new-001"})
        conn = _make_mock_connection(responses=[response])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.run(
                target_host="ci.example.com",
                target_user="deploy",
                natural_language="run unit tests",
            )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_with_system_name(self):
        """Named systems can be sent without an explicit SSH target."""
        response = _make_response("run", extra={"run_id": "run-new-002"})
        conn = _make_mock_connection(responses=[response])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.run(
                natural_language="run unit tests in system tuto",
                system_name="tuto",
            )

        assert result.success is True
        sent = conn.send.call_args_list[0][0][0]
        assert sent.payload["system_name"] == "tuto"
        assert "target_host" not in sent.payload

    @pytest.mark.asyncio
    async def test_run_with_confirmation_approved(self):
        """Daemon sends CONFIRM_PROMPT, user approves, daemon responds."""
        prompt = _make_confirm_prompt(command="pytest -v")
        final_response = _make_response("run", extra={"run_id": "run-002"})
        conn = _make_mock_connection(responses=[prompt, final_response])

        # Approve callback
        approve_callback = MagicMock(return_value=(True, None))
        client = ThinClient(on_confirm=approve_callback)

        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.run(
                target_host="ci.example.com",
                target_user="deploy",
                natural_language="run unit tests",
            )

        assert result.success is True
        approve_callback.assert_called_once()

        # Verify confirm reply was sent
        assert conn.send.call_count == 2  # run request + confirm reply
        confirm_reply = conn.send.call_args_list[1][0][0]
        assert confirm_reply.msg_type == MessageType.CONFIRM_REPLY
        assert confirm_reply.payload["approved"] is True
        assert confirm_reply.payload["original_msg_id"] == "daemon-prompt-001"

    @pytest.mark.asyncio
    async def test_run_with_confirmation_denied(self):
        """Daemon sends CONFIRM_PROMPT, user denies, daemon responds with error."""
        prompt = _make_confirm_prompt()
        error_response = _make_error("run", "Command denied by user", 403)
        conn = _make_mock_connection(responses=[prompt, error_response])

        deny_callback = MagicMock(return_value=(False, None))
        client = ThinClient(on_confirm=deny_callback)

        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.run(
                target_host="ci.example.com",
                target_user="deploy",
                natural_language="run tests",
            )

        assert result.success is False
        deny_callback.assert_called_once()

        # Verify deny was sent
        confirm_reply = conn.send.call_args_list[1][0][0]
        assert confirm_reply.payload["approved"] is False

    @pytest.mark.asyncio
    async def test_run_with_default_callback_denies(self):
        """Default callback always denies for security."""
        prompt = _make_confirm_prompt()
        error_response = _make_error("run", "Command denied", 403)
        conn = _make_mock_connection(responses=[prompt, error_response])

        client = ThinClient()  # default callback

        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.run(
                target_host="ci.example.com",
                target_user="deploy",
                natural_language="run tests",
            )

        # Default callback denies
        confirm_reply = conn.send.call_args_list[1][0][0]
        assert confirm_reply.payload["approved"] is False

    @pytest.mark.asyncio
    async def test_run_invalid_params(self):
        """Invalid parameters produce error without connecting."""
        client = ThinClient()
        result = await client.run(
            target_host="",  # invalid
            target_user="deploy",
            natural_language="run tests",
        )
        assert result.success is False
        assert "Invalid parameters" in result.rendered

    @pytest.mark.asyncio
    async def test_run_empty_natural_language(self):
        """Empty natural language produces error without connecting."""
        client = ThinClient()
        result = await client.run(
            target_host="ci.example.com",
            target_user="deploy",
            natural_language="",  # invalid
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_run_rejects_system_name_with_explicit_target(self):
        client = ThinClient()
        result = await client.run(
            natural_language="run tests",
            system_name="tuto",
            target_host="ci.example.com",
        )
        assert result.success is False
        assert "system_name cannot be combined" in result.rendered

    @pytest.mark.asyncio
    async def test_run_timeout_during_confirmation(self):
        """Timeout waiting for daemon response after confirm reply."""
        prompt = _make_confirm_prompt()
        conn = _make_mock_connection(responses=[prompt, None])

        approve = MagicMock(return_value=(True, None))
        client = ThinClient(on_confirm=approve)

        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.run(
                target_host="ci.example.com",
                target_user="deploy",
                natural_language="run tests",
            )

        assert result.success is False
        assert "No response" in result.rendered


class TestThinClientWatch:
    """Tests for ThinClient.watch() with streaming output."""

    @pytest.mark.asyncio
    async def test_watch_receives_lines(self):
        """Receives stream lines and end-of-stream."""
        sub_response = _make_response("watch", extra={
            "subscriber_id": "sub-001",
            "job_id": "run-abc",
        })
        line1 = _make_stream("PASS: test_auth", sequence=1)
        line2 = _make_stream("PASS: test_payment", sequence=2)
        end = _make_stream("", sequence=3, is_end=True)

        conn = _make_mock_connection(responses=[sub_response, line1, line2, end])

        received_lines: list[str] = []
        client = ThinClient()

        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.watch(on_line=received_lines.append)

        assert result.success is True
        assert len(received_lines) == 2
        assert received_lines[0] == "PASS: test_auth"
        assert received_lines[1] == "PASS: test_payment"
        assert "2 lines received" in result.rendered

    @pytest.mark.asyncio
    async def test_watch_max_lines(self):
        """Stops after max_lines is reached."""
        sub_response = _make_response("watch", extra={"subscriber_id": "sub-001"})
        lines = [_make_stream(f"line-{i}", sequence=i) for i in range(10)]

        conn = _make_mock_connection(responses=[sub_response] + lines)

        received: list[str] = []
        client = ThinClient()

        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.watch(
                on_line=received.append,
                max_lines=3,
            )

        assert result.success is True
        assert len(received) == 3
        assert "3 lines received" in result.rendered

    @pytest.mark.asyncio
    async def test_watch_subscription_error(self):
        """Error during subscription."""
        error = _make_error("watch", "No active run", 404)
        conn = _make_mock_connection(responses=[error])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.watch()

        assert result.success is False
        assert "No active run" in result.rendered

    @pytest.mark.asyncio
    async def test_watch_stream_error(self):
        """Error during streaming."""
        sub_response = _make_response("watch", extra={"subscriber_id": "sub-001"})
        line1 = _make_stream("PASS: test_auth", sequence=1)
        error = _make_error("watch", "SSH connection lost", 502)

        conn = _make_mock_connection(responses=[sub_response, line1, error])

        received: list[str] = []
        client = ThinClient()

        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.watch(on_line=received.append)

        assert result.success is False
        assert "SSH connection lost" in result.rendered
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_watch_connection_lost(self):
        """Connection drops during streaming (None response)."""
        sub_response = _make_response("watch", extra={"subscriber_id": "sub-001"})
        line1 = _make_stream("PASS: test_auth", sequence=1)

        conn = _make_mock_connection(responses=[sub_response, line1, None])

        received: list[str] = []
        client = ThinClient()

        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.watch(on_line=received.append)

        assert result.success is True  # graceful completion on disconnect
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_watch_invalid_tail_lines(self):
        """Invalid tail_lines produces error without connecting."""
        client = ThinClient()
        result = await client.watch(tail_lines=0)

        assert result.success is False
        assert "Invalid parameters" in result.rendered

    @pytest.mark.asyncio
    async def test_watch_no_subscription_response(self):
        """Timeout waiting for subscription response."""
        conn = _make_mock_connection(responses=[None])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.watch()

        assert result.success is False
        assert "No subscription response" in result.rendered

    @pytest.mark.asyncio
    async def test_watch_no_on_line_callback(self):
        """Watch works without an on_line callback."""
        sub_response = _make_response("watch", extra={"subscriber_id": "sub-001"})
        line1 = _make_stream("output", sequence=1)
        end = _make_stream("", sequence=2, is_end=True)

        conn = _make_mock_connection(responses=[sub_response, line1, end])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.watch()

        assert result.success is True
        assert "1 lines received" in result.rendered


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


class TestConnectionLifecycle:
    """Tests that connections are always closed properly."""

    @pytest.mark.asyncio
    async def test_close_called_on_success(self):
        response = _make_response("health")
        conn = _make_mock_connection(responses=[response])

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            await client.health()

        conn.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_called_on_send_error(self):
        conn = _make_mock_connection()
        conn.send = AsyncMock(side_effect=BrokenPipeError("Pipe broken"))

        client = ThinClient()
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.status()

        assert result.success is False
        conn.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_called_on_confirmation_error(self):
        prompt = _make_confirm_prompt()
        conn = _make_mock_connection(responses=[prompt])
        # Make the confirm reply send fail
        conn.send = AsyncMock(side_effect=[None, BrokenPipeError("broken")])

        client = ThinClient(on_confirm=MagicMock(return_value=(True, None)))
        with patch.object(client, "_create_connection", return_value=conn):
            result = await client.run(
                target_host="ci.example.com",
                target_user="deploy",
                natural_language="run tests",
            )

        # Connection should be closed even on error
        conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# Config propagation
# ---------------------------------------------------------------------------


class TestConfigPropagation:
    """Tests that config values are correctly propagated to connections."""

    @pytest.mark.asyncio
    async def test_socket_path_propagated(self):
        socket_path = Path("/tmp/test-jules.sock")
        config = ThinClientConfig(socket_path=socket_path)
        client = ThinClient(config=config)

        # Verify the connection config gets the right socket path
        conn = client._create_connection()
        assert conn.config.socket_path == socket_path

    @pytest.mark.asyncio
    async def test_connect_timeout_propagated(self):
        config = ThinClientConfig(connect_timeout=2.5)
        client = ThinClient(config=config)

        conn = client._create_connection()
        assert conn.config.connect_timeout == 2.5

    @pytest.mark.asyncio
    async def test_default_config_when_none(self):
        client = ThinClient()
        assert client.config.socket_path is None
        assert client.config.connect_timeout == 5.0
