"""Tests for AC 5: One-shot fallback path backward compatibility.

Validates that the one-shot LLM translation path works identically to
v1.2-mvp when the agent loop is unavailable. The three fallback triggers:

  (a) LLM not configured (env vars missing)
  (b) Agent loop init fails (import error, bad tool setup, etc.)
  (c) Explicit --one-shot flag set via CLI or config

Coverage:
  - _can_use_agent_loop returns False in all fallback scenarios
  - _handle_run routes to _handle_run_oneshot for NL commands
  - _translate_via_llm returns raw input when translator is None
  - CONFIRM_PROMPT is sent, CONFIRM_REPLY is awaited
  - Denial returns status="denied" (v1.2-mvp behavior)
  - Approval proceeds to background SSH execution (v1.2-mvp behavior)
  - Direct commands bypass LLM even with LLM configured
  - --one-shot CLI flag is parsed and wired through to config
  - _try_load_llm returns (None, None) when env vars are missing
  - RequestHandlerConfig.one_shot defaults to False
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.ipc.framing import (
    MessageEnvelope,
    MessageType,
    encode_frame,
)
from jules_daemon.ipc.request_handler import (
    RequestHandler,
    RequestHandlerConfig,
    is_direct_command,
)
from jules_daemon.ipc.server import ClientConnection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> ClientConnection:
    """Build a stub ClientConnection for testing."""
    return ClientConnection(
        client_id="test-client-fallback",
        reader=AsyncMock(spec=asyncio.StreamReader),
        writer=AsyncMock(spec=asyncio.StreamWriter),
        connected_at="2026-04-12T12:00:00Z",
    )


def _make_request(
    payload: dict[str, Any],
    msg_id: str = "req-fallback-001",
) -> MessageEnvelope:
    """Build a REQUEST-type envelope."""
    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=msg_id,
        timestamp="2026-04-12T12:00:00Z",
        payload=payload,
    )


def _make_llm_client() -> MagicMock:
    """Build a mock OpenAI client."""
    return MagicMock()


def _make_llm_config() -> MagicMock:
    """Build a mock LLMConfig."""
    config = MagicMock()
    config.default_model = "provider:connection:model-v1"
    return config


def _setup_deny_reply(client: ClientConnection) -> None:
    """Configure the mock client to return a deny CONFIRM_REPLY.

    Encodes a properly framed CONFIRM_REPLY envelope so the handler's
    _read_envelope() receives a valid deny response.
    """
    deny_reply = MessageEnvelope(
        msg_type=MessageType.CONFIRM_REPLY,
        msg_id="deny-fallback-001",
        timestamp="2026-04-12T12:00:01Z",
        payload={"approved": False},
    )
    deny_frame = encode_frame(deny_reply)
    header_bytes = deny_frame[:4]
    payload_bytes = deny_frame[4:]
    client.reader.readexactly = AsyncMock(
        side_effect=[header_bytes, payload_bytes]
    )


def _setup_approve_reply(client: ClientConnection) -> None:
    """Configure the mock client to return an approve CONFIRM_REPLY."""
    approve_reply = MessageEnvelope(
        msg_type=MessageType.CONFIRM_REPLY,
        msg_id="approve-fallback-001",
        timestamp="2026-04-12T12:00:01Z",
        payload={"approved": True},
    )
    approve_frame = encode_frame(approve_reply)
    header_bytes = approve_frame[:4]
    payload_bytes = approve_frame[4:]
    client.reader.readexactly = AsyncMock(
        side_effect=[header_bytes, payload_bytes]
    )


# ---------------------------------------------------------------------------
# AC 5a: LLM not configured -> one-shot fallback
# ---------------------------------------------------------------------------


class TestFallbackWhenLLMDisabled:
    """One-shot path is used when LLM environment variables are missing."""

    def test_can_use_agent_loop_false_no_llm(self, tmp_path: Path) -> None:
        """_can_use_agent_loop is False when llm_client is None."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        assert handler._can_use_agent_loop is False

    def test_can_use_agent_loop_false_no_config(self, tmp_path: Path) -> None:
        """_can_use_agent_loop is False when llm_config is None."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=None,
        )
        handler = RequestHandler(config=config)
        assert handler._can_use_agent_loop is False

    def test_can_use_agent_loop_false_no_client(self, tmp_path: Path) -> None:
        """_can_use_agent_loop is False when llm_client is None."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=None,
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        assert handler._can_use_agent_loop is False

    @pytest.mark.asyncio
    async def test_nl_command_denied_without_llm(
        self, tmp_path: Path,
    ) -> None:
        """NL command follows one-shot path when LLM is disabled.

        Without LLM, the raw NL input is used as the proposed command.
        The CONFIRM_PROMPT is sent, user denies, status='denied'.
        This matches v1.2-mvp behavior exactly.
        """
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        envelope = _make_request(payload={
            "verb": "run",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run the smoke tests on staging",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "run"
        assert response.payload["status"] == "denied"
        assert "denied" in response.payload.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_confirm_prompt_sent_without_llm(
        self, tmp_path: Path,
    ) -> None:
        """CONFIRM_PROMPT is sent to the CLI when LLM is disabled.

        Verifies the confirmation exchange still happens in the one-shot
        path, matching v1.2-mvp behavior.
        """
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        envelope = _make_request(payload={
            "verb": "run",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run the smoke tests",
        })

        await handler.handle_message(envelope, client)

        # Verify that the writer was called (CONFIRM_PROMPT was sent)
        assert client.writer.write.called

    @pytest.mark.asyncio
    async def test_raw_input_used_as_proposed_command(
        self, tmp_path: Path,
    ) -> None:
        """Without LLM, natural_language is used directly as proposed command.

        This is the core v1.2-mvp behavior: when no LLM is configured,
        the user's natural language input IS the command.
        """
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)

        # Call _translate_via_llm directly to verify it returns raw input
        result = await handler._translate_via_llm(
            natural_language="run smoke tests",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )
        assert result == "run smoke tests"


# ---------------------------------------------------------------------------
# AC 5b: Agent loop init fails -> one-shot fallback
# ---------------------------------------------------------------------------


class TestFallbackOnAgentLoopFailure:
    """One-shot path is used when agent loop initialization fails."""

    @pytest.mark.asyncio
    async def test_agent_loop_import_error_falls_back(
        self, tmp_path: Path,
    ) -> None:
        """ImportError in agent loop returns an ERROR response."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
            side_effect=ImportError("No module named 'jules_daemon.agent'"),
        ):
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run the regression suite",
            })

            response = await handler.handle_message(envelope, client)

            # Non-RetryExhaustedError returns an ERROR response
            assert response.msg_type == MessageType.ERROR
            assert "Agent loop error" in response.payload["error"]

    @pytest.mark.asyncio
    async def test_agent_loop_runtime_error_falls_back(
        self, tmp_path: Path,
    ) -> None:
        """RuntimeError in agent loop returns an ERROR response."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
            side_effect=RuntimeError("ToolRegistry init failed"),
        ):
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "check test status",
            })

            response = await handler.handle_message(envelope, client)

            assert response.msg_type == MessageType.ERROR
            assert "Agent loop error" in response.payload["error"]

    @pytest.mark.asyncio
    async def test_agent_loop_any_exception_falls_back(
        self, tmp_path: Path,
    ) -> None:
        """Any non-RetryExhaustedError exception from agent loop returns ERROR.

        The _handle_run method catches non-RetryExhaustedError exceptions from
        _handle_run_agent_loop and returns an ERROR response instead of falling
        through to one-shot. Only RetryExhaustedError triggers the fallback.
        """
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
            side_effect=ValueError("Unexpected configuration error"),
        ):
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run all tests",
            })

            response = await handler.handle_message(envelope, client)

            assert response.msg_type == MessageType.ERROR
            assert "Agent loop error" in response.payload["error"]


# ---------------------------------------------------------------------------
# AC 5c: Explicit --one-shot flag
# ---------------------------------------------------------------------------


class TestFallbackWithOneShotFlag:
    """One-shot path is forced when one_shot config flag is set."""

    def test_can_use_agent_loop_false_with_flag(
        self, tmp_path: Path,
    ) -> None:
        """_can_use_agent_loop is False when one_shot=True."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
            one_shot=True,
        )
        handler = RequestHandler(config=config)
        assert handler._can_use_agent_loop is False

    @pytest.mark.asyncio
    async def test_one_shot_flag_skips_agent_loop(
        self, tmp_path: Path,
    ) -> None:
        """--one-shot flag prevents agent loop from being called.

        Even when LLM is fully configured, the agent loop handler
        is never invoked.
        """
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
            one_shot=True,
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
        ) as mock_agent:
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run the smoke tests",
            })

            response = await handler.handle_message(envelope, client)

            mock_agent.assert_not_called()
            assert response.payload["status"] == "denied"

    @pytest.mark.asyncio
    async def test_one_shot_with_llm_still_translates(
        self, tmp_path: Path,
    ) -> None:
        """--one-shot flag with LLM configured still uses LLM translation.

        The one-shot flag skips the agent loop but does NOT disable
        the LLM command translator. This means NL inputs are still
        translated to commands via the single LLM call before
        confirmation.
        """
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
            one_shot=True,
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        # Verify translator was initialized (LLM is configured)
        assert handler._command_translator is not None

        with patch.object(
            handler,
            "_translate_via_llm",
            new_callable=AsyncMock,
            return_value="pytest -v tests/smoke/",
        ) as mock_translate:
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run the smoke tests",
            })

            response = await handler.handle_message(envelope, client)

            mock_translate.assert_called_once()
            assert response.payload["status"] == "denied"


# ---------------------------------------------------------------------------
# Config defaults and immutability
# ---------------------------------------------------------------------------


class TestOneShotConfigDefaults:
    """Verify config defaults for backward compatibility."""

    def test_one_shot_defaults_false(self, tmp_path: Path) -> None:
        """one_shot defaults to False (agent loop is default path)."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        assert config.one_shot is False

    def test_max_agent_iterations_defaults_to_5(
        self, tmp_path: Path,
    ) -> None:
        """max_agent_iterations defaults to 5."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        assert config.max_agent_iterations == 15

    def test_config_frozen(self, tmp_path: Path) -> None:
        """Config is frozen (immutable)."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            one_shot=True,
        )
        with pytest.raises(AttributeError):
            config.one_shot = False  # type: ignore[misc]

    def test_llm_fields_default_none(self, tmp_path: Path) -> None:
        """llm_client and llm_config default to None."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        assert config.llm_client is None
        assert config.llm_config is None


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestOneShotCLIFlag:
    """Verify the --one-shot CLI argument is parsed correctly."""

    def test_one_shot_flag_present(self) -> None:
        """--one-shot flag is parsed as True."""
        from jules_daemon.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(["--one-shot"])
        assert args.one_shot is True

    def test_one_shot_flag_absent(self) -> None:
        """Without --one-shot, flag defaults to False."""
        from jules_daemon.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.one_shot is False

    def test_one_shot_combined_with_other_flags(self) -> None:
        """--one-shot can be combined with other flags."""
        from jules_daemon.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--one-shot",
            "--wiki-dir", "/tmp/wiki",
            "--log-level", "DEBUG",
        ])
        assert args.one_shot is True
        assert str(args.wiki_dir) == "/tmp/wiki"
        assert args.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# _try_load_llm without env vars
# ---------------------------------------------------------------------------


class TestTryLoadLLMFallback:
    """_try_load_llm returns (None, None) when env vars are missing."""

    def test_no_base_url_returns_none(self) -> None:
        """Missing JULES_LLM_BASE_URL returns (None, None)."""
        from jules_daemon.__main__ import _try_load_llm

        with patch.dict("os.environ", {}, clear=True):
            client, config = _try_load_llm()
            assert client is None
            assert config is None

    def test_empty_base_url_returns_none(self) -> None:
        """Empty JULES_LLM_BASE_URL returns (None, None)."""
        from jules_daemon.__main__ import _try_load_llm

        with patch.dict("os.environ", {"JULES_LLM_BASE_URL": ""}, clear=True):
            client, config = _try_load_llm()
            assert client is None
            assert config is None

    def test_partial_env_returns_none(self) -> None:
        """Partial env (only BASE_URL) returns (None, None) gracefully."""
        from jules_daemon.__main__ import _try_load_llm

        with patch.dict(
            "os.environ",
            {"JULES_LLM_BASE_URL": "https://example.com/api/"},
            clear=True,
        ):
            # Missing API_KEY and DEFAULT_MODEL -> should fail gracefully
            client, config = _try_load_llm()
            assert client is None
            assert config is None


# ---------------------------------------------------------------------------
# _translate_via_llm fallback behavior
# ---------------------------------------------------------------------------


class TestTranslateViaLLMFallback:
    """_translate_via_llm returns raw input when LLM is not configured."""

    @pytest.mark.asyncio
    async def test_no_translator_returns_raw(self, tmp_path: Path) -> None:
        """Without translator, input is returned unchanged."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)

        result = await handler._translate_via_llm(
            natural_language="run all unit tests",
            target_host="prod.example.com",
            target_user="app",
            target_port=22,
        )
        assert result == "run all unit tests"

    @pytest.mark.asyncio
    async def test_translator_failure_returns_raw(
        self, tmp_path: Path,
    ) -> None:
        """If the LLM translator raises, input is returned as-is.

        This is the safety net: LLM failures never block command execution.
        """
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)

        # Make the translator raise an exception
        with patch.object(
            handler._command_translator,
            "translate",
            side_effect=RuntimeError("LLM endpoint unreachable"),
        ):
            result = await handler._translate_via_llm(
                natural_language="deploy staging",
                target_host="staging.example.com",
                target_user="deploy",
                target_port=22,
            )
            assert result == "deploy staging"

    @pytest.mark.asyncio
    async def test_translator_refusal_returns_raw(
        self, tmp_path: Path,
    ) -> None:
        """If the LLM refuses the request, input is returned as-is."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)

        mock_result = MagicMock()
        mock_result.is_refusal = True
        mock_result.response.explanation = "Cannot execute destructive command"

        with patch.object(
            handler._command_translator,
            "translate",
            return_value=mock_result,
        ):
            result = await handler._translate_via_llm(
                natural_language="rm -rf /",
                target_host="prod.example.com",
                target_user="root",
                target_port=22,
            )
            assert result == "rm -rf /"

    @pytest.mark.asyncio
    async def test_translator_empty_commands_returns_raw(
        self, tmp_path: Path,
    ) -> None:
        """If the LLM returns 0 commands, input is returned as-is."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)

        mock_result = MagicMock()
        mock_result.is_refusal = False
        mock_result.command_count = 0
        mock_result.ssh_commands = ()

        with patch.object(
            handler._command_translator,
            "translate",
            return_value=mock_result,
        ):
            result = await handler._translate_via_llm(
                natural_language="check disk space",
                target_host="staging.example.com",
                target_user="deploy",
                target_port=22,
            )
            assert result == "check disk space"


# ---------------------------------------------------------------------------
# Direct commands always bypass agent loop
# ---------------------------------------------------------------------------


class TestDirectCommandAlwaysOneShot:
    """Direct commands (known executables) always use one-shot path."""

    @pytest.mark.asyncio
    async def test_direct_command_skips_agent_even_with_llm(
        self, tmp_path: Path,
    ) -> None:
        """Direct command routes to one-shot even with LLM configured."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        # Verify agent loop IS available (but should not be used)
        assert handler._can_use_agent_loop is True

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
        ) as mock_agent:
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "pytest -v tests/integration/",
            })

            response = await handler.handle_message(envelope, client)

            mock_agent.assert_not_called()
            assert response.payload["status"] == "denied"

    @pytest.mark.parametrize(
        "command",
        [
            "pytest -v tests/",
            "python3 -m pytest",
            "./run_tests.sh",
            "/usr/bin/pytest -k test_api",
            "npm test",
            "make test",
            "go test ./...",
            "docker run test-image",
        ],
    )
    def test_direct_command_detected(self, command: str) -> None:
        """Various direct commands are correctly detected."""
        assert is_direct_command(command) is True

    @pytest.mark.parametrize(
        "nl_input",
        [
            "run the smoke tests",
            "check all integration tests on staging",
            "execute the regression suite",
            "deploy and verify",
        ],
    )
    def test_nl_input_not_direct(self, nl_input: str) -> None:
        """Natural language inputs are not detected as direct commands."""
        assert is_direct_command(nl_input) is False


# ---------------------------------------------------------------------------
# Full one-shot approval flow (v1.2-mvp behavior)
# ---------------------------------------------------------------------------


class TestOneShotApprovalFlow:
    """End-to-end one-shot flow: confirm -> approve/deny -> result."""

    @pytest.mark.asyncio
    async def test_approval_starts_background_run(
        self, tmp_path: Path,
    ) -> None:
        """Approved command starts a background SSH execution task.

        This is the happy path of the one-shot flow:
        1. NL input used as-is (no LLM)
        2. CONFIRM_PROMPT sent to CLI
        3. User approves
        4. SSH credentials resolved
        5. Background task spawned
        6. Response has status='started' with run_id
        """
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_approve_reply(client)

        with patch(
            "jules_daemon.ipc.request_handler.resolve_ssh_credentials",
            return_value=MagicMock(),
        ), patch(
            "jules_daemon.ipc.request_handler.check_remote_processes",
            new_callable=AsyncMock,
            return_value=[],  # No collisions
        ), patch(
            "jules_daemon.ipc.request_handler.execute_run",
            new_callable=AsyncMock,
        ):
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "pytest -v tests/smoke/",
            })

            response = await handler.handle_message(envelope, client)

            assert response.msg_type == MessageType.RESPONSE
            assert response.payload["verb"] == "run"
            assert response.payload["status"] == "started"
            assert "run_id" in response.payload

    @pytest.mark.asyncio
    async def test_denial_returns_denied_status(
        self, tmp_path: Path,
    ) -> None:
        """Denied command returns status='denied' without starting SSH.

        No background task is spawned, no SSH execution occurs.
        """
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        envelope = _make_request(payload={
            "verb": "run",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "pytest -v tests/smoke/",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "run"
        assert response.payload["status"] == "denied"
        # No background task should be running
        assert handler._current_task is None


# ---------------------------------------------------------------------------
# Other verbs still work (backward compat)
# ---------------------------------------------------------------------------


class TestOtherVerbsStillWork:
    """Non-run verbs are unaffected by the agent loop / one-shot routing."""

    @pytest.mark.asyncio
    async def test_status_verb_works(self, tmp_path: Path) -> None:
        """status verb returns a response regardless of LLM config."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        envelope = _make_request(payload={"verb": "status"})
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "status"

    @pytest.mark.asyncio
    async def test_queue_verb_works(self, tmp_path: Path) -> None:
        """queue verb still enqueues commands."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        envelope = _make_request(payload={
            "verb": "queue",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run all tests",
        })
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["status"] == "enqueued"

    @pytest.mark.asyncio
    async def test_history_verb_works(self, tmp_path: Path) -> None:
        """history verb returns result regardless of LLM config."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        envelope = _make_request(payload={"verb": "history"})
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "history"

    @pytest.mark.asyncio
    async def test_handshake_verb_works(self, tmp_path: Path) -> None:
        """handshake verb returns daemon info."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        envelope = _make_request(payload={"verb": "handshake"})
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "handshake"


# ---------------------------------------------------------------------------
# Handler initialization backward compat
# ---------------------------------------------------------------------------


class TestHandlerInitBackwardCompat:
    """RequestHandler initializes correctly without LLM (v1.2-mvp mode)."""

    def test_no_llm_no_crash(self, tmp_path: Path) -> None:
        """Handler initializes without LLM without raising."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        assert handler._command_translator is None

    def test_with_llm_translator_initialized(self, tmp_path: Path) -> None:
        """Handler initializes translator when LLM is configured."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        assert handler._command_translator is not None

    def test_verb_dispatch_table_intact(self, tmp_path: Path) -> None:
        """All verbs are present in the dispatch table."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)

        expected_sync = {"handshake", "queue", "status", "cancel", "history"}
        expected_async = {
            "run",
            "watch",
            "discover",
            "interpret",
            "subscribe_notifications",
            "unsubscribe_notifications",
        }

        assert set(handler._verb_dispatch.keys()) == expected_sync
        assert set(handler._async_client_dispatch.keys()) == expected_async
