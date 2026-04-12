"""Tests for AC 15 Sub-AC 2: Direct command bypass wiring.

Validates that detected direct commands skip the agent loop and go
straight to SSH execution (with human approval), preserving the
existing v1.2-mvp latency.

Coverage:
  - detect_direct_command is called in _handle_run before routing
  - Direct commands (pytest, python3, ./script) bypass agent loop
  - DirectCommandDetection is passed through to _handle_run_oneshot
  - _handle_run_oneshot reuses the pre-computed detection (no double call)
  - Agent loop is never invoked for direct commands
  - Human approval (CONFIRM_PROMPT) is still required for direct commands
  - Bypass preserves v1.2-mvp latency (no LLM call for direct commands)
  - NL commands still route to agent loop when available
  - Fallback path (no detection passed) re-runs detection internally
  - Env-var prefixed commands detected via new classifier
  - Sudo-prefixed commands detected via new classifier
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.classifier.direct_command import (
    DirectCommandDetection,
    detect_direct_command,
)
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
        client_id="test-client-bypass",
        reader=AsyncMock(spec=asyncio.StreamReader),
        writer=AsyncMock(spec=asyncio.StreamWriter),
        connected_at="2026-04-12T12:00:00Z",
    )


def _make_request(
    payload: dict[str, Any],
    msg_id: str = "req-bypass-001",
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
    """Configure the mock client to return a deny CONFIRM_REPLY."""
    deny_reply = MessageEnvelope(
        msg_type=MessageType.CONFIRM_REPLY,
        msg_id="deny-bypass-001",
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
        msg_id="approve-bypass-001",
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
# Bypass routing: direct commands skip agent loop
# ---------------------------------------------------------------------------


class TestDirectCommandBypassRouting:
    """Direct commands detected via detect_direct_command bypass the agent loop."""

    @pytest.mark.asyncio
    async def test_pytest_command_bypasses_agent_loop(
        self, tmp_path: Path,
    ) -> None:
        """A pytest command goes straight to one-shot SSH approval."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        # Agent loop IS available
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

            # Agent loop was never called
            mock_agent.assert_not_called()
            # Human approval flow was reached (user denied)
            assert response.payload["status"] == "denied"

    @pytest.mark.asyncio
    async def test_python3_command_bypasses_agent_loop(
        self, tmp_path: Path,
    ) -> None:
        """python3 command goes straight to SSH approval."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
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
                "natural_language": "python3 -m pytest --tb=short",
            })

            response = await handler.handle_message(envelope, client)

            mock_agent.assert_not_called()
            assert response.payload["status"] == "denied"

    @pytest.mark.asyncio
    async def test_dotslash_script_bypasses_agent_loop(
        self, tmp_path: Path,
    ) -> None:
        """./script.sh goes straight to SSH approval."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
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
                "natural_language": "./run_tests.sh --verbose",
            })

            response = await handler.handle_message(envelope, client)

            mock_agent.assert_not_called()
            assert response.payload["status"] == "denied"

    @pytest.mark.asyncio
    async def test_absolute_path_bypasses_agent_loop(
        self, tmp_path: Path,
    ) -> None:
        """/usr/bin/python3 bypasses agent loop."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
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
                "natural_language": "/usr/local/bin/pytest -v tests/",
            })

            response = await handler.handle_message(envelope, client)

            mock_agent.assert_not_called()
            assert response.payload["status"] == "denied"


# ---------------------------------------------------------------------------
# Enhanced detection: env vars and sudo prefixes
# ---------------------------------------------------------------------------


class TestEnhancedDetectionBypass:
    """New classifier handles env vars and sudo, legacy one does not."""

    @pytest.mark.asyncio
    async def test_env_prefixed_command_bypasses_agent_loop(
        self, tmp_path: Path,
    ) -> None:
        """PYTHONPATH=/opt/app pytest -v bypasses the agent loop.

        The legacy is_direct_command would NOT detect this because
        the first token starts with an env var, not an executable.
        The new detect_direct_command strips env prefixes first.
        """
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        command = "PYTHONPATH=/opt/app pytest -v tests/"

        # Verify the new classifier detects it
        detection = detect_direct_command(command)
        assert detection.bypass_agent_loop is True
        assert detection.executable == "pytest"

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
        ) as mock_agent:
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": command,
            })

            response = await handler.handle_message(envelope, client)

            mock_agent.assert_not_called()
            assert response.payload["status"] == "denied"

    @pytest.mark.asyncio
    async def test_sudo_prefixed_command_bypasses_agent_loop(
        self, tmp_path: Path,
    ) -> None:
        """sudo pytest -v bypasses the agent loop.

        The new detect_direct_command strips sudo prefix before checking.
        """
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        command = "sudo pytest -v tests/"

        detection = detect_direct_command(command)
        assert detection.bypass_agent_loop is True
        assert detection.executable == "pytest"

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
        ) as mock_agent:
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": command,
            })

            response = await handler.handle_message(envelope, client)

            mock_agent.assert_not_called()
            assert response.payload["status"] == "denied"

    @pytest.mark.asyncio
    async def test_env_and_sudo_combined_bypasses_agent_loop(
        self, tmp_path: Path,
    ) -> None:
        """LANG=C sudo pytest -v is detected and bypasses."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        command = "LANG=C sudo pytest -v tests/"

        detection = detect_direct_command(command)
        assert detection.bypass_agent_loop is True

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
        ) as mock_agent:
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": command,
            })

            response = await handler.handle_message(envelope, client)

            mock_agent.assert_not_called()
            assert response.payload["status"] == "denied"


# ---------------------------------------------------------------------------
# NL commands still use the agent loop
# ---------------------------------------------------------------------------


class TestNLCommandsUseAgentLoop:
    """Natural language commands still route to the agent loop."""

    @pytest.mark.asyncio
    async def test_nl_command_tries_agent_loop(
        self, tmp_path: Path,
    ) -> None:
        """NL input does NOT bypass and tries the agent loop."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
            # Simulate agent loop failure to fall through
            side_effect=RuntimeError("Agent loop test error"),
        ) as mock_agent:
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run the smoke tests on staging",
            })

            response = await handler.handle_message(envelope, client)

            # Agent loop WAS called (even though it failed)
            mock_agent.assert_called_once()
            # Fell back to one-shot path, user denied
            assert response.payload["status"] == "denied"

    @pytest.mark.asyncio
    async def test_daemon_verb_not_detected_as_direct(
        self, tmp_path: Path,
    ) -> None:
        """Daemon verbs (status, watch) are NOT direct commands."""
        detection = detect_direct_command("status")
        assert detection.bypass_agent_loop is False
        assert detection.is_direct_command is False


# ---------------------------------------------------------------------------
# Detection object is passed through (no double detection)
# ---------------------------------------------------------------------------


class TestDetectionPassthrough:
    """_handle_run passes detection to _handle_run_oneshot to avoid re-running."""

    @pytest.mark.asyncio
    async def test_detection_passed_to_oneshot(
        self, tmp_path: Path,
    ) -> None:
        """detect_direct_command is called once, result reused in oneshot."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        call_count = 0
        original_detect = detect_direct_command

        def counting_detect(raw: str, **kwargs: Any) -> DirectCommandDetection:
            nonlocal call_count
            call_count += 1
            return original_detect(raw, **kwargs)

        with patch(
            "jules_daemon.ipc.request_handler.detect_direct_command",
            side_effect=counting_detect,
        ):
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "pytest -v tests/",
            })

            await handler.handle_message(envelope, client)

            # detect_direct_command is called exactly once in _handle_run.
            # _handle_run_oneshot reuses the detection, not re-detecting.
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_oneshot_fallback_detects_when_no_detection_passed(
        self, tmp_path: Path,
    ) -> None:
        """When _handle_run_oneshot is called without detection, it detects."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        call_count = 0
        original_detect = detect_direct_command

        def counting_detect(raw: str, **kwargs: Any) -> DirectCommandDetection:
            nonlocal call_count
            call_count += 1
            return original_detect(raw, **kwargs)

        with patch(
            "jules_daemon.ipc.request_handler.detect_direct_command",
            side_effect=counting_detect,
        ):
            # Call _handle_run_oneshot directly without detection param
            # This simulates the fallback path (e.g. from agent loop failure)
            response = await handler._handle_run_oneshot(
                msg_id="req-test-001",
                parsed={
                    "verb": "run",
                    "target_host": "staging.example.com",
                    "target_user": "deploy",
                    "natural_language": "pytest -v tests/",
                },
                client=client,
            )

            # detection was called inside _handle_run_oneshot
            assert call_count == 1
            assert response.payload["status"] == "denied"


# ---------------------------------------------------------------------------
# Human approval still required for direct commands
# ---------------------------------------------------------------------------


class TestHumanApprovalRequired:
    """Direct commands still require CONFIRM_PROMPT human approval."""

    @pytest.mark.asyncio
    async def test_direct_command_sends_confirm_prompt(
        self, tmp_path: Path,
    ) -> None:
        """A direct command still sends CONFIRM_PROMPT to the CLI."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        envelope = _make_request(payload={
            "verb": "run",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "pytest -v tests/",
        })

        await handler.handle_message(envelope, client)

        # Writer was called (CONFIRM_PROMPT sent)
        assert client.writer.write.called

    @pytest.mark.asyncio
    async def test_direct_command_denied_returns_denied(
        self, tmp_path: Path,
    ) -> None:
        """User denial of a direct command returns status='denied'."""
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
        assert response.payload["status"] == "denied"

    @pytest.mark.asyncio
    async def test_direct_command_approved_starts_run(
        self, tmp_path: Path,
    ) -> None:
        """Approved direct command starts a background SSH execution."""
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
            return_value=[],
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
            assert response.payload["status"] == "started"
            assert "run_id" in response.payload


# ---------------------------------------------------------------------------
# LLM not called for direct commands (latency preservation)
# ---------------------------------------------------------------------------


class TestNoLLMCallForDirectCommands:
    """Direct commands skip LLM translation, preserving v1.2 latency."""

    @pytest.mark.asyncio
    async def test_direct_command_skips_llm_translation(
        self, tmp_path: Path,
    ) -> None:
        """Direct command does NOT invoke _translate_via_llm."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        with patch.object(
            handler,
            "_translate_via_llm",
            new_callable=AsyncMock,
            return_value="should not be called",
        ) as mock_translate:
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "pytest -v tests/",
            })

            await handler.handle_message(envelope, client)

            # LLM translation was NOT called
            mock_translate.assert_not_called()

    @pytest.mark.asyncio
    async def test_nl_command_does_call_llm_translation(
        self, tmp_path: Path,
    ) -> None:
        """NL command DOES invoke _translate_via_llm (when no agent loop)."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

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

            await handler.handle_message(envelope, client)

            # LLM translation WAS called for NL input
            mock_translate.assert_called_once()


# ---------------------------------------------------------------------------
# Parametrized bypass for all major executable categories
# ---------------------------------------------------------------------------


class TestBypassParametrized:
    """Parametrized coverage for direct commands across all categories."""

    @pytest.mark.parametrize(
        "command,expected_executable",
        [
            ("pytest -v tests/", "pytest"),
            ("python3 -m pytest", "python3"),
            ("python -c 'import sys; print(sys.version)'", "python"),
            ("npm test", "npm"),
            ("npx jest --verbose", "npx"),
            ("cargo test --release", "cargo"),
            ("go test ./...", "go"),
            ("make test", "make"),
            ("gradle test", "gradle"),
            ("./gradlew test", "gradlew"),
            ("bash run_tests.sh", "bash"),
            ("sh -c 'pytest -v'", "sh"),
            ("docker run --rm test-image", "docker"),
            ("kubectl get pods", "kubectl"),
            ("ls -la /opt/app", "ls"),
            ("cat /var/log/app.log", "cat"),
            ("grep -r FAIL tests/", "grep"),
            ("git status", "git"),
            ("pip install -r requirements.txt", "pip"),
            ("/usr/bin/python3 test.py", "python3"),
            ("./custom_script.sh", "custom_script.sh"),
        ],
    )
    def test_direct_command_bypass_flag(
        self, command: str, expected_executable: str,
    ) -> None:
        """Each direct command is detected with correct executable."""
        detection = detect_direct_command(command)
        assert detection.bypass_agent_loop is True
        assert detection.executable == expected_executable

    @pytest.mark.parametrize(
        "nl_input",
        [
            "run the smoke tests on staging",
            "can you check what's running?",
            "please execute the integration suite",
            "I need to see the test results",
            "deploy the latest build",
            "what tests failed yesterday?",
        ],
    )
    def test_nl_input_does_not_bypass(self, nl_input: str) -> None:
        """Natural language inputs do NOT trigger bypass."""
        detection = detect_direct_command(nl_input)
        assert detection.bypass_agent_loop is False
        assert detection.is_direct_command is False


# ---------------------------------------------------------------------------
# Legacy is_direct_command still works (backward compatibility)
# ---------------------------------------------------------------------------


class TestLegacyFunctionBackwardCompat:
    """Legacy is_direct_command is still exported and functional."""

    def test_legacy_function_still_available(self) -> None:
        """is_direct_command is still importable from request_handler."""
        assert callable(is_direct_command)

    @pytest.mark.parametrize(
        "command",
        ["pytest -v", "python3 test.py", "./run.sh", "make test"],
    )
    def test_legacy_detects_direct_commands(self, command: str) -> None:
        """Legacy function still detects basic direct commands."""
        assert is_direct_command(command) is True

    @pytest.mark.parametrize(
        "nl_input",
        ["run the tests", "what's running?"],
    )
    def test_legacy_rejects_nl(self, nl_input: str) -> None:
        """Legacy function still rejects NL inputs."""
        assert is_direct_command(nl_input) is False
