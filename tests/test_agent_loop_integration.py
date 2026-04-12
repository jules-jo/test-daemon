"""Tests for agent loop integration in request_handler.py (AC 4).

Validates that the agent loop replaces _handle_run as the default path
for NL commands, with proper fallback to one-shot when:
  (a) LLM is not configured
  (b) Agent loop initialization fails
  (c) Explicit --one-shot flag is set

Test strategy:
  - Mock the agent loop infrastructure (AgentLoop, ToolRegistry) to
    isolate the routing logic from actual LLM calls.
  - Verify the dispatch decision (agent loop vs one-shot) based on
    config flags and command type.
  - Verify response envelope structure for all paths.
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
        client_id="test-client-001",
        reader=AsyncMock(spec=asyncio.StreamReader),
        writer=AsyncMock(spec=asyncio.StreamWriter),
        connected_at="2026-04-12T12:00:00Z",
    )


def _make_request(
    payload: dict[str, Any],
    msg_id: str = "req-001",
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
        msg_id="deny-001",
        timestamp="2026-04-12T12:00:01Z",
        payload={"approved": False},
    )
    deny_frame = encode_frame(deny_reply)
    header_bytes = deny_frame[:4]
    payload_bytes = deny_frame[4:]
    client.reader.readexactly = AsyncMock(
        side_effect=[header_bytes, payload_bytes]
    )


# ---------------------------------------------------------------------------
# is_direct_command tests (existing function, baseline validation)
# ---------------------------------------------------------------------------


class TestIsDirectCommand:
    """Verify direct command detection still works as expected."""

    def test_python_is_direct(self) -> None:
        assert is_direct_command("python3 -m pytest tests/") is True

    def test_pytest_is_direct(self) -> None:
        assert is_direct_command("pytest -v tests/") is True

    def test_relative_path_is_direct(self) -> None:
        assert is_direct_command("./run_tests.sh") is True

    def test_absolute_path_is_direct(self) -> None:
        assert is_direct_command("/usr/bin/pytest") is True

    def test_natural_language_is_not_direct(self) -> None:
        assert is_direct_command("run the smoke tests") is False

    def test_empty_is_not_direct(self) -> None:
        assert is_direct_command("") is False

    def test_ambiguous_nl_is_not_direct(self) -> None:
        assert is_direct_command("check all unit tests on staging") is False


# ---------------------------------------------------------------------------
# Routing: _can_use_agent_loop property
# ---------------------------------------------------------------------------


class TestCanUseAgentLoop:
    """Tests for the _can_use_agent_loop property."""

    def test_true_when_llm_configured(self, tmp_path: Path) -> None:
        """Agent loop is available when LLM client and config are set."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        assert handler._can_use_agent_loop is True

    def test_false_when_llm_client_missing(self, tmp_path: Path) -> None:
        """Agent loop is unavailable without an LLM client."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=None,
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        assert handler._can_use_agent_loop is False

    def test_false_when_llm_config_missing(self, tmp_path: Path) -> None:
        """Agent loop is unavailable without an LLM config."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=None,
        )
        handler = RequestHandler(config=config)
        assert handler._can_use_agent_loop is False

    def test_false_when_one_shot_forced(self, tmp_path: Path) -> None:
        """Agent loop is unavailable when one_shot flag is set."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
            one_shot=True,
        )
        handler = RequestHandler(config=config)
        assert handler._can_use_agent_loop is False

    def test_false_when_all_missing(self, tmp_path: Path) -> None:
        """Agent loop is unavailable when nothing is configured."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        assert handler._can_use_agent_loop is False


# ---------------------------------------------------------------------------
# Routing: direct commands always use one-shot path
# ---------------------------------------------------------------------------


class TestDirectCommandRouting:
    """Direct commands should always route to one-shot, even with agent loop."""

    @pytest.mark.asyncio
    async def test_direct_command_uses_oneshot(self, tmp_path: Path) -> None:
        """A direct command (e.g. pytest) routes to _handle_run_oneshot."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        envelope = _make_request(payload={
            "verb": "run",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "pytest -v tests/",
        })

        response = await handler.handle_message(envelope, client)

        # Direct command should go through oneshot path
        # With deny reply, result is "denied"
        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "run"
        assert response.payload["status"] == "denied"


# ---------------------------------------------------------------------------
# Routing: NL commands use agent loop when available
# ---------------------------------------------------------------------------


class TestNLCommandAgentLoopRouting:
    """NL commands should route to agent loop when LLM is configured."""

    @pytest.mark.asyncio
    async def test_nl_command_uses_agent_loop(self, tmp_path: Path) -> None:
        """NL command routes to _handle_run_agent_loop when available."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()

        # Mock the agent loop to return a successful result
        mock_result = MagicMock()
        mock_result.final_state = MagicMock()
        mock_result.final_state.value = "complete"
        mock_result.iterations_used = 2
        mock_result.error_message = None

        # Import the actual state enum for comparison
        from jules_daemon.agent.agent_loop import AgentLoopState
        mock_result.final_state = AgentLoopState.COMPLETE

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
        ) as mock_agent:
            mock_agent.return_value = MessageEnvelope(
                msg_type=MessageType.RESPONSE,
                msg_id="req-001",
                timestamp="2026-04-12T12:00:00Z",
                payload={
                    "verb": "run",
                    "status": "completed",
                    "mode": "agent_loop",
                },
            )

            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run the smoke tests",
            })

            response = await handler.handle_message(envelope, client)

            mock_agent.assert_called_once()
            assert response.payload["mode"] == "agent_loop"

    @pytest.mark.asyncio
    async def test_nl_command_falls_back_on_agent_failure(
        self, tmp_path: Path,
    ) -> None:
        """NL command returns ERROR when agent loop raises a non-retry error."""
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
            side_effect=RuntimeError("Agent loop init failed"),
        ):
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run the smoke tests",
            })

            response = await handler.handle_message(envelope, client)

            # Non-RetryExhaustedError returns an ERROR response
            assert response.msg_type == MessageType.ERROR
            assert "Agent loop error" in response.payload["error"]


# ---------------------------------------------------------------------------
# Routing: NL commands use one-shot when LLM not configured
# ---------------------------------------------------------------------------


class TestNLCommandOneshotFallback:
    """NL commands should use one-shot when agent loop is unavailable."""

    @pytest.mark.asyncio
    async def test_no_llm_uses_oneshot(self, tmp_path: Path) -> None:
        """Without LLM, NL command goes through one-shot path."""
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

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "run"
        assert response.payload["status"] == "denied"

    @pytest.mark.asyncio
    async def test_one_shot_flag_forces_oneshot(self, tmp_path: Path) -> None:
        """Explicit one_shot flag forces one-shot even with LLM."""
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
            one_shot=True,
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        # Mock the agent loop to verify it's NOT called
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


# ---------------------------------------------------------------------------
# Config: max_agent_iterations
# ---------------------------------------------------------------------------


class TestAgentLoopConfig:
    """Tests for agent loop configuration on RequestHandlerConfig."""

    def test_default_max_iterations(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        assert config.max_agent_iterations == 15

    def test_custom_max_iterations(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            max_agent_iterations=10,
        )
        assert config.max_agent_iterations == 10

    def test_one_shot_default_false(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        assert config.one_shot is False


# ---------------------------------------------------------------------------
# Agent loop response handling
# ---------------------------------------------------------------------------


class TestAgentLoopResponseHandling:
    """Tests for how agent loop results are translated to envelopes."""

    @pytest.mark.asyncio
    async def test_agent_loop_complete_returns_completed(
        self, tmp_path: Path,
    ) -> None:
        """Agent loop COMPLETE state returns 'completed' status."""
        from jules_daemon.agent.agent_loop import (
            AgentLoopResult,
            AgentLoopState,
        )

        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()

        mock_result = AgentLoopResult(
            final_state=AgentLoopState.COMPLETE,
            iterations_used=2,
            history=(
                {"role": "system", "content": "..."},
                {"role": "user", "content": "run tests"},
            ),
            error_message=None,
        )

        with patch(
            "jules_daemon.agent.agent_loop.AgentLoop",
        ) as MockAgentLoop:
            mock_loop_instance = AsyncMock()
            mock_loop_instance.run.return_value = mock_result
            MockAgentLoop.return_value = mock_loop_instance

            with patch(
                "jules_daemon.agent.tools.registry_factory.build_tool_set",
            ) as mock_build:
                mock_build.return_value = ()

                with patch(
                    "jules_daemon.agent.llm_adapter.OpenAILLMAdapter",
                ):
                    with patch(
                        "jules_daemon.agent.tool_registry.ToolRegistry",
                    ) as MockRegistry:
                        mock_reg = MagicMock()
                        mock_reg.to_openai_schemas.return_value = ()
                        mock_reg.list_tool_names.return_value = ()
                        MockRegistry.return_value = mock_reg

                        envelope = _make_request(payload={
                            "verb": "run",
                            "target_host": "staging.example.com",
                            "target_user": "deploy",
                            "natural_language": "run the smoke tests",
                        })

                        response = await handler.handle_message(
                            envelope, client,
                        )

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["status"] == "completed"
        assert response.payload["mode"] == "agent_loop"
        assert response.payload["iterations_used"] == 2

    @pytest.mark.asyncio
    async def test_agent_loop_denied_returns_denied(
        self, tmp_path: Path,
    ) -> None:
        """Agent loop ERROR with 'denied' message returns denied status."""
        from jules_daemon.agent.agent_loop import (
            AgentLoopResult,
            AgentLoopState,
        )

        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()

        mock_result = AgentLoopResult(
            final_state=AgentLoopState.ERROR,
            iterations_used=1,
            history=(
                {"role": "system", "content": "..."},
                {"role": "user", "content": "run tests"},
            ),
            error_message="Tool 'propose_ssh_command' was denied: User denied",
        )

        with patch(
            "jules_daemon.agent.agent_loop.AgentLoop",
        ) as MockAgentLoop:
            mock_loop_instance = AsyncMock()
            mock_loop_instance.run.return_value = mock_result
            MockAgentLoop.return_value = mock_loop_instance

            with patch(
                "jules_daemon.agent.tools.registry_factory.build_tool_set",
            ) as mock_build:
                mock_build.return_value = ()

                with patch(
                    "jules_daemon.agent.llm_adapter.OpenAILLMAdapter",
                ):
                    with patch(
                        "jules_daemon.agent.tool_registry.ToolRegistry",
                    ) as MockRegistry:
                        mock_reg = MagicMock()
                        mock_reg.to_openai_schemas.return_value = ()
                        mock_reg.list_tool_names.return_value = ()
                        MockRegistry.return_value = mock_reg

                        envelope = _make_request(payload={
                            "verb": "run",
                            "target_host": "staging.example.com",
                            "target_user": "deploy",
                            "natural_language": "run the smoke tests",
                        })

                        response = await handler.handle_message(
                            envelope, client,
                        )

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["status"] == "denied"
        assert response.payload["mode"] == "agent_loop"

    @pytest.mark.asyncio
    async def test_agent_loop_error_returns_agent_error(
        self, tmp_path: Path,
    ) -> None:
        """Agent loop ERROR without 'denied' returns agent_error status."""
        from jules_daemon.agent.agent_loop import (
            AgentLoopResult,
            AgentLoopState,
        )

        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()

        mock_result = AgentLoopResult(
            final_state=AgentLoopState.ERROR,
            iterations_used=5,
            history=(
                {"role": "system", "content": "..."},
                {"role": "user", "content": "run tests"},
            ),
            error_message="Agent loop reached max iterations (5)",
        )

        with patch(
            "jules_daemon.agent.agent_loop.AgentLoop",
        ) as MockAgentLoop:
            mock_loop_instance = AsyncMock()
            mock_loop_instance.run.return_value = mock_result
            MockAgentLoop.return_value = mock_loop_instance

            with patch(
                "jules_daemon.agent.tools.registry_factory.build_tool_set",
            ) as mock_build:
                mock_build.return_value = ()

                with patch(
                    "jules_daemon.agent.llm_adapter.OpenAILLMAdapter",
                ):
                    with patch(
                        "jules_daemon.agent.tool_registry.ToolRegistry",
                    ) as MockRegistry:
                        mock_reg = MagicMock()
                        mock_reg.to_openai_schemas.return_value = ()
                        mock_reg.list_tool_names.return_value = ()
                        MockRegistry.return_value = mock_reg

                        envelope = _make_request(payload={
                            "verb": "run",
                            "target_host": "staging.example.com",
                            "target_user": "deploy",
                            "natural_language": "run the smoke tests",
                        })

                        response = await handler.handle_message(
                            envelope, client,
                        )

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["status"] == "agent_error"
        assert response.payload["mode"] == "agent_loop"
        assert "max iterations" in response.payload["error"]


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


class TestBuildAgentSystemPrompt:
    """Tests for _build_agent_system_prompt."""

    def test_includes_host_context(self, tmp_path: Path) -> None:
        """System prompt includes the SSH target information."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)

        prompt = handler._build_agent_system_prompt(
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )

        assert "staging.example.com" in prompt
        assert "deploy" in prompt
        assert "22" in prompt

    def test_includes_rules(self, tmp_path: Path) -> None:
        """System prompt includes behavioral rules."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)

        prompt = handler._build_agent_system_prompt(
            target_host="host",
            target_user="user",
            target_port=22,
        )

        assert "propose_ssh_command" in prompt
        assert "execute_ssh" in prompt
        assert "approval" in prompt.lower()
        assert "ask_user_question" in prompt

    def test_includes_wiki_context_when_available(
        self, tmp_path: Path,
    ) -> None:
        """System prompt includes wiki context when available."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)

        with patch.object(
            handler,
            "_build_wiki_context",
            return_value=["### USER-CORRECTED COMMANDS", "- 'test' -> `pytest`"],
        ):
            prompt = handler._build_agent_system_prompt(
                target_host="host",
                target_user="user",
                target_port=22,
            )

            assert "USER-CORRECTED COMMANDS" in prompt
            assert "Past Commands" in prompt


# ---------------------------------------------------------------------------
# Backward compatibility: all existing verbs still work
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _RegistryDispatcher adapter
# ---------------------------------------------------------------------------


class TestRegistryDispatcher:
    """Tests for the ToolRegistry -> ToolDispatcher adapter."""

    @pytest.mark.asyncio
    async def test_dispatch_delegates_to_execute(self) -> None:
        """dispatch() calls registry.execute()."""
        from jules_daemon.ipc.request_handler import _RegistryDispatcher

        mock_registry = AsyncMock()
        mock_result = MagicMock()
        mock_registry.execute.return_value = mock_result

        dispatcher = _RegistryDispatcher(mock_registry)
        mock_call = MagicMock()

        result = await dispatcher.dispatch(mock_call)

        mock_registry.execute.assert_called_once_with(mock_call)
        assert result is mock_result


# ---------------------------------------------------------------------------
# Backward compatibility: all existing verbs still work
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure the refactor does not break existing verb handlers."""

    @pytest.mark.asyncio
    async def test_status_still_works(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "status"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "status"

    @pytest.mark.asyncio
    async def test_queue_still_works(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={
            "verb": "queue",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run tests",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["status"] == "enqueued"

    @pytest.mark.asyncio
    async def test_cancel_still_works(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "cancel"})

        response = await handler.handle_message(envelope, client)

        # No task running -> error is expected
        assert response.msg_type == MessageType.ERROR

    @pytest.mark.asyncio
    async def test_history_still_works(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "history"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "history"

    @pytest.mark.asyncio
    async def test_run_denied_still_works_without_llm(
        self, tmp_path: Path,
    ) -> None:
        """The v1.2-mvp run denial flow still works identically."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        envelope = _make_request(payload={
            "verb": "run",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run the full regression suite",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "run"
        assert response.payload["status"] == "denied"
