"""Tests for the high-level translate_command LLM call + parse pipeline.

Covers:
    - Successful end-to-end flow: prompt -> LLM call -> parse -> SSHCommand
    - Handling of LLM refusal responses
    - Propagation of LLM client errors (connection, auth, etc.)
    - Propagation of parse errors from malformed responses
    - ChatCompletion response extraction (content from choices)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from jules_daemon.llm.config import LLMConfig
from jules_daemon.llm.errors import (
    LLMConnectionError,
    LLMParseError,
)
from jules_daemon.llm.prompts import HostContext
from jules_daemon.llm.response_parser import (
    Confidence,
    LLMCommandResponse,
    TranslateResult,
    translate_command,
)


# ---------------------------------------------------------------------------
# Helpers: build fake ChatCompletion responses
# ---------------------------------------------------------------------------


def _make_config() -> LLMConfig:
    return LLMConfig(
        base_url="https://mesh.example.com/api/v1/",
        api_key="test-key-123",
        default_model="openai:conn:gpt-4",
    )


def _make_host_context() -> HostContext:
    return HostContext(hostname="staging.example.com", user="deploy")


def _make_completion(content: str) -> ChatCompletion:
    """Build a minimal ChatCompletion with the given message content."""
    return ChatCompletion(
        id="chatcmpl-test-123",
        created=1700000000,
        model="openai:conn:gpt-4",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                ),
            )
        ],
    )


def _make_valid_response_json(
    commands: list[dict[str, Any]] | None = None,
    explanation: str = "Running the test suite",
    confidence: str = "high",
    warnings: list[str] | None = None,
) -> str:
    """Build a valid JSON response string."""
    if commands is None:
        commands = [
            {
                "command": "cd /opt/app && pytest -v --tb=short",
                "description": "Run the full test suite with verbose output",
                "working_directory": "/opt/app",
                "timeout_seconds": 300,
            }
        ]
    return json.dumps({
        "commands": commands,
        "explanation": explanation,
        "confidence": confidence,
        "warnings": warnings or [],
    })


# ---------------------------------------------------------------------------
# TranslateResult model
# ---------------------------------------------------------------------------


class TestTranslateResult:
    """Tests for the TranslateResult container."""

    def test_create_with_commands(self) -> None:
        from jules_daemon.ssh.command import SSHCommand

        cmds = [SSHCommand(command="pytest -v")]
        resp = LLMCommandResponse(
            commands=[{"command": "pytest -v", "description": "Run tests"}],
            explanation="Running tests",
            confidence=Confidence.HIGH,
        )
        result = TranslateResult(
            ssh_commands=cmds,
            llm_response=resp,
            raw_content="the raw text",
        )
        assert len(result.ssh_commands) == 1
        assert result.llm_response.confidence is Confidence.HIGH
        assert result.raw_content == "the raw text"
        assert result.is_refusal is False

    def test_refusal_result(self) -> None:
        resp = LLMCommandResponse(
            commands=[],
            explanation="Cannot do this",
            confidence=Confidence.LOW,
        )
        result = TranslateResult(
            ssh_commands=[],
            llm_response=resp,
            raw_content="the raw text",
        )
        assert result.is_refusal is True


# ---------------------------------------------------------------------------
# translate_command() end-to-end
# ---------------------------------------------------------------------------


class TestTranslateCommand:
    """Tests for the translate_command function."""

    @patch("jules_daemon.llm.response_parser.create_completion")
    @patch("jules_daemon.llm.response_parser.create_client")
    def test_successful_translation(
        self,
        mock_create_client: MagicMock,
        mock_create_completion: MagicMock,
    ) -> None:
        """Full success path: prompt -> LLM -> parse -> SSHCommand."""
        response_json = _make_valid_response_json()
        mock_create_completion.return_value = _make_completion(response_json)
        mock_create_client.return_value = MagicMock()

        config = _make_config()
        host = _make_host_context()

        result = translate_command(
            natural_language="run the full test suite",
            host_context=host,
            config=config,
        )

        assert isinstance(result, TranslateResult)
        assert len(result.ssh_commands) == 1
        assert result.ssh_commands[0].command == "cd /opt/app && pytest -v --tb=short"
        assert result.ssh_commands[0].working_directory == "/opt/app"
        assert result.llm_response.confidence is Confidence.HIGH
        assert result.is_refusal is False

    @patch("jules_daemon.llm.response_parser.create_completion")
    @patch("jules_daemon.llm.response_parser.create_client")
    def test_refusal_translation(
        self,
        mock_create_client: MagicMock,
        mock_create_completion: MagicMock,
    ) -> None:
        """LLM refuses the request -- returns empty commands."""
        response_json = _make_valid_response_json(
            commands=[],
            explanation="Cannot delete system files",
            confidence="low",
            warnings=["Forbidden operation"],
        )
        mock_create_completion.return_value = _make_completion(response_json)
        mock_create_client.return_value = MagicMock()

        result = translate_command(
            natural_language="delete all system files",
            host_context=_make_host_context(),
            config=_make_config(),
        )

        assert result.is_refusal is True
        assert result.ssh_commands == []
        assert result.llm_response.warnings == ["Forbidden operation"]

    @patch("jules_daemon.llm.response_parser.create_completion")
    @patch("jules_daemon.llm.response_parser.create_client")
    def test_code_fenced_response(
        self,
        mock_create_client: MagicMock,
        mock_create_completion: MagicMock,
    ) -> None:
        """LLM wraps JSON in markdown code fence."""
        inner_json = _make_valid_response_json()
        content = f"Here is the plan:\n```json\n{inner_json}\n```"
        mock_create_completion.return_value = _make_completion(content)
        mock_create_client.return_value = MagicMock()

        result = translate_command(
            natural_language="run tests",
            host_context=_make_host_context(),
            config=_make_config(),
        )

        assert len(result.ssh_commands) == 1

    @patch("jules_daemon.llm.response_parser.create_completion")
    @patch("jules_daemon.llm.response_parser.create_client")
    def test_llm_connection_error_propagates(
        self,
        mock_create_client: MagicMock,
        mock_create_completion: MagicMock,
    ) -> None:
        """LLM client errors propagate to caller."""
        mock_create_completion.side_effect = LLMConnectionError("unreachable")
        mock_create_client.return_value = MagicMock()

        with pytest.raises(LLMConnectionError, match="unreachable"):
            translate_command(
                natural_language="run tests",
                host_context=_make_host_context(),
                config=_make_config(),
            )

    @patch("jules_daemon.llm.response_parser.create_completion")
    @patch("jules_daemon.llm.response_parser.create_client")
    def test_malformed_json_raises_parse_error(
        self,
        mock_create_client: MagicMock,
        mock_create_completion: MagicMock,
    ) -> None:
        """Malformed LLM output raises LLMParseError."""
        mock_create_completion.return_value = _make_completion(
            "I cannot parse this {invalid json"
        )
        mock_create_client.return_value = MagicMock()

        with pytest.raises(LLMParseError):
            translate_command(
                natural_language="run tests",
                host_context=_make_host_context(),
                config=_make_config(),
            )

    @patch("jules_daemon.llm.response_parser.create_completion")
    @patch("jules_daemon.llm.response_parser.create_client")
    def test_empty_content_raises_parse_error(
        self,
        mock_create_client: MagicMock,
        mock_create_completion: MagicMock,
    ) -> None:
        """Empty LLM response content raises LLMParseError."""
        completion = _make_completion("")
        # Override content to None to test null handling
        completion.choices[0].message.content = None
        mock_create_completion.return_value = completion
        mock_create_client.return_value = MagicMock()

        with pytest.raises(LLMParseError, match="empty"):
            translate_command(
                natural_language="run tests",
                host_context=_make_host_context(),
                config=_make_config(),
            )

    @patch("jules_daemon.llm.response_parser.create_completion")
    @patch("jules_daemon.llm.response_parser.create_client")
    def test_no_choices_raises_parse_error(
        self,
        mock_create_client: MagicMock,
        mock_create_completion: MagicMock,
    ) -> None:
        """ChatCompletion with no choices raises LLMParseError."""
        completion = _make_completion("whatever")
        completion.choices = []
        mock_create_completion.return_value = completion
        mock_create_client.return_value = MagicMock()

        with pytest.raises(LLMParseError, match="No choices"):
            translate_command(
                natural_language="run tests",
                host_context=_make_host_context(),
                config=_make_config(),
            )

    @patch("jules_daemon.llm.response_parser.create_completion")
    @patch("jules_daemon.llm.response_parser.create_client")
    def test_multiple_commands_translation(
        self,
        mock_create_client: MagicMock,
        mock_create_completion: MagicMock,
    ) -> None:
        """Multiple commands map to multiple SSHCommand objects."""
        response_json = _make_valid_response_json(
            commands=[
                {
                    "command": "cd /opt/app",
                    "description": "Navigate to app directory",
                },
                {
                    "command": "pytest -v --tb=short",
                    "description": "Run tests",
                    "working_directory": "/opt/app",
                    "timeout_seconds": 600,
                },
            ],
            explanation="Navigate and run",
        )
        mock_create_completion.return_value = _make_completion(response_json)
        mock_create_client.return_value = MagicMock()

        result = translate_command(
            natural_language="run tests",
            host_context=_make_host_context(),
            config=_make_config(),
        )

        assert len(result.ssh_commands) == 2
        assert result.ssh_commands[0].command == "cd /opt/app"
        assert result.ssh_commands[1].command == "pytest -v --tb=short"
        assert result.ssh_commands[1].timeout == 600

    @patch("jules_daemon.llm.response_parser.create_completion")
    @patch("jules_daemon.llm.response_parser.create_client")
    def test_passes_client_to_completion(
        self,
        mock_create_client: MagicMock,
        mock_create_completion: MagicMock,
    ) -> None:
        """Verifies the client is created and passed to create_completion."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_create_completion.return_value = _make_completion(
            _make_valid_response_json()
        )

        config = _make_config()
        translate_command(
            natural_language="run tests",
            host_context=_make_host_context(),
            config=config,
        )

        mock_create_client.assert_called_once_with(config)
        call_kwargs = mock_create_completion.call_args
        assert call_kwargs.kwargs["client"] is mock_client
        assert call_kwargs.kwargs["config"] is config

    @patch("jules_daemon.llm.response_parser.create_completion")
    @patch("jules_daemon.llm.response_parser.create_client")
    def test_raw_content_preserved(
        self,
        mock_create_client: MagicMock,
        mock_create_completion: MagicMock,
    ) -> None:
        """The raw LLM content is preserved in the result for audit."""
        response_json = _make_valid_response_json()
        mock_create_completion.return_value = _make_completion(response_json)
        mock_create_client.return_value = MagicMock()

        result = translate_command(
            natural_language="run tests",
            host_context=_make_host_context(),
            config=_make_config(),
        )

        assert result.raw_content == response_json

    @patch("jules_daemon.llm.response_parser.create_completion")
    @patch("jules_daemon.llm.response_parser.create_client")
    def test_existing_client_reused(
        self,
        mock_create_client: MagicMock,
        mock_create_completion: MagicMock,
    ) -> None:
        """When an existing client is passed, create_client is not called."""
        mock_client = MagicMock()
        mock_create_completion.return_value = _make_completion(
            _make_valid_response_json()
        )

        translate_command(
            natural_language="run tests",
            host_context=_make_host_context(),
            config=_make_config(),
            client=mock_client,
        )

        mock_create_client.assert_not_called()
        call_kwargs = mock_create_completion.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs["client"] is mock_client
