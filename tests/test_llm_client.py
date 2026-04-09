"""Tests for the Dataiku Mesh LLM client wrapper."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI

from jules_daemon.llm.client import (
    create_client,
    create_completion,
)
from jules_daemon.llm.config import LLMConfig
from jules_daemon.llm.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    LLMToolCallingUnsupportedError,
)
from jules_daemon.llm.models import ToolCallingMode


def _make_config(**overrides: Any) -> LLMConfig:
    """Create a test config with sensible defaults."""
    defaults = {
        "base_url": "https://dss.example.com/public/api/projects/PROJ/llms/openai/v1/",
        "api_key": "dkuapi_test123",
        "default_model": "openai:my-conn:gpt-4",
    }
    return LLMConfig(**{**defaults, **overrides})


class TestCreateClient:
    """Tests for create_client factory function."""

    def test_returns_openai_client(self) -> None:
        config = _make_config()
        client = create_client(config)
        assert isinstance(client, OpenAI)

    def test_base_url_configured(self) -> None:
        config = _make_config(
            base_url="https://dss.example.com/public/api/projects/PROJ/llms/openai/v1/"
        )
        client = create_client(config)
        assert str(client.base_url).rstrip("/") == config.base_url.rstrip("/")

    def test_api_key_configured(self) -> None:
        config = _make_config(api_key="dkuapi_secretkey")
        client = create_client(config)
        assert client.api_key == "dkuapi_secretkey"

    def test_max_retries_configured(self) -> None:
        config = _make_config(max_retries=5)
        client = create_client(config)
        assert client.max_retries == 5

    def test_timeout_configured(self) -> None:
        config = _make_config(timeout=30.0)
        client = create_client(config)
        # OpenAI SDK may store timeout as float or httpx.Timeout
        timeout = client.timeout
        if hasattr(timeout, "read"):
            assert timeout.read == 30.0
        else:
            assert timeout == 30.0

    def test_ssl_verification_disabled(self) -> None:
        """When verify_ssl=False, the client is created without SSL verification."""
        config = _make_config(verify_ssl=False)
        # Should not raise even with verify_ssl=False
        client = create_client(config)
        assert isinstance(client, OpenAI)

    def test_ssl_verification_enabled_by_default(self) -> None:
        """When verify_ssl=True (default), standard SSL verification is used."""
        config = _make_config(verify_ssl=True)
        client = create_client(config)
        assert isinstance(client, OpenAI)


class TestCreateCompletion:
    """Tests for create_completion wrapper function."""

    def _mock_choice(
        self,
        content: str = "Hello!",
        tool_calls: list[Any] | None = None,
    ) -> MagicMock:
        """Build a mock ChatCompletion.choices[0]."""
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls
        message.role = "assistant"

        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop" if not tool_calls else "tool_calls"
        return choice

    def _mock_response(
        self,
        content: str = "Hello!",
        tool_calls: list[Any] | None = None,
        model: str = "openai:my-conn:gpt-4",
    ) -> MagicMock:
        """Build a mock ChatCompletion response."""
        response = MagicMock()
        response.choices = [self._mock_choice(content, tool_calls)]
        response.model = model
        response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        return response

    def _sample_tools(self) -> list[dict[str, Any]]:
        """Return a sample tool definition list."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

    def test_basic_completion(self) -> None:
        config = _make_config()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            content="Test passed!"
        )

        result = create_completion(
            client=mock_client,
            config=config,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.choices[0].message.content == "Test passed!"
        mock_client.chat.completions.create.assert_called_once()

    def test_uses_default_model(self) -> None:
        config = _make_config(default_model="openai:my-conn:gpt-4")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()

        create_completion(
            client=mock_client,
            config=config,
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "openai:my-conn:gpt-4"

    def test_model_override(self) -> None:
        config = _make_config(default_model="openai:my-conn:gpt-4")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()

        create_completion(
            client=mock_client,
            config=config,
            messages=[{"role": "user", "content": "Hi"}],
            model="anthropic:my-claude:claude-3-opus",
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "anthropic:my-claude:claude-3-opus"

    def test_with_tools_native_mode(self) -> None:
        config = _make_config()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()

        tools = self._sample_tools()

        create_completion(
            client=mock_client,
            config=config,
            messages=[{"role": "user", "content": "weather?"}],
            tools=tools,
            tool_calling_mode=ToolCallingMode.NATIVE,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["tools"] == tools

    def test_with_tools_prompt_based_mode(self) -> None:
        """In prompt-based mode, tools are injected into the system message."""
        config = _make_config()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()

        tools = self._sample_tools()

        create_completion(
            client=mock_client,
            config=config,
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "weather in NYC?"},
            ],
            tools=tools,
            tool_calling_mode=ToolCallingMode.PROMPT_BASED,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # In prompt-based mode, tools should NOT be in the API call
        assert "tools" not in call_kwargs
        # But the system message should contain tool descriptions
        messages = call_kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "get_weather" in system_msg["content"]

    def test_prompt_based_mode_no_existing_system_message(self) -> None:
        """Prompt-based mode should prepend a system message when none exists."""
        config = _make_config()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()

        tools = self._sample_tools()

        create_completion(
            client=mock_client,
            config=config,
            messages=[{"role": "user", "content": "weather in NYC?"}],
            tools=tools,
            tool_calling_mode=ToolCallingMode.PROMPT_BASED,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "get_weather" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_empty_tools_list_not_passed(self) -> None:
        """An empty tools list should not add a 'tools' key to the API call."""
        config = _make_config()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()

        create_completion(
            client=mock_client,
            config=config,
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs

    def test_excludes_unsupported_params(self) -> None:
        """Dataiku Mesh does not support parallel_tool_calls, n, seed, response_format."""
        config = _make_config()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()

        create_completion(
            client=mock_client,
            config=config,
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        for unsupported in ("parallel_tool_calls", "n", "seed", "response_format"):
            assert unsupported not in call_kwargs

    def test_supported_extra_kwargs_passed_through(self) -> None:
        """Supported extra kwargs like temperature should pass through."""
        config = _make_config()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()

        create_completion(
            client=mock_client,
            config=config,
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=500,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 500

    def test_unsupported_extra_kwargs_filtered_out(self) -> None:
        """Unsupported Dataiku params should be silently filtered."""
        config = _make_config()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()

        create_completion(
            client=mock_client,
            config=config,
            messages=[{"role": "user", "content": "Hi"}],
            parallel_tool_calls=True,
            seed=42,
            response_format={"type": "json_object"},
            n=3,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "parallel_tool_calls" not in call_kwargs
        assert "seed" not in call_kwargs
        assert "response_format" not in call_kwargs
        assert "n" not in call_kwargs

    def test_original_messages_not_mutated(self) -> None:
        """create_completion should not mutate the original messages list."""
        config = _make_config()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()

        original_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        original_content = original_messages[0]["content"]

        create_completion(
            client=mock_client,
            config=config,
            messages=original_messages,
            tools=self._sample_tools(),
            tool_calling_mode=ToolCallingMode.PROMPT_BASED,
        )

        # Original should be unchanged
        assert original_messages[0]["content"] == original_content

    def test_auth_error_wrapping(self) -> None:
        """Authentication errors from the SDK should be wrapped."""
        import openai

        config = _make_config()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="Invalid API key",
            response=mock_response,
            body=None,
        )

        with pytest.raises(LLMAuthenticationError, match="Invalid API key"):
            create_completion(
                client=mock_client,
                config=config,
                messages=[{"role": "user", "content": "Hi"}],
            )

    def test_connection_error_wrapping(self) -> None:
        """Connection errors should be wrapped in LLMConnectionError."""
        import openai

        config = _make_config()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock(),
        )

        with pytest.raises(LLMConnectionError, match="connection"):
            create_completion(
                client=mock_client,
                config=config,
                messages=[{"role": "user", "content": "Hi"}],
            )

    def test_connection_error_does_not_leak_base_url(self) -> None:
        """Connection error message should not contain the base URL."""
        import openai

        config = _make_config(
            base_url="https://internal-dss.corp.example.com/v1/"
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock(),
        )

        with pytest.raises(LLMConnectionError) as exc_info:
            create_completion(
                client=mock_client,
                config=config,
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert "internal-dss.corp.example.com" not in str(exc_info.value)

    def test_api_error_wrapping(self) -> None:
        """Generic API errors should be wrapped in LLMResponseError."""
        import openai

        config = _make_config()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_client.chat.completions.create.side_effect = openai.APIStatusError(
            message="Internal server error",
            response=mock_response,
            body=None,
        )

        with pytest.raises(LLMResponseError, match="API error"):
            create_completion(
                client=mock_client,
                config=config,
                messages=[{"role": "user", "content": "Hi"}],
            )

    def test_generic_openai_error_wrapping(self) -> None:
        """Catch-all for unexpected OpenAI errors."""
        import openai

        config = _make_config()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.OpenAIError(
            "Something unexpected"
        )

        with pytest.raises(LLMError, match="Unexpected LLM error"):
            create_completion(
                client=mock_client,
                config=config,
                messages=[{"role": "user", "content": "Hi"}],
            )
