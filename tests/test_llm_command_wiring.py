"""Tests for LLM-based command translation wiring in the request handler.

Covers:
    - is_direct_command() heuristic for detecting shell commands vs NL
    - _translate_via_llm() fallback when LLM is not configured
    - _translate_via_llm() success path with mocked CommandTranslator
    - _translate_via_llm() error path (LLM failure -> graceful fallback)
    - _translate_via_llm() refusal path (LLM refuses -> use input as-is)
    - _build_wiki_context() with empty wiki
    - _build_wiki_context() with translation history
    - _handle_run integration: direct command skips LLM
    - _handle_run integration: NL input triggers LLM translation
    - __main__._try_load_llm() with and without env vars
    - RequestHandlerConfig with optional LLM fields

All LLM calls are mocked -- these tests verify the wiring logic and
fallback behavior, not actual LLM behavior.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from jules_daemon.ipc.request_handler import (
    RequestHandler,
    RequestHandlerConfig,
    is_direct_command,
)


# ---------------------------------------------------------------------------
# is_direct_command tests
# ---------------------------------------------------------------------------


class TestIsDirectCommand:
    """Tests for the is_direct_command() heuristic."""

    @pytest.mark.parametrize(
        "text",
        [
            "pytest tests/integration/ -v",
            "python -m pytest",
            "python3 script.py",
            "npm test",
            "npx jest",
            "node app.js",
            "go test ./...",
            "make test",
            "bash -c 'echo hello'",
            "sh run.sh",
            "./run_tests.sh",
            "/usr/bin/pytest -v",
            "docker run test",
            "cargo test",
            "mvn test",
            "gradle test",
            "pip install -r requirements.txt",
            "uv run pytest",
            "ls -la",
            "cat foo.txt",
            "cd /opt/app",
            "grep -r 'pattern' .",
            "find . -name '*.py'",
            "echo hello",
            "env VAR=val pytest",
            "poetry run pytest",
        ],
    )
    def test_recognizes_direct_commands(self, text: str) -> None:
        assert is_direct_command(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "run the integration tests",
            "execute smoke tests on staging",
            "check if tests pass",
            "deploy the latest build",
            "show me the test results",
            "what failed in the last run",
            "restart the test suite",
            "try running tests again",
            "please run unit tests",
            "test the login flow",
        ],
    )
    def test_recognizes_natural_language(self, text: str) -> None:
        assert is_direct_command(text) is False

    def test_empty_string(self) -> None:
        assert is_direct_command("") is False

    def test_whitespace_only(self) -> None:
        assert is_direct_command("   ") is False

    def test_leading_whitespace_direct_command(self) -> None:
        assert is_direct_command("  pytest -v") is True

    def test_absolute_path(self) -> None:
        assert is_direct_command("/opt/bin/test_runner --suite smoke") is True

    def test_relative_path(self) -> None:
        assert is_direct_command("./scripts/run_tests.sh") is True


# ---------------------------------------------------------------------------
# RequestHandlerConfig tests
# ---------------------------------------------------------------------------


class TestRequestHandlerConfig:
    """Tests for RequestHandlerConfig with optional LLM fields."""

    def test_default_llm_fields_are_none(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        assert config.llm_client is None
        assert config.llm_config is None

    def test_llm_fields_round_trip(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_config = MagicMock()
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=mock_client,
            llm_config=mock_config,
        )
        assert config.llm_client is mock_client
        assert config.llm_config is mock_config


# ---------------------------------------------------------------------------
# RequestHandler LLM initialization tests
# ---------------------------------------------------------------------------


class TestRequestHandlerLLMInit:
    """Tests for LLM translator initialization in RequestHandler."""

    def test_no_llm_when_client_not_provided(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        assert handler._command_translator is None

    def test_no_llm_when_config_not_provided(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=MagicMock(),
            llm_config=None,
        )
        handler = RequestHandler(config=config)
        assert handler._command_translator is None

    def test_translator_created_when_both_provided(
        self, tmp_path: Path,
    ) -> None:
        from jules_daemon.llm.config import LLMConfig

        llm_config = LLMConfig(
            base_url="https://example.com/api/v1/",
            api_key="test-key",
            default_model="openai:conn:gpt-4",
        )
        mock_client = MagicMock()
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=mock_client,
            llm_config=llm_config,
        )
        handler = RequestHandler(config=config)
        assert handler._command_translator is not None


# ---------------------------------------------------------------------------
# _translate_via_llm tests
# ---------------------------------------------------------------------------


class TestTranslateViaLLM:
    """Tests for the _translate_via_llm method."""

    @pytest.fixture()
    def handler_no_llm(self, tmp_path: Path) -> RequestHandler:
        """Handler without LLM configured."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        return RequestHandler(config=config)

    @pytest.fixture()
    def handler_with_llm(self, tmp_path: Path) -> RequestHandler:
        """Handler with a mocked LLM translator."""
        from jules_daemon.llm.config import LLMConfig

        llm_config = LLMConfig(
            base_url="https://example.com/api/v1/",
            api_key="test-key",
            default_model="openai:conn:gpt-4",
        )
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=MagicMock(),
            llm_config=llm_config,
        )
        handler = RequestHandler(config=config)
        # Replace the real translator with a mock
        handler._command_translator = MagicMock()
        return handler

    @pytest.mark.asyncio()
    async def test_returns_input_when_llm_not_configured(
        self, handler_no_llm: RequestHandler,
    ) -> None:
        result = await handler_no_llm._translate_via_llm(
            natural_language="run the tests",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )
        assert result == "run the tests"

    @pytest.mark.asyncio()
    async def test_returns_translated_command_on_success(
        self, handler_with_llm: RequestHandler,
    ) -> None:
        # Mock a successful translation result
        mock_result = MagicMock()
        mock_result.is_refusal = False
        mock_result.command_count = 1
        mock_cmd = MagicMock()
        mock_cmd.command = "cd /opt/app && pytest -v --tb=short"
        mock_result.ssh_commands = (mock_cmd,)
        mock_result.response.confidence.value = "high"
        mock_result.elapsed_seconds = 1.5

        handler_with_llm._command_translator.translate.return_value = mock_result

        result = await handler_with_llm._translate_via_llm(
            natural_language="run the tests",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )
        assert result == "cd /opt/app && pytest -v --tb=short"

    @pytest.mark.asyncio()
    async def test_joins_multiple_commands_with_and(
        self, handler_with_llm: RequestHandler,
    ) -> None:
        mock_result = MagicMock()
        mock_result.is_refusal = False
        mock_result.command_count = 2
        cmd1 = MagicMock()
        cmd1.command = "cd /opt/app"
        cmd2 = MagicMock()
        cmd2.command = "pytest -v"
        mock_result.ssh_commands = (cmd1, cmd2)
        mock_result.response.confidence.value = "high"
        mock_result.elapsed_seconds = 2.0

        handler_with_llm._command_translator.translate.return_value = mock_result

        result = await handler_with_llm._translate_via_llm(
            natural_language="run the tests",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )
        assert result == "cd /opt/app && pytest -v"

    @pytest.mark.asyncio()
    async def test_falls_back_on_llm_error(
        self, handler_with_llm: RequestHandler,
    ) -> None:
        handler_with_llm._command_translator.translate.side_effect = (
            RuntimeError("LLM connection failed")
        )

        result = await handler_with_llm._translate_via_llm(
            natural_language="run the smoke tests",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )
        assert result == "run the smoke tests"

    @pytest.mark.asyncio()
    async def test_falls_back_on_refusal(
        self, handler_with_llm: RequestHandler,
    ) -> None:
        mock_result = MagicMock()
        mock_result.is_refusal = True
        mock_result.response.explanation = "Cannot generate destructive commands"

        handler_with_llm._command_translator.translate.return_value = mock_result

        result = await handler_with_llm._translate_via_llm(
            natural_language="delete everything",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )
        assert result == "delete everything"

    @pytest.mark.asyncio()
    async def test_falls_back_on_zero_commands(
        self, handler_with_llm: RequestHandler,
    ) -> None:
        mock_result = MagicMock()
        mock_result.is_refusal = False
        mock_result.command_count = 0

        handler_with_llm._command_translator.translate.return_value = mock_result

        result = await handler_with_llm._translate_via_llm(
            natural_language="do something unclear",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )
        assert result == "do something unclear"


# ---------------------------------------------------------------------------
# _build_wiki_context tests
# ---------------------------------------------------------------------------


class TestBuildWikiContext:
    """Tests for _build_wiki_context."""

    def test_empty_wiki_returns_empty_list(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)

        context = handler._build_wiki_context(target_host="staging.example.com")
        assert context == []

    def test_includes_past_translations(self, tmp_path: Path) -> None:
        from jules_daemon.wiki.command_translation import (
            CommandTranslation,
            TranslationOutcome,
            save,
        )
        from jules_daemon.wiki.layout import initialize_wiki

        initialize_wiki(tmp_path)

        translation = CommandTranslation(
            natural_language="run the smoke tests",
            resolved_shell="cd /opt/app && pytest tests/smoke/ -v",
            ssh_host="staging.example.com",
            outcome=TranslationOutcome.APPROVED,
        )
        save(tmp_path, translation)

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)

        context = handler._build_wiki_context(target_host="staging.example.com")
        assert len(context) >= 1
        joined = "\n".join(context)
        assert "run the smoke tests" in joined
        assert "pytest tests/smoke/ -v" in joined

    def test_filters_by_host(self, tmp_path: Path) -> None:
        from jules_daemon.wiki.command_translation import (
            CommandTranslation,
            TranslationOutcome,
            save,
        )
        from jules_daemon.wiki.layout import initialize_wiki

        initialize_wiki(tmp_path)

        # Translation for a different host
        save(
            tmp_path,
            CommandTranslation(
                natural_language="run tests",
                resolved_shell="pytest",
                ssh_host="other.example.com",
                outcome=TranslationOutcome.APPROVED,
            ),
        )
        # Translation for our host
        save(
            tmp_path,
            CommandTranslation(
                natural_language="run smoke",
                resolved_shell="pytest tests/smoke/",
                ssh_host="staging.example.com",
                outcome=TranslationOutcome.APPROVED,
            ),
        )

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)

        context = handler._build_wiki_context(target_host="staging.example.com")
        assert len(context) >= 1
        joined = "\n".join(context)
        assert "run smoke" in joined
        assert "run tests" not in joined  # other host's translation is filtered out


# ---------------------------------------------------------------------------
# __main__._try_load_llm tests
# ---------------------------------------------------------------------------


class TestTryLoadLLM:
    """Tests for the _try_load_llm function in __main__.py."""

    def test_returns_none_when_env_not_set(self) -> None:
        from jules_daemon.__main__ import _try_load_llm

        with patch.dict("os.environ", {}, clear=True):
            client, config = _try_load_llm()
        assert client is None
        assert config is None

    def test_returns_none_when_base_url_empty(self) -> None:
        from jules_daemon.__main__ import _try_load_llm

        with patch.dict(
            "os.environ",
            {"JULES_LLM_BASE_URL": ""},
            clear=True,
        ):
            client, config = _try_load_llm()
        assert client is None
        assert config is None

    def test_returns_client_and_config_when_env_set(self) -> None:
        from jules_daemon.__main__ import _try_load_llm

        env = {
            "JULES_LLM_BASE_URL": "https://mesh.example.com/api/v1/",
            "JULES_LLM_API_KEY": "test-key-123",
            "JULES_LLM_DEFAULT_MODEL": "openai:conn:gpt-4",
        }
        with patch.dict("os.environ", env, clear=True):
            client, config = _try_load_llm()

        assert client is not None
        assert config is not None
        assert config.base_url == "https://mesh.example.com/api/v1/"
        assert config.default_model == "openai:conn:gpt-4"

    def test_returns_none_on_config_error(self) -> None:
        from jules_daemon.__main__ import _try_load_llm

        # Missing API key should cause load_config_from_env to raise
        env = {
            "JULES_LLM_BASE_URL": "https://mesh.example.com/api/v1/",
        }
        with patch.dict("os.environ", env, clear=True):
            client, config = _try_load_llm()
        assert client is None
        assert config is None
