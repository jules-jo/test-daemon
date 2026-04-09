"""Tests for LLM configuration loading and validation."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from jules_daemon.llm.config import (
    LLMConfig,
    load_config,
    load_config_from_env,
)


class TestLLMConfig:
    """Tests for the frozen LLMConfig dataclass."""

    def test_create_with_required_fields(self) -> None:
        config = LLMConfig(
            base_url="https://dss.example.com/public/api/projects/PROJ/llms/openai/v1/",
            api_key="dkuapi_abc123",
            default_model="openai:my-conn:gpt-4",
        )
        assert config.base_url == "https://dss.example.com/public/api/projects/PROJ/llms/openai/v1/"
        assert config.api_key == "dkuapi_abc123"
        assert config.default_model == "openai:my-conn:gpt-4"

    def test_defaults(self) -> None:
        config = LLMConfig(
            base_url="https://dss.example.com/v1/",
            api_key="key",
            default_model="openai:conn:model",
        )
        assert config.timeout == 120.0
        assert config.max_retries == 2
        assert config.verify_ssl is True

    def test_custom_timeout(self) -> None:
        config = LLMConfig(
            base_url="https://x/v1/",
            api_key="k",
            default_model="m",
            timeout=60.0,
        )
        assert config.timeout == 60.0

    def test_custom_max_retries(self) -> None:
        config = LLMConfig(
            base_url="https://x/v1/",
            api_key="k",
            default_model="m",
            max_retries=5,
        )
        assert config.max_retries == 5

    def test_disable_ssl_verification(self) -> None:
        config = LLMConfig(
            base_url="https://x/v1/",
            api_key="k",
            default_model="m",
            verify_ssl=False,
        )
        assert config.verify_ssl is False

    def test_frozen(self) -> None:
        config = LLMConfig(
            base_url="https://x/v1/",
            api_key="k",
            default_model="m",
        )
        with pytest.raises(AttributeError):
            config.api_key = "new_key"  # type: ignore[misc]

    def test_empty_base_url_raises(self) -> None:
        with pytest.raises(ValueError, match="base_url must not be empty"):
            LLMConfig(base_url="", api_key="k", default_model="m")

    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key must not be empty"):
            LLMConfig(base_url="https://x/v1/", api_key="", default_model="m")

    def test_empty_default_model_raises(self) -> None:
        with pytest.raises(ValueError, match="default_model must not be empty"):
            LLMConfig(base_url="https://x/v1/", api_key="k", default_model="")

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout must be positive"):
            LLMConfig(
                base_url="https://x/v1/",
                api_key="k",
                default_model="m",
                timeout=-1.0,
            )

    def test_negative_max_retries_raises(self) -> None:
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            LLMConfig(
                base_url="https://x/v1/",
                api_key="k",
                default_model="m",
                max_retries=-1,
            )

    def test_base_url_trailing_slash_preserved(self) -> None:
        config = LLMConfig(
            base_url="https://x/v1/",
            api_key="k",
            default_model="m",
        )
        assert config.base_url == "https://x/v1/"


class TestLoadConfig:
    """Tests for load_config factory function."""

    def test_load_with_explicit_values(self) -> None:
        config = load_config(
            base_url="https://dss.example.com/public/api/projects/PROJ/llms/openai/v1/",
            api_key="dkuapi_test",
            default_model="openai:conn:gpt-4",
        )
        assert config.base_url == "https://dss.example.com/public/api/projects/PROJ/llms/openai/v1/"
        assert config.api_key == "dkuapi_test"
        assert config.default_model == "openai:conn:gpt-4"

    def test_load_with_all_optional_params(self) -> None:
        config = load_config(
            base_url="https://x/v1/",
            api_key="k",
            default_model="m",
            timeout=30.0,
            max_retries=5,
            verify_ssl=False,
        )
        assert config.timeout == 30.0
        assert config.max_retries == 5
        assert config.verify_ssl is False


class TestLoadConfigFromEnv:
    """Tests for environment-variable-based config loading."""

    def test_load_from_env(self) -> None:
        env = {
            "JULES_LLM_BASE_URL": "https://dss.example.com/public/api/projects/PROJ/llms/openai/v1/",
            "JULES_LLM_API_KEY": "dkuapi_secret",
            "JULES_LLM_DEFAULT_MODEL": "openai:conn:gpt-4",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config_from_env()
        assert config.base_url == env["JULES_LLM_BASE_URL"]
        assert config.api_key == env["JULES_LLM_API_KEY"]
        assert config.default_model == env["JULES_LLM_DEFAULT_MODEL"]

    def test_load_optional_env_vars(self) -> None:
        env = {
            "JULES_LLM_BASE_URL": "https://x/v1/",
            "JULES_LLM_API_KEY": "k",
            "JULES_LLM_DEFAULT_MODEL": "m",
            "JULES_LLM_TIMEOUT": "30.0",
            "JULES_LLM_MAX_RETRIES": "5",
            "JULES_LLM_VERIFY_SSL": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config_from_env()
        assert config.timeout == 30.0
        assert config.max_retries == 5
        assert config.verify_ssl is False

    def test_missing_base_url_raises(self) -> None:
        env = {
            "JULES_LLM_API_KEY": "k",
            "JULES_LLM_DEFAULT_MODEL": "m",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="JULES_LLM_BASE_URL"):
                load_config_from_env()

    def test_missing_api_key_raises(self) -> None:
        env = {
            "JULES_LLM_BASE_URL": "https://x/v1/",
            "JULES_LLM_DEFAULT_MODEL": "m",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="JULES_LLM_API_KEY"):
                load_config_from_env()

    def test_missing_default_model_raises(self) -> None:
        env = {
            "JULES_LLM_BASE_URL": "https://x/v1/",
            "JULES_LLM_API_KEY": "k",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="JULES_LLM_DEFAULT_MODEL"):
                load_config_from_env()

    def test_verify_ssl_truthy_values(self) -> None:
        base = {
            "JULES_LLM_BASE_URL": "https://x/v1/",
            "JULES_LLM_API_KEY": "k",
            "JULES_LLM_DEFAULT_MODEL": "m",
        }
        for truthy in ("true", "True", "TRUE", "1", "yes"):
            env = {**base, "JULES_LLM_VERIFY_SSL": truthy}
            with patch.dict(os.environ, env, clear=False):
                config = load_config_from_env()
            assert config.verify_ssl is True, f"Expected True for {truthy!r}"

    def test_verify_ssl_falsy_values(self) -> None:
        base = {
            "JULES_LLM_BASE_URL": "https://x/v1/",
            "JULES_LLM_API_KEY": "k",
            "JULES_LLM_DEFAULT_MODEL": "m",
        }
        for falsy in ("false", "False", "FALSE", "0", "no"):
            env = {**base, "JULES_LLM_VERIFY_SSL": falsy}
            with patch.dict(os.environ, env, clear=False):
                config = load_config_from_env()
            assert config.verify_ssl is False, f"Expected False for {falsy!r}"

    def test_invalid_timeout_raises(self) -> None:
        env = {
            "JULES_LLM_BASE_URL": "https://x/v1/",
            "JULES_LLM_API_KEY": "k",
            "JULES_LLM_DEFAULT_MODEL": "m",
            "JULES_LLM_TIMEOUT": "not_a_number",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError):
                load_config_from_env()

    def test_invalid_verify_ssl_raises(self) -> None:
        env = {
            "JULES_LLM_BASE_URL": "https://x/v1/",
            "JULES_LLM_API_KEY": "k",
            "JULES_LLM_DEFAULT_MODEL": "m",
            "JULES_LLM_VERIFY_SSL": "maybe",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="Cannot parse"):
                load_config_from_env()

    def test_empty_env_var_raises_specific_message(self) -> None:
        env = {
            "JULES_LLM_BASE_URL": "",
            "JULES_LLM_API_KEY": "k",
            "JULES_LLM_DEFAULT_MODEL": "m",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="set but empty"):
                load_config_from_env()
