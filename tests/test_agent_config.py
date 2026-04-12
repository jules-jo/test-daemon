"""Tests for jules_daemon.agent.config -- agent loop config loading."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from jules_daemon.agent.agent_loop import AgentLoopConfig
from jules_daemon.agent.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    load_agent_config,
    load_agent_config_from_env,
    resolve_max_iterations,
)


# ---------------------------------------------------------------------------
# resolve_max_iterations
# ---------------------------------------------------------------------------


class TestResolveMaxIterations:
    """Tests for resolve_max_iterations()."""

    def test_explicit_value_takes_precedence(self) -> None:
        result = resolve_max_iterations(explicit=10)
        assert result == 10

    def test_explicit_value_one(self) -> None:
        result = resolve_max_iterations(explicit=1)
        assert result == 1

    def test_explicit_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            resolve_max_iterations(explicit=0)

    def test_explicit_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            resolve_max_iterations(explicit=-1)

    def test_env_var_used_when_no_explicit(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "7"}):
            result = resolve_max_iterations()
            assert result == 7

    def test_default_when_no_explicit_and_no_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if present
            os.environ.pop("JULES_AGENT_MAX_ITERATIONS", None)
            result = resolve_max_iterations()
            assert result == DEFAULT_MAX_ITERATIONS

    def test_explicit_overrides_env_var(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "99"}):
            result = resolve_max_iterations(explicit=3)
            assert result == 3

    def test_env_var_invalid_string_raises(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "abc"}):
            with pytest.raises(ValueError, match="must be a valid integer"):
                resolve_max_iterations()

    def test_env_var_zero_raises(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "0"}):
            with pytest.raises(ValueError, match="must be >= 1"):
                resolve_max_iterations()

    def test_env_var_negative_raises(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "-5"}):
            with pytest.raises(ValueError, match="must be >= 1"):
                resolve_max_iterations()


# ---------------------------------------------------------------------------
# load_agent_config
# ---------------------------------------------------------------------------


class TestLoadAgentConfig:
    """Tests for load_agent_config()."""

    def test_defaults(self) -> None:
        config = load_agent_config()
        assert config.max_iterations == DEFAULT_MAX_ITERATIONS
        assert config.max_retries == DEFAULT_MAX_RETRIES
        assert config.retry_base_delay == DEFAULT_RETRY_BASE_DELAY

    def test_explicit_max_iterations(self) -> None:
        config = load_agent_config(max_iterations=10)
        assert config.max_iterations == 10

    def test_explicit_max_retries(self) -> None:
        config = load_agent_config(max_retries=5)
        assert config.max_retries == 5

    def test_explicit_retry_base_delay(self) -> None:
        config = load_agent_config(retry_base_delay=2.5)
        assert config.retry_base_delay == 2.5

    def test_all_explicit(self) -> None:
        config = load_agent_config(
            max_iterations=3,
            max_retries=1,
            retry_base_delay=0.5,
        )
        assert config.max_iterations == 3
        assert config.max_retries == 1
        assert config.retry_base_delay == 0.5

    def test_returns_agent_loop_config(self) -> None:
        config = load_agent_config()
        assert isinstance(config, AgentLoopConfig)


# ---------------------------------------------------------------------------
# load_agent_config_from_env
# ---------------------------------------------------------------------------


class TestLoadAgentConfigFromEnv:
    """Tests for load_agent_config_from_env()."""

    def test_defaults_when_no_env_vars(self) -> None:
        env = {
            k: v for k, v in os.environ.items()
            if k not in (
                "JULES_AGENT_MAX_ITERATIONS",
                "JULES_AGENT_MAX_RETRIES",
                "JULES_AGENT_RETRY_BASE_DELAY",
            )
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_agent_config_from_env()
            assert config.max_iterations == DEFAULT_MAX_ITERATIONS
            assert config.max_retries == DEFAULT_MAX_RETRIES
            assert config.retry_base_delay == DEFAULT_RETRY_BASE_DELAY

    def test_env_max_iterations(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "8"}):
            config = load_agent_config_from_env()
            assert config.max_iterations == 8

    def test_env_max_retries(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_MAX_RETRIES": "4"}):
            config = load_agent_config_from_env()
            assert config.max_retries == 4

    def test_env_max_retries_zero(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_MAX_RETRIES": "0"}):
            config = load_agent_config_from_env()
            assert config.max_retries == 0

    def test_env_retry_base_delay(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_RETRY_BASE_DELAY": "2.5"}):
            config = load_agent_config_from_env()
            assert config.retry_base_delay == 2.5

    def test_env_retry_base_delay_zero(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_RETRY_BASE_DELAY": "0.0"}):
            config = load_agent_config_from_env()
            assert config.retry_base_delay == 0.0

    def test_invalid_max_iterations_raises(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_MAX_ITERATIONS": "not_a_number"}):
            with pytest.raises(ValueError, match="must be a valid integer"):
                load_agent_config_from_env()

    def test_invalid_max_retries_raises(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_MAX_RETRIES": "abc"}):
            with pytest.raises(ValueError, match="must be a valid integer"):
                load_agent_config_from_env()

    def test_negative_max_retries_raises(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_MAX_RETRIES": "-1"}):
            with pytest.raises(ValueError, match="must be >= 0"):
                load_agent_config_from_env()

    def test_invalid_retry_base_delay_raises(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_RETRY_BASE_DELAY": "xyz"}):
            with pytest.raises(ValueError, match="must be a valid number"):
                load_agent_config_from_env()

    def test_negative_retry_base_delay_raises(self) -> None:
        with patch.dict(os.environ, {"JULES_AGENT_RETRY_BASE_DELAY": "-1.0"}):
            with pytest.raises(ValueError, match="must be >= 0.0"):
                load_agent_config_from_env()

    def test_returns_agent_loop_config(self) -> None:
        config = load_agent_config_from_env()
        assert isinstance(config, AgentLoopConfig)
