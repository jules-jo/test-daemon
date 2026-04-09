"""LLM client configuration.

Provides an immutable LLMConfig dataclass and factory functions for
loading configuration from explicit parameters or environment variables.
No secrets are ever hardcoded -- all sensitive values come from the
caller or the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


_ENV_PREFIX = "JULES_LLM_"
_TRUTHY = frozenset({"true", "1", "yes"})
_FALSY = frozenset({"false", "0", "no"})


@dataclass(frozen=True)
class LLMConfig:
    """Immutable configuration for the Dataiku Mesh LLM client.

    Attributes:
        base_url: Dataiku Mesh OpenAI-compatible endpoint URL.
            Format: ``https://<HOST>/public/api/projects/<PROJECT>/llms/openai/v1/``
        api_key: Dataiku API key for authentication.
        default_model: Default model in ``provider:connection:model`` format.
        timeout: HTTP request timeout in seconds.
        max_retries: Number of automatic retries on transient failures.
        verify_ssl: Whether to verify SSL certificates (disable for self-signed certs).
    """

    base_url: str
    api_key: str
    default_model: str
    timeout: float = 120.0
    max_retries: int = 2
    verify_ssl: bool = True

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        if not self.api_key:
            raise ValueError("api_key must not be empty")
        if not self.default_model:
            raise ValueError("default_model must not be empty")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )


def load_config(
    *,
    base_url: str,
    api_key: str,
    default_model: str,
    timeout: float = 120.0,
    max_retries: int = 2,
    verify_ssl: bool = True,
) -> LLMConfig:
    """Create an LLMConfig from explicit parameters.

    Args:
        base_url: Dataiku Mesh endpoint URL.
        api_key: Dataiku API key.
        default_model: Default model in ``provider:connection:model`` format.
        timeout: HTTP timeout in seconds.
        max_retries: Retry count for transient failures.
        verify_ssl: SSL verification toggle.

    Returns:
        Validated LLMConfig instance.
    """
    return LLMConfig(
        base_url=base_url,
        api_key=api_key,
        default_model=default_model,
        timeout=timeout,
        max_retries=max_retries,
        verify_ssl=verify_ssl,
    )


def _require_env(name: str) -> str:
    """Read a required environment variable or raise ValueError."""
    value = os.environ.get(name)
    if value is None:
        raise ValueError(f"Required environment variable {name} is not set")
    if not value:
        raise ValueError(f"Required environment variable {name} is set but empty")
    return value


def _parse_bool(raw: str) -> bool:
    """Parse a boolean from a string, case-insensitive."""
    lower = raw.strip().lower()
    if lower in _TRUTHY:
        return True
    if lower in _FALSY:
        return False
    raise ValueError(
        f"Cannot parse {raw!r} as boolean, expected one of: "
        f"{sorted(_TRUTHY | _FALSY)}"
    )


def load_config_from_env() -> LLMConfig:
    """Load LLMConfig from environment variables.

    Required environment variables:
        JULES_LLM_BASE_URL: Dataiku Mesh endpoint URL.
        JULES_LLM_API_KEY: Dataiku API key.
        JULES_LLM_DEFAULT_MODEL: Default model string.

    Optional environment variables:
        JULES_LLM_TIMEOUT: HTTP timeout in seconds (default: 120.0).
        JULES_LLM_MAX_RETRIES: Retry count (default: 2).
        JULES_LLM_VERIFY_SSL: SSL verification boolean (default: true).

    Returns:
        Validated LLMConfig instance.

    Raises:
        ValueError: If a required variable is missing or a value is invalid.
    """
    base_url = _require_env(f"{_ENV_PREFIX}BASE_URL")
    api_key = _require_env(f"{_ENV_PREFIX}API_KEY")
    default_model = _require_env(f"{_ENV_PREFIX}DEFAULT_MODEL")

    timeout_raw = os.environ.get(f"{_ENV_PREFIX}TIMEOUT")
    timeout = float(timeout_raw) if timeout_raw else 120.0

    retries_raw = os.environ.get(f"{_ENV_PREFIX}MAX_RETRIES")
    max_retries = int(retries_raw) if retries_raw else 2

    ssl_raw = os.environ.get(f"{_ENV_PREFIX}VERIFY_SSL")
    verify_ssl = _parse_bool(ssl_raw) if ssl_raw else True

    return LLMConfig(
        base_url=base_url,
        api_key=api_key,
        default_model=default_model,
        timeout=timeout,
        max_retries=max_retries,
        verify_ssl=verify_ssl,
    )
