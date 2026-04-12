"""Agent loop configuration with environment variable support.

Provides immutable configuration loading for the agent loop, including
environment variable overrides. The ``load_agent_config_from_env`` function
reads ``JULES_AGENT_MAX_ITERATIONS`` and ``JULES_AGENT_MAX_RETRIES`` from
the process environment, falling back to the ``AgentLoopConfig`` defaults
when not set.

The configuration flows through two layers:
    1. ``AgentLoopConfig`` -- core loop config (max_iterations, max_retries).
    2. ``RequestHandlerConfig.max_agent_iterations`` -- daemon-level config
       that propagates to ``AgentLoopConfig`` when the loop is instantiated.

Environment variables:
    JULES_AGENT_MAX_ITERATIONS: Override the hard cap on think-act-observe
        cycles per agent loop invocation. Must be a positive integer.
        Default: 5.
    JULES_AGENT_MAX_RETRIES: Override the transient error retry count
        within a single iteration. Must be a non-negative integer.
        Default: 2.

Usage::

    from jules_daemon.agent.config import load_agent_config_from_env

    config = load_agent_config_from_env()
    assert config.max_iterations == 5  # or env override
"""

from __future__ import annotations

import os

from jules_daemon.agent.agent_loop import AgentLoopConfig

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BASE_DELAY",
    "load_agent_config",
    "load_agent_config_from_env",
    "resolve_max_iterations",
]

# Canonical defaults -- single source of truth
DEFAULT_MAX_ITERATIONS: int = 15
DEFAULT_MAX_RETRIES: int = 2
DEFAULT_RETRY_BASE_DELAY: float = 1.0

_ENV_MAX_ITERATIONS = "JULES_AGENT_MAX_ITERATIONS"
_ENV_MAX_RETRIES = "JULES_AGENT_MAX_RETRIES"
_ENV_RETRY_BASE_DELAY = "JULES_AGENT_RETRY_BASE_DELAY"


def _parse_positive_int(raw: str, var_name: str) -> int:
    """Parse a string to a positive integer, raising ValueError on failure."""
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {var_name} must be a valid integer, "
            f"got {raw!r}"
        ) from None

    if value < 1:
        raise ValueError(
            f"Environment variable {var_name} must be >= 1, got {value}"
        )
    return value


def _parse_non_negative_int(raw: str, var_name: str) -> int:
    """Parse a string to a non-negative integer."""
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {var_name} must be a valid integer, "
            f"got {raw!r}"
        ) from None

    if value < 0:
        raise ValueError(
            f"Environment variable {var_name} must be >= 0, got {value}"
        )
    return value


def _parse_non_negative_float(raw: str, var_name: str) -> float:
    """Parse a string to a non-negative float."""
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {var_name} must be a valid number, "
            f"got {raw!r}"
        ) from None

    if value < 0.0:
        raise ValueError(
            f"Environment variable {var_name} must be >= 0.0, got {value}"
        )
    return value


def resolve_max_iterations(
    *,
    explicit: int | None = None,
    env_var: str = _ENV_MAX_ITERATIONS,
) -> int:
    """Resolve the max_iterations value from explicit param or env var.

    Priority order:
        1. Explicit parameter (if provided and not None)
        2. Environment variable (if set)
        3. Default (5)

    Args:
        explicit: Caller-provided value. Takes highest priority.
        env_var: Environment variable name to check.

    Returns:
        Resolved max_iterations value (always >= 1).

    Raises:
        ValueError: If the resolved value is invalid.
    """
    if explicit is not None:
        if explicit < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {explicit}"
            )
        return explicit

    raw = os.environ.get(env_var)
    if raw is not None:
        return _parse_positive_int(raw, env_var)

    return DEFAULT_MAX_ITERATIONS


def load_agent_config(
    *,
    max_iterations: int | None = None,
    max_retries: int | None = None,
    retry_base_delay: float | None = None,
) -> AgentLoopConfig:
    """Create an AgentLoopConfig from explicit parameters.

    Args:
        max_iterations: Hard cap on iterations. Default: 5.
        max_retries: Transient error retry count. Default: 2.
        retry_base_delay: Base delay for exponential backoff. Default: 1.0.

    Returns:
        Validated AgentLoopConfig instance.
    """
    return AgentLoopConfig(
        max_iterations=max_iterations if max_iterations is not None else DEFAULT_MAX_ITERATIONS,
        max_retries=max_retries if max_retries is not None else DEFAULT_MAX_RETRIES,
        retry_base_delay=retry_base_delay if retry_base_delay is not None else DEFAULT_RETRY_BASE_DELAY,
    )


def load_agent_config_from_env() -> AgentLoopConfig:
    """Load AgentLoopConfig from environment variables.

    Reads:
        JULES_AGENT_MAX_ITERATIONS: Positive integer (default: 5).
        JULES_AGENT_MAX_RETRIES: Non-negative integer (default: 2).
        JULES_AGENT_RETRY_BASE_DELAY: Non-negative float (default: 1.0).

    Returns:
        Validated AgentLoopConfig instance.

    Raises:
        ValueError: If an environment variable has an invalid value.
    """
    max_iter_raw = os.environ.get(_ENV_MAX_ITERATIONS)
    max_retries_raw = os.environ.get(_ENV_MAX_RETRIES)
    base_delay_raw = os.environ.get(_ENV_RETRY_BASE_DELAY)

    max_iterations = (
        _parse_positive_int(max_iter_raw, _ENV_MAX_ITERATIONS)
        if max_iter_raw is not None
        else DEFAULT_MAX_ITERATIONS
    )
    max_retries = (
        _parse_non_negative_int(max_retries_raw, _ENV_MAX_RETRIES)
        if max_retries_raw is not None
        else DEFAULT_MAX_RETRIES
    )
    retry_base_delay = (
        _parse_non_negative_float(base_delay_raw, _ENV_RETRY_BASE_DELAY)
        if base_delay_raw is not None
        else DEFAULT_RETRY_BASE_DELAY
    )

    return AgentLoopConfig(
        max_iterations=max_iterations,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )
