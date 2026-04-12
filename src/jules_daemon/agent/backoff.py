"""Exponential backoff with jitter calculator for the agent loop.

Pure utility module with no I/O or side effects. Provides configurable
exponential backoff with additive jitter for use in agent loop transient
error retry logic.

Delay formula:
    base = min(max_delay, base_delay * 2 ^ attempt)
    jitter = random_uniform(0, jitter_factor * base)
    total = base + jitter

The jitter is additive (always non-negative) to ensure the delay never
drops below the computed exponential base. This differs from the SSH
backoff module (``ssh.backoff``) which uses symmetric jitter centered
on zero -- the agent loop uses one-sided jitter because the base delay
already represents the minimum acceptable wait time between retries.

Usage::

    from jules_daemon.agent.backoff import (
        AgentBackoffConfig,
        calculate_agent_backoff,
    )

    config = AgentBackoffConfig(base_delay=1.0, max_delay=30.0, jitter_factor=0.25)
    delay = calculate_agent_backoff(
        base_delay=config.base_delay,
        max_delay=config.max_delay,
        attempt=retry_index,
        jitter_factor=config.jitter_factor,
    )
    await asyncio.sleep(delay.total)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

__all__ = [
    "AgentBackoffConfig",
    "AgentBackoffDelay",
    "calculate_agent_backoff",
]


# ---------------------------------------------------------------------------
# AgentBackoffConfig -- immutable configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentBackoffConfig:
    """Immutable configuration for agent loop exponential backoff.

    Attributes:
        base_delay: Initial delay in seconds before the first retry.
            Must be >= 0. A value of 0 disables backoff (useful for tests).
        max_delay: Upper bound on computed delay in seconds. Prevents
            unbounded growth for large attempt numbers. Must be >= base_delay.
        jitter_factor: Fraction of the computed base delay to add as
            random jitter. Must be between 0.0 (no jitter) and 1.0
            (up to 100% additional delay). The jitter is always additive
            (non-negative), so the total delay is always >= base_computed.
    """

    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter_factor: float = 0.25

    def __post_init__(self) -> None:
        if self.base_delay < 0.0:
            raise ValueError(
                f"base_delay must be >= 0, got {self.base_delay}"
            )
        if self.max_delay < 0.0:
            raise ValueError(
                f"max_delay must be >= 0, got {self.max_delay}"
            )
        if self.max_delay < self.base_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= base_delay "
                f"({self.base_delay})"
            )
        if not (0.0 <= self.jitter_factor <= 1.0):
            raise ValueError(
                f"jitter_factor must be between 0.0 and 1.0, "
                f"got {self.jitter_factor}"
            )


# ---------------------------------------------------------------------------
# AgentBackoffDelay -- immutable result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentBackoffDelay:
    """Computed delay for a single retry attempt.

    Attributes:
        attempt: Zero-indexed retry attempt number.
        base_computed: Exponential delay before jitter (capped at max_delay).
        jitter: Random additive offset applied to the base delay (>= 0).
        total: Final delay in seconds (base_computed + jitter, always >= 0).
    """

    attempt: int
    base_computed: float
    jitter: float
    total: float


# ---------------------------------------------------------------------------
# Pure calculation function
# ---------------------------------------------------------------------------


def calculate_agent_backoff(
    *,
    base_delay: float,
    max_delay: float,
    attempt: int,
    jitter_factor: float,
    rng: random.Random | None = None,
) -> AgentBackoffDelay:
    """Calculate the exponential backoff delay for a retry attempt.

    Pure function with no I/O or side effects. The only source of
    non-determinism is the random jitter, which can be made deterministic
    by passing a seeded ``rng`` instance.

    Formula:
        base_computed = min(max_delay, base_delay * 2^attempt)
        jitter = uniform(0, jitter_factor * base_computed)
        total = base_computed + jitter

    Args:
        base_delay: Initial delay in seconds. Must be >= 0.
        max_delay: Upper bound on the exponential delay. Must be >= 0
            and >= base_delay.
        attempt: Zero-indexed retry attempt number. 0 = first retry.
            Must be >= 0.
        jitter_factor: Fraction of base_computed to add as random jitter.
            Must be between 0.0 and 1.0 inclusive.
        rng: Optional seeded Random instance for deterministic testing.
            When None, uses the module-level random functions.

    Returns:
        AgentBackoffDelay with the computed delay components.

    Raises:
        ValueError: If any input violates its constraints.
    """
    if base_delay < 0.0:
        raise ValueError(f"base_delay must be >= 0, got {base_delay}")
    if max_delay < 0.0:
        raise ValueError(f"max_delay must be >= 0, got {max_delay}")
    if max_delay < base_delay:
        raise ValueError(
            f"max_delay ({max_delay}) must be >= base_delay ({base_delay})"
        )
    if attempt < 0:
        raise ValueError(f"attempt must be >= 0, got {attempt}")
    if not (0.0 <= jitter_factor <= 1.0):
        raise ValueError(
            f"jitter_factor must be between 0.0 and 1.0, got {jitter_factor}"
        )

    # Exponential base, capped at max_delay
    base_computed = min(max_delay, base_delay * (2 ** attempt))

    # Additive jitter: uniform in [0, jitter_factor * base_computed]
    if jitter_factor > 0.0 and base_computed > 0.0:
        jitter_range = jitter_factor * base_computed
        if rng is not None:
            jitter = rng.uniform(0.0, jitter_range)
        else:
            jitter = random.uniform(0.0, jitter_range)
    else:
        jitter = 0.0

    total = base_computed + jitter

    return AgentBackoffDelay(
        attempt=attempt,
        base_computed=base_computed,
        jitter=jitter,
        total=total,
    )
