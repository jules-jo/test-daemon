"""Configurable exponential backoff for SSH reconnection.

Provides an immutable BackoffConfig dataclass and pure functions for
calculating retry delays with jitter. No side effects -- the caller
is responsible for actually sleeping.

Delay formula:
    base = min(max_delay, base_delay * multiplier ^ attempt)
    jitter = random_uniform(-jitter_factor * base, +jitter_factor * base)
    total = max(0, base + jitter)

The jitter prevents thundering-herd effects when multiple daemon
instances reconnect simultaneously.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BackoffConfig:
    """Immutable configuration for exponential backoff.

    Attributes:
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Upper bound on computed delay (seconds).
        multiplier: Exponential growth factor per attempt.
        jitter_factor: Fraction of computed delay to add/subtract as
            random jitter (0.0 = no jitter, 1.0 = full jitter).
        max_retries: Maximum number of retry attempts before giving up.
            A value of 0 means no retries (only the initial attempt).
    """

    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter_factor: float = 0.1
    max_retries: int = 5

    def __post_init__(self) -> None:
        if self.base_delay <= 0:
            raise ValueError(
                f"base_delay must be positive, got {self.base_delay}"
            )
        if self.max_delay <= 0:
            raise ValueError(
                f"max_delay must be positive, got {self.max_delay}"
            )
        if self.max_delay < self.base_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= base_delay "
                f"({self.base_delay})"
            )
        if self.multiplier < 1.0:
            raise ValueError(
                f"multiplier must be >= 1.0, got {self.multiplier}"
            )
        if not (0.0 <= self.jitter_factor <= 1.0):
            raise ValueError(
                f"jitter_factor must be between 0.0 and 1.0, "
                f"got {self.jitter_factor}"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )


@dataclass(frozen=True)
class BackoffDelay:
    """Computed delay for a single retry attempt.

    Attributes:
        attempt: Zero-indexed retry attempt number.
        base_computed: Exponential delay before jitter (capped at max_delay).
        jitter: Random offset applied to the base delay.
        total: Final delay in seconds (always >= 0).
    """

    attempt: int
    base_computed: float
    jitter: float
    total: float


def calculate_delay(
    config: BackoffConfig,
    attempt: int,
    *,
    rng: random.Random | None = None,
) -> BackoffDelay:
    """Calculate the backoff delay for a specific retry attempt.

    Args:
        config: Backoff parameters.
        attempt: Zero-indexed retry attempt number (0 = first retry).
        rng: Optional seeded Random instance for deterministic testing.
            When None, uses the module-level random functions.

    Returns:
        BackoffDelay with the computed delay components.

    Raises:
        ValueError: If attempt is negative.
    """
    if attempt < 0:
        raise ValueError(f"attempt must be non-negative, got {attempt}")

    # Exponential base, capped at max_delay
    base_computed = min(
        config.max_delay,
        config.base_delay * (config.multiplier ** attempt),
    )

    # Jitter: uniform random in [-jitter_factor * base, +jitter_factor * base]
    if config.jitter_factor > 0:
        jitter_range = config.jitter_factor * base_computed
        if rng is not None:
            jitter = rng.uniform(-jitter_range, jitter_range)
        else:
            jitter = random.uniform(-jitter_range, jitter_range)
    else:
        jitter = 0.0

    total = max(0.0, base_computed + jitter)

    return BackoffDelay(
        attempt=attempt,
        base_computed=base_computed,
        jitter=jitter,
        total=total,
    )


def calculate_all_delays(
    config: BackoffConfig,
    *,
    rng: random.Random | None = None,
) -> tuple[BackoffDelay, ...]:
    """Calculate delays for all retry attempts up to max_retries.

    Useful for preview/logging to show the full retry schedule.

    Args:
        config: Backoff parameters.
        rng: Optional seeded Random for deterministic output.

    Returns:
        Tuple of BackoffDelay instances, one per retry attempt.
    """
    return tuple(
        calculate_delay(config, attempt, rng=rng)
        for attempt in range(config.max_retries)
    )
