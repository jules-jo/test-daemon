"""Tests for exponential backoff configuration and delay calculation.

Covers:
    - BackoffConfig validation (frozen, positive values, constraints)
    - calculate_delay() exponential growth and max_delay capping
    - calculate_delay() jitter bounds
    - calculate_delay() deterministic output with seeded RNG
    - calculate_delay() edge cases (attempt=0, large attempts)
    - calculate_all_delays() returns correct number of entries
"""

from __future__ import annotations

import random

import pytest

from jules_daemon.ssh.backoff import (
    BackoffConfig,
    BackoffDelay,
    calculate_all_delays,
    calculate_delay,
)


# ---------------------------------------------------------------------------
# BackoffConfig validation
# ---------------------------------------------------------------------------


class TestBackoffConfigDefaults:
    """Verify sensible defaults."""

    def test_default_values(self) -> None:
        config = BackoffConfig()
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.multiplier == 2.0
        assert config.jitter_factor == 0.1
        assert config.max_retries == 5

    def test_custom_values(self) -> None:
        config = BackoffConfig(
            base_delay=0.5,
            max_delay=30.0,
            multiplier=3.0,
            jitter_factor=0.2,
            max_retries=10,
        )
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.multiplier == 3.0
        assert config.jitter_factor == 0.2
        assert config.max_retries == 10


class TestBackoffConfigValidation:
    """Validation rules for BackoffConfig."""

    def test_base_delay_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="base_delay must be positive"):
            BackoffConfig(base_delay=0.0)

    def test_base_delay_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="base_delay must be positive"):
            BackoffConfig(base_delay=-1.0)

    def test_max_delay_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_delay must be positive"):
            BackoffConfig(max_delay=0.0)

    def test_max_delay_less_than_base_raises(self) -> None:
        with pytest.raises(ValueError, match="max_delay.*must be >= base_delay"):
            BackoffConfig(base_delay=10.0, max_delay=5.0)

    def test_multiplier_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="multiplier must be >= 1.0"):
            BackoffConfig(multiplier=0.5)

    def test_multiplier_exactly_one_ok(self) -> None:
        config = BackoffConfig(multiplier=1.0)
        assert config.multiplier == 1.0

    def test_jitter_factor_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="jitter_factor"):
            BackoffConfig(jitter_factor=-0.1)

    def test_jitter_factor_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="jitter_factor"):
            BackoffConfig(jitter_factor=1.1)

    def test_jitter_factor_zero_ok(self) -> None:
        config = BackoffConfig(jitter_factor=0.0)
        assert config.jitter_factor == 0.0

    def test_jitter_factor_one_ok(self) -> None:
        config = BackoffConfig(jitter_factor=1.0)
        assert config.jitter_factor == 1.0

    def test_max_retries_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            BackoffConfig(max_retries=-1)

    def test_max_retries_zero_ok(self) -> None:
        config = BackoffConfig(max_retries=0)
        assert config.max_retries == 0


class TestBackoffConfigFrozen:
    """BackoffConfig must be immutable."""

    def test_cannot_mutate_base_delay(self) -> None:
        config = BackoffConfig()
        with pytest.raises(AttributeError):
            config.base_delay = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# calculate_delay()
# ---------------------------------------------------------------------------


class TestCalculateDelayExponentialGrowth:
    """Verify exponential growth pattern with no jitter."""

    def test_attempt_zero(self) -> None:
        config = BackoffConfig(base_delay=1.0, multiplier=2.0, jitter_factor=0.0)
        result = calculate_delay(config, 0)
        assert result.attempt == 0
        assert result.base_computed == 1.0  # 1.0 * 2^0
        assert result.jitter == 0.0
        assert result.total == 1.0

    def test_attempt_one(self) -> None:
        config = BackoffConfig(base_delay=1.0, multiplier=2.0, jitter_factor=0.0)
        result = calculate_delay(config, 1)
        assert result.base_computed == 2.0  # 1.0 * 2^1

    def test_attempt_two(self) -> None:
        config = BackoffConfig(base_delay=1.0, multiplier=2.0, jitter_factor=0.0)
        result = calculate_delay(config, 2)
        assert result.base_computed == 4.0  # 1.0 * 2^2

    def test_attempt_three(self) -> None:
        config = BackoffConfig(base_delay=1.0, multiplier=2.0, jitter_factor=0.0)
        result = calculate_delay(config, 3)
        assert result.base_computed == 8.0  # 1.0 * 2^3

    def test_custom_base_and_multiplier(self) -> None:
        config = BackoffConfig(
            base_delay=0.5, max_delay=100.0, multiplier=3.0, jitter_factor=0.0
        )
        result = calculate_delay(config, 2)
        assert result.base_computed == 4.5  # 0.5 * 3^2 = 4.5

    def test_multiplier_one_gives_constant_delay(self) -> None:
        config = BackoffConfig(
            base_delay=5.0, multiplier=1.0, jitter_factor=0.0
        )
        for attempt in range(5):
            result = calculate_delay(config, attempt)
            assert result.base_computed == 5.0


class TestCalculateDelayMaxDelayCap:
    """Verify delay is capped at max_delay."""

    def test_caps_at_max_delay(self) -> None:
        config = BackoffConfig(
            base_delay=1.0, max_delay=10.0, multiplier=2.0, jitter_factor=0.0
        )
        # Attempt 10: 1.0 * 2^10 = 1024, should be capped at 10.0
        result = calculate_delay(config, 10)
        assert result.base_computed == 10.0
        assert result.total == 10.0

    def test_boundary_not_capped(self) -> None:
        config = BackoffConfig(
            base_delay=1.0, max_delay=8.0, multiplier=2.0, jitter_factor=0.0
        )
        # Attempt 3: 1.0 * 2^3 = 8.0, exactly at max_delay
        result = calculate_delay(config, 3)
        assert result.base_computed == 8.0

    def test_boundary_capped(self) -> None:
        config = BackoffConfig(
            base_delay=1.0, max_delay=7.0, multiplier=2.0, jitter_factor=0.0
        )
        # Attempt 3: 1.0 * 2^3 = 8.0, should be capped at 7.0
        result = calculate_delay(config, 3)
        assert result.base_computed == 7.0


class TestCalculateDelayJitter:
    """Verify jitter is bounded and deterministic with seeded RNG."""

    def test_jitter_within_bounds(self) -> None:
        config = BackoffConfig(
            base_delay=10.0, max_delay=60.0, multiplier=2.0, jitter_factor=0.2
        )
        rng = random.Random(42)
        for attempt in range(5):
            result = calculate_delay(config, attempt, rng=rng)
            max_jitter = config.jitter_factor * result.base_computed
            assert -max_jitter <= result.jitter <= max_jitter

    def test_jitter_deterministic_with_seed(self) -> None:
        config = BackoffConfig(jitter_factor=0.5)
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        for attempt in range(5):
            r1 = calculate_delay(config, attempt, rng=rng1)
            r2 = calculate_delay(config, attempt, rng=rng2)
            assert r1.jitter == r2.jitter
            assert r1.total == r2.total

    def test_no_jitter_when_factor_zero(self) -> None:
        config = BackoffConfig(jitter_factor=0.0)
        result = calculate_delay(config, 3)
        assert result.jitter == 0.0

    def test_total_never_negative(self) -> None:
        """Even with max jitter, total should be >= 0."""
        config = BackoffConfig(
            base_delay=0.1, max_delay=60.0, multiplier=1.0, jitter_factor=1.0
        )
        # Run many iterations to probabilistically test
        rng = random.Random(99)
        for attempt in range(100):
            result = calculate_delay(config, attempt % 5, rng=rng)
            assert result.total >= 0.0


class TestCalculateDelayValidation:
    """Validate inputs to calculate_delay()."""

    def test_negative_attempt_raises(self) -> None:
        config = BackoffConfig()
        with pytest.raises(ValueError, match="attempt must be non-negative"):
            calculate_delay(config, -1)


class TestCalculateDelayResultType:
    """Verify BackoffDelay is a frozen dataclass."""

    def test_returns_backoff_delay(self) -> None:
        config = BackoffConfig(jitter_factor=0.0)
        result = calculate_delay(config, 0)
        assert isinstance(result, BackoffDelay)

    def test_frozen(self) -> None:
        config = BackoffConfig(jitter_factor=0.0)
        result = calculate_delay(config, 0)
        with pytest.raises(AttributeError):
            result.total = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# calculate_all_delays()
# ---------------------------------------------------------------------------


class TestCalculateAllDelays:
    """Verify batch delay calculation."""

    def test_returns_correct_count(self) -> None:
        config = BackoffConfig(max_retries=5, jitter_factor=0.0)
        delays = calculate_all_delays(config)
        assert len(delays) == 5

    def test_zero_retries_returns_empty(self) -> None:
        config = BackoffConfig(max_retries=0)
        delays = calculate_all_delays(config)
        assert len(delays) == 0

    def test_delays_increase_monotonically_no_jitter(self) -> None:
        config = BackoffConfig(
            base_delay=1.0, max_delay=100.0, multiplier=2.0,
            jitter_factor=0.0, max_retries=5,
        )
        delays = calculate_all_delays(config)
        for i in range(len(delays) - 1):
            assert delays[i].total <= delays[i + 1].total

    def test_returns_tuple(self) -> None:
        config = BackoffConfig(max_retries=3, jitter_factor=0.0)
        delays = calculate_all_delays(config)
        assert isinstance(delays, tuple)

    def test_deterministic_with_rng(self) -> None:
        config = BackoffConfig(max_retries=5, jitter_factor=0.5)
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        d1 = calculate_all_delays(config, rng=rng1)
        d2 = calculate_all_delays(config, rng=rng2)
        for a, b in zip(d1, d2, strict=True):
            assert a.total == b.total
