"""Tests for agent loop exponential backoff with jitter calculator.

Covers:
    - Pure function behavior: no I/O, no side effects
    - Exponential growth: delay = min(max_delay, base_delay * 2^attempt)
    - Jitter: uniform random in [0, jitter_factor * base_computed]
    - Max delay cap prevents unbounded growth
    - Boundary conditions: attempt=0, large attempts, zero jitter
    - Deterministic output with seeded RNG
    - Input validation: negative values, invalid ranges
    - Immutability: frozen result dataclass
"""

from __future__ import annotations

import random

import pytest

from jules_daemon.agent.backoff import (
    AgentBackoffConfig,
    AgentBackoffDelay,
    calculate_agent_backoff,
)


# ---------------------------------------------------------------------------
# AgentBackoffConfig validation
# ---------------------------------------------------------------------------


class TestAgentBackoffConfigDefaults:
    """Verify sensible defaults for agent loop backoff."""

    def test_default_values(self) -> None:
        config = AgentBackoffConfig()
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.jitter_factor == 0.25

    def test_custom_values(self) -> None:
        config = AgentBackoffConfig(
            base_delay=0.5,
            max_delay=10.0,
            jitter_factor=0.5,
        )
        assert config.base_delay == 0.5
        assert config.max_delay == 10.0
        assert config.jitter_factor == 0.5


class TestAgentBackoffConfigValidation:
    """Validation rules for AgentBackoffConfig."""

    def test_base_delay_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="base_delay must be >= 0"):
            AgentBackoffConfig(base_delay=-0.1)

    def test_base_delay_zero_is_valid(self) -> None:
        config = AgentBackoffConfig(base_delay=0.0)
        assert config.base_delay == 0.0

    def test_max_delay_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="max_delay must be >= 0"):
            AgentBackoffConfig(max_delay=-1.0)

    def test_max_delay_zero_is_valid(self) -> None:
        config = AgentBackoffConfig(base_delay=0.0, max_delay=0.0)
        assert config.max_delay == 0.0

    def test_max_delay_less_than_base_raises(self) -> None:
        with pytest.raises(ValueError, match="max_delay.*must be >= base_delay"):
            AgentBackoffConfig(base_delay=10.0, max_delay=5.0)

    def test_jitter_factor_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="jitter_factor"):
            AgentBackoffConfig(jitter_factor=-0.1)

    def test_jitter_factor_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="jitter_factor"):
            AgentBackoffConfig(jitter_factor=1.1)

    def test_jitter_factor_zero_is_valid(self) -> None:
        config = AgentBackoffConfig(jitter_factor=0.0)
        assert config.jitter_factor == 0.0

    def test_jitter_factor_one_is_valid(self) -> None:
        config = AgentBackoffConfig(jitter_factor=1.0)
        assert config.jitter_factor == 1.0


class TestAgentBackoffConfigFrozen:
    """AgentBackoffConfig must be immutable."""

    def test_cannot_mutate_base_delay(self) -> None:
        config = AgentBackoffConfig()
        with pytest.raises(AttributeError):
            config.base_delay = 2.0  # type: ignore[misc]

    def test_cannot_mutate_max_delay(self) -> None:
        config = AgentBackoffConfig()
        with pytest.raises(AttributeError):
            config.max_delay = 100.0  # type: ignore[misc]

    def test_cannot_mutate_jitter_factor(self) -> None:
        config = AgentBackoffConfig()
        with pytest.raises(AttributeError):
            config.jitter_factor = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# calculate_agent_backoff() -- exponential growth
# ---------------------------------------------------------------------------


class TestCalculateAgentBackoffExponentialGrowth:
    """Verify exponential growth pattern with no jitter."""

    def test_attempt_zero(self) -> None:
        result = calculate_agent_backoff(
            base_delay=1.0, max_delay=60.0, attempt=0, jitter_factor=0.0,
        )
        assert result.base_computed == 1.0  # 1.0 * 2^0

    def test_attempt_one(self) -> None:
        result = calculate_agent_backoff(
            base_delay=1.0, max_delay=60.0, attempt=1, jitter_factor=0.0,
        )
        assert result.base_computed == 2.0  # 1.0 * 2^1

    def test_attempt_two(self) -> None:
        result = calculate_agent_backoff(
            base_delay=1.0, max_delay=60.0, attempt=2, jitter_factor=0.0,
        )
        assert result.base_computed == 4.0  # 1.0 * 2^2

    def test_attempt_three(self) -> None:
        result = calculate_agent_backoff(
            base_delay=1.0, max_delay=60.0, attempt=3, jitter_factor=0.0,
        )
        assert result.base_computed == 8.0  # 1.0 * 2^3

    def test_custom_base_delay(self) -> None:
        result = calculate_agent_backoff(
            base_delay=0.5, max_delay=100.0, attempt=3, jitter_factor=0.0,
        )
        assert result.base_computed == 4.0  # 0.5 * 2^3

    def test_total_equals_base_when_no_jitter(self) -> None:
        result = calculate_agent_backoff(
            base_delay=2.0, max_delay=60.0, attempt=2, jitter_factor=0.0,
        )
        assert result.total == result.base_computed
        assert result.jitter == 0.0


# ---------------------------------------------------------------------------
# calculate_agent_backoff() -- max delay cap
# ---------------------------------------------------------------------------


class TestCalculateAgentBackoffMaxDelayCap:
    """Verify delay is capped at max_delay."""

    def test_caps_at_max_delay(self) -> None:
        result = calculate_agent_backoff(
            base_delay=1.0, max_delay=10.0, attempt=10, jitter_factor=0.0,
        )
        # 1.0 * 2^10 = 1024, should be capped at 10.0
        assert result.base_computed == 10.0
        assert result.total == 10.0

    def test_exact_boundary(self) -> None:
        result = calculate_agent_backoff(
            base_delay=1.0, max_delay=8.0, attempt=3, jitter_factor=0.0,
        )
        # 1.0 * 2^3 = 8.0, exactly at max_delay
        assert result.base_computed == 8.0

    def test_just_over_boundary(self) -> None:
        result = calculate_agent_backoff(
            base_delay=1.0, max_delay=7.0, attempt=3, jitter_factor=0.0,
        )
        # 1.0 * 2^3 = 8.0, capped at 7.0
        assert result.base_computed == 7.0


# ---------------------------------------------------------------------------
# calculate_agent_backoff() -- jitter
# ---------------------------------------------------------------------------


class TestCalculateAgentBackoffJitter:
    """Verify jitter is bounded and deterministic with seeded RNG."""

    def test_jitter_within_bounds(self) -> None:
        rng = random.Random(42)
        for attempt in range(10):
            result = calculate_agent_backoff(
                base_delay=1.0,
                max_delay=60.0,
                attempt=attempt,
                jitter_factor=0.5,
                rng=rng,
            )
            max_jitter = 0.5 * result.base_computed
            assert 0.0 <= result.jitter <= max_jitter

    def test_deterministic_with_seed(self) -> None:
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        for attempt in range(5):
            r1 = calculate_agent_backoff(
                base_delay=1.0,
                max_delay=30.0,
                attempt=attempt,
                jitter_factor=0.25,
                rng=rng1,
            )
            r2 = calculate_agent_backoff(
                base_delay=1.0,
                max_delay=30.0,
                attempt=attempt,
                jitter_factor=0.25,
                rng=rng2,
            )
            assert r1.jitter == r2.jitter
            assert r1.total == r2.total

    def test_no_jitter_when_factor_zero(self) -> None:
        result = calculate_agent_backoff(
            base_delay=5.0, max_delay=60.0, attempt=3, jitter_factor=0.0,
        )
        assert result.jitter == 0.0

    def test_total_never_negative(self) -> None:
        """Even with full jitter, total should be >= 0."""
        rng = random.Random(99)
        for _ in range(200):
            result = calculate_agent_backoff(
                base_delay=0.1,
                max_delay=60.0,
                attempt=0,
                jitter_factor=1.0,
                rng=rng,
            )
            assert result.total >= 0.0

    def test_jitter_adds_randomness(self) -> None:
        """Two calls with different RNG state should produce different jitter."""
        rng = random.Random(42)
        r1 = calculate_agent_backoff(
            base_delay=1.0, max_delay=30.0, attempt=1, jitter_factor=0.5, rng=rng,
        )
        r2 = calculate_agent_backoff(
            base_delay=1.0, max_delay=30.0, attempt=1, jitter_factor=0.5, rng=rng,
        )
        # Same RNG, advanced state -- should produce different jitter
        # (unless astronomically unlikely collision)
        assert r1.jitter != r2.jitter


# ---------------------------------------------------------------------------
# calculate_agent_backoff() -- input validation
# ---------------------------------------------------------------------------


class TestCalculateAgentBackoffValidation:
    """Validate inputs to calculate_agent_backoff()."""

    def test_negative_attempt_raises(self) -> None:
        with pytest.raises(ValueError, match="attempt must be >= 0"):
            calculate_agent_backoff(
                base_delay=1.0, max_delay=30.0, attempt=-1, jitter_factor=0.0,
            )

    def test_negative_base_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="base_delay must be >= 0"):
            calculate_agent_backoff(
                base_delay=-1.0, max_delay=30.0, attempt=0, jitter_factor=0.0,
            )

    def test_negative_max_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="max_delay must be >= 0"):
            calculate_agent_backoff(
                base_delay=1.0, max_delay=-1.0, attempt=0, jitter_factor=0.0,
            )

    def test_negative_jitter_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="jitter_factor must be"):
            calculate_agent_backoff(
                base_delay=1.0, max_delay=30.0, attempt=0, jitter_factor=-0.1,
            )

    def test_jitter_factor_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="jitter_factor must be"):
            calculate_agent_backoff(
                base_delay=1.0, max_delay=30.0, attempt=0, jitter_factor=1.5,
            )

    def test_max_delay_less_than_base_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="max_delay.*must be >= base_delay"):
            calculate_agent_backoff(
                base_delay=10.0, max_delay=5.0, attempt=0, jitter_factor=0.0,
            )


# ---------------------------------------------------------------------------
# calculate_agent_backoff() -- edge cases
# ---------------------------------------------------------------------------


class TestCalculateAgentBackoffEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_base_delay(self) -> None:
        result = calculate_agent_backoff(
            base_delay=0.0, max_delay=0.0, attempt=5, jitter_factor=0.0,
        )
        assert result.base_computed == 0.0
        assert result.total == 0.0

    def test_large_attempt_number(self) -> None:
        result = calculate_agent_backoff(
            base_delay=1.0, max_delay=30.0, attempt=100, jitter_factor=0.0,
        )
        assert result.base_computed == 30.0  # capped at max

    def test_max_delay_equals_base_delay(self) -> None:
        result = calculate_agent_backoff(
            base_delay=5.0, max_delay=5.0, attempt=10, jitter_factor=0.0,
        )
        assert result.base_computed == 5.0
        assert result.total == 5.0


# ---------------------------------------------------------------------------
# calculate_agent_backoff() -- config convenience overload
# ---------------------------------------------------------------------------


class TestCalculateAgentBackoffFromConfig:
    """Verify the config-based convenience call pattern."""

    def test_from_config_matches_direct_call(self) -> None:
        config = AgentBackoffConfig(
            base_delay=2.0, max_delay=20.0, jitter_factor=0.0,
        )
        direct = calculate_agent_backoff(
            base_delay=config.base_delay,
            max_delay=config.max_delay,
            attempt=3,
            jitter_factor=config.jitter_factor,
        )
        assert direct.base_computed == 16.0  # 2.0 * 2^3
        assert direct.total == 16.0


# ---------------------------------------------------------------------------
# AgentBackoffDelay frozen dataclass
# ---------------------------------------------------------------------------


class TestAgentBackoffDelayFrozen:
    """AgentBackoffDelay must be immutable."""

    def test_returns_correct_type(self) -> None:
        result = calculate_agent_backoff(
            base_delay=1.0, max_delay=30.0, attempt=0, jitter_factor=0.0,
        )
        assert isinstance(result, AgentBackoffDelay)

    def test_frozen(self) -> None:
        result = calculate_agent_backoff(
            base_delay=1.0, max_delay=30.0, attempt=0, jitter_factor=0.0,
        )
        with pytest.raises(AttributeError):
            result.total = 999.0  # type: ignore[misc]

    def test_attempt_field(self) -> None:
        result = calculate_agent_backoff(
            base_delay=1.0, max_delay=30.0, attempt=3, jitter_factor=0.0,
        )
        assert result.attempt == 3
