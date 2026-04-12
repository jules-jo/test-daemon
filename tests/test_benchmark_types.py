"""Tests for benchmark harness data types and configuration.

Validates the frozen BenchmarkConfig and BenchmarkResult dataclasses
including construction, validation, serialization round-trips,
immutability, and computed statistics.
"""

from __future__ import annotations

import statistics

import pytest

from jules_daemon.agent.benchmark_types import (
    BenchmarkConfig,
    BenchmarkResult,
    compute_result,
)


# ---------------------------------------------------------------------------
# BenchmarkConfig tests
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    """Tests for the frozen BenchmarkConfig dataclass."""

    def test_create_with_defaults(self) -> None:
        config = BenchmarkConfig()
        assert config.iterations == 100
        assert config.warmup_count == 5
        assert config.label == ""

    def test_create_with_explicit_values(self) -> None:
        config = BenchmarkConfig(
            iterations=200,
            warmup_count=10,
            label="agent-loop-overhead",
        )
        assert config.iterations == 200
        assert config.warmup_count == 10
        assert config.label == "agent-loop-overhead"

    def test_frozen_immutability(self) -> None:
        config = BenchmarkConfig()
        with pytest.raises(AttributeError):
            config.iterations = 50  # type: ignore[misc]

    def test_zero_iterations_raises(self) -> None:
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            BenchmarkConfig(iterations=0)

    def test_negative_iterations_raises(self) -> None:
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            BenchmarkConfig(iterations=-5)

    def test_negative_warmup_raises(self) -> None:
        with pytest.raises(ValueError, match="warmup_count must be >= 0"):
            BenchmarkConfig(warmup_count=-1)

    def test_zero_warmup_allowed(self) -> None:
        config = BenchmarkConfig(warmup_count=0)
        assert config.warmup_count == 0

    def test_to_dict(self) -> None:
        config = BenchmarkConfig(
            iterations=50,
            warmup_count=3,
            label="unit-test",
        )
        result = config.to_dict()
        assert result == {
            "iterations": 50,
            "warmup_count": 3,
            "label": "unit-test",
        }

    def test_from_dict(self) -> None:
        data = {
            "iterations": 75,
            "warmup_count": 7,
            "label": "round-trip",
        }
        config = BenchmarkConfig.from_dict(data)
        assert config.iterations == 75
        assert config.warmup_count == 7
        assert config.label == "round-trip"

    def test_from_dict_with_defaults(self) -> None:
        config = BenchmarkConfig.from_dict({})
        assert config.iterations == 100
        assert config.warmup_count == 5
        assert config.label == ""

    def test_roundtrip_serialization(self) -> None:
        original = BenchmarkConfig(
            iterations=42,
            warmup_count=8,
            label="roundtrip-test",
        )
        restored = BenchmarkConfig.from_dict(original.to_dict())
        assert restored == original

    def test_equality(self) -> None:
        a = BenchmarkConfig(iterations=10, warmup_count=2, label="a")
        b = BenchmarkConfig(iterations=10, warmup_count=2, label="a")
        assert a == b

    def test_inequality(self) -> None:
        a = BenchmarkConfig(iterations=10, warmup_count=2)
        b = BenchmarkConfig(iterations=20, warmup_count=2)
        assert a != b


# ---------------------------------------------------------------------------
# BenchmarkResult tests
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """Tests for the frozen BenchmarkResult dataclass."""

    def test_create_with_all_fields(self) -> None:
        timings = (0.001, 0.002, 0.003, 0.004, 0.005)
        result = BenchmarkResult(
            raw_timings=timings,
            mean=0.003,
            stddev=0.001581,
            p50=0.003,
            p95=0.005,
            p99=0.005,
            min_time=0.001,
            max_time=0.005,
            samples=5,
            label="test-result",
        )
        assert result.raw_timings == timings
        assert result.mean == 0.003
        assert result.stddev == 0.001581
        assert result.p50 == 0.003
        assert result.p95 == 0.005
        assert result.p99 == 0.005
        assert result.min_time == 0.001
        assert result.max_time == 0.005
        assert result.samples == 5
        assert result.label == "test-result"

    def test_frozen_immutability(self) -> None:
        result = BenchmarkResult(
            raw_timings=(0.001,),
            mean=0.001,
            stddev=0.0,
            p50=0.001,
            p95=0.001,
            p99=0.001,
            min_time=0.001,
            max_time=0.001,
            samples=1,
            label="",
        )
        with pytest.raises(AttributeError):
            result.mean = 0.5  # type: ignore[misc]

    def test_negative_mean_raises(self) -> None:
        with pytest.raises(ValueError, match="mean must be >= 0.0"):
            BenchmarkResult(
                raw_timings=(0.001,),
                mean=-0.001,
                stddev=0.0,
                p50=0.001,
                p95=0.001,
                p99=0.001,
                min_time=0.001,
                max_time=0.001,
                samples=1,
                label="",
            )

    def test_negative_stddev_raises(self) -> None:
        with pytest.raises(ValueError, match="stddev must be >= 0.0"):
            BenchmarkResult(
                raw_timings=(0.001,),
                mean=0.001,
                stddev=-0.001,
                p50=0.001,
                p95=0.001,
                p99=0.001,
                min_time=0.001,
                max_time=0.001,
                samples=1,
                label="",
            )

    def test_zero_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="samples must be >= 1"):
            BenchmarkResult(
                raw_timings=(),
                mean=0.0,
                stddev=0.0,
                p50=0.0,
                p95=0.0,
                p99=0.0,
                min_time=0.0,
                max_time=0.0,
                samples=0,
                label="",
            )

    def test_negative_percentile_raises(self) -> None:
        with pytest.raises(ValueError, match="p50 must be >= 0.0"):
            BenchmarkResult(
                raw_timings=(0.001,),
                mean=0.001,
                stddev=0.0,
                p50=-0.001,
                p95=0.001,
                p99=0.001,
                min_time=0.001,
                max_time=0.001,
                samples=1,
                label="",
            )

    def test_negative_min_time_raises(self) -> None:
        with pytest.raises(ValueError, match="min_time must be >= 0.0"):
            BenchmarkResult(
                raw_timings=(0.001,),
                mean=0.001,
                stddev=0.0,
                p50=0.001,
                p95=0.001,
                p99=0.001,
                min_time=-0.001,
                max_time=0.001,
                samples=1,
                label="",
            )

    def test_to_dict(self) -> None:
        result = BenchmarkResult(
            raw_timings=(0.01, 0.02),
            mean=0.015,
            stddev=0.005,
            p50=0.015,
            p95=0.02,
            p99=0.02,
            min_time=0.01,
            max_time=0.02,
            samples=2,
            label="dict-test",
        )
        d = result.to_dict()
        assert d["raw_timings"] == [0.01, 0.02]
        assert d["mean"] == 0.015
        assert d["stddev"] == 0.005
        assert d["p50"] == 0.015
        assert d["p95"] == 0.02
        assert d["p99"] == 0.02
        assert d["min_time"] == 0.01
        assert d["max_time"] == 0.02
        assert d["samples"] == 2
        assert d["label"] == "dict-test"

    def test_from_dict(self) -> None:
        data = {
            "raw_timings": [0.01, 0.02, 0.03],
            "mean": 0.02,
            "stddev": 0.01,
            "p50": 0.02,
            "p95": 0.03,
            "p99": 0.03,
            "min_time": 0.01,
            "max_time": 0.03,
            "samples": 3,
            "label": "from-dict",
        }
        result = BenchmarkResult.from_dict(data)
        assert result.raw_timings == (0.01, 0.02, 0.03)
        assert result.mean == 0.02
        assert result.samples == 3
        assert result.label == "from-dict"

    def test_from_dict_defaults(self) -> None:
        """from_dict with minimal required fields."""
        data = {
            "raw_timings": [0.005],
            "mean": 0.005,
            "stddev": 0.0,
            "p50": 0.005,
            "p95": 0.005,
            "p99": 0.005,
            "min_time": 0.005,
            "max_time": 0.005,
            "samples": 1,
        }
        result = BenchmarkResult.from_dict(data)
        assert result.label == ""

    def test_roundtrip_serialization(self) -> None:
        original = BenchmarkResult(
            raw_timings=(0.001, 0.002, 0.003),
            mean=0.002,
            stddev=0.001,
            p50=0.002,
            p95=0.003,
            p99=0.003,
            min_time=0.001,
            max_time=0.003,
            samples=3,
            label="roundtrip",
        )
        restored = BenchmarkResult.from_dict(original.to_dict())
        assert restored == original

    def test_repr_includes_key_stats(self) -> None:
        result = BenchmarkResult(
            raw_timings=(0.001, 0.002),
            mean=0.0015,
            stddev=0.0005,
            p50=0.0015,
            p95=0.002,
            p99=0.002,
            min_time=0.001,
            max_time=0.002,
            samples=2,
            label="repr-test",
        )
        text = repr(result)
        assert "repr-test" in text
        assert "mean=" in text
        assert "p95=" in text
        assert "samples=2" in text


# ---------------------------------------------------------------------------
# compute_result tests
# ---------------------------------------------------------------------------


class TestComputeResult:
    """Tests for the compute_result factory function."""

    def test_basic_computation(self) -> None:
        timings = [0.010, 0.020, 0.030, 0.040, 0.050]
        result = compute_result(timings, label="basic")

        assert result.samples == 5
        assert result.label == "basic"
        assert result.raw_timings == tuple(timings)
        assert result.min_time == pytest.approx(0.010)
        assert result.max_time == pytest.approx(0.050)
        assert result.mean == pytest.approx(statistics.mean(timings))
        assert result.stddev == pytest.approx(statistics.stdev(timings))

    def test_single_sample(self) -> None:
        timings = [0.005]
        result = compute_result(timings, label="single")

        assert result.samples == 1
        assert result.mean == pytest.approx(0.005)
        assert result.stddev == 0.0  # single sample: stddev = 0
        assert result.p50 == pytest.approx(0.005)
        assert result.p95 == pytest.approx(0.005)
        assert result.p99 == pytest.approx(0.005)

    def test_empty_timings_raises(self) -> None:
        with pytest.raises(ValueError, match="timings must not be empty"):
            compute_result([], label="empty")

    def test_percentile_ordering(self) -> None:
        """p50 <= p95 <= p99 for any distribution."""
        timings = [float(i) / 1000.0 for i in range(1, 101)]
        result = compute_result(timings, label="ordering")

        assert result.p50 <= result.p95
        assert result.p95 <= result.p99
        assert result.min_time <= result.p50
        assert result.p99 <= result.max_time

    def test_uniform_distribution(self) -> None:
        """All-identical timings produce zero stddev."""
        timings = [0.010] * 50
        result = compute_result(timings, label="uniform")

        assert result.mean == pytest.approx(0.010)
        assert result.stddev == pytest.approx(0.0)
        assert result.p50 == pytest.approx(0.010)
        assert result.p95 == pytest.approx(0.010)
        assert result.p99 == pytest.approx(0.010)
        assert result.min_time == pytest.approx(0.010)
        assert result.max_time == pytest.approx(0.010)

    def test_default_label(self) -> None:
        result = compute_result([0.001, 0.002])
        assert result.label == ""

    def test_raw_timings_are_immutable_tuple(self) -> None:
        original = [0.001, 0.002, 0.003]
        result = compute_result(original, label="immutable")

        assert isinstance(result.raw_timings, tuple)
        # Mutating the original list must not affect the result
        original.append(0.999)
        assert len(result.raw_timings) == 3

    def test_negative_timing_raises(self) -> None:
        with pytest.raises(ValueError, match="All timings must be >= 0.0"):
            compute_result([0.001, -0.002, 0.003])
