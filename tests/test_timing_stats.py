"""Tests for the statistical analysis module (timing_stats).

Validates that the timing stats module correctly computes:
    - Percentiles (p50, p90, p95, p99) using nearest-rank method
    - Arithmetic mean
    - Standard deviation (population for 1 sample, sample for 2+)
    - Min and max values
    - Sample count
    - Serialization / deserialization roundtrip
    - Edge cases: single sample, two samples, identical values
    - Input validation: empty list, negative values
    - Immutability of frozen dataclass
    - Public percentile function for arbitrary percentiles
"""

from __future__ import annotations

import math

import pytest

from jules_daemon.agent.timing_stats import (
    TimingStats,
    compute_percentile,
    compute_timing_stats,
)


# ---------------------------------------------------------------------------
# compute_percentile (public function)
# ---------------------------------------------------------------------------


class TestComputePercentile:
    """Tests for the standalone percentile computation function."""

    def test_median_of_odd_count(self) -> None:
        data = (1.0, 2.0, 3.0, 4.0, 5.0)
        result = compute_percentile(data, 50)
        assert result == 3.0

    def test_median_of_even_count(self) -> None:
        data = (1.0, 2.0, 3.0, 4.0)
        result = compute_percentile(data, 50)
        assert result == 2.0

    def test_p0_returns_minimum(self) -> None:
        data = (10.0, 20.0, 30.0)
        result = compute_percentile(data, 0)
        assert result == 10.0

    def test_p100_returns_maximum(self) -> None:
        data = (10.0, 20.0, 30.0)
        result = compute_percentile(data, 100)
        assert result == 30.0

    def test_p90_of_ten_elements(self) -> None:
        """With 10 elements, p90 should return the 9th element (index 8)."""
        data = tuple(float(i) for i in range(1, 11))  # 1..10
        result = compute_percentile(data, 90)
        assert result == 9.0

    def test_p95_of_twenty_elements(self) -> None:
        data = tuple(float(i) for i in range(1, 21))  # 1..20
        result = compute_percentile(data, 95)
        assert result == 19.0

    def test_p99_of_hundred_elements(self) -> None:
        data = tuple(float(i) for i in range(1, 101))  # 1..100
        result = compute_percentile(data, 99)
        assert result == 99.0

    def test_empty_data_returns_zero(self) -> None:
        assert compute_percentile((), 50) == 0.0

    def test_single_element(self) -> None:
        data = (42.0,)
        assert compute_percentile(data, 0) == 42.0
        assert compute_percentile(data, 50) == 42.0
        assert compute_percentile(data, 100) == 42.0

    def test_unsorted_input_not_modified(self) -> None:
        """Function requires sorted input; unsorted input gives wrong results
        but must not raise (contract: caller sorts)."""
        data = (5.0, 1.0, 3.0)
        # Just verify it does not crash
        compute_percentile(data, 50)

    def test_negative_percentile_clamped(self) -> None:
        """Percentile below 0 should still return a valid value (clamped)."""
        data = (1.0, 2.0, 3.0)
        result = compute_percentile(data, -10)
        assert result == 1.0

    def test_percentile_above_100_clamped(self) -> None:
        """Percentile above 100 returns max element."""
        data = (1.0, 2.0, 3.0)
        result = compute_percentile(data, 110)
        assert result == 3.0


# ---------------------------------------------------------------------------
# TimingStats frozen dataclass
# ---------------------------------------------------------------------------


class TestTimingStatsDataclass:
    """Tests for TimingStats immutability and validation."""

    def test_fields_present(self) -> None:
        stats = TimingStats(
            raw_timings=(1.0, 2.0, 3.0),
            mean=2.0,
            stddev=1.0,
            p50=2.0,
            p90=3.0,
            p95=3.0,
            p99=3.0,
            min_val=1.0,
            max_val=3.0,
            sample_count=3,
        )
        assert stats.mean == 2.0
        assert stats.stddev == 1.0
        assert stats.p50 == 2.0
        assert stats.p90 == 3.0
        assert stats.p95 == 3.0
        assert stats.p99 == 3.0
        assert stats.min_val == 1.0
        assert stats.max_val == 3.0
        assert stats.sample_count == 3
        assert stats.raw_timings == (1.0, 2.0, 3.0)

    def test_frozen_immutability(self) -> None:
        stats = TimingStats(
            raw_timings=(1.0,),
            mean=1.0,
            stddev=0.0,
            p50=1.0,
            p90=1.0,
            p95=1.0,
            p99=1.0,
            min_val=1.0,
            max_val=1.0,
            sample_count=1,
        )
        with pytest.raises(AttributeError):
            stats.mean = 99.0  # type: ignore[misc]

    def test_negative_mean_rejected(self) -> None:
        with pytest.raises(ValueError, match="mean"):
            TimingStats(
                raw_timings=(1.0,),
                mean=-1.0,
                stddev=0.0,
                p50=1.0,
                p90=1.0,
                p95=1.0,
                p99=1.0,
                min_val=1.0,
                max_val=1.0,
                sample_count=1,
            )

    def test_negative_stddev_rejected(self) -> None:
        with pytest.raises(ValueError, match="stddev"):
            TimingStats(
                raw_timings=(1.0,),
                mean=1.0,
                stddev=-0.5,
                p50=1.0,
                p90=1.0,
                p95=1.0,
                p99=1.0,
                min_val=1.0,
                max_val=1.0,
                sample_count=1,
            )

    def test_zero_sample_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="sample_count"):
            TimingStats(
                raw_timings=(),
                mean=0.0,
                stddev=0.0,
                p50=0.0,
                p90=0.0,
                p95=0.0,
                p99=0.0,
                min_val=0.0,
                max_val=0.0,
                sample_count=0,
            )

    def test_negative_percentile_rejected(self) -> None:
        with pytest.raises(ValueError, match="p50"):
            TimingStats(
                raw_timings=(1.0,),
                mean=1.0,
                stddev=0.0,
                p50=-1.0,
                p90=1.0,
                p95=1.0,
                p99=1.0,
                min_val=1.0,
                max_val=1.0,
                sample_count=1,
            )

    def test_negative_min_val_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_val"):
            TimingStats(
                raw_timings=(1.0,),
                mean=1.0,
                stddev=0.0,
                p50=1.0,
                p90=1.0,
                p95=1.0,
                p99=1.0,
                min_val=-1.0,
                max_val=1.0,
                sample_count=1,
            )

    def test_negative_max_val_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_val"):
            TimingStats(
                raw_timings=(1.0,),
                mean=1.0,
                stddev=0.0,
                p50=1.0,
                p90=1.0,
                p95=1.0,
                p99=1.0,
                min_val=1.0,
                max_val=-1.0,
                sample_count=1,
            )

    def test_negative_p90_rejected(self) -> None:
        with pytest.raises(ValueError, match="p90"):
            TimingStats(
                raw_timings=(1.0,),
                mean=1.0,
                stddev=0.0,
                p50=1.0,
                p90=-1.0,
                p95=1.0,
                p99=1.0,
                min_val=1.0,
                max_val=1.0,
                sample_count=1,
            )

    def test_negative_p95_rejected(self) -> None:
        with pytest.raises(ValueError, match="p95"):
            TimingStats(
                raw_timings=(1.0,),
                mean=1.0,
                stddev=0.0,
                p50=1.0,
                p90=1.0,
                p95=-1.0,
                p99=1.0,
                min_val=1.0,
                max_val=1.0,
                sample_count=1,
            )

    def test_negative_p99_rejected(self) -> None:
        with pytest.raises(ValueError, match="p99"):
            TimingStats(
                raw_timings=(1.0,),
                mean=1.0,
                stddev=0.0,
                p50=1.0,
                p90=1.0,
                p95=1.0,
                p99=-1.0,
                min_val=1.0,
                max_val=1.0,
                sample_count=1,
            )


# ---------------------------------------------------------------------------
# TimingStats serialization
# ---------------------------------------------------------------------------


class TestTimingStatsSerialization:
    """Tests for to_dict / from_dict roundtrip."""

    def test_roundtrip(self) -> None:
        original = TimingStats(
            raw_timings=(0.1, 0.2, 0.3),
            mean=0.2,
            stddev=0.1,
            p50=0.2,
            p90=0.3,
            p95=0.3,
            p99=0.3,
            min_val=0.1,
            max_val=0.3,
            sample_count=3,
        )
        data = original.to_dict()
        restored = TimingStats.from_dict(data)

        assert restored == original

    def test_to_dict_raw_timings_is_list(self) -> None:
        """raw_timings should serialize as a list for JSON compatibility."""
        stats = TimingStats(
            raw_timings=(1.0, 2.0),
            mean=1.5,
            stddev=0.5,
            p50=1.0,
            p90=2.0,
            p95=2.0,
            p99=2.0,
            min_val=1.0,
            max_val=2.0,
            sample_count=2,
        )
        d = stats.to_dict()
        assert isinstance(d["raw_timings"], list)

    def test_from_dict_converts_list_to_tuple(self) -> None:
        data = {
            "raw_timings": [1.0, 2.0],
            "mean": 1.5,
            "stddev": 0.5,
            "p50": 1.0,
            "p90": 2.0,
            "p95": 2.0,
            "p99": 2.0,
            "min_val": 1.0,
            "max_val": 2.0,
            "sample_count": 2,
        }
        stats = TimingStats.from_dict(data)
        assert isinstance(stats.raw_timings, tuple)


# ---------------------------------------------------------------------------
# compute_timing_stats factory
# ---------------------------------------------------------------------------


class TestComputeTimingStats:
    """Tests for the compute_timing_stats factory function."""

    def test_returns_timing_stats(self) -> None:
        result = compute_timing_stats([1.0, 2.0, 3.0])
        assert isinstance(result, TimingStats)

    def test_sample_count(self) -> None:
        result = compute_timing_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.sample_count == 5

    def test_raw_timings_preserved_as_tuple(self) -> None:
        timings = [0.1, 0.2, 0.3]
        result = compute_timing_stats(timings)
        assert result.raw_timings == (0.1, 0.2, 0.3)

    def test_input_not_mutated(self) -> None:
        timings = [3.0, 1.0, 2.0]
        original = list(timings)
        compute_timing_stats(timings)
        assert timings == original

    def test_mean_computation(self) -> None:
        result = compute_timing_stats([2.0, 4.0, 6.0])
        assert result.mean == pytest.approx(4.0)

    def test_min_max(self) -> None:
        result = compute_timing_stats([5.0, 1.0, 10.0, 3.0])
        assert result.min_val == 1.0
        assert result.max_val == 10.0

    def test_stddev_single_sample(self) -> None:
        """Single sample: stddev should be 0.0."""
        result = compute_timing_stats([42.0])
        assert result.stddev == 0.0

    def test_stddev_two_samples(self) -> None:
        result = compute_timing_stats([2.0, 4.0])
        # statistics.stdev([2.0, 4.0]) = sqrt(2) ~= 1.4142
        assert result.stddev == pytest.approx(math.sqrt(2.0), rel=1e-6)

    def test_stddev_multiple_samples(self) -> None:
        result = compute_timing_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        # statistics.stdev([1,2,3,4,5]) = sqrt(2.5) ~= 1.5811
        assert result.stddev == pytest.approx(math.sqrt(2.5), rel=1e-6)

    def test_percentiles_ordered(self) -> None:
        """p50 <= p90 <= p95 <= p99 for monotonically increasing data."""
        timings = [float(i) for i in range(1, 101)]
        result = compute_timing_stats(timings)

        assert result.p50 <= result.p90
        assert result.p90 <= result.p95
        assert result.p95 <= result.p99

    def test_p50_correctness(self) -> None:
        """With 100 values [1..100], p50 should be around 50."""
        timings = [float(i) for i in range(1, 101)]
        result = compute_timing_stats(timings)
        assert result.p50 == 50.0

    def test_p90_correctness(self) -> None:
        """With 100 values [1..100], p90 should be around 90."""
        timings = [float(i) for i in range(1, 101)]
        result = compute_timing_stats(timings)
        assert result.p90 == 90.0

    def test_p95_correctness(self) -> None:
        """With 100 values [1..100], p95 should be around 95."""
        timings = [float(i) for i in range(1, 101)]
        result = compute_timing_stats(timings)
        assert result.p95 == 95.0

    def test_p99_correctness(self) -> None:
        """With 100 values [1..100], p99 should be around 99."""
        timings = [float(i) for i in range(1, 101)]
        result = compute_timing_stats(timings)
        assert result.p99 == 99.0

    def test_all_identical_values(self) -> None:
        """When all values are the same, all stats equal that value."""
        result = compute_timing_stats([5.0, 5.0, 5.0, 5.0])
        assert result.mean == 5.0
        assert result.stddev == 0.0
        assert result.p50 == 5.0
        assert result.p90 == 5.0
        assert result.p95 == 5.0
        assert result.p99 == 5.0
        assert result.min_val == 5.0
        assert result.max_val == 5.0

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            compute_timing_stats([])

    def test_negative_value_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            compute_timing_stats([1.0, -0.5, 2.0])

    def test_single_sample(self) -> None:
        result = compute_timing_stats([7.5])
        assert result.mean == 7.5
        assert result.stddev == 0.0
        assert result.p50 == 7.5
        assert result.p90 == 7.5
        assert result.p95 == 7.5
        assert result.p99 == 7.5
        assert result.min_val == 7.5
        assert result.max_val == 7.5
        assert result.sample_count == 1

    def test_two_samples(self) -> None:
        result = compute_timing_stats([10.0, 20.0])
        assert result.mean == pytest.approx(15.0)
        assert result.min_val == 10.0
        assert result.max_val == 20.0
        assert result.sample_count == 2

    def test_unsorted_input(self) -> None:
        """compute_timing_stats should handle unsorted input correctly."""
        result = compute_timing_stats([5.0, 1.0, 3.0, 2.0, 4.0])
        assert result.min_val == 1.0
        assert result.max_val == 5.0
        assert result.mean == pytest.approx(3.0)

    def test_zero_values_allowed(self) -> None:
        """Timing of 0.0 is valid (very fast operation)."""
        result = compute_timing_stats([0.0, 0.0, 0.0])
        assert result.mean == 0.0
        assert result.min_val == 0.0
        assert result.max_val == 0.0

    def test_label_parameter(self) -> None:
        result = compute_timing_stats([1.0, 2.0, 3.0], label="test-label")
        assert result.label == "test-label"

    def test_default_label(self) -> None:
        result = compute_timing_stats([1.0, 2.0, 3.0])
        assert result.label == ""

    def test_large_dataset(self) -> None:
        """Verify correctness with a larger dataset (1000 values)."""
        timings = [float(i) / 1000.0 for i in range(1, 1001)]
        result = compute_timing_stats(timings)

        assert result.sample_count == 1000
        assert result.min_val == pytest.approx(0.001)
        assert result.max_val == pytest.approx(1.0)
        assert result.mean == pytest.approx(0.5005, rel=1e-3)
        # Percentiles should be approximately correct
        assert result.p50 == pytest.approx(0.500, abs=0.002)
        assert result.p90 == pytest.approx(0.900, abs=0.002)
        assert result.p95 == pytest.approx(0.950, abs=0.002)
        assert result.p99 == pytest.approx(0.990, abs=0.002)


# ---------------------------------------------------------------------------
# TimingStats repr
# ---------------------------------------------------------------------------


class TestTimingStatsRepr:
    """Test __repr__ formatting."""

    def test_repr_contains_key_fields(self) -> None:
        stats = compute_timing_stats([0.001, 0.002, 0.003])
        text = repr(stats)
        assert "mean=" in text
        assert "p50=" in text
        assert "p90=" in text
        assert "p95=" in text
        assert "p99=" in text
        assert "stddev=" in text
        assert "samples=" in text

    def test_repr_label_when_set(self) -> None:
        stats = compute_timing_stats([1.0, 2.0], label="mytest")
        text = repr(stats)
        assert "mytest" in text
