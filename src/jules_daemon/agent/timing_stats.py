"""Statistical analysis module for raw timing data.

Provides immutable dataclasses and pure functions for computing
descriptive statistics from timing measurements. Designed as a
general-purpose statistics layer used by benchmarks, test-run
analysis, and the agent loop's performance monitoring.

Core API:

    ``compute_percentile(sorted_data, pct)``
        Public function computing a single percentile from pre-sorted data
        using the nearest-rank method.

    ``TimingStats``
        Frozen dataclass holding raw timings alongside pre-computed
        statistics: mean, standard deviation, percentiles (p50, p90, p95,
        p99), and min/max bounds.  Includes ``to_dict()`` / ``from_dict()``
        for JSON-compatible serialization.

    ``compute_timing_stats(timings, label)``
        Factory function that takes a list of raw timing values and
        produces a validated ``TimingStats`` with all statistics
        pre-computed.  Separates measurement (caller's responsibility)
        from statistical computation (this module's responsibility).

Usage::

    from jules_daemon.agent.timing_stats import (
        TimingStats,
        compute_timing_stats,
    )

    timings = [0.001, 0.002, 0.0015, 0.003, 0.0012]
    stats = compute_timing_stats(timings, label="agent-loop-overhead")

    assert stats.p95 < 0.050  # 50ms budget
    assert stats.mean < 0.010  # 10ms average

All timing values are in seconds (float).  All percentiles use the
nearest-rank method for consistency with the existing benchmark_types
module.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

__all__ = [
    "TimingStats",
    "compute_percentile",
    "compute_timing_stats",
]


# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------


def compute_percentile(sorted_data: tuple[float, ...], pct: float) -> float:
    """Compute a percentile from a pre-sorted tuple of floats.

    Uses the nearest-rank method: index = ceil(pct/100 * n) - 1,
    clamped to [0, n-1].  The input **must** be pre-sorted in ascending
    order; no sorting is performed internally.

    Args:
        sorted_data: Tuple of float values sorted in ascending order.
        pct: Percentile to compute (0--100).  Values outside [0, 100]
            are clamped to the nearest boundary.

    Returns:
        The value at the requested percentile, or 0.0 if the input
        is empty.
    """
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    idx = max(0, min(n - 1, int((pct / 100.0) * n + 0.5) - 1))
    return sorted_data[idx]


# ---------------------------------------------------------------------------
# TimingStats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingStats:
    """Immutable statistical summary of raw timing measurements.

    All timing values are in seconds (float).  The ``raw_timings`` field
    preserves the original measurement order (not sorted).

    Attributes:
        raw_timings: Tuple of individual timing samples in seconds,
            in the order they were collected (not sorted).
        mean: Arithmetic mean of all samples.
        stddev: Sample standard deviation (``statistics.stdev``).
            0.0 for single-sample runs.
        p50: 50th percentile (median).
        p90: 90th percentile.
        p95: 95th percentile.
        p99: 99th percentile.
        min_val: Minimum (fastest) sample.
        max_val: Maximum (slowest) sample.
        sample_count: Total number of timing samples.
        label: Human-readable label identifying this measurement set.
            Empty string when not specified.
    """

    raw_timings: tuple[float, ...]
    mean: float
    stddev: float
    p50: float
    p90: float
    p95: float
    p99: float
    min_val: float
    max_val: float
    sample_count: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.mean < 0.0:
            raise ValueError(f"mean must be >= 0.0, got {self.mean}")
        if self.stddev < 0.0:
            raise ValueError(f"stddev must be >= 0.0, got {self.stddev}")
        if self.p50 < 0.0:
            raise ValueError(f"p50 must be >= 0.0, got {self.p50}")
        if self.p90 < 0.0:
            raise ValueError(f"p90 must be >= 0.0, got {self.p90}")
        if self.p95 < 0.0:
            raise ValueError(f"p95 must be >= 0.0, got {self.p95}")
        if self.p99 < 0.0:
            raise ValueError(f"p99 must be >= 0.0, got {self.p99}")
        if self.min_val < 0.0:
            raise ValueError(f"min_val must be >= 0.0, got {self.min_val}")
        if self.max_val < 0.0:
            raise ValueError(f"max_val must be >= 0.0, got {self.max_val}")
        if self.sample_count < 1:
            raise ValueError(
                f"sample_count must be >= 1, got {self.sample_count}"
            )

    def __repr__(self) -> str:
        return (
            f"TimingStats("
            f"label={self.label!r}, "
            f"mean={self.mean * 1000:.3f}ms, "
            f"p50={self.p50 * 1000:.3f}ms, "
            f"p90={self.p90 * 1000:.3f}ms, "
            f"p95={self.p95 * 1000:.3f}ms, "
            f"p99={self.p99 * 1000:.3f}ms, "
            f"stddev={self.stddev * 1000:.3f}ms, "
            f"samples={self.sample_count})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for persistence or logging.

        Raw timings are stored as a list (not tuple) for JSON compatibility.
        """
        return {
            "raw_timings": list(self.raw_timings),
            "mean": self.mean,
            "stddev": self.stddev,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "sample_count": self.sample_count,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimingStats:
        """Deserialize from a plain dict.

        Converts ``raw_timings`` from list back to tuple for immutability.

        Args:
            data: Dictionary with keys matching field names.

        Returns:
            Validated TimingStats instance.
        """
        raw = data.get("raw_timings", ())
        if isinstance(raw, list):
            raw = tuple(raw)
        return cls(
            raw_timings=raw,
            mean=data["mean"],
            stddev=data["stddev"],
            p50=data["p50"],
            p90=data["p90"],
            p95=data["p95"],
            p99=data["p99"],
            min_val=data["min_val"],
            max_val=data["max_val"],
            sample_count=data["sample_count"],
            label=data.get("label", ""),
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def compute_timing_stats(
    timings: list[float],
    label: str = "",
) -> TimingStats:
    """Compute a TimingStats from raw timing measurements.

    Takes a list of raw timing values (in seconds) and computes all
    statistical fields: mean, stddev, percentiles (p50, p90, p95, p99),
    min, and max.

    The input list is not modified.  The resulting ``raw_timings`` field
    is an immutable tuple copy of the input.

    Args:
        timings: List of timing samples in seconds.  Must not be empty.
            All values must be >= 0.0.
        label: Human-readable label for the result.

    Returns:
        Fully computed TimingStats.

    Raises:
        ValueError: If timings is empty or contains negative values.
    """
    if not timings:
        raise ValueError("timings must not be empty")

    if any(t < 0.0 for t in timings):
        raise ValueError("All timings must be >= 0.0")

    raw = tuple(timings)
    sorted_timings = tuple(sorted(raw))
    n = len(raw)

    mean = statistics.mean(raw)

    if n < 2:
        stddev = 0.0
    else:
        stddev = statistics.stdev(raw)

    return TimingStats(
        raw_timings=raw,
        mean=mean,
        stddev=stddev,
        p50=compute_percentile(sorted_timings, 50),
        p90=compute_percentile(sorted_timings, 90),
        p95=compute_percentile(sorted_timings, 95),
        p99=compute_percentile(sorted_timings, 99),
        min_val=sorted_timings[0],
        max_val=sorted_timings[-1],
        sample_count=n,
        label=label,
    )
