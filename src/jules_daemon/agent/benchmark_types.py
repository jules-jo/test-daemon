"""Benchmark harness data types and configuration for agent loop performance.

Provides immutable dataclasses for configuring benchmark runs and storing
their results, following the project convention of frozen dataclasses
with validation, serialization, and computed statistics.

``BenchmarkConfig`` defines run parameters: how many timed iterations to
execute and how many warm-up passes to discard (to exclude first-call
initialization overhead like regex compilation, import-time work, etc.).

``BenchmarkResult`` captures the full statistical output of a benchmark
run: raw timing samples, computed percentiles (p50/p95/p99), mean,
standard deviation, and min/max bounds.

``compute_result`` is a factory function that takes a list of raw timing
values and produces a validated ``BenchmarkResult`` with all statistics
pre-computed.  This separates measurement (the caller's responsibility)
from statistical computation (this module's responsibility).

Usage::

    from jules_daemon.agent.benchmark_types import (
        BenchmarkConfig,
        BenchmarkResult,
        compute_result,
    )

    config = BenchmarkConfig(iterations=100, warmup_count=5)

    # ... run measurements and collect timings ...
    timings = [0.001, 0.002, 0.0015, ...]

    result = compute_result(timings, label="agent-loop-overhead")
    assert result.p95 < AGENT_LOOP_OVERHEAD_PER_ITERATION_S
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "compute_result",
]


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


def _percentile(sorted_data: tuple[float, ...], pct: float) -> float:
    """Compute the given percentile from a sorted tuple.

    Uses the nearest-rank method: index = ceil(pct/100 * n) - 1,
    clamped to [0, n-1].  The input must be pre-sorted ascending.

    Args:
        sorted_data: Sorted tuple of float values.
        pct: Percentile to compute (0--100).

    Returns:
        The value at the requested percentile, or 0.0 if empty.
    """
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    idx = max(0, min(n - 1, int((pct / 100.0) * n + 0.5) - 1))
    return sorted_data[idx]


# ---------------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable configuration for a benchmark run.

    Controls how many timed iterations to execute and how many untimed
    warm-up passes to perform beforehand.  Warm-up passes ensure that
    first-call initialization costs (regex compilation, module-level
    frozenset construction, JIT effects) are excluded from measurements.

    Attributes:
        iterations: Number of timed iterations to execute.  Each iteration
            produces one raw timing sample.  Must be >= 1.
        warmup_count: Number of untimed warm-up passes before measurement.
            Must be >= 0.  Set to 0 to disable warm-up.
        label: Human-readable label identifying this benchmark run
            (e.g., "agent-loop-overhead", "tool-registry-lookup").
            Empty string when not specified.
    """

    iterations: int = 100
    warmup_count: int = 5
    label: str = ""

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError(
                f"iterations must be >= 1, got {self.iterations}"
            )
        if self.warmup_count < 0:
            raise ValueError(
                f"warmup_count must be >= 0, got {self.warmup_count}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for persistence or logging."""
        return {
            "iterations": self.iterations,
            "warmup_count": self.warmup_count,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """Deserialize from a plain dict.

        Missing keys fall back to the dataclass defaults.

        Args:
            data: Dictionary with optional keys matching field names.

        Returns:
            Validated BenchmarkConfig instance.
        """
        return cls(
            iterations=data.get("iterations", 100),
            warmup_count=data.get("warmup_count", 5),
            label=data.get("label", ""),
        )


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    """Immutable result of a benchmark measurement run.

    Stores the raw timing data together with pre-computed statistics.
    All timing values are in seconds (float).

    Attributes:
        raw_timings: Tuple of individual timing samples in seconds,
            in the order they were collected (not sorted).
        mean: Arithmetic mean of all samples.
        stddev: Population standard deviation.  0.0 for single-sample runs.
        p50: 50th percentile (median) of timing samples.
        p95: 95th percentile.
        p99: 99th percentile.
        min_time: Fastest individual sample.
        max_time: Slowest individual sample.
        samples: Total number of timing samples.
        label: Human-readable label identifying this result
            (carried over from ``BenchmarkConfig.label``).
    """

    raw_timings: tuple[float, ...]
    mean: float
    stddev: float
    p50: float
    p95: float
    p99: float
    min_time: float
    max_time: float
    samples: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.mean < 0.0:
            raise ValueError(f"mean must be >= 0.0, got {self.mean}")
        if self.stddev < 0.0:
            raise ValueError(f"stddev must be >= 0.0, got {self.stddev}")
        if self.p50 < 0.0:
            raise ValueError(f"p50 must be >= 0.0, got {self.p50}")
        if self.p95 < 0.0:
            raise ValueError(f"p95 must be >= 0.0, got {self.p95}")
        if self.p99 < 0.0:
            raise ValueError(f"p99 must be >= 0.0, got {self.p99}")
        if self.min_time < 0.0:
            raise ValueError(f"min_time must be >= 0.0, got {self.min_time}")
        if self.max_time < 0.0:
            raise ValueError(f"max_time must be >= 0.0, got {self.max_time}")
        if self.samples < 1:
            raise ValueError(f"samples must be >= 1, got {self.samples}")

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult("
            f"label={self.label!r}, "
            f"mean={self.mean * 1000:.3f}ms, "
            f"p50={self.p50 * 1000:.3f}ms, "
            f"p95={self.p95 * 1000:.3f}ms, "
            f"p99={self.p99 * 1000:.3f}ms, "
            f"stddev={self.stddev * 1000:.3f}ms, "
            f"samples={self.samples})"
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
            "p95": self.p95,
            "p99": self.p99,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "samples": self.samples,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Deserialize from a plain dict.

        Converts ``raw_timings`` from list back to tuple for immutability.

        Args:
            data: Dictionary with keys matching field names.

        Returns:
            Validated BenchmarkResult instance.
        """
        raw = data.get("raw_timings", ())
        if isinstance(raw, list):
            raw = tuple(raw)
        return cls(
            raw_timings=raw,
            mean=data["mean"],
            stddev=data["stddev"],
            p50=data["p50"],
            p95=data["p95"],
            p99=data["p99"],
            min_time=data["min_time"],
            max_time=data["max_time"],
            samples=data["samples"],
            label=data.get("label", ""),
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def compute_result(
    timings: list[float],
    label: str = "",
) -> BenchmarkResult:
    """Compute a BenchmarkResult from raw timing measurements.

    Takes a list of raw timing values (in seconds) and computes all
    statistical fields: mean, stddev, percentiles, min, max.

    The input list is not modified.  The resulting ``raw_timings`` field
    is an immutable tuple copy of the input.

    Args:
        timings: List of timing samples in seconds.  Must not be empty.
            All values must be >= 0.0.
        label: Human-readable label for the result.

    Returns:
        Fully computed BenchmarkResult.

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

    return BenchmarkResult(
        raw_timings=raw,
        mean=mean,
        stddev=stddev,
        p50=_percentile(sorted_timings, 50),
        p95=_percentile(sorted_timings, 95),
        p99=_percentile(sorted_timings, 99),
        min_time=sorted_timings[0],
        max_time=sorted_timings[-1],
        samples=n,
        label=label,
    )
