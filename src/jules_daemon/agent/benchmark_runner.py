"""Core benchmark runner for agent loop performance measurement.

Provides functions that execute a callable N times with optional warm-up
iterations and collect raw latency measurements using ``time.monotonic()``.
Results are returned as immutable ``BenchmarkResult`` instances with
pre-computed statistics (mean, stddev, percentiles).

Two runner variants are provided:

    ``run_benchmark``
        Synchronous runner for CPU-bound callables (e.g., regex compilation,
        tool registry lookups, direct-command detection). Uses a tight
        ``time.monotonic()`` loop for minimal measurement overhead.

    ``run_benchmark_async``
        Async runner for coroutine callables (e.g., agent loop iterations,
        LLM adapter calls). Each iteration ``await``s the target coroutine
        and measures wall-clock time including event-loop scheduling.

Both runners share the same contract:

    1. Execute ``config.warmup_count`` untimed warm-up passes.
       Warm-up ensures first-call initialization costs (regex compilation,
       module-level frozenset construction, import-time work) are excluded
       from measurements.

    2. Execute ``config.iterations`` timed passes. Each pass records the
       wall-clock elapsed time via ``time.monotonic()`` before/after the
       callable invocation.

    3. Pass the collected timing samples to ``compute_result()`` to produce
       an immutable ``BenchmarkResult`` with statistics.

Errors raised by the target callable propagate immediately -- if the
callable fails during warm-up or a timed iteration, the runner does not
catch or suppress the error. This is intentional: benchmark targets should
be deterministic, and a failure indicates a bug in the target, not the
runner.

Usage::

    from jules_daemon.agent.benchmark_runner import run_benchmark
    from jules_daemon.agent.benchmark_types import BenchmarkConfig

    config = BenchmarkConfig(iterations=100, warmup_count=5, label="detect")

    def target():
        detect_direct_command("pytest -v tests/")

    result = run_benchmark(target, config)
    assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from jules_daemon.agent.benchmark_types import (
    BenchmarkConfig,
    BenchmarkResult,
    compute_result,
)

__all__ = [
    "BenchmarkTarget",
    "AsyncBenchmarkTarget",
    "run_benchmark",
    "run_benchmark_async",
]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BenchmarkTarget = Callable[[], None]
"""Synchronous callable with no arguments and no return value.

The benchmark runner invokes this callable once per iteration. Wrap
parameterized functions with ``functools.partial`` or a lambda before
passing them as a benchmark target.
"""

AsyncBenchmarkTarget = Callable[[], Awaitable[None]]
"""Async callable with no arguments and no return value.

Same contract as ``BenchmarkTarget`` but for coroutine functions. The
async runner ``await``s the returned awaitable on each iteration.
"""


# ---------------------------------------------------------------------------
# Synchronous runner
# ---------------------------------------------------------------------------


def run_benchmark(
    target: BenchmarkTarget,
    config: BenchmarkConfig | None = None,
    *,
    label: str | None = None,
) -> BenchmarkResult:
    """Execute a synchronous callable N times and collect latency samples.

    Runs ``config.warmup_count`` untimed warm-up passes followed by
    ``config.iterations`` timed passes. Each timed pass measures
    wall-clock time via ``time.monotonic()`` surrounding the target
    invocation.

    Args:
        target: Synchronous callable to benchmark. Must accept no arguments.
            Errors raised by the target propagate immediately.
        config: Benchmark configuration. If None, uses ``BenchmarkConfig()``
            defaults (100 iterations, 5 warm-up).
        label: Optional label override. If provided, takes precedence over
            ``config.label``. If None, ``config.label`` is used.

    Returns:
        Immutable ``BenchmarkResult`` with raw timings and computed
        statistics (mean, stddev, p50, p95, p99, min, max).

    Raises:
        Any exception raised by ``target`` during warm-up or timed
        iterations.
    """
    effective_config = config if config is not None else BenchmarkConfig()
    effective_label = label if label is not None else effective_config.label

    # -- Warm-up phase (untimed) --
    for _ in range(effective_config.warmup_count):
        target()

    # -- Timed phase --
    timings: list[float] = []
    for _ in range(effective_config.iterations):
        start = time.monotonic()
        target()
        elapsed = time.monotonic() - start
        timings.append(elapsed)

    return compute_result(timings, label=effective_label)


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------


async def run_benchmark_async(
    target: AsyncBenchmarkTarget,
    config: BenchmarkConfig | None = None,
    *,
    label: str | None = None,
) -> BenchmarkResult:
    """Execute an async callable N times and collect latency samples.

    Async counterpart of ``run_benchmark``. Runs warm-up passes and timed
    passes by ``await``ing the target coroutine. Measurement includes
    event-loop scheduling overhead, which is representative of real-world
    async workloads.

    Args:
        target: Async callable to benchmark. Must accept no arguments and
            return an awaitable. Errors propagate immediately.
        config: Benchmark configuration. If None, uses ``BenchmarkConfig()``
            defaults (100 iterations, 5 warm-up).
        label: Optional label override. If provided, takes precedence over
            ``config.label``. If None, ``config.label`` is used.

    Returns:
        Immutable ``BenchmarkResult`` with raw timings and computed
        statistics.

    Raises:
        Any exception raised by ``target`` during warm-up or timed
        iterations.
    """
    effective_config = config if config is not None else BenchmarkConfig()
    effective_label = label if label is not None else effective_config.label

    # -- Warm-up phase (untimed) --
    for _ in range(effective_config.warmup_count):
        await target()

    # -- Timed phase --
    timings: list[float] = []
    for _ in range(effective_config.iterations):
        start = time.monotonic()
        await target()
        elapsed = time.monotonic() - start
        timings.append(elapsed)

    return compute_result(timings, label=effective_label)
