"""Tests for the core benchmark runner function.

Validates that the benchmark runner:
    - Executes warm-up iterations without timing
    - Executes N timed iterations and collects raw latency samples
    - Supports both sync and async callables
    - Returns a valid BenchmarkResult with correct sample count
    - Respects BenchmarkConfig parameters (iterations, warmup_count, label)
    - Handles zero warm-up correctly
    - Handles callable errors by propagating them cleanly
    - Does not mutate the BenchmarkConfig
    - Produces monotonic-clock-based measurements (non-negative)
    - Supports async runner variant for async callables
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from jules_daemon.agent.benchmark_runner import (
    BenchmarkTarget,
    run_benchmark,
    run_benchmark_async,
)
from jules_daemon.agent.benchmark_types import (
    BenchmarkConfig,
    BenchmarkResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CallCounter:
    """Tracks the number of times a callable is invoked.

    Provides both sync and async entry points for benchmark runner tests.
    """

    def __init__(self, work_time: float = 0.0) -> None:
        self.call_count = 0
        self._work_time = work_time

    def sync_fn(self) -> None:
        self.call_count += 1
        if self._work_time > 0:
            # Busy-wait to simulate work without OS scheduling jitter.
            end = time.monotonic() + self._work_time
            while time.monotonic() < end:
                pass

    async def async_fn(self) -> None:
        self.call_count += 1
        if self._work_time > 0:
            await asyncio.sleep(self._work_time)


class FailAfterN:
    """Callable that succeeds N times then raises an exception."""

    def __init__(self, succeed_count: int) -> None:
        self._succeed_count = succeed_count
        self.call_count = 0

    def __call__(self) -> None:
        self.call_count += 1
        if self.call_count > self._succeed_count:
            raise RuntimeError(f"Deliberate failure on call {self.call_count}")


# ---------------------------------------------------------------------------
# run_benchmark (sync) tests
# ---------------------------------------------------------------------------


class TestRunBenchmarkBasic:
    """Core behavior of the synchronous benchmark runner."""

    def test_returns_benchmark_result(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(iterations=10, warmup_count=2, label="basic")
        result = run_benchmark(counter.sync_fn, config)

        assert isinstance(result, BenchmarkResult)

    def test_correct_sample_count(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(iterations=20, warmup_count=3)
        result = run_benchmark(counter.sync_fn, config)

        assert result.samples == 20
        assert len(result.raw_timings) == 20

    def test_total_calls_include_warmup(self) -> None:
        """Warmup + timed iterations = total invocations."""
        counter = CallCounter()
        config = BenchmarkConfig(iterations=10, warmup_count=5)
        run_benchmark(counter.sync_fn, config)

        assert counter.call_count == 15  # 5 warmup + 10 timed

    def test_zero_warmup(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(iterations=10, warmup_count=0)
        run_benchmark(counter.sync_fn, config)

        assert counter.call_count == 10

    def test_label_propagated(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(
            iterations=5, warmup_count=1, label="my-bench"
        )
        result = run_benchmark(counter.sync_fn, config)

        assert result.label == "my-bench"

    def test_label_override(self) -> None:
        """Explicit label parameter overrides config label."""
        counter = CallCounter()
        config = BenchmarkConfig(
            iterations=5, warmup_count=0, label="config-label"
        )
        result = run_benchmark(counter.sync_fn, config, label="override-label")

        assert result.label == "override-label"

    def test_timings_are_non_negative(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(iterations=50, warmup_count=2)
        result = run_benchmark(counter.sync_fn, config)

        for t in result.raw_timings:
            assert t >= 0.0

    def test_timings_reflect_work_duration(self) -> None:
        """Each sample should be at least as long as the work time."""
        work_ms = 0.002  # 2ms
        counter = CallCounter(work_time=work_ms)
        config = BenchmarkConfig(iterations=5, warmup_count=1)
        result = run_benchmark(counter.sync_fn, config)

        for t in result.raw_timings:
            # Allow some tolerance for timer resolution
            assert t >= work_ms * 0.5

    def test_single_iteration(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(iterations=1, warmup_count=0)
        result = run_benchmark(counter.sync_fn, config)

        assert result.samples == 1
        assert counter.call_count == 1


class TestRunBenchmarkConfig:
    """Config handling and immutability."""

    def test_config_not_mutated(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(iterations=10, warmup_count=3, label="orig")
        run_benchmark(counter.sync_fn, config)

        assert config.iterations == 10
        assert config.warmup_count == 3
        assert config.label == "orig"

    def test_default_config(self) -> None:
        """Passing None for config uses BenchmarkConfig defaults."""
        counter = CallCounter()
        result = run_benchmark(counter.sync_fn)

        assert result.samples == 100  # BenchmarkConfig default iterations
        assert counter.call_count == 105  # 100 + 5 warmup

    def test_statistics_computed(self) -> None:
        """Verify that the result has valid computed statistics."""
        counter = CallCounter()
        config = BenchmarkConfig(iterations=20, warmup_count=2)
        result = run_benchmark(counter.sync_fn, config)

        assert result.mean >= 0.0
        assert result.stddev >= 0.0
        assert result.p50 >= 0.0
        assert result.p95 >= result.p50
        assert result.p99 >= result.p95
        assert result.min_time <= result.mean
        assert result.max_time >= result.mean


class TestRunBenchmarkCallable:
    """Tests for different callable types."""

    def test_accepts_plain_function(self) -> None:
        call_count = 0

        def plain_fn() -> None:
            nonlocal call_count
            call_count += 1

        config = BenchmarkConfig(iterations=5, warmup_count=0)
        result = run_benchmark(plain_fn, config)

        assert result.samples == 5
        assert call_count == 5

    def test_accepts_lambda(self) -> None:
        config = BenchmarkConfig(iterations=5, warmup_count=0)
        result = run_benchmark(lambda: None, config)

        assert result.samples == 5

    def test_accepts_callable_object(self) -> None:
        class MyCallable:
            def __init__(self) -> None:
                self.count = 0

            def __call__(self) -> None:
                self.count += 1

        obj = MyCallable()
        config = BenchmarkConfig(iterations=5, warmup_count=0)
        result = run_benchmark(obj, config)

        assert result.samples == 5
        assert obj.count == 5

    def test_accepts_method(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(iterations=5, warmup_count=0)
        result = run_benchmark(counter.sync_fn, config)

        assert result.samples == 5
        assert counter.call_count == 5


class TestRunBenchmarkErrors:
    """Error handling during benchmark execution."""

    def test_error_during_warmup_propagates(self) -> None:
        """If the callable fails during warmup, the error propagates."""
        target = FailAfterN(succeed_count=2)
        config = BenchmarkConfig(iterations=10, warmup_count=5)

        with pytest.raises(RuntimeError, match="Deliberate failure"):
            run_benchmark(target, config)

    def test_error_during_timed_iteration_propagates(self) -> None:
        """If the callable fails during a timed iteration, error propagates."""
        target = FailAfterN(succeed_count=7)
        config = BenchmarkConfig(iterations=10, warmup_count=5)

        with pytest.raises(RuntimeError, match="Deliberate failure"):
            run_benchmark(target, config)


# ---------------------------------------------------------------------------
# run_benchmark_async tests
# ---------------------------------------------------------------------------


class TestRunBenchmarkAsync:
    """Tests for the async benchmark runner variant."""

    @pytest.mark.asyncio
    async def test_returns_benchmark_result(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(iterations=10, warmup_count=2, label="async")
        result = await run_benchmark_async(counter.async_fn, config)

        assert isinstance(result, BenchmarkResult)
        assert result.samples == 10
        assert result.label == "async"

    @pytest.mark.asyncio
    async def test_total_calls_include_warmup(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(iterations=10, warmup_count=5)
        await run_benchmark_async(counter.async_fn, config)

        assert counter.call_count == 15

    @pytest.mark.asyncio
    async def test_zero_warmup(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(iterations=8, warmup_count=0)
        await run_benchmark_async(counter.async_fn, config)

        assert counter.call_count == 8

    @pytest.mark.asyncio
    async def test_label_override(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(
            iterations=5, warmup_count=0, label="cfg"
        )
        result = await run_benchmark_async(
            counter.async_fn, config, label="override"
        )

        assert result.label == "override"

    @pytest.mark.asyncio
    async def test_timings_non_negative(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(iterations=20, warmup_count=1)
        result = await run_benchmark_async(counter.async_fn, config)

        for t in result.raw_timings:
            assert t >= 0.0

    @pytest.mark.asyncio
    async def test_default_config(self) -> None:
        counter = CallCounter()
        result = await run_benchmark_async(counter.async_fn)

        assert result.samples == 100
        assert counter.call_count == 105

    @pytest.mark.asyncio
    async def test_error_during_warmup_propagates(self) -> None:
        call_count = 0

        async def failing_fn() -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise RuntimeError("Async failure")

        config = BenchmarkConfig(iterations=10, warmup_count=5)

        with pytest.raises(RuntimeError, match="Async failure"):
            await run_benchmark_async(failing_fn, config)

    @pytest.mark.asyncio
    async def test_statistics_computed(self) -> None:
        counter = CallCounter()
        config = BenchmarkConfig(iterations=20, warmup_count=2)
        result = await run_benchmark_async(counter.async_fn, config)

        assert result.mean >= 0.0
        assert result.stddev >= 0.0
        assert result.p50 >= 0.0
        assert result.min_time <= result.mean
        assert result.max_time >= result.mean


# ---------------------------------------------------------------------------
# BenchmarkTarget type alias tests
# ---------------------------------------------------------------------------


class TestBenchmarkTarget:
    """Verify the BenchmarkTarget type alias works for expected signatures."""

    def test_sync_callable_is_valid_target(self) -> None:
        def sync_fn() -> None:
            pass

        target: BenchmarkTarget = sync_fn
        target()  # should not raise

    def test_lambda_is_valid_target(self) -> None:
        target: BenchmarkTarget = lambda: None
        target()  # should not raise
