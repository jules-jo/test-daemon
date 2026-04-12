"""Benchmark tests for the direct-command bypass path using the harness.

Exercises the full bypass pipeline through the ``run_benchmark`` harness,
compares results against the baseline fixture inputs, and asserts no
regression beyond the defined thresholds from ``performance_thresholds``.

Unlike ``test_performance_baselines.py`` (which uses raw ``time.monotonic``
loops via ``measure_detection_latency``), these tests use the structured
``BenchmarkConfig`` / ``run_benchmark`` / ``BenchmarkResult`` / suite
reporting infrastructure.  This ensures the benchmark harness itself is
validated as a regression-detection tool for the bypass path.

Test structure:
    1. Per-category benchmarks -- each baseline input category is measured
       independently through the harness and its p95 is checked against
       the detection threshold.
    2. Warm-path benchmarks -- identical measurement after warm-up, checked
       against the tighter warm threshold.
    3. Classification pipeline benchmarks -- the full ``classify()``
       pipeline is benchmarked and compared against the classification
       threshold.
    4. Suite-level regression gate -- all individual results are aggregated
       into a ``BenchmarkSuiteReport`` and the suite-level ``all_passed``
       flag is asserted.
    5. Per-input parametrized regression -- each individual baseline input
       is benchmarked as a single target to catch outlier inputs that
       might regress while the category average stays healthy.
    6. Cross-comparison -- bypass path must be faster than the full
       classification pipeline (structural invariant).

Usage::

    pytest tests/test_bypass_benchmark.py -v
"""

from __future__ import annotations

import pytest

from jules_daemon.agent.benchmark_report import (
    create_suite_report,
    format_suite_summary,
)
from jules_daemon.agent.benchmark_runner import run_benchmark
from jules_daemon.agent.benchmark_types import (
    BenchmarkConfig,
    BenchmarkResult,
)
from jules_daemon.agent.performance_thresholds import (
    BASELINE_ITERATION_COUNT,
    CLASSIFY_INPUT_THRESHOLD_S,
    DIRECT_COMMAND_DETECTION_THRESHOLD_S,
    WARM_DETECTION_THRESHOLD_S,
)
from jules_daemon.classifier.classify import classify
from jules_daemon.classifier.direct_command import detect_direct_command
from tests.fixtures.baseline_latency import (
    ALL_BASELINE_INPUTS,
    DIRECT_COMMAND_BASELINE_INPUTS,
    EDGE_CASE_BASELINE_INPUTS,
    ENV_PREFIX_BASELINE_INPUTS,
    NL_INPUT_BASELINE_INPUTS,
    PATH_EXECUTABLE_BASELINE_INPUTS,
    SUDO_PREFIX_BASELINE_INPUTS,
)


# ---------------------------------------------------------------------------
# Shared benchmark configuration
# ---------------------------------------------------------------------------

# Use the same iteration count as the performance contract
_BENCH_CONFIG = BenchmarkConfig(
    iterations=BASELINE_ITERATION_COUNT,
    warmup_count=10,
    label="bypass-default",
)

# Tighter config for per-input parametrized tests (lower iteration
# count keeps total runtime manageable while still catching outliers)
_PER_INPUT_CONFIG = BenchmarkConfig(
    iterations=50,
    warmup_count=5,
    label="per-input",
)

# Regression threshold mapping for suite-level assertions
_SUITE_THRESHOLDS: dict[str, float] = {
    "bypass-direct-commands": DIRECT_COMMAND_DETECTION_THRESHOLD_S,
    "bypass-nl-inputs": DIRECT_COMMAND_DETECTION_THRESHOLD_S,
    "bypass-env-prefix": DIRECT_COMMAND_DETECTION_THRESHOLD_S,
    "bypass-sudo-prefix": DIRECT_COMMAND_DETECTION_THRESHOLD_S,
    "bypass-path-executables": DIRECT_COMMAND_DETECTION_THRESHOLD_S,
    "bypass-edge-cases": DIRECT_COMMAND_DETECTION_THRESHOLD_S,
    "bypass-all-inputs": DIRECT_COMMAND_DETECTION_THRESHOLD_S,
    "warm-direct-commands": WARM_DETECTION_THRESHOLD_S,
    "warm-all-inputs": WARM_DETECTION_THRESHOLD_S,
    "classify-direct-commands": CLASSIFY_INPUT_THRESHOLD_S,
    "classify-nl-inputs": CLASSIFY_INPUT_THRESHOLD_S,
    "classify-all-inputs": CLASSIFY_INPUT_THRESHOLD_S,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection_target(inputs: tuple[str, ...]) -> callable:
    """Build a benchmark target that calls detect_direct_command on all inputs.

    Each invocation of the returned callable processes every input in
    the tuple once. The benchmark runner calls this N times, so total
    calls = iterations * len(inputs).
    """
    def target() -> None:
        for raw in inputs:
            detect_direct_command(raw)

    return target


def _make_classify_target(inputs: tuple[str, ...]) -> callable:
    """Build a benchmark target that calls classify() on all inputs."""
    def target() -> None:
        for raw in inputs:
            classify(raw)

    return target


def _make_single_detection_target(raw: str) -> callable:
    """Build a benchmark target for a single input string."""
    def target() -> None:
        detect_direct_command(raw)

    return target


def _bench_detect(
    inputs: tuple[str, ...],
    label: str,
) -> BenchmarkResult:
    """Run the bypass detection benchmark for a set of inputs."""
    config = BenchmarkConfig(
        iterations=BASELINE_ITERATION_COUNT,
        warmup_count=10,
        label=label,
    )
    return run_benchmark(_make_detection_target(inputs), config)


def _bench_classify(
    inputs: tuple[str, ...],
    label: str,
) -> BenchmarkResult:
    """Run the classification pipeline benchmark for a set of inputs."""
    config = BenchmarkConfig(
        iterations=BASELINE_ITERATION_COUNT,
        warmup_count=10,
        label=label,
    )
    return run_benchmark(_make_classify_target(inputs), config)


# ---------------------------------------------------------------------------
# Per-category bypass detection benchmarks
# ---------------------------------------------------------------------------


class TestBypassDetectionBenchmarks:
    """Exercise detect_direct_command through the harness for each category.

    Each test measures a category of baseline inputs, produces a
    BenchmarkResult, and asserts p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S.
    """

    def test_direct_commands_via_harness(self) -> None:
        """Known executables (pytest, npm, cargo, etc.) stay under threshold."""
        result = _bench_detect(
            DIRECT_COMMAND_BASELINE_INPUTS,
            label="bypass-direct-commands",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Direct commands p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )

    def test_nl_inputs_via_harness(self) -> None:
        """NL inputs (non-detection path) measured through the harness."""
        result = _bench_detect(
            NL_INPUT_BASELINE_INPUTS,
            label="bypass-nl-inputs",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"NL inputs p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )

    def test_env_prefix_via_harness(self) -> None:
        """Env-prefix stripping path measured through the harness."""
        result = _bench_detect(
            ENV_PREFIX_BASELINE_INPUTS,
            label="bypass-env-prefix",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Env prefix p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )

    def test_sudo_prefix_via_harness(self) -> None:
        """Sudo prefix stripping path measured through the harness."""
        result = _bench_detect(
            SUDO_PREFIX_BASELINE_INPUTS,
            label="bypass-sudo-prefix",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Sudo prefix p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )

    def test_path_executables_via_harness(self) -> None:
        """Absolute/relative path detection measured through the harness."""
        result = _bench_detect(
            PATH_EXECUTABLE_BASELINE_INPUTS,
            label="bypass-path-executables",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Path executables p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )

    def test_edge_cases_via_harness(self) -> None:
        """Edge cases (empty, whitespace, chains) measured through the harness."""
        result = _bench_detect(
            EDGE_CASE_BASELINE_INPUTS,
            label="bypass-edge-cases",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Edge cases p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )

    def test_all_inputs_via_harness(self) -> None:
        """Full corpus measured through the harness."""
        result = _bench_detect(
            ALL_BASELINE_INPUTS,
            label="bypass-all-inputs",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"All inputs p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )


# ---------------------------------------------------------------------------
# Warm-path benchmarks (tighter threshold)
# ---------------------------------------------------------------------------


class TestWarmPathBenchmarks:
    """Warm-path benchmarks using the tighter WARM_DETECTION_THRESHOLD_S.

    After warm-up, the regex patterns and frozensets are populated.
    These tests verify the hot-path performance stays within the
    tighter 0.5ms threshold.
    """

    def test_warm_direct_commands(self) -> None:
        """Warm direct-command detection under the tighter warm threshold."""
        config = BenchmarkConfig(
            iterations=BASELINE_ITERATION_COUNT,
            warmup_count=20,  # extra warm-up for hot-path
            label="warm-direct-commands",
        )
        result = run_benchmark(
            _make_detection_target(DIRECT_COMMAND_BASELINE_INPUTS),
            config,
        )
        assert result.p95 < WARM_DETECTION_THRESHOLD_S, (
            f"Warm direct commands p95={result.p95 * 1000:.3f}ms "
            f"exceeds warm threshold={WARM_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )

    def test_warm_all_inputs(self) -> None:
        """Warm full-corpus detection under the tighter warm threshold."""
        config = BenchmarkConfig(
            iterations=BASELINE_ITERATION_COUNT,
            warmup_count=20,
            label="warm-all-inputs",
        )
        result = run_benchmark(
            _make_detection_target(ALL_BASELINE_INPUTS),
            config,
        )
        assert result.p95 < WARM_DETECTION_THRESHOLD_S, (
            f"Warm all inputs p95={result.p95 * 1000:.3f}ms "
            f"exceeds warm threshold={WARM_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )


# ---------------------------------------------------------------------------
# Classification pipeline benchmarks
# ---------------------------------------------------------------------------


class TestClassifyPipelineBenchmarks:
    """Exercise the full classify() pipeline through the harness.

    The classify() function includes verb resolution, structuredness
    scoring, and NL extraction. Its threshold is higher than detection
    alone (CLASSIFY_INPUT_THRESHOLD_S = 2ms vs 1ms).
    """

    def test_classify_direct_commands(self) -> None:
        """classify() on direct-command inputs stays under threshold."""
        result = _bench_classify(
            DIRECT_COMMAND_BASELINE_INPUTS,
            label="classify-direct-commands",
        )
        assert result.p95 < CLASSIFY_INPUT_THRESHOLD_S, (
            f"Classify direct commands p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={CLASSIFY_INPUT_THRESHOLD_S * 1000:.3f}ms"
        )

    def test_classify_nl_inputs(self) -> None:
        """classify() on NL inputs stays under threshold."""
        result = _bench_classify(
            NL_INPUT_BASELINE_INPUTS,
            label="classify-nl-inputs",
        )
        assert result.p95 < CLASSIFY_INPUT_THRESHOLD_S, (
            f"Classify NL inputs p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={CLASSIFY_INPUT_THRESHOLD_S * 1000:.3f}ms"
        )

    def test_classify_all_inputs(self) -> None:
        """classify() on the full corpus stays under threshold."""
        result = _bench_classify(
            ALL_BASELINE_INPUTS,
            label="classify-all-inputs",
        )
        assert result.p95 < CLASSIFY_INPUT_THRESHOLD_S, (
            f"Classify all inputs p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={CLASSIFY_INPUT_THRESHOLD_S * 1000:.3f}ms"
        )


# ---------------------------------------------------------------------------
# Suite-level regression gate
# ---------------------------------------------------------------------------


class TestSuiteRegressionGate:
    """Aggregate all bypass benchmarks into a suite and assert all_passed.

    This is the definitive regression gate: if any individual benchmark
    exceeds its threshold, the suite fails. The suite report can be
    logged for CI dashboards.
    """

    def test_full_bypass_suite_passes(self) -> None:
        """All bypass path benchmarks pass within their thresholds."""
        results: list[BenchmarkResult] = [
            # Detection benchmarks
            _bench_detect(DIRECT_COMMAND_BASELINE_INPUTS, "bypass-direct-commands"),
            _bench_detect(NL_INPUT_BASELINE_INPUTS, "bypass-nl-inputs"),
            _bench_detect(ENV_PREFIX_BASELINE_INPUTS, "bypass-env-prefix"),
            _bench_detect(SUDO_PREFIX_BASELINE_INPUTS, "bypass-sudo-prefix"),
            _bench_detect(PATH_EXECUTABLE_BASELINE_INPUTS, "bypass-path-executables"),
            _bench_detect(EDGE_CASE_BASELINE_INPUTS, "bypass-edge-cases"),
            _bench_detect(ALL_BASELINE_INPUTS, "bypass-all-inputs"),
            # Warm benchmarks
            run_benchmark(
                _make_detection_target(DIRECT_COMMAND_BASELINE_INPUTS),
                BenchmarkConfig(
                    iterations=BASELINE_ITERATION_COUNT,
                    warmup_count=20,
                    label="warm-direct-commands",
                ),
            ),
            run_benchmark(
                _make_detection_target(ALL_BASELINE_INPUTS),
                BenchmarkConfig(
                    iterations=BASELINE_ITERATION_COUNT,
                    warmup_count=20,
                    label="warm-all-inputs",
                ),
            ),
            # Classification benchmarks
            _bench_classify(DIRECT_COMMAND_BASELINE_INPUTS, "classify-direct-commands"),
            _bench_classify(NL_INPUT_BASELINE_INPUTS, "classify-nl-inputs"),
            _bench_classify(ALL_BASELINE_INPUTS, "classify-all-inputs"),
        ]

        suite = create_suite_report(results, thresholds=_SUITE_THRESHOLDS)

        # Log for CI visibility (visible with pytest -s)
        print(f"\n{format_suite_summary(suite)}")

        assert suite.all_passed, (
            f"Bypass benchmark suite has failures. "
            f"See suite summary for details:\n{format_suite_summary(suite)}"
        )

    def test_suite_report_metadata_correct(self) -> None:
        """Suite report has correct result count and total samples."""
        results: list[BenchmarkResult] = [
            _bench_detect(DIRECT_COMMAND_BASELINE_INPUTS, "bypass-direct-commands"),
            _bench_detect(NL_INPUT_BASELINE_INPUTS, "bypass-nl-inputs"),
        ]

        suite = create_suite_report(results, thresholds=_SUITE_THRESHOLDS)

        assert len(suite.results) == 2
        assert suite.total_samples == sum(r.samples for r in results)
        assert all(tr.passed for tr in suite.threshold_results)


# ---------------------------------------------------------------------------
# Per-input parametrized regression checks
# ---------------------------------------------------------------------------


class TestPerInputRegression:
    """Parametrized per-input benchmarks to catch outlier inputs.

    Each baseline input is benchmarked individually. This catches cases
    where a single input regresses (e.g., a pathological regex match)
    while the category average remains healthy.
    """

    @pytest.mark.parametrize("command", DIRECT_COMMAND_BASELINE_INPUTS)
    def test_direct_command_input_regression(self, command: str) -> None:
        """Individual direct command stays under detection threshold."""
        result = run_benchmark(
            _make_single_detection_target(command),
            _PER_INPUT_CONFIG,
            label=f"input-{command[:30]}",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Input {command!r} p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )

    @pytest.mark.parametrize("nl_input", NL_INPUT_BASELINE_INPUTS)
    def test_nl_input_regression(self, nl_input: str) -> None:
        """Individual NL input stays under detection threshold."""
        result = run_benchmark(
            _make_single_detection_target(nl_input),
            _PER_INPUT_CONFIG,
            label=f"input-{nl_input[:30]}",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Input {nl_input!r} p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )

    @pytest.mark.parametrize("command", ENV_PREFIX_BASELINE_INPUTS)
    def test_env_prefix_input_regression(self, command: str) -> None:
        """Individual env-prefix command stays under detection threshold."""
        result = run_benchmark(
            _make_single_detection_target(command),
            _PER_INPUT_CONFIG,
            label=f"input-{command[:30]}",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Input {command!r} p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )

    @pytest.mark.parametrize("command", SUDO_PREFIX_BASELINE_INPUTS)
    def test_sudo_prefix_input_regression(self, command: str) -> None:
        """Individual sudo-prefix command stays under detection threshold."""
        result = run_benchmark(
            _make_single_detection_target(command),
            _PER_INPUT_CONFIG,
            label=f"input-{command[:30]}",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Input {command!r} p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )

    @pytest.mark.parametrize("command", PATH_EXECUTABLE_BASELINE_INPUTS)
    def test_path_executable_input_regression(self, command: str) -> None:
        """Individual path-executable command stays under detection threshold."""
        result = run_benchmark(
            _make_single_detection_target(command),
            _PER_INPUT_CONFIG,
            label=f"input-{command[:30]}",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Input {command!r} p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )

    @pytest.mark.parametrize("edge_input", EDGE_CASE_BASELINE_INPUTS)
    def test_edge_case_input_regression(self, edge_input: str) -> None:
        """Individual edge-case input stays under detection threshold."""
        result = run_benchmark(
            _make_single_detection_target(edge_input),
            _PER_INPUT_CONFIG,
            label=f"input-{edge_input[:30]}",
        )
        assert result.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Input {edge_input!r} p95={result.p95 * 1000:.3f}ms "
            f"exceeds threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S * 1000:.3f}ms"
        )


# ---------------------------------------------------------------------------
# Cross-comparison: bypass must be faster than full classification
# ---------------------------------------------------------------------------


class TestBypassFasterThanClassify:
    """Structural invariant: bypass detection is faster than full classify.

    The detect_direct_command function is a subset of the full classify()
    pipeline. Its p95 latency must always be lower than the classification
    pipeline's p95 latency. If this invariant breaks, it means the bypass
    path has regressed relative to the pipeline it is supposed to shortcut.
    """

    def test_detection_faster_than_classification(self) -> None:
        """detect_direct_command p95 < classify p95 on the same inputs."""
        detection_result = _bench_detect(
            DIRECT_COMMAND_BASELINE_INPUTS,
            label="cross-detect",
        )
        classify_result = _bench_classify(
            DIRECT_COMMAND_BASELINE_INPUTS,
            label="cross-classify",
        )

        assert detection_result.p95 < classify_result.p95, (
            f"Detection p95={detection_result.p95 * 1000:.3f}ms "
            f"is not faster than classify p95={classify_result.p95 * 1000:.3f}ms. "
            f"Bypass path has regressed relative to the classification pipeline."
        )

    def test_detection_mean_lower_than_classification(self) -> None:
        """detect_direct_command mean < classify mean on the same inputs."""
        detection_result = _bench_detect(
            ALL_BASELINE_INPUTS,
            label="cross-detect-all",
        )
        classify_result = _bench_classify(
            ALL_BASELINE_INPUTS,
            label="cross-classify-all",
        )

        assert detection_result.mean < classify_result.mean, (
            f"Detection mean={detection_result.mean * 1000:.3f}ms "
            f"is not lower than classify mean={classify_result.mean * 1000:.3f}ms."
        )


# ---------------------------------------------------------------------------
# Benchmark result structural validation
# ---------------------------------------------------------------------------


class TestBenchmarkResultStructure:
    """Validate that harness-produced results have expected statistical properties.

    These are meta-tests: they verify the benchmark harness produces
    internally consistent BenchmarkResult instances for bypass-path
    targets, not just that mock callables work.
    """

    def test_sample_count_matches_iterations(self) -> None:
        """Result sample count equals the configured iteration count."""
        config = BenchmarkConfig(iterations=42, warmup_count=3, label="count-check")
        result = run_benchmark(
            _make_detection_target(DIRECT_COMMAND_BASELINE_INPUTS),
            config,
        )
        assert result.samples == 42

    def test_percentile_ordering_on_real_target(self) -> None:
        """p50 <= p95 <= p99 for real bypass detection target."""
        result = _bench_detect(ALL_BASELINE_INPUTS, label="ordering-check")
        assert result.p50 <= result.p95
        assert result.p95 <= result.p99
        assert result.min_time <= result.p50
        assert result.p99 <= result.max_time

    def test_all_timings_non_negative(self) -> None:
        """All raw timings from bypass detection are non-negative."""
        result = _bench_detect(DIRECT_COMMAND_BASELINE_INPUTS, label="non-neg-check")
        for timing in result.raw_timings:
            assert timing >= 0.0

    def test_mean_within_bounds(self) -> None:
        """Mean is between min and max for a real detection benchmark."""
        result = _bench_detect(ALL_BASELINE_INPUTS, label="mean-check")
        assert result.min_time <= result.mean <= result.max_time

    def test_label_preserved(self) -> None:
        """The label survives the full harness pipeline."""
        result = _bench_detect(
            DIRECT_COMMAND_BASELINE_INPUTS,
            label="label-preservation-test",
        )
        assert result.label == "label-preservation-test"

    def test_raw_timings_are_immutable(self) -> None:
        """Raw timings returned as an immutable tuple."""
        result = _bench_detect(DIRECT_COMMAND_BASELINE_INPUTS, label="immutable-check")
        assert isinstance(result.raw_timings, tuple)

    def test_result_is_frozen(self) -> None:
        """BenchmarkResult from the harness is frozen (immutable)."""
        result = _bench_detect(DIRECT_COMMAND_BASELINE_INPUTS, label="frozen-check")
        with pytest.raises(AttributeError):
            result.mean = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Correctness sanity checks (bypass path + harness integration)
# ---------------------------------------------------------------------------


class TestBypassCorrectnessViaHarness:
    """Verify detection correctness is preserved when running through the harness.

    These are not timing tests -- they verify that after N benchmark
    iterations the bypass path still produces correct detection results.
    This catches regressions where the harness might interfere with
    function behavior (e.g., through repeated invocation side effects).
    """

    def test_direct_commands_still_detected_after_benchmark(self) -> None:
        """After 100 iterations, detection results are still correct."""
        # Run the benchmark to "stress" the code path
        _bench_detect(DIRECT_COMMAND_BASELINE_INPUTS, label="correctness-pre")

        # Now verify correctness
        for command in DIRECT_COMMAND_BASELINE_INPUTS:
            detection = detect_direct_command(command)
            assert detection.is_direct_command is True, (
                f"After benchmark, {command!r} should still be detected"
            )

    def test_nl_inputs_still_rejected_after_benchmark(self) -> None:
        """After 100 iterations, NL inputs are still rejected."""
        _bench_detect(NL_INPUT_BASELINE_INPUTS, label="correctness-pre-nl")

        for nl_input in NL_INPUT_BASELINE_INPUTS:
            detection = detect_direct_command(nl_input)
            assert detection.is_direct_command is False, (
                f"After benchmark, {nl_input!r} should still be rejected"
            )

    def test_env_prefix_still_detected_after_benchmark(self) -> None:
        """Env-prefix stripping still works after benchmark iterations."""
        _bench_detect(ENV_PREFIX_BASELINE_INPUTS, label="correctness-pre-env")

        for command in ENV_PREFIX_BASELINE_INPUTS:
            detection = detect_direct_command(command)
            assert detection.is_direct_command is True, (
                f"After benchmark, {command!r} should still be detected"
            )

    def test_sudo_prefix_still_detected_after_benchmark(self) -> None:
        """Sudo prefix stripping still works after benchmark iterations."""
        _bench_detect(SUDO_PREFIX_BASELINE_INPUTS, label="correctness-pre-sudo")

        for command in SUDO_PREFIX_BASELINE_INPUTS:
            detection = detect_direct_command(command)
            assert detection.is_direct_command is True, (
                f"After benchmark, {command!r} should still be detected"
            )
