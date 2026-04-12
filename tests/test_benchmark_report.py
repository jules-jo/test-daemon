"""Tests for benchmark reporting module.

Validates formatted reporting output for benchmark results:
    - Human-readable summary (single result and suite)
    - Structured JSON export (single result and suite)
    - Threshold comparison logic (pass/fail/no-threshold)
    - Suite aggregation (multiple results, mixed pass/fail)
    - Immutability of all output data types
    - JSON roundtrip (string serialization and deserialization)
    - Full harness integration (config -> runner -> report)
    - Edge cases: single sample, no label, no threshold, all pass, all fail
"""

from __future__ import annotations

import json

import pytest

from jules_daemon.agent.benchmark_report import (
    BenchmarkSuiteReport,
    BenchmarkThresholdResult,
    create_suite_report,
    format_json,
    format_json_string,
    format_suite_json,
    format_suite_json_string,
    format_suite_summary,
    format_summary,
)
from jules_daemon.agent.benchmark_runner import run_benchmark
from jules_daemon.agent.benchmark_types import (
    BenchmarkConfig,
    BenchmarkResult,
    compute_result,
)


# ---------------------------------------------------------------------------
# Test fixtures: reusable result builders
# ---------------------------------------------------------------------------


def _make_result(
    *,
    label: str = "test-bench",
    timings: list[float] | None = None,
) -> BenchmarkResult:
    """Build a BenchmarkResult from optional timings."""
    effective_timings = timings or [0.001, 0.002, 0.003, 0.004, 0.005]
    return compute_result(effective_timings, label=label)


def _make_fast_result(label: str = "fast") -> BenchmarkResult:
    """Build a result with very fast timings (sub-millisecond)."""
    return compute_result(
        [0.0001, 0.0002, 0.0003, 0.0001, 0.0002],
        label=label,
    )


def _make_slow_result(label: str = "slow") -> BenchmarkResult:
    """Build a result with slow timings (over 100ms)."""
    return compute_result(
        [0.100, 0.120, 0.150, 0.200, 0.180],
        label=label,
    )


# ---------------------------------------------------------------------------
# BenchmarkThresholdResult tests
# ---------------------------------------------------------------------------


class TestBenchmarkThresholdResult:
    """Tests for the frozen threshold comparison dataclass."""

    def test_frozen_immutability(self) -> None:
        tr = BenchmarkThresholdResult(
            label="x",
            passed=True,
            p95=0.001,
            threshold=0.005,
            margin=0.004,
        )
        with pytest.raises(AttributeError):
            tr.passed = False  # type: ignore[misc]

    def test_fields_present(self) -> None:
        tr = BenchmarkThresholdResult(
            label="detect",
            passed=True,
            p95=0.0005,
            threshold=0.001,
            margin=0.0005,
        )
        assert tr.label == "detect"
        assert tr.passed is True
        assert tr.p95 == 0.0005
        assert tr.threshold == 0.001
        assert tr.margin == 0.0005

    def test_no_threshold(self) -> None:
        tr = BenchmarkThresholdResult(
            label="no-th",
            passed=True,
            p95=0.010,
            threshold=None,
            margin=None,
        )
        assert tr.threshold is None
        assert tr.margin is None
        assert tr.passed is True

    def test_failed_threshold(self) -> None:
        tr = BenchmarkThresholdResult(
            label="over",
            passed=False,
            p95=0.010,
            threshold=0.005,
            margin=-0.005,
        )
        assert tr.passed is False
        assert tr.margin is not None
        assert tr.margin < 0


# ---------------------------------------------------------------------------
# format_summary (single result)
# ---------------------------------------------------------------------------


class TestFormatSummary:
    """Tests for human-readable single result formatting."""

    def test_contains_label(self) -> None:
        result = _make_result(label="my-bench")
        text = format_summary(result)
        assert "my-bench" in text

    def test_unlabeled_result(self) -> None:
        result = _make_result(label="")
        text = format_summary(result)
        assert "(unlabeled)" in text

    def test_contains_samples_count(self) -> None:
        result = _make_result()
        text = format_summary(result)
        assert f"Samples: {result.samples}" in text

    def test_contains_mean(self) -> None:
        result = _make_result()
        text = format_summary(result)
        assert "Mean:" in text
        assert "ms" in text

    def test_contains_percentiles(self) -> None:
        result = _make_result()
        text = format_summary(result)
        assert "P50:" in text
        assert "P95:" in text
        assert "P99:" in text

    def test_contains_stddev(self) -> None:
        result = _make_result()
        text = format_summary(result)
        assert "Stddev:" in text

    def test_contains_min_max(self) -> None:
        result = _make_result()
        text = format_summary(result)
        assert "Min:" in text
        assert "Max:" in text

    def test_no_threshold_status(self) -> None:
        result = _make_result()
        text = format_summary(result, threshold=None)
        assert "no threshold" in text

    def test_passing_threshold_status(self) -> None:
        result = _make_result(timings=[0.001, 0.002, 0.001, 0.002, 0.001])
        text = format_summary(result, threshold=0.010)
        assert "PASS" in text
        assert "threshold" in text

    def test_failing_threshold_status(self) -> None:
        result = _make_result(timings=[0.100, 0.200, 0.150, 0.180, 0.190])
        text = format_summary(result, threshold=0.001)
        assert "FAIL" in text

    def test_margin_shown_for_pass(self) -> None:
        result = _make_result(timings=[0.001, 0.002, 0.001, 0.002, 0.001])
        text = format_summary(result, threshold=0.010)
        assert "margin:" in text
        assert "+" in text  # positive margin for pass

    def test_margin_shown_for_fail(self) -> None:
        result = _make_result(timings=[0.100, 0.200, 0.150, 0.180, 0.190])
        text = format_summary(result, threshold=0.001)
        assert "margin:" in text

    def test_millisecond_formatting(self) -> None:
        result = _make_result(timings=[0.0015])
        text = format_summary(result)
        # 0.0015s = 1.500ms
        assert "1.500ms" in text

    def test_returns_string(self) -> None:
        result = _make_result()
        text = format_summary(result)
        assert isinstance(text, str)

    def test_multiline_output(self) -> None:
        result = _make_result()
        text = format_summary(result)
        lines = text.strip().split("\n")
        assert len(lines) >= 8  # header + 7 stats + status


# ---------------------------------------------------------------------------
# format_json (single result)
# ---------------------------------------------------------------------------


class TestFormatJson:
    """Tests for structured JSON export of a single result."""

    def test_contains_format_version(self) -> None:
        result = _make_result()
        data = format_json(result)
        assert data["format_version"] == "1.0"

    def test_contains_timestamp(self) -> None:
        result = _make_result()
        data = format_json(result)
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)

    def test_custom_timestamp(self) -> None:
        result = _make_result()
        ts = "2026-01-15T10:30:00+00:00"
        data = format_json(result, timestamp=ts)
        assert data["timestamp"] == ts

    def test_contains_result_data(self) -> None:
        result = _make_result()
        data = format_json(result)
        assert "result" in data
        result_data = data["result"]
        assert "mean" in result_data
        assert "p50" in result_data
        assert "p95" in result_data
        assert "p99" in result_data
        assert "raw_timings" in result_data

    def test_result_matches_to_dict(self) -> None:
        result = _make_result()
        data = format_json(result)
        assert data["result"] == result.to_dict()

    def test_threshold_section_without_threshold(self) -> None:
        result = _make_result()
        data = format_json(result)
        assert data["threshold"]["threshold_seconds"] is None
        assert data["threshold"]["passed"] is True
        assert data["threshold"]["margin_seconds"] is None

    def test_threshold_section_with_passing_threshold(self) -> None:
        result = _make_result(timings=[0.001, 0.002])
        data = format_json(result, threshold=0.010)
        assert data["threshold"]["passed"] is True
        assert data["threshold"]["threshold_seconds"] == 0.010
        assert data["threshold"]["margin_seconds"] is not None
        assert data["threshold"]["margin_seconds"] > 0

    def test_threshold_section_with_failing_threshold(self) -> None:
        result = _make_result(timings=[0.100, 0.200])
        data = format_json(result, threshold=0.001)
        assert data["threshold"]["passed"] is False
        assert data["threshold"]["margin_seconds"] is not None
        assert data["threshold"]["margin_seconds"] < 0

    def test_config_included_when_provided(self) -> None:
        result = _make_result()
        config = BenchmarkConfig(iterations=50, warmup_count=3, label="cfg")
        data = format_json(result, config=config)
        assert "config" in data
        assert data["config"] == config.to_dict()

    def test_config_absent_when_not_provided(self) -> None:
        result = _make_result()
        data = format_json(result)
        assert "config" not in data

    def test_json_serializable(self) -> None:
        """The output dict must be JSON-serializable without errors."""
        result = _make_result()
        config = BenchmarkConfig(iterations=10, warmup_count=2)
        data = format_json(result, threshold=0.005, config=config)
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_json_roundtrip(self) -> None:
        """Serialize to JSON string and deserialize back."""
        result = _make_result()
        data = format_json(result, threshold=0.005)
        json_str = json.dumps(data)
        restored = json.loads(json_str)
        assert restored["format_version"] == "1.0"
        assert restored["result"]["mean"] == data["result"]["mean"]
        assert restored["threshold"]["passed"] == data["threshold"]["passed"]


# ---------------------------------------------------------------------------
# format_json_string
# ---------------------------------------------------------------------------


class TestFormatJsonString:
    """Tests for the JSON string convenience wrapper."""

    def test_returns_string(self) -> None:
        result = _make_result()
        output = format_json_string(result)
        assert isinstance(output, str)

    def test_valid_json(self) -> None:
        result = _make_result()
        output = format_json_string(result, threshold=0.005)
        parsed = json.loads(output)
        assert "format_version" in parsed
        assert "result" in parsed

    def test_custom_indent(self) -> None:
        result = _make_result()
        compact = format_json_string(result, indent=0)
        pretty = format_json_string(result, indent=4)
        # Pretty-printed version should be longer due to whitespace
        assert len(pretty) > len(compact)

    def test_indent_two_default(self) -> None:
        result = _make_result()
        output = format_json_string(result)
        # Default indent=2 produces indented output
        assert "\n  " in output

    def test_timestamp_passthrough(self) -> None:
        result = _make_result()
        ts = "2026-03-20T12:00:00+00:00"
        output = format_json_string(result, timestamp=ts)
        parsed = json.loads(output)
        assert parsed["timestamp"] == ts

    def test_config_passthrough(self) -> None:
        result = _make_result()
        config = BenchmarkConfig(iterations=25, warmup_count=5, label="str")
        output = format_json_string(result, config=config)
        parsed = json.loads(output)
        assert parsed["config"]["iterations"] == 25


# ---------------------------------------------------------------------------
# BenchmarkSuiteReport tests
# ---------------------------------------------------------------------------


class TestBenchmarkSuiteReport:
    """Tests for the frozen suite report dataclass."""

    def test_frozen_immutability(self) -> None:
        suite = create_suite_report(
            [_make_result()],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        with pytest.raises(AttributeError):
            suite.all_passed = False  # type: ignore[misc]

    def test_fields_present(self) -> None:
        r1 = _make_result(label="a")
        r2 = _make_result(label="b")
        suite = create_suite_report(
            [r1, r2],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert len(suite.results) == 2
        assert suite.total_samples == r1.samples + r2.samples
        assert suite.timestamp == "2026-01-01T00:00:00+00:00"

    def test_all_passed_no_thresholds(self) -> None:
        suite = create_suite_report(
            [_make_result()],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert suite.all_passed is True

    def test_all_passed_with_passing_thresholds(self) -> None:
        r = _make_result(label="fast", timings=[0.001, 0.002])
        suite = create_suite_report(
            [r],
            thresholds={"fast": 1.0},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert suite.all_passed is True

    def test_failure_detected(self) -> None:
        r = _make_result(label="slow", timings=[0.100, 0.200])
        suite = create_suite_report(
            [r],
            thresholds={"slow": 0.001},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert suite.all_passed is False

    def test_mixed_pass_fail(self) -> None:
        fast = _make_result(label="fast", timings=[0.001])
        slow = _make_result(label="slow", timings=[0.500])
        suite = create_suite_report(
            [fast, slow],
            thresholds={"fast": 1.0, "slow": 0.001},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert suite.all_passed is False
        # Fast should pass, slow should fail
        assert suite.threshold_results[0].passed is True
        assert suite.threshold_results[1].passed is False

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="results must not be empty"):
            create_suite_report([])

    def test_threshold_for_unlabeled_result(self) -> None:
        """Unlabeled results get no threshold match."""
        r = _make_result(label="")
        suite = create_suite_report(
            [r],
            thresholds={"something": 0.001},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert suite.threshold_results[0].threshold is None
        assert suite.all_passed is True

    def test_auto_timestamp(self) -> None:
        """When no timestamp provided, one is generated."""
        suite = create_suite_report([_make_result()])
        assert suite.timestamp != ""
        assert "T" in suite.timestamp  # ISO-8601 format


# ---------------------------------------------------------------------------
# format_suite_summary
# ---------------------------------------------------------------------------


class TestFormatSuiteSummary:
    """Tests for human-readable suite report formatting."""

    def test_contains_header(self) -> None:
        suite = create_suite_report(
            [_make_result()],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        text = format_suite_summary(suite)
        assert "Benchmark Suite Report" in text

    def test_contains_timestamp(self) -> None:
        ts = "2026-04-10T08:00:00+00:00"
        suite = create_suite_report(
            [_make_result()],
            timestamp=ts,
        )
        text = format_suite_summary(suite)
        assert ts in text

    def test_contains_total_results(self) -> None:
        suite = create_suite_report(
            [_make_result(label="a"), _make_result(label="b")],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        text = format_suite_summary(suite)
        assert "Total Results: 2" in text

    def test_contains_total_samples(self) -> None:
        r1 = _make_result(timings=[0.001, 0.002, 0.003])
        r2 = _make_result(timings=[0.004, 0.005])
        suite = create_suite_report(
            [r1, r2],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        text = format_suite_summary(suite)
        assert f"Total Samples: {r1.samples + r2.samples}" in text

    def test_all_passed_status(self) -> None:
        suite = create_suite_report(
            [_make_result()],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        text = format_suite_summary(suite)
        assert "ALL PASSED" in text

    def test_failure_status(self) -> None:
        r = _make_result(label="slow", timings=[0.500])
        suite = create_suite_report(
            [r],
            thresholds={"slow": 0.001},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        text = format_suite_summary(suite)
        assert "FAILURES DETECTED" in text

    def test_contains_individual_results(self) -> None:
        r1 = _make_result(label="alpha")
        r2 = _make_result(label="beta")
        suite = create_suite_report(
            [r1, r2],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        text = format_suite_summary(suite)
        assert "alpha" in text
        assert "beta" in text

    def test_returns_string(self) -> None:
        suite = create_suite_report(
            [_make_result()],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        text = format_suite_summary(suite)
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# format_suite_json
# ---------------------------------------------------------------------------


class TestFormatSuiteJson:
    """Tests for structured JSON export of suite reports."""

    def test_contains_format_version(self) -> None:
        suite = create_suite_report(
            [_make_result()],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        data = format_suite_json(suite)
        assert data["format_version"] == "1.0"

    def test_contains_timestamp(self) -> None:
        ts = "2026-04-10T08:00:00+00:00"
        suite = create_suite_report(
            [_make_result()],
            timestamp=ts,
        )
        data = format_suite_json(suite)
        assert data["timestamp"] == ts

    def test_suite_status_passed(self) -> None:
        suite = create_suite_report(
            [_make_result()],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        data = format_suite_json(suite)
        assert data["suite_status"] == "passed"

    def test_suite_status_failed(self) -> None:
        r = _make_result(label="slow", timings=[0.500])
        suite = create_suite_report(
            [r],
            thresholds={"slow": 0.001},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        data = format_suite_json(suite)
        assert data["suite_status"] == "failed"

    def test_total_results_count(self) -> None:
        suite = create_suite_report(
            [_make_result(label="a"), _make_result(label="b")],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        data = format_suite_json(suite)
        assert data["total_results"] == 2

    def test_total_samples(self) -> None:
        r1 = _make_result(timings=[0.001, 0.002, 0.003])
        r2 = _make_result(timings=[0.004, 0.005])
        suite = create_suite_report(
            [r1, r2],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        data = format_suite_json(suite)
        assert data["total_samples"] == r1.samples + r2.samples

    def test_results_array_structure(self) -> None:
        suite = create_suite_report(
            [_make_result(label="x")],
            thresholds={"x": 1.0},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        data = format_suite_json(suite)
        results = data["results"]
        assert isinstance(results, list)
        assert len(results) == 1
        entry = results[0]
        assert "result" in entry
        assert "threshold" in entry

    def test_result_data_matches_to_dict(self) -> None:
        r = _make_result(label="y")
        suite = create_suite_report(
            [r],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        data = format_suite_json(suite)
        assert data["results"][0]["result"] == r.to_dict()

    def test_threshold_data_structure(self) -> None:
        r = _make_result(label="z", timings=[0.001])
        suite = create_suite_report(
            [r],
            thresholds={"z": 0.005},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        data = format_suite_json(suite)
        th = data["results"][0]["threshold"]
        assert "label" in th
        assert "passed" in th
        assert "p95_seconds" in th
        assert "threshold_seconds" in th
        assert "margin_seconds" in th

    def test_json_serializable(self) -> None:
        suite = create_suite_report(
            [_make_result()],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        data = format_suite_json(suite)
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_json_roundtrip(self) -> None:
        suite = create_suite_report(
            [_make_result(label="rt")],
            thresholds={"rt": 0.005},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        data = format_suite_json(suite)
        json_str = json.dumps(data)
        restored = json.loads(json_str)
        assert restored["format_version"] == data["format_version"]
        assert restored["suite_status"] == data["suite_status"]
        assert len(restored["results"]) == len(data["results"])


# ---------------------------------------------------------------------------
# format_suite_json_string
# ---------------------------------------------------------------------------


class TestFormatSuiteJsonString:
    """Tests for suite JSON string serialization."""

    def test_returns_string(self) -> None:
        suite = create_suite_report(
            [_make_result()],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        output = format_suite_json_string(suite)
        assert isinstance(output, str)

    def test_valid_json(self) -> None:
        suite = create_suite_report(
            [_make_result()],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        output = format_suite_json_string(suite)
        parsed = json.loads(output)
        assert "format_version" in parsed

    def test_custom_indent(self) -> None:
        suite = create_suite_report(
            [_make_result()],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        compact = format_suite_json_string(suite, indent=0)
        pretty = format_suite_json_string(suite, indent=4)
        assert len(pretty) > len(compact)


# ---------------------------------------------------------------------------
# Full harness integration tests
# ---------------------------------------------------------------------------


class TestFullHarnessIntegration:
    """End-to-end tests: config -> runner -> compute -> report.

    Validates that the benchmark pipeline works as a cohesive unit,
    from configuration through execution to formatted output.
    """

    def test_config_to_runner_to_summary(self) -> None:
        """Full pipeline: BenchmarkConfig -> run_benchmark -> format_summary."""
        config = BenchmarkConfig(
            iterations=10,
            warmup_count=2,
            label="integration-test",
        )

        call_count = 0

        def target() -> None:
            nonlocal call_count
            call_count += 1

        result = run_benchmark(target, config)
        text = format_summary(result, threshold=0.050)

        assert "integration-test" in text
        assert "PASS" in text
        assert call_count == 12  # 2 warmup + 10 timed

    def test_config_to_runner_to_json(self) -> None:
        """Full pipeline: BenchmarkConfig -> run_benchmark -> format_json."""
        config = BenchmarkConfig(
            iterations=10,
            warmup_count=2,
            label="json-integration",
        )

        result = run_benchmark(lambda: None, config)
        data = format_json(
            result,
            threshold=0.050,
            config=config,
            timestamp="2026-01-01T00:00:00+00:00",
        )

        assert data["format_version"] == "1.0"
        assert data["result"]["samples"] == 10
        assert data["result"]["label"] == "json-integration"
        assert data["config"]["iterations"] == 10
        assert data["threshold"]["passed"] is True

    def test_multi_result_suite_pipeline(self) -> None:
        """Full pipeline: multiple benchmarks -> suite report."""
        configs = [
            BenchmarkConfig(
                iterations=5,
                warmup_count=1,
                label="fast-op",
            ),
            BenchmarkConfig(
                iterations=5,
                warmup_count=1,
                label="medium-op",
            ),
        ]

        results: list[BenchmarkResult] = []
        for cfg in configs:
            result = run_benchmark(lambda: None, cfg)
            results.append(result)

        thresholds = {
            "fast-op": 0.050,
            "medium-op": 0.100,
        }

        suite = create_suite_report(
            results,
            thresholds=thresholds,
            timestamp="2026-01-01T00:00:00+00:00",
        )

        # Verify suite
        assert suite.all_passed is True
        assert len(suite.results) == 2
        assert suite.total_samples == 10  # 5 + 5

        # Verify summary
        text = format_suite_summary(suite)
        assert "ALL PASSED" in text
        assert "fast-op" in text
        assert "medium-op" in text

        # Verify JSON
        data = format_suite_json(suite)
        assert data["suite_status"] == "passed"
        assert data["total_results"] == 2

    def test_json_string_roundtrip(self) -> None:
        """Full pipeline: run -> format_json_string -> parse back."""
        config = BenchmarkConfig(iterations=5, warmup_count=1, label="rt")
        result = run_benchmark(lambda: None, config)

        json_str = format_json_string(
            result,
            threshold=0.050,
            config=config,
            timestamp="2026-01-01T00:00:00+00:00",
        )

        parsed = json.loads(json_str)
        assert parsed["format_version"] == "1.0"
        assert parsed["result"]["samples"] == 5
        assert parsed["result"]["label"] == "rt"
        assert parsed["threshold"]["passed"] is True

    def test_suite_json_string_roundtrip(self) -> None:
        """Full pipeline: run multiple -> suite -> JSON string -> parse back."""
        results = [
            run_benchmark(
                lambda: None,
                BenchmarkConfig(iterations=5, warmup_count=0, label="a"),
            ),
            run_benchmark(
                lambda: None,
                BenchmarkConfig(iterations=5, warmup_count=0, label="b"),
            ),
        ]

        suite = create_suite_report(
            results,
            thresholds={"a": 1.0, "b": 1.0},
            timestamp="2026-01-01T00:00:00+00:00",
        )

        json_str = format_suite_json_string(suite)
        parsed = json.loads(json_str)

        assert parsed["suite_status"] == "passed"
        assert len(parsed["results"]) == 2

    def test_compute_result_to_summary(self) -> None:
        """Direct compute_result -> format_summary path."""
        timings = [0.001, 0.002, 0.003, 0.004, 0.005]
        result = compute_result(timings, label="direct-compute")
        text = format_summary(result, threshold=0.010)

        assert "direct-compute" in text
        assert "PASS" in text
        assert "Samples: 5" in text

    def test_threshold_at_exact_boundary(self) -> None:
        """When p95 exactly equals threshold, it should pass."""
        timings = [0.005] * 20  # All identical -> p95 = 0.005
        result = compute_result(timings, label="boundary")
        text = format_summary(result, threshold=0.005)

        assert "PASS" in text

    def test_failing_suite_json_export(self) -> None:
        """Full pipeline with an intentionally failing benchmark."""
        # Create a result that will fail its threshold
        slow_timings = [0.500, 0.600, 0.700, 0.800, 0.900]
        slow_result = compute_result(slow_timings, label="intentionally-slow")

        fast_result = run_benchmark(
            lambda: None,
            BenchmarkConfig(iterations=5, warmup_count=0, label="fast"),
        )

        suite = create_suite_report(
            [fast_result, slow_result],
            thresholds={"fast": 1.0, "intentionally-slow": 0.001},
            timestamp="2026-01-01T00:00:00+00:00",
        )

        assert suite.all_passed is False

        data = format_suite_json(suite)
        assert data["suite_status"] == "failed"

        # Fast should pass
        assert data["results"][0]["threshold"]["passed"] is True
        # Slow should fail
        assert data["results"][1]["threshold"]["passed"] is False
        assert data["results"][1]["threshold"]["margin_seconds"] < 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_timing_sample(self) -> None:
        result = compute_result([0.042], label="single")
        text = format_summary(result, threshold=0.050)
        assert "PASS" in text
        assert "Samples: 1" in text

    def test_very_small_timings(self) -> None:
        """Sub-microsecond timings should format without error."""
        result = compute_result([0.000001, 0.000002], label="micro")
        text = format_summary(result)
        assert "micro" in text
        assert "ms" in text

    def test_very_large_timings(self) -> None:
        """Multi-second timings should format without error."""
        result = compute_result([10.0, 20.0, 30.0], label="large")
        text = format_summary(result)
        assert "large" in text

    def test_zero_timings(self) -> None:
        """Zero-valued timings should produce valid output."""
        result = compute_result([0.0, 0.0, 0.0], label="zero")
        text = format_summary(result, threshold=0.001)
        assert "PASS" in text
        assert "0.000ms" in text

    def test_suite_with_many_results(self) -> None:
        """Suite with 10 results should format correctly."""
        results = [
            compute_result(
                [float(i + 1) / 10000.0],
                label=f"bench-{i}",
            )
            for i in range(10)
        ]
        suite = create_suite_report(
            results,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        text = format_suite_summary(suite)
        assert "Total Results: 10" in text
        for i in range(10):
            assert f"bench-{i}" in text

    def test_suite_with_one_result(self) -> None:
        """Suite with a single result should work correctly."""
        result = compute_result([0.005], label="solo")
        suite = create_suite_report(
            [result],
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert len(suite.results) == 1
        assert suite.total_samples == 1

    def test_threshold_none_in_dict(self) -> None:
        """Results not in threshold dict should be treated as no threshold."""
        r = _make_result(label="missing-from-thresholds")
        suite = create_suite_report(
            [r],
            thresholds={"other-label": 0.001},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        # Should not fail -- label not in thresholds means no threshold
        assert suite.all_passed is True
        assert suite.threshold_results[0].threshold is None
