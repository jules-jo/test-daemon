"""Formatted reporting for benchmark results.

Provides two output formats for benchmark measurement data:

    1. **Human-readable summary** -- a compact text table showing key
       statistics (mean, p50, p95, p99, stddev, min, max) with optional
       threshold pass/fail indicators.  Suitable for terminal output,
       CI logs, and developer review.

    2. **Structured JSON export** -- a machine-readable dict (and JSON
       string) containing the full benchmark result with metadata
       (timestamp, config snapshot, threshold comparisons).  Suitable
       for persistence, dashboards, and regression tracking.

Both formats support single results and collections (suites) of
results.  The ``BenchmarkSuiteReport`` frozen dataclass aggregates
multiple results with a suite-level pass/fail summary.

Key design decisions:

    - All timing values displayed in milliseconds (user-friendly) while
      stored in seconds (internal convention).
    - Threshold comparisons are optional: when a threshold dict is
      provided, each result gets a pass/fail status.  Without thresholds,
      results are reported without judgment.
    - JSON export includes an ISO-8601 timestamp for when the report
      was generated, enabling time-series regression analysis.
    - Immutable frozen dataclasses throughout -- no mutation.

Usage::

    from jules_daemon.agent.benchmark_report import (
        format_summary,
        format_json,
        BenchmarkSuiteReport,
        create_suite_report,
    )
    from jules_daemon.agent.benchmark_types import BenchmarkResult

    result = compute_result(timings, label="detect")
    thresholds = {"detect": 0.001}  # 1ms threshold

    # Human-readable summary
    text = format_summary(result, threshold=0.001)
    print(text)

    # JSON export
    data = format_json(result, threshold=0.001)
    json_str = json.dumps(data, indent=2)

    # Suite report
    suite = create_suite_report([result1, result2], thresholds=thresholds)
    print(format_suite_summary(suite))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from jules_daemon.agent.benchmark_types import (
    BenchmarkConfig,
    BenchmarkResult,
)

__all__ = [
    "BenchmarkThresholdResult",
    "BenchmarkSuiteReport",
    "format_summary",
    "format_json",
    "format_json_string",
    "create_suite_report",
    "format_suite_summary",
    "format_suite_json",
    "format_suite_json_string",
]


# ---------------------------------------------------------------------------
# Threshold comparison result
# ---------------------------------------------------------------------------

_MS_PER_SECOND: float = 1000.0


@dataclass(frozen=True)
class BenchmarkThresholdResult:
    """Result of comparing a benchmark measurement against a threshold.

    Attributes:
        label: Human-readable label identifying the benchmark.
        passed: True if p95 is at or below the threshold.
        p95: 95th percentile timing in seconds.
        threshold: Threshold value in seconds, or None if no threshold.
        margin: Distance from p95 to threshold in seconds.
            Positive means under threshold (headroom).
            Negative means over threshold (violation).
            None if no threshold provided.
    """

    label: str
    passed: bool
    p95: float
    threshold: float | None
    margin: float | None


# ---------------------------------------------------------------------------
# Suite report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkSuiteReport:
    """Immutable aggregation of multiple benchmark results.

    Attributes:
        results: Tuple of individual benchmark results.
        threshold_results: Tuple of threshold comparison outcomes,
            one per result.  Empty if no thresholds provided.
        all_passed: True if every result with a threshold passed.
            True when no thresholds are provided (vacuously true).
        total_samples: Sum of all individual sample counts.
        timestamp: ISO-8601 UTC timestamp when the report was created.
    """

    results: tuple[BenchmarkResult, ...]
    threshold_results: tuple[BenchmarkThresholdResult, ...]
    all_passed: bool
    total_samples: int
    timestamp: str


# ---------------------------------------------------------------------------
# Threshold comparison
# ---------------------------------------------------------------------------


def _compare_threshold(
    result: BenchmarkResult,
    threshold: float | None,
) -> BenchmarkThresholdResult:
    """Compare a benchmark result's p95 against an optional threshold.

    Args:
        result: Benchmark measurement result.
        threshold: Maximum acceptable p95 in seconds, or None.

    Returns:
        Immutable threshold comparison result.
    """
    if threshold is None:
        return BenchmarkThresholdResult(
            label=result.label,
            passed=True,
            p95=result.p95,
            threshold=None,
            margin=None,
        )
    margin = threshold - result.p95
    passed = result.p95 <= threshold
    return BenchmarkThresholdResult(
        label=result.label,
        passed=passed,
        p95=result.p95,
        threshold=threshold,
        margin=margin,
    )


# ---------------------------------------------------------------------------
# Human-readable formatting: single result
# ---------------------------------------------------------------------------


def _format_ms(seconds: float) -> str:
    """Format a timing value in seconds as a millisecond string.

    Uses 3 decimal places for sub-millisecond precision.
    """
    return f"{seconds * _MS_PER_SECOND:.3f}ms"


def _format_status(threshold_result: BenchmarkThresholdResult) -> str:
    """Format the pass/fail status line for a threshold comparison."""
    if threshold_result.threshold is None:
        return "  Status: no threshold"
    status = "PASS" if threshold_result.passed else "FAIL"
    margin = threshold_result.margin
    margin_str = ""
    if margin is not None:
        sign = "+" if margin >= 0 else ""
        margin_str = f" (margin: {sign}{margin * _MS_PER_SECOND:.3f}ms)"
    return (
        f"  Status: {status} "
        f"(p95 {_format_ms(threshold_result.p95)} "
        f"vs threshold {_format_ms(threshold_result.threshold)})"
        f"{margin_str}"
    )


def format_summary(
    result: BenchmarkResult,
    *,
    threshold: float | None = None,
) -> str:
    """Format a single benchmark result as a human-readable text summary.

    Produces a compact multi-line report showing:
        - Label and sample count
        - Key percentiles (p50, p95, p99)
        - Mean and standard deviation
        - Min and max bounds
        - Threshold pass/fail status (when threshold provided)

    All timing values are displayed in milliseconds.

    Args:
        result: Benchmark measurement result to format.
        threshold: Optional maximum acceptable p95 in seconds.
            When provided, a pass/fail line is appended.

    Returns:
        Multi-line human-readable string.
    """
    label_display = result.label if result.label else "(unlabeled)"
    threshold_result = _compare_threshold(result, threshold)

    lines = [
        f"Benchmark: {label_display}",
        f"  Samples: {result.samples}",
        f"  Mean:    {_format_ms(result.mean)}",
        f"  Stddev:  {_format_ms(result.stddev)}",
        f"  P50:     {_format_ms(result.p50)}",
        f"  P95:     {_format_ms(result.p95)}",
        f"  P99:     {_format_ms(result.p99)}",
        f"  Min:     {_format_ms(result.min_time)}",
        f"  Max:     {_format_ms(result.max_time)}",
        _format_status(threshold_result),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON export: single result
# ---------------------------------------------------------------------------


def format_json(
    result: BenchmarkResult,
    *,
    threshold: float | None = None,
    config: BenchmarkConfig | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Export a single benchmark result as a structured JSON-compatible dict.

    The output dict contains:
        - ``result``: Full benchmark result data (via ``to_dict()``).
        - ``threshold``: Threshold comparison details (if threshold given).
        - ``config``: Benchmark config snapshot (if config given).
        - ``timestamp``: ISO-8601 UTC timestamp.
        - ``format_version``: Schema version for forward compatibility.

    Args:
        result: Benchmark measurement result to export.
        threshold: Optional maximum acceptable p95 in seconds.
        config: Optional benchmark config used for the run.
        timestamp: Optional ISO-8601 timestamp override. If None,
            uses the current UTC time.

    Returns:
        Dict ready for ``json.dumps()`` serialization.
    """
    effective_ts = timestamp or _utc_now_iso()
    threshold_result = _compare_threshold(result, threshold)

    output: dict[str, Any] = {
        "format_version": "1.0",
        "timestamp": effective_ts,
        "result": result.to_dict(),
    }

    if config is not None:
        output["config"] = config.to_dict()

    output["threshold"] = {
        "label": threshold_result.label,
        "passed": threshold_result.passed,
        "p95_seconds": threshold_result.p95,
        "threshold_seconds": threshold_result.threshold,
        "margin_seconds": threshold_result.margin,
    }

    return output


def format_json_string(
    result: BenchmarkResult,
    *,
    threshold: float | None = None,
    config: BenchmarkConfig | None = None,
    timestamp: str | None = None,
    indent: int = 2,
) -> str:
    """Export a single benchmark result as a formatted JSON string.

    Convenience wrapper around ``format_json()`` that serializes the
    result dict to a JSON string with configurable indentation.

    Args:
        result: Benchmark measurement result to export.
        threshold: Optional maximum acceptable p95 in seconds.
        config: Optional benchmark config used for the run.
        timestamp: Optional ISO-8601 timestamp override.
        indent: JSON indentation level (default 2).

    Returns:
        Formatted JSON string.
    """
    data = format_json(
        result,
        threshold=threshold,
        config=config,
        timestamp=timestamp,
    )
    return json.dumps(data, indent=indent)


# ---------------------------------------------------------------------------
# Suite report construction
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def create_suite_report(
    results: list[BenchmarkResult],
    *,
    thresholds: dict[str, float] | None = None,
    timestamp: str | None = None,
) -> BenchmarkSuiteReport:
    """Create an aggregated suite report from multiple benchmark results.

    Each result is compared against its matching threshold (by label).
    Results whose labels are not in the thresholds dict receive a
    "no threshold" status and are treated as passing.

    Args:
        results: List of benchmark measurement results.
        thresholds: Optional dict mapping result labels to maximum
            acceptable p95 values in seconds.
        timestamp: Optional ISO-8601 timestamp override.

    Returns:
        Immutable suite report with per-result threshold comparisons.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("results must not be empty")

    effective_thresholds = thresholds or {}
    effective_ts = timestamp or _utc_now_iso()

    threshold_results: list[BenchmarkThresholdResult] = []
    for r in results:
        th = effective_thresholds.get(r.label) if r.label else None
        threshold_results.append(_compare_threshold(r, th))

    all_passed = all(tr.passed for tr in threshold_results)
    total_samples = sum(r.samples for r in results)

    return BenchmarkSuiteReport(
        results=tuple(results),
        threshold_results=tuple(threshold_results),
        all_passed=all_passed,
        total_samples=total_samples,
        timestamp=effective_ts,
    )


# ---------------------------------------------------------------------------
# Human-readable formatting: suite
# ---------------------------------------------------------------------------


def format_suite_summary(suite: BenchmarkSuiteReport) -> str:
    """Format a suite report as a human-readable multi-line summary.

    Includes a header with suite-level status, followed by individual
    result summaries separated by blank lines.

    Args:
        suite: Suite report to format.

    Returns:
        Multi-line human-readable string.
    """
    status = "ALL PASSED" if suite.all_passed else "FAILURES DETECTED"
    header_lines = [
        "Benchmark Suite Report",
        f"  Timestamp:     {suite.timestamp}",
        f"  Total Results: {len(suite.results)}",
        f"  Total Samples: {suite.total_samples}",
        f"  Suite Status:  {status}",
        "",
    ]

    result_sections: list[str] = []
    for result, threshold_result in zip(
        suite.results, suite.threshold_results, strict=True
    ):
        result_sections.append(
            format_summary(result, threshold=threshold_result.threshold)
        )

    separator = "\n\n"
    return "\n".join(header_lines) + separator.join(result_sections)


# ---------------------------------------------------------------------------
# JSON export: suite
# ---------------------------------------------------------------------------


def format_suite_json(
    suite: BenchmarkSuiteReport,
) -> dict[str, Any]:
    """Export a suite report as a structured JSON-compatible dict.

    Args:
        suite: Suite report to export.

    Returns:
        Dict ready for ``json.dumps()`` serialization.
    """
    results_data: list[dict[str, Any]] = []
    for result, threshold_result in zip(
        suite.results, suite.threshold_results, strict=True
    ):
        entry: dict[str, Any] = {
            "result": result.to_dict(),
            "threshold": {
                "label": threshold_result.label,
                "passed": threshold_result.passed,
                "p95_seconds": threshold_result.p95,
                "threshold_seconds": threshold_result.threshold,
                "margin_seconds": threshold_result.margin,
            },
        }
        results_data.append(entry)

    return {
        "format_version": "1.0",
        "timestamp": suite.timestamp,
        "suite_status": "passed" if suite.all_passed else "failed",
        "total_results": len(suite.results),
        "total_samples": suite.total_samples,
        "results": results_data,
    }


def format_suite_json_string(
    suite: BenchmarkSuiteReport,
    *,
    indent: int = 2,
) -> str:
    """Export a suite report as a formatted JSON string.

    Args:
        suite: Suite report to export.
        indent: JSON indentation level (default 2).

    Returns:
        Formatted JSON string.
    """
    data = format_suite_json(suite)
    return json.dumps(data, indent=indent)
