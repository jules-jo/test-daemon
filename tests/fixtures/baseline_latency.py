"""Baseline latency fixtures for v1.2-mvp direct-command execution.

Captures timing data for the performance-critical fast path: input
arrives, ``detect_direct_command()`` classifies it, and the result
routes directly to SSH approval without LLM involvement.

These fixtures provide two things:

    1. **Representative inputs** -- frozen tuples of commands that
       exercise every code path in the detector (known executables,
       env prefixes, sudo, absolute paths, relative paths, NL inputs).

    2. **Measurement helpers** -- functions that run the detector in
       a tight loop and return percentile statistics (median, p95, p99).

The test suite uses these fixtures together with the threshold constants
from ``jules_daemon.agent.performance_thresholds`` to assert that
latency stays within bounds.

Usage::

    from tests.fixtures.baseline_latency import (
        DIRECT_COMMAND_BASELINE_INPUTS,
        NL_INPUT_BASELINE_INPUTS,
        measure_detection_latency,
    )

    stats = measure_detection_latency(
        inputs=DIRECT_COMMAND_BASELINE_INPUTS,
        iterations=100,
    )
    assert stats.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

from jules_daemon.classifier.direct_command import (
    DirectCommandDetection,
    detect_direct_command,
)

__all__ = [
    "DIRECT_COMMAND_BASELINE_INPUTS",
    "NL_INPUT_BASELINE_INPUTS",
    "ENV_PREFIX_BASELINE_INPUTS",
    "SUDO_PREFIX_BASELINE_INPUTS",
    "PATH_EXECUTABLE_BASELINE_INPUTS",
    "EDGE_CASE_BASELINE_INPUTS",
    "ALL_BASELINE_INPUTS",
    "LatencyStats",
    "measure_detection_latency",
    "measure_single_call",
    "measure_classification_latency",
]


# ---------------------------------------------------------------------------
# Representative input sets (frozen tuples for immutability)
# ---------------------------------------------------------------------------

DIRECT_COMMAND_BASELINE_INPUTS: tuple[str, ...] = (
    "pytest -v tests/",
    "python3 -m pytest --tb=short",
    "npm test",
    "cargo test --release",
    "go test ./...",
    "make test",
    "gradle test",
    "bash run_tests.sh",
    "docker run --rm test-image",
    "kubectl get pods -n testing",
    "ls -la /opt/app",
    "cat /var/log/test.log",
    "grep -r 'FAIL' test_output/",
    "git status",
    "java -jar runner.jar",
    "mvn test -pl module",
    "dotnet test --filter Category=Unit",
    "ruby -e 'puts :ok'",
    "node test.js",
    "pip install -r requirements.txt",
)
"""Commands starting with known executables (confidence 1.0).

Each exercises a different executable category: test runners,
shell utilities, container tools, sysadmin tools, and language
runtimes. All should produce ``is_direct_command=True``.
"""

NL_INPUT_BASELINE_INPUTS: tuple[str, ...] = (
    "run the smoke tests on staging",
    "can you check what's running?",
    "please execute the integration suite",
    "I need to see the test results",
    "what tests failed yesterday?",
    "deploy the latest build and verify",
    "show me the logs from the last run",
    "how long did the regression suite take?",
    "are there any flaky tests in the pipeline?",
    "kick off the nightly build",
)
"""Natural language inputs that should NOT trigger direct-command bypass.

All should produce ``is_direct_command=False`` with ``confidence=0.0``.
These exercise the full detection pipeline -- env prefix stripping,
sudo stripping, executable extraction -- only to fall through to
non-detection.
"""

ENV_PREFIX_BASELINE_INPUTS: tuple[str, ...] = (
    "PYTHONPATH=/opt/app pytest -v tests/",
    "DJANGO_SETTINGS_MODULE=settings python3 manage.py test",
    "LANG=C LC_ALL=C make test",
    "HOME=/tmp TERM=xterm cargo test",
    "NODE_ENV=test npm run test:unit",
)
"""Commands with environment variable prefixes.

The detector must strip ``VAR=value`` prefixes before matching the
executable, exercising the ``_strip_env_prefixes()`` path.
"""

SUDO_PREFIX_BASELINE_INPUTS: tuple[str, ...] = (
    "sudo pytest -v tests/",
    "sudo -u testuser python3 test.py",
    "sudo -u deploy cargo test --release",
    "sudo bash run_tests.sh",
)
"""Commands with sudo prefixes.

The detector must strip ``sudo [-flag [arg]]`` before matching,
exercising the ``_strip_sudo_prefix()`` path.
"""

PATH_EXECUTABLE_BASELINE_INPUTS: tuple[str, ...] = (
    "/usr/bin/python3 test.py",
    "/usr/local/bin/pytest -v tests/",
    "/opt/custom/runner --suite smoke",
    "./gradlew test",
    "./run_tests.sh --verbose",
    "./custom_script.sh",
)
"""Commands using absolute or relative executable paths.

Absolute paths exercise the path-based detection (confidence 0.8).
Relative ``./`` paths exercise the dot-slash stripping logic.
"""

EDGE_CASE_BASELINE_INPUTS: tuple[str, ...] = (
    "",
    "   ",
    "python3",
    "  pytest -v tests/  ",
    "frobnicator --test",
    "status",
    "watch --tail 100",
    "cancel --force",
    "history --limit 20",
    "cd /opt/app && pytest -v",
    "pytest -v 2>&1 | tee output.log",
)
"""Edge cases: empty, whitespace, bare executable, daemon verbs, chains.

Tests the boundaries of the detection algorithm to ensure no latency
spike on unusual inputs.
"""

ALL_BASELINE_INPUTS: tuple[str, ...] = (
    DIRECT_COMMAND_BASELINE_INPUTS
    + NL_INPUT_BASELINE_INPUTS
    + ENV_PREFIX_BASELINE_INPUTS
    + SUDO_PREFIX_BASELINE_INPUTS
    + PATH_EXECUTABLE_BASELINE_INPUTS
    + EDGE_CASE_BASELINE_INPUTS
)
"""Union of all baseline input sets for comprehensive measurement."""


# ---------------------------------------------------------------------------
# Latency measurement result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyStats:
    """Immutable percentile statistics from a latency measurement run.

    All values are in seconds (float).

    Attributes:
        min: Fastest single call.
        max: Slowest single call.
        mean: Arithmetic mean across all measured calls.
        median: 50th percentile (p50).
        p95: 95th percentile.
        p99: 99th percentile.
        samples: Total number of timed calls.
    """

    min: float
    max: float
    mean: float
    median: float
    p95: float
    p99: float
    samples: int

    def __repr__(self) -> str:
        return (
            f"LatencyStats(min={self.min*1000:.3f}ms, "
            f"median={self.median*1000:.3f}ms, "
            f"p95={self.p95*1000:.3f}ms, "
            f"p99={self.p99*1000:.3f}ms, "
            f"max={self.max*1000:.3f}ms, "
            f"samples={self.samples})"
        )


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------


def _percentile(data: list[float], pct: float) -> float:
    """Compute the given percentile from a sorted list.

    Uses nearest-rank method: finds the index at ``ceil(pct/100 * n) - 1``.
    The input list must be pre-sorted in ascending order.

    Args:
        data: Sorted list of float values.
        pct: Percentile to compute (0--100).

    Returns:
        The value at the requested percentile.
    """
    if not data:
        return 0.0
    n = len(data)
    # Nearest-rank: index = ceil(pct/100 * n) - 1, clamped to [0, n-1]
    idx = max(0, min(n - 1, int((pct / 100.0) * n + 0.5) - 1))
    return data[idx]


def measure_single_call(raw: str) -> float:
    """Measure the wall-clock time for a single ``detect_direct_command`` call.

    Uses ``time.monotonic()`` for high-resolution, monotonic timing.

    Args:
        raw: Input string to classify.

    Returns:
        Elapsed time in seconds.
    """
    start = time.monotonic()
    detect_direct_command(raw)
    return time.monotonic() - start


def measure_detection_latency(
    *,
    inputs: tuple[str, ...] | None = None,
    iterations: int = 100,
) -> LatencyStats:
    """Run ``detect_direct_command()`` in a tight loop and return percentile stats.

    Each input is processed ``iterations`` times. The total sample count
    is ``len(inputs) * iterations``.

    A single warm-up pass over all inputs is performed before measurement
    to ensure regex compilation, frozenset hashing, and module-level
    initialization are excluded from the timed data.

    Args:
        inputs: Tuple of input strings to classify. Defaults to
            ``ALL_BASELINE_INPUTS`` for comprehensive coverage.
        iterations: Number of times to process each input.

    Returns:
        LatencyStats with min/max/mean/median/p95/p99 across all samples.
    """
    effective_inputs = inputs if inputs is not None else ALL_BASELINE_INPUTS

    # Warm-up pass (untimed) to exclude first-call initialization costs
    for raw in effective_inputs:
        detect_direct_command(raw)

    # Timed measurement pass
    timings: list[float] = []
    for _ in range(iterations):
        for raw in effective_inputs:
            elapsed = measure_single_call(raw)
            timings.append(elapsed)

    timings.sort()
    return LatencyStats(
        min=timings[0],
        max=timings[-1],
        mean=statistics.mean(timings),
        median=statistics.median(timings),
        p95=_percentile(timings, 95),
        p99=_percentile(timings, 99),
        samples=len(timings),
    )


def measure_classification_latency(
    *,
    classify_fn: Callable[[str], object],
    inputs: tuple[str, ...],
    iterations: int = 100,
) -> LatencyStats:
    """Generic latency measurement for any classification function.

    Works with ``classify_input()``, ``detect_direct_command()``, or
    any single-argument callable that returns a result object.

    Args:
        classify_fn: The classification function to benchmark.
        inputs: Tuple of input strings to classify.
        iterations: Number of times to process each input.

    Returns:
        LatencyStats with percentile statistics.
    """
    # Warm-up pass
    for raw in inputs:
        classify_fn(raw)

    # Timed measurement
    timings: list[float] = []
    for _ in range(iterations):
        for raw in inputs:
            start = time.monotonic()
            classify_fn(raw)
            elapsed = time.monotonic() - start
            timings.append(elapsed)

    timings.sort()
    return LatencyStats(
        min=timings[0],
        max=timings[-1],
        mean=statistics.mean(timings),
        median=statistics.median(timings),
        p95=_percentile(timings, 95),
        p99=_percentile(timings, 99),
        samples=len(timings),
    )
