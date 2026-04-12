"""Failure rate spike tracker anomaly detector.

Concrete implementation of the ``AnomalyDetector`` protocol that uses a
sliding window counter to detect spikes in failure rates over a
configurable time window. When the number of failure events within the
window exceeds the configured threshold, the detector signals a spike.

Failure events are identified by matching SSH output lines against a
configurable regex pattern. Lines that match the regex are recorded as
failure timestamps in a deque-based sliding window. Timestamps older
than ``window_seconds`` are evicted on every operation, keeping memory
bounded and counts accurate.

Key properties:

- **Sliding window**: Uses a ``collections.deque`` of UTC timestamps to
  track failure events. Expired entries are evicted on every ``match()``,
  ``record_failure()``, and ``current_count`` access.

- **Configurable threshold**: Spike is signaled when the count of
  failures within the window reaches or exceeds ``threshold_count``
  from the ``FailureRatePattern`` configuration.

- **Regex-based failure identification**: Each output line is tested
  against a compiled regex (default: ``(?i)FAIL|ERROR|FAILED``). Only
  matching lines increment the failure counter.

- **Explicit recording**: The ``record_failure()`` method allows
  callers to record failure events directly (bypassing the regex),
  useful when failures are detected by other means (e.g., exit codes).

- **Injectable time function**: Accepts an optional ``time_func``
  callable for deterministic testing. Defaults to
  ``datetime.now(timezone.utc)``.

- **Thread-safe**: All mutable state access is guarded by a
  ``threading.Lock``, safe for concurrent reads and writes from
  multiple asyncio tasks or OS threads.

- **Protocol conformance**: Satisfies the ``AnomalyDetector`` protocol
  via structural subtyping. No inheritance required.

Usage::

    from jules_daemon.monitor.anomaly_models import (
        AnomalySeverity,
        FailureRatePattern,
    )
    from jules_daemon.monitor.failure_rate_spike_detector import (
        FailureRateSpikeDetector,
    )

    pattern = FailureRatePattern(
        name="high_fail",
        threshold_count=5,
        window_seconds=60.0,
        severity=AnomalySeverity.WARNING,
    )

    detector = FailureRateSpikeDetector(pattern=pattern)

    for line in ssh_output_lines:
        if detector.match(line):
            report = detector.report(line, session_id="run-1", detected_at=now)
            # ... notify user about the failure rate spike
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Callable, Final

from jules_daemon.monitor.anomaly_models import (
    AnomalyReport,
    FailureRatePattern,
    PatternType,
)

__all__ = ["FailureRateSpikeDetector"]

logger = logging.getLogger(__name__)

_DEFAULT_NAME: Final[str] = "failure_rate_spike_detector"
_DEFAULT_FAILURE_REGEX: Final[str] = r"(?i)FAIL|ERROR|FAILED"


def _utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class FailureRateSpikeDetector:
    """Sliding-window failure rate spike anomaly detector.

    Tracks failure events in a time-bounded sliding window and signals
    a spike when the failure count reaches or exceeds the configured
    threshold. Conforms to the ``AnomalyDetector`` protocol via
    structural subtyping.

    Attributes (read-only via properties):
        pattern_name: Unique identifier for this detector instance.
        pattern_type: Always ``PatternType.FAILURE_RATE``.
        threshold: Number of failures that triggers a spike.
        window_seconds: Size of the sliding time window in seconds.
        current_count: Current number of failures within the window.
        failure_regex: Source regex string for failure identification.
        compiled_failure_regex: Pre-compiled regex pattern.
    """

    __slots__ = (
        "_compiled_regex",
        "_failure_regex",
        "_lock",
        "_name",
        "_pattern",
        "_time_func",
        "_timestamps",
    )

    def __init__(
        self,
        *,
        pattern: FailureRatePattern,
        failure_regex: str = _DEFAULT_FAILURE_REGEX,
        name: str = _DEFAULT_NAME,
        time_func: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize with a failure rate pattern configuration.

        Args:
            pattern: Immutable ``FailureRatePattern`` providing the
                threshold count, window size, and severity.
            failure_regex: Regex to identify failure lines in SSH output.
                Lines matching this regex are counted as failure events.
                Must be a valid Python regex. Defaults to matching
                common failure keywords (FAIL, ERROR, FAILED).
            name: Identifier for this detector instance. Defaults to
                ``"failure_rate_spike_detector"``.
            time_func: Optional callable returning the current UTC time.
                Defaults to ``datetime.now(timezone.utc)``. Inject a
                custom function for deterministic testing.

        Raises:
            ValueError: If name is blank, failure_regex is blank or
                invalid.
        """
        stripped_name = name.strip()
        if not stripped_name:
            raise ValueError("name must not be empty")

        stripped_regex = failure_regex.strip()
        if not stripped_regex:
            raise ValueError("failure_regex must not be empty")

        try:
            compiled = re.compile(stripped_regex)
        except re.error as exc:
            raise ValueError(
                f"failure_regex is not a valid regular expression: {exc}"
            ) from exc

        self._name: Final[str] = stripped_name
        self._pattern: Final[FailureRatePattern] = pattern
        self._failure_regex: Final[str] = stripped_regex
        self._compiled_regex: Final[re.Pattern[str]] = compiled
        self._time_func: Final[Callable[[], datetime]] = (
            time_func if time_func is not None else _utc_now
        )
        self._timestamps: deque[datetime] = deque()
        self._lock: Final[threading.Lock] = threading.Lock()

    # ------------------------------------------------------------------
    # AnomalyDetector protocol properties
    # ------------------------------------------------------------------

    @property
    def pattern_name(self) -> str:
        """Unique identifier of this detector instance."""
        return self._name

    @property
    def pattern_type(self) -> PatternType:
        """Always ``PatternType.FAILURE_RATE`` for this detector."""
        return PatternType.FAILURE_RATE

    # ------------------------------------------------------------------
    # Configuration properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> int:
        """Number of failures within the window that triggers a spike."""
        return self._pattern.threshold_count

    @property
    def window_seconds(self) -> float:
        """Size of the sliding time window in seconds."""
        return self._pattern.window_seconds

    @property
    def failure_regex(self) -> str:
        """Source regex string for failure identification."""
        return self._failure_regex

    @property
    def compiled_failure_regex(self) -> re.Pattern[str]:
        """Pre-compiled regex pattern for failure identification."""
        return self._compiled_regex

    @property
    def current_count(self) -> int:
        """Current number of failures within the sliding window.

        Evicts expired timestamps before counting. Thread-safe.

        Returns:
            Number of failure events currently in the window.
        """
        now = self._time_func()
        with self._lock:
            self._evict_expired(now)
            return len(self._timestamps)

    # ------------------------------------------------------------------
    # AnomalyDetector protocol methods
    # ------------------------------------------------------------------

    def match(self, output_line: str) -> bool:
        """Test whether an output line is a failure and spike threshold met.

        If the line matches the failure regex, a timestamp is recorded
        in the sliding window. Then expired timestamps are evicted and
        the current count is compared against the threshold.

        Returns True only when the failure count within the window
        reaches or exceeds the threshold -- indicating a failure rate
        spike. Lines that do not match the failure regex return False
        without affecting the window.

        Args:
            output_line: A single line of SSH output to test.

        Returns:
            True if a failure rate spike is detected (count >= threshold).
        """
        if not self._compiled_regex.search(output_line):
            # Not a failure line -- still evict expired entries
            now = self._time_func()
            with self._lock:
                self._evict_expired(now)
            return False

        now = self._time_func()
        with self._lock:
            self._timestamps.append(now)
            self._evict_expired(now)
            return len(self._timestamps) >= self._pattern.threshold_count

    def report(
        self,
        output_line: str,
        *,
        session_id: str,
        detected_at: datetime,
    ) -> AnomalyReport:
        """Produce an anomaly report for a failure rate spike.

        Generates an ``AnomalyReport`` containing the current failure
        count, threshold, window size, and the triggering output line.
        Can be called regardless of whether a spike is active; the
        context always reflects the current window state.

        Args:
            output_line: The output line associated with this report.
            session_id: Identifier of the SSH session being monitored.
            detected_at: UTC timestamp of the detection.

        Returns:
            Immutable ``AnomalyReport`` with spike detection metadata.
        """
        now = self._time_func()
        with self._lock:
            self._evict_expired(now)
            count = len(self._timestamps)

        spike_active = count >= self._pattern.threshold_count

        if spike_active:
            message = (
                f"Failure rate spike '{self._pattern.name}': "
                f"{count} failures in {self._pattern.window_seconds}s "
                f"window (threshold: {self._pattern.threshold_count})"
            )
        else:
            message = (
                f"Failure rate '{self._pattern.name}': "
                f"{count}/{self._pattern.threshold_count} failures in "
                f"{self._pattern.window_seconds}s window"
            )

        return AnomalyReport(
            pattern_name=self._pattern.name,
            pattern_type=PatternType.FAILURE_RATE,
            severity=self._pattern.severity,
            message=message,
            detected_at=detected_at,
            session_id=session_id,
            context={
                "failure_count": count,
                "threshold": self._pattern.threshold_count,
                "window_seconds": self._pattern.window_seconds,
                "spike_active": spike_active,
                "output_line": output_line,
            },
        )

    # ------------------------------------------------------------------
    # Public state management
    # ------------------------------------------------------------------

    def record_failure(
        self,
        *,
        timestamp: datetime | None = None,
    ) -> bool:
        """Record a failure event explicitly.

        Use this method when failures are detected by means other than
        regex matching (e.g., process exit codes, external signals).
        The failure is recorded at the given timestamp (or the current
        time if not specified), and expired entries are evicted.

        Args:
            timestamp: UTC timestamp of the failure event. Defaults to
                the current time from the injected time function.

        Returns:
            True if the failure count now meets or exceeds the spike
            threshold.
        """
        now = self._time_func()
        ts = timestamp if timestamp is not None else now
        with self._lock:
            self._timestamps.append(ts)
            self._evict_expired(now)
            return len(self._timestamps) >= self._pattern.threshold_count

    def reset(self) -> None:
        """Clear all recorded failure timestamps.

        Resets the sliding window to empty, allowing spike detection
        to start from scratch. Thread-safe.
        """
        with self._lock:
            self._timestamps.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_expired(self, now: datetime) -> None:
        """Remove timestamps older than the sliding window.

        Timestamps are stored in chronological order (oldest first),
        so we pop from the left until all remaining entries are within
        the window. The boundary is inclusive: a timestamp exactly at
        ``now - window_seconds`` is retained.

        Must be called while holding ``self._lock``.

        Args:
            now: The current UTC time to compare against.
        """
        cutoff = now - timedelta(seconds=self._pattern.window_seconds)
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        count = 0
        with self._lock:
            count = len(self._timestamps)
        return (
            f"FailureRateSpikeDetector("
            f"name={self._name!r}, "
            f"pattern={self._pattern.name!r}, "
            f"count={count}/{self._pattern.threshold_count}, "
            f"window={self._pattern.window_seconds}s)"
        )
