"""Stall/hang detector anomaly detector.

Concrete implementation of the ``AnomalyDetector`` protocol that tracks
elapsed time since the last SSH output line was received and flags when
a configurable timeout threshold is exceeded. Indicates a potentially
hung or stalled test process.

Unlike the regex-based detectors (``ErrorKeywordDetector``,
``FailureRateSpikeDetector``) which analyze line content, this detector
is purely time-based. It records when output was last received and
compares elapsed time against the configured ``StallTimeoutPattern``
timeout.

Key properties:

- **Time-based detection**: Monitors the gap between output lines.
  A stall is detected when no new output has been received for longer
  than ``timeout_seconds`` from the ``StallTimeoutPattern`` config.

- **Dual detection modes**: Provides both ``match()`` for inline
  detection (called when output arrives -- detects stalls that just
  ended) and ``check_stall()`` for periodic polling (detects
  currently-active stalls when no output is arriving).

- **Strict-greater-than semantics**: Stall is flagged when elapsed
  time is strictly greater than the timeout threshold. At exactly the
  boundary, the detector reports no stall.

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
        StallTimeoutPattern,
    )
    from jules_daemon.monitor.stall_hang_detector import StallHangDetector

    pattern = StallTimeoutPattern(
        name="output_stall",
        timeout_seconds=300.0,
        severity=AnomalySeverity.WARNING,
    )

    detector = StallHangDetector(pattern=pattern)

    # Inline detection: called when output arrives
    if detector.match(line):
        # A stall just ended -- the gap before this line was too long
        report = detector.report("", session_id="run-1", detected_at=now)

    # Periodic polling: check if currently stalled
    if detector.check_stall():
        report = detector.report("", session_id="run-1", detected_at=now)
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Final

from jules_daemon.monitor.anomaly_models import (
    AnomalyReport,
    AnomalySeverity,
    PatternType,
    StallTimeoutPattern,
)

__all__ = ["StallHangDetector"]

logger = logging.getLogger(__name__)

_DEFAULT_NAME: Final[str] = "stall_hang_detector"


def _utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class StallHangDetector:
    """Elapsed-time-based anomaly detector for stalled SSH output.

    Tracks the timestamp of the most recent output line and detects
    stalls when the elapsed time exceeds the configured timeout.
    Conforms to the ``AnomalyDetector`` protocol via structural
    subtyping.

    Attributes (read-only via properties):
        pattern_name: Unique identifier for this detector instance.
        pattern_type: Always ``PatternType.STALL_TIMEOUT``.
        timeout_seconds: Maximum silence duration before stall is flagged.
        last_output_time: UTC timestamp of the most recent output.
        elapsed_seconds: Seconds since the last output was received.
        severity: Severity level from the pattern configuration.
    """

    __slots__ = (
        "_last_output_time",
        "_lock",
        "_name",
        "_pattern",
        "_time_func",
    )

    def __init__(
        self,
        *,
        pattern: StallTimeoutPattern,
        name: str = _DEFAULT_NAME,
        time_func: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize with a stall timeout pattern configuration.

        Args:
            pattern: Immutable ``StallTimeoutPattern`` providing the
                timeout threshold and severity level.
            name: Identifier for this detector instance. Defaults to
                ``"stall_hang_detector"``.
            time_func: Optional callable returning the current UTC time.
                Defaults to ``datetime.now(timezone.utc)``. Inject a
                custom function for deterministic testing.

        Raises:
            ValueError: If name is blank.
        """
        stripped_name = name.strip()
        if not stripped_name:
            raise ValueError("name must not be empty")

        self._name: Final[str] = stripped_name
        self._pattern: Final[StallTimeoutPattern] = pattern
        self._time_func: Final[Callable[[], datetime]] = (
            time_func if time_func is not None else _utc_now
        )
        self._last_output_time: datetime = self._time_func()
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
        """Always ``PatternType.STALL_TIMEOUT`` for this detector."""
        return PatternType.STALL_TIMEOUT

    # ------------------------------------------------------------------
    # Configuration properties
    # ------------------------------------------------------------------

    @property
    def timeout_seconds(self) -> float:
        """Maximum seconds of silence before a stall is flagged."""
        return self._pattern.timeout_seconds

    @property
    def severity(self) -> AnomalySeverity:
        """Severity level from the pattern configuration."""
        return self._pattern.severity

    @property
    def last_output_time(self) -> datetime:
        """UTC timestamp of the most recent output.

        Thread-safe.

        Returns:
            Timestamp of the last recorded output.
        """
        with self._lock:
            return self._last_output_time

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since the last output was received.

        Uses the injected time function for the current time. Thread-safe.

        Returns:
            Elapsed time in seconds since last output.
        """
        now = self._time_func()
        with self._lock:
            delta = now - self._last_output_time
            return delta.total_seconds()

    # ------------------------------------------------------------------
    # AnomalyDetector protocol methods
    # ------------------------------------------------------------------

    def match(self, output_line: str) -> bool:
        """Record output and detect if a stall just ended.

        Called when a new output line is received from the SSH session.
        Checks whether the elapsed time since the previous output
        exceeds the timeout threshold (indicating a stall that just
        resolved), then updates the last-output timestamp to the
        current time.

        The stall condition uses strict-greater-than semantics: elapsed
        time exactly at the timeout boundary does NOT trigger a stall.

        Args:
            output_line: A single line of SSH output. The content is
                not inspected -- any line counts as output received.

        Returns:
            True if a stall was detected (elapsed time since previous
            output exceeded the timeout before this line arrived).
        """
        now = self._time_func()
        with self._lock:
            delta = now - self._last_output_time
            was_stalled = delta.total_seconds() > self._pattern.timeout_seconds
            self._last_output_time = now
            return was_stalled

    def report(
        self,
        output_line: str,
        *,
        session_id: str,
        detected_at: datetime,
    ) -> AnomalyReport:
        """Produce an anomaly report for a stall event.

        Generates an ``AnomalyReport`` containing the current elapsed
        time, timeout threshold, and whether a stall is currently active.
        Can be called regardless of whether a stall is active; the
        context always reflects the current state.

        Args:
            output_line: The output line associated with this report.
                May be empty for stall reports generated by periodic
                polling (no output to reference).
            session_id: Identifier of the SSH session being monitored.
            detected_at: UTC timestamp of the detection.

        Returns:
            Immutable ``AnomalyReport`` with stall detection metadata.
        """
        now = self._time_func()
        with self._lock:
            delta = now - self._last_output_time
            elapsed = delta.total_seconds()

        stall_active = elapsed > self._pattern.timeout_seconds

        if stall_active:
            message = (
                f"Stall detected '{self._pattern.name}': "
                f"no output for {elapsed:.1f}s "
                f"(timeout: {self._pattern.timeout_seconds}s)"
            )
        else:
            message = (
                f"No stall '{self._pattern.name}': "
                f"{elapsed:.1f}s since last output "
                f"(timeout: {self._pattern.timeout_seconds}s)"
            )

        return AnomalyReport(
            pattern_name=self._pattern.name,
            pattern_type=PatternType.STALL_TIMEOUT,
            severity=self._pattern.severity,
            message=message,
            detected_at=detected_at,
            session_id=session_id,
            context={
                "elapsed_seconds": elapsed,
                "timeout_seconds": self._pattern.timeout_seconds,
                "stall_active": stall_active,
                "output_line": output_line,
            },
        )

    # ------------------------------------------------------------------
    # Public polling and state management
    # ------------------------------------------------------------------

    def check_stall(self) -> bool:
        """Check if the SSH session is currently stalled.

        This is a read-only check intended for periodic polling. It does
        NOT update the last-output timestamp. Use ``match()`` or
        ``record_output()`` when output is actually received.

        The stall condition uses strict-greater-than semantics: elapsed
        time exactly at the timeout boundary returns False.

        Returns:
            True if elapsed time since last output exceeds the timeout.
        """
        now = self._time_func()
        with self._lock:
            delta = now - self._last_output_time
            return delta.total_seconds() > self._pattern.timeout_seconds

    def record_output(
        self,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Record that output was received, resetting the stall timer.

        Use this method when output is detected by means other than
        ``match()`` (e.g., binary data, heartbeat signals). Updates
        the last-output timestamp without performing stall detection.

        Args:
            timestamp: UTC timestamp of the output event. Defaults to
                the current time from the injected time function.
        """
        ts = timestamp if timestamp is not None else self._time_func()
        with self._lock:
            self._last_output_time = ts

    def reset(self) -> None:
        """Reset the stall timer to the current time.

        Sets the last-output timestamp to the current time (from the
        injected time function), clearing any active stall condition.
        Thread-safe.
        """
        now = self._time_func()
        with self._lock:
            self._last_output_time = now

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            elapsed = (self._time_func() - self._last_output_time).total_seconds()
        return (
            f"StallHangDetector("
            f"name={self._name!r}, "
            f"pattern={self._pattern.name!r}, "
            f"elapsed={elapsed:.1f}s, "
            f"timeout={self._pattern.timeout_seconds}s)"
        )
