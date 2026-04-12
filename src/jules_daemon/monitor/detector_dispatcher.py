"""Fan-out dispatcher that sends each output line to all registered detectors.

Consumes output lines from the SSH monitoring pipeline and dispatches
each line to every detector registered in a ``DetectorRegistry``. All
detectors are evaluated concurrently using ``asyncio.gather`` with
``asyncio.to_thread`` for thread-safe execution.

The dispatcher is the bridge between the ``OutputMonitor`` (which
buffers raw lines) and the anomaly detection subsystem (which evaluates
patterns against those lines). Each dispatch produces an immutable
``DispatchResult`` containing any anomaly reports and any detector
errors.

Key properties:

- **Concurrent fan-out**: All registered detectors are evaluated in
  parallel via ``asyncio.gather(asyncio.to_thread(...))``. This
  ensures that slow detectors (e.g., those with lock contention)
  do not block faster ones.

- **Error isolation**: If a detector's ``match()`` or ``report()``
  method raises, the error is captured in a ``DetectorError`` record
  and included in the ``DispatchResult``. Other detectors continue
  running normally.

- **Immutable results**: ``DispatchResult``, ``DetectorError`` are
  frozen dataclasses. Safe to pass across async boundaries.

- **Session-scoped**: Each dispatch call requires a ``session_id``
  that is passed through to detector ``report()`` calls and captured
  in the result.

Usage::

    from jules_daemon.monitor.detector_dispatcher import DetectorDispatcher
    from jules_daemon.monitor.detector_registry import DetectorRegistry

    registry = DetectorRegistry()
    # ... register detectors ...

    dispatcher = DetectorDispatcher(registry=registry)

    result = await dispatcher.dispatch(
        "SIGSEGV at 0xdeadbeef",
        session_id="run-42",
    )

    if result.has_anomalies:
        for report in result.reports:
            print(f"Anomaly: {report.message}")

    if result.has_errors:
        for error in result.errors:
            print(f"Detector error: {error.detector_name}: {error.error}")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

from jules_daemon.monitor.anomaly_models import AnomalyDetector, AnomalyReport
from jules_daemon.monitor.detector_registry import DetectorRegistry

__all__ = [
    "DetectorDispatcher",
    "DetectorError",
    "DispatchResult",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data models (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorError:
    """Immutable record of a detector that raised during dispatch.

    Attributes:
        detector_name: The ``pattern_name`` of the detector that failed.
        error: Human-readable error message (str(exception)).
    """

    detector_name: str
    error: str


@dataclass(frozen=True)
class DispatchResult:
    """Immutable result of dispatching a line to all registered detectors.

    Contains the original output line, any anomaly reports produced by
    matching detectors, any errors from detectors that raised, and a
    timestamp of when the dispatch occurred.

    Attributes:
        output_line: The output line that was dispatched.
        session_id: Identifier of the SSH session being monitored.
        reports: Tuple of ``AnomalyReport`` instances from matching
            detectors. Empty when no detector matched.
        errors: Tuple of ``DetectorError`` records from detectors that
            raised exceptions. Empty when all detectors ran cleanly.
        dispatched_at: UTC timestamp of when the dispatch was initiated.
    """

    output_line: str
    session_id: str
    reports: tuple[AnomalyReport, ...]
    errors: tuple[DetectorError, ...]
    dispatched_at: datetime

    @property
    def has_anomalies(self) -> bool:
        """True if any detector produced an anomaly report."""
        return len(self.reports) > 0

    @property
    def has_errors(self) -> bool:
        """True if any detector raised an exception during dispatch."""
        return len(self.errors) > 0


# ---------------------------------------------------------------------------
# Internal result type for per-detector evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DetectorOutcome:
    """Internal: result of evaluating a single detector against a line."""

    detector_name: str
    report: AnomalyReport | None
    error: DetectorError | None


# ---------------------------------------------------------------------------
# DetectorDispatcher
# ---------------------------------------------------------------------------


class DetectorDispatcher:
    """Fan-out dispatcher that sends output lines to all registered detectors.

    Evaluates all detectors in the registry concurrently for each output
    line and collects results into an immutable ``DispatchResult``.

    Thread safety:
        This class is safe for single-threaded async use. The actual
        detector ``match()`` and ``report()`` calls are dispatched to
        the thread pool via ``asyncio.to_thread`` so thread-safe
        detectors (which use internal locks) work correctly.

    Args:
        registry: The ``DetectorRegistry`` containing detectors to
            dispatch to.
    """

    __slots__ = ("_registry",)

    def __init__(self, *, registry: DetectorRegistry) -> None:
        """Initialize with a detector registry.

        Args:
            registry: The registry providing detectors for dispatch.
        """
        self._registry: Final[DetectorRegistry] = registry

    @property
    def registry(self) -> DetectorRegistry:
        """The underlying detector registry."""
        return self._registry

    async def dispatch(
        self,
        output_line: str,
        *,
        session_id: str,
    ) -> DispatchResult:
        """Dispatch an output line to all registered detectors concurrently.

        Takes a snapshot of the current registry contents, then evaluates
        each detector's ``match()`` method in parallel. For detectors
        that match, calls ``report()`` to produce an ``AnomalyReport``.

        Detector exceptions are captured in ``DetectorError`` records and
        do not affect other detectors.

        Args:
            output_line: A single line of SSH output to evaluate.
            session_id: Identifier of the SSH session being monitored.

        Returns:
            Immutable ``DispatchResult`` containing any anomaly reports
            and any detector errors.
        """
        now = datetime.now(timezone.utc)
        detectors = self._registry.list_detectors()

        if not detectors:
            return DispatchResult(
                output_line=output_line,
                session_id=session_id,
                reports=(),
                errors=(),
                dispatched_at=now,
            )

        # Fan out to all detectors concurrently
        outcomes = await asyncio.gather(
            *(
                self._evaluate_detector(
                    detector=detector,
                    output_line=output_line,
                    session_id=session_id,
                    detected_at=now,
                )
                for detector in detectors
            ),
            return_exceptions=False,  # exceptions handled inside _evaluate
        )

        # Partition outcomes into reports and errors
        reports: list[AnomalyReport] = []
        errors: list[DetectorError] = []
        for outcome in outcomes:
            if outcome.error is not None:
                errors.append(outcome.error)
            elif outcome.report is not None:
                reports.append(outcome.report)

        result = DispatchResult(
            output_line=output_line,
            session_id=session_id,
            reports=tuple(reports),
            errors=tuple(errors),
            dispatched_at=now,
        )

        if result.has_anomalies:
            logger.info(
                "Dispatch produced %d anomaly report(s) for session %s",
                len(result.reports),
                session_id,
            )
        if result.has_errors:
            logger.warning(
                "Dispatch encountered %d detector error(s) for session %s",
                len(result.errors),
                session_id,
            )

        return result

    # ------------------------------------------------------------------
    # Internal: per-detector evaluation
    # ------------------------------------------------------------------

    @staticmethod
    async def _evaluate_detector(
        *,
        detector: AnomalyDetector,
        output_line: str,
        session_id: str,
        detected_at: datetime,
    ) -> _DetectorOutcome:
        """Evaluate a single detector against an output line.

        Runs the detector's ``match()`` and (if matched) ``report()``
        methods via ``asyncio.to_thread`` for thread-safe execution.
        Captures any exception as a ``DetectorError``.

        Args:
            detector: The detector to evaluate.
            output_line: The output line to test.
            session_id: SSH session identifier.
            detected_at: Timestamp for the report.

        Returns:
            ``_DetectorOutcome`` with either a report, an error, or
            neither (no match, no error).
        """
        name = detector.pattern_name
        try:
            matched = await asyncio.to_thread(
                detector.match, output_line
            )
            if not matched:
                return _DetectorOutcome(
                    detector_name=name,
                    report=None,
                    error=None,
                )

            report = await asyncio.to_thread(
                detector.report,
                output_line,
                session_id=session_id,
                detected_at=detected_at,
            )
            return _DetectorOutcome(
                detector_name=name,
                report=report,
                error=None,
            )
        except Exception as exc:
            logger.warning(
                "Detector %r raised during dispatch: %s",
                name,
                exc,
            )
            return _DetectorOutcome(
                detector_name=name,
                report=None,
                error=DetectorError(
                    detector_name=name,
                    error=str(exc),
                ),
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DetectorDispatcher(registry={self._registry!r})"
        )
