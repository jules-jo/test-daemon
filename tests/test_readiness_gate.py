"""Tests for the daemon readiness gate.

Verifies that the readiness gate:
- Starts in a not-ready state with an appropriate reason
- Blocks test-run requests while not ready, returning structured responses
- Transitions to ready after the scan-probe-mark pipeline completes
- Accepts test-run requests after transitioning to ready
- Records the startup result when transitioning to ready
- Is immutable in its response models
- Captures timing metadata (when the gate opened)
- Provides a human-readable reason for the not-ready state
- Handles duplicate ready transitions safely (idempotent)
- Queues requests when not ready rather than rejecting
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from jules_daemon.startup.readiness_gate import (
    GateState,
    NotReadyReason,
    NotReadyResponse,
    ReadinessGate,
    RequestVerdict,
)
from jules_daemon.startup.lifecycle import (
    DaemonPhase,
    StartupResult,
)


# ---------------------------------------------------------------------------
# GateState enum tests
# ---------------------------------------------------------------------------


class TestGateState:
    """Verify gate state enumeration."""

    def test_all_states_exist(self) -> None:
        assert GateState.NOT_READY.value == "not_ready"
        assert GateState.READY.value == "ready"


# ---------------------------------------------------------------------------
# NotReadyReason enum tests
# ---------------------------------------------------------------------------


class TestNotReadyReason:
    """Verify not-ready reason enumeration."""

    def test_all_reasons_exist(self) -> None:
        assert NotReadyReason.STARTUP_IN_PROGRESS.value == "startup_in_progress"
        assert NotReadyReason.PIPELINE_PENDING.value == "pipeline_pending"


# ---------------------------------------------------------------------------
# NotReadyResponse model tests
# ---------------------------------------------------------------------------


class TestNotReadyResponse:
    """Verify the immutable not-ready response model."""

    def test_frozen(self) -> None:
        response = NotReadyResponse(
            reason=NotReadyReason.STARTUP_IN_PROGRESS,
            message="Daemon is starting up",
            retry_after_seconds=5.0,
        )
        with pytest.raises(AttributeError):
            response.reason = NotReadyReason.PIPELINE_PENDING  # type: ignore[misc]

    def test_contains_reason_and_message(self) -> None:
        response = NotReadyResponse(
            reason=NotReadyReason.STARTUP_IN_PROGRESS,
            message="Daemon startup in progress, scan-probe-mark pipeline has not completed",
            retry_after_seconds=5.0,
        )
        assert response.reason == NotReadyReason.STARTUP_IN_PROGRESS
        assert "startup" in response.message.lower()
        assert response.retry_after_seconds == 5.0

    def test_default_retry_after(self) -> None:
        response = NotReadyResponse(
            reason=NotReadyReason.STARTUP_IN_PROGRESS,
            message="Starting up",
        )
        assert response.retry_after_seconds == 5.0


# ---------------------------------------------------------------------------
# RequestVerdict model tests
# ---------------------------------------------------------------------------


class TestRequestVerdict:
    """Verify the immutable request verdict model."""

    def test_frozen(self) -> None:
        verdict = RequestVerdict(
            allowed=True,
            not_ready_response=None,
        )
        with pytest.raises(AttributeError):
            verdict.allowed = False  # type: ignore[misc]

    def test_allowed_verdict(self) -> None:
        verdict = RequestVerdict(allowed=True, not_ready_response=None)
        assert verdict.allowed is True
        assert verdict.not_ready_response is None

    def test_blocked_verdict(self) -> None:
        response = NotReadyResponse(
            reason=NotReadyReason.STARTUP_IN_PROGRESS,
            message="Not ready",
        )
        verdict = RequestVerdict(allowed=False, not_ready_response=response)
        assert verdict.allowed is False
        assert verdict.not_ready_response is not None
        assert verdict.not_ready_response.reason == NotReadyReason.STARTUP_IN_PROGRESS


# ---------------------------------------------------------------------------
# ReadinessGate: initial state
# ---------------------------------------------------------------------------


class TestReadinessGateInitialState:
    """Gate starts in not-ready state."""

    def test_starts_not_ready(self) -> None:
        gate = ReadinessGate()
        assert gate.state == GateState.NOT_READY

    def test_is_ready_returns_false_initially(self) -> None:
        gate = ReadinessGate()
        assert gate.is_ready is False

    def test_ready_at_is_none_initially(self) -> None:
        gate = ReadinessGate()
        assert gate.ready_at is None

    def test_startup_result_is_none_initially(self) -> None:
        gate = ReadinessGate()
        assert gate.startup_result is None


# ---------------------------------------------------------------------------
# ReadinessGate: blocking requests
# ---------------------------------------------------------------------------


class TestReadinessGateBlocking:
    """Gate blocks test-run requests while not ready."""

    def test_check_returns_not_allowed_when_not_ready(self) -> None:
        gate = ReadinessGate()
        verdict = gate.check_request()
        assert verdict.allowed is False

    def test_not_ready_response_has_reason(self) -> None:
        gate = ReadinessGate()
        verdict = gate.check_request()
        assert verdict.not_ready_response is not None
        assert verdict.not_ready_response.reason == NotReadyReason.STARTUP_IN_PROGRESS

    def test_not_ready_response_has_message(self) -> None:
        gate = ReadinessGate()
        verdict = gate.check_request()
        assert verdict.not_ready_response is not None
        assert len(verdict.not_ready_response.message) > 0

    def test_not_ready_response_has_retry_after(self) -> None:
        gate = ReadinessGate()
        verdict = gate.check_request()
        assert verdict.not_ready_response is not None
        assert verdict.not_ready_response.retry_after_seconds > 0


# ---------------------------------------------------------------------------
# ReadinessGate: transitioning to ready
# ---------------------------------------------------------------------------


class TestReadinessGateTransition:
    """Gate transitions to ready after pipeline completes."""

    def _make_startup_result(
        self,
        phase: DaemonPhase = DaemonPhase.READY,
        error: str | None = None,
    ) -> StartupResult:
        return StartupResult(
            final_phase=phase,
            pipeline_result=None,
            duration_seconds=0.1,
            error=error,
            timestamp=datetime.now(timezone.utc),
        )

    def test_mark_ready_transitions_state(self) -> None:
        gate = ReadinessGate()
        result = self._make_startup_result()
        gate.mark_ready(result)
        assert gate.state == GateState.READY

    def test_is_ready_returns_true_after_transition(self) -> None:
        gate = ReadinessGate()
        result = self._make_startup_result()
        gate.mark_ready(result)
        assert gate.is_ready is True

    def test_ready_at_recorded(self) -> None:
        gate = ReadinessGate()
        before = datetime.now(timezone.utc)
        result = self._make_startup_result()
        gate.mark_ready(result)
        after = datetime.now(timezone.utc)
        assert gate.ready_at is not None
        assert before <= gate.ready_at <= after

    def test_startup_result_captured(self) -> None:
        gate = ReadinessGate()
        result = self._make_startup_result()
        gate.mark_ready(result)
        assert gate.startup_result is result


# ---------------------------------------------------------------------------
# ReadinessGate: accepting requests after ready
# ---------------------------------------------------------------------------


class TestReadinessGateAccepting:
    """Gate allows requests after transitioning to ready."""

    def _make_ready_gate(self) -> ReadinessGate:
        gate = ReadinessGate()
        result = StartupResult(
            final_phase=DaemonPhase.READY,
            pipeline_result=None,
            duration_seconds=0.1,
            error=None,
            timestamp=datetime.now(timezone.utc),
        )
        gate.mark_ready(result)
        return gate

    def test_check_returns_allowed_when_ready(self) -> None:
        gate = self._make_ready_gate()
        verdict = gate.check_request()
        assert verdict.allowed is True

    def test_no_not_ready_response_when_allowed(self) -> None:
        gate = self._make_ready_gate()
        verdict = gate.check_request()
        assert verdict.not_ready_response is None


# ---------------------------------------------------------------------------
# ReadinessGate: idempotent mark_ready
# ---------------------------------------------------------------------------


class TestReadinessGateIdempotent:
    """Duplicate mark_ready calls are safe."""

    def test_double_mark_ready_does_not_raise(self) -> None:
        gate = ReadinessGate()
        result = StartupResult(
            final_phase=DaemonPhase.READY,
            pipeline_result=None,
            duration_seconds=0.1,
            error=None,
            timestamp=datetime.now(timezone.utc),
        )
        gate.mark_ready(result)
        # Second call should not raise
        gate.mark_ready(result)
        assert gate.is_ready is True

    def test_first_result_preserved_on_double_call(self) -> None:
        gate = ReadinessGate()
        result1 = StartupResult(
            final_phase=DaemonPhase.READY,
            pipeline_result=None,
            duration_seconds=0.1,
            error=None,
            timestamp=datetime.now(timezone.utc),
        )
        result2 = StartupResult(
            final_phase=DaemonPhase.READY,
            pipeline_result=None,
            duration_seconds=0.2,
            error=None,
            timestamp=datetime.now(timezone.utc),
        )
        gate.mark_ready(result1)
        gate.mark_ready(result2)
        # First result is preserved (idempotent -- first wins)
        assert gate.startup_result is result1

    def test_first_ready_at_preserved_on_double_call(self) -> None:
        gate = ReadinessGate()
        result = StartupResult(
            final_phase=DaemonPhase.READY,
            pipeline_result=None,
            duration_seconds=0.1,
            error=None,
            timestamp=datetime.now(timezone.utc),
        )
        gate.mark_ready(result)
        first_ready_at = gate.ready_at
        gate.mark_ready(result)
        assert gate.ready_at == first_ready_at


# ---------------------------------------------------------------------------
# ReadinessGate: thread safety
# ---------------------------------------------------------------------------


class TestReadinessGateThreadSafety:
    """Gate operations are safe under concurrent access."""

    def test_concurrent_check_and_mark(self) -> None:
        gate = ReadinessGate()
        result = StartupResult(
            final_phase=DaemonPhase.READY,
            pipeline_result=None,
            duration_seconds=0.1,
            error=None,
            timestamp=datetime.now(timezone.utc),
        )

        errors: list[Exception] = []
        verdicts: list[RequestVerdict] = []

        def check_loop() -> None:
            try:
                for _ in range(100):
                    v = gate.check_request()
                    verdicts.append(v)
            except Exception as exc:
                errors.append(exc)

        def mark_once() -> None:
            try:
                gate.mark_ready(result)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=check_loop),
            threading.Thread(target=check_loop),
            threading.Thread(target=mark_once),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0
        assert gate.is_ready is True
        # All verdicts should be valid (either allowed or not)
        for v in verdicts:
            assert isinstance(v.allowed, bool)


# ---------------------------------------------------------------------------
# ReadinessGate: mark_ready with startup errors
# ---------------------------------------------------------------------------


class TestReadinessGateWithErrors:
    """Gate transitions to ready even when startup had errors.

    Per the lifecycle contract, errors do NOT prevent reaching READY.
    The gate respects this -- a StartupResult with final_phase=READY
    and an error still opens the gate.
    """

    def test_ready_with_pipeline_error(self) -> None:
        gate = ReadinessGate()
        result = StartupResult(
            final_phase=DaemonPhase.READY,
            pipeline_result=None,
            duration_seconds=1.5,
            error="Pipeline timed out after 30.0s -- continuing to READY",
            timestamp=datetime.now(timezone.utc),
        )
        gate.mark_ready(result)
        assert gate.is_ready is True
        assert gate.startup_result is not None
        assert gate.startup_result.error is not None


# ---------------------------------------------------------------------------
# ReadinessGate: snapshot method
# ---------------------------------------------------------------------------


class TestReadinessGateSnapshot:
    """Gate provides an immutable snapshot of its current state."""

    def test_snapshot_not_ready(self) -> None:
        gate = ReadinessGate()
        snapshot = gate.snapshot()
        assert snapshot.state == GateState.NOT_READY
        assert snapshot.ready_at is None
        assert snapshot.startup_error is None

    def test_snapshot_ready(self) -> None:
        gate = ReadinessGate()
        result = StartupResult(
            final_phase=DaemonPhase.READY,
            pipeline_result=None,
            duration_seconds=0.1,
            error=None,
            timestamp=datetime.now(timezone.utc),
        )
        gate.mark_ready(result)
        snapshot = gate.snapshot()
        assert snapshot.state == GateState.READY
        assert snapshot.ready_at is not None
        assert snapshot.startup_error is None

    def test_snapshot_ready_with_error(self) -> None:
        gate = ReadinessGate()
        result = StartupResult(
            final_phase=DaemonPhase.READY,
            pipeline_result=None,
            duration_seconds=1.0,
            error="Pipeline timed out",
            timestamp=datetime.now(timezone.utc),
        )
        gate.mark_ready(result)
        snapshot = gate.snapshot()
        assert snapshot.state == GateState.READY
        assert snapshot.startup_error == "Pipeline timed out"

    def test_snapshot_is_frozen(self) -> None:
        gate = ReadinessGate()
        snapshot = gate.snapshot()
        with pytest.raises(AttributeError):
            snapshot.state = GateState.READY  # type: ignore[misc]
