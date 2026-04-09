"""Readiness gate for daemon startup.

Blocks new test-run requests until the startup scan-probe-mark pipeline
has completed successfully. The gate starts in NOT_READY state and
transitions to READY when the startup lifecycle calls mark_ready().

This module enforces the invariant that no test-run request is processed
before the daemon has finished its startup cleanup, ensuring that
orphaned sessions are properly marked stale and the wiki is in a
consistent state.

Thread-safe: all state transitions use a lock to support concurrent
access from IPC handlers and the startup lifecycle.

Usage:
    from jules_daemon.startup.readiness_gate import ReadinessGate

    gate = ReadinessGate()

    # During startup...
    startup_result = await run_startup(wiki_root)
    gate.mark_ready(startup_result)

    # When a test-run request arrives...
    verdict = gate.check_request()
    if not verdict.allowed:
        return verdict.not_ready_response  # structured error response
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from jules_daemon.startup.lifecycle import StartupResult

__all__ = [
    "GateSnapshot",
    "GateState",
    "NotReadyReason",
    "NotReadyResponse",
    "ReadinessGate",
    "RequestVerdict",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RETRY_AFTER_SECONDS = 5.0
_NOT_READY_MESSAGE = (
    "Daemon startup in progress. The scan-probe-mark pipeline has not "
    "completed yet. Please retry after the daemon finishes initialization."
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class GateState(Enum):
    """Readiness gate state.

    Values:
        NOT_READY: Daemon has not completed startup. Requests are blocked.
        READY: Startup pipeline completed. Requests are accepted.
    """

    NOT_READY = "not_ready"
    READY = "ready"


class NotReadyReason(Enum):
    """Reason the daemon is not ready to accept requests.

    Values:
        STARTUP_IN_PROGRESS: Daemon is still initializing.
        PIPELINE_PENDING: Scan-probe-mark pipeline has not finished.
    """

    STARTUP_IN_PROGRESS = "startup_in_progress"
    PIPELINE_PENDING = "pipeline_pending"


# ---------------------------------------------------------------------------
# Response models (frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NotReadyResponse:
    """Structured response returned when the gate blocks a request.

    Attributes:
        reason: Enumerated reason for the not-ready state.
        message: Human-readable explanation of why the request was blocked.
        retry_after_seconds: Suggested delay before the client retries.
    """

    reason: NotReadyReason
    message: str
    retry_after_seconds: float = _DEFAULT_RETRY_AFTER_SECONDS


@dataclass(frozen=True)
class RequestVerdict:
    """Result of checking a test-run request against the readiness gate.

    Attributes:
        allowed: True if the request may proceed. False if blocked.
        not_ready_response: Structured response explaining the block.
            None when allowed is True.
    """

    allowed: bool
    not_ready_response: Optional[NotReadyResponse]


@dataclass(frozen=True)
class GateSnapshot:
    """Immutable point-in-time snapshot of the readiness gate state.

    Suitable for serialization and IPC responses.

    Attributes:
        state: Current gate state.
        ready_at: UTC datetime when the gate transitioned to READY.
            None if still NOT_READY.
        startup_error: Error from the startup lifecycle, if any.
            None when no error occurred or gate is not yet ready.
    """

    state: GateState
    ready_at: Optional[datetime]
    startup_error: Optional[str]


# ---------------------------------------------------------------------------
# ReadinessGate
# ---------------------------------------------------------------------------


class ReadinessGate:
    """Thread-safe readiness gate for daemon startup.

    Starts in NOT_READY state. Transitions to READY via mark_ready()
    after the startup lifecycle completes. The transition is idempotent
    -- subsequent mark_ready() calls are no-ops that preserve the
    original result.

    All public methods are thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = GateState.NOT_READY
        self._ready_at: Optional[datetime] = None
        self._startup_result: Optional[StartupResult] = None

    # -- Read-only properties (all protected by lock) --

    @property
    def state(self) -> GateState:
        """Current gate state."""
        with self._lock:
            return self._state

    @property
    def is_ready(self) -> bool:
        """True if the gate has transitioned to READY."""
        with self._lock:
            return self._state == GateState.READY

    @property
    def ready_at(self) -> Optional[datetime]:
        """UTC datetime when the gate transitioned to READY, or None."""
        with self._lock:
            return self._ready_at

    @property
    def startup_result(self) -> Optional[StartupResult]:
        """The StartupResult that opened the gate, or None."""
        with self._lock:
            return self._startup_result

    # -- State transition --

    def mark_ready(self, startup_result: StartupResult) -> None:
        """Transition the gate to READY.

        Idempotent: if already READY, subsequent calls are no-ops.
        The first StartupResult and ready_at timestamp are preserved.

        Args:
            startup_result: The completed startup lifecycle result.
        """
        with self._lock:
            if self._state == GateState.READY:
                logger.debug(
                    "ReadinessGate.mark_ready called but gate is already READY "
                    "-- ignoring (idempotent)"
                )
                return

            self._state = GateState.READY
            self._ready_at = datetime.now(timezone.utc)
            self._startup_result = startup_result

            logger.info(
                "ReadinessGate: transitioned to READY at %s "
                "(startup duration=%.3fs, error=%s)",
                self._ready_at.isoformat(),
                startup_result.duration_seconds,
                startup_result.error,
            )

    # -- Request checking --

    def check_request(self) -> RequestVerdict:
        """Check whether a test-run request should be allowed.

        Returns:
            RequestVerdict with allowed=True if the gate is READY,
            or allowed=False with a structured NotReadyResponse if
            the daemon is still starting up.
        """
        with self._lock:
            if self._state == GateState.READY:
                return RequestVerdict(allowed=True, not_ready_response=None)

            return RequestVerdict(
                allowed=False,
                not_ready_response=NotReadyResponse(
                    reason=NotReadyReason.STARTUP_IN_PROGRESS,
                    message=_NOT_READY_MESSAGE,
                    retry_after_seconds=_DEFAULT_RETRY_AFTER_SECONDS,
                ),
            )

    # -- Snapshot for IPC/serialization --

    def snapshot(self) -> GateSnapshot:
        """Return an immutable snapshot of the current gate state.

        Thread-safe. The snapshot captures the gate state at the moment
        of the call and is safe to pass across thread or process
        boundaries.

        Returns:
            Frozen GateSnapshot dataclass.
        """
        with self._lock:
            startup_error: Optional[str] = None
            if self._startup_result is not None:
                startup_error = self._startup_result.error

            return GateSnapshot(
                state=self._state,
                ready_at=self._ready_at,
                startup_error=startup_error,
            )
