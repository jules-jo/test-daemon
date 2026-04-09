"""Staleness guard for daemon status freshness validation.

Validates that status timestamps are within the acceptable freshness
threshold (default: 10 seconds) before returning status data to callers.
This ensures the ``status_freshness`` evaluation principle: running test
status is never more than 10 seconds stale.

Two API styles are provided for each status type:
- ``validate_*`` -- returns a FreshnessResult without raising
- ``require_fresh_*`` -- raises StaleStatusError when stale

Terminal and idle states bypass the staleness check entirely because
they represent final or inactive states that will not receive further
updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from jules_daemon.wiki.models import CurrentRun, RunStatus
from jules_daemon.wiki.monitor_status import MonitorStatus

__all__ = [
    "DEFAULT_STALENESS_THRESHOLD_SECONDS",
    "FreshnessResult",
    "StaleStatusError",
    "validate_monitor_freshness",
    "validate_run_freshness",
    "require_fresh_monitor_status",
    "require_fresh_run_status",
]

DEFAULT_STALENESS_THRESHOLD_SECONDS: float = 10.0

# Run statuses that are exempt from staleness checks.
# Terminal states will not receive new updates; idle has nothing to monitor.
_EXEMPT_RUN_STATUSES = frozenset({
    RunStatus.IDLE,
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
})


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _validate_threshold(threshold_seconds: float) -> None:
    """Raise ValueError if threshold is non-positive."""
    if threshold_seconds <= 0.0:
        raise ValueError(
            f"threshold_seconds must be positive, got {threshold_seconds}"
        )


def _compute_age_seconds(timestamp: datetime, now: datetime) -> float:
    """Compute the age in seconds between *timestamp* and *now*."""
    delta = now - timestamp
    return delta.total_seconds()


@dataclass(frozen=True)
class FreshnessResult:
    """Immutable result of a freshness validation check.

    Attributes:
        is_fresh: True if the status age is within the threshold.
        age_seconds: How old the status is, in seconds.
        threshold_seconds: The threshold that was applied.
    """

    is_fresh: bool
    age_seconds: float
    threshold_seconds: float

    @property
    def margin_seconds(self) -> float:
        """Remaining margin before staleness (negative if already stale)."""
        return self.threshold_seconds - self.age_seconds


class StaleStatusError(Exception):
    """Raised when a status snapshot exceeds the staleness threshold.

    Carries diagnostic attributes so callers can inspect the failure
    without parsing the error message.
    """

    def __init__(
        self,
        age_seconds: float,
        threshold_seconds: float,
        session_id: Optional[str] = None,
    ) -> None:
        self.age_seconds = age_seconds
        self.threshold_seconds = threshold_seconds
        self.session_id = session_id
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [
            f"Status is {self.age_seconds:.1f} seconds stale",
            f"(threshold: {self.threshold_seconds:.1f}s)",
        ]
        if self.session_id is not None:
            parts.append(f"[session: {self.session_id}]")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# MonitorStatus freshness
# ---------------------------------------------------------------------------


def validate_monitor_freshness(
    status: MonitorStatus,
    *,
    now: Optional[datetime] = None,
    threshold_seconds: float = DEFAULT_STALENESS_THRESHOLD_SECONDS,
) -> FreshnessResult:
    """Check whether a MonitorStatus snapshot is fresh.

    Terminal statuses (exit_status is set) are always reported as fresh
    because their timestamp will never advance.

    Args:
        status: The monitor status snapshot to validate.
        now: Reference time for age calculation (defaults to current UTC).
        threshold_seconds: Maximum acceptable age in seconds.

    Returns:
        FreshnessResult indicating fresh/stale and age details.

    Raises:
        ValueError: If threshold_seconds is non-positive.
    """
    _validate_threshold(threshold_seconds)

    if now is None:
        now = _now_utc()

    # Terminal statuses are exempt -- they represent final state.
    if status.is_terminal:
        age = _compute_age_seconds(status.timestamp, now)
        return FreshnessResult(
            is_fresh=True,
            age_seconds=age,
            threshold_seconds=threshold_seconds,
        )

    age = _compute_age_seconds(status.timestamp, now)
    is_fresh = age <= threshold_seconds

    return FreshnessResult(
        is_fresh=is_fresh,
        age_seconds=age,
        threshold_seconds=threshold_seconds,
    )


def require_fresh_monitor_status(
    status: MonitorStatus,
    *,
    now: Optional[datetime] = None,
    threshold_seconds: float = DEFAULT_STALENESS_THRESHOLD_SECONDS,
) -> MonitorStatus:
    """Return the status if fresh, or raise StaleStatusError.

    This is the ``require`` variant: use when stale data should halt
    the caller rather than be silently returned.

    Args:
        status: The monitor status snapshot to validate.
        now: Reference time for age calculation (defaults to current UTC).
        threshold_seconds: Maximum acceptable age in seconds.

    Returns:
        The same MonitorStatus instance (pass-through when fresh).

    Raises:
        StaleStatusError: If the status exceeds the threshold.
        ValueError: If threshold_seconds is non-positive.
    """
    result = validate_monitor_freshness(
        status, now=now, threshold_seconds=threshold_seconds
    )
    if not result.is_fresh:
        raise StaleStatusError(
            age_seconds=result.age_seconds,
            threshold_seconds=result.threshold_seconds,
            session_id=status.session_id,
        )
    return status


# ---------------------------------------------------------------------------
# CurrentRun freshness
# ---------------------------------------------------------------------------


def validate_run_freshness(
    run: CurrentRun,
    *,
    now: Optional[datetime] = None,
    threshold_seconds: float = DEFAULT_STALENESS_THRESHOLD_SECONDS,
) -> FreshnessResult:
    """Check whether a CurrentRun record is fresh.

    Only active run states (PENDING_APPROVAL, RUNNING) are checked for
    staleness.  Terminal and idle states are always reported as fresh.

    Args:
        run: The current run record to validate.
        now: Reference time for age calculation (defaults to current UTC).
        threshold_seconds: Maximum acceptable age in seconds.

    Returns:
        FreshnessResult indicating fresh/stale and age details.

    Raises:
        ValueError: If threshold_seconds is non-positive.
    """
    _validate_threshold(threshold_seconds)

    if now is None:
        now = _now_utc()

    age = _compute_age_seconds(run.updated_at, now)

    # Exempt statuses are always considered fresh.
    if run.status in _EXEMPT_RUN_STATUSES:
        return FreshnessResult(
            is_fresh=True,
            age_seconds=age,
            threshold_seconds=threshold_seconds,
        )

    is_fresh = age <= threshold_seconds

    return FreshnessResult(
        is_fresh=is_fresh,
        age_seconds=age,
        threshold_seconds=threshold_seconds,
    )


def require_fresh_run_status(
    run: CurrentRun,
    *,
    now: Optional[datetime] = None,
    threshold_seconds: float = DEFAULT_STALENESS_THRESHOLD_SECONDS,
) -> CurrentRun:
    """Return the run if fresh, or raise StaleStatusError.

    Args:
        run: The current run record to validate.
        now: Reference time for age calculation (defaults to current UTC).
        threshold_seconds: Maximum acceptable age in seconds.

    Returns:
        The same CurrentRun instance (pass-through when fresh).

    Raises:
        StaleStatusError: If the run exceeds the threshold.
        ValueError: If threshold_seconds is non-positive.
    """
    result = validate_run_freshness(
        run, now=now, threshold_seconds=threshold_seconds
    )
    if not result.is_fresh:
        raise StaleStatusError(
            age_seconds=result.age_seconds,
            threshold_seconds=result.threshold_seconds,
            session_id=run.run_id,
        )
    return run
