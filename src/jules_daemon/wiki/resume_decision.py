"""Resume-or-restart decision logic for interrupted test runs.

Takes an extracted Checkpoint (from checkpoint_extractor) and optional
CurrentRun metadata, applies configurable heuristics to decide whether
the daemon should resume from the checkpoint or restart the full suite.

Heuristics applied:
1. **Staleness**: How long since the checkpoint was last updated.
   Stale checkpoints indicate the test environment may have changed.
2. **Failure ratio**: What fraction of completed tests have failed.
   High failure rates suggest systemic issues that a restart after
   fixing is more useful than continuing.
3. **Completion percentage**: How much of the suite has been run.
   Very low completion makes a restart cheap and avoids complexity.
4. **Minimum tests completed**: Whether enough tests have actually
   run to justify the overhead of resuming.

Special cases:
- PENDING_APPROVAL: Always resumes (re-prompt the user for approval).
- Non-resumable checkpoints (corrupted, no state, terminal): Always
  restarts since there is nothing to resume from.

All thresholds are configurable via the ResumeThresholds dataclass.
The function never raises -- all error paths return RESTART with a
descriptive reason.

Usage:
    from pathlib import Path
    from jules_daemon.wiki.checkpoint_extractor import extract_checkpoint
    from jules_daemon.wiki.resume_decision import decide_resume_or_restart

    cp = extract_checkpoint(Path("wiki"))
    verdict = decide_resume_or_restart(cp)
    if verdict.should_resume:
        # Resume from cp.test_index
        ...
    else:
        # Restart the full suite
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from jules_daemon.wiki.checkpoint_extractor import Checkpoint
from jules_daemon.wiki.models import CurrentRun, RunStatus

__all__ = [
    "ResumeDecision",
    "ResumeThresholds",
    "ResumeVerdict",
    "ResumeVerdictFactor",
    "decide_resume_or_restart",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and configuration
# ---------------------------------------------------------------------------


class ResumeDecision(Enum):
    """The binary outcome of the resume-or-restart decision."""

    RESUME = "resume"
    RESTART = "restart"


@dataclass(frozen=True)
class ResumeThresholds:
    """Configurable thresholds for the resume-or-restart heuristics.

    All thresholds use "pass at boundary" semantics (<=, >=), meaning
    values exactly at the threshold pass the check.

    Attributes:
        max_staleness_seconds: Maximum age (seconds) of the checkpoint
            before it is considered too stale to resume. Default: 3600 (1h).
        max_failure_ratio: Maximum fraction of completed tests that may
            have failed before triggering a restart. Ratio is computed as
            tests_failed / tests_completed. Default: 0.5 (50%).
        min_completion_percent: Minimum completion percentage required
            to justify resuming. Below this threshold, restarting is
            considered cheaper. Default: 5.0%.
        min_tests_completed: Minimum number of individual tests that
            must have completed (passed + failed + skipped) to justify
            resuming. Default: 1.
    """

    max_staleness_seconds: float = 3600.0
    max_failure_ratio: float = 0.5
    min_completion_percent: float = 5.0
    min_tests_completed: int = 1


# ---------------------------------------------------------------------------
# Verdict and factor models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResumeVerdictFactor:
    """Result of a single heuristic evaluation.

    Each factor represents one check that was applied during the
    resume-or-restart decision.

    Attributes:
        name: Machine-readable identifier for the heuristic
            (e.g., "staleness", "failure_ratio", "completion", "min_tests").
        passed: Whether this specific check passed (True) or failed (False).
        detail: Human-readable description of the check result.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ResumeVerdict:
    """Immutable result of the resume-or-restart decision.

    Contains the binary decision, human-readable reasoning, the
    individual factor evaluations, and references to the input data.

    Attributes:
        decision: RESUME or RESTART.
        reason: Human-readable summary of why this decision was made.
        factors: Tuple of individual heuristic evaluations. Each factor
            indicates whether it passed or failed and why.
        checkpoint: Reference to the Checkpoint that was evaluated.
        run_id: The run identifier from the checkpoint.
    """

    decision: ResumeDecision
    reason: str
    factors: tuple[ResumeVerdictFactor, ...]
    checkpoint: Checkpoint
    run_id: str

    @property
    def should_resume(self) -> bool:
        """True if the decision is to resume from checkpoint."""
        return self.decision == ResumeDecision.RESUME

    @property
    def should_restart(self) -> bool:
        """True if the decision is to restart the full suite."""
        return self.decision == ResumeDecision.RESTART


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _compute_staleness_seconds(
    checkpoint: Checkpoint,
    run: Optional[CurrentRun],
    now: datetime,
) -> float:
    """Compute how many seconds have elapsed since the checkpoint was recorded.

    Uses checkpoint_at as the primary timestamp. Falls back to
    run.updated_at if checkpoint_at is None and a CurrentRun is provided.
    Falls back to now (0 staleness) as a last resort.

    Args:
        checkpoint: The extracted checkpoint.
        run: Optional CurrentRun for fallback timestamp.
        now: Reference time for computation.

    Returns:
        Non-negative float of elapsed seconds.
    """
    reference_time: datetime | None = checkpoint.checkpoint_at

    if reference_time is None and run is not None:
        reference_time = run.updated_at

    if reference_time is None:
        # No timestamp available -- treat as fresh to avoid false restarts
        return 0.0

    delta = now - reference_time
    return max(0.0, delta.total_seconds())


def _evaluate_staleness(
    checkpoint: Checkpoint,
    run: Optional[CurrentRun],
    thresholds: ResumeThresholds,
    now: datetime,
) -> ResumeVerdictFactor:
    """Evaluate the staleness heuristic.

    Returns a passing factor if the checkpoint age is within the threshold,
    or a failing factor if it is too stale.
    """
    staleness = _compute_staleness_seconds(checkpoint, run, now)
    passed = staleness <= thresholds.max_staleness_seconds

    if passed:
        detail = (
            f"Checkpoint age {staleness:.1f}s is within "
            f"{thresholds.max_staleness_seconds:.0f}s threshold"
        )
    else:
        detail = (
            f"Checkpoint age {staleness:.1f}s exceeds "
            f"{thresholds.max_staleness_seconds:.0f}s threshold -- too stale"
        )

    return ResumeVerdictFactor(name="staleness", passed=passed, detail=detail)


def _evaluate_failure_ratio(
    checkpoint: Checkpoint,
    thresholds: ResumeThresholds,
) -> ResumeVerdictFactor:
    """Evaluate the failure ratio heuristic.

    The ratio is tests_failed / tests_completed. When no tests have been
    completed, the check passes (no evidence of failures).
    """
    completed = checkpoint.tests_completed

    if completed == 0:
        return ResumeVerdictFactor(
            name="failure_ratio",
            passed=True,
            detail="No tests completed yet -- failure ratio check not applicable",
        )

    ratio = checkpoint.tests_failed / completed
    passed = ratio <= thresholds.max_failure_ratio

    if passed:
        detail = (
            f"Failure ratio {ratio:.2f} ({checkpoint.tests_failed}/{completed}) "
            f"is within {thresholds.max_failure_ratio:.2f} threshold"
        )
    else:
        detail = (
            f"Failure ratio {ratio:.2f} ({checkpoint.tests_failed}/{completed}) "
            f"exceeds {thresholds.max_failure_ratio:.2f} threshold -- too many failures"
        )

    return ResumeVerdictFactor(name="failure_ratio", passed=passed, detail=detail)


def _evaluate_completion(
    checkpoint: Checkpoint,
    thresholds: ResumeThresholds,
) -> ResumeVerdictFactor:
    """Evaluate the completion percentage heuristic.

    Checks whether enough of the suite has been completed to justify
    the overhead of resuming rather than starting fresh.
    """
    passed = checkpoint.percent >= thresholds.min_completion_percent

    if passed:
        detail = (
            f"Completion {checkpoint.percent:.1f}% meets "
            f"{thresholds.min_completion_percent:.1f}% minimum"
        )
    else:
        detail = (
            f"Completion {checkpoint.percent:.1f}% is below "
            f"{thresholds.min_completion_percent:.1f}% minimum -- "
            f"restarting is cheaper"
        )

    return ResumeVerdictFactor(name="completion", passed=passed, detail=detail)


def _evaluate_min_tests(
    checkpoint: Checkpoint,
    thresholds: ResumeThresholds,
) -> ResumeVerdictFactor:
    """Evaluate the minimum tests completed heuristic.

    Checks whether enough individual tests have been processed to
    justify resuming. A run that never completed any tests should
    restart cleanly.
    """
    completed = checkpoint.tests_completed
    passed = completed >= thresholds.min_tests_completed

    if passed:
        detail = (
            f"{completed} tests completed meets "
            f"{thresholds.min_tests_completed} minimum"
        )
    else:
        detail = (
            f"{completed} tests completed is below "
            f"{thresholds.min_tests_completed} minimum -- "
            f"not enough progress to resume"
        )

    return ResumeVerdictFactor(name="min_tests", passed=passed, detail=detail)


def _build_reason(
    decision: ResumeDecision,
    factors: tuple[ResumeVerdictFactor, ...],
) -> str:
    """Build a human-readable reason string from the decision and factors.

    For RESUME: summarizes that all checks passed.
    For RESTART: lists the specific checks that failed.
    """
    if decision == ResumeDecision.RESUME:
        return "All heuristics passed -- safe to resume from checkpoint"

    failed = [f for f in factors if not f.passed]
    if not failed:
        # Should not happen, but handle defensively
        return "Restart recommended"

    failed_names = ", ".join(f.name for f in failed)
    failed_details = "; ".join(f.detail for f in failed)
    return f"Restart recommended -- failed checks: {failed_names}. {failed_details}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decide_resume_or_restart(
    checkpoint: Checkpoint,
    *,
    run: Optional[CurrentRun] = None,
    thresholds: Optional[ResumeThresholds] = None,
    now: Optional[datetime] = None,
) -> ResumeVerdict:
    """Decide whether to resume from checkpoint or restart the full suite.

    Takes the extracted checkpoint and optional run metadata, applies
    configurable heuristics, and returns a ResumeVerdict with the
    binary decision, reasoning, and individual factor evaluations.

    Decision flow:
    1. If checkpoint is not resumable (corrupted, no state, terminal,
       idle): always RESTART.
    2. If checkpoint is PENDING_APPROVAL: always RESUME (re-prompt user).
    3. Otherwise, evaluate heuristics:
       a. Staleness: is the checkpoint too old?
       b. Failure ratio: have too many tests failed?
       c. Completion: has enough of the suite been completed?
       d. Minimum tests: have enough individual tests been processed?
    4. If all heuristics pass: RESUME. If any fail: RESTART.

    This function never raises. All error paths return RESTART with a
    descriptive reason.

    Args:
        checkpoint: The extracted checkpoint from the wiki state.
        run: Optional CurrentRun for supplementary metadata (e.g.,
            fallback timestamp when checkpoint_at is None).
        thresholds: Configurable heuristic thresholds. Uses defaults
            if not provided.
        now: Reference time for staleness computation. Defaults to
            current UTC time.

    Returns:
        ResumeVerdict with decision, reason, factors, and references.
    """
    if thresholds is None:
        thresholds = ResumeThresholds()

    if now is None:
        now = _now_utc()

    # -- Gate 1: Non-resumable checkpoints always restart --
    if not checkpoint.is_resumable:
        reason = (
            f"Checkpoint is not resumable (source={checkpoint.source.value}, "
            f"status={checkpoint.status.value})"
        )
        logger.info("Resume decision: RESTART -- %s", reason)
        return ResumeVerdict(
            decision=ResumeDecision.RESTART,
            reason=reason,
            factors=(),
            checkpoint=checkpoint,
            run_id=checkpoint.run_id,
        )

    # -- Gate 2: PENDING_APPROVAL always resumes (re-prompt user) --
    if checkpoint.status == RunStatus.PENDING_APPROVAL:
        reason = (
            "Pending approval detected -- resuming to re-prompt user "
            "for command confirmation"
        )
        logger.info("Resume decision: RESUME -- %s", reason)
        return ResumeVerdict(
            decision=ResumeDecision.RESUME,
            reason=reason,
            factors=(),
            checkpoint=checkpoint,
            run_id=checkpoint.run_id,
        )

    # -- Gate 3: Evaluate heuristics for RUNNING state --
    staleness_factor = _evaluate_staleness(checkpoint, run, thresholds, now)
    failure_factor = _evaluate_failure_ratio(checkpoint, thresholds)
    completion_factor = _evaluate_completion(checkpoint, thresholds)
    min_tests_factor = _evaluate_min_tests(checkpoint, thresholds)

    factors = (
        staleness_factor,
        failure_factor,
        completion_factor,
        min_tests_factor,
    )

    all_passed = all(f.passed for f in factors)
    decision = ResumeDecision.RESUME if all_passed else ResumeDecision.RESTART
    reason = _build_reason(decision, factors)

    logger.info(
        "Resume decision: %s -- run_id=%s. Factors: %s",
        decision.value,
        checkpoint.run_id,
        ", ".join(f"{f.name}={'pass' if f.passed else 'FAIL'}" for f in factors),
    )

    return ResumeVerdict(
        decision=decision,
        reason=reason,
        factors=factors,
        checkpoint=checkpoint,
        run_id=checkpoint.run_id,
    )
