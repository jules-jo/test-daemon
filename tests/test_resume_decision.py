"""Tests for resume-or-restart decision logic.

Verifies that the decision function:
- Takes a Checkpoint and optional CurrentRun to decide RESUME vs RESTART
- Applies staleness heuristic: stale checkpoints trigger restart
- Applies failure ratio heuristic: too many failures trigger restart
- Applies partial completion heuristic: minimal progress favors restart
- Handles non-resumable checkpoints (corrupted, no state, terminal)
- Returns immutable ResumeVerdict with decision, reason, and factors
- Supports configurable thresholds via ResumeThresholds
- Never raises -- returns RESTART with reason for all error paths
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jules_daemon.wiki.checkpoint_extractor import (
    Checkpoint,
    CheckpointPhase,
    CheckpointSource,
)
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    Progress,
    RunStatus,
    SSHTarget,
)
from jules_daemon.wiki.resume_decision import (
    ResumeDecision,
    ResumeThresholds,
    ResumeVerdict,
    decide_resume_or_restart,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(
    *,
    phase: CheckpointPhase = CheckpointPhase.RUNNING,
    source: CheckpointSource = CheckpointSource.WIKI_STATE,
    status: RunStatus = RunStatus.RUNNING,
    test_index: int = 10,
    tests_passed: int = 8,
    tests_failed: int = 1,
    tests_skipped: int = 2,
    tests_total: int = 50,
    percent: float = 22.0,
    marker: str = "PASSED test_checkout",
    run_id: str = "run-abc-123",
    checkpoint_at: datetime | None = None,
    error: str | None = None,
) -> Checkpoint:
    """Create a Checkpoint with sensible defaults for testing."""
    return Checkpoint(
        test_index=test_index,
        phase=phase,
        marker=marker,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_skipped=tests_skipped,
        tests_total=tests_total,
        percent=percent,
        checkpoint_at=checkpoint_at or datetime.now(timezone.utc),
        run_id=run_id,
        status=status,
        source=source,
        error=error,
    )


def _make_running_run(
    *,
    updated_at: datetime | None = None,
    percent: float = 50.0,
    passed: int = 10,
    failed: int = 1,
    total: int = 50,
) -> CurrentRun:
    """Create a CurrentRun in RUNNING state."""
    target = SSHTarget(host="prod.example.com", user="ci")
    cmd = Command(natural_language="run full regression")
    run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
    run = run.with_running("pytest --regression", remote_pid=5678)
    progress = Progress(
        percent=percent,
        tests_passed=passed,
        tests_failed=failed,
        tests_total=total,
    )
    run = run.with_progress(progress)
    return run


# ---------------------------------------------------------------------------
# ResumeVerdict dataclass
# ---------------------------------------------------------------------------


class TestResumeVerdict:
    """Verify the ResumeVerdict dataclass is frozen and well-formed."""

    def test_frozen(self) -> None:
        verdict = ResumeVerdict(
            decision=ResumeDecision.RESUME,
            reason="Test resume",
            factors=(),
            checkpoint=_make_checkpoint(),
            run_id="test-id",
        )
        with pytest.raises(AttributeError):
            verdict.decision = ResumeDecision.RESTART  # type: ignore[misc]

    def test_create_resume(self) -> None:
        verdict = ResumeVerdict(
            decision=ResumeDecision.RESUME,
            reason="All heuristics passed",
            factors=(),
            checkpoint=_make_checkpoint(),
            run_id="abc",
        )
        assert verdict.decision == ResumeDecision.RESUME
        assert verdict.should_resume is True
        assert verdict.should_restart is False

    def test_create_restart(self) -> None:
        verdict = ResumeVerdict(
            decision=ResumeDecision.RESTART,
            reason="Too stale",
            factors=(),
            checkpoint=_make_checkpoint(),
            run_id="abc",
        )
        assert verdict.decision == ResumeDecision.RESTART
        assert verdict.should_resume is False
        assert verdict.should_restart is True


class TestResumeDecisionEnum:
    """Verify enum values exist."""

    def test_values(self) -> None:
        assert ResumeDecision.RESUME.value == "resume"
        assert ResumeDecision.RESTART.value == "restart"


class TestResumeThresholds:
    """Verify the thresholds dataclass."""

    def test_defaults(self) -> None:
        t = ResumeThresholds()
        assert t.max_staleness_seconds > 0
        assert 0.0 < t.max_failure_ratio <= 1.0
        assert 0.0 <= t.min_completion_percent <= 100.0
        assert t.min_tests_completed >= 0

    def test_frozen(self) -> None:
        t = ResumeThresholds()
        with pytest.raises(AttributeError):
            t.max_staleness_seconds = 999.0  # type: ignore[misc]

    def test_custom_values(self) -> None:
        t = ResumeThresholds(
            max_staleness_seconds=600.0,
            max_failure_ratio=0.3,
            min_completion_percent=10.0,
            min_tests_completed=5,
        )
        assert t.max_staleness_seconds == 600.0
        assert t.max_failure_ratio == 0.3
        assert t.min_completion_percent == 10.0
        assert t.min_tests_completed == 5


# ---------------------------------------------------------------------------
# Non-resumable checkpoints -> always RESTART
# ---------------------------------------------------------------------------


class TestNonResumableCheckpoint:
    """When checkpoint is not resumable, always decide RESTART."""

    def test_no_state_returns_restart(self) -> None:
        cp = _make_checkpoint(
            source=CheckpointSource.NO_STATE,
            status=RunStatus.IDLE,
            phase=CheckpointPhase.NOT_STARTED,
        )
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision == ResumeDecision.RESTART

    def test_corrupted_returns_restart(self) -> None:
        cp = _make_checkpoint(
            source=CheckpointSource.CORRUPTED,
            status=RunStatus.IDLE,
            phase=CheckpointPhase.NOT_STARTED,
            error="parse error",
        )
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision == ResumeDecision.RESTART

    def test_completed_returns_restart(self) -> None:
        cp = _make_checkpoint(
            status=RunStatus.COMPLETED,
            phase=CheckpointPhase.COMPLETE,
            percent=100.0,
        )
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision == ResumeDecision.RESTART

    def test_failed_returns_restart(self) -> None:
        cp = _make_checkpoint(
            status=RunStatus.FAILED,
            phase=CheckpointPhase.FAILED,
            error="SSH timeout",
        )
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision == ResumeDecision.RESTART

    def test_cancelled_returns_restart(self) -> None:
        cp = _make_checkpoint(
            status=RunStatus.CANCELLED,
            phase=CheckpointPhase.CANCELLED,
        )
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision == ResumeDecision.RESTART

    def test_idle_returns_restart(self) -> None:
        cp = _make_checkpoint(
            status=RunStatus.IDLE,
            phase=CheckpointPhase.NOT_STARTED,
        )
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision == ResumeDecision.RESTART

    def test_reason_mentions_not_resumable(self) -> None:
        cp = _make_checkpoint(
            source=CheckpointSource.NO_STATE,
            status=RunStatus.IDLE,
            phase=CheckpointPhase.NOT_STARTED,
        )
        verdict = decide_resume_or_restart(cp)
        assert "not resumable" in verdict.reason.lower()


# ---------------------------------------------------------------------------
# Staleness heuristic
# ---------------------------------------------------------------------------


class TestStalenessHeuristic:
    """Stale checkpoints (too old) should trigger RESTART."""

    def test_recent_checkpoint_passes(self) -> None:
        """A checkpoint from seconds ago should pass staleness check."""
        cp = _make_checkpoint(checkpoint_at=datetime.now(timezone.utc))
        verdict = decide_resume_or_restart(cp)
        # Should not be RESTART due to staleness alone
        assert verdict.decision == ResumeDecision.RESUME

    def test_stale_checkpoint_triggers_restart(self) -> None:
        """A checkpoint from 2 hours ago should trigger RESTART."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        cp = _make_checkpoint(checkpoint_at=old_time)
        thresholds = ResumeThresholds(max_staleness_seconds=3600.0)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        assert verdict.decision == ResumeDecision.RESTART

    def test_staleness_at_exact_boundary_passes(self) -> None:
        """A checkpoint exactly at the threshold should still pass."""
        now = datetime.now(timezone.utc)
        threshold_seconds = 3600.0
        at_boundary = now - timedelta(seconds=threshold_seconds)
        cp = _make_checkpoint(checkpoint_at=at_boundary)
        thresholds = ResumeThresholds(max_staleness_seconds=threshold_seconds)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds, now=now)
        # At the boundary, we pass (<=, not <)
        assert verdict.decision == ResumeDecision.RESUME

    def test_staleness_just_over_boundary_fails(self) -> None:
        """A checkpoint one second past the threshold should trigger RESTART."""
        now = datetime.now(timezone.utc)
        threshold_seconds = 3600.0
        just_over = now - timedelta(seconds=threshold_seconds + 1.0)
        cp = _make_checkpoint(checkpoint_at=just_over)
        thresholds = ResumeThresholds(max_staleness_seconds=threshold_seconds)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds, now=now)
        assert verdict.decision == ResumeDecision.RESTART

    def test_staleness_reason_mentions_stale(self) -> None:
        old_time = datetime.now(timezone.utc) - timedelta(hours=5)
        cp = _make_checkpoint(checkpoint_at=old_time)
        thresholds = ResumeThresholds(max_staleness_seconds=3600.0)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        assert "stale" in verdict.reason.lower()

    def test_no_checkpoint_at_uses_updated_at_from_run(self) -> None:
        """When checkpoint_at is None, staleness check uses run updated_at."""
        cp = _make_checkpoint(checkpoint_at=None)
        old_run = _make_running_run()
        # A run with recent updated_at should pass staleness
        verdict = decide_resume_or_restart(cp, run=old_run)
        assert verdict.decision == ResumeDecision.RESUME

    def test_custom_staleness_threshold(self) -> None:
        """Custom threshold of 60s triggers restart for 2-minute-old checkpoint."""
        old_time = datetime.now(timezone.utc) - timedelta(minutes=2)
        cp = _make_checkpoint(checkpoint_at=old_time)
        thresholds = ResumeThresholds(max_staleness_seconds=60.0)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        assert verdict.decision == ResumeDecision.RESTART


# ---------------------------------------------------------------------------
# Failure ratio heuristic
# ---------------------------------------------------------------------------


class TestFailureRatioHeuristic:
    """High failure rates should trigger RESTART."""

    def test_low_failure_ratio_passes(self) -> None:
        """1 failure out of 50 tests should pass."""
        cp = _make_checkpoint(tests_failed=1, tests_total=50)
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision == ResumeDecision.RESUME

    def test_high_failure_ratio_triggers_restart(self) -> None:
        """30 failures out of 50 total (60%) should trigger RESTART."""
        cp = _make_checkpoint(
            tests_passed=10,
            tests_failed=30,
            tests_skipped=0,
            tests_total=50,
            test_index=39,  # 10+30-1
        )
        thresholds = ResumeThresholds(max_failure_ratio=0.5)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        assert verdict.decision == ResumeDecision.RESTART

    def test_failure_ratio_at_boundary_passes(self) -> None:
        """Exactly at the threshold ratio should pass."""
        cp = _make_checkpoint(
            tests_passed=5,
            tests_failed=5,
            tests_skipped=0,
            tests_total=20,
            test_index=9,  # 5+5-1
        )
        thresholds = ResumeThresholds(max_failure_ratio=0.5)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        # At boundary (50% == 50%), we pass (<=)
        assert verdict.decision == ResumeDecision.RESUME

    def test_failure_ratio_just_over_boundary_fails(self) -> None:
        """Just over the threshold should trigger RESTART."""
        # 6 failures out of 10 completed = 60% > 50%
        cp = _make_checkpoint(
            tests_passed=4,
            tests_failed=6,
            tests_skipped=0,
            tests_total=20,
            test_index=9,  # 4+6-1
        )
        thresholds = ResumeThresholds(max_failure_ratio=0.5)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        assert verdict.decision == ResumeDecision.RESTART

    def test_zero_completed_bypasses_failure_check(self) -> None:
        """When no tests completed, failure ratio is N/A (not a restart trigger)."""
        cp = _make_checkpoint(
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            tests_total=50,
            test_index=0,
            phase=CheckpointPhase.SETUP,
        )
        # Only the min_tests_completed heuristic would trigger restart, not failure
        verdict = decide_resume_or_restart(cp)
        # Result depends on other factors, but failure ratio should NOT be the reason
        for factor in verdict.factors:
            if factor.name == "failure_ratio":
                assert factor.passed is True

    def test_failure_reason_mentions_failures(self) -> None:
        cp = _make_checkpoint(
            tests_passed=2,
            tests_failed=20,
            tests_skipped=0,
            tests_total=50,
            test_index=21,
        )
        thresholds = ResumeThresholds(max_failure_ratio=0.5)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        assert "failure" in verdict.reason.lower() or "fail" in verdict.reason.lower()


# ---------------------------------------------------------------------------
# Completion heuristic
# ---------------------------------------------------------------------------


class TestCompletionHeuristic:
    """Minimal progress should favor RESTART (cheap to redo)."""

    def test_good_progress_passes(self) -> None:
        """50% done should pass the min completion check."""
        cp = _make_checkpoint(
            percent=50.0,
            tests_passed=25,
            tests_failed=0,
            tests_skipped=0,
            tests_total=50,
            test_index=24,
        )
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision == ResumeDecision.RESUME

    def test_minimal_progress_triggers_restart(self) -> None:
        """Only 1% done should trigger RESTART."""
        cp = _make_checkpoint(
            percent=1.0,
            tests_passed=1,
            tests_failed=0,
            tests_skipped=0,
            tests_total=100,
            test_index=0,
        )
        thresholds = ResumeThresholds(min_completion_percent=5.0)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        assert verdict.decision == ResumeDecision.RESTART

    def test_at_min_completion_passes(self) -> None:
        """Exactly at the min completion threshold should pass."""
        cp = _make_checkpoint(
            percent=5.0,
            tests_passed=5,
            tests_failed=0,
            tests_skipped=0,
            tests_total=100,
            test_index=4,
        )
        thresholds = ResumeThresholds(min_completion_percent=5.0)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        # Check that completion factor passes
        for factor in verdict.factors:
            if factor.name == "completion":
                assert factor.passed is True

    def test_min_tests_completed_check(self) -> None:
        """Must have completed at least N tests to resume."""
        cp = _make_checkpoint(
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            tests_total=50,
            test_index=0,
            percent=0.0,
            phase=CheckpointPhase.SETUP,
        )
        thresholds = ResumeThresholds(min_tests_completed=1)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        assert verdict.decision == ResumeDecision.RESTART

    def test_enough_tests_completed_passes(self) -> None:
        """Having completed enough tests should pass the check."""
        cp = _make_checkpoint(
            tests_passed=5,
            tests_failed=0,
            tests_skipped=0,
            tests_total=50,
            test_index=4,
            percent=10.0,
        )
        thresholds = ResumeThresholds(min_tests_completed=1)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        for factor in verdict.factors:
            if factor.name == "min_tests":
                assert factor.passed is True


# ---------------------------------------------------------------------------
# Pending approval special case
# ---------------------------------------------------------------------------


class TestPendingApprovalResume:
    """PENDING_APPROVAL state should always RESUME (re-prompt user)."""

    def test_pending_always_resumes(self) -> None:
        cp = _make_checkpoint(
            status=RunStatus.PENDING_APPROVAL,
            phase=CheckpointPhase.PENDING_APPROVAL,
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            tests_total=0,
            test_index=0,
            percent=0.0,
        )
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision == ResumeDecision.RESUME

    def test_pending_reason_mentions_approval(self) -> None:
        cp = _make_checkpoint(
            status=RunStatus.PENDING_APPROVAL,
            phase=CheckpointPhase.PENDING_APPROVAL,
            tests_passed=0,
            tests_failed=0,
            tests_total=0,
            test_index=0,
            percent=0.0,
        )
        verdict = decide_resume_or_restart(cp)
        assert "pending" in verdict.reason.lower() or "approval" in verdict.reason.lower()


# ---------------------------------------------------------------------------
# Happy path: all heuristics pass -> RESUME
# ---------------------------------------------------------------------------


class TestHappyPathResume:
    """When all heuristics pass, decide RESUME."""

    def test_healthy_running_checkpoint_resumes(self) -> None:
        """A recent checkpoint with good progress and low failures -> RESUME."""
        cp = _make_checkpoint(
            percent=50.0,
            tests_passed=25,
            tests_failed=1,
            tests_skipped=0,
            tests_total=50,
            test_index=25,
            checkpoint_at=datetime.now(timezone.utc),
        )
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision == ResumeDecision.RESUME

    def test_all_factors_pass(self) -> None:
        """All individual factors should report as passed."""
        cp = _make_checkpoint(
            percent=50.0,
            tests_passed=25,
            tests_failed=1,
            tests_skipped=0,
            tests_total=50,
            test_index=25,
            checkpoint_at=datetime.now(timezone.utc),
        )
        verdict = decide_resume_or_restart(cp)
        for factor in verdict.factors:
            assert factor.passed is True, f"Factor {factor.name} failed: {factor.detail}"

    def test_preserves_checkpoint_reference(self) -> None:
        cp = _make_checkpoint()
        verdict = decide_resume_or_restart(cp)
        assert verdict.checkpoint is cp

    def test_preserves_run_id(self) -> None:
        cp = _make_checkpoint(run_id="specific-run-id")
        verdict = decide_resume_or_restart(cp)
        assert verdict.run_id == "specific-run-id"


# ---------------------------------------------------------------------------
# Multiple failing heuristics
# ---------------------------------------------------------------------------


class TestMultipleFailingHeuristics:
    """When multiple heuristics fail, RESTART with combined reasoning."""

    def test_stale_and_high_failures(self) -> None:
        """Both stale and high failure rate should trigger RESTART."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=3)
        cp = _make_checkpoint(
            tests_passed=5,
            tests_failed=20,
            tests_skipped=0,
            tests_total=50,
            test_index=24,
            percent=50.0,
            checkpoint_at=old_time,
        )
        thresholds = ResumeThresholds(
            max_staleness_seconds=3600.0,
            max_failure_ratio=0.5,
        )
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        assert verdict.decision == ResumeDecision.RESTART

    def test_factors_show_which_failed(self) -> None:
        """Factors should indicate which specific checks failed."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=3)
        cp = _make_checkpoint(
            tests_passed=5,
            tests_failed=20,
            tests_skipped=0,
            tests_total=50,
            test_index=24,
            percent=50.0,
            checkpoint_at=old_time,
        )
        thresholds = ResumeThresholds(
            max_staleness_seconds=3600.0,
            max_failure_ratio=0.5,
        )
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)

        failed_names = {f.name for f in verdict.factors if not f.passed}
        assert "staleness" in failed_names
        assert "failure_ratio" in failed_names


# ---------------------------------------------------------------------------
# Optional CurrentRun parameter
# ---------------------------------------------------------------------------


class TestWithCurrentRun:
    """When a CurrentRun is provided, it supplements checkpoint data."""

    def test_run_updated_at_used_when_no_checkpoint_at(self) -> None:
        """If checkpoint_at is None, fall back to run.updated_at for staleness."""
        cp = _make_checkpoint(checkpoint_at=None)
        run = _make_running_run()
        # The run was just created, so it's fresh
        verdict = decide_resume_or_restart(cp, run=run)
        assert verdict.decision == ResumeDecision.RESUME

    def test_works_without_run(self) -> None:
        """Function works fine without a CurrentRun parameter."""
        cp = _make_checkpoint(checkpoint_at=datetime.now(timezone.utc))
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision in (ResumeDecision.RESUME, ResumeDecision.RESTART)


# ---------------------------------------------------------------------------
# Factor tuple integrity
# ---------------------------------------------------------------------------


class TestFactorIntegrity:
    """Factors should always be present and well-formed."""

    def test_factors_always_present(self) -> None:
        cp = _make_checkpoint()
        verdict = decide_resume_or_restart(cp)
        assert len(verdict.factors) > 0

    def test_factor_names_are_strings(self) -> None:
        cp = _make_checkpoint()
        verdict = decide_resume_or_restart(cp)
        for factor in verdict.factors:
            assert isinstance(factor.name, str)
            assert len(factor.name) > 0

    def test_factor_details_are_strings(self) -> None:
        cp = _make_checkpoint()
        verdict = decide_resume_or_restart(cp)
        for factor in verdict.factors:
            assert isinstance(factor.detail, str)

    def test_expected_factor_names(self) -> None:
        """The verdict should include the standard heuristic factor names."""
        cp = _make_checkpoint()
        verdict = decide_resume_or_restart(cp)
        factor_names = {f.name for f in verdict.factors}
        assert "staleness" in factor_names
        assert "failure_ratio" in factor_names
        assert "completion" in factor_names
        assert "min_tests" in factor_names

    def test_factors_are_frozen(self) -> None:
        cp = _make_checkpoint()
        verdict = decide_resume_or_restart(cp)
        for factor in verdict.factors:
            with pytest.raises(AttributeError):
                factor.passed = not factor.passed  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and edge conditions."""

    def test_zero_total_tests(self) -> None:
        """When tests_total is 0, heuristics handle gracefully."""
        cp = _make_checkpoint(
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            tests_total=0,
            test_index=0,
            percent=0.0,
            phase=CheckpointPhase.SETUP,
        )
        # Should not raise
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision in (ResumeDecision.RESUME, ResumeDecision.RESTART)

    def test_all_tests_failed(self) -> None:
        """When every completed test failed, should restart."""
        cp = _make_checkpoint(
            tests_passed=0,
            tests_failed=10,
            tests_skipped=0,
            tests_total=50,
            test_index=9,
            percent=20.0,
        )
        thresholds = ResumeThresholds(max_failure_ratio=0.5)
        verdict = decide_resume_or_restart(cp, thresholds=thresholds)
        assert verdict.decision == ResumeDecision.RESTART

    def test_all_tests_passed(self) -> None:
        """When all completed tests passed, should resume."""
        cp = _make_checkpoint(
            tests_passed=25,
            tests_failed=0,
            tests_skipped=0,
            tests_total=50,
            test_index=24,
            percent=50.0,
            checkpoint_at=datetime.now(timezone.utc),
        )
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision == ResumeDecision.RESUME

    def test_very_large_test_suite(self) -> None:
        """Handles large test counts without overflow."""
        cp = _make_checkpoint(
            tests_passed=50000,
            tests_failed=100,
            tests_skipped=50,
            tests_total=100000,
            test_index=50149,
            percent=50.15,
            checkpoint_at=datetime.now(timezone.utc),
        )
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision == ResumeDecision.RESUME

    def test_never_raises(self) -> None:
        """Even with odd inputs, function should not raise."""
        # Minimal checkpoint that is technically resumable
        cp = _make_checkpoint(
            status=RunStatus.RUNNING,
            phase=CheckpointPhase.RUNNING,
            source=CheckpointSource.WIKI_STATE,
            tests_passed=0,
            tests_failed=0,
            tests_total=0,
            test_index=0,
            percent=0.0,
            checkpoint_at=None,
        )
        # Should not raise
        verdict = decide_resume_or_restart(cp)
        assert verdict.decision in (ResumeDecision.RESUME, ResumeDecision.RESTART)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestDecisionPerformance:
    """Decision must be fast -- part of the 30s crash recovery SLA."""

    def test_completes_under_10ms(self) -> None:
        import time

        cp = _make_checkpoint(
            percent=75.0,
            tests_passed=150,
            tests_failed=3,
            tests_skipped=5,
            tests_total=200,
            test_index=157,
            checkpoint_at=datetime.now(timezone.utc),
        )
        run = _make_running_run(percent=75.0, passed=150, failed=3, total=200)

        start = time.monotonic()
        verdict = decide_resume_or_restart(cp, run=run)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 10.0, f"Decision took {elapsed_ms:.1f}ms (>10ms)"
        assert verdict.decision in (ResumeDecision.RESUME, ResumeDecision.RESTART)
