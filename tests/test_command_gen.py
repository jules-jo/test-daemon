"""Tests for command generation from resume/restart verdict.

Verifies that the command generator:
- Builds a restart command from the original shell command unmodified
- Builds a resume command with framework-specific resume flags
- Detects pytest and adds --start-at-test or equivalent resume markers
- Detects generic test runners and falls back to restart semantics
- Preserves working directory and environment from the original command
- Returns immutable GeneratedCommand with action, command, and context
- Validates inputs (non-empty shell, valid checkpoint)
- Handles edge cases (empty marker, zero test index, unknown framework)
- Associates the original checkpoint data for audit trail
- Never raises -- returns restart fallback for unrecoverable inputs
"""

from datetime import datetime, timezone

import pytest

from jules_daemon.ssh.command import SSHCommand
from jules_daemon.ssh.command_gen import (
    GeneratedCommand,
    RecoveryCommandAction,
    TestFramework,
    build_recovery_command,
    detect_framework,
)
from jules_daemon.wiki.checkpoint_extractor import (
    Checkpoint,
    CheckpointPhase,
    CheckpointSource,
)
from jules_daemon.wiki.models import RunStatus
from jules_daemon.wiki.resume_decision import (
    ResumeDecision,
    ResumeVerdict,
    ResumeVerdictFactor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(
    *,
    test_index: int = 10,
    tests_passed: int = 8,
    tests_failed: int = 1,
    tests_skipped: int = 2,
    tests_total: int = 50,
    percent: float = 22.0,
    marker: str = "PASSED test_checkout",
    run_id: str = "run-abc-123",
    phase: CheckpointPhase = CheckpointPhase.RUNNING,
    source: CheckpointSource = CheckpointSource.WIKI_STATE,
    status: RunStatus = RunStatus.RUNNING,
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


def _make_verdict(
    *,
    decision: ResumeDecision = ResumeDecision.RESUME,
    reason: str = "All heuristics passed",
    checkpoint: Checkpoint | None = None,
    run_id: str = "run-abc-123",
) -> ResumeVerdict:
    """Create a ResumeVerdict with defaults."""
    cp = checkpoint or _make_checkpoint()
    return ResumeVerdict(
        decision=decision,
        reason=reason,
        factors=(
            ResumeVerdictFactor(name="staleness", passed=True, detail="ok"),
        ),
        checkpoint=cp,
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# RecoveryCommandAction enum
# ---------------------------------------------------------------------------


class TestRecoveryCommandAction:
    def test_values_exist(self) -> None:
        assert RecoveryCommandAction.RESUME.value == "resume"
        assert RecoveryCommandAction.RESTART.value == "restart"


# ---------------------------------------------------------------------------
# TestFramework enum
# ---------------------------------------------------------------------------


class TestTestFramework:
    def test_values_exist(self) -> None:
        assert TestFramework.PYTEST.value == "pytest"
        assert TestFramework.NPM_TEST.value == "npm_test"
        assert TestFramework.CARGO_TEST.value == "cargo_test"
        assert TestFramework.GO_TEST.value == "go_test"
        assert TestFramework.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# GeneratedCommand frozen dataclass
# ---------------------------------------------------------------------------


class TestGeneratedCommand:
    def test_frozen(self) -> None:
        cmd = GeneratedCommand(
            action=RecoveryCommandAction.RESTART,
            ssh_command=SSHCommand(command="pytest"),
            original_shell="pytest",
            resume_context="Restarting full suite",
            checkpoint_marker="",
            test_index=0,
            framework=TestFramework.PYTEST,
            run_id="abc",
        )
        with pytest.raises(AttributeError):
            cmd.action = RecoveryCommandAction.RESUME  # type: ignore[misc]

    def test_has_all_fields(self) -> None:
        cmd = GeneratedCommand(
            action=RecoveryCommandAction.RESUME,
            ssh_command=SSHCommand(command="pytest --lf"),
            original_shell="pytest",
            resume_context="Resuming from test_checkout",
            checkpoint_marker="PASSED test_checkout",
            test_index=10,
            framework=TestFramework.PYTEST,
            run_id="run-xyz",
        )
        assert cmd.action == RecoveryCommandAction.RESUME
        assert cmd.ssh_command.command == "pytest --lf"
        assert cmd.original_shell == "pytest"
        assert cmd.resume_context == "Resuming from test_checkout"
        assert cmd.checkpoint_marker == "PASSED test_checkout"
        assert cmd.test_index == 10
        assert cmd.framework == TestFramework.PYTEST
        assert cmd.run_id == "run-xyz"

    def test_is_resume_property(self) -> None:
        cmd = GeneratedCommand(
            action=RecoveryCommandAction.RESUME,
            ssh_command=SSHCommand(command="pytest --lf"),
            original_shell="pytest",
            resume_context="",
            checkpoint_marker="",
            test_index=0,
            framework=TestFramework.PYTEST,
            run_id="abc",
        )
        assert cmd.is_resume is True
        assert cmd.is_restart is False

    def test_is_restart_property(self) -> None:
        cmd = GeneratedCommand(
            action=RecoveryCommandAction.RESTART,
            ssh_command=SSHCommand(command="pytest"),
            original_shell="pytest",
            resume_context="",
            checkpoint_marker="",
            test_index=0,
            framework=TestFramework.PYTEST,
            run_id="abc",
        )
        assert cmd.is_resume is False
        assert cmd.is_restart is True


# ---------------------------------------------------------------------------
# detect_framework
# ---------------------------------------------------------------------------


class TestDetectFramework:
    def test_detects_pytest(self) -> None:
        assert detect_framework("pytest -v --tb=short") == TestFramework.PYTEST

    def test_detects_pytest_with_cd(self) -> None:
        assert detect_framework("cd /app && pytest") == TestFramework.PYTEST

    def test_detects_python_m_pytest(self) -> None:
        assert detect_framework("python -m pytest") == TestFramework.PYTEST

    def test_detects_npm_test(self) -> None:
        assert detect_framework("npm test") == TestFramework.NPM_TEST

    def test_detects_npm_run_test(self) -> None:
        assert detect_framework("npm run test") == TestFramework.NPM_TEST

    def test_detects_npx_jest(self) -> None:
        assert detect_framework("npx jest") == TestFramework.NPM_TEST

    def test_detects_cargo_test(self) -> None:
        assert detect_framework("cargo test") == TestFramework.CARGO_TEST

    def test_detects_cargo_test_with_flags(self) -> None:
        assert detect_framework("cargo test --release") == TestFramework.CARGO_TEST

    def test_detects_go_test(self) -> None:
        assert detect_framework("go test ./...") == TestFramework.GO_TEST

    def test_detects_go_test_with_flags(self) -> None:
        assert detect_framework("go test -v -count=1 ./...") == TestFramework.GO_TEST

    def test_unknown_framework(self) -> None:
        assert detect_framework("make check") == TestFramework.UNKNOWN

    def test_empty_string_is_unknown(self) -> None:
        assert detect_framework("") == TestFramework.UNKNOWN

    def test_case_insensitive_detection(self) -> None:
        # command strings are usually lowercase, but be robust
        assert detect_framework("PYTEST -v") == TestFramework.PYTEST


# ---------------------------------------------------------------------------
# build_recovery_command: RESTART decision
# ---------------------------------------------------------------------------


class TestBuildRestartCommand:
    """When verdict is RESTART, the original command is used unmodified."""

    def test_restart_returns_original_command(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESTART)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest -v --tb=short",
        )
        assert result.action == RecoveryCommandAction.RESTART
        assert result.ssh_command.command == "pytest -v --tb=short"

    def test_restart_preserves_working_directory(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESTART)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest",
            working_directory="/opt/app",
        )
        assert result.ssh_command.working_directory == "/opt/app"

    def test_restart_preserves_environment(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESTART)
        env = {"CI": "true", "NODE_ENV": "test"}
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest",
            environment=env,
        )
        assert result.ssh_command.environment == env

    def test_restart_preserves_timeout(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESTART)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest",
            timeout=3600,
        )
        assert result.ssh_command.timeout == 3600

    def test_restart_context_mentions_restart(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESTART)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest",
        )
        assert "restart" in result.resume_context.lower()

    def test_restart_records_original_shell(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESTART)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest -x --tb=long",
        )
        assert result.original_shell == "pytest -x --tb=long"

    def test_restart_preserves_run_id(self) -> None:
        verdict = _make_verdict(
            decision=ResumeDecision.RESTART,
            run_id="specific-id",
        )
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest",
        )
        assert result.run_id == "specific-id"

    def test_restart_test_index_is_zero(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESTART)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest",
        )
        assert result.test_index == 0


# ---------------------------------------------------------------------------
# build_recovery_command: RESUME decision with pytest
# ---------------------------------------------------------------------------


class TestBuildResumePytest:
    """When verdict is RESUME and framework is pytest, add resume flags."""

    def test_resume_adds_last_failed_flag(self) -> None:
        """Pytest resume uses --lf (last failed) to re-run failures."""
        cp = _make_checkpoint(
            tests_failed=3,
            tests_passed=20,
            marker="FAILED test_payment",
        )
        verdict = _make_verdict(decision=ResumeDecision.RESUME, checkpoint=cp)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest -v --tb=short",
        )
        assert result.action == RecoveryCommandAction.RESUME
        assert "--lf" in result.ssh_command.command

    def test_resume_no_failures_uses_start_after(self) -> None:
        """When no failures, use nodeids or deselect to skip completed tests."""
        cp = _make_checkpoint(
            tests_failed=0,
            tests_passed=25,
            tests_skipped=0,
            tests_total=50,
            test_index=24,
            marker="PASSED test_last_passed",
        )
        verdict = _make_verdict(decision=ResumeDecision.RESUME, checkpoint=cp)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest -v",
        )
        assert result.action == RecoveryCommandAction.RESUME
        # Should use a pytest-specific mechanism to skip completed tests
        assert "pytest" in result.ssh_command.command

    def test_resume_preserves_working_directory(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESUME)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest -v",
            working_directory="/opt/app",
        )
        assert result.ssh_command.working_directory == "/opt/app"

    def test_resume_context_mentions_resume(self) -> None:
        cp = _make_checkpoint(marker="PASSED test_checkout")
        verdict = _make_verdict(decision=ResumeDecision.RESUME, checkpoint=cp)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest",
        )
        assert "resum" in result.resume_context.lower()

    def test_resume_records_checkpoint_marker(self) -> None:
        cp = _make_checkpoint(marker="FAILED test_payment_flow")
        verdict = _make_verdict(decision=ResumeDecision.RESUME, checkpoint=cp)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest",
        )
        assert result.checkpoint_marker == "FAILED test_payment_flow"

    def test_resume_records_test_index(self) -> None:
        cp = _make_checkpoint(test_index=42)
        verdict = _make_verdict(decision=ResumeDecision.RESUME, checkpoint=cp)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest",
        )
        assert result.test_index == 42

    def test_resume_detects_pytest_framework(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESUME)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest -v --tb=short",
        )
        assert result.framework == TestFramework.PYTEST


# ---------------------------------------------------------------------------
# build_recovery_command: RESUME decision with unknown framework
# ---------------------------------------------------------------------------


class TestBuildResumeUnknownFramework:
    """For unknown test frameworks, RESUME falls back to running original cmd."""

    def test_unknown_framework_uses_original_command(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESUME)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="make test",
        )
        assert result.ssh_command.command == "make test"
        assert result.framework == TestFramework.UNKNOWN

    def test_unknown_framework_context_mentions_fallback(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESUME)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="./run-tests.sh",
        )
        assert (
            "restart" in result.resume_context.lower()
            or "fallback" in result.resume_context.lower()
            or "unknown" in result.resume_context.lower()
        )


# ---------------------------------------------------------------------------
# build_recovery_command: RESUME with npm/cargo/go
# ---------------------------------------------------------------------------


class TestBuildResumeOtherFrameworks:
    def test_npm_test_uses_original_command(self) -> None:
        """npm test does not have a standard resume mechanism."""
        verdict = _make_verdict(decision=ResumeDecision.RESUME)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="npm test",
        )
        assert result.framework == TestFramework.NPM_TEST
        assert "npm" in result.ssh_command.command

    def test_cargo_test_uses_original_command(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESUME)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="cargo test",
        )
        assert result.framework == TestFramework.CARGO_TEST
        assert "cargo test" in result.ssh_command.command

    def test_go_test_uses_original_command(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESUME)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="go test ./...",
        )
        assert result.framework == TestFramework.GO_TEST
        assert "go test" in result.ssh_command.command


# ---------------------------------------------------------------------------
# build_recovery_command: input validation
# ---------------------------------------------------------------------------


class TestBuildRecoveryCommandValidation:
    def test_empty_shell_raises(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESTART)
        with pytest.raises(ValueError, match="original_shell"):
            build_recovery_command(
                verdict=verdict,
                original_shell="",
            )

    def test_whitespace_only_shell_raises(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESTART)
        with pytest.raises(ValueError, match="original_shell"):
            build_recovery_command(
                verdict=verdict,
                original_shell="   ",
            )


# ---------------------------------------------------------------------------
# build_recovery_command: edge cases
# ---------------------------------------------------------------------------


class TestBuildRecoveryCommandEdgeCases:
    def test_empty_marker_does_not_crash(self) -> None:
        cp = _make_checkpoint(marker="")
        verdict = _make_verdict(decision=ResumeDecision.RESUME, checkpoint=cp)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest",
        )
        assert result.checkpoint_marker == ""

    def test_zero_test_index(self) -> None:
        cp = _make_checkpoint(test_index=0, tests_passed=0, tests_failed=0)
        verdict = _make_verdict(decision=ResumeDecision.RESUME, checkpoint=cp)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest",
        )
        # With no progress, should effectively restart
        assert result.ssh_command.command is not None

    def test_pending_approval_checkpoint(self) -> None:
        """PENDING_APPROVAL with RESUME verdict uses original command."""
        cp = _make_checkpoint(
            status=RunStatus.PENDING_APPROVAL,
            phase=CheckpointPhase.PENDING_APPROVAL,
            tests_passed=0,
            tests_failed=0,
            test_index=0,
        )
        verdict = _make_verdict(decision=ResumeDecision.RESUME, checkpoint=cp)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest -v",
        )
        # For pending approval, the original command is used since no tests ran yet
        assert "pytest" in result.ssh_command.command

    def test_very_long_command(self) -> None:
        """Long but valid command should not be truncated."""
        long_cmd = "pytest " + " ".join(f"--opt{i}=val{i}" for i in range(50))
        verdict = _make_verdict(decision=ResumeDecision.RESTART)
        result = build_recovery_command(
            verdict=verdict,
            original_shell=long_cmd,
        )
        assert result.ssh_command.command == long_cmd

    def test_default_timeout_applied(self) -> None:
        verdict = _make_verdict(decision=ResumeDecision.RESTART)
        result = build_recovery_command(
            verdict=verdict,
            original_shell="pytest",
        )
        assert result.ssh_command.timeout > 0
