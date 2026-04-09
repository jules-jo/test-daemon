"""Command generation from resume/restart verdict.

Takes a ResumeVerdict (from resume_decision) and the original shell
command, then builds the appropriate SSH command string for either
resuming a partially completed test run or restarting the full suite.

Framework detection:
    The module inspects the original shell command to identify the test
    framework in use, then applies framework-specific resume strategies:

    - **pytest**: Uses ``--lf`` (last failed) when failures exist, or
      re-runs the original command with a ``--co -q`` deselect approach
      when all completed tests passed.
    - **npm test / jest**: No standard resume mechanism; falls back to
      running the original command.
    - **cargo test**: No standard resume mechanism; falls back to
      running the original command.
    - **go test**: No standard resume mechanism; falls back to running
      the original command.
    - **unknown**: Falls back to running the original command (effective
      restart semantics even when verdict is RESUME).

Design choices:
    - Framework detection is regex-based and conservative: only matches
      clear invocations. Ambiguous commands default to UNKNOWN.
    - Resume strategies never modify the command in ways that could
      change test semantics (no deselecting or skipping). The ``--lf``
      flag is the safest pytest-specific mechanism.
    - All results are immutable frozen dataclasses.
    - The function never raises for valid inputs. Invalid inputs
      (empty shell string) raise ValueError immediately.

Usage:
    from jules_daemon.ssh.command_gen import build_recovery_command
    from jules_daemon.wiki.resume_decision import decide_resume_or_restart

    verdict = decide_resume_or_restart(checkpoint)
    gen_cmd = build_recovery_command(
        verdict=verdict,
        original_shell="pytest -v --tb=short",
        working_directory="/opt/app",
    )
    # gen_cmd.ssh_command is ready for dispatch
    # gen_cmd.action tells you if this is RESUME or RESTART
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from jules_daemon.ssh.command import SSHCommand, DEFAULT_TIMEOUT
from jules_daemon.wiki.resume_decision import ResumeDecision, ResumeVerdict

__all__ = [
    "GeneratedCommand",
    "RecoveryCommandAction",
    "TestFramework",
    "build_recovery_command",
    "detect_framework",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RecoveryCommandAction(Enum):
    """Whether the generated command is a resume or restart."""

    RESUME = "resume"
    RESTART = "restart"


class TestFramework(Enum):
    """Detected test framework from the shell command string."""

    PYTEST = "pytest"
    NPM_TEST = "npm_test"
    CARGO_TEST = "cargo_test"
    GO_TEST = "go_test"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedCommand:
    """Immutable result of command generation from a resume/restart verdict.

    Contains the SSH command ready for dispatch, metadata about the
    action taken, and references to the checkpoint data for auditing.

    Attributes:
        action: Whether this is a RESUME or RESTART command.
        ssh_command: The validated SSHCommand ready for dispatch.
        original_shell: The original shell command before any modification.
        resume_context: Human-readable description of the action.
        checkpoint_marker: The checkpoint marker from the verdict (for audit).
        test_index: The test index from the checkpoint (for audit).
        framework: The detected test framework.
        run_id: The run identifier from the verdict.
    """

    action: RecoveryCommandAction
    ssh_command: SSHCommand
    original_shell: str
    resume_context: str
    checkpoint_marker: str
    test_index: int
    framework: TestFramework
    run_id: str

    @property
    def is_resume(self) -> bool:
        """True if the command is a resume action."""
        return self.action == RecoveryCommandAction.RESUME

    @property
    def is_restart(self) -> bool:
        """True if the command is a restart action."""
        return self.action == RecoveryCommandAction.RESTART


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------

# Compiled patterns for framework detection. Each pattern matches the
# test runner invocation in a shell command string.
_PYTEST_PATTERN = re.compile(
    r"(?:^|\s|&&\s*|;\s*)"      # start of string, whitespace, or chain
    r"(?:python\s+-m\s+)?"      # optional "python -m"
    r"pytest\b",                 # the pytest command itself
    re.IGNORECASE,
)

_NPM_TEST_PATTERN = re.compile(
    r"(?:^|\s|&&\s*|;\s*)"
    r"(?:npm\s+(?:test|run\s+test)|npx\s+jest)\b",
    re.IGNORECASE,
)

_CARGO_TEST_PATTERN = re.compile(
    r"(?:^|\s|&&\s*|;\s*)"
    r"cargo\s+test\b",
    re.IGNORECASE,
)

_GO_TEST_PATTERN = re.compile(
    r"(?:^|\s|&&\s*|;\s*)"
    r"go\s+test\b",
    re.IGNORECASE,
)


def detect_framework(shell_command: str) -> TestFramework:
    """Detect the test framework from a shell command string.

    Inspects the command using regex patterns to identify the test
    runner. Returns UNKNOWN if no recognized framework is detected.

    The detection order is deterministic: pytest, npm/jest, cargo, go.
    The first match wins.

    Args:
        shell_command: The shell command string to analyze.

    Returns:
        The detected TestFramework enum value.
    """
    if not shell_command.strip():
        return TestFramework.UNKNOWN

    if _PYTEST_PATTERN.search(shell_command):
        return TestFramework.PYTEST

    if _NPM_TEST_PATTERN.search(shell_command):
        return TestFramework.NPM_TEST

    if _CARGO_TEST_PATTERN.search(shell_command):
        return TestFramework.CARGO_TEST

    if _GO_TEST_PATTERN.search(shell_command):
        return TestFramework.GO_TEST

    return TestFramework.UNKNOWN


# ---------------------------------------------------------------------------
# Internal: framework-specific resume command builders
# ---------------------------------------------------------------------------


def _build_pytest_resume(
    original_shell: str,
    verdict: ResumeVerdict,
) -> str:
    """Build a pytest resume command based on checkpoint state.

    Strategy:
    - If there are failed tests: use ``--lf`` (last-failed) to re-run
      only the tests that failed in the previous session.
    - If no failures but progress exists: use ``--lf`` as well since
      pytest tracks the "last failed" set; if empty, it re-runs all.
      This is safe and idempotent.
    - If no progress at all (pending approval, setup phase): use the
      original command unmodified.

    The ``--lf`` flag is preferred because:
    1. It is built into pytest (no plugins needed)
    2. It is safe: only re-runs previously failed tests
    3. It requires the .pytest_cache directory on the remote host,
       which pytest creates by default

    Args:
        original_shell: The original pytest command string.
        verdict: The resume verdict with checkpoint data.

    Returns:
        Modified shell command string for resuming.
    """
    checkpoint = verdict.checkpoint

    # No progress yet: use original command as-is
    if checkpoint.tests_completed == 0:
        return original_shell

    # Has failed tests: use --lf to re-run failures
    if checkpoint.tests_failed > 0:
        # Check if --lf is already present
        if "--lf" in original_shell or "--last-failed" in original_shell:
            return original_shell
        return f"{original_shell} --lf"

    # All passed so far, no failures: use --lf (will re-run nothing
    # if cache says nothing failed, effectively running remaining).
    # This is a safe default. For more sophisticated deselection,
    # the LLM agent layer would need to generate specific node IDs.
    if "--lf" in original_shell or "--last-failed" in original_shell:
        return original_shell
    return f"{original_shell} --lf"


def _build_generic_resume(original_shell: str) -> str:
    """Build a resume command for frameworks without resume support.

    Falls back to the original command unmodified. This means a
    "resume" for unsupported frameworks is effectively a restart.

    Args:
        original_shell: The original test command string.

    Returns:
        The original command unchanged.
    """
    return original_shell


# ---------------------------------------------------------------------------
# Internal: build context string
# ---------------------------------------------------------------------------


def _build_restart_context(verdict: ResumeVerdict, framework: TestFramework) -> str:
    """Build a human-readable context string for a restart action."""
    cp = verdict.checkpoint
    return (
        f"Restarting full test suite ({framework.value}). "
        f"Reason: {verdict.reason}. "
        f"Prior progress: {cp.percent:.1f}% "
        f"({cp.tests_passed}p/{cp.tests_failed}f/{cp.tests_skipped}s)"
    )


def _build_resume_context(
    verdict: ResumeVerdict,
    framework: TestFramework,
    effective_cmd: str,
) -> str:
    """Build a human-readable context string for a resume action."""
    cp = verdict.checkpoint

    if framework == TestFramework.UNKNOWN:
        return (
            f"Resuming with unknown framework fallback (using original command). "
            f"Progress: {cp.percent:.1f}% at test index {cp.test_index}. "
            f"Marker: {cp.marker or '(none)'}"
        )

    return (
        f"Resuming {framework.value} run from checkpoint. "
        f"Progress: {cp.percent:.1f}% at test index {cp.test_index}. "
        f"Marker: {cp.marker or '(none)'}. "
        f"Command: {effective_cmd}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_recovery_command(
    *,
    verdict: ResumeVerdict,
    original_shell: str,
    working_directory: str | None = None,
    environment: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> GeneratedCommand:
    """Build the appropriate SSH command from a resume/restart verdict.

    For RESTART verdicts: the original shell command is used unmodified,
    with progress counters reset to zero semantically.

    For RESUME verdicts: the command is modified with framework-specific
    resume flags (e.g., ``--lf`` for pytest). Frameworks without resume
    support fall back to the original command.

    This function never raises for valid inputs. Invalid inputs (empty
    shell string) raise ValueError immediately.

    Args:
        verdict: The resume/restart verdict from decide_resume_or_restart().
        original_shell: The original approved shell command string.
        working_directory: Optional remote working directory (absolute path).
        environment: Optional environment variable overrides.
        timeout: Command execution timeout in seconds.

    Returns:
        GeneratedCommand with the SSH command ready for dispatch.

    Raises:
        ValueError: If original_shell is empty or whitespace-only.
    """
    stripped_shell = original_shell.strip()
    if not stripped_shell:
        raise ValueError(
            "original_shell must not be empty or whitespace-only"
        )

    framework = detect_framework(stripped_shell)
    checkpoint = verdict.checkpoint

    if verdict.decision == ResumeDecision.RESTART:
        effective_cmd = stripped_shell
        action = RecoveryCommandAction.RESTART
        context = _build_restart_context(verdict, framework)
        # Restart resets the test index to 0
        effective_test_index = 0
    else:
        # RESUME path: apply framework-specific modifications
        action = RecoveryCommandAction.RESUME
        effective_test_index = checkpoint.test_index

        if framework == TestFramework.PYTEST:
            effective_cmd = _build_pytest_resume(stripped_shell, verdict)
        else:
            # All other frameworks: fall back to original command
            effective_cmd = _build_generic_resume(stripped_shell)

        context = _build_resume_context(verdict, framework, effective_cmd)

    ssh_command = SSHCommand(
        command=effective_cmd,
        working_directory=working_directory,
        timeout=timeout,
        environment=environment or {},
    )

    result = GeneratedCommand(
        action=action,
        ssh_command=ssh_command,
        original_shell=stripped_shell,
        resume_context=context,
        checkpoint_marker=checkpoint.marker,
        test_index=effective_test_index,
        framework=framework,
        run_id=verdict.run_id,
    )

    logger.info(
        "Generated recovery command: action=%s framework=%s run_id=%s "
        "cmd=%r test_index=%d marker=%r",
        result.action.value,
        result.framework.value,
        result.run_id,
        effective_cmd[:80],
        effective_test_index,
        checkpoint.marker[:50] if checkpoint.marker else "",
    )

    return result
