"""Demo scenario fixtures for the agent loop acceptance criteria.

Each scenario bundles:
    - Natural-language input phrase(s)
    - Expected intermediate outputs at each pipeline stage
    - Tool call sequences the LLM would produce

Three demos are covered:
    Demo 1: Agent runs a named test end-to-end
    Demo 2: Agent self-corrects a failed command
    Demo 3: Agent asks for missing args via wiki test catalog

All data structures are frozen dataclasses to match the project-wide
immutability convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Frozen data structures for scenario definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedSpecLookup:
    """Expected output from lookup_test_spec for a given test name.

    Attributes:
        found: Whether the spec exists in the wiki.
        test_slug: Derived slug for the test command.
        command_pattern: Canonical command from the wiki entry.
        purpose: One-line purpose description.
        required_args: Arguments the test requires.
        common_failures: Known failure patterns.
        runs_observed: Number of historical runs.
    """

    found: bool
    test_slug: str
    command_pattern: str = ""
    purpose: str = ""
    required_args: tuple[str, ...] = ()
    common_failures: tuple[str, ...] = ()
    runs_observed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON dict format returned by lookup_test_spec."""
        if not self.found:
            return {
                "found": False,
                "test_slug": self.test_slug,
                "message": (
                    f"No test specification found for "
                    f"'{self.command_pattern}' (slug: {self.test_slug})"
                ),
            }
        return {
            "found": True,
            "test_slug": self.test_slug,
            "command_pattern": self.command_pattern,
            "purpose": self.purpose,
            "output_format": "",
            "common_failures": list(self.common_failures),
            "normal_behavior": "",
            "required_args": list(self.required_args),
            "runs_observed": self.runs_observed,
        }


@dataclass(frozen=True)
class ExpectedMissingArgs:
    """Expected missing-argument detection for a given NL input.

    Attributes:
        required_args: Arguments declared in the wiki spec.
        provided_args: Arguments the user supplied in the NL input.
        missing_args: Arguments still needed (required - provided).
    """

    required_args: tuple[str, ...]
    provided_args: tuple[str, ...]
    missing_args: tuple[str, ...]


@dataclass(frozen=True)
class ExpectedSSHCommand:
    """Expected SSH command proposed by the agent.

    Attributes:
        command: The full shell command string.
        working_directory: Remote working directory (absolute path or None).
        timeout: Execution timeout in seconds.
        description: Human-readable description of what the command does.
    """

    command: str
    working_directory: str | None = None
    timeout: int = 300
    description: str = ""


@dataclass(frozen=True)
class ExpectedSummaryTemplate:
    """Expected summary structure after test execution.

    Attributes:
        test_slug: Slug identifying the test.
        exit_code: Expected exit code from the command.
        passed: Number of passing tests (from parsed output).
        failed: Number of failing tests.
        skipped: Number of skipped tests.
        duration_hint: Expected duration description.
    """

    test_slug: str
    exit_code: int
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_hint: str = ""


@dataclass(frozen=True)
class DemoScenario:
    """Complete demo scenario with all intermediate expectations.

    Represents the full agent loop flow from NL input to completion,
    with expected outputs at each stage for test assertions.

    Attributes:
        name: Human-readable scenario name (e.g., "Demo 1: Named test").
        nl_inputs: Tuple of NL phrases that should trigger this scenario.
            The first is the canonical phrase; others are variations.
        expected_spec: Expected output from lookup_test_spec.
        expected_missing_args: Missing-argument detection result.
        expected_ssh_command: The SSH command the agent should propose.
        expected_summary: Summary structure after execution.
        expected_tool_sequence: Ordered list of tool names the agent
            should call (for validating the think-act cycle flow).
    """

    name: str
    nl_inputs: tuple[str, ...]
    expected_spec: ExpectedSpecLookup
    expected_missing_args: ExpectedMissingArgs
    expected_ssh_command: ExpectedSSHCommand
    expected_summary: ExpectedSummaryTemplate
    expected_tool_sequence: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Demo 1: Agent runs a named test end-to-end
# ---------------------------------------------------------------------------

DEMO_1_NL_INPUTS: tuple[str, ...] = (
    "run agent_test with 100 iterations on staging",
    "execute the agent test, 100 iterations, staging host",
    "run agent_test.py iterations=100 host=staging",
    "please run the agent stress test with 100 iterations on staging",
    "kick off agent_test on staging with 100 iterations",
)

DEMO_1_EXPECTED_SPEC = ExpectedSpecLookup(
    found=True,
    test_slug="agent-test-py",
    command_pattern="python3 ~/agent_test.py",
    purpose=(
        "Runs the agent loop stress test with configurable iterations and "
        "concurrency. Verifies that the agent can maintain state across "
        "multiple think-act cycles under load."
    ),
    required_args=("iterations", "host"),
    common_failures=(
        "timeout on large iteration counts (>500)",
        "connection refused when SSH agent is not forwarded",
        "ImportError: missing dependency on fresh hosts",
    ),
    runs_observed=42,
)

DEMO_1_EXPECTED_MISSING_ARGS = ExpectedMissingArgs(
    required_args=("iterations", "host"),
    provided_args=("iterations", "host"),
    missing_args=(),
)

DEMO_1_EXPECTED_SSH_COMMAND = ExpectedSSHCommand(
    command="python3 ~/agent_test.py --iterations 100 --host staging",
    working_directory=None,
    timeout=300,
    description=(
        "Run the agent stress test with 100 iterations targeting the staging host"
    ),
)

DEMO_1_EXPECTED_SUMMARY = ExpectedSummaryTemplate(
    test_slug="agent-test-py",
    exit_code=0,
    passed=100,
    failed=0,
    skipped=0,
    duration_hint="45-90 seconds typical for 100 iterations",
)

DEMO_1_EXPECTED_TOOL_SEQUENCE: tuple[str, ...] = (
    "lookup_test_spec",
    "propose_ssh_command",
    "execute_ssh",
    "read_output",
    "parse_test_output",
    "summarize_run",
)

DEMO_1_SCENARIO = DemoScenario(
    name="Demo 1: Agent runs a named test end-to-end",
    nl_inputs=DEMO_1_NL_INPUTS,
    expected_spec=DEMO_1_EXPECTED_SPEC,
    expected_missing_args=DEMO_1_EXPECTED_MISSING_ARGS,
    expected_ssh_command=DEMO_1_EXPECTED_SSH_COMMAND,
    expected_summary=DEMO_1_EXPECTED_SUMMARY,
    expected_tool_sequence=DEMO_1_EXPECTED_TOOL_SEQUENCE,
)


# ---------------------------------------------------------------------------
# Demo 2: Agent self-corrects a failed command
# ---------------------------------------------------------------------------

DEMO_2_NL_INPUTS: tuple[str, ...] = (
    "run the integration tests",
    "execute integration test suite",
    "run pytest integration tests with verbose output",
)

DEMO_2_EXPECTED_SPEC = ExpectedSpecLookup(
    found=True,
    test_slug="pytest-tests-integration",
    command_pattern="pytest tests/integration/ -v --tb=short",
    purpose=(
        "Runs the integration test suite against a live database. Covers "
        "API endpoints, message queue consumers, and cache invalidation."
    ),
    required_args=(),
    common_failures=(
        "FAILED tests/integration/test_api.py::test_health - ConnectionError",
        'fixture "db_session" not found (missing conftest on remote)',
    ),
    runs_observed=18,
)

DEMO_2_EXPECTED_MISSING_ARGS = ExpectedMissingArgs(
    required_args=(),
    provided_args=(),
    missing_args=(),
)

DEMO_2_FIRST_SSH_COMMAND = ExpectedSSHCommand(
    command="cd /opt/app && pytest tests/integration/ -v --tb=short",
    working_directory="/opt/app",
    timeout=300,
    description="Run the integration test suite with verbose output",
)

DEMO_2_CORRECTED_SSH_COMMAND = ExpectedSSHCommand(
    command=(
        "cd /opt/app && pip install -r requirements-test.txt "
        "&& pytest tests/integration/ -v --tb=short"
    ),
    working_directory="/opt/app",
    timeout=600,
    description=(
        "Install test dependencies first, then retry the integration test suite"
    ),
)

DEMO_2_FIRST_FAILURE_OUTPUT: str = (
    "FAILED tests/integration/test_api.py::test_health - ConnectionError: "
    "[Errno 111] Connection refused\n"
    "1 failed, 0 passed in 2.34s"
)

DEMO_2_CORRECTED_SUCCESS_OUTPUT: str = (
    "tests/integration/test_api.py::test_health PASSED\n"
    "tests/integration/test_api.py::test_create PASSED\n"
    "tests/integration/test_api.py::test_list PASSED\n"
    "3 passed in 12.56s"
)

DEMO_2_EXPECTED_SUMMARY = ExpectedSummaryTemplate(
    test_slug="pytest-tests-integration",
    exit_code=0,
    passed=3,
    failed=0,
    skipped=0,
    duration_hint="12-15 seconds after correction",
)

DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL: tuple[str, ...] = (
    # First attempt
    "lookup_test_spec",
    "propose_ssh_command",
    "execute_ssh",
    "read_output",
    "parse_test_output",
    # Self-correction cycle
    "propose_ssh_command",
    "execute_ssh",
    "read_output",
    "parse_test_output",
    "summarize_run",
)


# ---------------------------------------------------------------------------
# Demo 3: Agent asks for missing args via wiki test catalog
# ---------------------------------------------------------------------------

DEMO_3_NL_INPUTS: tuple[str, ...] = (
    "run agent_test",
    "execute the agent test",
    "run agent_test.py",
    "kick off the agent stress test",
)

DEMO_3_EXPECTED_SPEC = ExpectedSpecLookup(
    found=True,
    test_slug="agent-test-py",
    command_pattern="python3 ~/agent_test.py",
    purpose=(
        "Runs the agent loop stress test with configurable iterations and "
        "concurrency. Verifies that the agent can maintain state across "
        "multiple think-act cycles under load."
    ),
    required_args=("iterations", "host"),
    common_failures=(
        "timeout on large iteration counts (>500)",
        "connection refused when SSH agent is not forwarded",
        "ImportError: missing dependency on fresh hosts",
    ),
    runs_observed=42,
)

DEMO_3_EXPECTED_MISSING_ARGS = ExpectedMissingArgs(
    required_args=("iterations", "host"),
    provided_args=(),
    missing_args=("iterations", "host"),
)

DEMO_3_USER_RESPONSES: dict[str, str] = {
    "iterations": "50",
    "host": "staging.example.com",
}

DEMO_3_EXPECTED_SSH_COMMAND = ExpectedSSHCommand(
    command="python3 ~/agent_test.py --iterations 50 --host staging.example.com",
    working_directory=None,
    timeout=300,
    description=(
        "Run the agent stress test with 50 iterations targeting staging.example.com"
    ),
)

DEMO_3_EXPECTED_SUMMARY = ExpectedSummaryTemplate(
    test_slug="agent-test-py",
    exit_code=0,
    passed=50,
    failed=0,
    skipped=0,
    duration_hint="25-45 seconds typical for 50 iterations",
)

DEMO_3_EXPECTED_TOOL_SEQUENCE: tuple[str, ...] = (
    "lookup_test_spec",
    "ask_user_question",   # ask for "iterations"
    "ask_user_question",   # ask for "host"
    "propose_ssh_command",
    "execute_ssh",
    "read_output",
    "parse_test_output",
    "summarize_run",
)


# ---------------------------------------------------------------------------
# Test output samples for parse_test_output validation
# ---------------------------------------------------------------------------

AGENT_TEST_SUCCESS_OUTPUT: str = (
    "Iteration 1/100 ... OK\n"
    "Iteration 2/100 ... OK\n"
    "Iteration 3/100 ... OK\n"
    "...\n"
    "Iteration 99/100 ... OK\n"
    "Iteration 100/100 ... OK\n"
    "Result: 100 passed, 0 failed, 0 skipped in 67s"
)

AGENT_TEST_PARTIAL_FAILURE_OUTPUT: str = (
    "Iteration 1/100 ... OK\n"
    "Iteration 2/100 ... FAIL\n"
    "  Error: Connection timed out after 30s\n"
    "Iteration 3/100 ... OK\n"
    "...\n"
    "Iteration 100/100 ... OK\n"
    "Result: 98 passed, 2 failed, 0 skipped in 123s"
)

AGENT_TEST_IMPORT_ERROR_OUTPUT: str = (
    "Traceback (most recent call last):\n"
    '  File "/home/deploy/agent_test.py", line 3, in <module>\n'
    "    import agent_framework\n"
    "ModuleNotFoundError: No module named 'agent_framework'"
)

SMOKE_TEST_SUCCESS_OUTPUT: str = (
    "Checking redis ... PASS\n"
    "Checking postgres ... PASS\n"
    "Checking rabbitmq ... PASS\n"
    "All services reachable."
)


# ---------------------------------------------------------------------------
# Convenience: all scenarios in a tuple for parametrized tests
# ---------------------------------------------------------------------------

ALL_DEMO_SCENARIOS: tuple[DemoScenario, ...] = (
    DEMO_1_SCENARIO,
    DemoScenario(
        name="Demo 3: Agent asks for missing args",
        nl_inputs=DEMO_3_NL_INPUTS,
        expected_spec=DEMO_3_EXPECTED_SPEC,
        expected_missing_args=DEMO_3_EXPECTED_MISSING_ARGS,
        expected_ssh_command=DEMO_3_EXPECTED_SSH_COMMAND,
        expected_summary=DEMO_3_EXPECTED_SUMMARY,
        expected_tool_sequence=DEMO_3_EXPECTED_TOOL_SEQUENCE,
    ),
)
