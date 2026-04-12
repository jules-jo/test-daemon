"""Tests for demo scenario fixtures -- validates fixture data integrity.

Sub-AC 6.1: Verifies that the demo test fixtures are well-formed and
integrate correctly with the existing wiki persistence layer, tool types,
and agent loop infrastructure.

Coverage:
    - Wiki test spec entries parse correctly with the frontmatter module
    - Spec entries round-trip through save/load via test_knowledge
    - NL input phrases are non-empty and distinct per scenario
    - Expected intermediate outputs are structurally valid
    - lookup_test_spec tool returns expected data for fixture specs
    - Missing-args detection is consistent with required_args
    - ExpectedSSHCommand can construct a valid SSHCommand
    - Summary templates reference valid test slugs
    - Frozen dataclass immutability enforced
    - All three demo scenarios have complete data
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jules_daemon.agent.tool_types import ToolResultStatus
from jules_daemon.ssh.command import SSHCommand
from jules_daemon.wiki.frontmatter import WikiDocument, parse, serialize
from jules_daemon.wiki.test_knowledge import (
    TestKnowledge,
    derive_test_slug,
    load_test_knowledge,
    save_test_knowledge,
)

from tests.fixtures.demo_scenarios import (
    ALL_DEMO_SCENARIOS,
    AGENT_TEST_IMPORT_ERROR_OUTPUT,
    AGENT_TEST_PARTIAL_FAILURE_OUTPUT,
    AGENT_TEST_SUCCESS_OUTPUT,
    DEMO_1_EXPECTED_MISSING_ARGS,
    DEMO_1_EXPECTED_SPEC,
    DEMO_1_EXPECTED_SSH_COMMAND,
    DEMO_1_EXPECTED_SUMMARY,
    DEMO_1_EXPECTED_TOOL_SEQUENCE,
    DEMO_1_NL_INPUTS,
    DEMO_1_SCENARIO,
    DEMO_2_CORRECTED_SSH_COMMAND,
    DEMO_2_CORRECTED_SUCCESS_OUTPUT,
    DEMO_2_EXPECTED_SPEC,
    DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL,
    DEMO_2_FIRST_FAILURE_OUTPUT,
    DEMO_2_FIRST_SSH_COMMAND,
    DEMO_2_NL_INPUTS,
    DEMO_3_EXPECTED_MISSING_ARGS,
    DEMO_3_EXPECTED_SPEC,
    DEMO_3_EXPECTED_SSH_COMMAND,
    DEMO_3_EXPECTED_TOOL_SEQUENCE,
    DEMO_3_NL_INPUTS,
    DEMO_3_USER_RESPONSES,
    SMOKE_TEST_SUCCESS_OUTPUT,
    DemoScenario,
    ExpectedMissingArgs,
    ExpectedSSHCommand,
    ExpectedSpecLookup,
    ExpectedSummaryTemplate,
)
from tests.fixtures.wiki_test_specs import (
    AGENT_TEST_SPEC_RAW,
    PYTEST_INTEGRATION_SPEC_RAW,
    SMOKE_TEST_SPEC_RAW,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Create a temporary wiki root with the required directory structure."""
    from jules_daemon.wiki.layout import initialize_wiki

    initialize_wiki(tmp_path)
    return tmp_path


def _write_spec_file(wiki_root: Path, slug: str, raw_content: str) -> Path:
    """Write a raw wiki spec to the knowledge directory."""
    knowledge_dir = wiki_root / "pages" / "daemon" / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    file_path = knowledge_dir / f"test-{slug}.md"
    file_path.write_text(raw_content, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Wiki test spec parsing
# ---------------------------------------------------------------------------


class TestWikiTestSpecParsing:
    """Verify wiki test spec fixtures parse correctly."""

    def test_agent_test_spec_parses(self) -> None:
        """agent_test spec has valid YAML frontmatter."""
        doc = parse(AGENT_TEST_SPEC_RAW)
        assert isinstance(doc, WikiDocument)
        assert doc.frontmatter["test_slug"] == "agent-test-py"
        assert doc.frontmatter["type"] == "test-knowledge"

    def test_pytest_integration_spec_parses(self) -> None:
        """pytest integration spec has valid YAML frontmatter."""
        doc = parse(PYTEST_INTEGRATION_SPEC_RAW)
        assert doc.frontmatter["test_slug"] == "pytest-tests-integration"

    def test_smoke_test_spec_parses(self) -> None:
        """smoke_test spec has valid YAML frontmatter."""
        doc = parse(SMOKE_TEST_SPEC_RAW)
        assert doc.frontmatter["test_slug"] == "smoke-test-sh"

    @pytest.mark.parametrize(
        "raw_spec,expected_slug",
        [
            (AGENT_TEST_SPEC_RAW, "agent-test-py"),
            (PYTEST_INTEGRATION_SPEC_RAW, "pytest-tests-integration"),
            (SMOKE_TEST_SPEC_RAW, "smoke-test-sh"),
        ],
        ids=["agent_test", "pytest_integration", "smoke_test"],
    )
    def test_all_specs_have_required_frontmatter_keys(
        self, raw_spec: str, expected_slug: str
    ) -> None:
        """All spec fixtures must include the standard test-knowledge keys."""
        doc = parse(raw_spec)
        fm = doc.frontmatter
        required_keys = {
            "tags",
            "type",
            "test_slug",
            "command_pattern",
            "purpose",
            "required_args",
            "runs_observed",
            "last_updated",
        }
        missing = required_keys - set(fm.keys())
        assert not missing, f"Missing frontmatter keys for {expected_slug}: {missing}"

    @pytest.mark.parametrize(
        "raw_spec",
        [AGENT_TEST_SPEC_RAW, PYTEST_INTEGRATION_SPEC_RAW, SMOKE_TEST_SPEC_RAW],
        ids=["agent_test", "pytest_integration", "smoke_test"],
    )
    def test_specs_have_daemon_tag(self, raw_spec: str) -> None:
        """All specs must include the 'daemon' tag."""
        doc = parse(raw_spec)
        assert "daemon" in doc.frontmatter["tags"]

    @pytest.mark.parametrize(
        "raw_spec",
        [AGENT_TEST_SPEC_RAW, PYTEST_INTEGRATION_SPEC_RAW, SMOKE_TEST_SPEC_RAW],
        ids=["agent_test", "pytest_integration", "smoke_test"],
    )
    def test_specs_have_markdown_body(self, raw_spec: str) -> None:
        """All specs must have a non-empty markdown body."""
        doc = parse(raw_spec)
        assert doc.body.strip(), "Markdown body must not be empty"

    def test_agent_test_required_args(self) -> None:
        """agent_test spec must declare iterations and host as required."""
        doc = parse(AGENT_TEST_SPEC_RAW)
        required = doc.frontmatter["required_args"]
        assert "iterations" in required
        assert "host" in required

    def test_pytest_integration_no_required_args(self) -> None:
        """pytest integration spec has no required args."""
        doc = parse(PYTEST_INTEGRATION_SPEC_RAW)
        assert doc.frontmatter["required_args"] == []

    def test_smoke_test_no_required_args(self) -> None:
        """smoke_test spec has no required args."""
        doc = parse(SMOKE_TEST_SPEC_RAW)
        assert doc.frontmatter["required_args"] == []


# ---------------------------------------------------------------------------
# Wiki round-trip via test_knowledge
# ---------------------------------------------------------------------------


class TestWikiRoundTrip:
    """Verify fixtures round-trip through the wiki test_knowledge layer."""

    def test_agent_test_round_trip(self, wiki_root: Path) -> None:
        """Write agent_test spec, then load it back and verify fields."""
        _write_spec_file(wiki_root, "agent-test-py", AGENT_TEST_SPEC_RAW)

        knowledge = load_test_knowledge(wiki_root, "agent-test-py")
        assert knowledge is not None
        assert knowledge.test_slug == "agent-test-py"
        assert knowledge.command_pattern == "python3 ~/agent_test.py"
        assert "iterations" in knowledge.required_args
        assert "host" in knowledge.required_args
        assert knowledge.runs_observed == 42

    def test_pytest_integration_round_trip(self, wiki_root: Path) -> None:
        """Write pytest spec, then load it back and verify."""
        _write_spec_file(
            wiki_root, "pytest-tests-integration", PYTEST_INTEGRATION_SPEC_RAW
        )

        knowledge = load_test_knowledge(wiki_root, "pytest-tests-integration")
        assert knowledge is not None
        assert knowledge.test_slug == "pytest-tests-integration"
        assert knowledge.required_args == ()

    def test_smoke_test_round_trip(self, wiki_root: Path) -> None:
        """Write smoke_test spec, then load it back."""
        _write_spec_file(wiki_root, "smoke-test-sh", SMOKE_TEST_SPEC_RAW)

        knowledge = load_test_knowledge(wiki_root, "smoke-test-sh")
        assert knowledge is not None
        assert knowledge.test_slug == "smoke-test-sh"
        assert knowledge.runs_observed == 87

    def test_save_and_reload_preserves_fields(self, wiki_root: Path) -> None:
        """TestKnowledge save then reload must preserve all fields."""
        original = TestKnowledge(
            test_slug="agent-test-py",
            command_pattern="python3 ~/agent_test.py",
            purpose="Agent stress test",
            required_args=("iterations", "host"),
            common_failures=("timeout", "connection refused"),
            runs_observed=42,
        )
        save_test_knowledge(wiki_root, original)
        loaded = load_test_knowledge(wiki_root, "agent-test-py")

        assert loaded is not None
        assert loaded.test_slug == original.test_slug
        assert loaded.command_pattern == original.command_pattern
        assert loaded.purpose == original.purpose
        assert loaded.required_args == original.required_args
        assert loaded.common_failures == original.common_failures
        assert loaded.runs_observed == original.runs_observed


# ---------------------------------------------------------------------------
# NL input phrase validation
# ---------------------------------------------------------------------------


class TestNLInputPhrases:
    """Verify NL input phrases are well-formed and distinct."""

    def test_demo_1_has_multiple_phrases(self) -> None:
        """Demo 1 must provide at least 3 NL phrase variations."""
        assert len(DEMO_1_NL_INPUTS) >= 3

    def test_demo_2_has_multiple_phrases(self) -> None:
        """Demo 2 must provide at least 3 NL phrase variations."""
        assert len(DEMO_2_NL_INPUTS) >= 3

    def test_demo_3_has_multiple_phrases(self) -> None:
        """Demo 3 must provide at least 3 NL phrase variations."""
        assert len(DEMO_3_NL_INPUTS) >= 3

    def test_demo_1_phrases_are_distinct(self) -> None:
        """Demo 1 phrases must all be unique."""
        assert len(set(DEMO_1_NL_INPUTS)) == len(DEMO_1_NL_INPUTS)

    def test_demo_2_phrases_are_distinct(self) -> None:
        """Demo 2 phrases must all be unique."""
        assert len(set(DEMO_2_NL_INPUTS)) == len(DEMO_2_NL_INPUTS)

    def test_demo_3_phrases_are_distinct(self) -> None:
        """Demo 3 phrases must all be unique."""
        assert len(set(DEMO_3_NL_INPUTS)) == len(DEMO_3_NL_INPUTS)

    def test_all_phrases_non_empty(self) -> None:
        """Every NL phrase must be a non-empty stripped string."""
        all_phrases = DEMO_1_NL_INPUTS + DEMO_2_NL_INPUTS + DEMO_3_NL_INPUTS
        for phrase in all_phrases:
            assert phrase.strip(), f"Found empty NL phrase: {phrase!r}"

    def test_demo_1_phrases_mention_agent_test(self) -> None:
        """Demo 1 phrases should reference the agent test."""
        for phrase in DEMO_1_NL_INPUTS:
            assert "agent" in phrase.lower() or "test" in phrase.lower()

    def test_demo_3_phrases_omit_args(self) -> None:
        """Demo 3 phrases must not contain iteration counts (missing args)."""
        for phrase in DEMO_3_NL_INPUTS:
            # Phrases should NOT contain specific iteration counts
            # because the point of Demo 3 is that args are missing
            assert "100" not in phrase
            assert "50" not in phrase


# ---------------------------------------------------------------------------
# Expected spec lookup validation
# ---------------------------------------------------------------------------


class TestExpectedSpecLookup:
    """Verify expected spec lookups are structurally valid."""

    def test_demo_1_spec_found(self) -> None:
        """Demo 1 spec must be found with correct slug."""
        assert DEMO_1_EXPECTED_SPEC.found is True
        assert DEMO_1_EXPECTED_SPEC.test_slug == "agent-test-py"

    def test_demo_1_spec_has_required_args(self) -> None:
        """Demo 1 spec must declare iterations and host."""
        assert "iterations" in DEMO_1_EXPECTED_SPEC.required_args
        assert "host" in DEMO_1_EXPECTED_SPEC.required_args

    def test_demo_2_spec_found(self) -> None:
        """Demo 2 spec must be found."""
        assert DEMO_2_EXPECTED_SPEC.found is True
        assert DEMO_2_EXPECTED_SPEC.test_slug == "pytest-tests-integration"

    def test_demo_2_spec_no_required_args(self) -> None:
        """Demo 2 spec has no required args."""
        assert DEMO_2_EXPECTED_SPEC.required_args == ()

    def test_demo_3_spec_same_as_demo_1(self) -> None:
        """Demo 3 uses the same test spec as Demo 1."""
        assert DEMO_3_EXPECTED_SPEC.test_slug == DEMO_1_EXPECTED_SPEC.test_slug
        assert DEMO_3_EXPECTED_SPEC.required_args == DEMO_1_EXPECTED_SPEC.required_args

    def test_to_dict_found_format(self) -> None:
        """to_dict for found spec includes all expected keys."""
        data = DEMO_1_EXPECTED_SPEC.to_dict()
        assert data["found"] is True
        assert "test_slug" in data
        assert "command_pattern" in data
        assert "purpose" in data
        assert "required_args" in data
        assert isinstance(data["required_args"], list)

    def test_to_dict_not_found_format(self) -> None:
        """to_dict for not-found spec includes slug and message."""
        not_found = ExpectedSpecLookup(
            found=False,
            test_slug="unknown-test",
            command_pattern="unknown_command",
        )
        data = not_found.to_dict()
        assert data["found"] is False
        assert "test_slug" in data
        assert "message" in data

    def test_spec_lookup_frozen(self) -> None:
        """ExpectedSpecLookup must be immutable."""
        with pytest.raises(AttributeError):
            DEMO_1_EXPECTED_SPEC.found = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Missing args detection validation
# ---------------------------------------------------------------------------


class TestExpectedMissingArgs:
    """Verify missing-args fixtures are consistent."""

    def test_demo_1_all_args_provided(self) -> None:
        """Demo 1: user provides all required args, none missing."""
        args = DEMO_1_EXPECTED_MISSING_ARGS
        assert args.missing_args == ()
        assert set(args.provided_args) == set(args.required_args)

    def test_demo_2_no_required_args(self) -> None:
        """Demo 2: no required args, nothing missing."""
        from tests.fixtures.demo_scenarios import DEMO_2_EXPECTED_MISSING_ARGS

        args = DEMO_2_EXPECTED_MISSING_ARGS
        assert args.required_args == ()
        assert args.missing_args == ()

    def test_demo_3_all_args_missing(self) -> None:
        """Demo 3: user provides no args, all required args are missing."""
        args = DEMO_3_EXPECTED_MISSING_ARGS
        assert args.provided_args == ()
        assert set(args.missing_args) == set(args.required_args)
        assert len(args.missing_args) == 2

    def test_missing_equals_required_minus_provided(self) -> None:
        """missing_args must equal required_args - provided_args for all demos."""
        for args_fixture in [
            DEMO_1_EXPECTED_MISSING_ARGS,
            DEMO_3_EXPECTED_MISSING_ARGS,
        ]:
            expected_missing = set(args_fixture.required_args) - set(
                args_fixture.provided_args
            )
            assert set(args_fixture.missing_args) == expected_missing

    def test_user_responses_cover_missing_args(self) -> None:
        """Demo 3 user responses must provide values for all missing args."""
        missing = set(DEMO_3_EXPECTED_MISSING_ARGS.missing_args)
        provided = set(DEMO_3_USER_RESPONSES.keys())
        assert missing == provided


# ---------------------------------------------------------------------------
# Expected SSH command validation
# ---------------------------------------------------------------------------


class TestExpectedSSHCommand:
    """Verify expected SSH commands can construct valid SSHCommand objects."""

    def test_demo_1_ssh_command_valid(self) -> None:
        """Demo 1 SSH command must pass SSHCommand validation."""
        cmd = DEMO_1_EXPECTED_SSH_COMMAND
        ssh = SSHCommand(
            command=cmd.command,
            working_directory=cmd.working_directory,
            timeout=cmd.timeout,
        )
        assert ssh.command == cmd.command
        assert ssh.timeout == cmd.timeout

    def test_demo_2_first_command_valid(self) -> None:
        """Demo 2 first attempt SSH command must pass validation."""
        cmd = DEMO_2_FIRST_SSH_COMMAND
        ssh = SSHCommand(
            command=cmd.command,
            working_directory=cmd.working_directory,
            timeout=cmd.timeout,
        )
        assert ssh.command == cmd.command
        assert ssh.working_directory == "/opt/app"

    def test_demo_2_corrected_command_valid(self) -> None:
        """Demo 2 corrected SSH command must pass validation."""
        cmd = DEMO_2_CORRECTED_SSH_COMMAND
        ssh = SSHCommand(
            command=cmd.command,
            working_directory=cmd.working_directory,
            timeout=cmd.timeout,
        )
        assert ssh.command == cmd.command
        assert ssh.timeout == 600  # extended for install + test

    def test_demo_3_ssh_command_valid(self) -> None:
        """Demo 3 SSH command must pass validation."""
        cmd = DEMO_3_EXPECTED_SSH_COMMAND
        ssh = SSHCommand(
            command=cmd.command,
            working_directory=cmd.working_directory,
            timeout=cmd.timeout,
        )
        assert ssh.command == cmd.command

    def test_all_commands_non_empty(self) -> None:
        """Every expected command string must be non-empty."""
        all_commands = [
            DEMO_1_EXPECTED_SSH_COMMAND,
            DEMO_2_FIRST_SSH_COMMAND,
            DEMO_2_CORRECTED_SSH_COMMAND,
            DEMO_3_EXPECTED_SSH_COMMAND,
        ]
        for cmd in all_commands:
            assert cmd.command.strip()
            assert cmd.description.strip()

    def test_ssh_command_frozen(self) -> None:
        """ExpectedSSHCommand must be immutable."""
        with pytest.raises(AttributeError):
            DEMO_1_EXPECTED_SSH_COMMAND.command = "bad"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Expected summary template validation
# ---------------------------------------------------------------------------


class TestExpectedSummaryTemplate:
    """Verify summary templates are structurally valid."""

    def test_demo_1_summary_matches_spec(self) -> None:
        """Demo 1 summary slug must match the spec slug."""
        assert (
            DEMO_1_EXPECTED_SUMMARY.test_slug
            == DEMO_1_EXPECTED_SPEC.test_slug
        )

    def test_demo_1_summary_success(self) -> None:
        """Demo 1 expects a successful run (exit code 0)."""
        assert DEMO_1_EXPECTED_SUMMARY.exit_code == 0
        assert DEMO_1_EXPECTED_SUMMARY.failed == 0

    def test_demo_1_summary_counts(self) -> None:
        """Demo 1 expects 100 passed tests."""
        assert DEMO_1_EXPECTED_SUMMARY.passed == 100

    def test_demo_2_summary_success_after_correction(self) -> None:
        """Demo 2 expects success after self-correction."""
        from tests.fixtures.demo_scenarios import DEMO_2_EXPECTED_SUMMARY

        assert DEMO_2_EXPECTED_SUMMARY.exit_code == 0
        assert DEMO_2_EXPECTED_SUMMARY.passed == 3
        assert DEMO_2_EXPECTED_SUMMARY.failed == 0

    def test_summary_frozen(self) -> None:
        """ExpectedSummaryTemplate must be immutable."""
        with pytest.raises(AttributeError):
            DEMO_1_EXPECTED_SUMMARY.exit_code = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tool sequence validation
# ---------------------------------------------------------------------------


class TestToolSequences:
    """Verify expected tool call sequences are complete and valid."""

    VALID_TOOL_NAMES: frozenset[str] = frozenset({
        "read_wiki",
        "lookup_test_spec",
        "check_remote_processes",
        "propose_ssh_command",
        "execute_ssh",
        "read_output",
        "parse_test_output",
        "summarize_run",
        "ask_user_question",
        "notify_user",
    })

    def test_demo_1_tool_sequence_valid(self) -> None:
        """Demo 1 tool sequence uses only registered tool names."""
        for name in DEMO_1_EXPECTED_TOOL_SEQUENCE:
            assert name in self.VALID_TOOL_NAMES, f"Unknown tool: {name}"

    def test_demo_1_starts_with_lookup(self) -> None:
        """Demo 1 must start by looking up the test spec."""
        assert DEMO_1_EXPECTED_TOOL_SEQUENCE[0] == "lookup_test_spec"

    def test_demo_1_ends_with_summarize(self) -> None:
        """Demo 1 must end with summarize_run."""
        assert DEMO_1_EXPECTED_TOOL_SEQUENCE[-1] == "summarize_run"

    def test_demo_1_propose_before_execute(self) -> None:
        """propose_ssh_command must come before execute_ssh."""
        seq = DEMO_1_EXPECTED_TOOL_SEQUENCE
        propose_idx = seq.index("propose_ssh_command")
        execute_idx = seq.index("execute_ssh")
        assert propose_idx < execute_idx

    def test_demo_2_has_two_propose_execute_cycles(self) -> None:
        """Demo 2 must contain two propose->execute pairs (original + correction)."""
        seq = DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL
        propose_count = seq.count("propose_ssh_command")
        execute_count = seq.count("execute_ssh")
        assert propose_count == 2
        assert execute_count == 2

    def test_demo_2_all_tools_valid(self) -> None:
        """Demo 2 tool sequence uses only registered tool names."""
        for name in DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL:
            assert name in self.VALID_TOOL_NAMES, f"Unknown tool: {name}"

    def test_demo_3_includes_ask_user_question(self) -> None:
        """Demo 3 must include ask_user_question for missing args."""
        assert "ask_user_question" in DEMO_3_EXPECTED_TOOL_SEQUENCE

    def test_demo_3_ask_before_propose(self) -> None:
        """Demo 3: ask_user_question must precede propose_ssh_command."""
        seq = DEMO_3_EXPECTED_TOOL_SEQUENCE
        last_ask_idx = len(seq) - 1 - list(reversed(seq)).index("ask_user_question")
        propose_idx = seq.index("propose_ssh_command")
        assert last_ask_idx < propose_idx

    def test_demo_3_two_ask_calls_for_two_missing_args(self) -> None:
        """Demo 3 must ask twice (once per missing arg)."""
        ask_count = DEMO_3_EXPECTED_TOOL_SEQUENCE.count("ask_user_question")
        assert ask_count == len(DEMO_3_EXPECTED_MISSING_ARGS.missing_args)

    def test_demo_3_all_tools_valid(self) -> None:
        """Demo 3 tool sequence uses only registered tool names."""
        for name in DEMO_3_EXPECTED_TOOL_SEQUENCE:
            assert name in self.VALID_TOOL_NAMES, f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# DemoScenario completeness
# ---------------------------------------------------------------------------


class TestDemoScenarioCompleteness:
    """Verify the DemoScenario dataclass bundles all required data."""

    def test_demo_1_scenario_has_all_fields(self) -> None:
        """Demo 1 scenario must have all expected intermediate outputs."""
        s = DEMO_1_SCENARIO
        assert s.name
        assert len(s.nl_inputs) > 0
        assert s.expected_spec.found is True
        assert len(s.expected_tool_sequence) > 0

    def test_all_scenarios_have_names(self) -> None:
        """Every scenario in ALL_DEMO_SCENARIOS must have a non-empty name."""
        for scenario in ALL_DEMO_SCENARIOS:
            assert scenario.name.strip()

    def test_all_scenarios_have_nl_inputs(self) -> None:
        """Every scenario must have at least one NL input phrase."""
        for scenario in ALL_DEMO_SCENARIOS:
            assert len(scenario.nl_inputs) > 0

    def test_demo_scenario_frozen(self) -> None:
        """DemoScenario must be immutable."""
        with pytest.raises(AttributeError):
            DEMO_1_SCENARIO.name = "changed"  # type: ignore[misc]

    def test_all_scenarios_count(self) -> None:
        """We should have at least 2 demo scenarios registered."""
        assert len(ALL_DEMO_SCENARIOS) >= 2


# ---------------------------------------------------------------------------
# Test output samples validation
# ---------------------------------------------------------------------------


class TestOutputSamples:
    """Verify test output sample strings are well-formed."""

    def test_agent_test_success_has_result_line(self) -> None:
        """Success output must end with a Result: summary line."""
        assert "Result:" in AGENT_TEST_SUCCESS_OUTPUT
        assert "100 passed" in AGENT_TEST_SUCCESS_OUTPUT
        assert "0 failed" in AGENT_TEST_SUCCESS_OUTPUT

    def test_agent_test_failure_has_result_line(self) -> None:
        """Partial failure output must contain fail counts."""
        assert "FAIL" in AGENT_TEST_PARTIAL_FAILURE_OUTPUT
        assert "2 failed" in AGENT_TEST_PARTIAL_FAILURE_OUTPUT

    def test_agent_test_import_error_is_traceback(self) -> None:
        """Import error output must be a Python traceback."""
        assert "Traceback" in AGENT_TEST_IMPORT_ERROR_OUTPUT
        assert "ModuleNotFoundError" in AGENT_TEST_IMPORT_ERROR_OUTPUT

    def test_smoke_test_success_has_pass_lines(self) -> None:
        """Smoke test success output must show PASS for each service."""
        assert SMOKE_TEST_SUCCESS_OUTPUT.count("PASS") >= 3

    def test_demo_2_failure_output_has_error(self) -> None:
        """Demo 2 failure output must show the ConnectionError."""
        assert "ConnectionError" in DEMO_2_FIRST_FAILURE_OUTPUT
        assert "FAILED" in DEMO_2_FIRST_FAILURE_OUTPUT

    def test_demo_2_success_output_all_passed(self) -> None:
        """Demo 2 corrected output must show all passing."""
        assert "PASSED" in DEMO_2_CORRECTED_SUCCESS_OUTPUT
        assert "3 passed" in DEMO_2_CORRECTED_SUCCESS_OUTPUT


# ---------------------------------------------------------------------------
# Integration: lookup_test_spec tool with fixture wiki data
# ---------------------------------------------------------------------------


class TestLookupTestSpecWithFixtures:
    """Verify the lookup_test_spec tool returns expected data for fixtures."""

    @pytest.mark.asyncio
    async def test_agent_test_spec_via_tool(self, wiki_root: Path) -> None:
        """lookup_test_spec returns matching data for the agent_test fixture."""
        from jules_daemon.agent.tools.lookup_test_spec import LookupTestSpecTool

        _write_spec_file(wiki_root, "agent-test-py", AGENT_TEST_SPEC_RAW)

        tool = LookupTestSpecTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="fix1",
            args={"test_name": "python3 ~/agent_test.py"},
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["found"] is True
        assert data["test_slug"] == "agent-test-py"
        assert "iterations" in data["required_args"]
        assert "host" in data["required_args"]
        assert data["runs_observed"] == 42

    @pytest.mark.asyncio
    async def test_pytest_integration_spec_via_tool(self, wiki_root: Path) -> None:
        """lookup_test_spec returns matching data for the pytest fixture."""
        from jules_daemon.agent.tools.lookup_test_spec import LookupTestSpecTool

        _write_spec_file(
            wiki_root, "pytest-tests-integration", PYTEST_INTEGRATION_SPEC_RAW
        )

        tool = LookupTestSpecTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="fix2",
            args={"test_name": "pytest tests/integration/ -v --tb=short"},
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["found"] is True
        assert data["test_slug"] == "pytest-tests-integration"
        assert data["required_args"] == []

    @pytest.mark.asyncio
    async def test_smoke_test_spec_via_tool(self, wiki_root: Path) -> None:
        """lookup_test_spec returns matching data for the smoke_test fixture."""
        from jules_daemon.agent.tools.lookup_test_spec import LookupTestSpecTool

        _write_spec_file(wiki_root, "smoke-test-sh", SMOKE_TEST_SPEC_RAW)

        tool = LookupTestSpecTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="fix3",
            args={"test_name": "./smoke_test.sh"},
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["found"] is True
        assert data["test_slug"] == "smoke-test-sh"
        assert data["runs_observed"] == 87


# ---------------------------------------------------------------------------
# Slug derivation consistency
# ---------------------------------------------------------------------------


class TestSlugDerivationConsistency:
    """Verify slug derivation matches fixture expectations."""

    def test_agent_test_slug(self) -> None:
        """derive_test_slug for 'python3 ~/agent_test.py' must match fixture."""
        slug = derive_test_slug("python3 ~/agent_test.py")
        assert slug == "agent-test-py"

    def test_pytest_integration_slug(self) -> None:
        """derive_test_slug for pytest integration must match fixture."""
        slug = derive_test_slug("pytest tests/integration/ -v --tb=short")
        assert slug == "pytest-tests-integration"

    def test_smoke_test_slug(self) -> None:
        """derive_test_slug for smoke_test.sh must match fixture."""
        slug = derive_test_slug("./smoke_test.sh")
        assert slug == "smoke-test-sh"
