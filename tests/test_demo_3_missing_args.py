"""Demo 3: Agent asks for missing args via wiki test catalog (AC 8, Sub-AC 3).

End-to-end integration test for the full Demo 3 scenario:
    1. User provides partial input: "run agent_test" (no iterations, no host)
    2. Agent looks up wiki test spec -> finds required_args: ("iterations", "host")
    3. Agent iteratively asks user for each missing arg:
       - asks for iterations -> user says "50"
       - asks for host -> user says "staging.example.com"
    4. Agent proposes SSH command with collected args -> user approves
    5. Agent executes SSH -> success (stubbed)
    6. Agent reads output, parses results, summarizes -> COMPLETE

Uses REAL tools (LookupTestSpecTool, AskUserQuestionTool, ProposeSSHCommandTool,
ExecuteSSHTool, ParseTestOutputTool, ReadOutputTool, SummarizeRunTool) with:
- A ScriptedLLMClient that returns pre-defined tool call sequences
- Stubbed SSH execution via mock run_pipeline
- Real wiki test knowledge persistence
- Real ApprovalLedger and ToolRegistry

Key assertions:
    - Loop completes with COMPLETE state (not ERROR)
    - Tool sequence matches DEMO_3_EXPECTED_TOOL_SEQUENCE
    - Exactly 2 user questions asked (iterations and host)
    - User answers ("50" and "staging.example.com") appear in proposed command
    - Wiki spec lookup returns correct required_args
    - Approval ledger is consumed after execution
    - Conversation history has correct structure
    - User denial at ask_user_question terminates loop
"""

from __future__ import annotations

import json
import sys
import textwrap
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from jules_daemon.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopResult,
    AgentLoopState,
)
from jules_daemon.agent.tool_dispatch import ToolDispatchBridge
from jules_daemon.agent.tool_registry import ToolRegistry
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from jules_daemon.agent.tools.ask_user_question import AskUserQuestionTool
from jules_daemon.agent.tools.execute_ssh import ExecuteSSHTool
from jules_daemon.agent.tools.lookup_test_spec import LookupTestSpecTool
from jules_daemon.agent.tools.parse_test_output import ParseTestOutputTool
from jules_daemon.agent.tools.propose_ssh_command import (
    ApprovalLedger,
    ProposeSSHCommandTool,
)
from jules_daemon.agent.tools.read_output import ReadOutputTool
from jules_daemon.agent.tools.summarize_run import SummarizeRunTool
from jules_daemon.wiki.test_knowledge import (
    TestKnowledge,
    save_test_knowledge,
)

from tests.fixtures.demo_scenarios import (
    DEMO_3_EXPECTED_MISSING_ARGS,
    DEMO_3_EXPECTED_TOOL_SEQUENCE,
    DEMO_3_NL_INPUTS,
    DEMO_3_USER_RESPONSES,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_HOST = "staging.example.com"
_TEST_USER = "deploy"
_TEST_COMMAND = (
    "python3 ~/agent_test.py --iterations 50 --host staging.example.com"
)
_NL_INPUT = DEMO_3_NL_INPUTS[0]  # "run agent_test"
_SYSTEM_PROMPT = "You are a test runner assistant for remote SSH execution."

_STUB_STDOUT = textwrap.dedent("""\
    Running agent_test.py...
    Iteration 1/50: PASS (0.5s)
    Iteration 25/50: PASS (12.6s)
    Iteration 50/50: PASS (25.3s)
    ===========================
    50 passed, 0 failed, 0 skipped in 25.3s
""")

_STUB_STDERR = ""


# ---------------------------------------------------------------------------
# Scripted LLM Client
# ---------------------------------------------------------------------------


class ScriptedLLMClient:
    """LLM client returning pre-defined tool call batches per iteration.

    Entries can be static tuples of ToolCalls or callables that receive
    the current conversation history and return tool calls dynamically
    (used when the LLM needs approval_ids from earlier results).
    """

    def __init__(
        self,
        script: list[tuple[ToolCall, ...] | Any],
    ) -> None:
        self._script = list(script)
        self._call_count = 0
        self._messages_log: list[tuple[dict[str, Any], ...]] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def messages_log(self) -> list[tuple[dict[str, Any], ...]]:
        return list(self._messages_log)

    async def get_tool_calls(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self._call_count += 1
        self._messages_log.append(messages)

        if not self._script:
            return ()

        entry = self._script.pop(0)

        if callable(entry):
            return entry(messages)

        return entry


# ---------------------------------------------------------------------------
# Stub callbacks
# ---------------------------------------------------------------------------


class ApprovalTracker:
    """Auto-approves all commands, records each request."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, str]] = []

    async def confirm(
        self, command: str, target_host: str, explanation: str
    ) -> tuple[bool, str]:
        self.requests.append((command, target_host, explanation))
        return (True, command)


class QuestionTracker:
    """Returns canned answers keyed by question substring matches."""

    def __init__(
        self,
        answers: dict[str, str] | None = None,
    ) -> None:
        self._answers = dict(answers or {})
        self.questions: list[tuple[str, str]] = []

    async def ask(
        self, question: str, context: str
    ) -> str | None:
        self.questions.append((question, context))
        for substring, answer in self._answers.items():
            if substring.lower() in question.lower():
                return answer
        return None  # No match -> cancellation


class DenyingQuestionTracker:
    """Always cancels questions (returns None)."""

    def __init__(self) -> None:
        self.questions: list[tuple[str, str]] = []

    async def ask(
        self, question: str, context: str
    ) -> str | None:
        self.questions.append((question, context))
        return None


# ---------------------------------------------------------------------------
# Wiki fixture helpers
# ---------------------------------------------------------------------------


def _create_demo_3_wiki(wiki_root: Path) -> None:
    """Write test knowledge wiki files for agent_test with required args.

    Creates entries for both slug variants (agent-test, agent-test-py)
    so lookups work regardless of how the LLM refers to the test.
    """
    for slug in ("agent-test", "agent-test-py"):
        knowledge = TestKnowledge(
            test_slug=slug,
            command_pattern="python3 ~/agent_test.py",
            purpose=(
                "Runs the agent loop stress test with configurable "
                "iterations and concurrency."
            ),
            output_format=(
                "Line-delimited progress: 'Iteration N/M ... OK|FAIL'. "
                "Final summary: 'Result: X passed, Y failed, Z skipped in Ns'."
            ),
            common_failures=(
                "timeout on large iteration counts (>500)",
                "connection refused when SSH agent is not forwarded",
                "ImportError: missing dependency on fresh hosts",
            ),
            normal_behavior="All iterations complete within the timeout.",
            required_args=("iterations", "host"),
            runs_observed=42,
        )
        save_test_knowledge(wiki_root, knowledge)


# ---------------------------------------------------------------------------
# Dynamic script entry helpers
# ---------------------------------------------------------------------------


def _extract_approval_id_from_history(
    messages: tuple[dict[str, Any], ...],
) -> str | None:
    """Extract the most recent approval_id from tool result messages."""
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                if "approval_id" in data:
                    return data["approval_id"]
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _make_execute_ssh_calls(
    messages: tuple[dict[str, Any], ...],
) -> tuple[ToolCall, ...]:
    """Dynamic entry: execute_ssh with approval_id from propose result."""
    approval_id = _extract_approval_id_from_history(messages)
    if approval_id is None:
        raise AssertionError(
            "Expected an approval_id in the conversation history "
            "but none was found. propose_ssh_command did not succeed."
        )
    return (
        ToolCall(
            call_id="call_05_execute",
            tool_name="execute_ssh",
            arguments={"approval_id": approval_id},
        ),
    )


def _make_read_and_parse_calls(
    messages: tuple[dict[str, Any], ...],
) -> tuple[ToolCall, ...]:
    """Dynamic entry: read_output + parse_test_output after execution.

    read_output reads session history for the execute_ssh result.
    parse_test_output parses the raw stdout for test counts.
    """
    # Extract stdout from execute_ssh result in history
    raw_output = _STUB_STDOUT
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                if "stdout" in data and "exit_code" in data:
                    raw_output = data["stdout"]
                    break
            except (json.JSONDecodeError, TypeError):
                continue

    return (
        ToolCall(
            call_id="call_06_read",
            tool_name="read_output",
            arguments={
                "source": "session",
                "tool_name_filter": "execute_ssh",
            },
        ),
        ToolCall(
            call_id="call_07_parse",
            tool_name="parse_test_output",
            arguments={
                "raw_output": raw_output,
                "framework_hint": "auto",
            },
        ),
    )


def _make_summarize_calls(
    messages: tuple[dict[str, Any], ...],
) -> tuple[ToolCall, ...]:
    """Dynamic entry: summarize_run from the execute_ssh output."""
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                if "stdout" in data and "exit_code" in data:
                    return (
                        ToolCall(
                            call_id="call_08_summarize",
                            tool_name="summarize_run",
                            arguments={
                                "stdout": data["stdout"],
                                "stderr": data.get("stderr", ""),
                                "command": data.get(
                                    "command", _TEST_COMMAND
                                ),
                                "exit_code": data.get("exit_code", 0),
                            },
                        ),
                    )
            except (json.JSONDecodeError, TypeError):
                continue

    return (
        ToolCall(
            call_id="call_08_summarize",
            tool_name="summarize_run",
            arguments={
                "stdout": _STUB_STDOUT,
                "stderr": "",
                "command": _TEST_COMMAND,
                "exit_code": 0,
            },
        ),
    )


# ---------------------------------------------------------------------------
# Stub SSH execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StubRunResult:
    """Mimics jules_daemon.execution.run_pipeline.RunResult for tests."""

    success: bool
    run_id: str
    command: str
    target_host: str
    target_user: str
    exit_code: int | None
    stdout: str
    stderr: str
    error: str | None
    duration_seconds: float
    started_at: str = "2026-04-12T12:00:00Z"
    completed_at: str = "2026-04-12T12:00:25Z"


def _make_stub_execute_run() -> Any:
    """Create a fake execute_run that returns a successful StubRunResult."""

    async def _fake(**kwargs: Any) -> StubRunResult:
        return StubRunResult(
            success=True,
            run_id="run-demo3-001",
            command=kwargs.get("command", _TEST_COMMAND),
            target_host=kwargs.get("target_host", _TEST_HOST),
            target_user=kwargs.get("target_user", _TEST_USER),
            exit_code=0,
            stdout=_STUB_STDOUT,
            stderr=_STUB_STDERR,
            error=None,
            duration_seconds=25.3,
        )

    return _fake


def _install_mock_run_pipeline(execute_run_fn: Any) -> types.ModuleType:
    """Install a fake run_pipeline module into sys.modules."""
    mod = types.ModuleType("jules_daemon.execution.run_pipeline")
    mod.execute_run = execute_run_fn  # type: ignore[attr-defined]
    mod.RunResult = StubRunResult  # type: ignore[attr-defined]
    sys.modules["jules_daemon.execution.run_pipeline"] = mod
    return mod


def _uninstall_mock_run_pipeline() -> None:
    """Remove the fake run_pipeline module from sys.modules."""
    sys.modules.pop("jules_daemon.execution.run_pipeline", None)


# ---------------------------------------------------------------------------
# Demo 3 LLM script builder
# ---------------------------------------------------------------------------


def _build_demo_3_script() -> list[tuple[ToolCall, ...] | Any]:
    """Build the LLM tool call sequence for Demo 3.

    Matches DEMO_3_EXPECTED_TOOL_SEQUENCE:
        1. lookup_test_spec       (wiki lookup -> finds required_args)
        2. ask_user_question      (ask for "iterations")
        3. ask_user_question      (ask for "host")
        4. propose_ssh_command    (propose command with collected args)
        5. execute_ssh            (execute approved command)
        6. read_output + parse    (analyze output -- batched)
        7. summarize_run          (final summary)
        8. (empty -> COMPLETE)
    """
    return [
        # Iteration 1: look up the test spec from wiki
        (
            ToolCall(
                call_id="call_01_lookup",
                tool_name="lookup_test_spec",
                arguments={"test_name": "agent_test"},
            ),
        ),
        # Iteration 2: ask for the missing 'iterations' argument
        (
            ToolCall(
                call_id="call_02_ask_iterations",
                tool_name="ask_user_question",
                arguments={
                    "question": (
                        "How many iterations should I use for the "
                        "agent test?"
                    ),
                    "context": (
                        "The test specification requires 'iterations' "
                        "but no value was provided in your request"
                    ),
                },
            ),
        ),
        # Iteration 3: ask for the missing 'host' argument
        (
            ToolCall(
                call_id="call_03_ask_host",
                tool_name="ask_user_question",
                arguments={
                    "question": (
                        "Which host should I run the agent test on?"
                    ),
                    "context": (
                        "The test specification requires 'host' "
                        "but no value was provided in your request"
                    ),
                },
            ),
        ),
        # Iteration 4: propose the SSH command with collected args
        (
            ToolCall(
                call_id="call_04_propose",
                tool_name="propose_ssh_command",
                arguments={
                    "command": _TEST_COMMAND,
                    "target_host": _TEST_HOST,
                    "target_user": _TEST_USER,
                    "explanation": (
                        "Running agent_test with 50 iterations on "
                        "staging.example.com as requested"
                    ),
                },
            ),
        ),
        # Iteration 5: execute SSH (dynamic -- reads approval_id)
        _make_execute_ssh_calls,
        # Iteration 6: read output + parse test output (batch)
        _make_read_and_parse_calls,
        # Iteration 7: summarize run (dynamic -- reads stdout)
        _make_summarize_calls,
        # Iteration 8: no more tool calls -> COMPLETE
    ]


# ---------------------------------------------------------------------------
# Pipeline container and builder
# ---------------------------------------------------------------------------


@dataclass
class Demo3Pipeline:
    """Holds all Demo 3 pipeline components for test assertions."""

    loop: AgentLoop
    llm_client: ScriptedLLMClient
    bridge: ToolDispatchBridge
    registry: ToolRegistry
    approval_tracker: ApprovalTracker
    question_tracker: QuestionTracker
    ledger: ApprovalLedger
    wiki_root: Path


def _build_demo_3_pipeline(tmp_path: Path) -> Demo3Pipeline:
    """Build the complete Demo 3 pipeline with real tools."""
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    _create_demo_3_wiki(wiki_root)

    approval_tracker = ApprovalTracker()
    question_tracker = QuestionTracker(
        answers={
            "iteration": DEMO_3_USER_RESPONSES["iterations"],
            "host": DEMO_3_USER_RESPONSES["host"],
        }
    )
    ledger = ApprovalLedger()

    lookup_tool = LookupTestSpecTool(wiki_root=wiki_root)
    ask_tool = AskUserQuestionTool(ask_callback=question_tracker.ask)
    propose_tool = ProposeSSHCommandTool(
        confirm_callback=approval_tracker.confirm,
        ledger=ledger,
    )
    execute_tool = ExecuteSSHTool(
        wiki_root=wiki_root,
        ledger=ledger,
        confirm_callback=approval_tracker.confirm,
    )
    parse_tool = ParseTestOutputTool()
    summarize_tool = SummarizeRunTool()

    registry = ToolRegistry()
    registry.register(lookup_tool)
    registry.register(ask_tool)
    registry.register(propose_tool)
    registry.register(execute_tool)
    registry.register(parse_tool)
    registry.register(summarize_tool)

    bridge = ToolDispatchBridge(registry=registry)
    llm_client = ScriptedLLMClient(_build_demo_3_script())

    loop = AgentLoop(
        llm_client=llm_client,
        tool_dispatcher=bridge,
        system_prompt=_SYSTEM_PROMPT,
        config=AgentLoopConfig(max_iterations=12),
    )

    # Register read_output after loop creation so it can access
    # the loop's internal history via session_history_provider.
    read_output_tool = ReadOutputTool(
        wiki_root=wiki_root,
        session_history_provider=lambda: tuple(loop._history),
    )
    registry.register(read_output_tool)

    return Demo3Pipeline(
        loop=loop,
        llm_client=llm_client,
        bridge=bridge,
        registry=registry,
        approval_tracker=approval_tracker,
        question_tracker=question_tracker,
        ledger=ledger,
        wiki_root=wiki_root,
    )


async def _run_demo_3(
    tmp_path: Path,
) -> tuple[AgentLoopResult, Demo3Pipeline]:
    """Build and run the Demo 3 pipeline, returning result and components."""
    pipeline = _build_demo_3_pipeline(tmp_path)

    _install_mock_run_pipeline(_make_stub_execute_run())

    try:
        result = await pipeline.loop.run(_NL_INPUT)
    finally:
        _uninstall_mock_run_pipeline()

    return result, pipeline


# ---------------------------------------------------------------------------
# E2E Tests: Full Demo 3 Pipeline
# ---------------------------------------------------------------------------


class TestDemo3FullPipeline:
    """Demo 3: partial args -> wiki lookup -> iterative user prompts -> run.

    The complete flow:
        NL input ("run agent_test") -> lookup_test_spec (required_args)
        -> ask_user_question (iterations) -> ask_user_question (host)
        -> propose_ssh_command -> execute_ssh
        -> read_output + parse_test_output -> summarize_run -> COMPLETE
    """

    @pytest.mark.asyncio
    async def test_full_missing_args_flow_completes(
        self, tmp_path: Path
    ) -> None:
        """Full pipeline from partial input through prompts to completion."""
        result, _ = await _run_demo_3(tmp_path)

        assert result.final_state is AgentLoopState.COMPLETE, (
            f"Expected COMPLETE, got {result.final_state.value}. "
            f"Error: {result.error_message}"
        )
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_tool_sequence_matches_expected(
        self, tmp_path: Path
    ) -> None:
        """Dispatched tool names match DEMO_3_EXPECTED_TOOL_SEQUENCE."""
        _, pipeline = await _run_demo_3(tmp_path)

        dispatched_names = tuple(
            r.tool_name for r in pipeline.bridge.all_results
        )
        assert dispatched_names == DEMO_3_EXPECTED_TOOL_SEQUENCE

    @pytest.mark.asyncio
    async def test_two_user_questions_asked(
        self, tmp_path: Path
    ) -> None:
        """Exactly 2 user questions asked: one for iterations, one for host."""
        _, pipeline = await _run_demo_3(tmp_path)

        assert len(pipeline.question_tracker.questions) == 2

        q1_text = pipeline.question_tracker.questions[0][0].lower()
        assert "iteration" in q1_text

        q2_text = pipeline.question_tracker.questions[1][0].lower()
        assert "host" in q2_text

    @pytest.mark.asyncio
    async def test_user_answers_reflected_in_command(
        self, tmp_path: Path
    ) -> None:
        """Proposed SSH command includes user-provided values."""
        _, pipeline = await _run_demo_3(tmp_path)

        propose_result = next(
            r for r in pipeline.bridge.all_results
            if r.tool_name == "propose_ssh_command"
        )
        assert propose_result.status is ToolResultStatus.SUCCESS
        propose_data = json.loads(propose_result.output)

        assert propose_data["approved"] is True
        assert "50" in propose_data["command"]
        assert "staging.example.com" in propose_data["command"]

    @pytest.mark.asyncio
    async def test_wiki_spec_lookup_returns_required_args(
        self, tmp_path: Path
    ) -> None:
        """lookup_test_spec returns spec with required_args."""
        _, pipeline = await _run_demo_3(tmp_path)

        lookup_result = next(
            r for r in pipeline.bridge.all_results
            if r.tool_name == "lookup_test_spec"
        )
        assert lookup_result.status is ToolResultStatus.SUCCESS
        lookup_data = json.loads(lookup_result.output)

        assert lookup_data["found"] is True
        assert "iterations" in lookup_data["required_args"]
        assert "host" in lookup_data["required_args"]

    @pytest.mark.asyncio
    async def test_approval_ledger_consumed(
        self, tmp_path: Path
    ) -> None:
        """Approval ledger is empty after execution."""
        _, pipeline = await _run_demo_3(tmp_path)

        assert pipeline.ledger.pending_count == 0

    @pytest.mark.asyncio
    async def test_approval_tracker_saw_requests(
        self, tmp_path: Path
    ) -> None:
        """Approval tracker received propose and execute requests."""
        _, pipeline = await _run_demo_3(tmp_path)

        # 1 request: from propose_ssh_command only (execute_ssh no longer prompts)
        assert len(pipeline.approval_tracker.requests) == 1
        assert pipeline.approval_tracker.requests[0][1] == _TEST_HOST

    @pytest.mark.asyncio
    async def test_iterations_used(
        self, tmp_path: Path
    ) -> None:
        """Loop uses 8 iterations (7 with tool calls + 1 empty)."""
        result, _ = await _run_demo_3(tmp_path)

        assert result.iterations_used == 8

    @pytest.mark.asyncio
    async def test_llm_called_eight_times(
        self, tmp_path: Path
    ) -> None:
        """LLM is called once per iteration (8 total)."""
        _, pipeline = await _run_demo_3(tmp_path)

        assert pipeline.llm_client.call_count == 8

    @pytest.mark.asyncio
    async def test_conversation_history_structure(
        self, tmp_path: Path
    ) -> None:
        """History has system, user, assistant, and tool messages."""
        result, _ = await _run_demo_3(tmp_path)

        history = result.history
        # system + user + 7*(assistant + at_least_1_tool)
        assert len(history) >= 16

        assert history[0]["role"] == "system"
        assert history[0]["content"] == _SYSTEM_PROMPT
        assert history[1]["role"] == "user"
        assert history[1]["content"] == _NL_INPUT

        roles = {msg["role"] for msg in history}
        assert roles == {"system", "user", "assistant", "tool"}

    @pytest.mark.asyncio
    async def test_execute_ssh_succeeds(
        self, tmp_path: Path
    ) -> None:
        """execute_ssh returns SUCCESS with correct output."""
        _, pipeline = await _run_demo_3(tmp_path)

        execute_result = next(
            r for r in pipeline.bridge.all_results
            if r.tool_name == "execute_ssh"
        )
        assert execute_result.status is ToolResultStatus.SUCCESS
        execute_data = json.loads(execute_result.output)

        assert execute_data["success"] is True
        assert execute_data["exit_code"] == 0
        assert "50 passed" in execute_data["stdout"]

    @pytest.mark.asyncio
    async def test_ask_user_answers_match_fixture(
        self, tmp_path: Path
    ) -> None:
        """User answers match the DEMO_3_USER_RESPONSES fixture values."""
        _, pipeline = await _run_demo_3(tmp_path)

        ask_results = [
            r for r in pipeline.bridge.all_results
            if r.tool_name == "ask_user_question"
        ]
        assert len(ask_results) == 2
        for r in ask_results:
            assert r.status is ToolResultStatus.SUCCESS

        answers = {json.loads(r.output)["answer"] for r in ask_results}
        assert DEMO_3_USER_RESPONSES["iterations"] in answers
        assert DEMO_3_USER_RESPONSES["host"] in answers

    @pytest.mark.asyncio
    async def test_dispatch_count(
        self, tmp_path: Path
    ) -> None:
        """Exactly 8 tool calls dispatched across all iterations."""
        _, pipeline = await _run_demo_3(tmp_path)

        assert pipeline.bridge.dispatch_count == 8

    @pytest.mark.asyncio
    async def test_read_output_returns_session_data(
        self, tmp_path: Path
    ) -> None:
        """read_output returns session data with execute_ssh entries."""
        _, pipeline = await _run_demo_3(tmp_path)

        read_result = next(
            r for r in pipeline.bridge.all_results
            if r.tool_name == "read_output"
        )
        assert read_result.status is ToolResultStatus.SUCCESS
        read_data = json.loads(read_result.output)

        assert read_data["source"] == "session"
        assert read_data["tool_name_filter"] == "execute_ssh"
        assert read_data["returned_count"] >= 1

    @pytest.mark.asyncio
    async def test_summarize_run_produces_output(
        self, tmp_path: Path
    ) -> None:
        """summarize_run produces a structured summary."""
        _, pipeline = await _run_demo_3(tmp_path)

        summarize_result = next(
            r for r in pipeline.bridge.all_results
            if r.tool_name == "summarize_run"
        )
        assert summarize_result.status is ToolResultStatus.SUCCESS


# ---------------------------------------------------------------------------
# E2E Tests: User Denial Terminates Loop
# ---------------------------------------------------------------------------


class TestDemo3UserDenialAtQuestion:
    """User denial during question asking terminates the loop."""

    @pytest.mark.asyncio
    async def test_denial_at_first_question_terminates(
        self, tmp_path: Path
    ) -> None:
        """User cancels the first question -> loop terminates with ERROR."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_demo_3_wiki(wiki_root)

        denying_tracker = DenyingQuestionTracker()
        ask_tool = AskUserQuestionTool(ask_callback=denying_tracker.ask)
        lookup_tool = LookupTestSpecTool(wiki_root=wiki_root)

        registry = ToolRegistry()
        registry.register(lookup_tool)
        registry.register(ask_tool)

        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...]] = [
            (
                ToolCall(
                    call_id="call_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "agent_test"},
                ),
            ),
            (
                ToolCall(
                    call_id="call_ask",
                    tool_name="ask_user_question",
                    arguments={
                        "question": "How many iterations?",
                        "context": "Missing required arg",
                    },
                ),
            ),
        ]

        llm_client = ScriptedLLMClient(script)
        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
        )

        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.ERROR
        assert "denied" in (result.error_message or "").lower()
        assert len(denying_tracker.questions) == 1

    @pytest.mark.asyncio
    async def test_no_proposal_after_denial(
        self, tmp_path: Path
    ) -> None:
        """No propose_ssh_command is dispatched after question denial."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_demo_3_wiki(wiki_root)

        denying_tracker = DenyingQuestionTracker()
        ledger = ApprovalLedger()

        lookup_tool = LookupTestSpecTool(wiki_root=wiki_root)
        ask_tool = AskUserQuestionTool(ask_callback=denying_tracker.ask)
        propose_tool = ProposeSSHCommandTool(
            confirm_callback=ApprovalTracker().confirm,
            ledger=ledger,
        )

        registry = ToolRegistry()
        registry.register(lookup_tool)
        registry.register(ask_tool)
        registry.register(propose_tool)

        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...]] = [
            (
                ToolCall(
                    call_id="call_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "agent_test"},
                ),
            ),
            (
                ToolCall(
                    call_id="call_ask",
                    tool_name="ask_user_question",
                    arguments={
                        "question": "How many iterations?",
                        "context": "Missing required arg",
                    },
                ),
            ),
        ]

        llm_client = ScriptedLLMClient(script)
        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
        )

        await loop.run(_NL_INPUT)

        dispatched_names = tuple(
            r.tool_name for r in bridge.all_results
        )
        assert "propose_ssh_command" not in dispatched_names
        assert ledger.pending_count == 0


# ---------------------------------------------------------------------------
# Missing Args Detection from Wiki Spec
# ---------------------------------------------------------------------------


class TestDemo3MissingArgsDetection:
    """Verify wiki spec correctly surfaces required_args for Demo 3."""

    @pytest.mark.asyncio
    async def test_lookup_returns_both_required_args(
        self, tmp_path: Path
    ) -> None:
        """lookup_test_spec returns both 'iterations' and 'host'."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_demo_3_wiki(wiki_root)

        lookup_tool = LookupTestSpecTool(wiki_root=wiki_root)
        result = await lookup_tool.execute(
            call_id="test_call",
            args={"test_name": "agent_test"},
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["found"] is True
        assert set(data["required_args"]) == {"iterations", "host"}

    def test_fixture_missing_args_consistency(self) -> None:
        """DEMO_3_EXPECTED_MISSING_ARGS correctly identifies both as missing."""
        assert DEMO_3_EXPECTED_MISSING_ARGS.required_args == (
            "iterations",
            "host",
        )
        assert DEMO_3_EXPECTED_MISSING_ARGS.provided_args == ()
        assert DEMO_3_EXPECTED_MISSING_ARGS.missing_args == (
            "iterations",
            "host",
        )


# ---------------------------------------------------------------------------
# NL Input Variations
# ---------------------------------------------------------------------------


class TestDemo3InputVariations:
    """All Demo 3 NL input variations trigger correct wiki lookup."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "nl_input",
        DEMO_3_NL_INPUTS,
        ids=[f"input_{i}" for i in range(len(DEMO_3_NL_INPUTS))],
    )
    async def test_all_nl_inputs_trigger_lookup(
        self, tmp_path: Path, nl_input: str
    ) -> None:
        """Each NL input variation completes with a successful lookup."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_demo_3_wiki(wiki_root)

        lookup_tool = LookupTestSpecTool(wiki_root=wiki_root)
        registry = ToolRegistry()
        registry.register(lookup_tool)
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...]] = [
            (
                ToolCall(
                    call_id="call_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "agent_test"},
                ),
            ),
        ]

        llm_client = ScriptedLLMClient(script)
        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
        )

        result = await loop.run(nl_input)

        assert result.final_state is AgentLoopState.COMPLETE

        lookup_result = bridge.all_results[0]
        assert lookup_result.status is ToolResultStatus.SUCCESS
        data = json.loads(lookup_result.output)
        assert data["found"] is True
        assert "iterations" in data["required_args"]
        assert "host" in data["required_args"]


# ---------------------------------------------------------------------------
# Conversation History Flow Verification
# ---------------------------------------------------------------------------


class TestDemo3ConversationHistoryFlow:
    """Verify the LLM sees correct context at each iteration."""

    @pytest.mark.asyncio
    async def test_llm_sees_lookup_result_before_asking(
        self, tmp_path: Path
    ) -> None:
        """Iteration 2 (first ask) messages contain lookup_test_spec result."""
        _, pipeline = await _run_demo_3(tmp_path)

        # Iteration 2 messages (index 1, 0-based)
        assert pipeline.llm_client.call_count >= 2
        messages = pipeline.llm_client.messages_log[1]

        has_lookup = any(
            msg.get("role") == "tool"
            and isinstance(msg.get("content", ""), str)
            and "required_args" in msg["content"]
            for msg in messages
        )
        assert has_lookup, (
            "LLM must see lookup_test_spec result (with required_args) "
            "before asking for missing arguments"
        )

    @pytest.mark.asyncio
    async def test_llm_sees_iterations_answer_before_asking_host(
        self, tmp_path: Path
    ) -> None:
        """Iteration 3 (ask host) messages contain the iterations answer."""
        _, pipeline = await _run_demo_3(tmp_path)

        assert pipeline.llm_client.call_count >= 3
        messages = pipeline.llm_client.messages_log[2]

        tool_contents = " ".join(
            str(msg.get("content", ""))
            for msg in messages
            if msg.get("role") == "tool"
        )
        assert "50" in tool_contents, (
            "LLM must see iterations=50 answer before asking for host"
        )

    @pytest.mark.asyncio
    async def test_llm_sees_both_answers_before_proposing(
        self, tmp_path: Path
    ) -> None:
        """Iteration 4 (propose) messages contain both user answers."""
        _, pipeline = await _run_demo_3(tmp_path)

        assert pipeline.llm_client.call_count >= 4
        messages = pipeline.llm_client.messages_log[3]

        tool_contents = " ".join(
            str(msg.get("content", ""))
            for msg in messages
            if msg.get("role") == "tool"
        )
        assert "50" in tool_contents, (
            "LLM must see iterations=50 before proposing command"
        )
        assert "staging.example.com" in tool_contents, (
            "LLM must see host=staging.example.com before proposing"
        )

    @pytest.mark.asyncio
    async def test_history_contains_user_answers_in_tool_messages(
        self, tmp_path: Path
    ) -> None:
        """Final history has tool messages with the user-provided values."""
        result, _ = await _run_demo_3(tmp_path)

        tool_messages = [
            msg for msg in result.history
            if msg.get("role") == "tool"
        ]
        all_content = " ".join(
            str(m.get("content", "")) for m in tool_messages
        )
        assert "50" in all_content
        assert "staging.example.com" in all_content

    @pytest.mark.asyncio
    async def test_history_is_immutable_tuple(
        self, tmp_path: Path
    ) -> None:
        """Returned history is an immutable tuple."""
        result, _ = await _run_demo_3(tmp_path)
        assert isinstance(result.history, tuple)


# ---------------------------------------------------------------------------
# Iteration Budget Enforcement
# ---------------------------------------------------------------------------


class TestDemo3IterationBudget:
    """Verify Demo 3 respects iteration limits."""

    @pytest.mark.asyncio
    async def test_fails_with_insufficient_iterations(
        self, tmp_path: Path
    ) -> None:
        """Loop errors out when max_iterations < required iterations."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_demo_3_wiki(wiki_root)

        question_tracker = QuestionTracker(
            answers={
                "iteration": DEMO_3_USER_RESPONSES["iterations"],
                "host": DEMO_3_USER_RESPONSES["host"],
            }
        )
        approval_tracker = ApprovalTracker()
        ledger = ApprovalLedger()

        lookup_tool = LookupTestSpecTool(wiki_root=wiki_root)
        ask_tool = AskUserQuestionTool(ask_callback=question_tracker.ask)
        propose_tool = ProposeSSHCommandTool(
            confirm_callback=approval_tracker.confirm, ledger=ledger,
        )
        execute_tool = ExecuteSSHTool(
            wiki_root=wiki_root, ledger=ledger,
            confirm_callback=approval_tracker.confirm,
        )
        parse_tool = ParseTestOutputTool()
        summarize_tool = SummarizeRunTool()

        registry = ToolRegistry()
        registry.register(lookup_tool)
        registry.register(ask_tool)
        registry.register(propose_tool)
        registry.register(execute_tool)
        registry.register(parse_tool)
        registry.register(summarize_tool)

        bridge = ToolDispatchBridge(registry=registry)
        llm_client = ScriptedLLMClient(_build_demo_3_script())

        # Only 3 iterations -- not enough for the 8 needed
        config = AgentLoopConfig(max_iterations=3, max_retries=2)

        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=config,
        )

        _install_mock_run_pipeline(_make_stub_execute_run())
        try:
            result = await loop.run(_NL_INPUT)
        finally:
            _uninstall_mock_run_pipeline()

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 3
        assert result.error_message is not None
        assert "max iterations" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_succeeds_with_sufficient_iterations(
        self, tmp_path: Path
    ) -> None:
        """With enough iterations, the full Demo 3 flow completes."""
        result, _ = await _run_demo_3(tmp_path)

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used <= 12  # config max


# ---------------------------------------------------------------------------
# NL Input Variations with Full Pipeline
# ---------------------------------------------------------------------------


class TestDemo3NLInputFullPipeline:
    """All Demo 3 NL input variations complete the full pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "nl_input",
        DEMO_3_NL_INPUTS,
        ids=[f"phrase_{i}" for i in range(len(DEMO_3_NL_INPUTS))],
    )
    async def test_all_nl_phrases_complete_full_flow(
        self, tmp_path: Path, nl_input: str
    ) -> None:
        """Each NL input phrase drives the complete partial-args pipeline."""
        pipeline = _build_demo_3_pipeline(tmp_path)

        _install_mock_run_pipeline(_make_stub_execute_run())
        try:
            result = await pipeline.loop.run(nl_input)
        finally:
            _uninstall_mock_run_pipeline()

        assert result.final_state is AgentLoopState.COMPLETE
        tool_names = tuple(
            r.tool_name for r in pipeline.bridge.all_results
        )
        # All phrases must trigger the full tool sequence
        assert tool_names == DEMO_3_EXPECTED_TOOL_SEQUENCE
        # All phrases must ask exactly 2 questions
        assert len(pipeline.question_tracker.questions) == 2


# ---------------------------------------------------------------------------
# Proposal Denial After Questions
# ---------------------------------------------------------------------------


class TestDemo3ProposalDenialAfterQuestions:
    """Verify loop terminates if user denies SSH proposal after questions."""

    @pytest.mark.asyncio
    async def test_denial_at_proposal_terminates(
        self, tmp_path: Path
    ) -> None:
        """User denies propose_ssh_command after answering both questions."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_demo_3_wiki(wiki_root)

        question_tracker = QuestionTracker(
            answers={
                "iteration": DEMO_3_USER_RESPONSES["iterations"],
                "host": DEMO_3_USER_RESPONSES["host"],
            }
        )
        ledger = ApprovalLedger()

        class DenyingApprovalTracker:
            """Always denies proposals."""

            def __init__(self) -> None:
                self.requests: list[tuple[str, str, str]] = []

            async def confirm(
                self, command: str, target_host: str, explanation: str
            ) -> tuple[bool, str]:
                self.requests.append((command, target_host, explanation))
                return (False, command)

        deny_tracker = DenyingApprovalTracker()

        lookup_tool = LookupTestSpecTool(wiki_root=wiki_root)
        ask_tool = AskUserQuestionTool(ask_callback=question_tracker.ask)
        propose_tool = ProposeSSHCommandTool(
            confirm_callback=deny_tracker.confirm, ledger=ledger,
        )

        registry = ToolRegistry()
        registry.register(lookup_tool)
        registry.register(ask_tool)
        registry.register(propose_tool)

        bridge = ToolDispatchBridge(registry=registry)
        llm_client = ScriptedLLMClient(_build_demo_3_script())
        config = AgentLoopConfig(max_iterations=10, max_retries=2)

        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=config,
        )

        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.ERROR
        assert result.error_message is not None
        assert "denied" in result.error_message.lower()

        # Questions were asked before the denial
        assert len(question_tracker.questions) == 2
        # Proposal was attempted once
        assert len(deny_tracker.requests) == 1

        # No execute_ssh should have been dispatched
        dispatched_names = tuple(
            r.tool_name for r in bridge.all_results
        )
        assert "execute_ssh" not in dispatched_names

    @pytest.mark.asyncio
    async def test_denial_stops_at_correct_iteration(
        self, tmp_path: Path
    ) -> None:
        """Loop stops at iteration 4 (lookup, ask, ask, propose-denied)."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_demo_3_wiki(wiki_root)

        question_tracker = QuestionTracker(
            answers={
                "iteration": DEMO_3_USER_RESPONSES["iterations"],
                "host": DEMO_3_USER_RESPONSES["host"],
            }
        )

        async def deny_confirm(
            command: str, target_host: str, explanation: str
        ) -> tuple[bool, str]:
            return (False, command)

        ledger = ApprovalLedger()
        lookup_tool = LookupTestSpecTool(wiki_root=wiki_root)
        ask_tool = AskUserQuestionTool(ask_callback=question_tracker.ask)
        propose_tool = ProposeSSHCommandTool(
            confirm_callback=deny_confirm, ledger=ledger,
        )

        registry = ToolRegistry()
        registry.register(lookup_tool)
        registry.register(ask_tool)
        registry.register(propose_tool)

        bridge = ToolDispatchBridge(registry=registry)
        llm_client = ScriptedLLMClient(_build_demo_3_script())

        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=10),
        )

        result = await loop.run(_NL_INPUT)

        assert result.iterations_used == 4


# ---------------------------------------------------------------------------
# Approval Enforcement: Ordering
# ---------------------------------------------------------------------------


class TestDemo3ApprovalOrdering:
    """Verify that propose always precedes execute in the tool sequence."""

    @pytest.mark.asyncio
    async def test_propose_index_less_than_execute_index(
        self, tmp_path: Path
    ) -> None:
        """propose_ssh_command index < execute_ssh index in dispatch order."""
        _, pipeline = await _run_demo_3(tmp_path)

        names = tuple(r.tool_name for r in pipeline.bridge.all_results)

        propose_idx = names.index("propose_ssh_command")
        execute_idx = names.index("execute_ssh")
        assert propose_idx < execute_idx

    @pytest.mark.asyncio
    async def test_no_execute_without_prior_propose(
        self, tmp_path: Path
    ) -> None:
        """Every execute_ssh is preceded by a propose_ssh_command."""
        _, pipeline = await _run_demo_3(tmp_path)

        names = tuple(r.tool_name for r in pipeline.bridge.all_results)

        execute_indices = [
            i for i, n in enumerate(names) if n == "execute_ssh"
        ]
        propose_indices = [
            i for i, n in enumerate(names) if n == "propose_ssh_command"
        ]

        assert len(execute_indices) == 1
        assert len(propose_indices) == 1
        assert propose_indices[0] < execute_indices[0]


# ---------------------------------------------------------------------------
# Fixture Consistency Cross-Checks
# ---------------------------------------------------------------------------


class TestDemo3FixtureConsistency:
    """Verify Demo 3 test fixtures are internally consistent."""

    def test_user_responses_cover_missing_args(self) -> None:
        """DEMO_3_USER_RESPONSES keys match DEMO_3_EXPECTED_MISSING_ARGS."""
        missing = set(DEMO_3_EXPECTED_MISSING_ARGS.missing_args)
        responses = set(DEMO_3_USER_RESPONSES.keys())
        assert missing == responses

    def test_nl_inputs_omit_all_arg_values(self) -> None:
        """Demo 3 NL inputs do not contain iterations or host values."""
        for phrase in DEMO_3_NL_INPUTS:
            assert "50" not in phrase
            assert "100" not in phrase
            assert "staging.example.com" not in phrase
            assert "staging" not in phrase.lower()

    def test_expected_tool_sequence_starts_with_lookup(self) -> None:
        """Tool sequence starts with lookup_test_spec."""
        assert DEMO_3_EXPECTED_TOOL_SEQUENCE[0] == "lookup_test_spec"

    def test_expected_tool_sequence_has_two_asks(self) -> None:
        """Tool sequence has exactly 2 ask_user_question calls."""
        count = DEMO_3_EXPECTED_TOOL_SEQUENCE.count("ask_user_question")
        assert count == 2

    def test_expected_tool_sequence_ends_with_summarize(self) -> None:
        """Tool sequence ends with summarize_run."""
        assert DEMO_3_EXPECTED_TOOL_SEQUENCE[-1] == "summarize_run"

    def test_expected_tool_sequence_has_propose_before_execute(self) -> None:
        """propose_ssh_command comes before execute_ssh in expected sequence."""
        seq = DEMO_3_EXPECTED_TOOL_SEQUENCE
        propose_idx = seq.index("propose_ssh_command")
        execute_idx = seq.index("execute_ssh")
        assert propose_idx < execute_idx

    def test_missing_args_equals_required_minus_provided(self) -> None:
        """missing_args = required_args - provided_args."""
        args = DEMO_3_EXPECTED_MISSING_ARGS
        expected_missing = set(args.required_args) - set(args.provided_args)
        assert set(args.missing_args) == expected_missing
