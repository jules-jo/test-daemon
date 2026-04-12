"""E2E integration test for the agent loop full pipeline.

Drives the agent loop through the complete pipeline:

    NL input -> wiki test spec lookup -> ask missing args ->
    propose command -> approve -> SSH execute -> summary

Uses a scripted mock LLM client that simulates realistic multi-step
tool usage, a mocked approval callback (always approves), and a
stubbed SSH executor (returns canned output). Asserts correct outputs
at each pipeline stage.

This test validates Sub-AC 6.2: end-to-end agent loop behavior with
real ToolRegistry, real tool implementations, and real ApprovalLedger
-- only the LLM and SSH execution are faked.

Test stages and the tool calls the scripted LLM returns:

    Iteration 1: lookup_test_spec (wiki lookup)
    Iteration 2: ask_user_question (missing --iterations arg)
    Iteration 3: propose_ssh_command (propose the command)
    Iteration 4: execute_ssh (run with the approval_id from step 3)
    Iteration 5: summarize_run (produce a summary) -- then no more calls

Each iteration is verified for correct tool_name, arguments, result
status, and output content. The test also verifies:
    - ApprovalLedger enforcement (execute_ssh only runs approved commands)
    - Conversation history accumulates correctly across cycles
    - Loop terminates in COMPLETE state after the final iteration
    - Self-correction scenario: LLM observes an error and retries
"""

from __future__ import annotations

import json
import sys
import textwrap
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

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
from jules_daemon.agent.tools.propose_ssh_command import (
    ApprovalLedger,
    ProposeSSHCommandTool,
)
from jules_daemon.agent.tools.summarize_run import SummarizeRunTool
from jules_daemon.wiki.test_knowledge import (
    TestKnowledge,
    save_test_knowledge,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_HOST = "10.0.1.50"
_TEST_USER = "deploy"
_TEST_COMMAND = "python3 ~/agent_test.py --iterations 100"
_NL_INPUT = "run the agent test on staging with 100 iterations"
_SYSTEM_PROMPT = "You are a test runner assistant for remote SSH execution."

_STUB_STDOUT = textwrap.dedent("""\
    Running agent_test.py...
    Iteration 1/100: PASS (0.5s)
    Iteration 50/100: PASS (25.2s)
    Iteration 100/100: PASS (50.1s)
    ===========================
    100 passed, 0 failed, 0 skipped in 50.1s
""")

_STUB_STDERR = ""


# ---------------------------------------------------------------------------
# Scripted LLM Client
# ---------------------------------------------------------------------------


class ScriptedLLMClient:
    """LLM client that returns a predefined sequence of tool call batches.

    Each entry in the script is a tuple of ToolCalls the LLM would return
    for that iteration. Entries can also be callables that receive the
    current conversation history and return tool calls dynamically (used
    when the LLM needs to reference approval_ids from earlier results).

    Satisfies the LLMClient protocol from agent_loop.py.
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
        """All message histories received, for assertion."""
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

        # Support dynamic entries that read from conversation history
        if callable(entry):
            return entry(messages)

        return entry


# ---------------------------------------------------------------------------
# Stub callbacks
# ---------------------------------------------------------------------------


class ApprovalTracker:
    """Tracks all approval requests and auto-approves them.

    Records each (command, host, explanation) tuple for assertion.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, str]] = []

    async def confirm(
        self, command: str, target_host: str, explanation: str
    ) -> tuple[bool, str]:
        """Auto-approve every command, returning it unchanged."""
        self.requests.append((command, target_host, explanation))
        return (True, command)


class QuestionTracker:
    """Tracks all questions asked and returns canned answers.

    Maps question substrings to answers. If no match, returns "100"
    as a generic numeric answer.
    """

    def __init__(
        self,
        answers: dict[str, str] | None = None,
    ) -> None:
        self._answers = dict(answers or {})
        self.questions: list[tuple[str, str]] = []

    async def ask(
        self, question: str, context: str
    ) -> str | None:
        """Return a canned answer based on question content."""
        self.questions.append((question, context))
        for substring, answer in self._answers.items():
            if substring.lower() in question.lower():
                return answer
        return "100"


# ---------------------------------------------------------------------------
# Wiki fixture helpers
# ---------------------------------------------------------------------------


def _create_test_knowledge_wiki(wiki_root: Path) -> None:
    """Write test knowledge wiki files for agent_test.

    Creates two files: one for the slug ``agent-test`` (from bare name)
    and one for ``agent-test-py`` (from the .py filename). This ensures
    lookups work regardless of whether the LLM passes ``agent_test`` or
    ``agent_test.py``.
    """
    for slug in ("agent-test", "agent-test-py"):
        knowledge = TestKnowledge(
            test_slug=slug,
            command_pattern="python3 ~/agent_test.py",
            purpose="Runs the agent integration test with configurable iterations",
            output_format="Iteration progress lines followed by a summary",
            common_failures=("timeout on slow networks", "iteration count mismatch"),
            normal_behavior="All iterations pass within 60s",
            required_args=("iterations",),
            runs_observed=5,
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
    """Dynamic script entry: produce an execute_ssh call using the approval_id."""
    approval_id = _extract_approval_id_from_history(messages)
    if approval_id is None:
        raise AssertionError(
            "Expected an approval_id in the conversation history "
            "but none was found. This indicates propose_ssh_command "
            "did not succeed in a prior iteration."
        )
    return (
        ToolCall(
            call_id="call_execute",
            tool_name="execute_ssh",
            arguments={"approval_id": approval_id},
        ),
    )


def _make_summarize_calls(
    messages: tuple[dict[str, Any], ...],
) -> tuple[ToolCall, ...]:
    """Dynamic script entry: produce a summarize_run call using stdout from execute_ssh."""
    # Find the execute_ssh result in history
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                if "stdout" in data and "exit_code" in data:
                    return (
                        ToolCall(
                            call_id="call_summarize",
                            tool_name="summarize_run",
                            arguments={
                                "stdout": data["stdout"],
                                "stderr": data.get("stderr", ""),
                                "command": data.get("command", _TEST_COMMAND),
                                "exit_code": data.get("exit_code", 0),
                            },
                        ),
                    )
            except (json.JSONDecodeError, TypeError):
                continue

    # Fallback with canned data if execute_ssh result not found
    return (
        ToolCall(
            call_id="call_summarize",
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
# Stub SSH execution and run_pipeline module patching
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
    completed_at: str = "2026-04-12T12:00:50Z"


def _make_stub_execute_run(
    *,
    success: bool = True,
    exit_code: int = 0,
    stdout: str = _STUB_STDOUT,
    stderr: str = _STUB_STDERR,
) -> AsyncMock:
    """Create a fake execute_run async function that returns StubRunResult."""

    async def _fake(**kwargs: Any) -> StubRunResult:
        return StubRunResult(
            success=success,
            run_id="run-e2e-001",
            command=kwargs.get("command", _TEST_COMMAND),
            target_host=kwargs.get("target_host", _TEST_HOST),
            target_user=kwargs.get("target_user", _TEST_USER),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            error=None if success else "Command failed",
            duration_seconds=50.1,
        )

    return _fake


def _install_mock_run_pipeline(
    execute_run_fn: Any,
) -> types.ModuleType:
    """Install a fake ``jules_daemon.execution.run_pipeline`` module.

    The real module imports ``paramiko`` at module level, which is not
    available in the test environment. This function creates a minimal
    fake module with just the ``execute_run`` function and installs it
    into ``sys.modules`` so that lazy imports in execute_ssh.py resolve.

    Returns the fake module so callers can inspect or further patch it.
    """
    mod = types.ModuleType("jules_daemon.execution.run_pipeline")
    mod.execute_run = execute_run_fn  # type: ignore[attr-defined]
    mod.RunResult = StubRunResult  # type: ignore[attr-defined]
    sys.modules["jules_daemon.execution.run_pipeline"] = mod
    return mod


def _uninstall_mock_run_pipeline() -> None:
    """Remove the fake run_pipeline module from sys.modules."""
    sys.modules.pop("jules_daemon.execution.run_pipeline", None)


# ---------------------------------------------------------------------------
# E2E Test: Full Happy-Path Pipeline
# ---------------------------------------------------------------------------


class TestAgentLoopE2EFullPipeline:
    """E2E test driving the full agent loop pipeline.

    Pipeline:
        NL input -> lookup_test_spec -> ask_user_question ->
        propose_ssh_command -> execute_ssh -> summarize_run -> COMPLETE
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_happy_path(self, tmp_path: Path) -> None:
        """Full pipeline from NL input through SSH execution to summary."""
        # -- Setup wiki with test knowledge --
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_test_knowledge_wiki(wiki_root)

        # -- Setup callbacks --
        approval_tracker = ApprovalTracker()
        question_tracker = QuestionTracker(
            answers={"iteration": "100"}
        )

        # -- Setup approval ledger (shared between propose and execute) --
        ledger = ApprovalLedger()

        # -- Build real tools (only the 5 needed for the pipeline) --
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
        summarize_tool = SummarizeRunTool()

        # -- Register tools in the real ToolRegistry --
        registry = ToolRegistry()
        registry.register(lookup_tool)
        registry.register(ask_tool)
        registry.register(propose_tool)
        registry.register(execute_tool)
        registry.register(summarize_tool)

        assert len(registry) == 5

        # -- Build the dispatch bridge --
        bridge = ToolDispatchBridge(registry=registry)

        # -- Script the LLM responses (one entry per iteration) --
        script: list[tuple[ToolCall, ...] | Any] = [
            # Iteration 1: lookup the test spec
            (
                ToolCall(
                    call_id="call_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "agent_test"},
                ),
            ),
            # Iteration 2: ask for the missing 'iterations' argument
            (
                ToolCall(
                    call_id="call_ask",
                    tool_name="ask_user_question",
                    arguments={
                        "question": "How many iterations should I use for the agent test?",
                        "context": "The test spec requires --iterations but no value was provided",
                    },
                ),
            ),
            # Iteration 3: propose the SSH command
            (
                ToolCall(
                    call_id="call_propose",
                    tool_name="propose_ssh_command",
                    arguments={
                        "command": _TEST_COMMAND,
                        "target_host": _TEST_HOST,
                        "target_user": _TEST_USER,
                        "explanation": "Running agent_test with 100 iterations as requested",
                    },
                ),
            ),
            # Iteration 4: execute SSH (dynamic -- reads approval_id from history)
            _make_execute_ssh_calls,
            # Iteration 5: summarize the run (dynamic -- reads stdout from history)
            _make_summarize_calls,
            # Iteration 6: no more tool calls -> COMPLETE
        ]

        llm_client = ScriptedLLMClient(script)

        # -- Stub the SSH execution pipeline --
        fake_execute = _make_stub_execute_run(
            success=True,
            exit_code=0,
            stdout=_STUB_STDOUT,
        )
        _install_mock_run_pipeline(fake_execute)

        # -- Build and run the agent loop --
        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=10),
        )

        try:
            result = await loop.run(_NL_INPUT)
        finally:
            _uninstall_mock_run_pipeline()

        # -- Verify final state --
        assert result.final_state is AgentLoopState.COMPLETE, (
            f"Expected COMPLETE, got {result.final_state.value}. "
            f"Error: {result.error_message}"
        )
        # 5 iterations with tool calls + 1 final iteration with no calls
        assert result.iterations_used == 6
        assert result.error_message is None

        # -- Verify LLM was called 6 times --
        assert llm_client.call_count == 6

        # -- Verify dispatch count --
        assert bridge.dispatch_count == 5

        # -- Verify all tool results --
        all_results = bridge.all_results
        assert len(all_results) == 5

        # Result 0: lookup_test_spec -> SUCCESS with test spec data
        r_lookup = all_results[0]
        assert r_lookup.tool_name == "lookup_test_spec"
        assert r_lookup.status is ToolResultStatus.SUCCESS
        lookup_data = json.loads(r_lookup.output)
        assert lookup_data["found"] is True
        assert lookup_data["test_slug"] == "agent-test"
        assert "iterations" in lookup_data["required_args"]

        # Result 1: ask_user_question -> SUCCESS with user answer
        r_ask = all_results[1]
        assert r_ask.tool_name == "ask_user_question"
        assert r_ask.status is ToolResultStatus.SUCCESS
        ask_data = json.loads(r_ask.output)
        assert ask_data["answer"] == "100"

        # Result 2: propose_ssh_command -> SUCCESS with approval
        r_propose = all_results[2]
        assert r_propose.tool_name == "propose_ssh_command"
        assert r_propose.status is ToolResultStatus.SUCCESS
        propose_data = json.loads(r_propose.output)
        assert propose_data["approved"] is True
        assert "approval_id" in propose_data
        assert propose_data["command"] == _TEST_COMMAND

        # Result 3: execute_ssh -> SUCCESS with execution output
        r_execute = all_results[3]
        assert r_execute.tool_name == "execute_ssh"
        assert r_execute.status is ToolResultStatus.SUCCESS
        execute_data = json.loads(r_execute.output)
        assert execute_data["success"] is True
        assert execute_data["exit_code"] == 0
        assert "100 passed" in execute_data["stdout"]

        # Result 4: summarize_run -> SUCCESS with summary
        r_summarize = all_results[4]
        assert r_summarize.tool_name == "summarize_run"
        assert r_summarize.status is ToolResultStatus.SUCCESS

        # -- Verify approval tracker saw exactly 1 request --
        # (only from propose_ssh_command; execute_ssh no longer prompts)
        assert len(approval_tracker.requests) == 1
        assert approval_tracker.requests[0][0] == _TEST_COMMAND
        assert approval_tracker.requests[0][1] == _TEST_HOST

        # -- Verify question tracker saw 1 question about iterations --
        assert len(question_tracker.questions) == 1
        assert "iteration" in question_tracker.questions[0][0].lower()

        # -- Verify conversation history structure --
        history = result.history
        # Minimum: system + user + (assistant+tool)*5
        assert len(history) >= 12
        assert history[0]["role"] == "system"
        assert history[0]["content"] == _SYSTEM_PROMPT
        assert history[1]["role"] == "user"
        assert history[1]["content"] == _NL_INPUT

        # Verify all roles present
        roles = {msg["role"] for msg in history}
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

        # -- Verify the approval ledger is empty (all consumed) --
        assert ledger.pending_count == 0


# ---------------------------------------------------------------------------
# E2E Test: Denial Terminates Loop
# ---------------------------------------------------------------------------


class DenyingApprovalTracker:
    """Always denies proposals."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, str]] = []

    async def confirm(
        self, command: str, target_host: str, explanation: str
    ) -> tuple[bool, str]:
        self.requests.append((command, target_host, explanation))
        return (False, command)


class TestAgentLoopE2EDenial:
    """Test that user denial terminates the agent loop."""

    @pytest.mark.asyncio
    async def test_denial_at_proposal_terminates(self, tmp_path: Path) -> None:
        """User denies propose_ssh_command -> loop terminates with ERROR."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()

        denial_tracker = DenyingApprovalTracker()
        ledger = ApprovalLedger()

        propose_tool = ProposeSSHCommandTool(
            confirm_callback=denial_tracker.confirm,
            ledger=ledger,
        )

        registry = ToolRegistry()
        registry.register(propose_tool)
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...]] = [
            (
                ToolCall(
                    call_id="call_propose",
                    tool_name="propose_ssh_command",
                    arguments={
                        "command": "echo hello",
                        "target_host": _TEST_HOST,
                        "target_user": _TEST_USER,
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

        result = await loop.run("run something")

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 1
        assert result.error_message is not None
        assert "denied" in result.error_message.lower()

        # Verify the denial was recorded
        assert len(denial_tracker.requests) == 1
        assert denial_tracker.requests[0][0] == "echo hello"


# ---------------------------------------------------------------------------
# E2E Test: execute_ssh Requires Prior Approval
# ---------------------------------------------------------------------------


class TestAgentLoopE2EApprovalEnforcement:
    """Test that execute_ssh cannot run without a prior propose_ssh_command."""

    @pytest.mark.asyncio
    async def test_execute_without_approval_returns_error(
        self, tmp_path: Path,
    ) -> None:
        """execute_ssh with a bogus approval_id returns ERROR (not terminal)."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()

        approval_tracker = ApprovalTracker()
        ledger = ApprovalLedger()

        execute_tool = ExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=approval_tracker.confirm,
        )

        registry = ToolRegistry()
        registry.register(execute_tool)
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...]] = [
            # LLM tries to execute without proposing first
            (
                ToolCall(
                    call_id="call_bad_exec",
                    tool_name="execute_ssh",
                    arguments={"approval_id": "nonexistent-approval-id"},
                ),
            ),
            # After seeing the error, LLM gives up (no more tool calls)
        ]

        llm_client = ScriptedLLMClient(script)
        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
        )

        result = await loop.run("run something")

        # Loop should COMPLETE (error was non-terminal, LLM stopped)
        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 2

        # The execute_ssh call should have returned an error result
        exec_result = bridge.all_results[0]
        assert exec_result.tool_name == "execute_ssh"
        assert exec_result.status is ToolResultStatus.ERROR
        assert "no approved command" in exec_result.error_message.lower()

        # No approval confirmations should have been triggered
        assert len(approval_tracker.requests) == 0


# ---------------------------------------------------------------------------
# E2E Test: Self-Correction After Failure
# ---------------------------------------------------------------------------


class TestAgentLoopE2ESelfCorrection:
    """Test that the agent observes a failure and proposes a correction."""

    @pytest.mark.asyncio
    async def test_agent_corrects_after_error(self, tmp_path: Path) -> None:
        """Agent sees a tool error, then retries with different args."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()

        approval_tracker = ApprovalTracker()
        ledger = ApprovalLedger()

        propose_tool = ProposeSSHCommandTool(
            confirm_callback=approval_tracker.confirm,
            ledger=ledger,
        )
        execute_tool = ExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=approval_tracker.confirm,
        )

        registry = ToolRegistry()
        registry.register(propose_tool)
        registry.register(execute_tool)
        bridge = ToolDispatchBridge(registry=registry)

        # Script: first execute_ssh fails (bad approval_id), LLM observes
        # the error, proposes correctly, then executes
        script: list[tuple[ToolCall, ...] | Any] = [
            # Iteration 1: try execute_ssh with bad approval_id (error)
            (
                ToolCall(
                    call_id="call_bad",
                    tool_name="execute_ssh",
                    arguments={"approval_id": "bad-id"},
                ),
            ),
            # Iteration 2: LLM self-corrects by proposing first
            (
                ToolCall(
                    call_id="call_propose",
                    tool_name="propose_ssh_command",
                    arguments={
                        "command": "echo corrected",
                        "target_host": _TEST_HOST,
                        "target_user": _TEST_USER,
                        "explanation": "Corrected: proposing before executing",
                    },
                ),
            ),
            # Iteration 3: execute with real approval_id (dynamic)
            _make_execute_ssh_calls,
            # Iteration 4: done
        ]

        llm_client = ScriptedLLMClient(script)

        fake_execute = _make_stub_execute_run(
            success=True,
            exit_code=0,
            stdout="corrected output",
        )
        _install_mock_run_pipeline(fake_execute)

        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=10),
        )

        try:
            result = await loop.run("run something")
        finally:
            _uninstall_mock_run_pipeline()

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 4  # 3 with tools + 1 empty

        # Verify the error was observed in the first iteration
        r_bad = bridge.all_results[0]
        assert r_bad.status is ToolResultStatus.ERROR
        assert "no approved command" in r_bad.error_message.lower()

        # Verify self-correction worked
        r_propose = bridge.all_results[1]
        assert r_propose.status is ToolResultStatus.SUCCESS
        assert r_propose.tool_name == "propose_ssh_command"

        r_execute = bridge.all_results[2]
        assert r_execute.status is ToolResultStatus.SUCCESS
        assert r_execute.tool_name == "execute_ssh"


# ---------------------------------------------------------------------------
# E2E Test: Max Iterations Guard
# ---------------------------------------------------------------------------


class TestAgentLoopE2EMaxIterations:
    """Test that max_iterations guard stops the loop."""

    @pytest.mark.asyncio
    async def test_max_iterations_stops_loop(self, tmp_path: Path) -> None:
        """Loop terminates when max iterations is reached."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_test_knowledge_wiki(wiki_root)

        lookup_tool = LookupTestSpecTool(wiki_root=wiki_root)

        registry = ToolRegistry()
        registry.register(lookup_tool)
        bridge = ToolDispatchBridge(registry=registry)

        # Script: keep doing lookup_test_spec forever
        script: list[tuple[ToolCall, ...]] = [
            (
                ToolCall(
                    call_id=f"call_{i}",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "agent_test"},
                ),
            )
            for i in range(20)  # More than max_iterations
        ]

        llm_client = ScriptedLLMClient(script)
        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=3),
        )

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 3
        assert result.error_message is not None
        assert "max iterations" in result.error_message.lower()


# ---------------------------------------------------------------------------
# E2E Test: Wiki Not Found Scenario
# ---------------------------------------------------------------------------


class TestAgentLoopE2EWikiNotFound:
    """Test that missing wiki spec is handled gracefully."""

    @pytest.mark.asyncio
    async def test_lookup_nonexistent_test_returns_not_found(
        self, tmp_path: Path,
    ) -> None:
        """lookup_test_spec for unknown test returns found=false."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        # Do NOT create any wiki files

        lookup_tool = LookupTestSpecTool(wiki_root=wiki_root)

        registry = ToolRegistry()
        registry.register(lookup_tool)
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...]] = [
            (
                ToolCall(
                    call_id="call_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "nonexistent_test"},
                ),
            ),
            # LLM stops after seeing not-found
        ]

        llm_client = ScriptedLLMClient(script)
        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
        )

        result = await loop.run("run nonexistent test")

        assert result.final_state is AgentLoopState.COMPLETE

        r_lookup = bridge.all_results[0]
        assert r_lookup.status is ToolResultStatus.SUCCESS
        lookup_data = json.loads(r_lookup.output)
        assert lookup_data["found"] is False
        assert "nonexistent" in lookup_data["message"].lower()


# ---------------------------------------------------------------------------
# E2E Test: User Cancels Question
# ---------------------------------------------------------------------------


class CancellingQuestionTracker:
    """Returns None for all questions (simulates user cancel)."""

    def __init__(self) -> None:
        self.questions: list[tuple[str, str]] = []

    async def ask(
        self, question: str, context: str
    ) -> str | None:
        self.questions.append((question, context))
        return None


class TestAgentLoopE2EUserCancelsQuestion:
    """Test that user cancelling a question terminates the loop."""

    @pytest.mark.asyncio
    async def test_user_cancels_question_terminates(
        self, tmp_path: Path,
    ) -> None:
        """User cancels an ask_user_question -> DENIED -> loop stops."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()

        cancel_tracker = CancellingQuestionTracker()
        ask_tool = AskUserQuestionTool(ask_callback=cancel_tracker.ask)

        registry = ToolRegistry()
        registry.register(ask_tool)
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...]] = [
            (
                ToolCall(
                    call_id="call_ask",
                    tool_name="ask_user_question",
                    arguments={
                        "question": "What host should I target?",
                        "context": "No target host specified",
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

        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 1
        assert result.error_message is not None
        assert "denied" in result.error_message.lower()

        # Verify the question was asked
        assert len(cancel_tracker.questions) == 1


# ---------------------------------------------------------------------------
# E2E Test: Multiple Tool Calls in Single Iteration
# ---------------------------------------------------------------------------


class TestAgentLoopE2EMultipleToolCalls:
    """Test that multiple tool calls in a single iteration all execute."""

    @pytest.mark.asyncio
    async def test_multiple_tools_in_one_iteration(
        self, tmp_path: Path,
    ) -> None:
        """LLM returns two tool calls in one batch; both execute."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_test_knowledge_wiki(wiki_root)

        lookup_tool = LookupTestSpecTool(wiki_root=wiki_root)
        summarize_tool = SummarizeRunTool()

        registry = ToolRegistry()
        registry.register(lookup_tool)
        registry.register(summarize_tool)
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...]] = [
            # Single iteration with 2 tool calls
            (
                ToolCall(
                    call_id="call_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "agent_test"},
                ),
                ToolCall(
                    call_id="call_summarize",
                    tool_name="summarize_run",
                    arguments={
                        "stdout": _STUB_STDOUT,
                        "command": _TEST_COMMAND,
                    },
                ),
            ),
            # Done
        ]

        llm_client = ScriptedLLMClient(script)
        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
        )

        result = await loop.run("analyze recent test run")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 2  # 1 with tools + 1 empty
        assert bridge.dispatch_count == 2

        # Both tools executed
        r_lookup = bridge.all_results[0]
        assert r_lookup.tool_name == "lookup_test_spec"
        assert r_lookup.status is ToolResultStatus.SUCCESS

        r_summarize = bridge.all_results[1]
        assert r_summarize.tool_name == "summarize_run"
        assert r_summarize.status is ToolResultStatus.SUCCESS


# ---------------------------------------------------------------------------
# E2E Test: SSH Execution Failure -> Agent Observes and Reports
# ---------------------------------------------------------------------------


class TestAgentLoopE2ESSHFailure:
    """Test that a failed SSH execution is observable by the agent."""

    @pytest.mark.asyncio
    async def test_ssh_failure_is_observable(self, tmp_path: Path) -> None:
        """execute_ssh failure -> ERROR result -> agent observes and stops."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()

        approval_tracker = ApprovalTracker()
        ledger = ApprovalLedger()

        propose_tool = ProposeSSHCommandTool(
            confirm_callback=approval_tracker.confirm,
            ledger=ledger,
        )
        execute_tool = ExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=approval_tracker.confirm,
        )

        registry = ToolRegistry()
        registry.register(propose_tool)
        registry.register(execute_tool)
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            # Propose
            (
                ToolCall(
                    call_id="call_propose",
                    tool_name="propose_ssh_command",
                    arguments={
                        "command": "failing_command",
                        "target_host": _TEST_HOST,
                        "target_user": _TEST_USER,
                    },
                ),
            ),
            # Execute (dynamic)
            _make_execute_ssh_calls,
            # LLM sees the failure and stops
        ]

        llm_client = ScriptedLLMClient(script)

        fake_execute = _make_stub_execute_run(
            success=False,
            exit_code=1,
            stdout="Error: command not found",
            stderr="bash: failing_command: command not found",
        )
        _install_mock_run_pipeline(fake_execute)

        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=10),
        )

        try:
            result = await loop.run("run failing command")
        finally:
            _uninstall_mock_run_pipeline()

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 3  # propose + execute + empty

        # The execute result should reflect the SSH failure
        r_execute = bridge.all_results[1]
        assert r_execute.tool_name == "execute_ssh"
        assert r_execute.status is ToolResultStatus.ERROR
        execute_data = json.loads(r_execute.output)
        assert execute_data["success"] is False
        assert execute_data["exit_code"] == 1


# ---------------------------------------------------------------------------
# E2E Test: History Correctness Across All Stages
# ---------------------------------------------------------------------------


class TestAgentLoopE2EHistoryAccumulation:
    """Verify conversation history correctness through the pipeline."""

    @pytest.mark.asyncio
    async def test_history_structure_is_correct(self, tmp_path: Path) -> None:
        """Verify each message in history has correct role and content."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_test_knowledge_wiki(wiki_root)

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

        result = await loop.run("check test spec")

        history = result.history
        # Expected structure:
        # [0] system message
        # [1] user message
        # [2] assistant message (with tool_calls)
        # [3] tool result message
        assert len(history) >= 4

        # System message
        assert history[0]["role"] == "system"
        assert history[0]["content"] == _SYSTEM_PROMPT

        # User message
        assert history[1]["role"] == "user"
        assert history[1]["content"] == "check test spec"

        # Assistant message with tool_calls
        assert history[2]["role"] == "assistant"
        assert "tool_calls" in history[2]
        assert len(history[2]["tool_calls"]) == 1
        tc = history[2]["tool_calls"][0]
        assert tc["id"] == "call_lookup"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "lookup_test_spec"

        # Tool result message
        assert history[3]["role"] == "tool"
        assert history[3]["tool_call_id"] == "call_lookup"
        # Content should be the JSON output from lookup_test_spec
        content_data = json.loads(history[3]["content"])
        assert content_data["found"] is True
        assert content_data["test_slug"] == "agent-test"
