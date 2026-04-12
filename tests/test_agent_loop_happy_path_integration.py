"""Integration test: happy-path end-to-end agent loop (AC 130301, Sub-AC 1).

Validates the complete agent loop pipeline with mocked LLM responses
driving multiple think-act cycles through to clean completion. Each
test exercises the real AgentLoop state machine, real ToolRegistry,
real tool implementations, and real ApprovalLedger -- only the LLM
client and SSH execution backend are faked.

This test suite is distinct from test_agent_loop_e2e.py (Sub-AC 6.2)
in that it focuses on:

    1. Think-act-observe cycle mechanics across multiple iterations
    2. Read-only tools running without approval vs state-changing tools
       requiring CONFIRM_PROMPT
    3. Test catalog integration: wiki lookup -> missing-arg detection
       -> ask_user_question
    4. Notification delivery within the agent loop flow
    5. Conversation history accumulation and structural correctness
    6. Transient LLM error retry followed by successful completion
    7. Multiple read-only tool calls in a single iteration (parallel
       batch dispatch)

Test scenarios:

    TestHappyPathFullCycle
        Drives 6 iterations: read_wiki -> lookup_test_spec ->
        ask_user_question -> propose_ssh_command -> execute_ssh ->
        notify_user + summarize_run -> COMPLETE

    TestReadOnlyToolsRunFreely
        Verifies read-only tools (lookup_test_spec, read_wiki,
        summarize_run) execute without approval callback invocation

    TestTransientRetryThenComplete
        A transient LLM timeout on iteration 2 retries successfully
        and the loop completes normally

    TestConversationHistoryAccumulation
        Verifies that each think-act-observe cycle appends the
        correct messages (system, user, assistant+tool_calls, tool
        results) and that the history grows monotonically

    TestMultipleReadOnlyToolsInOneCycle
        Two read-only tools in a single iteration both execute and
        both results appear in the conversation history
"""

from __future__ import annotations

import json
import sys
import textwrap
import types
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from jules_daemon.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopState,
)
from jules_daemon.agent.tool_dispatch import ToolDispatchBridge
from jules_daemon.agent.tool_registry import ToolRegistry
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResultStatus,
)
from jules_daemon.agent.tools.ask_user_question import AskUserQuestionTool
from jules_daemon.agent.tools.execute_ssh import ExecuteSSHTool
from jules_daemon.agent.tools.lookup_test_spec import LookupTestSpecTool
from jules_daemon.agent.tools.notify_user import NotifyUserTool
from jules_daemon.agent.tools.propose_ssh_command import (
    ApprovalLedger,
    ProposeSSHCommandTool,
)
from jules_daemon.agent.tools.read_wiki import ReadWikiTool
from jules_daemon.agent.tools.summarize_run import SummarizeRunTool
from jules_daemon.wiki.test_knowledge import (
    TestKnowledge,
    save_test_knowledge,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_HOST = "staging-01.example.com"
_TEST_USER = "testrunner"
_TEST_COMMAND = "python3 ~/smoke_suite.py --env staging --iterations 50"
_NL_INPUT = "run the smoke suite on staging with 50 iterations"
_SYSTEM_PROMPT = (
    "You are a test-runner assistant. Use the available tools to look up "
    "test specs, ask the user for missing parameters, propose SSH commands, "
    "execute them after approval, and summarize results."
)

_STUB_STDOUT = textwrap.dedent("""\
    Running smoke_suite.py on staging...
    Iteration 1/50: PASS (0.3s)
    Iteration 25/50: PASS (12.6s)
    Iteration 50/50: PASS (25.0s)
    ===========================
    50 passed, 0 failed, 0 skipped in 25.0s
""")

_STUB_STDERR = ""


# ---------------------------------------------------------------------------
# Scripted LLM Client
# ---------------------------------------------------------------------------


class ScriptedLLMClient:
    """LLM client returning a predefined sequence of tool call batches.

    Each script entry is either a tuple of ToolCalls (static) or a
    callable that receives the current conversation history and returns
    tool calls dynamically (used when the LLM needs data from prior
    results, e.g., approval_id from propose_ssh_command).

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

        # Dynamic entries receive the conversation history
        if callable(entry):
            return entry(messages)

        return entry


# ---------------------------------------------------------------------------
# Scripted LLM Client with transient error injection
# ---------------------------------------------------------------------------


class TransientErrorLLMClient:
    """LLM client that raises a transient error once, then succeeds.

    Used to validate that the agent loop retries transient LLM errors
    within the same iteration and continues successfully.

    Parameters:
        script: Normal script entries (same as ScriptedLLMClient).
        error_at_call: 1-based call number at which to raise.
        error_count: How many consecutive times to raise before
            returning the next script entry.
        error_type: Exception class to raise (default: TimeoutError).
    """

    def __init__(
        self,
        script: list[tuple[ToolCall, ...] | Any],
        *,
        error_at_call: int,
        error_count: int = 1,
        error_type: type[Exception] = TimeoutError,
    ) -> None:
        self._script = list(script)
        self._call_count = 0
        self._error_at_call = error_at_call
        self._errors_remaining = error_count
        self._error_type = error_type
        self._total_errors_raised = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_errors_raised(self) -> int:
        return self._total_errors_raised

    async def get_tool_calls(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self._call_count += 1

        if (
            self._call_count >= self._error_at_call
            and self._errors_remaining > 0
        ):
            self._errors_remaining -= 1
            self._total_errors_raised += 1
            raise self._error_type("Simulated transient LLM timeout")

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
    """Auto-approves every command and tracks all approval requests."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, str]] = []

    async def confirm(
        self, command: str, target_host: str, explanation: str
    ) -> tuple[bool, str]:
        self.requests.append((command, target_host, explanation))
        return (True, command)


class QuestionTracker:
    """Returns canned answers by matching question substrings."""

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
        return "50"


class NotificationTracker:
    """Records all notifications and returns success."""

    def __init__(self) -> None:
        self.notifications: list[tuple[str, str]] = []

    async def notify(self, message: str, severity: str) -> bool:
        self.notifications.append((message, severity))
        return True


# ---------------------------------------------------------------------------
# Wiki fixture helpers
# ---------------------------------------------------------------------------


def _create_test_knowledge_wiki(wiki_root: Path) -> None:
    """Write test knowledge wiki files for smoke_suite."""
    for slug in ("smoke-suite", "smoke-suite-py"):
        knowledge = TestKnowledge(
            test_slug=slug,
            command_pattern="python3 ~/smoke_suite.py",
            purpose="Runs the smoke test suite on a target environment",
            output_format="Iteration progress lines followed by a summary",
            common_failures=("timeout on slow connections", "env var missing"),
            normal_behavior="All iterations pass within 60s",
            required_args=("env", "iterations"),
            runs_observed=12,
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
    """Dynamic entry: produce execute_ssh with approval_id from history."""
    approval_id = _extract_approval_id_from_history(messages)
    if approval_id is None:
        raise AssertionError(
            "Expected an approval_id in the conversation history "
            "but none was found. propose_ssh_command must succeed first."
        )
    return (
        ToolCall(
            call_id="call_execute",
            tool_name="execute_ssh",
            arguments={"approval_id": approval_id},
        ),
    )


def _make_notify_and_summarize_calls(
    messages: tuple[dict[str, Any], ...],
) -> tuple[ToolCall, ...]:
    """Dynamic entry: notify_user + summarize_run in one iteration."""
    # Extract stdout from execute_ssh result
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                if "stdout" in data and "exit_code" in data:
                    return (
                        ToolCall(
                            call_id="call_notify",
                            tool_name="notify_user",
                            arguments={
                                "message": "Smoke suite completed: all 50 passed",
                                "severity": "success",
                            },
                        ),
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

    # Fallback with canned data
    return (
        ToolCall(
            call_id="call_notify",
            tool_name="notify_user",
            arguments={
                "message": "Smoke suite completed",
                "severity": "success",
            },
        ),
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
    started_at: str = "2026-04-12T14:00:00Z"
    completed_at: str = "2026-04-12T14:00:25Z"


def _make_stub_execute_run(
    *,
    success: bool = True,
    exit_code: int = 0,
    stdout: str = _STUB_STDOUT,
    stderr: str = _STUB_STDERR,
) -> Any:
    """Create a fake execute_run async function."""

    async def _fake(**kwargs: Any) -> StubRunResult:
        return StubRunResult(
            success=success,
            run_id="run-happy-001",
            command=kwargs.get("command", _TEST_COMMAND),
            target_host=kwargs.get("target_host", _TEST_HOST),
            target_user=kwargs.get("target_user", _TEST_USER),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            error=None if success else "Command failed",
            duration_seconds=25.0,
        )

    return _fake


def _install_mock_run_pipeline(execute_run_fn: Any) -> types.ModuleType:
    """Install a fake run_pipeline module in sys.modules."""
    mod = types.ModuleType("jules_daemon.execution.run_pipeline")
    mod.execute_run = execute_run_fn  # type: ignore[attr-defined]
    mod.RunResult = StubRunResult  # type: ignore[attr-defined]
    sys.modules["jules_daemon.execution.run_pipeline"] = mod
    return mod


def _uninstall_mock_run_pipeline() -> None:
    """Remove the fake run_pipeline module from sys.modules."""
    sys.modules.pop("jules_daemon.execution.run_pipeline", None)


@pytest.fixture()
def stub_run_pipeline() -> Generator[types.ModuleType, None, None]:
    """Pytest fixture that installs/removes a fake run_pipeline module.

    Guarantees cleanup even if the test is skipped or raises unexpectedly.
    Yields the fake module for optional inspection.
    """
    fake_execute = _make_stub_execute_run(success=True, exit_code=0)
    mod = _install_mock_run_pipeline(fake_execute)
    yield mod
    _uninstall_mock_run_pipeline()


# ---------------------------------------------------------------------------
# Shared fixture: build registry with all tools for the happy path
# ---------------------------------------------------------------------------


def _build_full_registry(
    *,
    wiki_root: Path,
    ledger: ApprovalLedger,
    approval_tracker: ApprovalTracker,
    question_tracker: QuestionTracker,
    notification_tracker: NotificationTracker,
) -> ToolRegistry:
    """Build a ToolRegistry with all tools needed for the happy path.

    Tools registered:
        - read_wiki (read-only)
        - lookup_test_spec (read-only)
        - ask_user_question (read-only -- user interaction, no approval)
        - propose_ssh_command (CONFIRM_PROMPT)
        - execute_ssh (CONFIRM_PROMPT)
        - notify_user (read-only)
        - summarize_run (read-only)
    """
    registry = ToolRegistry()
    registry.register(ReadWikiTool(wiki_root=wiki_root))
    registry.register(LookupTestSpecTool(wiki_root=wiki_root))
    registry.register(AskUserQuestionTool(ask_callback=question_tracker.ask))
    registry.register(
        ProposeSSHCommandTool(
            confirm_callback=approval_tracker.confirm,
            ledger=ledger,
        )
    )
    registry.register(
        ExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=approval_tracker.confirm,
        )
    )
    registry.register(
        NotifyUserTool(notify_callback=notification_tracker.notify)
    )
    registry.register(SummarizeRunTool())
    return registry


# ---------------------------------------------------------------------------
# Test: Happy-Path Full Cycle (6 iterations + terminal empty)
# ---------------------------------------------------------------------------


class TestHappyPathFullCycle:
    """End-to-end happy path: 6 think-act-observe cycles to completion.

    Pipeline:
        Iteration 1: read_wiki (read-only, no approval)
        Iteration 2: lookup_test_spec (read-only, no approval)
        Iteration 3: ask_user_question (missing 'iterations' arg)
        Iteration 4: propose_ssh_command (CONFIRM_PROMPT, auto-approved)
        Iteration 5: execute_ssh (uses approval_id from step 4)
        Iteration 6: notify_user + summarize_run (2 tools, one batch)
        Iteration 7: LLM returns empty -> COMPLETE
    """

    @pytest.mark.asyncio
    async def test_full_cycle_completes_with_all_tools(
        self, tmp_path: Path, stub_run_pipeline: Any
    ) -> None:
        """Drive the full pipeline and verify every stage."""
        # -- Setup --
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_test_knowledge_wiki(wiki_root)

        approval_tracker = ApprovalTracker()
        question_tracker = QuestionTracker(answers={"iteration": "50"})
        notification_tracker = NotificationTracker()
        ledger = ApprovalLedger()

        registry = _build_full_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            approval_tracker=approval_tracker,
            question_tracker=question_tracker,
            notification_tracker=notification_tracker,
        )
        bridge = ToolDispatchBridge(registry=registry)

        # -- Script the LLM responses (one entry per iteration) --
        script: list[tuple[ToolCall, ...] | Any] = [
            # Iteration 1: read_wiki (read-only)
            (
                ToolCall(
                    call_id="call_read_wiki",
                    tool_name="read_wiki",
                    arguments={"query": "smoke suite staging"},
                ),
            ),
            # Iteration 2: lookup_test_spec (read-only)
            (
                ToolCall(
                    call_id="call_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "smoke_suite"},
                ),
            ),
            # Iteration 3: ask for missing 'iterations' argument
            (
                ToolCall(
                    call_id="call_ask",
                    tool_name="ask_user_question",
                    arguments={
                        "question": (
                            "How many iterations should I use for the "
                            "smoke suite?"
                        ),
                        "context": (
                            "The test spec requires --iterations but no "
                            "value was provided in the NL command."
                        ),
                    },
                ),
            ),
            # Iteration 4: propose the SSH command (CONFIRM_PROMPT)
            (
                ToolCall(
                    call_id="call_propose",
                    tool_name="propose_ssh_command",
                    arguments={
                        "command": _TEST_COMMAND,
                        "target_host": _TEST_HOST,
                        "target_user": _TEST_USER,
                        "explanation": (
                            "Running smoke_suite with 50 iterations on "
                            "staging as requested"
                        ),
                    },
                ),
            ),
            # Iteration 5: execute SSH (dynamic -- reads approval_id)
            _make_execute_ssh_calls,
            # Iteration 6: notify + summarize (2 tools in one batch)
            _make_notify_and_summarize_calls,
            # Iteration 7: no more tool calls -> COMPLETE
        ]

        llm_client = ScriptedLLMClient(script)

        # -- Build and run (stub_run_pipeline fixture handles cleanup) --
        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=10),
        )

        result = await loop.run(_NL_INPUT)

        # ===================== Assertions =====================

        # -- Terminal state --
        assert result.final_state is AgentLoopState.COMPLETE, (
            f"Expected COMPLETE, got {result.final_state.value}. "
            f"Error: {result.error_message}"
        )
        # 6 iterations with tools + 1 empty = 7 total
        assert result.iterations_used == 7
        assert result.error_message is None
        assert result.retry_exhausted is False

        # -- LLM called 7 times --
        assert llm_client.call_count == 7

        # -- 7 individual tool dispatches (6 iterations, iter 6 has 2) --
        assert bridge.dispatch_count == 7

        all_results = bridge.all_results
        assert len(all_results) == 7

        # -- Result 0: read_wiki -> SUCCESS --
        r_wiki = all_results[0]
        assert r_wiki.tool_name == "read_wiki"
        assert r_wiki.status is ToolResultStatus.SUCCESS
        wiki_data = json.loads(r_wiki.output)
        assert "query" in wiki_data

        # -- Result 1: lookup_test_spec -> SUCCESS with test spec --
        r_lookup = all_results[1]
        assert r_lookup.tool_name == "lookup_test_spec"
        assert r_lookup.status is ToolResultStatus.SUCCESS
        lookup_data = json.loads(r_lookup.output)
        assert lookup_data["found"] is True
        assert "smoke-suite" in lookup_data["test_slug"]
        assert "iterations" in lookup_data["required_args"]
        assert "env" in lookup_data["required_args"]

        # -- Result 2: ask_user_question -> SUCCESS with answer --
        r_ask = all_results[2]
        assert r_ask.tool_name == "ask_user_question"
        assert r_ask.status is ToolResultStatus.SUCCESS
        ask_data = json.loads(r_ask.output)
        assert ask_data["answer"] == "50"

        # -- Result 3: propose_ssh_command -> SUCCESS, approved --
        r_propose = all_results[3]
        assert r_propose.tool_name == "propose_ssh_command"
        assert r_propose.status is ToolResultStatus.SUCCESS
        propose_data = json.loads(r_propose.output)
        assert propose_data["approved"] is True
        assert "approval_id" in propose_data
        assert propose_data["command"] == _TEST_COMMAND

        # -- Result 4: execute_ssh -> SUCCESS with output --
        r_execute = all_results[4]
        assert r_execute.tool_name == "execute_ssh"
        assert r_execute.status is ToolResultStatus.SUCCESS
        execute_data = json.loads(r_execute.output)
        assert execute_data["success"] is True
        assert execute_data["exit_code"] == 0
        assert "50 passed" in execute_data["stdout"]

        # -- Result 5: notify_user -> SUCCESS --
        r_notify = all_results[5]
        assert r_notify.tool_name == "notify_user"
        assert r_notify.status is ToolResultStatus.SUCCESS
        notify_data = json.loads(r_notify.output)
        assert notify_data["delivered"] is True
        assert notify_data["severity"] == "success"

        # -- Result 6: summarize_run -> SUCCESS --
        r_summary = all_results[6]
        assert r_summary.tool_name == "summarize_run"
        assert r_summary.status is ToolResultStatus.SUCCESS
        summary_data = json.loads(r_summary.output)
        assert summary_data["overall_status"] == "PASSED"
        assert summary_data["passed"] == 50
        assert summary_data["failed"] == 0

        # -- Approval tracker: 1 request (propose only; execute no longer prompts) --
        assert len(approval_tracker.requests) == 1
        assert approval_tracker.requests[0][0] == _TEST_COMMAND
        assert approval_tracker.requests[0][1] == _TEST_HOST

        # -- Question tracker: 1 question about iterations --
        assert len(question_tracker.questions) == 1
        assert "iteration" in question_tracker.questions[0][0].lower()

        # -- Notification tracker: 1 success notification --
        assert len(notification_tracker.notifications) == 1
        assert "completed" in notification_tracker.notifications[0][0].lower()
        assert notification_tracker.notifications[0][1] == "success"

        # -- Approval ledger: all consumed --
        assert ledger.pending_count == 0

    @pytest.mark.asyncio
    async def test_conversation_history_structure(
        self, tmp_path: Path, stub_run_pipeline: Any
    ) -> None:
        """Verify conversation history has correct structure after run."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_test_knowledge_wiki(wiki_root)

        approval_tracker = ApprovalTracker()
        question_tracker = QuestionTracker(answers={"iteration": "50"})
        notification_tracker = NotificationTracker()
        ledger = ApprovalLedger()

        registry = _build_full_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            approval_tracker=approval_tracker,
            question_tracker=question_tracker,
            notification_tracker=notification_tracker,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...] | Any] = [
            (
                ToolCall(
                    call_id="call_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "smoke_suite"},
                ),
            ),
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
            _make_execute_ssh_calls,
            # done
        ]

        llm_client = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=10),
        )

        result = await loop.run(_NL_INPUT)

        assert result.final_state is AgentLoopState.COMPLETE

        history = result.history

        # First two messages: system + user
        assert history[0]["role"] == "system"
        assert history[0]["content"] == _SYSTEM_PROMPT
        assert history[1]["role"] == "user"
        assert history[1]["content"] == _NL_INPUT

        # All four roles should be present
        roles = {msg["role"] for msg in history}
        assert roles == {"system", "user", "assistant", "tool"}

        # Each iteration appends 1 assistant + N tool messages.
        # 3 iterations with 1 tool each = 3 assistant + 3 tool = 6 new msgs.
        # Plus the 2 initial = 8 messages total.
        assert len(history) == 8

        # Verify assistant messages have tool_calls arrays
        assistant_msgs = [m for m in history if m["role"] == "assistant"]
        assert len(assistant_msgs) == 3
        for assistant_msg in assistant_msgs:
            assert "tool_calls" in assistant_msg
            assert len(assistant_msg["tool_calls"]) >= 1

        # Verify tool messages have tool_call_id fields
        tool_msgs = [m for m in history if m["role"] == "tool"]
        assert len(tool_msgs) == 3
        for tool_msg in tool_msgs:
            assert "tool_call_id" in tool_msg

        # History grows monotonically across LLM calls
        for i in range(1, len(llm_client.messages_log)):
            prev_len = len(llm_client.messages_log[i - 1])
            curr_len = len(llm_client.messages_log[i])
            assert curr_len > prev_len, (
                f"History should grow monotonically: "
                f"call {i} had {curr_len} messages, "
                f"call {i - 1} had {prev_len} messages"
            )


# ---------------------------------------------------------------------------
# Test: Read-Only Tools Run Freely (no approval callbacks invoked)
# ---------------------------------------------------------------------------


class TestReadOnlyToolsRunFreely:
    """Read-only tools execute without triggering any approval callback."""

    @pytest.mark.asyncio
    async def test_readonly_tools_skip_approval(
        self, tmp_path: Path
    ) -> None:
        """lookup_test_spec, read_wiki, summarize_run run without approval."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_test_knowledge_wiki(wiki_root)

        approval_tracker = ApprovalTracker()
        question_tracker = QuestionTracker()
        notification_tracker = NotificationTracker()
        ledger = ApprovalLedger()

        registry = _build_full_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            approval_tracker=approval_tracker,
            question_tracker=question_tracker,
            notification_tracker=notification_tracker,
        )
        bridge = ToolDispatchBridge(registry=registry)

        # Only use read-only tools -- no propose/execute
        script: list[tuple[ToolCall, ...]] = [
            (
                ToolCall(
                    call_id="call_wiki",
                    tool_name="read_wiki",
                    arguments={"query": "smoke suite"},
                ),
            ),
            (
                ToolCall(
                    call_id="call_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "smoke_suite"},
                ),
            ),
            (
                ToolCall(
                    call_id="call_notify",
                    tool_name="notify_user",
                    arguments={
                        "message": "Checking test catalog",
                        "severity": "info",
                    },
                ),
            ),
            (
                ToolCall(
                    call_id="call_summarize",
                    tool_name="summarize_run",
                    arguments={
                        "stdout": _STUB_STDOUT,
                        "command": "pytest tests/",
                    },
                ),
            ),
            # done
        ]

        llm_client = ScriptedLLMClient(script)
        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=10),
        )

        result = await loop.run("check the smoke suite")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 5  # 4 with tools + 1 empty

        # All 4 tools returned SUCCESS
        for r in bridge.all_results:
            assert r.status is ToolResultStatus.SUCCESS, (
                f"Expected SUCCESS for {r.tool_name}, "
                f"got {r.status.value}: {r.error_message}"
            )

        # Zero approval requests -- read-only tools skip approval
        assert len(approval_tracker.requests) == 0

        # Notification was delivered
        assert len(notification_tracker.notifications) == 1


# ---------------------------------------------------------------------------
# Test: Transient LLM Error Retry Then Complete
# ---------------------------------------------------------------------------


class TestTransientRetryThenComplete:
    """Transient LLM timeout at iteration 2 retries and loop completes."""

    @pytest.mark.asyncio
    async def test_transient_timeout_retries_and_succeeds(
        self, tmp_path: Path
    ) -> None:
        """LLM raises TimeoutError once, retry succeeds, loop completes."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_test_knowledge_wiki(wiki_root)

        notification_tracker = NotificationTracker()
        registry = ToolRegistry()
        registry.register(LookupTestSpecTool(wiki_root=wiki_root))
        registry.register(
            NotifyUserTool(notify_callback=notification_tracker.notify)
        )
        bridge = ToolDispatchBridge(registry=registry)

        # Script: 2 iterations of read-only tools
        script: list[tuple[ToolCall, ...]] = [
            # Iteration 1: lookup (succeeds)
            (
                ToolCall(
                    call_id="call_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "smoke_suite"},
                ),
            ),
            # Iteration 2: will fail with transient error first, then
            # retry delivers this entry
            (
                ToolCall(
                    call_id="call_notify",
                    tool_name="notify_user",
                    arguments={
                        "message": "Lookup complete",
                        "severity": "info",
                    },
                ),
            ),
            # Iteration 3: done (empty)
        ]

        # Error on the 2nd LLM call (iteration 2), once
        llm_client = TransientErrorLLMClient(
            script,
            error_at_call=2,
            error_count=1,
            error_type=TimeoutError,
        )

        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(
                max_iterations=10,
                max_retries=2,
                retry_base_delay=0.0,  # No actual delay in tests
            ),
        )

        result = await loop.run("check smoke suite")

        # Loop should complete normally after the retry
        assert result.final_state is AgentLoopState.COMPLETE, (
            f"Expected COMPLETE, got {result.final_state.value}. "
            f"Error: {result.error_message}"
        )
        assert result.error_message is None
        assert result.retry_exhausted is False

        # LLM was called 4 times: 1 (iter 1) + 1 (error) + 1 (retry) + 1 (empty)
        assert llm_client.call_count == 4

        # Exactly 1 error was raised
        assert llm_client.total_errors_raised == 1

        # Both tools executed successfully
        assert bridge.dispatch_count == 2
        assert all(r.is_success for r in bridge.all_results)


# ---------------------------------------------------------------------------
# Test: Multiple Read-Only Tools in One Cycle
# ---------------------------------------------------------------------------


class TestMultipleReadOnlyToolsInOneCycle:
    """Two read-only tools dispatched in a single iteration."""

    @pytest.mark.asyncio
    async def test_two_tools_same_iteration(
        self, tmp_path: Path
    ) -> None:
        """LLM returns two tool calls in one batch; both execute."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_test_knowledge_wiki(wiki_root)

        notification_tracker = NotificationTracker()
        registry = ToolRegistry()
        registry.register(LookupTestSpecTool(wiki_root=wiki_root))
        registry.register(
            NotifyUserTool(notify_callback=notification_tracker.notify)
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...]] = [
            # Single iteration with 2 tool calls
            (
                ToolCall(
                    call_id="call_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "smoke_suite"},
                ),
                ToolCall(
                    call_id="call_notify",
                    tool_name="notify_user",
                    arguments={
                        "message": "Fetched test spec",
                        "severity": "info",
                    },
                ),
            ),
            # done
        ]

        llm_client = ScriptedLLMClient(script)
        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=5),
        )

        result = await loop.run("look up smoke suite")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used == 2  # 1 with tools + 1 empty

        # Both tools dispatched
        assert bridge.dispatch_count == 2
        assert bridge.all_results[0].tool_name == "lookup_test_spec"
        assert bridge.all_results[0].is_success
        assert bridge.all_results[1].tool_name == "notify_user"
        assert bridge.all_results[1].is_success

        # Both tool results appear in history
        tool_msgs = [
            m for m in result.history if m.get("role") == "tool"
        ]
        assert len(tool_msgs) == 2

        # The assistant message for that iteration has 2 tool_calls
        assistant_msgs = [
            m for m in result.history if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0]["tool_calls"]) == 2


# ---------------------------------------------------------------------------
# Test: Test Catalog Integration
# (wiki lookup -> detect missing args -> ask user)
# ---------------------------------------------------------------------------


class TestTestCatalogIntegration:
    """Agent reads wiki test spec, detects missing args, asks user."""

    @pytest.mark.asyncio
    async def test_wiki_lookup_then_ask_missing_args(
        self, tmp_path: Path
    ) -> None:
        """Wiki lookup reveals required_args; LLM asks for missing ones."""
        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        _create_test_knowledge_wiki(wiki_root)

        question_tracker = QuestionTracker(
            answers={"iteration": "50", "env": "staging"}
        )
        approval_tracker = ApprovalTracker()
        notification_tracker = NotificationTracker()
        ledger = ApprovalLedger()

        registry = _build_full_registry(
            wiki_root=wiki_root,
            ledger=ledger,
            approval_tracker=approval_tracker,
            question_tracker=question_tracker,
            notification_tracker=notification_tracker,
        )
        bridge = ToolDispatchBridge(registry=registry)

        script: list[tuple[ToolCall, ...]] = [
            # Iteration 1: lookup test spec
            (
                ToolCall(
                    call_id="call_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "smoke_suite"},
                ),
            ),
            # Iteration 2: ask for env (missing arg from test spec)
            (
                ToolCall(
                    call_id="call_ask_env",
                    tool_name="ask_user_question",
                    arguments={
                        "question": "What environment should I target?",
                        "context": "Required arg 'env' not provided",
                    },
                ),
            ),
            # Iteration 3: ask for iterations (another missing arg)
            (
                ToolCall(
                    call_id="call_ask_iter",
                    tool_name="ask_user_question",
                    arguments={
                        "question": (
                            "How many iterations for the smoke suite?"
                        ),
                        "context": "Required arg 'iterations' not provided",
                    },
                ),
            ),
            # done
        ]

        llm_client = ScriptedLLMClient(script)
        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=10),
        )

        result = await loop.run("run the smoke suite")

        assert result.final_state is AgentLoopState.COMPLETE

        # Verify test spec was found
        r_lookup = bridge.all_results[0]
        lookup_data = json.loads(r_lookup.output)
        assert lookup_data["found"] is True
        assert set(lookup_data["required_args"]) == {"env", "iterations"}

        # Verify both questions were asked
        assert len(question_tracker.questions) == 2

        # Verify env answer
        r_env = bridge.all_results[1]
        env_data = json.loads(r_env.output)
        assert env_data["answer"] == "staging"

        # Verify iterations answer
        r_iter = bridge.all_results[2]
        iter_data = json.loads(r_iter.output)
        assert iter_data["answer"] == "50"


# ---------------------------------------------------------------------------
# Test: Approval Enforcement in Happy Path
# (execute_ssh only works with prior propose_ssh_command approval)
# ---------------------------------------------------------------------------


class TestApprovalEnforcementInHappyPath:
    """Verify that execute_ssh requires a valid approval_id from ledger."""

    @pytest.mark.asyncio
    async def test_execute_with_valid_approval_succeeds(
        self, tmp_path: Path, stub_run_pipeline: Any
    ) -> None:
        """propose -> execute with correct approval_id succeeds."""
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
            (
                ToolCall(
                    call_id="call_propose",
                    tool_name="propose_ssh_command",
                    arguments={
                        "command": "echo approved-test",
                        "target_host": _TEST_HOST,
                        "target_user": _TEST_USER,
                    },
                ),
            ),
            _make_execute_ssh_calls,
            # done
        ]

        llm_client = ScriptedLLMClient(script)

        loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=bridge,
            system_prompt=_SYSTEM_PROMPT,
            config=AgentLoopConfig(max_iterations=10),
        )

        result = await loop.run("run approved test")

        assert result.final_state is AgentLoopState.COMPLETE

        # Both propose and execute succeeded
        assert bridge.all_results[0].tool_name == "propose_ssh_command"
        assert bridge.all_results[0].is_success
        assert bridge.all_results[1].tool_name == "execute_ssh"
        assert bridge.all_results[1].is_success

        # Approval was consumed from ledger
        assert ledger.pending_count == 0
