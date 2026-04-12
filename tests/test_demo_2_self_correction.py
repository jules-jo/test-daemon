"""Demo 2: Agent self-corrects a failed command (AC 7).

Validates the full self-correction flow:
    1. User says "run the integration tests"
    2. Agent looks up the test spec from the wiki catalog
    3. Agent proposes the initial SSH command -> user approves
    4. Agent executes -> command fails (non-zero exit, ConnectionError)
    5. Agent observes the error via read_output + parse_test_output
    6. Agent proposes a corrected command (installs deps first) -> user approves
    7. Agent executes the corrected command -> success
    8. Agent reads + parses output, then summarizes

Key assertions:
    - ERROR from execute_ssh is non-terminal (loop continues)
    - LLM observes the failure in conversation history
    - Corrected propose_ssh_command follows the failure
    - Second execute_ssh succeeds
    - Full tool sequence matches DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL
    - Loop terminates with COMPLETE (not ERROR)
    - Approval enforcement: two propose->execute pairs

Test strategy:
    Uses a ScriptedLLMClient that returns pre-defined tool calls per
    iteration, and a ScriptedToolDispatcher that returns pre-configured
    results per call_id. Both mock classes are self-contained and do
    not require any external services.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from jules_daemon.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopResult,
    AgentLoopState,
)
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResult,
    ToolResultStatus,
)

from tests.fixtures.demo_scenarios import (
    DEMO_2_CORRECTED_SSH_COMMAND,
    DEMO_2_CORRECTED_SUCCESS_OUTPUT,
    DEMO_2_EXPECTED_SPEC,
    DEMO_2_EXPECTED_SUMMARY,
    DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL,
    DEMO_2_FIRST_FAILURE_OUTPUT,
    DEMO_2_FIRST_SSH_COMMAND,
    DEMO_2_NL_INPUTS,
)


# ---------------------------------------------------------------------------
# Mock LLM client: scripted tool-call responses per iteration
# ---------------------------------------------------------------------------


class ScriptedLLMClient:
    """Mock LLM client that returns pre-scripted tool calls per iteration.

    Each call to get_tool_calls() pops the next batch from the queue.
    When the queue is exhausted, returns empty tuple (loop completion).

    Tracks call count and the messages it received for post-test
    assertions on conversation history propagation.
    """

    def __init__(
        self,
        responses: list[tuple[ToolCall, ...]],
    ) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self._received_messages: list[tuple[dict[str, Any], ...]] = []

    @property
    def call_count(self) -> int:
        """Number of times get_tool_calls was invoked."""
        return self._call_count

    @property
    def received_messages(self) -> list[tuple[dict[str, Any], ...]]:
        """Conversation histories received at each invocation."""
        return list(self._received_messages)

    async def get_tool_calls(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self._call_count += 1
        self._received_messages.append(messages)
        if self._responses:
            return self._responses.pop(0)
        return ()


# ---------------------------------------------------------------------------
# Mock tool dispatcher: returns pre-configured results by call_id
# ---------------------------------------------------------------------------


class ScriptedToolDispatcher:
    """Mock tool dispatcher with pre-configured results keyed by call_id.

    Falls back to a generic success result if no result is configured
    for a given call_id. Tracks all dispatched calls for sequence
    verification.
    """

    def __init__(
        self,
        results: dict[str, ToolResult],
    ) -> None:
        self._results = dict(results)
        self._dispatched: list[ToolCall] = []

    @property
    def dispatched_calls(self) -> tuple[ToolCall, ...]:
        """All tool calls that were dispatched, in order."""
        return tuple(self._dispatched)

    @property
    def dispatched_tool_names(self) -> tuple[str, ...]:
        """Tool names of all dispatched calls, in order."""
        return tuple(c.tool_name for c in self._dispatched)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        self._dispatched.append(call)
        if call.call_id in self._results:
            return self._results[call.call_id]
        # Default: generic success
        return ToolResult.success(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output=f"default success for {call.tool_name}",
        )


# ---------------------------------------------------------------------------
# Fixtures: build the scripted scenario
# ---------------------------------------------------------------------------

# Approval IDs used in the scenario
_APPROVAL_ID_1 = "approval-first-attempt"
_APPROVAL_ID_2 = "approval-corrected"


def _build_demo_2_tool_calls() -> list[tuple[ToolCall, ...]]:
    """Build the LLM tool call sequence for Demo 2.

    Returns a list of tool call batches, one per iteration.
    The sequence matches DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL:
        1. lookup_test_spec
        2. propose_ssh_command  (first attempt)
        3. execute_ssh  (first attempt -- will fail)
        4. read_output + parse_test_output  (observe failure)
        5. propose_ssh_command  (correction)
        6. execute_ssh  (retry -- will succeed)
        7. read_output + parse_test_output + summarize_run
        8. (empty -- done)
    """
    return [
        # Iteration 1: lookup the test spec
        (
            ToolCall(
                call_id="call_01_lookup",
                tool_name="lookup_test_spec",
                arguments={"test_name": "integration tests"},
            ),
        ),
        # Iteration 2: propose first SSH command
        (
            ToolCall(
                call_id="call_02_propose",
                tool_name="propose_ssh_command",
                arguments={
                    "command": DEMO_2_FIRST_SSH_COMMAND.command,
                    "target_host": "staging.example.com",
                    "target_user": "deploy",
                    "explanation": "Run the integration test suite",
                },
            ),
        ),
        # Iteration 3: execute first attempt (will fail)
        (
            ToolCall(
                call_id="call_03_execute",
                tool_name="execute_ssh",
                arguments={"approval_id": _APPROVAL_ID_1},
            ),
        ),
        # Iteration 4: observe failure (read_output + parse_test_output)
        (
            ToolCall(
                call_id="call_04_read",
                tool_name="read_output",
                arguments={"source": "session", "tool_name_filter": "execute_ssh"},
            ),
            ToolCall(
                call_id="call_05_parse",
                tool_name="parse_test_output",
                arguments={
                    "raw_output": DEMO_2_FIRST_FAILURE_OUTPUT,
                    "framework_hint": "pytest",
                },
            ),
        ),
        # Iteration 5: propose corrected command
        (
            ToolCall(
                call_id="call_06_propose_fix",
                tool_name="propose_ssh_command",
                arguments={
                    "command": DEMO_2_CORRECTED_SSH_COMMAND.command,
                    "target_host": "staging.example.com",
                    "target_user": "deploy",
                    "explanation": (
                        "Install test dependencies first to fix the "
                        "ConnectionError, then retry the integration tests"
                    ),
                },
            ),
        ),
        # Iteration 6: execute corrected command (will succeed)
        (
            ToolCall(
                call_id="call_07_execute_fix",
                tool_name="execute_ssh",
                arguments={"approval_id": _APPROVAL_ID_2},
            ),
        ),
        # Iteration 7: read + parse + summarize the success
        (
            ToolCall(
                call_id="call_08_read_success",
                tool_name="read_output",
                arguments={"source": "session", "tool_name_filter": "execute_ssh"},
            ),
            ToolCall(
                call_id="call_09_parse_success",
                tool_name="parse_test_output",
                arguments={
                    "raw_output": DEMO_2_CORRECTED_SUCCESS_OUTPUT,
                    "framework_hint": "pytest",
                },
            ),
            ToolCall(
                call_id="call_10_summarize",
                tool_name="summarize_run",
                arguments={
                    "stdout": DEMO_2_CORRECTED_SUCCESS_OUTPUT,
                    "stderr": "",
                    "command": DEMO_2_CORRECTED_SSH_COMMAND.command,
                    "exit_code": 0,
                },
            ),
        ),
        # Iteration 8: LLM returns empty -> COMPLETE
        (),
    ]


def _build_demo_2_results() -> dict[str, ToolResult]:
    """Build the pre-configured tool results for Demo 2.

    Maps call_id to ToolResult. Only calls that need specific
    results are listed; the dispatcher returns generic success
    for everything else.
    """
    spec_data = DEMO_2_EXPECTED_SPEC.to_dict()

    return {
        # lookup_test_spec: found the integration test spec
        "call_01_lookup": ToolResult.success(
            call_id="call_01_lookup",
            tool_name="lookup_test_spec",
            output=json.dumps(spec_data),
        ),
        # propose_ssh_command: user approves first attempt
        "call_02_propose": ToolResult.success(
            call_id="call_02_propose",
            tool_name="propose_ssh_command",
            output=json.dumps({
                "approved": True,
                "approval_id": _APPROVAL_ID_1,
                "command": DEMO_2_FIRST_SSH_COMMAND.command,
                "edited": False,
            }),
        ),
        # execute_ssh: FIRST ATTEMPT FAILS (non-zero exit)
        "call_03_execute": ToolResult.error(
            call_id="call_03_execute",
            tool_name="execute_ssh",
            error_message=(
                "Command exited with code 1: "
                "FAILED tests/integration/test_api.py::test_health "
                "- ConnectionError: [Errno 111] Connection refused"
            ),
            output=json.dumps({
                "success": False,
                "run_id": "run-001",
                "command": DEMO_2_FIRST_SSH_COMMAND.command,
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "exit_code": 1,
                "stdout": DEMO_2_FIRST_FAILURE_OUTPUT,
                "stderr": "",
                "error": "Command exited with code 1",
                "duration_seconds": 2.34,
            }),
        ),
        # read_output: session history shows the failure
        "call_04_read": ToolResult.success(
            call_id="call_04_read",
            tool_name="read_output",
            output=json.dumps({
                "source": "session",
                "total_tool_results": 3,
                "returned_count": 1,
                "last_n": 10,
                "tool_name_filter": "execute_ssh",
                "entries": [{
                    "tool_call_id": "call_03_execute",
                    "tool_name": "execute_ssh",
                    "arguments": {"approval_id": _APPROVAL_ID_1},
                    "is_error": True,
                    "content": (
                        "ERROR: Command exited with code 1: "
                        "FAILED tests/integration/test_api.py::test_health"
                    ),
                }],
            }),
        ),
        # parse_test_output: parsed failure result
        "call_05_parse": ToolResult.success(
            call_id="call_05_parse",
            tool_name="parse_test_output",
            output=json.dumps({
                "records": [{
                    "name": "test_health",
                    "status": "failed",
                    "module": "tests/integration/test_api.py",
                }],
                "truncated": False,
                "framework": "pytest",
                "total_lines_parsed": 2,
                "summary": {
                    "passed": 0,
                    "failed": 1,
                    "skipped": 0,
                    "error": 0,
                    "incomplete": 0,
                },
            }),
        ),
        # propose_ssh_command: user approves corrected command
        "call_06_propose_fix": ToolResult.success(
            call_id="call_06_propose_fix",
            tool_name="propose_ssh_command",
            output=json.dumps({
                "approved": True,
                "approval_id": _APPROVAL_ID_2,
                "command": DEMO_2_CORRECTED_SSH_COMMAND.command,
                "edited": False,
            }),
        ),
        # execute_ssh: CORRECTED COMMAND SUCCEEDS
        "call_07_execute_fix": ToolResult.success(
            call_id="call_07_execute_fix",
            tool_name="execute_ssh",
            output=json.dumps({
                "success": True,
                "run_id": "run-002",
                "command": DEMO_2_CORRECTED_SSH_COMMAND.command,
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "exit_code": 0,
                "stdout": DEMO_2_CORRECTED_SUCCESS_OUTPUT,
                "stderr": "",
                "error": None,
                "duration_seconds": 12.56,
            }),
        ),
        # read_output: session history now shows the success
        "call_08_read_success": ToolResult.success(
            call_id="call_08_read_success",
            tool_name="read_output",
            output=json.dumps({
                "source": "session",
                "total_tool_results": 5,
                "returned_count": 2,
                "last_n": 10,
                "tool_name_filter": "execute_ssh",
                "entries": [
                    {
                        "tool_call_id": "call_03_execute",
                        "tool_name": "execute_ssh",
                        "is_error": True,
                        "content": "ERROR: Command exited with code 1",
                    },
                    {
                        "tool_call_id": "call_07_execute_fix",
                        "tool_name": "execute_ssh",
                        "is_error": False,
                        "content": json.dumps({"success": True, "exit_code": 0}),
                    },
                ],
            }),
        ),
        # parse_test_output: parsed success result
        "call_09_parse_success": ToolResult.success(
            call_id="call_09_parse_success",
            tool_name="parse_test_output",
            output=json.dumps({
                "records": [
                    {"name": "test_health", "status": "passed", "module": "tests/integration/test_api.py"},
                    {"name": "test_create", "status": "passed", "module": "tests/integration/test_api.py"},
                    {"name": "test_list", "status": "passed", "module": "tests/integration/test_api.py"},
                ],
                "truncated": False,
                "framework": "pytest",
                "total_lines_parsed": 4,
                "summary": {
                    "passed": 3,
                    "failed": 0,
                    "skipped": 0,
                    "error": 0,
                    "incomplete": 0,
                },
            }),
        ),
        # summarize_run: final summary
        "call_10_summarize": ToolResult.success(
            call_id="call_10_summarize",
            tool_name="summarize_run",
            output=json.dumps({
                "parser": "pytest",
                "passed": DEMO_2_EXPECTED_SUMMARY.passed,
                "failed": DEMO_2_EXPECTED_SUMMARY.failed,
                "skipped": DEMO_2_EXPECTED_SUMMARY.skipped,
                "total": DEMO_2_EXPECTED_SUMMARY.passed,
                "duration_seconds": 12.56,
                "key_failures": [],
                "narrative": (
                    "All 3 integration tests passed after installing "
                    "test dependencies. The initial ConnectionError "
                    "was resolved by pip install -r requirements-test.txt."
                ),
                "raw_excerpt": DEMO_2_CORRECTED_SUCCESS_OUTPUT[:200],
            }),
        ),
    }


@pytest.fixture
def demo_2_llm() -> ScriptedLLMClient:
    """Scripted LLM client for Demo 2."""
    return ScriptedLLMClient(_build_demo_2_tool_calls())


@pytest.fixture
def demo_2_dispatcher() -> ScriptedToolDispatcher:
    """Scripted tool dispatcher for Demo 2."""
    return ScriptedToolDispatcher(_build_demo_2_results())


@pytest.fixture
def demo_2_config() -> AgentLoopConfig:
    """Agent loop config with enough iterations for Demo 2."""
    return AgentLoopConfig(max_iterations=10, max_retries=2)


# ---------------------------------------------------------------------------
# Core self-correction flow
# ---------------------------------------------------------------------------


class TestDemo2SelfCorrectionFlow:
    """Demo 2: Agent self-corrects a failed command.

    The complete flow:
        NL input -> lookup_test_spec -> propose_ssh_command -> execute_ssh (FAIL)
        -> read_output + parse_test_output (observe error)
        -> propose_ssh_command (correction) -> execute_ssh (SUCCESS)
        -> read_output + parse_test_output + summarize_run -> COMPLETE
    """

    @pytest.mark.asyncio
    async def test_full_self_correction_completes(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """The agent loop completes successfully after self-correction."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        result = await loop.run(DEMO_2_NL_INPUTS[0])

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_tool_sequence_matches_expected(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """Dispatched tool names match DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        await loop.run(DEMO_2_NL_INPUTS[0])

        assert (
            demo_2_dispatcher.dispatched_tool_names
            == DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL
        )

    @pytest.mark.asyncio
    async def test_two_propose_execute_pairs(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """The agent must issue exactly two propose->execute cycles."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        await loop.run(DEMO_2_NL_INPUTS[0])

        names = demo_2_dispatcher.dispatched_tool_names
        propose_count = names.count("propose_ssh_command")
        execute_count = names.count("execute_ssh")
        assert propose_count == 2, f"Expected 2 proposals, got {propose_count}"
        assert execute_count == 2, f"Expected 2 executions, got {execute_count}"

    @pytest.mark.asyncio
    async def test_iterations_used(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """Loop uses 8 iterations (7 active + 1 empty completion)."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        result = await loop.run(DEMO_2_NL_INPUTS[0])

        # 7 iterations with tool calls + 1 empty = 8 total
        assert result.iterations_used == 8

    @pytest.mark.asyncio
    async def test_all_ten_tools_dispatched(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """Exactly 10 tool calls are dispatched across all iterations."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        await loop.run(DEMO_2_NL_INPUTS[0])

        assert len(demo_2_dispatcher.dispatched_calls) == 10


# ---------------------------------------------------------------------------
# Error observation (non-terminal ERROR continues the loop)
# ---------------------------------------------------------------------------


class TestDemo2ErrorObservation:
    """Verify that execute_ssh ERROR status is non-terminal and observable."""

    @pytest.mark.asyncio
    async def test_first_execute_returns_error_status(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """The first execute_ssh returns ERROR (non-terminal)."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        await loop.run(DEMO_2_NL_INPUTS[0])

        # Find the first execute_ssh call
        first_execute = None
        for call in demo_2_dispatcher.dispatched_calls:
            if call.tool_name == "execute_ssh":
                first_execute = call
                break

        assert first_execute is not None
        assert first_execute.call_id == "call_03_execute"

    @pytest.mark.asyncio
    async def test_error_visible_in_history(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """The ERROR from execute_ssh appears in the conversation history."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        result = await loop.run(DEMO_2_NL_INPUTS[0])

        # Find tool messages in history with ERROR content
        error_tool_messages = [
            msg
            for msg in result.history
            if msg.get("role") == "tool"
            and isinstance(msg.get("content", ""), str)
            and msg["content"].startswith("ERROR:")
        ]

        assert len(error_tool_messages) >= 1
        # The error message should mention the ConnectionError
        error_content = error_tool_messages[0]["content"]
        assert "ConnectionError" in error_content or "exited with code 1" in error_content

    @pytest.mark.asyncio
    async def test_llm_sees_error_before_correction(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """LLM receives the error in conversation before proposing correction.

        After the first execute_ssh fails (iteration 3), the error is
        appended to the conversation. The LLM's 4th call should see this
        error in its messages.
        """
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        await loop.run(DEMO_2_NL_INPUTS[0])

        # The LLM is called 8 times (iterations 1-8).
        # After iteration 3 (execute_ssh fails), iteration 4 messages
        # should contain the error.
        assert demo_2_llm.call_count == 8

        # Iteration 4 messages (index 3, 0-based)
        iteration_4_messages = demo_2_llm.received_messages[3]

        # Find tool messages with ERROR prefix
        error_in_messages = any(
            msg.get("role") == "tool"
            and isinstance(msg.get("content", ""), str)
            and msg["content"].startswith("ERROR:")
            for msg in iteration_4_messages
        )
        assert error_in_messages, (
            "LLM must see the execute_ssh error in its conversation "
            "history before the self-correction iteration"
        )

    @pytest.mark.asyncio
    async def test_loop_does_not_terminate_on_tool_error(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """A tool ERROR (not DENIED) must not terminate the agent loop.

        The loop should continue to the next iteration where the LLM
        can observe the error and propose a correction.
        """
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        result = await loop.run(DEMO_2_NL_INPUTS[0])

        # Loop must complete, NOT error out
        assert result.final_state is AgentLoopState.COMPLETE
        # Must have dispatched more calls after the error
        assert len(demo_2_dispatcher.dispatched_calls) > 3


# ---------------------------------------------------------------------------
# Approval enforcement: propose before execute
# ---------------------------------------------------------------------------


class TestDemo2ApprovalEnforcement:
    """Verify that every execute_ssh is preceded by propose_ssh_command."""

    @pytest.mark.asyncio
    async def test_each_execute_preceded_by_propose(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """Every execute_ssh call must be preceded by a propose_ssh_command."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        await loop.run(DEMO_2_NL_INPUTS[0])

        names = demo_2_dispatcher.dispatched_tool_names

        # Find all execute_ssh indices and verify each has a prior propose
        execute_indices = [
            i for i, name in enumerate(names) if name == "execute_ssh"
        ]
        propose_indices = [
            i for i, name in enumerate(names) if name == "propose_ssh_command"
        ]

        assert len(execute_indices) == 2
        assert len(propose_indices) == 2

        # First execute must come after first propose
        assert propose_indices[0] < execute_indices[0]
        # Second execute must come after second propose
        assert propose_indices[1] < execute_indices[1]

    @pytest.mark.asyncio
    async def test_first_execute_uses_first_approval(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """First execute_ssh references the first approval_id."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        await loop.run(DEMO_2_NL_INPUTS[0])

        execute_calls = [
            c for c in demo_2_dispatcher.dispatched_calls
            if c.tool_name == "execute_ssh"
        ]
        assert len(execute_calls) == 2
        assert execute_calls[0].arguments["approval_id"] == _APPROVAL_ID_1

    @pytest.mark.asyncio
    async def test_second_execute_uses_second_approval(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """Second (corrected) execute_ssh references the second approval_id."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        await loop.run(DEMO_2_NL_INPUTS[0])

        execute_calls = [
            c for c in demo_2_dispatcher.dispatched_calls
            if c.tool_name == "execute_ssh"
        ]
        assert len(execute_calls) == 2
        assert execute_calls[1].arguments["approval_id"] == _APPROVAL_ID_2


# ---------------------------------------------------------------------------
# Corrected command differs from original
# ---------------------------------------------------------------------------


class TestDemo2CommandCorrection:
    """Verify the corrected command differs from the original."""

    @pytest.mark.asyncio
    async def test_corrected_command_differs_from_first(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """The corrected propose_ssh_command uses a different command string."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        await loop.run(DEMO_2_NL_INPUTS[0])

        propose_calls = [
            c for c in demo_2_dispatcher.dispatched_calls
            if c.tool_name == "propose_ssh_command"
        ]
        assert len(propose_calls) == 2

        first_cmd = propose_calls[0].arguments["command"]
        second_cmd = propose_calls[1].arguments["command"]

        assert first_cmd != second_cmd
        assert first_cmd == DEMO_2_FIRST_SSH_COMMAND.command
        assert second_cmd == DEMO_2_CORRECTED_SSH_COMMAND.command

    @pytest.mark.asyncio
    async def test_corrected_command_includes_dependency_install(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """The corrected command includes 'pip install' to fix dependencies."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        await loop.run(DEMO_2_NL_INPUTS[0])

        propose_calls = [
            c for c in demo_2_dispatcher.dispatched_calls
            if c.tool_name == "propose_ssh_command"
        ]
        corrected_cmd = propose_calls[1].arguments["command"]
        assert "pip install" in corrected_cmd


# ---------------------------------------------------------------------------
# History structure after self-correction
# ---------------------------------------------------------------------------


class TestDemo2HistoryStructure:
    """Verify conversation history reflects the self-correction flow."""

    @pytest.mark.asyncio
    async def test_history_starts_with_system_and_user(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """History starts with system prompt and user message."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        result = await loop.run(DEMO_2_NL_INPUTS[0])

        assert result.history[0]["role"] == "system"
        assert result.history[0]["content"] == "You are a test runner assistant."
        assert result.history[1]["role"] == "user"
        assert result.history[1]["content"] == DEMO_2_NL_INPUTS[0]

    @pytest.mark.asyncio
    async def test_history_contains_assistant_and_tool_roles(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """History includes both assistant (tool_calls) and tool (results) messages."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        result = await loop.run(DEMO_2_NL_INPUTS[0])

        roles = [msg["role"] for msg in result.history]
        assert "assistant" in roles
        assert "tool" in roles

    @pytest.mark.asyncio
    async def test_history_has_error_and_success_tool_messages(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """History contains both ERROR and SUCCESS tool result messages."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        result = await loop.run(DEMO_2_NL_INPUTS[0])

        tool_messages = [
            msg for msg in result.history
            if msg.get("role") == "tool"
        ]

        # Should have error messages (from first execute_ssh)
        error_msgs = [
            m for m in tool_messages
            if isinstance(m.get("content", ""), str)
            and m["content"].startswith("ERROR:")
        ]
        assert len(error_msgs) >= 1

        # Should have success messages (from corrected execute_ssh)
        success_msgs = [
            m for m in tool_messages
            if isinstance(m.get("content", ""), str)
            and not m["content"].startswith("ERROR:")
        ]
        assert len(success_msgs) >= 1

    @pytest.mark.asyncio
    async def test_history_is_immutable_tuple(
        self,
        demo_2_llm: ScriptedLLMClient,
        demo_2_dispatcher: ScriptedToolDispatcher,
        demo_2_config: AgentLoopConfig,
    ) -> None:
        """Returned history is a tuple (immutable)."""
        loop = AgentLoop(
            llm_client=demo_2_llm,
            tool_dispatcher=demo_2_dispatcher,
            system_prompt="You are a test runner assistant.",
            config=demo_2_config,
        )

        result = await loop.run(DEMO_2_NL_INPUTS[0])

        assert isinstance(result.history, tuple)


# ---------------------------------------------------------------------------
# Max iterations guard with self-correction
# ---------------------------------------------------------------------------


class TestDemo2IterationBudget:
    """Verify self-correction stays within iteration budget."""

    @pytest.mark.asyncio
    async def test_fails_if_max_iterations_too_low(self) -> None:
        """If max_iterations is too low, the loop cannot complete correction."""
        llm = ScriptedLLMClient(_build_demo_2_tool_calls())
        dispatcher = ScriptedToolDispatcher(_build_demo_2_results())

        # Only 3 iterations -- not enough for self-correction
        config = AgentLoopConfig(max_iterations=3, max_retries=2)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="You are a test runner assistant.",
            config=config,
        )

        result = await loop.run(DEMO_2_NL_INPUTS[0])

        # Should hit max iterations before completing
        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 3
        assert result.error_message is not None
        assert "max iterations" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_succeeds_with_sufficient_iterations(self) -> None:
        """With enough iterations, the full self-correction flow completes."""
        llm = ScriptedLLMClient(_build_demo_2_tool_calls())
        dispatcher = ScriptedToolDispatcher(_build_demo_2_results())

        # 10 iterations is sufficient for the 8 needed
        config = AgentLoopConfig(max_iterations=10, max_retries=2)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="You are a test runner assistant.",
            config=config,
        )

        result = await loop.run(DEMO_2_NL_INPUTS[0])

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.iterations_used <= 10


# ---------------------------------------------------------------------------
# NL input variations
# ---------------------------------------------------------------------------


class TestDemo2NLInputVariations:
    """Verify the flow works with all Demo 2 NL input phrases."""

    @pytest.mark.parametrize(
        "nl_input",
        DEMO_2_NL_INPUTS,
        ids=[f"phrase_{i}" for i in range(len(DEMO_2_NL_INPUTS))],
    )
    @pytest.mark.asyncio
    async def test_all_nl_phrases_complete(self, nl_input: str) -> None:
        """Each NL input phrase should drive the same self-correction flow."""
        llm = ScriptedLLMClient(_build_demo_2_tool_calls())
        dispatcher = ScriptedToolDispatcher(_build_demo_2_results())
        config = AgentLoopConfig(max_iterations=10, max_retries=2)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="You are a test runner assistant.",
            config=config,
        )

        result = await loop.run(nl_input)

        assert result.final_state is AgentLoopState.COMPLETE
        assert (
            dispatcher.dispatched_tool_names
            == DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL
        )


# ---------------------------------------------------------------------------
# Edge case: user denies the corrected command
# ---------------------------------------------------------------------------


class TestDemo2DenialDuringCorrection:
    """Verify loop terminates if user denies the corrected command."""

    @pytest.mark.asyncio
    async def test_denial_on_correction_terminates(self) -> None:
        """If user denies the corrected propose_ssh_command, loop stops."""
        # Build tool calls: same as Demo 2 but only up to correction proposal
        tool_calls = [
            # Iteration 1: lookup
            (
                ToolCall(
                    call_id="d_01_lookup",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "integration tests"},
                ),
            ),
            # Iteration 2: first propose (approved)
            (
                ToolCall(
                    call_id="d_02_propose",
                    tool_name="propose_ssh_command",
                    arguments={
                        "command": DEMO_2_FIRST_SSH_COMMAND.command,
                        "target_host": "staging.example.com",
                        "target_user": "deploy",
                        "explanation": "Run integration tests",
                    },
                ),
            ),
            # Iteration 3: execute (fails)
            (
                ToolCall(
                    call_id="d_03_execute",
                    tool_name="execute_ssh",
                    arguments={"approval_id": "approval-first"},
                ),
            ),
            # Iteration 4: propose correction (will be DENIED)
            (
                ToolCall(
                    call_id="d_04_propose_fix",
                    tool_name="propose_ssh_command",
                    arguments={
                        "command": DEMO_2_CORRECTED_SSH_COMMAND.command,
                        "target_host": "staging.example.com",
                        "target_user": "deploy",
                        "explanation": "Fix deps and retry",
                    },
                ),
            ),
        ]

        results = {
            "d_01_lookup": ToolResult.success(
                call_id="d_01_lookup",
                tool_name="lookup_test_spec",
                output=json.dumps(DEMO_2_EXPECTED_SPEC.to_dict()),
            ),
            "d_02_propose": ToolResult.success(
                call_id="d_02_propose",
                tool_name="propose_ssh_command",
                output=json.dumps({
                    "approved": True,
                    "approval_id": "approval-first",
                    "command": DEMO_2_FIRST_SSH_COMMAND.command,
                    "edited": False,
                }),
            ),
            "d_03_execute": ToolResult.error(
                call_id="d_03_execute",
                tool_name="execute_ssh",
                error_message="Command exited with code 1",
                output=json.dumps({
                    "success": False,
                    "exit_code": 1,
                    "stdout": DEMO_2_FIRST_FAILURE_OUTPUT,
                }),
            ),
            # User DENIES the corrected command
            "d_04_propose_fix": ToolResult.denied(
                call_id="d_04_propose_fix",
                tool_name="propose_ssh_command",
                error_message="User denied the corrected command",
            ),
        }

        llm = ScriptedLLMClient(tool_calls)
        dispatcher = ScriptedToolDispatcher(results)
        config = AgentLoopConfig(max_iterations=10, max_retries=2)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="You are a test runner assistant.",
            config=config,
        )

        result = await loop.run(DEMO_2_NL_INPUTS[0])

        # Loop must terminate with ERROR (denial is permanent)
        assert result.final_state is AgentLoopState.ERROR
        assert result.error_message is not None
        assert "denied" in result.error_message.lower()
        # Should have stopped at iteration 4
        assert result.iterations_used == 4


# ---------------------------------------------------------------------------
# Edge case: both attempts fail
# ---------------------------------------------------------------------------


class TestDemo2DoubleFailure:
    """Verify behavior when both original and corrected commands fail."""

    @pytest.mark.asyncio
    async def test_double_failure_hits_max_iterations(self) -> None:
        """If both attempts fail and LLM keeps trying, max iterations kicks in."""
        # Build a scenario where both execute_ssh calls return ERROR
        tool_calls = [
            (ToolCall(call_id="df_01", tool_name="lookup_test_spec",
                      arguments={"test_name": "integration tests"}),),
            (ToolCall(call_id="df_02", tool_name="propose_ssh_command",
                      arguments={"command": "cmd1", "target_host": "h", "target_user": "u"}),),
            (ToolCall(call_id="df_03", tool_name="execute_ssh",
                      arguments={"approval_id": "a1"}),),
            (ToolCall(call_id="df_04", tool_name="propose_ssh_command",
                      arguments={"command": "cmd2", "target_host": "h", "target_user": "u"}),),
            (ToolCall(call_id="df_05", tool_name="execute_ssh",
                      arguments={"approval_id": "a2"}),),
            # LLM still trying...
            (ToolCall(call_id="df_06", tool_name="propose_ssh_command",
                      arguments={"command": "cmd3", "target_host": "h", "target_user": "u"}),),
        ]

        results = {
            "df_01": ToolResult.success(
                call_id="df_01", tool_name="lookup_test_spec",
                output=json.dumps(DEMO_2_EXPECTED_SPEC.to_dict()),
            ),
            "df_02": ToolResult.success(
                call_id="df_02", tool_name="propose_ssh_command",
                output=json.dumps({"approved": True, "approval_id": "a1", "command": "cmd1"}),
            ),
            "df_03": ToolResult.error(
                call_id="df_03", tool_name="execute_ssh",
                error_message="Command exited with code 1",
            ),
            "df_04": ToolResult.success(
                call_id="df_04", tool_name="propose_ssh_command",
                output=json.dumps({"approved": True, "approval_id": "a2", "command": "cmd2"}),
            ),
            "df_05": ToolResult.error(
                call_id="df_05", tool_name="execute_ssh",
                error_message="Command exited with code 2 (different error)",
            ),
            "df_06": ToolResult.success(
                call_id="df_06", tool_name="propose_ssh_command",
                output=json.dumps({"approved": True, "approval_id": "a3", "command": "cmd3"}),
            ),
        }

        llm = ScriptedLLMClient(tool_calls)
        dispatcher = ScriptedToolDispatcher(results)
        # Only 5 iterations allowed (default)
        config = AgentLoopConfig(max_iterations=5, max_retries=2)

        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="You are a test runner assistant.",
            config=config,
        )

        result = await loop.run(DEMO_2_NL_INPUTS[0])

        # Should hit max iterations since both attempts failed
        assert result.final_state is AgentLoopState.ERROR
        assert result.iterations_used == 5
        assert "max iterations" in result.error_message.lower()


# ---------------------------------------------------------------------------
# Fixture consistency checks
# ---------------------------------------------------------------------------


class TestDemo2FixtureConsistency:
    """Verify Demo 2 fixture data is internally consistent."""

    def test_first_command_matches_fixture(self) -> None:
        """First SSH command matches DEMO_2_FIRST_SSH_COMMAND."""
        calls = _build_demo_2_tool_calls()
        # Iteration 2 (index 1) is the first propose_ssh_command
        propose_call = calls[1][0]
        assert propose_call.arguments["command"] == DEMO_2_FIRST_SSH_COMMAND.command

    def test_corrected_command_matches_fixture(self) -> None:
        """Corrected SSH command matches DEMO_2_CORRECTED_SSH_COMMAND."""
        calls = _build_demo_2_tool_calls()
        # Iteration 5 (index 4) is the correction propose_ssh_command
        propose_call = calls[4][0]
        assert propose_call.arguments["command"] == DEMO_2_CORRECTED_SSH_COMMAND.command

    def test_first_failure_output_matches_fixture(self) -> None:
        """First execute error output contains DEMO_2_FIRST_FAILURE_OUTPUT."""
        results = _build_demo_2_results()
        first_execute_result = results["call_03_execute"]
        result_data = json.loads(first_execute_result.output)
        assert result_data["stdout"] == DEMO_2_FIRST_FAILURE_OUTPUT

    def test_corrected_success_output_matches_fixture(self) -> None:
        """Corrected execute output contains DEMO_2_CORRECTED_SUCCESS_OUTPUT."""
        results = _build_demo_2_results()
        second_execute_result = results["call_07_execute_fix"]
        result_data = json.loads(second_execute_result.output)
        assert result_data["stdout"] == DEMO_2_CORRECTED_SUCCESS_OUTPUT

    def test_summary_counts_match_fixture(self) -> None:
        """Summary result matches DEMO_2_EXPECTED_SUMMARY counts."""
        results = _build_demo_2_results()
        summary_result = results["call_10_summarize"]
        summary_data = json.loads(summary_result.output)
        assert summary_data["passed"] == DEMO_2_EXPECTED_SUMMARY.passed
        assert summary_data["failed"] == DEMO_2_EXPECTED_SUMMARY.failed
        assert summary_data["skipped"] == DEMO_2_EXPECTED_SUMMARY.skipped

    def test_expected_tool_sequence_length(self) -> None:
        """DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL has exactly 10 entries."""
        assert len(DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL) == 10

    def test_expected_tool_sequence_has_two_propose_execute_pairs(self) -> None:
        """Sequence has 2 propose and 2 execute calls."""
        seq = DEMO_2_EXPECTED_TOOL_SEQUENCE_FULL
        assert seq.count("propose_ssh_command") == 2
        assert seq.count("execute_ssh") == 2
