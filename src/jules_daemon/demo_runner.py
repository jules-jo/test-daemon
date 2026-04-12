"""Interactive demo runner for the agent loop named-test flow.

Executes Demo 1 (named-test end-to-end) from the demo scenarios with
real human approval prompts on stdin/stdout and simulated SSH execution.
Prints each agent-loop stage transition to stdout for observability.

The demo wires up:
    - A scripted LLM client that replays the Demo 1 tool call sequence
    - Real ToolRegistry with all 10 tools registered
    - Interactive terminal-based approval prompts (stdin/stdout)
    - Simulated SSH execution (no real SSH connection required)
    - Stage transition logging to stdout

Usage::

    python -m jules_daemon.demo_runner
    python -m jules_daemon.demo_runner --max-iterations 10
    python -m jules_daemon.demo_runner --scenario demo1

Press Ctrl+C at any time to cancel the demo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from jules_daemon.agent.tools.base import BaseTool
from jules_daemon.agent.tools.execute_ssh import ExecuteSSHTool
from jules_daemon.agent.tools.notify_user import NotifyUserTool
from jules_daemon.agent.tools.parse_test_output import ParseTestOutputTool
from jules_daemon.agent.tools.propose_ssh_command import (
    ApprovalLedger,
    ProposeSSHCommandTool,
)
from jules_daemon.agent.tools.lookup_test_spec import LookupTestSpecTool
from jules_daemon.agent.tools.summarize_run import SummarizeRunTool


# ---------------------------------------------------------------------------
# Terminal formatting helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"

# State-to-color mapping for stage transitions
_STATE_COLORS: dict[str, str] = {
    "thinking": _CYAN,
    "acting": _YELLOW,
    "observing": _MAGENTA,
    "complete": _GREEN,
    "error": _RED,
}


def _print_banner() -> None:
    """Print the demo runner banner."""
    print(f"\n{_BOLD}{'=' * 70}{_RESET}")
    print(f"{_BOLD}{_CYAN}  Jules Agent Loop -- Interactive Demo Runner{_RESET}")
    print(f"{_BOLD}{'=' * 70}{_RESET}")
    print(f"{_DIM}  Demo 1: Named test end-to-end flow{_RESET}")
    print(f"{_DIM}  Press Ctrl+C at any time to cancel{_RESET}")
    print(f"{_BOLD}{'=' * 70}{_RESET}\n")


def _print_stage(state: str, message: str) -> None:
    """Print a stage transition with color coding."""
    color = _STATE_COLORS.get(state, _RESET)
    label = state.upper()
    print(f"  {color}{_BOLD}[{label}]{_RESET} {message}")


def _print_tool_call(call: ToolCall) -> None:
    """Print a tool call with formatted arguments."""
    print(f"    {_BLUE}-> {_BOLD}{call.tool_name}{_RESET}", end="")
    if call.arguments:
        # Truncate long argument values for readability
        short_args = {}
        for key, val in call.arguments.items():
            if key.startswith("_"):
                continue
            str_val = str(val)
            if len(str_val) > 80:
                short_args[key] = str_val[:77] + "..."
            else:
                short_args[key] = val
        if short_args:
            print(f"({json.dumps(short_args, default=str)})", end="")
    print()


def _print_tool_result(result: ToolResult) -> None:
    """Print a tool result with status coloring."""
    if result.is_success:
        status_str = f"{_GREEN}SUCCESS{_RESET}"
    elif result.is_denied:
        status_str = f"{_RED}DENIED{_RESET}"
    else:
        status_str = f"{_YELLOW}ERROR{_RESET}"

    print(f"    {_DIM}<-{_RESET} {result.tool_name}: {status_str}", end="")

    # Show a brief excerpt of the output
    if result.output:
        try:
            data = json.loads(result.output)
            # Show key fields depending on the tool
            excerpt_parts: list[str] = []
            for key in ("found", "approved", "approval_id", "exit_code",
                        "answer", "passed", "failed", "delivered"):
                if key in data:
                    excerpt_parts.append(f"{key}={data[key]}")
            if excerpt_parts:
                print(f" ({', '.join(excerpt_parts)})", end="")
        except (json.JSONDecodeError, TypeError):
            truncated = result.output[:60]
            if len(result.output) > 60:
                truncated += "..."
            print(f" ({truncated})", end="")

    if result.error_message:
        print(f" [{result.error_message[:60]}]", end="")

    print()


def _print_separator() -> None:
    """Print a visual separator between iterations."""
    print(f"  {_DIM}{'- ' * 35}{_RESET}")


def _print_result(result: AgentLoopResult) -> None:
    """Print the final agent loop result summary."""
    print(f"\n{_BOLD}{'=' * 70}{_RESET}")

    if result.final_state is AgentLoopState.COMPLETE:
        color = _GREEN
        label = "COMPLETED"
    else:
        color = _RED
        label = "ERROR"

    print(f"  {color}{_BOLD}Agent Loop {label}{_RESET}")
    print(f"  Iterations used: {result.iterations_used}")

    if result.error_message:
        print(f"  Error: {_RED}{result.error_message}{_RESET}")

    # Count tool calls in history
    tool_calls_count = sum(
        1 for msg in result.history
        if msg.get("role") == "tool"
    )
    print(f"  Tool results in history: {tool_calls_count}")
    print(f"{_BOLD}{'=' * 70}{_RESET}\n")


# ---------------------------------------------------------------------------
# Interactive terminal callbacks (real human approval)
# ---------------------------------------------------------------------------


async def _interactive_confirm(
    command: str,
    target_host: str,
    explanation: str,
) -> tuple[bool, str]:
    """Interactive confirmation prompt via stdin/stdout.

    Displays the proposed command and waits for the user to approve,
    deny, or edit it. This is a real human approval flow -- not mocked.

    Args:
        command: The proposed SSH command.
        target_host: The remote host.
        explanation: Why the command is proposed.

    Returns:
        Tuple of (approved, final_command).
    """
    print()
    print(f"  {_BOLD}{_YELLOW}--- APPROVAL REQUIRED ---{_RESET}")
    if explanation:
        print(f"  {_DIM}{explanation}{_RESET}")
    print(f"  Host:    {_BOLD}{target_host}{_RESET}")
    print(f"  Command: {_BOLD}{command}{_RESET}")
    print()
    print(f"  {_CYAN}[y]{_RESET} Approve  "
          f"{_RED}[n]{_RESET} Deny  "
          f"{_YELLOW}[e]{_RESET} Edit command")

    while True:
        try:
            response = await asyncio.to_thread(
                input, f"  {_BOLD}Your choice [y/n/e]: {_RESET}"
            )
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_RED}Cancelled{_RESET}")
            return (False, command)

        response = response.strip().lower()

        if response in ("y", "yes"):
            print(f"  {_GREEN}Approved{_RESET}")
            return (True, command)

        if response in ("n", "no"):
            print(f"  {_RED}Denied{_RESET}")
            return (False, command)

        if response in ("e", "edit"):
            try:
                edited = await asyncio.to_thread(
                    input, f"  {_BOLD}Enter edited command: {_RESET}"
                )
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {_RED}Cancelled{_RESET}")
                return (False, command)

            edited = edited.strip()
            if edited:
                print(f"  {_GREEN}Approved (edited){_RESET}")
                return (True, edited)
            print(f"  {_DIM}Empty edit, using original{_RESET}")
            print(f"  {_GREEN}Approved{_RESET}")
            return (True, command)

        print(f"  {_DIM}Please enter y, n, or e{_RESET}")


async def _interactive_ask(question: str, context: str) -> str | None:
    """Interactive question prompt via stdin/stdout.

    Args:
        question: The question to ask.
        context: Context for the question.

    Returns:
        User's answer, or None if cancelled.
    """
    print()
    print(f"  {_BOLD}{_CYAN}--- QUESTION ---{_RESET}")
    if context:
        print(f"  {_DIM}{context}{_RESET}")
    print(f"  {question}")

    try:
        answer = await asyncio.to_thread(
            input, f"  {_BOLD}Your answer (empty to cancel): {_RESET}"
        )
    except (EOFError, KeyboardInterrupt):
        print(f"\n  {_RED}Cancelled{_RESET}")
        return None

    answer = answer.strip()
    if not answer:
        print(f"  {_DIM}Cancelled{_RESET}")
        return None

    print(f"  {_GREEN}Answered: {answer}{_RESET}")
    return answer


async def _interactive_notify(message: str, severity: str = "info") -> bool:
    """Print a notification to stdout.

    Args:
        message: Notification message.
        severity: Severity level.

    Returns:
        Always True (delivery is stdout).
    """
    severity_colors = {
        "info": _CYAN,
        "warning": _YELLOW,
        "error": _RED,
        "success": _GREEN,
    }
    color = severity_colors.get(severity, _RESET)
    print(f"  {color}[NOTIFY:{severity.upper()}]{_RESET} {message}")
    return True


# ---------------------------------------------------------------------------
# Simulated wiki/SSH infrastructure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SimulatedRunResult:
    """Simulated result from SSH command execution."""

    success: bool
    run_id: str
    command: str
    target_host: str
    target_user: str
    exit_code: int
    stdout: str
    stderr: str
    error: str | None
    duration_seconds: float


def _write_test_knowledge(
    wiki_root: Path,
    *,
    slug: str,
    command_pattern: str,
    purpose: str,
    required_args: tuple[str, ...],
    common_failures: tuple[str, ...],
    runs_observed: int,
) -> None:
    """Write a test knowledge wiki file at the path expected by the wiki module.

    File location: ``{wiki_root}/pages/daemon/knowledge/test-{slug}.md``

    Uses the standard YAML frontmatter format that
    ``wiki.test_knowledge.load_test_knowledge`` parses.

    Args:
        wiki_root: Path to the wiki root.
        slug: Test slug (e.g., "agent-test").
        command_pattern: Canonical command string.
        purpose: One-line purpose description.
        required_args: Required argument names.
        common_failures: Known failure patterns.
        runs_observed: Number of observed runs.
    """
    knowledge_dir = wiki_root / "pages" / "daemon" / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    file_path = knowledge_dir / f"test-{slug}.md"

    # Build YAML-safe failure list
    failures_yaml = "\n".join(f"  - '{f}'" for f in common_failures)
    args_yaml = "\n".join(f"  - {a}" for a in required_args)

    file_path.write_text(
        f"---\n"
        f"tags: [daemon, test-knowledge, learning]\n"
        f"type: test-knowledge\n"
        f"test_slug: {slug}\n"
        f"command_pattern: {command_pattern}\n"
        f"purpose: >-\n"
        f"  {purpose}\n"
        f"output_format: \"\"\n"
        f"common_failures:\n"
        f"{failures_yaml}\n"
        f"normal_behavior: \"\"\n"
        f"required_args:\n"
        f"{args_yaml}\n"
        f"runs_observed: {runs_observed}\n"
        f"---\n"
        f"# Test: {slug}\n\n"
        f"{purpose}\n"
    )


def _make_simulated_wiki(tmp_dir: Path) -> Path:
    """Create a minimal wiki structure with a test spec for demo 1.

    Creates the wiki directory tree and a test knowledge file for
    the 'agent_test' test so that lookup_test_spec can find it.

    Args:
        tmp_dir: Temporary directory to create the wiki in.

    Returns:
        Path to the wiki root directory.
    """
    wiki_root = tmp_dir / "wiki"

    # Create directory structure matching the wiki layout expected by
    # test_knowledge.py: pages/daemon/knowledge/test-{slug}.md
    for subdir in [
        "pages/daemon/history",
        "pages/daemon/results",
        "pages/daemon/translations",
        "pages/daemon/audit",
        "pages/daemon/queue",
        "pages/daemon/knowledge",
        "pages/agents",
    ]:
        (wiki_root / subdir).mkdir(parents=True, exist_ok=True)

    # Create current-run.md (idle state)
    (wiki_root / "pages" / "daemon" / "current-run.md").write_text(
        "---\n"
        "tags: [daemon, state, current-run]\n"
        "type: daemon-state\n"
        "status: idle\n"
        "---\n"
        "# Current Run\n\n"
        "No active run.\n"
    )

    # Create test knowledge files at the path the wiki module expects.
    # derive_test_slug("agent_test") -> "agent-test"
    # File: pages/daemon/knowledge/test-agent-test.md
    _write_test_knowledge(
        wiki_root,
        slug="agent-test",
        command_pattern="python3 ~/agent_test.py",
        purpose=(
            "Runs the agent loop stress test with configurable iterations "
            "and concurrency. Verifies that the agent can maintain state "
            "across multiple think-act cycles under load."
        ),
        required_args=("iterations", "host"),
        common_failures=(
            "timeout on large iteration counts (>500)",
            "connection refused when SSH agent is not forwarded",
            "ImportError: missing dependency on fresh hosts",
        ),
        runs_observed=42,
    )

    # Also create one for the full command form:
    # derive_test_slug("python3 ~/agent_test.py") -> "agent-test-py"
    _write_test_knowledge(
        wiki_root,
        slug="agent-test-py",
        command_pattern="python3 ~/agent_test.py",
        purpose=(
            "Runs the agent loop stress test with configurable iterations "
            "and concurrency. Verifies that the agent can maintain state "
            "across multiple think-act cycles under load."
        ),
        required_args=("iterations", "host"),
        common_failures=(
            "timeout on large iteration counts (>500)",
            "connection refused when SSH agent is not forwarded",
            "ImportError: missing dependency on fresh hosts",
        ),
        runs_observed=42,
    )

    return wiki_root


# ---------------------------------------------------------------------------
# Scripted LLM client (replays Demo 1 tool call sequence)
# ---------------------------------------------------------------------------


class ScriptedLLMClient:
    """LLM client that replays a scripted sequence of tool calls.

    Simulates the LLM by returning pre-determined tool calls that
    follow the Demo 1 expected tool sequence. Each call to
    get_tool_calls() advances to the next step in the script.

    The script corresponds to the named-test end-to-end flow:
        1. lookup_test_spec -> find the test spec
        2. propose_ssh_command -> propose the command (needs approval)
        3. execute_ssh -> execute the approved command (needs approval)
        4. read_output -> read the execution output
        5. parse_test_output -> parse the test results
        6. summarize_run -> produce final summary
        7. (empty) -> signal completion
    """

    def __init__(self) -> None:
        self._step = 0
        self._approval_id: str | None = None
        self._stdout_output: str = ""

    async def get_tool_calls(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        """Return the next scripted tool call(s).

        Inspects the conversation history to extract dynamic values
        (approval_id from propose_ssh_command, stdout from execute_ssh)
        to use in subsequent calls.

        Args:
            messages: Current conversation history.

        Returns:
            Tuple of ToolCalls for this step, or empty tuple for completion.
        """
        # Extract dynamic values from history
        self._extract_dynamic_values(messages)

        step = self._step
        self._step += 1

        if step == 0:
            # Step 1: Look up the test spec
            return (
                ToolCall(
                    call_id="call-001",
                    tool_name="lookup_test_spec",
                    arguments={"test_name": "agent_test"},
                ),
            )

        if step == 1:
            # Step 2: Propose the SSH command
            return (
                ToolCall(
                    call_id="call-002",
                    tool_name="propose_ssh_command",
                    arguments={
                        "command": (
                            "python3 ~/agent_test.py "
                            "--iterations 100 --host staging"
                        ),
                        "target_host": "staging.example.com",
                        "target_user": "deploy",
                        "explanation": (
                            "Running the agent stress test with 100 iterations "
                            "targeting the staging host"
                        ),
                    },
                ),
            )

        if step == 2:
            # Step 3: Execute the approved command
            approval_id = self._approval_id or "unknown"
            return (
                ToolCall(
                    call_id="call-003",
                    tool_name="execute_ssh",
                    arguments={
                        "approval_id": approval_id,
                        "timeout": 300,
                    },
                ),
            )

        if step == 3:
            # Step 4: Read the output from the session
            return (
                ToolCall(
                    call_id="call-004",
                    tool_name="read_output",
                    arguments={"source": "session"},
                ),
            )

        if step == 4:
            # Step 5: Parse the test output
            return (
                ToolCall(
                    call_id="call-005",
                    tool_name="parse_test_output",
                    arguments={
                        "raw_output": self._stdout_output or (
                            "Iteration 1/100 ... OK\n"
                            "Iteration 50/100 ... OK\n"
                            "Iteration 100/100 ... OK\n"
                            "Result: 100 passed, 0 failed, 0 skipped in 67s"
                        ),
                        "framework_hint": "auto",
                    },
                ),
            )

        if step == 5:
            # Step 6: Summarize the run
            return (
                ToolCall(
                    call_id="call-006",
                    tool_name="summarize_run",
                    arguments={
                        "stdout": self._stdout_output or (
                            "Result: 100 passed, 0 failed, 0 skipped in 67s"
                        ),
                        "stderr": "",
                        "command": (
                            "python3 ~/agent_test.py "
                            "--iterations 100 --host staging"
                        ),
                        "exit_code": 0,
                    },
                ),
            )

        # Step 7+: Signal completion (no more tool calls)
        return ()

    def _extract_dynamic_values(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> None:
        """Extract approval_id and stdout from conversation history.

        Scans tool result messages for:
        - approval_id from propose_ssh_command result
        - stdout from execute_ssh result
        """
        for msg in messages:
            if msg.get("role") != "tool":
                continue

            content = msg.get("content", "")
            if not content or content.startswith("ERROR:"):
                continue

            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue

            # Extract approval_id from propose_ssh_command
            if (
                isinstance(data, dict)
                and data.get("approved") is True
                and "approval_id" in data
            ):
                self._approval_id = data["approval_id"]

            # Extract stdout from execute_ssh
            if isinstance(data, dict) and "stdout" in data and "exit_code" in data:
                self._stdout_output = data.get("stdout", "")


# ---------------------------------------------------------------------------
# Mock tool: execute_ssh with simulated SSH execution
# ---------------------------------------------------------------------------


class SimulatedExecuteSSHTool(BaseTool):
    """Execute_ssh tool that simulates SSH execution instead of connecting.

    Mirrors the real ExecuteSSHTool's approval enforcement and
    confirmation flow, but replaces the actual SSH connection with
    simulated output that matches Demo 1 expectations.
    """

    _spec = ExecuteSSHTool._spec

    def __init__(
        self,
        *,
        wiki_root: Path,
        ledger: ApprovalLedger,
        confirm_callback: Any,
    ) -> None:
        self._wiki_root = wiki_root
        self._ledger = ledger
        self._confirm_callback = confirm_callback

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute with simulated SSH output.

        Enforces the same approval_id validation and human confirmation
        gate as the real ExecuteSSHTool, but produces simulated output
        instead of making a real SSH connection.
        """
        approval_id = args.get("approval_id", "")
        call_id = args.get("_call_id", "execute_ssh")

        if not approval_id or not approval_id.strip():
            return ToolResult(
                call_id=call_id,
                tool_name="execute_ssh",
                status=ToolResultStatus.ERROR,
                output="",
                error_message=(
                    "approval_id is required. Use propose_ssh_command "
                    "first to get an approval_id."
                ),
            )

        approval_id = approval_id.strip()

        # Look up approval entry (peek without consuming)
        entry = self._ledger.get_approved_command(approval_id)
        if entry is None:
            return ToolResult(
                call_id=call_id,
                tool_name="execute_ssh",
                status=ToolResultStatus.ERROR,
                output="",
                error_message=(
                    f"No approved command found for approval_id={approval_id}. "
                    "Use propose_ssh_command first to get user approval."
                ),
            )

        # Human confirmation gate (real interactive prompt)
        _print_stage(
            "acting",
            f"Requesting execution confirmation for: {entry.command[:60]}...",
        )

        try:
            explanation = (
                f"Executing approved command (approval_id={entry.approval_id})"
            )
            approved, final_command = await self._confirm_callback(
                entry.command,
                entry.target_host,
                explanation,
            )
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                tool_name="execute_ssh",
                status=ToolResultStatus.ERROR,
                output="",
                error_message=f"Execution confirmation failed: {exc}",
            )

        if not approved:
            return ToolResult(
                call_id=call_id,
                tool_name="execute_ssh",
                status=ToolResultStatus.DENIED,
                output=json.dumps({
                    "approved": False,
                    "approval_id": approval_id,
                    "command": entry.command,
                }),
                error_message="User denied command execution",
            )

        # Consume the approval
        consumed = self._ledger.consume(approval_id)
        if consumed is None:
            return ToolResult(
                call_id=call_id,
                tool_name="execute_ssh",
                status=ToolResultStatus.ERROR,
                output="",
                error_message=(
                    f"Approval {approval_id} was consumed by another call."
                ),
            )

        command_to_execute = final_command if final_command else consumed.command

        # Simulate SSH execution with realistic output
        _print_stage(
            "acting",
            f"Simulating SSH execution: {command_to_execute[:60]}...",
        )
        print(f"    {_DIM}(Simulated -- no real SSH connection){_RESET}")

        # Brief pause to simulate execution time
        await asyncio.sleep(0.5)

        simulated_stdout = (
            "Iteration 1/100 ... OK\n"
            "Iteration 2/100 ... OK\n"
            "Iteration 3/100 ... OK\n"
            "...\n"
            "Iteration 99/100 ... OK\n"
            "Iteration 100/100 ... OK\n"
            "Result: 100 passed, 0 failed, 0 skipped in 67s"
        )

        result_data = {
            "success": True,
            "run_id": "demo-run-001",
            "command": command_to_execute,
            "target_host": consumed.target_host,
            "target_user": consumed.target_user,
            "exit_code": 0,
            "stdout": simulated_stdout,
            "stderr": "",
            "error": None,
            "duration_seconds": 67.0,
        }

        return ToolResult(
            call_id=call_id,
            tool_name="execute_ssh",
            status=ToolResultStatus.SUCCESS,
            output=json.dumps(result_data, default=str),
        )


# ---------------------------------------------------------------------------
# Observing dispatch bridge (wraps ToolDispatchBridge with stage logging)
# ---------------------------------------------------------------------------


class ObservableDispatchBridge:
    """Wraps ToolDispatchBridge to print stage transitions during dispatch.

    Satisfies the ToolDispatcher protocol while adding observability
    logging for the demo.
    """

    def __init__(self, *, registry: ToolRegistry) -> None:
        self._bridge = ToolDispatchBridge(registry=registry)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        """Dispatch a tool call with observability logging."""
        _print_tool_call(call)

        start = time.monotonic()
        result = await self._bridge.dispatch(call)
        elapsed = time.monotonic() - start

        _print_tool_result(result)
        print(f"    {_DIM}({elapsed:.3f}s){_RESET}")

        return result


# ---------------------------------------------------------------------------
# Observable agent loop (wraps AgentLoop with stage transition logging)
# ---------------------------------------------------------------------------


class ObservableAgentLoop:
    """Wraps AgentLoop to print stage transitions during execution.

    Reimplements the run() method to add stdout logging at each
    phase boundary, while delegating the actual logic to the same
    phase implementations used by AgentLoop.
    """

    def __init__(
        self,
        *,
        llm_client: ScriptedLLMClient,
        dispatcher: ObservableDispatchBridge,
        system_prompt: str,
        config: AgentLoopConfig,
    ) -> None:
        self._loop = AgentLoop(
            llm_client=llm_client,
            tool_dispatcher=dispatcher,
            system_prompt=system_prompt,
            config=config,
        )

    async def run(self, user_message: str) -> AgentLoopResult:
        """Execute the agent loop with stage transition logging.

        Delegates to the real AgentLoop but prints state transitions
        and iteration boundaries.

        Args:
            user_message: The natural-language user command.

        Returns:
            AgentLoopResult from the underlying agent loop.
        """
        print(f"\n{_BOLD}  User message:{_RESET} {_CYAN}{user_message}{_RESET}\n")
        _print_separator()

        result = await self._loop.run(user_message)

        return result


# ---------------------------------------------------------------------------
# Build the demo tool set
# ---------------------------------------------------------------------------


def _build_demo_registry(
    *,
    wiki_root: Path,
    ledger: ApprovalLedger,
) -> ToolRegistry:
    """Build a ToolRegistry with all tools wired for interactive demo.

    Uses:
    - Real read-only tools (lookup_test_spec, parse_test_output, etc.)
    - Interactive confirmation callbacks for propose_ssh_command
    - Simulated SSH execution for execute_ssh
    - Interactive ask callback for ask_user_question
    - Stdout-based notify callback for notify_user

    Args:
        wiki_root: Path to the demo wiki directory.
        ledger: Shared approval ledger.

    Returns:
        ToolRegistry with all 10 tools registered.
    """
    from jules_daemon.agent.tools.check_remote_processes import (
        CheckRemoteProcessesTool,
    )
    from jules_daemon.agent.tools.read_output import ReadOutputTool
    from jules_daemon.agent.tools.read_wiki import ReadWikiTool

    registry = ToolRegistry()

    # Read-only tools
    registry.register(ReadWikiTool(wiki_root=wiki_root))
    registry.register(LookupTestSpecTool(wiki_root=wiki_root))
    registry.register(CheckRemoteProcessesTool())
    registry.register(ReadOutputTool(wiki_root=wiki_root))
    registry.register(ParseTestOutputTool())

    # State-changing tools with interactive callbacks
    registry.register(ProposeSSHCommandTool(
        confirm_callback=_interactive_confirm,
        ledger=ledger,
    ))
    registry.register(SimulatedExecuteSSHTool(
        wiki_root=wiki_root,
        ledger=ledger,
        confirm_callback=_interactive_confirm,
    ))

    # User interaction tools
    registry.register(AskUserQuestionTool(ask_callback=_interactive_ask))
    registry.register(SummarizeRunTool(wiki_root=wiki_root))
    registry.register(NotifyUserTool(notify_callback=_interactive_notify))

    return registry



# ---------------------------------------------------------------------------
# System prompt for the demo
# ---------------------------------------------------------------------------

_DEMO_SYSTEM_PROMPT = """\
You are a test execution assistant for the Jules SSH Test Runner Daemon.
You have access to tools for looking up test specifications, proposing
and executing SSH commands, reading output, and parsing test results.

Target SSH host: staging.example.com
Target SSH user: deploy
Target SSH port: 22

Rules:
1. Always look up the test spec first using lookup_test_spec.
2. If the spec has required_args that the user did not provide, use
   ask_user_question to ask for each missing argument. Never guess.
3. Use propose_ssh_command to propose the command -- this requires
   human approval before the command can be executed.
4. Use execute_ssh with the approval_id from propose_ssh_command.
5. After execution, use read_output and parse_test_output to analyze results.
6. Summarize the run using summarize_run.
7. If a command fails, observe the error, propose a corrected command,
   and retry after getting fresh approval.
"""


# ---------------------------------------------------------------------------
# Main demo execution
# ---------------------------------------------------------------------------


async def run_demo(
    *,
    max_iterations: int = 10,
    scenario: str = "demo1",
) -> AgentLoopResult:
    """Execute the interactive demo.

    Sets up the full agent loop infrastructure with interactive
    callbacks and a scripted LLM, then runs the named-test flow.

    Args:
        max_iterations: Maximum agent loop iterations.
        scenario: Demo scenario to run (currently only 'demo1').

    Returns:
        The AgentLoopResult from the agent loop execution.
    """
    _print_banner()

    # Create temporary wiki directory
    import tempfile
    with tempfile.TemporaryDirectory(prefix="jules-demo-") as tmp_dir:
        wiki_root = _make_simulated_wiki(Path(tmp_dir))

        print(f"  {_DIM}Wiki root: {wiki_root}{_RESET}")
        print(f"  {_DIM}Max iterations: {max_iterations}{_RESET}")
        print(f"  {_DIM}Scenario: {scenario}{_RESET}")
        print()

        # Print registry info
        ledger = ApprovalLedger()
        registry = _build_demo_registry(
            wiki_root=wiki_root,
            ledger=ledger,
        )

        print(f"  {_BOLD}Registered tools ({len(registry)}):{_RESET}")
        for name in registry.list_tool_names():
            tool = registry.get(name)
            assert tool is not None
            approval_str = (
                f"{_RED}CONFIRM_PROMPT{_RESET}"
                if tool.spec.approval.value == "confirm_prompt"
                else f"{_GREEN}NONE{_RESET}"
            )
            print(f"    {name:30s} approval={approval_str}")

        print()
        _print_separator()

        # Build the agent loop
        llm_client = ScriptedLLMClient()
        dispatcher = ObservableDispatchBridge(registry=registry)

        config = AgentLoopConfig(
            max_iterations=max_iterations,
            max_retries=2,
        )

        loop = ObservableAgentLoop(
            llm_client=llm_client,
            dispatcher=dispatcher,
            system_prompt=_DEMO_SYSTEM_PROMPT,
            config=config,
        )

        # Select the user message based on scenario
        user_messages = {
            "demo1": "run agent_test with 100 iterations on staging",
            "demo2": "run the integration tests",
            "demo3": "run agent_test",
        }
        user_message = user_messages.get(scenario, user_messages["demo1"])

        # Run the agent loop
        result = await loop.run(user_message)

        # Print the result summary
        _print_result(result)

        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo runner."""
    parser = argparse.ArgumentParser(
        description="Jules Agent Loop -- Interactive Demo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m jules_daemon.demo_runner\n"
            "  python -m jules_daemon.demo_runner --max-iterations 10\n"
            "  python -m jules_daemon.demo_runner --scenario demo1\n"
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum agent loop iterations (default: 10)",
    )
    parser.add_argument(
        "--scenario",
        choices=["demo1", "demo2", "demo3"],
        default="demo1",
        help="Demo scenario to run (default: demo1)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the demo runner."""
    args = _parse_args()

    try:
        result = asyncio.run(
            run_demo(
                max_iterations=args.max_iterations,
                scenario=args.scenario,
            )
        )
    except KeyboardInterrupt:
        print(f"\n\n  {_RED}Demo cancelled by user{_RESET}\n")
        sys.exit(130)

    # Exit code based on result
    if result.final_state is AgentLoopState.COMPLETE:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
