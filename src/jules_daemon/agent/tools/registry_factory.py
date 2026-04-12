"""Factory for constructing the full tool set with proper dependency injection.

Creates all 10 agent tools with their required dependencies (wiki_root,
callbacks, ledger, LLM client) and returns them as a tuple ready for
registration in a ToolRegistry.

This is the single wiring point between the daemon's infrastructure
(IPC handlers, wiki, SSH) and the agent tool layer. Each tool wraps
existing daemon functionality via dependency injection -- no daemon
internals are reimplemented.

Usage::

    from jules_daemon.agent.tools.registry_factory import build_tool_set

    tools = build_tool_set(
        wiki_root=Path("/data/wiki"),
        confirm_callback=handler.confirm,
        ask_callback=handler.ask_user,
        notify_callback=event_bus.push,
        llm_client=openai_client,
        llm_model="gpt-4",
    )
    for tool in tools:
        registry.register(tool)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jules_daemon.agent.tools.ask_user_question import (
    AskCallback,
    AskUserQuestionTool,
)
from jules_daemon.agent.tools.base import BaseTool
from jules_daemon.agent.tools.check_remote_processes import (
    CheckRemoteProcessesTool,
)
from jules_daemon.agent.tools.execute_ssh import ExecuteSSHTool
from jules_daemon.agent.tools.lookup_test_spec import LookupTestSpecTool
from jules_daemon.agent.tools.notify_user import NotifyCallback, NotifyUserTool
from jules_daemon.agent.tools.parse_test_output import ParseTestOutputTool
from jules_daemon.agent.tools.propose_ssh_command import (
    ApprovalLedger,
    ConfirmCallback,
    ProposeSSHCommandTool,
)
from jules_daemon.agent.tools.read_output import ReadOutputTool, SessionHistoryProvider
from jules_daemon.agent.tools.read_wiki import ReadWikiTool
from jules_daemon.agent.tools.summarize_run import SummarizeRunTool

__all__ = ["build_tool_set"]


def build_tool_set(
    *,
    wiki_root: Path,
    confirm_callback: ConfirmCallback,
    ask_callback: AskCallback,
    notify_callback: NotifyCallback,
    llm_client: Any | None = None,
    llm_model: str | None = None,
    ledger: ApprovalLedger | None = None,
    session_history_provider: SessionHistoryProvider | None = None,
) -> tuple[BaseTool, ...]:
    """Construct all 10 agent tools with injected dependencies.

    Creates a fresh ApprovalLedger (shared between propose_ssh_command
    and execute_ssh) unless one is provided. This ensures the approval
    constraint is enforced within a single session.

    Args:
        wiki_root: Path to the wiki root directory for file-backed tools.
        confirm_callback: Async callback for user command confirmation.
        ask_callback: Async callback for asking user questions.
        notify_callback: Async callback for pushing notifications.
        llm_client: Optional OpenAI-compatible client for LLM-enhanced
            summarization. When None, summarize_run uses regex only.
        llm_model: LLM model identifier (required when llm_client is set).
        ledger: Optional pre-existing ApprovalLedger. When None, a fresh
            one is created for this tool set.
        session_history_provider: Optional callback that returns the current
            conversation history messages. Enables read_output to read
            prior tool results from the agent loop session.

    Returns:
        Tuple of 10 BaseTool instances, ready for registry registration.
    """
    if ledger is None:
        ledger = ApprovalLedger()

    return (
        # Read-only tools (no approval required)
        ReadWikiTool(wiki_root=wiki_root),
        LookupTestSpecTool(wiki_root=wiki_root),
        CheckRemoteProcessesTool(),
        ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=session_history_provider,
        ),
        ParseTestOutputTool(),
        # State-changing tools (require approval)
        ProposeSSHCommandTool(
            confirm_callback=confirm_callback,
            ledger=ledger,
        ),
        ExecuteSSHTool(
            wiki_root=wiki_root,
            ledger=ledger,
            confirm_callback=confirm_callback,
        ),
        # User interaction tools
        AskUserQuestionTool(ask_callback=ask_callback),
        SummarizeRunTool(
            llm_client=llm_client,
            llm_model=llm_model,
            wiki_root=wiki_root,
        ),
        NotifyUserTool(notify_callback=notify_callback),
    )
