"""Tool dispatch bridge and result collection for the agent loop.

Bridges the gap between the response parser (which produces
``ToolCallsResponse`` with parsed ``ToolCall`` instances) and the
``ToolRegistry`` (which validates and executes individual calls).
This module handles the **ACTING** and **OBSERVING** phases of the
agent loop's think-act-observe cycle:

    ACTING:     Route each parsed tool call to the ToolRegistry,
                collect execution results including errors/failures.
    OBSERVING:  Format results as OpenAI-compatible observation messages
                and append them to the immutable conversation history
                for the next iteration.

Key responsibilities:

    - ``dispatch(call)`` -- Dispatch a single ToolCall through the
      registry. Satisfies the ``ToolDispatcher`` protocol from
      ``agent_loop.py`` so the bridge can be used directly by the
      AgentLoop state machine.

    - ``dispatch_parsed_response(parsed)`` -- Batch dispatch all tool
      calls from a ``ToolCallsResponse``. Dispatches sequentially to
      preserve ordering semantics, collects all results, and classifies
      the overall outcome as CONTINUE or TERMINAL.

    - ``dispatch_tool_calls(calls)`` -- Same as above but accepts a raw
      tuple of ToolCalls. Useful when the caller has already extracted
      the calls from the parsed response.

    - ``format_observations(history, dispatch_result)`` -- Append the
      assistant message (with tool_calls) and per-call tool result
      messages to an immutable ``ConversationHistory``. Returns a new
      history instance; the original is never mutated.

Outcome classification:

    - **CONTINUE**: All results are non-terminal (SUCCESS, ERROR, or
      TIMEOUT). The agent loop should proceed to the next iteration.
      ERROR results are observable by the LLM and enable self-correction.

    - **TERMINAL**: At least one result has a terminal status (DENIED).
      The agent loop must stop immediately. The ``terminal_reason``
      field describes why.

Tracking:

    - ``dispatch_count`` -- Total number of individual tool calls
      dispatched through this bridge (across all batch calls).
    - ``all_results`` -- Ordered tuple of all ToolResults collected
      so far, for post-loop analysis and audit.

Thread-safety: Designed for single-threaded asyncio usage within a
single agent loop session. No locking is required.

Usage::

    from jules_daemon.agent.tool_dispatch import ToolDispatchBridge
    from jules_daemon.agent.tool_registry import ToolRegistry

    registry = ToolRegistry()
    # ... register tools ...

    bridge = ToolDispatchBridge(registry=registry)

    # ACTING phase
    dispatch_result = await bridge.dispatch_parsed_response(parsed_response)

    # OBSERVING phase
    new_history = bridge.format_observations(history, dispatch_result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from jules_daemon.agent.conversation_history import (
    ConversationHistory,
    append_assistant_message,
    append_tool_result,
)
from jules_daemon.agent.response_parser import ToolCallsResponse
from jules_daemon.agent.tool_registry import ToolRegistry
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResult,
)

__all__ = [
    "DispatchOutcome",
    "DispatchResult",
    "ToolDispatchBridge",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DispatchOutcome enum
# ---------------------------------------------------------------------------


class DispatchOutcome(Enum):
    """Classification of a batch dispatch result.

    CONTINUE: All results are non-terminal. The agent loop should
        proceed to the next think-act-observe iteration. ERROR and
        TIMEOUT results are included in the conversation history so
        the LLM can observe them and self-correct.

    TERMINAL: At least one result has a terminal status (DENIED).
        The agent loop must stop immediately. The ``terminal_reason``
        field on ``DispatchResult`` describes why.
    """

    CONTINUE = "continue"
    TERMINAL = "terminal"

    @property
    def is_terminal(self) -> bool:
        """Return True if this outcome should end the agent loop."""
        return self is DispatchOutcome.TERMINAL


# ---------------------------------------------------------------------------
# DispatchResult frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchResult:
    """Immutable result of dispatching one or more tool calls.

    Captures the complete output of the ACTING phase: the tool calls
    that were dispatched, the results from execution, and the overall
    outcome classification.

    Attributes:
        tool_calls: The tool calls that were dispatched, in order.
        results: The execution results, in the same order as tool_calls.
        outcome: CONTINUE if the loop should keep going, TERMINAL if
            a terminal result (e.g., DENIED) was encountered.
        terminal_reason: Human-readable description of why the loop
            should terminate. None when outcome is CONTINUE.
        assistant_text: Optional text content from the LLM response
            that accompanied the tool calls (reasoning preamble).
            Preserved so it can be included in the assistant message
            during the OBSERVING phase.
    """

    tool_calls: tuple[ToolCall, ...]
    results: tuple[ToolResult, ...]
    outcome: DispatchOutcome
    terminal_reason: str | None
    assistant_text: str | None = None

    @property
    def has_terminal(self) -> bool:
        """True if the outcome is terminal (loop should stop)."""
        return self.outcome.is_terminal

    @property
    def all_succeeded(self) -> bool:
        """True if every result has SUCCESS status."""
        return all(r.is_success for r in self.results)


# ---------------------------------------------------------------------------
# ToolDispatchBridge
# ---------------------------------------------------------------------------


class ToolDispatchBridge:
    """Bridges parsed tool-call requests to the ToolRegistry.

    Handles the ACTING phase (routing calls to the registry and
    collecting results) and the OBSERVING phase (formatting results
    as conversation history messages).

    Satisfies the ``ToolDispatcher`` protocol from ``agent_loop.py``
    via the ``dispatch(call)`` method, so it can be plugged directly
    into the AgentLoop state machine.

    Args:
        registry: The ToolRegistry containing registered tools.
    """

    def __init__(self, *, registry: ToolRegistry) -> None:
        self._registry = registry
        self._dispatch_count: int = 0
        self._all_results: list[ToolResult] = []

    # -- Tracking properties ------------------------------------------------

    @property
    def dispatch_count(self) -> int:
        """Total number of individual tool calls dispatched."""
        return self._dispatch_count

    @property
    def all_results(self) -> tuple[ToolResult, ...]:
        """All results collected so far, for post-loop analysis."""
        return tuple(self._all_results)

    # -- Single dispatch (ToolDispatcher protocol) --------------------------

    async def dispatch(self, call: ToolCall) -> ToolResult:
        """Dispatch a single tool call through the registry.

        Satisfies the ``ToolDispatcher`` protocol from ``agent_loop.py``.
        Delegates validation and execution to ``ToolRegistry.execute()``,
        which handles unknown tools, missing parameters, and unhandled
        exceptions by wrapping them in error ToolResults.

        Args:
            call: The ToolCall to dispatch.

        Returns:
            ToolResult from the registry execution.
        """
        logger.debug(
            "Dispatching tool call: %s (call_id=%s)",
            call.tool_name,
            call.call_id,
        )
        result = await self._registry.execute(call)

        self._dispatch_count += 1
        self._all_results.append(result)

        logger.debug(
            "Tool %s returned status=%s (call_id=%s)",
            call.tool_name,
            result.status.value,
            call.call_id,
        )
        return result

    # -- Batch dispatch from parsed response --------------------------------

    async def dispatch_parsed_response(
        self,
        parsed: ToolCallsResponse,
    ) -> DispatchResult:
        """Dispatch all tool calls from a parsed LLM response.

        Routes each call sequentially through the ToolRegistry to
        preserve ordering semantics (important when tools have
        side effects, e.g., propose_ssh_command then execute_ssh).

        Collects all results and classifies the batch outcome:
        - CONTINUE if all results are non-terminal
        - TERMINAL if any result has a terminal status (DENIED)

        Args:
            parsed: A ``ToolCallsResponse`` from the response parser
                containing one or more tool calls to dispatch.

        Returns:
            Immutable ``DispatchResult`` with all calls, results,
            and the overall outcome classification.
        """
        return await self._dispatch_calls_internal(
            tool_calls=parsed.tool_calls,
            assistant_text=parsed.assistant_text,
        )

    async def dispatch_tool_calls(
        self,
        tool_calls: tuple[ToolCall, ...],
    ) -> DispatchResult:
        """Dispatch a tuple of tool calls directly.

        Convenience method for callers that have already extracted the
        tool calls from the parsed response. Behaves identically to
        ``dispatch_parsed_response()`` but without the
        ``ToolCallsResponse`` wrapper.

        Args:
            tool_calls: Tuple of ToolCalls to dispatch.

        Returns:
            Immutable ``DispatchResult`` with all calls, results,
            and the overall outcome classification.
        """
        return await self._dispatch_calls_internal(
            tool_calls=tool_calls,
            assistant_text=None,
        )

    # -- Observation formatting (OBSERVING phase) ---------------------------

    def format_observations(
        self,
        history: ConversationHistory,
        dispatch_result: DispatchResult,
    ) -> ConversationHistory:
        """Append a tool-calling cycle to the conversation history.

        Adds an assistant message (with the tool_calls array and any
        accompanying text) followed by one tool result message per
        dispatched call. This produces the observation context that
        the LLM will see in the next iteration.

        Error results include an ``ERROR:`` prefix in the tool message
        content so the LLM gets a clear signal about what failed and
        can attempt self-correction.

        This method is pure: it returns a **new** ConversationHistory
        and never mutates the original.

        Args:
            history: The current (immutable) conversation history.
            dispatch_result: The result of the ACTING phase containing
                tool calls and their execution results.

        Returns:
            New ConversationHistory with the observation messages
            appended.
        """
        # Append assistant message with tool_calls (and optional text)
        updated = append_assistant_message(
            history,
            content=dispatch_result.assistant_text,
            tool_calls=dispatch_result.tool_calls,
        )

        # Append each tool result as a tool-role message
        for result in dispatch_result.results:
            updated = append_tool_result(updated, result)

        return updated

    # -- Internal helpers ---------------------------------------------------

    async def _dispatch_calls_internal(
        self,
        *,
        tool_calls: tuple[ToolCall, ...],
        assistant_text: str | None,
    ) -> DispatchResult:
        """Core dispatch logic shared by public dispatch methods.

        Dispatches each call sequentially, collects results, and
        classifies the batch outcome. Short-circuits immediately on
        terminal results (DENIED): remaining calls are not dispatched.
        This ensures permanent errors like user denial terminate
        without executing further side effects.

        Args:
            tool_calls: The tool calls to dispatch.
            assistant_text: Optional text from the LLM response.

        Returns:
            Immutable DispatchResult. The ``results`` tuple may be
            shorter than ``tool_calls`` if short-circuited.
        """
        results: list[ToolResult] = []
        dispatched_calls: list[ToolCall] = []
        terminal_reason: str | None = None

        for call in tool_calls:
            result = await self.dispatch(call)
            results.append(result)
            dispatched_calls.append(call)

            # Short-circuit on terminal result (e.g., user denial)
            if result.is_terminal:
                terminal_reason = (
                    f"Tool '{result.tool_name}' was denied: "
                    f"{result.error_message or 'User denied the operation'}"
                )
                logger.info(
                    "Terminal tool result during batch dispatch: "
                    "%s (status=%s) -- short-circuiting remaining calls",
                    result.tool_name,
                    result.status.value,
                )
                break

        outcome = (
            DispatchOutcome.TERMINAL
            if terminal_reason is not None
            else DispatchOutcome.CONTINUE
        )

        return DispatchResult(
            tool_calls=tuple(dispatched_calls),
            results=tuple(results),
            outcome=outcome,
            terminal_reason=terminal_reason,
            assistant_text=assistant_text,
        )
