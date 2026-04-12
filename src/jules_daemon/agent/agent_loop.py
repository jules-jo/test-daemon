"""Agent loop state machine for iterative think-act-observe cycles.

Replaces the one-shot LLM translation path with an iterative agent loop
where the LLM can call tools, observe results (including failures), and
propose corrections. The loop orchestrates the THINKING -> ACTING ->
OBSERVING cycle until a termination condition is met.

Termination conditions (OR-composed):
    - Max iterations reached (configurable, default 5)
    - LLM returns no tool calls (natural completion)
    - User cancels / denies an approval-required tool call
    - Permanent error (malformed LLM response, auth failure)

Transient errors (network blips, LLM timeouts) are retried within the
same iteration up to ``max_retries`` times. If all retries fail, the
loop terminates with an ERROR state and a fallback hint.

Protocol abstractions:
    - LLMClient: Protocol for getting tool calls from conversation history.
      Decoupled from the OpenAI SDK for testability with mocks.
    - ToolDispatcher: Protocol for dispatching tool calls to registered
      tools. Decoupled from ToolRegistry for testability with mocks.

Usage::

    from jules_daemon.agent.agent_loop import AgentLoop, AgentLoopConfig

    loop = AgentLoop(
        llm_client=my_llm_client,
        tool_dispatcher=my_dispatcher,
        system_prompt="You are a test runner assistant.",
        config=AgentLoopConfig(max_iterations=5),
    )
    result = await loop.run("run the smoke tests on staging")
    if result.final_state is AgentLoopState.COMPLETE:
        print("Agent completed successfully")
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from jules_daemon.agent.error_classification import (
    ClassifiedError,
    classify_error,
)
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResult,
)

__all__ = [
    "AgentLoop",
    "AgentLoopConfig",
    "AgentLoopError",
    "AgentLoopResult",
    "AgentLoopState",
    "LLMClient",
    "SleepFn",
    "ToolDispatcher",
    "_compute_backoff_delay",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backoff helpers
# ---------------------------------------------------------------------------


def _compute_backoff_delay(attempt: int, base_delay: float) -> float:
    """Compute exponential backoff delay for a retry attempt.

    Uses the formula: base_delay * 2^attempt, where ``attempt`` is the
    zero-based retry index (0 for the first retry, 1 for the second, etc.).

    Args:
        attempt: Zero-based retry index. 0 = first retry, 1 = second.
        base_delay: Base delay in seconds (typically 1.0).

    Returns:
        Delay in seconds before the next retry attempt.
    """
    return base_delay * (2 ** attempt)


# Default sleep function -- points to asyncio.sleep for production use.
# Tests can inject a no-op or recording replacement via the AgentLoop
# constructor for deterministic, fast execution.
SleepFn = Callable[[float], Awaitable[None]]


# ---------------------------------------------------------------------------
# AgentLoopState enum
# ---------------------------------------------------------------------------


class AgentLoopState(Enum):
    """State machine states for the agent loop.

    The loop transitions through these states during each iteration:

        THINKING -> ACTING -> OBSERVING -> THINKING -> ...

    Terminal states:
        COMPLETE: LLM signaled completion (no more tool calls).
        ERROR: Unrecoverable error or user denial.

    Non-terminal states:
        THINKING: Waiting for / processing LLM response.
        ACTING: Dispatching tool calls from the LLM response.
        OBSERVING: Appending tool results to conversation history.
    """

    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETE = "complete"
    ERROR = "error"

    @property
    def is_terminal(self) -> bool:
        """Return True if this state ends the agent loop."""
        return self in _TERMINAL_STATES


# Pre-computed frozenset for O(1) terminal check.
_TERMINAL_STATES: frozenset[AgentLoopState] = frozenset({
    AgentLoopState.COMPLETE,
    AgentLoopState.ERROR,
})


# ---------------------------------------------------------------------------
# AgentLoopConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentLoopConfig:
    """Immutable configuration for the agent loop.

    Attributes:
        max_iterations: Hard cap on think-act-observe cycles per command.
            One iteration = one LLM call + tool dispatch + result observation.
            The loop terminates with ERROR if this limit is reached.
        max_retries: Number of retry attempts for transient LLM errors
            (network blips, timeouts) within a single iteration. If all
            retries are exhausted, the loop terminates with ERROR.
        retry_base_delay: Base delay in seconds for exponential backoff
            between retry attempts. The actual delay for retry N (0-indexed)
            is ``retry_base_delay * 2^N``. For example, with a 1.0s base:
            first retry waits 1.0s, second retry waits 2.0s. Set to 0.0
            to disable backoff (useful for tests).
    """

    max_iterations: int = 5
    max_retries: int = 2
    retry_base_delay: float = 1.0

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {self.max_iterations}"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0, got {self.max_retries}"
            )
        if self.retry_base_delay < 0.0:
            raise ValueError(
                f"retry_base_delay must be >= 0.0, got {self.retry_base_delay}"
            )


# ---------------------------------------------------------------------------
# AgentLoopResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentLoopResult:
    """Immutable result of an agent loop execution.

    Attributes:
        final_state: The terminal state the loop ended in (COMPLETE or ERROR).
        iterations_used: Number of iterations consumed before termination.
        history: The full conversation history as a tuple of message dicts.
        error_message: Human-readable error description. None on success.
        retry_exhausted: True when the loop terminated because all transient
            error retries were consumed. The caller should fall back to the
            one-shot LLM translation path when this is True.
    """

    final_state: AgentLoopState
    iterations_used: int
    history: tuple[dict[str, Any], ...]
    error_message: str | None
    retry_exhausted: bool = False


# ---------------------------------------------------------------------------
# AgentLoopError
# ---------------------------------------------------------------------------


class AgentLoopError(Exception):
    """Base error for agent loop operations."""


# ---------------------------------------------------------------------------
# Protocol abstractions
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for getting tool calls from conversation history.

    Abstracts the LLM interaction so the agent loop skeleton can be
    tested with mock clients. The real implementation wraps the OpenAI
    SDK with Dataiku Mesh configuration.

    The method receives the full conversation history (system prompt,
    user message, and any prior assistant/tool messages) and returns
    the tool calls the LLM wants to make. An empty tuple signals that
    the LLM has no more actions (loop should complete).
    """

    async def get_tool_calls(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        """Request tool calls from the LLM.

        Args:
            messages: Immutable conversation history.

        Returns:
            Tuple of ToolCalls the LLM wants to invoke. Empty tuple
            signals natural completion.

        Raises:
            ConnectionError, TimeoutError: Transient errors (retryable).
            ValueError: Permanent errors (not retryable).
        """
        ...


@runtime_checkable
class ToolDispatcher(Protocol):
    """Protocol for dispatching tool calls to registered tools.

    Abstracts tool execution so the agent loop can be tested with mock
    dispatchers. The real implementation delegates to ToolRegistry.execute().

    The dispatcher handles approval gating for state-changing tools --
    the agent loop does not need to know about approval details.
    """

    async def dispatch(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result.

        Args:
            call: The ToolCall to execute.

        Returns:
            ToolResult with the execution outcome.
        """
        ...


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
    """Orchestrates iterative think-act-observe cycles.

    The loop receives a natural-language user command and drives the LLM
    through an iterative process:

        1. THINKING: Send conversation history to LLM, receive tool calls.
        2. ACTING: Dispatch each tool call through the ToolDispatcher.
        3. OBSERVING: Append tool results to conversation history.
        4. Repeat until a termination condition is met.

    The loop is single-use: call ``run()`` once per user command. Create
    a new AgentLoop instance for each command.

    Args:
        llm_client: LLM client satisfying the LLMClient protocol.
        tool_dispatcher: Tool dispatcher satisfying the ToolDispatcher protocol.
        system_prompt: The system prompt to prepend to conversation history.
        config: Loop configuration (max iterations, retries).
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_dispatcher: ToolDispatcher,
        system_prompt: str,
        config: AgentLoopConfig | None = None,
        sleep_fn: SleepFn | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._tool_dispatcher = tool_dispatcher
        self._system_prompt = system_prompt
        self._config = config or AgentLoopConfig()
        self._sleep = sleep_fn or asyncio.sleep
        self._state = AgentLoopState.THINKING
        self._history: list[dict[str, Any]] = []
        self._retry_exhausted: bool = False

    # -- Public properties ---------------------------------------------------

    @property
    def state(self) -> AgentLoopState:
        """Current state of the agent loop."""
        return self._state

    # -- Main entry point ----------------------------------------------------

    async def run(self, user_message: str) -> AgentLoopResult:
        """Execute the agent loop for a user command.

        Orchestrates the think-act-observe cycle until a termination
        condition is met. Returns an immutable result snapshot.

        Args:
            user_message: The natural-language user command to process.

        Returns:
            AgentLoopResult with the final state, iteration count,
            history, and any error message.
        """
        self._initialize_history(user_message)

        iteration = 0

        while iteration < self._config.max_iterations:
            iteration += 1

            logger.debug(
                "Agent loop iteration %d/%d",
                iteration,
                self._config.max_iterations,
            )

            # --- THINKING phase: get tool calls from LLM ---
            self._state = AgentLoopState.THINKING
            tool_calls = await self._thinking_phase(iteration)

            # If thinking phase returned None, an error occurred and
            # the loop has been terminated.
            if tool_calls is None:
                return self._build_result(iteration)

            # No tool calls = LLM signals completion
            if not tool_calls:
                self._state = AgentLoopState.COMPLETE
                logger.info(
                    "Agent loop completed: LLM returned no tool calls "
                    "at iteration %d",
                    iteration,
                )
                return self._build_result(iteration)

            # --- ACTING phase: dispatch tool calls ---
            self._state = AgentLoopState.ACTING
            results = await self._acting_phase(tool_calls)

            # --- OBSERVING phase: process results ---
            self._state = AgentLoopState.OBSERVING
            should_terminate = self._observing_phase(tool_calls, results)

            if should_terminate:
                return self._build_result(iteration)

        # Max iterations reached
        self._state = AgentLoopState.ERROR
        error_msg = (
            f"Agent loop reached max iterations ({self._config.max_iterations})"
        )
        logger.warning(error_msg)
        return self._build_result(
            iteration,
            error_message=error_msg,
        )

    # -- Phase implementations -----------------------------------------------

    async def _thinking_phase(
        self,
        iteration: int,
    ) -> tuple[ToolCall, ...] | None:
        """Execute the THINKING phase: call LLM with conversation history.

        Handles transient error retries with exponential backoff. Uses the
        dedicated ``classify_error()`` from ``error_classification`` to
        determine whether an error is transient (retryable) or permanent.

        Retry behavior:
            - Up to ``max_retries`` additional attempts for transient errors.
            - Each retry waits ``retry_base_delay * 2^attempt_index`` seconds
              (exponential backoff). Attempt 0 = first retry.
            - Conversation context (``self._history``) is preserved across
              all retries -- the same ``messages`` snapshot is re-sent.
            - Permanent errors terminate immediately without retry.
            - If all retries are exhausted, the loop terminates with ERROR
              and a fallback-to-one-shot hint.

        Args:
            iteration: Current iteration number (for logging).

        Returns:
            Tuple of tool calls from the LLM, or None on unrecoverable error.
        """
        messages = tuple(self._history)
        retries_remaining = self._config.max_retries
        retry_attempt = 0

        while True:
            try:
                return await self._llm_client.get_tool_calls(messages)
            except Exception as exc:
                classified: ClassifiedError = classify_error(exc)

                if classified.is_retryable and retries_remaining > 0:
                    delay = _compute_backoff_delay(
                        retry_attempt, self._config.retry_base_delay,
                    )
                    retries_remaining -= 1
                    retry_attempt += 1
                    logger.warning(
                        "Transient LLM error [%s] (retries remaining: %d, "
                        "backoff: %.2fs): %s",
                        classified.category.value,
                        retries_remaining,
                        delay,
                        exc,
                    )
                    if delay > 0:
                        await self._sleep(delay)
                    continue

                # Exhausted retries or permanent error
                if classified.is_retryable:
                    error_msg = (
                        f"LLM transient error after exhausting retries: "
                        f"falling back to one-shot path. "
                        f"[{classified.category.value}] {exc}"
                    )
                    self._retry_exhausted = True
                else:
                    error_msg = (
                        f"Permanent LLM error "
                        f"[{classified.category.value}]: {exc}"
                    )

                logger.error(error_msg)
                self._state = AgentLoopState.ERROR
                self._error_message = error_msg
                return None

    async def _acting_phase(
        self,
        tool_calls: tuple[ToolCall, ...],
    ) -> tuple[ToolResult, ...]:
        """Execute the ACTING phase: dispatch tool calls sequentially.

        Tool calls are dispatched sequentially to preserve ordering
        semantics. Each call goes through the ToolDispatcher which
        handles approval gating.

        Short-circuits immediately on terminal results (DENIED): if the
        user denies a tool call, remaining tool calls in the batch are
        not dispatched. This ensures permanent errors like user denial
        terminate the loop without executing further side effects.

        Args:
            tool_calls: Tool calls from the LLM response.

        Returns:
            Tuple of ToolResults for dispatched calls. May be shorter
            than tool_calls if a terminal result caused short-circuit.
        """
        results: list[ToolResult] = []
        for call in tool_calls:
            logger.debug("Dispatching tool call: %s", call.tool_name)
            result = await self._tool_dispatcher.dispatch(call)
            results.append(result)

            if result.is_terminal:
                logger.info(
                    "Terminal result from %s (status=%s) -- "
                    "short-circuiting remaining tool calls",
                    call.tool_name,
                    result.status.value,
                )
                break

        return tuple(results)

    def _observing_phase(
        self,
        tool_calls: tuple[ToolCall, ...],
        results: tuple[ToolResult, ...],
    ) -> bool:
        """Execute the OBSERVING phase: append results to history.

        Appends the assistant message (with tool_calls) and the tool
        result messages to the conversation history. Checks for terminal
        results (DENIED) that should stop the loop.

        Args:
            tool_calls: The tool calls that were dispatched.
            results: The results from dispatching.

        Returns:
            True if the loop should terminate (terminal result found).
        """
        # Append assistant message with tool_calls
        self._history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [call.to_openai_tool_call() for call in tool_calls],
        })

        # Append each tool result as a tool message
        should_terminate = False
        terminal_message: str | None = None

        for result in results:
            self._history.append(result.to_openai_tool_message())

            if result.is_terminal:
                should_terminate = True
                terminal_message = (
                    f"Tool '{result.tool_name}' was denied: "
                    f"{result.error_message or 'User denied the operation'}"
                )
                logger.info(
                    "Terminal tool result: %s (status=%s)",
                    result.tool_name,
                    result.status.value,
                )

        if should_terminate:
            self._state = AgentLoopState.ERROR
            self._error_message = terminal_message
            return True

        return False

    # -- History management ---------------------------------------------------

    def _initialize_history(self, user_message: str) -> None:
        """Set up the initial conversation history.

        Creates the system prompt message and user message that form
        the starting context for the LLM.

        Args:
            user_message: The natural-language user command.
        """
        self._history = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]
        self._error_message: str | None = None
        self._retry_exhausted = False

    # -- Result builder -------------------------------------------------------

    def _build_result(
        self,
        iterations_used: int,
        error_message: str | None = None,
    ) -> AgentLoopResult:
        """Build an immutable AgentLoopResult snapshot.

        Args:
            iterations_used: Number of iterations consumed.
            error_message: Override error message. If None, uses the
                internally tracked error message (if any).

        Returns:
            Frozen AgentLoopResult.
        """
        effective_error = error_message or getattr(self, "_error_message", None)

        return AgentLoopResult(
            final_state=self._state,
            iterations_used=iterations_used,
            history=tuple(self._history),
            error_message=effective_error,
            retry_exhausted=self._retry_exhausted,
        )
