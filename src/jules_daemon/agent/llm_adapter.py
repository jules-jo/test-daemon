"""OpenAI SDK adapter for the AgentLoop LLMClient protocol.

Wraps the OpenAI Chat Completions API client to satisfy the
LLMClient protocol required by AgentLoop. Provides two levels
of LLM interaction:

1. ``call_completion()``: Core LLM API call wrapper that sends the
   current message list to the LLM endpoint and returns the raw
   ChatCompletion response. Handles retry logic for transient errors
   (network blips, timeouts, rate limits), per-attempt timeout
   enforcement via ``asyncio.wait_for``, and error classification
   into transient vs permanent categories.

2. ``get_tool_calls()``: Higher-level method that calls
   ``call_completion()`` and parses the response into ToolCall tuples.
   Maintained for backward compatibility with the LLMClient protocol.

Error classification:
    - Transient (retryable): ConnectionError, TimeoutError, OSError,
      asyncio.TimeoutError, HTTP 408/429/500/502/503/504,
      OpenAI APIConnectionError, RateLimitError, APITimeoutError.
    - Permanent (not retryable): AuthenticationError, ValueError,
      HTTP 400/401/403/404, all others.

Usage::

    from jules_daemon.agent.llm_adapter import OpenAILLMAdapter

    adapter = OpenAILLMAdapter(
        client=openai_client,
        model="provider:connection:model",
        tool_schemas=registry.to_openai_schemas(),
    )

    # Low-level: raw response with retry/timeout handling
    result = await adapter.call_completion(messages, max_retries=2)
    raw_response = result.response

    # High-level: parsed tool calls (LLMClient protocol)
    tool_calls = await adapter.get_tool_calls(messages)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from jules_daemon.agent.tool_types import ToolCall

__all__ = [
    "LLMCallError",
    "LLMCallErrorKind",
    "LLMCallResult",
    "OpenAILLMAdapter",
    "classify_sdk_error",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class LLMCallErrorKind(Enum):
    """Classification of LLM call errors for retry decisions.

    TRANSIENT: Network blip, timeout, rate limit -- safe to retry.
        The underlying issue is likely temporary and a subsequent
        attempt may succeed.

    PERMANENT: Auth failure, malformed request, invalid model --
        retrying will not help. The loop should terminate or fall
        back to the one-shot path.
    """

    TRANSIENT = "transient"
    PERMANENT = "permanent"


class LLMCallError(Exception):
    """Error from an LLM API call with transient/permanent classification.

    Wraps the original exception with metadata to help the agent loop
    decide whether to retry (transient) or terminate (permanent).

    Attributes:
        kind: Whether this error is transient (retryable) or permanent.
        cause: The original exception that triggered this error.
        attempts: Number of attempts made before this error was raised.
    """

    def __init__(
        self,
        message: str,
        kind: LLMCallErrorKind,
        *,
        cause: BaseException | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.cause = cause
        self.attempts = attempts

    @property
    def is_transient(self) -> bool:
        """True if the error is transient and could be retried."""
        return self.kind is LLMCallErrorKind.TRANSIENT

    @property
    def is_permanent(self) -> bool:
        """True if the error is permanent and retrying will not help."""
        return self.kind is LLMCallErrorKind.PERMANENT


# ---------------------------------------------------------------------------
# Call result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMCallResult:
    """Immutable result of a successful LLM API call.

    Wraps the raw ChatCompletion response with metadata about timing
    and retry behavior, enabling callers to observe how the call
    performed without parsing the response themselves.

    Attributes:
        response: The raw OpenAI ChatCompletion response object.
        elapsed_seconds: Wall-clock time for the entire call sequence
            (includes any retry time if retries occurred).
        attempts: Total number of attempts made (1 = first try succeeded,
            2 = one retry needed, etc.).
        model: The model identifier used for the call.
    """

    response: Any  # openai.types.chat.ChatCompletion
    elapsed_seconds: float
    attempts: int
    model: str


# ---------------------------------------------------------------------------
# Transient error detection
# ---------------------------------------------------------------------------

# Built-in Python exceptions that indicate transient network/IO issues.
_TRANSIENT_BUILTIN_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# HTTP status codes that indicate transient server-side issues.
# These are safe to retry because the server may recover.
_TRANSIENT_HTTP_STATUS_CODES: frozenset[int] = frozenset({
    408,  # Request Timeout
    429,  # Too Many Requests (rate limit)
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
})


def classify_sdk_error(exc: BaseException) -> LLMCallErrorKind:
    """Classify an exception from the OpenAI SDK as transient or permanent.

    Checks multiple signals in priority order:
    1. Built-in Python exception types (ConnectionError, TimeoutError, OSError)
    2. OpenAI SDK-specific exception types (APIConnectionError, RateLimitError)
    3. HTTP status code on the exception (429, 5xx = transient)
    4. Default: permanent

    Args:
        exc: The exception raised during the LLM SDK call.

    Returns:
        TRANSIENT for retryable errors, PERMANENT for everything else.
    """
    # 1. Built-in network/OS errors are always transient
    if isinstance(exc, _TRANSIENT_BUILTIN_ERRORS):
        return LLMCallErrorKind.TRANSIENT

    # 2. OpenAI SDK-specific error types
    try:
        import openai

        if isinstance(exc, openai.APIConnectionError):
            return LLMCallErrorKind.TRANSIENT
        if isinstance(exc, openai.APITimeoutError):
            return LLMCallErrorKind.TRANSIENT
        if isinstance(exc, openai.RateLimitError):
            return LLMCallErrorKind.TRANSIENT
    except ImportError:
        pass  # openai not available; skip SDK-specific checks

    # 3. HTTP status code on the exception object (generic check)
    status_code = getattr(exc, "status_code", None)
    if status_code is not None and status_code in _TRANSIENT_HTTP_STATUS_CODES:
        return LLMCallErrorKind.TRANSIENT

    # 4. Default: permanent
    return LLMCallErrorKind.PERMANENT


# ---------------------------------------------------------------------------
# OpenAI LLM Adapter
# ---------------------------------------------------------------------------


class OpenAILLMAdapter:
    """Adapts an OpenAI SDK client to the AgentLoop LLMClient protocol.

    The adapter is stateless -- it translates between the OpenAI SDK's
    synchronous interface and the async protocols expected by the agent
    loop. All blocking SDK calls are offloaded to a thread pool via
    ``asyncio.to_thread``.

    Provides two levels of interaction:

    - ``call_completion()``: Low-level wrapper with retry and timeout
      handling. Returns the raw ChatCompletion response wrapped in an
      ``LLMCallResult``. This is the core function for the agent loop's
      think phase.

    - ``get_tool_calls()``: High-level method that calls
      ``call_completion()`` and parses the response into ToolCall tuples.
      Satisfies the ``LLMClient`` protocol for backward compatibility.

    Args:
        client: Pre-configured OpenAI SDK client instance.
        model: Model identifier (provider:connection:model format).
        tool_schemas: OpenAI-compatible function tool schemas from
            ToolRegistry.to_openai_schemas().
        default_max_retries: Default number of retry attempts for
            transient errors in ``call_completion()``. The total number
            of attempts is ``max_retries + 1``.
        default_timeout: Default per-attempt timeout in seconds for
            ``call_completion()``. None means no timeout enforcement.
    """

    def __init__(
        self,
        *,
        client: Any,  # openai.OpenAI -- kept as Any for decoupling
        model: str,
        tool_schemas: tuple[dict[str, Any], ...],
        default_max_retries: int = 2,
        default_timeout: float | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._tool_schemas = list(tool_schemas)
        self._default_max_retries = default_max_retries
        self._default_timeout = default_timeout

    # -- Core LLM API call wrapper -------------------------------------------

    async def call_completion(
        self,
        messages: tuple[dict[str, Any], ...],
        *,
        max_retries: int | None = None,
        timeout: float | None = ...,  # type: ignore[assignment]
    ) -> LLMCallResult:
        """Send messages to the LLM and return the raw completion response.

        This is the core LLM API call wrapper for the agent loop. It:

        1. Offloads the blocking OpenAI SDK call to a thread pool.
        2. Enforces a per-attempt timeout via ``asyncio.wait_for``.
        3. Retries transient errors (network, timeout, rate limit) up
           to ``max_retries`` times within the same call.
        4. Classifies errors as transient or permanent and wraps them
           in ``LLMCallError`` with metadata.

        A single call to ``call_completion`` corresponds to one
        think-act cycle's LLM interaction. The agent loop calls this
        once per iteration and handles the response.

        Args:
            messages: Immutable conversation history in OpenAI format.
                Includes system prompt, user message, and any prior
                assistant/tool messages from earlier iterations.
            max_retries: Maximum retry attempts for transient errors.
                The total number of attempts is ``max_retries + 1``.
                Defaults to the adapter's ``default_max_retries``.
            timeout: Per-attempt timeout in seconds. Each attempt gets
                the full timeout budget. ``None`` disables timeout
                enforcement. Defaults to ``default_timeout``.

        Returns:
            LLMCallResult wrapping the raw ChatCompletion response,
            timing metadata, and attempt count.

        Raises:
            LLMCallError: On transient error after all retries exhausted
                (kind=TRANSIENT), or immediately on permanent error
                (kind=PERMANENT). The ``cause`` attribute contains the
                original exception.
        """
        effective_retries = (
            max_retries if max_retries is not None
            else self._default_max_retries
        )
        effective_timeout = (
            timeout if timeout is not ...
            else self._default_timeout
        )

        messages_list = list(messages)
        start_time = time.monotonic()
        last_error: BaseException | None = None
        total_attempts = effective_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                response = await self._execute_with_timeout(
                    messages_list,
                    effective_timeout,
                )

                elapsed = time.monotonic() - start_time

                logger.debug(
                    "LLM call succeeded on attempt %d/%d in %.3fs",
                    attempt,
                    total_attempts,
                    elapsed,
                )

                return LLMCallResult(
                    response=response,
                    elapsed_seconds=elapsed,
                    attempts=attempt,
                    model=self._model,
                )

            except asyncio.TimeoutError as exc:
                last_error = exc
                logger.warning(
                    "LLM call attempt %d/%d timed out (timeout=%.1fs)",
                    attempt,
                    total_attempts,
                    effective_timeout or 0,
                )
                # asyncio.TimeoutError is always transient -- continue

            except Exception as exc:
                last_error = exc
                kind = classify_sdk_error(exc)

                if kind is LLMCallErrorKind.PERMANENT:
                    elapsed = time.monotonic() - start_time
                    logger.error(
                        "Permanent LLM error on attempt %d: %s",
                        attempt,
                        exc,
                    )
                    raise LLMCallError(
                        f"Permanent LLM error: {exc}",
                        kind=LLMCallErrorKind.PERMANENT,
                        cause=exc,
                        attempts=attempt,
                    ) from exc

                # Transient error -- log and continue to next attempt
                logger.warning(
                    "Transient LLM error on attempt %d/%d: %s",
                    attempt,
                    total_attempts,
                    exc,
                )

        # All attempts exhausted with transient errors
        elapsed = time.monotonic() - start_time
        raise LLMCallError(
            f"LLM call failed after {total_attempts} attempts: {last_error}",
            kind=LLMCallErrorKind.TRANSIENT,
            cause=last_error,
            attempts=total_attempts,
        )

    # -- High-level LLMClient protocol method --------------------------------

    async def get_tool_calls(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        """Send messages to the LLM and extract tool calls.

        Offloads the blocking OpenAI SDK call to a thread pool executor
        to avoid stalling the asyncio event loop.

        This method maintains backward compatibility with the LLMClient
        protocol. For the agent loop's think phase with retry/timeout
        handling, use ``call_completion()`` instead.

        Args:
            messages: Immutable conversation history.

        Returns:
            Tuple of ToolCall instances. Empty tuple signals completion.

        Raises:
            ConnectionError, TimeoutError: Transient errors (retryable).
            ValueError: Permanent errors (not retryable).
        """
        messages_list = list(messages)

        try:
            response = await asyncio.to_thread(
                self._call_llm,
                messages_list,
            )
        except (ConnectionError, TimeoutError, OSError):
            raise
        except Exception as exc:
            raise ValueError(f"LLM call failed: {exc}") from exc

        return self._parse_tool_calls(response)

    # -- Internal helpers ----------------------------------------------------

    async def _execute_with_timeout(
        self,
        messages: list[dict[str, Any]],
        timeout: float | None,
    ) -> Any:
        """Execute the synchronous LLM call with optional timeout.

        Offloads the blocking SDK call to a thread pool and optionally
        wraps it with ``asyncio.wait_for`` for timeout enforcement.

        Args:
            messages: Message list for the SDK call.
            timeout: Per-attempt timeout in seconds, or None for no limit.

        Returns:
            Raw ChatCompletion response object.

        Raises:
            asyncio.TimeoutError: If the call exceeds the timeout.
            Any exception from the underlying SDK call.
        """
        coro = asyncio.to_thread(self._call_llm, messages)

        if timeout is not None:
            return await asyncio.wait_for(coro, timeout=timeout)

        return await coro

    def _call_llm(self, messages: list[dict[str, Any]]) -> Any:
        """Synchronous LLM call (runs in thread pool).

        Sends the conversation history with tool schemas to the OpenAI
        Chat Completions API and returns the raw response object.

        Args:
            messages: Message list in OpenAI format.

        Returns:
            Raw ChatCompletion response from the SDK.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if self._tool_schemas:
            kwargs["tools"] = self._tool_schemas

        return self._client.chat.completions.create(**kwargs)

    @staticmethod
    def _parse_tool_calls(response: Any) -> tuple[ToolCall, ...]:
        """Parse tool calls from an OpenAI completion response.

        Handles the case where the LLM returns no tool calls (signals
        completion) or returns one or more tool calls. Malformed JSON
        in arguments defaults to an empty dict with a warning.

        Args:
            response: The raw OpenAI completion response.

        Returns:
            Tuple of ToolCall instances, or empty tuple on completion.
        """
        if not response.choices:
            return ()

        message = response.choices[0].message
        if not message.tool_calls:
            return ()

        calls: list[ToolCall] = []
        for tc in message.tool_calls:
            try:
                arguments = (
                    json.loads(tc.function.arguments)
                    if tc.function.arguments
                    else {}
                )
            except json.JSONDecodeError:
                arguments = {}
                logger.warning(
                    "Failed to parse tool call arguments for %s: %s",
                    tc.function.name,
                    tc.function.arguments,
                )

            calls.append(
                ToolCall(
                    call_id=tc.id,
                    tool_name=tc.function.name,
                    arguments=arguments,
                )
            )

        return tuple(calls)
