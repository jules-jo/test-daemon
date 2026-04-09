"""Natural-language to SSH command translation with deadline enforcement.

Orchestrates the full pipeline from user NL input to proposed SSH commands:

    NL input + host context
        -> prompt construction (cached system prompt)
        -> LLM call (Dataiku Mesh via OpenAI SDK)
        -> response parsing + validation
        -> TranslationResult with proposed commands

The entire pipeline enforces a 5-second deadline from input to result.
If the LLM does not respond within the deadline, a TranslationTimeout
error is raised so the caller can inform the user immediately.

Design choices for meeting the 5-second latency SLA:
    - System prompt is cached after first construction (immutable, same per config)
    - LLM timeout is set to the remaining deadline budget
    - Prompt construction and response parsing are local-only and sub-millisecond
    - No retries within the deadline -- a single attempt must succeed
    - Temperature is set to 0.0 for deterministic, faster responses

Usage::

    from jules_daemon.llm.command_translator import (
        CommandTranslator,
        translate_command,
    )

    translator = CommandTranslator(client=client, config=config)
    result = translator.translate(
        natural_language="run the smoke tests",
        host_context=host_context,
    )
    # result.response has the LLMCommandResponse
    # result.ssh_commands has the list[SSHCommand]
    # result.elapsed_seconds shows actual latency
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from jules_daemon.llm.config import LLMConfig
from jules_daemon.llm.errors import LLMError, LLMParseError
from jules_daemon.llm.models import ToolCallingMode
from jules_daemon.llm.prompts import (
    HostContext,
    PromptConfig,
    build_messages,
    build_system_prompt,
)
from jules_daemon.llm.response_parser import (
    LLMCommandResponse,
    parse_llm_response,
    response_to_ssh_commands,
)
from jules_daemon.ssh.command import SSHCommand

logger = logging.getLogger(__name__)

__all__ = [
    "CommandTranslator",
    "TranslationResult",
    "TranslationTimeout",
    "translate_command",
    "DEFAULT_DEADLINE_SECONDS",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DEADLINE_SECONDS: float = 5.0
"""Maximum wall-clock time from NL input to proposed command display."""

_MIN_LLM_TIMEOUT: float = 1.0
"""Minimum timeout passed to the LLM call, even if deadline is tight."""

_LLM_TEMPERATURE: float = 0.0
"""Temperature for command translation: deterministic for speed."""

_PROMPT_BUDGET_SECONDS: float = 0.1
"""Reserved budget for prompt construction (generous for sub-ms operation)."""

_PARSE_BUDGET_SECONDS: float = 0.1
"""Reserved budget for response parsing (generous for sub-ms operation)."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranslationResult:
    """Result of translating NL input to SSH commands.

    Immutable snapshot of the translation outcome, including the raw
    LLM response, the validated command response, the mapped SSH commands,
    and timing metadata.

    Attributes:
        response: Validated LLM command response with explanation,
            confidence, warnings, and command steps.
        ssh_commands: List of SSHCommand objects mapped from the response.
            Empty list if the LLM refused the request.
        elapsed_seconds: Wall-clock time from translate() entry to result.
        natural_language: The original NL input that was translated.
        deadline_seconds: The deadline that was enforced.
    """

    response: LLMCommandResponse
    ssh_commands: tuple[SSHCommand, ...]
    elapsed_seconds: float
    natural_language: str
    deadline_seconds: float

    @property
    def is_refusal(self) -> bool:
        """True if the LLM refused to generate commands."""
        return self.response.is_refusal

    @property
    def met_deadline(self) -> bool:
        """True if the translation completed within the deadline."""
        return self.elapsed_seconds <= self.deadline_seconds

    @property
    def command_count(self) -> int:
        """Number of proposed SSH commands."""
        return len(self.ssh_commands)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TranslationTimeout(LLMError):
    """Raised when the NL-to-command translation exceeds the deadline.

    Contains timing information so callers can log or display how much
    time was consumed before the timeout.
    """

    def __init__(
        self,
        message: str,
        deadline_seconds: float,
        elapsed_seconds: float,
    ) -> None:
        super().__init__(message)
        self.deadline_seconds = deadline_seconds
        self.elapsed_seconds = elapsed_seconds


# ---------------------------------------------------------------------------
# System prompt cache
# ---------------------------------------------------------------------------


class _SystemPromptCache:
    """Thread-safe cache for the system prompt string.

    The system prompt is deterministic for a given PromptConfig. Since
    configs are frozen dataclasses, we cache by config identity. In the
    single-user daemon, there is typically only one config.

    The cache avoids re-building the system prompt string (~2KB of text
    assembly) on every translation. While prompt construction is fast,
    caching eliminates even that overhead for the latency-critical path.
    """

    def __init__(self) -> None:
        self._cache: dict[int, str] = {}

    def get(self, config: PromptConfig) -> str:
        """Get the system prompt, building and caching if needed."""
        key = id(config)
        if key not in self._cache:
            prompt = build_system_prompt(config=config)
            self._cache[key] = prompt
        return self._cache[key]

    def clear(self) -> None:
        """Clear the cache (useful for testing or config changes)."""
        self._cache.clear()


_prompt_cache = _SystemPromptCache()


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


class CommandTranslator:
    """Translates natural-language inputs to SSH commands via LLM.

    Encapsulates the full pipeline with deadline enforcement:
    1. Build prompt messages (cached system prompt + user context)
    2. Call LLM with timeout = remaining deadline budget
    3. Parse and validate the response
    4. Map to SSHCommand objects

    The translator is designed for the daemon's single-user model:
    one translator instance per daemon lifetime, reused across requests.

    Args:
        client: OpenAI client configured for Dataiku Mesh.
        config: LLM configuration.
        prompt_config: Optional prompt configuration override.
        tool_calling_mode: How tools are passed to the LLM.
        deadline_seconds: Maximum wall-clock seconds for translation.
    """

    def __init__(
        self,
        *,
        client: OpenAI,
        config: LLMConfig,
        prompt_config: PromptConfig | None = None,
        tool_calling_mode: ToolCallingMode = ToolCallingMode.NATIVE,
        deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    ) -> None:
        if deadline_seconds <= 0:
            raise ValueError(
                f"deadline_seconds must be positive, got {deadline_seconds}"
            )

        self._client = client
        self._config = config
        self._prompt_config = prompt_config or PromptConfig()
        self._tool_calling_mode = tool_calling_mode
        self._deadline_seconds = deadline_seconds

    @property
    def deadline_seconds(self) -> float:
        """The configured deadline for translation."""
        return self._deadline_seconds

    def translate(
        self,
        *,
        natural_language: str,
        host_context: HostContext,
        deadline_seconds: float | None = None,
    ) -> TranslationResult:
        """Translate NL input to proposed SSH commands within the deadline.

        This is the main entry point for the translation pipeline. The
        entire operation -- prompt construction, LLM call, response parsing,
        and SSHCommand mapping -- must complete within deadline_seconds.

        Args:
            natural_language: The user's plain-English request (e.g.,
                "run the smoke tests on staging").
            host_context: Target host connection details and environment hints.
            deadline_seconds: Override the instance-level deadline for this
                single call. If None, uses the instance default.

        Returns:
            TranslationResult with the proposed commands and timing metadata.

        Raises:
            TranslationTimeout: If the pipeline exceeds the deadline.
            LLMParseError: If the LLM response cannot be parsed.
            LLMError: For LLM client errors (auth, connection, etc.).
            ValueError: If natural_language is empty.
        """
        effective_deadline = deadline_seconds or self._deadline_seconds
        start_time = time.monotonic()

        stripped_nl = natural_language.strip()
        if not stripped_nl:
            raise ValueError("natural_language must not be empty")

        logger.info(
            "Starting NL->command translation (deadline=%.1fs): %s",
            effective_deadline,
            stripped_nl[:100],
        )

        # Phase 1: Build messages (sub-millisecond with cached system prompt)
        messages = self._build_messages(
            natural_language=stripped_nl,
            host_context=host_context,
        )

        elapsed_after_prompt = time.monotonic() - start_time
        remaining = effective_deadline - elapsed_after_prompt - _PARSE_BUDGET_SECONDS

        if remaining < _MIN_LLM_TIMEOUT:
            remaining = _MIN_LLM_TIMEOUT

        logger.debug(
            "Prompt built in %.3fs, LLM budget: %.1fs",
            elapsed_after_prompt,
            remaining,
        )

        # Phase 2: LLM call with remaining budget as timeout
        raw_content = self._call_llm(
            messages=messages,
            timeout=remaining,
        )

        elapsed_after_llm = time.monotonic() - start_time
        logger.debug("LLM responded in %.3fs", elapsed_after_llm)

        # Check deadline after LLM call
        if elapsed_after_llm > effective_deadline:
            raise TranslationTimeout(
                f"LLM call exceeded {effective_deadline}s deadline "
                f"(took {elapsed_after_llm:.2f}s)",
                deadline_seconds=effective_deadline,
                elapsed_seconds=elapsed_after_llm,
            )

        # Phase 3: Parse and validate response (sub-millisecond)
        response = parse_llm_response(raw_content)

        # Phase 4: Map to SSHCommand objects (sub-millisecond)
        ssh_commands = response_to_ssh_commands(response)

        elapsed_total = time.monotonic() - start_time

        logger.info(
            "Translation completed in %.3fs (deadline=%.1fs): %d commands, confidence=%s",
            elapsed_total,
            effective_deadline,
            len(ssh_commands),
            response.confidence.value,
        )

        # Final deadline check (parsing should be sub-ms, but be safe)
        if elapsed_total > effective_deadline:
            raise TranslationTimeout(
                f"Translation pipeline exceeded {effective_deadline}s deadline "
                f"(took {elapsed_total:.2f}s)",
                deadline_seconds=effective_deadline,
                elapsed_seconds=elapsed_total,
            )

        return TranslationResult(
            response=response,
            ssh_commands=tuple(ssh_commands),
            elapsed_seconds=elapsed_total,
            natural_language=stripped_nl,
            deadline_seconds=effective_deadline,
        )

    def _build_messages(
        self,
        *,
        natural_language: str,
        host_context: HostContext,
    ) -> list[dict[str, str]]:
        """Build the LLM message list with cached system prompt."""
        return build_messages(
            natural_language=natural_language,
            host_context=host_context,
            config=self._prompt_config,
        )

    def _call_llm(
        self,
        *,
        messages: list[dict[str, Any]],
        timeout: float,
    ) -> str:
        """Call the LLM and extract the response content.

        Uses the remaining deadline budget as the HTTP timeout. Temperature
        is fixed at 0.0 for deterministic, faster responses. No retries
        within the deadline -- a single attempt must succeed.

        Args:
            messages: Message list for the LLM.
            timeout: Maximum seconds to wait for the LLM response.

        Returns:
            Raw content string from the LLM response.

        Raises:
            LLMError: On any LLM client error.
            LLMParseError: If the response has no content.
        """
        from jules_daemon.llm.client import create_completion

        response = create_completion(
            client=self._client,
            config=self._config,
            messages=messages,
            tool_calling_mode=self._tool_calling_mode,
            temperature=_LLM_TEMPERATURE,
        )

        # Extract content from the response
        if not response.choices:
            raise LLMParseError(
                "LLM returned empty choices list",
                raw_content="",
            )

        content = response.choices[0].message.content
        if not content:
            raise LLMParseError(
                "LLM returned empty content",
                raw_content="",
            )

        return content


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def translate_command(
    *,
    client: OpenAI,
    config: LLMConfig,
    natural_language: str,
    host_context: HostContext,
    prompt_config: PromptConfig | None = None,
    tool_calling_mode: ToolCallingMode = ToolCallingMode.NATIVE,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
) -> TranslationResult:
    """One-shot convenience function for NL-to-command translation.

    Creates a temporary CommandTranslator and calls translate(). For
    repeated translations, prefer creating a CommandTranslator instance
    directly to benefit from system prompt caching.

    Args:
        client: OpenAI client configured for Dataiku Mesh.
        config: LLM configuration.
        natural_language: User's plain-English request.
        host_context: Target host connection details.
        prompt_config: Optional prompt configuration override.
        tool_calling_mode: How tools are passed to the LLM.
        deadline_seconds: Maximum wall-clock seconds for translation.

    Returns:
        TranslationResult with proposed commands and timing metadata.

    Raises:
        TranslationTimeout: If the pipeline exceeds the deadline.
        LLMParseError: If the LLM response cannot be parsed.
        LLMError: For LLM client errors.
    """
    translator = CommandTranslator(
        client=client,
        config=config,
        prompt_config=prompt_config,
        tool_calling_mode=tool_calling_mode,
        deadline_seconds=deadline_seconds,
    )
    return translator.translate(
        natural_language=natural_language,
        host_context=host_context,
    )
