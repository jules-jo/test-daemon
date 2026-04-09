"""LLM response parsing, validation, and command translation.

Converts raw LLM text output into validated, immutable data structures
and orchestrates the end-to-end prompt -> LLM call -> parse pipeline.

Three-layer parsing pipeline:
    1. **JSON extraction** -- pull JSON from raw LLM text (handles plain
       JSON, markdown code fences, and JSON embedded in prose)
    2. **Schema validation** -- validate the extracted dict against the
       LLMCommandResponse Pydantic model
    3. **SSHCommand mapping** -- convert validated response steps into
       SSHCommand objects for the SSH execution layer

High-level entry point::

    from jules_daemon.llm.response_parser import translate_command
    from jules_daemon.llm.config import load_config_from_env
    from jules_daemon.llm.prompts import HostContext

    config = load_config_from_env()
    host = HostContext(hostname="staging.example.com", user="deploy")
    result = translate_command(
        natural_language="run the smoke tests",
        host_context=host,
        config=config,
    )
    for cmd in result.ssh_commands:
        print(cmd.command)

All models are frozen (immutable) Pydantic models.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from jules_daemon.llm.client import create_client, create_completion
from jules_daemon.llm.config import LLMConfig
from jules_daemon.llm.errors import LLMParseError
from jules_daemon.llm.prompts import HostContext, PromptConfig, build_messages
from jules_daemon.ssh.command import SSHCommand

logger = logging.getLogger(__name__)

__all__ = [
    "Confidence",
    "LLMCommandStep",
    "LLMCommandResponse",
    "TranslateResult",
    "extract_json_from_text",
    "parse_llm_response",
    "response_to_ssh_commands",
    "translate_command",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Confidence(Enum):
    """LLM's confidence in its generated commands."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Response models (immutable)
# ---------------------------------------------------------------------------


class LLMCommandStep(BaseModel):
    """Single command step in an LLM response.

    Attributes:
        command: Shell command string to execute.
        description: Human-readable description of what the command does.
        working_directory: Optional absolute path for cd before execution.
        timeout_seconds: Maximum execution time in seconds.
    """

    model_config = ConfigDict(frozen=True)

    command: str
    description: str
    working_directory: str | None = None
    timeout_seconds: int = 300

    @field_validator("command", mode="before")
    @classmethod
    def _strip_and_validate_command(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("command must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("command must not be empty or whitespace-only")
        return stripped

    @field_validator("description", mode="before")
    @classmethod
    def _strip_and_validate_description(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("description must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("description must not be empty or whitespace-only")
        return stripped

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def _validate_timeout(cls, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("timeout_seconds must be an integer")
        if value < 1:
            raise ValueError("timeout_seconds must be at least 1")
        return value


class LLMCommandResponse(BaseModel):
    """Complete validated LLM response containing proposed commands.

    Attributes:
        commands: List of command steps to execute sequentially.
        explanation: Brief explanation of the overall plan.
        confidence: LLM's confidence level in the generated commands.
        warnings: List of risk or caveat strings.
    """

    model_config = ConfigDict(frozen=True)

    commands: list[LLMCommandStep]
    explanation: str
    confidence: Confidence
    warnings: list[str] = Field(default_factory=list)

    @field_validator("explanation", mode="before")
    @classmethod
    def _strip_and_validate_explanation(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("explanation must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("explanation must not be empty or whitespace-only")
        return stripped

    @property
    def is_refusal(self) -> bool:
        """True if the LLM refused the request (no commands generated)."""
        return len(self.commands) == 0


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

# Pattern for ```json ... ``` code fences
_CODE_FENCE_JSON_RE = re.compile(
    r"```json\s*\n(.*?)```",
    re.DOTALL,
)

# Pattern for ``` ... ``` code fences (plain, no language tag)
_CODE_FENCE_PLAIN_RE = re.compile(
    r"```\s*\n(.*?)```",
    re.DOTALL,
)

# Pattern for finding JSON objects in text (brace matching).
# NOTE: This regex handles up to two levels of brace nesting. Deeply nested
# JSON (3+ levels) embedded in prose without code fences will not match.
# Strategies 1-3 (full parse, ```json, ```) handle those cases; this is a
# best-effort fallback for simple embedded objects.
_JSON_OBJECT_RE = re.compile(
    r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
    re.DOTALL,
)


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM response text.

    Tries these strategies in order:
    1. Parse the entire text as JSON
    2. Extract from ```json ... ``` code fence
    3. Extract from ``` ... ``` code fence
    4. Find the first JSON object in the text

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed JSON as a dict.

    Raises:
        LLMParseError: If no valid JSON object can be extracted.
    """
    stripped = text.strip()
    if not stripped:
        raise LLMParseError("No JSON found in empty response", raw_content=text)

    # Strategy 1: Try parsing the whole text as JSON
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            raise LLMParseError(
                "Expected a JSON object, got a JSON array",
                raw_content=text,
            )
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from ```json ... ``` code fence
    match = _CODE_FENCE_JSON_RE.search(stripped)
    if match:
        return _try_parse_json_block(match.group(1).strip(), text)

    # Strategy 3: Extract from ``` ... ``` code fence
    match = _CODE_FENCE_PLAIN_RE.search(stripped)
    if match:
        block = match.group(1).strip()
        try:
            return _try_parse_json_block(block, text)
        except LLMParseError:
            pass  # Fall through to strategy 4

    # Strategy 4: Find JSON objects in the text
    for match in _JSON_OBJECT_RE.finditer(stripped):
        try:
            candidate = match.group(0)
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    # Check if there's something that looks like JSON but is malformed
    if "{" in stripped:
        raise LLMParseError(
            "Invalid JSON found in response",
            raw_content=text,
        )

    raise LLMParseError("No JSON found in LLM response", raw_content=text)


def _try_parse_json_block(block: str, original_text: str) -> dict[str, Any]:
    """Parse a text block as JSON, raising LLMParseError on failure."""
    try:
        result = json.loads(block)
    except json.JSONDecodeError as exc:
        raise LLMParseError(
            f"Invalid JSON in code block: {exc}",
            raw_content=original_text,
        ) from exc

    if not isinstance(result, dict):
        raise LLMParseError(
            "Expected a JSON object in code block",
            raw_content=original_text,
        )
    return result


# ---------------------------------------------------------------------------
# Full parse pipeline
# ---------------------------------------------------------------------------


def parse_llm_response(text: str) -> LLMCommandResponse:
    """Parse raw LLM response text into a validated LLMCommandResponse.

    Pipeline:
    1. Extract JSON from the text (handles code fences, mixed text, etc.)
    2. Validate against the LLMCommandResponse schema

    Args:
        text: Raw LLM response text.

    Returns:
        Validated LLMCommandResponse.

    Raises:
        LLMParseError: If JSON extraction or schema validation fails.
    """
    data = extract_json_from_text(text)

    try:
        return LLMCommandResponse.model_validate(data)
    except ValidationError as exc:
        # Identify which field caused the issue for a helpful message
        error_str = str(exc)
        for field_name in ("commands", "explanation", "confidence"):
            if field_name in error_str.lower():
                raise LLMParseError(
                    f"Invalid or missing '{field_name}' in LLM response: {exc}",
                    raw_content=text,
                ) from exc
        raise LLMParseError(
            f"Failed to validate LLM response: {exc}",
            raw_content=text,
        ) from exc


# ---------------------------------------------------------------------------
# Mapping to SSHCommand
# ---------------------------------------------------------------------------


def response_to_ssh_commands(
    response: LLMCommandResponse,
) -> list[SSHCommand]:
    """Convert an LLMCommandResponse into a list of SSHCommand objects.

    Each LLMCommandStep is mapped to an SSHCommand with:
    - command -> command
    - working_directory -> working_directory
    - timeout_seconds -> timeout

    Args:
        response: Validated LLM response.

    Returns:
        List of SSHCommand objects. Empty list if the response is a refusal.

    Raises:
        LLMParseError: If an SSHCommand cannot be constructed from a step
            (e.g., invalid working directory).
    """
    if response.is_refusal:
        return []

    result: list[SSHCommand] = []
    for step in response.commands:
        try:
            ssh_cmd = SSHCommand(
                command=step.command,
                working_directory=step.working_directory,
                timeout=step.timeout_seconds,
            )
        except ValidationError as exc:
            raise LLMParseError(
                f"Cannot create SSHCommand from LLM step: {exc}",
                raw_content=step.command,
            ) from exc
        result.append(ssh_cmd)

    return result


# ---------------------------------------------------------------------------
# TranslateResult container
# ---------------------------------------------------------------------------


class TranslateResult(BaseModel):
    """Container for the full translate_command() output.

    Bundles the validated SSHCommands with the raw LLM response data
    for audit trail and debugging purposes.

    Attributes:
        ssh_commands: Validated SSHCommand objects ready for approval.
        llm_response: The parsed LLMCommandResponse.
        raw_content: The raw text content from the LLM response.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    ssh_commands: list[SSHCommand]
    llm_response: LLMCommandResponse
    raw_content: str

    @property
    def is_refusal(self) -> bool:
        """True if the LLM declined to generate commands."""
        return self.llm_response.is_refusal


# ---------------------------------------------------------------------------
# High-level: translate natural language -> SSHCommands via LLM
# ---------------------------------------------------------------------------


def _extract_content(completion: ChatCompletion) -> str:
    """Extract the text content from a ChatCompletion response.

    Args:
        completion: OpenAI ChatCompletion response.

    Returns:
        The assistant message content string.

    Raises:
        LLMParseError: If the response has no choices or empty content.
    """
    if not completion.choices:
        raise LLMParseError(
            "No choices in LLM response",
            raw_content=str(completion),
        )

    content = completion.choices[0].message.content
    if not content:
        raise LLMParseError(
            "LLM response content is empty or null",
            raw_content=str(completion),
        )

    return content


def translate_command(
    *,
    natural_language: str,
    host_context: HostContext,
    config: LLMConfig,
    prompt_config: PromptConfig | None = None,
    client: OpenAI | None = None,
) -> TranslateResult:
    """Translate a natural-language request into validated SSH commands.

    End-to-end pipeline:
    1. Build prompts from the natural-language request and host context
    2. Send to Dataiku Mesh LLM via the OpenAI SDK
    3. Extract JSON from the response
    4. Validate against the response schema
    5. Map to SSHCommand objects

    The caller is responsible for presenting the resulting commands to
    the user for approval before execution. No commands are executed
    by this function.

    Args:
        natural_language: User's plain-English request (e.g.,
            "run the smoke tests on staging").
        host_context: Connection details and environment hints.
        config: LLM client configuration.
        prompt_config: Optional prompt construction overrides.
        client: Optional pre-existing OpenAI client. If None, a new
            client is created from config. Pass an existing client
            for connection reuse across multiple calls.

    Returns:
        TranslateResult with validated SSHCommands, the parsed
        LLM response, and raw content for audit.

    Raises:
        LLMAuthenticationError: Invalid API key.
        LLMConnectionError: Cannot reach Dataiku Mesh endpoint.
        LLMResponseError: API returned non-success status.
        LLMParseError: Response cannot be parsed into valid commands.
    """
    # Build the message list
    messages = build_messages(
        natural_language=natural_language,
        host_context=host_context,
        config=prompt_config,
    )

    # Create or reuse client
    effective_client = client if client is not None else create_client(config)

    logger.info(
        "Translating command: %r -> LLM call to %s",
        natural_language,
        config.default_model,
    )

    # Call the LLM
    completion = create_completion(
        client=effective_client,
        config=config,
        messages=messages,
    )

    # Extract content from the ChatCompletion
    raw_content = _extract_content(completion)

    logger.debug("LLM raw content: %.500s", raw_content)

    # Parse the response into validated models
    llm_response = parse_llm_response(raw_content)

    # Map to SSHCommand objects
    ssh_commands = response_to_ssh_commands(llm_response)

    logger.info(
        "Translation complete: %d commands, confidence=%s, refusal=%s",
        len(ssh_commands),
        llm_response.confidence.value,
        llm_response.is_refusal,
    )

    return TranslateResult(
        ssh_commands=ssh_commands,
        llm_response=llm_response,
        raw_content=raw_content,
    )
