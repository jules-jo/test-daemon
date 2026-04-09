"""Dataiku Mesh LLM client wrapper.

Creates and configures an OpenAI Python SDK client pointed at a Dataiku
Mesh endpoint. Handles:

- Custom base_url and API key authentication
- Model routing in ``provider:connection:model`` format
- Exclusion of Dataiku-unsupported parameters (parallel_tool_calls, n, seed, response_format)
- Native tool calling and prompt-based simulation fallback
- Error wrapping into the project's LLM error hierarchy

Usage::

    from jules_daemon.llm.config import load_config_from_env
    from jules_daemon.llm.client import create_client, create_completion

    config = load_config_from_env()
    client = create_client(config)
    response = create_completion(
        client=client,
        config=config,
        messages=[{"role": "user", "content": "Run tests on staging"}],
    )
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import httpx
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

from jules_daemon.llm.config import LLMConfig
from jules_daemon.llm.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    LLMToolCallingUnsupportedError,
)
from jules_daemon.llm.models import ToolCallingMode

logger = logging.getLogger(__name__)

# Parameters that Dataiku Mesh does not support (as of DSS 14.x).
# These are silently excluded to prevent backend errors.
_UNSUPPORTED_PARAMS = frozenset({
    "parallel_tool_calls",
    "n",
    "seed",
    "response_format",
})


def create_client(config: LLMConfig) -> OpenAI:
    """Create an OpenAI client configured for Dataiku Mesh.

    When ``config.verify_ssl`` is False, an ``httpx.Client`` with SSL
    verification disabled is created and passed to the OpenAI client.
    The OpenAI SDK does not own the lifecycle of an externally provided
    ``http_client``. For long-lived daemon processes, callers should
    close the returned client when it is no longer needed by calling
    ``client.close()``, which will close the underlying httpx client.

    Args:
        config: LLM configuration with base_url, api_key, and connection settings.

    Returns:
        Configured OpenAI client instance.
    """
    http_client: httpx.Client | None = None
    if not config.verify_ssl:
        http_client = httpx.Client(verify=False)

    return OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
        max_retries=config.max_retries,
        http_client=http_client,
    )


def _build_tool_prompt_section(tools: list[dict[str, Any]]) -> str:
    """Build a text description of tools for prompt-based tool calling.

    When the backend does not support native tool calling, tool definitions
    are injected into the system prompt as structured text. The LLM is
    instructed to respond with a JSON block to invoke a tool.

    Args:
        tools: List of OpenAI-format tool definitions.

    Returns:
        Formatted string to append to the system message.
    """
    lines = [
        "",
        "---",
        "AVAILABLE TOOLS:",
        "You have access to the following tools. To use a tool, respond with "
        "a JSON block in this exact format:",
        "",
        '```json',
        '{',
        '  "tool_calls": [',
        '    {',
        '      "name": "<function_name>",',
        '      "arguments": { ... }',
        '    }',
        '  ]',
        '}',
        '```',
        "",
        "If you do not need to use a tool, respond normally without the JSON block.",
        "",
    ]

    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "No description")
        params = func.get("parameters", {})

        lines.append(f"### {name}")
        lines.append(f"Description: {desc}")
        lines.append(f"Parameters: {json.dumps(params, indent=2)}")
        lines.append("")

    lines.append("---")
    return "\n".join(lines)


def _inject_tools_into_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Inject tool definitions into the system message for prompt-based mode.

    Creates a new message list via deep copy (immutable pattern). If a system
    message exists, appends the tool section. Otherwise, prepends a new system
    message.

    Args:
        messages: Original message list (not mutated).
        tools: Tool definitions to inject.

    Returns:
        New message list with tools in the system prompt.
    """
    tool_section = _build_tool_prompt_section(tools)

    # Deep copy to avoid sharing nested mutable objects with the caller
    result: list[dict[str, Any]] = []
    system_found = False

    for msg in messages:
        if msg.get("role") == "system" and not system_found:
            # Append tool section to existing system message
            copied = copy.deepcopy(msg)
            copied["content"] = msg.get("content", "") + tool_section
            result.append(copied)
            system_found = True
        else:
            result.append(copy.deepcopy(msg))

    if not system_found:
        # No system message exists -- prepend one
        result.insert(0, {
            "role": "system",
            "content": tool_section.lstrip(),
        })

    return result


def create_completion(
    *,
    client: OpenAI,
    config: LLMConfig,
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_calling_mode: ToolCallingMode = ToolCallingMode.NATIVE,
    **extra_kwargs: Any,
) -> ChatCompletion:
    """Create a chat completion via the Dataiku Mesh endpoint.

    Wraps ``client.chat.completions.create()`` with:
    - Default model routing from config
    - Automatic exclusion of Dataiku-unsupported parameters
    - Tool calling mode selection (native vs prompt-based)
    - Error wrapping into project error types

    Args:
        client: OpenAI client from ``create_client()``.
        config: LLM config for default model and settings.
        messages: Conversation messages in OpenAI format.
        model: Model override. If None, uses ``config.default_model``.
        tools: Tool definitions in OpenAI format. Handling depends on
            ``tool_calling_mode``.
        tool_calling_mode: Whether to use native API tool calling or
            inject tools into the system prompt.
        **extra_kwargs: Additional parameters passed to the API call.
            Unsupported Dataiku parameters are automatically filtered out.

    Returns:
        ChatCompletion response object.

    Raises:
        LLMAuthenticationError: Invalid or expired API key.
        LLMConnectionError: Cannot reach the Dataiku Mesh endpoint.
        LLMResponseError: API returned a non-success status.
        LLMToolCallingUnsupportedError: Unknown tool calling mode.
        LLMError: Any other unexpected error.
    """
    effective_model = model or config.default_model
    effective_messages = list(messages)

    # Build the API call kwargs
    kwargs: dict[str, Any] = {
        "model": effective_model,
    }

    # Handle tool calling mode
    if tools:
        if tool_calling_mode == ToolCallingMode.NATIVE:
            kwargs["tools"] = tools
        elif tool_calling_mode == ToolCallingMode.PROMPT_BASED:
            effective_messages = _inject_tools_into_messages(
                effective_messages, tools
            )
        else:
            raise LLMToolCallingUnsupportedError(
                f"Unsupported tool calling mode: {tool_calling_mode!r}"
            )

    kwargs["messages"] = effective_messages

    # Merge extra kwargs, filtering out unsupported Dataiku parameters
    for key, value in extra_kwargs.items():
        if key not in _UNSUPPORTED_PARAMS:
            kwargs[key] = value

    logger.debug(
        "LLM request: model=%s, messages=%d, tools=%s, mode=%s",
        effective_model,
        len(effective_messages),
        len(tools) if tools else 0,
        tool_calling_mode.value,
    )

    try:
        response = client.chat.completions.create(**kwargs)
    except openai.AuthenticationError as exc:
        raise LLMAuthenticationError(
            f"Invalid API key for Dataiku Mesh: {exc}"
        ) from exc
    except openai.APIConnectionError as exc:
        logger.debug("Connection failed to %s", config.base_url)
        raise LLMConnectionError(
            "Failed to connect to Dataiku Mesh endpoint: connection error"
        ) from exc
    except openai.APIStatusError as exc:
        raise LLMResponseError(
            f"Dataiku Mesh API error (HTTP {exc.status_code}): {exc}",
            status_code=exc.status_code,
        ) from exc
    except openai.OpenAIError as exc:
        raise LLMError(f"Unexpected LLM error: {exc}") from exc

    logger.debug(
        "LLM response: model=%s, usage=%s",
        getattr(response, "model", "unknown"),
        getattr(response, "usage", "N/A"),
    )

    return response
