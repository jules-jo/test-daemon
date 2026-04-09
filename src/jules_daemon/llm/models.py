"""Dataiku Mesh model routing data types.

Dataiku Mesh uses the format ``provider:connection:model`` for model
identifiers (e.g., ``openai:my-openai:gpt-4``). This module provides
an immutable ModelID dataclass and a parser for that format.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ToolCallingMode(Enum):
    """Strategy for how tool definitions are passed to the LLM.

    NATIVE: Tools are passed via the ``tools`` parameter of the
        Chat Completions API. Requires the backend to support
        OpenAI-style function calling.

    PROMPT_BASED: Tool definitions are injected into the system
        prompt as structured text. The LLM responds with a
        JSON block that the agent loop parses. Works with any
        backend that can follow instructions.
    """

    NATIVE = "native"
    PROMPT_BASED = "prompt_based"


@dataclass(frozen=True)
class ModelID:
    """Parsed Dataiku Mesh model identifier.

    Format: ``provider:connection:model``

    Examples:
        - ``openai:my-openai-conn:gpt-4``
        - ``anthropic:my-claude:claude-3-opus``
        - ``azure-openai:eastus-conn:gpt-4-turbo``
    """

    provider: str
    connection: str
    model: str

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider must not be empty")
        if not self.connection:
            raise ValueError("connection must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")

    def to_string(self) -> str:
        """Serialize back to the ``provider:connection:model`` format."""
        return f"{self.provider}:{self.connection}:{self.model}"


def parse_model_id(raw: str) -> ModelID:
    """Parse a Dataiku Mesh model ID string.

    Args:
        raw: String in ``provider:connection:model`` format.
            Extra colons after the third part are included in the model
            name (e.g., ``openai:conn:gpt-4:latest`` -> model=``gpt-4:latest``).

    Returns:
        Parsed ModelID.

    Raises:
        ValueError: If the string is empty or has fewer than 3 colon-separated parts.
    """
    stripped = raw.strip()
    if not stripped:
        raise ValueError("Model ID string must not be empty")

    parts = stripped.split(":", maxsplit=2)
    if len(parts) < 3:
        raise ValueError(
            f"Model ID must be in 'provider:connection:model' format, "
            f"got {stripped!r} ({len(parts)} parts)"
        )

    return ModelID(
        provider=parts[0],
        connection=parts[1],
        model=parts[2],
    )
