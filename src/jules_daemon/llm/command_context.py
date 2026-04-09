"""Command context model and risk-level classification data types.

Provides an immutable data model for the output of the command context
classifier: a structured analysis of what an SSH command does, which
paths it affects, and its risk level.

The risk classifier sends an SSH command to the LLM backend, which
returns a JSON analysis. This module defines the schema for that
analysis and the parser that converts raw LLM text into a validated
``CommandContext`` instance.

Security invariant: ``requires_approval`` is always ``True``. Every
SSH command must go through explicit human approval before execution,
regardless of risk level.

Usage::

    from jules_daemon.llm.command_context import (
        CommandContext,
        RiskLevel,
        parse_context_response,
    )

    # From a pre-parsed dict (e.g., unit tests)
    ctx = CommandContext(
        command="pytest -v",
        explanation="Run the test suite with verbose output",
        affected_paths=("/opt/app/tests",),
        risk_level=RiskLevel.LOW,
    )

    # From raw LLM text (production path)
    ctx = parse_context_response(
        text=llm_response_text,
        command="pytest -v",
    )
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
)

from jules_daemon.llm.errors import LLMParseError
from jules_daemon.llm.response_parser import extract_json_from_text

__all__ = [
    "CommandContext",
    "RiskLevel",
    "parse_context_response",
]


# ---------------------------------------------------------------------------
# Risk level enum
# ---------------------------------------------------------------------------

# Severity ordering for comparisons. Lower number = lower risk.
_SEVERITY_ORDER: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


class RiskLevel(Enum):
    """Risk classification for an SSH command.

    Levels (ascending severity):
        LOW: Read-only operations -- listing files, checking versions,
            reading logs. No data is modified.
        MEDIUM: Operations that run tests or builds. May write temporary
            output but do not modify persistent state.
        HIGH: Operations that modify files, write data, or change
            configuration. Reversible but consequential.
        CRITICAL: Destructive or system-level operations -- disk
            formatting, package removal, firewall changes. Potentially
            irreversible.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def severity_order(self) -> int:
        """Numeric severity for comparison (0=low, 3=critical)."""
        return _SEVERITY_ORDER[self.value]


# ---------------------------------------------------------------------------
# Command context model
# ---------------------------------------------------------------------------


class CommandContext(BaseModel):
    """Structured analysis of an SSH command produced by the LLM classifier.

    Immutable (frozen) Pydantic model. All field mutations produce a new
    instance via ``model_copy(update=...)``.

    Attributes:
        command: The original SSH command string being analyzed.
        explanation: Human-readable description of what the command does.
        affected_paths: Filesystem paths the command reads or writes.
        risk_level: Classified risk level.
        risk_factors: Reasons for the assigned risk level.
        safe_to_execute: LLM's recommendation on whether execution is safe.
            This is advisory only -- human approval is always required.
        requires_approval: Always True. Security invariant enforcing that
            every command goes through explicit human approval.
    """

    model_config = ConfigDict(frozen=True)

    command: str
    explanation: str
    affected_paths: tuple[str, ...] = ()
    risk_level: RiskLevel
    risk_factors: tuple[str, ...] = ()
    safe_to_execute: bool = True
    requires_approval: bool = True

    # -- Validators --

    @field_validator("command", mode="before")
    @classmethod
    def _strip_and_validate_command(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("command must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("command must not be empty or whitespace-only")
        return stripped

    @field_validator("explanation", mode="before")
    @classmethod
    def _strip_and_validate_explanation(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("explanation must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "explanation must not be empty or whitespace-only"
            )
        return stripped

    @field_validator("affected_paths", mode="before")
    @classmethod
    def _coerce_affected_paths(cls, value: Any) -> tuple[str, ...]:
        """Accept lists (from JSON) and convert to tuple for immutability."""
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        raise ValueError("affected_paths must be a list or tuple of strings")

    @field_validator("risk_factors", mode="before")
    @classmethod
    def _coerce_risk_factors(cls, value: Any) -> tuple[str, ...]:
        """Accept lists (from JSON) and convert to tuple for immutability."""
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        raise ValueError("risk_factors must be a list or tuple of strings")

    @field_validator("requires_approval", mode="after")
    @classmethod
    def _enforce_approval_invariant(cls, value: bool) -> bool:
        """Security invariant: requires_approval is always True."""
        if not value:
            raise ValueError(
                "requires_approval must be True -- every SSH command "
                "requires explicit human approval"
            )
        return True

    # -- Serialization helpers --

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for wiki YAML persistence.

        The risk_level is serialized as its string value for readability.
        """
        data = self.model_dump()
        data["risk_level"] = self.risk_level.value
        return data

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return self.model_dump_json()


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def parse_context_response(
    text: str,
    *,
    command: str,
) -> CommandContext:
    """Parse raw LLM text into a validated CommandContext.

    Pipeline:
    1. Extract JSON from the text (handles code fences, mixed text)
    2. Override the 'command' field with the original command string
       (never trust the LLM to echo it back correctly)
    3. Validate against the CommandContext schema

    Args:
        text: Raw LLM response text containing a JSON analysis.
        command: The original SSH command string. This overrides whatever
            the LLM returns in the 'command' field, ensuring the context
            always references the actual command.

    Returns:
        Validated, immutable CommandContext.

    Raises:
        LLMParseError: If JSON extraction or schema validation fails.
    """
    data = extract_json_from_text(text)

    # Override command with the authoritative original
    data["command"] = command

    # Ensure requires_approval is always True (defense in depth)
    data["requires_approval"] = True

    try:
        return CommandContext.model_validate(data)
    except ValidationError as exc:
        error_str = str(exc)
        for field_name in ("explanation", "risk_level", "affected_paths"):
            if field_name in error_str.lower():
                raise LLMParseError(
                    f"Invalid or missing '{field_name}' in LLM context response: {exc}",
                    raw_content=text,
                ) from exc
        raise LLMParseError(
            f"Failed to validate LLM context response: {exc}",
            raw_content=text,
        ) from exc
