"""Tests for command context model and risk-level classification data types.

Covers:
    - RiskLevel enum values and string parsing
    - CommandContext frozen Pydantic model construction, validation, and defaults
    - CommandContext serialization to dict and JSON
    - parse_context_response: JSON extraction and schema validation
    - Edge cases: empty fields, invalid risk levels, missing required fields
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jules_daemon.llm.command_context import (
    CommandContext,
    RiskLevel,
    parse_context_response,
)
from jules_daemon.llm.errors import LLMParseError


# ---------------------------------------------------------------------------
# RiskLevel enum
# ---------------------------------------------------------------------------


class TestRiskLevel:
    """Tests for RiskLevel enum values and parsing."""

    def test_low_value(self) -> None:
        assert RiskLevel.LOW.value == "low"

    def test_medium_value(self) -> None:
        assert RiskLevel.MEDIUM.value == "medium"

    def test_high_value(self) -> None:
        assert RiskLevel.HIGH.value == "high"

    def test_critical_value(self) -> None:
        assert RiskLevel.CRITICAL.value == "critical"

    def test_from_string_low(self) -> None:
        assert RiskLevel("low") is RiskLevel.LOW

    def test_from_string_medium(self) -> None:
        assert RiskLevel("medium") is RiskLevel.MEDIUM

    def test_from_string_high(self) -> None:
        assert RiskLevel("high") is RiskLevel.HIGH

    def test_from_string_critical(self) -> None:
        assert RiskLevel("critical") is RiskLevel.CRITICAL

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            RiskLevel("extreme")

    def test_all_levels_are_four(self) -> None:
        assert len(RiskLevel) == 4

    def test_ordering_by_severity(self) -> None:
        """Verify severity ordering via the severity_order property."""
        assert RiskLevel.LOW.severity_order < RiskLevel.MEDIUM.severity_order
        assert RiskLevel.MEDIUM.severity_order < RiskLevel.HIGH.severity_order
        assert RiskLevel.HIGH.severity_order < RiskLevel.CRITICAL.severity_order


# ---------------------------------------------------------------------------
# CommandContext model: construction and validation
# ---------------------------------------------------------------------------


class TestCommandContextConstruction:
    """Tests for CommandContext creation and field validation."""

    def test_create_minimal(self) -> None:
        ctx = CommandContext(
            command="pytest -v",
            explanation="Run the pytest test suite with verbose output",
            affected_paths=("/opt/app/tests",),
            risk_level=RiskLevel.LOW,
        )
        assert ctx.command == "pytest -v"
        assert ctx.explanation == "Run the pytest test suite with verbose output"
        assert ctx.affected_paths == ("/opt/app/tests",)
        assert ctx.risk_level is RiskLevel.LOW
        assert ctx.risk_factors == ()
        assert ctx.safe_to_execute is True
        assert ctx.requires_approval is True

    def test_create_full(self) -> None:
        ctx = CommandContext(
            command="rm -rf /tmp/test-output && pytest -v",
            explanation="Remove old test output then run test suite",
            affected_paths=("/tmp/test-output", "/opt/app/tests"),
            risk_level=RiskLevel.HIGH,
            risk_factors=(
                "Uses rm -rf to delete directory",
                "Chained commands with &&",
            ),
            safe_to_execute=False,
        )
        assert ctx.risk_level is RiskLevel.HIGH
        assert len(ctx.risk_factors) == 2
        assert ctx.safe_to_execute is False
        assert ctx.requires_approval is True

    def test_frozen_immutability(self) -> None:
        ctx = CommandContext(
            command="ls -la",
            explanation="List files",
            affected_paths=("/home/user",),
            risk_level=RiskLevel.LOW,
        )
        with pytest.raises(ValidationError):
            ctx.command = "rm -rf /"  # type: ignore[misc]

    def test_empty_command_raises(self) -> None:
        with pytest.raises(ValidationError, match="command"):
            CommandContext(
                command="",
                explanation="Empty command",
                affected_paths=(),
                risk_level=RiskLevel.LOW,
            )

    def test_whitespace_only_command_raises(self) -> None:
        with pytest.raises(ValidationError, match="command"):
            CommandContext(
                command="   ",
                explanation="Blank command",
                affected_paths=(),
                risk_level=RiskLevel.LOW,
            )

    def test_empty_explanation_raises(self) -> None:
        with pytest.raises(ValidationError, match="explanation"):
            CommandContext(
                command="ls -la",
                explanation="",
                affected_paths=(),
                risk_level=RiskLevel.LOW,
            )

    def test_whitespace_only_explanation_raises(self) -> None:
        with pytest.raises(ValidationError, match="explanation"):
            CommandContext(
                command="ls -la",
                explanation="   ",
                affected_paths=(),
                risk_level=RiskLevel.LOW,
            )

    def test_command_gets_stripped(self) -> None:
        ctx = CommandContext(
            command="  pytest -v  ",
            explanation="Run tests",
            affected_paths=(),
            risk_level=RiskLevel.LOW,
        )
        assert ctx.command == "pytest -v"

    def test_explanation_gets_stripped(self) -> None:
        ctx = CommandContext(
            command="pytest -v",
            explanation="  Run tests  ",
            affected_paths=(),
            risk_level=RiskLevel.LOW,
        )
        assert ctx.explanation == "Run tests"

    def test_empty_affected_paths_allowed(self) -> None:
        ctx = CommandContext(
            command="echo hello",
            explanation="Print greeting",
            affected_paths=(),
            risk_level=RiskLevel.LOW,
        )
        assert ctx.affected_paths == ()

    def test_multiple_affected_paths(self) -> None:
        paths = ("/var/log", "/opt/app", "/tmp/output")
        ctx = CommandContext(
            command="test-runner",
            explanation="Run tests",
            affected_paths=paths,
            risk_level=RiskLevel.MEDIUM,
        )
        assert ctx.affected_paths == paths

    def test_requires_approval_always_true(self) -> None:
        """requires_approval is always True -- security invariant."""
        ctx = CommandContext(
            command="ls",
            explanation="List files",
            affected_paths=(),
            risk_level=RiskLevel.LOW,
        )
        assert ctx.requires_approval is True

    def test_requires_approval_false_raises(self) -> None:
        """Passing requires_approval=False must be rejected at validation."""
        with pytest.raises(ValidationError, match="requires_approval"):
            CommandContext(
                command="ls",
                explanation="List files",
                affected_paths=(),
                risk_level=RiskLevel.LOW,
                requires_approval=False,
            )

    def test_risk_level_from_string_in_model_validate(self) -> None:
        """model_validate should accept string risk levels."""
        data = {
            "command": "pytest -v",
            "explanation": "Run tests",
            "affected_paths": ["/opt/app"],
            "risk_level": "medium",
        }
        ctx = CommandContext.model_validate(data)
        assert ctx.risk_level is RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# CommandContext serialization
# ---------------------------------------------------------------------------


class TestCommandContextSerialization:
    """Tests for CommandContext serialization to dict and JSON."""

    def _make_context(self) -> CommandContext:
        return CommandContext(
            command="pytest -v --tb=short",
            explanation="Run the pytest suite with short tracebacks",
            affected_paths=("/opt/app/tests", "/opt/app/src"),
            risk_level=RiskLevel.LOW,
            risk_factors=("Read-only operation",),
            safe_to_execute=True,
        )

    def test_to_dict_contains_all_fields(self) -> None:
        ctx = self._make_context()
        d = ctx.to_dict()
        assert d["command"] == "pytest -v --tb=short"
        assert d["explanation"] == "Run the pytest suite with short tracebacks"
        assert d["affected_paths"] == ("/opt/app/tests", "/opt/app/src")
        assert d["risk_level"] == "low"
        assert d["risk_factors"] == ("Read-only operation",)
        assert d["safe_to_execute"] is True
        assert d["requires_approval"] is True

    def test_to_dict_round_trips(self) -> None:
        ctx = self._make_context()
        d = ctx.to_dict()
        restored = CommandContext.model_validate(d)
        assert restored == ctx

    def test_to_json_produces_valid_json(self) -> None:
        ctx = self._make_context()
        raw = ctx.to_json()
        parsed = json.loads(raw)
        assert parsed["command"] == "pytest -v --tb=short"
        assert parsed["risk_level"] == "low"

    def test_from_json_round_trips(self) -> None:
        ctx = self._make_context()
        raw = ctx.to_json()
        restored = CommandContext.model_validate_json(raw)
        assert restored == ctx


# ---------------------------------------------------------------------------
# parse_context_response: LLM text -> CommandContext
# ---------------------------------------------------------------------------


class TestParseContextResponse:
    """Tests for parsing raw LLM text into CommandContext."""

    def _make_valid_json(
        self,
        *,
        command: str = "pytest -v",
        explanation: str = "Run the test suite",
        affected_paths: list[str] | None = None,
        risk_level: str = "low",
        risk_factors: list[str] | None = None,
        safe_to_execute: bool = True,
    ) -> str:
        return json.dumps({
            "command": command,
            "explanation": explanation,
            "affected_paths": affected_paths or ["/opt/app/tests"],
            "risk_level": risk_level,
            "risk_factors": risk_factors or [],
            "safe_to_execute": safe_to_execute,
        })

    def test_parse_plain_json(self) -> None:
        text = self._make_valid_json()
        ctx = parse_context_response(text, command="pytest -v")
        assert isinstance(ctx, CommandContext)
        assert ctx.command == "pytest -v"
        assert ctx.risk_level is RiskLevel.LOW
        assert ctx.requires_approval is True

    def test_parse_code_fenced_json(self) -> None:
        inner = self._make_valid_json(risk_level="high")
        text = f"Here is the analysis:\n```json\n{inner}\n```\n"
        ctx = parse_context_response(text, command="pytest -v")
        assert ctx.risk_level is RiskLevel.HIGH

    def test_parse_uses_original_command(self) -> None:
        """The original command string overrides whatever the LLM returns."""
        text = self._make_valid_json(command="different command")
        ctx = parse_context_response(text, command="pytest -v")
        assert ctx.command == "pytest -v"

    def test_parse_critical_risk_level(self) -> None:
        text = self._make_valid_json(
            risk_level="critical",
            safe_to_execute=False,
            risk_factors=["Destructive operation"],
        )
        ctx = parse_context_response(text, command="pytest -v")
        assert ctx.risk_level is RiskLevel.CRITICAL
        assert ctx.safe_to_execute is False

    def test_parse_with_risk_factors(self) -> None:
        text = self._make_valid_json(
            risk_level="medium",
            risk_factors=["Writes to disk", "Modifies config"],
        )
        ctx = parse_context_response(text, command="pytest -v")
        assert len(ctx.risk_factors) == 2
        assert "Writes to disk" in ctx.risk_factors
        assert "Modifies config" in ctx.risk_factors

    def test_parse_empty_text_raises(self) -> None:
        with pytest.raises(LLMParseError):
            parse_context_response("", command="pytest -v")

    def test_parse_no_json_raises(self) -> None:
        with pytest.raises(LLMParseError):
            parse_context_response(
                "I cannot analyze this command.",
                command="pytest -v",
            )

    def test_parse_missing_explanation_raises(self) -> None:
        text = json.dumps({
            "command": "pytest -v",
            "affected_paths": [],
            "risk_level": "low",
        })
        with pytest.raises(LLMParseError, match="explanation"):
            parse_context_response(text, command="pytest -v")

    def test_parse_missing_risk_level_raises(self) -> None:
        text = json.dumps({
            "command": "pytest -v",
            "explanation": "Run tests",
            "affected_paths": [],
        })
        with pytest.raises(LLMParseError, match="risk_level"):
            parse_context_response(text, command="pytest -v")

    def test_parse_invalid_risk_level_raises(self) -> None:
        text = json.dumps({
            "command": "pytest -v",
            "explanation": "Run tests",
            "affected_paths": [],
            "risk_level": "extreme",
        })
        with pytest.raises(LLMParseError, match="risk_level"):
            parse_context_response(text, command="pytest -v")

    def test_parse_defaults_risk_factors_to_empty(self) -> None:
        text = json.dumps({
            "command": "pytest -v",
            "explanation": "Run tests",
            "affected_paths": ["/opt/app"],
            "risk_level": "low",
            "safe_to_execute": True,
        })
        ctx = parse_context_response(text, command="pytest -v")
        assert ctx.risk_factors == ()

    def test_parse_defaults_safe_to_execute_to_true(self) -> None:
        text = json.dumps({
            "command": "pytest -v",
            "explanation": "Run tests",
            "affected_paths": ["/opt/app"],
            "risk_level": "low",
        })
        ctx = parse_context_response(text, command="pytest -v")
        assert ctx.safe_to_execute is True

    def test_parse_extra_fields_ignored(self) -> None:
        text = json.dumps({
            "command": "pytest -v",
            "explanation": "Run tests",
            "affected_paths": [],
            "risk_level": "low",
            "extra_field": "should be ignored",
        })
        ctx = parse_context_response(text, command="pytest -v")
        assert ctx.explanation == "Run tests"

    def test_parse_affected_paths_converted_to_tuple(self) -> None:
        """JSON arrays should be converted to tuples for immutability."""
        text = self._make_valid_json(
            affected_paths=["/var/log", "/tmp/output"]
        )
        ctx = parse_context_response(text, command="pytest -v")
        assert isinstance(ctx.affected_paths, tuple)
        assert ctx.affected_paths == ("/var/log", "/tmp/output")
