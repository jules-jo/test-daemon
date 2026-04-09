"""Tests for LLM response parsing and validation.

Covers:
    - LLMCommandResponse Pydantic model construction and validation
    - JSON extraction from various LLM response formats (plain, code-fenced, mixed text)
    - Full response parsing pipeline: raw LLM text -> validated LLMCommandResponse
    - Error handling for malformed JSON, missing fields, and invalid values
    - Mapping from LLM response schema to SSHCommand objects
    - Prompt-based fallback parsing (extract JSON from free-form text)
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jules_daemon.llm.response_parser import (
    LLMCommandResponse,
    LLMCommandStep,
    Confidence,
    extract_json_from_text,
    parse_llm_response,
    response_to_ssh_commands,
)
from jules_daemon.llm.errors import LLMParseError
from jules_daemon.ssh.command import SSHCommand


# ---------------------------------------------------------------------------
# Confidence enum
# ---------------------------------------------------------------------------


class TestConfidence:
    """Tests for Confidence enum."""

    def test_high(self) -> None:
        assert Confidence.HIGH.value == "high"

    def test_medium(self) -> None:
        assert Confidence.MEDIUM.value == "medium"

    def test_low(self) -> None:
        assert Confidence.LOW.value == "low"

    def test_from_string(self) -> None:
        assert Confidence("high") is Confidence.HIGH
        assert Confidence("medium") is Confidence.MEDIUM
        assert Confidence("low") is Confidence.LOW

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            Confidence("very_high")


# ---------------------------------------------------------------------------
# LLMCommandStep model
# ---------------------------------------------------------------------------


class TestLLMCommandStep:
    """Tests for LLMCommandStep (single command in LLM response)."""

    def test_create_minimal(self) -> None:
        step = LLMCommandStep(
            command="pytest -v",
            description="Run tests",
        )
        assert step.command == "pytest -v"
        assert step.description == "Run tests"
        assert step.working_directory is None
        assert step.timeout_seconds == 300

    def test_create_full(self) -> None:
        step = LLMCommandStep(
            command="cd /opt/app && pytest -v --tb=short",
            description="Run the full test suite",
            working_directory="/opt/app",
            timeout_seconds=600,
        )
        assert step.command == "cd /opt/app && pytest -v --tb=short"
        assert step.working_directory == "/opt/app"
        assert step.timeout_seconds == 600

    def test_frozen(self) -> None:
        step = LLMCommandStep(command="echo hi", description="Greeting")
        with pytest.raises(ValidationError):
            step.command = "echo bye"  # type: ignore[misc]

    def test_empty_command_raises(self) -> None:
        with pytest.raises(ValidationError, match="command"):
            LLMCommandStep(command="", description="No command")

    def test_whitespace_only_command_raises(self) -> None:
        with pytest.raises(ValidationError, match="command"):
            LLMCommandStep(command="   ", description="Just spaces")

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValidationError, match="description"):
            LLMCommandStep(command="echo hi", description="")

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ValidationError, match="timeout_seconds"):
            LLMCommandStep(
                command="echo hi",
                description="test",
                timeout_seconds=-1,
            )

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValidationError, match="timeout_seconds"):
            LLMCommandStep(
                command="echo hi",
                description="test",
                timeout_seconds=0,
            )

    def test_stripped_values(self) -> None:
        step = LLMCommandStep(
            command="  pytest -v  ",
            description="  Run tests  ",
        )
        assert step.command == "pytest -v"
        assert step.description == "Run tests"


# ---------------------------------------------------------------------------
# LLMCommandResponse model
# ---------------------------------------------------------------------------


class TestLLMCommandResponse:
    """Tests for full LLM response model."""

    def _make_step(
        self,
        command: str = "pytest -v",
        description: str = "Run tests",
    ) -> dict:
        return {"command": command, "description": description}

    def test_create_with_commands(self) -> None:
        resp = LLMCommandResponse(
            commands=[self._make_step()],
            explanation="Running the test suite",
            confidence=Confidence.HIGH,
        )
        assert len(resp.commands) == 1
        assert resp.commands[0].command == "pytest -v"
        assert resp.explanation == "Running the test suite"
        assert resp.confidence is Confidence.HIGH
        assert resp.warnings == []

    def test_create_with_warnings(self) -> None:
        resp = LLMCommandResponse(
            commands=[self._make_step()],
            explanation="Running tests",
            confidence=Confidence.MEDIUM,
            warnings=["Might take a while"],
        )
        assert resp.warnings == ["Might take a while"]

    def test_create_empty_commands_refusal(self) -> None:
        resp = LLMCommandResponse(
            commands=[],
            explanation="Cannot fulfill this request",
            confidence=Confidence.LOW,
            warnings=["Request involves forbidden operations"],
        )
        assert len(resp.commands) == 0
        assert resp.is_refusal is True

    def test_is_refusal_false_when_commands_present(self) -> None:
        resp = LLMCommandResponse(
            commands=[self._make_step()],
            explanation="Running tests",
            confidence=Confidence.HIGH,
        )
        assert resp.is_refusal is False

    def test_frozen(self) -> None:
        resp = LLMCommandResponse(
            commands=[self._make_step()],
            explanation="test",
            confidence=Confidence.HIGH,
        )
        with pytest.raises(ValidationError):
            resp.explanation = "changed"  # type: ignore[misc]

    def test_empty_explanation_raises(self) -> None:
        with pytest.raises(ValidationError, match="explanation"):
            LLMCommandResponse(
                commands=[self._make_step()],
                explanation="",
                confidence=Confidence.HIGH,
            )

    def test_whitespace_explanation_raises(self) -> None:
        with pytest.raises(ValidationError, match="explanation"):
            LLMCommandResponse(
                commands=[self._make_step()],
                explanation="   ",
                confidence=Confidence.HIGH,
            )

    def test_multiple_commands(self) -> None:
        resp = LLMCommandResponse(
            commands=[
                self._make_step("cd /opt/app", "Navigate to app"),
                self._make_step("pytest -v", "Run tests"),
            ],
            explanation="Navigate and run",
            confidence=Confidence.HIGH,
        )
        assert len(resp.commands) == 2

    def test_from_dict(self) -> None:
        data = {
            "commands": [
                {
                    "command": "pytest -v",
                    "description": "Run tests",
                    "working_directory": "/opt/app",
                    "timeout_seconds": 300,
                }
            ],
            "explanation": "Running tests",
            "confidence": "high",
            "warnings": [],
        }
        resp = LLMCommandResponse.model_validate(data)
        assert resp.commands[0].command == "pytest -v"
        assert resp.confidence is Confidence.HIGH


# ---------------------------------------------------------------------------
# JSON extraction from LLM text
# ---------------------------------------------------------------------------


class TestExtractJsonFromText:
    """Tests for extract_json_from_text -- pulling JSON from LLM output."""

    def test_plain_json(self) -> None:
        text = '{"commands": [], "explanation": "no", "confidence": "low", "warnings": []}'
        result = extract_json_from_text(text)
        assert result["commands"] == []

    def test_json_in_markdown_code_fence(self) -> None:
        text = (
            "Here is the result:\n"
            "```json\n"
            '{"commands": [{"command": "ls", "description": "list"}], '
            '"explanation": "listing", "confidence": "high", "warnings": []}\n'
            "```\n"
            "Let me know if you need more."
        )
        result = extract_json_from_text(text)
        assert len(result["commands"]) == 1
        assert result["commands"][0]["command"] == "ls"

    def test_json_in_plain_code_fence(self) -> None:
        text = (
            "```\n"
            '{"commands": [], "explanation": "none", "confidence": "low", "warnings": []}\n'
            "```"
        )
        result = extract_json_from_text(text)
        assert result["explanation"] == "none"

    def test_json_with_surrounding_text(self) -> None:
        text = (
            "I analyzed your request. Here is the plan:\n\n"
            '{"commands": [{"command": "pytest", "description": "test"}], '
            '"explanation": "run tests", "confidence": "high", "warnings": []}\n\n'
            "The above command will run your tests."
        )
        result = extract_json_from_text(text)
        assert result["explanation"] == "run tests"

    def test_empty_text_raises(self) -> None:
        with pytest.raises(LLMParseError, match="No JSON"):
            extract_json_from_text("")

    def test_no_json_in_text_raises(self) -> None:
        with pytest.raises(LLMParseError, match="No JSON"):
            extract_json_from_text("This is plain text with no JSON at all.")

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(LLMParseError, match="Invalid JSON"):
            extract_json_from_text('{"broken": json{{{')

    def test_json_array_not_object_raises(self) -> None:
        with pytest.raises(LLMParseError, match="JSON object"):
            extract_json_from_text('[1, 2, 3]')

    def test_multiline_json(self) -> None:
        text = (
            "```json\n"
            "{\n"
            '  "commands": [\n'
            "    {\n"
            '      "command": "pytest -v --tb=short",\n'
            '      "description": "Run test suite"\n'
            "    }\n"
            "  ],\n"
            '  "explanation": "Running the test suite",\n'
            '  "confidence": "high",\n'
            '  "warnings": []\n'
            "}\n"
            "```"
        )
        result = extract_json_from_text(text)
        assert result["commands"][0]["command"] == "pytest -v --tb=short"

    def test_multiple_code_fences_uses_first_json(self) -> None:
        """When multiple code fences exist, extract from the first JSON one."""
        text = (
            "```bash\nls -la\n```\n"
            "```json\n"
            '{"commands": [], "explanation": "empty", "confidence": "low", "warnings": []}\n'
            "```"
        )
        result = extract_json_from_text(text)
        assert result["explanation"] == "empty"


# ---------------------------------------------------------------------------
# Full parse pipeline: raw text -> LLMCommandResponse
# ---------------------------------------------------------------------------


class TestParseLLMResponse:
    """Tests for parse_llm_response -- complete parsing pipeline."""

    def test_parse_valid_plain_json(self) -> None:
        text = json.dumps({
            "commands": [
                {
                    "command": "cd /opt/app && pytest -v",
                    "description": "Run the full test suite",
                    "working_directory": "/opt/app",
                    "timeout_seconds": 300,
                }
            ],
            "explanation": "Running the full pytest suite.",
            "confidence": "high",
            "warnings": [],
        })
        resp = parse_llm_response(text)
        assert isinstance(resp, LLMCommandResponse)
        assert len(resp.commands) == 1
        assert resp.commands[0].command == "cd /opt/app && pytest -v"
        assert resp.confidence is Confidence.HIGH

    def test_parse_code_fenced_json(self) -> None:
        text = (
            "Here is the plan:\n"
            "```json\n"
            + json.dumps({
                "commands": [
                    {"command": "make test", "description": "Run makefile tests"}
                ],
                "explanation": "Using make",
                "confidence": "medium",
                "warnings": ["Ensure make is installed"],
            })
            + "\n```"
        )
        resp = parse_llm_response(text)
        assert len(resp.commands) == 1
        assert resp.confidence is Confidence.MEDIUM
        assert resp.warnings == ["Ensure make is installed"]

    def test_parse_refusal(self) -> None:
        text = json.dumps({
            "commands": [],
            "explanation": "Cannot safely delete system files",
            "confidence": "low",
            "warnings": ["Request involves forbidden operations"],
        })
        resp = parse_llm_response(text)
        assert resp.is_refusal is True
        assert resp.explanation == "Cannot safely delete system files"

    def test_parse_missing_commands_field_raises(self) -> None:
        text = json.dumps({
            "explanation": "Missing commands",
            "confidence": "high",
            "warnings": [],
        })
        with pytest.raises(LLMParseError, match="commands"):
            parse_llm_response(text)

    def test_parse_missing_explanation_raises(self) -> None:
        text = json.dumps({
            "commands": [{"command": "ls", "description": "list"}],
            "confidence": "high",
            "warnings": [],
        })
        with pytest.raises(LLMParseError, match="explanation"):
            parse_llm_response(text)

    def test_parse_missing_confidence_raises(self) -> None:
        text = json.dumps({
            "commands": [{"command": "ls", "description": "list"}],
            "explanation": "listing",
            "warnings": [],
        })
        with pytest.raises(LLMParseError, match="confidence"):
            parse_llm_response(text)

    def test_parse_invalid_confidence_value_raises(self) -> None:
        text = json.dumps({
            "commands": [{"command": "ls", "description": "list"}],
            "explanation": "listing",
            "confidence": "super_high",
            "warnings": [],
        })
        with pytest.raises(LLMParseError, match="confidence"):
            parse_llm_response(text)

    def test_parse_no_json_raises(self) -> None:
        with pytest.raises(LLMParseError):
            parse_llm_response("I cannot help with that request.")

    def test_parse_empty_string_raises(self) -> None:
        with pytest.raises(LLMParseError):
            parse_llm_response("")

    def test_parse_extra_fields_ignored(self) -> None:
        """Extra fields from the LLM are silently ignored."""
        text = json.dumps({
            "commands": [{"command": "ls", "description": "list"}],
            "explanation": "listing",
            "confidence": "high",
            "warnings": [],
            "extra_field": "should be ignored",
        })
        resp = parse_llm_response(text)
        assert resp.explanation == "listing"

    def test_parse_defaults_warnings_to_empty(self) -> None:
        """If warnings is missing, default to empty list."""
        text = json.dumps({
            "commands": [{"command": "ls", "description": "list"}],
            "explanation": "listing",
            "confidence": "high",
        })
        resp = parse_llm_response(text)
        assert resp.warnings == []


# ---------------------------------------------------------------------------
# Mapping: LLMCommandResponse -> list[SSHCommand]
# ---------------------------------------------------------------------------


class TestResponseToSSHCommands:
    """Tests for converting LLMCommandResponse to SSHCommand objects."""

    def test_single_command_mapping(self) -> None:
        resp = LLMCommandResponse(
            commands=[
                LLMCommandStep(
                    command="pytest -v",
                    description="Run tests",
                    working_directory="/opt/app",
                    timeout_seconds=300,
                )
            ],
            explanation="Running tests",
            confidence=Confidence.HIGH,
        )
        ssh_cmds = response_to_ssh_commands(resp)
        assert len(ssh_cmds) == 1
        assert isinstance(ssh_cmds[0], SSHCommand)
        assert ssh_cmds[0].command == "pytest -v"
        assert ssh_cmds[0].working_directory == "/opt/app"
        assert ssh_cmds[0].timeout == 300

    def test_multiple_commands_mapping(self) -> None:
        resp = LLMCommandResponse(
            commands=[
                LLMCommandStep(
                    command="cd /opt/app",
                    description="Navigate",
                ),
                LLMCommandStep(
                    command="pytest -v",
                    description="Run tests",
                    timeout_seconds=600,
                ),
            ],
            explanation="Navigate and run",
            confidence=Confidence.HIGH,
        )
        ssh_cmds = response_to_ssh_commands(resp)
        assert len(ssh_cmds) == 2
        assert ssh_cmds[0].command == "cd /opt/app"
        assert ssh_cmds[1].command == "pytest -v"
        assert ssh_cmds[1].timeout == 600

    def test_refusal_returns_empty_list(self) -> None:
        resp = LLMCommandResponse(
            commands=[],
            explanation="Cannot do this",
            confidence=Confidence.LOW,
        )
        ssh_cmds = response_to_ssh_commands(resp)
        assert ssh_cmds == []

    def test_no_working_directory_maps_to_none(self) -> None:
        resp = LLMCommandResponse(
            commands=[
                LLMCommandStep(
                    command="pytest -v",
                    description="Run tests",
                )
            ],
            explanation="Running tests",
            confidence=Confidence.HIGH,
        )
        ssh_cmds = response_to_ssh_commands(resp)
        assert ssh_cmds[0].working_directory is None

    def test_timeout_mapping(self) -> None:
        resp = LLMCommandResponse(
            commands=[
                LLMCommandStep(
                    command="pytest -v",
                    description="Run tests",
                    timeout_seconds=120,
                )
            ],
            explanation="Running tests",
            confidence=Confidence.HIGH,
        )
        ssh_cmds = response_to_ssh_commands(resp)
        assert ssh_cmds[0].timeout == 120

    def test_command_with_invalid_working_dir_raises(self) -> None:
        """If the LLM returns a non-absolute path, SSHCommand validation catches it."""
        resp = LLMCommandResponse(
            commands=[
                LLMCommandStep(
                    command="pytest -v",
                    description="Run tests",
                    working_directory="relative/path",
                )
            ],
            explanation="Running tests",
            confidence=Confidence.HIGH,
        )
        with pytest.raises(LLMParseError, match="working_directory"):
            response_to_ssh_commands(resp)
