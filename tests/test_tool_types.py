"""Tests for agent tool protocol types: ToolCall, ToolResult, ToolParam, Tool.

Validates:
    - Frozen dataclass immutability
    - Validation rules (empty names, invalid statuses, etc.)
    - Serialization to/from dict
    - Tool protocol compliance
    - OpenAI-compatible function schema serialization
    - ApprovalRequirement classification (read-only vs state-changing)
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolCall,
    ToolParam,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# ToolParam tests
# ---------------------------------------------------------------------------


class TestToolParam:
    """Tests for the ToolParam frozen dataclass."""

    def test_create_minimal(self) -> None:
        param = ToolParam(
            name="host",
            description="Target hostname",
            json_type="string",
        )
        assert param.name == "host"
        assert param.description == "Target hostname"
        assert param.json_type == "string"
        assert param.required is True
        assert param.default is None
        assert param.enum is None

    def test_create_with_optional_fields(self) -> None:
        param = ToolParam(
            name="timeout",
            description="Timeout in seconds",
            json_type="integer",
            required=False,
            default=300,
            enum=None,
        )
        assert param.required is False
        assert param.default == 300

    def test_create_with_enum(self) -> None:
        param = ToolParam(
            name="level",
            description="Log level",
            json_type="string",
            enum=("debug", "info", "warning", "error"),
        )
        assert param.enum == ("debug", "info", "warning", "error")

    def test_create_array_with_items(self) -> None:
        param = ToolParam(
            name="summary_fields",
            description="Ordered summary fields",
            json_type="array",
            required=False,
            items={"type": "string"},
        )
        assert param.items == {"type": "string"}

    def test_frozen(self) -> None:
        param = ToolParam(name="x", description="d", json_type="string")
        with pytest.raises(FrozenInstanceError):
            param.name = "y"  # type: ignore[misc]

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            ToolParam(name="", description="d", json_type="string")

    def test_whitespace_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            ToolParam(name="   ", description="d", json_type="string")

    def test_empty_description_rejected(self) -> None:
        with pytest.raises(ValueError, match="description must not be empty"):
            ToolParam(name="x", description="", json_type="string")

    def test_invalid_json_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="json_type must be one of"):
            ToolParam(name="x", description="d", json_type="float")

    def test_array_without_items_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="array parameters must define an items schema"
        ):
            ToolParam(
                name="summary_fields",
                description="Ordered summary fields",
                json_type="array",
            )

    def test_non_array_with_items_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="items is only valid for array parameters"
        ):
            ToolParam(
                name="host",
                description="Target host",
                json_type="string",
                items={"type": "string"},
            )

    def test_to_json_schema_string(self) -> None:
        param = ToolParam(name="host", description="Target host", json_type="string")
        schema = param.to_json_schema()
        assert schema == {"type": "string", "description": "Target host"}

    def test_to_json_schema_with_enum(self) -> None:
        param = ToolParam(
            name="level",
            description="Log level",
            json_type="string",
            enum=("info", "debug"),
        )
        schema = param.to_json_schema()
        assert schema == {
            "type": "string",
            "description": "Log level",
            "enum": ["info", "debug"],
        }

    def test_to_json_schema_with_default(self) -> None:
        param = ToolParam(
            name="timeout",
            description="Timeout",
            json_type="integer",
            required=False,
            default=300,
        )
        schema = param.to_json_schema()
        assert schema["default"] == 300

    def test_to_json_schema_array_includes_items(self) -> None:
        param = ToolParam(
            name="summary_fields",
            description="Ordered summary fields",
            json_type="array",
            required=False,
            items={"type": "string"},
        )
        schema = param.to_json_schema()
        assert schema == {
            "type": "array",
            "description": "Ordered summary fields",
            "items": {"type": "string"},
        }


# ---------------------------------------------------------------------------
# ToolSpec tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Tests for the ToolSpec frozen dataclass."""

    def test_create_minimal(self) -> None:
        spec = ToolSpec(
            name="read_wiki",
            description="Read a wiki page by slug",
            parameters=(),
            approval=ApprovalRequirement.NONE,
        )
        assert spec.name == "read_wiki"
        assert spec.description == "Read a wiki page by slug"
        assert spec.parameters == ()
        assert spec.approval is ApprovalRequirement.NONE

    def test_create_with_parameters(self) -> None:
        params = (
            ToolParam(name="slug", description="Wiki page slug", json_type="string"),
        )
        spec = ToolSpec(
            name="read_wiki",
            description="Read a wiki page",
            parameters=params,
            approval=ApprovalRequirement.NONE,
        )
        assert len(spec.parameters) == 1
        assert spec.parameters[0].name == "slug"

    def test_frozen(self) -> None:
        spec = ToolSpec(
            name="read_wiki",
            description="Read a wiki page",
            parameters=(),
            approval=ApprovalRequirement.NONE,
        )
        with pytest.raises(FrozenInstanceError):
            spec.name = "other"  # type: ignore[misc]

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            ToolSpec(
                name="",
                description="Read a wiki page",
                parameters=(),
                approval=ApprovalRequirement.NONE,
            )

    def test_empty_description_rejected(self) -> None:
        with pytest.raises(ValueError, match="description must not be empty"):
            ToolSpec(
                name="read_wiki",
                description="",
                parameters=(),
                approval=ApprovalRequirement.NONE,
            )

    def test_duplicate_parameter_names_rejected(self) -> None:
        params = (
            ToolParam(name="slug", description="The slug", json_type="string"),
            ToolParam(name="slug", description="Duplicate slug", json_type="string"),
        )
        with pytest.raises(ValueError, match="Duplicate parameter names"):
            ToolSpec(
                name="read_wiki",
                description="Read wiki",
                parameters=params,
                approval=ApprovalRequirement.NONE,
            )

    def test_to_openai_schema_no_params(self) -> None:
        spec = ToolSpec(
            name="read_wiki",
            description="Read a wiki page",
            parameters=(),
            approval=ApprovalRequirement.NONE,
        )
        schema = spec.to_openai_function_schema()
        assert schema == {
            "type": "function",
            "function": {
                "name": "read_wiki",
                "description": "Read a wiki page",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    def test_to_openai_schema_with_params(self) -> None:
        params = (
            ToolParam(name="slug", description="Wiki slug", json_type="string"),
            ToolParam(
                name="section",
                description="Section name",
                json_type="string",
                required=False,
                default="overview",
            ),
        )
        spec = ToolSpec(
            name="read_wiki",
            description="Read a wiki page",
            parameters=params,
            approval=ApprovalRequirement.NONE,
        )
        schema = spec.to_openai_function_schema()
        fn = schema["function"]
        assert fn["name"] == "read_wiki"
        assert "slug" in fn["parameters"]["properties"]
        assert "section" in fn["parameters"]["properties"]
        assert "slug" in fn["parameters"]["required"]
        assert "section" not in fn["parameters"]["required"]

    def test_is_read_only_true(self) -> None:
        spec = ToolSpec(
            name="read_wiki",
            description="Read a wiki page",
            parameters=(),
            approval=ApprovalRequirement.NONE,
        )
        assert spec.is_read_only is True

    def test_is_read_only_false_for_confirm(self) -> None:
        spec = ToolSpec(
            name="propose_ssh_command",
            description="Propose an SSH command for approval",
            parameters=(),
            approval=ApprovalRequirement.CONFIRM_PROMPT,
        )
        assert spec.is_read_only is False

    def test_openai_schema_is_json_serializable(self) -> None:
        params = (
            ToolParam(name="slug", description="Wiki slug", json_type="string"),
        )
        spec = ToolSpec(
            name="read_wiki",
            description="Read a wiki page",
            parameters=params,
            approval=ApprovalRequirement.NONE,
        )
        schema = spec.to_openai_function_schema()
        # Must be JSON-serializable without errors
        serialized = json.dumps(schema)
        deserialized = json.loads(serialized)
        assert deserialized == schema


# ---------------------------------------------------------------------------
# ApprovalRequirement tests
# ---------------------------------------------------------------------------


class TestApprovalRequirement:
    """Tests for the ApprovalRequirement enum."""

    def test_values(self) -> None:
        assert ApprovalRequirement.NONE.value == "none"
        assert ApprovalRequirement.CONFIRM_PROMPT.value == "confirm_prompt"

    def test_all_members(self) -> None:
        members = set(ApprovalRequirement)
        assert members == {
            ApprovalRequirement.NONE,
            ApprovalRequirement.CONFIRM_PROMPT,
        }


# ---------------------------------------------------------------------------
# ToolResultStatus tests
# ---------------------------------------------------------------------------


class TestToolResultStatus:
    """Tests for the ToolResultStatus enum."""

    def test_values(self) -> None:
        assert ToolResultStatus.SUCCESS.value == "success"
        assert ToolResultStatus.ERROR.value == "error"
        assert ToolResultStatus.DENIED.value == "denied"
        assert ToolResultStatus.TIMEOUT.value == "timeout"

    def test_is_terminal_true_for_denied(self) -> None:
        assert ToolResultStatus.DENIED.is_terminal is True

    def test_is_terminal_false_for_success(self) -> None:
        assert ToolResultStatus.SUCCESS.is_terminal is False

    def test_is_terminal_false_for_error(self) -> None:
        assert ToolResultStatus.ERROR.is_terminal is False

    def test_is_terminal_false_for_timeout(self) -> None:
        assert ToolResultStatus.TIMEOUT.is_terminal is False


# ---------------------------------------------------------------------------
# ToolCall tests
# ---------------------------------------------------------------------------


class TestToolCall:
    """Tests for the ToolCall frozen dataclass."""

    def test_create_minimal(self) -> None:
        call = ToolCall(
            call_id="call_001",
            tool_name="read_wiki",
            arguments={},
        )
        assert call.call_id == "call_001"
        assert call.tool_name == "read_wiki"
        assert call.arguments == {}

    def test_create_with_arguments(self) -> None:
        args = {"slug": "smoke-tests", "section": "overview"}
        call = ToolCall(
            call_id="call_002",
            tool_name="read_wiki",
            arguments=args,
        )
        assert call.arguments == {"slug": "smoke-tests", "section": "overview"}

    def test_frozen(self) -> None:
        call = ToolCall(call_id="c1", tool_name="read_wiki", arguments={})
        with pytest.raises(FrozenInstanceError):
            call.call_id = "c2"  # type: ignore[misc]

    def test_empty_call_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="call_id must not be empty"):
            ToolCall(call_id="", tool_name="read_wiki", arguments={})

    def test_empty_tool_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="tool_name must not be empty"):
            ToolCall(call_id="c1", tool_name="", arguments={})

    def test_arguments_is_defensive_copy(self) -> None:
        """Verify that mutating the original dict does not affect the ToolCall."""
        original = {"slug": "test"}
        call = ToolCall(call_id="c1", tool_name="read_wiki", arguments=original)
        original["slug"] = "mutated"
        assert call.arguments["slug"] == "test"

    def test_to_dict(self) -> None:
        call = ToolCall(
            call_id="call_001",
            tool_name="read_wiki",
            arguments={"slug": "test"},
        )
        d = call.to_dict()
        assert d == {
            "call_id": "call_001",
            "tool_name": "read_wiki",
            "arguments": {"slug": "test"},
        }

    def test_from_dict(self) -> None:
        d = {
            "call_id": "call_001",
            "tool_name": "read_wiki",
            "arguments": {"slug": "test"},
        }
        call = ToolCall.from_dict(d)
        assert call.call_id == "call_001"
        assert call.tool_name == "read_wiki"
        assert call.arguments == {"slug": "test"}

    def test_from_dict_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            ToolCall.from_dict({"call_id": "c1", "tool_name": "read_wiki"})

    def test_to_openai_message_format(self) -> None:
        call = ToolCall(
            call_id="call_001",
            tool_name="read_wiki",
            arguments={"slug": "test"},
        )
        msg = call.to_openai_tool_call()
        assert msg == {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "read_wiki",
                "arguments": '{"slug": "test"}',
            },
        }


# ---------------------------------------------------------------------------
# ToolResult tests
# ---------------------------------------------------------------------------


class TestToolResult:
    """Tests for the ToolResult frozen dataclass."""

    def test_create_success(self) -> None:
        result = ToolResult(
            call_id="call_001",
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="# Smoke Tests\nRun with `pytest`.",
        )
        assert result.call_id == "call_001"
        assert result.tool_name == "read_wiki"
        assert result.status is ToolResultStatus.SUCCESS
        assert result.output == "# Smoke Tests\nRun with `pytest`."
        assert result.error_message is None

    def test_create_error(self) -> None:
        result = ToolResult(
            call_id="call_002",
            tool_name="execute_ssh",
            status=ToolResultStatus.ERROR,
            output="",
            error_message="Command returned exit code 1",
        )
        assert result.status is ToolResultStatus.ERROR
        assert result.error_message == "Command returned exit code 1"

    def test_create_denied(self) -> None:
        result = ToolResult(
            call_id="call_003",
            tool_name="propose_ssh_command",
            status=ToolResultStatus.DENIED,
            output="",
            error_message="User denied the proposed SSH command",
        )
        assert result.status is ToolResultStatus.DENIED
        assert result.is_terminal is True

    def test_frozen(self) -> None:
        result = ToolResult(
            call_id="c1",
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="ok",
        )
        with pytest.raises(FrozenInstanceError):
            result.output = "mutated"  # type: ignore[misc]

    def test_empty_call_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="call_id must not be empty"):
            ToolResult(
                call_id="",
                tool_name="read_wiki",
                status=ToolResultStatus.SUCCESS,
                output="ok",
            )

    def test_empty_tool_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="tool_name must not be empty"):
            ToolResult(
                call_id="c1",
                tool_name="",
                status=ToolResultStatus.SUCCESS,
                output="ok",
            )

    def test_is_terminal_delegates_to_status(self) -> None:
        denied = ToolResult(
            call_id="c1",
            tool_name="t",
            status=ToolResultStatus.DENIED,
            output="",
        )
        assert denied.is_terminal is True

        success = ToolResult(
            call_id="c2",
            tool_name="t",
            status=ToolResultStatus.SUCCESS,
            output="ok",
        )
        assert success.is_terminal is False

    def test_to_dict(self) -> None:
        result = ToolResult(
            call_id="call_001",
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="wiki content",
            error_message=None,
        )
        d = result.to_dict()
        assert d == {
            "call_id": "call_001",
            "tool_name": "read_wiki",
            "status": "success",
            "output": "wiki content",
            "error_message": None,
        }

    def test_from_dict(self) -> None:
        d = {
            "call_id": "call_001",
            "tool_name": "read_wiki",
            "status": "success",
            "output": "wiki content",
            "error_message": None,
        }
        result = ToolResult.from_dict(d)
        assert result.call_id == "call_001"
        assert result.status is ToolResultStatus.SUCCESS

    def test_to_openai_message_format(self) -> None:
        result = ToolResult(
            call_id="call_001",
            tool_name="read_wiki",
            status=ToolResultStatus.SUCCESS,
            output="wiki content",
        )
        msg = result.to_openai_tool_message()
        assert msg == {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": "wiki content",
        }

    def test_to_openai_message_format_error(self) -> None:
        result = ToolResult(
            call_id="call_002",
            tool_name="execute_ssh",
            status=ToolResultStatus.ERROR,
            output="",
            error_message="Command failed with exit code 1",
        )
        msg = result.to_openai_tool_message()
        assert msg == {
            "role": "tool",
            "tool_call_id": "call_002",
            "content": "ERROR: Command failed with exit code 1",
        }
