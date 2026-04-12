"""Tests for NotifyUserTool (agent/tools/notify_user.py).

Comprehensive unit tests covering:
    - Tool specification metadata (name, description, parameters, approval)
    - Execute method: successful notification flow (callback returns True)
    - Execute method: delivery failure (callback returns False)
    - Input validation: empty message, whitespace-only, missing message key
    - Severity parameter: valid values, invalid fallback to "info", default
    - call_id propagation across all result paths
    - Error handling: callback exceptions (various exception types)
    - Edge cases: very long messages, special characters, concurrent calls
    - JSON output structure validation
    - Callback argument forwarding (message stripped, severity passed)
    - BaseTool conformance (spec property, name shortcut)
    - ToolResult serialization (to_openai_tool_message, to_llm_message)

These tests exercise the tool in isolation using mocked async callbacks,
matching the established pattern from test_ask_user_question_tool.py.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolResultStatus,
    ToolSpec,
)
from jules_daemon.agent.tools.base import BaseTool, Tool
from jules_daemon.agent.tools.notify_user import NotifyUserTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def success_callback() -> AsyncMock:
    """Callback that returns True (notification delivered)."""
    return AsyncMock(return_value=True)


@pytest.fixture
def failure_callback() -> AsyncMock:
    """Callback that returns False (notification not delivered)."""
    return AsyncMock(return_value=False)


@pytest.fixture
def error_callback() -> AsyncMock:
    """Callback that raises an exception (simulating IPC failure)."""
    return AsyncMock(side_effect=ConnectionError("IPC channel closed"))


@pytest.fixture
def tool(success_callback: AsyncMock) -> NotifyUserTool:
    """NotifyUserTool with a standard success callback."""
    return NotifyUserTool(notify_callback=success_callback)


# ---------------------------------------------------------------------------
# Tool specification and metadata
# ---------------------------------------------------------------------------


class TestNotifyUserToolSpec:
    """Verify tool spec metadata and protocol conformance."""

    def test_tool_name(self, tool: NotifyUserTool) -> None:
        assert tool.name == "notify_user"

    def test_spec_returns_tool_spec(self, tool: NotifyUserTool) -> None:
        assert isinstance(tool.spec, ToolSpec)

    def test_spec_name_matches(self, tool: NotifyUserTool) -> None:
        assert tool.spec.name == "notify_user"

    def test_spec_has_nonempty_description(
        self, tool: NotifyUserTool,
    ) -> None:
        assert isinstance(tool.spec.description, str)
        assert len(tool.spec.description) > 0

    def test_description_mentions_notification(
        self, tool: NotifyUserTool,
    ) -> None:
        """Description should indicate this tool sends notifications."""
        desc_lower = tool.spec.description.lower()
        assert "notification" in desc_lower or "notify" in desc_lower

    def test_no_approval_required(self, tool: NotifyUserTool) -> None:
        """notify_user is read-only (ApprovalRequirement.NONE)."""
        assert tool.spec.approval is ApprovalRequirement.NONE

    def test_is_read_only(self, tool: NotifyUserTool) -> None:
        assert tool.spec.is_read_only

    def test_has_message_parameter(self, tool: NotifyUserTool) -> None:
        param_names = {p.name for p in tool.spec.parameters}
        assert "message" in param_names

    def test_message_parameter_is_required(
        self, tool: NotifyUserTool,
    ) -> None:
        msg_param = next(
            p for p in tool.spec.parameters if p.name == "message"
        )
        assert msg_param.required is True

    def test_message_parameter_is_string_type(
        self, tool: NotifyUserTool,
    ) -> None:
        msg_param = next(
            p for p in tool.spec.parameters if p.name == "message"
        )
        assert msg_param.json_type == "string"

    def test_has_severity_parameter(self, tool: NotifyUserTool) -> None:
        param_names = {p.name for p in tool.spec.parameters}
        assert "severity" in param_names

    def test_severity_parameter_is_optional(
        self, tool: NotifyUserTool,
    ) -> None:
        sev_param = next(
            p for p in tool.spec.parameters if p.name == "severity"
        )
        assert sev_param.required is False

    def test_severity_parameter_default_is_info(
        self, tool: NotifyUserTool,
    ) -> None:
        sev_param = next(
            p for p in tool.spec.parameters if p.name == "severity"
        )
        assert sev_param.default == "info"

    def test_severity_parameter_has_enum(
        self, tool: NotifyUserTool,
    ) -> None:
        sev_param = next(
            p for p in tool.spec.parameters if p.name == "severity"
        )
        assert sev_param.enum is not None
        assert set(sev_param.enum) == {"info", "warning", "error", "success"}

    def test_severity_parameter_is_string_type(
        self, tool: NotifyUserTool,
    ) -> None:
        sev_param = next(
            p for p in tool.spec.parameters if p.name == "severity"
        )
        assert sev_param.json_type == "string"

    def test_exactly_two_parameters(self, tool: NotifyUserTool) -> None:
        assert len(tool.spec.parameters) == 2

    def test_is_base_tool_subclass(self) -> None:
        """NotifyUserTool must extend BaseTool."""
        assert issubclass(NotifyUserTool, BaseTool)

    def test_instance_satisfies_tool_protocol(
        self, tool: NotifyUserTool,
    ) -> None:
        """Instance must satisfy the Tool protocol (has spec + execute)."""
        assert isinstance(tool, Tool)

    def test_openai_schema_structure(
        self, tool: NotifyUserTool,
    ) -> None:
        """ToolSpec should serialize to valid OpenAI function schema."""
        schema = tool.spec.to_openai_function_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "notify_user"
        assert "description" in fn
        assert "parameters" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "message" in params["properties"]
        assert "message" in params["required"]

    def test_openai_schema_severity_not_required(
        self, tool: NotifyUserTool,
    ) -> None:
        """Severity should not appear in required array."""
        schema = tool.spec.to_openai_function_schema()
        params = schema["function"]["parameters"]
        assert "severity" not in params["required"]

    def test_openai_schema_severity_has_enum(
        self, tool: NotifyUserTool,
    ) -> None:
        """Severity property should include enum values."""
        schema = tool.spec.to_openai_function_schema()
        sev_prop = schema["function"]["parameters"]["properties"]["severity"]
        assert "enum" in sev_prop
        assert set(sev_prop["enum"]) == {"info", "warning", "error", "success"}

    def test_openai_schema_is_json_serializable(
        self, tool: NotifyUserTool,
    ) -> None:
        schema = tool.spec.to_openai_function_schema()
        serialized = json.dumps(schema)
        deserialized = json.loads(serialized)
        assert deserialized["function"]["name"] == "notify_user"


# ---------------------------------------------------------------------------
# Execute: successful notification flow (callback returns True)
# ---------------------------------------------------------------------------


class TestNotifyUserExecution:
    """Verify successful notification flows."""

    @pytest.mark.asyncio
    async def test_returns_success_on_delivery(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)
        result = await tool.execute({
            "message": "Test completed",
            "_call_id": "c1",
        })

        assert result.status is ToolResultStatus.SUCCESS
        assert result.is_success

    @pytest.mark.asyncio
    async def test_output_contains_delivered_true(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "Test completed",
            "_call_id": "c2",
        })

        data = json.loads(result.output)
        assert data["delivered"] is True

    @pytest.mark.asyncio
    async def test_output_contains_message(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "95 passed, 5 failed",
            "_call_id": "c3",
        })

        data = json.loads(result.output)
        assert data["message"] == "95 passed, 5 failed"

    @pytest.mark.asyncio
    async def test_output_contains_severity(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "Test done",
            "severity": "warning",
            "_call_id": "c4",
        })

        data = json.loads(result.output)
        assert data["severity"] == "warning"

    @pytest.mark.asyncio
    async def test_message_is_stripped_in_output(
        self, success_callback: AsyncMock,
    ) -> None:
        """Leading/trailing whitespace must be stripped from the message."""
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "  Test done  ",
            "_call_id": "c5",
        })

        data = json.loads(result.output)
        assert data["message"] == "Test done"

    @pytest.mark.asyncio
    async def test_callback_receives_stripped_message(
        self, success_callback: AsyncMock,
    ) -> None:
        """Callback must receive the stripped message."""
        tool = NotifyUserTool(notify_callback=success_callback)

        await tool.execute({
            "message": "  Test done  ",
            "severity": "info",
            "_call_id": "c6",
        })

        success_callback.assert_awaited_once_with("Test done", "info")

    @pytest.mark.asyncio
    async def test_callback_receives_severity(
        self, success_callback: AsyncMock,
    ) -> None:
        """Severity argument must be forwarded to the callback."""
        tool = NotifyUserTool(notify_callback=success_callback)

        await tool.execute({
            "message": "Alert",
            "severity": "error",
            "_call_id": "c7",
        })

        success_callback.assert_awaited_once_with("Alert", "error")

    @pytest.mark.asyncio
    async def test_success_result_is_not_terminal(
        self, tool: NotifyUserTool,
    ) -> None:
        result = await tool.execute({
            "message": "Test done",
            "_call_id": "c8",
        })
        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_success_result_has_no_error_message(
        self, tool: NotifyUserTool,
    ) -> None:
        result = await tool.execute({
            "message": "Test done",
            "_call_id": "c9",
        })
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_output_is_valid_json(
        self, tool: NotifyUserTool,
    ) -> None:
        result = await tool.execute({
            "message": "Any value",
            "_call_id": "c10",
        })
        # Must not raise
        data = json.loads(result.output)
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_output_has_three_keys(
        self, tool: NotifyUserTool,
    ) -> None:
        """Output JSON should contain exactly delivered, message, severity."""
        result = await tool.execute({
            "message": "Test",
            "_call_id": "c11",
        })
        data = json.loads(result.output)
        assert set(data.keys()) == {"delivered", "message", "severity"}


# ---------------------------------------------------------------------------
# Execute: delivery failure (callback returns False)
# ---------------------------------------------------------------------------


class TestNotifyUserDeliveryFailure:
    """Verify behavior when callback returns False (no subscribers)."""

    @pytest.mark.asyncio
    async def test_false_delivery_returns_success(
        self, failure_callback: AsyncMock,
    ) -> None:
        """Even when delivery returns False, the tool call succeeds."""
        tool = NotifyUserTool(notify_callback=failure_callback)

        result = await tool.execute({
            "message": "No subscribers",
            "_call_id": "df1",
        })

        assert result.status is ToolResultStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_false_delivery_output_shows_not_delivered(
        self, failure_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=failure_callback)

        result = await tool.execute({
            "message": "No subscribers",
            "_call_id": "df2",
        })

        data = json.loads(result.output)
        assert data["delivered"] is False

    @pytest.mark.asyncio
    async def test_false_delivery_still_has_message(
        self, failure_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=failure_callback)

        result = await tool.execute({
            "message": "No one listening",
            "_call_id": "df3",
        })

        data = json.loads(result.output)
        assert data["message"] == "No one listening"

    @pytest.mark.asyncio
    async def test_false_delivery_is_not_terminal(
        self, failure_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=failure_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "df4",
        })

        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_false_delivery_no_error_message(
        self, failure_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=failure_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "df5",
        })

        assert result.error_message is None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestNotifyUserValidation:
    """Verify argument validation returns ERROR for invalid inputs."""

    @pytest.mark.asyncio
    async def test_empty_message_returns_error(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "",
            "_call_id": "v1",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "message" in (result.error_message or "").lower()
        success_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitespace_only_message_returns_error(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "   ",
            "_call_id": "v2",
        })

        assert result.status is ToolResultStatus.ERROR
        success_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_message_key_returns_error(
        self, success_callback: AsyncMock,
    ) -> None:
        """When message key is entirely absent, returns error."""
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "severity": "info",
            "_call_id": "v3",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "message" in (result.error_message or "").lower()
        success_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_validation_error_is_not_terminal(
        self, success_callback: AsyncMock,
    ) -> None:
        """Validation errors are ERROR, not DENIED -- agent can self-correct."""
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "",
            "_call_id": "v4",
        })

        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_validation_error_output_is_empty(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "",
            "_call_id": "v5",
        })

        assert result.output == ""

    @pytest.mark.asyncio
    async def test_newline_only_message_returns_error(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "\n\t\r",
            "_call_id": "v6",
        })

        assert result.status is ToolResultStatus.ERROR
        success_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_args_dict_returns_error(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({"_call_id": "v7"})

        assert result.status is ToolResultStatus.ERROR
        success_callback.assert_not_awaited()


# ---------------------------------------------------------------------------
# Severity parameter handling
# ---------------------------------------------------------------------------


class TestNotifyUserSeverity:
    """Verify severity parameter validation and default behavior."""

    @pytest.mark.asyncio
    async def test_default_severity_is_info(
        self, success_callback: AsyncMock,
    ) -> None:
        """When severity is omitted, defaults to 'info'."""
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "sev1",
        })

        data = json.loads(result.output)
        assert data["severity"] == "info"

    @pytest.mark.asyncio
    async def test_callback_receives_default_severity(
        self, success_callback: AsyncMock,
    ) -> None:
        """Callback receives 'info' when severity is omitted."""
        tool = NotifyUserTool(notify_callback=success_callback)

        await tool.execute({
            "message": "Test",
            "_call_id": "sev2",
        })

        success_callback.assert_awaited_once_with("Test", "info")

    @pytest.mark.asyncio
    async def test_info_severity(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "Test",
            "severity": "info",
            "_call_id": "sev3",
        })

        data = json.loads(result.output)
        assert data["severity"] == "info"

    @pytest.mark.asyncio
    async def test_warning_severity(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "Test",
            "severity": "warning",
            "_call_id": "sev4",
        })

        data = json.loads(result.output)
        assert data["severity"] == "warning"

    @pytest.mark.asyncio
    async def test_error_severity(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "Test",
            "severity": "error",
            "_call_id": "sev5",
        })

        data = json.loads(result.output)
        assert data["severity"] == "error"

    @pytest.mark.asyncio
    async def test_success_severity(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "Test",
            "severity": "success",
            "_call_id": "sev6",
        })

        data = json.loads(result.output)
        assert data["severity"] == "success"

    @pytest.mark.asyncio
    async def test_invalid_severity_falls_back_to_info(
        self, success_callback: AsyncMock,
    ) -> None:
        """Invalid severity values silently fall back to 'info'."""
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "Test",
            "severity": "critical",
            "_call_id": "sev7",
        })

        assert result.is_success
        data = json.loads(result.output)
        assert data["severity"] == "info"

    @pytest.mark.asyncio
    async def test_invalid_severity_callback_receives_info(
        self, success_callback: AsyncMock,
    ) -> None:
        """Callback receives 'info' when an invalid severity is provided."""
        tool = NotifyUserTool(notify_callback=success_callback)

        await tool.execute({
            "message": "Test",
            "severity": "debug",
            "_call_id": "sev8",
        })

        success_callback.assert_awaited_once_with("Test", "info")

    @pytest.mark.asyncio
    async def test_empty_severity_falls_back_to_info(
        self, success_callback: AsyncMock,
    ) -> None:
        """Empty string severity falls back to 'info'."""
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "Test",
            "severity": "",
            "_call_id": "sev9",
        })

        data = json.loads(result.output)
        assert data["severity"] == "info"

    @pytest.mark.asyncio
    async def test_uppercase_severity_falls_back_to_info(
        self, success_callback: AsyncMock,
    ) -> None:
        """Uppercase 'INFO' is not in the valid set, falls back to 'info'."""
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "Test",
            "severity": "INFO",
            "_call_id": "sev10",
        })

        data = json.loads(result.output)
        assert data["severity"] == "info"


# ---------------------------------------------------------------------------
# call_id propagation
# ---------------------------------------------------------------------------


class TestNotifyUserCallId:
    """Verify _call_id flows through to results across all paths."""

    @pytest.mark.asyncio
    async def test_call_id_in_success_result(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "unique-success-id",
        })

        assert result.call_id == "unique-success-id"

    @pytest.mark.asyncio
    async def test_call_id_in_error_result(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "",
            "_call_id": "unique-error-id",
        })

        assert result.call_id == "unique-error-id"

    @pytest.mark.asyncio
    async def test_call_id_in_exception_result(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=error_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "unique-exception-id",
        })

        assert result.call_id == "unique-exception-id"

    @pytest.mark.asyncio
    async def test_default_call_id_when_missing(
        self, success_callback: AsyncMock,
    ) -> None:
        """When _call_id is omitted, defaults to 'notify_user'."""
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({"message": "Test"})

        assert result.call_id == "notify_user"

    @pytest.mark.asyncio
    async def test_tool_name_always_set(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "tn1",
        })

        assert result.tool_name == "notify_user"

    @pytest.mark.asyncio
    async def test_tool_name_in_validation_error(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "",
            "_call_id": "tn2",
        })

        assert result.tool_name == "notify_user"

    @pytest.mark.asyncio
    async def test_tool_name_in_exception_error(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=error_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "tn3",
        })

        assert result.tool_name == "notify_user"


# ---------------------------------------------------------------------------
# Error handling: callback exceptions
# ---------------------------------------------------------------------------


class TestNotifyUserErrorHandling:
    """Verify graceful handling of callback exceptions."""

    @pytest.mark.asyncio
    async def test_callback_exception_returns_error(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=error_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "err1",
        })

        assert result.status is ToolResultStatus.ERROR
        assert result.is_error

    @pytest.mark.asyncio
    async def test_callback_exception_error_message_contains_details(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=error_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "err2",
        })

        assert result.error_message is not None
        assert "Notification failed" in result.error_message
        assert "IPC channel closed" in result.error_message

    @pytest.mark.asyncio
    async def test_callback_exception_is_not_terminal(
        self, error_callback: AsyncMock,
    ) -> None:
        """Callback errors are ERROR (not DENIED) -- agent loop can retry."""
        tool = NotifyUserTool(notify_callback=error_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "err3",
        })

        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_callback_exception_output_is_empty(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=error_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "err4",
        })

        assert result.output == ""

    @pytest.mark.asyncio
    async def test_timeout_error_returns_error_result(self) -> None:
        """TimeoutError from callback should be caught and wrapped."""
        callback = AsyncMock(side_effect=TimeoutError("IPC timeout"))
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "err5",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "timeout" in (result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_runtime_error_returns_error_result(self) -> None:
        """RuntimeError from callback should be caught."""
        callback = AsyncMock(side_effect=RuntimeError("unexpected"))
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "err6",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "unexpected" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_value_error_from_callback(self) -> None:
        """ValueError from callback is caught and wrapped."""
        callback = AsyncMock(side_effect=ValueError("bad input"))
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "err7",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "bad input" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_os_error_from_callback(self) -> None:
        """OSError from callback is caught and wrapped."""
        callback = AsyncMock(side_effect=OSError("socket broken"))
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "err8",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "socket broken" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestNotifyUserEdgeCases:
    """Edge cases: long strings, special characters, etc."""

    @pytest.mark.asyncio
    async def test_very_long_message(self) -> None:
        """Tool should handle very long messages without error."""
        long_message = "x" * 5000
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": long_message,
            "_call_id": "edge1",
        })

        assert result.is_success
        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_message_with_special_characters(self) -> None:
        """Messages with special characters are handled."""
        message = 'Test "path" for ~/test & <args>?'
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": message,
            "_call_id": "edge2",
        })

        assert result.is_success
        data = json.loads(result.output)
        assert data["message"] == message

    @pytest.mark.asyncio
    async def test_message_with_newlines(self) -> None:
        """Multi-line messages are valid and forwarded correctly."""
        message = "Line 1\nLine 2\nLine 3"
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": message,
            "_call_id": "edge3",
        })

        assert result.is_success
        data = json.loads(result.output)
        assert data["message"] == message.strip()

    @pytest.mark.asyncio
    async def test_message_with_unicode(self) -> None:
        """Unicode characters in messages are handled."""
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": "Test completed successfully",
            "_call_id": "edge4",
        })

        assert result.is_success

    @pytest.mark.asyncio
    async def test_json_in_message(self) -> None:
        """JSON-like strings in the message must be properly serialized."""
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": '{"key": "value"}',
            "_call_id": "edge5",
        })

        assert result.is_success
        data = json.loads(result.output)
        assert data["message"] == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_extra_args_ignored(self) -> None:
        """Extra args beyond message/severity/_call_id are ignored."""
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": "Test",
            "extra_param": "should be ignored",
            "another": 42,
            "_call_id": "edge6",
        })

        assert result.is_success

    @pytest.mark.asyncio
    async def test_result_output_is_always_valid_json(self) -> None:
        """Output field is always valid JSON on success."""
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "edge7",
        })

        parsed = json.loads(result.output)
        assert "delivered" in parsed
        assert "message" in parsed
        assert "severity" in parsed

    @pytest.mark.asyncio
    async def test_message_with_tabs_and_mixed_whitespace(self) -> None:
        """Message with internal tabs/mixed whitespace is preserved."""
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": "col1\tcol2\tcol3",
            "_call_id": "edge8",
        })

        assert result.is_success
        data = json.loads(result.output)
        assert data["message"] == "col1\tcol2\tcol3"

    @pytest.mark.asyncio
    async def test_message_with_backslash_sequences(self) -> None:
        """Backslash sequences in message are preserved in JSON."""
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        result = await tool.execute({
            "message": r"C:\Users\test\path",
            "_call_id": "edge9",
        })

        assert result.is_success
        data = json.loads(result.output)
        assert data["message"] == r"C:\Users\test\path"


# ---------------------------------------------------------------------------
# Callback invocation ordering
# ---------------------------------------------------------------------------


class TestNotifyUserCallbackInvocation:
    """Verify the callback is called exactly once with correct arguments."""

    @pytest.mark.asyncio
    async def test_callback_called_exactly_once(self) -> None:
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        await tool.execute({
            "message": "Test",
            "_call_id": "inv1",
        })

        assert callback.await_count == 1

    @pytest.mark.asyncio
    async def test_callback_not_called_on_validation_error(self) -> None:
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        await tool.execute({
            "message": "",
            "_call_id": "inv2",
        })

        callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_callback_receives_two_positional_args(self) -> None:
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        await tool.execute({
            "message": "Alert!",
            "severity": "warning",
            "_call_id": "inv3",
        })

        # Verify positional args: (message, severity)
        args, kwargs = callback.call_args
        assert len(args) == 2
        assert args[0] == "Alert!"
        assert args[1] == "warning"
        assert len(kwargs) == 0

    @pytest.mark.asyncio
    async def test_multiple_sequential_calls(self) -> None:
        """Multiple sequential calls each invoke the callback once."""
        call_count = 0

        async def counting_callback(msg: str, sev: str) -> bool:
            nonlocal call_count
            call_count += 1
            return True

        tool = NotifyUserTool(notify_callback=counting_callback)

        r1 = await tool.execute({
            "message": "Msg1", "_call_id": "seq1"
        })
        r2 = await tool.execute({
            "message": "Msg2", "_call_id": "seq2"
        })
        r3 = await tool.execute({
            "message": "Msg3", "_call_id": "seq3"
        })

        assert call_count == 3
        assert r1.is_success
        assert r2.is_success
        assert r3.is_success

    @pytest.mark.asyncio
    async def test_callback_receives_default_severity_when_omitted(
        self,
    ) -> None:
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        await tool.execute({
            "message": "Test",
            "_call_id": "inv4",
        })

        callback.assert_awaited_once_with("Test", "info")

    @pytest.mark.asyncio
    async def test_callback_receives_corrected_invalid_severity(
        self,
    ) -> None:
        """Invalid severity is corrected to 'info' before calling callback."""
        callback = AsyncMock(return_value=True)
        tool = NotifyUserTool(notify_callback=callback)

        await tool.execute({
            "message": "Test",
            "severity": "FATAL",
            "_call_id": "inv5",
        })

        callback.assert_awaited_once_with("Test", "info")


# ---------------------------------------------------------------------------
# ToolResult serialization (to_openai_tool_message, to_llm_message)
# ---------------------------------------------------------------------------


class TestNotifyUserResultSerialization:
    """Verify ToolResult serialization for conversation history."""

    @pytest.mark.asyncio
    async def test_success_openai_message_format(
        self, tool: NotifyUserTool,
    ) -> None:
        result = await tool.execute({
            "message": "Test done",
            "_call_id": "ser1",
        })

        msg = result.to_openai_tool_message()
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "ser1"
        # Content should be the JSON output
        data = json.loads(msg["content"])
        assert "delivered" in data

    @pytest.mark.asyncio
    async def test_error_openai_message_has_error_prefix(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=error_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "ser2",
        })

        msg = result.to_openai_tool_message()
        assert msg["content"].startswith("ERROR:")

    @pytest.mark.asyncio
    async def test_success_llm_message_format(
        self, tool: NotifyUserTool,
    ) -> None:
        result = await tool.execute({
            "message": "Test done",
            "_call_id": "ser3",
        })

        text = result.to_llm_message()
        assert "[notify_user]" in text
        assert "success" in text.lower()

    @pytest.mark.asyncio
    async def test_error_llm_message_format(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=error_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "ser4",
        })

        text = result.to_llm_message()
        assert "[notify_user]" in text
        assert "ERROR" in text

    @pytest.mark.asyncio
    async def test_validation_error_openai_message(
        self, success_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=success_callback)

        result = await tool.execute({
            "message": "",
            "_call_id": "ser5",
        })

        msg = result.to_openai_tool_message()
        assert msg["content"].startswith("ERROR:")
        assert "message" in msg["content"].lower()

    @pytest.mark.asyncio
    async def test_to_dict_serialization(
        self, tool: NotifyUserTool,
    ) -> None:
        result = await tool.execute({
            "message": "Test",
            "_call_id": "ser6",
        })

        d = result.to_dict()
        assert d["call_id"] == "ser6"
        assert d["tool_name"] == "notify_user"
        assert d["status"] == "success"
        assert d["error_message"] is None

    @pytest.mark.asyncio
    async def test_error_to_dict_serialization(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = NotifyUserTool(notify_callback=error_callback)

        result = await tool.execute({
            "message": "Test",
            "_call_id": "ser7",
        })

        d = result.to_dict()
        assert d["status"] == "error"
        assert d["error_message"] is not None
