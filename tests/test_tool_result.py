"""Tests for ToolResult factory methods and convenience properties.

The core ToolResult dataclass is tested in test_tool_types.py. This module
covers the additional factory classmethods (success, error, denied) and
boolean convenience properties (is_success, is_error, is_denied) that
simplify result construction and inspection in tool implementations.
"""

from __future__ import annotations

import json

import pytest

from jules_daemon.agent.tool_result import ToolResult, ToolResultStatus


# ---------------------------------------------------------------------------
# Factory classmethods
# ---------------------------------------------------------------------------


class TestToolResultSuccessFactory:
    """Tests for ToolResult.success() factory."""

    def test_basic_success(self) -> None:
        result = ToolResult.success(
            call_id="call_001",
            tool_name="read_wiki",
            output="wiki content here",
        )
        assert result.status is ToolResultStatus.SUCCESS
        assert result.call_id == "call_001"
        assert result.tool_name == "read_wiki"
        assert result.output == "wiki content here"
        assert result.error_message is None

    def test_success_is_frozen(self) -> None:
        result = ToolResult.success(
            call_id="c1",
            tool_name="read_wiki",
            output="ok",
        )
        with pytest.raises(AttributeError):
            result.output = "mutated"  # type: ignore[misc]


class TestToolResultErrorFactory:
    """Tests for ToolResult.error() factory."""

    def test_basic_error(self) -> None:
        result = ToolResult.error(
            call_id="call_002",
            tool_name="execute_ssh",
            error_message="Connection refused",
        )
        assert result.status is ToolResultStatus.ERROR
        assert result.error_message == "Connection refused"
        assert result.output == ""

    def test_error_with_output(self) -> None:
        result = ToolResult.error(
            call_id="call_003",
            tool_name="parse_test_output",
            error_message="Unexpected format",
            output="partial data before failure",
        )
        assert result.output == "partial data before failure"
        assert result.error_message == "Unexpected format"


class TestToolResultDeniedFactory:
    """Tests for ToolResult.denied() factory."""

    def test_basic_denied(self) -> None:
        result = ToolResult.denied(
            call_id="call_004",
            tool_name="propose_ssh_command",
            error_message="User denied: rm -rf /",
        )
        assert result.status is ToolResultStatus.DENIED
        assert result.error_message == "User denied: rm -rf /"
        assert result.output == ""

    def test_denied_is_terminal(self) -> None:
        result = ToolResult.denied(
            call_id="c1",
            tool_name="propose_ssh_command",
            error_message="No",
        )
        assert result.is_terminal is True


# ---------------------------------------------------------------------------
# Boolean convenience properties
# ---------------------------------------------------------------------------


class TestToolResultBooleanProperties:
    """Tests for is_success, is_error, is_denied properties."""

    def test_is_success_true(self) -> None:
        result = ToolResult.success(
            call_id="c1", tool_name="read_wiki", output="ok"
        )
        assert result.is_success is True
        assert result.is_error is False
        assert result.is_denied is False

    def test_is_error_true(self) -> None:
        result = ToolResult.error(
            call_id="c1", tool_name="t", error_message="fail"
        )
        assert result.is_success is False
        assert result.is_error is True
        assert result.is_denied is False

    def test_is_denied_true(self) -> None:
        result = ToolResult.denied(
            call_id="c1", tool_name="t", error_message="nope"
        )
        assert result.is_success is False
        assert result.is_error is False
        assert result.is_denied is True

    def test_timeout_is_not_success_or_denied(self) -> None:
        result = ToolResult(
            call_id="c1",
            tool_name="t",
            status=ToolResultStatus.TIMEOUT,
            output="",
            error_message="timed out",
        )
        assert result.is_success is False
        assert result.is_error is False
        assert result.is_denied is False


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


class TestToolResultToLlmMessage:
    """Tests for to_llm_message() formatting for LLM context."""

    def test_success_message_contains_output(self) -> None:
        result = ToolResult.success(
            call_id="c1",
            tool_name="read_wiki",
            output="wiki content",
        )
        msg = result.to_llm_message()
        assert "read_wiki" in msg
        assert "wiki content" in msg

    def test_error_message_contains_error(self) -> None:
        result = ToolResult.error(
            call_id="c1",
            tool_name="execute_ssh",
            error_message="connection refused",
        )
        msg = result.to_llm_message()
        assert "execute_ssh" in msg
        assert "connection refused" in msg

    def test_denied_message_indicates_denial(self) -> None:
        result = ToolResult.denied(
            call_id="c1",
            tool_name="propose_ssh_command",
            error_message="User denied the command",
        )
        msg = result.to_llm_message()
        assert "propose_ssh_command" in msg
        assert "denied" in msg.lower()

    def test_success_to_dict_roundtrip(self) -> None:
        result = ToolResult.success(
            call_id="c1",
            tool_name="parse_test_output",
            output="3 tests passed",
        )
        d = result.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["tool_name"] == "parse_test_output"
        assert deserialized["status"] == "success"
