"""Tests for the Tool protocol and InfoRetrievalTool base class.

Validates the Tool protocol contract, the InfoRetrievalTool base class
with its common validation, error handling, and result formatting, and
ensures the base class conforms to the Tool protocol.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from jules_daemon.agent.tool_base import InfoRetrievalTool, Tool
from jules_daemon.agent.tool_result import ToolResult, ToolResultStatus
from jules_daemon.agent.tool_types import ApprovalRequirement


# ---------------------------------------------------------------------------
# Concrete test implementation of InfoRetrievalTool
# ---------------------------------------------------------------------------


class FakeWikiTool(InfoRetrievalTool):
    """Minimal concrete implementation for testing the base class."""

    @property
    def name(self) -> str:
        return "fake_wiki"

    @property
    def description(self) -> str:
        return "Read a wiki page for testing"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "page_name": {
                    "type": "string",
                    "description": "Name of the wiki page to read",
                },
            },
            "required": ["page_name"],
        }

    async def _execute_impl(
        self, *, call_id: str, args: dict[str, Any]
    ) -> ToolResult:
        page_name = args["page_name"]
        return ToolResult.success(
            call_id=call_id,
            tool_name=self.name,
            output=f"Content of {page_name}",
        )


class FakeFailingTool(InfoRetrievalTool):
    """Tool whose _execute_impl raises an exception."""

    @property
    def name(self) -> str:
        return "fake_failing"

    @property
    def description(self) -> str:
        return "Always fails for testing error handling"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def _execute_impl(
        self, *, call_id: str, args: dict[str, Any]
    ) -> ToolResult:
        raise RuntimeError("Simulated failure in tool execution")


class FakeToolReturnsError(InfoRetrievalTool):
    """Tool whose _execute_impl returns an error ToolResult (no exception)."""

    @property
    def name(self) -> str:
        return "fake_returns_error"

    @property
    def description(self) -> str:
        return "Returns error result without raising"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def _execute_impl(
        self, *, call_id: str, args: dict[str, Any]
    ) -> ToolResult:
        return ToolResult.error(
            call_id=call_id,
            tool_name=self.name,
            error_message="Graceful error from tool",
        )


# ---------------------------------------------------------------------------
# Tool protocol conformance
# ---------------------------------------------------------------------------


class TestToolProtocol:
    """Ensure InfoRetrievalTool subclasses satisfy the Tool protocol."""

    def test_fake_wiki_tool_has_protocol_attributes(self) -> None:
        tool = FakeWikiTool()
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "parameters_schema")
        assert hasattr(tool, "execute")
        assert hasattr(tool, "requires_human_approval")

    def test_protocol_name_is_string(self) -> None:
        tool = FakeWikiTool()
        assert isinstance(tool.name, str)
        assert tool.name == "fake_wiki"

    def test_protocol_description_is_string(self) -> None:
        tool = FakeWikiTool()
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_protocol_parameters_schema_is_dict(self) -> None:
        tool = FakeWikiTool()
        schema = tool.parameters_schema
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"

    def test_requires_human_approval_default_false(self) -> None:
        """InfoRetrievalTool defaults to requires_human_approval=False."""
        tool = FakeWikiTool()
        assert tool.requires_human_approval is False

    def test_isinstance_check_with_tool_protocol(self) -> None:
        """InfoRetrievalTool instances are structurally compatible with Tool."""
        tool = FakeWikiTool()
        # Check structural compatibility (Tool is a Protocol)
        assert isinstance(tool.name, str)
        assert isinstance(tool.description, str)
        assert isinstance(tool.parameters_schema, dict)
        assert callable(tool.execute)


# ---------------------------------------------------------------------------
# InfoRetrievalTool.execute() -- the public entry point
# ---------------------------------------------------------------------------


class TestInfoRetrievalToolExecution:
    """Tests for the execute() wrapper method."""

    @pytest.mark.asyncio
    async def test_successful_execution(self) -> None:
        tool = FakeWikiTool()
        result = await tool.execute(call_id="c1", args={"page_name": "home"})
        assert result.is_success
        assert result.tool_name == "fake_wiki"
        assert result.call_id == "c1"
        assert "Content of home" in result.output

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self) -> None:
        tool = FakeWikiTool()
        result = await tool.execute(call_id="c2", args={"page_name": "test"})
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_exception_caught_returns_error_result(self) -> None:
        """Unhandled exceptions in _execute_impl are caught and wrapped."""
        tool = FakeFailingTool()
        result = await tool.execute(call_id="c3", args={})
        assert result.is_error
        assert result.tool_name == "fake_failing"
        assert result.call_id == "c3"
        assert "Simulated failure" in result.error_message

    @pytest.mark.asyncio
    async def test_error_result_passes_through(self) -> None:
        """An error ToolResult from _execute_impl is returned as-is."""
        tool = FakeToolReturnsError()
        result = await tool.execute(call_id="c4", args={})
        assert result.is_error
        assert result.error_message == "Graceful error from tool"


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestInfoRetrievalToolValidation:
    """Tests for argument validation in the base class."""

    @pytest.mark.asyncio
    async def test_missing_required_arg_returns_error(self) -> None:
        """Missing required parameter returns error without calling _execute_impl."""
        tool = FakeWikiTool()
        result = await tool.execute(call_id="c5", args={})
        assert result.is_error
        assert "page_name" in result.error_message

    @pytest.mark.asyncio
    async def test_none_args_returns_error(self) -> None:
        """None args are treated as empty dict and checked for required params."""
        tool = FakeWikiTool()
        result = await tool.execute(call_id="c6", args=None)  # type: ignore[arg-type]
        assert result.is_error

    @pytest.mark.asyncio
    async def test_extra_args_are_tolerated(self) -> None:
        """Extra arguments beyond the schema should not cause errors."""
        tool = FakeWikiTool()
        result = await tool.execute(
            call_id="c7", args={"page_name": "test", "extra": "ignored"}
        )
        assert result.is_success


# ---------------------------------------------------------------------------
# OpenAI-compatible schema serialization
# ---------------------------------------------------------------------------


class TestInfoRetrievalToolSchema:
    """Tests for to_openai_schema() serialization."""

    def test_to_openai_schema_structure(self) -> None:
        tool = FakeWikiTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert "function" in schema
        fn = schema["function"]
        assert fn["name"] == "fake_wiki"
        assert fn["description"] == "Read a wiki page for testing"
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"

    def test_to_openai_schema_includes_properties(self) -> None:
        tool = FakeWikiTool()
        schema = tool.to_openai_schema()
        params = schema["function"]["parameters"]
        assert "page_name" in params["properties"]

    def test_to_openai_schema_includes_required(self) -> None:
        tool = FakeWikiTool()
        schema = tool.to_openai_schema()
        params = schema["function"]["parameters"]
        assert "required" in params
        assert "page_name" in params["required"]

    def test_schema_is_json_serializable(self) -> None:
        """The OpenAI schema must be fully JSON-serializable."""
        tool = FakeWikiTool()
        schema = tool.to_openai_schema()
        serialized = json.dumps(schema)
        deserialized = json.loads(serialized)
        assert deserialized["function"]["name"] == "fake_wiki"


# ---------------------------------------------------------------------------
# to_tool_spec() conversion
# ---------------------------------------------------------------------------


class TestInfoRetrievalToolSpecConversion:
    """Tests for to_tool_spec() which produces a ToolSpec from the base class."""

    def test_to_tool_spec_name(self) -> None:
        tool = FakeWikiTool()
        spec = tool.to_tool_spec()
        assert spec.name == "fake_wiki"

    def test_to_tool_spec_description(self) -> None:
        tool = FakeWikiTool()
        spec = tool.to_tool_spec()
        assert spec.description == "Read a wiki page for testing"

    def test_to_tool_spec_approval_is_none_for_read_only(self) -> None:
        tool = FakeWikiTool()
        spec = tool.to_tool_spec()
        assert spec.approval is ApprovalRequirement.NONE

    def test_to_tool_spec_is_read_only(self) -> None:
        tool = FakeWikiTool()
        spec = tool.to_tool_spec()
        assert spec.is_read_only is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestInfoRetrievalToolEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_tool_with_no_required_params(self) -> None:
        """A tool with no required params should accept empty args."""

        class NoRequiredTool(InfoRetrievalTool):
            @property
            def name(self) -> str:
                return "no_required"

            @property
            def description(self) -> str:
                return "No required params"

            @property
            def parameters_schema(self) -> dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "optional_flag": {"type": "boolean"},
                    },
                }

            async def _execute_impl(
                self, *, call_id: str, args: dict[str, Any]
            ) -> ToolResult:
                return ToolResult.success(
                    call_id=call_id,
                    tool_name=self.name,
                    output="ran with no required params",
                )

        tool = NoRequiredTool()
        result = await tool.execute(call_id="c8", args={})
        assert result.is_success

    @pytest.mark.asyncio
    async def test_tool_with_empty_schema(self) -> None:
        """A tool that takes no parameters at all."""

        class EmptySchemaTool(InfoRetrievalTool):
            @property
            def name(self) -> str:
                return "empty_schema"

            @property
            def description(self) -> str:
                return "Takes nothing"

            @property
            def parameters_schema(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            async def _execute_impl(
                self, *, call_id: str, args: dict[str, Any]
            ) -> ToolResult:
                return ToolResult.success(
                    call_id=call_id,
                    tool_name=self.name,
                    output="done",
                )

        tool = EmptySchemaTool()
        result = await tool.execute(call_id="c9", args={})
        assert result.is_success
        assert result.output == "done"

    @pytest.mark.asyncio
    async def test_empty_call_id_returns_error(self) -> None:
        """An empty call_id should be caught by validation."""
        tool = FakeWikiTool()
        result = await tool.execute(call_id="", args={"page_name": "test"})
        assert result.is_error
        assert "call_id" in result.error_message
