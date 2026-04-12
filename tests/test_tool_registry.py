"""Tests for the ToolRegistry -- stores Tool instances by name.

Validates:
    - register/get/list operations
    - duplicate registration rejection
    - unknown tool lookup returns None
    - to_openai_schemas() serialization (structure, required, enum, defaults)
    - validate_call() checks required params and unknown tool names
    - ToolValidationError attributes (tool_name, missing_params)
    - execute() delegates to the tool after validation
    - execute() routes to correct tool when multiple registered
    - execute() argument isolation (no cross-call leakage)
    - read-only vs approval-required tool classification
    - immutability (registry returns copies, not mutable internals)
    - exception hierarchy (ToolRegistryError / ToolValidationError)
    - fluent chaining across multiple registrations
    - schema generation for tools with enum params and defaults
    - get_spec() returns correct ToolSpec with matching contents
"""

from __future__ import annotations

from typing import Any

import pytest

from jules_daemon.agent.tool_registry import (
    ToolRegistry,
    ToolRegistryError,
    ToolValidationError,
)
from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolCall,
    ToolParam,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)
from jules_daemon.agent.tools.base import BaseTool


# ---------------------------------------------------------------------------
# Test fixtures: concrete tool implementations
# ---------------------------------------------------------------------------


class FakeReadOnlyTool(BaseTool):
    """A read-only tool that echoes its input for testing."""

    _spec = ToolSpec(
        name="fake_readonly",
        description="A read-only tool for testing",
        parameters=(
            ToolParam(
                name="query",
                description="Search query",
                json_type="string",
            ),
        ),
        approval=ApprovalRequirement.NONE,
    )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        call_id = args.get("_call_id", "test")
        query = args.get("query", "")
        return ToolResult.success(
            call_id=call_id,
            tool_name=self.name,
            output=f"echo: {query}",
        )


class FakeApprovalTool(BaseTool):
    """A tool requiring human approval for testing."""

    _spec = ToolSpec(
        name="fake_approval",
        description="A tool requiring approval",
        parameters=(
            ToolParam(
                name="command",
                description="Command to propose",
                json_type="string",
            ),
            ToolParam(
                name="reason",
                description="Optional reason",
                json_type="string",
                required=False,
                default="none",
            ),
        ),
        approval=ApprovalRequirement.CONFIRM_PROMPT,
    )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        call_id = args.get("_call_id", "test")
        command = args.get("command", "")
        return ToolResult.success(
            call_id=call_id,
            tool_name=self.name,
            output=f"proposed: {command}",
        )


class FakeNoParamsTool(BaseTool):
    """A tool with no parameters."""

    _spec = ToolSpec(
        name="fake_no_params",
        description="A tool with no parameters",
        parameters=(),
        approval=ApprovalRequirement.NONE,
    )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        call_id = args.get("_call_id", "test")
        return ToolResult.success(
            call_id=call_id,
            tool_name=self.name,
            output="done",
        )


class FakeFailingTool(BaseTool):
    """A tool whose execute always raises."""

    _spec = ToolSpec(
        name="fake_failing",
        description="A tool that always fails",
        parameters=(),
        approval=ApprovalRequirement.NONE,
    )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("simulated failure")


class FakeEnumTool(BaseTool):
    """A tool with enum and default parameters for schema testing."""

    _spec = ToolSpec(
        name="fake_enum",
        description="A tool with enum and default parameters",
        parameters=(
            ToolParam(
                name="level",
                description="Log level",
                json_type="string",
                required=True,
                enum=("debug", "info", "warning", "error"),
            ),
            ToolParam(
                name="format",
                description="Output format",
                json_type="string",
                required=False,
                default="json",
                enum=("json", "text", "csv"),
            ),
            ToolParam(
                name="limit",
                description="Max results",
                json_type="integer",
                required=False,
                default=10,
            ),
        ),
        approval=ApprovalRequirement.NONE,
    )

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        call_id = args.get("_call_id", "test")
        return ToolResult.success(
            call_id=call_id,
            tool_name=self.name,
            output=f"level={args.get('level')}, format={args.get('format')}",
        )


class FakeCapturingTool(BaseTool):
    """A tool that captures received args for dispatch routing verification."""

    def __init__(self, name: str, approval: ApprovalRequirement) -> None:
        self._spec = ToolSpec(
            name=name,
            description=f"Capturing tool {name}",
            parameters=(
                ToolParam(
                    name="input",
                    description="Any input",
                    json_type="string",
                    required=False,
                ),
            ),
            approval=approval,
        )
        self.captured_args: list[dict[str, Any]] = []

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        self.captured_args.append(dict(args))
        call_id = args.get("_call_id", "test")
        return ToolResult.success(
            call_id=call_id,
            tool_name=self.name,
            output=f"captured by {self.name}",
        )


# ---------------------------------------------------------------------------
# ToolRegistry construction
# ---------------------------------------------------------------------------


class TestToolRegistryConstruction:
    """Tests for creating a ToolRegistry."""

    def test_empty_registry(self) -> None:
        registry = ToolRegistry()
        assert len(registry) == 0

    def test_len_reflects_registered_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        assert len(registry) == 1

    def test_bool_empty(self) -> None:
        registry = ToolRegistry()
        assert not registry

    def test_bool_non_empty(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        assert registry


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


class TestToolRegistryRegister:
    """Tests for the register() method."""

    def test_register_single_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        assert "fake_readonly" in registry

    def test_register_multiple_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        registry.register(FakeApprovalTool())
        assert len(registry) == 2

    def test_register_duplicate_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        with pytest.raises(ToolRegistryError, match="already registered"):
            registry.register(FakeReadOnlyTool())

    def test_register_returns_self_for_chaining(self) -> None:
        registry = ToolRegistry()
        result = registry.register(FakeReadOnlyTool())
        assert result is registry


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestToolRegistryGet:
    """Tests for the get() method."""

    def test_get_existing_tool(self) -> None:
        registry = ToolRegistry()
        tool = FakeReadOnlyTool()
        registry.register(tool)
        assert registry.get("fake_readonly") is tool

    def test_get_missing_tool_returns_none(self) -> None:
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_contains_check(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        assert "fake_readonly" in registry
        assert "nonexistent" not in registry


# ---------------------------------------------------------------------------
# list_tools()
# ---------------------------------------------------------------------------


class TestToolRegistryListTools:
    """Tests for the list_tools() method."""

    def test_list_tools_empty(self) -> None:
        registry = ToolRegistry()
        assert registry.list_tools() == ()

    def test_list_tools_returns_tuple(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        result = registry.list_tools()
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_list_tools_returns_all_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        registry.register(FakeApprovalTool())
        tools = registry.list_tools()
        names = {t.name for t in tools}
        assert names == {"fake_readonly", "fake_approval"}

    def test_list_tools_returns_snapshot(self) -> None:
        """Modifying the returned tuple should not affect the registry."""
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        tools = registry.list_tools()
        assert len(tools) == 1
        # Adding another tool should not affect the previously returned tuple
        registry.register(FakeApprovalTool())
        assert len(tools) == 1  # original tuple unchanged


# ---------------------------------------------------------------------------
# list_tool_names()
# ---------------------------------------------------------------------------


class TestToolRegistryListToolNames:
    """Tests for the list_tool_names() method."""

    def test_list_tool_names_empty(self) -> None:
        registry = ToolRegistry()
        assert registry.list_tool_names() == ()

    def test_list_tool_names_sorted(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeApprovalTool())
        registry.register(FakeReadOnlyTool())
        names = registry.list_tool_names()
        assert names == ("fake_approval", "fake_readonly")


# ---------------------------------------------------------------------------
# to_openai_schemas()
# ---------------------------------------------------------------------------


class TestToolRegistryOpenAISchemas:
    """Tests for OpenAI-compatible schema serialization."""

    def test_schemas_empty_registry(self) -> None:
        registry = ToolRegistry()
        assert registry.to_openai_schemas() == ()

    def test_schemas_single_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        schemas = registry.to_openai_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "fake_readonly"

    def test_schemas_multiple_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        registry.register(FakeApprovalTool())
        schemas = registry.to_openai_schemas()
        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"fake_readonly", "fake_approval"}

    def test_schemas_returns_tuple(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        schemas = registry.to_openai_schemas()
        assert isinstance(schemas, tuple)

    def test_schemas_include_parameters(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        schemas = registry.to_openai_schemas()
        fn = schemas[0]["function"]
        assert "parameters" in fn
        assert "query" in fn["parameters"]["properties"]
        assert "query" in fn["parameters"]["required"]


# ---------------------------------------------------------------------------
# validate_call()
# ---------------------------------------------------------------------------


class TestToolRegistryValidateCall:
    """Tests for validate_call() schema validation."""

    def test_validate_valid_call(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        call = ToolCall(call_id="c1", tool_name="fake_readonly", arguments={"query": "test"})
        # Should not raise
        registry.validate_call(call)

    def test_validate_unknown_tool_raises(self) -> None:
        registry = ToolRegistry()
        call = ToolCall(call_id="c1", tool_name="nonexistent", arguments={})
        with pytest.raises(ToolValidationError, match="not registered"):
            registry.validate_call(call)

    def test_validate_missing_required_param_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        call = ToolCall(call_id="c1", tool_name="fake_readonly", arguments={})
        with pytest.raises(ToolValidationError, match="query"):
            registry.validate_call(call)

    def test_validate_extra_params_tolerated(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        call = ToolCall(
            call_id="c1",
            tool_name="fake_readonly",
            arguments={"query": "test", "extra": "ignored"},
        )
        # Extra params are tolerated (LLMs sometimes hallucinate extra fields)
        registry.validate_call(call)

    def test_validate_optional_param_omitted(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeApprovalTool())
        call = ToolCall(
            call_id="c1",
            tool_name="fake_approval",
            arguments={"command": "ls -la"},
        )
        # "reason" is optional, should not raise
        registry.validate_call(call)

    def test_validate_no_params_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeNoParamsTool())
        call = ToolCall(call_id="c1", tool_name="fake_no_params", arguments={})
        # Should not raise
        registry.validate_call(call)

    def test_validate_multiple_missing_required(self) -> None:
        """When multiple required params are missing, all should be reported."""

        class MultiParamTool(BaseTool):
            _spec = ToolSpec(
                name="multi",
                description="Multi param tool",
                parameters=(
                    ToolParam(name="a", description="A", json_type="string"),
                    ToolParam(name="b", description="B", json_type="integer"),
                ),
                approval=ApprovalRequirement.NONE,
            )

            async def execute(self, args: dict[str, Any]) -> ToolResult:
                return ToolResult.success(
                    call_id=args.get("_call_id", "x"),
                    tool_name="multi",
                    output="ok",
                )

        registry = ToolRegistry()
        registry.register(MultiParamTool())
        call = ToolCall(call_id="c1", tool_name="multi", arguments={})
        with pytest.raises(ToolValidationError, match="a") as exc_info:
            registry.validate_call(call)
        # Both should be mentioned
        assert "b" in str(exc_info.value)


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


class TestToolRegistryExecute:
    """Tests for execute() -- validate then delegate."""

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        call = ToolCall(call_id="c1", tool_name="fake_readonly", arguments={"query": "hello"})
        result = await registry.execute(call)
        assert result.is_success
        assert result.call_id == "c1"
        assert result.tool_name == "fake_readonly"
        assert "echo: hello" in result.output

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self) -> None:
        registry = ToolRegistry()
        call = ToolCall(call_id="c1", tool_name="missing", arguments={})
        result = await registry.execute(call)
        assert result.is_error
        assert "not registered" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_missing_required_returns_error(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        call = ToolCall(call_id="c1", tool_name="fake_readonly", arguments={})
        result = await registry.execute(call)
        assert result.is_error
        assert "query" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_tool_exception_returns_error(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeFailingTool())
        call = ToolCall(call_id="c1", tool_name="fake_failing", arguments={})
        result = await registry.execute(call)
        assert result.is_error
        assert "simulated failure" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_injects_call_id(self) -> None:
        """execute() should inject _call_id into args for the tool."""
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        call = ToolCall(call_id="injected_id", tool_name="fake_readonly", arguments={"query": "x"})
        result = await registry.execute(call)
        assert result.call_id == "injected_id"


# ---------------------------------------------------------------------------
# read-only vs approval-required classification
# ---------------------------------------------------------------------------


class TestToolRegistryClassification:
    """Tests for read-only vs approval-required tool classification."""

    def test_list_read_only_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        registry.register(FakeApprovalTool())
        read_only = registry.list_read_only_tools()
        assert len(read_only) == 1
        assert read_only[0].name == "fake_readonly"

    def test_list_approval_required_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        registry.register(FakeApprovalTool())
        approval = registry.list_approval_required_tools()
        assert len(approval) == 1
        assert approval[0].name == "fake_approval"

    def test_requires_approval_check(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        registry.register(FakeApprovalTool())
        assert registry.requires_approval("fake_approval") is True
        assert registry.requires_approval("fake_readonly") is False

    def test_requires_approval_unknown_tool_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolRegistryError, match="not registered"):
            registry.requires_approval("unknown")


# ---------------------------------------------------------------------------
# get_spec()
# ---------------------------------------------------------------------------


class TestToolRegistryGetSpec:
    """Tests for get_spec() -- get a tool's specification by name."""

    def test_get_spec_existing(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        spec = registry.get_spec("fake_readonly")
        assert spec is not None
        assert spec.name == "fake_readonly"

    def test_get_spec_missing_returns_none(self) -> None:
        registry = ToolRegistry()
        assert registry.get_spec("nonexistent") is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestToolRegistryEdgeCases:
    """Edge cases and boundary conditions."""

    def test_register_preserves_tool_identity(self) -> None:
        """The exact same tool object should be returned by get()."""
        registry = ToolRegistry()
        tool = FakeReadOnlyTool()
        registry.register(tool)
        assert registry.get("fake_readonly") is tool

    def test_iter_returns_tool_names(self) -> None:
        """Iterating over the registry yields tool names."""
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        registry.register(FakeApprovalTool())
        names = set(registry)
        assert names == {"fake_readonly", "fake_approval"}

    def test_registry_repr(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        r = repr(registry)
        assert "ToolRegistry" in r
        assert "1" in r


# ---------------------------------------------------------------------------
# ToolValidationError attributes
# ---------------------------------------------------------------------------


class TestToolValidationErrorAttributes:
    """Tests for ToolValidationError exception attributes."""

    def test_error_has_tool_name(self) -> None:
        registry = ToolRegistry()
        call = ToolCall(call_id="c1", tool_name="unknown_tool", arguments={})
        with pytest.raises(ToolValidationError) as exc_info:
            registry.validate_call(call)
        assert exc_info.value.tool_name == "unknown_tool"

    def test_error_has_empty_missing_params_for_unknown_tool(self) -> None:
        registry = ToolRegistry()
        call = ToolCall(call_id="c1", tool_name="unknown_tool", arguments={})
        with pytest.raises(ToolValidationError) as exc_info:
            registry.validate_call(call)
        assert exc_info.value.missing_params == ()

    def test_error_has_missing_params_list(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        call = ToolCall(call_id="c1", tool_name="fake_readonly", arguments={})
        with pytest.raises(ToolValidationError) as exc_info:
            registry.validate_call(call)
        assert "query" in exc_info.value.missing_params

    def test_error_has_multiple_missing_params(self) -> None:
        class MultiReqTool(BaseTool):
            _spec = ToolSpec(
                name="multi_req",
                description="Multi required params",
                parameters=(
                    ToolParam(name="x", description="X", json_type="string"),
                    ToolParam(name="y", description="Y", json_type="integer"),
                    ToolParam(name="z", description="Z", json_type="boolean"),
                ),
                approval=ApprovalRequirement.NONE,
            )

            async def execute(self, args: dict[str, Any]) -> ToolResult:
                return ToolResult.success(
                    call_id=args.get("_call_id", "x"),
                    tool_name="multi_req",
                    output="ok",
                )

        registry = ToolRegistry()
        registry.register(MultiReqTool())
        call = ToolCall(call_id="c1", tool_name="multi_req", arguments={})
        with pytest.raises(ToolValidationError) as exc_info:
            registry.validate_call(call)
        assert set(exc_info.value.missing_params) == {"x", "y", "z"}

    def test_validation_error_is_registry_error(self) -> None:
        """ToolValidationError should be a subclass of ToolRegistryError."""
        assert issubclass(ToolValidationError, ToolRegistryError)

    def test_validation_error_str_contains_message(self) -> None:
        err = ToolValidationError(
            "missing params",
            tool_name="test_tool",
            missing_params=("a", "b"),
        )
        assert "missing params" in str(err)
        assert err.tool_name == "test_tool"
        assert err.missing_params == ("a", "b")


# ---------------------------------------------------------------------------
# Deep OpenAI schema structure validation
# ---------------------------------------------------------------------------


class TestToolRegistrySchemaStructure:
    """Deep validation of OpenAI-compatible schema output."""

    def test_schema_top_level_type_field(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        schema = registry.to_openai_schemas()[0]
        assert schema["type"] == "function"

    def test_schema_function_name(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        fn = registry.to_openai_schemas()[0]["function"]
        assert fn["name"] == "fake_readonly"

    def test_schema_function_description(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        fn = registry.to_openai_schemas()[0]["function"]
        assert fn["description"] == "A read-only tool for testing"

    def test_schema_parameters_object_type(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        params = registry.to_openai_schemas()[0]["function"]["parameters"]
        assert params["type"] == "object"

    def test_schema_properties_contain_param_definitions(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        props = registry.to_openai_schemas()[0]["function"]["parameters"]["properties"]
        assert "query" in props
        assert props["query"]["type"] == "string"
        assert "description" in props["query"]

    def test_schema_required_array_matches_required_params(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        required = registry.to_openai_schemas()[0]["function"]["parameters"]["required"]
        assert required == ["query"]

    def test_schema_optional_param_not_in_required(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeApprovalTool())
        schema = registry.to_openai_schemas()[0]
        required = schema["function"]["parameters"]["required"]
        assert "command" in required
        assert "reason" not in required

    def test_schema_optional_param_has_default(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeApprovalTool())
        props = registry.to_openai_schemas()[0]["function"]["parameters"]["properties"]
        assert "reason" in props
        assert props["reason"].get("default") == "none"

    def test_schema_no_params_tool_has_empty_properties(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeNoParamsTool())
        params = registry.to_openai_schemas()[0]["function"]["parameters"]
        assert params["properties"] == {}
        assert params["required"] == []

    def test_schema_enum_param_serialized(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeEnumTool())
        props = registry.to_openai_schemas()[0]["function"]["parameters"]["properties"]
        assert props["level"]["enum"] == ["debug", "info", "warning", "error"]

    def test_schema_optional_enum_with_default(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeEnumTool())
        props = registry.to_openai_schemas()[0]["function"]["parameters"]["properties"]
        assert props["format"]["enum"] == ["json", "text", "csv"]
        assert props["format"]["default"] == "json"

    def test_schema_integer_default_value(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeEnumTool())
        props = registry.to_openai_schemas()[0]["function"]["parameters"]["properties"]
        assert props["limit"]["type"] == "integer"
        assert props["limit"]["default"] == 10

    def test_schemas_are_json_serializable(self) -> None:
        """All schemas should be JSON-serializable (no custom objects)."""
        import json

        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        registry.register(FakeApprovalTool())
        registry.register(FakeEnumTool())
        schemas = registry.to_openai_schemas()
        # Should not raise
        serialized = json.dumps(schemas)
        deserialized = json.loads(serialized)
        assert len(deserialized) == 3


# ---------------------------------------------------------------------------
# Tool dispatch routing
# ---------------------------------------------------------------------------


class TestToolRegistryDispatchRouting:
    """Tests that execute() routes to the correct tool among many."""

    @pytest.mark.asyncio
    async def test_routes_to_correct_tool_by_name(self) -> None:
        """When multiple tools are registered, the correct one is called."""
        tool_a = FakeCapturingTool("tool_a", ApprovalRequirement.NONE)
        tool_b = FakeCapturingTool("tool_b", ApprovalRequirement.NONE)
        tool_c = FakeCapturingTool("tool_c", ApprovalRequirement.CONFIRM_PROMPT)

        registry = ToolRegistry()
        registry.register(tool_a)
        registry.register(tool_b)
        registry.register(tool_c)

        call = ToolCall(
            call_id="c1",
            tool_name="tool_b",
            arguments={"input": "routed_to_b"},
        )
        result = await registry.execute(call)

        assert result.is_success
        assert result.output == "captured by tool_b"
        assert len(tool_a.captured_args) == 0
        assert len(tool_b.captured_args) == 1
        assert len(tool_c.captured_args) == 0
        assert tool_b.captured_args[0]["input"] == "routed_to_b"

    @pytest.mark.asyncio
    async def test_routes_to_approval_tool(self) -> None:
        """Approval-required tools are dispatched the same way."""
        tool_ro = FakeCapturingTool("reader", ApprovalRequirement.NONE)
        tool_ap = FakeCapturingTool("writer", ApprovalRequirement.CONFIRM_PROMPT)

        registry = ToolRegistry()
        registry.register(tool_ro)
        registry.register(tool_ap)

        call = ToolCall(
            call_id="c1",
            tool_name="writer",
            arguments={"input": "write_data"},
        )
        result = await registry.execute(call)

        assert result.is_success
        assert len(tool_ro.captured_args) == 0
        assert len(tool_ap.captured_args) == 1

    @pytest.mark.asyncio
    async def test_sequential_dispatches_to_different_tools(self) -> None:
        """Multiple sequential dispatches go to different tools."""
        tool_a = FakeCapturingTool("tool_a", ApprovalRequirement.NONE)
        tool_b = FakeCapturingTool("tool_b", ApprovalRequirement.NONE)

        registry = ToolRegistry()
        registry.register(tool_a)
        registry.register(tool_b)

        call_a = ToolCall(call_id="c1", tool_name="tool_a", arguments={})
        call_b = ToolCall(call_id="c2", tool_name="tool_b", arguments={})
        call_a2 = ToolCall(call_id="c3", tool_name="tool_a", arguments={})

        await registry.execute(call_a)
        await registry.execute(call_b)
        await registry.execute(call_a2)

        assert len(tool_a.captured_args) == 2
        assert len(tool_b.captured_args) == 1


# ---------------------------------------------------------------------------
# execute() argument isolation
# ---------------------------------------------------------------------------


class TestToolRegistryArgumentIsolation:
    """Tests that execute() does not leak state between calls."""

    @pytest.mark.asyncio
    async def test_call_id_injected_correctly(self) -> None:
        """execute() injects _call_id into args dict for the tool."""
        tool = FakeCapturingTool("capturer", ApprovalRequirement.NONE)
        registry = ToolRegistry()
        registry.register(tool)

        call = ToolCall(call_id="unique_id_42", tool_name="capturer", arguments={})
        await registry.execute(call)

        assert tool.captured_args[0]["_call_id"] == "unique_id_42"

    @pytest.mark.asyncio
    async def test_original_call_arguments_not_mutated(self) -> None:
        """The original ToolCall arguments should not be mutated by execute()."""
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())

        original_args = {"query": "test"}
        call = ToolCall(call_id="c1", tool_name="fake_readonly", arguments=original_args)

        await registry.execute(call)

        # ToolCall makes defensive copy, so _call_id should not appear
        # in the original dict
        assert "_call_id" not in original_args

    @pytest.mark.asyncio
    async def test_separate_calls_have_isolated_args(self) -> None:
        """Each execute() call gets its own args dict."""
        tool = FakeCapturingTool("capturer", ApprovalRequirement.NONE)
        registry = ToolRegistry()
        registry.register(tool)

        call1 = ToolCall(call_id="c1", tool_name="capturer", arguments={"input": "first"})
        call2 = ToolCall(call_id="c2", tool_name="capturer", arguments={"input": "second"})

        await registry.execute(call1)
        await registry.execute(call2)

        assert tool.captured_args[0]["input"] == "first"
        assert tool.captured_args[0]["_call_id"] == "c1"
        assert tool.captured_args[1]["input"] == "second"
        assert tool.captured_args[1]["_call_id"] == "c2"


# ---------------------------------------------------------------------------
# Fluent chaining
# ---------------------------------------------------------------------------


class TestToolRegistryFluentChaining:
    """Tests for fluent chaining on register()."""

    def test_chain_multiple_registrations(self) -> None:
        registry = (
            ToolRegistry()
            .register(FakeReadOnlyTool())
            .register(FakeApprovalTool())
            .register(FakeNoParamsTool())
        )
        assert len(registry) == 3
        assert "fake_readonly" in registry
        assert "fake_approval" in registry
        assert "fake_no_params" in registry

    def test_chain_returns_same_instance(self) -> None:
        registry = ToolRegistry()
        result = registry.register(FakeReadOnlyTool()).register(FakeApprovalTool())
        assert result is registry


# ---------------------------------------------------------------------------
# get_spec() content verification
# ---------------------------------------------------------------------------


class TestToolRegistryGetSpecContents:
    """Tests that get_spec() returns correct ToolSpec contents."""

    def test_spec_name_matches_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeApprovalTool())
        spec = registry.get_spec("fake_approval")
        assert spec is not None
        assert spec.name == "fake_approval"

    def test_spec_description_matches_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeApprovalTool())
        spec = registry.get_spec("fake_approval")
        assert spec is not None
        assert spec.description == "A tool requiring approval"

    def test_spec_parameters_match_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeApprovalTool())
        spec = registry.get_spec("fake_approval")
        assert spec is not None
        param_names = [p.name for p in spec.parameters]
        assert "command" in param_names
        assert "reason" in param_names

    def test_spec_approval_matches_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeApprovalTool())
        spec = registry.get_spec("fake_approval")
        assert spec is not None
        assert spec.approval is ApprovalRequirement.CONFIRM_PROMPT

    def test_spec_read_only_flag(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        spec = registry.get_spec("fake_readonly")
        assert spec is not None
        assert spec.is_read_only is True

    def test_spec_approval_flag(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeApprovalTool())
        spec = registry.get_spec("fake_approval")
        assert spec is not None
        assert spec.is_read_only is False


# ---------------------------------------------------------------------------
# Classification with mixed tool sets
# ---------------------------------------------------------------------------


class TestToolRegistryMixedClassification:
    """Tests for classification with many tools of different types."""

    def _build_mixed_registry(self) -> ToolRegistry:
        """Build a registry simulating the 10-tool production set."""
        registry = ToolRegistry()
        # 5 read-only tools
        for i in range(5):
            registry.register(
                FakeCapturingTool(f"readonly_{i}", ApprovalRequirement.NONE)
            )
        # 3 approval-required tools
        for i in range(3):
            registry.register(
                FakeCapturingTool(
                    f"approval_{i}", ApprovalRequirement.CONFIRM_PROMPT
                )
            )
        return registry

    def test_total_count(self) -> None:
        registry = self._build_mixed_registry()
        assert len(registry) == 8

    def test_read_only_count(self) -> None:
        registry = self._build_mixed_registry()
        assert len(registry.list_read_only_tools()) == 5

    def test_approval_required_count(self) -> None:
        registry = self._build_mixed_registry()
        assert len(registry.list_approval_required_tools()) == 3

    def test_read_only_plus_approval_equals_total(self) -> None:
        registry = self._build_mixed_registry()
        total = (
            len(registry.list_read_only_tools())
            + len(registry.list_approval_required_tools())
        )
        assert total == len(registry)

    def test_requires_approval_for_each_tool_type(self) -> None:
        registry = self._build_mixed_registry()
        for name in registry.list_tool_names():
            if name.startswith("readonly_"):
                assert registry.requires_approval(name) is False
            elif name.startswith("approval_"):
                assert registry.requires_approval(name) is True

    def test_schemas_count_matches_total_tools(self) -> None:
        registry = self._build_mixed_registry()
        schemas = registry.to_openai_schemas()
        assert len(schemas) == len(registry)


# ---------------------------------------------------------------------------
# execute() error result structure
# ---------------------------------------------------------------------------


class TestToolRegistryExecuteErrorResults:
    """Tests for error result structure from execute()."""

    @pytest.mark.asyncio
    async def test_error_result_from_unknown_tool_has_correct_fields(self) -> None:
        registry = ToolRegistry()
        call = ToolCall(call_id="c1", tool_name="ghost", arguments={})
        result = await registry.execute(call)
        assert result.status is ToolResultStatus.ERROR
        assert result.call_id == "c1"
        assert result.tool_name == "ghost"
        assert result.error_message is not None
        assert result.output == ""

    @pytest.mark.asyncio
    async def test_error_result_from_missing_params_has_correct_fields(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        call = ToolCall(call_id="c2", tool_name="fake_readonly", arguments={})
        result = await registry.execute(call)
        assert result.status is ToolResultStatus.ERROR
        assert result.call_id == "c2"
        assert result.tool_name == "fake_readonly"
        assert "query" in result.error_message

    @pytest.mark.asyncio
    async def test_error_result_from_exception_wraps_message(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeFailingTool())
        call = ToolCall(call_id="c3", tool_name="fake_failing", arguments={})
        result = await registry.execute(call)
        assert result.status is ToolResultStatus.ERROR
        assert "simulated failure" in result.error_message
        assert result.is_error is True
        assert result.is_terminal is False  # errors are observable, not terminal


# ---------------------------------------------------------------------------
# Immutability guarantees
# ---------------------------------------------------------------------------


class TestToolRegistryImmutability:
    """Tests that the registry returns defensive copies of its internals."""

    def test_list_tools_returns_new_tuple_each_call(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        first = registry.list_tools()
        second = registry.list_tools()
        assert first == second
        assert first is not second  # different tuple objects

    def test_list_tool_names_returns_new_tuple_each_call(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        first = registry.list_tool_names()
        second = registry.list_tool_names()
        assert first == second
        assert first is not second

    def test_to_openai_schemas_returns_new_tuple_each_call(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        first = registry.to_openai_schemas()
        second = registry.to_openai_schemas()
        assert first == second
        assert first is not second

    def test_list_read_only_returns_new_tuple(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeReadOnlyTool())
        first = registry.list_read_only_tools()
        second = registry.list_read_only_tools()
        assert first is not second

    def test_list_approval_required_returns_new_tuple(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeApprovalTool())
        first = registry.list_approval_required_tools()
        second = registry.list_approval_required_tools()
        assert first is not second
