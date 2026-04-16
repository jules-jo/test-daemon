"""Tests for LookupTestSpecTool -- verifies InfoRetrievalTool integration.

Covers:
    - InfoRetrievalTool base class integration (validation, error wrapping)
    - Delegation to wiki.test_knowledge (derive_test_slug, load_test_knowledge)
    - required_args surfacing for agent loop missing-arg detection
    - Dual calling convention (InfoRetrievalTool + legacy BaseTool)
    - OpenAI schema serialization
    - ToolRegistry compatibility
    - Edge cases (empty args, whitespace, not found, wiki I/O errors)
    - Fallback kwargs calling convention
    - Thread pool delegation (asyncio.to_thread)
    - Multiple test specs disambiguation
    - JSON output structure conformance
    - Concurrent execution safety
    - Tool result LLM message formatting
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolResultStatus,
)
from jules_daemon.agent.tools.lookup_test_spec import LookupTestSpecTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Create a temporary wiki root with the required directory structure."""
    from jules_daemon.wiki.layout import initialize_wiki

    initialize_wiki(tmp_path)
    return tmp_path


@pytest.fixture
def tool(wiki_root: Path) -> LookupTestSpecTool:
    """Create a LookupTestSpecTool with a temp wiki root."""
    return LookupTestSpecTool(wiki_root=wiki_root)


def _save_knowledge(
    wiki_root: Path,
    *,
    slug: str = "agent-test-py",
    command: str = "python3 ~/agent_test.py",
    purpose: str = "Tests the agent loop",
    output_format: str = "iteration counts",
    test_file_path: str = "",
    summary_fields: tuple[str, ...] = ("passed", "failed", "iterations_done"),
    common_failures: tuple[str, ...] = ("timeout on large inputs",),
    normal_behavior: str = "All iterations pass",
    required_args: tuple[str, ...] = (),
    workflow_steps: tuple[str, ...] = (),
    prerequisites: tuple[str, ...] = (),
    artifact_requirements: tuple[str, ...] = (),
    when_missing_artifact_ask: str = "",
    success_criteria: str = "",
    failure_criteria: str = "",
    runs_observed: int = 10,
) -> None:
    """Helper to persist a test knowledge entry to the wiki."""
    from jules_daemon.wiki.test_knowledge import (
        TestKnowledge,
        save_test_knowledge,
    )

    knowledge = TestKnowledge(
        test_slug=slug,
        command_pattern=command,
        purpose=purpose,
        output_format=output_format,
        test_file_path=test_file_path,
        summary_fields=summary_fields,
        common_failures=common_failures,
        normal_behavior=normal_behavior,
        required_args=required_args,
        workflow_steps=workflow_steps,
        prerequisites=prerequisites,
        artifact_requirements=artifact_requirements,
        when_missing_artifact_ask=when_missing_artifact_ask,
        success_criteria=success_criteria,
        failure_criteria=failure_criteria,
        runs_observed=runs_observed,
    )
    save_test_knowledge(wiki_root, knowledge)


# ---------------------------------------------------------------------------
# InfoRetrievalTool protocol compliance
# ---------------------------------------------------------------------------


class TestInfoRetrievalToolCompliance:
    """Verify LookupTestSpecTool satisfies the InfoRetrievalTool contract."""

    def test_has_name_property(self, tool: LookupTestSpecTool) -> None:
        """Tool must expose a name property."""
        assert tool.name == "lookup_test_spec"

    def test_has_description_property(self, tool: LookupTestSpecTool) -> None:
        """Tool must expose a non-empty description."""
        assert tool.description
        assert "test specification" in tool.description.lower()

    def test_has_parameters_schema(self, tool: LookupTestSpecTool) -> None:
        """Tool must expose a valid JSON Schema for parameters."""
        schema = tool.parameters_schema
        assert schema["type"] == "object"
        assert "test_name" in schema["properties"]
        assert "test_name" in schema["required"]

    def test_requires_no_human_approval(self, tool: LookupTestSpecTool) -> None:
        """Read-only tool must not require human approval."""
        assert tool.requires_human_approval is False

    def test_openai_schema_format(self, tool: LookupTestSpecTool) -> None:
        """to_openai_schema must produce the OpenAI function-calling format."""
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "lookup_test_spec"
        assert func["description"]
        assert func["parameters"]["type"] == "object"
        assert "test_name" in func["parameters"]["properties"]

    def test_to_tool_spec_conversion(self, tool: LookupTestSpecTool) -> None:
        """to_tool_spec must produce a valid ToolSpec for the ToolRegistry."""
        spec = tool.to_tool_spec()
        assert spec.name == "lookup_test_spec"
        assert spec.approval is ApprovalRequirement.NONE
        param_names = [p.name for p in spec.parameters]
        assert "test_name" in param_names

    def test_spec_property_is_cached(self, tool: LookupTestSpecTool) -> None:
        """The spec property must return the same cached ToolSpec instance."""
        spec1 = tool.spec
        spec2 = tool.spec
        assert spec1 is spec2

    def test_spec_matches_to_tool_spec(self, tool: LookupTestSpecTool) -> None:
        """The spec property must match what to_tool_spec produces."""
        cached = tool.spec
        fresh = tool.to_tool_spec()
        assert cached.name == fresh.name
        assert cached.description == fresh.description
        assert cached.approval == fresh.approval


# ---------------------------------------------------------------------------
# Delegation to wiki.test_knowledge
# ---------------------------------------------------------------------------


class TestWikiDelegation:
    """Verify lookup_test_spec delegates to wiki.test_knowledge."""

    @pytest.mark.asyncio
    async def test_found_returns_full_spec(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """When a test spec exists, it must return all fields."""
        _save_knowledge(
            wiki_root,
            required_args=("iterations", "host"),
            workflow_steps=("setup-step", "main_check"),
            prerequisites=("setup-step",),
            artifact_requirements=("setup_ready_file",),
            when_missing_artifact_ask="Run setup first?",
            success_criteria="Main check passes.",
            failure_criteria="Main check fails.",
        )

        result = await tool.execute(
            call_id="c1", args={"test_name": "python3 ~/agent_test.py"}
        )

        assert result.status is ToolResultStatus.SUCCESS
        assert result.call_id == "c1"
        assert result.tool_name == "lookup_test_spec"

        data = json.loads(result.output)
        assert data["found"] is True
        assert data["test_slug"] == "agent-test-py"
        assert data["command_pattern"] == "python3 ~/agent_test.py"
        assert data["purpose"] == "Tests the agent loop"
        assert data["output_format"] == "iteration counts"
        assert data["summary_fields"] == [
            "passed",
            "failed",
            "iterations_done",
        ]
        assert "timeout on large inputs" in data["common_failures"]
        assert data["normal_behavior"] == "All iterations pass"
        assert data["required_args"] == ["iterations", "host"]
        assert data["workflow_steps"] == ["setup-step", "main_check"]
        assert data["prerequisites"] == ["setup-step"]
        assert data["artifact_requirements"] == ["setup_ready_file"]
        assert data["when_missing_artifact_ask"] == "Run setup first?"
        assert data["success_criteria"] == "Main check passes."
        assert data["failure_criteria"] == "Main check fails."
        assert data["runs_observed"] == 10

    @pytest.mark.asyncio
    async def test_not_found_returns_slug(
        self, tool: LookupTestSpecTool
    ) -> None:
        """When no test spec exists, return found=False with the derived slug."""
        result = await tool.execute(
            call_id="c2", args={"test_name": "nonexistent_test"}
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["found"] is False
        assert "test_slug" in data
        assert "message" in data

    @pytest.mark.asyncio
    async def test_delegates_to_derive_test_slug(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """The tool must call derive_test_slug to compute the wiki filename."""
        with patch(
            "jules_daemon.agent.tools.lookup_test_spec.LookupTestSpecTool._lookup",
            wraps=tool._lookup,
        ) as mock_lookup:
            await tool.execute(
                call_id="c3", args={"test_name": "pytest tests/unit"}
            )
            mock_lookup.assert_called_once_with("pytest tests/unit")

    @pytest.mark.asyncio
    async def test_empty_required_args_is_empty_list(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """When a test has no required_args, the field should be an empty list."""
        _save_knowledge(wiki_root, required_args=())

        result = await tool.execute(
            call_id="c4", args={"test_name": "python3 ~/agent_test.py"}
        )

        data = json.loads(result.output)
        assert data["found"] is True
        assert data["required_args"] == []


# ---------------------------------------------------------------------------
# required_args flow
# ---------------------------------------------------------------------------


class TestRequiredArgs:
    """Verify required_args surfacing for missing-arg detection."""

    @pytest.mark.asyncio
    async def test_required_args_roundtrip(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """required_args survive save -> load -> tool output."""
        _save_knowledge(
            wiki_root,
            required_args=("iterations", "concurrency", "environment"),
        )

        result = await tool.execute(
            call_id="c5", args={"test_name": "python3 ~/agent_test.py"}
        )

        data = json.loads(result.output)
        assert data["required_args"] == ["iterations", "concurrency", "environment"]

    @pytest.mark.asyncio
    async def test_required_args_deduplication(
        self, wiki_root: Path
    ) -> None:
        """Duplicate required_args should be deduplicated by the wiki layer."""
        from jules_daemon.wiki.test_knowledge import (
            TestKnowledge,
            save_test_knowledge,
        )

        knowledge = TestKnowledge(
            test_slug="dedup-test",
            command_pattern="./dedup_test.sh",
            required_args=("host", "host", "port"),
        )
        save_test_knowledge(wiki_root, knowledge)

        tool = LookupTestSpecTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="c6", args={"test_name": "./dedup_test.sh"}
        )

        data = json.loads(result.output)
        if data["found"]:
            # The coercion layer deduplicates on load
            assert len(data["required_args"]) == len(set(data["required_args"]))


# ---------------------------------------------------------------------------
# Dual calling convention
# ---------------------------------------------------------------------------


class TestDualCallingConvention:
    """Verify both InfoRetrievalTool and legacy BaseTool calling conventions."""

    @pytest.mark.asyncio
    async def test_keyword_convention(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """InfoRetrievalTool keyword convention: execute(call_id=..., args=...)."""
        _save_knowledge(wiki_root)

        result = await tool.execute(
            call_id="kw1",
            args={"test_name": "python3 ~/agent_test.py"},
        )

        assert result.status is ToolResultStatus.SUCCESS
        assert result.call_id == "kw1"
        data = json.loads(result.output)
        assert data["found"] is True

    @pytest.mark.asyncio
    async def test_positional_convention(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """InfoRetrievalTool positional convention: execute(call_id, args)."""
        _save_knowledge(wiki_root)

        result = await tool.execute(
            "pos1", {"test_name": "python3 ~/agent_test.py"}
        )

        assert result.status is ToolResultStatus.SUCCESS
        assert result.call_id == "pos1"
        data = json.loads(result.output)
        assert data["found"] is True

    @pytest.mark.asyncio
    async def test_legacy_dict_convention(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """Legacy BaseTool convention: execute({"test_name": ..., "_call_id": ...})."""
        _save_knowledge(wiki_root)

        result = await tool.execute({
            "test_name": "python3 ~/agent_test.py",
            "_call_id": "legacy1",
        })

        assert result.status is ToolResultStatus.SUCCESS
        assert result.call_id == "legacy1"
        data = json.loads(result.output)
        assert data["found"] is True

    @pytest.mark.asyncio
    async def test_legacy_dict_without_call_id(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """Legacy convention without _call_id should use default."""
        _save_knowledge(wiki_root)

        result = await tool.execute({"test_name": "python3 ~/agent_test.py"})

        assert result.status is ToolResultStatus.SUCCESS
        assert result.call_id == "lookup_test_spec"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify error and edge case handling."""

    @pytest.mark.asyncio
    async def test_empty_test_name_returns_error(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Empty test_name must produce an error result."""
        result = await tool.execute(call_id="e1", args={"test_name": ""})

        assert result.status is ToolResultStatus.ERROR
        assert result.error_message
        assert "test_name" in result.error_message

    @pytest.mark.asyncio
    async def test_whitespace_only_test_name_returns_error(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Whitespace-only test_name must produce an error result."""
        result = await tool.execute(
            call_id="e2", args={"test_name": "   "}
        )

        assert result.status is ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_missing_test_name_returns_error(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Missing test_name parameter should trigger base class validation."""
        result = await tool.execute(call_id="e3", args={})

        assert result.status is ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_empty_call_id_uses_sentinel(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Empty call_id should be handled by the base class."""
        result = await tool.execute(call_id="", args={"test_name": "test"})

        # Base class uses "unknown" as sentinel for empty call_id
        assert result.status is ToolResultStatus.ERROR
        assert result.call_id == "unknown"

    @pytest.mark.asyncio
    async def test_wiki_io_error_returns_error_result(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Wiki I/O errors should be caught and returned as error results."""
        with patch.object(
            tool, "_lookup", side_effect=OSError("disk full")
        ):
            result = await tool.execute(
                call_id="e4", args={"test_name": "any_test"}
            )

        assert result.status is ToolResultStatus.ERROR
        assert "disk full" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_result_is_not_terminal(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Error results from this tool should not be terminal."""
        result = await tool.execute(call_id="e5", args={"test_name": ""})
        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_test_name(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """Leading/trailing whitespace in test_name should be stripped."""
        _save_knowledge(wiki_root)

        result = await tool.execute(
            call_id="e6",
            args={"test_name": "  python3 ~/agent_test.py  "},
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["found"] is True


# ---------------------------------------------------------------------------
# ToolRegistry integration
# ---------------------------------------------------------------------------


class TestToolRegistryIntegration:
    """Verify the tool works correctly when registered in a ToolRegistry."""

    @pytest.mark.asyncio
    async def test_register_and_execute(
        self, wiki_root: Path
    ) -> None:
        """Tool must be registrable and executable through the ToolRegistry."""
        from jules_daemon.agent.tool_registry import ToolRegistry
        from jules_daemon.agent.tool_types import ToolCall

        _save_knowledge(wiki_root, required_args=("iterations",))

        registry = ToolRegistry()
        tool = LookupTestSpecTool(wiki_root=wiki_root)
        registry.register(tool)

        assert "lookup_test_spec" in registry
        assert not registry.requires_approval("lookup_test_spec")

        call = ToolCall(
            call_id="reg1",
            tool_name="lookup_test_spec",
            arguments={"test_name": "python3 ~/agent_test.py"},
        )

        result = await registry.execute(call)
        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["found"] is True
        assert data["required_args"] == ["iterations"]

    def test_listed_as_read_only(self, wiki_root: Path) -> None:
        """Tool must appear in the read-only tool list."""
        from jules_daemon.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        tool = LookupTestSpecTool(wiki_root=wiki_root)
        registry.register(tool)

        read_only_names = [t.name for t in registry.list_read_only_tools()]
        assert "lookup_test_spec" in read_only_names

    def test_not_listed_as_approval_required(self, wiki_root: Path) -> None:
        """Tool must not appear in the approval-required list."""
        from jules_daemon.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        tool = LookupTestSpecTool(wiki_root=wiki_root)
        registry.register(tool)

        approval_names = [
            t.name for t in registry.list_approval_required_tools()
        ]
        assert "lookup_test_spec" not in approval_names

    def test_openai_schema_in_registry(self, wiki_root: Path) -> None:
        """Registry OpenAI schema export should include this tool."""
        from jules_daemon.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        tool = LookupTestSpecTool(wiki_root=wiki_root)
        registry.register(tool)

        schemas = registry.to_openai_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "lookup_test_spec" in names


# ---------------------------------------------------------------------------
# Fallback kwargs calling convention (covers lines 162-164)
# ---------------------------------------------------------------------------


class TestFallbackKwargsConvention:
    """Verify the fallback branch where kwargs are used as legacy dict."""

    @pytest.mark.asyncio
    async def test_kwargs_as_legacy_dict(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """Passing test_name as a keyword arg triggers the fallback branch."""
        _save_knowledge(wiki_root)

        result = await tool.execute(test_name="python3 ~/agent_test.py")

        assert result.status is ToolResultStatus.SUCCESS
        # Default call_id comes from the fallback
        assert result.call_id == "lookup_test_spec"
        data = json.loads(result.output)
        assert data["found"] is True

    @pytest.mark.asyncio
    async def test_kwargs_with_call_id(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """Passing _call_id as a keyword arg alongside test_name."""
        _save_knowledge(wiki_root)

        result = await tool.execute(
            test_name="python3 ~/agent_test.py",
            _call_id="fallback1",
        )

        assert result.status is ToolResultStatus.SUCCESS
        assert result.call_id == "fallback1"
        data = json.loads(result.output)
        assert data["found"] is True

    @pytest.mark.asyncio
    async def test_kwargs_only_not_found(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Fallback branch with a test_name that does not exist."""
        result = await tool.execute(test_name="nonexistent_command")

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["found"] is False


# ---------------------------------------------------------------------------
# Positional call_id without args dict
# ---------------------------------------------------------------------------


class TestPositionalCallIdOnly:
    """Verify positional convention with call_id string but no args dict."""

    @pytest.mark.asyncio
    async def test_positional_call_id_without_args(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Positional call_id with no args dict produces validation error."""
        result = await tool.execute("pos-no-args")

        # Empty args dict => missing required 'test_name'
        assert result.status is ToolResultStatus.ERROR
        assert result.call_id == "pos-no-args"


# ---------------------------------------------------------------------------
# Thread pool delegation
# ---------------------------------------------------------------------------


class TestThreadPoolDelegation:
    """Verify _lookup runs in a thread pool via asyncio.to_thread."""

    @pytest.mark.asyncio
    async def test_lookup_runs_in_thread_pool(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """_lookup must be called via asyncio.to_thread (not directly)."""
        _save_knowledge(wiki_root)

        with patch(
            "jules_daemon.agent.tools.lookup_test_spec.asyncio.to_thread",
            wraps=asyncio.to_thread,
        ) as mock_to_thread:
            result = await tool.execute(
                call_id="thread1",
                args={"test_name": "python3 ~/agent_test.py"},
            )

        assert result.status is ToolResultStatus.SUCCESS
        mock_to_thread.assert_called_once()
        # First arg to to_thread is the callable (_lookup method)
        call_args = mock_to_thread.call_args
        assert call_args[0][0] == tool._lookup


# ---------------------------------------------------------------------------
# Multiple test specs disambiguation
# ---------------------------------------------------------------------------


class TestMultipleSpecs:
    """Verify the tool correctly disambiguates multiple test specs."""

    @pytest.mark.asyncio
    async def test_different_commands_different_slugs(
        self, wiki_root: Path
    ) -> None:
        """Different commands should produce different slugs and specs."""
        _save_knowledge(
            wiki_root,
            slug="agent-test-py",
            command="python3 ~/agent_test.py",
            purpose="Tests the agent loop",
        )
        _save_knowledge(
            wiki_root,
            slug="pytest-tests-integration",
            command="pytest tests/integration",
            purpose="Integration test suite",
        )

        tool = LookupTestSpecTool(wiki_root=wiki_root)

        result1 = await tool.execute(
            call_id="multi1",
            args={"test_name": "python3 ~/agent_test.py"},
        )
        result2 = await tool.execute(
            call_id="multi2",
            args={"test_name": "pytest tests/integration"},
        )

        data1 = json.loads(result1.output)
        data2 = json.loads(result2.output)

        assert data1["found"] is True
        assert data2["found"] is True
        assert data1["test_slug"] != data2["test_slug"]
        assert data1["purpose"] == "Tests the agent loop"
        assert data2["purpose"] == "Integration test suite"

    @pytest.mark.asyncio
    async def test_same_command_same_slug(
        self, wiki_root: Path
    ) -> None:
        """Same command looked up twice must return consistent results."""
        _save_knowledge(wiki_root)

        tool = LookupTestSpecTool(wiki_root=wiki_root)

        result1 = await tool.execute(
            call_id="same1",
            args={"test_name": "python3 ~/agent_test.py"},
        )
        result2 = await tool.execute(
            call_id="same2",
            args={"test_name": "python3 ~/agent_test.py"},
        )

        data1 = json.loads(result1.output)
        data2 = json.loads(result2.output)

        assert data1["test_slug"] == data2["test_slug"]
        assert data1["command_pattern"] == data2["command_pattern"]

    @pytest.mark.asyncio
    async def test_prefers_specific_name_over_generic_test_token(
        self, wiki_root: Path
    ) -> None:
        """A query like 'run the step test' should resolve the 'step' spec."""
        _save_knowledge(
            wiki_root,
            slug="step-py",
            command="python3 /root/step.py --target {target}",
            purpose="Step test",
            required_args=("target",),
        )
        _save_knowledge(
            wiki_root,
            slug="agent-test-py",
            command="python3 ~/agent_test.py --iterations {iterations} --host {host}",
            purpose="Agent loop test",
            required_args=("iterations", "host", "timeout"),
        )

        tool = LookupTestSpecTool(wiki_root=wiki_root)

        result = await tool.execute(
            call_id="specific1",
            args={"test_name": "run the step test"},
        )

        data = json.loads(result.output)
        assert data["found"] is True
        assert data["test_slug"] == "step-py"
        assert data["required_args"] == ["target"]
        assert data["command_pattern"] == "python3 /root/step.py --target {target}"

    @pytest.mark.asyncio
    async def test_loads_discovered_spec_command_template_as_command_pattern(
        self, wiki_root: Path
    ) -> None:
        """Discovery pages with command_template should still load cleanly."""
        file_path = wiki_root / "pages" / "daemon" / "knowledge" / "test-step-py.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "---\n"
            "type: test-spec\n"
            "name: step\n"
            "test_slug: step-py\n"
            "command_template: python3 /root/step.py --target {target}\n"
            "required_args:\n"
            "  - target\n"
            "---\n"
            "# Step\n",
            encoding="utf-8",
        )

        tool = LookupTestSpecTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="specific2",
            args={"test_name": "step"},
        )

        data = json.loads(result.output)
        assert data["found"] is True
        assert data["test_slug"] == "step-py"
        assert data["command_pattern"] == "python3 /root/step.py --target {target}"
        assert data["required_args"] == ["target"]


# ---------------------------------------------------------------------------
# JSON output structure conformance
# ---------------------------------------------------------------------------


class TestJsonOutputStructure:
    """Verify the JSON output matches the documented schema."""

    @pytest.mark.asyncio
    async def test_success_output_has_all_fields(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """Successful lookup must return all documented fields."""
        _save_knowledge(
            wiki_root,
            required_args=("iterations",),
            common_failures=("timeout", "connection refused"),
        )

        result = await tool.execute(
            call_id="schema1",
            args={"test_name": "python3 ~/agent_test.py"},
        )

        data = json.loads(result.output)
        expected_keys = {
            "found",
            "test_slug",
            "command_pattern",
            "purpose",
            "output_format",
            "test_file_path",
            "summary_fields",
            "common_failures",
            "normal_behavior",
            "required_args",
            "workflow_steps",
            "prerequisites",
            "artifact_requirements",
            "when_missing_artifact_ask",
            "success_criteria",
            "failure_criteria",
            "runs_observed",
        }
        assert set(data.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_not_found_output_has_required_fields(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Not-found result must have found, test_slug, and message."""
        result = await tool.execute(
            call_id="schema2",
            args={"test_name": "missing_test"},
        )

        data = json.loads(result.output)
        assert set(data.keys()) == {"found", "test_slug", "message"}
        assert data["found"] is False
        assert isinstance(data["test_slug"], str)
        assert isinstance(data["message"], str)

    @pytest.mark.asyncio
    async def test_common_failures_is_list(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """common_failures must be serialized as a JSON array."""
        _save_knowledge(
            wiki_root,
            common_failures=("err1", "err2", "err3"),
        )

        result = await tool.execute(
            call_id="schema3",
            args={"test_name": "python3 ~/agent_test.py"},
        )

        data = json.loads(result.output)
        assert isinstance(data["common_failures"], list)
        assert len(data["common_failures"]) == 3

    @pytest.mark.asyncio
    async def test_test_file_path_is_returned(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """test_file_path should round-trip through the lookup result."""
        _save_knowledge(
            wiki_root,
            command="python3 /root/tests/step.py --target {target}",
            test_file_path="/root/tests/step.py",
            required_args=("target",),
        )

        result = await tool.execute(
            call_id="schema-file-path",
            args={"test_name": "step"},
        )

        data = json.loads(result.output)
        assert data["found"] is True
        assert data["test_file_path"] == "/root/tests/step.py"

    @pytest.mark.asyncio
    async def test_required_args_is_list(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """required_args must be serialized as a JSON array."""
        _save_knowledge(
            wiki_root,
            required_args=("host", "port"),
        )

        result = await tool.execute(
            call_id="schema4",
            args={"test_name": "python3 ~/agent_test.py"},
        )

        data = json.loads(result.output)
        assert isinstance(data["required_args"], list)
        assert data["required_args"] == ["host", "port"]

    @pytest.mark.asyncio
    async def test_summary_fields_is_list(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """summary_fields must be serialized as a JSON array."""
        _save_knowledge(
            wiki_root,
            summary_fields=("passed", "failed", "skipped"),
        )

        result = await tool.execute(
            call_id="schema4b",
            args={"test_name": "python3 ~/agent_test.py"},
        )

        data = json.loads(result.output)
        assert isinstance(data["summary_fields"], list)
        assert data["summary_fields"] == ["passed", "failed", "skipped"]

    @pytest.mark.asyncio
    async def test_runs_observed_is_integer(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """runs_observed must be serialized as a JSON integer."""
        _save_knowledge(wiki_root, runs_observed=42)

        result = await tool.execute(
            call_id="schema5",
            args={"test_name": "python3 ~/agent_test.py"},
        )

        data = json.loads(result.output)
        assert isinstance(data["runs_observed"], int)
        assert data["runs_observed"] == 42

    @pytest.mark.asyncio
    async def test_output_is_valid_json(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """Tool output must always be valid JSON."""
        _save_knowledge(wiki_root)

        result = await tool.execute(
            call_id="json1",
            args={"test_name": "python3 ~/agent_test.py"},
        )

        # Must not raise
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Tool result LLM message formatting
# ---------------------------------------------------------------------------


class TestToolResultFormatting:
    """Verify ToolResult formatting for LLM conversation context."""

    @pytest.mark.asyncio
    async def test_success_result_to_llm_message(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """Successful result should format as a success message for the LLM."""
        _save_knowledge(wiki_root)

        result = await tool.execute(
            call_id="fmt1",
            args={"test_name": "python3 ~/agent_test.py"},
        )

        msg = result.to_llm_message()
        assert "[lookup_test_spec]" in msg
        assert "success" in msg.lower()

    @pytest.mark.asyncio
    async def test_error_result_to_llm_message(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Error result should format as an error message for the LLM."""
        result = await tool.execute(call_id="fmt2", args={"test_name": ""})

        msg = result.to_llm_message()
        assert "[lookup_test_spec]" in msg
        assert "ERROR" in msg

    @pytest.mark.asyncio
    async def test_success_result_to_openai_message(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """Successful result should serialize to valid OpenAI tool message."""
        _save_knowledge(wiki_root)

        result = await tool.execute(
            call_id="oai1",
            args={"test_name": "python3 ~/agent_test.py"},
        )

        oai_msg = result.to_openai_tool_message()
        assert oai_msg["role"] == "tool"
        assert oai_msg["tool_call_id"] == "oai1"
        # Content should be valid JSON
        parsed = json.loads(oai_msg["content"])
        assert parsed["found"] is True

    @pytest.mark.asyncio
    async def test_error_result_to_openai_message(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Error result should serialize with ERROR prefix in content."""
        result = await tool.execute(call_id="oai2", args={"test_name": ""})

        oai_msg = result.to_openai_tool_message()
        assert oai_msg["role"] == "tool"
        assert oai_msg["content"].startswith("ERROR:")


# ---------------------------------------------------------------------------
# Concurrent execution safety
# ---------------------------------------------------------------------------


class TestConcurrentExecution:
    """Verify the tool is safe for concurrent invocations."""

    @pytest.mark.asyncio
    async def test_concurrent_lookups(
        self, wiki_root: Path
    ) -> None:
        """Multiple concurrent lookups must not interfere with each other."""
        _save_knowledge(wiki_root)
        _save_knowledge(
            wiki_root,
            slug="run-sh",
            command="./run.sh",
            purpose="Shell runner",
        )

        tool = LookupTestSpecTool(wiki_root=wiki_root)

        results = await asyncio.gather(
            tool.execute(
                call_id="cc1",
                args={"test_name": "python3 ~/agent_test.py"},
            ),
            tool.execute(
                call_id="cc2",
                args={"test_name": "./run.sh"},
            ),
            tool.execute(
                call_id="cc3",
                args={"test_name": "nonexistent"},
            ),
        )

        data1 = json.loads(results[0].output)
        data2 = json.loads(results[1].output)
        data3 = json.loads(results[2].output)

        assert data1["found"] is True
        assert data1["test_slug"] == "agent-test-py"
        assert data2["found"] is True
        assert data2["test_slug"] == "run-sh"
        assert data3["found"] is False

    @pytest.mark.asyncio
    async def test_concurrent_mixed_success_and_error(
        self, wiki_root: Path
    ) -> None:
        """Concurrent mix of success and error calls must all return correctly."""
        _save_knowledge(wiki_root)

        tool = LookupTestSpecTool(wiki_root=wiki_root)

        results = await asyncio.gather(
            tool.execute(
                call_id="mix1",
                args={"test_name": "python3 ~/agent_test.py"},
            ),
            tool.execute(call_id="mix2", args={"test_name": ""}),
            tool.execute(call_id="mix3", args={}),
        )

        assert results[0].status is ToolResultStatus.SUCCESS
        assert results[1].status is ToolResultStatus.ERROR
        assert results[2].status is ToolResultStatus.ERROR


# ---------------------------------------------------------------------------
# Special character handling
# ---------------------------------------------------------------------------


class TestSpecialCharacters:
    """Verify handling of test names with special characters."""

    @pytest.mark.asyncio
    async def test_test_name_with_args(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """Test name containing CLI args should derive correct slug."""
        _save_knowledge(wiki_root)

        result = await tool.execute(
            call_id="sc1",
            args={"test_name": "python3 ~/agent_test.py --iterations 100"},
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        # Should still resolve to the same slug as without args
        assert data["test_slug"] == "agent-test-py"

    @pytest.mark.asyncio
    async def test_test_name_with_path_separators(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Test name with path separators produces a slug."""
        result = await tool.execute(
            call_id="sc2",
            args={"test_name": "pytest tests/integration/test_api.py"},
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        # Should produce a non-empty slug
        assert data["test_slug"]
        assert "/" not in data["test_slug"]

    @pytest.mark.asyncio
    async def test_test_name_with_tilde(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Test name with ~ should be handled cleanly."""
        result = await tool.execute(
            call_id="sc3",
            args={"test_name": "~/scripts/run_test.sh"},
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["test_slug"]

    @pytest.mark.asyncio
    async def test_test_name_unicode(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Unicode characters in test name should not crash."""
        result = await tool.execute(
            call_id="sc4",
            args={"test_name": "python3 test_unicode_\u00e9.py"},
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert "test_slug" in data


# ---------------------------------------------------------------------------
# Wiki root validation
# ---------------------------------------------------------------------------


class TestWikiRootEdgeCases:
    """Verify behavior with various wiki root states."""

    @pytest.mark.asyncio
    async def test_nonexistent_wiki_root(self, tmp_path: Path) -> None:
        """Tool with nonexistent wiki root returns not-found (not crash)."""
        fake_root = tmp_path / "nonexistent_wiki"
        tool = LookupTestSpecTool(wiki_root=fake_root)

        result = await tool.execute(
            call_id="wr1",
            args={"test_name": "some_test"},
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["found"] is False

    @pytest.mark.asyncio
    async def test_empty_wiki_root(self, wiki_root: Path) -> None:
        """Initialized but empty wiki returns not-found."""
        tool = LookupTestSpecTool(wiki_root=wiki_root)

        result = await tool.execute(
            call_id="wr2",
            args={"test_name": "any_test"},
        )

        assert result.status is ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["found"] is False


# ---------------------------------------------------------------------------
# Agent loop integration patterns
# ---------------------------------------------------------------------------


class TestAgentLoopIntegration:
    """Verify patterns used by the agent loop for self-correction."""

    @pytest.mark.asyncio
    async def test_output_parseable_by_agent(
        self, wiki_root: Path, tool: LookupTestSpecTool
    ) -> None:
        """Agent loop must be able to parse the output to detect missing args."""
        _save_knowledge(
            wiki_root,
            required_args=("host", "port", "timeout"),
        )

        result = await tool.execute(
            call_id="al1",
            args={"test_name": "python3 ~/agent_test.py"},
        )

        # Simulate what the agent loop does: parse output, check required_args
        data = json.loads(result.output)
        assert data["found"] is True
        missing_args = data["required_args"]
        assert len(missing_args) == 3
        assert "host" in missing_args
        assert "port" in missing_args
        assert "timeout" in missing_args

    @pytest.mark.asyncio
    async def test_not_found_triggers_agent_recovery(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Not-found result should be parseable so agent can adapt."""
        result = await tool.execute(
            call_id="al2",
            args={"test_name": "unknown_test_xyz"},
        )

        data = json.loads(result.output)
        assert data["found"] is False
        # Agent can use the message to explain to the user
        assert "unknown_test_xyz" in data["message"]
        # Agent can use the slug for further lookups
        assert data["test_slug"]

    @pytest.mark.asyncio
    async def test_error_result_is_observable(
        self, tool: LookupTestSpecTool
    ) -> None:
        """Error results must carry enough info for agent self-correction."""
        result = await tool.execute(call_id="al3", args={"test_name": ""})

        assert result.is_error
        assert not result.is_terminal
        assert result.error_message
        assert "test_name" in result.error_message
