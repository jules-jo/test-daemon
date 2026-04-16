"""Tests for ReadWikiTool extending InfoRetrievalTool base class.

Validates that ReadWikiTool:
- Extends InfoRetrievalTool (not just BaseTool)
- Exposes correct name, description, and parameters_schema properties
- Produces valid OpenAI-compatible schemas via to_openai_schema()
- Converts to ToolSpec via to_tool_spec() with correct fields
- Has requires_human_approval == False (read-only tool)
- Provides backward-compatible spec property for ToolRegistry
- Handles all calling conventions (InfoRetrievalTool keyword, positional,
  legacy BaseTool dict, and fallback kwargs)
- Validates required parameters via base class validation
- Catches exceptions and returns error ToolResults
- Delegates to wiki.command_translation.find_by_query (no reimplementation)
- Delegates to wiki.test_knowledge.load_test_knowledge (no reimplementation)
- Runs wiki I/O in thread pool via asyncio.to_thread
- Returns JSON output with correct structure and all expected fields
- Respects _MAX_TRANSLATIONS cap
- Strips whitespace from queries before searching
- Returns complete field sets for translations and test knowledge
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from jules_daemon.agent.tool_base import InfoRetrievalTool
from jules_daemon.agent.tool_result import ToolResult, ToolResultStatus
from jules_daemon.agent.tool_types import ApprovalRequirement, ToolSpec
from jules_daemon.agent.tools.read_wiki import ReadWikiTool


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
def tool(wiki_root: Path) -> ReadWikiTool:
    """Fresh ReadWikiTool instance with a temp wiki."""
    return ReadWikiTool(wiki_root=wiki_root)


# ---------------------------------------------------------------------------
# Base class conformance
# ---------------------------------------------------------------------------


class TestReadWikiToolBaseClass:
    """Verify ReadWikiTool extends InfoRetrievalTool."""

    def test_is_info_retrieval_tool_subclass(self) -> None:
        """ReadWikiTool must be a subclass of InfoRetrievalTool."""
        assert issubclass(ReadWikiTool, InfoRetrievalTool)

    def test_instance_is_info_retrieval_tool(self, tool: ReadWikiTool) -> None:
        """ReadWikiTool instance must be an InfoRetrievalTool instance."""
        assert isinstance(tool, InfoRetrievalTool)

    def test_requires_human_approval_false(self, tool: ReadWikiTool) -> None:
        """Read-only tool must not require human approval."""
        assert tool.requires_human_approval is False


# ---------------------------------------------------------------------------
# Protocol properties
# ---------------------------------------------------------------------------


class TestReadWikiToolProperties:
    """Verify name, description, and parameters_schema properties."""

    def test_name_is_read_wiki(self, tool: ReadWikiTool) -> None:
        assert tool.name == "read_wiki"

    def test_description_is_nonempty_string(self, tool: ReadWikiTool) -> None:
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_description_mentions_wiki(self, tool: ReadWikiTool) -> None:
        """Description should mention wiki for LLM context."""
        assert "wiki" in tool.description.lower()

    def test_parameters_schema_is_object_type(self, tool: ReadWikiTool) -> None:
        schema = tool.parameters_schema
        assert isinstance(schema, dict)
        assert schema["type"] == "object"

    def test_parameters_schema_has_query(self, tool: ReadWikiTool) -> None:
        schema = tool.parameters_schema
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"

    def test_parameters_schema_query_is_required(self, tool: ReadWikiTool) -> None:
        schema = tool.parameters_schema
        assert "query" in schema.get("required", [])

    def test_parameters_schema_has_ssh_host(self, tool: ReadWikiTool) -> None:
        schema = tool.parameters_schema
        assert "ssh_host" in schema["properties"]
        assert schema["properties"]["ssh_host"]["type"] == "string"

    def test_parameters_schema_ssh_host_is_optional(self, tool: ReadWikiTool) -> None:
        schema = tool.parameters_schema
        required = schema.get("required", [])
        assert "ssh_host" not in required


# ---------------------------------------------------------------------------
# OpenAI schema serialization
# ---------------------------------------------------------------------------


class TestReadWikiToolOpenAISchema:
    """Verify to_openai_schema() produces valid OpenAI function tool schemas."""

    def test_schema_structure(self, tool: ReadWikiTool) -> None:
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert "function" in schema
        fn = schema["function"]
        assert fn["name"] == "read_wiki"
        assert "description" in fn
        assert "parameters" in fn

    def test_schema_parameters_include_query(self, tool: ReadWikiTool) -> None:
        schema = tool.to_openai_schema()
        params = schema["function"]["parameters"]
        assert "query" in params["properties"]

    def test_schema_required_includes_query(self, tool: ReadWikiTool) -> None:
        schema = tool.to_openai_schema()
        params = schema["function"]["parameters"]
        assert "query" in params.get("required", [])

    def test_schema_is_json_serializable(self, tool: ReadWikiTool) -> None:
        schema = tool.to_openai_schema()
        serialized = json.dumps(schema)
        deserialized = json.loads(serialized)
        assert deserialized["function"]["name"] == "read_wiki"


# ---------------------------------------------------------------------------
# ToolSpec conversion
# ---------------------------------------------------------------------------


class TestReadWikiToolSpecConversion:
    """Verify to_tool_spec() and backward-compatible spec property."""

    def test_to_tool_spec_returns_tool_spec(self, tool: ReadWikiTool) -> None:
        spec = tool.to_tool_spec()
        assert isinstance(spec, ToolSpec)

    def test_to_tool_spec_name(self, tool: ReadWikiTool) -> None:
        spec = tool.to_tool_spec()
        assert spec.name == "read_wiki"

    def test_to_tool_spec_approval_none(self, tool: ReadWikiTool) -> None:
        spec = tool.to_tool_spec()
        assert spec.approval is ApprovalRequirement.NONE

    def test_to_tool_spec_is_read_only(self, tool: ReadWikiTool) -> None:
        spec = tool.to_tool_spec()
        assert spec.is_read_only is True

    def test_to_tool_spec_has_query_param(self, tool: ReadWikiTool) -> None:
        spec = tool.to_tool_spec()
        param_names = [p.name for p in spec.parameters]
        assert "query" in param_names

    def test_to_tool_spec_query_is_required(self, tool: ReadWikiTool) -> None:
        spec = tool.to_tool_spec()
        query_param = next(p for p in spec.parameters if p.name == "query")
        assert query_param.required is True

    def test_to_tool_spec_ssh_host_is_optional(self, tool: ReadWikiTool) -> None:
        spec = tool.to_tool_spec()
        ssh_param = next(p for p in spec.parameters if p.name == "ssh_host")
        assert ssh_param.required is False

    def test_spec_property_returns_tool_spec(self, tool: ReadWikiTool) -> None:
        """Backward-compatible spec property for ToolRegistry."""
        spec = tool.spec
        assert isinstance(spec, ToolSpec)
        assert spec.name == "read_wiki"

    def test_spec_property_is_cached(self, tool: ReadWikiTool) -> None:
        """Repeated access returns the same ToolSpec instance."""
        spec1 = tool.spec
        spec2 = tool.spec
        assert spec1 is spec2


# ---------------------------------------------------------------------------
# Execution -- InfoRetrievalTool calling convention
# ---------------------------------------------------------------------------


class TestReadWikiToolExecution:
    """Verify execution via InfoRetrievalTool calling convention."""

    @pytest.mark.asyncio
    async def test_successful_search_returns_success(
        self, tool: ReadWikiTool,
    ) -> None:
        result = await tool.execute(
            call_id="c1", args={"query": "run tests"}
        )
        assert result.is_success
        assert result.call_id == "c1"
        assert result.tool_name == "read_wiki"

    @pytest.mark.asyncio
    async def test_result_contains_translations_key(
        self, tool: ReadWikiTool,
    ) -> None:
        result = await tool.execute(
            call_id="c2", args={"query": "anything"}
        )
        data = json.loads(result.output)
        assert "translations" in data

    @pytest.mark.asyncio
    async def test_result_contains_test_knowledge_key(
        self, tool: ReadWikiTool,
    ) -> None:
        result = await tool.execute(
            call_id="c3", args={"query": "some test"}
        )
        data = json.loads(result.output)
        assert "test_knowledge" in data

    @pytest.mark.asyncio
    async def test_result_contains_query_echo(
        self, tool: ReadWikiTool,
    ) -> None:
        result = await tool.execute(
            call_id="c4", args={"query": "smoke"}
        )
        data = json.loads(result.output)
        assert data["query"] == "smoke"

    @pytest.mark.asyncio
    async def test_empty_wiki_returns_empty_translations(
        self, tool: ReadWikiTool,
    ) -> None:
        result = await tool.execute(
            call_id="c5", args={"query": "nonexistent"}
        )
        data = json.loads(result.output)
        assert data["translations"] == []
        assert data["test_knowledge"] is None


# ---------------------------------------------------------------------------
# Execution -- legacy BaseTool calling convention (backward compat)
# ---------------------------------------------------------------------------


class TestReadWikiToolLegacyConvention:
    """Verify backward-compatible single-dict calling convention."""

    @pytest.mark.asyncio
    async def test_legacy_call_works(self, tool: ReadWikiTool) -> None:
        """ToolRegistry passes a single dict with _call_id injected."""
        result = await tool.execute(
            {"query": "run tests", "_call_id": "legacy1"}
        )
        assert result.is_success
        assert result.call_id == "legacy1"
        assert result.tool_name == "read_wiki"

    @pytest.mark.asyncio
    async def test_legacy_call_default_call_id(self, tool: ReadWikiTool) -> None:
        """When _call_id is missing from the dict, use a default."""
        result = await tool.execute({"query": "run tests"})
        assert result.is_success
        assert result.call_id == "read_wiki"

    @pytest.mark.asyncio
    async def test_legacy_call_searches_wiki(
        self, wiki_root: Path,
    ) -> None:
        """Legacy call must still delegate to wiki modules."""
        from jules_daemon.wiki.command_translation import (
            CommandTranslation,
            save,
        )

        save(wiki_root, CommandTranslation(
            natural_language="run the smoke tests",
            resolved_shell="pytest tests/smoke -v",
            ssh_host="staging.example.com",
        ))

        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute(
            {"query": "smoke tests", "_call_id": "legacy2"}
        )

        data = json.loads(result.output)
        assert len(data["translations"]) >= 1
        assert data["translations"][0]["natural_language"] == "run the smoke tests"


# ---------------------------------------------------------------------------
# Validation (inherited from InfoRetrievalTool base class)
# ---------------------------------------------------------------------------


class TestReadWikiToolValidation:
    """Verify argument validation via InfoRetrievalTool base class."""

    @pytest.mark.asyncio
    async def test_missing_query_returns_error(self, tool: ReadWikiTool) -> None:
        """Missing required 'query' parameter returns an error result."""
        result = await tool.execute(call_id="v1", args={})
        assert result.is_error
        assert "query" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self, tool: ReadWikiTool) -> None:
        """Empty string query is caught by _execute_impl."""
        result = await tool.execute(call_id="v2", args={"query": ""})
        assert result.is_error
        assert "required" in (result.error_message or "").lower() or "empty" in (result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_whitespace_only_query_returns_error(
        self, tool: ReadWikiTool,
    ) -> None:
        result = await tool.execute(call_id="v3", args={"query": "   "})
        assert result.is_error

    @pytest.mark.asyncio
    async def test_empty_call_id_returns_error(self, tool: ReadWikiTool) -> None:
        """Empty call_id should be caught by base class validation."""
        result = await tool.execute(call_id="", args={"query": "test"})
        assert result.is_error
        assert "call_id" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_extra_args_tolerated(self, tool: ReadWikiTool) -> None:
        """Extra arguments beyond schema should not cause errors."""
        result = await tool.execute(
            call_id="v4", args={"query": "test", "extra": "ignored"}
        )
        assert result.is_success


# ---------------------------------------------------------------------------
# Wiki delegation
# ---------------------------------------------------------------------------


class TestReadWikiToolDelegation:
    """Verify delegation to existing wiki modules (no reimplementation)."""

    @pytest.mark.asyncio
    async def test_delegates_to_find_by_query(
        self, tool: ReadWikiTool,
    ) -> None:
        """Must call command_translation.find_by_query."""
        with patch(
            "jules_daemon.wiki.command_translation.find_by_query",
            return_value=[],
        ) as mock_find:
            with patch(
                "jules_daemon.wiki.test_knowledge.load_test_knowledge",
                return_value=None,
            ):
                await tool.execute(call_id="d1", args={"query": "test"})
            mock_find.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegates_to_load_test_knowledge(
        self, tool: ReadWikiTool,
    ) -> None:
        """Must call test_knowledge.load_test_knowledge."""
        with patch(
            "jules_daemon.wiki.command_translation.find_by_query",
            return_value=[],
        ):
            with patch(
                "jules_daemon.wiki.test_knowledge.load_test_knowledge",
                return_value=None,
            ) as mock_load:
                await tool.execute(call_id="d2", args={"query": "test"})
                mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegates_to_derive_test_slug(
        self, tool: ReadWikiTool,
    ) -> None:
        """Must call test_knowledge.derive_test_slug."""
        with patch(
            "jules_daemon.wiki.command_translation.find_by_query",
            return_value=[],
        ):
            with patch(
                "jules_daemon.wiki.test_knowledge.derive_test_slug",
                return_value="test-slug",
            ) as mock_slug:
                with patch(
                    "jules_daemon.wiki.test_knowledge.load_test_knowledge",
                    return_value=None,
                ):
                    await tool.execute(
                        call_id="d3", args={"query": "test"}
                    )
                    mock_slug.assert_called_once_with("test")

    @pytest.mark.asyncio
    async def test_ssh_host_passed_to_find_by_query(
        self, tool: ReadWikiTool,
    ) -> None:
        """ssh_host parameter must be forwarded to find_by_query."""
        with patch(
            "jules_daemon.wiki.command_translation.find_by_query",
            return_value=[],
        ) as mock_find:
            with patch(
                "jules_daemon.wiki.test_knowledge.load_test_knowledge",
                return_value=None,
            ):
                await tool.execute(
                    call_id="d4",
                    args={"query": "test", "ssh_host": "myhost"},
                )
            _, kwargs = mock_find.call_args
            # ssh_host should appear in the kwargs
            call_kwargs = mock_find.call_args
            assert "ssh_host" in call_kwargs.kwargs or (
                len(call_kwargs.args) > 0
                and any("myhost" in str(a) for a in call_kwargs.args)
            )


# ---------------------------------------------------------------------------
# Integration with real wiki data
# ---------------------------------------------------------------------------


class TestReadWikiToolIntegration:
    """Integration tests with real wiki file I/O."""

    @pytest.mark.asyncio
    async def test_finds_saved_translation(self, wiki_root: Path) -> None:
        from jules_daemon.wiki.command_translation import (
            CommandTranslation,
            save,
        )

        save(wiki_root, CommandTranslation(
            natural_language="run the smoke tests",
            resolved_shell="pytest tests/smoke -v",
            ssh_host="staging.example.com",
        ))

        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="int1", args={"query": "smoke tests"}
        )

        assert result.is_success
        data = json.loads(result.output)
        assert len(data["translations"]) >= 1
        assert data["translations"][0]["natural_language"] == "run the smoke tests"

    @pytest.mark.asyncio
    async def test_finds_saved_test_knowledge(self, wiki_root: Path) -> None:
        from jules_daemon.wiki.test_knowledge import (
            TestKnowledge,
            save_test_knowledge,
        )

        knowledge = TestKnowledge(
            test_slug="agent-test-py",
            command_pattern="python3 ~/agent_test.py",
            purpose="Tests the agent loop",
            runs_observed=5,
        )
        save_test_knowledge(wiki_root, knowledge)

        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="int2",
            args={"query": "python3 ~/agent_test.py"},
        )

        assert result.is_success
        data = json.loads(result.output)
        assert data["test_knowledge"] is not None
        assert data["test_knowledge"]["test_slug"] == "agent-test-py"
        assert data["test_knowledge"]["purpose"] == "Tests the agent loop"

    @pytest.mark.asyncio
    async def test_ssh_host_filter_works(self, wiki_root: Path) -> None:
        from jules_daemon.wiki.command_translation import (
            CommandTranslation,
            save,
        )

        save(wiki_root, CommandTranslation(
            natural_language="run tests",
            resolved_shell="pytest",
            ssh_host="host-a",
        ))
        save(wiki_root, CommandTranslation(
            natural_language="run tests",
            resolved_shell="pytest",
            ssh_host="host-b",
        ))

        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="int3",
            args={"query": "run tests", "ssh_host": "host-a"},
        )

        data = json.loads(result.output)
        for t in data["translations"]:
            assert t["ssh_host"] == "host-a"


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestReadWikiToolErrorResilience:
    """Verify graceful error handling for wiki I/O failures."""

    @pytest.mark.asyncio
    async def test_wiki_io_error_returns_error_result(
        self, tool: ReadWikiTool,
    ) -> None:
        """File system errors should be caught and returned as error results."""
        with patch(
            "jules_daemon.wiki.command_translation.find_by_query",
            side_effect=OSError("disk full"),
        ):
            result = await tool.execute(
                call_id="e1", args={"query": "test"}
            )
        assert result.is_error
        assert "disk full" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_result_is_never_terminal(self, tool: ReadWikiTool) -> None:
        """Read-only tool errors should never be terminal (not DENIED)."""
        result = await tool.execute(call_id="e2", args={"query": ""})
        assert result.is_error
        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_test_knowledge_load_error_returns_error_result(
        self, tool: ReadWikiTool,
    ) -> None:
        """Error in load_test_knowledge should be caught by base class."""
        with patch(
            "jules_daemon.wiki.command_translation.find_by_query",
            return_value=[],
        ):
            with patch(
                "jules_daemon.wiki.test_knowledge.load_test_knowledge",
                side_effect=RuntimeError("corrupted file"),
            ):
                result = await tool.execute(
                    call_id="e3", args={"query": "test"}
                )
        assert result.is_error
        assert "corrupted file" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_derive_slug_error_returns_error_result(
        self, tool: ReadWikiTool,
    ) -> None:
        """Error in derive_test_slug should be caught by base class."""
        with patch(
            "jules_daemon.wiki.command_translation.find_by_query",
            return_value=[],
        ):
            with patch(
                "jules_daemon.wiki.test_knowledge.derive_test_slug",
                side_effect=TypeError("unexpected input"),
            ):
                result = await tool.execute(
                    call_id="e4", args={"query": "test"}
                )
        assert result.is_error
        assert "unexpected input" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Positional calling convention (execute(call_id, args))
# ---------------------------------------------------------------------------


class TestReadWikiToolPositionalConvention:
    """Verify positional InfoRetrievalTool calling convention."""

    @pytest.mark.asyncio
    async def test_positional_call_with_args(self, tool: ReadWikiTool) -> None:
        """execute(call_id, args_dict) must work as positional args."""
        result = await tool.execute("pos1", {"query": "run tests"})
        assert result.is_success
        assert result.call_id == "pos1"
        assert result.tool_name == "read_wiki"

    @pytest.mark.asyncio
    async def test_positional_call_with_empty_args(
        self, tool: ReadWikiTool,
    ) -> None:
        """execute(call_id) with no second arg uses empty dict -- missing query."""
        result = await tool.execute("pos2")
        assert result.is_error
        assert "query" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_positional_call_output_is_valid_json(
        self, tool: ReadWikiTool,
    ) -> None:
        """Positional convention must produce valid JSON output."""
        result = await tool.execute("pos3", {"query": "check"})
        data = json.loads(result.output)
        assert "translations" in data
        assert "test_knowledge" in data
        assert "query" in data

    @pytest.mark.asyncio
    async def test_positional_call_query_echoed(
        self, tool: ReadWikiTool,
    ) -> None:
        """The query is echoed in the output for traceability."""
        result = await tool.execute("pos4", {"query": "my-query"})
        data = json.loads(result.output)
        assert data["query"] == "my-query"


# ---------------------------------------------------------------------------
# Fallback kwargs calling convention
# ---------------------------------------------------------------------------


class TestReadWikiToolFallbackConvention:
    """Verify the fallback kwargs-based calling convention."""

    @pytest.mark.asyncio
    async def test_fallback_kwargs_with_call_id(
        self, tool: ReadWikiTool,
    ) -> None:
        """execute(_call_id=..., query=...) uses fallback path."""
        result = await tool.execute(_call_id="fb1", query="run tests")
        assert result.is_success
        assert result.call_id == "fb1"

    @pytest.mark.asyncio
    async def test_fallback_kwargs_default_call_id(
        self, tool: ReadWikiTool,
    ) -> None:
        """Fallback without _call_id defaults to 'read_wiki'."""
        result = await tool.execute(query="run tests")
        assert result.is_success
        assert result.call_id == "read_wiki"

    @pytest.mark.asyncio
    async def test_fallback_kwargs_output_structure(
        self, tool: ReadWikiTool,
    ) -> None:
        """Fallback path must produce correct JSON structure."""
        result = await tool.execute(query="something")
        data = json.loads(result.output)
        assert data["query"] == "something"
        assert "translations" in data
        assert "test_knowledge" in data


# ---------------------------------------------------------------------------
# Output JSON structure completeness
# ---------------------------------------------------------------------------


class TestReadWikiToolOutputStructure:
    """Verify the JSON output contains all expected fields."""

    @pytest.mark.asyncio
    async def test_translation_fields_complete(
        self, wiki_root: Path,
    ) -> None:
        """Each translation dict must include all expected fields."""
        from jules_daemon.wiki.command_translation import (
            CommandTranslation,
            save,
        )

        save(wiki_root, CommandTranslation(
            natural_language="deploy service",
            resolved_shell="kubectl apply -f deploy.yaml",
            ssh_host="prod.example.com",
            model_id="gpt4-mesh",
        ))

        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="out1", args={"query": "deploy"}
        )
        data = json.loads(result.output)
        assert len(data["translations"]) >= 1

        translation = data["translations"][0]
        expected_keys = {
            "natural_language",
            "resolved_shell",
            "ssh_host",
            "outcome",
            "model_id",
        }
        assert set(translation.keys()) == expected_keys
        assert translation["natural_language"] == "deploy service"
        assert translation["resolved_shell"] == "kubectl apply -f deploy.yaml"
        assert translation["ssh_host"] == "prod.example.com"
        assert translation["outcome"] == "approved"
        assert translation["model_id"] == "gpt4-mesh"

    @pytest.mark.asyncio
    async def test_knowledge_fields_complete(
        self, wiki_root: Path,
    ) -> None:
        """Test knowledge dict must include all expected fields."""
        from jules_daemon.wiki.test_knowledge import (
            TestKnowledge,
            derive_test_slug,
            save_test_knowledge,
        )

        # Use the exact slug that derive_test_slug produces for our query
        query = "python3 ~/agent_test.py --iterations 100"
        expected_slug = derive_test_slug(query)

        knowledge = TestKnowledge(
            test_slug=expected_slug,
            command_pattern="python3 ~/agent_test.py",
            purpose="Test the agent loop",
            output_format="line-by-line status",
            common_failures=("timeout", "auth error"),
            normal_behavior="Completes in 30s",
            runs_observed=10,
        )
        save_test_knowledge(wiki_root, knowledge)

        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="out2", args={"query": query},
        )
        data = json.loads(result.output)
        kn = data["test_knowledge"]
        assert kn is not None

        expected_keys = {
            "test_slug",
            "command_pattern",
            "purpose",
            "output_format",
            "test_file_path",
            "summary_fields",
            "required_args",
            "workflow_steps",
            "prerequisites",
            "artifact_requirements",
            "when_missing_artifact_ask",
            "success_criteria",
            "failure_criteria",
            "common_failures",
            "normal_behavior",
            "runs_observed",
        }
        assert set(kn.keys()) == expected_keys
        assert kn["test_slug"] == expected_slug
        assert kn["command_pattern"] == "python3 ~/agent_test.py"
        assert kn["purpose"] == "Test the agent loop"
        assert kn["output_format"] == "line-by-line status"
        assert kn["common_failures"] == ["timeout", "auth error"]
        assert kn["normal_behavior"] == "Completes in 30s"
        assert kn["runs_observed"] == 10

    @pytest.mark.asyncio
    async def test_output_is_valid_json_string(
        self, tool: ReadWikiTool,
    ) -> None:
        """Output must be a valid JSON string, not a dict."""
        result = await tool.execute(call_id="out3", args={"query": "test"})
        assert isinstance(result.output, str)
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Query preprocessing
# ---------------------------------------------------------------------------


class TestReadWikiToolQueryPreprocessing:
    """Verify query whitespace stripping and edge cases."""

    @pytest.mark.asyncio
    async def test_leading_trailing_whitespace_stripped(
        self, wiki_root: Path,
    ) -> None:
        """Query with leading/trailing whitespace is stripped before search."""
        from jules_daemon.wiki.command_translation import (
            CommandTranslation,
            save,
        )

        save(wiki_root, CommandTranslation(
            natural_language="run smoke tests",
            resolved_shell="pytest smoke",
            ssh_host="staging",
        ))

        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="qp1", args={"query": "  smoke tests  "}
        )
        data = json.loads(result.output)
        # Stripped query should still match
        assert len(data["translations"]) >= 1
        # Echo should be the stripped version
        assert data["query"] == "smoke tests"

    @pytest.mark.asyncio
    async def test_query_with_special_characters(
        self, tool: ReadWikiTool,
    ) -> None:
        """Queries with special characters should not crash."""
        result = await tool.execute(
            call_id="qp2", args={"query": "test --flag=value /path/to/script"}
        )
        assert result.is_success
        data = json.loads(result.output)
        assert data["query"] == "test --flag=value /path/to/script"


# ---------------------------------------------------------------------------
# Max translations cap
# ---------------------------------------------------------------------------


class TestReadWikiToolMaxTranslations:
    """Verify the _MAX_TRANSLATIONS limit is respected."""

    @pytest.mark.asyncio
    async def test_max_translations_passed_to_find_by_query(
        self, tool: ReadWikiTool,
    ) -> None:
        """find_by_query must be called with max_results=_MAX_TRANSLATIONS."""
        with patch(
            "jules_daemon.wiki.command_translation.find_by_query",
            return_value=[],
        ) as mock_find:
            with patch(
                "jules_daemon.wiki.test_knowledge.load_test_knowledge",
                return_value=None,
            ):
                await tool.execute(
                    call_id="mt1", args={"query": "test"}
                )

            call_kwargs = mock_find.call_args.kwargs
            assert call_kwargs.get("max_results") == 5

    @pytest.mark.asyncio
    async def test_returns_at_most_max_translations(
        self, wiki_root: Path,
    ) -> None:
        """When more translations match, only _MAX_TRANSLATIONS returned."""
        from jules_daemon.wiki.command_translation import (
            CommandTranslation,
            save,
        )

        # Save 8 translations matching "run"
        for i in range(8):
            save(wiki_root, CommandTranslation(
                natural_language=f"run test suite {i}",
                resolved_shell=f"pytest tests/{i}",
                ssh_host="host",
            ))

        tool = ReadWikiTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="mt2", args={"query": "run"}
        )
        data = json.loads(result.output)
        assert len(data["translations"]) <= 5


# ---------------------------------------------------------------------------
# Thread pool execution verification
# ---------------------------------------------------------------------------


class TestReadWikiToolThreadPool:
    """Verify wiki I/O runs in thread pool via asyncio.to_thread."""

    @pytest.mark.asyncio
    async def test_search_wiki_runs_in_thread(
        self, tool: ReadWikiTool,
    ) -> None:
        """_search_wiki must be dispatched via asyncio.to_thread."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = {
                "translations": [],
                "test_knowledge": None,
                "query": "test",
            }
            result = await tool.execute(
                call_id="tp1", args={"query": "test"}
            )
            mock_to_thread.assert_called_once()
            # First arg to asyncio.to_thread should be _search_wiki
            called_func = mock_to_thread.call_args.args[0]
            assert called_func.__name__ == "_search_wiki"

    @pytest.mark.asyncio
    async def test_search_wiki_receives_correct_args(
        self, tool: ReadWikiTool,
    ) -> None:
        """_search_wiki receives the stripped query and ssh_host."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = {
                "translations": [],
                "test_knowledge": None,
                "query": "my query",
            }
            await tool.execute(
                call_id="tp2",
                args={"query": " my query ", "ssh_host": "myhost"},
            )
            call_args = mock_to_thread.call_args.args
            # args[0] is _search_wiki, args[1] is query, args[2] is ssh_host
            assert call_args[1] == "my query"
            assert call_args[2] == "myhost"

    @pytest.mark.asyncio
    async def test_search_wiki_none_ssh_host(
        self, tool: ReadWikiTool,
    ) -> None:
        """When ssh_host is not provided, None is passed to _search_wiki."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = {
                "translations": [],
                "test_knowledge": None,
                "query": "test",
            }
            await tool.execute(call_id="tp3", args={"query": "test"})
            call_args = mock_to_thread.call_args.args
            assert call_args[2] is None
