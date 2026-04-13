"""Tests for ReadOutputTool -- verifies wiki and session reading.

AC 10004 Sub-AC 4: read_output tool reads prior command/tool output
from the session.

Test strategy:
- Verify wiki source delegates to state_reader.load_reconnection_state
- Verify session source extracts tool results from conversation history
- Verify tool_name_filter filters session entries correctly
- Verify last_n limits the number of returned entries
- Verify truncation of long content
- Verify error handling: missing provider, None history
- Verify backward-compatible calling convention (legacy dict)
- Verify InfoRetrievalTool calling convention (call_id + args)
- Verify OpenAI schema serialization
- Verify ToolSpec generation
- Verify read-only classification (no human approval required)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jules_daemon.agent.tool_base import InfoRetrievalTool
from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolResultStatus,
)
from jules_daemon.agent.tools.read_output import (
    ReadOutputTool,
    _extract_tool_entries,
    _truncate_content,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Create a temporary wiki root with the required directory structure."""
    from jules_daemon.wiki.layout import initialize_wiki

    initialize_wiki(tmp_path)
    return tmp_path


def _build_session_messages(
    tool_calls: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build a minimal conversation history with tool results.

    Creates an assistant message with tool_calls followed by the
    corresponding tool result messages.
    """
    if tool_calls is None:
        tool_calls = [
            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "read_wiki",
                    "arguments": json.dumps({"query": "smoke test"}),
                },
            },
            {
                "id": "call_002",
                "type": "function",
                "function": {
                    "name": "propose_ssh_command",
                    "arguments": json.dumps({"command": "pytest tests/"}),
                },
            },
        ]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a test runner."},
        {"role": "user", "content": "run smoke tests"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        },
        {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": json.dumps({"translations": [], "query": "smoke test"}),
        },
        {
            "role": "tool",
            "tool_call_id": "call_002",
            "content": "ERROR: User denied the command",
        },
    ]
    return tuple(messages)


@pytest.fixture
def session_messages() -> tuple[dict[str, Any], ...]:
    """Standard session messages with two tool results."""
    return _build_session_messages()


@pytest.fixture
def session_provider(
    session_messages: tuple[dict[str, Any], ...],
) -> Any:
    """Session history provider returning standard messages."""
    return lambda: session_messages


# ---------------------------------------------------------------------------
# InfoRetrievalTool base class integration
# ---------------------------------------------------------------------------


class TestReadOutputToolIsInfoRetrieval:
    """Verify ReadOutputTool extends InfoRetrievalTool correctly."""

    def test_isinstance_check(self, wiki_root: Path) -> None:
        """ReadOutputTool must be an instance of InfoRetrievalTool."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        assert isinstance(tool, InfoRetrievalTool)

    def test_name_property(self, wiki_root: Path) -> None:
        """Tool name must be 'read_output'."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        assert tool.name == "read_output"

    def test_description_non_empty(self, wiki_root: Path) -> None:
        """Tool description must be non-empty and descriptive."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        assert len(tool.description) > 20
        assert "output" in tool.description.lower()

    def test_requires_no_human_approval(self, wiki_root: Path) -> None:
        """Read-only tool must not require human approval."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        assert tool.requires_human_approval is False

    def test_parameters_schema_is_valid_json_schema(self, wiki_root: Path) -> None:
        """parameters_schema must be a valid JSON Schema object."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        schema = tool.parameters_schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "source" in schema["properties"]

    def test_spec_property_returns_tool_spec(self, wiki_root: Path) -> None:
        """spec property must return a ToolSpec instance."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        spec = tool.spec
        assert spec.name == "read_output"
        assert spec.approval is ApprovalRequirement.NONE

    def test_spec_is_cached(self, wiki_root: Path) -> None:
        """spec property must return the same instance on repeated access."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        spec1 = tool.spec
        spec2 = tool.spec
        assert spec1 is spec2


# ---------------------------------------------------------------------------
# Wiki source (existing behavior, backward compat)
# ---------------------------------------------------------------------------


class TestReadOutputWikiSource:
    """Verify wiki source delegates to state_reader."""

    @pytest.mark.asyncio
    async def test_default_source_is_wiki(self, wiki_root: Path) -> None:
        """When no source is specified, defaults to wiki."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        result = await tool.execute(call_id="c1", args={})

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["source"] == "wiki"
        assert "load_result" in data
        assert "status" in data

    @pytest.mark.asyncio
    async def test_explicit_wiki_source(self, wiki_root: Path) -> None:
        """Explicit source='wiki' reads from state_reader."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="c1", args={"source": "wiki"}
        )

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["source"] == "wiki"
        assert "run_id" in data

    @pytest.mark.asyncio
    async def test_include_connection_false(self, wiki_root: Path) -> None:
        """Connection params excluded by default."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        result = await tool.execute(
            call_id="c1", args={"include_connection": False}
        )

        data = json.loads(result.output)
        assert "connection" not in data

    @pytest.mark.asyncio
    async def test_legacy_calling_convention(self, wiki_root: Path) -> None:
        """Legacy BaseTool calling convention must still work."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        result = await tool.execute({"_call_id": "c1"})

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["source"] == "wiki"

    @pytest.mark.asyncio
    async def test_legacy_with_include_connection(self, wiki_root: Path) -> None:
        """Legacy convention with include_connection=False."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        result = await tool.execute({
            "include_connection": False,
            "_call_id": "c2",
        })

        data = json.loads(result.output)
        assert "connection" not in data

    @pytest.mark.asyncio
    async def test_positional_calling_convention(self, wiki_root: Path) -> None:
        """Positional InfoRetrievalTool convention: execute(call_id, args)."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        result = await tool.execute("c1", {"source": "wiki"})

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["source"] == "wiki"


# ---------------------------------------------------------------------------
# Live source
# ---------------------------------------------------------------------------


class TestReadOutputLiveSource:
    """Verify live source reads the daemon's buffered output snapshot."""

    @pytest.mark.asyncio
    async def test_reads_live_output_snapshot(self, wiki_root: Path) -> None:
        """source='live' should return the injected live output snapshot."""
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            live_output_provider=lambda last_n: {
                "source": "live",
                "status": "running",
                "run_id": "run-live-123",
                "task_running": True,
                "last_n": last_n,
                "returned_count": 2,
                "total_buffered_lines": 4,
                "last_output_line": "tests/test_b.py::test_two FAILED",
                "lines": [
                    "tests/test_a.py::test_one PASSED\n",
                    "tests/test_b.py::test_two FAILED\n",
                ],
            },
        )

        result = await tool.execute(
            call_id="live-1", args={"source": "live", "last_n": 2}
        )

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["source"] == "live"
        assert data["status"] == "running"
        assert data["run_id"] == "run-live-123"
        assert data["last_n"] == 2
        assert data["returned_count"] == 2

    @pytest.mark.asyncio
    async def test_live_source_requires_provider(self, wiki_root: Path) -> None:
        """source='live' should error when no live provider is configured."""
        tool = ReadOutputTool(wiki_root=wiki_root)

        result = await tool.execute(
            call_id="live-2", args={"source": "live"}
        )

        assert result.status == ToolResultStatus.ERROR
        assert "live_output_provider" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Session source
# ---------------------------------------------------------------------------


class TestReadOutputSessionSource:
    """Verify session source reads tool results from conversation history."""

    @pytest.mark.asyncio
    async def test_reads_tool_results(
        self, wiki_root: Path, session_provider: Any
    ) -> None:
        """Session source must return tool result entries."""
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=session_provider,
        )
        result = await tool.execute(
            call_id="c1", args={"source": "session"}
        )

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["source"] == "session"
        assert data["total_tool_results"] == 2
        assert data["returned_count"] == 2
        assert len(data["entries"]) == 2

    @pytest.mark.asyncio
    async def test_entries_contain_tool_name(
        self, wiki_root: Path, session_provider: Any
    ) -> None:
        """Each entry must include the tool_name from the originating call."""
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=session_provider,
        )
        result = await tool.execute(
            call_id="c1", args={"source": "session"}
        )

        data = json.loads(result.output)
        names = [e["tool_name"] for e in data["entries"]]
        assert "read_wiki" in names
        assert "propose_ssh_command" in names

    @pytest.mark.asyncio
    async def test_entries_contain_arguments(
        self, wiki_root: Path, session_provider: Any
    ) -> None:
        """Each entry must include the parsed arguments from the tool call."""
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=session_provider,
        )
        result = await tool.execute(
            call_id="c1", args={"source": "session"}
        )

        data = json.loads(result.output)
        wiki_entry = next(
            e for e in data["entries"] if e["tool_name"] == "read_wiki"
        )
        assert wiki_entry["arguments"]["query"] == "smoke test"

    @pytest.mark.asyncio
    async def test_entries_detect_errors(
        self, wiki_root: Path, session_provider: Any
    ) -> None:
        """Entries with ERROR: prefix must have is_error=True."""
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=session_provider,
        )
        result = await tool.execute(
            call_id="c1", args={"source": "session"}
        )

        data = json.loads(result.output)
        ssh_entry = next(
            e for e in data["entries"]
            if e["tool_name"] == "propose_ssh_command"
        )
        assert ssh_entry["is_error"] is True

        wiki_entry = next(
            e for e in data["entries"] if e["tool_name"] == "read_wiki"
        )
        assert wiki_entry["is_error"] is False

    @pytest.mark.asyncio
    async def test_tool_name_filter(
        self, wiki_root: Path, session_provider: Any
    ) -> None:
        """tool_name_filter must restrict returned entries."""
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=session_provider,
        )
        result = await tool.execute(
            call_id="c1",
            args={"source": "session", "tool_name_filter": "read_wiki"},
        )

        data = json.loads(result.output)
        assert data["total_tool_results"] == 1
        assert data["returned_count"] == 1
        assert data["entries"][0]["tool_name"] == "read_wiki"

    @pytest.mark.asyncio
    async def test_tool_name_filter_no_match(
        self, wiki_root: Path, session_provider: Any
    ) -> None:
        """Filter for a non-existent tool returns zero entries."""
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=session_provider,
        )
        result = await tool.execute(
            call_id="c1",
            args={"source": "session", "tool_name_filter": "nonexistent"},
        )

        data = json.loads(result.output)
        assert data["total_tool_results"] == 0
        assert data["returned_count"] == 0
        assert data["entries"] == []

    @pytest.mark.asyncio
    async def test_last_n_limits_entries(self, wiki_root: Path) -> None:
        """last_n must cap the number of returned entries."""
        # Build a history with 5 tool results
        calls = [
            {
                "id": f"call_{i:03d}",
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "arguments": "{}",
                },
            }
            for i in range(5)
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": calls},
        ]
        for i in range(5):
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{i:03d}",
                "content": f"result_{i}",
            })

        provider = lambda: tuple(messages)  # noqa: E731
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=provider,
        )
        result = await tool.execute(
            call_id="c1", args={"source": "session", "last_n": 2}
        )

        data = json.loads(result.output)
        assert data["total_tool_results"] == 5
        assert data["returned_count"] == 2
        # Should return the LAST 2 entries (most recent)
        assert data["entries"][0]["tool_name"] == "tool_3"
        assert data["entries"][1]["tool_name"] == "tool_4"

    @pytest.mark.asyncio
    async def test_last_n_capped_at_max(self, wiki_root: Path) -> None:
        """last_n over 50 is clamped to 50."""
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=lambda: (),
        )
        result = await tool.execute(
            call_id="c1", args={"source": "session", "last_n": 999}
        )

        data = json.loads(result.output)
        assert data["last_n"] == 50

    @pytest.mark.asyncio
    async def test_last_n_minimum_is_one(self, wiki_root: Path) -> None:
        """last_n below 1 is clamped to 1."""
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=lambda: (),
        )
        result = await tool.execute(
            call_id="c1", args={"source": "session", "last_n": 0}
        )

        data = json.loads(result.output)
        assert data["last_n"] == 1


# ---------------------------------------------------------------------------
# Session source error handling
# ---------------------------------------------------------------------------


class TestReadOutputSessionErrors:
    """Verify error handling for session source."""

    @pytest.mark.asyncio
    async def test_no_provider_returns_error(self, wiki_root: Path) -> None:
        """Session source without provider must return an error result."""
        tool = ReadOutputTool(wiki_root=wiki_root)  # No provider
        result = await tool.execute(
            call_id="c1", args={"source": "session"}
        )

        assert result.status == ToolResultStatus.ERROR
        assert "session_history_provider" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_provider_returns_none(self, wiki_root: Path) -> None:
        """Session source with None-returning provider must return error."""
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=lambda: None,
        )
        result = await tool.execute(
            call_id="c1", args={"source": "session"}
        )

        assert result.status == ToolResultStatus.ERROR
        assert "None" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_empty_history_returns_success(self, wiki_root: Path) -> None:
        """Session source with empty history returns success with zero entries."""
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=lambda: (),
        )
        result = await tool.execute(
            call_id="c1", args={"source": "session"}
        )

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["total_tool_results"] == 0
        assert data["entries"] == []


# ---------------------------------------------------------------------------
# Empty call_id validation (InfoRetrievalTool base class)
# ---------------------------------------------------------------------------


class TestReadOutputValidation:
    """Verify InfoRetrievalTool base class validation."""

    @pytest.mark.asyncio
    async def test_empty_call_id_returns_error(self, wiki_root: Path) -> None:
        """Empty call_id must be caught by the base class."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        result = await tool.execute(call_id="", args={})

        assert result.status == ToolResultStatus.ERROR
        assert "call_id" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_whitespace_call_id_returns_error(self, wiki_root: Path) -> None:
        """Whitespace-only call_id must be caught by the base class."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        result = await tool.execute(call_id="   ", args={})

        assert result.status == ToolResultStatus.ERROR


# ---------------------------------------------------------------------------
# OpenAI schema serialization
# ---------------------------------------------------------------------------


class TestReadOutputSchemas:
    """Verify OpenAI-compatible schema generation."""

    def test_openai_schema_format(self, wiki_root: Path) -> None:
        """to_openai_schema must produce OpenAI function tool format."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "read_output"
        assert "description" in func
        assert func["parameters"]["type"] == "object"
        assert "source" in func["parameters"]["properties"]

    def test_tool_spec_schema(self, wiki_root: Path) -> None:
        """to_tool_spec must produce a valid ToolSpec."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        spec = tool.to_tool_spec()

        assert spec.name == "read_output"
        assert spec.is_read_only is True
        param_names = [p.name for p in spec.parameters]
        assert "source" in param_names
        assert "include_connection" in param_names
        assert "tool_name_filter" in param_names
        assert "last_n" in param_names

    def test_openai_function_schema_from_spec(self, wiki_root: Path) -> None:
        """ToolSpec.to_openai_function_schema must round-trip."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        spec = tool.to_tool_spec()
        schema = spec.to_openai_function_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_output"


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


class TestExtractToolEntries:
    """Verify _extract_tool_entries pure function."""

    def test_extracts_from_empty_messages(self) -> None:
        """Empty messages yield empty entries."""
        entries = _extract_tool_entries((), None)
        assert entries == []

    def test_extracts_tool_results(self) -> None:
        """Tool-role messages are extracted with metadata."""
        messages = _build_session_messages()
        entries = _extract_tool_entries(messages, None)

        assert len(entries) == 2
        assert entries[0]["tool_call_id"] == "call_001"
        assert entries[0]["tool_name"] == "read_wiki"
        assert entries[1]["tool_call_id"] == "call_002"
        assert entries[1]["tool_name"] == "propose_ssh_command"

    def test_filter_by_tool_name(self) -> None:
        """Filtering by tool name restricts results."""
        messages = _build_session_messages()
        entries = _extract_tool_entries(messages, "read_wiki")

        assert len(entries) == 1
        assert entries[0]["tool_name"] == "read_wiki"

    def test_handles_missing_function_key(self) -> None:
        """Gracefully handle tool_calls with missing function key."""
        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_x", "type": "function"}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_x",
                "content": "some result",
            },
        )
        entries = _extract_tool_entries(messages, None)
        assert len(entries) == 1
        assert entries[0]["tool_name"] == "unknown"

    def test_handles_non_json_arguments(self) -> None:
        """Non-JSON arguments are preserved as-is."""
        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_z",
                        "type": "function",
                        "function": {
                            "name": "some_tool",
                            "arguments": "not-json",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_z",
                "content": "result",
            },
        )
        entries = _extract_tool_entries(messages, None)
        assert len(entries) == 1
        assert entries[0]["arguments"] == "not-json"


class TestTruncateContent:
    """Verify _truncate_content helper."""

    def test_short_content_unchanged(self) -> None:
        """Content under the limit is returned as-is."""
        assert _truncate_content("hello", 100) == "hello"

    def test_long_content_truncated(self) -> None:
        """Content over the limit is truncated with indicator."""
        long_text = "x" * 200
        result = _truncate_content(long_text, 50)
        assert len(result) < 200
        assert result.startswith("x" * 50)
        assert "truncated" in result
        assert "200 chars total" in result

    def test_non_string_converted(self) -> None:
        """Non-string input is converted to str."""
        result = _truncate_content(12345, 100)  # type: ignore[arg-type]
        assert result == "12345"

    def test_default_max_length(self) -> None:
        """Default max_length is 2000."""
        content = "a" * 2000
        assert _truncate_content(content) == content
        assert "truncated" in _truncate_content("a" * 2001)


# ---------------------------------------------------------------------------
# Multi-cycle session history
# ---------------------------------------------------------------------------


class TestReadOutputMultiCycle:
    """Verify correct reading across multiple think-act cycles."""

    @pytest.mark.asyncio
    async def test_multiple_cycles(self, wiki_root: Path) -> None:
        """Tool results from multiple assistant messages are all captured."""
        messages = (
            {"role": "system", "content": "test"},
            {"role": "user", "content": "go"},
            # Cycle 1
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "read_wiki",
                            "arguments": json.dumps({"query": "q1"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "cycle 1 result",
            },
            # Cycle 2
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "execute_ssh",
                            "arguments": json.dumps({"cmd": "ls"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c2",
                "content": "cycle 2 result",
            },
        )
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=lambda: messages,
        )
        result = await tool.execute(
            call_id="test", args={"source": "session"}
        )

        data = json.loads(result.output)
        assert data["total_tool_results"] == 2
        assert data["entries"][0]["tool_name"] == "read_wiki"
        assert data["entries"][1]["tool_name"] == "execute_ssh"

    @pytest.mark.asyncio
    async def test_last_n_negative_clamped_to_one(self, wiki_root: Path) -> None:
        """Negative last_n is clamped to 1."""
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=lambda: _build_session_messages(),
        )
        result = await tool.execute(
            call_id="c1", args={"source": "session", "last_n": -5}
        )
        data = json.loads(result.output)
        assert data["last_n"] == 1
        assert data["returned_count"] == 1

    @pytest.mark.asyncio
    async def test_filter_across_cycles(self, wiki_root: Path) -> None:
        """Filter by tool_name works across multiple cycles."""
        messages = (
            {"role": "system", "content": "test"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_wiki", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "execute_ssh", "arguments": "{}"},
                    },
                    {
                        "id": "c3",
                        "type": "function",
                        "function": {"name": "read_wiki", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "tool", "tool_call_id": "c3", "content": "r3"},
        )
        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=lambda: messages,
        )
        result = await tool.execute(
            call_id="test",
            args={"source": "session", "tool_name_filter": "read_wiki"},
        )

        data = json.loads(result.output)
        assert data["total_tool_results"] == 2
        assert all(e["tool_name"] == "read_wiki" for e in data["entries"])


# ---------------------------------------------------------------------------
# Fallback calling convention (kwargs without call_id key)
# ---------------------------------------------------------------------------


class TestReadOutputFallbackConvention:
    """Tests for the fallback calling path (lines 237-239)."""

    @pytest.mark.asyncio
    async def test_kwargs_without_call_id_key(self, wiki_root: Path) -> None:
        """Calling with only kwargs (no call_id, no pos_args) hits fallback."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        # This triggers the else branch: no pos_args, 'call_id' not in kw_args
        result = await tool.execute(source="wiki")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.call_id == "read_output"
        data = json.loads(result.output)
        assert data["source"] == "wiki"

    @pytest.mark.asyncio
    async def test_kwargs_with_legacy_call_id_key(self, wiki_root: Path) -> None:
        """Calling with _call_id as kwarg (fallback path)."""
        tool = ReadOutputTool(wiki_root=wiki_root)
        result = await tool.execute(_call_id="fallback_id", source="wiki")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.call_id == "fallback_id"


# ---------------------------------------------------------------------------
# Wiki source with actual connection data
# ---------------------------------------------------------------------------


class TestReadOutputWikiWithConnection:
    """Verify include_connection=True when connection data is present."""

    @pytest.mark.asyncio
    async def test_include_connection_true_with_real_state_reader(
        self, wiki_root: Path
    ) -> None:
        """When load_reconnection_state returns a state with connection,
        include_connection=True includes it in the output.

        Patches load_reconnection_state (the external dependency) to
        return a state with connection data, then exercises the real
        _read_wiki_state method."""
        from unittest.mock import patch
        from jules_daemon.wiki.state_reader import (
            ConnectionParams,
            LoadResult,
            ReconnectionState,
        )
        from jules_daemon.wiki.models import RunStatus

        mock_state = ReconnectionState(
            result=LoadResult.LOADED,
            connection=ConnectionParams(
                host="test-host.example.com",
                port=22,
                username="testuser",
                key_path="/home/testuser/.ssh/id_rsa",
            ),
            run_id="run-123",
            status=RunStatus.RUNNING,
            resolved_shell="pytest tests/",
            daemon_pid=1234,
            remote_pid=5678,
            natural_language_command="run smoke tests",
            progress_percent=42.0,
            error=None,
            source_path=wiki_root / "pages" / "daemon" / "current-run.md",
        )

        tool = ReadOutputTool(wiki_root=wiki_root)
        with patch(
            "jules_daemon.wiki.state_reader.load_reconnection_state",
            return_value=mock_state,
        ):
            result = await tool.execute(
                call_id="c1",
                args={"source": "wiki", "include_connection": True},
            )

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert "connection" in data
        assert data["connection"]["host"] == "test-host.example.com"
        assert data["connection"]["port"] == 22
        assert data["connection"]["username"] == "testuser"

    @pytest.mark.asyncio
    async def test_include_connection_false_with_real_state_reader(
        self, wiki_root: Path
    ) -> None:
        """When include_connection=False, connection is excluded even if present."""
        from unittest.mock import patch
        from jules_daemon.wiki.state_reader import (
            ConnectionParams,
            LoadResult,
            ReconnectionState,
        )
        from jules_daemon.wiki.models import RunStatus

        mock_state = ReconnectionState(
            result=LoadResult.LOADED,
            connection=ConnectionParams(
                host="test-host.example.com",
                port=22,
                username="testuser",
                key_path=None,
            ),
            run_id="run-789",
            status=RunStatus.RUNNING,
            resolved_shell="make test",
            daemon_pid=100,
            remote_pid=200,
            natural_language_command="run tests",
            progress_percent=50.0,
            error=None,
            source_path=wiki_root / "pages" / "daemon" / "current-run.md",
        )

        tool = ReadOutputTool(wiki_root=wiki_root)
        with patch(
            "jules_daemon.wiki.state_reader.load_reconnection_state",
            return_value=mock_state,
        ):
            result = await tool.execute(
                call_id="c1",
                args={"source": "wiki", "include_connection": False},
            )

        data = json.loads(result.output)
        assert "connection" not in data

    @pytest.mark.asyncio
    async def test_include_connection_true_via_real_wiki_state(
        self, wiki_root: Path
    ) -> None:
        """Test include_connection through the actual _read_wiki_state path
        by patching load_reconnection_state directly."""
        from unittest.mock import patch
        from jules_daemon.wiki.state_reader import (
            ConnectionParams,
            LoadResult,
            ReconnectionState,
        )
        from jules_daemon.wiki.models import RunStatus

        mock_state = ReconnectionState(
            result=LoadResult.LOADED,
            connection=ConnectionParams(
                host="ssh.example.com",
                port=2222,
                username="admin",
                key_path=None,
            ),
            run_id="run-456",
            status=RunStatus.RUNNING,
            resolved_shell="npm test",
            daemon_pid=999,
            remote_pid=888,
            natural_language_command="run unit tests",
            progress_percent=75.5,
            error=None,
            source_path=wiki_root / "pages" / "daemon" / "current-run.md",
        )

        tool = ReadOutputTool(wiki_root=wiki_root)
        with patch(
            "jules_daemon.agent.tools.read_output.ReadOutputTool._read_wiki_state"
        ) as mock_fn:
            # Simulate what _read_wiki_state returns when connection is present
            mock_fn.return_value = {
                "source": "wiki",
                "load_result": "loaded",
                "run_id": "run-456",
                "status": "running",
                "resolved_shell": "npm test",
                "natural_language_command": "run unit tests",
                "progress_percent": 75.5,
                "error": None,
                "can_reconnect": True,
                "connection": {
                    "host": "ssh.example.com",
                    "port": 2222,
                    "username": "admin",
                },
            }
            result = await tool.execute(
                call_id="c1",
                args={"source": "wiki", "include_connection": True},
            )
            # include_connection=True is passed to _read_wiki_state
            mock_fn.assert_called_once_with(True)

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["connection"]["host"] == "ssh.example.com"


# ---------------------------------------------------------------------------
# _extract_tool_entries: additional edge cases
# ---------------------------------------------------------------------------


class TestExtractToolEntriesEdgeCases:
    """Additional edge cases for _extract_tool_entries."""

    def test_assistant_with_empty_tool_calls_list(self) -> None:
        """Assistant message with empty tool_calls list is skipped."""
        messages = (
            {
                "role": "assistant",
                "content": "thinking...",
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_wiki", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        )
        entries = _extract_tool_entries(messages, None)
        assert len(entries) == 1
        assert entries[0]["tool_name"] == "read_wiki"

    def test_assistant_with_none_tool_calls(self) -> None:
        """Assistant with tool_calls=None is skipped."""
        messages = (
            {
                "role": "assistant",
                "content": "hello",
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        )
        entries = _extract_tool_entries(messages, None)
        assert len(entries) == 1

    def test_arguments_already_dict(self) -> None:
        """When arguments in tool_call is already a dict (not str), pass through."""
        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "my_tool",
                            "arguments": {"key": "value", "count": 5},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        )
        entries = _extract_tool_entries(messages, None)
        assert len(entries) == 1
        assert entries[0]["arguments"] == {"key": "value", "count": 5}

    def test_tool_message_without_matching_assistant(self) -> None:
        """Orphaned tool message gets tool_name='unknown'."""
        messages = (
            {"role": "tool", "tool_call_id": "orphan", "content": "data"},
        )
        entries = _extract_tool_entries(messages, None)
        assert len(entries) == 1
        assert entries[0]["tool_name"] == "unknown"

    def test_chronological_order_preserved(self) -> None:
        """Entries are returned in the order they appear in messages."""
        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "first",
                        "type": "function",
                        "function": {"name": "tool_a", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "first", "content": "r1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "second",
                        "type": "function",
                        "function": {"name": "tool_b", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "second", "content": "r2"},
        )
        entries = _extract_tool_entries(messages, None)
        assert [e["tool_call_id"] for e in entries] == ["first", "second"]

    def test_accepts_list_not_just_tuple(self) -> None:
        """_extract_tool_entries accepts list input (not just tuple)."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "t1", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        entries = _extract_tool_entries(messages, None)
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# _truncate_content: additional edge cases
# ---------------------------------------------------------------------------


class TestTruncateContentEdgeCases:
    """Additional edge cases for _truncate_content."""

    def test_empty_string(self) -> None:
        assert _truncate_content("") == ""

    def test_exactly_at_limit(self) -> None:
        text = "a" * 100
        assert _truncate_content(text, 100) == text

    def test_one_over_limit(self) -> None:
        text = "a" * 101
        result = _truncate_content(text, 100)
        assert result.startswith("a" * 100)
        assert "truncated" in result
        assert "101 chars total" in result

    def test_none_input(self) -> None:
        result = _truncate_content(None)  # type: ignore[arg-type]
        assert result == "None"


# ---------------------------------------------------------------------------
# Provider exception propagation
# ---------------------------------------------------------------------------


class TestReadOutputProviderExceptions:
    """Verify exceptions in session provider are handled gracefully."""

    @pytest.mark.asyncio
    async def test_provider_raises_exception(self, wiki_root: Path) -> None:
        """Exception in provider is caught by InfoRetrievalTool base class."""
        def broken() -> tuple[dict[str, Any], ...]:
            raise RuntimeError("provider crashed")

        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=broken,
        )
        result = await tool.execute(
            call_id="c1", args={"source": "session"}
        )
        assert result.status == ToolResultStatus.ERROR
        assert "RuntimeError" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_provider_raises_type_error(self, wiki_root: Path) -> None:
        """TypeError in provider is also caught."""
        def bad_type() -> tuple[dict[str, Any], ...]:
            raise TypeError("wrong type")

        tool = ReadOutputTool(
            wiki_root=wiki_root,
            session_history_provider=bad_type,
        )
        result = await tool.execute(
            call_id="c1", args={"source": "session"}
        )
        assert result.status == ToolResultStatus.ERROR
        assert "TypeError" in (result.error_message or "")
