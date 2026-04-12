"""Coverage-focused tests for ReadOutputTool.

Targets the specific uncovered lines in jules_daemon.agent.tools.read_output:
    - Lines 225-226: execute() keyword call_id path
    - Lines 232-239: execute() legacy dict, positional str, fallback conventions
    - Line 263: _execute_impl session source dispatch
    - Line 300: _read_wiki_state with include_connection=True
    - Lines 322-360: _read_session -- no provider, provider returns None, normal
    - Lines 391-445: _extract_tool_entries -- call_id_to_info, filter, JSON parse
    - Lines 459-463: _truncate_content -- non-string, within-limit, over-limit
"""

from __future__ import annotations

import json
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from jules_daemon.agent.tools.read_output import (
    ReadOutputTool,
    _extract_tool_entries,
    _truncate_content,
)
from jules_daemon.agent.tool_types import ToolResult, ToolResultStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(
    messages: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None,
):
    """Return a callable that yields the given messages (or None)."""
    return lambda: tuple(messages) if messages is not None else None


def _build_messages_with_tool_results(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Build a minimal conversation containing assistant tool_calls + tool results."""
    return (
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        *tool_results,
    )


# ---------------------------------------------------------------------------
# Fake wiki state objects used with the mock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeConnection:
    host: str = "10.0.0.1"
    port: int = 22
    username: str = "deploy"


@dataclass(frozen=True)
class _FakeState:
    result: MagicMock = None  # type: ignore[assignment]
    run_id: str = "run-42"
    status: MagicMock = None  # type: ignore[assignment]
    resolved_shell: str = "pytest -x"
    natural_language_command: str = "run tests"
    progress_percent: float = 75.0
    error: Optional[str] = None
    can_reconnect: bool = True
    connection: Optional[_FakeConnection] = None

    @staticmethod
    def loaded_with_connection() -> "_FakeState":
        result_mock = MagicMock()
        result_mock.value = "loaded"
        status_mock = MagicMock()
        status_mock.value = "running"
        return _FakeState(
            result=result_mock,
            status=status_mock,
            connection=_FakeConnection(),
        )

    @staticmethod
    def loaded_without_connection() -> "_FakeState":
        result_mock = MagicMock()
        result_mock.value = "loaded"
        status_mock = MagicMock()
        status_mock.value = "idle"
        return _FakeState(
            result=result_mock,
            status=status_mock,
            connection=None,
        )


# ---------------------------------------------------------------------------
# Tests: execute() calling conventions (lines 225-226, 232-239)
# ---------------------------------------------------------------------------


class TestExecuteCallingConventions:
    """Cover every branch in execute() argument dispatch."""

    @pytest.mark.asyncio
    async def test_keyword_call_id_with_args(self) -> None:
        """Lines 225-226: keyword call_id + args dict."""
        tool = ReadOutputTool(
            wiki_root=Path("/tmp/test-wiki"),
            session_history_provider=_make_provider(()),
        )
        result = await tool.execute(
            call_id="kw-1",
            args={"source": "session"},
        )
        assert result.is_success
        data = json.loads(result.output)
        assert data["source"] == "session"

    @pytest.mark.asyncio
    async def test_keyword_call_id_without_args(self) -> None:
        """Lines 225-226: keyword call_id with None args (defaults to {})."""
        with patch(
            "jules_daemon.agent.tools.read_output.ReadOutputTool._read_wiki_state"
        ) as mock_ws:
            mock_ws.return_value = {"source": "wiki", "ok": True}
            tool = ReadOutputTool(wiki_root=Path("/tmp/test-wiki"))
            result = await tool.execute(call_id="kw-2", args=None)  # type: ignore[arg-type]
            assert result.is_success

    @pytest.mark.asyncio
    async def test_legacy_dict_convention(self) -> None:
        """Lines 228-231: first positional arg is a dict (legacy BaseTool)."""
        tool = ReadOutputTool(
            wiki_root=Path("/tmp/test-wiki"),
            session_history_provider=_make_provider(()),
        )
        result = await tool.execute({"source": "session", "_call_id": "leg-1"})
        assert result.is_success
        data = json.loads(result.output)
        assert data["source"] == "session"

    @pytest.mark.asyncio
    async def test_legacy_dict_convention_default_call_id(self) -> None:
        """Lines 228-231: legacy dict without _call_id uses default."""
        tool = ReadOutputTool(
            wiki_root=Path("/tmp/test-wiki"),
            session_history_provider=_make_provider(()),
        )
        result = await tool.execute({"source": "session"})
        assert result.is_success
        assert result.call_id == "read_output"

    @pytest.mark.asyncio
    async def test_positional_string_convention(self) -> None:
        """Lines 232-235: first positional arg is a string call_id."""
        tool = ReadOutputTool(
            wiki_root=Path("/tmp/test-wiki"),
            session_history_provider=_make_provider(()),
        )
        result = await tool.execute("pos-1", {"source": "session"})
        assert result.is_success
        data = json.loads(result.output)
        assert data["source"] == "session"

    @pytest.mark.asyncio
    async def test_positional_string_no_second_arg(self) -> None:
        """Lines 232-235: positional call_id string with no second arg defaults to {}."""
        with patch(
            "jules_daemon.agent.tools.read_output.ReadOutputTool._read_wiki_state"
        ) as mock_ws:
            mock_ws.return_value = {"source": "wiki", "ok": True}
            tool = ReadOutputTool(wiki_root=Path("/tmp/test-wiki"))
            result = await tool.execute("pos-2")
            assert result.is_success

    @pytest.mark.asyncio
    async def test_fallback_convention(self) -> None:
        """Lines 237-239: no positional args, no call_id keyword -- fallback."""
        tool = ReadOutputTool(
            wiki_root=Path("/tmp/test-wiki"),
            session_history_provider=_make_provider(()),
        )
        result = await tool.execute(source="session", _call_id="fb-1")
        assert result.is_success
        data = json.loads(result.output)
        assert data["source"] == "session"

    @pytest.mark.asyncio
    async def test_fallback_convention_default_call_id(self) -> None:
        """Lines 237-239: fallback without _call_id uses default 'read_output'."""
        tool = ReadOutputTool(
            wiki_root=Path("/tmp/test-wiki"),
            session_history_provider=_make_provider(()),
        )
        result = await tool.execute(source="session")
        assert result.is_success
        assert result.call_id == "read_output"


# ---------------------------------------------------------------------------
# Tests: _execute_impl session dispatch (line 263)
# ---------------------------------------------------------------------------


class TestExecuteImplSessionDispatch:
    """Cover source='session' branch in _execute_impl (line 263)."""

    @pytest.mark.asyncio
    async def test_session_dispatch(self) -> None:
        """Line 263: source='session' dispatches to _read_session."""
        messages = _build_messages_with_tool_results(
            tool_calls=[
                {
                    "id": "tc-1",
                    "type": "function",
                    "function": {"name": "execute_ssh", "arguments": '{"cmd": "ls"}'},
                },
            ],
            tool_results=[
                {"role": "tool", "tool_call_id": "tc-1", "content": "file.txt"},
            ],
        )
        tool = ReadOutputTool(
            wiki_root=Path("/tmp/test-wiki"),
            session_history_provider=_make_provider(messages),
        )
        result = await tool.execute(call_id="sd-1", args={"source": "session"})

        assert result.is_success
        data = json.loads(result.output)
        assert data["source"] == "session"
        assert data["total_tool_results"] == 1
        assert data["entries"][0]["tool_name"] == "execute_ssh"


# ---------------------------------------------------------------------------
# Tests: _read_wiki_state with connection (line 300)
# ---------------------------------------------------------------------------


class TestReadWikiStateWithConnection:
    """Cover include_connection branch (line 300)."""

    @pytest.mark.asyncio
    async def test_include_connection_true(self) -> None:
        """Line 300: when include_connection=True and state has connection."""
        fake = _FakeState.loaded_with_connection()
        with patch(
            "jules_daemon.wiki.state_reader.load_reconnection_state",
            return_value=fake,
        ):
            tool = ReadOutputTool(wiki_root=Path("/tmp/test-wiki"))
            result = await tool.execute(
                call_id="conn-1",
                args={"source": "wiki", "include_connection": True},
            )

            assert result.is_success
            data = json.loads(result.output)
            assert "connection" in data
            assert data["connection"]["host"] == "10.0.0.1"
            assert data["connection"]["port"] == 22
            assert data["connection"]["username"] == "deploy"

    @pytest.mark.asyncio
    async def test_include_connection_true_but_no_connection(self) -> None:
        """When include_connection=True but state.connection is None, no key."""
        fake = _FakeState.loaded_without_connection()
        with patch(
            "jules_daemon.wiki.state_reader.load_reconnection_state",
            return_value=fake,
        ):
            tool = ReadOutputTool(wiki_root=Path("/tmp/test-wiki"))
            result = await tool.execute(
                call_id="conn-2",
                args={"source": "wiki", "include_connection": True},
            )

            assert result.is_success
            data = json.loads(result.output)
            assert "connection" not in data


# ---------------------------------------------------------------------------
# Tests: _read_session (lines 322-360)
# ---------------------------------------------------------------------------


class TestReadSession:
    """Cover _read_session method including error paths and normal flow."""

    @pytest.mark.asyncio
    async def test_no_provider_error(self) -> None:
        """Lines 322-331: no session_history_provider returns error."""
        tool = ReadOutputTool(wiki_root=Path("/tmp/test-wiki"))
        result = await tool.execute(
            call_id="np-1", args={"source": "session"}
        )

        assert result.status == ToolResultStatus.ERROR
        assert "session_history_provider" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_provider_returns_none_error(self) -> None:
        """Lines 334-339: provider returns None."""
        tool = ReadOutputTool(
            wiki_root=Path("/tmp/test-wiki"),
            session_history_provider=lambda: None,
        )
        result = await tool.execute(
            call_id="none-1", args={"source": "session"}
        )

        assert result.status == ToolResultStatus.ERROR
        assert "None" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_normal_session_read_with_filter(self) -> None:
        """Lines 341-364: normal flow with tool_name_filter and last_n."""
        messages = _build_messages_with_tool_results(
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_wiki", "arguments": "{}"},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "execute_ssh", "arguments": '{"cmd": "ls"}'},
                },
                {
                    "id": "c3",
                    "type": "function",
                    "function": {"name": "execute_ssh", "arguments": '{"cmd": "pwd"}'},
                },
            ],
            tool_results=[
                {"role": "tool", "tool_call_id": "c1", "content": "wiki data"},
                {"role": "tool", "tool_call_id": "c2", "content": "output ls"},
                {"role": "tool", "tool_call_id": "c3", "content": "output pwd"},
            ],
        )
        tool = ReadOutputTool(
            wiki_root=Path("/tmp/test-wiki"),
            session_history_provider=_make_provider(messages),
        )

        # Filter by execute_ssh, last_n=1 -> only most recent execute_ssh entry
        result = await tool.execute(
            call_id="f-1",
            args={
                "source": "session",
                "tool_name_filter": "execute_ssh",
                "last_n": 1,
            },
        )

        assert result.is_success
        data = json.loads(result.output)
        assert data["source"] == "session"
        assert data["total_tool_results"] == 2  # 2 execute_ssh entries
        assert data["returned_count"] == 1  # capped by last_n
        assert data["last_n"] == 1
        assert data["tool_name_filter"] == "execute_ssh"
        assert data["entries"][0]["tool_name"] == "execute_ssh"
        assert data["entries"][0]["content"] == "output pwd"

    @pytest.mark.asyncio
    async def test_session_read_all_entries_within_last_n(self) -> None:
        """Lines 349: when total entries <= last_n, return all."""
        messages = _build_messages_with_tool_results(
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "some_tool", "arguments": "{}"},
                },
            ],
            tool_results=[
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            ],
        )
        tool = ReadOutputTool(
            wiki_root=Path("/tmp/test-wiki"),
            session_history_provider=_make_provider(messages),
        )
        result = await tool.execute(
            call_id="all-1",
            args={"source": "session", "last_n": 10},
        )

        data = json.loads(result.output)
        assert data["total_tool_results"] == 1
        assert data["returned_count"] == 1

    @pytest.mark.asyncio
    async def test_session_read_result_data_shape(self) -> None:
        """Lines 351-358: verify result_data contains all expected keys."""
        tool = ReadOutputTool(
            wiki_root=Path("/tmp/test-wiki"),
            session_history_provider=_make_provider(()),
        )
        result = await tool.execute(
            call_id="shape-1",
            args={"source": "session"},
        )

        data = json.loads(result.output)
        expected_keys = {
            "source",
            "total_tool_results",
            "returned_count",
            "last_n",
            "tool_name_filter",
            "entries",
        }
        assert set(data.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: _extract_tool_entries (lines 391-445)
# ---------------------------------------------------------------------------


class TestExtractToolEntriesCoverage:
    """Cover _extract_tool_entries including call_id_to_info build, filter, JSON parse."""

    def test_builds_call_id_to_info_lookup(self) -> None:
        """Lines 391-404: assistant messages with tool_calls build the lookup."""
        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-a",
                        "type": "function",
                        "function": {
                            "name": "execute_ssh",
                            "arguments": json.dumps({"cmd": "whoami"}),
                        },
                    },
                    {
                        "id": "tc-b",
                        "type": "function",
                        "function": {
                            "name": "read_wiki",
                            "arguments": json.dumps({"slug": "faq"}),
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc-a", "content": "root"},
            {"role": "tool", "tool_call_id": "tc-b", "content": "FAQ page"},
        )

        entries = _extract_tool_entries(messages, None)
        assert len(entries) == 2
        assert entries[0]["tool_name"] == "execute_ssh"
        assert entries[0]["arguments"] == {"cmd": "whoami"}
        assert entries[1]["tool_name"] == "read_wiki"
        assert entries[1]["arguments"] == {"slug": "faq"}

    def test_filter_excludes_non_matching_tools(self) -> None:
        """Lines 420-421: tool_name_filter excludes non-matching entries."""
        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-1",
                        "function": {"name": "tool_a", "arguments": "{}"},
                    },
                    {
                        "id": "tc-2",
                        "function": {"name": "tool_b", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc-1", "content": "a-result"},
            {"role": "tool", "tool_call_id": "tc-2", "content": "b-result"},
        )

        entries = _extract_tool_entries(messages, "tool_a")
        assert len(entries) == 1
        assert entries[0]["tool_name"] == "tool_a"

    def test_json_parse_valid_arguments(self) -> None:
        """Lines 426-427: valid JSON arguments are parsed into a dict."""
        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-j",
                        "function": {
                            "name": "tool_j",
                            "arguments": '{"key": "value", "num": 42}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc-j", "content": "ok"},
        )

        entries = _extract_tool_entries(messages, None)
        assert entries[0]["arguments"] == {"key": "value", "num": 42}

    def test_json_parse_invalid_arguments_preserved(self) -> None:
        """Lines 428-429: invalid JSON arguments are preserved as raw string."""
        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-bad",
                        "function": {
                            "name": "tool_bad",
                            "arguments": "not valid json {{{",
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc-bad", "content": "ok"},
        )

        entries = _extract_tool_entries(messages, None)
        assert entries[0]["arguments"] == "not valid json {{{"

    def test_non_string_arguments_passthrough(self) -> None:
        """Lines 430-431: non-string arguments (e.g. already a dict) pass through."""
        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-dict",
                        "function": {
                            "name": "tool_dict",
                            "arguments": {"already": "parsed"},
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc-dict", "content": "ok"},
        )

        entries = _extract_tool_entries(messages, None)
        assert entries[0]["arguments"] == {"already": "parsed"}

    def test_error_detection_in_content(self) -> None:
        """Lines 434: entries with ERROR: prefix set is_error=True."""
        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-err",
                        "function": {"name": "tool_err", "arguments": "{}"},
                    },
                    {
                        "id": "tc-ok",
                        "function": {"name": "tool_ok", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tc-err",
                "content": "ERROR: something failed",
            },
            {
                "role": "tool",
                "tool_call_id": "tc-ok",
                "content": "all good",
            },
        )

        entries = _extract_tool_entries(messages, None)
        assert entries[0]["is_error"] is True
        assert entries[1]["is_error"] is False

    def test_entry_structure_complete(self) -> None:
        """Lines 436-443: each entry has all expected keys."""
        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-s",
                        "function": {"name": "some_tool", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc-s", "content": "result here"},
        )

        entries = _extract_tool_entries(messages, None)
        assert len(entries) == 1
        entry = entries[0]
        assert set(entry.keys()) == {
            "tool_call_id",
            "tool_name",
            "arguments",
            "is_error",
            "content",
        }
        assert entry["tool_call_id"] == "tc-s"
        assert entry["tool_name"] == "some_tool"

    def test_unknown_tool_call_id_yields_unknown_name(self) -> None:
        """Lines 416-417: tool message with no matching assistant call_id."""
        messages = (
            {"role": "assistant", "content": "no tool_calls here"},
            {
                "role": "tool",
                "tool_call_id": "orphan-id",
                "content": "orphan result",
            },
        )

        entries = _extract_tool_entries(messages, None)
        assert len(entries) == 1
        assert entries[0]["tool_name"] == "unknown"

    def test_skips_non_tool_and_non_assistant_messages(self) -> None:
        """Lines 393-394, 409: non-assistant / non-tool messages are skipped."""
        messages = (
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "user input"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-x",
                        "function": {"name": "tool_x", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc-x", "content": "result"},
        )

        entries = _extract_tool_entries(messages, None)
        assert len(entries) == 1

    def test_assistant_without_tool_calls_skipped(self) -> None:
        """Lines 395-396: assistant message with no tool_calls is skipped."""
        messages = (
            {"role": "assistant", "content": "just text, no tool_calls"},
            {"role": "assistant", "content": None, "tool_calls": None},
            {"role": "assistant", "content": None, "tool_calls": []},
        )

        entries = _extract_tool_entries(messages, None)
        assert entries == []

    def test_content_truncation_in_entries(self) -> None:
        """Lines 441: long content is truncated via _truncate_content."""
        long_content = "x" * 5000
        messages = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc-long",
                        "function": {"name": "tool_long", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc-long", "content": long_content},
        )

        entries = _extract_tool_entries(messages, None)
        assert len(entries[0]["content"]) < 5000
        assert "truncated" in entries[0]["content"]


# ---------------------------------------------------------------------------
# Tests: _truncate_content (lines 459-463)
# ---------------------------------------------------------------------------


class TestTruncateContentCoverage:
    """Cover all branches in _truncate_content."""

    def test_non_string_input_converted(self) -> None:
        """Lines 459-460: non-string input is converted via str()."""
        assert _truncate_content(12345) == "12345"  # type: ignore[arg-type]
        assert _truncate_content(None) == "None"  # type: ignore[arg-type]
        assert _truncate_content(["a", "b"]) == "['a', 'b']"  # type: ignore[arg-type]

    def test_within_limit_unchanged(self) -> None:
        """Lines 461-462: content within max_length is returned unchanged."""
        short = "hello world"
        assert _truncate_content(short, max_length=100) == short
        assert _truncate_content(short, max_length=len(short)) == short

    def test_exactly_at_limit_unchanged(self) -> None:
        """Edge case: content length equals max_length."""
        text = "a" * 50
        assert _truncate_content(text, max_length=50) == text

    def test_over_limit_truncated(self) -> None:
        """Lines 462-463: content exceeding max_length is truncated."""
        text = "a" * 100
        result = _truncate_content(text, max_length=30)
        assert result.startswith("a" * 30)
        assert "truncated" in result
        assert "100 chars total" in result

    def test_default_max_length_is_2000(self) -> None:
        """Default max_length parameter is 2000."""
        text = "z" * 2001
        result = _truncate_content(text)
        assert "truncated" in result
        assert "2001 chars total" in result

        text_under = "z" * 2000
        result_under = _truncate_content(text_under)
        assert result_under == text_under

    def test_empty_string_unchanged(self) -> None:
        """Empty string is within any limit."""
        assert _truncate_content("", max_length=10) == ""
