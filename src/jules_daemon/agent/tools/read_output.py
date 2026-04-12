"""read_output tool -- reads prior command/tool output from the session.

Provides two complementary views of output data:

    source="wiki"    -- Reads the current run state from the wiki persistence
                        layer (status, resolved_shell, progress, error).
                        Delegates to wiki.state_reader.load_reconnection_state.

    source="session" -- Reads prior tool call results from the agent loop
                        conversation history. Enables the LLM to review what
                        happened in earlier think-act cycles and self-correct.

Extends InfoRetrievalTool (the base class for read-only tools) which
provides:
    - Automatic argument validation against parameters_schema
    - Exception-safe execution wrapping
    - OpenAI-compatible schema serialization (to_openai_schema)
    - ToolSpec conversion (to_tool_spec) for ToolRegistry integration

Delegates to:
    - jules_daemon.wiki.state_reader.load_reconnection_state (wiki source)
    - Session history callback (session source)

No business logic is reimplemented -- this tool composes existing
functions and formats their output for the LLM conversation.

Usage::

    from jules_daemon.agent.tools.read_output import ReadOutputTool

    # Wiki-only mode (no session history provider)
    tool = ReadOutputTool(wiki_root=Path("/data/wiki"))

    # With session history provider for agent loop integration
    tool = ReadOutputTool(
        wiki_root=Path("/data/wiki"),
        session_history_provider=lambda: conversation_history,
    )

    # InfoRetrievalTool calling convention
    result = await tool.execute(call_id="c1", args={"source": "wiki"})

    # Legacy BaseTool calling convention (backward compat with ToolRegistry)
    result = await tool.execute({"source": "session", "_call_id": "c1"})
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

from jules_daemon.agent.tool_base import InfoRetrievalTool
from jules_daemon.agent.tool_result import ToolResult
from jules_daemon.agent.tool_types import ToolSpec

__all__ = ["ReadOutputTool"]

logger = logging.getLogger(__name__)

_DEFAULT_LAST_N = 10
"""Default number of recent session entries to return."""

_MAX_LAST_N = 50
"""Hard cap on session entries to prevent unbounded output."""

# Type alias for the session history provider callback.
# Returns the current conversation history messages as a tuple of dicts.
SessionHistoryProvider = Callable[[], tuple[dict[str, Any], ...] | None]


class ReadOutputTool(InfoRetrievalTool):
    """Read prior command/tool output from the wiki or session history.

    Wraps:
        - wiki.state_reader.load_reconnection_state (wiki source)
        - Session history callback (session source)

    This is a read-only tool (ApprovalRequirement.NONE) that extends
    InfoRetrievalTool. Inherits argument validation, error handling,
    and OpenAI schema serialization from the base class.

    The tool supports two output sources:

    ``source="wiki"`` (default):
        Reads the current run state from the wiki file. Returns run
        status, resolved shell command, progress, error messages, and
        optionally SSH connection parameters.

    ``source="session"``:
        Reads recent tool call results from the agent loop conversation
        history. Returns the N most recent tool result entries, optionally
        filtered by tool name. This allows the LLM to review earlier
        cycle outputs for self-correction.
    """

    def __init__(
        self,
        *,
        wiki_root: Path,
        session_history_provider: SessionHistoryProvider | None = None,
    ) -> None:
        self._wiki_root = wiki_root
        self._session_history_provider = session_history_provider
        self._spec_cache: ToolSpec | None = None

    # -- Protocol-required properties (InfoRetrievalTool) ------------------

    @property
    def name(self) -> str:
        """Unique tool identifier (function name in OpenAI API)."""
        return "read_output"

    @property
    def description(self) -> str:
        """Human-readable description shown to the LLM."""
        return (
            "Read prior command or tool output from the current session. "
            "Use source='wiki' (default) to read the current run's status, "
            "resolved shell command, progress, and error messages from the "
            "wiki. Use source='session' to review recent tool call results "
            "from earlier think-act cycles in this agent loop session, "
            "which is useful for observing past failures and self-correcting."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema dict describing accepted arguments.

        Parameters:
            source (string, optional): Output source -- 'wiki' for current
                run state, 'session' for recent tool results. Default: 'wiki'.
            include_connection (boolean, optional): Include SSH connection
                params in wiki output. Default: false. Ignored for session.
            tool_name_filter (string, optional): When source is 'session',
                filter results to only this tool name. Ignored for wiki.
            last_n (integer, optional): When source is 'session', return
                the N most recent tool results. Default: 10. Max: 50.
                Ignored for wiki.
        """
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": (
                        "Output source: 'wiki' for current run state, "
                        "'session' for recent tool call results from this "
                        "agent loop session"
                    ),
                    "enum": ["wiki", "session"],
                    "default": "wiki",
                },
                "include_connection": {
                    "type": "boolean",
                    "description": (
                        "Whether to include SSH connection parameters "
                        "in the wiki output (ignored for session source)"
                    ),
                    "default": False,
                },
                "tool_name_filter": {
                    "type": "string",
                    "description": (
                        "When source is 'session', filter results to only "
                        "this tool name (e.g., 'execute_ssh', 'propose_ssh_command'). "
                        "Ignored for wiki source"
                    ),
                },
                "last_n": {
                    "type": "integer",
                    "description": (
                        "When source is 'session', return the N most recent "
                        "tool results. Default: 10, max: 50. Ignored for wiki"
                    ),
                    "default": _DEFAULT_LAST_N,
                },
            },
            "required": [],
        }

    # -- Backward-compatible spec property for ToolRegistry ----------------

    @property
    def spec(self) -> ToolSpec:
        """Return a ToolSpec for ToolRegistry integration.

        Cached on first access. Equivalent to calling to_tool_spec()
        but avoids reconstructing the ToolSpec on every access.
        """
        if self._spec_cache is None:
            self._spec_cache = self.to_tool_spec()
        return self._spec_cache

    # -- Dual calling convention for execute -------------------------------

    async def execute(  # type: ignore[override]
        self, *pos_args: Any, **kw_args: Any
    ) -> ToolResult:
        """Execute with support for both calling conventions.

        InfoRetrievalTool convention::

            result = await tool.execute(call_id="c1", args={"source": "wiki"})
            result = await tool.execute("c1", {"source": "wiki"})

        Legacy BaseTool convention (used by ToolRegistry)::

            result = await tool.execute({"source": "wiki", "_call_id": "c1"})

        When called with a dict as the first positional argument, the
        ``_call_id`` key is extracted and the rest is used as args.

        Returns:
            ToolResult -- never raises.
        """
        # Detect calling convention
        call_id: str
        args: dict[str, Any]

        if "call_id" in kw_args:
            # Keyword-based InfoRetrievalTool convention
            call_id = str(kw_args["call_id"])
            args = dict(kw_args.get("args") or {})
        elif len(pos_args) >= 1 and isinstance(pos_args[0], dict):
            # Legacy BaseTool convention: execute(args_dict)
            legacy_args = dict(pos_args[0])
            call_id = str(legacy_args.pop("_call_id", "read_output"))
            args = legacy_args
        elif len(pos_args) >= 1 and isinstance(pos_args[0], str):
            # Positional InfoRetrievalTool convention: execute(call_id, args)
            call_id = pos_args[0]
            args = dict(pos_args[1]) if len(pos_args) > 1 else {}
        else:
            # Fallback -- try kwargs as legacy dict
            call_id = str(kw_args.pop("_call_id", "read_output"))
            args = dict(kw_args)

        return await super().execute(call_id=call_id, args=args)

    # -- Core execution logic ----------------------------------------------

    async def _execute_impl(
        self, *, call_id: str, args: dict[str, Any]
    ) -> ToolResult:
        """Read output from the requested source.

        Dispatches to the appropriate reader based on the ``source``
        parameter.

        Args:
            call_id: Unique identifier for this invocation.
            args: Validated arguments.

        Returns:
            ToolResult with JSON output from the requested source.
        """
        source = args.get("source", "wiki")

        if source == "session":
            return self._read_session(call_id=call_id, args=args)

        # Default: wiki source
        include_connection = args.get("include_connection", False)
        result_data = await asyncio.to_thread(
            self._read_wiki_state, include_connection
        )
        return ToolResult.success(
            call_id=call_id,
            tool_name=self.name,
            output=json.dumps(result_data, default=str),
        )

    # -- Wiki source reader ------------------------------------------------

    def _read_wiki_state(self, include_connection: bool) -> dict[str, Any]:
        """Blocking wiki read -- runs in thread pool.

        Delegates to existing wiki.state_reader module.
        """
        from jules_daemon.wiki.state_reader import load_reconnection_state

        state = load_reconnection_state(self._wiki_root)

        result: dict[str, Any] = {
            "source": "wiki",
            "load_result": state.result.value,
            "run_id": state.run_id,
            "status": state.status.value,
            "resolved_shell": state.resolved_shell,
            "natural_language_command": state.natural_language_command,
            "progress_percent": state.progress_percent,
            "error": state.error,
            "can_reconnect": state.can_reconnect,
        }

        if include_connection and state.connection is not None:
            result["connection"] = {
                "host": state.connection.host,
                "port": state.connection.port,
                "username": state.connection.username,
            }

        return result

    # -- Session source reader ---------------------------------------------

    def _read_session(
        self, *, call_id: str, args: dict[str, Any]
    ) -> ToolResult:
        """Read recent tool results from the session conversation history.

        Extracts tool-role messages from the conversation history,
        optionally filters by tool name, and returns the most recent N
        entries.

        This runs synchronously (no I/O) -- the conversation history is
        an in-memory data structure.
        """
        if self._session_history_provider is None:
            return ToolResult.error(
                call_id=call_id,
                tool_name=self.name,
                error_message=(
                    "Session history is not available. The read_output tool "
                    "was not configured with a session_history_provider. "
                    "Use source='wiki' to read current run state instead."
                ),
            )

        messages = self._session_history_provider()
        if messages is None:
            return ToolResult.error(
                call_id=call_id,
                tool_name=self.name,
                error_message="Session history provider returned None",
            )

        tool_name_filter = args.get("tool_name_filter")
        raw_last_n = args.get("last_n", _DEFAULT_LAST_N)
        last_n = min(max(int(raw_last_n), 1), _MAX_LAST_N)

        # Extract tool-role messages from the conversation history
        tool_entries = _extract_tool_entries(messages, tool_name_filter)

        # Take the most recent N entries
        recent = tool_entries[-last_n:] if len(tool_entries) > last_n else tool_entries

        result_data: dict[str, Any] = {
            "source": "session",
            "total_tool_results": len(tool_entries),
            "returned_count": len(recent),
            "last_n": last_n,
            "tool_name_filter": tool_name_filter,
            "entries": recent,
        }

        return ToolResult.success(
            call_id=call_id,
            tool_name=self.name,
            output=json.dumps(result_data, default=str),
        )


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions)
# ---------------------------------------------------------------------------


def _extract_tool_entries(
    messages: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    tool_name_filter: str | None,
) -> list[dict[str, Any]]:
    """Extract tool-role messages from conversation history.

    Walks the message sequence and collects entries with ``role: "tool"``.
    Correlates each tool message with its originating tool call from the
    preceding assistant message to include the tool name and arguments.

    Args:
        messages: Ordered conversation messages (OpenAI format).
        tool_name_filter: If set, only include entries from this tool.

    Returns:
        List of extracted tool entry dicts, in chronological order.
    """
    # Build a lookup from call_id to (tool_name, arguments) from
    # assistant messages that contain tool_calls.
    call_id_to_info: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            fn = tc.get("function", {})
            call_id_to_info[tc_id] = {
                "tool_name": fn.get("name", "unknown"),
                "arguments": fn.get("arguments", "{}"),
            }

    # Collect tool-role messages and correlate with call info
    entries: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue

        tool_call_id = msg.get("tool_call_id", "")
        content = msg.get("content", "")

        # Look up originating call info
        call_info = call_id_to_info.get(tool_call_id, {})
        tool_name = call_info.get("tool_name", "unknown")

        # Apply filter
        if tool_name_filter and tool_name != tool_name_filter:
            continue

        # Parse arguments string if it looks like JSON
        arguments_raw = call_info.get("arguments", "{}")
        if isinstance(arguments_raw, str):
            try:
                arguments = json.loads(arguments_raw)
            except (json.JSONDecodeError, TypeError):
                arguments = arguments_raw
        else:
            arguments = arguments_raw

        # Detect error vs success from content prefix
        is_error = isinstance(content, str) and content.startswith("ERROR:")

        entry: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "is_error": is_error,
            "content": _truncate_content(content),
        }
        entries.append(entry)

    return entries


def _truncate_content(content: str, max_length: int = 2000) -> str:
    """Truncate long content to prevent oversized responses.

    Args:
        content: The content string to potentially truncate.
        max_length: Maximum character length. Default: 2000.

    Returns:
        The original content if within limits, or a truncated version
        with a trailing indicator.
    """
    if not isinstance(content, str):
        return str(content)
    if len(content) <= max_length:
        return content
    return content[:max_length] + f"... [truncated, {len(content)} chars total]"
