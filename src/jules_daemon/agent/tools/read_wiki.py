"""read_wiki tool -- wraps wiki.test_knowledge.

Searches the wiki for accumulated test knowledge only. Translation-history
pages are intentionally excluded from the active agent path so past NL-to-
command mappings do not bias current command generation.

Extends InfoRetrievalTool (the base class for read-only tools) which
provides:
    - Automatic argument validation against parameters_schema
    - Exception-safe execution wrapping
    - OpenAI-compatible schema serialization (to_openai_schema)
    - ToolSpec conversion (to_tool_spec) for ToolRegistry integration

Delegates to:
    - jules_daemon.wiki.test_knowledge.load_test_knowledge
    - jules_daemon.wiki.test_knowledge.derive_test_slug

No business logic is reimplemented -- this tool composes existing
functions and formats their output for the LLM conversation.

Usage::

    tool = ReadWikiTool(wiki_root=Path("/data/wiki"))

    # InfoRetrievalTool calling convention
    result = await tool.execute(call_id="c1", args={"query": "run tests"})

    # Legacy BaseTool calling convention (backward compat with ToolRegistry)
    result = await tool.execute({"query": "run tests", "_call_id": "c1"})
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from jules_daemon.agent.tool_base import InfoRetrievalTool
from jules_daemon.agent.tool_result import ToolResult
from jules_daemon.agent.tool_types import ToolSpec

__all__ = ["ReadWikiTool"]

logger = logging.getLogger(__name__)

_MAX_TRANSLATIONS = 5
"""Legacy compatibility field; translation results are always empty."""


class ReadWikiTool(InfoRetrievalTool):
    """Search wiki for accumulated test knowledge.

    Wraps:
        - wiki.test_knowledge.load_test_knowledge (accumulated test knowledge)
        - wiki.test_knowledge.derive_test_slug (slug derivation)

    This is a read-only tool (ApprovalRequirement.NONE) that extends
    InfoRetrievalTool. Inherits argument validation, error handling,
    and OpenAI schema serialization from the base class.
    """

    def __init__(self, *, wiki_root: Path) -> None:
        self._wiki_root = wiki_root
        self._spec_cache: ToolSpec | None = None

    # -- Protocol-required properties (InfoRetrievalTool) ------------------

    @property
    def name(self) -> str:
        """Unique tool identifier (function name in OpenAI API)."""
        return "read_wiki"

    @property
    def description(self) -> str:
        """Human-readable description shown to the LLM."""
        return (
            "Search the wiki for accumulated test knowledge. Returns any "
            "known test metadata such as purpose, output format, summary "
            "fields, common failures, normal behavior, and required "
            "arguments."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema dict describing accepted arguments.

        Parameters:
            query (string, required): Search string to match against
                known test knowledge.
            ssh_host (string, optional): Legacy no-op field kept for schema
                compatibility. Translation history is not consulted.
        """
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search string to match against known test knowledge"
                    ),
                },
                "ssh_host": {
                    "type": "string",
                    "description": (
                        "Legacy no-op field kept for compatibility; "
                        "translation history is not used"
                    ),
                },
            },
            "required": ["query"],
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

            result = await tool.execute(call_id="c1", args={"query": "test"})
            result = await tool.execute("c1", {"query": "test"})

        Legacy BaseTool convention (used by ToolRegistry)::

            result = await tool.execute({"query": "test", "_call_id": "c1"})

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
            call_id = str(legacy_args.pop("_call_id", "read_wiki"))
            args = legacy_args
        elif len(pos_args) >= 1 and isinstance(pos_args[0], str):
            # Positional InfoRetrievalTool convention: execute(call_id, args)
            call_id = pos_args[0]
            args = dict(pos_args[1]) if len(pos_args) > 1 else {}
        else:
            # Fallback -- try kwargs as legacy dict
            call_id = str(kw_args.pop("_call_id", "read_wiki"))
            args = dict(kw_args)

        return await super().execute(call_id=call_id, args=args)

    # -- Core execution logic ----------------------------------------------

    async def _execute_impl(
        self, *, call_id: str, args: dict[str, Any]
    ) -> ToolResult:
        """Search wiki for accumulated test knowledge.

        Runs wiki I/O in a thread pool via asyncio.to_thread to avoid
        blocking the event loop.

        Args:
            call_id: Unique identifier for this invocation.
            args: Validated arguments with at least the 'query' key.

        Returns:
            ToolResult with JSON output containing test knowledge and an
            empty translations list for backward compatibility, or an
            error result if the query is empty.
        """
        query = args.get("query", "")
        ssh_host = args.get("ssh_host")

        if not query or not query.strip():
            return ToolResult.error(
                call_id=call_id,
                tool_name=self.name,
                error_message="query parameter is required and must not be empty",
            )

        result_data = await asyncio.to_thread(
            self._search_wiki, query.strip(), ssh_host
        )
        return ToolResult.success(
            call_id=call_id,
            tool_name=self.name,
            output=json.dumps(result_data, default=str),
        )

    # -- Blocking wiki search (runs in thread pool) ------------------------

    def _search_wiki(
        self, query: str, ssh_host: str | None
    ) -> dict[str, Any]:
        """Blocking wiki search -- runs in thread pool.

        Delegates to wiki.test_knowledge for accumulated test knowledge.
        Translation pages are intentionally ignored in the active path.
        """
        from jules_daemon.wiki.test_knowledge import (
            derive_test_slug,
            load_test_knowledge,
        )

        # Look up test knowledge by deriving a slug from the query
        test_slug = derive_test_slug(query)
        knowledge = load_test_knowledge(self._wiki_root, test_slug)

        knowledge_data: dict[str, Any] | None = None
        if knowledge is not None:
            knowledge_data = {
                "test_slug": knowledge.test_slug,
                "command_pattern": knowledge.command_pattern,
                "purpose": knowledge.purpose,
                "output_format": knowledge.output_format,
                "common_failures": list(knowledge.common_failures),
                "normal_behavior": knowledge.normal_behavior,
                "runs_observed": knowledge.runs_observed,
            }

        return {
            "translations": [],
            "test_knowledge": knowledge_data,
            "query": query,
        }
