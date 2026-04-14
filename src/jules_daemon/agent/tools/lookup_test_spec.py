"""lookup_test_spec tool -- wraps wiki.test_knowledge for structured test specs.

Looks up a test specification from the wiki test catalog by name or slug.
The catalog uses hybrid wiki files: user creates starter specs (command_template,
required_args), daemon augments (typical_duration, failure_patterns, summary_fields).

Extends InfoRetrievalTool (the base class for read-only tools) which
provides:
    - Automatic argument validation against parameters_schema
    - Exception-safe execution wrapping
    - OpenAI-compatible schema serialization (to_openai_schema)
    - ToolSpec conversion (to_tool_spec) for ToolRegistry integration

Delegates to:
    - jules_daemon.wiki.test_knowledge.derive_test_slug
    - jules_daemon.wiki.test_knowledge.load_test_knowledge

No business logic is reimplemented -- this tool composes existing
functions and formats their output for the LLM conversation.

Usage::

    tool = LookupTestSpecTool(wiki_root=Path("/data/wiki"))

    # InfoRetrievalTool calling convention
    result = await tool.execute(call_id="c1", args={"test_name": "agent_test"})

    # Legacy BaseTool calling convention (backward compat with ToolRegistry)
    result = await tool.execute({"test_name": "agent_test", "_call_id": "c1"})
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

__all__ = ["LookupTestSpecTool"]

logger = logging.getLogger(__name__)

_QUERY_STOPWORDS: frozenset[str] = frozenset({
    "a",
    "an",
    "about",
    "can",
    "current",
    "file",
    "for",
    "give",
    "learn",
    "me",
    "please",
    "run",
    "script",
    "show",
    "status",
    "test",
    "tests",
    "the",
    "this",
})


class LookupTestSpecTool(InfoRetrievalTool):
    """Look up a test specification from the wiki test catalog.

    Wraps:
        - wiki.test_knowledge.derive_test_slug (slug from test name)
        - wiki.test_knowledge.load_test_knowledge (load catalog entry)

    This is a read-only tool (ApprovalRequirement.NONE) that extends
    InfoRetrievalTool. Inherits argument validation, error handling,
    and OpenAI schema serialization from the base class.

    The tool surfaces required_args from the test spec so that the
    agent loop can detect missing arguments and prompt the user via
    the ask_user_question tool.
    """

    def __init__(self, *, wiki_root: Path) -> None:
        self._wiki_root = wiki_root
        self._spec_cache: ToolSpec | None = None

    # -- Protocol-required properties (InfoRetrievalTool) ------------------

    @property
    def name(self) -> str:
        """Unique tool identifier (function name in OpenAI API)."""
        return "lookup_test_spec"

    @property
    def description(self) -> str:
        """Human-readable description shown to the LLM."""
        return (
            "Look up a test specification from the wiki test catalog. "
            "Returns the test's command pattern, purpose, output format, "
            "summary fields, common failures, normal behavior, required "
            "arguments, and run count. Use this to understand what a "
            "test does and what arguments it needs before proposing an "
            "SSH command."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema dict describing accepted arguments.

        Parameters:
            test_name (string, required): Name or command of the test
                to look up (e.g., 'agent_test', 'pytest tests/integration').
        """
        return {
            "type": "object",
            "properties": {
                "test_name": {
                    "type": "string",
                    "description": (
                        "Name or command of the test to look up "
                        "(e.g., 'agent_test', 'pytest tests/integration')"
                    ),
                },
            },
            "required": ["test_name"],
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

            result = await tool.execute(call_id="c1", args={"test_name": "test"})
            result = await tool.execute("c1", {"test_name": "test"})

        Legacy BaseTool convention (used by ToolRegistry)::

            result = await tool.execute({"test_name": "test", "_call_id": "c1"})

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
            call_id = str(legacy_args.pop("_call_id", "lookup_test_spec"))
            args = legacy_args
        elif len(pos_args) >= 1 and isinstance(pos_args[0], str):
            # Positional InfoRetrievalTool convention: execute(call_id, args)
            call_id = pos_args[0]
            args = dict(pos_args[1]) if len(pos_args) > 1 else {}
        else:
            # Fallback -- try kwargs as legacy dict
            call_id = str(kw_args.pop("_call_id", "lookup_test_spec"))
            args = dict(kw_args)

        return await super().execute(call_id=call_id, args=args)

    # -- Core execution logic ----------------------------------------------

    async def _execute_impl(
        self, *, call_id: str, args: dict[str, Any]
    ) -> ToolResult:
        """Look up a test spec by name or command string.

        Runs wiki I/O in a thread pool via asyncio.to_thread to avoid
        blocking the event loop.

        Args:
            call_id: Unique identifier for this invocation.
            args: Validated arguments with at least the 'test_name' key.

        Returns:
            ToolResult with JSON output containing the test specification
            data, or a not-found indicator.
        """
        test_name = args.get("test_name", "")

        if not test_name or not test_name.strip():
            return ToolResult.error(
                call_id=call_id,
                tool_name=self.name,
                error_message="test_name parameter is required and must not be empty",
            )

        result_data = await asyncio.to_thread(
            self._lookup, test_name.strip()
        )
        return ToolResult.success(
            call_id=call_id,
            tool_name=self.name,
            output=json.dumps(result_data, default=str),
        )

    # -- Blocking wiki lookup (runs in thread pool) ------------------------

    def _lookup(self, test_name: str) -> dict[str, Any]:
        """Blocking wiki lookup -- runs in thread pool.

        Tries multiple slug strategies to find the test spec:
        1. Direct slug from derive_test_slug (handles command-style input)
        2. Simple slugify of the test name (handles "ld test" -> "ld-test")
        3. Glob search for partial matches in the knowledge directory
        """
        import re
        from jules_daemon.wiki.test_knowledge import (
            KNOWLEDGE_DIR,
            derive_test_slug,
            load_test_knowledge,
        )

        def _normalize_text(value: str) -> str:
            return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()

        def _meaningful_tokens(value: str) -> list[str]:
            normalized = _normalize_text(value)
            return [
                token
                for token in normalized.split()
                if len(token) >= 2
                and token not in _QUERY_STOPWORDS
                and not token.isdigit()
            ]

        # Strategy 1: derive_test_slug (best for command-style input)
        slug = derive_test_slug(test_name)
        knowledge = load_test_knowledge(self._wiki_root, slug)

        # Strategy 2: simple slugify (best for test names like "ld test")
        if knowledge is None:
            simple_slug = re.sub(r"[^a-z0-9]+", "-", test_name.lower()).strip("-")
            if simple_slug and simple_slug != slug:
                knowledge = load_test_knowledge(self._wiki_root, simple_slug)
                if knowledge is not None:
                    slug = simple_slug

        # Strategy 3: scan all spec files and match against frontmatter
        # fields: name, test_slug, command_template
        if knowledge is None:
            knowledge_dir = self._wiki_root / KNOWLEDGE_DIR
            if knowledge_dir.is_dir():
                search_normalized = _normalize_text(test_name)
                search_tokens = _meaningful_tokens(test_name)
                search_phrase = " ".join(search_tokens)
                best_match: tuple[int, str] | None = None

                for md_file in sorted(knowledge_dir.glob("test-*.md")):
                    try:
                        from jules_daemon.wiki import frontmatter
                        raw = md_file.read_text(encoding="utf-8")
                        doc = frontmatter.parse(raw)
                        fm = doc.frontmatter

                        # Score exact/near matches above loose token matches.
                        fm_name = _normalize_text(str(fm.get("name", "")))
                        fm_slug = _normalize_text(str(fm.get("test_slug", "")))
                        fm_cmd = _normalize_text(
                            str(fm.get("command_pattern") or fm.get("command_template", ""))
                        )
                        field_scores = [
                            (fm_name, 4),
                            (fm_slug, 3),
                            (fm_cmd, 1),
                        ]

                        score = 0
                        for field_value, field_weight in field_scores:
                            if not field_value:
                                continue
                            if search_phrase and search_phrase == field_value:
                                score = max(score, 100 + field_weight)
                            elif search_normalized and search_normalized == field_value:
                                score = max(score, 95 + field_weight)
                            elif search_phrase and search_phrase in field_value:
                                score = max(score, 80 + field_weight)
                            elif search_tokens and all(token in field_value for token in search_tokens):
                                score = max(score, 70 + field_weight)
                            elif search_tokens:
                                exact_token_hits = sum(
                                    1
                                    for token in search_tokens
                                    if token in field_value.split()
                                )
                                partial_token_hits = sum(
                                    1
                                    for token in search_tokens
                                    if token in field_value
                                )
                                if exact_token_hits > 0 or partial_token_hits > 0:
                                    score = max(
                                        score,
                                        exact_token_hits * 15
                                        + partial_token_hits * 5
                                        + field_weight,
                                    )

                        if score <= 0:
                            continue

                        match_slug = md_file.stem.removeprefix("test-")
                        if best_match is None or score > best_match[0]:
                            best_match = (score, match_slug)
                    except Exception:
                        continue

                if best_match is not None:
                    match_slug = best_match[1]
                    knowledge = load_test_knowledge(self._wiki_root, match_slug)
                    if knowledge is not None:
                        slug = match_slug

        if knowledge is None:
            return {
                "found": False,
                "test_slug": slug,
                "message": f"No test specification found for '{test_name}' (slug: {slug})",
            }

        return {
            "found": True,
            "test_slug": knowledge.test_slug,
            "command_pattern": knowledge.command_pattern,
            "purpose": knowledge.purpose,
            "output_format": knowledge.output_format,
            "test_file_path": knowledge.test_file_path,
            "summary_fields": list(knowledge.summary_fields),
            "common_failures": list(knowledge.common_failures),
            "normal_behavior": knowledge.normal_behavior,
            "required_args": list(knowledge.required_args),
            "runs_observed": knowledge.runs_observed,
        }
