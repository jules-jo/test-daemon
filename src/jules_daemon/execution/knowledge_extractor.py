"""LLM-driven extraction of durable test knowledge from a completed run.

After each test run finishes, this module asks the configured LLM
client to distill durable observations about the test (purpose, output
format, common failures, normal behavior) so that future runs can be
summarized with richer context. The output is a small JSON dict that
the caller hands to
:func:`jules_daemon.wiki.test_knowledge.merge_knowledge`.

Design notes:

- Always fail-soft: any error (timeout, network, malformed JSON,
  missing client) returns ``None`` so the audit flow keeps moving.
- The blocking LLM call is dispatched via :func:`asyncio.to_thread` to
  avoid stalling the asyncio event loop.
- The same 10-second timeout used by the output summarizer is reused
  for symmetry; the constant is imported so a single source of truth
  governs the LLM budget.
- The prompt instructs the model to skip unchanged content when prior
  knowledge already exists, which keeps the wiki page from being
  rewritten on every run.

Usage::

    from jules_daemon.execution.knowledge_extractor import extract_knowledge

    new_observations = await extract_knowledge(
        command="pytest tests/integration",
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        existing_knowledge=current_knowledge,
        llm_client=handler.llm_client,
        llm_model=handler.llm_model,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

from jules_daemon.execution.output_summarizer import (
    LLM_SUMMARIZER_TIMEOUT_SECONDS,
)
from jules_daemon.wiki.test_knowledge import TestKnowledge

__all__ = [
    "extract_knowledge",
    "KNOWLEDGE_EXTRACTOR_TIMEOUT_SECONDS",
]

logger = logging.getLogger(__name__)


# Reuse the summarizer timeout so the daemon has a single LLM budget.
KNOWLEDGE_EXTRACTOR_TIMEOUT_SECONDS: float = LLM_SUMMARIZER_TIMEOUT_SECONDS

# Bound the amount of stdout/stderr forwarded to the LLM so prompts
# stay reasonable regardless of how chatty the test was.
_LLM_PROMPT_OUTPUT_CHARS: int = 8_000

# Maximum length of any single value the LLM may return. Prevents
# accidentally accepting a 50KB "purpose" field that would dominate
# subsequent prompts.
_MAX_FIELD_CHARS: int = 600

# Maximum number of common failure patterns we accept from a single
# extraction call. The merge layer caps the global list separately.
_MAX_FAILURES_PER_RUN: int = 5


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_existing_knowledge(existing: Optional[TestKnowledge]) -> str:
    """Render the prior knowledge (if any) for inclusion in the prompt."""
    if existing is None:
        return "(no prior knowledge captured for this test)"
    rendered = existing.to_prompt_context()
    return rendered or "(no prior knowledge captured for this test)"


def _tail(text: str, limit: int) -> str:
    """Return the last *limit* characters of *text* (empty string on ``None``)."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _combined_output(stdout: str, stderr: str) -> str:
    """Combine stdout and stderr into one string for the prompt."""
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        if parts:
            parts.append("\n--- stderr ---\n")
        parts.append(stderr)
    return "".join(parts)


def _build_prompt(
    *,
    command: str,
    exit_code: Optional[int],
    output_tail: str,
    existing_text: str,
) -> str:
    """Build the user prompt sent to the LLM extractor."""
    exit_display = str(exit_code) if exit_code is not None else "(none)"
    return (
        "You are analyzing a test run to extract DURABLE knowledge that "
        "would help summarize future runs of the same test.\n\n"
        f"Command: {command}\n"
        f"Exit code: {exit_display}\n\n"
        "Existing knowledge (do NOT restate fields that are already "
        "captured -- leave them blank):\n"
        f"{existing_text}\n\n"
        "Output (truncated):\n"
        f"{output_tail}\n\n"
        "Extract:\n"
        "1. purpose -- one short sentence describing what this test does\n"
        "2. output_format -- how to interpret the output (counts, log "
        "lines, custom shape, etc.)\n"
        "3. common_failures -- up to 5 short failure patterns observable "
        "in this run (skip if none)\n"
        "4. normal_behavior -- one or two sentences describing what a "
        "healthy run looks like\n\n"
        "Rules:\n"
        "- Be brief. Each field <= 600 characters.\n"
        "- Only include observations that are clearly visible in the "
        "command and output. Skip fields you are not confident about.\n"
        "- If a field is already captured by existing knowledge, return "
        "an empty string for it.\n\n"
        "Respond as strict JSON with exactly these keys:\n"
        "{\n"
        '  "purpose": "...",\n'
        '  "output_format": "...",\n'
        '  "common_failures": ["...", "..."],\n'
        '  "normal_behavior": "..."\n'
        "}"
    )


# ---------------------------------------------------------------------------
# LLM call + JSON parsing
# ---------------------------------------------------------------------------


def _extract_text_from_response(response: Any) -> str:
    """Pull the assistant text out of an OpenAI ChatCompletion response.

    Tolerates both SDK object and plain-dict shapes so tests can stub
    the client with lightweight fakes.
    """
    try:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None and isinstance(first, dict):
            message = first.get("message")
        if message is None:
            return ""
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        return content or ""
    except Exception:  # pragma: no cover - defensive
        return ""


def _parse_llm_json(text: str) -> Optional[dict[str, Any]]:
    """Parse JSON from a possibly-wrapped LLM response.

    Mirrors the parser in :mod:`output_summarizer` but lives here so
    the two modules stay independent.
    """
    if not text:
        return None
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except (json.JSONDecodeError, ValueError):
        pass
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL
    )
    if fence_match:
        try:
            value = json.loads(fence_match.group(1))
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, ValueError):
            pass
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        try:
            value = json.loads(brace_match.group(0))
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _coerce_field(value: Any) -> str:
    """Coerce a JSON value to a trimmed, length-bounded string."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if len(cleaned) > _MAX_FIELD_CHARS:
        cleaned = cleaned[: _MAX_FIELD_CHARS - 3] + "..."
    return cleaned


def _coerce_failures(value: Any) -> list[str]:
    """Coerce the ``common_failures`` field into a bounded list."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _coerce_field(item)
        if not text:
            continue
        if text not in out:
            out.append(text)
        if len(out) >= _MAX_FAILURES_PER_RUN:
            break
    return out


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply field-level coercion to a parsed LLM JSON payload."""
    return {
        "purpose": _coerce_field(payload.get("purpose")),
        "output_format": _coerce_field(payload.get("output_format")),
        "normal_behavior": _coerce_field(payload.get("normal_behavior")),
        "common_failures": _coerce_failures(payload.get("common_failures")),
    }


def _call_llm_blocking(
    *,
    llm_client: Any,
    llm_model: str,
    prompt: str,
) -> str:
    """Synchronous LLM call -- runs inside :func:`asyncio.to_thread`."""
    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract durable, reusable knowledge about a "
                    "test from one of its runs. Always answer with a "
                    "single JSON object and never add prose around it."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return _extract_text_from_response(response)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def extract_knowledge(
    *,
    command: str,
    stdout: str,
    stderr: str,
    exit_code: Optional[int],
    existing_knowledge: Optional[TestKnowledge],
    llm_client: Any | None,
    llm_model: Optional[str],
) -> Optional[dict[str, Any]]:
    """Ask the LLM to extract durable test knowledge from a run.

    Returns a dict with the keys ``purpose``, ``output_format``,
    ``common_failures``, and ``normal_behavior``. Each value is already
    coerced (string fields are trimmed to a sane length and failure
    lists are deduped). Returns ``None`` on any error so the caller can
    skip the merge step without special-casing exceptions.

    Args:
        command: The full shell command that produced the output.
        stdout: Captured standard output (may be empty).
        stderr: Captured standard error (may be empty).
        exit_code: The process exit code, or ``None`` if no code was
            captured (e.g. connection failure).
        existing_knowledge: Prior accumulated knowledge for this test,
            or ``None`` for the first observation. The function uses it
            only to build a context block for the prompt -- it does not
            mutate or read individual fields beyond that.
        llm_client: Optional OpenAI-compatible client. When ``None``,
            the function returns ``None`` immediately.
        llm_model: Default model identifier to pass to the client.
            Required when ``llm_client`` is provided.

    Returns:
        A dict with the four extraction fields, or ``None`` on any
        failure.
    """
    if llm_client is None or not llm_model:
        return None
    if not command or not command.strip():
        return None

    combined = _combined_output(stdout, stderr)
    output_tail = _tail(combined, _LLM_PROMPT_OUTPUT_CHARS)
    existing_text = _format_existing_knowledge(existing_knowledge)
    prompt = _build_prompt(
        command=command,
        exit_code=exit_code,
        output_tail=output_tail,
        existing_text=existing_text,
    )

    try:
        raw_text = await asyncio.wait_for(
            asyncio.to_thread(
                _call_llm_blocking,
                llm_client=llm_client,
                llm_model=llm_model,
                prompt=prompt,
            ),
            timeout=KNOWLEDGE_EXTRACTOR_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.debug(
            "Knowledge extractor timed out after %ss",
            KNOWLEDGE_EXTRACTOR_TIMEOUT_SECONDS,
        )
        return None
    except Exception as exc:
        logger.debug("Knowledge extractor LLM call failed: %s", exc)
        return None

    parsed = _parse_llm_json(raw_text)
    if parsed is None:
        logger.debug(
            "Knowledge extractor returned unparseable text (%d chars)",
            len(raw_text),
        )
        return None

    return _normalize_payload(parsed)
