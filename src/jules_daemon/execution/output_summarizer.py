"""Hybrid test output summarizer for audit records.

Produces a high-level :class:`OutputSummary` from captured stdout/stderr
without storing the raw output verbatim in the audit record. The design
uses two cooperating layers:

1. **Regex layer (free, fast)** -- tries a small battery of patterns for
   the common test frameworks (pytest, unittest, jest, and a generic
   iteration counter used by custom test loops). Yields counts, duration,
   and a deterministic parser label.
2. **LLM layer (optional)** -- when an OpenAI-style client is available,
   the raw tail of output is sent to the model with a strict JSON prompt.
   The LLM is used for two purposes:

   - **Narrative synthesis**: always requested if enabled, produces a
     one- or two-sentence human summary of the run.
   - **Fallback parsing**: if the regex layer produced no counts, the
     LLM's structured output is used to populate them.

The module is deliberately resilient:

- No mutation of inputs; all intermediate state flows through immutable
  dataclasses.
- Any error -- JSON parse failure, LLM timeout, network error, missing
  client -- is swallowed. The caller always receives a valid
  :class:`OutputSummary` (possibly with narrative == "" or parser == "none").
- The blocking LLM call is offloaded to :func:`asyncio.to_thread` so the
  asyncio event loop is never stalled by network I/O.
- No secrets or credentials are ever referenced here -- only the caller's
  own stdout/stderr/command strings are processed.

Usage::

    from jules_daemon.execution.output_summarizer import summarize_output

    summary = await summarize_output(
        stdout=result.stdout,
        stderr=result.stderr,
        command=result.command,
        exit_code=result.exit_code,
        llm_client=handler_config.llm_client,
        llm_model=handler_config.llm_config.default_model,
    )
    print(summary.narrative, summary.passed, summary.failed)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "OutputSummary",
    "summarize_output",
    "LLM_SUMMARIZER_TIMEOUT_SECONDS",
]

logger = logging.getLogger(__name__)

# How much of the combined stdout/stderr is forwarded to the LLM. Keeps
# prompt tokens bounded regardless of pathological output lengths.
_LLM_PROMPT_OUTPUT_CHARS: int = 8_000

# Raw output excerpt size stored on the OutputSummary for downstream
# debugging. The audit writer can embed this in the record so a human
# reading the wiki file still sees the tail of the failing output.
_RAW_EXCERPT_CHARS: int = 500

# Per-failure description cap (used when coercing LLM output).
_MAX_FAILURE_DESCRIPTION_CHARS: int = 100

# Maximum number of ``key_failures`` surfaced in a summary. The LLM
# prompt asks for up to five -- anything beyond that is dropped to
# keep audit summaries scannable.
_MAX_FAILURES: int = 5

# Timeout applied to the blocking LLM call. If the endpoint is slow or
# unreachable, the summarizer degrades gracefully to a regex-only result.
LLM_SUMMARIZER_TIMEOUT_SECONDS: float = 10.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputSummary:
    """Structured summary of a test/command output.

    Attributes:
        parser: Identifier for the layer that produced the counts
            (``"pytest"``, ``"unittest"``, ``"jest"``, ``"iteration"``,
            ``"llm"``, or ``"none"``).
        passed: Number of passing tests detected (``0`` if not
            applicable).
        failed: Number of failing tests detected.
        skipped: Number of skipped tests detected.
        total: Total test count detected. When the parser only reports
            ``passed``/``failed``/``skipped``, callers can use
            ``passed + failed + skipped`` instead.
        duration_seconds: Wall-clock duration in seconds if reported by
            the framework, otherwise ``None``.
        key_failures: Up to :data:`_MAX_FAILURES` short descriptions of
            the failing tests. Each entry is capped at
            :data:`_MAX_FAILURE_DESCRIPTION_CHARS` characters.
        narrative: 1-2 sentence human summary of what happened. Empty
            when the LLM is disabled or the call failed.
        raw_excerpt: Last :data:`_RAW_EXCERPT_CHARS` characters of the
            combined stdout/stderr, suitable for embedding in the audit
            file for debugging context.
    """

    parser: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    total: int = 0
    duration_seconds: float | None = None
    key_failures: tuple[str, ...] = field(default_factory=tuple)
    narrative: str = ""
    raw_excerpt: str = ""


# ---------------------------------------------------------------------------
# Regex layer
# ---------------------------------------------------------------------------


_PATTERNS: dict[str, re.Pattern[str]] = {
    "pytest": re.compile(
        r"=+\s*(?:(?P<passed>\d+)\s+passed)?[,\s]*"
        r"(?:(?P<failed>\d+)\s+failed)?[,\s]*"
        r"(?:(?P<skipped>\d+)\s+skipped)?[,\s]*"
        r"(?:(?P<errors>\d+)\s+error[s]?)?.*?"
        r"in\s+(?P<duration>[\d.]+)s"
    ),
    "unittest": re.compile(
        r"Ran\s+(?P<total>\d+)\s+tests?\s+in\s+(?P<duration>[\d.]+)s"
    ),
    "jest": re.compile(
        r"Tests:\s+(?:(?P<failed>\d+)\s+failed,\s+)?"
        r"(?:(?P<passed>\d+)\s+passed,\s+)?"
        r"(?P<total>\d+)\s+total"
    ),
    "iteration": re.compile(
        r"[Ii]teration\s+\d+(?:/\d+)?:\s+(?P<outcome>PASSED|FAILED)"
    ),
}


def _safe_int(value: str | None) -> int:
    """Convert a capture group to int, defaulting to ``0`` on ``None``."""
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: str | None) -> float | None:
    """Convert a capture group to float, returning ``None`` on ``None``."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _regex_summary(text: str) -> OutputSummary | None:
    """Try each parser pattern in order and return the first match.

    Returns ``None`` if no parser matched -- the caller may then fall
    back to the LLM layer. The returned summary has an empty
    ``narrative`` and ``raw_excerpt`` -- those are populated by
    :func:`summarize_output` after the regex layer runs.
    """
    if not text:
        return None

    # pytest: e.g. "==== 95 passed, 5 failed in 1.23s ===="
    match = _PATTERNS["pytest"].search(text)
    if match and any(
        match.group(name) for name in ("passed", "failed", "skipped", "errors")
    ):
        passed = _safe_int(match.group("passed"))
        failed = _safe_int(match.group("failed"))
        skipped = _safe_int(match.group("skipped"))
        errors = _safe_int(match.group("errors"))
        duration = _safe_float(match.group("duration"))
        # pytest reports errors separately from failures; roll them in
        # so audit summaries always show a single "failed" number.
        failed = failed + errors
        total = passed + failed + skipped
        return OutputSummary(
            parser="pytest",
            passed=passed,
            failed=failed,
            skipped=skipped,
            total=total,
            duration_seconds=duration,
        )

    # unittest: e.g. "Ran 42 tests in 0.500s"
    match = _PATTERNS["unittest"].search(text)
    if match:
        total = _safe_int(match.group("total"))
        duration = _safe_float(match.group("duration"))
        # unittest reports failures with "FAILED (failures=N, errors=M)";
        # look for the companion line to split passed/failed out.
        failures, errors = _extract_unittest_failures(text)
        failed = failures + errors
        # Skipped is typically reported separately -- look it up.
        skipped = _extract_unittest_skipped(text)
        passed = max(total - failed - skipped, 0)
        return OutputSummary(
            parser="unittest",
            passed=passed,
            failed=failed,
            skipped=skipped,
            total=total,
            duration_seconds=duration,
        )

    # jest: e.g. "Tests:       5 failed, 95 passed, 100 total"
    match = _PATTERNS["jest"].search(text)
    if match:
        passed = _safe_int(match.group("passed"))
        failed = _safe_int(match.group("failed"))
        total = _safe_int(match.group("total"))
        skipped = max(total - passed - failed, 0)
        return OutputSummary(
            parser="jest",
            passed=passed,
            failed=failed,
            skipped=skipped,
            total=total,
        )

    # iteration: count PASSED/FAILED lines individually, e.g.
    # "Iteration 1: PASSED", "Iteration 2/100: FAILED"
    iter_matches = list(_PATTERNS["iteration"].finditer(text))
    if iter_matches:
        passed = sum(
            1 for m in iter_matches if m.group("outcome").upper() == "PASSED"
        )
        failed = sum(
            1 for m in iter_matches if m.group("outcome").upper() == "FAILED"
        )
        total = passed + failed
        return OutputSummary(
            parser="iteration",
            passed=passed,
            failed=failed,
            skipped=0,
            total=total,
        )

    return None


_UNITTEST_FAILURES_PATTERN = re.compile(
    r"FAILED\s*\((?:[^)]*failures=(?P<failures>\d+))?"
    r"(?:[^)]*errors=(?P<errors>\d+))?[^)]*\)"
)

_UNITTEST_SKIPPED_PATTERN = re.compile(
    r"OK\s*\([^)]*skipped=(?P<skipped>\d+)[^)]*\)"
    r"|FAILED\s*\([^)]*skipped=(?P<skipped2>\d+)[^)]*\)"
)


def _extract_unittest_failures(text: str) -> tuple[int, int]:
    """Extract (failures, errors) from a unittest trailer line."""
    match = _UNITTEST_FAILURES_PATTERN.search(text)
    if not match:
        return 0, 0
    return _safe_int(match.group("failures")), _safe_int(match.group("errors"))


def _extract_unittest_skipped(text: str) -> int:
    """Extract the skipped count from a unittest trailer line."""
    match = _UNITTEST_SKIPPED_PATTERN.search(text)
    if not match:
        return 0
    return _safe_int(match.group("skipped") or match.group("skipped2"))


# ---------------------------------------------------------------------------
# LLM layer
# ---------------------------------------------------------------------------


def _tail(text: str, limit: int) -> str:
    """Return the last *limit* characters of *text* (empty string on ``None``)."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _combined_output(stdout: str, stderr: str) -> str:
    """Combine stdout and stderr into a single string for parsing."""
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
    exit_code: int | None,
    output_tail: str,
    wiki_context: str = "",
) -> str:
    """Build the user prompt sent to the LLM.

    The prompt is deliberately explicit about the desired JSON shape so
    the model's response can be parsed reliably without ``response_format``
    (Dataiku Mesh does not support it).

    When ``wiki_context`` is non-empty it is injected ahead of the test
    output so the LLM can leverage prior accumulated knowledge about
    this command (purpose, output format, normal behavior, common
    failure patterns). The context block is rendered by
    :meth:`jules_daemon.wiki.test_knowledge.TestKnowledge.to_prompt_context`.
    """
    exit_display = str(exit_code) if exit_code is not None else "(none)"
    context_block = ""
    if wiki_context:
        context_block = (
            f"{wiki_context}\n\n"
            "Use that prior knowledge to make the narrative more "
            "specific (mention what passed/failed relative to normal "
            "behavior).\n\n"
        )
    return (
        "You are analyzing the output of a test run.\n\n"
        f"Command: {command}\n"
        f"Exit code: {exit_display}\n\n"
        f"{context_block}"
        "Output (truncated):\n"
        f"{output_tail}\n\n"
        "Extract:\n"
        "1. Test counts (passed, failed, skipped) -- use 0 if not applicable\n"
        "2. Key failures (up to 5 short descriptions, each under 100 chars)\n"
        "3. Narrative summary (1-2 sentences): what happened overall\n\n"
        "Respond with JSON:\n"
        "{\n"
        '  "passed": 0,\n'
        '  "failed": 0,\n'
        '  "skipped": 0,\n'
        '  "total": 0,\n'
        '  "key_failures": ["...", "..."],\n'
        '  "narrative": "..."\n'
        "}"
    )


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


def _parse_llm_json(text: str) -> dict[str, Any] | None:
    """Parse JSON from a possibly-wrapped LLM response.

    Handles the common cases where the model wraps JSON in
    ```` ```json ... ``` ```` fences or adds prose before/after. Returns
    ``None`` on any failure so callers can fall through to the regex
    result.
    """
    if not text:
        return None
    # Try direct JSON parse first.
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except (json.JSONDecodeError, ValueError):
        pass
    # Look for a ```json``` fenced block.
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
    # Look for the first top-level JSON object.
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        try:
            value = json.loads(brace_match.group(0))
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _coerce_failures(value: Any) -> tuple[str, ...]:
    """Coerce the LLM ``key_failures`` field into a bounded tuple."""
    if not isinstance(value, list):
        return ()
    coerced: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if not stripped:
            continue
        if len(stripped) > _MAX_FAILURE_DESCRIPTION_CHARS:
            stripped = stripped[: _MAX_FAILURE_DESCRIPTION_CHARS - 3] + "..."
        coerced.append(stripped)
        if len(coerced) >= _MAX_FAILURES:
            break
    return tuple(coerced)


def _coerce_narrative(value: Any) -> str:
    """Coerce the LLM ``narrative`` field into a trimmed string."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _call_llm_blocking(
    *,
    llm_client: Any,
    llm_model: str,
    prompt: str,
) -> str:
    """Invoke the LLM synchronously and return the assistant text.

    Wraps the OpenAI-style ``chat.completions.create`` call. All
    exceptions propagate to the asyncio wrapper so it can handle them
    uniformly alongside timeouts.
    """
    response = llm_client.chat.completions.create(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize test and command output into "
                    "strict JSON for an audit system. Never add prose "
                    "outside the JSON object."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return _extract_text_from_response(response)


async def _llm_summary(
    *,
    llm_client: Any,
    llm_model: str,
    command: str,
    exit_code: int | None,
    output_tail: str,
    wiki_context: str = "",
) -> dict[str, Any] | None:
    """Async wrapper around the blocking LLM call.

    Returns the parsed JSON payload on success or ``None`` on any
    failure (timeout, network error, malformed response). Errors are
    logged at debug level so production noise stays low.
    """
    prompt = _build_prompt(
        command=command,
        exit_code=exit_code,
        output_tail=output_tail,
        wiki_context=wiki_context,
    )
    try:
        raw_text = await asyncio.wait_for(
            asyncio.to_thread(
                _call_llm_blocking,
                llm_client=llm_client,
                llm_model=llm_model,
                prompt=prompt,
            ),
            timeout=LLM_SUMMARIZER_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.debug("LLM summarizer timed out after %ss", LLM_SUMMARIZER_TIMEOUT_SECONDS)
        return None
    except Exception as exc:
        logger.debug("LLM summarizer call failed: %s", exc)
        return None

    parsed = _parse_llm_json(raw_text)
    if parsed is None:
        logger.debug("LLM summarizer returned unparseable text (%d chars)", len(raw_text))
    return parsed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def summarize_output(
    *,
    stdout: str,
    stderr: str,
    command: str,
    exit_code: int | None,
    llm_client: Any | None = None,
    llm_model: str | None = None,
    wiki_context: str = "",
) -> OutputSummary:
    """Hybrid summarization: regex first, LLM for narrative and fallback.

    The function always returns a valid :class:`OutputSummary`. When no
    regex pattern matches and the LLM is unavailable, the parser label
    is ``"none"`` and all counts are zero -- callers can render a
    best-effort message using the :attr:`OutputSummary.raw_excerpt`
    field.

    Args:
        stdout: Captured standard output (may be empty).
        stderr: Captured standard error (may be empty).
        command: The shell command that produced the output. Included
            in the LLM prompt for extra context.
        exit_code: The process exit code, or ``None`` if no code was
            captured (e.g. connection failure).
        llm_client: Optional OpenAI-compatible client. When ``None``,
            the LLM layer is skipped entirely.
        llm_model: Default model identifier to pass to the client.
            Required when ``llm_client`` is provided; otherwise the
            client call is skipped (treated as disabled).
        wiki_context: Optional formatted prior knowledge about this
            test (typically the output of
            :meth:`jules_daemon.wiki.test_knowledge.TestKnowledge.to_prompt_context`).
            When non-empty it is included in the LLM prompt so the
            narrative can leverage past observations. Empty string is
            equivalent to "no prior knowledge" and the prompt remains
            backward-compatible.

    Returns:
        A frozen :class:`OutputSummary`.
    """
    combined = _combined_output(stdout, stderr)
    raw_excerpt = _tail(combined, _RAW_EXCERPT_CHARS)

    regex_result = _regex_summary(combined)

    # When the LLM is disabled or misconfigured, return the regex
    # result as-is (or a default "none" entry).
    if llm_client is None or not llm_model:
        base = regex_result or OutputSummary(parser="none")
        return OutputSummary(
            parser=base.parser,
            passed=base.passed,
            failed=base.failed,
            skipped=base.skipped,
            total=base.total,
            duration_seconds=base.duration_seconds,
            key_failures=base.key_failures,
            narrative=base.narrative,
            raw_excerpt=raw_excerpt,
        )

    output_tail = _tail(combined, _LLM_PROMPT_OUTPUT_CHARS)
    llm_payload = await _llm_summary(
        llm_client=llm_client,
        llm_model=llm_model,
        command=command,
        exit_code=exit_code,
        output_tail=output_tail,
        wiki_context=wiki_context,
    )

    narrative = ""
    key_failures: tuple[str, ...] = ()
    if llm_payload is not None:
        narrative = _coerce_narrative(llm_payload.get("narrative"))
        key_failures = _coerce_failures(llm_payload.get("key_failures"))

    # If regex produced counts, keep them and only inherit narrative +
    # key failures from the LLM. If regex failed, fall back fully to
    # the LLM's counts (parser == "llm") when they are available.
    if regex_result is not None:
        return OutputSummary(
            parser=regex_result.parser,
            passed=regex_result.passed,
            failed=regex_result.failed,
            skipped=regex_result.skipped,
            total=regex_result.total,
            duration_seconds=regex_result.duration_seconds,
            key_failures=key_failures,
            narrative=narrative,
            raw_excerpt=raw_excerpt,
        )

    if llm_payload is None:
        return OutputSummary(parser="none", raw_excerpt=raw_excerpt)

    passed = _safe_int(str(llm_payload.get("passed", 0)))
    failed = _safe_int(str(llm_payload.get("failed", 0)))
    skipped = _safe_int(str(llm_payload.get("skipped", 0)))
    total_raw = llm_payload.get("total", 0)
    total = _safe_int(str(total_raw)) or (passed + failed + skipped)
    return OutputSummary(
        parser="llm",
        passed=passed,
        failed=failed,
        skipped=skipped,
        total=total,
        duration_seconds=None,
        key_failures=key_failures,
        narrative=narrative,
        raw_excerpt=raw_excerpt,
    )
