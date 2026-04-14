"""Build typed *Args dataclasses from classification extracted_args dicts.

Converts the untyped ``extracted_args`` dictionary produced by the input
classifier (both structured and NL paths) into the verb-specific frozen
``*Args`` dataclass expected by the handler registry and dispatcher.

This module is the bridge that ensures **identical argument structures**
reach handlers regardless of input style: whether the user typed a
structured CLI command (``run deploy@host run tests``) or natural
language (``run the smoke tests on staging``), the handler receives
the same ``RunArgs`` dataclass.

Design:
    - Per-verb builder functions extract known keys from the dict
    - Type coercion for common mismatches (string -> int, string -> bool)
    - Missing optional fields get their dataclass defaults
    - Missing required fields produce a human-readable error string
    - Unknown keys are silently ignored (forward compatibility)
    - Never raises exceptions -- returns either *Args or error string

Usage::

    from jules_daemon.cli.args_builder import build_verb_args

    # From NL classification
    args = build_verb_args("run", {
        "target_host": "staging",
        "target_user": "ci",
        "natural_language": "run smoke tests",
    })
    # -> RunArgs(target_host="staging", target_user="ci", ...)

    # From structured classification (also works)
    args = build_verb_args("status", {"verbose": True})
    # -> StatusArgs(verbose=True)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jules_daemon.cli.verbs import (
    CancelArgs,
    HistoryArgs,
    QueueArgs,
    RunArgs,
    StatusArgs,
    VerbArgs,
    WatchArgs,
)

__all__ = [
    "build_verb_args",
]


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------


def _coerce_bool(value: Any) -> bool:
    """Coerce a value to bool, handling common string representations.

    Treats "true", "yes", "1" (case-insensitive) as True.
    Treats "false", "no", "0", empty string as False.
    Falls back to Python's truthiness for other types.

    Args:
        value: Value to coerce.

    Returns:
        Boolean interpretation of the value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _coerce_int(value: Any, field_name: str) -> int | str:
    """Coerce a value to int, returning an error string on failure.

    Args:
        value: Value to coerce (typically int or str).
        field_name: Field name for error messages.

    Returns:
        Integer on success, or a human-readable error string on failure.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return int(stripped)
        except ValueError:
            return f"{field_name} must be a number, got {stripped!r}"
    if isinstance(value, float):
        return int(value)
    return f"{field_name} must be a number, got {type(value).__name__}"


def _coerce_optional_str(value: Any) -> str | None:
    """Coerce a value to an optional string.

    Returns None for None, empty strings, and whitespace-only strings.
    Strips leading/trailing whitespace from non-empty strings.

    Args:
        value: Value to coerce.

    Returns:
        Stripped string or None.
    """
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


# ---------------------------------------------------------------------------
# Per-verb builder functions
# ---------------------------------------------------------------------------


def _build_status_args(extracted: dict[str, Any]) -> StatusArgs | str:
    """Build StatusArgs from an extracted_args dict.

    Optional fields:
        verbose (bool): Default False.
    """
    verbose = _coerce_bool(extracted.get("verbose", False))
    try:
        return StatusArgs(verbose=verbose)
    except ValueError as exc:
        return str(exc)


def _build_watch_args(extracted: dict[str, Any]) -> WatchArgs | str:
    """Build WatchArgs from an extracted_args dict.

    Optional fields:
        run_id (str | None): Default None.
        tail_lines (int): Default 50.
        follow (bool): Default False.
        output_format (str): Default "text". Must be one of text, json, summary.
    """
    run_id = _coerce_optional_str(extracted.get("run_id"))

    raw_tail = extracted.get("tail_lines", 50)
    tail_result = _coerce_int(raw_tail, "tail_lines")
    if isinstance(tail_result, str):
        return tail_result

    follow = _coerce_bool(extracted.get("follow", False))

    output_format = str(extracted.get("output_format", "text")).strip()
    if not output_format:
        output_format = "text"

    try:
        return WatchArgs(
            run_id=run_id,
            tail_lines=tail_result,
            follow=follow,
            output_format=output_format,
        )
    except ValueError as exc:
        return str(exc)


def _extract_ssh_nl_fields(
    extracted: dict[str, Any],
) -> tuple[str, str, str, int, str | None] | str:
    """Extract and validate common SSH target + NL fields.

    Shared by ``_build_run_args`` and ``_build_queue_args`` to avoid
    duplication. Mirrors the approach in ``parser.py::_parse_target_and_nl``.

    Returns:
        Tuple of (host, user, natural_language, port, key_path) on success,
        or an error string on failure.
    """
    target_host = _coerce_optional_str(extracted.get("target_host"))
    if target_host is None:
        return "Missing required field: target_host"

    target_user = _coerce_optional_str(extracted.get("target_user"))
    if target_user is None:
        return "Missing required field: target_user"

    natural_language = _coerce_optional_str(extracted.get("natural_language"))
    if natural_language is None:
        return "Missing required field: natural_language"

    raw_port = extracted.get("target_port", 22)
    port_result = _coerce_int(raw_port, "target_port")
    if isinstance(port_result, str):
        return port_result

    key_path = _coerce_optional_str(extracted.get("key_path"))

    return (target_host, target_user, natural_language, port_result, key_path)


def _build_run_args(extracted: dict[str, Any]) -> RunArgs | str:
    """Build RunArgs from an extracted_args dict.

    Required fields:
        natural_language (str): What tests to run.

    Optional fields:
        system_name (str): Named system alias defined in the wiki.
        target_host (str): Remote hostname.
        target_user (str): SSH username.
        target_port (int): Default 22.
        key_path (str | None): Default None.
    """
    system_name = _coerce_optional_str(extracted.get("system_name"))
    if system_name is not None:
        natural_language = _coerce_optional_str(extracted.get("natural_language"))
        if natural_language is None:
            return "Missing required field: natural_language"
        try:
            return RunArgs(
                natural_language=natural_language,
                system_name=system_name,
            )
        except ValueError as exc:
            return str(exc)

    infer_target = extracted.get("infer_target") is True
    if infer_target:
        natural_language = _coerce_optional_str(extracted.get("natural_language"))
        if natural_language is None:
            return "Missing required field: natural_language"
        try:
            return RunArgs(
                natural_language=natural_language,
                infer_target=True,
            )
        except ValueError as exc:
            return str(exc)

    interpret_request = extracted.get("interpret_request") is True
    if interpret_request:
        natural_language = _coerce_optional_str(extracted.get("natural_language"))
        if natural_language is None:
            return "Missing required field: natural_language"
        try:
            return RunArgs(
                natural_language=natural_language,
                interpret_request=True,
            )
        except ValueError as exc:
            return str(exc)

    fields = _extract_ssh_nl_fields(extracted)
    if isinstance(fields, str):
        return fields
    host, user, nl, port, key_path = fields

    try:
        return RunArgs(
            target_host=host,
            target_user=user,
            natural_language=nl,
            target_port=port,
            key_path=key_path,
        )
    except ValueError as exc:
        return str(exc)


def _build_queue_args(extracted: dict[str, Any]) -> QueueArgs | str:
    """Build QueueArgs from an extracted_args dict.

    Required fields:
        target_host (str): Remote hostname.
        target_user (str): SSH username.
        natural_language (str): What tests to run.

    Optional fields:
        target_port (int): Default 22.
        key_path (str | None): Default None.
        priority (int): Default 0.
    """
    fields = _extract_ssh_nl_fields(extracted)
    if isinstance(fields, str):
        return fields
    host, user, nl, port, key_path = fields

    raw_priority = extracted.get("priority", 0)
    priority_result = _coerce_int(raw_priority, "priority")
    if isinstance(priority_result, str):
        return priority_result

    try:
        return QueueArgs(
            target_host=host,
            target_user=user,
            natural_language=nl,
            target_port=port,
            key_path=key_path,
            priority=priority_result,
        )
    except ValueError as exc:
        return str(exc)


def _build_cancel_args(extracted: dict[str, Any]) -> CancelArgs | str:
    """Build CancelArgs from an extracted_args dict.

    Optional fields:
        run_id (str | None): Default None.
        force (bool): Default False.
        reason (str | None): Default None.
    """
    run_id = _coerce_optional_str(extracted.get("run_id"))
    force = _coerce_bool(extracted.get("force", False))
    reason = _coerce_optional_str(extracted.get("reason"))

    try:
        return CancelArgs(run_id=run_id, force=force, reason=reason)
    except ValueError as exc:
        return str(exc)


def _build_history_args(extracted: dict[str, Any]) -> HistoryArgs | str:
    """Build HistoryArgs from an extracted_args dict.

    Optional fields:
        limit (int): Default 20.
        status_filter (str | None): Default None.
        host_filter (str | None): Default None.
        verbose (bool): Default False.
    """
    raw_limit = extracted.get("limit", 20)
    limit_result = _coerce_int(raw_limit, "limit")
    if isinstance(limit_result, str):
        return limit_result

    status_filter = _coerce_optional_str(extracted.get("status_filter"))
    host_filter = _coerce_optional_str(extracted.get("host_filter"))
    verbose = _coerce_bool(extracted.get("verbose", False))

    try:
        return HistoryArgs(
            limit=limit_result,
            status_filter=status_filter,
            host_filter=host_filter,
            verbose=verbose,
        )
    except ValueError as exc:
        return str(exc)


# ---------------------------------------------------------------------------
# Verb -> builder dispatch table
# ---------------------------------------------------------------------------


# Type alias for per-verb builder functions.
_BuilderFn = Callable[[dict[str, Any]], VerbArgs | str]

_VERB_BUILDERS: dict[str, _BuilderFn] = {
    "status": _build_status_args,
    "watch": _build_watch_args,
    "run": _build_run_args,
    "queue": _build_queue_args,
    "cancel": _build_cancel_args,
    "history": _build_history_args,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_verb_args(
    canonical_verb: str,
    extracted_args: dict[str, Any],
) -> VerbArgs | str:
    """Build a typed *Args dataclass from a canonical verb and extracted args.

    This is the central bridge between the input classifier's untyped
    output and the handler registry's typed input. Both the structured
    path (CLI parser) and the NL path (heuristic extractor / LLM
    classifier) produce ``extracted_args`` dicts that this function
    converts into identical ``*Args`` dataclasses.

    The function never raises exceptions. Invalid input produces a
    human-readable error string instead.

    Args:
        canonical_verb: One of the six canonical verb strings
            (status, watch, run, queue, cancel, history).
        extracted_args: Dictionary of extracted parameters. Keys
            depend on the verb. Unknown keys are ignored.

    Returns:
        The appropriate ``*Args`` frozen dataclass on success,
        or a human-readable error string on failure.
    """
    normalized = canonical_verb.strip().lower() if canonical_verb else ""
    if not normalized:
        return "Verb must not be empty"

    builder = _VERB_BUILDERS.get(normalized)
    if builder is None:
        valid = ", ".join(sorted(_VERB_BUILDERS.keys()))
        return f"Unknown canonical verb {canonical_verb!r}. Valid verbs: {valid}"

    return builder(extracted_args)
