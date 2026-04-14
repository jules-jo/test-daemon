"""CLI verb parser: tokenize, match, extract, validate.

Parses raw user input strings into structured ``ParsedCommand`` instances
or ``ParseError`` results. This is the main entry point for converting
CLI text into typed, validated daemon commands.

Pipeline:
    1. Tokenize the raw input (shell-style quoting via ``shlex``)
    2. Match the first token against both exact verbs and the
       canonical verb registry (40+ aliases normalized to 6 verbs)
    3. Dispatch to a per-verb argument parser
    4. Validate and construct the appropriate ``*Args`` dataclass
    5. Return ``ParsedCommand`` or ``ParseError``

The parser never raises exceptions for invalid user input -- it always
returns a ``ParseError`` with a human-readable message. This makes it
safe to use as the front-end of an IPC handler where exceptions would
be difficult to serialize.

Verb normalization: the parser first tries direct ``Verb`` enum matching,
then falls back to ``resolve_canonical_verb()`` from the verb registry.
This means aliases like ``execute``, ``stop``, ``tail``, ``enqueue``,
``check``, and ``past`` all resolve to their canonical verbs.

Note on negative numbers: tokens starting with ``-`` are treated as
flags, so ``--tail -5`` is parsed as two flags (``--tail`` and ``-5``),
not a flag with a negative value. Use ``--tail=-5`` to pass negative
values via the ``=`` syntax.

Usage::

    from jules_daemon.cli.parser import parse_command

    result = parse_command("run deploy@staging run the regression tests")
    if isinstance(result, ParsedCommand):
        print(f"verb={result.verb}, args={result.args}")
    else:
        print(f"Error: {result.message}")

    # Alias-based input works identically:
    result = parse_command("execute deploy@staging run the regression tests")
    # result.verb == Verb.RUN

    # Deterministic classification (pre-LLM):
    from jules_daemon.cli.parser import classify_structured_command
    classification = classify_structured_command("stop --run-id abc-123")
    # classification.canonical_verb == "cancel", confidence_score == 0.9
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from jules_daemon.classifier.models import ClassificationResult, InputType
from jules_daemon.classifier.verb_registry import resolve_canonical_verb
from jules_daemon.cli.verbs import (
    CancelArgs,
    HistoryArgs,
    ParsedCommand,
    QueueArgs,
    RunArgs,
    StatusArgs,
    Verb,
    WatchArgs,
    parse_verb,
)
from jules_daemon.cli.watch_parser import parse_watch_tokens

__all__ = [
    "ParseError",
    "classify_structured_command",
    "normalize_verb",
    "parse_command",
    "tokenize",
]

# Mapping from canonical verb string -> Verb enum member.
# Used to bridge the verb registry (string-based) to the Verb enum.
_CANONICAL_TO_VERB: dict[str, Verb] = {v.value: v for v in Verb}


# ---------------------------------------------------------------------------
# ParseError
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParseError:
    """Structured error result from verb parsing.

    Returned instead of raising exceptions so callers (especially IPC
    handlers) can serialize the error without try/except ceremony.

    Attributes:
        message: Human-readable description of what went wrong.
        raw_input: The original unmodified input string.
        verb: The recognized verb, if parsing got that far. None when
            the verb itself was unrecognized or missing.
    """

    message: str
    raw_input: str
    verb: Verb | None = None

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def tokenize(raw: str) -> list[str]:
    """Tokenize a raw input string using POSIX shell quoting rules.

    Handles single-quoted strings, double-quoted strings, and escaped
    characters. Whitespace between tokens is collapsed. Delegates to
    ``shlex.split``, which raises ``ValueError`` for unterminated quotes.

    Args:
        raw: Raw user input string.

    Returns:
        List of tokens. Empty list for empty/whitespace-only input.

    Raises:
        ValueError: If the input contains unterminated quotes.
    """
    stripped = raw.strip()
    if not stripped:
        return []

    return shlex.split(stripped)


# ---------------------------------------------------------------------------
# Verb normalization (bridges verb registry to Verb enum)
# ---------------------------------------------------------------------------


def normalize_verb(raw: str) -> Verb | None:
    """Normalize a raw verb token to a Verb enum via the canonical registry.

    Lookup strategy:
    1. Try direct ``Verb`` enum match (exact canonical verbs).
    2. Fall back to ``resolve_canonical_verb()`` from the verb registry
       (handles 40+ aliases like execute->run, stop->cancel, etc.).
    3. Return None if neither resolves.

    This function never raises for unrecognized input.

    Args:
        raw: Raw verb token from user input.

    Returns:
        The resolved ``Verb`` enum member, or None if unrecognized.
    """
    # Fast path: try exact Verb enum match first
    try:
        return parse_verb(raw)
    except ValueError:
        pass

    # Slow path: resolve via verb registry alias table
    canonical = resolve_canonical_verb(raw)
    if canonical is None:
        return None

    return _CANONICAL_TO_VERB.get(canonical)


# ---------------------------------------------------------------------------
# Internal helpers: flag parsing
# ---------------------------------------------------------------------------


def _split_flags_and_positionals(
    tokens: list[str],
) -> tuple[list[str], dict[str, str | None]]:
    """Split tokens into positional args and flag key-value pairs.

    Flags start with ``-`` or ``--``. A flag followed by a non-flag
    token consumes that token as its value. Boolean flags (no value)
    get ``None`` as their value.

    Supports ``--flag=value`` syntax by splitting on the first ``=``.

    Note: Tokens starting with ``-`` are always treated as flags, so
    negative numeric values must use ``--flag=-5`` syntax.

    Args:
        tokens: Token list (verb already removed).

    Returns:
        Tuple of (positional_tokens, flag_dict).
        Flag keys are lowercased with leading dashes preserved.
    """
    positionals: list[str] = []
    flags: dict[str, str | None] = {}

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.startswith("-"):
            # Handle --flag=value syntax
            if "=" in token:
                key, _, value = token.partition("=")
                flags[key.lower()] = value
                i += 1
                continue

            key = token.lower()
            # Check if next token is a value (not a flag itself)
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                flags[key] = tokens[i + 1]
                i += 2
            else:
                flags[key] = None
                i += 1
        else:
            positionals.append(token)
            i += 1

    return positionals, flags


def _check_unknown_flags(
    flags: dict[str, str | None],
    known: frozenset[str],
    verb_name: str,
) -> str | None:
    """Return an error message if any unknown flags are present.

    Args:
        flags: Parsed flags dictionary.
        known: Set of recognized flag keys.
        verb_name: Verb name for the error message.

    Returns:
        Error message string, or None if all flags are known.
    """
    unknown = set(flags.keys()) - known
    if unknown:
        return f"Unknown flag(s) for {verb_name}: {', '.join(sorted(unknown))}"
    return None


def _require_flag_value(
    flags: dict[str, str | None],
    key: str,
    flag_name: str,
) -> str | None:
    """Extract a required-value flag, returning None if not present.

    Args:
        flags: The parsed flags dictionary.
        key: The normalized flag key (e.g., ``--run-id``).
        flag_name: Human-readable name for error messages.

    Returns:
        The flag value, or None if the flag is not present.

    Raises:
        ValueError: If the flag is present but has no value.
    """
    if key not in flags:
        return None
    value = flags[key]
    if value is None:
        raise ValueError(f"{flag_name} requires a value")
    return value


def _parse_int_flag(
    flags: dict[str, str | None],
    key: str,
    default: int,
) -> int | str:
    """Parse an integer flag value, returning a default when absent.

    Args:
        flags: Parsed flags dictionary.
        key: The normalized flag key (e.g., ``--port``).
        default: Value to return when the flag is not present.

    Returns:
        Parsed integer on success, or an error message string on failure.
    """
    if key not in flags:
        return default
    value = flags[key]
    if value is None:
        return f"{key} requires a value"
    try:
        return int(value)
    except ValueError:
        return f"{key} must be a number, got {value!r}"


# ---------------------------------------------------------------------------
# Internal helpers: SSH target parsing
# ---------------------------------------------------------------------------


def _parse_ssh_target(
    raw: str,
) -> tuple[str, str, int] | str:
    """Parse a ``user@host[:port]`` SSH target string.

    Args:
        raw: String in ``user@host`` or ``user@host:port`` format.

    Returns:
        Tuple of (user, host, port) on success, or an error message string.
    """
    if "@" not in raw:
        return f"Invalid SSH target {raw!r}: expected user@host[:port] format"

    user_part, _, host_part = raw.partition("@")
    if not user_part:
        return f"Invalid SSH target {raw!r}: user must not be empty"

    # Check for port
    if ":" in host_part:
        host, _, port_str = host_part.partition(":")
        if not host:
            return f"Invalid SSH target {raw!r}: host must not be empty"
        try:
            port = int(port_str)
        except ValueError:
            return f"Invalid SSH target {raw!r}: port {port_str!r} is not a number"
        if not (1 <= port <= 65535):
            return (
                f"Invalid SSH target {raw!r}: port must be 1-65535, got {port}"
            )
    else:
        host = host_part
        port = 22
        if not host:
            return f"Invalid SSH target {raw!r}: host must not be empty"

    return (user_part, host, port)


def _parse_target_and_nl(
    positionals: list[str],
    flags: dict[str, str | None],
    verb_name: str,
) -> tuple[str, str, int, str | None, str] | str:
    """Parse SSH target, natural language, port override, and key path.

    Shared logic for the ``run`` and ``queue`` verbs.

    Args:
        positionals: Positional tokens (target + NL words).
        flags: Parsed flags dictionary (may contain --port, --key).
        verb_name: Verb name for error messages.

    Returns:
        Tuple of (user, host, port, key_path, natural_language) on success,
        or an error message string on failure.
    """
    if not positionals:
        return f"{verb_name} requires a target in user@host[:port] format"

    target_result = _parse_ssh_target(positionals[0])
    if isinstance(target_result, str):
        return target_result
    user, host, port = target_result

    nl_parts = positionals[1:]
    if not nl_parts:
        return f"{verb_name} requires a natural-language command after the target"
    natural_language = " ".join(nl_parts)

    # Override port from flag
    port_result = _parse_int_flag(flags, "--port", port)
    if isinstance(port_result, str):
        return port_result
    port = port_result

    # Key path
    try:
        key_path = _require_flag_value(flags, "--key", "--key")
    except ValueError as exc:
        return str(exc)

    return (user, host, port, key_path, natural_language)


# ---------------------------------------------------------------------------
# Per-verb parsers
# ---------------------------------------------------------------------------


def _parse_status_args(tokens: list[str]) -> StatusArgs | str:
    """Parse arguments for the ``status`` verb.

    Recognized flags: --verbose / -v
    """
    _, flags = _split_flags_and_positionals(tokens)

    err = _check_unknown_flags(flags, frozenset({"--verbose", "-v"}), "status")
    if err is not None:
        return err

    verbose = "--verbose" in flags or "-v" in flags
    return StatusArgs(verbose=verbose)


def _parse_watch_args(tokens: list[str]) -> WatchArgs | str:
    """Parse arguments for the ``watch`` verb.

    Delegates to the argparse-based ``parse_watch_tokens()`` from
    ``watch_parser.py``. Supports:
        --run-id <id>    Target a specific job/run by ID
        --follow / -f    Continuously stream output
        --format <fmt>   Output format (text, json, summary)
        --tail <n>       Number of recent lines to show
    """
    return parse_watch_tokens(tokens)


def _parse_run_args(tokens: list[str]) -> RunArgs | str:
    """Parse arguments for the ``run`` verb.

    Syntax:
      run user@host[:port] <natural language> [--port N] [--key PATH]
      run --system NAME <natural language>
      run --infer-target <natural language>
    """
    positionals, flags = _split_flags_and_positionals(tokens)

    err = _check_unknown_flags(
        flags,
        frozenset({"--port", "--key", "--system", "--infer-target"}),
        "run",
    )
    if err is not None:
        return err

    try:
        system_name = _require_flag_value(flags, "--system", "--system")
    except ValueError as exc:
        return str(exc)
    infer_target = "--infer-target" in flags

    # ``_split_flags_and_positionals()`` is intentionally generic and treats a
    # non-flag token after any flag as that flag's value. ``--infer-target`` is
    # a boolean flag, so if the user writes ``run --infer-target run tests`` the
    # leading ``run`` would otherwise be swallowed. Put it back into the NL
    # positionals for this verb-specific boolean flag.
    infer_target_captured = flags.get("--infer-target")
    if infer_target and infer_target_captured is not None:
        positionals = [infer_target_captured, *positionals]
        flags["--infer-target"] = None

    if system_name is not None:
        if infer_target:
            return "--system cannot be combined with --infer-target"
        if "--port" in flags or "--key" in flags:
            return "--system cannot be combined with --port or --key"
        if not positionals:
            return "run requires a natural-language command after --system"
        if "@" in positionals[0]:
            return "run cannot combine a user@host target with --system"
        natural_language = " ".join(positionals)
        try:
            return RunArgs(
                natural_language=natural_language,
                system_name=system_name,
            )
        except ValueError as exc:
            return str(exc)

    if infer_target:
        if "--port" in flags or "--key" in flags:
            return "--infer-target cannot be combined with --port or --key"
        if not positionals:
            return "run requires a natural-language command after --infer-target"
        if "@" in positionals[0]:
            return "run cannot combine a user@host target with --infer-target"
        natural_language = " ".join(positionals)
        try:
            return RunArgs(
                natural_language=natural_language,
                infer_target=True,
            )
        except ValueError as exc:
            return str(exc)

    parsed = _parse_target_and_nl(positionals, flags, "run")
    if isinstance(parsed, str):
        return parsed
    user, host, port, key_path, natural_language = parsed

    try:
        return RunArgs(
            target_host=host,
            target_user=user,
            natural_language=natural_language,
            target_port=port,
            key_path=key_path,
        )
    except ValueError as exc:
        return str(exc)


def _parse_queue_args(tokens: list[str]) -> QueueArgs | str:
    """Parse arguments for the ``queue`` verb.

    Syntax: queue user@host[:port] <NL> [--port N] [--key PATH] [--priority N]
    """
    positionals, flags = _split_flags_and_positionals(tokens)

    err = _check_unknown_flags(
        flags, frozenset({"--port", "--key", "--priority"}), "queue",
    )
    if err is not None:
        return err

    parsed = _parse_target_and_nl(positionals, flags, "queue")
    if isinstance(parsed, str):
        return parsed
    user, host, port, key_path, natural_language = parsed

    priority_result = _parse_int_flag(flags, "--priority", 0)
    if isinstance(priority_result, str):
        return priority_result

    try:
        return QueueArgs(
            target_host=host,
            target_user=user,
            natural_language=natural_language,
            target_port=port,
            key_path=key_path,
            priority=priority_result,
        )
    except ValueError as exc:
        return str(exc)


def _parse_cancel_args(tokens: list[str]) -> CancelArgs | str:
    """Parse arguments for the ``cancel`` verb.

    Recognized flags: --run-id <id>, --force / -f, --reason <text>
    """
    _, flags = _split_flags_and_positionals(tokens)

    err = _check_unknown_flags(
        flags, frozenset({"--run-id", "--force", "-f", "--reason"}), "cancel",
    )
    if err is not None:
        return err

    try:
        run_id = _require_flag_value(flags, "--run-id", "--run-id")
    except ValueError as exc:
        return str(exc)

    force = "--force" in flags or "-f" in flags

    try:
        reason = _require_flag_value(flags, "--reason", "--reason")
    except ValueError as exc:
        return str(exc)

    try:
        return CancelArgs(run_id=run_id, force=force, reason=reason)
    except ValueError as exc:
        return str(exc)


def _parse_history_args(tokens: list[str]) -> HistoryArgs | str:
    """Parse arguments for the ``history`` verb.

    Recognized flags: --limit <n>, --status <s>, --host <h>, --verbose / -v
    """
    _, flags = _split_flags_and_positionals(tokens)

    err = _check_unknown_flags(
        flags,
        frozenset({"--limit", "--status", "--host", "--verbose", "-v"}),
        "history",
    )
    if err is not None:
        return err

    verbose = "--verbose" in flags or "-v" in flags

    limit_result = _parse_int_flag(flags, "--limit", 20)
    if isinstance(limit_result, str):
        return limit_result

    try:
        status_filter = _require_flag_value(flags, "--status", "--status")
    except ValueError as exc:
        return str(exc)

    try:
        host_filter = _require_flag_value(flags, "--host", "--host")
    except ValueError as exc:
        return str(exc)

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
# Verb -> parser dispatch table
# ---------------------------------------------------------------------------

_VERB_PARSERS = {
    Verb.STATUS: _parse_status_args,
    Verb.WATCH: _parse_watch_args,
    Verb.RUN: _parse_run_args,
    Verb.QUEUE: _parse_queue_args,
    Verb.CANCEL: _parse_cancel_args,
    Verb.HISTORY: _parse_history_args,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_command(raw: str) -> ParsedCommand | ParseError:
    """Parse a raw CLI input string into a validated ParsedCommand.

    This is the main entry point for the CLI verb parser. It tokenizes
    the input, normalizes the first token against both exact verb names
    and the canonical verb registry (40+ aliases), dispatches to the
    appropriate per-verb argument parser, and returns a structured result.

    The function never raises exceptions for invalid user input. All
    errors are returned as ``ParseError`` instances with descriptive
    messages.

    Args:
        raw: Raw user input string. Supports both exact verbs
            (e.g., ``"run deploy@host run tests"``) and aliases
            (e.g., ``"execute deploy@host run tests"``).

    Returns:
        ``ParsedCommand`` on success, ``ParseError`` on any failure.
    """
    try:
        tokens = tokenize(raw)
    except ValueError as exc:
        return ParseError(
            message=f"Tokenization error: {exc}",
            raw_input=raw,
        )

    if not tokens:
        return ParseError(
            message="Empty input: expected a verb "
            "(status, watch, run, queue, cancel, history)",
            raw_input=raw,
        )

    verb_token = tokens[0]

    # Normalize the verb through the canonical registry.
    # This handles both exact verbs (run, status) and aliases
    # (execute->run, stop->cancel, tail->watch, etc.).
    verb = normalize_verb(verb_token)
    if verb is None:
        valid = ", ".join(sorted(v.value for v in Verb))
        return ParseError(
            message=(
                f"Unknown verb {verb_token.strip()!r}. "
                f"Valid verbs: {valid}"
            ),
            raw_input=raw,
        )

    parser_fn = _VERB_PARSERS[verb]
    result = parser_fn(tokens[1:])

    if isinstance(result, str):
        return ParseError(message=result, raw_input=raw, verb=verb)

    return ParsedCommand(verb=verb, args=result)


# ---------------------------------------------------------------------------
# Deterministic structured classification
# ---------------------------------------------------------------------------


def classify_structured_command(raw: str) -> ClassificationResult | None:
    """Deterministic pre-LLM classification of structured verb-style input.

    Pattern-matches the raw input against the verb registry, extracts
    positional and keyword arguments, and returns a ``ClassificationResult``
    with a confidence score. This is the fast path that avoids LLM calls
    for well-formed structured commands.

    Confidence scoring:
    - 1.0: Exact canonical verb match (e.g., ``run``, ``status``)
    - 0.9: Alias match via verb registry (e.g., ``execute`` -> ``run``)

    Returns None when the input cannot be classified deterministically
    (empty input, unterminated quotes, or unrecognized verb). The caller
    should fall through to the LLM-powered IntentClassifier.

    Args:
        raw: Raw user input string to classify.

    Returns:
        ``ClassificationResult`` if deterministically classified,
        None if the input should be forwarded to the LLM classifier.
    """
    try:
        tokens = tokenize(raw)
    except ValueError:
        return None

    if not tokens:
        return None

    verb_token = tokens[0]
    remaining_tokens = tokens[1:]

    # Determine confidence based on match type
    is_exact = _is_exact_canonical_verb(verb_token)
    canonical = _resolve_to_canonical_string(verb_token)

    if canonical is None:
        return None

    confidence = 1.0 if is_exact else 0.9

    # Extract positional and keyword arguments from remaining tokens
    positionals, flags = _split_flags_and_positionals(remaining_tokens)

    # Build the extracted args dict with separated positionals and keywords
    extracted_args: dict[str, Any] = {
        "positional_args": tuple(positionals),
        "keyword_args": _flags_to_serializable(flags),
    }

    return ClassificationResult(
        canonical_verb=canonical,
        extracted_args=extracted_args,
        confidence_score=confidence,
        input_type=InputType.COMMAND,
    )


def _is_exact_canonical_verb(token: str) -> bool:
    """Check if the token is an exact canonical verb (not an alias).

    Args:
        token: Raw verb token.

    Returns:
        True if the token exactly matches one of the 6 canonical verbs.
    """
    try:
        parse_verb(token)
        return True
    except ValueError:
        return False


def _resolve_to_canonical_string(token: str) -> str | None:
    """Resolve a token to its canonical verb string.

    Tries exact match first, then alias resolution.

    Args:
        token: Raw verb token.

    Returns:
        Canonical verb string (e.g., "run"), or None.
    """
    # Try exact Verb enum match
    try:
        verb = parse_verb(token)
        return verb.value
    except ValueError:
        pass

    # Try verb registry alias
    return resolve_canonical_verb(token)


def _flags_to_serializable(
    flags: dict[str, str | None],
) -> dict[str, str | None]:
    """Create a new dict from flags (immutability).

    Args:
        flags: Parsed flags dictionary.

    Returns:
        New dict copy of the flags.
    """
    return dict(flags)
