"""Jules CLI client entry point.

Connects to the daemon via the thin client and either sends a single
command (when arguments are provided) or enters an interactive REPL
(when no arguments are given).

Usage::

    # Interactive mode
    jules

    # Single command mode
    jules status
    jules run deploy@staging run the smoke tests
    jules watch --follow
    jules history --limit 10
    jules cancel --force
"""

from __future__ import annotations

import asyncio
import difflib
import sys
from pathlib import Path

from jules_daemon.classifier.classify import classify
from jules_daemon.classifier.models import InputType
from jules_daemon.ipc.framing import MessageEnvelope
from jules_daemon.thin_client.client import ThinClient, ThinClientConfig
from jules_daemon.thin_client.renderer import render_confirm_prompt

_PROMPT = "jules> "
_QUIT_COMMANDS = frozenset({"quit", "exit", "q", "c"})

# Verb routing table: maps user-facing verb strings to thin client methods.
_VERB_MAP = frozenset({
    "status", "history", "cancel", "run", "watch", "health", "discover",
})

# Ordered tuple of verbs used for fuzzy typo correction. Kept as a
# tuple because difflib.get_close_matches requires a sequence and we
# want deterministic iteration order for stable match ranking.
_KNOWN_VERBS: tuple[str, ...] = (
    "run",
    "status",
    "watch",
    "history",
    "cancel",
    "queue",
    "health",
    "discover",
)

# Minimum number of characters in the first token before we attempt
# fuzzy matching. Single-character tokens are too short for difflib
# similarity ratios to be meaningful and would produce noisy matches.
_FUZZY_MIN_LENGTH: int = 2

# difflib cutoff for fuzzy verb matching. Tuned to catch common typos
# (wajch, statuz, histroy, helath) without matching unrelated words.
_FUZZY_CUTOFF: float = 0.6

# Minimum classifier confidence required before we trust a natural
# language interpretation. Matches the classifier's own internal
# threshold for "confident" results.
_CLASSIFIER_CONFIDENCE_THRESHOLD: float = 0.7


# ---------------------------------------------------------------------------
# Confirmation callback
# ---------------------------------------------------------------------------


def _interactive_confirm(envelope: MessageEnvelope) -> tuple[bool, str | None]:
    """Prompt the user to approve, deny, edit, or answer a question.

    Handles two types of CONFIRM_PROMPT:
    1. SSH command approval: shows command, asks [A]pprove/[D]eny/[E]dit
    2. Question: shows the question, reads a free-text answer

    The prompt type is determined by the payload's 'type' field.

    Args:
        envelope: The CONFIRM_PROMPT envelope from the daemon.

    Returns:
        (approved, answer_or_edit) tuple.
        For SSH approvals: (True/False, edited_command or None)
        For questions: (True, user's text answer) or (False, None) if cancelled
    """
    payload = envelope.payload
    prompt_type = payload.get("type", "")

    # Handle question-type prompts (from ask_user_question tool)
    if prompt_type == "question":
        question = payload.get("question") or payload.get("message") or "?"
        context = payload.get("context", "")
        print()
        if context:
            print(f"  {context}")
        print(f"  {question}")
        print()
        try:
            answer = input("  Your answer (or 'skip' to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False, None
        if not answer or answer.lower() == "skip":
            return False, None
        # Return the answer as the "edited_command" field -- the IPC
        # bridge reads it from payload["answer"] or payload["text"]
        return True, answer

    # Standard SSH command approval
    prompt_text = render_confirm_prompt(envelope)
    print(prompt_text)

    original_command = (
        payload.get("proposed_command")
        or payload.get("command")
        or ""
    )

    try:
        answer = input("[A]pprove / [D]eny / [E]dit? ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False, None

    if answer in ("a", "approve", "y", "yes"):
        return True, None

    if answer in ("e", "edit"):
        print(f"Current: {original_command}")
        try:
            edited = input("New command: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False, None
        if not edited:
            print("Empty command, denying.")
            return False, None
        return True, edited

    # Anything else: deny
    return False, None


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------


async def _execute_single(
    client: ThinClient,
    raw_args: list[str],
) -> int:
    """Execute a single command and return the exit code.

    Parses the verb from the first argument and delegates to the
    appropriate thin client method.

    Args:
        client: Connected thin client instance.
        raw_args: Command arguments (verb + optional params).

    Returns:
        0 on success, 1 on failure.
    """
    if not raw_args:
        result = await client.status()
        print(result.rendered)
        return 0 if result.success else 1

    verb = raw_args[0].lower()
    remaining = raw_args[1:]

    if verb == "health":
        result = await client.health()

    elif verb == "status":
        verbose = "--verbose" in remaining or "-v" in remaining
        result = await client.status(verbose=verbose)

    elif verb == "history":
        limit = 20
        status_filter = None
        host_filter = None
        i = 0
        while i < len(remaining):
            if remaining[i] == "--limit" and i + 1 < len(remaining):
                limit = int(remaining[i + 1])
                i += 2
            elif remaining[i] == "--status" and i + 1 < len(remaining):
                status_filter = remaining[i + 1]
                i += 2
            elif remaining[i] == "--host" and i + 1 < len(remaining):
                host_filter = remaining[i + 1]
                i += 2
            else:
                i += 1
        result = await client.history(
            limit=limit,
            status_filter=status_filter,
            host_filter=host_filter,
        )

    elif verb == "cancel":
        force = "--force" in remaining or "-f" in remaining
        run_id = None
        reason = None
        filtered = [a for a in remaining if a not in ("--force", "-f")]
        i = 0
        while i < len(filtered):
            if filtered[i] == "--run-id" and i + 1 < len(filtered):
                run_id = filtered[i + 1]
                i += 2
            elif filtered[i] == "--reason" and i + 1 < len(filtered):
                reason = filtered[i + 1]
                i += 2
            else:
                i += 1
        result = await client.cancel(
            run_id=run_id,
            force=force,
            reason=reason,
        )

    elif verb == "run":
        # Parse: run user@host <natural language description>
        if not remaining:
            print("Usage: jules run user@host <description>")
            return 1

        target_spec = remaining[0]
        nl_parts = remaining[1:]

        if "@" not in target_spec:
            print("Error: target must be in user@host format")
            return 1

        user, host_part = target_spec.split("@", 1)
        port = 22
        if ":" in host_part:
            host, port_str = host_part.rsplit(":", 1)
            port = int(port_str)
        else:
            host = host_part

        if not nl_parts:
            print("Error: natural language description required")
            return 1

        natural_language = " ".join(nl_parts)
        result = await client.run(
            target_host=host,
            target_user=user,
            natural_language=natural_language,
            target_port=port,
        )

    elif verb == "discover":
        # discover user@host command [args...]
        if not remaining:
            print("Usage: jules discover user@host command [args...]")
            return 1

        target_spec = remaining[0]
        command_parts = remaining[1:]

        if "@" not in target_spec:
            print("Error: target must be in user@host format")
            return 1

        user, host_part = target_spec.split("@", 1)
        port = 22
        if ":" in host_part:
            host, port_str = host_part.rsplit(":", 1)
            port = int(port_str)
        else:
            host = host_part

        if not command_parts:
            print("Error: command to discover is required")
            return 1

        command_str = " ".join(command_parts)
        result = await client.discover(
            target_host=host,
            target_user=user,
            command=command_str,
            target_port=port,
        )

    elif verb == "watch":
        run_id = None
        tail_lines = 50
        i = 0
        while i < len(remaining):
            if remaining[i] == "--run-id" and i + 1 < len(remaining):
                run_id = remaining[i + 1]
                i += 2
            elif remaining[i] == "--tail" and i + 1 < len(remaining):
                tail_lines = int(remaining[i + 1])
                i += 2
            else:
                i += 1
        result = await client.watch(
            run_id=run_id,
            tail_lines=tail_lines,
            on_line=lambda line: print(line, end=""),
        )

    else:
        print(f"Unknown command: {verb}")
        print(f"Available commands: {', '.join(sorted(_VERB_MAP))}")
        return 1

    print(result.rendered)
    return 0 if result.success else 1


# ---------------------------------------------------------------------------
# Hybrid verb resolution
# ---------------------------------------------------------------------------


def _fuzzy_match_verb(first_word: str) -> str | None:
    """Attempt to fuzzy-match a single token against the known verbs.

    Uses ``difflib.get_close_matches`` with a conservative cutoff to
    catch common typos (``wajch``, ``statuz``, ``histroy``) without
    mapping unrelated words. Single-character tokens are skipped to
    avoid noisy matches -- a lone ``c`` should never silently become
    ``cancel``.

    Args:
        first_word: The first whitespace-delimited token from the
            user's input. Caller is responsible for lower-casing and
            stripping whitespace.

    Returns:
        A canonical verb string when a close match is found, or
        ``None`` when no candidate exceeds the cutoff or the token is
        too short for meaningful fuzzy matching.
    """
    if len(first_word) < _FUZZY_MIN_LENGTH:
        return None
    matches = difflib.get_close_matches(
        first_word,
        _KNOWN_VERBS,
        n=1,
        cutoff=_FUZZY_CUTOFF,
    )
    if matches:
        return matches[0]
    return None


def _classify_natural_language(raw: str) -> str | None:
    """Ask the pattern-based classifier to interpret natural language.

    Delegates to the existing deterministic ``classify()`` entry point
    (no LLM calls) and accepts the result only when the classifier
    reports a confident, non-ambiguous interpretation. The confidence
    gate matches the classifier's own internal "is confident" threshold
    so that borderline guesses are rejected rather than silently
    reinterpreting user input.

    TODO(future): Swap to LLM-based classification for more robust
    natural language understanding. The current pattern-based classifier
    is brittle -- only pre-written keyword phrases work ("what's running"
    matches, but "are the tests done yet?" won't). To upgrade, replace
    the body of this function with a call to the Dataiku Mesh LLM that
    returns the canonical verb. The signature stays the same -- no
    other code in the CLI needs to change. Consider caching recent
    classifications to reduce LLM calls for repeated queries.

    Args:
        raw: The raw, stripped user input string.

    Returns:
        The canonical verb string when the classifier is confident,
        otherwise ``None`` so the caller can emit an ``Unknown
        command`` error.
    """
    result = classify(raw)
    if result.input_type == InputType.AMBIGUOUS:
        return None
    if result.confidence_score < _CLASSIFIER_CONFIDENCE_THRESHOLD:
        return None
    return result.canonical_verb


def _resolve_verb(raw: str) -> tuple[list[str], str | None] | None:
    """Resolve raw REPL input into command parts via a 3-layer lookup.

    The resolver never calls the network or the LLM. It runs three
    progressively more forgiving layers:

    1. Exact match: if the first token is already a known verb, the
       original ``raw.split()`` tokens are returned unchanged and no
       hint is emitted.
    2. Fuzzy match: the first token is compared against the known
       verbs with ``difflib``; a close match replaces only the first
       token (preserving remaining ``run`` arguments such as the
       ``user@host`` target and free-form command description) and a
       subtle hint is emitted so the user sees how the input was
       rewritten.
    3. Classifier fallback: the full raw input is passed to the
       pattern-based classifier, which resolves natural language like
       ``"what's running?"`` or ``"kill the test"`` to a canonical
       verb. Any extracted arguments are dropped here because
       ``_execute_single`` relies on positional ``argv`` tokens; when
       the classifier picks a verb that requires arguments (such as
       ``run``), dispatch will surface the normal usage error.

    Args:
        raw: The stripped, non-empty REPL input line.

    Returns:
        A ``(parts, hint)`` tuple suitable for passing to
        ``_execute_single``, where ``hint`` is an optional one-line
        interpretation message to print before dispatch, or ``None``
        when the input cannot be resolved and the caller should emit
        an ``Unknown command`` error.
    """
    parts = raw.split()
    if not parts:
        return None

    first_word = parts[0].lower()

    # Layer 1: exact match against the known verb set. We intentionally
    # do not lower-case the rest of the tokens -- "run" arguments such
    # as natural-language test descriptions must preserve their
    # original casing.
    if first_word in _KNOWN_VERBS:
        return parts, None

    # Layer 2: fuzzy match to correct typos. Only the first token is
    # rewritten; the remaining tokens (e.g., ``user@host`` targets and
    # test descriptions for ``run``) are passed through verbatim.
    fuzzy_verb = _fuzzy_match_verb(first_word)
    if fuzzy_verb is not None:
        corrected = [fuzzy_verb, *parts[1:]]
        return corrected, f"(interpreted as '{fuzzy_verb}')"

    # Layer 3: classifier fallback for natural language. When the
    # classifier is confident we dispatch with just the canonical verb
    # -- extracted args are not threaded through because the current
    # ``_execute_single`` dispatcher parses positional argv tokens.
    classified_verb = _classify_natural_language(raw)
    if classified_verb is not None:
        return [classified_verb], f"(interpreted as '{classified_verb}')"

    return None


# ---------------------------------------------------------------------------
# REPL mode
# ---------------------------------------------------------------------------


async def _repl(client: ThinClient) -> int:
    """Run the interactive REPL loop.

    Reads commands from stdin, sends them to the daemon, and displays
    responses. Exits on quit/exit/q or EOF.

    Args:
        client: Thin client instance.

    Returns:
        0 on normal exit.
    """
    print("Jules SSH Test Runner -- Interactive Mode")
    print("Type 'help' for commands, 'c' or Ctrl+C to exit")
    print()

    while True:
        try:
            raw = input(_PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        if raw.lower() in _QUIT_COMMANDS:
            break

        if raw.lower() == "help":
            _print_help()
            continue

        resolved = _resolve_verb(raw)
        if resolved is None:
            first_word = raw.split()[0]
            print(f"Unknown command: {first_word}")
            print("Try 'help' to see available commands")
            print()
            continue

        parts, hint = resolved
        if hint is not None:
            print(hint)

        exit_code = await _execute_single(client, parts)
        if exit_code != 0:
            print(f"(exit code: {exit_code})")
        print()

    return 0


def _print_help() -> None:
    """Print available commands for the REPL."""
    print("Available commands:")
    print("  status [--verbose]            Query current run state")
    print("  watch [--run-id ID] [--tail N] Stream output from a run")
    print("  run user@host description     Start a test execution")
    print("  discover user@host command    Auto-discover test spec via -h")
    print("  cancel [--force] [--run-id ID] Cancel a run")
    print("  history [--limit N] [--status S] View past results")
    print("  health                        Check daemon liveness")
    print("  help                          Show this help")
    print("  quit / c / Ctrl+C             Exit the CLI")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _async_main(args: list[str], socket_path: Path | None) -> int:
    """Async entry point for the CLI client.

    Args:
        args: Command-line arguments (excluding program name).
        socket_path: Explicit socket path, or None for auto-discovery.

    Returns:
        Exit code.
    """
    config = ThinClientConfig(
        socket_path=socket_path,
    )
    client = ThinClient(
        config=config,
        on_confirm=_interactive_confirm,
    )

    if args:
        return await _execute_single(client, args)
    return await _repl(client)


def main() -> None:
    """Parse arguments and run the CLI client."""
    args = sys.argv[1:]

    # Extract --socket-path if present
    socket_path: Path | None = None
    filtered_args: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--socket-path" and i + 1 < len(args):
            socket_path = Path(args[i + 1])
            i += 2
        else:
            filtered_args.append(args[i])
            i += 1

    try:
        exit_code = asyncio.run(_async_main(filtered_args, socket_path))
    except KeyboardInterrupt:
        print("\nBye!")
        exit_code = 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
