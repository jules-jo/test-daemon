"""Jules CLI client entry point.

Connects to the daemon via the thin client and either sends a single
command (when arguments are provided) or enters an interactive REPL
(when no arguments are given).

Usage::

    # Interactive mode
    jules

    # Single command mode
    jules status
    jules "run the smoke tests on deploy@staging"
    jules run deploy@staging run the smoke tests
    jules watch --follow
    jules history --limit 10
    jules cancel --force
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import difflib
import sys
from pathlib import Path

from jules_daemon.classifier.classify import classify
from jules_daemon.classifier.models import ClassificationResult, InputType
from jules_daemon.classifier.nl_extractor import extract_from_natural_language
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
_RUN_CLASSIFIER_CONFIDENCE_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class _ResolvedInput:
    """Resolved CLI input ready for thin-client dispatch."""

    parts: tuple[str, ...] | None
    hint: str | None = None
    error: str | None = None


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
        # Parse:
        #   run user@host <natural language description>
        #   run --system <name> <natural language description>
        #   run --infer-target <natural language description>
        #   run --interpret-request <natural language description>
        if not remaining:
            print("Usage: jules run user@host <description>")
            print("   or: jules run --system <name> <description>")
            print("   or: jules run --infer-target <description>")
            print("   or: jules run --interpret-request <description>")
            return 1

        if remaining[0] == "--system" or remaining[0].startswith("--system="):
            if remaining[0] == "--system":
                if len(remaining) < 3:
                    print("Usage: jules run --system <name> <description>")
                    return 1
                system_name = remaining[1].strip()
                nl_parts = remaining[2:]
            else:
                _, _, system_name = remaining[0].partition("=")
                if not system_name.strip() or len(remaining) < 2:
                    print("Usage: jules run --system <name> <description>")
                    return 1
                nl_parts = remaining[1:]

            if not nl_parts:
                print("Error: natural language description required")
                return 1

            natural_language = " ".join(nl_parts)
            result = await client.run(
                natural_language=natural_language,
                system_name=system_name,
            )
        elif remaining[0] == "--infer-target":
            if len(remaining) < 2:
                print("Usage: jules run --infer-target <description>")
                return 1
            natural_language = " ".join(remaining[1:])
            result = await client.run(
                natural_language=natural_language,
                infer_target=True,
            )
        elif remaining[0] == "--interpret-request":
            if len(remaining) < 2:
                print("Usage: jules run --interpret-request <description>")
                return 1
            natural_language = " ".join(remaining[1:])
            result = await client.run(
                natural_language=natural_language,
                interpret_request=True,
            )
        else:
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


def _classify_interpreted_input(raw: str) -> ClassificationResult | None:
    """Classify free-form input and return a trusted result object.

    Accepts the general confidence threshold for most verbs, but uses a
    slightly lower threshold for ``run`` because the heuristic extractor
    intentionally assigns simple run requests scores around 0.5-0.6.
    """
    result = classify(raw)
    if result.input_type == InputType.AMBIGUOUS:
        return None
    if result.confidence_score >= _CLASSIFIER_CONFIDENCE_THRESHOLD:
        return result
    if (
        result.canonical_verb == "run"
        and result.confidence_score >= _RUN_CLASSIFIER_CONFIDENCE_THRESHOLD
    ):
        return result
    return None


def _extract_natural_language_run(raw: str) -> dict[str, object] | None:
    """Force the NL extractor for conversational run requests.

    This avoids the structured classifier path misclassifying inputs like
    ``run the smoke tests on deploy@staging`` as verb-style commands.
    """
    extraction = extract_from_natural_language(raw)
    if extraction.canonical_verb != "run":
        return None
    if extraction.confidence < _RUN_CLASSIFIER_CONFIDENCE_THRESHOLD:
        return None
    return dict(extraction.extracted_args)


def _looks_like_structured_run(parts: list[str]) -> bool:
    """Return True when argv already matches the explicit ``run`` syntax."""
    if len(parts) < 2:
        return False
    if "@" in parts[1]:
        return True
    if parts[1] == "--system":
        return len(parts) >= 4 and bool(parts[2].strip())
    if parts[1].startswith("--system="):
        return len(parts) >= 3 and bool(parts[1].partition("=")[2].strip())
    if parts[1] == "--infer-target":
        return len(parts) >= 3
    if parts[1] == "--interpret-request":
        return len(parts) >= 3
    return False


def _parse_target_spec(target_spec: str) -> tuple[str, str, int] | str:
    """Parse ``user@host[:port]`` into structured SSH target fields."""
    if "@" not in target_spec:
        return "target must be in user@host[:port] format"

    user, host_part = target_spec.split("@", 1)
    if not user.strip():
        return "target user must not be empty"

    host = host_part
    port = 22
    if ":" in host_part:
        host, port_str = host_part.rsplit(":", 1)
        if not port_str:
            return "target port must not be empty"
        try:
            port = int(port_str)
        except ValueError:
            return f"target port must be numeric, got {port_str!r}"

    if not host.strip():
        return "target host must not be empty"
    if not (1 <= port <= 65535):
        return f"target port must be 1-65535, got {port}"
    return user.strip(), host.strip(), port


def _format_target_spec(user: str, host: str, port: int) -> str:
    """Format SSH target fields back into ``user@host[:port]``."""
    if port == 22:
        return f"{user}@{host}"
    return f"{user}@{host}:{port}"


def _prompt_for_target() -> tuple[str, str, int] | None:
    """Interactively ask for a missing SSH target."""
    print("A run request needs an SSH target.")
    while True:
        try:
            raw = input("Target (user@host[:port], or 'skip' to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if not raw or raw.lower() == "skip":
            return None

        parsed = _parse_target_spec(raw)
        if isinstance(parsed, str):
            print(f"Error: {parsed}")
            continue
        return parsed


def _resolve_natural_language_run(
    raw: str,
    extracted_args: dict[str, object],
    *,
    allow_prompt: bool,
) -> _ResolvedInput:
    """Resolve a natural-language run request into explicit argv tokens."""
    target_user = extracted_args.get("target_user")
    target_host = extracted_args.get("target_host")
    system_name = extracted_args.get("system_name")
    infer_target = extracted_args.get("infer_target") is True
    raw_port = extracted_args.get("target_port", 22)

    if isinstance(target_user, str) and isinstance(target_host, str):
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            port = 22
        target_spec = _format_target_spec(
            target_user.strip(),
            target_host.strip(),
            port,
        )
        return _ResolvedInput(
            parts=("run", target_spec, raw),
            hint="(interpreted as 'run')",
        )

    if isinstance(system_name, str) and system_name.strip():
        return _ResolvedInput(
            parts=("run", "--system", system_name.strip(), raw),
            hint="(interpreted as 'run')",
        )

    if infer_target:
        return _ResolvedInput(
            parts=("run", "--infer-target", raw),
            hint="(interpreted as 'run')",
        )

    return _ResolvedInput(
        parts=("run", "--interpret-request", raw),
        hint="(interpreted as 'run')",
    )


def _resolve_verb(raw: str) -> tuple[list[str], str | None] | None:
    """Compatibility wrapper around ``_resolve_input`` for non-interactive use."""
    resolved = _resolve_input(raw, allow_prompt=False)
    if resolved.parts is None:
        return None
    return list(resolved.parts), resolved.hint


def _resolve_input(
    raw: str,
    *,
    allow_prompt: bool,
) -> _ResolvedInput:
    """Resolve raw CLI input into dispatchable argv tokens.

    Structured commands pass through unchanged. Bare natural-language run
    requests are rewritten into the existing explicit ``run`` syntax so the
    rest of the thin client can stay unchanged.
    """
    parts = raw.split()
    if not parts:
        return _ResolvedInput(parts=None, error="Empty input.")

    first_word = parts[0].lower()

    # Layer 1: exact match against the known verb set. Keep the explicit
    # syntax for structured commands, but let conversational ``run ...``
    # input fall through to NL resolution when the target token is missing.
    if first_word in _KNOWN_VERBS:
        if first_word != "run" or _looks_like_structured_run(parts):
            return _ResolvedInput(parts=tuple(parts))
        extracted_args = _extract_natural_language_run(raw)
        if extracted_args is None:
            return _ResolvedInput(
                parts=None,
                error=(
                    "Run requests must include an SSH target like user@host[:port]. "
                    "Example: run the smoke tests on deploy@staging"
                ),
            )
        return _resolve_natural_language_run(
            raw,
            extracted_args,
            allow_prompt=allow_prompt,
        )

    interpreted = _classify_interpreted_input(raw)
    if interpreted is not None and interpreted.input_type != InputType.COMMAND:
        if interpreted.canonical_verb == "run":
            return _resolve_natural_language_run(
                raw,
                dict(interpreted.extracted_args),
                allow_prompt=allow_prompt,
            )
        return _ResolvedInput(
            parts=(interpreted.canonical_verb,),
            hint=f"(interpreted as '{interpreted.canonical_verb}')",
        )

    # Layer 2: fuzzy match to correct typos. Only the first token is
    # rewritten; the remaining tokens (e.g., ``user@host`` targets and
    # test descriptions for ``run``) are passed through verbatim.
    fuzzy_verb = _fuzzy_match_verb(first_word)
    if fuzzy_verb is not None:
        corrected = [fuzzy_verb, *parts[1:]]
        if fuzzy_verb == "run" and not _looks_like_structured_run(corrected):
            extracted_args = _extract_natural_language_run(raw)
            if extracted_args is not None:
                return _resolve_natural_language_run(
                    raw,
                    extracted_args,
                    allow_prompt=allow_prompt,
                )
        return _ResolvedInput(
            parts=tuple(corrected),
            hint=f"(interpreted as '{fuzzy_verb}')",
        )

    # Layer 3: classifier fallback for natural language. When the
    # classifier is confident we dispatch to the canonical verb. ``run``
    # is special because it needs both a target and the original free-form
    # prompt text.
    interpreted = _classify_interpreted_input(raw)
    if interpreted is None:
        return _ResolvedInput(parts=None, error=f"Unknown command: {first_word}")

    if interpreted.canonical_verb == "run":
        return _resolve_natural_language_run(
            raw,
            dict(interpreted.extracted_args),
            allow_prompt=allow_prompt,
        )

    return _ResolvedInput(
        parts=(interpreted.canonical_verb,),
        hint=f"(interpreted as '{interpreted.canonical_verb}')",
    )


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

        resolved = _resolve_input(raw, allow_prompt=True)
        if resolved.parts is None:
            print(resolved.error or "Unknown command")
            print("Try 'help' to see available commands")
            print()
            continue

        if resolved.hint is not None:
            print(resolved.hint)

        exit_code = await _execute_single(client, list(resolved.parts))
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
    print("  run --system NAME description Start a test execution via system alias")
    print("  run --infer-target description Ask daemon to infer system alias from NL")
    print("  run --interpret-request description")
    print("                                Ask daemon to interpret unresolved run prompts")
    print("  run the smoke tests on deploy@staging")
    print("                                Natural-language run request")
    print("  run the smoke tests in tuto")
    print("                                Natural-language run request via inferred system alias")
    print("  run the smoke tests in system tuto")
    print("                                Natural-language run request via system alias")
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
        resolved = _resolve_input(" ".join(args), allow_prompt=sys.stdin.isatty())
        if resolved.parts is None:
            print(resolved.error or "Unknown command")
            return 1
        if resolved.hint is not None:
            print(resolved.hint)
        return await _execute_single(client, list(resolved.parts))
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
