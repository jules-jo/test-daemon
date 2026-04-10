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
import sys
from pathlib import Path

from jules_daemon.ipc.framing import MessageEnvelope
from jules_daemon.thin_client.client import ThinClient, ThinClientConfig
from jules_daemon.thin_client.renderer import render_confirm_prompt

_PROMPT = "jules> "
_QUIT_COMMANDS = frozenset({"quit", "exit", "q", "c"})

# Verb routing table: maps user-facing verb strings to thin client methods.
_VERB_MAP = frozenset({
    "status", "history", "cancel", "run", "watch", "health",
})


# ---------------------------------------------------------------------------
# Confirmation callback
# ---------------------------------------------------------------------------


def _interactive_confirm(envelope: MessageEnvelope) -> bool:
    """Prompt the user to approve or deny an SSH command.

    Displays the proposed command from the daemon's CONFIRM_PROMPT
    envelope and asks for y/n confirmation.

    Args:
        envelope: The CONFIRM_PROMPT envelope from the daemon.

    Returns:
        True if the user approved, False otherwise.
    """
    prompt_text = render_confirm_prompt(envelope)
    print(prompt_text)

    try:
        answer = input("Approve? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    return answer in ("y", "yes")


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

        parts = raw.split()
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
