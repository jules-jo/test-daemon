"""Argparse-based argument parser for the ``watch`` CLI command.

Provides a dedicated ``argparse.ArgumentParser`` for the watch verb,
supporting the following options:

    --run-id <id>    Target a specific job/run by ID (optional)
    --follow / -f    Continuously stream output (like ``tail -f``)
    --format <fmt>   Output format: text, json, or summary
    --tail <n>       Number of recent lines to show on attach

The parser is built as a separate module (not inlined in ``parser.py``)
so that:
    1. Argparse handles flag validation, help text, and type conversion
    2. The watch command's argument schema is self-documenting
    3. The parser can be tested in isolation from the verb routing layer

The ``parse_watch_tokens`` function is the main entry point. It accepts
a list of pre-tokenized strings (the tokens after the ``watch`` verb has
been stripped) and returns either a validated ``WatchArgs`` dataclass or
a human-readable error string. It never raises exceptions for invalid
user input.

Usage::

    from jules_daemon.cli.watch_parser import parse_watch_tokens

    result = parse_watch_tokens(["--run-id", "abc", "--follow", "--format", "json"])
    if isinstance(result, WatchArgs):
        dispatch(result)
    else:
        print(f"Error: {result}")
"""

from __future__ import annotations

import argparse
from typing import NoReturn

from jules_daemon.cli.verbs import VALID_OUTPUT_FORMATS, WatchArgs

__all__ = [
    "VALID_OUTPUT_FORMATS",
    "build_watch_argparser",
    "parse_watch_tokens",
]


# ---------------------------------------------------------------------------
# Non-exiting ArgumentParser subclass
# ---------------------------------------------------------------------------


class _NoExitArgumentParser(argparse.ArgumentParser):
    """ArgumentParser subclass that raises instead of calling sys.exit.

    Standard argparse calls ``sys.exit()`` on parse errors, which is
    unacceptable in a daemon context. This subclass captures the error
    message and raises ``SystemExit`` with the message string, which
    the caller catches and converts to an error string.

    On Python 3.14+ with ``exit_on_error=False``, argparse raises
    ``argparse.ArgumentError`` directly for unrecognized arguments and
    type errors (bypassing ``error()``). The caller must catch both
    ``SystemExit`` and ``ArgumentError`` to handle all error paths.
    """

    def error(self, message: str) -> NoReturn:
        """Override to raise SystemExit with the error message.

        Args:
            message: Error description from argparse.

        Raises:
            SystemExit: Always, with the error message as the argument.
        """
        raise SystemExit(message)

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        """Override to prevent sys.exit calls.

        Args:
            status: Exit status code (ignored).
            message: Optional message (used as the error).

        Raises:
            SystemExit: Always, with the message as the argument.
        """
        raise SystemExit(message or "")


# ---------------------------------------------------------------------------
# Parser factory
# ---------------------------------------------------------------------------

# Sorted tuple for deterministic help text and error messages.
_SORTED_FORMATS = tuple(sorted(VALID_OUTPUT_FORMATS))


def build_watch_argparser() -> _NoExitArgumentParser:
    """Build an argparse.ArgumentParser for the ``watch`` command.

    The parser is configured with ``exit_on_error=False`` and uses the
    ``_NoExitArgumentParser`` subclass to prevent ``sys.exit()`` calls.
    ``add_help=False`` prevents argparse from writing to stdout when
    ``--help`` is passed (important in a daemon context where stdout
    may be a socket or redirected). All flags use long-form names;
    ``--follow`` also accepts ``-f``.

    Returns:
        Configured ArgumentParser for the watch command.
    """
    parser = _NoExitArgumentParser(
        prog="watch",
        description="Live-stream output from a running test session",
        exit_on_error=False,
        add_help=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        default=False,
        help="Show help message",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        metavar="ID",
        help="Target a specific run/job by its unique identifier",
    )

    parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        default=False,
        help="Continuously stream new output as it arrives (like tail -f)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=_SORTED_FORMATS,
        default="text",
        dest="output_format",
        metavar="FORMAT",
        help=(
            f"Output format: {', '.join(_SORTED_FORMATS)} "
            f"(default: text)"
        ),
    )

    parser.add_argument(
        "--tail",
        type=int,
        default=50,
        metavar="N",
        help="Number of recent output lines to show on initial attach (default: 50)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_watch_tokens(tokens: list[str]) -> WatchArgs | str:
    """Parse pre-tokenized watch command arguments into a WatchArgs.

    Accepts the token list remaining after the ``watch`` verb has been
    stripped by the top-level parser. Uses argparse for flag validation,
    type conversion, and help text.

    The function never raises exceptions for invalid user input. Parse
    errors, unknown flags, and validation failures are all returned as
    human-readable error strings.

    Args:
        tokens: List of argument tokens (e.g., ["--run-id", "abc", "-f"]).

    Returns:
        ``WatchArgs`` on success, or a human-readable error string on
        any failure (unknown flags, invalid types, validation errors).
    """
    parser = build_watch_argparser()

    try:
        namespace = parser.parse_args(tokens)
    except SystemExit as exc:
        # Our _NoExitArgumentParser.error() raises SystemExit with the message
        error_msg = str(exc) if str(exc) else "Invalid watch arguments"
        return error_msg
    except argparse.ArgumentError as exc:
        # Python 3.14+ with exit_on_error=False raises ArgumentError directly
        return str(exc)

    # Handle --help: return formatted help as the error string so it stays
    # in the return channel and does not pollute stdout.
    if getattr(namespace, "help", False):
        return parser.format_help()

    # Extract parsed values (argparse has already validated types and choices)
    run_id: str | None = namespace.run_id
    follow: bool = namespace.follow
    output_format: str = namespace.output_format
    tail_lines: int = namespace.tail

    # Construct the immutable WatchArgs dataclass.
    # Range validation (e.g., tail_lines positivity, output_format membership)
    # is handled by WatchArgs.__post_init__. The ValueError is caught below.
    try:
        return WatchArgs(
            run_id=run_id,
            tail_lines=tail_lines,
            follow=follow,
            output_format=output_format,
        )
    except ValueError as exc:
        return str(exc)
