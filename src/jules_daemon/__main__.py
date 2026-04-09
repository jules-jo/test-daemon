"""Jules Daemon entry point.

Parses CLI arguments, initializes the wiki directory structure,
loads LLM configuration, sets up the IPC server with request
handlers, runs crash recovery, and starts the async event loop.

Usage::

    python -m jules_daemon --wiki-dir ./wiki --log-level DEBUG
    jules-daemon --socket-path /tmp/jules.sock
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from jules_daemon.ipc.request_handler import RequestHandler, RequestHandlerConfig
from jules_daemon.ipc.server import ServerConfig, SocketServer
from jules_daemon.ipc.socket_discovery import default_socket_path
from jules_daemon.startup.lifecycle import StartupHookConfig, run_startup
from jules_daemon.wiki.current_run import exists as current_run_exists
from jules_daemon.wiki.current_run import read as read_current_run
from jules_daemon.wiki.layout import initialize_wiki

logger = logging.getLogger("jules_daemon")

_DEFAULT_WIKI_DIR = "wiki"
_DEFAULT_LOG_LEVEL = "INFO"
_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the daemon.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="jules-daemon",
        description="Jules SSH Test Runner Daemon",
    )
    parser.add_argument(
        "--wiki-dir",
        type=Path,
        default=Path(_DEFAULT_WIKI_DIR),
        help=f"Path to the wiki directory (default: {_DEFAULT_WIKI_DIR})",
    )
    parser.add_argument(
        "--socket-path",
        type=Path,
        default=None,
        help="Path to the Unix domain socket (default: auto-discover)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=_DEFAULT_LOG_LEVEL,
        choices=_VALID_LOG_LEVELS,
        help=f"Logging level (default: {_DEFAULT_LOG_LEVEL})",
    )
    parser.add_argument(
        "--skip-startup-scan",
        action="store_true",
        default=False,
        help="Skip the scan-probe-mark pipeline on startup",
    )
    return parser


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(level_name: str) -> None:
    """Configure structured logging for the daemon.

    Args:
        level_name: Log level string (DEBUG, INFO, etc.).
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Crash recovery check
# ---------------------------------------------------------------------------


def _check_crash_recovery(wiki_dir: Path) -> bool:
    """Check if crash recovery is needed based on current-run state.

    If a non-idle current-run record exists, a previous daemon instance
    may have crashed. The startup lifecycle's scan-probe-mark pipeline
    handles the actual recovery.

    Args:
        wiki_dir: Path to the wiki root directory.

    Returns:
        True if a non-idle current-run record exists.
    """
    if not current_run_exists(wiki_dir):
        return False

    run = read_current_run(wiki_dir)
    if run is None:
        return False

    from jules_daemon.wiki.models import RunStatus

    if run.status != RunStatus.IDLE:
        logger.warning(
            "Crash recovery needed: found non-idle current-run "
            "(status=%s, run_id=%s)",
            run.status.value,
            run.run_id,
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------


async def _run_daemon(
    wiki_dir: Path,
    socket_path: Path,
    skip_scan: bool,
) -> int:
    """Run the daemon lifecycle.

    1. Initialize wiki directory structure
    2. Check for crash recovery
    3. Run startup lifecycle (scan-probe-mark pipeline)
    4. Start IPC server
    5. Wait for shutdown signal

    Args:
        wiki_dir: Path to the wiki root directory.
        socket_path: Path to the Unix domain socket.
        skip_scan: Whether to skip the startup scan pipeline.

    Returns:
        Exit code (0 on success).
    """
    # Step 1: Initialize wiki
    wiki_dir = wiki_dir.resolve()
    logger.info("Initializing wiki at %s", wiki_dir)
    initialize_wiki(wiki_dir)

    # Step 2: Check for crash recovery
    needs_recovery = _check_crash_recovery(wiki_dir)
    if needs_recovery:
        logger.info("Crash recovery will be handled by startup pipeline")

    # Step 3: Run startup lifecycle
    startup_config = StartupHookConfig(
        run_scan_probe_mark=not skip_scan,
    )
    startup_result = await run_startup(wiki_dir, config=startup_config)

    if startup_result.is_ready:
        logger.info(
            "Daemon startup complete in %.3fs",
            startup_result.duration_seconds,
        )
    else:
        logger.error(
            "Daemon startup failed: phase=%s, error=%s",
            startup_result.final_phase.value,
            startup_result.error,
        )
        return 1

    if startup_result.error:
        logger.warning("Startup warning: %s", startup_result.error)

    # Step 4: Set up IPC server
    handler_config = RequestHandlerConfig(wiki_root=wiki_dir)
    handler = RequestHandler(config=handler_config)

    server_config = ServerConfig(socket_path=socket_path)
    server = SocketServer(config=server_config, handler=handler)

    # Step 5: Set up shutdown signal handling
    shutdown_event = asyncio.Event()

    def _signal_handler(sig: int) -> None:
        sig_name = signal.Signals(sig).name
        logger.info("Received %s, initiating graceful shutdown", sig_name)
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler, sig)

    # Step 6: Start server and wait for shutdown
    async with server:
        logger.info(
            "Jules daemon is ready (socket=%s, wiki=%s, pid=%d)",
            socket_path,
            wiki_dir,
            os.getpid(),
        )
        await shutdown_event.wait()

    logger.info("Jules daemon stopped")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the daemon."""
    parser = build_parser()
    args = parser.parse_args()

    _configure_logging(args.log_level)

    # Resolve socket path
    socket_path: Path
    if args.socket_path is not None:
        socket_path = args.socket_path.resolve()
    else:
        socket_path = default_socket_path()

    logger.info(
        "Starting Jules daemon (wiki=%s, socket=%s, log_level=%s)",
        args.wiki_dir,
        socket_path,
        args.log_level,
    )

    exit_code = asyncio.run(
        _run_daemon(
            wiki_dir=args.wiki_dir,
            socket_path=socket_path,
            skip_scan=args.skip_startup_scan,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
