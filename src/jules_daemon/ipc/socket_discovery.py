"""IPC daemon socket path discovery for CLI clients.

Locates the daemon's Unix domain socket file using a prioritized search
order. The CLI client calls ``discover_socket_path()`` before connecting
to determine where the daemon is (or should be) listening.

Search order:
    1. **JULES_SOCKET_PATH** environment variable -- explicit override.
       When set to a non-empty string, this path is used unconditionally.
       The file does not need to exist (the daemon may not be started yet).

    2. **XDG_RUNTIME_DIR** -- standard Linux per-user runtime directory.
       Looks for ``$XDG_RUNTIME_DIR/jules/daemon.sock``. This is the
       preferred location on Linux systems with systemd (which sets
       XDG_RUNTIME_DIR to ``/run/user/{uid}``).

    3. **/tmp/jules-{uid}/daemon.sock** -- fallback when XDG_RUNTIME_DIR
       is not available (macOS, containers, minimal Linux installs).
       Uses the numeric UID to namespace the socket per user.

The search always returns a result. When no existing socket file is
found, the result still contains the best candidate path (where the
daemon *should* create its socket). This allows the CLI to attempt
connection and report a meaningful "daemon not running" error rather
than "cannot determine socket path".

Usage::

    from jules_daemon.ipc.socket_discovery import discover_socket_path

    result = discover_socket_path()
    if result.found:
        print(f"Daemon socket found at {result.path} (via {result.source.value})")
    else:
        print(f"No daemon running; expected socket at {result.path}")

    # Or use the convenience function:
    from jules_daemon.ipc.socket_discovery import default_socket_path
    path = default_socket_path()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = [
    "DEFAULT_SOCKET_FILENAME",
    "DEFAULT_SUBDIRECTORY",
    "DiscoveryConfig",
    "DiscoverySource",
    "SocketPathResult",
    "default_socket_path",
    "discover_socket_path",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SOCKET_FILENAME: str = "daemon.sock"
"""Default filename for the Unix domain socket."""

DEFAULT_SUBDIRECTORY: str = "jules"
"""Default subdirectory name under XDG_RUNTIME_DIR or /tmp."""

_ENV_VAR_NAME: str = "JULES_SOCKET_PATH"
"""Environment variable for explicit socket path override."""

_ENV_XDG_RUNTIME: str = "XDG_RUNTIME_DIR"
"""Standard XDG runtime directory environment variable."""


# ---------------------------------------------------------------------------
# DiscoverySource enum
# ---------------------------------------------------------------------------


class DiscoverySource(Enum):
    """How the socket path was determined.

    Values:
        ENV_VAR:     Resolved from the JULES_SOCKET_PATH environment variable.
        XDG_RUNTIME: Resolved from $XDG_RUNTIME_DIR/jules/daemon.sock.
        TMPDIR:      Resolved from /tmp/jules-{uid}/daemon.sock fallback.
        NOT_FOUND:   No candidate path could be determined (should not
                     normally occur since the TMPDIR fallback always works).
    """

    ENV_VAR = "env_var"
    XDG_RUNTIME = "xdg_runtime"
    TMPDIR = "tmpdir"
    NOT_FOUND = "not_found"


# ---------------------------------------------------------------------------
# DiscoveryConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveryConfig:
    """Immutable configuration for socket path discovery.

    Allows customizing the socket filename and subdirectory name
    for testing or non-standard deployments.

    Attributes:
        socket_filename: Name of the socket file. Default: "daemon.sock".
        subdirectory:    Subdirectory name under the runtime dir.
                         Default: "jules".
    """

    socket_filename: str = DEFAULT_SOCKET_FILENAME
    subdirectory: str = DEFAULT_SUBDIRECTORY

    def __post_init__(self) -> None:
        if not self.socket_filename or not self.socket_filename.strip():
            raise ValueError("socket_filename must not be empty")
        if not self.subdirectory or not self.subdirectory.strip():
            raise ValueError("subdirectory must not be empty")


# ---------------------------------------------------------------------------
# SocketPathResult dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SocketPathResult:
    """Immutable result of socket path discovery.

    Always contains a candidate path (even when no live socket was found).
    The ``source`` field explains how the path was determined, and
    ``checked`` lists all paths that were examined during the search.

    Attributes:
        path:    Resolved socket path. None only when no candidate could
                 be determined at all (extremely rare).
        source:  How the path was determined (env var, XDG, tmpdir, etc.).
        checked: All paths that were examined during the discovery search,
                 in the order they were checked.
    """

    path: Path | None
    source: DiscoverySource
    checked: tuple[Path, ...]

    @property
    def found(self) -> bool:
        """True if a socket path was successfully determined."""
        return self.path is not None


# ---------------------------------------------------------------------------
# Internal: discovery strategies
# ---------------------------------------------------------------------------


def _try_env_var() -> Path | None:
    """Check JULES_SOCKET_PATH environment variable.

    Returns the path if the env var is set to a non-empty, non-whitespace
    string. The file does not need to exist.

    Returns:
        Path from the environment variable, or None if unset/empty.
    """
    raw = os.environ.get(_ENV_VAR_NAME, "")
    stripped = raw.strip()
    if not stripped:
        return None
    return Path(stripped)


def _try_xdg_runtime(config: DiscoveryConfig) -> Path | None:
    """Check XDG_RUNTIME_DIR for the socket file.

    Returns the expected socket path if XDG_RUNTIME_DIR is set and
    non-empty. The actual file does not need to exist.

    Args:
        config: Discovery configuration with subdirectory and filename.

    Returns:
        Expected socket path under XDG_RUNTIME_DIR, or None if the
        env var is unset/empty.
    """
    xdg_dir = os.environ.get(_ENV_XDG_RUNTIME, "")
    if not xdg_dir.strip():
        return None
    return Path(xdg_dir) / config.subdirectory / config.socket_filename


def _tmpdir_fallback(config: DiscoveryConfig) -> Path:
    """Build a per-user fallback socket path in the temp directory.

    On Linux/macOS: /tmp/jules-{uid}/daemon.sock
    On Windows: %TEMP%/jules-{username}/daemon.sock

    Always returns a path. The directory and file may not exist.

    Args:
        config: Discovery configuration with subdirectory and filename.

    Returns:
        Fallback socket path namespaced by user identity.
    """
    if hasattr(os, "getuid"):
        user_id = str(os.getuid())
    else:
        user_id = os.environ.get("USERNAME", os.environ.get("USER", "default"))
    import tempfile
    tmp_base = Path(tempfile.gettempdir())
    return tmp_base / f"{config.subdirectory}-{user_id}" / config.socket_filename


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_socket_path(
    config: DiscoveryConfig | None = None,
) -> SocketPathResult:
    """Discover the daemon socket path using the standard search order.

    Checks each candidate location in priority order and returns the
    first match. A "match" means either:
    - The env var is set (highest priority, used unconditionally).
    - The XDG runtime path can be determined (used if env var is unset).
    - The tmpdir fallback (always available).

    The returned path may or may not have a live socket. The caller
    should attempt connection to determine if the daemon is running.

    Args:
        config: Optional custom discovery configuration. Uses defaults
            if None.

    Returns:
        SocketPathResult with the resolved path, source, and list
        of all paths checked during discovery.
    """
    effective_config = config if config is not None else DiscoveryConfig()
    checked: list[Path] = []

    # Strategy 1: Environment variable override
    env_path = _try_env_var()
    if env_path is not None:
        checked.append(env_path)
        logger.debug(
            "Socket path from %s: %s",
            _ENV_VAR_NAME,
            env_path,
        )
        return SocketPathResult(
            path=env_path,
            source=DiscoverySource.ENV_VAR,
            checked=tuple(checked),
        )

    # Strategy 2: XDG_RUNTIME_DIR
    xdg_path = _try_xdg_runtime(effective_config)
    if xdg_path is not None:
        checked.append(xdg_path)
        logger.debug(
            "Socket path from %s: %s",
            _ENV_XDG_RUNTIME,
            xdg_path,
        )
        return SocketPathResult(
            path=xdg_path,
            source=DiscoverySource.XDG_RUNTIME,
            checked=tuple(checked),
        )

    # Strategy 3: /tmp fallback
    tmp_path = _tmpdir_fallback(effective_config)
    checked.append(tmp_path)
    logger.debug("Socket path from tmpdir fallback: %s", tmp_path)
    return SocketPathResult(
        path=tmp_path,
        source=DiscoverySource.TMPDIR,
        checked=tuple(checked),
    )


def default_socket_path(
    config: DiscoveryConfig | None = None,
) -> Path:
    """Convenience function: discover and return the socket path.

    Unlike ``discover_socket_path()``, this always returns a ``Path``
    (never None). If discovery somehow fails completely, raises a
    RuntimeError.

    Args:
        config: Optional custom discovery configuration.

    Returns:
        The resolved socket path.

    Raises:
        RuntimeError: If no socket path could be determined (should
            not occur under normal circumstances).
    """
    result = discover_socket_path(config=config)
    if result.path is None:
        raise RuntimeError(
            "Failed to determine daemon socket path. "
            f"Checked: {[str(p) for p in result.checked]}"
        )
    return result.path
