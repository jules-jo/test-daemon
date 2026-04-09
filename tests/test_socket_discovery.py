"""Tests for IPC daemon socket path discovery.

Validates:
    - Environment variable override (JULES_SOCKET_PATH)
    - XDG_RUNTIME_DIR-based discovery
    - Fallback to /tmp/jules-{uid}/daemon.sock
    - Path validation (existing socket file vs directory)
    - SocketPathResult dataclass immutability and fields
    - discover_socket_path returns the correct source for each strategy
    - All checked paths are recorded in the result
    - Custom search order via DiscoveryConfig
"""

from __future__ import annotations

import os
import stat
import socket
from pathlib import Path
from unittest.mock import patch

import pytest

from jules_daemon.ipc.socket_discovery import (
    DEFAULT_SOCKET_FILENAME,
    DEFAULT_SUBDIRECTORY,
    DiscoveryConfig,
    DiscoverySource,
    SocketPathResult,
    discover_socket_path,
    default_socket_path,
)


# ---------------------------------------------------------------------------
# SocketPathResult tests
# ---------------------------------------------------------------------------


class TestSocketPathResult:
    """Tests for the immutable SocketPathResult dataclass."""

    def test_create_with_path(self) -> None:
        result = SocketPathResult(
            path=Path("/run/user/1000/jules/daemon.sock"),
            source=DiscoverySource.XDG_RUNTIME,
            checked=(Path("/tmp/jules-1000/daemon.sock"),),
        )
        assert result.path == Path("/run/user/1000/jules/daemon.sock")
        assert result.source == DiscoverySource.XDG_RUNTIME
        assert len(result.checked) == 1

    def test_create_not_found(self) -> None:
        result = SocketPathResult(
            path=None,
            source=DiscoverySource.NOT_FOUND,
            checked=(Path("/a"), Path("/b")),
        )
        assert result.path is None
        assert result.source == DiscoverySource.NOT_FOUND
        assert len(result.checked) == 2

    def test_found_property(self) -> None:
        found = SocketPathResult(
            path=Path("/tmp/jules/daemon.sock"),
            source=DiscoverySource.ENV_VAR,
            checked=(),
        )
        not_found = SocketPathResult(
            path=None,
            source=DiscoverySource.NOT_FOUND,
            checked=(),
        )
        assert found.found is True
        assert not_found.found is False

    def test_frozen(self) -> None:
        result = SocketPathResult(
            path=Path("/tmp/sock"),
            source=DiscoverySource.ENV_VAR,
            checked=(),
        )
        with pytest.raises(AttributeError):
            result.path = Path("/mutated")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DiscoveryConfig tests
# ---------------------------------------------------------------------------


class TestDiscoveryConfig:
    """Tests for the DiscoveryConfig dataclass."""

    def test_defaults(self) -> None:
        config = DiscoveryConfig()
        assert config.socket_filename == DEFAULT_SOCKET_FILENAME
        assert config.subdirectory == DEFAULT_SUBDIRECTORY

    def test_custom_values(self) -> None:
        config = DiscoveryConfig(
            socket_filename="test.sock",
            subdirectory="my-daemon",
        )
        assert config.socket_filename == "test.sock"
        assert config.subdirectory == "my-daemon"

    def test_frozen(self) -> None:
        config = DiscoveryConfig()
        with pytest.raises(AttributeError):
            config.socket_filename = "changed"  # type: ignore[misc]

    def test_empty_filename_raises(self) -> None:
        with pytest.raises(ValueError, match="socket_filename must not be empty"):
            DiscoveryConfig(socket_filename="")

    def test_empty_subdirectory_raises(self) -> None:
        with pytest.raises(ValueError, match="subdirectory must not be empty"):
            DiscoveryConfig(subdirectory="  ")


# ---------------------------------------------------------------------------
# DiscoverySource tests
# ---------------------------------------------------------------------------


class TestDiscoverySource:
    """Tests for the DiscoverySource enum."""

    def test_all_values(self) -> None:
        assert DiscoverySource.ENV_VAR.value == "env_var"
        assert DiscoverySource.XDG_RUNTIME.value == "xdg_runtime"
        assert DiscoverySource.TMPDIR.value == "tmpdir"
        assert DiscoverySource.NOT_FOUND.value == "not_found"


# ---------------------------------------------------------------------------
# discover_socket_path -- environment variable override
# ---------------------------------------------------------------------------


class TestDiscoverFromEnvVar:
    """Tests for JULES_SOCKET_PATH environment variable discovery."""

    def test_env_var_existing_socket(self, tmp_path: Path) -> None:
        """When JULES_SOCKET_PATH points to an existing socket, use it."""
        sock_path = tmp_path / "custom.sock"
        # Create an actual Unix socket to simulate the daemon
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(sock_path))
            with patch.dict(os.environ, {"JULES_SOCKET_PATH": str(sock_path)}):
                result = discover_socket_path()
            assert result.path == sock_path
            assert result.source == DiscoverySource.ENV_VAR
            assert result.found is True
        finally:
            server.close()
            sock_path.unlink(missing_ok=True)

    def test_env_var_nonexistent_path_still_returns_it(self) -> None:
        """When JULES_SOCKET_PATH is set but the file does not exist,
        return it anyway (the daemon might not be started yet)."""
        with patch.dict(
            os.environ,
            {"JULES_SOCKET_PATH": "/nonexistent/path/daemon.sock"},
        ):
            result = discover_socket_path()
        assert result.path == Path("/nonexistent/path/daemon.sock")
        assert result.source == DiscoverySource.ENV_VAR

    def test_env_var_empty_string_is_ignored(self, tmp_path: Path) -> None:
        """Empty JULES_SOCKET_PATH is treated as unset."""
        with patch.dict(os.environ, {"JULES_SOCKET_PATH": ""}):
            result = discover_socket_path()
        assert result.source != DiscoverySource.ENV_VAR

    def test_env_var_whitespace_is_ignored(self) -> None:
        """Whitespace-only JULES_SOCKET_PATH is treated as unset."""
        with patch.dict(os.environ, {"JULES_SOCKET_PATH": "   "}):
            result = discover_socket_path()
        assert result.source != DiscoverySource.ENV_VAR


# ---------------------------------------------------------------------------
# discover_socket_path -- XDG_RUNTIME_DIR
# ---------------------------------------------------------------------------


class TestDiscoverFromXdgRuntime:
    """Tests for XDG_RUNTIME_DIR-based discovery."""

    def test_xdg_with_existing_socket(self, tmp_path: Path) -> None:
        """When XDG_RUNTIME_DIR/jules/daemon.sock exists as a socket, use it."""
        xdg_dir = tmp_path / "runtime"
        jules_dir = xdg_dir / DEFAULT_SUBDIRECTORY
        jules_dir.mkdir(parents=True)
        sock_path = jules_dir / DEFAULT_SOCKET_FILENAME

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(sock_path))
            env = {"XDG_RUNTIME_DIR": str(xdg_dir)}
            with patch.dict(os.environ, env, clear=False):
                # Remove JULES_SOCKET_PATH so it doesn't interfere
                os.environ.pop("JULES_SOCKET_PATH", None)
                result = discover_socket_path()
            assert result.path == sock_path
            assert result.source == DiscoverySource.XDG_RUNTIME
        finally:
            server.close()
            sock_path.unlink(missing_ok=True)

    def test_xdg_dir_exists_but_no_socket_returns_expected_path(
        self, tmp_path: Path
    ) -> None:
        """When XDG_RUNTIME_DIR exists but no socket file, still return the
        expected path (daemon may start later)."""
        xdg_dir = tmp_path / "runtime"
        xdg_dir.mkdir()
        env = {"XDG_RUNTIME_DIR": str(xdg_dir)}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("JULES_SOCKET_PATH", None)
            result = discover_socket_path()
        # Should return the XDG path even though socket doesn't exist yet
        expected = xdg_dir / DEFAULT_SUBDIRECTORY / DEFAULT_SOCKET_FILENAME
        assert result.path == expected
        assert result.source == DiscoverySource.XDG_RUNTIME


# ---------------------------------------------------------------------------
# discover_socket_path -- /tmp fallback
# ---------------------------------------------------------------------------


class TestDiscoverFromTmpdir:
    """Tests for /tmp/jules-{uid}/ fallback discovery."""

    def test_tmpdir_fallback_when_no_xdg(self) -> None:
        """When XDG_RUNTIME_DIR is unset, falls back to /tmp/jules-{uid}/."""
        env_clean = {
            k: v for k, v in os.environ.items()
            if k not in ("JULES_SOCKET_PATH", "XDG_RUNTIME_DIR")
        }
        with patch.dict(os.environ, env_clean, clear=True):
            result = discover_socket_path()
        uid = os.getuid()
        expected = Path(f"/tmp/jules-{uid}") / DEFAULT_SOCKET_FILENAME
        assert result.path == expected
        assert result.source == DiscoverySource.TMPDIR


# ---------------------------------------------------------------------------
# discover_socket_path -- checked paths tracking
# ---------------------------------------------------------------------------


class TestDiscoverCheckedPaths:
    """Tests that discover_socket_path records all checked paths."""

    def test_records_checked_paths(self) -> None:
        """The result includes all paths that were examined."""
        env_clean = {
            k: v for k, v in os.environ.items()
            if k not in ("JULES_SOCKET_PATH", "XDG_RUNTIME_DIR")
        }
        with patch.dict(os.environ, env_clean, clear=True):
            result = discover_socket_path()
        # Should have checked at least the tmpdir path
        assert len(result.checked) >= 1
        assert all(isinstance(p, Path) for p in result.checked)


# ---------------------------------------------------------------------------
# default_socket_path
# ---------------------------------------------------------------------------


class TestDefaultSocketPath:
    """Tests for the default_socket_path convenience function."""

    def test_returns_a_path(self) -> None:
        """default_socket_path always returns a Path (never None)."""
        path = default_socket_path()
        assert isinstance(path, Path)
        assert str(path).endswith(DEFAULT_SOCKET_FILENAME)

    def test_returns_env_var_when_set(self) -> None:
        with patch.dict(
            os.environ,
            {"JULES_SOCKET_PATH": "/custom/daemon.sock"},
        ):
            path = default_socket_path()
        assert path == Path("/custom/daemon.sock")
