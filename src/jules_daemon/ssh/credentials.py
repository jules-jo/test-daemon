"""SSH credential lookup for password-based authentication.

Provides a credential resolution chain for SSH password auth:

1. Per-host credentials file (~/.jules/ssh_credentials.yaml or
   path from JULES_SSH_CREDENTIALS_FILE env var)
2. Single-host fallback via JULES_SSH_PASSWORD env var
3. No password (fall back to key-based auth)

The credentials file format:

    hosts:
      staging-server-2:
        username: deploy
        password: "secret"
      prod-test:
        username: testrunner
        password: "other-secret"

Security:
    - Passwords are NEVER logged or included in error messages
    - The credentials file permissions are checked; a warning is
      emitted if the file is readable by group or others
    - The REDACTED constant is used in any log/audit output

Usage:
    from jules_daemon.ssh.credentials import resolve_ssh_credentials

    creds = resolve_ssh_credentials("staging-server-2")
    if creds is not None:
        # Use creds.username and creds.password for paramiko auth
        ...
    else:
        # Fall back to key-based auth
        ...
"""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "REDACTED",
    "SSHCredential",
    "load_credentials_file",
    "resolve_ssh_credentials",
]

logger = logging.getLogger(__name__)

REDACTED: str = "********"
"""Placeholder used in logs and audit output instead of real passwords."""

_ENV_PASSWORD = "JULES_SSH_PASSWORD"
_ENV_CREDENTIALS_FILE = "JULES_SSH_CREDENTIALS_FILE"
_DEFAULT_CREDENTIALS_PATH = Path.home() / ".jules" / "ssh_credentials.yaml"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SSHCredential:
    """Immutable container for a resolved SSH credential.

    Attributes:
        username: SSH username (may differ from the SSHTarget user if
            the credentials file specifies a per-host override).
        password: The plaintext password for authentication.
        source: Human-readable description of where the credential
            was loaded from (for audit logging -- never contains the
            actual password).
    """

    username: str | None
    password: str
    source: str

    def __repr__(self) -> str:
        """Redacted repr -- never expose the password."""
        return (
            f"SSHCredential(username={self.username!r}, "
            f"password='{REDACTED}', source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# File permission check
# ---------------------------------------------------------------------------


def _check_file_permissions(path: Path) -> None:
    """Warn if the credentials file is readable by group or others.

    Mirrors the behavior of OpenSSH when key file permissions are too
    open. Does not block loading -- only logs a warning.

    Args:
        path: Absolute path to the credentials file.
    """
    try:
        file_stat = path.stat()
        mode = file_stat.st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            logger.warning(
                "SSH credentials file %s has permissions %o which are too "
                "open. It is recommended that the file is not accessible "
                "by group or others (chmod 600).",
                path,
                stat.S_IMODE(mode),
            )
    except OSError:
        # If we cannot stat the file, the load will fail anyway
        pass


# ---------------------------------------------------------------------------
# Credentials file loading
# ---------------------------------------------------------------------------


def load_credentials_file(
    path: Path | None = None,
) -> dict[str, SSHCredential]:
    """Load per-host credentials from a YAML file.

    The file must contain a top-level ``hosts`` mapping where each key
    is a hostname and each value has ``username`` and ``password`` fields.

    Args:
        path: Explicit path to the credentials file. If None, uses
            the JULES_SSH_CREDENTIALS_FILE env var or the default
            path (~/.jules/ssh_credentials.yaml).

    Returns:
        A dict mapping hostname to SSHCredential. Returns an empty
        dict if the file does not exist or cannot be parsed.
    """
    resolved_path = _resolve_credentials_path(path)

    if not resolved_path.is_file():
        return {}

    _check_file_permissions(resolved_path)

    try:
        raw = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "Failed to read SSH credentials file %s: %s",
            resolved_path,
            exc,
        )
        return {}

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning(
            "Failed to parse SSH credentials file %s: %s",
            resolved_path,
            exc,
        )
        return {}

    return _parse_credentials_data(data, str(resolved_path))


def _resolve_credentials_path(explicit: Path | None) -> Path:
    """Determine the credentials file path from args, env, or default."""
    if explicit is not None:
        return explicit

    env_path = os.environ.get(_ENV_CREDENTIALS_FILE)
    if env_path:
        return Path(env_path)

    return _DEFAULT_CREDENTIALS_PATH


def _parse_credentials_data(
    data: Any,
    source_path: str,
) -> dict[str, SSHCredential]:
    """Parse validated credential entries from raw YAML data.

    Args:
        data: Parsed YAML content (expected to be a dict with a
            ``hosts`` key).
        source_path: File path string for the ``source`` field.

    Returns:
        Dict mapping hostname to SSHCredential.
    """
    if not isinstance(data, dict):
        logger.warning(
            "SSH credentials file must contain a YAML mapping, "
            "got %s",
            type(data).__name__,
        )
        return {}

    hosts = data.get("hosts")
    if not isinstance(hosts, dict):
        logger.warning(
            "SSH credentials file must have a 'hosts' mapping"
        )
        return {}

    result: dict[str, SSHCredential] = {}
    for hostname, entry in hosts.items():
        if not isinstance(hostname, str) or not hostname.strip():
            logger.warning(
                "Skipping invalid hostname in credentials file: %r",
                hostname,
            )
            continue

        if not isinstance(entry, dict):
            logger.warning(
                "Skipping non-mapping entry for host %r in credentials file",
                hostname,
            )
            continue

        password = entry.get("password")
        if not isinstance(password, str) or not password:
            logger.warning(
                "Skipping host %r: missing or empty 'password' field",
                hostname,
            )
            continue

        username = entry.get("username")
        if username is not None and not isinstance(username, str):
            logger.warning(
                "Skipping host %r: 'username' must be a string",
                hostname,
            )
            continue

        clean_hostname = hostname.strip()
        result[clean_hostname] = SSHCredential(
            username=username if username else None,
            password=password,
            source=f"credentials_file:{source_path}",
        )

    return result


# ---------------------------------------------------------------------------
# Public resolution API
# ---------------------------------------------------------------------------


def resolve_ssh_credentials(
    hostname: str,
    *,
    credentials_file_path: Path | None = None,
) -> SSHCredential | None:
    """Resolve SSH password credentials for a given hostname.

    Resolution order:
    1. Per-host entry in the credentials file
    2. JULES_SSH_PASSWORD environment variable (single-host MVP)
    3. None (caller should fall back to key-based auth)

    The password is NEVER logged. Only the source is recorded for audit.

    Args:
        hostname: The SSH target hostname to look up.
        credentials_file_path: Optional explicit path to the credentials
            file. If None, uses env var or default path.

    Returns:
        SSHCredential if a password was found, None otherwise.
    """
    # 1. Check credentials file (per-host)
    creds_map = load_credentials_file(credentials_file_path)
    if hostname in creds_map:
        logger.info(
            "SSH credentials resolved for host %r from credentials file",
            hostname,
        )
        return creds_map[hostname]

    # 2. Check JULES_SSH_PASSWORD env var
    env_password = os.environ.get(_ENV_PASSWORD)
    if env_password:
        logger.info(
            "SSH credentials resolved for host %r from %s env var",
            hostname,
            _ENV_PASSWORD,
        )
        return SSHCredential(
            username=None,
            password=env_password,
            source=f"env:{_ENV_PASSWORD}",
        )

    # 3. No password available -- fall back to key auth
    logger.debug(
        "No SSH password credentials found for host %r",
        hostname,
    )
    return None
