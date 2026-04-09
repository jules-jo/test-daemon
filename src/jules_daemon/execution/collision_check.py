"""Remote process collision detection via SSH.

Connects to a remote host via paramiko and runs ``ps aux`` to detect
running test processes (pytest, python, npm test, node, go test) that
might conflict with a new test run.

This module is intentionally decoupled from the startup collision
detector, which scans for *daemon* processes locally. This module
checks for *test* processes on the remote target host.

Usage::

    from jules_daemon.execution.collision_check import check_remote_processes

    processes = await check_remote_processes(
        host="10.0.1.50", port=22, username="root", credential=None,
    )
    if processes:
        for p in processes:
            print(f"PID={p.pid} CMD={p.command}")
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

import paramiko

from jules_daemon.ssh.credentials import SSHCredential

__all__ = [
    "ProcessInfo",
    "check_remote_processes",
    "format_collision_warning",
]

logger = logging.getLogger(__name__)

# Patterns that indicate a test process is running.
# Each pattern is compiled as a case-insensitive regex.
_TEST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bpytest\b", re.IGNORECASE),
    re.compile(r"\bpython\b.*\btest", re.IGNORECASE),
    re.compile(r"\bnpm\s+test\b", re.IGNORECASE),
    re.compile(r"\bnode\b.*\btest", re.IGNORECASE),
    re.compile(r"\bgo\s+test\b", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessInfo:
    """A single remote process detected as a potential collision.

    Attributes:
        pid: Remote process ID.
        command: Full command line of the process.
    """

    pid: int
    command: str


# ---------------------------------------------------------------------------
# Internal: blocking paramiko call
# ---------------------------------------------------------------------------


def _run_ps_via_ssh(
    *,
    host: str,
    port: int,
    username: str,
    credential: SSHCredential | None,
) -> str:
    """SSH into the host and run ``ps aux`` (blocking).

    Returns the raw stdout output. Raises on connection failure.
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs: dict[str, Any] = {
        "hostname": host,
        "port": port,
        "username": username,
        "timeout": 15,
        "allow_agent": True,
        "look_for_keys": True,
    }

    if credential is not None:
        connect_kwargs["password"] = credential.password
        connect_kwargs["allow_agent"] = False
        connect_kwargs["look_for_keys"] = False
        if credential.username is not None:
            connect_kwargs["username"] = credential.username

    try:
        client.connect(**connect_kwargs)
        _, stdout_ch, _ = client.exec_command("ps aux", timeout=15)
        output = stdout_ch.read().decode("utf-8", errors="replace")
        return output
    finally:
        client.close()


def _parse_ps_aux(raw_output: str) -> list[ProcessInfo]:
    """Parse ``ps aux`` output and filter for test process patterns.

    Args:
        raw_output: Raw stdout from ``ps aux``.

    Returns:
        List of ProcessInfo for matching lines.
    """
    if not raw_output or not raw_output.strip():
        return []

    lines = raw_output.strip().splitlines()
    if len(lines) <= 1:
        return []

    results: list[ProcessInfo] = []
    for line in lines[1:]:  # skip header
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line matches any test pattern
        if not any(pat.search(stripped) for pat in _TEST_PATTERNS):
            continue

        # Parse PID from ps aux output (USER PID ...)
        parts = stripped.split(None, 10)
        if len(parts) < 11:
            # Minimal: USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
            # At least need USER + PID + some command
            if len(parts) >= 2:
                try:
                    pid = int(parts[1])
                except ValueError:
                    continue
                cmd = " ".join(parts[10:]) if len(parts) > 10 else stripped
                results.append(ProcessInfo(pid=pid, command=cmd))
            continue

        try:
            pid = int(parts[1])
        except ValueError:
            continue

        command = parts[10]
        results.append(ProcessInfo(pid=pid, command=command))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_collision_warning(processes: list[ProcessInfo]) -> str:
    """Format detected remote processes into a human-readable warning.

    Args:
        processes: List of detected test processes.

    Returns:
        Multi-line warning string suitable for display in the CLI.
    """
    lines = [
        "",
        "WARNING: Test processes detected on remote host:",
        "",
    ]
    for proc in processes:
        lines.append(f"  PID {proc.pid}: {proc.command[:120]}")
    lines.append("")
    lines.append("Proceeding may cause conflicts with existing test runs.")
    lines.append("")
    return "\n".join(lines)


async def check_remote_processes(
    host: str,
    port: int,
    username: str,
    credential: SSHCredential | None,
) -> list[ProcessInfo]:
    """SSH into host, run ps aux, grep for test processes.

    The paramiko call runs in a thread pool to avoid blocking the
    asyncio event loop.

    Args:
        host: Remote hostname or IP address.
        port: SSH port number.
        username: SSH login username.
        credential: Resolved SSH credential (password), or None for
            key-based auth.

    Returns:
        List of ProcessInfo for detected test processes. Returns an
        empty list on connection failure (logs a warning).
    """
    try:
        raw_output = await asyncio.to_thread(
            _run_ps_via_ssh,
            host=host,
            port=port,
            username=username,
            credential=credential,
        )
    except Exception as exc:
        logger.warning(
            "Collision check failed for %s@%s:%d: %s",
            username,
            host,
            port,
            exc,
        )
        return []

    return _parse_ps_aux(raw_output)
