"""Remote workflow preflight helpers.

This module performs deterministic preflight checks before a workflow-
aware run enters the agent loop. The first supported probe type is a
remote path existence check over SSH for artifact requirements that look
like concrete filesystem paths.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from collections.abc import Iterable
from typing import Any

import paramiko

from jules_daemon.ssh.credentials import resolve_ssh_credentials
from jules_daemon.ssh.errors import SSHAuthenticationError, SSHConnectionError
from jules_daemon.workflows.models import ArtifactState, ArtifactStatus

__all__ = [
    "artifact_presence_from_states",
    "build_unverified_artifact_prompt",
    "inspect_workflow_artifacts",
    "is_probeable_artifact_requirement",
]

logger = logging.getLogger(__name__)


def is_probeable_artifact_requirement(value: str) -> bool:
    """Return True when an artifact requirement looks like a remote path."""
    stripped = value.strip()
    if not stripped:
        return False
    return stripped.startswith(("/", "./", "../", "~/")) or "/" in stripped


def artifact_presence_from_states(
    states: Iterable[ArtifactState],
) -> dict[str, bool | None]:
    """Convert artifact states into planner-ready presence values."""
    presence: dict[str, bool | None] = {}
    for state in states:
        if state.status is ArtifactStatus.PRESENT:
            presence[state.name] = True
        elif state.status is ArtifactStatus.MISSING:
            presence[state.name] = False
        else:
            presence[state.name] = None
    return presence


def build_unverified_artifact_prompt(
    artifact_names: Iterable[str],
    *,
    prerequisite_steps: Iterable[str] = (),
) -> str:
    """Build an accurate user-facing prompt for unverified artifacts."""
    names = [name.strip() for name in artifact_names if name.strip()]
    prereq_names = [name.strip() for name in prerequisite_steps if name.strip()]
    if not names:
        return "Jules could not verify some required artifacts automatically."
    if prereq_names:
        return (
            "Jules could not verify required artifacts automatically: "
            + ", ".join(names)
            + ". Do you want me to run "
            + ", ".join(prereq_names)
            + " first?"
        )
    return (
        "Jules could not verify required artifacts automatically: "
        + ", ".join(names)
        + ". Do you want me to continue anyway?"
    )


def _build_artifact_probe_command(artifact_paths: Iterable[str]) -> str:
    """Build one remote shell command that prints presence per path."""
    commands: list[str] = []
    for path in artifact_paths:
        quoted_path = shlex.quote(path)
        commands.append(
            "if test -e -- {path}; then "
            "printf '%s\\tpresent\\n' {path}; "
            "else printf '%s\\tmissing\\n' {path}; fi".format(
                path=quoted_path,
            )
        )
    return "sh -lc " + shlex.quote("; ".join(commands))


def _probe_artifact_paths_via_ssh(
    *,
    host: str,
    port: int,
    username: str,
    artifact_paths: tuple[str, ...],
) -> dict[str, ArtifactStatus]:
    """SSH into the host and probe remote path existence (blocking)."""
    credential = resolve_ssh_credentials(host)
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
    except paramiko.AuthenticationException as exc:
        raise SSHAuthenticationError(
            f"Authentication failed for {username}@{host}:{port}: {exc}"
        ) from exc
    except (
        paramiko.SSHException,
        OSError,
        TimeoutError,
        ConnectionRefusedError,
    ) as exc:
        raise SSHConnectionError(
            f"Connection failed to {host}:{port}: {exc}"
        ) from exc

    try:
        command = _build_artifact_probe_command(artifact_paths)
        _, stdout_channel, stderr_channel = client.exec_command(command, timeout=20)
        stdout_text = stdout_channel.read().decode("utf-8", errors="replace")
        stderr_text = stderr_channel.read().decode("utf-8", errors="replace")
        exit_code = stdout_channel.channel.recv_exit_status()
        if exit_code != 0:
            summary = stderr_text.strip() or stdout_text.strip() or "probe failed"
            raise RuntimeError(summary)

        statuses: dict[str, ArtifactStatus] = {}
        for raw_line in stdout_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            path, _, raw_status = line.partition("\t")
            status = raw_status.strip().lower()
            if not path:
                continue
            if status == "present":
                statuses[path] = ArtifactStatus.PRESENT
            elif status == "missing":
                statuses[path] = ArtifactStatus.MISSING
        return statuses
    finally:
        client.close()


async def inspect_workflow_artifacts(
    *,
    host: str,
    port: int,
    username: str,
    artifact_requirements: Iterable[str],
) -> tuple[ArtifactState, ...]:
    """Inspect workflow artifacts and return durable artifact states.

    Probeable requirements (path-like strings) are checked remotely. All
    other artifact requirements remain ``UNKNOWN`` so higher layers can
    decide whether to ask the user or continue.
    """
    artifact_names = tuple(
        requirement.strip()
        for requirement in artifact_requirements
        if requirement.strip()
    )
    if not artifact_names:
        return ()

    probeable = tuple(
        name for name in artifact_names if is_probeable_artifact_requirement(name)
    )
    statuses: dict[str, ArtifactStatus] = {}
    probe_error: str | None = None
    if probeable:
        try:
            statuses = await asyncio.to_thread(
                _probe_artifact_paths_via_ssh,
                host=host,
                port=port,
                username=username,
                artifact_paths=probeable,
            )
        except Exception as exc:
            probe_error = str(exc)
            logger.warning(
                "Workflow artifact probe failed for %s@%s:%d: %s",
                username,
                host,
                port,
                exc,
            )

    states: list[ArtifactState] = []
    for artifact_name in artifact_names:
        if artifact_name in statuses:
            status = statuses[artifact_name]
            details = (
                "Verified remote path exists."
                if status is ArtifactStatus.PRESENT
                else "Verified remote path is missing."
            )
        elif is_probeable_artifact_requirement(artifact_name):
            status = ArtifactStatus.UNKNOWN
            details = (
                f"Automatic path check failed: {probe_error}"
                if probe_error
                else "Path check did not return a result."
            )
        else:
            status = ArtifactStatus.UNKNOWN
            details = (
                "Artifact requirement is not an explicit remote path, "
                "so Jules cannot verify it automatically yet."
            )
        states.append(
            ArtifactState(
                name=artifact_name,
                status=status,
                details=details,
            )
        )
    return tuple(states)
