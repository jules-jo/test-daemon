"""check_remote_processes tool -- wraps execution.collision_check.

SSHes into the target host and runs ``ps aux`` to detect running test
processes that might conflict with a new test run. The LLM uses this
to decide whether to warn the user or proceed.

Extends InfoRetrievalTool (the base class for read-only tools) which
provides:
    - Automatic argument validation against parameters_schema
    - Exception-safe execution wrapping
    - OpenAI-compatible schema serialization (to_openai_schema)
    - ToolSpec conversion (to_tool_spec) for ToolRegistry integration

Delegates to:
    - jules_daemon.execution.collision_check.check_remote_processes
    - jules_daemon.ssh.credentials.resolve_ssh_credentials

No business logic is reimplemented -- this tool composes existing
functions and formats their output for the LLM conversation.

Usage::

    tool = CheckRemoteProcessesTool()

    # InfoRetrievalTool calling convention
    result = await tool.execute(call_id="c1", args={
        "host": "10.0.1.50",
        "username": "root",
        "port": 22,
    })

    # Legacy BaseTool calling convention (backward compat with ToolRegistry)
    result = await tool.execute({
        "host": "10.0.1.50",
        "username": "root",
        "port": 22,
        "_call_id": "c1",
    })
"""

from __future__ import annotations

import json
import logging
from typing import Any

from jules_daemon.agent.tool_base import InfoRetrievalTool
from jules_daemon.agent.tool_result import ToolResult
from jules_daemon.agent.tool_types import ToolSpec

__all__ = ["CheckRemoteProcessesTool"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports -- avoid importing paramiko at module level so tests can run
# without SSH infrastructure.
# ---------------------------------------------------------------------------


def _check_remote_processes(
    *,
    host: str,
    port: int,
    username: str,
    credential: Any,
    filter_pattern: str | None = None,
) -> Any:
    """Lazy wrapper for collision_check.check_remote_processes.

    Imported on first call to avoid paramiko import at module level.
    """
    from jules_daemon.execution.collision_check import check_remote_processes

    return check_remote_processes(
        host=host,
        port=port,
        username=username,
        credential=credential,
    )


def _resolve_ssh_credentials(host: str) -> Any:
    """Lazy wrapper for ssh.credentials.resolve_ssh_credentials.

    Imported on first call to avoid loading credentials infrastructure
    at module level.
    """
    from jules_daemon.ssh.credentials import resolve_ssh_credentials

    return resolve_ssh_credentials(host)


# ---------------------------------------------------------------------------
# CheckRemoteProcessesTool
# ---------------------------------------------------------------------------


class CheckRemoteProcessesTool(InfoRetrievalTool):
    """Check for running test processes on a remote host.

    Wraps:
        - execution.collision_check.check_remote_processes (SSH + ps aux)
        - ssh.credentials.resolve_ssh_credentials (credential lookup)

    This is a read-only tool (ApprovalRequirement.NONE) that extends
    InfoRetrievalTool. Inherits argument validation, error handling,
    and OpenAI schema serialization from the base class.
    """

    def __init__(self) -> None:
        self._spec_cache: ToolSpec | None = None

    # -- Protocol-required properties (InfoRetrievalTool) ------------------

    @property
    def name(self) -> str:
        """Unique tool identifier (function name in OpenAI API)."""
        return "check_remote_processes"

    @property
    def description(self) -> str:
        """Human-readable description shown to the LLM."""
        return (
            "Check for running test processes (pytest, python, npm test, etc.) "
            "on a remote host via SSH. Returns a list of detected processes "
            "with their PIDs and command lines. Use this before proposing "
            "a new test command to avoid conflicts."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema dict describing accepted arguments.

        Parameters:
            host (string, required): Remote hostname or IP address.
            username (string, required): SSH login username.
            port (integer, optional): SSH port number. Defaults to 22.
            filter_pattern (string, optional): Additional regex pattern to
                filter processes beyond the default test patterns.
        """
        return {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Remote hostname or IP address",
                },
                "username": {
                    "type": "string",
                    "description": "SSH login username",
                },
                "port": {
                    "type": "integer",
                    "description": "SSH port number",
                    "default": 22,
                },
                "filter_pattern": {
                    "type": "string",
                    "description": (
                        "Additional regex pattern to filter processes "
                        "beyond the default test patterns"
                    ),
                },
            },
            "required": ["host", "username"],
        }

    # -- Backward-compatible spec property for ToolRegistry ----------------

    @property
    def spec(self) -> ToolSpec:
        """Return a ToolSpec for ToolRegistry integration.

        Cached on first access. Equivalent to calling to_tool_spec()
        but avoids reconstructing the ToolSpec on every access.
        """
        if self._spec_cache is None:
            self._spec_cache = self.to_tool_spec()
        return self._spec_cache

    # -- Dual calling convention for execute -------------------------------

    async def execute(  # type: ignore[override]
        self, *pos_args: Any, **kw_args: Any
    ) -> ToolResult:
        """Execute with support for both calling conventions.

        InfoRetrievalTool convention::

            result = await tool.execute(call_id="c1", args={"host": "x", ...})
            result = await tool.execute("c1", {"host": "x", ...})

        Legacy BaseTool convention (used by ToolRegistry)::

            result = await tool.execute({"host": "x", ..., "_call_id": "c1"})

        When called with a dict as the first positional argument, the
        ``_call_id`` key is extracted and the rest is used as args.

        Returns:
            ToolResult -- never raises.
        """
        call_id: str
        args: dict[str, Any]

        if "call_id" in kw_args:
            # Keyword-based InfoRetrievalTool convention
            call_id = str(kw_args["call_id"])
            args = dict(kw_args.get("args") or {})
        elif len(pos_args) >= 1 and isinstance(pos_args[0], dict):
            # Legacy BaseTool convention: execute(args_dict)
            legacy_args = dict(pos_args[0])
            call_id = str(legacy_args.pop("_call_id", "check_remote_processes"))
            args = legacy_args
        elif len(pos_args) >= 1 and isinstance(pos_args[0], str):
            # Positional InfoRetrievalTool convention: execute(call_id, args)
            call_id = pos_args[0]
            args = dict(pos_args[1]) if len(pos_args) > 1 else {}
        else:
            # Fallback -- try kwargs as legacy dict
            call_id = str(kw_args.pop("_call_id", "check_remote_processes"))
            args = dict(kw_args)

        return await super().execute(call_id=call_id, args=args)

    # -- Core execution logic ----------------------------------------------

    async def _execute_impl(
        self, *, call_id: str, args: dict[str, Any]
    ) -> ToolResult:
        """Check for running test processes on the remote host.

        Delegates to the existing collision_check module which handles
        SSH connection via asyncio.to_thread internally.

        Args:
            call_id: Unique identifier for this invocation.
            args: Validated arguments with at least 'host' and 'username'.

        Returns:
            ToolResult with JSON output containing host, processes_found,
            and a processes list, or an error result on failure.
        """
        host = args.get("host", "")
        username = args.get("username", "")
        port = args.get("port", 22)
        filter_pattern = args.get("filter_pattern")

        if not host or not host.strip():
            return ToolResult.error(
                call_id=call_id,
                tool_name=self.name,
                error_message="host parameter is required and must not be empty",
            )
        if not username or not username.strip():
            return ToolResult.error(
                call_id=call_id,
                tool_name=self.name,
                error_message="username parameter is required and must not be empty",
            )

        clean_host = host.strip()
        clean_username = username.strip()

        try:
            credential = _resolve_ssh_credentials(clean_host)
        except Exception as exc:
            logger.warning(
                "Credential resolution failed for %s: %s",
                clean_host,
                exc,
            )
            return ToolResult.error(
                call_id=call_id,
                tool_name=self.name,
                error_message=f"Remote process check failed: {exc}",
            )

        try:
            processes = await _check_remote_processes(
                host=clean_host,
                port=int(port),
                username=clean_username,
                credential=credential,
                filter_pattern=filter_pattern,
            )
        except Exception as exc:
            logger.warning(
                "check_remote_processes failed for %s@%s:%s: %s",
                clean_username,
                clean_host,
                port,
                exc,
            )
            return ToolResult.error(
                call_id=call_id,
                tool_name=self.name,
                error_message=f"Remote process check failed: {exc}",
            )

        result_data = {
            "host": clean_host,
            "processes_found": len(processes),
            "processes": [
                {"pid": p.pid, "command": p.command}
                for p in processes
            ],
        }

        return ToolResult.success(
            call_id=call_id,
            tool_name=self.name,
            output=json.dumps(result_data),
        )
