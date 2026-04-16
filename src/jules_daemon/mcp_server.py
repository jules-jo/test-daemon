"""MCP server adapter for exposing Jules daemon capabilities to agent frontends.

This is the first integration slice for a Copilot-SDK-style frontend:

- keep Jules as the durable backend/runtime
- expose a safe subset of Jules over MCP
- let an external chat/session layer orchestrate via MCP tools

The current tool surface intentionally focuses on conversational/status
operations that already behave well over a request/response boundary:

- ``jules_chat``   -- free-form informational prompt routed to daemon ``interpret``
- ``jules_status`` -- current workflow/run status
- ``jules_history`` -- recent run history
- ``jules_health`` -- backend liveness
- ``jules_cancel`` -- cancel current or specific run

Execution/proposal tools are intentionally deferred until the approval bridge is
designed cleanly for MCP-driven frontends.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from jules_daemon.thin_client.client import (
    CommandResult,
    ThinClient,
    ThinClientConfig,
)

__all__ = [
    "JulesMCPAdapter",
    "JulesMCPConfig",
    "build_server",
    "main",
]


_ENV_SOCKET_PATH = "JULES_SOCKET_PATH"


@dataclass(frozen=True)
class JulesMCPConfig:
    """Configuration for the Jules MCP adapter."""

    socket_path: Path | None = None

    @classmethod
    def from_env(cls) -> "JulesMCPConfig":
        raw = os.environ.get(_ENV_SOCKET_PATH, "").strip()
        return cls(socket_path=Path(raw) if raw else None)


def _build_client(config: JulesMCPConfig) -> ThinClient:
    return ThinClient(config=ThinClientConfig(socket_path=config.socket_path))


class JulesMCPAdapter:
    """Thin async adapter from MCP tools to the existing Jules daemon client."""

    def __init__(
        self,
        *,
        config: JulesMCPConfig | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._config = config or JulesMCPConfig.from_env()
        self._client_factory = client_factory or (
            lambda: _build_client(self._config)
        )

    @staticmethod
    def _command_result_to_payload(result: CommandResult) -> dict[str, Any]:
        response_payload = result.response.payload if result.response else None
        message = (
            response_payload.get("message")
            if isinstance(response_payload, dict)
            else None
        )
        return {
            "success": result.success,
            "verb": result.verb,
            "message": message or result.rendered,
            "rendered": result.rendered,
            "error": result.error,
            "payload": response_payload,
        }

    async def chat(self, prompt: str) -> dict[str, Any]:
        client = self._client_factory()
        result = await client.interpret(input_text=prompt)
        return self._command_result_to_payload(result)

    async def health(self) -> dict[str, Any]:
        client = self._client_factory()
        result = await client.health()
        return self._command_result_to_payload(result)

    async def status(self, *, verbose: bool = False) -> dict[str, Any]:
        client = self._client_factory()
        result = await client.status(verbose=verbose)
        return self._command_result_to_payload(result)

    async def history(
        self,
        *,
        limit: int = 10,
        status_filter: str | None = None,
        host_filter: str | None = None,
    ) -> dict[str, Any]:
        client = self._client_factory()
        result = await client.history(
            limit=limit,
            status_filter=status_filter,
            host_filter=host_filter,
        )
        return self._command_result_to_payload(result)

    async def cancel(
        self,
        *,
        run_id: str | None = None,
        force: bool = False,
        reason: str | None = None,
    ) -> dict[str, Any]:
        client = self._client_factory()
        result = await client.cancel(
            run_id=run_id,
            force=force,
            reason=reason,
        )
        return self._command_result_to_payload(result)


def build_server(
    *,
    adapter: JulesMCPAdapter | None = None,
) -> FastMCP:
    """Build the Jules MCP server with its registered tool set."""
    adapter = adapter or JulesMCPAdapter()
    server = FastMCP(
        name="jules",
        instructions=(
            "Jules is a backend runtime for remote test workflows. "
            "Use jules_chat for free-form informational questions, "
            "jules_status for current workflow state, "
            "jules_history for recent completed runs, "
            "jules_health for liveness checks, and "
            "jules_cancel to stop an active run."
        ),
    )

    @server.tool(
        name="jules_chat",
        description=(
            "Ask Jules a free-form informational question about tests, "
            "workflow status, or saved daemon knowledge."
        ),
    )
    async def jules_chat(prompt: str) -> dict[str, Any]:
        return await adapter.chat(prompt)

    @server.tool(
        name="jules_health",
        description="Check that the Jules backend daemon is reachable.",
    )
    async def jules_health() -> dict[str, Any]:
        return await adapter.health()

    @server.tool(
        name="jules_status",
        description="Get the current Jules workflow/run status.",
    )
    async def jules_status(verbose: bool = False) -> dict[str, Any]:
        return await adapter.status(verbose=verbose)

    @server.tool(
        name="jules_history",
        description="Get recent Jules run history entries.",
    )
    async def jules_history(
        limit: int = 10,
        status_filter: str | None = None,
        host_filter: str | None = None,
    ) -> dict[str, Any]:
        return await adapter.history(
            limit=limit,
            status_filter=status_filter,
            host_filter=host_filter,
        )

    @server.tool(
        name="jules_cancel",
        description="Cancel the current Jules run or a specific run id.",
    )
    async def jules_cancel(
        run_id: str | None = None,
        force: bool = False,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return await adapter.cancel(
            run_id=run_id,
            force=force,
            reason=reason,
        )

    return server


def main() -> None:
    """Run the Jules MCP server over stdio."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
