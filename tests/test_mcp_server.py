from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from jules_daemon.ipc.framing import MessageEnvelope, MessageType
from jules_daemon.mcp_server import (
    JulesMCPAdapter,
    JulesMCPConfig,
    build_server,
)
from jules_daemon.thin_client.client import CommandResult


@dataclass
class _FakeClient:
    interpret_result: CommandResult | None = None
    health_result: CommandResult | None = None
    status_result: CommandResult | None = None
    history_result: CommandResult | None = None
    cancel_result: CommandResult | None = None

    last_prompt: str | None = None
    last_verbose: bool | None = None
    last_history: tuple[int, str | None, str | None] | None = None
    last_cancel: tuple[str | None, bool, str | None] | None = None

    async def interpret(self, *, input_text: str) -> CommandResult:
        self.last_prompt = input_text
        assert self.interpret_result is not None
        return self.interpret_result

    async def health(self) -> CommandResult:
        assert self.health_result is not None
        return self.health_result

    async def status(self, *, verbose: bool = False) -> CommandResult:
        self.last_verbose = verbose
        assert self.status_result is not None
        return self.status_result

    async def history(
        self,
        *,
        limit: int = 20,
        status_filter: str | None = None,
        host_filter: str | None = None,
    ) -> CommandResult:
        self.last_history = (limit, status_filter, host_filter)
        assert self.history_result is not None
        return self.history_result

    async def cancel(
        self,
        *,
        run_id: str | None = None,
        force: bool = False,
        reason: str | None = None,
    ) -> CommandResult:
        self.last_cancel = (run_id, force, reason)
        assert self.cancel_result is not None
        return self.cancel_result


def _result(
    *,
    verb: str,
    success: bool = True,
    payload: dict | None = None,
    rendered: str = "",
    error: str | None = None,
) -> CommandResult:
    response = None
    if payload is not None:
        response = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="mcp-test",
            timestamp="2026-04-16T00:00:00Z",
            payload=payload,
        )
    rendered_text = rendered
    if not rendered_text and payload is not None:
        rendered_text = str(payload.get("message", ""))
    return CommandResult(
        success=success,
        verb=verb,
        response=response,
        rendered=rendered_text,
        error=error,
    )


class TestJulesMCPAdapter:
    @pytest.mark.asyncio
    async def test_chat_uses_interpret_and_prefers_payload_message(self) -> None:
        fake = _FakeClient(
            interpret_result=_result(
                verb="interpret",
                payload={
                    "verb": "interpret",
                    "status": "answered",
                    "message": "The saved test file path is /root/step.py",
                },
            ),
        )
        adapter = JulesMCPAdapter(client_factory=lambda: fake)

        result = await adapter.chat("where is the test script?")

        assert fake.last_prompt == "where is the test script?"
        assert result["success"] is True
        assert result["message"] == "The saved test file path is /root/step.py"
        assert result["payload"]["status"] == "answered"

    @pytest.mark.asyncio
    async def test_status_passes_verbose_flag(self) -> None:
        fake = _FakeClient(
            status_result=_result(
                verb="status",
                payload={"verb": "status", "status": "idle"},
            ),
        )
        adapter = JulesMCPAdapter(client_factory=lambda: fake)

        result = await adapter.status(verbose=True)

        assert fake.last_verbose is True
        assert result["payload"]["status"] == "idle"

    @pytest.mark.asyncio
    async def test_history_passes_filters(self) -> None:
        fake = _FakeClient(
            history_result=_result(
                verb="history",
                payload={"verb": "history", "entries": []},
            ),
        )
        adapter = JulesMCPAdapter(client_factory=lambda: fake)

        result = await adapter.history(
            limit=5,
            status_filter="failed",
            host_filter="tuto",
        )

        assert fake.last_history == (5, "failed", "tuto")
        assert result["verb"] == "history"

    @pytest.mark.asyncio
    async def test_cancel_passes_args(self) -> None:
        fake = _FakeClient(
            cancel_result=_result(
                verb="cancel",
                payload={"verb": "cancel", "status": "cancelled"},
            ),
        )
        adapter = JulesMCPAdapter(client_factory=lambda: fake)

        result = await adapter.cancel(
            run_id="run-123",
            force=True,
            reason="user requested stop",
        )

        assert fake.last_cancel == ("run-123", True, "user requested stop")
        assert result["payload"]["status"] == "cancelled"


class TestJulesMCPConfig:
    def test_from_env_reads_socket_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JULES_SOCKET_PATH", "/tmp/jules.sock")

        config = JulesMCPConfig.from_env()

        assert config.socket_path == Path("/tmp/jules.sock")


@pytest.mark.asyncio
async def test_build_server_registers_expected_tools() -> None:
    server = build_server(adapter=JulesMCPAdapter(client_factory=lambda: _FakeClient()))
    tool_names = {tool.name for tool in await server.list_tools()}

    assert tool_names == {
        "jules_chat",
        "jules_health",
        "jules_status",
        "jules_history",
        "jules_cancel",
    }
