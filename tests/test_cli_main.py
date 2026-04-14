"""Tests for the legacy CLI entry point's active input resolution path."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from jules_daemon.cli_main import (
    _notification_socket_path,
    _repl,
    _resolve_input,
)
from jules_daemon.thin_client.client import ThinClient, ThinClientConfig


class TestResolveInput:
    """The active CLI flow should route user text to daemon interpretation."""

    def test_empty_input_returns_error(self) -> None:
        result = _resolve_input("", allow_prompt=False)
        assert result.parts is None
        assert result.error == "Empty input."

    def test_health_remains_local_escape_hatch(self) -> None:
        result = _resolve_input("health", allow_prompt=False)
        assert result.parts == ("health",)
        assert result.hint is None
        assert result.error is None

    def test_interpret_remains_explicit_debug_escape_hatch(self) -> None:
        result = _resolve_input(
            "interpret give me the current status",
            allow_prompt=False,
        )
        assert result.parts == ("interpret", "give me the current status")
        assert result.hint is None
        assert result.error is None

    def test_structured_status_is_forwarded_to_daemon_interpretation(self) -> None:
        raw = "status --verbose"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("interpret", raw)
        assert result.hint == "(sent to daemon for interpretation)"
        assert result.error is None

    def test_structured_run_is_forwarded_to_daemon_interpretation(self) -> None:
        raw = "run deploy@staging run the smoke tests"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("interpret", raw)
        assert result.hint == "(sent to daemon for interpretation)"
        assert result.error is None

    def test_conversational_status_is_forwarded_to_daemon_interpretation(self) -> None:
        raw = "give me the current status"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("interpret", raw)
        assert result.hint == "(sent to daemon for interpretation)"
        assert result.error is None

    def test_conversational_run_is_forwarded_to_daemon_interpretation(self) -> None:
        raw = "run the smoke tests in tuto. 1 iteration"
        result = _resolve_input(raw, allow_prompt=False)
        assert result.parts == ("interpret", raw)
        assert result.hint == "(sent to daemon for interpretation)"
        assert result.error is None

    def test_no_local_target_prompt_for_conversational_run(self) -> None:
        raw = "run the smoke tests there"
        with patch("builtins.input") as mock_input:
            result = _resolve_input(raw, allow_prompt=True)
        assert result.parts == ("interpret", raw)
        assert result.hint == "(sent to daemon for interpretation)"
        assert result.error is None
        mock_input.assert_not_called()


class TestNotificationListener:
    """Interactive REPL should keep a background notification subscriber alive."""

    def test_notification_socket_path_uses_client_config(self) -> None:
        client = ThinClient(
            config=ThinClientConfig(socket_path=Path("/tmp/jules.sock"))
        )
        assert _notification_socket_path(client) == Path("/tmp/jules.sock")

    def test_notification_socket_path_uses_discovery_when_missing(self) -> None:
        client = ThinClient()
        with patch(
            "jules_daemon.ipc.socket_discovery.default_socket_path",
            return_value=Path("/tmp/discovered.sock"),
        ):
            assert _notification_socket_path(client) == Path("/tmp/discovered.sock")

    @pytest.mark.asyncio
    async def test_repl_starts_and_stops_notification_listener(self) -> None:
        client = ThinClient()
        task = asyncio.create_task(asyncio.sleep(3600))
        stop_mock = AsyncMock()

        with patch(
            "jules_daemon.cli_main._start_notification_listener",
            return_value=task,
        ) as start_mock, patch(
            "jules_daemon.cli_main._stop_notification_listener",
            stop_mock,
        ), patch(
            "builtins.input",
            side_effect=["quit"],
        ):
            exit_code = await _repl(client)

        assert exit_code == 0
        start_mock.assert_called_once_with(client)
        stop_mock.assert_awaited_once_with(task)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
