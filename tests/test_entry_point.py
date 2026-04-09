"""Tests for the unified CLI entry point.

Verifies that the entry point:
- Accepts raw user input (structured commands, NL text, verb aliases)
- Passes input through the classifier to determine input type
- Resolves the canonical verb via the handler registry
- Dispatches to the matched handler with typed *Args dataclasses
- Produces identical handler calls regardless of input style
- Returns structured InputProcessingResult (never raises)
- Handles empty/whitespace input gracefully
- Reports classification details for diagnostics
- Falls back from structured path to NL path when needed
- Handles missing handler registration gracefully
- Works without LLM classifier (pure deterministic mode)
"""

from __future__ import annotations

from typing import Any

import pytest

from jules_daemon.cli.dispatcher import CommandDispatcher
from jules_daemon.cli.entry_point import (
    InputProcessingResult,
    process_input,
)
from jules_daemon.cli.registry import CommandHandlerRegistry
from jules_daemon.cli.verbs import (
    CancelArgs,
    HistoryArgs,
    ParsedCommand,
    QueueArgs,
    RunArgs,
    StatusArgs,
    Verb,
    VerbArgs,
    WatchArgs,
)
from jules_daemon.classifier.models import InputType


# ---------------------------------------------------------------------------
# Fake handlers
# ---------------------------------------------------------------------------


async def fake_status_handler(args: VerbArgs) -> dict[str, Any]:
    status_args: StatusArgs = args  # type: ignore[assignment]
    return {"status": "idle", "verbose": status_args.verbose}


async def fake_watch_handler(args: VerbArgs) -> dict[str, Any]:
    watch_args: WatchArgs = args  # type: ignore[assignment]
    return {"watching": True, "run_id": watch_args.run_id}


async def fake_run_handler(args: VerbArgs) -> dict[str, Any]:
    run_args: RunArgs = args  # type: ignore[assignment]
    return {"started": True, "host": run_args.target_host}


async def fake_queue_handler(args: VerbArgs) -> dict[str, Any]:
    queue_args: QueueArgs = args  # type: ignore[assignment]
    return {"queued": True, "priority": queue_args.priority}


async def fake_cancel_handler(args: VerbArgs) -> dict[str, Any]:
    cancel_args: CancelArgs = args  # type: ignore[assignment]
    return {"cancelled": True, "force": cancel_args.force}


async def fake_history_handler(args: VerbArgs) -> dict[str, Any]:
    history_args: HistoryArgs = args  # type: ignore[assignment]
    return {"records": [], "limit": history_args.limit}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_registry() -> CommandHandlerRegistry:
    """Registry with all six verbs registered."""
    return (
        CommandHandlerRegistry()
        .register(
            verb=Verb.STATUS,
            handler=fake_status_handler,
            description="Query current run state",
            parameter_schema=StatusArgs,
        )
        .register(
            verb=Verb.WATCH,
            handler=fake_watch_handler,
            description="Live-stream output",
            parameter_schema=WatchArgs,
        )
        .register(
            verb=Verb.RUN,
            handler=fake_run_handler,
            description="Start test execution",
            parameter_schema=RunArgs,
        )
        .register(
            verb=Verb.QUEUE,
            handler=fake_queue_handler,
            description="Queue command for later",
            parameter_schema=QueueArgs,
        )
        .register(
            verb=Verb.CANCEL,
            handler=fake_cancel_handler,
            description="Cancel run",
            parameter_schema=CancelArgs,
        )
        .register(
            verb=Verb.HISTORY,
            handler=fake_history_handler,
            description="View past results",
            parameter_schema=HistoryArgs,
        )
    )


@pytest.fixture
def full_dispatcher() -> CommandDispatcher:
    """Dispatcher with all six verb handlers."""
    return (
        CommandDispatcher()
        .with_handler(Verb.STATUS, fake_status_handler)
        .with_handler(Verb.WATCH, fake_watch_handler)
        .with_handler(Verb.RUN, fake_run_handler)
        .with_handler(Verb.QUEUE, fake_queue_handler)
        .with_handler(Verb.CANCEL, fake_cancel_handler)
        .with_handler(Verb.HISTORY, fake_history_handler)
    )


# ---------------------------------------------------------------------------
# Structured command dispatch
# ---------------------------------------------------------------------------


class TestStructuredCommandDispatch:
    """Tests for processing structured CLI-style input."""

    @pytest.mark.asyncio
    async def test_status_command(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "status",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is True
        assert result.dispatch_response is not None
        assert result.dispatch_response.success is True
        assert result.dispatch_response.verb == Verb.STATUS
        assert result.dispatch_response.payload == {
            "status": "idle",
            "verbose": False,
        }

    @pytest.mark.asyncio
    async def test_status_verbose(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "status --verbose",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is True
        assert result.dispatch_response is not None
        assert result.dispatch_response.payload["verbose"] is True

    @pytest.mark.asyncio
    async def test_run_command(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "run deploy@staging.example.com run the smoke tests",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is True
        assert result.dispatch_response is not None
        assert result.dispatch_response.verb == Verb.RUN
        assert result.dispatch_response.payload["started"] is True
        assert result.dispatch_response.payload["host"] == "staging.example.com"

    @pytest.mark.asyncio
    async def test_cancel_command(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "cancel --force",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is True
        assert result.dispatch_response is not None
        assert result.dispatch_response.verb == Verb.CANCEL
        assert result.dispatch_response.payload["force"] is True

    @pytest.mark.asyncio
    async def test_history_command(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "history --limit 50 --status failed",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is True
        assert result.dispatch_response is not None
        assert result.dispatch_response.verb == Verb.HISTORY
        assert result.dispatch_response.payload["limit"] == 50

    @pytest.mark.asyncio
    async def test_watch_command(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "watch --run-id abc-123",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is True
        assert result.dispatch_response is not None
        assert result.dispatch_response.verb == Verb.WATCH

    @pytest.mark.asyncio
    async def test_queue_command(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "queue ci@prod.example.com run smoke tests --priority 5",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is True
        assert result.dispatch_response is not None
        assert result.dispatch_response.verb == Verb.QUEUE
        assert result.dispatch_response.payload["priority"] == 5


# ---------------------------------------------------------------------------
# Alias-based command dispatch
# ---------------------------------------------------------------------------


class TestAliasCommandDispatch:
    """Tests for processing alias-based structured input."""

    @pytest.mark.asyncio
    async def test_execute_alias_routes_to_run(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "execute ci@staging run the tests",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is True
        assert result.dispatch_response is not None
        assert result.dispatch_response.verb == Verb.RUN

    @pytest.mark.asyncio
    async def test_stop_alias_routes_to_cancel(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "stop --force",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is True
        assert result.dispatch_response is not None
        assert result.dispatch_response.verb == Verb.CANCEL

    @pytest.mark.asyncio
    async def test_tail_alias_routes_to_watch(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "tail",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is True
        assert result.dispatch_response is not None
        assert result.dispatch_response.verb == Verb.WATCH

    @pytest.mark.asyncio
    async def test_check_alias_routes_to_status(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "check --verbose",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is True
        assert result.dispatch_response is not None
        assert result.dispatch_response.verb == Verb.STATUS

    @pytest.mark.asyncio
    async def test_past_alias_routes_to_history(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "past",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is True
        assert result.dispatch_response is not None
        assert result.dispatch_response.verb == Verb.HISTORY


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error conditions in the entry point."""

    @pytest.mark.asyncio
    async def test_empty_input(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_whitespace_input(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "   ",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_parse_error_structured(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        """Structured parse error (missing required target for run)."""
        result = await process_input(
            "run",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        # Should fail because run requires user@host + NL
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_unregistered_handler(
        self,
        full_registry: CommandHandlerRegistry,
    ) -> None:
        """Dispatching to an empty dispatcher should fail gracefully."""
        empty_dispatcher = CommandDispatcher()
        result = await process_input(
            "status",
            registry=full_registry,
            dispatcher=empty_dispatcher,
        )
        # Dispatch should fail because no handler registered
        assert result.success is False
        assert result.dispatch_response is not None
        assert result.dispatch_response.success is False


# ---------------------------------------------------------------------------
# InputProcessingResult structure
# ---------------------------------------------------------------------------


class TestInputProcessingResult:
    """Tests for the InputProcessingResult data model."""

    @pytest.mark.asyncio
    async def test_result_is_frozen(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "status",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_result_contains_classification(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "status --verbose",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.classification is not None
        assert result.classification.canonical_verb == "status"

    @pytest.mark.asyncio
    async def test_result_contains_parsed_command(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "cancel --force",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result.parsed_command is not None
        assert result.parsed_command.verb == Verb.CANCEL

    @pytest.mark.asyncio
    async def test_result_to_dict(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        result = await process_input(
            "status",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert "classification" in d
        assert "dispatch_response" in d


# ---------------------------------------------------------------------------
# Cross-path equivalence: structured vs alias produce identical dispatch
# ---------------------------------------------------------------------------


class TestCrossPathEquivalenceE2E:
    """End-to-end tests ensuring identical handler calls from different paths.

    The unified entry point contract: regardless of how the user expresses
    their intent (exact verb, alias, or NL), the handler receives the same
    typed *Args dataclass.
    """

    @pytest.mark.asyncio
    async def test_cancel_vs_stop_same_handler_result(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        """'cancel --force' and 'stop --force' must produce identical results."""
        result_cancel = await process_input(
            "cancel --force",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        result_stop = await process_input(
            "stop --force",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result_cancel.success is True
        assert result_stop.success is True
        assert result_cancel.dispatch_response is not None
        assert result_stop.dispatch_response is not None
        assert (
            result_cancel.dispatch_response.payload
            == result_stop.dispatch_response.payload
        )
        assert (
            result_cancel.dispatch_response.verb
            == result_stop.dispatch_response.verb
        )

    @pytest.mark.asyncio
    async def test_status_vs_check_same_handler_result(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        """'status' and 'check' must produce identical results."""
        result_status = await process_input(
            "status",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        result_check = await process_input(
            "check",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result_status.success is True
        assert result_check.success is True
        assert result_status.dispatch_response is not None
        assert result_check.dispatch_response is not None
        assert (
            result_status.dispatch_response.payload
            == result_check.dispatch_response.payload
        )

    @pytest.mark.asyncio
    async def test_history_vs_past_same_handler_result(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        """'history' and 'past' must produce identical results."""
        result_history = await process_input(
            "history",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        result_past = await process_input(
            "past",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result_history.success is True
        assert result_past.success is True
        assert result_history.dispatch_response is not None
        assert result_past.dispatch_response is not None
        assert (
            result_history.dispatch_response.payload
            == result_past.dispatch_response.payload
        )

    @pytest.mark.asyncio
    async def test_run_vs_execute_same_handler_result(
        self,
        full_registry: CommandHandlerRegistry,
        full_dispatcher: CommandDispatcher,
    ) -> None:
        """'run' and 'execute' with same args produce identical results."""
        result_run = await process_input(
            "run ci@staging run the tests",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        result_exec = await process_input(
            "execute ci@staging run the tests",
            registry=full_registry,
            dispatcher=full_dispatcher,
        )
        assert result_run.success is True
        assert result_exec.success is True
        assert result_run.dispatch_response is not None
        assert result_exec.dispatch_response is not None
        assert (
            result_run.dispatch_response.payload
            == result_exec.dispatch_response.payload
        )
