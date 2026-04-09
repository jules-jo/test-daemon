"""Tests for the CLI command dispatcher.

Verifies that the command dispatcher:
- Maintains an immutable registry mapping Verb -> handler callable
- Routes ParsedCommand to the correct handler with parsed arguments
- Returns a structured DispatchResponse with success/failure info
- Rejects unknown verbs with a clear error (no crash)
- Supports registering handlers for all six verbs
- Is frozen (immutable after creation)
- Creates new dispatcher instances via with_handler (no mutation)
- Handles handler exceptions gracefully (captures in result)
- Validates handler callable at registration time
- Provides a factory function for convenient construction
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

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
from jules_daemon.cli.dispatcher import (
    CommandDispatcher,
    DispatchResponse,
    create_dispatcher,
)


# ---------------------------------------------------------------------------
# Fake handlers for testing
# ---------------------------------------------------------------------------


async def fake_status_handler(args: VerbArgs) -> dict[str, Any]:
    """Return a mock status response."""
    # ParsedCommand.__post_init__ guarantees args type matches verb
    status_args: StatusArgs = args  # type: ignore[assignment]
    return {"status": "idle", "verbose": status_args.verbose}


async def fake_watch_handler(args: VerbArgs) -> dict[str, Any]:
    """Return a mock watch response."""
    watch_args: WatchArgs = args  # type: ignore[assignment]
    return {"watching": True, "run_id": watch_args.run_id}


async def fake_run_handler(args: VerbArgs) -> dict[str, Any]:
    """Return a mock run response."""
    run_args: RunArgs = args  # type: ignore[assignment]
    return {"started": True, "host": run_args.target_host}


async def fake_queue_handler(args: VerbArgs) -> dict[str, Any]:
    """Return a mock queue response."""
    queue_args: QueueArgs = args  # type: ignore[assignment]
    return {"queued": True, "priority": queue_args.priority}


async def fake_cancel_handler(args: VerbArgs) -> dict[str, Any]:
    """Return a mock cancel response."""
    cancel_args: CancelArgs = args  # type: ignore[assignment]
    return {"cancelled": True, "force": cancel_args.force}


async def fake_history_handler(args: VerbArgs) -> dict[str, Any]:
    """Return a mock history response."""
    history_args: HistoryArgs = args  # type: ignore[assignment]
    return {"records": [], "limit": history_args.limit}


async def exploding_handler(args: VerbArgs) -> dict[str, Any]:
    """Handler that always raises an exception."""
    raise RuntimeError("handler blew up")


# ---------------------------------------------------------------------------
# DispatchResponse data model
# ---------------------------------------------------------------------------


class TestDispatchResponse:
    """Tests for the DispatchResponse frozen dataclass."""

    def test_frozen(self) -> None:
        resp = DispatchResponse(
            success=True,
            verb=Verb.STATUS,
            payload={"status": "idle"},
            error=None,
        )
        with pytest.raises(FrozenInstanceError):
            resp.success = False  # type: ignore[misc]

    def test_success_response(self) -> None:
        resp = DispatchResponse(
            success=True,
            verb=Verb.RUN,
            payload={"started": True},
            error=None,
        )
        assert resp.success is True
        assert resp.verb == Verb.RUN
        assert resp.payload == {"started": True}
        assert resp.error is None

    def test_error_response(self) -> None:
        resp = DispatchResponse(
            success=False,
            verb=Verb.CANCEL,
            payload=None,
            error="No active run to cancel",
        )
        assert resp.success is False
        assert resp.verb == Verb.CANCEL
        assert resp.payload is None
        assert resp.error == "No active run to cancel"


# ---------------------------------------------------------------------------
# CommandDispatcher: construction and immutability
# ---------------------------------------------------------------------------


class TestDispatcherConstruction:
    """Tests for dispatcher construction and immutability."""

    def test_empty_dispatcher(self) -> None:
        dispatcher = CommandDispatcher()
        assert dispatcher.registered_verbs == frozenset()

    def test_single_handler_registration(self) -> None:
        dispatcher = CommandDispatcher().with_handler(
            Verb.STATUS, fake_status_handler
        )
        assert Verb.STATUS in dispatcher.registered_verbs

    def test_multiple_handler_registration(self) -> None:
        dispatcher = (
            CommandDispatcher()
            .with_handler(Verb.STATUS, fake_status_handler)
            .with_handler(Verb.RUN, fake_run_handler)
            .with_handler(Verb.CANCEL, fake_cancel_handler)
        )
        assert dispatcher.registered_verbs == frozenset({
            Verb.STATUS, Verb.RUN, Verb.CANCEL
        })

    def test_with_handler_returns_new_instance(self) -> None:
        original = CommandDispatcher()
        updated = original.with_handler(Verb.STATUS, fake_status_handler)
        assert original is not updated
        assert Verb.STATUS not in original.registered_verbs
        assert Verb.STATUS in updated.registered_verbs

    def test_handler_replacement_returns_new_instance(self) -> None:
        first = CommandDispatcher().with_handler(
            Verb.STATUS, fake_status_handler
        )
        second = first.with_handler(Verb.STATUS, exploding_handler)
        assert first is not second
        # Both have STATUS registered but with different handlers
        assert Verb.STATUS in first.registered_verbs
        assert Verb.STATUS in second.registered_verbs

    def test_frozen_registry(self) -> None:
        """The internal registry should not be directly mutable."""
        dispatcher = CommandDispatcher().with_handler(
            Verb.STATUS, fake_status_handler
        )
        # registered_verbs returns a frozenset, which is inherently immutable
        verbs = dispatcher.registered_verbs
        assert isinstance(verbs, frozenset)
        # The underlying MappingProxyType rejects item assignment
        with pytest.raises(TypeError):
            dispatcher._registry[Verb.RUN] = fake_run_handler  # type: ignore[index]

    def test_has_handler(self) -> None:
        dispatcher = CommandDispatcher().with_handler(
            Verb.STATUS, fake_status_handler
        )
        assert dispatcher.has_handler(Verb.STATUS) is True
        assert dispatcher.has_handler(Verb.RUN) is False


# ---------------------------------------------------------------------------
# CommandDispatcher: dispatch routing
# ---------------------------------------------------------------------------


class TestDispatchRouting:
    """Tests for correct routing of ParsedCommand to handlers."""

    @pytest.fixture
    def full_dispatcher(self) -> CommandDispatcher:
        """Dispatcher with all six verbs registered."""
        return (
            CommandDispatcher()
            .with_handler(Verb.STATUS, fake_status_handler)
            .with_handler(Verb.WATCH, fake_watch_handler)
            .with_handler(Verb.RUN, fake_run_handler)
            .with_handler(Verb.QUEUE, fake_queue_handler)
            .with_handler(Verb.CANCEL, fake_cancel_handler)
            .with_handler(Verb.HISTORY, fake_history_handler)
        )

    @pytest.mark.asyncio
    async def test_dispatch_status(
        self, full_dispatcher: CommandDispatcher
    ) -> None:
        cmd = ParsedCommand(verb=Verb.STATUS, args=StatusArgs(verbose=True))
        result = await full_dispatcher.dispatch(cmd)
        assert result.success is True
        assert result.verb == Verb.STATUS
        assert result.payload == {"status": "idle", "verbose": True}
        assert result.error is None

    @pytest.mark.asyncio
    async def test_dispatch_watch(
        self, full_dispatcher: CommandDispatcher
    ) -> None:
        cmd = ParsedCommand(
            verb=Verb.WATCH,
            args=WatchArgs(run_id="abc-123", tail_lines=100),
        )
        result = await full_dispatcher.dispatch(cmd)
        assert result.success is True
        assert result.verb == Verb.WATCH
        assert result.payload == {"watching": True, "run_id": "abc-123"}

    @pytest.mark.asyncio
    async def test_dispatch_run(
        self, full_dispatcher: CommandDispatcher
    ) -> None:
        cmd = ParsedCommand(
            verb=Verb.RUN,
            args=RunArgs(
                target_host="staging.example.com",
                target_user="ci",
                natural_language="run the full regression suite",
            ),
        )
        result = await full_dispatcher.dispatch(cmd)
        assert result.success is True
        assert result.verb == Verb.RUN
        assert result.payload == {
            "started": True,
            "host": "staging.example.com",
        }

    @pytest.mark.asyncio
    async def test_dispatch_queue(
        self, full_dispatcher: CommandDispatcher
    ) -> None:
        cmd = ParsedCommand(
            verb=Verb.QUEUE,
            args=QueueArgs(
                target_host="prod.example.com",
                target_user="deploy",
                natural_language="run smoke tests",
                priority=5,
            ),
        )
        result = await full_dispatcher.dispatch(cmd)
        assert result.success is True
        assert result.verb == Verb.QUEUE
        assert result.payload == {"queued": True, "priority": 5}

    @pytest.mark.asyncio
    async def test_dispatch_cancel(
        self, full_dispatcher: CommandDispatcher
    ) -> None:
        cmd = ParsedCommand(
            verb=Verb.CANCEL,
            args=CancelArgs(force=True),
        )
        result = await full_dispatcher.dispatch(cmd)
        assert result.success is True
        assert result.verb == Verb.CANCEL
        assert result.payload == {"cancelled": True, "force": True}

    @pytest.mark.asyncio
    async def test_dispatch_history(
        self, full_dispatcher: CommandDispatcher
    ) -> None:
        cmd = ParsedCommand(
            verb=Verb.HISTORY,
            args=HistoryArgs(limit=50, verbose=True),
        )
        result = await full_dispatcher.dispatch(cmd)
        assert result.success is True
        assert result.verb == Verb.HISTORY
        assert result.payload == {"records": [], "limit": 50}


# ---------------------------------------------------------------------------
# CommandDispatcher: error handling
# ---------------------------------------------------------------------------


class TestDispatchErrors:
    """Tests for error handling during dispatch."""

    @pytest.mark.asyncio
    async def test_unregistered_verb_returns_error(self) -> None:
        dispatcher = CommandDispatcher().with_handler(
            Verb.STATUS, fake_status_handler
        )
        cmd = ParsedCommand(
            verb=Verb.RUN,
            args=RunArgs(
                target_host="host",
                target_user="user",
                natural_language="run tests",
            ),
        )
        result = await dispatcher.dispatch(cmd)
        assert result.success is False
        assert result.verb == Verb.RUN
        assert result.error is not None
        assert "run" in result.error.lower()
        assert result.payload is None

    @pytest.mark.asyncio
    async def test_handler_exception_captured_in_result(self) -> None:
        dispatcher = CommandDispatcher().with_handler(
            Verb.STATUS, exploding_handler
        )
        cmd = ParsedCommand(verb=Verb.STATUS, args=StatusArgs())
        result = await dispatcher.dispatch(cmd)
        assert result.success is False
        assert result.verb == Verb.STATUS
        assert result.error is not None
        assert "handler blew up" in result.error
        assert result.payload is None

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_propagate(self) -> None:
        """Dispatch must never raise -- it always returns a DispatchResponse."""
        dispatcher = CommandDispatcher().with_handler(
            Verb.STATUS, exploding_handler
        )
        cmd = ParsedCommand(verb=Verb.STATUS, args=StatusArgs())
        # This should NOT raise
        result = await dispatcher.dispatch(cmd)
        assert isinstance(result, DispatchResponse)


# ---------------------------------------------------------------------------
# create_dispatcher factory
# ---------------------------------------------------------------------------


class TestCreateDispatcher:
    """Tests for the create_dispatcher convenience factory."""

    def test_create_with_mapping(self) -> None:
        dispatcher = create_dispatcher({
            Verb.STATUS: fake_status_handler,
            Verb.RUN: fake_run_handler,
        })
        assert dispatcher.registered_verbs == frozenset({
            Verb.STATUS, Verb.RUN
        })

    def test_create_empty(self) -> None:
        dispatcher = create_dispatcher({})
        assert dispatcher.registered_verbs == frozenset()

    @pytest.mark.asyncio
    async def test_created_dispatcher_routes_correctly(self) -> None:
        dispatcher = create_dispatcher({
            Verb.CANCEL: fake_cancel_handler,
        })
        cmd = ParsedCommand(
            verb=Verb.CANCEL,
            args=CancelArgs(reason="test cleanup"),
        )
        result = await dispatcher.dispatch(cmd)
        assert result.success is True
        assert result.payload == {"cancelled": True, "force": False}

    def test_create_all_six_verbs(self) -> None:
        dispatcher = create_dispatcher({
            Verb.STATUS: fake_status_handler,
            Verb.WATCH: fake_watch_handler,
            Verb.RUN: fake_run_handler,
            Verb.QUEUE: fake_queue_handler,
            Verb.CANCEL: fake_cancel_handler,
            Verb.HISTORY: fake_history_handler,
        })
        assert dispatcher.registered_verbs == frozenset(Verb)


# ---------------------------------------------------------------------------
# Handler callable type validation
# ---------------------------------------------------------------------------


class TestHandlerValidation:
    """Tests for handler callable validation at registration."""

    def test_non_callable_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CommandDispatcher().with_handler(
                Verb.STATUS, "not a function"  # type: ignore[arg-type]
            )

    def test_none_handler_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CommandDispatcher().with_handler(
                Verb.STATUS, None  # type: ignore[arg-type]
            )

    def test_integer_handler_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CommandDispatcher().with_handler(
                Verb.STATUS, 42  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Dispatcher with_handlers (bulk registration)
# ---------------------------------------------------------------------------


class TestWithHandlers:
    """Tests for bulk handler registration via with_handlers."""

    def test_bulk_register(self) -> None:
        dispatcher = CommandDispatcher().with_handlers({
            Verb.STATUS: fake_status_handler,
            Verb.RUN: fake_run_handler,
            Verb.CANCEL: fake_cancel_handler,
        })
        assert dispatcher.registered_verbs == frozenset({
            Verb.STATUS, Verb.RUN, Verb.CANCEL
        })

    def test_bulk_register_returns_new_instance(self) -> None:
        original = CommandDispatcher()
        updated = original.with_handlers({
            Verb.STATUS: fake_status_handler,
        })
        assert original is not updated

    def test_bulk_register_empty_preserves_verbs(self) -> None:
        original = CommandDispatcher().with_handler(
            Verb.STATUS, fake_status_handler
        )
        updated = original.with_handlers({})
        # Returns a new instance (immutable builder pattern) with same verbs
        assert original is not updated
        assert updated.registered_verbs == frozenset({Verb.STATUS})

    @pytest.mark.asyncio
    async def test_bulk_registered_handlers_dispatch(self) -> None:
        dispatcher = CommandDispatcher().with_handlers({
            Verb.STATUS: fake_status_handler,
            Verb.HISTORY: fake_history_handler,
        })
        cmd = ParsedCommand(
            verb=Verb.HISTORY,
            args=HistoryArgs(limit=10),
        )
        result = await dispatcher.dispatch(cmd)
        assert result.success is True
        assert result.payload == {"records": [], "limit": 10}

    def test_bulk_register_validates_callables(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CommandDispatcher().with_handlers({
                Verb.STATUS: fake_status_handler,
                Verb.RUN: "not_callable",  # type: ignore[dict-item]
            })
