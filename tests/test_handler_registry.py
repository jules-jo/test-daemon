"""Tests for the command handler registry.

Verifies that the registry:
- Maps canonical verb names to handler entries (callable + metadata)
- Supports registration via immutable builder pattern (no mutation)
- Supports lookup by Verb enum (structured CLI path)
- Supports lookup by string name (NL/LLM path)
- Provides introspection for LLM prompt construction (verb_descriptions)
- Validates handler callables and metadata at registration time
- Returns None for unregistered verbs (no crash)
- Produces new instances on every mutation (frozen)
- Supports bulk registration
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
from jules_daemon.cli.registry import (
    CommandHandlerRegistry,
    HandlerEntry,
    create_registry,
)


# ---------------------------------------------------------------------------
# Fake handlers for testing
# ---------------------------------------------------------------------------


async def fake_status_handler(args: VerbArgs) -> dict[str, Any]:
    """Return a mock status response."""
    return {"status": "idle"}


async def fake_watch_handler(args: VerbArgs) -> dict[str, Any]:
    """Return a mock watch response."""
    return {"watching": True}


async def fake_run_handler(args: VerbArgs) -> dict[str, Any]:
    """Return a mock run response."""
    return {"started": True}


async def fake_queue_handler(args: VerbArgs) -> dict[str, Any]:
    """Return a mock queue response."""
    return {"queued": True}


async def fake_cancel_handler(args: VerbArgs) -> dict[str, Any]:
    """Return a mock cancel response."""
    return {"cancelled": True}


async def fake_history_handler(args: VerbArgs) -> dict[str, Any]:
    """Return a mock history response."""
    return {"records": []}


async def replacement_status_handler(args: VerbArgs) -> dict[str, Any]:
    """Replacement handler for testing handler override."""
    return {"status": "replaced"}


# ---------------------------------------------------------------------------
# HandlerEntry: data model
# ---------------------------------------------------------------------------


class TestHandlerEntry:
    """Tests for the HandlerEntry frozen dataclass."""

    def test_construction(self) -> None:
        entry = HandlerEntry(
            verb=Verb.STATUS,
            handler=fake_status_handler,
            description="Query the current run state",
            parameter_schema=StatusArgs,
        )
        assert entry.verb == Verb.STATUS
        assert entry.handler is fake_status_handler
        assert entry.description == "Query the current run state"
        assert entry.parameter_schema is StatusArgs

    def test_frozen(self) -> None:
        entry = HandlerEntry(
            verb=Verb.STATUS,
            handler=fake_status_handler,
            description="Query the current run state",
            parameter_schema=StatusArgs,
        )
        with pytest.raises(FrozenInstanceError):
            entry.description = "mutated"  # type: ignore[misc]

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="description"):
            HandlerEntry(
                verb=Verb.STATUS,
                handler=fake_status_handler,
                description="",
                parameter_schema=StatusArgs,
            )

    def test_whitespace_description_raises(self) -> None:
        with pytest.raises(ValueError, match="description"):
            HandlerEntry(
                verb=Verb.STATUS,
                handler=fake_status_handler,
                description="   ",
                parameter_schema=StatusArgs,
            )

    def test_non_callable_handler_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            HandlerEntry(
                verb=Verb.STATUS,
                handler="not a function",  # type: ignore[arg-type]
                description="desc",
                parameter_schema=StatusArgs,
            )


# ---------------------------------------------------------------------------
# CommandHandlerRegistry: construction
# ---------------------------------------------------------------------------


class TestRegistryConstruction:
    """Tests for registry construction and immutability."""

    def test_empty_registry(self) -> None:
        registry = CommandHandlerRegistry()
        assert registry.registered_verbs == frozenset()
        assert len(registry) == 0

    def test_register_single_handler(self) -> None:
        registry = CommandHandlerRegistry().register(
            verb=Verb.STATUS,
            handler=fake_status_handler,
            description="Query the current run state",
            parameter_schema=StatusArgs,
        )
        assert Verb.STATUS in registry.registered_verbs
        assert len(registry) == 1

    def test_register_returns_new_instance(self) -> None:
        original = CommandHandlerRegistry()
        updated = original.register(
            verb=Verb.STATUS,
            handler=fake_status_handler,
            description="Query status",
            parameter_schema=StatusArgs,
        )
        assert original is not updated
        assert len(original) == 0
        assert len(updated) == 1

    def test_register_multiple_verbs(self) -> None:
        registry = (
            CommandHandlerRegistry()
            .register(
                verb=Verb.STATUS,
                handler=fake_status_handler,
                description="Query status",
                parameter_schema=StatusArgs,
            )
            .register(
                verb=Verb.RUN,
                handler=fake_run_handler,
                description="Start test execution",
                parameter_schema=RunArgs,
            )
            .register(
                verb=Verb.CANCEL,
                handler=fake_cancel_handler,
                description="Cancel a run",
                parameter_schema=CancelArgs,
            )
        )
        assert registry.registered_verbs == frozenset({
            Verb.STATUS, Verb.RUN, Verb.CANCEL,
        })
        assert len(registry) == 3

    def test_frozen_registry(self) -> None:
        """The internal mapping should not be directly mutable."""
        registry = CommandHandlerRegistry().register(
            verb=Verb.STATUS,
            handler=fake_status_handler,
            description="Query status",
            parameter_schema=StatusArgs,
        )
        with pytest.raises(TypeError):
            registry._entries[Verb.RUN] = None  # type: ignore[index]

    def test_handler_replacement_preserves_immutability(self) -> None:
        first = CommandHandlerRegistry().register(
            verb=Verb.STATUS,
            handler=fake_status_handler,
            description="Original",
            parameter_schema=StatusArgs,
        )
        second = first.register(
            verb=Verb.STATUS,
            handler=replacement_status_handler,
            description="Replaced",
            parameter_schema=StatusArgs,
        )
        assert first is not second
        # Original is unchanged
        assert first.lookup(Verb.STATUS) is not None
        assert first.lookup(Verb.STATUS).handler is fake_status_handler  # type: ignore[union-attr]
        # Updated has replacement
        assert second.lookup(Verb.STATUS) is not None
        assert second.lookup(Verb.STATUS).handler is replacement_status_handler  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# CommandHandlerRegistry: lookup by Verb enum (structured path)
# ---------------------------------------------------------------------------


class TestLookupByVerb:
    """Tests for lookup by Verb enum -- the structured CLI path."""

    def test_lookup_registered_verb(self) -> None:
        registry = CommandHandlerRegistry().register(
            verb=Verb.STATUS,
            handler=fake_status_handler,
            description="Query status",
            parameter_schema=StatusArgs,
        )
        entry = registry.lookup(Verb.STATUS)
        assert entry is not None
        assert entry.verb == Verb.STATUS
        assert entry.handler is fake_status_handler
        assert entry.description == "Query status"
        assert entry.parameter_schema is StatusArgs

    def test_lookup_unregistered_verb_returns_none(self) -> None:
        registry = CommandHandlerRegistry()
        assert registry.lookup(Verb.RUN) is None

    def test_has_handler_true(self) -> None:
        registry = CommandHandlerRegistry().register(
            verb=Verb.WATCH,
            handler=fake_watch_handler,
            description="Watch output",
            parameter_schema=WatchArgs,
        )
        assert registry.has_handler(Verb.WATCH) is True

    def test_has_handler_false(self) -> None:
        registry = CommandHandlerRegistry()
        assert registry.has_handler(Verb.CANCEL) is False


# ---------------------------------------------------------------------------
# CommandHandlerRegistry: lookup by string name (NL path)
# ---------------------------------------------------------------------------


class TestLookupByName:
    """Tests for lookup by string name -- the NL/LLM path."""

    @pytest.fixture
    def populated_registry(self) -> CommandHandlerRegistry:
        """Registry with status and run registered."""
        return (
            CommandHandlerRegistry()
            .register(
                verb=Verb.STATUS,
                handler=fake_status_handler,
                description="Query status",
                parameter_schema=StatusArgs,
            )
            .register(
                verb=Verb.RUN,
                handler=fake_run_handler,
                description="Start tests",
                parameter_schema=RunArgs,
            )
        )

    def test_lookup_by_name_exact(
        self, populated_registry: CommandHandlerRegistry
    ) -> None:
        entry = populated_registry.lookup_by_name("status")
        assert entry is not None
        assert entry.verb == Verb.STATUS

    def test_lookup_by_name_case_insensitive(
        self, populated_registry: CommandHandlerRegistry
    ) -> None:
        entry = populated_registry.lookup_by_name("STATUS")
        assert entry is not None
        assert entry.verb == Verb.STATUS

    def test_lookup_by_name_mixed_case(
        self, populated_registry: CommandHandlerRegistry
    ) -> None:
        entry = populated_registry.lookup_by_name("Run")
        assert entry is not None
        assert entry.verb == Verb.RUN

    def test_lookup_by_name_with_whitespace(
        self, populated_registry: CommandHandlerRegistry
    ) -> None:
        entry = populated_registry.lookup_by_name("  status  ")
        assert entry is not None
        assert entry.verb == Verb.STATUS

    def test_lookup_by_name_unregistered_returns_none(
        self, populated_registry: CommandHandlerRegistry
    ) -> None:
        assert populated_registry.lookup_by_name("cancel") is None

    def test_lookup_by_name_invalid_verb_returns_none(
        self, populated_registry: CommandHandlerRegistry
    ) -> None:
        assert populated_registry.lookup_by_name("deploy") is None

    def test_lookup_by_name_empty_returns_none(
        self, populated_registry: CommandHandlerRegistry
    ) -> None:
        assert populated_registry.lookup_by_name("") is None

    def test_lookup_by_name_whitespace_only_returns_none(
        self, populated_registry: CommandHandlerRegistry
    ) -> None:
        assert populated_registry.lookup_by_name("   ") is None


# ---------------------------------------------------------------------------
# CommandHandlerRegistry: introspection for LLM prompt construction
# ---------------------------------------------------------------------------


class TestRegistryIntrospection:
    """Tests for introspection methods used by the NL/LLM path."""

    @pytest.fixture
    def full_registry(self) -> CommandHandlerRegistry:
        """Registry with all six verbs registered."""
        return (
            CommandHandlerRegistry()
            .register(
                verb=Verb.STATUS,
                handler=fake_status_handler,
                description="Query the current run state",
                parameter_schema=StatusArgs,
            )
            .register(
                verb=Verb.WATCH,
                handler=fake_watch_handler,
                description="Live-stream test output",
                parameter_schema=WatchArgs,
            )
            .register(
                verb=Verb.RUN,
                handler=fake_run_handler,
                description="Start test execution via SSH",
                parameter_schema=RunArgs,
            )
            .register(
                verb=Verb.QUEUE,
                handler=fake_queue_handler,
                description="Queue a command for later execution",
                parameter_schema=QueueArgs,
            )
            .register(
                verb=Verb.CANCEL,
                handler=fake_cancel_handler,
                description="Cancel the current or queued run",
                parameter_schema=CancelArgs,
            )
            .register(
                verb=Verb.HISTORY,
                handler=fake_history_handler,
                description="View past test run results",
                parameter_schema=HistoryArgs,
            )
        )

    def test_all_entries(
        self, full_registry: CommandHandlerRegistry
    ) -> None:
        entries = full_registry.all_entries()
        assert isinstance(entries, tuple)
        assert len(entries) == 6
        verbs_found = {e.verb for e in entries}
        assert verbs_found == frozenset(Verb)

    def test_all_entries_immutable(
        self, full_registry: CommandHandlerRegistry
    ) -> None:
        entries = full_registry.all_entries()
        # tuple is immutable
        assert isinstance(entries, tuple)

    def test_all_entries_empty_registry(self) -> None:
        registry = CommandHandlerRegistry()
        entries = registry.all_entries()
        assert entries == ()

    def test_verb_descriptions(
        self, full_registry: CommandHandlerRegistry
    ) -> None:
        descriptions = full_registry.verb_descriptions()
        assert isinstance(descriptions, dict)
        assert len(descriptions) == 6
        assert descriptions["status"] == "Query the current run state"
        assert descriptions["run"] == "Start test execution via SSH"
        assert descriptions["cancel"] == "Cancel the current or queued run"

    def test_verb_descriptions_keys_are_string_values(
        self, full_registry: CommandHandlerRegistry
    ) -> None:
        """Keys should be verb string values (e.g., 'status'), not enum names."""
        descriptions = full_registry.verb_descriptions()
        for key in descriptions:
            assert isinstance(key, str)
            # Verify they match Verb enum values (lowercase)
            assert key == key.lower()

    def test_verb_descriptions_empty_registry(self) -> None:
        registry = CommandHandlerRegistry()
        assert registry.verb_descriptions() == {}

    def test_registered_verbs(
        self, full_registry: CommandHandlerRegistry
    ) -> None:
        verbs = full_registry.registered_verbs
        assert isinstance(verbs, frozenset)
        assert verbs == frozenset(Verb)

    def test_len(self, full_registry: CommandHandlerRegistry) -> None:
        assert len(full_registry) == 6


# ---------------------------------------------------------------------------
# CommandHandlerRegistry: validation
# ---------------------------------------------------------------------------


class TestRegistryValidation:
    """Tests for validation at registration time."""

    def test_non_callable_handler_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CommandHandlerRegistry().register(
                verb=Verb.STATUS,
                handler="not a function",  # type: ignore[arg-type]
                description="desc",
                parameter_schema=StatusArgs,
            )

    def test_none_handler_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CommandHandlerRegistry().register(
                verb=Verb.STATUS,
                handler=None,  # type: ignore[arg-type]
                description="desc",
                parameter_schema=StatusArgs,
            )

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="description"):
            CommandHandlerRegistry().register(
                verb=Verb.STATUS,
                handler=fake_status_handler,
                description="",
                parameter_schema=StatusArgs,
            )

    def test_whitespace_description_raises(self) -> None:
        with pytest.raises(ValueError, match="description"):
            CommandHandlerRegistry().register(
                verb=Verb.STATUS,
                handler=fake_status_handler,
                description="   \t  ",
                parameter_schema=StatusArgs,
            )


# ---------------------------------------------------------------------------
# Bulk registration
# ---------------------------------------------------------------------------


class TestBulkRegistration:
    """Tests for registering multiple handlers at once."""

    def test_register_many(self) -> None:
        registry = CommandHandlerRegistry().register_many([
            {
                "verb": Verb.STATUS,
                "handler": fake_status_handler,
                "description": "Query status",
                "parameter_schema": StatusArgs,
            },
            {
                "verb": Verb.RUN,
                "handler": fake_run_handler,
                "description": "Start tests",
                "parameter_schema": RunArgs,
            },
        ])
        assert registry.registered_verbs == frozenset({Verb.STATUS, Verb.RUN})

    def test_register_many_returns_new_instance(self) -> None:
        original = CommandHandlerRegistry()
        updated = original.register_many([
            {
                "verb": Verb.STATUS,
                "handler": fake_status_handler,
                "description": "Query status",
                "parameter_schema": StatusArgs,
            },
        ])
        assert original is not updated
        assert len(original) == 0
        assert len(updated) == 1

    def test_register_many_empty_list(self) -> None:
        original = CommandHandlerRegistry().register(
            verb=Verb.STATUS,
            handler=fake_status_handler,
            description="Query status",
            parameter_schema=StatusArgs,
        )
        updated = original.register_many([])
        # Returns a new instance with same entries (immutable builder)
        assert len(updated) == 1

    def test_register_many_validates_all_before_applying(self) -> None:
        """If any entry is invalid, none should be applied."""
        with pytest.raises(TypeError, match="callable"):
            CommandHandlerRegistry().register_many([
                {
                    "verb": Verb.STATUS,
                    "handler": fake_status_handler,
                    "description": "Valid",
                    "parameter_schema": StatusArgs,
                },
                {
                    "verb": Verb.RUN,
                    "handler": "not callable",  # invalid
                    "description": "Invalid",
                    "parameter_schema": RunArgs,
                },
            ])


# ---------------------------------------------------------------------------
# create_registry factory function
# ---------------------------------------------------------------------------


class TestCreateRegistry:
    """Tests for the create_registry convenience factory."""

    def test_create_with_entries(self) -> None:
        registry = create_registry([
            {
                "verb": Verb.STATUS,
                "handler": fake_status_handler,
                "description": "Query status",
                "parameter_schema": StatusArgs,
            },
            {
                "verb": Verb.CANCEL,
                "handler": fake_cancel_handler,
                "description": "Cancel run",
                "parameter_schema": CancelArgs,
            },
        ])
        assert registry.registered_verbs == frozenset({
            Verb.STATUS, Verb.CANCEL,
        })

    def test_create_empty(self) -> None:
        registry = create_registry([])
        assert registry.registered_verbs == frozenset()
        assert len(registry) == 0

    def test_create_all_six(self) -> None:
        registry = create_registry([
            {
                "verb": Verb.STATUS,
                "handler": fake_status_handler,
                "description": "Query status",
                "parameter_schema": StatusArgs,
            },
            {
                "verb": Verb.WATCH,
                "handler": fake_watch_handler,
                "description": "Watch output",
                "parameter_schema": WatchArgs,
            },
            {
                "verb": Verb.RUN,
                "handler": fake_run_handler,
                "description": "Start tests",
                "parameter_schema": RunArgs,
            },
            {
                "verb": Verb.QUEUE,
                "handler": fake_queue_handler,
                "description": "Queue command",
                "parameter_schema": QueueArgs,
            },
            {
                "verb": Verb.CANCEL,
                "handler": fake_cancel_handler,
                "description": "Cancel run",
                "parameter_schema": CancelArgs,
            },
            {
                "verb": Verb.HISTORY,
                "handler": fake_history_handler,
                "description": "View history",
                "parameter_schema": HistoryArgs,
            },
        ])
        assert registry.registered_verbs == frozenset(Verb)
        assert len(registry) == 6


# ---------------------------------------------------------------------------
# Integration: structured path lookup
# ---------------------------------------------------------------------------


class TestStructuredPathLookup:
    """Verify the registry can resolve handlers for ParsedCommand verbs."""

    def test_lookup_from_parsed_command_verb(self) -> None:
        registry = CommandHandlerRegistry().register(
            verb=Verb.RUN,
            handler=fake_run_handler,
            description="Start tests",
            parameter_schema=RunArgs,
        )
        cmd = ParsedCommand(
            verb=Verb.RUN,
            args=RunArgs(
                target_host="host",
                target_user="user",
                natural_language="run tests",
            ),
        )
        entry = registry.lookup(cmd.verb)
        assert entry is not None
        assert entry.handler is fake_run_handler


# ---------------------------------------------------------------------------
# Integration: NL path lookup
# ---------------------------------------------------------------------------


class TestNLPathLookup:
    """Verify the registry can resolve handlers from NL-classified verb strings."""

    def test_lookup_from_classified_verb_string(self) -> None:
        """Simulate the NL path: intent classifier returns a verb string."""
        registry = CommandHandlerRegistry().register(
            verb=Verb.CANCEL,
            handler=fake_cancel_handler,
            description="Cancel run",
            parameter_schema=CancelArgs,
        )
        # The NL classifier returns a verb string like "cancel"
        classified_verb_str = "cancel"
        entry = registry.lookup_by_name(classified_verb_str)
        assert entry is not None
        assert entry.verb == Verb.CANCEL
        assert entry.handler is fake_cancel_handler

    def test_verb_descriptions_usable_for_prompt(self) -> None:
        """Verify verb_descriptions can be used to build LLM prompts."""
        registry = (
            CommandHandlerRegistry()
            .register(
                verb=Verb.STATUS,
                handler=fake_status_handler,
                description="Query the current run state",
                parameter_schema=StatusArgs,
            )
            .register(
                verb=Verb.RUN,
                handler=fake_run_handler,
                description="Start test execution via SSH",
                parameter_schema=RunArgs,
            )
        )
        descriptions = registry.verb_descriptions()
        # Build a simple prompt fragment from descriptions
        prompt_lines = [
            f"- {verb}: {desc}" for verb, desc in descriptions.items()
        ]
        prompt = "\n".join(prompt_lines)
        assert "status: Query the current run state" in prompt
        assert "run: Start test execution via SSH" in prompt
