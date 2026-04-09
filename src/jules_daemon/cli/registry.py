"""Command handler registry: maps canonical verb names to handler entries.

The single source of truth for verb -> handler resolution, targeted by
both the structured CLI path (parser -> ParsedCommand -> lookup) and
the natural-language path (LLM intent classifier -> verb string -> lookup).

Design principles:
- Immutable: the registry is a ``MappingProxyType``. Registration produces
  a new registry instance (builder pattern via ``register``).
- Introspectable: ``verb_descriptions()`` and ``all_entries()`` provide
  metadata for LLM prompt construction and debugging.
- Dual lookup: ``lookup(Verb)`` for the structured path,
  ``lookup_by_name(str)`` for the NL path.
- Validated: handler callables and descriptions are validated at
  registration time (fail fast, no partial updates on bulk registration).

Usage::

    from jules_daemon.cli.registry import CommandHandlerRegistry, create_registry
    from jules_daemon.cli.verbs import Verb, StatusArgs

    async def handle_status(args):
        return {"status": "idle"}

    # Builder pattern
    registry = (
        CommandHandlerRegistry()
        .register(
            verb=Verb.STATUS,
            handler=handle_status,
            description="Query the current run state",
            parameter_schema=StatusArgs,
        )
    )

    # Structured path: lookup by Verb enum
    entry = registry.lookup(Verb.STATUS)

    # NL path: lookup by string name
    entry = registry.lookup_by_name("status")

    # LLM prompt construction
    descriptions = registry.verb_descriptions()
    # {"status": "Query the current run state", ...}
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from jules_daemon.cli.verbs import Verb, VerbArgs

__all__ = [
    "CommandHandlerRegistry",
    "HandlerCallable",
    "HandlerEntry",
    "create_registry",
]

logger = logging.getLogger(__name__)

# Type alias for handler callables.
# Each handler is an async function that receives the verb-specific args
# and returns an arbitrary payload dict (or any serializable value).
HandlerCallable = Callable[[VerbArgs], Awaitable[Any]]

# Empty registry constant, shared by all empty registries.
_EMPTY_ENTRIES: MappingProxyType[Verb, "HandlerEntry"] = MappingProxyType({})

# Lookup table for case-insensitive verb name resolution.
_VERB_NAME_LOOKUP: dict[str, Verb] = {v.value: v for v in Verb}


# ---------------------------------------------------------------------------
# HandlerEntry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandlerEntry:
    """Metadata and callable for a registered verb handler.

    Binds a ``Verb`` to its async handler callable, a human-readable
    description (used for LLM prompt construction), and the expected
    parameter schema type (the ``*Args`` dataclass).

    Immutable: all fields are set at construction time and validated
    in ``__post_init__``.

    Attributes:
        verb: The canonical CLI verb this entry handles.
        handler: Async callable that processes the verb's arguments.
        description: Human-readable description of what the handler does.
            Used by the NL intent classifier to build LLM prompts.
        parameter_schema: The ``*Args`` dataclass type that defines
            the expected argument schema for this verb.
    """

    verb: Verb
    handler: HandlerCallable
    description: str
    parameter_schema: type

    def __post_init__(self) -> None:
        if not callable(self.handler):
            raise TypeError(
                f"Handler for verb {self.verb.value!r} must be callable, "
                f"got {type(self.handler).__name__}"
            )
        if not self.description or not self.description.strip():
            raise ValueError(
                f"description for verb {self.verb.value!r} must not be "
                f"empty or whitespace-only"
            )


# ---------------------------------------------------------------------------
# HandlerEntrySpec (for bulk registration)
# ---------------------------------------------------------------------------

# Type alias for the dict format accepted by register_many / create_registry.
# Keys: verb, handler, description, parameter_schema
HandlerEntrySpec = dict[str, Any]


# ---------------------------------------------------------------------------
# CommandHandlerRegistry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandHandlerRegistry:
    """Immutable command handler registry with verb -> handler entry mapping.

    The single source of truth for verb -> handler resolution. Both the
    structured CLI path (``ParsedCommand.verb``) and the NL/LLM path
    (intent classifier verb string) target this registry.

    The registry is immutable. Use the builder methods (``register``,
    ``register_many``) to produce new registry instances with additional
    or replaced entries.
    """

    _entries: MappingProxyType[Verb, HandlerEntry] = field(
        default=_EMPTY_ENTRIES,
    )

    # -- Query methods --

    @property
    def registered_verbs(self) -> frozenset[Verb]:
        """Return the set of verbs that have registered handlers."""
        return frozenset(self._entries.keys())

    def __len__(self) -> int:
        """Return the number of registered handlers."""
        return len(self._entries)

    def has_handler(self, verb: Verb) -> bool:
        """Check whether a handler is registered for the given verb.

        Args:
            verb: The Verb enum member to check.

        Returns:
            True if a handler is registered for the verb.
        """
        return verb in self._entries

    # -- Lookup methods --

    def lookup(self, verb: Verb) -> HandlerEntry | None:
        """Look up a handler entry by Verb enum (structured CLI path).

        Args:
            verb: The Verb enum member to look up.

        Returns:
            The HandlerEntry if registered, None otherwise.
        """
        return self._entries.get(verb)

    def lookup_by_name(self, name: str) -> HandlerEntry | None:
        """Look up a handler entry by verb string name (NL/LLM path).

        Case-insensitive with leading/trailing whitespace stripped.
        Returns None for empty strings, unknown verb names, or verbs
        without a registered handler.

        Args:
            name: Verb name string (e.g., "status", "RUN", "  cancel  ").

        Returns:
            The HandlerEntry if registered, None otherwise.
        """
        normalized = name.strip().lower()
        if not normalized:
            return None

        verb = _VERB_NAME_LOOKUP.get(normalized)
        if verb is None:
            return None

        return self._entries.get(verb)

    # -- Introspection methods (for LLM prompt construction) --

    def all_entries(self) -> tuple[HandlerEntry, ...]:
        """Return all registered handler entries as an immutable tuple.

        Entries are sorted by verb value for deterministic ordering.

        Returns:
            Tuple of HandlerEntry instances, sorted by verb name.
        """
        return tuple(
            entry
            for _, entry in sorted(
                self._entries.items(),
                key=lambda item: item[0].value,
            )
        )

    def verb_descriptions(self) -> dict[str, str]:
        """Return a mapping of verb name -> description for all registered verbs.

        Keys are the lowercase string values of the Verb enum (e.g.,
        "status", "run"). Useful for building LLM system prompts that
        describe available actions.

        Returns:
            Dict mapping verb string value to description.
        """
        return {
            verb.value: entry.description
            for verb, entry in sorted(
                self._entries.items(),
                key=lambda item: item[0].value,
            )
        }

    # -- Builder methods (return new instances) --

    def register(
        self,
        *,
        verb: Verb,
        handler: HandlerCallable,
        description: str,
        parameter_schema: type,
    ) -> CommandHandlerRegistry:
        """Return a new registry with the given handler registered.

        If a handler is already registered for the verb, it is replaced.
        The original registry is not modified.

        Args:
            verb: The verb to register the handler for.
            handler: Async callable that processes the verb's arguments.
            description: Human-readable description of the handler.
            parameter_schema: The ``*Args`` dataclass type for the verb.

        Returns:
            New CommandHandlerRegistry with the handler added.

        Raises:
            TypeError: If handler is not callable.
            ValueError: If description is empty or whitespace-only.
        """
        # Validation happens in HandlerEntry.__post_init__
        entry = HandlerEntry(
            verb=verb,
            handler=handler,
            description=description,
            parameter_schema=parameter_schema,
        )

        new_mapping = dict(self._entries)
        new_mapping[verb] = entry
        return CommandHandlerRegistry(
            _entries=MappingProxyType(new_mapping),
        )

    def register_many(
        self,
        specs: list[HandlerEntrySpec],
    ) -> CommandHandlerRegistry:
        """Return a new registry with multiple handlers registered.

        Validates all entries before applying any changes. If any entry
        is invalid, raises without modifying the registry (fail fast,
        no partial updates).

        Each spec dict must have keys: verb, handler, description,
        parameter_schema.

        Args:
            specs: List of handler entry specification dicts.

        Returns:
            New CommandHandlerRegistry with all handlers added.

        Raises:
            TypeError: If any handler is not callable.
            ValueError: If any description is empty.
        """
        # Validate all entries first (fail fast)
        validated_entries: list[HandlerEntry] = []
        for spec in specs:
            entry = HandlerEntry(
                verb=spec["verb"],
                handler=spec["handler"],
                description=spec["description"],
                parameter_schema=spec["parameter_schema"],
            )
            validated_entries.append(entry)

        # Apply all validated entries
        new_mapping = dict(self._entries)
        for entry in validated_entries:
            new_mapping[entry.verb] = entry

        return CommandHandlerRegistry(
            _entries=MappingProxyType(new_mapping),
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_registry(
    specs: list[HandlerEntrySpec],
) -> CommandHandlerRegistry:
    """Create a CommandHandlerRegistry with pre-registered handlers.

    Convenience factory that validates all entries and constructs
    the registry in one step.

    Each spec dict must have keys: verb, handler, description,
    parameter_schema.

    Args:
        specs: List of handler entry specification dicts.

    Returns:
        CommandHandlerRegistry with the given handlers registered.

    Raises:
        TypeError: If any handler is not callable.
        ValueError: If any description is empty.
    """
    if not specs:
        return CommandHandlerRegistry()

    return CommandHandlerRegistry().register_many(specs)
