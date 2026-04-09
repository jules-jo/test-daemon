"""CLI command dispatcher: route ParsedCommands to handler callables.

Maintains an immutable registry mapping each ``Verb`` to an async handler
callable. When the daemon receives a ``ParsedCommand`` from the IPC layer,
the dispatcher looks up the handler for the command's verb and invokes it
with the parsed arguments.

Design principles:
- Immutable: the registry is a ``MappingProxyType``. Registration produces
  a new dispatcher instance (builder pattern via ``with_handler``).
- Safe: dispatch never raises. Handler exceptions are captured in the
  ``DispatchResponse.error`` field.
- Typed: handlers receive the verb-specific ``*Args`` dataclass.

Usage::

    from jules_daemon.cli.dispatcher import CommandDispatcher, create_dispatcher
    from jules_daemon.cli.verbs import Verb

    async def handle_status(args):
        return {"status": "idle"}

    dispatcher = create_dispatcher({Verb.STATUS: handle_status})
    result = await dispatcher.dispatch(parsed_command)
    if result.success:
        send_to_cli(result.payload)
    else:
        send_error(result.error)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from jules_daemon.cli.registry import HandlerCallable
from jules_daemon.cli.verbs import ParsedCommand, Verb

__all__ = [
    "CommandDispatcher",
    "DispatchResponse",
    "HandlerCallable",
    "create_dispatcher",
]

logger = logging.getLogger(__name__)

# Empty registry constant, shared by all empty dispatchers.
_EMPTY_REGISTRY: MappingProxyType[Verb, HandlerCallable] = MappingProxyType({})


# ---------------------------------------------------------------------------
# DispatchResponse
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchResponse:
    """Immutable result of dispatching a command to its handler.

    Attributes:
        success: True if the handler completed without error.
        verb: The verb that was dispatched.
        payload: Handler return value on success, None on failure.
        error: Human-readable error description on failure, None on success.
    """

    success: bool
    verb: Verb
    payload: Any
    error: str | None


# ---------------------------------------------------------------------------
# CommandDispatcher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandDispatcher:
    """Immutable command dispatcher with a verb -> handler registry.

    The registry maps each ``Verb`` to an async handler callable. Use the
    builder methods (``with_handler``, ``with_handlers``) to produce new
    dispatcher instances with additional or replaced handlers.

    The dispatcher never raises during ``dispatch()`` -- all errors are
    captured in the ``DispatchResponse``.
    """

    _registry: MappingProxyType[Verb, HandlerCallable] = field(
        default=_EMPTY_REGISTRY,
    )

    # -- Query methods --

    @property
    def registered_verbs(self) -> frozenset[Verb]:
        """Return the set of verbs that have registered handlers."""
        return frozenset(self._registry.keys())

    def has_handler(self, verb: Verb) -> bool:
        """Check whether a handler is registered for the given verb."""
        return verb in self._registry

    # -- Builder methods (return new instances) --

    def with_handler(
        self,
        verb: Verb,
        handler: HandlerCallable,
    ) -> CommandDispatcher:
        """Return a new dispatcher with the given handler registered.

        If a handler is already registered for the verb, it is replaced.
        The original dispatcher is not modified.

        Args:
            verb: The verb to register the handler for.
            handler: Async callable that processes the verb's arguments.

        Returns:
            New CommandDispatcher with the handler added.

        Raises:
            TypeError: If handler is not callable.
        """
        if not callable(handler):
            raise TypeError(
                f"Handler for verb {verb.value!r} must be callable, "
                f"got {type(handler).__name__}"
            )

        new_mapping = dict(self._registry)
        new_mapping[verb] = handler
        return CommandDispatcher(
            _registry=MappingProxyType(new_mapping),
        )

    def with_handlers(
        self,
        handlers: dict[Verb, HandlerCallable],
    ) -> CommandDispatcher:
        """Return a new dispatcher with multiple handlers registered.

        Validates all handlers before applying any changes. If any handler
        is not callable, raises TypeError without modifying the registry.

        Args:
            handlers: Mapping of verb -> handler callable.

        Returns:
            New CommandDispatcher with all handlers added.

        Raises:
            TypeError: If any handler is not callable.
        """
        # Validate all handlers first (fail fast, no partial updates)
        for verb, handler in handlers.items():
            if not callable(handler):
                raise TypeError(
                    f"Handler for verb {verb.value!r} must be callable, "
                    f"got {type(handler).__name__}"
                )

        new_mapping = dict(self._registry)
        new_mapping.update(handlers)
        return CommandDispatcher(
            _registry=MappingProxyType(new_mapping),
        )

    # -- Dispatch --

    async def dispatch(self, command: ParsedCommand) -> DispatchResponse:
        """Route a parsed command to its registered handler.

        Looks up the handler for the command's verb and invokes it with
        the parsed arguments. Returns a ``DispatchResponse`` with the
        handler's return value or an error description.

        This method never raises. All errors (missing handler, handler
        exception) are captured in the response.

        Args:
            command: The parsed command to dispatch.

        Returns:
            DispatchResponse with success status and handler result.
        """
        verb = command.verb
        handler = self._registry.get(verb)

        if handler is None:
            logger.warning(
                "No handler registered for verb %r", verb.value
            )
            return DispatchResponse(
                success=False,
                verb=verb,
                payload=None,
                error=(
                    f"No handler registered for verb {verb.value!r}. "
                    f"Registered: {', '.join(v.value for v in sorted(self._registry, key=lambda v: v.value))}"
                ),
            )

        try:
            payload = await handler(command.args)
        except Exception as exc:
            logger.error(
                "Handler for verb %r raised %s: %s",
                verb.value,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            # Return sanitized error to caller -- full details stay in logs
            return DispatchResponse(
                success=False,
                verb=verb,
                payload=None,
                error=f"Handler error: {exc}",
            )

        return DispatchResponse(
            success=True,
            verb=verb,
            payload=payload,
            error=None,
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_dispatcher(
    handlers: dict[Verb, HandlerCallable],
) -> CommandDispatcher:
    """Create a CommandDispatcher with pre-registered handlers.

    Convenience factory that validates all handlers and constructs
    the dispatcher in one step.

    Args:
        handlers: Mapping of verb -> async handler callable.

    Returns:
        CommandDispatcher with the given handlers registered.

    Raises:
        TypeError: If any handler is not callable.
    """
    if not handlers:
        return CommandDispatcher()

    return CommandDispatcher().with_handlers(handlers)
