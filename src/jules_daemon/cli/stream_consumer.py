"""Event stream consumer/dispatcher for daemon notification events.

Subscribes to the daemon's notification stream via ``SubscriptionClient``,
routes each incoming ``NotificationEnvelope`` to the appropriate format
function via a ``RendererRegistry``, and writes the assembled output to
the terminal.

Architecture::

    SubscriptionClient (async iterator of NotificationEnvelope)
        |
        v
    EventStreamConsumer.consume_iter() / consume_client()
        |
        +--> RendererRegistry.get(event_type)
        |        |
        |        v
        |    EventRenderer.render(payload, context)
        |        |
        |        v
        |    RenderedOutput (text + severity)
        |        |
        |        +--> severity filter (optional)
        |        +--> _write_output() --> TextIO (terminal)
        |
        v
    ConsumerResult (events processed / displayed / dropped)

Key components:

    **RendererRegistry**: Immutable registry mapping event type strings
    to ``EventRenderer`` instances. Follows the project-wide builder
    pattern -- ``register()`` returns a new registry with the renderer
    added. Provides ``get()``, ``registered_types``, and ``__len__``.

    **EventStreamConsumer**: The main async consumer. Reads events from
    an async iterable of ``NotificationEnvelope`` objects, dispatches
    each event to the appropriate renderer, and writes the formatted
    output to a ``TextIO`` stream. Tracks event counts and provides
    a ``ConsumerResult`` when the stream ends.

    **StreamConsumerConfig**: Frozen dataclass with configuration for
    the consumer: verbosity, style settings, heartbeat display, and
    optional minimum severity filter.

    **ConsumerExitReason**: Why the consumer stopped (stream end, user
    cancel, error, or subscription lost).

    **ConsumerResult**: Frozen dataclass summarizing the consumption
    session (event counts and exit reason).

    **create_default_registry()**: Factory that creates a registry
    pre-loaded with renderers for all notification event types
    (completion, alert, heartbeat) and optionally all agent loop
    event types (tool_call, approval_prompt, observation, error,
    status_change).

Usage::

    from jules_daemon.cli.stream_consumer import (
        EventStreamConsumer,
        StreamConsumerConfig,
        create_default_registry,
    )

    registry = create_default_registry()
    config = StreamConsumerConfig(verbose=True)
    consumer = EventStreamConsumer(registry=registry, config=config)

    # With SubscriptionClient (implements __aiter__)
    result = await consumer.consume_client(subscription_client)

    # Or with any async iterable
    result = await consumer.consume_iter(async_event_generator)

    print(f"Processed {result.events_processed}, "
          f"displayed {result.events_displayed}")
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TextIO

from jules_daemon.cli.event_renderer import (
    EventRenderer,
    EventSeverity,
    RenderContext,
    RenderedOutput,
)
from jules_daemon.cli.styles import StyleConfig
from jules_daemon.protocol.notifications import NotificationEnvelope

__all__ = [
    "ConsumerExitReason",
    "ConsumerResult",
    "EventStreamConsumer",
    "RendererRegistry",
    "StreamConsumerConfig",
    "create_default_registry",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity ordering for filtering
# ---------------------------------------------------------------------------

_SEVERITY_ORDER: dict[EventSeverity, int] = {
    EventSeverity.DEBUG: 0,
    EventSeverity.INFO: 1,
    EventSeverity.SUCCESS: 2,
    EventSeverity.WARNING: 3,
    EventSeverity.ERROR: 4,
}


# ---------------------------------------------------------------------------
# ConsumerExitReason enum
# ---------------------------------------------------------------------------


class ConsumerExitReason(Enum):
    """Why the event stream consumer stopped.

    Values:
        STREAM_END:        The event source was exhausted normally.
        USER_CANCEL:       The user cancelled via Ctrl+C or task cancel.
        CONSUMER_ERROR:    An unrecoverable error occurred during processing.
        SUBSCRIPTION_LOST: The underlying subscription connection dropped.
    """

    STREAM_END = "stream_end"
    USER_CANCEL = "user_cancel"
    CONSUMER_ERROR = "consumer_error"
    SUBSCRIPTION_LOST = "subscription_lost"


# ---------------------------------------------------------------------------
# StreamConsumerConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamConsumerConfig:
    """Immutable configuration for the event stream consumer.

    Attributes:
        verbose:          When True, renderers include extra detail.
        style:            Terminal styling configuration.
        show_heartbeats:  When True, heartbeat events are displayed.
                          Default False (heartbeats are routine noise).
        min_severity:     Optional minimum severity filter. Events with
                          severity below this threshold are suppressed.
                          When None, all severities are displayed.
    """

    verbose: bool = False
    style: StyleConfig = field(default_factory=StyleConfig)
    show_heartbeats: bool = False
    min_severity: EventSeverity | None = None


# ---------------------------------------------------------------------------
# ConsumerResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumerResult:
    """Immutable result of a stream consumption session.

    Attributes:
        events_processed: Total notification envelopes received.
        events_displayed: Events that matched a renderer and passed
                          filters, resulting in terminal output.
        events_dropped:   Events with no matching renderer or that
                          were filtered out.
        exit_reason:      Why the consumer stopped.
        error_message:    Human-readable error description, or None.
    """

    events_processed: int
    events_displayed: int
    events_dropped: int
    exit_reason: ConsumerExitReason
    error_message: str | None = None


# ---------------------------------------------------------------------------
# RendererRegistry
# ---------------------------------------------------------------------------

# Empty registry constant, shared by all empty registries
_EMPTY_RENDERERS: MappingProxyType[str, EventRenderer] = MappingProxyType({})


@dataclass(frozen=True)
class RendererRegistry:
    """Immutable registry mapping event type strings to EventRenderers.

    The registry follows the project-wide builder pattern. Mutation
    methods return new instances with the change applied -- the original
    is never modified.

    The event type string used as the key is the ``event_type`` property
    of the registered ``EventRenderer`` instance.

    Attributes:
        _renderers: Internal mapping (MappingProxyType for immutability).
    """

    _renderers: MappingProxyType[str, EventRenderer] = field(
        default=_EMPTY_RENDERERS,
    )

    def register(self, renderer: EventRenderer) -> RendererRegistry:
        """Return a new registry with the given renderer added.

        If a renderer is already registered for the same event type,
        it is replaced. The original registry is not modified.

        Args:
            renderer: An EventRenderer to register.

        Returns:
            New RendererRegistry with the renderer added.

        Raises:
            TypeError: If renderer does not satisfy the EventRenderer protocol.
        """
        if not isinstance(renderer, EventRenderer):
            raise TypeError(
                f"Expected EventRenderer, got {type(renderer).__name__}"
            )

        new_map = dict(self._renderers)
        new_map[renderer.event_type] = renderer
        return RendererRegistry(_renderers=MappingProxyType(new_map))

    def register_all(
        self, renderers: tuple[EventRenderer, ...],
    ) -> RendererRegistry:
        """Return a new registry with all given renderers added.

        Validates all renderers before applying changes. If any renderer
        is invalid, raises TypeError without modifying the registry.

        Args:
            renderers: Tuple of EventRenderers to register.

        Returns:
            New RendererRegistry with all renderers added.

        Raises:
            TypeError: If any renderer does not satisfy the EventRenderer
                       protocol.
        """
        # Validate all first (fail fast, no partial updates)
        for renderer in renderers:
            if not isinstance(renderer, EventRenderer):
                raise TypeError(
                    f"Expected EventRenderer, got {type(renderer).__name__}"
                )

        new_map = dict(self._renderers)
        for renderer in renderers:
            new_map[renderer.event_type] = renderer
        return RendererRegistry(_renderers=MappingProxyType(new_map))

    def get(self, event_type: str) -> EventRenderer | None:
        """Look up a renderer by event type string.

        Args:
            event_type: The event type to look up.

        Returns:
            The registered EventRenderer, or None if not found.
        """
        return self._renderers.get(event_type)

    @property
    def registered_types(self) -> frozenset[str]:
        """The set of event type strings with registered renderers."""
        return frozenset(self._renderers.keys())

    def __len__(self) -> int:
        """Return the number of registered renderers."""
        return len(self._renderers)

    def __contains__(self, event_type: str) -> bool:
        """Check if a renderer is registered for the event type."""
        return event_type in self._renderers


# ---------------------------------------------------------------------------
# EventStreamConsumer
# ---------------------------------------------------------------------------


class EventStreamConsumer:
    """Async consumer that reads notification events and writes formatted output.

    Subscribes to a daemon notification stream (or any async iterable of
    ``NotificationEnvelope`` objects), routes each event to the appropriate
    renderer via the ``RendererRegistry``, and writes the formatted text
    to the terminal.

    The consumer tracks three event counters:
    - ``events_processed``: Total envelopes received from the stream.
    - ``events_displayed``: Events that produced terminal output.
    - ``events_dropped``: Events with no renderer or filtered out.

    The consumer is designed for single-use: create a new instance for
    each consumption session.

    Note on mutability: Unlike the frozen dataclass value objects in this
    module, this class intentionally holds mutable internal state (event
    counters, exit tracking) because it is an active coroutine, not a
    value object.

    Args:
        registry: The renderer registry for event dispatch.
        config:   Consumer configuration.
        output:   Text IO stream for writing output. Defaults to sys.stdout.
    """

    def __init__(
        self,
        *,
        registry: RendererRegistry,
        config: StreamConsumerConfig | None = None,
        output: TextIO | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or StreamConsumerConfig()
        self._output = output or sys.stdout

        # Mutable per-session state
        self._events_processed: int = 0
        self._events_displayed: int = 0
        self._events_dropped: int = 0

    # -- Properties -----------------------------------------------------------

    @property
    def registry(self) -> RendererRegistry:
        """The renderer registry used for event dispatch."""
        return self._registry

    @property
    def config(self) -> StreamConsumerConfig:
        """The consumer configuration."""
        return self._config

    @property
    def events_processed(self) -> int:
        """Total notification envelopes received so far."""
        return self._events_processed

    @property
    def events_displayed(self) -> int:
        """Events that produced terminal output so far."""
        return self._events_displayed

    @property
    def events_dropped(self) -> int:
        """Events with no renderer or filtered out so far."""
        return self._events_dropped

    # -- Public API -----------------------------------------------------------

    async def consume_iter(
        self,
        event_source: AsyncIterator[NotificationEnvelope],
    ) -> ConsumerResult:
        """Consume events from an async iterator until exhaustion.

        Reads ``NotificationEnvelope`` objects from the source, dispatches
        each to the appropriate renderer, and writes the formatted output
        to the terminal.

        The consumer handles three stop conditions:
        - Stream exhaustion (StopAsyncIteration): normal completion.
        - Cancellation (CancelledError): user cancel via Ctrl+C.
        - Unexpected error: logged and captured in the result.

        Args:
            event_source: Async iterator yielding NotificationEnvelope
                          objects. The ``SubscriptionClient`` satisfies
                          this interface via ``__aiter__``.

        Returns:
            ConsumerResult summarizing the session.
        """
        exit_reason = ConsumerExitReason.STREAM_END
        error_message: str | None = None

        try:
            async for envelope in event_source:
                self._process_envelope(envelope)
        except asyncio.CancelledError:
            exit_reason = ConsumerExitReason.USER_CANCEL
            logger.info("Stream consumer cancelled by user")
        except ConnectionError as exc:
            exit_reason = ConsumerExitReason.SUBSCRIPTION_LOST
            error_message = f"Subscription lost: {exc}"
            logger.warning("Subscription connection lost: %s", exc)
        except Exception as exc:
            exit_reason = ConsumerExitReason.CONSUMER_ERROR
            error_message = f"Consumer error: {type(exc).__name__}: {exc}"
            logger.error(
                "Stream consumer error: %s: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )

        return ConsumerResult(
            events_processed=self._events_processed,
            events_displayed=self._events_displayed,
            events_dropped=self._events_dropped,
            exit_reason=exit_reason,
            error_message=error_message,
        )

    async def consume_client(
        self,
        client: object,
    ) -> ConsumerResult:
        """Consume events from a SubscriptionClient.

        Convenience method that extracts the async iterator from the
        client and delegates to ``consume_iter()``. Accepts any object
        that implements ``__aiter__`` returning ``NotificationEnvelope``
        objects.

        Args:
            client: An object implementing ``__aiter__`` (e.g.,
                    ``SubscriptionClient``).

        Returns:
            ConsumerResult summarizing the session.

        Raises:
            TypeError: If client does not implement ``__aiter__``.
        """
        if not hasattr(client, "__aiter__"):
            raise TypeError(
                f"Client must implement __aiter__, "
                f"got {type(client).__name__}"
            )

        return await self.consume_iter(client.__aiter__())

    # -- Internal: event processing -------------------------------------------

    def _process_envelope(
        self,
        envelope: NotificationEnvelope,
    ) -> None:
        """Process a single notification envelope.

        Determines the event type, looks up the renderer, applies
        filters, formats the event, and writes to the terminal.

        Args:
            envelope: The notification envelope to process.
        """
        self._events_processed += 1
        event_type = envelope.event_type.value

        # Skip heartbeats unless configured to show them
        if event_type == "heartbeat" and not self._config.show_heartbeats:
            self._events_dropped += 1
            logger.debug("Heartbeat suppressed (show_heartbeats=False)")
            return

        # Look up renderer
        renderer = self._registry.get(event_type)
        if renderer is None:
            self._events_dropped += 1
            logger.debug(
                "No renderer for event type %r; dropping event %s",
                event_type,
                envelope.event_id,
            )
            return

        # Build render context
        context = RenderContext(
            style=self._config.style,
            verbose=self._config.verbose,
        )

        # Render the event
        try:
            rendered = renderer.render(envelope.payload, context)
        except (TypeError, ValueError) as exc:
            self._events_dropped += 1
            logger.warning(
                "Renderer for %r raised %s: %s",
                event_type,
                type(exc).__name__,
                exc,
            )
            return

        # Apply severity filter
        if not self._meets_severity_threshold(rendered.severity):
            self._events_dropped += 1
            return

        # Write to terminal
        self._write_output(rendered)
        self._events_displayed += 1

    def _meets_severity_threshold(self, severity: EventSeverity) -> bool:
        """Check if a severity meets the configured minimum threshold.

        When no minimum severity is configured, all events pass.

        Args:
            severity: The severity to check.

        Returns:
            True if the severity meets or exceeds the threshold.
        """
        if self._config.min_severity is None:
            return True

        event_order = _SEVERITY_ORDER.get(severity, 0)
        threshold_order = _SEVERITY_ORDER.get(self._config.min_severity, 0)
        return event_order >= threshold_order

    def _write_output(self, rendered: RenderedOutput) -> None:
        """Write rendered output text to the terminal.

        Skips empty output. Flushes the stream after each write to
        ensure the user sees output immediately.

        Args:
            rendered: The rendered output to write.
        """
        if not rendered.text:
            return

        self._output.write(rendered.text)
        self._output.write("\n")
        self._output.flush()


# ---------------------------------------------------------------------------
# Factory: create_default_registry
# ---------------------------------------------------------------------------


def create_default_registry(
    *,
    include_agent_events: bool = False,
) -> RendererRegistry:
    """Create a RendererRegistry pre-loaded with notification renderers.

    Registers renderers for all three daemon notification event types:
    completion, alert, and heartbeat.

    When ``include_agent_events`` is True, also registers renderers for
    the five agent loop event types: tool_call, approval_prompt,
    observation, error, and status_change.

    Args:
        include_agent_events: Whether to include agent loop event
                              renderers. Default False.

    Returns:
        Pre-configured RendererRegistry.
    """
    from jules_daemon.cli.notification_formats import (
        AlertRenderer,
        CompletionRenderer,
        HeartbeatRenderer,
    )

    registry = RendererRegistry()
    registry = registry.register_all((
        CompletionRenderer(),
        AlertRenderer(),
        HeartbeatRenderer(),
    ))

    if include_agent_events:
        from jules_daemon.cli.event_formats import (
            ApprovalPromptRenderer,
            ErrorRenderer,
            ObservationRenderer,
            StatusChangeRenderer,
            ToolCallRenderer,
        )

        registry = registry.register_all((
            ToolCallRenderer(),
            ApprovalPromptRenderer(),
            ObservationRenderer(),
            ErrorRenderer(),
            StatusChangeRenderer(),
        ))

    return registry
