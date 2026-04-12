"""Tests for the event stream consumer/dispatcher.

Covers:
- RendererRegistry: register, get, register_all, immutability, contains
- StreamConsumerConfig: defaults, frozen, validation
- ConsumerExitReason: enum values
- ConsumerResult: frozen dataclass
- EventStreamConsumer: routing, filtering, terminal output writing
- EventStreamConsumer.consume_iter: async iteration, cancellation, errors
- EventStreamConsumer.consume_client: delegation to consume_iter
- Heartbeat suppression logic
- Severity filtering
- Unregistered event type handling
- create_default_registry: with and without agent events
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest

from jules_daemon.cli.event_renderer import (
    EventRenderer,
    EventSeverity,
    RenderContext,
    RenderedOutput,
)
from jules_daemon.cli.notification_formats import (
    AlertRenderer,
    CompletionRenderer,
    HeartbeatRenderer,
)
from jules_daemon.cli.stream_consumer import (
    ConsumerExitReason,
    ConsumerResult,
    EventStreamConsumer,
    RendererRegistry,
    StreamConsumerConfig,
    create_default_registry,
)
from jules_daemon.cli.styles import StyleConfig
from jules_daemon.protocol.notifications import (
    AlertNotification,
    CompletionNotification,
    HeartbeatNotification,
    NotificationEnvelope,
    NotificationEventType,
    NotificationSeverity,
    TestOutcomeSummary,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_TS = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


def _make_completion_envelope(
    run_id: str = "run-001",
    exit_status: int = 0,
    event_id: str = "evt-001",
) -> NotificationEnvelope:
    """Build a completion notification envelope."""
    return NotificationEnvelope(
        channel_version="1.0.0",
        event_id=event_id,
        timestamp=_TS,
        event_type=NotificationEventType.COMPLETION,
        payload=CompletionNotification(
            run_id=run_id,
            natural_language_command="Run tests",
            exit_status=exit_status,
        ),
    )


def _make_alert_envelope(
    title: str = "Alert title",
    severity: NotificationSeverity = NotificationSeverity.WARNING,
    event_id: str = "evt-002",
) -> NotificationEnvelope:
    """Build an alert notification envelope."""
    return NotificationEnvelope(
        channel_version="1.0.0",
        event_id=event_id,
        timestamp=_TS,
        event_type=NotificationEventType.ALERT,
        payload=AlertNotification(
            severity=severity,
            title=title,
            message="Alert message body",
        ),
    )


def _make_heartbeat_envelope(
    uptime: float = 120.0,
    event_id: str = "evt-003",
) -> NotificationEnvelope:
    """Build a heartbeat notification envelope."""
    return NotificationEnvelope(
        channel_version="1.0.0",
        event_id=event_id,
        timestamp=_TS,
        event_type=NotificationEventType.HEARTBEAT,
        payload=HeartbeatNotification(
            daemon_uptime_seconds=uptime,
        ),
    )


async def _async_iter(
    envelopes: list[NotificationEnvelope],
) -> AsyncIterator[NotificationEnvelope]:
    """Create an async iterator from a list of envelopes."""
    for envelope in envelopes:
        yield envelope


class _StubRenderer:
    """Minimal renderer for testing registry dispatch."""

    def __init__(self, event_type: str, text: str = "rendered") -> None:
        self._event_type = event_type
        self._text = text

    @property
    def event_type(self) -> str:
        return self._event_type

    def render(self, event: object, context: RenderContext) -> RenderedOutput:
        return RenderedOutput(text=self._text, line_count=1)


class _ErrorRenderer:
    """Renderer that always raises TypeError."""

    @property
    def event_type(self) -> str:
        return "error_type"

    def render(self, event: object, context: RenderContext) -> RenderedOutput:
        raise TypeError("Intentional test error")


class _AsyncIterableClient:
    """Fake client that implements __aiter__ for testing consume_client."""

    def __init__(self, envelopes: list[NotificationEnvelope]) -> None:
        self._envelopes = envelopes

    def __aiter__(self) -> AsyncIterator[NotificationEnvelope]:
        return _async_iter(self._envelopes)


# ---------------------------------------------------------------------------
# RendererRegistry tests
# ---------------------------------------------------------------------------


class TestRendererRegistry:
    """Tests for the immutable RendererRegistry."""

    def test_empty_registry(self) -> None:
        registry = RendererRegistry()
        assert len(registry) == 0
        assert registry.registered_types == frozenset()

    def test_register_single(self) -> None:
        registry = RendererRegistry()
        renderer = _StubRenderer("test_type")
        new_registry = registry.register(renderer)

        assert len(new_registry) == 1
        assert "test_type" in new_registry
        assert new_registry.get("test_type") is renderer

    def test_register_preserves_original(self) -> None:
        """Registering a renderer returns a new registry; original is unchanged."""
        original = RendererRegistry()
        renderer = _StubRenderer("test_type")
        _new = original.register(renderer)

        assert len(original) == 0
        assert "test_type" not in original

    def test_register_replaces_existing(self) -> None:
        """Registering for an existing type replaces the renderer."""
        renderer1 = _StubRenderer("same_type", text="first")
        renderer2 = _StubRenderer("same_type", text="second")

        registry = RendererRegistry().register(renderer1)
        updated = registry.register(renderer2)

        assert updated.get("same_type") is renderer2

    def test_register_all(self) -> None:
        renderers = (
            _StubRenderer("type_a"),
            _StubRenderer("type_b"),
            _StubRenderer("type_c"),
        )
        registry = RendererRegistry().register_all(renderers)

        assert len(registry) == 3
        assert registry.registered_types == frozenset({"type_a", "type_b", "type_c"})

    def test_register_all_validates_all_first(self) -> None:
        """register_all validates all renderers before applying changes."""
        with pytest.raises(TypeError, match="Expected EventRenderer"):
            RendererRegistry().register_all(
                (_StubRenderer("valid"), "not a renderer"),  # type: ignore[arg-type]
            )

    def test_register_invalid_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected EventRenderer"):
            RendererRegistry().register("not a renderer")  # type: ignore[arg-type]

    def test_get_missing_returns_none(self) -> None:
        registry = RendererRegistry()
        assert registry.get("nonexistent") is None

    def test_contains_true(self) -> None:
        registry = RendererRegistry().register(_StubRenderer("my_type"))
        assert "my_type" in registry

    def test_contains_false(self) -> None:
        registry = RendererRegistry()
        assert "my_type" not in registry

    def test_frozen(self) -> None:
        registry = RendererRegistry()
        with pytest.raises(AttributeError):
            registry._renderers = {}  # type: ignore[misc]

    def test_registered_types_returns_frozenset(self) -> None:
        renderers = (_StubRenderer("a"), _StubRenderer("b"))
        registry = RendererRegistry().register_all(renderers)
        types = registry.registered_types
        assert isinstance(types, frozenset)
        assert types == frozenset({"a", "b"})


# ---------------------------------------------------------------------------
# StreamConsumerConfig tests
# ---------------------------------------------------------------------------


class TestStreamConsumerConfig:
    """Tests for the StreamConsumerConfig frozen dataclass."""

    def test_defaults(self) -> None:
        config = StreamConsumerConfig()
        assert config.verbose is False
        assert config.show_heartbeats is False
        assert config.min_severity is None
        assert isinstance(config.style, StyleConfig)

    def test_frozen(self) -> None:
        config = StreamConsumerConfig()
        with pytest.raises(AttributeError):
            config.verbose = True  # type: ignore[misc]

    def test_custom_values(self) -> None:
        style = StyleConfig(color_enabled=False)
        config = StreamConsumerConfig(
            verbose=True,
            style=style,
            show_heartbeats=True,
            min_severity=EventSeverity.WARNING,
        )
        assert config.verbose is True
        assert config.show_heartbeats is True
        assert config.min_severity == EventSeverity.WARNING
        assert config.style.color_enabled is False


# ---------------------------------------------------------------------------
# ConsumerExitReason tests
# ---------------------------------------------------------------------------


class TestConsumerExitReason:
    """Tests for the ConsumerExitReason enum."""

    def test_all_values(self) -> None:
        assert ConsumerExitReason.STREAM_END.value == "stream_end"
        assert ConsumerExitReason.USER_CANCEL.value == "user_cancel"
        assert ConsumerExitReason.CONSUMER_ERROR.value == "consumer_error"
        assert ConsumerExitReason.SUBSCRIPTION_LOST.value == "subscription_lost"

    def test_count(self) -> None:
        assert len(ConsumerExitReason) == 4


# ---------------------------------------------------------------------------
# ConsumerResult tests
# ---------------------------------------------------------------------------


class TestConsumerResult:
    """Tests for the ConsumerResult frozen dataclass."""

    def test_creation(self) -> None:
        result = ConsumerResult(
            events_processed=10,
            events_displayed=8,
            events_dropped=2,
            exit_reason=ConsumerExitReason.STREAM_END,
        )
        assert result.events_processed == 10
        assert result.events_displayed == 8
        assert result.events_dropped == 2
        assert result.exit_reason == ConsumerExitReason.STREAM_END
        assert result.error_message is None

    def test_with_error_message(self) -> None:
        result = ConsumerResult(
            events_processed=5,
            events_displayed=3,
            events_dropped=2,
            exit_reason=ConsumerExitReason.CONSUMER_ERROR,
            error_message="Something went wrong",
        )
        assert result.error_message == "Something went wrong"

    def test_frozen(self) -> None:
        result = ConsumerResult(
            events_processed=0,
            events_displayed=0,
            events_dropped=0,
            exit_reason=ConsumerExitReason.STREAM_END,
        )
        with pytest.raises(AttributeError):
            result.events_processed = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EventStreamConsumer -- basic routing
# ---------------------------------------------------------------------------


class TestEventStreamConsumerRouting:
    """Tests for event routing through the consumer."""

    @pytest.mark.asyncio
    async def test_routes_completion_to_renderer(self) -> None:
        """Completion events are routed to the CompletionRenderer."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            style=StyleConfig(color_enabled=False),
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelope = _make_completion_envelope(exit_status=0)
        result = await consumer.consume_iter(_async_iter([envelope]))

        assert result.events_processed == 1
        assert result.events_displayed == 1
        assert "PASSED" in output.getvalue()

    @pytest.mark.asyncio
    async def test_routes_alert_to_renderer(self) -> None:
        """Alert events are routed to the AlertRenderer."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            style=StyleConfig(color_enabled=False),
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelope = _make_alert_envelope(title="Test alert")
        result = await consumer.consume_iter(_async_iter([envelope]))

        assert result.events_processed == 1
        assert result.events_displayed == 1
        assert "Test alert" in output.getvalue()

    @pytest.mark.asyncio
    async def test_routes_heartbeat_when_enabled(self) -> None:
        """Heartbeat events are routed when show_heartbeats=True."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            style=StyleConfig(color_enabled=False),
            show_heartbeats=True,
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelope = _make_heartbeat_envelope()
        result = await consumer.consume_iter(_async_iter([envelope]))

        assert result.events_processed == 1
        assert result.events_displayed == 1
        assert "Heartbeat" in output.getvalue()

    @pytest.mark.asyncio
    async def test_multiple_events_routed(self) -> None:
        """Multiple events of different types are all routed correctly."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            style=StyleConfig(color_enabled=False),
            show_heartbeats=True,
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelopes = [
            _make_completion_envelope(event_id="e1"),
            _make_alert_envelope(event_id="e2"),
            _make_heartbeat_envelope(event_id="e3"),
        ]
        result = await consumer.consume_iter(_async_iter(envelopes))

        assert result.events_processed == 3
        assert result.events_displayed == 3
        text = output.getvalue()
        assert "PASSED" in text
        assert "Alert title" in text
        assert "Heartbeat" in text

    @pytest.mark.asyncio
    async def test_empty_stream(self) -> None:
        """Empty event stream produces zero-count result."""
        output = io.StringIO()
        registry = create_default_registry()
        consumer = EventStreamConsumer(
            registry=registry, output=output,
        )

        result = await consumer.consume_iter(_async_iter([]))

        assert result.events_processed == 0
        assert result.events_displayed == 0
        assert result.events_dropped == 0
        assert result.exit_reason == ConsumerExitReason.STREAM_END
        assert output.getvalue() == ""


# ---------------------------------------------------------------------------
# EventStreamConsumer -- heartbeat suppression
# ---------------------------------------------------------------------------


class TestHeartbeatSuppression:
    """Tests for heartbeat event suppression."""

    @pytest.mark.asyncio
    async def test_heartbeats_suppressed_by_default(self) -> None:
        """Heartbeat events are dropped when show_heartbeats=False (default)."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(show_heartbeats=False)
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelope = _make_heartbeat_envelope()
        result = await consumer.consume_iter(_async_iter([envelope]))

        assert result.events_processed == 1
        assert result.events_displayed == 0
        assert result.events_dropped == 1
        assert output.getvalue() == ""

    @pytest.mark.asyncio
    async def test_non_heartbeat_not_affected(self) -> None:
        """Non-heartbeat events are unaffected by heartbeat suppression."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            show_heartbeats=False,
            style=StyleConfig(color_enabled=False),
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelopes = [
            _make_completion_envelope(event_id="e1"),
            _make_heartbeat_envelope(event_id="e2"),
            _make_alert_envelope(event_id="e3"),
        ]
        result = await consumer.consume_iter(_async_iter(envelopes))

        assert result.events_processed == 3
        assert result.events_displayed == 2  # completion + alert
        assert result.events_dropped == 1   # heartbeat
        assert "PASSED" in output.getvalue()
        assert "Heartbeat" not in output.getvalue()


# ---------------------------------------------------------------------------
# EventStreamConsumer -- severity filtering
# ---------------------------------------------------------------------------


class TestSeverityFiltering:
    """Tests for minimum severity filtering."""

    @pytest.mark.asyncio
    async def test_no_filter_shows_all(self) -> None:
        """With no severity filter, all events are displayed."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            style=StyleConfig(color_enabled=False),
            show_heartbeats=True,
            min_severity=None,
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelopes = [
            _make_completion_envelope(event_id="e1"),
            _make_heartbeat_envelope(event_id="e2"),
        ]
        result = await consumer.consume_iter(_async_iter(envelopes))

        assert result.events_displayed == 2

    @pytest.mark.asyncio
    async def test_warning_filter_drops_low_severity(self) -> None:
        """WARNING filter drops INFO and DEBUG events."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            style=StyleConfig(color_enabled=False),
            show_heartbeats=True,
            min_severity=EventSeverity.WARNING,
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelopes = [
            # Heartbeat -> DEBUG severity -> dropped
            _make_heartbeat_envelope(event_id="e1"),
            # Completion success -> SUCCESS severity -> dropped (below WARNING)
            _make_completion_envelope(exit_status=0, event_id="e2"),
            # Alert warning -> WARNING severity -> displayed
            _make_alert_envelope(
                severity=NotificationSeverity.WARNING, event_id="e3",
            ),
        ]
        result = await consumer.consume_iter(_async_iter(envelopes))

        assert result.events_processed == 3
        assert result.events_displayed == 1  # only the warning alert
        assert "Alert title" in output.getvalue()

    @pytest.mark.asyncio
    async def test_error_filter_shows_only_errors(self) -> None:
        """ERROR filter shows only error-severity events."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            style=StyleConfig(color_enabled=False),
            min_severity=EventSeverity.ERROR,
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelopes = [
            # Completion failure -> ERROR severity -> displayed
            _make_completion_envelope(exit_status=1, event_id="e1"),
            # Completion success -> SUCCESS severity -> dropped
            _make_completion_envelope(exit_status=0, event_id="e2"),
        ]
        result = await consumer.consume_iter(_async_iter(envelopes))

        assert result.events_displayed == 1
        assert "FAILED" in output.getvalue()
        assert "PASSED" not in output.getvalue()


# ---------------------------------------------------------------------------
# EventStreamConsumer -- unregistered event types
# ---------------------------------------------------------------------------


class TestUnregisteredEventTypes:
    """Tests for handling events without a matching renderer."""

    @pytest.mark.asyncio
    async def test_unregistered_type_dropped(self) -> None:
        """Events with no matching renderer are dropped gracefully."""
        output = io.StringIO()
        # Empty registry -- no renderers at all
        registry = RendererRegistry()
        consumer = EventStreamConsumer(
            registry=registry, output=output,
        )

        envelope = _make_completion_envelope()
        result = await consumer.consume_iter(_async_iter([envelope]))

        assert result.events_processed == 1
        assert result.events_displayed == 0
        assert result.events_dropped == 1
        assert output.getvalue() == ""


# ---------------------------------------------------------------------------
# EventStreamConsumer -- renderer error handling
# ---------------------------------------------------------------------------


class TestRendererErrorHandling:
    """Tests for handling renderer exceptions."""

    @pytest.mark.asyncio
    async def test_renderer_type_error_drops_event(self) -> None:
        """A renderer that raises TypeError drops the event gracefully."""
        output = io.StringIO()
        error_renderer = _ErrorRenderer()
        # Create a registry that maps 'completion' to the error renderer
        # We need to override the event_type. Use a stub.
        stub = _StubRenderer("completion")
        registry = RendererRegistry().register(error_renderer)

        # Build an envelope that maps to "error_type" (the error renderer)
        # We need a custom envelope - reuse alert since we can control type
        # Actually the event_type must be NotificationEventType, so let's
        # route completion to the error renderer via a custom wrapper

        # Simpler approach: just use a registry with only the error renderer
        # and send an envelope with an event_type whose .value matches
        consumer = EventStreamConsumer(
            registry=registry, output=output,
        )

        # We can't change NotificationEventType.COMPLETION.value,
        # but we can make an _ErrorRenderer with event_type="completion"
        class CompletionErrorRenderer:
            @property
            def event_type(self) -> str:
                return "completion"

            def render(self, event: object, context: RenderContext) -> RenderedOutput:
                raise TypeError("bad payload")

        registry2 = RendererRegistry().register(CompletionErrorRenderer())
        consumer2 = EventStreamConsumer(
            registry=registry2, output=output,
        )

        envelope = _make_completion_envelope()
        result = await consumer2.consume_iter(_async_iter([envelope]))

        assert result.events_processed == 1
        assert result.events_displayed == 0
        assert result.events_dropped == 1


# ---------------------------------------------------------------------------
# EventStreamConsumer -- cancellation and errors
# ---------------------------------------------------------------------------


class TestConsumerLifecycle:
    """Tests for consumer lifecycle and error handling."""

    @pytest.mark.asyncio
    async def test_stream_end_reason(self) -> None:
        """Normal stream exhaustion produces STREAM_END exit reason."""
        output = io.StringIO()
        registry = create_default_registry()
        consumer = EventStreamConsumer(registry=registry, output=output)

        result = await consumer.consume_iter(_async_iter([]))

        assert result.exit_reason == ConsumerExitReason.STREAM_END
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_cancellation_produces_user_cancel(self) -> None:
        """CancelledError during consumption produces USER_CANCEL."""
        output = io.StringIO()
        registry = create_default_registry()
        consumer = EventStreamConsumer(registry=registry, output=output)

        async def _hanging_iter() -> AsyncIterator[NotificationEnvelope]:
            yield _make_completion_envelope()
            # Hang forever to allow cancellation
            await asyncio.sleep(3600)
            yield _make_completion_envelope()  # pragma: no cover

        task = asyncio.create_task(
            consumer.consume_iter(_hanging_iter())
        )
        await asyncio.sleep(0.01)
        task.cancel()

        result = await task
        assert result.exit_reason == ConsumerExitReason.USER_CANCEL

    @pytest.mark.asyncio
    async def test_connection_error_produces_subscription_lost(self) -> None:
        """ConnectionError during iteration produces SUBSCRIPTION_LOST."""
        output = io.StringIO()
        registry = create_default_registry()
        consumer = EventStreamConsumer(registry=registry, output=output)

        async def _error_iter() -> AsyncIterator[NotificationEnvelope]:
            yield _make_completion_envelope()
            raise ConnectionError("Socket closed")

        result = await consumer.consume_iter(_error_iter())

        assert result.exit_reason == ConsumerExitReason.SUBSCRIPTION_LOST
        assert result.error_message is not None
        assert "Socket closed" in result.error_message

    @pytest.mark.asyncio
    async def test_unexpected_error_produces_consumer_error(self) -> None:
        """Unexpected exceptions produce CONSUMER_ERROR."""
        output = io.StringIO()
        registry = create_default_registry()
        consumer = EventStreamConsumer(registry=registry, output=output)

        async def _error_iter() -> AsyncIterator[NotificationEnvelope]:
            yield _make_completion_envelope()
            raise RuntimeError("unexpected boom")

        result = await consumer.consume_iter(_error_iter())

        assert result.exit_reason == ConsumerExitReason.CONSUMER_ERROR
        assert result.error_message is not None
        assert "unexpected boom" in result.error_message


# ---------------------------------------------------------------------------
# EventStreamConsumer.consume_client
# ---------------------------------------------------------------------------


class TestConsumeClient:
    """Tests for the consume_client convenience method."""

    @pytest.mark.asyncio
    async def test_delegates_to_consume_iter(self) -> None:
        """consume_client delegates to consume_iter via __aiter__."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            style=StyleConfig(color_enabled=False),
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        client = _AsyncIterableClient([_make_completion_envelope()])
        result = await consumer.consume_client(client)

        assert result.events_processed == 1
        assert result.events_displayed == 1
        assert "PASSED" in output.getvalue()

    @pytest.mark.asyncio
    async def test_raises_for_non_iterable(self) -> None:
        """consume_client raises TypeError for non-iterable clients."""
        registry = create_default_registry()
        consumer = EventStreamConsumer(registry=registry)

        with pytest.raises(TypeError, match="must implement __aiter__"):
            await consumer.consume_client("not iterable")


# ---------------------------------------------------------------------------
# EventStreamConsumer -- output writing
# ---------------------------------------------------------------------------


class TestOutputWriting:
    """Tests for terminal output writing behavior."""

    @pytest.mark.asyncio
    async def test_output_ends_with_newline(self) -> None:
        """Each rendered event is followed by a newline."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            style=StyleConfig(color_enabled=False),
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelope = _make_completion_envelope()
        await consumer.consume_iter(_async_iter([envelope]))

        text = output.getvalue()
        assert text.endswith("\n")

    @pytest.mark.asyncio
    async def test_multiple_events_produce_separate_blocks(self) -> None:
        """Each event's rendered output is separated by newlines."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            style=StyleConfig(color_enabled=False),
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelopes = [
            _make_completion_envelope(run_id="run-a", event_id="e1"),
            _make_completion_envelope(run_id="run-b", event_id="e2"),
        ]
        await consumer.consume_iter(_async_iter(envelopes))

        text = output.getvalue()
        assert "run-a" in text
        assert "run-b" in text

    @pytest.mark.asyncio
    async def test_verbose_mode_passes_to_renderer(self) -> None:
        """Verbose mode in config is passed through to the render context."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            verbose=True,
            style=StyleConfig(color_enabled=False),
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelope = NotificationEnvelope(
            channel_version="1.0.0",
            event_id="e1",
            timestamp=_TS,
            event_type=NotificationEventType.COMPLETION,
            payload=CompletionNotification(
                run_id="run-001",
                natural_language_command="Run tests",
                resolved_shell="cd /app && pytest",
                exit_status=0,
            ),
        )
        await consumer.consume_iter(_async_iter([envelope]))

        # In verbose mode, resolved_shell should be shown
        assert "cd /app && pytest" in output.getvalue()


# ---------------------------------------------------------------------------
# EventStreamConsumer -- counter tracking
# ---------------------------------------------------------------------------


class TestCounterTracking:
    """Tests for accurate event counter tracking."""

    @pytest.mark.asyncio
    async def test_counters_accurate(self) -> None:
        """Event counters accurately reflect processing outcomes."""
        output = io.StringIO()
        # Registry with only completion renderer
        registry = RendererRegistry().register(CompletionRenderer())
        config = StreamConsumerConfig(
            style=StyleConfig(color_enabled=False),
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        envelopes = [
            _make_completion_envelope(event_id="e1"),  # displayed
            _make_alert_envelope(event_id="e2"),        # dropped (no renderer)
            _make_heartbeat_envelope(event_id="e3"),    # dropped (suppressed)
            _make_completion_envelope(event_id="e4"),   # displayed
        ]
        result = await consumer.consume_iter(_async_iter(envelopes))

        assert result.events_processed == 4
        assert result.events_displayed == 2
        assert result.events_dropped == 2

    @pytest.mark.asyncio
    async def test_properties_track_progress(self) -> None:
        """Consumer properties track running totals during consumption."""
        output = io.StringIO()
        registry = create_default_registry()
        config = StreamConsumerConfig(
            style=StyleConfig(color_enabled=False),
        )
        consumer = EventStreamConsumer(
            registry=registry, config=config, output=output,
        )

        assert consumer.events_processed == 0
        assert consumer.events_displayed == 0
        assert consumer.events_dropped == 0

        await consumer.consume_iter(
            _async_iter([_make_completion_envelope()])
        )

        assert consumer.events_processed == 1
        assert consumer.events_displayed == 1


# ---------------------------------------------------------------------------
# create_default_registry tests
# ---------------------------------------------------------------------------


class TestCreateDefaultRegistry:
    """Tests for the create_default_registry factory."""

    def test_notification_types_registered(self) -> None:
        """Default registry has all three notification renderers."""
        registry = create_default_registry()

        assert len(registry) == 3
        assert "completion" in registry
        assert "alert" in registry
        assert "heartbeat" in registry

    def test_agent_events_excluded_by_default(self) -> None:
        """Agent loop event renderers are excluded by default."""
        registry = create_default_registry()

        assert "tool_call" not in registry
        assert "approval_prompt" not in registry
        assert "observation" not in registry
        assert "error" not in registry
        assert "status_change" not in registry

    def test_include_agent_events(self) -> None:
        """With include_agent_events=True, all 8 renderers are registered."""
        registry = create_default_registry(include_agent_events=True)

        assert len(registry) == 8
        assert "completion" in registry
        assert "alert" in registry
        assert "heartbeat" in registry
        assert "tool_call" in registry
        assert "approval_prompt" in registry
        assert "observation" in registry
        assert "error" in registry
        assert "status_change" in registry

    def test_renderers_satisfy_protocol(self) -> None:
        """All registered renderers satisfy the EventRenderer protocol."""
        registry = create_default_registry(include_agent_events=True)

        for event_type in registry.registered_types:
            renderer = registry.get(event_type)
            assert isinstance(renderer, EventRenderer), (
                f"Renderer for {event_type!r} does not satisfy EventRenderer"
            )


# ---------------------------------------------------------------------------
# EventStreamConsumer -- properties
# ---------------------------------------------------------------------------


class TestConsumerProperties:
    """Tests for the consumer's read-only properties."""

    def test_registry_property(self) -> None:
        registry = create_default_registry()
        consumer = EventStreamConsumer(registry=registry)
        assert consumer.registry is registry

    def test_config_default(self) -> None:
        registry = create_default_registry()
        consumer = EventStreamConsumer(registry=registry)
        assert isinstance(consumer.config, StreamConsumerConfig)
        assert consumer.config.verbose is False

    def test_config_custom(self) -> None:
        registry = create_default_registry()
        config = StreamConsumerConfig(verbose=True)
        consumer = EventStreamConsumer(registry=registry, config=config)
        assert consumer.config.verbose is True
