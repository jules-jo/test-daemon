"""CLI layer for daemon interaction.

Modules:
    args_builder: Build typed *Args dataclasses from classification dicts
    confirmation: Terminal-based editable confirmation prompt for SSH commands
    dispatcher: Command dispatcher routing ParsedCommands to handler callables
    entry_point: Unified CLI entry point (classify -> resolve -> build -> dispatch)
    event_renderer: Base renderer types and EventRenderer protocol
    notification_formats: Format functions for daemon notification events
    parser: Tokenizer and verb parser for raw CLI input strings
    registry: Command handler registry mapping verbs to handler entries
    signal_handler: Signal trap handlers for graceful daemon socket detach
    stream_consumer: Event stream consumer/dispatcher for notification routing
    styles: Terminal styling utilities (colors, icons, layout helpers)
    terminal_renderer: Terminal output renderer with progress indicators
    verbs: Typed data models for the six CLI verbs and their argument schemas
"""

from jules_daemon.cli.args_builder import build_verb_args
from jules_daemon.cli.event_renderer import (
    EventRenderer,
    EventSeverity,
    RenderContext,
    RenderedOutput,
    render_footer,
    render_header,
)
from jules_daemon.cli.notification_formats import (
    AlertRenderer,
    CompletionRenderer,
    HeartbeatRenderer,
    format_alert,
    format_completion,
    format_heartbeat,
)
from jules_daemon.cli.confirmation import (
    ConfirmationRequest,
    ConfirmationResult,
    Decision,
    confirm_ssh_command,
    render_confirmation_display,
)
from jules_daemon.cli.dispatcher import (
    CommandDispatcher,
    DispatchResponse,
    create_dispatcher,
)
from jules_daemon.cli.entry_point import (
    InputProcessingResult,
    process_input,
)
from jules_daemon.cli.signal_handler import (
    DetachReason,
    DetachResult,
    SignalState,
    SignalTrapConfig,
    SignalTrapHandler,
    create_signal_handler,
    format_detach_message,
)
from jules_daemon.cli.parser import (
    ParseError,
    classify_structured_command,
    normalize_verb,
    parse_command,
    tokenize,
)
from jules_daemon.cli.registry import (
    CommandHandlerRegistry,
    HandlerCallable,
    HandlerEntry,
    create_registry,
)
from jules_daemon.cli.stream_consumer import (
    ConsumerExitReason,
    ConsumerResult,
    EventStreamConsumer,
    RendererRegistry,
    StreamConsumerConfig,
    create_default_registry,
)
from jules_daemon.cli.styles import (
    Color,
    ICON_APPROVE,
    ICON_ARROW,
    ICON_BULLET,
    ICON_CANCEL,
    ICON_DONE,
    ICON_ERROR,
    ICON_INFO,
    ICON_PENDING,
    ICON_REJECT,
    ICON_RUNNING,
    ICON_TOOL,
    ICON_WARNING,
    StyleConfig,
    box_bottom,
    box_line,
    box_top,
    horizontal_rule,
    indent,
    pad_right,
    styled,
    truncate,
)
from jules_daemon.cli.terminal_renderer import (
    ProgressMatch,
    ProgressType,
    RenderResult,
    RendererConfig,
    TerminalExitReason,
    TerminalRenderer,
    detect_progress_pattern,
    format_progress_bar,
    format_spinner_frame,
)
from jules_daemon.cli.verbs import (
    CancelArgs,
    HistoryArgs,
    ParsedCommand,
    QueueArgs,
    RunArgs,
    StatusArgs,
    Verb,
    WatchArgs,
    parse_verb,
)

__all__ = [
    "AlertRenderer",
    "CancelArgs",
    "Color",
    "CommandDispatcher",
    "CommandHandlerRegistry",
    "CompletionRenderer",
    "ConfirmationRequest",
    "ConfirmationResult",
    "ConsumerExitReason",
    "ConsumerResult",
    "Decision",
    "DetachReason",
    "DetachResult",
    "DispatchResponse",
    "EventRenderer",
    "EventSeverity",
    "EventStreamConsumer",
    "HandlerCallable",
    "HandlerEntry",
    "HeartbeatRenderer",
    "HistoryArgs",
    "ICON_APPROVE",
    "ICON_ARROW",
    "ICON_BULLET",
    "ICON_CANCEL",
    "ICON_DONE",
    "ICON_ERROR",
    "ICON_INFO",
    "ICON_PENDING",
    "ICON_REJECT",
    "ICON_RUNNING",
    "ICON_TOOL",
    "ICON_WARNING",
    "InputProcessingResult",
    "ParseError",
    "ParsedCommand",
    "ProgressMatch",
    "ProgressType",
    "QueueArgs",
    "RenderContext",
    "RendererRegistry",
    "RenderResult",
    "RenderedOutput",
    "RendererConfig",
    "RunArgs",
    "SignalState",
    "SignalTrapConfig",
    "SignalTrapHandler",
    "StatusArgs",
    "StreamConsumerConfig",
    "StyleConfig",
    "TerminalExitReason",
    "TerminalRenderer",
    "Verb",
    "WatchArgs",
    "box_bottom",
    "box_line",
    "box_top",
    "build_verb_args",
    "classify_structured_command",
    "confirm_ssh_command",
    "create_default_registry",
    "create_dispatcher",
    "create_registry",
    "create_signal_handler",
    "detect_progress_pattern",
    "format_alert",
    "format_completion",
    "format_detach_message",
    "format_heartbeat",
    "format_progress_bar",
    "format_spinner_frame",
    "horizontal_rule",
    "indent",
    "normalize_verb",
    "pad_right",
    "parse_command",
    "parse_verb",
    "process_input",
    "render_confirmation_display",
    "render_footer",
    "render_header",
    "styled",
    "tokenize",
    "truncate",
]
