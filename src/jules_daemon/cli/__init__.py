"""CLI layer for daemon interaction.

Modules:
    args_builder: Build typed *Args dataclasses from classification dicts
    confirmation: Terminal-based editable confirmation prompt for SSH commands
    dispatcher: Command dispatcher routing ParsedCommands to handler callables
    entry_point: Unified CLI entry point (classify -> resolve -> build -> dispatch)
    parser: Tokenizer and verb parser for raw CLI input strings
    registry: Command handler registry mapping verbs to handler entries
    signal_handler: Signal trap handlers for graceful daemon socket detach
    terminal_renderer: Terminal output renderer with progress indicators
    verbs: Typed data models for the six CLI verbs and their argument schemas
"""

from jules_daemon.cli.args_builder import build_verb_args
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
    "CancelArgs",
    "CommandDispatcher",
    "CommandHandlerRegistry",
    "ConfirmationRequest",
    "ConfirmationResult",
    "Decision",
    "DetachReason",
    "DetachResult",
    "DispatchResponse",
    "HandlerCallable",
    "HandlerEntry",
    "HistoryArgs",
    "InputProcessingResult",
    "ParseError",
    "ParsedCommand",
    "ProgressMatch",
    "ProgressType",
    "QueueArgs",
    "RenderResult",
    "RendererConfig",
    "RunArgs",
    "SignalState",
    "SignalTrapConfig",
    "SignalTrapHandler",
    "StatusArgs",
    "TerminalExitReason",
    "TerminalRenderer",
    "Verb",
    "WatchArgs",
    "build_verb_args",
    "classify_structured_command",
    "confirm_ssh_command",
    "create_dispatcher",
    "create_registry",
    "create_signal_handler",
    "detect_progress_pattern",
    "format_detach_message",
    "format_progress_bar",
    "format_spinner_frame",
    "normalize_verb",
    "parse_command",
    "parse_verb",
    "process_input",
    "render_confirmation_display",
    "tokenize",
]
