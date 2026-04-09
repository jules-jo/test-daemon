"""Monitoring layer for autonomous SSH session observation.

Modules:
    polling_loop: Async polling loop that reads SSH output at configurable
        intervals and feeds chunks to MonitorStatus update methods
    monitor_transition: Transition logic that bridges fast-forward
        resumption and the live polling loop, replaying gap events
        and re-registering callbacks from the reconciled state
    process_state: Local process-state checker that probes the OS process
        table via os.kill(pid, 0) to determine alive/dead status
    session_liveness: Per-session liveness aggregator that combines the
        process-state verdict and SSH-connection verdict into a unified
        LivenessResult
    stale_session_detector: Heartbeat/keepalive-based staleness detector
        that evaluates SSH sessions against configurable timeout
        thresholds and transport-layer status checks
    output_broadcaster: Server-side fan-out broadcaster that buffers recent
        SSH output lines per job and delivers them to multiple async
        subscriber queues
    formatting_pipeline: Output formatting pipeline that processes raw
        SSH output chunks by preserving/normalizing/stripping ANSI codes
        and prepending configurable timestamps to each line
    queue_consumer: Background asyncio task that drains enqueued commands
        from the AsyncCommandQueue after the current run completes,
        without interrupting or blocking active runs
"""

from jules_daemon.monitor.monitor_transition import (
    TransitionConfig,
    TransitionOutcome,
    TransitionPhase,
    TransitionResult,
    build_initial_status,
    prepare_transition,
    replay_gap_events,
)
from jules_daemon.monitor.polling_loop import (
    PollingConfig,
    PollingLoop,
    PollingState,
    StatusCallback,
)
from jules_daemon.monitor.process_state import (
    ProcessCheckResult,
    ProcessVerdict,
    check_pid,
    check_pids,
)
from jules_daemon.monitor.session_liveness import (
    LivenessResult,
    SessionHealth,
    SessionInfo,
    check_session_liveness,
)
from jules_daemon.monitor.formatting_pipeline import (
    AnsiMode,
    FormattedChunk,
    FormatterConfig,
    format_chunk,
    normalize_ansi,
    prepend_timestamps,
    strip_ansi,
)
from jules_daemon.monitor.output_broadcaster import (
    BroadcasterConfig,
    JobOutputBroadcaster,
    OutputLine,
    SubscriberHandle,
)
from jules_daemon.monitor.queue_consumer import (
    AsyncCommandHandler,
    ConsumerConfig,
    ConsumerState,
    DrainResult,
    QueueConsumer,
)
from jules_daemon.monitor.stale_session_detector import (
    DetectorConfig,
    HeartbeatRecord,
    HeartbeatTracker,
    StalenessDetection,
    StalenessReason,
    detect_batch_staleness,
    detect_session_staleness,
)

__all__ = [
    "AnsiMode",
    "AsyncCommandHandler",
    "BroadcasterConfig",
    "ConsumerConfig",
    "ConsumerState",
    "FormattedChunk",
    "FormatterConfig",
    "DrainResult",
    "DetectorConfig",
    "HeartbeatRecord",
    "HeartbeatTracker",
    "JobOutputBroadcaster",
    "LivenessResult",
    "OutputLine",
    "PollingConfig",
    "PollingLoop",
    "PollingState",
    "QueueConsumer",
    "ProcessCheckResult",
    "ProcessVerdict",
    "SessionHealth",
    "SessionInfo",
    "StalenessDetection",
    "StalenessReason",
    "StatusCallback",
    "SubscriberHandle",
    "TransitionConfig",
    "TransitionOutcome",
    "TransitionPhase",
    "TransitionResult",
    "build_initial_status",
    "check_pid",
    "check_pids",
    "format_chunk",
    "check_session_liveness",
    "detect_batch_staleness",
    "detect_session_staleness",
    "normalize_ansi",
    "prepend_timestamps",
    "prepare_transition",
    "replay_gap_events",
    "strip_ansi",
]
