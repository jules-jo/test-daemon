"""Monitoring layer for autonomous SSH session observation.

Modules:
    anomaly_models: Anomaly pattern configuration schema and base detector
        interface. Defines immutable data models for error keyword patterns
        (regex-based), failure-rate thresholds (count/window), and stall
        timeouts (elapsed-time). Provides the AnomalyDetector protocol
        that concrete detectors implement.
    error_keyword_detector: Concrete implementation of the AnomalyDetector
        protocol for regex-based error keyword matching. Scans SSH output
        lines against configurable ErrorKeywordPattern instances.
    failure_rate_spike_detector: Concrete implementation of the
        AnomalyDetector protocol for failure-rate spike detection. Uses
        a sliding window counter to detect spikes in failure rates over
        a configurable time window.
    stall_hang_detector: Concrete implementation of the AnomalyDetector
        protocol for stall/hang detection. Tracks elapsed time since
        last SSH output and flags when a configurable timeout threshold
        is exceeded.
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
    detector_registry: Centralized registry for AnomalyDetector instances
        with register/unregister pattern. Thread-safe, keyed by
        pattern_name, provides snapshot-based listing.
    detector_dispatcher: Fan-out dispatcher that sends each output line
        to all registered detectors concurrently via asyncio.gather
        and asyncio.to_thread. Collects anomaly reports and captures
        detector errors without crashing.
    alert_dedup: Alert deduplication and priority-scoring processor.
        Sits between DetectorDispatcher output and AlertCollector input
        to suppress duplicate alerts within a configurable time window
        and assign priority scores based on severity, pattern type,
        and occurrence frequency.
    alert_query: Query API for filtered and prioritized alert retrieval.
        Provides rich multi-criteria filtering (severity, status, pattern
        type, session, time range), priority-based sorting, result
        limiting, and agent-friendly output formats. Designed for the
        agent loop to consume alerts during think-act cycles.
"""

from jules_daemon.monitor.anomaly_models import (
    AnomalyDetector,
    AnomalyPatternConfig,
    AnomalyReport,
    AnomalySeverity,
    ErrorKeywordPattern,
    FailureRatePattern,
    PatternType,
    StallTimeoutPattern,
)
from jules_daemon.monitor.error_keyword_detector import ErrorKeywordDetector
from jules_daemon.monitor.failure_rate_spike_detector import (
    FailureRateSpikeDetector,
)
from jules_daemon.monitor.stall_hang_detector import StallHangDetector
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
from jules_daemon.monitor.detector_registry import DetectorRegistry
from jules_daemon.monitor.detector_dispatcher import (
    DetectorDispatcher,
    DetectorError,
    DispatchResult,
)
from jules_daemon.monitor.alert_dedup import (
    AlertProcessor,
    AlertProcessorConfig,
    DeduplicationKey,
    PriorityScore,
    ProcessedAlert,
    ProcessingResult,
    compute_dedup_key,
    compute_priority_score,
)
from jules_daemon.monitor.alert_query import (
    AlertQuery,
    AlertQueryResult,
    AlertQueryService,
    AlertSortOrder,
    QueryableAlert,
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
    "AlertQuery",
    "AlertQueryResult",
    "AlertQueryService",
    "AlertProcessor",
    "AlertProcessorConfig",
    "AlertSortOrder",
    "AnomalyDetector",
    "AnomalyPatternConfig",
    "AnomalyReport",
    "AnomalySeverity",
    "AnsiMode",
    "AsyncCommandHandler",
    "BroadcasterConfig",
    "ConsumerConfig",
    "ConsumerState",
    "DeduplicationKey",
    "DetectorDispatcher",
    "DetectorError",
    "DetectorRegistry",
    "DispatchResult",
    "FormattedChunk",
    "FormatterConfig",
    "DrainResult",
    "DetectorConfig",
    "ErrorKeywordDetector",
    "ErrorKeywordPattern",
    "FailureRatePattern",
    "FailureRateSpikeDetector",
    "HeartbeatRecord",
    "HeartbeatTracker",
    "JobOutputBroadcaster",
    "LivenessResult",
    "OutputLine",
    "PatternType",
    "PollingConfig",
    "PriorityScore",
    "ProcessedAlert",
    "ProcessingResult",
    "PollingLoop",
    "PollingState",
    "QueryableAlert",
    "QueueConsumer",
    "ProcessCheckResult",
    "ProcessVerdict",
    "SessionHealth",
    "SessionInfo",
    "StallHangDetector",
    "StallTimeoutPattern",
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
    "compute_dedup_key",
    "compute_priority_score",
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
