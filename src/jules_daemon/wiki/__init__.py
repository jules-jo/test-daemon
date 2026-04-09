"""Wiki persistence layer for daemon state.

Modules:
    frontmatter: YAML frontmatter parser/serializer for wiki files
    models: Immutable data models for run state
    layout: Wiki directory layout manager -- central registry for all paths
    current_run: Read/write current-run wiki file
    boot_reader: Load and extract status fields on daemon boot
    interrupted_run: Detect interrupted runs and return recovery verdicts
    state_reader: Load and extract connection params + run metadata for reconnection
    crash_recovery: Unified crash recovery detection at daemon startup
    checkpoint_extractor: Extract last completed checkpoint from wiki state
    checkpoint_recovery: Monitoring checkpoint recovery and persistence
    resume_decision: Resume-or-restart decision logic for interrupted test runs
    command_translation: Store NL-to-command mappings as wiki pages for learning
    session_scanner: Discover and parse all session entries for liveness evaluation
    stale_session_marker: Mark non-live sessions as stale with immutable write-new-file semantics
    output_fast_forward: Fast-forward scanner for SSH stream re-attachment after crash recovery
    run_promotion: Promote terminal current-run records to completed run history
    test_result_writer: Persist AssembledTestResult as Karpathy-style wiki entries
    audit_writer: Persist completed audit chains as wiki pages with full traceability
    audit_age_scanner: Identify aged audit-log entries for archival candidates
    audit_archiver: Move approved audit entries to archive with updated frontmatter
    archival_approval: User approval prompt flow for audit log archival
    queue_models: Immutable data models for the command queue (QueuedCommand, QueueStatus, QueuePriority)
    command_queue: Thread-safe command queue with wiki-backed persistence
    async_queue: Async wrapper with non-blocking put/get for the daemon event loop
    watch_session_models: Immutable data models for watch-mode session metadata
        (WatcherRecord, StreamRecord, WatchSessionSnapshot)
    watch_session: Persist watch-mode session metadata (active watchers, stream state)
        to a wiki file with YAML frontmatter + markdown body
    monitor_status: Timestamped immutable status model for SSH monitoring sessions
    connection_status: Wiki persistence for SSH connection liveness status
    staleness_guard: Staleness guard enforcing 10s freshness threshold on status data
    path_router: Fresh-start vs recovery path router for daemon boot
    recovery_log: Wiki persistence for recovery outcome logging
    recovery_orchestrator: 30-second recovery timeout orchestrator for daemon crash recovery
    resumption_reconciler: Resumption state reconciliation for crash recovery reconnection
    assembled_result: Structured result dataclass for assembled test execution outcomes
    partial_result_assembler: Partial result assembler for interrupted or chunked test output
    session_persistence: Session state persistence to wiki on unexpected disconnect
    session_recovery: Session recovery detection and resume/discard offer handling
"""
