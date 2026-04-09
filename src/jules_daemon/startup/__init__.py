"""Daemon startup lifecycle management.

Modules:
    scan_probe_mark: Three-phase pipeline that scans wiki sessions, probes
        their liveness (PID + endpoint), and marks stale ones before the
        daemon accepts commands.
    lifecycle: Startup lifecycle orchestrator that runs initialization hooks
        (including scan-probe-mark) and transitions the daemon to READY.
    readiness_gate: Thread-safe gate that blocks new test-run requests until
        the startup scan-probe-mark pipeline has completed, returning a
        structured not-ready response for early requests.
    stale_session_logger: Structured logging for stale sessions detected
        during the startup scan. Emits WARNING-level log records with a
        defined schema (session_id, host, last_activity_timestamp,
        staleness_reason) for each stale session.
    collision_detector: Discovers existing daemon processes by scanning the
        OS process table and wiki active sessions. Returns structured info
        (PID, command, start time, duration) for warn-and-allow collision
        detection.
    collision_prompt: Formats collision details (PID, command, duration) as
        a terminal warning and presents an interactive prompt with three
        choices: Proceed (warn-and-allow), Abort, or Force-replace.
"""
