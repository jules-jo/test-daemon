"""SSH connectivity layer for remote test execution.

Modules:
    command: Validated Pydantic model for SSH commands
    credentials: SSH credential resolution (password auth support)
    reader: Non-blocking async SSH output reader
    buffer_reader: SSH output buffer reader for stale/interrupted sessions
    errors: SSH error hierarchy (transient vs permanent classification)
    backoff: Configurable exponential backoff delay calculation
    reconnect: SSH reconnection orchestrator with retry logic
    executor: SSH command executor with 3-retry exponential backoff
    reestablish: SSH re-establishment from recovered wiki run records
    liveness: SSH connection liveness validator (established session)
    endpoint_probe: TCP/SSH-banner reachability checker (pre-connection)
    pid_liveness: Remote PID liveness validation (kill -0 + /proc fallback)
    reattach: Process output re-attachment (probe + stream output lines)
    execution_audit: Audit instrumentation for SSH dispatch (command, host, outcome)
"""
