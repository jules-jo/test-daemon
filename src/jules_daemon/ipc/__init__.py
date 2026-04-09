"""IPC (Inter-Process Communication) package for CLI-daemon messaging.

Provides twelve layers:

    framing               -- wire format (length-prefixed JSON envelopes)
    event_bus             -- async publish/subscribe event routing
    connection_manager    -- client registry with lifecycle event emission
    connection_dispatcher -- concurrent connection dispatcher with explicit
                            task spawning, semaphore-based concurrency limits,
                            and ConnectionManager integration
    session_models        -- immutable data models for per-client session tracking
    session_registry      -- per-client session registry with lifecycle management,
                            unique session IDs, metadata tracking, and state transitions
    client_io             -- per-client read/write coroutines with framing integration
    server                -- async Unix domain socket server with lifecycle management
    socket_discovery      -- daemon socket path discovery (env, XDG, tmpdir)
    client_connection     -- client-side connection with handshake protocol
    watch_client          -- client-side watch command for real-time output streaming
    request_validator     -- request payload validation (verb, fields, constraints)
    request_handler       -- concrete ClientHandler bridging validation and dispatch
"""
