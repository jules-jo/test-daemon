"""Resource cleanup package for deterministic teardown on disconnect.

Provides handlers that release socket connections, close SSH channels,
and flush pending I/O buffers when disconnect events are detected.

Modules:
    resource_types: Protocols and immutable data types for cleanup operations.
    disconnect_handler: Event-driven cleanup coordinator.
    channel_guard: Context manager for SSH channel lifecycle.
"""
