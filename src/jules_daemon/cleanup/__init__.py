"""Resource cleanup package for deterministic teardown on disconnect.

Provides handlers that release socket connections, close SSH channels,
flush pending I/O buffers, and clean up notification subscribers when
disconnect events are detected.

Modules:
    resource_types: Protocols and immutable data types for cleanup operations.
    disconnect_handler: Event-driven cleanup coordinator.
    channel_guard: Context manager for SSH channel lifecycle.
    subscriber_cleanup: Subscriber cleanup for notification broadcaster resources.
    subscriber_sweep: Periodic background sweep for orphaned/stale subscribers.
"""
