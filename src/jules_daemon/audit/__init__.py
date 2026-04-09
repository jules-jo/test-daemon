"""Audit trail for command execution pipeline.

Modules:
    models: Immutable data models for the full-chain audit record
    chain: Generic AuditEntry and immutable AuditChain accumulator
    snapshot: Pipeline stage state capture into immutable snapshots
    instrumentation: Decorator and context manager for stage audit tracking
"""
