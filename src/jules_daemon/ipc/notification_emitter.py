"""Notification event emitters for the agent loop and task lifecycle.

Bridges the agent loop result types and SSH run pipeline results to the
notification broadcaster. Each emitter function builds the appropriate
notification payload (``CompletionNotification`` or ``AlertNotification``)
from domain results and broadcasts it through the ``NotificationBroadcaster``.

Two emission paths:

    **Agent loop completion** (``emit_agent_loop_completion``):
        Creates a COMPLETION or ALERT notification from ``AgentLoopResult``.
        COMPLETE state emits a CompletionNotification with exit_status=0.
        ERROR state emits an AlertNotification with the loop's error details.

    **Run pipeline completion** (``emit_run_completion``):
        Creates a COMPLETION notification from ``RunResult``. Always emits
        a CompletionNotification since the run pipeline always produces an
        exit code (even on failure).

    **Alert emission** (``emit_alert``):
        Creates an ALERT notification from explicit parameters. Used by
        background monitoring and anomaly detection.

All emitter functions are async and safe to call without a broadcaster --
they return ``None`` when the broadcaster is ``None`` (the common case
when no CLI subscribers are connected or the feature is disabled).

The ``BroadcastResult`` is returned so callers can inspect delivery
outcomes without coupling to the broadcaster internals.

Usage::

    from jules_daemon.ipc.notification_emitter import (
        emit_agent_loop_completion,
        emit_run_completion,
        emit_alert,
    )

    # After agent loop finishes
    result = await agent_loop.run(user_message)
    await emit_agent_loop_completion(
        broadcaster=broadcaster,
        loop_result=result,
        natural_language_command=user_message,
        run_id="run-abc",
    )

    # After SSH pipeline finishes
    run_result = await execute_run(...)
    await emit_run_completion(
        broadcaster=broadcaster,
        run_result=run_result,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from jules_daemon.ipc.notification_broadcaster import (
    BroadcastResult,
    NotificationBroadcaster,
)
from jules_daemon.protocol.notifications import (
    AlertNotification,
    CompletionNotification,
    NotificationEventType,
    NotificationSeverity,
    create_notification_envelope,
)

__all__ = [
    "emit_agent_loop_completion",
    "emit_alert",
    "emit_run_completion",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent loop completion
# ---------------------------------------------------------------------------


async def emit_agent_loop_completion(
    *,
    broadcaster: NotificationBroadcaster | None,
    loop_result: Any,
    natural_language_command: str,
    run_id: str | None = None,
    resolved_shell: str | None = None,
) -> BroadcastResult | None:
    """Emit a notification after the agent loop completes.

    For COMPLETE state, emits a COMPLETION notification with exit_status=0
    (the agent loop finished its task successfully). For ERROR state, emits
    an ALERT notification with the error details.

    Args:
        broadcaster: The notification broadcaster. When None, no event
            is emitted and None is returned.
        loop_result: An ``AgentLoopResult`` (imported lazily to avoid
            top-level coupling to the agent module).
        natural_language_command: The original user NL command.
        run_id: Optional run ID to associate with the notification.
            When None, a synthetic ID is generated from the loop state.
        resolved_shell: Optional resolved shell command (if known).

    Returns:
        BroadcastResult if an event was emitted, None if the broadcaster
        is None or no event was applicable.
    """
    if broadcaster is None:
        return None

    # Lazy import to avoid coupling at module level
    from jules_daemon.agent.agent_loop import AgentLoopState

    effective_run_id = run_id or f"agent-loop-{id(loop_result):x}"

    if loop_result.final_state is AgentLoopState.COMPLETE:
        payload = CompletionNotification(
            run_id=effective_run_id,
            natural_language_command=natural_language_command,
            resolved_shell=resolved_shell,
            exit_status=0,
        )
        envelope = create_notification_envelope(
            event_type=NotificationEventType.COMPLETION,
            payload=payload,
        )
        logger.debug(
            "Emitting agent loop completion notification: "
            "run_id=%s, iterations=%d",
            effective_run_id,
            loop_result.iterations_used,
        )
        return await broadcaster.broadcast(envelope)

    if loop_result.final_state is AgentLoopState.ERROR:
        error_msg = loop_result.error_message or "Agent loop terminated"

        # Classify severity: user denial is INFO, retries exhausted is
        # WARNING, other errors are ERROR.
        if "denied" in error_msg.lower():
            severity = NotificationSeverity.WARNING
            title = "Agent loop: command denied"
        elif loop_result.retry_exhausted:
            severity = NotificationSeverity.WARNING
            title = "Agent loop: retries exhausted"
        else:
            severity = NotificationSeverity.ERROR
            title = "Agent loop: error"

        payload = AlertNotification(
            run_id=effective_run_id,
            severity=severity,
            title=title,
            message=error_msg,
            details={
                "iterations_used": loop_result.iterations_used,
                "retry_exhausted": loop_result.retry_exhausted,
                "natural_language_command": natural_language_command,
            },
        )
        envelope = create_notification_envelope(
            event_type=NotificationEventType.ALERT,
            payload=payload,
        )
        logger.debug(
            "Emitting agent loop alert notification: "
            "run_id=%s, severity=%s, error=%s",
            effective_run_id,
            severity.value,
            error_msg[:80],
        )
        return await broadcaster.broadcast(envelope)

    # Non-terminal states should not reach here, but be defensive
    logger.warning(
        "emit_agent_loop_completion called with non-terminal state: %s",
        loop_result.final_state.value,
    )
    return None


# ---------------------------------------------------------------------------
# Run pipeline completion
# ---------------------------------------------------------------------------


async def emit_run_completion(
    *,
    broadcaster: NotificationBroadcaster | None,
    run_result: Any,
    natural_language_command: str | None = None,
) -> BroadcastResult | None:
    """Emit a COMPLETION notification after the SSH run pipeline finishes.

    Always emits a CompletionNotification since the run pipeline always
    produces an exit code (even on connection failure, the exit_code is
    None but we use -1 as a sentinel).

    Args:
        broadcaster: The notification broadcaster. When None, no event
            is emitted and None is returned.
        run_result: A ``RunResult`` from the execution pipeline.
        natural_language_command: Optional override for the NL command.
            When None, uses ``run_result.command``.

    Returns:
        BroadcastResult if an event was emitted, None if the broadcaster
        is None.
    """
    if broadcaster is None:
        return None

    nl_command = natural_language_command or run_result.command
    exit_code = run_result.exit_code if run_result.exit_code is not None else -1

    payload = CompletionNotification(
        run_id=run_result.run_id,
        natural_language_command=nl_command,
        resolved_shell=run_result.command,
        exit_status=exit_code,
        duration_seconds=run_result.duration_seconds or None,
        error_message=run_result.error,
    )
    envelope = create_notification_envelope(
        event_type=NotificationEventType.COMPLETION,
        payload=payload,
    )
    logger.debug(
        "Emitting run completion notification: "
        "run_id=%s, success=%s, exit_code=%s",
        run_result.run_id,
        run_result.success,
        exit_code,
    )
    return await broadcaster.broadcast(envelope)


# ---------------------------------------------------------------------------
# Generic alert emission
# ---------------------------------------------------------------------------


async def emit_alert(
    *,
    broadcaster: NotificationBroadcaster | None,
    severity: NotificationSeverity,
    title: str,
    message: str,
    run_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> BroadcastResult | None:
    """Emit an ALERT notification through the broadcaster.

    General-purpose alert emitter for background monitoring, anomaly
    detection, and other daemon-originated events.

    Args:
        broadcaster: The notification broadcaster. When None, no event
            is emitted and None is returned.
        severity: Alert severity level.
        title: Short summary line for the alert.
        message: Detailed description of the alert condition.
        run_id: Optional associated run ID.
        details: Optional machine-readable details dict.

    Returns:
        BroadcastResult if an event was emitted, None if the broadcaster
        is None.
    """
    if broadcaster is None:
        return None

    payload = AlertNotification(
        run_id=run_id,
        severity=severity,
        title=title,
        message=message,
        details=details,
    )
    envelope = create_notification_envelope(
        event_type=NotificationEventType.ALERT,
        payload=payload,
    )
    logger.debug(
        "Emitting alert notification: severity=%s, title=%s, run_id=%s",
        severity.value,
        title,
        run_id,
    )
    return await broadcaster.broadcast(envelope)
