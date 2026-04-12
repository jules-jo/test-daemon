"""Notification event types and streaming protocol for daemon push events.

Defines the shared data models and constants used by both the daemon
(sender) and CLI (subscriber) for the persistent notification channel.
This channel delivers real-time events independently of the
request-response IPC cycle.

Event Types
-----------

Three notification event types cover the daemon's push scenarios:

    ``COMPLETION``  -- A test run finished (success or failure). Carries
                       the run ID, exit status, duration, and an optional
                       test outcome summary.
    ``ALERT``       -- An anomaly or warning detected by the daemon's
                       background monitoring (e.g., high failure rate,
                       pattern match on SIGSEGV, SSH disconnect).
    ``HEARTBEAT``   -- Periodic liveness signal so the CLI can detect
                       stale connections. Carries daemon uptime and
                       current run state.

Streaming Protocol
------------------

The notification channel uses a subscribe/unsubscribe handshake over the
existing IPC framed transport. Once subscribed, the daemon pushes
``NotificationEnvelope`` messages to the client. The envelope carries:

    - ``channel_version``: Semver for the notification channel (independent
      of the main IPC protocol version).
    - ``event_id``: Unique identifier per event (for dedup and ordering).
    - ``timestamp``: UTC-aware datetime.
    - ``event_type``: Discriminates which payload model to expect.
    - ``payload``: One of CompletionNotification, AlertNotification,
      or HeartbeatNotification.

This design mirrors SSE semantics (event type + payload + id) while
running over the existing length-prefixed IPC transport rather than HTTP.

All models are frozen (immutable) via Pydantic's ``ConfigDict(frozen=True)``
to match the project-wide immutability convention.

Usage::

    from jules_daemon.protocol.notifications import (
        AlertNotification,
        CompletionNotification,
        HeartbeatNotification,
        NotificationEventType,
        NotificationSeverity,
        create_notification_envelope,
    )

    # Build a completion event
    payload = CompletionNotification(
        run_id="run-abc",
        natural_language_command="Run pytest on auth",
        exit_status=0,
        outcome=TestOutcomeSummary(tests_passed=95, tests_total=100),
    )
    envelope = create_notification_envelope(
        event_type=NotificationEventType.COMPLETION,
        payload=payload,
    )
    wire_json = envelope.model_dump_json()
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

__all__ = [
    "HEARTBEAT_DEFAULT_INTERVAL_SECONDS",
    "NOTIFICATION_CHANNEL_VERSION",
    "AlertNotification",
    "CompletionNotification",
    "HeartbeatNotification",
    "NotificationEnvelope",
    "NotificationEventType",
    "NotificationPayloadType",
    "NotificationSeverity",
    "SubscribeRequest",
    "SubscribeResponse",
    "TestOutcomeSummary",
    "UnsubscribeRequest",
    "UnsubscribeResponse",
    "create_notification_envelope",
    "parse_notification_event_type",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOTIFICATION_CHANNEL_VERSION: str = "1.0.0"
"""Semver version for the notification channel protocol.

Independent of the main IPC protocol version. Bumped when the
notification envelope format or event payload schemas change.
"""

HEARTBEAT_DEFAULT_INTERVAL_SECONDS: int = 30
"""Default interval between heartbeat events (in seconds).

The daemon sends a heartbeat at this interval to each subscriber
so the CLI can detect stale connections. Configurable at the daemon
level; this constant serves as the default and as documentation.
"""


# ---------------------------------------------------------------------------
# Base configuration -- internal, not exported
# ---------------------------------------------------------------------------


class _FrozenModel(BaseModel):
    """Base for all notification models -- frozen to enforce immutability.

    Matches the convention established in ``protocol.schemas``.
    """

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# NotificationEventType enum
# ---------------------------------------------------------------------------


class NotificationEventType(Enum):
    """Categories for notification events pushed by the daemon.

    Values:
        COMPLETION: A test run finished (success or failure).
        ALERT:      An anomaly or warning from background monitoring.
        HEARTBEAT:  Periodic liveness signal from the daemon.
    """

    COMPLETION = "completion"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"


# Lookup table: lowered value -> NotificationEventType
_EVENT_TYPE_LOOKUP: dict[str, NotificationEventType] = {
    et.value: et for et in NotificationEventType
}


def parse_notification_event_type(raw: str) -> NotificationEventType:
    """Parse a raw string into a NotificationEventType enum member.

    Matching is case-insensitive with leading/trailing whitespace stripped.

    Args:
        raw: Wire-format event type string (e.g., ``"completion"``).

    Returns:
        The matching ``NotificationEventType`` member.

    Raises:
        ValueError: If the string is empty or does not match any type.
    """
    normalized = raw.strip().lower()
    if not normalized:
        raise ValueError("Notification event type must not be empty")

    event_type = _EVENT_TYPE_LOOKUP.get(normalized)
    if event_type is None:
        valid = ", ".join(sorted(_EVENT_TYPE_LOOKUP))
        raise ValueError(
            f"Unknown notification event type {raw.strip()!r}. "
            f"Valid types: {valid}"
        )
    return event_type


# ---------------------------------------------------------------------------
# NotificationSeverity enum
# ---------------------------------------------------------------------------


class NotificationSeverity(Enum):
    """Severity level for alert notifications.

    Used by AlertNotification to classify the urgency of the alert.
    Maps to the same severity levels used by the notify_user tool.

    Values:
        INFO:    Informational message (no action required).
        WARNING: Potential issue that may need attention.
        ERROR:   Error condition that likely requires action.
        SUCCESS: Positive outcome notification.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


# ---------------------------------------------------------------------------
# TestOutcomeSummary model
# ---------------------------------------------------------------------------


class TestOutcomeSummary(_FrozenModel):
    """Summary of test execution results.

    Carried as an optional field in CompletionNotification when the
    daemon has parsed the test output and extracted counts.

    Fields:
        tests_passed:  Number of tests that passed.
        tests_failed:  Number of tests that failed.
        tests_skipped: Number of tests skipped.
        tests_total:   Total number of tests discovered.
    """

    tests_passed: int = Field(default=0, ge=0)
    tests_failed: int = Field(default=0, ge=0)
    tests_skipped: int = Field(default=0, ge=0)
    tests_total: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# CompletionNotification model
# ---------------------------------------------------------------------------


class CompletionNotification(_FrozenModel):
    """Notification payload for a completed test run.

    Sent when a test run finishes, regardless of success or failure.
    Carries enough context for the CLI to display a meaningful summary
    without querying the daemon for status.

    Fields:
        event_type: Literal discriminator for payload routing.
        run_id: Unique identifier of the completed run.
        natural_language_command: The original user command.
        resolved_shell: The shell command that was executed (if approved).
        exit_status: Remote process exit code.
        duration_seconds: Wall-clock execution time (if available).
        outcome: Parsed test result summary (if available).
        error_message: Error description (if the run failed before
            producing test results).
    """

    event_type: Literal["completion"] = "completion"
    run_id: str = Field(..., min_length=1)
    natural_language_command: str = Field(..., min_length=1)
    resolved_shell: str | None = None
    exit_status: int
    duration_seconds: float | None = None
    outcome: TestOutcomeSummary | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# AlertNotification model
# ---------------------------------------------------------------------------


class AlertNotification(_FrozenModel):
    """Notification payload for an anomaly or warning alert.

    Sent by the daemon's background monitoring when a pattern match,
    anomaly, or exceptional condition is detected. The CLI can display
    these in real time to keep the user informed.

    Fields:
        event_type: Literal discriminator for payload routing.
        run_id: Associated run ID (None if not run-specific).
        severity: Urgency classification (info, warning, error, success).
        title: Short summary line for the alert.
        message: Detailed description of the alert condition.
        details: Optional machine-readable details (pattern matches,
            counts, thresholds, etc.). Values must be JSON-serializable.
    """

    event_type: Literal["alert"] = "alert"
    run_id: str | None = None
    severity: NotificationSeverity
    title: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    details: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# HeartbeatNotification model
# ---------------------------------------------------------------------------


class HeartbeatNotification(_FrozenModel):
    """Notification payload for periodic daemon liveness heartbeats.

    Sent at a configurable interval (default: HEARTBEAT_DEFAULT_INTERVAL_SECONDS)
    to all subscribers so the CLI can detect stale connections. Also
    carries lightweight daemon state for display purposes.

    Fields:
        event_type: Literal discriminator for payload routing.
        daemon_uptime_seconds: How long the daemon has been running.
        active_run_id: Current active run (None if idle).
        queue_depth: Number of commands waiting in the queue.
    """

    event_type: Literal["heartbeat"] = "heartbeat"
    daemon_uptime_seconds: float = Field(..., ge=0.0)
    active_run_id: str | None = None
    queue_depth: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Discriminated union of notification payloads
# ---------------------------------------------------------------------------

NotificationPayloadType = Annotated[
    Union[
        CompletionNotification,
        AlertNotification,
        HeartbeatNotification,
    ],
    Field(discriminator="event_type"),
]
"""Discriminated union of all notification payload types.

Keyed on the ``event_type`` literal field for unambiguous deserialization.
"""


# ---------------------------------------------------------------------------
# NotificationEnvelope model
# ---------------------------------------------------------------------------


class NotificationEnvelope(_FrozenModel):
    """Envelope wrapping every notification event pushed to subscribers.

    Mirrors SSE semantics (event type + data + id) while using the
    existing length-prefixed IPC transport. The envelope provides
    metadata for dedup, ordering, and version negotiation independently
    of the payload content.

    Fields:
        channel_version: Semver for the notification channel protocol.
        event_id: Unique identifier for this event (UUID-based).
            Enables client-side deduplication and resume-from.
        timestamp: UTC-aware datetime when the event was created.
        event_type: Category of the notification (completion, alert,
            heartbeat). Used for routing and filtering.
        payload: The typed notification payload (discriminated by
            event_type).
    """

    channel_version: str = Field(..., min_length=1)
    event_id: str = Field(..., min_length=1)
    timestamp: AwareDatetime
    event_type: NotificationEventType
    payload: NotificationPayloadType


# ---------------------------------------------------------------------------
# Subscription protocol models
# ---------------------------------------------------------------------------


class SubscribeRequest(_FrozenModel):
    """CLI -> Daemon: Subscribe to the notification channel.

    Starts a persistent subscription. The daemon will push
    NotificationEnvelope messages for matching event types.

    Fields:
        payload_type: Discriminator literal for IPC payload routing.
        event_filter: Optional tuple of event types to receive. When
            None, all event types are delivered (default).
    """

    payload_type: Literal["subscribe_notifications"] = "subscribe_notifications"
    event_filter: tuple[NotificationEventType, ...] | None = None


class SubscribeResponse(_FrozenModel):
    """Daemon -> CLI: Acknowledge a notification subscription.

    Confirms the subscription and provides the subscription ID for
    later unsubscription. Also communicates the heartbeat interval
    so the CLI knows how often to expect liveness signals.

    Fields:
        payload_type: Discriminator literal for IPC payload routing.
        subscription_id: Unique identifier for this subscription.
            Used in UnsubscribeRequest to remove the subscription.
        heartbeat_interval_seconds: How often the daemon will send
            heartbeat events (in seconds).
    """

    payload_type: Literal["subscribe_notifications_response"] = (
        "subscribe_notifications_response"
    )
    subscription_id: str = Field(..., min_length=1)
    heartbeat_interval_seconds: int = Field(..., ge=1)


class UnsubscribeRequest(_FrozenModel):
    """CLI -> Daemon: Remove a notification subscription.

    Stops the persistent subscription identified by subscription_id.
    The daemon will stop pushing events to this subscriber.

    Fields:
        payload_type: Discriminator literal for IPC payload routing.
        subscription_id: The subscription to remove (from SubscribeResponse).
    """

    payload_type: Literal["unsubscribe_notifications"] = "unsubscribe_notifications"
    subscription_id: str = Field(..., min_length=1)


class UnsubscribeResponse(_FrozenModel):
    """Daemon -> CLI: Acknowledge a notification unsubscription.

    Confirms that the subscription has been removed. The daemon
    will no longer push events for this subscriber.

    Fields:
        payload_type: Discriminator literal for IPC payload routing.
        subscription_id: The subscription that was removed.
    """

    payload_type: Literal["unsubscribe_notifications_response"] = (
        "unsubscribe_notifications_response"
    )
    subscription_id: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def create_notification_envelope(
    *,
    event_type: NotificationEventType,
    payload: NotificationPayloadType,
    event_id: str | None = None,
) -> NotificationEnvelope:
    """Create a NotificationEnvelope with auto-populated metadata.

    Generates a UUID-based event_id and UTC timestamp automatically.

    Args:
        event_type: The category of notification event.
        payload: The typed notification payload model.
        event_id: Optional override for the event ID.

    Returns:
        A fully populated, immutable NotificationEnvelope.
    """
    return NotificationEnvelope(
        channel_version=NOTIFICATION_CHANNEL_VERSION,
        event_id=event_id or f"evt-{uuid.uuid4().hex[:16]}",
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        payload=payload,
    )
