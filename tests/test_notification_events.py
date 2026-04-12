"""Tests for notification event types and streaming protocol models.

Covers NotificationEventType enum, NotificationSeverity enum, all
notification payload models (completion, alert, heartbeat), the
subscription protocol models, and serialization round-trips.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from jules_daemon.protocol.notifications import (
    HEARTBEAT_DEFAULT_INTERVAL_SECONDS,
    NOTIFICATION_CHANNEL_VERSION,
    AlertNotification,
    CompletionNotification,
    HeartbeatNotification,
    NotificationEnvelope,
    NotificationEventType,
    NotificationSeverity,
    SubscribeRequest,
    SubscribeResponse,
    TestOutcomeSummary,
    UnsubscribeRequest,
    UnsubscribeResponse,
    create_notification_envelope,
    parse_notification_event_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)
_RUN_ID = "run-abc-123"
_SUB_ID = "sub-xyz-789"


# ---------------------------------------------------------------------------
# NotificationEventType enum
# ---------------------------------------------------------------------------


class TestNotificationEventType:
    """NotificationEventType enum tests."""

    def test_completion_value(self) -> None:
        assert NotificationEventType.COMPLETION.value == "completion"

    def test_alert_value(self) -> None:
        assert NotificationEventType.ALERT.value == "alert"

    def test_heartbeat_value(self) -> None:
        assert NotificationEventType.HEARTBEAT.value == "heartbeat"

    def test_all_members_have_unique_values(self) -> None:
        values = [m.value for m in NotificationEventType]
        assert len(values) == len(set(values))

    def test_member_count(self) -> None:
        assert len(NotificationEventType) == 3


class TestParseNotificationEventType:
    """Tests for parse_notification_event_type helper."""

    def test_parse_completion(self) -> None:
        result = parse_notification_event_type("completion")
        assert result is NotificationEventType.COMPLETION

    def test_parse_alert(self) -> None:
        result = parse_notification_event_type("alert")
        assert result is NotificationEventType.ALERT

    def test_parse_heartbeat(self) -> None:
        result = parse_notification_event_type("heartbeat")
        assert result is NotificationEventType.HEARTBEAT

    def test_parse_case_insensitive(self) -> None:
        assert parse_notification_event_type("COMPLETION") is NotificationEventType.COMPLETION
        assert parse_notification_event_type("Alert") is NotificationEventType.ALERT
        assert parse_notification_event_type("HEARTBEAT") is NotificationEventType.HEARTBEAT

    def test_parse_with_whitespace(self) -> None:
        assert parse_notification_event_type("  completion  ") is NotificationEventType.COMPLETION

    def test_parse_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            parse_notification_event_type("")

    def test_parse_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown notification event type"):
            parse_notification_event_type("unknown_type")


# ---------------------------------------------------------------------------
# NotificationSeverity enum
# ---------------------------------------------------------------------------


class TestNotificationSeverity:
    """NotificationSeverity enum tests."""

    def test_info_value(self) -> None:
        assert NotificationSeverity.INFO.value == "info"

    def test_warning_value(self) -> None:
        assert NotificationSeverity.WARNING.value == "warning"

    def test_error_value(self) -> None:
        assert NotificationSeverity.ERROR.value == "error"

    def test_success_value(self) -> None:
        assert NotificationSeverity.SUCCESS.value == "success"

    def test_member_count(self) -> None:
        assert len(NotificationSeverity) == 4


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Notification channel constants."""

    def test_channel_version_is_string(self) -> None:
        assert isinstance(NOTIFICATION_CHANNEL_VERSION, str)

    def test_channel_version_semver_format(self) -> None:
        parts = NOTIFICATION_CHANNEL_VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_heartbeat_interval_positive(self) -> None:
        assert HEARTBEAT_DEFAULT_INTERVAL_SECONDS > 0

    def test_heartbeat_interval_is_int(self) -> None:
        assert isinstance(HEARTBEAT_DEFAULT_INTERVAL_SECONDS, int)


# ---------------------------------------------------------------------------
# TestOutcomeSummary model
# ---------------------------------------------------------------------------


class TestTestOutcomeSummary:
    """TestOutcomeSummary model tests."""

    def test_valid_construction(self) -> None:
        summary = TestOutcomeSummary(
            tests_passed=90,
            tests_failed=5,
            tests_skipped=3,
            tests_total=98,
        )
        assert summary.tests_passed == 90
        assert summary.tests_failed == 5
        assert summary.tests_skipped == 3
        assert summary.tests_total == 98

    def test_defaults_to_zero(self) -> None:
        summary = TestOutcomeSummary()
        assert summary.tests_passed == 0
        assert summary.tests_failed == 0
        assert summary.tests_skipped == 0
        assert summary.tests_total == 0

    def test_immutable(self) -> None:
        summary = TestOutcomeSummary(tests_passed=10)
        with pytest.raises(ValidationError):
            summary.tests_passed = 20  # type: ignore[misc]

    def test_negative_values_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TestOutcomeSummary(tests_passed=-1)

    def test_serialization_round_trip(self) -> None:
        summary = TestOutcomeSummary(
            tests_passed=10,
            tests_failed=2,
            tests_skipped=1,
            tests_total=13,
        )
        data = summary.model_dump()
        restored = TestOutcomeSummary.model_validate(data)
        assert restored == summary


# ---------------------------------------------------------------------------
# CompletionNotification model
# ---------------------------------------------------------------------------


class TestCompletionNotification:
    """CompletionNotification model tests."""

    def test_valid_construction(self) -> None:
        notification = CompletionNotification(
            run_id=_RUN_ID,
            natural_language_command="Run pytest on auth",
            resolved_shell="pytest tests/auth/",
            exit_status=0,
            duration_seconds=45.2,
            outcome=TestOutcomeSummary(
                tests_passed=95,
                tests_failed=5,
                tests_total=100,
            ),
        )
        assert notification.event_type == "completion"
        assert notification.run_id == _RUN_ID
        assert notification.exit_status == 0
        assert notification.outcome.tests_passed == 95

    def test_event_type_literal(self) -> None:
        notification = CompletionNotification(
            run_id=_RUN_ID,
            natural_language_command="test",
            exit_status=0,
        )
        assert notification.event_type == "completion"

    def test_immutable(self) -> None:
        notification = CompletionNotification(
            run_id=_RUN_ID,
            natural_language_command="test",
            exit_status=0,
        )
        with pytest.raises(ValidationError):
            notification.exit_status = 1  # type: ignore[misc]

    def test_optional_fields_default_none(self) -> None:
        notification = CompletionNotification(
            run_id=_RUN_ID,
            natural_language_command="test",
            exit_status=0,
        )
        assert notification.resolved_shell is None
        assert notification.duration_seconds is None
        assert notification.outcome is None
        assert notification.error_message is None

    def test_run_id_required(self) -> None:
        with pytest.raises(ValidationError):
            CompletionNotification(
                run_id="",
                natural_language_command="test",
                exit_status=0,
            )

    def test_serialization_round_trip(self) -> None:
        notification = CompletionNotification(
            run_id=_RUN_ID,
            natural_language_command="Run pytest",
            resolved_shell="pytest",
            exit_status=0,
            duration_seconds=12.3,
            outcome=TestOutcomeSummary(tests_passed=10, tests_total=10),
        )
        data = notification.model_dump()
        restored = CompletionNotification.model_validate(data)
        assert restored == notification

    def test_json_round_trip(self) -> None:
        notification = CompletionNotification(
            run_id=_RUN_ID,
            natural_language_command="Run pytest",
            exit_status=1,
            error_message="SSH auth failed",
        )
        json_str = notification.model_dump_json()
        restored = CompletionNotification.model_validate_json(json_str)
        assert restored == notification


# ---------------------------------------------------------------------------
# AlertNotification model
# ---------------------------------------------------------------------------


class TestAlertNotification:
    """AlertNotification model tests."""

    def test_valid_construction(self) -> None:
        alert = AlertNotification(
            run_id=_RUN_ID,
            severity=NotificationSeverity.WARNING,
            title="High failure rate",
            message="50% of tests are failing",
        )
        assert alert.event_type == "alert"
        assert alert.severity is NotificationSeverity.WARNING
        assert alert.title == "High failure rate"

    def test_event_type_literal(self) -> None:
        alert = AlertNotification(
            severity=NotificationSeverity.INFO,
            title="Info alert",
            message="Test started",
        )
        assert alert.event_type == "alert"

    def test_run_id_optional(self) -> None:
        alert = AlertNotification(
            severity=NotificationSeverity.ERROR,
            title="Connection lost",
            message="SSH dropped",
        )
        assert alert.run_id is None

    def test_immutable(self) -> None:
        alert = AlertNotification(
            severity=NotificationSeverity.INFO,
            title="Test",
            message="Test message",
        )
        with pytest.raises(ValidationError):
            alert.title = "Changed"  # type: ignore[misc]

    def test_title_required(self) -> None:
        with pytest.raises(ValidationError):
            AlertNotification(
                severity=NotificationSeverity.INFO,
                title="",
                message="test",
            )

    def test_message_required(self) -> None:
        with pytest.raises(ValidationError):
            AlertNotification(
                severity=NotificationSeverity.INFO,
                title="test",
                message="",
            )

    def test_details_optional(self) -> None:
        alert = AlertNotification(
            severity=NotificationSeverity.WARNING,
            title="Anomaly",
            message="Pattern detected",
            details={"pattern": "SIGSEGV", "count": 3},
        )
        assert alert.details == {"pattern": "SIGSEGV", "count": 3}

    def test_serialization_round_trip(self) -> None:
        alert = AlertNotification(
            run_id=_RUN_ID,
            severity=NotificationSeverity.ERROR,
            title="Critical",
            message="Node crash",
            details={"node": "worker-1"},
        )
        data = alert.model_dump()
        restored = AlertNotification.model_validate(data)
        assert restored == alert


# ---------------------------------------------------------------------------
# HeartbeatNotification model
# ---------------------------------------------------------------------------


class TestHeartbeatNotification:
    """HeartbeatNotification model tests."""

    def test_valid_construction(self) -> None:
        hb = HeartbeatNotification(
            daemon_uptime_seconds=3600.0,
            active_run_id=_RUN_ID,
            queue_depth=2,
        )
        assert hb.event_type == "heartbeat"
        assert hb.daemon_uptime_seconds == 3600.0
        assert hb.active_run_id == _RUN_ID
        assert hb.queue_depth == 2

    def test_event_type_literal(self) -> None:
        hb = HeartbeatNotification(daemon_uptime_seconds=0.0)
        assert hb.event_type == "heartbeat"

    def test_defaults(self) -> None:
        hb = HeartbeatNotification(daemon_uptime_seconds=100.0)
        assert hb.active_run_id is None
        assert hb.queue_depth == 0

    def test_immutable(self) -> None:
        hb = HeartbeatNotification(daemon_uptime_seconds=10.0)
        with pytest.raises(ValidationError):
            hb.daemon_uptime_seconds = 20.0  # type: ignore[misc]

    def test_negative_uptime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HeartbeatNotification(daemon_uptime_seconds=-1.0)

    def test_negative_queue_depth_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HeartbeatNotification(
                daemon_uptime_seconds=10.0,
                queue_depth=-1,
            )

    def test_serialization_round_trip(self) -> None:
        hb = HeartbeatNotification(
            daemon_uptime_seconds=7200.5,
            active_run_id=_RUN_ID,
            queue_depth=3,
        )
        data = hb.model_dump()
        restored = HeartbeatNotification.model_validate(data)
        assert restored == hb


# ---------------------------------------------------------------------------
# NotificationEnvelope model
# ---------------------------------------------------------------------------


class TestNotificationEnvelope:
    """NotificationEnvelope model tests."""

    def test_completion_envelope(self) -> None:
        payload = CompletionNotification(
            run_id=_RUN_ID,
            natural_language_command="test",
            exit_status=0,
        )
        envelope = NotificationEnvelope(
            channel_version=NOTIFICATION_CHANNEL_VERSION,
            event_id="evt-001",
            timestamp=_NOW,
            event_type=NotificationEventType.COMPLETION,
            payload=payload,
        )
        assert envelope.event_type is NotificationEventType.COMPLETION
        assert isinstance(envelope.payload, CompletionNotification)

    def test_alert_envelope(self) -> None:
        payload = AlertNotification(
            severity=NotificationSeverity.WARNING,
            title="Warning",
            message="Slow test",
        )
        envelope = NotificationEnvelope(
            channel_version=NOTIFICATION_CHANNEL_VERSION,
            event_id="evt-002",
            timestamp=_NOW,
            event_type=NotificationEventType.ALERT,
            payload=payload,
        )
        assert envelope.event_type is NotificationEventType.ALERT

    def test_heartbeat_envelope(self) -> None:
        payload = HeartbeatNotification(daemon_uptime_seconds=100.0)
        envelope = NotificationEnvelope(
            channel_version=NOTIFICATION_CHANNEL_VERSION,
            event_id="evt-003",
            timestamp=_NOW,
            event_type=NotificationEventType.HEARTBEAT,
            payload=payload,
        )
        assert envelope.event_type is NotificationEventType.HEARTBEAT

    def test_immutable(self) -> None:
        payload = HeartbeatNotification(daemon_uptime_seconds=10.0)
        envelope = NotificationEnvelope(
            channel_version=NOTIFICATION_CHANNEL_VERSION,
            event_id="evt-004",
            timestamp=_NOW,
            event_type=NotificationEventType.HEARTBEAT,
            payload=payload,
        )
        with pytest.raises(ValidationError):
            envelope.event_id = "changed"  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
        payload = CompletionNotification(
            run_id=_RUN_ID,
            natural_language_command="Run pytest",
            exit_status=0,
            outcome=TestOutcomeSummary(tests_passed=5, tests_total=5),
        )
        envelope = NotificationEnvelope(
            channel_version=NOTIFICATION_CHANNEL_VERSION,
            event_id="evt-005",
            timestamp=_NOW,
            event_type=NotificationEventType.COMPLETION,
            payload=payload,
        )
        json_str = envelope.model_dump_json()
        restored = NotificationEnvelope.model_validate_json(json_str)
        assert restored.event_id == envelope.event_id
        assert restored.event_type == envelope.event_type


# ---------------------------------------------------------------------------
# create_notification_envelope factory
# ---------------------------------------------------------------------------


class TestCreateNotificationEnvelope:
    """Tests for the create_notification_envelope factory."""

    def test_creates_completion_envelope(self) -> None:
        payload = CompletionNotification(
            run_id=_RUN_ID,
            natural_language_command="test",
            exit_status=0,
        )
        envelope = create_notification_envelope(
            event_type=NotificationEventType.COMPLETION,
            payload=payload,
        )
        assert envelope.event_type is NotificationEventType.COMPLETION
        assert envelope.channel_version == NOTIFICATION_CHANNEL_VERSION
        assert envelope.event_id  # auto-generated, non-empty
        assert envelope.timestamp  # auto-generated

    def test_creates_alert_envelope(self) -> None:
        payload = AlertNotification(
            severity=NotificationSeverity.ERROR,
            title="Crash",
            message="Node crashed",
        )
        envelope = create_notification_envelope(
            event_type=NotificationEventType.ALERT,
            payload=payload,
        )
        assert envelope.event_type is NotificationEventType.ALERT

    def test_creates_heartbeat_envelope(self) -> None:
        payload = HeartbeatNotification(daemon_uptime_seconds=100.0)
        envelope = create_notification_envelope(
            event_type=NotificationEventType.HEARTBEAT,
            payload=payload,
        )
        assert envelope.event_type is NotificationEventType.HEARTBEAT

    def test_custom_event_id(self) -> None:
        payload = HeartbeatNotification(daemon_uptime_seconds=10.0)
        envelope = create_notification_envelope(
            event_type=NotificationEventType.HEARTBEAT,
            payload=payload,
            event_id="custom-id-001",
        )
        assert envelope.event_id == "custom-id-001"

    def test_auto_generated_event_id_unique(self) -> None:
        payload = HeartbeatNotification(daemon_uptime_seconds=10.0)
        env1 = create_notification_envelope(
            event_type=NotificationEventType.HEARTBEAT,
            payload=payload,
        )
        env2 = create_notification_envelope(
            event_type=NotificationEventType.HEARTBEAT,
            payload=payload,
        )
        assert env1.event_id != env2.event_id


# ---------------------------------------------------------------------------
# SubscribeRequest / SubscribeResponse models
# ---------------------------------------------------------------------------


class TestSubscribeRequest:
    """SubscribeRequest model tests."""

    def test_valid_construction(self) -> None:
        req = SubscribeRequest()
        assert req.payload_type == "subscribe_notifications"

    def test_with_event_filter(self) -> None:
        req = SubscribeRequest(
            event_filter=(
                NotificationEventType.COMPLETION,
                NotificationEventType.ALERT,
            ),
        )
        assert len(req.event_filter) == 2
        assert NotificationEventType.COMPLETION in req.event_filter
        assert NotificationEventType.ALERT in req.event_filter

    def test_default_subscribes_all(self) -> None:
        req = SubscribeRequest()
        assert req.event_filter is None

    def test_immutable(self) -> None:
        req = SubscribeRequest()
        with pytest.raises(ValidationError):
            req.payload_type = "changed"  # type: ignore[misc]


class TestSubscribeResponse:
    """SubscribeResponse model tests."""

    def test_valid_construction(self) -> None:
        resp = SubscribeResponse(
            subscription_id=_SUB_ID,
            heartbeat_interval_seconds=30,
        )
        assert resp.payload_type == "subscribe_notifications_response"
        assert resp.subscription_id == _SUB_ID
        assert resp.heartbeat_interval_seconds == 30

    def test_subscription_id_required(self) -> None:
        with pytest.raises(ValidationError):
            SubscribeResponse(
                subscription_id="",
                heartbeat_interval_seconds=30,
            )

    def test_immutable(self) -> None:
        resp = SubscribeResponse(
            subscription_id=_SUB_ID,
            heartbeat_interval_seconds=30,
        )
        with pytest.raises(ValidationError):
            resp.subscription_id = "changed"  # type: ignore[misc]


class TestUnsubscribeRequest:
    """UnsubscribeRequest model tests."""

    def test_valid_construction(self) -> None:
        req = UnsubscribeRequest(subscription_id=_SUB_ID)
        assert req.payload_type == "unsubscribe_notifications"
        assert req.subscription_id == _SUB_ID

    def test_subscription_id_required(self) -> None:
        with pytest.raises(ValidationError):
            UnsubscribeRequest(subscription_id="")


class TestUnsubscribeResponse:
    """UnsubscribeResponse model tests."""

    def test_valid_construction(self) -> None:
        resp = UnsubscribeResponse(subscription_id=_SUB_ID)
        assert resp.payload_type == "unsubscribe_notifications_response"
        assert resp.subscription_id == _SUB_ID

    def test_subscription_id_required(self) -> None:
        with pytest.raises(ValidationError):
            UnsubscribeResponse(subscription_id="")
