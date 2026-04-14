"""Tests for thin client command envelope factories.

Validates:
    - Each verb factory produces a valid MessageEnvelope
    - Envelope fields (msg_type, msg_id, timestamp, payload) are correct
    - SSHTargetParams validation rejects invalid inputs
    - Argument validation (limits, empty strings, invalid ports)
    - Confirm reply carries the original_msg_id and decision
    - All factories produce REQUEST type except confirm (CONFIRM_REPLY)
"""

from __future__ import annotations

import pytest

from jules_daemon.ipc.framing import MessageType
from jules_daemon.thin_client.commands import (
    SSHTargetParams,
    build_cancel_request,
    build_confirm_reply,
    build_health_request,
    build_history_request,
    build_interpret_request,
    build_run_request,
    build_status_request,
    build_watch_request,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_valid_envelope(envelope, expected_verb: str, expected_type: MessageType = MessageType.REQUEST):
    """Assert common envelope invariants."""
    assert envelope.msg_type == expected_type
    assert envelope.msg_id.startswith("thin-")
    assert len(envelope.msg_id) > 5
    assert envelope.timestamp  # non-empty
    assert isinstance(envelope.payload, dict)
    assert envelope.payload.get("verb") == expected_verb


# ---------------------------------------------------------------------------
# SSHTargetParams
# ---------------------------------------------------------------------------


class TestSSHTargetParams:
    """Tests for the SSHTargetParams frozen dataclass."""

    def test_valid_minimal(self):
        target = SSHTargetParams(host="ci.example.com", user="deploy")
        assert target.host == "ci.example.com"
        assert target.user == "deploy"
        assert target.port == 22
        assert target.key_path is None

    def test_valid_full(self):
        target = SSHTargetParams(
            host="10.0.0.1",
            user="root",
            port=2222,
            key_path="/home/user/.ssh/id_ed25519",
        )
        assert target.port == 2222
        assert target.key_path == "/home/user/.ssh/id_ed25519"

    def test_empty_host_rejected(self):
        with pytest.raises(ValueError, match="host must not be empty"):
            SSHTargetParams(host="", user="deploy")

    def test_whitespace_host_rejected(self):
        with pytest.raises(ValueError, match="host must not be empty"):
            SSHTargetParams(host="   ", user="deploy")

    def test_empty_user_rejected(self):
        with pytest.raises(ValueError, match="user must not be empty"):
            SSHTargetParams(host="ci.example.com", user="")

    def test_port_zero_rejected(self):
        with pytest.raises(ValueError, match="port must be 1-65535"):
            SSHTargetParams(host="ci.example.com", user="deploy", port=0)

    def test_port_too_high_rejected(self):
        with pytest.raises(ValueError, match="port must be 1-65535"):
            SSHTargetParams(host="ci.example.com", user="deploy", port=70000)

    def test_relative_key_path_rejected(self):
        with pytest.raises(ValueError, match="key_path must be an absolute path"):
            SSHTargetParams(
                host="ci.example.com",
                user="deploy",
                key_path="relative/path",
            )

    def test_to_payload_dict_minimal(self):
        target = SSHTargetParams(host="ci.example.com", user="deploy")
        result = target.to_payload_dict()
        assert result == {
            "target_host": "ci.example.com",
            "target_user": "deploy",
            "target_port": 22,
        }
        assert "key_path" not in result

    def test_to_payload_dict_with_key(self):
        target = SSHTargetParams(
            host="ci.example.com",
            user="deploy",
            key_path="/home/deploy/.ssh/id_rsa",
        )
        result = target.to_payload_dict()
        assert result["key_path"] == "/home/deploy/.ssh/id_rsa"

    def test_frozen(self):
        target = SSHTargetParams(host="ci.example.com", user="deploy")
        with pytest.raises(AttributeError):
            target.host = "other.com"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Health request
# ---------------------------------------------------------------------------


class TestBuildHealthRequest:
    """Tests for the health request factory."""

    def test_valid_health_request(self):
        envelope = build_health_request()
        _assert_valid_envelope(envelope, "health")

    def test_unique_msg_ids(self):
        e1 = build_health_request()
        e2 = build_health_request()
        assert e1.msg_id != e2.msg_id


# ---------------------------------------------------------------------------
# Status request
# ---------------------------------------------------------------------------


class TestBuildStatusRequest:
    """Tests for the status request factory."""

    def test_default_non_verbose(self):
        envelope = build_status_request()
        _assert_valid_envelope(envelope, "status")
        assert envelope.payload["verbose"] is False

    def test_verbose_true(self):
        envelope = build_status_request(verbose=True)
        assert envelope.payload["verbose"] is True


# ---------------------------------------------------------------------------
# History request
# ---------------------------------------------------------------------------


class TestBuildHistoryRequest:
    """Tests for the history request factory."""

    def test_default_parameters(self):
        envelope = build_history_request()
        _assert_valid_envelope(envelope, "history")
        assert envelope.payload["limit"] == 20
        assert "status_filter" not in envelope.payload
        assert "host_filter" not in envelope.payload

    def test_custom_limit(self):
        envelope = build_history_request(limit=5)
        assert envelope.payload["limit"] == 5

    def test_with_filters(self):
        envelope = build_history_request(
            limit=10,
            status_filter="completed",
            host_filter="ci.example.com",
        )
        assert envelope.payload["status_filter"] == "completed"
        assert envelope.payload["host_filter"] == "ci.example.com"

    def test_zero_limit_rejected(self):
        with pytest.raises(ValueError, match="limit must be positive"):
            build_history_request(limit=0)

    def test_negative_limit_rejected(self):
        with pytest.raises(ValueError, match="limit must be positive"):
            build_history_request(limit=-1)

    def test_excessive_limit_rejected(self):
        with pytest.raises(ValueError, match="limit must not exceed 1000"):
            build_history_request(limit=1001)

    def test_boundary_limit_1(self):
        envelope = build_history_request(limit=1)
        assert envelope.payload["limit"] == 1

    def test_boundary_limit_1000(self):
        envelope = build_history_request(limit=1000)
        assert envelope.payload["limit"] == 1000


# ---------------------------------------------------------------------------
# Interpret request
# ---------------------------------------------------------------------------


class TestBuildInterpretRequest:
    """Tests for the daemon-side conversational interpretation request."""

    def test_valid_interpret_request(self):
        envelope = build_interpret_request(input_text="give me the current status")
        _assert_valid_envelope(envelope, "interpret")
        assert envelope.payload["input_text"] == "give me the current status"

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="input_text must not be empty"):
            build_interpret_request(input_text="")


# ---------------------------------------------------------------------------
# Cancel request
# ---------------------------------------------------------------------------


class TestBuildCancelRequest:
    """Tests for the cancel request factory."""

    def test_cancel_current_run(self):
        envelope = build_cancel_request()
        _assert_valid_envelope(envelope, "cancel")
        assert envelope.payload["force"] is False
        assert "run_id" not in envelope.payload
        assert "reason" not in envelope.payload

    def test_cancel_specific_run(self):
        envelope = build_cancel_request(run_id="run-abc-123")
        assert envelope.payload["run_id"] == "run-abc-123"

    def test_cancel_with_force(self):
        envelope = build_cancel_request(force=True)
        assert envelope.payload["force"] is True

    def test_cancel_with_reason(self):
        envelope = build_cancel_request(reason="Tests are hanging")
        assert envelope.payload["reason"] == "Tests are hanging"

    def test_cancel_all_options(self):
        envelope = build_cancel_request(
            run_id="run-xyz",
            force=True,
            reason="Emergency stop",
        )
        assert envelope.payload["run_id"] == "run-xyz"
        assert envelope.payload["force"] is True
        assert envelope.payload["reason"] == "Emergency stop"


# ---------------------------------------------------------------------------
# Run request
# ---------------------------------------------------------------------------


class TestBuildRunRequest:
    """Tests for the run request factory."""

    def test_valid_run_request(self):
        target = SSHTargetParams(host="ci.example.com", user="deploy")
        envelope = build_run_request(
            target=target,
            natural_language="run the unit tests",
        )
        _assert_valid_envelope(envelope, "run")
        assert envelope.payload["natural_language"] == "run the unit tests"
        assert envelope.payload["target_host"] == "ci.example.com"
        assert envelope.payload["target_user"] == "deploy"
        assert envelope.payload["target_port"] == 22

    def test_empty_natural_language_rejected(self):
        target = SSHTargetParams(host="ci.example.com", user="deploy")
        with pytest.raises(ValueError, match="natural_language must not be empty"):
            build_run_request(target=target, natural_language="")

    def test_whitespace_natural_language_rejected(self):
        target = SSHTargetParams(host="ci.example.com", user="deploy")
        with pytest.raises(ValueError, match="natural_language must not be empty"):
            build_run_request(target=target, natural_language="   ")

    def test_run_with_key_path(self):
        target = SSHTargetParams(
            host="ci.example.com",
            user="deploy",
            key_path="/home/deploy/.ssh/id_rsa",
        )
        envelope = build_run_request(
            target=target,
            natural_language="run all tests",
        )
        assert envelope.payload["key_path"] == "/home/deploy/.ssh/id_rsa"

    def test_run_request_with_system_name(self):
        envelope = build_run_request(
            natural_language="run the unit tests",
            system_name="tuto",
        )
        _assert_valid_envelope(envelope, "run")
        assert envelope.payload["system_name"] == "tuto"
        assert "target_host" not in envelope.payload

    def test_run_request_with_infer_target(self):
        envelope = build_run_request(
            natural_language="run the unit tests in tuto",
            infer_target=True,
        )
        _assert_valid_envelope(envelope, "run")
        assert envelope.payload["infer_target"] is True
        assert "target_host" not in envelope.payload
        assert "system_name" not in envelope.payload

    def test_run_request_with_interpret_request(self):
        envelope = build_run_request(
            natural_language="run the unit tests",
            interpret_request=True,
        )
        _assert_valid_envelope(envelope, "run")
        assert envelope.payload["interpret_request"] is True
        assert "target_host" not in envelope.payload
        assert "system_name" not in envelope.payload

    def test_run_request_rejects_missing_target_selection(self):
        with pytest.raises(
            ValueError,
            match="exactly one of target, system_name, infer_target, or interpret_request",
        ):
            build_run_request(natural_language="run the unit tests")

    def test_run_request_rejects_empty_system_name(self):
        with pytest.raises(ValueError, match="system_name must not be empty"):
            build_run_request(
                natural_language="run the unit tests",
                system_name="   ",
            )


# ---------------------------------------------------------------------------
# Watch request
# ---------------------------------------------------------------------------


class TestBuildWatchRequest:
    """Tests for the watch request factory."""

    def test_default_watch(self):
        envelope = build_watch_request()
        _assert_valid_envelope(envelope, "watch")
        assert envelope.payload["tail_lines"] == 50
        assert envelope.payload["follow"] is True
        assert "run_id" not in envelope.payload

    def test_watch_specific_run(self):
        envelope = build_watch_request(run_id="run-abc")
        assert envelope.payload["run_id"] == "run-abc"

    def test_custom_tail_lines(self):
        envelope = build_watch_request(tail_lines=100)
        assert envelope.payload["tail_lines"] == 100

    def test_no_follow(self):
        envelope = build_watch_request(follow=False)
        assert envelope.payload["follow"] is False

    def test_zero_tail_lines_rejected(self):
        with pytest.raises(ValueError, match="tail_lines must be positive"):
            build_watch_request(tail_lines=0)

    def test_negative_tail_lines_rejected(self):
        with pytest.raises(ValueError, match="tail_lines must be positive"):
            build_watch_request(tail_lines=-5)


# ---------------------------------------------------------------------------
# Confirm reply
# ---------------------------------------------------------------------------


class TestBuildConfirmReply:
    """Tests for the confirm reply factory."""

    def test_approve(self):
        envelope = build_confirm_reply(
            approved=True,
            original_msg_id="daemon-abc-123",
        )
        assert envelope.msg_type == MessageType.CONFIRM_REPLY
        assert envelope.payload["verb"] == "confirm"
        assert envelope.payload["approved"] is True
        assert envelope.payload["original_msg_id"] == "daemon-abc-123"
        assert "edited_command" not in envelope.payload

    def test_deny(self):
        envelope = build_confirm_reply(
            approved=False,
            original_msg_id="daemon-xyz",
        )
        assert envelope.payload["approved"] is False

    def test_approve_with_edit(self):
        envelope = build_confirm_reply(
            approved=True,
            original_msg_id="daemon-abc",
            edited_command="cd /opt && pytest -v",
        )
        assert envelope.payload["approved"] is True
        assert envelope.payload["edited_command"] == "cd /opt && pytest -v"

    def test_msg_type_is_confirm_reply(self):
        envelope = build_confirm_reply(
            approved=True,
            original_msg_id="daemon-abc",
        )
        assert envelope.msg_type == MessageType.CONFIRM_REPLY

    def test_msg_id_prefix(self):
        envelope = build_confirm_reply(
            approved=True,
            original_msg_id="daemon-abc",
        )
        assert envelope.msg_id.startswith("thin-")


# ---------------------------------------------------------------------------
# Cross-cutting: immutability of envelopes
# ---------------------------------------------------------------------------


class TestEnvelopeImmutability:
    """Verify that all produced envelopes are frozen."""

    def test_health_envelope_frozen(self):
        envelope = build_health_request()
        with pytest.raises(AttributeError):
            envelope.msg_id = "tampered"  # type: ignore[misc]

    def test_status_envelope_frozen(self):
        envelope = build_status_request()
        with pytest.raises(AttributeError):
            envelope.payload = {"tampered": True}  # type: ignore[misc]

    def test_run_envelope_frozen(self):
        target = SSHTargetParams(host="ci.example.com", user="deploy")
        envelope = build_run_request(target=target, natural_language="run tests")
        with pytest.raises(AttributeError):
            envelope.timestamp = "fake"  # type: ignore[misc]
