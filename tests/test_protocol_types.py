"""Tests for the IPC protocol types module.

Covers protocol version constants, MessageKind enum, StatusCode enum,
and all validation/lookup helpers.
"""

from __future__ import annotations

import pytest

from jules_daemon.protocol.types import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MAJOR,
    PROTOCOL_VERSION_MINOR,
    PROTOCOL_VERSION_PATCH,
    MessageKind,
    StatusCode,
    is_client_error,
    is_server_error,
    is_success,
    is_terminal_message,
    parse_message_kind,
    parse_status_code,
    status_code_to_reason,
)


# ---------------------------------------------------------------------------
# Protocol version constants
# ---------------------------------------------------------------------------


class TestProtocolVersion:
    """Protocol version constant tests."""

    def test_version_is_string(self) -> None:
        assert isinstance(PROTOCOL_VERSION, str)

    def test_version_format_semver(self) -> None:
        parts = PROTOCOL_VERSION.split(".")
        assert len(parts) == 3, "version must be semver (MAJOR.MINOR.PATCH)"
        for part in parts:
            assert part.isdigit(), f"version part {part!r} must be numeric"

    def test_version_components_match_string(self) -> None:
        expected = f"{PROTOCOL_VERSION_MAJOR}.{PROTOCOL_VERSION_MINOR}.{PROTOCOL_VERSION_PATCH}"
        assert PROTOCOL_VERSION == expected

    def test_major_is_positive_int(self) -> None:
        assert isinstance(PROTOCOL_VERSION_MAJOR, int)
        assert PROTOCOL_VERSION_MAJOR >= 1

    def test_minor_is_non_negative_int(self) -> None:
        assert isinstance(PROTOCOL_VERSION_MINOR, int)
        assert PROTOCOL_VERSION_MINOR >= 0

    def test_patch_is_non_negative_int(self) -> None:
        assert isinstance(PROTOCOL_VERSION_PATCH, int)
        assert PROTOCOL_VERSION_PATCH >= 0

    def test_protocol_name(self) -> None:
        assert isinstance(PROTOCOL_NAME, str)
        assert PROTOCOL_NAME.strip() != ""


# ---------------------------------------------------------------------------
# MessageKind enum
# ---------------------------------------------------------------------------


class TestMessageKind:
    """MessageKind enum tests."""

    def test_has_request(self) -> None:
        assert MessageKind.REQUEST.value == "request"

    def test_has_response(self) -> None:
        assert MessageKind.RESPONSE.value == "response"

    def test_has_notification(self) -> None:
        assert MessageKind.NOTIFICATION.value == "notification"

    def test_has_error(self) -> None:
        assert MessageKind.ERROR.value == "error"

    def test_has_stream(self) -> None:
        assert MessageKind.STREAM.value == "stream"

    def test_has_confirm_prompt(self) -> None:
        assert MessageKind.CONFIRM_PROMPT.value == "confirm_prompt"

    def test_has_confirm_reply(self) -> None:
        assert MessageKind.CONFIRM_REPLY.value == "confirm_reply"

    def test_all_values_are_lowercase_strings(self) -> None:
        for member in MessageKind:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()
            assert member.value.strip() != ""

    def test_values_are_unique(self) -> None:
        values = [m.value for m in MessageKind]
        assert len(values) == len(set(values))

    def test_total_member_count(self) -> None:
        # Ensure we know exactly how many kinds exist
        assert len(MessageKind) == 7

    def test_members_are_immutable(self) -> None:
        # Enum members should not be reassignable
        with pytest.raises(AttributeError):
            MessageKind.REQUEST = "something_else"  # type: ignore[misc]


class TestParseMessageKind:
    """parse_message_kind() tests."""

    def test_roundtrip_all_members(self) -> None:
        for member in MessageKind:
            assert parse_message_kind(member.value) is member

    def test_case_insensitive(self) -> None:
        assert parse_message_kind("REQUEST") is MessageKind.REQUEST
        assert parse_message_kind("Request") is MessageKind.REQUEST
        assert parse_message_kind("rEqUeSt") is MessageKind.REQUEST

    def test_strips_whitespace(self) -> None:
        assert parse_message_kind("  response  ") is MessageKind.RESPONSE

    def test_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown message kind"):
            parse_message_kind("unknown_type")

    def test_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            parse_message_kind("")

    def test_whitespace_only_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            parse_message_kind("   ")


class TestIsTerminalMessage:
    """is_terminal_message() tests."""

    def test_response_is_terminal(self) -> None:
        assert is_terminal_message(MessageKind.RESPONSE) is True

    def test_error_is_terminal(self) -> None:
        assert is_terminal_message(MessageKind.ERROR) is True

    def test_request_is_not_terminal(self) -> None:
        assert is_terminal_message(MessageKind.REQUEST) is False

    def test_notification_is_not_terminal(self) -> None:
        assert is_terminal_message(MessageKind.NOTIFICATION) is False

    def test_stream_is_not_terminal(self) -> None:
        assert is_terminal_message(MessageKind.STREAM) is False

    def test_confirm_prompt_is_not_terminal(self) -> None:
        assert is_terminal_message(MessageKind.CONFIRM_PROMPT) is False

    def test_confirm_reply_is_not_terminal(self) -> None:
        assert is_terminal_message(MessageKind.CONFIRM_REPLY) is False


# ---------------------------------------------------------------------------
# StatusCode enum
# ---------------------------------------------------------------------------


class TestStatusCode:
    """StatusCode enum tests."""

    # -- Success codes --

    def test_ok(self) -> None:
        assert StatusCode.OK.value == 200

    def test_accepted(self) -> None:
        assert StatusCode.ACCEPTED.value == 202

    def test_no_content(self) -> None:
        assert StatusCode.NO_CONTENT.value == 204

    # -- Client error codes --

    def test_bad_request(self) -> None:
        assert StatusCode.BAD_REQUEST.value == 400

    def test_unauthorized(self) -> None:
        assert StatusCode.UNAUTHORIZED.value == 401

    def test_forbidden(self) -> None:
        assert StatusCode.FORBIDDEN.value == 403

    def test_not_found(self) -> None:
        assert StatusCode.NOT_FOUND.value == 404

    def test_conflict(self) -> None:
        assert StatusCode.CONFLICT.value == 409

    def test_unprocessable(self) -> None:
        assert StatusCode.UNPROCESSABLE.value == 422

    # -- Server error codes --

    def test_internal_error(self) -> None:
        assert StatusCode.INTERNAL_ERROR.value == 500

    def test_not_implemented(self) -> None:
        assert StatusCode.NOT_IMPLEMENTED.value == 501

    def test_service_unavailable(self) -> None:
        assert StatusCode.SERVICE_UNAVAILABLE.value == 502

    def test_busy(self) -> None:
        assert StatusCode.BUSY.value == 503

    def test_timeout(self) -> None:
        assert StatusCode.TIMEOUT.value == 504

    def test_all_values_are_ints(self) -> None:
        for member in StatusCode:
            assert isinstance(member.value, int)

    def test_values_are_unique(self) -> None:
        values = [m.value for m in StatusCode]
        assert len(values) == len(set(values))

    def test_total_member_count(self) -> None:
        assert len(StatusCode) == 14


class TestParseStatusCode:
    """parse_status_code() tests."""

    def test_roundtrip_all_members(self) -> None:
        for member in StatusCode:
            assert parse_status_code(member.value) is member

    def test_unknown_code_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown status code"):
            parse_status_code(999)


class TestStatusCodeClassifiers:
    """is_success(), is_client_error(), is_server_error() tests."""

    def test_ok_is_success(self) -> None:
        assert is_success(StatusCode.OK) is True

    def test_accepted_is_success(self) -> None:
        assert is_success(StatusCode.ACCEPTED) is True

    def test_no_content_is_success(self) -> None:
        assert is_success(StatusCode.NO_CONTENT) is True

    def test_bad_request_is_not_success(self) -> None:
        assert is_success(StatusCode.BAD_REQUEST) is False

    def test_internal_error_is_not_success(self) -> None:
        assert is_success(StatusCode.INTERNAL_ERROR) is False

    def test_bad_request_is_client_error(self) -> None:
        assert is_client_error(StatusCode.BAD_REQUEST) is True

    def test_forbidden_is_client_error(self) -> None:
        assert is_client_error(StatusCode.FORBIDDEN) is True

    def test_ok_is_not_client_error(self) -> None:
        assert is_client_error(StatusCode.OK) is False

    def test_internal_error_is_not_client_error(self) -> None:
        assert is_client_error(StatusCode.INTERNAL_ERROR) is False

    def test_internal_error_is_server_error(self) -> None:
        assert is_server_error(StatusCode.INTERNAL_ERROR) is True

    def test_busy_is_server_error(self) -> None:
        assert is_server_error(StatusCode.BUSY) is True

    def test_ok_is_not_server_error(self) -> None:
        assert is_server_error(StatusCode.OK) is False

    def test_bad_request_is_not_server_error(self) -> None:
        assert is_server_error(StatusCode.BAD_REQUEST) is False


class TestStatusCodeToReason:
    """status_code_to_reason() tests."""

    def test_all_codes_have_reason(self) -> None:
        for member in StatusCode:
            reason = status_code_to_reason(member)
            assert isinstance(reason, str)
            assert reason.strip() != ""

    def test_ok_reason(self) -> None:
        assert status_code_to_reason(StatusCode.OK) == "OK"

    def test_bad_request_reason(self) -> None:
        assert status_code_to_reason(StatusCode.BAD_REQUEST) == "Bad Request"

    def test_internal_error_reason(self) -> None:
        assert status_code_to_reason(StatusCode.INTERNAL_ERROR) == "Internal Error"

    def test_busy_reason(self) -> None:
        assert status_code_to_reason(StatusCode.BUSY) == "Busy"
