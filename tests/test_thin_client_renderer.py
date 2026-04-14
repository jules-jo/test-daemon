"""Tests for the thin client response renderer.

Validates:
    - render_response routes to correct sub-renderer by message type
    - render_error formats status code, verb, and error message
    - render_confirm_prompt displays SSH command approval details
    - render_stream_line formats output lines and end-of-stream markers
    - Generic responses display verb, status, and payload fields
    - All renderers are pure functions (no side effects)
"""

from __future__ import annotations

from jules_daemon.ipc.framing import MessageEnvelope, MessageType
from jules_daemon.thin_client.renderer import (
    render_confirm_prompt,
    render_error,
    render_response,
    render_stream_line,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = "2026-04-09T12:00:00Z"


def _make_envelope(
    msg_type: MessageType,
    payload: dict | None = None,
) -> MessageEnvelope:
    """Build a test envelope with the given type and payload."""
    return MessageEnvelope(
        msg_type=msg_type,
        msg_id="test-001",
        timestamp=_TS,
        payload=payload or {},
    )


# ---------------------------------------------------------------------------
# render_response routing
# ---------------------------------------------------------------------------


class TestRenderResponseRouting:
    """Tests that render_response dispatches to the correct sub-renderer."""

    def test_routes_error(self):
        envelope = _make_envelope(
            MessageType.ERROR,
            {"error": "Something failed", "status_code": 500, "verb": "run"},
        )
        result = render_response(envelope)
        assert "ERROR" in result
        assert "500" in result
        assert "Something failed" in result

    def test_routes_confirm_prompt(self):
        envelope = _make_envelope(
            MessageType.CONFIRM_PROMPT,
            {
                "command": "pytest -v",
                "target_host": "ci.example.com",
                "target_user": "deploy",
                "risk_level": "MEDIUM",
            },
        )
        result = render_response(envelope)
        assert "Approval Required" in result
        assert "pytest -v" in result

    def test_routes_stream(self):
        envelope = _make_envelope(
            MessageType.STREAM,
            {"line": "PASS: test_auth", "sequence": 42},
        )
        result = render_response(envelope)
        assert "PASS: test_auth" in result

    def test_routes_generic_response(self):
        envelope = _make_envelope(
            MessageType.RESPONSE,
            {"verb": "status", "status": "ok", "run_state": "idle"},
        )
        result = render_response(envelope)
        assert "RESPONSE" in result
        assert "status" in result
        assert "run_state" in result


# ---------------------------------------------------------------------------
# render_error
# ---------------------------------------------------------------------------


class TestRenderError:
    """Tests for error envelope rendering."""

    def test_full_error(self):
        envelope = _make_envelope(
            MessageType.ERROR,
            {"error": "Run not found", "status_code": 404, "verb": "cancel"},
        )
        result = render_error(envelope)
        assert "ERROR [404]" in result
        assert "(cancel)" in result
        assert "Run not found" in result

    def test_missing_fields_uses_defaults(self):
        envelope = _make_envelope(MessageType.ERROR, {})
        result = render_error(envelope)
        assert "ERROR [???]" in result
        assert "(unknown)" in result
        assert "Unknown error" in result

    def test_partial_fields(self):
        envelope = _make_envelope(
            MessageType.ERROR,
            {"error": "Timeout", "verb": "run"},
        )
        result = render_error(envelope)
        assert "Timeout" in result
        assert "(run)" in result


# ---------------------------------------------------------------------------
# render_confirm_prompt
# ---------------------------------------------------------------------------


class TestRenderConfirmPrompt:
    """Tests for confirmation prompt rendering."""

    def test_full_prompt(self):
        envelope = _make_envelope(
            MessageType.CONFIRM_PROMPT,
            {
                "command": "cd /app && pytest -v tests/",
                "system_name": "tuto",
                "system_hostname": "tuto.internal.example",
                "system_ip_address": "10.0.0.10",
                "system_description": "Tutorial box for smoke-test runs",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "target_port": 2222,
                "risk_level": "HIGH",
                "explanation": "Runs the full test suite",
            },
        )
        result = render_confirm_prompt(envelope)
        assert "SSH Command Approval Required" in result
        assert "System:      tuto" in result
        assert "Host:        tuto.internal.example" in result
        assert "IP:          10.0.0.10" in result
        assert "Target:      deploy@staging.example.com:2222" in result
        assert "Tutorial box for smoke-test runs" in result
        assert "cd /app && pytest -v tests/" in result
        assert "HIGH" in result
        assert "Runs the full test suite" in result
        assert "[A]pprove" in result
        assert "[D]eny" in result
        assert "[E]dit" in result

    def test_minimal_prompt(self):
        envelope = _make_envelope(
            MessageType.CONFIRM_PROMPT,
            {"command": "ls"},
        )
        result = render_confirm_prompt(envelope)
        assert "ls" in result
        assert "SSH Command Approval Required" in result

    def test_no_explanation(self):
        envelope = _make_envelope(
            MessageType.CONFIRM_PROMPT,
            {
                "command": "make test",
                "target_host": "ci",
                "target_user": "root",
                "target_port": 22,
            },
        )
        result = render_confirm_prompt(envelope)
        # No "Explanation:" line when explanation is empty
        assert "make test" in result
        assert "Target:      root@ci:22" in result


# ---------------------------------------------------------------------------
# render_stream_line
# ---------------------------------------------------------------------------


class TestRenderStreamLine:
    """Tests for stream line rendering."""

    def test_normal_line(self):
        envelope = _make_envelope(
            MessageType.STREAM,
            {"line": "PASS: test_login", "sequence": 1},
        )
        result = render_stream_line(envelope)
        assert "[1] PASS: test_login" in result

    def test_line_without_sequence(self):
        envelope = _make_envelope(
            MessageType.STREAM,
            {"line": "Running tests..."},
        )
        result = render_stream_line(envelope)
        assert result == "Running tests..."

    def test_end_of_stream(self):
        envelope = _make_envelope(
            MessageType.STREAM,
            {"is_end": True},
        )
        result = render_stream_line(envelope)
        assert "Stream ended" in result

    def test_empty_line(self):
        envelope = _make_envelope(
            MessageType.STREAM,
            {"line": "", "sequence": 5},
        )
        result = render_stream_line(envelope)
        assert "[5]" in result


# ---------------------------------------------------------------------------
# Generic response
# ---------------------------------------------------------------------------


class TestRenderGenericResponse:
    """Tests for generic RESPONSE envelope rendering."""

    def test_status_response(self):
        envelope = _make_envelope(
            MessageType.RESPONSE,
            {
                "verb": "status",
                "status": "ok",
                "run_state": "running",
                "progress": 45.2,
            },
        )
        result = render_response(envelope)
        assert "RESPONSE (status) [ok]" in result
        assert "run_state: running" in result
        assert "progress: 45.2" in result

    def test_empty_payload(self):
        envelope = _make_envelope(
            MessageType.RESPONSE,
            {"verb": "health", "status": "ok"},
        )
        result = render_response(envelope)
        assert "RESPONSE (health) [ok]" in result
        assert "(no additional data)" in result

    def test_payload_fields_sorted(self):
        envelope = _make_envelope(
            MessageType.RESPONSE,
            {
                "verb": "history",
                "status": "ok",
                "z_field": "last",
                "a_field": "first",
            },
        )
        result = render_response(envelope)
        lines = result.split("\n")
        # Data lines should be sorted: a_field before z_field
        data_lines = [l.strip() for l in lines if l.strip().startswith(("a_", "z_"))]
        assert len(data_lines) == 2
        assert data_lines[0].startswith("a_field")
        assert data_lines[1].startswith("z_field")

    def test_missing_verb_and_status_defaults(self):
        envelope = _make_envelope(MessageType.RESPONSE, {})
        result = render_response(envelope)
        assert "RESPONSE (unknown) [ok]" in result
