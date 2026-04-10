"""Minimal response renderer for the thin client.

Formats daemon responses as human-readable text for terminal display.
Intentionally simple -- no ANSI colors, no rich formatting, just
structured text. This demonstrates that rendering is completely
decoupled from the IPC layer.

All functions are pure (no side effects) and return strings. The
caller decides where to write them.

Usage::

    from jules_daemon.thin_client.renderer import render_response

    text = render_response(envelope)
    print(text)
"""

from __future__ import annotations

from jules_daemon.ipc.framing import MessageEnvelope, MessageType

__all__ = [
    "render_confirm_prompt",
    "render_error",
    "render_response",
    "render_stream_line",
]


def render_response(envelope: MessageEnvelope) -> str:
    """Render a daemon response envelope as human-readable text.

    Routes to the appropriate sub-renderer based on the message type.
    Returns a formatted string suitable for terminal display.

    Args:
        envelope: The response envelope from the daemon.

    Returns:
        Formatted text representation of the response.
    """
    if envelope.msg_type == MessageType.ERROR:
        return render_error(envelope)

    if envelope.msg_type == MessageType.CONFIRM_PROMPT:
        return render_confirm_prompt(envelope)

    if envelope.msg_type == MessageType.STREAM:
        return render_stream_line(envelope)

    return _render_generic_response(envelope)


def render_error(envelope: MessageEnvelope) -> str:
    """Render an ERROR envelope as a formatted error message.

    Args:
        envelope: An ERROR-type envelope from the daemon.

    Returns:
        Formatted error text with status code and message.
    """
    payload = envelope.payload
    error_msg = payload.get("error", "Unknown error")
    status_code = payload.get("status_code", "???")
    verb = payload.get("verb", "unknown")

    lines = [
        f"ERROR [{status_code}] ({verb})",
        f"  {error_msg}",
    ]
    return "\n".join(lines)


def render_confirm_prompt(envelope: MessageEnvelope) -> str:
    """Render a CONFIRM_PROMPT envelope as a confirmation display.

    Presents the SSH command details that require human approval
    in a clear, structured format.

    Args:
        envelope: A CONFIRM_PROMPT-type envelope from the daemon.

    Returns:
        Formatted confirmation display text.
    """
    payload = envelope.payload
    command = payload.get("proposed_command") or payload.get("command") or "<no command>"
    target_host = payload.get("target_host", "<unknown>")
    target_user = payload.get("target_user", "<unknown>")
    risk_level = payload.get("risk_level", "unknown")
    explanation = payload.get("explanation", "")

    separator = "-" * 50

    lines = [
        "",
        separator,
        "  SSH Command Approval Required",
        separator,
        f"  Target:      {target_user}@{target_host}",
        f"  Command:     {command}",
        f"  Risk:        {risk_level}",
    ]

    if explanation:
        lines.append(f"  Explanation: {explanation}")

    lines.extend([
        separator,
        "  [A]pprove  [D]eny  [E]dit",
        separator,
        "",
    ])
    return "\n".join(lines)


def render_stream_line(envelope: MessageEnvelope) -> str:
    """Render a STREAM envelope as a single output line.

    Args:
        envelope: A STREAM-type envelope from the daemon.

    Returns:
        The output line text, or an end-of-stream marker.
    """
    payload = envelope.payload
    is_end = payload.get("is_end", False)

    if is_end:
        return "-- Stream ended --"

    line = payload.get("line", "")
    sequence = payload.get("sequence", "")

    if sequence:
        return f"[{sequence}] {line}"
    return line


def _render_generic_response(envelope: MessageEnvelope) -> str:
    """Render a generic RESPONSE envelope as formatted key-value pairs.

    Args:
        envelope: A RESPONSE-type envelope from the daemon.

    Returns:
        Formatted text with verb, status, and payload fields.
    """
    payload = envelope.payload
    verb = payload.get("verb", "unknown")
    # Prefer the explicit status field, fall back to state, then "ok"
    status = payload.get("status") or payload.get("state") or "ok"

    lines = [f"RESPONSE ({verb}) [{status}]"]

    # Render payload fields (excluding meta fields already shown)
    meta_keys = frozenset({"verb", "status"})
    data_fields = {
        k: v for k, v in sorted(payload.items()) if k not in meta_keys
    }

    if data_fields:
        for key, value in data_fields.items():
            lines.append(f"  {key}: {value}")
    else:
        lines.append("  (no additional data)")

    return "\n".join(lines)
