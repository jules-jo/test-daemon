"""Tests for the discover verb and test_discovery module.

Covers:
- DiscoveredTestSpec construction and immutability
- format_spec_preview output formatting
- save_discovered_spec wiki file creation and content
- _parse_llm_response JSON extraction (including markdown fences)
- discover_test fallback when LLM is unavailable
- Verb enum includes DISCOVER
- Request validator accepts "discover" verb
- build_discover_request envelope construction
- CLI verb routing for discover
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.cli.verbs import DiscoverArgs, Verb, parse_verb
from jules_daemon.execution.test_discovery import (
    DiscoveredTestSpec,
    _parse_llm_response,
    build_discovery_help_command,
    build_discovery_help_commands,
    format_spec_preview,
    normalize_discovery_command,
    resolve_discovery_command_candidates,
    save_discovered_spec,
)
from jules_daemon.ipc.framing import MessageEnvelope, MessageType
from jules_daemon.ipc.request_validator import validate_request
from jules_daemon.thin_client.commands import (
    SSHTargetParams,
    build_discover_request,
)


# ---------------------------------------------------------------------------
# DiscoveredTestSpec
# ---------------------------------------------------------------------------


class TestDiscoveredTestSpec:
    """Tests for the DiscoveredTestSpec frozen dataclass."""

    def test_construction(self) -> None:
        spec = DiscoveredTestSpec(
            command_template="python3 test.py -n {log_folder}",
            required_args=("log_folder",),
            optional_args=("verbose",),
            arg_descriptions={"log_folder": "Name of the log folder"},
            typical_duration=60,
            raw_help_text="usage: test.py [-h] -n LOG_FOLDER",
        )
        assert spec.command_template == "python3 test.py -n {log_folder}"
        assert spec.required_args == ("log_folder",)
        assert spec.optional_args == ("verbose",)
        assert spec.typical_duration == 60

    def test_immutability(self) -> None:
        spec = DiscoveredTestSpec(
            command_template="cmd",
            required_args=(),
            optional_args=(),
            arg_descriptions={},
            typical_duration=None,
            raw_help_text="help",
        )
        with pytest.raises(AttributeError):
            spec.command_template = "other"  # type: ignore[misc]

    def test_none_duration(self) -> None:
        spec = DiscoveredTestSpec(
            command_template="cmd",
            required_args=(),
            optional_args=(),
            arg_descriptions={},
            typical_duration=None,
            raw_help_text="help",
        )
        assert spec.typical_duration is None


# ---------------------------------------------------------------------------
# Discovery command normalization
# ---------------------------------------------------------------------------


class TestDiscoveryCommandNormalization:
    """Tests for discovery command normalization helpers."""

    def test_normalize_bare_python_script_prefers_python3(self) -> None:
        assert (
            normalize_discovery_command("/root/step.py")
            == "python3 /root/step.py"
        )

    def test_normalize_python_command_is_unchanged(self) -> None:
        assert (
            normalize_discovery_command("python3 /root/step.py --flag")
            == "python3 /root/step.py --flag"
        )

    def test_resolve_python_script_candidates_in_order(self) -> None:
        assert resolve_discovery_command_candidates("/root/step.py") == (
            "python3 /root/step.py",
            "python /root/step.py",
        )

    def test_build_help_command_uses_first_python3_candidate(self) -> None:
        assert (
            build_discovery_help_command("/root/step.py")
            == "python3 /root/step.py -h"
        )

    def test_build_help_commands_includes_python_fallback(self) -> None:
        assert build_discovery_help_commands("/root/step.py") == (
            "python3 /root/step.py -h",
            "python /root/step.py -h",
        )


# ---------------------------------------------------------------------------
# format_spec_preview
# ---------------------------------------------------------------------------


class TestFormatSpecPreview:
    """Tests for the format_spec_preview function."""

    def test_full_preview(self) -> None:
        spec = DiscoveredTestSpec(
            command_template="python3 test.py -n {log_folder} -t {test_name}",
            required_args=("log_folder", "test_name"),
            optional_args=("verbose",),
            arg_descriptions={
                "log_folder": "Log folder name",
                "test_name": "Test to run",
            },
            typical_duration=120,
            raw_help_text="usage: ...",
        )
        preview = format_spec_preview(spec)
        assert "Command: python3 test.py" in preview
        assert "Required args: log_folder, test_name" in preview
        assert "Optional args: verbose" in preview
        assert "Typical duration: 120s" in preview
        assert "log_folder: Log folder name" in preview

    def test_minimal_preview(self) -> None:
        spec = DiscoveredTestSpec(
            command_template="cmd",
            required_args=(),
            optional_args=(),
            arg_descriptions={},
            typical_duration=None,
            raw_help_text="help",
        )
        preview = format_spec_preview(spec)
        assert "Command: cmd" in preview
        assert "Required args" not in preview
        assert "Optional args" not in preview
        assert "Typical duration" not in preview


# ---------------------------------------------------------------------------
# save_discovered_spec
# ---------------------------------------------------------------------------


class TestSaveDiscoveredSpec:
    """Tests for wiki file persistence."""

    def test_creates_wiki_file(self, tmp_path: Path) -> None:
        spec = DiscoveredTestSpec(
            command_template="python3.8 /root/tests/my_test.py -n {log}",
            required_args=("log",),
            optional_args=(),
            arg_descriptions={"log": "Log folder"},
            typical_duration=None,
            raw_help_text="usage: my_test.py",
        )
        result_path = save_discovered_spec(
            wiki_root=tmp_path,
            spec=spec,
            command="python3.8 /root/tests/my_test.py",
            host="10.74.30.211",
        )
        assert result_path.exists()
        assert result_path.suffix == ".md"
        content = result_path.read_text(encoding="utf-8")
        assert "test-spec" in content
        assert "python3.8" in content or "my-test-py" in content

    def test_file_in_knowledge_dir(self, tmp_path: Path) -> None:
        spec = DiscoveredTestSpec(
            command_template="cmd",
            required_args=(),
            optional_args=(),
            arg_descriptions={},
            typical_duration=None,
            raw_help_text="help",
        )
        result_path = save_discovered_spec(
            wiki_root=tmp_path,
            spec=spec,
            command="./run_test.sh",
            host="host1",
        )
        assert "pages/daemon/knowledge" in str(result_path)

    def test_frontmatter_fields(self, tmp_path: Path) -> None:
        spec = DiscoveredTestSpec(
            command_template="python3 test.py -n {name}",
            required_args=("name",),
            optional_args=("verbose",),
            arg_descriptions={"name": "The name arg"},
            typical_duration=30,
            raw_help_text="usage: test.py [-h] -n NAME",
        )
        result_path = save_discovered_spec(
            wiki_root=tmp_path,
            spec=spec,
            command="python3 test.py",
            host="myhost",
        )
        content = result_path.read_text(encoding="utf-8")
        assert "type: test-spec" in content
        assert "discovered_from_host: myhost" in content
        assert "command_pattern: python3 test.py -n {name}" in content
        assert "test_file_path: test.py" in content

    def test_persists_full_test_file_path(self, tmp_path: Path) -> None:
        spec = DiscoveredTestSpec(
            command_template="python3 /root/tests/step.py --target {target}",
            required_args=("target",),
            optional_args=(),
            arg_descriptions={"target": "Target id"},
            typical_duration=None,
            raw_help_text="usage: step.py [-h] --target TARGET",
        )
        result_path = save_discovered_spec(
            wiki_root=tmp_path,
            spec=spec,
            command="python3 /root/tests/step.py",
            host="myhost",
        )
        content = result_path.read_text(encoding="utf-8")
        assert "test_file_path: /root/tests/step.py" in content
        assert "`/root/tests/step.py`" in content

    def test_no_temp_file_left(self, tmp_path: Path) -> None:
        spec = DiscoveredTestSpec(
            command_template="cmd",
            required_args=(),
            optional_args=(),
            arg_descriptions={},
            typical_duration=None,
            raw_help_text="help",
        )
        save_discovered_spec(
            wiki_root=tmp_path,
            spec=spec,
            command="cmd",
            host="host",
        )
        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------


class TestParseLlmResponse:
    """Tests for JSON extraction from LLM output."""

    def test_plain_json(self) -> None:
        raw = json.dumps({
            "command_template": "cmd -n {name}",
            "required_args": ["name"],
            "optional_args": [],
            "arg_descriptions": {"name": "The name"},
            "typical_duration": None,
        })
        result = _parse_llm_response(raw)
        assert result is not None
        assert result["command_template"] == "cmd -n {name}"
        assert result["required_args"] == ["name"]

    def test_markdown_fenced_json(self) -> None:
        raw = "```json\n" + json.dumps({
            "command_template": "cmd",
            "required_args": [],
            "optional_args": [],
            "arg_descriptions": {},
            "typical_duration": None,
        }) + "\n```"
        result = _parse_llm_response(raw)
        assert result is not None
        assert result["command_template"] == "cmd"

    def test_invalid_json(self) -> None:
        result = _parse_llm_response("this is not json")
        assert result is None

    def test_non_dict_json(self) -> None:
        result = _parse_llm_response("[1, 2, 3]")
        assert result is None

    def test_empty_string(self) -> None:
        result = _parse_llm_response("")
        assert result is None


# ---------------------------------------------------------------------------
# Verb enum
# ---------------------------------------------------------------------------


class TestVerbEnum:
    """Tests that DISCOVER is in the Verb enum."""

    def test_discover_in_enum(self) -> None:
        assert Verb.DISCOVER.value == "discover"

    def test_parse_verb_discover(self) -> None:
        assert parse_verb("discover") == Verb.DISCOVER

    def test_parse_verb_discover_case_insensitive(self) -> None:
        assert parse_verb("DISCOVER") == Verb.DISCOVER
        assert parse_verb("  Discover  ") == Verb.DISCOVER


# ---------------------------------------------------------------------------
# DiscoverArgs
# ---------------------------------------------------------------------------


class TestDiscoverArgs:
    """Tests for the DiscoverArgs frozen dataclass."""

    def test_valid_construction(self) -> None:
        args = DiscoverArgs(
            target_host="10.0.0.1",
            target_user="root",
            command="python3 test.py",
        )
        assert args.target_host == "10.0.0.1"
        assert args.target_port == 22

    def test_custom_port(self) -> None:
        args = DiscoverArgs(
            target_host="host",
            target_user="user",
            command="cmd",
            target_port=2222,
        )
        assert args.target_port == 2222

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError, match="target_host"):
            DiscoverArgs(
                target_host="  ",
                target_user="root",
                command="cmd",
            )

    def test_empty_user_raises(self) -> None:
        with pytest.raises(ValueError, match="target_user"):
            DiscoverArgs(
                target_host="host",
                target_user="",
                command="cmd",
            )

    def test_empty_command_raises(self) -> None:
        with pytest.raises(ValueError, match="command"):
            DiscoverArgs(
                target_host="host",
                target_user="root",
                command="  ",
            )

    def test_invalid_port_raises(self) -> None:
        with pytest.raises(ValueError, match="target_port"):
            DiscoverArgs(
                target_host="host",
                target_user="root",
                command="cmd",
                target_port=0,
            )


# ---------------------------------------------------------------------------
# Request validator
# ---------------------------------------------------------------------------


class TestRequestValidator:
    """Tests that the validator accepts discover requests."""

    def test_valid_discover_request(self) -> None:
        envelope = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="test-123",
            timestamp="2024-01-01T00:00:00Z",
            payload={
                "verb": "discover",
                "target_host": "10.0.0.1",
                "target_user": "root",
                "command": "python3 test.py",
            },
        )
        result = validate_request(envelope)
        assert result.is_valid
        assert result.verb == "discover"
        assert result.parsed_payload["target_host"] == "10.0.0.1"
        assert result.parsed_payload["command"] == "python3 test.py"

    def test_missing_command_field(self) -> None:
        envelope = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="test-456",
            timestamp="2024-01-01T00:00:00Z",
            payload={
                "verb": "discover",
                "target_host": "host",
                "target_user": "root",
            },
        )
        result = validate_request(envelope)
        assert not result.is_valid
        error_fields = [e.field for e in result.errors]
        assert "command" in error_fields

    def test_missing_host_and_user(self) -> None:
        envelope = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="test-789",
            timestamp="2024-01-01T00:00:00Z",
            payload={
                "verb": "discover",
                "command": "test.py",
            },
        )
        result = validate_request(envelope)
        assert not result.is_valid
        error_fields = [e.field for e in result.errors]
        assert "target_host" in error_fields
        assert "target_user" in error_fields

    def test_optional_port_defaults(self) -> None:
        envelope = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="test-port",
            timestamp="2024-01-01T00:00:00Z",
            payload={
                "verb": "discover",
                "target_host": "host",
                "target_user": "root",
                "command": "cmd",
            },
        )
        result = validate_request(envelope)
        assert result.is_valid
        assert result.parsed_payload["target_port"] == 22


# ---------------------------------------------------------------------------
# build_discover_request
# ---------------------------------------------------------------------------


class TestBuildDiscoverRequest:
    """Tests for the thin client discover request builder."""

    def test_builds_valid_envelope(self) -> None:
        target = SSHTargetParams(host="10.0.0.1", user="root")
        envelope = build_discover_request(
            target=target,
            command="python3 test.py",
        )
        assert envelope.msg_type == MessageType.REQUEST
        assert envelope.payload["verb"] == "discover"
        assert envelope.payload["command"] == "python3 test.py"
        assert envelope.payload["target_host"] == "10.0.0.1"
        assert envelope.payload["target_user"] == "root"

    def test_empty_command_raises(self) -> None:
        target = SSHTargetParams(host="host", user="user")
        with pytest.raises(ValueError, match="command"):
            build_discover_request(target=target, command="")

    def test_whitespace_command_raises(self) -> None:
        target = SSHTargetParams(host="host", user="user")
        with pytest.raises(ValueError, match="command"):
            build_discover_request(target=target, command="   ")

    def test_custom_port(self) -> None:
        target = SSHTargetParams(host="host", user="user", port=2222)
        envelope = build_discover_request(
            target=target,
            command="cmd",
        )
        assert envelope.payload["target_port"] == 2222


# ---------------------------------------------------------------------------
# discover_test (async, mocked SSH)
# ---------------------------------------------------------------------------


class TestDiscoverTestAsync:
    """Tests for the async discover_test function with mocked SSH."""

    @pytest.mark.asyncio
    async def test_returns_none_on_ssh_failure(self) -> None:
        from jules_daemon.execution.test_discovery import discover_test

        with patch(
            "jules_daemon.execution.test_discovery.resolve_ssh_credentials",
            return_value=None,
        ), patch(
            "jules_daemon.execution.test_discovery._fetch_help_text",
            return_value=None,
        ):
            result = await discover_test(
                host="10.0.0.1",
                user="root",
                command="test.py",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_raw_spec_without_llm(self) -> None:
        from jules_daemon.execution.test_discovery import (
            _HelpProbeResult,
            discover_test,
        )

        with patch(
            "jules_daemon.execution.test_discovery.resolve_ssh_credentials",
            return_value=None,
        ), patch(
            "jules_daemon.execution.test_discovery._fetch_help_text",
            return_value=_HelpProbeResult(
                help_text="usage: test.py [-h] -n NAME",
                executed_command="python3 test.py",
            ),
        ):
            result = await discover_test(
                host="10.0.0.1",
                user="root",
                command="python3 test.py",
            )
            assert result is not None
            assert result.raw_help_text == "usage: test.py [-h] -n NAME"
            assert result.command_template == "python3 test.py"
            assert result.required_args == ()

    @pytest.mark.asyncio
    async def test_returns_raw_spec_with_normalized_python_script_command(self) -> None:
        from jules_daemon.execution.test_discovery import (
            _HelpProbeResult,
            discover_test,
        )

        with patch(
            "jules_daemon.execution.test_discovery.resolve_ssh_credentials",
            return_value=None,
        ), patch(
            "jules_daemon.execution.test_discovery._fetch_help_text",
            return_value=_HelpProbeResult(
                help_text="usage: step.py [-h] --name NAME",
                executed_command="python3 /root/step.py",
            ),
        ):
            result = await discover_test(
                host="10.0.0.1",
                user="root",
                command="/root/step.py",
            )
            assert result is not None
            assert result.command_template == "python3 /root/step.py"

    @pytest.mark.asyncio
    async def test_normalizes_llm_parsed_python_script_template(self) -> None:
        from jules_daemon.execution.test_discovery import (
            _HelpProbeResult,
            discover_test,
        )

        parsed_spec = DiscoveredTestSpec(
            command_template="/root/step.py --name {name}",
            required_args=("name",),
            optional_args=(),
            arg_descriptions={"name": "Name to use"},
            typical_duration=None,
            raw_help_text="usage: step.py [-h] --name NAME",
        )

        with patch(
            "jules_daemon.execution.test_discovery.resolve_ssh_credentials",
            return_value=None,
        ), patch(
            "jules_daemon.execution.test_discovery._fetch_help_text",
            return_value=_HelpProbeResult(
                help_text="usage: step.py [-h] --name NAME",
                executed_command="python3 /root/step.py",
            ),
        ), patch(
            "jules_daemon.execution.test_discovery._parse_help_with_llm",
            AsyncMock(return_value=parsed_spec),
        ):
            result = await discover_test(
                host="10.0.0.1",
                user="root",
                command="/root/step.py",
                llm_client=MagicMock(),
                llm_config=MagicMock(),
            )
            assert result is not None
            assert result.command_template == "python3 /root/step.py --name {name}"

    @pytest.mark.asyncio
    async def test_fetch_help_text_falls_back_from_python3_to_python(self) -> None:
        from jules_daemon.execution.test_discovery import _fetch_help_text

        with patch(
            "jules_daemon.execution.test_discovery._run_help_via_paramiko",
            side_effect=[
                (127, "", "python3: command not found"),
                (127, "", "python3: command not found"),
                (0, "usage: step.py [-h] --name NAME", ""),
            ],
        ) as mock_run:
            result = await _fetch_help_text(
                host="10.0.0.1",
                port=22,
                username="root",
                credential=None,
                command="/root/step.py",
            )

        assert result is not None
        assert result.executed_command == "python /root/step.py"
        assert result.help_text == "usage: step.py [-h] --name NAME"
        commands = [call.kwargs["command"] for call in mock_run.call_args_list]
        assert commands == [
            "python3 /root/step.py -h",
            "python3 /root/step.py --help",
            "python /root/step.py -h",
        ]

    @pytest.mark.asyncio
    async def test_fetch_help_text_raises_on_ssh_auth_failure(self) -> None:
        from jules_daemon.execution.test_discovery import _fetch_help_text
        from jules_daemon.ssh.errors import SSHAuthenticationError

        with patch(
            "jules_daemon.execution.test_discovery._run_help_via_paramiko",
            side_effect=SSHAuthenticationError("Authentication failed"),
        ):
            with pytest.raises(SSHAuthenticationError, match="Authentication failed"):
                await _fetch_help_text(
                    host="10.0.0.1",
                    port=22,
                    username="root",
                    credential=None,
                    command="/root/step.py",
                )

    @pytest.mark.asyncio
    async def test_fetch_help_text_raises_probe_error_with_root_cause(self) -> None:
        from jules_daemon.execution.test_discovery import (
            DiscoveryProbeError,
            _fetch_help_text,
        )

        with patch(
            "jules_daemon.execution.test_discovery._run_help_via_paramiko",
            side_effect=[
                (1, "", "python3: can't open file '/root/step.py': [Errno 2]"),
                (1, "", "python3: can't open file '/root/step.py': [Errno 2]"),
            ],
        ):
            with pytest.raises(DiscoveryProbeError) as exc_info:
                await _fetch_help_text(
                    host="10.0.0.1",
                    port=22,
                    username="root",
                    credential=None,
                    command="python3 /root/step.py",
                )

        err = exc_info.value
        assert err.executed_command == "python3 /root/step.py"
        assert err.exit_code == 1
        assert err.last_attempted_command == "python3 /root/step.py --help"
        assert "can't open file" in err.summary_text
        assert "exited with code 1" in err.format_user_message()
