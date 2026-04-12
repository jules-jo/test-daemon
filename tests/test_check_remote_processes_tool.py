"""Tests for the check_remote_processes agent tool.

Sub-AC 2b-1 + Sub-AC 3: Verify the check_remote_processes tool correctly
queries and returns running process info from the remote host via SSH.

Test coverage:
    - Tool metadata (name, description, parameters, approval classification)
    - InfoRetrievalTool protocol compliance (dual calling convention)
    - ToolSpec generation and OpenAI schema serialization
    - Delegation to collision_check.check_remote_processes
    - Parameter validation (host, username, port)
    - Error handling (SSH failure, timeout, connection refused)
    - Output format (JSON with host, processes_found, processes)
    - Default port (22) when not specified
    - Custom filter_pattern parameter
    - Empty process list (no conflicts)
    - Large process list handling
    - ToolRegistry integration (register, execute, classification)
    - to_openai_schema direct serialization
    - Result serialization (to_llm_message, to_openai_tool_message)
    - Error result non-terminal status
    - Kwargs fallback calling convention
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolResultStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockProcessInfo:
    """Mock ProcessInfo matching the collision_check module's interface."""

    pid: int
    command: str


@pytest.fixture
def mock_processes() -> list[MockProcessInfo]:
    """Standard set of mock processes for testing."""
    return [
        MockProcessInfo(pid=1234, command="pytest tests/"),
        MockProcessInfo(pid=5678, command="python3 test_runner.py"),
    ]


@pytest.fixture
def empty_processes() -> list[MockProcessInfo]:
    """Empty process list."""
    return []


@pytest.fixture
def single_process() -> list[MockProcessInfo]:
    """Single detected process."""
    return [MockProcessInfo(pid=42, command="npm test")]


# ---------------------------------------------------------------------------
# Tool metadata and classification
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesToolMetadata:
    """Verify tool metadata matches spec requirements."""

    def test_tool_name(self) -> None:
        """Tool name must be 'check_remote_processes'."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        assert tool.name == "check_remote_processes"

    def test_tool_description_not_empty(self) -> None:
        """Tool must have a non-empty description for LLM context."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        assert len(tool.description) > 20

    def test_read_only_classification(self) -> None:
        """check_remote_processes is a read-only tool (no approval needed)."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        assert tool.spec.approval is ApprovalRequirement.NONE

    def test_does_not_require_human_approval(self) -> None:
        """Read-only tool must not require human approval."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        assert tool.requires_human_approval is False

    def test_spec_has_host_parameter(self) -> None:
        """Spec must define a required 'host' parameter."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        spec = tool.spec
        host_params = [p for p in spec.parameters if p.name == "host"]
        assert len(host_params) == 1
        assert host_params[0].required is True
        assert host_params[0].json_type == "string"

    def test_spec_has_username_parameter(self) -> None:
        """Spec must define a required 'username' parameter."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        spec = tool.spec
        user_params = [p for p in spec.parameters if p.name == "username"]
        assert len(user_params) == 1
        assert user_params[0].required is True
        assert user_params[0].json_type == "string"

    def test_spec_has_port_parameter_optional(self) -> None:
        """Spec must define an optional 'port' parameter with default 22."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        spec = tool.spec
        port_params = [p for p in spec.parameters if p.name == "port"]
        assert len(port_params) == 1
        assert port_params[0].required is False
        assert port_params[0].json_type == "integer"
        assert port_params[0].default == 22

    def test_spec_has_filter_pattern_parameter_optional(self) -> None:
        """Spec must define an optional 'filter_pattern' parameter."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        spec = tool.spec
        fp_params = [p for p in spec.parameters if p.name == "filter_pattern"]
        assert len(fp_params) == 1
        assert fp_params[0].required is False
        assert fp_params[0].json_type == "string"


# ---------------------------------------------------------------------------
# OpenAI schema serialization
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesOpenAISchema:
    """Verify OpenAI-compatible schema generation."""

    def test_openai_schema_structure(self) -> None:
        """Schema must follow OpenAI function-calling format."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        schema = tool.spec.to_openai_function_schema()

        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "check_remote_processes"
        assert "description" in fn
        assert "parameters" in fn

        params = fn["parameters"]
        assert params["type"] == "object"
        assert "host" in params["properties"]
        assert "username" in params["properties"]
        assert "port" in params["properties"]
        assert "host" in params["required"]
        assert "username" in params["required"]

    def test_port_not_in_required(self) -> None:
        """Port is optional and must not be in the required list."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        schema = tool.spec.to_openai_function_schema()
        required = schema["function"]["parameters"]["required"]
        assert "port" not in required


# ---------------------------------------------------------------------------
# InfoRetrievalTool protocol compliance
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesProtocol:
    """Verify the tool satisfies the InfoRetrievalTool protocol."""

    def test_has_name_property(self) -> None:
        """Tool must expose a name property."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        assert hasattr(tool, "name")
        assert isinstance(tool.name, str)

    def test_has_description_property(self) -> None:
        """Tool must expose a description property."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        assert hasattr(tool, "description")
        assert isinstance(tool.description, str)

    def test_has_parameters_schema_property(self) -> None:
        """Tool must expose a parameters_schema property."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        assert hasattr(tool, "parameters_schema")
        schema = tool.parameters_schema
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema

    def test_has_requires_human_approval_property(self) -> None:
        """Tool must expose requires_human_approval property."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        assert hasattr(tool, "requires_human_approval")
        assert tool.requires_human_approval is False

    def test_has_spec_property(self) -> None:
        """Tool must expose a spec property for ToolRegistry."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        assert hasattr(tool, "spec")
        spec = tool.spec
        assert spec.name == "check_remote_processes"

    def test_spec_is_cached(self) -> None:
        """Spec property should return the same object on repeated access."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        spec_a = tool.spec
        spec_b = tool.spec
        assert spec_a is spec_b


# ---------------------------------------------------------------------------
# Execution -- delegation to collision_check
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesExecution:
    """Verify the tool correctly delegates to collision_check."""

    @pytest.mark.asyncio
    async def test_delegates_to_collision_check(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Tool must delegate to collision_check.check_remote_processes."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "10.0.1.50",
                "username": "root",
                "port": 22,
                "_call_id": "c1",
            })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["host"] == "10.0.1.50"
        assert data["processes_found"] == 2
        assert data["processes"][0]["pid"] == 1234
        assert data["processes"][0]["command"] == "pytest tests/"
        assert data["processes"][1]["pid"] == 5678

    @pytest.mark.asyncio
    async def test_passes_credential_to_check(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Tool must resolve credentials and pass them to check_remote_processes."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_credential = MagicMock()
        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=mock_credential)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            await tool.execute({
                "host": "myhost",
                "username": "user1",
                "_call_id": "c2",
            })

        mock_resolve_fn.assert_called_once_with("myhost")
        mock_check_fn.assert_called_once()
        call_kwargs = mock_check_fn.call_args[1]
        assert call_kwargs["host"] == "myhost"
        assert call_kwargs["username"] == "user1"
        assert call_kwargs["credential"] is mock_credential

    @pytest.mark.asyncio
    async def test_default_port_22(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Port must default to 22 when not specified."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "c3",
            })

        call_kwargs = mock_check_fn.call_args[1]
        assert call_kwargs["port"] == 22

    @pytest.mark.asyncio
    async def test_custom_port(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Tool must respect a custom port value."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            await tool.execute({
                "host": "server",
                "username": "root",
                "port": 2222,
                "_call_id": "c4",
            })

        call_kwargs = mock_check_fn.call_args[1]
        assert call_kwargs["port"] == 2222

    @pytest.mark.asyncio
    async def test_empty_process_list(
        self, empty_processes: list[MockProcessInfo]
    ) -> None:
        """Tool must handle empty process list (no conflicts)."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=empty_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "c5",
            })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["processes_found"] == 0
        assert data["processes"] == []

    @pytest.mark.asyncio
    async def test_single_process(
        self, single_process: list[MockProcessInfo]
    ) -> None:
        """Tool must handle single detected process."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=single_process)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "c6",
            })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["processes_found"] == 1
        assert data["processes"][0]["pid"] == 42
        assert data["processes"][0]["command"] == "npm test"

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_host(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Tool must strip whitespace from host and username."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "  server  ",
                "username": "  root  ",
                "_call_id": "c7",
            })

        assert result.status == ToolResultStatus.SUCCESS
        mock_resolve_fn.assert_called_once_with("server")
        call_kwargs = mock_check_fn.call_args[1]
        assert call_kwargs["host"] == "server"
        assert call_kwargs["username"] == "root"

    @pytest.mark.asyncio
    async def test_filter_pattern_passed_to_check(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Tool must pass filter_pattern to the underlying check function."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "filter_pattern": "java.*selenium",
                "_call_id": "c8",
            })

        assert result.status == ToolResultStatus.SUCCESS
        call_kwargs = mock_check_fn.call_args[1]
        assert call_kwargs.get("filter_pattern") == "java.*selenium"


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesValidation:
    """Verify parameter validation and error handling."""

    @pytest.mark.asyncio
    async def test_empty_host_returns_error(self) -> None:
        """Empty host must return ERROR status."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        result = await tool.execute({
            "host": "",
            "username": "root",
            "_call_id": "v1",
        })
        assert result.status == ToolResultStatus.ERROR
        assert "host" in (result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_whitespace_only_host_returns_error(self) -> None:
        """Whitespace-only host must return ERROR status."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        result = await tool.execute({
            "host": "   ",
            "username": "root",
            "_call_id": "v2",
        })
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_empty_username_returns_error(self) -> None:
        """Empty username must return ERROR status."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        result = await tool.execute({
            "host": "server",
            "username": "",
            "_call_id": "v3",
        })
        assert result.status == ToolResultStatus.ERROR
        assert "username" in (result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_whitespace_only_username_returns_error(self) -> None:
        """Whitespace-only username must return ERROR status."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        result = await tool.execute({
            "host": "server",
            "username": "   ",
            "_call_id": "v4",
        })
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_missing_host_key_returns_error(self) -> None:
        """Missing host key in args must return ERROR status."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        result = await tool.execute({
            "username": "root",
            "_call_id": "v5",
        })
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_missing_username_key_returns_error(self) -> None:
        """Missing username key in args must return ERROR status."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        result = await tool.execute({
            "host": "server",
            "_call_id": "v6",
        })
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_call_id_present_in_result(self) -> None:
        """Result call_id must match the input call_id."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        result = await tool.execute({
            "host": "",
            "username": "root",
            "_call_id": "my-id-123",
        })
        assert result.call_id == "my-id-123"

    @pytest.mark.asyncio
    async def test_tool_name_in_result(self) -> None:
        """Result tool_name must be 'check_remote_processes'."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        result = await tool.execute({
            "host": "",
            "username": "root",
            "_call_id": "c1",
        })
        assert result.tool_name == "check_remote_processes"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesErrorHandling:
    """Verify error handling for SSH failures."""

    @pytest.mark.asyncio
    async def test_ssh_connection_failure_returns_error(self) -> None:
        """SSH connection failure must return ERROR (not raise)."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(side_effect=ConnectionRefusedError("Connection refused"))
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "unreachable",
                "username": "root",
                "_call_id": "e1",
            })

        assert result.status == ToolResultStatus.ERROR
        assert result.error_message is not None
        assert "failed" in result.error_message.lower() or "error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_ssh_timeout_returns_error(self) -> None:
        """SSH timeout must return ERROR status."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(side_effect=TimeoutError("Connection timed out"))
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "slow-server",
                "username": "root",
                "_call_id": "e2",
            })

        assert result.status == ToolResultStatus.ERROR
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_ssh_auth_failure_returns_error(self) -> None:
        """SSH authentication failure must return ERROR."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(
            side_effect=PermissionError("Authentication failed")
        )
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "secured",
                "username": "hacker",
                "_call_id": "e3",
            })

        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_credential_resolution_failure_returns_error(self) -> None:
        """Credential resolution failure must return ERROR."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_resolve_fn = MagicMock(side_effect=RuntimeError("Credentials file corrupt"))

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "e4",
            })

        assert result.status == ToolResultStatus.ERROR
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_error_does_not_raise(self) -> None:
        """Tool execute() must never raise -- errors become ToolResult."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(side_effect=RuntimeError("Unexpected"))
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            # Must NOT raise
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "e5",
            })

        assert result.status == ToolResultStatus.ERROR


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesOutput:
    """Verify output format and content."""

    @pytest.mark.asyncio
    async def test_output_is_valid_json(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Output must be valid JSON."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "o1",
            })

        data = json.loads(result.output)
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_output_contains_required_fields(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Output JSON must contain host, processes_found, and processes."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "o2",
            })

        data = json.loads(result.output)
        assert "host" in data
        assert "processes_found" in data
        assert "processes" in data
        assert isinstance(data["processes"], list)

    @pytest.mark.asyncio
    async def test_each_process_has_pid_and_command(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Each process entry must have 'pid' and 'command' keys."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "o3",
            })

        data = json.loads(result.output)
        for proc in data["processes"]:
            assert "pid" in proc
            assert "command" in proc
            assert isinstance(proc["pid"], int)
            assert isinstance(proc["command"], str)

    @pytest.mark.asyncio
    async def test_processes_found_matches_list_length(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """processes_found count must match length of processes list."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "o4",
            })

        data = json.loads(result.output)
        assert data["processes_found"] == len(data["processes"])


# ---------------------------------------------------------------------------
# Dual calling convention support
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesDualCallingConvention:
    """Verify both InfoRetrievalTool and BaseTool calling conventions work."""

    @pytest.mark.asyncio
    async def test_legacy_basemtool_convention(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Legacy convention: execute(args_dict) with _call_id in args."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "legacy1",
            })

        assert result.status == ToolResultStatus.SUCCESS
        assert result.call_id == "legacy1"

    @pytest.mark.asyncio
    async def test_info_retrieval_keyword_convention(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """InfoRetrievalTool convention: execute(call_id=..., args=...)."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute(
                call_id="kw1",
                args={"host": "server", "username": "root"},
            )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.call_id == "kw1"

    @pytest.mark.asyncio
    async def test_positional_convention(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Positional convention: execute(call_id, args_dict)."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute(
                "pos1",
                {"host": "server", "username": "root"},
            )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.call_id == "pos1"

    @pytest.mark.asyncio
    async def test_default_call_id_when_missing(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """When _call_id is missing from legacy dict, use tool name as default."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
            })

        assert result.status == ToolResultStatus.SUCCESS
        assert result.call_id == "check_remote_processes"

    @pytest.mark.asyncio
    async def test_kwargs_fallback_convention(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Fallback convention: execute(host=..., username=..., _call_id=...).

        Covers lines 222-224 -- the else branch that handles
        keyword-only arguments without call_id or dict positional.
        """
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute(
                host="server",
                username="root",
                _call_id="fallback1",
            )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.call_id == "fallback1"

    @pytest.mark.asyncio
    async def test_kwargs_fallback_without_call_id(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Fallback convention without _call_id uses tool name default."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute(
                host="server",
                username="root",
            )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.call_id == "check_remote_processes"


# ---------------------------------------------------------------------------
# ToolRegistry integration
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesRegistryIntegration:
    """Verify the tool works correctly within a ToolRegistry."""

    def _make_tool(self) -> "CheckRemoteProcessesTool":
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )
        return CheckRemoteProcessesTool()

    def test_register_in_registry(self) -> None:
        """Tool must be registrable in a ToolRegistry."""
        from jules_daemon.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        tool = self._make_tool()
        registry.register(tool)
        assert "check_remote_processes" in registry

    def test_listed_as_read_only(self) -> None:
        """Tool must appear in the read-only tool list."""
        from jules_daemon.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(self._make_tool())
        read_only_names = [t.name for t in registry.list_read_only_tools()]
        assert "check_remote_processes" in read_only_names

    def test_not_listed_as_approval_required(self) -> None:
        """Tool must not appear in the approval-required list."""
        from jules_daemon.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(self._make_tool())
        approval_names = [
            t.name for t in registry.list_approval_required_tools()
        ]
        assert "check_remote_processes" not in approval_names

    def test_requires_approval_returns_false(self) -> None:
        """Registry.requires_approval must return False for this tool."""
        from jules_daemon.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(self._make_tool())
        assert not registry.requires_approval("check_remote_processes")

    def test_openai_schema_in_registry(self) -> None:
        """Registry OpenAI schema export must include this tool."""
        from jules_daemon.agent.tool_registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(self._make_tool())
        schemas = registry.to_openai_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "check_remote_processes" in names

    @pytest.mark.asyncio
    async def test_execute_via_registry(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Tool must be executable through the ToolRegistry.execute path."""
        from jules_daemon.agent.tool_registry import ToolRegistry
        from jules_daemon.agent.tool_types import ToolCall

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        registry = ToolRegistry()
        registry.register(self._make_tool())

        call = ToolCall(
            call_id="reg1",
            tool_name="check_remote_processes",
            arguments={"host": "10.0.1.50", "username": "root"},
        )

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await registry.execute(call)

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["host"] == "10.0.1.50"
        assert data["processes_found"] == 2

    @pytest.mark.asyncio
    async def test_registry_execute_missing_args_returns_error(self) -> None:
        """Registry execute with missing required args must return error."""
        from jules_daemon.agent.tool_registry import ToolRegistry
        from jules_daemon.agent.tool_types import ToolCall

        registry = ToolRegistry()
        registry.register(self._make_tool())

        call = ToolCall(
            call_id="reg2",
            tool_name="check_remote_processes",
            arguments={"host": "server"},  # missing username
        )

        result = await registry.execute(call)
        assert result.status == ToolResultStatus.ERROR


# ---------------------------------------------------------------------------
# Direct OpenAI schema serialization (via InfoRetrievalTool.to_openai_schema)
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesDirectSchema:
    """Verify to_openai_schema() produces valid OpenAI function-calling format."""

    def test_to_openai_schema_structure(self) -> None:
        """to_openai_schema must produce the correct structure."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "check_remote_processes"
        assert func["description"]
        assert func["parameters"]["type"] == "object"
        assert "host" in func["parameters"]["properties"]
        assert "username" in func["parameters"]["properties"]
        assert "port" in func["parameters"]["properties"]
        assert "filter_pattern" in func["parameters"]["properties"]

    def test_to_openai_schema_matches_spec_schema(self) -> None:
        """to_openai_schema output must match the ToolSpec schema output."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        direct = tool.to_openai_schema()
        via_spec = tool.spec.to_openai_function_schema()

        assert direct["function"]["name"] == via_spec["function"]["name"]
        assert direct["function"]["description"] == via_spec["function"]["description"]

    def test_to_tool_spec_parameters_match(self) -> None:
        """to_tool_spec parameters must match the parameters_schema."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        spec = tool.to_tool_spec()

        param_names = {p.name for p in spec.parameters}
        assert param_names == {"host", "username", "port", "filter_pattern"}

        host_param = next(p for p in spec.parameters if p.name == "host")
        assert host_param.required is True
        assert host_param.json_type == "string"

        port_param = next(p for p in spec.parameters if p.name == "port")
        assert port_param.required is False
        assert port_param.json_type == "integer"
        assert port_param.default == 22


# ---------------------------------------------------------------------------
# Result serialization and status checks
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesResultSerialization:
    """Verify ToolResult serialization from tool execution."""

    @pytest.mark.asyncio
    async def test_success_result_to_openai_tool_message(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Successful result must serialize to an OpenAI tool message."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "ser1",
            })

        msg = result.to_openai_tool_message()
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "ser1"
        # Content should be valid JSON (the success output)
        data = json.loads(msg["content"])
        assert data["processes_found"] == 2

    @pytest.mark.asyncio
    async def test_error_result_to_openai_tool_message(self) -> None:
        """Error result must serialize with ERROR prefix."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        result = await tool.execute({
            "host": "",
            "username": "root",
            "_call_id": "ser2",
        })

        msg = result.to_openai_tool_message()
        assert msg["role"] == "tool"
        assert msg["content"].startswith("ERROR:")

    @pytest.mark.asyncio
    async def test_success_result_to_llm_message(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Success result to_llm_message must include tool name and status."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "ser3",
            })

        llm_msg = result.to_llm_message()
        assert "check_remote_processes" in llm_msg
        assert "success" in llm_msg.lower()

    @pytest.mark.asyncio
    async def test_error_result_is_not_terminal(self) -> None:
        """Error results from a read-only tool must NOT be terminal."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(
            side_effect=ConnectionRefusedError("refused")
        )
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "ser4",
            })

        assert result.is_error
        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_success_result_is_success(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Successful execution must set is_success=True."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "ser5",
            })

        assert result.is_success
        assert not result.is_error
        assert not result.is_denied
        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_result_to_dict_roundtrip(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """ToolResult.to_dict should produce a serializable dict."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "ser6",
            })

        d = result.to_dict()
        assert d["call_id"] == "ser6"
        assert d["tool_name"] == "check_remote_processes"
        assert d["status"] == "success"
        assert d["error_message"] is None
        # Output should be valid JSON
        json.loads(d["output"])


# ---------------------------------------------------------------------------
# Large process list handling
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesLargeOutput:
    """Verify behavior with large numbers of detected processes."""

    @pytest.mark.asyncio
    async def test_many_processes_all_included(self) -> None:
        """All detected processes must be included in the output."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        large_list = [
            MockProcessInfo(pid=i, command=f"test_process_{i}")
            for i in range(100)
        ]

        mock_check_fn = AsyncMock(return_value=large_list)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "busy-server",
                "username": "root",
                "_call_id": "lg1",
            })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["processes_found"] == 100
        assert len(data["processes"]) == 100

    @pytest.mark.asyncio
    async def test_process_pids_preserved_in_order(self) -> None:
        """Process PIDs and commands must be preserved in order."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        ordered_list = [
            MockProcessInfo(pid=10, command="first"),
            MockProcessInfo(pid=20, command="second"),
            MockProcessInfo(pid=30, command="third"),
        ]

        mock_check_fn = AsyncMock(return_value=ordered_list)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "lg2",
            })

        data = json.loads(result.output)
        assert data["processes"][0]["pid"] == 10
        assert data["processes"][0]["command"] == "first"
        assert data["processes"][1]["pid"] == 20
        assert data["processes"][2]["pid"] == 30


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestCheckRemoteProcessesEdgeCases:
    """Verify behavior at edge boundaries."""

    @pytest.mark.asyncio
    async def test_port_coerced_to_int(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """String port value should be coerced to int."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "server",
                "username": "root",
                "port": "2222",
                "_call_id": "ec1",
            })

        assert result.status == ToolResultStatus.SUCCESS
        call_kwargs = mock_check_fn.call_args[1]
        assert call_kwargs["port"] == 2222
        assert isinstance(call_kwargs["port"], int)

    @pytest.mark.asyncio
    async def test_filter_pattern_none_when_omitted(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """filter_pattern should be None when not provided."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "ec2",
            })

        call_kwargs = mock_check_fn.call_args[1]
        assert call_kwargs["filter_pattern"] is None

    @pytest.mark.asyncio
    async def test_credential_passed_is_none_when_not_found(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """When resolve_ssh_credentials returns None, None is passed through."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            await tool.execute({
                "host": "server",
                "username": "root",
                "_call_id": "ec3",
            })

        call_kwargs = mock_check_fn.call_args[1]
        assert call_kwargs["credential"] is None

    @pytest.mark.asyncio
    async def test_host_with_special_characters(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Host with valid special chars (IP, FQDN) should work."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            result = await tool.execute({
                "host": "test-server.example.com",
                "username": "deploy-user",
                "_call_id": "ec4",
            })

        assert result.status == ToolResultStatus.SUCCESS
        data = json.loads(result.output)
        assert data["host"] == "test-server.example.com"

    @pytest.mark.asyncio
    async def test_empty_call_id_uses_sentinel(self) -> None:
        """Empty call_id should trigger base class validation error."""
        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        tool = CheckRemoteProcessesTool()
        result = await tool.execute(
            call_id="",
            args={"host": "server", "username": "root"},
        )

        # Base class uses "unknown" as sentinel for empty call_id
        assert result.status == ToolResultStatus.ERROR
        assert result.call_id == "unknown"

    @pytest.mark.asyncio
    async def test_concurrent_executions_independent(
        self, mock_processes: list[MockProcessInfo]
    ) -> None:
        """Multiple concurrent executions must not interfere."""
        import asyncio

        from jules_daemon.agent.tools.check_remote_processes import (
            CheckRemoteProcessesTool,
        )

        mock_check_fn = AsyncMock(return_value=mock_processes)
        mock_resolve_fn = MagicMock(return_value=None)

        tool = CheckRemoteProcessesTool()

        with patch(
            "jules_daemon.agent.tools.check_remote_processes._check_remote_processes",
            mock_check_fn,
        ), patch(
            "jules_daemon.agent.tools.check_remote_processes._resolve_ssh_credentials",
            mock_resolve_fn,
        ):
            results = await asyncio.gather(
                tool.execute({
                    "host": "server-a",
                    "username": "user-a",
                    "_call_id": "cc1",
                }),
                tool.execute({
                    "host": "server-b",
                    "username": "user-b",
                    "_call_id": "cc2",
                }),
            )

        assert results[0].call_id == "cc1"
        assert results[1].call_id == "cc2"
        assert results[0].status == ToolResultStatus.SUCCESS
        assert results[1].status == ToolResultStatus.SUCCESS

        data_a = json.loads(results[0].output)
        data_b = json.loads(results[1].output)
        assert data_a["host"] == "server-a"
        assert data_b["host"] == "server-b"
