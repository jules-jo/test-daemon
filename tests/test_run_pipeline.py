"""Focused tests for the SSH run pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from jules_daemon.execution.run_pipeline import execute_run
from jules_daemon.ssh.errors import SSHAuthenticationError


class TestExecuteRun:
    @pytest.mark.asyncio
    async def test_missing_password_auth_failure_includes_guidance(
        self,
        tmp_path: Path,
    ) -> None:
        with patch(
            "jules_daemon.execution.run_pipeline.resolve_ssh_credentials",
            return_value=None,
        ), patch(
            "jules_daemon.execution.run_pipeline.current_run_io.write",
        ), patch(
            "jules_daemon.execution.run_pipeline.promote_run",
        ), patch(
            "jules_daemon.execution.run_pipeline.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=SSHAuthenticationError(
                "Authentication failed for root@10.0.0.10:22: bad credentials"
            ),
        ):
            result = await execute_run(
                target_host="10.0.0.10",
                target_user="root",
                command="pytest -q",
                target_port=22,
                wiki_root=tmp_path,
            )

        assert result.success is False
        assert result.error is not None
        assert "Authentication failed for root@10.0.0.10:22" in result.error
        assert "ssh_credentials.yaml" in result.error
        assert "JULES_SSH_PASSWORD" in result.error
