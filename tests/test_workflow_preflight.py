"""Tests for remote workflow artifact preflight helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jules_daemon.workflows.models import ArtifactStatus
from jules_daemon.workflows.preflight import inspect_workflow_artifacts


class TestInspectWorkflowArtifacts:
    """Artifact inspection should preserve path vs non-path semantics."""

    @pytest.mark.asyncio
    async def test_non_path_requirements_remain_unknown(self) -> None:
        states = await inspect_workflow_artifacts(
            host="10.0.0.10",
            port=22,
            username="root",
            artifact_requirements=("calibration_file",),
        )

        assert len(states) == 1
        assert states[0].name == "calibration_file"
        assert states[0].status is ArtifactStatus.UNKNOWN
        assert "cannot verify" in (states[0].details or "").lower()

    @pytest.mark.asyncio
    async def test_path_requirements_use_remote_probe(self) -> None:
        with patch(
            "jules_daemon.workflows.preflight._probe_artifact_paths_via_ssh",
            return_value={
                "/tmp/calibration.json": ArtifactStatus.MISSING,
                "/tmp/existing.txt": ArtifactStatus.PRESENT,
            },
        ) as mock_probe:
            states = await inspect_workflow_artifacts(
                host="10.0.0.10",
                port=22,
                username="root",
                artifact_requirements=(
                    "/tmp/calibration.json",
                    "/tmp/existing.txt",
                ),
            )

        mock_probe.assert_called_once()
        by_name = {state.name: state for state in states}
        assert by_name["/tmp/calibration.json"].status is ArtifactStatus.MISSING
        assert by_name["/tmp/existing.txt"].status is ArtifactStatus.PRESENT
