"""Read-side workflow status assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jules_daemon.workflows.store import (
    list_steps,
    read_latest_workflow,
    read_workflow,
)


def _artifact_snapshot(artifact: Any) -> dict[str, Any]:
    """Convert one artifact state into a plain dict."""
    return {
        "name": artifact.name,
        "status": artifact.status.value,
        "details": artifact.details,
        "checked_at": artifact.checked_at.isoformat(),
    }


def _step_snapshot(step: Any) -> dict[str, Any]:
    """Convert one workflow step into a plain dict."""
    snapshot: dict[str, Any] = {
        "step_id": step.step_id,
        "name": step.name,
        "kind": step.kind,
        "status": step.status.value,
    }
    if step.run_id:
        snapshot["run_id"] = step.run_id
    if step.command:
        snapshot["command"] = step.command
    if step.target_host:
        snapshot["target_host"] = step.target_host
    if step.target_user:
        snapshot["target_user"] = step.target_user
    if step.exit_code is not None:
        snapshot["exit_code"] = step.exit_code
    if step.summary:
        snapshot["summary"] = step.summary
    if step.error:
        snapshot["error"] = step.error
    if step.last_output_line:
        snapshot["last_output_line"] = step.last_output_line
    if step.started_at:
        snapshot["started_at"] = step.started_at.isoformat()
    if step.completed_at:
        snapshot["completed_at"] = step.completed_at.isoformat()
    return snapshot


def build_workflow_status(
    wiki_root: Path,
    workflow_id: str,
) -> dict[str, Any] | None:
    """Build a user-facing status snapshot for one workflow."""
    workflow = read_workflow(wiki_root, workflow_id)
    if workflow is None:
        return None

    steps = list_steps(wiki_root, workflow_id)
    active_step = None
    if workflow.current_step_id:
        for step in steps:
            if step.step_id == workflow.current_step_id:
                active_step = step
                break
    if active_step is None and steps:
        active_step = steps[-1]

    snapshot: dict[str, Any] = {
        "workflow_id": workflow.workflow_id,
        "workflow_kind": workflow.workflow_kind,
        "request_text": workflow.request_text,
        "status": workflow.status.value,
        "step_count": len(steps),
    }
    if workflow.run_id:
        snapshot["run_id"] = workflow.run_id
    if workflow.current_step_id:
        snapshot["current_step_id"] = workflow.current_step_id
    if workflow.target_host:
        snapshot["target_host"] = workflow.target_host
    if workflow.target_user:
        snapshot["target_user"] = workflow.target_user
    if workflow.summary:
        snapshot["summary"] = workflow.summary
    if workflow.error:
        snapshot["error"] = workflow.error
    if workflow.started_at:
        snapshot["started_at"] = workflow.started_at.isoformat()
    if workflow.completed_at:
        snapshot["completed_at"] = workflow.completed_at.isoformat()
    if workflow.artifact_states:
        snapshot["artifact_states"] = [
            _artifact_snapshot(artifact)
            for artifact in workflow.artifact_states
        ]
    if active_step is not None:
        snapshot["active_step"] = _step_snapshot(active_step)
    if steps:
        snapshot["steps"] = [_step_snapshot(step) for step in steps]
    return snapshot


def build_latest_workflow_status(wiki_root: Path) -> dict[str, Any] | None:
    """Build a status snapshot for the most recently updated workflow."""
    workflow = read_latest_workflow(wiki_root)
    if workflow is None:
        return None
    return build_workflow_status(wiki_root, workflow.workflow_id)

