"""Tests for workflow models, persistence, and read-side status."""

from __future__ import annotations

from jules_daemon.wiki.layout import initialize_wiki
from jules_daemon.workflows.models import WorkflowRecord, WorkflowStepRecord
from jules_daemon.workflows.status import (
    build_latest_workflow_status,
    build_workflow_status,
)
from jules_daemon.workflows.store import (
    list_steps,
    read_latest_workflow,
    read_step,
    read_workflow,
    save_step,
    save_workflow,
)


def test_workflow_round_trip(tmp_path) -> None:
    initialize_wiki(tmp_path)
    workflow = WorkflowRecord(
        workflow_id="run-abc123",
        request_text="run main check",
    ).with_running(
        current_step_id="primary-run",
        run_id="run-abc123",
        target_host="10.0.0.10",
        target_user="root",
    )
    step = WorkflowStepRecord(
        workflow_id="run-abc123",
        step_id="primary-run",
        name="primary run",
    ).with_running(
        run_id="run-abc123",
        command="python3 step.py --target 5",
        target_host="10.0.0.10",
        target_user="root",
    )

    save_workflow(tmp_path, workflow)
    save_step(tmp_path, step)

    loaded_workflow = read_workflow(tmp_path, "run-abc123")
    loaded_step = read_step(tmp_path, "run-abc123", "primary-run")

    assert loaded_workflow is not None
    assert loaded_workflow.status.value == "running"
    assert loaded_workflow.request_text == "run main check"
    assert loaded_step is not None
    assert loaded_step.command == "python3 step.py --target 5"
    assert loaded_step.status.value == "running"


def test_workflow_status_snapshot_includes_active_step(tmp_path) -> None:
    initialize_wiki(tmp_path)
    workflow = WorkflowRecord(
        workflow_id="run-xyz789",
        request_text="run smoke test",
    ).with_running(
        current_step_id="primary-run",
        run_id="run-xyz789",
        target_host="host.example.com",
        target_user="deploy",
    )
    step = WorkflowStepRecord(
        workflow_id="run-xyz789",
        step_id="primary-run",
        name="primary run",
    ).with_running(
        run_id="run-xyz789",
        command="pytest -q",
        target_host="host.example.com",
        target_user="deploy",
    )
    save_workflow(tmp_path, workflow)
    save_step(tmp_path, step)

    snapshot = build_workflow_status(tmp_path, "run-xyz789")

    assert snapshot is not None
    assert snapshot["workflow_id"] == "run-xyz789"
    assert snapshot["status"] == "running"
    assert snapshot["active_step"]["step_id"] == "primary-run"
    assert snapshot["active_step"]["command"] == "pytest -q"
    assert snapshot["step_count"] == 1
    assert len(snapshot["steps"]) == 1


def test_latest_workflow_prefers_most_recent_update(tmp_path) -> None:
    initialize_wiki(tmp_path)
    older = WorkflowRecord(
        workflow_id="run-older",
        request_text="run older test",
    )
    newer = WorkflowRecord(
        workflow_id="run-newer",
        request_text="run newer test",
    ).with_running(
        current_step_id="primary-run",
        run_id="run-newer",
    )

    save_workflow(tmp_path, older)
    save_workflow(tmp_path, newer)

    latest = read_latest_workflow(tmp_path)
    latest_snapshot = build_latest_workflow_status(tmp_path)

    assert latest is not None
    assert latest.workflow_id == "run-newer"
    assert latest_snapshot is not None
    assert latest_snapshot["workflow_id"] == "run-newer"
    assert list_steps(tmp_path, "run-older") == ()
