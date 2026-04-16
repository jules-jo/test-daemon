"""Wiki-backed persistence for workflow and workflow-step state."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument
from jules_daemon.workflows.models import (
    ArtifactState,
    ArtifactStatus,
    WorkflowRecord,
    WorkflowStatus,
    WorkflowStepRecord,
    WorkflowStepStatus,
)

_WORKFLOWS_DIR = "pages/daemon/workflows"
_WORKFLOW_STEPS_DIR = "pages/daemon/workflow-steps"
_WORKFLOW_TAGS = ["daemon", "workflow"]
_WORKFLOW_TYPE = "workflow-state"
_WORKFLOW_STEP_TAGS = ["daemon", "workflow-step"]
_WORKFLOW_STEP_TYPE = "workflow-step-state"


def _slug_component(value: str) -> str:
    """Convert an identifier into a filesystem-safe component."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = normalized.strip("-")
    return cleaned or "item"


def _ensure_parent_directory(path: Path) -> None:
    """Create a file's parent directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _datetime_to_iso(value: datetime | None) -> str | None:
    """Convert a datetime to an ISO string."""
    return value.isoformat() if value is not None else None


def _iso_to_datetime(value: str | None) -> datetime | None:
    """Parse an ISO datetime string."""
    return datetime.fromisoformat(value) if value else None


def workflow_file_path(wiki_root: Path, workflow_id: str) -> Path:
    """Return the canonical file path for one workflow."""
    return wiki_root / _WORKFLOWS_DIR / f"workflow-{_slug_component(workflow_id)}.md"


def workflow_step_file_path(
    wiki_root: Path,
    workflow_id: str,
    step_id: str,
) -> Path:
    """Return the canonical file path for one workflow step."""
    return (
        wiki_root
        / _WORKFLOW_STEPS_DIR
        / (
            f"workflow-{_slug_component(workflow_id)}"
            f"--{_slug_component(step_id)}.md"
        )
    )


def _artifact_to_dict(artifact: ArtifactState) -> dict[str, Any]:
    """Serialize one artifact state."""
    return {
        "name": artifact.name,
        "status": artifact.status.value,
        "details": artifact.details,
        "checked_at": _datetime_to_iso(artifact.checked_at),
    }


def _dict_to_artifact(data: dict[str, Any]) -> ArtifactState:
    """Deserialize one artifact state."""
    return ArtifactState(
        name=str(data["name"]),
        status=ArtifactStatus(data.get("status", ArtifactStatus.UNKNOWN.value)),
        details=data.get("details"),
        checked_at=(
            _iso_to_datetime(data.get("checked_at"))
            or datetime.now(timezone.utc)
        ),
    )


def _workflow_to_frontmatter(workflow: WorkflowRecord) -> dict[str, Any]:
    """Serialize a workflow record into YAML-safe frontmatter."""
    return {
        "tags": list(_WORKFLOW_TAGS),
        "type": _WORKFLOW_TYPE,
        "workflow_id": workflow.workflow_id,
        "workflow_kind": workflow.workflow_kind,
        "request_text": workflow.request_text,
        "status": workflow.status.value,
        "run_id": workflow.run_id,
        "current_step_id": workflow.current_step_id,
        "target_host": workflow.target_host,
        "target_user": workflow.target_user,
        "summary": workflow.summary,
        "error": workflow.error,
        "artifact_states": [
            _artifact_to_dict(artifact) for artifact in workflow.artifact_states
        ],
        "started_at": _datetime_to_iso(workflow.started_at),
        "completed_at": _datetime_to_iso(workflow.completed_at),
        "created": _datetime_to_iso(workflow.created_at),
        "updated": _datetime_to_iso(workflow.updated_at),
    }


def _frontmatter_to_workflow(fm: dict[str, Any]) -> WorkflowRecord:
    """Deserialize frontmatter into a workflow record."""
    raw_artifacts = fm.get("artifact_states") or []
    artifact_states = tuple(
        _dict_to_artifact(item)
        for item in raw_artifacts
        if isinstance(item, dict)
    )
    created_at = _iso_to_datetime(fm.get("created")) or datetime.now(timezone.utc)
    updated_at = _iso_to_datetime(fm.get("updated")) or created_at
    return WorkflowRecord(
        workflow_id=str(fm["workflow_id"]),
        request_text=str(fm["request_text"]),
        workflow_kind=str(fm.get("workflow_kind", "test_run")),
        status=WorkflowStatus(fm.get("status", WorkflowStatus.PLANNING.value)),
        run_id=fm.get("run_id"),
        current_step_id=fm.get("current_step_id"),
        target_host=fm.get("target_host"),
        target_user=fm.get("target_user"),
        summary=fm.get("summary"),
        error=fm.get("error"),
        artifact_states=artifact_states,
        started_at=_iso_to_datetime(fm.get("started_at")),
        completed_at=_iso_to_datetime(fm.get("completed_at")),
        created_at=created_at,
        updated_at=updated_at,
    )


def _step_to_frontmatter(step: WorkflowStepRecord) -> dict[str, Any]:
    """Serialize a workflow step into YAML-safe frontmatter."""
    return {
        "tags": list(_WORKFLOW_STEP_TAGS),
        "type": _WORKFLOW_STEP_TYPE,
        "workflow_id": step.workflow_id,
        "step_id": step.step_id,
        "name": step.name,
        "kind": step.kind,
        "status": step.status.value,
        "run_id": step.run_id,
        "command": step.command,
        "target_host": step.target_host,
        "target_user": step.target_user,
        "exit_code": step.exit_code,
        "summary": step.summary,
        "error": step.error,
        "last_output_line": step.last_output_line,
        "parsed_status": step.parsed_status,
        "started_at": _datetime_to_iso(step.started_at),
        "completed_at": _datetime_to_iso(step.completed_at),
        "created": _datetime_to_iso(step.created_at),
        "updated": _datetime_to_iso(step.updated_at),
    }


def _frontmatter_to_step(fm: dict[str, Any]) -> WorkflowStepRecord:
    """Deserialize frontmatter into a workflow step record."""
    created_at = _iso_to_datetime(fm.get("created")) or datetime.now(timezone.utc)
    updated_at = _iso_to_datetime(fm.get("updated")) or created_at
    exit_code_raw = fm.get("exit_code")
    exit_code = int(exit_code_raw) if isinstance(exit_code_raw, int) else None
    return WorkflowStepRecord(
        workflow_id=str(fm["workflow_id"]),
        step_id=str(fm["step_id"]),
        name=str(fm.get("name", fm["step_id"])),
        kind=str(fm.get("kind", "run")),
        status=WorkflowStepStatus(
            fm.get("status", WorkflowStepStatus.PENDING.value)
        ),
        run_id=fm.get("run_id"),
        command=fm.get("command"),
        target_host=fm.get("target_host"),
        target_user=fm.get("target_user"),
        exit_code=exit_code,
        summary=fm.get("summary"),
        error=fm.get("error"),
        last_output_line=fm.get("last_output_line"),
        parsed_status=(
            fm.get("parsed_status")
            if isinstance(fm.get("parsed_status"), dict)
            else None
        ),
        started_at=_iso_to_datetime(fm.get("started_at")),
        completed_at=_iso_to_datetime(fm.get("completed_at")),
        created_at=created_at,
        updated_at=updated_at,
    )


def _build_workflow_body(workflow: WorkflowRecord) -> str:
    """Render a human-readable workflow markdown body."""
    lines = [
        f"# Workflow {workflow.workflow_id}",
        "",
        f"*Workflow kind: {workflow.workflow_kind}*",
        "",
        "## Request",
        "",
        workflow.request_text,
        "",
        "## State",
        "",
        f"- **Status:** {workflow.status.value}",
    ]
    if workflow.run_id:
        lines.append(f"- **Run ID:** {workflow.run_id}")
    if workflow.current_step_id:
        lines.append(f"- **Current Step:** {workflow.current_step_id}")
    if workflow.target_host:
        lines.append(f"- **Target Host:** {workflow.target_host}")
    if workflow.target_user:
        lines.append(f"- **Target User:** {workflow.target_user}")
    if workflow.started_at:
        lines.append(f"- **Started:** {workflow.started_at.isoformat()}")
    if workflow.completed_at:
        lines.append(f"- **Completed:** {workflow.completed_at.isoformat()}")
    lines.append("")

    if workflow.artifact_states:
        lines.extend(["## Artifacts", ""])
        for artifact in workflow.artifact_states:
            detail = f" -- {artifact.details}" if artifact.details else ""
            lines.append(f"- **{artifact.name}:** {artifact.status.value}{detail}")
        lines.append("")

    if workflow.summary:
        lines.extend(["## Summary", "", workflow.summary, ""])

    if workflow.error:
        lines.extend(["## Error", "", "```", workflow.error, "```", ""])

    return "\n".join(lines)


def _build_step_body(step: WorkflowStepRecord) -> str:
    """Render a human-readable workflow step markdown body."""
    lines = [
        f"# Workflow Step {step.step_id}",
        "",
        f"*Workflow: {step.workflow_id}*",
        "",
        "## State",
        "",
        f"- **Name:** {step.name}",
        f"- **Kind:** {step.kind}",
        f"- **Status:** {step.status.value}",
    ]
    if step.run_id:
        lines.append(f"- **Run ID:** {step.run_id}")
    if step.target_host:
        lines.append(f"- **Target Host:** {step.target_host}")
    if step.target_user:
        lines.append(f"- **Target User:** {step.target_user}")
    if step.exit_code is not None:
        lines.append(f"- **Exit Code:** {step.exit_code}")
    if step.started_at:
        lines.append(f"- **Started:** {step.started_at.isoformat()}")
    if step.completed_at:
        lines.append(f"- **Completed:** {step.completed_at.isoformat()}")
    lines.append("")

    if step.command:
        lines.extend(["## Command", "", f"`{step.command}`", ""])
    if step.last_output_line:
        lines.extend(["## Last Output", "", "```", step.last_output_line, "```", ""])
    if step.parsed_status:
        lines.extend(["## Parsed Status", ""])
        progress_message = step.parsed_status.get("progress_message")
        if isinstance(progress_message, str) and progress_message.strip():
            lines.append(f"- **Progress:** {progress_message}")
        state = step.parsed_status.get("state")
        if isinstance(state, str) and state.strip():
            lines.append(f"- **State:** {state}")
        summary_fields = step.parsed_status.get("summary_fields")
        if isinstance(summary_fields, dict) and summary_fields:
            for key, value in summary_fields.items():
                lines.append(f"- **{key}:** {value}")
        lines.append("")
    if step.summary:
        lines.extend(["## Summary", "", step.summary, ""])
    if step.error:
        lines.extend(["## Error", "", "```", step.error, "```", ""])

    return "\n".join(lines)


def save_workflow(wiki_root: Path, workflow: WorkflowRecord) -> Path:
    """Persist one workflow record with an atomic write."""
    file_path = workflow_file_path(wiki_root, workflow.workflow_id)
    _ensure_parent_directory(file_path)
    doc = WikiDocument(
        frontmatter=_workflow_to_frontmatter(workflow),
        body=_build_workflow_body(workflow),
    )
    content = frontmatter.serialize(doc)
    tmp_path = file_path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(str(tmp_path), str(file_path))
    return file_path


def save_step(wiki_root: Path, step: WorkflowStepRecord) -> Path:
    """Persist one workflow-step record with an atomic write."""
    file_path = workflow_step_file_path(wiki_root, step.workflow_id, step.step_id)
    _ensure_parent_directory(file_path)
    doc = WikiDocument(
        frontmatter=_step_to_frontmatter(step),
        body=_build_step_body(step),
    )
    content = frontmatter.serialize(doc)
    tmp_path = file_path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(str(tmp_path), str(file_path))
    return file_path


def read_workflow(wiki_root: Path, workflow_id: str) -> WorkflowRecord | None:
    """Load one workflow record."""
    file_path = workflow_file_path(wiki_root, workflow_id)
    if not file_path.exists():
        return None
    doc = frontmatter.parse(file_path.read_text(encoding="utf-8"))
    return _frontmatter_to_workflow(doc.frontmatter)


def read_step(
    wiki_root: Path,
    workflow_id: str,
    step_id: str,
) -> WorkflowStepRecord | None:
    """Load one workflow-step record."""
    file_path = workflow_step_file_path(wiki_root, workflow_id, step_id)
    if not file_path.exists():
        return None
    doc = frontmatter.parse(file_path.read_text(encoding="utf-8"))
    return _frontmatter_to_step(doc.frontmatter)


def list_steps(wiki_root: Path, workflow_id: str) -> tuple[WorkflowStepRecord, ...]:
    """List all steps for one workflow, sorted by creation time."""
    steps_dir = wiki_root / _WORKFLOW_STEPS_DIR
    if not steps_dir.exists():
        return ()

    prefix = f"workflow-{_slug_component(workflow_id)}--"
    records: list[WorkflowStepRecord] = []
    for file_path in steps_dir.glob(f"{prefix}*.md"):
        try:
            doc = frontmatter.parse(file_path.read_text(encoding="utf-8"))
            records.append(_frontmatter_to_step(doc.frontmatter))
        except Exception:
            continue
    return tuple(sorted(records, key=lambda item: (item.created_at, item.step_id)))


def list_workflows(
    wiki_root: Path,
    *,
    limit: int | None = None,
) -> tuple[WorkflowRecord, ...]:
    """List workflows sorted by most recently updated."""
    workflows_dir = wiki_root / _WORKFLOWS_DIR
    if not workflows_dir.exists():
        return ()

    records: list[WorkflowRecord] = []
    for file_path in workflows_dir.glob("workflow-*.md"):
        try:
            doc = frontmatter.parse(file_path.read_text(encoding="utf-8"))
            records.append(_frontmatter_to_workflow(doc.frontmatter))
        except Exception:
            continue
    records.sort(key=lambda item: (item.updated_at, item.created_at), reverse=True)
    if limit is not None:
        records = records[:limit]
    return tuple(records)


def read_latest_workflow(wiki_root: Path) -> WorkflowRecord | None:
    """Return the most recently updated workflow, if any."""
    workflows = list_workflows(wiki_root, limit=1)
    return workflows[0] if workflows else None
