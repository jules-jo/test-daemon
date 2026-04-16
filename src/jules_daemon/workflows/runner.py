"""Deterministic helpers for executable workflow steps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from string import Formatter
from typing import Any, Mapping

from jules_daemon.wiki.test_knowledge import TestKnowledge

__all__ = [
    "WorkflowExecutionPlan",
    "WorkflowExecutionStep",
    "build_required_command_args",
    "build_workflow_step_id",
    "normalize_step_name",
    "render_command_pattern",
]


def normalize_step_name(value: str) -> str:
    """Normalize a logical workflow step name for matching."""
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _command_placeholders(command_pattern: str) -> tuple[str, ...]:
    """Return unique named placeholders referenced by a command pattern."""
    placeholders: list[str] = []
    seen: set[str] = set()
    for _literal, field_name, _format_spec, _conversion in Formatter().parse(
        command_pattern
    ):
        if field_name is None:
            continue
        cleaned = field_name.strip()
        if not cleaned or cleaned.isdigit() or cleaned in seen:
            continue
        seen.add(cleaned)
        placeholders.append(cleaned)
    return tuple(placeholders)


def build_required_command_args(knowledge: TestKnowledge) -> tuple[str, ...]:
    """Return the ordered arg names needed to materialize a command."""
    ordered: list[str] = []
    seen: set[str] = set()
    for arg_name in knowledge.required_args + _command_placeholders(
        knowledge.command_pattern
    ):
        cleaned = arg_name.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return tuple(ordered)


def render_command_pattern(
    *,
    command_pattern: str,
    args: Mapping[str, Any],
) -> str:
    """Render a command pattern with explicit placeholder validation."""
    required = _command_placeholders(command_pattern)
    missing = tuple(name for name in required if name not in args)
    if missing:
        raise ValueError(
            "Missing command arguments: " + ", ".join(missing)
        )

    rendered_args = {
        key: str(value)
        for key, value in args.items()
        if value is not None
    }
    return command_pattern.format_map(rendered_args)


def build_workflow_step_id(index: int, step_name: str) -> str:
    """Build a durable per-workflow step identifier."""
    normalized = normalize_step_name(step_name) or f"step-{index:02d}"
    return f"step-{index:02d}-{normalized}"


@dataclass(frozen=True)
class WorkflowExecutionStep:
    """Concrete executable step in a workflow plan."""

    step_id: str
    step_name: str
    phase: str
    test_slug: str
    command: str
    command_pattern: str
    required_args: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowExecutionPlan:
    """Executable workflow prepared before background execution starts."""

    workflow_id: str
    request_text: str
    target_host: str
    target_user: str
    target_port: int
    steps: tuple[WorkflowExecutionStep, ...]
