"""Deterministic workflow planning from structured test knowledge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from jules_daemon.wiki.test_knowledge import TestKnowledge


@dataclass(frozen=True)
class PlannedWorkflowStep:
    """One planned workflow step."""

    name: str
    phase: str


@dataclass(frozen=True)
class WorkflowQuestion:
    """A user-facing workflow question produced by planning or preflight."""

    kind: str
    prompt: str


@dataclass(frozen=True)
class TestWorkflowPlan:
    """Resolved workflow plan for one test request."""

    test_slug: str
    request_text: str
    workflow_steps: tuple[PlannedWorkflowStep, ...]
    prerequisite_steps: tuple[PlannedWorkflowStep, ...] = ()
    main_steps: tuple[PlannedWorkflowStep, ...] = ()
    artifact_requirements: tuple[str, ...] = ()
    when_missing_artifact_ask: str = ""
    success_criteria: str = ""
    failure_criteria: str = ""
    notes: tuple[str, ...] = ()


def resolve_test_workflow(
    *,
    request_text: str,
    knowledge: TestKnowledge,
) -> TestWorkflowPlan:
    """Resolve a generic workflow plan from structured test knowledge.

    Current behavior:
    - If ``workflow_steps`` is present, treat the final step as the primary
      step and all earlier steps as prerequisites.
    - Otherwise, build a single-step plan around ``test_slug``.
    - If explicit ``prerequisites`` are present but ``workflow_steps`` are
      not, model them as prerequisite steps ahead of the primary step.
    """
    ordered_step_names: tuple[str, ...]
    if knowledge.workflow_steps:
        ordered_step_names = knowledge.workflow_steps
    elif knowledge.prerequisites:
        ordered_step_names = knowledge.prerequisites + (knowledge.test_slug,)
    else:
        ordered_step_names = (knowledge.test_slug,)

    planned_steps: list[PlannedWorkflowStep] = []
    prerequisite_steps: list[PlannedWorkflowStep] = []
    main_steps: list[PlannedWorkflowStep] = []

    for index, step_name in enumerate(ordered_step_names):
        phase = "main" if index == len(ordered_step_names) - 1 else "prerequisite"
        step = PlannedWorkflowStep(name=step_name, phase=phase)
        planned_steps.append(step)
        if phase == "main":
            main_steps.append(step)
        else:
            prerequisite_steps.append(step)

    notes: list[str] = []
    if knowledge.prerequisites:
        notes.append(
            "Configured prerequisites: " + ", ".join(knowledge.prerequisites)
        )
    if knowledge.artifact_requirements:
        notes.append(
            "Artifact requirements: "
            + ", ".join(knowledge.artifact_requirements)
        )

    return TestWorkflowPlan(
        test_slug=knowledge.test_slug,
        request_text=request_text,
        workflow_steps=tuple(planned_steps),
        prerequisite_steps=tuple(prerequisite_steps),
        main_steps=tuple(main_steps),
        artifact_requirements=knowledge.artifact_requirements,
        when_missing_artifact_ask=knowledge.when_missing_artifact_ask,
        success_criteria=knowledge.success_criteria,
        failure_criteria=knowledge.failure_criteria,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class WorkflowPreflightDecision:
    """Deterministic decision about whether a plan is ready to run."""

    missing_artifacts: tuple[str, ...] = ()
    unknown_artifacts: tuple[str, ...] = ()
    ready_to_run: bool = True
    requires_user_confirmation: bool = False
    question: WorkflowQuestion | None = None
    notes: tuple[str, ...] = ()


def evaluate_workflow_preflight(
    *,
    plan: TestWorkflowPlan,
    artifact_presence: Mapping[str, bool | None],
) -> WorkflowPreflightDecision:
    """Evaluate artifact requirements for a workflow plan.

    This is intentionally deterministic. Remote artifact probing can be
    layered on later; this function only interprets the presence map.
    """
    missing: list[str] = []
    unknown: list[str] = []
    for artifact_name in plan.artifact_requirements:
        presence = artifact_presence.get(artifact_name)
        if presence is True:
            continue
        if presence is False:
            missing.append(artifact_name)
            continue
        unknown.append(artifact_name)

    if not missing and not unknown:
        return WorkflowPreflightDecision(
            missing_artifacts=(),
            unknown_artifacts=(),
            ready_to_run=True,
            requires_user_confirmation=False,
        )

    if missing and plan.when_missing_artifact_ask:
        prompt = plan.when_missing_artifact_ask
    elif missing and plan.prerequisite_steps:
        prereq_names = ", ".join(step.name for step in plan.prerequisite_steps)
        prompt = (
            "Required artifacts are missing: "
            + ", ".join(missing)
            + f". Do you want me to run {prereq_names} first?"
        )
    elif missing:
        prompt = (
            "Required artifacts are missing: "
            + ", ".join(missing)
            + ". Do you want me to continue anyway?"
        )
    elif plan.prerequisite_steps:
        prereq_names = ", ".join(step.name for step in plan.prerequisite_steps)
        prompt = (
            "Jules could not verify required artifacts automatically: "
            + ", ".join(unknown)
            + f". Do you want me to run {prereq_names} first?"
        )
    else:
        prompt = (
            "Jules could not verify required artifacts automatically: "
            + ", ".join(unknown)
            + ". Do you want me to continue anyway?"
        )

    return WorkflowPreflightDecision(
        missing_artifacts=tuple(missing),
        unknown_artifacts=tuple(unknown),
        ready_to_run=False,
        requires_user_confirmation=True,
        question=WorkflowQuestion(
            kind="missing_artifact" if missing else "unverified_artifact",
            prompt=prompt,
        ),
        notes=(
            (
                "Preflight blocked on missing artifacts: "
                + ", ".join(missing)
            )
            if missing
            else (
                "Preflight requires confirmation because artifacts "
                "could not be verified automatically: "
                + ", ".join(unknown)
            ),
        ),
    )
