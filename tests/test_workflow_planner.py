"""Tests for workflow knowledge planning and preflight."""

from __future__ import annotations

from jules_daemon.wiki.test_knowledge import TestKnowledge
from jules_daemon.workflows.planner import (
    evaluate_workflow_preflight,
    resolve_test_workflow,
)


def _knowledge(**overrides: object) -> TestKnowledge:
    """Build a workflow-aware test knowledge record."""
    base = dict(
        test_slug="main-check",
        command_pattern="python3 /root/main_check.py",
        workflow_steps=("setup-step", "main_check"),
        prerequisites=("setup-step",),
        artifact_requirements=("setup_ready_file",),
        when_missing_artifact_ask=(
            "There is no setup file. Do you want me to run the setup step first?"
        ),
        success_criteria="Main check summary reports zero failures.",
        failure_criteria="Setup fails or main check reports any failure.",
    )
    base.update(overrides)
    return TestKnowledge(**base)


def test_resolve_test_workflow_uses_configured_steps() -> None:
    plan = resolve_test_workflow(
        request_text="run main check",
        knowledge=_knowledge(),
    )

    assert plan.test_slug == "main-check"
    assert [step.name for step in plan.workflow_steps] == [
        "setup-step",
        "main_check",
    ]
    assert [step.phase for step in plan.workflow_steps] == [
        "prerequisite",
        "main",
    ]
    assert [step.name for step in plan.prerequisite_steps] == ["setup-step"]
    assert [step.name for step in plan.main_steps] == ["main_check"]
    assert plan.artifact_requirements == ("setup_ready_file",)
    assert "Configured prerequisites: setup-step" in plan.notes


def test_resolve_test_workflow_defaults_to_single_step() -> None:
    plan = resolve_test_workflow(
        request_text="run smoke test",
        knowledge=_knowledge(
            test_slug="smoke-test",
            command_pattern="./smoke.sh",
            workflow_steps=(),
            prerequisites=(),
            artifact_requirements=(),
            when_missing_artifact_ask="",
            success_criteria="",
            failure_criteria="",
        ),
    )

    assert [step.name for step in plan.workflow_steps] == ["smoke-test"]
    assert plan.prerequisite_steps == ()
    assert [step.name for step in plan.main_steps] == ["smoke-test"]


def test_evaluate_workflow_preflight_ready_when_artifacts_present() -> None:
    plan = resolve_test_workflow(
        request_text="run main check",
        knowledge=_knowledge(),
    )
    decision = evaluate_workflow_preflight(
        plan=plan,
        artifact_presence={"setup_ready_file": True},
    )

    assert decision.ready_to_run is True
    assert decision.requires_user_confirmation is False
    assert decision.missing_artifacts == ()
    assert decision.unknown_artifacts == ()
    assert decision.question is None


def test_evaluate_workflow_preflight_asks_configured_prompt_when_missing() -> None:
    plan = resolve_test_workflow(
        request_text="run main check",
        knowledge=_knowledge(),
    )
    decision = evaluate_workflow_preflight(
        plan=plan,
        artifact_presence={"setup_ready_file": False},
    )

    assert decision.ready_to_run is False
    assert decision.requires_user_confirmation is True
    assert decision.missing_artifacts == ("setup_ready_file",)
    assert decision.question is not None
    assert (
        decision.question.prompt
        == "There is no setup file. Do you want me to run the setup step first?"
    )


def test_evaluate_workflow_preflight_asks_when_artifacts_are_unverified() -> None:
    plan = resolve_test_workflow(
        request_text="run main check",
        knowledge=_knowledge(when_missing_artifact_ask=""),
    )
    decision = evaluate_workflow_preflight(
        plan=plan,
        artifact_presence={},
    )

    assert decision.ready_to_run is False
    assert decision.requires_user_confirmation is True
    assert decision.missing_artifacts == ()
    assert decision.unknown_artifacts == ("setup_ready_file",)
    assert decision.question is not None
    assert "could not verify required artifacts automatically" in (
        decision.question.prompt.lower()
    )
    assert "run setup-step first" in decision.question.prompt.lower()


def test_evaluate_workflow_preflight_builds_generic_prompt_without_custom_one() -> None:
    plan = resolve_test_workflow(
        request_text="run main check",
        knowledge=_knowledge(when_missing_artifact_ask=""),
    )
    decision = evaluate_workflow_preflight(
        plan=plan,
        artifact_presence={"setup_ready_file": False},
    )

    assert decision.question is not None
    assert "Do you want me to run setup-step first?" in decision.question.prompt
