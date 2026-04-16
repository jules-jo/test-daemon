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
        test_slug="lt-test",
        command_pattern="python3 /root/lt.py",
        workflow_steps=("calibration", "lt_test"),
        prerequisites=("calibration",),
        artifact_requirements=("calibration_file",),
        when_missing_artifact_ask=(
            "There is no calibration file. Do you want me to run calibration first?"
        ),
        success_criteria="LT summary reports zero failures.",
        failure_criteria="Calibration fails or LT reports any failure.",
    )
    base.update(overrides)
    return TestKnowledge(**base)


def test_resolve_test_workflow_uses_configured_steps() -> None:
    plan = resolve_test_workflow(
        request_text="run lt test",
        knowledge=_knowledge(),
    )

    assert plan.test_slug == "lt-test"
    assert [step.name for step in plan.workflow_steps] == [
        "calibration",
        "lt_test",
    ]
    assert [step.phase for step in plan.workflow_steps] == [
        "prerequisite",
        "main",
    ]
    assert [step.name for step in plan.prerequisite_steps] == ["calibration"]
    assert [step.name for step in plan.main_steps] == ["lt_test"]
    assert plan.artifact_requirements == ("calibration_file",)
    assert "Configured prerequisites: calibration" in plan.notes


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
        request_text="run lt test",
        knowledge=_knowledge(),
    )
    decision = evaluate_workflow_preflight(
        plan=plan,
        artifact_presence={"calibration_file": True},
    )

    assert decision.ready_to_run is True
    assert decision.requires_user_confirmation is False
    assert decision.missing_artifacts == ()
    assert decision.unknown_artifacts == ()
    assert decision.question is None


def test_evaluate_workflow_preflight_asks_configured_prompt_when_missing() -> None:
    plan = resolve_test_workflow(
        request_text="run lt test",
        knowledge=_knowledge(),
    )
    decision = evaluate_workflow_preflight(
        plan=plan,
        artifact_presence={"calibration_file": False},
    )

    assert decision.ready_to_run is False
    assert decision.requires_user_confirmation is True
    assert decision.missing_artifacts == ("calibration_file",)
    assert decision.question is not None
    assert (
        decision.question.prompt
        == "There is no calibration file. Do you want me to run calibration first?"
    )


def test_evaluate_workflow_preflight_asks_when_artifacts_are_unverified() -> None:
    plan = resolve_test_workflow(
        request_text="run lt test",
        knowledge=_knowledge(when_missing_artifact_ask=""),
    )
    decision = evaluate_workflow_preflight(
        plan=plan,
        artifact_presence={},
    )

    assert decision.ready_to_run is False
    assert decision.requires_user_confirmation is True
    assert decision.missing_artifacts == ()
    assert decision.unknown_artifacts == ("calibration_file",)
    assert decision.question is not None
    assert "could not verify required artifacts automatically" in (
        decision.question.prompt.lower()
    )
    assert "run calibration first" in decision.question.prompt.lower()


def test_evaluate_workflow_preflight_builds_generic_prompt_without_custom_one() -> None:
    plan = resolve_test_workflow(
        request_text="run lt test",
        knowledge=_knowledge(when_missing_artifact_ask=""),
    )
    decision = evaluate_workflow_preflight(
        plan=plan,
        artifact_presence={"calibration_file": False},
    )

    assert decision.question is not None
    assert "Do you want me to run calibration first?" in decision.question.prompt
