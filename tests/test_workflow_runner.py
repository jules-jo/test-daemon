"""Tests for deterministic workflow runner helpers."""

from __future__ import annotations

import pytest

from jules_daemon.wiki.test_knowledge import TestKnowledge
from jules_daemon.workflows.runner import (
    build_required_command_args,
    build_workflow_step_id,
    normalize_step_name,
    render_command_pattern,
)


def _knowledge(**overrides: object) -> TestKnowledge:
    base = dict(
        test_slug="main-check",
        command_pattern="python3 /root/main_check.py --target {target}",
        required_args=("target",),
    )
    base.update(overrides)
    return TestKnowledge(**base)


def test_build_required_command_args_includes_template_placeholders() -> None:
    knowledge = _knowledge(
        command_pattern=(
            "python3 /root/main_check.py --target {target} --env {env}"
        ),
        required_args=("target",),
    )

    assert build_required_command_args(knowledge) == ("target", "env")


def test_render_command_pattern_requires_all_placeholders() -> None:
    with pytest.raises(ValueError, match="Missing command arguments: env"):
        render_command_pattern(
            command_pattern="python3 /root/main_check.py --target {target} --env {env}",
            args={"target": "5"},
        )


def test_render_command_pattern_formats_string_values() -> None:
    rendered = render_command_pattern(
        command_pattern="python3 /root/main_check.py --target {target}",
        args={"target": 5},
    )

    assert rendered == "python3 /root/main_check.py --target 5"


def test_build_workflow_step_id_is_stable_and_slugged() -> None:
    assert build_workflow_step_id(2, "Main Check") == "step-02-main-check"
    assert normalize_step_name("Setup_File") == "setup-file"
