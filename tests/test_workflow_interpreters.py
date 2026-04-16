"""Tests for workflow step output interpreters."""

from __future__ import annotations

from jules_daemon.workflows.interpreters.generic import (
    interpret_generic_step_output,
)
from jules_daemon.workflows.interpreters.registry import StepInterpreterRegistry


def test_generic_interpreter_parses_pytest_like_output() -> None:
    parsed = interpret_generic_step_output(
        raw_output=(
            "tests/test_demo.py::test_ok PASSED\n"
            "tests/test_demo.py::test_bad FAILED\n"
        ),
        command="pytest -q",
        success=False,
        active=False,
    )

    assert parsed is not None
    assert parsed["state"] == "completed_failure"
    assert parsed["summary_fields"]["framework"] == "pytest"
    assert parsed["summary_fields"]["passed"] == 1
    assert parsed["summary_fields"]["failed"] == 1
    assert parsed["failure_detected"] is True
    assert "test output summary" in parsed["progress_message"]


def test_generic_interpreter_falls_back_to_last_meaningful_line() -> None:
    parsed = interpret_generic_step_output(
        raw_output="setup started\nstill warming hardware\n",
        command="python3 /root/setup_step.py",
        success=None,
        active=True,
    )

    assert parsed is not None
    assert parsed["state"] == "running"
    assert parsed["summary_fields"] == {}
    assert parsed["progress_message"] == "still warming hardware"


def test_registry_routes_to_generic_interpreter() -> None:
    registry = StepInterpreterRegistry()

    parsed = registry.interpret(
        step_name="setup-step",
        command="python3 /root/setup_step.py",
        raw_output="tests/test_setup.py::test_ready PASSED\n",
        success=True,
        active=False,
    )

    assert parsed is not None
    assert parsed["state"] == "completed_success"
    assert parsed["summary_fields"]["passed"] == 1
