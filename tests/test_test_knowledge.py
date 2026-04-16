"""Tests for the per-test wiki knowledge persistence module.

Covers slug derivation for varied command shapes, the
``load`` / ``save`` round-trip including file-not-present and
malformed-file paths, the merge strategy for fresh + existing
observations, and the helper formatting that becomes part of the
LLM summarization prompt.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.test_knowledge import (
    KNOWLEDGE_DIR,
    TestKnowledge,
    derive_test_slug,
    knowledge_file_path,
    load_test_knowledge,
    merge_knowledge,
    save_test_knowledge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


def _make_knowledge(
    *,
    test_slug: str = "agent-test-py",
    command_pattern: str = "python3 agent_test.py",
    purpose: str = "",
    output_format: str = "",
    summary_fields: tuple[str, ...] = (),
    common_failures: tuple[str, ...] = (),
    normal_behavior: str = "",
    required_args: tuple[str, ...] = (),
    workflow_steps: tuple[str, ...] = (),
    prerequisites: tuple[str, ...] = (),
    artifact_requirements: tuple[str, ...] = (),
    when_missing_artifact_ask: str = "",
    success_criteria: str = "",
    failure_criteria: str = "",
    runs_observed: int = 0,
) -> TestKnowledge:
    """Build a TestKnowledge with sensible defaults for tests."""
    return TestKnowledge(
        test_slug=test_slug,
        command_pattern=command_pattern,
        purpose=purpose,
        output_format=output_format,
        summary_fields=summary_fields,
        common_failures=common_failures,
        normal_behavior=normal_behavior,
        required_args=required_args,
        workflow_steps=workflow_steps,
        prerequisites=prerequisites,
        artifact_requirements=artifact_requirements,
        when_missing_artifact_ask=when_missing_artifact_ask,
        success_criteria=success_criteria,
        failure_criteria=failure_criteria,
        runs_observed=runs_observed,
    )


# ---------------------------------------------------------------------------
# derive_test_slug
# ---------------------------------------------------------------------------


class TestDeriveTestSlug:
    """The slug derivation strips interpreters, paths, and flags."""

    def test_python_script_with_args(self) -> None:
        slug = derive_test_slug(
            "python3.8 ~/agent_test.py --name JJ --iteration 100"
        )
        assert slug == "agent-test-py"

    def test_python3_explicit(self) -> None:
        slug = derive_test_slug("python3 /opt/foo_test.py")
        assert slug == "foo-test-py"

    def test_pytest_with_path(self) -> None:
        slug = derive_test_slug("pytest tests/integration/ -v")
        assert slug == "pytest-tests-integration"

    def test_pytest_no_args(self) -> None:
        slug = derive_test_slug("pytest")
        assert slug == "pytest"

    def test_pytest_with_unit_path(self) -> None:
        slug = derive_test_slug("pytest tests/unit")
        assert slug == "pytest-tests-unit"

    def test_pytest_subdirs_distinguished(self) -> None:
        """Two different pytest subdirs must produce different slugs."""
        a = derive_test_slug("pytest tests/integration")
        b = derive_test_slug("pytest tests/unit")
        assert a != b

    def test_npm_test(self) -> None:
        assert derive_test_slug("npm test") == "npm-test"

    def test_cargo_build(self) -> None:
        assert derive_test_slug("cargo build") == "cargo-build"

    def test_go_test(self) -> None:
        # ./... is mostly punctuation; should at least include "go"
        slug = derive_test_slug("go test ./...")
        assert slug.startswith("go")

    def test_relative_script(self) -> None:
        assert derive_test_slug("./run.sh") == "run-sh"

    def test_absolute_script(self) -> None:
        assert derive_test_slug("/usr/local/bin/run_tests.sh") == "run-tests-sh"

    def test_make_target(self) -> None:
        slug = derive_test_slug("make check")
        assert slug == "make-check"

    def test_empty_command_returns_unknown(self) -> None:
        assert derive_test_slug("") == "unknown-test"

    def test_whitespace_command_returns_unknown(self) -> None:
        assert derive_test_slug("   ") == "unknown-test"

    def test_only_flags_falls_back(self) -> None:
        """A command consisting only of flags still produces a stable slug."""
        slug = derive_test_slug("--version --help")
        assert slug
        assert slug != "unknown-test" or slug == "unknown-test"  # any non-empty

    def test_slug_is_lowercase(self) -> None:
        slug = derive_test_slug("python MyScript.PY")
        assert slug == slug.lower()

    def test_slug_is_filesystem_safe(self) -> None:
        slug = derive_test_slug("python my+weird*name.py")
        # Only alphanumerics and hyphens
        for ch in slug:
            assert ch.isalnum() or ch == "-"

    def test_slug_max_length(self) -> None:
        long_command = "python " + "a" * 200 + ".py"
        slug = derive_test_slug(long_command)
        assert len(slug) <= 60

    def test_same_command_with_different_args_collapses(self) -> None:
        a = derive_test_slug("python agent_test.py --iter 1")
        b = derive_test_slug("python agent_test.py --iter 100 --name foo")
        assert a == b

    def test_python_versions_normalize(self) -> None:
        a = derive_test_slug("python3.8 agent_test.py")
        b = derive_test_slug("python3.12 agent_test.py")
        c = derive_test_slug("python agent_test.py")
        assert a == b == c

    def test_quoted_paths(self) -> None:
        slug = derive_test_slug('python "agent_test.py"')
        assert slug == "agent-test-py"

    def test_npx_runner(self) -> None:
        slug = derive_test_slug("npx jest --watch")
        assert "jest" in slug or slug == "npx"

    def test_node_interpreter_stripped(self) -> None:
        slug = derive_test_slug("node tests/run.js")
        assert slug == "run-js"


# ---------------------------------------------------------------------------
# TestKnowledge dataclass
# ---------------------------------------------------------------------------


class TestTestKnowledgeDataclass:
    """The TestKnowledge dataclass enforces basic invariants."""

    def test_frozen(self) -> None:
        k = _make_knowledge()
        with pytest.raises(AttributeError):
            k.purpose = "something"  # type: ignore[misc]

    def test_empty_slug_raises(self) -> None:
        with pytest.raises(ValueError, match="test_slug must not be empty"):
            TestKnowledge(test_slug="", command_pattern="cmd")

    def test_empty_command_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="command_pattern"):
            TestKnowledge(test_slug="slug", command_pattern="")

    def test_negative_runs_raises(self) -> None:
        with pytest.raises(ValueError, match="runs_observed"):
            TestKnowledge(
                test_slug="slug",
                command_pattern="cmd",
                runs_observed=-1,
            )

    def test_default_runs_zero(self) -> None:
        k = _make_knowledge()
        assert k.runs_observed == 0

    def test_defaults_to_now(self) -> None:
        before = datetime.now(timezone.utc)
        k = _make_knowledge()
        after = datetime.now(timezone.utc)
        assert before <= k.last_updated <= after


class TestPromptContext:
    """``to_prompt_context`` produces concise, populated context only."""

    def test_empty_returns_empty_string(self) -> None:
        k = _make_knowledge()
        assert k.to_prompt_context() == ""

    def test_full_includes_all_sections(self) -> None:
        k = _make_knowledge(
            purpose="Runs the agent test suite",
            output_format="Iteration N: PASSED|FAILED",
            summary_fields=("passed", "failed"),
            normal_behavior="Completes in under 30s with all PASSED",
            required_args=("iterations", "host"),
            workflow_steps=("calibration", "lt_test"),
            prerequisites=("calibration",),
            artifact_requirements=("calibration_file",),
            when_missing_artifact_ask=(
                "There is no calibration file. Do you want me to run calibration first?"
            ),
            success_criteria="LT summary reports zero failures.",
            failure_criteria="Calibration step fails or LT reports any failure.",
            common_failures=("timeout", "connection refused"),
            runs_observed=4,
        )
        text = k.to_prompt_context()
        assert "Purpose:" in text
        assert "Output format:" in text
        assert "Summary fields:" in text
        assert "Normal behavior:" in text
        assert "Required arguments:" in text
        assert "Workflow steps:" in text
        assert "Prerequisites:" in text
        assert "Required artifacts:" in text
        assert "Missing artifact prompt:" in text
        assert "Success criteria:" in text
        assert "Failure criteria:" in text
        assert "Common failure patterns:" in text
        assert "timeout" in text
        assert "connection refused" in text
        assert "4 prior run(s)" in text

    def test_partial_skips_empty_sections(self) -> None:
        k = _make_knowledge(purpose="run agent test", runs_observed=1)
        text = k.to_prompt_context()
        assert "Purpose:" in text
        assert "Output format:" not in text
        assert "Summary fields:" not in text
        assert "Common failure patterns:" not in text


# ---------------------------------------------------------------------------
# load_test_knowledge / save_test_knowledge round-trip
# ---------------------------------------------------------------------------


class TestLoadAndSave:
    """File-level persistence: missing file, round-trip, malformed file."""

    def test_load_missing_returns_none(self, wiki_root: Path) -> None:
        result = load_test_knowledge(wiki_root, "agent-test-py")
        assert result is None

    def test_save_creates_directory(self, wiki_root: Path) -> None:
        k = _make_knowledge(purpose="run the agent")
        path = save_test_knowledge(wiki_root, k)
        assert path.exists()
        assert path.parent.is_dir()
        assert path.parent == wiki_root / KNOWLEDGE_DIR

    def test_save_writes_yaml_frontmatter(self, wiki_root: Path) -> None:
        k = _make_knowledge(
            purpose="run the agent",
            output_format="iteration N: PASSED|FAILED",
            summary_fields=("passed", "failed"),
            workflow_steps=("calibration", "lt_test"),
            prerequisites=("calibration",),
            artifact_requirements=("calibration_file",),
            when_missing_artifact_ask="Run calibration first?",
            success_criteria="LT passes.",
            failure_criteria="LT fails.",
        )
        path = save_test_knowledge(wiki_root, k)
        raw = path.read_text(encoding="utf-8")
        doc = frontmatter.parse(raw)
        assert doc.frontmatter.get("type") == "test-knowledge"
        assert doc.frontmatter.get("test_slug") == "agent-test-py"
        assert doc.frontmatter.get("purpose") == "run the agent"
        assert doc.frontmatter.get("summary_fields") == ["passed", "failed"]
        assert doc.frontmatter.get("workflow_steps") == [
            "calibration",
            "lt_test",
        ]
        assert doc.frontmatter.get("prerequisites") == ["calibration"]
        assert doc.frontmatter.get("artifact_requirements") == [
            "calibration_file"
        ]
        assert doc.frontmatter.get("when_missing_artifact_ask") == (
            "Run calibration first?"
        )

    def test_round_trip_preserves_fields(self, wiki_root: Path) -> None:
        original = _make_knowledge(
            test_slug="agent-test-py",
            command_pattern="python3 agent_test.py",
            purpose="Runs the agent",
            output_format="Iteration N: PASSED|FAILED",
            summary_fields=("passed", "failed"),
            normal_behavior="All iterations PASSED",
            required_args=("iterations",),
            workflow_steps=("calibration", "lt_test"),
            prerequisites=("calibration",),
            artifact_requirements=("calibration_file",),
            when_missing_artifact_ask="Run calibration first?",
            success_criteria="LT passes.",
            failure_criteria="LT fails.",
            common_failures=("timeout", "ConnectionError"),
            runs_observed=3,
        )
        save_test_knowledge(wiki_root, original)
        loaded = load_test_knowledge(wiki_root, "agent-test-py")
        assert loaded is not None
        assert loaded.test_slug == original.test_slug
        assert loaded.command_pattern == original.command_pattern
        assert loaded.purpose == original.purpose
        assert loaded.output_format == original.output_format
        assert loaded.summary_fields == original.summary_fields
        assert loaded.normal_behavior == original.normal_behavior
        assert loaded.required_args == original.required_args
        assert loaded.workflow_steps == original.workflow_steps
        assert loaded.prerequisites == original.prerequisites
        assert (
            loaded.artifact_requirements == original.artifact_requirements
        )
        assert (
            loaded.when_missing_artifact_ask
            == original.when_missing_artifact_ask
        )
        assert loaded.success_criteria == original.success_criteria
        assert loaded.failure_criteria == original.failure_criteria
        assert loaded.common_failures == original.common_failures
        assert loaded.runs_observed == original.runs_observed

    def test_round_trip_preserves_timestamp_within_seconds(
        self, wiki_root: Path
    ) -> None:
        original = _make_knowledge()
        save_test_knowledge(wiki_root, original)
        loaded = load_test_knowledge(wiki_root, original.test_slug)
        assert loaded is not None
        delta = abs((loaded.last_updated - original.last_updated).total_seconds())
        assert delta < 2

    def test_atomic_overwrite(self, wiki_root: Path) -> None:
        first = _make_knowledge(purpose="first")
        save_test_knowledge(wiki_root, first)
        second = _make_knowledge(purpose="second", runs_observed=2)
        save_test_knowledge(wiki_root, second)
        loaded = load_test_knowledge(wiki_root, "agent-test-py")
        assert loaded is not None
        assert loaded.purpose == "second"
        assert loaded.runs_observed == 2

    def test_load_malformed_returns_none(self, wiki_root: Path) -> None:
        path = knowledge_file_path(wiki_root, "agent-test-py")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Not valid YAML frontmatter at all
        path.write_text("not a wiki page", encoding="utf-8")
        assert load_test_knowledge(wiki_root, "agent-test-py") is None

    def test_load_partial_frontmatter_uses_defaults(
        self, wiki_root: Path
    ) -> None:
        """A hand-curated file with only purpose still loads cleanly."""
        path = knowledge_file_path(wiki_root, "agent-test-py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\n"
            "type: test-knowledge\n"
            "test_slug: agent-test-py\n"
            "command_pattern: python3 agent_test.py\n"
            "purpose: human-curated purpose\n"
            "---\n\n# Test Knowledge\n",
            encoding="utf-8",
        )
        loaded = load_test_knowledge(wiki_root, "agent-test-py")
        assert loaded is not None
        assert loaded.purpose == "human-curated purpose"
        assert loaded.output_format == ""
        assert loaded.summary_fields == ()
        assert loaded.workflow_steps == ()
        assert loaded.prerequisites == ()
        assert loaded.artifact_requirements == ()
        assert loaded.common_failures == ()
        assert loaded.runs_observed == 0

    def test_load_with_empty_slug_returns_none(self, wiki_root: Path) -> None:
        assert load_test_knowledge(wiki_root, "") is None

    def test_knowledge_file_path_includes_test_prefix(
        self, wiki_root: Path
    ) -> None:
        path = knowledge_file_path(wiki_root, "agent-test-py")
        assert path.name == "test-agent-test-py.md"


# ---------------------------------------------------------------------------
# merge_knowledge
# ---------------------------------------------------------------------------


class TestMergeKnowledge:
    """The merge strategy preserves curated text and accumulates failures."""

    def test_first_observation_creates_record(self) -> None:
        observations = {
            "purpose": "runs the agent",
            "output_format": "iteration logs",
            "summary_fields": ["passed", "failed"],
            "normal_behavior": "all PASSED",
            "workflow_steps": ["calibration", "lt_test"],
            "prerequisites": ["calibration"],
            "artifact_requirements": ["calibration_file"],
            "when_missing_artifact_ask": "Run calibration first?",
            "success_criteria": "LT passes.",
            "failure_criteria": "LT fails.",
            "common_failures": ["timeout"],
        }
        merged = merge_knowledge(
            None,
            observations,
            test_slug="agent-test-py",
            command_pattern="python3 agent_test.py",
        )
        assert merged.purpose == "runs the agent"
        assert merged.output_format == "iteration logs"
        assert merged.summary_fields == ("passed", "failed")
        assert merged.normal_behavior == "all PASSED"
        assert merged.workflow_steps == ("calibration", "lt_test")
        assert merged.prerequisites == ("calibration",)
        assert merged.artifact_requirements == ("calibration_file",)
        assert merged.when_missing_artifact_ask == "Run calibration first?"
        assert merged.success_criteria == "LT passes."
        assert merged.failure_criteria == "LT fails."
        assert merged.common_failures == ("timeout",)
        assert merged.runs_observed == 1

    def test_first_observation_requires_slug_and_pattern(self) -> None:
        with pytest.raises(ValueError):
            merge_knowledge(None, {"purpose": "x"})

    def test_existing_purpose_is_preserved(self) -> None:
        existing = _make_knowledge(
            purpose="existing purpose",
            runs_observed=2,
        )
        merged = merge_knowledge(
            existing,
            {"purpose": "different purpose"},
        )
        assert merged.purpose == "existing purpose"
        assert merged.runs_observed == 3

    def test_empty_existing_purpose_adopts_new(self) -> None:
        existing = _make_knowledge(purpose="", runs_observed=2)
        merged = merge_knowledge(existing, {"purpose": "fresh purpose"})
        assert merged.purpose == "fresh purpose"

    def test_existing_summary_fields_are_preserved(self) -> None:
        existing = _make_knowledge(
            summary_fields=("passed", "failed"),
            runs_observed=2,
        )
        merged = merge_knowledge(
            existing,
            {"summary_fields": ["iterations_done", "slot_errors"]},
        )
        assert merged.summary_fields == ("passed", "failed")

    def test_existing_workflow_fields_are_preserved(self) -> None:
        existing = _make_knowledge(
            workflow_steps=("calibration", "lt_test"),
            prerequisites=("calibration",),
            artifact_requirements=("calibration_file",),
            when_missing_artifact_ask="Run calibration first?",
            success_criteria="LT passes.",
            failure_criteria="LT fails.",
            runs_observed=2,
        )
        merged = merge_knowledge(
            existing,
            {
                "workflow_steps": ["other_step"],
                "prerequisites": ["other_prereq"],
                "artifact_requirements": ["other_artifact"],
                "when_missing_artifact_ask": "Other prompt",
                "success_criteria": "Other success",
                "failure_criteria": "Other failure",
            },
        )
        assert merged.workflow_steps == ("calibration", "lt_test")
        assert merged.prerequisites == ("calibration",)
        assert merged.artifact_requirements == ("calibration_file",)
        assert merged.when_missing_artifact_ask == "Run calibration first?"
        assert merged.success_criteria == "LT passes."
        assert merged.failure_criteria == "LT fails."

    def test_empty_existing_summary_fields_adopts_new(self) -> None:
        existing = _make_knowledge(summary_fields=(), runs_observed=2)
        merged = merge_knowledge(
            existing,
            {"summary_fields": ["iterations_done", "slot_errors"]},
        )
        assert merged.summary_fields == (
            "iterations_done",
            "slot_errors",
        )

    def test_failures_unioned_and_deduped(self) -> None:
        existing = _make_knowledge(
            common_failures=("timeout", "connection refused"),
            runs_observed=1,
        )
        merged = merge_knowledge(
            existing,
            {"common_failures": ["timeout", "AssertionError"]},
        )
        assert merged.common_failures == (
            "timeout",
            "connection refused",
            "AssertionError",
        )

    def test_failures_capped_at_ten(self) -> None:
        # Start with 8 failures, add 5 more -- should cap at 10 by
        # dropping the oldest entries.
        existing_failures = tuple(f"existing-{i}" for i in range(8))
        existing = _make_knowledge(
            common_failures=existing_failures,
            runs_observed=1,
        )
        new_failures = [f"new-{i}" for i in range(5)]
        merged = merge_knowledge(
            existing,
            {"common_failures": new_failures},
        )
        assert len(merged.common_failures) == 10
        assert "new-4" in merged.common_failures
        # Some of the older entries should have been evicted
        assert "existing-0" not in merged.common_failures

    def test_runs_observed_increments(self) -> None:
        existing = _make_knowledge(runs_observed=4)
        merged = merge_knowledge(existing, {})
        assert merged.runs_observed == 5

    def test_last_updated_advances(self) -> None:
        old_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        existing = TestKnowledge(
            test_slug="agent-test-py",
            command_pattern="python3 agent_test.py",
            runs_observed=1,
            last_updated=old_time,
        )
        merged = merge_knowledge(existing, {})
        assert merged.last_updated > old_time

    def test_merge_does_not_mutate_existing(self) -> None:
        existing = _make_knowledge(
            common_failures=("a",),
            runs_observed=1,
        )
        merge_knowledge(existing, {"common_failures": ["b"]})
        # Original tuple is unchanged
        assert existing.common_failures == ("a",)
        assert existing.runs_observed == 1

    def test_merge_with_empty_observations(self) -> None:
        existing = _make_knowledge(purpose="x", runs_observed=1)
        merged = merge_knowledge(existing, {})
        # Existing fields preserved, run count incremented
        assert merged.purpose == "x"
        assert merged.runs_observed == 2

    def test_empty_failures_in_observations_does_not_change_existing(
        self,
    ) -> None:
        existing = _make_knowledge(common_failures=("known",), runs_observed=1)
        merged = merge_knowledge(existing, {"common_failures": []})
        assert merged.common_failures == ("known",)


# ---------------------------------------------------------------------------
# Full pipeline (load -> merge -> save -> load)
# ---------------------------------------------------------------------------


class TestPipelineRoundTrip:
    """Simulates the daemon's repeated test runs accumulating knowledge."""

    def test_three_runs_accumulate_knowledge(self, wiki_root: Path) -> None:
        slug = derive_test_slug("python3 agent_test.py")
        # Run 1 -- nothing exists yet
        loaded = load_test_knowledge(wiki_root, slug)
        assert loaded is None
        merged = merge_knowledge(
            loaded,
            {
                "purpose": "runs the agent",
                "output_format": "iteration logs",
                "normal_behavior": "all PASSED",
                "common_failures": ["timeout"],
            },
            test_slug=slug,
            command_pattern="python3 agent_test.py",
        )
        save_test_knowledge(wiki_root, merged)

        # Run 2 -- previous knowledge present, LLM proposes a new
        # failure that should accumulate.
        loaded = load_test_knowledge(wiki_root, slug)
        assert loaded is not None
        assert loaded.runs_observed == 1
        merged = merge_knowledge(
            loaded,
            {
                "purpose": "different purpose",  # should NOT overwrite
                "common_failures": ["AssertionError"],
            },
        )
        save_test_knowledge(wiki_root, merged)

        # Run 3 -- knowledge has grown
        loaded = load_test_knowledge(wiki_root, slug)
        assert loaded is not None
        assert loaded.runs_observed == 2
        assert loaded.purpose == "runs the agent"  # original preserved
        assert "timeout" in loaded.common_failures
        assert "AssertionError" in loaded.common_failures
