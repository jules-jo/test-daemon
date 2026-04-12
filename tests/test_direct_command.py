"""Tests for the direct-command detector.

Validates that the detector:
1. Identifies inputs starting with known executables
2. Returns a bypass flag (is_direct_command=True) for direct commands
3. Returns is_direct_command=False for natural language inputs
4. Returns is_direct_command=False for daemon verb commands
5. Handles edge cases (empty, whitespace, paths, env prefixes)
6. Produces immutable, frozen results
"""

from __future__ import annotations

import pytest

from jules_daemon.classifier.direct_command import (
    DEFAULT_KNOWN_EXECUTABLES,
    DirectCommandDetection,
    detect_direct_command,
)


class TestKnownExecutablesDetection:
    """Commands starting with known executables should be detected."""

    def test_pytest_detected(self) -> None:
        result = detect_direct_command("pytest -v tests/")
        assert result.is_direct_command is True
        assert result.executable == "pytest"

    def test_python_detected(self) -> None:
        result = detect_direct_command("python -m pytest --tb=short")
        assert result.is_direct_command is True
        assert result.executable == "python"

    def test_python3_detected(self) -> None:
        result = detect_direct_command("python3 test_runner.py")
        assert result.is_direct_command is True
        assert result.executable == "python3"

    def test_npm_detected(self) -> None:
        result = detect_direct_command("npm test")
        assert result.is_direct_command is True
        assert result.executable == "npm"

    def test_npx_detected(self) -> None:
        result = detect_direct_command("npx jest --verbose")
        assert result.is_direct_command is True
        assert result.executable == "npx"

    def test_cargo_detected(self) -> None:
        result = detect_direct_command("cargo test --release")
        assert result.is_direct_command is True
        assert result.executable == "cargo"

    def test_go_detected(self) -> None:
        result = detect_direct_command("go test ./...")
        assert result.is_direct_command is True
        assert result.executable == "go"

    def test_make_detected(self) -> None:
        result = detect_direct_command("make test")
        assert result.is_direct_command is True
        assert result.executable == "make"

    def test_bash_detected(self) -> None:
        result = detect_direct_command("bash run_tests.sh")
        assert result.is_direct_command is True
        assert result.executable == "bash"

    def test_sh_detected(self) -> None:
        result = detect_direct_command("sh -c 'pytest -v'")
        assert result.is_direct_command is True
        assert result.executable == "sh"

    def test_java_detected(self) -> None:
        result = detect_direct_command("java -jar runner.jar")
        assert result.is_direct_command is True
        assert result.executable == "java"

    def test_mvn_detected(self) -> None:
        result = detect_direct_command("mvn test -pl module")
        assert result.is_direct_command is True
        assert result.executable == "mvn"

    def test_gradle_detected(self) -> None:
        result = detect_direct_command("gradle test")
        assert result.is_direct_command is True
        assert result.executable == "gradle"

    def test_gradlew_detected(self) -> None:
        result = detect_direct_command("./gradlew test")
        assert result.is_direct_command is True
        assert result.executable == "gradlew"

    def test_dotslash_prefix_stripped(self) -> None:
        result = detect_direct_command("./run_tests.sh")
        assert result.is_direct_command is True
        assert result.executable == "run_tests.sh"

    def test_ls_detected(self) -> None:
        result = detect_direct_command("ls -la /opt/app")
        assert result.is_direct_command is True
        assert result.executable == "ls"

    def test_cat_detected(self) -> None:
        result = detect_direct_command("cat /var/log/test.log")
        assert result.is_direct_command is True
        assert result.executable == "cat"

    def test_grep_detected(self) -> None:
        result = detect_direct_command("grep -r 'FAIL' test_output/")
        assert result.is_direct_command is True
        assert result.executable == "grep"

    def test_docker_detected(self) -> None:
        result = detect_direct_command("docker run --rm test-image")
        assert result.is_direct_command is True
        assert result.executable == "docker"

    def test_kubectl_detected(self) -> None:
        result = detect_direct_command("kubectl get pods -n testing")
        assert result.is_direct_command is True
        assert result.executable == "kubectl"

    def test_dotnet_detected(self) -> None:
        result = detect_direct_command("dotnet test --filter Category=Unit")
        assert result.is_direct_command is True
        assert result.executable == "dotnet"

    def test_ruby_detected(self) -> None:
        result = detect_direct_command("ruby -e 'puts :ok'")
        assert result.is_direct_command is True
        assert result.executable == "ruby"

    def test_node_detected(self) -> None:
        result = detect_direct_command("node test.js")
        assert result.is_direct_command is True
        assert result.executable == "node"

    def test_perl_detected(self) -> None:
        result = detect_direct_command("perl test.pl")
        assert result.is_direct_command is True
        assert result.executable == "perl"


class TestAbsolutePathExecutables:
    """Commands starting with absolute paths to executables."""

    def test_absolute_path_detected(self) -> None:
        result = detect_direct_command("/usr/bin/python3 test.py")
        assert result.is_direct_command is True
        assert result.executable == "python3"

    def test_absolute_path_unknown_executable(self) -> None:
        result = detect_direct_command("/opt/custom/runner --suite smoke")
        assert result.is_direct_command is True
        assert result.executable == "runner"

    def test_nested_absolute_path(self) -> None:
        result = detect_direct_command("/usr/local/bin/pytest -v")
        assert result.is_direct_command is True
        assert result.executable == "pytest"


class TestEnvironmentVariablePrefixes:
    """Commands prefixed with VAR=value should still be detected."""

    def test_env_prefix_single(self) -> None:
        result = detect_direct_command("PYTHONPATH=/opt/app pytest -v")
        assert result.is_direct_command is True
        assert result.executable == "pytest"

    def test_env_prefix_multiple(self) -> None:
        result = detect_direct_command(
            "PYTHONPATH=/opt/app DJANGO_SETTINGS_MODULE=settings pytest"
        )
        assert result.is_direct_command is True
        assert result.executable == "pytest"

    def test_env_prefix_with_quoted_value(self) -> None:
        result = detect_direct_command("HOME=/tmp python3 test.py")
        assert result.is_direct_command is True
        assert result.executable == "python3"


class TestSudoPrefixes:
    """Commands prefixed with sudo should still be detected."""

    def test_sudo_prefix(self) -> None:
        result = detect_direct_command("sudo pytest -v")
        assert result.is_direct_command is True
        assert result.executable == "pytest"

    def test_sudo_with_flag(self) -> None:
        result = detect_direct_command("sudo -u testuser python3 test.py")
        assert result.is_direct_command is True
        assert result.executable == "python3"


class TestNonDirectCommands:
    """Natural language and daemon verbs should NOT be detected as direct."""

    def test_natural_language_question(self) -> None:
        result = detect_direct_command("what's running right now?")
        assert result.is_direct_command is False
        assert result.executable is None

    def test_natural_language_request(self) -> None:
        result = detect_direct_command(
            "can you run the smoke tests on staging?"
        )
        assert result.is_direct_command is False
        assert result.executable is None

    def test_natural_language_polite(self) -> None:
        result = detect_direct_command(
            "please run the integration tests"
        )
        assert result.is_direct_command is False
        assert result.executable is None

    def test_conversational_input(self) -> None:
        result = detect_direct_command(
            "I want to see the results from the last test run"
        )
        assert result.is_direct_command is False
        assert result.executable is None

    def test_daemon_verb_status(self) -> None:
        """Daemon verbs (status, watch, etc.) are NOT direct commands --
        they are handled by the daemon's verb pipeline, not SSH."""
        result = detect_direct_command("status")
        assert result.is_direct_command is False
        assert result.executable is None

    def test_daemon_verb_watch(self) -> None:
        result = detect_direct_command("watch --tail 100")
        assert result.is_direct_command is False
        assert result.executable is None

    def test_daemon_verb_cancel(self) -> None:
        result = detect_direct_command("cancel --force")
        assert result.is_direct_command is False
        assert result.executable is None

    def test_daemon_verb_history(self) -> None:
        result = detect_direct_command("history --limit 20")
        assert result.is_direct_command is False
        assert result.executable is None

    def test_daemon_verb_queue(self) -> None:
        result = detect_direct_command("queue deploy@host run tests")
        assert result.is_direct_command is False
        assert result.executable is None

    def test_daemon_alias_execute(self) -> None:
        """'execute' is a verb alias for 'run' -- should not be direct."""
        result = detect_direct_command("execute deploy@host tests")
        assert result.is_direct_command is False
        assert result.executable is None

    def test_daemon_alias_exec(self) -> None:
        """'exec' is a verb alias for 'run' -- should not be direct."""
        result = detect_direct_command("exec deploy@host tests")
        assert result.is_direct_command is False
        assert result.executable is None


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_string(self) -> None:
        result = detect_direct_command("")
        assert result.is_direct_command is False
        assert result.executable is None

    def test_whitespace_only(self) -> None:
        result = detect_direct_command("   ")
        assert result.is_direct_command is False
        assert result.executable is None

    def test_single_known_executable(self) -> None:
        """A bare executable name with no args should still be detected."""
        result = detect_direct_command("python3")
        assert result.is_direct_command is True
        assert result.executable == "python3"

    def test_leading_whitespace_stripped(self) -> None:
        result = detect_direct_command("  pytest -v tests/")
        assert result.is_direct_command is True
        assert result.executable == "pytest"

    def test_trailing_whitespace_stripped(self) -> None:
        result = detect_direct_command("pytest -v tests/  ")
        assert result.is_direct_command is True
        assert result.executable == "pytest"

    def test_case_insensitive_executable(self) -> None:
        """Executables should match case-insensitively."""
        result = detect_direct_command("Python3 test.py")
        assert result.is_direct_command is True
        assert result.executable == "python3"

    def test_unknown_word(self) -> None:
        """Random words should not be detected as direct commands."""
        result = detect_direct_command("frobnicator --test")
        assert result.is_direct_command is False
        assert result.executable is None

    def test_preserves_raw_command(self) -> None:
        raw = "pytest -v tests/"
        result = detect_direct_command(raw)
        assert result.raw_command == raw


class TestResultImmutability:
    """DirectCommandDetection should be frozen (immutable)."""

    def test_result_is_frozen(self) -> None:
        result = detect_direct_command("pytest -v")
        with pytest.raises(AttributeError):
            result.is_direct_command = False  # type: ignore[misc]

    def test_result_is_frozen_executable(self) -> None:
        result = detect_direct_command("pytest -v")
        with pytest.raises(AttributeError):
            result.executable = "other"  # type: ignore[misc]


class TestResultProperties:
    """DirectCommandDetection property accessors."""

    def test_bypass_agent_loop_true_for_direct(self) -> None:
        result = detect_direct_command("pytest -v tests/")
        assert result.bypass_agent_loop is True

    def test_bypass_agent_loop_false_for_nl(self) -> None:
        result = detect_direct_command("what's running?")
        assert result.bypass_agent_loop is False

    def test_to_dict_round_trips(self) -> None:
        result = detect_direct_command("pytest -v tests/")
        d = result.to_dict()
        assert d["is_direct_command"] is True
        assert d["executable"] == "pytest"
        assert d["raw_command"] == "pytest -v tests/"
        assert isinstance(d["confidence"], float)


class TestCustomExecutables:
    """detect_direct_command should accept custom executable sets."""

    def test_custom_executables_override(self) -> None:
        custom = frozenset({"myrunner", "custom_tool"})
        result = detect_direct_command("myrunner --suite smoke", known_executables=custom)
        assert result.is_direct_command is True
        assert result.executable == "myrunner"

    def test_custom_executables_reject_default(self) -> None:
        custom = frozenset({"myrunner"})
        result = detect_direct_command("pytest -v", known_executables=custom)
        assert result.is_direct_command is False

    def test_empty_custom_set_rejects_all(self) -> None:
        result = detect_direct_command("pytest -v", known_executables=frozenset())
        assert result.is_direct_command is False


class TestDefaultKnownExecutables:
    """Validate the default known executables set."""

    def test_contains_test_runners(self) -> None:
        test_runners = {"pytest", "python", "python3", "npm", "npx",
                        "cargo", "go", "make", "gradle", "gradlew",
                        "mvn", "dotnet", "node", "java"}
        assert test_runners.issubset(DEFAULT_KNOWN_EXECUTABLES)

    def test_contains_shell_utilities(self) -> None:
        utils = {"bash", "sh", "ls", "cat", "grep", "find", "tail",
                 "head", "wc", "sort", "awk", "sed"}
        assert utils.issubset(DEFAULT_KNOWN_EXECUTABLES)

    def test_contains_container_tools(self) -> None:
        containers = {"docker", "kubectl"}
        assert containers.issubset(DEFAULT_KNOWN_EXECUTABLES)

    def test_does_not_contain_daemon_verbs(self) -> None:
        """Daemon verbs should NEVER be in the known executables set."""
        daemon_verbs = {"status", "watch", "cancel", "history", "queue"}
        assert daemon_verbs.isdisjoint(DEFAULT_KNOWN_EXECUTABLES)

    def test_is_frozenset(self) -> None:
        assert isinstance(DEFAULT_KNOWN_EXECUTABLES, frozenset)


class TestChainedCommands:
    """Commands chained with && or ; should still be detected."""

    def test_chained_with_and(self) -> None:
        result = detect_direct_command("cd /opt/app && pytest -v")
        assert result.is_direct_command is True
        assert result.executable == "cd"

    def test_chained_with_semicolon(self) -> None:
        result = detect_direct_command("cd /opt/app; pytest -v")
        assert result.is_direct_command is True
        assert result.executable == "cd"

    def test_piped_command(self) -> None:
        result = detect_direct_command("pytest -v 2>&1 | tee output.log")
        assert result.is_direct_command is True
        assert result.executable == "pytest"
