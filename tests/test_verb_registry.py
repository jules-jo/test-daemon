"""Tests for the canonical verb registry.

Validates the alias-to-verb mapping, resolution function, and
registry introspection utilities.
"""

from __future__ import annotations

import pytest

from jules_daemon.classifier.verb_registry import (
    CANONICAL_VERBS,
    VERB_ALIASES,
    get_aliases_for_verb,
    resolve_canonical_verb,
)


class TestCanonicalVerbs:
    """Tests for the CANONICAL_VERBS frozenset."""

    def test_contains_all_six_verbs(self) -> None:
        expected = frozenset({"run", "status", "cancel", "watch", "queue", "history"})
        assert CANONICAL_VERBS == expected

    def test_is_frozenset(self) -> None:
        assert isinstance(CANONICAL_VERBS, frozenset)

    def test_immutable(self) -> None:
        with pytest.raises(AttributeError):
            CANONICAL_VERBS.add("destroy")  # type: ignore[attr-defined]


class TestVerbAliases:
    """Tests for the VERB_ALIASES mapping structure."""

    def test_identity_mappings_present(self) -> None:
        """Every canonical verb must map to itself."""
        for verb in CANONICAL_VERBS:
            assert VERB_ALIASES[verb] == verb, (
                f"Canonical verb {verb!r} must be present as its own alias"
            )

    def test_all_values_are_canonical(self) -> None:
        """Every alias must resolve to a valid canonical verb."""
        for alias, canonical in VERB_ALIASES.items():
            assert canonical in CANONICAL_VERBS, (
                f"Alias {alias!r} maps to {canonical!r} which is not canonical"
            )

    def test_all_keys_are_lowercase(self) -> None:
        """All alias keys must be lowercase for consistent lookup."""
        for alias in VERB_ALIASES:
            assert alias == alias.lower(), (
                f"Alias key {alias!r} must be lowercase"
            )

    def test_no_empty_keys(self) -> None:
        for alias in VERB_ALIASES:
            assert alias.strip(), "Alias keys must not be empty or whitespace"

    def test_run_aliases(self) -> None:
        """Common synonyms for 'run' should be registered."""
        run_aliases = {"run", "execute", "start", "launch", "begin", "test", "exec"}
        for alias in run_aliases:
            assert VERB_ALIASES.get(alias) == "run", (
                f"Expected {alias!r} to map to 'run'"
            )

    def test_status_aliases(self) -> None:
        """Common synonyms for 'status' should be registered."""
        status_aliases = {"status", "check", "state", "info"}
        for alias in status_aliases:
            assert VERB_ALIASES.get(alias) == "status", (
                f"Expected {alias!r} to map to 'status'"
            )

    def test_cancel_aliases(self) -> None:
        """Common synonyms for 'cancel' should be registered."""
        cancel_aliases = {"cancel", "stop", "abort", "kill", "terminate"}
        for alias in cancel_aliases:
            assert VERB_ALIASES.get(alias) == "cancel", (
                f"Expected {alias!r} to map to 'cancel'"
            )

    def test_watch_aliases(self) -> None:
        """Common synonyms for 'watch' should be registered."""
        watch_aliases = {"watch", "tail", "follow", "stream", "monitor", "logs"}
        for alias in watch_aliases:
            assert VERB_ALIASES.get(alias) == "watch", (
                f"Expected {alias!r} to map to 'watch'"
            )

    def test_queue_aliases(self) -> None:
        """Common synonyms for 'queue' should be registered."""
        queue_aliases = {"queue", "enqueue", "schedule", "defer"}
        for alias in queue_aliases:
            assert VERB_ALIASES.get(alias) == "queue", (
                f"Expected {alias!r} to map to 'queue'"
            )

    def test_history_aliases(self) -> None:
        """Common synonyms for 'history' should be registered."""
        history_aliases = {"history", "past", "results", "previous", "log"}
        for alias in history_aliases:
            assert VERB_ALIASES.get(alias) == "history", (
                f"Expected {alias!r} to map to 'history'"
            )


class TestResolveCanonicalVerb:
    """Tests for the resolve_canonical_verb function."""

    def test_resolve_exact_canonical_verb(self) -> None:
        for verb in CANONICAL_VERBS:
            assert resolve_canonical_verb(verb) == verb

    def test_resolve_known_alias(self) -> None:
        assert resolve_canonical_verb("execute") == "run"
        assert resolve_canonical_verb("stop") == "cancel"
        assert resolve_canonical_verb("tail") == "watch"
        assert resolve_canonical_verb("enqueue") == "queue"
        assert resolve_canonical_verb("check") == "status"
        assert resolve_canonical_verb("past") == "history"

    def test_case_insensitive(self) -> None:
        assert resolve_canonical_verb("RUN") == "run"
        assert resolve_canonical_verb("Execute") == "run"
        assert resolve_canonical_verb("STOP") == "cancel"
        assert resolve_canonical_verb("Status") == "status"

    def test_strips_whitespace(self) -> None:
        assert resolve_canonical_verb("  run  ") == "run"
        assert resolve_canonical_verb("\texecute\n") == "run"

    def test_unknown_alias_returns_none(self) -> None:
        assert resolve_canonical_verb("destroy") is None
        assert resolve_canonical_verb("obliterate") is None
        assert resolve_canonical_verb("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert resolve_canonical_verb("   ") is None
        assert resolve_canonical_verb("\t\n") is None


class TestGetAliasesForVerb:
    """Tests for the get_aliases_for_verb introspection function."""

    def test_returns_all_aliases_for_run(self) -> None:
        aliases = get_aliases_for_verb("run")
        assert "run" in aliases
        assert "execute" in aliases
        assert "start" in aliases

    def test_returns_all_aliases_for_cancel(self) -> None:
        aliases = get_aliases_for_verb("cancel")
        assert "cancel" in aliases
        assert "stop" in aliases
        assert "abort" in aliases
        assert "kill" in aliases

    def test_returns_frozenset(self) -> None:
        aliases = get_aliases_for_verb("status")
        assert isinstance(aliases, frozenset)

    def test_unknown_verb_returns_empty(self) -> None:
        aliases = get_aliases_for_verb("destroy")
        assert aliases == frozenset()

    def test_every_canonical_verb_has_at_least_itself(self) -> None:
        for verb in CANONICAL_VERBS:
            aliases = get_aliases_for_verb(verb)
            assert verb in aliases, (
                f"Canonical verb {verb!r} should be in its own alias set"
            )
