"""Canonical verb registry for deterministic input classification.

Maps verb aliases and synonyms to the six normalized canonical verbs
used by the Jules daemon. This registry enables fast, deterministic
resolution of common verb forms without requiring an LLM call.

The six canonical verbs are:
    run     -- start test execution on a remote host
    status  -- query current run state
    cancel  -- cancel a running or queued test
    watch   -- live-stream output from a running test
    queue   -- queue a command for later execution
    history -- view past test run results

Each canonical verb has multiple aliases (synonyms, shortened forms,
common misspellings) registered in the VERB_ALIASES mapping. Lookup
is always case-insensitive with whitespace stripped.

The registry is immutable after module load. All data structures are
frozen (frozenset, MappingProxyType) to prevent runtime mutation.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

__all__ = [
    "CANONICAL_VERBS",
    "VERB_ALIASES",
    "get_aliases_for_verb",
    "resolve_canonical_verb",
]


# ---------------------------------------------------------------------------
# Canonical verb set
# ---------------------------------------------------------------------------

CANONICAL_VERBS: frozenset[str] = frozenset({
    "run",
    "status",
    "cancel",
    "watch",
    "queue",
    "history",
})
"""The six recognized canonical verbs for the Jules daemon."""


# ---------------------------------------------------------------------------
# Alias -> canonical verb mapping
# ---------------------------------------------------------------------------

# Internal mutable dict used only during module initialization.
# Exposed via MappingProxyType for immutability.
_aliases: dict[str, str] = {
    # -- run: start test execution --
    "run": "run",
    "execute": "run",
    "exec": "run",
    "start": "run",
    "launch": "run",
    "begin": "run",
    "test": "run",
    "kick": "run",
    "trigger": "run",

    # -- status: query current state --
    "status": "status",
    "check": "status",
    "state": "status",
    "info": "status",
    "progress": "status",
    "ping": "status",

    # -- cancel: stop a running or queued test --
    "cancel": "cancel",
    "stop": "cancel",
    "abort": "cancel",
    "kill": "cancel",
    "terminate": "cancel",
    "halt": "cancel",

    # -- watch: live-stream output --
    "watch": "watch",
    "tail": "watch",
    "follow": "watch",
    "stream": "watch",
    "monitor": "watch",
    "logs": "watch",
    "output": "watch",
    "attach": "watch",

    # -- queue: defer execution --
    "queue": "queue",
    "enqueue": "queue",
    "schedule": "queue",
    "defer": "queue",
    "later": "queue",

    # -- history: past run results --
    "history": "history",
    "past": "history",
    "results": "history",
    "previous": "history",
    "log": "history",
    "report": "history",
    "reports": "history",
}

VERB_ALIASES: Mapping[str, str] = MappingProxyType(_aliases)
"""Immutable mapping from alias strings to canonical verb strings.

All keys are lowercase. Every canonical verb appears as its own alias
(identity mapping). Values are always members of CANONICAL_VERBS.
"""


# ---------------------------------------------------------------------------
# Pre-computed reverse index: canonical verb -> frozenset of aliases
# ---------------------------------------------------------------------------

_reverse_index: dict[str, frozenset[str]] = {}
_alias: str = ""
_canonical: str = ""
for _alias, _canonical in _aliases.items():
    if _canonical not in _reverse_index:
        _reverse_index[_canonical] = frozenset()
    _reverse_index[_canonical] = _reverse_index[_canonical] | frozenset({_alias})

_REVERSE_INDEX: Mapping[str, frozenset[str]] = MappingProxyType(_reverse_index)

# Clean up module-level loop variables
del _alias, _canonical


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_canonical_verb(alias: str) -> str | None:
    """Resolve a verb alias to its canonical form.

    Performs case-insensitive lookup with leading/trailing whitespace
    stripped. Returns None for unrecognized aliases instead of raising
    an exception, allowing callers to fall through to LLM classification.

    Args:
        alias: Raw verb alias string from user input.

    Returns:
        The canonical verb string (e.g., ``"run"``), or None if the
        alias is not recognized in the registry.
    """
    normalized = alias.strip().lower()
    if not normalized:
        return None
    return VERB_ALIASES.get(normalized)


def get_aliases_for_verb(canonical_verb: str) -> frozenset[str]:
    """Return all registered aliases for a canonical verb.

    Useful for help text, documentation, and introspection. The
    returned set always includes the canonical verb itself (identity
    mapping).

    Args:
        canonical_verb: A canonical verb string (e.g., ``"run"``).

    Returns:
        Frozenset of alias strings that map to this verb.
        Empty frozenset if the verb is not recognized.
    """
    return _REVERSE_INDEX.get(canonical_verb, frozenset())
