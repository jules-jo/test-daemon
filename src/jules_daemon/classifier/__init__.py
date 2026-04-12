"""Deterministic input classifier for the Jules daemon.

Provides a fast, pre-LLM classification layer that resolves common verb
aliases and extracts basic parameters without requiring an LLM call.
Falls through to the LLM-powered IntentClassifier (in ``llm.intent_classifier``)
when deterministic classification is ambiguous.

Modules:
    models: ClassificationResult dataclass and InputType enum
    verb_registry: Canonical verb definitions and alias-to-verb mapping
    scorer: Structuredness scorer (0.0 = NL, 1.0 = structured)
    nl_extractor: Heuristic NL intent and argument extraction
    classify: Unified classify() entry point
    direct_command: Direct-command detector for agent loop bypass
"""

from jules_daemon.classifier.classify import classify
from jules_daemon.classifier.direct_command import (
    DEFAULT_KNOWN_EXECUTABLES,
    DirectCommandDetection,
    detect_direct_command,
)
from jules_daemon.classifier.models import (
    ClassificationResult,
    InputType,
)
from jules_daemon.classifier.nl_audit import (
    NLAuditResult,
    classify_with_audit,
)
from jules_daemon.classifier.nl_extractor import (
    NLExtraction,
    extract_from_natural_language,
)
from jules_daemon.classifier.scorer import (
    STRUCTURED_THRESHOLD,
    compute_structuredness_score,
)
from jules_daemon.classifier.verb_registry import (
    CANONICAL_VERBS,
    VERB_ALIASES,
    get_aliases_for_verb,
    resolve_canonical_verb,
)

__all__ = [
    "CANONICAL_VERBS",
    "ClassificationResult",
    "DEFAULT_KNOWN_EXECUTABLES",
    "DirectCommandDetection",
    "InputType",
    "NLAuditResult",
    "NLExtraction",
    "STRUCTURED_THRESHOLD",
    "VERB_ALIASES",
    "classify",
    "classify_with_audit",
    "compute_structuredness_score",
    "detect_direct_command",
    "extract_from_natural_language",
    "get_aliases_for_verb",
    "resolve_canonical_verb",
]
