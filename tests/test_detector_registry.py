"""Tests for the DetectorRegistry register/unregister pattern.

Verifies that the DetectorRegistry:
- Registers AnomalyDetector instances by pattern_name
- Rejects duplicate registrations with a clear error
- Unregisters detectors by name, returning the removed detector
- Raises KeyError when unregistering an unknown name
- Looks up detectors by name (returns None for unknown names)
- Lists all registered detectors as an immutable tuple
- Reports detector names as a frozenset
- Reports detector count accurately
- Handles bulk registration and unregistration
- Provides clear repr for debugging
- Is safe to iterate while empty
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jules_daemon.monitor.anomaly_models import (
    AnomalyDetector,
    AnomalyReport,
    AnomalySeverity,
    ErrorKeywordPattern,
    FailureRatePattern,
    PatternType,
    StallTimeoutPattern,
)
from jules_daemon.monitor.detector_registry import DetectorRegistry
from jules_daemon.monitor.error_keyword_detector import ErrorKeywordDetector
from jules_daemon.monitor.failure_rate_spike_detector import (
    FailureRateSpikeDetector,
)
from jules_daemon.monitor.stall_hang_detector import StallHangDetector


# ---------------------------------------------------------------------------
# Helpers: minimal stub detector for isolation tests
# ---------------------------------------------------------------------------


class _StubDetector:
    """Minimal AnomalyDetector-conformant stub for registry tests."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def pattern_name(self) -> str:
        return self._name

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.ERROR_KEYWORD

    def match(self, output_line: str) -> bool:
        return False

    def report(
        self,
        output_line: str,
        *,
        session_id: str,
        detected_at: datetime,
    ) -> AnomalyReport:
        return AnomalyReport(
            pattern_name=self._name,
            pattern_type=PatternType.ERROR_KEYWORD,
            severity=AnomalySeverity.INFO,
            message=f"stub report from {self._name}",
            detected_at=detected_at,
            session_id=session_id,
        )


# ---------------------------------------------------------------------------
# Construction / empty state
# ---------------------------------------------------------------------------


class TestDetectorRegistryEmpty:
    """Tests for the registry in its initial empty state."""

    def test_empty_registry_count_is_zero(self) -> None:
        registry = DetectorRegistry()
        assert registry.count == 0

    def test_empty_registry_list_detectors_returns_empty_tuple(self) -> None:
        registry = DetectorRegistry()
        assert registry.list_detectors() == ()

    def test_empty_registry_detector_names_returns_empty_frozenset(self) -> None:
        registry = DetectorRegistry()
        assert registry.detector_names == frozenset()

    def test_empty_registry_get_returns_none(self) -> None:
        registry = DetectorRegistry()
        assert registry.get("nonexistent") is None

    def test_empty_registry_repr(self) -> None:
        registry = DetectorRegistry()
        result = repr(registry)
        assert "DetectorRegistry" in result
        assert "0" in result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestDetectorRegistryRegister:
    """Tests for registering detectors."""

    def test_register_single_detector(self) -> None:
        registry = DetectorRegistry()
        detector = _StubDetector("alpha")
        registry.register(detector)
        assert registry.count == 1
        assert registry.get("alpha") is detector

    def test_register_multiple_detectors(self) -> None:
        registry = DetectorRegistry()
        d1 = _StubDetector("alpha")
        d2 = _StubDetector("beta")
        d3 = _StubDetector("gamma")
        registry.register(d1)
        registry.register(d2)
        registry.register(d3)
        assert registry.count == 3
        assert registry.detector_names == frozenset({"alpha", "beta", "gamma"})

    def test_register_duplicate_name_raises_value_error(self) -> None:
        registry = DetectorRegistry()
        d1 = _StubDetector("alpha")
        d2 = _StubDetector("alpha")
        registry.register(d1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(d2)

    def test_register_real_error_keyword_detector(self) -> None:
        registry = DetectorRegistry()
        detector = ErrorKeywordDetector(
            patterns=(
                ErrorKeywordPattern(name="oom", regex=r"OOM"),
            ),
            name="oom_detector",
        )
        registry.register(detector)
        assert registry.get("oom_detector") is detector

    def test_register_real_failure_rate_detector(self) -> None:
        registry = DetectorRegistry()
        pattern = FailureRatePattern(
            name="high_fail",
            threshold_count=5,
            window_seconds=60.0,
        )
        detector = FailureRateSpikeDetector(pattern=pattern)
        registry.register(detector)
        assert registry.get("failure_rate_spike_detector") is detector

    def test_register_real_stall_detector(self) -> None:
        registry = DetectorRegistry()
        pattern = StallTimeoutPattern(
            name="output_stall",
            timeout_seconds=300.0,
        )
        detector = StallHangDetector(pattern=pattern)
        registry.register(detector)
        assert registry.get("stall_hang_detector") is detector

    def test_register_enforces_anomaly_detector_protocol(self) -> None:
        """Registry should accept any object satisfying AnomalyDetector protocol."""
        registry = DetectorRegistry()
        stub = _StubDetector("proto_check")
        # Verify the stub satisfies the protocol
        assert isinstance(stub, AnomalyDetector)
        registry.register(stub)
        assert registry.count == 1


# ---------------------------------------------------------------------------
# Unregistration
# ---------------------------------------------------------------------------


class TestDetectorRegistryUnregister:
    """Tests for unregistering detectors."""

    def test_unregister_returns_removed_detector(self) -> None:
        registry = DetectorRegistry()
        detector = _StubDetector("alpha")
        registry.register(detector)
        removed = registry.unregister("alpha")
        assert removed is detector
        assert registry.count == 0

    def test_unregister_unknown_name_raises_key_error(self) -> None:
        registry = DetectorRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            registry.unregister("nonexistent")

    def test_unregister_from_multiple(self) -> None:
        registry = DetectorRegistry()
        d1 = _StubDetector("alpha")
        d2 = _StubDetector("beta")
        registry.register(d1)
        registry.register(d2)

        removed = registry.unregister("alpha")
        assert removed is d1
        assert registry.count == 1
        assert registry.get("alpha") is None
        assert registry.get("beta") is d2

    def test_unregister_then_re_register_same_name(self) -> None:
        registry = DetectorRegistry()
        d1 = _StubDetector("alpha")
        d2 = _StubDetector("alpha")
        registry.register(d1)
        registry.unregister("alpha")
        registry.register(d2)
        assert registry.get("alpha") is d2

    def test_unregister_all(self) -> None:
        registry = DetectorRegistry()
        names = ["alpha", "beta", "gamma"]
        for name in names:
            registry.register(_StubDetector(name))
        assert registry.count == 3

        for name in names:
            registry.unregister(name)
        assert registry.count == 0
        assert registry.list_detectors() == ()


# ---------------------------------------------------------------------------
# Lookup and listing
# ---------------------------------------------------------------------------


class TestDetectorRegistryLookup:
    """Tests for get, list_detectors, and detector_names."""

    def test_get_returns_registered_detector(self) -> None:
        registry = DetectorRegistry()
        detector = _StubDetector("alpha")
        registry.register(detector)
        assert registry.get("alpha") is detector

    def test_get_returns_none_for_unknown(self) -> None:
        registry = DetectorRegistry()
        registry.register(_StubDetector("alpha"))
        assert registry.get("unknown") is None

    def test_list_detectors_returns_tuple(self) -> None:
        registry = DetectorRegistry()
        d1 = _StubDetector("alpha")
        d2 = _StubDetector("beta")
        registry.register(d1)
        registry.register(d2)
        result = registry.list_detectors()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert d1 in result
        assert d2 in result

    def test_detector_names_returns_frozenset(self) -> None:
        registry = DetectorRegistry()
        registry.register(_StubDetector("alpha"))
        registry.register(_StubDetector("beta"))
        names = registry.detector_names
        assert isinstance(names, frozenset)
        assert names == frozenset({"alpha", "beta"})

    def test_list_detectors_is_snapshot(self) -> None:
        """Returned tuple must not be affected by later mutations."""
        registry = DetectorRegistry()
        registry.register(_StubDetector("alpha"))
        snapshot = registry.list_detectors()
        registry.register(_StubDetector("beta"))
        assert len(snapshot) == 1  # unchanged


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestDetectorRegistryRepr:
    """Tests for the registry's string representation."""

    def test_repr_with_detectors(self) -> None:
        registry = DetectorRegistry()
        registry.register(_StubDetector("alpha"))
        registry.register(_StubDetector("beta"))
        result = repr(registry)
        assert "DetectorRegistry" in result
        assert "2" in result

    def test_repr_empty(self) -> None:
        result = repr(DetectorRegistry())
        assert "DetectorRegistry" in result
