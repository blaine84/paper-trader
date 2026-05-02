"""
Drift Warning Tests — Property 4: Drift Warning Threshold

Validates: Requirements 2.12, 3.6

Tests verify that the drift warning fires when:
  - raw gap_and_go count >= 10 AND reclassified/raw > 0.10
And does NOT fire when:
  - raw gap_and_go count < 10 (regardless of proportion)
  - reclassified/raw <= 0.10

The drift warning logic lives in agents/analyst.py run() and operates
on the signals dict after ThreadPoolExecutor completes.
"""

import logging

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


def _compute_drift_warning(signals: dict, logger: logging.Logger) -> bool:
    """
    Replicate the drift warning logic from agents/analyst.py run().

    Returns True if a WARNING was logged, False otherwise.
    This mirrors the exact logic in the production code.
    """
    raw_gap_and_go = 0
    reclassified_gap_and_go = 0
    for sig in signals.values():
        is_current_gap = sig.get("setup_type") == "gap_and_go"
        is_reclassified_from_gap = (
            sig.get("setup_reclassified") is True
            and sig.get("original_setup_type") == "gap_and_go"
        )
        if is_current_gap or is_reclassified_from_gap:
            raw_gap_and_go += 1
        if is_reclassified_from_gap:
            reclassified_gap_and_go += 1

    if raw_gap_and_go >= 10 and (reclassified_gap_and_go / raw_gap_and_go) > 0.10:
        logger.warning(
            "Signal drift detected: %d/%d raw gap_and_go signals were reclassified (%.1f%%). "
            "LLM may be over-assigning gap_and_go to non-stock symbols.",
            reclassified_gap_and_go,
            raw_gap_and_go,
            (reclassified_gap_and_go / raw_gap_and_go) * 100,
        )
        return True
    return False


def _build_signals(
    current_gap_count: int,
    reclassified_count: int,
    other_count: int = 0,
) -> dict:
    """Build a signals dict with the specified counts."""
    signals = {}
    idx = 0

    # Signals that currently have gap_and_go (unknown symbols that kept it)
    for _ in range(current_gap_count):
        signals[f"SYM{idx}"] = {
            "setup_type": "gap_and_go",
            "setup_reclassified": False,
        }
        idx += 1

    # Signals that were reclassified FROM gap_and_go (non-stock symbols)
    for _ in range(reclassified_count):
        signals[f"SYM{idx}"] = {
            "setup_type": "technical_breakout",
            "setup_reclassified": True,
            "original_setup_type": "gap_and_go",
        }
        idx += 1

    # Other signals (non-gap_and_go)
    for _ in range(other_count):
        signals[f"SYM{idx}"] = {
            "setup_type": "momentum_fade",
            "setup_reclassified": False,
        }
        idx += 1

    return signals


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDriftWarningUnit:
    """Unit tests for drift warning threshold logic."""

    def test_drift_warning_fires_above_threshold(self, caplog):
        """10 raw gap_and_go, 2 reclassified (20%) → warning fires."""
        signals = _build_signals(current_gap_count=8, reclassified_count=2)
        logger = logging.getLogger("test_drift")

        with caplog.at_level(logging.WARNING, logger="test_drift"):
            fired = _compute_drift_warning(signals, logger)

        assert fired is True
        assert "Signal drift detected" in caplog.text

    def test_drift_warning_suppressed_below_10(self, caplog):
        """9 raw gap_and_go, 5 reclassified (55%) → warning suppressed (< 10 raw)."""
        signals = _build_signals(current_gap_count=4, reclassified_count=5)
        logger = logging.getLogger("test_drift")

        with caplog.at_level(logging.WARNING, logger="test_drift"):
            fired = _compute_drift_warning(signals, logger)

        assert fired is False
        assert "Signal drift detected" not in caplog.text

    def test_drift_warning_suppressed_at_10_percent(self, caplog):
        """10 raw gap_and_go, 1 reclassified (10%) → warning suppressed (not > 10%)."""
        signals = _build_signals(current_gap_count=9, reclassified_count=1)
        logger = logging.getLogger("test_drift")

        with caplog.at_level(logging.WARNING, logger="test_drift"):
            fired = _compute_drift_warning(signals, logger)

        assert fired is False
        assert "Signal drift detected" not in caplog.text

    def test_drift_warning_fires_just_above_10_percent(self, caplog):
        """10 raw gap_and_go, 2 reclassified (20%) → warning fires (> 10%)."""
        signals = _build_signals(current_gap_count=8, reclassified_count=2)
        logger = logging.getLogger("test_drift")

        with caplog.at_level(logging.WARNING, logger="test_drift"):
            fired = _compute_drift_warning(signals, logger)

        assert fired is True

    def test_drift_warning_zero_gap_and_go(self, caplog):
        """0 raw gap_and_go → warning suppressed."""
        signals = _build_signals(current_gap_count=0, reclassified_count=0, other_count=20)
        logger = logging.getLogger("test_drift")

        with caplog.at_level(logging.WARNING, logger="test_drift"):
            fired = _compute_drift_warning(signals, logger)

        assert fired is False

    def test_drift_warning_exactly_10_raw_above_proportion(self, caplog):
        """Exactly 10 raw gap_and_go, 2 reclassified (20%) → warning fires."""
        signals = _build_signals(current_gap_count=8, reclassified_count=2)
        logger = logging.getLogger("test_drift")

        with caplog.at_level(logging.WARNING, logger="test_drift"):
            fired = _compute_drift_warning(signals, logger)

        assert fired is True

    def test_drift_warning_non_gap_signals_ignored(self, caplog):
        """Non-gap_and_go signals don't count toward raw total."""
        signals = _build_signals(current_gap_count=5, reclassified_count=0, other_count=50)
        logger = logging.getLogger("test_drift")

        with caplog.at_level(logging.WARNING, logger="test_drift"):
            fired = _compute_drift_warning(signals, logger)

        assert fired is False


# ═══════════════════════════════════════════════════════════════════════
# Property-Based Test — Property 4: Drift Warning Threshold
#
# Validates: Requirements 2.12, 3.6
# ═══════════════════════════════════════════════════════════════════════


@given(
    current_gap=st.integers(min_value=0, max_value=50),
    reclassified=st.integers(min_value=0, max_value=50),
    other=st.integers(min_value=0, max_value=50),
)
@settings(max_examples=200, deadline=None)
def test_property_drift_warning_threshold(current_gap, reclassified, other):
    """
    **Validates: Requirements 2.12, 3.6**

    Property 4: Drift Warning Threshold — For any run configuration:
    - If raw_gap_and_go >= 10 AND reclassified/raw > 0.10 → warning fires
    - If raw_gap_and_go < 10 → warning does NOT fire regardless of proportion
    """
    raw_total = current_gap + reclassified
    logger = logging.getLogger("test_drift_pbt")
    signals = _build_signals(current_gap, reclassified, other)

    fired = _compute_drift_warning(signals, logger)

    if raw_total < 10:
        # Requirement 3.6: suppress when raw count < 10
        assert fired is False, (
            f"Drift warning fired with raw_gap_and_go={raw_total} < 10. "
            f"current_gap={current_gap}, reclassified={reclassified}."
        )
    elif raw_total >= 10 and reclassified / raw_total > 0.10:
        # Requirement 2.12: fire when >= 10 and proportion > 10%
        assert fired is True, (
            f"Drift warning did NOT fire with raw_gap_and_go={raw_total}, "
            f"reclassified={reclassified} ({reclassified/raw_total:.1%}). "
            f"Expected warning."
        )
    else:
        # raw >= 10 but proportion <= 10%
        assert fired is False, (
            f"Drift warning fired with proportion={reclassified}/{raw_total} "
            f"= {reclassified/raw_total:.1%} which is <= 10%."
        )
