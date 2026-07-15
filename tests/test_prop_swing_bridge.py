"""Property-based tests for swing candidate bridge feature flag behavior and observability.

Tests Property 21 (Feature Flag Mode Behavior) and Property 26 (Observability
Structured Log Completeness) from the design document.

Uses unittest.mock to patch _get_swing_mode and validates that:
- disabled mode: zero registrations, zero log entries, zero events
- observe mode: structured log entries but zero registrations and zero events
- enabled mode: full normalization, geometry, registration for eligible signals
- structured logs contain all required fields
"""

from __future__ import annotations

import logging
import logging.handlers
from decimal import Decimal
from unittest.mock import patch, MagicMock

from hypothesis import given, settings, strategies as st, assume

from utils.swing_candidate_bridge import process_swing_signals, SwingBridgeResult


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

symbol_st = st.text(min_size=1, max_size=5, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
direction_st = st.sampled_from(["LONG", "SHORT"])
strength_st = st.sampled_from(["weak", "moderate", "strong"])
confidence_st = st.sampled_from(["low", "medium", "high"])
ema_trend_st = st.sampled_from(["bullish", "bearish", "neutral"])
market_regime_st = st.sampled_from(["risk_on", "risk_off", "mixed"])
profile_st = st.sampled_from(["conservative", "moderate", "aggressive"])

# Price strategies that produce valid swing geometry (stop >1.5% from entry)
entry_price_st = st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False)


def _make_signal(symbol="AAPL", setup_type="sector_rotation", direction="LONG",
                 strength="strong", confidence="high",
                 ema_trend="bullish", market_regime="risk_on",
                 entry_price=100.0, stop_price=95.0, target_price=115.0):
    """Create a well-formed signal dict for testing."""
    return {
        "symbol": symbol,
        "setup_type": setup_type,
        "direction": direction,
        "strength": strength,
        "confidence": confidence,
        "key_levels": {"support": 95.0, "resistance": 110.0},
        "ema_trend": ema_trend,
        "market_regime": market_regime,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "signal_age_hours": 0,
        "catalyst_freshness": "fresh",
        "sector": "technology",
    }


# ---------------------------------------------------------------------------
# Property 21: Feature Flag Mode Behavior
# Validates: Requirements 10.2, 10.3, 10.4
# ---------------------------------------------------------------------------


@given(
    symbol=symbol_st,
    direction=direction_st,
    strength=strength_st,
    confidence=confidence_st,
    ema_trend=ema_trend_st,
    market_regime=market_regime_st,
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="disabled")
def test_disabled_mode_zero_output(mock_mode, symbol, direction, strength,
                                   confidence, ema_trend, market_regime):
    """Property 21 (disabled): Zero registrations, zero log entries, zero events.

    When SWING_CANDIDATE_MODE is "disabled", the bridge returns [] immediately
    with no logging or event persistence.

    **Validates: Requirements 10.2**
    """
    signals = {
        "sig-1": _make_signal(
            symbol=symbol, direction=direction, strength=strength,
            confidence=confidence, ema_trend=ema_trend, market_regime=market_regime,
        )
    }
    result = process_swing_signals(
        signals=signals,
        profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1",
        db=None,
        engine=None,
    )
    # Zero registrations
    assert result == []


@given(
    symbol=symbol_st,
    direction=direction_st,
    strength=strength_st,
    confidence=confidence_st,
    ema_trend=ema_trend_st,
    market_regime=market_regime_st,
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="disabled")
def test_disabled_mode_no_log_entries(mock_mode, symbol, direction, strength,
                                      confidence, ema_trend, market_regime):
    """Property 21 (disabled): No structured log entries emitted.

    **Validates: Requirements 10.2**
    """
    bridge_logger = logging.getLogger("utils.swing_candidate_bridge")
    handler = logging.handlers.MemoryHandler(capacity=1000)
    handler.setLevel(logging.INFO)
    bridge_logger.addHandler(handler)
    bridge_logger.setLevel(logging.INFO)
    try:
        signals = {
            "sig-1": _make_signal(
                symbol=symbol, direction=direction, strength=strength,
                confidence=confidence, ema_trend=ema_trend, market_regime=market_regime,
            )
        }
        process_swing_signals(
            signals=signals,
            profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"},
            portfolio={"equity": 100000},
            cycle_id="cycle-1",
            db=None,
            engine=None,
        )
        # Zero swing log entries
        handler.flush()
        swing_logs = [r for r in handler.buffer if "swing_bridge_signal" in r.getMessage()]
        assert len(swing_logs) == 0
    finally:
        bridge_logger.removeHandler(handler)


@given(
    symbol=symbol_st,
    strength=strength_st,
    confidence=confidence_st,
    ema_trend=ema_trend_st,
    market_regime=market_regime_st,
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_observe_mode_logs_but_no_registrations(mock_mode, symbol, strength,
                                                 confidence, ema_trend,
                                                 market_regime):
    """Property 21 (observe): Structured log entries emitted but zero registrations.

    In observe mode, normalization runs and logs are emitted, but no candidates
    are registered and no pm_candidate_events rows are written.

    **Validates: Requirements 10.3**
    """
    bridge_logger = logging.getLogger("utils.swing_candidate_bridge")
    handler = logging.handlers.MemoryHandler(capacity=1000)
    handler.setLevel(logging.INFO)
    bridge_logger.addHandler(handler)
    bridge_logger.setLevel(logging.INFO)
    try:
        # Use sector_rotation with LONG direction to ensure normalization proceeds
        signals = {
            "sig-1": _make_signal(
                symbol=symbol, setup_type="sector_rotation",
                direction="LONG", strength=strength, confidence=confidence,
                ema_trend=ema_trend, market_regime=market_regime,
            )
        }
        result = process_swing_signals(
            signals=signals,
            profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"},
            portfolio={"equity": 100000},
            cycle_id="cycle-1",
            db=None,
            engine=None,
        )
        # Zero registrations in observe mode
        assert result == []
        # Should have at least one log entry (normalization always produces a log)
        handler.flush()
        swing_logs = [r for r in handler.buffer if "swing_bridge_signal" in r.getMessage()]
        assert len(swing_logs) >= 1
    finally:
        bridge_logger.removeHandler(handler)


@given(profile_id=profile_st)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
def test_enabled_mode_registers_eligible_signals(mock_mode, profile_id):
    """Property 21 (enabled): Full normalization, geometry, registration for eligible signals.

    When SWING_CANDIDATE_MODE is "enabled" and the signal passes all checks
    (normalization, geometry, profile policy, sizing), the bridge returns
    registered candidates.

    **Validates: Requirements 10.4**
    """
    # Construct a signal that will pass for moderate/aggressive profiles
    # For conservative: needs high confidence, strong strength, and R:R >= 3.0
    if profile_id == "conservative":
        confidence = "high"
        strength = "strong"
        # Need R:R >= 3.0 for conservative policy — 20/5 = 4.0
        entry, stop, target = 100.0, 95.0, 120.0
    else:
        confidence = "high"
        strength = "strong"
        # R:R = 15/5 = 3.0 for moderate (needs >= 1.5)
        entry, stop, target = 100.0, 95.0, 115.0

    signals = {
        "sig-1": _make_signal(
            symbol="AAPL", setup_type="sector_rotation",
            direction="LONG", strength=strength, confidence=confidence,
            ema_trend="bullish", market_regime="risk_on",
            entry_price=entry, stop_price=stop, target_price=target,
        )
    }
    result = process_swing_signals(
        signals=signals,
        profile_id=profile_id,
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1",
        db=None,
        engine=None,
    )
    # Eligible signal should produce at least one registered candidate
    assert len(result) >= 1
    assert result[0]["symbol"] == "AAPL"


@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_observe_mode_no_events_written(mock_mode):
    """Property 21 (observe): Zero pm_candidate_events rows written.

    In observe mode, db is not used for event persistence. Even if db is
    provided, no events should be written because the bridge returns early
    in observe mode before reaching event emission for registrations.

    **Validates: Requirements 10.3**
    """
    mock_db = MagicMock()
    signals = {
        "sig-1": _make_signal(
            symbol="AAPL", setup_type="sector_rotation",
            direction="LONG", strength="strong", confidence="high",
            ema_trend="bullish", market_regime="risk_on",
        )
    }
    result = process_swing_signals(
        signals=signals,
        profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1",
        db=mock_db,
        engine=None,
    )
    assert result == []
    # No calls to db.connect() for event writing in observe mode
    mock_db.connect.assert_not_called()


@given(num_signals=st.integers(min_value=1, max_value=5))
@settings(max_examples=50)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="disabled")
def test_disabled_mode_multiple_signals_zero_output(mock_mode, num_signals):
    """Property 21 (disabled): Multiple signals still produce zero output.

    **Validates: Requirements 10.2**
    """
    signals = {
        f"sig-{i}": _make_signal(symbol=f"SYM{i}")
        for i in range(num_signals)
    }
    result = process_swing_signals(
        signals=signals,
        profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1",
        db=None,
        engine=None,
    )
    assert result == []


# ---------------------------------------------------------------------------
# Property 26: Observability Structured Log Completeness
# Validates: Requirements 8.1, 8.8
# ---------------------------------------------------------------------------


@given(
    symbol=symbol_st,
    setup_type=st.sampled_from([
        "sector_rotation", "risk_off_macro_short",
        "breakout_retest", "pullback_continuation",
        "relative_strength_swing", "unknown_label_xyz",
    ]),
    direction=direction_st,
    strength=strength_st,
    confidence=confidence_st,
    ema_trend=ema_trend_st,
    market_regime=market_regime_st,
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_structured_log_contains_all_required_fields(mock_mode, symbol, setup_type,
                                                      direction, strength, confidence,
                                                      ema_trend, market_regime):
    """Property 26: Structured log entries contain all required fields.

    When the bridge processes a signal, it emits a structured log entry at INFO
    containing: signal_id, symbol, raw_label, normalized_label, rejection_reason,
    construction_attempted, construction_succeeded.

    **Validates: Requirements 8.1, 8.8**
    """
    bridge_logger = logging.getLogger("utils.swing_candidate_bridge")
    handler = logging.handlers.MemoryHandler(capacity=1000)
    handler.setLevel(logging.INFO)
    bridge_logger.addHandler(handler)
    bridge_logger.setLevel(logging.INFO)
    try:
        signals = {
            "sig-test-1": _make_signal(
                symbol=symbol, setup_type=setup_type,
                direction=direction, strength=strength, confidence=confidence,
                ema_trend=ema_trend, market_regime=market_regime,
            )
        }
        process_swing_signals(
            signals=signals,
            profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"},
            portfolio={"equity": 100000},
            cycle_id="cycle-1",
            db=None,
            engine=None,
        )

        # Find the swing_bridge_signal log entry
        handler.flush()
        swing_logs = [r for r in handler.buffer if "swing_bridge_signal" in r.getMessage()]
        assert len(swing_logs) >= 1, "Expected at least one swing_bridge_signal log entry"

        msg = swing_logs[0].getMessage()
        # All required fields must be present in the structured log
        assert "signal_id=" in msg, f"Missing signal_id in log: {msg}"
        assert "symbol=" in msg, f"Missing symbol in log: {msg}"
        assert "raw_label=" in msg, f"Missing raw_label in log: {msg}"
        assert "normalized_label=" in msg, f"Missing normalized_label in log: {msg}"
        assert "rejection_reason=" in msg, f"Missing rejection_reason in log: {msg}"
        assert "construction_attempted=" in msg, f"Missing construction_attempted in log: {msg}"
        assert "construction_succeeded=" in msg, f"Missing construction_succeeded in log: {msg}"
    finally:
        bridge_logger.removeHandler(handler)


@given(
    symbol=symbol_st,
    setup_type=st.sampled_from([
        "sector_rotation", "risk_off_macro_short",
        "breakout_retest", "pullback_continuation",
    ]),
    direction=direction_st,
    strength=strength_st,
    confidence=confidence_st,
    ema_trend=ema_trend_st,
    market_regime=market_regime_st,
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
def test_enabled_mode_log_completeness(mock_mode, symbol, setup_type,
                                        direction, strength, confidence,
                                        ema_trend, market_regime):
    """Property 26: Enabled mode also emits complete structured logs.

    Even in enabled mode, each processed signal gets a structured log entry
    with all required fields, regardless of whether it was accepted or rejected.

    **Validates: Requirements 8.1, 8.8**
    """
    bridge_logger = logging.getLogger("utils.swing_candidate_bridge")
    handler = logging.handlers.MemoryHandler(capacity=1000)
    handler.setLevel(logging.INFO)
    bridge_logger.addHandler(handler)
    bridge_logger.setLevel(logging.INFO)
    try:
        signals = {
            "sig-test-2": _make_signal(
                symbol=symbol, setup_type=setup_type,
                direction=direction, strength=strength, confidence=confidence,
                ema_trend=ema_trend, market_regime=market_regime,
            )
        }
        process_swing_signals(
            signals=signals,
            profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"},
            portfolio={"equity": 100000},
            cycle_id="cycle-1",
            db=None,
            engine=None,
        )

        # Find the swing_bridge_signal log entry
        handler.flush()
        swing_logs = [r for r in handler.buffer if "swing_bridge_signal" in r.getMessage()]
        assert len(swing_logs) >= 1, "Expected at least one swing_bridge_signal log entry"

        msg = swing_logs[0].getMessage()
        # All required fields must be present
        assert "signal_id=" in msg
        assert "symbol=" in msg
        assert "raw_label=" in msg
        assert "normalized_label=" in msg
        assert "rejection_reason=" in msg
        assert "construction_attempted=" in msg
        assert "construction_succeeded=" in msg
    finally:
        bridge_logger.removeHandler(handler)


@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_log_level_is_info(mock_mode):
    """Property 26: Log entries are emitted at INFO level.

    **Validates: Requirements 8.1**
    """
    bridge_logger = logging.getLogger("utils.swing_candidate_bridge")
    handler = logging.handlers.MemoryHandler(capacity=1000)
    handler.setLevel(logging.INFO)
    bridge_logger.addHandler(handler)
    bridge_logger.setLevel(logging.INFO)
    try:
        signals = {
            "sig-1": _make_signal(
                symbol="MSFT", setup_type="sector_rotation",
                direction="LONG", strength="strong", confidence="high",
                ema_trend="bullish", market_regime="risk_on",
            )
        }
        process_swing_signals(
            signals=signals,
            profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"},
            portfolio={"equity": 100000},
            cycle_id="cycle-1",
            db=None,
            engine=None,
        )

        handler.flush()
        swing_logs = [r for r in handler.buffer if "swing_bridge_signal" in r.getMessage()]
        assert len(swing_logs) >= 1
        # All swing bridge signal logs should be at INFO level
        for record in swing_logs:
            assert record.levelno == logging.INFO
    finally:
        bridge_logger.removeHandler(handler)
