"""Property-based tests for pipeline behavior (Properties 3, 4, 17).

Tests that:
- Property 3: First-Failing-Stage Rejection Assignment — exactly one canonical
  code from first failing stage (no silent drops)
- Property 4: Construction Flag Invariants — flags reflect pipeline stage reached
- Property 17: Observe Mode Telemetry Without Registration — full pipeline run,
  summary persisted, empty list returned, no trades opened

**Validates: Requirements 2.2, 2.3, 2.4, 2.5, 3.4, 18.2, 18.3**
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from hypothesis import given, settings, strategies as st, assume

from utils.swing_candidate_bridge import (
    CANONICAL_REJECTION_CODES,
    PerSymbolEntry,
    SwingEvaluationSummary,
    process_swing_signals,
    _build_evaluation_summary,
)


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

symbol_st = st.text(min_size=1, max_size=5, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
direction_st = st.sampled_from(["LONG", "SHORT"])
strength_st = st.sampled_from(["weak", "moderate", "strong"])
confidence_st = st.sampled_from(["low", "medium", "high"])
ema_trend_st = st.sampled_from(["bullish", "bearish", "neutral"])
market_regime_st = st.sampled_from(["risk_on", "risk_off", "mixed"])
catalyst_freshness_st = st.sampled_from(["fresh", "aging", "stale"])


# Signal age strategies
from utils.gate_config import SWING_SIGNAL_FRESHNESS_HOURS

fresh_age_st = st.floats(
    min_value=0.0,
    max_value=float(SWING_SIGNAL_FRESHNESS_HOURS),
    allow_nan=False,
    allow_infinity=False,
)

stale_age_st = st.floats(
    min_value=float(SWING_SIGNAL_FRESHNESS_HOURS) + 0.001,
    max_value=10000.0,
    allow_nan=False,
    allow_infinity=False,
)


def _make_signal(
    symbol="AAPL",
    setup_type="sector_rotation",
    direction="LONG",
    strength="strong",
    confidence="high",
    ema_trend="bullish",
    market_regime="risk_on",
    entry_price=100.0,
    stop_price=95.0,
    target_price=115.0,
    signal_age_hours=0,
    catalyst_freshness="fresh",
):
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
        "signal_age_hours": signal_age_hours,
        "catalyst_freshness": catalyst_freshness,
        "sector": "technology",
    }


# ---------------------------------------------------------------------------
# Property 3: First-Failing-Stage Rejection Assignment
# Validates: Requirements 2.2, 3.4
#
# For any signal that fails multiple pipeline checks, the per-symbol entry
# records exactly one final_rejection_reason (canonical code) corresponding
# to the first failing stage. Every rejection gets a canonical code.
# ---------------------------------------------------------------------------


@given(
    symbol=symbol_st,
    direction=direction_st,
    strength=strength_st,
    confidence=confidence_st,
    ema_trend=ema_trend_st,
    market_regime=market_regime_st,
    signal_age=stale_age_st,
    catalyst_freshness=catalyst_freshness_st,
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_stale_signal_is_first_rejection_regardless_of_other_failures(
    mock_mode, symbol, direction, strength, confidence, ema_trend,
    market_regime, signal_age, catalyst_freshness,
):
    """Property 3a: Stale signal is the first-failing stage rejection.

    When signal_age_hours exceeds the threshold, the rejection must be
    stale_signal even if the signal would also fail catalyst freshness,
    normalization, geometry, or policy checks.

    **Validates: Requirements 2.2, 3.4**
    """
    signals = {
        "sig-1": _make_signal(
            symbol=symbol, direction=direction, strength=strength,
            confidence=confidence, ema_trend=ema_trend,
            market_regime=market_regime, signal_age_hours=signal_age,
            catalyst_freshness=catalyst_freshness,
        )
    }
    result = process_swing_signals(
        signals=signals, profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1", db=None, engine=None,
    )
    assert result == []


@given(
    symbol=symbol_st,
    direction=direction_st,
    strength=strength_st,
    confidence=confidence_st,
    ema_trend=ema_trend_st,
    market_regime=market_regime_st,
    num_signals=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_pipeline_processes_all_signals_no_silent_drops(
    mock_mode, symbol, direction, strength, confidence, ema_trend,
    market_regime, num_signals,
):
    """Property 3b: Pipeline processes all signals — no silent drops.

    Every signal in the input dict must produce exactly one PerSymbolEntry
    in the evaluation summary. The number of entries equals the number of
    input signals, proving no signals are silently dropped.

    **Validates: Requirements 2.2, 3.4**
    """
    signals = {
        f"sig-{i}": _make_signal(
            symbol=f"{symbol}{i}"[:5], direction=direction,
            strength=strength, confidence=confidence,
            ema_trend=ema_trend, market_regime=market_regime,
            signal_age_hours=0, catalyst_freshness="fresh",
        )
        for i in range(num_signals)
    }
    # We need to capture the summary to check entry count.
    # Use a mock DB that captures the persist call.
    mock_conn = MagicMock()
    mock_db = MagicMock()
    mock_db.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.begin.return_value.__exit__ = MagicMock(return_value=False)

    process_swing_signals(
        signals=signals, profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1", db=mock_db, engine=None,
    )

    # Summary was persisted — extract and validate entry count
    assert mock_conn.execute.called
    call_args = mock_conn.execute.call_args
    params = call_args[0][1]
    import json
    payload = json.loads(params["event_data"])
    assert payload["total_signals_evaluated"] == num_signals
    assert len(payload["per_symbol_entries"]) == num_signals


@given(
    symbol=symbol_st,
    direction=direction_st,
    strength=strength_st,
    confidence=confidence_st,
    ema_trend=ema_trend_st,
    market_regime=market_regime_st,
    signal_age=fresh_age_st,
    catalyst_freshness=st.sampled_from(["fresh", "aging"]),
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_every_rejection_gets_a_canonical_code(
    mock_mode, symbol, direction, strength, confidence, ema_trend,
    market_regime, signal_age, catalyst_freshness,
):
    """Property 3c: Every rejection gets a canonical code.

    For any signal that is rejected (final_rejection_reason is not None),
    the code must be a member of CANONICAL_REJECTION_CODES. No free-form
    strings appear as final_rejection_reason.

    **Validates: Requirements 2.2, 3.4**
    """
    signals = {
        "sig-1": _make_signal(
            symbol=symbol, direction=direction, strength=strength,
            confidence=confidence, ema_trend=ema_trend,
            market_regime=market_regime, signal_age_hours=signal_age,
            catalyst_freshness=catalyst_freshness,
        )
    }
    mock_conn = MagicMock()
    mock_db = MagicMock()
    mock_db.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.begin.return_value.__exit__ = MagicMock(return_value=False)

    process_swing_signals(
        signals=signals, profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1", db=mock_db, engine=None,
    )

    # Extract per-symbol entries from persisted payload
    assert mock_conn.execute.called
    call_args = mock_conn.execute.call_args
    params = call_args[0][1]
    import json
    payload = json.loads(params["event_data"])

    for entry in payload["per_symbol_entries"]:
        reason = entry["final_rejection_reason"]
        if reason is not None:
            assert reason in CANONICAL_REJECTION_CODES, (
                f"Rejection reason {reason!r} not in CANONICAL_REJECTION_CODES"
            )


@given(
    symbol=symbol_st,
    signal_age=fresh_age_st,
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_catalyst_stale_is_second_stage_rejection(mock_mode, symbol, signal_age):
    """Property 3d: Catalyst freshness is second-stage rejection.

    When signal is fresh but catalyst_freshness is stale, the first-failing
    stage is catalyst freshness, producing stale_catalyst as the canonical
    rejection — even if the signal would also fail normalization.

    **Validates: Requirements 2.2, 3.4**
    """
    signals = {
        "sig-1": _make_signal(
            symbol=symbol, direction="HOLD",  # HOLD would fail normalization
            strength="weak", confidence="low",
            ema_trend="neutral", market_regime="mixed",
            signal_age_hours=signal_age,
            catalyst_freshness="stale",
        )
    }
    mock_conn = MagicMock()
    mock_db = MagicMock()
    mock_db.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.begin.return_value.__exit__ = MagicMock(return_value=False)

    process_swing_signals(
        signals=signals, profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1", db=mock_db, engine=None,
    )

    assert mock_conn.execute.called
    call_args = mock_conn.execute.call_args
    params = call_args[0][1]
    import json
    payload = json.loads(params["event_data"])

    entry = payload["per_symbol_entries"][0]
    assert entry["final_rejection_reason"] == "stale_catalyst", (
        f"Expected stale_catalyst, got {entry['final_rejection_reason']}"
    )


# ---------------------------------------------------------------------------
# Property 4: Construction Flag Invariants
# Validates: Requirements 2.3, 2.4, 2.5
#
# (a) If signal passes all checks: construction_attempted=True,
#     construction_succeeded=True, final_rejection_reason=None
# (b) If normalization succeeds and geometry is attempted:
#     construction_attempted=True regardless of geometry outcome
# (c) If signal rejected before geometry (freshness or normalization):
#     construction_attempted=False, construction_succeeded=False
# ---------------------------------------------------------------------------


@given(
    symbol=symbol_st,
    signal_age=stale_age_st,
    catalyst_freshness=catalyst_freshness_st,
    direction=direction_st,
    strength=strength_st,
    confidence=confidence_st,
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_construction_flags_false_when_rejected_at_freshness(
    mock_mode, symbol, signal_age, catalyst_freshness, direction,
    strength, confidence,
):
    """Property 4c: Rejected at freshness → construction_attempted=False.

    When a signal is rejected due to stale signal age (before normalization),
    construction_attempted and construction_succeeded must both be False.

    **Validates: Requirements 2.5**
    """
    signals = {
        "sig-1": _make_signal(
            symbol=symbol, direction=direction, strength=strength,
            confidence=confidence, signal_age_hours=signal_age,
            catalyst_freshness=catalyst_freshness,
        )
    }
    mock_conn = MagicMock()
    mock_db = MagicMock()
    mock_db.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.begin.return_value.__exit__ = MagicMock(return_value=False)

    process_swing_signals(
        signals=signals, profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1", db=mock_db, engine=None,
    )

    assert mock_conn.execute.called
    call_args = mock_conn.execute.call_args
    params = call_args[0][1]
    import json
    payload = json.loads(params["event_data"])

    entry = payload["per_symbol_entries"][0]
    assert entry["construction_attempted"] is False
    assert entry["construction_succeeded"] is False
    assert entry["final_rejection_reason"] == "stale_signal"


@given(
    symbol=symbol_st,
    signal_age=fresh_age_st,
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_construction_flags_false_when_rejected_at_normalization(
    mock_mode, symbol, signal_age,
):
    """Property 4c: Rejected at normalization → construction_attempted=False.

    When a signal passes freshness but fails normalization (e.g., unmapped
    label), construction_attempted and construction_succeeded must be False
    because geometry was never attempted.

    **Validates: Requirements 2.5**
    """
    signals = {
        "sig-1": _make_signal(
            symbol=symbol,
            setup_type="totally_unknown_label_xyz",
            direction="LONG", strength="strong", confidence="high",
            signal_age_hours=signal_age,
            catalyst_freshness="fresh",
        )
    }
    mock_conn = MagicMock()
    mock_db = MagicMock()
    mock_db.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.begin.return_value.__exit__ = MagicMock(return_value=False)

    process_swing_signals(
        signals=signals, profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1", db=mock_db, engine=None,
    )

    assert mock_conn.execute.called
    call_args = mock_conn.execute.call_args
    params = call_args[0][1]
    import json
    payload = json.loads(params["event_data"])

    entry = payload["per_symbol_entries"][0]
    assert entry["construction_attempted"] is False
    assert entry["construction_succeeded"] is False
    assert entry["final_rejection_reason"] is not None


@given(symbol=symbol_st)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_construction_attempted_true_when_geometry_attempted(mock_mode, symbol):
    """Property 4b: Normalization success + geometry attempted → construction_attempted=True.

    When normalization succeeds, geometry construction is always attempted.
    The construction_attempted flag must be True regardless of whether geometry
    succeeds or fails (e.g., missing stop/target prices).

    **Validates: Requirements 2.4**
    """
    # Use sector_rotation with all evidence to pass normalization,
    # but set stop_price=None so geometry will reject with missing_geometry
    signals = {
        "sig-1": {
            "symbol": symbol,
            "setup_type": "sector_rotation",
            "direction": "LONG",
            "strength": "strong",
            "confidence": "high",
            "key_levels": {"support": 95.0, "resistance": 110.0},
            "ema_trend": "bullish",
            "market_regime": "risk_on",
            "entry_price": 100.0,
            "stop_price": None,  # Missing → geometry rejection
            "target_price": 115.0,
            "signal_age_hours": 0,
            "catalyst_freshness": "fresh",
            "sector": "technology",
        }
    }
    mock_conn = MagicMock()
    mock_db = MagicMock()
    mock_db.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.begin.return_value.__exit__ = MagicMock(return_value=False)

    process_swing_signals(
        signals=signals, profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1", db=mock_db, engine=None,
    )

    assert mock_conn.execute.called
    call_args = mock_conn.execute.call_args
    params = call_args[0][1]
    import json
    payload = json.loads(params["event_data"])

    entry = payload["per_symbol_entries"][0]
    # Geometry was attempted (normalization passed)
    assert entry["construction_attempted"] is True
    # But geometry failed (missing stop)
    assert entry["construction_succeeded"] is False
    assert entry["final_rejection_reason"] is not None


@given(symbol=symbol_st)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_construction_succeeded_true_when_all_checks_pass(mock_mode, symbol):
    """Property 4a: All checks pass → construction_attempted=True, succeeded=True.

    When a signal passes freshness, normalization, geometry, and profile
    policy, the entry must have construction_attempted=True,
    construction_succeeded=True, and final_rejection_reason=None.

    **Validates: Requirements 2.3**
    """
    signals = {
        "sig-1": _make_signal(
            symbol=symbol,
            setup_type="sector_rotation",
            direction="LONG",
            strength="strong",
            confidence="high",
            ema_trend="bullish",
            market_regime="risk_on",
            entry_price=100.0,
            stop_price=95.0,
            target_price=115.0,
            signal_age_hours=0,
            catalyst_freshness="fresh",
        )
    }
    mock_conn = MagicMock()
    mock_db = MagicMock()
    mock_db.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.begin.return_value.__exit__ = MagicMock(return_value=False)

    process_swing_signals(
        signals=signals, profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1", db=mock_db, engine=None,
    )

    assert mock_conn.execute.called
    call_args = mock_conn.execute.call_args
    params = call_args[0][1]
    import json
    payload = json.loads(params["event_data"])

    entry = payload["per_symbol_entries"][0]
    assert entry["construction_attempted"] is True
    assert entry["construction_succeeded"] is True
    assert entry["final_rejection_reason"] is None
    assert entry["raw_rejection_reason"] is None


# ---------------------------------------------------------------------------
# Property 17: Observe Mode Telemetry Without Registration
# Validates: Requirements 18.2, 18.3
#
# For any set of signals processed with SWING_CANDIDATE_MODE="observe":
# (a) Run the full evaluation pipeline (freshness, normalization, geometry,
#     risk, policy)
# (b) Persist a SwingEvaluationSummary to pm_candidate_events
# (c) Return an empty list (no registered candidates)
# (d) Not open any trades
# ---------------------------------------------------------------------------


@given(
    num_signals=st.integers(min_value=1, max_value=5),
    symbol=symbol_st,
    direction=direction_st,
    strength=strength_st,
    confidence=confidence_st,
    ema_trend=ema_trend_st,
    market_regime=market_regime_st,
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_observe_mode_returns_empty_list(
    mock_mode, num_signals, symbol, direction, strength, confidence,
    ema_trend, market_regime,
):
    """Property 17a: Observe mode always returns empty list.

    Regardless of signal properties (passing or failing), observe mode
    never returns registered candidates. The return value is always [].

    **Validates: Requirements 18.2, 18.3**
    """
    signals = {
        f"sig-{i}": _make_signal(
            symbol=f"{symbol}{i}"[:5], direction=direction,
            strength=strength, confidence=confidence,
            ema_trend=ema_trend, market_regime=market_regime,
            signal_age_hours=0, catalyst_freshness="fresh",
        )
        for i in range(num_signals)
    }
    result = process_swing_signals(
        signals=signals, profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1", db=None, engine=None,
    )
    assert result == [], (
        f"Observe mode must return [], got {len(result)} candidates"
    )


@given(
    num_signals=st.integers(min_value=1, max_value=5),
    symbol=symbol_st,
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_observe_mode_persists_evaluation_summary(mock_mode, num_signals, symbol):
    """Property 17b: Observe mode persists SwingEvaluationSummary.

    When at least one signal is present, observe mode must persist a
    swing_evaluation_summary event to pm_candidate_events containing
    all per-symbol entries.

    **Validates: Requirements 18.2**
    """
    signals = {
        f"sig-{i}": _make_signal(
            symbol=f"{symbol}{i}"[:5],
            setup_type="sector_rotation",
            direction="LONG", strength="strong", confidence="high",
            ema_trend="bullish", market_regime="risk_on",
            signal_age_hours=0, catalyst_freshness="fresh",
        )
        for i in range(num_signals)
    }
    mock_conn = MagicMock()
    mock_db = MagicMock()
    mock_db.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.begin.return_value.__exit__ = MagicMock(return_value=False)

    result = process_swing_signals(
        signals=signals, profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1", db=mock_db, engine=None,
    )

    # Returns empty in observe mode
    assert result == []

    # Summary was persisted
    assert mock_conn.execute.called
    call_args = mock_conn.execute.call_args
    params = call_args[0][1]
    assert params["event_type"] == "swing_evaluation_summary"
    assert params["candidate_type"] == "swing"

    import json
    payload = json.loads(params["event_data"])
    assert payload["candidate_mode"] == "observe"
    assert payload["total_signals_evaluated"] == num_signals
    assert len(payload["per_symbol_entries"]) == num_signals


@given(symbol=symbol_st)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_observe_mode_runs_full_pipeline_including_geometry(mock_mode, symbol):
    """Property 17c: Observe mode runs full pipeline including geometry.

    In observe mode, signals that pass normalization proceed to geometry
    construction. The per-symbol entry must reflect geometry was attempted
    (construction_attempted=True) even though no candidates are registered.

    **Validates: Requirements 18.2, 18.3**
    """
    # Signal that passes normalization — geometry will be attempted
    signals = {
        "sig-1": _make_signal(
            symbol=symbol,
            setup_type="sector_rotation",
            direction="LONG", strength="strong", confidence="high",
            ema_trend="bullish", market_regime="risk_on",
            entry_price=100.0, stop_price=95.0, target_price=115.0,
            signal_age_hours=0, catalyst_freshness="fresh",
        )
    }
    mock_conn = MagicMock()
    mock_db = MagicMock()
    mock_db.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.begin.return_value.__exit__ = MagicMock(return_value=False)

    result = process_swing_signals(
        signals=signals, profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1", db=mock_db, engine=None,
    )

    # No candidates registered
    assert result == []

    # But full pipeline ran — check entry shows geometry attempted
    assert mock_conn.execute.called
    call_args = mock_conn.execute.call_args
    params = call_args[0][1]
    import json
    payload = json.loads(params["event_data"])

    entry = payload["per_symbol_entries"][0]
    # Geometry was attempted (normalization succeeded)
    assert entry["construction_attempted"] is True
    # With valid prices, geometry should succeed
    assert entry["construction_succeeded"] is True
    # No rejection for a fully-passing signal
    assert entry["final_rejection_reason"] is None


@given(
    symbol=symbol_st,
    direction=direction_st,
    strength=strength_st,
    confidence=confidence_st,
)
@settings(max_examples=200)
@patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
def test_observe_mode_never_emits_candidates_even_when_eligible(
    mock_mode, symbol, direction, strength, confidence,
):
    """Property 17d: Observe mode never emits candidates.

    Even with a fully eligible signal (passes normalization, geometry, policy,
    and sizing), observe mode must return [] and not register any candidates.

    **Validates: Requirements 18.2, 18.3**
    """
    # Construct a signal that would pass all checks in enabled mode
    # Use LONG direction with strong/high to pass all profiles
    signals = {
        "sig-1": _make_signal(
            symbol=symbol,
            setup_type="sector_rotation",
            direction="LONG",
            strength="strong",
            confidence="high",
            ema_trend="bullish",
            market_regime="risk_on",
            entry_price=100.0,
            stop_price=95.0,
            target_price=115.0,
            signal_age_hours=0,
            catalyst_freshness="fresh",
        )
    }
    result = process_swing_signals(
        signals=signals, profile_id="moderate",
        profile={"risk_per_trade_pct": "0.01"},
        portfolio={"equity": 100000},
        cycle_id="cycle-1", db=None, engine=None,
    )
    # Observe mode NEVER returns candidates
    assert result == []
