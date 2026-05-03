"""
Integration tests for the PM pipeline (execute_trade) verifying the
Tier-1 modules (edge score, similarity, portfolio risk) work together.

Each test uses an in-memory SQLite database and mocks external
dependencies (LLM calls, Finnhub API, case library queries).
"""

import json
import logging
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Balance, Trade, Position, AgentMemory, get_session
from models.case import Case  # noqa: F401 — registers with Base
from agents.portfolio_manager import execute_trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _make_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


def _seed_balance(db, profile_id: str, cash: float = 100_000.0):
    """Insert a Balance record so execute_trade can find cash."""
    db.add(Balance(profile=profile_id, cash=cash))
    db.commit()


def _seed_analyst_signal(db, symbol: str, signal: dict):
    """Insert an analyst signal into AgentMemory."""
    db.add(AgentMemory(
        agent="analyst",
        symbol=symbol,
        key="signal",
        value=json.dumps(signal),
    ))
    db.commit()


def _strong_signal(symbol: str = "AAPL") -> dict:
    """Return a strong analyst signal dict that produces a high edge score."""
    return {
        "signal": "LONG",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "gap_and_go",
        "market_regime": "risk_on",
        "bias": "LONG",
        "indicators": {
            "above_vwap": True,
            "ema_trend": "bullish",
            "rsi": 55.0,
            "macd_bias": "bullish",
            "bb_position": "upper",
        },
    }


def _weak_signal(symbol: str = "AAPL") -> dict:
    """Return a weak signal dict that produces a low edge score."""
    return {
        "signal": "LONG",
        "strength": "weak",
        "confidence": "low",
        "setup_type": "unknown_setup",
        "market_regime": "mixed",
        "bias": "LONG",
        "indicators": {},
    }


def _good_sim_stats():
    """Similarity stats that contribute positively to edge score."""
    return {
        "similarity_winrate": 0.80,
        "similarity_avg_r": 2.5,
        "sample_size": 15,
        "similarity_confidence": 1.0,
    }


def _zero_sim_stats():
    """Similarity stats with no data (skip_similarity)."""
    return {
        "similarity_winrate": 0.0,
        "similarity_avg_r": 0.0,
        "sample_size": 0,
        "similarity_confidence": 0.0,
        "skip_similarity": True,
    }


def _base_decision(symbol: str = "AAPL", action: str = "BUY",
                    quantity: int = 50, price: float = 150.0,
                    stop: float = 145.0, target: float = 160.0) -> dict:
    """Build a minimal valid decision dict."""
    return {
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "price": price,
        "stop_loss": stop,
        "target": target,
        "rationale": "integration test trade",
        "setup_type": "gap_and_go",
        "market_regime": "risk_on",
    }


# Common mock targets
_FIND_SIMILAR = "agents.portfolio_manager.find_similar_cases"
_COMPUTE_SIM_STATS = "agents.portfolio_manager.compute_similarity_stats"
_ADJUST_CONFIDENCE = "utils.trade_validator.adjust_confidence"
_VALIDATE_TRADE = "utils.trade_validator.validate_trade"
_CHECK_CORRELATION = "utils.trade_validator.check_correlation"

# Default adjust_confidence return that doesn't block
_CONF_OK = {
    "modifier": 1.0,
    "block": False,
    "reason": "no adjustment",
    "win_rate": 0.60,
    "total_cases": 3,
}


# ---------------------------------------------------------------------------
# 1. test_edge_score_rejection
# ---------------------------------------------------------------------------

def test_edge_score_rejection():
    """
    A weak signal + low confidence + zero similarity → edge score < 0.4 → rejected.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "aggressive"
    _seed_balance(db, profile_id)

    # Seed a weak analyst signal
    _seed_analyst_signal(db, "AAPL", _weak_signal())

    decision = _base_decision(symbol="AAPL", quantity=50, price=150.0,
                              stop=145.0, target=160.0)
    decision["setup_type"] = "unknown_setup"
    decision["market_regime"] = "mixed"

    with (
        patch(_FIND_SIMILAR, return_value=[]),
        patch(_COMPUTE_SIM_STATS, return_value=_zero_sim_stats()),
        patch(_ADJUST_CONFIDENCE, return_value={
            "modifier": 1.0, "block": False, "reason": "no data",
            "win_rate": 0.0, "total_cases": 0,
        }),
        patch(_VALIDATE_TRADE),
        patch(_CHECK_CORRELATION, return_value=""),
    ):
        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is False
    assert "edge score too low" in msg.lower() or "Edge score too low" in msg

    db.close()


# ---------------------------------------------------------------------------
# 2. test_hard_rejection_for_proven_bad_setup
# ---------------------------------------------------------------------------

def test_hard_rejection_for_proven_bad_setup():
    """
    case_stats with sample_size >= 10 and win_rate < 0.35 → hard rejection
    before edge score is even computed.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "moderate"
    _seed_balance(db, profile_id)

    _seed_analyst_signal(db, "AAPL", _strong_signal())

    decision = _base_decision(symbol="AAPL", quantity=30, price=150.0,
                              stop=145.0, target=160.0)

    # adjust_confidence returns terrible stats → _build_case_stats picks them up
    bad_conf = {
        "modifier": 0.0,
        "block": False,  # We don't want adjust_confidence to block at the validator level
        "reason": "bad setup",
        "win_rate": 0.20,
        "total_cases": 15,
    }

    with (
        patch(_FIND_SIMILAR, return_value=[]),
        patch(_COMPUTE_SIM_STATS, return_value=_zero_sim_stats()),
        patch(_ADJUST_CONFIDENCE, return_value=bad_conf),
        patch(_VALIDATE_TRADE),
        patch(_CHECK_CORRELATION, return_value=""),
    ):
        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is False
    assert "hard reject" in msg.lower() or "Hard reject" in msg

    db.close()


# ---------------------------------------------------------------------------
# 3. test_position_size_scaling_with_cap
# ---------------------------------------------------------------------------

def test_position_size_scaling_with_cap():
    """
    Strong signal → high edge score. Verify the final quantity is scaled by
    edge_score and capped at base_size * 1.2.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "aggressive"
    _seed_balance(db, profile_id, cash=100_000.0)

    _seed_analyst_signal(db, "AAPL", _strong_signal())

    base_qty = 100
    decision = _base_decision(symbol="AAPL", quantity=base_qty, price=150.0,
                              stop=145.0, target=160.0)

    good_conf = {
        "modifier": 1.0, "block": False, "reason": "ok",
        "win_rate": 0.80, "total_cases": 20,
    }

    with (
        patch(_FIND_SIMILAR, return_value=[]),
        patch(_COMPUTE_SIM_STATS, return_value=_good_sim_stats()),
        patch(_ADJUST_CONFIDENCE, return_value=good_conf),
        patch(_VALIDATE_TRADE),
        patch(_CHECK_CORRELATION, return_value=""),
    ):
        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is True, f"Trade should have succeeded: {msg}"

    # Verify the trade was recorded with a capped quantity
    trade = db.query(Trade).filter_by(symbol="AAPL", profile=profile_id).first()
    assert trade is not None

    max_allowed = int(base_qty * 1.2)
    assert trade.quantity <= max_allowed, (
        f"Quantity {trade.quantity} exceeds cap {max_allowed}"
    )
    # Quantity should also be > 0
    assert trade.quantity > 0

    db.close()


# ---------------------------------------------------------------------------
# 4. test_adaptive_risk_throttling_after_loss_streak
# ---------------------------------------------------------------------------

def test_adaptive_risk_throttling_after_loss_streak():
    """
    Insert 3+ consecutive losing closed trades. Execute a new BUY.
    Verify the quantity is reduced by 25-50%.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "aggressive"
    _seed_balance(db, profile_id, cash=100_000.0)

    _seed_analyst_signal(db, "AAPL", _strong_signal())

    # Insert 4 consecutive losing trades (most recent first by exit_time)
    now = datetime.utcnow()
    for i in range(4):
        db.add(Trade(
            symbol="SPY",
            direction="LONG",
            quantity=10,
            entry_price=450.0,
            exit_price=445.0,
            entry_time=now - timedelta(hours=8 - i),
            exit_time=now - timedelta(hours=4 - i),
            status="closed",
            pnl=-50.0,
            pnl_pct=-1.1,
            profile=profile_id,
        ))
    db.commit()

    base_qty = 100
    decision = _base_decision(symbol="AAPL", quantity=base_qty, price=150.0,
                              stop=145.0, target=160.0)

    good_conf = {
        "modifier": 1.0, "block": False, "reason": "ok",
        "win_rate": 0.80, "total_cases": 20,
    }

    with (
        patch(_FIND_SIMILAR, return_value=[]),
        patch(_COMPUTE_SIM_STATS, return_value=_good_sim_stats()),
        patch(_ADJUST_CONFIDENCE, return_value=good_conf),
        patch(_VALIDATE_TRADE),
        patch(_CHECK_CORRELATION, return_value=""),
    ):
        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is True, f"Trade should have succeeded: {msg}"

    trade = db.query(Trade).filter_by(
        symbol="AAPL", profile=profile_id, status="open"
    ).first()
    assert trade is not None

    # With 4 losses and adaptive throttle, quantity should be reduced.
    # Edge score scales first, then throttle reduces by 25-50%.
    # The final quantity must be strictly less than the cap (base * 1.2).
    max_unthrottled = int(base_qty * 1.2)
    assert trade.quantity < max_unthrottled, (
        f"Expected throttled quantity < {max_unthrottled}, got {trade.quantity}"
    )
    # Must still be positive
    assert trade.quantity >= 1

    db.close()


# ---------------------------------------------------------------------------
# 5. test_trade_record_persistence
# ---------------------------------------------------------------------------

def test_trade_record_persistence():
    """
    Execute a successful BUY. Verify edge_score, similarity_winrate,
    similarity_sample_size, and similarity_confidence are populated on
    the Trade record.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "aggressive"
    _seed_balance(db, profile_id, cash=100_000.0)

    _seed_analyst_signal(db, "AAPL", _strong_signal())

    decision = _base_decision(symbol="AAPL", quantity=50, price=150.0,
                              stop=145.0, target=160.0)

    good_conf = {
        "modifier": 1.0, "block": False, "reason": "ok",
        "win_rate": 0.75, "total_cases": 12,
    }

    with (
        patch(_FIND_SIMILAR, return_value=[]),
        patch(_COMPUTE_SIM_STATS, return_value=_good_sim_stats()),
        patch(_ADJUST_CONFIDENCE, return_value=good_conf),
        patch(_VALIDATE_TRADE),
        patch(_CHECK_CORRELATION, return_value=""),
    ):
        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is True, f"Trade should have succeeded: {msg}"

    trade = db.query(Trade).filter_by(symbol="AAPL", profile=profile_id).first()
    assert trade is not None

    # All four edge data fields must be populated (not None)
    assert trade.edge_score is not None, "edge_score should be populated"
    assert trade.similarity_winrate is not None, "similarity_winrate should be populated"
    assert trade.similarity_sample_size is not None, "similarity_sample_size should be populated"
    assert trade.similarity_confidence is not None, "similarity_confidence should be populated"

    # Sanity ranges
    assert 0.0 <= trade.edge_score <= 1.0
    assert 0.0 <= trade.similarity_winrate <= 1.0
    assert trade.similarity_sample_size >= 0
    assert 0.0 <= trade.similarity_confidence <= 1.0

    db.close()


# ---------------------------------------------------------------------------
# 6. test_portfolio_risk_rejection_multi_bucket
# ---------------------------------------------------------------------------

def test_portfolio_risk_rejection_multi_bucket():
    """
    Fill a bucket near 50% with existing positions, then try to add another
    trade in the same bucket → rejected by portfolio risk.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "aggressive"
    cash = 100_000.0
    _seed_balance(db, profile_id, cash=cash)

    _seed_analyst_signal(db, "AMD", _strong_signal("AMD"))

    # Fill the "semis" bucket close to 50%.
    # With NVDA at 200 * 480 = 96,000 and cash = 100k,
    # total_equity = 100k + 96k = 196k.
    # semis bucket = 96k / 196k ≈ 49%.
    # Adding even a small AMD position pushes it over 50%.
    db.add(Position(
        symbol="NVDA",
        quantity=200,
        avg_cost=480.0,
        profile=profile_id,
        side="long",
    ))
    db.commit()

    # Try to add AMD (also in semis bucket) — should push semis over 50%
    decision = _base_decision(symbol="AMD", action="BUY", quantity=30,
                              price=150.0, stop=145.0, target=160.0)

    good_conf = {
        "modifier": 1.0, "block": False, "reason": "ok",
        "win_rate": 0.80, "total_cases": 20,
    }

    with (
        patch(_FIND_SIMILAR, return_value=[]),
        patch(_COMPUTE_SIM_STATS, return_value=_good_sim_stats()),
        patch(_ADJUST_CONFIDENCE, return_value=good_conf),
        patch(_VALIDATE_TRADE),
        patch(_CHECK_CORRELATION, return_value=""),
    ):
        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is False, f"Trade should have been rejected: {msg}"
    assert "exposure" in msg.lower() or "exceeded" in msg.lower(), (
        f"Expected portfolio risk rejection message, got: {msg}"
    )

    db.close()


# ---------------------------------------------------------------------------
# 7. test_structured_logging_output
# ---------------------------------------------------------------------------

def test_structured_logging_output(caplog):
    """
    Capture log output during a trade execution and verify EDGE SCORE,
    PORTFOLIO RISK, and DECISION log blocks are present.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "aggressive"
    _seed_balance(db, profile_id, cash=100_000.0)

    _seed_analyst_signal(db, "AAPL", _strong_signal())

    decision = _base_decision(symbol="AAPL", quantity=50, price=150.0,
                              stop=145.0, target=160.0)

    good_conf = {
        "modifier": 1.0, "block": False, "reason": "ok",
        "win_rate": 0.75, "total_cases": 12,
    }

    with caplog.at_level(logging.INFO, logger="agents.portfolio_manager"):
        with (
            patch(_FIND_SIMILAR, return_value=[]),
            patch(_COMPUTE_SIM_STATS, return_value=_good_sim_stats()),
            patch(_ADJUST_CONFIDENCE, return_value=good_conf),
            patch(_VALIDATE_TRADE),
            patch(_CHECK_CORRELATION, return_value=""),
        ):
            ok, msg = execute_trade(db, decision, profile_id)

    log_text = caplog.text

    assert "EDGE SCORE" in log_text, (
        f"Expected 'EDGE SCORE' in logs, got:\n{log_text}"
    )
    assert "PORTFOLIO RISK" in log_text, (
        f"Expected 'PORTFOLIO RISK' in logs, got:\n{log_text}"
    )
    assert "DECISION" in log_text, (
        f"Expected 'DECISION' in logs, got:\n{log_text}"
    )

    db.close()


# ---------------------------------------------------------------------------
# 8. test_entry_contract_persisted_on_buy
# ---------------------------------------------------------------------------

def test_entry_contract_persisted_on_buy():
    """
    Execute a successful BUY with a signal containing structured invalidation.
    Verify thesis, setup_type, and invalidators are persisted on the Trade record.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "aggressive"
    _seed_balance(db, profile_id, cash=100_000.0)

    signal_with_invalidation = {
        **_strong_signal(),
        "invalidation": [
            {
                "type": "price_below_level",
                "reference": "145.00",
                "confirmation": "5m_close",
                "lookback_bars": 1,
            }
        ],
    }
    _seed_analyst_signal(db, "AAPL", signal_with_invalidation)

    decision = _base_decision(symbol="AAPL", quantity=50, price=150.0,
                              stop=145.0, target=160.0)

    good_conf = {
        "modifier": 1.0, "block": False, "reason": "ok",
        "win_rate": 0.75, "total_cases": 12,
    }

    with (
        patch(_FIND_SIMILAR, return_value=[]),
        patch(_COMPUTE_SIM_STATS, return_value=_good_sim_stats()),
        patch(_ADJUST_CONFIDENCE, return_value=good_conf),
        patch(_VALIDATE_TRADE),
        patch(_CHECK_CORRELATION, return_value=""),
    ):
        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is True, f"Trade should have succeeded: {msg}"

    trade = db.query(Trade).filter_by(symbol="AAPL", profile=profile_id).first()
    assert trade is not None

    # Entry contract fields must be populated
    assert trade.thesis is not None, "thesis should be populated"
    assert trade.thesis != "", "thesis should not be empty"
    assert trade.setup_type is not None, "setup_type should be populated"
    assert trade.setup_type == "gap_and_go", "setup_type should match signal"

    # Invalidators should be valid JSON
    assert trade.invalidators is not None, "invalidators should be populated"
    invalidators = json.loads(trade.invalidators)
    assert isinstance(invalidators, list)
    assert len(invalidators) >= 1
    assert invalidators[0]["type"] == "price_below_level"
    assert invalidators[0]["reference"] == "145.00"

    db.close()


# ---------------------------------------------------------------------------
# 9. test_entry_contract_persisted_on_short
# ---------------------------------------------------------------------------

def test_entry_contract_persisted_on_short():
    """
    Execute a successful SHORT. Verify entry contract fields are persisted.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "aggressive"
    _seed_balance(db, profile_id, cash=100_000.0)

    signal = {
        **_strong_signal(),
        "bias": "SHORT",
        "signal": "SHORT",
    }
    _seed_analyst_signal(db, "AAPL", signal)

    decision = _base_decision(symbol="AAPL", action="SHORT", quantity=50,
                              price=150.0, stop=155.0, target=140.0)

    good_conf = {
        "modifier": 1.0, "block": False, "reason": "ok",
        "win_rate": 0.75, "total_cases": 12,
    }

    with (
        patch(_FIND_SIMILAR, return_value=[]),
        patch(_COMPUTE_SIM_STATS, return_value=_good_sim_stats()),
        patch(_ADJUST_CONFIDENCE, return_value=good_conf),
        patch(_VALIDATE_TRADE),
        patch(_CHECK_CORRELATION, return_value=""),
    ):
        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is True, f"Trade should have succeeded: {msg}"

    trade = db.query(Trade).filter_by(symbol="AAPL", profile=profile_id).first()
    assert trade is not None

    # Entry contract fields must be populated
    assert trade.thesis is not None, "thesis should be populated"
    assert trade.setup_type is not None, "setup_type should be populated"
    assert trade.invalidators is not None, "invalidators should be populated"

    # Invalidators should be valid JSON with at least one entry
    invalidators = json.loads(trade.invalidators)
    assert isinstance(invalidators, list)
    assert len(invalidators) >= 1

    db.close()


# ---------------------------------------------------------------------------
# 10. test_entry_contract_fallback_to_stop_price_default
# ---------------------------------------------------------------------------

def test_entry_contract_fallback_to_stop_price_default():
    """
    Execute a BUY with a signal that has no invalidation field.
    Verify the default stop-price-based invalidator is used.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "aggressive"
    _seed_balance(db, profile_id, cash=100_000.0)

    # Signal without invalidation field
    _seed_analyst_signal(db, "AAPL", _strong_signal())

    decision = _base_decision(symbol="AAPL", quantity=50, price=150.0,
                              stop=145.0, target=160.0)

    good_conf = {
        "modifier": 1.0, "block": False, "reason": "ok",
        "win_rate": 0.75, "total_cases": 12,
    }

    with (
        patch(_FIND_SIMILAR, return_value=[]),
        patch(_COMPUTE_SIM_STATS, return_value=_good_sim_stats()),
        patch(_ADJUST_CONFIDENCE, return_value=good_conf),
        patch(_VALIDATE_TRADE),
        patch(_CHECK_CORRELATION, return_value=""),
    ):
        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is True, f"Trade should have succeeded: {msg}"

    trade = db.query(Trade).filter_by(symbol="AAPL", profile=profile_id).first()
    assert trade is not None
    assert trade.invalidators is not None

    invalidators = json.loads(trade.invalidators)
    assert len(invalidators) >= 1
    # Default invalidator should reference the stop price
    assert invalidators[0]["reference"] == "145.0"
    assert invalidators[0]["type"] == "price_below_level"

    db.close()

# ---------------------------------------------------------------------------
# 10. test_high_winrate_fast_intraday_stop_buffer_is_enforced
# ---------------------------------------------------------------------------

def test_high_winrate_fast_intraday_stop_buffer_is_enforced():
    """
    A setup with >60% historical WR and sub-60-minute intraday horizon must
    persist a stop at least 1.5% from entry, even if the PM proposed tighter.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "aggressive"
    _seed_balance(db, profile_id, cash=100_000.0)

    signal = _strong_signal("AMD")
    signal["timeframe"] = "5-15 min"
    _seed_analyst_signal(db, "AMD", signal)

    decision = _base_decision(
        symbol="AMD", quantity=50, price=100.0,
        stop=99.25, target=104.0,
    )
    decision["timeframe"] = "5-15 min"

    high_wr_conf = {
        "modifier": 1.0, "block": False, "reason": "ok",
        "win_rate": 0.72, "total_cases": 20,
    }

    with (
        patch(_FIND_SIMILAR, return_value=[]),
        patch(_COMPUTE_SIM_STATS, return_value=_good_sim_stats()),
        patch(_ADJUST_CONFIDENCE, return_value=high_wr_conf),
        patch(_VALIDATE_TRADE),
        patch(_CHECK_CORRELATION, return_value=""),
        patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
    ):
        mock_fh = MagicMock()
        mock_fh.get_quote.return_value = {"price": 100.0}
        mock_fh_cls.return_value = mock_fh
        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is True, f"Trade should have succeeded: {msg}"
    trade = db.query(Trade).filter_by(symbol="AMD", profile=profile_id).first()
    assert trade is not None
    assert trade.stop_price == 98.5
    assert decision["stop_loss"] == 98.5

    db.close()


# ---------------------------------------------------------------------------
# 11. test_high_momentum_asset_cooldown_blocks_cascading_reentry
# ---------------------------------------------------------------------------

def test_high_momentum_asset_cooldown_blocks_cascading_reentry():
    """
    A fresh AMD stop/loss in any profile should block another profile from
    immediately re-entering and creating cascading correlated exits.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "aggressive"
    _seed_balance(db, profile_id, cash=100_000.0)
    _seed_analyst_signal(db, "AMD", _strong_signal("AMD"))

    now = datetime.utcnow()
    db.add(Trade(
        symbol="AMD",
        direction="LONG",
        quantity=25,
        entry_price=100.0,
        exit_price=98.5,
        entry_time=now - timedelta(minutes=20),
        exit_time=now - timedelta(minutes=5),
        status="closed",
        pnl=-37.5,
        pnl_pct=-1.5,
        profile="moderate",
        reason_exit="Stop loss hit",
    ))
    db.commit()

    decision = _base_decision(symbol="AMD", quantity=50, price=100.0,
                              stop=98.0, target=105.0)
    high_wr_conf = {
        "modifier": 1.0, "block": False, "reason": "ok",
        "win_rate": 0.72, "total_cases": 20,
    }

    with (
        patch(_FIND_SIMILAR, return_value=[]),
        patch(_COMPUTE_SIM_STATS, return_value=_good_sim_stats()),
        patch(_ADJUST_CONFIDENCE, return_value=high_wr_conf),
        patch(_VALIDATE_TRADE),
        patch(_CHECK_CORRELATION, return_value=""),
    ):
        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is False
    assert "cooldown active for AMD".lower() in msg.lower()
    assert db.query(Trade).filter_by(symbol="AMD", profile=profile_id, status="open").first() is None

    db.close()
