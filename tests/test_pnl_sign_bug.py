"""
Bug Condition Exploration Test — Property 1: PnL Sign Consistency

Validates: Requirements 1.1, 1.2, 1.3, 1.4

This test encodes the EXPECTED (correct) behavior: when a CLOSE action is
executed with a negative quantity, the resulting trade should have
sign(pnl) == sign(pnl_pct) or pnl == 0.

On UNFIXED code this test is EXPECTED TO FAIL — failure confirms the bug
exists (negative close_qty flips dollar PnL sign while pnl_pct double-negates
back to the correct sign, producing sign-inconsistent trade records).
"""

import math
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Balance, Trade, Position
from models.case import Case  # noqa: F401 — registers with Base
from agents.portfolio_manager import execute_trade


# ── Helpers ──

def _make_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _make_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


def _sign(x: float) -> int:
    """Return -1, 0, or 1."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


# ── Hypothesis strategies ──

side_strategy = st.sampled_from(["long", "short"])

# Positive floats for prices — avoid subnormals and extremes
entry_price_strategy = st.floats(
    min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)

exit_price_strategy = st.floats(
    min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)

# Negative integers for the bug condition
negative_quantity_strategy = st.integers(min_value=-500, max_value=-1)


# ── Property-based test ──

@given(
    side=side_strategy,
    entry_price=entry_price_strategy,
    exit_price=exit_price_strategy,
    negative_quantity=negative_quantity_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_pnl_sign_consistency_negative_qty(
    side, entry_price, exit_price, negative_quantity
):
    """
    **Validates: Requirements 1.1, 1.2, 1.3**

    Property: For all CLOSE trades where decision_quantity < 0
    (isBugCondition holds), the closed trade must have
    sign(pnl) == sign(pnl_pct) or pnl == 0.

    On UNFIXED code this FAILS because close_qty is negative, flipping
    the dollar PnL sign while pnl_pct double-negates back to the correct sign.
    """
    # Avoid breakeven (entry == exit) to ensure pnl != 0
    assume(abs(entry_price - exit_price) > 0.01)

    # Position quantity must be positive and larger than abs(negative_quantity)
    # so the bug condition triggers: negative_quantity < pos.quantity
    pos_quantity = abs(negative_quantity) + 10

    engine = _make_engine()
    db = _make_session(engine)

    profile_id = "aggressive"

    # Seed balance
    db.add(Balance(profile=profile_id, cash=100_000.0))

    # Seed position
    db.add(Position(
        profile=profile_id,
        symbol="TEST",
        side=side,
        quantity=pos_quantity,
        avg_cost=entry_price,
        opened_at=datetime.utcnow(),
    ))

    # Seed open trade
    db.add(Trade(
        profile=profile_id,
        symbol="TEST",
        direction=side.upper(),
        quantity=pos_quantity,
        entry_price=entry_price,
        entry_time=datetime.utcnow(),
        status="open",
    ))
    db.commit()

    # Build CLOSE decision with negative quantity
    decision = {
        "symbol": "TEST",
        "action": "CLOSE",
        "quantity": negative_quantity,
        "price": exit_price,
        "rationale": "bug exploration test",
    }

    # Mock FinnhubClient.get_quote to return the decision price
    mock_quote = {"price": exit_price, "symbol": "TEST"}
    with patch("agents.portfolio_manager.FinnhubClient") as MockFH:
        mock_instance = MagicMock()
        mock_instance.get_quote.return_value = mock_quote
        MockFH.return_value = mock_instance

        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is True, f"execute_trade should succeed, got: {msg}"

    # Fetch the closed trade
    closed_trade = db.query(Trade).filter_by(
        symbol="TEST", profile=profile_id, status="closed"
    ).first()

    assert closed_trade is not None, "Trade should be closed"
    assert closed_trade.pnl is not None, "pnl should be set"
    assert closed_trade.pnl_pct is not None, "pnl_pct should be set"

    # EXPECTED BEHAVIOR: sign(pnl) == sign(pnl_pct) or pnl == 0
    # On UNFIXED code, this assertion FAILS — proving the bug.
    pnl = closed_trade.pnl
    pnl_pct = closed_trade.pnl_pct

    assert pnl == 0 or _sign(pnl) == _sign(pnl_pct), (
        f"Bug confirmed: PnL sign mismatch for {side} position. "
        f"entry_price={entry_price}, exit_price={exit_price}, "
        f"negative_quantity={negative_quantity}, close_qty used={negative_quantity}. "
        f"pnl={pnl}, pnl_pct={pnl_pct} — "
        f"sign(pnl)={_sign(pnl)}, sign(pnl_pct)={_sign(pnl_pct)}. "
        f"Dollar PnL sign is flipped by negative close_qty while "
        f"pnl_pct double-negates back to the correct sign."
    )

    db.close()
