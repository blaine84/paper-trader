"""
Preservation Property Tests — Property 2: Positive Quantity PnL Unchanged

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5

These tests verify that CLOSE actions with positive quantities produce
correct PnL values on the UNFIXED code. They establish a baseline that
must be preserved after the bugfix is applied.

Run BEFORE implementing the fix. All tests should PASS on unfixed code.
"""

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


def _setup_and_close(side, entry_price, exit_price, pos_quantity, close_quantity):
    """
    Set up an in-memory DB with a position and open trade, then execute
    a CLOSE decision. Returns the closed Trade record.
    """
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

    decision = {
        "symbol": "TEST",
        "action": "CLOSE",
        "quantity": close_quantity,
        "price": exit_price,
        "rationale": "preservation test",
    }

    mock_quote = {"price": exit_price, "symbol": "TEST"}
    with patch("agents.portfolio_manager.FinnhubClient") as MockFH:
        mock_instance = MagicMock()
        mock_instance.get_quote.return_value = mock_quote
        MockFH.return_value = mock_instance

        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is True, f"execute_trade should succeed, got: {msg}"

    closed_trade = db.query(Trade).filter_by(
        symbol="TEST", profile=profile_id, status="closed"
    ).first()

    assert closed_trade is not None, "Trade should be closed"
    assert closed_trade.pnl is not None, "pnl should be set"
    assert closed_trade.pnl_pct is not None, "pnl_pct should be set"

    return closed_trade


# ═══════════════════════════════════════════════════════════════════════
# Concrete observation tests — verify expected values on UNFIXED code
# ═══════════════════════════════════════════════════════════════════════

class TestConcreteObservations:
    """Observe PnL values for specific scenarios on unfixed code."""

    def test_long_winning_full_close(self):
        """long 100 shares at $200, CLOSE qty=100 at $206 → pnl=600.00, pnl_pct=3.00"""
        trade = _setup_and_close("long", 200.0, 206.0, 100, 100)
        assert trade.pnl == pytest.approx(600.00)
        assert trade.pnl_pct == pytest.approx(3.00)

    def test_short_winning_full_close(self):
        """short 50 shares at $10, CLOSE qty=50 at $9 → pnl=50.00, pnl_pct=10.00"""
        trade = _setup_and_close("short", 10.0, 9.0, 50, 50)
        assert trade.pnl == pytest.approx(50.00)
        assert trade.pnl_pct == pytest.approx(10.00)

    def test_long_losing_full_close(self):
        """long 100 shares at $200, CLOSE qty=100 at $194 → pnl=-600.00, pnl_pct=-3.00"""
        trade = _setup_and_close("long", 200.0, 194.0, 100, 100)
        assert trade.pnl == pytest.approx(-600.00)
        assert trade.pnl_pct == pytest.approx(-3.00)

    def test_short_losing_full_close(self):
        """short 50 shares at $10, CLOSE qty=50 at $11 → pnl=-50.00, pnl_pct=-10.00"""
        trade = _setup_and_close("short", 10.0, 11.0, 50, 50)
        assert trade.pnl == pytest.approx(-50.00)
        assert trade.pnl_pct == pytest.approx(-10.00)

    def test_long_breakeven_full_close(self):
        """long 100 shares at $200, CLOSE qty=100 at $200 → pnl=0.00, pnl_pct=0.00"""
        trade = _setup_and_close("long", 200.0, 200.0, 100, 100)
        assert trade.pnl == pytest.approx(0.00)
        assert trade.pnl_pct == pytest.approx(0.00)

    def test_long_winning_partial_close(self):
        """long 100 shares at $200, CLOSE qty=40 at $206 → pnl=240.00, pnl_pct=3.00"""
        trade = _setup_and_close("long", 200.0, 206.0, 100, 40)
        assert trade.pnl == pytest.approx(240.00)
        assert trade.pnl_pct == pytest.approx(3.00)


# ═══════════════════════════════════════════════════════════════════════
# Hypothesis strategies for property-based tests
# ═══════════════════════════════════════════════════════════════════════

side_strategy = st.sampled_from(["long", "short"])

# Positive floats for prices — avoid subnormals and extremes
price_strategy = st.floats(
    min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)

# Positive integers for quantity
quantity_strategy = st.integers(min_value=1, max_value=500)


# ═══════════════════════════════════════════════════════════════════════
# Property 2a: Long PnL formula preservation
# ═══════════════════════════════════════════════════════════════════════

@given(
    entry_price=price_strategy,
    exit_price=price_strategy,
    quantity=quantity_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_2a_long_pnl_formula(entry_price, exit_price, quantity):
    """
    **Validates: Requirements 3.1**

    Property 2a: For all long positions with positive quantity,
    pnl == (exit_price - entry_price) * quantity and
    pnl_pct == pnl / (entry_price * quantity) * 100
    """
    pos_quantity = quantity + 10  # ensure pos.quantity > close quantity for full close path

    trade = _setup_and_close("long", entry_price, exit_price, pos_quantity, quantity)

    expected_pnl = round((exit_price - entry_price) * quantity, 2)
    expected_pnl_pct = round(
        (exit_price - entry_price) * quantity / (entry_price * quantity) * 100, 2
    )

    assert trade.pnl == pytest.approx(expected_pnl, abs=0.01), (
        f"Long PnL mismatch: entry={entry_price}, exit={exit_price}, qty={quantity}. "
        f"Got pnl={trade.pnl}, expected={expected_pnl}"
    )
    assert trade.pnl_pct == pytest.approx(expected_pnl_pct, abs=0.01), (
        f"Long PnL% mismatch: entry={entry_price}, exit={exit_price}, qty={quantity}. "
        f"Got pnl_pct={trade.pnl_pct}, expected={expected_pnl_pct}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Property 2b: Short PnL formula preservation
# ═══════════════════════════════════════════════════════════════════════

@given(
    entry_price=price_strategy,
    exit_price=price_strategy,
    quantity=quantity_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_2b_short_pnl_formula(entry_price, exit_price, quantity):
    """
    **Validates: Requirements 3.1**

    Property 2b: For all short positions with positive quantity,
    pnl == (entry_price - exit_price) * quantity and
    pnl_pct == pnl / (entry_price * quantity) * 100
    """
    pos_quantity = quantity + 10

    trade = _setup_and_close("short", entry_price, exit_price, pos_quantity, quantity)

    expected_pnl = round((entry_price - exit_price) * quantity, 2)
    expected_pnl_pct = round(
        (entry_price - exit_price) * quantity / (entry_price * quantity) * 100, 2
    )

    assert trade.pnl == pytest.approx(expected_pnl, abs=0.01), (
        f"Short PnL mismatch: entry={entry_price}, exit={exit_price}, qty={quantity}. "
        f"Got pnl={trade.pnl}, expected={expected_pnl}"
    )
    assert trade.pnl_pct == pytest.approx(expected_pnl_pct, abs=0.01), (
        f"Short PnL% mismatch: entry={entry_price}, exit={exit_price}, qty={quantity}. "
        f"Got pnl_pct={trade.pnl_pct}, expected={expected_pnl_pct}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Property 2c: Breakeven preservation
# ═══════════════════════════════════════════════════════════════════════

@given(
    side=side_strategy,
    entry_price=price_strategy,
    quantity=quantity_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_2c_breakeven_pnl_zero(side, entry_price, quantity):
    """
    **Validates: Requirements 3.4**

    Property 2c: For all positions where exit_price == entry_price
    with positive quantity, pnl == 0 and pnl_pct == 0.
    """
    pos_quantity = quantity + 10

    trade = _setup_and_close(side, entry_price, entry_price, pos_quantity, quantity)

    assert trade.pnl == pytest.approx(0.0, abs=0.01), (
        f"Breakeven PnL should be 0: side={side}, price={entry_price}, qty={quantity}. "
        f"Got pnl={trade.pnl}"
    )
    assert trade.pnl_pct == pytest.approx(0.0, abs=0.01), (
        f"Breakeven PnL% should be 0: side={side}, price={entry_price}, qty={quantity}. "
        f"Got pnl_pct={trade.pnl_pct}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Property 2d: Partial close PnL preservation
# ═══════════════════════════════════════════════════════════════════════

@given(
    side=side_strategy,
    entry_price=price_strategy,
    exit_price=price_strategy,
    pos_quantity=st.integers(min_value=2, max_value=500),
)
@settings(max_examples=100, deadline=None)
def test_property_2d_partial_close_pnl(side, entry_price, exit_price, pos_quantity):
    """
    **Validates: Requirements 3.5**

    Property 2d: For all partial closes (0 < quantity < pos.quantity)
    with positive quantity, PnL is calculated on the partial quantity correctly.
    """
    # close_quantity must be strictly less than pos_quantity
    assume(pos_quantity >= 2)
    close_quantity = pos_quantity // 2  # always < pos_quantity and > 0

    trade = _setup_and_close(side, entry_price, exit_price, pos_quantity, close_quantity)

    # Compute expected values the same way the code does:
    # raw pnl first, then pnl_pct from raw pnl, then round each independently
    if side == "long":
        raw_pnl = (exit_price - entry_price) * close_quantity
    else:
        raw_pnl = (entry_price - exit_price) * close_quantity
    raw_pnl_pct = raw_pnl / (entry_price * close_quantity) * 100
    expected_pnl = round(raw_pnl, 2)
    expected_pnl_pct = round(raw_pnl_pct, 2)

    assert trade.pnl == pytest.approx(expected_pnl, abs=0.01), (
        f"Partial close PnL mismatch: side={side}, entry={entry_price}, "
        f"exit={exit_price}, pos_qty={pos_quantity}, close_qty={close_quantity}. "
        f"Got pnl={trade.pnl}, expected={expected_pnl}"
    )
    assert trade.pnl_pct == pytest.approx(expected_pnl_pct, abs=0.01), (
        f"Partial close PnL% mismatch: side={side}, entry={entry_price}, "
        f"exit={exit_price}, pos_qty={pos_quantity}, close_qty={close_quantity}. "
        f"Got pnl_pct={trade.pnl_pct}, expected={expected_pnl_pct}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Property 2e: Sign consistency for positive quantities
# ═══════════════════════════════════════════════════════════════════════

@given(
    side=side_strategy,
    entry_price=price_strategy,
    exit_price=price_strategy,
    quantity=quantity_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_2e_sign_consistency_positive_qty(side, entry_price, exit_price, quantity):
    """
    **Validates: Requirements 3.1, 3.2, 3.3**

    Property 2e: For all positive-quantity closes,
    sign(pnl) == sign(pnl_pct) or pnl == 0 (signs are always consistent).
    """
    pos_quantity = quantity + 10

    trade = _setup_and_close(side, entry_price, exit_price, pos_quantity, quantity)

    pnl = trade.pnl
    pnl_pct = trade.pnl_pct

    assert pnl == 0 or _sign(pnl) == _sign(pnl_pct), (
        f"Sign inconsistency for positive qty: side={side}, "
        f"entry={entry_price}, exit={exit_price}, qty={quantity}. "
        f"pnl={pnl} (sign={_sign(pnl)}), pnl_pct={pnl_pct} (sign={_sign(pnl_pct)})"
    )
