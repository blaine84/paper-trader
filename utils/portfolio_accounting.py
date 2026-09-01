"""Portfolio accounting helpers.

Balances are snapshots written by execution code, not the source of truth.
These helpers derive cash/equity from the trade ledger plus current positions
so a bad balance row cannot permanently distort portfolio state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from db.schema import Position, Trade


@dataclass(frozen=True)
class PortfolioAccountingSnapshot:
    starting_balance: float
    realized_pnl: float
    unrealized_pnl: float
    reserved_capital: float
    cash: float
    total_equity: float
    position_value: float


def _trade_pnl(trade: Trade) -> float:
    if trade.pnl is not None:
        return float(trade.pnl)
    if trade.exit_price is None:
        return 0.0

    qty = float(trade.quantity or 0)
    entry = float(trade.entry_price or 0)
    exit_price = float(trade.exit_price or 0)
    if str(trade.direction or "").upper() == "SHORT":
        return (entry - exit_price) * qty
    return (exit_price - entry) * qty


def compute_portfolio_accounting(
    db,
    profile_id: str,
    starting_balance: float,
    current_prices: Mapping[str, float] | None = None,
) -> PortfolioAccountingSnapshot:
    """Return a reconciled portfolio snapshot for a PM profile."""
    prices = {
        str(k).upper(): float(v)
        for k, v in (current_prices or {}).items()
        if v is not None
    }

    closed_trades = (
        db.query(Trade)
        .filter_by(profile=profile_id, status="closed")
        .all()
    )
    realized_pnl = sum(_trade_pnl(trade) for trade in closed_trades)

    positions = db.query(Position).filter_by(profile=profile_id).all()
    reserved_capital = 0.0
    position_value = 0.0
    unrealized_pnl = 0.0

    for position in positions:
        qty = float(position.quantity or 0)
        avg_cost = float(position.avg_cost or 0)
        reserve = qty * avg_cost
        price = prices.get(str(position.symbol or "").upper(), avg_cost)
        market_value = qty * price

        reserved_capital += reserve
        position_value += market_value
        if str(position.side or "").lower() == "short":
            unrealized_pnl += (avg_cost - price) * qty
        else:
            unrealized_pnl += (price - avg_cost) * qty

    cash = float(starting_balance) + realized_pnl - reserved_capital
    total_equity = float(starting_balance) + realized_pnl + unrealized_pnl

    return PortfolioAccountingSnapshot(
        starting_balance=float(starting_balance),
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        reserved_capital=reserved_capital,
        cash=cash,
        total_equity=total_equity,
        position_value=position_value,
    )
