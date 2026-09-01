from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Balance, Position, Trade
from models.case import Case  # noqa: F401
from utils.portfolio_accounting import compute_portfolio_accounting
from web.app import get_portfolio_summary


def _make_session():
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_accounting_ignores_poisoned_balance_snapshot():
    db = _make_session()
    db.add(Balance(profile="aggressive", cash=83_444.82))
    db.add(
        Trade(
            profile="aggressive",
            symbol="AMD",
            direction="LONG",
            quantity=10,
            entry_price=100.0,
            exit_price=90.0,
            status="closed",
            pnl=-100.0,
            entry_time=datetime.utcnow(),
            exit_time=datetime.utcnow(),
        )
    )
    db.add(
        Position(
            profile="aggressive",
            symbol="NVDA",
            side="long",
            quantity=1,
            avg_cost=220.59,
        )
    )
    db.commit()

    snapshot = compute_portfolio_accounting(db, "aggressive", 100_000.0)

    assert round(snapshot.cash, 2) == 99_679.41
    assert round(snapshot.total_equity, 2) == 99_900.00
    assert round(snapshot.realized_pnl, 2) == -100.00


def test_dashboard_summary_uses_reconciled_accounting():
    db = _make_session()
    db.add(Balance(profile="aggressive", cash=83_444.82))
    db.add(
        Trade(
            profile="aggressive",
            symbol="AMD",
            direction="LONG",
            quantity=10,
            entry_price=100.0,
            exit_price=90.0,
            status="closed",
            pnl=-100.0,
            entry_time=datetime.utcnow(),
            exit_time=datetime.utcnow(),
        )
    )
    db.add(
        Position(
            profile="aggressive",
            symbol="NVDA",
            side="long",
            quantity=1,
            avg_cost=220.59,
        )
    )
    db.commit()

    summary = get_portfolio_summary(db)["aggressive"]

    assert summary["cash"] == 99_679.41
    assert summary["equity"] == 99_900.00
    assert summary["pnl"] == -100.00
