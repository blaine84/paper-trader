from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agents.profit_manager import _partial_close
from db.schema import Base, Balance, Position, ReviewQueue, Trade, TradeEvent


def _session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


def test_partial_profit_reduces_trade_lot_quantity():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = _session(engine)
    db.add(Balance(profile="moderate", cash=100_000))
    db.add(Position(profile="moderate", symbol="META", side="long", quantity=20, avg_cost=100))
    trade = Trade(
        profile="moderate",
        symbol="META",
        direction="LONG",
        quantity=20,
        entry_price=100,
        status="open",
    )
    db.add(trade)
    db.commit()
    trade_id = trade.id
    db.close()

    _partial_close(engine, trade, 0.25, 110, "take some off")

    db = _session(engine)
    updated_trade = db.query(Trade).filter_by(id=trade_id).one()
    updated_position = db.query(Position).filter_by(symbol="META", profile="moderate").one()
    event = db.query(TradeEvent).filter_by(trade_id=trade_id, event_type="partial_profit").one()

    assert updated_trade.status == "open"
    assert updated_trade.quantity == 15
    assert updated_position.quantity == 15
    assert '"trade_remaining_qty": 15' in event.payload_json
    db.close()


def test_partial_profit_closes_trade_when_lot_is_depleted():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = _session(engine)
    db.add(Balance(profile="moderate", cash=100_000))
    db.add(Position(profile="moderate", symbol="META", side="long", quantity=4, avg_cost=100))
    trade = Trade(
        profile="moderate",
        symbol="META",
        direction="LONG",
        quantity=4,
        entry_price=100,
        status="open",
    )
    db.add(trade)
    db.commit()
    trade_id = trade.id
    db.close()

    _partial_close(engine, trade, 1.0, 112, "final partial")

    db = _session(engine)
    updated_trade = db.query(Trade).filter_by(id=trade_id).one()
    remaining_position = db.query(Position).filter_by(symbol="META", profile="moderate").first()
    review = db.query(ReviewQueue).filter_by(trade_id=trade_id).one()

    assert updated_trade.status == "closed"
    assert updated_trade.quantity == 0
    assert updated_trade.exit_price == 112
    assert updated_trade.pnl == 48
    assert remaining_position is None
    assert review.status == "pending"
    db.close()
