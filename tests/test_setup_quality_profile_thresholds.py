from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base
from models.case import Case
from utils.setup_quality_gate import evaluate_setup_quality


def _engine_and_session():
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session()


def _seed_cases(db, setup_type="momentum_fade"):
    # 10 cases, 2 wins = 20% all-time WR. The most recent 5 include 2 wins
    # = 40% rolling WR, matching the real NVDA shape that exposed this gate.
    outcomes = ["success", "failure", "success", "failure", "failure", "failure", "failure", "failure", "failure", "failure"]
    now = datetime.utcnow()
    for idx, outcome in enumerate(outcomes):
        db.add(Case(
            symbol="NVDA",
            date=(now - timedelta(days=idx)).strftime("%Y-%m-%d"),
            setup_type=setup_type,
            outcome=outcome,
            pnl_pct=1.0 if outcome == "success" else -0.5,
            created_at=now - timedelta(days=idx),
        ))
    db.commit()


def test_momentum_fade_setup_quality_is_profile_aware():
    engine, db = _engine_and_session()
    _seed_cases(db)

    moderate = evaluate_setup_quality(engine, db, "momentum_fade", profile="moderate", symbol="NVDA")
    aggressive = evaluate_setup_quality(engine, db, "momentum_fade", profile="aggressive", symbol="NVDA")

    assert moderate["threshold"] == 0.30
    assert moderate["decision"] == "reject"
    assert moderate["reason_type"] == "historical_underperformance"

    assert aggressive["threshold"] == 0.20
    assert aggressive["decision"] == "downgrade"
    assert aggressive["reason_type"] == "weak_but_allowed"


def test_conservative_keeps_original_momentum_fade_floor():
    engine, db = _engine_and_session()
    _seed_cases(db)

    conservative = evaluate_setup_quality(engine, db, "momentum_fade", profile="conservative", symbol="NVDA")

    assert conservative["threshold"] == 0.35
    assert conservative["decision"] == "reject"
