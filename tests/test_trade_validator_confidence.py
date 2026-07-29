from sqlalchemy import create_engine

from db.schema import Base, get_session
from models.case import Case
from utils.trade_validator import adjust_confidence


def _make_engine():
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _seed_cases(engine, setup_type, outcomes, market_regime="risk_off"):
    db = get_session(engine)
    for index, outcome in enumerate(outcomes):
        db.add(Case(
            symbol="AMD",
            date=f"2026-07-{index + 1:02d}",
            setup_type=setup_type,
            market_regime=market_regime,
            outcome=outcome,
            lesson="test case",
        ))
    db.commit()
    db.close()


def test_low_case_library_winrate_warns_without_blocking():
    engine = _make_engine()
    _seed_cases(
        engine,
        "momentum_fade",
        ["success"] * 2 + ["failure"] * 8,
        market_regime="risk_off",
    )

    result = adjust_confidence(engine, "momentum_fade", "risk_off")

    assert result["block"] is False
    assert result["modifier"] == 0.4
    assert result["win_rate"] == 0.2
    assert result["total_cases"] == 10
    assert "warning-only" in result["reason"]
    assert "BLOCKED" not in result["reason"]
