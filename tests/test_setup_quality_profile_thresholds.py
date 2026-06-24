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
            lesson="test case",
            created_at=now - timedelta(days=idx),
        ))
    db.commit()


def _add_cases(db, outcomes, setup_type="news_breakout", pnls=None):
    now = datetime.utcnow()
    pnls = pnls or [1.0 if outcome == "success" else -0.5 for outcome in outcomes]
    for idx, (outcome, pnl_pct) in enumerate(zip(outcomes, pnls)):
        db.add(Case(
            symbol="AMD",
            date=(now - timedelta(days=idx)).strftime("%Y-%m-%d"),
            setup_type=setup_type,
            outcome=outcome,
            pnl_pct=pnl_pct,
            lesson="test case",
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


def test_vwap_reclaim_at_thirty_percent_is_allowed_for_moderate():
    engine, db = _engine_and_session()
    _add_cases(
        db,
        [
            "success",
            "failure",
            "success",
            "failure",
            "failure",
            "success",
            "failure",
            "failure",
            "failure",
            "failure",
        ],
        setup_type="vwap_reclaim",
    )

    moderate = evaluate_setup_quality(
        engine, db, "vwap_reclaim", profile="moderate", symbol="AMD"
    )
    conservative = evaluate_setup_quality(
        engine, db, "vwap_reclaim", profile="conservative", symbol="AMD"
    )

    assert moderate["threshold"] == 0.30
    assert moderate["decision"] == "downgrade"
    assert moderate["reason_type"] == "weak_but_allowed"

    assert conservative["threshold"] == 0.35
    assert conservative["decision"] == "reject"
    assert conservative["reason_type"] == "historical_underperformance"


def test_profitable_partial_breaks_consecutive_loss_pause_without_counting_as_win():
    engine, db = _engine_and_session()
    _add_cases(
        db,
        ["partial", "failure", "failure", "success", "success"],
        pnls=[0.27, -3.60, -3.60, 1.0, 1.0],
    )

    result = evaluate_setup_quality(
        engine, db, "news_breakout", profile="aggressive", symbol="AMD"
    )

    assert result["win_rate"] == 0.40
    assert result["reason_type"] != "consecutive_losses"
    assert result["decision"] == "downgrade"


def test_non_profitable_partial_still_counts_toward_consecutive_loss_pause():
    engine, db = _engine_and_session()
    _add_cases(
        db,
        ["partial", "failure", "failure", "success", "success"],
        pnls=[0.0, -3.60, -3.60, 1.0, 1.0],
    )

    result = evaluate_setup_quality(
        engine, db, "news_breakout", profile="aggressive", symbol="AMD"
    )

    assert result["decision"] == "warn"
    assert result["reason_type"] == "consecutive_losses"


def test_consecutive_loss_pause_warns_without_rejecting():
    engine, db = _engine_and_session()
    _add_cases(
        db,
        ["failure", "failure", "failure", "success", "success"],
        setup_type="technical_breakout",
    )

    result = evaluate_setup_quality(
        engine, db, "technical_breakout", profile="moderate", symbol="XLK"
    )

    assert result["decision"] == "warn"
    assert result["canonical_decision"] == "warn"
    assert result["reason_type"] == "consecutive_losses"
    assert "warning only" in result["reason"]


def test_gap_and_go_is_exempt_from_consecutive_loss_pause():
    engine, db = _engine_and_session()
    _add_cases(
        db,
        ["failure", "failure", "failure", "success", "success"],
        setup_type="gap_and_go",
    )

    result = evaluate_setup_quality(
        engine, db, "gap_and_go", profile="moderate", symbol="AMD"
    )

    assert result["win_rate"] == 0.40
    assert result["decision"] != "reject"
    assert result["reason_type"] != "consecutive_losses"


def test_recovery_override_can_fire_with_the_configured_rolling_window():
    engine, db = _engine_and_session()
    _add_cases(
        db,
        [
            "success", "failure", "success", "failure", "success",
            "failure", "failure", "failure", "failure", "failure",
        ],
        pnls=[1.0, -0.2, 1.0, -0.2, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    )

    result = evaluate_setup_quality(
        engine, db, "news_breakout", profile="conservative", symbol="AMD"
    )

    assert result["win_rate"] == 0.30
    assert result["rolling_sample_size"] == 5
    assert result["decision"] == "allow"
    assert result["reason_type"] == "recovery_override"
