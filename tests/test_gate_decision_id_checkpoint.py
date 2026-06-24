"""Checkpoint test: verify gate_decision_id is present in all recovery-related results.

This test validates that the _evaluate_rolling_underperformance helper
generates a valid UUID4 gate_decision_id for all profile branches.
"""

import uuid
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


def _seed_rolling_underperformance(db, setup_type="test_setup"):
    """Seed cases where all-time WR >= floor but rolling WR < floor.

    Cases are ordered most-recent first. We need:
    - All-time WR >= all profile floors (>= 0.35 for conservative)
    - Rolling WR (last 5) < all profile floors (< 0.25 for aggressive)
    - Most recent 3 cases must NOT all be losses (avoid consecutive_losses)

    Solution: 12 cases total.
    Most recent 5: [failure, success, failure, failure, failure] = 1/5 = 20% rolling WR
      (The 2nd most recent is success, breaking consecutive-loss streak at position 0,2,3,4)
    Older 7: all success = 7 wins
    All-time: 8 wins / 12 = 66.7% (above all floors)
    Rolling: 1/5 = 20% (below aggressive 25%, moderate 35%, conservative 35%)
    Consecutive losses from most recent: only 1 (position 0), then success at position 1.
    """
    # Most recent first
    outcomes = ["failure", "success", "failure", "failure", "failure"] + ["success"] * 7
    now = datetime.utcnow()
    for idx, outcome in enumerate(outcomes):
        db.add(Case(
            symbol="TEST",
            date=(now - timedelta(days=idx)).strftime("%Y-%m-%d"),
            setup_type=setup_type,
            outcome=outcome,
            pnl_pct=1.0 if outcome == "success" else -0.5,
            lesson="checkpoint test",
            created_at=now - timedelta(days=idx),
        ))
    db.commit()


def test_aggressive_recovery_probe_has_gate_decision_id():
    engine, db = _engine_and_session()
    _seed_rolling_underperformance(db)

    result = evaluate_setup_quality(
        engine, db, "test_setup", profile="aggressive", symbol="TEST"
    )

    assert result["decision"] == "reduce_size"
    assert result["reason_type"] == "rolling_underperformance_recovery_probe"
    assert "gate_decision_id" in result
    # Validate it's a proper UUID4 string
    parsed = uuid.UUID(result["gate_decision_id"])
    assert parsed.version == 4


def test_moderate_recovery_probe_has_gate_decision_id():
    engine, db = _engine_and_session()
    _seed_rolling_underperformance(db)

    result = evaluate_setup_quality(
        engine, db, "test_setup", profile="moderate", symbol="TEST",
        confidence_score=9.0,
    )

    assert result["decision"] == "reduce_size"
    assert result["reason_type"] == "rolling_underperformance_recovery_probe"
    assert "gate_decision_id" in result
    parsed = uuid.UUID(result["gate_decision_id"])
    assert parsed.version == 4


def test_moderate_warning_has_gate_decision_id():
    engine, db = _engine_and_session()
    _seed_rolling_underperformance(db)

    result = evaluate_setup_quality(
        engine, db, "test_setup", profile="moderate", symbol="TEST",
        confidence_score=5.0,
    )

    assert result["decision"] == "warn"
    assert result["reason_type"] == "rolling_underperformance_confirmation_required"
    assert "gate_decision_id" in result
    parsed = uuid.UUID(result["gate_decision_id"])
    assert parsed.version == 4


def test_conservative_warning_has_gate_decision_id():
    engine, db = _engine_and_session()
    _seed_rolling_underperformance(db)

    result = evaluate_setup_quality(
        engine, db, "test_setup", profile="conservative", symbol="TEST"
    )

    assert result["decision"] == "warn"
    assert result["reason_type"] == "rolling_underperformance_conservative_reject"
    assert "gate_decision_id" in result
    parsed = uuid.UUID(result["gate_decision_id"])
    assert parsed.version == 4


def test_unknown_profile_warning_has_gate_decision_id():
    engine, db = _engine_and_session()
    _seed_rolling_underperformance(db)

    result = evaluate_setup_quality(
        engine, db, "test_setup", profile=None, symbol="TEST"
    )

    assert result["decision"] == "warn"
    assert result["reason_type"] == "rolling_underperformance"
    assert "gate_decision_id" in result
    parsed = uuid.UUID(result["gate_decision_id"])
    assert parsed.version == 4


def test_non_recovery_results_do_not_have_gate_decision_id():
    """Non-recovery decisions (allow, downgrade, etc.) should NOT have gate_decision_id."""
    engine, db = _engine_and_session()
    # Seed insufficient data (< MIN_CASES_FOR_BLOCK)
    now = datetime.utcnow()
    db.add(Case(
        symbol="TEST",
        date=now.strftime("%Y-%m-%d"),
        setup_type="rare_setup",
        outcome="success",
        pnl_pct=1.0,
        lesson="test",
        created_at=now,
    ))
    db.commit()

    result = evaluate_setup_quality(
        engine, db, "rare_setup", profile="aggressive", symbol="TEST"
    )

    assert result["decision"] == "allow"
    assert result["reason_type"] == "insufficient_data"
    assert "gate_decision_id" not in result
