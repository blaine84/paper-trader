import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine

from db.schema import (
    Base,
    AnalystFeedbackQueue,
    AnalystMitigation,
    get_session,
)
from feedback_loop.analyst_feedback import (
    apply_signal_mitigation,
    evaluate_auto_mitigation,
    maybe_reset_weekly_mitigations,
    queue_reviewer_flags,
)


@pytest.fixture
def engine():
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    return eng


def test_queue_reviewer_flags_derives_and_dedupes(engine):
    cases = [{
        "trade_id": 7,
        "symbol": "NVDA",
        "date": "2026-04-29",
        "setup_type": "gap_and_go",
        "selection_score": 4.0,
        "signal_strength": "weak",
        "signal_confidence": "low",
        "holding_minutes": 90,
        "outcome": "failure",
    }]

    first = queue_reviewer_flags(engine, cases)
    second = queue_reviewer_flags(engine, cases)

    db = get_session(engine)
    rows = db.query(AnalystFeedbackQueue).all()
    db.close()

    flag_types = {row.flag_type for row in rows}
    assert len(first) == 4
    assert second == []
    assert len(rows) == 4
    assert "selection_score_below_threshold" in flag_types
    assert "signal_strength_below_threshold" in flag_types
    assert "signal_confidence_below_threshold" in flag_types
    assert "gap_and_go_hold_time_violated" in flag_types


def test_unsupported_rejects_trigger_setup_mitigation(engine):
    now = datetime.utcnow()
    db = get_session(engine)
    db.add_all([
        AnalystFeedbackQueue(
            symbol="TSLA",
            setup_type="gap_and_go",
            date="2026-04-29",
            flag_type="selection_score_below_threshold",
            severity="high",
            recommendation="Tighten classification",
            reviewer_context=json.dumps({}),
            due_at=now,
            responded_at=now,
            status="responded",
            analyst_response="reject",
            analyst_supporting_data=json.dumps([]),
            no_data_reject=True,
        ),
        AnalystFeedbackQueue(
            symbol="TSLA",
            setup_type="gap_and_go",
            date="2026-04-29",
            flag_type="signal_strength_below_threshold",
            severity="medium",
            recommendation="Require stronger confirmation",
            reviewer_context=json.dumps({}),
            due_at=now,
            responded_at=now,
            status="responded",
            analyst_response="reject",
            analyst_supporting_data=json.dumps([]),
            no_data_reject=True,
        ),
    ])
    db.commit()
    db.close()

    evaluate_auto_mitigation(engine)

    db = get_session(engine)
    mitigation = db.query(AnalystMitigation).filter_by(setup_type="gap_and_go").first()
    db.close()

    assert mitigation is not None
    assert mitigation.active is True
    assert mitigation.level == 1
    assert mitigation.deployment_multiplier == 0.75
    assert mitigation.signal_threshold_bump == 0.5

    signal = {
        "signal": "LONG",
        "strength": "moderate",
        "confidence": "medium",
        "setup_type": "gap_and_go",
        "reasoning": "Base signal.",
    }
    adjusted = apply_signal_mitigation(signal, {
        "gap_and_go": {
            "level": mitigation.level,
            "deployment_multiplier": mitigation.deployment_multiplier,
            "signal_threshold_bump": mitigation.signal_threshold_bump,
        }
    })
    assert adjusted["signal"] == "HOLD"
    assert adjusted["original_signal"] == "LONG"


def test_weekly_reset_clears_mitigation_after_high_acceptance(engine):
    now = datetime.utcnow()
    db = get_session(engine)
    db.add(AnalystMitigation(
        setup_type="vwap_reclaim",
        level=2,
        deployment_multiplier=0.5,
        signal_threshold_bump=1.0,
        active=True,
        applied_at=now - timedelta(days=2),
    ))
    db.add_all([
        AnalystFeedbackQueue(
            symbol="AMD",
            setup_type="vwap_reclaim",
            date="2026-04-29",
            flag_type="selection_score_below_threshold",
            severity="high",
            recommendation="Tighten classification",
            reviewer_context=json.dumps({}),
            due_at=now,
            responded_at=now - timedelta(hours=4),
            status="responded",
            analyst_response="accept",
            analyst_supporting_data=json.dumps([]),
            no_data_reject=False,
        ),
        AnalystFeedbackQueue(
            symbol="AMD",
            setup_type="vwap_reclaim",
            date="2026-04-29",
            flag_type="signal_confidence_below_threshold",
            severity="medium",
            recommendation="Raise confidence bar",
            reviewer_context=json.dumps({}),
            due_at=now,
            responded_at=now - timedelta(hours=2),
            status="responded",
            analyst_response="modify",
            analyst_supporting_data=json.dumps([]),
            no_data_reject=False,
        ),
    ])
    db.commit()
    db.close()

    maybe_reset_weekly_mitigations(engine)

    db = get_session(engine)
    mitigation = db.query(AnalystMitigation).filter_by(setup_type="vwap_reclaim").first()
    db.close()

    assert mitigation.active is False
    assert mitigation.level == 0
    assert mitigation.deployment_multiplier == 1.0
    assert mitigation.signal_threshold_bump == 0.0
