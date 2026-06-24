import json

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from db.schema import init_db
from utils.candidate_pipeline import (
    ResolvedOrder,
    _commit_candidate_pipeline_session,
    _record_pipeline_shadow_block,
)
from utils.position_sizer import SizingResult
from utils.shadow_ledger import ensure_shadow_ledger_schema


def test_candidate_pipeline_gate_reject_is_mirrored_to_shadow_ledger(tmp_path):
    engine = init_db(str(tmp_path / "paper.db"))
    ensure_shadow_ledger_schema(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    resolved_order = ResolvedOrder(
        candidate_id="cand-123",
        execution_key="exec-123",
        symbol="NVDA",
        action="BUY",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        setup_type="momentum_fade",
        risk_reward=2.0,
        source_signal={"symbol": "NVDA", "confidence_score": 7.2},
        profile_id="moderate",
        geometry_name="base_breakout",
        risk_multiplier=1.0,
        pm_rationale="Accepted by PM",
    )
    sizing_result = SizingResult(
        quantity=10,
        dollar_risk=20.0,
        position_value=1000.0,
        sizing_method="standard",
        applied_multiplier=1.0,
    )
    gate_notes = [
        {
            "gate": "setup_quality_gate",
            "decision": "reject",
            "reason_type": "historical_underperformance",
            "reason": "Setup WR below threshold",
        }
    ]

    _record_pipeline_shadow_block(
        db,
        resolved_order,
        sizing_result,
        outcome="gate_rejected",
        block_reason="Setup WR below threshold",
        blocked_by="setup_quality_gate",
        reason_code="historical_underperformance",
        gate_notes=gate_notes,
    )
    _commit_candidate_pipeline_session(db, resolved_order.candidate_id, "test")

    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT symbol, action, profile, blocked_by, block_reason, reason_code,
                       quantity, decision_snapshot_json, signal_snapshot_json,
                       gate_notes_json, source
                FROM blocked_trade_candidates
                """
            )
        ).mappings().one()

    assert row["symbol"] == "NVDA"
    assert row["action"] == "BUY"
    assert row["profile"] == "moderate"
    assert row["blocked_by"] == "setup_quality_gate"
    assert row["block_reason"] == "Setup WR below threshold"
    assert row["reason_code"] == "historical_underperformance"
    assert row["quantity"] == 10
    assert row["source"] == "candidate_id_pipeline"

    decision_snapshot = json.loads(row["decision_snapshot_json"])
    assert decision_snapshot["candidate_id"] == "cand-123"
    assert decision_snapshot["pipeline_outcome"] == "gate_rejected"
    assert decision_snapshot["sizing"]["quantity"] == 10

    signal_snapshot = json.loads(row["signal_snapshot_json"])
    assert signal_snapshot["symbol"] == "NVDA"

    stored_gate_notes = json.loads(row["gate_notes_json"])
    assert stored_gate_notes[0]["reason_type"] == "historical_underperformance"
