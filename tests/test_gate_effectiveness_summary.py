"""Unit tests for get_gate_effectiveness_summary()."""

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from db.schema import init_db
from utils.shadow_ledger import ensure_shadow_ledger_schema
from utils.shadow_outcomes import get_gate_effectiveness_summary


def _setup_db(tmp_path):
    db_path = tmp_path / "paper.db"
    engine = init_db(str(db_path))
    ensure_shadow_ledger_schema(engine)
    return engine


def _insert_candidate(conn, candidate_id, blocked_by="setup_quality_gate", created_at=None):
    """Insert a minimal blocked_trade_candidates row."""
    if created_at is None:
        created_at = datetime.now(timezone.utc) - timedelta(days=5)
    conn.execute(
        text("""
            INSERT INTO blocked_trade_candidates (id, created_at, symbol, action, direction, profile,
                entry_price, stop_price, target_price, quantity, blocked_by, block_reason)
            VALUES (:id, :created_at, 'AAPL', 'BUY', 'long', 'moderate',
                150.0, 148.0, 155.0, 10, :blocked_by, 'test reason')
        """),
        {
            "id": candidate_id,
            "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "blocked_by": blocked_by,
        },
    )


def _insert_outcome(conn, candidate_id, eval_window, gate_verdict, outcome_label, pnl_pct, notes_json=None):
    """Insert a blocked_trade_candidate_outcomes row."""
    if notes_json is None:
        notes_json = json.dumps({"candidate_source_classification": "gate_rejection"})
    conn.execute(
        text("""
            INSERT INTO blocked_trade_candidate_outcomes
                (blocked_candidate_id, eval_window, evaluated_at, eval_price,
                 pnl_pct, gate_verdict, outcome_label, notes_json)
            VALUES (:cid, :window, datetime('now'), 150.0,
                    :pnl, :verdict, :label, :notes)
        """),
        {
            "cid": candidate_id,
            "window": eval_window,
            "pnl": pnl_pct,
            "verdict": gate_verdict,
            "label": outcome_label,
            "notes": notes_json,
        },
    )


def test_basic_counts(tmp_path):
    """Test that blocked_winners, saved_us, and neutral are counted correctly."""
    engine = _setup_db(tmp_path)
    with engine.begin() as conn:
        _insert_candidate(conn, 1)
        _insert_candidate(conn, 2)
        _insert_candidate(conn, 3)

        _insert_outcome(conn, 1, "60m", "blocked_winner", "blocked_winner", 1.5)
        _insert_outcome(conn, 2, "60m", "saved_us", "would_hit_stop", -0.8)
        _insert_outcome(conn, 3, "60m", "neutral", "flat_so_far", 0.05)

    result = get_gate_effectiveness_summary(engine)

    assert result["blocked_winners"] == 1
    assert result["saved_us"] == 1
    assert result["neutral"] == 1
    assert result["unscorable_excluded"] == 0
    assert result["malformed_excluded"] == 0
    assert result["period_days"] == 30
    assert result["gate_name"] == "all"


def test_excludes_unscorable(tmp_path):
    """Test that unscorable 60m outcomes are excluded from counts."""
    engine = _setup_db(tmp_path)
    with engine.begin() as conn:
        _insert_candidate(conn, 1)
        _insert_candidate(conn, 2)

        _insert_outcome(conn, 1, "60m", "blocked_winner", "blocked_winner", 1.5)
        _insert_outcome(conn, 2, "60m", "unscorable", "unscorable_missing_entry_price", None)

    result = get_gate_effectiveness_summary(engine)

    assert result["blocked_winners"] == 1
    assert result["saved_us"] == 0
    assert result["neutral"] == 0
    assert result["unscorable_excluded"] == 1


def test_excludes_malformed_decision(tmp_path):
    """Test that candidates classified as malformed_decision are excluded."""
    engine = _setup_db(tmp_path)
    malformed_notes = json.dumps({"candidate_source_classification": "malformed_decision"})
    with engine.begin() as conn:
        _insert_candidate(conn, 1)
        _insert_candidate(conn, 2, blocked_by="pm_normalizer")

        _insert_outcome(conn, 1, "60m", "blocked_winner", "blocked_winner", 1.5)
        _insert_outcome(conn, 2, "60m", "blocked_winner", "blocked_winner", 2.0,
                        notes_json=malformed_notes)

    result = get_gate_effectiveness_summary(engine)

    assert result["blocked_winners"] == 1
    assert result["malformed_excluded"] == 1


def test_only_uses_60m_window(tmp_path):
    """Test that only 60m outcomes are counted, not 15m or 30m."""
    engine = _setup_db(tmp_path)
    with engine.begin() as conn:
        _insert_candidate(conn, 1)

        # 15m and 30m outcomes should be ignored
        _insert_outcome(conn, 1, "15m", "blocked_winner", "blocked_winner", 1.0)
        _insert_outcome(conn, 1, "30m", "blocked_winner", "blocked_winner", 1.2)
        _insert_outcome(conn, 1, "60m", "saved_us", "would_hit_stop", -0.5)

    result = get_gate_effectiveness_summary(engine)

    assert result["blocked_winners"] == 0
    assert result["saved_us"] == 1


def test_gate_name_filter(tmp_path):
    """Test that gate_name filters to specific gate's candidates."""
    engine = _setup_db(tmp_path)
    with engine.begin() as conn:
        _insert_candidate(conn, 1, blocked_by="setup_quality_gate")
        _insert_candidate(conn, 2, blocked_by="risk_geometry_gate")

        _insert_outcome(conn, 1, "60m", "blocked_winner", "blocked_winner", 1.5)
        _insert_outcome(conn, 2, "60m", "saved_us", "would_hit_stop", -0.8)

    result = get_gate_effectiveness_summary(engine, gate_name="setup_quality_gate")

    assert result["blocked_winners"] == 1
    assert result["saved_us"] == 0
    assert result["gate_name"] == "setup_quality_gate"


def test_avg_pnl_pct(tmp_path):
    """Test that avg_pnl_pct is computed correctly from included outcomes."""
    engine = _setup_db(tmp_path)
    with engine.begin() as conn:
        _insert_candidate(conn, 1)
        _insert_candidate(conn, 2)

        _insert_outcome(conn, 1, "60m", "blocked_winner", "blocked_winner", 2.0)
        _insert_outcome(conn, 2, "60m", "saved_us", "would_hit_stop", -1.0)

    result = get_gate_effectiveness_summary(engine)

    assert result["avg_pnl_pct"] == 0.5  # (2.0 + -1.0) / 2


def test_lookback_days_filters_old(tmp_path):
    """Test that candidates outside lookback_days are excluded."""
    engine = _setup_db(tmp_path)
    with engine.begin() as conn:
        # Recent candidate (within 7 days)
        _insert_candidate(conn, 1, created_at=datetime.now(timezone.utc) - timedelta(days=3))
        # Old candidate (outside 7 days)
        _insert_candidate(conn, 2, created_at=datetime.now(timezone.utc) - timedelta(days=10))

        _insert_outcome(conn, 1, "60m", "blocked_winner", "blocked_winner", 1.5)
        _insert_outcome(conn, 2, "60m", "blocked_winner", "blocked_winner", 2.0)

    result = get_gate_effectiveness_summary(engine, lookback_days=7)

    assert result["blocked_winners"] == 1
    assert result["period_days"] == 7


def test_empty_results(tmp_path):
    """Test that an empty database returns zero counts."""
    engine = _setup_db(tmp_path)

    result = get_gate_effectiveness_summary(engine)

    assert result["blocked_winners"] == 0
    assert result["saved_us"] == 0
    assert result["neutral"] == 0
    assert result["unscorable_excluded"] == 0
    assert result["malformed_excluded"] == 0
    assert result["avg_pnl_pct"] == 0.0
