"""Unit tests for AgentMemory analyst signal record writing (Task 6.2).

Verifies that _write_signal_memory() and run_funnel_analysis() produce
standard AgentMemory records (agent="analyst", key="signal") that are:
1. Created for promoted candidates
2. JSON schema-compatible with existing PM/Analyst code paths
3. Queryable by the same filter pattern used across the codebase

Requirements: 5.6
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine

from db.schema import Base, FunnelCandidate, AgentMemory, get_session
from utils.funnel_analyst import (
    AnalysisDecision,
    run_funnel_analysis,
    _write_signal_memory,
    _normalize_direction,
    _normalize_strength,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _make_funnel_candidate(
    engine,
    symbol: str = "AAPL",
    scout_score: float = 85.0,
    direction_bias: str = "bullish",
    stage_status: str = "awaiting_analysis",
) -> FunnelCandidate:
    """Create and persist a FunnelCandidate in the DB for testing."""
    ny_tz = ZoneInfo("America/New_York")
    today_ny = datetime.now(timezone.utc).astimezone(ny_tz).date()

    scout_decision = {
        "agent": "scout",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": "promoted",
        "reasoning": "Strong price action with catalyst",
        "evidence": {"scout_score": scout_score, "direction_bias": direction_bias},
        "next_stage": "awaiting_research",
    }
    researcher_decision = {
        "agent": "researcher",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": "promoted",
        "reasoning": "Fresh specific catalyst confirmed",
        "evidence": {"catalyst_validation": {"freshness": "fresh", "specific": True}},
        "next_stage": "awaiting_analysis",
    }

    candidate = FunnelCandidate(
        candidate_id=str(uuid.uuid4()),
        date=today_ny,
        symbol=symbol,
        discovered_at=datetime.now(timezone.utc),
        source_run="premarket",
        selection_mode="deterministic_fallback",
        scout_rank=1,
        scout_score=scout_score,
        direction_bias=direction_bias,
        catalyst_evidence=json.dumps({
            "news_headlines": [{"headline": f"{symbol} beats earnings"}],
            "news_freshness_minutes": 30.0,
        }),
        selection_reason="Strong earnings beat",
        primary_risk="Valuation stretched",
        sector_context=json.dumps({"sector": "tech", "sector_name": "Technology"}),
        preliminary_setup_type="gap_and_go",
        stage_status=stage_status,
        stage_decisions=json.dumps([scout_decision, researcher_decision]),
        expired=False,
    )

    session = get_session(engine)
    session.add(candidate)
    session.commit()
    session.refresh(candidate)
    session.expunge(candidate)
    session.close()
    return candidate


def _mock_analysis_response(
    setup_type: str = "gap_and_go",
    signal_direction: str = "LONG",
    signal_strength: str = "strong",
    confidence: str = "high",
    decision: str = "promoted",
    reasoning: str = "Clear gap-and-go setup with strong volume",
    key_levels: dict | None = None,
    invalidation: str = "Close below VWAP",
    volume_requirements: str = "Volume > 2x average in first 15 min",
    catalyst_dependence: str = "medium",
    catalyst_freshness: str = "fresh",
) -> dict:
    """Build a mock LLM analysis response."""
    if key_levels is None:
        key_levels = {
            "support": 175.50,
            "resistance": 182.00,
            "vwap": 178.25,
            "entry_zone": "178.50-179.00",
            "stop_level": 175.00,
            "target_1": 181.00,
            "target_2": 184.50,
        }
    return {
        "setup_type": setup_type,
        "signal_direction": signal_direction,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "key_levels": key_levels,
        "invalidation": invalidation,
        "volume_requirements": volume_requirements,
        "catalyst_dependence": catalyst_dependence,
        "catalyst_freshness": catalyst_freshness,
        "reasoning": reasoning,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# _write_signal_memory unit tests
# ---------------------------------------------------------------------------


class TestWriteSignalMemory:
    """Direct unit tests for the _write_signal_memory helper function."""

    def test_creates_agent_memory_record(self, in_memory_engine):
        """_write_signal_memory creates an AgentMemory row with correct filters."""
        analysis = _mock_analysis_response()

        _write_signal_memory(
            in_memory_engine, "AAPL", analysis, "gap_and_go"
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        assert mem is not None
        assert mem.agent == "analyst"
        assert mem.key == "signal"
        assert mem.symbol == "AAPL"
        session.close()

    def test_signal_record_json_schema_has_required_fields(self, in_memory_engine):
        """Signal record value JSON contains all fields expected by PM code path."""
        analysis = _mock_analysis_response()

        _write_signal_memory(
            in_memory_engine, "TSLA", analysis, "breakout"
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="TSLA")
            .first()
        )
        data = json.loads(mem.value)

        # These fields are required by the existing PM code path
        # (see normalize_analyst_signal_shape and PM signal reads)
        assert "symbol" in data
        assert "signal" in data  # direction: LONG/SHORT/HOLD
        assert "strength" in data  # weak/moderate/strong
        assert "confidence" in data  # low/medium/high
        assert "setup_type" in data
        assert "reasoning" in data
        assert "key_levels" in data
        assert "invalidation" in data
        session.close()

    def test_signal_direction_values_match_existing_schema(self, in_memory_engine):
        """Signal direction is LONG/SHORT/HOLD — same as existing analyst."""
        analysis = _mock_analysis_response(signal_direction="LONG")

        _write_signal_memory(
            in_memory_engine, "AAPL", analysis, "gap_and_go"
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        data = json.loads(mem.value)
        assert data["signal"] in ("LONG", "SHORT", "HOLD")
        assert data["signal"] == "LONG"
        session.close()

    def test_strength_values_match_existing_schema(self, in_memory_engine):
        """Strength is weak/moderate/strong — same as existing analyst."""
        analysis = _mock_analysis_response(signal_strength="moderate")

        _write_signal_memory(
            in_memory_engine, "AAPL", analysis, "breakout"
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        data = json.loads(mem.value)
        assert data["strength"] in ("weak", "moderate", "strong")
        assert data["strength"] == "moderate"
        session.close()

    def test_confidence_values_match_existing_schema(self, in_memory_engine):
        """Confidence is low/medium/high — same as existing analyst."""
        analysis = _mock_analysis_response(confidence="high")

        _write_signal_memory(
            in_memory_engine, "AAPL", analysis, "gap_and_go"
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        data = json.loads(mem.value)
        assert data["confidence"] in ("low", "medium", "high")
        assert data["confidence"] == "high"
        session.close()

    def test_setup_type_uses_authoritative(self, in_memory_engine):
        """setup_type in signal uses the authoritative_setup_type parameter."""
        analysis = _mock_analysis_response(setup_type="momentum_continuation")

        _write_signal_memory(
            in_memory_engine, "AAPL", analysis, "breakout"
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        data = json.loads(mem.value)
        # authoritative_setup_type parameter takes precedence
        assert data["setup_type"] == "breakout"
        session.close()

    def test_key_levels_is_dict(self, in_memory_engine):
        """key_levels is a dict (PM reads support/resistance from it)."""
        analysis = _mock_analysis_response(
            key_levels={"support": 150.0, "resistance": 165.0, "vwap": 157.5}
        )

        _write_signal_memory(
            in_memory_engine, "AAPL", analysis, "gap_and_go"
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        data = json.loads(mem.value)
        assert isinstance(data["key_levels"], dict)
        assert data["key_levels"]["support"] == 150.0
        assert data["key_levels"]["resistance"] == 165.0
        session.close()

    def test_symbol_field_matches_query_symbol(self, in_memory_engine):
        """Symbol in JSON value matches the AgentMemory.symbol column."""
        analysis = _mock_analysis_response()

        _write_signal_memory(
            in_memory_engine, "MSFT", analysis, "breakout"
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="MSFT")
            .first()
        )
        data = json.loads(mem.value)
        assert data["symbol"] == "MSFT"
        session.close()

    def test_handles_missing_analysis_fields_gracefully(self, in_memory_engine):
        """Missing optional fields in analysis produce valid defaults."""
        # Minimal analysis dict
        analysis = {
            "signal_direction": "LONG",
            "signal_strength": "moderate",
        }

        _write_signal_memory(
            in_memory_engine, "AAPL", analysis, "unknown"
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        data = json.loads(mem.value)
        # Should produce valid record with defaults
        assert data["signal"] == "LONG"
        assert data["strength"] == "moderate"
        assert data["confidence"] == "low"  # default
        assert data["setup_type"] == "unknown"
        assert data["reasoning"] == ""  # default empty
        assert data["key_levels"] == {}  # default empty dict
        assert data["invalidation"] == ""  # default empty
        session.close()

    def test_normalizes_alternate_direction_formats(self, in_memory_engine):
        """Alternate direction formats (BUY, BULLISH) are normalized."""
        analysis = _mock_analysis_response(signal_direction="BUY")

        _write_signal_memory(
            in_memory_engine, "AAPL", analysis, "gap_and_go"
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        data = json.loads(mem.value)
        assert data["signal"] == "LONG"
        session.close()

    def test_normalizes_alternate_strength_formats(self, in_memory_engine):
        """Alternate strength formats (high, medium) are normalized."""
        analysis = _mock_analysis_response(signal_strength="high")

        _write_signal_memory(
            in_memory_engine, "AAPL", analysis, "gap_and_go"
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        data = json.loads(mem.value)
        assert data["strength"] == "strong"
        session.close()


# ---------------------------------------------------------------------------
# PM code path compatibility tests
# ---------------------------------------------------------------------------


class TestPMCodePathCompatibility:
    """Tests that signal records are queryable by the existing PM code path.

    The PM queries analyst signals with:
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol=X, key="signal")
        .order_by(AgentMemory.timestamp.desc())
        .first()

    Then parses: json.loads(mem.value)
    And accesses: sig.get("key_levels", {}), levels.get("support"), levels.get("resistance")
    """

    def test_queryable_by_pm_filter_pattern(self, in_memory_engine):
        """Record is found by the exact filter pattern used in portfolio_manager.py."""
        analysis = _mock_analysis_response()
        _write_signal_memory(in_memory_engine, "AAPL", analysis, "gap_and_go")

        # Simulate PM query pattern
        session = get_session(in_memory_engine)
        sig_mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", symbol="AAPL", key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        assert sig_mem is not None

        # PM parses value as JSON
        sig = json.loads(sig_mem.value)
        assert isinstance(sig, dict)
        session.close()

    def test_pm_can_read_key_levels_support(self, in_memory_engine):
        """PM can derive stop from support level in signal key_levels."""
        analysis = _mock_analysis_response(
            key_levels={"support": 175.50, "resistance": 182.00, "vwap": 178.25}
        )
        _write_signal_memory(in_memory_engine, "AAPL", analysis, "gap_and_go")

        session = get_session(in_memory_engine)
        sig_mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", symbol="AAPL", key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        sig = json.loads(sig_mem.value)

        # PM code: levels = sig.get("key_levels", {})
        levels = sig.get("key_levels", {})
        assert levels.get("support") == 175.50
        assert levels.get("resistance") == 182.00

        # PM derives stop: round(float(levels["support"]) * 0.995, 2)
        stop = round(float(levels["support"]) * 0.995, 2)
        assert stop == 174.62
        session.close()

    def test_pm_can_read_setup_type(self, in_memory_engine):
        """PM can read setup_type for confidence adjustment."""
        analysis = _mock_analysis_response(setup_type="momentum_continuation")
        _write_signal_memory(in_memory_engine, "AAPL", analysis, "momentum_continuation")

        session = get_session(in_memory_engine)
        sig_mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", symbol="AAPL", key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        sig = json.loads(sig_mem.value)
        assert sig.get("setup_type") == "momentum_continuation"
        session.close()

    def test_pm_can_read_strength_and_confidence(self, in_memory_engine):
        """PM can read strength and confidence for threshold comparisons."""
        analysis = _mock_analysis_response(signal_strength="strong", confidence="high")
        _write_signal_memory(in_memory_engine, "AAPL", analysis, "gap_and_go")

        session = get_session(in_memory_engine)
        sig_mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", symbol="AAPL", key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        sig = json.loads(sig_mem.value)

        # PM uses these in STRENGTH_ORDER comparisons
        assert sig.get("strength") == "strong"
        assert sig.get("confidence") == "high"
        session.close()

    def test_multiple_signals_ordered_by_timestamp(self, in_memory_engine):
        """When multiple signals exist, PM gets the latest via timestamp ordering."""
        analysis1 = _mock_analysis_response(signal_direction="HOLD", confidence="low")
        analysis2 = _mock_analysis_response(signal_direction="LONG", confidence="high")

        _write_signal_memory(in_memory_engine, "AAPL", analysis1, "unknown")
        _write_signal_memory(in_memory_engine, "AAPL", analysis2, "gap_and_go")

        session = get_session(in_memory_engine)
        sig_mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", symbol="AAPL", key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        sig = json.loads(sig_mem.value)
        # Latest should be the second write
        assert sig["signal"] == "LONG"
        assert sig["confidence"] == "high"
        session.close()


# ---------------------------------------------------------------------------
# Integration: run_funnel_analysis writes signal memory for promoted candidates
# ---------------------------------------------------------------------------


class TestRunFunnelAnalysisSignalMemory:
    """Integration tests verifying run_funnel_analysis writes signal records."""

    @patch("utils.funnel_analyst._evaluate_candidate_setup")
    def test_promoted_candidate_gets_signal_record(self, mock_eval, in_memory_engine):
        """Promoted candidates get an AgentMemory analyst signal record."""
        mock_eval.return_value = _mock_analysis_response(decision="promoted")

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        run_funnel_analysis(in_memory_engine, [candidate])

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        assert mem is not None
        data = json.loads(mem.value)
        assert data["symbol"] == "AAPL"
        assert data["signal"] == "LONG"
        assert data["setup_type"] == "gap_and_go"
        session.close()

    @patch("utils.funnel_analyst._evaluate_candidate_setup")
    def test_needs_confirmation_candidate_gets_signal_record(self, mock_eval, in_memory_engine):
        """needs_confirmation candidates also get a signal record (per code logic)."""
        mock_eval.return_value = _mock_analysis_response(decision="needs_confirmation")

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        run_funnel_analysis(in_memory_engine, [candidate])

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        assert mem is not None
        session.close()

    @patch("utils.funnel_analyst._evaluate_candidate_setup")
    def test_rejected_candidate_no_signal_record(self, mock_eval, in_memory_engine):
        """Rejected candidates do NOT get an AgentMemory signal record."""
        mock_eval.return_value = _mock_analysis_response(
            decision="rejected",
            signal_direction="HOLD",
            signal_strength="weak",
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        run_funnel_analysis(in_memory_engine, [candidate])

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        assert mem is None
        session.close()

    @patch("utils.funnel_analyst._evaluate_candidate_setup")
    def test_failed_candidate_no_signal_record(self, mock_eval, in_memory_engine):
        """Candidates that fail evaluation do NOT get a signal record."""
        mock_eval.side_effect = RuntimeError("LLM timeout")

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        run_funnel_analysis(in_memory_engine, [candidate])

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        assert mem is None
        session.close()

    @patch("utils.funnel_analyst._evaluate_candidate_setup")
    def test_signal_record_contains_authoritative_setup_type(self, mock_eval, in_memory_engine):
        """Signal record setup_type matches the authoritative type set on FunnelCandidate."""
        mock_eval.return_value = _mock_analysis_response(
            setup_type="breakout", decision="promoted"
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        run_funnel_analysis(in_memory_engine, [candidate])

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        data = json.loads(mem.value)
        assert data["setup_type"] == "breakout"

        # Also verify FunnelCandidate has the same authoritative type
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.authoritative_setup_type == "breakout"
        session.close()

    @patch("utils.funnel_analyst._evaluate_candidate_setup")
    def test_multiple_promoted_candidates_each_get_signal(self, mock_eval, in_memory_engine):
        """Each promoted candidate gets its own signal record."""
        mock_eval.return_value = _mock_analysis_response(decision="promoted")

        c1 = _make_funnel_candidate(in_memory_engine, "AAPL", scout_score=90.0)
        c2 = _make_funnel_candidate(in_memory_engine, "MSFT", scout_score=85.0)

        run_funnel_analysis(in_memory_engine, [c1, c2])

        session = get_session(in_memory_engine)
        aapl_mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        msft_mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="MSFT")
            .first()
        )
        assert aapl_mem is not None
        assert msft_mem is not None

        aapl_data = json.loads(aapl_mem.value)
        msft_data = json.loads(msft_mem.value)
        assert aapl_data["symbol"] == "AAPL"
        assert msft_data["symbol"] == "MSFT"
        session.close()

    @patch("utils.funnel_analyst._evaluate_candidate_setup")
    def test_wrong_stage_status_skipped_no_signal(self, mock_eval, in_memory_engine):
        """Candidates not in awaiting_analysis are skipped — no signal written."""
        mock_eval.return_value = _mock_analysis_response(decision="promoted")

        candidate = _make_funnel_candidate(
            in_memory_engine, "AAPL", stage_status="awaiting_research"
        )
        run_funnel_analysis(in_memory_engine, [candidate])

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="analyst", key="signal", symbol="AAPL")
            .first()
        )
        assert mem is None
        mock_eval.assert_not_called()
        session.close()
