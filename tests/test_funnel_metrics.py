"""Unit tests for funnel stage quality metrics and daily review reporting (Task 14.1).

Tests:
- generate_daily_review() — shortlist count, per-stage stats, fallback/timeout,
  PM handoff, trade outcomes, shadow outcomes, top 3 rejection reasons
- mark_shadow_eligible_candidates() — links rejected candidates with geometry
  to blocked_trade_candidates
- compute_stage_classifications() — trailing 5 day health classification
- Unique (date, symbol) counting — no inflation from re-evaluations

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine, text

from db.schema import Base, FunnelCandidate, FunnelRunLog, get_session
from utils.funnel_metrics import (
    generate_daily_review,
    mark_shadow_eligible_candidates,
    compute_stage_classifications,
    _compute_stage_stats,
    _compute_top_rejection_reasons,
    _classify_stage,
    _extract_geometry,
    _parse_stage_decisions,
    PIPELINE_STAGES,
)

_NY_TZ = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    # Create shadow ledger tables
    _create_shadow_tables(engine)
    return engine


def _create_shadow_tables(engine):
    """Create blocked_trade_candidates and outcomes tables for testing."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS blocked_trade_candidates (
                id INTEGER PRIMARY KEY,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR(10),
                action VARCHAR(16) NOT NULL,
                direction VARCHAR(10),
                profile VARCHAR(16),
                setup_type VARCHAR(64),
                entry_price REAL,
                stop_price REAL,
                target_price REAL,
                quantity REAL,
                blocked_by VARCHAR(64) NOT NULL,
                block_reason TEXT NOT NULL,
                reason_code VARCHAR(64),
                gate_notes_json TEXT,
                decision_snapshot_json TEXT,
                signal_snapshot_json TEXT,
                source VARCHAR(64),
                agent VARCHAR(64),
                dedupe_key VARCHAR(255),
                trade_event_id INTEGER
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS blocked_trade_candidate_outcomes (
                id INTEGER PRIMARY KEY,
                blocked_candidate_id INTEGER NOT NULL,
                eval_window VARCHAR(16) NOT NULL,
                evaluated_at DATETIME NOT NULL,
                eval_price REAL,
                pnl_pct REAL,
                mfe_pct REAL,
                mae_pct REAL,
                stop_hit BOOLEAN DEFAULT 0,
                target_hit BOOLEAN DEFAULT 0,
                first_hit VARCHAR(16),
                first_hit_at DATETIME,
                outcome_label VARCHAR(64),
                gate_verdict VARCHAR(64),
                notes_json TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(blocked_candidate_id, eval_window)
            )
        """))


def _make_candidate(
    engine,
    symbol: str = "AAPL",
    scout_rank: int = 1,
    scout_score: float = 85.0,
    stage_status: str = "pm_eligible",
    candidate_date: date | None = None,
    stage_decisions: list[dict] | None = None,
    direction_bias: str = "bullish",
    trade_event_id: int | None = None,
    blocked_candidate_id: int | None = None,
) -> FunnelCandidate:
    """Create and persist a FunnelCandidate for testing."""
    if candidate_date is None:
        candidate_date = datetime.now(_NY_TZ).date()

    if stage_decisions is None:
        stage_decisions = [
            {
                "agent": "scout",
                "timestamp": "2025-01-15T11:00:00Z",
                "decision": "promoted",
                "reasoning": "Strong mover",
                "evidence": {"catalyst": "earnings"},
                "next_stage": "awaiting_research",
            }
        ]

    candidate = FunnelCandidate(
        candidate_id=str(uuid.uuid4()),
        date=candidate_date,
        symbol=symbol,
        discovered_at=datetime(2025, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
        source_run="premarket",
        selection_mode="chief_scout",
        scout_rank=scout_rank,
        scout_score=scout_score,
        direction_bias=direction_bias,
        catalyst_evidence=json.dumps({"type": "earnings"}),
        selection_reason="Strong momentum",
        primary_risk="Market reversal",
        sector_context=json.dumps({"sector": "tech"}),
        stage_status=stage_status,
        stage_decisions=json.dumps(stage_decisions),
        trade_event_id=trade_event_id,
        blocked_candidate_id=blocked_candidate_id,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    session = get_session(engine)
    try:
        session.add(candidate)
        session.commit()
        session.refresh(candidate)
        return candidate
    finally:
        session.close()


def _make_run_log(
    engine,
    stage: str = "discovery",
    result_status: str = "completed",
    log_date: date | None = None,
    candidates_promoted: int = 3,
    candidates_rejected: int = 2,
) -> FunnelRunLog:
    """Create and persist a FunnelRunLog for testing."""
    if log_date is None:
        log_date = datetime.now(_NY_TZ).date()

    log_entry = FunnelRunLog(
        date=log_date,
        stage=stage,
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        duration_seconds=12.5,
        budget_seconds=90.0,
        result_status=result_status,
        candidates_input=5,
        candidates_promoted=candidates_promoted,
        candidates_rejected=candidates_rejected,
    )

    session = get_session(engine)
    try:
        session.add(log_entry)
        session.commit()
        session.refresh(log_entry)
        return log_entry
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tests: generate_daily_review
# ---------------------------------------------------------------------------


class TestGenerateDailyReview:
    """Tests for generate_daily_review()."""

    def test_empty_day_returns_zero_counts(self, in_memory_engine):
        """No candidates returns all-zero review."""
        today = datetime.now(_NY_TZ).date()
        review = generate_daily_review(in_memory_engine, today)

        assert review.review_date == today
        assert review.shortlist_count == 0
        assert review.pm_handoff_count == 0
        assert review.fallback_count == 0
        assert review.timeout_count == 0

    def test_shortlist_count_is_unique_candidates(self, in_memory_engine):
        """Shortlist count = number of unique (date, symbol) candidates."""
        today = datetime.now(_NY_TZ).date()
        _make_candidate(in_memory_engine, symbol="AAPL", candidate_date=today)
        _make_candidate(in_memory_engine, symbol="MSFT", candidate_date=today, scout_rank=2)
        _make_candidate(in_memory_engine, symbol="GOOG", candidate_date=today, scout_rank=3)

        review = generate_daily_review(in_memory_engine, today)
        assert review.shortlist_count == 3

    def test_pm_handoff_count(self, in_memory_engine):
        """PM handoff counts pm_eligible and executed candidates."""
        today = datetime.now(_NY_TZ).date()
        _make_candidate(in_memory_engine, symbol="AAPL", stage_status="pm_eligible",
                        candidate_date=today)
        _make_candidate(in_memory_engine, symbol="MSFT", stage_status="executed",
                        candidate_date=today, scout_rank=2)
        _make_candidate(in_memory_engine, symbol="GOOG", stage_status="rejected_research",
                        candidate_date=today, scout_rank=3)

        review = generate_daily_review(in_memory_engine, today)
        assert review.pm_handoff_count == 2

    def test_fallback_count_from_run_logs(self, in_memory_engine):
        """Fallback count from degraded discovery run logs."""
        today = datetime.now(_NY_TZ).date()
        _make_run_log(in_memory_engine, stage="discovery",
                      result_status="degraded", log_date=today)
        _make_run_log(in_memory_engine, stage="discovery",
                      result_status="completed", log_date=today)
        _make_candidate(in_memory_engine, symbol="AAPL", candidate_date=today)

        review = generate_daily_review(in_memory_engine, today)
        assert review.fallback_count == 1

    def test_timeout_count_includes_logs_and_decisions(self, in_memory_engine):
        """Timeout count includes run log timeouts and candidate decision timeouts."""
        today = datetime.now(_NY_TZ).date()
        _make_run_log(in_memory_engine, stage="confirmation",
                      result_status="timed_out", log_date=today)
        _make_candidate(
            in_memory_engine, symbol="AAPL", candidate_date=today,
            stage_status="awaiting_confirmation",
            stage_decisions=[
                {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                 "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                {"agent": "confirmation", "timestamp": "T2", "decision": "timed_out",
                 "reasoning": "budget exhausted", "evidence": {},
                 "next_stage": "awaiting_confirmation"},
            ],
        )

        review = generate_daily_review(in_memory_engine, today)
        # 1 from run log + 1 from candidate decision
        assert review.timeout_count == 2

    def test_stage_stats_per_stage(self, in_memory_engine):
        """Stage stats track promotions and rejections per stage."""
        today = datetime.now(_NY_TZ).date()
        # Candidate promoted through research, rejected at analysis
        _make_candidate(
            in_memory_engine, symbol="AAPL", candidate_date=today,
            stage_status="rejected_analysis",
            stage_decisions=[
                {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                 "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                {"agent": "researcher", "timestamp": "T2", "decision": "promoted",
                 "reasoning": "fresh catalyst", "evidence": {},
                 "next_stage": "awaiting_analysis"},
                {"agent": "analyst", "timestamp": "T3", "decision": "rejected",
                 "reasoning": "weak setup", "evidence": {},
                 "next_stage": "rejected_analysis"},
            ],
        )
        # Candidate rejected at research
        _make_candidate(
            in_memory_engine, symbol="MSFT", candidate_date=today, scout_rank=2,
            stage_status="rejected_research",
            stage_decisions=[
                {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                 "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                {"agent": "researcher", "timestamp": "T2", "decision": "rejected",
                 "reasoning": "stale catalyst", "evidence": {},
                 "next_stage": "rejected_research"},
            ],
        )

        review = generate_daily_review(in_memory_engine, today)

        # Discovery: 2 entered, 2 promoted (all discovered candidates)
        assert review.stage_stats["discovery"].candidates_entered == 2
        assert review.stage_stats["discovery"].promoted == 2

        # Research: 2 entered, 1 promoted, 1 rejected
        assert review.stage_stats["research"].candidates_entered == 2
        assert review.stage_stats["research"].promoted == 1
        assert review.stage_stats["research"].rejected == 1

        # Analysis: 1 entered, 0 promoted, 1 rejected
        assert review.stage_stats["analysis"].candidates_entered == 1
        assert review.stage_stats["analysis"].rejected == 1

    def test_top_rejection_reasons(self, in_memory_engine):
        """Top 3 rejection reasons sorted by frequency."""
        today = datetime.now(_NY_TZ).date()
        # 3 candidates rejected for different reasons
        for i, (sym, reason) in enumerate([
            ("A", "stale catalyst"),
            ("B", "stale catalyst"),
            ("C", "weak setup"),
            ("D", "stale catalyst"),
            ("E", "no volume"),
        ], start=1):
            _make_candidate(
                in_memory_engine, symbol=sym, candidate_date=today,
                scout_rank=i, stage_status="rejected_research",
                stage_decisions=[
                    {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                     "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                    {"agent": "researcher", "timestamp": "T2", "decision": "rejected",
                     "reasoning": reason, "evidence": {},
                     "next_stage": "rejected_research"},
                ],
            )

        review = generate_daily_review(in_memory_engine, today)

        assert len(review.top_rejection_reasons) == 3
        # "stale catalyst" appears 3 times — should be first
        assert review.top_rejection_reasons[0][0] == "stale catalyst"
        assert review.top_rejection_reasons[0][1] == 3

    def test_counts_unique_candidate_once(self, in_memory_engine):
        """Each (date, symbol) counted once even with multiple decisions."""
        today = datetime.now(_NY_TZ).date()
        # Candidate with multiple re-evaluation decisions (same candidate)
        _make_candidate(
            in_memory_engine, symbol="AAPL", candidate_date=today,
            stage_status="pm_eligible",
            stage_decisions=[
                {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                 "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                {"agent": "scout", "timestamp": "T1b", "decision": "promoted",
                 "reasoning": "re-discovery", "evidence": {},
                 "next_stage": "awaiting_research"},
                {"agent": "researcher", "timestamp": "T2", "decision": "promoted",
                 "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_analysis"},
                {"agent": "analyst", "timestamp": "T3", "decision": "promoted",
                 "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_confirmation"},
                {"agent": "confirmation", "timestamp": "T4", "decision": "promoted",
                 "reasoning": "ok", "evidence": {}, "next_stage": "pm_eligible"},
            ],
        )

        review = generate_daily_review(in_memory_engine, today)
        # Only 1 candidate — counted once
        assert review.shortlist_count == 1
        assert review.pm_handoff_count == 1


# ---------------------------------------------------------------------------
# Tests: mark_shadow_eligible_candidates
# ---------------------------------------------------------------------------


class TestMarkShadowEligible:
    """Tests for mark_shadow_eligible_candidates()."""

    def test_links_rejected_with_geometry(self, in_memory_engine):
        """Rejected candidate with geometry is linked to shadow ledger."""
        today = datetime.now(_NY_TZ).date()
        _make_candidate(
            in_memory_engine, symbol="NVDA", candidate_date=today,
            stage_status="rejected_analysis",
            stage_decisions=[
                {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                 "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                {"agent": "analyst", "timestamp": "T2", "decision": "rejected",
                 "reasoning": "weak setup",
                 "evidence": {
                     "key_levels": {
                         "entry_price": 150.0,
                         "stop_price": 145.0,
                         "target_price": 165.0,
                     }
                 },
                 "next_stage": "rejected_analysis"},
            ],
        )

        linked = mark_shadow_eligible_candidates(in_memory_engine, today)
        assert len(linked) == 1

        # Verify FunnelCandidate is now linked
        session = get_session(in_memory_engine)
        try:
            candidate = session.query(FunnelCandidate).filter(
                FunnelCandidate.symbol == "NVDA"
            ).first()
            assert candidate.blocked_candidate_id is not None
        finally:
            session.close()

    def test_skips_candidates_without_geometry(self, in_memory_engine):
        """Rejected candidate without geometry is not linked."""
        today = datetime.now(_NY_TZ).date()
        _make_candidate(
            in_memory_engine, symbol="AAPL", candidate_date=today,
            stage_status="rejected_research",
            stage_decisions=[
                {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                 "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                {"agent": "researcher", "timestamp": "T2", "decision": "rejected",
                 "reasoning": "stale catalyst", "evidence": {},
                 "next_stage": "rejected_research"},
            ],
        )

        linked = mark_shadow_eligible_candidates(in_memory_engine, today)
        assert len(linked) == 0

    def test_skips_non_rejected_candidates(self, in_memory_engine):
        """Non-rejected candidates are not linked to shadow ledger."""
        today = datetime.now(_NY_TZ).date()
        _make_candidate(
            in_memory_engine, symbol="AAPL", candidate_date=today,
            stage_status="pm_eligible",
            stage_decisions=[
                {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                 "reasoning": "ok",
                 "evidence": {"entry_price": 100, "stop_price": 95, "target_price": 110},
                 "next_stage": "awaiting_research"},
            ],
        )

        linked = mark_shadow_eligible_candidates(in_memory_engine, today)
        assert len(linked) == 0

    def test_skips_already_linked_candidates(self, in_memory_engine):
        """Candidates already linked to blocked_candidate_id are skipped."""
        today = datetime.now(_NY_TZ).date()
        _make_candidate(
            in_memory_engine, symbol="AAPL", candidate_date=today,
            stage_status="rejected_analysis",
            blocked_candidate_id=42,
            stage_decisions=[
                {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                 "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                {"agent": "analyst", "timestamp": "T2", "decision": "rejected",
                 "reasoning": "weak",
                 "evidence": {
                     "key_levels": {
                         "entry_price": 100, "stop_price": 95, "target_price": 110,
                     }
                 },
                 "next_stage": "rejected_analysis"},
            ],
        )

        linked = mark_shadow_eligible_candidates(in_memory_engine, today)
        assert len(linked) == 0


# ---------------------------------------------------------------------------
# Tests: compute_stage_classifications
# ---------------------------------------------------------------------------


class TestStageClassifications:
    """Tests for compute_stage_classifications()."""

    def test_healthy_stage_no_classification(self, in_memory_engine):
        """Stage with balanced rates has no classification."""
        today = datetime.now(_NY_TZ).date()
        # Create 12 candidates over the past few days with mixed decisions
        for i in range(12):
            _make_candidate(
                in_memory_engine, symbol=f"SYM{i}", candidate_date=today,
                scout_rank=i + 1,
                stage_status="awaiting_analysis" if i < 6 else "rejected_research",
                stage_decisions=[
                    {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                     "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                    {"agent": "researcher", "timestamp": "T2",
                     "decision": "promoted" if i < 6 else "rejected",
                     "reasoning": "ok" if i < 6 else "stale",
                     "evidence": {},
                     "next_stage": "awaiting_analysis" if i < 6 else "rejected_research"},
                ],
            )

        classifications = compute_stage_classifications(in_memory_engine, today)
        # Research stage: 6 promoted / 12 entered = 50% — healthy
        assert classifications["research"].classification is None

    def test_overly_permissive_classification(self, in_memory_engine):
        """Stage promoting >90% is classified as overly_permissive."""
        today = datetime.now(_NY_TZ).date()
        # 10 candidates, 10 promoted (100% promotion rate)
        for i in range(10):
            _make_candidate(
                in_memory_engine, symbol=f"SYM{i}", candidate_date=today,
                scout_rank=i + 1,
                stage_status="awaiting_analysis",
                stage_decisions=[
                    {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                     "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                    {"agent": "researcher", "timestamp": "T2", "decision": "promoted",
                     "reasoning": "ok", "evidence": {},
                     "next_stage": "awaiting_analysis"},
                ],
            )

        classifications = compute_stage_classifications(in_memory_engine, today)
        assert classifications["research"].classification == "overly_permissive"
        assert classifications["research"].promotion_rate > 0.90

    def test_overly_restrictive_classification(self, in_memory_engine):
        """Stage rejecting >90% is classified as overly_restrictive."""
        today = datetime.now(_NY_TZ).date()
        # 10 candidates, 10 rejected (100% rejection rate)
        for i in range(10):
            _make_candidate(
                in_memory_engine, symbol=f"SYM{i}", candidate_date=today,
                scout_rank=i + 1,
                stage_status="rejected_research",
                stage_decisions=[
                    {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                     "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                    {"agent": "researcher", "timestamp": "T2", "decision": "rejected",
                     "reasoning": "stale", "evidence": {},
                     "next_stage": "rejected_research"},
                ],
            )

        classifications = compute_stage_classifications(in_memory_engine, today)
        assert classifications["research"].classification == "overly_restrictive"
        assert classifications["research"].rejection_rate > 0.90

    def test_operationally_failing_classification(self, in_memory_engine):
        """Stage timing out >30% is classified as operationally_failing."""
        today = datetime.now(_NY_TZ).date()
        # 10 candidates: 4 timed out, 6 promoted (40% timeout rate > 30%)
        for i in range(10):
            is_timeout = i < 4
            _make_candidate(
                in_memory_engine, symbol=f"SYM{i}", candidate_date=today,
                scout_rank=i + 1,
                stage_status="awaiting_confirmation" if not is_timeout else "awaiting_confirmation",
                stage_decisions=[
                    {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                     "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                    {"agent": "researcher", "timestamp": "T2",
                     "decision": "timed_out" if is_timeout else "promoted",
                     "reasoning": "budget" if is_timeout else "ok",
                     "evidence": {},
                     "next_stage": "awaiting_research" if is_timeout else "awaiting_analysis"},
                ],
            )

        classifications = compute_stage_classifications(in_memory_engine, today)
        assert classifications["research"].classification == "operationally_failing"
        assert classifications["research"].timeout_rate > 0.30

    def test_insufficient_data_no_classification(self, in_memory_engine):
        """Stage with fewer than 10 candidates gets no classification."""
        today = datetime.now(_NY_TZ).date()
        # Only 5 candidates — below minimum threshold
        for i in range(5):
            _make_candidate(
                in_memory_engine, symbol=f"SYM{i}", candidate_date=today,
                scout_rank=i + 1,
                stage_status="rejected_research",
                stage_decisions=[
                    {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                     "reasoning": "ok", "evidence": {}, "next_stage": "awaiting_research"},
                    {"agent": "researcher", "timestamp": "T2", "decision": "rejected",
                     "reasoning": "stale", "evidence": {},
                     "next_stage": "rejected_research"},
                ],
            )

        classifications = compute_stage_classifications(in_memory_engine, today)
        # Not enough data to classify
        assert classifications["research"].classification is None
        assert classifications["research"].candidates_in_window == 5

    def test_trailing_window_uses_multiple_days(self, in_memory_engine):
        """Classification window spans up to 5 trading days."""
        today = datetime.now(_NY_TZ).date()
        # Spread 12 candidates across 3 days
        for day_offset in range(3):
            day = today - timedelta(days=day_offset)
            for i in range(4):
                sym = f"D{day_offset}S{i}"
                _make_candidate(
                    in_memory_engine, symbol=sym, candidate_date=day,
                    scout_rank=i + 1,
                    stage_status="rejected_research",
                    stage_decisions=[
                        {"agent": "scout", "timestamp": "T1", "decision": "promoted",
                         "reasoning": "ok", "evidence": {},
                         "next_stage": "awaiting_research"},
                        {"agent": "researcher", "timestamp": "T2",
                         "decision": "rejected",
                         "reasoning": "stale", "evidence": {},
                         "next_stage": "rejected_research"},
                    ],
                )

        classifications = compute_stage_classifications(in_memory_engine, today)
        # 12 candidates across 3 days — enough to classify
        assert classifications["research"].candidates_in_window == 12
        assert classifications["research"].days_in_window == 3


# ---------------------------------------------------------------------------
# Tests: Internal helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal helper functions."""

    def test_parse_stage_decisions_valid_json(self):
        """Parses valid JSON stage_decisions."""
        candidate = FunnelCandidate(
            stage_decisions=json.dumps([
                {"agent": "scout", "decision": "promoted"}
            ])
        )
        result = _parse_stage_decisions(candidate)
        assert len(result) == 1
        assert result[0]["agent"] == "scout"

    def test_parse_stage_decisions_invalid_json(self):
        """Returns empty list for invalid JSON."""
        candidate = FunnelCandidate(stage_decisions="not json")
        result = _parse_stage_decisions(candidate)
        assert result == []

    def test_parse_stage_decisions_none(self):
        """Returns empty list for None."""
        candidate = FunnelCandidate(stage_decisions=None)
        result = _parse_stage_decisions(candidate)
        assert result == []

    def test_extract_geometry_from_analyst(self):
        """Extracts geometry from Analyst evidence key_levels."""
        candidate = FunnelCandidate(
            stage_decisions=json.dumps([
                {"agent": "scout", "decision": "promoted", "evidence": {}},
                {"agent": "analyst", "decision": "rejected",
                 "evidence": {
                     "key_levels": {
                         "entry_price": 100.0,
                         "stop_price": 95.0,
                         "target_price": 110.0,
                     }
                 }},
            ])
        )
        geo = _extract_geometry(candidate)
        assert geo is not None
        assert geo["entry_price"] == 100.0
        assert geo["stop_price"] == 95.0
        assert geo["target_price"] == 110.0

    def test_extract_geometry_none_when_missing(self):
        """Returns None when no geometry fields found."""
        candidate = FunnelCandidate(
            stage_decisions=json.dumps([
                {"agent": "scout", "decision": "promoted", "evidence": {}},
            ])
        )
        geo = _extract_geometry(candidate)
        assert geo is None

    def test_extract_geometry_from_scout_evidence(self):
        """Falls back to Scout evidence when Analyst has no geometry."""
        candidate = FunnelCandidate(
            stage_decisions=json.dumps([
                {"agent": "scout", "decision": "promoted",
                 "evidence": {
                     "entry_price": 50.0,
                     "stop_price": 48.0,
                     "target_price": 55.0,
                 }},
            ])
        )
        geo = _extract_geometry(candidate)
        assert geo is not None
        assert geo["entry_price"] == 50.0
