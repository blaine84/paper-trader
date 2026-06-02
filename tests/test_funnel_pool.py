"""Unit tests for funnel pool functions (Task 9.1).

Tests:
- get_pm_eligible_candidates() — deterministic ordering, max_handoff ceiling,
  empty results, expired candidates excluded
- get_candidate_context() — full context with stage decisions organized by agent
- expire_daily_candidates() — marks yesterday's candidates as expired without
  deleting records
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine

from db.schema import Base, FunnelCandidate, get_session
from utils.funnel_pool import (
    get_pm_eligible_candidates,
    get_candidate_context,
    expire_daily_candidates,
    _get_confirmation_timestamp,
    _extract_agent_context,
    _extract_analyst_context,
    _safe_json_parse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NY_TZ = ZoneInfo("America/New_York")


@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _make_funnel_candidate(
    engine,
    symbol: str = "AAPL",
    scout_rank: int = 1,
    scout_score: float = 85.0,
    stage_status: str = "pm_eligible",
    expired: bool = False,
    candidate_date: date | None = None,
    confirmation_ts: str | None = None,
    direction_bias: str = "bullish",
) -> FunnelCandidate:
    """Create and persist a FunnelCandidate for testing."""
    if candidate_date is None:
        candidate_date = datetime.now(_NY_TZ).date()

    # Build stage_decisions with confirmation timestamp
    scout_decision = {
        "agent": "scout",
        "timestamp": "2025-01-15T11:00:00Z",
        "decision": "promoted",
        "reasoning": "Strong mover",
        "evidence": {"catalyst": "earnings"},
        "next_stage": "awaiting_research",
    }
    researcher_decision = {
        "agent": "researcher",
        "timestamp": "2025-01-15T11:30:00Z",
        "decision": "promoted",
        "reasoning": "Fresh catalyst confirmed",
        "evidence": {"catalyst_validation": {"freshness": "fresh", "specificity": True}},
        "next_stage": "awaiting_analysis",
    }
    analyst_decision = {
        "agent": "analyst",
        "timestamp": "2025-01-15T12:00:00Z",
        "decision": "promoted",
        "reasoning": "Gap and go setup",
        "evidence": {
            "authoritative_setup_type": "gap_and_go",
            "signal_direction": "LONG",
            "signal_strength": "strong",
            "key_levels": {"support": 145.0, "resistance": 160.0},
            "invalidation": "Close below 144.0",
            "volume_requirements": "Volume above 1M",
        },
        "next_stage": "awaiting_confirmation",
    }
    conf_ts = confirmation_ts or "2025-01-15T14:35:00Z"
    confirmation_decision = {
        "agent": "confirmation",
        "timestamp": conf_ts,
        "decision": "promoted",
        "reasoning": "Volume and price confirmed",
        "evidence": {"volume_confirmed": True, "price_behavior_ok": True},
        "next_stage": "pm_eligible",
    }

    stage_decisions = [scout_decision, researcher_decision, analyst_decision, confirmation_decision]

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
        catalyst_evidence=json.dumps({"type": "earnings", "detail": "Beat Q4 estimates"}),
        selection_reason="Strong momentum with catalyst",
        primary_risk="Market reversal",
        sector_context=json.dumps({"sector": "tech", "etf": "XLK"}),
        preliminary_setup_type="gap_and_go",
        authoritative_setup_type="gap_and_go" if stage_status == "pm_eligible" else None,
        stage_status=stage_status,
        stage_decisions=json.dumps(stage_decisions),
        expired=expired,
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


# ---------------------------------------------------------------------------
# Tests: get_pm_eligible_candidates
# ---------------------------------------------------------------------------


class TestGetPmEligibleCandidates:
    """Tests for get_pm_eligible_candidates()."""

    def test_returns_empty_when_no_candidates(self, in_memory_engine):
        """No pm_eligible candidates today returns empty list."""
        result = get_pm_eligible_candidates(in_memory_engine)
        assert result == []

    def test_returns_pm_eligible_symbols(self, in_memory_engine):
        """Returns symbols for today's pm_eligible candidates."""
        _make_funnel_candidate(in_memory_engine, symbol="AAPL", scout_rank=1)
        _make_funnel_candidate(in_memory_engine, symbol="MSFT", scout_rank=2)

        result = get_pm_eligible_candidates(in_memory_engine)
        assert set(result) == {"AAPL", "MSFT"}

    def test_excludes_expired_candidates(self, in_memory_engine):
        """Expired candidates are not returned."""
        _make_funnel_candidate(in_memory_engine, symbol="AAPL", expired=False)
        _make_funnel_candidate(in_memory_engine, symbol="MSFT", expired=True)

        result = get_pm_eligible_candidates(in_memory_engine)
        assert result == ["AAPL"]

    def test_excludes_non_pm_eligible_status(self, in_memory_engine):
        """Only pm_eligible candidates returned, not other statuses."""
        _make_funnel_candidate(in_memory_engine, symbol="AAPL", stage_status="pm_eligible")
        _make_funnel_candidate(in_memory_engine, symbol="MSFT", stage_status="awaiting_confirmation")
        _make_funnel_candidate(in_memory_engine, symbol="GOOG", stage_status="rejected_research")

        result = get_pm_eligible_candidates(in_memory_engine)
        assert result == ["AAPL"]

    def test_enforces_max_handoff_ceiling(self, in_memory_engine):
        """Returns at most max_handoff symbols."""
        for i, sym in enumerate(["AAPL", "MSFT", "GOOG", "AMZN", "META"], start=1):
            _make_funnel_candidate(
                in_memory_engine,
                symbol=sym,
                scout_rank=i,
                confirmation_ts=f"2025-01-15T14:{35 + i}:00Z",
            )

        result = get_pm_eligible_candidates(in_memory_engine, max_handoff=3)
        assert len(result) == 3

    def test_deterministic_ordering_by_confirmation_timestamp(self, in_memory_engine):
        """Orders by confirmation timestamp ASC (earliest first)."""
        _make_funnel_candidate(
            in_memory_engine, symbol="LATE", scout_rank=1,
            confirmation_ts="2025-01-15T14:40:00Z",
        )
        _make_funnel_candidate(
            in_memory_engine, symbol="EARLY", scout_rank=2,
            confirmation_ts="2025-01-15T14:35:00Z",
        )

        result = get_pm_eligible_candidates(in_memory_engine, max_handoff=10)
        assert result[0] == "EARLY"
        assert result[1] == "LATE"

    def test_deterministic_ordering_tiebreak_scout_rank(self, in_memory_engine):
        """Same confirmation timestamp breaks tie by scout_rank ASC."""
        _make_funnel_candidate(
            in_memory_engine, symbol="RANK3", scout_rank=3,
            confirmation_ts="2025-01-15T14:35:00Z",
        )
        _make_funnel_candidate(
            in_memory_engine, symbol="RANK1", scout_rank=1,
            confirmation_ts="2025-01-15T14:35:00Z",
        )

        result = get_pm_eligible_candidates(in_memory_engine, max_handoff=10)
        assert result[0] == "RANK1"
        assert result[1] == "RANK3"

    def test_deterministic_ordering_tiebreak_scout_score(self, in_memory_engine):
        """Same confirmation timestamp and rank breaks tie by scout_score DESC."""
        _make_funnel_candidate(
            in_memory_engine, symbol="LOW", scout_rank=1, scout_score=70.0,
            confirmation_ts="2025-01-15T14:35:00Z",
        )
        _make_funnel_candidate(
            in_memory_engine, symbol="HIGH", scout_rank=1, scout_score=95.0,
            confirmation_ts="2025-01-15T14:35:00Z",
        )

        result = get_pm_eligible_candidates(in_memory_engine, max_handoff=10)
        assert result[0] == "HIGH"
        assert result[1] == "LOW"

    def test_excludes_other_dates(self, in_memory_engine):
        """Only today's candidates are returned."""
        yesterday = datetime.now(_NY_TZ).date() - timedelta(days=1)
        _make_funnel_candidate(
            in_memory_engine, symbol="OLD", candidate_date=yesterday
        )
        _make_funnel_candidate(in_memory_engine, symbol="TODAY")

        result = get_pm_eligible_candidates(in_memory_engine)
        assert result == ["TODAY"]


# ---------------------------------------------------------------------------
# Tests: get_candidate_context
# ---------------------------------------------------------------------------


class TestGetCandidateContext:
    """Tests for get_candidate_context()."""

    def test_returns_none_for_unknown_symbol(self, in_memory_engine):
        """Returns None when symbol doesn't exist today."""
        result = get_candidate_context(in_memory_engine, "UNKNOWN")
        assert result is None

    def test_returns_none_for_expired_candidate(self, in_memory_engine):
        """Returns None for expired candidates."""
        _make_funnel_candidate(in_memory_engine, symbol="AAPL", expired=True)

        result = get_candidate_context(in_memory_engine, "AAPL")
        assert result is None

    def test_returns_full_context_for_valid_candidate(self, in_memory_engine):
        """Returns complete context including all agent decisions."""
        _make_funnel_candidate(in_memory_engine, symbol="AAPL")

        result = get_candidate_context(in_memory_engine, "AAPL")
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["stage_status"] == "pm_eligible"

    def test_context_includes_scout_fields(self, in_memory_engine):
        """Context includes Scout selection_reason and scout_score."""
        _make_funnel_candidate(in_memory_engine, symbol="AAPL", scout_score=92.0)

        result = get_candidate_context(in_memory_engine, "AAPL")
        assert result["scout"]["scout_score"] == 92.0
        assert result["scout"]["selection_reason"] == "Strong momentum with catalyst"
        assert result["scout"]["direction_bias"] == "bullish"

    def test_context_includes_researcher_decision(self, in_memory_engine):
        """Context includes Researcher catalyst_validation and reasoning."""
        _make_funnel_candidate(in_memory_engine, symbol="AAPL")

        result = get_candidate_context(in_memory_engine, "AAPL")
        assert result["researcher"] is not None
        assert result["researcher"]["decision"] == "promoted"
        assert "catalyst_validation" in result["researcher"]["evidence"]

    def test_context_includes_analyst_context(self, in_memory_engine):
        """Context includes Analyst authoritative_setup_type and key_levels."""
        _make_funnel_candidate(in_memory_engine, symbol="AAPL")

        result = get_candidate_context(in_memory_engine, "AAPL")
        assert result["analyst"] is not None
        assert result["analyst"]["authoritative_setup_type"] == "gap_and_go"
        assert result["analyst"]["key_levels"] == {"support": 145.0, "resistance": 160.0}
        assert result["analyst"]["signal_direction"] == "LONG"

    def test_context_includes_confirmation_decision(self, in_memory_engine):
        """Context includes Confirmation decision with reasoning."""
        _make_funnel_candidate(in_memory_engine, symbol="AAPL")

        result = get_candidate_context(in_memory_engine, "AAPL")
        assert result["confirmation"] is not None
        assert result["confirmation"]["decision"] == "promoted"

    def test_context_includes_full_stage_decisions_history(self, in_memory_engine):
        """Context includes the full ordered stage_decisions list."""
        _make_funnel_candidate(in_memory_engine, symbol="AAPL")

        result = get_candidate_context(in_memory_engine, "AAPL")
        assert len(result["stage_decisions"]) == 4
        agents = [sd["agent"] for sd in result["stage_decisions"]]
        assert agents == ["scout", "researcher", "analyst", "confirmation"]


# ---------------------------------------------------------------------------
# Tests: expire_daily_candidates
# ---------------------------------------------------------------------------


class TestExpireDailyCandidates:
    """Tests for expire_daily_candidates()."""

    def test_expires_yesterday_candidates(self, in_memory_engine):
        """Marks yesterday's active candidates as expired."""
        yesterday = datetime.now(_NY_TZ).date() - timedelta(days=1)
        _make_funnel_candidate(
            in_memory_engine, symbol="OLD1", candidate_date=yesterday
        )
        _make_funnel_candidate(
            in_memory_engine, symbol="OLD2", candidate_date=yesterday, scout_rank=2
        )

        count = expire_daily_candidates(in_memory_engine)
        assert count == 2

        # Verify expired flag is set
        session = get_session(in_memory_engine)
        try:
            candidates = session.query(FunnelCandidate).filter(
                FunnelCandidate.date == yesterday
            ).all()
            assert all(c.expired is True for c in candidates)
        finally:
            session.close()

    def test_does_not_expire_today_candidates(self, in_memory_engine):
        """Today's candidates are not expired."""
        _make_funnel_candidate(in_memory_engine, symbol="TODAY")

        count = expire_daily_candidates(in_memory_engine)
        assert count == 0

        # Verify today's candidate is still active
        session = get_session(in_memory_engine)
        try:
            candidate = session.query(FunnelCandidate).filter(
                FunnelCandidate.symbol == "TODAY"
            ).first()
            assert candidate.expired is False
        finally:
            session.close()

    def test_does_not_double_expire(self, in_memory_engine):
        """Already expired candidates are not counted again."""
        yesterday = datetime.now(_NY_TZ).date() - timedelta(days=1)
        _make_funnel_candidate(
            in_memory_engine, symbol="ALREADY", candidate_date=yesterday, expired=True
        )

        count = expire_daily_candidates(in_memory_engine)
        assert count == 0

    def test_preserves_records_not_deletes(self, in_memory_engine):
        """Records remain queryable after expiry (not deleted)."""
        yesterday = datetime.now(_NY_TZ).date() - timedelta(days=1)
        _make_funnel_candidate(
            in_memory_engine, symbol="KEEP", candidate_date=yesterday
        )

        expire_daily_candidates(in_memory_engine)

        # Record still exists in DB
        session = get_session(in_memory_engine)
        try:
            candidate = session.query(FunnelCandidate).filter(
                FunnelCandidate.symbol == "KEEP"
            ).first()
            assert candidate is not None
            assert candidate.expired is True
            assert candidate.stage_decisions is not None
        finally:
            session.close()

    def test_returns_zero_when_no_candidates(self, in_memory_engine):
        """Returns 0 when no candidates exist for yesterday."""
        count = expire_daily_candidates(in_memory_engine)
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: Internal Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal helper functions."""

    def test_get_confirmation_timestamp_finds_promoted(self):
        """Extracts timestamp from confirmation promoted decision."""
        candidate = FunnelCandidate(
            stage_decisions=json.dumps([
                {"agent": "scout", "timestamp": "2025-01-15T11:00:00Z", "decision": "promoted"},
                {"agent": "confirmation", "timestamp": "2025-01-15T14:35:00Z", "decision": "promoted"},
            ])
        )
        ts = _get_confirmation_timestamp(candidate)
        assert ts == "2025-01-15T14:35:00Z"

    def test_get_confirmation_timestamp_sentinel_when_missing(self):
        """Returns sentinel value when no confirmation decision exists."""
        candidate = FunnelCandidate(
            stage_decisions=json.dumps([
                {"agent": "scout", "timestamp": "2025-01-15T11:00:00Z", "decision": "promoted"},
            ])
        )
        ts = _get_confirmation_timestamp(candidate)
        assert ts == "9999-12-31T23:59:59Z"

    def test_safe_json_parse_valid_json(self):
        """Parses valid JSON text."""
        result = _safe_json_parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_safe_json_parse_invalid_json(self):
        """Returns raw text for invalid JSON."""
        result = _safe_json_parse("not json")
        assert result == "not json"

    def test_safe_json_parse_none(self):
        """Returns None for None input."""
        result = _safe_json_parse(None)
        assert result is None

    def test_extract_agent_context_empty(self):
        """Returns None for empty decisions list."""
        result = _extract_agent_context([])
        assert result is None

    def test_extract_agent_context_prefers_promoted(self):
        """Returns most recent promoted/needs_confirmation decision."""
        decisions = [
            {"decision": "promoted", "reasoning": "first", "evidence": {}, "timestamp": "T1"},
            {"decision": "rejected", "reasoning": "later", "evidence": {}, "timestamp": "T2"},
        ]
        result = _extract_agent_context(decisions)
        # Returns the promoted decision (reversed search finds it)
        assert result["decision"] == "promoted"
        assert result["reasoning"] == "first"


# ---------------------------------------------------------------------------
# Tests: build_full_watchlist_with_funnel
# ---------------------------------------------------------------------------


class TestBuildFullWatchlistWithFunnel:
    """Tests for build_full_watchlist_with_funnel() (Task 9.2).

    Validates:
    - Deduplication across core, scout_picks, and funnel candidates
    - max_pm_handoff ceiling on funnel additions
    - PM input includes required context (Scout/Researcher/Analyst/Confirmation)
    - Existing risk gates apply unchanged (no special treatment)
    - Empty funnel returns core + scout_picks only
    """

    def test_combines_core_scout_and_funnel(self, in_memory_engine):
        """Combines all three sources into one deduplicated watchlist."""
        from utils.funnel_pool import build_full_watchlist_with_funnel

        _make_funnel_candidate(in_memory_engine, symbol="FUNNEL1", scout_rank=1)
        _make_funnel_candidate(in_memory_engine, symbol="FUNNEL2", scout_rank=2)

        result = build_full_watchlist_with_funnel(
            in_memory_engine,
            core_watchlist=["SPY", "QQQ"],
            scout_picks=["AAPL", "MSFT"],
            max_pm_handoff=3,
        )

        wl = result["full_watchlist"]
        # Core first, then scout, then funnel
        assert wl[:2] == ["SPY", "QQQ"]
        assert wl[2:4] == ["AAPL", "MSFT"]
        assert set(wl[4:]) == {"FUNNEL1", "FUNNEL2"}
        assert len(wl) == 6

    def test_deduplicates_funnel_already_in_core(self, in_memory_engine):
        """Funnel candidate already in core watchlist is not duplicated."""
        from utils.funnel_pool import build_full_watchlist_with_funnel

        _make_funnel_candidate(in_memory_engine, symbol="SPY", scout_rank=1)

        result = build_full_watchlist_with_funnel(
            in_memory_engine,
            core_watchlist=["SPY", "QQQ"],
            scout_picks=[],
            max_pm_handoff=3,
        )

        # SPY appears only once (from core)
        assert result["full_watchlist"].count("SPY") == 1
        # But funnel_symbols still lists SPY for context purposes
        assert "SPY" in result["funnel_symbols"]

    def test_deduplicates_funnel_already_in_scout_picks(self, in_memory_engine):
        """Funnel candidate already in scout_picks is not duplicated."""
        from utils.funnel_pool import build_full_watchlist_with_funnel

        _make_funnel_candidate(in_memory_engine, symbol="AAPL", scout_rank=1)

        result = build_full_watchlist_with_funnel(
            in_memory_engine,
            core_watchlist=["SPY"],
            scout_picks=["AAPL"],
            max_pm_handoff=3,
        )

        assert result["full_watchlist"].count("AAPL") == 1
        assert "AAPL" in result["funnel_symbols"]

    def test_enforces_max_pm_handoff_ceiling(self, in_memory_engine):
        """At most max_pm_handoff funnel candidates are included."""
        from utils.funnel_pool import build_full_watchlist_with_funnel

        for i, sym in enumerate(["F1", "F2", "F3", "F4", "F5"], start=1):
            _make_funnel_candidate(
                in_memory_engine, symbol=sym, scout_rank=i,
                confirmation_ts=f"2025-01-15T14:{35 + i}:00Z",
            )

        result = build_full_watchlist_with_funnel(
            in_memory_engine,
            core_watchlist=["SPY"],
            scout_picks=[],
            max_pm_handoff=2,
        )

        # Only 2 funnel symbols included
        assert len(result["funnel_symbols"]) == 2

    def test_empty_funnel_returns_core_plus_scout_only(self, in_memory_engine):
        """No funnel candidates returns just core + scout_picks."""
        from utils.funnel_pool import build_full_watchlist_with_funnel

        result = build_full_watchlist_with_funnel(
            in_memory_engine,
            core_watchlist=["SPY", "QQQ"],
            scout_picks=["AAPL"],
            max_pm_handoff=3,
        )

        assert result["full_watchlist"] == ["SPY", "QQQ", "AAPL"]
        assert result["funnel_symbols"] == []
        assert result["funnel_context"] == {}

    def test_funnel_context_includes_required_fields(self, in_memory_engine):
        """Funnel context includes Scout, Researcher, Analyst, Confirmation info."""
        from utils.funnel_pool import build_full_watchlist_with_funnel

        _make_funnel_candidate(in_memory_engine, symbol="NVDA", scout_rank=1)

        result = build_full_watchlist_with_funnel(
            in_memory_engine,
            core_watchlist=["SPY"],
            scout_picks=[],
            max_pm_handoff=3,
        )

        ctx = result["funnel_context"]["NVDA"]
        # Scout selection_reason and scout_score (Req 9.3)
        assert ctx["scout"]["selection_reason"] is not None
        assert ctx["scout"]["scout_score"] > 0
        # Researcher catalyst_validation (Req 9.3)
        assert ctx["researcher"] is not None
        assert ctx["researcher"]["decision"] == "promoted"
        # Analyst setup_type/levels/invalidation (Req 9.3)
        assert ctx["analyst"] is not None
        assert ctx["analyst"]["authoritative_setup_type"] == "gap_and_go"
        assert ctx["analyst"]["key_levels"] is not None
        assert ctx["analyst"]["invalidation"] is not None
        # Confirmation decision (Req 9.3)
        assert ctx["confirmation"] is not None
        assert ctx["confirmation"]["decision"] == "promoted"

    def test_deduplication_preserves_order(self, in_memory_engine):
        """Core symbols always come first regardless of funnel ordering."""
        from utils.funnel_pool import build_full_watchlist_with_funnel

        _make_funnel_candidate(in_memory_engine, symbol="NEW1", scout_rank=1)

        result = build_full_watchlist_with_funnel(
            in_memory_engine,
            core_watchlist=["QQQ", "SPY"],
            scout_picks=["MSFT"],
            max_pm_handoff=3,
        )

        wl = result["full_watchlist"]
        assert wl.index("QQQ") < wl.index("SPY") < wl.index("MSFT") < wl.index("NEW1")

    def test_core_and_scout_duplicates_handled(self, in_memory_engine):
        """Symbols in both core and scout_picks appear only once."""
        from utils.funnel_pool import build_full_watchlist_with_funnel

        result = build_full_watchlist_with_funnel(
            in_memory_engine,
            core_watchlist=["SPY", "QQQ"],
            scout_picks=["QQQ", "AAPL"],
            max_pm_handoff=3,
        )

        # QQQ from core takes priority, not duplicated from scout
        assert result["full_watchlist"] == ["SPY", "QQQ", "AAPL"]
