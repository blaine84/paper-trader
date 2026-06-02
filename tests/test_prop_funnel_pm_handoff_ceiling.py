"""
Property-based test for PM handoff ceiling (Property 11).

For any set of pm_eligible candidates, get_pm_eligible_candidates(max_handoff=N)
returns at most N symbols in deterministic order.

Feature: premarket-candidate-funnel

**Validates: Requirements 9.2, 9.6**
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, FunnelCandidate


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# max_handoff ceiling: 1 to 15
st_max_handoff = st.integers(min_value=1, max_value=15)

# Number of pm_eligible candidates: 0 to 30 (can greatly exceed ceiling)
st_num_eligible = st.integers(min_value=0, max_value=30)

_NY_TZ = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_engine_and_tables():
    """Create a fresh in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _get_today_ny() -> date:
    """Get today's date in New York timezone (matches production logic)."""
    return datetime.now(_NY_TZ).date()


def _make_pm_eligible_candidate(engine, index: int, today: date) -> FunnelCandidate:
    """Create and persist a FunnelCandidate in pm_eligible status."""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # Vary confirmation timestamps so ordering is testable
        confirmation_ts = datetime(
            today.year, today.month, today.day,
            9, 35, index % 60, tzinfo=timezone.utc
        ).isoformat()

        stage_decisions = json.dumps([
            {
                "agent": "scout",
                "timestamp": datetime(
                    today.year, today.month, today.day, 6, 0, 0, tzinfo=timezone.utc
                ).isoformat(),
                "decision": "promoted",
                "reasoning": "Discovery promotion",
                "evidence": {},
                "next_stage": "awaiting_research",
            },
            {
                "agent": "researcher",
                "timestamp": datetime(
                    today.year, today.month, today.day, 6, 30, 0, tzinfo=timezone.utc
                ).isoformat(),
                "decision": "promoted",
                "reasoning": "Catalyst validated",
                "evidence": {},
                "next_stage": "awaiting_analysis",
            },
            {
                "agent": "analyst",
                "timestamp": datetime(
                    today.year, today.month, today.day, 7, 15, 0, tzinfo=timezone.utc
                ).isoformat(),
                "decision": "promoted",
                "reasoning": "Setup classified",
                "evidence": {},
                "next_stage": "awaiting_confirmation",
            },
            {
                "agent": "confirmation",
                "timestamp": confirmation_ts,
                "decision": "promoted",
                "reasoning": "Confirmed at open",
                "evidence": {},
                "next_stage": "pm_eligible",
            },
        ])

        candidate = FunnelCandidate(
            candidate_id=str(uuid.uuid4()),
            date=today,
            symbol=f"S{index:03d}"[:5],
            discovered_at=datetime.now(timezone.utc),
            source_run="premarket",
            selection_mode="deterministic_fallback",
            scout_rank=index + 1,
            scout_score=90.0 - index * 0.5,
            direction_bias="bullish",
            catalyst_evidence=json.dumps({"headline": f"Catalyst {index}"}),
            selection_reason=f"Candidate {index} selection reason",
            primary_risk="Market risk",
            sector_context=json.dumps({"sector": "technology"}),
            preliminary_setup_type="momentum_breakout",
            authoritative_setup_type="momentum_breakout",
            stage_status="pm_eligible",
            stage_decisions=stage_decisions,
            expired=False,
        )
        session.add(candidate)
        session.commit()
        session.refresh(candidate)
        return candidate
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Property 11: PM handoff ceiling enforced
# Feature: premarket-candidate-funnel
# ---------------------------------------------------------------------------


class TestProperty11PMHandoffCeiling:
    """
    For any set of pm_eligible candidates, get_pm_eligible_candidates(max_handoff=N)
    returns at most N symbols in deterministic order.

    **Validates: Requirements 9.2, 9.6**
    """

    @given(
        num_eligible=st_num_eligible,
        max_handoff=st_max_handoff,
    )
    @settings(max_examples=100)
    def test_returned_count_never_exceeds_max_handoff(
        self,
        num_eligible: int,
        max_handoff: int,
    ):
        """
        get_pm_eligible_candidates() with any number of pm_eligible candidates
        returns at most max_handoff symbols.

        **Validates: Requirements 9.2, 9.6**
        """
        from utils.funnel_pool import get_pm_eligible_candidates

        engine = _create_engine_and_tables()
        # Use the same NY date logic as production code
        today = _get_today_ny()

        # Create N pm_eligible candidates in the database
        for i in range(num_eligible):
            _make_pm_eligible_candidate(engine, i, today)

        result = get_pm_eligible_candidates(engine, max_handoff=max_handoff)

        # PROPERTY: returned count NEVER exceeds max_handoff
        assert len(result) <= max_handoff, (
            f"get_pm_eligible_candidates returned {len(result)} symbols "
            f"but max_handoff ceiling is {max_handoff} "
            f"(pm_eligible candidates: {num_eligible})"
        )

        # Additional invariant: returned count equals min(num_eligible, max_handoff)
        expected_count = min(num_eligible, max_handoff)
        assert len(result) == expected_count, (
            f"Expected {expected_count} symbols "
            f"(min({num_eligible}, {max_handoff})) "
            f"but got {len(result)}"
        )

        # All returned values are strings (symbols)
        for sym in result:
            assert isinstance(sym, str), f"Expected string symbol, got {type(sym)}"

    @given(
        num_eligible=st.integers(min_value=2, max_value=20),
        max_handoff=st.integers(min_value=1, max_value=15),
    )
    @settings(max_examples=50)
    def test_returned_order_is_deterministic(
        self,
        num_eligible: int,
        max_handoff: int,
    ):
        """
        Calling get_pm_eligible_candidates() multiple times with the same data
        returns results in the same deterministic order.

        **Validates: Requirements 9.2, 9.6**
        """
        from utils.funnel_pool import get_pm_eligible_candidates

        engine = _create_engine_and_tables()
        today = _get_today_ny()

        # Create pm_eligible candidates
        for i in range(num_eligible):
            _make_pm_eligible_candidate(engine, i, today)

        result1 = get_pm_eligible_candidates(engine, max_handoff=max_handoff)
        result2 = get_pm_eligible_candidates(engine, max_handoff=max_handoff)

        # PROPERTY: order is deterministic across calls
        assert result1 == result2, (
            f"Non-deterministic ordering detected:\n"
            f"  Call 1: {result1}\n"
            f"  Call 2: {result2}"
        )
