"""
Property-based test for PM pool only pm_eligible (Property 5).

For any call to get_pm_eligible_candidates(), every returned symbol has
stage_status=pm_eligible, expired=False, date=today. No other stage_status appears.

Feature: premarket-candidate-funnel

**Validates: Requirements 9.1**
"""

from datetime import date, datetime, timedelta, timezone
import json
import uuid

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch
from zoneinfo import ZoneInfo

from db.schema import Base, FunnelCandidate
from utils.funnel_pool import get_pm_eligible_candidates


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

_NY_TZ = ZoneInfo("America/New_York")

# All valid stage_status values from the state machine
ALL_STAGE_STATUSES = [
    "awaiting_research",
    "awaiting_analysis",
    "awaiting_confirmation",
    "pm_eligible",
    "executed",
    "expired",
    "rejected_research",
    "rejected_analysis",
    "rejected_confirmation",
]

st_stage_status = st.sampled_from(ALL_STAGE_STATUSES)
st_expired = st.booleans()
st_source_run = st.sampled_from(["premarket", "confirmation", "manual_intraday"])
st_selection_mode = st.sampled_from(["chief_scout", "deterministic_fallback"])
st_max_handoff = st.integers(min_value=1, max_value=10)

# Generate ticker symbols: 1-5 uppercase letters
st_symbol = st.text(
    alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    min_size=1,
    max_size=5,
)

# Strategy for a single candidate's properties
st_candidate_props = st.fixed_dictionaries({
    "stage_status": st_stage_status,
    "expired": st_expired,
    "is_today": st.booleans(),  # Whether the candidate is for today or yesterday
})

# Strategy for a list of candidates (1 to 20)
st_candidates = st.lists(st_candidate_props, min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_engine_and_tables():
    """Create a fresh in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _make_confirmation_decision(timestamp: str) -> dict:
    """Create a confirmation promoted stage decision."""
    return {
        "agent": "confirmation",
        "timestamp": timestamp,
        "decision": "promoted",
        "reasoning": "All checks passed",
        "evidence": {"volume_confirmed": True, "price_behavior_ok": True},
        "next_stage": "pm_eligible",
    }


def _insert_candidate(
    session,
    symbol: str,
    candidate_date: date,
    stage_status: str,
    expired: bool,
    scout_rank: int = 1,
    scout_score: float = 75.0,
) -> FunnelCandidate:
    """Insert a FunnelCandidate row with given properties."""
    confirmation_ts = datetime.now(timezone.utc).isoformat()
    stage_decisions = json.dumps([
        {
            "agent": "scout",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": "promoted",
            "reasoning": "Test candidate",
            "evidence": {},
            "next_stage": "awaiting_research",
        },
        _make_confirmation_decision(confirmation_ts),
    ])

    candidate = FunnelCandidate(
        candidate_id=str(uuid.uuid4()),
        date=candidate_date,
        symbol=symbol,
        discovered_at=datetime.now(timezone.utc),
        source_run="premarket",
        selection_mode="chief_scout",
        scout_rank=scout_rank,
        scout_score=scout_score,
        direction_bias="bullish",
        catalyst_evidence=json.dumps({"event": "earnings beat"}),
        selection_reason="Strong momentum",
        primary_risk="Sector rotation",
        sector_context=json.dumps({"sector": "technology"}),
        stage_status=stage_status,
        stage_decisions=stage_decisions,
        expired=expired,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Property 5: PM pool contains only pm_eligible candidates
# Feature: premarket-candidate-funnel
# ---------------------------------------------------------------------------


class TestProperty5PmPoolOnlyPmEligible:
    """
    For any call to get_pm_eligible_candidates(), every returned symbol has
    stage_status=pm_eligible, expired=False, date=today. No other stage_status appears.

    **Validates: Requirements 9.1**
    """

    @given(
        candidates_props=st_candidates,
        max_handoff=st_max_handoff,
    )
    @settings(max_examples=100)
    def test_returned_symbols_all_pm_eligible_not_expired_today(
        self,
        candidates_props: list[dict],
        max_handoff: int,
    ):
        """
        Given a database populated with candidates having various stage_statuses,
        expired flags, and dates, get_pm_eligible_candidates() returns only
        symbols where stage_status='pm_eligible', expired=False, and date=today.
        """
        engine = _create_engine_and_tables()
        Session = sessionmaker(bind=engine)
        session = Session()

        today = datetime.now(_NY_TZ).date()
        yesterday = today - timedelta(days=1)

        # Track which symbols should be returned
        expected_eligible_symbols = []

        try:
            for i, props in enumerate(candidates_props):
                symbol = f"T{i:03d}"
                candidate_date = today if props["is_today"] else yesterday
                stage_status = props["stage_status"]
                expired = props["expired"]

                _insert_candidate(
                    session=session,
                    symbol=symbol,
                    candidate_date=candidate_date,
                    stage_status=stage_status,
                    expired=expired,
                    scout_rank=i + 1,
                    scout_score=90.0 - i,
                )

                # A symbol should be returned only if ALL three conditions hold:
                # date == today, stage_status == "pm_eligible", expired == False
                if (
                    candidate_date == today
                    and stage_status == "pm_eligible"
                    and expired is False
                ):
                    expected_eligible_symbols.append(symbol)

            session.commit()

            # Call the function under test
            result = get_pm_eligible_candidates(engine, max_handoff=max_handoff)

            # Property: every returned symbol must be pm_eligible, not expired, today
            # Verify by checking against the database
            for symbol in result:
                candidate = (
                    session.query(FunnelCandidate)
                    .filter(
                        FunnelCandidate.symbol == symbol,
                        FunnelCandidate.date == today,
                    )
                    .first()
                )
                assert candidate is not None, (
                    f"Returned symbol '{symbol}' not found in database for today"
                )
                assert candidate.stage_status == "pm_eligible", (
                    f"Symbol '{symbol}' has stage_status='{candidate.stage_status}' "
                    f"but should be 'pm_eligible'"
                )
                assert candidate.expired is False, (
                    f"Symbol '{symbol}' has expired=True but should be False"
                )
                assert candidate.date == today, (
                    f"Symbol '{symbol}' has date={candidate.date} but should be {today}"
                )

            # Property: no eligible candidate is missed (up to max_handoff)
            assert set(result) <= set(expected_eligible_symbols), (
                f"Result contains symbols not in expected eligible set. "
                f"Result: {result}, Expected eligible: {expected_eligible_symbols}"
            )

            # The result should be at most max_handoff in size
            assert len(result) <= max_handoff, (
                f"Result has {len(result)} symbols but max_handoff is {max_handoff}"
            )

            # If fewer eligible candidates exist than max_handoff, all should be returned
            if len(expected_eligible_symbols) <= max_handoff:
                assert set(result) == set(expected_eligible_symbols), (
                    f"Expected all {len(expected_eligible_symbols)} eligible candidates "
                    f"to be returned when count <= max_handoff={max_handoff}. "
                    f"Got: {result}, Expected: {expected_eligible_symbols}"
                )

        finally:
            session.close()

    @given(
        non_eligible_statuses=st.lists(
            st.sampled_from([
                s for s in ALL_STAGE_STATUSES if s != "pm_eligible"
            ]),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_non_pm_eligible_never_returned(
        self,
        non_eligible_statuses: list[str],
    ):
        """
        When no candidates have stage_status='pm_eligible', the result is empty
        regardless of how many other-status candidates exist for today.
        """
        engine = _create_engine_and_tables()
        Session = sessionmaker(bind=engine)
        session = Session()

        today = datetime.now(_NY_TZ).date()

        try:
            for i, status in enumerate(non_eligible_statuses):
                _insert_candidate(
                    session=session,
                    symbol=f"N{i:03d}",
                    candidate_date=today,
                    stage_status=status,
                    expired=False,
                    scout_rank=i + 1,
                    scout_score=85.0 - i,
                )
            session.commit()

            result = get_pm_eligible_candidates(engine, max_handoff=10)

            # Property: no non-pm_eligible candidate is ever returned
            assert result == [], (
                f"Expected empty result when no pm_eligible candidates exist, "
                f"but got: {result} (statuses were: {non_eligible_statuses})"
            )

        finally:
            session.close()

    @given(
        num_eligible=st.integers(min_value=1, max_value=8),
        num_expired_eligible=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50)
    def test_expired_pm_eligible_never_returned(
        self,
        num_eligible: int,
        num_expired_eligible: int,
    ):
        """
        Candidates with stage_status='pm_eligible' but expired=True are never returned.
        Only non-expired pm_eligible candidates appear in the result.
        """
        engine = _create_engine_and_tables()
        Session = sessionmaker(bind=engine)
        session = Session()

        today = datetime.now(_NY_TZ).date()

        try:
            # Insert non-expired pm_eligible candidates (should be returned)
            for i in range(num_eligible):
                _insert_candidate(
                    session=session,
                    symbol=f"E{i:03d}",
                    candidate_date=today,
                    stage_status="pm_eligible",
                    expired=False,
                    scout_rank=i + 1,
                    scout_score=90.0 - i,
                )

            # Insert expired pm_eligible candidates (should NOT be returned)
            for i in range(num_expired_eligible):
                _insert_candidate(
                    session=session,
                    symbol=f"X{i:03d}",
                    candidate_date=today,
                    stage_status="pm_eligible",
                    expired=True,
                    scout_rank=num_eligible + i + 1,
                    scout_score=70.0 - i,
                )

            session.commit()

            result = get_pm_eligible_candidates(engine, max_handoff=20)

            # Property: no expired candidate is returned
            expired_symbols = {f"X{i:03d}" for i in range(num_expired_eligible)}
            for symbol in result:
                assert symbol not in expired_symbols, (
                    f"Expired pm_eligible symbol '{symbol}' was returned"
                )

            # All non-expired pm_eligible should be returned (max_handoff=20 is large enough)
            non_expired_symbols = {f"E{i:03d}" for i in range(num_eligible)}
            assert set(result) == non_expired_symbols, (
                f"Expected all non-expired pm_eligible symbols {non_expired_symbols} "
                f"but got {set(result)}"
            )

        finally:
            session.close()
