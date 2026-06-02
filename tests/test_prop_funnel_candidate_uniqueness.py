"""
Property-based test for FunnelCandidate uniqueness per day (Property 10).

For any date and symbol pair, at most one FunnelCandidate record exists.
The unique index on (date, symbol) enforces this at the database level.

Feature: premarket-candidate-funnel

**Validates: Requirements 3.6, 3.7**
"""

from datetime import date, datetime, timezone
import json
import uuid

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from db.schema import Base, FunnelCandidate


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Generate arbitrary trading dates within a reasonable range
st_date = st.dates(min_value=date(2024, 1, 1), max_value=date(2030, 12, 31))

# Generate ticker symbols: 1-5 uppercase letters
st_symbol = st.text(
    alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    min_size=1,
    max_size=5,
)

st_source_run = st.sampled_from(["premarket", "confirmation", "manual_intraday"])
st_selection_mode = st.sampled_from(["chief_scout", "deterministic_fallback"])
st_scout_rank = st.integers(min_value=1, max_value=5)
st_scout_score = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_funnel_candidate(trading_date: date, symbol: str, **overrides) -> dict:
    """Build kwargs for a FunnelCandidate row."""
    defaults = {
        "candidate_id": str(uuid.uuid4()),
        "date": trading_date,
        "symbol": symbol,
        "discovered_at": datetime.now(timezone.utc),
        "source_run": "premarket",
        "selection_mode": "chief_scout",
        "scout_rank": 1,
        "scout_score": 75.0,
        "direction_bias": "bullish",
        "catalyst_evidence": json.dumps({"event": "earnings"}),
        "selection_reason": "Strong catalyst",
        "primary_risk": "Sector rotation",
        "sector_context": json.dumps({"sector": "technology"}),
        "stage_status": "awaiting_research",
        "stage_decisions": json.dumps([{
            "agent": "scout",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": "promoted",
            "reasoning": "High composite score",
            "evidence": {},
            "next_stage": "awaiting_research",
        }]),
    }
    defaults.update(overrides)
    return defaults


def _fresh_session():
    """Create a fresh in-memory SQLite database session with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


# ---------------------------------------------------------------------------
# Property 10: Candidate uniqueness per day
# Feature: premarket-candidate-funnel
# ---------------------------------------------------------------------------


class TestProperty10CandidateUniquenessPerDay:
    """
    For any date and symbol pair, at most one FunnelCandidate record exists.
    The unique index on (date, symbol) enforces this at the database level.

    **Validates: Requirements 3.6, 3.7**
    """

    @given(
        trading_date=st_date,
        symbol=st_symbol,
        source_run_1=st_source_run,
        source_run_2=st_source_run,
        selection_mode_1=st_selection_mode,
        selection_mode_2=st_selection_mode,
        scout_rank_1=st_scout_rank,
        scout_rank_2=st_scout_rank,
        scout_score_1=st_scout_score,
        scout_score_2=st_scout_score,
    )
    @settings(max_examples=100)
    def test_duplicate_date_symbol_raises_integrity_error(
        self,
        trading_date,
        symbol,
        source_run_1,
        source_run_2,
        selection_mode_1,
        selection_mode_2,
        scout_rank_1,
        scout_rank_2,
        scout_score_1,
        scout_score_2,
    ):
        """Inserting two FunnelCandidate records with the same (date, symbol) raises IntegrityError."""
        session = _fresh_session()

        try:
            # First insert should succeed
            row1_kwargs = _make_funnel_candidate(
                trading_date,
                symbol,
                source_run=source_run_1,
                selection_mode=selection_mode_1,
                scout_rank=scout_rank_1,
                scout_score=scout_score_1,
            )
            row1 = FunnelCandidate(**row1_kwargs)
            session.add(row1)
            session.commit()

            # Second insert with same (date, symbol) but different other fields must fail
            row2_kwargs = _make_funnel_candidate(
                trading_date,
                symbol,
                source_run=source_run_2,
                selection_mode=selection_mode_2,
                scout_rank=scout_rank_2,
                scout_score=scout_score_2,
            )
            row2 = FunnelCandidate(**row2_kwargs)
            session.add(row2)

            try:
                session.commit()
                # If we reach here, the unique constraint was NOT enforced
                assert False, (
                    f"Expected IntegrityError for duplicate (date={trading_date}, symbol={symbol}) "
                    f"but commit succeeded"
                )
            except IntegrityError:
                # Expected: unique index on (date, symbol) prevents duplicates
                session.rollback()
        finally:
            session.close()
