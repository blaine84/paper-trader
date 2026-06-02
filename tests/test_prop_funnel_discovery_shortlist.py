"""
Property-based test for discovery shortlist bounded (Property 1).

For any discovery run with any number of enabled sectors and finalists,
persisted FunnelCandidate rows never exceed max_discovery_shortlist ceiling.

Feature: premarket-candidate-funnel

**Validates: Requirements 1.6**
"""

from datetime import date, datetime, timezone
import json
import uuid

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, FunnelCandidate
from utils.sector_scout_models import CandidateRow
from utils.funnel_discovery import persist_discovery_candidates


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# max_shortlist ceiling: 1 to 20 (default is 5 but test wider range)
st_max_shortlist = st.integers(min_value=1, max_value=20)

# Number of finalists: 0 to 50 (can exceed ceiling by a lot)
st_num_finalists = st.integers(min_value=0, max_value=50)

st_source_run = st.sampled_from(["premarket", "confirmation", "manual_intraday"])
st_selection_mode = st.sampled_from(["chief_scout", "deterministic_fallback"])

# Generate ticker symbols: 1-5 uppercase letters
st_symbol = st.text(
    alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    min_size=1,
    max_size=5,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate_row(symbol: str, score: float, sector: str = "technology") -> CandidateRow:
    """Create a minimal CandidateRow for testing."""
    return CandidateRow(
        symbol=symbol,
        sector=sector,
        sector_name=f"{sector.title()} Sector",
        current_price=100.0,
        prev_close=95.0,
        move_pct=5.26,
        current_volume=1_000_000,
        average_volume=500_000,
        relative_volume=2.0,
        dollar_volume=100_000_000,
        news_headlines=[{"title": "Test news", "age_minutes": 30}],
        news_freshness_minutes=30.0,
        sector_etf="XLK",
        sector_etf_move_pct=1.5,
        sector_confirmed=True,
        hard_gate_passed=True,
        scout_score=score,
        collected_at=datetime.now(timezone.utc).isoformat(),
        run_type="premarket",
    )


def _make_unique_finalists(num: int) -> list[CandidateRow]:
    """Generate a list of CandidateRow objects with unique symbols."""
    finalists = []
    for i in range(num):
        # Generate unique symbols by combining letters with index
        symbol = f"S{i:03d}"[:5]  # e.g. S000, S001, ..., S049
        score = 80.0 - (i * 0.5)  # Descending scores
        finalists.append(_make_candidate_row(symbol, score))
    return finalists


def _create_engine_and_tables():
    """Create a fresh in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _count_funnel_candidates(engine) -> int:
    """Count all FunnelCandidate rows in the database."""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        return session.query(FunnelCandidate).count()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Property 1: Discovery shortlist is bounded
# Feature: premarket-candidate-funnel
# ---------------------------------------------------------------------------


class TestProperty1DiscoveryShortlistBounded:
    """
    For any discovery run with any number of enabled sectors and finalists,
    persisted FunnelCandidate rows never exceed max_discovery_shortlist ceiling.

    **Validates: Requirements 1.6**
    """

    @given(
        num_finalists=st_num_finalists,
        max_shortlist=st_max_shortlist,
        selection_mode=st_selection_mode,
        source_run=st_source_run,
    )
    @settings(max_examples=100)
    def test_persisted_candidates_never_exceed_max_shortlist(
        self,
        num_finalists: int,
        max_shortlist: int,
        selection_mode: str,
        source_run: str,
    ):
        """
        persist_discovery_candidates() with any number of finalists
        never persists more rows than max_shortlist ceiling.
        """
        engine = _create_engine_and_tables()

        # Generate finalists (may exceed ceiling)
        finalists = _make_unique_finalists(num_finalists)

        # Call persist_discovery_candidates with the ceiling
        result = persist_discovery_candidates(
            engine=engine,
            finalists=finalists,
            selection_mode=selection_mode,
            source_run=source_run,
            max_shortlist=max_shortlist,
        )

        # Property: returned count never exceeds ceiling
        assert len(result) <= max_shortlist, (
            f"persist_discovery_candidates returned {len(result)} candidates "
            f"but max_shortlist ceiling is {max_shortlist}"
        )

        # Property: persisted rows in DB never exceed ceiling
        db_count = _count_funnel_candidates(engine)
        assert db_count <= max_shortlist, (
            f"Database contains {db_count} FunnelCandidate rows "
            f"but max_shortlist ceiling is {max_shortlist}"
        )

        # Additional invariant: returned count equals min(num_finalists, max_shortlist)
        expected = min(num_finalists, max_shortlist)
        assert len(result) == expected, (
            f"Expected {expected} candidates (min({num_finalists}, {max_shortlist})) "
            f"but got {len(result)}"
        )
