"""
Property-based tests for PM-side idempotency using Hypothesis.

Property 8: PM idempotency — duplicate (alert_intent_id, profile_id) produces no-op.
Validates that claim_alert_for_processing uses atomic INSERT-or-reject via
UNIQUE constraint on (alert_intent_id, profile_id) to guarantee idempotency.

**Validates: Requirements 5.1, 5.2, 5.3**
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine, text

from agents.portfolio_manager import claim_alert_for_processing
from utils.alert_dispatch_schema import init_alert_dispatch_schema


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Alert intent IDs — unique string identifiers
alert_intent_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=5,
    max_size=30,
)

# Symbols — uppercase stock tickers
symbol_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu",)),
    min_size=1,
    max_size=6,
)

# Alert types from the known set
alert_type_strategy = st.sampled_from(["entry_alert", "breakout", "rapid_move", "target_hit"])

# Profile IDs
profile_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=3,
    max_size=20,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create a fresh in-memory SQLite database with alert dispatch schema."""
    eng = create_engine("sqlite:///:memory:")
    init_alert_dispatch_schema(eng)
    return eng


# ---------------------------------------------------------------------------
# Property 8: PM idempotency — duplicate (alert_intent_id, profile_id) produces no-op
# **Validates: Requirements 5.1, 5.2, 5.3**
# ---------------------------------------------------------------------------


class TestProperty8PMIdempotency:
    """
    Property 8: PM idempotency — duplicate (alert_intent_id, profile_id) produces no-op.

    1. First call to claim_alert_for_processing returns None (proceed)
    2. Second call with SAME (alert_intent_id, profile_id) returns {"outcome": "duplicate_noop"}
    3. Different profile_ids can independently claim the same alert_intent_id (both return None)
    4. No duplicate rows created in pm_alert_claims for the same (intent, profile) pair

    **Validates: Requirements 5.1, 5.2, 5.3**
    """

    @given(
        alert_intent_id=alert_intent_id_strategy,
        symbol=symbol_strategy,
        alert_type=alert_type_strategy,
        profile_id=profile_id_strategy,
    )
    @settings(max_examples=50)
    def test_first_claim_returns_none(
        self, alert_intent_id: str, symbol: str, alert_type: str, profile_id: str
    ):
        """First call to claim_alert_for_processing returns None (proceed)."""
        assume(len(alert_intent_id.strip()) > 0)
        assume(len(symbol.strip()) > 0)
        assume(len(profile_id.strip()) > 0)

        eng = create_engine("sqlite:///:memory:")
        init_alert_dispatch_schema(eng)

        result = claim_alert_for_processing(eng, alert_intent_id, symbol, alert_type, profile_id)

        assert result is None, (
            f"Expected None (proceed) on first claim, got {result} "
            f"for alert_intent_id={alert_intent_id!r}, profile_id={profile_id!r}"
        )

    @given(
        alert_intent_id=alert_intent_id_strategy,
        symbol=symbol_strategy,
        alert_type=alert_type_strategy,
        profile_id=profile_id_strategy,
    )
    @settings(max_examples=50)
    def test_second_claim_same_pair_returns_duplicate_noop(
        self, alert_intent_id: str, symbol: str, alert_type: str, profile_id: str
    ):
        """Second call with SAME (alert_intent_id, profile_id) returns duplicate_noop."""
        assume(len(alert_intent_id.strip()) > 0)
        assume(len(symbol.strip()) > 0)
        assume(len(profile_id.strip()) > 0)

        eng = create_engine("sqlite:///:memory:")
        init_alert_dispatch_schema(eng)

        # First claim — should succeed
        first_result = claim_alert_for_processing(eng, alert_intent_id, symbol, alert_type, profile_id)
        assert first_result is None

        # Second claim — same (alert_intent_id, profile_id) must produce duplicate_noop
        second_result = claim_alert_for_processing(eng, alert_intent_id, symbol, alert_type, profile_id)

        assert second_result is not None, (
            f"Expected duplicate_noop dict on second claim, got None "
            f"for alert_intent_id={alert_intent_id!r}, profile_id={profile_id!r}"
        )
        assert second_result["outcome"] == "duplicate_noop", (
            f"Expected outcome='duplicate_noop', got {second_result['outcome']!r} "
            f"for alert_intent_id={alert_intent_id!r}, profile_id={profile_id!r}"
        )

    @given(
        alert_intent_id=alert_intent_id_strategy,
        symbol=symbol_strategy,
        alert_type=alert_type_strategy,
        profile_id_a=profile_id_strategy,
        profile_id_b=profile_id_strategy,
    )
    @settings(max_examples=50)
    def test_different_profiles_independently_claim_same_intent(
        self,
        alert_intent_id: str,
        symbol: str,
        alert_type: str,
        profile_id_a: str,
        profile_id_b: str,
    ):
        """Different profile_ids can independently claim the same alert_intent_id (both return None)."""
        assume(len(alert_intent_id.strip()) > 0)
        assume(len(symbol.strip()) > 0)
        assume(len(profile_id_a.strip()) > 0)
        assume(len(profile_id_b.strip()) > 0)
        assume(profile_id_a != profile_id_b)

        eng = create_engine("sqlite:///:memory:")
        init_alert_dispatch_schema(eng)

        # Profile A claims first
        result_a = claim_alert_for_processing(eng, alert_intent_id, symbol, alert_type, profile_id_a)
        assert result_a is None, (
            f"Expected None for profile_a={profile_id_a!r} first claim, got {result_a}"
        )

        # Profile B claims same alert_intent_id independently
        result_b = claim_alert_for_processing(eng, alert_intent_id, symbol, alert_type, profile_id_b)
        assert result_b is None, (
            f"Expected None for profile_b={profile_id_b!r} independent claim of same intent, "
            f"got {result_b}. Different profiles must be able to claim independently."
        )

    @given(
        alert_intent_id=alert_intent_id_strategy,
        symbol=symbol_strategy,
        alert_type=alert_type_strategy,
        profile_id=profile_id_strategy,
    )
    @settings(max_examples=50)
    def test_no_duplicate_rows_in_pm_alert_claims(
        self, alert_intent_id: str, symbol: str, alert_type: str, profile_id: str
    ):
        """No duplicate rows created in pm_alert_claims for the same (intent, profile) pair."""
        assume(len(alert_intent_id.strip()) > 0)
        assume(len(symbol.strip()) > 0)
        assume(len(profile_id.strip()) > 0)

        eng = create_engine("sqlite:///:memory:")
        init_alert_dispatch_schema(eng)

        # Claim twice — first succeeds, second is duplicate_noop
        claim_alert_for_processing(eng, alert_intent_id, symbol, alert_type, profile_id)
        claim_alert_for_processing(eng, alert_intent_id, symbol, alert_type, profile_id)

        # Verify exactly 1 row exists for this (alert_intent_id, profile_id) pair
        with eng.connect() as conn:
            row_count = conn.execute(text("""
                SELECT COUNT(*) FROM pm_alert_claims
                WHERE alert_intent_id = :aid AND profile_id = :pid
            """), {"aid": alert_intent_id, "pid": profile_id}).scalar()

        assert row_count == 1, (
            f"Expected exactly 1 row in pm_alert_claims for "
            f"(alert_intent_id={alert_intent_id!r}, profile_id={profile_id!r}), "
            f"got {row_count} rows"
        )
