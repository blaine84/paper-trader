"""
Property-based tests for AlertIntentStore.query_by_lifecycle() using Hypothesis.

Property 15: Lifecycle query returns correct filtered results.

Validates that query_by_lifecycle():
1. Always returns a list (never raises)
2. All returned items match the provided filters
3. Results are ordered by first_seen_at DESC
4. Result set size is <= 1000
5. Empty results when no matches exist returns [] not an error

**Validates: Requirements 9.4, 9.5**
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from utils.alert_intent_store import AlertIntentStore


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

SYMBOLS = ["NVDA", "AAPL", "TSLA"]
ALERT_TYPES = ["entry_alert", "breakout", "rapid_move", "target_hit"]
DISPATCH_STATUSES = ["pending", "dispatched", "expired", "consumed"]

st_symbol = st.sampled_from(SYMBOLS)
st_alert_type = st.sampled_from(ALERT_TYPES)
st_dispatch_status = st.sampled_from(DISPATCH_STATUSES)

# Generate a random timestamp within a fixed 7-day window for test determinism
_BASE_TIME = datetime(2025, 1, 15, 0, 0, 0)

st_offset_minutes = st.integers(min_value=0, max_value=7 * 24 * 60)


@st.composite
def st_intent_data(draw):
    """Generate randomized intent data for insertion."""
    symbol = draw(st_symbol)
    alert_type = draw(st_alert_type)
    dispatch_status = draw(st_dispatch_status)
    offset = draw(st_offset_minutes)
    first_seen_at = _BASE_TIME + timedelta(minutes=offset)

    return {
        "symbol": symbol,
        "alert_type": alert_type,
        "dispatch_status": dispatch_status,
        "first_seen_at": first_seen_at,
        "dedupe_key": str(uuid.uuid4()),
    }


@st.composite
def st_filter_combo(draw):
    """Generate a random filter combination (any subset of filters may be None)."""
    use_status = draw(st.booleans())
    use_alert_type = draw(st.booleans())
    use_symbol = draw(st.booleans())
    use_start_time = draw(st.booleans())
    use_end_time = draw(st.booleans())

    filters = {}

    if use_status:
        # Pick 1-3 statuses from the pool
        filters["status_list"] = draw(
            st.lists(st_dispatch_status, min_size=1, max_size=3, unique=True)
        )

    if use_alert_type:
        filters["alert_type"] = draw(st_alert_type)

    if use_symbol:
        filters["symbol"] = draw(st_symbol)

    if use_start_time:
        offset = draw(st.integers(min_value=0, max_value=7 * 24 * 60))
        filters["start_time"] = (_BASE_TIME + timedelta(minutes=offset)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )

    if use_end_time:
        offset = draw(st.integers(min_value=0, max_value=7 * 24 * 60))
        filters["end_time"] = (_BASE_TIME + timedelta(minutes=offset)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )

    return filters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ISO_FMT = "%Y-%m-%dT%H:%M:%S.000Z"


def _create_store():
    """Create an in-memory SQLite engine with schema and return a store."""
    engine = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(engine)
    return AlertIntentStore(engine)


def _insert_intent(store, intent_data: dict):
    """Insert a test intent and set the desired dispatch_status."""
    data = {
        "symbol": intent_data["symbol"],
        "alert_type": intent_data["alert_type"],
        "direction": "long",
        "trigger_price": "100.00",
        "source_level": None,
        "urgency": "medium",
        "reason": None,
        "dedupe_key": intent_data["dedupe_key"],
        "filter_status": "passed",
        "first_seen_at": intent_data["first_seen_at"].strftime(_ISO_FMT),
        "last_seen_at": intent_data["first_seen_at"].strftime(_ISO_FMT),
        "expiration_at": "2025-01-30T23:00:00.000Z",
    }
    intent = store.record_or_update_intent(data)

    # If we need a non-pending status, update directly
    target_status = intent_data["dispatch_status"]
    if target_status != "pending":
        with store._engine.begin() as conn:
            conn.execute(
                text("UPDATE alert_intents SET dispatch_status = :status WHERE id = :id"),
                {"status": target_status, "id": intent.id},
            )

    return intent


def _matches_filters(intent_data: dict, filters: dict) -> bool:
    """Check if an intent would match the given filter combination."""
    if "status_list" in filters:
        if intent_data["dispatch_status"] not in filters["status_list"]:
            return False

    if "alert_type" in filters:
        if intent_data["alert_type"] != filters["alert_type"]:
            return False

    if "symbol" in filters:
        if intent_data["symbol"] != filters["symbol"]:
            return False

    first_seen_str = intent_data["first_seen_at"].strftime(_ISO_FMT)

    if "start_time" in filters:
        if first_seen_str < filters["start_time"]:
            return False

    if "end_time" in filters:
        if first_seen_str > filters["end_time"]:
            return False

    return True


# ---------------------------------------------------------------------------
# Property 15: Lifecycle query returns correct filtered results
# ---------------------------------------------------------------------------


class TestProperty15LifecycleQueryFiltering:
    """
    Property 15: Lifecycle query returns correct filtered results.

    Given a set of inserted intents and a random filter combination,
    query_by_lifecycle() always:
    1. Returns a list (never raises)
    2. All returned items match all provided filters
    3. Results are ordered by first_seen_at DESC
    4. Result set size is <= 1000
    5. Empty results when no matches returns [] not an error

    **Validates: Requirements 9.4, 9.5**
    """

    @given(
        intents=st.lists(st_intent_data(), min_size=0, max_size=20),
        filters=st_filter_combo(),
    )
    @settings(max_examples=50)
    def test_query_returns_list_and_respects_filters(
        self, intents: list[dict], filters: dict
    ):
        """query_by_lifecycle always returns a list with items matching all filters."""
        store = _create_store()

        # Insert all generated intents
        for intent_data in intents:
            _insert_intent(store, intent_data)

        # Execute query with the random filter combination
        result = store.query_by_lifecycle(**filters)

        # 1. Always returns a list
        assert isinstance(result, list)

        # 4. Result set size <= 1000
        assert len(result) <= 1000

        # 2. All returned items match all provided filters
        for item in result:
            if "status_list" in filters:
                assert item.dispatch_status in filters["status_list"], (
                    f"Item dispatch_status={item.dispatch_status} not in "
                    f"filter status_list={filters['status_list']}"
                )

            if "alert_type" in filters:
                assert item.alert_type == filters["alert_type"], (
                    f"Item alert_type={item.alert_type} != "
                    f"filter alert_type={filters['alert_type']}"
                )

            if "symbol" in filters:
                assert item.symbol == filters["symbol"], (
                    f"Item symbol={item.symbol} != filter symbol={filters['symbol']}"
                )

            if "start_time" in filters:
                item_ts = item.first_seen_at.strftime(_ISO_FMT)
                assert item_ts >= filters["start_time"], (
                    f"Item first_seen_at={item_ts} < start_time={filters['start_time']}"
                )

            if "end_time" in filters:
                item_ts = item.first_seen_at.strftime(_ISO_FMT)
                assert item_ts <= filters["end_time"], (
                    f"Item first_seen_at={item_ts} > end_time={filters['end_time']}"
                )

        # 3. Results are ordered by first_seen_at DESC
        for i in range(len(result) - 1):
            assert result[i].first_seen_at >= result[i + 1].first_seen_at, (
                f"Results not sorted DESC: index {i} has "
                f"first_seen_at={result[i].first_seen_at}, "
                f"index {i+1} has first_seen_at={result[i+1].first_seen_at}"
            )

    @given(
        intents=st.lists(st_intent_data(), min_size=1, max_size=15),
        filters=st_filter_combo(),
    )
    @settings(max_examples=50)
    def test_query_result_count_matches_expected(
        self, intents: list[dict], filters: dict
    ):
        """The number of results matches the count of intents matching filters."""
        store = _create_store()

        # Insert all generated intents
        for intent_data in intents:
            _insert_intent(store, intent_data)

        # Execute query
        result = store.query_by_lifecycle(**filters)

        # Count how many inserted intents should match
        expected_count = sum(
            1 for intent_data in intents if _matches_filters(intent_data, filters)
        )

        # Result count should match (capped at 1000)
        assert len(result) == min(expected_count, 1000), (
            f"Expected {min(expected_count, 1000)} results, got {len(result)}. "
            f"Filters: {filters}"
        )

    @given(filters=st_filter_combo())
    @settings(max_examples=50)
    def test_empty_db_returns_empty_list(self, filters: dict):
        """Query on empty DB returns [] regardless of filters (never raises)."""
        store = _create_store()

        result = store.query_by_lifecycle(**filters)

        assert result == []
        assert isinstance(result, list)

    @given(
        intents=st.lists(st_intent_data(), min_size=1, max_size=10),
    )
    @settings(max_examples=50)
    def test_no_matching_filters_returns_empty_list(self, intents: list[dict]):
        """When filters exclude all intents, returns [] not an error."""
        store = _create_store()

        for intent_data in intents:
            _insert_intent(store, intent_data)

        # Use a symbol that doesn't exist in SYMBOLS list
        result = store.query_by_lifecycle(symbol="NONEXISTENT_SYMBOL_XYZ")

        assert result == []
        assert isinstance(result, list)
