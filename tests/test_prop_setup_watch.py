"""
Property-based tests for the Setup Watch Layer.

Tests 8 universal invariants using Hypothesis:
1. Maturity score always in [0.0, 1.0]
2. Score invariant to condition list ordering
3. Every permitted transition succeeds; every non-permitted raises
4. Terminal states are absorbing
5. integrity_hash is deterministic
6. expire_elapsed never expires a future-dated watch
7. Condition definition columns are immutable under registry operations
8. MFE >= MAE for any candle series

**Validates: Requirements 3.2, 4.3, 4.12, 11.4, 12.6, 12.10**
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine, text

from db.schema import init_setup_watch_schema
from utils.setup_watch_evaluator import evaluate_maturation_conditions
from utils.setup_watch_outcomes import score_watch_outcome
from utils.setup_watch_registry import (
    ACTIVE_STATES,
    PERMITTED_TRANSITIONS,
    TERMINAL_STATES,
    SetupWatch,
    SetupWatchRegistry,
    SetupWatchRegistryError,
    WatchState,
    compute_watch_integrity_hash,
)

NOW = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

st_weight = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)

st_maturation_type = st.sampled_from([
    "price_zone", "regime_aligned", "catalyst_fresh", "time_window", "key_level_proximity",
])

st_unknown_type = st.sampled_from(["volume_threshold", "spread_acceptable", "foobar_unknown"])

st_condition_type = st.one_of(st_maturation_type, st_unknown_type)

st_price = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)

st_side = st.sampled_from(["BUY", "SHORT"])

st_state = st.sampled_from(list(WatchState))


def st_condition(cond_type=None):
    """Strategy for a single maturation condition dict."""
    if cond_type is None:
        cond_type = st_condition_type
    return st.fixed_dictionaries({
        "type": cond_type,
        "weight": st_weight,
        "params": st.fixed_dictionaries({
            "low": st_price,
            "high": st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        }),
    })


st_condition_list = st.lists(
    st_condition(),
    min_size=1,
    max_size=10,
)

st_known_condition_list = st.lists(
    st_condition(st_maturation_type),
    min_size=1,
    max_size=10,
)


def _market_context(current_price: float = 150.0) -> dict:
    """A market context that satisfies all condition types when prices match."""
    return {
        "current_price": str(current_price),
        "market_regime": "bullish",
        "catalyst_age_minutes": 10,
        "current_et_hour": 10,
        "key_levels": {"support": [str(current_price - 5)], "resistance": [str(current_price + 5)]},
        "held_symbols": [],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_setup_watch_schema(eng)
    return eng


@pytest.fixture
def registry(engine):
    return SetupWatchRegistry(engine)


def _make_watch(**overrides) -> SetupWatch:
    """Build a SetupWatch dataclass with sensible defaults."""
    defaults = dict(
        watch_id=str(uuid.uuid4()),
        profile_id="moderate",
        symbol="AAPL",
        side="BUY",
        setup_type="technical_breakout",
        state=WatchState.WATCHING,
        thesis="Stock approaching key support with strong volume",
        source_type="analyst",
        source_id="signal_123",
        source_cycle_id="cycle_001",
        maturation_conditions_json=json.dumps([
            {"type": "price_zone", "params": {"low": "140", "high": "160"}, "weight": 1.0},
            {"type": "regime_aligned", "params": {"required_regime": "bullish"}, "weight": 1.0},
        ]),
        invalidation_conditions_json=json.dumps([
            {"type": "price_breach", "params": {"level": "130", "direction": "below"}},
        ]),
        last_evaluation_json=None,
        entry_zone_json=None,
        draft_geometry_json=None,
        maturity_score=0.0,
        created_at=NOW,
        updated_at=NOW,
        expires_at=NOW + timedelta(hours=6),
        state_changed_at=None,
        observed_cycles=0,
        ready_at=None,
        ready_reference_price=None,
        terminal_reason=None,
        promoted_cycle_id=None,
        execution_ref_type=None,
        execution_ref_id=None,
        integrity_hash="",
    )
    defaults.update(overrides)
    return SetupWatch(**defaults)


# ---------------------------------------------------------------------------
# Property 1: Any valid condition set yields maturity_score in [0.0, 1.0]
# **Validates: Requirements 4.3**
# ---------------------------------------------------------------------------


@given(conditions=st_condition_list)
@settings(max_examples=200)
def test_maturity_score_bounded(conditions):
    """maturity_score is always in [0.0, 1.0] regardless of condition set."""
    conditions_json = json.dumps(conditions)
    ctx = _market_context()

    score, _ = evaluate_maturation_conditions(conditions_json, ctx)

    assert 0.0 <= score <= 1.0, f"Score {score} is out of bounds [0.0, 1.0]"


# ---------------------------------------------------------------------------
# Property 2: Score is invariant to condition list ordering
# **Validates: Requirements 4.3**
# ---------------------------------------------------------------------------


@given(conditions=st_known_condition_list, data=st.data())
@settings(max_examples=200)
def test_score_invariant_to_ordering(conditions, data):
    """Reordering the condition list does not change the maturity score."""
    assume(len(conditions) >= 2)

    # Generate a permutation via shuffle
    shuffled = data.draw(st.permutations(conditions))
    assume(shuffled != conditions)  # ensure actually different ordering

    ctx = _market_context()

    score_original, _ = evaluate_maturation_conditions(json.dumps(conditions), ctx)
    score_shuffled, _ = evaluate_maturation_conditions(json.dumps(shuffled), ctx)

    assert abs(score_original - score_shuffled) < 1e-10, (
        f"Score changed with reordering: {score_original} vs {score_shuffled}"
    )


# ---------------------------------------------------------------------------
# Property 3: Every pair in PERMITTED_TRANSITIONS succeeds; every pair outside raises
# **Validates: Requirements 3.2**
# ---------------------------------------------------------------------------


def test_permitted_transitions_exhaustive(engine):
    """Every permitted transition succeeds via CAS; every non-permitted raises."""
    registry = SetupWatchRegistry(engine)
    all_states = list(WatchState)

    # Test permitted transitions
    for from_state, to_state in PERMITTED_TRANSITIONS:
        watch = _make_watch(state=WatchState.WATCHING)
        watch_id = registry.create_watch(watch)

        # Manually set to from_state if not WATCHING (create always sets WATCHING)
        if from_state != WatchState.WATCHING:
            with engine.connect() as conn:
                conn.execute(
                    text("UPDATE setup_watches SET state = :state WHERE watch_id = :wid"),
                    {"state": from_state.value, "wid": watch_id},
                )
                conn.commit()

        # The transition should succeed
        kwargs = {}
        if to_state in TERMINAL_STATES:
            kwargs["terminal_reason"] = "test_reason"
        if to_state == WatchState.ORDERED:
            kwargs["execution_ref_type"] = "trade"
            kwargs["execution_ref_id"] = "trade_123"
        if to_state == WatchState.PROMOTED:
            kwargs["promoted_cycle_id"] = "cycle_x"
        if to_state == WatchState.READY:
            kwargs["ready_reference_price"] = 150.0

        registry.transition_state(
            watch_id, from_state, to_state, **kwargs
        )

        with engine.connect() as conn:
            actual = conn.execute(
                text("SELECT state FROM setup_watches WHERE watch_id = :wid"),
                {"wid": watch_id},
            ).scalar()
        assert actual == to_state.value

    # Test non-permitted transitions raise
    for from_state in all_states:
        for to_state in all_states:
            if (from_state, to_state) in PERMITTED_TRANSITIONS:
                continue
            if from_state == to_state:
                continue

            watch = _make_watch(state=WatchState.WATCHING)
            watch_id = registry.create_watch(watch)

            if from_state != WatchState.WATCHING:
                with engine.connect() as conn:
                    conn.execute(
                        text("UPDATE setup_watches SET state = :state WHERE watch_id = :wid"),
                        {"state": from_state.value, "wid": watch_id},
                    )
                    conn.commit()

            with pytest.raises(SetupWatchRegistryError):
                registry.transition_state(
                    watch_id, from_state, to_state,
                    terminal_reason="test" if to_state in TERMINAL_STATES else None,
                    execution_ref_type="trade" if to_state == WatchState.ORDERED else None,
                    execution_ref_id="t1" if to_state == WatchState.ORDERED else None,
                )


# ---------------------------------------------------------------------------
# Property 4: Terminal states are absorbing under fuzzed transition sequences
# **Validates: Requirements 3.2**
# ---------------------------------------------------------------------------


@given(
    initial_state=st.sampled_from(list(TERMINAL_STATES)),
    transition_attempts=st.lists(st_state, min_size=1, max_size=15),
)
@settings(max_examples=200)
def test_terminal_states_absorbing(initial_state, transition_attempts):
    """Once a watch reaches a terminal state, no further transition is possible."""
    engine = create_engine("sqlite:///:memory:")
    init_setup_watch_schema(engine)
    registry = SetupWatchRegistry(engine)

    watch = _make_watch(state=WatchState.WATCHING)
    watch_id = registry.create_watch(watch)

    # Force to terminal state
    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE setup_watches SET state = :state, terminal_reason = :reason "
                "WHERE watch_id = :wid"
            ),
            {"state": initial_state.value, "reason": "test", "wid": watch_id},
        )
        conn.commit()

    # Attempt transitions — all must raise
    for target_state in transition_attempts:
        if target_state == initial_state:
            continue
        with pytest.raises(SetupWatchRegistryError):
            kwargs = {}
            if target_state in TERMINAL_STATES:
                kwargs["terminal_reason"] = "fuzz"
            if target_state == WatchState.ORDERED:
                kwargs["execution_ref_type"] = "trade"
                kwargs["execution_ref_id"] = "t1"
            registry.transition_state(
                watch_id, initial_state, target_state, **kwargs
            )

    # Verify state unchanged
    with engine.connect() as conn:
        final = conn.execute(
            text("SELECT state FROM setup_watches WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).scalar()
    assert final == initial_state.value


# ---------------------------------------------------------------------------
# Property 5: integrity_hash is deterministic across repeated computation
# **Validates: Requirements 12.6**
# ---------------------------------------------------------------------------


@given(
    profile_id=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L", "N"))),
    symbol=st.text(min_size=1, max_size=10, alphabet=st.characters(categories=("Lu",))),
    side=st_side,
    setup_type=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L", "N"))),
    thesis=st.text(min_size=1, max_size=100),
)
@settings(max_examples=200)
def test_integrity_hash_deterministic(profile_id, symbol, side, setup_type, thesis):
    """Same identity fields always produce the same hash."""
    watch = _make_watch(
        profile_id=profile_id,
        symbol=symbol,
        side=side,
        setup_type=setup_type,
        thesis=thesis,
    )

    hash1 = compute_watch_integrity_hash(watch)
    hash2 = compute_watch_integrity_hash(watch)
    hash3 = compute_watch_integrity_hash(watch)

    assert hash1 == hash2 == hash3
    assert len(hash1) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# Property 6: expire_elapsed never expires a future-dated watch
# **Validates: Requirements 12.10**
# ---------------------------------------------------------------------------


@given(
    hours_ahead=st.floats(min_value=0.1, max_value=168.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_expire_elapsed_never_expires_future(hours_ahead):
    """A watch whose expires_at is in the future is never expired."""
    engine = create_engine("sqlite:///:memory:")
    init_setup_watch_schema(engine)
    registry = SetupWatchRegistry(engine)

    future_expiry = NOW + timedelta(hours=hours_ahead)
    watch = _make_watch(expires_at=future_expiry, created_at=NOW, updated_at=NOW)
    watch_id = registry.create_watch(watch)

    # Expire with a "now" that is before the expiry
    # The registry uses now_utc() internally, so we patch it to be before expiry
    from unittest.mock import patch as _patch
    check_time = future_expiry - timedelta(minutes=1)
    with _patch("utils.setup_watch_registry.now_utc", return_value=check_time):
        with _patch("utils.setup_watch_registry.to_iso", side_effect=lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")):
            expired_count = registry.expire_elapsed("moderate")

    assert expired_count == 0

    # Verify watch is still active
    with engine.connect() as conn:
        state = conn.execute(
            text("SELECT state FROM setup_watches WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).scalar()
    assert state == WatchState.WATCHING.value


# ---------------------------------------------------------------------------
# Property 7: Condition definition columns are unchanged after operations
# **Validates: Requirements 4.12, 12.6**
# ---------------------------------------------------------------------------


@given(
    num_evaluations=st.integers(min_value=1, max_value=5),
    scores=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=5),
    increment_count=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=200)
def test_condition_columns_immutable_under_operations(num_evaluations, scores, increment_count):
    """maturation_conditions_json and invalidation_conditions_json stay byte-identical
    after arbitrary sequences of update_evaluation and increment_observed_cycles."""
    engine = create_engine("sqlite:///:memory:")
    init_setup_watch_schema(engine)
    registry = SetupWatchRegistry(engine)

    watch = _make_watch()
    watch_id = registry.create_watch(watch)

    # Record original conditions
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT maturation_conditions_json, invalidation_conditions_json "
                "FROM setup_watches WHERE watch_id = :wid"
            ),
            {"wid": watch_id},
        ).fetchone()
    original_mat = row[0]
    original_inv = row[1]

    # Perform arbitrary operations
    for i in range(min(num_evaluations, len(scores))):
        registry.update_evaluation(
            watch_id,
            maturity_score=scores[i],
            last_evaluation_json=json.dumps({"cycle": i, "score": scores[i]}),
        )

    for _ in range(increment_count):
        registry.increment_observed_cycles([watch_id])

    # Also try a state transition
    registry.transition_state(watch_id, WatchState.WATCHING, WatchState.MATURING)
    registry.update_evaluation(watch_id, maturity_score=0.9, last_evaluation_json='{"final": true}')

    # Verify conditions are byte-identical
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT maturation_conditions_json, invalidation_conditions_json "
                "FROM setup_watches WHERE watch_id = :wid"
            ),
            {"wid": watch_id},
        ).fetchone()
    assert row[0] == original_mat, "maturation_conditions_json was modified!"
    assert row[1] == original_inv, "invalidation_conditions_json was modified!"


# ---------------------------------------------------------------------------
# Property 8: MFE >= MAE for any candle series
# **Validates: Requirements 11.4**
# ---------------------------------------------------------------------------


@st.composite
def st_candle(draw):
    """Generate a valid OHLC candle with high >= max(open,close), low <= min(open,close)."""
    open_price = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    close_price = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))

    # High must be >= max(open, close)
    base_high = max(open_price, close_price)
    high_extra = draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False))
    high_price = base_high + high_extra

    # Low must be <= min(open, close)
    base_low = min(open_price, close_price)
    low_sub = draw(st.floats(min_value=0.0, max_value=min(base_low - 0.01, 50.0), allow_nan=False, allow_infinity=False))
    low_price = base_low - low_sub

    # Timestamp within the scoring window
    minutes_offset = draw(st.integers(min_value=0, max_value=59))
    ts = NOW + timedelta(minutes=minutes_offset)

    return {
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@given(
    candles=st.lists(st_candle(), min_size=1, max_size=20),
    side=st_side,
    ref_price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_mfe_gte_mae(candles, side, ref_price):
    """MFE (maximum favorable excursion) >= MAE (maximum adverse excursion).

    The best possible movement in the trade direction can never be below the
    worst possible movement against the trade direction. By construction, MFE
    measures the ceiling of favorable movement and MAE measures the floor of
    adverse movement (as a negative), so MFE >= MAE always holds.
    """
    # Build a minimal watch-like object with the fields score_watch_outcome needs
    class _MockWatch:
        def __init__(self):
            self.watch_id = str(uuid.uuid4())
            self.profile_id = "moderate"
            self.symbol = "TEST"
            self.side = side
            self.ready_reference_price = ref_price
            self.ready_at = NOW
            self.entry_zone_json = None
            self.draft_geometry_json = None

    watch = _MockWatch()

    outcome = score_watch_outcome(
        watch,
        window_label="w60",
        window_minutes=60,
        candles=candles,
    )

    if outcome.scorable == 0:
        # Unscorable outcomes don't have MFE/MAE — skip
        return

    assert outcome.mfe_pct is not None
    assert outcome.mae_pct is not None
    assert outcome.mfe_pct >= outcome.mae_pct, (
        f"MFE ({outcome.mfe_pct}) < MAE ({outcome.mae_pct}) — "
        f"favorable ceiling is below adverse floor for side={side}, ref={ref_price}"
    )
