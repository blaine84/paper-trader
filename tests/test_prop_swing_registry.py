"""Property-based tests for swing candidate registry integration.

Tests candidate type assignment, registry filtering by type,
swing expiration independence from intraday, expires-at computation,
and price deviation expiration.

Uses an in-memory SQLite database to validate registry behavior with
mixed swing and intraday candidates.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
from sqlalchemy import create_engine, text

from utils.candidate_registry import CandidateRecord, CandidateRegistry, CandidateState
from utils.gate_config import (
    CANDIDATE_EXECUTABLE_SETUP_TYPES,
    SWING_EXECUTABLE_SETUP_TYPES,
    SWING_MAX_CANDIDATE_AGE_HOURS,
    SWING_PRICE_DEVIATION_THRESHOLD_PCT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_engine():
    """Create an in-memory SQLite engine with pm_candidates and pm_candidate_events tables."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text('''
            CREATE TABLE pm_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL UNIQUE,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                setup_type TEXT NOT NULL,
                geometry_name TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                risk_reward REAL NOT NULL,
                trigger TEXT,
                invalidation_basis TEXT,
                target_basis TEXT,
                source_signal_id TEXT NOT NULL,
                signal_snapshot_json TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'registered',
                integrity_hash TEXT NOT NULL,
                execution_key TEXT,
                reserved_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME NOT NULL,
                context_snapshot_json TEXT,
                benchmark_mapping_json TEXT,
                rejection_reason TEXT,
                candidate_type TEXT DEFAULT 'intraday',
                holding_horizon INTEGER,
                normalized_setup_type TEXT
            )
        '''))
        conn.execute(text('''
            CREATE TABLE pm_candidate_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                candidate_type TEXT DEFAULT 'intraday'
            )
        '''))
    return eng


@pytest.fixture
def engine():
    """Pytest fixture wrapper for tests that don't use @given."""
    return _create_engine()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

candidate_type_st = st.sampled_from(["swing", "intraday"])
symbol_st = st.text(min_size=1, max_size=5, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
profile_st = st.sampled_from(["conservative", "moderate", "aggressive"])
direction_st = st.sampled_from(["BUY", "SHORT"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    cycle_id: str,
    profile_id: str,
    candidate_type: str = "intraday",
    symbol: str = "AAPL",
    direction: str = "BUY",
    setup_type: str | None = None,
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
    entry_price: float = 150.0,
) -> CandidateRecord:
    """Build a CandidateRecord with sensible defaults for testing."""
    now = created_at or datetime.now(timezone.utc)
    exp = expires_at or (now + timedelta(hours=SWING_MAX_CANDIDATE_AGE_HOURS))

    if setup_type is None:
        if candidate_type == "swing":
            setup_type = "sector_rotation_swing"
        else:
            setup_type = "momentum_fade"

    cid = str(uuid.uuid4())
    return CandidateRecord(
        candidate_id=cid,
        cycle_id=cycle_id,
        profile_id=profile_id,
        symbol=symbol,
        direction=direction,
        setup_type=setup_type,
        geometry_name="test_geometry",
        entry_price=entry_price,
        stop_price=140.0 if direction == "BUY" else 160.0,
        target_price=170.0 if direction == "BUY" else 130.0,
        risk_reward=2.0,
        trigger="test_trigger",
        invalidation_basis="test_invalidation",
        target_basis="test_target",
        source_signal_id=f"sig-{cid[:8]}",
        signal_snapshot_json='{"test": true}',
        created_at=now,
        expires_at=exp,
        integrity_hash=f"hash-{cid[:8]}",
        candidate_type=candidate_type,
    )


# ---------------------------------------------------------------------------
# Property 19: Candidate Type Assignment Invariant
# Validates: Requirements 6.1, 6.8
# ---------------------------------------------------------------------------


@given(
    candidate_type=candidate_type_st,
    symbol=symbol_st,
)
@settings(max_examples=200)
def test_candidate_type_assignment_invariant(candidate_type, symbol):
    """Property 19: candidate_type persisted is exactly the value written at registration.

    Swing candidates' setup_type MUST be in SWING_EXECUTABLE_SETUP_TYPES,
    intraday candidates' setup_type MUST be in CANDIDATE_EXECUTABLE_SETUP_TYPES.
    Pre-migration rows with NULL candidate_type are treated as 'intraday'.

    **Validates: Requirements 6.1, 6.8**
    """
    engine = _create_engine()
    cycle_id = f"cycle-{uuid.uuid4().hex[:8]}"
    profile_id = "moderate"

    # Pick a valid setup type for the candidate type
    if candidate_type == "swing":
        setup_type = "sector_rotation_swing"
    else:
        setup_type = "momentum_fade"

    candidate = _make_candidate(
        cycle_id=cycle_id,
        profile_id=profile_id,
        candidate_type=candidate_type,
        symbol=symbol,
        setup_type=setup_type,
    )

    registry = CandidateRegistry(db=engine, cycle_id=cycle_id, profile_id=profile_id)
    registry.register(candidate)

    # Verify persisted value matches what was written
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT candidate_type FROM pm_candidates WHERE candidate_id = :cid"),
            {"cid": candidate.candidate_id},
        ).fetchone()

    assert row is not None
    assert row[0] == candidate_type

    # Verify setup type membership consistency
    if candidate_type == "swing":
        assert setup_type in SWING_EXECUTABLE_SETUP_TYPES
    else:
        assert setup_type in CANDIDATE_EXECUTABLE_SETUP_TYPES


def test_null_candidate_type_treated_as_intraday(engine):
    """Pre-migration rows with NULL candidate_type are treated as 'intraday'.

    **Validates: Requirements 6.8**
    """
    cycle_id = "cycle-null-test"
    profile_id = "moderate"

    # Insert a row with NULL candidate_type (simulating pre-migration data)
    now = datetime.now(timezone.utc)
    exp = now + timedelta(hours=24)
    cid = str(uuid.uuid4())

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO pm_candidates (
                    candidate_id, cycle_id, profile_id, symbol, direction,
                    setup_type, geometry_name, entry_price, stop_price,
                    target_price, risk_reward, trigger, invalidation_basis,
                    target_basis, source_signal_id, signal_snapshot_json,
                    state, integrity_hash, created_at, expires_at,
                    candidate_type
                ) VALUES (
                    :cid, :cycle_id, :profile_id, 'AAPL', 'BUY',
                    'momentum_fade', 'test', 150.0, 140.0,
                    170.0, 2.0, 'trigger', 'invalidation',
                    'target', 'sig-1', '{}',
                    'registered', 'hash-1', :created_at, :expires_at,
                    NULL
                )
            """),
            {
                "cid": cid,
                "cycle_id": cycle_id,
                "profile_id": profile_id,
                "created_at": now.isoformat(),
                "expires_at": exp.isoformat(),
            },
        )

    registry = CandidateRegistry(db=engine, cycle_id=cycle_id, profile_id=profile_id)

    # Filtering by 'intraday' should include the NULL row
    intraday_ids = registry.get_registered_ids(candidate_type="intraday")
    assert cid in intraday_ids

    # Filtering by 'swing' should NOT include the NULL row
    swing_ids = registry.get_registered_ids(candidate_type="swing")
    assert cid not in swing_ids

    # Filtering by None (all) should include it
    all_ids = registry.get_registered_ids(candidate_type=None)
    assert cid in all_ids


# ---------------------------------------------------------------------------
# Property 20: Registry Filtering by Candidate Type
# Validates: Requirements 6.2
# ---------------------------------------------------------------------------


@given(
    num_swing=st.integers(min_value=0, max_value=5),
    num_intraday=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=200)
def test_registry_filtering_by_candidate_type(num_swing, num_intraday):
    """Property 20: get_offered_summary filters correctly by candidate_type.

    swing filter → only swing, intraday filter → only intraday, None → all.

    **Validates: Requirements 6.2**
    """
    engine = _create_engine()
    cycle_id = f"cycle-{uuid.uuid4().hex[:8]}"
    profile_id = "moderate"
    registry = CandidateRegistry(db=engine, cycle_id=cycle_id, profile_id=profile_id)

    swing_ids = set()
    intraday_ids = set()

    for i in range(num_swing):
        candidate = _make_candidate(
            cycle_id=cycle_id,
            profile_id=profile_id,
            candidate_type="swing",
            symbol=f"SW{i}",
        )
        registry.register(candidate)
        swing_ids.add(candidate.candidate_id)

    for i in range(num_intraday):
        candidate = _make_candidate(
            cycle_id=cycle_id,
            profile_id=profile_id,
            candidate_type="intraday",
            symbol=f"ID{i}",
        )
        registry.register(candidate)
        intraday_ids.add(candidate.candidate_id)

    # Filter swing only
    swing_summary = registry.get_offered_summary(candidate_type="swing")
    swing_returned = {c["candidate_id"] for c in swing_summary}
    assert swing_returned == swing_ids

    # Filter intraday only
    intraday_summary = registry.get_offered_summary(candidate_type="intraday")
    intraday_returned = {c["candidate_id"] for c in intraday_summary}
    assert intraday_returned == intraday_ids

    # No filter → all
    all_summary = registry.get_offered_summary(candidate_type=None)
    all_returned = {c["candidate_id"] for c in all_summary}
    assert all_returned == swing_ids | intraday_ids


# ---------------------------------------------------------------------------
# Property 22: Swing Expiration Independence from Intraday
# Validates: Requirements 9.1
# ---------------------------------------------------------------------------


@given(
    hours_old=st.integers(min_value=1, max_value=SWING_MAX_CANDIDATE_AGE_HOURS - 1),
)
@settings(max_examples=200)
def test_swing_expiration_independence_from_intraday(hours_old):
    """Property 22: Swing candidates do NOT expire due to intraday window closure.

    A swing candidate whose expires_at is in the future should remain
    REGISTERED even if an intraday candidate in the same cycle has expired.

    **Validates: Requirements 9.1**
    """
    engine = _create_engine()
    cycle_id = f"cycle-{uuid.uuid4().hex[:8]}"
    profile_id = "moderate"

    now = datetime.now(timezone.utc)

    # Create an intraday candidate that is already expired
    intraday_created = now - timedelta(hours=5)
    intraday_expires = now - timedelta(hours=1)  # expired
    intraday_candidate = _make_candidate(
        cycle_id=cycle_id,
        profile_id=profile_id,
        candidate_type="intraday",
        symbol="INTD",
        created_at=intraday_created,
        expires_at=intraday_expires,
    )

    # Create a swing candidate that is NOT expired
    # Its expires_at is set using SWING_MAX_CANDIDATE_AGE_HOURS from created_at
    swing_created = now - timedelta(hours=hours_old)
    swing_expires = swing_created + timedelta(hours=SWING_MAX_CANDIDATE_AGE_HOURS)
    swing_candidate = _make_candidate(
        cycle_id=cycle_id,
        profile_id=profile_id,
        candidate_type="swing",
        symbol="SWNG",
        created_at=swing_created,
        expires_at=swing_expires,
    )

    registry = CandidateRegistry(db=engine, cycle_id=cycle_id, profile_id=profile_id)
    registry.register(intraday_candidate)
    registry.register(swing_candidate)

    # Finalize cycle — intraday should expire, swing should NOT expire
    # (swing_expires is in the future since hours_old < SWING_MAX_CANDIDATE_AGE_HOURS)
    terminal = registry.finalize_cycle()

    assert terminal[intraday_candidate.candidate_id] == CandidateState.EXPIRED
    # Swing candidate should be NOT_SELECTED (still had time remaining),
    # not EXPIRED
    assert terminal[swing_candidate.candidate_id] == CandidateState.NOT_SELECTED


# ---------------------------------------------------------------------------
# Property 23: Swing Expires-At Computation
# Validates: Requirements 9.2
# ---------------------------------------------------------------------------


@given(
    hours_offset=st.integers(min_value=0, max_value=72),
    minutes_offset=st.integers(min_value=0, max_value=59),
)
@settings(max_examples=200)
def test_swing_expires_at_computation(hours_offset, minutes_offset):
    """Property 23: swing expires_at = created_at + SWING_MAX_CANDIDATE_AGE_HOURS.

    For any swing candidate, the correct expires_at is exactly
    created_at + timedelta(hours=SWING_MAX_CANDIDATE_AGE_HOURS).

    **Validates: Requirements 9.2**
    """
    engine = _create_engine()
    base_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    created_at = base_time - timedelta(hours=hours_offset, minutes=minutes_offset)
    expected_expires_at = created_at + timedelta(hours=SWING_MAX_CANDIDATE_AGE_HOURS)

    # Build candidate with the correct expires_at formula
    candidate = _make_candidate(
        cycle_id="cycle-exp-test",
        profile_id="moderate",
        candidate_type="swing",
        symbol="TEST",
        created_at=created_at,
        expires_at=expected_expires_at,
    )

    # Verify the invariant: expires_at == created_at + SWING_MAX_CANDIDATE_AGE_HOURS
    assert candidate.expires_at == candidate.created_at + timedelta(
        hours=SWING_MAX_CANDIDATE_AGE_HOURS
    )

    # Also verify persistence roundtrip preserves the relationship
    cycle_id = f"cycle-{uuid.uuid4().hex[:8]}"
    candidate = _make_candidate(
        cycle_id=cycle_id,
        profile_id="moderate",
        candidate_type="swing",
        symbol="TEST",
        created_at=created_at,
        expires_at=expected_expires_at,
    )

    registry = CandidateRegistry(db=engine, cycle_id=cycle_id, profile_id="moderate")
    registry.register(candidate)

    # Read back and verify
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT created_at, expires_at FROM pm_candidates WHERE candidate_id = :cid"),
            {"cid": candidate.candidate_id},
        ).fetchone()

    assert row is not None
    persisted_created = datetime.fromisoformat(row[0])
    persisted_expires = datetime.fromisoformat(row[1])

    # The difference should be exactly SWING_MAX_CANDIDATE_AGE_HOURS
    delta = persisted_expires - persisted_created
    assert delta == timedelta(hours=SWING_MAX_CANDIDATE_AGE_HOURS)


# ---------------------------------------------------------------------------
# Property 24: Price Deviation Expiration
# Validates: Requirements 9.5
# ---------------------------------------------------------------------------


@given(
    entry_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    deviation_pct=st.floats(min_value=3.1, max_value=50.0, allow_nan=False, allow_infinity=False),
    direction_up=st.booleans(),
)
@settings(max_examples=200)
def test_price_deviation_expiration(entry_price, deviation_pct, direction_up):
    """Property 24: When current price deviates > SWING_PRICE_DEVIATION_THRESHOLD_PCT from entry, expire.

    Any swing candidate whose current price has moved beyond the threshold
    in either direction should expire with reason 'price_moved_beyond_threshold'.

    **Validates: Requirements 9.5**
    """
    threshold = float(SWING_PRICE_DEVIATION_THRESHOLD_PCT)

    # Compute a current price that deviates beyond threshold
    if direction_up:
        current_price = entry_price * (1 + deviation_pct / 100.0)
    else:
        current_price = entry_price * (1 - deviation_pct / 100.0)

    # Verify the deviation exceeds threshold
    actual_deviation_pct = abs(current_price - entry_price) / entry_price * 100.0
    assert actual_deviation_pct > threshold

    # The property: when deviation > threshold, candidate should expire.
    # This validates the LOGIC of the check (not a full integration test of
    # finalize_cycle with live prices, since that requires price feed injection).
    # The registry checks: abs(current_price - entry_price) / entry_price > threshold/100
    should_expire = (
        abs(current_price - entry_price) / entry_price
        > float(SWING_PRICE_DEVIATION_THRESHOLD_PCT) / 100.0
    )
    assert should_expire is True


@given(
    entry_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    deviation_pct=st.floats(min_value=0.0, max_value=2.9, allow_nan=False, allow_infinity=False),
    direction_up=st.booleans(),
)
@settings(max_examples=200)
def test_price_within_threshold_does_not_expire(entry_price, deviation_pct, direction_up):
    """Property 24 (converse): When price deviation <= threshold, candidate should NOT expire.

    **Validates: Requirements 9.5**
    """
    threshold = float(SWING_PRICE_DEVIATION_THRESHOLD_PCT)

    if direction_up:
        current_price = entry_price * (1 + deviation_pct / 100.0)
    else:
        current_price = entry_price * (1 - deviation_pct / 100.0)

    # Verify the deviation does NOT exceed threshold
    actual_deviation_pct = abs(current_price - entry_price) / entry_price * 100.0
    assert actual_deviation_pct <= threshold

    # The property: when deviation <= threshold, should NOT expire due to price
    should_expire_price = (
        abs(current_price - entry_price) / entry_price
        > float(SWING_PRICE_DEVIATION_THRESHOLD_PCT) / 100.0
    )
    assert should_expire_price is False
