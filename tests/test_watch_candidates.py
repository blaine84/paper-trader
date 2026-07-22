"""Unit tests for watch candidate lifecycle management.

Requirements: 17.5
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect as sa_inspect, text

from orchestrator import _ensure_watch_candidate_tables
from utils.watch_candidates import (
    WatchCandidate,
    evaluate_active_watch_candidates,
    expire_session_watch_candidates,
    _insert_watch_candidate,
    _transition_watch_state,
    _evaluate_watch,
    _threshold_crossed,
)


@pytest.fixture
def engine():
    """In-memory SQLite engine with watch_candidates table."""
    eng = create_engine("sqlite:///:memory:")
    inspector = sa_inspect(eng)
    _ensure_watch_candidate_tables(eng, inspector)
    return eng


def _make_watch(
    symbol="AAPL",
    profile_id="test_profile",
    direction="LONG",
    activation=None,
    invalidation=None,
    expires_hours=7,
    watch_id=None,
    state="active",
) -> WatchCandidate:
    """Helper to build a test WatchCandidate."""
    now = datetime.now(timezone.utc)
    return WatchCandidate(
        watch_id=watch_id or f"watch-{symbol}-{direction}-{now.timestamp()}",
        symbol=symbol,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        expires_at=(now + timedelta(hours=expires_hours)).isoformat(),
        source_cycle_id="cycle-123",
        profile_id=profile_id,
        market_state="compression_under_resistance",
        setup_lifecycle_state="compression_watch",
        timeframe_authority_json=json.dumps({"authority": "aligned"}),
        direction_watch=direction,
        trade_posture="watch_long_trigger",
        activation_conditions_json=json.dumps(
            activation or [{"condition": "price > above resistance", "threshold": 110.0}]
        ),
        invalidation_conditions_json=json.dumps(
            invalidation or [{"condition": "price below support", "threshold": 95.0}]
        ),
        key_levels_json=json.dumps({"resistance": 110.0, "support": 95.0}),
        trigger_status_json=json.dumps({}),
        reason="Test watch candidate",
        source_signal_snapshot_json=json.dumps(
            {"signal": "LONG", "setup_type": "technical_breakout"}
        ),
        state=state,
    )


# ─── Happy Path Insert ────────────────────────────────────────────────────────


def test_create_watch_candidate_happy_path(engine):
    """Verify INSERT and field values after creation."""
    watch = _make_watch(symbol="AAPL", direction="LONG")
    result = _insert_watch_candidate(engine, watch)
    assert result is True

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": watch.watch_id},
        ).fetchone()

    assert row is not None
    # Map column names to values
    cols = row._mapping
    assert cols["symbol"] == "AAPL"
    assert cols["direction_watch"] == "LONG"
    assert cols["profile_id"] == "test_profile"
    assert cols["state"] == "active"
    assert cols["market_state"] == "compression_under_resistance"
    assert cols["setup_lifecycle_state"] == "compression_watch"
    assert json.loads(cols["activation_conditions_json"])[0]["threshold"] == 110.0
    assert json.loads(cols["invalidation_conditions_json"])[0]["threshold"] == 95.0


# ─── Deduplication ────────────────────────────────────────────────────────────


def test_deduplication_expires_existing(engine):
    """Same (symbol, profile_id, direction) → old watch expired, new inserted."""
    watch1 = _make_watch(symbol="TSLA", direction="LONG", watch_id="watch-old")
    watch2 = _make_watch(symbol="TSLA", direction="LONG", watch_id="watch-new")

    assert _insert_watch_candidate(engine, watch1) is True
    assert _insert_watch_candidate(engine, watch2) is True

    with engine.connect() as conn:
        old = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-old"},
        ).fetchone()
        new = conn.execute(
            text("SELECT state FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-new"},
        ).fetchone()

    assert old._mapping["state"] == "expired"
    outcome = json.loads(old._mapping["outcome_json"])
    assert outcome["terminal_reason"] == "replaced_by_newer_watch"
    assert new._mapping["state"] == "active"


def test_deduplication_unique_index_race(engine):
    """Simulate IntegrityError from unique index race — verify graceful handling."""
    watch = _make_watch(symbol="RACE", direction="SHORT", watch_id="watch-race-1")
    assert _insert_watch_candidate(engine, watch) is True

    # Manually set the first one back to active so unique index conflicts
    # by inserting a second with same unique key directly
    with engine.connect() as conn:
        # Force a conflict by inserting directly with same unique combo
        try:
            conn.execute(
                text(
                    "INSERT INTO watch_candidates "
                    "(watch_id, symbol, created_at, updated_at, expires_at, source_cycle_id, "
                    " profile_id, market_state, setup_lifecycle_state, timeframe_authority_json, "
                    " direction_watch, trade_posture, activation_conditions_json, "
                    " invalidation_conditions_json, key_levels_json, trigger_status_json, "
                    " reason, source_signal_snapshot_json, state) "
                    "VALUES (:watch_id, :symbol, :created_at, :updated_at, :expires_at, "
                    " :source_cycle_id, :profile_id, :market_state, :setup_lifecycle_state, "
                    " :timeframe_authority_json, :direction_watch, :trade_posture, "
                    " :activation_conditions_json, :invalidation_conditions_json, "
                    " :key_levels_json, :trigger_status_json, :reason, "
                    " :source_signal_snapshot_json, :state)"
                ),
                {
                    "watch_id": "watch-race-1",
                    "symbol": "RACE",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "expires_at": (datetime.now(timezone.utc) + timedelta(hours=7)).isoformat(),
                    "source_cycle_id": "cycle-456",
                    "profile_id": "test_profile",
                    "market_state": "confounded",
                    "setup_lifecycle_state": "no_setup",
                    "timeframe_authority_json": "{}",
                    "direction_watch": "SHORT",
                    "trade_posture": "flat",
                    "activation_conditions_json": "[]",
                    "invalidation_conditions_json": "[]",
                    "key_levels_json": "{}",
                    "trigger_status_json": "{}",
                    "reason": "race test",
                    "source_signal_snapshot_json": "{}",
                    "state": "active",
                },
            )
            conn.commit()
        except Exception:
            # Expected — UNIQUE constraint violation on watch_id
            pass

    # The _insert_watch_candidate function should handle this gracefully
    # (the dedup logic expires existing, but a true race with same watch_id returns False)
    watch_dup = _make_watch(symbol="RACE", direction="SHORT", watch_id="watch-race-1")
    result = _insert_watch_candidate(engine, watch_dup)
    # Returns False because of UNIQUE constraint on watch_id
    assert result is False


# ─── Threshold Evaluation ─────────────────────────────────────────────────────


def test_evaluate_invalidation_threshold_crossed():
    """Price below support → invalidated."""
    watch = _make_watch(
        invalidation=[{"condition": "price below support", "threshold": 95.0}],
        activation=[{"condition": "price > above resistance", "threshold": 110.0}],
    )
    result = _evaluate_watch(watch, current_price=94.0)
    assert result == "invalidated"


def test_evaluate_activation_threshold_crossed():
    """Price above resistance → promotion_eligible."""
    watch = _make_watch(
        invalidation=[{"condition": "price below support", "threshold": 95.0}],
        activation=[{"condition": "price > above resistance", "threshold": 110.0}],
    )
    result = _evaluate_watch(watch, current_price=111.0)
    assert result == "promotion_eligible"


def test_invalidation_priority_over_activation():
    """Both thresholds crossed → invalidated wins (conservative, per Req 10.5)."""
    # Set up a situation where both conditions are simultaneously true
    watch = _make_watch(
        invalidation=[{"condition": "price below support", "threshold": 115.0}],
        activation=[{"condition": "price > above resistance", "threshold": 110.0}],
    )
    # Price is 112: above activation (110) AND below invalidation (115)
    result = _evaluate_watch(watch, current_price=112.0)
    assert result == "invalidated"


def test_expired_candidate_not_evaluated(engine):
    """Only active candidates are evaluated — expired ones are skipped."""
    watch = _make_watch(symbol="MSFT", direction="LONG", watch_id="watch-expired-1")
    _insert_watch_candidate(engine, watch)

    # Manually expire the candidate
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE watch_candidates SET state = 'expired' WHERE watch_id = :wid"),
            {"wid": "watch-expired-1"},
        )
        conn.commit()

    # evaluate_active_watch_candidates queries WHERE state = 'active'
    signals = {"MSFT": {"current_price": 111.0}}
    counts = evaluate_active_watch_candidates(engine, signals, "test_profile")

    # The expired candidate should not be counted in any category
    assert counts["invalidated"] == 0
    assert counts["promotion_eligible"] == 0
    assert counts["still_active"] == 0


# ─── Promotion in Enforcing Mode ─────────────────────────────────────────────


@patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing")
def test_promotion_enforcing_mode(engine):
    """Activation + directional signal + executable setup → promoted."""
    watch = _make_watch(
        symbol="NVDA",
        direction="LONG",
        watch_id="watch-promo-1",
        activation=[{"condition": "price > above resistance", "threshold": 110.0}],
        invalidation=[{"condition": "price below support", "threshold": 95.0}],
    )
    _insert_watch_candidate(engine, watch)

    signals = {
        "NVDA": {
            "current_price": 115.0,
            "signal": "LONG",
            "setup_type": "technical_breakout",
        }
    }
    counts = evaluate_active_watch_candidates(engine, signals, "test_profile")
    assert counts["promotion_eligible"] == 1

    # Verify state transitioned to promoted
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-promo-1"},
        ).fetchone()

    assert row._mapping["state"] == "promoted"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_state"] == "promoted"


@patch("utils.watch_candidates.MARKET_STATE_MODE", "observe")
def test_promotion_observe_mode_disabled(engine):
    """Activation in observe mode → expired with outcome_json (no promotion)."""
    watch = _make_watch(
        symbol="AMZN",
        direction="LONG",
        watch_id="watch-obs-1",
        activation=[{"condition": "price > above resistance", "threshold": 110.0}],
        invalidation=[{"condition": "price below support", "threshold": 95.0}],
    )
    _insert_watch_candidate(engine, watch)

    signals = {
        "AMZN": {
            "current_price": 115.0,
            "signal": "LONG",
            "setup_type": "technical_breakout",
        }
    }
    counts = evaluate_active_watch_candidates(engine, signals, "test_profile")
    assert counts["promotion_eligible"] == 1

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-obs-1"},
        ).fetchone()

    assert row._mapping["state"] == "expired"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_reason"] == "activation_observed_in_observe_mode"


@patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing")
def test_promotion_fails_hold_signal(engine):
    """Activation but HOLD signal → expired (cannot promote without direction)."""
    watch = _make_watch(
        symbol="META",
        direction="LONG",
        watch_id="watch-hold-1",
        activation=[{"condition": "price > above resistance", "threshold": 110.0}],
        invalidation=[{"condition": "price below support", "threshold": 95.0}],
    )
    _insert_watch_candidate(engine, watch)

    signals = {
        "META": {
            "current_price": 115.0,
            "signal": "HOLD",
            "setup_type": "technical_breakout",
        }
    }
    counts = evaluate_active_watch_candidates(engine, signals, "test_profile")
    assert counts["promotion_eligible"] == 1

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-hold-1"},
        ).fetchone()

    assert row._mapping["state"] == "expired"
    outcome = json.loads(row._mapping["outcome_json"])
    assert "promotion_blocked" in outcome["terminal_reason"] or "blocked" in outcome.get("block_reason", "")


@patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing")
def test_promotion_fails_non_executable_setup(engine):
    """Activation but non-executable setup type → expired."""
    watch = _make_watch(
        symbol="GOOG",
        direction="LONG",
        watch_id="watch-bad-setup-1",
        activation=[{"condition": "price > above resistance", "threshold": 110.0}],
        invalidation=[{"condition": "price below support", "threshold": 95.0}],
    )
    _insert_watch_candidate(engine, watch)

    signals = {
        "GOOG": {
            "current_price": 115.0,
            "signal": "LONG",
            "setup_type": "unclear_direction",  # Not in CANDIDATE_EXECUTABLE_SETUP_TYPES
        }
    }
    counts = evaluate_active_watch_candidates(engine, signals, "test_profile")
    assert counts["promotion_eligible"] == 1

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-bad-setup-1"},
        ).fetchone()

    assert row._mapping["state"] == "expired"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_reason"] == "activation_detected_but_promotion_blocked"


# ─── Session Expiration ───────────────────────────────────────────────────────


def test_expire_session_sweeps_active(engine):
    """Past expires_at → expired by session sweep."""
    # Create a watch that already expired (expires_at in the past)
    watch = _make_watch(
        symbol="SPOT",
        direction="LONG",
        watch_id="watch-sweep-1",
        expires_hours=-1,  # Already expired 1 hour ago
    )
    _insert_watch_candidate(engine, watch)

    expired_count = expire_session_watch_candidates(engine)
    assert expired_count == 1

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-sweep-1"},
        ).fetchone()

    assert row._mapping["state"] == "expired"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_reason"] == "session_expired"


def test_expire_session_idempotent(engine):
    """Already expired → no change on second sweep."""
    watch = _make_watch(
        symbol="UBER",
        direction="SHORT",
        watch_id="watch-idem-1",
        expires_hours=-2,
    )
    _insert_watch_candidate(engine, watch)

    # First sweep
    count1 = expire_session_watch_candidates(engine)
    assert count1 == 1

    # Second sweep — already expired, should not re-expire
    count2 = expire_session_watch_candidates(engine)
    assert count2 == 0


# ─── Shadow Outcome Recording ────────────────────────────────────────────────


def test_shadow_outcome_recorded_on_invalidation(engine):
    """Invalidation records outcome_json with terminal details."""
    watch = _make_watch(
        symbol="DIS",
        direction="LONG",
        watch_id="watch-inv-shadow-1",
        invalidation=[{"condition": "price below support", "threshold": 95.0}],
        activation=[{"condition": "price > above resistance", "threshold": 110.0}],
    )
    _insert_watch_candidate(engine, watch)

    signals = {"DIS": {"current_price": 90.0}}
    evaluate_active_watch_candidates(engine, signals, "test_profile")

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-inv-shadow-1"},
        ).fetchone()

    assert row._mapping["state"] == "invalidated"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_state"] == "invalidated"
    assert outcome["terminal_reason"] == "invalidation_threshold_crossed"
    assert outcome["price_at_invalidation"] == 90.0


@patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing")
def test_shadow_outcome_recorded_on_promotion(engine):
    """Promotion records outcome_json with activation details."""
    watch = _make_watch(
        symbol="COIN",
        direction="LONG",
        watch_id="watch-promo-shadow-1",
        activation=[{"condition": "price > above resistance", "threshold": 110.0}],
        invalidation=[{"condition": "price below support", "threshold": 95.0}],
    )
    _insert_watch_candidate(engine, watch)

    signals = {
        "COIN": {
            "current_price": 112.0,
            "signal": "LONG",
            "setup_type": "technical_breakout",
        }
    }
    evaluate_active_watch_candidates(engine, signals, "test_profile")

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-promo-shadow-1"},
        ).fetchone()

    assert row._mapping["state"] == "promoted"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_state"] == "promoted"
    assert outcome["price_at_activation"] == 112.0
