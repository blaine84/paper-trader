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
    get_promotable_candidates,
    _insert_watch_candidate,
    _transition_watch_state,
    _evaluate_watch,
    _threshold_crossed,
    _check_structural_invalidation,
    _check_same_cycle_policy,
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
    # Enforcing mode requires a cycle_id for promotion (Req 4.10).
    counts = evaluate_active_watch_candidates(
        engine, signals, "test_profile", cycle_id="cycle-promo"
    )
    assert counts["promotion_eligible"] == 1

    # Verify state transitioned to promoted and promoted_cycle_id recorded atomically
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT state, outcome_json, promoted_cycle_id "
                "FROM watch_candidates WHERE watch_id = :wid"
            ),
            {"wid": "watch-promo-1"},
        ).fetchone()

    assert row._mapping["state"] == "promoted"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_state"] == "promoted"
    assert row._mapping["promoted_cycle_id"] == "cycle-promo"


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
    # Enforcing mode requires a cycle_id to reach the promote/block branch (Req 4.10).
    counts = evaluate_active_watch_candidates(
        engine, signals, "test_profile", cycle_id="cycle-hold"
    )
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
    # Enforcing mode requires a cycle_id to reach the promote/block branch (Req 4.10).
    counts = evaluate_active_watch_candidates(
        engine, signals, "test_profile", cycle_id="cycle-bad-setup"
    )
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
    assert outcome["terminal_reason"] == "ttl_expired"


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
    # Enforcing mode requires a cycle_id for promotion (Req 4.10).
    evaluate_active_watch_candidates(
        engine, signals, "test_profile", cycle_id="cycle-promo-shadow"
    )

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-promo-shadow-1"},
        ).fetchone()

    assert row._mapping["state"] == "promoted"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_state"] == "promoted"
    assert outcome["price_at_activation"] == 112.0


# ─── get_promotable_candidates cycle_id scoping ───────────────────────────────


def _insert_promoted(engine, watch_id, symbol, promoted_cycle_id):
    """Insert a promoted watch row with a given promoted_cycle_id."""
    watch = _make_watch(symbol=symbol, direction="LONG", watch_id=watch_id)
    _insert_watch_candidate(engine, watch)
    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE watch_candidates "
                "SET state = 'promoted', promoted_cycle_id = :cid "
                "WHERE watch_id = :wid"
            ),
            {"cid": promoted_cycle_id, "wid": watch_id},
        )
        conn.commit()


@patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing")
def test_get_promotable_enforcing_none_cycle_returns_empty(engine):
    """Enforcing mode + cycle_id=None → empty list (Req 1.8)."""
    _insert_promoted(engine, "watch-e-1", "AAPL", "cycle-A")
    signals = {"AAPL": {"signal": "LONG", "setup_type": "technical_breakout"}}

    result = get_promotable_candidates(engine, signals, "test_profile")
    assert result == []


@patch("utils.watch_candidates.MARKET_STATE_MODE", "observe")
def test_get_promotable_nonenforcing_none_cycle_returns_all(engine):
    """Non-enforcing mode + cycle_id=None → all promoted for profile (Req 1.9)."""
    _insert_promoted(engine, "watch-n-1", "AAPL", "cycle-A")
    _insert_promoted(engine, "watch-n-2", "MSFT", "cycle-B")
    signals = {
        "AAPL": {"signal": "LONG", "setup_type": "technical_breakout"},
        "MSFT": {"signal": "LONG", "setup_type": "technical_breakout"},
    }

    result = get_promotable_candidates(engine, signals, "test_profile")
    symbols = {r["symbol"] for r in result}
    assert symbols == {"AAPL", "MSFT"}


@patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing")
def test_get_promotable_filters_by_cycle_id(engine):
    """cycle_id provided → only rows with matching promoted_cycle_id (Req 1.1)."""
    _insert_promoted(engine, "watch-c-1", "AAPL", "cycle-A")
    _insert_promoted(engine, "watch-c-2", "MSFT", "cycle-B")
    signals = {
        "AAPL": {"signal": "LONG", "setup_type": "technical_breakout"},
        "MSFT": {"signal": "LONG", "setup_type": "technical_breakout"},
    }

    result = get_promotable_candidates(engine, signals, "test_profile", cycle_id="cycle-A")
    symbols = {r["symbol"] for r in result}
    assert symbols == {"AAPL"}


@patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing")
def test_get_promotable_cycle_id_excludes_nonexecutable(engine):
    """cycle-scoped query still applies directional/executable-setup filter."""
    _insert_promoted(engine, "watch-x-1", "AAPL", "cycle-A")
    signals = {"AAPL": {"signal": "LONG", "setup_type": "unclear_direction"}}

    result = get_promotable_candidates(engine, signals, "test_profile", cycle_id="cycle-A")
    assert result == []


# ─── Structural Invalidation (_check_structural_invalidation) ─────────────────
#
# Unit tests for _check_structural_invalidation(watch, current_signal) -> str | None.
# Returns "structural_degradation", "key_level_drift", or None (fail-open).
# Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.8.


def _make_structural_watch(
    watch_id="watch-struct-1",
    key_levels=None,
) -> WatchCandidate:
    """Build a watch with a specific stored key_levels_json for drift tests.

    Defaults to stored resistance=110.0, support=95.0 (matching _make_watch).
    """
    if key_levels is None:
        key_levels = {"resistance": 110.0, "support": 95.0}
    watch = _make_watch(symbol="AAPL", direction="LONG", watch_id=watch_id)
    watch.key_levels_json = json.dumps(key_levels)
    return watch


# --- Lifecycle state regression (Req 3.1, 3.2) --------------------------------


def test_structural_lifecycle_regression_invalidates():
    """Current lifecycle not in WATCHABLE_LIFECYCLE_STATES → structural_degradation."""
    watch = _make_structural_watch()
    signal = {
        "setup_lifecycle_state": "no_setup",  # not watchable
        "key_levels": {"resistance": 110.0, "support": 95.0},
    }
    assert _check_structural_invalidation(watch, signal) == "structural_degradation"


def test_structural_lifecycle_watchable_state_not_invalidated():
    """Current lifecycle still in WATCHABLE_LIFECYCLE_STATES → no structural degradation."""
    watch = _make_structural_watch()
    signal = {
        "setup_lifecycle_state": "breakout_watch",  # watchable
        "key_levels": {"resistance": 110.0, "support": 95.0},  # no drift
    }
    assert _check_structural_invalidation(watch, signal) is None


def test_structural_lifecycle_takes_priority_over_drift():
    """Lifecycle regression is checked before drift → structural_degradation wins."""
    watch = _make_structural_watch()
    signal = {
        "setup_lifecycle_state": "no_setup",  # not watchable
        "key_levels": {"resistance": 200.0, "support": 10.0},  # huge drift too
    }
    # Lifecycle check runs first, so structural_degradation is returned.
    assert _check_structural_invalidation(watch, signal) == "structural_degradation"


# --- Key-level drift (Req 3.3, 3.4) -------------------------------------------


@patch("utils.watch_candidates.WATCH_KEY_LEVEL_DRIFT_PCT", 2.0)
def test_structural_key_level_drift_above_threshold_resistance():
    """Resistance drift > threshold → key_level_drift."""
    watch = _make_structural_watch(key_levels={"resistance": 110.0, "support": 95.0})
    # 115 vs 110 → |115-110|/110*100 = 4.545% > 2.0%
    signal = {
        "setup_lifecycle_state": "breakout_watch",
        "key_levels": {"resistance": 115.0, "support": 95.0},
    }
    assert _check_structural_invalidation(watch, signal) == "key_level_drift"


@patch("utils.watch_candidates.WATCH_KEY_LEVEL_DRIFT_PCT", 2.0)
def test_structural_key_level_drift_above_threshold_support():
    """Support drift > threshold → key_level_drift."""
    watch = _make_structural_watch(key_levels={"resistance": 110.0, "support": 95.0})
    # 90 vs 95 → |90-95|/95*100 = 5.26% > 2.0%
    signal = {
        "setup_lifecycle_state": "breakout_watch",
        "key_levels": {"resistance": 110.0, "support": 90.0},
    }
    assert _check_structural_invalidation(watch, signal) == "key_level_drift"


@patch("utils.watch_candidates.WATCH_KEY_LEVEL_DRIFT_PCT", 2.0)
def test_structural_key_level_drift_within_threshold_no_invalidation():
    """Drift within threshold on both levels → None (no invalidation)."""
    watch = _make_structural_watch(key_levels={"resistance": 110.0, "support": 95.0})
    # resistance: 111 vs 110 → 0.9%; support: 95.5 vs 95 → 0.53% — both < 2%
    signal = {
        "setup_lifecycle_state": "breakout_watch",
        "key_levels": {"resistance": 111.0, "support": 95.5},
    }
    assert _check_structural_invalidation(watch, signal) is None


@patch("utils.watch_candidates.WATCH_KEY_LEVEL_DRIFT_PCT", 2.0)
def test_structural_key_level_drift_exactly_at_threshold_not_invalidated():
    """Drift exactly equal to threshold is NOT > threshold → None."""
    watch = _make_structural_watch(key_levels={"resistance": 100.0, "support": 95.0})
    # resistance: 102 vs 100 → exactly 2.0%, which is not strictly greater than 2.0
    signal = {
        "setup_lifecycle_state": "breakout_watch",
        "key_levels": {"resistance": 102.0, "support": 95.0},
    }
    assert _check_structural_invalidation(watch, signal) is None


# --- Non-numeric / <= 0 levels are skipped (Req 3.8) --------------------------


@patch("utils.watch_candidates.WATCH_KEY_LEVEL_DRIFT_PCT", 2.0)
def test_structural_skips_nonpositive_stored_level():
    """Stored level <= 0 is skipped from drift comparison → None."""
    watch = _make_structural_watch(key_levels={"resistance": 0, "support": 0})
    # Current values would represent huge drift, but stored <= 0 → skipped.
    signal = {
        "setup_lifecycle_state": "breakout_watch",
        "key_levels": {"resistance": 500.0, "support": 300.0},
    }
    assert _check_structural_invalidation(watch, signal) is None


@patch("utils.watch_candidates.WATCH_KEY_LEVEL_DRIFT_PCT", 2.0)
def test_structural_skips_nonnumeric_stored_level():
    """Non-numeric stored level (string / None) is skipped → None."""
    watch = _make_structural_watch(key_levels={"resistance": "abc", "support": None})
    signal = {
        "setup_lifecycle_state": "breakout_watch",
        "key_levels": {"resistance": 500.0, "support": 300.0},
    }
    assert _check_structural_invalidation(watch, signal) is None


@patch("utils.watch_candidates.WATCH_KEY_LEVEL_DRIFT_PCT", 2.0)
def test_structural_skips_nonnumeric_current_level():
    """Non-numeric / <= 0 current level is skipped → None."""
    watch = _make_structural_watch(key_levels={"resistance": 110.0, "support": 95.0})
    signal = {
        "setup_lifecycle_state": "breakout_watch",
        "key_levels": {"resistance": None, "support": -5.0},
    }
    assert _check_structural_invalidation(watch, signal) is None


@patch("utils.watch_candidates.WATCH_KEY_LEVEL_DRIFT_PCT", 2.0)
def test_structural_skipped_level_does_not_mask_valid_drift():
    """One level skipped (non-numeric) but the other drifts > threshold → key_level_drift."""
    watch = _make_structural_watch(key_levels={"resistance": 110.0, "support": None})
    # support stored is None (skipped); resistance 120 vs 110 → 9.09% > 2%
    signal = {
        "setup_lifecycle_state": "breakout_watch",
        "key_levels": {"resistance": 120.0, "support": 95.0},
    }
    assert _check_structural_invalidation(watch, signal) == "key_level_drift"


# --- Missing signal data → None (fail-open) (Req 3.6) -------------------------


def test_structural_missing_signal_returns_none():
    """Empty signal dict → None (fail-open)."""
    watch = _make_structural_watch()
    assert _check_structural_invalidation(watch, {}) is None


def test_structural_none_signal_returns_none():
    """None signal → None (fail-open)."""
    watch = _make_structural_watch()
    assert _check_structural_invalidation(watch, None) is None


def test_structural_no_lifecycle_no_keylevels_returns_none():
    """Signal with no lifecycle and no key_levels → None (fail-open)."""
    watch = _make_structural_watch()
    signal = {"current_price": 105.0}
    assert _check_structural_invalidation(watch, signal) is None


def test_structural_no_lifecycle_but_drift_still_checked():
    """Missing lifecycle data skips lifecycle check but drift is still evaluated."""
    watch = _make_structural_watch(key_levels={"resistance": 110.0, "support": 95.0})
    # No setup_lifecycle_state → lifecycle check skipped; resistance drifts 9%.
    signal = {"key_levels": {"resistance": 120.0, "support": 95.0}}
    with patch("utils.watch_candidates.WATCH_KEY_LEVEL_DRIFT_PCT", 2.0):
        assert _check_structural_invalidation(watch, signal) == "key_level_drift"


def test_structural_missing_current_key_levels_returns_none():
    """Watchable lifecycle but current signal has no key_levels → None (fail-open)."""
    watch = _make_structural_watch(key_levels={"resistance": 110.0, "support": 95.0})
    signal = {"setup_lifecycle_state": "breakout_watch"}
    assert _check_structural_invalidation(watch, signal) is None


# ─── Same-Cycle Promotion Policy (_check_same_cycle_policy) ───────────────────
#
# Unit tests for _check_same_cycle_policy(watch, current_signal, cycle_id) -> bool.
# Returns True (promotion BLOCKED) / False (promotion ALLOWED).
# Same-cycle means watch.source_cycle_id == cycle_id.
# _make_watch defaults source_cycle_id to "cycle-123".
# Requirements: 4.4, 4.5, 4.6, 4.7, 4.12.


# --- Policy "never" blocks all same-cycle (Req 4.4, 4.7) ----------------------


@patch("utils.watch_candidates.WATCH_SAME_CYCLE_PROMOTION_POLICY", "never")
def test_same_cycle_never_blocks_same_cycle():
    """policy 'never' + same-cycle → blocked regardless of lifecycle."""
    watch = _make_watch()  # source_cycle_id = "cycle-123"
    signal = {"setup_lifecycle_state": "activation_pending"}
    assert _check_same_cycle_policy(watch, signal, "cycle-123") is True


@patch("utils.watch_candidates.WATCH_SAME_CYCLE_PROMOTION_POLICY", "never")
def test_same_cycle_never_allows_different_cycle():
    """policy 'never' but different source_cycle_id → not same-cycle → allowed."""
    watch = _make_watch()  # source_cycle_id = "cycle-123"
    signal = {"setup_lifecycle_state": "activation_pending"}
    assert _check_same_cycle_policy(watch, signal, "cycle-999") is False


# --- Policy "always" allows all same-cycle (Req 4.6, 4.7) ---------------------


@patch("utils.watch_candidates.WATCH_SAME_CYCLE_PROMOTION_POLICY", "always")
def test_same_cycle_always_allows_same_cycle():
    """policy 'always' + same-cycle → allowed regardless of lifecycle."""
    watch = _make_watch()  # source_cycle_id = "cycle-123"
    signal = {"setup_lifecycle_state": "compression_watch"}  # non-pending
    assert _check_same_cycle_policy(watch, signal, "cycle-123") is False


@patch("utils.watch_candidates.WATCH_SAME_CYCLE_PROMOTION_POLICY", "always")
def test_same_cycle_always_allows_missing_lifecycle():
    """policy 'always' + same-cycle + missing lifecycle data → allowed."""
    watch = _make_watch()
    signal = {}  # no setup_lifecycle_state
    assert _check_same_cycle_policy(watch, signal, "cycle-123") is False


# --- Policy "activation_pending_only" is lifecycle-gated (Req 4.5, 4.12) ------


@patch("utils.watch_candidates.WATCH_SAME_CYCLE_PROMOTION_POLICY", "activation_pending_only")
def test_same_cycle_pending_only_allows_matching_lifecycle():
    """policy 'activation_pending_only' + same-cycle + activation_pending → allowed."""
    watch = _make_watch()  # source_cycle_id = "cycle-123"
    signal = {"setup_lifecycle_state": "activation_pending"}
    assert _check_same_cycle_policy(watch, signal, "cycle-123") is False


@patch("utils.watch_candidates.WATCH_SAME_CYCLE_PROMOTION_POLICY", "activation_pending_only")
def test_same_cycle_pending_only_blocks_nonmatching_lifecycle():
    """policy 'activation_pending_only' + same-cycle + non-pending lifecycle → blocked."""
    watch = _make_watch()
    signal = {"setup_lifecycle_state": "compression_watch"}
    assert _check_same_cycle_policy(watch, signal, "cycle-123") is True


@patch("utils.watch_candidates.WATCH_SAME_CYCLE_PROMOTION_POLICY", "activation_pending_only")
def test_same_cycle_pending_only_blocks_missing_lifecycle():
    """policy 'activation_pending_only' + same-cycle + missing lifecycle data → blocked (Req 4.12)."""
    watch = _make_watch()
    signal = {}  # no setup_lifecycle_state
    assert _check_same_cycle_policy(watch, signal, "cycle-123") is True


@patch("utils.watch_candidates.WATCH_SAME_CYCLE_PROMOTION_POLICY", "activation_pending_only")
def test_same_cycle_pending_only_allows_different_cycle():
    """policy 'activation_pending_only' + different source_cycle_id → allowed (not same-cycle)."""
    watch = _make_watch()  # source_cycle_id = "cycle-123"
    signal = {"setup_lifecycle_state": "compression_watch"}
    assert _check_same_cycle_policy(watch, signal, "cycle-999") is False


# --- cycle_id=None skips policy (backward compat) -----------------------------


@patch("utils.watch_candidates.WATCH_SAME_CYCLE_PROMOTION_POLICY", "never")
def test_same_cycle_none_cycle_id_skips_policy():
    """cycle_id=None → policy skipped → allowed even under 'never'."""
    watch = _make_watch()
    signal = {"setup_lifecycle_state": "compression_watch"}
    assert _check_same_cycle_policy(watch, signal, None) is False


# ─── Task 2.10: stale cleanup, TTL sweep, atomic transition, enforcing cycle_id=None
#
# Unit tests for:
#   - expire_stale_promoted_watches()  (Req 1.11)
#   - expire_session_watch_candidates() promoted sweep  (Req 1.13)
#   - _transition_watch_state() atomic promoted_cycle_id + COALESCE  (Req 1.12)
#   - evaluate_active_watch_candidates() enforcing + cycle_id=None  (Req 4.10)

import logging

from utils.watch_candidates import expire_stale_promoted_watches


def _insert_promoted_with_cycle(engine, watch_id, symbol, promoted_cycle_id):
    """Insert a watch and force it into 'promoted' state with a given (or NULL) cycle."""
    watch = _make_watch(symbol=symbol, direction="LONG", watch_id=watch_id)
    _insert_watch_candidate(engine, watch)
    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE watch_candidates "
                "SET state = 'promoted', promoted_cycle_id = :cid "
                "WHERE watch_id = :wid"
            ),
            {"cid": promoted_cycle_id, "wid": watch_id},
        )
        conn.commit()


# --- expire_stale_promoted_watches (Req 1.11) ---------------------------------


def test_expire_stale_promoted_mismatched_cycle(engine):
    """Promoted row with promoted_cycle_id != cycle_id → expired, stale reason."""
    _insert_promoted_with_cycle(engine, "watch-stale-1", "AAPL", "cycle-A")

    expired = expire_stale_promoted_watches(engine, "test_profile", "cycle-B")
    assert expired == 1

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-stale-1"},
        ).fetchone()

    assert row._mapping["state"] == "expired"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_reason"] == "promotion_expired_stale_cycle"


def test_expire_stale_promoted_null_cycle(engine):
    """Promoted row with NULL promoted_cycle_id → expired, stale reason."""
    _insert_promoted_with_cycle(engine, "watch-stale-null", "MSFT", None)

    expired = expire_stale_promoted_watches(engine, "test_profile", "cycle-B")
    assert expired == 1

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-stale-null"},
        ).fetchone()

    assert row._mapping["state"] == "expired"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_reason"] == "promotion_expired_stale_cycle"


def test_expire_stale_promoted_matching_cycle_not_expired(engine):
    """Promoted row whose promoted_cycle_id == current cycle_id → left untouched."""
    _insert_promoted_with_cycle(engine, "watch-current-1", "NVDA", "cycle-A")

    expired = expire_stale_promoted_watches(engine, "test_profile", "cycle-A")
    assert expired == 0

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, promoted_cycle_id FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-current-1"},
        ).fetchone()

    assert row._mapping["state"] == "promoted"
    assert row._mapping["promoted_cycle_id"] == "cycle-A"


def test_expire_stale_promoted_mixed_only_stale_expired(engine):
    """Only stale promoted rows are expired; matching-cycle rows remain promoted."""
    _insert_promoted_with_cycle(engine, "watch-mix-stale", "AAPL", "cycle-old")
    _insert_promoted_with_cycle(engine, "watch-mix-null", "MSFT", None)
    _insert_promoted_with_cycle(engine, "watch-mix-current", "NVDA", "cycle-now")

    expired = expire_stale_promoted_watches(engine, "test_profile", "cycle-now")
    assert expired == 2

    with engine.connect() as conn:
        states = {
            r._mapping["watch_id"]: r._mapping["state"]
            for r in conn.execute(
                text("SELECT watch_id, state FROM watch_candidates")
            ).fetchall()
        }

    assert states["watch-mix-stale"] == "expired"
    assert states["watch-mix-null"] == "expired"
    assert states["watch-mix-current"] == "promoted"


def test_expire_stale_promoted_column_missing_returns_zero():
    """Pre-migration table lacking promoted_cycle_id → fail-open, returns 0."""
    eng = create_engine("sqlite:///:memory:")
    with eng.connect() as conn:
        # Minimal watch_candidates table WITHOUT the promoted_cycle_id column.
        conn.execute(
            text(
                "CREATE TABLE watch_candidates ("
                " watch_id TEXT NOT NULL UNIQUE,"
                " symbol TEXT NOT NULL,"
                " profile_id TEXT NOT NULL,"
                " state TEXT NOT NULL DEFAULT 'active',"
                " outcome_json TEXT,"
                " state_changed_at TEXT,"
                " updated_at TEXT"
                ")"
            )
        )
        conn.execute(
            text(
                "INSERT INTO watch_candidates (watch_id, symbol, profile_id, state) "
                "VALUES ('w-pre', 'AAPL', 'test_profile', 'promoted')"
            )
        )
        conn.commit()

    # Query references the missing column → OperationalError → fail-open return 0.
    expired = expire_stale_promoted_watches(eng, "test_profile", "cycle-B")
    assert expired == 0


# --- expire_session_watch_candidates: promoted sweep (Req 1.13) ---------------


def test_expire_session_sweeps_promoted(engine):
    """A stale 'promoted' row past expires_at is swept to 'expired' via TTL."""
    watch = _make_watch(
        symbol="STALE",
        direction="LONG",
        watch_id="watch-promo-ttl",
        expires_hours=-1,  # already expired
    )
    _insert_watch_candidate(engine, watch)
    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE watch_candidates SET state = 'promoted', "
                "promoted_cycle_id = 'cycle-A' WHERE watch_id = :wid"
            ),
            {"wid": "watch-promo-ttl"},
        )
        conn.commit()

    expired_count = expire_session_watch_candidates(engine, "test_profile")
    assert expired_count == 1

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-promo-ttl"},
        ).fetchone()
    assert row._mapping["state"] == "expired"


def test_expire_session_sweeps_both_active_and_promoted(engine):
    """TTL sweep expires BOTH stale active and stale promoted rows past expires_at."""
    active_watch = _make_watch(
        symbol="ACT", direction="LONG", watch_id="watch-act-ttl", expires_hours=-1
    )
    promoted_watch = _make_watch(
        symbol="PRM", direction="LONG", watch_id="watch-prm-ttl", expires_hours=-1
    )
    _insert_watch_candidate(engine, active_watch)
    _insert_watch_candidate(engine, promoted_watch)
    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE watch_candidates SET state = 'promoted', "
                "promoted_cycle_id = 'cycle-A' WHERE watch_id = :wid"
            ),
            {"wid": "watch-prm-ttl"},
        )
        conn.commit()

    expired_count = expire_session_watch_candidates(engine, "test_profile")
    assert expired_count == 2

    with engine.connect() as conn:
        states = {
            r._mapping["watch_id"]: r._mapping["state"]
            for r in conn.execute(
                text("SELECT watch_id, state FROM watch_candidates")
            ).fetchall()
        }
    assert states["watch-act-ttl"] == "expired"
    assert states["watch-prm-ttl"] == "expired"


# --- _transition_watch_state: atomic promoted_cycle_id + COALESCE (Req 1.12) --


def test_transition_writes_promoted_cycle_id_atomically(engine):
    """active → promoted with promoted_cycle_id parameter writes the column atomically."""
    watch = _make_watch(symbol="ATOM", direction="LONG", watch_id="watch-atom-1")
    _insert_watch_candidate(engine, watch)

    outcome = json.dumps({"terminal_state": "promoted"})
    ok = _transition_watch_state(
        engine, "watch-atom-1", "promoted", outcome, promoted_cycle_id="cycle-X"
    )
    assert ok is True

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT state, promoted_cycle_id FROM watch_candidates WHERE watch_id = :wid"
            ),
            {"wid": "watch-atom-1"},
        ).fetchone()
    assert row._mapping["state"] == "promoted"
    assert row._mapping["promoted_cycle_id"] == "cycle-X"


def test_transition_coalesce_preserves_promoted_cycle_id(engine):
    """A non-promotion transition (promoted→registered) with no cycle preserves the value."""
    watch = _make_watch(symbol="COAL", direction="LONG", watch_id="watch-coal-1")
    _insert_watch_candidate(engine, watch)

    # First: active → promoted with a cycle id.
    _transition_watch_state(
        engine, "watch-coal-1", "promoted",
        json.dumps({"terminal_state": "promoted"}),
        promoted_cycle_id="cycle-keep",
    )
    # Then: promoted → registered without passing promoted_cycle_id (None).
    ok = _transition_watch_state(
        engine, "watch-coal-1", "registered",
        json.dumps({"terminal_state": "registered"}),
        expected_state="promoted",
    )
    assert ok is True

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT state, promoted_cycle_id FROM watch_candidates WHERE watch_id = :wid"
            ),
            {"wid": "watch-coal-1"},
        ).fetchone()
    assert row._mapping["state"] == "registered"
    # COALESCE(NULL, existing) preserved the original promoted_cycle_id.
    assert row._mapping["promoted_cycle_id"] == "cycle-keep"


def test_transition_non_promotion_leaves_cycle_id_null(engine):
    """active → invalidated with promoted_cycle_id=None leaves the column NULL."""
    watch = _make_watch(symbol="NULLC", direction="LONG", watch_id="watch-nullc-1")
    _insert_watch_candidate(engine, watch)

    ok = _transition_watch_state(
        engine, "watch-nullc-1", "invalidated",
        json.dumps({"terminal_state": "invalidated"}),
    )
    assert ok is True

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT promoted_cycle_id FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-nullc-1"},
        ).fetchone()
    assert row._mapping["promoted_cycle_id"] is None


# --- evaluate_active_watch_candidates: enforcing + cycle_id=None (Req 4.10) ---


@patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing")
def test_evaluate_enforcing_none_cycle_blocks_promotion(engine, caplog):
    """Enforcing + cycle_id=None: activation detected but NO promotion, WARNING logged."""
    watch = _make_watch(
        symbol="BLOCK",
        direction="LONG",
        watch_id="watch-block-1",
        activation=[{"condition": "price > above resistance", "threshold": 110.0}],
        invalidation=[{"condition": "price below support", "threshold": 95.0}],
    )
    _insert_watch_candidate(engine, watch)

    signals = {
        "BLOCK": {
            "current_price": 115.0,  # activation crossed
            "signal": "LONG",
            "setup_type": "technical_breakout",
        }
    }

    with caplog.at_level(logging.WARNING, logger="utils.watch_candidates"):
        counts = evaluate_active_watch_candidates(engine, signals, "test_profile")

    # No promotion occurred; watch remains active.
    assert counts["promotion_eligible"] == 0
    assert counts["still_active"] == 1

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT state, promoted_cycle_id FROM watch_candidates WHERE watch_id = :wid"
            ),
            {"wid": "watch-block-1"},
        ).fetchone()
    assert row._mapping["state"] == "active"
    assert row._mapping["promoted_cycle_id"] is None
    assert "cycle_id is required" in caplog.text


@patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing")
def test_evaluate_enforcing_none_cycle_still_invalidates(engine):
    """Enforcing + cycle_id=None: invalidation checks still run (price-threshold)."""
    watch = _make_watch(
        symbol="INVAL",
        direction="LONG",
        watch_id="watch-inval-1",
        activation=[{"condition": "price > above resistance", "threshold": 110.0}],
        invalidation=[{"condition": "price below support", "threshold": 95.0}],
    )
    _insert_watch_candidate(engine, watch)

    signals = {"INVAL": {"current_price": 90.0}}  # below support → invalidated
    counts = evaluate_active_watch_candidates(engine, signals, "test_profile")

    assert counts["invalidated"] == 1

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-inval-1"},
        ).fetchone()
    assert row._mapping["state"] == "invalidated"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_reason"] == "invalidation_threshold_crossed"


@patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing")
def test_evaluate_enforcing_none_cycle_still_structurally_invalidates(engine):
    """Enforcing + cycle_id=None: structural invalidation still runs."""
    watch = _make_watch(
        symbol="STRUCT",
        direction="LONG",
        watch_id="watch-struct-eval-1",
        activation=[{"condition": "price > above resistance", "threshold": 110.0}],
        invalidation=[{"condition": "price below support", "threshold": 95.0}],
    )
    _insert_watch_candidate(engine, watch)

    signals = {
        "STRUCT": {
            "current_price": 115.0,
            "setup_lifecycle_state": "no_setup",  # not watchable → structural degradation
        }
    }
    counts = evaluate_active_watch_candidates(engine, signals, "test_profile")

    assert counts["invalidated"] == 1

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": "watch-struct-eval-1"},
        ).fetchone()
    assert row._mapping["state"] == "invalidated"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_reason"] == "structural_degradation"
