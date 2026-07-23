"""Property-based tests for the watch-candidate hardening feature.

Feature: watch-candidate-hardening
These tests validate the universal correctness properties enumerated in
design.md §"Correctness Properties". This file currently implements
Properties 1-10; Properties 11-20 are appended in a later batch, so each
property lives in its own clearly-named test function and all shared imports
and helpers are defined at the top of the module.

CRITICAL Hypothesis + DB isolation note:
    A function-scoped pytest ``engine`` fixture must NOT be bound to an
    ``@given`` test — Hypothesis reuses the single fixture instance across all
    generated examples, leaking state between examples. Instead we build a fresh
    in-memory engine per example via the module-level ``_new_engine()`` helper,
    called INSIDE each test body.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from hypothesis import given, strategies as st, settings, assume
from sqlalchemy import create_engine, inspect as sa_inspect, text

from orchestrator import _ensure_watch_candidate_tables
from utils.gate_config import CANDIDATE_EXECUTABLE_SETUP_TYPES
from utils.market_state import WATCHABLE_LIFECYCLE_STATES
from utils.candidate_builder import _process_promoted_watch, build_candidate_set
from utils.candidate_registry import CandidateRegistry, CandidateRegistryError
from utils.watch_candidates import (
    WatchCandidate,
    _insert_watch_candidate,
    _transition_watch_state,
    _check_structural_invalidation,
    _check_same_cycle_policy,
    evaluate_active_watch_candidates,
    get_promotable_candidates,
    expire_stale_promoted_watches,
    expire_session_watch_candidates,
)

# Sorted lists for deterministic Hypothesis sampling.
EXECUTABLE_SETUP_TYPES = sorted(CANDIDATE_EXECUTABLE_SETUP_TYPES)
PROFILE_ID = "test_profile"

# A representative set of lifecycle states that are NOT watchable (used to drive
# structural-degradation properties).
NON_WATCHABLE_LIFECYCLE_STATES = [
    "no_setup",
    "confounded",
    "triggered",
    "invalidated_setup",
    "breakout_confirmed",
    "exhausted",
    "none",
]


# ---------------------------------------------------------------------------
# Module-level helpers (called INSIDE each test body — never bound as fixtures)
# ---------------------------------------------------------------------------


def _create_pm_candidates_table(engine) -> None:
    """Create the pm_candidates table (mirrors the integration-test schema)."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE pm_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
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
                    state TEXT NOT NULL,
                    integrity_hash TEXT NOT NULL,
                    execution_key TEXT,
                    reserved_at TEXT,
                    created_at TEXT,
                    expires_at TEXT NOT NULL,
                    context_snapshot_json TEXT,
                    benchmark_mapping_json TEXT,
                    rejection_reason TEXT,
                    candidate_lineage_id TEXT,
                    candidate_type TEXT DEFAULT 'intraday',
                    holding_horizon INTEGER,
                    normalized_setup_type TEXT
                )
                """
            )
        )


def _new_engine(with_pm: bool = False):
    """Create a fresh in-memory SQLite engine with the watch_candidates table.

    Args:
        with_pm: when True, also create the pm_candidates table (needed by the
            promotion / dedup properties).
    """
    eng = create_engine("sqlite:///:memory:")
    _ensure_watch_candidate_tables(eng, sa_inspect(eng))
    if with_pm:
        _create_pm_candidates_table(eng)
    return eng


def _make_watch(
    symbol: str = "AAPL",
    profile_id: str = PROFILE_ID,
    direction: str = "LONG",
    watch_id: str | None = None,
    key_levels: dict | None = None,
    source_cycle_id: str = "cycle-src",
    activation: list | None = None,
    invalidation: list | None = None,
    expires_hours: int = 7,
) -> WatchCandidate:
    """Build a test WatchCandidate row object."""
    now = datetime.now(timezone.utc)
    if key_levels is None:
        key_levels = {"resistance": 110.0, "support": 95.0}
    return WatchCandidate(
        watch_id=watch_id or f"watch-{uuid.uuid4()}",
        symbol=symbol,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        expires_at=(now + timedelta(hours=expires_hours)).isoformat(),
        source_cycle_id=source_cycle_id,
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
        key_levels_json=json.dumps(key_levels),
        trigger_status_json=json.dumps({}),
        reason="prop test watch",
        source_signal_snapshot_json=json.dumps(
            {"signal": "LONG", "setup_type": "technical_breakout"}
        ),
        state="active",
    )


def _insert_promoted(
    engine,
    watch_id: str,
    symbol: str,
    promoted_cycle_id: str | None,
    profile_id: str = PROFILE_ID,
    state: str = "promoted",
) -> None:
    """Insert a watch and force it into a target state with a promoted_cycle_id."""
    watch = _make_watch(symbol=symbol, watch_id=watch_id, profile_id=profile_id)
    _insert_watch_candidate(engine, watch)
    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE watch_candidates "
                "SET state = :st, promoted_cycle_id = :cid "
                "WHERE watch_id = :wid"
            ),
            {"st": state, "cid": promoted_cycle_id, "wid": watch_id},
        )
        conn.commit()


def _insert_pm_candidate(
    engine, candidate_id, cycle_id, profile_id, symbol, source_signal_id
) -> None:
    """Insert a minimal registered pm_candidates row (used for dedup tests)."""
    now = datetime.now(timezone.utc).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pm_candidates
                (candidate_id, cycle_id, profile_id, symbol, direction, setup_type,
                 geometry_name, entry_price, stop_price, target_price, risk_reward,
                 source_signal_id, signal_snapshot_json, state, integrity_hash,
                 created_at, expires_at, candidate_type)
                VALUES
                (:candidate_id, :cycle_id, :profile_id, :symbol, 'BUY', 'momentum_fade',
                 'base_breakout', 150.0, 148.0, 154.0, 2.0,
                 :source_signal_id, '{}', 'registered', 'hash-existing',
                 :now, :future, 'intraday')
                """
            ),
            {
                "candidate_id": candidate_id,
                "cycle_id": cycle_id,
                "profile_id": profile_id,
                "symbol": symbol,
                "source_signal_id": source_signal_id,
                "now": now,
                "future": future,
            },
        )


def _read_watch(engine, watch_id):
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT state, outcome_json FROM watch_candidates WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).fetchone()


def _executable_signal(symbol, strength="strong", direction="LONG", setup="technical_breakout"):
    """A directional, executable-setup-type signal accepted by get_promotable."""
    return {
        "symbol": symbol,
        "signal": direction,
        "strength": strength,
        "setup_type": setup,
        "current_price": 150.0,
    }


def _ok_scaffold(signal, profile_id=None, profile_context=None, **kwargs):
    """A valid geometry scaffold with exactly one candidate."""
    return {
        "symbol": signal.get("symbol", "AAA"),
        "direction": "LONG",
        "status": "ok",
        "candidates": [
            {
                "name": "base_breakout",
                "entry_price": 150.0,
                "stop_loss": 148.0,
                "target": 154.0,
                "risk_reward": 2.0,
                "trigger": "Price breaks above",
                "invalidation_basis": "Falls below stop",
                "target_basis": "Entry + RR * risk",
            }
        ],
    }


def _failed_scaffold(signal, profile_id=None, profile_context=None, **kwargs):
    """A non-ok geometry scaffold (status != 'ok')."""
    return {
        "symbol": signal.get("symbol", "AAA"),
        "direction": "LONG",
        "status": "error",
        "reason": "insufficient_geometry",
        "candidates": [],
    }


def _empty_ok_scaffold(signal, profile_id=None, profile_context=None, **kwargs):
    """A valid-status scaffold that nonetheless yields zero candidates."""
    return {
        "symbol": signal.get("symbol", "AAA"),
        "direction": "LONG",
        "status": "ok",
        "candidates": [],
    }


CYCLES = ["cycle-A", "cycle-B", "cycle-C"]


# ===========================================================================
# Property 1: Promoted-cycle scoping excludes cross-cycle watches
# Validates: Requirements 1.1, 1.6
# ===========================================================================


@given(
    assignments=st.lists(st.sampled_from(CYCLES), min_size=1, max_size=8),
    target=st.sampled_from(CYCLES),
)
@settings(max_examples=50)
def test_prop1_promoted_cycle_scoping_excludes_cross_cycle(assignments, target):
    """get_promotable_candidates(cycle_id=X) returns only rows with promoted_cycle_id == X."""
    with patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing"):
        engine = _new_engine()
        signals = {}
        expected = set()
        for i, cid in enumerate(assignments):
            symbol = f"SYM{i}"
            wid = f"w-{i}"
            _insert_promoted(engine, wid, symbol, cid)
            # Every symbol has a directional + executable signal so the only
            # thing that can exclude a row is the cycle scoping.
            signals[symbol] = _executable_signal(symbol)
            if cid == target:
                expected.add(wid)

        result = get_promotable_candidates(engine, signals, PROFILE_ID, cycle_id=target)
        returned = {r["watch_id"] for r in result}
        assert returned == expected


# ===========================================================================
# Property 2: Registered state is query-invisible
# Validates: Requirements 1.4
# ===========================================================================


@given(
    states=st.lists(
        st.sampled_from(["active", "promoted", "registered", "invalidated", "expired"]),
        min_size=1,
        max_size=8,
    ),
)
@settings(max_examples=50)
def test_prop2_registered_state_is_query_invisible(states):
    """A 'registered' watch is never returned by get_promotable_candidates nor
    counted among active/promoted rows."""
    cycle = "cycle-A"
    with patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing"):
        engine = _new_engine()
        signals = {}
        registered_wids = set()
        for i, stt in enumerate(states):
            symbol = f"SYM{i}"
            wid = f"w-{i}"
            _insert_promoted(engine, wid, symbol, cycle, state=stt)
            signals[symbol] = _executable_signal(symbol)
            if stt == "registered":
                registered_wids.add(wid)

        result = get_promotable_candidates(engine, signals, PROFILE_ID, cycle_id=cycle)
        returned = {r["watch_id"] for r in result}
        assert returned.isdisjoint(registered_wids)

        # Registered watches must also never appear in an active/promoted query.
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT watch_id FROM watch_candidates "
                    "WHERE state IN ('active', 'promoted')"
                )
            ).fetchall()
        active_or_promoted = {r[0] for r in rows}
        assert active_or_promoted.isdisjoint(registered_wids)


# ===========================================================================
# Property 3: Successful promotion produces registered terminal state
# Validates: Requirements 1.2
# ===========================================================================


@given(
    strength=st.sampled_from(["moderate", "strong"]),
    setup=st.sampled_from(EXECUTABLE_SETUP_TYPES),
)
@settings(max_examples=50)
def test_prop3_successful_promotion_produces_registered(strength, setup):
    """A promoted watch that completes geometry + register transitions to 'registered'."""
    engine = _new_engine(with_pm=True)
    cycle = "cycle-A"
    wid = f"w-{uuid.uuid4()}"
    symbol = "AAA"
    _insert_promoted(engine, wid, symbol, cycle)
    registry = CandidateRegistry(engine, cycle, PROFILE_ID)
    promo = {
        "watch_id": wid,
        "symbol": symbol,
        "signal": _executable_signal(symbol, strength=strength, setup=setup),
    }

    with patch("utils.candidate_builder.build_entry_geometry_scaffold", _ok_scaffold):
        _process_promoted_watch(
            engine, promo, registry, set(), "moderate", PROFILE_ID, cycle, None
        )

    row = _read_watch(engine, wid)
    assert row._mapping["state"] == "registered"
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM pm_candidates WHERE source_signal_id = :w"),
            {"w": wid},
        ).fetchone()[0]
    assert count == 1


# ===========================================================================
# Property 4: Idempotent promotion consumption
# Validates: Requirements 1.7
# ===========================================================================


@given(
    strength=st.sampled_from(["moderate", "strong"]),
    setup=st.sampled_from(EXECUTABLE_SETUP_TYPES),
)
@settings(max_examples=50)
def test_prop4_idempotent_promotion_consumption(strength, setup):
    """A pre-existing PM candidate for (watch_id, profile, cycle) → skip register,
    no duplicate row, and the watch transitions to 'registered'."""
    engine = _new_engine(with_pm=True)
    cycle = "cycle-A"
    wid = f"w-{uuid.uuid4()}"
    symbol = "AAA"
    _insert_promoted(engine, wid, symbol, cycle)
    # Pre-insert the PM candidate the dedup check will find.
    _insert_pm_candidate(
        engine, str(uuid.uuid4()), cycle, PROFILE_ID, symbol, source_signal_id=wid
    )
    registry = CandidateRegistry(engine, cycle, PROFILE_ID)
    promo = {
        "watch_id": wid,
        "symbol": symbol,
        "signal": _executable_signal(symbol, strength=strength, setup=setup),
    }

    # Geometry is patched to ok so that if dedup DID NOT short-circuit, a
    # duplicate would be created — proving the dedup path when count stays 1.
    with patch("utils.candidate_builder.build_entry_geometry_scaffold", _ok_scaffold):
        _process_promoted_watch(
            engine, promo, registry, set(), "moderate", PROFILE_ID, cycle, None
        )

    row = _read_watch(engine, wid)
    assert row._mapping["state"] == "registered"
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM pm_candidates WHERE source_signal_id = :w"),
            {"w": wid},
        ).fetchone()[0]
    assert count == 1


# ===========================================================================
# Property 5: Enforcing mode requires cycle_id for get_promotable_candidates
# Validates: Requirements 1.8
# ===========================================================================


@given(
    assignments=st.lists(st.sampled_from(CYCLES), min_size=0, max_size=8),
)
@settings(max_examples=50)
def test_prop5_enforcing_mode_requires_cycle_id(assignments):
    """In enforcing mode, cycle_id=None → empty list regardless of promoted rows."""
    with patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing"):
        engine = _new_engine()
        signals = {}
        for i, cid in enumerate(assignments):
            symbol = f"SYM{i}"
            wid = f"w-{i}"
            _insert_promoted(engine, wid, symbol, cid)
            signals[symbol] = _executable_signal(symbol)

        result = get_promotable_candidates(engine, signals, PROFILE_ID)
        assert result == []


# ===========================================================================
# Property 6: Held-symbol exclusion blocks promotion with short-circuit
# Validates: Requirements 2.1, 2.3, 2.4
# ===========================================================================


@given(
    strength=st.sampled_from(["weak", "moderate", "strong"]),
    setup=st.sampled_from(EXECUTABLE_SETUP_TYPES),
)
@settings(max_examples=50)
def test_prop6_held_symbol_exclusion_short_circuits(strength, setup):
    """A held symbol → promotion_blocked_held_symbol, no PM candidate, and the
    strength check is never evaluated (short-circuit even for a weak signal)."""
    engine = _new_engine(with_pm=True)
    cycle = "cycle-A"
    wid = f"w-{uuid.uuid4()}"
    symbol = "AAA"
    _insert_promoted(engine, wid, symbol, cycle)
    registry = CandidateRegistry(engine, cycle, PROFILE_ID)
    promo = {
        "watch_id": wid,
        "symbol": symbol,
        "signal": _executable_signal(symbol, strength=strength, setup=setup),
    }

    with patch("utils.candidate_builder.build_entry_geometry_scaffold", _ok_scaffold):
        _process_promoted_watch(
            engine, promo, registry, {symbol}, "moderate", PROFILE_ID, cycle, None
        )

    row = _read_watch(engine, wid)
    assert row._mapping["state"] == "expired"
    outcome = json.loads(row._mapping["outcome_json"])
    # Held-symbol wins the short-circuit even when the signal is also weak.
    assert outcome["terminal_reason"] == "promotion_blocked_held_symbol"

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM pm_candidates WHERE source_signal_id = :w"),
            {"w": wid},
        ).fetchone()[0]
    assert count == 0


# ===========================================================================
# Property 7: Signal strength threshold blocks weak promotions
# Validates: Requirements 2.2, 2.5
# ===========================================================================


@given(
    pair=st.sampled_from([("strong", "weak"), ("strong", "moderate"), ("moderate", "weak")]),
    setup=st.sampled_from(EXECUTABLE_SETUP_TYPES),
)
@settings(max_examples=50)
def test_prop7_signal_strength_threshold_blocks_weak(pair, setup):
    """A signal strength below threshold (and symbol not held) →
    promotion_blocked_weak_signal, no PM candidate."""
    threshold, strength = pair
    engine = _new_engine(with_pm=True)
    cycle = "cycle-A"
    wid = f"w-{uuid.uuid4()}"
    symbol = "AAA"
    _insert_promoted(engine, wid, symbol, cycle)
    registry = CandidateRegistry(engine, cycle, PROFILE_ID)
    promo = {
        "watch_id": wid,
        "symbol": symbol,
        "signal": _executable_signal(symbol, strength=strength, setup=setup),
    }

    with patch("utils.candidate_builder.build_entry_geometry_scaffold", _ok_scaffold):
        _process_promoted_watch(
            engine, promo, registry, set(), threshold, PROFILE_ID, cycle, None
        )

    row = _read_watch(engine, wid)
    assert row._mapping["state"] == "expired"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_reason"] == "promotion_blocked_weak_signal"

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM pm_candidates WHERE source_signal_id = :w"),
            {"w": wid},
        ).fetchone()[0]
    assert count == 0


# ===========================================================================
# Property 8: Promotion loop terminal failures produce correct terminal reasons
# Validates: Requirements 2.8, 2.9, 2.10
# ===========================================================================


@given(
    mode=st.sampled_from(["geometry_failed", "no_candidates", "registry_error"]),
    setup=st.sampled_from(EXECUTABLE_SETUP_TYPES),
)
@settings(max_examples=50)
def test_prop8_promotion_loop_terminal_failures(mode, setup):
    """Geometry not-ok, empty candidates, and registry errors each map to their
    dedicated terminal_reason."""
    engine = _new_engine(with_pm=True)
    cycle = "cycle-A"
    wid = f"w-{uuid.uuid4()}"
    symbol = "AAA"
    _insert_promoted(engine, wid, symbol, cycle)
    registry = CandidateRegistry(engine, cycle, PROFILE_ID)
    # Strong + not-held so eligibility passes and we reach geometry/registry.
    promo = {
        "watch_id": wid,
        "symbol": symbol,
        "signal": _executable_signal(symbol, strength="strong", setup=setup),
    }

    expected = {
        "geometry_failed": "promotion_blocked_geometry_failed",
        "no_candidates": "promotion_blocked_no_geometry_candidates",
        "registry_error": "promotion_blocked_registry_error",
    }[mode]

    if mode == "geometry_failed":
        with patch("utils.candidate_builder.build_entry_geometry_scaffold", _failed_scaffold):
            _process_promoted_watch(
                engine, promo, registry, set(), "moderate", PROFILE_ID, cycle, None
            )
    elif mode == "no_candidates":
        with patch("utils.candidate_builder.build_entry_geometry_scaffold", _empty_ok_scaffold):
            _process_promoted_watch(
                engine, promo, registry, set(), "moderate", PROFILE_ID, cycle, None
            )
    else:  # registry_error
        def _raise_register(self, candidate):
            raise CandidateRegistryError("simulated registry failure")

        with patch("utils.candidate_builder.build_entry_geometry_scaffold", _ok_scaffold), \
             patch.object(CandidateRegistry, "register", _raise_register):
            _process_promoted_watch(
                engine, promo, registry, set(), "moderate", PROFILE_ID, cycle, None
            )

    row = _read_watch(engine, wid)
    assert row._mapping["state"] == "expired"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_reason"] == expected


# ===========================================================================
# Property 9: Structural degradation invalidates active watches
# Validates: Requirements 3.1, 3.2, 3.5
# ===========================================================================


@given(
    lifecycle=st.one_of(
        st.sampled_from(NON_WATCHABLE_LIFECYCLE_STATES),
        st.text(min_size=1, max_size=12),
    ),
)
@settings(max_examples=50)
def test_prop9_structural_degradation_invalidates(lifecycle):
    """A current lifecycle not in WATCHABLE_LIFECYCLE_STATES → structural_degradation
    from the pure check AND an 'invalidated' transition through evaluate()."""
    assume(lifecycle and lifecycle not in WATCHABLE_LIFECYCLE_STATES)

    # Pure-function check: lifecycle regression is detected regardless of price.
    watch = _make_watch(key_levels={"resistance": 110.0, "support": 95.0})
    signal = {
        "setup_lifecycle_state": lifecycle,
        "key_levels": {"resistance": 110.0, "support": 95.0},
    }
    assert _check_structural_invalidation(watch, signal) == "structural_degradation"

    # Integration through evaluate_active_watch_candidates → 'invalidated'.
    with patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing"):
        engine = _new_engine()
        wid = f"w-{uuid.uuid4()}"
        symbol = "AAA"
        _insert_watch_candidate(
            engine, _make_watch(symbol=symbol, watch_id=wid)
        )
        signals = {symbol: {"setup_lifecycle_state": lifecycle, "current_price": 115.0}}
        counts = evaluate_active_watch_candidates(
            engine, signals, PROFILE_ID, cycle_id="cycle-A"
        )
        assert counts["invalidated"] == 1
        row = _read_watch(engine, wid)
        assert row._mapping["state"] == "invalidated"
        outcome = json.loads(row._mapping["outcome_json"])
        assert outcome["terminal_reason"] == "structural_degradation"


# ===========================================================================
# Property 10: Key-level drift invalidates active watches
# Validates: Requirements 3.3, 3.4
# ===========================================================================


@given(
    stored=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    drift_pct=st.floats(min_value=2.01, max_value=200.0, allow_nan=False, allow_infinity=False),
    which=st.sampled_from(["support", "resistance"]),
    sign=st.sampled_from([1, -1]),
)
@settings(max_examples=50)
def test_prop10_key_level_drift_invalidates(stored, drift_pct, which, sign):
    """A support/resistance drift exceeding WATCH_KEY_LEVEL_DRIFT_PCT →
    key_level_drift from the pure check AND an 'invalidated' evaluate() transition."""
    current = stored * (1 + sign * drift_pct / 100.0)
    # Skip cases where the drifted value would be non-positive — those are
    # intentionally excluded by the drift math (covered by Property 11).
    assume(current > 0)

    stored_levels = {"support": stored, "resistance": stored}
    current_levels = {"support": stored, "resistance": stored}
    current_levels[which] = current

    with patch("utils.watch_candidates.WATCH_KEY_LEVEL_DRIFT_PCT", 2.0):
        # Pure-function check.
        watch = _make_watch(key_levels=stored_levels)
        signal = {"setup_lifecycle_state": "breakout_watch", "key_levels": current_levels}
        assert _check_structural_invalidation(watch, signal) == "key_level_drift"

        # Integration through evaluate_active_watch_candidates → 'invalidated'.
        with patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing"):
            engine = _new_engine()
            wid = f"w-{uuid.uuid4()}"
            symbol = "AAA"
            _insert_watch_candidate(
                engine,
                _make_watch(symbol=symbol, watch_id=wid, key_levels=stored_levels),
            )
            signals = {
                symbol: {
                    "setup_lifecycle_state": "breakout_watch",
                    "key_levels": current_levels,
                    "current_price": stored,
                }
            }
            counts = evaluate_active_watch_candidates(
                engine, signals, PROFILE_ID, cycle_id="cycle-A"
            )
            assert counts["invalidated"] == 1
            row = _read_watch(engine, wid)
            assert row._mapping["state"] == "invalidated"
            outcome = json.loads(row._mapping["outcome_json"])
            assert outcome["terminal_reason"] == "key_level_drift"


# ===========================================================================
# Property 11: Drift computation skips non-numeric and non-positive levels
# Validates: Requirements 3.8
# ===========================================================================


@given(
    good_stored=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    which_bad=st.sampled_from(["support", "resistance"]),
    bad_in=st.sampled_from(["stored", "current"]),
    bad_value=st.sampled_from([None, "not-a-number", 0.0, -0.01, -5.0, True, False]),
    good_drifts=st.booleans(),
)
@settings(max_examples=200)
def test_prop11_drift_skips_non_numeric_and_non_positive(
    good_stored, which_bad, bad_in, bad_value, good_drifts
):
    """A level whose stored or current value is non-numeric or <= 0 is skipped
    from the drift comparison (never causes invalidation), and skipping it must
    not mask real drift on the OTHER level."""
    which_good = "resistance" if which_bad == "support" else "support"

    # The good level either drifts far beyond threshold (50%) or not at all.
    current_good = good_stored * 1.5 if good_drifts else good_stored

    stored_levels = {which_good: good_stored, which_bad: good_stored}
    current_levels = {which_good: current_good, which_bad: good_stored}

    # Inject the "bad" value on either the stored or current side of the bad
    # level. The other side of the bad level stays a large valid number so that,
    # were the bad level NOT skipped, it would register huge drift.
    if bad_in == "stored":
        stored_levels[which_bad] = bad_value
        current_levels[which_bad] = good_stored * 10.0
    else:
        stored_levels[which_bad] = good_stored * 10.0
        current_levels[which_bad] = bad_value

    with patch("utils.watch_candidates.WATCH_KEY_LEVEL_DRIFT_PCT", 2.0):
        watch = _make_watch(key_levels=stored_levels)
        # Watchable lifecycle so structural degradation never fires — isolate drift.
        signal = {"setup_lifecycle_state": "breakout_watch", "key_levels": current_levels}
        result = _check_structural_invalidation(watch, signal)

    if good_drifts:
        # Real drift on the good level must still be detected (skip didn't mask it).
        assert result == "key_level_drift"
    else:
        # Bad level skipped and good level stable → no invalidation.
        assert result is None


# ===========================================================================
# Property 12: Same-cycle "never" policy blocks all same-cycle promotions
# Validates: Requirements 4.4, 4.7
# ===========================================================================


@given(
    source_cycle=st.sampled_from(CYCLES),
    cycle_id=st.one_of(st.sampled_from(CYCLES), st.none()),
    lifecycle=st.sampled_from(
        ["activation_pending", "breakout_watch", "compression_watch", "", "no_setup"]
    ),
)
@settings(max_examples=200)
def test_prop12_same_cycle_never_policy_blocks(source_cycle, cycle_id, lifecycle):
    """policy='never': _check_same_cycle_policy returns True (blocked) whenever
    source_cycle_id == cycle_id, and False when they differ (or cycle_id None)."""
    watch = _make_watch(source_cycle_id=source_cycle)
    signal = {"setup_lifecycle_state": lifecycle}

    with patch("utils.watch_candidates.WATCH_SAME_CYCLE_PROMOTION_POLICY", "never"):
        blocked = _check_same_cycle_policy(watch, signal, cycle_id)

    if cycle_id is None:
        # Backward-compat: no cycle context → policy skipped (allowed).
        assert blocked is False
    elif source_cycle == cycle_id:
        assert blocked is True
    else:
        assert blocked is False


# ===========================================================================
# Property 13: Same-cycle "activation_pending_only" policy is lifecycle-gated
# Validates: Requirements 4.5, 4.7, 4.12
# ===========================================================================


@given(
    source_cycle=st.sampled_from(CYCLES),
    cycle_id=st.one_of(st.sampled_from(CYCLES), st.none()),
    lifecycle=st.sampled_from(
        ["activation_pending", "breakout_watch", "compression_watch", "no_setup", ""]
    ),
    include_lifecycle=st.booleans(),
)
@settings(max_examples=200)
def test_prop13_same_cycle_activation_pending_only(
    source_cycle, cycle_id, lifecycle, include_lifecycle
):
    """policy='activation_pending_only': same-cycle → allowed iff current signal
    lifecycle == 'activation_pending' (missing lifecycle → blocked); a different
    cycle is always allowed; cycle_id None is always allowed."""
    watch = _make_watch(source_cycle_id=source_cycle)
    signal = {"setup_lifecycle_state": lifecycle} if include_lifecycle else {}

    with patch(
        "utils.watch_candidates.WATCH_SAME_CYCLE_PROMOTION_POLICY",
        "activation_pending_only",
    ):
        blocked = _check_same_cycle_policy(watch, signal, cycle_id)

    if cycle_id is None or source_cycle != cycle_id:
        # Not same-cycle (or no cycle context) → policy does not block.
        assert blocked is False
    else:
        # Same-cycle: allowed only when lifecycle explicitly activation_pending.
        should_allow = include_lifecycle and lifecycle == "activation_pending"
        assert blocked is (not should_allow)


# ===========================================================================
# Property 14: Evaluation order — structural precedes price, same-cycle policy
#              gates only after activation
# Validates: Requirements 3.5, 4.13
# ===========================================================================


@given(
    lifecycle=st.sampled_from(NON_WATCHABLE_LIFECYCLE_STATES),
    policy=st.sampled_from(["never", "activation_pending_only", "always"]),
    same_cycle=st.booleans(),
)
@settings(max_examples=50)
def test_prop14_structural_precedes_price_and_policy(lifecycle, policy, same_cycle):
    """When structural degradation co-occurs with a crossed activation threshold,
    the watch is invalidated (structural) — price/promotion logic never runs, and
    the same-cycle policy never pre-empts structural invalidation."""
    cycle = "cycle-A"
    source_cycle = cycle if same_cycle else "cycle-OTHER"

    with patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing"), \
         patch("utils.watch_candidates.WATCH_SAME_CYCLE_PROMOTION_POLICY", policy):
        engine = _new_engine()
        wid = f"w-{uuid.uuid4()}"
        symbol = "AAA"
        _insert_watch_candidate(
            engine,
            _make_watch(symbol=symbol, watch_id=wid, source_cycle_id=source_cycle),
        )
        # Price is WAY above the activation threshold (110) → activation would
        # cross; but a non-watchable lifecycle forces structural degradation first.
        signals = {
            symbol: {
                "setup_lifecycle_state": lifecycle,
                "current_price": 999.0,
                "signal": "LONG",
                "setup_type": "technical_breakout",
            }
        }
        counts = evaluate_active_watch_candidates(
            engine, signals, PROFILE_ID, cycle_id=cycle
        )

    assert counts["invalidated"] == 1
    assert counts["promotion_eligible"] == 0
    row = _read_watch(engine, wid)
    assert row._mapping["state"] == "invalidated"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_reason"] == "structural_degradation"


# ===========================================================================
# Property 15: Disabled mode executes no hardening logic
# Validates: Requirements 5.1
# ===========================================================================


@given(
    n_watches=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=50)
def test_prop15_disabled_mode_no_hardening(n_watches):
    """MARKET_STATE_MODE='disabled': build_candidate_set performs no watch
    creation/evaluation/promotion (the watch_candidates functions are never
    called) and existing watch rows are untouched."""
    engine = _new_engine(with_pm=True)

    # Seed active watch rows with unique symbols.
    wids = []
    for i in range(n_watches):
        wid = f"w-{i}-{uuid.uuid4()}"
        _insert_watch_candidate(engine, _make_watch(symbol=f"SYM{i}", watch_id=wid))
        wids.append(wid)

    signals = {f"SYM{i}": _executable_signal(f"SYM{i}") for i in range(n_watches)}
    profile = {"min_signal_strength": "moderate"}
    portfolio = {"positions": []}

    mock_eval = MagicMock()
    mock_create = MagicMock()
    mock_promotable = MagicMock(return_value=[])
    mock_stale = MagicMock(return_value=0)

    with patch("utils.gate_config.MARKET_STATE_MODE", "disabled"), \
         patch("utils.watch_candidates.MARKET_STATE_MODE", "disabled"), \
         patch("utils.candidate_builder.build_entry_geometry_scaffold", _failed_scaffold), \
         patch("utils.watch_candidates.evaluate_active_watch_candidates", mock_eval), \
         patch("utils.watch_candidates.evaluate_and_create_watch_candidates", mock_create), \
         patch("utils.watch_candidates.get_promotable_candidates", mock_promotable), \
         patch("utils.watch_candidates.expire_stale_promoted_watches", mock_stale):
        build_candidate_set(
            engine, signals, PROFILE_ID, profile, portfolio, "cycle-A"
        )

    # No hardening function was invoked.
    assert mock_eval.call_count == 0
    assert mock_create.call_count == 0
    assert mock_promotable.call_count == 0
    assert mock_stale.call_count == 0

    # Every seeded watch row is untouched (still active).
    for wid in wids:
        row = _read_watch(engine, wid)
        assert row._mapping["state"] == "active"


# ===========================================================================
# Property 16: Hardening exceptions are fail-open (evaluation)
# Validates: Requirements 5.4
# ===========================================================================


@given(
    which=st.sampled_from(["structural", "same_cycle"]),
)
@settings(max_examples=50)
def test_prop16_hardening_exceptions_fail_open(which):
    """If _check_structural_invalidation or _check_same_cycle_policy raises,
    evaluate_active_watch_candidates does not crash, still returns counts, and
    leaves the watch in a safe (uncorrupted) state."""
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated hardening failure")

    cycle = "cycle-A"
    with patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing"):
        engine = _new_engine()
        wid = f"w-{uuid.uuid4()}"
        symbol = "AAA"
        _insert_watch_candidate(engine, _make_watch(symbol=symbol, watch_id=wid))

        if which == "structural":
            # No activation crossing → after fail-open the watch stays active.
            signals = {
                symbol: {"setup_lifecycle_state": "breakout_watch", "current_price": 100.0}
            }
            patch_target = "utils.watch_candidates._check_structural_invalidation"
        else:
            # Activation threshold crossed so the same-cycle policy path runs.
            signals = {
                symbol: {
                    "setup_lifecycle_state": "breakout_watch",
                    "current_price": 999.0,
                    "signal": "LONG",
                    "setup_type": "technical_breakout",
                }
            }
            patch_target = "utils.watch_candidates._check_same_cycle_policy"

        with patch(patch_target, _boom):
            counts = evaluate_active_watch_candidates(
                engine, signals, PROFILE_ID, cycle_id=cycle
            )

    # No exception propagated; a valid counts dict is returned.
    assert isinstance(counts, dict)
    assert {"invalidated", "promotion_eligible", "still_active"} <= set(counts)

    # The watch is in a valid (non-corrupted) terminal/transient state.
    row = _read_watch(engine, wid)
    assert row._mapping["state"] in {"active", "promoted", "invalidated", "expired"}


# ===========================================================================
# Property 17: Eligibility exceptions are fail-closed (promotion)
# Validates: Requirements 5.5
# ===========================================================================


@given(
    setup=st.sampled_from(EXECUTABLE_SETUP_TYPES),
    strength=st.sampled_from(["moderate", "strong"]),
)
@settings(max_examples=50)
def test_prop17_eligibility_exceptions_fail_closed(setup, strength):
    """If the eligibility check raises inside _process_promoted_watch, the watch
    is expired with terminal_reason 'promotion_blocked_eligibility_error' and no
    PM candidate is created (fail-closed)."""
    engine = _new_engine(with_pm=True)
    cycle = "cycle-A"
    wid = f"w-{uuid.uuid4()}"
    symbol = "AAA"
    _insert_promoted(engine, wid, symbol, cycle)
    registry = CandidateRegistry(engine, cycle, PROFILE_ID)
    promo = {
        "watch_id": wid,
        "symbol": symbol,
        "signal": _executable_signal(symbol, strength=strength, setup=setup),
    }

    def _raise_threshold(*args, **kwargs):
        raise RuntimeError("simulated eligibility failure")

    # Symbol NOT held so the strength check (which calls _meets_threshold) runs
    # and raises — the fail-closed path must expire the watch.
    with patch("utils.candidate_builder._meets_threshold", _raise_threshold), \
         patch("utils.candidate_builder.build_entry_geometry_scaffold", _ok_scaffold):
        _process_promoted_watch(
            engine, promo, registry, set(), "moderate", PROFILE_ID, cycle, None
        )

    row = _read_watch(engine, wid)
    assert row._mapping["state"] == "expired"
    outcome = json.loads(row._mapping["outcome_json"])
    assert outcome["terminal_reason"] == "promotion_blocked_eligibility_error"

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM pm_candidates WHERE source_signal_id = :w"),
            {"w": wid},
        ).fetchone()[0]
    assert count == 0


# ===========================================================================
# Property 18: Stale promoted watches are expired at cycle start
# Validates: Requirements 1.11
# ===========================================================================


@given(
    assignments=st.lists(
        st.one_of(st.sampled_from(CYCLES), st.none()), min_size=1, max_size=8
    ),
    target=st.sampled_from(CYCLES),
)
@settings(max_examples=50)
def test_prop18_stale_promoted_expired_at_cycle_start(assignments, target):
    """expire_stale_promoted_watches expires every promoted row whose
    promoted_cycle_id != target (including NULL) with promotion_expired_stale_cycle,
    and leaves matching-cycle rows promoted."""
    engine = _new_engine()
    stale_wids = set()
    matching_wids = set()
    for i, cid in enumerate(assignments):
        wid = f"w-{i}"
        _insert_promoted(engine, wid, f"SYM{i}", cid)
        if cid == target:
            matching_wids.add(wid)
        else:
            stale_wids.add(wid)

    expired = expire_stale_promoted_watches(engine, PROFILE_ID, target)
    assert expired == len(stale_wids)

    for wid in stale_wids:
        row = _read_watch(engine, wid)
        assert row._mapping["state"] == "expired"
        outcome = json.loads(row._mapping["outcome_json"])
        assert outcome["terminal_reason"] == "promotion_expired_stale_cycle"

    for wid in matching_wids:
        row = _read_watch(engine, wid)
        assert row._mapping["state"] == "promoted"


# ===========================================================================
# Property 19: Enforcing mode + cycle_id=None blocks all promotion
# Validates: Requirements 4.10
# ===========================================================================


@given(
    specs=st.lists(st.sampled_from(["activate", "invalidate"]), min_size=1, max_size=8),
)
@settings(max_examples=50)
def test_prop19_enforcing_cycle_id_none_blocks_promotion(specs):
    """In enforcing mode with cycle_id=None, no active watch is transitioned to
    'promoted' (activated ones stay active) while invalidation still occurs."""
    with patch("utils.watch_candidates.MARKET_STATE_MODE", "enforcing"):
        engine = _new_engine()
        signals = {}
        activate_wids = set()
        invalidate_wids = set()
        for i, kind in enumerate(specs):
            wid = f"w-{i}"
            symbol = f"SYM{i}"
            _insert_watch_candidate(engine, _make_watch(symbol=symbol, watch_id=wid))
            if kind == "activate":
                # Price crosses activation threshold (>110); executable + directional.
                signals[symbol] = {
                    "setup_lifecycle_state": "breakout_watch",
                    "current_price": 200.0,
                    "signal": "LONG",
                    "setup_type": "technical_breakout",
                }
                activate_wids.add(wid)
            else:
                # Price crosses invalidation threshold (<95, below support).
                signals[symbol] = {
                    "setup_lifecycle_state": "breakout_watch",
                    "current_price": 50.0,
                }
                invalidate_wids.add(wid)

        counts = evaluate_active_watch_candidates(
            engine, signals, PROFILE_ID, cycle_id=None
        )

    # Nothing promoted; invalidations still happened.
    with engine.connect() as conn:
        promoted = conn.execute(
            text("SELECT COUNT(*) FROM watch_candidates WHERE state = 'promoted'")
        ).fetchone()[0]
    assert promoted == 0

    for wid in activate_wids:
        assert _read_watch(engine, wid)._mapping["state"] == "active"
    for wid in invalidate_wids:
        assert _read_watch(engine, wid)._mapping["state"] == "invalidated"
    assert counts["invalidated"] == len(invalidate_wids)


# ===========================================================================
# Property 20: TTL sweep expires stale promoted rows
# Validates: Requirements 1.13
# ===========================================================================


@given(
    specs=st.lists(
        st.tuples(
            st.sampled_from(["active", "promoted"]),
            st.booleans(),  # is_past
        ),
        min_size=1,
        max_size=8,
    ),
)
@settings(max_examples=50)
def test_prop20_ttl_sweep_expires_stale_promoted(specs):
    """expire_session_watch_candidates sweeps promoted (and active) rows past
    expires_at to expired; rows not past expires_at are untouched."""
    engine = _new_engine()
    past_wids = set()
    live_wids = {}  # wid -> original state
    for i, (state, is_past) in enumerate(specs):
        wid = f"w-{i}"
        symbol = f"SYM{i}"
        expires_hours = -1 if is_past else 5
        _insert_watch_candidate(
            engine, _make_watch(symbol=symbol, watch_id=wid, expires_hours=expires_hours)
        )
        if state == "promoted":
            with engine.connect() as conn:
                conn.execute(
                    text("UPDATE watch_candidates SET state = 'promoted' WHERE watch_id = :w"),
                    {"w": wid},
                )
                conn.commit()
        if is_past:
            past_wids.add(wid)
        else:
            live_wids[wid] = state

    expire_session_watch_candidates(engine, PROFILE_ID)

    for wid in past_wids:
        assert _read_watch(engine, wid)._mapping["state"] == "expired"
    for wid, original in live_wids.items():
        assert _read_watch(engine, wid)._mapping["state"] == original
