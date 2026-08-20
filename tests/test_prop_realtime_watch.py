"""
Property-based tests for Realtime Watch Maturity feature.

Tests correctness properties from the design document using Hypothesis.
Feature: realtime-watch-maturity

Properties tested:
  4: _safe_decimal Never Raises for Any Input
  1: Side-Consistency Matching
  2: Maturity Evidence Annotated on Flipped Conditions
  5: Entry Zone Requires Two+ Valid Levels
  6: Draft Geometry Directional Consistency
  7: Entry Zone Replacement Iff Strictly Tighter
  8: Missed-Move Detection Side-Symmetric
  10: Source Provenance Validation Asymmetric
  3: State Advance Iff Score Meets Threshold
  9: Promotion Snapshot All Required Keys
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.setup_watch_evaluator import _safe_decimal
from utils.setup_watch_bridge import (
    _is_side_consistent,
    _record_maturity_evidence,
)
from utils.draft_geometry import (
    compute_entry_zone,
    compute_draft_geometry,
    should_replace_entry_zone,
    EntryZone,
    DraftGeometry,
    SWING_SETUP_TYPES,
)
from utils.missed_move_detector import (
    check_missed_move,
    check_target_crossed_for_pending_order,
    MissedMoveResult,
)
from utils.setup_watch_manager import _validate_source_provenance
from utils.setup_watch_registry import WatchState, SetupWatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_watch(
    watch_id: str = "sw_test",
    state: WatchState = WatchState.READY,
    side: str = "BUY",
    draft_geometry_json: str | None = None,
    **kwargs,
) -> SetupWatch:
    """Construct a minimal SetupWatch for testing."""
    now = datetime.now(timezone.utc)
    defaults = dict(
        watch_id=watch_id,
        profile_id="profile_1",
        symbol="TEST",
        side=side,
        setup_type="support_bounce_swing",
        state=state,
        thesis="test thesis",
        source_type="analyst",
        source_id="sig_001",
        source_cycle_id="cycle_001",
        maturation_conditions_json='[{"type":"price_zone","weight":1}]',
        invalidation_conditions_json='[{"type":"price_below","level":100}]',
        last_evaluation_json=None,
        entry_zone_json=None,
        draft_geometry_json=draft_geometry_json,
        maturity_score=0.0,
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(hours=48),
        state_changed_at=now,
        observed_cycles=3,
        ready_at=now,
        ready_reference_price=150.0,
        terminal_reason=None,
        promoted_cycle_id=None,
        execution_ref_type=None,
        execution_ref_id=None,
        integrity_hash="hash_test",
    )
    defaults.update(kwargs)
    return SetupWatch(**defaults)


# ---------------------------------------------------------------------------
# Property 4: _safe_decimal Never Raises for Any Input
# **Validates: Requirements 10.1, 10.2, 10.4, 2.7, 2.9, 3.6**
# ---------------------------------------------------------------------------


@given(
    value=st.one_of(
        st.none(),
        st.integers(),
        st.floats(),
        st.text(),
        st.binary(),
        st.dictionaries(st.text(), st.text()),
        st.lists(st.integers()),
    )
)
@settings(max_examples=200)
def test_safe_decimal_never_raises(value):
    """_safe_decimal must return Decimal or None, never raise, for any input.

    **Validates: Requirements 10.1, 10.2, 10.4, 2.7, 2.9, 3.6**
    """
    result = _safe_decimal(value)
    assert result is None or isinstance(result, Decimal)


# ---------------------------------------------------------------------------
# Property 1: Side-Consistency Matching
# **Validates: Requirements 1.1**
# ---------------------------------------------------------------------------

# Strategy: valid setup types including the special ones
st_setup_type = st.sampled_from([
    "support_bounce_swing",
    "pullback_continuation",
    "momentum_fade",
    "breakout_retest",
    "failed_breakdown_reclaim",
])

st_level_name = st.sampled_from([
    "support", "resistance", "vwap", "ma_20", "moving_average",
])

st_side = st.sampled_from(["BUY", "SHORT"])


@given(
    side=st_side,
    setup_type=st_setup_type,
    level_name=st_level_name,
    price=st.decimals(min_value="0.01", max_value="10000", places=2),
    level_value=st.decimals(min_value="0.01", max_value="10000", places=2),
)
@settings(max_examples=200)
def test_side_consistency_directional(side, setup_type, level_name, price, level_value):
    """_is_side_consistent returns True iff alert direction is favorable per rules.

    Default: BUY matches price > level; SHORT matches price < level.
    breakout_retest BUY: matches price < level (resistance from below).
    breakout_retest SHORT: matches price > level (support from above).
    failed_breakdown_reclaim BUY: matches price > level (reclaiming).
    failed_breakdown_reclaim SHORT: uses default (price < level).

    **Validates: Requirements 1.1**
    """
    # Skip when price == level_value (boundary isn't directional)
    assume(price != level_value)

    result = _is_side_consistent(side, setup_type, level_name, price, level_value)

    # Verify against documented rules
    if setup_type == "breakout_retest":
        if side == "BUY":
            expected = price < level_value
        else:
            expected = price > level_value
    elif setup_type == "failed_breakdown_reclaim":
        if side == "BUY":
            expected = price > level_value
        else:
            expected = price < level_value
    else:
        # Default matching
        if side == "BUY":
            expected = price > level_value
        else:
            expected = price < level_value

    assert result == expected, (
        f"side={side}, setup_type={setup_type}, level_name={level_name}, "
        f"price={price}, level_value={level_value}: expected={expected}, got={result}"
    )


# ---------------------------------------------------------------------------
# Property 2: Maturity Evidence Annotated on Flipped Conditions
# **Validates: Requirements 1.3**
# ---------------------------------------------------------------------------

@dataclass
class _MockConditionResult:
    """Minimal condition result for _record_maturity_evidence."""
    condition_type: str
    met: bool
    detail: str = ""


@dataclass
class _MockEvaluationResult:
    """Minimal evaluation result."""
    condition_results: list


@given(
    num_conditions=st.integers(min_value=1, max_value=5),
    prev_met_mask=st.lists(st.booleans(), min_size=1, max_size=5),
    new_met_mask=st.lists(st.booleans(), min_size=1, max_size=5),
)
@settings(max_examples=200)
def test_maturity_evidence_flipped_conditions(num_conditions, prev_met_mask, new_met_mask):
    """Evidence count equals number of conditions that flipped False→True.

    _record_maturity_evidence returns the count of conditions that transitioned
    from unmet in previous evaluation to met in new evaluation.

    **Validates: Requirements 1.3**
    """
    # Normalize masks to same length
    n = min(num_conditions, len(prev_met_mask), len(new_met_mask))
    assume(n >= 1)

    condition_types = [f"cond_{i}" for i in range(n)]

    # Build previous evaluation
    previous_eval = {
        "condition_results": [
            {"condition_type": condition_types[i], "met": prev_met_mask[i]}
            for i in range(n)
        ]
    }

    # Build new evaluation result
    new_results = [
        _MockConditionResult(
            condition_type=condition_types[i],
            met=new_met_mask[i],
            detail=f"detail_{i}",
        )
        for i in range(n)
    ]
    new_eval = _MockEvaluationResult(condition_results=new_results)

    alert = {
        "price": 150.0,
        "level_name": "support",
        "level_value": 148.0,
        "distance_pct": 1.3,
    }

    flipped_count = _record_maturity_evidence(previous_eval, new_eval, alert)

    # Count expected flips: was False, now True
    expected_flips = sum(
        1 for i in range(n)
        if not prev_met_mask[i] and new_met_mask[i]
    )

    assert flipped_count == expected_flips


# ---------------------------------------------------------------------------
# Property 5: Entry Zone Requires Two+ Valid Levels
# **Validates: Requirements 2.1, 2.4**
# ---------------------------------------------------------------------------

@given(
    key_levels=st.dictionaries(
        st.sampled_from(["support", "resistance", "vwap", "ma_20"]),
        st.one_of(
            st.none(),
            st.floats(min_value=0.01, max_value=10000, allow_nan=False, allow_infinity=False),
            st.text(max_size=10),
        ),
    ),
    side=st_side,
)
@settings(max_examples=200)
def test_entry_zone_requires_two_valid(key_levels, side):
    """compute_entry_zone returns non-None iff 2+ values parse to valid Decimals.

    **Validates: Requirements 2.1, 2.4**
    """
    result = compute_entry_zone(key_levels, side)

    # Count how many values can be parsed via _safe_decimal
    valid_count = 0
    for v in key_levels.values():
        if isinstance(v, list):
            for item in v:
                if _safe_decimal(item) is not None:
                    valid_count += 1
        else:
            if _safe_decimal(v) is not None:
                valid_count += 1

    if valid_count < 2:
        assert result is None, (
            f"Expected None with {valid_count} valid levels, got {result}"
        )
    else:
        # With 2+ valid levels, result should be non-None UNLESS all values
        # are identical (low == high is rejected)
        valid_values = []
        for v in key_levels.values():
            if isinstance(v, list):
                for item in v:
                    parsed = _safe_decimal(item)
                    if parsed is not None:
                        valid_values.append(parsed)
            else:
                parsed = _safe_decimal(v)
                if parsed is not None:
                    valid_values.append(parsed)
        if len(set(valid_values)) >= 2:
            assert result is not None, (
                f"Expected EntryZone with {valid_count} valid distinct levels, "
                f"got None. key_levels={key_levels}"
            )
            assert isinstance(result, EntryZone)
            assert result.low <= result.high
        # If all values are identical, None is acceptable


# ---------------------------------------------------------------------------
# Property 6: Draft Geometry Directional Consistency
# **Validates: Requirements 2.3**
# ---------------------------------------------------------------------------

@given(
    side=st_side,
    support=st.decimals(min_value="10", max_value="500", places=2),
    resistance=st.decimals(min_value="10", max_value="500", places=2),
)
@settings(max_examples=200)
def test_draft_geometry_directional_consistency(side, support, resistance):
    """Any non-None DraftGeometry satisfies directional ordering invariant.

    BUY: stop < entry < target. SHORT: stop > entry > target.

    **Validates: Requirements 2.3**
    """
    # Ensure there's a meaningful spread
    assume(abs(support - resistance) > Decimal("1"))
    if side == "BUY":
        assume(support < resistance)
    else:
        assume(resistance < support)

    key_levels = {
        "support": float(support),
        "resistance": float(resistance),
    }

    result = compute_draft_geometry(
        key_levels, "support_bounce_swing", side
    )

    if result is not None:
        if side == "BUY":
            assert result.stop < result.entry < result.target, (
                f"BUY invariant violated: stop={result.stop}, "
                f"entry={result.entry}, target={result.target}"
            )
        else:
            assert result.stop > result.entry > result.target, (
                f"SHORT invariant violated: stop={result.stop}, "
                f"entry={result.entry}, target={result.target}"
            )


# ---------------------------------------------------------------------------
# Property 7: Entry Zone Replacement Iff Strictly Tighter
# **Validates: Requirements 2.5**
# ---------------------------------------------------------------------------

@given(
    old_low=st.decimals(min_value="1", max_value="400", places=2),
    old_spread=st.decimals(min_value="0.01", max_value="100", places=2),
    new_low=st.decimals(min_value="1", max_value="400", places=2),
    new_spread=st.decimals(min_value="0.01", max_value="100", places=2),
)
@settings(max_examples=200)
def test_entry_zone_replacement_iff_strictly_tighter(old_low, old_spread, new_low, new_spread):
    """should_replace_entry_zone returns True iff new zone is strictly tighter.

    Strictly tighter means (new_high - new_low) < (old_high - old_low).
    Equal widths do NOT trigger replacement.

    **Validates: Requirements 2.5**
    """
    old_high = old_low + old_spread
    new_high = new_low + new_spread

    existing_json = json.dumps({
        "low": str(old_low),
        "high": str(old_high),
    })
    new_zone = EntryZone(low=new_low, high=new_high)

    result = should_replace_entry_zone(existing_json, new_zone)

    old_width = old_high - old_low
    new_width = new_high - new_low

    expected = new_width < old_width
    assert result == expected, (
        f"old_width={old_width}, new_width={new_width}: "
        f"expected={expected}, got={result}"
    )


# ---------------------------------------------------------------------------
# Property 8: Missed-Move Detection Side-Symmetric
# **Validates: Requirements 4.2, 4.3, 7.2**
# ---------------------------------------------------------------------------

@given(
    target=st.decimals(min_value="0.01", max_value="10000", places=2),
    price=st.decimals(min_value="0.01", max_value="10000", places=2),
    side=st_side,
)
@settings(max_examples=200)
def test_missed_move_side_symmetric(target, price, side):
    """BUY missed iff price >= target; SHORT missed iff price <= target.

    This property holds for both check_missed_move (bridge) and
    check_target_crossed_for_pending_order (pending-order guard) when
    geometry is valid.

    **Validates: Requirements 4.2, 4.3, 7.2**
    """
    geometry_json = json.dumps({
        "entry": str(target - Decimal("5") if side == "BUY" else target + Decimal("5")),
        "stop": str(target - Decimal("10") if side == "BUY" else target + Decimal("10")),
        "target": str(target),
        "risk_reward": "2.00",
    })

    watch = _make_watch(
        state=WatchState.READY,
        side=side,
        draft_geometry_json=geometry_json,
    )

    # Test check_missed_move
    result = check_missed_move(watch, str(price))

    if side == "BUY":
        expected_missed = price >= target
    else:
        expected_missed = price <= target

    assert result.missed == expected_missed, (
        f"check_missed_move: side={side}, price={price}, target={target}: "
        f"expected_missed={expected_missed}, got={result.missed}"
    )

    # Test check_target_crossed_for_pending_order (same logic)
    result_pending = check_target_crossed_for_pending_order(watch, str(price))

    assert result_pending.missed == expected_missed, (
        f"check_target_crossed: side={side}, price={price}, target={target}: "
        f"expected_missed={expected_missed}, got={result_pending.missed}"
    )


# ---------------------------------------------------------------------------
# Property 10: Source Provenance Validation Asymmetric
# **Validates: Requirements 11.6, 11.8**
# ---------------------------------------------------------------------------

_KNOWN_SOURCE_TYPES = frozenset({
    "analyst", "candidate_reject", "market_state", "pm_defer",
    "price_monitor", "scout",
})
_LEGACY_SOURCE_TYPES = frozenset({
    "analyst", "candidate_reject", "market_state", "pm_defer",
})
_NEW_SOURCE_TYPES = _KNOWN_SOURCE_TYPES - _LEGACY_SOURCE_TYPES


@given(
    source_type=st.one_of(
        st.sampled_from(sorted(_KNOWN_SOURCE_TYPES)),
        st.text(min_size=1, max_size=20),
    ),
    source_id=st.one_of(st.none(), st.text(min_size=1, max_size=30)),
)
@settings(max_examples=200)
def test_source_provenance_validation_asymmetric(source_type, source_id):
    """Source provenance validation follows asymmetric rules:

    - Unknown type → reject (False)
    - New type + null id → reject (False)
    - Legacy type + null id → accept (True)
    - Valid type + non-null id → accept (True)

    **Validates: Requirements 11.6, 11.8**
    """
    result = _validate_source_provenance(source_type, source_id)

    if source_type not in _KNOWN_SOURCE_TYPES:
        # Unknown type → always rejected
        assert result is False, (
            f"Unknown source_type={source_type!r} should be rejected"
        )
    elif source_type in _NEW_SOURCE_TYPES and not source_id:
        # New type with null/empty source_id → rejected
        assert result is False, (
            f"New source_type={source_type!r} with null source_id should be rejected"
        )
    elif source_type in _LEGACY_SOURCE_TYPES and not source_id:
        # Legacy type with null source_id → accepted
        assert result is True, (
            f"Legacy source_type={source_type!r} with null source_id should be accepted"
        )
    else:
        # Valid type with non-null source_id → accepted
        assert result is True, (
            f"source_type={source_type!r} with source_id={source_id!r} should be accepted"
        )


# ---------------------------------------------------------------------------
# Property 3: State Advance Iff Score Meets Threshold
# **Validates: Requirements 1.4**
# ---------------------------------------------------------------------------

@given(
    current_state=st.sampled_from([WatchState.WATCHING, WatchState.MATURING]),
    score=st.floats(min_value=-0.1, max_value=1.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_state_advance_iff_score_meets_threshold(current_state, score):
    """Transition attempted iff score crosses boundary for current state.

    WATCHING → MATURING: iff score > 0
    MATURING → READY: iff score >= SETUP_WATCH_MATURITY_THRESHOLD (0.7)

    **Validates: Requirements 1.4**
    """
    from utils.gate_config import SETUP_WATCH_MATURITY_THRESHOLD

    watch = _make_watch(state=current_state, maturity_score=0.0)

    # Mock the registry to track whether transition_state is called
    mock_registry = MagicMock()
    mock_registry.transition_state = MagicMock()

    from utils.setup_watch_bridge import _attempt_state_advance

    result = _attempt_state_advance(
        mock_registry, watch, score, Decimal("150.00")
    )

    if current_state == WatchState.WATCHING:
        if score > 0:
            # Transition should be attempted
            assert mock_registry.transition_state.called or result is not None
        else:
            # No transition
            assert result is None
            assert not mock_registry.transition_state.called
    elif current_state == WatchState.MATURING:
        if score >= SETUP_WATCH_MATURITY_THRESHOLD:
            # Transition should be attempted
            assert mock_registry.transition_state.called or result is not None
        else:
            # No transition
            assert result is None
            assert not mock_registry.transition_state.called


# ---------------------------------------------------------------------------
# Property 9: Promotion Snapshot All Required Keys
# **Validates: Requirements 6.1–6.7**
# ---------------------------------------------------------------------------

_REQUIRED_SNAPSHOT_KEYS = frozenset({
    "source_type",
    "watch_id",
    "setup_type",
    "thesis",
    "maturity_score",
    "observed_cycles",
    "last_evaluation_json",
    "ready_reference_price",
    "ready_at",
    "source_trigger",
    "entry_zone",
    "draft_geometry",
    "current_price",
    "key_levels",
    "source_provenance",
})


@given(
    has_eval=st.booleans(),
    has_geometry=st.booleans(),
    has_entry_zone=st.booleans(),
    has_signal=st.booleans(),
)
@settings(max_examples=200)
def test_promotion_snapshot_all_required_keys(
    has_eval, has_geometry, has_entry_zone, has_signal
):
    """Promoted watch snapshot contains all required keys. NULL fields present as null.

    **Validates: Requirements 6.1–6.7**
    """
    from utils.setup_watch_manager import _build_evidence_package

    now = datetime.now(timezone.utc)

    watch = _make_watch(
        state=WatchState.PROMOTED,
        last_evaluation_json=json.dumps({
            "maturity_score": 0.85,
            "conditions": [{"type": "price_zone", "met": True}],
        }) if has_eval else None,
        draft_geometry_json=json.dumps({
            "entry": "149.50",
            "stop": "147.00",
            "target": "155.00",
            "risk_reward": "2.20",
        }) if has_geometry else None,
        entry_zone_json=json.dumps({
            "low": "148.00",
            "high": "150.00",
        }) if has_entry_zone else None,
        maturity_score=0.85,
        ready_at=now,
        ready_reference_price=149.50,
    )

    signals = None
    if has_signal:
        signals = {
            "TEST": {
                "current_price": 149.80,
                "key_levels": {"support": 148.0, "resistance": 152.0},
            }
        }

    evidence = _build_evidence_package(watch, "cycle_001", signals)

    # All required keys must be present (even if value is None/null)
    for key in _REQUIRED_SNAPSHOT_KEYS:
        assert key in evidence, (
            f"Required key '{key}' missing from evidence package. "
            f"Present keys: {sorted(evidence.keys())}"
        )

    # Verify NULL fields are present as None (JSON null), never omitted
    if not has_eval:
        assert "last_evaluation_json" in evidence
        assert evidence["last_evaluation_json"] is None
    if not has_geometry:
        assert "draft_geometry" in evidence
        assert evidence["draft_geometry"] is None
    if not has_entry_zone:
        assert "entry_zone" in evidence
        assert evidence["entry_zone"] is None
    if not has_signal:
        assert "current_price" in evidence
        assert evidence["current_price"] is None
        assert "key_levels" in evidence
        assert evidence["key_levels"] is None

    # source_provenance must always be a dict with required sub-keys
    assert isinstance(evidence["source_provenance"], dict)
    assert "source_type" in evidence["source_provenance"]
    assert "source_id" in evidence["source_provenance"]
    assert "source_cycle_id" in evidence["source_provenance"]
