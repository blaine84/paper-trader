"""Unit tests for utils/missed_move_detector.py.

Tests: check_missed_move, apply_missed_move_transition,
       check_target_crossed_for_pending_order, MissedMoveResult

Requirements: 4.1-4.9, 7.1-7.8
"""
from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from utils.missed_move_detector import (
    MissedMoveResult,
    apply_missed_move_transition,
    check_missed_move,
    check_target_crossed_for_pending_order,
)
from utils.setup_watch_registry import SetupWatchRegistryError, WatchState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_watch(
    state=WatchState.READY,
    side="BUY",
    draft_geometry_json=None,
    watch_id="w-1",
    ready_at=None,
):
    """Create a mock watch with the minimal fields needed by the detector."""
    return SimpleNamespace(
        watch_id=watch_id,
        state=state,
        side=side,
        draft_geometry_json=draft_geometry_json,
        ready_at=ready_at,
        symbol="AAPL",
        profile_id="p1",
        maturity_score=0.8,
    )


def _geometry_json(target, entry="100.00", stop="95.00"):
    """Helper to build a valid draft_geometry_json string."""
    return json.dumps({"entry": entry, "stop": stop, "target": str(target)})


# ---------------------------------------------------------------------------
# 14.2 BUY: missed when current_price >= target; not missed when < target
# ---------------------------------------------------------------------------

class TestBuyDirectionalLogic:
    """Req 4.2: BUY missed iff current_price >= target."""

    def test_buy_missed_when_price_at_target(self):
        watch = _make_watch(side="BUY", draft_geometry_json=_geometry_json("150.00"))
        result = check_missed_move(watch, "150.00")
        assert result.missed is True
        assert result.reason == "target_already_crossed"

    def test_buy_missed_when_price_above_target(self):
        watch = _make_watch(side="BUY", draft_geometry_json=_geometry_json("150.00"))
        result = check_missed_move(watch, "155.00")
        assert result.missed is True
        assert result.reason == "target_already_crossed"

    def test_buy_not_missed_when_price_below_target(self):
        watch = _make_watch(side="BUY", draft_geometry_json=_geometry_json("150.00"))
        result = check_missed_move(watch, "149.99")
        assert result.missed is False
        assert result.reason is None


# ---------------------------------------------------------------------------
# 14.3 SHORT: missed when current_price <= target; not missed when > target
# ---------------------------------------------------------------------------

class TestShortDirectionalLogic:
    """Req 4.3: SHORT missed iff current_price <= target."""

    def test_short_missed_when_price_at_target(self):
        watch = _make_watch(side="SHORT", draft_geometry_json=_geometry_json("90.00"))
        result = check_missed_move(watch, "90.00")
        assert result.missed is True
        assert result.reason == "target_already_crossed"

    def test_short_missed_when_price_below_target(self):
        watch = _make_watch(side="SHORT", draft_geometry_json=_geometry_json("90.00"))
        result = check_missed_move(watch, "85.00")
        assert result.missed is True
        assert result.reason == "target_already_crossed"

    def test_short_not_missed_when_price_above_target(self):
        watch = _make_watch(side="SHORT", draft_geometry_json=_geometry_json("90.00"))
        result = check_missed_move(watch, "90.01")
        assert result.missed is False
        assert result.reason is None


# ---------------------------------------------------------------------------
# 14.4 Watch in WATCHING state → skipped (returns missed=False)
# ---------------------------------------------------------------------------

class TestStateFilteringWatching:
    """Req 4.4: Only READY/PROMOTED evaluated; WATCHING skipped."""

    def test_watching_state_skipped(self):
        watch = _make_watch(
            state=WatchState.WATCHING,
            side="BUY",
            draft_geometry_json=_geometry_json("150.00"),
        )
        result = check_missed_move(watch, "200.00")  # way above target
        assert result.missed is False
        assert result.target_price is None
        assert result.current_price is None


# ---------------------------------------------------------------------------
# 14.5 Watch in MATURING state → skipped (returns missed=False)
# ---------------------------------------------------------------------------

class TestStateFilteringMaturing:
    """Req 4.4: Only READY/PROMOTED evaluated; MATURING skipped."""

    def test_maturing_state_skipped(self):
        watch = _make_watch(
            state=WatchState.MATURING,
            side="SHORT",
            draft_geometry_json=_geometry_json("50.00"),
        )
        result = check_missed_move(watch, "10.00")  # way below target
        assert result.missed is False
        assert result.target_price is None
        assert result.current_price is None


# ---------------------------------------------------------------------------
# 14.6 No draft_geometry_json (NULL) → returns missed=False
# ---------------------------------------------------------------------------

class TestNullGeometry:
    """Req 4.6: No geometry means skip check."""

    def test_null_geometry_returns_not_missed(self):
        watch = _make_watch(side="BUY", draft_geometry_json=None)
        result = check_missed_move(watch, "999.99")
        assert result.missed is False
        assert result.target_price is None


# ---------------------------------------------------------------------------
# 14.7 draft_geometry_json present but JSON parse fails → missed=False
# ---------------------------------------------------------------------------

class TestMalformedGeometryBridgePath:
    """Req 4.6: JSON parse failure on bridge path → skip (fail-open)."""

    def test_invalid_json_returns_not_missed(self):
        watch = _make_watch(side="BUY", draft_geometry_json="not valid json {{")
        result = check_missed_move(watch, "200.00")
        assert result.missed is False
        assert result.target_price is None


# ---------------------------------------------------------------------------
# 14.8 draft_geometry_json valid but no 'target' key → missed=False
# ---------------------------------------------------------------------------

class TestGeometryNoTargetKey:
    """Req 4.6: Valid JSON without target key → skip check."""

    def test_no_target_key_returns_not_missed(self):
        geom = json.dumps({"entry": "100.00", "stop": "95.00"})
        watch = _make_watch(side="BUY", draft_geometry_json=geom)
        result = check_missed_move(watch, "200.00")
        assert result.missed is False
        assert result.target_price is None


# ---------------------------------------------------------------------------
# 14.9 apply_missed_move_transition succeeds on CAS rowcount == 1
# ---------------------------------------------------------------------------

class TestApplyTransitionSuccess:
    """Req 4.5: CAS succeeds → watch transitioned to MISSED."""

    def test_apply_transition_returns_true_on_success(self):
        registry = MagicMock()
        registry.transition_state = MagicMock()  # succeeds (no exception)

        watch = _make_watch(side="BUY", draft_geometry_json=_geometry_json("150.00"))
        result = MissedMoveResult(
            watch_id="w-1",
            missed=True,
            target_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            side="BUY",
            reason="target_already_crossed",
        )

        success = apply_missed_move_transition(registry, watch, result)
        assert success is True
        registry.transition_state.assert_called_once_with(
            "w-1",
            WatchState.READY,
            WatchState.MISSED,
            terminal_reason="target_already_crossed",
        )

    def test_apply_transition_returns_false_when_not_missed(self):
        registry = MagicMock()
        watch = _make_watch(side="BUY")
        result = MissedMoveResult(
            watch_id="w-1",
            missed=False,
            target_price=None,
            current_price=None,
            side="BUY",
            reason=None,
        )

        success = apply_missed_move_transition(registry, watch, result)
        assert success is False
        registry.transition_state.assert_not_called()


# ---------------------------------------------------------------------------
# 14.10 apply_missed_move_transition logs WARNING on CAS failure, no raise
# ---------------------------------------------------------------------------

class TestApplyTransitionCASFailure:
    """Req 4.9: CAS fails → log WARNING, return False, do not raise."""

    def test_cas_failure_does_not_raise(self):
        registry = MagicMock()
        registry.transition_state.side_effect = SetupWatchRegistryError("CAS failed")

        watch = _make_watch(side="BUY", draft_geometry_json=_geometry_json("150.00"))
        result = MissedMoveResult(
            watch_id="w-1",
            missed=True,
            target_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            side="BUY",
            reason="target_already_crossed",
        )

        # Should not raise
        success = apply_missed_move_transition(registry, watch, result)
        assert success is False


# ---------------------------------------------------------------------------
# 14.11 check_target_crossed_for_pending_order: NULL geometry → missed=False
# ---------------------------------------------------------------------------

class TestPendingOrderNullGeometry:
    """Req 7.6: NULL geometry → skip check, allow order."""

    def test_null_geometry_allows_order(self):
        watch = _make_watch(side="BUY", draft_geometry_json=None)
        result = check_target_crossed_for_pending_order(watch, "200.00")
        assert result.missed is False
        assert result.reason is None


# ---------------------------------------------------------------------------
# 14.12 check_target_crossed_for_pending_order: malformed → missed=True
# ---------------------------------------------------------------------------

class TestPendingOrderMalformedGeometry:
    """Req 7.6: Malformed geometry → fail-closed, missed=True."""

    def test_malformed_json_fails_closed(self):
        watch = _make_watch(side="BUY", draft_geometry_json="not json!!")
        result = check_target_crossed_for_pending_order(watch, "100.00")
        assert result.missed is True
        assert result.reason == "malformed_geometry"

    def test_no_target_key_fails_closed(self):
        geom = json.dumps({"entry": "100.00", "stop": "95.00"})
        watch = _make_watch(side="BUY", draft_geometry_json=geom)
        result = check_target_crossed_for_pending_order(watch, "200.00")
        assert result.missed is True
        assert result.reason == "malformed_geometry"

    def test_unparseable_target_value_fails_closed(self):
        geom = json.dumps({"entry": "100.00", "stop": "95.00", "target": "not_a_number"})
        watch = _make_watch(side="BUY", draft_geometry_json=geom)
        result = check_target_crossed_for_pending_order(watch, "200.00")
        assert result.missed is True
        assert result.reason == "malformed_geometry"


# ---------------------------------------------------------------------------
# 14.13 check_target_crossed_for_pending_order: valid, target crossed
# ---------------------------------------------------------------------------

class TestPendingOrderTargetCrossed:
    """Req 7.2: Valid geometry + target crossed → block order."""

    def test_buy_target_crossed(self):
        watch = _make_watch(side="BUY", draft_geometry_json=_geometry_json("150.00"))
        result = check_target_crossed_for_pending_order(watch, "150.00")
        assert result.missed is True
        assert result.reason == "target_already_crossed"

    def test_short_target_crossed(self):
        watch = _make_watch(side="SHORT", draft_geometry_json=_geometry_json("80.00"))
        result = check_target_crossed_for_pending_order(watch, "80.00")
        assert result.missed is True
        assert result.reason == "target_already_crossed"


# ---------------------------------------------------------------------------
# 14.14 check_target_crossed_for_pending_order: valid, target NOT crossed
# ---------------------------------------------------------------------------

class TestPendingOrderTargetNotCrossed:
    """Req 7.2: Valid geometry + target NOT crossed → allow order."""

    def test_buy_target_not_crossed(self):
        watch = _make_watch(side="BUY", draft_geometry_json=_geometry_json("150.00"))
        result = check_target_crossed_for_pending_order(watch, "148.50")
        assert result.missed is False
        assert result.reason is None

    def test_short_target_not_crossed(self):
        watch = _make_watch(side="SHORT", draft_geometry_json=_geometry_json("80.00"))
        result = check_target_crossed_for_pending_order(watch, "85.50")
        assert result.missed is False
        assert result.reason is None


# ---------------------------------------------------------------------------
# 14.15 Decimal arithmetic precision: prices at exact boundary
# ---------------------------------------------------------------------------

class TestDecimalBoundaryPrecision:
    """Req 4.2, 7.5: Decimal arithmetic ensures exact boundary correctness."""

    def test_buy_at_exact_boundary_is_missed(self):
        """price == target exactly → missed for BUY."""
        target = "123.456789012345678901234567"
        watch = _make_watch(side="BUY", draft_geometry_json=_geometry_json(target))
        result = check_missed_move(watch, target)
        assert result.missed is True

    def test_short_at_exact_boundary_is_missed(self):
        """price == target exactly → missed for SHORT."""
        target = "87.654321098765432109876543"
        watch = _make_watch(side="SHORT", draft_geometry_json=_geometry_json(target))
        result = check_missed_move(watch, target)
        assert result.missed is True

    def test_buy_one_tick_below_not_missed(self):
        """BUY price one tiny increment below target → NOT missed."""
        target = "150.000000000000000000000000"
        price = "149.999999999999999999999999"
        watch = _make_watch(side="BUY", draft_geometry_json=_geometry_json(target))
        result = check_missed_move(watch, price)
        assert result.missed is False

    def test_short_one_tick_above_not_missed(self):
        """SHORT price one tiny increment above target → NOT missed."""
        target = "80.000000000000000000000000"
        price = "80.000000000000000000000001"
        watch = _make_watch(side="SHORT", draft_geometry_json=_geometry_json(target))
        result = check_missed_move(watch, price)
        assert result.missed is False

    def test_float_input_does_not_cause_drift(self):
        """Float inputs are handled via _safe_decimal without drift."""
        watch = _make_watch(side="BUY", draft_geometry_json=_geometry_json("100.00"))
        # 100.0 as float should be exactly equal to target
        result = check_missed_move(watch, 100.0)
        assert result.missed is True

    def test_pending_order_boundary_precision(self):
        """Pending-order guard uses same Decimal precision."""
        target = "200.123456789012345678901234"
        watch = _make_watch(side="BUY", draft_geometry_json=_geometry_json(target))
        result = check_target_crossed_for_pending_order(watch, target)
        assert result.missed is True
        assert result.target_price == Decimal(target)
