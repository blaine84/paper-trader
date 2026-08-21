"""Tests for the fast-path outcome evaluator.

Validates:
- Target crossed (BUY above target) -> missed_move
- Target crossed (SHORT below target) -> missed_move
- Stale market data -> stand_down
- Invalid geometry -> stand_down
- Entry too far -> stand_down
- Trigger not met -> returns None
- Trigger met + all gates pass -> trade_executed
- Trigger met + gate rejects -> stand_down
- Price away + valid limit -> pending_order_created
- Needs confirmation -> watch_created

Requirements: 3.1-3.10, cross-cutting acceptance tests 1-3
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from utils.fast_path_evaluator import (
    FastPathOutcome,
    evaluate_trigger,
    price_away_but_limit_valid,
    requires_confirmation,
    target_crossed,
    trigger_condition_met,
)
from utils.fast_path_registry import TriggerRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trigger(**overrides) -> TriggerRecord:
    """Build a minimally valid SHORT momentum_fade TriggerRecord."""
    defaults = {
        "trigger_id": str(uuid.uuid4()),
        "symbol": "TSLA",
        "profile_id": "moderate",
        "direction": "SHORT",
        "setup_type": "momentum_fade",
        "trigger_type": "entry_zone",
        "trigger_level": 351.61,
        "trigger_zone_upper": 352.00,
        "trigger_zone_lower": 350.50,
        "entry_price": 351.61,
        "stop_price": 355.00,
        "target_price": 348.00,
        "geometry_name": "momentum_fade_short",
        "source_signal_id": "sig-001",
        "source_watch_id": None,
        "invalidation_basis": "close above 355",
        "target_basis": "prior support",
        "state": "active",
        "registered_at": "2026-08-20T14:30:00+00:00",
        "expires_at": "2026-08-20T14:35:00+00:00",
        "signal_snapshot_json": '{"setup_type":"momentum_fade"}',
        "context_json": None,
    }
    defaults.update(overrides)
    return TriggerRecord(**defaults)


def _make_buy_trigger(**overrides) -> TriggerRecord:
    """Build a minimally valid BUY technical_breakout TriggerRecord."""
    defaults = {
        "trigger_id": str(uuid.uuid4()),
        "symbol": "AAPL",
        "profile_id": "moderate",
        "direction": "BUY",
        "setup_type": "technical_breakout",
        "trigger_type": "level_break",
        "trigger_level": 180.00,
        "trigger_zone_upper": None,
        "trigger_zone_lower": None,
        "entry_price": 180.00,
        "stop_price": 175.00,
        "target_price": 195.00,
        "geometry_name": "breakout_long",
        "source_signal_id": "sig-002",
        "source_watch_id": None,
        "invalidation_basis": "close below 175",
        "target_basis": "measured move",
        "state": "active",
        "registered_at": "2026-08-20T14:30:00+00:00",
        "expires_at": "2026-08-20T14:35:00+00:00",
        "signal_snapshot_json": '{"setup_type":"technical_breakout"}',
        "context_json": None,
    }
    defaults.update(overrides)
    return TriggerRecord(**defaults)


def _quote(price: float, age_ms: int = 500, reliable: bool = True) -> dict:
    """Build a simple quote dict."""
    return {"price": price, "age_ms": age_ms, "reliable": reliable}


# ---------------------------------------------------------------------------
# Standalone function tests: target_crossed
# ---------------------------------------------------------------------------


class TestTargetCrossed:
    """Tests for the target_crossed standalone function."""

    def test_buy_above_target_is_crossed(self):
        assert target_crossed("BUY", 195.0, 190.0) is True

    def test_short_below_target_is_crossed(self):
        assert target_crossed("SHORT", 345.0, 348.0) is True

    def test_buy_below_target_not_crossed(self):
        assert target_crossed("BUY", 185.0, 190.0) is False

    def test_short_above_target_not_crossed(self):
        assert target_crossed("SHORT", 350.0, 348.0) is False

    def test_buy_at_target_is_crossed(self):
        assert target_crossed("BUY", 190.0, 190.0) is True

    def test_short_at_target_is_crossed(self):
        assert target_crossed("SHORT", 348.0, 348.0) is True


# ---------------------------------------------------------------------------
# Standalone function tests: trigger_condition_met
# ---------------------------------------------------------------------------


class TestTriggerConditionMet:
    """Tests for the trigger_condition_met standalone function."""

    def test_entry_zone_price_inside_zone_met(self):
        trigger = _make_trigger(
            trigger_type="entry_zone",
            trigger_zone_lower=350.50,
            trigger_zone_upper=352.00,
        )
        assert trigger_condition_met(trigger, _quote(351.00)) is True

    def test_entry_zone_price_outside_zone_not_met(self):
        trigger = _make_trigger(
            trigger_type="entry_zone",
            trigger_zone_lower=350.50,
            trigger_zone_upper=352.00,
        )
        assert trigger_condition_met(trigger, _quote(349.00)) is False

    def test_entry_zone_price_at_lower_bound_met(self):
        trigger = _make_trigger(
            trigger_type="entry_zone",
            trigger_zone_lower=350.50,
            trigger_zone_upper=352.00,
        )
        assert trigger_condition_met(trigger, _quote(350.50)) is True

    def test_entry_zone_price_at_upper_bound_met(self):
        trigger = _make_trigger(
            trigger_type="entry_zone",
            trigger_zone_lower=350.50,
            trigger_zone_upper=352.00,
        )
        assert trigger_condition_met(trigger, _quote(352.00)) is True

    def test_level_break_buy_price_above_level_met(self):
        trigger = _make_buy_trigger(
            trigger_type="level_break",
            trigger_level=180.00,
        )
        assert trigger_condition_met(trigger, _quote(180.50)) is True

    def test_level_break_buy_price_below_level_not_met(self):
        trigger = _make_buy_trigger(
            trigger_type="level_break",
            trigger_level=180.00,
        )
        assert trigger_condition_met(trigger, _quote(179.50)) is False

    def test_level_break_short_price_below_level_met(self):
        trigger = _make_trigger(
            trigger_type="level_break",
            trigger_level=351.00,
            direction="SHORT",
        )
        assert trigger_condition_met(trigger, _quote(350.50)) is True

    def test_level_break_short_price_above_level_not_met(self):
        trigger = _make_trigger(
            trigger_type="level_break",
            trigger_level=351.00,
            direction="SHORT",
        )
        assert trigger_condition_met(trigger, _quote(352.00)) is False


# ---------------------------------------------------------------------------
# Standalone function tests: requires_confirmation
# ---------------------------------------------------------------------------


class TestRequiresConfirmation:
    """Tests for the requires_confirmation standalone function."""

    def test_level_reject_requires_confirmation(self):
        trigger = _make_trigger(trigger_type="level_reject")
        assert requires_confirmation(trigger) is True

    def test_level_break_does_not_require_confirmation(self):
        trigger = _make_buy_trigger(trigger_type="level_break")
        assert requires_confirmation(trigger) is False

    def test_entry_zone_does_not_require_confirmation(self):
        trigger = _make_trigger(trigger_type="entry_zone")
        assert requires_confirmation(trigger) is False

    def test_invalidation_basis_with_retest_requires_confirmation(self):
        trigger = _make_buy_trigger(
            trigger_type="level_break",
            invalidation_basis="needs retest of 180 before entry",
        )
        assert requires_confirmation(trigger) is True

    def test_invalidation_basis_with_confirmation_requires_confirmation(self):
        trigger = _make_trigger(
            trigger_type="entry_zone",
            invalidation_basis="await confirmation at support",
        )
        assert requires_confirmation(trigger) is True

    def test_invalidation_basis_without_keywords_does_not_require(self):
        trigger = _make_trigger(
            trigger_type="entry_zone",
            invalidation_basis="close above 355",
        )
        assert requires_confirmation(trigger) is False


# ---------------------------------------------------------------------------
# Standalone function tests: price_away_but_limit_valid
# ---------------------------------------------------------------------------


class TestPriceAwayButLimitValid:
    """Tests for the price_away_but_limit_valid standalone function."""

    def test_buy_price_above_entry_target_ahead(self):
        # BUY: price ran past entry (180) to 182 but target (195) still ahead
        trigger = _make_buy_trigger(
            entry_price=180.00,
            stop_price=175.00,
            target_price=195.00,
        )
        assert price_away_but_limit_valid(trigger, _quote(182.00)) is True

    def test_short_price_below_entry_target_ahead(self):
        # SHORT: price ran past entry (351.61) to 350 but target (348) still ahead
        trigger = _make_trigger(
            entry_price=351.61,
            stop_price=355.00,
            target_price=348.00,
        )
        assert price_away_but_limit_valid(trigger, _quote(350.00)) is True

    def test_buy_price_below_entry_not_away(self):
        # BUY: price (179) hasn't run past entry (180)
        trigger = _make_buy_trigger(
            entry_price=180.00,
            stop_price=175.00,
            target_price=195.00,
        )
        assert price_away_but_limit_valid(trigger, _quote(179.00)) is False

    def test_buy_price_past_target_not_valid(self):
        # BUY: price (196) past target (195) — target crossed
        trigger = _make_buy_trigger(
            entry_price=180.00,
            stop_price=175.00,
            target_price=195.00,
        )
        assert price_away_but_limit_valid(trigger, _quote(196.00)) is False


# ---------------------------------------------------------------------------
# evaluate_trigger integration tests
# ---------------------------------------------------------------------------


class TestEvaluateTriggerMissedMove:
    """Tests for missed_move outcomes from evaluate_trigger."""

    def test_buy_above_target_produces_missed_move(self):
        """BUY trigger where current price already above target -> missed_move.
        Cross-cutting acceptance test 1.
        """
        trigger = _make_buy_trigger(
            entry_price=180.00,
            stop_price=175.00,
            target_price=195.00,
        )
        quote = _quote(196.00)  # Price above target
        result = evaluate_trigger(trigger, quote, {})

        assert result is not None
        assert result.outcome_type == "missed_move"
        assert result.outcome_reason_code == "target_already_crossed"
        assert result.symbol == "AAPL"
        assert result.current_price == 196.00

    def test_short_below_target_produces_missed_move(self):
        """SHORT trigger where current price already below target -> missed_move.
        Cross-cutting acceptance test 1.
        """
        trigger = _make_trigger(
            entry_price=351.61,
            stop_price=355.00,
            target_price=348.00,
        )
        quote = _quote(347.00)  # Price below target
        result = evaluate_trigger(trigger, quote, {})

        assert result is not None
        assert result.outcome_type == "missed_move"
        assert result.outcome_reason_code == "target_already_crossed"
        assert result.symbol == "TSLA"
        assert result.current_price == 347.00


class TestEvaluateTriggerStandDown:
    """Tests for stand_down outcomes from evaluate_trigger."""

    def test_stale_market_data_produces_stand_down(self):
        """Quote older than FAST_PATH_MAX_TRIGGER_AGE_SECONDS -> stand_down."""
        trigger = _make_trigger()
        # age_ms > 300 * 1000 = 300000
        quote = _quote(351.00, age_ms=400000)
        result = evaluate_trigger(trigger, quote, {})

        assert result is not None
        assert result.outcome_type == "stand_down"
        assert result.outcome_reason_code == "stale_market_data"

    def test_unreliable_market_data_produces_stand_down(self):
        """Unreliable market data -> stand_down."""
        trigger = _make_trigger()
        quote = _quote(351.00, reliable=False)
        result = evaluate_trigger(trigger, quote, {})

        assert result is not None
        assert result.outcome_type == "stand_down"
        assert result.outcome_reason_code == "market_data_unreliable"

    def test_invalid_geometry_produces_stand_down(self):
        """Invalid geometry (stop on wrong side) -> stand_down."""
        # BUY with stop ABOVE entry — invalid geometry
        trigger = _make_buy_trigger(
            entry_price=180.00,
            stop_price=185.00,  # Stop above entry for BUY is invalid
            target_price=195.00,
        )
        quote = _quote(180.00)
        result = evaluate_trigger(trigger, quote, {})

        assert result is not None
        assert result.outcome_type == "stand_down"
        assert result.outcome_reason_code == "invalid_geometry"

    def test_entry_too_far_produces_stand_down(self):
        """Price far from entry (>5%) and not limit-eligible -> stand_down."""
        # BUY with entry at 180, price at 170 (deviation ~5.5%)
        # Price below entry for BUY, so not limit-eligible (price hasn't run past)
        # And trigger condition: level_break at 180, price 170 doesn't meet it either
        # But we need trigger to fire first, then check deviation.
        # Use entry_zone trigger where price is in zone but far from entry.
        trigger = _make_buy_trigger(
            trigger_type="level_break",
            trigger_level=170.00,  # Set level_break lower so trigger fires at 170
            entry_price=180.00,
            stop_price=175.00,
            target_price=195.00,
        )
        # Price is 170. Trigger (level_break BUY >= 170) met.
        # Price below entry for BUY → not limit-eligible.
        # Deviation: |170 - 180| / 180 = 5.5% > 5% → stand_down
        quote = _quote(170.00)
        result = evaluate_trigger(trigger, quote, {})

        assert result is not None
        assert result.outcome_type == "stand_down"
        assert result.outcome_reason_code == "entry_too_far_from_price"

    @patch("utils.fast_path_evaluator._run_gates")
    def test_gate_rejects_produces_stand_down(self, mock_run_gates):
        """Trigger met but gate pipeline rejects -> stand_down."""
        mock_run_gates.return_value = (False, "risk_geometry")

        # SHORT entry_zone trigger, price in zone and AT/above entry
        # so price_away_but_limit_valid returns False (SHORT: price >= entry is NOT "past")
        trigger = _make_trigger(
            entry_price=351.00,
            stop_price=355.00,
            target_price=348.00,
            trigger_type="entry_zone",
            trigger_zone_lower=350.50,
            trigger_zone_upper=352.00,
        )
        # Price 351.50 is in zone [350.50, 352.00] and price >= entry (351.50 >= 351.00)
        # So SHORT hasn't run past entry → not limit-eligible
        # Deviation: |351.50 - 351.00|/351.00 = 0.14% < 5%
        quote = _quote(351.50)
        result = evaluate_trigger(trigger, quote, {})

        assert result is not None
        assert result.outcome_type == "stand_down"
        assert result.outcome_reason_code == "gate_rejected:risk_geometry"
        assert result.blocking_rule_name == "risk_geometry"


class TestEvaluateTriggerNone:
    """Tests for None return (trigger not fired, stays active)."""

    def test_trigger_not_met_returns_none(self):
        """Entry zone trigger with price outside zone -> None (stays active)."""
        trigger = _make_trigger(
            trigger_type="entry_zone",
            trigger_zone_lower=350.50,
            trigger_zone_upper=352.00,
            entry_price=351.61,
            stop_price=355.00,
            target_price=348.00,
        )
        # Price 353 is outside the entry zone
        quote = _quote(353.00)
        result = evaluate_trigger(trigger, quote, {})

        assert result is None


class TestEvaluateTriggerTradeExecuted:
    """Tests for trade_executed outcomes from evaluate_trigger."""

    def test_trigger_met_all_gates_pass_produces_trade_executed(self):
        """Trigger condition met + all gates pass -> trade_executed.
        Uses stub _run_gates which returns (True, None) by default.
        """
        # SHORT entry_zone trigger, price in zone, at/above entry
        # so price_away_but_limit_valid = False (SHORT: price >= entry is not "past")
        trigger = _make_trigger(
            entry_price=351.00,
            stop_price=355.00,
            target_price=348.00,
            trigger_type="entry_zone",
            trigger_zone_lower=350.50,
            trigger_zone_upper=352.00,
        )
        # Price 351.50 is within [350.50, 352.00] zone
        # price >= entry (351.50 >= 351.00) → not price-away for SHORT
        # Deviation: |351.50 - 351.00|/351.00 = 0.14% < 5%
        quote = _quote(351.50)
        result = evaluate_trigger(trigger, quote, {})

        assert result is not None
        assert result.outcome_type == "trade_executed"
        assert result.outcome_reason_code == "all_gates_passed"
        assert result.symbol == "TSLA"
        assert result.direction == "SHORT"
        assert result.entry_price == 351.00
        assert result.stop_price == 355.00
        assert result.target_price == 348.00
        assert result.reward_to_risk is not None


class TestEvaluateTriggerPendingOrder:
    """Tests for pending_order_created outcomes from evaluate_trigger."""

    def test_price_away_valid_limit_produces_pending_order(self):
        """Price ran past entry but target still ahead -> pending_order_created.
        Cross-cutting acceptance test 2.
        """
        # SHORT: entry at 351.61, price ran past (below) to 350.00
        # Target 348 still ahead (current 350 > 348)
        # Geometry valid, R:R acceptable
        trigger = _make_trigger(
            entry_price=351.61,
            stop_price=355.00,
            target_price=348.00,
            trigger_type="entry_zone",
            trigger_zone_lower=349.00,
            trigger_zone_upper=351.00,
        )
        # Price 350.00: in zone [349, 351], SHORT price < entry (350 < 351.61) → price away
        quote = _quote(350.00)
        result = evaluate_trigger(trigger, quote, {})

        assert result is not None
        assert result.outcome_type == "pending_order_created"
        assert result.outcome_reason_code == "limit_order_valid"

    def test_buy_price_away_valid_limit_produces_pending_order(self):
        """BUY: price ran past entry but target still ahead -> pending_order_created."""
        # BUY: entry at 180, price ran past (above) to 182
        # Target 195 still ahead (current 182 < 195)
        trigger = _make_buy_trigger(
            entry_price=180.00,
            stop_price=175.00,
            target_price=195.00,
            trigger_type="level_break",
            trigger_level=180.00,
        )
        # Price 182: level_break BUY (>= 180) met, price > entry (182 > 180)
        quote = _quote(182.00)
        result = evaluate_trigger(trigger, quote, {})

        assert result is not None
        assert result.outcome_type == "pending_order_created"
        assert result.outcome_reason_code == "limit_order_valid"


class TestEvaluateTriggerWatchCreated:
    """Tests for watch_created outcomes from evaluate_trigger."""

    def test_needs_confirmation_produces_watch_created(self):
        """Level_reject trigger (requires confirmation) -> watch_created.
        Cross-cutting acceptance test 3.
        """
        # level_reject trigger: requires_confirmation returns True
        trigger = _make_trigger(
            trigger_type="level_reject",
            trigger_level=351.61,
            entry_price=351.61,
            stop_price=355.00,
            target_price=348.00,
        )
        # Price 351.50: within 0.5% of trigger_level → trigger_condition_met
        quote = _quote(351.50)
        result = evaluate_trigger(trigger, quote, {})

        assert result is not None
        assert result.outcome_type == "watch_created"
        assert result.outcome_reason_code == "awaiting_confirmation"


# ---------------------------------------------------------------------------
# FastPathOutcome dataclass tests
# ---------------------------------------------------------------------------


class TestFastPathOutcome:
    """Tests for the FastPathOutcome frozen dataclass."""

    def test_outcome_is_frozen(self):
        outcome = FastPathOutcome(
            outcome_type="stand_down",
            outcome_reason_code="stale_market_data",
            trigger_id="t-001",
            symbol="TSLA",
            profile_id="moderate",
            direction="SHORT",
            setup_type="momentum_fade",
            current_price=351.00,
        )
        with pytest.raises(Exception):
            outcome.outcome_type = "trade_executed"  # type: ignore

    def test_outcome_optional_fields_default_to_none(self):
        outcome = FastPathOutcome(
            outcome_type="missed_move",
            outcome_reason_code="target_already_crossed",
            trigger_id="t-002",
            symbol="AAPL",
            profile_id="moderate",
            direction="BUY",
            setup_type="technical_breakout",
            current_price=196.00,
        )
        assert outcome.entry_price is None
        assert outcome.stop_price is None
        assert outcome.target_price is None
        assert outcome.reward_to_risk is None
        assert outcome.blocking_rule_name is None
        assert outcome.blocking_rule_threshold is None
        assert outcome.metadata is None
