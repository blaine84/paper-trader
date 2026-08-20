"""Tests for utils/setup_watch_bridge.py — Watch Maturity Bridge.

Requirements: 1.1–1.9, 9.2–9.4
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from utils.setup_watch_bridge import (
    BridgeEvaluationResult,
    _attempt_state_advance,
    _build_bridge_market_context,
    _is_side_consistent,
    _record_maturity_evidence,
    evaluate_alerts,
)
from utils.setup_watch_evaluator import ConditionResult, EvaluationResult
from utils.setup_watch_registry import (
    SetupWatch,
    SetupWatchRegistry,
    SetupWatchRegistryError,
    WatchState,
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

NOW = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)


def _make_watch(
    *,
    watch_id: str | None = None,
    symbol: str = "AMD",
    side: str = "BUY",
    setup_type: str = "pullback_continuation",
    state: WatchState = WatchState.WATCHING,
    maturity_score: float = 0.0,
    last_evaluation_json: str | None = None,
    draft_geometry_json: str | None = None,
    entry_zone_json: str | None = None,
    maturation_conditions_json: str | None = None,
    invalidation_conditions_json: str | None = None,
) -> SetupWatch:
    """Create a minimal SetupWatch for testing."""
    return SetupWatch(
        watch_id=watch_id or str(uuid.uuid4()),
        profile_id="profile_1",
        symbol=symbol,
        side=side,
        setup_type=setup_type,
        state=state,
        thesis="Test thesis",
        source_type="analyst",
        source_id=None,
        source_cycle_id="cycle_001",
        maturation_conditions_json=maturation_conditions_json or json.dumps([
            {"type": "price_zone", "weight": 1.0, "params": {"level": "150.0"}},
            {"type": "regime_aligned", "weight": 1.0, "params": {}},
        ]),
        invalidation_conditions_json=invalidation_conditions_json or json.dumps([
            {"type": "price_breach", "params": {"level": "140.0"}},
        ]),
        last_evaluation_json=last_evaluation_json,
        entry_zone_json=entry_zone_json,
        draft_geometry_json=draft_geometry_json,
        maturity_score=maturity_score,
        created_at=NOW,
        updated_at=NOW,
        expires_at=NOW + timedelta(hours=48),
        state_changed_at=NOW,
        observed_cycles=2,
        ready_at=None,
        ready_reference_price=None,
        terminal_reason=None,
        promoted_cycle_id=None,
        execution_ref_type=None,
        execution_ref_id=None,
        integrity_hash="test_hash",
    )


# ────────────────────────────────────────────────────────────────────────────
# 15.2: _is_side_consistent — default BUY matches support from above
# ────────────────────────────────────────────────────────────────────────────


class TestIsSideConsistentDefault:
    """Default matching rules for _is_side_consistent."""

    def test_buy_matches_support_from_above(self):
        """15.2: BUY matches when price > level (approaching support from above)."""
        assert _is_side_consistent(
            "BUY", "pullback_continuation", "support", Decimal("151.0"), Decimal("150.0")
        ) is True

    def test_buy_rejects_support_from_below(self):
        """BUY does NOT match when price < level."""
        assert _is_side_consistent(
            "BUY", "pullback_continuation", "support", Decimal("149.0"), Decimal("150.0")
        ) is False

    def test_short_matches_resistance_from_below(self):
        """15.3: SHORT matches when price < level (approaching resistance from below)."""
        assert _is_side_consistent(
            "SHORT", "momentum_fade", "resistance", Decimal("149.0"), Decimal("150.0")
        ) is True

    def test_short_rejects_resistance_from_above(self):
        """SHORT does NOT match when price > level."""
        assert _is_side_consistent(
            "SHORT", "momentum_fade", "resistance", Decimal("151.0"), Decimal("150.0")
        ) is False


# ────────────────────────────────────────────────────────────────────────────
# 15.4–15.5: _is_side_consistent — breakout_retest special cases
# ────────────────────────────────────────────────────────────────────────────


class TestIsSideConsistentBreakoutRetest:
    """breakout_retest has inverted matching."""

    def test_breakout_retest_buy_matches_resistance_from_below(self):
        """15.4: breakout_retest BUY matches resistance from below (price < level)."""
        assert _is_side_consistent(
            "BUY", "breakout_retest", "resistance", Decimal("149.0"), Decimal("150.0")
        ) is True

    def test_breakout_retest_buy_rejects_from_above(self):
        """breakout_retest BUY does NOT match when price > level."""
        assert _is_side_consistent(
            "BUY", "breakout_retest", "resistance", Decimal("151.0"), Decimal("150.0")
        ) is False

    def test_breakout_retest_short_matches_support_from_above(self):
        """15.5: breakout_retest SHORT matches support from above (price > level)."""
        assert _is_side_consistent(
            "SHORT", "breakout_retest", "support", Decimal("151.0"), Decimal("150.0")
        ) is True

    def test_breakout_retest_short_rejects_from_below(self):
        """breakout_retest SHORT does NOT match when price < level."""
        assert _is_side_consistent(
            "SHORT", "breakout_retest", "support", Decimal("149.0"), Decimal("150.0")
        ) is False


# ────────────────────────────────────────────────────────────────────────────
# 15.6: _is_side_consistent — failed_breakdown_reclaim BUY
# ────────────────────────────────────────────────────────────────────────────


class TestIsSideConsistentFailedBreakdown:
    """failed_breakdown_reclaim special cases."""

    def test_failed_breakdown_reclaim_buy_matches_from_above(self):
        """15.6: failed_breakdown_reclaim BUY matches support from above."""
        assert _is_side_consistent(
            "BUY", "failed_breakdown_reclaim", "support", Decimal("151.0"), Decimal("150.0")
        ) is True

    def test_failed_breakdown_reclaim_buy_rejects_from_below(self):
        """failed_breakdown_reclaim BUY rejects when price < level."""
        assert _is_side_consistent(
            "BUY", "failed_breakdown_reclaim", "support", Decimal("149.0"), Decimal("150.0")
        ) is False

    def test_failed_breakdown_reclaim_short_uses_default(self):
        """failed_breakdown_reclaim SHORT uses default SHORT logic (price < level)."""
        assert _is_side_consistent(
            "SHORT", "failed_breakdown_reclaim", "resistance", Decimal("149.0"), Decimal("150.0")
        ) is True


# ────────────────────────────────────────────────────────────────────────────
# 15.7: _is_side_consistent — pullback_continuation uses default
# ────────────────────────────────────────────────────────────────────────────


class TestIsSideConsistentPullbackContinuation:
    """pullback_continuation uses default matching (no override)."""

    def test_pullback_continuation_buy_default(self):
        """15.7: pullback_continuation BUY uses default (price > level)."""
        assert _is_side_consistent(
            "BUY", "pullback_continuation", "support", Decimal("151.0"), Decimal("150.0")
        ) is True

    def test_pullback_continuation_short_default(self):
        """pullback_continuation SHORT uses default (price < level)."""
        assert _is_side_consistent(
            "SHORT", "pullback_continuation", "resistance", Decimal("149.0"), Decimal("150.0")
        ) is True


# ────────────────────────────────────────────────────────────────────────────
# 15.8: _build_bridge_market_context
# ────────────────────────────────────────────────────────────────────────────


class TestBuildBridgeMarketContext:
    """_build_bridge_market_context produces correct fields."""

    def test_produces_expected_fields_from_alert(self):
        """15.8: Correct fields populated from alert and watch."""
        alert = {
            "symbol": "AMD",
            "price": 151.5,
            "level_name": "support",
            "level_value": 150.0,
            "distance_pct": 1.0,
        }
        watch = _make_watch(side="BUY", setup_type="pullback_continuation")

        ctx = _build_bridge_market_context(alert, watch)

        assert ctx["symbol"] == "AMD"
        assert ctx["current_price"] == 151.5
        assert ctx["level_name"] == "support"
        assert ctx["level_value"] == 150.0
        assert ctx["distance_pct"] == 1.0
        assert ctx["side"] == "BUY"
        assert ctx["setup_type"] == "pullback_continuation"
        assert ctx["source"] == "price_monitor"

    def test_merges_stored_price_history(self):
        """Carries forward price_history from last_evaluation_json."""
        last_eval = json.dumps({
            "price_history": [149.0, 150.0, 151.0],
            "key_levels": {"support": 148.0},
        })
        watch = _make_watch(last_evaluation_json=last_eval)
        alert = {"symbol": "AMD", "price": 151.5, "level_name": "support",
                 "level_value": 150.0, "distance_pct": 1.0}

        ctx = _build_bridge_market_context(alert, watch)

        assert ctx["price_history"] == [149.0, 150.0, 151.0]
        assert ctx["key_levels"] == {"support": 148.0}

    def test_handles_no_last_evaluation(self):
        """Handles watch with no prior evaluation gracefully."""
        watch = _make_watch(last_evaluation_json=None)
        alert = {"symbol": "AMD", "price": 151.5, "level_name": "support",
                 "level_value": 150.0, "distance_pct": 1.0}

        ctx = _build_bridge_market_context(alert, watch)

        assert ctx["current_price"] == 151.5
        assert "price_history" not in ctx


# ────────────────────────────────────────────────────────────────────────────
# 15.9–15.10: _record_maturity_evidence
# ────────────────────────────────────────────────────────────────────────────


class TestRecordMaturityEvidence:
    """Evidence annotation for condition flips."""

    def test_counts_flipped_conditions(self):
        """15.9: Counts conditions that flipped unmet → met."""
        # Previous: price_zone was unmet
        previous_eval = {
            "condition_results": [
                {"condition_type": "price_zone", "met": False},
                {"condition_type": "regime_aligned", "met": False},
            ]
        }
        # New: price_zone is now met, regime_aligned still unmet
        new_eval = EvaluationResult(
            invalidated=False,
            invalidation_reason=None,
            maturity_score=0.5,
            condition_results=[
                ConditionResult(condition_type="price_zone", met=True, detail="in zone"),
                ConditionResult(condition_type="regime_aligned", met=False, detail=None),
            ],
            evaluation_timestamp="2026-08-14T14:30:00Z",
        )
        alert = {"price": 151.0, "level_value": 150.0, "distance_pct": 0.66}

        flipped = _record_maturity_evidence(previous_eval, new_eval, alert)

        assert flipped == 1

    def test_already_met_conditions_not_counted(self):
        """15.10: Already-met conditions don't generate new evidence."""
        previous_eval = {
            "condition_results": [
                {"condition_type": "price_zone", "met": True},
                {"condition_type": "regime_aligned", "met": True},
            ]
        }
        new_eval = EvaluationResult(
            invalidated=False,
            invalidation_reason=None,
            maturity_score=1.0,
            condition_results=[
                ConditionResult(condition_type="price_zone", met=True, detail="in zone"),
                ConditionResult(condition_type="regime_aligned", met=True, detail="aligned"),
            ],
            evaluation_timestamp="2026-08-14T14:30:00Z",
        )
        alert = {"price": 151.0, "level_value": 150.0, "distance_pct": 0.66}

        flipped = _record_maturity_evidence(previous_eval, new_eval, alert)

        assert flipped == 0

    def test_no_previous_eval_all_met_are_flips(self):
        """When no previous eval, all met conditions count as flips."""
        new_eval = EvaluationResult(
            invalidated=False,
            invalidation_reason=None,
            maturity_score=0.5,
            condition_results=[
                ConditionResult(condition_type="price_zone", met=True, detail="in zone"),
                ConditionResult(condition_type="regime_aligned", met=False, detail=None),
            ],
            evaluation_timestamp="2026-08-14T14:30:00Z",
        )
        alert = {"price": 151.0, "level_value": 150.0, "distance_pct": 0.66}

        flipped = _record_maturity_evidence(None, new_eval, alert)

        assert flipped == 1


# ────────────────────────────────────────────────────────────────────────────
# 15.11–15.13: _attempt_state_advance
# ────────────────────────────────────────────────────────────────────────────


class TestAttemptStateAdvance:
    """CAS state advance based on score thresholds."""

    def test_watching_to_maturing_on_score_gt_zero(self):
        """15.11: Transitions watching → maturing when score > 0."""
        registry = MagicMock(spec=SetupWatchRegistry)
        watch = _make_watch(state=WatchState.WATCHING, maturity_score=0.0)

        result = _attempt_state_advance(registry, watch, 0.3, Decimal("151.0"))

        assert result == WatchState.MATURING
        registry.transition_state.assert_called_once_with(
            watch.watch_id,
            WatchState.WATCHING,
            WatchState.MATURING,
        )

    def test_maturing_to_ready_on_score_gte_threshold(self):
        """15.12: Transitions maturing → ready when score >= threshold."""
        registry = MagicMock(spec=SetupWatchRegistry)
        watch = _make_watch(state=WatchState.MATURING, maturity_score=0.3)

        # Default threshold is 0.7
        with patch("utils.setup_watch_bridge.SETUP_WATCH_MATURITY_THRESHOLD", 0.7):
            result = _attempt_state_advance(registry, watch, 0.7, Decimal("151.0"))

        assert result == WatchState.READY
        registry.transition_state.assert_called_once_with(
            watch.watch_id,
            WatchState.MATURING,
            WatchState.READY,
            ready_reference_price=151.0,
        )

    def test_no_retry_on_cas_failure(self):
        """15.13: Does NOT retry on CAS failure — returns None."""
        registry = MagicMock(spec=SetupWatchRegistry)
        registry.transition_state.side_effect = SetupWatchRegistryError("CAS failed")
        watch = _make_watch(state=WatchState.WATCHING, maturity_score=0.0)

        result = _attempt_state_advance(registry, watch, 0.5, Decimal("151.0"))

        assert result is None
        # Called exactly once, no retry
        assert registry.transition_state.call_count == 1

    def test_no_transition_when_score_zero_and_watching(self):
        """No transition when watching and score == 0."""
        registry = MagicMock(spec=SetupWatchRegistry)
        watch = _make_watch(state=WatchState.WATCHING, maturity_score=0.0)

        result = _attempt_state_advance(registry, watch, 0.0, Decimal("151.0"))

        assert result is None
        registry.transition_state.assert_not_called()

    def test_no_transition_maturing_below_threshold(self):
        """No transition when maturing but score < threshold."""
        registry = MagicMock(spec=SetupWatchRegistry)
        watch = _make_watch(state=WatchState.MATURING, maturity_score=0.3)

        with patch("utils.setup_watch_bridge.SETUP_WATCH_MATURITY_THRESHOLD", 0.7):
            result = _attempt_state_advance(registry, watch, 0.5, Decimal("151.0"))

        assert result is None
        registry.transition_state.assert_not_called()


# ────────────────────────────────────────────────────────────────────────────
# 15.14: evaluate_alerts returns zero-result when disabled
# ────────────────────────────────────────────────────────────────────────────


class TestEvaluateAlertsDisabled:
    """evaluate_alerts with SETUP_WATCH_REALTIME_MODE=disabled."""

    def test_returns_zero_result_when_disabled(self, monkeypatch):
        """15.14: Zero-result when mode is disabled."""
        monkeypatch.setattr(
            "utils.setup_watch_bridge.SETUP_WATCH_REALTIME_MODE", "disabled"
        )
        engine = MagicMock()
        alerts = [{"symbol": "AMD", "price": 151.0, "level_name": "support",
                   "level_value": 150.0, "distance_pct": 0.66}]

        result = evaluate_alerts(engine, alerts)

        assert result == BridgeEvaluationResult()
        assert result.alerts_processed == 0
        assert result.watches_evaluated == 0
        assert result.state_transitions == 0
        # Engine should never be touched
        engine.connect.assert_not_called()


# ────────────────────────────────────────────────────────────────────────────
# 15.15: evaluate_alerts in "observe" mode
# ────────────────────────────────────────────────────────────────────────────


class TestEvaluateAlertsObserve:
    """evaluate_alerts in observe mode — events but no transitions."""

    @patch("utils.setup_watch_bridge._get_active_watches_for_symbols")
    @patch("utils.setup_watch_bridge.evaluate_watch")
    def test_observe_mode_no_state_transitions(
        self, mock_evaluate_watch, mock_get_watches, monkeypatch
    ):
        """15.15: Observe mode emits events but zero state transitions."""
        monkeypatch.setattr(
            "utils.setup_watch_bridge.SETUP_WATCH_REALTIME_MODE", "observe"
        )
        watch = _make_watch(
            state=WatchState.WATCHING,
            maturity_score=0.0,
        )
        mock_get_watches.return_value = [watch]
        mock_evaluate_watch.return_value = EvaluationResult(
            invalidated=False,
            invalidation_reason=None,
            maturity_score=0.5,
            condition_results=[
                ConditionResult(condition_type="price_zone", met=True, detail="in zone"),
            ],
            evaluation_timestamp="2026-08-14T14:30:00Z",
        )

        engine = MagicMock()
        alerts = [{"symbol": "AMD", "price": 151.0, "level_name": "support",
                   "level_value": 150.0, "distance_pct": 0.66}]

        with patch.object(SetupWatchRegistry, "update_evaluation"):
            with patch.object(SetupWatchRegistry, "transition_state") as mock_transition:
                with patch.object(SetupWatchRegistry, "_emit_event"):
                    result = evaluate_alerts(engine, alerts)

        assert result.state_transitions == 0
        mock_transition.assert_not_called()


# ────────────────────────────────────────────────────────────────────────────
# 15.16: evaluate_alerts in "enabled" mode — transitions and promotion
# ────────────────────────────────────────────────────────────────────────────


class TestEvaluateAlertsEnabled:
    """evaluate_alerts in enabled mode — transitions and promotion."""

    @patch("utils.setup_watch_bridge._try_promote_ready_watch")
    @patch("utils.setup_watch_bridge._get_active_watches_for_symbols")
    @patch("utils.setup_watch_bridge.evaluate_watch")
    def test_enabled_mode_transitions_and_promotes(
        self, mock_evaluate_watch, mock_get_watches, mock_promote, monkeypatch
    ):
        """15.16: Enabled mode performs transitions and invokes promotion."""
        monkeypatch.setattr(
            "utils.setup_watch_bridge.SETUP_WATCH_REALTIME_MODE", "enabled"
        )
        monkeypatch.setattr(
            "utils.setup_watch_bridge.SETUP_WATCH_MATURITY_THRESHOLD", 0.7
        )
        watch = _make_watch(
            state=WatchState.MATURING,
            maturity_score=0.5,
        )
        mock_get_watches.return_value = [watch]
        mock_evaluate_watch.return_value = EvaluationResult(
            invalidated=False,
            invalidation_reason=None,
            maturity_score=0.8,
            condition_results=[
                ConditionResult(condition_type="price_zone", met=True, detail="in zone"),
                ConditionResult(condition_type="regime_aligned", met=True, detail="aligned"),
            ],
            evaluation_timestamp="2026-08-14T14:30:00Z",
        )

        engine = MagicMock()
        alerts = [{"symbol": "AMD", "price": 151.0, "level_name": "support",
                   "level_value": 150.0, "distance_pct": 0.66}]

        with patch.object(SetupWatchRegistry, "update_evaluation"):
            with patch.object(SetupWatchRegistry, "transition_state") as mock_transition:
                with patch.object(SetupWatchRegistry, "_emit_event"):
                    result = evaluate_alerts(engine, alerts)

        assert result.state_transitions == 1
        mock_transition.assert_called_once()
        # Should have called promotion since watch reached READY
        mock_promote.assert_called_once()


# ────────────────────────────────────────────────────────────────────────────
# 15.17: Per-watch error does not abort remaining watches
# ────────────────────────────────────────────────────────────────────────────


class TestEvaluateAlertsPerWatchError:
    """Per-watch error handling."""

    @patch("utils.setup_watch_bridge._get_active_watches_for_symbols")
    @patch("utils.setup_watch_bridge.evaluate_watch")
    def test_per_watch_error_does_not_abort_remaining(
        self, mock_evaluate_watch, mock_get_watches, monkeypatch
    ):
        """15.17: One watch error doesn't stop other watches from being evaluated."""
        monkeypatch.setattr(
            "utils.setup_watch_bridge.SETUP_WATCH_REALTIME_MODE", "enabled"
        )
        watch1 = _make_watch(watch_id="watch_1", state=WatchState.WATCHING)
        watch2 = _make_watch(watch_id="watch_2", state=WatchState.WATCHING)
        mock_get_watches.return_value = [watch1, watch2]

        # First call raises, second succeeds
        mock_evaluate_watch.side_effect = [
            RuntimeError("Unexpected error in watch1"),
            EvaluationResult(
                invalidated=False,
                invalidation_reason=None,
                maturity_score=0.3,
                condition_results=[
                    ConditionResult(condition_type="price_zone", met=True, detail="ok"),
                ],
                evaluation_timestamp="2026-08-14T14:30:00Z",
            ),
        ]

        engine = MagicMock()
        alerts = [{"symbol": "AMD", "price": 151.0, "level_name": "support",
                   "level_value": 150.0, "distance_pct": 0.66}]

        with patch.object(SetupWatchRegistry, "update_evaluation"):
            with patch.object(SetupWatchRegistry, "_emit_event"):
                result = evaluate_alerts(engine, alerts)

        # One error, one successful evaluation
        assert result.errors == 1
        assert result.watches_evaluated == 1


# ────────────────────────────────────────────────────────────────────────────
# 15.18: Zero matching alerts → BridgeEvaluationResult with all zeros
# ────────────────────────────────────────────────────────────────────────────


class TestEvaluateAlertsZeroMatches:
    """Bridge with zero matching alerts."""

    @patch("utils.setup_watch_bridge._get_active_watches_for_symbols")
    def test_zero_matches_all_zeros(self, mock_get_watches, monkeypatch):
        """15.18: No matching watches → all zeros in result."""
        monkeypatch.setattr(
            "utils.setup_watch_bridge.SETUP_WATCH_REALTIME_MODE", "enabled"
        )
        # No watches for the alerted symbol
        mock_get_watches.return_value = []

        engine = MagicMock()
        alerts = [{"symbol": "TSLA", "price": 250.0, "level_name": "resistance",
                   "level_value": 255.0, "distance_pct": 2.0}]

        result = evaluate_alerts(engine, alerts)

        assert result.alerts_processed == 1
        assert result.watches_evaluated == 0
        assert result.conditions_flipped == 0
        assert result.state_transitions == 0
        assert result.missed_moves_detected == 0
        assert result.errors == 0
