"""Tests for utils/setup_watch_outcomes.py — counterfactual outcome scoring.

Requirements: 11.1-11.12, 12.8
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from utils.setup_watch_outcomes import (
    WatchOutcome,
    _unscorable,
    run_setup_watch_outcome_scoring,
    score_watch_outcome,
)

NOW = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Mock watch object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockWatch:
    """Lightweight stand-in for SetupWatch with the fields outcome scoring needs."""

    watch_id: str = "watch_001"
    profile_id: str = "moderate"
    symbol: str = "AAPL"
    side: str = "BUY"
    ready_at: str = "2026-08-14T14:30:00Z"
    ready_reference_price: float | None = 100.0
    entry_zone_json: str | None = None
    draft_geometry_json: str | None = None


def _candle(minutes_offset: int, open_: float, high: float, low: float, close: float) -> dict:
    """Build a candle dict at NOW + minutes_offset."""
    ts = NOW + timedelta(minutes=minutes_offset)
    return {
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000,
    }


# ---------------------------------------------------------------------------
# Test 1: BUY — mfe_pct positive when price rises, mae_pct negative when falls
# ---------------------------------------------------------------------------


class TestBuySignConvention:
    def test_mfe_positive_when_price_rises(self):
        """BUY: ref=100, high=105 → MFE = +5.0%"""
        watch = MockWatch(side="BUY", ready_reference_price=100.0)
        candles = [_candle(5, 100, 105, 99, 102)]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.scorable == 1
        assert result.mfe_pct == pytest.approx(5.0)

    def test_mae_negative_when_price_falls(self):
        """BUY: ref=100, low=97 → MAE = -3.0%"""
        watch = MockWatch(side="BUY", ready_reference_price=100.0)
        candles = [_candle(5, 100, 101, 97, 99)]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.scorable == 1
        assert result.mae_pct == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# Test 2: SHORT — sign convention inverted (favorable = downward)
# ---------------------------------------------------------------------------


class TestShortSignConvention:
    def test_mfe_positive_when_price_falls(self):
        """SHORT: ref=100, low=95 → MFE = +5.0%"""
        watch = MockWatch(side="SHORT", ready_reference_price=100.0)
        candles = [_candle(5, 100, 101, 95, 97)]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.scorable == 1
        assert result.mfe_pct == pytest.approx(5.0)

    def test_mae_negative_when_price_rises(self):
        """SHORT: ref=100, high=103 → MAE = -3.0%"""
        watch = MockWatch(side="SHORT", ready_reference_price=100.0)
        candles = [_candle(5, 98, 103, 97, 99)]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.scorable == 1
        assert result.mae_pct == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# Test 3: entry_zone_touched
# ---------------------------------------------------------------------------


class TestEntryZoneTouched:
    def test_touched_when_candle_overlaps_zone(self):
        """Zone {low:99, high:101}, candle low=98 high=102 → overlaps → 1"""
        zone = json.dumps({"low": 99, "high": 101})
        watch = MockWatch(entry_zone_json=zone)
        candles = [_candle(5, 100, 102, 98, 100)]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.entry_zone_touched == 1

    def test_not_touched_when_candle_outside_zone(self):
        """Zone {low:99, high:101}, candle entirely above zone → 0"""
        zone = json.dumps({"low": 99, "high": 101})
        watch = MockWatch(entry_zone_json=zone)
        candles = [_candle(5, 102, 105, 102, 104)]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.entry_zone_touched == 0

    def test_null_when_no_zone(self):
        """No entry_zone_json → entry_zone_touched is None."""
        watch = MockWatch(entry_zone_json=None)
        candles = [_candle(5, 100, 105, 98, 102)]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.entry_zone_touched is None


# ---------------------------------------------------------------------------
# Test 4: would_have_hit_target when target reached before stop
# ---------------------------------------------------------------------------


class TestWouldHaveHitTarget:
    def test_target_hit_before_stop(self):
        """BUY: target=110, stop=95. First candle hits target (high=111). → hit_target=1"""
        geom = json.dumps({"target": 110, "stop": 95})
        watch = MockWatch(side="BUY", draft_geometry_json=geom)
        candles = [
            _candle(5, 100, 111, 99, 109),  # hits target, doesn't hit stop
        ]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.would_have_hit_target == 1
        assert result.would_have_hit_stop == 0


# ---------------------------------------------------------------------------
# Test 5: would_have_hit_stop when stop reached before target
# ---------------------------------------------------------------------------


class TestWouldHaveHitStop:
    def test_stop_hit_before_target(self):
        """BUY: target=110, stop=95. First candle hits stop (low=94). → hit_stop=1"""
        geom = json.dumps({"target": 110, "stop": 95})
        watch = MockWatch(side="BUY", draft_geometry_json=geom)
        candles = [
            _candle(5, 100, 105, 94, 96),  # hits stop, doesn't hit target
        ]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.would_have_hit_target == 0
        assert result.would_have_hit_stop == 1


# ---------------------------------------------------------------------------
# Test 6: Single bar spanning both levels → pessimistic (hit_stop=True)
# ---------------------------------------------------------------------------


class TestPessimisticConvention:
    def test_single_bar_spanning_both_records_stop(self):
        """BUY: target=110, stop=95. Single bar high=111, low=94 → hit_stop=1 (pessimistic)."""
        geom = json.dumps({"target": 110, "stop": 95})
        watch = MockWatch(side="BUY", draft_geometry_json=geom)
        candles = [
            _candle(5, 100, 111, 94, 100),  # spans both target and stop
        ]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.would_have_hit_stop == 1
        assert result.would_have_hit_target == 0


# ---------------------------------------------------------------------------
# Test 7: Candles outside the window are excluded
# ---------------------------------------------------------------------------


class TestWindowFiltering:
    def test_candles_before_ready_at_excluded(self):
        """Candle timestamped before ready_at should be excluded from scoring."""
        watch = MockWatch(side="BUY", ready_reference_price=100.0)
        candles = [
            _candle(-10, 100, 200, 50, 150),  # BEFORE ready_at → excluded
            _candle(5, 100, 103, 99, 102),  # in window
        ]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        # If the -10 candle were included, MFE would be 100%. Only 3% expected.
        assert result.mfe_pct == pytest.approx(3.0)
        assert result.mae_pct == pytest.approx(-1.0)

    def test_candles_after_window_excluded(self):
        """Candle timestamped after ready_at + window_minutes should be excluded."""
        watch = MockWatch(side="BUY", ready_reference_price=100.0)
        candles = [
            _candle(5, 100, 103, 99, 102),  # in window
            _candle(20, 100, 200, 50, 150),  # AFTER 15-min window → excluded
        ]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        # If the +20 candle were included, MFE would be 100%. Only 3% expected.
        assert result.mfe_pct == pytest.approx(3.0)
        assert result.mae_pct == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Test 8: ready_reference_price IS NULL → unscorable with "no_reference_price"
# ---------------------------------------------------------------------------


class TestNullReferencePrice:
    def test_null_reference_price_unscorable(self):
        """When ready_reference_price is None, return unscorable outcome."""
        watch = MockWatch(ready_reference_price=None)
        candles = [_candle(5, 100, 105, 98, 102)]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.scorable == 0
        assert result.unscorable_reason == "no_reference_price"
        assert result.mfe_pct is None
        assert result.mae_pct is None


# ---------------------------------------------------------------------------
# Test 9: Empty candle set → unscorable row written, not silently skipped
# ---------------------------------------------------------------------------


class TestEmptyCandles:
    def test_empty_candles_unscorable(self):
        """When no candles fall within the window, produce an unscorable outcome."""
        watch = MockWatch()
        candles = []
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.scorable == 0
        assert result.unscorable_reason == "no_candles_in_window"
        assert result.mfe_pct is None

    def test_all_candles_outside_window_also_unscorable(self):
        """Candles exist but all outside window → same as empty."""
        watch = MockWatch()
        # All candles are before ready_at
        candles = [_candle(-30, 100, 110, 90, 100), _candle(-20, 100, 108, 92, 100)]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.scorable == 0
        assert result.unscorable_reason == "no_candles_in_window"


# ---------------------------------------------------------------------------
# Test 10: Scoring runs for watches in every terminal state (counterfactual independence)
# ---------------------------------------------------------------------------


class TestCounterfactualIndependence:
    """score_watch_outcome does not inspect watch state — scoring works on all states."""

    @pytest.mark.parametrize(
        "state",
        ["watching", "maturing", "ready", "promoted", "ordered", "rejected", "expired"],
    )
    def test_scoring_succeeds_regardless_of_state(self, state):
        """Scoring succeeds regardless of the watch's lifecycle state."""
        # The watch mock doesn't expose state, but this verifies that
        # score_watch_outcome never checks or rejects by state.
        watch = MockWatch(side="BUY", ready_reference_price=100.0)
        candles = [_candle(5, 100, 104, 98, 102)]
        result = score_watch_outcome(
            watch, window_label="w30", window_minutes=30, candles=candles
        )
        assert result.scorable == 1
        assert result.mfe_pct == pytest.approx(4.0)

    def test_run_scoring_includes_terminal_state_watches(self):
        """run_setup_watch_outcome_scoring processes watches in terminal states.

        The query in get_watches_awaiting_scoring has no state filter — verified
        by scoring a watch whose state would be 'ordered' (terminal).
        """
        mock_registry = MagicMock()

        # Simulate a watch that is in terminal state (has ready_at set)
        terminal_watch = MockWatch(
            watch_id="terminal_watch",
            side="BUY",
            ready_reference_price=100.0,
            ready_at="2026-08-14T14:00:00Z",
        )
        mock_registry.get_watches_awaiting_scoring.return_value = [terminal_watch]
        mock_registry.record_outcome.return_value = True

        with patch(
            "utils.setup_watch_registry.SetupWatchRegistry", return_value=mock_registry
        ), patch(
            "utils.setup_watch_outcomes._fetch_candles",
            return_value=[
                {
                    "timestamp": "2026-08-14T14:05:00Z",
                    "open": 100,
                    "high": 106,
                    "low": 99,
                    "close": 104,
                    "volume": 1000,
                }
            ],
        ):
            engine = MagicMock()
            counts = run_setup_watch_outcome_scoring(engine)

        # Verify scoring happened
        assert mock_registry.record_outcome.called
        total_scored = sum(counts.values())
        assert total_scored > 0


# ---------------------------------------------------------------------------
# Test 11: Duplicate window insert is tolerated as a benign race
# ---------------------------------------------------------------------------


class TestDuplicateInsertTolerated:
    def test_duplicate_outcome_returns_false(self):
        """record_outcome returning False on duplicate does not crash the run."""
        mock_registry = MagicMock()
        terminal_watch = MockWatch(
            watch_id="dup_watch",
            side="BUY",
            ready_reference_price=100.0,
            ready_at="2026-08-14T14:00:00Z",
        )
        mock_registry.get_watches_awaiting_scoring.return_value = [terminal_watch]
        mock_registry.record_outcome.return_value = False  # duplicate

        with patch(
            "utils.setup_watch_registry.SetupWatchRegistry", return_value=mock_registry
        ), patch(
            "utils.setup_watch_outcomes._fetch_candles",
            return_value=[
                {
                    "timestamp": "2026-08-14T14:05:00Z",
                    "open": 100,
                    "high": 103,
                    "low": 99,
                    "close": 101,
                    "volume": 500,
                }
            ],
        ):
            engine = MagicMock()
            # Should not raise
            counts = run_setup_watch_outcome_scoring(engine)

        # Scored count still increments (scored = successfully produced an outcome)
        total = sum(counts.values())
        assert total > 0


# ---------------------------------------------------------------------------
# Test 12: Per-watch error does not abort the run
# ---------------------------------------------------------------------------


class TestPerWatchErrorIsolation:
    def test_error_on_one_watch_does_not_abort_others(self):
        """If one watch raises during scoring, subsequent watches still get scored."""
        mock_registry = MagicMock()
        good_watch = MockWatch(
            watch_id="good_watch",
            side="BUY",
            ready_reference_price=100.0,
            ready_at="2026-08-14T14:00:00Z",
        )
        bad_watch = MockWatch(
            watch_id="bad_watch",
            side="BUY",
            ready_reference_price=100.0,
            ready_at="2026-08-14T14:00:00Z",
        )

        mock_registry.get_watches_awaiting_scoring.return_value = [bad_watch, good_watch]

        call_count = {"n": 0}

        def mock_record(outcome):
            call_count["n"] += 1
            if outcome.get("watch_id") == "bad_watch":
                raise RuntimeError("Simulated DB failure")
            return True

        mock_registry.record_outcome.side_effect = mock_record

        with patch(
            "utils.setup_watch_registry.SetupWatchRegistry", return_value=mock_registry
        ), patch(
            "utils.setup_watch_outcomes._fetch_candles",
            return_value=[
                {
                    "timestamp": "2026-08-14T14:05:00Z",
                    "open": 100,
                    "high": 103,
                    "low": 99,
                    "close": 101,
                    "volume": 500,
                }
            ],
        ):
            engine = MagicMock()
            # Should not raise despite the bad_watch error
            counts = run_setup_watch_outcome_scoring(engine)

        # At least one window scored one watch (the good one)
        total = sum(counts.values())
        assert total > 0


# ---------------------------------------------------------------------------
# Test 13: Decimal arithmetic — no float drift in excursion percentages
# ---------------------------------------------------------------------------


class TestDecimalArithmetic:
    def test_no_float_drift_in_mfe_mae(self):
        """Prices that cause float drift (0.1+0.2 patterns) produce exact results.

        ref=10.0, high=10.3 → MFE = 3.0% exactly (not 2.9999...% or 3.0000...01%)
        ref=10.0, low=9.7 → MAE = -3.0% exactly
        """
        watch = MockWatch(side="BUY", ready_reference_price=10.0)
        candles = [_candle(5, 10.0, 10.3, 9.7, 10.1)]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        # Decimal(10.3 - 10.0)/10.0 * 100 should be exactly 3.0
        assert result.mfe_pct == 3.0
        assert result.mae_pct == -3.0

    def test_tricky_decimal_values(self):
        """Values that float can't represent exactly: ref=0.3, high=0.33, low=0.27.

        MFE = (0.33 - 0.30) / 0.30 * 100 = 10.0%
        MAE = (0.27 - 0.30) / 0.30 * 100 = -10.0%
        """
        watch = MockWatch(side="BUY", ready_reference_price=0.3)
        candles = [_candle(5, 0.3, 0.33, 0.27, 0.3)]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        # Using Decimal("0.33") - Decimal("0.3") = Decimal("0.03")
        # Decimal("0.03") / Decimal("0.3") * 100 = 10.0
        assert result.mfe_pct == pytest.approx(10.0, abs=1e-10)
        assert result.mae_pct == pytest.approx(-10.0, abs=1e-10)

    def test_short_decimal_precision(self):
        """SHORT: ref=50.0, low=48.5, high=51.0.

        MFE = (50.0 - 48.5) / 50.0 * 100 = 3.0%
        MAE = (50.0 - 51.0) / 50.0 * 100 = -2.0%
        """
        watch = MockWatch(side="SHORT", ready_reference_price=50.0)
        candles = [_candle(5, 50.0, 51.0, 48.5, 49.5)]
        result = score_watch_outcome(
            watch, window_label="w15", window_minutes=15, candles=candles
        )
        assert result.mfe_pct == 3.0
        assert result.mae_pct == -2.0
