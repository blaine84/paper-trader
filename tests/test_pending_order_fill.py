"""Tests for utils/pending_order_fill.py — crossing detection and fill pricing.

The load-bearing test in this file is
``test_bar_predating_creation_does_not_fill``: it encodes the anti-stale-fill
guarantee, which is the reason this feature is safe to enable at all.

Requirements: 4.4, 4.5, 4.6, 4.7, 4.8, 4.10, 5.1, 5.2, 5.3, 5.4, 10.5, 13.4
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from utils.pending_order_fill import (
    FILL_POLICY_LIMIT_PRICE,
    Bar,
    bars_from_candles,
    detect_crossing,
    eligible_bars,
    resolve_fill_price,
)

CREATED = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)
EXPIRES = CREATED + timedelta(hours=2)
LIMIT = Decimal("593.87")
GAP_PCT = Decimal("0.015")


def bar(
    minutes_after_creation: float,
    *,
    open_=600.0,
    high=601.0,
    low=599.0,
    close=600.5,
) -> Bar:
    """A Bar positioned relative to CREATED, defaulting to no crossing."""
    return Bar(
        ts=CREATED + timedelta(minutes=minutes_after_creation),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
    )


def cross(minutes: float, **kwargs) -> Bar:
    """A bar that crosses a BUY limit of 593.87 (low dips to 593.00)."""
    defaults = {"open_": 596.0, "high": 597.0, "low": 593.0, "close": 594.0}
    defaults.update(kwargs)
    return bar(minutes, **defaults)


# ---------------------------------------------------------------------------
# eligible_bars — window and watermark
# ---------------------------------------------------------------------------


def test_bar_predating_creation_does_not_fill():
    """THE anti-stale-fill guarantee.

    Bar timestamps mark the bar's START, so a bar straddling creation may have
    printed its low before the order existed. Filling from it would be exactly
    the stale favorable fill this feature is meant to prevent.
    """
    before = cross(-1)
    assert eligible_bars([before], created_at=CREATED, expires_at=EXPIRES) == []


def test_bar_exactly_at_creation_does_not_fill():
    """The lower bound is strict, not inclusive."""
    at_creation = cross(0)
    assert eligible_bars([at_creation], created_at=CREATED, expires_at=EXPIRES) == []


def test_bar_just_after_creation_is_eligible():
    just_after = cross(1)
    result = eligible_bars([just_after], created_at=CREATED, expires_at=EXPIRES)
    assert result == [just_after]


def test_bar_after_expiry_is_not_eligible():
    late = cross(121)  # EXPIRES is +120 minutes
    assert eligible_bars([late], created_at=CREATED, expires_at=EXPIRES) == []


def test_bar_exactly_at_expiry_is_eligible():
    """The upper bound is inclusive."""
    at_expiry = cross(120)
    result = eligible_bars([at_expiry], created_at=CREATED, expires_at=EXPIRES)
    assert result == [at_expiry]


def test_bars_at_or_below_the_watermark_are_skipped():
    first, second, third = cross(1), cross(2), cross(3)
    result = eligible_bars(
        [first, second, third],
        created_at=CREATED,
        expires_at=EXPIRES,
        watermark=second.ts,
    )
    assert result == [third]


def test_watermark_none_evaluates_everything_in_window():
    bars = [cross(1), cross(2)]
    result = eligible_bars(
        bars, created_at=CREATED, expires_at=EXPIRES, watermark=None
    )
    assert result == bars


def test_eligible_bars_returns_sorted_output():
    out_of_order = [cross(5), cross(1), cross(3)]
    result = eligible_bars(out_of_order, created_at=CREATED, expires_at=EXPIRES)
    assert [b.ts for b in result] == sorted(b.ts for b in result)


def test_naive_bounds_are_normalized_not_rejected():
    """Callers legitimately pass naive values read back from SQLite."""
    naive_created = CREATED.replace(tzinfo=None)
    naive_expires = EXPIRES.replace(tzinfo=None)

    result = eligible_bars(
        [cross(1)], created_at=naive_created, expires_at=naive_expires
    )
    assert len(result) == 1


def test_empty_input_returns_empty():
    assert eligible_bars([], created_at=CREATED, expires_at=EXPIRES) == []


# ---------------------------------------------------------------------------
# detect_crossing — BUY
# ---------------------------------------------------------------------------


def test_buy_fills_when_low_dips_below_limit():
    result = detect_crossing(
        [cross(1)], side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT
    )
    assert result.crossed is True
    assert result.gap_through is False


def test_buy_fills_when_low_touches_limit_exactly():
    """Boundary is inclusive — a touch is a fill."""
    touching = bar(1, open_=596.0, high=597.0, low=float(LIMIT), close=595.0)
    result = detect_crossing(
        [touching], side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT
    )
    assert result.crossed is True


def test_buy_does_not_fill_when_low_stays_one_cent_above():
    near_miss = bar(1, open_=596.0, high=597.0, low=593.88, close=595.0)
    result = detect_crossing(
        [near_miss], side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT
    )
    assert result.crossed is False
    assert result.closest_approach_distance == Decimal("0.01")


# ---------------------------------------------------------------------------
# detect_crossing — SHORT
# ---------------------------------------------------------------------------


def test_short_fills_when_high_rises_above_limit():
    short_limit = Decimal("601.00")
    rising = bar(1, open_=598.0, high=602.0, low=597.0, close=601.5)
    result = detect_crossing(
        [rising], side="SHORT", limit_price=short_limit, gap_through_pct=GAP_PCT
    )
    assert result.crossed is True


def test_short_fills_when_high_touches_limit_exactly():
    short_limit = Decimal("601.00")
    touching = bar(1, open_=598.0, high=float(short_limit), low=597.0, close=600.0)
    result = detect_crossing(
        [touching], side="SHORT", limit_price=short_limit, gap_through_pct=GAP_PCT
    )
    assert result.crossed is True


def test_short_does_not_fill_when_high_stays_below():
    short_limit = Decimal("601.00")
    near_miss = bar(1, open_=598.0, high=600.99, low=597.0, close=600.0)
    result = detect_crossing(
        [near_miss], side="SHORT", limit_price=short_limit, gap_through_pct=GAP_PCT
    )
    assert result.crossed is False
    assert result.closest_approach_distance == Decimal("0.01")


# ---------------------------------------------------------------------------
# Earliest-crossing selection
# ---------------------------------------------------------------------------


def test_multiple_crossings_return_the_earliest_not_the_lowest():
    """The fill must reflect the first opportunity, not the best one.

    The second bar dips far lower, so a "most favorable" implementation would
    pick it. Constructed so the two are distinguishable.
    """
    first = cross(1, low=593.50)
    deepest = cross(2, low=570.00)
    later = cross(3, low=590.00)

    result = detect_crossing(
        [first, deepest, later],
        side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT,
    )

    assert result.bar is first
    assert result.bar.low == Decimal("593.5")
    assert result.bar is not deepest


def test_earliest_crossing_is_chosen_regardless_of_input_order():
    first = cross(1, low=593.50)
    deepest = cross(2, low=570.00)

    result = detect_crossing(
        [deepest, first],  # reversed
        side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT,
    )
    assert result.bar is first


def test_non_crossing_bars_before_a_crossing_are_skipped():
    result = detect_crossing(
        [bar(1), bar(2), cross(3), cross(4)],
        side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT,
    )
    assert result.crossed is True
    assert result.bar.ts == CREATED + timedelta(minutes=3)


# ---------------------------------------------------------------------------
# Gap-through
# ---------------------------------------------------------------------------


def test_buy_gap_through_fires_when_open_is_far_below_limit():
    # Threshold is 1.5% of 593.87 == 8.908; open must be < 584.96 to trip it.
    gapped = bar(1, open_=580.00, high=585.0, low=575.0, close=578.0)
    result = detect_crossing(
        [gapped], side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT
    )
    assert result.crossed is True
    assert result.gap_through is True


def test_buy_gap_through_does_not_fire_just_inside_the_threshold():
    inside = bar(1, open_=585.50, high=590.0, low=584.0, close=586.0)
    result = detect_crossing(
        [inside], side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT
    )
    assert result.crossed is True
    assert result.gap_through is False


def test_short_gap_through_fires_when_open_is_far_above_limit():
    short_limit = Decimal("601.00")
    # 1.5% of 601 == 9.015; open must exceed 610.015.
    gapped = bar(1, open_=615.0, high=620.0, low=612.0, close=618.0)
    result = detect_crossing(
        [gapped], side="SHORT", limit_price=short_limit, gap_through_pct=GAP_PCT
    )
    assert result.crossed is True
    assert result.gap_through is True


def test_short_gap_through_does_not_fire_just_inside_the_threshold():
    short_limit = Decimal("601.00")
    inside = bar(1, open_=609.00, high=612.0, low=607.0, close=610.0)
    result = detect_crossing(
        [inside], side="SHORT", limit_price=short_limit, gap_through_pct=GAP_PCT
    )
    assert result.crossed is True
    assert result.gap_through is False


def test_gap_through_is_false_when_nothing_crossed():
    result = detect_crossing(
        [bar(1)], side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT
    )
    assert result.crossed is False
    assert result.gap_through is False


def test_zero_gap_threshold_flags_any_open_beyond_the_limit():
    just_below = bar(1, open_=593.86, high=595.0, low=593.0, close=594.0)
    result = detect_crossing(
        [just_below], side="BUY", limit_price=LIMIT,
        gap_through_pct=Decimal("0"),
    )
    assert result.gap_through is True


# ---------------------------------------------------------------------------
# Watermark advance
# ---------------------------------------------------------------------------


def test_newest_bar_ts_is_set_even_without_a_crossing():
    """The watermark must advance on every tick that saw bars."""
    result = detect_crossing(
        [bar(1), bar(2), bar(3)],
        side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT,
    )
    assert result.crossed is False
    assert result.newest_bar_ts == CREATED + timedelta(minutes=3)


def test_newest_bar_ts_is_the_maximum_not_the_crossing_bar():
    result = detect_crossing(
        [cross(1), bar(2), bar(3)],
        side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT,
    )
    assert result.bar.ts == CREATED + timedelta(minutes=1)
    assert result.newest_bar_ts == CREATED + timedelta(minutes=3)


def test_empty_bar_list_yields_empty_result():
    result = detect_crossing(
        [], side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT
    )
    assert result.crossed is False
    assert result.newest_bar_ts is None
    assert result.closest_approach_price is None


# ---------------------------------------------------------------------------
# Near-miss telemetry
# ---------------------------------------------------------------------------


def test_closest_approach_tracks_the_best_buy_low():
    result = detect_crossing(
        [bar(1, low=598.0), bar(2, low=595.0), bar(3, low=596.0)],
        side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT,
    )
    assert result.closest_approach_price == Decimal("595")
    assert result.closest_approach_distance == Decimal("1.13")


def test_closest_approach_tracks_the_best_short_high():
    short_limit = Decimal("601.00")
    result = detect_crossing(
        [bar(1, high=597.0), bar(2, high=600.0), bar(3, high=598.0)],
        side="SHORT", limit_price=short_limit, gap_through_pct=GAP_PCT,
    )
    assert result.closest_approach_price == Decimal("600")
    assert result.closest_approach_distance == Decimal("1")


def test_closest_approach_distance_is_negative_or_zero_when_crossed():
    result = detect_crossing(
        [cross(1, low=593.00)],
        side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT,
    )
    assert result.crossed is True
    assert result.closest_approach_distance <= 0


# ---------------------------------------------------------------------------
# resolve_fill_price — always the limit
# ---------------------------------------------------------------------------


def test_fill_price_is_the_limit_not_the_bar_low():
    deep = cross(1, low=550.00)
    price = resolve_fill_price(deep, side="BUY", limit_price=LIMIT)
    assert price == LIMIT


def test_fill_price_ignores_a_bar_that_traded_far_beyond():
    """No hindsight-favorable fill, even on an extreme bar."""
    extreme = bar(1, open_=500.0, high=505.0, low=450.0, close=460.0)
    assert resolve_fill_price(extreme, side="BUY", limit_price=LIMIT) == LIMIT


def test_fill_price_is_the_limit_for_shorts_too():
    short_limit = Decimal("601.00")
    spiking = bar(1, open_=615.0, high=650.0, low=610.0, close=640.0)
    assert (
        resolve_fill_price(spiking, side="SHORT", limit_price=short_limit)
        == short_limit
    )


def test_fill_price_accepts_float_and_string_limits():
    b = cross(1)
    assert resolve_fill_price(b, side="BUY", limit_price=593.87) == LIMIT
    assert resolve_fill_price(b, side="BUY", limit_price="593.87") == LIMIT


def test_fill_policy_constant_is_stable():
    """Recorded on every fill event; renaming it would break audit queries."""
    assert FILL_POLICY_LIMIT_PRICE == "limit_price"


# ---------------------------------------------------------------------------
# Side validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["buy", "Buy", " BUY ", "short", "SHORT"])
def test_side_is_normalized_case_and_whitespace_insensitively(side):
    result = detect_crossing(
        [bar(1)], side=side, limit_price=LIMIT, gap_through_pct=GAP_PCT
    )
    assert result.crossed in (True, False)


@pytest.mark.parametrize("side", ["LONG", "SELL", "", "CLOSE", None])
def test_unsupported_sides_raise(side):
    with pytest.raises(ValueError, match="[Uu]nsupported side"):
        detect_crossing(
            [bar(1)], side=side, limit_price=LIMIT, gap_through_pct=GAP_PCT
        )


def test_negative_gap_threshold_raises():
    with pytest.raises(ValueError, match="non-negative"):
        detect_crossing(
            [cross(1)], side="BUY", limit_price=LIMIT,
            gap_through_pct=Decimal("-0.01"),
        )


# ---------------------------------------------------------------------------
# bars_from_candles
# ---------------------------------------------------------------------------


def _candles(**overrides) -> dict:
    payload = {
        "symbol": "META",
        "resolution": "1",
        "timestamps": [1786732200, 1786732260, 1786732320],
        "open": [600.0, 599.0, 598.0],
        "high": [601.0, 600.0, 599.0],
        "low": [599.0, 598.0, 593.0],
        "close": [599.5, 598.5, 594.0],
        "volume": [1000, 1100, 1200],
        "source": "alpaca",
    }
    payload.update(overrides)
    return payload


def test_bars_from_candles_converts_all_rows():
    bars = bars_from_candles(_candles())
    assert len(bars) == 3
    assert all(b.ts.tzinfo is not None for b in bars)
    assert all(isinstance(b.low, Decimal) for b in bars)


def test_bars_from_candles_returns_sorted_bars():
    unsorted_payload = _candles(timestamps=[1786732320, 1786732200, 1786732260])
    bars = bars_from_candles(unsorted_payload)
    assert [b.ts for b in bars] == sorted(b.ts for b in bars)


@pytest.mark.parametrize("empty", [None, {}, {"timestamps": []}])
def test_bars_from_candles_handles_empty_payloads(empty):
    """get_candles() returns {} when every provider failed."""
    assert bars_from_candles(empty) == []


def test_bars_from_candles_drops_rows_with_none_fields():
    """Provider payloads are occasionally ragged; drop the row, keep the tick."""
    bars = bars_from_candles(_candles(low=[599.0, None, 593.0]))
    assert len(bars) == 2


def test_bars_from_candles_drops_non_finite_prices():
    bars = bars_from_candles(_candles(high=[601.0, float("nan"), 599.0]))
    assert len(bars) == 2


def test_bars_from_candles_drops_incoherent_bars():
    """low > high could fabricate a crossing that never happened."""
    bars = bars_from_candles(_candles(low=[599.0, 700.0, 593.0]))
    assert len(bars) == 2


def test_bars_from_candles_tolerates_truncated_arrays():
    """Uses the shortest array so index access cannot go out of range."""
    bars = bars_from_candles(_candles(close=[599.5, 598.5]))
    assert len(bars) == 2


def test_bars_from_candles_uses_decimal_not_float_arithmetic():
    """Decimal(str(x)) avoids the binary artifacts of Decimal(float)."""
    bars = bars_from_candles(_candles(low=[593.87, 598.0, 593.0]))
    assert bars[0].low == Decimal("593.87")


def test_bars_from_candles_rejects_millisecond_timestamps():
    """A ms payload would otherwise silently fail every window comparison."""
    bars = bars_from_candles(_candles(timestamps=[1786732200000] * 3))
    assert bars == []


# ---------------------------------------------------------------------------
# Integration of the three stages
# ---------------------------------------------------------------------------


def test_full_pipeline_from_provider_payload_to_fill_price():
    """The realistic path: payload -> window filter -> crossing -> price."""
    created = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)
    expires = created + timedelta(hours=2)

    payload = _candles(
        timestamps=[
            int((created - timedelta(minutes=1)).timestamp()),  # pre-creation
            int((created + timedelta(minutes=1)).timestamp()),  # no cross
            int((created + timedelta(minutes=2)).timestamp()),  # crosses
        ],
        open=[596.0, 599.0, 596.0],
        high=[597.0, 600.0, 597.0],
        low=[590.0, 598.0, 593.0],  # the pre-creation bar also crosses
        close=[591.0, 598.5, 594.0],
    )

    bars = bars_from_candles(payload)
    assert len(bars) == 3

    windowed = eligible_bars(bars, created_at=created, expires_at=expires)
    assert len(windowed) == 2, "the pre-creation bar must be excluded"

    result = detect_crossing(
        windowed, side="BUY", limit_price=LIMIT, gap_through_pct=GAP_PCT
    )
    assert result.crossed is True
    assert result.bar.ts == created + timedelta(minutes=2)
    assert result.gap_through is False

    assert resolve_fill_price(result.bar, side="BUY", limit_price=LIMIT) == LIMIT
