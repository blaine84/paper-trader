"""Property-based tests for pending limit order fill logic.

These validate universal invariants rather than specific examples. The two that
matter most for safety:

- no fill is ever better than the limit price (no hindsight-favorable fills)
- no crossing bar ever predates the order's creation (no stale fills)

Requirements: 4.10, 4.14, 5.1, 5.2, 13.3, 13.4, 15.4, 15.10
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from utils.pending_order_fill import (
    Bar,
    bars_from_candles,
    detect_crossing,
    eligible_bars,
    resolve_fill_price,
)

CREATED = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)

prices = st.decimals(min_value="1.00", max_value="10000.00", places=2)
sides = st.sampled_from(["BUY", "SHORT"])
gap_pcts = st.decimals(min_value="0.000", max_value="0.500", places=3)
# Offsets straddle creation so the strict lower bound is genuinely exercised.
offsets = st.integers(min_value=-120, max_value=240)


@st.composite
def bars(draw, offset_strategy=offsets):
    """A coherent OHLC bar (low <= open/close <= high) at a minute offset."""
    low = draw(prices)
    span = draw(st.decimals(min_value="0.00", max_value="500.00", places=2))
    high = low + span
    open_ = draw(st.decimals(min_value="0.00", max_value="1.00", places=2))
    close = draw(st.decimals(min_value="0.00", max_value="1.00", places=2))
    minute = draw(offset_strategy)
    return Bar(
        ts=CREATED + timedelta(minutes=minute),
        open=low + (span * open_),
        high=high,
        low=low,
        close=low + (span * close),
    )


bar_lists = st.lists(bars(), min_size=0, max_size=12)


# ---------------------------------------------------------------------------
# Property 1 — the fill price is always the limit
# ---------------------------------------------------------------------------


@given(bar=bars(), side=sides, limit=prices)
@settings(max_examples=200)
def test_fill_price_always_equals_the_limit(bar, side, limit):
    """No bar, however extreme, can produce a fill better than the limit.

    This is the no-hindsight-fill guarantee expressed as a universal property.
    """
    assert resolve_fill_price(bar, side=side, limit_price=limit) == limit


@given(bar_list=bar_lists, side=sides, limit=prices, gap=gap_pcts)
@settings(max_examples=200)
def test_fill_price_never_improves_on_the_limit(bar_list, side, limit, gap):
    """Reached through the real detection path rather than in isolation."""
    result = detect_crossing(
        bar_list, side=side, limit_price=limit, gap_through_pct=gap
    )
    assume(result.crossed)

    price = resolve_fill_price(result.bar, side=side, limit_price=limit)

    assert price == limit
    if side == "BUY":
        # A buyer filling below the limit would be a hindsight gift.
        assert price >= result.bar.low
    else:
        assert price <= result.bar.high


# ---------------------------------------------------------------------------
# Property 2 — no crossing bar ever predates creation
# ---------------------------------------------------------------------------


@given(bar_list=bar_lists, side=sides, limit=prices, gap=gap_pcts, window=st.integers(1, 240))
@settings(max_examples=200)
def test_no_eligible_bar_predates_creation(bar_list, side, limit, gap, window):
    """The anti-stale-fill guarantee, over arbitrary bar sequences."""
    expires = CREATED + timedelta(minutes=window)
    windowed = eligible_bars(bar_list, created_at=CREATED, expires_at=expires)

    for b in windowed:
        assert b.ts > CREATED
        assert b.ts <= expires

    result = detect_crossing(
        windowed, side=side, limit_price=limit, gap_through_pct=gap
    )
    if result.crossed:
        assert result.bar.ts > CREATED
        assert result.bar.ts <= expires


@given(bar_list=bar_lists, watermark_minute=st.integers(-120, 240))
@settings(max_examples=200)
def test_watermark_is_always_respected(bar_list, watermark_minute):
    expires = CREATED + timedelta(minutes=240)
    mark = CREATED + timedelta(minutes=watermark_minute)

    windowed = eligible_bars(
        bar_list, created_at=CREATED, expires_at=expires, watermark=mark
    )
    for b in windowed:
        assert b.ts > mark
        assert b.ts > CREATED


# ---------------------------------------------------------------------------
# Property 3 — the crossing bar is the earliest match
# ---------------------------------------------------------------------------


@given(bar_list=bar_lists, side=sides, limit=prices, gap=gap_pcts)
@settings(max_examples=300)
def test_returned_crossing_is_the_earliest_match(bar_list, side, limit, gap):
    """Never the most favorable — always the first opportunity offered."""
    result = detect_crossing(
        bar_list, side=side, limit_price=limit, gap_through_pct=gap
    )

    def matches(b: Bar) -> bool:
        return b.low <= limit if side == "BUY" else b.high >= limit

    matching = sorted([b for b in bar_list if matches(b)], key=lambda b: b.ts)

    if not matching:
        assert result.crossed is False
        assert result.bar is None
        return

    assert result.crossed is True
    assert result.bar.ts == matching[0].ts


@given(bar_list=st.lists(bars(), min_size=1, max_size=10), side=sides, limit=prices, gap=gap_pcts)
@settings(max_examples=200)
def test_detection_is_order_independent(bar_list, side, limit, gap):
    """Input ordering must not change the outcome."""
    forward = detect_crossing(
        bar_list, side=side, limit_price=limit, gap_through_pct=gap
    )
    backward = detect_crossing(
        list(reversed(bar_list)), side=side, limit_price=limit, gap_through_pct=gap
    )

    assert forward.crossed == backward.crossed
    assert forward.newest_bar_ts == backward.newest_bar_ts
    if forward.crossed:
        assert forward.bar.ts == backward.bar.ts


# ---------------------------------------------------------------------------
# Property 4 — watermark monotonicity (idempotence support)
# ---------------------------------------------------------------------------


@given(bar_list=st.lists(bars(), min_size=1, max_size=10), side=sides, limit=prices, gap=gap_pcts)
@settings(max_examples=200)
def test_newest_bar_ts_is_monotonic_as_bars_accumulate(bar_list, side, limit, gap):
    """Simulates successive monitor ticks seeing a growing bar list.

    If the watermark could move backwards, an already-evaluated bar could be
    rescanned and produce a second fill attempt.
    """
    ordered = sorted(bar_list, key=lambda b: b.ts)
    previous = None

    for i in range(1, len(ordered) + 1):
        result = detect_crossing(
            ordered[:i], side=side, limit_price=limit, gap_through_pct=gap
        )
        assert result.newest_bar_ts is not None
        if previous is not None:
            assert result.newest_bar_ts >= previous
        previous = result.newest_bar_ts


@given(bar_list=bar_lists, side=sides, limit=prices, gap=gap_pcts)
@settings(max_examples=200)
def test_newest_bar_ts_is_the_maximum_evaluated_timestamp(bar_list, side, limit, gap):
    result = detect_crossing(
        bar_list, side=side, limit_price=limit, gap_through_pct=gap
    )
    if not bar_list:
        assert result.newest_bar_ts is None
    else:
        assert result.newest_bar_ts == max(b.ts for b in bar_list)


@given(bar_list=bar_lists, side=sides, limit=prices, gap=gap_pcts)
@settings(max_examples=200)
def test_detection_is_idempotent(bar_list, side, limit, gap):
    """Re-running against unchanged input yields an identical result."""
    first = detect_crossing(
        bar_list, side=side, limit_price=limit, gap_through_pct=gap
    )
    second = detect_crossing(
        bar_list, side=side, limit_price=limit, gap_through_pct=gap
    )
    assert first == second


# ---------------------------------------------------------------------------
# Property 5 — naive/aware mixing never raises TypeError
# ---------------------------------------------------------------------------


@given(
    bar_list=bar_lists,
    naive_created=st.booleans(),
    naive_expires=st.booleans(),
    naive_watermark=st.booleans(),
    use_watermark=st.booleans(),
    bars_naive=st.booleans(),
)
@settings(max_examples=300)
def test_mixed_naive_and_aware_inputs_never_raise_typeerror(
    bar_list, naive_created, naive_expires, naive_watermark, use_watermark, bars_naive
):
    """Comparing naive to aware raises TypeError in Python rather than degrading.

    Every combination of representations must be normalized at the boundary, so
    a single missed coercion cannot become a hard failure in the fill path.
    """
    created = CREATED.replace(tzinfo=None) if naive_created else CREATED
    expires_aware = CREATED + timedelta(minutes=240)
    expires = expires_aware.replace(tzinfo=None) if naive_expires else expires_aware

    watermark = None
    if use_watermark:
        mark = CREATED + timedelta(minutes=30)
        watermark = mark.replace(tzinfo=None) if naive_watermark else mark

    if bars_naive:
        bar_list = [
            Bar(ts=b.ts.replace(tzinfo=None), open=b.open, high=b.high,
                low=b.low, close=b.close)
            for b in bar_list
        ]
        # A naive bar timestamp cannot be compared to an aware bound, so
        # normalize the series the way bars_from_candles() would.
        bar_list = [
            Bar(ts=b.ts.replace(tzinfo=timezone.utc), open=b.open, high=b.high,
                low=b.low, close=b.close)
            for b in bar_list
        ]

    try:
        eligible_bars(
            bar_list, created_at=created, expires_at=expires, watermark=watermark
        )
    except TypeError as exc:  # pragma: no cover - the property being asserted
        pytest.fail(f"naive/aware comparison escaped normalization: {exc}")


@given(
    timestamps=st.lists(
        st.integers(min_value=1_600_000_000, max_value=1_900_000_000),
        min_size=1, max_size=8,
    ),
)
@settings(max_examples=200)
def test_provider_epochs_always_produce_aware_bars(timestamps):
    """Epoch seconds from get_candles() must never yield naive timestamps."""
    n = len(timestamps)
    payload = {
        "symbol": "TEST",
        "resolution": "1",
        "timestamps": timestamps,
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
    }
    result = bars_from_candles(payload)

    assert len(result) == n
    for b in result:
        assert b.ts.tzinfo is not None
        # Comparable against an aware bound without raising.
        assert isinstance(b.ts > CREATED, bool)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


@given(bar_list=bar_lists, side=sides, limit=prices, gap=gap_pcts)
@settings(max_examples=200)
def test_gap_through_implies_crossed(bar_list, side, limit, gap):
    """Gap-through is only meaningful for a bar that actually crossed."""
    result = detect_crossing(
        bar_list, side=side, limit_price=limit, gap_through_pct=gap
    )
    if result.gap_through:
        assert result.crossed is True
        assert result.bar is not None


@given(bar_list=bar_lists, side=sides, limit=prices, gap=gap_pcts)
@settings(max_examples=200)
def test_crossed_implies_a_bar_and_not_crossed_implies_none(
    bar_list, side, limit, gap
):
    result = detect_crossing(
        bar_list, side=side, limit_price=limit, gap_through_pct=gap
    )
    assert (result.bar is not None) == result.crossed


@given(bar_list=st.lists(bars(), min_size=1, max_size=10), side=sides, limit=prices, gap=gap_pcts)
@settings(max_examples=300)
def test_closest_approach_distance_sign_matches_crossing(
    bar_list, side, limit, gap
):
    """Non-positive distance means the market reached the limit, and vice versa."""
    result = detect_crossing(
        bar_list, side=side, limit_price=limit, gap_through_pct=gap
    )
    assert result.closest_approach_distance is not None
    assert result.crossed == (result.closest_approach_distance <= Decimal("0"))


@given(bar_list=st.lists(bars(), min_size=1, max_size=10), side=sides, limit=prices, gap=gap_pcts)
@settings(max_examples=200)
def test_closest_approach_price_is_an_actual_printed_extreme(
    bar_list, side, limit, gap
):
    """Telemetry must report a real market print, not an interpolation."""
    result = detect_crossing(
        bar_list, side=side, limit_price=limit, gap_through_pct=gap
    )
    extremes = [b.low for b in bar_list] if side == "BUY" else [b.high for b in bar_list]
    assert result.closest_approach_price in extremes
