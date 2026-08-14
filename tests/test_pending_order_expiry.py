"""Tests for utils/pending_order_expiry.py.

The clamp that matters most is the ENTRY_WINDOW_LIMITS one: without it, a
resting order could fill after the setup's own entry window closed, which would
make pending orders a hole in an existing live rule.

Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 15.6
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pytz import timezone as tz

from utils.pending_order_expiry import (
    bar_interval_minutes,
    resolve_expiry,
    session_bounds_utc,
)

EASTERN = tz("America/New_York")


def et(year, month, day, hour, minute) -> datetime:
    """An aware UTC datetime built from an ET wall clock."""
    return EASTERN.localize(datetime(year, month, day, hour, minute)).astimezone(
        tz("UTC")
    )


def as_et(moment: datetime) -> datetime:
    return moment.astimezone(EASTERN)


# A regular Friday inside US DST (EDT, UTC-4).
TRADING_DAY = (2026, 8, 14)


# ---------------------------------------------------------------------------
# Session bounds
# ---------------------------------------------------------------------------


def test_session_bounds_are_930_to_1600_eastern():
    open_utc, close_utc = session_bounds_utc(et(*TRADING_DAY, 12, 0))

    assert (as_et(open_utc).hour, as_et(open_utc).minute) == (9, 30)
    assert (as_et(close_utc).hour, as_et(close_utc).minute) == (16, 0)
    assert open_utc.tzinfo is not None and close_utc.tzinfo is not None


@pytest.mark.parametrize(
    "date_parts,expected_utc_open_hour,label",
    [
        ((2026, 8, 14), 13, "EDT: 09:30 ET == 13:30 UTC"),
        ((2026, 1, 15), 14, "EST: 09:30 ET == 14:30 UTC"),
    ],
)
def test_session_bounds_handle_dst(date_parts, expected_utc_open_hour, label):
    """Localizing the wall clock, not offsetting UTC, is what makes this right."""
    open_utc, _ = session_bounds_utc(et(*date_parts, 12, 0))
    assert open_utc.hour == expected_utc_open_hour, label


def test_session_bounds_accept_naive_input():
    open_utc, close_utc = session_bounds_utc(datetime(2026, 8, 14, 17, 0))
    assert open_utc < close_utc


# ---------------------------------------------------------------------------
# Clamp 1 — setup-specific base window
# ---------------------------------------------------------------------------


def test_setup_specific_window_is_used_when_configured():
    """technical_breakout is configured at 120 minutes."""
    created = et(*TRADING_DAY, 10, 0)
    expires = resolve_expiry(created_at_utc=created, setup_type="technical_breakout")

    assert expires == created + timedelta(minutes=120)


def test_default_window_is_used_for_unconfigured_setups():
    """An unlisted setup falls back to PENDING_ORDER_DEFAULT_EXPIRY_MINUTES (120)."""
    created = et(*TRADING_DAY, 10, 0)
    expires = resolve_expiry(created_at_utc=created, setup_type="some_new_setup")

    assert expires == created + timedelta(minutes=120)


def test_none_setup_type_falls_back_to_the_default():
    created = et(*TRADING_DAY, 10, 0)
    expires = resolve_expiry(created_at_utc=created, setup_type=None)

    assert expires == created + timedelta(minutes=120)


def test_momentum_fade_uses_its_shorter_window():
    """Configured at 45 minutes, and not in ENTRY_WINDOW_LIMITS."""
    created = et(*TRADING_DAY, 11, 0)
    expires = resolve_expiry(created_at_utc=created, setup_type="momentum_fade")

    assert expires == created + timedelta(minutes=45)


# ---------------------------------------------------------------------------
# Clamp 2 — ENTRY_WINDOW_LIMITS
# ---------------------------------------------------------------------------


def test_gap_and_go_created_at_1015_clamps_to_1030_not_1045():
    """gap_and_go: 30-minute base window, but ENTRY_WINDOW_LIMITS caps at
    09:30 + 60 = 10:30 ET. Without this clamp the order would rest to 10:45 and
    could fill after the setup's own entry window closed.
    """
    created = et(*TRADING_DAY, 10, 15)
    expires = resolve_expiry(created_at_utc=created, setup_type="gap_and_go")

    assert expires is not None
    assert (as_et(expires).hour, as_et(expires).minute) == (10, 30)
    assert expires < created + timedelta(minutes=30)


@pytest.mark.parametrize("setup_type", ["gap_and_go", "orb", "short_squeeze"])
def test_every_entry_window_limited_setup_is_clamped(setup_type):
    """All three ENTRY_WINDOW_LIMITS setups cap at 10:30 ET."""
    created = et(*TRADING_DAY, 10, 20)
    expires = resolve_expiry(created_at_utc=created, setup_type=setup_type)

    assert expires is not None
    assert (as_et(expires).hour, as_et(expires).minute) == (10, 30)


def test_entry_window_clamp_does_not_extend_a_shorter_base_window():
    """The clamp is a ceiling, never a floor."""
    created = et(*TRADING_DAY, 9, 35)
    expires = resolve_expiry(created_at_utc=created, setup_type="gap_and_go")

    # 30-minute base lands at 10:05, well inside the 10:30 entry cap.
    assert expires == created + timedelta(minutes=30)
    assert (as_et(expires).hour, as_et(expires).minute) == (10, 5)


def test_gap_and_go_created_past_its_entry_window_is_unusable():
    """Created at 10:45 ET, after the 10:30 cap — no valid window remains."""
    created = et(*TRADING_DAY, 10, 45)
    assert resolve_expiry(created_at_utc=created, setup_type="gap_and_go") is None


def test_unlimited_setup_is_not_clamped_by_entry_window():
    created = et(*TRADING_DAY, 10, 15)
    expires = resolve_expiry(created_at_utc=created, setup_type="vwap_reclaim")

    # vwap_reclaim: 60-minute window, absent from ENTRY_WINDOW_LIMITS.
    assert expires == created + timedelta(minutes=60)


# ---------------------------------------------------------------------------
# Clamp 3 — session close
# ---------------------------------------------------------------------------


def test_technical_breakout_created_at_1500_clamps_to_the_close():
    """120-minute base would reach 17:00 ET; must clamp to 16:00."""
    created = et(*TRADING_DAY, 15, 0)
    expires = resolve_expiry(created_at_utc=created, setup_type="technical_breakout")

    assert expires is not None
    assert (as_et(expires).hour, as_et(expires).minute) == (16, 0)


def test_no_expiry_ever_falls_outside_the_creating_session():
    """No overnight resting in v1."""
    for hour, minute in [(9, 31), (10, 0), (12, 30), (14, 45), (15, 30), (15, 55)]:
        created = et(*TRADING_DAY, hour, minute)
        expires = resolve_expiry(
            created_at_utc=created, setup_type="technical_breakout"
        )
        if expires is None:
            continue
        _, session_close = session_bounds_utc(created)
        assert expires <= session_close
        assert as_et(expires).date() == as_et(created).date()


def test_creation_after_the_close_is_unusable():
    created = et(*TRADING_DAY, 16, 30)
    assert resolve_expiry(created_at_utc=created, setup_type="technical_breakout") is None


def test_creation_exactly_at_the_close_is_unusable():
    created = et(*TRADING_DAY, 16, 0)
    assert resolve_expiry(created_at_utc=created, setup_type="technical_breakout") is None


# ---------------------------------------------------------------------------
# Window-too-short
# ---------------------------------------------------------------------------


def test_creation_at_1559_returns_none_for_a_one_minute_bar():
    """A window of exactly one bar interval is rejected.

    eligible_bars() requires `created_at < ts <= expires_at`, and bar timestamps
    sit on interval boundaries. Creating at 15:59:00 with expiry clamped to
    16:00:00 leaves exactly one candidate boundary — 16:00:00 — which is the bar
    *starting at the close* and therefore never present in regular-session data.
    The 15:59:00 bar is excluded by the strict lower bound. So no bar can ever
    fill this order, and returning None is correct.
    """
    created = et(*TRADING_DAY, 15, 59)
    result = resolve_expiry(created_at_utc=created, setup_type="technical_breakout")
    assert result is None


def test_window_must_exceed_one_bar_interval_regardless_of_alignment():
    """The >1-interval rule is what makes usability alignment-independent.

    A window of exactly one interval may or may not contain a real boundary
    depending on whether creation fell mid-bar. Requiring strictly more than one
    interval guarantees at least one boundary falls inside either way, so
    resolve_expiry() never hands back a window that cannot possibly fill.
    """
    interval = timedelta(minutes=bar_interval_minutes())

    for minute, second in [(59, 0), (59, 30), (58, 45)]:
        created = EASTERN.localize(
            datetime(*TRADING_DAY, 15, minute, second)
        ).astimezone(tz("UTC"))
        expires = resolve_expiry(
            created_at_utc=created, setup_type="technical_breakout"
        )
        if expires is not None:
            assert expires - created > interval, (
                f"created 15:{minute}:{second} produced a window of "
                f"{expires - created}, which is not longer than one bar"
            )


def test_creation_at_1558_leaves_a_usable_window():
    created = et(*TRADING_DAY, 15, 58)
    expires = resolve_expiry(created_at_utc=created, setup_type="technical_breakout")

    assert expires is not None
    assert expires - created >= timedelta(minutes=bar_interval_minutes())


# ---------------------------------------------------------------------------
# Premarket creation
# ---------------------------------------------------------------------------


def test_premarket_creation_anchors_the_window_at_the_open():
    """An order created at 08:00 ET must not expire before the bell.

    Anchoring the base window at max(created, open) gives it a full window of
    tradable time instead of a window that elapses during premarket.
    """
    created = et(*TRADING_DAY, 8, 0)
    expires = resolve_expiry(created_at_utc=created, setup_type="gap_and_go")

    assert expires is not None
    session_open, _ = session_bounds_utc(created)
    assert expires > session_open, "must not expire before the market opens"
    assert (as_et(expires).hour, as_et(expires).minute) == (10, 0)


def test_premarket_gap_and_go_still_respects_the_entry_window():
    created = et(*TRADING_DAY, 7, 0)
    expires = resolve_expiry(created_at_utc=created, setup_type="gap_and_go")

    assert expires is not None
    assert as_et(expires) <= as_et(et(*TRADING_DAY, 10, 30))


def test_premarket_creation_of_a_long_window_still_clamps_to_close():
    created = et(*TRADING_DAY, 4, 0)
    expires = resolve_expiry(created_at_utc=created, setup_type="technical_breakout")

    assert expires is not None
    _, session_close = session_bounds_utc(created)
    assert expires <= session_close


# ---------------------------------------------------------------------------
# Non-trading days
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "date_parts,label",
    [
        ((2026, 8, 15), "Saturday"),
        ((2026, 8, 16), "Sunday"),
    ],
)
def test_weekend_creation_is_unusable(date_parts, label):
    created = et(*date_parts, 12, 0)
    assert resolve_expiry(created_at_utc=created, setup_type="technical_breakout") is None, label


# ---------------------------------------------------------------------------
# DST correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "date_parts,label",
    [
        ((2026, 8, 14), "EDT (summer)"),
        ((2026, 1, 15), "EST (winter)"),
        ((2026, 3, 9), "first trading day after spring forward"),
        ((2026, 11, 2), "first trading day after fall back"),
    ],
)
def test_expiry_lands_on_the_correct_et_wall_clock_across_dst(date_parts, label):
    """A 120-minute window from 10:00 ET must end at 12:00 ET year-round."""
    created = et(*date_parts, 10, 0)
    expires = resolve_expiry(created_at_utc=created, setup_type="technical_breakout")

    assert expires is not None, label
    assert (as_et(expires).hour, as_et(expires).minute) == (12, 0), label


def test_session_close_clamp_is_correct_in_both_dst_regimes():
    for date_parts, expected_utc_hour in [((2026, 8, 14), 20), ((2026, 1, 15), 21)]:
        created = et(*date_parts, 15, 0)
        expires = resolve_expiry(
            created_at_utc=created, setup_type="technical_breakout"
        )
        assert expires is not None
        assert expires.hour == expected_utc_hour
        assert (as_et(expires).hour, as_et(expires).minute) == (16, 0)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_naive_created_at_is_treated_as_utc():
    aware = et(*TRADING_DAY, 10, 0)
    naive = aware.replace(tzinfo=None)

    assert resolve_expiry(
        created_at_utc=naive, setup_type="technical_breakout"
    ) == resolve_expiry(created_at_utc=aware, setup_type="technical_breakout")


def test_returned_expiry_is_always_aware():
    expires = resolve_expiry(
        created_at_utc=et(*TRADING_DAY, 10, 0), setup_type="technical_breakout"
    )
    assert expires is not None
    assert expires.tzinfo is not None


def test_missing_created_at_raises():
    with pytest.raises(ValueError):
        resolve_expiry(created_at_utc=None, setup_type="technical_breakout")


# ---------------------------------------------------------------------------
# bar_interval_minutes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resolution,expected",
    [("1", 1), ("5", 5), ("15", 15), ("30", 30), ("60", 60)],
)
def test_known_resolutions_map_to_minutes(resolution, expected):
    assert bar_interval_minutes(resolution) == expected


@pytest.mark.parametrize("resolution", ["D", "W", "M", "", "bogus", "7"])
def test_unknown_resolutions_fall_back_to_one_minute(resolution):
    """Falls back rather than raising: a misconfigured resolution should not
    block order creation, and get_candles() will surface the real problem."""
    assert bar_interval_minutes(resolution) == 1


def test_default_resolution_comes_from_config():
    assert bar_interval_minutes() == 1
