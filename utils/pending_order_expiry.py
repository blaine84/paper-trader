"""Active-window resolution for pending limit orders.

A resting order's lifetime is bounded by three clamps applied in order:

1. **Setup window** — how long this setup type's premise stays credible.
2. **Entry window** — ``ENTRY_WINDOW_LIMITS`` already caps when certain setups
   may be entered at all. Without this clamp, a resting order would become a
   hole in that existing rule.
3. **Session close** — 16:00 ET the same trading day. Orders never rest
   overnight in v1, so unattended positions cannot be opened into a gap.

Eastern-time reasoning is deliberately confined to this module. Everything else
in the pending-order subsystem works in aware UTC.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 15.6
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from utils.gate_config import (
    PENDING_ORDER_BAR_RESOLUTION,
    PENDING_ORDER_DEFAULT_EXPIRY_MINUTES,
    PENDING_ORDER_EXPIRY_MINUTES_BY_SETUP,
)
from utils.pending_order_time import to_utc

logger = logging.getLogger(__name__)

EASTERN_TZ_NAME = "America/New_York"

# Regular US session, Eastern.
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# ENTRY_WINDOW_LIMITS counts minutes from the 9:30 ET open.
_ENTRY_WINDOW_ANCHOR_HOUR = 9
_ENTRY_WINDOW_ANCHOR_MINUTE = 30

# Minutes per bar, by get_candles() resolution string.
_RESOLUTION_MINUTES = {
    "1": 1,
    "5": 5,
    "15": 15,
    "30": 30,
    "60": 60,
}

__all__ = [
    "bar_interval_minutes",
    "resolve_expiry",
    "session_bounds_utc",
]


def bar_interval_minutes(resolution: str | None = None) -> int:
    """Minutes covered by one bar at the configured resolution.

    Falls back to 1 minute for unrecognized values rather than raising, since a
    misconfigured resolution should not prevent order creation outright — the
    monitor's own ``get_candles()`` call will surface the real problem.
    """
    raw = PENDING_ORDER_BAR_RESOLUTION if resolution is None else resolution
    key = str(raw).strip()
    if key in _RESOLUTION_MINUTES:
        return _RESOLUTION_MINUTES[key]

    logger.warning(
        "Unrecognized bar resolution %r for pending order windows; "
        "assuming 1 minute. Daily/weekly resolutions are not meaningful for "
        "intraday resting orders.",
        raw,
    )
    return 1


def session_bounds_utc(moment: datetime) -> tuple[datetime, datetime]:
    """Regular-session open and close, in aware UTC, for the ET day of ``moment``.

    Args:
        moment: Any instant; normalized to UTC then interpreted in ET to pick
            the calendar day.

    Returns:
        ``(open_utc, close_utc)`` for 09:30–16:00 ET on that ET calendar day.
        DST is handled by localizing the naive wall-clock times rather than
        arithmetic on UTC offsets.
    """
    from pytz import timezone as _tz

    eastern = _tz(EASTERN_TZ_NAME)
    normalized = to_utc(moment)
    if normalized is None:
        raise ValueError("session_bounds_utc requires a datetime")

    local = normalized.astimezone(eastern)

    open_naive = datetime(
        local.year, local.month, local.day, MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE
    )
    close_naive = datetime(
        local.year, local.month, local.day, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE
    )

    # localize() picks the correct offset for the date, which a
    # replace(tzinfo=...) would get wrong across DST boundaries.
    open_utc = eastern.localize(open_naive).astimezone(normalized.tzinfo)
    close_utc = eastern.localize(close_naive).astimezone(normalized.tzinfo)
    return open_utc, close_utc


def resolve_expiry(
    *,
    created_at_utc: datetime,
    setup_type: str | None,
) -> datetime | None:
    """Resolve ``expires_at`` for a new pending order, or None if unusable.

    Clamps, applied in order:

    1. **Base window.** ``PENDING_ORDER_EXPIRY_MINUTES_BY_SETUP[setup_type]``
       when present, else ``PENDING_ORDER_DEFAULT_EXPIRY_MINUTES``. Anchored at
       ``max(created_at, session_open)``, so an order created premarket gets its
       full window of *tradable* time instead of expiring before the bell.
    2. **Entry window.** When ``setup_type`` appears in ``ENTRY_WINDOW_LIMITS``,
       clamp to ``09:30 ET + limit``. Otherwise a resting order could fill after
       the setup's own entry window closed, bypassing an existing rule.
    3. **Session close.** Clamp to 16:00 ET the same day. No overnight resting.

    Args:
        created_at_utc: Order creation time. Naive values are treated as UTC.
        setup_type: The setup type, used for both the base and entry clamps.

    Returns:
        Aware UTC ``expires_at``, or ``None`` when the resulting window is
        unusable — after the close, on a non-trading day, or shorter than one
        bar interval. Callers turn ``None`` into a ``window_too_short`` decline.
    """
    created = to_utc(created_at_utc)
    if created is None:
        raise ValueError("resolve_expiry requires created_at_utc")

    session_open, session_close = session_bounds_utc(created)

    # A window that has already closed can never fill.
    if created >= session_close:
        logger.debug(
            "Pending order window unusable: created %s is at or past the "
            "session close %s",
            created.isoformat(), session_close.isoformat(),
        )
        return None

    if not _is_trading_day_utc(created):
        logger.debug(
            "Pending order window unusable: %s is not a trading day",
            created.date().isoformat(),
        )
        return None

    # ── Clamp 1: base setup window, anchored at the first tradable moment ──
    minutes = _base_window_minutes(setup_type)
    anchor = max(created, session_open)
    expires = anchor + timedelta(minutes=minutes)

    # ── Clamp 2: the setup's own entry window ──
    entry_window_cap = _entry_window_cap(created, setup_type)
    if entry_window_cap is not None:
        expires = min(expires, entry_window_cap)

    # ── Clamp 3: regular session close ──
    expires = min(expires, session_close)

    # The window must be strictly LONGER than one bar interval, not merely equal
    # to it. eligible_bars() requires `created_at < ts <= expires_at`, and bar
    # timestamps sit on interval boundaries. A window of exactly one interval
    # only contains a boundary when creation happens to fall mid-bar; when
    # creation lands on a boundary the sole candidate is `created_at + interval`,
    # which for an end-of-session window is the bar starting at the close and so
    # never exists in regular-session data. Requiring more than one interval
    # guarantees at least one real boundary falls inside, whatever the alignment.
    interval = timedelta(minutes=bar_interval_minutes())
    if expires - created <= interval:
        logger.debug(
            "Pending order window unusable: %s to %s is shorter than one "
            "%s-minute bar",
            created.isoformat(), expires.isoformat(), bar_interval_minutes(),
        )
        return None

    return expires


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _base_window_minutes(setup_type: str | None) -> int:
    """Setup-specific window length, falling back to the configured default."""
    if setup_type:
        configured = PENDING_ORDER_EXPIRY_MINUTES_BY_SETUP.get(setup_type)
        if configured is not None:
            return configured
    return PENDING_ORDER_DEFAULT_EXPIRY_MINUTES


def _entry_window_cap(moment: datetime, setup_type: str | None) -> datetime | None:
    """Latest fill time allowed by ``ENTRY_WINDOW_LIMITS``, or None if unlimited.

    ``ENTRY_WINDOW_LIMITS`` lives in ``agents/portfolio_manager.py`` and is
    imported lazily: that module is ~5000 lines and imports this subsystem's
    creation path, so a module-scope import would be both heavy and circular.
    """
    if not setup_type:
        return None

    try:
        from agents.portfolio_manager import ENTRY_WINDOW_LIMITS
    except Exception:  # pragma: no cover - defensive
        logger.warning(
            "Could not load ENTRY_WINDOW_LIMITS; skipping the entry-window "
            "clamp for setup_type=%s", setup_type,
            exc_info=True,
        )
        return None

    limit_minutes = ENTRY_WINDOW_LIMITS.get(setup_type)
    if limit_minutes is None:
        return None

    from pytz import timezone as _tz

    eastern = _tz(EASTERN_TZ_NAME)
    local = moment.astimezone(eastern)
    anchor_naive = datetime(
        local.year, local.month, local.day,
        _ENTRY_WINDOW_ANCHOR_HOUR, _ENTRY_WINDOW_ANCHOR_MINUTE,
    )
    anchor_utc = eastern.localize(anchor_naive).astimezone(moment.tzinfo)
    return anchor_utc + timedelta(minutes=limit_minutes)


def _is_trading_day_utc(moment: datetime) -> bool:
    """Whether ``moment`` falls on a trading day, evaluated in ET.

    ``is_trading_day()`` expects an ET datetime. Imported from
    ``utils.position_lifecycle_governance`` rather than ``orchestrator``, which
    cannot be imported on every platform.
    """
    try:
        from pytz import timezone as _tz

        from utils.position_lifecycle_governance import is_trading_day

        return bool(is_trading_day(moment.astimezone(_tz(EASTERN_TZ_NAME))))
    except Exception:  # pragma: no cover - defensive
        # Fail open: a calendar lookup failure should not silently suppress
        # every pending order. The monitor is independently guarded by
        # _skip_outside_regular_market_job(), so a bad day cannot produce fills.
        logger.warning(
            "Trading-day check failed for %s; assuming tradable",
            moment.isoformat(), exc_info=True,
        )
        return True
