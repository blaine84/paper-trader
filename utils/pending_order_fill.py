"""Pure crossing detection and fill pricing for pending limit orders.

No I/O, no database, no provider calls — everything here is a deterministic
function of its arguments, which is what makes the safety properties testable
exhaustively.

All price arithmetic uses ``Decimal`` with ``Context(prec=28,
ROUND_HALF_UP)``, matching ``utils/entry_zone.py`` and
``utils/geometry_calculator.py``.

The central safety property lives in :func:`eligible_bars`: a bar is only
eligible when its timestamp is **strictly after** the order's creation. Bar
timestamps mark the bar's START, so a bar straddling creation may have printed
its low or high before the order existed. Honoring it would manufacture exactly
the stale favorable fill this whole feature exists to prevent.

Requirements: 4.2, 4.4, 4.5, 4.6, 4.7, 4.8, 4.10, 4.12, 5.1, 5.2, 5.3, 5.4,
              5.7, 10.5, 13.3, 13.4, 15.1, 15.2, 15.9
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Context, Decimal, InvalidOperation, ROUND_HALF_UP

from utils.pending_order_time import to_utc

logger = logging.getLogger(__name__)

# Fixed Decimal context — consistent with entry_zone / geometry_calculator.
_CTX = Context(prec=28, rounding=ROUND_HALF_UP)

# The only fill policy in v1. Recorded on every fill event so a later policy
# change stays distinguishable in the audit trail.
FILL_POLICY_LIMIT_PRICE = "limit_price"

_LONG = "BUY"
_SHORT = "SHORT"

__all__ = [
    "Bar",
    "CrossingResult",
    "FILL_POLICY_LIMIT_PRICE",
    "bars_from_candles",
    "detect_crossing",
    "eligible_bars",
    "resolve_fill_price",
]


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bar:
    """A single OHLC bar with an aware-UTC start timestamp."""

    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


@dataclass(frozen=True)
class CrossingResult:
    """Outcome of evaluating a bar series against a resting limit price."""

    crossed: bool
    bar: Bar | None
    gap_through: bool

    # Near-miss telemetry, recorded on expiry so review can distinguish
    # "nowhere close" from "missed by a cent" (Requirement 10.5).
    closest_approach_price: Decimal | None
    closest_approach_distance: Decimal | None

    # Watermark advance target. Set whenever any bar was evaluated, whether or
    # not a crossing occurred, so repeated ticks do not rescan the same bars.
    newest_bar_ts: datetime | None

    @classmethod
    def empty(cls) -> "CrossingResult":
        """No evaluable bars — leaves the order untouched."""
        return cls(
            crossed=False,
            bar=None,
            gap_through=False,
            closest_approach_price=None,
            closest_approach_distance=None,
            newest_bar_ts=None,
        )


# ---------------------------------------------------------------------------
# Provider payload conversion
# ---------------------------------------------------------------------------


def bars_from_candles(candles: dict | None) -> list[Bar]:
    """Convert ``FinnhubClient.get_candles()`` parallel arrays into sorted Bars.

    ``get_candles()`` returns ``{symbol, resolution, timestamps, open, high,
    low, close, volume, source}`` with parallel lists, or ``{}`` when every
    provider failed.

    Ragged payloads are tolerated rather than fatal: any index with a missing,
    non-numeric, or non-finite field is dropped and the rest are kept. Provider
    payloads are occasionally uneven, and losing one bar is far better than
    losing the whole tick.

    Args:
        candles: The provider payload, or None/empty.

    Returns:
        Bars sorted ascending by timestamp. Empty list when nothing is usable.
    """
    if not candles:
        return []

    timestamps = candles.get("timestamps") or []
    opens = candles.get("open") or []
    highs = candles.get("high") or []
    lows = candles.get("low") or []
    closes = candles.get("close") or []

    # Arrays should be parallel; if a provider truncates one, use the shortest
    # so index access can never go out of range.
    usable = min(len(timestamps), len(opens), len(highs), len(lows), len(closes))
    if usable == 0:
        return []

    if not (
        len(timestamps) == len(opens) == len(highs) == len(lows) == len(closes)
    ):
        logger.warning(
            "Ragged candle payload for %s: lengths ts=%d o=%d h=%d l=%d c=%d; "
            "using first %d",
            candles.get("symbol"),
            len(timestamps), len(opens), len(highs), len(lows), len(closes),
            usable,
        )

    bars: list[Bar] = []
    dropped = 0
    for i in range(usable):
        try:
            ts = to_utc(timestamps[i])
            if ts is None:
                dropped += 1
                continue
            bar = Bar(
                ts=ts,
                open=_to_decimal(opens[i]),
                high=_to_decimal(highs[i]),
                low=_to_decimal(lows[i]),
                close=_to_decimal(closes[i]),
            )
        except (ValueError, TypeError, InvalidOperation):
            dropped += 1
            continue

        # A bar whose low exceeds its high is incoherent; trusting it could
        # fabricate a crossing that never happened.
        if bar.low > bar.high:
            dropped += 1
            continue

        bars.append(bar)

    if dropped:
        logger.warning(
            "Dropped %d unusable bar(s) of %d for %s",
            dropped, usable, candles.get("symbol"),
        )

    bars.sort(key=lambda b: b.ts)
    return bars


# ---------------------------------------------------------------------------
# Window and watermark filtering — the anti-stale-fill guarantee
# ---------------------------------------------------------------------------


def eligible_bars(
    bars: list[Bar],
    *,
    created_at: datetime,
    expires_at: datetime,
    watermark: datetime | None = None,
) -> list[Bar]:
    """Filter bars to those eligible to fill an order.

    A bar qualifies when all of the following hold:

    - ``ts > created_at``  — **strict**. Bar timestamps mark the bar's start, so
      a bar straddling creation may have printed its extreme before the order
      existed. Honoring it would be a stale favorable fill. The cost is up to
      one bar interval of latency after creation, which is accepted.
    - ``ts <= expires_at`` — nothing outside the active window may fill.
    - ``ts > watermark``   — when set, skips bars already evaluated on an
      earlier tick, which bounds work and keeps evaluation idempotent.

    All three bounds are normalized through ``to_utc()`` on entry, because
    callers legitimately hand over naive values read back from SQLite.

    Args:
        bars: Candidate bars, in any order.
        created_at: When the order was created.
        expires_at: End of the active window, inclusive.
        watermark: Newest bar already evaluated, or None.

    Returns:
        Eligible bars, sorted ascending by timestamp.
    """
    if not bars:
        return []

    lower = to_utc(created_at)
    upper = to_utc(expires_at)
    mark = to_utc(watermark)

    if lower is None or upper is None:
        raise ValueError("created_at and expires_at are required to filter bars")

    selected = [
        bar
        for bar in bars
        if bar.ts > lower and bar.ts <= upper and (mark is None or bar.ts > mark)
    ]
    selected.sort(key=lambda b: b.ts)
    return selected


# ---------------------------------------------------------------------------
# Crossing detection
# ---------------------------------------------------------------------------


def detect_crossing(
    bars: list[Bar],
    *,
    side: str,
    limit_price: Decimal | float | str,
    gap_through_pct: Decimal | float | str,
) -> CrossingResult:
    """Find the earliest bar that crossed the limit price.

    Crossing predicate:

    - ``BUY``   crosses when ``bar.low <= limit_price``
    - ``SHORT`` crosses when ``bar.high >= limit_price``

    Returns the **earliest** crossing bar, never the most favorable one. The
    fill should reflect the first opportunity the market offered, not the best
    one available in hindsight.

    Gap-through detection, evaluated only on the crossing bar and
    direction-aware:

    - ``BUY``   gapped when ``bar.open < limit * (1 - gap_through_pct)``
    - ``SHORT`` gapped when ``bar.open > limit * (1 + gap_through_pct)``

    A gap means the market jumped past the level rather than trading down to
    it, which invalidates the stop and target derived from the pre-gap
    structure. The caller cancels such orders rather than filling them.

    Args:
        bars: Already window-filtered bars (see :func:`eligible_bars`).
        side: ``"BUY"`` or ``"SHORT"``.
        limit_price: The resting limit price.
        gap_through_pct: Fractional gap threshold (0.015 == 1.5%).

    Returns:
        A CrossingResult. ``newest_bar_ts`` is populated whenever any bar was
        evaluated, so the caller can advance its watermark regardless of
        outcome.
    """
    normalized_side = _normalize_side(side)
    if not bars:
        return CrossingResult.empty()

    limit = _to_decimal(limit_price)
    gap_pct = _to_decimal(gap_through_pct)

    ordered = sorted(bars, key=lambda b: b.ts)
    newest_ts = ordered[-1].ts

    crossing_bar: Bar | None = None
    best_price: Decimal | None = None

    for bar in ordered:
        extreme = bar.low if normalized_side == _LONG else bar.high

        # Track the nearest approach across every evaluated bar, including
        # after a crossing is found, so expiry telemetry is complete.
        if best_price is None:
            best_price = extreme
        elif normalized_side == _LONG:
            best_price = min(best_price, extreme)
        else:
            best_price = max(best_price, extreme)

        if crossing_bar is None and _crosses(normalized_side, extreme, limit):
            crossing_bar = bar

    distance = None
    if best_price is not None:
        # Signed: positive means the market never reached the limit.
        if normalized_side == _LONG:
            distance = _CTX.subtract(best_price, limit)
        else:
            distance = _CTX.subtract(limit, best_price)

    if crossing_bar is None:
        return CrossingResult(
            crossed=False,
            bar=None,
            gap_through=False,
            closest_approach_price=best_price,
            closest_approach_distance=distance,
            newest_bar_ts=newest_ts,
        )

    return CrossingResult(
        crossed=True,
        bar=crossing_bar,
        gap_through=_is_gap_through(
            normalized_side, crossing_bar.open, limit, gap_pct
        ),
        closest_approach_price=best_price,
        closest_approach_distance=distance,
        newest_bar_ts=newest_ts,
    )


# ---------------------------------------------------------------------------
# Fill pricing
# ---------------------------------------------------------------------------


def resolve_fill_price(
    bar: Bar,
    *,
    side: str,
    limit_price: Decimal | float | str,
) -> Decimal:
    """Resolve the simulated fill price under the v1 ``limit_price`` policy.

    Always returns the limit price, deliberately ignoring the bar's actual
    prices. This is the conservative direction: a resting buy limit that the
    market gapped through would in reality fill *better* than the limit, so
    recording the limit understates performance rather than inflating it. That
    satisfies the no-hindsight-fill rule.

    Large gaps never reach here — :func:`detect_crossing` flags them as
    ``gap_through`` and the caller cancels the order instead of filling it.

    ``bar`` and ``side`` are part of the signature so that additional policies
    can be added later without changing call sites. v1 does not read them.

    Args:
        bar: The crossing bar (unused in v1; kept for interface stability).
        side: ``"BUY"`` or ``"SHORT"`` (validated, then unused in v1).
        limit_price: The order's limit price.

    Returns:
        The fill price as a Decimal — always equal to ``limit_price``.
    """
    _normalize_side(side)  # validate even though v1 does not branch on it
    del bar  # v1 policy is bar-independent by design
    return _to_decimal(limit_price)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _normalize_side(side: str) -> str:
    """Validate and canonicalize an order side."""
    normalized = str(side).strip().upper()
    if normalized not in (_LONG, _SHORT):
        raise ValueError(f"Unsupported side {side!r}; expected 'BUY' or 'SHORT'")
    return normalized


def _crosses(side: str, extreme: Decimal, limit: Decimal) -> bool:
    """Whether a bar extreme reached the limit. Boundary is inclusive."""
    if side == _LONG:
        return extreme <= limit
    return extreme >= limit


def _is_gap_through(
    side: str, bar_open: Decimal, limit: Decimal, gap_pct: Decimal
) -> bool:
    """Whether the crossing bar opened beyond the limit by more than the gap."""
    if gap_pct < 0:
        raise ValueError(f"gap_through_pct must be non-negative, got {gap_pct}")

    tolerance = _CTX.multiply(limit, gap_pct)
    if side == _LONG:
        return bar_open < _CTX.subtract(limit, tolerance)
    return bar_open > _CTX.add(limit, tolerance)


def _to_decimal(value) -> Decimal:
    """Convert to Decimal via str, rejecting non-finite values.

    ``Decimal(str(value))`` is the codebase idiom (see plan_executor), and it
    avoids binary float artifacts that direct ``Decimal(float)`` introduces.
    """
    if value is None:
        raise ValueError("Cannot convert None to Decimal")
    if isinstance(value, bool):
        raise ValueError(f"Refusing to convert bool {value!r} to a price")

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Cannot convert non-finite value {value!r} to Decimal")

    result = _CTX.create_decimal(Decimal(str(value)))
    if not result.is_finite():
        raise ValueError(f"Cannot convert non-finite value {value!r} to Decimal")
    return result
