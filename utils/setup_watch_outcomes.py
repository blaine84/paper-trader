"""Setup Watch Outcomes — counterfactual scoring for matured watches.

Measures what price did after a watch reached `ready`, independent of whether
the watch was promoted or traded. Without this, observe mode reports activity
(how many watches matured) rather than evidence (whether maturation was right).

Mirrors utils/shadow_outcomes.py in structure, cadence, and unscorable handling.

Requirements: 11.1-11.13
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from utils.gate_config import SETUP_WATCH_OUTCOME_WINDOWS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchOutcome:
    """Scored result for one watch at one window."""

    watch_id: str
    profile_id: str
    symbol: str
    side: str
    window_label: str
    window_minutes: int
    reference_price: float
    evaluated_at: str  # ISO8601
    mfe_pct: float | None
    mae_pct: float | None
    entry_zone_touched: int | None  # 0/1/None
    would_have_hit_target: int | None  # 0/1/None
    would_have_hit_stop: int | None  # 0/1/None
    scorable: int  # 0/1
    unscorable_reason: str | None
    created_at: str  # ISO8601


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_dt(value) -> datetime | None:
    """Parse an ISO8601 string into a timezone-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        s = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _unscorable(watch, window_label: str, window_minutes: int, reason: str) -> WatchOutcome:
    """Build a terminal outcome record for an unscorable watch/window."""
    ready_at = _parse_dt(watch.ready_at)
    if ready_at is not None:
        evaluated_at = (ready_at + timedelta(minutes=window_minutes)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    else:
        evaluated_at = _now_iso()

    return WatchOutcome(
        watch_id=watch.watch_id,
        profile_id=watch.profile_id,
        symbol=watch.symbol,
        side=watch.side,
        window_label=window_label,
        window_minutes=window_minutes,
        reference_price=watch.ready_reference_price or 0.0,
        evaluated_at=evaluated_at,
        mfe_pct=None,
        mae_pct=None,
        entry_zone_touched=None,
        would_have_hit_target=None,
        would_have_hit_stop=None,
        scorable=0,
        unscorable_reason=reason,
        created_at=_now_iso(),
    )


def _outcome_to_dict(outcome: WatchOutcome) -> dict:
    """Convert a WatchOutcome to a dict suitable for record_outcome()."""
    return {
        "watch_id": outcome.watch_id,
        "profile_id": outcome.profile_id,
        "symbol": outcome.symbol,
        "side": outcome.side,
        "window_label": outcome.window_label,
        "window_minutes": outcome.window_minutes,
        "reference_price": outcome.reference_price,
        "evaluated_at": outcome.evaluated_at,
        "mfe_pct": outcome.mfe_pct,
        "mae_pct": outcome.mae_pct,
        "entry_zone_touched": outcome.entry_zone_touched,
        "would_have_hit_target": outcome.would_have_hit_target,
        "would_have_hit_stop": outcome.would_have_hit_stop,
        "scorable": outcome.scorable,
        "unscorable_reason": outcome.unscorable_reason,
        "created_at": outcome.created_at,
    }


def _filter_candles_to_window(
    candles: list[dict], ready_at: datetime, window_minutes: int
) -> list[dict]:
    """Filter and sort candles to the [ready_at, ready_at + window_minutes] range."""
    window_end = ready_at + timedelta(minutes=window_minutes)
    filtered = []
    for c in candles:
        ts = _parse_dt(c.get("timestamp"))
        if ts is None:
            continue
        if ready_at <= ts <= window_end:
            filtered.append(c)
    return sorted(filtered, key=lambda c: _parse_dt(c["timestamp"]))


def _compute_entry_zone_touched(
    candles: list[dict], entry_zone_json: str | None
) -> int | None:
    """Check if any candle's range overlaps the entry zone. Returns 0/1/None."""
    if not entry_zone_json:
        return None
    try:
        zone = json.loads(entry_zone_json)
    except (json.JSONDecodeError, TypeError):
        return None

    zone_low = zone.get("low")
    zone_high = zone.get("high")
    if zone_low is None or zone_high is None:
        return None

    zone_low_d = Decimal(str(zone_low))
    zone_high_d = Decimal(str(zone_high))

    for c in candles:
        candle_low = Decimal(str(c.get("low", 0)))
        candle_high = Decimal(str(c.get("high", 0)))
        # Overlap: candle range intersects entry zone
        if candle_low <= zone_high_d and candle_high >= zone_low_d:
            return 1
    return 0


def _compute_target_stop(
    candles: list[dict], draft_geometry_json: str | None, side: str
) -> tuple[int | None, int | None]:
    """Walk candles chronologically to determine target/stop hits.

    Returns (would_have_hit_target, would_have_hit_stop).
    Single bar spanning both levels → pessimistic: would_have_hit_stop = True.
    """
    if not draft_geometry_json:
        return None, None
    try:
        geom = json.loads(draft_geometry_json)
    except (json.JSONDecodeError, TypeError):
        return None, None

    target = geom.get("target")
    stop = geom.get("stop")
    if target is None or stop is None:
        return None, None

    target_d = Decimal(str(target))
    stop_d = Decimal(str(stop))

    hit_target = False
    hit_stop = False

    for c in candles:
        candle_high = Decimal(str(c.get("high", 0)))
        candle_low = Decimal(str(c.get("low", 0)))

        if side == "BUY":
            bar_hits_target = candle_high >= target_d
            bar_hits_stop = candle_low <= stop_d
        else:  # SHORT
            bar_hits_target = candle_low <= target_d
            bar_hits_stop = candle_high >= stop_d

        if bar_hits_target and bar_hits_stop:
            # Ambiguous bar: pessimistic convention
            hit_stop = True
            break
        elif bar_hits_target:
            hit_target = True
            break
        elif bar_hits_stop:
            hit_stop = True
            break

    return (1 if hit_target else 0), (1 if hit_stop else 0)


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------


def score_watch_outcome(
    watch,
    *,
    window_label: str,
    window_minutes: int,
    candles: list[dict],
) -> WatchOutcome:
    """Score one watch for one elapsed outcome window.

    Returns a WatchOutcome (scorable or unscorable).

    Requirements: 11.1-11.8
    """
    # Unscorable: no reference price
    if watch.ready_reference_price is None:
        return _unscorable(watch, window_label, window_minutes, "no_reference_price")

    # Parse ready_at
    ready_at = _parse_dt(watch.ready_at)
    if ready_at is None:
        return _unscorable(watch, window_label, window_minutes, "no_ready_at")

    # Filter candles to the scoring window
    window_candles = _filter_candles_to_window(candles, ready_at, window_minutes)

    # Unscorable: no candles in window
    if not window_candles:
        return _unscorable(watch, window_label, window_minutes, "no_candles_in_window")

    # Compute MFE/MAE with Decimal precision
    ref = Decimal(str(watch.ready_reference_price))
    if ref == 0:
        return _unscorable(watch, window_label, window_minutes, "zero_reference_price")

    highs = [Decimal(str(c.get("high", 0))) for c in window_candles]
    lows = [Decimal(str(c.get("low", 0))) for c in window_candles]

    if watch.side == "BUY":
        mfe = (max(highs) - ref) / ref * 100  # favorable = up
        mae = (min(lows) - ref) / ref * 100  # adverse = down (negative)
    else:  # SHORT
        mfe = (ref - min(lows)) / ref * 100  # favorable = down
        mae = (ref - max(highs)) / ref * 100  # adverse = up (negative)

    # Compute entry_zone_touched
    entry_zone_touched = _compute_entry_zone_touched(
        window_candles, watch.entry_zone_json
    )

    # Compute would_have_hit_target / would_have_hit_stop
    would_have_hit_target, would_have_hit_stop = _compute_target_stop(
        window_candles, watch.draft_geometry_json, watch.side
    )

    evaluated_at = (ready_at + timedelta(minutes=window_minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    return WatchOutcome(
        watch_id=watch.watch_id,
        profile_id=watch.profile_id,
        symbol=watch.symbol,
        side=watch.side,
        window_label=window_label,
        window_minutes=window_minutes,
        reference_price=float(ref),
        evaluated_at=evaluated_at,
        mfe_pct=float(mfe),
        mae_pct=float(mae),
        entry_zone_touched=entry_zone_touched,
        would_have_hit_target=would_have_hit_target,
        would_have_hit_stop=would_have_hit_stop,
        scorable=1,
        unscorable_reason=None,
        created_at=_now_iso(),
    )


# ---------------------------------------------------------------------------
# Candle fetching
# ---------------------------------------------------------------------------


def _fetch_candles(symbol: str, start: datetime, end: datetime) -> list[dict]:
    """Fetch 1-minute candles for the given symbol and time range.

    Uses FinnhubClient (which internally tries Alpaca → yfinance → Finnhub).
    Returns a list of dicts with keys: timestamp, open, high, low, close, volume.
    """
    try:
        from utils.finnhub_client import FinnhubClient

        now_utc = datetime.now(timezone.utc)
        days = max(1, int((now_utc - start.astimezone(timezone.utc)).total_seconds() // 86400) + 2)
        data = FinnhubClient().get_candles(symbol, resolution="1", days=days)
        if not data:
            return []

        records: list[dict] = []
        timestamps = data.get("timestamps", [])
        opens = data.get("open", [])
        highs = data.get("high", [])
        lows = data.get("low", [])
        closes = data.get("close", [])
        volumes = data.get("volume", [])

        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc) + timedelta(minutes=2)

        for i, ts in enumerate(timestamps):
            candle_ts = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            if start_utc <= candle_ts <= end_utc:
                records.append({
                    "timestamp": candle_ts,
                    "open": opens[i] if i < len(opens) else None,
                    "high": highs[i] if i < len(highs) else None,
                    "low": lows[i] if i < len(lows) else None,
                    "close": closes[i] if i < len(closes) else None,
                    "volume": volumes[i] if i < len(volumes) else 0,
                })
        return sorted(records, key=lambda c: c["timestamp"])
    except Exception as exc:
        log.warning(
            "Setup watch outcome: candle fetch failed for %s: %s", symbol, exc
        )
        return []


# ---------------------------------------------------------------------------
# Batch scoring orchestration
# ---------------------------------------------------------------------------


def run_setup_watch_outcome_scoring(engine) -> dict[str, int]:
    """Score all eligible watches across all outcome windows.

    For each window in SETUP_WATCH_OUTCOME_WINDOWS:
      - Select watches awaiting scoring (ready_at old enough, no existing row)
      - Fetch candles for the window range
      - Score each watch
      - Record the outcome (idempotent via unique index)

    Watches are scored regardless of terminal state (counterfactual independence).
    Per-watch try/except ensures one failure never aborts the rest.

    Returns dict of {window_label: count_scored} for observability.

    Requirements: 11.9-11.13
    """
    from utils.setup_watch_registry import SetupWatchRegistry

    registry = SetupWatchRegistry(engine)
    counts: dict[str, int] = {}

    for window_label, window_minutes in SETUP_WATCH_OUTCOME_WINDOWS:
        scored = 0
        try:
            watches = registry.get_watches_awaiting_scoring(window_label, window_minutes)
        except Exception as exc:
            log.warning(
                "Setup watch outcome: failed to query watches for %s: %s",
                window_label, exc,
            )
            counts[window_label] = 0
            continue

        for watch in watches:
            try:
                # Fetch candles covering the scoring window
                ready_at = _parse_dt(watch.ready_at)
                if ready_at is None:
                    outcome = _unscorable(watch, window_label, window_minutes, "no_ready_at")
                else:
                    window_end = ready_at + timedelta(minutes=window_minutes)
                    candles = _fetch_candles(watch.symbol, ready_at, window_end)
                    outcome = score_watch_outcome(
                        watch,
                        window_label=window_label,
                        window_minutes=window_minutes,
                        candles=candles,
                    )

                registry.record_outcome(_outcome_to_dict(outcome))
                scored += 1
            except Exception as exc:
                log.warning(
                    "Setup watch outcome: failed to score watch %s for %s: %s",
                    watch.watch_id, window_label, exc,
                )
                continue

        counts[window_label] = scored
        if scored > 0:
            log.info(
                "Setup watch outcomes: scored %d watches for window %s",
                scored, window_label,
            )

    return counts
