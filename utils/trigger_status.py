"""Deterministic breakout/pullback trigger-state classification."""

from __future__ import annotations

import math
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _levels_from(signal: dict, indicators: dict | None = None) -> dict[str, float]:
    levels: dict[str, float] = {}
    key_levels = signal.get("key_levels") if isinstance(signal, dict) else {}
    if isinstance(key_levels, dict):
        for key, value in key_levels.items():
            num = _safe_float(value)
            if isinstance(key, str) and num is not None and num > 0:
                levels[key.lower()] = num

    for source in (signal, indicators or {}):
        if not isinstance(source, dict):
            continue
        for key in ("support", "resistance", "vwap", "day_high", "day_low", "prior_high", "prior_low"):
            if key not in levels:
                num = _safe_float(source.get(key))
                if num is not None and num > 0:
                    levels[key] = num
    return levels


def _nearest_level(current: float, levels: dict[str, float], keys: tuple[str, ...], *, side: str) -> tuple[str | None, float | None]:
    candidates = []
    for key in keys:
        level = levels.get(key)
        if level is None or level <= 0:
            continue
        if side == "above" and level >= current * 0.985:
            candidates.append((abs(level - current), key, level))
        elif side == "below" and level <= current * 1.015:
            candidates.append((abs(level - current), key, level))
    if not candidates:
        return None, None
    _, key, level = min(candidates, key=lambda item: item[0])
    return key, level


def _active_breakout_level(current: float, levels: dict[str, float]) -> tuple[str | None, float | None]:
    breakout_levels = []
    for key in ("resistance", "day_high", "prior_high"):
        level = levels.get(key)
        if level is not None and level > 0:
            breakout_levels.append((key, level))
    if not breakout_levels:
        return None, None

    overhead = [(level - current, key, level) for key, level in breakout_levels if level >= current]
    if overhead:
        _, key, level = min(overhead, key=lambda item: item[0])
        return key, level

    # All tracked breakout levels are below current; confirm against the
    # highest cleared level instead of the easiest prior level.
    key, level = max(breakout_levels, key=lambda item: item[1])
    return key, level


def _distance_pct(current: float, level: float) -> float:
    return round(((current - level) / level) * 100, 4)


def compute_trigger_status(signal: dict, quote: dict, indicators: dict | None = None) -> dict:
    """Return deterministic breakout/pullback state for PM and dashboard context.

    This is not a trading decision and does not override Analyst direction. It
    turns existing quote/key-level data into a stable status object that PM can
    consume without parsing prose.
    """
    indicators = indicators or {}
    current = _safe_float(
        quote.get("price") if isinstance(quote, dict) else None
    ) or _safe_float(signal.get("current_price") if isinstance(signal, dict) else None)
    levels = _levels_from(signal if isinstance(signal, dict) else {}, indicators)
    if current is None or current <= 0:
        return {
            "status": "unknown",
            "entry_trigger": "unknown",
            "reason": "missing_current_price",
            "breakout": {"status": "unknown"},
            "pullback": {"status": "unknown"},
        }

    resistance_name, resistance = _active_breakout_level(current, levels)
    support_name, support = _nearest_level(
        current, levels, ("vwap", "support", "day_low", "prior_low"), side="below"
    )
    vwap = levels.get("vwap")

    breakout = {"status": "none"}
    if resistance is not None:
        dist = _distance_pct(current, resistance)
        if dist >= 0.10:
            breakout = {
                "status": "confirmed",
                "level_name": resistance_name,
                "level": resistance,
                "distance_pct": dist,
                "reason": "price_above_resistance",
            }
        elif dist >= -0.75:
            breakout = {
                "status": "approaching",
                "level_name": resistance_name,
                "level": resistance,
                "distance_pct": dist,
                "reason": "price_within_0.75pct_of_resistance",
            }
        else:
            breakout = {
                "status": "waiting",
                "level_name": resistance_name,
                "level": resistance,
                "distance_pct": dist,
                "reason": "price_below_resistance",
            }

    pullback = {"status": "none"}
    if support is not None:
        dist = _distance_pct(current, support)
        if dist < -0.25:
            pullback = {
                "status": "failed",
                "level_name": support_name,
                "level": support,
                "distance_pct": dist,
                "reason": "price_lost_support_level",
            }
        elif abs(dist) <= 0.45:
            pullback = {
                "status": "at_level",
                "level_name": support_name,
                "level": support,
                "distance_pct": dist,
                "reason": "price_testing_support_or_vwap",
            }
        elif dist <= 2.0:
            pullback = {
                "status": "holding_above_level",
                "level_name": support_name,
                "level": support,
                "distance_pct": dist,
                "reason": "price_holding_near_support_or_vwap",
            }
        else:
            pullback = {
                "status": "extended_from_level",
                "level_name": support_name,
                "level": support,
                "distance_pct": dist,
                "reason": "price_extended_above_support_or_vwap",
            }

    if vwap is not None and support_name != "vwap":
        vwap_dist = _distance_pct(current, vwap)
        pullback["vwap"] = {"level": vwap, "distance_pct": vwap_dist}

    if breakout.get("status") == "confirmed":
        entry_trigger = "breakout_confirmed"
        status = "breakout_confirmed"
    elif breakout.get("status") == "approaching":
        entry_trigger = "breakout_approaching"
        status = "waiting_for_breakout"
    elif pullback.get("status") in {"at_level", "holding_above_level"}:
        entry_trigger = "pullback_validating"
        status = "pullback_validating"
    elif pullback.get("status") == "failed":
        entry_trigger = "pullback_failed"
        status = "trigger_failed"
    else:
        entry_trigger = "no_trigger"
        status = "no_trigger"

    return {
        "status": status,
        "entry_trigger": entry_trigger,
        "current_price": current,
        "breakout": breakout,
        "pullback": pullback,
    }
