"""Deterministic focus-list selection for constrained LLM cycles."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any

from db.schema import AgentMemory, get_session
from utils.finnhub_client import FinnhubClient
from utils.symbol_class import classify_symbol

logger = logging.getLogger(__name__)


def _dedupe(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in symbols:
        sym = str(item or "").strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            result.append(sym)
    return result


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _score_quote(symbol: str, quote: dict[str, Any], source_bonus: float) -> tuple[float, list[str]]:
    score = float(source_bonus)
    reasons: list[str] = []
    if source_bonus:
        reasons.append(f"source_bonus={source_bonus:g}")

    symbol_class = classify_symbol(symbol)
    if symbol_class == "stock":
        score += 0.75
        reasons.append("stock_symbol")

    change_pct = _safe_float(quote.get("change_pct"))
    price = _safe_float(quote.get("price"))
    high = _safe_float(quote.get("high"))
    low = _safe_float(quote.get("low"))
    prev_close = _safe_float(quote.get("prev_close"))

    if change_pct is not None:
        abs_change = abs(change_pct)
        score += min(abs_change * 1.4, 8.0)
        reasons.append(f"abs_change={abs_change:.2f}%")
        if 0.6 <= abs_change <= 6.0:
            score += 1.0
            reasons.append("tradable_move_size")
        elif abs_change > 8.0:
            score -= 1.5
            reasons.append("overextended_move")

    if prev_close and high is not None and low is not None and high > low:
        range_pct = ((high - low) / prev_close) * 100
        score += min(range_pct, 4.0)
        reasons.append(f"day_range={range_pct:.2f}%")

        if price is not None:
            range_pos = (price - low) / (high - low)
            if change_pct is not None and change_pct > 0 and range_pos >= 0.75:
                score += 1.0
                reasons.append("near_high_with_positive_tape")
            elif change_pct is not None and change_pct < 0 and range_pos <= 0.25:
                score += 1.0
                reasons.append("near_low_with_negative_tape")

    return round(score, 4), reasons


def select_focus_symbols(
    engine,
    candidate_symbols: list[str],
    *,
    max_symbols: int = 3,
    required_symbols: list[str] | None = None,
    source_bonuses: dict[str, float] | None = None,
    context: str = "cycle",
    fh: FinnhubClient | None = None,
) -> list[str]:
    """Rank candidate symbols and return a compact focus list.

    Required symbols are always included first, even if that makes the returned
    list longer than max_symbols. This protects open-position maintenance and
    alert-specific cycles while still constraining new analyst/PM attention.
    """
    max_symbols = max(1, int(max_symbols))
    candidates = _dedupe(candidate_symbols)
    required = _dedupe(required_symbols or [])
    source_bonuses = {str(k).upper(): float(v) for k, v in (source_bonuses or {}).items()}

    client = fh or FinnhubClient()
    ranked: list[dict[str, Any]] = []
    required_set = set(required)

    for idx, sym in enumerate(candidates):
        try:
            quote = client.get_quote(sym)
            score, reasons = _score_quote(sym, quote, source_bonuses.get(sym, 0.0))
        except Exception as exc:
            quote = {"symbol": sym, "error": str(exc)[:200]}
            score = source_bonuses.get(sym, 0.0) - 5.0
            reasons = [f"quote_error={type(exc).__name__}"]

        # Stable low-order preference for earlier candidates without letting
        # broad core-watchlist ordering dominate the actual tape.
        score -= idx * 0.001
        ranked.append({
            "symbol": sym,
            "score": round(score, 4),
            "required": sym in required_set,
            "source_bonus": source_bonuses.get(sym, 0.0),
            "reasons": reasons,
            "quote": quote,
            "input_rank": idx + 1,
        })

    ranked.sort(key=lambda row: (-row["score"], row["input_rank"], row["symbol"]))
    required_ranked = [row["symbol"] for row in ranked if row["required"]]
    optional_ranked = [row["symbol"] for row in ranked if not row["required"]]
    selected = _dedupe(required_ranked + optional_ranked[:max_symbols])

    _persist_focus_selection(engine, context, max_symbols, selected, ranked)
    logger.info(
        "FOCUS_LIST_SELECTED: context=%s max=%s selected=%s candidates=%s",
        context,
        max_symbols,
        selected,
        len(candidates),
    )
    return selected


def _persist_focus_selection(
    engine,
    context: str,
    max_symbols: int,
    selected: list[str],
    ranked: list[dict[str, Any]],
) -> None:
    try:
        db = get_session(engine)
        today = datetime.now(timezone.utc).date().isoformat()
        db.add(AgentMemory(
            agent="scout",
            symbol=None,
            key=f"focus_list:{today}:{context}",
            value=json.dumps({
                "date": today,
                "context": context,
                "max_symbols": max_symbols,
                "selected": selected,
                "ranked": ranked,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }, default=str),
        ))
        db.commit()
    except Exception as exc:
        logger.warning("Focus-list persistence failed: %s", exc)
    finally:
        try:
            db.close()
        except Exception:
            pass
