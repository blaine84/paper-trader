"""Outcome tracking for blocked trade candidates.

This module turns the shadow ledger from "we captured the reject" into
"was the reject right in hindsight?" by periodically scoring blocked
candidates against later market prices.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text

from utils.gate_config import SHADOW_SCORE_MAX_ENTRY_DEVIATION_PCT

log = logging.getLogger(__name__)

OUTCOME_WINDOWS: tuple[tuple[str, int], ...] = (
    ("15m", 15),
    ("30m", 30),
    ("60m", 60),
)
MARKET_TZ = ZoneInfo("America/New_York")

KNOWN_GATE_NAMES: set[str] = {
    "setup_quality_gate",
    "risk_geometry_gate",
    "catalyst_specificity_gate",
    "entry_timing_gate",
    "position_risk_gate",
    "dollar_risk_gate",
    "track_record_gate",
    "alert_cooldown",
}


def classify_candidate_source(blocked_by: str | None) -> str:
    """Classify whether a blocked candidate is a malformed_decision or gate_rejection.

    Returns: 'malformed_decision' | 'gate_rejection' | 'unknown_source'
    """
    if blocked_by == "pm_normalizer":
        return "malformed_decision"
    if blocked_by in KNOWN_GATE_NAMES:
        return "gate_rejection"
    return "unknown_source"


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            try:
                dt = datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
    else:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if out > 0 else None
    except (TypeError, ValueError):
        return None


def _is_positive_number(value: Any) -> bool:
    """Check if value is a positive finite number."""
    if value is None:
        return False
    try:
        v = float(value)
        return v > 0 and math.isfinite(v)
    except (TypeError, ValueError):
        return False


# Patterns that indicate a placeholder symbol (sector/category/theme concepts)
_INVALID_SYMBOL_PATTERNS: set[str] = {
    "sector_",
    "industry_",
    "category_",
    "theme_",
    "ETF_",
}


def validate_candidate_scorability(
    candidate: dict[str, Any],
    engine: Any | None = None,
) -> tuple[bool, str | None]:
    """Verify a blocked candidate is scorable before requesting market data.

    Pre-flight checks (Requirements 10.1–10.6):
    1. Candidate has a valid candidate_id that exists in pm_candidates table
       (verifies it originated from a registered deterministic candidate).
       If engine is None, skip this check (backward compatibility).
    2. Candidate has a valid supported symbol (non-empty string, not a
       sector/category placeholder).
    3. Candidate has complete numeric entry, stop, and target fields.
    4. Candidate has a recognized direction (BUY or SHORT).
    5. Candidate has a deterministic execution quantity OR we can derive
       a shadow quantity (non-zero numeric quantity field present).

    Args:
        candidate: Dict with candidate data (from blocked_trade_candidates table).
        engine: SQLAlchemy engine for pm_candidates lookup (optional).

    Returns:
        (is_scorable, reason) — reason is None if scorable.
    """
    # Check 1: Candidate registry origin (Requirement 10.1)
    if engine is not None:
        candidate_id = candidate.get("geometry_candidate_id") or candidate.get("candidate_id")
        if candidate_id:
            from sqlalchemy import text as _text

            try:
                with engine.connect() as conn:
                    row = conn.execute(
                        _text("SELECT 1 FROM pm_candidates WHERE candidate_id = :cid"),
                        {"cid": candidate_id},
                    ).fetchone()
                    if row is None:
                        return (False, "candidate_not_in_registry")
            except Exception:
                # If table doesn't exist yet (pre-migration), skip this check
                pass
        # If no candidate_id field at all, this may be a legacy blocked candidate.
        # Allow through for backward compatibility (legacy path didn't use candidate_ids).

    # Check 2: Valid supported symbol (Requirement 10.2, 10.6)
    symbol = candidate.get("symbol")
    if not symbol or not isinstance(symbol, str):
        return (False, "missing_symbol")
    # Reject sector/category placeholders
    if any(symbol.startswith(p) for p in _INVALID_SYMBOL_PATTERNS):
        return (False, "symbol_is_placeholder")

    # Check 3: Complete numeric geometry (Requirement 10.3)
    entry = candidate.get("entry_price")
    stop = candidate.get("stop_price")
    target = candidate.get("target_price")
    if not _is_positive_number(entry) or not _is_positive_number(stop) or not _is_positive_number(target):
        return (False, "incomplete_geometry")

    # Check 4: Recognized direction (Requirement 10.4)
    action = candidate.get("action") or candidate.get("direction") or ""
    if str(action).upper() not in ("BUY", "SHORT", "LONG"):
        return (False, "unrecognized_direction")

    # Check 5: Has quantity (deterministic or shadow) (Requirement 10.3)
    quantity = candidate.get("quantity")
    if quantity is None or (isinstance(quantity, (int, float)) and quantity <= 0):
        return (False, "missing_quantity")

    return (True, None)


def validate_scorability(
    candidate: dict[str, Any],
    first_candle_close: float | None,
    max_deviation_pct: float = SHADOW_SCORE_MAX_ENTRY_DEVIATION_PCT,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Check if a candidate can be meaningfully scored.

    Returns:
        (is_scorable, reason, extra_notes)
        - is_scorable: True if scoring should proceed
        - reason: None if scorable, else one of: missing_entry_price,
          missing_stop_price, missing_target_price, missing_quantity,
          missing_symbol, entry_price_deviation_exceeded, no_reference_price
        - extra_notes: Additional context (e.g., deviation_pct) or None
    """
    # Check required numeric fields for null/zero
    entry_price = _num(candidate.get("entry_price"))
    if entry_price is None:
        return (False, "missing_entry_price", None)

    stop_price = _num(candidate.get("stop_price"))
    if stop_price is None:
        return (False, "missing_stop_price", None)

    target_price = _num(candidate.get("target_price"))
    if target_price is None:
        return (False, "missing_target_price", None)

    quantity = _num(candidate.get("quantity"))
    if quantity is None:
        return (False, "missing_quantity", None)

    # Check symbol for null
    symbol = candidate.get("symbol")
    if not symbol:
        return (False, "missing_symbol", None)

    # Check reference price availability
    if first_candle_close is None or first_candle_close == 0:
        return (False, "no_reference_price", None)

    # Compute entry price deviation
    deviation_pct = abs(entry_price - first_candle_close) / first_candle_close
    if deviation_pct > max_deviation_pct:
        return (False, "entry_price_deviation_exceeded", {"deviation_pct": deviation_pct})

    return (True, None, None)


def _direction(row: dict[str, Any]) -> str | None:
    direction = (row.get("direction") or "").lower()
    action = (row.get("action") or "").upper()
    if direction in {"long", "short"}:
        return direction
    if action == "BUY":
        return "long"
    if action == "SHORT":
        return "short"
    return None


def _overlaps_regular_session(start: datetime, end: datetime) -> bool:
    """Return whether an interval includes any regular NYSE session time."""
    start_et = start.astimezone(MARKET_TZ)
    end_et = end.astimezone(MARKET_TZ)
    day = start_et.date()
    while day <= end_et.date():
        if day.weekday() < 5:
            market_open = datetime.combine(day, datetime.min.time(), tzinfo=MARKET_TZ).replace(
                hour=9, minute=30
            )
            market_close = market_open.replace(hour=16, minute=0)
            if start_et <= market_close and end_et >= market_open:
                return True
        day += timedelta(days=1)
    return False


def _unscorable_outcome(candidate: dict[str, Any], window_label: str, window_minutes: int) -> dict[str, Any]:
    """Build a terminal outcome record for a window with no market candles by design."""
    created_at = _parse_dt(candidate.get("created_at"))
    evaluated_at = created_at + timedelta(minutes=window_minutes)
    return {
        "blocked_candidate_id": candidate.get("id"),
        "eval_window": window_label,
        "evaluated_at": evaluated_at.replace(tzinfo=None),
        "eval_price": None,
        "pnl_pct": None,
        "mfe_pct": None,
        "mae_pct": None,
        "stop_hit": 0,
        "target_hit": 0,
        "first_hit": None,
        "first_hit_at": None,
        "outcome_label": "unscorable_no_regular_session",
        "gate_verdict": "unscorable",
        "notes_json": json.dumps({
            "reason": "evaluation window does not overlap regular trading session",
            "candles_scored": 0,
        }),
    }


def _as_candles(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candles: list[dict[str, Any]] = []
    for r in records:
        ts = _parse_dt(r.get("timestamp") or r.get("time") or r.get("datetime"))
        close = _num(r.get("close"))
        if ts is None or close is None:
            continue
        high = _num(r.get("high")) or close
        low = _num(r.get("low")) or close
        candles.append({"timestamp": ts, "high": high, "low": low, "close": close})
    return sorted(candles, key=lambda c: c["timestamp"])


def _fetch_yfinance_candles(symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Fetch one-minute candles as normalized UTC records."""
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(
            start=start.astimezone(timezone.utc),
            end=end.astimezone(timezone.utc) + timedelta(minutes=2),
            interval="1m",
            auto_adjust=False,
            prepost=False,
        )
        if hist is None or hist.empty:
            return []

        records: list[dict[str, Any]] = []
        for idx, row in hist.iterrows():
            ts = idx.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            records.append({
                "timestamp": ts,
                "high": float(row.get("High")),
                "low": float(row.get("Low")),
                "close": float(row.get("Close")),
            })
        return _as_candles(records)
    except Exception as exc:
        log.warning("Shadow outcome: yfinance candles failed for %s: %s", symbol, exc)
        return []


def score_blocked_candidate(
    candidate: dict[str, Any],
    *,
    window_label: str,
    window_minutes: int,
    candles: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Score one blocked candidate for one elapsed outcome window.

    Returns a dict ready for insertion into blocked_trade_candidate_outcomes,
    or None when the candidate cannot be scored yet.
    """
    created_at = _parse_dt(candidate.get("created_at"))
    if created_at is None:
        return None

    eval_at = created_at + timedelta(minutes=window_minutes)
    window_candles = [c for c in _as_candles(candles) if created_at <= c["timestamp"] <= eval_at]

    # Determine first candle close for scorability validation
    first_candle_close: float | None = window_candles[0]["close"] if window_candles else None

    # Validate scorability before any P&L calculation
    is_scorable, reason, extra_notes = validate_scorability(candidate, first_candle_close)

    # Classify candidate source for all outcomes
    candidate_source_classification = classify_candidate_source(candidate.get("blocked_by"))

    if not is_scorable:
        # Build notes_json for unscorable outcome
        notes: dict[str, Any] = {
            "reason": reason,
            "candidate_source_classification": candidate_source_classification,
        }
        # Include deviation_pct if present in extra_notes
        if extra_notes and "deviation_pct" in extra_notes:
            notes["deviation_pct"] = extra_notes["deviation_pct"]
            notes["entry_price"] = _num(candidate.get("entry_price"))
            notes["first_candle_close"] = first_candle_close

        return {
            "blocked_candidate_id": candidate.get("id"),
            "eval_window": window_label,
            "evaluated_at": eval_at.replace(tzinfo=None),
            "eval_price": None,
            "pnl_pct": None,
            "mfe_pct": None,
            "mae_pct": None,
            "stop_hit": 0,
            "target_hit": 0,
            "first_hit": None,
            "first_hit_at": None,
            "outcome_label": f"unscorable_{reason}",
            "gate_verdict": "unscorable",
            "notes_json": json.dumps(notes),
        }

    # Scorable path — proceed with P&L calculation
    entry = _num(candidate.get("entry_price"))
    direction = _direction(candidate)
    if entry is None or direction is None:
        return None

    if not window_candles:
        return None

    eval_price = window_candles[-1]["close"]
    highs = [c["high"] for c in window_candles]
    lows = [c["low"] for c in window_candles]
    stop = _num(candidate.get("stop_price"))
    target = _num(candidate.get("target_price"))

    if direction == "long":
        pnl_pct = (eval_price - entry) / entry * 100
        mfe_pct = (max(highs) - entry) / entry * 100
        mae_pct = (min(lows) - entry) / entry * 100
    else:
        pnl_pct = (entry - eval_price) / entry * 100
        mfe_pct = (entry - min(lows)) / entry * 100
        mae_pct = (entry - max(highs)) / entry * 100

    first_hit = None
    first_hit_at = None
    stop_hit = False
    target_hit = False
    for candle in window_candles:
        if direction == "long":
            hit_stop = stop is not None and candle["low"] <= stop
            hit_target = target is not None and candle["high"] >= target
        else:
            hit_stop = stop is not None and candle["high"] >= stop
            hit_target = target is not None and candle["low"] <= target

        stop_hit = stop_hit or hit_stop
        target_hit = target_hit or hit_target
        if first_hit is None and (hit_stop or hit_target):
            first_hit = "ambiguous" if hit_stop and hit_target else ("stop" if hit_stop else "target")
            first_hit_at = candle["timestamp"]
            if first_hit != "ambiguous":
                break

    if first_hit == "target":
        outcome_label = "would_hit_target"
        gate_verdict = "blocked_winner"
    elif first_hit in {"stop", "ambiguous"}:
        outcome_label = "would_hit_stop" if first_hit == "stop" else "ambiguous_stop_and_target"
        gate_verdict = "saved_us" if first_hit == "stop" else "ambiguous"
    elif pnl_pct > 0.15:
        outcome_label = "winner_so_far"
        gate_verdict = "possibly_overblocked"
    elif pnl_pct < -0.15:
        outcome_label = "loser_so_far"
        gate_verdict = "saved_us"
    else:
        outcome_label = "flat_so_far"
        gate_verdict = "neutral"

    return {
        "blocked_candidate_id": candidate.get("id"),
        "eval_window": window_label,
        "evaluated_at": eval_at.replace(tzinfo=None),
        "eval_price": round(eval_price, 4),
        "pnl_pct": round(pnl_pct, 4),
        "mfe_pct": round(mfe_pct, 4),
        "mae_pct": round(mae_pct, 4),
        "stop_hit": int(stop_hit),
        "target_hit": int(target_hit),
        "first_hit": first_hit,
        "first_hit_at": first_hit_at.replace(tzinfo=None) if first_hit_at else None,
        "outcome_label": outcome_label,
        "gate_verdict": gate_verdict,
        "notes_json": json.dumps({
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "direction": direction,
            "candles_scored": len(window_candles),
            "candidate_source_classification": candidate_source_classification,
        }),
    }


def get_gate_effectiveness_summary(
    engine,
    *,
    gate_name: str | None = None,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Compute gate effectiveness using only 60-minute opportunity outcomes.

    Excludes:
    - Candidates with unscorable 60m outcomes
    - Candidates with no 60m outcome row
    - Candidates classified as malformed_decision

    Returns:
        {
            "gate_name": str,
            "blocked_winners": int,
            "saved_us": int,
            "neutral": int,
            "unscorable_excluded": int,
            "malformed_excluded": int,
            "avg_pnl_pct": float,
            "period_days": int,
        }
    """
    with engine.connect() as conn:
        cutoff = f"-{lookback_days} days"

        # --- Count unscorable exclusions ---
        unscorable_params: dict[str, Any] = {"cutoff": cutoff}
        unscorable_sql = """
            SELECT COUNT(*) FROM blocked_trade_candidate_outcomes o
            JOIN blocked_trade_candidates b ON o.blocked_candidate_id = b.id
            WHERE o.eval_window = '60m'
              AND o.gate_verdict = 'unscorable'
              AND datetime(b.created_at) >= datetime('now', :cutoff)
        """
        if gate_name is not None:
            unscorable_sql += " AND b.blocked_by = :gate_name"
            unscorable_params["gate_name"] = gate_name

        unscorable_excluded = conn.execute(
            text(unscorable_sql), unscorable_params
        ).scalar() or 0

        # --- Count malformed_decision exclusions ---
        malformed_params: dict[str, Any] = {"cutoff": cutoff}
        malformed_sql = """
            SELECT COUNT(*) FROM blocked_trade_candidate_outcomes o
            JOIN blocked_trade_candidates b ON o.blocked_candidate_id = b.id
            WHERE o.eval_window = '60m'
              AND o.gate_verdict != 'unscorable'
              AND json_extract(o.notes_json, '$.candidate_source_classification') = 'malformed_decision'
              AND datetime(b.created_at) >= datetime('now', :cutoff)
        """
        if gate_name is not None:
            malformed_sql += " AND b.blocked_by = :gate_name"
            malformed_params["gate_name"] = gate_name

        malformed_excluded = conn.execute(
            text(malformed_sql), malformed_params
        ).scalar() or 0

        # --- Query valid 60m outcomes for effectiveness ---
        effectiveness_params: dict[str, Any] = {"cutoff": cutoff}
        effectiveness_sql = """
            SELECT o.outcome_label, o.gate_verdict, o.pnl_pct
            FROM blocked_trade_candidate_outcomes o
            JOIN blocked_trade_candidates b ON o.blocked_candidate_id = b.id
            WHERE o.eval_window = '60m'
              AND o.gate_verdict != 'unscorable'
              AND json_extract(o.notes_json, '$.candidate_source_classification') != 'malformed_decision'
              AND datetime(b.created_at) >= datetime('now', :cutoff)
        """
        if gate_name is not None:
            effectiveness_sql += " AND b.blocked_by = :gate_name"
            effectiveness_params["gate_name"] = gate_name

        rows = conn.execute(
            text(effectiveness_sql), effectiveness_params
        ).mappings().all()

    # Tally counts
    blocked_winners = 0
    saved_us = 0
    neutral = 0
    pnl_values: list[float] = []

    for row in rows:
        outcome_label = row["outcome_label"] or ""
        gate_verdict = row["gate_verdict"] or ""
        pnl = row["pnl_pct"]

        if gate_verdict == "blocked_winner" or outcome_label == "blocked_winner":
            blocked_winners += 1
        elif gate_verdict == "saved_us":
            saved_us += 1
        else:
            neutral += 1

        if pnl is not None:
            pnl_values.append(float(pnl))

    avg_pnl_pct = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0

    return {
        "gate_name": gate_name or "all",
        "blocked_winners": blocked_winners,
        "saved_us": saved_us,
        "neutral": neutral,
        "unscorable_excluded": unscorable_excluded,
        "malformed_excluded": malformed_excluded,
        "avg_pnl_pct": round(avg_pnl_pct, 4),
        "period_days": lookback_days,
    }


def update_blocked_candidate_outcomes(
    engine,
    *,
    now: datetime | None = None,
    max_rows: int = 50,
    candle_fetcher=_fetch_yfinance_candles,
) -> dict[str, int]:
    """Score due blocked candidates and insert companion outcome rows."""
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    inserted = 0
    skipped = 0
    candidates_seen = 0

    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT * FROM blocked_trade_candidates
                WHERE action IN ('BUY', 'SHORT')
                  AND datetime(created_at) <= datetime(:now, '-15 minutes')
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {"now": now_utc.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"), "limit": max_rows},
        ).mappings().all()

        for row in rows:
            candidate = dict(row)
            candidates_seen += 1
            created_at = _parse_dt(candidate.get("created_at"))
            if created_at is None:
                skipped += 1
                continue

            due_windows: list[tuple[str, int]] = []
            for label, minutes in OUTCOME_WINDOWS:
                if created_at + timedelta(minutes=minutes) > now_utc:
                    continue
                exists = conn.execute(
                    text(
                        """
                        SELECT 1 FROM blocked_trade_candidate_outcomes
                        WHERE blocked_candidate_id = :id AND eval_window = :window
                        LIMIT 1
                        """
                    ),
                    {"id": candidate["id"], "window": label},
                ).scalar()
                if not exists:
                    due_windows.append((label, minutes))

            if not due_windows:
                continue

            regular_session_windows = []
            for label, minutes in due_windows:
                if _overlaps_regular_session(created_at, created_at + timedelta(minutes=minutes)):
                    regular_session_windows.append((label, minutes))
                    continue
                conn.execute(
                    text(
                        """
                        INSERT OR IGNORE INTO blocked_trade_candidate_outcomes (
                            blocked_candidate_id, eval_window, evaluated_at, eval_price,
                            pnl_pct, mfe_pct, mae_pct, stop_hit, target_hit, first_hit,
                            first_hit_at, outcome_label, gate_verdict, notes_json
                        ) VALUES (
                            :blocked_candidate_id, :eval_window, :evaluated_at, :eval_price,
                            :pnl_pct, :mfe_pct, :mae_pct, :stop_hit, :target_hit, :first_hit,
                            :first_hit_at, :outcome_label, :gate_verdict, :notes_json
                        )
                        """
                    ),
                    _unscorable_outcome(candidate, label, minutes),
                )
                inserted += 1

            if not regular_session_windows:
                continue

            # Guard: skip candle fetch for symbol-null candidates — they will
            # be marked unscorable by validate_scorability() inside the scorer.
            if not candidate.get("symbol"):
                candles: list[dict[str, Any]] = []
            else:
                max_minutes = max(minutes for _, minutes in regular_session_windows)
                candles = candle_fetcher(
                    candidate["symbol"],
                    created_at - timedelta(minutes=1),
                    created_at + timedelta(minutes=max_minutes),
                )
                if not candles:
                    skipped += len(regular_session_windows)
                    continue

            for label, minutes in regular_session_windows:
                scored = score_blocked_candidate(
                    candidate,
                    window_label=label,
                    window_minutes=minutes,
                    candles=candles,
                )
                if not scored:
                    skipped += 1
                    continue
                conn.execute(
                    text(
                        """
                        INSERT OR IGNORE INTO blocked_trade_candidate_outcomes (
                            blocked_candidate_id, eval_window, evaluated_at, eval_price,
                            pnl_pct, mfe_pct, mae_pct, stop_hit, target_hit, first_hit,
                            first_hit_at, outcome_label, gate_verdict, notes_json
                        ) VALUES (
                            :blocked_candidate_id, :eval_window, :evaluated_at, :eval_price,
                            :pnl_pct, :mfe_pct, :mae_pct, :stop_hit, :target_hit, :first_hit,
                            :first_hit_at, :outcome_label, :gate_verdict, :notes_json
                        )
                        """
                    ),
                    scored,
                )
                inserted += 1

    return {"candidates_seen": candidates_seen, "inserted": inserted, "skipped": skipped}
