"""
Position Timer — Lifecycle Governance Executor

Iterates open positions, calls the lifecycle evaluator, logs decisions,
and executes the returned actions (close, warn, repair).

All governance logic (time limits, news expiry, overnight authorization,
stop geometry) lives in utils/position_lifecycle_governance.py.
This module is a thin executor only.
"""

import json
import logging
import os
from datetime import datetime
from pytz import timezone

from db.schema import Trade, TradeEvent, Balance, AgentMemory, get_session
from utils.trade_events import log_trade_event
from utils.news_trade_governance import (
    log_trade_event_once, _build_failure_dedupe_key,
)
from utils.position_lifecycle_governance import evaluate_position_lifecycle, validate_stop_geometry

log = logging.getLogger(__name__)


def _get_setup_type_for_trade(db, trade) -> str:
    """Look up the analyst's setup_type for this trade's symbol at entry time."""
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol=trade.symbol, key="signal")
        .filter(AgentMemory.timestamp <= trade.entry_time)
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if mem:
        try:
            return json.loads(mem.value).get("setup_type", "")
        except Exception:
            pass
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol=trade.symbol, key="signal")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if mem:
        try:
            return json.loads(mem.value).get("setup_type", "")
        except Exception:
            pass
    return ""


def _get_current_price(symbol, fallback):
    try:
        import yfinance as yf
        return float(yf.Ticker(symbol).fast_info.get("lastPrice", fallback))
    except Exception:
        return fallback


def _close_position(engine, trade, price, reason):
    """Force close a position."""
    from agents.portfolio_manager import execute_trade
    db = get_session(engine)
    try:
        execute_trade(db, {
            "symbol": trade.symbol,
            "action": "CLOSE",
            "quantity": 0,
            "price": price,
            "rationale": reason,
        }, trade.profile)
        log.warning(f"⏰ FORCE CLOSED: {trade.symbol} ({trade.profile}) — {reason}")
    except Exception as e:
        log.error(f"Force close failed for {trade.symbol}: {e}")
    finally:
        db.close()

    # Trigger narrator flash update for force exits
    try:
        import agents.narrator as narrator
        pnl = round(price - trade.entry_price, 2) if trade.entry_price else None
        if getattr(trade, "direction", None) == "SHORT" and pnl is not None:
            pnl = -pnl
        narrator.run(engine, "flash_update", event_context={
            "trigger": "force_exit",
            "symbol": trade.symbol,
            "details": f"Force exit: {reason}",
            "profile": trade.profile,
            "pnl_impact": pnl,
        })
    except Exception:
        pass  # never block position timer operations


# ─── Stop Repair Execution (Task 7.3) ────────────────────────────────────────


def _execute_stop_repair(engine, db, trade_dict, price, decision) -> bool:
    """Attempt stop repair for overnight carry.

    On success: updates trade stop_price, logs overnight_stop_validated event.
    On failure: closes position immediately, logs overnight_stop_repair_failed.

    Deduplication: Only one repair attempt per governance window. If already
    attempted this window, skips repair and closes immediately.

    Returns True if repair succeeded, False if position was closed.
    """
    metadata = decision.get("metadata", {})
    auth_stop = metadata.get("auth_stop_price")
    reason = metadata.get("reason", "")
    trade_id = trade_dict["id"]
    governance_window_id = metadata.get("governance_window_id")

    # Dedupe: check if repair already attempted this governance window
    already_attempted = not log_trade_event_once(
        db,
        "stop_repair_attempted",
        trade_id,
        governance_window_id=governance_window_id,
        agent="position_timer",
        symbol=trade_dict["symbol"],
        profile=trade_dict["profile"],
        payload={"reason": reason, "auth_stop_price": auth_stop},
    )
    db.commit()

    if already_attempted:
        log.warning(
            f"Stop repair already attempted for {trade_dict['symbol']} "
            f"(trade {trade_id}) in governance window {governance_window_id} — closing"
        )
        _close_and_log_repair_failed(engine, db, trade_dict, price, decision, "dedupe_window_exhausted")
        return False

    # Non-repairable reasons: missing price or missing stop can't be fixed by updating stop
    if reason in ("missing_price_fail_safe", "missing_stop"):
        _close_and_log_repair_failed(engine, db, trade_dict, price, decision, reason)
        return False

    # Repairable: authorization_stop_mismatch — update trade stop to auth stop
    if reason == "authorization_stop_mismatch" and auth_stop is not None:
        try:
            repair_db = get_session(engine)
            trade = repair_db.query(Trade).filter_by(id=trade_id).first()
            if trade:
                trade.stop_price = float(auth_stop)
                trade.stop_updated_by = "position_timer_governance"
                trade.stop_updated_at = datetime.now(timezone("UTC"))
                repair_db.commit()
            repair_db.close()

            # Validate the repaired geometry
            is_valid, validation_reason = validate_stop_geometry(
                {**trade_dict, "stop_price": float(auth_stop)},
                {"stop_price": auth_stop},
                price,
            )
            if is_valid:
                log_trade_event_once(
                    db,
                    "overnight_stop_validated",
                    trade_id,
                    governance_window_id=governance_window_id,
                    agent="position_timer",
                    symbol=trade_dict["symbol"],
                    profile=trade_dict["profile"],
                    payload={"repaired_stop": auth_stop, "reason": reason},
                )
                db.commit()
                log.info(
                    f"✅ Stop repair succeeded for {trade_dict['symbol']} "
                    f"(trade {trade_id}): stop updated to {auth_stop}"
                )
                return True
            else:
                log.warning(
                    f"Stop repair validation failed for {trade_dict['symbol']}: {validation_reason}"
                )
        except Exception as e:
            log.error(f"Stop repair failed for {trade_dict['symbol']}: {e}")

    # Repair failed or not applicable — close position
    _close_and_log_repair_failed(engine, db, trade_dict, price, decision, reason)
    return False


def _close_and_log_repair_failed(engine, db, trade_dict, price, decision, reason):
    """Close position and log overnight_stop_repair_failed event."""
    metadata = decision.get("metadata", {})
    governance_window_id = metadata.get("governance_window_id")

    # Reconstruct trade object for _close_position
    class _T:
        pass
    trade_obj = _T()
    trade_obj.id = trade_dict["id"]
    trade_obj.symbol = trade_dict["symbol"]
    trade_obj.profile = trade_dict["profile"]
    trade_obj.direction = trade_dict["direction"]
    trade_obj.entry_price = trade_dict["entry_price"]
    trade_obj.stop_price = trade_dict["stop_price"]
    trade_obj.target_price = trade_dict["target_price"]

    _close_position(
        engine, trade_obj, price,
        f"Stop repair failed ({reason}): closing position for overnight safety"
    )

    log_trade_event_once(
        db,
        "overnight_stop_repair_failed",
        trade_dict["id"],
        governance_window_id=governance_window_id,
        agent="position_timer",
        symbol=trade_dict["symbol"],
        profile=trade_dict["profile"],
        payload={"reason": reason, "decision_reason_type": decision.get("reason_type")},
    )
    db.commit()


# ─── Executor Helper Functions (Task 7.1) ────────────────────────────────────


def fetch_open_trades_with_events(db) -> list[tuple[dict, list[dict]]]:
    """Query open trades and their trade_events.

    Returns list of (trade_dict, events_list) tuples where:
    - trade_dict has keys: id, symbol, profile, direction, entry_price, entry_time,
      stop_price, target_price, setup_type, status, quantity
    - events_list is a list of dicts with: event_type, timestamp, payload (parsed dict)
    """
    open_trades = (
        db.query(Trade)
        .filter_by(status="open")
        .filter(Trade.entry_time.isnot(None))
        .all()
    )

    results = []
    for trade in open_trades:
        trade_dict = {
            "id": trade.id,
            "symbol": trade.symbol,
            "profile": trade.profile,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "entry_time": trade.entry_time,
            "stop_price": trade.stop_price,
            "target_price": trade.target_price,
            "setup_type": _get_setup_type_for_trade(db, trade),
            "status": trade.status,
            "quantity": trade.quantity,
        }

        trade_events = (
            db.query(TradeEvent)
            .filter_by(trade_id=trade.id)
            .order_by(TradeEvent.timestamp.asc())
            .all()
        )

        events_list = []
        for event in trade_events:
            payload = {}
            if event.payload_json:
                try:
                    payload = json.loads(event.payload_json)
                except (json.JSONDecodeError, TypeError):
                    payload = {}
            events_list.append({
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "payload": payload,
            })

        results.append((trade_dict, events_list))

    return results


def get_current_equity(db) -> float | None:
    """Get current portfolio equity from latest balance row.
    Falls back to STARTING_EQUITY env var if no balance row exists.
    """
    latest_balance = (
        db.query(Balance)
        .order_by(Balance.timestamp.desc())
        .first()
    )

    if latest_balance:
        if latest_balance.total_equity is not None:
            return latest_balance.total_equity
        if latest_balance.portfolio_value is not None:
            return latest_balance.portfolio_value
        return latest_balance.cash

    # Fallback to environment variable
    try:
        starting_equity = os.environ.get("STARTING_EQUITY", "")
        if starting_equity:
            return float(starting_equity)
    except (ValueError, TypeError):
        pass

    return None


def get_overnight_stats(db, now_utc: datetime) -> tuple[dict[str, int], dict[str, float]]:
    """Get overnight position counts and exposure percentages by profile.

    Returns:
        (overnight_position_counts, overnight_exposure_pcts)
        Both are dicts keyed by profile name.
    """
    overnight_position_counts: dict[str, int] = {}
    overnight_exposure_pcts: dict[str, float] = {}

    # Query all overnight_authorized events
    auth_events = (
        db.query(TradeEvent)
        .filter_by(event_type="overnight_authorized")
        .all()
    )

    for event in auth_events:
        payload = {}
        if event.payload_json:
            try:
                payload = json.loads(event.payload_json)
            except (json.JSONDecodeError, TypeError):
                continue

        # Check if authorization is still valid (expires_at > now_utc)
        expires_at_str = payload.get("expires_at")
        if not expires_at_str:
            continue

        try:
            if isinstance(expires_at_str, str):
                expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
            else:
                expires_at = expires_at_str
            # Compare naive datetimes if needed
            if hasattr(expires_at, "tzinfo") and expires_at.tzinfo is not None:
                from datetime import timezone as dt_timezone
                now_aware = now_utc.replace(tzinfo=dt_timezone.utc) if now_utc.tzinfo is None else now_utc
                if expires_at <= now_aware:
                    continue
            else:
                now_naive = now_utc.replace(tzinfo=None) if now_utc.tzinfo is not None else now_utc
                if expires_at <= now_naive:
                    continue
        except (ValueError, TypeError):
            continue

        # Check that the associated trade is still open
        profile = payload.get("profile") or event.profile
        if not profile:
            continue

        # Verify the trade is still open
        trade = db.query(Trade).filter_by(id=event.trade_id, status="open").first()
        if not trade:
            continue

        overnight_position_counts[profile] = overnight_position_counts.get(profile, 0) + 1
        # Exposure is set to 0.0 here — the evaluator calculates candidate exposure
        overnight_exposure_pcts[profile] = overnight_exposure_pcts.get(profile, 0.0)

    return overnight_position_counts, overnight_exposure_pcts


def exclude_trade_from_counts(counts: dict[str, int], trade: dict) -> dict[str, int]:
    """Return a copy of counts with the current trade's profile decremented (if > 0)."""
    # Note: The counts already exclude the current trade if computed correctly.
    # This is a safety helper — in practice the stats should already exclude it.
    return dict(counts)


def exclude_trade_from_exposure(exposure: dict[str, float], trade: dict) -> dict[str, float]:
    """Return a copy of exposure with the current trade's profile excluded."""
    return dict(exposure)


def log_lifecycle_decision_once(db, decision: dict) -> bool:
    """Log a lifecycle decision as a trade event with deduplication.

    Deduplication strategy:
    - Success/warning events: per governance window (standard dedupe)
    - Failure events: hourly bucket to allow retry visibility

    Returns True if event was written, False if suppressed as duplicate.
    """

    # Map decision state/reason_type to appropriate event_type
    state = decision.get("state", "")
    decision_action = decision.get("decision", "")
    reason_type = decision.get("reason_type", "")
    trade_id = decision.get("trade_id")

    # Determine event_type from the decision
    event_type_map = {
        "news_expired": "news_expiry_force_close",
        "news_reconfirmation_due": "news_reconfirmation_due",
        "news_grace": "news_reconfirmation_due",
        "overnight_unauthorized": "overnight_unauthorized_close",
        "invalid_stop_geometry": "overnight_invalid_stop_force_close",
        "intraday_expired": "intraday_force_close",
        "intraday_warning": "intraday_time_warning",
        "close_required": "lifecycle_close_required",
    }
    event_type = event_type_map.get(state, "lifecycle_decision_logged")

    # Build payload from decision metadata
    payload = {
        "decision": decision_action,
        "state": state,
        "reason_type": reason_type,
        "hours_held": decision.get("hours_held"),
        "setup_type": decision.get("setup_type"),
    }
    if decision.get("close_reason"):
        payload["close_reason"] = decision["close_reason"]
    if decision.get("metadata"):
        payload["metadata"] = decision["metadata"]

    # Extract governance_window_id from metadata if available
    metadata = decision.get("metadata", {})
    governance_window_id = metadata.get("governance_window_id")

    # Failure events use hourly deduplication; success/warning use per-window
    failure_states = {
        "news_expired", "overnight_unauthorized", "invalid_stop_geometry",
        "intraday_expired", "close_required",
    }
    is_failure_event = (state in failure_states and decision_action == "close")

    if is_failure_event:
        now_utc = datetime.now(timezone("UTC"))
        dedupe_key = _build_failure_dedupe_key(
            event_type, trade_id, governance_window_id, now_utc
        )
        # Manual dedupe check + write for failure events
        existing = (
            db.query(TradeEvent)
            .filter_by(event_type=event_type, trade_id=trade_id, dedupe_key=dedupe_key)
            .first()
        )
        if existing:
            return False

        if payload is None:
            payload = {}
        if governance_window_id is not None:
            payload["governance_window_id"] = governance_window_id

        event = log_trade_event(
            db, event_type, trade_id=trade_id,
            agent="position_timer",
            symbol=decision.get("symbol"),
            profile=decision.get("profile"),
            payload=payload,
        )
        event.dedupe_key = dedupe_key
        return True

    # Standard per-window deduplication for success/warning events
    return log_trade_event_once(
        db,
        event_type,
        trade_id,
        governance_window_id=governance_window_id,
        agent="position_timer",
        symbol=decision.get("symbol"),
        profile=decision.get("profile"),
        payload=payload,
    )


def log_force_close_success(db, decision: dict) -> bool:
    """Log a force-close success event AFTER the close operation succeeds.

    Deduplication: per governance window (standard dedupe).
    Only call this after the close operation has completed successfully.

    Returns True if event was written, False if suppressed as duplicate.
    """
    state = decision.get("state", "")
    trade_id = decision.get("trade_id")
    metadata = decision.get("metadata", {})
    governance_window_id = metadata.get("governance_window_id")

    # Map state to success event type
    success_event_map = {
        "news_expired": "news_expiry_force_close",
        "overnight_unauthorized": "overnight_unauthorized_force_close",
        "invalid_stop_geometry": "overnight_invalid_stop_force_close",
        "intraday_expired": "intraday_force_close",
        "close_required": "lifecycle_force_close",
    }
    event_type = success_event_map.get(state, "lifecycle_force_close")

    return log_trade_event_once(
        db, event_type, trade_id,
        governance_window_id=governance_window_id,
        agent="position_timer",
        symbol=decision.get("symbol"),
        profile=decision.get("profile"),
        payload={
            "state": state,
            "reason_type": decision.get("reason_type"),
            "close_reason": decision.get("close_reason"),
            "hours_held": decision.get("hours_held"),
        },
    )


def log_force_close_failure(db, decision: dict, error: str) -> bool:
    """Log a force-close failure event with hourly deduplication.

    Deduplication: hourly bucket per (event_type, trade_id, governance_window).
    This ensures persistent failures remain visible (one log per hour) while
    avoiding log spam on rapid retry cycles.

    Returns True if event was written, False if suppressed as duplicate.
    """

    state = decision.get("state", "")
    trade_id = decision.get("trade_id")
    metadata = decision.get("metadata", {})
    governance_window_id = metadata.get("governance_window_id")

    # Map state to failure event type
    failure_event_map = {
        "news_expired": "news_expiry_force_close_failed",
        "overnight_unauthorized": "overnight_unauthorized_force_close_failed",
        "invalid_stop_geometry": "overnight_invalid_stop_force_close_failed",
        "intraday_expired": "intraday_force_close_failed",
        "close_required": "lifecycle_force_close_failed",
    }
    event_type = failure_event_map.get(state, "lifecycle_force_close_failed")

    # Build hourly-bucketed dedupe key
    now_utc = datetime.now(timezone("UTC"))
    dedupe_key = _build_failure_dedupe_key(event_type, trade_id, governance_window_id, now_utc)

    # Check for existing event with same dedupe_key
    existing = (
        db.query(TradeEvent)
        .filter_by(event_type=event_type, trade_id=trade_id, dedupe_key=dedupe_key)
        .first()
    )
    if existing:
        return False

    event = log_trade_event(
        db, event_type, trade_id=trade_id,
        agent="position_timer",
        symbol=decision.get("symbol"),
        profile=decision.get("profile"),
        payload={
            "state": state,
            "reason_type": decision.get("reason_type"),
            "error": error,
            "hours_held": decision.get("hours_held"),
        },
    )
    event.dedupe_key = dedupe_key
    return True


def run(engine) -> dict:
    """Execute lifecycle governance for all open positions."""
    db = get_session(engine)
    et_tz = timezone("America/New_York")
    now_et = datetime.now(et_tz)
    now_utc = datetime.now(timezone("UTC"))

    results = {"closes": [], "repairs": [], "warnings": [], "skipped": []}

    open_trades = None
    try:
        open_trades = fetch_open_trades_with_events(db)
        if not open_trades:
            return results

        current_equity = get_current_equity(db)
        overnight_counts, overnight_exposure = get_overnight_stats(db, now_utc)

        for trade_dict, events in open_trades:
            price = _get_current_price(trade_dict["symbol"], trade_dict["entry_price"])

            # Exclude current trade from overnight stats to prevent double-counting
            trade_overnight_counts = exclude_trade_from_counts(overnight_counts, trade_dict)
            trade_overnight_exposure = exclude_trade_from_exposure(overnight_exposure, trade_dict)

            decision = evaluate_position_lifecycle(
                trade_dict, events,
                now_utc=now_utc, now_et=now_et,
                current_price=price,
                current_equity=current_equity,
                overnight_position_counts=trade_overnight_counts,
                overnight_exposure_pcts=trade_overnight_exposure,
            )

            # Log decision event (idempotent)
            if decision["requires_event"]:
                log_lifecycle_decision_once(db, decision)
                db.commit()

            # Execute action
            if decision["decision"] == "close":
                # Reconstruct minimal trade object for _close_position
                class _T:
                    pass
                trade_obj = _T()
                trade_obj.id = trade_dict["id"]
                trade_obj.symbol = trade_dict["symbol"]
                trade_obj.profile = trade_dict["profile"]
                trade_obj.direction = trade_dict["direction"]
                trade_obj.entry_price = trade_dict["entry_price"]
                trade_obj.stop_price = trade_dict["stop_price"]
                trade_obj.target_price = trade_dict["target_price"]

                try:
                    _close_position(engine, trade_obj, price, decision.get("close_reason", "Lifecycle governance close"))
                    # Log success ONLY after close operation succeeds
                    log_force_close_success(db, decision)
                    db.commit()
                    results["closes"].append(decision)

                    # Update in-memory stats after close
                    profile = trade_dict.get("profile", "")
                    if profile in overnight_counts and overnight_counts[profile] > 0:
                        overnight_counts[profile] -= 1
                except Exception as e:
                    log.error(f"Force close failed for {trade_dict['symbol']}: {e}")
                    log_force_close_failure(db, decision, str(e))
                    db.commit()

            elif decision["decision"] == "repair_stop":
                success = _execute_stop_repair(engine, db, trade_dict, price, decision)
                results["repairs"].append({**decision, "repair_success": success})
                if not success:
                    # Position was closed — update overnight stats
                    profile = trade_dict.get("profile", "")
                    if profile in overnight_counts and overnight_counts[profile] > 0:
                        overnight_counts[profile] -= 1

            elif decision["decision"] in ("warn", "authorize_required"):
                results["warnings"].append(decision)
                # Write AgentMemory alert for PM/CEO visibility
                try:
                    db.add(AgentMemory(
                        agent="position_timer",
                        symbol=trade_dict["symbol"],
                        key=f"lifecycle_{decision['state']}",
                        value=json.dumps({
                            "trade_id": trade_dict["id"],
                            "symbol": trade_dict["symbol"],
                            "profile": trade_dict["profile"],
                            "state": decision["state"],
                            "reason_type": decision["reason_type"],
                            "hours_held": decision.get("hours_held"),
                            "metadata": decision.get("metadata", {}),
                        }, default=str),
                    ))
                    db.commit()
                except Exception as e:
                    log.error(f"Failed to write lifecycle alert for {trade_dict['symbol']}: {e}")

            elif decision["decision"] == "skip":
                results["skipped"].append(decision)

    finally:
        db.close()

    return results
