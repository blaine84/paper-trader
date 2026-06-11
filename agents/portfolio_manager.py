"""
Portfolio Manager Agent
Each profile (conservative/moderate/aggressive) runs with its own isolated portfolio.
Scout provides symbols only — all entry/exit decisions are made here.
"""

import collections
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from utils.finnhub_client import FinnhubClient
from utils.llm import call_llm, parse_json_response
from db.schema import AgentMemory, Position, Balance, Trade, DynamicStrategy, get_session
from utils.trade_events import log_trade_event
from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES
from utils.case_library import get_relevant_cases, get_win_rate_by_setup
from utils.prompt_compaction import format_cases_digest_for_pm, compact_signal_for_pm
from utils.entry_geometry import build_entry_geometry_scaffold
from agents.quant_researcher import build_pm_strategy_context
from agents.lesson_registry import check_track_record
from core.similarity import find_similar_cases, compute_similarity_stats
from core.edge_score import (
    compute_edge_score, check_hard_rejection, cap_position_size,
    confluence_score, similarity_quality,
)
from core.portfolio_risk import (
    validate_portfolio_risk, compute_portfolio_risk, adaptive_risk_throttle,
)
from utils.setup_quality_gate import evaluate_setup_quality, resolve_setup_type
from utils.pre_trade_quality_gate import evaluate_pre_trade_quality
from utils.catalyst_specificity import evaluate_catalyst_specificity
from utils.stop_authority import apply_stop_update
from utils.shadow_ledger import record_blocked_candidate, write_pilot_counterfactual_row

log = logging.getLogger(__name__)


# ── Entry Normalizer Dataclasses ──────────────────────────────────────────────

@dataclass
class NormalizedOrder:
    """Structured result from the entry normalizer for a single LLM decision."""
    ok: bool
    order: dict | None = None
    warnings: list[str] = field(default_factory=list)
    reason_code: str | None = None
    reason: str | None = None
    details: dict = field(default_factory=dict)
    raw_decision: dict = field(default_factory=dict)


@dataclass
class NonOrderRecord:
    """Record for a normalized non-order LLM output (HOLD/PASS/AVOID/WATCH)."""
    action: str
    symbol: str | None = None
    raw_decision: dict = field(default_factory=dict)


@dataclass
class NormalizationResult:
    """Aggregate result from normalizing all decisions in a PM cycle."""
    orders: list[NormalizedOrder] = field(default_factory=list)
    rejections: list[NormalizedOrder] = field(default_factory=list)
    non_orders: list[NonOrderRecord] = field(default_factory=list)


# ── Entry Normalizer Constants & Helpers ──────────────────────────────────────

_ENTRY_ACTIONS = {"BUY", "SHORT"}
_NON_ORDER_ACTIONS = {"HOLD", "PASS", "AVOID", "WATCH"}
_ACTION_ALIASES = {"SKIP": "PASS"}
MAX_PM_PORTFOLIO_NOTE_CHARS = 420
MAX_PM_RATIONALE_CHARS = 280


def _compact_text_for_decision_log(value, max_chars: int) -> str:
    """Keep operator-facing decision log text concise and single-line.

    Local/specialized models can obey the JSON shape while still writing long
    narrative commentary.  The dashboard decision log is an operator feed, not
    a scratchpad, so clamp prose deterministically before persistence.
    """
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text

    # Prefer a clean sentence boundary, but only if it does not discard too
    # much useful context. Otherwise cut at the last word boundary.
    boundary = max(text.rfind(". ", 0, max_chars - 1), text.rfind("; ", 0, max_chars - 1))
    if boundary >= int(max_chars * 0.55):
        return text[: boundary + 1].strip()

    clipped = text[: max_chars - 1].rstrip()
    space = clipped.rfind(" ")
    if space >= int(max_chars * 0.70):
        clipped = clipped[:space].rstrip()
    return clipped.rstrip(".,;:") + "…"


def _enforce_pm_decision_log_contract(result: dict) -> dict:
    """Clamp PM notes/rationales to the dashboard-facing contract."""
    if not isinstance(result, dict):
        return result

    sanitized = dict(result)
    sanitized["portfolio_notes"] = _compact_text_for_decision_log(
        sanitized.get("portfolio_notes", ""),
        MAX_PM_PORTFOLIO_NOTE_CHARS,
    )

    decisions = sanitized.get("decisions")
    if isinstance(decisions, list):
        clean_decisions = []
        for decision in decisions:
            if isinstance(decision, dict):
                d = dict(decision)
                if "rationale" in d:
                    d["rationale"] = _compact_text_for_decision_log(
                        d.get("rationale", ""),
                        MAX_PM_RATIONALE_CHARS,
                    )
                clean_decisions.append(d)
            else:
                clean_decisions.append(decision)
        sanitized["decisions"] = clean_decisions

    return sanitized


def _validate_action(decision: dict) -> tuple[str, str | None, str | None]:
    """Classify action from a raw LLM decision dict.

    Returns (category, normalized_action, reason_code) where:
        category: "entry" | "non_order" | "rejected"
        normalized_action: uppercase action string, or None if missing/empty
        reason_code: None for "entry"/"non_order", or a rejection reason code
    """
    decision_type_raw = decision.get("decision_type")
    if (
        isinstance(decision_type_raw, str)
        and decision_type_raw.strip().lower() == "reject"
    ):
        return ("non_order", "REJECT", None)

    action_raw = decision.get("action")

    # Missing, None, or empty action
    if action_raw is None or (isinstance(action_raw, str) and not action_raw.strip()):
        return ("rejected", None, "missing_action")

    # Non-string actions (e.g., int, bool, list) → treat as unsupported
    if not isinstance(action_raw, str):
        return ("rejected", None, "missing_action")

    normalized = action_raw.strip().upper()

    if not normalized:
        return ("rejected", None, "missing_action")

    normalized = _ACTION_ALIASES.get(normalized, normalized)

    if normalized in _ENTRY_ACTIONS:
        return ("entry", normalized, None)

    if normalized in _NON_ORDER_ACTIONS:
        return ("non_order", normalized, None)

    return ("rejected", normalized, "unsupported_action")


def _apply_scaffold_geometry_defaults(
    decisions: list[dict],
    scaffold_results: dict[str, dict | None],
) -> list[dict]:
    """Fill missing executable geometry from the selected scaffold candidate.

    The PM prompt requires accept/adjust decisions to include a
    geometry_candidate_id. If the model names a scaffold candidate but omits
    one of the candidate-owned fields, repair only missing values from the
    deterministic scaffold. Existing PM-adjusted values are preserved.
    """
    if not decisions or not scaffold_results:
        return decisions

    scaffold_lookup = {sym.upper(): result for sym, result in scaffold_results.items()}
    repaired: list[dict] = []

    for item in decisions:
        if not isinstance(item, dict):
            repaired.append(item)
            continue

        decision = dict(item)
        decision_type = str(decision.get("decision_type") or "").strip().lower()
        if decision_type not in {"accept", "adjust"}:
            repaired.append(decision)
            continue

        symbol_raw = decision.get("symbol")
        candidate_id = decision.get("geometry_candidate_id")
        if not isinstance(symbol_raw, str) or not isinstance(candidate_id, str):
            repaired.append(decision)
            continue

        scaffold = scaffold_lookup.get(symbol_raw.strip().upper())
        if not isinstance(scaffold, dict):
            repaired.append(decision)
            continue

        candidates = scaffold.get("candidates")
        if not isinstance(candidates, list):
            repaired.append(decision)
            continue

        selected = None
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("candidate_id") == candidate_id:
                selected = candidate
                break

        if selected is None:
            repaired.append(decision)
            continue

        for field in ("entry_price", "stop_loss", "target"):
            if decision.get(field) in (None, "") and selected.get(field) not in (None, ""):
                decision[field] = selected[field]

        if decision.get("geometry_candidate_name") in (None, "") and selected.get("name"):
            decision["geometry_candidate_name"] = selected["name"]

        repaired.append(decision)

    return repaired


def _validate_symbol(
    symbol_raw: Any, entry_signals: dict[str, dict]
) -> tuple[bool, str | None, str | None, dict | None]:
    """Validate and normalize symbol against allowed set.

    Returns (valid, canonical_symbol, reason_code, details).
    """
    # Missing, None, or non-string → missing_symbol
    if symbol_raw is None or not isinstance(symbol_raw, str):
        return (False, None, "missing_symbol", None)

    stripped = symbol_raw.strip()

    # Empty or whitespace-only → missing_symbol
    if not stripped:
        return (False, None, "missing_symbol", None)

    # Build case-insensitive lookup: {KEY_UPPER: original_key}
    lookup = {key.upper(): key for key in entry_signals}

    canonical = lookup.get(stripped.upper())
    if canonical is not None:
        return (True, canonical, None, None)

    # Not a member — provide extra detail if it looks like a concept
    if "/" in stripped or " " in stripped:
        return (
            False,
            None,
            "unsupported_symbol",
            {"symbol": stripped, "note": "appears to be a concept rather than a ticker"},
        )

    return (False, None, "unsupported_symbol", {"symbol": stripped})


def _validate_quantity(quantity_raw: Any) -> tuple[bool, int | None, str | None]:
    """Validate quantity is a positive whole integer.

    Returns (valid, coerced_quantity, reason_code).
    Rejects: None, booleans, fractional floats, fractional strings,
             zero, negative, non-numeric strings.
    """
    import math

    # Reject None
    if quantity_raw is None:
        return (False, None, "invalid_quantity")

    # Reject booleans FIRST — isinstance(True, int) is True in Python
    if isinstance(quantity_raw, bool):
        return (False, None, "invalid_quantity")

    # Integer path
    if isinstance(quantity_raw, int):
        if quantity_raw >= 1:
            return (True, quantity_raw, None)
        return (False, None, "invalid_quantity")

    # Float path
    if isinstance(quantity_raw, float):
        # Reject NaN and Inf
        if math.isnan(quantity_raw) or math.isinf(quantity_raw):
            return (False, None, "invalid_quantity")
        # Check if it's a whole number
        if quantity_raw != int(quantity_raw):
            return (False, None, "invalid_quantity")
        int_val = int(quantity_raw)
        if int_val >= 1:
            return (True, int_val, None)
        return (False, None, "invalid_quantity")

    # String path
    if isinstance(quantity_raw, str):
        stripped = quantity_raw.strip()
        if not stripped:
            return (False, None, "invalid_quantity")
        try:
            float_val = float(stripped)
        except (ValueError, OverflowError):
            return (False, None, "invalid_quantity")
        # Reject NaN/Inf strings
        if math.isnan(float_val) or math.isinf(float_val):
            return (False, None, "invalid_quantity")
        # Check if it's a whole number
        if float_val != int(float_val):
            return (False, None, "invalid_quantity")
        int_val = int(float_val)
        if int_val >= 1:
            return (True, int_val, None)
        return (False, None, "invalid_quantity")

    # Everything else → reject
    return (False, None, "invalid_quantity")


def _try_positive_float(value) -> float | None:
    """Attempt to coerce a value to a positive float. Returns None on failure."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if f > 0:
            return f
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            f = float(stripped)
        except (ValueError, OverflowError):
            return None
        if f > 0:
            return f
        return None
    return None


def _resolve_price(
    decision: dict,
    symbol: str,
    get_live_quote: Callable[[str], float | None] | None,
) -> tuple[bool, float | None, str | None, str | None]:
    """Resolve entry price with fallback to live quote.

    Priority: entry_price > price > live quote repair.
    Returns (valid, price, price_source, reason_code).
    price_source is one of: "entry_price", "price", "live_quote", or None on failure.
    """
    # 1. Check entry_price key first
    ep = _try_positive_float(decision.get("entry_price"))
    if ep is not None:
        return (True, ep, "entry_price", None)

    # 2. Check price key
    p = _try_positive_float(decision.get("price"))
    if p is not None:
        return (True, p, "price", None)

    # 3. Attempt live quote repair
    if get_live_quote is not None:
        quote = get_live_quote(symbol)
        live = _try_positive_float(quote)
        if live is not None:
            return (True, live, "live_quote", None)

    # 4. All paths failed
    return (False, None, None, "invalid_price")


def _resolve_stop(decision: dict) -> tuple[bool, float | None, str | None]:
    """Resolve stop price from multiple possible keys.

    Keys checked in priority order: stop_loss, stop_price, stop.
    Returns (valid, stop_price, reason_code).
    """
    for key in ("stop_loss", "stop_price", "stop"):
        val = _try_positive_float(decision.get(key))
        if val is not None:
            return (True, val, None)
    return (False, None, "missing_stop_loss")


def _resolve_target(decision: dict) -> tuple[bool, float | None, str | None]:
    """Resolve target price from multiple possible keys.

    Keys checked in priority order: target, target_price, profit_target.
    Returns (valid, target_price, reason_code).
    """
    for key in ("target", "target_price", "profit_target"):
        val = _try_positive_float(decision.get(key))
        if val is not None:
            return (True, val, None)
    return (False, None, "missing_target")


def _validate_geometry(
    action: str, entry_price: float, stop: float, target: float
) -> tuple[bool, str | None]:
    """Validate directional geometry consistency.

    BUY:   stop < entry < target
    SHORT: target < entry < stop
    Returns (valid, reason_code).
    """
    if action == "BUY":
        if stop < entry_price < target:
            return (True, None)
        return (False, "invalid_geometry")

    # SHORT
    if target < entry_price < stop:
        return (True, None)
    return (False, "invalid_geometry")


def normalize_pm_entry_decisions(
    decisions: list[dict],
    entry_signals: dict[str, dict],
    *,
    get_live_quote: Callable[[str], float | None] | None = None,
) -> NormalizationResult:
    """Validate and normalize raw LLM decisions into structured results.

    Args:
        decisions: Raw decision dicts from LLM JSON response.
        entry_signals: Dict of {symbol: signal_dict} for this PM cycle.
        get_live_quote: Optional callable that returns a live price for a symbol.

    Returns:
        NormalizationResult with orders, rejections, and non_orders categorized.

    Validation order (short-circuits on first failure):
        1. Action validation (BUY/SHORT vs non-order vs unsupported)
        2. Symbol validation (membership in entry_signals, case-insensitive)
        3. Quantity validation (positive whole integer)
        4. Price validation (entry_price > price > live repair)
        5. Stop presence (stop_loss / stop_price / stop)
        6. Target presence (target / target_price / profit_target)
        7. Directional geometry (stop/entry/target consistency)
        8. Setup type resolution (via resolve_setup_type — warning only, not rejection)
    """
    orders: list[NormalizedOrder] = []
    rejections: list[NormalizedOrder] = []
    non_orders: list[NonOrderRecord] = []

    for item in decisions:
        # ── Non-dict items → immediate rejection ──
        if not isinstance(item, dict):
            rejections.append(NormalizedOrder(
                ok=False,
                reason_code="invalid_decision_shape",
                reason="Decision is not a dict",
                details={"repr": repr(item)[:200]},
                raw_decision={},
            ))
            continue

        decision = item

        # ── 1. Validate action ──
        category, action, reason_code = _validate_action(decision)

        if category == "non_order":
            non_orders.append(NonOrderRecord(
                action=action or "",
                symbol=decision.get("symbol") if isinstance(decision.get("symbol"), str) else None,
                raw_decision=decision,
            ))
            continue

        if category == "rejected":
            rejections.append(NormalizedOrder(
                ok=False,
                reason_code=reason_code,
                reason=f"Action rejected: {reason_code}",
                details={"action": decision.get("action")},
                raw_decision=decision,
            ))
            continue

        # category == "entry" — continue validation
        # ── 2. Validate symbol ──
        valid, canonical_symbol, sym_reason, sym_details = _validate_symbol(
            decision.get("symbol"), entry_signals
        )
        if not valid:
            rejections.append(NormalizedOrder(
                ok=False,
                reason_code=sym_reason,
                reason=f"Symbol rejected: {sym_reason}",
                details=sym_details or {},
                raw_decision=decision,
            ))
            continue

        # ── 3. Validate quantity ──
        valid, quantity, qty_reason = _validate_quantity(decision.get("quantity"))
        if not valid:
            rejections.append(NormalizedOrder(
                ok=False,
                reason_code=qty_reason,
                reason=f"Quantity rejected: {qty_reason}",
                details={"quantity": decision.get("quantity")},
                raw_decision=decision,
            ))
            continue

        # ── 4. Resolve price ──
        valid, price, price_source, price_reason = _resolve_price(
            decision, canonical_symbol, get_live_quote
        )
        if not valid:
            rejections.append(NormalizedOrder(
                ok=False,
                reason_code=price_reason,
                reason=f"Price rejected: {price_reason}",
                details={"entry_price": decision.get("entry_price"), "price": decision.get("price")},
                raw_decision=decision,
            ))
            continue

        # ── 5. Resolve stop ──
        valid, stop, stop_reason = _resolve_stop(decision)
        if not valid:
            rejections.append(NormalizedOrder(
                ok=False,
                reason_code=stop_reason,
                reason=f"Stop rejected: {stop_reason}",
                details={},
                raw_decision=decision,
            ))
            continue

        # ── 6. Resolve target ──
        valid, target, target_reason = _resolve_target(decision)
        if not valid:
            rejections.append(NormalizedOrder(
                ok=False,
                reason_code=target_reason,
                reason=f"Target rejected: {target_reason}",
                details={},
                raw_decision=decision,
            ))
            continue

        # ── 7. Validate geometry ──
        valid, geom_reason = _validate_geometry(action, price, stop, target)
        if not valid:
            rejections.append(NormalizedOrder(
                ok=False,
                reason_code=geom_reason,
                reason=f"Geometry rejected: {geom_reason}",
                details={"action": action, "entry_price": price, "stop": stop, "target": target},
                raw_decision=decision,
            ))
            continue

        # ── 8. Resolve setup type (warning only, not rejection) ──
        warnings: list[str] = []
        setup_type = resolve_setup_type(decision, entry_signals.get(canonical_symbol, {}))
        if setup_type is None:
            warnings.append(
                "Setup type could not be resolved — will be evaluated by gate pipeline"
            )

        # Add live quote repair warning if applicable
        if price_source == "live_quote":
            warnings.append(f"Price repaired from live quote: {price:.2f}")

        # ── Build validated order ──
        order = {
            "action": action,
            "symbol": canonical_symbol,
            "quantity": quantity,
            "entry_price": price,
            "stop": stop,
            "target": target,
            "setup_type": setup_type,
            "rationale": decision.get("rationale", ""),
            "price_source": price_source,
            "price_repaired_from_live_quote": price_source == "live_quote",
        }

        # Preserve optional override metadata (coerce, validate range, or drop)
        raw_conf = decision.get("override_confidence_score")
        if raw_conf is not None:
            try:
                score = float(raw_conf)
                if math.isfinite(score) and 0.0 <= score <= 10.0:
                    order["override_confidence_score"] = score
                # else: drop — NaN, inf, negatives, >10.0 are invalid
            except (TypeError, ValueError):
                pass  # Drop malformed values — gate receives None (fail-closed)

        raw_reason = decision.get("override_reason")
        if isinstance(raw_reason, str) and raw_reason.strip():
            order["override_reason"] = raw_reason.strip()

        orders.append(NormalizedOrder(
            ok=True,
            order=order,
            warnings=warnings,
            raw_decision=decision,
        ))

    return NormalizationResult(
        orders=orders,
        rejections=rejections,
        non_orders=non_orders,
    )


def _log_pm_decision_rejected(
    db, rejection: NormalizedOrder, profile_id: str
):
    """Log a pm_decision_rejected trade event for a rejected decision.

    Truncates raw decision JSON to 2000 characters max.
    Includes normalized symbol if parseable, null otherwise.

    Returns the TradeEvent object from log_trade_event for trade_event_id linkage.
    """
    raw_json = json.dumps(rejection.raw_decision, default=str)
    if len(raw_json) > 2000:
        raw_json = raw_json[:1997] + "..."

    # Resolve symbol: try details first, then raw_decision
    symbol = rejection.details.get("symbol")
    if symbol is None:
        raw_sym = rejection.raw_decision.get("symbol")
        if isinstance(raw_sym, str) and raw_sym.strip():
            symbol = raw_sym.strip()

    payload = {
        "reason_code": rejection.reason_code,
        "reason": rejection.reason,
        "symbol": symbol,
        "raw_decision": raw_json,
        "profile": profile_id,
    }

    return log_trade_event(
        db,
        "pm_decision_rejected",
        agent="portfolio_manager",
        symbol=symbol,
        profile=profile_id,
        payload=payload,
    )


def _store_pm_cycle_note(db, profile_id: str, notes: str) -> str:
    """Persist an operator-facing PM cycle note for the decision log."""
    stored_notes = _compact_text_for_decision_log(notes, MAX_PM_PORTFOLIO_NOTE_CHARS)
    if stored_notes:
        db.add(AgentMemory(
            agent=f"pm_{profile_id}",
            symbol=None,
            key="notes",
            value=stored_notes,
        ))
        db.commit()
    return stored_notes


def _log_no_trade_outcome(
    db,
    symbol: str,
    profile_id: str,
    category: str,
    *,
    scaffold_result: dict | None = None,
    rejection_rationale: str | None = None,
    gate_name: str | None = None,
    gate_reason: str | None = None,
    scaffold_candidate: dict | None = None,
    pm_adjusted_geometry: dict | None = None,
):
    """Log a no-trade outcome with correct blocker attribution.

    Categorizes no-trade outcomes as exactly one of:
      - "scaffold_unavailable": scaffold returned None or status != "ok"
      - "candidates_rejected_by_pm": PM rejected all scaffold candidates
      - "candidate_rejected_by_gate": PM selected a candidate but a Gate rejected it

    Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
    """
    # Build the log message based on category (Req 12.1: no Analyst-blaming language)
    if category == "scaffold_unavailable":
        # Req 12.2: log "geometry scaffold unavailable" with reason
        reason = ""
        if scaffold_result and isinstance(scaffold_result, dict):
            reason = scaffold_result.get("reason", "")
        elif scaffold_result is None:
            reason = "internal scaffold error"
        message = f"geometry scaffold unavailable: {reason}" if reason else "geometry scaffold unavailable"

    elif category == "candidates_rejected_by_pm":
        # Req 12.3: log PM rejection rationale referencing concrete factors
        message = f"PM rejected all scaffold candidates: {rejection_rationale}" if rejection_rationale else "PM rejected all scaffold candidates"

    elif category == "candidate_rejected_by_gate":
        # Req 12.4: log specific Gate name and rejection reason
        message = f"Gate rejected selected candidate: {gate_name}: {gate_reason}" if gate_name and gate_reason else "Gate rejected selected candidate"

    else:
        message = f"no-trade outcome: {category}"

    # Req 12.5: include scaffold candidate geometry in rejection records
    payload: dict = {
        "no_trade_outcome": category,
        "symbol": symbol,
        "profile": profile_id,
        "message": message,
    }

    if scaffold_candidate and isinstance(scaffold_candidate, dict):
        payload["scaffold_candidate"] = {
            "candidate_id": scaffold_candidate.get("candidate_id"),
            "name": scaffold_candidate.get("name"),
            "entry_price": scaffold_candidate.get("entry_price"),
            "stop_loss": scaffold_candidate.get("stop_loss"),
            "target": scaffold_candidate.get("target"),
            "risk_reward": scaffold_candidate.get("risk_reward"),
        }

    if pm_adjusted_geometry and isinstance(pm_adjusted_geometry, dict):
        payload["pm_adjusted_geometry"] = {
            "entry_price": pm_adjusted_geometry.get("entry_price"),
            "stop_loss": pm_adjusted_geometry.get("stop_loss"),
            "target": pm_adjusted_geometry.get("target"),
        }

    if gate_name:
        payload["gate_name"] = gate_name
    if gate_reason:
        payload["gate_reason"] = gate_reason

    log_trade_event(
        db,
        "no_trade_outcome",
        agent=f"pm_{profile_id}",
        symbol=symbol,
        profile=profile_id,
        message=message,
        payload=payload,
    )


def _should_retry_pm_contract(
    original_decisions: list[dict],
    norm_result: NormalizationResult,
) -> bool:
    """Determine whether a contract validation retry is warranted.

    Returns True when ALL of the following hold:
    1. PM_CONTRACT_RETRY_ENABLED env var is truthy
    2. original_decisions was non-empty (PM attempted something)
    3. norm_result.rejections is non-empty
    4. At least one rejection has a raw_decision with action
       matching BUY or SHORT (case-insensitive)

    Gate rejections (valid orders blocked downstream) are NOT in
    norm_result.rejections and thus never trigger retry.
    """
    # 1. Check env var
    if os.environ.get("PM_CONTRACT_RETRY_ENABLED", "").strip().lower() not in ("true", "1", "yes"):
        return False

    # 2. Original decisions must be non-empty
    if not original_decisions:
        return False

    # 3. Rejections must be non-empty
    if not norm_result.rejections:
        return False

    # 4. At least one rejection must have a BUY or SHORT action
    for rejection in norm_result.rejections:
        action_raw = rejection.raw_decision.get("action")
        if isinstance(action_raw, str) and action_raw.strip().upper() in ("BUY", "SHORT"):
            return True

    return False


def _summarize_normalization_rejections(
    rejections: list[NormalizedOrder],
) -> str:
    """Build a compact human-readable summary of normalizer rejections.

    Returns a string like:
        - AMD BUY: invalid_quantity, missing_stop_loss
        - NVDA SHORT: invalid_geometry

    Groups rejections by (symbol, action) and lists all reason_codes.
    Used in the retry prompt to show the PM what went wrong.
    """
    from collections import OrderedDict

    # Group reason_codes by (symbol, action), preserving insertion order
    groups: OrderedDict[tuple[str, str], list[str]] = OrderedDict()

    for r in rejections:
        symbol = "UNKNOWN"
        action = "UNKNOWN"

        raw = r.raw_decision
        if isinstance(raw, dict):
            sym_val = raw.get("symbol")
            if isinstance(sym_val, str) and sym_val.strip():
                symbol = sym_val.strip().upper()
            act_val = raw.get("action")
            if isinstance(act_val, str) and act_val.strip():
                action = act_val.strip().upper()

        reason = r.reason_code or "unknown"
        key = (symbol, action)
        if key not in groups:
            groups[key] = []
        groups[key].append(reason)

    lines = []
    for (symbol, action), reasons in groups.items():
        lines.append(f"- {symbol} {action}: {', '.join(reasons)}")

    return "\n".join(lines)


def _build_pm_contract_retry_prompt(
    original_response: dict,
    rejections: list[NormalizedOrder],
    entry_signals: dict[str, dict],
) -> tuple[str, str]:
    """Build the system and user prompts for a contract validation retry.

    Args:
        original_response: The full parsed JSON from the first PM call
            (contains both decisions and portfolio_notes).
        rejections: The NormalizedOrder(ok=False) items from normalization.
        entry_signals: The allowed entry signals dict for this cycle.

    Returns:
        (system_prompt, user_prompt) tuple for the retry LLM call.

    The retry prompt includes:
    - The original PM JSON response (decisions + portfolio_notes)
    - A rejection summary (symbol, action, reason_code for each)
    - Allowed symbols for this cycle
    - Compact signal context for rejected symbols that are in entry_signals
    - The strict output contract
    - Instruction: "Only return corrected versions of rejected decisions;
      do not repeat already-valid orders."
    - Instruction: "If missing fields cannot be recovered, return empty
      decisions and move the idea to portfolio_notes."

    The retry prompt does NOT include:
    - Full system prompt context (signals, positions, portfolio state)
    - Maintenance/reversal review context
    """
    # ── System prompt: minimal contract-focused ──
    system_prompt = (
        "You are a portfolio manager correcting malformed trade decisions. "
        "Your ONLY job is to fix rejected decisions into valid executable orders "
        "or move them to portfolio_notes. Follow the output contract exactly."
    )

    # ── User prompt construction ──
    parts: list[str] = []

    # 1. Original PM JSON response
    parts.append("## YOUR PREVIOUS RESPONSE\n")
    parts.append(json.dumps(original_response, indent=2, default=str))

    # 2. Rejection summary
    parts.append("\n\n## REJECTION SUMMARY\n")
    parts.append("The following decisions were rejected by the normalizer:\n")
    parts.append(_summarize_normalization_rejections(rejections))

    # 3. Allowed symbols
    allowed_symbols = sorted(entry_signals.keys())
    parts.append("\n\n## ALLOWED SYMBOLS\n")
    parts.append(", ".join(allowed_symbols))

    # 4. Compact signal context for rejected symbols that exist in entry_signals
    # Build set of rejected symbols (case-insensitive lookup)
    rejected_symbols: set[str] = set()
    for r in rejections:
        raw = r.raw_decision
        if isinstance(raw, dict):
            sym_val = raw.get("symbol")
            if isinstance(sym_val, str) and sym_val.strip():
                rejected_symbols.add(sym_val.strip().upper())

    # Match rejected symbols against entry_signals (case-insensitive)
    signal_lookup = {k.upper(): k for k in entry_signals}
    signal_context_lines: list[str] = []
    for rejected_sym in sorted(rejected_symbols):
        canonical_key = signal_lookup.get(rejected_sym)
        if canonical_key is not None:
            sig = entry_signals[canonical_key]
            context = {
                "direction": sig.get("direction"),
                "strength": sig.get("strength"),
                "entry_price": sig.get("entry_price"),
                "stop_loss": sig.get("stop_loss"),
                "target": sig.get("target"),
            }
            signal_context_lines.append(
                f"  {canonical_key}: {json.dumps(context, default=str)}"
            )

    if signal_context_lines:
        parts.append("\n\n## SIGNAL CONTEXT FOR REJECTED SYMBOLS\n")
        parts.append("\n".join(signal_context_lines))

    # 5. Strict output contract
    parts.append("\n\n## OUTPUT CONTRACT\n")
    parts.append(
        "Return valid JSON with this structure:\n"
        '{"decisions": [...], "portfolio_notes": "..."}\n\n'
        "Each decision in decisions[] MUST have ALL of these fields:\n"
        "- symbol: string (must be from ALLOWED SYMBOLS above)\n"
        "- action: BUY or SHORT\n"
        "- quantity: positive integer (no nulls, no zero)\n"
        "- entry_price: positive float\n"
        "- stop_loss: positive float\n"
        "- target: positive float\n"
        "- setup_type: string\n"
        "- rationale: string\n\n"
        "Geometry rules:\n"
        "- BUY: stop_loss < entry_price < target\n"
        "- SHORT: target < entry_price < stop_loss"
    )

    # 6. Instructions
    parts.append("\n\n## INSTRUCTIONS\n")
    parts.append(
        "Only return corrected versions of rejected decisions; "
        "do not repeat already-valid orders.\n\n"
        "If missing fields cannot be recovered from your previous response "
        "or the signal context provided, return empty decisions and move the "
        "idea to portfolio_notes. Do not invent order geometry."
    )

    user_prompt = "".join(parts)

    return (system_prompt, user_prompt)


def _deduplicate_retry_orders(
    original_orders: list[NormalizedOrder],
    retry_orders: list[NormalizedOrder],
) -> list[NormalizedOrder]:
    """Remove retry orders that conflict with an original valid order by symbol.

    A retry order is considered a conflict if its symbol matches any order
    in original_orders (case-insensitive comparison). Same profile, same cycle,
    same symbol should have at most one entry intent.

    Also deduplicates within retry_orders itself: if retry returns multiple
    orders for the same symbol, only the first is kept.

    Returns: list of retry orders that are NOT conflicts/duplicates.
    """
    # Build set of symbols already present in original valid orders
    original_symbols: set[str] = set()
    for o in original_orders:
        if o.order and isinstance(o.order.get("symbol"), str):
            original_symbols.add(o.order["symbol"].upper())

    result: list[NormalizedOrder] = []
    seen_retry_symbols: set[str] = set()

    for retry_order in retry_orders:
        symbol_raw = retry_order.order.get("symbol", "") if retry_order.order else ""
        symbol_key = symbol_raw.upper() if isinstance(symbol_raw, str) else ""

        # Check conflict with original orders
        if symbol_key in original_symbols:
            log.info(
                "Retry order for %s skipped: conflicts with original valid order",
                symbol_key,
            )
            continue

        # Check duplicate within retry orders (first-in-list wins)
        if symbol_key in seen_retry_symbols:
            log.info(
                "Retry order for %s skipped: duplicate within retry orders",
                symbol_key,
            )
            continue

        seen_retry_symbols.add(symbol_key)
        result.append(retry_order)

    return result


def _merge_retry_notes(
    original_notes: str,
    retry_notes: str | None,
) -> str:
    """Merge original portfolio_notes with retry notes.

    Rules:
    - Original notes are always preserved.
    - If retry_notes is non-empty, append with prefix:
      "\\n\\nContract retry note: " + retry_notes
    - If retry_notes is empty/None, return original_notes unchanged.

    This ensures that even when retry returns decisions:[] with useful
    explanatory notes (e.g., "AMD not executable — no valid stop level"),
    that context is preserved for the operator.
    """
    if retry_notes:
        return original_notes + "\n\nContract retry note: " + retry_notes
    return original_notes


def _log_retry_triggered(
    db, profile_id: str, norm_result: NormalizationResult,
    entry_signals: dict[str, dict],
) -> None:
    """Log pm_contract_retry_triggered trade event."""
    reason_counts = collections.Counter(
        r.reason_code for r in norm_result.rejections
    )
    log_trade_event(
        db,
        "pm_contract_retry_triggered",
        agent="portfolio_manager",
        symbol=None,
        profile=profile_id,
        payload={
            "profile": profile_id,
            "reason_counts": dict(reason_counts),
            "allowed_symbols": sorted(entry_signals.keys()),
            "initial_rejection_count": len(norm_result.rejections),
        },
    )


def _log_retry_succeeded(
    db, profile_id: str,
    initial_norm: NormalizationResult,
    retry_norm: NormalizationResult,
    deduped_retry: list[NormalizedOrder],
) -> None:
    """Log pm_contract_retry_succeeded trade event."""
    log_trade_event(
        db,
        "pm_contract_retry_succeeded",
        agent="portfolio_manager",
        symbol=None,
        profile=profile_id,
        payload={
            "profile": profile_id,
            "initial_rejection_count": len(initial_norm.rejections),
            "retry_rejection_count": len(retry_norm.rejections),
            "retry_order_count": len(deduped_retry),
        },
    )


def _log_retry_failed(
    db, profile_id: str,
    initial_norm: NormalizationResult,
    retry_norm: NormalizationResult,
) -> None:
    """Log pm_contract_retry_failed trade event."""
    retry_reason_counts = collections.Counter(
        r.reason_code for r in retry_norm.rejections
    )
    log_trade_event(
        db,
        "pm_contract_retry_failed",
        agent="portfolio_manager",
        symbol=None,
        profile=profile_id,
        payload={
            "profile": profile_id,
            "initial_rejection_count": len(initial_norm.rejections),
            "retry_rejection_count": len(retry_norm.rejections),
            "retry_reason_counts": dict(retry_reason_counts),
        },
    )


def _log_pm_decision_corrected(
    db, rejection: NormalizedOrder, profile_id: str,
) -> None:
    """Log a pm_contract_retry_corrected trade event for a rejection that was
    subsequently fixed by the retry.

    These are NOT counted as normal pm_decision_rejected — they represent
    transient malformed output that the PM self-corrected.
    """
    raw_json = json.dumps(rejection.raw_decision, default=str)
    if len(raw_json) > 2000:
        raw_json = raw_json[:1997] + "..."
    # Resolve symbol robustly: try raw_decision first, fall back to details
    symbol = (
        rejection.raw_decision.get("symbol")
        or rejection.details.get("symbol")
    )
    log_trade_event(
        db,
        "pm_contract_retry_corrected",
        agent="portfolio_manager",
        symbol=None,
        profile=profile_id,
        payload={
            "profile": profile_id,
            "reason_code": rejection.reason_code,
            "symbol": symbol,
            "raw_decision": raw_json,
        },
    )


def _coerce_price(value, field_name: str, symbol: str) -> float | None:
    """Normalize LLM/database price values before persisting or %.2f logging."""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip().replace("$", "").replace(",", "")
        return float(value)
    except (TypeError, ValueError):
        log.warning(
            "Ignoring invalid %s for %s from maintenance review: %r",
            field_name, symbol, value,
        )
        return None


def should_suppress_maintenance_stop(side, old_stop, new_stop_raw):
    """Determine whether a maintenance tighten_stop proposal should be suppressed.

    Pure deterministic function — no side effects, no logging, no I/O.
    Called from the maintenance review tighten_stop branch BEFORE _coerce_price()
    to intercept invalid/non-monotonic proposals before they reach apply_stop_update().

    Parameters
    ----------
    side : str | None
        Trade side (long/LONG/buy, short/SHORT/sell, or unknown/None).
    old_stop : float | int | str | None
        Current stop value for the position.
    new_stop_raw : any
        Raw proposed stop value from the LLM (before coercion).

    Returns
    -------
    tuple[bool, str | None]
        (should_suppress, reason) where reason is one of:
        - "invalid_stop_value" — new_stop is null, non-numeric, zero, or negative
        - "non_monotonic_or_noop" — new_stop is not strictly tighter than old_stop
        - None — proposal should pass through to apply_stop_update()
    """
    # --- Normalize side case-insensitively ---
    normalized_side = None
    if isinstance(side, str) and side.strip():
        lower_side = side.strip().lower()
        if lower_side in ("long", "buy"):
            normalized_side = "long"
        elif lower_side in ("short", "sell"):
            normalized_side = "short"

    # Unknown/missing side → delegate to stop authority
    if normalized_side is None:
        return (False, None)

    # --- Validate old_stop ---
    # If old_stop is None, non-positive, or non-numeric → delegate to stop authority
    if old_stop is None:
        return (False, None)
    try:
        old_stop_f = float(old_stop)
    except (TypeError, ValueError):
        return (False, None)
    if old_stop_f <= 0 or not math.isfinite(old_stop_f):
        return (False, None)

    # --- Validate new_stop_raw ---
    # If new_stop_raw is None, non-numeric, zero, negative, or non-finite → suppress
    if new_stop_raw is None:
        return (True, "invalid_stop_value")
    try:
        new_stop_f = float(new_stop_raw)
    except (TypeError, ValueError):
        return (True, "invalid_stop_value")
    if new_stop_f <= 0 or not math.isfinite(new_stop_f):
        return (True, "invalid_stop_value")

    # --- Monotonicity check (side-aware) ---
    if normalized_side == "long" and new_stop_f <= old_stop_f:
        return (True, "non_monotonic_or_noop")
    if normalized_side == "short" and new_stop_f >= old_stop_f:
        return (True, "non_monotonic_or_noop")

    # Valid monotonic proposal → pass through
    return (False, None)


def _log_maintenance_stop_suppressed(
    db,
    *,
    trade_id: int,
    symbol: str,
    profile: str,
    side: str,
    old_stop,
    new_stop_raw,
    reason: str,
) -> None:
    """Log a maintenance_stop_suppressed event for a suppressed tighten_stop proposal.

    Emits exactly one event per trade per maintenance review invocation.
    The ``proposed_stop_raw`` field is included in the payload whenever
    ``new_stop_raw`` is not a usable stop value — i.e. None, non-parseable,
    zero, negative, or non-finite (NaN/inf).

    Parameters
    ----------
    db : Session
        SQLAlchemy session (caller owns commit/rollback).
    trade_id : int
        ID of the trade whose stop proposal was suppressed.
    symbol : str
        Ticker symbol for the position.
    profile : str
        Profile ID that owns the position.
    side : str
        Trade side ("long" or "short").
    old_stop : float | None
        Current stop price for the position.
    new_stop_raw : any
        Raw proposed stop value from the LLM (before coercion).
    reason : str
        Suppression reason ("non_monotonic_or_noop" or "invalid_stop_value").
    """
    # Determine proposed_stop (valid positive finite float, else None)
    # Always include proposed_stop_raw when proposed_stop is not a usable value
    proposed_stop = None
    include_raw = False
    if new_stop_raw is None:
        include_raw = True
    else:
        try:
            val = float(new_stop_raw)
            if val > 0 and math.isfinite(val):
                proposed_stop = val
            else:
                include_raw = True
        except (TypeError, ValueError):
            include_raw = True

    payload = {
        "trade_id": trade_id,
        "symbol": symbol,
        "profile": profile,
        "side": side,
        "old_stop": old_stop,
        "proposed_stop": proposed_stop,
        "reason": reason,
        "source_action": "tighten_stop",
    }

    # Include proposed_stop_raw when proposed_stop is not a usable value
    if include_raw:
        payload["proposed_stop_raw"] = str(new_stop_raw)

    log_trade_event(
        db,
        "maintenance_stop_suppressed",
        trade_id=trade_id,
        agent="portfolio_manager",
        symbol=symbol,
        profile=profile,
        payload=payload,
    )


def _coerce_quantity(value, *, symbol: str = "UNKNOWN") -> int:
    """Normalize LLM quantity values before sizing math.

    JSON-repair and local LLM output occasionally preserve explicit
    ``null`` quantities. Treat those as zero instead of letting downstream
    multipliers raise ``TypeError: unsupported operand type(s) for *:
    'NoneType' and 'float'``.
    """
    if value in (None, ""):
        return 0
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        log.warning("Ignoring invalid quantity for %s: %r", symbol, value)
        return 0

# Max minutes after market open (9:30 AM ET) that each setup type may be entered.
# Setup types NOT listed here have no entry-window restriction.
ENTRY_WINDOW_LIMITS = {
    "gap_and_go": 60,
    "orb": 60,
    "short_squeeze": 60,
}

# Hard guardrails from 2026-04-30 AMD review. High-WR, fast intraday
# setups need enough breathing room; otherwise execution noise negates the edge.
HIGH_WR_STOP_BUFFER_THRESHOLD = 0.60
MIN_HIGH_WR_INTRADAY_STOP_BUFFER_PCT = 0.015
HIGH_MOMENTUM_ASSETS = {"AMD", "NVDA", "TSLA"}
HIGH_MOMENTUM_COOLDOWN_MINUTES = 30

def _market_time_context() -> dict[str, str | int]:
    """Return explicit market/local time labels for LLM prompts.

    The model was previously shown only UTC (e.g. 14:00 UTC), which it can
    misread as 2 PM market time. Keep UTC for auditability, but lead with ET.
    """
    from pytz import timezone as _tz

    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(_tz("America/New_York"))
    now_mt = now_utc.astimezone(_tz("America/Denver"))
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_since_open = int((now_et - market_open).total_seconds() // 60)
    return {
        "et": now_et.strftime("%Y-%m-%d %H:%M %Z"),
        "mt": now_mt.strftime("%Y-%m-%d %H:%M %Z"),
        "utc": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "minutes_since_open": minutes_since_open,
    }

_FAST_INTRADAY_SETUPS = {
    "gap_and_go",
    "vwap_reclaim",
    "orb",
    "momentum_fade",
    "trend_pullback",
    "news_catalyst",
    "short_squeeze",
}


def _extract_minutes(value) -> int | None:
    """Best-effort conversion of timeframe/horizon fields into minutes."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).lower()
    if any(token in text for token in ("swing", "overnight", "multi-day", "multiday")):
        return None

    import re
    minute_values = [int(n) for n in re.findall(r"(\d+)\s*(?:m|min|mins|minute|minutes)\b", text)]
    if minute_values:
        return max(minute_values)

    hour_values = [int(n) * 60 for n in re.findall(r"(\d+)\s*(?:h|hr|hrs|hour|hours)\b", text)]
    if hour_values:
        return max(hour_values)

    return None




def _format_execution_audit(
    executed: list[dict],
    gate_blocked: list[dict] | None = None,
    rejections: list | None = None,
    non_orders: list | None = None,
    *,
    retry_info: dict | None = None,
) -> str:
    """Categorized execution audit for PM notes.

    Args:
        executed: Decisions that were filled (ok from execute_trade).
        gate_blocked: Decisions that passed normalizer but were rejected by
            the gate pipeline. Each dict has at minimum: symbol, action, message.
        rejections: NormalizedOrder(ok=False) results from the normalizer.
            Only UNRESOLVED rejections (corrected rejections excluded).
        non_orders: NonOrderRecord items (HOLD/PASS/AVOID/WATCH).
        retry_info: Optional dict with keys: attempted (bool), succeeded (bool),
            corrected_count (int). Describes the outcome of a contract retry.

    Categories:
      EXECUTED: Full trade details (action, qty, symbol, price, message)
      BLOCKED BY GATE: Individual rows with gate reason
      REJECTED MALFORMED PM OUTPUT: Count by reason_code (no individual rows)
      IGNORED NON-ORDER: Count by action type (no individual rows)
    """
    gate_blocked = gate_blocked or []
    rejections = rejections or []
    non_orders = non_orders or []

    if not executed and not gate_blocked and not rejections and not non_orders:
        return "Execution audit: no decisions this cycle."

    lines = ["Execution audit:"]

    # ── EXECUTED ──
    if executed:
        lines.append(f"── EXECUTED ({len(executed)}) ──")
        for d in executed:
            action = str(d.get("action") or "?").upper()
            symbol = d.get("symbol") or "?"
            qty = d.get("quantity") or 0
            price = d.get("price") or d.get("entry_price") or 0
            msg = d.get("message") or ""
            line = f"  • {action} {qty} {symbol} @ ${float(price or 0):.2f}"
            if msg:
                line += f" — {msg}"
            lines.append(line)

    # ── BLOCKED BY GATE ──
    if gate_blocked:
        lines.append(f"── BLOCKED BY GATE ({len(gate_blocked)}) ──")
        for d in gate_blocked:
            action = str(d.get("action") or "?").upper()
            symbol = d.get("symbol") or "?"
            msg = d.get("message") or ""
            line = f"  • {action} {symbol}"
            if msg:
                line += f" — {msg}"
            lines.append(line)

    # ── REJECTED MALFORMED PM OUTPUT ──
    if rejections:
        lines.append(f"── REJECTED MALFORMED PM OUTPUT ({len(rejections)}) ──")
        reason_counts = collections.Counter(
            getattr(r, "reason_code", None) or "unknown" for r in rejections
        )
        for reason_code, count in reason_counts.most_common():
            lines.append(f"  • {reason_code}: {count}")

    # ── Retry summary line ──
    if retry_info and retry_info.get("corrected_count", 0) > 0:
        lines.append(f"  • Contract retry corrected {retry_info['corrected_count']} malformed PM output(s).")
    elif retry_info and retry_info.get("attempted") and not retry_info.get("succeeded"):
        lines.append("  • Contract retry attempted; malformed output still rejected.")
    elif rejections:
        lines.append("  • Malformed order-shaped commentary was ignored before execution.")

    # ── IGNORED NON-ORDER ──
    if non_orders:
        lines.append(f"── IGNORED NON-ORDER ({len(non_orders)}) ──")
        action_counts = collections.Counter(
            str(getattr(no, "action", None) or "?").upper() for no in non_orders
        )
        summary = ", ".join(f"{act}: {cnt}" for act, cnt in action_counts.most_common())
        lines.append(f"  • {summary}")

    return "\n".join(lines)

def _is_fast_intraday_trade(decision: dict, signal: dict) -> bool:
    """Return True when the trade is clearly intraday and intended under ~60 minutes."""
    setup_type = (
        decision.get("setup_type") or decision.get("setup")
        or signal.get("setup_type") or signal.get("setup") or ""
    )
    setup_type = str(setup_type).lower()

    for key in (
        "holding_minutes", "expected_holding_minutes", "max_holding_minutes",
        "time_horizon_minutes", "duration_minutes",
    ):
        minutes = _extract_minutes(decision.get(key) or signal.get(key))
        if minutes is not None:
            return minutes <= 60

    timeframe = (
        decision.get("timeframe") or decision.get("time_horizon") or decision.get("duration")
        or signal.get("timeframe") or signal.get("time_horizon") or signal.get("duration")
    )
    minutes = _extract_minutes(timeframe)
    if minutes is not None:
        return minutes <= 60

    text = str(timeframe or "").lower()
    if "intraday" in text and not any(token in text for token in ("multi-day", "multiday", "swing")):
        return True

    return setup_type in _FAST_INTRADAY_SETUPS


def _apply_high_wr_stop_buffer(
    action: str,
    price: float,
    stop: float | None,
    decision: dict,
    signal: dict,
    case_stats: dict,
) -> float | None:
    """Widen too-tight stops for high-win-rate, fast intraday setups."""
    if action not in ("BUY", "SHORT") or not price or not stop:
        return stop

    win_rate = float(case_stats.get("win_rate") or 0.0)
    if win_rate <= HIGH_WR_STOP_BUFFER_THRESHOLD:
        return stop
    if not _is_fast_intraday_trade(decision, signal):
        return stop

    min_distance = price * MIN_HIGH_WR_INTRADAY_STOP_BUFFER_PCT
    if action == "BUY":
        min_stop = round(price - min_distance, 2)
        if stop > min_stop:
            log.warning(
                "Stop buffer enforced for %s: high-WR setup %.0f%% requires >=%.1f%% breathing room; %.2f → %.2f",
                decision.get("symbol", "?"), win_rate * 100,
                MIN_HIGH_WR_INTRADAY_STOP_BUFFER_PCT * 100, stop, min_stop,
            )
            return min_stop
    else:
        min_stop = round(price + min_distance, 2)
        if stop < min_stop:
            log.warning(
                "Stop buffer enforced for %s: high-WR setup %.0f%% requires >=%.1f%% breathing room; %.2f → %.2f",
                decision.get("symbol", "?"), win_rate * 100,
                MIN_HIGH_WR_INTRADAY_STOP_BUFFER_PCT * 100, stop, min_stop,
            )
            return min_stop

    return stop


def _high_momentum_cooldown_message(db, symbol: str) -> str | None:
    """Block fresh entries shortly after a stop/loss on volatile names across profiles."""
    symbol = (symbol or "").upper()
    if symbol not in HIGH_MOMENTUM_ASSETS:
        return None

    cutoff = datetime.utcnow() - timedelta(minutes=HIGH_MOMENTUM_COOLDOWN_MINUTES)
    recent = (
        db.query(Trade)
        .filter_by(symbol=symbol, status="closed")
        .filter(Trade.exit_time >= cutoff)
        .order_by(Trade.exit_time.desc())
        .first()
    )
    if not recent:
        return None

    reason = (recent.reason_exit or "").lower()
    stopped_or_loss = "stop" in reason or (recent.pnl is not None and recent.pnl < 0)
    if not stopped_or_loss:
        return None

    return (
        f"Cooldown active for {symbol}: recent stop/loss in {recent.profile} profile "
        f"within {HIGH_MOMENTUM_COOLDOWN_MINUTES} minutes — blocking cascading re-entry"
    )


# Recovery probe contract — injected into SYSTEM_PROMPT_TEMPLATE only for moderate profile
RECOVERY_PROBE_CONTRACT = (
    "\n\nRECOVERY PROBE FIELDS (moderate profile only):\n"
    "- override_confidence_score: optional float (0.0-10.0). "
    "Provide when entering a setup that is in rolling underperformance for your profile. "
    "Required for recovery-probe eligibility.\n"
    "- override_reason: optional string. Brief justification for the override confidence.\n"
)


SYSTEM_PROMPT_TEMPLATE = """You are a portfolio manager for paper day trading.
Profile: {profile_name} {emoji}
{personality}

You receive analyst signals (direction, setup quality, key levels, invalidation, confidence)
along with geometry scaffold candidates (entry/stop/target proposals), your current positions,
cash balance, and reviewer feedback.

The Analyst tells you: direction, setup quality, key levels, invalidation, confidence.
The Analyst does NOT tell you how to trade. That is entirely your job.
The Geometry Scaffold proposes candidate trade plans with concrete entry/stop/target numbers.
You decide whether to accept, adjust, or reject those candidates.

Given the Analyst's read and the scaffold candidates, you decide:
  - Whether to act at all (maybe the setup is right but timing is wrong for your profile)
  - decision_type: "accept" (use scaffold candidate as-is), "adjust" (modify scaffold candidate), or "reject" (no trade)
  - Action: BUY / SHORT / pass
  - Entry price (from scaffold candidate, or adjusted based on your judgment)
  - Stop placement (from scaffold candidate, or adjusted for your profile's risk tolerance)
  - Target (from scaffold candidate, or adjusted for your profile's R:R requirements)
  - Position size (based on your profile's max position % and stop distance)
  - Scale in / scale out if appropriate for your profile

Your constraints:
- Max positions: {max_positions}
- Max position size: {max_position_pct}% of portfolio
- Minimum R:R ratio: {min_risk_reward}:1
- Min signal strength to act: {min_signal_strength}
- Daily loss limit: {max_daily_loss_pct}%
- Avoid first {avoid_first_minutes} min and last {avoid_last_minutes} min of session

Decide which trades to make. For each:
{{
  "decisions": [
    {{
      "symbol": "SPY",
      "action": "BUY",
      "decision_type": "accept",
      "geometry_candidate_id": "spy_long_support_bounce_1",
      "geometry_candidate_name": "support_bounce",
      "quantity": 10,
      "entry_price": 450.00,
      "stop_loss": 447.00,
      "target": 455.00,
      "setup_type": "breakout_pullback",
      "rationale": "why you're doing this given your risk profile"
    }}
  ],
  "portfolio_notes": "your overall thinking this cycle"
}}

DECISION FORMAT RULES:
- The `decisions` array is for executable accept/adjust entries and explicit reject records only.
- Valid actions: BUY or SHORT. No other actions belong in decisions[].
- Symbols in decisions[] MUST come from the ALLOWED ENTRY SYMBOLS block in the user prompt.
- Do NOT create trades from breaking news, strategy context, sector themes, baskets, or general market opinions unless that exact ticker is also in ALLOWED ENTRY SYMBOLS.
- HOLD, PASS, AVOID, WATCH, OVERWEIGHT, UNDERWEIGHT, CLOSE, SELL, TRIM
  are portfolio commentary — express them in `portfolio_notes` only.
- Every decision MUST include: symbol, action (BUY/SHORT), decision_type, quantity (positive integer),
  entry_price (positive number), stop_loss, target, setup_type, and rationale.
- decision_type MUST be one of: "accept", "adjust", or "reject".
- For decision_type "accept" or "adjust": include geometry_candidate_id and geometry_candidate_name from the scaffold candidate you selected.
- For decision_type "adjust": also include adjustment_rationale explaining what you changed and why.
- For decision_type "reject": this is a non-executable candidate rejection; geometry fields (entry_price, stop_loss, target, quantity) are NOT required. Include rationale explaining why no candidate was acceptable.
If no allowed entry symbol has a fully executable setup, return an empty decisions array.

STRICT OUTPUT CONTRACT — READ CAREFULLY

If any required field (quantity, entry_price, stop_loss, target) is unknown,
uncertain, conditional, placeholder, or null — DO NOT emit a decision.
Put that idea in portfolio_notes instead.

Forbidden inside decisions[]:
- null values, zero quantities, placeholder prices, textual targets
- conditional actions: BUY ON DIP, ACCUMULATE, OVERWEIGHT, UNDERWEIGHT
- non-entry actions: HOLD, PASS, AVOID, WATCH, CLOSE, SELL, TRIM
- sector concepts, baskets, unsupported symbols

When no executable order exists, return:
{{"decisions": [], "portfolio_notes": "..."}}

DECISION LOG BREVITY CONTRACT:
- `portfolio_notes` is an operator dashboard note, not chain-of-thought.
- Keep `portfolio_notes` <= 420 characters, max 3 terse clauses/sentences.
- Each decision `rationale` must be <= 280 characters.
- No markdown headings, numbered analysis, long narratives, or step-by-step deliberation.
- State only: action/pass, primary blocker or catalyst, and risk control.

Example — VALID (goes in decisions[]):
{{"symbol":"AMD","action":"BUY","decision_type":"accept","geometry_candidate_id":"amd_long_support_bounce_1","geometry_candidate_name":"support_bounce","quantity":10,"entry_price":161.0,"stop_loss":152.0,"target":179.0,"setup_type":"news_catalyst_breakout","rationale":"Strong momentum with clear levels"}}

Example — VALID non-executable candidate reject (audited, not traded):
{{"symbol":"AMD","action":"BUY","decision_type":"reject","rationale":"R:R below threshold, price extended beyond entry candidates"}}

Example — INVALID (belongs in portfolio_notes, NOT decisions[]):
{{"symbol":"AMD","action":"BUY","quantity":null,"target":"next resistance","rationale":"AMD looks attractive"}}
{recovery_probe_contract}"""


MAINTENANCE_REVIEW_PROMPT = """You are a portfolio manager performing a routine maintenance review of open positions.
Your job is to decide whether to HOLD, TIGHTEN STOP, RAISE TARGET, or TRIM a partial position.
You CANNOT close a position in this review. Closing requires a separate Reversal/Close Review triggered by thesis invalidation.

Profile: {profile_name} {emoji}

ENTRY CONTRACT (the original trade thesis — your anchor):
  Thesis: {thesis}
  Setup Type: {setup_type}
  Entry Price: ${entry_price}
  Stop Price: ${stop_price}
  Target Price: ${target_price}
  Invalidators: {invalidators}

CURRENT POSITION:
  Symbol: {symbol}
  Side: {side}
  Quantity: {quantity}
  Current Price: ${current_price}
  Unrealized P&L: {unrealized_pnl_pct}%
  DRIFTING: {drifting} (no recent analyst signal since entry)

CURRENT INDICATORS:
{indicators_text}

ADVISORY ANALYST SIGNALS (informational only — do NOT override the Entry Contract):
{advisory_signals_text}

POSITION HEALTH ASSESSMENT:
{health_text}

RULES:
1. The Entry Contract thesis is your anchor. Hold unless there is a clear reason to adjust.
2. You CANNOT produce a close action. Only Reversal/Close Review can close positions.
3. If the position is DRIFTING, that alone is NOT a reason to act. Evaluate against the Entry Contract.
4. Tighten stop only if the position has moved favorably and you want to lock in gains.
5. Raise target only if new evidence supports a higher target while the thesis remains intact.
6. Trim partial only if the position is significantly profitable and you want to reduce risk.
7. For LONG positions: `tighten_stop` is valid only if `new_stop > current stop`.
8. For SHORT positions: `tighten_stop` is valid only if `new_stop < current stop`.
9. If your proposed stop equals the current stop, use `hold` instead.
10. If you are uncertain about the stop value, use `hold`.
11. `tighten_stop` must never loosen risk controls; it must strictly tighten them.

Respond with JSON only:
{{
  "reviews": [
    {{
      "symbol": "{symbol}",
      "action": "hold|tighten_stop|raise_target|trim_partial",
      "new_stop": null,
      "new_target": null,
      "trim_pct": null,
      "reasoning": "why you chose this action, referencing the Entry Contract thesis"
    }}
  ],
  "notes": "overall maintenance review summary"
}}

VALID ACTIONS: hold, tighten_stop, raise_target, trim_partial
DO NOT use close, close_full, close_partial, or CLOSE.
"""

VALID_MAINTENANCE_ACTIONS = {"hold", "tighten_stop", "raise_target", "trim_partial"}


REVERSAL_CLOSE_PROMPT = """You are a portfolio manager performing a Reversal/Close Review.
A specific trigger has fired that may warrant closing this position. Your job is to evaluate
whether the original trade thesis is truly broken and decide whether to CLOSE FULL, CLOSE PARTIAL,
or HOLD with a tightened stop.

Profile: {profile_name} {emoji}

TRIGGER THAT CAUSED THIS REVIEW:
  Trigger Type: {trigger_type}
  Trigger Details: {trigger_details}

ENTRY CONTRACT (the original trade thesis — your anchor):
  Thesis: {thesis}
  Setup Type: {setup_type}
  Entry Price: ${entry_price}
  Stop Price: ${stop_price}
  Target Price: ${target_price}
  Invalidators: {invalidators}

CURRENT POSITION:
  Symbol: {symbol}
  Side: {side}
  Quantity: {quantity}
  Current Price: ${current_price}
  Unrealized P&L: {unrealized_pnl_pct}%

CURRENT MARKET CONDITIONS:
{market_conditions_text}

OPPOSING EVIDENCE:
{opposing_evidence_text}

RULES:
1. This review was triggered by: {trigger_type}. Evaluate whether the thesis is truly broken.
2. If the thesis is clearly invalidated (e.g., key level lost on confirmed close), close the position.
3. If the evidence is ambiguous, hold with a tightened stop to protect capital while giving the trade room.
4. close_partial is appropriate when some thesis elements are broken but others remain intact.
5. hold_tighten means: tighten stop to breakeven if profitable, otherwise hold current stop.

Respond with JSON only:
{{
  "symbol": "{symbol}",
  "action": "close_full|close_partial|hold_tighten",
  "reasoning": "why you chose this action, referencing the Entry Contract thesis and the trigger",
  "trigger": "{trigger_type}",
  "invalidator": {invalidator_json}
}}

VALID ACTIONS: close_full, close_partial, hold_tighten
"""

VALID_REVERSAL_ACTIONS = {"close_full", "close_partial", "hold_tighten"}


def run_maintenance_review(position_data: dict, profile: dict, tier: str = "high") -> dict:
    """
    Run a Maintenance Review for a single open position.

    Accepts position data (with Entry Contract, current price, indicators,
    advisory signals, health data, drifting state), formats the prompt,
    calls the LLM, validates the action, and returns the parsed review result.

    On LLM failure, defaults to "hold" (no action taken).

    Args:
        position_data: dict with keys:
            symbol, side, quantity, entry_price, stop_price, target_price,
            current_price, unrealized_pnl_pct, drifting,
            thesis, setup_type, invalidators,
            indicators, advisory_signals, health_text
        profile: PM profile dict from PM_PROFILES
        tier: LLM tier to use (default "high")

    Returns:
        dict with keys: symbol, action, new_stop, new_target, trim_pct, reasoning
    """
    symbol = position_data.get("symbol", "UNKNOWN")

    # Format invalidators for display
    invalidators_raw = position_data.get("invalidators")
    if isinstance(invalidators_raw, str):
        try:
            invalidators_list = json.loads(invalidators_raw)
        except (json.JSONDecodeError, TypeError):
            invalidators_list = []
    elif isinstance(invalidators_raw, list):
        invalidators_list = invalidators_raw
    else:
        invalidators_list = []
    invalidators_text = json.dumps(invalidators_list, indent=2) if invalidators_list else "None"

    # Format indicators
    indicators = position_data.get("indicators")
    if isinstance(indicators, dict):
        indicators_text = json.dumps(indicators, indent=2)
    elif isinstance(indicators, str):
        indicators_text = indicators
    else:
        indicators_text = "No indicator data available"

    # Format advisory signals
    advisory_signals = position_data.get("advisory_signals")
    if isinstance(advisory_signals, dict):
        advisory_signals_text = json.dumps(advisory_signals, indent=2)
    elif isinstance(advisory_signals, str):
        advisory_signals_text = advisory_signals
    else:
        advisory_signals_text = "No advisory signals available"

    # Format health text
    health_text = position_data.get("health_text")
    if not health_text:
        health_text = "No health assessment available"

    prompt = MAINTENANCE_REVIEW_PROMPT.format(
        profile_name=profile.get("name", "Unknown"),
        emoji=profile.get("emoji", ""),
        thesis=position_data.get("thesis") or "No thesis recorded",
        setup_type=position_data.get("setup_type") or "unknown",
        entry_price=position_data.get("entry_price") or "N/A",
        stop_price=position_data.get("stop_price") or "N/A",
        target_price=position_data.get("target_price") or "N/A",
        invalidators=invalidators_text,
        symbol=symbol,
        side=position_data.get("side") or "unknown",
        quantity=position_data.get("quantity") or 0,
        current_price=position_data.get("current_price") or "N/A",
        unrealized_pnl_pct=position_data.get("unrealized_pnl_pct") or 0,
        drifting="YES" if position_data.get("drifting") else "NO",
        indicators_text=indicators_text,
        advisory_signals_text=advisory_signals_text,
        health_text=health_text,
    )

    system_prompt = (
        "You are a portfolio manager performing routine maintenance reviews. "
        "Respond with valid JSON only. No markdown, no explanation outside the JSON."
    )

    # Default result on failure
    default_result = {
        "symbol": symbol,
        "action": "hold",
        "new_stop": None,
        "new_target": None,
        "trim_pct": None,
        "reasoning": "LLM review failed — defaulting to hold (no action taken)",
    }

    try:
        raw = call_llm(system_prompt, prompt, json_mode=True, tier=tier, purpose=f"pm_maintenance:{profile.get('name', 'unknown')}:{symbol}")
        result = parse_json_response(raw)
    except Exception as exc:
        log.error("Maintenance Review LLM call failed for %s: %s", symbol, exc)
        return default_result

    # Extract the review for this symbol from the reviews array
    reviews = result.get("reviews", [])
    review = None
    for r in reviews:
        if r.get("symbol") == symbol:
            review = r
            break
    # If no matching symbol found, use the first review or default
    if review is None:
        review = reviews[0] if reviews else {}

    # Validate the action
    action = review.get("action", "hold")
    if action not in VALID_MAINTENANCE_ACTIONS:
        log.warning(
            "Maintenance Review returned invalid action '%s' for %s. Defaulting to hold.",
            action, symbol,
        )
        action = "hold"

    return {
        "symbol": symbol,
        "action": action,
        "new_stop": review.get("new_stop"),
        "new_target": review.get("new_target"),
        "trim_pct": review.get("trim_pct"),
        "reasoning": review.get("reasoning") or "No reasoning provided",
    }


def run_reversal_close_review(position_data: dict, trigger_info: dict, profile: dict, tier: str = "high") -> dict:
    """
    Run a Reversal/Close Review for a single open position.

    Only invoked when a trigger fires (thesis_invalidation, opposing signal,
    explicit CLOSE). Evaluates the Entry Contract thesis against current
    market conditions and produces one of: close_full, close_partial,
    or hold_tighten.

    On LLM failure, defaults to "hold_tighten" (tighten stop to breakeven
    if profitable, otherwise hold).

    Args:
        position_data: dict with keys:
            symbol, side, quantity, entry_price, stop_price, target_price,
            current_price, unrealized_pnl_pct,
            thesis, setup_type, invalidators,
            market_conditions, opposing_evidence
        trigger_info: dict with keys:
            type: str — one of "thesis_invalidation", "opposing_signal", "explicit_close"
            details: str — human-readable description of the trigger
            invalidator: dict | None — the specific invalidator that was breached (if applicable)
        profile: PM profile dict from PM_PROFILES
        tier: LLM tier to use (default "high")

    Returns:
        dict with keys: symbol, action, reasoning, trigger, invalidator
    """
    symbol = position_data.get("symbol", "UNKNOWN")
    trigger_type = trigger_info.get("type", "unknown")
    trigger_details = trigger_info.get("details", "No details available")
    trigger_invalidator = trigger_info.get("invalidator")

    # Log the specific trigger that caused this review
    log.info(
        "Reversal/Close Review triggered for %s: trigger=%s, details=%s",
        symbol, trigger_type, trigger_details,
    )

    # Format invalidators for display
    invalidators_raw = position_data.get("invalidators")
    if isinstance(invalidators_raw, str):
        try:
            invalidators_list = json.loads(invalidators_raw)
        except (json.JSONDecodeError, TypeError):
            invalidators_list = []
    elif isinstance(invalidators_raw, list):
        invalidators_list = invalidators_raw
    else:
        invalidators_list = []
    invalidators_text = json.dumps(invalidators_list, indent=2) if invalidators_list else "None"

    # Format market conditions
    market_conditions = position_data.get("market_conditions")
    if isinstance(market_conditions, dict):
        market_conditions_text = json.dumps(market_conditions, indent=2)
    elif isinstance(market_conditions, str):
        market_conditions_text = market_conditions
    else:
        market_conditions_text = "No market condition data available"

    # Format opposing evidence
    opposing_evidence = position_data.get("opposing_evidence")
    if isinstance(opposing_evidence, dict):
        opposing_evidence_text = json.dumps(opposing_evidence, indent=2)
    elif isinstance(opposing_evidence, str):
        opposing_evidence_text = opposing_evidence
    else:
        opposing_evidence_text = "No opposing evidence available"

    # Format the trigger invalidator for the prompt
    if trigger_invalidator and isinstance(trigger_invalidator, dict):
        invalidator_json = json.dumps(trigger_invalidator)
    else:
        invalidator_json = "null"

    prompt = REVERSAL_CLOSE_PROMPT.format(
        profile_name=profile.get("name", "Unknown"),
        emoji=profile.get("emoji", ""),
        trigger_type=trigger_type,
        trigger_details=trigger_details,
        thesis=position_data.get("thesis") or "No thesis recorded",
        setup_type=position_data.get("setup_type") or "unknown",
        entry_price=position_data.get("entry_price") or "N/A",
        stop_price=position_data.get("stop_price") or "N/A",
        target_price=position_data.get("target_price") or "N/A",
        invalidators=invalidators_text,
        symbol=symbol,
        side=position_data.get("side") or "unknown",
        quantity=position_data.get("quantity") or 0,
        current_price=position_data.get("current_price") or "N/A",
        unrealized_pnl_pct=position_data.get("unrealized_pnl_pct") or 0,
        market_conditions_text=market_conditions_text,
        opposing_evidence_text=opposing_evidence_text,
        invalidator_json=invalidator_json,
    )

    system_prompt = (
        "You are a portfolio manager performing a Reversal/Close Review. "
        "Respond with valid JSON only. No markdown, no explanation outside the JSON."
    )

    # Default result on failure — hold_tighten is conservative
    default_result = {
        "symbol": symbol,
        "action": "hold_tighten",
        "reasoning": "LLM review failed — defaulting to hold_tighten (tighten stop to breakeven if profitable)",
        "trigger": trigger_type,
        "invalidator": trigger_invalidator,
    }

    try:
        raw = call_llm(system_prompt, prompt, json_mode=True, tier=tier, purpose=f"pm_reversal:{profile.get('name', 'unknown')}:{symbol}")
        result = parse_json_response(raw)
    except Exception as exc:
        log.error("Reversal/Close Review LLM call failed for %s: %s", symbol, exc)
        return default_result

    # Validate the action
    action = result.get("action", "hold_tighten")
    if action not in VALID_REVERSAL_ACTIONS:
        log.warning(
            "Reversal/Close Review returned invalid action '%s' for %s. Defaulting to hold_tighten.",
            action, symbol,
        )
        action = "hold_tighten"

    return {
        "symbol": symbol,
        "action": action,
        "reasoning": result.get("reasoning") or "No reasoning provided",
        "trigger": result.get("trigger") or trigger_type,
        "invalidator": result.get("invalidator") or trigger_invalidator,
    }


def get_portfolio_for_profile(db, fh, profile_id: str) -> dict:
    """Build portfolio snapshot for a specific profile."""
    positions = db.query(Position).filter_by(profile=profile_id).all()
    pos_data = []
    total_pos_value = 0.0

    for p in positions:
        try:
            quote = fh.get_quote(p.symbol)
            price = quote["price"]
        except Exception:
            price = p.avg_cost
        market_value = p.quantity * price
        if p.side == "short":
            unrealized_pnl = (p.avg_cost - price) * p.quantity
        else:
            unrealized_pnl = (price - p.avg_cost) * p.quantity
        total_pos_value += market_value

        # Get stop/target from the open trade
        open_trade = (
            db.query(Trade)
            .filter_by(symbol=p.symbol, profile=profile_id, status="open")
            .order_by(Trade.entry_time.desc())
            .first()
        )

        # Detect DRIFTING state for this position
        drifting = detect_drifting(db, open_trade) if open_trade else True

        pos_data.append({
            "symbol": p.symbol,
            "side": p.side,
            "quantity": p.quantity,
            "avg_cost": p.avg_cost,
            "current_price": price,
            "market_value": round(market_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pnl_pct": round(unrealized_pnl / (p.avg_cost * p.quantity) * 100, 2) if p.avg_cost and p.quantity else 0,
            "stop_price": open_trade.stop_price if open_trade else None,
            "target_price": open_trade.target_price if open_trade else None,
            "entry_time": open_trade.entry_time.isoformat() if open_trade and open_trade.entry_time else None,
            "drifting": drifting,
        })

    bal = (
        db.query(Balance)
        .filter_by(profile=profile_id)
        .order_by(Balance.timestamp.desc())
        .first()
    )
    starting = PM_PROFILES[profile_id]["starting_balance"]
    cash = bal.cash if bal else float(starting)
    total_equity = cash + total_pos_value

    # Today's realized P&L
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
    today_trades = (
        db.query(Trade)
        .filter_by(profile=profile_id, status="closed")
        .filter(Trade.exit_time >= today_start)
        .all()
    )
    daily_pnl = sum(t.pnl or 0 for t in today_trades)

    return {
        "profile": profile_id,
        "cash": round(cash, 2),
        "positions": pos_data,
        "total_equity": round(total_equity, 2),
        "position_count": len(pos_data),
        "daily_pnl": round(daily_pnl, 2),
        "daily_pnl_pct": round(daily_pnl / starting * 100, 2),
        "starting_balance": starting,
    }


def _count_recent_consecutive_losses(db, profile_id: str) -> int:
    """Count consecutive recent losing trades for a profile (most recent first)."""
    recent_trades = (
        db.query(Trade)
        .filter_by(profile=profile_id, status="closed")
        .order_by(Trade.exit_time.desc())
        .limit(20)
        .all()
    )
    count = 0
    for t in recent_trades:
        if t.pnl is not None and t.pnl < 0:
            count += 1
        else:
            break
    return count


def _build_signal_for_symbol(db, symbol: str, decision: dict) -> dict:
    """Build a signal dict for the similarity/edge score engines from analyst memory."""
    sig_mem = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol=symbol, key="signal")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if sig_mem:
        try:
            return json.loads(sig_mem.value)
        except Exception:
            pass
    # Fallback: build minimal signal from decision fields
    return {
        "setup_type": decision.get("setup_type") or decision.get("setup") or "",
        "market_regime": decision.get("market_regime") or decision.get("regime") or "",
        "strength": decision.get("strength") or "moderate",
        "confidence": decision.get("confidence") or "medium",
        "bias": "LONG" if decision.get("action") == "BUY" else "SHORT",
        "indicators": {},
    }


def _build_case_stats(db, setup_type: str, market_regime: str = None) -> dict:
    """Build case_stats dict from the case library for edge score computation."""
    from utils.trade_validator import adjust_confidence
    conf_result = adjust_confidence(db.bind, setup_type, market_regime)
    return {
        "win_rate": conf_result.get("win_rate") or 0.0,
        "sample_size": conf_result.get("total_cases", 0),
    }


# Strength ordering for opposing evidence threshold comparison (Req 7.5)
# Higher value = stronger signal. A signal "meets" a threshold when its
# numeric strength >= the threshold's numeric strength.
STRENGTH_ORDER = {"weak": 1, "moderate": 2, "strong": 3}
RISK_GATE_SIGNAL_STRENGTH = {"weak": 3.0, "moderate": 6.0, "strong": 10.0}


def _meets_threshold(signal_strength: str, threshold: str) -> bool:
    """Return True if signal_strength meets or exceeds the opposing_evidence_threshold."""
    sig_val = STRENGTH_ORDER.get(str(signal_strength).lower(), 0)
    thr_val = STRENGTH_ORDER.get(str(threshold).lower(), 0)
    return sig_val >= thr_val


def summarize_entry_signal_filter(
    signals: dict[str, dict],
    held_symbols: set[str],
    min_signal_strength: str,
) -> dict:
    """Explain why PM entry filtering produced zero or few candidates."""
    totals = {
        "total": len(signals),
        "held": 0,
        "eligible": 0,
        "hold": 0,
        "below_threshold": 0,
        "direction_counts": {},
        "strength_counts": {},
        "confidence_counts": {},
        "setup_counts": {},
        "sanity_conflicts": [],
        "veto_required": 0,
        "veto_present": 0,
        "veto_missing": 0,
        "min_signal_strength": min_signal_strength,
    }

    for sym, sig in signals.items():
        if sym in held_symbols:
            totals["held"] += 1
            continue

        direction = str(sig.get("signal", "") or "").upper() or "UNKNOWN"
        strength = str(sig.get("strength", "weak") or "weak").lower()
        confidence = str(sig.get("confidence", "unknown") or "unknown").lower()
        setup_type = str(sig.get("setup_type", "unknown") or "unknown")

        totals["direction_counts"][direction] = totals["direction_counts"].get(direction, 0) + 1
        totals["strength_counts"][strength] = totals["strength_counts"].get(strength, 0) + 1
        totals["confidence_counts"][confidence] = totals["confidence_counts"].get(confidence, 0) + 1
        totals["setup_counts"][setup_type] = totals["setup_counts"].get(setup_type, 0) + 1

        sanity = sig.get("deterministic_sanity")
        if sig.get("llm_veto_required"):
            totals["veto_required"] += 1
        if sig.get("llm_veto_present"):
            totals["veto_present"] += 1
        if sig.get("llm_veto_missing"):
            totals["veto_missing"] += 1

        if isinstance(sanity, dict) and sanity.get("conflict"):
            totals["sanity_conflicts"].append({
                "symbol": sym,
                "llm_signal": sanity.get("llm_signal"),
                "deterministic_bias": sanity.get("bias"),
                "score": sanity.get("score"),
                "veto_required": bool(sig.get("llm_veto_required")),
                "veto_present": bool(sig.get("llm_veto_present")),
                "veto_missing": bool(sig.get("llm_veto_missing")),
                "veto_reason": str(sig.get("llm_veto_reason") or "")[:160],
                "reasons": sanity.get("reasons", [])[:6],
            })

        if direction == "HOLD":
            totals["hold"] += 1
        elif not _meets_threshold(strength, min_signal_strength):
            totals["below_threshold"] += 1
        else:
            totals["eligible"] += 1

    return totals


def format_entry_signal_filter_summary(summary: dict) -> str:
    """Compact one-line PM skip reason for logs and cycle notes."""
    parts = [
        f"total={summary.get('total', 0)}",
        f"eligible={summary.get('eligible', 0)}",
        f"held={summary.get('held', 0)}",
        f"hold={summary.get('hold', 0)}",
        f"below_threshold={summary.get('below_threshold', 0)}",
        f"min_strength={summary.get('min_signal_strength')}",
        f"directions={summary.get('direction_counts', {})}",
        f"strengths={summary.get('strength_counts', {})}",
        f"confidences={summary.get('confidence_counts', {})}",
        f"setups={summary.get('setup_counts', {})}",
        f"veto_required={summary.get('veto_required', 0)}",
        f"veto_present={summary.get('veto_present', 0)}",
        f"veto_missing={summary.get('veto_missing', 0)}",
    ]
    conflicts = summary.get("sanity_conflicts") or []
    if conflicts:
        parts.append(f"sanity_conflicts={conflicts[:5]}")
    return "; ".join(parts)


def _check_reversal_triggers(
    db, trade, position_data: dict, signal: dict | None, profile: dict
) -> dict | None:
    """
    Check whether a position has any Reversal/Close Review triggers.

    Returns a trigger_info dict if a trigger is found, or None if the
    position should go to Maintenance Review.

    Trigger types:
      - thesis_invalidation: Price Monitor detected an invalidator breach
      - opposing_signal: Analyst signal contradicts Entry Contract direction
        and meets the profile's opposing_evidence_threshold
      - explicit_close: Analyst signal contains an explicit CLOSE action
    """
    symbol = trade.symbol

    # 1. Check AgentMemory for thesis_invalidation triggers from Price Monitor
    invalidation_mem = (
        db.query(AgentMemory)
        .filter_by(agent="price_monitor", symbol=symbol, key="thesis_invalidation")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if invalidation_mem:
        try:
            inv_data = json.loads(invalidation_mem.value)
        except (json.JSONDecodeError, TypeError):
            inv_data = {}
        return {
            "type": "thesis_invalidation",
            "details": (
                f"Price Monitor detected thesis invalidation for {symbol}: "
                f"{inv_data.get('invalidator', 'unknown invalidator')}"
            ),
            "invalidator": inv_data.get("invalidator"),
        }

    if not signal:
        return None

    # 2. Check for explicit CLOSE signal
    signal_action = str(signal.get("signal", "") or signal.get("action", "")).upper()
    if signal_action == "CLOSE":
        return {
            "type": "explicit_close",
            "details": f"Analyst issued explicit CLOSE signal for {symbol}",
            "invalidator": None,
        }

    # 3. Check for opposing signal that meets the profile threshold
    signal_bias = str(signal.get("bias", "")).upper()
    trade_direction = str(trade.direction).upper()  # LONG or SHORT

    # Determine if the signal opposes the position direction
    is_opposing = (
        (trade_direction == "LONG" and signal_bias == "SHORT")
        or (trade_direction == "SHORT" and signal_bias == "LONG")
    )

    if is_opposing:
        signal_strength = str(signal.get("strength", "")).lower()
        threshold = profile.get("opposing_evidence_threshold", "strong")
        if _meets_threshold(signal_strength, threshold):
            return {
                "type": "opposing_signal",
                "details": (
                    f"Opposing {signal_bias} signal (strength={signal_strength}) "
                    f"for {symbol} meets {threshold} threshold"
                ),
                "invalidator": None,
            }

    return None


VALID_INVALIDATOR_TYPES = {"price_below_level", "price_above_level", "structure_break"}
VALID_CONFIRMATION_METHODS = {"tick", "5m_close"}


def _parse_invalidator(raw: dict) -> dict | None:
    """Parse and validate a single invalidator dict. Returns None if invalid."""
    if not isinstance(raw, dict):
        return None
    inv_type = raw.get("type", "")
    reference = raw.get("reference", "")
    confirmation = raw.get("confirmation", "5m_close")
    lookback_bars = raw.get("lookback_bars", 1)

    # Validate type
    if inv_type not in VALID_INVALIDATOR_TYPES:
        return None
    # Reference must be a non-empty string
    if not reference:
        return None
    reference = str(reference)
    # Validate confirmation
    if confirmation not in VALID_CONFIRMATION_METHODS:
        confirmation = "5m_close"
    # Validate lookback_bars
    try:
        lookback_bars = int(lookback_bars)
        if lookback_bars < 0:
            lookback_bars = 0
    except (TypeError, ValueError):
        lookback_bars = 1

    return {
        "type": inv_type,
        "reference": reference,
        "confirmation": confirmation,
        "lookback_bars": lookback_bars,
    }


def _default_invalidator(stop: float) -> dict:
    """Build a default stop-price-based invalidator."""
    return {
        "type": "price_below_level",
        "reference": str(stop),
        "confirmation": "5m_close",
        "lookback_bars": 1,
    }


def build_entry_contract(decision: dict, signal: dict, stop: float, target: float) -> dict:
    """
    Build an Entry Contract from a trade decision and analyst signal.

    Extracts thesis, setup_type, and structured invalidators.
    Falls back to a stop-price-based default invalidator when the signal
    lacks an invalidation field.

    Returns:
        {"thesis": str, "setup_type": str, "invalidators": list[dict]}
    """
    # --- Thesis ---
    rationale = decision.get("rationale") or ""
    signal_context_parts = []
    if signal.get("bias"):
        signal_context_parts.append(f"Bias: {signal['bias']}")
    if signal.get("confidence"):
        signal_context_parts.append(f"Confidence: {signal['confidence']}")
    if signal.get("setup_type") or signal.get("setup"):
        signal_context_parts.append(
            f"Setup: {signal.get('setup_type') or signal.get('setup')}"
        )
    if signal.get("key_levels"):
        signal_context_parts.append(f"Key levels: {signal['key_levels']}")

    signal_context = "; ".join(signal_context_parts)
    if rationale and signal_context:
        thesis = f"{rationale} [Signal context: {signal_context}]"
    elif rationale:
        thesis = rationale
    elif signal_context:
        thesis = f"[Signal context: {signal_context}]"
    else:
        thesis = "No thesis recorded"

    # --- Setup type ---
    setup_type = (
        signal.get("setup_type")
        or signal.get("setup")
        or decision.get("setup_type")
        or decision.get("setup")
        or "unknown"
    )

    # --- Invalidators ---
    invalidators = []
    invalidation_raw = signal.get("invalidation")

    if invalidation_raw:
        # Parse invalidation field — could be a list of dicts, a single dict,
        # or a string description
        parsed_any = False
        if isinstance(invalidation_raw, list):
            for item in invalidation_raw:
                inv = _parse_invalidator(item)
                if inv:
                    invalidators.append(inv)
                    parsed_any = True
        elif isinstance(invalidation_raw, dict):
            inv = _parse_invalidator(invalidation_raw)
            if inv:
                invalidators.append(inv)
                parsed_any = True
        # If invalidation was present but we couldn't parse any structured
        # invalidators from it, fall back to default
        if not parsed_any:
            log.warning(
                "Could not parse structured invalidators from signal invalidation "
                "field (value: %s). Falling back to stop-price default.",
                invalidation_raw,
            )
            invalidators.append(_default_invalidator(stop))
    else:
        # No invalidation field at all — use stop-price default
        log.warning(
            "Signal lacks invalidation field. Using stop-price default "
            "invalidator (stop=%.2f).",
            stop,
        )
        invalidators.append(_default_invalidator(stop))

    return {
        "thesis": thesis,
        "setup_type": setup_type,
        "invalidators": invalidators,
    }


def build_legacy_entry_contract(trade) -> dict | None:
    """
    Build a best-effort Entry Contract for a legacy trade that was opened
    before the thesis-anchored exits feature.

    Migration rules:
      - If trade.thesis is already populated → return None (no migration needed)
      - If trade.stop_price and trade.target_price exist → full Entry Contract
      - If only trade.stop_price exists → partial contract with stop-based invalidator
      - If neither exists → return None (fall back to signal-based evaluation)

    Logs a warning for each legacy trade migrated, identifying the trade
    and which fields were missing or inferred.

    Returns:
        dict with keys {"thesis", "setup_type", "invalidators"} or None
    """
    # Already has a thesis — no migration needed
    if trade.thesis:
        return None

    has_stop = trade.stop_price is not None
    has_target = trade.target_price is not None

    # Neither stop nor target — cannot construct a meaningful contract
    if not has_stop and not has_target:
        return None

    # Build thesis from reason_entry or use a default
    thesis = trade.reason_entry or "Legacy trade — no thesis recorded"
    setup_type = "unknown"

    # Build invalidator from stop price
    invalidators = []
    if has_stop:
        invalidators.append(_default_invalidator(trade.stop_price))

    # Determine what was missing/inferred for the log message
    missing_parts = []
    if not trade.reason_entry:
        missing_parts.append("reason_entry (used default thesis)")
    if not has_target:
        missing_parts.append("target_price")
    missing_parts.append("setup_type (inferred as 'unknown')")

    trade_id = getattr(trade, "id", "?")
    symbol = getattr(trade, "symbol", "?")
    log.warning(
        "Legacy trade migration: trade_id=%s symbol=%s — "
        "constructed %s Entry Contract. Missing/inferred: %s",
        trade_id,
        symbol,
        "full" if (has_stop and has_target) else "partial",
        ", ".join(missing_parts),
    )

    return {
        "thesis": thesis,
        "setup_type": setup_type,
        "invalidators": invalidators,
    }


def detect_drifting(db, trade) -> bool:
    """
    Detect whether a position is in DRIFTING state.

    A position is DRIFTING when no analyst signal for the trade's symbol
    has been recorded after the trade's entry_time. This is a computed
    state — not stored in the DB — to avoid stale state if signals arrive
    between cycles.

    Args:
        db: SQLAlchemy session
        trade: Trade record with .symbol and .entry_time

    Returns:
        True if no analyst signal exists after entry_time (drifting),
        False if a signal exists after entry_time (not drifting).
    """
    if not trade.entry_time:
        # No entry time recorded — treat as drifting (conservative)
        return True

    latest_signal = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol=trade.symbol, key="signal")
        .filter(AgentMemory.timestamp > trade.entry_time)
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )

    return latest_signal is None


def get_strategy_position_multiplier(engine, strategy_key: str) -> float:
    """Return position size multiplier based on pipeline stage.

    Queries the DynamicStrategy table for the given key.  Hardcoded strategies
    (not present in the table) always get a 1.0 multiplier.

    Returns:
        0.5 for live_50 strategies
        1.0 for live_100 strategies (and hardcoded)
        0.0 for strategies not yet in live stages (should not be called)
    """
    db = get_session(engine)
    try:
        strategy = db.query(DynamicStrategy).filter_by(key=strategy_key).first()
        if strategy is None:
            # Not a dynamic strategy — treat as hardcoded → full allocation
            return 1.0
        if strategy.status == "live_50":
            return 0.5
        if strategy.status == "live_100":
            return 1.0
        # Strategy exists but is not in a live stage
        return 0.0
    finally:
        db.close()


def _compute_max_dollar_risk(profile_id: str, total_equity: float) -> float:
    """Compute the maximum dollar risk per trade from the active profile.

    Uses the profile's risk_per_trade_pct (or equivalent field) applied to
    total_equity. This is the single authoritative source for max dollar risk
    used by both initial PM sizing and the RiskGeometryGate.

    Falls back to a conservative default (1% of equity) if the profile field is missing.
    """
    profile = PM_PROFILES.get(profile_id, {})
    risk_pct = profile.get("risk_per_trade_pct", 0.01)
    return total_equity * risk_pct


def _run_gate_pipeline(db, engine, decision, signal, profile_id):
    """
    Run all active gates in sequence. Returns (proceed, notes, cumulative_multiplier, multiplier_breakdown).

    Gate order:
    1. Setup Quality Gate (Phase 1)
    2. Pre-Trade Quality Gate (Phase 1)
    2.5. Catalyst Specificity Gate (Phase 1)
    3. Risk Geometry Gate (Phase 1)
    4. Catalyst Timing Gate (Phase 2 — skipped until wired)
    5. Concentration Gate (Phase 2 — skipped until wired)

    Short-circuits on reject or override_required. Accumulates size multipliers on reduce_size.
    Each gate call is wrapped in try/except — on unexpected exception, log gate_error event and treat as warn.
    """
    notes = []
    cumulative_multiplier = 1.0
    multiplier_breakdown = []

    # Gate 1: Setup Quality
    # PM decisions often omit setup metadata; fall back to the latest Analyst
    # signal so the gate evaluates the actual setup instead of treating it as
    # empty/insufficient-data.
    signal = signal or {}
    setup_type = resolve_setup_type(decision, signal)
    if setup_type is None:
        notes.append({"gate": "setup_type_resolution", "decision": "reject", "reason": "missing setup_type"})
        return False, notes, cumulative_multiplier, multiplier_breakdown

    market_regime = (
        decision.get("market_regime")
        or decision.get("regime")
        or signal.get("market_regime")
        or signal.get("regime")
    )
    symbol = decision.get("symbol") or signal.get("symbol")

    # Extract override_confidence_score for moderate high-confirmation check
    confidence_score = None
    raw_conf = decision.get("override_confidence_score")
    if raw_conf is not None:
        try:
            score = float(raw_conf)
            if math.isfinite(score) and 0.0 <= score <= 10.0:
                confidence_score = score
        except (TypeError, ValueError):
            confidence_score = None

    # Extract near-miss pilot confirming signal fields from decision/signal
    catalyst_type = decision.get("catalyst_type") or (
        signal.get("catalyst_type") if signal else None
    )
    price_above_vwap = decision.get("price_above_vwap")
    if price_above_vwap is None and signal:
        price_above_vwap = signal.get("price_above_vwap")
    volume_ratio = None
    raw_vr = decision.get("volume_ratio") or (
        signal.get("volume_ratio") if signal else None
    )
    if raw_vr is not None:
        try:
            volume_ratio = float(raw_vr)
        except (TypeError, ValueError):
            volume_ratio = None

    try:
        setup_result = evaluate_setup_quality(
            engine, db, setup_type, market_regime,
            symbol=symbol, profile=profile_id,
            confidence_score=confidence_score,
            catalyst_type=catalyst_type,
            price_above_vwap=price_above_vwap,
            volume_ratio=volume_ratio,
        )
        notes.append({"gate": "setup_quality_gate", **setup_result})
        if setup_result["decision"] == "reject":
            return False, notes, cumulative_multiplier, multiplier_breakdown
        if setup_result.get("size_multiplier") and setup_result["decision"] == "reduce_size":
            sqg_multiplier = setup_result["size_multiplier"]
            # Write pilot counterfactual row if this is a near-miss pilot override
            if setup_result.get("reason_type") == "near_miss_pilot_override":
                try:
                    write_pilot_counterfactual_row(
                        db,
                        symbol=symbol,
                        action=decision.get("action", ""),
                        blocked_by="setup_quality_gate",
                        block_reason="historical_underperformance",
                        direction=decision.get("direction"),
                        profile=profile_id,
                        setup_type=setup_type,
                        entry_price=decision.get("price") or decision.get("entry_price"),
                        stop_price=decision.get("stop") or decision.get("stop_price"),
                        target_price=decision.get("target") or decision.get("target_price"),
                        quantity=_coerce_quantity(decision.get("quantity", 0), symbol=symbol),
                        gate_result=setup_result,
                        gate_notes=notes,
                        signal_snapshot=signal,
                        agent=f"pm_{profile_id}",
                    )
                except Exception as exc:
                    log.warning("Failed to write pilot counterfactual row for setup_quality_gate: %s", exc)
            # Update working quantity so downstream gates see reduced size
            original_qty = _coerce_quantity(decision.get("quantity", 0), symbol=symbol)
            reduced_qty = max(1, int(original_qty * sqg_multiplier))
            log.info(
                "SETUP QUALITY GATE: %s reduce_size — qty %d → %d "
                "(multiplier=%.2f, reason_type=%s)",
                symbol, original_qty, reduced_qty,
                sqg_multiplier, setup_result.get("reason_type", ""),
            )
            decision["quantity"] = reduced_qty
            cumulative_multiplier *= sqg_multiplier
            multiplier_breakdown.append({"gate": "setup_quality_gate", "multiplier": sqg_multiplier})
    except Exception as exc:
        log.error("Setup quality gate error (treating as warn): %s", exc)
        log_trade_event(
            db, "gate_error",
            agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id,
            message=f"Setup quality gate error: {exc}",
            payload={"gate_name": "setup_quality_gate", "error": str(exc)},
        )
        notes.append({"gate": "setup_quality_gate", "decision": "warn", "reason_type": "gate_error", "reason": str(exc)})

    # Gate 2: Pre-Trade Quality
    try:
        quality_result = evaluate_pre_trade_quality(
            db, decision, signal,
            symbol=symbol, profile=profile_id,
        )
        notes.append({"gate": "pre_trade_quality_gate", **quality_result})
        if quality_result["decision"] == "reject":
            return False, notes, cumulative_multiplier, multiplier_breakdown
        if quality_result["decision"] == "override_required":
            return False, notes, cumulative_multiplier, multiplier_breakdown
        if quality_result.get("size_multiplier") and quality_result["decision"] == "reduce_size":
            ptq_multiplier = quality_result["size_multiplier"]
            # Update working quantity so downstream gates see reduced size
            original_qty = _coerce_quantity(decision.get("quantity", 0), symbol=symbol)
            reduced_qty = max(1, int(original_qty * ptq_multiplier))
            log.info(
                "PRE-TRADE QUALITY GATE: %s reduce_size — qty %d → %d "
                "(multiplier=%.2f, reason_type=%s)",
                symbol, original_qty, reduced_qty,
                ptq_multiplier, quality_result.get("reason_type", ""),
            )
            decision["quantity"] = reduced_qty
            cumulative_multiplier *= ptq_multiplier
            multiplier_breakdown.append({"gate": "pre_trade_quality_gate", "multiplier": ptq_multiplier})
    except Exception as exc:
        log.error("Pre-trade quality gate error (treating as warn): %s", exc)
        log_trade_event(
            db, "gate_error",
            agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id,
            message=f"Pre-trade quality gate error: {exc}",
            payload={"gate_name": "pre_trade_quality_gate", "error": str(exc)},
        )
        notes.append({"gate": "pre_trade_quality_gate", "decision": "warn", "reason_type": "gate_error", "reason": str(exc)})

    # Gate 2.5: Catalyst Specificity Gate
    try:
        catalyst_result = evaluate_catalyst_specificity(
            decision=decision,
            signal=signal,
            profile=profile_id,
            db=db,
        )
        notes.append({"gate": "catalyst_specificity_gate", **catalyst_result})

        catalyst_decision = catalyst_result.get("decision", "allow")

        if catalyst_decision == "block":
            return False, notes, cumulative_multiplier, multiplier_breakdown

        if catalyst_decision == "reduce_size":
            cat_multiplier = catalyst_result.get("size_multiplier", 1.0)
            if cat_multiplier < 1.0:
                # Update decision["quantity"] directly so risk geometry evaluates reduced size
                original_qty = _coerce_quantity(decision.get("quantity", 0), symbol=symbol)
                reduced_qty = max(1, int(original_qty * cat_multiplier))
                log.info(
                    "CATALYST SPECIFICITY GATE: %s reduce_size — qty %d → %d "
                    "(multiplier=%.2f, reason_type=%s, score=%d)",
                    symbol, original_qty, reduced_qty,
                    cat_multiplier, catalyst_result.get("reason_type", ""),
                    catalyst_result.get("score", 0),
                )
                decision["quantity"] = reduced_qty
                cumulative_multiplier *= cat_multiplier
                multiplier_breakdown.append({"gate": "catalyst_specificity_gate", "multiplier": cat_multiplier})

        elif catalyst_decision == "warn":
            log.info(
                "CATALYST SPECIFICITY GATE: %s warn — score=%d, reason_type=%s, reason=%s",
                symbol, catalyst_result.get("score", 0),
                catalyst_result.get("reason_type", ""),
                catalyst_result.get("reason", ""),
            )

    except Exception as exc:
        log.error("Catalyst specificity gate error (treating as warn): %s", exc)
        log_trade_event(
            db, "gate_error",
            agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id,
            message=f"Catalyst specificity gate error: {exc}",
            payload={"gate_name": "catalyst_specificity_gate", "error": str(exc)},
        )
        notes.append({"gate": "catalyst_specificity_gate", "decision": "warn", "reason_type": "gate_error", "reason": str(exc)})

    # Gate 3: Risk Geometry Gate
    try:
        from utils.risk_geometry_gate import evaluate_risk_geometry
        from utils.atr_helper import compute_intraday_atr

        # Extract trade parameters from decision
        price = decision.get("price") or decision.get("entry_price") or 0
        try:
            price = float(price) if price not in (None, "") else 0
        except (TypeError, ValueError):
            price = 0

        stop = decision.get("stop") or decision.get("stop_price") or decision.get("stop_loss")
        target = decision.get("target") or decision.get("target_price") or decision.get("profit_target")
        quantity = _coerce_quantity(decision.get("quantity", 0), symbol=symbol)
        action = decision.get("action", "")

        # Only run the gate if we have the minimum required parameters
        if price and price > 0 and stop and target:
            try:
                stop = float(stop)
                target = float(target)
            except (TypeError, ValueError):
                stop = 0
                target = 0

            if stop > 0 and target > 0:
                # Compute ATR for the symbol
                atr_data = compute_intraday_atr(symbol)

                # Compute total_equity for max_dollar_risk
                starting = PM_PROFILES[profile_id]["starting_balance"]
                bal = (
                    db.query(Balance)
                    .filter_by(profile=profile_id)
                    .order_by(Balance.timestamp.desc())
                    .first()
                )
                cash = bal.cash if bal else float(starting)
                positions = db.query(Position).filter_by(profile=profile_id).all()
                pos_value = sum(p.quantity * p.avg_cost for p in positions)
                total_equity = cash + pos_value

                max_dollar_risk = _compute_max_dollar_risk(profile_id, total_equity)

                geometry_result = evaluate_risk_geometry(
                    entry_price=price,
                    stop_price=stop,
                    target_price=target,
                    quantity=quantity,
                    direction=action,
                    symbol=symbol,
                    setup_type=setup_type,
                    atr_5min=atr_data.get("atr"),
                    atr_timestamp=atr_data.get("timestamp"),
                    atr_source=atr_data.get("source"),
                    trade_timestamp=datetime.now(timezone.utc),
                    max_dollar_risk=max_dollar_risk,
                    quantity_policy="whole_share",
                    db=db,
                    profile=profile_id,
                    signal_strength=RISK_GATE_SIGNAL_STRENGTH.get(
                        str(signal.get("strength", "")).lower()
                    ),
                    confidence_level=signal.get("confidence"),
                )

                notes.append({"gate": "risk_geometry_gate", **geometry_result})

                if geometry_result["decision"] == "rejected":
                    return False, notes, cumulative_multiplier, multiplier_breakdown

                if geometry_result["decision"] == "adjusted_allowed":
                    # Write pilot counterfactual row if this is a pilot R:R override
                    if geometry_result.get("pilot_override"):
                        try:
                            write_pilot_counterfactual_row(
                                db,
                                symbol=symbol,
                                action=decision.get("action", ""),
                                blocked_by="risk_geometry_gate",
                                block_reason="rr_degradation",
                                direction=decision.get("direction"),
                                profile=profile_id,
                                setup_type=setup_type,
                                entry_price=price,
                                stop_price=stop,
                                target_price=target,
                                quantity=quantity,
                                gate_result=geometry_result,
                                gate_notes=notes,
                                signal_snapshot=signal,
                                agent=f"pm_{profile_id}",
                            )
                        except Exception as exc:
                            log.warning("Failed to write pilot counterfactual row for risk_geometry_gate: %s", exc)
                    # Propagate adjusted params for downstream use
                    decision["stop"] = geometry_result["stop_price"]
                    decision["stop_loss"] = geometry_result["stop_price"]
                    decision["stop_price"] = geometry_result["stop_price"]
                    decision["quantity"] = geometry_result["quantity"]

    except Exception as exc:
        log.error("Risk geometry gate error (treating as warn): %s", exc)
        log_trade_event(
            db, "gate_error",
            agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id,
            message=f"Risk geometry gate error: {exc}",
            payload={"gate_name": "risk_geometry_gate", "risk_geometry_gate_failed_open": True, "error": str(exc)},
        )
        notes.append({"gate": "risk_geometry_gate", "decision": "warn", "reason_type": "gate_error", "risk_geometry_gate_failed_open": True, "reason": str(exc)})

    # Phase 2 gates (catalyst timing, concentration) — placeholders
    # Will be wired in Phase 2 tasks

    return True, notes, cumulative_multiplier, multiplier_breakdown


def execute_trade(db, decision: dict, profile_id: str, *, normalized: bool = False):
    """
    Apply a trade decision to the paper portfolio.

    Supported actions:
      BUY   — open or add to a long position
      SHORT — open or add to a short position
      CLOSE — close an existing long or short position

    Args:
        normalized: If True, the decision has passed the Entry Normalizer
            and stop/target are guaranteed present. Fallback stop derivation
            is skipped. If a normalized BUY/SHORT somehow reaches the fallback
            path without a stop, execution is REJECTED (fail closed).
    """
    action = str(decision["action"]).upper()
    decision["action"] = action
    symbol = decision["symbol"]

    if action not in {"BUY", "SHORT", "CLOSE"}:
        return False, f"Unsupported action: {action}"

    quantity = _coerce_quantity(decision.get("quantity", 0), symbol=symbol)
    decision["quantity"] = quantity
    price = decision.get("price") or decision.get("entry_price") or 0
    try:
        price = float(price) if price not in (None, "") else 0
    except (TypeError, ValueError):
        log.warning("Decision for %s had non-numeric price %r", symbol, price)
        price = 0

    # Sanity-check the LLM's price against a live quote. If the LLM omitted
    # price (or JSON repair produced price=0), use a live quote for valid
    # symbols instead of creating noisy "No price in decision" rejects.
    try:
        fh = FinnhubClient()
        live_quote = fh.get_quote(symbol)
        live_price = live_quote.get("price", 0)
        try:
            live_price = float(live_price) if live_price not in (None, "") else 0
        except (TypeError, ValueError):
            live_price = 0
        if (not price or price <= 0) and live_price and live_price > 0:
            log.warning(
                "Decision for %s had missing/zero price; using live price %.2f",
                symbol, live_price,
            )
            price = live_price
            decision["price"] = price
            decision["entry_price"] = price
        if not price or price <= 0:
            return False, "No price in decision and live quote unavailable"
        if live_price and live_price > 0 and action != "CLOSE":
            deviation = abs(price - live_price) / live_price

            if deviation > 0.10:
                # Tier 3 — extreme deviation (>10%): reject outright
                log.warning(
                    "Extreme price deviation for %s: LLM price=%.2f, "
                    "live price=%.2f (%.1f%% deviation). Rejecting trade — "
                    "stale context or hallucinated price.",
                    symbol, price, live_price, deviation * 100,
                )
                return False, (
                    f"Extreme price deviation for {symbol}: "
                    f"LLM price={price:.2f}, live price={live_price:.2f} "
                    f"({deviation * 100:.1f}% deviation). "
                    f"Trade rejected — price mismatch too large for repair."
                )

            elif deviation > 0.05:
                # Tier 2 — moderate deviation (>5% to ≤10%): proportional repair
                original_entry = price
                original_stop = (
                    decision.get("stop")
                    or decision.get("stop_price")
                    or decision.get("stop_loss")
                )
                original_target = (
                    decision.get("target")
                    or decision.get("target_price")
                    or decision.get("profit_target")
                )

                price = live_price
                decision["price"] = price
                decision["entry_price"] = price

                if original_stop and original_entry:
                    stop_ratio = (original_stop - original_entry) / original_entry
                    new_stop = round(live_price * (1 + stop_ratio), 2)
                else:
                    new_stop = None

                if original_target and original_entry:
                    target_ratio = (original_target - original_entry) / original_entry
                    new_target = round(live_price * (1 + target_ratio), 2)
                else:
                    new_target = None

                # Validate repaired geometry before accepting
                geometry_valid = True
                rejection_reason = ""
                if new_stop is not None and new_target is not None:
                    if action == "BUY":
                        # LONG: stop must be below entry, target must be above entry
                        if new_stop >= live_price:
                            geometry_valid = False
                            rejection_reason = (
                                f"Repaired stop ({new_stop:.2f}) >= entry ({live_price:.2f}) for BUY"
                            )
                        elif new_target <= live_price:
                            geometry_valid = False
                            rejection_reason = (
                                f"Repaired target ({new_target:.2f}) <= entry ({live_price:.2f}) for BUY"
                            )
                        else:
                            # Check R:R >= 1:1
                            risk = live_price - new_stop
                            reward = new_target - live_price
                            if risk > 0 and reward / risk < 1.0:
                                geometry_valid = False
                                rejection_reason = (
                                    f"Repaired R:R ({reward / risk:.2f}) < 1:1 for BUY"
                                )
                    elif action == "SHORT":
                        # SHORT: stop must be above entry, target must be below entry
                        if new_stop <= live_price:
                            geometry_valid = False
                            rejection_reason = (
                                f"Repaired stop ({new_stop:.2f}) <= entry ({live_price:.2f}) for SHORT"
                            )
                        elif new_target >= live_price:
                            geometry_valid = False
                            rejection_reason = (
                                f"Repaired target ({new_target:.2f}) >= entry ({live_price:.2f}) for SHORT"
                            )
                        else:
                            # Check R:R >= 1:1
                            risk = new_stop - live_price
                            reward = live_price - new_target
                            if risk > 0 and reward / risk < 1.0:
                                geometry_valid = False
                                rejection_reason = (
                                    f"Repaired R:R ({reward / risk:.2f}) < 1:1 for SHORT"
                                )

                if not geometry_valid:
                    log.warning(
                        "Price repair geometry invalid for %s: %s. "
                        "Original entry=%.2f, stop=%s, target=%s. "
                        "Repaired entry=%.2f, stop=%s, target=%s. Rejecting.",
                        symbol, rejection_reason,
                        original_entry, original_stop, original_target,
                        live_price, new_stop, new_target,
                    )
                    return False, (
                        f"Price repair geometry invalid for {symbol}: {rejection_reason}"
                    )

                # Write repaired values back to decision dict so downstream
                # extraction picks up corrected values
                if new_stop is not None:
                    decision["stop"] = new_stop
                    if "stop_price" in decision:
                        decision["stop_price"] = new_stop
                    if "stop_loss" in decision:
                        decision["stop_loss"] = new_stop

                if new_target is not None:
                    decision["target"] = new_target
                    if "target_price" in decision:
                        decision["target_price"] = new_target
                    if "profit_target" in decision:
                        decision["profit_target"] = new_target

                log.warning(
                    "Proportional price repair for %s: deviation=%.1f%%. "
                    "Entry: %.2f → %.2f. Stop: %s → %s (ratio=%.4f). "
                    "Target: %s → %s (ratio=%.4f).",
                    symbol, deviation * 100,
                    original_entry, live_price,
                    original_stop, new_stop,
                    stop_ratio if original_stop and original_entry else 0,
                    original_target, new_target,
                    target_ratio if original_target and original_entry else 0,
                )

            # Tier 1 — ≤5% deviation: no change (existing passthrough)

    except Exception as exc:
        log.warning("Could not verify price for %s: %s", symbol, exc)
        if not price or price <= 0:
            return False, "No price in decision and live quote unavailable"

    # Extract stop/target from multiple possible keys the LLM might use
    stop = decision.get("stop") or decision.get("stop_price") or decision.get("stop_loss")
    target = decision.get("target") or decision.get("target_price") or decision.get("profit_target")

    # If stop/target are in rationale text but not fields, try to parse them out
    if not stop or not target:
        rationale = decision.get("rationale", "")
        import re
        if not stop:
            m = re.search(r'stop[:\s]*\$?([\d.]+)', rationale, re.IGNORECASE)
            if m:
                try: stop = float(m.group(1))
                except: pass
        if not target:
            m = re.search(r'target[:\s]*\$?([\d.]+)', rationale, re.IGNORECASE)
            if m:
                try: target = float(m.group(1))
                except: pass

    # If still no stop, derive from ATR or analyst key levels — never use flat %
    if action in ("BUY", "SHORT") and not stop and price:
        if normalized:
            # Normalized orders have stop guaranteed by the Entry Normalizer.
            # If we reach here, it's an invariant violation — fail closed.
            log.error(
                "NORMALIZED_ORDER_INVARIANT_VIOLATED: %s reached fallback stop "
                "derivation with no stop. This should never happen. "
                "Rejecting trade.",
                symbol,
            )
            return False, (
                f"Normalized order for {symbol} missing stop — "
                f"invariant violation, trade rejected"
            )

        import logging
        _log = logging.getLogger(__name__)

        # Try 1: ATR-based stop (1.5x ATR from entry)
        try:
            from utils.technicals import compute_indicators
            fh = FinnhubClient()
            candles = fh.get_candles(symbol, resolution="5", days=2)
            indicators = compute_indicators(candles)
            atr = indicators.get("atr")
            if atr and atr > 0:
                if action == "BUY":
                    stop = round(price - (atr * 1.5), 2)
                else:
                    stop = round(price + (atr * 1.5), 2)
                _log.info(f"Stop derived from ATR ({atr:.2f} × 1.5) for {symbol}: {stop}")
        except Exception:
            pass

        # Try 2: Key level from analyst signal
        if not stop:
            try:
                sig_mem = (
                    db.query(AgentMemory)
                    .filter_by(agent="analyst", symbol=symbol, key="signal")
                    .order_by(AgentMemory.timestamp.desc())
                    .first()
                )
                if sig_mem:
                    import json as _json
                    sig = _json.loads(sig_mem.value)
                    levels = sig.get("key_levels", {})
                    if action == "BUY" and levels.get("support"):
                        stop = round(float(levels["support"]) * 0.995, 2)  # just below support
                        _log.info(f"Stop derived from support level for {symbol}: {stop}")
                    elif action == "SHORT" and levels.get("resistance"):
                        stop = round(float(levels["resistance"]) * 1.005, 2)  # just above resistance
                        _log.info(f"Stop derived from resistance level for {symbol}: {stop}")
            except Exception:
                pass

        # Try 3: Last resort — 2x ATR or 1.5% (whichever is available)
        if not stop:
            if action == "BUY":
                stop = round(price * 0.985, 2)
            else:
                stop = round(price * 1.015, 2)
            _log.warning(f"No ATR or key level for {symbol}, using 1.5% fallback: {stop}")

    if action in ("BUY", "SHORT", "CLOSE"):
        log_trade_event(
            db,
            "entry_requested" if action in ("BUY", "SHORT") else "exit_requested",
            agent=f"pm_{profile_id}",
            symbol=symbol,
            profile=profile_id,
            price=price,
            message=decision.get("rationale"),
            payload={
                "action": action,
                "quantity": quantity,
                "stop": stop,
                "target": target,
                "decision": decision,
            },
        )

    starting = PM_PROFILES[profile_id]["starting_balance"]
    bal = (
        db.query(Balance)
        .filter_by(profile=profile_id)
        .order_by(Balance.timestamp.desc())
        .first()
    )
    cash = bal.cash if bal else float(starting)

    # ── Gate Pipeline: deterministic pre-trade safety gates ──
    _gate_notes = []
    _gate_multiplier = 1.0
    _gate_multiplier_breakdown = []
    _gate_decision_id = None  # Correlation ID from recovery-probe gate decisions
    _pilot_override_info = None  # Pilot override metadata for trade event tagging
    _candidate_lineage_id = None  # Replay lineage ID propagated to downstream tables
    if action in ("BUY", "SHORT"):
        signal_for_gates = _build_signal_for_symbol(db, symbol, decision)

        # ── Decision Snapshot: persist BEFORE gates (Requirement 3.1, 3.6) ──
        try:
            from utils.decision_snapshot import (
                build_and_persist_snapshot,
                generate_candidate_lineage_id,
                SnapshotPersistenceError,
            )

            _candidate_lineage_id = generate_candidate_lineage_id()

            # Compute account equity for snapshot
            _snap_positions = db.query(Position).filter_by(profile=profile_id).all()
            _snap_pos_value = sum(p.quantity * p.avg_cost for p in _snap_positions)
            _snap_equity = cash + _snap_pos_value
            _snap_open_positions = [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_cost": float(p.avg_cost),
                    "direction": p.direction if hasattr(p, "direction") else "long",
                }
                for p in _snap_positions
            ]

            # Collect gate config and feature flags for snapshot
            _snap_gate_config = {}
            _snap_feature_flags = {}
            try:
                from core.replay.policy_version import build_current_policy_version
                _snap_policy_version = build_current_policy_version()
                _snap_policy_version_id = f"{_snap_policy_version.gate_revision}:{_snap_policy_version.config_digest[:16]}"
                _snap_feature_flags = _snap_policy_version.feature_flags

                # Capture gate config values for snapshot
                from utils import gate_config as _gc
                _snap_gate_config = {
                    "min_win_rate_by_setup": _gc.MIN_WIN_RATE_BY_SETUP,
                    "default_min_win_rate": _gc.DEFAULT_MIN_WIN_RATE,
                    "stop_distance_rules": getattr(_gc, "STOP_DISTANCE_RULES", {}),
                    "default_stop_distance_rule": getattr(_gc, "DEFAULT_STOP_DISTANCE_RULE", {}),
                    "reduced_rr_thresholds_by_profile": getattr(_gc, "REDUCED_RR_THRESHOLDS_BY_PROFILE", {}),
                    "qualifying_min_signal_strength": getattr(_gc, "QUALIFYING_MIN_SIGNAL_STRENGTH", 0),
                    "qualifying_setup_types": list(getattr(_gc, "QUALIFYING_SETUP_TYPES", set())),
                    "override_min_confidence_score": getattr(_gc, "OVERRIDE_MIN_CONFIDENCE_SCORE", 0),
                }
            except Exception as _pv_exc:
                log.debug("Could not build policy version for snapshot: %s", _pv_exc)
                _snap_policy_version_id = "unknown"

            build_and_persist_snapshot(
                engine=db.bind,
                candidate_lineage_id=_candidate_lineage_id,
                decision=decision,
                signal=signal_for_gates,
                profile_id=profile_id,
                account_equity=_snap_equity,
                available_cash=cash,
                open_positions=_snap_open_positions,
                gate_config=_snap_gate_config,
                feature_flags=_snap_feature_flags,
                policy_version_id=_snap_policy_version_id,
            )
        except SnapshotPersistenceError as _snap_err:
            # Requirement 3.6: If snapshot persistence fails, BLOCK gate evaluation
            # UNLESS the table simply doesn't exist yet (migration not applied)
            _err_str = str(_snap_err)
            if "no such table" in _err_str:
                # Table doesn't exist — schema not migrated yet. Non-blocking.
                log.warning(
                    "Decision snapshot table not available for %s (schema migration pending): %s",
                    symbol, _snap_err,
                )
            else:
                log.error(
                    "DECISION SNAPSHOT FAILED for %s — gate evaluation BLOCKED: %s",
                    symbol, _snap_err,
                )
                log_trade_event(
                    db,
                    "snapshot_persistence_failed",
                    agent=f"pm_{profile_id}",
                    symbol=symbol,
                    profile=profile_id,
                    message=f"Decision snapshot persistence failed: {_snap_err}",
                    payload={"error": str(_snap_err), "candidate_lineage_id": _candidate_lineage_id},
                )
                return False, f"Snapshot persistence failed — gate evaluation blocked: {_snap_err}"
        except Exception as _snap_exc:
            # Non-fatal: log but allow gate pipeline to proceed
            # This handles cases where imports fail, table doesn't exist, etc.
            log.warning(
                "Decision snapshot creation failed (non-blocking) for %s: %s",
                symbol, _snap_exc,
            )

        # Store lineage ID on decision for downstream propagation
        if _candidate_lineage_id:
            decision["candidate_lineage_id"] = _candidate_lineage_id

        proceed, _gate_notes, _gate_multiplier, _gate_multiplier_breakdown = _run_gate_pipeline(
            db, db.bind, decision, signal_for_gates, profile_id,
        )
        if not proceed:
            gate_rejection_reasons = "; ".join(
                n.get("reason", "") for n in _gate_notes if n.get("decision") in ("reject", "rejected", "override_required")
            )
            trade_event = log_trade_event(
                db,
                "gate_rejected",
                agent=f"pm_{profile_id}",
                symbol=symbol,
                profile=profile_id,
                price=price,
                message=f"Trade rejected by gate pipeline: {gate_rejection_reasons}",
                payload={"gate_notes": _gate_notes, "candidate_lineage_id": _candidate_lineage_id},
            )
            db.flush()  # ensure trade_event.id is populated for shadow ledger linkage

            # Determine blocked_by: single rejecting gate or "gate_pipeline"
            rejecting_gates = [
                n.get("gate", "unknown")
                for n in _gate_notes
                if n.get("decision") in ("reject", "rejected", "override_required", "block")
            ]
            if len(rejecting_gates) == 1:
                blocked_by = rejecting_gates[0]
            else:
                blocked_by = "gate_pipeline"

            # Build block_reason — include "block" decisions for completeness
            block_reason = gate_rejection_reasons or "; ".join(
                n.get("reason", "") for n in _gate_notes if n.get("decision") in ("block",)
            ) or "Gate pipeline rejected"

            record_blocked_candidate(
                db,
                symbol=symbol,
                action=action,
                blocked_by=blocked_by,
                block_reason=block_reason,
                profile=profile_id,
                entry_price=price,
                stop_price=(
                    decision.get("stop")
                    or decision.get("stop_price")
                    or decision.get("stop_loss")
                ),
                target_price=(
                    decision.get("target")
                    or decision.get("target_price")
                    or decision.get("profit_target")
                ),
                quantity=quantity,
                gate_notes=_gate_notes,
                decision_snapshot=decision,
                signal_snapshot=signal_for_gates,
                source="analyst_signal",
                agent=f"pm_{profile_id}",
                trade_event_id=trade_event.id if trade_event else None,
                geometry_candidate_id=decision.get("geometry_candidate_id"),
                geometry_candidate_name=decision.get("geometry_candidate_name"),
                candidate_lineage_id=_candidate_lineage_id,
            )

            log.warning(
                "DECISION: status=GATE_REJECTED symbol=%s profile=%s gate=%s reason=%s",
                symbol, profile_id, blocked_by, gate_rejection_reasons,
            )
            return False, f"Gate rejected ({blocked_by}): {gate_rejection_reasons}"

        # Apply cumulative size multiplier from gates
        pre_gate_qty = quantity  # Capture before any gate reduction for sizing invariant
        if _gate_multiplier < 1.0 and quantity > 0:
            quantity = max(1, int(quantity * _gate_multiplier))
            decision["quantity"] = quantity
            log.info(
                "Gate pipeline size multiplier applied: %.2f × %d → %d",
                _gate_multiplier, pre_gate_qty, quantity,
            )

        # Propagate any adjusted parameters from the risk geometry gate
        # (decision dict was updated inside _run_gate_pipeline)
        rg_note = next((n for n in _gate_notes if n.get("gate") == "risk_geometry_gate" and n.get("decision") == "adjusted_allowed"), None)
        if rg_note:
            stop = rg_note["stop_price"]
            rg_quantity = rg_note["quantity"]
            # Sizing invariant: geometry may reduce but never restore above reduced cap
            reduced_cap = max(1, int(pre_gate_qty * _gate_multiplier)) if _gate_multiplier < 1.0 else pre_gate_qty
            quantity = min(reduced_cap, rg_quantity)
            decision["stop"] = stop
            decision["stop_loss"] = stop
            decision["stop_price"] = stop
            decision["quantity"] = quantity

        # Extract gate_decision_id from gate_notes for correlation (recovery probes).
        # The setup_quality_gate note carries gate_decision_id when a recovery-probe
        # decision was made. This is propagated to entry_filled events downstream.
        _gate_decision_id = None
        sqg_note = next((n for n in _gate_notes if n.get("gate") == "setup_quality_gate"), None)
        if sqg_note:
            _gate_decision_id = sqg_note.get("gate_decision_id")

        # Detect pilot override from gate notes for trade event tagging.
        # If any gate applied a pilot override (near-miss or R:R), propagate
        # pilot_override: true and pilot_size_multiplier to the entry_filled event.
        _pilot_override_info = None
        for _gn in _gate_notes:
            if _gn.get("pilot_override") is True or _gn.get("reason_type") == "near_miss_pilot_override":
                _pilot_override_info = {
                    "pilot_override": True,
                    "pilot_size_multiplier": _gn.get("size_multiplier", 0.25),
                }
                break

    # ── Tier-1 pre-validation: similarity → edge score → portfolio risk ──
    # Tracks edge data to store on the Trade record later
    _edge_data = {}

    if action in ("BUY", "SHORT"):
        base_quantity = quantity  # preserve original for cap calculation

        # --- Build signal context for this symbol ---
        signal_for_symbol = _build_signal_for_symbol(db, symbol, decision)

        # --- 1. Similarity engine (fail-open: proceed with zero stats on error) ---
        sim_stats = {
            "similarity_winrate": 0.0, "similarity_avg_r": 0.0,
            "sample_size": 0, "similarity_confidence": 0.0, "skip_similarity": True,
        }
        try:
            similar_cases = find_similar_cases(signal_for_symbol, db.bind)
            sim_stats = compute_similarity_stats(similar_cases)
        except Exception as exc:
            log.warning("Similarity engine error (proceeding with zero stats): %s", exc)

        # --- 2. Case stats from existing win rate data ---
        setup_type = decision.get("setup_type") or decision.get("setup") or signal_for_symbol.get("setup_type") or ""
        regime = decision.get("market_regime") or decision.get("regime") or signal_for_symbol.get("market_regime")
        case_stats = _build_case_stats(db, setup_type, regime)

        # High-WR fast intraday setups need mandatory breathing room. Enforce
        # before edge/validation so the persisted stop and Entry Contract agree.
        adjusted_stop = _apply_high_wr_stop_buffer(
            action, price, stop, decision, signal_for_symbol, case_stats
        )
        if adjusted_stop != stop:
            stop = adjusted_stop
            decision["stop"] = stop
            decision["stop_loss"] = stop

        cooldown_msg = _high_momentum_cooldown_message(db, symbol)
        if cooldown_msg:
            log.warning(cooldown_msg)
            return False, cooldown_msg

        # --- 3. Hard rejection check (fail-closed) ---
        try:
            if check_hard_rejection(case_stats):
                log.warning(
                    "DECISION: status=REJECTED reason=hard_rejection "
                    "setup_winrate=%.2f sample_size=%d",
                    case_stats["win_rate"], case_stats["sample_size"],
                )
                return False, (
                    f"Hard reject: setup winrate too low "
                    f"({case_stats['win_rate']:.2f} over {case_stats['sample_size']} cases)"
                )
        except Exception as exc:
            log.error("Hard rejection check failed (rejecting trade): %s", exc)
            return False, f"Edge score pre-check error: {exc}"

        # --- 4. Compute edge score (fail-closed) ---
        try:
            edge = compute_edge_score(signal_for_symbol, case_stats, sim_stats)
        except Exception as exc:
            log.error("Edge score computation failed (rejecting trade): %s", exc)
            return False, f"Edge score computation error: {exc}"

        # Compute sub-components for logging
        _confluence = confluence_score(
            signal_for_symbol.get("indicators", {}),
            signal_for_symbol.get("bias", ""),
        )
        _sim_qual = similarity_quality(sim_stats.get("sample_size", 0))

        # --- EDGE SCORE structured log ---
        log.info(
            "EDGE SCORE: %.3f | setup_winrate=%.2f (n=%d) | "
            "similarity_winrate=%.2f (n=%d) | similarity_confidence=%.2f | "
            "confluence=%.2f | similarity_quality=%.2f",
            edge,
            case_stats.get("win_rate", 0), case_stats.get("sample_size", 0),
            sim_stats.get("similarity_winrate", 0), sim_stats.get("sample_size", 0),
            sim_stats.get("similarity_confidence", 0),
            _confluence, _sim_qual,
        )

        if edge < 0.4:
            log.info(
                "DECISION: status=REJECTED reason=edge_score_too_low (%.3f < 0.4)", edge
            )
            return False, f"Edge score too low ({edge:.3f})"

        # --- 5. Scale position size by edge score, cap at 1.2× base ---
        scaled_size = max(1, int(quantity * edge))
        quantity = int(cap_position_size(scaled_size, base_quantity))
        decision["quantity"] = quantity

        # --- 5b. Apply pipeline stage position multiplier ---
        try:
            strategy_key = (
                decision.get("strategy_key")
                or decision.get("setup_type")
                or decision.get("setup")
                or signal_for_symbol.get("setup_type")
                or ""
            )
            if strategy_key:
                multiplier = get_strategy_position_multiplier(db.bind, strategy_key)
                if multiplier < 1.0 and multiplier > 0:
                    pre_mult_qty = quantity
                    quantity = max(1, int(quantity * multiplier))
                    decision["quantity"] = quantity
                    log.info(
                        "Pipeline multiplier applied for %s: %.1f × %d → %d",
                        strategy_key, multiplier, pre_mult_qty, quantity,
                    )
                elif multiplier == 0.0:
                    log.warning(
                        "Strategy %s not in a live stage (multiplier=0.0), rejecting trade",
                        strategy_key,
                    )
                    return False, f"Strategy {strategy_key} not in a live pipeline stage"
        except Exception as exc:
            log.warning("Pipeline multiplier lookup error (proceeding without scaling): %s", exc)

        # --- 6. Adaptive risk throttling (fail-open) ---
        try:
            recent_losses = _count_recent_consecutive_losses(db, profile_id)
            if recent_losses >= 3:
                throttled = adaptive_risk_throttle(quantity, recent_losses)
                quantity = max(1, int(throttled))
                decision["quantity"] = quantity
                log.info(
                    "Adaptive risk throttle: recent_losses=%d, size %d → %d",
                    recent_losses, scaled_size, quantity,
                )
        except Exception as exc:
            log.warning("Adaptive risk throttle error (proceeding): %s", exc)
            recent_losses = 0

        # --- 7. Portfolio risk validation (fail-open) ---
        try:
            positions = db.query(Position).filter_by(profile=profile_id).all()
            pos_list = [
                {"symbol": p.symbol, "quantity": p.quantity, "avg_cost": p.avg_cost, "side": p.side}
                for p in positions
            ]
            pos_value = sum(p.quantity * p.avg_cost for p in positions)
            total_equity = cash + pos_value

            risk_result = compute_portfolio_risk(pos_list, total_equity)

            # --- PORTFOLIO RISK structured log ---
            bucket_str = ", ".join(
                f"{k}={v:.2f}" for k, v in risk_result.get("bucket_exposure", {}).items()
            )
            log.info(
                "PORTFOLIO RISK: total_exposure=%.2f | %s",
                risk_result.get("total_exposure", 0), bucket_str,
            )

            risk_ok, risk_msg = validate_portfolio_risk(
                {"symbol": symbol, "quantity": quantity, "price": price},
                pos_list, total_equity,
            )
            if not risk_ok:
                log.info("DECISION: status=REJECTED reason=%s", risk_msg)
                return False, risk_msg
        except Exception as exc:
            log.warning("Portfolio risk check error (proceeding with existing validation): %s", exc)

        # --- Store edge data for Trade record (Task 5.4) ---
        _edge_data = {
            "edge_score": round(edge, 4),
            "similarity_winrate": round(sim_stats.get("similarity_winrate", 0), 4),
            "similarity_sample_size": sim_stats.get("sample_size", 0),
            "similarity_confidence": round(sim_stats.get("similarity_confidence", 0), 4),
        }

        # --- DECISION structured log (executed) ---
        log.info(
            "DECISION: size_scaled=%d status=EXECUTED edge=%.3f",
            quantity, edge,
        )

    # Validate trade before execution (existing validation)
    if action in ("BUY", "SHORT"):
        from utils.trade_validator import validate_trade, TradeValidationError
        direction = "LONG" if action == "BUY" else "SHORT"
        # Build a normalized decision for validation
        validated = {**decision, "price": price, "stop": stop, "target": target, "quantity": quantity}
        positions = db.query(Position).filter_by(profile=profile_id).all()
        pos_value = sum(p.quantity * p.avg_cost for p in positions)
        total_equity = cash + pos_value
        try:
            validate_trade(validated, profile_id, cash, total_equity, direction)
        except TradeValidationError as e:
            import logging
            logging.getLogger(__name__).warning(f"Trade rejected: {e}")
            return False, str(e)

        # Check correlated exposure
        from utils.trade_validator import check_correlation
        corr_warning = check_correlation(symbol, direction, profile_id, db)
        if corr_warning:
            import logging
            logging.getLogger(__name__).warning(f"Trade rejected: {corr_warning}")
            return False, corr_warning

        # Confidence adjustment based on case library win rates
        from utils.trade_validator import adjust_confidence
        setup_type = decision.get("setup_type") or decision.get("setup") or ""
        regime = decision.get("market_regime") or decision.get("regime")
        conf_adj = adjust_confidence(db.bind, setup_type, regime)
        if conf_adj["block"]:
            import logging
            logging.getLogger(__name__).warning(f"Trade BLOCKED: {conf_adj['reason']}")
            return False, conf_adj["reason"]
        if conf_adj["modifier"] < 1.0:
            import logging
            logging.getLogger(__name__).info(f"Confidence adjusted: {conf_adj['reason']}")

    # ── Build Entry Contract for BUY/SHORT actions ──
    _entry_contract = {}
    if action in ("BUY", "SHORT"):
        try:
            signal_for_contract = _build_signal_for_symbol(db, symbol, decision)
            _entry_contract = build_entry_contract(
                decision, signal_for_contract,
                stop or 0.0, target or 0.0,
            )
            log.info(
                "Entry contract built for %s: setup_type=%s, invalidators=%d",
                symbol,
                _entry_contract.get("setup_type", "unknown"),
                len(_entry_contract.get("invalidators", [])),
            )
        except Exception as exc:
            log.warning("Failed to build entry contract for %s: %s", symbol, exc)

    if action == "BUY":
        cost = quantity * price
        if cost > cash:
            return False, "Insufficient cash"

        pos = db.query(Position).filter_by(
            symbol=symbol, profile=profile_id, side="long"
        ).first()
        if pos:
            total_qty = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + price * quantity) / total_qty
            pos.quantity = total_qty
        else:
            pos = Position(
                symbol=symbol, quantity=quantity,
                avg_cost=price, profile=profile_id, side="long"
            )
            db.add(pos)

        trade = Trade(
            symbol=symbol, direction="LONG", quantity=quantity,
            entry_price=price, reason_entry=decision.get("rationale"),
            stop_price=stop,
            target_price=target,
            profile=profile_id,
            edge_score=_edge_data.get("edge_score"),
            similarity_winrate=_edge_data.get("similarity_winrate"),
            similarity_sample_size=_edge_data.get("similarity_sample_size"),
            similarity_confidence=_edge_data.get("similarity_confidence"),
            thesis=_entry_contract.get("thesis"),
            setup_type=_entry_contract.get("setup_type"),
            invalidators=json.dumps(_entry_contract["invalidators"]) if _entry_contract.get("invalidators") else None,
        )
        db.add(trade)
        db.flush()
        _entry_filled_payload = {"action": action, "quantity": quantity, "side": "long", "edge": _edge_data}
        if _gate_decision_id is not None:
            _entry_filled_payload["gate_decision_id"] = _gate_decision_id
        if _pilot_override_info is not None:
            _entry_filled_payload["pilot_override"] = True
            _entry_filled_payload["pilot_size_multiplier"] = _pilot_override_info["pilot_size_multiplier"]
        log_trade_event(db, "entry_filled", trade_id=trade.id, agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id, price=price, message=decision.get("rationale"), payload=_entry_filled_payload)
        if stop:
            log_trade_event(db, "stop_set", trade_id=trade.id, agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id, price=float(stop), message="Initial stop set", payload={"stop_price": stop})
        if target:
            log_trade_event(db, "target_set", trade_id=trade.id, agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id, price=float(target), message="Initial target set", payload={"target_price": target})
        db.add(Balance(cash=cash - cost, profile=profile_id))

    elif action == "SHORT":
        # Paper short: reserve margin equal to position value
        margin_required = quantity * price
        if margin_required > cash:
            return False, "Insufficient margin for short"

        pos = db.query(Position).filter_by(
            symbol=symbol, profile=profile_id, side="short"
        ).first()
        if pos:
            total_qty = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + price * quantity) / total_qty
            pos.quantity = total_qty
        else:
            pos = Position(
                symbol=symbol, quantity=quantity,
                avg_cost=price, profile=profile_id, side="short"
            )
            db.add(pos)

        trade = Trade(
            symbol=symbol, direction="SHORT", quantity=quantity,
            entry_price=price, reason_entry=decision.get("rationale"),
            stop_price=stop,
            target_price=target,
            profile=profile_id,
            edge_score=_edge_data.get("edge_score"),
            similarity_winrate=_edge_data.get("similarity_winrate"),
            similarity_sample_size=_edge_data.get("similarity_sample_size"),
            similarity_confidence=_edge_data.get("similarity_confidence"),
            thesis=_entry_contract.get("thesis"),
            setup_type=_entry_contract.get("setup_type"),
            invalidators=json.dumps(_entry_contract["invalidators"]) if _entry_contract.get("invalidators") else None,
        )
        db.add(trade)
        db.flush()
        _entry_filled_payload = {"action": action, "quantity": quantity, "side": "short", "edge": _edge_data}
        if _gate_decision_id is not None:
            _entry_filled_payload["gate_decision_id"] = _gate_decision_id
        if _pilot_override_info is not None:
            _entry_filled_payload["pilot_override"] = True
            _entry_filled_payload["pilot_size_multiplier"] = _pilot_override_info["pilot_size_multiplier"]
        log_trade_event(db, "entry_filled", trade_id=trade.id, agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id, price=price, message=decision.get("rationale"), payload=_entry_filled_payload)
        if stop:
            log_trade_event(db, "stop_set", trade_id=trade.id, agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id, price=float(stop), message="Initial stop set", payload={"stop_price": stop})
        if target:
            log_trade_event(db, "target_set", trade_id=trade.id, agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id, price=float(target), message="Initial target set", payload={"target_price": target})
        # Deduct margin from cash (returned + P&L on close)
        db.add(Balance(cash=cash - margin_required, profile=profile_id))

    elif action == "CLOSE":
        # Find the open position (long or short)
        pos = db.query(Position).filter_by(
            symbol=symbol, profile=profile_id
        ).first()
        if not pos:
            return False, "No position to close"

        close_qty = quantity if quantity and quantity < pos.quantity else pos.quantity
        close_qty = abs(close_qty)  # Guard against negative qty from LLM decisions
        side = pos.side

        # Find ALL open trades for this symbol/profile (handles averaged-in positions)
        open_trades = (
            db.query(Trade)
            .filter_by(symbol=symbol, status="open", profile=profile_id)
            .order_by(Trade.entry_time)
            .all()
        )

        pnl_total = 0.0
        remaining_to_close = close_qty
        first_trade = open_trades[0] if open_trades else None

        for open_trade in open_trades:
            if remaining_to_close <= 0:
                break
            trade_close_qty = min(open_trade.quantity, remaining_to_close)
            remaining_to_close -= trade_close_qty

            if side == "long":
                pnl = (price - open_trade.entry_price) * trade_close_qty
            else:  # short: profit when price falls
                pnl = (open_trade.entry_price - price) * trade_close_qty
            pnl_pct = pnl / (open_trade.entry_price * trade_close_qty) * 100

            open_trade.exit_price = price
            open_trade.exit_time = datetime.utcnow()
            open_trade.status = "closed"
            open_trade.pnl = round(pnl, 2)
            open_trade.pnl_pct = round(pnl_pct, 2)
            open_trade.reason_exit = decision.get("rationale")

            # Post-trade PnL sign consistency check
            if pnl != 0 and ((pnl > 0) != (pnl_pct > 0)):
                log.warning(
                    "PnL sign mismatch for %s: pnl=%.2f, pnl_pct=%.2f — signs should be consistent",
                    symbol, pnl, pnl_pct,
                )

            log_trade_event(
                db, "exit_filled", trade_id=open_trade.id, agent=f"pm_{profile_id}",
                symbol=symbol, profile=profile_id, price=price, message=decision.get("rationale"),
                payload={"quantity": trade_close_qty, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2), "side": side},
            )

            # Queue for review
            from db.schema import ReviewQueue
            db.add(ReviewQueue(trade_id=open_trade.id))

            pnl_total += pnl

        if close_qty >= pos.quantity:
            db.delete(pos)
        else:
            pos.quantity -= close_qty

        # Return margin + P&L to cash
        if side == "long":
            cash_delta = close_qty * price
        else:
            # Return original margin + profit (or minus loss)
            margin_back = close_qty * (first_trade.entry_price if first_trade else price)
            profit = (first_trade.entry_price - price) * close_qty if first_trade else 0
            cash_delta = margin_back + profit

        db.add(Balance(cash=cash + cash_delta, profile=profile_id))

    db.commit()
    return True, "OK"


def _compute_shadow_agreement(candidate_results, legacy_results):
    """Compute simple agreement summary between candidate and legacy paths."""
    candidate_symbols = {r["symbol"] for r in candidate_results if r.get("outcome") == "executed"}
    legacy_symbols = {e.get("symbol") for e in legacy_results if e.get("executed")}
    common = candidate_symbols & legacy_symbols
    candidate_only = candidate_symbols - legacy_symbols
    legacy_only = legacy_symbols - candidate_symbols
    return json.dumps({
        "both_executed": sorted(common),
        "candidate_only": sorted(candidate_only),
        "legacy_only": sorted(legacy_only),
    })


def run_profile(engine, symbols: list[str], profile_id: str, tier: str = "high") -> dict:
    """
    Run a single PM profile for one cycle with two-tier review routing.

    The decision loop:
    1. Load all open positions with their Entry Contracts
    2. Check for pending Reversal triggers (thesis_invalidation from AgentMemory,
       opposing signals, explicit CLOSE)
    3. For positions WITH a Reversal trigger → call Reversal/Close Review
    4. For positions WITHOUT a Reversal trigger → call Maintenance Review
    5. For NEW entries (no existing position) → use existing entry logic unchanged
    6. Execute resulting decisions via execute_trade()
    7. Log each cycle whether signals were used in advisory or authoritative capacity

    tier controls which LLM is used.
    """
    profile = PM_PROFILES[profile_id]
    fh = FinnhubClient()
    db = get_session(engine)

    # Get analyst signals
    signals = {}
    for sym in symbols:
        sig = (
            db.query(AgentMemory)
            .filter_by(agent="analyst", symbol=sym, key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if sig:
            signals[sym] = json.loads(sig.value)

    # Get profile-specific execution feedback from Reviewer
    exec_fb = (
        db.query(AgentMemory)
        .filter_by(agent="reviewer", key=f"execution_feedback_{profile_id}")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    # Fall back to general execution feedback
    if not exec_fb:
        exec_fb = (
            db.query(AgentMemory)
            .filter_by(agent="reviewer", key="execution_feedback")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
    feedback_text = exec_fb.value if exec_fb else "No execution feedback yet."

    # Meta-reviewer recommendations for this PM profile
    meta_rec = (
        db.query(AgentMemory)
        .filter_by(agent="meta_reviewer", symbol=f"pm_{profile_id}", key="agent_recommendation")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    meta_text = meta_rec.value if meta_rec else ""

    portfolio = get_portfolio_for_profile(db, fh, profile_id)

    # Check daily loss limit before doing any work
    max_loss = portfolio["starting_balance"] * profile["max_daily_loss_pct"]
    if abs(portfolio["daily_pnl"]) >= max_loss and portfolio["daily_pnl"] < 0:
        notes = f"Daily loss limit hit (${portfolio['daily_pnl']:,.2f}). No more trades today."
        stored_notes = _store_pm_cycle_note(db, profile_id, notes)
        db.close()
        return {"decisions": [], "portfolio_notes": stored_notes, "profile": profile_id}

    # ── PHASE 1: Two-tier review for existing open positions ──
    # Track signal usage for audit logging (Req 4.4)
    signal_usage_log = []  # list of {"symbol", "usage": "advisory"|"authoritative"}
    review_decisions = []

    open_positions = db.query(Position).filter_by(profile=profile_id).all()
    open_symbols = {p.symbol for p in open_positions}

    # Position health from health monitor. Never feed stale/global health into
    # new-entry decisions: old XLE/TSLA assessments can make a flat portfolio
    # look like it still has critical positions.
    if not open_positions:
        health_text = "No open positions for this profile; no position-health context applies."
    else:
        health_mem = (
            db.query(AgentMemory)
            .filter_by(agent="position_health", key="health_check")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        health_text = "No fresh matching position-health data."
        if health_mem and health_mem.timestamp:
            age = datetime.utcnow() - health_mem.timestamp.replace(tzinfo=None)
            if age <= timedelta(hours=2):
                try:
                    health_data = json.loads(health_mem.value)
                    assessments = [
                        a for a in health_data.get("assessments", [])
                        if a.get("profile") == profile_id and a.get("symbol") in open_symbols
                    ]
                    if assessments:
                        health_text = json.dumps({
                            "assessments": assessments,
                            "summary": health_data.get("summary", ""),
                        })
                except Exception as exc:
                    log.warning("Failed to parse position health memory for %s: %s", profile_id, exc)
            else:
                health_text = (
                    f"Latest position-health memory is stale ({age.total_seconds() / 3600:.1f}h old); "
                    "ignored for this cycle."
                )

    for pos in open_positions:
        symbol = pos.symbol

        # Load the open trade with Entry Contract
        open_trade = (
            db.query(Trade)
            .filter_by(symbol=symbol, profile=profile_id, status="open")
            .order_by(Trade.entry_time.desc())
            .first()
        )
        if not open_trade:
            continue

        # Check if this position has an Entry Contract
        has_entry_contract = bool(open_trade.thesis)

        if not has_entry_contract:
            # Attempt legacy migration before skipping (Req 8.2, 8.3)
            contract = build_legacy_entry_contract(open_trade)
            if contract:
                open_trade.thesis = contract["thesis"]
                open_trade.setup_type = contract["setup_type"]
                open_trade.invalidators = json.dumps(contract["invalidators"])
                db.commit()
                has_entry_contract = True
            else:
                # No Entry Contract and migration not possible — skip two-tier review.
                log.info(
                    "Position %s has no Entry Contract — skipping two-tier review "
                    "(will be handled by legacy migration or existing logic).",
                    symbol,
                )
                continue

        # Get current price
        try:
            quote = fh.get_quote(symbol)
            current_price = quote["price"]
        except Exception:
            current_price = pos.avg_cost

        # Compute unrealized P&L
        if pos.side == "short":
            unrealized_pnl = (pos.avg_cost - current_price) * pos.quantity
        else:
            unrealized_pnl = (current_price - pos.avg_cost) * pos.quantity
        unrealized_pnl_pct = round(
            unrealized_pnl / (pos.avg_cost * pos.quantity) * 100, 2
        ) if pos.avg_cost and pos.quantity else 0

        # Detect DRIFTING state
        drifting = detect_drifting(db, open_trade)

        # Get analyst signal for this symbol (if any)
        signal_for_symbol = signals.get(symbol)

        # Check for Reversal triggers
        trigger_info = _check_reversal_triggers(
            db, open_trade, {}, signal_for_symbol, profile
        )

        # Build position data dict for review handlers
        position_data = {
            "symbol": symbol,
            "side": pos.side,
            "quantity": pos.quantity,
            "entry_price": open_trade.entry_price,
            "stop_price": open_trade.stop_price,
            "target_price": open_trade.target_price,
            "current_price": current_price,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "drifting": drifting,
            "thesis": open_trade.thesis,
            "setup_type": open_trade.setup_type,
            "invalidators": open_trade.invalidators,
            "indicators": signal_for_symbol.get("indicators") if signal_for_symbol else None,
            "advisory_signals": signal_for_symbol if signal_for_symbol else None,
            "health_text": health_text,
            "market_conditions": signal_for_symbol if signal_for_symbol else None,
            "opposing_evidence": signal_for_symbol if (
                trigger_info and trigger_info.get("type") == "opposing_signal"
            ) else None,
        }

        if trigger_info:
            # ── Reversal/Close Review (authoritative signal usage) ──
            log.info(
                "Routing %s to Reversal/Close Review: trigger=%s",
                symbol, trigger_info["type"],
            )
            review_result = run_reversal_close_review(
                position_data, trigger_info, profile, tier=tier
            )

            # Log authoritative signal usage (Req 4.4)
            if signal_for_symbol:
                signal_usage_log.append({
                    "symbol": symbol,
                    "usage": "authoritative",
                    "trigger": trigger_info["type"],
                })

            # Convert review result to an executable decision
            action = review_result.get("action", "hold_tighten")
            if action in ("close_full", "close_partial"):
                close_decision = {
                    "symbol": symbol,
                    "action": "CLOSE",
                    "quantity": pos.quantity if action == "close_full" else max(1, int(pos.quantity * 0.5)),
                    "price": current_price,
                    "rationale": (
                        f"Reversal/Close Review ({trigger_info['type']}): "
                        f"{review_result.get('reasoning', 'No reasoning')}"
                    ),
                }
                review_decisions.append(close_decision)
            elif action == "hold_tighten":
                # Tighten stop to breakeven if profitable
                if unrealized_pnl > 0 and open_trade.stop_price:
                    new_stop = open_trade.entry_price  # breakeven
                    # Fetch ATR for minimum-change threshold
                    try:
                        from utils.atr_helper import compute_intraday_atr
                        atr_data = compute_intraday_atr(symbol)
                    except Exception:
                        atr_data = None
                    atr_value = atr_data.get("atr") if atr_data else None
                    result = apply_stop_update(
                        db,
                        trade=open_trade,
                        new_stop=new_stop,
                        source_agent="portfolio_manager",
                        stop_role="maintenance_tighten",
                        reason=f"Reversal/Close Review hold_tighten for {symbol}",
                        current_price=current_price,
                        atr=atr_value,
                    )
                    db.commit()
                    if result.valid:
                        log.info(
                            "Reversal/Close Review hold_tighten for %s: "
                            "tightened stop to breakeven (%.2f)",
                            symbol, new_stop,
                        )
                    else:
                        log.warning(
                            "Reversal/Close Review hold_tighten rejected for %s: %s",
                            symbol, result.reason,
                        )
        else:
            # ── Maintenance Review (advisory signal usage) ──
            log.info("Routing %s to Maintenance Review", symbol)
            review_result = run_maintenance_review(
                position_data, profile, tier=tier
            )

            # Log advisory signal usage (Req 4.4)
            if signal_for_symbol:
                signal_usage_log.append({
                    "symbol": symbol,
                    "usage": "advisory",
                })

            # Apply maintenance actions
            action = review_result.get("action", "hold")
            if action == "tighten_stop":
                new_stop_raw = review_result.get("new_stop")
                # Suppression check: intercept invalid/non-monotonic proposals
                # BEFORE _coerce_price() so null/non-numeric values produce a
                # maintenance_stop_suppressed event rather than being silently
                # swallowed by _coerce_price() → None → continue.
                suppressed, suppress_reason = should_suppress_maintenance_stop(
                    pos.side, open_trade.stop_price, new_stop_raw
                )
                if suppressed:
                    _log_maintenance_stop_suppressed(
                        db,
                        trade_id=open_trade.id,
                        symbol=symbol,
                        profile=profile_id,
                        side=pos.side,
                        old_stop=open_trade.stop_price,
                        new_stop_raw=new_stop_raw,
                        reason=suppress_reason,
                    )
                    db.commit()
                    continue

                new_stop = _coerce_price(new_stop_raw, "new_stop", symbol)
                if new_stop is None:
                    continue
                # Fetch ATR for minimum-change threshold
                try:
                    from utils.atr_helper import compute_intraday_atr
                    atr_data = compute_intraday_atr(symbol)
                except Exception:
                    atr_data = None
                atr_value = atr_data.get("atr") if atr_data else None
                result = apply_stop_update(
                    db,
                    trade=open_trade,
                    new_stop=new_stop,
                    source_agent="portfolio_manager",
                    stop_role="maintenance_tighten",
                    reason=f"Maintenance Review tighten_stop for {symbol}",
                    current_price=current_price,
                    atr=atr_value,
                )
                db.commit()
                if result.valid:
                    log.info(
                        "Maintenance Review tighten_stop for %s: new stop=%.2f",
                        symbol, new_stop,
                    )
                else:
                    log.warning(
                        "Maintenance Review tighten_stop rejected for %s: %s",
                        symbol, result.reason,
                    )
            elif action == "raise_target" and review_result.get("new_target"):
                new_target = _coerce_price(review_result.get("new_target"), "new_target", symbol)
                if new_target is None:
                    continue
                open_trade.target_price = new_target
                db.commit()
                log.info(
                    "Maintenance Review raise_target for %s: new target=%.2f",
                    symbol, new_target,
                )
            elif action == "trim_partial" and review_result.get("trim_pct"):
                trim_qty = max(1, int(pos.quantity * review_result["trim_pct"] / 100))
                trim_decision = {
                    "symbol": symbol,
                    "action": "CLOSE",
                    "quantity": trim_qty,
                    "price": current_price,
                    "rationale": (
                        f"Maintenance Review trim_partial ({review_result['trim_pct']}%): "
                        f"{review_result.get('reasoning', 'No reasoning')}"
                    ),
                }
                review_decisions.append(trim_decision)

    # Log signal usage summary for this cycle (Req 4.4)
    advisory_count = sum(1 for s in signal_usage_log if s["usage"] == "advisory")
    authoritative_count = sum(1 for s in signal_usage_log if s["usage"] == "authoritative")
    log.info(
        "SIGNAL USAGE [%s]: advisory=%d, authoritative=%d, details=%s",
        profile_id, advisory_count, authoritative_count,
        json.dumps(signal_usage_log) if signal_usage_log else "no signals used",
    )

    # Execute review-generated decisions (close/trim from two-tier review)
    executed = []
    for decision in review_decisions:
        ok, msg = execute_trade(db, decision, profile_id)
        executed.append({
            **decision, "executed": ok, "message": msg,
            "profile": profile_id, "source": "two_tier_review",
        })

    # ── PHASE 2: Existing entry logic for NEW positions (unchanged) ──
    # Symbols that already have open positions are excluded from new entry consideration.
    held_symbols = {p.symbol for p in db.query(Position).filter_by(profile=profile_id).all()}

    # ── CANDIDATE-ID SELECTION BRANCH ──
    from utils.gate_config import PM_CANDIDATE_MODE

    def _record_candidate_event(engine, candidate_id, cycle_id, profile_id, event_type, event_data):
        """Record a single audit event to pm_candidate_events table (Requirement 12)."""
        from sqlalchemy import text as _sql_text
        with engine.connect() as conn:
            conn.execute(
                _sql_text("""
                    INSERT INTO pm_candidate_events
                    (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                    VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
                """),
                {
                    "cid": candidate_id,
                    "cycle_id": cycle_id,
                    "profile_id": profile_id,
                    "event_type": event_type,
                    "event_data": json.dumps(event_data, default=str) if event_data is not None else None,
                    "created_at": datetime.utcnow().isoformat(),
                },
            )
            conn.commit()

    if PM_CANDIDATE_MODE in ("enabled", "shadow"):
        import copy
        import uuid
        from utils.candidate_registry import recover_stale_reservations
        from utils.candidate_builder import build_candidate_set
        from utils.candidate_prompt_builder import build_candidate_pm_prompt, build_decision_schema
        from utils.decision_contract import (
            parse_decision_contract,
            should_retry_candidate_contract,
            build_candidate_retry_prompt,
        )
        from utils.candidate_pipeline import execute_candidate_pipeline, dry_run_candidate_pipeline

        # Recover any stale reservations from prior cycles/crashes
        recover_stale_reservations(engine)

        # Generate unique cycle ID
        cycle_id = f"cycle_{profile_id}_{uuid.uuid4().hex[:12]}"

        # Capture frozen portfolio snapshot for consistent evaluation
        portfolio_snapshot = copy.deepcopy(portfolio)

        # Build candidate set from eligible signals
        registry = build_candidate_set(
            engine, signals, profile_id, profile, portfolio, cycle_id
        )

        if registry.is_empty and PM_CANDIDATE_MODE == "enabled":
            notes = "Candidate-ID mode: no eligible candidates for this cycle."
            stored_notes = _store_pm_cycle_note(db, profile_id, notes)
            db.close()
            return {
                "decisions": executed,
                "portfolio_notes": stored_notes,
                "profile": profile_id,
            }

        if registry.is_empty:
            log.info(
                "Candidate-ID shadow mode produced no eligible candidates for "
                "profile=%s; continuing with legacy entry path",
                profile_id,
            )

        if not registry.is_empty:
            # Build PM prompt with candidate summaries
            candidate_summaries = registry.get_offered_summary()

            # Record offered candidates for audit (Requirement 12.1)
            for summary in candidate_summaries:
                _record_candidate_event(
                    engine, summary["candidate_id"], cycle_id, profile_id,
                    "offered",
                    {"symbol": summary["symbol"], "direction": summary["direction"], "risk_reward": summary.get("risk_reward")},
                )

            pm_prompt = build_candidate_pm_prompt(
                candidate_summaries, portfolio_snapshot, profile, profile_id
            )

            # Build dynamic schema for structured output
            schema = build_decision_schema(registry.get_registered_ids())

            # Call LLM with candidate prompt
            candidate_system_prompt = f"You are the {profile['name']} Portfolio Manager. {profile['personality']}"
            raw = call_llm(
                candidate_system_prompt,
                pm_prompt,
                json_mode=True,
                tier=tier,
                purpose=f"pm_candidate_entry:{profile_id}",
            )
            raw_response = parse_json_response(raw)

            # Parse decision contract
            candidate_metadata = registry.get_candidate_metadata()
            parse_result = parse_decision_contract(
                raw_response, registry.get_registered_ids(), candidate_metadata
            )

            # Handle contract retry (at most once)
            if should_retry_candidate_contract(parse_result):
                retry_prompt = build_candidate_retry_prompt(parse_result, registry)
                raw_retry = call_llm(
                    candidate_system_prompt,
                    retry_prompt,
                    json_mode=True,
                    tier=tier,
                    purpose=f"pm_candidate_retry:{profile_id}",
                )
                raw_response = parse_json_response(raw_retry)
                parse_result = parse_decision_contract(
                    raw_response, registry.get_registered_ids(), candidate_metadata
                )

            # Record PM decisions for audit (Requirements 12.2, 12.4)
            for decision in parse_result.accepted:
                _record_candidate_event(
                    engine, decision.candidate_id, cycle_id, profile_id,
                    "pm_accept",
                    {"rationale": decision.rationale, "risk_multiplier": decision.risk_multiplier},
                )
            for decision in parse_result.rejected:
                _record_candidate_event(
                    engine, decision.candidate_id, cycle_id, profile_id,
                    "pm_reject",
                    {"rationale": decision.rationale},
                )
            for ns_id in parse_result.not_selected_ids:
                _record_candidate_event(
                    engine, ns_id, cycle_id, profile_id,
                    "pm_not_selected",
                    None,
                )

            # Record PM rejection decisions
            if PM_CANDIDATE_MODE == "enabled":
                for decision in parse_result.rejected:
                    registry.mark_rejected(decision.candidate_id, decision.rationale)

                # Execute each accepted candidate through the pipeline
                for decision in parse_result.accepted:
                    pipeline_result = execute_candidate_pipeline(
                        db, engine, registry, decision,
                        portfolio_snapshot, profile, profile_id,
                        recovery_multiplier=1.0,
                    )

                    # Record pipeline result for audit (Requirements 12.3, 12.5, 12.6, 12.7)
                    _record_candidate_event(
                        engine, pipeline_result.candidate_id, cycle_id, profile_id,
                        f"pipeline_{pipeline_result.outcome}",
                        {
                            "outcome": pipeline_result.outcome,
                            "sizing": {
                                "quantity": pipeline_result.sizing_result.quantity,
                                "dollar_risk": pipeline_result.sizing_result.dollar_risk,
                            } if pipeline_result.sizing_result else None,
                            "gate_notes": pipeline_result.gate_notes,
                            "error": pipeline_result.error,
                        },
                    )

                    executed.append({
                        "symbol": pipeline_result.resolved_order.symbol if pipeline_result.resolved_order else "unknown",
                        "action": pipeline_result.resolved_order.action if pipeline_result.resolved_order else "unknown",
                        "executed": pipeline_result.outcome == "executed",
                        "outcome": pipeline_result.outcome,
                        "profile": profile_id,
                        "source": "candidate_pipeline",
                        "candidate_id": pipeline_result.candidate_id,
                    })
            else:
                # Shadow mode: dry-run candidate path, record hypothetical results
                from utils.trade_events import log_trade_event
                candidate_results = []
                for decision in parse_result.accepted:
                    pipeline_result = dry_run_candidate_pipeline(
                        db, engine, registry, decision,
                        portfolio_snapshot, profile, profile_id,
                        recovery_multiplier=1.0,
                    )

                    # Record shadow pipeline result for audit (Requirements 12.3, 12.5, 12.6, 12.7)
                    _record_candidate_event(
                        engine, pipeline_result.candidate_id, cycle_id, profile_id,
                        f"shadow_pipeline_{pipeline_result.outcome}",
                        {
                            "outcome": pipeline_result.outcome,
                            "sizing": {
                                "quantity": pipeline_result.sizing_result.quantity,
                                "dollar_risk": pipeline_result.sizing_result.dollar_risk,
                            } if pipeline_result.sizing_result else None,
                            "gate_notes": pipeline_result.gate_notes,
                            "error": pipeline_result.error,
                        },
                    )

                    candidate_results.append({
                        "candidate_id": pipeline_result.candidate_id,
                        "outcome": pipeline_result.outcome,
                        "symbol": pipeline_result.resolved_order.symbol if pipeline_result.resolved_order else None,
                        "action": pipeline_result.resolved_order.action if pipeline_result.resolved_order else None,
                        "quantity": pipeline_result.sizing_result.quantity if pipeline_result.sizing_result else None,
                    })

                # Record hypothetical PM rejections as events (no state mutation)
                for decision in parse_result.rejected:
                    log_trade_event(
                        db,
                        "pm_reject_hypothetical",
                        agent=f"pm_{profile_id}",
                        symbol="candidate_pipeline",
                        profile=profile_id,
                        message=f"Shadow mode: PM rejected candidate {decision.candidate_id}: {decision.rationale}",
                        payload={"candidate_id": decision.candidate_id, "rationale": decision.rationale},
                    )

                # Store candidate results for shadow comparison after legacy executes
                _shadow_candidate_results = candidate_results
                _shadow_parse_result = parse_result

            # Finalize cycle (assign terminal states to remaining candidates)
            registry.finalize_cycle()

            # In shadow mode, fall through to legacy entry path below
            if PM_CANDIDATE_MODE == "enabled":
                notes = f"Candidate-ID mode: processed {len(parse_result.accepted)} accepts, {len(parse_result.rejected)} rejects."
                stored_notes = _store_pm_cycle_note(db, profile_id, notes)
                db.close()
                return {
                    "decisions": executed,
                    "portfolio_notes": stored_notes,
                    "profile": profile_id,
                }
            # If shadow mode, continue to legacy path below...

    # Build profile-specific system prompt for entry decisions
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        profile_name=profile["name"],
        emoji=profile["emoji"],
        personality=profile["personality"],
        max_positions=profile["max_positions"],
        max_position_pct=int(profile["max_position_pct"] * 100),
        min_risk_reward=profile["min_risk_reward"],
        min_signal_strength=profile["min_signal_strength"],
        avoid_first_minutes=profile["avoid_first_minutes"],
        avoid_last_minutes=profile["avoid_last_minutes"],
        max_daily_loss_pct=int(profile["max_daily_loss_pct"] * 100),
        recovery_probe_contract=RECOVERY_PROBE_CONTRACT if profile_id == "moderate" else "",
    )

    # Pull weekly stance if available (written Sunday, applies Mon–Fri)
    from datetime import date, timedelta as td
    weekly_stance_mem = (
        db.query(AgentMemory)
        .filter_by(agent="weekly_prep", key=f"weekly_stance_{profile_id}")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    weekly_stance_text = ""
    if weekly_stance_mem:
        data = json.loads(weekly_stance_mem.value)
        week_str = data.get("week", "")
        if week_str >= (date.today() - td(days=6)).isoformat():
            weekly_stance_text = (
                f"\nWEEKLY STANCE (set Sunday):\n"
                f"  stance: {data.get('weekly_stance')}\n"
                f"  reason: {data.get('stance_reason')}\n"
                f"  size_adjustment: {data.get('size_adjustment') or 0:+.0%}\n"
                f"  signal_threshold: {data.get('signal_threshold_adjustment', 'normal')}\n"
                f"  avoid: {data.get('symbols_avoid', [])}\n"
                f"  favor: {data.get('symbols_favor', [])}\n"
                f"  short_bias: {data.get('symbols_short_bias', [])}\n"
                f"  notes: {data.get('notes', '')}"
            )

    # Query case library — find cases relevant to this profile's style
    case_context = {
        "market_regime": None,  # will match broadly
        "bias": "long",
    }
    relevant_cases = get_relevant_cases(engine, case_context, limit=5)
    cases_text = format_cases_digest_for_pm(relevant_cases)
    strategy_context = build_pm_strategy_context(engine)

    # Win rates by setup type — PM uses this to adjust sizing
    win_rates = get_win_rate_by_setup(engine)

    # Filter signals to only symbols without open positions (entry candidates),
    # excluding HOLD signals and signals below the profile's strength threshold.
    entry_signals = {
        sym: sig for sym, sig in signals.items()
        if sym not in held_symbols
        and sig.get("signal", "").upper() != "HOLD"
        and _meets_threshold(sig.get("strength", "weak"), profile["min_signal_strength"])
    }

    # Log filtered-out signals at DEBUG level for observability
    for sym, sig in signals.items():
        if sym in held_symbols:
            continue  # already excluded by held_symbols filter, not new
        direction = sig.get("signal", "").upper()
        strength = sig.get("strength", "weak")
        if direction == "HOLD":
            log.debug(
                "Signal filtered out: symbol=%s direction=%s strength=%s reason=HOLD_direction",
                sym, direction, strength,
            )
        elif not _meets_threshold(strength, profile["min_signal_strength"]):
            log.debug(
                "Signal filtered out: symbol=%s direction=%s strength=%s reason=below_threshold (requires %s)",
                sym, direction, strength, profile["min_signal_strength"],
            )

    filter_summary = summarize_entry_signal_filter(
        signals, held_symbols, profile["min_signal_strength"]
    )

    if not entry_signals:
        summary_text = format_entry_signal_filter_summary(filter_summary)
        log.info(
            "No eligible PM entry signals for profile=%s after filtering; %s; skipping entry LLM to avoid invented/malformed decisions",
            profile_id, summary_text,
        )
        notes = (
            "No eligible entry signals after filtering; skipped new-entry decision cycle. "
            f"Filter summary: {summary_text}"
        )
        stored_notes = _store_pm_cycle_note(db, profile_id, notes)
        db.close()
        return {
            "decisions": [],
            "portfolio_notes": stored_notes,
            "profile": profile_id,
        }

    # Filter win rates to only include setup types matching entry candidates' signals
    entry_setup_types = {sig.get("setup_type") for sig in entry_signals.values() if sig.get("setup_type")}
    if entry_setup_types:
        win_rates = [wr for wr in win_rates if wr.get("setup_type") in entry_setup_types]

    if win_rates:
        win_rate_lines = ["Setup type win rates from case library:"]
        for r in sorted(win_rates, key=lambda x: x["win_rate"], reverse=True):
            flag = " ⚠️ avoid or reduce size" if r["win_rate"] < 40 and r["total"] >= 5 else ""
            win_rate_lines.append(
                f"  {r['setup_type']}: {r['win_rate']}% ({r['wins']}/{r['total']}) "
                f"avg pnl {r['avg_pnl_pct'] or 0:+.1f}%{flag}"
            )
        win_rate_text = "\n".join(win_rate_lines)
    else:
        win_rate_text = "No setup win rate data yet."

    # Breaking news from news monitor
    news_mem = (
        db.query(AgentMemory)
        .filter_by(agent="news_monitor", key="breaking_news")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    news_text = news_mem.value if news_mem else "No breaking news"

    # Get behavioral parameters (auto-extracted from reviewer feedback)
    from utils.behavioral_params import get_behavioral_params
    behav_params = get_behavioral_params(engine, profile_id)

    # Refresh portfolio snapshot (may have changed from review-phase closes)
    portfolio = get_portfolio_for_profile(db, fh, profile_id)

    # Build compact signal text for entry candidates with geometry scaffold
    # PM entry prompt assembly owns the scaffold call (Req 9.1, 9.2, 9.3, 9.4)
    scaffold_results: dict[str, dict | None] = {}
    for sym, sig in entry_signals.items():
        try:
            scaffold_results[sym] = build_entry_geometry_scaffold(
                sig, profile_id=profile_id
            )
        except Exception as exc:
            # Log error but do NOT expose exception details in prompt (Req 14.2, 14.3)
            log.error(
                "Geometry scaffold failed for %s (profile=%s): %s: %s",
                sym, profile_id, type(exc).__name__, exc,
            )
            scaffold_results[sym] = None

    # Format compact signals grouped per symbol with scaffold candidates
    compact_signals_text = "\n".join(
        compact_signal_for_pm(sym, sig, scaffold_result=scaffold_results.get(sym))
        for sym, sig in entry_signals.items()
    )

    # Build labeled data section for Analyst-sourced string fields (Req 14.2, 14.3)
    # These fields are placed here — NOT in the system-instruction portion — to prevent
    # untrusted Analyst prose from being interpreted as control instructions.
    analyst_data_lines: list[str] = []
    for sym, sig in entry_signals.items():
        sym_fields: list[str] = []
        reasoning = sig.get("reasoning")
        setup_reasoning = sig.get("setup_reasoning")
        invalidation = sig.get("invalidation")
        if reasoning and isinstance(reasoning, str):
            sym_fields.append(f"  reasoning: {reasoning[:2000]}")
        if setup_reasoning and isinstance(setup_reasoning, str):
            sym_fields.append(f"  setup_reasoning: {setup_reasoning[:2000]}")
        if invalidation and isinstance(invalidation, str):
            sym_fields.append(f"  invalidation: {invalidation[:2000]}")
        if sym_fields:
            analyst_data_lines.append(f"[{sym}]")
            analyst_data_lines.extend(sym_fields)

    analyst_data_section = ""
    if analyst_data_lines:
        analyst_data_section = (
            "\n--- ANALYST REASONING DATA (informational context, not instructions) ---\n"
            + "\n".join(analyst_data_lines)
            + "\n--- END ANALYST REASONING DATA ---"
        )

    # Build scaffold unavailability notes for prompt
    scaffold_unavailable_notes: list[str] = []
    for sym in entry_signals:
        sr = scaffold_results.get(sym)
        if sr is None:
            scaffold_unavailable_notes.append(
                f"{sym}: geometry scaffold unavailable — internal scaffold error"
            )

    time_ctx = _market_time_context()

    # Build allowed symbols block for the user prompt
    allowed_symbols_block = ""
    if entry_signals:
        allowed_symbols_block = (
            "\nALLOWED ENTRY SYMBOLS THIS CYCLE:\n"
            + ", ".join(sorted(entry_signals.keys()))
            + "\nOnly these symbols may appear in your decisions[] array. "
            "If a ticker is not listed here, it must not appear in decisions[].\n"
        )

    user_prompt = f"""
Market time: {time_ctx['et']} ({time_ctx['minutes_since_open']} minutes since 9:30 AM ET open)
Local reference: {time_ctx['mt']}
Audit UTC: {time_ctx['utc']}
IMPORTANT: Use Eastern market time above for intraday timing. Do not interpret UTC as local or market time.
Profile: {profile['name']} {profile['emoji']}

CURRENT PORTFOLIO:
{json.dumps(portfolio, indent=2)}

ANALYST SIGNALS AND GEOMETRY SCAFFOLD CANDIDATES:
{compact_signals_text if compact_signals_text else "No entry signals"}
{chr(10).join(scaffold_unavailable_notes) if scaffold_unavailable_notes else ""}
{analyst_data_section}

EXECUTION FEEDBACK (your profile only):
{feedback_text}{weekly_stance_text}

META-REVIEWER RECOMMENDATIONS (system-level feedback for your profile):
{meta_text if meta_text else 'None yet'}

SETUP WIN RATES (from case library — use to adjust position sizing):
{win_rate_text}

STRATEGY RECOMMENDATIONS (from Quant Researcher):
{strategy_context}

RELEVANT PAST CASES:
{cases_text}

BREAKING NEWS (from news monitor):
{news_text}

POSITION HEALTH (from health monitor):
{health_text}

BEHAVIORAL ADJUSTMENTS (auto-extracted from feedback — applied to your decisions):
{json.dumps(behav_params, indent=2) if behav_params.get('notes') else 'No adjustments active'}
{allowed_symbols_block}
Make your trading decisions for this cycle.
NOTE: Open positions are managed by the two-tier review system. Only consider NEW entries here.
"""

    raw = call_llm(system_prompt, user_prompt, json_mode=True, tier=tier, purpose=f"pm_entry:{profile_id}")
    result = _enforce_pm_decision_log_contract(parse_json_response(raw))
    if isinstance(result.get("decisions"), list):
        result["decisions"] = _apply_scaffold_geometry_defaults(
            result["decisions"],
            scaffold_results,
        )

    # ── Entry Normalizer (before behavioral params) ──
    from utils.behavioral_params import apply_params_to_decision

    def _safe_live_quote(sym: str) -> float | None:
        try:
            quote = fh.get_quote(sym)
            price = quote.get("price", 0)
            return float(price) if price and float(price) > 0 else None
        except Exception as exc:
            log.warning("Live quote fetch failed for %s: %s", sym, exc)
            return None

    raw_decisions = result.get("decisions", [])
    norm_result = normalize_pm_entry_decisions(
        raw_decisions,
        entry_signals,
        get_live_quote=_safe_live_quote,
    )

    # ── Contract Validation Retry ──
    # NOTE: Do NOT log pm_decision_rejected yet — retry may resolve some rejections.
    final_orders = norm_result.orders
    final_notes = result.get("portfolio_notes", "")
    unresolved_rejections = norm_result.rejections  # default: all rejections unresolved
    corrected_rejections = []  # rejections that retry resolved
    retry_info = None

    if _should_retry_pm_contract(raw_decisions, norm_result):
        # Telemetry: retry triggered
        _log_retry_triggered(db, profile_id, norm_result, entry_signals)

        retry_system, retry_user = _build_pm_contract_retry_prompt(
            result, norm_result.rejections, entry_signals,
        )

        try:
            retry_raw = call_llm(
                retry_system, retry_user,
                json_mode=True, tier=tier,
                purpose=f"pm_contract_retry:{profile_id}",
            )
            retry_result = parse_json_response(retry_raw)
            retry_decisions = retry_result.get("decisions", [])
            retry_norm_result = normalize_pm_entry_decisions(
                retry_decisions, entry_signals, get_live_quote=_safe_live_quote,
            )

            if retry_norm_result.orders:
                # Deduplicate against original valid orders (by symbol)
                deduped_retry = _deduplicate_retry_orders(
                    norm_result.orders, retry_norm_result.orders,
                )
                final_orders = norm_result.orders + deduped_retry

                # Classify initial rejections: corrected vs unresolved
                retry_symbols = {o.order["symbol"].upper() for o in deduped_retry}
                corrected_rejections = [
                    r for r in norm_result.rejections
                    if r.raw_decision.get("symbol", "").upper().strip() in retry_symbols
                ]
                unresolved_rejections = [
                    r for r in norm_result.rejections
                    if r not in corrected_rejections
                ]

                # Merge notes
                retry_notes = retry_result.get("portfolio_notes", "")
                final_notes = _merge_retry_notes(final_notes, retry_notes)

                # Telemetry: retry succeeded
                _log_retry_succeeded(db, profile_id, norm_result, retry_norm_result, deduped_retry)
                retry_info = {"attempted": True, "succeeded": True, "corrected_count": len(corrected_rejections)}
            else:
                # Retry failed — keep original valid orders
                unresolved_rejections = norm_result.rejections
                corrected_rejections = []

                # Merge retry notes even on failure
                retry_notes = retry_result.get("portfolio_notes", "")
                final_notes = _merge_retry_notes(final_notes, retry_notes)

                # Telemetry: retry failed
                _log_retry_failed(db, profile_id, norm_result, retry_norm_result)
                retry_info = {"attempted": True, "succeeded": False, "corrected_count": 0}

        except Exception as exc:
            log.error("PM contract retry LLM call failed for %s: %s", profile_id, exc)
            # Emit pm_contract_retry_failed telemetry for the exception case
            log_trade_event(
                db,
                "pm_contract_retry_failed",
                agent="portfolio_manager",
                symbol=None,
                profile=profile_id,
                payload={
                    "profile": profile_id,
                    "initial_rejection_count": len(norm_result.rejections),
                    "retry_rejection_count": 0,
                    "retry_reason_counts": {},
                    "error": str(exc)[:500],
                },
            )
            # Fall through — final_orders remains norm_result.orders
            retry_info = {"attempted": True, "succeeded": False, "corrected_count": 0}

        # NOTE: Do NOT log retry_norm_result.rejections as pm_decision_rejected.
        # Retry's own malformed outputs are captured inside pm_contract_retry_failed
        # payload (retry_reason_counts). Logging them as normal rejections would
        # pollute dashboard counts with the retry attempt itself.

    # ── Now log rejection events with correct final disposition ──
    for rejection in unresolved_rejections:
        trade_event = _log_pm_decision_rejected(db, rejection, profile_id)
        db.flush()  # ensure trade_event.id is populated for shadow ledger linkage

        # ── Shadow ledger: record PM normalizer rejection ──
        raw = rejection.raw_decision
        sym = raw.get("symbol") if isinstance(raw, dict) else None
        if sym and isinstance(sym, str) and sym.strip():
            record_blocked_candidate(
                db,
                symbol=sym.strip(),
                action=raw.get("action", "UNKNOWN") if isinstance(raw, dict) else "UNKNOWN",
                blocked_by="pm_normalizer",
                block_reason=f"{rejection.reason_code}: {rejection.reason}",
                reason_code=rejection.reason_code,
                entry_price=_try_positive_float(raw.get("entry_price")),
                stop_price=_try_positive_float(
                    raw.get("stop_loss") or raw.get("stop_price") or raw.get("stop")
                ),
                target_price=_try_positive_float(
                    raw.get("target") or raw.get("target_price") or raw.get("profit_target")
                ),
                decision_snapshot=raw,
                source="pm_llm_decision",
                agent=f"pm_{profile_id}",
                trade_event_id=trade_event.id if trade_event else None,
            )

    for rejection in corrected_rejections:
        _log_pm_decision_corrected(db, rejection, profile_id)

    # Execute entry decisions — only final_orders proceed
    for norm_order in final_orders:
        decision = norm_order.order
        decision = apply_params_to_decision(decision, behav_params, profile)
        if decision.get("action") == "PASS":
            executed.append({
                **decision, "executed": False,
                "message": "Blocked by behavioral params",
                "profile": profile_id, "source": "entry_logic",
            })
            continue

        # Filter out CLOSE/HOLD actions for held symbols — those are handled
        # by the two-tier review system above (defense-in-depth; normalizer
        # only passes BUY/SHORT but keep as a no-op guard)
        action = decision.get("action", "").upper()
        sym = decision.get("symbol", "")

        # Guardrail: defense-in-depth symbol check (normalizer already validates
        # symbols against entry_signals, but keep as a safety net)
        if sym and sym not in set(symbols) and sym not in held_symbols:
            msg = f"Rejected invented/out-of-scope symbol: {sym}"
            log.warning(msg)
            executed.append({
                **decision, "executed": False,
                "message": msg,
                "profile": profile_id, "source": "entry_logic",
            })
            continue

        if action == "CLOSE" and sym in held_symbols:
            log.info(
                "Ignoring LLM CLOSE for %s — close decisions are handled by "
                "two-tier review system only.",
                sym,
            )
            continue
        if action == "HOLD":
            continue

        # ── Entry timing gate ──
        setup = decision.get("setup_type") or ""
        if not setup:
            signal_for_timing = _build_signal_for_symbol(db, sym, decision)
            setup = signal_for_timing.get("setup_type", "")
        max_minutes = ENTRY_WINDOW_LIMITS.get(setup)
        if max_minutes is not None:
            from pytz import timezone as _tz
            et_tz = _tz("America/New_York")
            now_et = datetime.now(et_tz)
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            minutes_since_open = (now_et - market_open).total_seconds() / 60
            if minutes_since_open > max_minutes:
                log.info(
                    "ENTRY TIMING GATE: %s %s blocked — %s entry window closed "
                    "(%d min since open, max %d)",
                    sym, setup, setup, int(minutes_since_open), max_minutes,
                )
                record_blocked_candidate(
                    db, symbol=sym, action=decision["action"],
                    blocked_by="entry_timing_gate",
                    block_reason=f"{setup} window closed ({int(minutes_since_open)} min, max {max_minutes})",
                    profile=profile_id, setup_type=setup,
                    entry_price=decision.get("entry_price"),
                    stop_price=decision.get("stop_loss") or decision.get("stop_price"),
                    target_price=decision.get("target") or decision.get("target_price"),
                    decision_snapshot=decision, signal_snapshot=entry_signals.get(sym),
                    source="analyst_signal", agent=f"pm_{profile_id}",
                )
                decision["action"] = "PASS"
                decision["rationale"] = (
                    f"Entry timing gate: {setup} window closed "
                    f"({int(minutes_since_open)} min since open, max {max_minutes})"
                )
                executed.append({
                    **decision, "executed": False,
                    "message": f"Blocked by entry timing gate",
                    "profile": profile_id, "source": "entry_logic",
                })
                continue

        # ── LessonRegistry pre-trade gate ──
        signal = _build_signal_for_symbol(db, sym, decision)
        verdict = check_track_record(
            engine,
            symbol=sym,
            bias=signal.get("bias", ""),
            setup_type=signal.get("setup_type", ""),
        )

        if verdict["verdict"] == "BLOCK":
            log.warning(
                "LESSON BLOCK: %s %s %s — avg_score=%.2f (n=%d). Trade blocked.",
                sym, signal.get("bias"), signal.get("setup_type"),
                verdict.get("avg_score_5") or verdict.get("avg_score_3") or 0,
                verdict["sample_size"],
            )
            avg_score = verdict.get("avg_score_5") or verdict.get("avg_score_3") or 0
            record_blocked_candidate(
                db, symbol=sym, action=action,
                blocked_by="track_record_gate",
                block_reason=f"Track record BLOCK: avg_score={avg_score}, sample_size={verdict['sample_size']}, match_type={verdict.get('match_type', '')}",
                profile=profile_id, setup_type=signal.get("setup_type", ""),
                gate_notes={
                    "avg_score_3": verdict.get("avg_score_3"),
                    "avg_score_5": verdict.get("avg_score_5"),
                    "sample_size": verdict.get("sample_size"),
                    "size_multiplier": verdict.get("size_multiplier"),
                    "match_type": verdict.get("match_type"),
                    "bias": verdict.get("bias"),
                    "setup_type": verdict.get("setup_type"),
                },
                decision_snapshot=decision, signal_snapshot=signal,
                source="analyst_signal", agent=f"pm_{profile_id}",
            )
            executed.append({
                **decision, "executed": False,
                "message": f"Blocked by LessonRegistry: {verdict['verdict']}",
                "profile": profile_id, "source": "entry_logic",
            })
            continue

        if verdict["verdict"] == "POOR_TRACK_RECORD":
            original_qty = _coerce_quantity(decision.get("quantity", 0), symbol=sym)
            decision["quantity"] = max(1, int(original_qty * verdict["size_multiplier"]))
            log.info(
                "LESSON WARNING: %s %s %s — avg_score=%.2f (n=%d). "
                "Reducing qty %d → %d.",
                sym, signal.get("bias"), signal.get("setup_type"),
                verdict.get("avg_score_3") or 0, verdict["sample_size"],
                original_qty, decision["quantity"],
            )

        ok, msg = execute_trade(db, decision, profile_id, normalized=True)
        executed.append({
            **decision, "executed": ok, "message": msg,
            "profile": profile_id, "source": "entry_logic",
        })

    # ── No-trade outcome logging (Req 12.1–12.6) ──
    # Categorize each no-trade outcome as exactly one of:
    # "scaffold_unavailable", "candidates_rejected_by_pm", "candidate_rejected_by_gate"
    #
    # Split executed into actual fills vs gate-blocked (needed for outcome categorization)
    actual_fills = [d for d in executed if d.get("executed")]
    gate_blocked_items = [d for d in executed if not d.get("executed")]
    filled_symbols = {str(d.get("symbol", "")).upper() for d in actual_fills if d.get("symbol")}

    # 1. scaffold_unavailable: scaffold returned None or status != "ok".
    # Do not log a no-trade outcome for a symbol that actually filled in this
    # same PM/profile cycle; the fill is the authoritative outcome and logging
    # both makes the decision log contradict itself.
    for sym in entry_signals:
        if str(sym).upper() in filled_symbols:
            continue
        sr = scaffold_results.get(sym)
        if sr is None:
            _log_no_trade_outcome(
                db, sym, profile_id, "scaffold_unavailable",
                scaffold_result=None,
            )
        elif isinstance(sr, dict) and sr.get("status") not in ("ok",):
            _log_no_trade_outcome(
                db, sym, profile_id, "scaffold_unavailable",
                scaffold_result=sr,
            )

    # 2. candidates_rejected_by_pm: PM explicitly rejected all candidates
    # Look for decision_type == "reject" in the raw decisions
    all_raw_decisions = result.get("decisions", [])
    for raw_dec in all_raw_decisions:
        if not isinstance(raw_dec, dict):
            continue
        if raw_dec.get("decision_type") == "reject":
            sym = raw_dec.get("symbol")
            if sym and isinstance(sym, str) and sym.strip():
                rationale = raw_dec.get("rationale", "")
                _log_no_trade_outcome(
                    db, sym.strip(), profile_id, "candidates_rejected_by_pm",
                    rejection_rationale=rationale if rationale else None,
                )

    # 3. candidate_rejected_by_gate: PM selected a candidate but a Gate rejected it
    # These are in gate_blocked_items where the message indicates gate rejection
    for blocked in gate_blocked_items:
        msg = blocked.get("message", "")
        sym = blocked.get("symbol", "")
        if not sym or ("Gate" not in msg and "gate" not in msg.lower()):
            continue
        # Extract gate name and reason from the message format:
        # "Gate rejected ({gate_name}): {reason}"
        gate_name = None
        gate_reason = msg
        import re
        gate_match = re.match(r"Gate rejected \(([^)]+)\): (.+)", msg)
        if gate_match:
            gate_name = gate_match.group(1)
            gate_reason = gate_match.group(2)

        # Find the scaffold candidate geometry from the decision
        scaffold_candidate = None
        candidate_id = blocked.get("geometry_candidate_id")
        candidate_name = blocked.get("geometry_candidate_name")
        sr = scaffold_results.get(sym)
        if sr and isinstance(sr, dict) and candidate_id:
            candidates = sr.get("candidates", [])
            for c in candidates:
                if isinstance(c, dict) and c.get("candidate_id") == candidate_id:
                    scaffold_candidate = c
                    break

        # Determine PM's final adjusted geometry if modified
        pm_adjusted_geometry = None
        if blocked.get("decision_type") == "adjust":
            pm_adjusted_geometry = {
                "entry_price": blocked.get("entry_price"),
                "stop_loss": blocked.get("stop_loss") or blocked.get("stop"),
                "target": blocked.get("target"),
            }

        _log_no_trade_outcome(
            db, sym, profile_id, "candidate_rejected_by_gate",
            gate_name=gate_name,
            gate_reason=gate_reason,
            scaffold_candidate=scaffold_candidate,
            pm_adjusted_geometry=pm_adjusted_geometry,
        )

    # Save PM notes with a post-validation execution audit.  The LLM's
    # portfolio_notes are written before validation/edge gates finish, so the
    # audit prevents the dashboard from implying rejected ideas became trades.
    notes = _compact_text_for_decision_log(final_notes, MAX_PM_PORTFOLIO_NOTE_CHARS)
    if notes or executed or unresolved_rejections or norm_result.non_orders:
        audit = _format_execution_audit(
            actual_fills,
            gate_blocked=gate_blocked_items,
            rejections=unresolved_rejections,
            non_orders=norm_result.non_orders,
            retry_info=retry_info,
        )
        stored_notes = f"{notes}\n\n{audit}" if notes else audit
        stored_notes = _store_pm_cycle_note(db, profile_id, stored_notes)
    else:
        stored_notes = notes

    # ── Shadow comparison recording (if shadow mode was active) ──
    if PM_CANDIDATE_MODE == "shadow" and '_shadow_candidate_results' in locals():
        try:
            from sqlalchemy import text as _text
            comparison_data = {
                "candidate_results_json": json.dumps(_shadow_candidate_results, default=str),
                "legacy_results_json": json.dumps(executed, default=str),
                "agreement_summary": _compute_shadow_agreement(
                    _shadow_candidate_results, executed
                ),
                "malformed_count": len(_shadow_parse_result.violations) if _shadow_parse_result else 0,
                "hypothetical_diffs": json.dumps({
                    "candidate_accepts": len(_shadow_parse_result.accepted) if _shadow_parse_result else 0,
                    "candidate_rejects": len(_shadow_parse_result.rejected) if _shadow_parse_result else 0,
                    "legacy_executed": len([e for e in executed if e.get("executed")]),
                }, default=str),
            }
            with engine.connect() as conn:
                conn.execute(
                    _text("""
                        INSERT INTO candidate_shadow_comparison
                        (cycle_id, profile_id, candidate_results_json, legacy_results_json,
                         agreement_summary, malformed_count, hypothetical_diffs)
                        VALUES (:cycle_id, :profile_id, :candidate_results_json,
                                :legacy_results_json, :agreement_summary,
                                :malformed_count, :hypothetical_diffs)
                    """),
                    {"cycle_id": cycle_id, "profile_id": profile_id, **comparison_data},
                )
                conn.commit()
        except Exception as exc:
            log.warning("Failed to record shadow comparison: %s", exc)

    db.close()
    return {"decisions": executed, "portfolio_notes": stored_notes, "profile": profile_id}


def run(engine, symbols: list[str]) -> dict:
    """Run all active PM profiles in sequence."""
    all_results = {}
    for profile_id in ACTIVE_PROFILES:
        all_results[profile_id] = run_profile(engine, symbols, profile_id)
    return all_results
