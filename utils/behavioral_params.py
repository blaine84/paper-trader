"""
Behavioral Parameters
Converts reviewer feedback into executable parameter adjustments.
The Reviewer writes these. The PM reads and applies them before deciding.

Parameters are per-profile and override/modify the base profile settings.
Stored in agent_memory as structured JSON, not prose.
"""

import json
import logging
from datetime import datetime, timedelta
from db.schema import AgentMemory, get_session

log = logging.getLogger(__name__)

# Default behavioral params (no adjustments)
DEFAULT_PARAMS = {
    "entry_offset_pct": 0.0,          # adjust entry price (negative = earlier/more aggressive)
    "size_multiplier": 1.0,           # scale position size (0.5 = half size, 1.5 = 150%)
    "reduce_size_on_low_confidence": False,  # halve size when confidence is "low"
    "min_r_override": None,           # override profile's min R:R (e.g. 2.5)
    "avoid_setups": [],               # setup types to skip entirely
    "favor_setups": [],               # setup types to boost confidence on
    "max_hold_override_min": None,    # override max hold time
    "stop_buffer_pct": 0.0,           # widen/tighten stops (positive = wider)
    "avoid_first_minutes_override": None,  # override avoid-first-minutes
    "avoid_last_minutes_override": None,   # override avoid-last-minutes
    "notes": "",                       # human-readable summary of why
    "updated_at": None,
}


def get_behavioral_params(engine, profile_id: str) -> dict:
    """Get current behavioral parameters for a profile."""
    db = get_session(engine)
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="reviewer", key=f"behavioral_params_{profile_id}")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    db.close()

    if mem:
        try:
            params = json.loads(mem.value)
            # Merge with defaults so new fields are always present
            merged = {**DEFAULT_PARAMS, **params}
            return merged
        except Exception:
            pass
    return DEFAULT_PARAMS.copy()


def store_behavioral_params(engine, profile_id: str, params: dict):
    """Store updated behavioral parameters for a profile."""
    params["updated_at"] = datetime.utcnow().isoformat()
    db = get_session(engine)
    db.add(AgentMemory(
        agent="reviewer",
        key=f"behavioral_params_{profile_id}",
        value=json.dumps(params),
    ))
    db.commit()
    db.close()
    log.info(f"Behavioral params updated for {profile_id}: {json.dumps(params, indent=2)}")


def apply_params_to_decision(decision: dict, params: dict, profile: dict) -> dict:
    """
    Apply behavioral parameters to a trade decision before execution.
    Modifies the decision dict in place and returns it.
    """
    if not decision.get("action") or decision["action"] == "CLOSE":
        return decision

    symbol = decision.get("symbol", "")
    setup = decision.get("setup_type") or decision.get("setup") or ""
    confidence = decision.get("confidence", "medium")

    # Block avoided setups
    if setup in params.get("avoid_setups", []):
        log.info(f"Behavioral: blocking {symbol} — setup '{setup}' is on avoid list")
        decision["action"] = "PASS"
        decision["rationale"] = f"Blocked by behavioral params: {setup} on avoid list"
        return decision

    # Entry offset
    offset = params.get("entry_offset_pct", 0.0)
    if offset and decision.get("price"):
        original = decision["price"]
        decision["price"] = round(original * (1 + offset / 100), 2)
        if offset != 0:
            log.info(f"Behavioral: {symbol} entry adjusted {offset:+.1f}% (${original} → ${decision['price']})")

    # Size multiplier
    size_mult = params.get("size_multiplier", 1.0)
    if size_mult != 1.0 and decision.get("quantity"):
        original_qty = decision["quantity"]
        decision["quantity"] = max(1, int(original_qty * size_mult))
        log.info(f"Behavioral: {symbol} size adjusted {size_mult:.0%} ({original_qty} → {decision['quantity']})")

    # Reduce size on low confidence
    if params.get("reduce_size_on_low_confidence") and confidence == "low":
        if decision.get("quantity"):
            original_qty = decision["quantity"]
            decision["quantity"] = max(1, int(original_qty * 0.5))
            log.info(f"Behavioral: {symbol} size halved (low confidence) ({original_qty} → {decision['quantity']})")

    # Favor setups — boost by increasing quantity slightly
    if setup in params.get("favor_setups", []):
        if decision.get("quantity"):
            original_qty = decision["quantity"]
            decision["quantity"] = int(original_qty * 1.2)
            log.info(f"Behavioral: {symbol} size boosted 20% (favored setup '{setup}')")

    # Stop buffer
    stop_buffer = params.get("stop_buffer_pct", 0.0)
    if stop_buffer and decision.get("stop"):
        original_stop = decision["stop"]
        if decision.get("action") == "BUY":
            decision["stop"] = round(original_stop * (1 - stop_buffer / 100), 2)  # wider = lower for long
        elif decision.get("action") == "SHORT":
            decision["stop"] = round(original_stop * (1 + stop_buffer / 100), 2)  # wider = higher for short
        if stop_buffer != 0:
            log.info(f"Behavioral: {symbol} stop adjusted {stop_buffer:+.1f}% (${original_stop} → ${decision['stop']})")

    # Min R:R override
    min_rr = params.get("min_r_override")
    if min_rr and decision.get("price") and decision.get("stop") and decision.get("target"):
        price = decision["price"]
        stop = decision["stop"]
        target = decision["target"]
        if decision["action"] == "BUY":
            risk = price - stop
            reward = target - price
        else:
            risk = stop - price
            reward = price - target
        if risk > 0:
            rr = reward / risk
            if rr < min_rr:
                log.info(f"Behavioral: {symbol} R:R {rr:.1f} below override minimum {min_rr} — blocking")
                decision["action"] = "PASS"
                decision["rationale"] = f"Blocked by behavioral params: R:R {rr:.1f} < {min_rr} minimum"

    return decision


# ─── REVIEWER INTEGRATION ─────────────────────────────────────────────────────

PARAM_EXTRACTION_PROMPT = """You are converting trade execution feedback into specific parameter adjustments.

Given the feedback text for a PM profile, extract concrete behavioral changes.

Respond in JSON:
{
  "entry_offset_pct": float (-1.0 to +1.0, negative = enter earlier/more aggressively),
  "size_multiplier": float (0.5 to 1.5, 1.0 = no change),
  "reduce_size_on_low_confidence": bool,
  "min_r_override": float or null (e.g. 2.5 to require higher R:R),
  "avoid_setups": ["setup_type1", ...] or [],
  "favor_setups": ["setup_type1", ...] or [],
  "stop_buffer_pct": float (-1.0 to +2.0, positive = wider stops),
  "avoid_first_minutes_override": int or null,
  "avoid_last_minutes_override": int or null,
  "notes": "one sentence summary of changes"
}

Only change parameters where the feedback clearly indicates a problem.
Leave everything else at default (0, 1.0, false, null, []).
Be conservative — small adjustments, not dramatic swings.
"""


def extract_params_from_feedback(engine, profile_id: str, feedback_text: str) -> dict:
    """Use LLM to convert prose feedback into behavioral parameters."""
    from utils.llm import call_llm, parse_json_response

    prompt = f"""
Profile: {profile_id}
Current feedback:
{feedback_text}

Extract behavioral parameter adjustments from this feedback.
"""

    try:
        raw = call_llm(PARAM_EXTRACTION_PROMPT, prompt, json_mode=True, tier="medium")
        params = parse_json_response(raw)
        # Validate types
        params["entry_offset_pct"] = float(params.get("entry_offset_pct", 0))
        params["size_multiplier"] = max(0.25, min(2.0, float(params.get("size_multiplier", 1.0))))
        params["reduce_size_on_low_confidence"] = bool(params.get("reduce_size_on_low_confidence", False))
        params["avoid_setups"] = list(params.get("avoid_setups", []))
        params["favor_setups"] = list(params.get("favor_setups", []))
        params["stop_buffer_pct"] = float(params.get("stop_buffer_pct", 0))
        store_behavioral_params(engine, profile_id, params)
        return params
    except Exception as e:
        log.error(f"Failed to extract behavioral params for {profile_id}: {e}")
        return DEFAULT_PARAMS.copy()
