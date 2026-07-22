"""Deterministic market-state computation and conditional trigger models.

Single module containing all market-state classification logic. No LLM calls.
No external API calls. Operates only on data already present in the signal
enrichment pipeline.

See: requirements.md §2–§7, design.md §1–§15
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frozen Dataclass Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeframeAuthority:
    higher_timeframe_trend: str   # "bullish" | "bearish" | "neutral"
    intraday_trend: str           # "bullish" | "bearish" | "neutral"
    authority: str                # "higher_timeframe" | "intraday" | "aligned" | "confounded"
    conflict: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "higher_timeframe_trend": self.higher_timeframe_trend,
            "intraday_trend": self.intraday_trend,
            "authority": self.authority,
            "conflict": self.conflict,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SetupReclassification:
    original_setup_type: str
    reclassified_setup_type: str
    reason: str
    trade_posture: str  # flat|watch_long_trigger|watch_short_trigger|watch_retest|eligible_for_pm_review|veto_long|veto_short

    def to_dict(self) -> dict:
        return {
            "original_setup_type": self.original_setup_type,
            "reclassified_setup_type": self.reclassified_setup_type,
            "reason": self.reason,
            "trade_posture": self.trade_posture,
        }


@dataclass(frozen=True)
class IfThenTrigger:
    id: str
    condition: str
    threshold: float | None
    confirmation: str
    then: str
    trade_posture: str
    invalidates: str | None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "condition": self.condition,
            "threshold": self.threshold,
            "confirmation": self.confirmation,
            "then": self.then,
            "trade_posture": self.trade_posture,
            "invalidates": self.invalidates,
        }


@dataclass(frozen=True)
class MarketStateResult:
    market_state: str
    timeframe_authority: TimeframeAuthority
    setup_reclassification: SetupReclassification | None
    if_then_triggers: list[IfThenTrigger]
    setup_lifecycle_state: str
    veto_reason_category: str | None

    def to_dict(self) -> dict:
        return {
            "market_state": self.market_state,
            "timeframe_authority": self.timeframe_authority.to_dict(),
            "setup_reclassification": (
                self.setup_reclassification.to_dict()
                if self.setup_reclassification else None
            ),
            "if_then_triggers": [t.to_dict() for t in self.if_then_triggers],
            "setup_lifecycle_state": self.setup_lifecycle_state,
            "veto_reason_category": self.veto_reason_category,
        }


# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

EXTENDED_FROM_VWAP_THRESHOLD_PCT: float = 1.5
BREAKOUT_CONFIRMATION_DISTANCE_PCT: float = 0.10
PULLBACK_PROXIMITY_PCT: float = 3.0  # Only generate pullback trigger when within this % of support


# ---------------------------------------------------------------------------
# Enum Frozensets (closed enums)
# ---------------------------------------------------------------------------

# Valid market states (closed enum per Requirement 2.3)
VALID_MARKET_STATES: frozenset[str] = frozenset({
    "trend_aligned_breakout",
    "breakout_extended",
    "breakout_retest_watch",
    "compression_under_resistance",
    "counter_trend_retracement_under_resistance",
    "range_bound_churn",
    "confounded",
    "risk_off_suppression",
    "pullback_validating",
    "pullback_failed",
})

# Valid lifecycle states (closed enum per Requirement 7.2)
VALID_LIFECYCLE_STATES: frozenset[str] = frozenset({
    "no_setup",
    "early_watch",
    "compression_watch",
    "breakout_watch",
    "breakout_confirmed_wait_retest",
    "pullback_watch",
    "pullback_validating",
    "activation_pending",
    "activated_for_pm_review",
    "invalidated",
    "expired",
})

# Valid trade postures
VALID_TRADE_POSTURES: frozenset[str] = frozenset({
    "flat",
    "watch_long_trigger",
    "watch_short_trigger",
    "watch_retest",
    "eligible_for_pm_review",
    "veto_long",
    "veto_short",
})

# Lifecycle states eligible for watch-candidate creation (Design §10)
WATCHABLE_LIFECYCLE_STATES: frozenset[str] = frozenset({
    "compression_watch",
    "breakout_watch",
    "breakout_confirmed_wait_retest",
    "pullback_watch",
    "pullback_validating",
    "activation_pending",
})


# ---------------------------------------------------------------------------
# Timeframe Authority Computation (Design §5, Requirements 3.1–3.7)
# ---------------------------------------------------------------------------


def _compute_timeframe_authority(mtf_context: dict) -> TimeframeAuthority:
    """Compute timeframe authority from multi-timeframe context.

    Determines which timeframe (daily or intraday 5m) has directional
    authority based on trend agreement or conflict.

    Default: confounded with conflict=False when data missing/malformed.
    """
    try:
        timeframes = mtf_context.get("timeframes", {})
        daily = timeframes.get("daily", {})
        intraday_5m = timeframes.get("5m", {})

        # Higher timeframe = daily trend
        ht_trend = daily.get("trend")
        # Intraday = 5m trend
        id_trend = intraday_5m.get("trend")

        # Normalize None → "neutral"
        ht = ht_trend if ht_trend in ("bullish", "bearish") else "neutral"
        intra = id_trend if id_trend in ("bullish", "bearish") else "neutral"

        # Determine authority
        if ht == intra and ht != "neutral":
            authority = "aligned"
            conflict = False
            reason = f"Both timeframes agree: {ht}"
        elif ht == "neutral" and intra == "neutral":
            authority = "confounded"
            conflict = False
            reason = "Neither timeframe provides clear direction"
        elif ht != "neutral" and intra != "neutral" and ht != intra:
            authority = "higher_timeframe"
            conflict = True
            reason = f"Intraday {intra} conflicts with daily {ht}; daily controls"
        elif ht != "neutral" and intra == "neutral":
            authority = "higher_timeframe"
            conflict = False
            reason = f"Daily {ht} with flat intraday"
        else:  # ht neutral, intra directional
            authority = "intraday"
            conflict = False
            reason = f"Intraday {intra} with no daily trend"

        return TimeframeAuthority(
            higher_timeframe_trend=ht,
            intraday_trend=intra,
            authority=authority,
            conflict=conflict,
            reason=reason,
        )
    except Exception as exc:
        logger.warning("Timeframe authority computation failed: %s", exc)
        return TimeframeAuthority(
            higher_timeframe_trend="neutral",
            intraday_trend="neutral",
            authority="confounded",
            conflict=False,
            reason="computation_failed",
        )


# ---------------------------------------------------------------------------
# Market State Classification (Design §4, Requirements 2.1, 2.3, 2.5–2.7)
# ---------------------------------------------------------------------------


def _classify_market_state(
    signal: dict,
    quote: dict,
    indicators: dict,
    timeframe_authority: TimeframeAuthority,
    market_regime: str | None = None,
) -> str:
    """Classify market state from deterministic inputs.

    Decision tree:
    1. risk_off + bearish alignment → risk_off_suppression
    2. breakout confirmed → check extended/conflict/aligned
    3. breakout approaching → check aligned vs not
    4. pullback at_level/holding → check authority
    5. pullback failed → pullback_failed
    6. conflicted agreement → confounded
    7. no_trigger + mixed bias → range_bound_churn
    8. DEFAULT → confounded

    Always returns a valid member of VALID_MARKET_STATES.
    """
    try:
        trigger_status = signal.get("trigger_status", {})
        breakout = trigger_status.get("breakout", {})
        pullback = trigger_status.get("pullback", {})
        key_levels = signal.get("key_levels", {})

        # Get current price from quote
        current_price = quote.get("price") or signal.get("current_price")

        # Get directional alignment info from multi-timeframe context
        mtf_context = signal.get("multitimeframe_context", {})
        directional_alignment = mtf_context.get("directional_alignment", {})

        # 1. Risk-off suppression
        if market_regime == "risk_off" and directional_alignment.get("bias") == "bearish":
            return "risk_off_suppression"

        # 2. Breakout confirmed
        if breakout.get("status") == "confirmed":
            # Check extended from VWAP
            vwap = key_levels.get("vwap")
            if current_price and vwap and vwap > 0:
                vwap_distance_pct = ((current_price - vwap) / vwap) * 100
                if vwap_distance_pct > EXTENDED_FROM_VWAP_THRESHOLD_PCT:
                    return "breakout_extended"

            # Check timeframe conflict
            if timeframe_authority.conflict and timeframe_authority.authority == "higher_timeframe":
                return "counter_trend_retracement_under_resistance"

            return "trend_aligned_breakout"

        # 3. Breakout approaching
        if breakout.get("status") == "approaching":
            if timeframe_authority.authority == "aligned":
                return "breakout_retest_watch"
            return "compression_under_resistance"

        # 4. Pullback at level or holding
        pullback_status = pullback.get("status", "")
        if pullback_status in ("at_level", "holding_above_level"):
            if timeframe_authority.authority in ("aligned", "intraday"):
                return "pullback_validating"
            return "counter_trend_retracement_under_resistance"

        # 5. Pullback failed
        if pullback_status == "failed":
            return "pullback_failed"

        # 6. Conflicted agreement
        if directional_alignment.get("agreement") == "conflicted":
            return "confounded"

        # 7. No trigger + mixed bias
        trigger_main_status = trigger_status.get("status", "")
        if trigger_main_status == "no_trigger" and directional_alignment.get("bias") == "mixed":
            return "range_bound_churn"

        # 8. Default
        return "confounded"

    except Exception as exc:
        logger.warning("Market state classification failed: %s", exc)
        return "confounded"


# ---------------------------------------------------------------------------
# Setup Reclassification (Design §6, Requirements 4.1–4.6)
# ---------------------------------------------------------------------------


def _compute_setup_reclassification(
    signal: dict,
    market_state: str,
    timeframe_authority: TimeframeAuthority,
) -> SetupReclassification | None:
    """Compute setup reclassification when market state contradicts raw setup.

    Only fires when:
    1. Signal direction is LONG or SHORT (not HOLD — HOLD is already safe).
    2. Market state or timeframe authority contradicts the raw setup semantics.

    Returns SetupReclassification or None if no contradiction exists.
    Wrapped in try/except returning None on error (fail-open).
    """
    try:
        # Only fire for directional signals
        signal_direction = signal.get("signal", "").upper()
        if signal_direction not in ("LONG", "SHORT"):
            return None

        setup_type = (signal.get("setup_type") or "").lower()
        ht_trend = timeframe_authority.higher_timeframe_trend
        authority = timeframe_authority.authority

        # Rule 1: technical_breakout + HT bearish → counter_trend_retracement_under_resistance
        if setup_type == "technical_breakout" and ht_trend == "bearish":
            return SetupReclassification(
                original_setup_type=setup_type,
                reclassified_setup_type="counter_trend_retracement_under_resistance",
                reason="technical_breakout under bearish higher-timeframe trend",
                trade_posture="watch_retest",
            )

        # Rule 2: any breakout_* prefix + market_state is breakout_extended → breakout_extended
        if setup_type.startswith("breakout") and market_state == "breakout_extended":
            return SetupReclassification(
                original_setup_type=setup_type,
                reclassified_setup_type="breakout_extended",
                reason=f"{setup_type} extended beyond VWAP threshold",
                trade_posture="watch_retest",
            )

        # Rule 3: vwap_reclaim + HT bearish + market_state != trend_aligned_breakout
        if (
            setup_type == "vwap_reclaim"
            and ht_trend == "bearish"
            and market_state != "trend_aligned_breakout"
        ):
            return SetupReclassification(
                original_setup_type=setup_type,
                reclassified_setup_type="compression_under_resistance",
                reason="vwap_reclaim under bearish HT without confirmed breakout",
                trade_posture="watch_long_trigger",
            )

        # Rule 4: momentum_fade + HT bullish + pullback_validating
        if (
            setup_type == "momentum_fade"
            and ht_trend == "bullish"
            and market_state == "pullback_validating"
        ):
            return SetupReclassification(
                original_setup_type=setup_type,
                reclassified_setup_type="pullback_validating",
                reason="momentum_fade in bullish HT pullback context",
                trade_posture="eligible_for_pm_review",
            )

        # Rule 5: gap_and_go + HT bearish + authority == intraday
        if (
            setup_type == "gap_and_go"
            and ht_trend == "bearish"
            and authority == "intraday"
        ):
            return SetupReclassification(
                original_setup_type=setup_type,
                reclassified_setup_type="counter_trend_retracement_under_resistance",
                reason="gap_and_go counter to bearish HT with intraday-only authority",
                trade_posture="veto_long",
            )

        # No reclassification applies
        return None

    except Exception as exc:
        logger.warning("Setup reclassification computation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Veto Reason Categorization (Design §13, Requirements 13.1–13.5)
# ---------------------------------------------------------------------------


def _categorize_veto_reason(signal: dict) -> str | None:
    """Categorize LLM veto reason into a known category via keyword matching.

    Returns category string if keywords match, None otherwise.
    Case-insensitive matching.
    """
    try:
        veto_reason = (signal.get("llm_veto_reason") or "").lower()
        if not veto_reason:
            return None

        # Category → keywords mapping (checked in priority order)
        categories = [
            ("higher_timeframe_resistance", ["higher timeframe", "daily resistance", "weekly resistance", "ht bearish"]),
            ("counter_trend_move", ["counter trend", "counter-trend", "against trend", "opposing trend"]),
            ("extended_from_vwap", ["extended", "far from vwap", "vwap distance", "overextended"]),
            ("compression_no_trigger", ["compression", "squeeze", "no trigger", "waiting for break"]),
            ("range_bound", ["range bound", "range-bound", "chop", "no direction", "sideways"]),
            ("risk_off_regime", ["risk off", "risk-off", "defensive", "flight to safety"]),
        ]

        for category, keywords in categories:
            for kw in keywords:
                if kw in veto_reason:
                    return category

        return None

    except Exception as exc:
        logger.warning("Veto reason categorization failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# If-Then Trigger Computation (Design §7, Requirements 5.1–5.6)
# ---------------------------------------------------------------------------


def _compute_if_then_triggers(
    signal: dict,
    current_price: float | None,
    timeframe_authority: TimeframeAuthority,
) -> list[IfThenTrigger]:
    """Generate conditional if-then triggers based on current state.

    Produces up to 4 triggers (capped). Skips triggers when required
    levels are missing. Returns empty list on error (fail-open).
    """
    try:
        triggers: list[IfThenTrigger] = []
        key_levels = signal.get("key_levels", {})
        resistance = key_levels.get("resistance")
        support = key_levels.get("support")
        vwap = key_levels.get("vwap")

        if current_price is None:
            return triggers

        ht_trend = timeframe_authority.higher_timeframe_trend
        authority = timeframe_authority.authority

        # 1. Long Breakout trigger
        if resistance is not None and authority != "higher_timeframe":
            triggers.append(IfThenTrigger(
                id="long_breakout",
                condition=f"price > {resistance}",
                threshold=resistance,
                confirmation="5m close above resistance with volume",
                then="Long entry triggered",
                trade_posture="watch_long_trigger",
                invalidates=None,
            ))

        # 2. Pullback Hold trigger (only when within proximity of support)
        if support is not None and authority in ("aligned", "intraday"):
            if current_price > 0 and support > 0:
                distance_pct = ((current_price - support) / support) * 100
                if 0 <= distance_pct <= PULLBACK_PROXIMITY_PCT:
                    triggers.append(IfThenTrigger(
                        id="pullback_hold",
                        condition=f"price holds above {support}",
                        threshold=support,
                        confirmation="Hold above support for 2 candles",
                        then="Pullback continuation entry",
                        trade_posture="watch_long_trigger",
                        invalidates=f"price closes below {support}",
                    ))

        # 3. VWAP Veto trigger (if extended beyond threshold)
        if vwap is not None and vwap > 0:
            vwap_distance_pct = ((current_price - vwap) / vwap) * 100
            if vwap_distance_pct > EXTENDED_FROM_VWAP_THRESHOLD_PCT:
                triggers.append(IfThenTrigger(
                    id="vwap_veto",
                    condition=f"price extended > {EXTENDED_FROM_VWAP_THRESHOLD_PCT}% from VWAP",
                    threshold=vwap,
                    confirmation="N/A - immediate veto",
                    then="Veto long entry until VWAP reversion",
                    trade_posture="veto_long",
                    invalidates="price returns within VWAP band",
                ))

        # 4. Short Rejection trigger (HT bearish + price near resistance)
        if ht_trend == "bearish" and resistance is not None:
            triggers.append(IfThenTrigger(
                id="short_rejection",
                condition=f"rejection at {resistance}",
                threshold=resistance,
                confirmation="Rejection candle at resistance",
                then="Short entry on failed breakout",
                trade_posture="watch_short_trigger",
                invalidates=f"Clean break and hold above {resistance}",
            ))

        # Cap at 4 triggers
        return triggers[:4]

    except Exception as exc:
        logger.warning("If-then trigger computation failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Setup Lifecycle Classification (Design §8, Requirements 7.1–7.6)
# ---------------------------------------------------------------------------


def _classify_setup_lifecycle(
    market_state: str,
    signal_direction: str,
    timeframe_authority: TimeframeAuthority,
) -> str:
    """Classify setup lifecycle state from market state and signal context.

    Returns a valid member of VALID_LIFECYCLE_STATES.
    Default: 'no_setup' (fail-safe).
    """
    try:
        direction = signal_direction.upper() if signal_direction else ""
        is_directional = direction in ("LONG", "SHORT")
        authority = timeframe_authority.authority

        # States that always map to no_setup
        if market_state in ("confounded", "risk_off_suppression", "range_bound_churn"):
            return "no_setup"

        # compression_under_resistance
        if market_state == "compression_under_resistance":
            return "compression_watch" if is_directional else "no_setup"

        # breakout_retest_watch
        if market_state == "breakout_retest_watch":
            if is_directional and authority == "aligned":
                return "breakout_watch"
            elif is_directional:
                return "compression_watch"
            return "no_setup"

        # trend_aligned_breakout
        if market_state == "trend_aligned_breakout":
            if is_directional and authority in ("aligned", "intraday"):
                return "activation_pending"
            return "breakout_watch"

        # breakout_extended
        if market_state == "breakout_extended":
            return "breakout_confirmed_wait_retest"

        # counter_trend_retracement_under_resistance
        if market_state == "counter_trend_retracement_under_resistance":
            return "early_watch"

        # pullback_validating
        if market_state == "pullback_validating":
            if is_directional and authority in ("aligned", "intraday"):
                return "pullback_validating"
            return "pullback_watch"

        # pullback_failed
        if market_state == "pullback_failed":
            return "invalidated"

        # Default
        return "no_setup"

    except Exception as exc:
        logger.warning("Setup lifecycle classification failed: %s", exc)
        return "no_setup"


# ---------------------------------------------------------------------------
# Public Entry Point (Design §1, §12, Requirements 1.5, 2.6, 14.2)
# ---------------------------------------------------------------------------


def compute_market_state(
    signal: dict,
    quote: dict,
    indicators: dict,
    market_regime: str | None = None,
) -> MarketStateResult:
    """Compute full market-state analysis for a signal.

    Orchestrates:
    1. Timeframe authority computation
    2. Market state classification
    3. Setup reclassification (if applicable)
    4. If-then trigger generation
    5. Setup lifecycle classification
    6. Veto reason categorization

    Fail-open: returns safe defaults (confounded, no_setup, no triggers)
    on any unexpected failure. Never raises.
    """
    try:
        # 1. Compute timeframe authority
        mtf_context = signal.get("multitimeframe_context", {})
        timeframe_authority = _compute_timeframe_authority(mtf_context)

        # 2. Classify market state
        market_state = _classify_market_state(
            signal, quote, indicators, timeframe_authority, market_regime
        )

        # 3. Compute setup reclassification
        setup_reclassification = _compute_setup_reclassification(
            signal, market_state, timeframe_authority
        )

        # 4. Compute if-then triggers
        current_price = quote.get("price") or signal.get("current_price")
        if_then_triggers = _compute_if_then_triggers(
            signal, current_price, timeframe_authority
        )

        # 5. Classify setup lifecycle
        signal_direction = signal.get("signal", "")
        setup_lifecycle_state = _classify_setup_lifecycle(
            market_state, signal_direction, timeframe_authority
        )

        # 6. Categorize veto reason
        veto_reason_category = _categorize_veto_reason(signal)

        return MarketStateResult(
            market_state=market_state,
            timeframe_authority=timeframe_authority,
            setup_reclassification=setup_reclassification,
            if_then_triggers=if_then_triggers,
            setup_lifecycle_state=setup_lifecycle_state,
            veto_reason_category=veto_reason_category,
        )

    except Exception as exc:
        logger.warning("compute_market_state() failed: %s; returning safe defaults", exc)
        safe_authority = TimeframeAuthority(
            higher_timeframe_trend="neutral",
            intraday_trend="neutral",
            authority="confounded",
            conflict=False,
            reason="computation_failed",
        )
        return MarketStateResult(
            market_state="confounded",
            timeframe_authority=safe_authority,
            setup_reclassification=None,
            if_then_triggers=[],
            setup_lifecycle_state="no_setup",
            veto_reason_category=None,
        )
