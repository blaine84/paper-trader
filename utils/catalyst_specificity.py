"""Catalyst Specificity Gate — config loader and scoring utilities.

Loads externalized alias and relationship maps from JSON, with hardcoded
fallback and cached reload support.  The gate evaluates news-driven trade
candidates on catalyst relevance to the traded symbol.

See: .kiro/specs/catalyst-specificity-gate/design.md §Component 3
"""

import json
import logging
import os
import re
from typing import Optional

from utils.gate_config import (
    CATALYST_SPECIFICITY_PROFILE_THRESHOLDS,
    CATALYST_SPECIFICITY_SECTOR_SYMPATHY_SIZE_MULTIPLIER,
)
from utils.trade_events import log_trade_event

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level config cache (lazy-loaded on first call)
# ---------------------------------------------------------------------------

_CONFIG_CACHE: dict | None = None

# ---------------------------------------------------------------------------
# Default config path — co-located JSON file
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "catalyst_config.json")

# ---------------------------------------------------------------------------
# Hardcoded defaults — used when config file is missing or malformed
# ---------------------------------------------------------------------------

_HARDCODED_DEFAULTS: dict = {
    "symbol_aliases": {
        "AMD": ["AMD", "Advanced Micro Devices"],
        "NVDA": ["NVDA", "Nvidia", "NVIDIA"],
        "TSLA": ["TSLA", "Tesla"],
        "MSFT": ["MSFT", "Microsoft"],
        "SPY": ["SPY", "S&P 500", "S&P", "large-cap index"],
        "QQQ": ["QQQ", "Nasdaq", "Nasdaq 100", "NDX"],
        "IWM": ["IWM", "Russell 2000", "small caps"],
        "DIA": ["DIA", "Dow", "Dow Jones"],
        "TLT": ["TLT", "Treasury", "Treasuries", "yields", "rates"],
        "GLD": ["GLD", "gold", "bullion"],
        "XLK": ["XLK", "technology sector", "tech sector"],
        "XLF": ["XLF", "financials", "banks"],
        "XLE": ["XLE", "energy sector", "oil", "gas"],
    },
    "readthrough_relationships": {
        "NVDA": ["Lumentum", "TSMC", "Super Micro", "Dell", "Microsoft", "Meta", "Amazon", "Google"],
        "AMD": ["TSMC", "Super Micro", "Microsoft", "Meta", "Amazon", "Google"],
        "TSLA": ["Panasonic", "BYD", "CATL", "lithium", "EV deliveries"],
        "MSFT": ["OpenAI", "Azure", "LinkedIn", "Activision"],
        "XLE": ["WTI", "Brent", "OPEC", "oil inventory", "crude"],
        "TLT": ["Treasury yields", "Fed", "CPI", "PCE", "jobs report"],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def should_apply_gate(decision: dict, signal: dict | None = None) -> bool:
    """Determine whether the catalyst specificity gate should evaluate this candidate.

    Returns True if the merged decision+signal context has news-driven
    characteristics that warrant catalyst specificity scoring.  Returns False
    for purely technical setups.

    For gap_and_go setup_type: requires EXPLICIT catalyst fields (catalyst_type
    or catalyst) — generic momentum/opening-move language in rationale does NOT
    trigger the gate.
    """
    NEWS_SETUP_TYPES = {
        "news_breakout",
        "news_catalyst",
        "news_catalyst_breakout",
        "catalyst_breakout",
        "gap_and_go",
    }
    # Only strong news-specific terms — exclude generic momentum words
    # that appear in technical setups (momentum, breakout, opening, move)
    NEWS_TERMS = {
        "catalyst",
        "headline",
        "upgrade",
        "downgrade",
        "earnings",
        "guidance",
        "contract",
        "customer",
        "supplier",
        "fda",
        "regulatory",
        "macro shock",
    }

    # Merge setup_type from decision and signal
    setup_type = (
        decision.get("setup_type")
        or (signal.get("setup_type") if signal else None)
        or ""
    ).lower()

    # Check setup_type match
    if any(ns in setup_type for ns in NEWS_SETUP_TYPES):
        # For gap_and_go, require EXPLICIT catalyst fields present
        # Do NOT gate gap_and_go just because rationale says "momentum"
        if "gap_and_go" in setup_type:
            has_catalyst = (
                decision.get("catalyst_type")
                or decision.get("catalyst")
                or (signal.get("catalyst_type") if signal else None)
                or (signal.get("catalyst") if signal else None)
            )
            if not has_catalyst:
                return False
        return True

    # Check catalyst_type exists (from decision OR signal)
    catalyst_type = decision.get("catalyst_type") or (
        signal.get("catalyst_type") if signal else None
    )
    if catalyst_type:
        return True

    # Check rationale/thesis for news terms
    text = " ".join(
        [
            decision.get("rationale", ""),
            decision.get("thesis", ""),
        ]
    ).lower()

    if any(term in text for term in NEWS_TERMS):
        return True

    return False


def load_catalyst_config(
    config_path: str | None = None, force_reload: bool = False
) -> dict:
    """Load symbol aliases and readthrough relationships from JSON config.

    Returns {"aliases": {...}, "relationships": {...}}.

    Uses a module-level cache.  First call loads from disk; subsequent calls
    return the cached result unless *force_reload=True*.  This makes tests
    and config hot-reloads easy without paying disk I/O per evaluation.

    Falls back to hardcoded defaults if the file is missing or malformed,
    logging a warning on fallback.
    """
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    path = config_path or _DEFAULT_CONFIG_PATH

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Validate expected structure
        aliases = raw.get("symbol_aliases")
        relationships = raw.get("readthrough_relationships")

        if not isinstance(aliases, dict) or not isinstance(relationships, dict):
            raise ValueError(
                "Config must contain 'symbol_aliases' and 'readthrough_relationships' dicts"
            )

        _CONFIG_CACHE = {
            "aliases": aliases,
            "relationships": relationships,
        }

    except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError) as exc:
        log.warning(
            "Catalyst config load failed (%s: %s); using hardcoded defaults.",
            type(exc).__name__,
            exc,
        )
        _CONFIG_CACHE = {
            "aliases": _HARDCODED_DEFAULTS["symbol_aliases"],
            "relationships": _HARDCODED_DEFAULTS["readthrough_relationships"],
        }

    return _CONFIG_CACHE


# ---------------------------------------------------------------------------
# Helper functions for text extraction and matching
# ---------------------------------------------------------------------------

# Sector/theme keyword mapping for contains_sector_terms
_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "XLK": ["technology", "tech", "software", "semiconductor", "chip", "AI", "cloud"],
    "XLF": ["financials", "banks", "banking", "insurance", "lending", "credit"],
    "XLE": ["energy", "oil", "gas", "crude", "petroleum", "drilling", "refining"],
    "NVDA": ["semiconductor", "chip", "AI", "GPU", "data center", "gaming"],
    "AMD": ["semiconductor", "chip", "AI", "CPU", "GPU", "data center"],
    "TSLA": ["EV", "electric vehicle", "autonomous", "battery", "solar", "energy storage"],
    "MSFT": ["cloud", "software", "AI", "enterprise", "Azure"],
    "SPY": ["market", "equities", "large-cap", "S&P"],
    "QQQ": ["tech", "Nasdaq", "growth", "large-cap tech"],
    "IWM": ["small-cap", "Russell", "domestic"],
    "DIA": ["blue-chip", "industrial", "Dow"],
    "TLT": ["bonds", "Treasury", "yields", "rates", "fixed income", "duration"],
    "GLD": ["gold", "precious metals", "bullion", "safe haven", "inflation hedge"],
}

# Macro terms for is_macro_catalyst
_MACRO_TERMS: list[str] = [
    "fed", "cpi", "pce", "jobs report", "fomc", "gdp", "inflation",
    "interest rate", "treasury", "yields", "nonfarm", "unemployment",
    "federal reserve", "rate decision", "rate cut", "rate hike",
]

# Direction inference keywords — CONSERVATIVE, only clear unambiguous words
_BULLISH_KEYWORDS: list[str] = [
    "beat", "raises", "upgrade", "approval", "contract win",
    "strong demand", "accelerating", "record revenue",
]
_BEARISH_KEYWORDS: list[str] = [
    "downgrade", "miss", "cut", "probe", "recall",
    "weak", "warning", "layoffs", "investigation",
]


def extract_catalyst_text(decision: dict, signal: dict | None = None) -> str:
    """Gather all catalyst-related text fields from merged context.

    Decision fields take priority; signal provides fallbacks.
    Gathers from: catalyst, rationale, thesis, news_catalyst fields.
    """
    parts: list[str] = []

    # Fields to extract (decision overrides signal)
    text_fields = ["catalyst", "rationale", "thesis", "news_catalyst"]

    for field in text_fields:
        # Decision value takes priority
        val = decision.get(field)
        if val and isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif signal:
            # Fall back to signal
            sig_val = signal.get(field)
            if sig_val and isinstance(sig_val, str) and sig_val.strip():
                parts.append(sig_val.strip())

    return " ".join(parts)


def any_mention(text: str, names: list[str]) -> bool:
    """Case-insensitive substring match — returns True if any name appears in text.

    Returns False if text is empty or names is empty.
    """
    if not text or not names:
        return False
    text_lower = text.lower()
    return any(name.lower() in text_lower for name in names)


def contains_sector_terms(text: str, symbol: str) -> bool:
    """Check for sector/theme keywords relevant to the given symbol.

    Uses the _SECTOR_KEYWORDS mapping to find sector terms for the symbol.
    Returns True if any sector keyword for the symbol is found in the text.
    """
    if not text or not symbol:
        return False

    keywords = _SECTOR_KEYWORDS.get(symbol.upper(), [])
    if not keywords:
        return False

    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def assess_freshness(decision: dict, signal: dict | None = None) -> str:
    """Determine catalyst freshness: 'intraday', 'same_day', or 'stale'.

    Checks for timestamp fields and freshness keywords in the merged context.
    - 'intraday': explicit intraday/today keywords or fresh timestamp indicators
    - 'same_day': same-day but unclear timing
    - 'stale': yesterday/last week or no freshness indicators
    """
    # Gather freshness-relevant fields
    catalyst_type = decision.get("catalyst_type", "") or ""
    catalyst_text = extract_catalyst_text(decision, signal)
    text_lower = catalyst_text.lower() + " " + catalyst_type.lower()

    # Check for timestamp fields in signal
    quote_timestamp = None
    if signal:
        quote_timestamp = signal.get("quote_timestamp")

    # Check for explicit intraday indicators
    intraday_terms = [
        "intraday", "today", "this morning", "just reported",
        "breaking", "just announced", "moments ago", "pre-market today",
        "after hours today",
    ]
    stale_terms = [
        "yesterday", "last week", "last month", "days ago",
        "weeks ago", "prior session", "previous day",
    ]

    has_intraday = any(term in text_lower for term in intraday_terms)
    has_stale = any(term in text_lower for term in stale_terms)

    # If quote_timestamp is present, that's a strong intraday signal
    if quote_timestamp:
        has_intraday = True

    # Determine freshness
    if has_stale and not has_intraday:
        return "stale"
    elif has_intraday:
        return "intraday"
    else:
        # No clear indicators either way — default to same_day
        return "same_day"


def extract_indicators(decision: dict, signal: dict | None = None) -> dict:
    """Merge indicator dicts from decision and signal (decision takes priority).

    Gracefully handles missing volume/price data by returning an empty dict
    or partial dict with whatever is available.
    """
    indicators: dict = {}

    # Start with signal indicators as base
    if signal:
        sig_indicators = signal.get("indicators")
        if isinstance(sig_indicators, dict):
            indicators.update(sig_indicators)
        # Also pull top-level quote fields from signal
        for field in [
            "relative_volume", "volume_ratio", "current_price",
            "change_pct", "day_high", "day_low", "prev_close",
        ]:
            val = signal.get(field)
            if val is not None:
                indicators.setdefault(field, val)

    # Decision indicators override signal
    dec_indicators = decision.get("indicators")
    if isinstance(dec_indicators, dict):
        indicators.update(dec_indicators)

    return indicators


def infer_catalyst_direction(text: str) -> Optional[str]:
    """CONSERVATIVE direction inference — only classify clear unambiguous words.

    Returns:
        "BULLISH" — if only bullish keywords found (no bearish)
        "BEARISH" — if only bearish keywords found (no bullish)
        None — if ambiguous, mixed, or no directional keywords found

    False conflicts are expensive (they apply -3 penalty), so this function
    errs heavily on the side of returning None.
    """
    if not text:
        return None

    text_lower = text.lower()

    has_bullish = any(kw in text_lower for kw in _BULLISH_KEYWORDS)
    has_bearish = any(kw in text_lower for kw in _BEARISH_KEYWORDS)

    # Mixed signals → ambiguous → None
    if has_bullish and has_bearish:
        return None

    if has_bullish:
        return "BULLISH"
    if has_bearish:
        return "BEARISH"

    # No clear directional keywords
    return None


def directions_match(trade_dir: str, catalyst_dir: Optional[str]) -> bool:
    """Check if trade direction and catalyst direction are aligned.

    trade_dir: typically "LONG"/"BUY" (bullish) or "SHORT"/"SELL" (bearish)
    catalyst_dir: "BULLISH", "BEARISH", or None

    Returns False if either direction is None/empty.
    """
    if not trade_dir or not catalyst_dir:
        return False

    trade_upper = trade_dir.upper()
    bullish_trade = trade_upper in ("LONG", "BUY", "BULLISH")
    bearish_trade = trade_upper in ("SHORT", "SELL", "BEARISH")

    if bullish_trade and catalyst_dir == "BULLISH":
        return True
    if bearish_trade and catalyst_dir == "BEARISH":
        return True

    return False


def directions_conflict(trade_dir: str, catalyst_dir: Optional[str]) -> bool:
    """Check if trade direction and catalyst direction clearly conflict.

    Returns True only when there's a clear directional mismatch:
    - LONG/BUY trade + BEARISH catalyst
    - SHORT/SELL trade + BULLISH catalyst

    Returns False if either direction is None/empty (no conflict without evidence).
    """
    if not trade_dir or not catalyst_dir:
        return False

    trade_upper = trade_dir.upper()
    bullish_trade = trade_upper in ("LONG", "BUY", "BULLISH")
    bearish_trade = trade_upper in ("SHORT", "SELL", "BEARISH")

    if bullish_trade and catalyst_dir == "BEARISH":
        return True
    if bearish_trade and catalyst_dir == "BULLISH":
        return True

    return False


def is_breaking_level(indicators: dict) -> bool:
    """Check if price is breaking a key level based on available indicators.

    Looks for evidence that price is at or breaking through significant levels
    (day high, day low, prev close, or explicit key_levels).
    """
    if not indicators:
        return False

    current_price = indicators.get("current_price")
    if current_price is None:
        return False

    # Check if price is near/breaking day high
    day_high = indicators.get("day_high")
    if day_high and current_price >= day_high * 0.995:
        return True

    # Check if price is near/breaking day low (for shorts)
    day_low = indicators.get("day_low")
    if day_low and day_low > 0 and current_price <= day_low * 1.005:
        return True

    # Check explicit breaking_level flag
    if indicators.get("breaking_level"):
        return True

    return False


def is_strong_signal(signal: dict | None) -> bool:
    """Check if the analyst signal indicates strong conviction.

    Looks for strength/conviction fields in the signal dict.
    """
    if not signal:
        return False

    strength = signal.get("strength", "").lower() if signal.get("strength") else ""
    if strength in ("strong", "high", "very_strong"):
        return True

    conviction = signal.get("conviction", "").lower() if signal.get("conviction") else ""
    if conviction in ("high", "very_high"):
        return True

    return False


def is_overextended(indicators: dict) -> bool:
    """Check if the candidate is overextended without support.

    Looks for large intraday moves without retest or support levels.
    """
    if not indicators:
        return False

    change_pct = indicators.get("change_pct") or indicators.get("price_change_pct")
    if change_pct is not None:
        # Overextended if moved more than 5% intraday without support
        if abs(change_pct) > 5.0:
            # Check if there's support (retest or key level nearby)
            if not indicators.get("has_support") and not indicators.get("retested"):
                return True

    return False


def is_macro_catalyst(text: str) -> bool:
    """Detect broad macro terms in catalyst text.

    Returns True if the text contains macro-economic terms like:
    Fed, CPI, PCE, jobs report, FOMC, GDP, inflation, interest rate,
    Treasury, yields, etc.
    """
    if not text:
        return False

    text_lower = text.lower()
    return any(term in text_lower for term in _MACRO_TERMS)


# ---------------------------------------------------------------------------
# Macro instrument classification helpers
# ---------------------------------------------------------------------------

# Rate/commodity instruments whose primary driver IS macro
_RATE_COMMODITY_INSTRUMENTS: set[str] = {"TLT", "GLD"}

# Primary driver keywords for rate/commodity instruments
_PRIMARY_DRIVER_KEYWORDS: dict[str, list[str]] = {
    "TLT": ["fed", "yields", "cpi", "pce", "jobs report", "fomc", "interest rate",
             "treasury", "rate decision", "rate cut", "rate hike", "nonfarm",
             "unemployment", "federal reserve"],
    "GLD": ["gold", "bullion", "inflation", "precious metals", "safe haven",
             "inflation hedge"],
}

# Broad equity ETFs — get readthrough (3) for macro unless explicitly macro-focused
_BROAD_EQUITY_ETFS: set[str] = {"SPY", "QQQ", "IWM", "DIA"}

# Sector ETFs — get direct (4) for matching sector, sympathy (2) for different
_SECTOR_ETFS: set[str] = {"XLK", "XLF", "XLE"}

# Macro-focused setup types that allow broad equity ETFs to get full direct
_MACRO_SETUP_TYPES: set[str] = {
    "macro", "index", "macro_breakout", "index_breakout",
    "macro_catalyst", "fed_play", "rate_play",
}


def _is_macro_focused_setup(decision: dict, signal: dict | None) -> bool:
    """Check if the setup_type is explicitly macro/index-focused."""
    setup_type = (
        decision.get("setup_type")
        or (signal.get("setup_type") if signal else None)
        or ""
    ).lower()
    return any(ms in setup_type for ms in _MACRO_SETUP_TYPES)


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------


def compute_catalyst_score(
    symbol: str,
    decision: dict,
    signal: dict | None,
    aliases: dict[str, list[str]],
    relationships: dict[str, list[str]],
) -> tuple[int, str, list[str], list[str]]:
    """Compute catalyst specificity score for a trade candidate.

    Returns (score, reason_type, evidence_list, missing_list).

    Scoring components:
      - Symbol mention: +0 to +4
      - Freshness: -2 to +2
      - Confirmation: -1 to +2
      - Direction consistency: +1 / -3

    Final score is clamped to [0, 10].
    """
    score = 0
    evidence: list[str] = []
    missing: list[str] = []

    # Extract catalyst text from all available fields
    catalyst_text = extract_catalyst_text(decision, signal)

    if not catalyst_text.strip():
        return (0, "unknown", [], ["no catalyst evidence found"])

    # --- Symbol Mention Scoring (+0 to +4) ---
    symbol_upper = symbol.upper()
    symbol_names = aliases.get(symbol_upper, [symbol_upper])
    relationship_names = relationships.get(symbol_upper, [])

    # Macro instrument guardrails (Requirement 15)
    if symbol_upper in _RATE_COMMODITY_INSTRUMENTS:
        # TLT/GLD: full direct (4) for their primary driver
        primary_keywords = _PRIMARY_DRIVER_KEYWORDS.get(symbol_upper, [])
        if any_mention(catalyst_text, primary_keywords):
            score += 4
            evidence.append(f"catalyst directly relates to {symbol_upper} primary driver")
            mention_type = "direct"
        elif any_mention(catalyst_text, symbol_names):
            score += 4
            evidence.append(f"headline mentions {symbol_upper}")
            mention_type = "direct"
        elif any_mention(catalyst_text, relationship_names):
            score += 3
            evidence.append(f"headline mentions linked entity for {symbol_upper}")
            mention_type = "readthrough"
        elif contains_sector_terms(catalyst_text, symbol_upper):
            score += 2
            evidence.append("sector/theme terms present")
            mention_type = "sector"
        else:
            score += 0
            missing.append(f"no {symbol_upper}-specific headline")
            mention_type = "none"

    elif symbol_upper in _BROAD_EQUITY_ETFS:
        # SPY/QQQ/IWM/DIA: readthrough (3) for broad macro unless macro-focused setup
        if any_mention(catalyst_text, symbol_names):
            score += 4
            evidence.append(f"headline mentions {symbol_upper}")
            mention_type = "direct"
        elif is_macro_catalyst(catalyst_text):
            if _is_macro_focused_setup(decision, signal):
                # Explicitly macro-focused setup → full direct
                score += 4
                evidence.append(f"macro catalyst with macro-focused setup for {symbol_upper}")
                mention_type = "direct"
            else:
                # Broad macro → readthrough level only
                score += 3
                evidence.append(f"broad macro catalyst for {symbol_upper} (readthrough level)")
                mention_type = "readthrough"
        elif any_mention(catalyst_text, relationship_names):
            score += 3
            evidence.append(f"headline mentions linked entity for {symbol_upper}")
            mention_type = "readthrough"
        elif contains_sector_terms(catalyst_text, symbol_upper):
            score += 2
            evidence.append("sector/theme terms present")
            mention_type = "sector"
        else:
            score += 0
            missing.append(f"no {symbol_upper}-specific headline")
            mention_type = "none"

    elif symbol_upper in _SECTOR_ETFS:
        # Sector ETFs: direct (4) for matching sector, sympathy (2) for different
        if any_mention(catalyst_text, symbol_names):
            score += 4
            evidence.append(f"headline mentions {symbol_upper}")
            mention_type = "direct"
        elif contains_sector_terms(catalyst_text, symbol_upper):
            # Matching sector → direct
            score += 4
            evidence.append(f"catalyst matches {symbol_upper} sector (direct)")
            mention_type = "direct"
        elif any_mention(catalyst_text, relationship_names):
            score += 3
            evidence.append(f"headline mentions linked entity for {symbol_upper}")
            mention_type = "readthrough"
        else:
            # Check if it's a different sector's catalyst
            other_sector_match = False
            for other_etf in _SECTOR_ETFS:
                if other_etf != symbol_upper and contains_sector_terms(catalyst_text, other_etf):
                    other_sector_match = True
                    break
            if other_sector_match:
                score += 2
                evidence.append("sector sympathy (different sector catalyst)")
                mention_type = "sector"
            elif is_macro_catalyst(catalyst_text):
                score += 2
                evidence.append("broad macro catalyst for sector ETF")
                mention_type = "sector"
            else:
                score += 0
                missing.append(f"no {symbol_upper}-specific headline")
                mention_type = "none"

    else:
        # Standard symbol — normal scoring path
        if any_mention(catalyst_text, symbol_names):
            score += 4
            evidence.append(f"headline mentions {symbol_upper}")
            mention_type = "direct"
        elif any_mention(catalyst_text, relationship_names):
            score += 3
            evidence.append(f"headline mentions linked company for {symbol_upper}")
            mention_type = "readthrough"
        elif contains_sector_terms(catalyst_text, symbol_upper):
            score += 2
            evidence.append("sector/theme terms present")
            mention_type = "sector"
        else:
            score += 0
            missing.append(f"no {symbol_upper}-specific headline")
            mention_type = "none"

    # --- Freshness Scoring (-2 to +2) ---
    freshness = assess_freshness(decision, signal)
    if freshness == "intraday":
        score += 2
        evidence.append("catalyst is intraday/fresh")
    elif freshness == "same_day":
        score += 1
        evidence.append("catalyst is same-day")
    elif freshness == "stale":
        score -= 2
        missing.append("catalyst appears stale")

    # --- Confirmation Scoring (-1 to +2) ---
    indicators = extract_indicators(decision, signal)
    vol_ratio = indicators.get("volume_ratio") or indicators.get("relative_volume")

    if vol_ratio is not None and vol_ratio >= 1.5:
        score += 2
        evidence.append(f"relative volume {vol_ratio:.1f}x")
    elif is_breaking_level(indicators) or is_strong_signal(signal):
        score += 1
        evidence.append("breaking key level or strong signal")
    elif is_overextended(indicators):
        score -= 1
        missing.append("overextended without support")
    # else: no indicators → default 0 (no bonus, no penalty)

    # --- Direction Consistency (+1 / -3) ---
    trade_direction = (
        decision.get("bias")
        or decision.get("direction")
        or decision.get("action", "")
    ).upper()
    catalyst_direction = infer_catalyst_direction(catalyst_text)

    if catalyst_direction and trade_direction:
        if directions_match(trade_direction, catalyst_direction):
            score += 1
            evidence.append("catalyst direction matches trade")
        elif directions_conflict(trade_direction, catalyst_direction):
            score -= 3
            missing.append("catalyst direction conflicts with trade")

    # --- Clamp ---
    score = max(0, min(10, score))

    # --- Classify reason_type ---
    if directions_conflict(trade_direction, catalyst_direction):
        reason_type = "mismatch"
    elif mention_type == "direct":
        reason_type = "direct_symbol"
    elif mention_type == "readthrough":
        reason_type = "named_readthrough"
    elif mention_type == "sector":
        reason_type = "sector_sympathy"
    elif is_macro_catalyst(catalyst_text):
        reason_type = "macro_only"
    else:
        reason_type = "unknown"

    return (score, reason_type, evidence, missing)


# ---------------------------------------------------------------------------
# Gate Decision Engine
# ---------------------------------------------------------------------------


def apply_gate_decision(
    score: int,
    reason_type: str,
    profile: str,
    mode: str,
) -> tuple[str, float, str, float]:
    """Convert a catalyst specificity score into a gate decision.

    Uses profile-aware thresholds to determine allow/warn/reduce_size/block.
    In log_only mode, always returns allow with size_multiplier=1.0 but
    preserves the intended decision for logging.

    Args:
        score: Integer score in [0, 10].
        reason_type: Classification from compute_catalyst_score().
        profile: One of "conservative", "moderate", "aggressive".
        mode: Either "enforce" or "log_only".

    Returns:
        Tuple of (decision, size_multiplier, intended_decision, intended_size_multiplier).
        - decision: "allow" | "warn" | "reduce_size" | "block"
        - size_multiplier: 0.0–1.0 (always 1.0 in log_only mode)
        - intended_decision: what enforcement WOULD have done
        - intended_size_multiplier: what enforcement WOULD have applied
    """
    # Default to moderate for unknown profiles (Requirement 8.4)
    thresholds = CATALYST_SPECIFICITY_PROFILE_THRESHOLDS.get(
        profile, CATALYST_SPECIFICITY_PROFILE_THRESHOLDS["moderate"]
    )
    allow_threshold = thresholds["allow"]
    warn_threshold = thresholds["warn"]

    size_multiplier = 1.0

    if score >= allow_threshold:
        decision = "allow"
    elif score >= warn_threshold:
        if reason_type == "sector_sympathy":
            decision = "reduce_size"
            size_multiplier = CATALYST_SPECIFICITY_SECTOR_SYMPATHY_SIZE_MULTIPLIER.get(
                profile, CATALYST_SPECIFICITY_SECTOR_SYMPATHY_SIZE_MULTIPLIER["moderate"]
            )
        else:
            decision = "warn"
    else:
        # Below warn threshold — deterministic rules
        if reason_type == "sector_sympathy":
            # Sector sympathy below warn → always block
            decision = "block"
            size_multiplier = 0.0
        elif profile == "conservative":
            # Conservative blocks all non-sector below warn
            decision = "block"
            size_multiplier = 0.0
        else:
            # Moderate/aggressive reduce non-sector below warn
            decision = "reduce_size"
            size_multiplier = 0.5

    # Log-only mode override (Requirements 11.1, 11.2, 11.3)
    if mode == "log_only":
        intended_decision = decision
        intended_size_multiplier = size_multiplier
        decision = "allow"
        size_multiplier = 1.0
    else:
        intended_decision = decision
        intended_size_multiplier = size_multiplier

    return (decision, size_multiplier, intended_decision, intended_size_multiplier)


# ---------------------------------------------------------------------------
# Public API — Main Orchestrator
# ---------------------------------------------------------------------------


def evaluate_catalyst_specificity(
    decision: dict,
    signal: dict | None = None,
    profile: str = "moderate",
    db=None,
) -> dict:
    """Evaluate a trade candidate through the catalyst specificity gate.

    Orchestrates: check enabled flag → check applicability → compute score
    using merged context → apply decision → log events → return result.

    Args:
        decision: PM entry decision dict (must contain at least 'symbol').
        signal: Optional analyst signal dict with quote data, indicators,
                catalyst freshness, key_levels.
        profile: One of "conservative", "moderate", "aggressive".
        db: Optional SQLAlchemy session for trade event logging.

    Returns:
        Dict matching the Gate Result Schema with decision, score, evidence, etc.
    """
    # --- Read environment configuration ---
    gate_enabled = os.environ.get("CATALYST_SPECIFICITY_GATE_ENABLED", "true").lower()
    mode = os.environ.get("CATALYST_SPECIFICITY_GATE_MODE", "log_only").lower()
    if mode not in ("enforce", "log_only"):
        mode = "log_only"

    symbol = decision.get("symbol", "UNKNOWN")
    setup_type = (
        decision.get("setup_type")
        or (signal.get("setup_type") if signal else None)
        or ""
    )

    # --- Gate disabled: immediate allow ---
    if gate_enabled == "false":
        return {
            "gate": "catalyst_specificity_gate",
            "mode": mode,
            "decision": "allow",
            "intended_decision": "allow",
            "reason_type": "gate_disabled",
            "score": 10,
            "threshold": CATALYST_SPECIFICITY_PROFILE_THRESHOLDS.get(
                profile, CATALYST_SPECIFICITY_PROFILE_THRESHOLDS["moderate"]
            )["allow"],
            "symbol": symbol,
            "setup_type": setup_type,
            "evidence": [],
            "missing": [],
            "size_multiplier": 1.0,
            "intended_size_multiplier": 1.0,
            "reason": "Catalyst specificity gate is disabled",
            "quantity_before": None,
            "quantity_after": None,
        }

    # --- Check applicability ---
    if not should_apply_gate(decision, signal):
        return {
            "gate": "catalyst_specificity_gate",
            "mode": mode,
            "decision": "allow",
            "intended_decision": "allow",
            "reason_type": "not_applicable",
            "score": 10,
            "threshold": CATALYST_SPECIFICITY_PROFILE_THRESHOLDS.get(
                profile, CATALYST_SPECIFICITY_PROFILE_THRESHOLDS["moderate"]
            )["allow"],
            "symbol": symbol,
            "setup_type": setup_type,
            "evidence": [],
            "missing": [],
            "size_multiplier": 1.0,
            "intended_size_multiplier": 1.0,
            "reason": "Gate does not apply to this setup",
            "quantity_before": None,
            "quantity_after": None,
        }

    # --- Load config (cached) ---
    config = load_catalyst_config()
    aliases = config.get("aliases", {})
    relationships = config.get("relationships", {})

    # --- Compute score using merged context ---
    score, reason_type, evidence, missing = compute_catalyst_score(
        symbol=symbol,
        decision=decision,
        signal=signal,
        aliases=aliases,
        relationships=relationships,
    )

    # --- Apply gate decision ---
    gate_decision, size_multiplier, intended_decision, intended_size_multiplier = (
        apply_gate_decision(
            score=score,
            reason_type=reason_type,
            profile=profile,
            mode=mode,
        )
    )

    # --- Compute quantity_before / quantity_after ---
    quantity_before: int | None = None
    quantity_after: int | None = None

    if gate_decision == "reduce_size" and size_multiplier < 1.0:
        raw_qty = decision.get("quantity")
        if raw_qty is not None:
            quantity_before = int(raw_qty)
            quantity_after = max(1, int(raw_qty * size_multiplier))

    # --- Get threshold for result ---
    thresholds = CATALYST_SPECIFICITY_PROFILE_THRESHOLDS.get(
        profile, CATALYST_SPECIFICITY_PROFILE_THRESHOLDS["moderate"]
    )
    allow_threshold = thresholds["allow"]

    # --- Generate human-readable reason string ---
    reason = _build_reason_string(
        gate_decision=gate_decision,
        intended_decision=intended_decision,
        reason_type=reason_type,
        score=score,
        allow_threshold=allow_threshold,
        symbol=symbol,
        mode=mode,
        evidence=evidence,
        missing=missing,
    )

    # --- Build result dict ---
    result = {
        "gate": "catalyst_specificity_gate",
        "mode": mode,
        "decision": gate_decision,
        "intended_decision": intended_decision,
        "reason_type": reason_type,
        "score": score,
        "threshold": allow_threshold,
        "symbol": symbol,
        "setup_type": setup_type,
        "evidence": evidence,
        "missing": missing,
        "size_multiplier": size_multiplier,
        "intended_size_multiplier": intended_size_multiplier,
        "reason": reason,
        "quantity_before": quantity_before,
        "quantity_after": quantity_after,
    }

    # --- Log trade events (Requirements 10.1, 10.2, 10.3, 11.2, 11.3) ---
    if db is not None:
        # Build payload with all required fields
        event_payload = {
            "score": score,
            "threshold": allow_threshold,
            "decision": gate_decision,
            "intended_decision": intended_decision,
            "reason_type": reason_type,
            "size_multiplier": size_multiplier,
            "intended_size_multiplier": intended_size_multiplier,
            "mode": mode,
            "evidence": evidence,
            "missing": missing,
            "setup_type": setup_type,
        }

        # Include quantity_before/quantity_after when size is reduced
        if quantity_before is not None:
            event_payload["quantity_before"] = quantity_before
        if quantity_after is not None:
            event_payload["quantity_after"] = quantity_after

        # Always log "catalyst_specificity_gate_evaluated" for every evaluation
        log_trade_event(
            db,
            "catalyst_specificity_gate_evaluated",
            symbol=symbol,
            profile=profile,
            payload=event_payload,
        )

        # Log "gate_rejected" ONLY when decision=="block" AND mode=="enforce"
        # NEVER emit gate_rejected in log_only mode, even if intended_decision is "block"
        if gate_decision == "block" and mode == "enforce":
            log_trade_event(
                db,
                "gate_rejected",
                symbol=symbol,
                profile=profile,
                payload={
                    "gate_name": "catalyst_specificity_gate",
                    "score": score,
                    "threshold": allow_threshold,
                    "decision": gate_decision,
                    "intended_decision": intended_decision,
                    "reason_type": reason_type,
                    "size_multiplier": size_multiplier,
                    "intended_size_multiplier": intended_size_multiplier,
                    "mode": mode,
                    "evidence": evidence,
                    "missing": missing,
                },
            )

    return result


def _build_reason_string(
    *,
    gate_decision: str,
    intended_decision: str,
    reason_type: str,
    score: int,
    allow_threshold: int,
    symbol: str,
    mode: str,
    evidence: list[str],
    missing: list[str],
) -> str:
    """Build a human-readable reason string for the gate result."""
    parts: list[str] = []

    # Decision summary
    if mode == "log_only" and intended_decision != "allow":
        parts.append(
            f"[log_only] Would {intended_decision} {symbol} "
            f"(score {score}/{allow_threshold})"
        )
    else:
        parts.append(
            f"{gate_decision.replace('_', ' ').capitalize()} {symbol} "
            f"(score {score}/{allow_threshold})"
        )

    # Reason type context
    reason_labels = {
        "direct_symbol": "direct symbol catalyst",
        "named_readthrough": "named read-through catalyst",
        "sector_sympathy": "sector sympathy only",
        "macro_only": "broad macro catalyst",
        "mismatch": "catalyst direction conflicts with trade",
        "unknown": "no parseable catalyst evidence",
    }
    label = reason_labels.get(reason_type, reason_type)
    parts.append(f"— {label}")

    # Evidence summary
    if evidence:
        parts.append(f"[+{', '.join(evidence[:3])}]")
    if missing:
        parts.append(f"[-{', '.join(missing[:2])}]")

    return " ".join(parts)
