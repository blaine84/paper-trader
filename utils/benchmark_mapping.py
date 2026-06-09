"""Benchmark Mapping — trusted configuration for sector/industry benchmark instruments.

Provides the canonical mapping from symbol sector/industry classifications to
specific benchmark instruments. NOT a universal ETF — each sector has its own
appropriate benchmark.

Requirements: 14.2, 14.3, 14.4, 16.1-16.10
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Mapping version — increment when mappings change
MAPPING_VERSION = "1.0.0"
MAPPING_SOURCE = "paper_trader_config"

# Broad market benchmark (default)
BROAD_MARKET_BENCHMARK = "SPY"

# Sector benchmark mappings (sector -> benchmark ETF)
SECTOR_BENCHMARKS: dict[str, str] = {
    "technology": "XLK",
    "healthcare": "XLV",
    "financials": "XLF",
    "energy": "XLE",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "industrials": "XLI",
    "materials": "XLB",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "communication_services": "XLC",
}

# Industry benchmark mappings (industry -> specific benchmark ETF)
# Not universal ETF — each industry gets its own appropriate benchmark (Req 14.3)
INDUSTRY_BENCHMARKS: dict[str, str] = {
    "semiconductors": "SMH",
    "software": "IGV",
    "biotech": "XBI",
    "internet": "FDN",
    "cybersecurity": "HACK",
    "cloud_computing": "SKYY",
    "banking": "KBE",
    "regional_banks": "KRE",
    "oil_exploration": "XOP",
    "homebuilders": "XHB",
    "retail": "XRT",
    "aerospace_defense": "ITA",
    "gold_miners": "GDX",
}

# Symbol to sector/industry classification
# This is the trusted config — not from PM or Analyst prose (Req 14.2)
SYMBOL_CLASSIFICATION: dict[str, dict[str, str | None]] = {
    # Semiconductors
    "NVDA": {"sector": "technology", "industry": "semiconductors"},
    "AMD": {"sector": "technology", "industry": "semiconductors"},
    "AVGO": {"sector": "technology", "industry": "semiconductors"},
    "SMCI": {"sector": "technology", "industry": "semiconductors"},
    "ARM": {"sector": "technology", "industry": "semiconductors"},
    "MU": {"sector": "technology", "industry": "semiconductors"},
    "INTC": {"sector": "technology", "industry": "semiconductors"},
    # Software/Cloud
    "MSFT": {"sector": "technology", "industry": "software"},
    "CRM": {"sector": "technology", "industry": "software"},
    "NOW": {"sector": "technology", "industry": "software"},
    "PLTR": {"sector": "technology", "industry": "software"},
    # Internet
    "GOOGL": {"sector": "communication_services", "industry": "internet"},
    "META": {"sector": "communication_services", "industry": "internet"},
    "AMZN": {"sector": "consumer_discretionary", "industry": "internet"},
    "NFLX": {"sector": "communication_services", "industry": "internet"},
    # General tech (no specific industry)
    "AAPL": {"sector": "technology", "industry": None},
    "TSLA": {"sector": "consumer_discretionary", "industry": None},
    # Financials
    "JPM": {"sector": "financials", "industry": "banking"},
    "BAC": {"sector": "financials", "industry": "banking"},
    "GS": {"sector": "financials", "industry": "banking"},
    # Energy
    "XOM": {"sector": "energy", "industry": "oil_exploration"},
    "CVX": {"sector": "energy", "industry": "oil_exploration"},
    # Healthcare
    "JNJ": {"sector": "healthcare", "industry": None},
    "UNH": {"sector": "healthcare", "industry": None},
    "LLY": {"sector": "healthcare", "industry": "biotech"},
    "MRNA": {"sector": "healthcare", "industry": "biotech"},
    # Crypto proxy
    "COIN": {"sector": "financials", "industry": None},
    "MSTR": {"sector": "technology", "industry": None},
}

# Supported symbols for market data (benchmark must be in this set) — Requirement 16.1
SUPPORTED_SYMBOLS: set[str] = (
    set(SECTOR_BENCHMARKS.values())
    | set(INDUSTRY_BENCHMARKS.values())
    | {BROAD_MARKET_BENCHMARK}
    | set(SYMBOL_CLASSIFICATION.keys())
)

# Freshness configuration (Requirement 16.10)
DEFAULT_FRESHNESS_CONFIG: dict = {
    "max_age_minutes": 15,
    "required_fields": ["symbol_price"],
    "version": "1.0.0",
}


def get_benchmark_mapping(symbol: str) -> dict | None:
    """Get the benchmark mapping for a symbol.

    Returns a dict suitable for build_context_snapshot(), or None if the
    symbol has no trusted classification.

    The mapping comes from trusted configuration only — never from PM
    or Analyst prose (Requirement 14.2).
    """
    classification = SYMBOL_CLASSIFICATION.get(symbol)
    if classification is None:
        # Unknown symbol — no trusted classification available
        return None

    sector = classification["sector"]
    industry = classification.get("industry")

    sector_benchmark = SECTOR_BENCHMARKS.get(sector)
    if sector_benchmark is None:
        logger.warning("No sector benchmark for sector=%s (symbol=%s)", sector, symbol)
        return None

    industry_benchmark = INDUSTRY_BENCHMARKS.get(industry) if industry else None

    return {
        "broad_market_benchmark": BROAD_MARKET_BENCHMARK,
        "sector": sector,
        "sector_benchmark": sector_benchmark,
        "industry": industry,
        "industry_benchmark": industry_benchmark,
        "mapping_source": MAPPING_SOURCE,
        "mapping_version": MAPPING_VERSION,
    }


def validate_benchmark_mapping(mapping: dict) -> tuple[bool, str | None]:
    """Validate that a benchmark mapping is internally consistent.

    Checks (Requirements 16.1-16.5):
    1. Benchmark IDs are supported symbols
    2. Mappings are compatible with classification
    3. Required fields present

    Returns (valid, reason).
    """
    # Check required fields
    required = ["broad_market_benchmark", "sector", "sector_benchmark", "mapping_source", "mapping_version"]
    for field in required:
        if not mapping.get(field):
            return (False, f"missing required field: {field}")

    # Check benchmark IDs are supported (Req 16.1)
    for bm_field in ["broad_market_benchmark", "sector_benchmark", "industry_benchmark"]:
        bm_symbol = mapping.get(bm_field)
        if bm_symbol and bm_symbol not in SUPPORTED_SYMBOLS:
            return (False, f"unsupported benchmark symbol: {bm_symbol} in {bm_field}")

    # Check sector mapping exists (Req 16.2)
    sector = mapping.get("sector")
    if sector and sector not in SECTOR_BENCHMARKS:
        return (False, f"unknown sector: {sector}")

    # Check industry mapping if provided (Req 16.2)
    industry = mapping.get("industry")
    if industry and industry not in INDUSTRY_BENCHMARKS:
        # Industry without a benchmark is OK (optional) — just no industry_benchmark
        if mapping.get("industry_benchmark"):
            return (False, f"industry benchmark provided but industry unknown: {industry}")

    return (True, None)


def validate_context_for_prompt(
    context_json: str | None,
    freshness_config: dict | None = None,
) -> tuple[bool, str, str | None]:
    """Validate context snapshot before including in PM prompt.

    Requirements 16.3-16.7:
    - Numeric fields must be finite and correctly typed
    - Timestamps meet configured freshness limits
    - If required context missing → exclude candidate
    - If only optional context missing → include with degraded marker

    Args:
        context_json: Serialized ContextSnapshot JSON, or None.
        freshness_config: Freshness configuration (or use defaults).

    Returns:
        (include, state, reason)
        - include: True if candidate should be included in prompt
        - state: "complete" | "degraded" | "excluded"
        - reason: Reason for exclusion/degradation (None if complete)
    """
    if context_json is None:
        return (False, "excluded", "no context snapshot")

    config = freshness_config or DEFAULT_FRESHNESS_CONFIG

    try:
        import json
        data = json.loads(context_json)
    except (json.JSONDecodeError, TypeError):
        return (False, "excluded", "invalid context JSON")

    # Check numeric fields are finite (Req 16.3)
    numeric_fields = ["symbol_price", "symbol_session_change_pct"]
    for field in numeric_fields:
        value = data.get(field)
        if value is not None:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return (False, "excluded", f"non-finite numeric field: {field}")

    # Check required fields present (Req 16.6)
    required_fields = config.get("required_fields", ["symbol_price"])
    missing_required = [f for f in required_fields if data.get(f) is None]
    if missing_required:
        return (False, "excluded", f"missing required fields: {missing_required}")

    # Check timestamp freshness (Req 16.4)
    snapshot_ts = data.get("snapshot_timestamp")
    if snapshot_ts:
        try:
            ts = datetime.fromisoformat(snapshot_ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_minutes = (datetime.now(timezone.utc) - ts).total_seconds() / 60
            max_age = config.get("max_age_minutes", 15)
            if age_minutes > max_age:
                return (False, "excluded", f"context stale ({age_minutes:.0f}min > {max_age}min)")
        except (ValueError, TypeError):
            return (False, "excluded", "invalid snapshot_timestamp")

    # Check state (Req 16.6, 16.7)
    context_state = data.get("context_state", "complete")
    if context_state == "excluded":
        return (False, "excluded", "context_state is excluded")
    if context_state == "degraded":
        return (True, "degraded", f"degraded fields: {data.get('degraded_fields', [])}")

    return (True, "complete", None)
