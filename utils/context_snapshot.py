"""Context Snapshot Builder — frozen point-in-time market context for candidates.

Builds a ContextSnapshot containing benchmark-aware sector/industry data
for each candidate. The snapshot is frozen at registration time and stored
as canonical JSON in pm_candidates.context_snapshot_json.

Requirements: 14.1, 14.2, 14.3, 14.4, 15.1, 15.2, 16.8
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextSnapshot:
    """Frozen point-in-time market context for a candidate."""
    symbol: str
    broad_market_benchmark: str  # e.g., "SPY"
    sector: str
    sector_benchmark: str  # e.g., "XLK", "XLF"
    industry: str | None
    industry_benchmark: str | None  # e.g., "SMH", "SOXX"
    mapping_source: str
    mapping_version: str
    snapshot_timestamp: datetime
    # Symbol measurements
    symbol_price: float | None
    symbol_session_change_pct: float | None
    symbol_vs_vwap: float | None
    symbol_momentum: str | None  # "bullish" | "bearish" | "neutral" | None
    # Benchmark measurements (keyed by benchmark ticker)
    benchmark_measurements: dict[str, dict] = field(default_factory=dict)
    # Relative strength
    relative_strength: dict[str, float | None] = field(default_factory=dict)
    # Data quality
    freshness_ok: bool = True
    degraded_fields: list[str] = field(default_factory=list)
    context_state: str = "complete"  # "complete" | "degraded" | "excluded"

    def to_json(self) -> str:
        """Serialize to canonical JSON for storage."""
        data = {
            "symbol": self.symbol,
            "broad_market_benchmark": self.broad_market_benchmark,
            "sector": self.sector,
            "sector_benchmark": self.sector_benchmark,
            "industry": self.industry,
            "industry_benchmark": self.industry_benchmark,
            "mapping_source": self.mapping_source,
            "mapping_version": self.mapping_version,
            "snapshot_timestamp": self.snapshot_timestamp.isoformat(),
            "symbol_price": self.symbol_price,
            "symbol_session_change_pct": self.symbol_session_change_pct,
            "symbol_vs_vwap": self.symbol_vs_vwap,
            "symbol_momentum": self.symbol_momentum,
            "benchmark_measurements": self.benchmark_measurements,
            "relative_strength": self.relative_strength,
            "freshness_ok": self.freshness_ok,
            "degraded_fields": self.degraded_fields,
            "context_state": self.context_state,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def build_context_snapshot(
    symbol: str,
    benchmark_mapping: dict,
    market_data_provider: Any,
    freshness_config: dict,
) -> ContextSnapshot | None:
    """Build frozen context snapshot for a candidate.

    Returns None if required fields are unavailable (candidate excluded).
    Returns ContextSnapshot with context_state="degraded" if only optional fields missing.
    Result is immediately serialized to canonical JSON for storage.

    Never replaces missing context with model-generated values (Requirement 16.8).

    Args:
        symbol: The candidate's trading symbol.
        benchmark_mapping: Dict with keys: broad_market_benchmark, sector,
            sector_benchmark, industry, industry_benchmark, mapping_source,
            mapping_version.
        market_data_provider: Object with get_quote(symbol) method returning
            dict with price, change_pct, etc.
        freshness_config: Dict with keys: max_age_minutes (int), required_fields (list).
    """
    # Validate required mapping fields
    required_mapping_fields = [
        "broad_market_benchmark",
        "sector",
        "sector_benchmark",
        "mapping_source",
        "mapping_version",
    ]
    for field_name in required_mapping_fields:
        if not benchmark_mapping.get(field_name):
            logger.warning(
                "Context excluded for %s: missing %s in benchmark_mapping",
                symbol,
                field_name,
            )
            return None

    now = datetime.now(timezone.utc)
    degraded_fields: list[str] = []

    # Get symbol measurements
    symbol_price = None
    symbol_session_change_pct = None
    symbol_vs_vwap = None
    symbol_momentum = None

    try:
        quote = market_data_provider.get_quote(symbol)
        symbol_price = quote.get("price") if quote else None
        symbol_session_change_pct = quote.get("change_pct") if quote else None
        symbol_vs_vwap = quote.get("vs_vwap") if quote else None
        symbol_momentum = quote.get("momentum") if quote else None
    except Exception as exc:
        logger.warning("Failed to get quote for %s: %s", symbol, exc)

    # Check if required symbol measurements are present
    required_fields = freshness_config.get("required_fields", ["symbol_price"])
    if "symbol_price" in required_fields and (symbol_price is None or symbol_price <= 0):
        logger.warning("Context excluded for %s: symbol_price unavailable", symbol)
        return None

    # Optional fields that are degraded
    if symbol_session_change_pct is None:
        degraded_fields.append("symbol_session_change_pct")
    if symbol_vs_vwap is None:
        degraded_fields.append("symbol_vs_vwap")
    if symbol_momentum is None:
        degraded_fields.append("symbol_momentum")

    # Get benchmark measurements
    benchmarks_to_check = [
        benchmark_mapping["broad_market_benchmark"],
        benchmark_mapping["sector_benchmark"],
    ]
    if benchmark_mapping.get("industry_benchmark"):
        benchmarks_to_check.append(benchmark_mapping["industry_benchmark"])

    benchmark_measurements: dict[str, dict] = {}
    relative_strength: dict[str, float | None] = {}

    for bm_symbol in benchmarks_to_check:
        try:
            bm_quote = market_data_provider.get_quote(bm_symbol)
            if bm_quote:
                benchmark_measurements[bm_symbol] = {
                    "price": bm_quote.get("price"),
                    "session_change_pct": bm_quote.get("change_pct"),
                    "vs_vwap": bm_quote.get("vs_vwap"),
                    "momentum": bm_quote.get("momentum"),
                }
                # Compute relative strength
                if (
                    symbol_session_change_pct is not None
                    and bm_quote.get("change_pct") is not None
                ):
                    relative_strength[bm_symbol] = (
                        symbol_session_change_pct - bm_quote["change_pct"]
                    )
                else:
                    relative_strength[bm_symbol] = None
            else:
                benchmark_measurements[bm_symbol] = {}
                relative_strength[bm_symbol] = None
                degraded_fields.append(f"benchmark_{bm_symbol}")
        except Exception as exc:
            logger.warning("Failed to get benchmark quote for %s: %s", bm_symbol, exc)
            benchmark_measurements[bm_symbol] = {}
            relative_strength[bm_symbol] = None
            degraded_fields.append(f"benchmark_{bm_symbol}")

    # Determine context state
    context_state = "degraded" if degraded_fields else "complete"

    return ContextSnapshot(
        symbol=symbol,
        broad_market_benchmark=benchmark_mapping["broad_market_benchmark"],
        sector=benchmark_mapping["sector"],
        sector_benchmark=benchmark_mapping["sector_benchmark"],
        industry=benchmark_mapping.get("industry"),
        industry_benchmark=benchmark_mapping.get("industry_benchmark"),
        mapping_source=benchmark_mapping["mapping_source"],
        mapping_version=benchmark_mapping["mapping_version"],
        snapshot_timestamp=now,
        symbol_price=symbol_price,
        symbol_session_change_pct=symbol_session_change_pct,
        symbol_vs_vwap=symbol_vs_vwap,
        symbol_momentum=symbol_momentum,
        benchmark_measurements=benchmark_measurements,
        relative_strength=relative_strength,
        freshness_ok=True,
        degraded_fields=degraded_fields,
        context_state=context_state,
    )
