"""Data models and validation constants for the Sector Scout pipeline.

Defines the CandidateRow dataclass (full schema for screened symbols),
RunSummary TypedDict (per-run audit record), ChiefScoutPick TypedDict
(LLM curation output), and CooldownState TypedDict (reanalysis gating).

See: design.md §Data Models, requirements.md §1.5, §2.9
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


# ---------------------------------------------------------------------------
# Validation Constants
# ---------------------------------------------------------------------------

VALID_DIRECTION_BIAS: set[str] = {"bullish", "bearish", "neutral"}
VALID_CONVICTION: set[str] = {"low", "medium", "high"}
REQUIRED_PICK_FIELDS: set[str] = {
    "symbol",
    "sector",
    "direction_bias",
    "conviction",
    "catalyst_summary",
    "reason",
    "risk",
    "source_candidate_score",
}


# ---------------------------------------------------------------------------
# CandidateRow — Full schema for a screened symbol
# ---------------------------------------------------------------------------

@dataclass
class CandidateRow:
    """Normalized row representing one screened candidate symbol.

    All numeric market-data fields are Optional (None when unavailable).
    The pipeline never fabricates values — missing data is flagged explicitly.
    """

    # Identity
    symbol: str
    sector: str
    sector_name: str

    # Market data
    current_price: float | None = None
    prev_close: float | None = None
    move_pct: float | None = None
    current_volume: float | None = None
    average_volume: float | None = None
    relative_volume: float | None = None
    dollar_volume: float | None = None

    # News
    news_headlines: list[dict] | None = None
    news_freshness_minutes: float | None = None

    # Sector context
    sector_etf: str | None = None
    sector_etf_move_pct: float | None = None
    sector_confirmed: bool | None = None

    # Spread
    bid: float | None = None
    ask: float | None = None
    spread_pct: float | None = None
    spread_status: str = "unknown"  # "known" | "unknown"

    # Market cap
    market_cap: float | None = None
    market_cap_source: str | None = None  # "api" | "proxy" | None
    microcap_proxy_used: bool = False

    # Quality flags
    missing_data_flags: list[str] = field(default_factory=list)
    hard_gate_passed: bool = False
    reason_codes: list[str] = field(default_factory=list)

    # Scoring
    scout_score: float = 0.0
    component_scores: dict[str, float] = field(default_factory=dict)
    penalties_applied: list[dict] = field(default_factory=list)

    # Metadata
    collected_at: str = ""  # ISO timestamp
    run_type: str = ""  # "premarket" | "confirmation" | "midday"


# ---------------------------------------------------------------------------
# RunSummary — Per-run audit record persisted to AgentMemory
# ---------------------------------------------------------------------------

class RunSummary(TypedDict):
    """Structured summary of a single sector scout pipeline execution."""

    run_type: str  # "premarket" | "confirmation" | "midday"
    timestamp: str  # ISO
    sectors_scanned: int
    total_candidates_evaluated: int
    hard_gate_rejections: int
    finalists_count: int
    chief_scout_picks: list[dict]
    fallback_used: bool
    expanded_watchlist_symbols: list[str]
    expanded_watchlist_size: int
    reason_counts: dict[str, int]
    budget_hits: list[str]
    duration_seconds: float


# ---------------------------------------------------------------------------
# ChiefScoutPick — Structured output from Chief Scout LLM curation
# ---------------------------------------------------------------------------

class ChiefScoutPick(TypedDict):
    """A single pick returned by the Chief Scout LLM."""

    symbol: str
    sector: str
    direction_bias: str  # "bullish" | "bearish" | "neutral"
    conviction: str  # "low" | "medium" | "high"
    catalyst_summary: str
    reason: str
    risk: str
    source_candidate_score: float


# ---------------------------------------------------------------------------
# CooldownState — Reanalysis gating state per symbol
# ---------------------------------------------------------------------------

class CooldownState(TypedDict):
    """Tracks prior analysis state for reanalysis cooldown decisions."""

    symbol: str
    last_scout_score: float
    last_move_pct: float
    last_news_headline: str | None
    last_analyzed_at: str  # ISO timestamp
