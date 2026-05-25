"""Chief Scout LLM curation module for the Sector Scout pipeline.

Receives deterministic finalists from sector screeners and uses a single
LLM call to curate 0–8 final picks for the Expanded Watchlist.

The Chief Scout SHALL NOT output PM decisions, entries, stops, targets,
quantities, or portfolio actions.

See: design.md §3 (Chief Scout LLM Curation), requirements.md §6
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from utils.case_library import get_selection_feedback, get_win_rate_by_setup
from utils.llm import call_llm, parse_json_response
from utils.scout_logging import emit_scout_event
from utils.sector_scout_models import (
    CandidateRow,
    REQUIRED_PICK_FIELDS,
    VALID_CONVICTION,
    VALID_DIRECTION_BIAS,
    ChiefScoutPick,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conviction ordering for truncation (highest first)
# ---------------------------------------------------------------------------

_CONVICTION_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}

# ---------------------------------------------------------------------------
# Chief Scout System Prompt
# ---------------------------------------------------------------------------

CHIEF_SCOUT_SYSTEM_PROMPT = """You are the Chief Scout for a day-trading system. Your role is to curate
the best trading candidates from a set of pre-screened, deterministically-scored finalists.

You receive:
- The Core Watchlist (already covered by other agents — do NOT duplicate these)
- Finalist candidates from multiple sector screeners, each with component scores
- Sector summaries showing overall sector activity
- Case library feedback showing what setups have worked historically

Your job:
1. Review the finalist candidates and their scores
2. Select 0–8 symbols that represent the best opportunities for today
3. For each pick, provide structured reasoning

CRITICAL RULES:
- You may ONLY select symbols from the finalist candidates provided
- Do NOT invent symbols or select symbols not in the finalist list
- Do NOT output trade entries, stops, targets, quantities, or portfolio actions
- Do NOT make PM decisions — you are a scout, not a portfolio manager
- Quality over quantity — return 0 picks if nothing stands out
- Each pick must include: symbol, sector, direction_bias, conviction, catalyst_summary, reason, risk, source_candidate_score

Respond in JSON format:
{
  "picks": [
    {
      "symbol": "AVGO",
      "sector": "ai_semi",
      "direction_bias": "bullish|bearish|neutral",
      "conviction": "low|medium|high",
      "catalyst_summary": "brief catalyst description",
      "reason": "why this is worth watching",
      "risk": "main risk to the thesis",
      "source_candidate_score": 72.5
    }
  ]
}

Valid direction_bias values: bullish, bearish, neutral
Valid conviction values: low, medium, high
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_chief_scout_pick(pick: dict, finalist_symbols: set[str]) -> bool:
    """Validate a single Chief Scout pick. Returns True if valid, False to discard.

    Checks:
    - All required fields are present
    - direction_bias is a valid enum value
    - conviction is a valid enum value
    - symbol is in the finalist set
    """
    if not isinstance(pick, dict):
        return False
    if not REQUIRED_PICK_FIELDS.issubset(pick.keys()):
        return False
    if pick.get("direction_bias") not in VALID_DIRECTION_BIAS:
        return False
    if pick.get("conviction") not in VALID_CONVICTION:
        return False
    if pick.get("symbol") not in finalist_symbols:
        return False
    return True


# ---------------------------------------------------------------------------
# Prompt Building
# ---------------------------------------------------------------------------


def _build_chief_scout_prompt(
    finalists_by_sector: dict,
    core_watchlist: list[str],
    engine,
) -> str:
    """Build the user prompt for the Chief Scout LLM call.

    Includes:
    - Core Watchlist for context
    - Finalist rows with component scores per sector
    - Sector summaries
    - Case library feedback when available
    """
    parts: list[str] = []

    # Header
    parts.append(f"Today: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    parts.append("")

    # Core Watchlist
    parts.append(f"CORE WATCHLIST (already covered, do NOT duplicate): {', '.join(core_watchlist)}")
    parts.append("")

    # Sector summaries and finalist rows
    parts.append("FINALIST CANDIDATES BY SECTOR:")
    parts.append("=" * 60)

    for sector_key, finalists in finalists_by_sector.items():
        if not finalists:
            continue

        # Sector summary
        sector_name = ""
        scores = []
        for f in finalists:
            if hasattr(f, "sector_name"):
                sector_name = f.sector_name
            if hasattr(f, "scout_score"):
                scores.append(f.scout_score)
            elif isinstance(f, dict):
                sector_name = f.get("sector_name", sector_key)
                scores.append(f.get("scout_score", 0))

        if not sector_name:
            sector_name = sector_key

        avg_score = sum(scores) / len(scores) if scores else 0
        parts.append(f"\n--- {sector_name} ({sector_key}) ---")
        parts.append(f"  Finalists: {len(finalists)} | Avg Score: {avg_score:.1f}")
        parts.append("")

        # Individual candidate rows
        for candidate in finalists:
            if hasattr(candidate, "symbol"):
                # CandidateRow dataclass
                parts.append(f"  {candidate.symbol}:")
                parts.append(f"    Scout Score: {candidate.scout_score:.1f}")
                parts.append(f"    Move: {candidate.move_pct:.2f}%" if candidate.move_pct is not None else "    Move: N/A")
                parts.append(f"    Relative Volume: {candidate.relative_volume:.2f}x" if candidate.relative_volume is not None else "    Relative Volume: N/A")
                parts.append(f"    Dollar Volume: ${candidate.dollar_volume:,.0f}" if candidate.dollar_volume is not None else "    Dollar Volume: N/A")
                parts.append(f"    News Freshness: {candidate.news_freshness_minutes:.0f} min" if candidate.news_freshness_minutes is not None else "    News Freshness: N/A")
                parts.append(f"    Sector Confirmed: {candidate.sector_confirmed}")
                parts.append(f"    Spread: {candidate.spread_pct:.2f}% ({candidate.spread_status})" if candidate.spread_pct is not None else f"    Spread: {candidate.spread_status}")

                # Component scores
                if candidate.component_scores:
                    comp_parts = [f"{k}={v:.1f}" for k, v in candidate.component_scores.items()]
                    parts.append(f"    Components: {', '.join(comp_parts)}")

                # Penalties
                if candidate.penalties_applied:
                    pen_parts = [f"{p['type']}(-{p['deduction']:.1f})" for p in candidate.penalties_applied]
                    parts.append(f"    Penalties: {', '.join(pen_parts)}")

                parts.append("")
            elif isinstance(candidate, dict):
                # Dict-based candidate
                sym = candidate.get("symbol", "?")
                parts.append(f"  {sym}:")
                parts.append(f"    Scout Score: {candidate.get('scout_score', 0):.1f}")
                move = candidate.get("move_pct")
                parts.append(f"    Move: {move:.2f}%" if move is not None else "    Move: N/A")
                rvol = candidate.get("relative_volume")
                parts.append(f"    Relative Volume: {rvol:.2f}x" if rvol is not None else "    Relative Volume: N/A")
                dvol = candidate.get("dollar_volume")
                parts.append(f"    Dollar Volume: ${dvol:,.0f}" if dvol is not None else "    Dollar Volume: N/A")
                news = candidate.get("news_freshness_minutes")
                parts.append(f"    News Freshness: {news:.0f} min" if news is not None else "    News Freshness: N/A")
                parts.append(f"    Sector Confirmed: {candidate.get('sector_confirmed')}")
                spread = candidate.get("spread_pct")
                spread_status = candidate.get("spread_status", "unknown")
                parts.append(f"    Spread: {spread:.2f}% ({spread_status})" if spread is not None else f"    Spread: {spread_status}")

                comp_scores = candidate.get("component_scores", {})
                if comp_scores:
                    comp_parts = [f"{k}={v:.1f}" for k, v in comp_scores.items()]
                    parts.append(f"    Components: {', '.join(comp_parts)}")

                penalties = candidate.get("penalties_applied", [])
                if penalties:
                    pen_parts = [f"{p['type']}(-{p['deduction']:.1f})" for p in penalties]
                    parts.append(f"    Penalties: {', '.join(pen_parts)}")

                parts.append("")

    # Case library feedback (when available)
    parts.append("")
    parts.append("CASE LIBRARY FEEDBACK:")
    try:
        selection_fb = get_selection_feedback(engine, limit=10)
        parts.append(selection_fb if selection_fb else "No selection feedback available.")
    except Exception as exc:
        logger.debug("Could not load selection feedback: %s", exc)
        parts.append("No selection feedback available.")

    parts.append("")
    parts.append("SETUP WIN RATES:")
    try:
        win_rates = get_win_rate_by_setup(engine)
        if win_rates:
            for r in win_rates:
                parts.append(
                    f"  {r['setup_type']}: {r['win_rate']}% win rate over "
                    f"{r['total']} cases (avg pnl: {r['avg_pnl_pct']}%)"
                )
        else:
            parts.append("  No case history yet.")
    except Exception as exc:
        logger.debug("Could not load win rates: %s", exc)
        parts.append("  No case history yet.")

    parts.append("")
    parts.append("Select 0–8 of the best candidates from the finalists above.")
    parts.append("Return your picks as JSON with the required fields.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def run_chief_scout(
    finalists_by_sector: dict,
    core_watchlist: list[str],
    config: dict,
    engine,
) -> dict:
    """Call Chief Scout LLM to curate final picks from deterministic finalists.

    Args:
        finalists_by_sector: Top N candidates per sector from screening.
        core_watchlist: For context in the prompt.
        config: Chief scout configuration.
        engine: DB engine for case library / feedback context.

    Returns:
        {
            "picks": [ChiefScoutPick, ...],  # 0-8 items
            "fallback_used": bool,
            "llm_error": str | None,
        }
    """
    chief_cfg = config.get("chief_scout", {})
    max_picks = chief_cfg.get("max_picks", 8)

    # Collect all finalist symbols for validation
    finalist_symbols: set[str] = set()
    for sector_key, finalists in finalists_by_sector.items():
        for candidate in finalists:
            if hasattr(candidate, "symbol"):
                finalist_symbols.add(candidate.symbol)
            elif isinstance(candidate, dict):
                finalist_symbols.add(candidate.get("symbol", ""))

    # If no finalists at all, return empty (no LLM call needed)
    if not finalist_symbols:
        logger.info("Chief Scout: no finalists to curate, returning empty picks")
        return {
            "picks": [],
            "fallback_used": False,
            "llm_error": None,
        }

    # Build prompt
    user_prompt = _build_chief_scout_prompt(
        finalists_by_sector, core_watchlist, engine
    )

    # Call LLM
    try:
        raw_response = call_llm(
            CHIEF_SCOUT_SYSTEM_PROMPT,
            user_prompt,
            json_mode=True,
            tier="low",
            purpose="chief_scout",
        )
    except Exception as exc:
        error_msg = f"Chief Scout LLM call failed: {type(exc).__name__}: {exc}"
        logger.error(error_msg)
        return {
            "picks": [],
            "fallback_used": False,
            "llm_error": error_msg,
        }

    # Parse JSON response
    try:
        parsed = parse_json_response(raw_response)
    except (ValueError, json.JSONDecodeError) as exc:
        error_msg = f"Chief Scout LLM returned malformed JSON: {exc}"
        logger.error(error_msg)
        return {
            "picks": [],
            "fallback_used": False,
            "llm_error": error_msg,
        }

    # Extract picks list
    raw_picks = parsed.get("picks", [])
    if not isinstance(raw_picks, list):
        error_msg = "Chief Scout LLM response 'picks' field is not a list"
        logger.error(error_msg)
        return {
            "picks": [],
            "fallback_used": False,
            "llm_error": error_msg,
        }

    # Validate each pick — discard invalid, don't repair
    valid_picks: list[dict] = []
    for i, pick in enumerate(raw_picks):
        if validate_chief_scout_pick(pick, finalist_symbols):
            valid_picks.append(pick)
        else:
            logger.warning(
                "Chief Scout pick %d discarded (validation failed): %s",
                i,
                pick.get("symbol", "?") if isinstance(pick, dict) else "?",
            )

    # If ALL picks failed validation → treat as LLM failure
    if raw_picks and not valid_picks:
        error_msg = (
            f"Chief Scout: all {len(raw_picks)} picks failed validation, "
            "treating as LLM failure"
        )
        logger.error(error_msg)
        return {
            "picks": [],
            "fallback_used": False,
            "llm_error": error_msg,
        }

    # Truncate to max_picks, keeping highest conviction first
    if len(valid_picks) > max_picks:
        valid_picks.sort(
            key=lambda p: _CONVICTION_ORDER.get(p.get("conviction", "low"), 2)
        )
        valid_picks = valid_picks[:max_picks]

    # Cast to ChiefScoutPick typed dicts
    typed_picks: list[ChiefScoutPick] = []
    for p in valid_picks:
        typed_picks.append(
            ChiefScoutPick(
                symbol=p["symbol"],
                sector=p["sector"],
                direction_bias=p["direction_bias"],
                conviction=p["conviction"],
                catalyst_summary=p["catalyst_summary"],
                reason=p["reason"],
                risk=p["risk"],
                source_candidate_score=float(p["source_candidate_score"]),
            )
        )

    logger.info(
        "Chief Scout: %d valid picks from %d LLM responses (max_picks=%d)",
        len(typed_picks),
        len(raw_picks),
        max_picks,
    )

    # Emit structured CHIEF_SCOUT event
    emit_scout_event("CHIEF_SCOUT", {
        "picks_count": len(typed_picks),
        "fallback_used": False,
        "symbols_selected": [p["symbol"] for p in typed_picks],
    })

    return {
        "picks": typed_picks,
        "fallback_used": False,
        "llm_error": None,
    }


# ---------------------------------------------------------------------------
# Deterministic Fallback
# ---------------------------------------------------------------------------


def _fallback_sort_key(row) -> tuple:
    """Return a composite sort key for stable, deterministic ordering.

    Uses the same tie-breaking logic as rank_candidates in sector_scout.py:
      1. scout_score descending
      2. relative_volume descending (None treated as 0)
      3. dollar_volume descending (None treated as 0)
      4. symbol ascending (lexicographic)
    """
    if hasattr(row, "scout_score"):
        score = row.scout_score
        rvol = row.relative_volume if row.relative_volume is not None else 0.0
        dvol = row.dollar_volume if row.dollar_volume is not None else 0.0
        symbol = row.symbol
    elif isinstance(row, dict):
        score = row.get("scout_score", 0.0)
        rvol = row.get("relative_volume") or 0.0
        dvol = row.get("dollar_volume") or 0.0
        symbol = row.get("symbol", "")
    else:
        return (0.0, 0.0, 0.0, "")

    return (-score, -rvol, -dvol, symbol)


def chief_scout_fallback(
    finalists_by_sector: dict,
    config: dict,
    current_watchlist_size: int = 0,
) -> dict:
    """Deterministic fallback when Chief Scout LLM fails.

    Returns top N candidates by scout_score from the finalist pool.
    Respects max_expanded_watchlist ceiling.

    Args:
        finalists_by_sector: Top N candidates per sector from screening.
        config: Parsed config dict.
        current_watchlist_size: Current number of symbols in today's expanded watchlist.

    Returns:
        {
            "picks": [ChiefScoutPick, ...],
            "fallback_used": True,
            "llm_error": str | None,
        }
    """
    chief_cfg = config.get("chief_scout", {})
    fallback_limit: int = int(chief_cfg.get("fallback_limit", 3))

    budget_ceilings = config.get("budget_ceilings", {})
    max_expanded_watchlist: int = int(budget_ceilings.get("max_expanded_watchlist", 12))

    # 1. Collect all finalists from all sectors into a single list
    all_finalists: list = []
    for sector_key, finalists in finalists_by_sector.items():
        if finalists:
            all_finalists.extend(finalists)

    # 2. Sort by scout_score descending with stable tie-breaking
    all_finalists.sort(key=_fallback_sort_key)

    # 3. Determine how many to return: min(fallback_limit, max_expanded_watchlist - current_watchlist_size)
    available_slots = max_expanded_watchlist - current_watchlist_size
    n_to_return = min(fallback_limit, available_slots)

    # 4. If that number is 0 or negative (watchlist already full), return empty picks
    if n_to_return <= 0 or not all_finalists:
        if not all_finalists:
            logger.info(
                "Scout: no expanded candidates — "
                "both screening and Chief Scout produced nothing"
            )
        else:
            logger.info(
                "Scout: no expanded candidates — "
                "watchlist at capacity (%d/%d), fallback cannot add symbols",
                current_watchlist_size,
                max_expanded_watchlist,
            )
        return {
            "picks": [],
            "fallback_used": True,
            "llm_error": None,
        }

    # 5. Take the top N candidates
    top_candidates = all_finalists[:n_to_return]

    # 6. Convert to ChiefScoutPick format
    picks: list[ChiefScoutPick] = []
    for candidate in top_candidates:
        if hasattr(candidate, "symbol"):
            # CandidateRow dataclass
            picks.append(
                ChiefScoutPick(
                    symbol=candidate.symbol,
                    sector=candidate.sector,
                    direction_bias="neutral",
                    conviction="low",
                    catalyst_summary="Deterministic fallback - top by scout score",
                    reason="Selected by deterministic fallback (LLM unavailable)",
                    risk="No LLM risk assessment available",
                    source_candidate_score=float(candidate.scout_score),
                )
            )
        elif isinstance(candidate, dict):
            # Dict-based candidate
            picks.append(
                ChiefScoutPick(
                    symbol=candidate.get("symbol", ""),
                    sector=candidate.get("sector", ""),
                    direction_bias="neutral",
                    conviction="low",
                    catalyst_summary="Deterministic fallback - top by scout score",
                    reason="Selected by deterministic fallback (LLM unavailable)",
                    risk="No LLM risk assessment available",
                    source_candidate_score=float(candidate.get("scout_score", 0.0)),
                )
            )

    logger.info(
        "Chief Scout fallback: returning top %d candidates by scout_score "
        "(fallback_limit=%d, available_slots=%d)",
        len(picks),
        fallback_limit,
        available_slots,
    )

    # Emit structured CHIEF_SCOUT event for fallback
    emit_scout_event("CHIEF_SCOUT", {
        "picks_count": len(picks),
        "fallback_used": True,
        "symbols_selected": [p["symbol"] for p in picks],
    })

    # 7. Return with fallback_used = True
    return {
        "picks": picks,
        "fallback_used": True,
        "llm_error": None,
    }
