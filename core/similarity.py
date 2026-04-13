"""
Similarity Matching Engine — deterministic, LLM-free case matching.

Queries the SQLite case library to find historically similar trades using
weighted scoring across setup_type, market_regime, RSI distance, VWAP
alignment, and EMA trend alignment. Returns top matches ranked by
similarity score and computes aggregate performance statistics.
"""

import logging
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

log = logging.getLogger(__name__)

# Weighted similarity criteria (sum to 1.0)
SIMILARITY_WEIGHTS = {
    "setup_match": 0.30,
    "regime_match": 0.25,
    "rsi_distance": 0.15,
    "vwap_alignment": 0.15,
    "trend_alignment": 0.15,
}


def compute_similarity_score(case: dict, signal: dict) -> float:
    """
    Compute weighted similarity score between a historical case and the
    current signal.  Returns a float in [0.0, 1.0].

    Criteria:
      - setup_type:    1.0 if exact match, else 0.0
      - market_regime: 1.0 if exact match, else 0.0
      - rsi_distance:  1.0 - abs(case_rsi - signal_rsi) / 100, clamped [0, 1]
      - vwap_alignment: 1.0 if both above_vwap values match, else 0.0
      - trend_alignment: 1.0 if ema_trend matches, else 0.0
    """
    w = SIMILARITY_WEIGHTS

    # setup_type
    setup = 1.0 if case.get("setup_type") == signal.get("setup_type") else 0.0

    # market_regime
    regime = 1.0 if case.get("market_regime") == signal.get("market_regime") else 0.0

    # rsi_distance (continuous)
    try:
        case_rsi = float(case.get("rsi_at_entry", 50))
        signal_rsi = float(signal.get("rsi", signal.get("rsi_at_entry", 50)))
        rsi_score = max(0.0, min(1.0, 1.0 - abs(case_rsi - signal_rsi) / 100.0))
    except (TypeError, ValueError):
        rsi_score = 0.0

    # vwap_alignment
    case_vwap = str(case.get("above_vwap", "")).lower()
    signal_vwap = signal.get("above_vwap")
    # Normalize signal vwap to string for comparison
    if isinstance(signal_vwap, bool):
        signal_vwap_str = "true" if signal_vwap else "false"
    else:
        signal_vwap_str = str(signal_vwap).lower()
    vwap = 1.0 if case_vwap == signal_vwap_str else 0.0

    # trend_alignment
    case_trend = str(case.get("ema_trend", "")).lower()
    signal_trend = str(signal.get("ema_trend", "")).lower()
    trend = 1.0 if case_trend == signal_trend and case_trend != "" else 0.0

    score = (
        w["setup_match"] * setup
        + w["regime_match"] * regime
        + w["rsi_distance"] * rsi_score
        + w["vwap_alignment"] * vwap
        + w["trend_alignment"] * trend
    )
    return max(0.0, min(1.0, score))


def find_similar_cases(signal: dict, engine) -> list[dict]:
    """
    Query the case library for historically similar trades using weighted
    scoring.  Returns top 10 matches sorted by descending similarity score.

    Handles DB errors gracefully — logs and returns an empty list.
    """
    from models.case import Case
    import json

    try:
        SessionFactory = sessionmaker(bind=engine)
        db = SessionFactory()

        cases = db.query(Case).all()

        results = []
        for c in cases:
            case_dict = {
                "id": c.id,
                "date": c.date,
                "symbol": c.symbol,
                "setup_type": c.setup_type,
                "catalyst_type": c.catalyst_type,
                "float_profile": c.float_profile,
                "sector": c.sector,
                "premarket_gap_pct": c.premarket_gap_pct,
                "premarket_volume_rank": c.premarket_volume_rank,
                "market_regime": c.market_regime,
                "entry_timing": c.entry_timing,
                "bias": c.bias,
                "signal_strength": c.signal_strength,
                "rsi_at_entry": c.rsi_at_entry,
                "above_vwap": c.above_vwap,
                "above_daily_resistance": c.above_daily_resistance,
                "ema_trend": c.ema_trend,
                "outcome": c.outcome,
                "pnl_pct": c.pnl_pct,
                "holding_minutes": c.holding_minutes,
                "lesson": c.lesson,
                "conditions_for_success": (
                    json.loads(c.conditions_for_success)
                    if c.conditions_for_success
                    else []
                ),
                "conditions_to_avoid": (
                    json.loads(c.conditions_to_avoid)
                    if c.conditions_to_avoid
                    else []
                ),
                "confidence": c.confidence,
                "selection_score": c.selection_score,
                "execution_score": c.execution_score,
                "review_score": c.review_score,
                "profile": c.profile,
            }
            score = compute_similarity_score(case_dict, signal)
            case_dict["similarity_score"] = score
            results.append(case_dict)

        db.close()

        # Sort by descending similarity score, return top 10
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:10]

    except Exception as exc:
        log.error("Similarity engine DB error: %s", exc)
        return []


def compute_similarity_stats(cases: list[dict]) -> dict:
    """
    Aggregate performance statistics from matched cases.

    When cases is empty, returns a dict with skip_similarity=True so the
    edge score computation skips similarity weighting entirely.

    Otherwise returns:
      - similarity_winrate: fraction of cases with outcome="success"
      - similarity_avg_r: average pnl_pct (excluding nulls)
      - sample_size: number of cases
      - similarity_confidence: min(1.0, sample_size / 10)
    """
    if not cases:
        return {
            "similarity_winrate": 0.0,
            "similarity_avg_r": 0.0,
            "sample_size": 0,
            "similarity_confidence": 0.0,
            "skip_similarity": True,
        }

    sample_size = len(cases)
    wins = sum(1 for c in cases if c.get("outcome") == "success")
    winrate = wins / sample_size

    pnl_values = [c["pnl_pct"] for c in cases if c.get("pnl_pct") is not None]
    avg_r = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0

    confidence = min(1.0, sample_size / 10.0)

    return {
        "similarity_winrate": winrate,
        "similarity_avg_r": avg_r,
        "sample_size": sample_size,
        "similarity_confidence": confidence,
    }
