"""
Edge Score Calculator — deterministic, LLM-free trade scoring.

Computes a continuous 0.0–1.0 edge score for proposed trades using
six weighted components: setup win rate, similarity win rate, signal
strength, signal confidence, indicator confluence, and similarity quality.

Includes hard rejection for proven-bad setups and a position sizing cap.
"""

# 6-component weights (sum to 1.0)
WEIGHTS = {
    "setup_winrate": 0.25,
    "similarity_winrate": 0.20,
    "signal_strength": 0.15,
    "signal_confidence": 0.10,
    "confluence": 0.15,
    "similarity_quality": 0.15,
}

_STRENGTH_MAP = {
    "strong": 1.0,
    "moderate": 0.6,
    "weak": 0.3,
}

_CONFIDENCE_MAP = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
}


def normalize_winrate(win_rate: float) -> float:
    """Clamp *win_rate* to [0.0, 1.0]. Returns 0.0 for non-numeric inputs."""
    try:
        return max(0.0, min(1.0, float(win_rate)))
    except (TypeError, ValueError):
        return 0.0


def map_strength(strength: str) -> float:
    """Map strength string to numeric: strong=1.0, moderate=0.6, weak=0.3, default=0.0."""
    return _STRENGTH_MAP.get(str(strength).lower(), 0.0)


def map_confidence(confidence: str) -> float:
    """Map confidence string to numeric: high=1.0, medium=0.6, low=0.3, default=0.0."""
    return _CONFIDENCE_MAP.get(str(confidence).lower(), 0.0)



def confluence_score(indicators: dict, bias: str) -> float:
    """
    Count aligned indicators and return a 0.0–1.0 score.

    Checks five indicators — each aligned one contributes 0.2:
      • above_vwap: True for LONG, False for SHORT
      • ema_trend: "bullish" for LONG, "bearish" for SHORT
      • rsi: 30–70 for LONG, outside that range inverted for SHORT
      • macd_bias: "bullish" for LONG, "bearish" for SHORT
      • bb_position: "upper" for LONG, "lower" for SHORT
    """
    if not indicators:
        return 0.0

    aligned = 0
    bias_upper = str(bias).upper()
    is_long = bias_upper == "LONG"

    # 1. above_vwap
    above_vwap = indicators.get("above_vwap", False)
    if (is_long and above_vwap) or (not is_long and not above_vwap):
        aligned += 1

    # 2. ema_trend
    ema_trend = str(indicators.get("ema_trend", "")).lower()
    if (is_long and ema_trend == "bullish") or (not is_long and ema_trend == "bearish"):
        aligned += 1

    # 3. rsi — favorable range
    try:
        rsi = float(indicators.get("rsi", -1))
        if is_long and 30 <= rsi <= 70:
            aligned += 1
        elif not is_long and (rsi > 70 or rsi < 30):
            aligned += 1
    except (TypeError, ValueError):
        pass  # non-numeric → not aligned

    # 4. macd_bias
    macd = str(indicators.get("macd_bias", "")).lower()
    if (is_long and macd == "bullish") or (not is_long and macd == "bearish"):
        aligned += 1

    # 5. bb_position
    bb = str(indicators.get("bb_position", "")).lower()
    if (is_long and bb == "upper") or (not is_long and bb == "lower"):
        aligned += 1

    return aligned / 5.0


def similarity_quality(similarity_sample_size: int) -> float:
    """Sample-size-aware confidence: min(1.0, similarity_sample_size / 10). Returns 0.0 for non-numeric."""
    try:
        return min(1.0, max(0, int(similarity_sample_size)) / 10.0)
    except (TypeError, ValueError):
        return 0.0


def check_hard_rejection(case_stats: dict) -> bool:
    """
    Return True if the trade should be hard-rejected.

    Triggers when case_stats sample_size >= 10 AND win_rate < 0.35.
    """
    sample_size = case_stats.get("sample_size", 0)
    win_rate = case_stats.get("win_rate", 0.0)
    return sample_size >= 10 and win_rate < 0.35


def cap_position_size(scaled_size: float, base_size: float) -> float:
    """Cap scaled position size at base_size × 1.2."""
    return min(float(scaled_size), float(base_size) * 1.2)


def compute_edge_score(
    signal: dict, case_stats: dict, similarity_stats: dict
) -> float:
    """
    Compute a 0.0–1.0 edge score for a proposed trade.

    Six weighted components (see WEIGHTS):
      0.25 × normalize_winrate(case_stats.win_rate)
      0.20 × normalize_winrate(similarity_stats.similarity_winrate)
      0.15 × map_strength(signal.strength)
      0.10 × map_confidence(signal.confidence)
      0.15 × confluence_score(signal.indicators, signal.bias)
      0.15 × similarity_quality(similarity_stats.sample_size)

    Returns a float clamped to [0.0, 1.0].
    """
    setup_wr = normalize_winrate(case_stats.get("win_rate", 0.0))
    sim_wr = normalize_winrate(similarity_stats.get("similarity_winrate", 0.0))
    strength = map_strength(signal.get("strength", ""))
    confidence = map_confidence(signal.get("confidence", ""))
    confluence = confluence_score(signal.get("indicators", {}), signal.get("bias", ""))
    sim_qual = similarity_quality(similarity_stats.get("sample_size", 0))

    raw = (
        WEIGHTS["setup_winrate"] * setup_wr
        + WEIGHTS["similarity_winrate"] * sim_wr
        + WEIGHTS["signal_strength"] * strength
        + WEIGHTS["signal_confidence"] * confidence
        + WEIGHTS["confluence"] * confluence
        + WEIGHTS["similarity_quality"] * sim_qual
    )

    return max(0.0, min(1.0, raw))
