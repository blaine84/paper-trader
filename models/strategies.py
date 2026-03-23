"""
Strategy Library
Encoded definitions of known day trading strategies.
Each strategy has documented edge, ideal conditions, failure modes,
execution notes, and which agents it most affects.
"""

STRATEGIES = {

    "gap_and_go": {
        "name": "Gap and Go",
        "description": "Trade in the direction of a pre-market gap, expecting continuation at open.",
        "timeframe": "5-15 min",
        "bias": "LONG or SHORT depending on gap direction",
        "ideal_conditions": {
            "market_regime": ["risk_on"],
            "premarket_gap_pct": "> 2.0",
            "premarket_volume_rank": ["high", "extreme"],
            "entry_timing": ["first_15min"],
            "float_profile": ["small_cap", "mid_cap"],
            "catalyst_required": True,
        },
        "failure_conditions": [
            "market_regime=risk_off",
            "premarket_gap_pct < 1.5 (weak gap, fades)",
            "no clear catalyst — technical gap only",
            "entry_timing=open (too early, needs confirmation)",
            "rsi_at_entry > 80 (overextended at open)",
            "large_cap or mega_cap (gaps fill more often)",
        ],
        "execution_notes": [
            "Wait for first 5-min candle to close above VWAP before entry",
            "Stop below pre-market low or first candle low",
            "Target: measured move = gap size projected from breakout",
            "Avoid if market opens weak regardless of pre-market strength",
        ],
        "win_rate_documented": 0.58,   # documented historical baseline
        "avg_rr_documented": 2.1,
        "affects": ["analyst", "pm"],
        "tags": ["momentum", "gap", "open"],
    },

    "vwap_reclaim": {
        "name": "VWAP Reclaim",
        "description": "Stock dips below VWAP, then reclaims it — long the reclaim for continuation.",
        "timeframe": "5-15 min",
        "bias": "LONG",
        "ideal_conditions": {
            "market_regime": ["risk_on", "mixed"],
            "entry_timing": ["first_30min", "mid_day"],
            "above_vwap": False,              # price was below, now reclaiming
            "ema_trend": ["bullish"],
            "float_profile": ["mid_cap", "large_cap"],
        },
        "failure_conditions": [
            "market_regime=risk_off",
            "ema_trend=bearish (reclaim fails against trend)",
            "high volume on the dip (distribution, not pullback)",
            "reclaim attempt #3+ (multiple failures = weak)",
            "entry_timing=close (late day reclaims unreliable)",
        ],
        "execution_notes": [
            "Entry: on reclaim candle close above VWAP",
            "Stop: below the reclaim candle low",
            "Target: prior high or R:R 2:1 minimum",
            "Higher quality if VWAP is rising (not flat or declining)",
        ],
        "win_rate_documented": 0.55,
        "avg_rr_documented": 1.9,
        "affects": ["analyst", "pm"],
        "tags": ["vwap", "pullback", "intraday"],
    },

    "orb": {
        "name": "Opening Range Breakout (ORB)",
        "description": "Define the high/low of the first 15 or 30 minutes. Trade the breakout of that range.",
        "timeframe": "15-30 min range, then 5-min entries",
        "bias": "LONG or SHORT depending on breakout direction",
        "ideal_conditions": {
            "market_regime": ["risk_on"],
            "entry_timing": ["first_30min"],
            "premarket_volume_rank": ["medium", "high", "extreme"],
            "float_profile": ["large_cap", "mega_cap", "etf"],
        },
        "failure_conditions": [
            "market_regime=risk_off (breakouts fail, reverse)",
            "low premarket volume (false breakouts)",
            "very wide opening range > 2% (less reliable)",
            "choppy pre-market (indecision = choppy open)",
            "chasing breakout > 0.5% past range high/low",
        ],
        "execution_notes": [
            "Define range: high and low of first 15 (or 30) minutes",
            "Entry: on break and close outside the range",
            "Stop: middle of the range or opposite side of range",
            "Target: range height projected from breakout level",
            "Works well on index ETFs (SPY, QQQ, IWM)",
        ],
        "win_rate_documented": 0.52,
        "avg_rr_documented": 2.4,
        "affects": ["analyst", "pm"],
        "tags": ["breakout", "open", "etf"],
    },

    "momentum_fade": {
        "name": "Momentum Fade",
        "description": "Fade an overextended move — short a parabolic spike or long a capitulation flush.",
        "timeframe": "1-5 min entry, 5-15 min hold",
        "bias": "SHORT on spikes, LONG on flushes",
        "ideal_conditions": {
            "market_regime": ["risk_off", "mixed"],
            "rsi_at_entry": "> 80 (short) or < 20 (long)",
            "bb_position": ["outside_upper", "outside_lower"],
            "entry_timing": ["first_30min", "power_hour"],
            "float_profile": ["small_cap", "mid_cap"],
            "catalyst_required": False,
        },
        "failure_conditions": [
            "market_regime=risk_on (momentum carries further than expected)",
            "strong catalyst driving the move (news = trend, not fade)",
            "small float stock (can squeeze beyond reason)",
            "entry_timing=mid_day (lower volume = less reliable fade)",
            "fading against the broader market trend",
        ],
        "execution_notes": [
            "Wait for exhaustion candle (high volume, long wick, no follow-through)",
            "Stop: above the spike high (short) or below flush low (long)",
            "Target: VWAP reclaim or prior support/resistance",
            "Requires fast execution — fades can reverse quickly",
            "Never fade strong fundamental catalysts",
        ],
        "win_rate_documented": 0.51,
        "avg_rr_documented": 1.8,
        "affects": ["analyst", "pm"],
        "tags": ["reversal", "fade", "overextended"],
    },

    "trend_pullback": {
        "name": "Trend Pullback",
        "description": "In a clear intraday trend, enter on pullbacks to EMA or VWAP.",
        "timeframe": "5-15 min",
        "bias": "LONG in uptrend, SHORT in downtrend",
        "ideal_conditions": {
            "market_regime": ["risk_on"],
            "ema_trend": ["bullish"],
            "entry_timing": ["first_30min", "mid_day", "power_hour"],
            "above_vwap": True,
            "float_profile": ["mid_cap", "large_cap", "mega_cap"],
        },
        "failure_conditions": [
            "market_regime=risk_off",
            "pullback depth > 50% of prior move (too deep, trend may be broken)",
            "volume spike on pullback (distribution)",
            "multiple failed attempts to resume trend",
            "entry_timing=close (late day trend entries often reverse)",
        ],
        "execution_notes": [
            "Entry: on touch of 9 or 21 EMA, or VWAP, with bullish candle",
            "Stop: below EMA or below VWAP",
            "Target: new high (long) or new low (short) with R:R > 2:1",
            "Higher quality if market (SPY/QQQ) is also trending same direction",
        ],
        "win_rate_documented": 0.57,
        "avg_rr_documented": 2.2,
        "affects": ["analyst", "pm"],
        "tags": ["trend", "pullback", "ema"],
    },

    "news_catalyst": {
        "name": "News Catalyst Trade",
        "description": "Trade a directional move driven by a clear, material news catalyst.",
        "timeframe": "5-30 min",
        "bias": "Direction of catalyst",
        "ideal_conditions": {
            "catalyst_type": ["earnings_beat", "analyst_upgrade", "product_launch",
                              "short_squeeze", "regulatory"],
            "premarket_volume_rank": ["high", "extreme"],
            "market_regime": ["risk_on", "mixed"],
            "float_profile": ["small_cap", "mid_cap"],
        },
        "failure_conditions": [
            "catalyst already known / priced in",
            "weak catalyst — rumor, speculative, vague",
            "market_regime=risk_off (even good news fades)",
            "earnings beat but guidance cut (sell the news)",
            "analyst upgrade on overextended stock",
        ],
        "execution_notes": [
            "Verify catalyst is material, not priced in",
            "Entry: after initial volatility settles (5-15 min after open)",
            "Stop: below catalyst candle low or pre-market low",
            "Target: measured move or key resistance",
            "Avoid earnings plays at open — too volatile",
        ],
        "win_rate_documented": 0.60,
        "avg_rr_documented": 2.0,
        "affects": ["scout", "researcher", "analyst", "pm"],
        "tags": ["catalyst", "news", "momentum"],
    },

    "sector_rotation": {
        "name": "Sector Rotation",
        "description": "Trade into a sector receiving capital inflows, out of one losing flows.",
        "timeframe": "intraday to multi-day",
        "bias": "LONG strong sector, SHORT weak sector",
        "ideal_conditions": {
            "market_regime": ["risk_on", "mixed"],
            "catalyst_type": ["macro_event", "sector_move"],
            "float_profile": ["large_cap", "mega_cap", "etf"],
        },
        "failure_conditions": [
            "rotation signal is noise, not sustained",
            "trading individual names instead of sector ETFs (more risk)",
            "market_regime=risk_off (rotation pauses, everything sells)",
        ],
        "execution_notes": [
            "Use sector ETFs (XLK, XLE, XLF) for cleaner exposure",
            "Confirm rotation with relative strength vs SPY",
            "Look for sector ETF breaking out while SPY is flat or down",
        ],
        "win_rate_documented": 0.54,
        "avg_rr_documented": 2.0,
        "affects": ["researcher", "scout", "pm"],
        "tags": ["sector", "rotation", "etf"],
    },

    "short_squeeze": {
        "name": "Short Squeeze",
        "description": "High short interest stock gets a catalyst — forced covering amplifies upside.",
        "timeframe": "5-30 min, can run for days",
        "bias": "LONG",
        "ideal_conditions": {
            "catalyst_type": ["short_squeeze", "earnings_beat", "analyst_upgrade"],
            "premarket_gap_pct": "> 5.0",
            "premarket_volume_rank": ["extreme"],
            "float_profile": ["small_cap", "micro_cap"],
            "market_regime": ["risk_on"],
        },
        "failure_conditions": [
            "no confirmed short interest data",
            "large_cap or mega_cap (not enough short pressure)",
            "market_regime=risk_off",
            "entry after 50%+ move already (too late)",
            "chasing into resistance without pullback",
        ],
        "execution_notes": [
            "Very high risk — use smaller position size",
            "Entry: on first pullback after initial spike, not at top",
            "Stop: tight — below pullback low",
            "Target: take partial profits at resistance, trail stop on rest",
            "Can move much further than expected — don't cap upside too early",
        ],
        "win_rate_documented": 0.45,
        "avg_rr_documented": 3.5,
        "affects": ["scout", "pm_aggressive"],
        "tags": ["squeeze", "catalyst", "high_risk"],
    },

}

# Strategies indexed by tag for fast lookup
STRATEGY_TAGS = {}
for key, strat in STRATEGIES.items():
    for tag in strat.get("tags", []):
        STRATEGY_TAGS.setdefault(tag, []).append(key)

# Strategies by setup_type alignment (maps case library setup_type → strategy key)
SETUP_TYPE_MAP = {
    "gap_and_go":           "gap_and_go",
    "vwap_reclaim":         "vwap_reclaim",
    "range_breakout":       "orb",
    "technical_breakout":   "orb",
    "momentum_fade":        "momentum_fade",
    "reversal":             "momentum_fade",
    "trend_continuation":   "trend_pullback",
    "news_breakout":        "news_catalyst",
    "earnings_reaction":    "news_catalyst",
    "sector_rotation":      "sector_rotation",
    "short_squeeze":        "short_squeeze",
}
