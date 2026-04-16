"""
Portfolio Manager Risk Profiles
Each profile defines its own trading personality and constraints.
All profiles run in parallel with isolated portfolios.
"""

PM_PROFILES = {
    "conservative": {
        "name": "Conservative",
        "emoji": "🛡️",
        "description": "Capital preservation first. Only high-conviction setups with tight risk.",
        "starting_balance": 100_000,
        "max_positions": 2,
        "max_position_pct": 0.15,        # Max 15% of portfolio per trade
        "min_risk_reward": 3.0,          # Only take R:R >= 3:1
        "min_signal_strength": "strong", # Only strong signals
        "min_conviction": "high",        # Only high-conviction scout picks
        "stop_loss_multiplier": 1.0,     # Use analyst stop as-is
        "target_multiplier": 1.0,        # Use analyst target as-is
        "avoid_first_minutes": 30,       # No trades in first 30 min
        "avoid_last_minutes": 30,        # No trades in last 30 min
        "max_daily_loss_pct": 0.02,      # Stop trading if down 2% on day
        "opposing_evidence_threshold": "moderate",  # Moderate opposing signal triggers Reversal Review
        "personality": """
You are a conservative portfolio manager. Capital preservation is your top priority.
Rules:
- Only take trades with strong signals and high conviction
- Minimum 3:1 risk/reward ratio
- Max 2 positions at once, max 15% of portfolio per trade
- Avoid the first and last 30 minutes of trading
- If down 2% on the day, stop trading
- When in doubt, HOLD. Missing a trade is better than a bad trade.
- Prefer ETFs (SPY, QQQ, IWM) over individual stocks
""",
    },

    "moderate": {
        "name": "Moderate",
        "emoji": "⚖️",
        "description": "Balanced risk/reward. Takes quality setups across the watchlist.",
        "starting_balance": 100_000,
        "max_positions": 3,
        "max_position_pct": 0.25,
        "min_risk_reward": 2.0,
        "min_signal_strength": "moderate",
        "min_conviction": "medium",
        "stop_loss_multiplier": 1.0,
        "target_multiplier": 1.0,
        "avoid_first_minutes": 15,
        "avoid_last_minutes": 15,
        "max_daily_loss_pct": 0.03,
        "opposing_evidence_threshold": "strong",  # Only strong opposing signals trigger Reversal Review
        "personality": """
You are a balanced portfolio manager. You seek quality setups with good risk/reward.
Rules:
- Take moderate-to-strong signals with at least 2:1 risk/reward
- Max 3 positions, max 25% per trade
- Avoid first and last 15 minutes
- Stop trading if down 3% on the day
- Diversify across sectors when possible
- Trust the analyst but use your judgment
""",
    },

    "aggressive": {
        "name": "Aggressive",
        "emoji": "🔥",
        "description": "High risk, high reward. Trades more frequently, larger size, chases momentum.",
        "starting_balance": 100_000,
        "max_positions": 4,
        "max_position_pct": 0.35,
        "min_risk_reward": 1.5,
        "min_signal_strength": "weak",   # Will trade weaker signals
        "min_conviction": "low",
        "stop_loss_multiplier": 1.2,     # Wider stops
        "target_multiplier": 1.3,        # More ambitious targets
        "avoid_first_minutes": 5,
        "avoid_last_minutes": 5,
        "max_daily_loss_pct": 0.05,
        "opposing_evidence_threshold": "strong",  # Only strong opposing signals trigger Reversal Review
        "personality": """
You are an aggressive day trader. You chase momentum and aren't afraid of risk.
Rules:
- Take any signal with at least 1.5:1 risk/reward
- Up to 4 positions, up to 35% per trade
- Embrace volatility — individual stocks (TSLA, NVDA, AMD) preferred
- Use wider stops and bigger targets
- Act on Scout picks early — they often move fast
- Stop trading only if down 5% on the day
- Don't be afraid to pyramid into winning positions
""",
    },
}

# Which profiles are active (all run in parallel)
ACTIVE_PROFILES = ["conservative", "moderate", "aggressive"]
