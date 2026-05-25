"""
Case Library Schema
Structured trade cases stored as typed records, not prose.
Think: each closed trade generates a case. Agents query cases by setup type,
catalyst, outcome, etc. to inform future decisions.
"""

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Text, Index
)
from datetime import datetime
from db.schema import Base


class Case(Base):
    """
    A structured lesson extracted from a closed trade.
    This is the core of the agent memory system — a queryable case library.
    """
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    trade_id = Column(Integer, nullable=True)         # FK to trades.id
    profile = Column(String(16), nullable=True)       # which PM profile

    # --- Setup Classification ---
    symbol = Column(String(10), nullable=False)
    date = Column(String(10), nullable=False)         # YYYY-MM-DD

    setup_type = Column(String(32), nullable=True)
    # e.g.: news_breakout | gap_and_go | technical_breakout | momentum_fade |
    #       gap_fill | range_breakout | vwap_reclaim | earnings_reaction |
    #       sector_rotation | reversal

    catalyst_type = Column(String(32), nullable=True)
    # e.g.: analyst_upgrade | analyst_downgrade | earnings_beat | earnings_miss |
    #       macro_event | product_launch | sector_move | short_squeeze |
    #       technical_only | news_headline | regulatory

    # --- Stock Characteristics ---
    float_profile = Column(String(12), nullable=True)
    # e.g.: micro_cap | small_cap | mid_cap | large_cap | mega_cap

    sector = Column(String(32), nullable=True)
    # e.g.: tech | energy | financials | healthcare | consumer | industrials | etf

    # --- Pre-market / Entry Context ---
    premarket_gap_pct = Column(Float, nullable=True)  # % gap from prev close
    premarket_volume_rank = Column(String(8), nullable=True)  # low|medium|high|extreme
    market_regime = Column(String(12), nullable=True)         # risk_on|risk_off|mixed
    entry_timing = Column(String(16), nullable=True)
    # e.g.: open | first_15min | first_30min | mid_day | power_hour | close

    # --- Analyst Signal ---
    bias = Column(String(5), nullable=True)             # LONG | SHORT
    signal_strength = Column(String(8), nullable=True)  # weak|moderate|strong
    signal_confidence = Column(String(8), nullable=True) # low|medium|high
    invalidation = Column(Text, nullable=True)          # what would have killed the setup
    rsi_at_entry = Column(Float, nullable=True)
    above_vwap = Column(String(5), nullable=True)       # true|false
    above_daily_resistance = Column(String(5), nullable=True)  # true|false
    ema_trend = Column(String(8), nullable=True)        # bullish|bearish|neutral
    bb_position = Column(String(16), nullable=True)     # upper|middle|lower|outside_*

    # --- PM Execution (separate from signal) ---
    entry_vs_level = Column(String(16), nullable=True)
    # how PM entered relative to key levels: at_support|above_vwap|at_breakout|chased|etc

    # --- Outcome ---
    outcome = Column(String(8), nullable=False)       # success | failure | partial
    pnl_pct = Column(Float, nullable=True)
    holding_minutes = Column(Integer, nullable=True)  # how long the trade was held
    exit_category = Column(String(40), nullable=True)
    # One of: bad_entry | valid_entry_bad_exit_policy |
    #         valid_exit_thesis_invalidated | forced_exit_missing_metadata

    # --- Structured Lesson ---
    lesson = Column(Text, nullable=False)
    # One actionable sentence. e.g.:
    # "strongest when market regime risk_on and stock above daily resistance"

    conditions_for_success = Column(Text, nullable=True)
    # JSON list of conditions that made this work (or would have made it work)
    # e.g.: ["market_regime=risk_on", "above_daily_resistance=true", "premarket_gap_pct>3"]

    conditions_to_avoid = Column(Text, nullable=True)
    # JSON list of conditions that hurt this trade
    # e.g.: ["entry_timing=open", "rsi_at_entry>75"]

    confidence = Column(String(8), nullable=True)     # low|medium|high

    # --- Separated Scores ---
    selection_score = Column(Float, nullable=True)
    # 1-10: Did Scout/Analyst identify the right stock at the right time?
    # Did the setup and catalyst actually play out as expected?
    # Feeds back to: Scout + Analyst

    execution_score = Column(Float, nullable=True)
    # 1-10: Given we were in the trade, did PM enter, size, and exit well?
    # Entry timing, stop placement, target, position sizing, exit discipline
    # Feeds back to: PM (per profile)

    review_score = Column(Float, nullable=True)
    # 1-10: Composite overall score (avg of above, kept for backward compat)


# Indexes for fast case lookup
Index("ix_cases_setup_outcome", Case.setup_type, Case.outcome)
Index("ix_cases_catalyst_outcome", Case.catalyst_type, Case.outcome)
Index("ix_cases_symbol_date", Case.symbol, Case.date)
Index("ix_cases_market_regime", Case.market_regime, Case.outcome)
