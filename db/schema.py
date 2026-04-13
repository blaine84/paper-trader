"""
Database schema and initialization.
Uses SQLite via SQLAlchemy.
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, 
    DateTime, Text, Boolean, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()


class Trade(Base):
    """A paper trade (open or closed)."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    profile = Column(String(16), default="moderate")  # conservative|moderate|aggressive
    symbol = Column(String(10), nullable=False)
    direction = Column(String(5), nullable=False)  # LONG | SHORT
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    status = Column(String(8), default="open")  # open | closed
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    reason_entry = Column(Text, nullable=True)
    reason_exit = Column(Text, nullable=True)
    stop_price = Column(Float, nullable=True)     # PM's stop loss level
    target_price = Column(Float, nullable=True)    # PM's profit target
    review_score = Column(Float, nullable=True)  # 1-10 from Reviewer
    review_notes = Column(Text, nullable=True)
    edge_score = Column(Float, nullable=True)                # 0.0-1.0
    similarity_winrate = Column(Float, nullable=True)        # 0.0-1.0
    similarity_sample_size = Column(Integer, nullable=True)  # count of matched cases
    similarity_confidence = Column(Float, nullable=True)     # min(1.0, sample_size/10)


class Position(Base):
    """Current open positions (long or short)."""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    profile = Column(String(16), default="moderate")  # which PM owns this
    symbol = Column(String(10), nullable=False)
    side = Column(String(5), default="long")          # long | short
    quantity = Column(Float, nullable=False)           # always positive
    avg_cost = Column(Float, nullable=False)           # entry price
    opened_at = Column(DateTime, default=datetime.utcnow)


class Balance(Base):
    """Cash balance snapshots."""
    __tablename__ = "balance"

    id = Column(Integer, primary_key=True)
    profile = Column(String(16), default="moderate")  # which PM portfolio
    timestamp = Column(DateTime, default=datetime.utcnow)
    cash = Column(Float, nullable=False)
    portfolio_value = Column(Float, nullable=True)
    total_equity = Column(Float, nullable=True)


class AgentMemory(Base):
    """Persistent notes/feedback shared between agents."""
    __tablename__ = "agent_memory"

    id = Column(Integer, primary_key=True)
    agent = Column(String(32), nullable=False)   # researcher|analyst|pm|reviewer
    symbol = Column(String(10), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    key = Column(String(64), nullable=False)     # e.g. "lesson", "signal", "feedback"
    value = Column(Text, nullable=False)


class ReviewQueue(Base):
    """Queue for trades pending review."""
    __tablename__ = "review_queue"

    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, nullable=False)
    status = Column(String(16), default="pending")  # pending | reviewed | failed
    queued_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)


class DynamicStrategy(Base):
    """Agent-proposed strategies that supplement the hardcoded strategy library."""
    __tablename__ = "dynamic_strategies"

    id = Column(Integer, primary_key=True)
    key = Column(String(64), nullable=False, unique=True)  # e.g. "vwap_fade_eod"
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=False)
    timeframe = Column(String(32))
    bias = Column(String(32))                    # LONG | SHORT | either
    ideal_conditions = Column(Text)              # JSON
    failure_conditions = Column(Text)            # JSON
    execution_notes = Column(Text)               # JSON
    proposed_by = Column(String(32), default="quant_researcher")
    status = Column(String(16), default="active")  # active | retired | probation
    total_trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    avg_pnl_pct = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    retired_at = Column(DateTime, nullable=True)
    retire_reason = Column(Text, nullable=True)


class DailyLog(Base):
    """End-of-day summaries."""
    __tablename__ = "daily_log"

    id = Column(Integer, primary_key=True)
    date = Column(String(10), nullable=False)    # YYYY-MM-DD
    starting_equity = Column(Float)
    ending_equity = Column(Float)
    trades_taken = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    daily_pnl = Column(Float)
    daily_pnl_pct = Column(Float)
    notes = Column(Text)


def init_db(db_path: str = "db/paper_trader.db"):
    engine = create_engine(f"sqlite:///{db_path}")
    # Import Case model so it registers with Base before create_all
    from models.case import Case  # noqa: F401
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
