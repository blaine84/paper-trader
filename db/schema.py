"""
Database schema and initialization.
Supports both SQLite (development/test) and Postgres (production) via DATABASE_URL.
"""

import logging
import os

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Date, Text, Boolean, ForeignKey, Index,
    event, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

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
    # Entry Contract fields (thesis-anchored exits)
    thesis = Column(Text, nullable=True)                     # trade thesis narrative
    setup_type = Column(String(64), nullable=True)           # analyst's setup classification
    invalidators = Column(Text, nullable=True)               # JSON array of invalidator objects

    # Stop metadata (StopAuthority)
    stop_role = Column(String(32), default="initial")       # initial|breakeven|trail|manual|maintenance_tighten
    stop_updated_by = Column(String(64), nullable=True)     # agent that last modified stop
    stop_updated_at = Column(DateTime, nullable=True)       # when stop was last modified
    candidate_lineage_id = Column(String(36), nullable=True, index=True)
    pm_candidate_id = Column(String(36), nullable=True, index=True)



class TradeEvent(Base):
    """Normalized audit log for trade lifecycle decisions and outcomes."""
    __tablename__ = "trade_events"

    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    event_type = Column(String(64), nullable=False)
    agent = Column(String(64), nullable=True)
    symbol = Column(String(10), nullable=True)
    profile = Column(String(16), nullable=True)
    price = Column(Float, nullable=True)
    message = Column(Text, nullable=True)
    payload_json = Column(Text, nullable=True)
    dedupe_key = Column(String(256), nullable=True, index=True)
    candidate_lineage_id = Column(String(36), nullable=True, index=True)
    pm_candidate_id = Column(String(36), nullable=True, index=True)


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


class AnalystFeedbackQueue(Base):
    """Reviewer-raised quality flags that require an analyst response."""
    __tablename__ = "analyst_feedback_queue"

    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, nullable=True)
    symbol = Column(String(10), nullable=False)
    setup_type = Column(String(64), nullable=True)
    date = Column(String(10), nullable=False)  # YYYY-MM-DD case date
    flag_type = Column(String(64), nullable=False)
    severity = Column(String(16), nullable=False)  # low | medium | high | critical
    recommendation = Column(Text, nullable=False)
    reviewer_context = Column(Text, nullable=True)  # JSON payload from reviewer case
    status = Column(String(16), default="pending")  # pending | responded | overdue
    created_at = Column(DateTime, default=datetime.utcnow)
    due_at = Column(DateTime, nullable=False)
    responded_at = Column(DateTime, nullable=True)
    analyst_response = Column(String(16), nullable=True)  # accept | reject | modify
    analyst_response_note = Column(Text, nullable=True)
    analyst_supporting_data = Column(Text, nullable=True)  # JSON array or object
    no_data_reject = Column(Boolean, default=False)


class AnalystMitigation(Base):
    """Active conservative throttles applied to analyst setup classifications."""
    __tablename__ = "analyst_mitigations"

    id = Column(Integer, primary_key=True)
    setup_type = Column(String(64), nullable=False, unique=True)
    level = Column(Integer, default=0)
    deployment_multiplier = Column(Float, default=1.0)
    signal_threshold_bump = Column(Float, default=0.0)
    active = Column(Boolean, default=False)
    reason = Column(Text, nullable=True)
    applied_at = Column(DateTime, nullable=True)
    last_triggered_at = Column(DateTime, nullable=True)
    reset_at = Column(DateTime, nullable=True)


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
    status = Column(String(16), default="active")  # active | retired | probation | backtest | paper_trade | live_50 | live_100 | backtest_failed
    total_trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    avg_pnl_pct = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    retired_at = Column(DateTime, nullable=True)
    retire_reason = Column(Text, nullable=True)
    # Pipeline tracking columns
    pipeline_stage = Column(String(16), nullable=True)        # backtest | paper_trade | live_50 | live_100
    backtest_report_id = Column(String(128), nullable=True)   # AgentMemory key reference
    paper_trade_start_date = Column(DateTime, nullable=True)
    live_50_start_date = Column(DateTime, nullable=True)
    live_100_start_date = Column(DateTime, nullable=True)
    failure_stage = Column(String(16), nullable=True)         # which stage caused failure
    failure_reason = Column(Text, nullable=True)              # human-readable reason


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


class FunnelCandidate(Base):
    """Persistent premarket candidate funnel record with stage history."""
    __tablename__ = "funnel_candidates"

    id = Column(Integer, primary_key=True)
    candidate_id = Column(String(36), nullable=False, default=lambda: str(uuid.uuid4()))
    date = Column(Date, nullable=False)  # New York trading date (America/New_York)
    symbol = Column(String(10), nullable=False)
    discovered_at = Column(DateTime, nullable=False)  # UTC timestamp
    source_run = Column(String(32), nullable=False)  # premarket|confirmation|manual_intraday
    selection_mode = Column(String(32), nullable=False)  # chief_scout|deterministic_fallback
    scout_rank = Column(Integer, nullable=False)
    scout_score = Column(Float, nullable=False)
    direction_bias = Column(String(10), nullable=True)  # bullish|bearish|neutral
    catalyst_evidence = Column(Text, nullable=False)  # JSON
    selection_reason = Column(Text, nullable=False)
    primary_risk = Column(Text, nullable=False)
    sector_context = Column(Text, nullable=True)  # JSON
    preliminary_setup_type = Column(String(32), nullable=True)
    authoritative_setup_type = Column(String(32), nullable=True)
    stage_status = Column(String(32), nullable=False, default="awaiting_research")
    stage_decisions = Column(Text, nullable=False, default="[]")  # JSON array
    trade_event_id = Column(Integer, ForeignKey("trade_events.id"), nullable=True)
    blocked_candidate_id = Column(Integer, nullable=True)
    expired = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_funnel_date_status", "date", "stage_status"),
        Index("ix_funnel_date_symbol", "date", "symbol", unique=True),
        Index("ix_funnel_candidate_id", "candidate_id", unique=True),
    )


class FunnelRunLog(Base):
    """Operational log for each funnel pipeline execution."""
    __tablename__ = "funnel_run_logs"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    stage = Column(String(32), nullable=False)  # discovery|research|analysis|confirmation
    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    budget_seconds = Column(Float, nullable=False)
    result_status = Column(String(32), nullable=False)  # completed|timed_out|degraded|error
    sectors_completed = Column(Text, nullable=True)  # JSON array
    sectors_timed_out = Column(Text, nullable=True)  # JSON array
    candidates_input = Column(Integer, nullable=True)
    candidates_promoted = Column(Integer, nullable=True)
    candidates_rejected = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def is_sqlite(engine) -> bool:
    """Return True if the engine dialect is SQLite."""
    return engine.dialect.name == "sqlite"


def init_db(db_path: str = "db/paper_trader.db"):
    database_url = os.environ.get("DATABASE_URL", "").strip()

    if database_url:
        # Postgres branch
        engine = create_engine(database_url, pool_pre_ping=True)
        # Verify connection at startup (fail-closed)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        # No PRAGMA listener is registered here.
    else:
        # SQLite branch
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"timeout": 30},
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    # Common path (both dialects)
    from models.case import Case  # noqa: F401
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


def verify_wal_mode(engine) -> bool:
    """Verify WAL mode and busy_timeout are configured on the SQLite database.

    Checks PRAGMA journal_mode and busy_timeout. If either is not set correctly,
    applies the correct settings and logs a warning.

    Returns True if settings were already correct, False if corrections were made.

    Requirements: 12.1, 12.2
    """
    corrections_made = False

    with engine.connect() as conn:
        # Check journal_mode
        result = conn.execute(text("PRAGMA journal_mode")).scalar()
        if result and result.lower() != "wal":
            conn.execute(text("PRAGMA journal_mode=WAL"))
            logger.warning(
                "WAL mode correction: journal_mode was '%s', set to WAL",
                result,
            )
            corrections_made = True

        # Check busy_timeout
        timeout = conn.execute(text("PRAGMA busy_timeout")).scalar()
        if timeout is not None and int(timeout) == 0:
            conn.execute(text("PRAGMA busy_timeout=30000"))
            logger.warning(
                "WAL mode correction: busy_timeout was 0, set to 30000",
            )
            corrections_made = True

        conn.commit()

    return not corrections_made

# ---------------------------------------------------------------------------
# Triggered trade plans schema (non-destructive, IF NOT EXISTS)
#
# Trade plans have their own lifecycle/state machine, separate from
# pm_candidates. DDL below is idempotent and safe to run on every startup.
#
# Requirements: 1.8, 8.1, 9.1
# ---------------------------------------------------------------------------

_TRADE_PLANS_DDL = """
CREATE TABLE IF NOT EXISTS trade_plans (
    plan_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    geometry_name TEXT,

    entry_reference REAL NOT NULL,
    entry_zone_upper REAL NOT NULL,
    entry_zone_lower REAL NOT NULL,

    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    risk_reward REAL NOT NULL,

    trigger_type TEXT NOT NULL,
    trigger_condition_json TEXT NOT NULL,
    trigger_confirmation_required INTEGER NOT NULL DEFAULT 0,

    invalidation_logic_json TEXT,

    analyst_reasoning TEXT,
    pm_rationale TEXT,
    source_signal_id TEXT,
    signal_snapshot_json TEXT,

    state TEXT NOT NULL DEFAULT 'planned',
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    triggered_at TEXT,
    executed_at TEXT,
    missed_at TEXT,
    miss_reason TEXT,
    rejection_reason TEXT,

    integrity_hash TEXT NOT NULL
)
"""

_TRADE_PLANS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_trade_plans_state ON trade_plans(state)",
    "CREATE INDEX IF NOT EXISTS idx_trade_plans_symbol_state ON trade_plans(symbol, state)",
    "CREATE INDEX IF NOT EXISTS idx_trade_plans_candidate ON trade_plans(candidate_id)",
    "CREATE INDEX IF NOT EXISTS idx_trade_plans_cycle ON trade_plans(cycle_id, profile_id)",
]


# ---------------------------------------------------------------------------
# trade_plan_events (immutable, append-only audit trail)
#
# Every plan state transition emits a row here. Rows are never updated or
# deleted — immutability is enforced by database triggers.
#
# Requirements: 8.5, 9.4, 9.5
# ---------------------------------------------------------------------------

_TRADE_PLAN_EVENTS_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS trade_plan_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT,
    fresh_price REAL,
    from_state TEXT,
    to_state TEXT,
    created_at TEXT NOT NULL
)
"""

_TRADE_PLAN_EVENTS_DDL_POSTGRES = """
CREATE TABLE IF NOT EXISTS trade_plan_events (
    id SERIAL PRIMARY KEY,
    plan_id TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT,
    fresh_price DOUBLE PRECISION,
    from_state TEXT,
    to_state TEXT,
    created_at TEXT NOT NULL
)
"""

_TRADE_PLAN_EVENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_trade_plan_events_plan ON trade_plan_events(plan_id)",
    "CREATE INDEX IF NOT EXISTS idx_trade_plan_events_type ON trade_plan_events(event_type)",
]

_TRADE_PLAN_EVENTS_IMMUTABILITY_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS trade_plan_events_no_update
        BEFORE UPDATE ON trade_plan_events
    BEGIN
        SELECT RAISE(ABORT, 'trade_plan_events is immutable: UPDATE prohibited');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trade_plan_events_no_delete
        BEFORE DELETE ON trade_plan_events
    BEGIN
        SELECT RAISE(ABORT, 'trade_plan_events is immutable: DELETE prohibited');
    END
    """,
]


def init_trade_plan_schema(engine):
    """Create trade_plans and trade_plan_events tables if they do not exist.

    HISTORICAL COMPATIBILITY ONLY — the triggered-plan entry execution
    architecture was retired in the limit-order-mode-cleanup spec.
    No active runtime code inserts into these tables. They are retained so
    that:
    - Existing databases are not broken by a missing table reference.
    - Historical review/CEO queries that read trade_plan_events continue to work.
    - The immutability triggers remain in place protecting audit data.

    The 'missed_setup' trade event type is likewise dormant vocabulary — all
    three emitters (plan_monitor, plan_executor, TradePlanRegistry) were removed.
    Historical rows remain queryable.

    Do not add new INSERT paths to these tables.
    """
    sqlite = is_sqlite(engine)

    with engine.begin() as conn:
        conn.execute(text(_TRADE_PLANS_DDL))
        for stmt in _TRADE_PLANS_INDEXES:
            conn.execute(text(stmt))

        conn.execute(text(
            _TRADE_PLAN_EVENTS_DDL_SQLITE if sqlite
            else _TRADE_PLAN_EVENTS_DDL_POSTGRES
        ))
        for stmt in _TRADE_PLAN_EVENTS_INDEXES:
            conn.execute(text(stmt))

        if sqlite:
            for stmt in _TRADE_PLAN_EVENTS_IMMUTABILITY_TRIGGERS:
                conn.execute(text(stmt))
        else:
            # Postgres: shared raise_immutable() function + per-op triggers
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION raise_immutable() RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION '% is immutable: % prohibited', TG_TABLE_NAME, TG_OP;
                END;
                $$ LANGUAGE plpgsql
            """))
            for op in ("update", "delete"):
                conn.execute(text(
                    f"DROP TRIGGER IF EXISTS trade_plan_events_no_{op} "
                    f"ON trade_plan_events"
                ))
                conn.execute(text(
                    f"CREATE TRIGGER trade_plan_events_no_{op} "
                    f"BEFORE {op.upper()} ON trade_plan_events "
                    f"FOR EACH ROW EXECUTE FUNCTION raise_immutable()"
                ))

    logger.debug("Trade plan schema verified (trade_plans, trade_plan_events)")


# ---------------------------------------------------------------------------
# pending_orders (resting paper limit orders)
#
# A pending order is a PM-approved entry intent whose limit price was not
# executable when the decision was made, because the fresh quote had already
# run away from the intended entry. It rests until market data crosses the
# limit inside its active window, or until it expires or is canceled.
#
# Deliberately a separate table from trade_plans rather than an extension of
# it: trade_plans declares entry_zone_upper/lower, trigger_type and
# trigger_condition_json NOT NULL, none of which mean anything for a
# single-price resting order, and coupling the two would tie this feature's
# rollout to the (now-retired) triggered-plan subsystem.
#
# All linkage columns are nullable on purpose. The live PM path runs with
# PM_CANDIDATE_MODE disabled and therefore produces no candidate_id.
#
# Requirements: 2.8, 2.12, 7.4
# ---------------------------------------------------------------------------

_PENDING_ORDERS_DDL = """
CREATE TABLE IF NOT EXISTS pending_orders (
    order_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    geometry_name TEXT,

    candidate_id TEXT,
    cycle_id TEXT,
    source_signal_id TEXT,
    plan_id TEXT,

    limit_price REAL NOT NULL,
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    risk_reward REAL NOT NULL,
    intended_quantity INTEGER,

    fresh_price_at_creation REAL NOT NULL,
    runaway_pct_at_creation REAL NOT NULL,
    pm_rationale TEXT,
    signal_snapshot_json TEXT,

    state TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    last_evaluated_bar_ts TEXT,
    filled_at TEXT,
    terminal_at TEXT,
    fill_price REAL,
    fill_policy TEXT,
    fill_bar_ts TEXT,
    terminal_reason TEXT,
    trade_id INTEGER,

    integrity_hash TEXT NOT NULL
)
"""

_PENDING_ORDERS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pending_orders_state "
    "ON pending_orders(state)",
    "CREATE INDEX IF NOT EXISTS idx_pending_orders_symbol_state "
    "ON pending_orders(symbol, state)",
    "CREATE INDEX IF NOT EXISTS idx_pending_orders_profile_state "
    "ON pending_orders(profile_id, state)",
    "CREATE INDEX IF NOT EXISTS idx_pending_orders_candidate "
    "ON pending_orders(candidate_id)",
    # Enforces Requirement 7.4 at the storage layer rather than trusting
    # application logic: at most one ACTIVE order per
    # (profile_id, symbol, side, setup_type). Terminal rows are excluded, so
    # history accumulates freely. Creation supersedes any existing active
    # order for the key before inserting, so this only fires on a genuine
    # race — where an IntegrityError is the correct, fail-closed outcome.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_orders_active_key "
    "ON pending_orders(profile_id, symbol, side, setup_type) "
    "WHERE state IN ('pending', 'filling')",
]


# ---------------------------------------------------------------------------
# pending_order_events (immutable, append-only audit trail)
#
# Every state transition emits a row here. Rows are never updated or deleted —
# immutability is enforced by database triggers, matching trade_plan_events,
# decision_snapshots and provenance_events.
#
# Requirements: 2.9, 2.11, 9.10, 10.8
# ---------------------------------------------------------------------------

_PENDING_ORDER_EVENTS_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS pending_order_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT,
    from_state TEXT,
    to_state TEXT,
    reference_price REAL,
    created_at TEXT NOT NULL
)
"""

_PENDING_ORDER_EVENTS_DDL_POSTGRES = """
CREATE TABLE IF NOT EXISTS pending_order_events (
    id SERIAL PRIMARY KEY,
    order_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT,
    from_state TEXT,
    to_state TEXT,
    reference_price DOUBLE PRECISION,
    created_at TEXT NOT NULL
)
"""

_PENDING_ORDER_EVENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pending_order_events_order "
    "ON pending_order_events(order_id)",
    "CREATE INDEX IF NOT EXISTS idx_pending_order_events_type "
    "ON pending_order_events(event_type)",
]

_PENDING_ORDER_EVENTS_IMMUTABILITY_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS pending_order_events_no_update
        BEFORE UPDATE ON pending_order_events
    BEGIN
        SELECT RAISE(ABORT, 'pending_order_events is immutable: UPDATE prohibited');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS pending_order_events_no_delete
        BEFORE DELETE ON pending_order_events
    BEGIN
        SELECT RAISE(ABORT, 'pending_order_events is immutable: DELETE prohibited');
    END
    """,
]


def init_pending_order_schema(engine):
    """Create pending-limit-order tables, indexes, and triggers if missing.

    Non-destructive and idempotent: uses CREATE TABLE / CREATE INDEX /
    CREATE TRIGGER IF NOT EXISTS so it can run on every orchestrator startup.
    Existing rows and columns are never modified.

    `pending_order_events` is an append-only audit trail: UPDATE and DELETE are
    blocked by immutability triggers, matching the pattern used by
    trade_plan_events, decision_snapshots and provenance_events.

    A partial UNIQUE index enforces at most one active order per
    (profile_id, symbol, side, setup_type), which is Requirement 7.4 expressed
    as a storage constraint instead of an application-layer convention.

    Requirements: 2.8, 2.9, 2.10, 2.11, 2.12, 7.4
    """
    sqlite = is_sqlite(engine)

    with engine.begin() as conn:
        conn.execute(text(_PENDING_ORDERS_DDL))
        for stmt in _PENDING_ORDERS_INDEXES:
            conn.execute(text(stmt))

        conn.execute(text(
            _PENDING_ORDER_EVENTS_DDL_SQLITE if sqlite
            else _PENDING_ORDER_EVENTS_DDL_POSTGRES
        ))
        for stmt in _PENDING_ORDER_EVENTS_INDEXES:
            conn.execute(text(stmt))

        if sqlite:
            for stmt in _PENDING_ORDER_EVENTS_IMMUTABILITY_TRIGGERS:
                conn.execute(text(stmt))
        else:
            # Postgres: shared raise_immutable() function + per-op triggers.
            # CREATE OR REPLACE makes this safe alongside init_trade_plan_schema,
            # which defines the same function.
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION raise_immutable() RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION '% is immutable: % prohibited', TG_TABLE_NAME, TG_OP;
                END;
                $$ LANGUAGE plpgsql
            """))
            for op in ("update", "delete"):
                conn.execute(text(
                    f"DROP TRIGGER IF EXISTS pending_order_events_no_{op} "
                    f"ON pending_order_events"
                ))
                conn.execute(text(
                    f"CREATE TRIGGER pending_order_events_no_{op} "
                    f"BEFORE {op.upper()} ON pending_order_events "
                    f"FOR EACH ROW EXECUTE FUNCTION raise_immutable()"
                ))

    logger.debug(
        "Pending order schema verified (pending_orders, pending_order_events)"
    )


# ---------------------------------------------------------------------------
# Setup Watch Layer — Tables, Indexes, Triggers
#
# The setup watch layer adds a structured intermediate stage between upstream
# discovery (analyst, scout, market-state) and PM-approved trade execution.
# Watches track thesis maturation through condition evaluation, independent of
# the existing watch_candidates table.
#
# Three tables:
#   - setup_watches: lifecycle state machine with CAS transitions
#   - setup_watch_events: immutable audit trail of all state changes
#   - setup_watch_outcomes: counterfactual scoring at 15/30/60-minute windows
#
# Requirements: 1.1–1.12
# ---------------------------------------------------------------------------

_SETUP_WATCHES_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS setup_watches (
    watch_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'watching',

    thesis TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT,
    source_cycle_id TEXT NOT NULL,

    maturation_conditions_json TEXT NOT NULL,
    invalidation_conditions_json TEXT NOT NULL,

    last_evaluation_json TEXT,

    entry_zone_json TEXT,
    draft_geometry_json TEXT,

    maturity_score REAL DEFAULT 0.0,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    state_changed_at TEXT,

    observed_cycles INTEGER DEFAULT 0,

    ready_at TEXT,
    ready_reference_price REAL,

    terminal_reason TEXT,
    promoted_cycle_id TEXT,

    execution_ref_type TEXT,
    execution_ref_id TEXT,

    integrity_hash TEXT NOT NULL
)
"""

_SETUP_WATCHES_DDL_POSTGRES = """
CREATE TABLE IF NOT EXISTS setup_watches (
    watch_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'watching',

    thesis TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT,
    source_cycle_id TEXT NOT NULL,

    maturation_conditions_json TEXT NOT NULL,
    invalidation_conditions_json TEXT NOT NULL,

    last_evaluation_json TEXT,

    entry_zone_json TEXT,
    draft_geometry_json TEXT,

    maturity_score DOUBLE PRECISION DEFAULT 0.0,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    state_changed_at TEXT,

    observed_cycles INTEGER DEFAULT 0,

    ready_at TEXT,
    ready_reference_price DOUBLE PRECISION,

    terminal_reason TEXT,
    promoted_cycle_id TEXT,

    execution_ref_type TEXT,
    execution_ref_id TEXT,

    integrity_hash TEXT NOT NULL
)
"""

_SETUP_WATCHES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_setup_watches_profile_state "
    "ON setup_watches(profile_id, state)",

    "CREATE INDEX IF NOT EXISTS idx_setup_watches_symbol_state "
    "ON setup_watches(symbol, state)",

    "CREATE INDEX IF NOT EXISTS idx_setup_watches_state_expires "
    "ON setup_watches(state, expires_at)",

    # At most one active watch per (profile, symbol, side, setup_type).
    # Terminal rows excluded so history accumulates freely.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_setup_watches_active_key "
    "ON setup_watches(profile_id, symbol, side, setup_type) "
    "WHERE state NOT IN ('expired', 'rejected', 'ordered')",
]


# ---------------------------------------------------------------------------
# setup_watch_events (immutable, append-only audit trail)
#
# Every state transition emits a row here. Rows are never updated or deleted —
# immutability is enforced by database triggers.
#
# Requirements: 1.4, 1.5
# ---------------------------------------------------------------------------

_SETUP_WATCH_EVENTS_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS setup_watch_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT,
    from_state TEXT,
    to_state TEXT,
    maturity_score REAL,
    created_at TEXT NOT NULL
)
"""

_SETUP_WATCH_EVENTS_DDL_POSTGRES = """
CREATE TABLE IF NOT EXISTS setup_watch_events (
    id SERIAL PRIMARY KEY,
    watch_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT,
    from_state TEXT,
    to_state TEXT,
    maturity_score DOUBLE PRECISION,
    created_at TEXT NOT NULL
)
"""

_SETUP_WATCH_EVENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_setup_watch_events_watch "
    "ON setup_watch_events(watch_id)",

    "CREATE INDEX IF NOT EXISTS idx_setup_watch_events_type "
    "ON setup_watch_events(event_type)",
]

_SETUP_WATCH_EVENTS_IMMUTABILITY_TRIGGERS_SQLITE = [
    """
    CREATE TRIGGER IF NOT EXISTS setup_watch_events_no_update
        BEFORE UPDATE ON setup_watch_events
    BEGIN
        SELECT RAISE(ABORT, 'setup_watch_events is immutable: UPDATE prohibited');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS setup_watch_events_no_delete
        BEFORE DELETE ON setup_watch_events
    BEGIN
        SELECT RAISE(ABORT, 'setup_watch_events is immutable: DELETE prohibited');
    END
    """,
]


# ---------------------------------------------------------------------------
# setup_watch_outcomes (immutable, counterfactual scoring)
#
# One row per (watch_id, window_label). Measures what price did after a watch
# reached `ready`, independent of whether the watch was promoted or traded.
#
# Requirements: 1.7, 1.8
# ---------------------------------------------------------------------------

_SETUP_WATCH_OUTCOMES_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS setup_watch_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    window_label TEXT NOT NULL,
    window_minutes INTEGER NOT NULL,
    reference_price REAL NOT NULL,
    evaluated_at TEXT NOT NULL,
    mfe_pct REAL,
    mae_pct REAL,
    entry_zone_touched INTEGER,
    would_have_hit_target INTEGER,
    would_have_hit_stop INTEGER,
    scorable INTEGER NOT NULL,
    unscorable_reason TEXT,
    created_at TEXT NOT NULL
)
"""

_SETUP_WATCH_OUTCOMES_DDL_POSTGRES = """
CREATE TABLE IF NOT EXISTS setup_watch_outcomes (
    id SERIAL PRIMARY KEY,
    watch_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    window_label TEXT NOT NULL,
    window_minutes INTEGER NOT NULL,
    reference_price DOUBLE PRECISION NOT NULL,
    evaluated_at TEXT NOT NULL,
    mfe_pct DOUBLE PRECISION,
    mae_pct DOUBLE PRECISION,
    entry_zone_touched INTEGER,
    would_have_hit_target INTEGER,
    would_have_hit_stop INTEGER,
    scorable INTEGER NOT NULL,
    unscorable_reason TEXT,
    created_at TEXT NOT NULL
)
"""

_SETUP_WATCH_OUTCOMES_INDEXES = [
    # Idempotency: each watch scored at most once per window
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_setup_watch_outcomes_watch_window "
    "ON setup_watch_outcomes(watch_id, window_label)",

    "CREATE INDEX IF NOT EXISTS idx_setup_watch_outcomes_profile_window "
    "ON setup_watch_outcomes(profile_id, window_label)",
]

_SETUP_WATCH_OUTCOMES_IMMUTABILITY_TRIGGERS_SQLITE = [
    """
    CREATE TRIGGER IF NOT EXISTS setup_watch_outcomes_no_update
        BEFORE UPDATE ON setup_watch_outcomes
    BEGIN
        SELECT RAISE(ABORT, 'setup_watch_outcomes is immutable: UPDATE prohibited');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS setup_watch_outcomes_no_delete
        BEFORE DELETE ON setup_watch_outcomes
    BEGIN
        SELECT RAISE(ABORT, 'setup_watch_outcomes is immutable: DELETE prohibited');
    END
    """,
]


def init_setup_watch_schema(engine):
    """Create setup watch tables, indexes, and triggers if missing.

    Non-destructive and idempotent. Uses CREATE TABLE / INDEX / TRIGGER
    IF NOT EXISTS. Existing rows and columns are never modified.

    Three tables:
      - setup_watches: lifecycle state machine
      - setup_watch_events: immutable audit trail
      - setup_watch_outcomes: counterfactual scoring

    Requirements: 1.1–1.12
    """
    sqlite = is_sqlite(engine)

    with engine.begin() as conn:
        # --- setup_watches table ---
        conn.execute(text(
            _SETUP_WATCHES_DDL_SQLITE if sqlite else _SETUP_WATCHES_DDL_POSTGRES
        ))
        for stmt in _SETUP_WATCHES_INDEXES:
            conn.execute(text(stmt))

        # --- setup_watch_events table ---
        conn.execute(text(
            _SETUP_WATCH_EVENTS_DDL_SQLITE if sqlite
            else _SETUP_WATCH_EVENTS_DDL_POSTGRES
        ))
        for stmt in _SETUP_WATCH_EVENTS_INDEXES:
            conn.execute(text(stmt))

        # --- setup_watch_outcomes table ---
        conn.execute(text(
            _SETUP_WATCH_OUTCOMES_DDL_SQLITE if sqlite
            else _SETUP_WATCH_OUTCOMES_DDL_POSTGRES
        ))
        for stmt in _SETUP_WATCH_OUTCOMES_INDEXES:
            conn.execute(text(stmt))

        # --- Immutability triggers ---
        if sqlite:
            for stmt in _SETUP_WATCH_EVENTS_IMMUTABILITY_TRIGGERS_SQLITE:
                conn.execute(text(stmt))
            for stmt in _SETUP_WATCH_OUTCOMES_IMMUTABILITY_TRIGGERS_SQLITE:
                conn.execute(text(stmt))
        else:
            # Postgres: shared raise_immutable() function + per-op triggers.
            # CREATE OR REPLACE makes this safe alongside init_trade_plan_schema
            # and init_pending_order_schema, which define the same function.
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION raise_immutable() RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION '% is immutable: % prohibited', TG_TABLE_NAME, TG_OP;
                END;
                $$ LANGUAGE plpgsql
            """))
            for table in ("setup_watch_events", "setup_watch_outcomes"):
                for op in ("update", "delete"):
                    conn.execute(text(
                        f"DROP TRIGGER IF EXISTS {table}_no_{op} ON {table}"
                    ))
                    conn.execute(text(
                        f"CREATE TRIGGER {table}_no_{op} "
                        f"BEFORE {op.upper()} ON {table} "
                        f"FOR EACH ROW EXECUTE FUNCTION raise_immutable()"
                    ))

    logger.debug(
        "Setup watch schema verified "
        "(setup_watches, setup_watch_events, setup_watch_outcomes)"
    )
