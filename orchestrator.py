"""
Orchestrator
Runs the full agent pipeline on a schedule aligned with market hours.
Uses APScheduler for intraday loops.
"""

import os
import json
import signal
import logging
import logging.handlers
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from rich.console import Console

load_dotenv()

from db.schema import init_db, Balance
from sqlalchemy.orm import sessionmaker
from utils.shadow_ledger import ensure_shadow_ledger_schema
from utils.shadow_outcomes import update_blocked_candidate_outcomes
import agents.researcher as researcher
import agents.analyst as analyst
import agents.portfolio_manager as pm
import agents.bookkeeper as bookkeeper
import agents.reviewer as reviewer
import agents.scout as scout
from agents.scout import run_intraday_scan
import agents.weekly_prep as weekly_prep
import agents.quant_researcher as quant_researcher
import agents.daily_review as daily_review
import agents.ceo as ceo
from utils.expanded_watchlist import get_expanded_watchlist
from utils.funnel_transition import get_funnel_or_fallback_candidates, build_deduplicated_watchlist
from utils.focus_list import select_focus_symbols
from utils.sector_scout_outcomes import (
    record_analyst_outcome,
    record_pm_outcome,
    record_trade_outcome,
)
from utils.position_lifecycle_governance import is_trading_day
from utils.funnel_config import load_funnel_config
from utils.funnel_discovery import run_funnel_discovery
from utils.funnel_researcher import run_funnel_qualification
from utils.funnel_analyst import run_funnel_analysis
from utils.funnel_confirmation import run_opening_confirmation, run_confirmation_retry
from db.provenance_schema import init_provenance_schema
from db.blocker_mitigation_schema import init_blocker_mitigation_schema

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            "logs/orchestrator.log", maxBytes=10_000_000, backupCount=5
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

def _parse_symbol_list(raw: str) -> list[str]:
    """Parse a comma-separated symbol list, preserving order and de-duping."""
    seen = set()
    symbols = []
    for item in raw.split(","):
        sym = item.strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            symbols.append(sym)
    return symbols


WATCHLIST = _parse_symbol_list(os.getenv("WATCHLIST", "SPY,QQQ,IWM,DIA,TLT,GLD,XLK,XLF,XLE,TSLA,NVDA,AMD"))
SCOUT_CANDIDATES = _parse_symbol_list(os.getenv("SCOUT_CANDIDATES", "AAPL,MSFT,META,AMZN,GOOGL,AVGO,SMCI,PLTR,COIN,MSTR,ARM,MU,INTC,NFLX"))
PM_TRADABLE_SYMBOLS = _parse_symbol_list(os.getenv("PM_TRADABLE_SYMBOLS", "META,MU"))
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL_MINUTES", 15))


def _pm_base_watchlist() -> list[str]:
    """Core symbols plus configured non-core names that PM may trade."""
    return _parse_symbol_list(",".join(WATCHLIST + PM_TRADABLE_SYMBOLS))


def _env_flag_enabled(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("true", "1", "yes", "enabled")


def _analyst_focus_enabled() -> bool:
    return _env_flag_enabled("ANALYST_FOCUS_MODE", "disabled")


def _analyst_focus_max_symbols() -> int:
    try:
        return max(1, int(os.getenv("ANALYST_FOCUS_MAX_SYMBOLS", "3")))
    except ValueError:
        log.warning("Invalid ANALYST_FOCUS_MAX_SYMBOLS=%r; using 3", os.getenv("ANALYST_FOCUS_MAX_SYMBOLS"))
        return 3


def _source_bonuses(
    base_symbols: list[str],
    scout_picks: list[str],
    expanded_symbols: list[str],
    alert_symbols: list[str] | None = None,
) -> dict[str, float]:
    bonuses: dict[str, float] = {}
    for sym in base_symbols:
        bonuses[sym] = max(bonuses.get(sym, 0.0), 0.5)
    for sym in scout_picks:
        bonuses[sym] = max(bonuses.get(sym, 0.0), 2.0)
    for sym in expanded_symbols:
        bonuses[sym] = max(bonuses.get(sym, 0.0), 3.0)
    for sym in alert_symbols or []:
        bonuses[sym] = max(bonuses.get(sym, 0.0), 4.0)
    return bonuses


def _open_position_symbols(engine) -> list[str]:
    from db.schema import Position, get_session
    db = get_session(engine)
    try:
        return _parse_symbol_list(",".join(p.symbol for p in db.query(Position).all()))
    finally:
        db.close()


_focus_list_cache = {}
_focus_list_lock = threading.Lock()


def _apply_focus_list(
    engine,
    candidate_symbols: list[str],
    *,
    scout_picks: list[str],
    expanded_symbols: list[str],
    context: str,
    required_symbols: list[str] | None = None,
    alert_symbols: list[str] | None = None,
) -> list[str]:
    if not _analyst_focus_enabled():
        return candidate_symbols

    max_symbols = _analyst_focus_max_symbols()
    required_symbols = required_symbols or []
    alert_symbols = alert_symbols or []
    focus_bucket = datetime.utcnow().replace(second=0, microsecond=0)
    cache_key = (
        focus_bucket.isoformat(),
        max_symbols,
        tuple(candidate_symbols),
        tuple(required_symbols),
        tuple(alert_symbols),
    )

    with _focus_list_lock:
        focused = _focus_list_cache.get(cache_key)
        if focused is None:
            focused = select_focus_symbols(
                engine,
                candidate_symbols,
                max_symbols=max_symbols,
                required_symbols=required_symbols,
                source_bonuses=_source_bonuses(_pm_base_watchlist(), scout_picks, expanded_symbols, alert_symbols),
                context=context,
            )
            _focus_list_cache[cache_key] = list(focused)
            if len(_focus_list_cache) > 12:
                _focus_list_cache.clear()
                _focus_list_cache[cache_key] = list(focused)
        else:
            log.info(
                "FOCUS_LIST_CACHE_HIT: context=%s selected=%s candidates=%s",
                context,
                focused,
                len(candidate_symbols),
            )

    log.info(
        "FOCUS_LIST_APPLIED: context=%s before=%d after=%d symbols=%s",
        context,
        len(candidate_symbols),
        len(focused),
        focused,
    )
    return focused


_engine = None
_market_closed_skips_logged = set()
_regular_market_skips_logged = set()

# Funnel job blocking state: when PM cycle is active, new funnel jobs must wait.
# An in-progress premarket job may complete within its budget, but no NEW funnel
# jobs should start until the PM cycle finishes. (Requirement 7.5)
_pm_cycle_active = False
_pm_cycle_owner = None
_pm_cycle_lock = threading.Lock()

# Alert dispatcher instance (constructed in main() when PM_ALERT_DISPATCH_MODE != "disabled")
_alert_dispatcher = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = init_db("db/paper_trader.db")
    return _engine


def _skip_closed_market_job(job_name: str, now_et=None) -> bool:
    """Return True when a market-day job should not run on a closed session."""
    if now_et is None:
        from pytz import timezone
        now_et = datetime.now(timezone("America/New_York"))

    if is_trading_day(now_et):
        return False

    key = (job_name, now_et.date())
    if key not in _market_closed_skips_logged:
        log.info(
            "MARKET_CLOSED_SKIP: job=%s date=%s reason=holiday_or_weekend",
            job_name,
            now_et.date().isoformat(),
        )
        _market_closed_skips_logged.add(key)
    return True


def _skip_outside_regular_market_job(job_name: str, now_et=None) -> bool:
    """Return True when a regular-session job should not run outside 9:30-16:00 ET."""
    if now_et is None:
        from pytz import timezone
        now_et = datetime.now(timezone("America/New_York"))

    if _skip_closed_market_job(job_name, now_et):
        return True

    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    if market_open <= now_et <= market_close:
        return False

    if now_et < market_open:
        reason = "before_regular_open"
    else:
        reason = "after_regular_close"

    key = (job_name, now_et.date(), reason)
    if key not in _regular_market_skips_logged:
        log.info(
            "REGULAR_MARKET_SKIP: job=%s time=%s reason=%s",
            job_name,
            now_et.isoformat(),
            reason,
        )
        _regular_market_skips_logged.add(key)
    return True


# ---------------------------------------------------------------------------
# Funnel Missed-Job Detection and Error Handling (Task 10.2)
# Requirements: 12.7, 12.8, 7.5
# ---------------------------------------------------------------------------


def _log_missed_funnel_job(engine, stage: str, scheduled_time_str: str, budget_seconds: float) -> None:
    """Create a FunnelRunLog entry for a missed funnel job.

    When the orchestrator starts after a scheduled funnel job time, it skips
    that job and logs an error entry so downstream logic (fallback, observability)
    treats the day correctly.

    Args:
        engine: SQLAlchemy engine.
        stage: Pipeline stage name (discovery, research, analysis, confirmation).
        scheduled_time_str: Human-readable scheduled time (e.g. "06:00 ET").
        budget_seconds: Budget that was configured for this stage.
    """
    from db.schema import FunnelRunLog, get_session
    from zoneinfo import ZoneInfo

    try:
        session = get_session(engine)
        now_utc = datetime.utcnow()
        today_ny = datetime.now(ZoneInfo("America/New_York")).date()

        log_entry = FunnelRunLog(
            date=today_ny,
            stage=stage,
            started_at=now_utc,
            ended_at=now_utc,
            duration_seconds=0.0,
            budget_seconds=budget_seconds,
            result_status="error",
            candidates_input=0,
            candidates_promoted=0,
            candidates_rejected=0,
            error_message=f"missed scheduled start at {scheduled_time_str}",
        )
        session.add(log_entry)
        session.commit()
        session.close()
    except Exception as e:
        log.error(
            "Failed to write missed-job FunnelRunLog for stage=%s: %s",
            stage, e
        )


def _log_funnel_job_error(engine, stage: str, budget_seconds: float, error: Exception) -> None:
    """Create a FunnelRunLog entry for an unhandled funnel job exception.

    Catches and records the exception so other jobs can continue.

    Args:
        engine: SQLAlchemy engine.
        stage: Pipeline stage name (discovery, research, analysis, confirmation).
        budget_seconds: Budget that was configured for this stage.
        error: The exception that caused the failure.
    """
    from db.schema import FunnelRunLog, get_session
    from zoneinfo import ZoneInfo

    try:
        session = get_session(engine)
        now_utc = datetime.utcnow()
        today_ny = datetime.now(ZoneInfo("America/New_York")).date()

        log_entry = FunnelRunLog(
            date=today_ny,
            stage=stage,
            started_at=now_utc,
            ended_at=now_utc,
            duration_seconds=0.0,
            budget_seconds=budget_seconds,
            result_status="error",
            candidates_input=0,
            candidates_promoted=0,
            candidates_rejected=0,
            error_message=f"{type(error).__name__}: {str(error)[:500]}",
        )
        session.add(log_entry)
        session.commit()
        session.close()
    except Exception as log_err:
        log.error(
            "Failed to write FunnelRunLog error entry for stage=%s: %s",
            stage, log_err
        )


def _check_missed_funnel_jobs(engine, funnel_config: dict) -> None:
    """Detect and log missed funnel jobs when the orchestrator starts late.

    Called at orchestrator startup. For each funnel job whose scheduled time
    has already passed today, creates a FunnelRunLog with result_status="error"
    and error_message indicating the missed time.

    If the orchestrator starts before the first PM cycle (09:30 ET), missed
    premarket jobs are logged. If it starts after confirmation time, all
    premarket jobs that weren't run are logged.

    Only logs a missed job if no FunnelRunLog entry already exists for that
    stage + date (to prevent duplicates on restart).

    Args:
        engine: SQLAlchemy engine.
        funnel_config: Funnel pipeline configuration dict.

    Requirements: 12.7
    """
    from zoneinfo import ZoneInfo
    from db.schema import FunnelRunLog, get_session

    now_ny = datetime.now(ZoneInfo("America/New_York"))
    today_ny = now_ny.date()

    # Only check on trading days
    if not is_trading_day(now_ny):
        return

    funnel = funnel_config.get("funnel", funnel_config)
    schedule = funnel.get("schedule", {})
    budgets = funnel.get("budgets", {})

    # Define funnel job schedule: (stage, time_str, budget_key, default_budget)
    funnel_jobs = [
        ("discovery", schedule.get("discovery_time", "06:00"), "total_pipeline_seconds", 90),
        ("research", schedule.get("research_time", "06:30"), "total_pipeline_seconds", 90),
        ("analysis", schedule.get("analysis_time", "07:15"), "total_pipeline_seconds", 90),
        ("confirmation", schedule.get("confirmation_time", "09:35"), "confirmation_budget_seconds", 45),
    ]

    # Check which jobs already have log entries for today
    session = get_session(engine)
    try:
        existing_logs = (
            session.query(FunnelRunLog.stage)
            .filter(FunnelRunLog.date == today_ny)
            .all()
        )
        logged_stages = {row.stage for row in existing_logs}
    finally:
        session.close()

    for stage, time_str, budget_key, default_budget in funnel_jobs:
        # Parse scheduled time
        try:
            parts = time_str.split(":")
            sched_hour = int(parts[0])
            sched_minute = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            continue

        sched_time = now_ny.replace(
            hour=sched_hour, minute=sched_minute, second=0, microsecond=0
        )

        # If we started after this job's scheduled time and no log exists for it
        if now_ny > sched_time and stage not in logged_stages:
            budget = float(budgets.get(budget_key, default_budget))
            scheduled_label = f"{time_str} ET"
            log.warning(
                "MISSED_FUNNEL_JOB: stage=%s scheduled=%s current=%s — "
                "orchestrator started after scheduled time, skipping job",
                stage,
                scheduled_label,
                now_ny.strftime("%H:%M ET"),
            )
            _log_missed_funnel_job(engine, stage, scheduled_label, budget)


def _is_pm_cycle_blocking_funnel() -> bool:
    """Return True if a PM cycle is active and new funnel jobs should be blocked.

    Per requirement 7.5: in-progress premarket jobs complete within their
    budget, but new funnel jobs must not start while a PM cycle is running.
    """
    return _pm_cycle_active


def _set_pm_cycle_active(active: bool) -> None:
    """Set the PM cycle blocking state."""
    global _pm_cycle_active
    _pm_cycle_active = active


def _try_begin_pm_cycle(owner: str) -> bool:
    """Try to begin a new-entry PM cycle without overlapping another one."""
    global _pm_cycle_active, _pm_cycle_owner
    if not _pm_cycle_lock.acquire(blocking=False):
        log.info(
            "PM_CYCLE_SKIP: owner=%s reason=pm_cycle_already_active active_owner=%s",
            owner,
            _pm_cycle_owner,
        )
        return False

    _pm_cycle_active = True
    _pm_cycle_owner = owner
    return True


def _end_pm_cycle(owner: str) -> None:
    """End a PM cycle started by _try_begin_pm_cycle()."""
    global _pm_cycle_active, _pm_cycle_owner
    if _pm_cycle_owner not in (None, owner):
        log.warning(
            "PM_CYCLE_OWNER_MISMATCH: ending_owner=%s active_owner=%s",
            owner,
            _pm_cycle_owner,
        )
    _pm_cycle_active = False
    _pm_cycle_owner = None
    if _pm_cycle_lock.locked():
        _pm_cycle_lock.release()


def _candidate_mode_requires_serial_pm_profiles() -> bool:
    """Return True when PM profiles should not run concurrently."""
    try:
        from utils.gate_config import PM_CANDIDATE_MODE
        return PM_CANDIDATE_MODE in ("enabled", "shadow")
    except Exception:
        return False


def _run_pm_profile_jobs(profile_ids, run_one_profile):
    """Run profile jobs, serializing them when candidate mode writes registries."""
    profile_ids = list(profile_ids)
    if _candidate_mode_requires_serial_pm_profiles():
        log.info(
            "PM_PROFILE_EXECUTION: mode=serial reason=candidate_mode profiles=%s",
            profile_ids,
        )
        for profile_id in profile_ids:
            run_one_profile(profile_id)
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=len(profile_ids) or 1) as executor:
        futures = [executor.submit(run_one_profile, pid) for pid in profile_ids]
        for f in as_completed(futures):
            pass


def ensure_initial_balance(engine):
    """Seed starting cash balance for all profiles if DB is fresh."""
    from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES
    Session = sessionmaker(bind=engine)
    db = Session()
    for profile_id in ACTIVE_PROFILES:
        count = db.query(Balance).filter_by(profile=profile_id).count()
        if count == 0:
            starting = float(PM_PROFILES[profile_id]["starting_balance"])
            db.add(Balance(cash=starting, profile=profile_id))
            log.info(f"Seeded {profile_id} balance: ${starting:,.2f}")
    db.commit()
    db.close()


def _ensure_funnel_tables(engine, inspector):
    """Create FunnelCandidate and FunnelRunLog tables if they don't exist.

    Called during check_schema() startup. Non-destructive — only creates
    tables that are missing. WAL mode is already applied at engine level
    via the _set_sqlite_pragma event listener in init_db().

    Requirements: 3.6, 12.1
    """
    from sqlalchemy import text

    if not inspector.has_table("funnel_candidates"):
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS funnel_candidates (
                        id INTEGER PRIMARY KEY,
                        candidate_id VARCHAR(36) NOT NULL,
                        date DATE NOT NULL,
                        symbol VARCHAR(10) NOT NULL,
                        discovered_at DATETIME NOT NULL,
                        source_run VARCHAR(32) NOT NULL,
                        selection_mode VARCHAR(32) NOT NULL,
                        scout_rank INTEGER NOT NULL,
                        scout_score REAL NOT NULL,
                        direction_bias VARCHAR(10),
                        catalyst_evidence TEXT NOT NULL,
                        selection_reason TEXT NOT NULL,
                        primary_risk TEXT NOT NULL,
                        sector_context TEXT,
                        preliminary_setup_type VARCHAR(32),
                        authoritative_setup_type VARCHAR(32),
                        stage_status VARCHAR(32) NOT NULL DEFAULT 'awaiting_research',
                        stage_decisions TEXT NOT NULL DEFAULT '[]',
                        trade_event_id INTEGER REFERENCES trade_events(id),
                        blocked_candidate_id INTEGER,
                        expired BOOLEAN DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_funnel_date_status "
                    "ON funnel_candidates (date, stage_status)"
                )
            )
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_funnel_date_symbol "
                    "ON funnel_candidates (date, symbol)"
                )
            )
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_funnel_candidate_id "
                    "ON funnel_candidates (candidate_id)"
                )
            )
            conn.commit()
        log.warning("Schema migration: created funnel_candidates table with indexes")

    if not inspector.has_table("funnel_run_logs"):
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS funnel_run_logs (
                        id INTEGER PRIMARY KEY,
                        date DATE NOT NULL,
                        stage VARCHAR(32) NOT NULL,
                        started_at DATETIME NOT NULL,
                        ended_at DATETIME,
                        duration_seconds REAL,
                        budget_seconds REAL NOT NULL,
                        result_status VARCHAR(32) NOT NULL,
                        sectors_completed TEXT,
                        sectors_timed_out TEXT,
                        candidates_input INTEGER,
                        candidates_promoted INTEGER,
                        candidates_rejected INTEGER,
                        error_message TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            )
            conn.commit()
        log.warning("Schema migration: created funnel_run_logs table")


def _ensure_candidate_tables(engine, inspector):
    """Create pm_candidates, pm_candidate_events, and candidate_shadow_comparison tables if missing.

    Called during check_schema() startup. Non-destructive — only creates
    tables that are missing. WAL mode is already applied at engine level
    via the _set_sqlite_pragma event listener in init_db().

    Requirements: 1.3, 12.1, 12.2, 12.5, 13.2
    """
    from sqlalchemy import text
    from utils.dialect_sql import _pk_column, _default_timestamp

    pk = _pk_column(engine)
    ts_default = _default_timestamp(engine)

    if not inspector.has_table("pm_candidates"):
        with engine.connect() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS pm_candidates (
                        {pk},
                        candidate_id VARCHAR(36) NOT NULL UNIQUE,
                        cycle_id VARCHAR(64) NOT NULL,
                        profile_id VARCHAR(64) NOT NULL,
                        symbol VARCHAR(10) NOT NULL,
                        direction VARCHAR(10) NOT NULL,
                        setup_type VARCHAR(64) NOT NULL,
                        geometry_name VARCHAR(64) NOT NULL,
                        entry_price REAL NOT NULL,
                        stop_price REAL NOT NULL,
                        target_price REAL NOT NULL,
                        risk_reward REAL NOT NULL,
                        trigger TEXT,
                        invalidation_basis TEXT,
                        target_basis TEXT,
                        source_signal_id VARCHAR(64) NOT NULL,
                        signal_snapshot_json TEXT NOT NULL,
                        state VARCHAR(32) NOT NULL DEFAULT 'registered',
                        integrity_hash VARCHAR(64) NOT NULL,
                        execution_key VARCHAR(128),
                        reserved_at DATETIME,
                        created_at DATETIME {ts_default},
                        expires_at DATETIME NOT NULL,
                        context_snapshot_json TEXT,
                        benchmark_mapping_json TEXT,
                        rejection_reason TEXT,
                        candidate_type TEXT DEFAULT 'intraday',
                        holding_horizon INTEGER,
                        normalized_setup_type TEXT
                    )
                    """
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_pm_candidates_cycle_profile "
                    "ON pm_candidates (cycle_id, profile_id)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_pm_candidates_state "
                    "ON pm_candidates (state)"
                )
            )
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_pm_candidates_execution_key "
                    "ON pm_candidates (execution_key) WHERE execution_key IS NOT NULL"
                )
            )
            conn.commit()
        log.warning("Schema migration: created pm_candidates table with indexes")

    if not inspector.has_table("pm_candidate_events"):
        with engine.connect() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS pm_candidate_events (
                        {pk},
                        candidate_id VARCHAR(36) NOT NULL,
                        cycle_id VARCHAR(64) NOT NULL,
                        profile_id VARCHAR(64) NOT NULL,
                        event_type VARCHAR(64) NOT NULL,
                        event_data TEXT,
                        created_at DATETIME {ts_default},
                        candidate_type TEXT DEFAULT 'intraday'
                    )
                    """
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_pm_candidate_events_candidate "
                    "ON pm_candidate_events (candidate_id)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_pm_candidate_events_cycle "
                    "ON pm_candidate_events (cycle_id, profile_id)"
                )
            )
            conn.commit()
        log.warning("Schema migration: created pm_candidate_events table with indexes")

    if not inspector.has_table("candidate_shadow_comparison"):
        with engine.connect() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS candidate_shadow_comparison (
                        {pk},
                        cycle_id VARCHAR(64) NOT NULL,
                        profile_id VARCHAR(64) NOT NULL,
                        candidate_results_json TEXT NOT NULL,
                        legacy_results_json TEXT NOT NULL,
                        agreement_summary TEXT,
                        malformed_count INTEGER DEFAULT 0,
                        hypothetical_diffs TEXT,
                        created_at DATETIME {ts_default}
                    )
                    """
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_shadow_comparison_cycle "
                    "ON candidate_shadow_comparison (cycle_id, profile_id)"
                )
            )
            conn.commit()
        log.warning("Schema migration: created candidate_shadow_comparison table with indexes")


def _ensure_postgres_identity_default(engine, inspector, table_name: str) -> None:
    """Repair migrated Postgres integer primary keys missing a sequence default.

    SQLite's ``INTEGER PRIMARY KEY`` auto-generates values. During the
    SQLite-to-Postgres migration, the same DDL can leave an integer primary key
    with no sequence/default, causing inserts that omit ``id`` to fail.
    """
    from sqlalchemy import text
    from db.schema import is_sqlite

    if is_sqlite(engine) or not inspector.has_table(table_name):
        return

    columns = {col["name"]: col for col in inspector.get_columns(table_name)}
    id_col = columns.get("id")
    if not id_col or id_col.get("default"):
        return

    sequence_name = f"{table_name}_id_seq"
    with engine.begin() as conn:
        conn.execute(
            text(
                f"CREATE SEQUENCE IF NOT EXISTS {sequence_name} "
                f"OWNED BY {table_name}.id"
            )
        )
        conn.execute(
            text(
                "SELECT setval("
                f"'{sequence_name}', "
                f"COALESCE((SELECT MAX(id) FROM {table_name}), 0) + 1, "
                "false)"
            )
        )
        conn.execute(
            text(
                f"ALTER TABLE {table_name} "
                f"ALTER COLUMN id SET DEFAULT nextval('{sequence_name}')"
            )
        )
    log.warning("Schema migration: repaired %s.id sequence default", table_name)


def _ensure_pm_candidate_events_identity_default(engine, inspector):
    _ensure_postgres_identity_default(engine, inspector, "pm_candidate_events")


def _ensure_pm_candidates_identity_default(engine, inspector):
    _ensure_postgres_identity_default(engine, inspector, "pm_candidates")


def _ensure_decision_snapshots_identity_default(engine, inspector):
    _ensure_postgres_identity_default(engine, inspector, "decision_snapshots")


def _ensure_checkpoint_tables(engine, inspector):
    """Create checkpoint_events table if missing.

    Stores structured funnel events for every entry-opportunity stage transition.
    Used by CheckpointLogger for observability into candidate flow.
    Called during check_schema() startup. Non-destructive.

    Requirements: 11.1
    """
    from sqlalchemy import text
    from utils.dialect_sql import _pk_column, _default_timestamp

    if inspector.has_table("checkpoint_events"):
        return

    pk = _pk_column(engine)
    ts_default = _default_timestamp(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS checkpoint_events (
                    {pk},
                    stage VARCHAR(64) NOT NULL,
                    outcome_category VARCHAR(64),
                    cycle_id VARCHAR(36) NOT NULL,
                    candidate_id VARCHAR(36),
                    lineage_id VARCHAR(36),
                    profile VARCHAR(36) NOT NULL,
                    symbol VARCHAR(10),
                    setup_type VARCHAR(64),
                    decision VARCHAR(32),
                    reason_code VARCHAR(64),
                    metadata_json TEXT,
                    created_at DATETIME NOT NULL {ts_default}
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_checkpoint_events_cycle "
                "ON checkpoint_events (cycle_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_checkpoint_events_candidate "
                "ON checkpoint_events (candidate_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_checkpoint_events_stage "
                "ON checkpoint_events (stage)"
            )
        )
        conn.commit()
    log.warning("Schema migration: created checkpoint_events table with indexes")


def _ensure_watch_candidate_tables(engine, inspector):
    """Create watch_candidates table if missing.

    Called during check_schema() startup. Non-destructive — only creates
    tables that are missing.

    Requirements: 8.2, 8.3, 8.5.1, 15.1, 15.3
    """
    from sqlalchemy import text
    from utils.dialect_sql import _pk_column, _default_timestamp

    if inspector.has_table("watch_candidates"):
        return

    pk = _pk_column(engine)
    ts_default = _default_timestamp(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS watch_candidates (
                    {pk},
                    watch_id TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    created_at TEXT NOT NULL {ts_default},
                    updated_at TEXT NOT NULL {ts_default},
                    expires_at TEXT NOT NULL,
                    source_cycle_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    market_state TEXT NOT NULL,
                    setup_lifecycle_state TEXT NOT NULL,
                    timeframe_authority_json TEXT NOT NULL,
                    direction_watch TEXT NOT NULL,
                    trade_posture TEXT NOT NULL,
                    activation_conditions_json TEXT NOT NULL,
                    invalidation_conditions_json TEXT NOT NULL,
                    key_levels_json TEXT NOT NULL,
                    trigger_status_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    source_signal_snapshot_json TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'active',
                    state_changed_at TEXT,
                    outcome_json TEXT,
                    promoted_cycle_id TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_watch_candidates_active "
                "ON watch_candidates (profile_id, state, symbol) "
                "WHERE state = 'active'"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_watch_candidates_promoted_cycle "
                "ON watch_candidates (profile_id, promoted_cycle_id) "
                "WHERE state = 'promoted'"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_watch_candidates_expires "
                "ON watch_candidates (expires_at) "
                "WHERE state = 'active'"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_watch_candidates_active_unique "
                "ON watch_candidates (symbol, profile_id, direction_watch) "
                "WHERE state = 'active'"
            )
        )
        conn.commit()
    log.warning("Schema migration: created watch_candidates table with indexes")


def check_schema(engine):
    """Verify the DB schema has all expected columns. Fail fast if not."""
    from sqlalchemy import inspect as sa_inspect, text
    from db.schema import verify_wal_mode, is_sqlite

    # --- Verify WAL mode and busy_timeout (non-destructive, SQLite only) ---
    if is_sqlite(engine):
        verify_wal_mode(engine)

    inspector = sa_inspect(engine)

    # --- Auto-create funnel tables if missing (non-destructive) ---
    _ensure_funnel_tables(engine, inspector)

    # --- Auto-create candidate-ID selection tables if missing (non-destructive) ---
    _ensure_candidate_tables(engine, inspector)
    _ensure_watch_candidate_tables(engine, inspector)
    _ensure_pm_candidates_identity_default(engine, inspector)
    _ensure_pm_candidate_events_identity_default(engine, inspector)

    # --- Auto-create checkpoint funnel logging tables if missing (non-destructive) ---
    _ensure_checkpoint_tables(engine, inspector)

    # --- Auto-create provenance tables if missing (non-destructive) ---
    init_provenance_schema(engine)

    # --- Auto-create blocker mitigation tables if missing (non-destructive) ---
    init_blocker_mitigation_schema(engine)

    # --- Auto-create decision replay tables and lineage columns if missing ---
    from db.replay_schema import init_replay_db
    init_replay_db(engine)
    inspector = sa_inspect(engine)
    _ensure_decision_snapshots_identity_default(engine, inspector)

    # Expected columns per table that have been added over time.
    # If a column is missing, the system will crash on first query anyway —
    # better to catch it here with a clear message and auto-fix.
    expected = {
        "trades": ["thesis", "setup_type", "invalidators", "stop_role", "stop_updated_by", "stop_updated_at"],
        "trade_events": ["dedupe_key"],
        "cases": ["exit_category"],
        "pm_candidates": ["candidate_type", "holding_horizon", "normalized_setup_type", "rejection_reason_code"],
        "pm_candidate_events": ["candidate_type"],
        "watch_candidates": ["promoted_cycle_id"],
    }

    missing = {}
    for table, columns in expected.items():
        if not inspector.has_table(table):
            continue
        existing = {col["name"] for col in inspector.get_columns(table)}
        table_missing = [c for c in columns if c not in existing]
        if table_missing:
            missing[table] = table_missing

    if not missing:
        return

    # Auto-fix: add missing columns
    col_types = {
        "thesis": "TEXT",
        "setup_type": "VARCHAR(64)",
        "invalidators": "TEXT",
        "stop_role": "VARCHAR(32) DEFAULT 'initial'",
        "stop_updated_by": "VARCHAR(64)",
        "stop_updated_at": "DATETIME",
        "dedupe_key": "VARCHAR(256)",
        "exit_category": "VARCHAR(40)",
        "candidate_type": "TEXT DEFAULT 'intraday'",
        "holding_horizon": "INTEGER",
        "normalized_setup_type": "TEXT",
        "rejection_reason_code": "VARCHAR(64)",
        "promoted_cycle_id": "TEXT",
    }

    with engine.begin() as conn:
        for table, cols in missing.items():
            for col in cols:
                col_type = col_types.get(col, "TEXT")
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
                log.warning(f"Schema migration: added {table}.{col} ({col_type})")

    # Create unique index for trade_events dedupe_key if column was just added
    if "trade_events" in missing and "dedupe_key" in missing["trade_events"]:
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_trade_events_dedupe "
                "ON trade_events(event_type, trade_id, dedupe_key)"
            ))
        log.warning("Schema migration: created unique index ix_trade_events_dedupe on trade_events(event_type, trade_id, dedupe_key)")

    # Create partial promoted-cycle index if promoted_cycle_id was just added
    if "watch_candidates" in missing and "promoted_cycle_id" in missing["watch_candidates"]:
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_watch_candidates_promoted_cycle "
                "ON watch_candidates (profile_id, promoted_cycle_id) "
                "WHERE state = 'promoted'"
            ))
        log.warning(
            "Schema migration: created partial index idx_watch_candidates_promoted_cycle "
            "on watch_candidates (profile_id, promoted_cycle_id) WHERE state = 'promoted'"
        )

    # If stop_role was just added, backfill existing open trades
    if "trades" in missing and "stop_role" in missing["trades"]:
        backfill_stop_roles(engine)

    console.print(f"   [yellow]⚠ Auto-migrated missing columns: {missing}[/yellow]")


def backfill_stop_roles(engine):
    """Backfill stop_role for existing open trades after column migration.

    Idempotent — safe to run multiple times. Only updates trades where
    stop_role is NULL or empty string.

    Logic:
    1. Query all open trades where stop_role is NULL or empty.
    2. For each trade, look for the most recent stop-related event from
       profit_manager (event_type in 'stop_set', 'stop_update_accepted').
    3. If found: parse payload to infer breakeven/trail/initial.
    4. If not found: set stop_role = 'initial'.

    Requirements: 12.4, 12.5
    """
    import json
    import re
    from db.schema import Trade, TradeEvent, get_session

    db = get_session(engine)
    try:
        # Find open trades with NULL or empty stop_role
        trades = (
            db.query(Trade)
            .filter(
                Trade.status == "open",
                (Trade.stop_role == None) | (Trade.stop_role == ""),  # noqa: E711
            )
            .all()
        )

        if not trades:
            return

        for trade in trades:
            # Look for the most recent stop-related event from profit_manager
            latest_event = (
                db.query(TradeEvent)
                .filter(
                    TradeEvent.trade_id == trade.id,
                    TradeEvent.agent == "profit_manager",
                    TradeEvent.event_type.in_(["stop_set", "stop_update_accepted"]),
                )
                .order_by(TradeEvent.timestamp.desc())
                .first()
            )

            if latest_event:
                # Parse payload to determine role
                role = "initial"
                # Check message field
                msg = (latest_event.message or "").lower()
                # Check payload_json field
                payload_text = ""
                if latest_event.payload_json:
                    try:
                        payload = json.loads(latest_event.payload_json)
                        if isinstance(payload, dict):
                            payload_text = " ".join(
                                str(v) for v in payload.values()
                            ).lower()
                            # Also check 'reason' or 'message' keys specifically
                            reason = str(payload.get("reason", "")).lower()
                            payload_text += " " + reason
                    except (json.JSONDecodeError, TypeError):
                        payload_text = str(latest_event.payload_json).lower()

                combined = msg + " " + payload_text

                if "breakeven" in combined:
                    role = "breakeven"
                elif "trail" in combined or re.search(r"\+\d*r", combined):
                    role = "trail"

                trade.stop_role = role
            else:
                # No stop event history → initial
                trade.stop_role = "initial"

        db.commit()
        log.info(f"Backfilled stop_role for {len(trades)} open trade(s)")
    except Exception as e:
        db.rollback()
        log.error(f"Backfill stop_roles error: {e}", exc_info=True)
    finally:
        db.close()


def run_pre_market():
    """8:30 AM ET — Research + Analysis prep before open."""
    if _skip_closed_market_job("pre_market"):
        return
    log.info("=== PRE-MARKET RUN ===")
    engine = get_engine()

    # Scout — find additional symbols
    scout_symbols = []
    try:
        console.print("[bold cyan]🔭 Scout scanning for movers...[/bold cyan]")
        scout_result = scout.run(engine, WATCHLIST, scout_candidates=SCOUT_CANDIDATES)
        scout_symbols = scout_result.get("symbols", [])
        if scout_symbols:
            log.info(f"Scout picks: {', '.join(scout_symbols)} (tone: {scout_result.get('market_tone')})")
        else:
            log.info("Scout: no picks today")
    except Exception as e:
        log.error(f"Scout error: {e}", exc_info=True)

    full_watchlist = WATCHLIST + scout_symbols

    # Researcher
    market_regime = None
    try:
        console.print("[bold yellow]📰 Researcher running...[/bold yellow]")
        res = researcher.run(engine, full_watchlist)
        market_regime = res.get("market_regime")
        log.info(f"Researcher: {res.get('market_context', '')[:100]}")
    except Exception as e:
        log.error(f"Researcher error: {e}", exc_info=True)

    # Quant Researcher
    try:
        console.print("[bold blue]📐 Quant Researcher: matching strategies to conditions...[/bold blue]")
        qr_result = quant_researcher.run(engine, market_regime=market_regime)
        quant_researcher.print_report(qr_result)
        log.info(f"Quant Researcher: primary={qr_result.get('primary_strategy')} avoid={qr_result.get('strategies_to_avoid')}")
    except Exception as e:
        log.error(f"Quant Researcher error: {e}", exc_info=True)

    # Pipeline evaluation — evaluate all dynamic strategies in pipeline stages
    try:
        console.print("[bold magenta]🔄 Pipeline: evaluating strategy stages...[/bold magenta]")
        pipeline_results = run_pipeline_evaluation()
        for r in pipeline_results:
            log.info(f"Pipeline: {r['strategy_key']} @ {r['stage']} → {r['decision']} ({r['reason']})")
        if not pipeline_results:
            log.info("Pipeline: no strategies in pipeline stages")
    except Exception as e:
        log.error(f"Pipeline evaluation error: {e}", exc_info=True)

    # Analyst
    try:
        console.print("[bold blue]📊 Analyst running...[/bold blue]")
        sigs = analyst.run(engine, full_watchlist)
        for sym, sig in sigs.items():
            log.info(f"  {sym}: {sig.get('signal')} ({sig.get('strength')})")
    except Exception as e:
        log.error(f"Analyst error: {e}", exc_info=True)

    # Slack morning report — non-blocking, failures never affect trading
    try:
        from utils.slack_notifier import SlackNotifier
        notifier = SlackNotifier()
        if notifier.is_enabled():
            notifier.send_morning_report(engine)
    except Exception as e:
        log.error(f"Slack morning report error: {e}", exc_info=True)

    # Narrator morning briefing - non-blocking, failures never affect trading
    try:
        import agents.narrator as narrator
        console.print("[bold cyan]📝 Narrator: morning briefing...[/bold cyan]")
        narrator.run(engine, "morning_briefing")
    except Exception as e:
        log.error(f"Narrator morning briefing error: {e}", exc_info=True)


def run_pipeline_evaluation():
    """Pre-market: evaluate all strategies in pipeline stages.

    1. Trigger backtests for new proposals (status=backtest, no report yet).
    2. Run the full pipeline evaluation for all strategies in pipeline stages.
    """
    from deployment_pipeline import (
        run_pipeline_evaluation as evaluate,
        evaluate_backtest_gate,
        apply_gate_result,
    )
    from strategy_backtester import StrategyBacktester
    from db.schema import get_session, DynamicStrategy

    engine = get_engine()

    # --- Phase 1: trigger backtests for pending proposals ---
    db = get_session(engine)
    try:
        pending = (
            db.query(DynamicStrategy)
            .filter(
                DynamicStrategy.status == "backtest",
                (DynamicStrategy.backtest_report_id == None) | (DynamicStrategy.backtest_report_id == ""),  # noqa: E711
            )
            .all()
        )
        # Detach so we can close the session before long-running backtests
        for s in pending:
            db.expunge(s)
    finally:
        db.close()

    for strategy in pending:
        try:
            log.info(f"Pipeline: triggering backtest for '{strategy.key}'")
            backtester = StrategyBacktester(engine)
            report = backtester.run(strategy)

            # Update the strategy's backtest_report_id
            memory_key = f"backtest_report_{strategy.key}"
            db = get_session(engine)
            try:
                strat = db.query(DynamicStrategy).filter_by(id=strategy.id).first()
                if strat:
                    strat.backtest_report_id = memory_key
                    db.commit()
            finally:
                db.close()

            # Evaluate the backtest gate and apply the result
            gate_result = evaluate_backtest_gate(report)
            apply_gate_result(engine, strategy, gate_result)

            log.info(
                f"Pipeline: backtest for '{strategy.key}' → {gate_result.decision} ({gate_result.reason})"
            )
        except Exception as e:
            log.error(f"Pipeline: backtest failed for '{strategy.key}': {e}", exc_info=True)

    # --- Phase 2: evaluate all strategies in pipeline stages ---
    results = evaluate(engine)
    for r in results:
        log.info(f"Pipeline: {r['strategy_key']} @ {r['stage']} → {r['decision']} ({r['reason']})")
    return results


def run_coordinated_market_cycle():
    """Coordinated market cycle — replaces independent analyst/PM jobs when enabled.

    When PM_CYCLE_COORDINATOR_MODE == "enabled":
        Runs a single sequential pipeline: focus → analyst → freshness gate → PM → safety.

    When disabled:
        Falls back to legacy independent jobs (run_analyst_refresh + run_intraday).
    """
    from utils.gate_config import PM_CYCLE_COORDINATOR_MODE

    if PM_CYCLE_COORDINATOR_MODE != "enabled":
        # Legacy path — run independent jobs sequentially
        run_analyst_refresh()
        run_intraday()
        return

    if _skip_outside_regular_market_job("coordinated_cycle"):
        return

    log.info("=== COORDINATED MARKET CYCLE ===")
    engine = get_engine()

    from utils.cycle_coordinator import CycleCoordinator

    coordinator = CycleCoordinator(engine)
    try:
        summary = coordinator.run_market_cycle(trigger_source="scheduled")
        log.info(
            "Coordinated cycle completed: cycle_id=%s, duration=%.1fs, fresh=%d, stale=%d",
            summary.cycle_id,
            summary.total_duration_seconds,
            summary.symbols_fresh,
            summary.symbols_stale_skipped,
        )
    except Exception as exc:
        log.error("Coordinated market cycle failed: %s", exc, exc_info=True)


def _run_manual_coordinated_cycle():
    """Manual coordinated market cycle triggered via SIGUSR2 or CLI.

    Bypasses market-hours guard so the operator can test the coordinated path
    at any time. Always uses trigger_source="manual".
    """
    log.info("=== MANUAL COORDINATED MARKET CYCLE ===")
    engine = get_engine()

    from utils.cycle_coordinator import CycleCoordinator

    coordinator = CycleCoordinator(engine)
    try:
        summary = coordinator.run_market_cycle(trigger_source="manual")
        log.info(
            "Manual coordinated cycle completed: cycle_id=%s, duration=%.1fs, fresh=%d, stale=%d",
            summary.cycle_id,
            summary.total_duration_seconds,
            summary.symbols_fresh,
            summary.symbols_stale_skipped,
        )
        console.print(f"[green]Coordinated cycle completed: {summary.cycle_id} ({summary.total_duration_seconds:.1f}s)[/green]")
    except Exception as exc:
        log.error("Manual coordinated market cycle failed: %s", exc, exc_info=True)
        console.print(f"[red]Coordinated cycle failed: {exc}[/red]")


def run_analyst_refresh():
    """Analyst-only refresh — morning every 15 min, afternoon every 30 min."""
    if _skip_outside_regular_market_job("analyst_refresh"):
        return
    log.info("=== ANALYST REFRESH ===")
    engine = get_engine()

    scout_picks = []
    try:
        scout_picks = scout.get_todays_picks(engine)
    except Exception as e:
        log.error(f"Scout picks error: {e}", exc_info=True)

    # Funnel transition (Req 13.1–13.4): Use get_pm_eligible_candidates() as
    # primary source. Fall back to get_expanded_watchlist() ONLY if no valid
    # FunnelRunLog exists for today's discovery. Deduplicate across sources.
    expanded_symbols = []
    try:
        funnel_config = load_funnel_config()
        max_handoff = funnel_config.get("funnel", {}).get("ceilings", {}).get("max_pm_handoff", 3)
        expanded_symbols, source = get_funnel_or_fallback_candidates(engine, max_pm_handoff=max_handoff)
        log.info(f"Analyst refresh expanded symbols source: {source}, symbols: {expanded_symbols}")
    except Exception as e:
        log.error(f"Funnel/expanded watchlist error: {e}", exc_info=True)

    full_watchlist = build_deduplicated_watchlist(_pm_base_watchlist(), scout_picks, expanded_symbols)
    analyst_symbols = _apply_focus_list(
        engine,
        full_watchlist,
        scout_picks=scout_picks,
        expanded_symbols=expanded_symbols,
        context="analyst_refresh",
    )

    try:
        console.print("[bold blue]📊 Analyst refresh...[/bold blue]")
        signals = analyst.run(engine, analyst_symbols)

        # Outcome tracking hook (Req 10.5): record analyst signals for expanded candidates
        if expanded_symbols and signals:
            today = datetime.now().strftime("%Y-%m-%d")
            for sym in expanded_symbols:
                sig = signals.get(sym, {})
                signal_direction = (sig.get("signal") or "NO_SIGNAL").upper()
                try:
                    record_analyst_outcome(engine, sym, today, signal_direction)
                except Exception as e:
                    log.debug(f"Outcome tracking (analyst) error for {sym}: {e}")
    except Exception as e:
        log.error(f"Analyst error: {e}", exc_info=True)


def run_intraday():
    """PM decisions + stop checks — runs on the split schedule."""
    if _skip_outside_regular_market_job("intraday"):
        return
    if not _try_begin_pm_cycle("intraday"):
        return
    log.info("=== INTRADAY CYCLE ===")
    engine = get_engine()

    # Block new funnel jobs while PM cycle is active (Requirement 7.5).
    # In-progress premarket jobs can still complete within their budget.
    try:
        _run_intraday_inner(engine)
    finally:
        _end_pm_cycle("intraday")


def _run_intraday_inner(engine):
    """Internal intraday logic — separated so PM cycle blocking can wrap it."""

    # Check stop losses
    try:
        stops = bookkeeper.check_stop_losses(engine)
        if stops:
            log.warning(f"Stop losses triggered: {[(s['symbol'], s['profile']) for s in stops]}")
            from db.schema import get_session
            from agents.portfolio_manager import execute_trade
            for stop in stops:
                db = get_session(engine)
                execute_trade(db, {
                    "symbol": stop["symbol"],
                    "action": "CLOSE",
                    "quantity": 0,
                    "price": stop["price"],
                    "rationale": f"Stop loss hit at {stop['stop_loss']}",
                }, stop["profile"])
                db.close()
    except Exception as e:
        log.error(f"Stop loss check error: {e}", exc_info=True)

    # Full watchlist
    scout_picks = []
    try:
        scout_picks = scout.get_todays_picks(engine)
    except Exception as e:
        log.error(f"Scout picks error: {e}", exc_info=True)

    # Funnel transition (Req 13.1–13.4): Use get_pm_eligible_candidates() as
    # primary source. Fall back to get_expanded_watchlist() ONLY if no valid
    # FunnelRunLog exists for today's discovery. Deduplicate across sources.
    expanded_symbols = []
    try:
        funnel_config = load_funnel_config()
        max_handoff = funnel_config.get("funnel", {}).get("ceilings", {}).get("max_pm_handoff", 3)
        expanded_symbols, source = get_funnel_or_fallback_candidates(engine, max_pm_handoff=max_handoff)
        log.info(f"Intraday expanded symbols source: {source}, symbols: {expanded_symbols}")
    except Exception as e:
        log.error(f"Funnel/expanded watchlist error: {e}", exc_info=True)

    full_watchlist = build_deduplicated_watchlist(_pm_base_watchlist(), scout_picks, expanded_symbols)

    # ─── Alert intent scheduled consumption ────────────────────────────
    _alert_intent_ids = []
    _alert_symbols = []
    from utils.gate_config import PM_ALERT_DISPATCH_MODE
    if PM_ALERT_DISPATCH_MODE == "enabled" and _alert_dispatcher is not None:
        try:
            extra_symbols, _alert_intent_ids = _alert_dispatcher.consume_for_scheduled_cycle()
            if extra_symbols:
                _alert_symbols = list(extra_symbols)
                # Add intent symbols to the watchlist (deduplicating)
                for sym in extra_symbols:
                    if sym not in full_watchlist:
                        full_watchlist.append(sym)
                log.info(
                    "INTRADAY_ALERT_CONSUME: added %d intent symbols to watchlist: %s",
                    len(extra_symbols), extra_symbols,
                )
        except Exception as e:
            log.error(f"Alert intent consumption error: {e}", exc_info=True)
            _alert_intent_ids = []

    full_watchlist = _apply_focus_list(
        engine,
        full_watchlist,
        scout_picks=scout_picks,
        expanded_symbols=expanded_symbols,
        context="intraday_pm",
        required_symbols=_open_position_symbols(engine),
        alert_symbols=_alert_symbols,
    )

    # Analyst signals are refreshed by the separate run_analyst_refresh job
    # on the same schedule — no need to duplicate here.

    # PM profiles decide. Candidate-ID mode writes a registry per profile, so
    # those profiles run serially to avoid SQLite writer contention.
    console.print("[bold green]🧠 Portfolio Managers deciding...[/bold green]")

    def _run_pm(profile_id):
        try:
            eng = get_engine()
            result = pm.run_profile(eng, full_watchlist, profile_id)
            for d in result.get("decisions", []):
                status = "✅" if d.get("executed") else "❌"
                log.info(f"  [{profile_id}] {status} {d['action']} {d.get('quantity', '')} {d['symbol']} @ ${d.get('price', 0):.2f}")

            # Outcome tracking hook (Req 10.5): record PM status for expanded candidates
            if expanded_symbols:
                today = datetime.now().strftime("%Y-%m-%d")
                decisions = result.get("decisions", [])
                decided_symbols = set()
                for d in decisions:
                    sym = d.get("symbol", "")
                    if sym not in expanded_symbols:
                        continue
                    decided_symbols.add(sym)
                    if d.get("executed"):
                        pm_status = "executed"
                    else:
                        pm_status = "rejected"
                    try:
                        record_pm_outcome(eng, sym, today, pm_status)
                    except Exception:
                        pass
                # Symbols in expanded watchlist that PM considered eligible
                # (had a signal) but didn't produce a decision → "no_entry"
                for sym in expanded_symbols:
                    if sym not in decided_symbols:
                        try:
                            record_pm_outcome(eng, sym, today, "no_entry")
                        except Exception:
                            pass
        except Exception as e:
            log.error(f"PM {profile_id} error: {e}", exc_info=True)

    try:
        _run_pm_profile_jobs(pm.ACTIVE_PROFILES, _run_pm)
    except Exception as e:
        # ─── Revert alert intent claim on PM failure ───────────────────
        if _alert_intent_ids and _alert_dispatcher is not None:
            try:
                _alert_dispatcher.revert_scheduled_claim(_alert_intent_ids, str(e))
            except Exception as revert_err:
                log.error(f"Alert intent revert error: {revert_err}", exc_info=True)
        raise
    else:
        # ─── Confirm alert intent consumption on PM success ────────────
        if _alert_intent_ids and _alert_dispatcher is not None:
            try:
                _alert_dispatcher.confirm_scheduled_consumption(_alert_intent_ids)
            except Exception as e:
                log.error(f"Alert intent confirm error: {e}", exc_info=True)

    # Dashboard
    try:
        bookkeeper.print_dashboard(engine)
    except Exception as e:
        log.error(f"Dashboard error: {e}", exc_info=True)


def run_sector_scout_confirmation():
    """10:00 AM ET — Sector scout confirmation scan after initial session volatility."""
    if _skip_closed_market_job("sector_scout_confirmation"):
        return
    log.info("=== SECTOR SCOUT CONFIRMATION ===")
    engine = get_engine()
    try:
        console.print("[bold cyan]🔭 Sector Scout confirmation scan...[/bold cyan]")
        result = run_intraday_scan(engine, WATCHLIST, run_type="confirmation")
        picks = result.get("picks", [])
        if picks:
            symbols = [p.get("symbol", p) if isinstance(p, dict) else str(p) for p in picks]
            log.info(f"Sector Scout confirmation: {len(picks)} picks — {', '.join(symbols)}")
        else:
            log.info("Sector Scout confirmation: no new picks")
        if result.get("skipped_symbols"):
            log.info(f"  Cooldown skipped: {', '.join(result['skipped_symbols'])}")
    except Exception as e:
        log.error(f"Sector Scout confirmation error: {e}", exc_info=True)


def run_sector_scout_midday():
    """12:30 PM ET — Sector scout midday unusual-mover scan. DISABLED in v1."""
    # Disabled in v1 per requirement 8.1 — no broad multi-sector discovery
    # during market hours. Kept as a no-op for reference; not registered in scheduler.
    log.info("sector_scout_midday: DISABLED in v1 (no broad market-hours scan)")
    return


# ---------------------------------------------------------------------------
# Premarket Candidate Funnel Jobs
# ---------------------------------------------------------------------------


def run_funnel_discovery_job():
    """06:00 ET — Premarket funnel discovery (bounded sector scanning).

    Guards: Blocks new broad discovery scans after the configured
    confirmation_time (default 09:35 ET) per Requirement 7.1.
    Also blocks when PM cycle is active (Requirement 7.5).
    """
    if _skip_closed_market_job("funnel_discovery"):
        return

    if _is_pm_cycle_blocking_funnel():
        log.info("FUNNEL_BLOCKED: funnel_discovery skipped — PM cycle active (Req 7.5)")
        return

    # Market-hours resource protection guard (Req 7.1, 8.2):
    # Block new broad discovery after confirmation_time.
    from utils.funnel_orchestrator import is_discovery_allowed
    funnel_cfg = load_funnel_config()
    if not is_discovery_allowed(funnel_cfg):
        log.info(
            "FUNNEL_DISCOVERY_BLOCKED: broad discovery not allowed after "
            "confirmation_time — skipping"
        )
        return

    log.info("=== FUNNEL DISCOVERY ===")
    engine = get_engine()
    try:
        from utils.sector_scout import load_sector_scout_config

        console.print("[bold cyan]🔭 Funnel discovery (premarket)...[/bold cyan]")
        config = load_sector_scout_config()
        funnel_config = load_funnel_config()

        result = run_funnel_discovery(engine, config, funnel_config)
        n_candidates = len(result.candidates)
        log.info(
            f"Funnel discovery: {n_candidates} candidates persisted "
            f"({result.selection_mode}), "
            f"sectors completed={result.sectors_completed}, "
            f"timed_out={result.sectors_timed_out}, "
            f"skipped={result.sectors_skipped}, "
            f"duration={result.total_duration_seconds:.1f}s"
        )
        if n_candidates:
            symbols = [c.symbol for c in result.candidates]
            console.print(f"  [green]Shortlist: {', '.join(symbols)}[/green]")
        else:
            console.print("  [yellow]No candidates qualified[/yellow]")
    except Exception as e:
        log.error(f"Funnel discovery error: {e}", exc_info=True)
        try:
            funnel_config = load_funnel_config()
            budget = (
                funnel_config.get("funnel", {})
                .get("budgets", {})
                .get("total_pipeline_seconds", 90)
            )
        except Exception:
            budget = 90
        _log_funnel_job_error(engine, stage="discovery", budget_seconds=budget, error=e)


def run_funnel_research_job():
    """06:30 ET — Funnel researcher qualification of discovery shortlist."""
    if _skip_closed_market_job("funnel_research"):
        return
    if _is_pm_cycle_blocking_funnel():
        log.info("FUNNEL_BLOCKED: funnel_research skipped — PM cycle active (Req 7.5)")
        return
    log.info("=== FUNNEL RESEARCH ===")
    engine = get_engine()
    try:
        from datetime import date as date_type
        from zoneinfo import ZoneInfo
        from db.schema import FunnelCandidate, get_session

        console.print("[bold cyan]📋 Funnel research qualification...[/bold cyan]")
        funnel_config = load_funnel_config()
        ceilings = funnel_config.get("funnel", {}).get("ceilings", {})
        max_promoted = ceilings.get("max_researcher_promoted", 3)

        # Query today's awaiting_research candidates
        today_ny = datetime.now(ZoneInfo("America/New_York")).date()
        session = get_session(engine)
        try:
            candidates = (
                session.query(FunnelCandidate)
                .filter(
                    FunnelCandidate.date == today_ny,
                    FunnelCandidate.stage_status == "awaiting_research",
                    FunnelCandidate.expired == False,  # noqa: E712
                )
                .all()
            )
            session.expunge_all()
        finally:
            session.close()

        if not candidates:
            log.info("Funnel research: no awaiting_research candidates for %s", today_ny)
            return

        decisions = run_funnel_qualification(
            engine, candidates, config={}, max_promoted=max_promoted
        )
        promoted = sum(1 for d in decisions if d.decision == "promoted")
        rejected = sum(1 for d in decisions if d.decision == "rejected")
        log.info(
            f"Funnel research: {promoted} promoted, {rejected} rejected "
            f"out of {len(candidates)} candidates"
        )
    except Exception as e:
        log.error(f"Funnel research error: {e}", exc_info=True)
        try:
            funnel_config = load_funnel_config()
            budget = (
                funnel_config.get("funnel", {})
                .get("budgets", {})
                .get("total_pipeline_seconds", 90)
            )
        except Exception:
            budget = 90
        _log_funnel_job_error(engine, stage="research", budget_seconds=budget, error=e)


def run_funnel_analysis_job():
    """07:15 ET — Funnel analyst setup classification."""
    if _skip_closed_market_job("funnel_analysis"):
        return
    if _is_pm_cycle_blocking_funnel():
        log.info("FUNNEL_BLOCKED: funnel_analysis skipped — PM cycle active (Req 7.5)")
        return
    log.info("=== FUNNEL ANALYSIS ===")
    engine = get_engine()
    try:
        from zoneinfo import ZoneInfo
        from db.schema import FunnelCandidate, get_session

        console.print("[bold cyan]📊 Funnel analyst setup classification...[/bold cyan]")

        # Query today's awaiting_analysis candidates
        today_ny = datetime.now(ZoneInfo("America/New_York")).date()
        session = get_session(engine)
        try:
            candidates = (
                session.query(FunnelCandidate)
                .filter(
                    FunnelCandidate.date == today_ny,
                    FunnelCandidate.stage_status == "awaiting_analysis",
                    FunnelCandidate.expired == False,  # noqa: E712
                )
                .all()
            )
            session.expunge_all()
        finally:
            session.close()

        if not candidates:
            log.info("Funnel analysis: no awaiting_analysis candidates for %s", today_ny)
            return

        decisions = run_funnel_analysis(engine, candidates)
        promoted = sum(1 for d in decisions if d.decision == "promoted")
        rejected = sum(1 for d in decisions if d.decision == "rejected")
        log.info(
            f"Funnel analysis: {promoted} promoted, {rejected} rejected "
            f"out of {len(candidates)} candidates"
        )
    except Exception as e:
        log.error(f"Funnel analysis error: {e}", exc_info=True)
        _log_funnel_job_error(engine, stage="analysis", budget_seconds=90, error=e)


def run_funnel_confirmation_job():
    """09:35 ET — Opening confirmation of funnel shortlist."""
    if _skip_closed_market_job("funnel_confirmation"):
        return
    if _is_pm_cycle_blocking_funnel():
        log.info("FUNNEL_BLOCKED: funnel_confirmation skipped — PM cycle active (Req 7.5)")
        return
    log.info("=== FUNNEL CONFIRMATION ===")
    engine = get_engine()
    try:
        from zoneinfo import ZoneInfo
        from db.schema import FunnelCandidate, get_session

        console.print("[bold cyan]✅ Funnel opening confirmation...[/bold cyan]")
        funnel_config = load_funnel_config()
        budgets = funnel_config.get("funnel", {}).get("budgets", {})
        budget_seconds = budgets.get("confirmation_budget_seconds", 45)

        # Query today's awaiting_confirmation candidates
        today_ny = datetime.now(ZoneInfo("America/New_York")).date()
        session = get_session(engine)
        try:
            candidates = (
                session.query(FunnelCandidate)
                .filter(
                    FunnelCandidate.date == today_ny,
                    FunnelCandidate.stage_status == "awaiting_confirmation",
                    FunnelCandidate.expired == False,  # noqa: E712
                )
                .all()
            )
            session.expunge_all()
        finally:
            session.close()

        if not candidates:
            log.info("Funnel confirmation: no awaiting_confirmation candidates for %s", today_ny)
            return

        decisions = run_opening_confirmation(
            engine, candidates, budget_seconds=budget_seconds
        )
        promoted = sum(1 for d in decisions if d.decision == "promoted")
        rejected = sum(1 for d in decisions if d.decision == "rejected")
        log.info(
            f"Funnel confirmation: {promoted} promoted, {rejected} rejected "
            f"out of {len(candidates)} candidates"
        )
    except Exception as e:
        log.error(f"Funnel confirmation error: {e}", exc_info=True)
        try:
            funnel_config = load_funnel_config()
            budget = (
                funnel_config.get("funnel", {})
                .get("budgets", {})
                .get("confirmation_budget_seconds", 45)
            )
        except Exception:
            budget = 45
        _log_funnel_job_error(engine, stage="confirmation", budget_seconds=budget, error=e)


def run_funnel_confirmation_retry_job():
    """10:00 ET — Bounded shortlist confirmation retry (replaces broad sector scan)."""
    if _skip_closed_market_job("funnel_confirmation_retry"):
        return
    if _is_pm_cycle_blocking_funnel():
        log.info("FUNNEL_BLOCKED: funnel_confirmation_retry skipped — PM cycle active (Req 7.5)")
        return
    log.info("=== FUNNEL CONFIRMATION RETRY ===")
    engine = get_engine()
    try:
        console.print("[bold cyan]🔄 Funnel confirmation retry (10:00 ET)...[/bold cyan]")
        decisions = run_confirmation_retry(engine)
        if decisions:
            promoted = sum(1 for d in decisions if d.decision == "promoted")
            rejected = sum(1 for d in decisions if d.decision == "rejected")
            log.info(
                f"Funnel confirmation retry: {promoted} promoted, {rejected} rejected "
                f"out of {len(decisions)} candidates"
            )
        else:
            log.info("Funnel confirmation retry: no candidates to retry")
    except Exception as e:
        log.error(f"Funnel confirmation retry error: {e}", exc_info=True)
        try:
            funnel_config = load_funnel_config()
            budget = (
                funnel_config.get("funnel", {})
                .get("budgets", {})
                .get("market_hours_confirmation_budget_seconds", 60)
            )
        except Exception:
            budget = 60
        _log_funnel_job_error(engine, stage="confirmation_retry", budget_seconds=budget, error=e)


def run_manual_intraday_discovery_job():
    """Operator-triggered intraday discovery — NOT scheduled automatically.

    Runs the same bounded discovery pipeline as premarket but labels results
    with source_run="manual_intraday". Enforces the same Total_Pipeline_Budget
    (default 90s) and max_discovery_shortlist ceiling (default 5).

    Guards:
    - Enforces market_hours_confirmation_budget_seconds on any subsequent
      market-hours confirmation.
    - Logs prominently in FunnelRunLog so manual runs are distinguishable.

    Requirements: 8.3
    """
    if _skip_closed_market_job("manual_intraday_discovery"):
        return
    log.info("=== MANUAL INTRADAY DISCOVERY (operator-triggered) ===")
    engine = get_engine()
    try:
        from utils.sector_scout import load_sector_scout_config
        from utils.funnel_orchestrator import run_manual_intraday_discovery

        console.print("[bold cyan]🔭 Manual intraday discovery (operator-triggered)...[/bold cyan]")
        config = load_sector_scout_config()
        funnel_config = load_funnel_config()

        result = run_manual_intraday_discovery(engine, config, funnel_config)
        n_candidates = len(result.candidates)
        log.info(
            f"Manual intraday discovery: {n_candidates} candidates persisted "
            f"(mode={result.selection_mode}, "
            f"source_run=manual_intraday, "
            f"duration={result.total_duration_seconds:.1f}s)"
        )
        if n_candidates:
            symbols = [c.symbol for c in result.candidates]
            console.print(f"  [green]Manual shortlist: {', '.join(symbols)}[/green]")
        else:
            console.print("  [yellow]No candidates qualified[/yellow]")
    except Exception as e:
        log.error(f"Manual intraday discovery error: {e}", exc_info=True)


def run_weekly_prep():
    """Sunday 5:00 PM ET — Weekly prep, sets Monday context."""
    log.info("=== SUNDAY WEEKLY PREP ===")
    engine = get_engine()
    try:
        result = weekly_prep.run(engine, WATCHLIST)
        log.info(f"Weekly prep complete. Regime: {result['briefing'].get('market_regime')}")
        for profile_id, stance in result["stances"].items():
            log.info(f"  {profile_id}: {stance.get('weekly_stance')} | {stance.get('stance_reason', '')}")
    except Exception as e:
        log.error(f"Weekly prep error: {e}", exc_info=True)

    # Meta Reviewer — runs after weekly prep
    try:
        import agents.meta_reviewer as meta_reviewer
        console.print("[bold magenta]🔬 Meta Reviewer analyzing system performance...[/bold magenta]")
        meta_result = meta_reviewer.run(engine)
        log.info(f"Meta Reviewer: {meta_result.get('overall_assessment', '')[:200]}")
    except Exception as e:
        log.error(f"Meta Reviewer error: {e}", exc_info=True)


def _record_expanded_trade_outcomes(engine) -> None:
    """Record trade outcomes for expanded watchlist candidates (Req 10.5).

    Checks today's closed trades and records outcomes for any symbols
    that were in the expanded watchlist.
    """
    from db.schema import Trade, get_session

    today = datetime.now().strftime("%Y-%m-%d")
    expanded_symbols = []
    try:
        expanded_symbols = get_expanded_watchlist(engine)
    except Exception:
        return

    if not expanded_symbols:
        return

    db = get_session(engine)
    try:
        # Find trades closed today for expanded symbols
        today_start = datetime.strptime(today, "%Y-%m-%d")
        today_end = today_start + timedelta(days=1)

        closed_trades = (
            db.query(Trade)
            .filter(
                Trade.symbol.in_(expanded_symbols),
                Trade.status == "closed",
                Trade.exit_time >= today_start,
                Trade.exit_time < today_end,
            )
            .all()
        )

        for trade in closed_trades:
            outcome = {
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "profile": trade.profile,
                "setup_type": trade.setup_type,
            }
            try:
                record_trade_outcome(engine, trade.symbol, today, outcome)
            except Exception:
                pass
    finally:
        db.close()


def run_post_market():
    """4:15 PM ET — End of day wrap-up."""
    if _skip_closed_market_job("post_market"):
        return
    log.info("=== POST-MARKET / END OF DAY ===")
    engine = get_engine()

    # Expire session watch candidates at end of day
    from utils.gate_config import MARKET_STATE_MODE
    if MARKET_STATE_MODE != "disabled":
        try:
            from utils.watch_candidates import expire_session_watch_candidates
            expire_session_watch_candidates(engine)
        except Exception as exc:
            log.warning("End-of-day watch candidate expiration failed: %s", exc)

    # Outcome tracking hook (Req 10.5): record trade outcomes for expanded candidates
    try:
        _record_expanded_trade_outcomes(engine)
    except Exception as e:
        log.debug(f"Outcome tracking (trade) error: {e}")

    # Reviewer runs independently — failure here doesn't block EOD
    try:
        console.print("[bold magenta]🔍 Reviewer scoring trades...[/bold magenta]")
        review = reviewer.run(engine, min_unreviewed=1)
        log.info(f"Reviewer: {review.get('batch_feedback', review.get('message', ''))[:200]}")
    except Exception as e:
        log.error(f"Reviewer error: {e}", exc_info=True)

    # EOD bookkeeping always runs
    try:
        summary = bookkeeper.end_of_day(engine)
        log.info(f"Day summary: P&L ${summary['daily_pnl']:+,.2f} | {summary['trades']} trades | {summary['wins']}W {summary['losses']}L")
        bookkeeper.print_dashboard(engine)
    except Exception as e:
        log.error(f"EOD bookkeeper error: {e}", exc_info=True)


def run_daily_review():
    """4:30 PM ET — Daily review journal generation."""
    if _skip_closed_market_job("daily_review"):
        return
    log.info("=== DAILY REVIEW ===")
    engine = get_engine()
    try:
        console.print("[bold cyan]📓 Daily Review generating journal...[/bold cyan]")
        result = daily_review.run(engine)
        log.info(f"Daily Review: {result.get('date', 'unknown')} — confidence: {result.get('completeness', {}).get('confidence', 'unknown')}")
    except Exception as e:
        log.error(f"Daily Review error: {e}", exc_info=True)

    # Slack afternoon report — non-blocking, failures never affect trading
    try:
        from utils.slack_notifier import SlackNotifier
        notifier = SlackNotifier()
        if notifier.is_enabled():
            notifier.send_afternoon_report(engine)
    except Exception as e:
        log.error(f"Slack afternoon report error: {e}", exc_info=True)


def run_shadow_outcomes():
    """Score blocked trade candidates after their outcome windows mature."""
    if _skip_outside_regular_market_job("shadow_outcomes"):
        return
    engine = get_engine()
    try:
        result = update_blocked_candidate_outcomes(engine)
        if result.get("inserted"):
            log.info(f"Shadow outcomes: {result}")
    except Exception as e:
        log.error(f"Shadow outcomes error: {e}", exc_info=True)


def run_ceo_daily():
    """4:45 PM ET — CEO daily operating memo."""
    if _skip_closed_market_job("ceo_daily"):
        return
    log.info("=== CEO DAILY MEMO ===")
    engine = get_engine()
    try:
        console.print("[bold magenta]🧭 CEO: daily operating memo...[/bold magenta]")
        memo = ceo.run(engine, period="daily")
        log.info(f"CEO daily: constraint={memo.get('biggest_constraint', '')[:200]}")
    except Exception as e:
        log.error(f"CEO daily memo error: {e}", exc_info=True)


def run_ceo_weekly():
    """Friday 4:50 PM ET — CEO weekly strategy memo."""
    if _skip_closed_market_job("ceo_weekly"):
        return
    log.info("=== CEO WEEKLY STRATEGY MEMO ===")
    engine = get_engine()
    try:
        console.print("[bold magenta]🧭 CEO: weekly strategy memo...[/bold magenta]")
        memo = ceo.run(engine, period="weekly")
        log.info(f"CEO weekly: constraint={memo.get('biggest_constraint', '')[:200]}")
    except Exception as e:
        log.error(f"CEO weekly memo error: {e}", exc_info=True)


def run_once():
    """Run a single full cycle manually (for testing)."""
    engine = get_engine()
    ensure_initial_balance(engine)
    check_schema(engine)
    ensure_shadow_ledger_schema(engine)
    console.print("[bold]Running single cycle...[/bold]")

    from utils.gate_config import PM_CYCLE_COORDINATOR_MODE

    if PM_CYCLE_COORDINATOR_MODE == "enabled":
        from utils.cycle_coordinator import CycleCoordinator
        coordinator = CycleCoordinator(engine)
        summary = coordinator.run_market_cycle(trigger_source="manual")
        console.print(f"[green]Coordinated cycle completed: {summary.cycle_id} ({summary.total_duration_seconds:.1f}s)[/green]")
    else:
        run_pre_market()
        run_intraday()


def check_llm_connectivity():
    """Optionally ping LLM providers at startup to catch config issues early."""
    enabled = os.getenv("STARTUP_LLM_CONNECTIVITY_CHECKS", "false").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        log.info("Startup LLM connectivity checks disabled")
        return

    from utils.llm import call_llm
    probe = "Reply with one word: ok"

    # High tier
    try:
        result = call_llm("You are a test.", probe, tier="high", purpose="startup_probe:high")
        console.print(f"   [green]✓ LLM high tier OK[/green] ({os.getenv('LLM_PROVIDER')} / {os.getenv('LLM_MODEL')})")
    except Exception as e:
        console.print(f"   [red]✗ LLM high tier FAILED: {e}[/red]")
        log.error(f"LLM high tier check failed: {e}")

    # Low tier (only if configured separately)
    if os.getenv("LLM_LOW_PROVIDER"):
        try:
            result = call_llm("You are a test.", probe, tier="low", purpose="startup_probe:low")
            console.print(f"   [green]✓ LLM low tier OK[/green] ({os.getenv('LLM_LOW_PROVIDER')} / {os.getenv('LLM_LOW_MODEL')})")
        except Exception as e:
            console.print(f"   [yellow]⚠ LLM low tier FAILED: {e} (will use fallback)[/yellow]")
            log.warning(f"LLM low tier check failed: {e}")

    # Finance tier (only if configured)
    if os.getenv("LLM_FINANCE_PROVIDER"):
        try:
            result = call_llm("You are a test.", probe, tier="finance", purpose="startup_probe:finance")
            console.print(f"   [green]✓ LLM finance tier OK[/green] "
                          f"({os.getenv('LLM_FINANCE_PROVIDER')} / {os.getenv('LLM_FINANCE_MODEL')})")
        except Exception as e:
            console.print(f"   [yellow]⚠ LLM finance tier FAILED: {e} (will use medium fallback)[/yellow]")
            log.warning(f"LLM finance tier check failed: {e}")


def run_price_monitor():
    """Every 60 seconds — check prices against stops/targets/key levels."""
    # Only run during market hours (9:30 AM - 4:00 PM ET)
    from pytz import timezone
    et = datetime.now(timezone("America/New_York"))
    if _skip_closed_market_job("price_monitor", et):
        return
    market_open = et.replace(hour=9, minute=30, second=0)
    market_close = et.replace(hour=16, minute=0, second=0)
    if not (market_open <= et <= market_close):
        return

    engine = get_engine()
    try:
        import agents.price_monitor as price_monitor
        result = price_monitor.run(engine)

        # Persist live alerts for the web UI
        all_live_alerts = []
        for t in result.get("entry_triggers", []):
            all_live_alerts.append({
                "type": t.get("type", "entry"),
                "symbol": t.get("symbol"),
                "price": t.get("price"),
                "detail": f"{t.get('signal', '')} broke {t.get('level_name', '')}={t.get('level', '')}",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        for a in result.get("momentum_alerts", []):
            if a["type"] == "rapid_move":
                all_live_alerts.append({
                    "type": "rapid_move",
                    "symbol": a["symbol"],
                    "price": a["price"],
                    "detail": f"{a['direction']} {a['change_pct']}% in {a['window_minutes']}min",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
            elif a["type"] == "approaching_level":
                all_live_alerts.append({
                    "type": "approaching",
                    "symbol": a["symbol"],
                    "price": a["price"],
                    "detail": f"within {a['distance_pct']}% of {a['level_name']}={a['level_value']}",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
        for t in result.get("stop_triggers", []):
            all_live_alerts.append({
                "type": t.get("type", "stop"),
                "symbol": t.get("symbol"),
                "price": t.get("price"),
                "detail": f"{t['type']} at {t.get('level', '')} ({t.get('profile', '')})",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        if all_live_alerts:
            from db.schema import get_session as _gs, AgentMemory
            _db = _gs(engine)
            _db.add(AgentMemory(
                agent="price_monitor",
                symbol=None,
                key="live_alerts",
                value=json.dumps(all_live_alerts),
            ))
            _db.commit()
            _db.close()

        # Execute stop losses immediately
        for trigger in result.get("stop_triggers", []):
            try:
                from db.schema import get_session
                from agents.portfolio_manager import execute_trade
                db = get_session(engine)
                execute_trade(db, {
                    "symbol": trigger["symbol"],
                    "action": "CLOSE",
                    "quantity": 0,
                    "price": trigger["price"],
                    "rationale": f"Price monitor: {trigger['type']} at {trigger['price']} (level: {trigger['level']})",
                }, trigger["profile"])
                db.close()
                log.warning(f"Price monitor closed {trigger['symbol']} ({trigger['profile']}): {trigger['type']}")
            except Exception as e:
                log.error(f"Price monitor close error: {e}")

        # For entry triggers or rapid moves, route based on dispatch mode
        entry_triggers = result.get("entry_triggers", [])
        momentum_alerts = [a for a in result.get("momentum_alerts", []) if a["type"] == "rapid_move"]
        all_alerts = entry_triggers + momentum_alerts

        if all_alerts:
            from utils.gate_config import PM_ALERT_DISPATCH_MODE

            if PM_ALERT_DISPATCH_MODE != "disabled":
                # New path: record durable alert intents for deferred dispatch
                from agents.price_monitor import record_alert_intent
                for alert in all_alerts:
                    record_alert_intent(engine, alert)
            else:
                # Legacy path: inline LLM filter + PM spawn (preserved when disabled)
                from agents.price_monitor import filter_alert_with_llm
                actionable_symbols = []
                for alert in all_alerts:
                    try:
                        assessment = filter_alert_with_llm(alert, engine)
                        if assessment.get("actionable"):
                            actionable_symbols.append(alert["symbol"])
                            log.info(f"  Alert ACTIONABLE: {alert['symbol']} — {assessment.get('reasoning', '')}")
                        else:
                            log.info(f"  Alert filtered out: {alert['symbol']} — {assessment.get('reasoning', '')}")
                    except Exception as e:
                        actionable_symbols.append(alert["symbol"])  # pass through on error

                action_symbols = list(set(actionable_symbols))
                if action_symbols:
                    if not _try_begin_pm_cycle("price_monitor"):
                        log.info(
                            "Price monitor: PM trigger skipped because another PM cycle is active for %s",
                            action_symbols,
                        )
                        return
                    log.info(f"Price monitor: triggering PM (local LLM) for {action_symbols}")
                    def _run_pm_async(syms):
                        try:
                            eng = get_engine()
                            def _run_pm(pid):
                                try:
                                    pm_result = pm.run_profile(eng, WATCHLIST + syms, pid, tier="medium")
                                    for d in pm_result.get("decisions", []):
                                        if d.get("executed"):
                                            log.info(f"  ⚡ [{pid}] {d['action']} {d.get('quantity','')} {d['symbol']} @ ${d.get('price',0):.2f}")
                                except Exception as e:
                                    log.error(f"Price monitor PM {pid} error: {e}")
                            _run_pm_profile_jobs(pm.ACTIVE_PROFILES, _run_pm)
                        finally:
                            _end_pm_cycle("price_monitor")
                    threading.Thread(target=_run_pm_async, args=(action_symbols,), daemon=True).start()

    except Exception as e:
        log.error(f"Price monitor error: {e}", exc_info=True)


def run_news_monitor():
    """Every 2 hours — check for breaking news catalysts."""
    if os.getenv("NEWS_MONITOR_ENABLED", "true").strip().lower() in ("false", "0", "no", "disabled"):
        log.info("News monitor disabled by NEWS_MONITOR_ENABLED")
        return
    if _skip_closed_market_job("news_monitor"):
        return
    log.info("=== NEWS MONITOR ===")
    engine = get_engine()
    try:
        import agents.news_monitor as news_monitor
        result = news_monitor.run(engine)
        if result.get("alerts"):
            log.info(f"News monitor: {len(result['alerts'])} alerts found")
        else:
            log.info(f"News monitor: {result.get('market_update', 'no updates')}")
    except Exception as e:
        log.error(f"News monitor error: {e}", exc_info=True)


def run_position_health():
    """Every hour — review open position health."""
    if _skip_closed_market_job("position_health"):
        return
    log.info("=== POSITION HEALTH CHECK ===")
    engine = get_engine()
    try:
        import agents.position_health as position_health
        result = position_health.run(engine)
        log.info(f"Position health: {result.get('summary', '')}")
    except Exception as e:
        log.error(f"Position health error: {e}", exc_info=True)


def run_price_spike_news_check():
    """Every ~15 min — check for price spikes and fetch news for spiking symbols."""
    from pytz import timezone
    et = datetime.now(timezone("America/New_York"))
    if _skip_closed_market_job("price_spike_news", et):
        return
    market_open = et.replace(hour=9, minute=30, second=0)
    market_close = et.replace(hour=16, minute=0, second=0)
    if not (market_open <= et <= market_close):
        return

    engine = get_engine()
    from agents.price_monitor import get_batch_quotes, get_price_history
    from utils.catalyst_freshness import PRICE_SPIKE_THRESHOLD_PCT, PRICE_SPIKE_WINDOW_MINUTES

    quotes = get_batch_quotes(WATCHLIST)

    spiking = []
    for sym, price in quotes.items():
        history = get_price_history(sym)
        if len(history) < 2:
            continue
        # Find price from ~PRICE_SPIKE_WINDOW_MINUTES ago
        cutoff = datetime.utcnow() - timedelta(minutes=PRICE_SPIKE_WINDOW_MINUTES)
        old_prices = [(t, p) for t, p in history if t <= cutoff]
        if not old_prices:
            continue
        old_price = old_prices[-1][1]
        change_pct = abs((price - old_price) / old_price) * 100
        if change_pct >= PRICE_SPIKE_THRESHOLD_PCT:
            spiking.append(sym)
            log.info(f"📰 Price spike detected: {sym} moved {change_pct:.1f}% in {PRICE_SPIKE_WINDOW_MINUTES}min — fetching news")

    if spiking:
        from agents.news_monitor import fetch_and_store_news
        fetch_and_store_news(engine, spiking, source_tag="price_spike")


def run_position_news_poll():
    """Every ~30 min — fetch news for symbols with open positions."""
    from pytz import timezone
    et = datetime.now(timezone("America/New_York"))
    if _skip_closed_market_job("position_news_poll", et):
        return
    market_open = et.replace(hour=9, minute=30, second=0)
    market_close = et.replace(hour=16, minute=0, second=0)
    if not (market_open <= et <= market_close):
        return

    engine = get_engine()
    from db.schema import get_session, Position
    db = get_session(engine)
    positions = db.query(Position).all()
    db.close()

    held_symbols = list(set(p.symbol for p in positions))
    if not held_symbols:
        return

    log.info(f"📰 Position news poll for: {', '.join(held_symbols)}")
    from agents.news_monitor import fetch_and_store_news
    fetch_and_store_news(engine, held_symbols, source_tag="position_poll")


def run_position_timer():
    """Every 5 minutes — check position hold times and enforce exits."""
    if _skip_outside_regular_market_job("position_timer"):
        return
    engine = get_engine()
    try:
        import agents.position_timer as position_timer
        result = position_timer.run(engine)
        if result.get("closes"):
            log.warning(f"Position timer: {len(result['closes'])} closes")
        if result.get("repairs"):
            log.info(f"Position timer: {len(result['repairs'])} stop repairs")
        if result.get("skipped"):
            log.info(f"Position timer: {len(result['skipped'])} skipped")
    except Exception as e:
        log.error(f"Position timer error: {e}", exc_info=True)


def run_narrator(update_type: str):
    """Generic narrator runner for cron-triggered update types."""
    if os.getenv("NARRATOR_ENABLED", "true").strip().lower() in ("false", "0", "no", "disabled"):
        log.info(f"Narrator {update_type} disabled by NARRATOR_ENABLED")
        return
    if update_type != "sunday_prep" and _skip_closed_market_job(f"narrator_{update_type}"):
        return
    log.info(f"=== NARRATOR: {update_type} ===")
    engine = get_engine()
    try:
        import agents.narrator as narrator
        console.print(f"[bold cyan]📝 Narrator: {update_type}...[/bold cyan]")
        result = narrator.run(engine, update_type)
        if result.get("skipped"):
            log.info(f"Narrator {update_type}: skipped (already exists)")
        else:
            log.info(f"Narrator {update_type}: generated")
    except Exception as e:
        log.error(f"Narrator {update_type} error: {e}", exc_info=True)


def main():
    engine = get_engine()
    ensure_initial_balance(engine)
    check_schema(engine)
    ensure_shadow_ledger_schema(engine)

    # Expire stale watch candidates (crash recovery / idempotent startup sweep)
    from utils.gate_config import MARKET_STATE_MODE
    if MARKET_STATE_MODE != "disabled":
        try:
            from sqlalchemy import text as _text
            with engine.connect() as conn:
                expired = conn.execute(
                    _text(
                        "UPDATE watch_candidates SET state = 'expired', "
                        "state_changed_at = CURRENT_TIMESTAMP, "
                        "updated_at = CURRENT_TIMESTAMP "
                        "WHERE state = 'active' AND expires_at < CURRENT_TIMESTAMP"
                    )
                )
                if expired.rowcount > 0:
                    log.info("Startup: expired %d stale watch candidates", expired.rowcount)
                conn.commit()
        except Exception as exc:
            log.warning("Startup watch candidate expiration sweep failed: %s", exc)

    # Initialize alert dispatch schema (idempotent IF NOT EXISTS DDL)
    from utils.alert_dispatch_schema import init_alert_dispatch_schema
    init_alert_dispatch_schema(engine)

    check_llm_connectivity()

    # Detect and log any funnel jobs that were missed due to late startup (Req 12.7)
    funnel_config = load_funnel_config()
    _check_missed_funnel_jobs(engine, funnel_config)

    # ─── Alert Dispatcher Construction ─────────────────────────────────
    global _alert_dispatcher
    from utils.gate_config import PM_ALERT_DISPATCH_MODE, PM_ALERT_DISPATCHER_INTERVAL_SECONDS

    if PM_ALERT_DISPATCH_MODE != "disabled":
        from utils.alert_intent_store import AlertIntentStore
        from utils.alert_dispatcher import AlertDispatcher

        _alert_intent_store = AlertIntentStore(engine)
        _alert_dispatcher = AlertDispatcher(
            engine=engine,
            intent_store=_alert_intent_store,
            begin_pm_cycle=_try_begin_pm_cycle,
            end_pm_cycle=_end_pm_cycle,
        )
        # Startup crash recovery
        _alert_dispatcher._recover_stale_intents()

    scheduler = BlockingScheduler(timezone="America/New_York")

    # ===================================================================
    # PREMARKET FUNNEL JOBS (earliest — before market open)
    # These are optional discovery/qualification jobs that run Mon-Fri
    # before the market session. They do NOT delay position monitoring,
    # stop enforcement, or active-position lifecycle jobs.
    # ===================================================================

    # Funnel discovery: 06:00 ET, Mon-Fri
    scheduler.add_job(
        run_funnel_discovery_job,
        CronTrigger(day_of_week="mon-fri", hour=6, minute=0, timezone="America/New_York"),
        id="funnel_discovery",
        max_instances=1,
        coalesce=True,
    )

    # Funnel research qualification: 06:30 ET, Mon-Fri
    scheduler.add_job(
        run_funnel_research_job,
        CronTrigger(day_of_week="mon-fri", hour=6, minute=30, timezone="America/New_York"),
        id="funnel_research",
        max_instances=1,
        coalesce=True,
    )

    # Funnel analyst setup classification: 07:15 ET, Mon-Fri
    scheduler.add_job(
        run_funnel_analysis_job,
        CronTrigger(day_of_week="mon-fri", hour=7, minute=15, timezone="America/New_York"),
        id="funnel_analysis",
        max_instances=1,
        coalesce=True,
    )

    # Funnel opening confirmation: 09:35 ET, Mon-Fri
    scheduler.add_job(
        run_funnel_confirmation_job,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=35, timezone="America/New_York"),
        id="funnel_confirmation",
        max_instances=1,
        coalesce=True,
    )

    # ===================================================================
    # POSITION MONITORING & SAFETY-CRITICAL JOBS (highest priority)
    # These are registered before optional funnel confirmation/intraday
    # jobs per requirement 7.4 — they must not be delayed by funnel work.
    # ===================================================================

    # Pre-market: 8:30 AM ET, Mon-Fri
    scheduler.add_job(
        run_pre_market,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=30, timezone="America/New_York"),
        id="pre_market",
    )

    # ===================================================================
    # ANALYST + PM JOBS: coordinated cycle vs. legacy independent jobs
    # When PM_CYCLE_COORDINATOR_MODE == "enabled", a single coordinated
    # market cycle replaces the separate analyst_refresh + intraday jobs.
    # ===================================================================
    from utils.gate_config import PM_CYCLE_COORDINATOR_MODE

    if PM_CYCLE_COORDINATOR_MODE == "enabled":
        # Coordinated cycle: morning every 15 min (9:00–11:45 ET)
        scheduler.add_job(
            run_coordinated_market_cycle,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-11",
                minute=f"*/{LOOP_INTERVAL}",
                timezone="America/New_York",
            ),
            id="coordinated_cycle_morning",
        )

        # Coordinated cycle: afternoon every 30 min (12:00–15:30 ET)
        scheduler.add_job(
            run_coordinated_market_cycle,
            CronTrigger(
                day_of_week="mon-fri",
                hour="12-15",
                minute="0,30",
                timezone="America/New_York",
            ),
            id="coordinated_cycle_afternoon",
        )
    else:
        # Legacy: independent analyst and PM jobs
        # Analyst refresh: morning every 15 min (9:00–11:45 ET)
        scheduler.add_job(
            run_analyst_refresh,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-11",
                minute=f"*/{LOOP_INTERVAL}",
                timezone="America/New_York",
            ),
            id="analyst_refresh_morning",
        )

        # Analyst refresh: afternoon every 30 min (12:00–15:30 ET)
        scheduler.add_job(
            run_analyst_refresh,
            CronTrigger(
                day_of_week="mon-fri",
                hour="12-15",
                minute="0,30",
                timezone="America/New_York",
            ),
            id="analyst_refresh_afternoon",
        )

        # Intraday morning (PM decisions): every 15 min, 9:30 AM – 12:00 PM ET
        scheduler.add_job(
            run_intraday,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-11",
                minute=f"*/{LOOP_INTERVAL}",
                timezone="America/New_York",
            ),
            id="intraday_morning",
        )

        # Intraday afternoon: every 30 min, 12:00 PM – 4:00 PM ET
        scheduler.add_job(
            run_intraday,
            CronTrigger(
                day_of_week="mon-fri",
                hour="12-15",
                minute="0,30",
                timezone="America/New_York",
            ),
            id="intraday_afternoon",
        )

    # Price monitor: every 60 seconds during market hours only (uses yfinance, free).
    # Offset from :00 to avoid colliding with analyst/PM batch jobs and quote-provider bursts.
    from apscheduler.triggers.cron import CronTrigger as CT
    scheduler.add_job(
        run_price_monitor,
        CT(day_of_week="mon-fri", hour="9-15", second="50", timezone="America/New_York"),
        id="price_monitor",
        max_instances=1,
        coalesce=True,
    )

    # Alert dispatcher: every N seconds during market hours (configurable)
    if PM_ALERT_DISPATCH_MODE != "disabled":
        from apscheduler.triggers.interval import IntervalTrigger

        def run_alert_dispatcher():
            """Wrapper to invoke the alert dispatcher with market-hours guard."""
            from pytz import timezone as pytz_tz
            et_now = datetime.now(pytz_tz("America/New_York"))
            # Only run during market hours (Mon-Fri, 09:30-16:00 ET)
            if et_now.weekday() > 4:
                return
            market_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
            if not (market_open <= et_now <= market_close):
                return
            _alert_dispatcher.evaluate_and_dispatch()

        scheduler.add_job(
            run_alert_dispatcher,
            IntervalTrigger(seconds=PM_ALERT_DISPATCHER_INTERVAL_SECONDS),
            id="alert_dispatcher",
            max_instances=1,
            coalesce=True,
        )

    # Lower-priority jobs: offset by 5 min when coordinator is active
    # to avoid contention with the coordinated cycle's decision window.
    # Requirements: 7.4, 9.4
    _lp_offset = 5 if PM_CYCLE_COORDINATOR_MODE == "enabled" else 0

    # Sector Scout confirmation scan: 10:00 AM ET — NOW bounded shortlist retry
    # (Replaces former broad sector scan per requirement 12.5)
    scheduler.add_job(
        run_funnel_confirmation_retry_job,
        CronTrigger(day_of_week="mon-fri", hour=10, minute=_lp_offset, timezone="America/New_York"),
        id="funnel_confirmation_retry",
        max_instances=1,
        coalesce=True,
    )

    # Sector Scout midday scan: DISABLED in v1 (requirement 8.1, 12.4)
    # No broad multi-sector discovery during market hours.
    # scheduler.add_job(run_sector_scout_midday, ...) — NOT REGISTERED

    # News monitor: every 2 hours during market hours (local LLM, free)
    scheduler.add_job(
        run_news_monitor,
        CronTrigger(day_of_week="mon-fri", hour="10,12,14", minute=_lp_offset, timezone="America/New_York"),
        id="news_monitor",
    )

    # Position health check: every hour during market hours (local LLM, free)
    scheduler.add_job(
        run_position_health,
        CronTrigger(day_of_week="mon-fri", hour="10-15", minute=30 + _lp_offset, timezone="America/New_York"),
        id="position_health",
    )

    # Position timer: every 5 minutes during market hours (no LLM, pure math)
    scheduler.add_job(
        run_position_timer,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/5", timezone="America/New_York"),
        id="position_timer",
        max_instances=1,
        coalesce=True,
    )

    # Price-spike news check: every 15 min during market hours
    scheduler.add_job(
        run_price_spike_news_check,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/15", timezone="America/New_York"),
        id="price_spike_news",
        max_instances=1,
        coalesce=True,
    )

    # Shadow outcomes: score blocked candidates after 15/30/60 minute windows mature
    scheduler.add_job(
        run_shadow_outcomes,
        CronTrigger(day_of_week="mon-fri", hour="9-16", minute="*/5", timezone="America/New_York"),
        id="shadow_outcomes",
        max_instances=1,
        coalesce=True,
    )

    # Position-based news poll: every 30 min during market hours
    scheduler.add_job(
        run_position_news_poll,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="0,30", timezone="America/New_York"),
        id="position_news_poll",
        max_instances=1,
        coalesce=True,
    )

    # Reviewer queue: every 15 min during market hours, process pending reviews
    def run_reviewer_queue():
        if _skip_closed_market_job("reviewer_queue"):
            return
        engine = get_engine()
        try:
            review = reviewer.run(engine, min_unreviewed=1)
            msg = review.get("batch_feedback", review.get("message", ""))
            if msg:
                log.info(f"Reviewer queue: {str(msg)[:100]}")

            # Check for stale pending reviews (>24 hours)
            from db.schema import get_session, ReviewQueue
            from datetime import timedelta as td24
            db = get_session(engine)
            stale_cutoff = datetime.utcnow() - td24(hours=24)
            stale = db.query(ReviewQueue).filter_by(status="pending").filter(
                ReviewQueue.queued_at < stale_cutoff
            ).count()
            if stale > 0:
                log.critical(f"🚨 STALE REVIEWS: {stale} trades pending review for >24 hours!")
            db.close()
        except Exception as e:
            log.error(f"Reviewer queue error: {e}", exc_info=True)

    scheduler.add_job(
        run_reviewer_queue,
        CronTrigger(day_of_week="mon-fri", hour="10-16", minute="*/15", timezone="America/New_York"),
        id="reviewer_queue",
        max_instances=1,
        coalesce=True,
    )

    # Post-market: 4:15 PM ET
    scheduler.add_job(
        run_post_market,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=15, timezone="America/New_York"),
        id="post_market",
    )

    # Daily Review: 4:30 PM ET (after Reviewer + Bookkeeper at 4:15)
    scheduler.add_job(
        run_daily_review,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone="America/New_York"),
        id="daily_review",
    )

    # CEO daily memo remains available via the manual `ceo-daily` command.
    # The scheduled CEO memo is weekly-only to avoid duplicating the afternoon report.

    # CEO weekly strategy memo: Friday after close and daily review.
    scheduler.add_job(
        run_ceo_weekly,
        CronTrigger(day_of_week="fri", hour=16, minute=50, timezone="America/New_York"),
        id="ceo_weekly",
        max_instances=1,
        coalesce=True,
    )

    # Sunday weekly prep: 5:00 PM ET
    scheduler.add_job(
        run_weekly_prep,
        CronTrigger(day_of_week="sun", hour=17, minute=0, timezone="America/New_York"),
        id="weekly_prep",
    )

    # --- Narrator cron jobs ---

    # Hourly recaps: 10 AM, 11 AM, 12 PM ET, Mon-Fri
    scheduler.add_job(
        lambda: run_narrator("hourly_recap"),
        CronTrigger(day_of_week="mon-fri", hour="10,11,12", minute=0,
                    timezone="America/New_York"),
        id="narrator_hourly",
    )

    # Afternoon recap: 2 PM ET, Mon-Fri
    scheduler.add_job(
        lambda: run_narrator("afternoon_recap"),
        CronTrigger(day_of_week="mon-fri", hour=14, minute=0,
                    timezone="America/New_York"),
        id="narrator_afternoon",
    )

    # Daily wrap: 4:15 PM ET, Mon-Thu
    scheduler.add_job(
        lambda: run_narrator("daily_wrap"),
        CronTrigger(day_of_week="mon-thu", hour=16, minute=15,
                    timezone="America/New_York"),
        id="narrator_daily_wrap",
    )

    # Weekly wrap: 4:15 PM ET, Friday
    scheduler.add_job(
        lambda: run_narrator("weekly_wrap"),
        CronTrigger(day_of_week="fri", hour=16, minute=15,
                    timezone="America/New_York"),
        id="narrator_weekly_wrap",
    )

    # Sunday prep: 5:15 PM ET, Sunday
    scheduler.add_job(
        lambda: run_narrator("sunday_prep"),
        CronTrigger(day_of_week="sun", hour=17, minute=15,
                    timezone="America/New_York"),
        id="narrator_sunday_prep",
    )

    console.print(f"[bold green]🚀 Paper Trader started[/bold green]")
    console.print(f"   Watchlist: {', '.join(WATCHLIST)}")
    if PM_TRADABLE_SYMBOLS:
        console.print(f"   PM tradable symbols: {', '.join(PM_TRADABLE_SYMBOLS)}")
    console.print(f"   Loop interval: {LOOP_INTERVAL} min")
    console.print(f"   Schedule: Sun 5PM weekly prep | 8:30 pre-market | 9:30-12 every {LOOP_INTERVAL}min | 12-4 every 30min | 4:15 EOD | 4:30 daily review")
    console.print(f"   Analyst: every {LOOP_INTERVAL}min morning, every 30min afternoon")

    def _shutdown(signum, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        scheduler.shutdown(wait=False)

    def _refresh(signum, frame):
        log.info("SIGUSR1 received — triggering manual refresh")
        console.print("[bold cyan]⚡ Manual refresh triggered...[/bold cyan]")
        scheduler.add_job(run_pre_market, id="manual_refresh", replace_existing=True)

    def _intraday_now(signum, frame):
        from utils.gate_config import PM_CYCLE_COORDINATOR_MODE

        if PM_CYCLE_COORDINATOR_MODE == "enabled":
            log.info("SIGUSR2 received — triggering manual coordinated market cycle")
            console.print("[bold cyan]⚡ Manual coordinated market cycle triggered...[/bold cyan]")
            scheduler.add_job(
                _run_manual_coordinated_cycle, id="manual_coordinated_cycle", replace_existing=True
            )
        else:
            log.info("SIGUSR2 received — triggering manual intraday cycle")
            console.print("[bold cyan]⚡ Manual intraday cycle triggered...[/bold cyan]")
            scheduler.add_job(run_intraday, id="manual_intraday", replace_existing=True)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _refresh)
        signal.signal(signal.SIGUSR2, _intraday_now)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        console.print("\n[yellow]Shutting down...[/yellow]")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        run_once()
    elif len(sys.argv) > 1 and sys.argv[1] == "weekly":
        engine = get_engine()
        ensure_initial_balance(engine)
        run_weekly_prep()
    elif len(sys.argv) > 1 and sys.argv[1] == "ceo-daily":
        engine = get_engine()
        ensure_initial_balance(engine)
        check_schema(engine)
        run_ceo_daily()
    elif len(sys.argv) > 1 and sys.argv[1] == "ceo-weekly":
        engine = get_engine()
        ensure_initial_balance(engine)
        check_schema(engine)
        run_ceo_weekly()
    elif len(sys.argv) > 2 and sys.argv[1] == "manual" and sys.argv[2] == "market-cycle":
        # Always runs coordinated path regardless of PM_CYCLE_COORDINATOR_MODE flag
        engine = get_engine()
        ensure_initial_balance(engine)
        check_schema(engine)
        ensure_shadow_ledger_schema(engine)
        console.print("[bold]Running manual coordinated market cycle...[/bold]")
        _run_manual_coordinated_cycle()
    elif len(sys.argv) > 2 and sys.argv[1] == "manual" and sys.argv[2] == "pm":
        # Always calls run_intraday() directly (legacy path) regardless of flag
        engine = get_engine()
        ensure_initial_balance(engine)
        check_schema(engine)
        ensure_shadow_ledger_schema(engine)
        console.print("[bold]Running manual PM (legacy intraday)...[/bold]")
        run_intraday()
    elif len(sys.argv) > 2 and sys.argv[1] == "manual" and sys.argv[2] == "analyst":
        # Always calls run_analyst_refresh() directly (bypass coordinator)
        engine = get_engine()
        ensure_initial_balance(engine)
        check_schema(engine)
        ensure_shadow_ledger_schema(engine)
        console.print("[bold]Running manual analyst refresh...[/bold]")
        run_analyst_refresh()
    else:
        main()
