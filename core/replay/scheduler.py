"""
Scheduling and market-hour constraints for the Decision Replay Agent.

Enforces that scheduled replay runs execute only outside regular market hours
(before 9:30 AM ET or after 4:00 PM ET on trading days). Provides:

- Market-hour detection
- Checkpoint-and-suspend: if replay is running at 9:30 AM ET, save progress and suspend
- Default batch window: candidates reaching terminal state since previous
  successful run's batch-start timestamp
- Idempotent overlap window: 60-second re-processing before the captured
  batch-start timestamp to avoid edge-case misses
- Batch checkpoint at candidate granularity (>50 candidates)

Requirements: 10.1, 10.2, 10.3, 10.6
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import text

from utils.position_lifecycle_governance import is_trading_day

log = logging.getLogger(__name__)

# US Eastern timezone for market-hour calculations
_ET = ZoneInfo("America/New_York")

# Market hours boundaries
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Overlap window duration in seconds (Requirement 10.3)
OVERLAP_WINDOW_SECONDS = 60

# Batch checkpoint threshold (Requirement 10.6)
BATCH_CHECKPOINT_THRESHOLD = 50


def is_market_hours(now_utc: datetime | None = None) -> bool:
    """Return True if the current time is during regular US market hours.

    Market hours are 9:30 AM – 4:00 PM ET on trading days (weekdays excluding
    market holidays).

    Args:
        now_utc: Optional UTC datetime for testing. If None, uses current time.

    Returns:
        True if currently within market hours, False otherwise.

    Requirement 10.1: Scheduled replay SHALL run only outside regular market
    hours (before 9:30 AM ET or after 4:00 PM ET on trading days).
    """
    if now_utc is None:
        now_et = datetime.now(_ET)
    else:
        now_et = now_utc.astimezone(_ET)

    # Not a trading day → not market hours
    if not is_trading_day(now_et):
        return False

    # Check time boundaries: 9:30 AM ≤ now < 4:00 PM
    market_open = now_et.replace(
        hour=MARKET_OPEN_HOUR,
        minute=MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0,
    )
    market_close = now_et.replace(
        hour=MARKET_CLOSE_HOUR,
        minute=MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0,
    )

    return market_open <= now_et < market_close


def should_suspend(now_utc: datetime | None = None) -> bool:
    """Return True if a running replay should checkpoint and suspend.

    A replay should suspend when market hours are active or imminent. This
    checks if the current time has reached or passed 9:30 AM ET on a trading
    day (i.e., market is open).

    Requirement 10.2: IF a replay job is still running when regular market
    hours begin (9:30 AM ET), THEN checkpoint progress and suspend.

    Args:
        now_utc: Optional UTC datetime for testing. If None, uses current time.

    Returns:
        True if replay should suspend (market hours active), False otherwise.
    """
    return is_market_hours(now_utc)


def get_default_batch_window(
    session,
    now_utc: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Compute the default batch window for a scheduled replay run.

    The default batch evaluates candidates that reached a terminal state
    (consumed, rejected, not_selected, or expired) since the PREVIOUS
    successful run's batch-start timestamp.

    An idempotent overlap window (60 seconds before the previous batch-start)
    is applied to avoid missing candidates that completed while the prior run
    was executing.

    Requirement 10.3: Default scheduled run SHALL evaluate candidates that
    reached terminal state since the previous successful run's batch-start
    timestamp with an idempotent overlap window.

    Args:
        session: SQLAlchemy session for querying replay_batch_runs.
        now_utc: Optional UTC datetime for the batch-end boundary.
                 If None, uses current time.

    Returns:
        Tuple of (window_start, window_end) datetimes (UTC).
        - window_start: previous batch-start minus 60s overlap
        - window_end: current time (batch-start for this run)
    """
    if now_utc is None:
        now_utc = datetime.utcnow()

    # Query the previous successful batch run's started_at (batch-start timestamp)
    previous_batch_start = _get_previous_successful_batch_start(session)

    if previous_batch_start is None:
        # No previous successful run — use a wide default window (7 days back)
        window_start = now_utc - timedelta(days=7)
        log.info(
            "No previous successful batch run found. "
            "Using default 7-day lookback: %s",
            window_start.isoformat(),
        )
    else:
        # Apply the 60-second overlap window before previous batch-start
        window_start = get_overlap_window(previous_batch_start)
        log.info(
            "Previous batch-start: %s, overlap window start: %s",
            previous_batch_start.isoformat(),
            window_start.isoformat(),
        )

    return (window_start, now_utc)


def get_overlap_window(previous_batch_start: datetime) -> datetime:
    """Compute the overlap window start time.

    Returns the timestamp 60 seconds before the previous batch-start.
    This ensures candidates whose terminal timestamp falls within that
    60-second window are re-processed (idempotent), avoiding edge-case
    misses from candidates that completed while the prior run was executing.

    Requirement 10.3: Idempotent overlap window — re-process candidates
    whose terminal timestamp falls within 60 seconds before the captured
    batch-start timestamp.

    Args:
        previous_batch_start: The started_at timestamp of the previous
                              successful batch run.

    Returns:
        Overlap start datetime (previous_batch_start minus 60 seconds).
    """
    return previous_batch_start - timedelta(seconds=OVERLAP_WINDOW_SECONDS)


def should_checkpoint_batch(candidates_total: int) -> bool:
    """Return True if the batch should use candidate-level checkpointing.

    Requirement 10.6: When a replay batch exceeds 50 candidates, checkpoint
    progress at the granularity of individual candidates so that a restart
    resumes from the last successfully completed candidate.

    The BatchManager already processes items individually — this function
    indicates whether the batch is large enough to require explicit
    checkpoint-and-suspend awareness.

    Args:
        candidates_total: Total number of candidates in the batch.

    Returns:
        True if batch exceeds checkpoint threshold (50 candidates).
    """
    return candidates_total > BATCH_CHECKPOINT_THRESHOLD


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_previous_successful_batch_start(session) -> datetime | None:
    """Query the most recent successful batch run's started_at timestamp.

    Looks for the latest batch run with status='completed' and mode='batch'.

    Args:
        session: SQLAlchemy session.

    Returns:
        The started_at datetime of the previous successful run, or None.
    """
    result = session.execute(
        text("""
            SELECT started_at
            FROM replay_batch_runs
            WHERE status = 'completed'
              AND mode = 'batch'
            ORDER BY started_at DESC
            LIMIT 1
        """)
    )
    row = result.fetchone()
    if row is None:
        return None

    started_at = row[0]
    if isinstance(started_at, str):
        return datetime.fromisoformat(started_at)
    return started_at
