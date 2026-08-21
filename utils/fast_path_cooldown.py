"""Fast-path cooldown checks.

Determines whether a new fast-path outcome for a given symbol should be
blocked based on recent activity.  Cooldown semantics differ from the
existing PM-cycle cooldowns in one key respect: *missed moves and watches
do not suppress subsequent alerts* for the same symbol.

Blocking rules (by prior outcome):
    - ``trade_executed`` in the last ``TRADE_COOLDOWN_MINUTES`` → block.
    - ``pending_order_created`` with an active order still resting → block.
    - ``missed_move`` → does NOT block.
    - ``stand_down`` → does NOT block (unless churn protection triggers).
    - ``watch_created`` / ``watch_promoted`` → does NOT block.

Churn protection:
    3+ ``stand_down`` for the same symbol + setup_type within
    ``CHURN_WINDOW_MINUTES`` → block.

Fail-mode:
    - **Execution path** (trade or pending order): fail-closed.  If the DB
      query raises, return a ``CooldownBlock`` so the evaluator produces
      ``stand_down`` — never allow a trade through unverifiable cooldown.
    - **Non-execution path** (watch, annotation, narration): fail-open.
      Log the error and return ``None``, allowing the benign outcome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from utils.fast_path_config import CHURN_MAX_STANDDOWNS, CHURN_WINDOW_MINUTES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How long after a trade_executed event the same symbol is blocked.
TRADE_COOLDOWN_MINUTES: int = 15


# ---------------------------------------------------------------------------
# CooldownBlock — frozen value object returned when a cooldown blocks.
# (Task 7.2 — Requirements 6.4, 6.6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CooldownBlock:
    """Indicates that a cooldown rule is blocking the proposed outcome.

    Attributes:
        reason_code: Machine-readable code identifying the blocking reason.
        message: Human-readable explanation suitable for audit logs.
        blocking_outcome_type: The outcome_type of the event that caused
            the block (e.g. ``"trade_executed"``), or ``None`` if the block
            is from a structural check (e.g. churn protection).
        blocking_event_id: The ``event_id`` of the specific event that
            caused the block, or ``None`` when not attributable to a
            single event.
    """

    reason_code: str
    message: str
    blocking_outcome_type: str | None = None
    blocking_event_id: str | None = None


# ---------------------------------------------------------------------------
# Main cooldown check
# (Task 7.1 — Requirements 6.1–6.8)
# ---------------------------------------------------------------------------


def check_fast_path_cooldown(
    symbol: str,
    setup_type: str,
    profile_id: str,
    db,
    *,
    execution_path: bool = True,
) -> CooldownBlock | None:
    """Check whether a fast-path outcome for *symbol* should be blocked.

    Parameters
    ----------
    symbol:
        Ticker symbol being evaluated.
    setup_type:
        The setup type of the trigger (e.g. ``"momentum_fade"``).
    profile_id:
        Active trading profile id.
    db:
        SQLAlchemy engine (used via ``db.connect()``).
    execution_path:
        If ``True``, DB errors produce a ``CooldownBlock`` (fail-closed).
        If ``False``, DB errors log a warning and return ``None`` (fail-open).

    Returns
    -------
    CooldownBlock | None
        A ``CooldownBlock`` when the outcome should be suppressed, otherwise
        ``None`` (no cooldown active).
    """
    try:
        return _check_cooldown_inner(symbol, setup_type, profile_id, db)
    except Exception as exc:  # noqa: BLE001
        if execution_path:
            logger.error(
                "Cooldown check failed for %s (execution_path=True), "
                "returning fail-closed block: %s",
                symbol,
                exc,
            )
            return CooldownBlock(
                reason_code="cooldown_check_failed",
                message=f"Cooldown DB query failed for {symbol}: {exc}",
                blocking_outcome_type=None,
                blocking_event_id=None,
            )
        else:
            logger.warning(
                "Cooldown check failed for %s (execution_path=False), "
                "failing open: %s",
                symbol,
                exc,
            )
            return None


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _check_cooldown_inner(
    symbol: str,
    setup_type: str,
    profile_id: str,
    db,
) -> CooldownBlock | None:
    """Core cooldown logic — may raise on DB errors."""

    now = datetime.now(timezone.utc)
    cooldown_cutoff = (now - timedelta(minutes=TRADE_COOLDOWN_MINUTES)).isoformat()
    churn_cutoff = (now - timedelta(minutes=CHURN_WINDOW_MINUTES)).isoformat()

    with db.connect() as conn:
        # 1. Check for recent trade_executed events within cooldown window.
        result = conn.execute(
            text(
                "SELECT event_id, outcome_type FROM fast_path_events "
                "WHERE symbol = :symbol "
                "AND profile_id = :profile_id "
                "AND outcome_type = 'trade_executed' "
                "AND evaluated_at >= :cutoff "
                "ORDER BY evaluated_at DESC "
                "LIMIT 1"
            ),
            {"symbol": symbol, "profile_id": profile_id, "cutoff": cooldown_cutoff},
        )
        row = result.mappings().first()
        if row:
            return CooldownBlock(
                reason_code="recent_trade",
                message=(
                    f"{symbol} had a trade_executed event within the last "
                    f"{TRADE_COOLDOWN_MINUTES} minutes"
                ),
                blocking_outcome_type="trade_executed",
                blocking_event_id=row["event_id"],
            )

        # 1b. Also check the trades table for recent entries by this profile.
        result = conn.execute(
            text(
                "SELECT id FROM trades "
                "WHERE symbol = :symbol "
                "AND profile = :profile_id "
                "AND entry_time >= :cutoff "
                "ORDER BY entry_time DESC "
                "LIMIT 1"
            ),
            {"symbol": symbol, "profile_id": profile_id, "cutoff": cooldown_cutoff},
        )
        row = result.mappings().first()
        if row:
            return CooldownBlock(
                reason_code="recent_trade",
                message=(
                    f"{symbol} has a trade opened within the last "
                    f"{TRADE_COOLDOWN_MINUTES} minutes"
                ),
                blocking_outcome_type="trade_executed",
                blocking_event_id=None,
            )

        # 2. Check for pending_order_created with an active pending order.
        result = conn.execute(
            text(
                "SELECT order_id FROM pending_orders "
                "WHERE symbol = :symbol "
                "AND profile_id = :profile_id "
                "AND state IN ('pending', 'filling') "
                "LIMIT 1"
            ),
            {"symbol": symbol, "profile_id": profile_id},
        )
        row = result.mappings().first()
        if row:
            return CooldownBlock(
                reason_code="active_pending_order",
                message=(
                    f"{symbol} has an active pending order "
                    f"(order_id={row['order_id']})"
                ),
                blocking_outcome_type="pending_order_created",
                blocking_event_id=None,
            )

        # 3. Churn protection: 3+ stand_down for same symbol+setup_type
        #    within the churn window.
        result = conn.execute(
            text(
                "SELECT COUNT(*) AS cnt FROM fast_path_events "
                "WHERE symbol = :symbol "
                "AND profile_id = :profile_id "
                "AND setup_type = :setup_type "
                "AND outcome_type = 'stand_down' "
                "AND evaluated_at >= :cutoff"
            ),
            {
                "symbol": symbol,
                "profile_id": profile_id,
                "setup_type": setup_type,
                "cutoff": churn_cutoff,
            },
        )
        row = result.mappings().first()
        if row and row["cnt"] >= CHURN_MAX_STANDDOWNS:
            return CooldownBlock(
                reason_code="churn_protection",
                message=(
                    f"{symbol}/{setup_type} has {row['cnt']} stand_down events "
                    f"in the last {CHURN_WINDOW_MINUTES} minutes "
                    f"(threshold: {CHURN_MAX_STANDDOWNS})"
                ),
                blocking_outcome_type="stand_down",
                blocking_event_id=None,
            )

    # No cooldown triggered.
    return None
