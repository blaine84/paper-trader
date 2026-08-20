"""Missed-Move Detector — identify when price has crossed a watch's target.

Detects watches where the profit target has already been reached before the
system could execute. Transitions such watches to MISSED state via CAS.
This is a SAFETY feature — fail-closed on logic, tolerant of CAS contention.
A watch whose target is already crossed must never produce a candidate.

Requirements: 4.1-4.9, 7.1-7.8
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, Context, ROUND_HALF_UP

from utils.setup_watch_evaluator import _safe_decimal
from utils.setup_watch_registry import (
    SetupWatch,
    SetupWatchRegistry,
    SetupWatchRegistryError,
    WatchState,
)

logger = logging.getLogger(__name__)

_CTX = Context(prec=28, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class MissedMoveResult:
    """Result of missed-move check for a single watch."""

    watch_id: str
    missed: bool
    target_price: Decimal | None
    current_price: Decimal | None
    side: str
    reason: str | None


def _parse_target_from_geometry(draft_geometry_json: str | None) -> Decimal | None:
    """Parse the target price from draft_geometry_json.

    Returns Decimal on success, None on any failure (NULL, parse error, missing key).
    """
    if draft_geometry_json is None:
        return None
    try:
        geom = json.loads(draft_geometry_json)
    except (json.JSONDecodeError, TypeError):
        return None
    raw_target = geom.get("target")
    if raw_target is None:
        return None
    return _safe_decimal(raw_target)


def _is_target_crossed(current_price: Decimal, target: Decimal, side: str) -> bool:
    """Determine if the target has been crossed based on side.

    BUY: missed iff current_price >= target
    SHORT: missed iff current_price <= target

    All comparisons use Decimal with Context(prec=28, rounding=ROUND_HALF_UP).
    """
    if side.upper() == "BUY":
        return _CTX.compare(current_price, target) >= 0
    else:
        # SHORT
        return _CTX.compare(current_price, target) <= 0


def check_missed_move(
    watch: SetupWatch,
    current_price_raw: float | str | Decimal,
) -> MissedMoveResult:
    """Check if price has crossed a watch's draft geometry target.

    Only evaluates watches in READY or PROMOTED state. For other states,
    returns MissedMoveResult(missed=False).

    Returns MissedMoveResult(missed=False) if:
      - Watch is not in READY or PROMOTED state
      - No draft_geometry_json (NULL)
      - draft_geometry_json fails JSON parsing
      - Parsed JSON has no "target" key
      - Target value cannot be parsed to Decimal

    This is the BRIDGE path — fail-open on parse errors (skip check).

    Parameters
    ----------
    watch : SetupWatch
        The watch to check.
    current_price_raw : float | str | Decimal
        Current market price for comparison.

    Returns
    -------
    MissedMoveResult
    """
    side = watch.side.upper() if watch.side else "BUY"

    # Only evaluate READY or PROMOTED state
    if watch.state not in (WatchState.READY, WatchState.PROMOTED):
        return MissedMoveResult(
            watch_id=watch.watch_id,
            missed=False,
            target_price=None,
            current_price=None,
            side=side,
            reason=None,
        )

    # Parse current price
    current_price = _safe_decimal(current_price_raw)
    if current_price is None:
        return MissedMoveResult(
            watch_id=watch.watch_id,
            missed=False,
            target_price=None,
            current_price=None,
            side=side,
            reason=None,
        )

    # Parse target from draft geometry — fail-open (bridge path)
    target = _parse_target_from_geometry(watch.draft_geometry_json)
    if target is None:
        return MissedMoveResult(
            watch_id=watch.watch_id,
            missed=False,
            target_price=None,
            current_price=current_price,
            side=side,
            reason=None,
        )

    # Directional comparison using Decimal arithmetic
    missed = _is_target_crossed(current_price, target, side)

    return MissedMoveResult(
        watch_id=watch.watch_id,
        missed=missed,
        target_price=target,
        current_price=current_price,
        side=side,
        reason="target_already_crossed" if missed else None,
    )


def apply_missed_move_transition(
    registry: SetupWatchRegistry,
    watch: SetupWatch,
    result: MissedMoveResult,
) -> bool:
    """Transition a watch to MISSED state via CAS if missed-move detected.

    Fail-closed on logic (a missed watch must not proceed) but tolerant
    of CAS contention (SetupWatchRegistryError on rowcount == 0 means
    another transition won the race — log WARNING and continue).

    Records event data: current_price, target_price, side, timestamp.

    Parameters
    ----------
    registry : SetupWatchRegistry
    watch : SetupWatch
    result : MissedMoveResult

    Returns
    -------
    bool
        True if watch was transitioned successfully.
        False if CAS failed (watch already in terminal state).
    """
    if not result.missed:
        return False

    try:
        registry.transition_state(
            watch.watch_id,
            watch.state,
            WatchState.MISSED,
            terminal_reason="target_already_crossed",
        )

        # Req 8.3: WARNING log with symbol, side, target, current price,
        # and elapsed minutes since ready_at
        elapsed_minutes = None
        if watch.ready_at is not None:
            elapsed_td = datetime.now(timezone.utc) - watch.ready_at
            elapsed_minutes = int(elapsed_td.total_seconds() / 60)

        logger.warning(
            "Missed move detected: symbol=%s, side=%s, target=%s, "
            "current=%s, elapsed_minutes_since_ready=%s",
            watch.symbol,
            result.side,
            result.target_price,
            result.current_price,
            elapsed_minutes if elapsed_minutes is not None else "N/A",
        )
        return True
    except SetupWatchRegistryError:
        # CAS failure — watch already moved to another terminal state
        logger.warning(
            "CAS transition to MISSED failed for watch %s "
            "(already terminal or state changed concurrently): "
            "side=%s, target=%s, current=%s",
            watch.watch_id,
            result.side,
            result.target_price,
            result.current_price,
        )
        return False


def check_target_crossed_for_pending_order(
    watch: SetupWatch,
    fresh_price_raw: float | str | Decimal,
) -> MissedMoveResult:
    """Check if fresh price crosses target for the pending-order guard.

    Same directional logic as check_missed_move but called at the
    pending-order creation boundary. Uses Decimal with Context(prec=28).

    Geometry handling (fail-CLOSED for corrupted data):
      - draft_geometry_json IS NULL → MissedMoveResult(missed=False)
        (no geometry to compare against, allow order)
      - draft_geometry_json IS NOT NULL but fails JSON parse or has
        malformed/missing target → MissedMoveResult(missed=True)
        with reason="malformed_geometry" (fail-closed on corrupted watch data)
      - draft_geometry_json valid with parseable target → normal directional check

    Parameters
    ----------
    watch : SetupWatch
        The watch originating the pending order.
    fresh_price_raw : float | str | Decimal
        Fresh market price for comparison.

    Returns
    -------
    MissedMoveResult

    Requirements: 7.1-7.8
    """
    side = watch.side.upper() if watch.side else "BUY"

    # Parse fresh price
    fresh_price = _safe_decimal(fresh_price_raw)
    if fresh_price is None:
        # Cannot evaluate — fail-closed with reason
        return MissedMoveResult(
            watch_id=watch.watch_id,
            missed=True,
            target_price=None,
            current_price=None,
            side=side,
            reason="malformed_geometry",
        )

    # Case 1: NULL geometry → skip check, allow order
    if watch.draft_geometry_json is None:
        return MissedMoveResult(
            watch_id=watch.watch_id,
            missed=False,
            target_price=None,
            current_price=fresh_price,
            side=side,
            reason=None,
        )

    # Case 2/3: Present geometry — try to parse
    try:
        geom = json.loads(watch.draft_geometry_json)
    except (json.JSONDecodeError, TypeError):
        # Present but malformed JSON → fail-closed
        return MissedMoveResult(
            watch_id=watch.watch_id,
            missed=True,
            target_price=None,
            current_price=fresh_price,
            side=side,
            reason="malformed_geometry",
        )

    # Check for target key
    raw_target = geom.get("target")
    if raw_target is None:
        # Present JSON but no target key → fail-closed (malformed)
        return MissedMoveResult(
            watch_id=watch.watch_id,
            missed=True,
            target_price=None,
            current_price=fresh_price,
            side=side,
            reason="malformed_geometry",
        )

    # Parse target value
    target = _safe_decimal(raw_target)
    if target is None:
        # Target present but unparseable → fail-closed (malformed)
        return MissedMoveResult(
            watch_id=watch.watch_id,
            missed=True,
            target_price=None,
            current_price=fresh_price,
            side=side,
            reason="malformed_geometry",
        )

    # Case 3: Valid geometry — normal directional check
    missed = _is_target_crossed(fresh_price, target, side)

    return MissedMoveResult(
        watch_id=watch.watch_id,
        missed=missed,
        target_price=target,
        current_price=fresh_price,
        side=side,
        reason="target_already_crossed" if missed else None,
    )
