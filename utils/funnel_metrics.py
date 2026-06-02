"""Funnel stage quality metrics and daily review reporting.

Generates daily review stats, stage health classifications, and shadow
outcome linkage for premarket funnel candidates. Supports Requirement 10
(Stage Quality Measurement) of the premarket-candidate-funnel spec.

Key behaviors:
- Retains rejected candidates with complete stage_decisions history (10.1)
- Marks candidates with geometry as eligible for shadow-outcome evaluation (10.2)
- Generates daily review stats: shortlist count, per-stage promotion/rejection,
  fallback/timeout, PM handoff, trade outcomes, shadow outcomes, top 3 reasons (10.3)
- Counts each unique (date, symbol) exactly once per metric (10.4)
- Classifies stage health over trailing 5 trading days (10.5)

See: design.md §Correctness Properties, requirements.md §10
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text

from db.schema import FunnelCandidate, FunnelRunLog, get_session

logger = logging.getLogger(__name__)

_NY_TZ = ZoneInfo("America/New_York")

# Stage names in pipeline order
PIPELINE_STAGES = ("discovery", "research", "analysis", "confirmation")

# Stage status prefixes for rejection detection
_REJECTION_STATUSES = {
    "discovery": "rejected_discovery",
    "research": "rejected_research",
    "analysis": "rejected_analysis",
    "confirmation": "rejected_confirmation",
}

# Thresholds for stage health classification (Requirement 10.5)
_OVERLY_PERMISSIVE_THRESHOLD = 0.90
_OVERLY_RESTRICTIVE_THRESHOLD = 0.90
_OPERATIONALLY_FAILING_THRESHOLD = 0.30
_TRAILING_DAYS = 5
_MIN_CANDIDATES_FOR_CLASSIFICATION = 10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StageStats:
    """Per-stage promotion/rejection/timeout counts."""

    stage: str
    candidates_entered: int = 0
    promoted: int = 0
    rejected: int = 0
    needs_confirmation: int = 0
    timed_out: int = 0
    not_evaluated: int = 0


@dataclass
class TradeOutcomes:
    """Trade outcome counts for promoted candidates."""

    total: int = 0
    wins: int = 0
    losses: int = 0
    open_trades: int = 0


@dataclass
class ShadowOutcomes:
    """Shadow outcome counts for filtered candidates."""

    total: int = 0
    target_hit: int = 0
    stop_hit: int = 0
    timed_exit: int = 0
    insufficient_data: int = 0


@dataclass
class StageClassification:
    """Stage health classification over trailing window."""

    stage: str
    classification: str | None = None  # overly_permissive | overly_restrictive | operationally_failing | None (healthy)
    promotion_rate: float = 0.0
    rejection_rate: float = 0.0
    timeout_rate: float = 0.0
    candidates_in_window: int = 0
    days_in_window: int = 0


@dataclass
class DailyReview:
    """Complete daily review report for funnel stage quality."""

    review_date: date
    shortlist_count: int = 0
    stage_stats: dict[str, StageStats] = field(default_factory=dict)
    fallback_count: int = 0
    timeout_count: int = 0
    pm_handoff_count: int = 0
    trade_outcomes: TradeOutcomes = field(default_factory=TradeOutcomes)
    shadow_outcomes: ShadowOutcomes = field(default_factory=ShadowOutcomes)
    top_rejection_reasons: list[tuple[str, int]] = field(default_factory=list)
    stage_classifications: dict[str, StageClassification] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_daily_review(engine, review_date: date | None = None) -> DailyReview:
    """Generate the daily review report for funnel stage quality.

    Counts each unique (date, symbol) exactly once per metric regardless
    of re-evaluations. Reports shortlist count, per-stage stats, fallback/
    timeout counts, PM handoff count, trade outcomes, shadow outcomes, and
    top 3 rejection reasons.

    Args:
        engine: SQLAlchemy engine.
        review_date: The trading date to review. Defaults to today (NY time).

    Returns:
        DailyReview dataclass with all computed metrics.
    """
    if review_date is None:
        review_date = datetime.now(_NY_TZ).date()

    session = get_session(engine)
    try:
        # Query all candidates for this date (unique by date+symbol per schema)
        candidates = (
            session.query(FunnelCandidate)
            .filter(FunnelCandidate.date == review_date)
            .all()
        )

        review = DailyReview(review_date=review_date)

        # Shortlist count = total unique (date, symbol) candidates discovered
        review.shortlist_count = len(candidates)

        # Compute per-stage stats from stage_decisions
        review.stage_stats = _compute_stage_stats(candidates)

        # Fallback/timeout counts from FunnelRunLog
        run_logs = (
            session.query(FunnelRunLog)
            .filter(FunnelRunLog.date == review_date)
            .all()
        )
        review.fallback_count = _count_fallbacks(run_logs)
        review.timeout_count = _count_timeouts(run_logs, candidates)

        # PM handoff count
        review.pm_handoff_count = sum(
            1 for c in candidates
            if c.stage_status in ("pm_eligible", "executed")
        )

        # Trade outcomes for candidates that got executed
        review.trade_outcomes = _compute_trade_outcomes(engine, candidates)

        # Shadow outcomes for filtered candidates
        review.shadow_outcomes = _compute_shadow_outcomes(engine, candidates)

        # Top 3 rejection reasons
        review.top_rejection_reasons = _compute_top_rejection_reasons(candidates, top_n=3)

        # Stage classifications over trailing window
        review.stage_classifications = compute_stage_classifications(
            engine, review_date
        )

        return review

    finally:
        session.close()


def mark_shadow_eligible_candidates(engine, review_date: date | None = None) -> list[int]:
    """Mark rejected candidates with geometry as eligible for shadow-outcome evaluation.

    For each rejected candidate that has non-null entry_price, stop_price, and
    target_price in its Analyst or Scout evidence (stage_decisions), links it
    to a blocked_trade_candidates record using the existing shadow ledger.

    Args:
        engine: SQLAlchemy engine.
        review_date: Date to check. Defaults to today (NY time).

    Returns:
        List of blocked_candidate_ids that were created/linked.
    """
    if review_date is None:
        review_date = datetime.now(_NY_TZ).date()

    session = get_session(engine)
    try:
        # Get rejected candidates not yet linked to shadow ledger
        rejected_candidates = (
            session.query(FunnelCandidate)
            .filter(
                FunnelCandidate.date == review_date,
                FunnelCandidate.stage_status.like("rejected_%"),
                FunnelCandidate.blocked_candidate_id == None,  # noqa: E711
            )
            .all()
        )

        linked_ids: list[int] = []

        for candidate in rejected_candidates:
            geometry = _extract_geometry(candidate)
            if geometry is None:
                continue

            entry_price = geometry.get("entry_price")
            stop_price = geometry.get("stop_price")
            target_price = geometry.get("target_price")

            if entry_price is None or stop_price is None or target_price is None:
                continue

            # Record in shadow ledger
            blocked_id = _record_shadow_candidate(
                engine, candidate, entry_price, stop_price, target_price
            )

            if blocked_id is not None:
                # Link FunnelCandidate to blocked record
                candidate.blocked_candidate_id = blocked_id
                candidate.updated_at = datetime.now(timezone.utc)
                linked_ids.append(blocked_id)

        if linked_ids:
            session.commit()
            logger.info(
                "Linked %d rejected candidates to shadow ledger for date %s",
                len(linked_ids), review_date
            )

        return linked_ids

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def compute_stage_classifications(
    engine, as_of_date: date | None = None
) -> dict[str, StageClassification]:
    """Classify each stage's health over trailing 5 trading days.

    A stage is classified as:
    - overly_permissive: >90% promotion rate
    - overly_restrictive: >90% rejection rate
    - operationally_failing: >30% timeout rate

    Requires minimum 10 candidates entering the stage in the window.

    Args:
        engine: SQLAlchemy engine.
        as_of_date: End date for trailing window. Defaults to today (NY time).

    Returns:
        Dict mapping stage name to StageClassification.
    """
    if as_of_date is None:
        as_of_date = datetime.now(_NY_TZ).date()

    # Compute trailing trading days (approximate — uses calendar days -7 to catch 5 weekdays)
    window_start = as_of_date - timedelta(days=7)

    session = get_session(engine)
    try:
        candidates = (
            session.query(FunnelCandidate)
            .filter(
                FunnelCandidate.date >= window_start,
                FunnelCandidate.date <= as_of_date,
            )
            .all()
        )

        # Group by actual dates present to count trading days
        dates_present = set(c.date for c in candidates)
        # Sort and take up to 5 most recent dates
        sorted_dates = sorted(dates_present, reverse=True)[:_TRAILING_DAYS]

        if not sorted_dates:
            return {
                stage: StageClassification(stage=stage)
                for stage in PIPELINE_STAGES
            }

        # Filter candidates to only those in the trailing window dates
        window_candidates = [c for c in candidates if c.date in set(sorted_dates)]

        classifications: dict[str, StageClassification] = {}
        for stage in PIPELINE_STAGES:
            classifications[stage] = _classify_stage(
                stage, window_candidates, len(sorted_dates)
            )

        return classifications

    finally:
        session.close()


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _compute_stage_stats(candidates: list[FunnelCandidate]) -> dict[str, StageStats]:
    """Compute per-stage promotion/rejection/timeout counts.

    Each candidate is counted exactly once per stage it entered.
    Decisions are read from stage_decisions JSON array.
    """
    stats: dict[str, StageStats] = {
        stage: StageStats(stage=stage) for stage in PIPELINE_STAGES
    }

    for candidate in candidates:
        decisions = _parse_stage_decisions(candidate)

        # Every candidate entered discovery (was discovered by scout)
        stats["discovery"].candidates_entered += 1
        # Scout decision is always "promoted" for persisted candidates
        stats["discovery"].promoted += 1

        # Track which stages this candidate entered based on decisions
        for decision in decisions:
            agent = decision.get("agent", "")
            dec_value = decision.get("decision", "")

            if agent == "researcher":
                stats["research"].candidates_entered += 1
                if dec_value == "promoted":
                    stats["research"].promoted += 1
                elif dec_value == "rejected":
                    stats["research"].rejected += 1
                elif dec_value == "needs_confirmation":
                    stats["research"].needs_confirmation += 1
                elif dec_value == "timed_out":
                    stats["research"].timed_out += 1
                elif dec_value == "not_evaluated":
                    stats["research"].not_evaluated += 1

            elif agent == "analyst":
                stats["analysis"].candidates_entered += 1
                if dec_value == "promoted":
                    stats["analysis"].promoted += 1
                elif dec_value == "rejected":
                    stats["analysis"].rejected += 1
                elif dec_value == "needs_confirmation":
                    stats["analysis"].needs_confirmation += 1
                elif dec_value == "timed_out":
                    stats["analysis"].timed_out += 1
                elif dec_value == "not_evaluated":
                    stats["analysis"].not_evaluated += 1

            elif agent == "confirmation":
                stats["confirmation"].candidates_entered += 1
                if dec_value == "promoted":
                    stats["confirmation"].promoted += 1
                elif dec_value == "rejected":
                    stats["confirmation"].rejected += 1
                elif dec_value == "needs_confirmation":
                    stats["confirmation"].needs_confirmation += 1
                elif dec_value == "timed_out":
                    stats["confirmation"].timed_out += 1
                elif dec_value == "not_evaluated":
                    stats["confirmation"].not_evaluated += 1

    return stats


def _count_fallbacks(run_logs: list[FunnelRunLog]) -> int:
    """Count discovery runs that used deterministic fallback."""
    return sum(
        1 for log_entry in run_logs
        if log_entry.stage == "discovery"
        and log_entry.result_status == "degraded"
    )


def _count_timeouts(
    run_logs: list[FunnelRunLog], candidates: list[FunnelCandidate]
) -> int:
    """Count timeout events from run logs and candidate decisions."""
    log_timeouts = sum(
        1 for log_entry in run_logs
        if log_entry.result_status == "timed_out"
    )

    # Also count individual candidate timeout decisions
    candidate_timeouts = 0
    for candidate in candidates:
        decisions = _parse_stage_decisions(candidate)
        for dec in decisions:
            if dec.get("decision") == "timed_out":
                candidate_timeouts += 1

    return log_timeouts + candidate_timeouts


def _compute_trade_outcomes(
    engine, candidates: list[FunnelCandidate]
) -> TradeOutcomes:
    """Compute trade outcomes for candidates that were executed.

    Links through trade_event_id to the trades table to determine
    win/loss/open status.
    """
    outcomes = TradeOutcomes()

    # Get candidates with trade linkage
    executed = [c for c in candidates if c.trade_event_id is not None]
    if not executed:
        return outcomes

    outcomes.total = len(executed)

    # Query trade outcomes via trade_events → trades
    session = get_session(engine)
    try:
        for candidate in executed:
            try:
                result = session.execute(
                    text(
                        """
                        SELECT t.status, t.pnl
                        FROM trade_events te
                        JOIN trades t ON t.id = te.trade_id
                        WHERE te.id = :event_id
                        LIMIT 1
                        """
                    ),
                    {"event_id": candidate.trade_event_id},
                )
                row = result.fetchone()
                if row is None:
                    continue

                status = row[0]
                pnl = row[1]

                if status == "open":
                    outcomes.open_trades += 1
                elif status == "closed":
                    if pnl is not None and pnl > 0:
                        outcomes.wins += 1
                    else:
                        outcomes.losses += 1
            except Exception as e:
                logger.debug("Error fetching trade outcome for event_id=%s: %s",
                             candidate.trade_event_id, e)
                continue
    finally:
        session.close()

    return outcomes


def _compute_shadow_outcomes(
    engine, candidates: list[FunnelCandidate]
) -> ShadowOutcomes:
    """Compute shadow outcome counts for filtered candidates.

    Links through blocked_candidate_id to blocked_trade_candidate_outcomes
    to determine target_hit/stop_hit/timed_exit/insufficient_data.
    """
    outcomes = ShadowOutcomes()

    # Get candidates with shadow linkage
    shadow_linked = [c for c in candidates if c.blocked_candidate_id is not None]
    if not shadow_linked:
        return outcomes

    outcomes.total = len(shadow_linked)

    session = get_session(engine)
    try:
        for candidate in shadow_linked:
            try:
                result = session.execute(
                    text(
                        """
                        SELECT outcome_label, target_hit, stop_hit
                        FROM blocked_trade_candidate_outcomes
                        WHERE blocked_candidate_id = :blocked_id
                        ORDER BY eval_window DESC
                        LIMIT 1
                        """
                    ),
                    {"blocked_id": candidate.blocked_candidate_id},
                )
                row = result.fetchone()
                if row is None:
                    outcomes.insufficient_data += 1
                    continue

                outcome_label = row[0]
                target_hit = row[1]
                stop_hit = row[2]

                if target_hit:
                    outcomes.target_hit += 1
                elif stop_hit:
                    outcomes.stop_hit += 1
                elif outcome_label and "timed" in outcome_label.lower():
                    outcomes.timed_exit += 1
                else:
                    outcomes.insufficient_data += 1
            except Exception as e:
                logger.debug(
                    "Error fetching shadow outcome for blocked_id=%s: %s",
                    candidate.blocked_candidate_id, e
                )
                outcomes.insufficient_data += 1
                continue
    finally:
        session.close()

    return outcomes


def _compute_top_rejection_reasons(
    candidates: list[FunnelCandidate], top_n: int = 3
) -> list[tuple[str, int]]:
    """Extract top N rejection reasons by frequency.

    Parses stage_decisions for rejected decisions and counts reasoning
    strings (normalized to lowercase for grouping).
    """
    reason_counter: Counter[str] = Counter()

    for candidate in candidates:
        decisions = _parse_stage_decisions(candidate)
        for dec in decisions:
            if dec.get("decision") == "rejected":
                reasoning = dec.get("reasoning", "unknown")
                # Normalize for grouping
                normalized = reasoning.strip().lower()[:100] if reasoning else "unknown"
                reason_counter[normalized] += 1

    return reason_counter.most_common(top_n)


def _classify_stage(
    stage: str, candidates: list[FunnelCandidate], days_in_window: int
) -> StageClassification:
    """Classify a single stage's health from trailing window candidates.

    Computes promotion/rejection/timeout rates and classifies as:
    - overly_permissive if promotion rate > 90%
    - overly_restrictive if rejection rate > 90%
    - operationally_failing if timeout rate > 30%
    """
    classification = StageClassification(
        stage=stage,
        days_in_window=days_in_window,
    )

    # Count decisions for this stage across all candidates in window
    entered = 0
    promoted = 0
    rejected = 0
    timed_out = 0

    agent_map = {
        "discovery": "scout",
        "research": "researcher",
        "analysis": "analyst",
        "confirmation": "confirmation",
    }
    target_agent = agent_map.get(stage, stage)

    for candidate in candidates:
        decisions = _parse_stage_decisions(candidate)

        # For discovery, every candidate counts as entered and promoted
        if stage == "discovery":
            entered += 1
            promoted += 1
            continue

        # For other stages, look at the agent's decision
        for dec in decisions:
            if dec.get("agent") == target_agent:
                entered += 1
                dec_value = dec.get("decision", "")
                if dec_value == "promoted":
                    promoted += 1
                elif dec_value == "rejected":
                    rejected += 1
                elif dec_value in ("timed_out", "not_evaluated"):
                    timed_out += 1
                break  # Count each candidate once per stage

    classification.candidates_in_window = entered

    if entered < _MIN_CANDIDATES_FOR_CLASSIFICATION:
        # Not enough data to classify
        return classification

    classification.promotion_rate = promoted / entered if entered > 0 else 0.0
    classification.rejection_rate = rejected / entered if entered > 0 else 0.0
    classification.timeout_rate = timed_out / entered if entered > 0 else 0.0

    # Apply classification thresholds
    if classification.promotion_rate > _OVERLY_PERMISSIVE_THRESHOLD:
        classification.classification = "overly_permissive"
    elif classification.rejection_rate > _OVERLY_RESTRICTIVE_THRESHOLD:
        classification.classification = "overly_restrictive"
    elif classification.timeout_rate > _OPERATIONALLY_FAILING_THRESHOLD:
        classification.classification = "operationally_failing"

    return classification


def _extract_geometry(candidate: FunnelCandidate) -> dict[str, Any] | None:
    """Extract entry_price/stop_price/target_price geometry from stage decisions.

    Looks in Analyst evidence first (authoritative), then Scout evidence.
    Returns dict with geometry fields or None if not found.
    """
    decisions = _parse_stage_decisions(candidate)

    # Check Analyst evidence first (more authoritative)
    for dec in reversed(decisions):
        if dec.get("agent") == "analyst":
            evidence = dec.get("evidence", {})
            if isinstance(evidence, dict):
                key_levels = evidence.get("key_levels", {})
                if isinstance(key_levels, dict):
                    entry = key_levels.get("entry_price")
                    stop = key_levels.get("stop_price") or key_levels.get("support")
                    target = key_levels.get("target_price") or key_levels.get("resistance")
                    if entry is not None and stop is not None and target is not None:
                        return {
                            "entry_price": entry,
                            "stop_price": stop,
                            "target_price": target,
                        }

    # Fall back to Scout evidence
    for dec in reversed(decisions):
        if dec.get("agent") == "scout":
            evidence = dec.get("evidence", {})
            if isinstance(evidence, dict):
                entry = evidence.get("entry_price")
                stop = evidence.get("stop_price")
                target = evidence.get("target_price")
                if entry is not None and stop is not None and target is not None:
                    return {
                        "entry_price": entry,
                        "stop_price": stop,
                        "target_price": target,
                    }

    return None


def _record_shadow_candidate(
    engine,
    candidate: FunnelCandidate,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> int | None:
    """Record a rejected funnel candidate in the shadow ledger.

    Uses the existing record_blocked_candidate() function from shadow_ledger.
    The shadow ledger expects a Session-like object (with .execute and .flush).

    Returns:
        blocked_candidate_id on success, None on failure/dedup.
    """
    try:
        from utils.shadow_ledger import record_blocked_candidate

        # Determine direction from candidate's direction_bias
        direction = None
        if candidate.direction_bias == "bullish":
            direction = "long"
        elif candidate.direction_bias == "bearish":
            direction = "short"

        # Determine the rejecting stage
        stage_status = candidate.stage_status or ""
        if "research" in stage_status:
            blocked_by = "funnel_researcher"
        elif "analysis" in stage_status:
            blocked_by = "funnel_analyst"
        elif "confirmation" in stage_status:
            blocked_by = "funnel_confirmation"
        else:
            blocked_by = "funnel_pipeline"

        # Get rejection reasoning from the last rejection decision
        decisions = _parse_stage_decisions(candidate)
        block_reason = "Rejected by funnel pipeline"
        for dec in reversed(decisions):
            if dec.get("decision") == "rejected":
                block_reason = dec.get("reasoning", block_reason)
                break

        # record_blocked_candidate expects a Session (calls .flush())
        session = get_session(engine)
        try:
            blocked_id = record_blocked_candidate(
                session,
                symbol=candidate.symbol,
                action="BUY" if direction == "long" else "SHORT",
                blocked_by=blocked_by,
                block_reason=block_reason,
                direction=direction,
                setup_type=candidate.authoritative_setup_type or candidate.preliminary_setup_type,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                source="funnel_pipeline",
                agent=blocked_by,
            )
            session.commit()
            return blocked_id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    except Exception as e:
        logger.error(
            "Failed to record shadow candidate for %s: %s",
            candidate.symbol, e
        )
        return None


def _parse_stage_decisions(candidate: FunnelCandidate) -> list[dict]:
    """Parse stage_decisions JSON field safely."""
    try:
        decisions = json.loads(candidate.stage_decisions or "[]")
        if isinstance(decisions, list):
            return decisions
    except (json.JSONDecodeError, TypeError):
        pass
    return []
