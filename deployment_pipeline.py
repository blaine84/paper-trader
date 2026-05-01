"""
Deployment Pipeline
Staged gate evaluation for dynamic strategies:
  Backtest → Paper Trade → Live 50% → Live 100%

Each stage has quantitative thresholds. Strategies that fail a gate
are reverted to backtest_failed and the Quant Researcher is notified.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime

from db.schema import get_session, DynamicStrategy, AgentMemory
from models.case import Case

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_STAGES = ["backtest", "paper_trade", "live_50", "live_100"]

BACKTEST_MIN_TRADES = 50
WIN_RATE_THRESHOLD = 0.55
TIME_GATE_DAYS = 7
TIME_GATE_MIN_TRADES = 5  # minimum trades required during a time-gated stage


# ---------------------------------------------------------------------------
# GateResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Result of a pipeline gate evaluation."""
    decision: str          # "advance" | "fail" | "wait"
    next_stage: str | None  # target stage if advancing
    reason: str            # human-readable explanation
    metrics: dict          # e.g. {"total_trades": 67, "win_rate": 0.58}


# ---------------------------------------------------------------------------
# Stage progression map
# ---------------------------------------------------------------------------

_NEXT_STAGE = {
    "backtest": "paper_trade",
    "paper_trade": "live_50",
    "live_50": "live_100",
}

_STAGE_START_DATE_FIELD = {
    "paper_trade": "paper_trade_start_date",
    "live_50": "live_50_start_date",
    "live_100": "live_100_start_date",
}


# ---------------------------------------------------------------------------
# Gate evaluation functions
# ---------------------------------------------------------------------------

def evaluate_backtest_gate(report: dict) -> GateResult:
    """
    Evaluate a backtest report against the minimum quality thresholds.

    Checks:
      - total_trades >= BACKTEST_MIN_TRADES (50)
      - win_rate > WIN_RATE_THRESHOLD (0.55)

    Returns a GateResult with decision advance, fail, or wait.
    """
    summary = report.get("summary", {})
    total_trades = summary.get("total_trades", 0)
    win_rate = summary.get("win_rate", 0.0)

    metrics = {"total_trades": total_trades, "win_rate": win_rate}

    if total_trades < BACKTEST_MIN_TRADES:
        return GateResult(
            decision="fail",
            next_stage=None,
            reason=f"insufficient trades ({total_trades} < {BACKTEST_MIN_TRADES})",
            metrics=metrics,
        )

    if win_rate <= WIN_RATE_THRESHOLD:
        return GateResult(
            decision="fail",
            next_stage=None,
            reason=f"win rate below threshold ({win_rate:.4f} <= {WIN_RATE_THRESHOLD})",
            metrics=metrics,
        )

    return GateResult(
        decision="advance",
        next_stage="paper_trade",
        reason=f"backtest passed ({total_trades} trades, {win_rate:.4f} win rate)",
        metrics=metrics,
    )


def evaluate_time_gated_stage(
    strategy: DynamicStrategy,
    stage: str,
    current_date: datetime,
    win_rate: float,
    total_trades: int,
) -> GateResult:
    """
    Evaluate a time-gated stage (paper_trade or live_50) for advancement.

    Checks:
      1. Elapsed time >= TIME_GATE_DAYS (7 days)
      2. Minimum trades >= TIME_GATE_MIN_TRADES (5)
      3. win_rate > WIN_RATE_THRESHOLD (0.55)

    If time hasn't elapsed, returns decision="wait".
    If time elapsed but not enough trades, returns decision="wait".
    If time elapsed and win_rate fails, returns decision="fail".
    If all pass, returns decision="advance".
    """
    # Determine stage start date
    start_date_field = _STAGE_START_DATE_FIELD.get(stage)
    start_date = getattr(strategy, start_date_field, None) if start_date_field else None

    if start_date is None:
        return GateResult(
            decision="wait",
            next_stage=None,
            reason=f"no start date recorded for {stage}",
            metrics={"total_trades": total_trades, "win_rate": win_rate},
        )

    elapsed = (current_date - start_date).days
    metrics = {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "elapsed_days": elapsed,
    }

    # Time gate not yet met
    if elapsed < TIME_GATE_DAYS:
        return GateResult(
            decision="wait",
            next_stage=None,
            reason=f"{stage}: {elapsed} days elapsed (need {TIME_GATE_DAYS})",
            metrics=metrics,
        )

    # Time elapsed but not enough trades — wait for more data
    if total_trades < TIME_GATE_MIN_TRADES:
        return GateResult(
            decision="wait",
            next_stage=None,
            reason=f"{stage}: not enough trades ({total_trades} < {TIME_GATE_MIN_TRADES}) after {elapsed} days",
            metrics=metrics,
        )

    # Time elapsed and enough trades — evaluate win rate
    next_stage = _NEXT_STAGE.get(stage)

    if win_rate <= WIN_RATE_THRESHOLD:
        return GateResult(
            decision="fail",
            next_stage=None,
            reason=f"{stage} win rate below threshold ({win_rate:.4f} <= {WIN_RATE_THRESHOLD})",
            metrics=metrics,
        )

    return GateResult(
        decision="advance",
        next_stage=next_stage,
        reason=f"{stage} passed ({total_trades} trades, {win_rate:.4f} win rate, {elapsed} days)",
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Win rate computation from Case records
# ---------------------------------------------------------------------------

def compute_stage_win_rate(
    engine, strategy_key: str, since_date: datetime
) -> tuple[float, int]:
    """
    Compute win rate from Case records since a given date.

    Queries cases where setup_type matches the strategy key and
    the case date is on or after since_date.

    Returns:
        (win_rate, total_trades) tuple.
        If no trades found, returns (0.0, 0).
    """
    db = get_session(engine)
    try:
        since_str = since_date.strftime("%Y-%m-%d")

        cases = (
            db.query(Case)
            .filter(
                Case.setup_type == strategy_key,
                Case.date >= since_str,
            )
            .all()
        )

        total = len(cases)
        if total == 0:
            return 0.0, 0

        wins = sum(1 for c in cases if c.outcome == "success")
        win_rate = wins / total

        return win_rate, total
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Apply gate result to strategy record
# ---------------------------------------------------------------------------

def apply_gate_result(
    engine, strategy: DynamicStrategy, result: GateResult
) -> None:
    """
    Apply a gate evaluation result to a DynamicStrategy record.

    - "advance": update status to next_stage, set appropriate start date,
                 update pipeline_stage.
    - "fail": set status to backtest_failed, record failure metadata,
              call escalate_failure.
    - "wait": do nothing.
    """
    if result.decision == "wait":
        return

    db = get_session(engine)
    try:
        # Re-fetch the strategy within this session
        strat = db.query(DynamicStrategy).filter_by(id=strategy.id).first()
        if strat is None:
            log.error(f"Strategy id={strategy.id} not found during apply_gate_result")
            return

        if result.decision == "advance":
            next_stage = result.next_stage
            strat.status = next_stage
            strat.pipeline_stage = next_stage

            # Set the appropriate start date for the new stage
            start_field = _STAGE_START_DATE_FIELD.get(next_stage)
            if start_field:
                setattr(strat, start_field, datetime.utcnow())

            log.info(
                f"Strategy '{strat.key}' advanced to {next_stage}"
            )

        elif result.decision == "fail":
            current_stage = strat.pipeline_stage or strat.status
            strat.status = "backtest_failed"
            strat.pipeline_stage = None
            strat.failure_stage = current_stage
            strat.failure_reason = result.reason

            log.warning(
                f"Strategy '{strat.key}' failed at {current_stage}: {result.reason}"
            )

        db.commit()
    except Exception as e:
        db.rollback()
        log.error(f"Failed to apply gate result for strategy {strategy.key}: {e}")
        raise
    finally:
        db.close()

    # Escalate failure outside the DB session
    if result.decision == "fail":
        current_stage = strategy.pipeline_stage or strategy.status
        try:
            escalate_failure(
                engine, strategy, current_stage, result.reason, result.metrics
            )
        except Exception as e:
            log.error(
                f"Failed to escalate failure for {strategy.key}: {e}"
            )


# ---------------------------------------------------------------------------
# Failure escalation
# ---------------------------------------------------------------------------

def escalate_failure(
    engine,
    strategy: DynamicStrategy,
    stage: str,
    reason: str,
    metrics: dict,
) -> None:
    """
    Create an AgentMemory record to notify the Quant Researcher
    about a pipeline failure.

    Record format:
      agent = "quant_researcher"
      key   = "pipeline_failure_{strategy_key}"
      value = JSON with strategy_key, failed_stage, failure_reason,
              performance_snapshot
    """
    strategy_key = strategy.key if hasattr(strategy, "key") else str(strategy)

    escalation = {
        "strategy_key": strategy_key,
        "failed_stage": stage,
        "failure_reason": reason,
        "performance_snapshot": metrics,
    }

    db = get_session(engine)
    try:
        memory_key = f"pipeline_failure_{strategy_key}"

        # Upsert: update existing or create new
        existing = (
            db.query(AgentMemory)
            .filter_by(agent="quant_researcher", key=memory_key)
            .first()
        )
        if existing:
            existing.value = json.dumps(escalation)
            existing.timestamp = datetime.utcnow()
        else:
            record = AgentMemory(
                agent="quant_researcher",
                key=memory_key,
                value=json.dumps(escalation),
            )
            db.add(record)

        db.commit()
        log.info(f"Escalated pipeline failure for {strategy_key} at {stage}")
    except Exception as e:
        db.rollback()
        log.error(f"Failed to write escalation for {strategy_key}: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Main pipeline evaluation
# ---------------------------------------------------------------------------

def run_pipeline_evaluation(engine) -> list[dict]:
    """
    Evaluate all strategies currently in pipeline stages.
    Called by the orchestrator during the pre-market cycle.

    For each strategy:
      - backtest: evaluate backtest gate (report must already exist)
      - paper_trade: evaluate time-gated stage with live win rate
      - live_50: evaluate time-gated stage with live win rate
      - live_100: no evaluation needed (terminal stage)

    Returns a list of result dicts for logging:
      [{"strategy_key": ..., "stage": ..., "decision": ..., "reason": ..., "metrics": ...}]
    """
    results = []
    current_date = datetime.utcnow()

    db = get_session(engine)
    try:
        # Query all strategies in pipeline stages
        strategies = (
            db.query(DynamicStrategy)
            .filter(DynamicStrategy.status.in_(PIPELINE_STAGES))
            .all()
        )
        # Detach from session so we can close it before processing
        strategy_data = []
        for s in strategies:
            db.expunge(s)
            strategy_data.append(s)
    finally:
        db.close()

    for strategy in strategy_data:
        try:
            result = _evaluate_single_strategy(engine, strategy, current_date)
            if result is None:
                continue

            results.append({
                "strategy_key": strategy.key,
                "stage": strategy.status,
                "decision": result.decision,
                "reason": result.reason,
                "metrics": result.metrics,
            })

            # Apply the gate result
            apply_gate_result(engine, strategy, result)

        except Exception as e:
            log.error(f"Error evaluating strategy {strategy.key}: {e}")
            results.append({
                "strategy_key": strategy.key,
                "stage": strategy.status,
                "decision": "error",
                "reason": str(e),
                "metrics": {},
            })

    return results


def _evaluate_single_strategy(
    engine, strategy: DynamicStrategy, current_date: datetime
) -> GateResult | None:
    """
    Evaluate a single strategy based on its current pipeline stage.
    Returns a GateResult or None if no evaluation is needed.
    """
    stage = strategy.status

    if stage == "backtest":
        return _evaluate_backtest_strategy(engine, strategy)

    elif stage in ("paper_trade", "live_50"):
        # Compute win rate from Case records since stage start
        start_field = _STAGE_START_DATE_FIELD.get(stage)
        start_date = getattr(strategy, start_field, None) if start_field else None

        if start_date is None:
            log.warning(
                f"Strategy '{strategy.key}' in {stage} has no start date"
            )
            return GateResult(
                decision="wait",
                next_stage=None,
                reason=f"no start date for {stage}",
                metrics={},
            )

        win_rate, total_trades = compute_stage_win_rate(
            engine, strategy.key, start_date
        )

        return evaluate_time_gated_stage(
            strategy, stage, current_date, win_rate, total_trades
        )

    elif stage == "live_100":
        # Terminal stage — no automatic advancement
        return None

    return None


def _evaluate_backtest_strategy(
    engine, strategy: DynamicStrategy
) -> GateResult | None:
    """
    Evaluate a strategy in the backtest stage.
    Looks up the stored backtest report and evaluates the gate.
    Returns None if no report exists yet (backtest hasn't run).
    """
    report_key = strategy.backtest_report_id
    if not report_key:
        # No backtest report yet — the orchestrator will trigger the backtest
        return None

    # Load the backtest report from AgentMemory
    db = get_session(engine)
    try:
        memory = (
            db.query(AgentMemory)
            .filter_by(agent="strategy_backtester", key=report_key)
            .first()
        )
        if memory is None:
            log.warning(
                f"Backtest report '{report_key}' not found for {strategy.key}"
            )
            return None

        report = json.loads(memory.value)
    finally:
        db.close()

    return evaluate_backtest_gate(report)
