"""
Strategy Store
Merges hardcoded strategies from models/strategies.py with
dynamic strategies proposed by the Quant Researcher.
"""

import json
from db.schema import get_session, DynamicStrategy
from models.strategies import STRATEGIES, SETUP_TYPE_MAP


LIVE_DYNAMIC_STRATEGY_STATUSES = ("active", "live_50", "live_100")


def get_all_strategies(engine) -> dict:
    """Return merged dict of hardcoded + live dynamic strategies.

    Hardcoded strategies are always included with source="hardcoded".
    Dynamic strategies are included only when status is 'live_50' or 'live_100'.
    """
    all_strats = {}

    # Hardcoded — always included, unchanged
    for key, strat in STRATEGIES.items():
        all_strats[key] = {**strat, "source": "hardcoded"}

    # Dynamic — only live_50 and live_100 strategies
    db = get_session(engine)
    dynamic = db.query(DynamicStrategy).filter(
        DynamicStrategy.status.in_(["live_50", "live_100"])
    ).all()
    for d in dynamic:
        all_strats[d.key] = {
            "name": d.name,
            "description": d.description,
            "timeframe": d.timeframe or "",
            "bias": d.bias or "",
            "ideal_conditions": json.loads(d.ideal_conditions) if d.ideal_conditions else {},
            "failure_conditions": json.loads(d.failure_conditions) if d.failure_conditions else [],
            "execution_notes": json.loads(d.execution_notes) if d.execution_notes else [],
            "win_rate_documented": d.win_rate,
            "total_trades": d.total_trades,
            "source": "dynamic",
            "status": d.status,
            "pipeline_stage": d.pipeline_stage,
        }
    db.close()
    return all_strats


def get_pipeline_strategies(engine, stage: str = None) -> list[DynamicStrategy]:
    """Query DynamicStrategy records by pipeline stage for evaluation.

    Args:
        engine: SQLAlchemy engine instance.
        stage: Pipeline stage to filter by (backtest, paper_trade, live_50, live_100).
               If None, returns all strategies in any pipeline stage.

    Returns:
        List of DynamicStrategy records matching the stage filter.
    """
    pipeline_stages = ["backtest", "paper_trade", "live_50", "live_100"]
    db = get_session(engine)
    if stage is not None:
        strategies = db.query(DynamicStrategy).filter(
            DynamicStrategy.status == stage
        ).all()
    else:
        strategies = db.query(DynamicStrategy).filter(
            DynamicStrategy.status.in_(pipeline_stages)
        ).all()
    db.close()
    return strategies


def get_all_setup_types(engine) -> list[str]:
    """Return all valid setup type names (hardcoded + live dynamic).

    Setup-type aliases in ``SETUP_TYPE_MAP`` are valid analyst labels even
    when they map onto a broader strategy, e.g. ``technical_breakout`` -> ORB.
    """
    types = list(SETUP_TYPE_MAP.keys()) + list(STRATEGIES.keys())
    db = get_session(engine)
    dynamic = db.query(DynamicStrategy).filter(
        DynamicStrategy.status.in_(LIVE_DYNAMIC_STRATEGY_STATUSES)
    ).all()
    types += [d.key for d in dynamic]
    db.close()
    return sorted(set(types))


def propose_strategy(engine, strategy_data: dict) -> DynamicStrategy:
    """
    Add a new dynamic strategy proposed by an agent.
    strategy_data keys: key, name, description, timeframe, bias,
                        ideal_conditions, failure_conditions, execution_notes
    """
    db = get_session(engine)

    # Check if already exists
    existing = db.query(DynamicStrategy).filter_by(key=strategy_data["key"]).first()
    if existing:
        # Update if retired, skip if active
        if existing.status == "retired":
            existing.status = "backtest"
            existing.pipeline_stage = "backtest"
            existing.retired_at = None
            existing.retire_reason = None
            existing.failure_stage = None
            existing.failure_reason = None
            existing.description = strategy_data.get("description", existing.description)
            db.commit()
            db.close()
            return existing
        db.close()
        return existing

    strat = DynamicStrategy(
        key=strategy_data["key"],
        name=strategy_data["name"],
        description=strategy_data["description"],
        timeframe=strategy_data.get("timeframe"),
        bias=strategy_data.get("bias"),
        ideal_conditions=json.dumps(strategy_data.get("ideal_conditions", {})),
        failure_conditions=json.dumps(strategy_data.get("failure_conditions", [])),
        execution_notes=json.dumps(strategy_data.get("execution_notes", [])),
        proposed_by=strategy_data.get("proposed_by", "quant_researcher"),
        status="backtest",
        pipeline_stage="backtest",
    )
    db.add(strat)
    db.commit()
    db.refresh(strat)
    db.close()
    return strat


def update_strategy_stats(engine):
    """
    Update win rates for all dynamic strategies based on case library data.
    Retire strategies that underperform after enough trades.
    """
    from models.case import Case
    from datetime import datetime

    db = get_session(engine)
    dynamic = db.query(DynamicStrategy).filter(
        DynamicStrategy.status.in_(["active", "probation", "backtest", "paper_trade", "live_50", "live_100"])
    ).all()

    for strat in dynamic:
        cases = db.query(Case).filter_by(setup_type=strat.key).all()
        if not cases:
            continue

        total = len(cases)
        wins = sum(1 for c in cases if c.outcome == "success")
        avg_pnl = sum(c.pnl_pct or 0 for c in cases) / total

        strat.total_trades = total
        strat.wins = wins
        strat.win_rate = round(wins / total * 100, 1) if total else None
        strat.avg_pnl_pct = round(avg_pnl, 2)

        # Retire if enough data and consistently losing.
        # Only retire strategies that have reached live stages (live_50 or live_100)
        # or legacy statuses (active, probation). Strategies still in early pipeline
        # stages (backtest, paper_trade) are managed by the deployment pipeline.
        if total >= 10 and strat.win_rate < 35 and avg_pnl < 0:
            if strat.status in ("active", "probation", "live_50", "live_100"):
                strat.status = "retired"
                strat.retired_at = datetime.utcnow()
                strat.retire_reason = f"Win rate {strat.win_rate}% with avg P&L {avg_pnl:.2f}% over {total} trades"

    db.commit()
    db.close()
