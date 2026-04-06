"""
Strategy Store
Merges hardcoded strategies from models/strategies.py with
dynamic strategies proposed by the Quant Researcher.
"""

import json
from db.schema import get_session, DynamicStrategy
from models.strategies import STRATEGIES, SETUP_TYPE_MAP


def get_all_strategies(engine) -> dict:
    """Return merged dict of hardcoded + active dynamic strategies."""
    all_strats = {}

    # Hardcoded
    for key, strat in STRATEGIES.items():
        all_strats[key] = {**strat, "source": "hardcoded"}

    # Dynamic (active only)
    db = get_session(engine)
    dynamic = db.query(DynamicStrategy).filter_by(status="active").all()
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
        }
    db.close()
    return all_strats


def get_all_setup_types(engine) -> list[str]:
    """Return all valid setup type names (hardcoded + dynamic)."""
    types = list(SETUP_TYPE_MAP.keys()) + list(STRATEGIES.keys())
    db = get_session(engine)
    dynamic = db.query(DynamicStrategy).filter_by(status="active").all()
    types += [d.key for d in dynamic]
    db.close()
    return list(set(types))


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
            existing.status = "probation"
            existing.retired_at = None
            existing.retire_reason = None
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
        status="active",
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
        DynamicStrategy.status.in_(["active", "probation"])
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

        # Retire if enough data and consistently losing
        if total >= 10 and strat.win_rate < 35 and avg_pnl < 0:
            strat.status = "retired"
            strat.retired_at = datetime.utcnow()
            strat.retire_reason = f"Win rate {strat.win_rate}% with avg P&L {avg_pnl:.2f}% over {total} trades"

    db.commit()
    db.close()
