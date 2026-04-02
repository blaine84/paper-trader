"""
Case Library
Read/write interface for the structured case memory system.
Agents call this to store lessons and query relevant past cases.
"""

import json
from sqlalchemy.orm import Session
from models.case import Case
from db.schema import get_session


def store_case(engine, case_data: dict) -> Case:
    """
    Write a structured case to the library.
    case_data should match Case column names.
    """
    import json as _json
    # Serialize any list fields to JSON strings
    for field in ("conditions_for_success", "conditions_to_avoid"):
        if isinstance(case_data.get(field), list):
            case_data[field] = _json.dumps(case_data[field])
    db = get_session(engine)
    case = Case(**{k: v for k, v in case_data.items() if hasattr(Case, k)})
    db.add(case)
    db.commit()
    db.refresh(case)
    db.close()
    return case


def query_cases(
    engine,
    setup_type: str = None,
    catalyst_type: str = None,
    symbol: str = None,
    market_regime: str = None,
    outcome: str = None,
    bias: str = None,
    float_profile: str = None,
    sector: str = None,
    limit: int = 10,
) -> list[dict]:
    """
    Query cases by any combination of fields.
    Returns list of dicts (most recent first).
    """
    db = get_session(engine)
    q = db.query(Case)

    if setup_type:
        q = q.filter(Case.setup_type == setup_type)
    if catalyst_type:
        q = q.filter(Case.catalyst_type == catalyst_type)
    if symbol:
        q = q.filter(Case.symbol == symbol)
    if market_regime:
        q = q.filter(Case.market_regime == market_regime)
    if outcome:
        q = q.filter(Case.outcome == outcome)
    if bias:
        q = q.filter(Case.bias == bias)
    if float_profile:
        q = q.filter(Case.float_profile == float_profile)
    if sector:
        q = q.filter(Case.sector == sector)

    cases = q.order_by(Case.created_at.desc()).limit(limit).all()

    result = []
    for c in cases:
        result.append({
            "id": c.id,
            "date": c.date,
            "symbol": c.symbol,
            "setup_type": c.setup_type,
            "catalyst_type": c.catalyst_type,
            "float_profile": c.float_profile,
            "sector": c.sector,
            "premarket_gap_pct": c.premarket_gap_pct,
            "premarket_volume_rank": c.premarket_volume_rank,
            "market_regime": c.market_regime,
            "entry_timing": c.entry_timing,
            "bias": c.bias,
            "signal_strength": c.signal_strength,
            "rsi_at_entry": c.rsi_at_entry,
            "above_vwap": c.above_vwap,
            "above_daily_resistance": c.above_daily_resistance,
            "ema_trend": c.ema_trend,
            "outcome": c.outcome,
            "pnl_pct": c.pnl_pct,
            "holding_minutes": c.holding_minutes,
            "lesson": c.lesson,
            "conditions_for_success": json.loads(c.conditions_for_success) if c.conditions_for_success else [],
            "conditions_to_avoid": json.loads(c.conditions_to_avoid) if c.conditions_to_avoid else [],
            "confidence": c.confidence,
            "selection_score": c.selection_score,
            "execution_score": c.execution_score,
            "review_score": c.review_score,
            "profile": c.profile,
        })

    db.close()
    return result


def get_relevant_cases(engine, context: dict, limit: int = 5) -> list[dict]:
    """
    Smart lookup: find cases most relevant to a given trade context.
    Tries progressively broader queries until it finds matches.
    context keys: setup_type, catalyst_type, market_regime, symbol, bias, etc.
    """
    # Tier 1: exact match on setup + catalyst + regime
    cases = query_cases(
        engine,
        setup_type=context.get("setup_type"),
        catalyst_type=context.get("catalyst_type"),
        market_regime=context.get("market_regime"),
        limit=limit,
    )
    if cases:
        return cases

    # Tier 2: setup + regime
    cases = query_cases(
        engine,
        setup_type=context.get("setup_type"),
        market_regime=context.get("market_regime"),
        limit=limit,
    )
    if cases:
        return cases

    # Tier 3: catalyst + bias
    cases = query_cases(
        engine,
        catalyst_type=context.get("catalyst_type"),
        bias=context.get("bias"),
        limit=limit,
    )
    if cases:
        return cases

    # Tier 4: just setup type
    cases = query_cases(
        engine,
        setup_type=context.get("setup_type"),
        limit=limit,
    )

    return cases


def get_selection_feedback(engine, limit: int = 10) -> str:
    """
    Summarize selection score trends for Scout + Analyst context.
    Returns a compact block of cases sorted by selection_score.
    """
    db = get_session(engine)
    cases = (
        db.query(Case)
        .filter(Case.selection_score != None)
        .order_by(Case.created_at.desc())
        .limit(limit)
        .all()
    )
    db.close()

    if not cases:
        return "No selection scores yet."

    lines = []
    for c in cases:
        lines.append(
            f"  [{c.date} {c.symbol}] setup: {c.setup_type} | catalyst: {c.catalyst_type} "
            f"| regime: {c.market_regime} | outcome: {c.outcome} "
            f"| selection_score: {c.selection_score}/10"
            + (f" | lesson: {c.lesson}" if c.lesson else "")
        )
    return "\n".join(lines)


def get_execution_feedback(engine, profile_id: str = None, limit: int = 10) -> str:
    """
    Summarize execution score trends for PM context.
    Optionally filtered by profile so each PM gets its own feedback.
    """
    db = get_session(engine)
    q = db.query(Case).filter(Case.execution_score != None)
    if profile_id:
        q = q.filter(Case.profile == profile_id)
    cases = q.order_by(Case.created_at.desc()).limit(limit).all()
    db.close()

    if not cases:
        return "No execution scores yet."

    lines = []
    for c in cases:
        lines.append(
            f"  [{c.date} {c.symbol}] profile: {c.profile} | entry_timing: {c.entry_timing} "
            f"| above_vwap: {c.above_vwap} | rsi: {c.rsi_at_entry} "
            f"| pnl: {c.pnl_pct}% | execution_score: {c.execution_score}/10"
            + (f" | avoid_when: {c.conditions_to_avoid}" if c.conditions_to_avoid else "")
        )
    return "\n".join(lines)


def get_win_rate_by_setup(engine) -> list[dict]:
    """Aggregate win rates grouped by setup_type. Useful for dashboard/reporting."""
    db = get_session(engine)
    from sqlalchemy import func, Integer

    results = (
        db.query(
            Case.setup_type,
            func.count(Case.id).label("total"),
            func.sum(
                (Case.outcome == "success").cast(Integer)
            ).label("wins"),
            func.avg(Case.pnl_pct).label("avg_pnl_pct"),
            func.avg(Case.review_score).label("avg_score"),
        )
        .filter(Case.setup_type != None)
        .group_by(Case.setup_type)
        .all()
    )

    db.close()
    return [
        {
            "setup_type": r.setup_type,
            "total": r.total,
            "wins": r.wins or 0,
            "win_rate": round((r.wins or 0) / r.total * 100, 1),
            "avg_pnl_pct": round(r.avg_pnl_pct or 0, 2),
            "avg_score": round(r.avg_score or 0, 1),
        }
        for r in results
    ]


def format_cases_for_prompt(cases: list[dict]) -> str:
    """
    Format cases as a compact block for injection into agent prompts.
    Omits nulls to keep tokens lean.
    """
    if not cases:
        return "No relevant past cases found."

    lines = []
    for c in cases:
        parts = [f"[{c['date']} {c['symbol']}]"]
        for field in [
            "setup_type", "catalyst_type", "float_profile", "market_regime",
            "premarket_gap_pct", "premarket_volume_rank", "bias",
            "entry_timing", "above_vwap", "above_daily_resistance",
            "rsi_at_entry", "ema_trend", "signal_strength",
        ]:
            val = c.get(field)
            if val is not None:
                parts.append(f"{field}: {val}")
        parts.append(f"outcome: {c['outcome']} (pnl: {c.get('pnl_pct', '?')}%)")
        if c.get("lesson"):
            parts.append(f"lesson: {c['lesson']}")
        if c.get("conditions_for_success"):
            parts.append(f"works_when: {', '.join(c['conditions_for_success'])}")
        if c.get("conditions_to_avoid"):
            parts.append(f"avoid_when: {', '.join(c['conditions_to_avoid'])}")
        lines.append("  " + " | ".join(parts))

    return "\n".join(lines)
