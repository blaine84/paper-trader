"""Analyst feedback loop driven by Reviewer quality flags."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from db.schema import (
    AnalystFeedbackQueue,
    AnalystMitigation,
    get_session,
)
from utils.llm import call_llm, parse_json_response

log = logging.getLogger(__name__)

RESPONSE_DEADLINE_HOURS = 24
LOOKBACK_DAYS = 7
RESET_ACCEPTANCE_THRESHOLD = 0.80
NO_DATA_REJECT_THRESHOLD = 0.50
MIN_RESPONSES_FOR_MITIGATION = 2

SIGNAL_SCORES = {"weak": 1.0, "moderate": 2.0, "strong": 3.0}
CONFIDENCE_SCORES = {"low": 1.0, "medium": 2.0, "high": 3.0}
ENTRY_WINDOW_LIMITS = {
    "gap_and_go": 60,
    "orb": 60,
    "momentum_fade": 60,
    "short_squeeze": 60,
}

ANALYST_RESPONSE_PROMPT = """You are the Analyst responding to Reviewer quality flags.

For each flag, respond with exactly one action:
- accept: the Reviewer is right and the Analyst will adopt the recommendation
- reject: only if you can cite concrete supporting data
- modify: partially accept and specify a narrower adjustment

Rules:
- Rejects MUST include supporting_data with concrete evidence.
- If you modify, include a concise mitigation_plan.
- Keep notes short and operational.

Return JSON:
{
  "responses": [
    {
      "id": 1,
      "action": "accept|reject|modify",
      "note": "short rationale",
      "supporting_data": ["evidence item"] or [],
      "mitigation_plan": "optional narrower adjustment"
    }
  ]
}
"""


def queue_reviewer_flags(engine, cases: list[dict]) -> list[dict]:
    """Derive analyst-facing quality flags from reviewer cases and persist them."""
    db = get_session(engine)
    queued = []

    for case in cases or []:
        for flag in _derive_flags(case):
            exists = (
                db.query(AnalystFeedbackQueue)
                .filter_by(
                    symbol=flag["symbol"],
                    date=flag["date"],
                    setup_type=flag.get("setup_type"),
                    flag_type=flag["flag_type"],
                )
                .first()
            )
            if exists:
                continue

            row = AnalystFeedbackQueue(
                trade_id=flag.get("trade_id"),
                symbol=flag["symbol"],
                setup_type=flag.get("setup_type"),
                date=flag["date"],
                flag_type=flag["flag_type"],
                severity=flag["severity"],
                recommendation=flag["recommendation"],
                reviewer_context=json.dumps(flag["reviewer_context"]),
                due_at=datetime.utcnow() + timedelta(hours=RESPONSE_DEADLINE_HOURS),
            )
            db.add(row)
            queued.append(flag)

    db.commit()
    db.close()
    return queued


def process_pending_feedback(engine) -> dict:
    """Have the Analyst respond to pending reviewer flags, then update mitigations."""
    db = get_session(engine)
    _mark_overdue_rows(db)
    pending = (
        db.query(AnalystFeedbackQueue)
        .filter(AnalystFeedbackQueue.responded_at == None)  # noqa: E711
        .order_by(AnalystFeedbackQueue.created_at.asc())
        .all()
    )
    if not pending:
        db.commit()
        db.close()
        maybe_reset_weekly_mitigations(engine)
        return {"responses": []}

    prompt_rows = []
    for row in pending:
        context = {}
        try:
            context = json.loads(row.reviewer_context) if row.reviewer_context else {}
        except Exception:
            context = {}
        prompt_rows.append(
            {
                "id": row.id,
                "symbol": row.symbol,
                "setup_type": row.setup_type,
                "date": row.date,
                "flag_type": row.flag_type,
                "severity": row.severity,
                "recommendation": row.recommendation,
                "reviewer_context": context,
            }
        )

    recent_feedback = (
        db.query(AgentMemory)
        .filter_by(agent="reviewer", key="selection_feedback")
        .order_by(AgentMemory.timestamp.desc())
        .limit(3)
        .all()
    )
    recent_text = "\n\n".join(m.value for m in recent_feedback) if recent_feedback else "None"
    db.close()

    user_prompt = (
        f"Current time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"Recent reviewer selection feedback:\n{recent_text}\n\n"
        f"Flags requiring response:\n{json.dumps(prompt_rows, indent=2)}"
    )

    try:
        raw = call_llm(ANALYST_RESPONSE_PROMPT, user_prompt, json_mode=True, tier="low")
        parsed = parse_json_response(raw)
        responses = parsed.get("responses", [])
    except Exception as exc:
        log.warning("Analyst feedback response failed, defaulting to accept: %s", exc)
        responses = [
            {
                "id": row["id"],
                "action": "accept",
                "note": "Fallback acceptance after response generation failure.",
                "supporting_data": [],
                "mitigation_plan": "",
            }
            for row in prompt_rows
        ]

    result = _persist_responses(engine, responses)
    evaluate_auto_mitigation(engine)
    maybe_reset_weekly_mitigations(engine)
    return result


def get_active_mitigations(engine) -> dict[str, dict]:
    db = get_session(engine)
    rows = db.query(AnalystMitigation).filter_by(active=True).all()
    result = {
        row.setup_type: {
            "level": row.level,
            "deployment_multiplier": row.deployment_multiplier,
            "signal_threshold_bump": row.signal_threshold_bump,
            "reason": row.reason or "",
        }
        for row in rows
    }
    db.close()
    return result


def build_feedback_prompt_context(engine) -> str:
    """Build analyst prompt context from pending flags and active mitigations."""
    db = get_session(engine)
    _mark_overdue_rows(db)
    pending = (
        db.query(AnalystFeedbackQueue)
        .filter(AnalystFeedbackQueue.responded_at == None)  # noqa: E711
        .order_by(AnalystFeedbackQueue.created_at.desc())
        .limit(5)
        .all()
    )
    mitigations = db.query(AnalystMitigation).filter_by(active=True).all()
    db.commit()
    db.close()

    lines = []
    if pending:
        lines.append("PENDING REVIEWER FLAGS (respond within 24h):")
        for row in pending:
            setup = f" [{row.setup_type}]" if row.setup_type else ""
            lines.append(
                f"- {row.symbol}{setup}: {row.flag_type} ({row.severity}) -> {row.recommendation}"
            )

    if mitigations:
        lines.append("\nACTIVE ANALYST MITIGATIONS:")
        for row in mitigations:
            lines.append(
                f"- {row.setup_type}: level {row.level}, deployment {row.deployment_multiplier:.2f}x, "
                f"signal threshold +{row.signal_threshold_bump:.1f}"
            )
        lines.append("When mitigations are active, prefer HOLD unless evidence is clearly stronger than normal.")

    return "\n".join(lines) if lines else "No pending analyst feedback flags or active mitigations."


def apply_signal_mitigation(signal: dict, mitigations: dict[str, dict]) -> dict:
    """Apply deterministic conservative throttles to an analyst signal."""
    if not isinstance(signal, dict):
        return signal

    setup_type = signal.get("setup_type")
    mitigation = mitigations.get(setup_type or "")
    if not mitigation or signal.get("signal") == "HOLD":
        return signal

    strength_score = SIGNAL_SCORES.get(str(signal.get("strength", "")).lower(), 1.0)
    confidence_score = CONFIDENCE_SCORES.get(str(signal.get("confidence", "")).lower(), 1.0)
    effective_score = ((strength_score + confidence_score) / 2.0) * mitigation["deployment_multiplier"]
    required_score = 2.0 + mitigation["signal_threshold_bump"]

    signal["mitigation"] = {
        "setup_type": setup_type,
        "level": mitigation["level"],
        "deployment_multiplier": mitigation["deployment_multiplier"],
        "signal_threshold_bump": mitigation["signal_threshold_bump"],
    }

    if effective_score >= required_score:
        return signal

    original_signal = signal.get("signal")
    signal["original_signal"] = original_signal
    signal["signal"] = "HOLD"
    signal["strength"] = "weak"
    signal["confidence"] = "low"
    note = (
        f" Auto-mitigation converted {original_signal} to HOLD for {setup_type}: "
        f"effective score {effective_score:.2f} below required {required_score:.2f}."
    )
    signal["reasoning"] = f"{signal.get('reasoning', '').strip()}{note}".strip()
    return signal


def get_quality_metrics(engine) -> dict:
    """Return dashboard metrics for today's analyst quality loop."""
    db = get_session(engine)
    today = datetime.utcnow().date()
    today_start = datetime.combine(today, datetime.min.time())

    _mark_overdue_rows(db)

    rows = (
        db.query(AnalystFeedbackQueue)
        .filter(AnalystFeedbackQueue.created_at >= today_start)
        .all()
    )
    responded = [r for r in rows if r.responded_at]
    due = [r for r in rows if r.due_at <= datetime.utcnow()]
    valid = [r for r in responded if not (r.analyst_response == "reject" and not r.analyst_supporting_data)]
    accepted = [r for r in valid if r.analyst_response in {"accept", "modify"}]

    avg_response_time = None
    if responded:
        hours = [
            (r.responded_at - r.created_at).total_seconds() / 3600.0
            for r in responded
            if r.responded_at and r.created_at
        ]
        if hours:
            avg_response_time = round(sum(hours) / len(hours), 2)

    mitigations = (
        db.query(AnalystMitigation)
        .filter_by(active=True)
        .order_by(AnalystMitigation.level.desc(), AnalystMitigation.setup_type.asc())
        .all()
    )
    highest_level = max((m.level for m in mitigations), default=0)

    result = {
        "flags_received": len(rows),
        "flags_accepted": len(accepted),
        "avg_response_time": avg_response_time,
        "response_rate": round(len(responded) / len(due), 2) if due else 1.0,
        "acceptance_rate": round(len(accepted) / len(valid), 2) if valid else 1.0,
        "current_mitigation_level": highest_level,
        "active_mitigations": [
            {
                "setup_type": m.setup_type,
                "level": m.level,
                "deployment_multiplier": m.deployment_multiplier,
                "signal_threshold_bump": m.signal_threshold_bump,
            }
            for m in mitigations
        ],
    }
    db.commit()
    db.close()
    return result


def evaluate_auto_mitigation(engine) -> None:
    """Apply setup-specific conservative mitigations after unsupported rejects."""
    db = get_session(engine)
    _mark_overdue_rows(db)
    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
    rows = (
        db.query(AnalystFeedbackQueue)
        .filter(AnalystFeedbackQueue.responded_at != None)  # noqa: E711
        .filter(AnalystFeedbackQueue.responded_at >= cutoff)
        .all()
    )

    by_setup = defaultdict(list)
    for row in rows:
        if row.setup_type:
            by_setup[row.setup_type].append(row)

    now = datetime.utcnow()
    for setup_type, setup_rows in by_setup.items():
        if len(setup_rows) < MIN_RESPONSES_FOR_MITIGATION:
            continue
        no_data_rejects = [
            r for r in setup_rows if r.analyst_response == "reject" and r.no_data_reject
        ]
        reject_rate = len(no_data_rejects) / len(setup_rows)
        if reject_rate <= NO_DATA_REJECT_THRESHOLD:
            continue

        mitigation = db.query(AnalystMitigation).filter_by(setup_type=setup_type).first()
        if mitigation is None:
            mitigation = AnalystMitigation(setup_type=setup_type)
            db.add(mitigation)

        mitigation.level = max(1, (mitigation.level or 0) + 1)
        mitigation.deployment_multiplier = max(0.25, round(1.0 - 0.25 * mitigation.level, 2))
        mitigation.signal_threshold_bump = round(0.5 * mitigation.level, 2)
        mitigation.active = True
        mitigation.reason = (
            f"Unsupported rejects {len(no_data_rejects)}/{len(setup_rows)} "
            f"({reject_rate:.0%}) over last {LOOKBACK_DAYS}d"
        )
        mitigation.applied_at = mitigation.applied_at or now
        mitigation.last_triggered_at = now
        mitigation.reset_at = None

    db.commit()
    db.close()


def maybe_reset_weekly_mitigations(engine) -> None:
    """Reset active mitigations after a strong week of valid analyst acceptance."""
    db = get_session(engine)
    active = db.query(AnalystMitigation).filter_by(active=True).all()
    if not active:
        db.close()
        return

    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
    for mitigation in active:
        rows = (
            db.query(AnalystFeedbackQueue)
            .filter_by(setup_type=mitigation.setup_type)
            .filter(AnalystFeedbackQueue.responded_at != None)  # noqa: E711
            .filter(AnalystFeedbackQueue.responded_at >= cutoff)
            .all()
        )
        valid = [
            r for r in rows
            if not (r.analyst_response == "reject" and not r.analyst_supporting_data)
        ]
        if not valid:
            continue
        accepted = [r for r in valid if r.analyst_response in {"accept", "modify"}]
        acceptance_rate = len(accepted) / len(valid)
        if acceptance_rate > RESET_ACCEPTANCE_THRESHOLD:
            mitigation.active = False
            mitigation.level = 0
            mitigation.deployment_multiplier = 1.0
            mitigation.signal_threshold_bump = 0.0
            mitigation.reset_at = datetime.utcnow()
            mitigation.reason = (
                f"Weekly reset after {acceptance_rate:.0%} acceptance on valid flags."
            )

    db.commit()
    db.close()


def _derive_flags(case: dict) -> list[dict]:
    flags = []
    symbol = case.get("symbol")
    date = case.get("date") or datetime.utcnow().strftime("%Y-%m-%d")
    setup_type = case.get("setup_type")
    selection_score = float(case.get("selection_score") or 0)
    signal_strength = str(case.get("signal_strength") or "").lower()
    signal_confidence = str(case.get("signal_confidence") or "").lower()
    holding_minutes = case.get("holding_minutes") or 0
    outcome = str(case.get("outcome") or "")

    def add(flag_type: str, severity: str, recommendation: str):
        flags.append(
            {
                "trade_id": case.get("trade_id"),
                "symbol": symbol,
                "setup_type": setup_type,
                "date": date,
                "flag_type": flag_type,
                "severity": severity,
                "recommendation": recommendation,
                "reviewer_context": case,
            }
        )

    if selection_score and selection_score <= 5:
        severity = "critical" if selection_score <= 3 else "high"
        add(
            "selection_score_below_threshold",
            severity,
            f"Tighten {setup_type or 'current'} setup classification rules and re-check catalyst/regime alignment.",
        )

    if signal_strength == "weak" and outcome != "success":
        add(
            "signal_strength_below_threshold",
            "medium",
            f"Require stronger confirmation before issuing {setup_type or 'this'} setup as tradable.",
        )

    if signal_confidence == "low" and outcome != "success":
        add(
            "signal_confidence_below_threshold",
            "medium",
            f"Raise confidence threshold or downgrade {setup_type or 'current'} setup to HOLD sooner.",
        )

    limit = ENTRY_WINDOW_LIMITS.get(setup_type or "")
    if limit and holding_minutes and holding_minutes > limit:
        add(
            f"{setup_type}_hold_time_violated",
            "medium",
            f"Review whether {setup_type} was misclassified or needed a tighter invalidation/time expectation.",
        )

    return flags


def _persist_responses(engine, responses: list[dict]) -> dict:
    db = get_session(engine)
    now = datetime.utcnow()
    stored = []

    rows = {
        row.id: row
        for row in db.query(AnalystFeedbackQueue)
        .filter(AnalystFeedbackQueue.responded_at == None)  # noqa: E711
        .all()
    }

    for response in responses or []:
        row = rows.get(response.get("id"))
        if row is None:
            continue

        action = str(response.get("action") or "accept").lower()
        note = response.get("note") or ""
        supporting_data = response.get("supporting_data") or []
        if isinstance(supporting_data, dict):
            supporting_data = [supporting_data]

        if action not in {"accept", "reject", "modify"}:
            action = "accept"

        if action == "modify" and response.get("mitigation_plan"):
            note = f"{note} Mitigation: {response['mitigation_plan']}".strip()

        row.status = "responded"
        row.analyst_response = action
        row.analyst_response_note = note
        row.analyst_supporting_data = json.dumps(supporting_data)
        row.no_data_reject = action == "reject" and len(supporting_data) == 0
        row.responded_at = now

        stored.append(
            {
                "id": row.id,
                "action": action,
                "no_data_reject": row.no_data_reject,
            }
        )

    db.commit()
    db.close()
    return {"responses": stored}


def _mark_overdue_rows(db) -> None:
    now = datetime.utcnow()
    overdue = (
        db.query(AnalystFeedbackQueue)
        .filter(AnalystFeedbackQueue.responded_at == None)  # noqa: E711
        .filter(AnalystFeedbackQueue.due_at < now)
        .all()
    )
    for row in overdue:
        row.status = "overdue"
