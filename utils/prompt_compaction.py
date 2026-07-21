"""
Prompt Compaction Helpers

Dedicated helper functions that produce compact representations of cases, signals,
daily logs, and case trends for injection into PM entry and Narrator prompts.

These functions do NOT modify the shared formatters (format_cases_for_prompt,
build_strategy_context) — they are new, purpose-built alternatives for specific
prompt assembly paths that need smaller token footprints.
"""

import json


def format_cases_digest_for_pm(cases: list[dict]) -> str:
    """
    Compact case digest for PM entry prompts.

    Takes the same list[dict] as format_cases_for_prompt() but returns a much
    shorter digest per case containing only decision-critical fields:
    setup_type, outcome, pnl_pct, key lesson (truncated to ~80 chars),
    and 1-line failure mode from conditions_to_avoid.

    Target: ~300 chars/case, ~1.5k total for 5 cases.
    Does NOT modify format_cases_for_prompt().
    """
    if not cases:
        return "No relevant past cases found."

    lines = []
    for c in cases:
        symbol = c.get("symbol", "?")
        case_date = c.get("date", "?")
        setup_type = c.get("setup_type", "?")
        outcome = c.get("outcome", "?")
        pnl_pct = c.get("pnl_pct", "?")

        # Truncate lesson to ~80 chars
        lesson = c.get("lesson") or ""
        if len(lesson) > 80:
            lesson = lesson[:77] + "..."

        # Extract 1-line failure mode from conditions_to_avoid
        avoid = c.get("conditions_to_avoid") or []
        if isinstance(avoid, str):
            try:
                avoid = json.loads(avoid)
            except (json.JSONDecodeError, TypeError):
                avoid = [avoid] if avoid else []
        avoid_text = ", ".join(avoid[:2]) if avoid else "none"

        parts = [
            f"[{case_date} {symbol}]",
            f"setup: {setup_type}",
            f"outcome: {outcome} ({pnl_pct}%)",
        ]
        if lesson:
            parts.append(f"lesson: {lesson}")
        parts.append(f"avoid: {avoid_text}")

        lines.append("  " + " | ".join(parts))

    return "\n".join(lines)


def compact_signal_for_pm(symbol: str, signal: dict, scaffold_result: dict | None = None) -> str:
    """
    Compact analyst signal representation for PM entry prompts.

    Displays only Analyst-owned fields: symbol, direction, strength, setup_type,
    confidence, current_price, key_levels (support, resistance, VWAP), and
    invalidation. Omits fields that are missing or null rather than showing
    placeholders.

    Does NOT display entry_price, stop_loss, or target fields — those are
    owned by the geometry scaffold, not the Analyst.

    When scaffold_result is provided, appends formatted scaffold candidates
    or failure reason via format_scaffold_for_pm().
    """
    # --- Analyst-owned fields only; omit missing/null ---
    direction = signal.get("signal")
    strength = signal.get("strength")
    setup_type = signal.get("setup_type")
    confidence = signal.get("confidence")
    current_price = signal.get("current_price")
    invalidation = signal.get("invalidation")

    # Header line always includes symbol and direction (if available)
    header_parts = [symbol]
    if direction:
        if strength:
            header_parts[0] = f"{symbol}: {direction} ({strength})"
        else:
            header_parts[0] = f"{symbol}: {direction}"
    parts = [header_parts[0]]

    if setup_type:
        parts.append(f"setup: {setup_type}")

    if confidence:
        parts.append(f"confidence: {confidence}")

    if current_price is not None:
        parts.append(f"current_price: {current_price}")

    # Key levels — extract support, resistance, VWAP from key_levels dict
    key_levels = signal.get("key_levels")
    if key_levels and isinstance(key_levels, dict):
        kl_parts = []
        support = key_levels.get("support")
        resistance = key_levels.get("resistance")
        vwap = key_levels.get("vwap") or key_levels.get("VWAP")
        if support is not None:
            kl_parts.append(f"support: {support}")
        if resistance is not None:
            kl_parts.append(f"resistance: {resistance}")
        if vwap is not None:
            kl_parts.append(f"VWAP: {vwap}")
        if kl_parts:
            parts.append(f"key_levels: [{', '.join(kl_parts)}]")
    elif key_levels:
        # Legacy list or string format
        if isinstance(key_levels, list):
            levels_str = ", ".join(str(lv) for lv in key_levels)
        else:
            levels_str = str(key_levels)
        parts.append(f"key_levels: [{levels_str}]")

    if invalidation:
        parts.append(f"invalidation: {invalidation}")

    trigger_status = signal.get("trigger_status")
    if isinstance(trigger_status, dict):
        trigger_parts = []
        status = trigger_status.get("status")
        entry_trigger = trigger_status.get("entry_trigger")
        breakout = trigger_status.get("breakout")
        pullback = trigger_status.get("pullback")
        if status:
            trigger_parts.append(f"status: {status}")
        if entry_trigger:
            trigger_parts.append(f"entry_trigger: {entry_trigger}")
        if isinstance(breakout, dict) and breakout.get("status"):
            text = f"breakout: {breakout.get('status')}"
            if breakout.get("level") is not None:
                text += f" @ {breakout.get('level')}"
            trigger_parts.append(text)
        if isinstance(pullback, dict) and pullback.get("status"):
            text = f"pullback: {pullback.get('status')}"
            if pullback.get("level") is not None:
                text += f" @ {pullback.get('level')}"
            trigger_parts.append(text)
        if trigger_parts:
            parts.append(f"trigger_status: [{', '.join(trigger_parts)}]")

    result = " | ".join(parts)

    # Append scaffold section
    if scaffold_result is not None:
        result += "\n" + format_scaffold_for_pm(scaffold_result)
    else:
        result += "\n" + "No geometry scaffold available — PM must not trade this signal."

    return result


def format_scaffold_for_pm(scaffold_result: dict) -> str:
    """Format geometry scaffold output for PM prompt injection.

    If status == "ok": renders candidates table with candidate_id, name,
    entry_price, stop_loss, target, risk_reward, trigger.

    If status != "ok": renders failure message with reason.
    """
    status = scaffold_result.get("status", "")
    reason = scaffold_result.get("reason", "")

    if status == "ok":
        candidates = scaffold_result.get("candidates", [])
        if not candidates:
            return "No geometry scaffold available — PM must not trade this signal."

        lines = ["--- Geometry Scaffold Candidates ---"]
        header = f"{'ID':<40} {'Name':<25} {'Entry':<10} {'Stop':<10} {'Target':<10} {'R:R':<6} {'Trigger'}"
        lines.append(header)
        lines.append("-" * len(header))

        for c in candidates:
            cid = str(c.get("candidate_id", ""))
            name = str(c.get("name", ""))
            entry_price = c.get("entry_price", "")
            stop_loss = c.get("stop_loss", "")
            target = c.get("target", "")
            rr = c.get("risk_reward", "")
            trigger = str(c.get("trigger", ""))
            lines.append(
                f"{cid:<40} {name:<25} {entry_price:<10} {stop_loss:<10} {target:<10} {rr:<6} {trigger}"
            )

        return "\n".join(lines)

    elif status == "insufficient_data":
        msg = "No geometry scaffold available — PM must not trade this signal."
        if reason:
            msg += f"\nReason: {reason}"
        return msg

    elif status == "not_tradeable_signal":
        msg = "No executable geometry scaffold candidates — PM must not trade this signal."
        if reason:
            msg += f"\nReason: {reason}"
        return msg

    else:
        # Unknown status — fail safe
        msg = "No geometry scaffold available — PM must not trade this signal."
        if reason:
            msg += f"\nReason: {reason}"
        return msg


def compact_daily_log_for_narrator(log: dict) -> str:
    """
    Compact DailyLog representation for Narrator weekly_wrap prompts.

    Returns compact format: date, P&L, trade count, 1-line note summary
    (truncated notes to ~100 chars). Target < 200 chars per log entry.
    """
    log_date = log.get("date", "?")
    daily_pnl = log.get("daily_pnl", 0)
    daily_pnl_pct = log.get("daily_pnl_pct", 0)
    trades_taken = log.get("trades_taken", 0)
    winning = log.get("winning_trades", 0)
    losing = log.get("losing_trades", 0)

    # Truncate notes to ~100 chars
    notes = log.get("notes") or ""
    if len(notes) > 100:
        notes = notes[:97] + "..."

    parts = [
        f"[{log_date}]",
        f"P&L: ${daily_pnl:+.0f} ({daily_pnl_pct:+.2f}%)",
        f"trades: {trades_taken} (W{winning}/L{losing})",
    ]
    if notes:
        parts.append(f"note: {notes}")

    return " | ".join(parts)


def compact_case_trends_for_narrator(case_trends: list[dict]) -> str:
    """
    Aggregated case trend statistics for Narrator weekly_wrap prompts.

    Takes list of case trend dicts and returns aggregated statistics:
    - Win rate by setup_type
    - Top 3 lessons
    - Notable failures (cases with outcome="failure" and pnl_pct < -2%)

    Replaces individual case object dump.
    """
    if not case_trends:
        return "No case trends this week."

    # --- Win rate by setup_type ---
    setup_stats: dict[str, dict] = {}
    for c in case_trends:
        st = c.get("setup_type", "unknown")
        if st not in setup_stats:
            setup_stats[st] = {"total": 0, "wins": 0}
        setup_stats[st]["total"] += 1
        if c.get("outcome") == "success":
            setup_stats[st]["wins"] += 1

    wr_lines = ["Win rate by setup:"]
    for st, stats in sorted(setup_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        total = stats["total"]
        wins = stats["wins"]
        rate = round(wins / total * 100, 1) if total > 0 else 0.0
        wr_lines.append(f"  {st}: {rate}% ({wins}/{total})")

    # --- Top 3 lessons (deduplicated, most common) ---
    lesson_counts: dict[str, int] = {}
    for c in case_trends:
        lesson = c.get("lesson") or ""
        if lesson:
            # Truncate for dedup key
            key = lesson[:80]
            lesson_counts[key] = lesson_counts.get(key, 0) + 1

    top_lessons = sorted(lesson_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    lesson_lines = ["Top lessons:"]
    for lesson, count in top_lessons:
        lesson_lines.append(f"  - {lesson} (x{count})")

    # --- Notable failures ---
    failures = [
        c for c in case_trends
        if c.get("outcome") == "failure" and (c.get("pnl_pct") or 0) < -2.0
    ]
    failure_lines = ["Notable failures:"]
    if failures:
        for f in failures:
            sym = f.get("symbol", "?")
            st = f.get("setup_type", "?")
            pnl = f.get("pnl_pct", "?")
            failure_lines.append(f"  {sym} ({st}): {pnl}%")
    else:
        failure_lines.append("  None")

    return "\n".join(wr_lines + [""] + lesson_lines + [""] + failure_lines)
