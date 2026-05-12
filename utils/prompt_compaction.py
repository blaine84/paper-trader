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


def compact_signal_for_pm(symbol: str, signal: dict) -> str:
    """
    Compact analyst signal representation for PM entry prompts.

    Preserves all decision-critical fields: signal direction, strength,
    setup_type, entry/stop/target levels, confidence, invalidation level,
    key_levels, symbol_class/setup reclassification metadata,
    freshness/catalyst warning (if present), and reasoning_summary
    (first 240 chars of reasoning).

    Omits full indicator detail objects and reasoning beyond 240 chars.
    """
    direction = signal.get("signal", "?")
    strength = signal.get("strength", "?")
    setup_type = signal.get("setup_type", "?")
    confidence = signal.get("confidence", "?")

    entry = signal.get("entry", "?")
    stop = signal.get("stop", "?")
    target = signal.get("target", "?")

    parts = [
        f"{symbol}: {direction} ({strength})",
        f"setup: {setup_type}",
        f"entry: {entry} / stop: {stop} / target: {target}",
        f"confidence: {confidence}",
    ]

    # Invalidation level
    invalidation = signal.get("invalidation")
    if invalidation:
        parts.append(f"invalidation: {invalidation}")

    # Current price and quote-derived key levels (for catalyst specificity gate)
    current_price = signal.get("current_price")
    if current_price is not None:
        parts.append(f"current_price: {current_price}")

    # Day high/low and prev_close from structured quote context
    day_high = signal.get("day_high")
    day_low = signal.get("day_low")
    prev_close = signal.get("prev_close")
    quote_levels = []
    if day_high is not None:
        quote_levels.append(f"H:{day_high}")
    if day_low is not None:
        quote_levels.append(f"L:{day_low}")
    if prev_close is not None:
        quote_levels.append(f"PC:{prev_close}")
    if quote_levels:
        parts.append(f"levels: {'/'.join(quote_levels)}")

    # Key levels (analyst-defined support/resistance)
    key_levels = signal.get("key_levels")
    if key_levels:
        if isinstance(key_levels, list):
            levels_str = ", ".join(str(lv) for lv in key_levels)
        else:
            levels_str = str(key_levels)
        parts.append(f"key_levels: [{levels_str}]")

    # Symbol class / setup reclassification metadata
    symbol_class = signal.get("symbol_class")
    if symbol_class:
        parts.append(f"symbol_class: {symbol_class}")

    # Freshness / catalyst warning
    catalyst_warning = signal.get("catalyst_warning")
    if catalyst_warning:
        parts.append(f"⚠ {catalyst_warning}")

    freshness_warning = signal.get("freshness_warning")
    if freshness_warning:
        parts.append(f"⚠ freshness: {freshness_warning}")

    # Reasoning summary — first 240 chars
    reasoning = signal.get("reasoning") or ""
    if reasoning:
        if len(reasoning) > 240:
            reasoning_summary = reasoning[:237] + "..."
        else:
            reasoning_summary = reasoning
        parts.append(f"reasoning: {reasoning_summary}")

    return " | ".join(parts)


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
