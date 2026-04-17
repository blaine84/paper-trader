"""
Slack report delivery.
Formats and sends morning/afternoon trading reports to Slack via the Web API.
Credentials are read from CEO_SLACK_BOT_TOKEN and CEO_SLACK_CHANNEL_ID env vars.
"""

import json
import logging
import os
import time

import requests

logger = logging.getLogger("slack_notifier")

# ---------------------------------------------------------------------------
# Strength ordering for analyst signal cap (higher = stronger)
# ---------------------------------------------------------------------------
_STRENGTH_ORDER = {"strong": 3, "moderate": 2, "weak": 1}

# Emoji scheme
_SIGNAL_EMOJI = {
    "bullish": "📈",
    "bearish": "📉",
    "neutral": "➡️",
}


def truncate_text(text: str, max_len: int = 2000) -> str:
    """Truncate *text* to at most *max_len* characters.

    If the text fits, it is returned unchanged.  Otherwise the last character
    is replaced with "…" so the result is exactly *max_len* characters.
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _section(mrkdwn_text: str) -> dict:
    """Return a Slack Block Kit section block with truncated mrkdwn text."""
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": truncate_text(mrkdwn_text)},
    }


def format_morning_report(data: dict) -> list[dict]:
    """Pure function: assembled morning data dict → Slack Block Kit blocks.

    Always produces blocks for ALL five sections (regime, recommended
    strategies, avoid strategies, scout picks, analyst signals) even when
    data is ``"Data unavailable"``.  The output is capped at 15 blocks;
    content within sections is truncated rather than sections being dropped.

    Args:
        data: Dict with keys ``date``, ``regime``, ``strategies``,
              ``scout_picks``, ``analyst_signals``.

    Returns:
        A list of Slack Block Kit block dicts (max 15).
    """
    from datetime import datetime  # local import to keep module-level light

    # --- header + divider (2 blocks) ---
    raw_date = data.get("date", "")
    try:
        pretty_date = datetime.fromisoformat(raw_date).strftime("%b %d")
    except (ValueError, TypeError):
        pretty_date = str(raw_date) if raw_date else "Unknown"

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": truncate_text(f"📊 Morning Report — {pretty_date}", 150),
            },
        },
        {"type": "divider"},
    ]

    # --- 1. Market regime (1 block) ---
    regime = data.get("regime", "Data unavailable")
    if regime == "Data unavailable":
        regime_text = "*Market Regime:* Data unavailable"
    else:
        regime_display = str(regime).replace("_", " ").title()
        regime_text = f"*Market Regime:* {regime_display}"
    blocks.append(_section(regime_text))

    # --- 2. Recommended strategies (1 block) ---
    strategies = data.get("strategies", {})
    recommended = strategies.get("recommended", "Data unavailable") if isinstance(strategies, dict) else "Data unavailable"
    if recommended == "Data unavailable":
        rec_text = "*Recommended Strategies:* Data unavailable"
    elif isinstance(recommended, list):
        names = ", ".join(str(s).replace("_", " ") for s in recommended) if recommended else "None"
        rec_text = f"*Recommended Strategies:* {names}"
    else:
        rec_text = f"*Recommended Strategies:* {recommended}"
    blocks.append(_section(rec_text))

    # --- 3. Strategies to avoid (1 block) ---
    avoid = strategies.get("avoid", "Data unavailable") if isinstance(strategies, dict) else "Data unavailable"
    if avoid == "Data unavailable":
        avoid_text = "*Strategies to Avoid:* Data unavailable"
    elif isinstance(avoid, list):
        names = ", ".join(str(s).replace("_", " ") for s in avoid) if avoid else "None"
        avoid_text = f"*Strategies to Avoid:* {names}"
    else:
        avoid_text = f"*Strategies to Avoid:* {avoid}"
    blocks.append(_section(avoid_text))

    # --- 4. Scout picks (1 block — combine all picks into one section) ---
    scout_picks = data.get("scout_picks", "Data unavailable")
    if scout_picks == "Data unavailable":
        picks_text = "*Scout Picks:* Data unavailable"
    elif isinstance(scout_picks, list) and len(scout_picks) == 0:
        picks_text = "*Scout Picks:* None"
    elif isinstance(scout_picks, list):
        lines = []
        for p in scout_picks:
            sym = p.get("symbol", "?")
            reason = p.get("reasoning", "")
            lines.append(f"• *{sym}* — {reason}")
        picks_text = "*Scout Picks:*\n" + "\n".join(lines)
    else:
        picks_text = f"*Scout Picks:* {scout_picks}"
    blocks.append(_section(picks_text))

    # --- 5. Analyst signals (1 block for header + up to 6 signal blocks) ---
    analyst_signals = data.get("analyst_signals", "Data unavailable")
    if analyst_signals == "Data unavailable":
        blocks.append(_section("*Analyst Signals:* Data unavailable"))
    elif isinstance(analyst_signals, list) and len(analyst_signals) == 0:
        blocks.append(_section("*Analyst Signals:* None"))
    elif isinstance(analyst_signals, list):
        # Sort by strength descending, cap at 6
        sorted_signals = sorted(
            analyst_signals,
            key=lambda s: _STRENGTH_ORDER.get(str(s.get("strength", "")).lower(), 0),
            reverse=True,
        )
        total = len(sorted_signals)
        top_signals = sorted_signals[:6]

        # Build a single combined text block for all signals
        lines = ["*Analyst Signals:*"]
        for sig in top_signals:
            sym = sig.get("symbol", "?")
            direction = str(sig.get("signal", "neutral")).lower()
            emoji = _SIGNAL_EMOJI.get(direction, "➡️")
            strength = sig.get("strength", "")
            lines.append(f"{emoji} *{sym}* — {strength}")

        if total > 6:
            omitted = total - 6
            lines.append(f"_…and {omitted} more signal{'s' if omitted != 1 else ''} omitted_")

        blocks.append(_section("\n".join(lines)))
    else:
        blocks.append(_section(f"*Analyst Signals:* {analyst_signals}"))

    # --- Enforce 15-block cap ---
    # The structure above produces at most: 2 (header+divider) + 5 sections = 7 blocks
    # which is well under 15.  But if future changes add more, truncate here.
    if len(blocks) > 15:
        blocks = blocks[:15]

    return blocks


def format_afternoon_report(data: dict) -> list[dict]:
    """Pure function: daily review dict → Slack Block Kit blocks.

    Handles two cases:
    - **Missing review**: ``{"missing": True, "date": "..."}`` → single section
      block with a "not available" message.
    - **Valid review**: produces blocks for day classification, executive
      summary, total P&L, win/loss count, per-profile breakdown, what worked,
      what failed, highest leverage fix, and tomorrow's focus.

    Emoji scheme:
        🟢 positive P&L, 🔴 negative P&L, ✅ wins, ❌ losses, 🎯 focus items

    The output is capped at **20 blocks**.  All mrkdwn text fields are
    truncated via :func:`truncate_text`.

    Args:
        data: The daily review dict from AgentMemory, or a fallback dict
              with ``{"missing": True, "date": ...}``.

    Returns:
        A list of Slack Block Kit block dicts (max 20).
    """
    from datetime import datetime  # local import to keep module-level light

    # ------------------------------------------------------------------
    # Fallback / missing review
    # ------------------------------------------------------------------
    if data.get("missing"):
        date_str = data.get("date", "unknown date")
        return [_section(f"Daily review not available for {date_str}")]

    # ------------------------------------------------------------------
    # Header + divider (2 blocks)
    # ------------------------------------------------------------------
    raw_date = data.get("date", "")
    try:
        pretty_date = datetime.fromisoformat(raw_date).strftime("%b %d")
    except (ValueError, TypeError):
        pretty_date = str(raw_date) if raw_date else "Unknown"

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": truncate_text(
                    f"📊 Afternoon Report — {pretty_date}", 150
                ),
            },
        },
        {"type": "divider"},
    ]

    # ------------------------------------------------------------------
    # 1. Day classification (1 block)
    # ------------------------------------------------------------------
    day_class = data.get("day_classification", "N/A")
    day_class_display = str(day_class).replace("_", " ").title()
    blocks.append(_section(f"*Day Classification:* {day_class_display}"))

    # ------------------------------------------------------------------
    # 2. Executive summary (1 block)
    # ------------------------------------------------------------------
    exec_summary = data.get("executive_summary", "N/A")
    blocks.append(_section(f"*Executive Summary:*\n{exec_summary}"))

    # ------------------------------------------------------------------
    # 3. Total P&L — dollar and percentage (1 block)
    # ------------------------------------------------------------------
    trade_perf = data.get("trade_performance", {}) or {}
    total_pnl = trade_perf.get("total_pnl", 0) or 0
    total_pnl_pct = trade_perf.get("total_pnl_pct", 0) or 0

    pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
    pnl_pct_emoji = "🟢" if total_pnl_pct >= 0 else "🔴"
    blocks.append(
        _section(
            f"*Total P&L:* {pnl_emoji} ${total_pnl:+,.2f}  |  "
            f"{pnl_pct_emoji} {total_pnl_pct:+.2f}%"
        )
    )

    # ------------------------------------------------------------------
    # 4. Win / Loss count (1 block)
    # ------------------------------------------------------------------
    wins = trade_perf.get("wins", 0) or 0
    losses = trade_perf.get("losses", 0) or 0
    total_trades = trade_perf.get("total_trades", 0) or 0
    blocks.append(
        _section(
            f"*Trades:* {total_trades}  |  ✅ {wins} wins  |  ❌ {losses} losses"
        )
    )

    # ------------------------------------------------------------------
    # 5. Per-profile breakdown (1 block — combined into one section)
    # ------------------------------------------------------------------
    per_profile = trade_perf.get("per_profile", {}) or {}
    if per_profile:
        lines = ["*Per-Profile Breakdown:*"]
        for profile_name in sorted(per_profile.keys()):
            p = per_profile[profile_name]
            p_pnl = p.get("pnl", 0) or 0
            p_emoji = "🟢" if p_pnl >= 0 else "🔴"
            p_trades = p.get("trades", 0) or 0
            p_wins = p.get("wins", 0) or 0
            p_pnl_pct = p.get("pnl_pct", 0) or 0
            lines.append(
                f"• *{profile_name.title()}*: {p_trades} trades, "
                f"{p_wins} wins, {p_emoji} ${p_pnl:+,.2f} ({p_pnl_pct:+.2f}%)"
            )
        blocks.append(_section("\n".join(lines)))
    else:
        blocks.append(_section("*Per-Profile Breakdown:* N/A"))

    # ------------------------------------------------------------------
    # 6. What worked (1 block)
    # ------------------------------------------------------------------
    what_worked = data.get("what_worked", [])
    if isinstance(what_worked, list) and what_worked:
        items = "\n".join(f"• {item}" for item in what_worked)
        blocks.append(_section(f"*What Worked:*\n{items}"))
    else:
        blocks.append(_section("*What Worked:* N/A"))

    # ------------------------------------------------------------------
    # 7. What failed (1 block)
    # ------------------------------------------------------------------
    what_failed = data.get("what_failed", [])
    if isinstance(what_failed, list) and what_failed:
        items = "\n".join(f"• {item}" for item in what_failed)
        blocks.append(_section(f"*What Failed:*\n{items}"))
    else:
        blocks.append(_section("*What Failed:* N/A"))

    # ------------------------------------------------------------------
    # 8. Highest leverage fix (1 block)
    # ------------------------------------------------------------------
    hlf = data.get("highest_leverage_fix", "N/A") or "N/A"
    blocks.append(_section(f"*Highest Leverage Fix:*\n{hlf}"))

    # ------------------------------------------------------------------
    # 9. Tomorrow's focus (1 block)
    # ------------------------------------------------------------------
    focus = data.get("tomorrows_focus", [])
    if isinstance(focus, list) and focus:
        items = "\n".join(f"🎯 {item}" for item in focus)
        blocks.append(_section(f"*Tomorrow's Focus:*\n{items}"))
    else:
        blocks.append(_section("*Tomorrow's Focus:* N/A"))

    # ------------------------------------------------------------------
    # Enforce 20-block cap
    # ------------------------------------------------------------------
    if len(blocks) > 20:
        blocks = blocks[:20]

    return blocks


def assemble_morning_data(engine) -> dict:
    """Query AgentMemory for morning report data sources.

    Returns a structured dict with keys: date, regime, strategies,
    scout_picks, analyst_signals.  Any missing or unparseable source
    is replaced with ``"Data unavailable"``.
    """
    from datetime import date as _date
    from db.schema import AgentMemory, get_session

    today = _date.today().isoformat()
    result: dict = {
        "date": today,
        "regime": "Data unavailable",
        "strategies": {
            "recommended": "Data unavailable",
            "avoid": "Data unavailable",
        },
        "scout_picks": "Data unavailable",
        "analyst_signals": "Data unavailable",
    }

    try:
        db = get_session(engine)
    except Exception:
        logger.error("Failed to open DB session for morning data assembly")
        return result

    try:
        # --- Market regime ---
        try:
            regime_mem = (
                db.query(AgentMemory)
                .filter_by(agent="quant_researcher", key="regime")
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if regime_mem:
                try:
                    parsed = json.loads(regime_mem.value)
                    if isinstance(parsed, dict):
                        result["regime"] = parsed.get("regime", parsed.get("market_regime", str(parsed)))
                    else:
                        result["regime"] = str(parsed) if parsed else "Data unavailable"
                except json.JSONDecodeError:
                    # Value might be a plain string like "risk_on"
                    result["regime"] = regime_mem.value
        except Exception as exc:
            logger.error("Error querying regime: %s", exc)

        # --- Strategy recommendation ---
        try:
            strat_mem = (
                db.query(AgentMemory)
                .filter_by(agent="quant_researcher", key="strategy_recommendation")
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if strat_mem:
                try:
                    parsed = json.loads(strat_mem.value)
                    recommended = parsed.get("recommended_strategies", parsed.get("strategies", "Data unavailable"))
                    avoid = parsed.get("strategies_to_avoid", "Data unavailable")
                    result["strategies"] = {
                        "recommended": recommended if recommended else "Data unavailable",
                        "avoid": avoid if avoid else "Data unavailable",
                    }
                except json.JSONDecodeError:
                    logger.warning("Failed to parse strategy_recommendation JSON")
        except Exception as exc:
            logger.error("Error querying strategy_recommendation: %s", exc)

        # --- Scout picks ---
        try:
            scout_mem = (
                db.query(AgentMemory)
                .filter_by(agent="scout", key="daily_picks")
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if scout_mem:
                try:
                    parsed = json.loads(scout_mem.value)
                    picks = parsed.get("picks", parsed)
                    if isinstance(picks, list):
                        result["scout_picks"] = picks
                    else:
                        result["scout_picks"] = "Data unavailable"
                except json.JSONDecodeError:
                    logger.warning("Failed to parse daily_picks JSON")
        except Exception as exc:
            logger.error("Error querying scout picks: %s", exc)

        # --- Analyst signals ---
        try:
            signal_mems = (
                db.query(AgentMemory)
                .filter_by(agent="analyst", key="signal")
                .order_by(AgentMemory.timestamp.desc())
                .all()
            )
            if signal_mems:
                signals = []
                seen_symbols: set = set()
                for mem in signal_mems:
                    try:
                        parsed = json.loads(mem.value)
                        sym = parsed.get("symbol", mem.symbol)
                        if sym and sym not in seen_symbols:
                            seen_symbols.add(sym)
                            signals.append(parsed)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse analyst signal JSON for %s", mem.symbol)
                result["analyst_signals"] = signals if signals else "Data unavailable"
        except Exception as exc:
            logger.error("Error querying analyst signals: %s", exc)

    except Exception as exc:
        logger.error("Unexpected error during morning data assembly: %s", exc)
    finally:
        try:
            db.close()
        except Exception:
            pass

    return result


def assemble_afternoon_data(engine) -> dict:
    """Query AgentMemory for today's daily review.

    Returns the parsed daily review dict, or a fallback dict
    ``{"missing": True, "date": today}`` when the review is absent
    or unparseable.
    """
    from datetime import date as _date
    from db.schema import AgentMemory, get_session

    today = _date.today().isoformat()
    fallback = {"missing": True, "date": today}

    try:
        db = get_session(engine)
    except Exception:
        logger.error("Failed to open DB session for afternoon data assembly")
        return fallback

    try:
        review_mem = (
            db.query(AgentMemory)
            .filter_by(agent="daily_review", symbol=today, key="daily_review")
            .first()
        )
        if not review_mem:
            return fallback

        try:
            return json.loads(review_mem.value)
        except json.JSONDecodeError:
            logger.warning("Failed to parse daily_review JSON for %s", today)
            return fallback

    except Exception as exc:
        logger.error("Error querying daily review: %s", exc)
        return fallback
    finally:
        try:
            db.close()
        except Exception:
            pass


class SlackNotifier:
    """Slack report delivery. Reads credentials from env vars at init."""

    def __init__(self):
        self._token = (os.getenv("CEO_SLACK_BOT_TOKEN") or "").strip()
        self._channel = (os.getenv("CEO_SLACK_CHANNEL_ID") or "").strip()
        if not self._token:
            logger.warning("CEO_SLACK_BOT_TOKEN not set — Slack delivery disabled")
        if not self._channel:
            logger.warning("CEO_SLACK_CHANNEL_ID not set — Slack delivery disabled")

    def is_enabled(self) -> bool:
        """True only when both token and channel are present and non-empty."""
        return bool(self._token) and bool(self._channel)

    def send_blocks(self, blocks: list[dict], text: str) -> dict:
        """POST blocks to Slack chat.postMessage with retry on network errors.

        Args:
            blocks: List of Slack Block Kit block dicts.
            text: Fallback plain-text summary shown in notifications.

        Returns:
            {"ok": bool, "ts": str | None, "error": str | None}
        """
        if not self.is_enabled():
            return {"ok": False, "error": "disabled", "ts": None}

        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        payload = {
            "channel": self._channel,
            "blocks": blocks,
            "text": text,
        }

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                data = resp.json()

                if data.get("ok"):
                    ts = data.get("ts")
                    logger.info("Slack message sent successfully (ts=%s)", ts)
                    return {"ok": True, "ts": ts, "error": None}

                # Slack API returned ok: false
                error_code = data.get("error", "unknown_error")
                logger.error(
                    "Slack API error: %s — %s",
                    error_code,
                    data.get("response_metadata", {}).get("messages", ""),
                )
                return {"ok": False, "error": error_code, "ts": None}

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                logger.warning(
                    "Slack request failed (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt < max_attempts:
                    time.sleep(5)

        # All retries exhausted
        logger.error("Slack delivery failed after %d attempts", max_attempts)
        return {"ok": False, "error": "timeout", "ts": None}

    def send_morning_report(self, engine) -> dict:
        """Assemble, format, and send the morning report to Slack.

        Args:
            engine: SQLAlchemy engine for querying AgentMemory.

        Returns:
            Result dict from :meth:`send_blocks`.
        """
        data = assemble_morning_data(engine)
        blocks = format_morning_report(data)
        return self.send_blocks(blocks, f"Morning Report — {data.get('date', 'today')}")

    def send_afternoon_report(self, engine) -> dict:
        """Assemble, format, and send the afternoon report to Slack.

        Args:
            engine: SQLAlchemy engine for querying AgentMemory.

        Returns:
            Result dict from :meth:`send_blocks`.
        """
        data = assemble_afternoon_data(engine)
        blocks = format_afternoon_report(data)
        return self.send_blocks(blocks, f"Afternoon Report — {data.get('date', 'today')}")
