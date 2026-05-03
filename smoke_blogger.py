"""
Smoke test: run the weekly-wrap narrator pipeline end-to-end
and publish to Blogger.

Usage:
    python smoke_blogger.py              # run weekly wrap, publish to Blogger
    python smoke_blogger.py --dry-run    # run weekly wrap, print narrative, skip publish
    python smoke_blogger.py --force      # delete any existing dedup entry first
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-24s %(levelname)-5s %(message)s",
)
log = logging.getLogger("smoke_blogger")

from db.schema import init_db, get_session, AgentMemory
from agents.narrator import (
    assemble_weekly_wrap,
    build_system_prompt,
    build_user_prompt,
    compute_confidence_regime,
    format_blog_title,
)
from utils.llm import call_llm
from utils.blogger_publisher import BloggerPublisher


def main():
    parser = argparse.ArgumentParser(description="Smoke-test weekly wrap + Blogger")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate narrative but skip Blogger publish")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing dedup entry so narrator regenerates")
    args = parser.parse_args()

    engine = init_db("db/paper_trader.db")
    update_type = "weekly_wrap"
    today = "2026-04-25"

    # --- Optionally clear dedup so we can regenerate ---
    if args.force:
        db = get_session(engine)
        deleted = (
            db.query(AgentMemory)
            .filter_by(agent="narrator", key=update_type, symbol=today)
            .delete()
        )
        db.commit()
        db.close()
        log.info(f"Cleared {deleted} existing dedup entry(ies) for {update_type} [{today}]")

    # --- 1. Assemble context ---
    log.info("Assembling weekly-wrap context...")
    ctx = assemble_weekly_wrap(engine)

    try:
        ctx["confidence_regime"] = compute_confidence_regime(engine)
    except Exception as e:
        log.warning(f"Confidence regime failed: {e}")
        ctx["confidence_regime"] = {}

    log.info("Context keys: %s", list(ctx.keys()))
    log.info("Week P&L: %s", json.dumps(ctx.get("week_pnl", {}), indent=2, default=str))
    log.info("Best trades: %d, Worst trades: %d",
             len(ctx.get("best_trades", [])), len(ctx.get("worst_trades", [])))

    # --- 2. Generate narrative via LLM ---
    log.info("Calling LLM for weekly-wrap narrative...")
    system_prompt = build_system_prompt(update_type)
    user_prompt = build_user_prompt(update_type, ctx)
    narrative = call_llm(system_prompt, user_prompt, tier="medium")

    if not narrative:
        log.error("LLM returned empty narrative — aborting")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("WEEKLY WRAP NARRATIVE")
    print("=" * 60)
    print(narrative)
    print("=" * 60 + "\n")

    # --- 3. Store to AgentMemory (dedup) ---
    result = {
        "narrative": narrative,
        "update_type": update_type,
        "date": today,
        "dedup_key": today,
        "skipped": False,
    }
    try:
        db = get_session(engine)
        db.add(AgentMemory(
            agent="narrator",
            symbol=today,
            key=update_type,
            value=json.dumps(result, default=str),
        ))
        db.commit()
        db.close()
        log.info("Narrative stored in AgentMemory")
    except Exception as e:
        log.warning(f"AgentMemory write failed (maybe dupe): {e}")

    # --- 4. Publish to Blogger ---
    if args.dry_run:
        log.info("--dry-run: skipping Blogger publish")
        return

    publisher = BloggerPublisher()
    if not publisher.is_enabled():
        log.error("Blogger disabled — check env vars: BLOGGER_BLOG_ID, "
                   "GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN")
        sys.exit(1)

    title = format_blog_title(update_type, today)
    log.info(f"Publishing to Blogger: {title}")
    pub_result = publisher.publish(title=title, content=narrative)
    print(pub_result)

    if pub_result["ok"]:
        print(f"\nSUCCESS — post live at {pub_result['url']}")
    else:
        print(f"\nFAIL — {pub_result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
