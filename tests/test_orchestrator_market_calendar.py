"""Tests for orchestrator closed-market scheduling guards."""

import logging
from datetime import datetime
from unittest.mock import patch

from pytz import timezone

import orchestrator


def _et(year, month, day, hour=10, minute=0):
    return timezone("America/New_York").localize(
        datetime(year, month, day, hour, minute)
    )


def setup_function():
    orchestrator._market_closed_skips_logged.clear()


def test_memorial_day_market_job_is_skipped_and_logged_once(caplog):
    memorial_day = _et(2026, 5, 25)

    with caplog.at_level(logging.INFO):
        assert orchestrator._skip_closed_market_job("price_monitor", memorial_day)
        assert orchestrator._skip_closed_market_job("price_monitor", memorial_day)

    messages = [
        record.message for record in caplog.records
        if "MARKET_CLOSED_SKIP" in record.message
    ]
    assert messages == [
        "MARKET_CLOSED_SKIP: job=price_monitor date=2026-05-25 "
        "reason=holiday_or_weekend"
    ]


def test_open_market_day_is_not_skipped():
    assert not orchestrator._skip_closed_market_job(
        "price_monitor", _et(2026, 5, 26)
    )


def test_market_session_jobs_return_before_engine_access_when_closed():
    guarded_jobs = [
        orchestrator.run_pre_market,
        orchestrator.run_analyst_refresh,
        orchestrator.run_intraday,
        orchestrator.run_sector_scout_confirmation,
        orchestrator.run_sector_scout_midday,
        orchestrator.run_post_market,
        orchestrator.run_daily_review,
        orchestrator.run_shadow_outcomes,
        orchestrator.run_ceo_daily,
        orchestrator.run_ceo_weekly,
        orchestrator.run_price_monitor,
        orchestrator.run_news_monitor,
        orchestrator.run_position_health,
        orchestrator.run_price_spike_news_check,
        orchestrator.run_position_news_poll,
        orchestrator.run_position_timer,
    ]

    with (
        patch.object(orchestrator, "is_trading_day", return_value=False),
        patch.object(orchestrator, "get_engine") as get_engine,
    ):
        for job in guarded_jobs:
            job()
        orchestrator.run_narrator("hourly_recap")

    get_engine.assert_not_called()
