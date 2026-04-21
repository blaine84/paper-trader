"""
Tests for utils.catalyst_freshness — compute_freshness_state, compute_confidence,
get_breaking_news_for_symbols, and get_researcher_timestamps.

Covers task 1.2 requirements: 2.2, 2.3, 2.4, 2.7.
Covers task 1.3 requirements: 1.1, 1.2, 1.3, 1.4, 2.1.
"""

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import AgentMemory, Base
from utils.catalyst_freshness import (
    CONFIDENCE_MAP,
    ET,
    build_freshness_label,
    compute_catalyst_freshness,
    compute_confidence,
    compute_freshness_state,
    get_breaking_news_for_symbols,
    get_researcher_timestamps,
)


# ===================================================================
# Unit tests for compute_freshness_state
# ===================================================================


class TestComputeFreshnessState:
    """Unit tests for compute_freshness_state."""

    def _now(self):
        return datetime.now(timezone.utc)

    def test_fresh_when_recent(self):
        """Data less than 60 minutes old → fresh."""
        now = self._now()
        ts = now - timedelta(minutes=30)
        assert compute_freshness_state(ts, None, now) == "fresh"

    def test_aging_at_exactly_60_minutes(self):
        """Exactly 60 minutes → aging (not fresh)."""
        now = self._now()
        ts = now - timedelta(minutes=60)
        assert compute_freshness_state(ts, None, now) == "aging"

    def test_aging_at_61_minutes(self):
        now = self._now()
        ts = now - timedelta(minutes=61)
        assert compute_freshness_state(ts, None, now) == "aging"

    def test_aging_at_179_minutes(self):
        now = self._now()
        ts = now - timedelta(minutes=179)
        assert compute_freshness_state(ts, None, now) == "aging"

    def test_stale_at_exactly_180_minutes(self):
        """Exactly 180 minutes → stale (not aging)."""
        now = self._now()
        ts = now - timedelta(minutes=180)
        assert compute_freshness_state(ts, None, now) == "stale"

    def test_stale_at_181_minutes(self):
        now = self._now()
        ts = now - timedelta(minutes=181)
        assert compute_freshness_state(ts, None, now) == "stale"

    def test_both_none_returns_stale(self):
        """If both timestamps are None, return stale."""
        now = self._now()
        assert compute_freshness_state(None, None, now) == "stale"

    def test_uses_later_timestamp_researcher_newer(self):
        """When researcher is newer, use researcher timestamp."""
        now = self._now()
        researcher_ts = now - timedelta(minutes=30)  # fresh
        breaking_ts = now - timedelta(minutes=200)    # stale on its own
        assert compute_freshness_state(researcher_ts, breaking_ts, now) == "fresh"

    def test_uses_later_timestamp_breaking_newer(self):
        """When breaking news is newer, use breaking news timestamp."""
        now = self._now()
        researcher_ts = now - timedelta(minutes=200)  # stale on its own
        breaking_ts = now - timedelta(minutes=30)     # fresh
        assert compute_freshness_state(researcher_ts, breaking_ts, now) == "fresh"

    def test_only_breaking_news_ts(self):
        """When only breaking_news_ts is provided, use it."""
        now = self._now()
        breaking_ts = now - timedelta(minutes=90)
        assert compute_freshness_state(None, breaking_ts, now) == "aging"

    def test_fresh_at_59_minutes(self):
        now = self._now()
        ts = now - timedelta(minutes=59)
        assert compute_freshness_state(ts, None, now) == "fresh"


# ===================================================================
# Unit tests for compute_confidence
# ===================================================================


class TestComputeConfidence:
    """Unit tests for compute_confidence."""

    def test_high_fresh(self):
        assert compute_confidence("high", "fresh") == 0.9

    def test_high_aging(self):
        assert compute_confidence("high", "aging") == 0.6

    def test_high_stale(self):
        assert compute_confidence("high", "stale") == 0.3

    def test_medium_fresh(self):
        assert compute_confidence("medium", "fresh") == 0.7

    def test_medium_aging(self):
        assert compute_confidence("medium", "aging") == 0.4

    def test_medium_stale(self):
        assert compute_confidence("medium", "stale") == 0.2

    def test_low_fresh(self):
        assert compute_confidence("low", "fresh") == 0.4

    def test_low_aging(self):
        assert compute_confidence("low", "aging") == 0.2

    def test_low_stale(self):
        assert compute_confidence("low", "stale") == 0.0

    def test_unknown_level_returns_zero(self):
        """Unknown confidence level returns 0.0."""
        assert compute_confidence("unknown", "fresh") == 0.0

    def test_unknown_level_logs_warning(self, caplog):
        """Unknown confidence level logs a warning."""
        with caplog.at_level(logging.WARNING):
            compute_confidence("unknown", "fresh")
        assert "Unknown confidence combination" in caplog.text
        assert "unknown" in caplog.text

    def test_unknown_state_returns_zero(self):
        """Unknown freshness state returns 0.0."""
        assert compute_confidence("high", "unknown_state") == 0.0

    def test_all_map_entries_covered(self):
        """Every entry in CONFIDENCE_MAP is reachable."""
        for (level, state), expected in CONFIDENCE_MAP.items():
            assert compute_confidence(level, state) == expected


# ===================================================================
# Fixtures for DB-backed tests
# ===================================================================


@pytest.fixture
def db_session():
    """Create an in-memory SQLite DB with the schema and return a session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# ===================================================================
# Unit tests for get_breaking_news_for_symbols
# ===================================================================


class TestGetBreakingNewsForSymbols:
    """Unit tests for get_breaking_news_for_symbols."""

    def test_returns_alerts_for_matching_symbols(self, db_session):
        """Alerts for watchlist symbols are returned; others are excluded."""
        now = datetime.now(timezone.utc)
        alerts = [
            {"symbol": "TSLA", "headline": "Tesla news", "impact": "bullish", "urgency": "high", "summary": "big"},
            {"symbol": "NVDA", "headline": "Nvidia news", "impact": "bearish", "urgency": "low", "summary": "small"},
            {"symbol": "AAPL", "headline": "Apple news", "impact": "neutral", "urgency": "medium", "summary": "ok"},
        ]
        db_session.add(AgentMemory(
            agent="news_monitor", symbol=None, key="breaking_news",
            value=json.dumps({"alerts": alerts}), timestamp=now,
        ))
        db_session.commit()

        result = get_breaking_news_for_symbols(db_session, ["TSLA", "NVDA"], now - timedelta(hours=1))
        assert len(result["TSLA"]) == 1
        assert result["TSLA"][0]["headline"] == "Tesla news"
        assert len(result["NVDA"]) == 1
        assert result["NVDA"][0]["headline"] == "Nvidia news"

    def test_empty_list_for_symbols_without_alerts(self, db_session):
        """Symbols with no matching alerts get an empty list."""
        now = datetime.now(timezone.utc)
        alerts = [{"symbol": "TSLA", "headline": "Tesla news", "impact": "bullish", "urgency": "high", "summary": "x"}]
        db_session.add(AgentMemory(
            agent="news_monitor", symbol=None, key="breaking_news",
            value=json.dumps({"alerts": alerts}), timestamp=now,
        ))
        db_session.commit()

        result = get_breaking_news_for_symbols(db_session, ["TSLA", "AMD"], now - timedelta(hours=1))
        assert len(result["TSLA"]) == 1
        assert result["AMD"] == []

    def test_ignores_records_before_market_day_start(self, db_session):
        """Records with timestamps before market_day_start are excluded."""
        now = datetime.now(timezone.utc)
        market_day_start = now - timedelta(hours=1)
        old_ts = now - timedelta(hours=2)  # before market_day_start

        alerts = [{"symbol": "TSLA", "headline": "Old news", "impact": "bullish", "urgency": "high", "summary": "x"}]
        db_session.add(AgentMemory(
            agent="news_monitor", symbol=None, key="breaking_news",
            value=json.dumps({"alerts": alerts}), timestamp=old_ts,
        ))
        db_session.commit()

        result = get_breaking_news_for_symbols(db_session, ["TSLA"], market_day_start)
        assert result["TSLA"] == []

    def test_deduplicates_by_headline(self, db_session):
        """Duplicate headlines for the same symbol are deduplicated."""
        now = datetime.now(timezone.utc)
        alerts1 = [{"symbol": "TSLA", "headline": "Same headline", "impact": "bullish", "urgency": "high", "summary": "x"}]
        alerts2 = [{"symbol": "TSLA", "headline": "Same headline", "impact": "bearish", "urgency": "low", "summary": "y"}]
        db_session.add(AgentMemory(
            agent="news_monitor", symbol=None, key="breaking_news",
            value=json.dumps({"alerts": alerts1}), timestamp=now,
        ))
        db_session.add(AgentMemory(
            agent="news_monitor", symbol=None, key="breaking_news",
            value=json.dumps({"alerts": alerts2}), timestamp=now - timedelta(minutes=5),
        ))
        db_session.commit()

        result = get_breaking_news_for_symbols(db_session, ["TSLA"], now - timedelta(hours=1))
        assert len(result["TSLA"]) == 1

    def test_no_records_returns_empty_lists(self, db_session):
        """When no breaking_news records exist, all symbols get empty lists."""
        now = datetime.now(timezone.utc)
        result = get_breaking_news_for_symbols(db_session, ["TSLA", "NVDA"], now - timedelta(hours=1))
        assert result == {"TSLA": [], "NVDA": []}

    def test_malformed_json_is_skipped(self, db_session):
        """Rows with invalid JSON are silently skipped."""
        now = datetime.now(timezone.utc)
        db_session.add(AgentMemory(
            agent="news_monitor", symbol=None, key="breaking_news",
            value="not valid json", timestamp=now,
        ))
        db_session.commit()

        result = get_breaking_news_for_symbols(db_session, ["TSLA"], now - timedelta(hours=1))
        assert result["TSLA"] == []

    def test_multiple_rows_aggregated(self, db_session):
        """Alerts from multiple rows are aggregated."""
        now = datetime.now(timezone.utc)
        alerts1 = [{"symbol": "TSLA", "headline": "News 1", "impact": "bullish", "urgency": "high", "summary": "x"}]
        alerts2 = [{"symbol": "TSLA", "headline": "News 2", "impact": "bearish", "urgency": "low", "summary": "y"}]
        db_session.add(AgentMemory(
            agent="news_monitor", symbol=None, key="breaking_news",
            value=json.dumps({"alerts": alerts1}), timestamp=now,
        ))
        db_session.add(AgentMemory(
            agent="news_monitor", symbol=None, key="breaking_news",
            value=json.dumps({"alerts": alerts2}), timestamp=now - timedelta(minutes=10),
        ))
        db_session.commit()

        result = get_breaking_news_for_symbols(db_session, ["TSLA"], now - timedelta(hours=1))
        assert len(result["TSLA"]) == 2
        headlines = {a["headline"] for a in result["TSLA"]}
        assert headlines == {"News 1", "News 2"}


# ===================================================================
# Unit tests for get_researcher_timestamps
# ===================================================================


class TestGetResearcherTimestamps:
    """Unit tests for get_researcher_timestamps."""

    def test_returns_timestamp_and_confidence(self, db_session):
        """Returns the most recent timestamp and confidence for each symbol."""
        now = datetime.now(timezone.utc)
        data = {"sentiment": "bullish", "confidence": "high", "catalysts": [], "risks": [], "summary": "good"}
        db_session.add(AgentMemory(
            agent="researcher", symbol="TSLA", key="sentiment",
            value=json.dumps(data), timestamp=now,
        ))
        db_session.commit()

        result = get_researcher_timestamps(db_session, ["TSLA"])
        assert "TSLA" in result
        ts, conf = result["TSLA"]
        assert conf == "high"
        # SQLite stores naive datetimes, so compare without tzinfo
        assert ts.replace(tzinfo=None) == now.replace(tzinfo=None)

    def test_returns_most_recent_record(self, db_session):
        """When multiple records exist, returns the most recent one."""
        now = datetime.now(timezone.utc)
        old_data = {"sentiment": "bearish", "confidence": "low", "catalysts": [], "risks": [], "summary": "bad"}
        new_data = {"sentiment": "bullish", "confidence": "medium", "catalysts": [], "risks": [], "summary": "ok"}
        db_session.add(AgentMemory(
            agent="researcher", symbol="TSLA", key="sentiment",
            value=json.dumps(old_data), timestamp=now - timedelta(hours=2),
        ))
        db_session.add(AgentMemory(
            agent="researcher", symbol="TSLA", key="sentiment",
            value=json.dumps(new_data), timestamp=now,
        ))
        db_session.commit()

        result = get_researcher_timestamps(db_session, ["TSLA"])
        _, conf = result["TSLA"]
        assert conf == "medium"

    def test_missing_symbol_not_in_result(self, db_session):
        """Symbols with no researcher data are absent from the result."""
        result = get_researcher_timestamps(db_session, ["TSLA", "NVDA"])
        assert result == {}

    def test_defaults_confidence_to_low(self, db_session):
        """When confidence field is missing from JSON, defaults to 'low'."""
        now = datetime.now(timezone.utc)
        data = {"sentiment": "neutral", "catalysts": [], "risks": [], "summary": "meh"}
        db_session.add(AgentMemory(
            agent="researcher", symbol="TSLA", key="sentiment",
            value=json.dumps(data), timestamp=now,
        ))
        db_session.commit()

        result = get_researcher_timestamps(db_session, ["TSLA"])
        _, conf = result["TSLA"]
        assert conf == "low"

    def test_malformed_json_is_skipped(self, db_session):
        """Rows with invalid JSON are silently skipped."""
        now = datetime.now(timezone.utc)
        db_session.add(AgentMemory(
            agent="researcher", symbol="TSLA", key="sentiment",
            value="not json", timestamp=now,
        ))
        db_session.commit()

        result = get_researcher_timestamps(db_session, ["TSLA"])
        assert "TSLA" not in result

    def test_multiple_symbols(self, db_session):
        """Returns data for multiple symbols independently."""
        now = datetime.now(timezone.utc)
        tsla_data = {"sentiment": "bullish", "confidence": "high", "catalysts": [], "risks": [], "summary": "x"}
        nvda_data = {"sentiment": "bearish", "confidence": "low", "catalysts": [], "risks": [], "summary": "y"}
        db_session.add(AgentMemory(
            agent="researcher", symbol="TSLA", key="sentiment",
            value=json.dumps(tsla_data), timestamp=now,
        ))
        db_session.add(AgentMemory(
            agent="researcher", symbol="NVDA", key="sentiment",
            value=json.dumps(nvda_data), timestamp=now - timedelta(hours=1),
        ))
        db_session.commit()

        result = get_researcher_timestamps(db_session, ["TSLA", "NVDA"])
        assert result["TSLA"][1] == "high"
        assert result["NVDA"][1] == "low"


# ===================================================================
# Unit tests for compute_catalyst_freshness
# ===================================================================


class TestComputeCatalystFreshness:
    """Unit tests for compute_catalyst_freshness (task 1.4)."""

    def test_returns_dict_for_all_symbols(self, db_session):
        """Returns a freshness dict for every requested symbol."""
        now = datetime.now(timezone.utc)
        result = compute_catalyst_freshness(db_session, ["TSLA", "NVDA"], now=now)
        assert "TSLA" in result
        assert "NVDA" in result

    def test_keys_present_in_result(self, db_session):
        """Each symbol dict has the required keys."""
        now = datetime.now(timezone.utc)
        result = compute_catalyst_freshness(db_session, ["TSLA"], now=now)
        entry = result["TSLA"]
        assert "last_researcher_update" in entry
        assert "last_breaking_news_update" in entry
        assert "freshness_state" in entry
        assert "source_type" in entry
        assert "confidence" in entry

    def test_stale_when_no_data(self, db_session):
        """Symbol with no researcher or breaking news data → stale."""
        now = datetime.now(timezone.utc)
        result = compute_catalyst_freshness(db_session, ["TSLA"], now=now)
        assert result["TSLA"]["freshness_state"] == "stale"
        assert result["TSLA"]["last_researcher_update"] is None
        assert result["TSLA"]["last_breaking_news_update"] is None

    def test_source_type_premarket_when_no_breaking_news(self, db_session):
        """Source type is premarket_synthesis when no breaking news exists."""
        now = datetime.now(timezone.utc)
        # Add researcher data only
        data = {"sentiment": "bullish", "confidence": "high", "catalysts": [], "risks": [], "summary": "x"}
        db_session.add(AgentMemory(
            agent="researcher", symbol="TSLA", key="sentiment",
            value=json.dumps(data), timestamp=now - timedelta(minutes=30),
        ))
        db_session.commit()

        result = compute_catalyst_freshness(db_session, ["TSLA"], now=now)
        assert result["TSLA"]["source_type"] == "premarket_synthesis"

    def test_source_type_intraday_when_breaking_news_exists(self, db_session):
        """Source type is intraday_alert when breaking news exists for current market day."""
        now = datetime.now(timezone.utc)
        now_et = now.astimezone(ET)
        # Add breaking news for the current market day
        alerts = [{"symbol": "TSLA", "headline": "Tesla news", "impact": "bullish", "urgency": "high", "summary": "x"}]
        db_session.add(AgentMemory(
            agent="news_monitor", symbol=None, key="breaking_news",
            value=json.dumps({"alerts": alerts}), timestamp=now,
        ))
        db_session.commit()

        result = compute_catalyst_freshness(db_session, ["TSLA"], now=now)
        assert result["TSLA"]["source_type"] == "intraday_alert"

    def test_confidence_uses_researcher_level(self, db_session):
        """Confidence is computed from researcher confidence level and freshness state."""
        now = datetime.now(timezone.utc)
        data = {"sentiment": "bullish", "confidence": "high", "catalysts": [], "risks": [], "summary": "x"}
        db_session.add(AgentMemory(
            agent="researcher", symbol="TSLA", key="sentiment",
            value=json.dumps(data), timestamp=now - timedelta(minutes=30),
        ))
        db_session.commit()

        result = compute_catalyst_freshness(db_session, ["TSLA"], now=now)
        # 30 min old + high confidence → fresh → 0.9
        assert result["TSLA"]["confidence"] == 0.9

    def test_confidence_zero_when_no_researcher(self, db_session):
        """Confidence is 0.0 when no researcher data exists (defaults to low + stale)."""
        now = datetime.now(timezone.utc)
        result = compute_catalyst_freshness(db_session, ["TSLA"], now=now)
        assert result["TSLA"]["confidence"] == 0.0

    def test_uses_prefetched_breaking_news(self, db_session):
        """When breaking_news_by_symbol is provided, uses it instead of querying DB."""
        now = datetime.now(timezone.utc)
        prefetched = {
            "TSLA": [{"symbol": "TSLA", "headline": "Prefetched news", "impact": "bullish", "urgency": "high", "summary": "x"}],
        }
        result = compute_catalyst_freshness(
            db_session, ["TSLA"], now=now, breaking_news_by_symbol=prefetched
        )
        assert result["TSLA"]["source_type"] == "intraday_alert"
        assert result["TSLA"]["last_breaking_news_update"] is not None

    def test_researcher_timestamp_is_iso_string(self, db_session):
        """last_researcher_update is an ISO format string."""
        now = datetime.now(timezone.utc)
        data = {"sentiment": "bullish", "confidence": "medium", "catalysts": [], "risks": [], "summary": "x"}
        db_session.add(AgentMemory(
            agent="researcher", symbol="TSLA", key="sentiment",
            value=json.dumps(data), timestamp=now,
        ))
        db_session.commit()

        result = compute_catalyst_freshness(db_session, ["TSLA"], now=now)
        iso_str = result["TSLA"]["last_researcher_update"]
        assert iso_str is not None
        # Should be parseable as ISO
        parsed = datetime.fromisoformat(iso_str)
        assert parsed is not None

    def test_multiple_symbols_independent(self, db_session):
        """Each symbol is computed independently."""
        now = datetime.now(timezone.utc)
        # TSLA has recent researcher data → fresh
        tsla_data = {"sentiment": "bullish", "confidence": "high", "catalysts": [], "risks": [], "summary": "x"}
        db_session.add(AgentMemory(
            agent="researcher", symbol="TSLA", key="sentiment",
            value=json.dumps(tsla_data), timestamp=now - timedelta(minutes=10),
        ))
        # NVDA has old researcher data → stale
        nvda_data = {"sentiment": "bearish", "confidence": "low", "catalysts": [], "risks": [], "summary": "y"}
        db_session.add(AgentMemory(
            agent="researcher", symbol="NVDA", key="sentiment",
            value=json.dumps(nvda_data), timestamp=now - timedelta(hours=4),
        ))
        db_session.commit()

        result = compute_catalyst_freshness(db_session, ["TSLA", "NVDA"], now=now)
        assert result["TSLA"]["freshness_state"] == "fresh"
        assert result["NVDA"]["freshness_state"] == "stale"


# ===================================================================
# Unit tests for build_freshness_label
# ===================================================================


class TestBuildFreshnessLabel:
    """Unit tests for build_freshness_label (task 1.4)."""

    def test_label_contains_symbol(self):
        """Label contains the symbol name."""
        freshness = {
            "last_researcher_update": "2025-01-15T13:30:00+00:00",
            "last_breaking_news_update": None,
            "freshness_state": "fresh",
            "source_type": "premarket_synthesis",
            "confidence": 0.9,
        }
        label = build_freshness_label("TSLA", freshness)
        assert "TSLA" in label

    def test_label_contains_freshness_state(self):
        """Label contains the freshness state."""
        freshness = {
            "last_researcher_update": "2025-01-15T13:30:00+00:00",
            "last_breaking_news_update": None,
            "freshness_state": "aging",
            "source_type": "premarket_synthesis",
            "confidence": 0.6,
        }
        label = build_freshness_label("TSLA", freshness)
        assert "aging" in label

    def test_label_no_breaking_news(self):
        """When no breaking news, label says 'no intraday news updates'."""
        freshness = {
            "last_researcher_update": "2025-01-15T13:30:00+00:00",
            "last_breaking_news_update": None,
            "freshness_state": "fresh",
            "source_type": "premarket_synthesis",
            "confidence": 0.9,
        }
        label = build_freshness_label("TSLA", freshness)
        assert "no intraday news updates" in label

    def test_label_with_breaking_news_time(self):
        """When breaking news exists, label contains a time in HH:MM AM/PM format."""
        freshness = {
            "last_researcher_update": "2025-01-15T13:30:00+00:00",
            "last_breaking_news_update": "2025-01-15T17:05:00+00:00",
            "freshness_state": "fresh",
            "source_type": "intraday_alert",
            "confidence": 0.7,
        }
        label = build_freshness_label("TSLA", freshness)
        # 17:05 UTC → 12:05 PM ET
        assert "12:05 PM" in label
        assert "no intraday news updates" not in label

    def test_label_source_description_premarket(self):
        """Label contains 'premarket synthesis' for premarket_synthesis source type."""
        freshness = {
            "last_researcher_update": "2025-01-15T13:30:00+00:00",
            "last_breaking_news_update": None,
            "freshness_state": "fresh",
            "source_type": "premarket_synthesis",
            "confidence": 0.9,
        }
        label = build_freshness_label("TSLA", freshness)
        assert "premarket synthesis" in label

    def test_label_source_description_intraday(self):
        """Label contains 'intraday alert' for intraday_alert source type."""
        freshness = {
            "last_researcher_update": "2025-01-15T13:30:00+00:00",
            "last_breaking_news_update": "2025-01-15T17:05:00+00:00",
            "freshness_state": "fresh",
            "source_type": "intraday_alert",
            "confidence": 0.7,
        }
        label = build_freshness_label("TSLA", freshness)
        assert "intraday alert" in label

    def test_label_researcher_time_in_et(self):
        """Researcher time is displayed in ET HH:MM AM/PM format."""
        # 13:30 UTC = 8:30 AM ET (during EST)
        freshness = {
            "last_researcher_update": "2025-01-15T13:30:00+00:00",
            "last_breaking_news_update": None,
            "freshness_state": "fresh",
            "source_type": "premarket_synthesis",
            "confidence": 0.9,
        }
        label = build_freshness_label("TSLA", freshness)
        assert "8:30 AM" in label

    def test_label_empty_freshness_dict(self):
        """Empty freshness dict returns unavailable message."""
        label = build_freshness_label("TSLA", {})
        assert "TSLA" in label
        assert "unavailable" in label

    def test_label_format_structure(self):
        """Label follows the expected format pattern."""
        freshness = {
            "last_researcher_update": "2025-01-15T13:30:00+00:00",
            "last_breaking_news_update": None,
            "freshness_state": "stale",
            "source_type": "premarket_synthesis",
            "confidence": 0.3,
        }
        label = build_freshness_label("TSLA", freshness)
        assert "catalyst view is based on" in label
        assert "last intraday news update was" in label
        assert "catalyst freshness is" in label
        assert label.endswith(".")
