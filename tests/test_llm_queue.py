"""Unit tests for QueueConfig environment variable loading and MarketSessionHelper."""

from __future__ import annotations

import os
from datetime import datetime, time
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import pytest

from utils.llm_queue.config import QueueConfig
from utils.llm_queue.market_session import MarketSessionHelper


class TestQueueConfigDefaults:
    """Verify safe defaults when no env vars are set."""

    def test_default_mode_is_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.mode == "disabled"

    def test_default_global_concurrency(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.global_concurrency == 1

    def test_default_max_queue_size(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.max_queue_size == 10

    def test_default_market_hour_deadline_factor(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.market_hour_deadline_factor == 0.5

    def test_default_prompt_token_warn_threshold(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.prompt_token_warn_threshold == 8000

    def test_default_prompt_token_reject_threshold(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.prompt_token_reject_threshold == 16000

    def test_default_fallback_deadline_buffer_seconds(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.fallback_deadline_buffer_seconds == 15.0

    def test_default_per_model_concurrency_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.per_model_concurrency == {}

    def test_default_deadlines_match_spec(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.deadlines["execution_critical"] == 120.0
            assert cfg.deadlines["market_analysis"] == 180.0
            assert cfg.deadlines["position_management"] == 180.0
            assert cfg.deadlines["repair"] == 60.0
            assert cfg.deadlines["research"] == 600.0
            assert cfg.deadlines["review"] == 600.0
            assert cfg.deadlines["advisory"] == 900.0
            assert cfg.deadlines["startup_probe"] == 30.0

    def test_default_max_queue_waits_match_spec(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.max_queue_waits["execution_critical"] == 30.0
            assert cfg.max_queue_waits["market_analysis"] == 60.0
            assert cfg.max_queue_waits["position_management"] == 60.0
            assert cfg.max_queue_waits["repair"] == 15.0
            assert cfg.max_queue_waits["research"] == 180.0
            assert cfg.max_queue_waits["review"] == 180.0
            assert cfg.max_queue_waits["advisory"] == 300.0
            assert cfg.max_queue_waits["startup_probe"] == 10.0

    def test_default_stale_after_match_spec(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.stale_after["execution_critical"] == 60.0
            assert cfg.stale_after["market_analysis"] == 120.0
            assert cfg.stale_after["position_management"] == 120.0
            assert cfg.stale_after["repair"] == 30.0
            assert cfg.stale_after["research"] == 300.0
            assert cfg.stale_after["review"] == 300.0
            assert cfg.stale_after["advisory"] == 600.0
            assert cfg.stale_after["startup_probe"] == 15.0


class TestQueueConfigModeValidation:
    """Verify mode env var handling."""

    def test_enforcing_mode(self):
        with patch.dict(os.environ, {"LLM_QUEUE_MODE": "enforcing"}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.mode == "enforcing"

    def test_observe_mode(self):
        with patch.dict(os.environ, {"LLM_QUEUE_MODE": "observe"}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.mode == "observe"

    def test_invalid_mode_defaults_to_disabled(self):
        with patch.dict(os.environ, {"LLM_QUEUE_MODE": "invalid_value"}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.mode == "disabled"

    def test_empty_mode_defaults_to_disabled(self):
        with patch.dict(os.environ, {"LLM_QUEUE_MODE": ""}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.mode == "disabled"


class TestQueueConfigGlobalConcurrency:
    """Verify LLM_QUEUE_GLOBAL_CONCURRENCY parsing."""

    def test_valid_concurrency(self):
        with patch.dict(os.environ, {"LLM_QUEUE_GLOBAL_CONCURRENCY": "3"}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.global_concurrency == 3

    def test_negative_concurrency_produces_default(self):
        with patch.dict(os.environ, {"LLM_QUEUE_GLOBAL_CONCURRENCY": "-1"}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.global_concurrency == 1

    def test_zero_concurrency_produces_default(self):
        with patch.dict(os.environ, {"LLM_QUEUE_GLOBAL_CONCURRENCY": "0"}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.global_concurrency == 1

    def test_non_numeric_concurrency_produces_default(self):
        with patch.dict(os.environ, {"LLM_QUEUE_GLOBAL_CONCURRENCY": "abc"}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.global_concurrency == 1


class TestQueueConfigModelConcurrency:
    """Verify LLM_QUEUE_MODEL_CONCURRENCY JSON parsing."""

    def test_valid_json_model_concurrency(self):
        env = {"LLM_QUEUE_MODEL_CONCURRENCY": '{"model1": 2}'}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.per_model_concurrency == {"model1": 2}

    def test_multi_model_concurrency(self):
        env = {"LLM_QUEUE_MODEL_CONCURRENCY": '{"model1": 2, "model2": 4}'}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.per_model_concurrency == {"model1": 2, "model2": 4}

    def test_invalid_json_produces_empty_dict(self):
        env = {"LLM_QUEUE_MODEL_CONCURRENCY": "invalid_json"}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.per_model_concurrency == {}

    def test_non_object_json_produces_empty_dict(self):
        env = {"LLM_QUEUE_MODEL_CONCURRENCY": "[1, 2, 3]"}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.per_model_concurrency == {}


class TestQueueConfigPerClassDeadlineOverride:
    """Verify per-class deadline env var overrides."""

    def test_deadline_override_execution_critical(self):
        env = {"LLM_QUEUE_DEADLINE_EXECUTION_CRITICAL": "90"}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.deadlines["execution_critical"] == 90.0

    def test_deadline_override_does_not_affect_other_classes(self):
        env = {"LLM_QUEUE_DEADLINE_EXECUTION_CRITICAL": "90"}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.deadlines["advisory"] == 900.0

    def test_invalid_deadline_keeps_default(self):
        env = {"LLM_QUEUE_DEADLINE_EXECUTION_CRITICAL": "not_a_number"}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.deadlines["execution_critical"] == 120.0

    def test_negative_deadline_keeps_default(self):
        env = {"LLM_QUEUE_DEADLINE_EXECUTION_CRITICAL": "-10"}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.deadlines["execution_critical"] == 120.0


class TestQueueConfigPerClassStaleOverride:
    """Verify per-class stale_after env var overrides."""

    def test_stale_override_advisory(self):
        env = {"LLM_QUEUE_STALE_ADVISORY": "1200"}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.stale_after["advisory"] == 1200.0

    def test_stale_override_does_not_affect_other_classes(self):
        env = {"LLM_QUEUE_STALE_ADVISORY": "1200"}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.stale_after["execution_critical"] == 60.0


class TestQueueConfigFallbackModels:
    """Verify approved fallback model env var parsing."""

    def test_fallback_models_comma_separated(self):
        env = {"LLM_QUEUE_FALLBACK_MODELS_EXECUTION_CRITICAL": "claude-3-haiku,gpt-4o-mini"}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.approved_fallback_models["execution_critical"] == [
                "claude-3-haiku",
                "gpt-4o-mini",
            ]

    def test_fallback_models_empty_produces_empty_list(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.approved_fallback_models["execution_critical"] == []

    def test_fallback_models_whitespace_trimmed(self):
        env = {"LLM_QUEUE_FALLBACK_MODELS_EXECUTION_CRITICAL": " claude-3-haiku , gpt-4o-mini "}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.approved_fallback_models["execution_critical"] == [
                "claude-3-haiku",
                "gpt-4o-mini",
            ]


class TestQueueConfigMaxQueueSizeByClass:
    """Verify per-class max queue size env var overrides."""

    def test_max_size_by_class(self):
        env = {"LLM_QUEUE_MAX_SIZE_ADVISORY": "3"}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert cfg.max_queue_size_by_class["advisory"] == 3

    def test_max_size_by_class_invalid_skipped(self):
        env = {"LLM_QUEUE_MAX_SIZE_ADVISORY": "abc"}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert "advisory" not in cfg.max_queue_size_by_class

    def test_max_size_by_class_negative_skipped(self):
        env = {"LLM_QUEUE_MAX_SIZE_ADVISORY": "-2"}
        with patch.dict(os.environ, env, clear=True):
            cfg = QueueConfig.from_environment()
            assert "advisory" not in cfg.max_queue_size_by_class


# ---------------------------------------------------------------------------
# MarketSessionHelper Tests
# ---------------------------------------------------------------------------

_ET = ZoneInfo("America/New_York")


class TestMarketSessionHelperCurrentSession:
    """Test current_session() returns correct session for various times."""

    def _make_helper_with_time(self, dt: datetime) -> str:
        """Use internal method to test specific datetimes."""
        helper = MarketSessionHelper()
        return helper._session_for_datetime(dt)

    def test_premarket_start_boundary(self):
        # Monday 4:00 AM ET -> premarket
        dt = datetime(2024, 1, 8, 4, 0, 0, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "premarket"

    def test_premarket_mid(self):
        # Wednesday 7:00 AM ET -> premarket
        dt = datetime(2024, 1, 10, 7, 0, 0, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "premarket"

    def test_premarket_end_boundary(self):
        # Tuesday 9:29:59 ET -> premarket
        dt = datetime(2024, 1, 9, 9, 29, 59, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "premarket"

    def test_regular_open_boundary(self):
        # Monday 9:30 AM ET -> regular
        dt = datetime(2024, 1, 8, 9, 30, 0, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "regular"

    def test_regular_mid(self):
        # Thursday 12:00 PM ET -> regular
        dt = datetime(2024, 1, 11, 12, 0, 0, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "regular"

    def test_regular_end_boundary(self):
        # Friday 15:59:59 ET -> regular
        dt = datetime(2024, 1, 12, 15, 59, 59, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "regular"

    def test_postmarket_open_boundary(self):
        # Monday 16:00 ET -> postmarket
        dt = datetime(2024, 1, 8, 16, 0, 0, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "postmarket"

    def test_postmarket_mid(self):
        # Tuesday 18:00 ET -> postmarket
        dt = datetime(2024, 1, 9, 18, 0, 0, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "postmarket"

    def test_postmarket_end_boundary(self):
        # Wednesday 19:59:59 ET -> postmarket
        dt = datetime(2024, 1, 10, 19, 59, 59, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "postmarket"

    def test_closed_after_postmarket(self):
        # Thursday 20:00 ET -> closed
        dt = datetime(2024, 1, 11, 20, 0, 0, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "closed"

    def test_closed_late_night(self):
        # Friday 23:00 ET -> closed
        dt = datetime(2024, 1, 12, 23, 0, 0, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "closed"

    def test_closed_early_morning(self):
        # Monday 3:59 AM ET -> closed
        dt = datetime(2024, 1, 8, 3, 59, 0, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "closed"

    def test_saturday_always_closed(self):
        # Saturday 10:00 AM ET -> closed (even during would-be regular hours)
        dt = datetime(2024, 1, 13, 10, 0, 0, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "closed"

    def test_sunday_always_closed(self):
        # Sunday 14:00 ET -> closed
        dt = datetime(2024, 1, 14, 14, 0, 0, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "closed"

    def test_saturday_premarket_time_still_closed(self):
        # Saturday 5:00 AM ET -> closed (weekend overrides time-of-day)
        dt = datetime(2024, 1, 13, 5, 0, 0, tzinfo=_ET)
        assert self._make_helper_with_time(dt) == "closed"


class TestMarketSessionHelperIsMarketHours:
    """Test is_market_hours() returns True only during regular session."""

    def _is_regular(self, dt: datetime) -> bool:
        helper = MarketSessionHelper()
        return helper._is_regular_session(dt)

    def test_regular_hours_true(self):
        dt = datetime(2024, 1, 8, 12, 0, 0, tzinfo=_ET)
        assert self._is_regular(dt) is True

    def test_premarket_false(self):
        dt = datetime(2024, 1, 8, 7, 0, 0, tzinfo=_ET)
        assert self._is_regular(dt) is False

    def test_postmarket_false(self):
        dt = datetime(2024, 1, 8, 17, 0, 0, tzinfo=_ET)
        assert self._is_regular(dt) is False

    def test_closed_false(self):
        dt = datetime(2024, 1, 8, 22, 0, 0, tzinfo=_ET)
        assert self._is_regular(dt) is False

    def test_weekend_false(self):
        dt = datetime(2024, 1, 13, 12, 0, 0, tzinfo=_ET)
        assert self._is_regular(dt) is False

    def test_boundary_930_true(self):
        dt = datetime(2024, 1, 8, 9, 30, 0, tzinfo=_ET)
        assert self._is_regular(dt) is True

    def test_boundary_1600_false(self):
        dt = datetime(2024, 1, 8, 16, 0, 0, tzinfo=_ET)
        assert self._is_regular(dt) is False


class TestMarketSessionHelperLiveCall:
    """Test that the live current_session() and is_market_hours() methods work."""

    def test_current_session_returns_valid_string(self):
        helper = MarketSessionHelper()
        session = helper.current_session()
        assert session in ("regular", "premarket", "postmarket", "closed")

    def test_is_market_hours_returns_bool(self):
        helper = MarketSessionHelper()
        result = helper.is_market_hours()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# RequestClassifier Purpose Prefix Mapping Tests
# ---------------------------------------------------------------------------


class TestRequestClassifierPrefixMappings:
    """Verify all known purpose prefix → (agent, request_class, priority) mappings."""

    @pytest.fixture
    def classifier(self):
        """Create a RequestClassifier with default config and mocked market session."""
        cfg = QueueConfig()
        from utils.llm_queue.classifier import RequestClassifier

        return RequestClassifier(cfg)

    def _classify(self, classifier, purpose, request_class_override=None):
        """Classify a purpose string with deterministic market session = closed."""
        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value="closed",
        ):
            return classifier.classify(
                purpose=purpose,
                tier="medium",
                provider="ollama",
                model="llama3",
                json_mode=False,
                prompt_chars=1000,
                request_class_override=request_class_override,
            )

    # --- pm agent: execution_critical (priority 0) ---

    def test_pm_entry(self, classifier):
        rec = self._classify(classifier, "pm_entry")
        assert rec.agent == "pm"
        assert rec.request_class == "execution_critical"
        assert rec.priority == 0

    def test_pm_candidate(self, classifier):
        rec = self._classify(classifier, "pm_candidate")
        assert rec.agent == "pm"
        assert rec.request_class == "execution_critical"
        assert rec.priority == 0

    # --- pm agent: position_management (priority 2) ---

    def test_pm_maintenance(self, classifier):
        rec = self._classify(classifier, "pm_maintenance")
        assert rec.agent == "pm"
        assert rec.request_class == "position_management"
        assert rec.priority == 2

    def test_pm_reversal(self, classifier):
        rec = self._classify(classifier, "pm_reversal")
        assert rec.agent == "pm"
        assert rec.request_class == "position_management"
        assert rec.priority == 2

    # --- analyst agent: market_analysis (priority 1) ---

    def test_analyst_signal(self, classifier):
        rec = self._classify(classifier, "analyst_signal")
        assert rec.agent == "analyst"
        assert rec.request_class == "market_analysis"
        assert rec.priority == 1

    def test_analyst_veto(self, classifier):
        rec = self._classify(classifier, "analyst_veto")
        assert rec.agent == "analyst"
        assert rec.request_class == "market_analysis"
        assert rec.priority == 1

    # --- price_monitor agent: market_analysis (priority 1) ---

    def test_price_monitor_filter(self, classifier):
        rec = self._classify(classifier, "price_monitor_filter")
        assert rec.agent == "price_monitor"
        assert rec.request_class == "market_analysis"
        assert rec.priority == 1

    # --- system agent: repair (priority 3) ---

    def test_json_repair(self, classifier):
        rec = self._classify(classifier, "json_repair")
        assert rec.agent == "system"
        assert rec.request_class == "repair"
        assert rec.priority == 3

    # --- researcher/quant/scout agents: research (priority 4) ---

    def test_researcher_premarket(self, classifier):
        rec = self._classify(classifier, "researcher_premarket")
        assert rec.agent == "researcher"
        assert rec.request_class == "research"
        assert rec.priority == 4

    def test_quant_researcher(self, classifier):
        rec = self._classify(classifier, "quant_researcher")
        assert rec.agent == "quant"
        assert rec.request_class == "research"
        assert rec.priority == 4

    def test_sector_scout(self, classifier):
        rec = self._classify(classifier, "sector_scout")
        assert rec.agent == "scout"
        assert rec.request_class == "research"
        assert rec.priority == 4

    # --- reviewer agent: review (priority 5) ---

    def test_reviewer_trade(self, classifier):
        rec = self._classify(classifier, "reviewer_trade")
        assert rec.agent == "reviewer"
        assert rec.request_class == "review"
        assert rec.priority == 5

    def test_daily_review(self, classifier):
        rec = self._classify(classifier, "daily_review")
        assert rec.agent == "reviewer"
        assert rec.request_class == "review"
        assert rec.priority == 5

    def test_meta_reviewer(self, classifier):
        rec = self._classify(classifier, "meta_reviewer")
        assert rec.agent == "reviewer"
        assert rec.request_class == "review"
        assert rec.priority == 5

    # --- ceo/narrator agents: advisory (priority 6) ---

    def test_ceo(self, classifier):
        rec = self._classify(classifier, "ceo")
        assert rec.agent == "ceo"
        assert rec.request_class == "advisory"
        assert rec.priority == 6

    def test_weekly_prep(self, classifier):
        rec = self._classify(classifier, "weekly_prep")
        assert rec.agent == "ceo"
        assert rec.request_class == "advisory"
        assert rec.priority == 6

    def test_narrator(self, classifier):
        rec = self._classify(classifier, "narrator")
        assert rec.agent == "narrator"
        assert rec.request_class == "advisory"
        assert rec.priority == 6

    # --- system agent: startup_probe (priority 7) ---

    def test_startup_probe(self, classifier):
        rec = self._classify(classifier, "startup_probe")
        assert rec.agent == "system"
        assert rec.request_class == "startup_probe"
        assert rec.priority == 7

    # --- Unknown purpose → defaults ---

    def test_unknown_purpose_defaults(self, classifier):
        rec = self._classify(classifier, "completely_unknown_purpose")
        assert rec.agent == "unknown"
        assert rec.request_class == "advisory"
        assert rec.priority == 6

    # --- Suffix after prefix does not affect classification ---

    def test_suffix_does_not_affect_pm_entry(self, classifier):
        rec = self._classify(classifier, "pm_entry_AAPL")
        assert rec.agent == "pm"
        assert rec.request_class == "execution_critical"
        assert rec.priority == 0

    def test_suffix_does_not_affect_analyst_signal(self, classifier):
        rec = self._classify(classifier, "analyst_signal_strong_buy_MSFT")
        assert rec.agent == "analyst"
        assert rec.request_class == "market_analysis"
        assert rec.priority == 1

    def test_suffix_does_not_affect_researcher_premarket(self, classifier):
        rec = self._classify(classifier, "researcher_premarket_sector_scan")
        assert rec.agent == "researcher"
        assert rec.request_class == "research"
        assert rec.priority == 4

    # --- request_class_override works ---

    def test_request_class_override_changes_class_and_priority(self, classifier):
        rec = self._classify(
            classifier, "narrator", request_class_override="execution_critical"
        )
        assert rec.agent == "narrator"
        assert rec.request_class == "execution_critical"
        assert rec.priority == 0

    def test_request_class_override_invalid_value_ignored(self, classifier):
        rec = self._classify(
            classifier, "narrator", request_class_override="nonexistent_class"
        )
        assert rec.agent == "narrator"
        assert rec.request_class == "advisory"
        assert rec.priority == 6


# ---------------------------------------------------------------------------
# PriorityQueue Eviction and Depth Tracking Tests
# ---------------------------------------------------------------------------

import uuid
from datetime import timedelta, timezone

from utils.llm_queue.priority_queue import PriorityQueue
from utils.llm_queue.models import RequestRecord


def _make_record(priority: int, request_class: str = "test", created_at=None):
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    return RequestRecord(
        request_id=str(uuid.uuid4()),
        purpose="test",
        agent="test",
        request_class=request_class,
        tier="medium",
        provider="ollama",
        model="llama3",
        json_mode=False,
        priority=priority,
        deadline_seconds=120.0,
        max_queue_wait_seconds=30.0,
        fallback_policy="none",
        stale_after_seconds=60.0,
        created_at=created_at,
        prompt_chars=100,
        approx_prompt_tokens=25,
        market_session="closed",
    )


class TestPriorityQueueEvictionAndDepth:
    """Unit tests for queue eviction and depth tracking."""

    def test_evict_lowest_removes_lowest_priority_item(self):
        """evict_lowest() removes the record with the highest priority value (lowest urgency)."""
        q = PriorityQueue(QueueConfig())
        r0 = _make_record(priority=0)
        r3 = _make_record(priority=3)
        r6 = _make_record(priority=6)
        q.put(r0)
        q.put(r3)
        q.put(r6)

        evicted = q.evict_lowest()
        assert evicted is r6
        assert q.depth() == 2

    def test_evict_lowest_fifo_tiebreak(self):
        """Within same priority, evict_lowest() removes the LATER record (less urgent)."""
        q = PriorityQueue(QueueConfig())
        t1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = t1 + timedelta(seconds=5)
        r_early = _make_record(priority=5, created_at=t1)
        r_late = _make_record(priority=5, created_at=t2)
        q.put(r_early)
        q.put(r_late)

        evicted = q.evict_lowest()
        assert evicted is r_late

    def test_evict_lowest_empty_returns_none(self):
        """evict_lowest() on empty queue returns None."""
        q = PriorityQueue(QueueConfig())
        assert q.evict_lowest() is None

    def test_depth_by_class_accurate(self):
        """depth_by_class() returns correct per-class counts."""
        q = PriorityQueue(QueueConfig())
        q.put(_make_record(priority=0, request_class="execution_critical"))
        q.put(_make_record(priority=1, request_class="market_analysis"))
        q.put(_make_record(priority=1, request_class="market_analysis"))
        q.put(_make_record(priority=6, request_class="advisory"))

        counts = q.depth_by_class()
        assert counts == {
            "execution_critical": 1,
            "market_analysis": 2,
            "advisory": 1,
        }

    def test_depth_by_class_empty(self):
        """Empty queue returns empty dict from depth_by_class()."""
        q = PriorityQueue(QueueConfig())
        assert q.depth_by_class() == {}

    def test_depth_increments_on_put(self):
        """depth() increases with each put()."""
        q = PriorityQueue(QueueConfig())
        assert q.depth() == 0
        q.put(_make_record(priority=1))
        assert q.depth() == 1
        q.put(_make_record(priority=2))
        assert q.depth() == 2
        q.put(_make_record(priority=3))
        assert q.depth() == 3

    def test_depth_decrements_on_get(self):
        """depth() decreases with each get()."""
        q = PriorityQueue(QueueConfig())
        q.put(_make_record(priority=1))
        q.put(_make_record(priority=2))
        q.put(_make_record(priority=3))
        assert q.depth() == 3

        q.get(timeout=0.1)
        assert q.depth() == 2
        q.get(timeout=0.1)
        assert q.depth() == 1
        q.get(timeout=0.1)
        assert q.depth() == 0

    def test_get_timeout_returns_none(self):
        """get(timeout=0.05) on empty queue returns None quickly."""
        q = PriorityQueue(QueueConfig())
        result = q.get(timeout=0.05)
        assert result is None

    def test_peek_lowest_priority_does_not_remove(self):
        """peek_lowest_priority() returns the lowest-priority item but depth stays the same."""
        q = PriorityQueue(QueueConfig())
        r0 = _make_record(priority=0)
        r6 = _make_record(priority=6)
        q.put(r0)
        q.put(r6)

        peeked = q.peek_lowest_priority()
        assert peeked is r6
        assert q.depth() == 2



# ---------------------------------------------------------------------------
# DeadlineEnforcer Tests
# ---------------------------------------------------------------------------

from utils.llm_queue.deadline import DeadlineEnforcer


class TestDeadlineEnforcerCheckQueueWait:
    """Test check_queue_wait returns (expired, elapsed) correctly."""

    @pytest.fixture
    def enforcer(self):
        return DeadlineEnforcer(QueueConfig())

    def test_not_expired_when_within_limit(self, enforcer):
        """Request created just now should not have expired."""
        record = _make_record(priority=0)
        expired, elapsed = enforcer.check_queue_wait(record)
        assert expired is False
        assert elapsed >= 0.0
        assert elapsed < 1.0  # should be near-instant

    def test_expired_when_past_max_queue_wait(self, enforcer):
        """Request created far in the past should be expired."""
        old_time = datetime.now(timezone.utc) - timedelta(seconds=60)
        record = _make_record(priority=0, created_at=old_time)
        # default max_queue_wait_seconds for the record is 30.0
        expired, elapsed = enforcer.check_queue_wait(record)
        assert expired is True
        assert elapsed >= 59.0

    def test_boundary_not_expired_at_exact_limit(self, enforcer):
        """Request at exactly max_queue_wait boundary should NOT be expired (needs > not >=)."""
        # Create a record with max_queue_wait_seconds=30 and created_at exactly 30s ago
        exactly_at = datetime.now(timezone.utc) - timedelta(seconds=29.9)
        record = _make_record(priority=0, created_at=exactly_at)
        expired, elapsed = enforcer.check_queue_wait(record)
        assert expired is False


class TestDeadlineEnforcerCheckDeadline:
    """Test check_deadline returns (expired, elapsed) correctly."""

    @pytest.fixture
    def enforcer(self):
        return DeadlineEnforcer(QueueConfig())

    def test_not_expired_when_within_deadline(self, enforcer):
        """Fresh request should not have expired deadline."""
        record = _make_record(priority=0)
        expired, elapsed = enforcer.check_deadline(record)
        assert expired is False
        assert elapsed >= 0.0
        assert elapsed < 1.0

    def test_expired_when_past_deadline(self, enforcer):
        """Request created far in the past should have expired deadline."""
        old_time = datetime.now(timezone.utc) - timedelta(seconds=200)
        record = _make_record(priority=0, created_at=old_time)
        # default deadline_seconds is 120.0
        expired, elapsed = enforcer.check_deadline(record)
        assert expired is True
        assert elapsed >= 199.0

    def test_boundary_not_expired_just_before_deadline(self, enforcer):
        """Request just before deadline should NOT be expired."""
        just_before = datetime.now(timezone.utc) - timedelta(seconds=119.0)
        record = _make_record(priority=0, created_at=just_before)
        expired, elapsed = enforcer.check_deadline(record)
        assert expired is False


class TestDeadlineEnforcerCanFallbackMeetDeadline:
    """Test can_fallback_meet_deadline checks remaining time correctly."""

    @pytest.fixture
    def enforcer(self):
        return DeadlineEnforcer(QueueConfig())

    def test_can_meet_deadline_with_plenty_of_time(self, enforcer):
        """Fresh request should have plenty of remaining time for fallback."""
        record = _make_record(priority=0)
        # deadline_seconds=120, just created, so ~120s remaining
        assert enforcer.can_fallback_meet_deadline(record, 15.0) is True

    def test_cannot_meet_deadline_with_insufficient_time(self, enforcer):
        """Request near deadline cannot meet fallback budget."""
        near_deadline = datetime.now(timezone.utc) - timedelta(seconds=115)
        record = _make_record(priority=0, created_at=near_deadline)
        # deadline_seconds=120, elapsed ~115, remaining ~5s, budget=15s
        assert enforcer.can_fallback_meet_deadline(record, 15.0) is False

    def test_can_meet_deadline_at_exact_boundary(self, enforcer):
        """When remaining time equals fallback budget, should return True (>=)."""
        # deadline=120, need remaining >= 15, so elapsed should be 105
        at_boundary = datetime.now(timezone.utc) - timedelta(seconds=104.5)
        record = _make_record(priority=0, created_at=at_boundary)
        # remaining ~15.5, budget=15
        assert enforcer.can_fallback_meet_deadline(record, 15.0) is True

    def test_cannot_meet_deadline_past_deadline(self, enforcer):
        """Request past deadline has negative remaining time."""
        past_deadline = datetime.now(timezone.utc) - timedelta(seconds=200)
        record = _make_record(priority=0, created_at=past_deadline)
        # remaining is negative
        assert enforcer.can_fallback_meet_deadline(record, 15.0) is False


# ---------------------------------------------------------------------------
# AdmissionController Tests
# ---------------------------------------------------------------------------

from utils.llm_queue.admission import AdmissionController


class TestAdmissionControllerPromptSize:
    """Test prompt size guardrail (Rule 1)."""

    def test_prompt_too_large_rejected(self):
        """Prompt exceeding reject threshold is rejected with prompt_too_large."""
        cfg = QueueConfig(prompt_token_reject_threshold=16000)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        # 16000 tokens = reject threshold (>= threshold means reject)
        record = _make_record(priority=6, request_class="advisory")
        # Override approx_prompt_tokens to exceed threshold
        record = RequestRecord(
            request_id=record.request_id,
            purpose=record.purpose,
            agent=record.agent,
            request_class=record.request_class,
            tier=record.tier,
            provider=record.provider,
            model=record.model,
            json_mode=record.json_mode,
            priority=record.priority,
            deadline_seconds=record.deadline_seconds,
            max_queue_wait_seconds=record.max_queue_wait_seconds,
            fallback_policy=record.fallback_policy,
            stale_after_seconds=record.stale_after_seconds,
            created_at=record.created_at,
            prompt_chars=64000,
            approx_prompt_tokens=16000,
            market_session="closed",
        )
        admitted, reason = ctrl.admit(record)
        assert admitted is False
        assert reason == "prompt_too_large"

    def test_prompt_below_threshold_admitted(self):
        """Prompt below reject threshold is admitted."""
        cfg = QueueConfig(prompt_token_reject_threshold=16000)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        record = _make_record(priority=6, request_class="advisory")
        admitted, reason = ctrl.admit(record)
        assert admitted is True
        assert reason is None

    def test_execution_critical_prompt_too_large_still_rejected(self):
        """Even execution_critical is rejected for prompt_too_large (Rule 1 runs first)."""
        cfg = QueueConfig(prompt_token_reject_threshold=16000)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        record = RequestRecord(
            request_id="test-id",
            purpose="pm_entry",
            agent="pm",
            request_class="execution_critical",
            tier="finance",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=0,
            deadline_seconds=120.0,
            max_queue_wait_seconds=30.0,
            fallback_policy="remote_approved",
            stale_after_seconds=60.0,
            created_at=datetime.now(timezone.utc),
            prompt_chars=80000,
            approx_prompt_tokens=20000,
            market_session="regular",
        )
        admitted, reason = ctrl.admit(record)
        assert admitted is False
        assert reason == "prompt_too_large"


class TestAdmissionControllerQueueFull:
    """Test global queue depth rejection (Rule 2)."""

    def test_queue_full_rejects_low_priority(self):
        """When queue is at max, lower-priority requests are rejected."""
        cfg = QueueConfig(max_queue_size=3)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        # Fill the queue with medium-priority requests
        for _ in range(3):
            q.put(_make_record(priority=3, request_class="repair"))

        # Try to admit a low-priority request
        record = _make_record(priority=6, request_class="advisory")
        admitted, reason = ctrl.admit(record)
        assert admitted is False
        assert reason == "queue_full"

    def test_queue_not_full_admits(self):
        """When queue has room, request is admitted."""
        cfg = QueueConfig(max_queue_size=5)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        q.put(_make_record(priority=3))
        q.put(_make_record(priority=3))

        record = _make_record(priority=6, request_class="advisory")
        admitted, reason = ctrl.admit(record)
        assert admitted is True
        assert reason is None


class TestAdmissionControllerPerClassLimit:
    """Test per-class queue limit rejection (Rule 3)."""

    def test_class_queue_full_rejects(self):
        """Per-class limit exceeded rejects with class_queue_full."""
        cfg = QueueConfig(
            max_queue_size=10,
            max_queue_size_by_class={"advisory": 2},
        )
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        # Fill advisory slots
        q.put(_make_record(priority=6, request_class="advisory"))
        q.put(_make_record(priority=6, request_class="advisory"))

        record = _make_record(priority=6, request_class="advisory")
        admitted, reason = ctrl.admit(record)
        assert admitted is False
        assert reason == "class_queue_full"

    def test_class_limit_not_exceeded_admits(self):
        """Per-class limit not yet reached allows admission."""
        cfg = QueueConfig(
            max_queue_size=10,
            max_queue_size_by_class={"advisory": 3},
        )
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        q.put(_make_record(priority=6, request_class="advisory"))
        q.put(_make_record(priority=6, request_class="advisory"))

        record = _make_record(priority=6, request_class="advisory")
        admitted, reason = ctrl.admit(record)
        assert admitted is True
        assert reason is None

    def test_different_class_not_affected_by_other_class_limit(self):
        """Other class limit doesn't affect unrelated classes."""
        cfg = QueueConfig(
            max_queue_size=10,
            max_queue_size_by_class={"advisory": 1},
        )
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        q.put(_make_record(priority=6, request_class="advisory"))

        # research class has no per-class limit
        record = _make_record(priority=4, request_class="research")
        admitted, reason = ctrl.admit(record)
        assert admitted is True
        assert reason is None


class TestAdmissionControllerMarketHourPressure:
    """Test market-hour pressure policy (Rule 4)."""

    def test_advisory_rejected_during_market_hours_at_50pct(self):
        """Advisory rejected at 50% capacity during market hours."""
        cfg = QueueConfig(max_queue_size=10)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        # Fill to 50% (5 items in a queue of max 10)
        for _ in range(5):
            q.put(_make_record(priority=3, request_class="repair"))

        record = RequestRecord(
            request_id="test-advisory",
            purpose="ceo",
            agent="ceo",
            request_class="advisory",
            tier="low",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=6,
            deadline_seconds=900.0,
            max_queue_wait_seconds=300.0,
            fallback_policy="defer",
            stale_after_seconds=600.0,
            created_at=datetime.now(timezone.utc),
            prompt_chars=1000,
            approx_prompt_tokens=250,
            market_session="regular",
        )
        admitted, reason = ctrl.admit(record)
        assert admitted is False
        assert reason == "market_hour_pressure"

    def test_review_rejected_during_market_hours_at_50pct(self):
        """Review rejected at 50% capacity during market hours."""
        cfg = QueueConfig(max_queue_size=10)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        # Fill to 50%
        for _ in range(5):
            q.put(_make_record(priority=3, request_class="repair"))

        record = RequestRecord(
            request_id="test-review",
            purpose="reviewer_trade",
            agent="reviewer",
            request_class="review",
            tier="low",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=5,
            deadline_seconds=600.0,
            max_queue_wait_seconds=180.0,
            fallback_policy="defer",
            stale_after_seconds=300.0,
            created_at=datetime.now(timezone.utc),
            prompt_chars=1000,
            approx_prompt_tokens=250,
            market_session="regular",
        )
        admitted, reason = ctrl.admit(record)
        assert admitted is False
        assert reason == "market_hour_pressure"

    def test_advisory_admitted_outside_market_hours(self):
        """Advisory admitted when not in regular market hours (no pressure)."""
        cfg = QueueConfig(max_queue_size=10)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        # Fill to 50%
        for _ in range(5):
            q.put(_make_record(priority=3, request_class="repair"))

        record = RequestRecord(
            request_id="test-advisory-closed",
            purpose="ceo",
            agent="ceo",
            request_class="advisory",
            tier="low",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=6,
            deadline_seconds=900.0,
            max_queue_wait_seconds=300.0,
            fallback_policy="defer",
            stale_after_seconds=600.0,
            created_at=datetime.now(timezone.utc),
            prompt_chars=1000,
            approx_prompt_tokens=250,
            market_session="closed",
        )
        admitted, reason = ctrl.admit(record)
        assert admitted is True
        assert reason is None

    def test_advisory_admitted_during_market_hours_below_50pct(self):
        """Advisory admitted during market hours when queue below 50%."""
        cfg = QueueConfig(max_queue_size=10)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        # Fill to 40% (4 items, threshold is 5)
        for _ in range(4):
            q.put(_make_record(priority=3, request_class="repair"))

        record = RequestRecord(
            request_id="test-advisory-low",
            purpose="ceo",
            agent="ceo",
            request_class="advisory",
            tier="low",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=6,
            deadline_seconds=900.0,
            max_queue_wait_seconds=300.0,
            fallback_policy="defer",
            stale_after_seconds=600.0,
            created_at=datetime.now(timezone.utc),
            prompt_chars=1000,
            approx_prompt_tokens=250,
            market_session="regular",
        )
        admitted, reason = ctrl.admit(record)
        assert admitted is True
        assert reason is None

    def test_non_pressure_class_not_affected_during_market_hours(self):
        """market_analysis is NOT affected by market-hour pressure rule."""
        cfg = QueueConfig(max_queue_size=10)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        # Fill to 60%
        for _ in range(6):
            q.put(_make_record(priority=4, request_class="research"))

        record = RequestRecord(
            request_id="test-market-analysis",
            purpose="analyst_signal",
            agent="analyst",
            request_class="market_analysis",
            tier="medium",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=1,
            deadline_seconds=180.0,
            max_queue_wait_seconds=60.0,
            fallback_policy="smaller_local",
            stale_after_seconds=120.0,
            created_at=datetime.now(timezone.utc),
            prompt_chars=1000,
            approx_prompt_tokens=250,
            market_session="regular",
        )
        admitted, reason = ctrl.admit(record)
        assert admitted is True
        assert reason is None


class TestAdmissionControllerExecutionCritical:
    """Test execution_critical reserved admission and eviction (Rule 5)."""

    def test_execution_critical_admitted_when_queue_not_full(self):
        """execution_critical admitted normally when space available."""
        cfg = QueueConfig(max_queue_size=5)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        q.put(_make_record(priority=6, request_class="advisory"))
        q.put(_make_record(priority=6, request_class="advisory"))

        record = RequestRecord(
            request_id="test-exec-crit",
            purpose="pm_entry",
            agent="pm",
            request_class="execution_critical",
            tier="finance",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=0,
            deadline_seconds=120.0,
            max_queue_wait_seconds=30.0,
            fallback_policy="remote_approved",
            stale_after_seconds=60.0,
            created_at=datetime.now(timezone.utc),
            prompt_chars=1000,
            approx_prompt_tokens=250,
            market_session="regular",
        )
        admitted, reason = ctrl.admit(record)
        assert admitted is True
        assert reason is None

    def test_execution_critical_evicts_lowest_when_queue_full(self):
        """execution_critical evicts lowest-priority item when queue is full."""
        cfg = QueueConfig(max_queue_size=3)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        # Fill queue with non-critical items
        q.put(_make_record(priority=4, request_class="research"))
        q.put(_make_record(priority=5, request_class="review"))
        q.put(_make_record(priority=6, request_class="advisory"))
        assert q.depth() == 3

        record = RequestRecord(
            request_id="test-exec-crit-evict",
            purpose="pm_entry",
            agent="pm",
            request_class="execution_critical",
            tier="finance",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=0,
            deadline_seconds=120.0,
            max_queue_wait_seconds=30.0,
            fallback_policy="remote_approved",
            stale_after_seconds=60.0,
            created_at=datetime.now(timezone.utc),
            prompt_chars=1000,
            approx_prompt_tokens=250,
            market_session="regular",
        )
        admitted, reason = ctrl.admit(record)
        assert admitted is True
        assert reason is None
        # One item evicted, depth should still be 2 (evicted 1 from 3)
        assert q.depth() == 2

    def test_execution_critical_no_admission_path_when_all_critical(self):
        """execution_critical fails with no_admission_path when queue full of critical."""
        cfg = QueueConfig(max_queue_size=3)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        # Fill queue with execution_critical items
        q.put(_make_record(priority=0, request_class="execution_critical"))
        q.put(_make_record(priority=0, request_class="execution_critical"))
        q.put(_make_record(priority=0, request_class="execution_critical"))

        record = RequestRecord(
            request_id="test-exec-crit-no-path",
            purpose="pm_entry",
            agent="pm",
            request_class="execution_critical",
            tier="finance",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=0,
            deadline_seconds=120.0,
            max_queue_wait_seconds=30.0,
            fallback_policy="remote_approved",
            stale_after_seconds=60.0,
            created_at=datetime.now(timezone.utc),
            prompt_chars=1000,
            approx_prompt_tokens=250,
            market_session="regular",
        )
        admitted, reason = ctrl.admit(record)
        assert admitted is False
        assert reason == "no_admission_path"

    def test_execution_critical_not_affected_by_market_hour_pressure(self):
        """execution_critical skips market-hour pressure and queue_full rules."""
        cfg = QueueConfig(max_queue_size=10)
        q = PriorityQueue(cfg)
        ctrl = AdmissionController(cfg, q)
        # Fill to 80%
        for _ in range(8):
            q.put(_make_record(priority=6, request_class="advisory"))

        record = RequestRecord(
            request_id="test-exec-crit-no-pressure",
            purpose="pm_entry",
            agent="pm",
            request_class="execution_critical",
            tier="finance",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=0,
            deadline_seconds=120.0,
            max_queue_wait_seconds=30.0,
            fallback_policy="remote_approved",
            stale_after_seconds=60.0,
            created_at=datetime.now(timezone.utc),
            prompt_chars=1000,
            approx_prompt_tokens=250,
            market_session="regular",
        )
        admitted, reason = ctrl.admit(record)
        assert admitted is True
        assert reason is None


# ---------------------------------------------------------------------------
# TelemetryEmitter Tests
# ---------------------------------------------------------------------------

from utils.llm_queue.telemetry import TelemetryEmitter
from utils.llm_queue.models import DispatchResult


def _make_test_record(
    request_class="advisory",
    priority=6,
    purpose="ceo",
    agent="ceo",
    model="llama3",
    market_session="closed",
):
    """Helper to create a RequestRecord for telemetry tests."""
    return RequestRecord(
        request_id=str(uuid.uuid4()),
        purpose=purpose,
        agent=agent,
        request_class=request_class,
        tier="medium",
        provider="ollama",
        model=model,
        json_mode=False,
        priority=priority,
        deadline_seconds=120.0,
        max_queue_wait_seconds=30.0,
        fallback_policy="none",
        stale_after_seconds=60.0,
        created_at=datetime.now(timezone.utc),
        prompt_chars=1000,
        approx_prompt_tokens=250,
        market_session=market_session,
    )


def _make_test_result(
    status="completed",
    queue_wait_seconds=1.0,
    elapsed_seconds=5.0,
    model="llama3",
    fallback_used=False,
    stale=False,
):
    """Helper to create a DispatchResult for telemetry tests."""
    return DispatchResult(
        ok=(status == "completed"),
        text="response text" if status == "completed" else "",
        request_id=str(uuid.uuid4()),
        status=status,
        reason_code=None if status == "completed" else status,
        provider="ollama",
        model=model,
        fallback_used=fallback_used,
        queue_wait_seconds=queue_wait_seconds,
        elapsed_seconds=elapsed_seconds,
        stale=stale,
    )


class TestTelemetryEmitter:
    """Unit tests for TelemetryEmitter fail-open and cycle summary."""

    def test_emit_does_not_raise_on_logger_error(self):
        """Patching the logger to raise does not propagate exceptions from emit()."""
        emitter = TelemetryEmitter()
        record = _make_test_record()
        result = _make_test_result()

        with patch("utils.llm_queue.telemetry.logger") as mock_logger:
            mock_logger.info.side_effect = RuntimeError("logger broken")
            mock_logger.warning.side_effect = RuntimeError("warning also broken")
            # Should NOT raise
            emitter.emit(record, result, queue_depth=5)

    def test_emit_does_not_raise_on_bad_record(self):
        """Passing None or broken objects as record does not propagate exceptions."""
        emitter = TelemetryEmitter()
        result = _make_test_result()

        # None record
        emitter.emit(None, result, queue_depth=0)

        # Broken object with no attributes
        emitter.emit(object(), result, queue_depth=0)

    def test_cycle_summary_counts_correctly(self):
        """Emit several results with different statuses, verify all counts are correct."""
        emitter = TelemetryEmitter()

        # 2 completed
        for _ in range(2):
            emitter.emit(
                _make_test_record(),
                _make_test_result(status="completed"),
                queue_depth=1,
            )
        # 1 timed_out
        emitter.emit(
            _make_test_record(),
            _make_test_result(status="timed_out"),
            queue_depth=2,
        )
        # 1 rejected
        emitter.emit(
            _make_test_record(),
            _make_test_result(status="rejected"),
            queue_depth=3,
        )
        # 1 fallback (completed but fallback_used=True)
        emitter.emit(
            _make_test_record(),
            _make_test_result(status="completed", fallback_used=True),
            queue_depth=2,
        )
        # 1 stale
        emitter.emit(
            _make_test_record(),
            _make_test_result(status="completed", stale=True),
            queue_depth=1,
        )

        summary = emitter.get_cycle_summary()
        assert summary["total_requests"] == 6
        assert summary["completed"] == 4  # 2 + 1 fallback + 1 stale
        assert summary["timed_out"] == 1
        assert summary["rejected"] == 1
        assert summary["fallback_used"] == 1
        assert summary["stale_results"] == 1

    def test_cycle_summary_avg_queue_wait(self):
        """Emit 3 results with queue_wait_seconds of 1.0, 2.0, 3.0. Verify avg = 2.0."""
        emitter = TelemetryEmitter()

        for wait in [1.0, 2.0, 3.0]:
            emitter.emit(
                _make_test_record(),
                _make_test_result(queue_wait_seconds=wait),
                queue_depth=1,
            )

        summary = emitter.get_cycle_summary()
        assert abs(summary["avg_queue_wait_seconds"] - 2.0) < 0.001

    def test_cycle_summary_max_queue_wait(self):
        """Emit results with varying waits. Verify max is correct."""
        emitter = TelemetryEmitter()

        for wait in [1.5, 4.2, 2.8, 0.3]:
            emitter.emit(
                _make_test_record(),
                _make_test_result(queue_wait_seconds=wait),
                queue_depth=1,
            )

        summary = emitter.get_cycle_summary()
        assert abs(summary["max_queue_wait_seconds"] - 4.2) < 0.001

    def test_reset_cycle_clears_counters(self):
        """Emit some results, call reset_cycle(), verify get_cycle_summary() returns zeros."""
        emitter = TelemetryEmitter()

        # Emit several events
        for _ in range(3):
            emitter.emit(
                _make_test_record(),
                _make_test_result(status="completed", queue_wait_seconds=2.0),
                queue_depth=1,
            )

        # Verify non-zero before reset
        summary_before = emitter.get_cycle_summary()
        assert summary_before["total_requests"] == 3

        # Reset
        emitter.reset_cycle()

        # Verify zeros after reset
        summary = emitter.get_cycle_summary()
        assert summary["total_requests"] == 0
        assert summary["completed"] == 0
        assert summary["timed_out"] == 0
        assert summary["rejected"] == 0
        assert summary["deferred"] == 0
        assert summary["fallback_used"] == 0
        assert summary["stale_results"] == 0
        assert summary["avg_queue_wait_seconds"] == 0.0
        assert summary["max_queue_wait_seconds"] == 0.0
        assert summary["by_class"] == {}
        assert summary["by_model"] == {}

    def test_cycle_summary_by_class(self):
        """Emit results with different request_classes, verify by_class breakdown."""
        emitter = TelemetryEmitter()

        # 2 advisory requests
        for _ in range(2):
            emitter.emit(
                _make_test_record(request_class="advisory"),
                _make_test_result(queue_wait_seconds=3.0),
                queue_depth=1,
            )
        # 1 execution_critical request
        emitter.emit(
            _make_test_record(request_class="execution_critical", priority=0),
            _make_test_result(queue_wait_seconds=1.0),
            queue_depth=1,
        )

        summary = emitter.get_cycle_summary()
        by_class = summary["by_class"]

        assert "advisory" in by_class
        assert by_class["advisory"]["count"] == 2
        assert abs(by_class["advisory"]["avg_wait_seconds"] - 3.0) < 0.001

        assert "execution_critical" in by_class
        assert by_class["execution_critical"]["count"] == 1
        assert abs(by_class["execution_critical"]["avg_wait_seconds"] - 1.0) < 0.001

    def test_cycle_summary_by_model(self):
        """Emit results with different models, verify by_model breakdown."""
        emitter = TelemetryEmitter()

        # 2 requests served by llama3
        for _ in range(2):
            emitter.emit(
                _make_test_record(model="llama3"),
                _make_test_result(model="llama3", elapsed_seconds=4.0),
                queue_depth=1,
            )
        # 1 request served by mistral
        emitter.emit(
            _make_test_record(model="mistral"),
            _make_test_result(model="mistral", elapsed_seconds=10.0),
            queue_depth=1,
        )

        summary = emitter.get_cycle_summary()
        by_model = summary["by_model"]

        assert "llama3" in by_model
        assert by_model["llama3"]["count"] == 2
        assert abs(by_model["llama3"]["avg_elapsed_seconds"] - 4.0) < 0.001

        assert "mistral" in by_model
        assert by_model["mistral"]["count"] == 1
        assert abs(by_model["mistral"]["avg_elapsed_seconds"] - 10.0) < 0.001


# ---------------------------------------------------------------------------
# FallbackRouter Tests
# ---------------------------------------------------------------------------

from utils.llm_queue.fallback import FallbackRouter
from utils.llm_queue.models import FallbackDecision


def _make_fallback_record(
    request_class: str = "execution_critical",
    created_at=None,
    deadline_seconds: float = 120.0,
    max_queue_wait_seconds: float = 30.0,
    stale_after_seconds: float = 60.0,
):
    """Helper to build a RequestRecord for fallback tests."""
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    return RequestRecord(
        request_id=str(uuid.uuid4()),
        purpose="test_fallback",
        agent="pm",
        request_class=request_class,
        tier="medium",
        provider="ollama",
        model="llama3",
        json_mode=False,
        priority=0,
        deadline_seconds=deadline_seconds,
        max_queue_wait_seconds=max_queue_wait_seconds,
        fallback_policy="remote_approved",
        stale_after_seconds=stale_after_seconds,
        created_at=created_at,
        prompt_chars=1000,
        approx_prompt_tokens=250,
        market_session="regular",
    )


class TestFallbackRouter:
    """Unit tests for FallbackRouter fallback routing scenarios."""

    def test_execution_critical_with_approved_model_and_time(self):
        """execution_critical with approved model and sufficient time returns FallbackDecision."""
        cfg = QueueConfig(
            approved_fallback_models={"execution_critical": ["claude-3-haiku"]},
            fallback_deadline_buffer_seconds=15.0,
        )
        router = FallbackRouter(cfg)
        # Record created now → remaining ≈ 120s (plenty of time)
        record = _make_fallback_record(
            request_class="execution_critical",
            deadline_seconds=120.0,
        )
        decision = router.resolve_fallback(record, reason="queue_timeout")
        assert decision is not None
        assert isinstance(decision, FallbackDecision)
        assert decision.provider == "anthropic"
        assert decision.model == "claude-3-haiku"

    def test_execution_critical_no_approved_models(self):
        """execution_critical with empty approved list returns None."""
        cfg = QueueConfig(
            approved_fallback_models={"execution_critical": []},
            fallback_deadline_buffer_seconds=15.0,
        )
        router = FallbackRouter(cfg)
        record = _make_fallback_record(request_class="execution_critical")
        decision = router.resolve_fallback(record, reason="queue_timeout")
        assert decision is None

    def test_execution_critical_insufficient_time(self):
        """execution_critical with insufficient remaining time returns None (fail closed)."""
        cfg = QueueConfig(
            approved_fallback_models={"execution_critical": ["claude-3-haiku"]},
            fallback_deadline_buffer_seconds=15.0,
        )
        router = FallbackRouter(cfg)
        # created_at far in the past so remaining < buffer (15s)
        # deadline=120, elapsed≈115, remaining≈5 < buffer=15
        old_time = datetime.now(timezone.utc) - timedelta(seconds=115)
        record = _make_fallback_record(
            request_class="execution_critical",
            deadline_seconds=120.0,
            created_at=old_time,
        )
        decision = router.resolve_fallback(record, reason="queue_timeout")
        assert decision is None

    def test_startup_probe_no_fallback(self):
        """startup_probe always returns None regardless of config."""
        cfg = QueueConfig(
            approved_fallback_models={"startup_probe": ["claude-3-haiku"]},
            fallback_deadline_buffer_seconds=15.0,
        )
        router = FallbackRouter(cfg)
        record = _make_fallback_record(
            request_class="startup_probe",
            deadline_seconds=30.0,
        )
        decision = router.resolve_fallback(record, reason="connectivity_failed")
        assert decision is None

    def test_advisory_defers_when_time_remaining(self):
        """advisory with plenty of time remaining defers (returns None)."""
        cfg = QueueConfig(
            approved_fallback_models={"advisory": ["claude-3-haiku"]},
            fallback_deadline_buffer_seconds=15.0,
        )
        router = FallbackRouter(cfg)
        # Record just created → remaining ≈ 900s, max_queue_wait=300s
        # remaining (900) >= max_queue_wait (300) → defer preferred
        record = _make_fallback_record(
            request_class="advisory",
            deadline_seconds=900.0,
            max_queue_wait_seconds=300.0,
        )
        decision = router.resolve_fallback(record, reason="queue_timeout")
        assert decision is None

    def test_advisory_fallback_when_deadline_imminent(self):
        """advisory with imminent deadline and approved models returns FallbackDecision."""
        cfg = QueueConfig(
            approved_fallback_models={"advisory": ["claude-3-haiku"]},
            fallback_deadline_buffer_seconds=15.0,
        )
        router = FallbackRouter(cfg)
        # deadline=900, max_queue_wait=300
        # Need remaining < max_queue_wait (300) but > buffer (15)
        # elapsed ≈ 700 → remaining ≈ 200, which is < 300 and > 15
        old_time = datetime.now(timezone.utc) - timedelta(seconds=700)
        record = _make_fallback_record(
            request_class="advisory",
            deadline_seconds=900.0,
            max_queue_wait_seconds=300.0,
            created_at=old_time,
        )
        decision = router.resolve_fallback(record, reason="queue_timeout")
        assert decision is not None
        assert isinstance(decision, FallbackDecision)
        assert decision.provider == "anthropic"
        assert decision.model == "claude-3-haiku"

    def test_market_analysis_with_approved_model(self):
        """market_analysis with approved model and sufficient time returns FallbackDecision."""
        cfg = QueueConfig(
            approved_fallback_models={"market_analysis": ["claude-3-haiku"]},
            fallback_deadline_buffer_seconds=15.0,
        )
        router = FallbackRouter(cfg)
        # deadline=180, stale_after=120, just created → plenty of time
        record = _make_fallback_record(
            request_class="market_analysis",
            deadline_seconds=180.0,
            stale_after_seconds=120.0,
        )
        decision = router.resolve_fallback(record, reason="queue_timeout")
        assert decision is not None
        assert isinstance(decision, FallbackDecision)
        assert decision.provider == "anthropic"
        assert decision.model == "claude-3-haiku"

    def test_gpt_model_infers_openai_provider(self):
        """Approved model containing 'gpt' infers provider='openai'."""
        cfg = QueueConfig(
            approved_fallback_models={"execution_critical": ["gpt-4o-mini"]},
            fallback_deadline_buffer_seconds=15.0,
        )
        router = FallbackRouter(cfg)
        record = _make_fallback_record(
            request_class="execution_critical",
            deadline_seconds=120.0,
        )
        decision = router.resolve_fallback(record, reason="queue_timeout")
        assert decision is not None
        assert decision.provider == "openai"
        assert decision.model == "gpt-4o-mini"



# ===========================================================================
# Observe Mode Tests
# ===========================================================================

from utils.llm_queue.dispatcher import OllamaDispatcher


class TestObserveModeDispatch:
    """Verify observe mode: classify, virtual admission, telemetry, no enforcement."""

    def _make_observe_dispatcher(self):
        """Create a dispatcher in observe mode."""
        cfg = QueueConfig(mode="observe")
        return OllamaDispatcher(cfg)

    def test_observe_dispatch_returns_request_record(self):
        """observe_dispatch() returns a valid RequestRecord."""
        dispatcher = self._make_observe_dispatcher()
        record = dispatcher.observe_dispatch(
            system_prompt="You are a PM.",
            user_prompt="Analyze AAPL.",
            model="fin-llama3.1:8b-finance",
            purpose="pm_entry_AAPL",
            tier="finance",
            json_mode=False,
            timeout=120,
            num_ctx=4096,
        )
        assert record is not None
        assert record.purpose == "pm_entry_AAPL"
        assert record.request_class == "execution_critical"
        assert record.priority == 0
        assert record.model == "fin-llama3.1:8b-finance"

    def test_observe_dispatch_classifies_unknown_purpose_as_advisory(self):
        """observe_dispatch() classifies unknown purposes as advisory."""
        dispatcher = self._make_observe_dispatcher()
        record = dispatcher.observe_dispatch(
            system_prompt="Hello",
            user_prompt="World",
            model="llama3",
            purpose="unknown_weird_purpose",
            tier="medium",
            json_mode=False,
            timeout=60,
            num_ctx=2048,
        )
        assert record.request_class == "advisory"
        assert record.priority == 6

    def test_observe_dispatch_does_not_enqueue(self):
        """observe_dispatch() does NOT put anything in the queue."""
        dispatcher = self._make_observe_dispatcher()
        assert dispatcher._queue.depth() == 0

        dispatcher.observe_dispatch(
            system_prompt="You are a PM.",
            user_prompt="Analyze AAPL.",
            model="fin-llama3.1:8b-finance",
            purpose="pm_entry_AAPL",
            tier="finance",
            json_mode=False,
            timeout=120,
            num_ctx=4096,
        )
        # Queue depth should still be zero — observe never enqueues
        assert dispatcher._queue.depth() == 0

    def test_observe_dispatch_emits_telemetry(self):
        """observe_dispatch() emits telemetry for the virtual decision."""
        dispatcher = self._make_observe_dispatcher()

        dispatcher.observe_dispatch(
            system_prompt="You are a PM.",
            user_prompt="Analyze AAPL.",
            model="fin-llama3.1:8b-finance",
            purpose="pm_entry_AAPL",
            tier="finance",
            json_mode=False,
            timeout=120,
            num_ctx=4096,
        )

        summary = dispatcher.get_cycle_summary()
        assert summary["total_requests"] == 1

    def test_observe_dispatch_logs_would_reject_large_prompt(self, caplog):
        """observe_dispatch() logs virtual rejection for oversized prompts."""
        cfg = QueueConfig(mode="observe", prompt_token_reject_threshold=100)
        dispatcher = OllamaDispatcher(cfg)

        # Create prompt exceeding 100 * 4 = 400 chars (threshold in tokens)
        large_prompt = "x" * 500  # 500 chars → ~125 tokens > 100 threshold

        import logging
        with caplog.at_level(logging.INFO):
            record = dispatcher.observe_dispatch(
                system_prompt=large_prompt,
                user_prompt="extra",
                model="llama3",
                purpose="ceo_summary",
                tier="medium",
                json_mode=False,
                timeout=60,
                num_ctx=2048,
            )

        # Should log "would have rejected"
        assert any("would have rejected" in msg for msg in caplog.messages)

    def test_observe_dispatch_logs_would_admit(self, caplog):
        """observe_dispatch() logs virtual admission for normal requests."""
        dispatcher = self._make_observe_dispatcher()

        import logging
        with caplog.at_level(logging.INFO):
            dispatcher.observe_dispatch(
                system_prompt="Hello",
                user_prompt="World",
                model="llama3",
                purpose="analyst_signal_MSFT",
                tier="medium",
                json_mode=False,
                timeout=60,
                num_ctx=2048,
            )

        # Should log "would have admitted"
        assert any("would have admitted" in msg for msg in caplog.messages)

    def test_observe_dispatch_does_not_start_workers(self):
        """observe_dispatch() does NOT start worker threads."""
        dispatcher = self._make_observe_dispatcher()

        dispatcher.observe_dispatch(
            system_prompt="Hello",
            user_prompt="World",
            model="llama3",
            purpose="pm_entry_test",
            tier="finance",
            json_mode=False,
            timeout=60,
            num_ctx=2048,
        )

        assert dispatcher._workers_started is False
        assert len(dispatcher._workers) == 0


# ===========================================================================
# Disabled Mode Passthrough Tests
# ===========================================================================


class TestDisabledModePassthrough:
    """Verify disabled mode: classify at DEBUG level, no queue/admission logic."""

    def _make_disabled_dispatcher(self):
        """Create a dispatcher in disabled mode."""
        cfg = QueueConfig(mode="disabled")
        return OllamaDispatcher(cfg)

    def _make_enforcing_dispatcher(self):
        """Create a dispatcher in enforcing mode."""
        cfg = QueueConfig(mode="enforcing")
        return OllamaDispatcher(cfg)

    def test_is_enabled_false_in_disabled_mode(self):
        """is_enabled() returns False when mode is disabled."""
        dispatcher = self._make_disabled_dispatcher()
        assert dispatcher.is_enabled() is False

    def test_is_enabled_true_in_enforcing_mode(self):
        """is_enabled() returns True when mode is enforcing."""
        dispatcher = self._make_enforcing_dispatcher()
        assert dispatcher.is_enabled() is True

    def test_is_enabled_false_in_observe_mode(self):
        """is_enabled() returns False when mode is observe."""
        cfg = QueueConfig(mode="observe")
        dispatcher = OllamaDispatcher(cfg)
        assert dispatcher.is_enabled() is False

    def test_disabled_classify_returns_request_record(self):
        """disabled_classify() returns a valid RequestRecord."""
        dispatcher = self._make_disabled_dispatcher()
        record = dispatcher.disabled_classify(
            purpose="pm_entry_AAPL",
            tier="finance",
            model="fin-llama3.1:8b-finance",
            json_mode=False,
            prompt_chars=2000,
        )
        assert record is not None
        assert record.purpose == "pm_entry_AAPL"
        assert record.request_class == "execution_critical"
        assert record.priority == 0
        assert record.model == "fin-llama3.1:8b-finance"

    def test_disabled_classify_unknown_purpose(self):
        """disabled_classify() classifies unknown purposes as advisory."""
        dispatcher = self._make_disabled_dispatcher()
        record = dispatcher.disabled_classify(
            purpose="unknown_thing",
            tier="medium",
            model="llama3",
            json_mode=False,
            prompt_chars=500,
        )
        assert record is not None
        assert record.request_class == "advisory"
        assert record.priority == 6

    def test_disabled_classify_logs_at_debug(self, caplog):
        """disabled_classify() logs classification at DEBUG level."""
        import logging

        dispatcher = self._make_disabled_dispatcher()
        with caplog.at_level(logging.DEBUG, logger="utils.llm_queue.dispatcher"):
            record = dispatcher.disabled_classify(
                purpose="analyst_signal_check",
                tier="medium",
                model="llama3",
                json_mode=False,
                prompt_chars=1000,
            )
        assert "Disabled mode classify" in caplog.text
        assert record.request_id in caplog.text

    def test_disabled_classify_does_not_enqueue(self):
        """disabled_classify() does not put anything in the priority queue."""
        cfg = QueueConfig(mode="disabled")
        dispatcher = OllamaDispatcher(cfg)
        dispatcher.disabled_classify(
            purpose="pm_entry_AAPL",
            tier="finance",
            model="fin-llama3.1:8b-finance",
            json_mode=False,
            prompt_chars=2000,
        )
        assert dispatcher._queue.depth() == 0

    def test_disabled_classify_does_not_emit_telemetry(self):
        """disabled_classify() does not emit telemetry events."""
        cfg = QueueConfig(mode="disabled")
        dispatcher = OllamaDispatcher(cfg)
        dispatcher.disabled_classify(
            purpose="pm_entry_AAPL",
            tier="finance",
            model="fin-llama3.1:8b-finance",
            json_mode=False,
            prompt_chars=2000,
        )
        summary = dispatcher.get_cycle_summary()
        assert summary["total_requests"] == 0


# ===========================================================================
# Dispatcher Lifecycle Tests
# ===========================================================================


class TestDispatcherLifecycle:
    """Unit tests for dispatcher lifecycle scenarios: happy path, failure,
    admission rejection, mode switching."""

    def _make_enforcing_config(self, **overrides):
        """Create an enforcing QueueConfig with optional overrides."""
        defaults = dict(
            mode="enforcing",
            global_concurrency=1,
            max_queue_size=10,
        )
        defaults.update(overrides)
        return QueueConfig(**defaults)

    def test_happy_path_complete(self):
        """Full lifecycle: classify → admit → queue → execute → complete.

        Verifies result.ok=True, result.text matches execute_fn output,
        and result.status is 'completed'.
        """
        def execute_fn(system_prompt, user_prompt, model, json_mode, timeout, num_ctx):
            return "model response"

        cfg = self._make_enforcing_config()
        dispatcher = OllamaDispatcher(cfg, execute_fn=execute_fn)

        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value="closed",
        ):
            result = dispatcher.dispatch(
                system_prompt="You are a PM.",
                user_prompt="Analyze AAPL.",
                model="llama3",
                purpose="pm_entry_AAPL",
                tier="finance",
                json_mode=False,
                timeout=120,
                num_ctx=4096,
            )

        assert result.ok is True
        assert result.text == "model response"
        assert result.status == "completed"

    def test_execution_failure_returns_structured_failure(self):
        """execute_fn raises Exception → result.ok=False, reason_code='local_execution_failed'."""
        def execute_fn(system_prompt, user_prompt, model, json_mode, timeout, num_ctx):
            raise RuntimeError("ollama connection refused")

        cfg = self._make_enforcing_config()
        dispatcher = OllamaDispatcher(cfg, execute_fn=execute_fn)

        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value="closed",
        ):
            result = dispatcher.dispatch(
                system_prompt="You are a PM.",
                user_prompt="Analyze AAPL.",
                model="llama3",
                purpose="pm_entry_AAPL",
                tier="finance",
                json_mode=False,
                timeout=120,
                num_ctx=4096,
            )

        assert result.ok is False
        assert result.reason_code == "local_execution_failed"

    def test_admission_rejection_returns_structured_failure(self):
        """Prompt exceeding token reject threshold → result.ok=False, status='prompt_too_large'."""
        def execute_fn(system_prompt, user_prompt, model, json_mode, timeout, num_ctx):
            return "should not reach here"

        cfg = self._make_enforcing_config(prompt_token_reject_threshold=10)
        dispatcher = OllamaDispatcher(cfg, execute_fn=execute_fn)

        # >10 tokens means >40 chars (approx 4 chars per token)
        long_prompt = "x" * 50  # 50 chars → ~12 tokens, exceeds threshold of 10

        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value="closed",
        ):
            result = dispatcher.dispatch(
                system_prompt=long_prompt,
                user_prompt="extra text here",
                model="llama3",
                purpose="pm_entry_AAPL",
                tier="finance",
                json_mode=False,
                timeout=120,
                num_ctx=4096,
            )

        assert result.ok is False
        assert result.status == "prompt_too_large"

    def test_mode_disabled_is_not_enabled(self):
        """QueueConfig(mode='disabled') → dispatcher.is_enabled() == False."""
        cfg = QueueConfig(mode="disabled")
        dispatcher = OllamaDispatcher(cfg)
        assert dispatcher.is_enabled() is False

    def test_mode_enforcing_is_enabled(self):
        """QueueConfig(mode='enforcing') → dispatcher.is_enabled() == True."""
        cfg = QueueConfig(mode="enforcing")
        dispatcher = OllamaDispatcher(cfg)
        assert dispatcher.is_enabled() is True

    def test_observe_mode_does_not_enqueue(self):
        """QueueConfig(mode='observe') → observe_dispatch() keeps queue depth at 0."""
        cfg = QueueConfig(mode="observe")
        dispatcher = OllamaDispatcher(cfg)

        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value="closed",
        ):
            dispatcher.observe_dispatch(
                system_prompt="You are a PM.",
                user_prompt="Analyze AAPL.",
                model="llama3",
                purpose="pm_entry_AAPL",
                tier="finance",
                json_mode=False,
                timeout=120,
                num_ctx=4096,
            )

        assert dispatcher._queue.depth() == 0


# ===========================================================================
# Integration Tests for call_llm() Compatibility (Task 11.6)
# ===========================================================================

from utils.llm import call_llm, parse_json_response


class TestCallLlmCompatibility:
    """Integration tests verifying call_llm() backward compatibility across all queue modes."""

    @patch("utils.llm.requests.post")
    def test_call_llm_returns_string_disabled_mode(self, mock_post, monkeypatch):
        """With LLM_QUEUE_MODE=disabled (default), call_llm(tier='low') returns a string."""
        monkeypatch.delenv("LLM_QUEUE_MODE", raising=False)
        monkeypatch.setenv("LLM_LOW_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_LOW_MODEL", "llama3")
        monkeypatch.setenv("OLLAMA_SERIALIZE_REQUESTS", "false")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": '{"decisions": []}'}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = call_llm(
            "You are a test assistant.",
            "Return empty decisions.",
            json_mode=True,
            tier="low",
            purpose="test_disabled_mode",
        )

        assert isinstance(result, str)
        assert result == '{"decisions": []}'

    @patch("utils.llm.requests.post")
    def test_call_llm_returns_string_observe_mode(self, mock_post, monkeypatch):
        """With LLM_QUEUE_MODE=observe, call_llm() returns string unchanged."""
        monkeypatch.setenv("LLM_QUEUE_MODE", "observe")
        monkeypatch.setenv("LLM_LOW_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_LOW_MODEL", "llama3")
        monkeypatch.setenv("OLLAMA_SERIALIZE_REQUESTS", "false")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": '{"decisions": [{"symbol": "AAPL"}]}'}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = call_llm(
            "You are a test assistant.",
            "Return a decision.",
            json_mode=True,
            tier="low",
            purpose="test_observe_mode",
        )

        assert isinstance(result, str)
        assert result == '{"decisions": [{"symbol": "AAPL"}]}'

    @patch("utils.llm.requests.post")
    def test_empty_response_handling(self, mock_post, monkeypatch):
        """Mock ollama returning empty string. Verify call_llm retries once and returns empty string."""
        monkeypatch.delenv("LLM_QUEUE_MODE", raising=False)
        monkeypatch.setenv("LLM_LOW_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_LOW_MODEL", "llama3")
        monkeypatch.setenv("OLLAMA_SERIALIZE_REQUESTS", "false")

        # Both attempts return empty content
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": ""}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = call_llm(
            "You are a test assistant.",
            "Return something.",
            json_mode=True,
            tier="low",
            purpose="test_empty_response",
        )

        # call_llm retries once on empty, then returns empty string
        assert isinstance(result, str)
        assert result == ""
        # Should have been called twice (original + 1 retry)
        assert mock_post.call_count == 2

    @patch("utils.llm._call_ollama")
    def test_finance_tier_fallback(self, mock_ollama, monkeypatch):
        """Mock finance ollama call raising, verify fallback to medium tier works."""
        monkeypatch.delenv("LLM_QUEUE_MODE", raising=False)
        monkeypatch.setenv("LLM_FINANCE_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_FINANCE_MODEL", "fin-llama3.1:8b-finance")
        monkeypatch.setenv("LLM_MED_MODEL", "llama3.1:8b")

        # First call (finance model) raises, second call (medium model) succeeds
        mock_ollama.side_effect = [
            ConnectionError("Connection refused"),
            '{"decisions": [], "portfolio_notes": "fallback worked"}',
        ]

        result = call_llm(
            "You are a PM.",
            "Analyze portfolio.",
            json_mode=True,
            tier="finance",
            purpose="pm_entry_test",
        )

        assert isinstance(result, str)
        assert "fallback worked" in result
        # Verify two calls were made (first fails, second is fallback)
        assert mock_ollama.call_count == 2

    def test_json_repair_recursion_guard(self):
        """Call parse_json_response with non-JSON text where repair also returns non-JSON.

        Verify it doesn't infinitely recurse (max 2 attempts).
        """
        non_json_text = "This is plain prose with no JSON structure at all. No trades recommended."

        # Mock call_llm to return non-JSON for the repair attempt too
        with patch("utils.llm.call_llm") as mock_repair_llm:
            # The repair call also returns prose (non-JSON), triggering recursion guard
            mock_repair_llm.return_value = "Still not JSON, cannot extract anything meaningful."

            # parse_json_response should NOT infinitely recurse.
            # It should detect "no action" intent from the original text
            # since it contains "No trades recommended"
            result = parse_json_response(non_json_text)

            # Should return the no-action fallback (detected from prose)
            assert isinstance(result, dict)
            assert result["decisions"] == []
            # Repair was attempted at most once (recursion guard caps at 2)
            assert mock_repair_llm.call_count <= 1

    def test_json_repair_recursion_guard_no_action_phrases(self):
        """Verify recursion guard works when text has no no-action phrases — raises ValueError."""
        # Text that has no JSON and no no-action phrases
        non_json_text = "The market is very interesting today with lots of activity happening."

        with patch("utils.llm.call_llm") as mock_repair_llm:
            # Repair also returns non-JSON
            mock_repair_llm.return_value = "Cannot help with that request."

            # Without no-action phrases, should raise ValueError after failing repair
            with pytest.raises(ValueError, match="Failed to parse LLM JSON response"):
                parse_json_response(non_json_text)

            # Repair was called but recursion was limited
            assert mock_repair_llm.call_count <= 1
