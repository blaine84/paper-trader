"""Unit tests for funnel opening confirmation evaluator (Task 7.1).

Tests:
- run_opening_confirmation() — budget enforcement, deterministic ordering,
  data unavailability handling, promote/reject/needs_confirmation decisions
- _check_volume() — volume confirmation logic
- _check_price_position() — price vs key levels
- _check_price_behavior() — directional price behavior
- _check_catalyst_freshness() — 24h freshness window
- Budget exhaustion: not_evaluated for remaining, stage_status unchanged
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine

from db.schema import Base, FunnelCandidate, get_session
from utils.funnel_confirmation import (
    ConfirmationDecision,
    run_opening_confirmation,
    _check_volume,
    _check_price_position,
    _check_price_behavior,
    _check_catalyst_freshness,
    _get_analyst_plan,
    _make_confirmation_decision,
    _to_float,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _make_analyst_evidence() -> dict:
    """Build analyst evidence payload with key levels and volume requirements."""
    return {
        "authoritative_setup_type": "gap_and_go",
        "signal_direction": "LONG",
        "signal_strength": "strong",
        "confidence": "high",
        "key_levels": {
            "support": 145.0,
            "resistance": 160.0,
            "vwap": 152.0,
            "entry_zone": 150.0,
            "stop_level": 144.0,
            "target_1": 158.0,
            "target_2": 165.0,
        },
        "invalidation": "Close below 144.0",
        "volume_requirements": "Volume above 1M in first 15 minutes",
        "catalyst_dependence": "medium",
        "catalyst_freshness": "fresh",
    }


def _make_funnel_candidate(
    engine,
    symbol: str = "AAPL",
    scout_rank: int = 1,
    scout_score: float = 85.0,
    direction_bias: str = "bullish",
    stage_status: str = "awaiting_confirmation",
    discovered_at: datetime | None = None,
    analyst_evidence: dict | None = None,
) -> FunnelCandidate:
    """Create and persist a FunnelCandidate for testing confirmation."""
    if discovered_at is None:
        discovered_at = datetime.now(timezone.utc) - timedelta(hours=3)

    if analyst_evidence is None:
        analyst_evidence = _make_analyst_evidence()

    ny_tz = ZoneInfo("America/New_York")
    today_ny = datetime.now(timezone.utc).astimezone(ny_tz).date()

    scout_decision = {
        "agent": "scout",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": "promoted",
        "reasoning": "Strong mover with catalyst",
        "evidence": {"scout_score": scout_score},
        "next_stage": "awaiting_research",
    }
    analyst_decision = {
        "agent": "analyst",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": "promoted",
        "reasoning": "Clear gap_and_go setup",
        "evidence": analyst_evidence,
        "next_stage": "awaiting_confirmation",
    }

    candidate = FunnelCandidate(
        candidate_id=str(uuid.uuid4()),
        date=today_ny,
        symbol=symbol,
        discovered_at=discovered_at,
        source_run="premarket",
        selection_mode="deterministic_fallback",
        scout_rank=scout_rank,
        scout_score=scout_score,
        direction_bias=direction_bias,
        catalyst_evidence=json.dumps({
            "news_headlines": [{"headline": f"{symbol} beats expectations"}],
            "timestamp": discovered_at.isoformat(),
        }),
        selection_reason="Strong catalyst",
        primary_risk="Valuation",
        sector_context=json.dumps({"sector": "tech"}),
        preliminary_setup_type="gap_and_go",
        authoritative_setup_type="gap_and_go",
        stage_status=stage_status,
        stage_decisions=json.dumps([scout_decision, analyst_decision]),
        expired=False,
    )

    session = get_session(engine)
    session.add(candidate)
    session.commit()
    session.refresh(candidate)
    session.expunge(candidate)
    session.close()
    return candidate


# ---------------------------------------------------------------------------
# _check_volume tests
# ---------------------------------------------------------------------------


class TestCheckVolume:
    """Tests for volume confirmation check."""

    def test_positive_volume_confirmed(self):
        assert _check_volume({"volume": 1_000_000}, {}) is True

    def test_zero_volume_not_confirmed(self):
        assert _check_volume({"volume": 0}, {}) is False

    def test_none_volume_not_confirmed(self):
        assert _check_volume({"volume": None}, {}) is False

    def test_missing_volume_not_confirmed(self):
        assert _check_volume({}, {}) is False

    def test_negative_volume_not_confirmed(self):
        assert _check_volume({"volume": -1}, {}) is False


# ---------------------------------------------------------------------------
# _check_price_position tests
# ---------------------------------------------------------------------------


class TestCheckPricePosition:
    """Tests for price position vs key levels."""

    def test_long_above_support_confirmed(self):
        live = {"price": 150.0}
        plan = {
            "signal_direction": "LONG",
            "key_levels": {"support": 145.0, "stop_level": 144.0},
        }
        assert _check_price_position(live, plan) is True

    def test_long_below_stop_not_confirmed(self):
        live = {"price": 143.0}
        plan = {
            "signal_direction": "LONG",
            "key_levels": {"support": 145.0, "stop_level": 144.0},
        }
        assert _check_price_position(live, plan) is False

    def test_short_below_resistance_confirmed(self):
        live = {"price": 155.0}
        plan = {
            "signal_direction": "SHORT",
            "key_levels": {"resistance": 160.0, "stop_level": 162.0},
        }
        assert _check_price_position(live, plan) is True

    def test_short_above_stop_not_confirmed(self):
        live = {"price": 163.0}
        plan = {
            "signal_direction": "SHORT",
            "key_levels": {"resistance": 160.0, "stop_level": 162.0},
        }
        assert _check_price_position(live, plan) is False

    def test_no_key_levels_returns_none(self):
        assert _check_price_position({"price": 150.0}, {}) is None
        assert _check_price_position({"price": 150.0}, {"key_levels": None}) is None

    def test_no_price_returns_none(self):
        plan = {"signal_direction": "LONG", "key_levels": {"support": 145.0}}
        assert _check_price_position({"price": None}, plan) is None
        assert _check_price_position({}, plan) is None


# ---------------------------------------------------------------------------
# _check_price_behavior tests
# ---------------------------------------------------------------------------


class TestCheckPriceBehavior:
    """Tests for price behavior direction check."""

    def test_long_price_up_from_open_ok(self):
        """LONG signal + price above open = OK."""
        live = {"price": 152.0, "open": 150.0, "prev_close": 148.0}
        candidate = MagicMock(direction_bias="bullish")
        plan = {"signal_direction": "LONG"}
        assert _check_price_behavior(live, candidate, plan) is True

    def test_long_price_down_2pct_or_less_ok(self):
        """LONG signal + price down less than 2% from open = still OK."""
        live = {"price": 149.0, "open": 150.0, "prev_close": 148.0}
        candidate = MagicMock(direction_bias="bullish")
        plan = {"signal_direction": "LONG"}
        # -0.67% is within -2% tolerance
        assert _check_price_behavior(live, candidate, plan) is True

    def test_long_price_down_more_than_2pct_not_ok(self):
        """LONG signal + price down more than 2% from open = NOT OK."""
        live = {"price": 145.0, "open": 150.0, "prev_close": 148.0}
        candidate = MagicMock(direction_bias="bullish")
        plan = {"signal_direction": "LONG"}
        # -3.33% exceeds tolerance
        assert _check_price_behavior(live, candidate, plan) is False

    def test_short_price_down_from_open_ok(self):
        """SHORT signal + price below open = OK."""
        live = {"price": 148.0, "open": 150.0, "prev_close": 152.0}
        candidate = MagicMock(direction_bias="bearish")
        plan = {"signal_direction": "SHORT"}
        assert _check_price_behavior(live, candidate, plan) is True

    def test_short_price_up_more_than_2pct_not_ok(self):
        """SHORT signal + price up more than 2% from open = NOT OK."""
        live = {"price": 154.0, "open": 150.0, "prev_close": 152.0}
        candidate = MagicMock(direction_bias="bearish")
        plan = {"signal_direction": "SHORT"}
        # +2.67% exceeds tolerance
        assert _check_price_behavior(live, candidate, plan) is False

    def test_no_direction_is_permissive(self):
        """No direction signal → permissive (any movement OK)."""
        live = {"price": 155.0, "open": 150.0, "prev_close": 148.0}
        candidate = MagicMock(direction_bias="neutral")
        plan = {"signal_direction": ""}
        assert _check_price_behavior(live, candidate, plan) is True

    def test_falls_back_to_direction_bias(self):
        """Uses candidate.direction_bias when analyst plan has no direction."""
        live = {"price": 152.0, "open": 150.0, "prev_close": 148.0}
        candidate = MagicMock(direction_bias="bullish")
        plan = {}
        assert _check_price_behavior(live, candidate, plan) is True

    def test_no_price_returns_false(self):
        """Missing price returns False."""
        live = {"price": None, "open": 150.0}
        candidate = MagicMock(direction_bias="bullish")
        plan = {"signal_direction": "LONG"}
        assert _check_price_behavior(live, candidate, plan) is False


# ---------------------------------------------------------------------------
# _check_catalyst_freshness tests
# ---------------------------------------------------------------------------


class TestCheckCatalystFreshness:
    """Tests for catalyst freshness (≤24h)."""

    def test_discovered_within_24h_fresh(self):
        """Candidate discovered 3 hours ago is fresh."""
        candidate = MagicMock()
        candidate.discovered_at = datetime.now(timezone.utc) - timedelta(hours=3)
        candidate.catalyst_evidence = json.dumps({})
        assert _check_catalyst_freshness(candidate) is True

    def test_discovered_over_24h_stale(self):
        """Candidate discovered 25 hours ago is stale."""
        candidate = MagicMock()
        candidate.discovered_at = datetime.now(timezone.utc) - timedelta(hours=25)
        candidate.catalyst_evidence = json.dumps({})
        assert _check_catalyst_freshness(candidate) is False

    def test_discovered_just_under_24h_fresh(self):
        """Candidate discovered just under 24h ago is still fresh."""
        candidate = MagicMock()
        candidate.discovered_at = datetime.now(timezone.utc) - timedelta(hours=23, minutes=59)
        candidate.catalyst_evidence = json.dumps({})
        assert _check_catalyst_freshness(candidate) is True

    def test_catalyst_evidence_timestamp_stale(self):
        """Catalyst evidence with old timestamp makes it stale."""
        candidate = MagicMock()
        candidate.discovered_at = datetime.now(timezone.utc) - timedelta(hours=2)
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        candidate.catalyst_evidence = json.dumps({"timestamp": old_ts})
        assert _check_catalyst_freshness(candidate) is False

    def test_no_discovered_at_is_fresh(self):
        """If discovered_at is None, consider fresh (no info to reject)."""
        candidate = MagicMock()
        candidate.discovered_at = None
        candidate.catalyst_evidence = json.dumps({})
        assert _check_catalyst_freshness(candidate) is True


# ---------------------------------------------------------------------------
# _get_analyst_plan tests
# ---------------------------------------------------------------------------


class TestGetAnalystPlan:
    """Tests for analyst plan extraction from stage decisions."""

    def test_extracts_analyst_promoted_plan(self, in_memory_engine):
        """Extracts plan from promoted analyst decision."""
        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        plan = _get_analyst_plan(candidate)

        assert plan["signal_direction"] == "LONG"
        assert plan["signal_strength"] == "strong"
        assert plan["key_levels"]["support"] == 145.0
        assert plan["volume_requirements"] == "Volume above 1M in first 15 minutes"

    def test_empty_stage_decisions_returns_empty(self, in_memory_engine):
        """Empty stage_decisions returns empty dict."""
        candidate = MagicMock()
        candidate.stage_decisions = "[]"
        assert _get_analyst_plan(candidate) == {}

    def test_no_analyst_decision_returns_empty(self, in_memory_engine):
        """No analyst decision in history returns empty dict."""
        candidate = MagicMock()
        candidate.stage_decisions = json.dumps([{
            "agent": "scout",
            "decision": "promoted",
            "evidence": {},
        }])
        assert _get_analyst_plan(candidate) == {}


# ---------------------------------------------------------------------------
# _make_confirmation_decision tests
# ---------------------------------------------------------------------------


class TestMakeConfirmationDecision:
    """Tests for confirmation decision logic."""

    def test_all_checks_pass_promotes(self):
        """Volume + price + catalyst fresh = promoted."""
        candidate = MagicMock(symbol="AAPL")
        decision, reasoning, next_stage = _make_confirmation_decision(
            volume_ok=True, vwap_ok=True, price_ok=True, catalyst_fresh=True,
            candidate=candidate, live_data={}, analyst_plan={},
        )
        assert decision == "promoted"
        assert next_stage == "pm_eligible"

    def test_stale_catalyst_rejects(self):
        """Stale catalyst always leads to rejection."""
        candidate = MagicMock(symbol="AAPL")
        decision, reasoning, next_stage = _make_confirmation_decision(
            volume_ok=True, vwap_ok=True, price_ok=True, catalyst_fresh=False,
            candidate=candidate, live_data={}, analyst_plan={},
        )
        assert decision == "rejected"
        assert next_stage == "rejected_confirmation"
        assert "fresh" in reasoning.lower() or "24h" in reasoning

    def test_volume_missing_needs_confirmation(self):
        """Volume not confirmed + price OK = needs_confirmation."""
        candidate = MagicMock(symbol="AAPL")
        decision, reasoning, next_stage = _make_confirmation_decision(
            volume_ok=False, vwap_ok=True, price_ok=True, catalyst_fresh=True,
            candidate=candidate, live_data={}, analyst_plan={},
        )
        assert decision == "needs_confirmation"
        assert next_stage == "awaiting_confirmation"

    def test_price_behavior_adverse_rejects(self):
        """Adverse price behavior leads to rejection."""
        candidate = MagicMock(symbol="AAPL")
        decision, reasoning, next_stage = _make_confirmation_decision(
            volume_ok=True, vwap_ok=True, price_ok=False, catalyst_fresh=True,
            candidate=candidate, live_data={}, analyst_plan={},
        )
        assert decision == "rejected"
        assert next_stage == "rejected_confirmation"

    def test_vwap_false_rejects_despite_volume_price(self):
        """Price below invalidation level rejects even with volume/price OK."""
        candidate = MagicMock(symbol="AAPL")
        decision, reasoning, next_stage = _make_confirmation_decision(
            volume_ok=True, vwap_ok=False, price_ok=True, catalyst_fresh=True,
            candidate=candidate, live_data={}, analyst_plan={},
        )
        assert decision == "rejected"
        assert next_stage == "rejected_confirmation"

    def test_vwap_none_does_not_block_promotion(self):
        """vwap_ok=None (unavailable) does not prevent promotion."""
        candidate = MagicMock(symbol="AAPL")
        decision, reasoning, next_stage = _make_confirmation_decision(
            volume_ok=True, vwap_ok=None, price_ok=True, catalyst_fresh=True,
            candidate=candidate, live_data={}, analyst_plan={},
        )
        assert decision == "promoted"
        assert next_stage == "pm_eligible"


# ---------------------------------------------------------------------------
# run_opening_confirmation integration tests
# ---------------------------------------------------------------------------


class TestRunOpeningConfirmation:
    """Integration tests for the full confirmation pipeline."""

    @patch("utils.funnel_confirmation._get_live_data")
    def test_promotes_candidate_all_checks_pass(self, mock_live, in_memory_engine):
        """Candidate with all confirmation checks passing is promoted."""
        mock_live.return_value = {
            "price": 152.0, "open": 150.0, "prev_close": 148.0,
            "volume": 2_000_000, "high": 153.0, "low": 149.5,
        }

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        decisions = run_opening_confirmation(in_memory_engine, [candidate])

        assert len(decisions) == 1
        assert decisions[0].decision == "promoted"
        assert decisions[0].volume_confirmed is True
        assert decisions[0].price_behavior_ok is True
        assert decisions[0].catalyst_still_fresh is True

        # Verify DB state
        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "pm_eligible"
        stage_decisions = json.loads(row.stage_decisions)
        # scout + analyst + confirmation
        assert len(stage_decisions) == 3
        assert stage_decisions[-1]["agent"] == "confirmation"
        assert stage_decisions[-1]["decision"] == "promoted"
        session.close()

    @patch("utils.funnel_confirmation._get_live_data")
    def test_rejects_stale_catalyst(self, mock_live, in_memory_engine):
        """Candidate with stale catalyst (>24h) is rejected."""
        mock_live.return_value = {
            "price": 152.0, "open": 150.0, "prev_close": 148.0,
            "volume": 2_000_000,
        }

        # Discovered 25 hours ago
        old_discovered = datetime.now(timezone.utc) - timedelta(hours=25)
        candidate = _make_funnel_candidate(
            in_memory_engine, "AAPL", discovered_at=old_discovered
        )
        decisions = run_opening_confirmation(in_memory_engine, [candidate])

        assert decisions[0].decision == "rejected"
        assert decisions[0].catalyst_still_fresh is False

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "rejected_confirmation"
        session.close()

    @patch("utils.funnel_confirmation._get_live_data")
    def test_data_unavailability_needs_confirmation(self, mock_live, in_memory_engine):
        """Data unavailability records needs_confirmation (Req 6.8)."""
        mock_live.return_value = None

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        decisions = run_opening_confirmation(in_memory_engine, [candidate])

        assert decisions[0].decision == "needs_confirmation"
        assert "unavailable" in decisions[0].reasoning.lower()

        # stage_status stays awaiting_confirmation
        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "awaiting_confirmation"
        session.close()


    @patch("utils.funnel_confirmation._get_live_data")
    def test_budget_exhaustion_not_evaluated(self, mock_live, in_memory_engine):
        """Budget exhaustion appends not_evaluated, leaves stage_status unchanged (Req 6.6)."""
        # Simulate slow data fetch that exhausts budget
        def slow_fetch(symbol):
            time.sleep(0.1)
            return {
                "price": 152.0, "open": 150.0, "prev_close": 148.0,
                "volume": 2_000_000,
            }
        mock_live.side_effect = slow_fetch

        # Create 3 candidates but with a tiny budget (0.05s)
        candidates = [
            _make_funnel_candidate(in_memory_engine, "AAPL", scout_rank=1),
            _make_funnel_candidate(in_memory_engine, "MSFT", scout_rank=2),
            _make_funnel_candidate(in_memory_engine, "GOOGL", scout_rank=3),
        ]

        decisions = run_opening_confirmation(
            in_memory_engine, candidates, budget_seconds=0.05
        )

        # At least some should be not_evaluated due to budget
        not_evaluated = [d for d in decisions if d.decision == "not_evaluated"]
        assert len(not_evaluated) >= 1

        # not_evaluated candidates stay in awaiting_confirmation
        session = get_session(in_memory_engine)
        for d in not_evaluated:
            row = (
                session.query(FunnelCandidate)
                .filter(FunnelCandidate.candidate_id == d.candidate_id)
                .first()
            )
            assert row.stage_status == "awaiting_confirmation"
            # Stage decision appended
            stage_decs = json.loads(row.stage_decisions)
            last_dec = stage_decs[-1]
            assert last_dec["agent"] == "confirmation"
            assert last_dec["decision"] == "not_evaluated"
            assert "budget" in last_dec["reasoning"].lower()
        session.close()

    @patch("utils.funnel_confirmation._get_live_data")
    def test_deterministic_order(self, mock_live, in_memory_engine):
        """Candidates processed in scout_rank ASC, scout_score DESC order (Req 6.7)."""
        mock_live.return_value = {
            "price": 152.0, "open": 150.0, "prev_close": 148.0,
            "volume": 2_000_000,
        }

        # Create candidates out of order
        c1 = _make_funnel_candidate(
            in_memory_engine, "AAPL", scout_rank=2, scout_score=90.0
        )
        c2 = _make_funnel_candidate(
            in_memory_engine, "MSFT", scout_rank=1, scout_score=85.0
        )
        c3 = _make_funnel_candidate(
            in_memory_engine, "GOOGL", scout_rank=2, scout_score=95.0
        )

        decisions = run_opening_confirmation(
            in_memory_engine, [c1, c2, c3], budget_seconds=45
        )

        # Expected order: MSFT (rank 1), GOOGL (rank 2, score 95), AAPL (rank 2, score 90)
        assert len(decisions) == 3
        # Find the candidates by id - they should be in deterministic order
        session = get_session(in_memory_engine)
        msft = session.query(FunnelCandidate).filter_by(symbol="MSFT").first()
        googl = session.query(FunnelCandidate).filter_by(symbol="GOOGL").first()
        aapl = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        session.close()

        assert decisions[0].candidate_id == msft.candidate_id
        assert decisions[1].candidate_id == googl.candidate_id
        assert decisions[2].candidate_id == aapl.candidate_id


    @patch("utils.funnel_confirmation._get_live_data")
    def test_only_awaiting_confirmation_processed(self, mock_live, in_memory_engine):
        """Only candidates with stage_status=awaiting_confirmation are evaluated (Req 6.1)."""
        mock_live.return_value = {
            "price": 152.0, "open": 150.0, "prev_close": 148.0,
            "volume": 2_000_000,
        }

        # Create candidates with different stage_status
        c1 = _make_funnel_candidate(in_memory_engine, "AAPL", stage_status="awaiting_confirmation")
        c2 = _make_funnel_candidate(in_memory_engine, "MSFT", stage_status="awaiting_analysis")
        c3 = _make_funnel_candidate(in_memory_engine, "GOOGL", stage_status="pm_eligible")

        decisions = run_opening_confirmation(
            in_memory_engine, [c1, c2, c3], budget_seconds=45
        )

        # Only AAPL should be evaluated
        assert len(decisions) == 1
        assert decisions[0].candidate_id == c1.candidate_id

    @patch("utils.funnel_confirmation._get_live_data")
    def test_exception_handling_needs_confirmation(self, mock_live, in_memory_engine):
        """Exception during evaluation records needs_confirmation (Req 6.8)."""
        mock_live.side_effect = RuntimeError("API connection failed")

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        decisions = run_opening_confirmation(in_memory_engine, [candidate])

        assert len(decisions) == 1
        assert decisions[0].decision == "needs_confirmation"
        assert "error" in decisions[0].reasoning.lower()

        # Stage_status stays awaiting_confirmation
        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "awaiting_confirmation"
        session.close()

    @patch("utils.funnel_confirmation._get_live_data")
    def test_empty_candidate_list(self, mock_live, in_memory_engine):
        """Empty candidate list returns empty decisions."""
        decisions = run_opening_confirmation(in_memory_engine, [], budget_seconds=45)
        assert decisions == []
        mock_live.assert_not_called()

    @patch("utils.funnel_confirmation._get_live_data")
    def test_rejects_adverse_price_behavior(self, mock_live, in_memory_engine):
        """Candidate with adverse price behavior is rejected."""
        # LONG signal but price dropped >2% from open
        mock_live.return_value = {
            "price": 140.0, "open": 150.0, "prev_close": 148.0,
            "volume": 2_000_000,
        }

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        decisions = run_opening_confirmation(in_memory_engine, [candidate])

        assert decisions[0].decision == "rejected"
        assert decisions[0].price_behavior_ok is False

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "rejected_confirmation"
        session.close()

    @patch("utils.funnel_confirmation._get_live_data")
    def test_stage_decisions_preserve_prior(self, mock_live, in_memory_engine):
        """Confirmation decision does not overwrite prior stage decisions."""
        mock_live.return_value = {
            "price": 152.0, "open": 150.0, "prev_close": 148.0,
            "volume": 2_000_000,
        }

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")

        # Record original decisions count
        original_decisions = json.loads(candidate.stage_decisions)
        original_count = len(original_decisions)

        run_opening_confirmation(in_memory_engine, [candidate])

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        final_decisions = json.loads(row.stage_decisions)
        # Should have original + 1 new confirmation decision
        assert len(final_decisions) == original_count + 1
        # Prior decisions unchanged
        for i in range(original_count):
            assert final_decisions[i] == original_decisions[i]
        session.close()


# ---------------------------------------------------------------------------
# _to_float helper tests
# ---------------------------------------------------------------------------


class TestToFloat:
    """Tests for the float conversion helper."""

    def test_converts_int(self):
        assert _to_float(145) == 145.0

    def test_converts_float(self):
        assert _to_float(145.5) == 145.5

    def test_converts_string(self):
        assert _to_float("145.5") == 145.5

    def test_none_returns_none(self):
        assert _to_float(None) is None

    def test_invalid_string_returns_none(self):
        assert _to_float("not_a_number") is None

    def test_empty_string_returns_none(self):
        assert _to_float("") is None


# ---------------------------------------------------------------------------
# run_confirmation_retry tests (Task 7.2)
# ---------------------------------------------------------------------------


class TestRunConfirmationRetry:
    """Tests for the 10:00 ET bounded shortlist confirmation retry pass."""

    @patch("utils.funnel_confirmation._get_live_data")
    def test_retry_evaluates_awaiting_confirmation_candidates(
        self, mock_live, in_memory_engine
    ):
        """Retry queries and evaluates only today's awaiting_confirmation candidates."""
        from utils.funnel_confirmation import run_confirmation_retry

        mock_live.return_value = {
            "price": 152.0, "open": 150.0, "prev_close": 148.0,
            "volume": 2_000_000, "high": 153.0, "low": 149.5,
        }

        # Create candidates: one awaiting_confirmation, one already promoted
        _make_funnel_candidate(in_memory_engine, "AAPL", stage_status="awaiting_confirmation")
        _make_funnel_candidate(in_memory_engine, "MSFT", stage_status="pm_eligible")

        funnel_config = {
            "funnel": {
                "budgets": {"market_hours_confirmation_budget_seconds": 60},
            }
        }

        decisions = run_confirmation_retry(in_memory_engine, funnel_config=funnel_config)

        # Only AAPL should be evaluated (awaiting_confirmation)
        assert len(decisions) == 1
        assert decisions[0].decision == "promoted"

        # Verify AAPL is now pm_eligible
        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "pm_eligible"
        session.close()

    @patch("utils.funnel_confirmation._get_live_data")
    def test_retry_uses_market_hours_budget(self, mock_live, in_memory_engine):
        """Retry uses market_hours_confirmation_budget_seconds (60s) not primary (45s)."""
        from utils.funnel_confirmation import run_confirmation_retry, FunnelRunLog

        mock_live.return_value = {
            "price": 152.0, "open": 150.0, "prev_close": 148.0,
            "volume": 2_000_000,
        }

        _make_funnel_candidate(in_memory_engine, "AAPL", stage_status="awaiting_confirmation")

        funnel_config = {
            "funnel": {
                "budgets": {"market_hours_confirmation_budget_seconds": 60},
            }
        }

        run_confirmation_retry(in_memory_engine, funnel_config=funnel_config)

        # Verify FunnelRunLog records the 60s budget
        from db.schema import FunnelRunLog as FRL
        session = get_session(in_memory_engine)
        log = session.query(FRL).filter_by(stage="confirmation_retry").first()
        assert log is not None
        assert log.budget_seconds == 60
        session.close()

    @patch("utils.funnel_confirmation._get_live_data")
    def test_retry_empty_candidates_records_completed(self, mock_live, in_memory_engine):
        """Retry with no awaiting_confirmation candidates records completed empty run."""
        from utils.funnel_confirmation import run_confirmation_retry
        from db.schema import FunnelRunLog as FRL

        funnel_config = {
            "funnel": {
                "budgets": {"market_hours_confirmation_budget_seconds": 60},
            }
        }

        decisions = run_confirmation_retry(in_memory_engine, funnel_config=funnel_config)

        assert decisions == []
        mock_live.assert_not_called()

        # FunnelRunLog should exist with result_status=completed, candidates_input=0
        session = get_session(in_memory_engine)
        log = session.query(FRL).filter_by(stage="confirmation_retry").first()
        assert log is not None
        assert log.result_status == "completed"
        assert log.candidates_input == 0
        assert log.candidates_promoted == 0
        session.close()

    @patch("utils.funnel_confirmation._get_live_data")
    def test_retry_excludes_expired_candidates(self, mock_live, in_memory_engine):
        """Retry excludes candidates marked as expired."""
        from utils.funnel_confirmation import run_confirmation_retry

        mock_live.return_value = {
            "price": 152.0, "open": 150.0, "prev_close": 148.0,
            "volume": 2_000_000,
        }

        # Create an expired awaiting_confirmation candidate
        ny_tz = ZoneInfo("America/New_York")
        today_ny = datetime.now(timezone.utc).astimezone(ny_tz).date()

        session = get_session(in_memory_engine)
        candidate = FunnelCandidate(
            candidate_id=str(uuid.uuid4()),
            date=today_ny,
            symbol="OLD",
            discovered_at=datetime.now(timezone.utc) - timedelta(hours=3),
            source_run="premarket",
            selection_mode="deterministic_fallback",
            scout_rank=1,
            scout_score=80.0,
            direction_bias="bullish",
            catalyst_evidence=json.dumps({}),
            selection_reason="Test",
            primary_risk="Risk",
            sector_context=json.dumps({}),
            stage_status="awaiting_confirmation",
            stage_decisions=json.dumps([]),
            expired=True,  # Expired!
        )
        session.add(candidate)
        session.commit()
        session.close()

        funnel_config = {
            "funnel": {
                "budgets": {"market_hours_confirmation_budget_seconds": 60},
            }
        }

        decisions = run_confirmation_retry(in_memory_engine, funnel_config=funnel_config)

        # Expired candidate not evaluated
        assert decisions == []

    @patch("utils.funnel_confirmation._get_live_data")
    def test_retry_records_funnel_run_log(self, mock_live, in_memory_engine):
        """Retry records a FunnelRunLog entry with stage=confirmation_retry."""
        from utils.funnel_confirmation import run_confirmation_retry
        from db.schema import FunnelRunLog as FRL

        mock_live.return_value = {
            "price": 152.0, "open": 150.0, "prev_close": 148.0,
            "volume": 2_000_000,
        }

        _make_funnel_candidate(in_memory_engine, "AAPL", stage_status="awaiting_confirmation")
        _make_funnel_candidate(in_memory_engine, "MSFT", stage_status="awaiting_confirmation")

        funnel_config = {
            "funnel": {
                "budgets": {"market_hours_confirmation_budget_seconds": 60},
            }
        }

        run_confirmation_retry(in_memory_engine, funnel_config=funnel_config)

        session = get_session(in_memory_engine)
        log = session.query(FRL).filter_by(stage="confirmation_retry").first()
        assert log is not None
        assert log.candidates_input == 2
        assert log.candidates_promoted == 2  # Both promoted with good data
        assert log.result_status == "completed"
        assert log.duration_seconds is not None
        assert log.budget_seconds == 60
        session.close()

    @patch("utils.funnel_confirmation._get_live_data")
    def test_retry_loads_default_config_when_none(self, mock_live, in_memory_engine):
        """When funnel_config is None, loads from default path."""
        from utils.funnel_confirmation import run_confirmation_retry

        mock_live.return_value = {
            "price": 152.0, "open": 150.0, "prev_close": 148.0,
            "volume": 2_000_000,
        }

        _make_funnel_candidate(in_memory_engine, "AAPL", stage_status="awaiting_confirmation")

        # Passing None should load from default config (which has 60s budget)
        decisions = run_confirmation_retry(in_memory_engine, funnel_config=None)

        assert len(decisions) == 1
        assert decisions[0].decision == "promoted"
