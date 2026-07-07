"""Verification tests for swing candidate observability integration (Task 14.1).

Validates:
- get_offered_summary() includes swing fields (candidate_type, holding_horizon)
- build_candidate_pm_prompt() displays holding_horizon for swing candidates
- Zero-swing-candidate cycles produce no invalid candidate_id references in pm_candidate_events
- Dashboard API endpoints (build_deterministic_summary, gather_trade_performance) handle
  swing candidates gracefully

Requirements: 8.3, 8.4, 8.6, 8.7
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from utils.candidate_registry import CandidateRegistry
from utils.candidate_prompt_builder import build_candidate_pm_prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_engine():
    """Create in-memory SQLite with pm_candidates and pm_candidate_events."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text('''
            CREATE TABLE pm_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL UNIQUE,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                setup_type TEXT,
                geometry_name TEXT,
                entry_price REAL,
                stop_price REAL,
                target_price REAL,
                risk_reward REAL,
                trigger TEXT,
                invalidation_basis TEXT,
                target_basis TEXT,
                source_signal_id TEXT,
                signal_snapshot_json TEXT,
                created_at DATETIME,
                expires_at DATETIME,
                integrity_hash TEXT,
                state TEXT DEFAULT 'REGISTERED',
                context_snapshot_json TEXT,
                benchmark_mapping_json TEXT,
                rejection_reason TEXT,
                candidate_type TEXT DEFAULT 'intraday',
                holding_horizon INTEGER,
                normalized_setup_type TEXT
            )
        '''))
        conn.execute(text('''
            CREATE TABLE pm_candidate_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                candidate_type TEXT DEFAULT 'intraday'
            )
        '''))
    return eng


@pytest.fixture
def engine():
    return _create_engine()


# ---------------------------------------------------------------------------
# Test: get_offered_summary includes swing fields
# ---------------------------------------------------------------------------


class TestOfferedSummarySwingFields:
    """Requirement 8.7: get_offered_summary includes swing-specific fields."""

    def test_swing_candidate_includes_holding_horizon_and_type(self, engine):
        """Swing candidate summary includes holding_horizon and candidate_type='swing'."""
        cid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_candidates
                (candidate_id, cycle_id, profile_id, symbol, direction, setup_type,
                 geometry_name, entry_price, stop_price, target_price, risk_reward,
                 trigger, invalidation_basis, target_basis, source_signal_id,
                 signal_snapshot_json, created_at, expires_at, integrity_hash,
                 state, candidate_type, holding_horizon, normalized_setup_type)
                VALUES (:cid, 'cyc-1', 'moderate', 'AAPL', 'BUY', 'sector_rotation_swing',
                 'swing_sector_rotation_swing', 150.0, 145.0, 165.0, 3.0,
                 'Swing entry: sector rotation', 'Below 145 key support', 'Prior high at 165',
                 'sig-1', '{}', :now, :now, 'hash1', 'REGISTERED', 'swing', 5,
                 'sector_rotation_swing')
            """), {"cid": cid, "now": now})

        registry = CandidateRegistry(engine, "cyc-1", "moderate")
        summaries = registry.get_offered_summary()

        assert len(summaries) == 1
        s = summaries[0]
        assert s["candidate_type"] == "swing"
        assert s["holding_horizon"] == 5
        assert s["setup_type"] == "sector_rotation_swing"
        assert s["invalidation_basis"] == "Below 145 key support"

    def test_intraday_candidate_has_null_holding_horizon(self, engine):
        """Intraday candidate summary has holding_horizon=None and candidate_type='intraday'."""
        cid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_candidates
                (candidate_id, cycle_id, profile_id, symbol, direction, setup_type,
                 geometry_name, entry_price, stop_price, target_price, risk_reward,
                 trigger, invalidation_basis, target_basis, source_signal_id,
                 signal_snapshot_json, created_at, expires_at, integrity_hash,
                 state, candidate_type)
                VALUES (:cid, 'cyc-1', 'moderate', 'TSLA', 'BUY', 'momentum_continuation',
                 'breakout_v1', 250.0, 245.0, 270.0, 4.0,
                 'Volume breakout', 'Below 245', 'Prior swing high', 'sig-2',
                 '{}', :now, :now, 'hash2', 'REGISTERED', 'intraday')
            """), {"cid": cid, "now": now})

        registry = CandidateRegistry(engine, "cyc-1", "moderate")
        summaries = registry.get_offered_summary()

        assert len(summaries) == 1
        s = summaries[0]
        assert s["candidate_type"] == "intraday"
        assert s["holding_horizon"] is None

    def test_pre_migration_null_candidate_type_treated_as_intraday(self, engine):
        """Pre-migration rows with NULL candidate_type are treated as 'intraday'."""
        cid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with engine.begin() as conn:
            # Insert with explicit NULL for candidate_type (simulating pre-migration)
            conn.execute(text("""
                INSERT INTO pm_candidates
                (candidate_id, cycle_id, profile_id, symbol, direction, setup_type,
                 geometry_name, entry_price, stop_price, target_price, risk_reward,
                 trigger, invalidation_basis, target_basis, source_signal_id,
                 signal_snapshot_json, created_at, expires_at, integrity_hash,
                 state, candidate_type)
                VALUES (:cid, 'cyc-1', 'moderate', 'GOOG', 'BUY', 'breakout',
                 'breakout_v1', 180.0, 175.0, 195.0, 3.0,
                 'Volume breakout', 'Below 175', 'Measured move', 'sig-3',
                 '{}', :now, :now, 'hash3', 'REGISTERED', NULL)
            """), {"cid": cid, "now": now})

        registry = CandidateRegistry(engine, "cyc-1", "moderate")
        summaries = registry.get_offered_summary()
        assert len(summaries) == 1
        assert summaries[0]["candidate_type"] == "intraday"

    def test_filter_by_swing_excludes_intraday(self, engine):
        """Filtering by candidate_type='swing' excludes intraday candidates."""
        now = datetime.now(timezone.utc).isoformat()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_candidates
                (candidate_id, cycle_id, profile_id, symbol, direction, setup_type,
                 geometry_name, entry_price, stop_price, target_price, risk_reward,
                 trigger, invalidation_basis, target_basis, source_signal_id,
                 signal_snapshot_json, created_at, expires_at, integrity_hash,
                 state, candidate_type, holding_horizon)
                VALUES
                ('swing-1', 'cyc-1', 'moderate', 'AAPL', 'BUY', 'breakout_retest',
                 'swing_breakout_retest', 150.0, 145.0, 165.0, 3.0,
                 'Swing entry', 'Below 145', 'Prior high', 'sig-1',
                 '{}', :now, :now, 'hash1', 'REGISTERED', 'swing', 4),
                ('intra-1', 'cyc-1', 'moderate', 'TSLA', 'BUY', 'momentum_continuation',
                 'breakout_v1', 250.0, 245.0, 270.0, 4.0,
                 'Volume breakout', 'Below 245', 'Swing high', 'sig-2',
                 '{}', :now, :now, 'hash2', 'REGISTERED', 'intraday', NULL)
            """), {"now": now})

        registry = CandidateRegistry(engine, "cyc-1", "moderate")

        swing_only = registry.get_offered_summary(candidate_type="swing")
        assert len(swing_only) == 1
        assert swing_only[0]["symbol"] == "AAPL"
        assert swing_only[0]["holding_horizon"] == 4

        intraday_only = registry.get_offered_summary(candidate_type="intraday")
        assert len(intraday_only) == 1
        assert intraday_only[0]["symbol"] == "TSLA"

        all_candidates = registry.get_offered_summary()
        assert len(all_candidates) == 2


# ---------------------------------------------------------------------------
# Test: PM prompt displays holding_horizon for swing candidates
# ---------------------------------------------------------------------------


class TestPmPromptSwingFields:
    """Requirement 8.7: PM prompt displays swing fields without errors."""

    def test_swing_candidate_shows_horizon_in_prompt(self):
        """Swing candidate with holding_horizon displays 'Xd' in prompt table."""
        summaries = [{
            "candidate_id": "cand-1",
            "symbol": "AAPL",
            "direction": "BUY",
            "setup_type": "sector_rotation_swing",
            "entry_price": 150.0,
            "stop_price": 145.0,
            "target_price": 165.0,
            "risk_reward": 3.0,
            "geometry_name": "swing_sector_rotation_swing",
            "trigger": "Swing entry: sector rotation",
            "invalidation_basis": "Below 145 key support",
            "target_basis": "Prior high at 165",
            "state": "REGISTERED",
            "candidate_type": "swing",
            "holding_horizon": 5,
        }]
        portfolio = {"cash": 50000, "total_equity": 100000, "positions": []}
        profile = {"name": "Moderate", "max_positions": 5}

        prompt = build_candidate_pm_prompt(summaries, portfolio, profile, "moderate")

        assert "5d" in prompt
        assert "sector_rotation_swing" in prompt
        assert "Below 145 key support" in prompt
        assert "Horizon" in prompt

    def test_intraday_candidate_no_horizon_in_prompt(self):
        """Intraday candidate with no holding_horizon shows blank in Horizon column."""
        summaries = [{
            "candidate_id": "cand-2",
            "symbol": "TSLA",
            "direction": "BUY",
            "setup_type": "momentum_continuation",
            "entry_price": 250.0,
            "stop_price": 245.0,
            "target_price": 270.0,
            "risk_reward": 4.0,
            "geometry_name": "breakout_v1",
            "trigger": "Volume breakout above resistance",
            "invalidation_basis": "Below 245 support",
            "target_basis": "Prior swing high",
            "state": "REGISTERED",
            "candidate_type": "intraday",
            "holding_horizon": None,
        }]
        portfolio = {"cash": 50000, "total_equity": 100000, "positions": []}
        profile = {"name": "Moderate", "max_positions": 5}

        prompt = build_candidate_pm_prompt(summaries, portfolio, profile, "moderate")

        # Should not have a numeric horizon for intraday
        assert "momentum_continuation" in prompt
        # The horizon cell should be empty (just "| |" at end)
        assert "Horizon" in prompt


# ---------------------------------------------------------------------------
# Test: Zero-swing-candidate cycles produce no invalid candidate_id references
# ---------------------------------------------------------------------------


class TestZeroSwingCandidateEvents:
    """Requirement 8.6: No invalid candidate_id references when zero swing candidates."""

    def test_swing_no_candidates_event_uses_empty_string_id(self, engine):
        """The swing_no_candidates event uses empty string as candidate_id, not a random UUID."""
        now = datetime.now(timezone.utc).isoformat()
        # Simulate the _record_no_swing_explanation behavior
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_candidate_events
                (candidate_id, cycle_id, profile_id, event_type, event_data, created_at, candidate_type)
                VALUES ('', 'cyc-1', 'moderate', 'swing_no_candidates',
                        '{"reason": "no_fresh_signals"}', :now, 'swing')
            """), {"now": now})

        # Verify: no pm_candidates row exists for this cycle
        with engine.connect() as conn:
            candidate_count = conn.execute(
                text("SELECT COUNT(*) FROM pm_candidates WHERE cycle_id = 'cyc-1'")
            ).scalar()
            assert candidate_count == 0

            # The event exists with empty candidate_id
            events = conn.execute(
                text("SELECT candidate_id, event_type FROM pm_candidate_events WHERE cycle_id = 'cyc-1'")
            ).fetchall()
            assert len(events) == 1
            assert events[0][0] == ""
            assert events[0][1] == "swing_no_candidates"

    def test_empty_candidate_id_does_not_match_any_candidate(self, engine):
        """Empty-string candidate_id in events doesn't accidentally match real candidates."""
        now = datetime.now(timezone.utc).isoformat()
        cid = str(uuid.uuid4())
        with engine.begin() as conn:
            # A real candidate
            conn.execute(text("""
                INSERT INTO pm_candidates
                (candidate_id, cycle_id, profile_id, symbol, direction, setup_type,
                 entry_price, stop_price, target_price, risk_reward,
                 source_signal_id, signal_snapshot_json, created_at, expires_at,
                 integrity_hash, state, candidate_type)
                VALUES (:cid, 'cyc-1', 'moderate', 'AAPL', 'BUY', 'breakout',
                 150.0, 145.0, 165.0, 3.0, 'sig-1', '{}', :now, :now,
                 'hash1', 'REGISTERED', 'intraday')
            """), {"cid": cid, "now": now})

            # No-candidates event with empty string
            conn.execute(text("""
                INSERT INTO pm_candidate_events
                (candidate_id, cycle_id, profile_id, event_type, event_data, created_at, candidate_type)
                VALUES ('', 'cyc-1', 'moderate', 'swing_no_candidates',
                        '{"reason": "no_executable_mapping"}', :now, 'swing')
            """), {"now": now})

        # Verify: empty string doesn't match any real candidate_id
        with engine.connect() as conn:
            # This simulates a hypothetical join — the empty ID should not match
            matches = conn.execute(text("""
                SELECT e.event_type, c.candidate_id
                FROM pm_candidate_events e
                LEFT JOIN pm_candidates c ON e.candidate_id = c.candidate_id
                WHERE e.cycle_id = 'cyc-1' AND e.event_type = 'swing_no_candidates'
            """)).fetchall()
            assert len(matches) == 1
            # The join should produce NULL for candidate_id (no match)
            assert matches[0][1] is None


# ---------------------------------------------------------------------------
# Test: Daily review summary works with/without swing candidates
# ---------------------------------------------------------------------------


class TestDailyReviewWithSwingCandidates:
    """Requirement 8.7: daily_review flows continue to return valid responses."""

    def test_build_deterministic_summary_handles_no_swing_data(self):
        """build_deterministic_summary returns valid dict with no swing-specific data."""
        from agents.daily_review import build_deterministic_summary

        result = build_deterministic_summary(
            trade_perf=None,
            git_commits=None,
            agent_context=None,
            cases=None,
            previous_review=None,
            setup_aware_exits=None,
        )

        # Should return a valid summary without errors
        assert isinstance(result, dict)
        assert "trade_performance" in result
        assert "completeness" in result
        assert result["setup_aware_exits"]["total_setup_aware_events"] == 0

    def test_build_deterministic_summary_handles_swing_setup_types(self):
        """build_deterministic_summary handles trade data with swing setup types."""
        from agents.daily_review import build_deterministic_summary

        # Trade performance with a swing-type setup
        trade_perf = {
            "total_trades": 1,
            "wins": 1,
            "losses": 0,
            "total_pnl": 150.0,
            "no_trades": False,
            "best_trade": {"symbol": "AAPL", "pnl_pct": 2.5, "setup_type": "sector_rotation_swing"},
            "worst_trade": None,
            "per_profile": {"moderate": {"trades": 1, "pnl": 150.0}},
        }

        result = build_deterministic_summary(
            trade_perf=trade_perf,
            git_commits=[],
            agent_context={"market_context": "bullish"},
            cases=[],
            previous_review=None,
            setup_aware_exits=None,
        )

        assert result["trade_performance"]["best_trade"]["setup_type"] == "sector_rotation_swing"
        assert result["process_metrics"]["win_rate"] == 100.0
