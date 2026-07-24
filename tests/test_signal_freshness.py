"""Tests for utils/signal_freshness.py — freshness gate evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, strategies as st, settings, assume
from sqlalchemy import create_engine, text

from utils.signal_freshness import (
    FreshnessResult,
    StaleSignalSkip,
    check_signal_freshness,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite engine with agent_memory table."""
    eng = create_engine("sqlite:///:memory:")
    with eng.connect() as conn:
        conn.execute(text("""
            CREATE TABLE agent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent VARCHAR(32) NOT NULL,
                symbol VARCHAR(10),
                timestamp DATETIME,
                key VARCHAR(64) NOT NULL,
                value TEXT NOT NULL
            )
        """))
        conn.commit()
    return eng


def _insert_signal(engine, symbol: str, cycle_id: str | None, timestamp: datetime):
    """Helper to insert an analyst signal into agent_memory."""
    signal_data = {"strength": 7.5, "setup_type": "momentum_fade"}
    if cycle_id is not None:
        signal_data["_cycle_id"] = cycle_id
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO agent_memory (agent, symbol, timestamp, key, value)
            VALUES (:agent, :symbol, :timestamp, :key, :value)
        """), {
            "agent": "analyst",
            "symbol": symbol,
            "timestamp": timestamp,
            "key": "signal",
            "value": json.dumps(signal_data),
        })
        conn.commit()


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Test that dataclasses are frozen and have correct fields."""

    def test_freshness_result_is_frozen(self):
        result = FreshnessResult(
            fresh_symbols=("AAPL",),
            stale_symbols=(),
            missing_symbols=(),
            error_symbols=(),
            skip_events=(),
        )
        with pytest.raises(AttributeError):
            result.fresh_symbols = ("TSLA",)  # type: ignore[misc]

    def test_stale_signal_skip_is_frozen(self):
        skip = StaleSignalSkip(
            cycle_id="20260723_1030_scheduled_a3f2",
            symbol="TSLA",
            signal_age_seconds=300.0,
            freshness_threshold_seconds=120,
            signal_timestamp=datetime(2026, 7, 23, 10, 25, tzinfo=timezone.utc),
            reason="stale_signal",
        )
        with pytest.raises(AttributeError):
            skip.reason = "other"  # type: ignore[misc]

    def test_freshness_result_fields(self):
        result = FreshnessResult(
            fresh_symbols=("AAPL", "MSFT"),
            stale_symbols=("TSLA",),
            missing_symbols=("GOOG",),
            error_symbols=("AMZN",),
            skip_events=(),
        )
        assert result.fresh_symbols == ("AAPL", "MSFT")
        assert result.stale_symbols == ("TSLA",)
        assert result.missing_symbols == ("GOOG",)
        assert result.error_symbols == ("AMZN",)


# ---------------------------------------------------------------------------
# check_signal_freshness tests
# ---------------------------------------------------------------------------


class TestCheckSignalFreshness:
    """Unit tests for check_signal_freshness()."""

    def test_empty_symbols_returns_empty_result(self, engine):
        """No symbols → empty result, no DB query."""
        result = check_signal_freshness(engine, [], "cycle_123", 120)
        assert result == FreshnessResult(
            fresh_symbols=(),
            stale_symbols=(),
            missing_symbols=(),
            error_symbols=(),
            skip_events=(),
        )

    def test_fresh_signal_same_cycle_id(self, engine):
        """Signal with matching cycle_id is classified as fresh."""
        now = datetime.now(timezone.utc)
        cycle_id = "20260723_1030_scheduled_a3f2"
        _insert_signal(engine, "AAPL", cycle_id, now - timedelta(seconds=200))

        result = check_signal_freshness(engine, ["AAPL"], cycle_id, 120)

        assert result.fresh_symbols == ("AAPL",)
        assert result.stale_symbols == ()
        assert result.missing_symbols == ()
        assert result.error_symbols == ()

    def test_fresh_signal_within_window(self, engine):
        """Signal within freshness window (different cycle) is fresh."""
        now = datetime.now(timezone.utc)
        _insert_signal(engine, "MSFT", "old_cycle_id", now - timedelta(seconds=60))

        result = check_signal_freshness(
            engine, ["MSFT"], "new_cycle_id", freshness_window_seconds=120
        )

        assert result.fresh_symbols == ("MSFT",)
        assert result.stale_symbols == ()

    def test_stale_signal_old_timestamp(self, engine):
        """Signal older than window with no cycle_id match is stale."""
        now = datetime.now(timezone.utc)
        _insert_signal(engine, "TSLA", None, now - timedelta(seconds=300))

        result = check_signal_freshness(
            engine, ["TSLA"], "new_cycle_id", freshness_window_seconds=120
        )

        assert result.fresh_symbols == ()
        assert result.stale_symbols == ("TSLA",)
        assert len(result.skip_events) == 1
        skip = result.skip_events[0]
        assert skip.symbol == "TSLA"
        assert skip.reason == "stale_signal"
        assert skip.signal_age_seconds >= 300.0

    def test_stale_signal_previous_cycle(self, engine):
        """Signal from a different cycle_id with old timestamp gets previous_cycle reason."""
        now = datetime.now(timezone.utc)
        _insert_signal(engine, "GOOG", "old_cycle_abc", now - timedelta(seconds=300))

        result = check_signal_freshness(
            engine, ["GOOG"], "new_cycle_xyz", freshness_window_seconds=120
        )

        assert result.stale_symbols == ("GOOG",)
        assert result.skip_events[0].reason == "previous_cycle"

    def test_missing_signal(self, engine):
        """Symbol with no signal in DB is classified as missing."""
        result = check_signal_freshness(
            engine, ["NVDA"], "cycle_123", freshness_window_seconds=120
        )

        assert result.fresh_symbols == ()
        assert result.missing_symbols == ("NVDA",)
        assert len(result.skip_events) == 1
        assert result.skip_events[0].reason == "missing_signal"

    def test_db_error_classifies_all_as_error(self):
        """DB error → all symbols classified as freshness_gate_error (fail-closed)."""
        broken_engine = MagicMock()
        broken_engine.connect.side_effect = RuntimeError("DB connection failed")

        result = check_signal_freshness(
            broken_engine, ["AAPL", "TSLA"], "cycle_123", 120
        )

        assert result.fresh_symbols == ()
        assert result.stale_symbols == ()
        assert result.missing_symbols == ()
        assert set(result.error_symbols) == {"AAPL", "TSLA"}
        assert all(s.reason == "freshness_gate_error" for s in result.skip_events)

    def test_json_parse_error_single_symbol(self, engine):
        """Invalid JSON for one symbol → that symbol is error, others unaffected."""
        now = datetime.now(timezone.utc)
        cycle_id = "cycle_123"

        # Insert valid signal for AAPL
        _insert_signal(engine, "AAPL", cycle_id, now - timedelta(seconds=10))

        # Insert invalid JSON for TSLA
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO agent_memory (agent, symbol, timestamp, key, value)
                VALUES (:agent, :symbol, :timestamp, :key, :value)
            """), {
                "agent": "analyst",
                "symbol": "TSLA",
                "timestamp": now - timedelta(seconds=10),
                "key": "signal",
                "value": "not valid json {{{",
            })
            conn.commit()

        result = check_signal_freshness(engine, ["AAPL", "TSLA"], cycle_id, 120)

        assert result.fresh_symbols == ("AAPL",)
        assert result.error_symbols == ("TSLA",)
        assert any(
            s.symbol == "TSLA" and s.reason == "freshness_gate_error"
            for s in result.skip_events
        )

    def test_multiple_signals_takes_latest(self, engine):
        """When multiple signals exist for a symbol, the latest is used."""
        now = datetime.now(timezone.utc)
        cycle_id = "current_cycle"

        # Insert old signal (stale)
        _insert_signal(engine, "AAPL", "old_cycle", now - timedelta(seconds=600))
        # Insert newer signal (fresh, matching cycle)
        _insert_signal(engine, "AAPL", cycle_id, now - timedelta(seconds=5))

        result = check_signal_freshness(engine, ["AAPL"], cycle_id, 120)

        assert result.fresh_symbols == ("AAPL",)

    def test_mixed_classification(self, engine):
        """Multiple symbols get correctly classified in one call."""
        now = datetime.now(timezone.utc)
        cycle_id = "test_cycle"

        # Fresh (same cycle)
        _insert_signal(engine, "AAPL", cycle_id, now - timedelta(seconds=200))
        # Fresh (within window)
        _insert_signal(engine, "MSFT", "other_cycle", now - timedelta(seconds=60))
        # Stale (old)
        _insert_signal(engine, "TSLA", "old_cycle", now - timedelta(seconds=300))
        # Missing (no insert for GOOG)

        result = check_signal_freshness(
            engine, ["AAPL", "MSFT", "TSLA", "GOOG"], cycle_id, 120
        )

        assert set(result.fresh_symbols) == {"AAPL", "MSFT"}
        assert result.stale_symbols == ("TSLA",)
        assert result.missing_symbols == ("GOOG",)

    def test_skip_event_includes_required_fields(self, engine):
        """StaleSignalSkip events contain all required fields per Req 3.4."""
        now = datetime.now(timezone.utc)
        cycle_id = "test_cycle_abc"
        _insert_signal(engine, "AMD", None, now - timedelta(seconds=500))

        result = check_signal_freshness(engine, ["AMD"], cycle_id, 120)

        assert len(result.skip_events) == 1
        skip = result.skip_events[0]
        assert skip.cycle_id == cycle_id
        assert skip.symbol == "AMD"
        assert skip.signal_age_seconds >= 500.0
        assert skip.freshness_threshold_seconds == 120
        assert isinstance(skip.signal_timestamp, datetime)
        assert skip.reason in ("stale_signal", "missing_signal", "previous_cycle", "freshness_gate_error")

    def test_all_symbols_stale_returns_empty_fresh(self, engine):
        """When all symbols fail freshness, PM gets zero candidates (Req 3.5)."""
        now = datetime.now(timezone.utc)
        _insert_signal(engine, "AAPL", "old", now - timedelta(seconds=300))
        _insert_signal(engine, "TSLA", "old", now - timedelta(seconds=400))

        result = check_signal_freshness(
            engine, ["AAPL", "TSLA"], "new_cycle", 120
        )

        assert result.fresh_symbols == ()
        assert len(result.stale_symbols) == 2

    def test_cycle_id_match_overrides_old_timestamp(self, engine):
        """Signal matching current cycle_id is fresh regardless of age (Req 3.7)."""
        now = datetime.now(timezone.utc)
        cycle_id = "current_cycle"
        # Signal is very old by wall-clock but matches cycle_id
        _insert_signal(engine, "AAPL", cycle_id, now - timedelta(seconds=9999))

        result = check_signal_freshness(engine, ["AAPL"], cycle_id, 120)

        assert result.fresh_symbols == ("AAPL",)


    def test_error_symbols_never_in_fresh_symbols(self):
        """Error symbols are never included in fresh_symbols (fail-closed invariant).

        Validates: Requirements 3.8, 3.9
        """
        broken_engine = MagicMock()
        broken_engine.connect.side_effect = RuntimeError("DB connection failed")

        result = check_signal_freshness(
            broken_engine, ["AAPL", "TSLA", "GOOG"], "cycle_123", 120
        )

        # Verify no overlap between error_symbols and fresh_symbols
        assert set(result.fresh_symbols).isdisjoint(set(result.error_symbols))
        # All symbols must be in error_symbols (DB failure is fail-closed for all)
        assert set(result.error_symbols) == {"AAPL", "TSLA", "GOOG"}
        assert result.fresh_symbols == ()

    def test_error_symbols_never_in_fresh_symbols_json_parse(self, engine):
        """JSON parse error symbol never appears in fresh_symbols.

        Validates: Requirements 3.8, 3.9
        """
        now = datetime.now(timezone.utc)
        cycle_id = "cycle_abc"

        # Insert invalid JSON for multiple symbols
        for sym in ("BAD1", "BAD2"):
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO agent_memory (agent, symbol, timestamp, key, value)
                    VALUES (:agent, :symbol, :timestamp, :key, :value)
                """), {
                    "agent": "analyst",
                    "symbol": sym,
                    "timestamp": now - timedelta(seconds=10),
                    "key": "signal",
                    "value": "{{not json}}",
                })
                conn.commit()

        # Insert a valid fresh signal
        _insert_signal(engine, "GOOD", cycle_id, now - timedelta(seconds=5))

        result = check_signal_freshness(
            engine, ["GOOD", "BAD1", "BAD2"], cycle_id, 120
        )

        # error_symbols and fresh_symbols must be disjoint
        assert set(result.fresh_symbols).isdisjoint(set(result.error_symbols))
        assert result.fresh_symbols == ("GOOD",)
        assert set(result.error_symbols) == {"BAD1", "BAD2"}


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


class TestSignalFreshnessProperties:
    """Property-based tests for freshness gate invariants.

    Validates: Requirements 3.1-3.9
    """

    @given(
        num_fresh=st.integers(min_value=0, max_value=5),
        num_stale=st.integers(min_value=0, max_value=5),
        num_missing=st.integers(min_value=0, max_value=5),
        freshness_window=st.integers(min_value=30, max_value=600),
    )
    @settings(max_examples=200)
    def test_fresh_symbols_disjoint_from_error_symbols(
        self, num_fresh, num_stale, num_missing, freshness_window
    ):
        """For any valid input, fresh_symbols ∩ error_symbols == ∅.

        This is the core fail-closed invariant: if a symbol is in error_symbols,
        it must NEVER appear in fresh_symbols. PM must never evaluate a symbol
        whose freshness could not be verified.

        Validates: Requirements 3.8, 3.9
        """
        # Build an in-memory DB per test case
        eng = create_engine("sqlite:///:memory:")
        with eng.connect() as conn:
            conn.execute(text("""
                CREATE TABLE agent_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent VARCHAR(32) NOT NULL,
                    symbol VARCHAR(10),
                    timestamp DATETIME,
                    key VARCHAR(64) NOT NULL,
                    value TEXT NOT NULL
                )
            """))
            conn.commit()

        now = datetime.now(timezone.utc)
        cycle_id = "prop_test_cycle"
        all_symbols: list[str] = []

        # Insert fresh signals (matching cycle_id)
        for i in range(num_fresh):
            sym = f"FRESH{i}"
            all_symbols.append(sym)
            signal_data = json.dumps({"_cycle_id": cycle_id, "strength": 5.0})
            with eng.connect() as conn:
                conn.execute(text("""
                    INSERT INTO agent_memory (agent, symbol, timestamp, key, value)
                    VALUES (:agent, :symbol, :timestamp, :key, :value)
                """), {
                    "agent": "analyst",
                    "symbol": sym,
                    "timestamp": now - timedelta(seconds=10),
                    "key": "signal",
                    "value": signal_data,
                })
                conn.commit()

        # Insert stale signals (old timestamp, different cycle)
        for i in range(num_stale):
            sym = f"STALE{i}"
            all_symbols.append(sym)
            signal_data = json.dumps({"_cycle_id": "old_cycle", "strength": 3.0})
            with eng.connect() as conn:
                conn.execute(text("""
                    INSERT INTO agent_memory (agent, symbol, timestamp, key, value)
                    VALUES (:agent, :symbol, :timestamp, :key, :value)
                """), {
                    "agent": "analyst",
                    "symbol": sym,
                    "timestamp": now - timedelta(seconds=freshness_window + 100),
                    "key": "signal",
                    "value": signal_data,
                })
                conn.commit()

        # Missing symbols (no insert)
        for i in range(num_missing):
            sym = f"MISS{i}"
            all_symbols.append(sym)

        # Skip if no symbols generated
        assume(len(all_symbols) > 0)

        result = check_signal_freshness(eng, all_symbols, cycle_id, freshness_window)

        # INVARIANT: fresh_symbols ∩ error_symbols == ∅
        fresh_set = set(result.fresh_symbols)
        error_set = set(result.error_symbols)
        assert fresh_set.isdisjoint(error_set), (
            f"Overlap found: {fresh_set & error_set}"
        )

        # Additional structural invariant: every input symbol appears in exactly one bucket
        stale_set = set(result.stale_symbols)
        missing_set = set(result.missing_symbols)
        all_classified = fresh_set | stale_set | missing_set | error_set
        assert all_classified == set(all_symbols), (
            f"Not all symbols classified. Input: {set(all_symbols)}, Classified: {all_classified}"
        )

    @given(
        num_symbols=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=200)
    def test_db_error_puts_all_in_error_never_fresh(self, num_symbols):
        """On DB error, ALL symbols end up in error_symbols and NONE in fresh_symbols.

        This validates the fail-closed behavior: a database failure means we
        cannot verify freshness for any symbol, so none may pass to PM.

        Validates: Requirements 3.8, 3.9
        """
        symbols = [f"SYM{i}" for i in range(num_symbols)]
        broken_engine = MagicMock()
        broken_engine.connect.side_effect = RuntimeError("Simulated DB failure")

        result = check_signal_freshness(broken_engine, symbols, "any_cycle", 120)

        # All symbols must be in error_symbols
        assert set(result.error_symbols) == set(symbols)
        # fresh_symbols must be empty (fail-closed)
        assert result.fresh_symbols == ()
        # No overlap possible
        assert set(result.fresh_symbols).isdisjoint(set(result.error_symbols))
