"""Tests for CandidateRegistry.expire_swing_candidates() method.

Validates swing-specific expiration logic:
- Max age exceeded: created_at + SWING_MAX_CANDIDATE_AGE_HOURS < now
- Price deviation: current price vs entry price beyond threshold
- Intraday candidates are NOT affected by swing expiration
- Proper event emission and reason codes
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from utils.candidate_registry import (
    CandidateRegistry,
    CandidateState,
)


def _create_tables(engine):
    """Create minimal pm_candidates and pm_candidate_events tables."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE pm_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                setup_type TEXT NOT NULL,
                geometry_name TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                risk_reward REAL NOT NULL,
                trigger TEXT,
                invalidation_basis TEXT,
                target_basis TEXT,
                source_signal_id TEXT NOT NULL,
                signal_snapshot_json TEXT NOT NULL,
                state TEXT NOT NULL,
                integrity_hash TEXT NOT NULL,
                execution_key TEXT,
                reserved_at TEXT,
                created_at TEXT,
                expires_at TEXT NOT NULL,
                context_snapshot_json TEXT,
                benchmark_mapping_json TEXT,
                rejection_reason TEXT,
                candidate_lineage_id TEXT,
                candidate_type TEXT DEFAULT 'intraday'
            )
        """))
        conn.execute(text("""
            CREATE TABLE pm_candidate_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                created_at TEXT NOT NULL,
                candidate_type TEXT
            )
        """))


def _insert_candidate(
    engine,
    *,
    candidate_id="swing-1",
    symbol="AAPL",
    entry_price=150.0,
    candidate_type="swing",
    created_at=None,
    state="registered",
):
    """Insert a test candidate with given parameters."""
    now = datetime.now(timezone.utc)
    if created_at is None:
        created_at = now - timedelta(hours=2)
    expires_at = created_at + timedelta(hours=24)

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO pm_candidates (
                    candidate_id, cycle_id, profile_id, symbol, direction,
                    setup_type, geometry_name, entry_price, stop_price,
                    target_price, risk_reward, source_signal_id,
                    signal_snapshot_json, state, integrity_hash,
                    created_at, expires_at, candidate_type
                ) VALUES (
                    :candidate_id, 'cycle-1', 'moderate', :symbol, 'BUY',
                    'trend_pullback', 'support_bounce', :entry_price, 145.0,
                    160.0, 2.0, 'sig-1', '{}', :state, 'hash-1',
                    :created_at, :expires_at, :candidate_type
                )
            """),
            {
                "candidate_id": candidate_id,
                "symbol": symbol,
                "entry_price": entry_price,
                "candidate_type": candidate_type,
                "created_at": created_at.isoformat(),
                "expires_at": expires_at.isoformat(),
                "state": state,
            },
        )


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    _create_tables(eng)
    return eng


@pytest.fixture
def registry(engine):
    return CandidateRegistry(engine, cycle_id="cycle-1", profile_id="moderate")


class TestSwingExpirationMaxAge:
    """Tests for max-age-based swing expiration."""

    @patch("utils.gate_config.SWING_MAX_CANDIDATE_AGE_HOURS", 24)
    def test_expires_swing_past_max_age(self, engine, registry):
        """Swing candidate past max age is expired with correct reason."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=25)
        _insert_candidate(engine, candidate_id="swing-old", created_at=created_at)

        expired = registry.expire_swing_candidates()

        assert expired == ["swing-old"]
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT state, rejection_reason FROM pm_candidates WHERE candidate_id = 'swing-old'")
            ).fetchone()
        assert row[0] == CandidateState.EXPIRED.value
        assert row[1] == "max_age_exceeded"

    @patch("utils.gate_config.SWING_MAX_CANDIDATE_AGE_HOURS", 24)
    def test_does_not_expire_swing_within_max_age(self, engine, registry):
        """Swing candidate within max age is NOT expired."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=10)
        _insert_candidate(engine, candidate_id="swing-young", created_at=created_at)

        expired = registry.expire_swing_candidates()

        assert expired == []
        with engine.connect() as conn:
            state = conn.execute(
                text("SELECT state FROM pm_candidates WHERE candidate_id = 'swing-young'")
            ).scalar_one()
        assert state == CandidateState.REGISTERED.value

    @patch("utils.gate_config.SWING_MAX_CANDIDATE_AGE_HOURS", 24)
    def test_does_not_expire_intraday_candidates(self, engine, registry):
        """Intraday candidates are never affected by swing expiration."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=25)
        _insert_candidate(
            engine,
            candidate_id="intra-old",
            created_at=created_at,
            candidate_type="intraday",
        )

        expired = registry.expire_swing_candidates()

        assert expired == []
        with engine.connect() as conn:
            state = conn.execute(
                text("SELECT state FROM pm_candidates WHERE candidate_id = 'intra-old'")
            ).scalar_one()
        assert state == CandidateState.REGISTERED.value

    @patch("utils.gate_config.SWING_MAX_CANDIDATE_AGE_HOURS", 24)
    def test_emits_swing_expired_event_for_max_age(self, engine, registry):
        """swing_expired event is emitted with reason max_age_exceeded."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=25)
        _insert_candidate(engine, candidate_id="swing-ev", created_at=created_at)

        registry.expire_swing_candidates()

        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT event_type, event_data, candidate_type
                    FROM pm_candidate_events
                    WHERE candidate_id = 'swing-ev'
                """)
            ).fetchone()
        assert row[0] == "swing_expired"
        assert json.loads(row[1]) == {"reason": "max_age_exceeded"}
        assert row[2] == "swing"


class TestSwingExpirationPriceDeviation:
    """Tests for price-deviation-based swing expiration."""

    @patch("utils.gate_config.SWING_MAX_CANDIDATE_AGE_HOURS", 24)
    @patch("utils.gate_config.SWING_PRICE_DEVIATION_THRESHOLD_PCT", 3.0)
    def test_expires_swing_with_price_above_threshold(self, engine, registry):
        """Swing expired when current price deviates > threshold from entry."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        _insert_candidate(
            engine,
            candidate_id="swing-dev",
            symbol="TSLA",
            entry_price=100.0,
            created_at=created_at,
        )

        # 5% deviation > 3% threshold
        expired = registry.expire_swing_candidates(current_prices={"TSLA": 105.0})

        assert expired == ["swing-dev"]
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT state, rejection_reason FROM pm_candidates WHERE candidate_id = 'swing-dev'")
            ).fetchone()
        assert row[0] == CandidateState.EXPIRED.value
        assert row[1] == "price_moved_beyond_threshold"

    @patch("utils.gate_config.SWING_MAX_CANDIDATE_AGE_HOURS", 24)
    @patch("utils.gate_config.SWING_PRICE_DEVIATION_THRESHOLD_PCT", 3.0)
    def test_expires_swing_with_price_below_threshold_negative(self, engine, registry):
        """Swing expired when current price drops beyond threshold."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        _insert_candidate(
            engine,
            candidate_id="swing-drop",
            symbol="MSFT",
            entry_price=100.0,
            created_at=created_at,
        )

        # -4% deviation > 3% threshold
        expired = registry.expire_swing_candidates(current_prices={"MSFT": 96.0})

        assert expired == ["swing-drop"]

    @patch("utils.gate_config.SWING_MAX_CANDIDATE_AGE_HOURS", 24)
    @patch("utils.gate_config.SWING_PRICE_DEVIATION_THRESHOLD_PCT", 3.0)
    def test_does_not_expire_within_deviation_threshold(self, engine, registry):
        """Swing not expired when price is within threshold."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        _insert_candidate(
            engine,
            candidate_id="swing-ok",
            symbol="NVDA",
            entry_price=100.0,
            created_at=created_at,
        )

        # 1% deviation < 3% threshold
        expired = registry.expire_swing_candidates(current_prices={"NVDA": 101.0})

        assert expired == []
        with engine.connect() as conn:
            state = conn.execute(
                text("SELECT state FROM pm_candidates WHERE candidate_id = 'swing-ok'")
            ).scalar_one()
        assert state == CandidateState.REGISTERED.value

    @patch("utils.gate_config.SWING_MAX_CANDIDATE_AGE_HOURS", 24)
    @patch("utils.gate_config.SWING_PRICE_DEVIATION_THRESHOLD_PCT", 3.0)
    def test_skips_symbol_not_in_current_prices(self, engine, registry):
        """Swing candidate whose symbol is not in current_prices is skipped."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        _insert_candidate(
            engine,
            candidate_id="swing-no-price",
            symbol="AMD",
            entry_price=100.0,
            created_at=created_at,
        )

        expired = registry.expire_swing_candidates(current_prices={"TSLA": 200.0})

        assert expired == []

    @patch("utils.gate_config.SWING_MAX_CANDIDATE_AGE_HOURS", 24)
    @patch("utils.gate_config.SWING_PRICE_DEVIATION_THRESHOLD_PCT", 3.0)
    def test_emits_swing_expired_event_for_price_deviation(self, engine, registry):
        """swing_expired event includes price deviation details."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        _insert_candidate(
            engine,
            candidate_id="swing-pev",
            symbol="GOOG",
            entry_price=100.0,
            created_at=created_at,
        )

        registry.expire_swing_candidates(current_prices={"GOOG": 104.0})

        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT event_type, event_data, candidate_type
                    FROM pm_candidate_events
                    WHERE candidate_id = 'swing-pev'
                """)
            ).fetchone()
        assert row[0] == "swing_expired"
        data = json.loads(row[1])
        assert data["reason"] == "price_moved_beyond_threshold"
        assert data["symbol"] == "GOOG"
        assert data["entry_price"] == 100.0
        assert data["current_price"] == 104.0
        assert data["deviation_pct"] == 4.0
        assert row[2] == "swing"


class TestSwingExpirationCombined:
    """Tests combining both expiration conditions."""

    @patch("utils.gate_config.SWING_MAX_CANDIDATE_AGE_HOURS", 24)
    @patch("utils.gate_config.SWING_PRICE_DEVIATION_THRESHOLD_PCT", 3.0)
    def test_both_age_and_price_expire_different_candidates(self, engine, registry):
        """Max-age and price-deviation can expire different candidates in one call."""
        # Old candidate (will expire by age)
        created_old = datetime.now(timezone.utc) - timedelta(hours=25)
        _insert_candidate(
            engine, candidate_id="swing-age", symbol="AAPL",
            entry_price=150.0, created_at=created_old,
        )
        # Young candidate with price deviation
        created_young = datetime.now(timezone.utc) - timedelta(hours=2)
        _insert_candidate(
            engine, candidate_id="swing-price", symbol="TSLA",
            entry_price=100.0, created_at=created_young,
        )

        expired = registry.expire_swing_candidates(current_prices={"TSLA": 110.0})

        assert "swing-age" in expired
        assert "swing-price" in expired
        assert len(expired) == 2

    @patch("utils.gate_config.SWING_MAX_CANDIDATE_AGE_HOURS", 24)
    @patch("utils.gate_config.SWING_PRICE_DEVIATION_THRESHOLD_PCT", 3.0)
    def test_no_prices_only_checks_max_age(self, engine, registry):
        """When current_prices is None, only max-age check runs."""
        created_old = datetime.now(timezone.utc) - timedelta(hours=25)
        _insert_candidate(engine, candidate_id="swing-old", created_at=created_old)

        created_young = datetime.now(timezone.utc) - timedelta(hours=2)
        _insert_candidate(
            engine, candidate_id="swing-young", symbol="TSLA",
            entry_price=100.0, created_at=created_young,
        )

        expired = registry.expire_swing_candidates()  # No current_prices

        assert expired == ["swing-old"]

    @patch("utils.gate_config.SWING_MAX_CANDIDATE_AGE_HOURS", 24)
    @patch("utils.gate_config.SWING_PRICE_DEVIATION_THRESHOLD_PCT", 3.0)
    def test_does_not_expire_non_registered_swing(self, engine, registry):
        """Already-reserved swing candidates are not expired."""
        created_old = datetime.now(timezone.utc) - timedelta(hours=25)
        _insert_candidate(
            engine, candidate_id="swing-reserved",
            created_at=created_old, state="reserved",
        )

        expired = registry.expire_swing_candidates()

        assert expired == []
