"""Performance benchmark tests for the Watch Maturity Bridge.

Validates that bridge evaluation of 30 active watches with 10 approaching-level
alerts completes within 200ms on in-memory SQLite, and makes zero network calls.

Requirements: Design performance constraint (60-second budget, 200ms hard ceiling)
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import init_setup_watch_schema
from utils.pending_order_time import to_iso


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

NOW = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)
FUTURE = NOW + timedelta(hours=48)
CYCLE = "perf_cycle_001"

# 5 symbols, 6 watches each = 30 watches
SYMBOLS = ["AMD", "NVDA", "TSLA", "AAPL", "MSFT"]
WATCHES_PER_SYMBOL = 6


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


def _insert_watch(
    conn,
    *,
    symbol: str,
    side: str = "BUY",
    setup_type: str = "pullback_continuation",
    state: str = "watching",
    maturity_score: float = 0.0,
) -> str:
    """Insert a single watch row into the in-memory DB."""
    watch_id = str(uuid.uuid4())
    now_iso = to_iso(NOW)
    exp_iso = to_iso(FUTURE)

    maturation_conditions = json.dumps([
        {"type": "price_zone", "weight": 1.0, "params": {"low": "145.0", "high": "155.0"}},
        {"type": "regime_aligned", "weight": 1.0, "params": {"expected_regime": "bullish"}},
        {"type": "key_level_proximity", "weight": 1.0, "params": {"level": "150.0", "distance_pct": 2.0}},
    ])
    invalidation_conditions = json.dumps([
        {"type": "price_breach", "params": {"level": "140.0", "direction": "below"}},
    ])

    conn.execute(
        text(
            "INSERT INTO setup_watches "
            "(watch_id, profile_id, symbol, side, setup_type, state, "
            " thesis, source_type, source_id, source_cycle_id, "
            " maturation_conditions_json, invalidation_conditions_json, "
            " maturity_score, created_at, updated_at, expires_at, "
            " observed_cycles, integrity_hash, state_changed_at) "
            "VALUES "
            "(:wid, :pid, :sym, :side, :stype, :state, "
            " :thesis, :src_type, :src_id, :cycle_id, "
            " :mat_json, :inv_json, "
            " :score, :now, :now, :exp, "
            " :cycles, :hash, :state_changed_at)"
        ),
        {
            "wid": watch_id,
            "pid": "profile_perf",
            "sym": symbol,
            "side": side,
            "stype": setup_type,
            "state": state,
            "thesis": f"Performance test thesis for {symbol}",
            "src_type": "analyst",
            "src_id": "signal_perf",
            "cycle_id": CYCLE,
            "mat_json": maturation_conditions,
            "inv_json": invalidation_conditions,
            "score": maturity_score,
            "now": now_iso,
            "exp": exp_iso,
            "cycles": 3,
            "hash": "perf_hash_" + watch_id[:8],
            "state_changed_at": now_iso if state != "watching" else None,
        },
    )
    return watch_id


@pytest.fixture
def engine_with_watches():
    """In-memory SQLite with full schema and 30 pre-populated watches.

    30 watches across 5 symbols (6 per symbol) in various active states.
    """
    eng = create_engine("sqlite:///:memory:")
    init_setup_watch_schema(eng)

    with eng.connect() as conn:
        for i, symbol in enumerate(SYMBOLS):
            for j in range(WATCHES_PER_SYMBOL):
                # Vary states: 3 watching, 2 maturing, 1 ready per symbol
                if j < 3:
                    state = "watching"
                    score = 0.0
                elif j < 5:
                    state = "maturing"
                    score = 0.3
                else:
                    state = "ready"
                    score = 0.8

                # Alternate sides
                side = "BUY" if j % 2 == 0 else "SHORT"

                # Alternate setup types
                setup_types = [
                    "pullback_continuation",
                    "support_bounce_swing",
                    "momentum_fade",
                    "breakout_retest",
                    "failed_breakdown_reclaim",
                    "pullback_continuation",
                ]
                _insert_watch(
                    conn,
                    symbol=symbol,
                    side=side,
                    setup_type=setup_types[j],
                    state=state,
                    maturity_score=score,
                )
        conn.commit()

    return eng


def _build_approaching_alerts(count: int = 10) -> list[dict]:
    """Build approaching_level alerts distributed across symbols."""
    alerts = []
    for i in range(count):
        symbol = SYMBOLS[i % len(SYMBOLS)]
        alerts.append({
            "type": "approaching_level",
            "symbol": symbol,
            "price": 150.0 + i * 0.1,
            "level_name": "support" if i % 2 == 0 else "resistance",
            "level_value": 148.0 + i * 0.05,
            "distance_pct": 1.5 - i * 0.1,
        })
    return alerts


# ────────────────────────────────────────────────────────────────────────────
# Test: Performance benchmark (20.2, 20.3, 20.4)
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_bridge.SETUP_WATCH_REALTIME_MODE", "enabled")
def test_bridge_30_watches_10_alerts_within_200ms(engine_with_watches):
    """30 active watches + 10 approaching_level alerts completes within 200ms.

    Runs 5 iterations to account for variance. Each must individually complete
    within the 200ms ceiling. Uses in-memory SQLite with pre-populated watches.
    """
    from utils.setup_watch_bridge import evaluate_alerts

    alerts = _build_approaching_alerts(10)

    elapsed_times = []
    for _ in range(5):
        start = time.perf_counter()
        result = evaluate_alerts(engine_with_watches, alerts)
        elapsed = time.perf_counter() - start
        elapsed_times.append(elapsed)

        # Sanity: bridge actually processed something
        assert result.alerts_processed == 10

    # Assert ALL 5 runs completed within 200ms
    for i, elapsed in enumerate(elapsed_times):
        assert elapsed < 0.200, (
            f"Run {i + 1}: bridge took {elapsed * 1000:.1f}ms, "
            f"exceeding 200ms ceiling"
        )


# ────────────────────────────────────────────────────────────────────────────
# Test: Zero network calls (20.5)
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_bridge.SETUP_WATCH_REALTIME_MODE", "enabled")
def test_zero_network_calls_during_bridge_execution(engine_with_watches, monkeypatch):
    """Bridge evaluation makes zero network calls (HTTP, LLM, external APIs).

    Mocks external HTTP libraries and LLM provider calls, then asserts none
    were invoked during bridge execution.
    """
    from utils.setup_watch_bridge import evaluate_alerts

    # Mock HTTP/network libraries
    mock_requests_get = MagicMock()
    mock_requests_post = MagicMock()
    mock_httpx_get = MagicMock()
    mock_httpx_post = MagicMock()

    # Patch common network entry points
    monkeypatch.setattr("requests.get", mock_requests_get, raising=False)
    monkeypatch.setattr("requests.post", mock_requests_post, raising=False)

    # Mock LLM providers (these are the providers used per the project architecture)
    mock_openai = MagicMock()
    mock_anthropic = MagicMock()
    monkeypatch.setattr("openai.OpenAI", mock_openai, raising=False)
    monkeypatch.setattr("anthropic.Anthropic", mock_anthropic, raising=False)

    # Mock the shared LLM utility if it exists
    mock_llm_call = MagicMock()
    monkeypatch.setattr("utils.llm.call_llm", mock_llm_call, raising=False)

    alerts = _build_approaching_alerts(10)

    # Execute bridge
    result = evaluate_alerts(engine_with_watches, alerts)

    # Verify bridge actually ran (not short-circuited)
    assert result.alerts_processed == 10

    # Assert zero network calls
    mock_requests_get.assert_not_called()
    mock_requests_post.assert_not_called()
    mock_openai.assert_not_called()
    mock_anthropic.assert_not_called()
    mock_llm_call.assert_not_called()
