from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from db.schema import init_db
from utils.shadow_ledger import ensure_shadow_ledger_schema, record_blocked_candidate
from utils import shadow_outcomes
from utils.shadow_outcomes import score_blocked_candidate, update_blocked_candidate_outcomes


def _candles(start, closes):
    rows = []
    for i, close in enumerate(closes):
        rows.append({
            "timestamp": start + timedelta(minutes=i),
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
        })
    return rows


def test_provider_candle_fetcher_uses_shared_candle_client(monkeypatch):
    start = datetime(2026, 6, 23, 14, 30, tzinfo=timezone.utc)
    end = start + timedelta(minutes=15)
    calls = []

    class FakeFinnhubClient:
        def get_candles(self, symbol, resolution, days):
            calls.append((symbol, resolution, days))
            return {
                "symbol": symbol,
                "resolution": resolution,
                "timestamps": [
                    int((start - timedelta(minutes=1)).timestamp()),
                    int(start.timestamp()),
                    int((start + timedelta(minutes=5)).timestamp()),
                    int((end + timedelta(minutes=3)).timestamp()),
                ],
                "open": [99.0, 100.0, 101.0, 102.0],
                "high": [99.5, 100.5, 101.5, 102.5],
                "low": [98.5, 99.5, 100.5, 101.5],
                "close": [99.2, 100.2, 101.2, 102.2],
                "volume": [100, 200, 300, 400],
                "source": "alpaca",
            }

    monkeypatch.setattr("utils.finnhub_client.FinnhubClient", FakeFinnhubClient)

    candles = shadow_outcomes._fetch_provider_candles("SPY", start, end)

    assert calls
    assert calls[0][0] == "SPY"
    assert calls[0][1] == "1"
    assert [c["close"] for c in candles] == [100.2, 101.2]


def test_score_blocked_candidate_classifies_saved_us_when_stop_hits_first():
    created = datetime(2026, 5, 20, 14, 0, tzinfo=timezone.utc)
    candidate = {
        "id": 1,
        "created_at": created,
        "symbol": "NVDA",
        "action": "BUY",
        "direction": "long",
        "entry_price": 100.0,
        "stop_price": 99.0,
        "target_price": 105.0,
        "quantity": 10,
    }
    candles = _candles(created, [100.0, 99.8, 98.9, 99.2])

    outcome = score_blocked_candidate(candidate, window_label="15m", window_minutes=15, candles=candles)

    assert outcome["first_hit"] == "stop"
    assert outcome["outcome_label"] == "would_hit_stop"
    assert outcome["gate_verdict"] == "saved_us"
    assert outcome["pnl_pct"] < 0


def test_update_blocked_candidate_outcomes_inserts_companion_rows(tmp_path):
    db_path = tmp_path / "paper.db"
    engine = init_db(str(db_path))
    ensure_shadow_ledger_schema(engine)

    created = datetime(2026, 5, 20, 14, 0, tzinfo=timezone.utc)
    now = created + timedelta(minutes=61)

    def candle_fetcher(symbol, start, end):
        assert symbol == "NVDA"
        return _candles(created, [100.0] * 20 + [101.0] * 20 + [102.0] * 25)

    Session = sessionmaker(bind=engine)
    db = Session()
    candidate_id = record_blocked_candidate(
        db,
        "NVDA",
        "BUY",
        "risk_geometry_gate",
        "Adjusted R:R below minimum",
        direction="long",
        profile="moderate",
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
    )
    db.commit()
    db.close()

    with engine.begin() as conn:
        conn.execute(
            text("UPDATE blocked_trade_candidates SET created_at = :created WHERE id = :id"),
            {"created": created.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"), "id": candidate_id},
        )

    result = update_blocked_candidate_outcomes(engine, now=now, candle_fetcher=candle_fetcher)

    assert result["inserted"] == 3
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT eval_window, gate_verdict FROM blocked_trade_candidate_outcomes ORDER BY eval_window")
        ).fetchall()
    assert {r[0] for r in rows} == {"15m", "30m", "60m"}


def test_update_blocked_candidate_outcomes_closes_preopen_window_without_fetching(tmp_path):
    engine = init_db(str(tmp_path / "paper.db"))
    ensure_shadow_ledger_schema(engine)
    created = datetime(2026, 5, 20, 13, 0, tzinfo=timezone.utc)  # 09:00 ET

    Session = sessionmaker(bind=engine)
    db = Session()
    candidate_id = record_blocked_candidate(
        db,
        "XLE",
        "BUY",
        "risk_geometry_gate",
        "Adjusted R:R below minimum",
        direction="long",
        profile="moderate",
        entry_price=60.82,
        stop_price=60.70,
        target_price=61.06,
    )
    db.commit()
    db.close()

    with engine.begin() as conn:
        conn.execute(
            text("UPDATE blocked_trade_candidates SET created_at = :created WHERE id = :id"),
            {"created": created.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"), "id": candidate_id},
        )

    def candle_fetcher(*args):
        raise AssertionError("pre-open 15m window must not request market candles")

    result = update_blocked_candidate_outcomes(
        engine,
        now=created + timedelta(minutes=16),
        candle_fetcher=candle_fetcher,
    )

    assert result["inserted"] == 1
    with engine.connect() as conn:
        outcome = conn.execute(
            text(
                """
                SELECT eval_window, outcome_label, gate_verdict
                FROM blocked_trade_candidate_outcomes
                WHERE blocked_candidate_id = :id
                """
            ),
            {"id": candidate_id},
        ).fetchone()
    assert outcome == ("15m", "unscorable_no_regular_session", "unscorable")
