import json

from sqlalchemy import create_engine

from db.schema import AgentMemory, Base, TradeEvent, get_session
from utils.market_data_health import reset_market_data_health_throttle
from utils.cycle_coordinator import CycleContext, CycleCoordinator
from datetime import datetime, timedelta, timezone


def _engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


def test_analyst_provider_error_records_signal_unavailable_not_fresh_signal(monkeypatch):
    import agents.analyst as analyst

    engine = _engine()

    class BrokenFinnhub:
        def get_quote(self, symbol):
            raise RuntimeError(
                "HTTPSConnectionPool(host='api.finnhub.io'): "
                "Failed to resolve 'api.finnhub.io'"
            )

    monkeypatch.setattr(analyst, "FinnhubClient", BrokenFinnhub)
    monkeypatch.setattr(analyst, "process_pending_feedback", lambda engine: None)
    monkeypatch.setattr(analyst, "write_feedback_health_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(analyst, "build_strategy_context", lambda engine: "")
    monkeypatch.setattr(analyst, "build_feedback_prompt_context", lambda engine: "")
    monkeypatch.setattr(analyst, "get_active_mitigations", lambda engine: [])
    monkeypatch.setattr("utils.strategy_store.get_all_setup_types", lambda engine: ["technical_breakout"])

    result = analyst.run(engine, ["AMD"], cycle_id="cycle-outage")

    assert result["AMD"]["data_unavailable"] is True
    assert result["AMD"]["skip_signal_memory"] is True

    db = get_session(engine)
    try:
        signal_rows = (
            db.query(AgentMemory)
            .filter_by(agent="analyst", symbol="AMD", key="signal")
            .all()
        )
        unavailable = (
            db.query(AgentMemory)
            .filter_by(agent="analyst", symbol="AMD", key="signal_unavailable")
            .one()
        )
        outage_event = (
            db.query(TradeEvent)
            .filter_by(event_type="signal_unavailable", symbol="AMD")
            .one()
        )
    finally:
        db.close()

    assert signal_rows == []
    payload = json.loads(unavailable.value)
    assert payload["setup_type"] == "error"
    assert "api.finnhub.io" in payload["reasoning"]
    assert "api.finnhub.io" in outage_event.message


def test_price_monitor_empty_quote_batch_records_market_data_outage(monkeypatch):
    import agents.price_monitor as price_monitor

    engine = _engine()
    price_monitor._last_empty_quote_batch_log = 0.0
    reset_market_data_health_throttle()

    monkeypatch.setenv("WATCHLIST", "AMD,MSFT")
    monkeypatch.setattr(price_monitor, "get_batch_quotes", lambda symbols, **kwargs: {})

    alerts = price_monitor.check_momentum(engine)

    assert alerts == []

    db = get_session(engine)
    try:
        marker = (
            db.query(AgentMemory)
            .filter_by(agent="market_data", key="quote_outage")
            .one()
        )
        event = (
            db.query(TradeEvent)
            .filter_by(event_type="market_data_unavailable")
            .one()
        )
        health = (
            db.query(AgentMemory)
            .filter_by(agent="market_data", key="health_alert")
            .one()
        )
        live = (
            db.query(AgentMemory)
            .filter_by(agent="price_monitor", key="live_alerts")
            .one()
        )
    finally:
        db.close()

    payload = json.loads(marker.value)
    assert payload["consumer"] == "momentum"
    assert payload["reason"] == "all_quote_providers_unavailable"
    assert set(payload["symbols"]) == {"AMD", "MSFT"}
    assert event.agent == "price_monitor"
    assert json.loads(health.value)["severity"] == "critical"
    live_alert = json.loads(live.value)[0]
    assert live_alert["type"] == "market_data_outage"
    assert live_alert["symbol"] == "SYSTEM"


def test_price_monitor_run_returns_market_data_outages(monkeypatch):
    import agents.price_monitor as price_monitor

    engine = _engine()
    price_monitor._last_empty_quote_batch_log = 0.0
    reset_market_data_health_throttle()

    monkeypatch.setenv("WATCHLIST", "AMD,MSFT")
    monkeypatch.setattr(price_monitor, "get_batch_quotes", lambda symbols, **kwargs: {})

    result = price_monitor.run(engine)

    assert result["market_data_outages"]
    assert any(o["consumer"] == "momentum" for o in result["market_data_outages"])


def test_coordinator_all_symbol_analyst_outage_records_health_alert(monkeypatch):
    import agents.analyst as analyst

    engine = _engine()
    reset_market_data_health_throttle()

    def fake_run(engine, symbols, cycle_id=None):
        return {
            sym: {"data_unavailable": True, "skip_signal_memory": True}
            for sym in symbols
        }

    monkeypatch.setattr(analyst, "run", fake_run)

    now = datetime.now(timezone.utc)
    ctx = CycleContext(
        cycle_id="cycle-dns",
        trigger_source="scheduled",
        started_at=now,
        focused_symbols=("AMD", "MSFT"),
        decision_window_end=now + timedelta(seconds=60),
        analyst_timeout_seconds=5,
        pm_timeout_seconds=5,
        freshness_window_seconds=60,
        finnhub_budget=10,
    )

    result = CycleCoordinator(engine)._phase_analyst_refresh(ctx, ["AMD", "MSFT"])

    assert set(result) == {"AMD", "MSFT"}

    db = get_session(engine)
    try:
        health = (
            db.query(AgentMemory)
            .filter_by(agent="market_data", key="health_alert")
            .one()
        )
        live = (
            db.query(AgentMemory)
            .filter_by(agent="price_monitor", key="live_alerts")
            .one()
        )
        event = (
            db.query(TradeEvent)
            .filter_by(event_type="market_data_unavailable")
            .one()
        )
    finally:
        db.close()

    payload = json.loads(health.value)
    assert payload["source"] == "cycle_coordinator"
    assert payload["consumer"] == "analyst_refresh"
    assert payload["details"]["cycle_id"] == "cycle-dns"
    assert json.loads(live.value)[0]["type"] == "market_data_outage"
    assert event.agent == "cycle_coordinator"
