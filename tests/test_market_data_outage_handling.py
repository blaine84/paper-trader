import json

from sqlalchemy import create_engine

from db.schema import AgentMemory, Base, TradeEvent, get_session


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
    finally:
        db.close()

    payload = json.loads(marker.value)
    assert payload["consumer"] == "momentum"
    assert payload["reason"] == "all_quote_providers_unavailable"
    assert set(payload["symbols"]) == {"AMD", "MSFT"}
    assert event.agent == "price_monitor"
