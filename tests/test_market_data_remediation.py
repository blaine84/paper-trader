import json

from utils.market_data_remediation import maybe_restart_orchestrator_after_market_data_outage


def _payload(source="price_monitor"):
    return {
        "source": source,
        "consumer": "momentum",
        "reason": "all_quote_providers_unavailable",
        "symbols": ["AMD", "MSFT"],
    }


def test_market_data_remediation_disabled_by_default(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("MARKET_DATA_OUTAGE_RESTART_ENABLED", raising=False)
    monkeypatch.setenv("MARKET_DATA_OUTAGE_RESTART_STATE_PATH", str(tmp_path / "state.json"))

    scheduled = maybe_restart_orchestrator_after_market_data_outage(
        _payload(),
        restart_func=lambda code: calls.append(code),
        delay_seconds=0,
    )

    assert scheduled is False
    assert calls == []


def test_market_data_remediation_schedules_restart_when_enabled(monkeypatch, tmp_path):
    calls = []
    state = tmp_path / "state.json"
    monkeypatch.setenv("MARKET_DATA_OUTAGE_RESTART_ENABLED", "true")
    monkeypatch.setenv("MARKET_DATA_OUTAGE_RESTART_STATE_PATH", str(state))
    monkeypatch.setenv("MARKET_DATA_OUTAGE_RESTART_EXIT_CODE", "75")

    scheduled = maybe_restart_orchestrator_after_market_data_outage(
        _payload(),
        restart_func=lambda code: calls.append(code),
        delay_seconds=0,
    )

    assert scheduled is True
    assert calls == [75]
    saved = json.loads(state.read_text())
    assert saved["payload"]["source"] == "price_monitor"


def test_market_data_remediation_respects_cooldown(monkeypatch, tmp_path):
    calls = []
    state = tmp_path / "state.json"
    monkeypatch.setenv("MARKET_DATA_OUTAGE_RESTART_ENABLED", "true")
    monkeypatch.setenv("MARKET_DATA_OUTAGE_RESTART_STATE_PATH", str(state))
    monkeypatch.setenv("MARKET_DATA_OUTAGE_RESTART_COOLDOWN_SECONDS", "900")

    first = maybe_restart_orchestrator_after_market_data_outage(
        _payload(),
        restart_func=lambda code: calls.append(code),
        delay_seconds=0,
    )
    second = maybe_restart_orchestrator_after_market_data_outage(
        _payload(),
        restart_func=lambda code: calls.append(code),
        delay_seconds=0,
    )

    assert first is True
    assert second is False
    assert calls == [75]


def test_market_data_remediation_respects_source_filter(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("MARKET_DATA_OUTAGE_RESTART_ENABLED", "true")
    monkeypatch.setenv("MARKET_DATA_OUTAGE_RESTART_SOURCES", "price_monitor")
    monkeypatch.setenv("MARKET_DATA_OUTAGE_RESTART_STATE_PATH", str(tmp_path / "state.json"))

    scheduled = maybe_restart_orchestrator_after_market_data_outage(
        _payload(source="analyst"),
        restart_func=lambda code: calls.append(code),
        delay_seconds=0,
    )

    assert scheduled is False
    assert calls == []
