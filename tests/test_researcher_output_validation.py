import json
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine

from agents import researcher
from db.schema import AgentMemory, Base, get_session


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _valid_result(symbols):
    return {
        "market_context": "Market context.",
        "market_regime": "mixed",
        "symbols": {
            sym: {
                "sentiment": "neutral",
                "confidence": "medium",
                "catalysts": [f"Catalyst for {sym}"],
                "risks": [],
                "summary": "Summary.",
            }
            for sym in symbols
        },
    }


@patch("agents.researcher.FinnhubClient")
@patch("agents.researcher.call_llm")
def test_retries_primary_tier_when_medium_output_omits_symbols(mock_llm, mock_client, engine):
    symbols = ["SPY", "AMD"]
    mock_client.return_value.get_market_news.return_value = []
    mock_client.return_value.get_news.return_value = []
    mock_client.return_value.get_quote.return_value = {}
    mock_llm.side_effect = [json.dumps({"analysis": "wrong shape"}), json.dumps(_valid_result(symbols))]

    result = researcher.run(engine, symbols)

    assert result["symbols"]["AMD"]["sentiment"] == "neutral"
    assert mock_llm.call_args_list[0].kwargs["tier"] == "medium"
    assert mock_llm.call_args_list[1].kwargs["tier"] == "high"
    db = get_session(engine)
    rows = db.query(AgentMemory).filter_by(agent="researcher", key="sentiment").all()
    assert {row.symbol for row in rows} == set(symbols)
    db.close()


@patch("agents.researcher.FinnhubClient")
@patch("agents.researcher.call_llm")
def test_does_not_write_empty_success_when_fallback_remains_incomplete(mock_llm, mock_client, engine):
    mock_client.return_value.get_market_news.return_value = []
    mock_client.return_value.get_news.return_value = []
    mock_client.return_value.get_quote.return_value = {}
    mock_llm.return_value = json.dumps({"market_context": ""})

    with pytest.raises(ValueError, match="incomplete after fallback"):
        researcher.run(engine, ["SPY"])

    db = get_session(engine)
    assert db.query(AgentMemory).filter_by(agent="researcher").count() == 0
    db.close()
