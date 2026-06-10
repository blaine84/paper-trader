from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agents.portfolio_manager import _run_gate_pipeline


def test_pm_passes_analyst_strength_and_confidence_to_risk_geometry():
    db = MagicMock()
    db.bind = MagicMock()
    query = db.query.return_value
    query.filter_by.return_value.order_by.return_value.first.return_value = SimpleNamespace(
        cash=100_000.0
    )
    query.filter_by.return_value.all.return_value = []

    decision = {
        "symbol": "AMD",
        "action": "BUY",
        "quantity": 10,
        "price": 100.0,
        "stop": 98.0,
        "target": 102.0,
        "setup_type": "technical_breakout",
    }
    signal = {
        "symbol": "AMD",
        "signal": "LONG",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "technical_breakout",
    }

    with (
        patch(
            "agents.portfolio_manager.evaluate_setup_quality",
            return_value={"decision": "allow", "reason": "allowed"},
        ),
        patch(
            "agents.portfolio_manager.evaluate_pre_trade_quality",
            return_value={"decision": "allow", "reason": "allowed"},
        ),
        patch(
            "agents.portfolio_manager.evaluate_catalyst_specificity",
            return_value={"decision": "allow", "reason": "allowed"},
        ),
        patch(
            "utils.atr_helper.compute_intraday_atr",
            return_value={"atr": 1.0, "timestamp": None, "source": "test"},
        ),
        patch(
            "utils.risk_geometry_gate.evaluate_risk_geometry",
            return_value={"decision": "passed_unchanged", "reason": "allowed"},
        ) as evaluate_geometry,
    ):
        proceed, _, _, _ = _run_gate_pipeline(
            db, db.bind, decision, signal, "aggressive"
        )

    assert proceed is True
    assert evaluate_geometry.call_args.kwargs["signal_strength"] == 10.0
    assert evaluate_geometry.call_args.kwargs["confidence_level"] == "high"
