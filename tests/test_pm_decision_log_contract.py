from agents.portfolio_manager import (
    MAX_PM_PORTFOLIO_NOTE_CHARS,
    MAX_PM_RATIONALE_CHARS,
    _compact_text_for_decision_log,
    _enforce_pm_decision_log_contract,
)


def test_compact_text_for_decision_log_collapses_whitespace_and_clamps():
    long_text = "\n".join([
        "DECISION SUMMARY:",
        "This is a very long explanation with multiple details about timing, catalyst quality, risk controls, and position management that should never fully spill into the operator dashboard decision log because it becomes unreadable during live trading.",
        "Second paragraph adds even more narrative and pseudo-deliberation that belongs in model scratch space, not the persisted notes feed.",
    ])

    compact = _compact_text_for_decision_log(long_text, 160)

    assert "\n" not in compact
    assert len(compact) <= 160
    assert compact.endswith("…") or compact.endswith(".")


def test_enforce_pm_decision_log_contract_clamps_notes_and_rationales():
    result = {
        "decisions": [
            {
                "symbol": "TSLA",
                "action": "BUY",
                "quantity": 1,
                "entry_price": 450.0,
                "stop_loss": 440.0,
                "target": 470.0,
                "setup_type": "news_breakout",
                "rationale": "Catalyst is strong. " * 40,
            }
        ],
        "portfolio_notes": "No new entries because the setup is extended and risk controls are binding. " * 30,
    }

    sanitized = _enforce_pm_decision_log_contract(result)

    assert len(sanitized["portfolio_notes"]) <= MAX_PM_PORTFOLIO_NOTE_CHARS
    assert len(sanitized["decisions"][0]["rationale"]) <= MAX_PM_RATIONALE_CHARS
    # Original object is not mutated.
    assert len(result["portfolio_notes"]) > MAX_PM_PORTFOLIO_NOTE_CHARS
    assert len(result["decisions"][0]["rationale"]) > MAX_PM_RATIONALE_CHARS
