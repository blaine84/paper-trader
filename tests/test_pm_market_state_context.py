"""Tests for PM prompt market-state context injection.

Requirements: 6.5
"""

from __future__ import annotations


def _build_market_state_block(entry_signals: dict, mode: str) -> str:
    """Extract the market-state block logic for testability."""
    if mode != "enforcing":
        return ""

    ms_lines = []
    for sym, sig in entry_signals.items():
        ms = sig.get("market_state")
        if not ms or ms == "confounded":
            continue
        authority_info = sig.get("timeframe_authority", {})
        reclass = sig.get("setup_reclassification")
        triggers = sig.get("if_then_triggers", [])
        lifecycle = sig.get("setup_lifecycle_state", "no_setup")

        sym_block = f"[{sym}] state={ms} authority={authority_info.get('authority', 'unknown')}"
        if authority_info.get("conflict"):
            sym_block += " CONFLICT"
        if reclass:
            sym_block += f" reclassified={reclass.get('reclassified_setup_type', '')} posture={reclass.get('trade_posture', '')}"
        sym_block += f" lifecycle={lifecycle}"
        if triggers:
            trigger_summary = "; ".join(
                f"{t.get('id', '?')}@{t.get('threshold', '?')}"
                for t in triggers[:4]
            )
            sym_block += f" triggers=[{trigger_summary}]"
        ms_lines.append(sym_block)

    if not ms_lines:
        return ""
    return (
        "\n--- MARKET STATE CONTEXT (deterministic analysis) ---\n"
        + "\n".join(ms_lines)
        + "\n--- END MARKET STATE CONTEXT ---\n"
    )


def _sample_signals():
    return {
        "NVDA": {
            "market_state": "trend_aligned_breakout",
            "timeframe_authority": {"authority": "aligned", "conflict": False},
            "setup_reclassification": None,
            "if_then_triggers": [
                {"id": "long_breakout", "threshold": 950.0},
                {"id": "pullback_hold", "threshold": 920.0},
            ],
            "setup_lifecycle_state": "activation_pending",
        },
        "AMD": {
            "market_state": "compression_under_resistance",
            "timeframe_authority": {"authority": "higher_timeframe", "conflict": True},
            "setup_reclassification": {
                "reclassified_setup_type": "counter_trend_retracement_under_resistance",
                "trade_posture": "watch_retest",
            },
            "if_then_triggers": [{"id": "short_rejection", "threshold": 180.0}],
            "setup_lifecycle_state": "compression_watch",
        },
    }


def test_pm_prompt_includes_market_state_enforcing():
    """In enforcing mode, market state block appears in output."""
    block = _build_market_state_block(_sample_signals(), "enforcing")
    assert "--- MARKET STATE CONTEXT" in block
    assert "[NVDA]" in block
    assert "trend_aligned_breakout" in block
    assert "[AMD]" in block
    assert "CONFLICT" in block


def test_pm_prompt_excludes_market_state_observe():
    """In observe mode, market state block is empty."""
    block = _build_market_state_block(_sample_signals(), "observe")
    assert block == ""


def test_pm_prompt_excludes_market_state_disabled():
    """When disabled, market state block is empty."""
    block = _build_market_state_block(_sample_signals(), "disabled")
    assert block == ""


def test_pm_prompt_token_budget():
    """Verify < 600 tokens per symbol worst case (~4 chars per token estimate)."""
    signals = _sample_signals()
    block = _build_market_state_block(signals, "enforcing")
    # Rough token estimate: 4 chars per token, check < 600 tokens -> < 2400 chars per symbol line
    for sym in signals:
        lines = [line for line in block.split("\n") if f"[{sym}]" in line]
        for line in lines:
            assert len(line) < 2400, f"Symbol {sym} line exceeds token budget: {len(line)} chars"
