"""
Bug Condition Exploration Test — Property 1: Bug Condition
Known Non-Single-Stock gap_and_go Passes Through Unchanged

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 2.10, 2.11

This test encodes the EXPECTED (correct) behavior: when a known
non-single-stock symbol (ETF, index, currency index) is assigned
setup_type = "gap_and_go", the validate_setup_for_symbol() function
SHALL reclassify it to "technical_breakout" with reclassification
metadata attached.

On UNFIXED code this test is EXPECTED TO FAIL — failure confirms the
bug exists because validate_setup_for_symbol does not exist yet
(ImportError), proving there is no validation or reclassification logic.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from utils.symbol_class import validate_setup_for_symbol


# ── Canonical non-single-stock symbols (all 14 from the spec) ──

NON_SINGLE_STOCK_SYMBOLS = [
    # Broad ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # Sector ETFs
    "XLK", "XLF", "XLE",
    # Commodity ETFs
    "GLD", "SLV",
    # Bond ETFs
    "TLT",
    # Volatility ETFs
    "UVXY", "VXX",
    # Index
    "VIX",
    # Currency Index
    "DXY",
]


# ── Hypothesis strategy ──

non_stock_symbol_strategy = st.sampled_from(NON_SINGLE_STOCK_SYMBOLS)


# ── Property-based test ──

@given(symbol=non_stock_symbol_strategy)
@settings(max_examples=100, deadline=None)
def test_property_bug_condition_non_stock_gap_and_go_reclassified(symbol):
    """
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 2.10, 2.11**

    Property 1: Bug Condition — For all known non-single-stock symbols
    paired with setup_type = "gap_and_go", validate_setup_for_symbol()
    SHALL return setup_type = "technical_breakout", setup_reclassified = True,
    original_setup_type = "gap_and_go", and a non-empty reclassification_reason.

    On UNFIXED code this FAILS because validate_setup_for_symbol does not
    exist (ImportError), confirming the bug: no validation function exists
    to catch invalid setup_type/symbol-class combinations.
    """
    result = validate_setup_for_symbol(symbol, "gap_and_go")

    assert result["setup_type"] == "technical_breakout", (
        f"Bug confirmed: {symbol} + gap_and_go was NOT reclassified. "
        f"Got setup_type={result.get('setup_type')!r}. "
        f"Expected 'technical_breakout' for known non-single-stock symbol."
    )

    assert result["setup_reclassified"] is True, (
        f"Bug confirmed: {symbol} + gap_and_go missing setup_reclassified=True. "
        f"Got setup_reclassified={result.get('setup_reclassified')!r}."
    )

    assert result["original_setup_type"] == "gap_and_go", (
        f"Bug confirmed: {symbol} + gap_and_go missing original_setup_type. "
        f"Got original_setup_type={result.get('original_setup_type')!r}. "
        f"Expected 'gap_and_go'."
    )

    assert result.get("reclassification_reason"), (
        f"Bug confirmed: {symbol} + gap_and_go has empty reclassification_reason. "
        f"Got reclassification_reason={result.get('reclassification_reason')!r}."
    )


# ═══════════════════════════════════════════════════════════════════════
# Preservation Property Tests — Property 2: Preservation
# Non-Buggy Signals Pass Through Unchanged
#
# Validates: Requirements 2.9, 2.13, 3.1, 3.2, 3.3, 3.4, 3.5
#
# These tests encode the preservation contract from the design:
# all signals where the bug condition does NOT hold must persist
# unchanged. On UNFIXED code these tests FAIL (ImportError —
# validate_setup_for_symbol and classify_symbol do not exist yet).
# After implementation they MUST PASS.
# ═══════════════════════════════════════════════════════════════════════

from utils.symbol_class import classify_symbol


# ── Valid symbol classes (the eight defined in the spec) ──

VALID_SYMBOL_CLASSES = frozenset({
    "broad_etf", "sector_etf", "commodity_etf", "bond_etf",
    "volatility_etf", "index", "currency_index", "unknown",
})

# ── Non-gap_and_go setup types used in the system ──

NON_GAP_AND_GO_SETUP_TYPES = [
    "technical_breakout", "momentum_fade", "sector_rotation",
    "orb", "mean_reversion", "earnings_play",
]


# ── Hypothesis strategies ──

# Symbols guaranteed NOT to be in any canonical non-stock set
unknown_symbol_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu",)),
    min_size=1,
    max_size=10,
).filter(lambda s: s.upper() not in {
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE",
    "GLD", "SLV",
    "TLT",
    "UVXY", "VXX",
    "VIX",
    "DXY",
})

any_setup_type_strategy = st.sampled_from(
    NON_GAP_AND_GO_SETUP_TYPES + ["gap_and_go"]
)

non_gap_and_go_setup_strategy = st.sampled_from(NON_GAP_AND_GO_SETUP_TYPES)

all_symbols_strategy = st.one_of(
    non_stock_symbol_strategy,
    unknown_symbol_strategy,
)


# ── Property 2a: Unknown symbols with any setup_type — preserved unchanged ──

@given(symbol=unknown_symbol_strategy, setup_type=any_setup_type_strategy)
@settings(max_examples=100, deadline=None)
def test_property_preservation_unknown_symbol_any_setup_preserved(symbol, setup_type):
    """
    **Validates: Requirements 3.1, 3.4**

    Property 2a: For unknown symbols (any string NOT in canonical non-stock
    sets) with any setup_type (including gap_and_go), validate_setup_for_symbol()
    SHALL return setup_type preserved unchanged and setup_reclassified = False.

    On UNFIXED code this FAILS because validate_setup_for_symbol does not
    exist (ImportError).
    """
    result = validate_setup_for_symbol(symbol, setup_type)

    assert result["setup_type"] == setup_type, (
        f"Preservation violated: unknown symbol {symbol!r} with setup_type "
        f"{setup_type!r} was changed to {result.get('setup_type')!r}. "
        f"Expected setup_type to be preserved unchanged."
    )

    assert result["setup_reclassified"] is False, (
        f"Preservation violated: unknown symbol {symbol!r} with setup_type "
        f"{setup_type!r} was reclassified. "
        f"Got setup_reclassified={result.get('setup_reclassified')!r}. "
        f"Expected False."
    )


# ── Property 2b: Any symbol with non-gap_and_go setup — preserved unchanged ──

@given(symbol=all_symbols_strategy, setup_type=non_gap_and_go_setup_strategy)
@settings(max_examples=100, deadline=None)
def test_property_preservation_any_symbol_non_gap_and_go_preserved(symbol, setup_type):
    """
    **Validates: Requirements 3.2, 3.3, 3.5**

    Property 2b: For any symbol (including known non-stock symbols) with
    setup_type != "gap_and_go", validate_setup_for_symbol() SHALL return
    setup_type preserved unchanged and setup_reclassified = False.

    On UNFIXED code this FAILS because validate_setup_for_symbol does not
    exist (ImportError).
    """
    result = validate_setup_for_symbol(symbol, setup_type)

    assert result["setup_type"] == setup_type, (
        f"Preservation violated: symbol {symbol!r} with non-gap_and_go "
        f"setup_type {setup_type!r} was changed to {result.get('setup_type')!r}. "
        f"Expected setup_type to be preserved unchanged."
    )

    assert result["setup_reclassified"] is False, (
        f"Preservation violated: symbol {symbol!r} with non-gap_and_go "
        f"setup_type {setup_type!r} was reclassified. "
        f"Got setup_reclassified={result.get('setup_reclassified')!r}. "
        f"Expected False."
    )


# ── Property 2c: Unknown symbol + gap_and_go — preserved with flagging metadata ──

@given(symbol=unknown_symbol_strategy)
@settings(max_examples=100, deadline=None)
def test_property_preservation_unknown_symbol_gap_and_go_flagged(symbol):
    """
    **Validates: Requirements 2.9, 2.13**

    Property 2c: For unknown symbols with setup_type = "gap_and_go",
    validate_setup_for_symbol() SHALL preserve setup_type = "gap_and_go",
    attach classification_confidence = "low" and
    needs_symbol_class_review = True.

    On UNFIXED code this FAILS because validate_setup_for_symbol does not
    exist (ImportError).
    """
    result = validate_setup_for_symbol(symbol, "gap_and_go")

    assert result["setup_type"] == "gap_and_go", (
        f"Flagging violated: unknown symbol {symbol!r} with gap_and_go "
        f"was reclassified to {result.get('setup_type')!r}. "
        f"Expected 'gap_and_go' preserved for unknown symbols."
    )

    assert result.get("classification_confidence") == "low", (
        f"Flagging violated: unknown symbol {symbol!r} with gap_and_go "
        f"missing classification_confidence='low'. "
        f"Got {result.get('classification_confidence')!r}."
    )

    assert result.get("needs_symbol_class_review") is True, (
        f"Flagging violated: unknown symbol {symbol!r} with gap_and_go "
        f"missing needs_symbol_class_review=True. "
        f"Got {result.get('needs_symbol_class_review')!r}."
    )


# ═══════════════════════════════════════════════════════════════════════
# Classification Determinism Property — Property 3
#
# Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9
#
# For any symbol string, classify_symbol() returns the same value on
# repeated calls and the value is one of the eight valid classes.
# On UNFIXED code this FAILS (ImportError — classify_symbol does not
# exist yet).
# ═══════════════════════════════════════════════════════════════════════

@given(symbol=st.text(min_size=0, max_size=20))
@settings(max_examples=200, deadline=None)
def test_property_classification_determinism(symbol):
    """
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9**

    Property 3: Classification Determinism — For any symbol string,
    classify_symbol(sym) returns the same value on repeated calls and
    the value is one of the eight valid classes (broad_etf, sector_etf,
    commodity_etf, bond_etf, volatility_etf, index, currency_index,
    unknown).

    On UNFIXED code this FAILS because classify_symbol does not exist
    (ImportError).
    """
    result1 = classify_symbol(symbol)
    result2 = classify_symbol(symbol)

    assert result1 == result2, (
        f"Determinism violated: classify_symbol({symbol!r}) returned "
        f"{result1!r} on first call and {result2!r} on second call."
    )

    assert result1 in VALID_SYMBOL_CLASSES, (
        f"Invalid class: classify_symbol({symbol!r}) returned {result1!r}, "
        f"which is not one of the eight valid classes: {VALID_SYMBOL_CLASSES}."
    )
