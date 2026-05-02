"""
Deterministic symbol classification and setup-type validation.

Classifies ticker symbols into asset classes and validates whether a
given setup_type is appropriate for that class.  No LLM involvement —
purely deterministic lookups against canonical symbol sets.

See .kiro/specs/symbol-class-validation/design.md for the full spec.
"""

import logging

logger = logging.getLogger(__name__)

# ── Canonical non-single-stock symbol sets (frozenset for O(1) lookup) ──

BROAD_ETF_SYMBOLS: frozenset[str] = frozenset({"SPY", "QQQ", "IWM", "DIA"})
SECTOR_ETF_SYMBOLS: frozenset[str] = frozenset({"XLK", "XLF", "XLE"})
COMMODITY_ETF_SYMBOLS: frozenset[str] = frozenset({"GLD", "SLV"})
BOND_ETF_SYMBOLS: frozenset[str] = frozenset({"TLT"})
VOLATILITY_ETF_SYMBOLS: frozenset[str] = frozenset({"UVXY", "VXX"})
INDEX_SYMBOLS: frozenset[str] = frozenset({"VIX"})
CURRENCY_INDEX_SYMBOLS: frozenset[str] = frozenset({"DXY"})

# ── Set of class names that trigger reclassification ──

NON_SINGLE_STOCK_CLASSES: frozenset[str] = frozenset({
    "broad_etf",
    "sector_etf",
    "commodity_etf",
    "bond_etf",
    "volatility_etf",
    "index",
    "currency_index",
})


def classify_symbol(sym: str) -> str:
    """Classify a ticker symbol into one of eight asset classes.

    Uppercases the input and checks membership in each canonical
    non-stock set.  Returns the matching class or ``"unknown"`` with
    a DEBUG log when the symbol is not in any canonical set.

    Returns one of: ``broad_etf``, ``sector_etf``, ``commodity_etf``,
    ``bond_etf``, ``volatility_etf``, ``index``, ``currency_index``,
    ``unknown``.
    """
    upper = sym.upper()

    if upper in BROAD_ETF_SYMBOLS:
        return "broad_etf"
    if upper in SECTOR_ETF_SYMBOLS:
        return "sector_etf"
    if upper in COMMODITY_ETF_SYMBOLS:
        return "commodity_etf"
    if upper in BOND_ETF_SYMBOLS:
        return "bond_etf"
    if upper in VOLATILITY_ETF_SYMBOLS:
        return "volatility_etf"
    if upper in INDEX_SYMBOLS:
        return "index"
    if upper in CURRENCY_INDEX_SYMBOLS:
        return "currency_index"

    logger.debug(
        "Symbol %s not in canonical sets, classified as 'unknown' by default",
        sym,
    )
    return "unknown"


def validate_setup_for_symbol(symbol: str, setup_type: str) -> dict:
    """Validate and potentially reclassify a setup_type for a given symbol.

    Three-way logic:

    1. **Known non-stock class + gap_and_go** → reclassify to
       ``technical_breakout`` with audit metadata.  Logs at INFO.
    2. **Unknown class + gap_and_go** → preserve ``setup_type``, attach
       low-confidence flagging metadata.  Logs a WARNING.
    3. **Everything else** → pass through unchanged with ``symbol_class``.

    Returns a dict with at minimum ``setup_type``, ``setup_reclassified``,
    and ``symbol_class``.
    """
    symbol_class = classify_symbol(symbol)

    # Case 1: Known non-single-stock class + gap_and_go → reclassify
    if symbol_class in NON_SINGLE_STOCK_CLASSES and setup_type == "gap_and_go":
        logger.info(
            "Reclassifying %s from gap_and_go to technical_breakout "
            "(symbol_class=%s)",
            symbol,
            symbol_class,
        )
        return {
            "setup_type": "technical_breakout",
            "setup_reclassified": True,
            "original_setup_type": "gap_and_go",
            "reclassification_reason": (
                f"{symbol} is classified as {symbol_class}; "
                f"gap_and_go only applies to single stocks"
            ),
            "symbol_class": symbol_class,
        }

    # Case 2: Unknown class + gap_and_go → preserve but flag
    if symbol_class == "unknown" and setup_type == "gap_and_go":
        logger.warning(
            "Unknown symbol %s assigned gap_and_go — preserving setup_type "
            "with low-confidence flag for review",
            symbol,
        )
        return {
            "setup_type": "gap_and_go",
            "setup_reclassified": False,
            "symbol_class": "unknown",
            "classification_confidence": "low",
            "needs_symbol_class_review": True,
        }

    # Case 3: Everything else → pass through
    return {
        "setup_type": setup_type,
        "setup_reclassified": False,
        "symbol_class": symbol_class,
    }
