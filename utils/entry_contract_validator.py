"""
Entry Contract Validator — Validates entry metadata for setup-aware exit governance.

Ensures that PM entry decisions carry sufficient exit metadata at trade open
so the Lifecycle_Evaluator can make setup-aware decisions later without
requiring additional PM input.

Thesis-development setups (news_breakout, news_catalyst, trend_pullback) require
richer metadata for full extension eligibility. Fast tactical setups
(momentum_fade, orb, short_squeeze, gap_and_go, vwap_reclaim) require only
basic stop/entry data.
"""

from utils.setup_time_policy import get_policy, is_thesis_development_setup


# ---------------------------------------------------------------------------
# Fast Tactical Setup Types
# ---------------------------------------------------------------------------

FAST_TACTICAL_SETUPS = frozenset({
    "momentum_fade",
    "orb",
    "short_squeeze",
    "gap_and_go",
    "vwap_reclaim",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_entry_for_exit_governance(
    entry: dict,
    setup_type: str,
) -> tuple[bool, str, str]:
    """Validate that an entry has sufficient metadata for setup-aware exit governance.

    Args:
        entry: The PM entry decision dict.
        setup_type: The setup type classification.

    Returns:
        (is_valid, eligibility_status, reason)
        - is_valid: True if all required metadata is present
        - eligibility_status: "reject" | "execute_without_extension" | "full_eligibility"
        - reason: Explanation of validation result

    Note: eligibility_status uses three values:
        - "full_eligibility": all metadata present, trade qualifies for extension
        - "execute_without_extension": partial metadata, trade executes but cannot extend
        - "reject": insufficient metadata, entry should not execute
    """
    if is_thesis_development_setup(setup_type):
        return _validate_thesis_development_entry(entry, setup_type)
    elif setup_type in FAST_TACTICAL_SETUPS:
        return _validate_fast_tactical_entry(entry, setup_type)
    else:
        # Unknown setup type — apply fast tactical validation as minimum
        return _validate_fast_tactical_entry(entry, setup_type)


# ---------------------------------------------------------------------------
# Internal Validators
# ---------------------------------------------------------------------------


def _validate_thesis_development_entry(
    entry: dict,
    setup_type: str,
) -> tuple[bool, str, str]:
    """Validate a thesis-development setup entry.

    Required for full_eligibility:
        setup_type, entry_price, stop_price, target_price, thesis,
        invalidation_basis, structural levels (when available but not required)

    Fallback behavior per setup type (from Requirement 5.7):
        - news_breakout/news_catalyst:
            - Missing both stop_price AND invalidation_basis → reject
            - Has stop_price but missing invalidation_basis → execute_without_extension
            - All present → full_eligibility
        - trend_pullback:
            - Missing stop_price → reject
            - Has stop_price but missing other thesis fields → execute_without_extension
            - All present → full_eligibility
    """
    stop_price = _get_stop_price(entry)
    entry_price = entry.get("entry_price")
    target_price = entry.get("target_price")
    thesis = entry.get("thesis")
    invalidation_basis = entry.get("invalidation_basis")

    # Check for entry_price — always required
    if not _is_valid_number(entry_price):
        return (False, "reject", "Missing or invalid entry_price")

    # Setup-type-specific fallback logic
    if setup_type in ("news_breakout", "news_catalyst"):
        return _validate_news_setup(
            entry, setup_type, stop_price, entry_price, target_price, thesis, invalidation_basis
        )
    elif setup_type == "trend_pullback":
        return _validate_trend_pullback(
            entry, setup_type, stop_price, entry_price, target_price, thesis, invalidation_basis
        )
    else:
        # Generic thesis-development fallback (same as trend_pullback logic)
        return _validate_trend_pullback(
            entry, setup_type, stop_price, entry_price, target_price, thesis, invalidation_basis
        )


def _validate_news_setup(
    entry: dict,
    setup_type: str,
    stop_price: float | None,
    entry_price: float,
    target_price: float | None,
    thesis: str | None,
    invalidation_basis: str | None,
) -> tuple[bool, str, str]:
    """Validate news_breakout or news_catalyst entry.

    Fallback behavior:
        - Missing both stop_price AND invalidation_basis → reject
        - Has stop_price but missing invalidation_basis → execute_without_extension
        - All required fields present → full_eligibility
    """
    has_stop = _is_valid_number(stop_price)
    has_invalidation_basis = _is_non_empty_string(invalidation_basis)

    # Critical: both stop_price and invalidation_basis missing → reject
    if not has_stop and not has_invalidation_basis:
        return (
            False,
            "reject",
            f"{setup_type} entry missing both stop_price and invalidation_basis; "
            "insufficient metadata for execution",
        )

    # Has stop_price but missing invalidation_basis → execute_without_extension
    if has_stop and not has_invalidation_basis:
        return (
            False,
            "execute_without_extension",
            f"{setup_type} entry has stop_price but missing invalidation_basis; "
            "trade may execute but is not eligible for time extension",
        )

    # Has invalidation_basis but missing stop_price → execute_without_extension
    if not has_stop and has_invalidation_basis:
        return (
            False,
            "execute_without_extension",
            f"{setup_type} entry has invalidation_basis but missing stop_price; "
            "trade may execute but is not eligible for time extension",
        )

    # Both stop_price and invalidation_basis present — check remaining fields
    has_target = _is_valid_number(target_price)
    has_thesis = _is_non_empty_string(thesis)

    if not has_target or not has_thesis:
        missing = []
        if not has_target:
            missing.append("target_price")
        if not has_thesis:
            missing.append("thesis")
        return (
            False,
            "execute_without_extension",
            f"{setup_type} entry missing: {', '.join(missing)}; "
            "trade may execute but is not eligible for time extension",
        )

    # All required fields present → full_eligibility
    return (
        True,
        "full_eligibility",
        f"{setup_type} entry has all required metadata for setup-aware exit governance",
    )


def _validate_trend_pullback(
    entry: dict,
    setup_type: str,
    stop_price: float | None,
    entry_price: float,
    target_price: float | None,
    thesis: str | None,
    invalidation_basis: str | None,
) -> tuple[bool, str, str]:
    """Validate trend_pullback entry.

    Fallback behavior:
        - Missing stop_price → reject
        - Has stop_price but missing other thesis fields → execute_without_extension
        - All required fields present → full_eligibility
    """
    has_stop = _is_valid_number(stop_price)

    # Missing stop_price → reject
    if not has_stop:
        return (
            False,
            "reject",
            f"{setup_type} entry missing stop_price; "
            "insufficient metadata for execution",
        )

    # Has stop_price — check remaining thesis fields
    has_target = _is_valid_number(target_price)
    has_thesis = _is_non_empty_string(thesis)
    has_invalidation_basis = _is_non_empty_string(invalidation_basis)

    if not has_target or not has_thesis or not has_invalidation_basis:
        missing = []
        if not has_target:
            missing.append("target_price")
        if not has_thesis:
            missing.append("thesis")
        if not has_invalidation_basis:
            missing.append("invalidation_basis")
        return (
            False,
            "execute_without_extension",
            f"{setup_type} entry has stop_price but missing: {', '.join(missing)}; "
            "trade may execute but is not eligible for time extension",
        )

    # All required fields present → full_eligibility
    return (
        True,
        "full_eligibility",
        f"{setup_type} entry has all required metadata for setup-aware exit governance",
    )


def _validate_fast_tactical_entry(
    entry: dict,
    setup_type: str,
) -> tuple[bool, str, str]:
    """Validate a fast tactical setup entry.

    Required: setup_type, entry_price, stop_price (or stop_loss)
    If all present → full_eligibility
    If stop_price missing → reject
    """
    entry_price = entry.get("entry_price")
    stop_price = _get_stop_price(entry)

    if not _is_valid_number(entry_price):
        return (False, "reject", "Missing or invalid entry_price")

    if not _is_valid_number(stop_price):
        return (
            False,
            "reject",
            f"{setup_type} entry missing stop_price; "
            "insufficient metadata for execution",
        )

    # All required fields present → full_eligibility
    return (
        True,
        "full_eligibility",
        f"{setup_type} entry has all required metadata for setup-aware exit governance",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_stop_price(entry: dict) -> float | None:
    """Extract stop price from entry, checking both 'stop_price' and 'stop_loss' fields."""
    stop = entry.get("stop_price")
    if stop is None:
        stop = entry.get("stop_loss")
    return stop


def _is_valid_number(value) -> bool:
    """Check if value is a valid non-None, non-zero number."""
    if value is None:
        return False
    try:
        return float(value) != 0.0
    except (TypeError, ValueError):
        return False


def _is_non_empty_string(value) -> bool:
    """Check if value is a non-None, non-empty string."""
    if value is None:
        return False
    if not isinstance(value, str):
        return False
    return len(value.strip()) > 0
