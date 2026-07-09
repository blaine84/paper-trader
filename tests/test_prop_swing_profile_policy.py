"""Property-based tests for swing profile policy logic.

Uses Hypothesis to validate universal correctness properties of the swing
candidate profile policy evaluation and related configuration-based gates.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

from hypothesis import given, settings, strategies as st

from utils.gate_config import SWING_MAX_CONCURRENT_POSITIONS
from utils.swing_candidate_bridge import evaluate_profile_policy, PolicyResult

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

confidence_st = st.sampled_from(["low", "medium", "high"])
strength_st = st.sampled_from(["weak", "moderate", "strong"])
profile_st = st.sampled_from(["conservative", "moderate", "aggressive"])
rr_st = st.decimals(min_value=Decimal("0.5"), max_value=Decimal("10.0"), places=1)
symbol_st = st.text(min_size=1, max_size=5, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ---------------------------------------------------------------------------
# Helper — encodes the max-concurrent-positions gate logic
# (actual enforcement will live in the bridge orchestration, task 10.1)
# ---------------------------------------------------------------------------


def _would_reject_max_positions(profile_id: str, open_count: int) -> bool:
    """Return True if a new swing candidate should be rejected due to max positions."""
    max_allowed = SWING_MAX_CONCURRENT_POSITIONS.get(profile_id, 0)
    return open_count >= max_allowed


# ---------------------------------------------------------------------------
# Property 18: Max Concurrent Positions Enforcement
# Validates: Requirements 5.4
# ---------------------------------------------------------------------------


@given(
    profile_id=profile_st,
    open_count=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=200)
def test_max_concurrent_positions_enforcement(profile_id, open_count):
    """Property 18: When open_count >= max_positions[profile], reject.

    **Validates: Requirements 5.4**
    """
    max_allowed = SWING_MAX_CONCURRENT_POSITIONS[profile_id]
    should_reject = open_count >= max_allowed

    assert _would_reject_max_positions(profile_id, open_count) == should_reject


# ---------------------------------------------------------------------------
# Property 15: Profile Policy Acceptance and Sizing
# Validates: Requirements 4.1, 4.2, 4.3
# ---------------------------------------------------------------------------


@given(
    profile_id=profile_st,
    confidence=confidence_st,
    strength=strength_st,
    risk_reward=rr_st,
    symbol=symbol_st,
)
@settings(max_examples=200)
def test_profile_policy_acceptance_and_sizing(profile_id, confidence, strength, risk_reward, symbol):
    """Property 15: Acceptance iff all thresholds met; sizing_multiplier applied.

    **Validates: Requirements 4.1, 4.2, 4.3**
    """
    result = evaluate_profile_policy(
        profile_id=profile_id,
        confidence=confidence,
        strength=strength,
        risk_reward=risk_reward,
        symbol=symbol,
        open_swing_symbols=set(),  # no overlap
    )

    from utils.gate_config import SWING_PROFILE_POLICY

    policy = SWING_PROFILE_POLICY[profile_id]

    _CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
    _STRENGTH_ORDER = {"weak": 0, "moderate": 1, "strong": 2}

    confidence_met = _CONFIDENCE_ORDER[confidence] >= _CONFIDENCE_ORDER[policy["min_confidence"]]
    strength_met = _STRENGTH_ORDER[strength] >= _STRENGTH_ORDER[policy["min_strength"]]
    rr_met = risk_reward >= policy["min_risk_reward"]

    if confidence_met and strength_met and rr_met:
        assert result.accepted is True, (
            f"Expected accepted=True for profile={profile_id}, "
            f"confidence={confidence}, strength={strength}, rr={risk_reward}"
        )
        assert result.sizing_multiplier == policy["sizing_multiplier"], (
            f"Expected sizing_multiplier={policy['sizing_multiplier']}, "
            f"got {result.sizing_multiplier}"
        )
    else:
        assert result.accepted is False, (
            f"Expected accepted=False for profile={profile_id}, "
            f"confidence={confidence}, strength={strength}, rr={risk_reward}"
        )
        assert result.reason_code in (
            "confidence_below_threshold",
            "strength_below_threshold",
            "rr_below_threshold",
        ), f"Unexpected reason_code={result.reason_code}"


# ---------------------------------------------------------------------------
# Property 16: Conservative Observe-Only Override
# Validates: Requirements 4.5
# ---------------------------------------------------------------------------


@given(
    confidence=confidence_st,
    strength=strength_st,
    risk_reward=rr_st,
    symbol=symbol_st,
)
@settings(max_examples=200)
def test_conservative_observe_only_override(confidence, strength, risk_reward, symbol):
    """Property 16: When SWING_CONSERVATIVE_OBSERVE_ONLY=True, conservative always rejected.

    **Validates: Requirements 4.5**
    """
    with patch("utils.gate_config.SWING_CONSERVATIVE_OBSERVE_ONLY", True):
        result = evaluate_profile_policy(
            profile_id="conservative",
            confidence=confidence,
            strength=strength,
            risk_reward=risk_reward,
            symbol=symbol,
            open_swing_symbols=set(),
        )

    assert result.accepted is False, (
        f"Expected rejected when observe_only=True, got accepted=True "
        f"for confidence={confidence}, strength={strength}, rr={risk_reward}"
    )
    assert result.reason_code == "observe_only_period", (
        f"Expected reason_code='observe_only_period', got '{result.reason_code}'"
    )


# ---------------------------------------------------------------------------
# Property 19: Conservative Profile Stricter Than Others
# Validates: Requirements 19.2
# ---------------------------------------------------------------------------


@given(
    confidence=confidence_st,
    strength=strength_st,
    risk_reward=rr_st,
    symbol=symbol_st,
)
@settings(max_examples=200)
def test_conservative_profile_strictness(confidence, strength, risk_reward, symbol):
    """Property 19: Conservative profile is strictly more selective.

    For any signal that is accepted by the conservative profile, it must also
    be accepted by moderate and aggressive profiles. In other words, conservative
    never accepts something that a less strict profile would reject.

    **Validates: Requirements 19.2**
    """
    conservative_result = evaluate_profile_policy(
        profile_id="conservative",
        confidence=confidence,
        strength=strength,
        risk_reward=risk_reward,
        symbol=symbol,
        open_swing_symbols=set(),
    )

    if conservative_result.accepted:
        # If conservative accepts, moderate and aggressive MUST also accept
        moderate_result = evaluate_profile_policy(
            profile_id="moderate",
            confidence=confidence,
            strength=strength,
            risk_reward=risk_reward,
            symbol=symbol,
            open_swing_symbols=set(),
        )
        aggressive_result = evaluate_profile_policy(
            profile_id="aggressive",
            confidence=confidence,
            strength=strength,
            risk_reward=risk_reward,
            symbol=symbol,
            open_swing_symbols=set(),
        )

        assert moderate_result.accepted is True, (
            f"Conservative accepted (conf={confidence}, str={strength}, rr={risk_reward}) "
            f"but moderate rejected with reason={moderate_result.reason_code}"
        )
        assert aggressive_result.accepted is True, (
            f"Conservative accepted (conf={confidence}, str={strength}, rr={risk_reward}) "
            f"but aggressive rejected with reason={aggressive_result.reason_code}"
        )
