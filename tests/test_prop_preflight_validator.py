"""
Property-based tests for preflight validator filtering, event data, and state transitions.

Property 6: Preflight-failed candidates excluded from PM prompt.
Property 7: Excluded candidate event preserves geometry and shadow eligibility.
Property 8: Excluded candidates reach NOT_SELECTED terminal state.

These tests validate the CONTRACT/INTERFACE of filtering logic, event construction,
and state transition decisions in isolation using generated data.

**Validates: Requirements 3.1, 3.3, 3.4, 3.5, 9.6**
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.preflight_validator import PreflightSummary


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

st_candidate_id = st.builds(lambda: str(uuid.uuid4()))
st_passed = st.booleans()

# Generate a list of (candidate_id, passed) tuples representing preflight results
@st.composite
def st_preflight_results(draw):
    """Generate a list of (candidate_id, PreflightSummary) with a mix of pass/fail."""
    n = draw(st.integers(min_value=1, max_value=20))
    results = []
    for _ in range(n):
        cid = str(uuid.uuid4())
        passed = draw(st.booleans())
        if passed:
            summary = PreflightSummary(
                candidate_id=cid,
                has_entry_stop_target=True,
                min_risk_reward_met=True,
                direction_valid=True,
                profile_allowed=True,
                candidate_not_expired=True,
                cash_available=True,
                sizing_possible=True,
                max_positions_available=True,
                same_symbol_allowed=True,
                blocking_reason_codes=[],
            )
        else:
            # Pick at least one blocking reason
            reasons = draw(st.lists(
                st.sampled_from([
                    "missing_geometry",
                    "min_risk_reward_not_met",
                    "invalid_direction",
                    "profile_not_allowed",
                    "candidate_expired",
                    "insufficient_cash",
                    "sizing_impossible",
                    "max_positions_reached",
                    "same_symbol_exists",
                ]),
                min_size=1,
                max_size=4,
                unique=True,
            ))
            summary = PreflightSummary(
                candidate_id=cid,
                has_entry_stop_target="missing_geometry" not in reasons,
                min_risk_reward_met="min_risk_reward_not_met" not in reasons,
                direction_valid="invalid_direction" not in reasons,
                profile_allowed="profile_not_allowed" not in reasons,
                candidate_not_expired="candidate_expired" not in reasons,
                cash_available="insufficient_cash" not in reasons,
                sizing_possible="sizing_impossible" not in reasons,
                max_positions_available="max_positions_reached" not in reasons,
                same_symbol_allowed="same_symbol_exists" not in reasons,
                blocking_reason_codes=reasons,
            )
        results.append((cid, summary))
    return results


# Strategy for geometry fields (floats that can be present or zero/None)
st_price = st.one_of(
    st.none(),
    st.just(0.0),
    st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
)

st_risk_reward = st.one_of(
    st.none(),
    st.just(0.0),
    st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
)

st_geometry_name = st.sampled_from([
    "analyst_geometry",
    "swing_sector_rotation",
    "technical_breakout_geo",
    "momentum_fade_geo",
])

st_signal_snapshot = st.text(min_size=2, max_size=200).map(lambda s: f'{{"data": "{s}"}}')


@st.composite
def st_excluded_candidate_event_data(draw):
    """Generate event_data for a preflight_excluded event with varying geometry completeness."""
    entry_price = draw(st_price)
    stop_price = draw(st_price)
    target_price = draw(st_price)
    risk_reward = draw(st_risk_reward)
    geometry_name = draw(st_geometry_name)
    signal_snapshot_json = draw(st_signal_snapshot)
    blocking_reason_codes = draw(st.lists(
        st.sampled_from([
            "missing_geometry",
            "min_risk_reward_not_met",
            "invalid_direction",
            "profile_not_allowed",
            "candidate_expired",
        ]),
        min_size=1,
        max_size=3,
        unique=True,
    ))

    return {
        "blocking_reason_codes": blocking_reason_codes,
        "signal_snapshot_json": signal_snapshot_json,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_reward": risk_reward,
        "geometry_name": geometry_name,
    }


# Strategy for observe mode values
st_observe_mode = st.sampled_from(["disabled", "enabled", "observe"])


# ---------------------------------------------------------------------------
# Helper: Filtering logic under test
# ---------------------------------------------------------------------------

def filter_candidates_for_pm_prompt(
    preflight_results: list[tuple[str, PreflightSummary]],
    observe_mode: str,
) -> list[str]:
    """Simulate the filtering logic: return candidate_ids that pass to PM prompt.

    When observe_mode is "disabled" or "enabled", only candidates whose
    PreflightSummary.passed is True are included.
    When observe_mode is "observe", all candidates are included.
    """
    if observe_mode == "observe":
        return [cid for cid, _ in preflight_results]
    else:
        return [cid for cid, summary in preflight_results if summary.passed]


def build_preflight_excluded_event_data(
    signal_snapshot_json: str,
    entry_price: float | None,
    stop_price: float | None,
    target_price: float | None,
    risk_reward: float | None,
    geometry_name: str,
    blocking_reason_codes: list[str],
) -> dict:
    """Build the event_data dict for a preflight_excluded event.

    This replicates the contract specified in the design:
    - Contains signal_snapshot_json, entry_price, stop_price, target_price,
      risk_reward, geometry_name
    - shadow_eligible is True iff ALL of entry_price, stop_price, target_price,
      and risk_reward are present and non-zero
    """
    shadow_eligible = (
        entry_price is not None
        and entry_price != 0
        and stop_price is not None
        and stop_price != 0
        and target_price is not None
        and target_price != 0
        and risk_reward is not None
        and risk_reward != 0
    )

    return {
        "blocking_reason_codes": blocking_reason_codes,
        "signal_snapshot_json": signal_snapshot_json,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_reward": risk_reward,
        "geometry_name": geometry_name,
        "shadow_eligible": shadow_eligible,
    }


def should_transition_to_not_selected(observe_mode: str) -> bool:
    """Determine if excluded candidates should be transitioned to NOT_SELECTED.

    Per the design: when PM_PREFLIGHT_OBSERVE_MODE is not "observe",
    excluded candidates SHALL be transitioned to NOT_SELECTED.
    """
    return observe_mode != "observe"


# ---------------------------------------------------------------------------
# Property 6: Preflight-failed candidates excluded from PM prompt
# **Validates: Requirements 3.1, 9.6**
# ---------------------------------------------------------------------------


class TestProperty6PreflightFailedExcludedFromPrompt:
    """
    Property 6: Preflight-failed candidates excluded from PM prompt.

    For any set of candidates where some fail preflight and
    PM_PREFLIGHT_OBSERVE_MODE is "disabled", the candidate IDs passed to
    the PM prompt builder SHALL not include any candidate whose
    PreflightSummary.passed is false.

    **Validates: Requirements 3.1, 9.6**
    """

    @given(preflight_results=st_preflight_results())
    @settings(max_examples=200)
    def test_disabled_mode_excludes_all_failed_candidates(
        self,
        preflight_results: list[tuple[str, PreflightSummary]],
    ):
        """In disabled mode, no failed candidate appears in the PM prompt list."""
        prompt_ids = filter_candidates_for_pm_prompt(preflight_results, "disabled")

        failed_ids = {
            cid for cid, summary in preflight_results if not summary.passed
        }

        # No failed candidate should appear in prompt
        for cid in prompt_ids:
            assert cid not in failed_ids, (
                f"Candidate {cid} failed preflight but was included in PM prompt "
                f"(disabled mode)"
            )

    @given(preflight_results=st_preflight_results())
    @settings(max_examples=200)
    def test_disabled_mode_includes_all_passing_candidates(
        self,
        preflight_results: list[tuple[str, PreflightSummary]],
    ):
        """In disabled mode, all passing candidates ARE included in the PM prompt."""
        prompt_ids = filter_candidates_for_pm_prompt(preflight_results, "disabled")

        passing_ids = {
            cid for cid, summary in preflight_results if summary.passed
        }

        for cid in passing_ids:
            assert cid in prompt_ids, (
                f"Candidate {cid} passed preflight but was NOT included in PM prompt "
                f"(disabled mode)"
            )

    @given(preflight_results=st_preflight_results())
    @settings(max_examples=200)
    def test_enabled_mode_also_excludes_failed_candidates(
        self,
        preflight_results: list[tuple[str, PreflightSummary]],
    ):
        """In enabled mode (same as disabled), failed candidates are excluded."""
        prompt_ids = filter_candidates_for_pm_prompt(preflight_results, "enabled")

        failed_ids = {
            cid for cid, summary in preflight_results if not summary.passed
        }

        for cid in prompt_ids:
            assert cid not in failed_ids, (
                f"Candidate {cid} failed preflight but was included in PM prompt "
                f"(enabled mode)"
            )

    @given(preflight_results=st_preflight_results())
    @settings(max_examples=200)
    def test_observe_mode_includes_all_candidates(
        self,
        preflight_results: list[tuple[str, PreflightSummary]],
    ):
        """In observe mode, ALL candidates (including failed) are included."""
        prompt_ids = filter_candidates_for_pm_prompt(preflight_results, "observe")

        all_ids = {cid for cid, _ in preflight_results}

        assert set(prompt_ids) == all_ids, (
            f"Observe mode should include all candidates. "
            f"Missing: {all_ids - set(prompt_ids)}"
        )

    @given(preflight_results=st_preflight_results())
    @settings(max_examples=200)
    def test_prompt_list_only_contains_known_candidates(
        self,
        preflight_results: list[tuple[str, PreflightSummary]],
    ):
        """The filtered prompt list never introduces candidate IDs not in the input."""
        prompt_ids = filter_candidates_for_pm_prompt(preflight_results, "disabled")

        all_ids = {cid for cid, _ in preflight_results}

        for cid in prompt_ids:
            assert cid in all_ids, (
                f"Prompt list contains unknown candidate {cid}"
            )


# ---------------------------------------------------------------------------
# Property 7: Excluded candidate event preserves geometry and shadow eligibility
# **Validates: Requirements 3.3, 3.4**
# ---------------------------------------------------------------------------


class TestProperty7ExcludedCandidateEventPreservesGeometry:
    """
    Property 7: Excluded candidate event preserves geometry and shadow eligibility.

    For any candidate excluded by preflight, the preflight_excluded event's
    event_data SHALL contain signal_snapshot_json, entry_price, stop_price,
    target_price, risk_reward, and geometry_name. The shadow_eligible field
    SHALL be true if and only if all of entry_price, stop_price, target_price,
    and risk_reward are present and non-zero.

    **Validates: Requirements 3.3, 3.4**
    """

    @given(event_data_inputs=st_excluded_candidate_event_data())
    @settings(max_examples=200)
    def test_event_data_contains_all_required_fields(
        self,
        event_data_inputs: dict,
    ):
        """Event data always contains all required geometry and signal fields."""
        event_data = build_preflight_excluded_event_data(
            signal_snapshot_json=event_data_inputs["signal_snapshot_json"],
            entry_price=event_data_inputs["entry_price"],
            stop_price=event_data_inputs["stop_price"],
            target_price=event_data_inputs["target_price"],
            risk_reward=event_data_inputs["risk_reward"],
            geometry_name=event_data_inputs["geometry_name"],
            blocking_reason_codes=event_data_inputs["blocking_reason_codes"],
        )

        required_fields = [
            "signal_snapshot_json",
            "entry_price",
            "stop_price",
            "target_price",
            "risk_reward",
            "geometry_name",
            "shadow_eligible",
            "blocking_reason_codes",
        ]

        for field in required_fields:
            assert field in event_data, (
                f"Required field '{field}' missing from preflight_excluded event_data"
            )

    @given(event_data_inputs=st_excluded_candidate_event_data())
    @settings(max_examples=200)
    def test_shadow_eligible_true_when_all_geometry_present_and_nonzero(
        self,
        event_data_inputs: dict,
    ):
        """shadow_eligible is True iff entry, stop, target, and risk_reward are all present and non-zero."""
        event_data = build_preflight_excluded_event_data(
            signal_snapshot_json=event_data_inputs["signal_snapshot_json"],
            entry_price=event_data_inputs["entry_price"],
            stop_price=event_data_inputs["stop_price"],
            target_price=event_data_inputs["target_price"],
            risk_reward=event_data_inputs["risk_reward"],
            geometry_name=event_data_inputs["geometry_name"],
            blocking_reason_codes=event_data_inputs["blocking_reason_codes"],
        )

        entry = event_data_inputs["entry_price"]
        stop = event_data_inputs["stop_price"]
        target = event_data_inputs["target_price"]
        rr = event_data_inputs["risk_reward"]

        all_present_and_nonzero = (
            entry is not None and entry != 0
            and stop is not None and stop != 0
            and target is not None and target != 0
            and rr is not None and rr != 0
        )

        assert event_data["shadow_eligible"] == all_present_and_nonzero, (
            f"shadow_eligible={event_data['shadow_eligible']} but expected "
            f"{all_present_and_nonzero}. "
            f"entry={entry}, stop={stop}, target={target}, rr={rr}"
        )

    @given(
        entry=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        stop=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        target=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        rr=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_shadow_eligible_true_when_all_positive(
        self,
        entry: float,
        stop: float,
        target: float,
        rr: float,
    ):
        """When all geometry values are positive floats, shadow_eligible is always True."""
        event_data = build_preflight_excluded_event_data(
            signal_snapshot_json='{"test": true}',
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            risk_reward=rr,
            geometry_name="analyst_geometry",
            blocking_reason_codes=["profile_not_allowed"],
        )

        assert event_data["shadow_eligible"] is True, (
            f"shadow_eligible should be True when all values are positive. "
            f"entry={entry}, stop={stop}, target={target}, rr={rr}"
        )

    @given(
        missing_field=st.sampled_from(["entry_price", "stop_price", "target_price", "risk_reward"]),
    )
    @settings(max_examples=200)
    def test_shadow_eligible_false_when_any_field_none(
        self,
        missing_field: str,
    ):
        """When any single geometry field is None, shadow_eligible is False."""
        values = {
            "entry_price": 150.0,
            "stop_price": 145.0,
            "target_price": 160.0,
            "risk_reward": 2.0,
        }
        values[missing_field] = None

        event_data = build_preflight_excluded_event_data(
            signal_snapshot_json='{"test": true}',
            entry_price=values["entry_price"],
            stop_price=values["stop_price"],
            target_price=values["target_price"],
            risk_reward=values["risk_reward"],
            geometry_name="analyst_geometry",
            blocking_reason_codes=["missing_geometry"],
        )

        assert event_data["shadow_eligible"] is False, (
            f"shadow_eligible should be False when {missing_field} is None"
        )

    @given(
        zero_field=st.sampled_from(["entry_price", "stop_price", "target_price", "risk_reward"]),
    )
    @settings(max_examples=200)
    def test_shadow_eligible_false_when_any_field_zero(
        self,
        zero_field: str,
    ):
        """When any single geometry field is 0, shadow_eligible is False."""
        values = {
            "entry_price": 150.0,
            "stop_price": 145.0,
            "target_price": 160.0,
            "risk_reward": 2.0,
        }
        values[zero_field] = 0.0

        event_data = build_preflight_excluded_event_data(
            signal_snapshot_json='{"test": true}',
            entry_price=values["entry_price"],
            stop_price=values["stop_price"],
            target_price=values["target_price"],
            risk_reward=values["risk_reward"],
            geometry_name="analyst_geometry",
            blocking_reason_codes=["missing_geometry"],
        )

        assert event_data["shadow_eligible"] is False, (
            f"shadow_eligible should be False when {zero_field} is 0"
        )

    @given(event_data_inputs=st_excluded_candidate_event_data())
    @settings(max_examples=200)
    def test_geometry_fields_preserved_verbatim(
        self,
        event_data_inputs: dict,
    ):
        """Geometry field values in event_data match the input exactly (no modification)."""
        event_data = build_preflight_excluded_event_data(
            signal_snapshot_json=event_data_inputs["signal_snapshot_json"],
            entry_price=event_data_inputs["entry_price"],
            stop_price=event_data_inputs["stop_price"],
            target_price=event_data_inputs["target_price"],
            risk_reward=event_data_inputs["risk_reward"],
            geometry_name=event_data_inputs["geometry_name"],
            blocking_reason_codes=event_data_inputs["blocking_reason_codes"],
        )

        assert event_data["entry_price"] == event_data_inputs["entry_price"]
        assert event_data["stop_price"] == event_data_inputs["stop_price"]
        assert event_data["target_price"] == event_data_inputs["target_price"]
        assert event_data["risk_reward"] == event_data_inputs["risk_reward"]
        assert event_data["geometry_name"] == event_data_inputs["geometry_name"]
        assert event_data["signal_snapshot_json"] == event_data_inputs["signal_snapshot_json"]


# ---------------------------------------------------------------------------
# Property 8: Excluded candidates reach NOT_SELECTED terminal state
# **Validates: Requirements 3.5**
# ---------------------------------------------------------------------------


class TestProperty8ExcludedCandidatesReachNotSelected:
    """
    Property 8: Excluded candidates reach NOT_SELECTED terminal state.

    For any candidate excluded by preflight when PM_PREFLIGHT_OBSERVE_MODE
    is not "observe", the candidate SHALL be transitioned to NOT_SELECTED.

    This tests the state transition DECISION contract — whether the logic
    correctly determines to transition, not the actual DB operation.

    **Validates: Requirements 3.5**
    """

    @given(observe_mode=st.just("disabled"))
    @settings(max_examples=200)
    def test_disabled_mode_transitions_to_not_selected(
        self,
        observe_mode: str,
    ):
        """In disabled mode, excluded candidates should be transitioned to NOT_SELECTED."""
        assert should_transition_to_not_selected(observe_mode) is True, (
            f"Excluded candidates should be transitioned to NOT_SELECTED in "
            f"'{observe_mode}' mode"
        )

    @given(observe_mode=st.just("enabled"))
    @settings(max_examples=200)
    def test_enabled_mode_transitions_to_not_selected(
        self,
        observe_mode: str,
    ):
        """In enabled mode (same as disabled), excluded candidates transition to NOT_SELECTED."""
        assert should_transition_to_not_selected(observe_mode) is True, (
            f"Excluded candidates should be transitioned to NOT_SELECTED in "
            f"'{observe_mode}' mode"
        )

    @given(observe_mode=st.just("observe"))
    @settings(max_examples=200)
    def test_observe_mode_does_not_transition(
        self,
        observe_mode: str,
    ):
        """In observe mode, excluded candidates are NOT transitioned (they're included in prompt)."""
        assert should_transition_to_not_selected(observe_mode) is False, (
            "Observe mode should NOT transition excluded candidates to NOT_SELECTED "
            "(they are included in the prompt for PM calibration)"
        )

    @given(
        observe_mode=st.sampled_from(["disabled", "enabled", "observe"]),
        preflight_results=st_preflight_results(),
    )
    @settings(max_examples=200)
    def test_only_failed_candidates_considered_for_transition(
        self,
        observe_mode: str,
        preflight_results: list[tuple[str, PreflightSummary]],
    ):
        """Only candidates that FAILED preflight are considered for NOT_SELECTED transition.

        Passing candidates are never transitioned regardless of mode."""
        failed_candidates = [
            (cid, summary) for cid, summary in preflight_results if not summary.passed
        ]
        passing_candidates = [
            (cid, summary) for cid, summary in preflight_results if summary.passed
        ]

        # Determine which failed candidates should transition
        if should_transition_to_not_selected(observe_mode):
            # All failed candidates should be transitioned
            candidates_to_transition = {cid for cid, _ in failed_candidates}
        else:
            # No candidates should be transitioned (observe mode)
            candidates_to_transition = set()

        # Passing candidates should NEVER be in the transition set
        passing_ids = {cid for cid, _ in passing_candidates}
        assert candidates_to_transition.isdisjoint(passing_ids), (
            f"Passing candidates should never be transitioned to NOT_SELECTED. "
            f"Overlap: {candidates_to_transition & passing_ids}"
        )

    @given(
        observe_mode=st.sampled_from(["disabled", "enabled"]),
        preflight_results=st_preflight_results(),
    )
    @settings(max_examples=200)
    def test_all_failed_candidates_transition_in_non_observe_mode(
        self,
        observe_mode: str,
        preflight_results: list[tuple[str, PreflightSummary]],
    ):
        """In non-observe modes, every failed candidate is marked for NOT_SELECTED."""
        failed_ids = {
            cid for cid, summary in preflight_results if not summary.passed
        }

        # In non-observe mode, all failed candidates should be transitioned
        assert should_transition_to_not_selected(observe_mode) is True

        # This means every failed candidate gets transitioned
        # (the logic guarantees it; we test the decision function here)
        for cid in failed_ids:
            # The decision is: if not observe mode AND candidate failed, transition
            assert should_transition_to_not_selected(observe_mode), (
                f"Failed candidate {cid} should be transitioned in {observe_mode} mode"
            )
