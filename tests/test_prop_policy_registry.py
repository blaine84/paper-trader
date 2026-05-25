"""
Property-based tests for Setup Time Policy Registry using Hypothesis.

Tests universal correctness properties that must hold across all valid inputs.
"""

from datetime import time

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.setup_time_policy import (
    SETUP_TIME_POLICY_REGISTRY,
    DEFAULT_POLICY,
    SetupTimePolicy,
    get_policy,
    is_extension_eligible,
)


# ---------------------------------------------------------------------------
# Property 1: Policy schema completeness
# Feature: setup-aware-exit-governance, Property 1: Policy schema completeness
# ---------------------------------------------------------------------------


class TestProperty1PolicySchemaCompleteness:
    """
    For any setup type entry in the Setup_Time_Policy registry, the entry SHALL
    contain all required fields with correct types: alert_minutes and
    force_close_minutes as positive integers, extension_eligible as boolean,
    nullable fields as int or None, and eod_hard_wall as a time value.

    If extension_eligible is False, max_extension_minutes must be None.
    If extension_eligible is True, max_extension_minutes must be a positive int.

    **Validates: Requirements 1.2**
    """

    @given(setup_type=st.sampled_from(list(SETUP_TIME_POLICY_REGISTRY.keys())))
    @settings(max_examples=200)
    def test_all_registry_entries_have_correct_types_and_required_fields(
        self, setup_type: str
    ):
        """All registry entries have correct types and required fields."""
        policy = SETUP_TIME_POLICY_REGISTRY[setup_type]

        # Must be a SetupTimePolicy instance
        assert isinstance(policy, SetupTimePolicy), (
            f"{setup_type}: expected SetupTimePolicy, got {type(policy)}"
        )

        # alert_minutes is a positive int
        assert isinstance(policy.alert_minutes, int), (
            f"{setup_type}: alert_minutes must be int, got {type(policy.alert_minutes)}"
        )
        assert policy.alert_minutes > 0, (
            f"{setup_type}: alert_minutes must be positive, got {policy.alert_minutes}"
        )

        # force_close_minutes is a positive int
        assert isinstance(policy.force_close_minutes, int), (
            f"{setup_type}: force_close_minutes must be int, got {type(policy.force_close_minutes)}"
        )
        assert policy.force_close_minutes > 0, (
            f"{setup_type}: force_close_minutes must be positive, got {policy.force_close_minutes}"
        )

        # extension_eligible is a bool
        assert isinstance(policy.extension_eligible, bool), (
            f"{setup_type}: extension_eligible must be bool, got {type(policy.extension_eligible)}"
        )

        # revalidate_minutes is int or None
        assert policy.revalidate_minutes is None or isinstance(policy.revalidate_minutes, int), (
            f"{setup_type}: revalidate_minutes must be int or None, got {type(policy.revalidate_minutes)}"
        )

        # max_extension_minutes is int or None
        assert policy.max_extension_minutes is None or isinstance(policy.max_extension_minutes, int), (
            f"{setup_type}: max_extension_minutes must be int or None, got {type(policy.max_extension_minutes)}"
        )

        # revalidation_interval_minutes is int or None
        assert policy.revalidation_interval_minutes is None or isinstance(policy.revalidation_interval_minutes, int), (
            f"{setup_type}: revalidation_interval_minutes must be int or None, got {type(policy.revalidation_interval_minutes)}"
        )

        # eod_hard_wall is a time value
        assert isinstance(policy.eod_hard_wall, time), (
            f"{setup_type}: eod_hard_wall must be datetime.time, got {type(policy.eod_hard_wall)}"
        )

        # If extension_eligible is False, max_extension_minutes must be None
        if not policy.extension_eligible:
            assert policy.max_extension_minutes is None, (
                f"{setup_type}: extension_eligible=False but max_extension_minutes={policy.max_extension_minutes} (expected None)"
            )

        # If extension_eligible is True, max_extension_minutes must be a positive int
        if policy.extension_eligible:
            assert isinstance(policy.max_extension_minutes, int) and policy.max_extension_minutes > 0, (
                f"{setup_type}: extension_eligible=True but max_extension_minutes={policy.max_extension_minutes} (expected positive int)"
            )


# ---------------------------------------------------------------------------
# Property 2: Unknown setup types receive default policy with no extensions
# Feature: setup-aware-exit-governance, Property 2: Unknown setup types receive default policy with no extensions
# ---------------------------------------------------------------------------


class TestProperty2UnknownSetupTypesReceiveDefaultPolicy:
    """
    For any setup type string not present in the Setup_Time_Policy registry
    (including empty string and random strings), the evaluator SHALL apply the
    default policy (alert_minutes=60, force_close_minutes=90,
    extension_eligible=False) and SHALL NOT grant any extension beyond the
    default force_close_minutes.

    **Validates: Requirements 1.9, 7.2, 7.3**
    """

    @given(setup_type=st.text(min_size=0, max_size=100))
    @settings(max_examples=200)
    def test_unknown_setup_types_get_default_policy(self, setup_type: str):
        """Any random string not in registry gets DEFAULT_POLICY with no extensions."""
        # Skip known setup types — we only test unknown ones
        assume(setup_type not in SETUP_TIME_POLICY_REGISTRY)

        policy = get_policy(setup_type)

        # Must return the DEFAULT_POLICY
        assert policy is DEFAULT_POLICY, (
            f"Unknown setup_type '{setup_type}' did not return DEFAULT_POLICY"
        )

        # Verify default policy values
        assert policy.alert_minutes == 60, (
            f"DEFAULT_POLICY alert_minutes should be 60, got {policy.alert_minutes}"
        )
        assert policy.force_close_minutes == 90, (
            f"DEFAULT_POLICY force_close_minutes should be 90, got {policy.force_close_minutes}"
        )
        assert policy.extension_eligible is False, (
            f"DEFAULT_POLICY extension_eligible should be False, got {policy.extension_eligible}"
        )
        assert policy.max_extension_minutes is None, (
            f"DEFAULT_POLICY max_extension_minutes should be None, got {policy.max_extension_minutes}"
        )
        assert policy.revalidation_interval_minutes is None, (
            f"DEFAULT_POLICY revalidation_interval_minutes should be None, got {policy.revalidation_interval_minutes}"
        )


# ---------------------------------------------------------------------------
# Property 3: Non-extension-eligible setups are never extended
# Feature: setup-aware-exit-governance, Property 3: Non-extension-eligible setups are never extended
# ---------------------------------------------------------------------------


class TestProperty3NonExtensionEligibleSetupsNeverExtended:
    """
    For all non-extension-eligible entries in the registry, max_extension_minutes
    is None and revalidation_interval_minutes is None.

    **Validates: Requirements 1.10**
    """

    @given(setup_type=st.sampled_from(list(SETUP_TIME_POLICY_REGISTRY.keys())))
    @settings(max_examples=200)
    def test_non_extension_eligible_setups_have_no_extension_fields(
        self, setup_type: str
    ):
        """Non-extension-eligible setups have max_extension_minutes=None and revalidation_interval_minutes=None."""
        policy = SETUP_TIME_POLICY_REGISTRY[setup_type]

        # Only check non-extension-eligible setups
        assume(not policy.extension_eligible)

        assert policy.max_extension_minutes is None, (
            f"{setup_type}: extension_eligible=False but max_extension_minutes={policy.max_extension_minutes}"
        )
        assert policy.revalidation_interval_minutes is None, (
            f"{setup_type}: extension_eligible=False but revalidation_interval_minutes={policy.revalidation_interval_minutes}"
        )
