"""Unit tests for Setup Time Policy Registry exact values.

Validates that each setup type's policy matches the exact values specified
in the requirements document:
- Requirement 1.5: news_breakout policy values
- Requirement 1.6: news_catalyst policy values
- Requirement 1.7: Fast tactical setup constraints
- Requirement 1.8: Default/unknown fallback values
- Requirement 1.11: trend_pullback policy values
"""

from datetime import time

from utils.setup_time_policy import (
    DEFAULT_POLICY,
    SETUP_TIME_POLICY_REGISTRY,
    THESIS_DEVELOPMENT_SETUPS,
    SetupTimePolicy,
    get_policy,
    is_extension_eligible,
    is_thesis_development_setup,
)


class TestNewsBreakoutPolicy:
    """Test news_breakout policy matches exact values from Requirement 1.5."""

    def test_alert_minutes(self):
        policy = get_policy("news_breakout")
        assert policy.alert_minutes == 60

    def test_revalidate_minutes(self):
        policy = get_policy("news_breakout")
        assert policy.revalidate_minutes == 90

    def test_force_close_minutes(self):
        policy = get_policy("news_breakout")
        assert policy.force_close_minutes == 120

    def test_extension_eligible(self):
        policy = get_policy("news_breakout")
        assert policy.extension_eligible is True

    def test_max_extension_minutes(self):
        policy = get_policy("news_breakout")
        assert policy.max_extension_minutes == 180

    def test_revalidation_interval_minutes(self):
        policy = get_policy("news_breakout")
        assert policy.revalidation_interval_minutes == 30

    def test_eod_hard_wall(self):
        policy = get_policy("news_breakout")
        assert policy.eod_hard_wall == time(15, 45)

    def test_setup_type_field(self):
        policy = get_policy("news_breakout")
        assert policy.setup_type == "news_breakout"


class TestNewsCatalystPolicy:
    """Test news_catalyst policy matches exact values from Requirement 1.6."""

    def test_alert_minutes(self):
        policy = get_policy("news_catalyst")
        assert policy.alert_minutes == 60

    def test_revalidate_minutes(self):
        policy = get_policy("news_catalyst")
        assert policy.revalidate_minutes == 90

    def test_force_close_minutes(self):
        policy = get_policy("news_catalyst")
        assert policy.force_close_minutes == 120

    def test_extension_eligible(self):
        policy = get_policy("news_catalyst")
        assert policy.extension_eligible is True

    def test_max_extension_minutes(self):
        policy = get_policy("news_catalyst")
        assert policy.max_extension_minutes == 180

    def test_revalidation_interval_minutes(self):
        policy = get_policy("news_catalyst")
        assert policy.revalidation_interval_minutes == 30

    def test_eod_hard_wall(self):
        policy = get_policy("news_catalyst")
        assert policy.eod_hard_wall == time(15, 45)

    def test_setup_type_field(self):
        policy = get_policy("news_catalyst")
        assert policy.setup_type == "news_catalyst"


class TestTrendPullbackPolicy:
    """Test trend_pullback policy matches exact values from Requirement 1.11."""

    def test_alert_minutes(self):
        policy = get_policy("trend_pullback")
        assert policy.alert_minutes == 90

    def test_revalidate_minutes(self):
        policy = get_policy("trend_pullback")
        assert policy.revalidate_minutes == 120

    def test_force_close_minutes(self):
        policy = get_policy("trend_pullback")
        assert policy.force_close_minutes == 150

    def test_extension_eligible(self):
        policy = get_policy("trend_pullback")
        assert policy.extension_eligible is True

    def test_max_extension_minutes(self):
        policy = get_policy("trend_pullback")
        assert policy.max_extension_minutes == 180

    def test_revalidation_interval_minutes(self):
        policy = get_policy("trend_pullback")
        assert policy.revalidation_interval_minutes == 30

    def test_eod_hard_wall(self):
        policy = get_policy("trend_pullback")
        assert policy.eod_hard_wall == time(15, 45)

    def test_setup_type_field(self):
        policy = get_policy("trend_pullback")
        assert policy.setup_type == "trend_pullback"


class TestFastTacticalSetups:
    """Test fast tactical setups meet constraints from Requirement 1.7."""

    def test_momentum_fade_alert_at_most_45(self):
        policy = get_policy("momentum_fade")
        assert policy.alert_minutes <= 45

    def test_momentum_fade_force_close_at_most_75(self):
        policy = get_policy("momentum_fade")
        assert policy.force_close_minutes <= 75

    def test_momentum_fade_not_extension_eligible(self):
        policy = get_policy("momentum_fade")
        assert policy.extension_eligible is False

    def test_orb_alert_at_most_45(self):
        policy = get_policy("orb")
        assert policy.alert_minutes <= 45

    def test_orb_force_close_at_most_75(self):
        policy = get_policy("orb")
        assert policy.force_close_minutes <= 75

    def test_orb_not_extension_eligible(self):
        policy = get_policy("orb")
        assert policy.extension_eligible is False

    def test_short_squeeze_alert_at_most_30(self):
        policy = get_policy("short_squeeze")
        assert policy.alert_minutes <= 30

    def test_short_squeeze_force_close_at_most_60(self):
        policy = get_policy("short_squeeze")
        assert policy.force_close_minutes <= 60

    def test_short_squeeze_not_extension_eligible(self):
        policy = get_policy("short_squeeze")
        assert policy.extension_eligible is False

    def test_momentum_fade_no_revalidation(self):
        policy = get_policy("momentum_fade")
        assert policy.revalidate_minutes is None

    def test_orb_no_revalidation(self):
        policy = get_policy("orb")
        assert policy.revalidate_minutes is None

    def test_short_squeeze_no_revalidation(self):
        policy = get_policy("short_squeeze")
        assert policy.revalidate_minutes is None

    def test_momentum_fade_no_max_extension(self):
        policy = get_policy("momentum_fade")
        assert policy.max_extension_minutes is None

    def test_orb_no_max_extension(self):
        policy = get_policy("orb")
        assert policy.max_extension_minutes is None

    def test_short_squeeze_no_max_extension(self):
        policy = get_policy("short_squeeze")
        assert policy.max_extension_minutes is None


class TestDefaultFallbackPolicy:
    """Test default fallback matches Requirement 1.8."""

    def test_alert_minutes(self):
        assert DEFAULT_POLICY.alert_minutes == 60

    def test_revalidate_minutes_is_none(self):
        assert DEFAULT_POLICY.revalidate_minutes is None

    def test_force_close_minutes(self):
        assert DEFAULT_POLICY.force_close_minutes == 90

    def test_extension_eligible_false(self):
        assert DEFAULT_POLICY.extension_eligible is False

    def test_max_extension_minutes_is_none(self):
        assert DEFAULT_POLICY.max_extension_minutes is None

    def test_revalidation_interval_minutes_is_none(self):
        assert DEFAULT_POLICY.revalidation_interval_minutes is None

    def test_eod_hard_wall(self):
        assert DEFAULT_POLICY.eod_hard_wall == time(15, 45)

    def test_unknown_setup_type_returns_default(self):
        policy = get_policy("nonexistent_setup")
        assert policy is DEFAULT_POLICY

    def test_empty_string_returns_default(self):
        policy = get_policy("")
        assert policy is DEFAULT_POLICY


class TestIsThesisDevelopmentSetup:
    """Test is_thesis_development_setup() helper."""

    def test_news_breakout_is_thesis_development(self):
        assert is_thesis_development_setup("news_breakout") is True

    def test_news_catalyst_is_thesis_development(self):
        assert is_thesis_development_setup("news_catalyst") is True

    def test_trend_pullback_is_thesis_development(self):
        assert is_thesis_development_setup("trend_pullback") is True

    def test_momentum_fade_is_not_thesis_development(self):
        assert is_thesis_development_setup("momentum_fade") is False

    def test_orb_is_not_thesis_development(self):
        assert is_thesis_development_setup("orb") is False

    def test_short_squeeze_is_not_thesis_development(self):
        assert is_thesis_development_setup("short_squeeze") is False

    def test_gap_and_go_is_not_thesis_development(self):
        assert is_thesis_development_setup("gap_and_go") is False

    def test_vwap_reclaim_is_not_thesis_development(self):
        assert is_thesis_development_setup("vwap_reclaim") is False

    def test_unknown_is_not_thesis_development(self):
        assert is_thesis_development_setup("unknown_type") is False


class TestIsExtensionEligible:
    """Test is_extension_eligible() helper."""

    def test_news_breakout_is_extension_eligible(self):
        assert is_extension_eligible("news_breakout") is True

    def test_news_catalyst_is_extension_eligible(self):
        assert is_extension_eligible("news_catalyst") is True

    def test_trend_pullback_is_extension_eligible(self):
        assert is_extension_eligible("trend_pullback") is True

    def test_momentum_fade_not_extension_eligible(self):
        assert is_extension_eligible("momentum_fade") is False

    def test_orb_not_extension_eligible(self):
        assert is_extension_eligible("orb") is False

    def test_short_squeeze_not_extension_eligible(self):
        assert is_extension_eligible("short_squeeze") is False

    def test_gap_and_go_not_extension_eligible(self):
        assert is_extension_eligible("gap_and_go") is False

    def test_vwap_reclaim_not_extension_eligible(self):
        assert is_extension_eligible("vwap_reclaim") is False

    def test_unknown_type_not_extension_eligible(self):
        assert is_extension_eligible("unknown_type") is False
