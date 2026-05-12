"""
Tests for utils.catalyst_specificity — should_apply_gate().

Covers task 2.1 requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7.
"""

import pytest

from utils.catalyst_specificity import should_apply_gate


# ===================================================================
# Requirement 1.1: News setup_types trigger the gate
# ===================================================================


class TestNewsSetupTypes:
    """Gate applies for news-driven setup_types in decision or signal."""

    @pytest.mark.parametrize(
        "setup_type",
        [
            "news_breakout",
            "news_catalyst",
            "news_catalyst_breakout",
            "catalyst_breakout",
        ],
    )
    def test_news_setup_type_in_decision_triggers_gate(self, setup_type):
        decision = {"setup_type": setup_type}
        assert should_apply_gate(decision) is True

    @pytest.mark.parametrize(
        "setup_type",
        [
            "news_breakout",
            "news_catalyst",
            "news_catalyst_breakout",
            "catalyst_breakout",
        ],
    )
    def test_news_setup_type_in_signal_triggers_gate(self, setup_type):
        """Requirement 1.7: signal fields are also inspected."""
        decision = {}
        signal = {"setup_type": setup_type}
        assert should_apply_gate(decision, signal) is True

    def test_news_setup_type_case_insensitive(self):
        decision = {"setup_type": "News_Breakout"}
        assert should_apply_gate(decision) is True

    def test_news_setup_type_as_substring(self):
        """setup_type containing a news type still triggers."""
        decision = {"setup_type": "aggressive_news_catalyst_breakout"}
        assert should_apply_gate(decision) is True


# ===================================================================
# Requirement 1.2: gap_and_go WITH catalyst fields triggers gate
# ===================================================================


class TestGapAndGoWithCatalyst:
    """Gate applies for gap_and_go when explicit catalyst fields exist."""

    def test_gap_and_go_with_catalyst_type_in_decision(self):
        decision = {"setup_type": "gap_and_go", "catalyst_type": "earnings_beat"}
        assert should_apply_gate(decision) is True

    def test_gap_and_go_with_catalyst_in_decision(self):
        decision = {
            "setup_type": "gap_and_go",
            "catalyst": "AMD beats earnings expectations",
        }
        assert should_apply_gate(decision) is True

    def test_gap_and_go_with_catalyst_type_in_signal(self):
        decision = {"setup_type": "gap_and_go"}
        signal = {"catalyst_type": "upgrade"}
        assert should_apply_gate(decision, signal) is True

    def test_gap_and_go_with_catalyst_in_signal(self):
        decision = {"setup_type": "gap_and_go"}
        signal = {"catalyst": "FDA approval announced"}
        assert should_apply_gate(decision, signal) is True

    def test_gap_and_go_setup_type_in_signal_with_catalyst_in_decision(self):
        decision = {"catalyst_type": "earnings"}
        signal = {"setup_type": "gap_and_go"}
        assert should_apply_gate(decision, signal) is True


# ===================================================================
# Requirement 1.3: gap_and_go WITHOUT catalyst fields skips gate
# ===================================================================


class TestGapAndGoWithoutCatalyst:
    """Gate does NOT apply for gap_and_go without explicit catalyst fields."""

    def test_gap_and_go_no_catalyst_fields(self):
        decision = {"setup_type": "gap_and_go"}
        assert should_apply_gate(decision) is False

    def test_gap_and_go_with_momentum_rationale_does_not_trigger(self):
        """Generic momentum language in rationale does NOT trigger for gap_and_go."""
        decision = {
            "setup_type": "gap_and_go",
            "rationale": "Strong momentum on the opening move, gap up with volume",
        }
        assert should_apply_gate(decision) is False

    def test_gap_and_go_with_empty_catalyst_type(self):
        decision = {"setup_type": "gap_and_go", "catalyst_type": ""}
        assert should_apply_gate(decision) is False

    def test_gap_and_go_with_none_catalyst(self):
        decision = {"setup_type": "gap_and_go", "catalyst": None}
        assert should_apply_gate(decision) is False


# ===================================================================
# Requirement 1.4: Non-empty catalyst_type triggers gate
# ===================================================================


class TestCatalystTypeField:
    """Gate applies when catalyst_type is present regardless of setup_type."""

    def test_catalyst_type_in_decision_triggers(self):
        decision = {"setup_type": "technical_breakout", "catalyst_type": "earnings"}
        assert should_apply_gate(decision) is True

    def test_catalyst_type_in_signal_triggers(self):
        decision = {"setup_type": "technical_breakout"}
        signal = {"catalyst_type": "upgrade"}
        assert should_apply_gate(decision, signal) is True

    def test_empty_catalyst_type_does_not_trigger(self):
        decision = {"setup_type": "technical_breakout", "catalyst_type": ""}
        assert should_apply_gate(decision) is False

    def test_none_catalyst_type_does_not_trigger(self):
        decision = {"setup_type": "technical_breakout", "catalyst_type": None}
        assert should_apply_gate(decision) is False


# ===================================================================
# Requirement 1.5: News terms in rationale/thesis trigger gate
# ===================================================================


class TestNewsTermsInText:
    """Gate applies when rationale or thesis contains news-specific terms."""

    @pytest.mark.parametrize(
        "term",
        [
            "catalyst",
            "headline",
            "upgrade",
            "downgrade",
            "earnings",
            "guidance",
            "contract",
            "customer",
            "supplier",
            "fda",
            "regulatory",
            "macro shock",
        ],
    )
    def test_news_term_in_rationale_triggers(self, term):
        decision = {
            "setup_type": "technical_breakout",
            "rationale": f"Trading based on {term} event",
        }
        assert should_apply_gate(decision) is True

    def test_news_term_in_thesis_triggers(self):
        decision = {
            "setup_type": "technical_breakout",
            "thesis": "Earnings beat expected to drive price higher",
        }
        assert should_apply_gate(decision) is True

    def test_news_term_case_insensitive(self):
        decision = {
            "setup_type": "technical_breakout",
            "rationale": "FDA approval pending",
        }
        assert should_apply_gate(decision) is True

    def test_generic_momentum_words_do_not_trigger(self):
        """Generic momentum/breakout words are NOT in NEWS_TERMS."""
        decision = {
            "setup_type": "technical_breakout",
            "rationale": "Strong momentum breakout with opening move above resistance",
        }
        assert should_apply_gate(decision) is False


# ===================================================================
# Requirement 1.6: Purely technical setups return False
# ===================================================================


class TestPurelyTechnicalSetups:
    """Gate does NOT apply for purely technical setups."""

    def test_technical_breakout_no_catalyst(self):
        decision = {
            "setup_type": "technical_breakout",
            "rationale": "Price breaking above resistance with volume",
        }
        assert should_apply_gate(decision) is False

    def test_pullback_entry_no_catalyst(self):
        decision = {
            "setup_type": "pullback_entry",
            "rationale": "Retest of breakout level holding as support",
        }
        assert should_apply_gate(decision) is False

    def test_empty_decision(self):
        decision = {}
        assert should_apply_gate(decision) is False

    def test_no_setup_type_no_catalyst_no_news_terms(self):
        decision = {
            "rationale": "Technical pattern forming on the daily chart",
            "thesis": "Bullish flag breakout setup",
        }
        assert should_apply_gate(decision) is False


# ===================================================================
# Requirement 1.7: Function accepts both decision and optional signal
# ===================================================================


class TestSignalParameter:
    """Verify signal parameter handling."""

    def test_signal_none_works(self):
        decision = {"setup_type": "news_breakout"}
        assert should_apply_gate(decision, None) is True

    def test_signal_not_provided_works(self):
        decision = {"setup_type": "news_breakout"}
        assert should_apply_gate(decision) is True

    def test_decision_setup_type_takes_priority_over_signal(self):
        """Decision setup_type is checked first."""
        decision = {"setup_type": "news_catalyst"}
        signal = {"setup_type": "technical_breakout"}
        assert should_apply_gate(decision, signal) is True

    def test_signal_provides_setup_type_when_decision_lacks_it(self):
        decision = {"rationale": "some text"}
        signal = {"setup_type": "catalyst_breakout"}
        assert should_apply_gate(decision, signal) is True

    def test_signal_provides_catalyst_type_when_decision_lacks_it(self):
        decision = {"setup_type": "pullback_entry"}
        signal = {"catalyst_type": "earnings"}
        assert should_apply_gate(decision, signal) is True

    def test_empty_signal_does_not_affect_result(self):
        decision = {"setup_type": "technical_breakout"}
        signal = {}
        assert should_apply_gate(decision, signal) is False
