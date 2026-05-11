"""Unit tests for validate_stop_geometry in position_lifecycle_governance.

Tests the stop geometry validation helper that checks whether a trade's stop
is valid for overnight carry.
"""

import pytest

from utils.position_lifecycle_governance import validate_stop_geometry


# ---------------------------------------------------------------------------
# Test: missing_price_fail_safe (current_price is None)
# ---------------------------------------------------------------------------


class TestMissingPriceFailSafe:
    """When current_price is None, fail safe to invalid."""

    def test_none_price_returns_missing_price_fail_safe(self):
        trade = {"stop_price": 95.0, "direction": "LONG"}
        is_valid, reason = validate_stop_geometry(trade, None, None)
        assert is_valid is False
        assert reason == "missing_price_fail_safe"

    def test_none_price_with_auth_still_fails_safe(self):
        trade = {"stop_price": 95.0, "direction": "LONG"}
        auth = {"stop_price": 95.0}
        is_valid, reason = validate_stop_geometry(trade, auth, None)
        assert is_valid is False
        assert reason == "missing_price_fail_safe"


# ---------------------------------------------------------------------------
# Test: missing_stop (trade has no stop_price or stop_price is 0)
# ---------------------------------------------------------------------------


class TestMissingStop:
    """When trade has no stop_price (None or 0), return missing_stop."""

    def test_none_stop_price(self):
        trade = {"stop_price": None, "direction": "LONG"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is False
        assert reason == "missing_stop"

    def test_zero_stop_price(self):
        trade = {"stop_price": 0, "direction": "LONG"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is False
        assert reason == "missing_stop"

    def test_missing_stop_price_key(self):
        trade = {"direction": "LONG"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is False
        assert reason == "missing_stop"


# ---------------------------------------------------------------------------
# Test: authorization_stop_mismatch
# ---------------------------------------------------------------------------


class TestAuthorizationStopMismatch:
    """When trade stop != auth stop, return authorization_stop_mismatch."""

    def test_mismatch_detected(self):
        trade = {"stop_price": 95.0, "direction": "LONG"}
        auth = {"stop_price": 90.0}
        is_valid, reason = validate_stop_geometry(trade, auth, 100.0)
        assert is_valid is False
        assert reason == "authorization_stop_mismatch"

    def test_matching_stops_pass(self):
        trade = {"stop_price": 95.0, "direction": "LONG"}
        auth = {"stop_price": 95.0}
        is_valid, reason = validate_stop_geometry(trade, auth, 100.0)
        assert is_valid is True
        assert reason == ""

    def test_no_auth_skips_mismatch_check(self):
        """When overnight_auth is None, skip the mismatch check entirely."""
        trade = {"stop_price": 95.0, "direction": "LONG"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is True
        assert reason == ""

    def test_auth_without_stop_price_key_skips_check(self):
        """When auth dict exists but has no stop_price key, skip mismatch check."""
        trade = {"stop_price": 95.0, "direction": "LONG"}
        auth = {"overnight_thesis": "some thesis"}
        is_valid, reason = validate_stop_geometry(trade, auth, 100.0)
        assert is_valid is True
        assert reason == ""


# ---------------------------------------------------------------------------
# Test: inverted_long (LONG stop >= current_price)
# ---------------------------------------------------------------------------


class TestInvertedLong:
    """LONG position with stop >= current_price is inverted."""

    def test_stop_above_price(self):
        trade = {"stop_price": 105.0, "direction": "LONG"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is False
        assert reason == "inverted_long"

    def test_stop_equal_to_price(self):
        trade = {"stop_price": 100.0, "direction": "LONG"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is False
        assert reason == "inverted_long"

    def test_stop_below_price_valid(self):
        trade = {"stop_price": 95.0, "direction": "LONG"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is True
        assert reason == ""


# ---------------------------------------------------------------------------
# Test: inverted_short (SHORT stop <= current_price)
# ---------------------------------------------------------------------------


class TestInvertedShort:
    """SHORT position with stop <= current_price is inverted."""

    def test_stop_below_price(self):
        trade = {"stop_price": 95.0, "direction": "SHORT"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is False
        assert reason == "inverted_short"

    def test_stop_equal_to_price(self):
        trade = {"stop_price": 100.0, "direction": "SHORT"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is False
        assert reason == "inverted_short"

    def test_stop_above_price_valid(self):
        trade = {"stop_price": 105.0, "direction": "SHORT"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is True
        assert reason == ""


# ---------------------------------------------------------------------------
# Test: Direction case normalization
# ---------------------------------------------------------------------------


class TestDirectionCaseNormalization:
    """Direction should be normalized to upper case."""

    def test_lowercase_long(self):
        trade = {"stop_price": 105.0, "direction": "long"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is False
        assert reason == "inverted_long"

    def test_mixed_case_short(self):
        trade = {"stop_price": 95.0, "direction": "Short"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is False
        assert reason == "inverted_short"

    def test_uppercase_long_valid(self):
        trade = {"stop_price": 95.0, "direction": "LONG"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is True
        assert reason == ""


# ---------------------------------------------------------------------------
# Test: Priority order (check order matters)
# ---------------------------------------------------------------------------


class TestCheckOrder:
    """Verify that checks are evaluated in the correct priority order."""

    def test_missing_price_takes_priority_over_missing_stop(self):
        """current_price=None should be checked before stop_price."""
        trade = {"stop_price": None, "direction": "LONG"}
        is_valid, reason = validate_stop_geometry(trade, None, None)
        assert reason == "missing_price_fail_safe"

    def test_missing_stop_takes_priority_over_mismatch(self):
        """missing_stop should be checked before authorization_stop_mismatch."""
        trade = {"stop_price": None, "direction": "LONG"}
        auth = {"stop_price": 90.0}
        is_valid, reason = validate_stop_geometry(trade, auth, 100.0)
        assert reason == "missing_stop"

    def test_mismatch_takes_priority_over_inversion(self):
        """authorization_stop_mismatch should be checked before inversion."""
        # Trade has stop=105 (inverted for LONG), auth has stop=90 (mismatch)
        trade = {"stop_price": 105.0, "direction": "LONG"}
        auth = {"stop_price": 90.0}
        is_valid, reason = validate_stop_geometry(trade, auth, 100.0)
        assert reason == "authorization_stop_mismatch"


# ---------------------------------------------------------------------------
# Test: Valid scenarios (all checks pass)
# ---------------------------------------------------------------------------


class TestValidScenarios:
    """All checks pass → (True, "")."""

    def test_valid_long_with_auth(self):
        trade = {"stop_price": 95.0, "direction": "LONG"}
        auth = {"stop_price": 95.0}
        is_valid, reason = validate_stop_geometry(trade, auth, 100.0)
        assert is_valid is True
        assert reason == ""

    def test_valid_short_with_auth(self):
        trade = {"stop_price": 105.0, "direction": "SHORT"}
        auth = {"stop_price": 105.0}
        is_valid, reason = validate_stop_geometry(trade, auth, 100.0)
        assert is_valid is True
        assert reason == ""

    def test_valid_long_no_auth(self):
        trade = {"stop_price": 95.0, "direction": "LONG"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is True
        assert reason == ""

    def test_valid_short_no_auth(self):
        trade = {"stop_price": 105.0, "direction": "SHORT"}
        is_valid, reason = validate_stop_geometry(trade, None, 100.0)
        assert is_valid is True
        assert reason == ""
