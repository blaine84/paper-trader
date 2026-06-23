"""Tests for validate_candidate_scorability in utils/shadow_outcomes.py."""

import pytest
from sqlalchemy import create_engine, text

from utils.shadow_outcomes import validate_candidate_scorability, _is_positive_number


@pytest.fixture
def valid_candidate():
    return {
        "symbol": "AAPL",
        "entry_price": 150.0,
        "stop_price": 145.0,
        "target_price": 160.0,
        "action": "BUY",
        "quantity": 10,
    }


@pytest.fixture
def engine_with_registry():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE pm_candidates (candidate_id TEXT PRIMARY KEY)"
        ))
        conn.execute(text(
            "INSERT INTO pm_candidates (candidate_id) VALUES ('real-id-123')"
        ))
    return engine


@pytest.fixture
def engine_no_table():
    return create_engine("sqlite:///:memory:")


class TestIsPositiveNumber:
    def test_positive_float(self):
        assert _is_positive_number(100.0) is True

    def test_positive_string(self):
        assert _is_positive_number("10.5") is True

    def test_zero(self):
        assert _is_positive_number(0) is False

    def test_negative(self):
        assert _is_positive_number(-5) is False

    def test_none(self):
        assert _is_positive_number(None) is False

    def test_non_numeric_string(self):
        assert _is_positive_number("abc") is False

    def test_infinity(self):
        assert _is_positive_number(float("inf")) is False

    def test_nan(self):
        assert _is_positive_number(float("nan")) is False


class TestValidateCandidateScorability:
    def test_valid_candidate_no_engine(self, valid_candidate):
        result = validate_candidate_scorability(valid_candidate)
        assert result == (True, None)

    def test_missing_symbol(self, valid_candidate):
        valid_candidate["symbol"] = None
        assert validate_candidate_scorability(valid_candidate) == (False, "missing_symbol")

    def test_empty_symbol(self, valid_candidate):
        valid_candidate["symbol"] = ""
        assert validate_candidate_scorability(valid_candidate) == (False, "missing_symbol")

    def test_non_string_symbol(self, valid_candidate):
        valid_candidate["symbol"] = 123
        assert validate_candidate_scorability(valid_candidate) == (False, "missing_symbol")

    @pytest.mark.parametrize("prefix", ["sector_", "industry_", "category_", "theme_", "ETF_"])
    def test_placeholder_symbol(self, valid_candidate, prefix):
        valid_candidate["symbol"] = f"{prefix}tech"
        assert validate_candidate_scorability(valid_candidate) == (False, "symbol_is_placeholder")

    def test_incomplete_geometry_no_entry(self, valid_candidate):
        valid_candidate["entry_price"] = None
        assert validate_candidate_scorability(valid_candidate) == (False, "incomplete_geometry")

    def test_incomplete_geometry_no_stop(self, valid_candidate):
        valid_candidate["stop_price"] = 0
        assert validate_candidate_scorability(valid_candidate) == (False, "incomplete_geometry")

    def test_incomplete_geometry_no_target(self, valid_candidate):
        valid_candidate["target_price"] = -10
        assert validate_candidate_scorability(valid_candidate) == (False, "incomplete_geometry")

    def test_unrecognized_direction(self, valid_candidate):
        valid_candidate["action"] = "SELL"
        assert validate_candidate_scorability(valid_candidate) == (False, "unrecognized_direction")

    def test_no_direction_field(self, valid_candidate):
        del valid_candidate["action"]
        assert validate_candidate_scorability(valid_candidate) == (False, "unrecognized_direction")

    @pytest.mark.parametrize("direction", ["BUY", "SHORT", "LONG", "buy", "short", "long"])
    def test_recognized_directions(self, valid_candidate, direction):
        valid_candidate["action"] = direction
        assert validate_candidate_scorability(valid_candidate) == (True, None)

    def test_direction_field_fallback(self, valid_candidate):
        del valid_candidate["action"]
        valid_candidate["direction"] = "SHORT"
        assert validate_candidate_scorability(valid_candidate) == (True, None)

    def test_missing_quantity(self, valid_candidate):
        valid_candidate["quantity"] = None
        assert validate_candidate_scorability(valid_candidate) == (False, "missing_quantity")

    def test_zero_quantity(self, valid_candidate):
        valid_candidate["quantity"] = 0
        assert validate_candidate_scorability(valid_candidate) == (False, "missing_quantity")

    def test_negative_quantity(self, valid_candidate):
        valid_candidate["quantity"] = -5
        assert validate_candidate_scorability(valid_candidate) == (False, "missing_quantity")

    # --- Registry checks with engine ---

    def test_registry_hit(self, valid_candidate, engine_with_registry):
        valid_candidate["candidate_id"] = "real-id-123"
        result = validate_candidate_scorability(valid_candidate, engine=engine_with_registry)
        assert result == (True, None)

    def test_registry_miss(self, valid_candidate, engine_with_registry):
        valid_candidate["candidate_id"] = "fake-id-999"
        result = validate_candidate_scorability(valid_candidate, engine=engine_with_registry)
        assert result == (False, "candidate_not_in_registry")

    def test_legacy_no_candidate_id_field(self, valid_candidate, engine_with_registry):
        # No candidate_id field → legacy candidate, allow through
        result = validate_candidate_scorability(valid_candidate, engine=engine_with_registry)
        assert result == (True, None)

    def test_geometry_candidate_id_used(self, valid_candidate, engine_with_registry):
        # geometry_candidate_id takes precedence
        valid_candidate["geometry_candidate_id"] = "real-id-123"
        valid_candidate["candidate_id"] = "fake-id"
        result = validate_candidate_scorability(valid_candidate, engine=engine_with_registry)
        assert result == (True, None)

    def test_missing_table_skips_check(self, valid_candidate, engine_no_table):
        valid_candidate["candidate_id"] = "some-id"
        result = validate_candidate_scorability(valid_candidate, engine=engine_no_table)
        assert result == (True, None)

    def test_engine_none_skips_registry(self, valid_candidate):
        valid_candidate["candidate_id"] = "nonexistent-id"
        result = validate_candidate_scorability(valid_candidate, engine=None)
        assert result == (True, None)
