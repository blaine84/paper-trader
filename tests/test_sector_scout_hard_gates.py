"""Unit tests for apply_hard_gates() in the sector scout pipeline."""

from __future__ import annotations

import pytest

from utils.sector_scout import apply_hard_gates
from utils.sector_scout_models import CandidateRow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> dict:
    """Minimal config dict with hard_gates section using defaults."""
    return {
        "hard_gates": {
            "min_price": 5.0,
            "max_spread_pct": 5.0,
            "min_market_cap": 500_000_000,
            "proxy_market_cap_enabled": True,
        }
    }


def _make_row(**overrides) -> CandidateRow:
    """Create a valid CandidateRow with sensible defaults, applying overrides."""
    defaults = {
        "symbol": "AAPL",
        "sector": "tech",
        "sector_name": "Technology",
        "current_price": 150.0,
        "average_volume": 50_000_000,
        "market_cap": 2_000_000_000_000.0,
        "spread_status": "known",
        "spread_pct": 0.5,
    }
    defaults.update(overrides)
    return CandidateRow(**defaults)


# ---------------------------------------------------------------------------
# Happy path — candidate passes all gates
# ---------------------------------------------------------------------------


def test_valid_candidate_passes(default_config):
    """A well-formed candidate with good data passes all hard gates."""
    row = _make_row()
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is True
    assert reason is None
    assert row.hard_gate_passed is True
    assert row.reason_codes == []


# ---------------------------------------------------------------------------
# Gate 1: Malformed/missing critical fields
# ---------------------------------------------------------------------------


def test_missing_symbol_rejects(default_config):
    """Empty symbol triggers malformed_row rejection."""
    row = _make_row(symbol="")
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is False
    assert reason == "hard_gate:malformed_row"
    assert "hard_gate:malformed_row" in row.reason_codes


def test_missing_sector_rejects(default_config):
    """Empty sector triggers malformed_row rejection."""
    row = _make_row(sector="")
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is False
    assert reason == "hard_gate:malformed_row"
    assert "hard_gate:malformed_row" in row.reason_codes


# ---------------------------------------------------------------------------
# Gate 2: Missing or zero price
# ---------------------------------------------------------------------------


def test_none_price_rejects(default_config):
    """None price triggers missing_or_zero_price rejection."""
    row = _make_row(current_price=None)
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is False
    assert "hard_gate:missing_or_zero_price" in row.reason_codes


def test_zero_price_rejects(default_config):
    """Zero price triggers missing_or_zero_price rejection."""
    row = _make_row(current_price=0)
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is False
    assert "hard_gate:missing_or_zero_price" in row.reason_codes


# ---------------------------------------------------------------------------
# Gate 3: Price below minimum
# ---------------------------------------------------------------------------


def test_price_below_minimum_rejects(default_config):
    """Price below min_price threshold triggers price_below_minimum."""
    row = _make_row(current_price=3.50)
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is False
    assert "hard_gate:price_below_minimum" in row.reason_codes


def test_price_at_minimum_passes(default_config):
    """Price exactly at min_price does NOT trigger rejection."""
    row = _make_row(current_price=5.0)
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is True
    assert "hard_gate:price_below_minimum" not in row.reason_codes


# ---------------------------------------------------------------------------
# Gate 4: Spread too wide
# ---------------------------------------------------------------------------


def test_spread_too_wide_rejects(default_config):
    """Spread above max_spread_pct with known status triggers rejection."""
    row = _make_row(spread_pct=6.0, spread_status="known")
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is False
    assert "hard_gate:spread_too_wide" in row.reason_codes


def test_spread_unknown_does_not_reject(default_config):
    """Wide spread with unknown status does NOT trigger hard gate."""
    row = _make_row(spread_pct=10.0, spread_status="unknown")
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is True
    assert "hard_gate:spread_too_wide" not in row.reason_codes


def test_spread_at_threshold_passes(default_config):
    """Spread exactly at max_spread_pct does NOT trigger rejection."""
    row = _make_row(spread_pct=5.0, spread_status="known")
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is True


# ---------------------------------------------------------------------------
# Gate 5: Market cap below minimum
# ---------------------------------------------------------------------------


def test_market_cap_below_minimum_rejects(default_config):
    """Market cap below min_market_cap triggers rejection."""
    row = _make_row(market_cap=100_000_000)
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is False
    assert "hard_gate:below_min_market_cap" in row.reason_codes


def test_market_cap_at_minimum_passes(default_config):
    """Market cap exactly at min_market_cap does NOT trigger rejection."""
    row = _make_row(market_cap=500_000_000)
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is True


# ---------------------------------------------------------------------------
# Gate 5b: Market cap proxy
# ---------------------------------------------------------------------------


def test_proxy_market_cap_rejects_when_below(default_config):
    """When market_cap is None, proxy (price * avg_volume) below threshold rejects."""
    # proxy = 10.0 * 1_000_000 = 10M < 500M
    row = _make_row(market_cap=None, current_price=10.0, average_volume=1_000_000)
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is False
    assert "hard_gate:below_min_market_cap" in row.reason_codes
    assert row.microcap_proxy_used is True
    assert row.market_cap_source == "proxy"


def test_proxy_market_cap_passes_when_above(default_config):
    """When market_cap is None, proxy above threshold passes."""
    # proxy = 150.0 * 50_000_000 = 7.5B > 500M
    row = _make_row(market_cap=None, current_price=150.0, average_volume=50_000_000)
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is True
    assert row.microcap_proxy_used is True
    assert row.market_cap_source == "proxy"


def test_proxy_inconclusive_does_not_reject(default_config):
    """When proxy data is missing, do NOT reject (flag for penalty later)."""
    row = _make_row(market_cap=None, current_price=150.0, average_volume=None)
    passed, reason = apply_hard_gates(row, default_config)

    # Should pass hard gates (inconclusive proxy → penalty, not rejection)
    assert passed is True
    assert "hard_gate:below_min_market_cap" not in row.reason_codes


def test_proxy_disabled_does_not_reject(default_config):
    """When proxy_market_cap_enabled is False, missing market_cap is not checked."""
    default_config["hard_gates"]["proxy_market_cap_enabled"] = False
    row = _make_row(market_cap=None, current_price=2.0, average_volume=100)
    passed, reason = apply_hard_gates(row, default_config)

    # Price below minimum will still reject, but market cap proxy won't
    assert "hard_gate:below_min_market_cap" not in row.reason_codes


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_output(default_config):
    """Same input always produces same output."""
    for _ in range(10):
        row = _make_row(current_price=3.0)
        passed, reason = apply_hard_gates(row, default_config)
        assert passed is False
        assert reason == "hard_gate:price_below_minimum"


# ---------------------------------------------------------------------------
# Multiple rejections
# ---------------------------------------------------------------------------


def test_multiple_reasons_collected(default_config):
    """Multiple gate violations are all recorded in reason_codes."""
    row = _make_row(
        symbol="",
        current_price=0,
        spread_pct=10.0,
        spread_status="known",
    )
    passed, reason = apply_hard_gates(row, default_config)

    assert passed is False
    assert "hard_gate:malformed_row" in row.reason_codes
    assert "hard_gate:missing_or_zero_price" in row.reason_codes
    # First reason returned is the first gate checked
    assert reason == "hard_gate:malformed_row"


def test_return_first_reason_code(default_config):
    """The returned reason_code is the first rejection encountered."""
    row = _make_row(current_price=None)
    passed, reason = apply_hard_gates(row, default_config)

    assert reason == "hard_gate:missing_or_zero_price"
