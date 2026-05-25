"""Unit tests for apply_score_penalties() in the sector scout pipeline."""

from __future__ import annotations

import pytest

from utils.sector_scout import apply_score_penalties
from utils.sector_scout_models import CandidateRow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> dict:
    """Minimal config dict with score_penalties section using defaults."""
    return {
        "score_penalties": {
            "stale_news_threshold_minutes": 120,
            "stale_news_deduction": 15.0,
            "missing_news_deduction": 20.0,
            "unknown_spread_deduction": 10.0,
            "weak_sector_confirmation_deduction": 10.0,
            "low_rvol_threshold": 1.2,
            "low_rvol_deduction": 12.0,
            "low_dollar_volume_threshold": 5_000_000,
            "low_dollar_volume_deduction": 8.0,
        }
    }


def _make_row(**overrides) -> CandidateRow:
    """Create a CandidateRow that triggers NO penalties by default."""
    defaults = {
        "symbol": "AAPL",
        "sector": "tech",
        "sector_name": "Technology",
        "current_price": 150.0,
        "scout_score": 80.0,
        # Fresh news (no penalty)
        "news_freshness_minutes": 30.0,
        # Known spread (no penalty)
        "spread_status": "known",
        "spread_pct": 0.5,
        # Sector confirmed (no penalty)
        "sector_confirmed": True,
        # High relative volume (no penalty)
        "relative_volume": 2.5,
        # High dollar volume (no penalty)
        "dollar_volume": 50_000_000.0,
    }
    defaults.update(overrides)
    return CandidateRow(**defaults)


# ---------------------------------------------------------------------------
# Happy path — no penalties applied
# ---------------------------------------------------------------------------


def test_no_penalties_when_all_conditions_good(default_config):
    """A candidate with good data gets no penalties and score unchanged."""
    row = _make_row()
    result = apply_score_penalties(row, default_config)

    assert result.scout_score == 80.0
    assert result.penalties_applied == []
    assert result.reason_codes == []


# ---------------------------------------------------------------------------
# Penalty 1: missing_news (news_freshness_minutes is None)
# ---------------------------------------------------------------------------


def test_missing_news_penalty_when_none(default_config):
    """news_freshness_minutes=None triggers missing_news penalty."""
    row = _make_row(news_freshness_minutes=None)
    result = apply_score_penalties(row, default_config)

    assert any(p["type"] == "missing_news" for p in result.penalties_applied)
    assert "penalty:missing_news:20.0" in result.reason_codes
    assert result.scout_score == 80.0 - 20.0


def test_missing_news_does_not_trigger_stale_news(default_config):
    """When news_freshness_minutes is None, stale_news is NOT applied."""
    row = _make_row(news_freshness_minutes=None)
    result = apply_score_penalties(row, default_config)

    assert not any(p["type"] == "stale_news" for p in result.penalties_applied)


# ---------------------------------------------------------------------------
# Penalty 2: stale_news (news_freshness_minutes > threshold)
# ---------------------------------------------------------------------------


def test_stale_news_penalty_when_above_threshold(default_config):
    """news_freshness_minutes > 120 triggers stale_news penalty."""
    row = _make_row(news_freshness_minutes=150.0)
    result = apply_score_penalties(row, default_config)

    assert any(p["type"] == "stale_news" for p in result.penalties_applied)
    assert "penalty:stale_news:15.0" in result.reason_codes
    assert result.scout_score == 80.0 - 15.0


def test_stale_news_does_not_trigger_missing_news(default_config):
    """When news_freshness_minutes is stale, missing_news is NOT applied."""
    row = _make_row(news_freshness_minutes=150.0)
    result = apply_score_penalties(row, default_config)

    assert not any(p["type"] == "missing_news" for p in result.penalties_applied)


def test_no_news_penalty_when_fresh(default_config):
    """news_freshness_minutes <= threshold triggers no news penalty."""
    row = _make_row(news_freshness_minutes=120.0)
    result = apply_score_penalties(row, default_config)

    assert not any(p["type"] == "stale_news" for p in result.penalties_applied)
    assert not any(p["type"] == "missing_news" for p in result.penalties_applied)


def test_no_news_penalty_at_exact_threshold(default_config):
    """news_freshness_minutes exactly at threshold does NOT trigger stale_news."""
    row = _make_row(news_freshness_minutes=120.0)
    result = apply_score_penalties(row, default_config)

    assert not any(p["type"] == "stale_news" for p in result.penalties_applied)


# ---------------------------------------------------------------------------
# Penalty 3: unknown_spread
# ---------------------------------------------------------------------------


def test_unknown_spread_penalty(default_config):
    """spread_status='unknown' triggers unknown_spread penalty."""
    row = _make_row(spread_status="unknown")
    result = apply_score_penalties(row, default_config)

    assert any(p["type"] == "unknown_spread" for p in result.penalties_applied)
    assert "penalty:unknown_spread:10.0" in result.reason_codes
    assert result.scout_score == 80.0 - 10.0


def test_known_spread_no_penalty(default_config):
    """spread_status='known' does NOT trigger unknown_spread penalty."""
    row = _make_row(spread_status="known")
    result = apply_score_penalties(row, default_config)

    assert not any(p["type"] == "unknown_spread" for p in result.penalties_applied)


# ---------------------------------------------------------------------------
# Penalty 4: weak_sector_confirmation
# ---------------------------------------------------------------------------


def test_weak_sector_false_penalty(default_config):
    """sector_confirmed=False triggers weak_sector_confirmation penalty."""
    row = _make_row(sector_confirmed=False)
    result = apply_score_penalties(row, default_config)

    assert any(p["type"] == "weak_sector_confirmation" for p in result.penalties_applied)
    assert "penalty:weak_sector_confirmation:10.0" in result.reason_codes
    assert result.scout_score == 80.0 - 10.0


def test_weak_sector_none_penalty(default_config):
    """sector_confirmed=None triggers weak_sector_confirmation penalty."""
    row = _make_row(sector_confirmed=None)
    result = apply_score_penalties(row, default_config)

    assert any(p["type"] == "weak_sector_confirmation" for p in result.penalties_applied)


def test_sector_confirmed_no_penalty(default_config):
    """sector_confirmed=True does NOT trigger weak_sector penalty."""
    row = _make_row(sector_confirmed=True)
    result = apply_score_penalties(row, default_config)

    assert not any(p["type"] == "weak_sector_confirmation" for p in result.penalties_applied)


# ---------------------------------------------------------------------------
# Penalty 5: low_rvol
# ---------------------------------------------------------------------------


def test_low_rvol_penalty(default_config):
    """relative_volume < 1.2 triggers low_rvol penalty."""
    row = _make_row(relative_volume=0.8)
    result = apply_score_penalties(row, default_config)

    assert any(p["type"] == "low_rvol" for p in result.penalties_applied)
    assert "penalty:low_rvol:12.0" in result.reason_codes
    assert result.scout_score == 80.0 - 12.0


def test_rvol_at_threshold_no_penalty(default_config):
    """relative_volume exactly at threshold does NOT trigger low_rvol."""
    row = _make_row(relative_volume=1.2)
    result = apply_score_penalties(row, default_config)

    assert not any(p["type"] == "low_rvol" for p in result.penalties_applied)


def test_rvol_none_no_penalty(default_config):
    """relative_volume=None does NOT trigger low_rvol penalty."""
    row = _make_row(relative_volume=None)
    result = apply_score_penalties(row, default_config)

    assert not any(p["type"] == "low_rvol" for p in result.penalties_applied)


# ---------------------------------------------------------------------------
# Penalty 6: low_dollar_volume
# ---------------------------------------------------------------------------


def test_low_dollar_volume_penalty(default_config):
    """dollar_volume < $5M triggers low_dollar_volume penalty."""
    row = _make_row(dollar_volume=3_000_000.0)
    result = apply_score_penalties(row, default_config)

    assert any(p["type"] == "low_dollar_volume" for p in result.penalties_applied)
    assert "penalty:low_dollar_volume:8.0" in result.reason_codes
    assert result.scout_score == 80.0 - 8.0


def test_dollar_volume_at_threshold_no_penalty(default_config):
    """dollar_volume exactly at threshold does NOT trigger penalty."""
    row = _make_row(dollar_volume=5_000_000.0)
    result = apply_score_penalties(row, default_config)

    assert not any(p["type"] == "low_dollar_volume" for p in result.penalties_applied)


def test_dollar_volume_none_no_penalty(default_config):
    """dollar_volume=None does NOT trigger low_dollar_volume penalty."""
    row = _make_row(dollar_volume=None)
    result = apply_score_penalties(row, default_config)

    assert not any(p["type"] == "low_dollar_volume" for p in result.penalties_applied)


# ---------------------------------------------------------------------------
# Multiple penalties stacking
# ---------------------------------------------------------------------------


def test_multiple_penalties_stack(default_config):
    """Multiple conditions trigger multiple penalties that all deduct."""
    row = _make_row(
        scout_score=80.0,
        news_freshness_minutes=None,  # missing_news: -20
        spread_status="unknown",       # unknown_spread: -10
        sector_confirmed=False,        # weak_sector: -10
        relative_volume=0.5,           # low_rvol: -12
        dollar_volume=1_000_000.0,     # low_dollar_volume: -8
    )
    result = apply_score_penalties(row, default_config)

    # Total deduction: 20 + 10 + 10 + 12 + 8 = 60
    assert result.scout_score == 80.0 - 60.0
    assert len(result.penalties_applied) == 5
    assert len(result.reason_codes) == 5


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------


def test_score_clamped_to_zero(default_config):
    """Score cannot go below 0.0 even with massive penalties."""
    row = _make_row(
        scout_score=10.0,
        news_freshness_minutes=None,   # -20
        spread_status="unknown",        # -10
        sector_confirmed=None,          # -10
        relative_volume=0.1,            # -12
        dollar_volume=100.0,            # -8
    )
    result = apply_score_penalties(row, default_config)

    # Total deduction: 60, but score was only 10 → clamped to 0
    assert result.scout_score == 0.0


def test_score_clamped_to_100(default_config):
    """Score cannot exceed 100.0 (edge case: no penalties on high score)."""
    row = _make_row(scout_score=100.0)
    result = apply_score_penalties(row, default_config)

    assert result.scout_score == 100.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_output(default_config):
    """Same input always produces same output."""
    for _ in range(10):
        row = _make_row(
            scout_score=75.0,
            news_freshness_minutes=200.0,
            spread_status="unknown",
        )
        result = apply_score_penalties(row, default_config)
        # stale_news: -15, unknown_spread: -10 → 75 - 25 = 50
        assert result.scout_score == 50.0
        assert len(result.penalties_applied) == 2


# ---------------------------------------------------------------------------
# Mutual exclusivity of missing_news and stale_news
# ---------------------------------------------------------------------------


def test_mutual_exclusivity_never_both(default_config):
    """missing_news and stale_news can never both be applied."""
    # Case 1: None → only missing_news
    row1 = _make_row(news_freshness_minutes=None)
    result1 = apply_score_penalties(row1, default_config)
    types1 = [p["type"] for p in result1.penalties_applied]
    assert "missing_news" in types1
    assert "stale_news" not in types1

    # Case 2: stale → only stale_news
    row2 = _make_row(news_freshness_minutes=200.0)
    result2 = apply_score_penalties(row2, default_config)
    types2 = [p["type"] for p in result2.penalties_applied]
    assert "stale_news" in types2
    assert "missing_news" not in types2

    # Case 3: fresh → neither
    row3 = _make_row(news_freshness_minutes=60.0)
    result3 = apply_score_penalties(row3, default_config)
    types3 = [p["type"] for p in result3.penalties_applied]
    assert "missing_news" not in types3
    assert "stale_news" not in types3


# ---------------------------------------------------------------------------
# Config override
# ---------------------------------------------------------------------------


def test_custom_config_values():
    """Custom config values are respected for thresholds and deductions."""
    config = {
        "score_penalties": {
            "stale_news_threshold_minutes": 60,
            "stale_news_deduction": 5.0,
            "missing_news_deduction": 10.0,
            "unknown_spread_deduction": 3.0,
            "weak_sector_confirmation_deduction": 4.0,
            "low_rvol_threshold": 2.0,
            "low_rvol_deduction": 6.0,
            "low_dollar_volume_threshold": 10_000_000,
            "low_dollar_volume_deduction": 2.0,
        }
    }
    # news_freshness=90 > custom threshold of 60 → stale_news with custom deduction
    row = _make_row(scout_score=50.0, news_freshness_minutes=90.0)
    result = apply_score_penalties(row, config)

    assert any(p["type"] == "stale_news" and p["deduction"] == 5.0 for p in result.penalties_applied)
    assert result.scout_score == 50.0 - 5.0
