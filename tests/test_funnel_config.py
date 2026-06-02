"""Unit tests for the funnel configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.funnel_config import load_funnel_config, _DEFAULTS, _deep_merge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_config_path():
    """Return the path to the real funnel config file."""
    return Path(__file__).resolve().parent.parent / "config" / "funnel_config.yaml"


@pytest.fixture
def tmp_config(tmp_path):
    """Helper to write a temporary YAML config file."""

    def _write(content: str) -> Path:
        p = tmp_path / "funnel_config.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    return _write


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_valid_config(valid_config_path):
    """Loading the real config file succeeds and returns expected structure."""
    config = load_funnel_config(valid_config_path)
    assert isinstance(config, dict)
    assert "funnel" in config
    funnel = config["funnel"]
    assert funnel["enabled"] is True
    assert "schedule" in funnel
    assert "ceilings" in funnel
    assert "budgets" in funnel
    assert "midday_scan" in funnel


def test_schedule_values(valid_config_path):
    """Schedule contains all expected time fields."""
    config = load_funnel_config(valid_config_path)
    schedule = config["funnel"]["schedule"]
    assert schedule["discovery_time"] == "06:00"
    assert schedule["research_time"] == "06:30"
    assert schedule["analysis_time"] == "07:15"
    assert schedule["confirmation_time"] == "09:35"
    assert schedule["confirmation_end"] == "09:45"


def test_ceilings_values(valid_config_path):
    """Ceilings have correct default values."""
    config = load_funnel_config(valid_config_path)
    ceilings = config["funnel"]["ceilings"]
    assert ceilings["max_discovery_shortlist"] == 5
    assert ceilings["max_researcher_promoted"] == 3
    assert ceilings["max_pm_handoff"] == 3


def test_budgets_values(valid_config_path):
    """Budgets have correct default values."""
    config = load_funnel_config(valid_config_path)
    budgets = config["funnel"]["budgets"]
    assert budgets["per_sector_seconds"] == 15
    assert budgets["total_pipeline_seconds"] == 90
    assert budgets["confirmation_budget_seconds"] == 45
    assert budgets["market_hours_confirmation_budget_seconds"] == 60


def test_midday_scan_disabled(valid_config_path):
    """Midday scan is disabled in v1."""
    config = load_funnel_config(valid_config_path)
    midday = config["funnel"]["midday_scan"]
    assert midday["enabled"] is False
    assert midday["revalidation_only"] is True


# ---------------------------------------------------------------------------
# Missing file — returns defaults gracefully
# ---------------------------------------------------------------------------


def test_missing_file_returns_defaults():
    """When config file is missing, returns defaults without raising."""
    fake_path = Path("/nonexistent/path/funnel_config.yaml")
    config = load_funnel_config(fake_path)
    assert config == _DEFAULTS


def test_missing_file_has_full_structure():
    """Defaults contain all required sub-keys."""
    fake_path = Path("/nonexistent/does_not_exist.yaml")
    config = load_funnel_config(fake_path)
    funnel = config["funnel"]
    assert funnel["enabled"] is True
    assert funnel["schedule"]["discovery_time"] == "06:00"
    assert funnel["ceilings"]["max_discovery_shortlist"] == 5
    assert funnel["budgets"]["per_sector_seconds"] == 15
    assert funnel["midday_scan"]["enabled"] is False


# ---------------------------------------------------------------------------
# Malformed / partial YAML — returns defaults gracefully
# ---------------------------------------------------------------------------


def test_invalid_yaml_returns_defaults(tmp_config):
    """Malformed YAML returns defaults without raising."""
    path = tmp_config("funnel: [unterminated")
    config = load_funnel_config(path)
    assert config == _DEFAULTS


def test_non_mapping_returns_defaults(tmp_config):
    """YAML root that is not a mapping returns defaults."""
    path = tmp_config("- item1\n- item2\n")
    config = load_funnel_config(path)
    assert config == _DEFAULTS


def test_empty_file_returns_defaults(tmp_config):
    """Empty config file returns defaults."""
    path = tmp_config("")
    config = load_funnel_config(path)
    assert config == _DEFAULTS


# ---------------------------------------------------------------------------
# Partial config — merges with defaults
# ---------------------------------------------------------------------------


def test_partial_config_fills_missing_fields(tmp_config):
    """Partially defined config fills missing fields from defaults."""
    content = "funnel:\n  enabled: false\n  ceilings:\n    max_discovery_shortlist: 10\n"
    path = tmp_config(content)
    config = load_funnel_config(path)
    funnel = config["funnel"]
    # Overridden values
    assert funnel["enabled"] is False
    assert funnel["ceilings"]["max_discovery_shortlist"] == 10
    # Default-filled values
    assert funnel["ceilings"]["max_researcher_promoted"] == 3
    assert funnel["schedule"]["discovery_time"] == "06:00"
    assert funnel["budgets"]["total_pipeline_seconds"] == 90
    assert funnel["midday_scan"]["enabled"] is False


def test_override_single_budget(tmp_config):
    """Overriding one budget field preserves other budget defaults."""
    content = "funnel:\n  budgets:\n    per_sector_seconds: 30\n"
    path = tmp_config(content)
    config = load_funnel_config(path)
    budgets = config["funnel"]["budgets"]
    assert budgets["per_sector_seconds"] == 30
    assert budgets["total_pipeline_seconds"] == 90
    assert budgets["confirmation_budget_seconds"] == 45
    assert budgets["market_hours_confirmation_budget_seconds"] == 60


# ---------------------------------------------------------------------------
# Deep merge utility
# ---------------------------------------------------------------------------


def test_deep_merge_nested():
    """Deep merge correctly handles nested dicts."""
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 99}, "e": 5}
    result = _deep_merge(base, override)
    assert result == {"a": {"b": 99, "c": 2}, "d": 3, "e": 5}


def test_deep_merge_does_not_mutate_base():
    """Deep merge returns a new dict without mutating the base."""
    base = {"a": {"b": 1}}
    override = {"a": {"b": 99}}
    _deep_merge(base, override)
    assert base == {"a": {"b": 1}}
