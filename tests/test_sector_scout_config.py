"""Unit tests for the sector scout config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.sector_scout import load_sector_scout_config, REQUIRED_CONFIG_SECTIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_config_path():
    """Return the path to the real config file."""
    return Path(__file__).resolve().parent.parent / "config" / "sector_scout_config.yaml"


@pytest.fixture
def tmp_config(tmp_path):
    """Helper to write a temporary YAML config file."""

    def _write(content: str) -> Path:
        p = tmp_path / "sector_scout_config.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    return _write


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_valid_config(valid_config_path):
    """Loading the real config file succeeds and returns a dict with all sections."""
    config = load_sector_scout_config(valid_config_path)
    assert isinstance(config, dict)
    for section in REQUIRED_CONFIG_SECTIONS:
        assert section in config


def test_config_returns_expected_types(valid_config_path):
    """Spot-check that key sections have expected types."""
    config = load_sector_scout_config(valid_config_path)
    assert isinstance(config["enabled"], bool)
    assert isinstance(config["sector_buckets"], dict)
    assert isinstance(config["hard_gates"], dict)
    assert isinstance(config["scoring_weights"], dict)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_missing_file_raises_file_not_found():
    """FileNotFoundError raised when config path does not exist."""
    fake_path = Path("/nonexistent/path/config.yaml")
    with pytest.raises(FileNotFoundError, match="not found"):
        load_sector_scout_config(fake_path)


def test_invalid_yaml_raises_value_error(tmp_config):
    """ValueError raised when YAML is malformed."""
    path = tmp_config("enabled: [unterminated")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_sector_scout_config(path)


def test_empty_file_raises_value_error(tmp_config):
    """ValueError raised when config file is empty."""
    path = tmp_config("")
    with pytest.raises(ValueError, match="empty"):
        load_sector_scout_config(path)


def test_non_mapping_raises_value_error(tmp_config):
    """ValueError raised when YAML root is not a mapping."""
    path = tmp_config("- item1\n- item2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_sector_scout_config(path)


def test_missing_sections_raises_value_error(tmp_config):
    """ValueError raised when required sections are absent."""
    # Only provide 'enabled' — all others missing
    path = tmp_config("enabled: true\n")
    with pytest.raises(ValueError, match="missing required sections"):
        load_sector_scout_config(path)


def test_missing_single_section_reports_which(tmp_config):
    """Error message identifies the specific missing section(s)."""
    # Provide all except reanalysis_cooldown
    sections = REQUIRED_CONFIG_SECTIONS - {"reanalysis_cooldown"}
    content = "\n".join(f"{s}: {{}}" for s in sections)
    path = tmp_config(content)
    with pytest.raises(ValueError, match="reanalysis_cooldown"):
        load_sector_scout_config(path)
