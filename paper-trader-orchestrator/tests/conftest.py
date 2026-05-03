"""Shared test fixtures for paper-trader-orchestrator."""

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def amd_overlap_snapshot():
    """Load the AMD moderate/aggressive overlap snapshot fixture."""
    path = FIXTURES_DIR / "amd_overlap_snapshot.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def no_overlap_snapshot():
    """Load the no-overlap snapshot fixture."""
    path = FIXTURES_DIR / "no_overlap_snapshot.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def malformed_snapshot():
    """Load the malformed snapshot fixture (missing required fields)."""
    path = FIXTURES_DIR / "malformed_snapshot.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def valid_diagnostic():
    """Load a valid LLM diagnostic output fixture."""
    path = FIXTURES_DIR / "valid_diagnostic.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def invalid_diagnostic():
    """Load an invalid LLM diagnostic output fixture."""
    path = FIXTURES_DIR / "invalid_diagnostic.json"
    with open(path) as f:
        return json.load(f)
