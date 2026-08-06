"""Unit tests for AlertDispatcher._is_deferred() deferral logic — Task 5.7.

Tests the flag-gated deferral behavior:
- When ALERT_MATERIAL_OCCURRENCE_MODE == "enabled": uses material_occurrence_count
- When "disabled": uses occurrence_count (legacy behavior)

Requirements: 4.1–4.5
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from utils.alert_dispatcher import AlertDispatcher
from utils.alert_intent_store import AlertIntent, AlertIntentStore


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def engine():
    """In-memory SQLite engine with alert dispatch schema initialized."""
    eng = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(eng)
    return eng


@pytest.fixture
def store(engine):
    """AlertIntentStore backed by an in-memory engine."""
    return AlertIntentStore(engine)


@pytest.fixture
def dispatcher(engine, store):
    """AlertDispatcher with mocked PM callbacks."""
    begin_pm = MagicMock(return_value=True)
    end_pm = MagicMock()
    return AlertDispatcher(
        engine=engine,
        intent_store=store,
        begin_pm_cycle=begin_pm,
        end_pm_cycle=end_pm,
    )


def _make_intent(**overrides) -> AlertIntent:
    """Factory for AlertIntent with sensible defaults for deferral tests."""
    now = datetime(2025, 1, 15, 10, 30, 0)
    defaults = dict(
        id=1,
        alert_intent_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        symbol="AMD",
        alert_type="entry_alert",
        direction="long",
        trigger_price=Decimal("155.20"),
        source_level="resistance_155",
        urgency="medium",
        reason="Price crossed resistance",
        dedupe_key="AMD:entry_alert:long:resistance_155",
        filter_status="passed",
        first_seen_at=now,
        last_seen_at=now,
        occurrence_count=5,
        material_occurrence_count=1,
        expiration_at=now + timedelta(hours=6),
        dispatch_status="pending",
        dispatch_reason=None,
        dispatched_at=None,
        deferred_until=now + timedelta(minutes=15),
        occurrence_count_at_deferral=1,
        dispatch_attempt_count=0,
        last_dispatch_error=None,
    )
    defaults.update(overrides)
    return AlertIntent(**defaults)


# ─── Tests: enabled mode ────────────────────────────────────────────────────


class TestIsDeferredEnabledMode:
    """Tests for _is_deferred when ALERT_MATERIAL_OCCURRENCE_MODE == 'enabled'."""

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    def test_deferred_future_material_occ_equals_snapshot(self, dispatcher):
        """deferred_until in future + material_occ == snapshot → _is_deferred returns True.

        The intent is still deferred because no material change occurred.
        """
        now = datetime(2025, 1, 15, 10, 30, 0)
        intent = _make_intent(
            deferred_until=now + timedelta(minutes=15),
            material_occurrence_count=1,
            occurrence_count_at_deferral=1,
            occurrence_count=5,  # raw count is higher but irrelevant in enabled mode
        )

        result = dispatcher._is_deferred(intent, now)

        assert result is True

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    def test_deferred_future_material_occ_greater_than_snapshot(self, dispatcher):
        """deferred_until in future + material_occ > snapshot → _is_deferred returns False.

        Material change occurred — bypass deferral, re-evaluate.
        """
        now = datetime(2025, 1, 15, 10, 30, 0)
        intent = _make_intent(
            deferred_until=now + timedelta(minutes=15),
            material_occurrence_count=2,
            occurrence_count_at_deferral=1,
            occurrence_count=5,
        )

        result = dispatcher._is_deferred(intent, now)

        assert result is False

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    def test_deferred_until_expired(self, dispatcher):
        """deferred_until expired (in the past) → _is_deferred returns False.

        Deferral window has passed — re-evaluate regardless of counters.
        """
        now = datetime(2025, 1, 15, 10, 30, 0)
        intent = _make_intent(
            deferred_until=now - timedelta(minutes=1),
            material_occurrence_count=1,
            occurrence_count_at_deferral=1,
        )

        result = dispatcher._is_deferred(intent, now)

        assert result is False

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    def test_no_deferred_until(self, dispatcher):
        """No deferred_until set (None) → _is_deferred returns False.

        Intent was never deferred — evaluate normally.
        """
        now = datetime(2025, 1, 15, 10, 30, 0)
        intent = _make_intent(
            deferred_until=None,
            material_occurrence_count=1,
            occurrence_count_at_deferral=0,
        )

        result = dispatcher._is_deferred(intent, now)

        assert result is False


# ─── Tests: disabled mode ───────────────────────────────────────────────────


class TestIsDeferredDisabledMode:
    """Tests for _is_deferred when ALERT_MATERIAL_OCCURRENCE_MODE == 'disabled'."""

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "disabled")
    def test_disabled_mode_uses_occurrence_count(self, dispatcher):
        """Disabled mode compares occurrence_count vs occurrence_count_at_deferral.

        occurrence_count > snapshot → returns False (re-evaluate).
        This is the legacy behavior.
        """
        now = datetime(2025, 1, 15, 10, 30, 0)
        intent = _make_intent(
            deferred_until=now + timedelta(minutes=15),
            occurrence_count=5,
            occurrence_count_at_deferral=3,
            material_occurrence_count=1,  # irrelevant in disabled mode
        )

        result = dispatcher._is_deferred(intent, now)

        assert result is False

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "disabled")
    def test_disabled_mode_deferred_when_occ_equals_snapshot(self, dispatcher):
        """Disabled mode: occurrence_count == snapshot → still deferred (True)."""
        now = datetime(2025, 1, 15, 10, 30, 0)
        intent = _make_intent(
            deferred_until=now + timedelta(minutes=15),
            occurrence_count=3,
            occurrence_count_at_deferral=3,
            material_occurrence_count=2,  # higher but irrelevant in disabled mode
        )

        result = dispatcher._is_deferred(intent, now)

        assert result is True
