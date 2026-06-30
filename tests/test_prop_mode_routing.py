"""
Property-based tests for mode-based dispatch routing correctness.

Property 3: Mode-based dispatch routing correctness.
Validates that the dispatcher routes intents correctly based on their effective
mode: disabled intents are skipped entirely (no audit, no PM), observe intents
get a would_dispatch audit record but no PM invocation, and dispatch intents
are added to the dispatch candidate pipeline.

**Validates: Requirements 1.2, 1.3, 1.4**
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch, call

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.alert_dispatcher import AlertDispatcher
from utils.alert_intent_store import AlertIntent


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

alert_type_strategy = st.sampled_from(["entry_alert", "breakout", "rapid_move", "target_hit"])
effective_mode_strategy = st.sampled_from(["dispatch", "observe", "disabled"])

# Generate symbols
symbol_strategy = st.sampled_from(["AAPL", "NVDA", "MSFT", "TSLA", "GOOGL", "AMZN"])

# Urgency levels
urgency_strategy = st.sampled_from(["high", "medium", "low"])

# Occurrence counts (non-negative integers)
occurrence_count_strategy = st.integers(min_value=1, max_value=100)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dispatcher() -> AlertDispatcher:
    """Create a minimal AlertDispatcher with mock dependencies."""
    engine = MagicMock()
    store = MagicMock()
    # Methods called during evaluate_and_dispatch() before mode routing
    store.recover_stale_active_intents.return_value = 0
    store.query_unclassified.return_value = []
    store.query_active_past_expiration.return_value = []
    # has_would_dispatch_for_occurrence returns False (first observation)
    store.has_would_dispatch_for_occurrence.return_value = False
    begin_pm_cycle = MagicMock(return_value=True)
    end_pm_cycle = MagicMock()
    dispatcher = AlertDispatcher(engine, store, begin_pm_cycle, end_pm_cycle)
    # Mock _recover_stale_intents to avoid needing a real DB connection
    # for recover_stale_claims (which does raw SQL against the engine)
    dispatcher._recover_stale_intents = MagicMock()
    return dispatcher


def _make_fresh_intent(
    alert_type: str,
    symbol: str = "AAPL",
    urgency: str = "high",
    occurrence_count: int = 1,
) -> AlertIntent:
    """Create a fresh AlertIntent that will pass freshness checks.

    Sets last_seen_at to very recent (1 minute ago) so freshness check passes
    for any alert_type with default freshness limits. Sets deferred_until to None
    so deferral check passes.
    """
    now = datetime.utcnow()
    return AlertIntent(
        id=1,
        alert_intent_id="test-intent-001",
        symbol=symbol,
        alert_type=alert_type,
        direction="long",
        trigger_price=Decimal("150.00"),
        source_level="support_bounce",
        urgency=urgency,
        reason="Test intent for routing",
        dedupe_key=f"{symbol}:{alert_type}:abc123",
        filter_status="passed",
        first_seen_at=now - timedelta(minutes=2),
        last_seen_at=now - timedelta(minutes=1),
        occurrence_count=occurrence_count,
        expiration_at=now + timedelta(hours=2),
        dispatch_status="pending",
        dispatch_reason=None,
        dispatched_at=None,
        deferred_until=None,
        occurrence_count_at_deferral=0,
        dispatch_attempt_count=0,
        last_dispatch_error=None,
    )


# ---------------------------------------------------------------------------
# Property 3: Mode-based dispatch routing correctness
# **Validates: Requirements 1.2, 1.3, 1.4**
# ---------------------------------------------------------------------------


class TestProperty3ModeBasedDispatchRoutingCorrectness:
    """
    Property 3: Mode-based dispatch routing correctness.

    For any (alert_type, effective_mode) combination:
    - disabled: no record_audit_log call, no dispatch candidate, only DEBUG log
    - observe: _handle_observe called (which writes audit), NOT added to dispatch candidates
    - dispatch: added to dispatch candidates, NOT routed through _handle_observe

    **Validates: Requirements 1.2, 1.3, 1.4**
    """

    @given(
        alert_type=alert_type_strategy,
        symbol=symbol_strategy,
        urgency=urgency_strategy,
        occurrence_count=occurrence_count_strategy,
    )
    @settings(max_examples=100)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_BREAKOUT_MINUTES", 10)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES", 5)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 30)
    def test_disabled_mode_never_dispatches_or_observes(
        self,
        alert_type: str,
        symbol: str,
        urgency: str,
        occurrence_count: int,
    ):
        """When effective mode is 'disabled', intent is skipped entirely:
        no audit row, no PM invocation, no dispatch candidate."""
        dispatcher = _make_dispatcher()
        intent = _make_fresh_intent(
            alert_type=alert_type,
            symbol=symbol,
            urgency=urgency,
            occurrence_count=occurrence_count,
        )

        # Set up store to return our single intent
        dispatcher._store.query_pending.return_value = [intent]

        # Patch mode_config to return "disabled" for ALL alert types
        mock_mode_config = MagicMock()
        mock_mode_config.global_mode = "dispatch"  # global is not disabled (so we enter the loop)
        mock_mode_config.effective_mode.return_value = "disabled"

        with patch("utils.dispatch_mode.build_dispatch_mode_config", return_value=mock_mode_config):
            result = dispatcher.evaluate_and_dispatch()

        # Assertions for disabled mode:
        # 1. No audit log record written (no record_audit_log call)
        dispatcher._store.record_audit_log.assert_not_called()

        # 2. No observe handling (has_would_dispatch_for_occurrence not called for routing)
        dispatcher._store.has_would_dispatch_for_occurrence.assert_not_called()

        # 3. No PM cycle triggered (begin_pm_cycle not called)
        dispatcher._begin_pm_cycle.assert_not_called()

        # 4. Result should be None (nothing dispatched or observed)
        assert result is None, (
            f"Disabled mode should return None, got {result} for "
            f"alert_type={alert_type}, symbol={symbol}"
        )

    @given(
        alert_type=alert_type_strategy,
        symbol=symbol_strategy,
        urgency=urgency_strategy,
        occurrence_count=occurrence_count_strategy,
    )
    @settings(max_examples=100)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_BREAKOUT_MINUTES", 10)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES", 5)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 30)
    def test_observe_mode_records_audit_without_pm_invocation(
        self,
        alert_type: str,
        symbol: str,
        urgency: str,
        occurrence_count: int,
    ):
        """When effective mode is 'observe', a would_dispatch audit row is written
        but no PM invocation occurs and intent is NOT a dispatch candidate."""
        # For observe mode, the intent must pass freshness. target_hit always fails
        # freshness (_check_freshness returns False for target_hit), so it gets
        # expired rather than observed. We need alert types that CAN pass freshness.
        assume(alert_type != "target_hit")

        dispatcher = _make_dispatcher()
        intent = _make_fresh_intent(
            alert_type=alert_type,
            symbol=symbol,
            urgency=urgency,
            occurrence_count=occurrence_count,
        )

        # Set up store to return our single intent
        dispatcher._store.query_pending.return_value = [intent]

        # Mock mode_config to return "observe" for all alert types
        mock_mode_config = MagicMock()
        mock_mode_config.global_mode = "dispatch"  # global allows processing
        mock_mode_config.effective_mode.return_value = "observe"

        with patch("utils.dispatch_mode.build_dispatch_mode_config", return_value=mock_mode_config):
            result = dispatcher.evaluate_and_dispatch()

        # Assertions for observe mode:
        # 1. Audit log IS written (would_dispatch via _handle_observe)
        dispatcher._store.record_audit_log.assert_called()
        audit_call_kwargs = dispatcher._store.record_audit_log.call_args
        # Verify it's a would_dispatch record
        if audit_call_kwargs.kwargs:
            assert audit_call_kwargs.kwargs.get("dispatch_status") == "would_dispatch", (
                f"Expected dispatch_status='would_dispatch', got "
                f"{audit_call_kwargs.kwargs.get('dispatch_status')}"
            )
        else:
            # positional args — dispatch_status is the 5th arg
            assert audit_call_kwargs.args[4] == "would_dispatch"

        # 2. No PM cycle triggered (begin_pm_cycle not called)
        dispatcher._begin_pm_cycle.assert_not_called()

        # 3. Result indicates observe mode, not dispatch
        assert result is not None, "Observe mode should return a summary dict"
        assert result.get("mode") == "observe", (
            f"Expected mode='observe' in result, got {result}"
        )
        assert result.get("would_dispatch", 0) >= 1, (
            f"Expected would_dispatch >= 1, got {result}"
        )

    @given(
        alert_type=alert_type_strategy,
        symbol=symbol_strategy,
        urgency=urgency_strategy,
        occurrence_count=occurrence_count_strategy,
    )
    @settings(max_examples=100)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_BREAKOUT_MINUTES", 10)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES", 5)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 30)
    def test_dispatch_mode_adds_to_candidates_not_observe(
        self,
        alert_type: str,
        symbol: str,
        urgency: str,
        occurrence_count: int,
    ):
        """When effective mode is 'dispatch', intent is added to dispatch candidates
        and NOT routed through _handle_observe (no would_dispatch audit)."""
        # target_hit fails freshness check, so it never reaches dispatch routing
        assume(alert_type != "target_hit")

        dispatcher = _make_dispatcher()
        intent = _make_fresh_intent(
            alert_type=alert_type,
            symbol=symbol,
            urgency=urgency,
            occurrence_count=occurrence_count,
        )

        # Set up store to return our single intent
        dispatcher._store.query_pending.return_value = [intent]

        # Mock mode_config to return "dispatch" for all alert types
        mock_mode_config = MagicMock()
        mock_mode_config.global_mode = "dispatch"
        mock_mode_config.effective_mode.return_value = "dispatch"

        # We need to intercept the dispatch path. After mode routing,
        # the intent goes through _check_eligibility and _select_dispatch_set.
        # We'll let those proceed and check the PM trigger attempt.
        # Mock _check_eligibility to pass through
        with patch("utils.dispatch_mode.build_dispatch_mode_config", return_value=mock_mode_config):
            # Mock the PM cycle trigger to capture what gets dispatched
            with patch.object(dispatcher, "_trigger_pm_cycle", return_value={"dispatched": True}) as mock_trigger:
                with patch.object(dispatcher, "_check_eligibility", return_value=[intent]) as mock_eligibility:
                    with patch.object(dispatcher, "_select_dispatch_set", return_value=[intent]) as mock_select:
                        result = dispatcher.evaluate_and_dispatch()

        # Assertions for dispatch mode:
        # 1. No would_dispatch audit row (that's observe behavior)
        # record_audit_log may be called for other reasons (e.g., dispatch audit in 8.3)
        # but NOT with dispatch_status="would_dispatch"
        for call_item in dispatcher._store.record_audit_log.call_args_list:
            if call_item.kwargs:
                assert call_item.kwargs.get("dispatch_status") != "would_dispatch", (
                    "Dispatch mode should NOT write would_dispatch audit rows"
                )

        # 2. has_would_dispatch_for_occurrence should NOT be called (not observe path)
        dispatcher._store.has_would_dispatch_for_occurrence.assert_not_called()

        # 3. The intent reached the dispatch pipeline (eligibility check was called)
        mock_eligibility.assert_called_once()
        # Verify our intent was in the candidates list
        candidates_arg = mock_eligibility.call_args[0][0]
        assert intent in candidates_arg, (
            f"Intent should be in dispatch candidates, got {candidates_arg}"
        )

        # 4. PM trigger was attempted (dispatch path completed)
        mock_trigger.assert_called_once()

    @given(
        alert_type=alert_type_strategy,
        effective_mode=effective_mode_strategy,
        symbol=symbol_strategy,
        urgency=urgency_strategy,
    )
    @settings(max_examples=100)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_BREAKOUT_MINUTES", 10)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES", 5)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 30)
    def test_routing_exclusivity_each_mode_uses_exactly_one_path(
        self,
        alert_type: str,
        effective_mode: str,
        symbol: str,
        urgency: str,
    ):
        """Each mode routes through exactly ONE path: disabled skips, observe
        writes audit without PM, dispatch goes to PM without would_dispatch.
        No intent should be routed through multiple paths simultaneously."""
        # target_hit always fails freshness for observe/dispatch modes
        if effective_mode != "disabled":
            assume(alert_type != "target_hit")

        dispatcher = _make_dispatcher()
        intent = _make_fresh_intent(
            alert_type=alert_type,
            symbol=symbol,
            urgency=urgency,
        )

        dispatcher._store.query_pending.return_value = [intent]

        mock_mode_config = MagicMock()
        mock_mode_config.global_mode = "dispatch"  # allow processing to enter the loop
        mock_mode_config.effective_mode.return_value = effective_mode

        with patch("utils.dispatch_mode.build_dispatch_mode_config", return_value=mock_mode_config):
            with patch.object(dispatcher, "_trigger_pm_cycle", return_value={"dispatched": True}) as mock_trigger:
                with patch.object(dispatcher, "_check_eligibility", return_value=[intent]) as mock_eligibility:
                    with patch.object(dispatcher, "_select_dispatch_set", return_value=[intent]):
                        result = dispatcher.evaluate_and_dispatch()

        # Count which paths were taken
        audit_called = dispatcher._store.record_audit_log.called
        observe_dedup_called = dispatcher._store.has_would_dispatch_for_occurrence.called
        pm_triggered = mock_trigger.called
        eligibility_checked = mock_eligibility.called

        if effective_mode == "disabled":
            # Disabled: nothing happens
            assert not audit_called, "Disabled mode must not write audit"
            assert not observe_dedup_called, "Disabled mode must not check observe dedup"
            assert not pm_triggered, "Disabled mode must not trigger PM"
            assert not eligibility_checked, "Disabled mode must not check eligibility"

        elif effective_mode == "observe":
            # Observe: audit is written, PM is NOT triggered
            assert audit_called, "Observe mode must write audit"
            assert not pm_triggered, "Observe mode must NOT trigger PM"
            assert not eligibility_checked, "Observe mode must NOT check eligibility"

        elif effective_mode == "dispatch":
            # Dispatch: eligibility is checked, PM is triggered, no would_dispatch audit
            assert eligibility_checked, "Dispatch mode must check eligibility"
            assert pm_triggered, "Dispatch mode must trigger PM"
            # Verify no would_dispatch audit was written
            for call_item in dispatcher._store.record_audit_log.call_args_list:
                if call_item.kwargs:
                    assert call_item.kwargs.get("dispatch_status") != "would_dispatch", (
                        "Dispatch mode must not write would_dispatch audit rows"
                    )
