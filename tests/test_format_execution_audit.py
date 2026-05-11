"""Unit tests for _format_execution_audit categorized output."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from agents.portfolio_manager import _format_execution_audit, NormalizedOrder, NonOrderRecord


class TestFormatExecutionAuditEmpty:
    """Tests for empty/no-decisions case."""

    def test_all_empty_returns_no_decisions_message(self):
        result = _format_execution_audit([], gate_blocked=[], rejections=[], non_orders=[])
        assert result == "Execution audit: no decisions this cycle."

    def test_defaults_none_returns_no_decisions_message(self):
        result = _format_execution_audit([])
        assert result == "Execution audit: no decisions this cycle."


class TestFormatExecutionAuditExecuted:
    """Tests for EXECUTED category."""

    def test_single_executed_trade(self):
        executed = [{"action": "BUY", "symbol": "NVDA", "quantity": 50, "entry_price": 130.50, "message": "filled"}]
        result = _format_execution_audit(executed)
        assert "── EXECUTED (1) ──" in result
        assert "• BUY 50 NVDA @ $130.50 — filled" in result

    def test_multiple_executed_trades(self):
        executed = [
            {"action": "BUY", "symbol": "NVDA", "quantity": 50, "entry_price": 130.50, "message": "filled"},
            {"action": "SHORT", "symbol": "AMD", "quantity": 30, "price": 95.00, "message": "filled"},
        ]
        result = _format_execution_audit(executed)
        assert "── EXECUTED (2) ──" in result
        assert "NVDA" in result
        assert "AMD" in result


class TestFormatExecutionAuditGateBlocked:
    """Tests for BLOCKED BY GATE category."""

    def test_single_gate_blocked(self):
        gate_blocked = [{"action": "BUY", "symbol": "AMD", "message": "missing setup_type"}]
        result = _format_execution_audit([], gate_blocked=gate_blocked)
        assert "── BLOCKED BY GATE (1) ──" in result
        assert "• BUY AMD — missing setup_type" in result

    def test_gate_blocked_no_message(self):
        gate_blocked = [{"action": "SHORT", "symbol": "TSLA"}]
        result = _format_execution_audit([], gate_blocked=gate_blocked)
        assert "── BLOCKED BY GATE (1) ──" in result
        assert "• SHORT TSLA" in result


class TestFormatExecutionAuditRejections:
    """Tests for REJECTED MALFORMED PM OUTPUT category."""

    def test_rejections_counted_by_reason_code(self):
        rejections = [
            NormalizedOrder(ok=False, reason_code="unsupported_symbol", reason="bad symbol", raw_decision={}),
            NormalizedOrder(ok=False, reason_code="unsupported_symbol", reason="bad symbol", raw_decision={}),
            NormalizedOrder(ok=False, reason_code="invalid_quantity", reason="bad qty", raw_decision={}),
        ]
        result = _format_execution_audit([], rejections=rejections)
        assert "── REJECTED MALFORMED PM OUTPUT (3) ──" in result
        assert "unsupported_symbol: 2" in result
        assert "invalid_quantity: 1" in result

    def test_rejections_no_individual_rows(self):
        """Rejections should NOT show individual trade details like BUY 0 X @ $0.00."""
        rejections = [
            NormalizedOrder(ok=False, reason_code="unsupported_symbol", reason="bad", raw_decision={"action": "BUY", "symbol": "XAU/USD"}),
        ]
        result = _format_execution_audit([], rejections=rejections)
        assert "XAU/USD" not in result
        assert "BUY" not in result.split("REJECTED")[1] or "unsupported_symbol" in result


class TestFormatExecutionAuditNonOrders:
    """Tests for IGNORED NON-ORDER category."""

    def test_non_orders_counted_by_action(self):
        non_orders = [
            NonOrderRecord(action="HOLD", symbol="NVDA", raw_decision={}),
            NonOrderRecord(action="WATCH", symbol="AMD", raw_decision={}),
            NonOrderRecord(action="HOLD", symbol="TSLA", raw_decision={}),
        ]
        result = _format_execution_audit([], non_orders=non_orders)
        assert "── IGNORED NON-ORDER (3) ──" in result
        assert "HOLD: 2" in result
        assert "WATCH: 1" in result

    def test_non_orders_no_individual_rows(self):
        """Non-orders should NOT show individual symbols."""
        non_orders = [
            NonOrderRecord(action="HOLD", symbol="NVDA", raw_decision={}),
        ]
        result = _format_execution_audit([], non_orders=non_orders)
        assert "NVDA" not in result


class TestFormatExecutionAuditMixed:
    """Tests for mixed categories."""

    def test_all_categories_present(self):
        executed = [{"action": "BUY", "symbol": "NVDA", "quantity": 50, "entry_price": 130.50, "message": "filled"}]
        gate_blocked = [{"action": "BUY", "symbol": "AMD", "message": "missing setup_type"}]
        rejections = [
            NormalizedOrder(ok=False, reason_code="unsupported_symbol", reason="bad", raw_decision={}),
        ]
        non_orders = [
            NonOrderRecord(action="HOLD", symbol="TSLA", raw_decision={}),
        ]
        result = _format_execution_audit(executed, gate_blocked=gate_blocked, rejections=rejections, non_orders=non_orders)
        assert "── EXECUTED (1) ──" in result
        assert "── BLOCKED BY GATE (1) ──" in result
        assert "── REJECTED MALFORMED PM OUTPUT (1) ──" in result
        assert "── IGNORED NON-ORDER (1) ──" in result

    def test_empty_categories_not_shown(self):
        """Only categories with items should appear."""
        executed = [{"action": "BUY", "symbol": "NVDA", "quantity": 50, "entry_price": 130.50, "message": "filled"}]
        result = _format_execution_audit(executed)
        assert "EXECUTED" in result
        assert "BLOCKED BY GATE" not in result
        assert "REJECTED MALFORMED" not in result
        assert "IGNORED NON-ORDER" not in result
