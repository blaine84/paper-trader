"""Unit tests for _is_valid_reconfirmation and updated compute_news_governance_state."""

from datetime import datetime, timezone, timedelta

from utils.position_lifecycle_governance import (
    _is_valid_reconfirmation,
    compute_news_governance_state,
    HOLD_BLOCKING_THESIS_STATUSES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENTRY_TIME = datetime(2025, 1, 15, 9, 30, 0, tzinfo=timezone.utc)


def _valid_payload(**overrides):
    """Create a valid RECONFIRM_AND_HOLD payload."""
    base = {
        "decision": "RECONFIRM_AND_HOLD",
        "decided_at": "2025-01-15T14:00:00+00:00",
        "decided_by": "pm_john",
        "new_expiry_time": "2025-01-16T09:30:00+00:00",
        "risk_plan": "Tight stop at support, scale out at resistance",
        "original_catalyst": "AAPL earnings beat Q4 2025",
        "fresh_catalyst_evidence": "New analyst upgrade from Goldman Sachs with PT raise to $250",
        "fresh_catalyst_timestamp": "2025-01-15T12:00:00+00:00",
    }
    base.update(overrides)
    return base


def _make_trade(**overrides):
    """Create a minimal trade dict."""
    trade = {
        "id": 1,
        "symbol": "AAPL",
        "profile": "moderate",
        "setup_type": "news_catalyst",
        "entry_time": ENTRY_TIME,
        "status": "open",
    }
    trade.update(overrides)
    return trade


def _make_reconfirm_event(payload):
    """Wrap a payload in a news_reconfirmation_submitted event."""
    return {
        "event_type": "news_reconfirmation_submitted",
        "timestamp": payload.get("decided_at", "2025-01-15T14:00:00+00:00"),
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# _is_valid_reconfirmation: Rule 1 - Required fields
# ---------------------------------------------------------------------------


def test_valid_payload_passes():
    assert _is_valid_reconfirmation(_valid_payload(), ENTRY_TIME) is True


def test_missing_decided_at_invalid():
    payload = _valid_payload()
    del payload["decided_at"]
    assert _is_valid_reconfirmation(payload, ENTRY_TIME) is False


def test_missing_decided_by_invalid():
    payload = _valid_payload()
    del payload["decided_by"]
    assert _is_valid_reconfirmation(payload, ENTRY_TIME) is False


def test_missing_new_expiry_time_invalid():
    payload = _valid_payload()
    del payload["new_expiry_time"]
    assert _is_valid_reconfirmation(payload, ENTRY_TIME) is False


def test_missing_risk_plan_invalid():
    payload = _valid_payload()
    del payload["risk_plan"]
    assert _is_valid_reconfirmation(payload, ENTRY_TIME) is False


def test_missing_original_catalyst_invalid():
    payload = _valid_payload()
    del payload["original_catalyst"]
    assert _is_valid_reconfirmation(payload, ENTRY_TIME) is False


def test_missing_fresh_catalyst_evidence_invalid():
    payload = _valid_payload()
    del payload["fresh_catalyst_evidence"]
    assert _is_valid_reconfirmation(payload, ENTRY_TIME) is False


def test_missing_fresh_catalyst_timestamp_invalid():
    payload = _valid_payload()
    del payload["fresh_catalyst_timestamp"]
    assert _is_valid_reconfirmation(payload, ENTRY_TIME) is False


def test_empty_decided_by_invalid():
    assert _is_valid_reconfirmation(_valid_payload(decided_by=""), ENTRY_TIME) is False


def test_whitespace_decided_by_invalid():
    assert _is_valid_reconfirmation(_valid_payload(decided_by="   "), ENTRY_TIME) is False


def test_empty_risk_plan_invalid():
    assert _is_valid_reconfirmation(_valid_payload(risk_plan=""), ENTRY_TIME) is False


def test_empty_original_catalyst_invalid():
    assert _is_valid_reconfirmation(_valid_payload(original_catalyst=""), ENTRY_TIME) is False


# ---------------------------------------------------------------------------
# _is_valid_reconfirmation: Rule 2 - fresh_catalyst_evidence >= 20 chars
# ---------------------------------------------------------------------------


def test_evidence_exactly_20_chars_valid():
    evidence = "A" * 20
    assert _is_valid_reconfirmation(_valid_payload(fresh_catalyst_evidence=evidence), ENTRY_TIME) is True


def test_evidence_19_chars_invalid():
    evidence = "A" * 19
    assert _is_valid_reconfirmation(_valid_payload(fresh_catalyst_evidence=evidence), ENTRY_TIME) is False


def test_evidence_empty_invalid():
    assert _is_valid_reconfirmation(_valid_payload(fresh_catalyst_evidence=""), ENTRY_TIME) is False


# ---------------------------------------------------------------------------
# _is_valid_reconfirmation: Rule 3 - fresh_catalyst_timestamp > entry_time
# ---------------------------------------------------------------------------


def test_catalyst_ts_after_entry_valid():
    ts = (ENTRY_TIME + timedelta(hours=1)).isoformat()
    assert _is_valid_reconfirmation(_valid_payload(fresh_catalyst_timestamp=ts), ENTRY_TIME) is True


def test_catalyst_ts_equal_to_entry_invalid():
    ts = ENTRY_TIME.isoformat()
    assert _is_valid_reconfirmation(_valid_payload(fresh_catalyst_timestamp=ts), ENTRY_TIME) is False


def test_catalyst_ts_before_entry_invalid():
    ts = (ENTRY_TIME - timedelta(hours=1)).isoformat()
    assert _is_valid_reconfirmation(_valid_payload(fresh_catalyst_timestamp=ts), ENTRY_TIME) is False


# ---------------------------------------------------------------------------
# _is_valid_reconfirmation: Rule 4 - fresh_catalyst_timestamp > prior's
# ---------------------------------------------------------------------------


def test_catalyst_ts_after_prior_valid():
    prior = _valid_payload(fresh_catalyst_timestamp="2025-01-15T12:00:00+00:00")
    current = _valid_payload(
        decided_at="2025-01-15T18:00:00+00:00",
        fresh_catalyst_timestamp="2025-01-15T16:00:00+00:00",
        new_expiry_time="2025-01-16T14:00:00+00:00",
    )
    assert _is_valid_reconfirmation(current, ENTRY_TIME, prior_reconfirmation=prior) is True


def test_catalyst_ts_equal_to_prior_invalid():
    prior = _valid_payload(fresh_catalyst_timestamp="2025-01-15T12:00:00+00:00")
    current = _valid_payload(
        decided_at="2025-01-15T18:00:00+00:00",
        fresh_catalyst_timestamp="2025-01-15T12:00:00+00:00",
        new_expiry_time="2025-01-16T14:00:00+00:00",
    )
    assert _is_valid_reconfirmation(current, ENTRY_TIME, prior_reconfirmation=prior) is False


def test_catalyst_ts_before_prior_invalid():
    prior = _valid_payload(fresh_catalyst_timestamp="2025-01-15T12:00:00+00:00")
    current = _valid_payload(
        decided_at="2025-01-15T18:00:00+00:00",
        fresh_catalyst_timestamp="2025-01-15T11:00:00+00:00",
        new_expiry_time="2025-01-16T14:00:00+00:00",
    )
    assert _is_valid_reconfirmation(current, ENTRY_TIME, prior_reconfirmation=prior) is False


def test_no_prior_reconfirmation_skips_rule4():
    """When prior_reconfirmation is None, rule 4 is not checked."""
    assert _is_valid_reconfirmation(_valid_payload(), ENTRY_TIME, prior_reconfirmation=None) is True


# ---------------------------------------------------------------------------
# _is_valid_reconfirmation: Rule 5 - thesis_status not in blocking set
# ---------------------------------------------------------------------------


def test_deteriorating_thesis_invalid():
    assert _is_valid_reconfirmation(
        _valid_payload(thesis_status="deteriorating"), ENTRY_TIME
    ) is False


def test_invalidated_thesis_invalid():
    assert _is_valid_reconfirmation(
        _valid_payload(thesis_status="invalidated"), ENTRY_TIME
    ) is False


def test_healthy_thesis_valid():
    assert _is_valid_reconfirmation(
        _valid_payload(thesis_status="healthy"), ENTRY_TIME
    ) is True


def test_missing_thesis_status_valid():
    """thesis_status is not a required field; missing means not blocking."""
    payload = _valid_payload()
    # thesis_status not in payload at all
    assert "thesis_status" not in payload
    assert _is_valid_reconfirmation(payload, ENTRY_TIME) is True


def test_empty_thesis_status_valid():
    """Empty string thesis_status is not in the blocking set."""
    assert _is_valid_reconfirmation(
        _valid_payload(thesis_status=""), ENTRY_TIME
    ) is True


# ---------------------------------------------------------------------------
# _is_valid_reconfirmation: Rule 6 - new_expiry_time <= decided_at + 24h
# ---------------------------------------------------------------------------


def test_expiry_within_24h_valid():
    decided = "2025-01-15T14:00:00+00:00"
    expiry = "2025-01-16T14:00:00+00:00"  # exactly 24h
    assert _is_valid_reconfirmation(
        _valid_payload(decided_at=decided, new_expiry_time=expiry), ENTRY_TIME
    ) is True


def test_expiry_exceeds_24h_invalid():
    decided = "2025-01-15T14:00:00+00:00"
    expiry = "2025-01-16T14:00:01+00:00"  # 24h + 1 second
    assert _is_valid_reconfirmation(
        _valid_payload(decided_at=decided, new_expiry_time=expiry), ENTRY_TIME
    ) is False


def test_expiry_well_within_24h_valid():
    decided = "2025-01-15T14:00:00+00:00"
    expiry = "2025-01-15T20:00:00+00:00"  # 6h
    assert _is_valid_reconfirmation(
        _valid_payload(decided_at=decided, new_expiry_time=expiry), ENTRY_TIME
    ) is True


# ---------------------------------------------------------------------------
# _is_valid_reconfirmation: Edge cases
# ---------------------------------------------------------------------------


def test_unparseable_fresh_catalyst_timestamp_invalid():
    assert _is_valid_reconfirmation(
        _valid_payload(fresh_catalyst_timestamp="not-a-date"), ENTRY_TIME
    ) is False


def test_unparseable_decided_at_invalid():
    assert _is_valid_reconfirmation(
        _valid_payload(decided_at="not-a-date"), ENTRY_TIME
    ) is False


def test_unparseable_new_expiry_time_invalid():
    assert _is_valid_reconfirmation(
        _valid_payload(new_expiry_time="not-a-date"), ENTRY_TIME
    ) is False


# ---------------------------------------------------------------------------
# compute_news_governance_state: Sequential validation
# ---------------------------------------------------------------------------


def test_valid_reconfirmation_extends_expiry():
    """A valid reconfirmation should extend the effective expiry."""
    trade = _make_trade()
    new_expiry = "2025-01-16T14:00:00+00:00"
    events = [_make_reconfirm_event(_valid_payload(new_expiry_time=new_expiry))]

    # Set now_utc to 3h before new expiry (within warning window)
    now_utc = datetime(2025, 1, 16, 11, 0, 0, tzinfo=timezone.utc)
    result = compute_news_governance_state(trade, events, now_utc)

    assert result is not None
    assert result["state"] == "news_reconfirmation_due"
    assert "2025-01-16" in result["metadata"]["effective_expiry"]
    assert "14:00:00" in result["metadata"]["effective_expiry"]


def test_invalid_reconfirmation_ignored():
    """An invalid reconfirmation (short evidence) should be treated as non-existent."""
    trade = _make_trade()
    # Evidence too short (< 20 chars)
    payload = _valid_payload(fresh_catalyst_evidence="too short")
    events = [_make_reconfirm_event(payload)]

    # now_utc past entry_time + 24h + 30min → should be news_expired
    now_utc = ENTRY_TIME + timedelta(hours=24, minutes=31)
    result = compute_news_governance_state(trade, events, now_utc)

    assert result is not None
    assert result["state"] == "news_expired"
    assert result["decision"] == "close"


def test_sequential_validation_skips_invalid_middle():
    """Invalid reconfirmation in the middle is skipped; latest valid one wins."""
    trade = _make_trade()

    # First valid reconfirmation
    r1 = _valid_payload(
        decided_at="2025-01-15T14:00:00+00:00",
        fresh_catalyst_timestamp="2025-01-15T12:00:00+00:00",
        new_expiry_time="2025-01-16T09:30:00+00:00",
    )

    # Second reconfirmation: invalid (thesis_status = deteriorating)
    r2 = _valid_payload(
        decided_at="2025-01-15T18:00:00+00:00",
        fresh_catalyst_timestamp="2025-01-15T16:00:00+00:00",
        new_expiry_time="2025-01-16T14:00:00+00:00",
        thesis_status="deteriorating",
    )

    # Third valid reconfirmation (uses r1 as prior since r2 was invalid)
    r3 = _valid_payload(
        decided_at="2025-01-15T22:00:00+00:00",
        fresh_catalyst_timestamp="2025-01-15T20:00:00+00:00",
        new_expiry_time="2025-01-16T18:00:00+00:00",
    )

    events = [_make_reconfirm_event(r1), _make_reconfirm_event(r2), _make_reconfirm_event(r3)]

    # now_utc within warning window of r3's expiry
    now_utc = datetime(2025, 1, 16, 15, 0, 0, tzinfo=timezone.utc)
    result = compute_news_governance_state(trade, events, now_utc)

    assert result is not None
    assert result["state"] == "news_reconfirmation_due"
    assert "2025-01-16" in result["metadata"]["effective_expiry"]
    assert "18:00:00" in result["metadata"]["effective_expiry"]


def test_all_reconfirmations_invalid_uses_entry_plus_24h():
    """When all reconfirmations are invalid, effective_expiry = entry_time + 24h."""
    trade = _make_trade()

    # Invalid: evidence too short
    r1 = _valid_payload(fresh_catalyst_evidence="short")

    events = [_make_reconfirm_event(r1)]

    # now_utc past entry + 24h + 30min
    now_utc = ENTRY_TIME + timedelta(hours=24, minutes=31)
    result = compute_news_governance_state(trade, events, now_utc)

    assert result is not None
    assert result["state"] == "news_expired"
    assert result["decision"] == "close"


def test_hold_blocking_thesis_statuses_constant():
    """Verify the module constant contains the expected values."""
    assert HOLD_BLOCKING_THESIS_STATUSES == {"deteriorating", "invalidated"}
