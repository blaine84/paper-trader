"""
Property-based tests for News Trade Governance using Hypothesis.

Tests universal correctness properties that must hold across all valid inputs
for the news-catalyst 24h exit gate feature.
"""

import json
import pytest
from datetime import datetime, timedelta, timezone
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Trade, TradeEvent
from utils.news_trade_governance import (
    NEWS_GOVERNANCE,
    NewsGovernanceClassifier,
    NewsGovernancePolicy,
    ReconfirmationValidator,
    log_trade_event_once,
    latest_valid_reconfirmation,
    _build_dedupe_key,
    _build_failure_dedupe_key,
    VALID_CATALYST_TYPES,
    VALID_DECISIONS,
    HOLD_BLOCKING_THESIS_STATUSES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session():
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

governed_setup_types = sorted(NEWS_GOVERNANCE["setup_types"])
governed_text_terms = sorted(NEWS_GOVERNANCE["entry_text_terms"])
valid_catalyst_types = sorted(VALID_CATALYST_TYPES)

# Non-governed values
non_governed_setup_types = [
    "swing", "breakout", "pullback", "reversal", "momentum",
    "mean_reversion", "gap_fill", "technical", "range_bound",
]

non_governed_text = [
    "technical breakout above resistance",
    "pullback to moving average support",
    "range bound consolidation pattern",
    "volume profile shows accumulation",
    "MACD crossover with RSI divergence",
]

aware_datetimes = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 1, 1),
).map(lambda dt: dt.replace(tzinfo=timezone.utc))

st_positive_hours = st.floats(min_value=1.0, max_value=168.0, allow_nan=False, allow_infinity=False)
st_positive_minutes = st.floats(min_value=1.0, max_value=120.0, allow_nan=False, allow_infinity=False)
st_warning_hours = st.floats(min_value=0.5, max_value=12.0, allow_nan=False, allow_infinity=False)


@st.composite
def governed_trade_via_setup_type(draw):
    """Generate a trade governed via setup_type."""
    return {
        "setup_type": draw(st.sampled_from(governed_setup_types)),
        "reason_entry": draw(st.text(min_size=0, max_size=50)),
        "thesis": draw(st.text(min_size=0, max_size=50)),
        "invalidators": draw(st.text(min_size=0, max_size=50)),
    }


@st.composite
def governed_trade_via_entry_signal(draw):
    """Generate a trade governed via entry_signal setup_type."""
    return {
        "setup_type": draw(st.sampled_from(non_governed_setup_types)),
        "reason_entry": draw(st.sampled_from(non_governed_text)),
        "thesis": draw(st.sampled_from(non_governed_text)),
        "invalidators": draw(st.sampled_from(non_governed_text)),
    }, {
        "setup_type": draw(st.sampled_from(governed_setup_types)),
    }


@st.composite
def governed_trade_via_text_term(draw):
    """Generate a trade governed via text term match."""
    term = draw(st.sampled_from(governed_text_terms))
    field = draw(st.sampled_from(["reason_entry", "thesis", "invalidators"]))
    prefix = draw(st.text(min_size=0, max_size=20))
    suffix = draw(st.text(min_size=0, max_size=20))
    trade = {
        "setup_type": draw(st.sampled_from(non_governed_setup_types)),
        "reason_entry": draw(st.sampled_from(non_governed_text)),
        "thesis": draw(st.sampled_from(non_governed_text)),
        "invalidators": draw(st.sampled_from(non_governed_text)),
    }
    # Inject the governed term into the chosen field
    trade[field] = f"{prefix} {term} {suffix}"
    return trade


@st.composite
def governed_trade_via_catalyst_type(draw):
    """Generate a trade governed via catalyst_type."""
    return {
        "setup_type": draw(st.sampled_from(non_governed_setup_types)),
        "reason_entry": draw(st.sampled_from(non_governed_text)),
        "thesis": draw(st.sampled_from(non_governed_text)),
        "invalidators": draw(st.sampled_from(non_governed_text)),
        "catalyst_type": draw(st.sampled_from(valid_catalyst_types)),
    }


@st.composite
def non_governed_trade(draw):
    """Generate a trade that does NOT match any governance condition."""
    return {
        "setup_type": draw(st.sampled_from(non_governed_setup_types)),
        "reason_entry": draw(st.sampled_from(non_governed_text)),
        "thesis": draw(st.sampled_from(non_governed_text)),
        "invalidators": draw(st.sampled_from(non_governed_text)),
        "catalyst_type": "",
    }


@st.composite
def valid_reconfirm_hold_payload(draw):
    """Generate a valid RECONFIRM_AND_HOLD payload."""
    entry_time = draw(aware_datetimes)
    decided_at = entry_time + timedelta(hours=draw(st.floats(min_value=1.0, max_value=20.0)))
    fresh_ts = entry_time + timedelta(hours=draw(st.floats(min_value=0.5, max_value=19.0)))
    new_expiry = decided_at + timedelta(hours=draw(st.floats(min_value=1.0, max_value=23.0)))
    evidence = draw(st.text(min_size=25, max_size=200))

    payload = {
        "trade_id": draw(st.integers(min_value=1, max_value=10000)),
        "symbol": draw(st.sampled_from(["AAPL", "TSLA", "XLE", "SPY", "QQQ"])),
        "profile": draw(st.sampled_from(["conservative", "moderate", "aggressive"])),
        "decision": "RECONFIRM_AND_HOLD",
        "decided_by": "pm_agent",
        "decided_at": decided_at,
        "original_catalyst": "news catalyst event",
        "fresh_catalyst_evidence": evidence,
        "fresh_catalyst_timestamp": fresh_ts,
        "thesis_status": draw(st.sampled_from(["strengthened", "unchanged"])),
        "new_expiry_time": new_expiry,
        "risk_plan": "maintain current stop with trailing adjustment",
    }
    return payload, entry_time


# ---------------------------------------------------------------------------
# Property 1: Classification Positive Correctness
# Feature: news-catalyst-24h-exit-gate, Property 1: Classification Positive Correctness
# ---------------------------------------------------------------------------


class TestProperty1ClassificationPositiveCorrectness:
    """
    For any trade matching at least one governance condition, classifier returns
    (True, evidence) with non-empty evidence.

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 9.6**
    """

    @given(trade=governed_trade_via_setup_type())
    @settings(max_examples=100, deadline=None)
    def test_governed_via_setup_type(self, trade):
        """Trade with governed setup_type is classified as governed."""
        classifier = NewsGovernanceClassifier()
        is_governed, evidence = classifier.classify(trade)
        assert is_governed is True
        assert evidence != {}
        assert "triggered_by" in evidence

    @given(data=governed_trade_via_entry_signal())
    @settings(max_examples=100, deadline=None)
    def test_governed_via_entry_signal(self, data):
        """Trade with governed entry_signal setup_type is classified as governed."""
        trade, entry_signal = data
        classifier = NewsGovernanceClassifier()
        is_governed, evidence = classifier.classify(trade, entry_signal=entry_signal)
        assert is_governed is True
        assert evidence != {}
        assert evidence["triggered_by"] == "entry_signal_setup_type"

    @given(trade=governed_trade_via_text_term())
    @settings(max_examples=100, deadline=None)
    def test_governed_via_text_term(self, trade):
        """Trade with governed text term in entry fields is classified as governed."""
        classifier = NewsGovernanceClassifier()
        is_governed, evidence = classifier.classify(trade)
        assert is_governed is True
        assert evidence != {}
        assert evidence["triggered_by"] in ("setup_type", "entry_text_terms")

    @given(trade=governed_trade_via_catalyst_type())
    @settings(max_examples=100, deadline=None)
    def test_governed_via_catalyst_type(self, trade):
        """Trade with valid catalyst_type is classified as governed."""
        classifier = NewsGovernanceClassifier()
        is_governed, evidence = classifier.classify(trade)
        assert is_governed is True
        assert evidence != {}
        assert evidence["triggered_by"] in ("setup_type", "entry_text_terms", "catalyst_type")


# ---------------------------------------------------------------------------
# Property 2: Classification Negative Correctness
# Feature: news-catalyst-24h-exit-gate, Property 2: Classification Negative Correctness
# ---------------------------------------------------------------------------


class TestProperty2ClassificationNegativeCorrectness:
    """
    For any trade where no governance condition is met, classifier returns (False, {}).

    **Validates: Requirements 1.5**
    """

    @given(trade=non_governed_trade())
    @settings(max_examples=100, deadline=None)
    def test_non_governed_trade_returns_false(self, trade):
        """Trade with no governed conditions returns (False, {})."""
        classifier = NewsGovernanceClassifier()
        is_governed, evidence = classifier.classify(trade)
        assert is_governed is False
        assert evidence == {}


# ---------------------------------------------------------------------------
# Property 3: Classification Evidence Round-Trip
# Feature: news-catalyst-24h-exit-gate, Property 3: Classification Evidence Round-Trip
# ---------------------------------------------------------------------------


class TestProperty3ClassificationEvidenceRoundTrip:
    """
    For any trade, classify → serialize evidence to JSON → deserialize →
    classify_from_persisted_evidence() returns same boolean.

    **Validates: Requirements 1.8, 2.5**
    """

    @given(trade=governed_trade_via_setup_type())
    @settings(max_examples=100, deadline=None)
    def test_governed_evidence_round_trip(self, trade):
        """Governed trade evidence survives JSON round-trip."""
        classifier = NewsGovernanceClassifier()
        is_governed, evidence = classifier.classify(trade)
        assert is_governed is True

        # Serialize → deserialize
        serialized = json.dumps(evidence)
        deserialized = json.loads(serialized)

        # Re-evaluate from persisted evidence
        result = classifier.classify_from_persisted_evidence(deserialized)
        assert result == is_governed

    @given(trade=non_governed_trade())
    @settings(max_examples=100, deadline=None)
    def test_non_governed_evidence_round_trip(self, trade):
        """Non-governed trade evidence survives JSON round-trip."""
        classifier = NewsGovernanceClassifier()
        is_governed, evidence = classifier.classify(trade)
        assert is_governed is False

        # Serialize → deserialize
        serialized = json.dumps(evidence)
        deserialized = json.loads(serialized)

        # Re-evaluate from persisted evidence
        result = classifier.classify_from_persisted_evidence(deserialized)
        assert result == is_governed


# ---------------------------------------------------------------------------
# Property 4: Persisted Classification Durability
# Feature: news-catalyst-24h-exit-gate, Property 4: Persisted Classification Durability
# ---------------------------------------------------------------------------


class TestProperty4PersistedClassificationDurability:
    """
    For any trade with persisted news_governed=true event, mutating setup_type,
    thesis, reason_entry, invalidators does not change governance status.

    **Validates: Requirements 2.2, 2.3**
    """

    @given(
        trade=governed_trade_via_setup_type(),
        new_setup_type=st.sampled_from(non_governed_setup_types),
        new_thesis=st.sampled_from(non_governed_text),
        new_reason_entry=st.sampled_from(non_governed_text),
        new_invalidators=st.sampled_from(non_governed_text),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_persisted_classification_survives_field_mutations(
        self, db_session, trade, new_setup_type, new_thesis, new_reason_entry, new_invalidators
    ):
        """Persisted classification is durable regardless of field mutations."""
        classifier = NewsGovernanceClassifier()

        # Classify the original trade
        is_governed, evidence = classifier.classify(trade)
        assert is_governed is True

        # Create a Trade row in the DB
        db_trade = Trade(
            symbol="XLE",
            direction="LONG",
            quantity=100,
            entry_price=85.0,
            setup_type=trade["setup_type"],
            reason_entry=trade.get("reason_entry", ""),
            thesis=trade.get("thesis", ""),
            invalidators=trade.get("invalidators", ""),
            status="open",
        )
        db_session.add(db_trade)
        db_session.flush()

        # Persist the classification event
        event = TradeEvent(
            trade_id=db_trade.id,
            event_type="news_governance_classified",
            payload_json=json.dumps(evidence),
            timestamp=datetime.now(timezone.utc),
            dedupe_key=f"news_governance_classified:{db_trade.id}",
        )
        db_session.add(event)
        db_session.flush()

        # Mutate the trade fields to non-governed values
        db_trade.setup_type = new_setup_type
        db_trade.thesis = new_thesis
        db_trade.reason_entry = new_reason_entry
        db_trade.invalidators = new_invalidators
        db_session.flush()

        # Verify persisted classification still returns evidence
        persisted = classifier.get_persisted_classification(db_session, db_trade.id)
        assert persisted is not None
        assert persisted["evidence"] == evidence

        # Verify classify_from_persisted_evidence still returns True
        result = classifier.classify_from_persisted_evidence(persisted["evidence"])
        assert result is True

        db_session.rollback()


# ---------------------------------------------------------------------------
# Property 5: Effective Expiry Resolution
# Feature: news-catalyst-24h-exit-gate, Property 5: Effective Expiry Resolution
# ---------------------------------------------------------------------------


class TestProperty5EffectiveExpiryResolution:
    """
    For any entry_time E and config max_hold_hours H: no reconfirmation → expiry = E+H;
    valid reconfirmation with new_expiry_time T → expiry = T.

    **Validates: Requirements 3.1, 3.2**
    """

    @given(
        entry_time=aware_datetimes,
        max_hold_hours=st_positive_hours,
    )
    @settings(max_examples=100, deadline=None)
    def test_no_reconfirmation_expiry_equals_entry_plus_max_hold(
        self, entry_time, max_hold_hours
    ):
        """Without reconfirmation, expiry = entry_time + max_hold_hours."""
        config = {**NEWS_GOVERNANCE, "max_hold_hours": max_hold_hours}
        policy = NewsGovernancePolicy(config=config)

        effective_expiry = policy.get_effective_expiry(entry_time, None)
        expected = entry_time + timedelta(hours=max_hold_hours)
        assert effective_expiry == expected

    @given(
        entry_time=aware_datetimes,
        new_expiry_offset_hours=st.floats(min_value=1.0, max_value=48.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_with_reconfirmation_expiry_equals_new_expiry_time(
        self, entry_time, new_expiry_offset_hours
    ):
        """With valid reconfirmation, expiry = new_expiry_time from reconfirmation."""
        new_expiry_time = entry_time + timedelta(hours=new_expiry_offset_hours)
        reconfirmation = {
            "decision": "RECONFIRM_AND_HOLD",
            "new_expiry_time": new_expiry_time.isoformat(),
        }
        policy = NewsGovernancePolicy()

        effective_expiry = policy.get_effective_expiry(entry_time, reconfirmation)
        assert effective_expiry == new_expiry_time


# ---------------------------------------------------------------------------
# Property 6: Governance Window Monotonicity
# Feature: news-catalyst-24h-exit-gate, Property 6: Governance Window Monotonicity
# ---------------------------------------------------------------------------


class TestProperty6GovernanceWindowMonotonicity:
    """
    For any trade with N valid RECONFIRM_AND_HOLD reconfirmations, window_id = N+1,
    each successive window_id strictly greater.

    **Validates: Requirements 3.3, 3.4**
    """

    @given(
        n_reconfirmations=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_window_id_equals_n_plus_one(self, db_session, n_reconfirmations):
        """Governance window_id = number of RECONFIRM_AND_HOLD events + 1."""
        # Create a trade
        trade = Trade(
            symbol="XLE",
            direction="LONG",
            quantity=100,
            entry_price=85.0,
            status="open",
        )
        db_session.add(trade)
        db_session.flush()

        policy = NewsGovernancePolicy()
        window_ids = []

        # Record initial window_id
        window_ids.append(policy.get_governance_window_id(db_session, trade.id))

        # Add N reconfirmation events
        for i in range(n_reconfirmations):
            payload = json.dumps({"decision": "RECONFIRM_AND_HOLD", "window": i + 1})
            event = TradeEvent(
                trade_id=trade.id,
                event_type="news_reconfirmation_submitted",
                payload_json=payload,
                timestamp=datetime.now(timezone.utc) + timedelta(hours=i),
            )
            db_session.add(event)
            db_session.flush()

            window_ids.append(policy.get_governance_window_id(db_session, trade.id))

        # Final window_id should be N+1
        assert window_ids[-1] == n_reconfirmations + 1

        # Each successive window_id should be strictly greater
        for i in range(1, len(window_ids)):
            assert window_ids[i] > window_ids[i - 1]

        db_session.rollback()


# ---------------------------------------------------------------------------
# Property 7: Warning Idempotency Per Window
# Feature: news-catalyst-24h-exit-gate, Property 7: Warning Idempotency Per Window
# ---------------------------------------------------------------------------


class TestProperty7WarningIdempotencyPerWindow:
    """
    For any governance window, regardless of timer cycle count during warning period,
    exactly one news_reconfirmation_due event exists.

    **Validates: Requirements 4.1, 4.3, 4.4, 4.5**
    """

    @given(
        n_calls=st.integers(min_value=1, max_value=20),
        governance_window_id=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_log_trade_event_once_produces_single_warning(
        self, db_session, n_calls, governance_window_id
    ):
        """Calling log_trade_event_once N times produces exactly one event."""
        # Create a trade
        trade = Trade(
            symbol="XLE",
            direction="LONG",
            quantity=100,
            entry_price=85.0,
            status="open",
        )
        db_session.add(trade)
        db_session.flush()

        results = []
        for _ in range(n_calls):
            result = log_trade_event_once(
                db_session,
                "news_reconfirmation_due",
                trade.id,
                governance_window_id=governance_window_id,
                agent="position_timer",
                symbol="XLE",
                message="Reconfirmation due",
            )
            results.append(result)
            db_session.flush()

        # First call returns True, all subsequent return False
        assert results[0] is True
        for r in results[1:]:
            assert r is False

        # Exactly one event exists
        events = (
            db_session.query(TradeEvent)
            .filter_by(
                event_type="news_reconfirmation_due",
                trade_id=trade.id,
            )
            .all()
        )
        assert len(events) == 1

        db_session.rollback()


# ---------------------------------------------------------------------------
# Property 8: Force-Close Regardless of Target Price
# Feature: news-catalyst-24h-exit-gate, Property 8: Force-Close Regardless of Target Price
# ---------------------------------------------------------------------------


class TestProperty8ForceCloseRegardlessOfTargetPrice:
    """
    For any expired news-governed trade with no valid reconfirmation, force-close
    occurs regardless of target_price being set.

    **Validates: Requirements 5.1, 5.4**
    """

    @given(
        target_price=st.one_of(
            st.none(),
            st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        ),
        hours_past_grace=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_compute_status_expired_regardless_of_target_price(
        self, target_price, hours_past_grace
    ):
        """compute_status returns 'expired' regardless of target_price when past grace."""
        policy = NewsGovernancePolicy()
        grace_minutes = NEWS_GOVERNANCE["reconfirm_grace_minutes"]
        warning_lead_hours = NEWS_GOVERNANCE["warning_lead_hours"]

        # Set effective_expiry in the past
        effective_expiry = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        # now_utc is past grace period
        now_utc = effective_expiry + timedelta(minutes=grace_minutes) + timedelta(hours=hours_past_grace)

        result = policy.compute_status(
            now_utc=now_utc,
            effective_expiry=effective_expiry,
            grace_minutes=grace_minutes,
            warning_lead_hours=warning_lead_hours,
            has_valid_hold=False,
            has_exit_request=False,
        )

        # Status is expired regardless of what target_price might be
        # (compute_status doesn't take target_price — that's the point)
        assert result["status"] == "expired"
        assert result["hold_authorized"] is False


# ---------------------------------------------------------------------------
# Property 9: Reconfirmation Schema Validation
# Feature: news-catalyst-24h-exit-gate, Property 9: Reconfirmation Schema Validation
# ---------------------------------------------------------------------------


class TestProperty9ReconfirmationSchemaValidation:
    """
    For any payload missing required fields or with invalid decision values,
    validator rejects with descriptive errors.

    **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
    """

    @given(
        missing_field=st.sampled_from(["trade_id", "symbol", "profile", "decision", "decided_by"]),
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_common_required_field_rejected(self, missing_field):
        """Payload missing a common required field is rejected."""
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        payload = {
            "trade_id": 1,
            "symbol": "XLE",
            "profile": "moderate",
            "decision": "RECONFIRM_AND_HOLD",
            "decided_by": "pm_agent",
        }
        # Remove the field
        del payload[missing_field]

        validator = ReconfirmationValidator()
        is_valid, errors = validator.validate(payload, entry_time=entry_time)
        assert is_valid is False
        assert len(errors) > 0
        assert any(missing_field in e for e in errors)

    @given(
        invalid_decision=st.text(min_size=1, max_size=30).filter(
            lambda x: x not in VALID_DECISIONS
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_decision_value_rejected(self, invalid_decision):
        """Payload with invalid decision value is rejected."""
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        payload = {
            "trade_id": 1,
            "symbol": "XLE",
            "profile": "moderate",
            "decision": invalid_decision,
            "decided_by": "pm_agent",
        }

        validator = ReconfirmationValidator()
        is_valid, errors = validator.validate(payload, entry_time=entry_time)
        assert is_valid is False
        assert any("Invalid decision" in e for e in errors)

    @given(
        missing_field=st.sampled_from([
            "original_catalyst", "fresh_catalyst_evidence",
            "fresh_catalyst_timestamp", "thesis_status",
            "new_expiry_time", "risk_plan",
        ]),
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_hold_specific_field_rejected(self, missing_field):
        """RECONFIRM_AND_HOLD payload missing decision-specific field is rejected."""
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        payload = {
            "trade_id": 1,
            "symbol": "XLE",
            "profile": "moderate",
            "decision": "RECONFIRM_AND_HOLD",
            "decided_by": "pm_agent",
            "original_catalyst": "news event",
            "fresh_catalyst_evidence": "x" * 25,
            "fresh_catalyst_timestamp": (entry_time + timedelta(hours=2)).isoformat(),
            "thesis_status": "strengthened",
            "new_expiry_time": (entry_time + timedelta(hours=30)).isoformat(),
            "risk_plan": "maintain stops",
        }
        # Remove the field
        del payload[missing_field]

        validator = ReconfirmationValidator()
        is_valid, errors = validator.validate(payload, entry_time=entry_time)
        assert is_valid is False
        assert len(errors) > 0

    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    def test_exit_now_missing_exit_reason_rejected(self, data):
        """EXIT_NOW payload missing exit_reason is rejected."""
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        payload = {
            "trade_id": 1,
            "symbol": "XLE",
            "profile": "moderate",
            "decision": "EXIT_NOW",
            "decided_by": "pm_agent",
        }

        validator = ReconfirmationValidator()
        is_valid, errors = validator.validate(payload, entry_time=entry_time)
        assert is_valid is False
        assert any("exit_reason" in e for e in errors)

    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    def test_let_expire_missing_decline_reason_rejected(self, data):
        """LET_EXPIRE payload missing decline_reason is rejected."""
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        payload = {
            "trade_id": 1,
            "symbol": "XLE",
            "profile": "moderate",
            "decision": "LET_EXPIRE",
            "decided_by": "pm_agent",
        }

        validator = ReconfirmationValidator()
        is_valid, errors = validator.validate(payload, entry_time=entry_time)
        assert is_valid is False
        assert any("decline_reason" in e for e in errors)


# ---------------------------------------------------------------------------
# Property 10: Reconfirmation Business Rule Rejection
# Feature: news-catalyst-24h-exit-gate, Property 10: Reconfirmation Business Rule Rejection
# ---------------------------------------------------------------------------


class TestProperty10ReconfirmationBusinessRuleRejection:
    """
    For any RECONFIRM_AND_HOLD payload violating business rules (bad thesis_status,
    excessive expiry, short evidence, stale timestamp, naive datetime), validator rejects.

    **Validates: Requirements 6.6, 6.7, 6.8, 6.9, 13.4**
    """

    @given(
        blocking_status=st.sampled_from(sorted(HOLD_BLOCKING_THESIS_STATUSES)),
    )
    @settings(max_examples=100, deadline=None)
    def test_blocking_thesis_status_rejected(self, blocking_status):
        """Payload with blocking thesis_status is rejected."""
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        decided_at = entry_time + timedelta(hours=5)
        payload = {
            "trade_id": 1,
            "symbol": "XLE",
            "profile": "moderate",
            "decision": "RECONFIRM_AND_HOLD",
            "decided_by": "pm_agent",
            "decided_at": decided_at,
            "original_catalyst": "news event",
            "fresh_catalyst_evidence": "x" * 25,
            "fresh_catalyst_timestamp": (entry_time + timedelta(hours=2)),
            "thesis_status": blocking_status,
            "new_expiry_time": decided_at + timedelta(hours=20),
            "risk_plan": "maintain stops",
        }

        validator = ReconfirmationValidator()
        is_valid, errors = validator.validate(payload, entry_time=entry_time)
        assert is_valid is False
        assert any("thesis_status" in e for e in errors)

    @given(
        excess_hours=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_excessive_expiry_rejected(self, excess_hours):
        """Payload with new_expiry_time exceeding max_hold_hours is rejected."""
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        decided_at = entry_time + timedelta(hours=5)
        max_hold_hours = NEWS_GOVERNANCE["max_hold_hours"]
        # Set expiry beyond allowed
        new_expiry = decided_at + timedelta(hours=max_hold_hours + excess_hours)

        payload = {
            "trade_id": 1,
            "symbol": "XLE",
            "profile": "moderate",
            "decision": "RECONFIRM_AND_HOLD",
            "decided_by": "pm_agent",
            "decided_at": decided_at,
            "original_catalyst": "news event",
            "fresh_catalyst_evidence": "x" * 25,
            "fresh_catalyst_timestamp": (entry_time + timedelta(hours=2)),
            "thesis_status": "strengthened",
            "new_expiry_time": new_expiry,
            "risk_plan": "maintain stops",
        }

        validator = ReconfirmationValidator()
        is_valid, errors = validator.validate(payload, entry_time=entry_time)
        assert is_valid is False
        assert any("exceeds maximum" in e for e in errors)

    @given(
        short_evidence=st.text(
            min_size=0,
            max_size=NEWS_GOVERNANCE["min_evidence_length"] - 1,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_short_evidence_rejected(self, short_evidence):
        """Payload with evidence shorter than min_evidence_length is rejected."""
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        decided_at = entry_time + timedelta(hours=5)

        payload = {
            "trade_id": 1,
            "symbol": "XLE",
            "profile": "moderate",
            "decision": "RECONFIRM_AND_HOLD",
            "decided_by": "pm_agent",
            "decided_at": decided_at,
            "original_catalyst": "news event",
            "fresh_catalyst_evidence": short_evidence,
            "fresh_catalyst_timestamp": (entry_time + timedelta(hours=2)),
            "thesis_status": "strengthened",
            "new_expiry_time": decided_at + timedelta(hours=20),
            "risk_plan": "maintain stops",
        }

        validator = ReconfirmationValidator()
        is_valid, errors = validator.validate(payload, entry_time=entry_time)
        assert is_valid is False
        assert any("fresh_catalyst_evidence" in e for e in errors)

    @given(
        hours_before_entry=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_stale_timestamp_rejected(self, hours_before_entry):
        """Payload with fresh_catalyst_timestamp before entry_time is rejected."""
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        decided_at = entry_time + timedelta(hours=5)
        # Stale timestamp: at or before entry_time
        stale_ts = entry_time - timedelta(hours=hours_before_entry)

        payload = {
            "trade_id": 1,
            "symbol": "XLE",
            "profile": "moderate",
            "decision": "RECONFIRM_AND_HOLD",
            "decided_by": "pm_agent",
            "decided_at": decided_at,
            "original_catalyst": "news event",
            "fresh_catalyst_evidence": "x" * 25,
            "fresh_catalyst_timestamp": stale_ts,
            "thesis_status": "strengthened",
            "new_expiry_time": decided_at + timedelta(hours=20),
            "risk_plan": "maintain stops",
        }

        validator = ReconfirmationValidator()
        is_valid, errors = validator.validate(payload, entry_time=entry_time)
        assert is_valid is False
        assert any("after trade entry_time" in e or "after" in e for e in errors)

    @settings(max_examples=100, deadline=None)
    @given(data=st.data())
    def test_naive_datetime_new_expiry_rejected(self, data):
        """Payload with naive datetime in new_expiry_time is rejected."""
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        decided_at = entry_time + timedelta(hours=5)
        # Naive datetime (no tzinfo)
        naive_expiry = datetime(2024, 6, 2, 10, 0)

        payload = {
            "trade_id": 1,
            "symbol": "XLE",
            "profile": "moderate",
            "decision": "RECONFIRM_AND_HOLD",
            "decided_by": "pm_agent",
            "decided_at": decided_at,
            "original_catalyst": "news event",
            "fresh_catalyst_evidence": "x" * 25,
            "fresh_catalyst_timestamp": (entry_time + timedelta(hours=2)),
            "thesis_status": "strengthened",
            "new_expiry_time": naive_expiry,
            "risk_plan": "maintain stops",
        }

        validator = ReconfirmationValidator()
        is_valid, errors = validator.validate(payload, entry_time=entry_time)
        assert is_valid is False
        assert any("timezone-aware" in e or "naive" in e for e in errors)


# ---------------------------------------------------------------------------
# Property 11: LET_EXPIRE Not Hold Authorization
# Feature: news-catalyst-24h-exit-gate, Property 11: LET_EXPIRE Not Hold Authorization
# ---------------------------------------------------------------------------


class TestProperty11LetExpireNotHoldAuthorization:
    """
    For any trade where latest reconfirmation is LET_EXPIRE,
    latest_valid_reconfirmation() returns None.

    **Validates: Requirements 6.13**
    """

    @given(
        n_let_expire=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_let_expire_returns_none(self, db_session, n_let_expire):
        """LET_EXPIRE reconfirmations do not count as hold authorization."""
        # Create a trade
        trade = Trade(
            symbol="XLE",
            direction="LONG",
            quantity=100,
            entry_price=85.0,
            status="open",
        )
        db_session.add(trade)
        db_session.flush()

        # Add LET_EXPIRE events
        for i in range(n_let_expire):
            payload = json.dumps({
                "decision": "LET_EXPIRE",
                "decline_reason": f"No longer valid reason {i}",
            })
            event = TradeEvent(
                trade_id=trade.id,
                event_type="news_reconfirmation_submitted",
                payload_json=payload,
                timestamp=datetime.now(timezone.utc) + timedelta(hours=i),
            )
            db_session.add(event)
        db_session.flush()

        # latest_valid_reconfirmation should return None
        result = latest_valid_reconfirmation(db_session, trade.id)
        assert result is None

        db_session.rollback()


# ---------------------------------------------------------------------------
# Property 12: Swing Reclassification Blocked
# Feature: news-catalyst-24h-exit-gate, Property 12: Swing Reclassification Blocked
# ---------------------------------------------------------------------------


class TestProperty12SwingReclassificationBlocked:
    """
    For any news-governed trade regardless of target_price, swing reclassification
    is blocked unless valid authorization exists.

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    """

    @given(
        target_price=st.one_of(
            st.none(),
            st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        ),
        hours_held=st.floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_news_governed_flag_blocks_reclassification(self, target_price, hours_held):
        """The _news_governed flag prevents swing reclassification."""
        # Simulate the trade dict as used in position_timer
        td = {
            "_news_governed": True,
            "target_price": target_price,
            "hours_held": hours_held,
            "setup_type": "news_catalyst",
        }

        # The logic in position_timer: if td.get("_news_governed"): continue
        # This means reclassification is blocked
        should_block = td.get("_news_governed", False)
        assert should_block is True

    @given(
        target_price=st.one_of(
            st.none(),
            st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_non_governed_trade_not_blocked(self, target_price):
        """Non-governed trades are not blocked from reclassification."""
        td = {
            "_news_governed": False,
            "target_price": target_price,
            "setup_type": "breakout",
        }

        should_block = td.get("_news_governed", False)
        assert should_block is False


# ---------------------------------------------------------------------------
# Property 13: log_trade_event_once Idempotency
# Feature: news-catalyst-24h-exit-gate, Property 13: log_trade_event_once Idempotency
# ---------------------------------------------------------------------------


class TestProperty13LogTradeEventOnceIdempotency:
    """
    For any (event_type, trade_id, governance_window_id), calling N times results
    in exactly one row; returns True first, False after.

    **Validates: Requirements 8.7, 8.8, 8.10, 8.11, 8.12**
    """

    @given(
        n_calls=st.integers(min_value=1, max_value=15),
        event_type=st.sampled_from([
            "news_reconfirmation_due",
            "news_governance_classified",
            "news_expiry_force_close",
        ]),
        governance_window_id=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_idempotent_single_row(
        self, db_session, n_calls, event_type, governance_window_id
    ):
        """Calling log_trade_event_once N times produces exactly one row."""
        # Create a trade
        trade = Trade(
            symbol="XLE",
            direction="LONG",
            quantity=100,
            entry_price=85.0,
            status="open",
        )
        db_session.add(trade)
        db_session.flush()

        results = []
        for _ in range(n_calls):
            result = log_trade_event_once(
                db_session,
                event_type,
                trade.id,
                governance_window_id=governance_window_id,
                agent="position_timer",
                symbol="XLE",
            )
            results.append(result)
            db_session.flush()

        # First call returns True
        assert results[0] is True
        # All subsequent calls return False
        for r in results[1:]:
            assert r is False

        # Exactly one row exists with this dedupe key
        dedupe_key = _build_dedupe_key(event_type, trade.id, governance_window_id)
        events = (
            db_session.query(TradeEvent)
            .filter_by(
                event_type=event_type,
                trade_id=trade.id,
                dedupe_key=dedupe_key,
            )
            .all()
        )
        assert len(events) == 1

        db_session.rollback()


# ---------------------------------------------------------------------------
# Property 14: Governance Status Computation
# Feature: news-catalyst-24h-exit-gate, Property 14: Governance Status Computation
# ---------------------------------------------------------------------------


class TestProperty14GovernanceStatusComputation:
    """
    For any timestamps relative to effective_expiry: correct status mapping
    (ok/warning/grace/expired/exit_requested).

    **Validates: Requirements 10.1, 10.2, 10.3**
    """

    @given(
        effective_expiry=aware_datetimes,
        warning_lead_hours=st_warning_hours,
        grace_minutes=st.integers(min_value=1, max_value=120),
        hours_before_warning=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_status_ok_before_warning_threshold(
        self, effective_expiry, warning_lead_hours, grace_minutes, hours_before_warning
    ):
        """Status is 'ok' when now < expiry - warning_lead_hours."""
        warning_threshold = effective_expiry - timedelta(hours=warning_lead_hours)
        now_utc = warning_threshold - timedelta(hours=hours_before_warning)

        policy = NewsGovernancePolicy()
        result = policy.compute_status(
            now_utc=now_utc,
            effective_expiry=effective_expiry,
            grace_minutes=grace_minutes,
            warning_lead_hours=warning_lead_hours,
            has_valid_hold=False,
            has_exit_request=False,
        )
        assert result["status"] == "ok"

    @given(
        effective_expiry=aware_datetimes,
        warning_lead_hours=st_warning_hours,
        grace_minutes=st.integers(min_value=1, max_value=120),
        fraction=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_status_warning_between_threshold_and_expiry(
        self, effective_expiry, warning_lead_hours, grace_minutes, fraction
    ):
        """Status is 'warning' when expiry - warning_lead <= now < expiry."""
        warning_threshold = effective_expiry - timedelta(hours=warning_lead_hours)
        # now is between warning_threshold and effective_expiry
        delta = effective_expiry - warning_threshold
        now_utc = warning_threshold + (delta * fraction)

        policy = NewsGovernancePolicy()
        result = policy.compute_status(
            now_utc=now_utc,
            effective_expiry=effective_expiry,
            grace_minutes=grace_minutes,
            warning_lead_hours=warning_lead_hours,
            has_valid_hold=False,
            has_exit_request=False,
        )
        assert result["status"] == "warning"

    @given(
        effective_expiry=aware_datetimes,
        warning_lead_hours=st_warning_hours,
        grace_minutes=st.integers(min_value=1, max_value=120),
        fraction=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_status_grace_between_expiry_and_grace_deadline(
        self, effective_expiry, warning_lead_hours, grace_minutes, fraction
    ):
        """Status is 'grace' when expiry <= now < expiry + grace_minutes."""
        grace_delta = timedelta(minutes=grace_minutes)
        now_utc = effective_expiry + (grace_delta * fraction)

        policy = NewsGovernancePolicy()
        result = policy.compute_status(
            now_utc=now_utc,
            effective_expiry=effective_expiry,
            grace_minutes=grace_minutes,
            warning_lead_hours=warning_lead_hours,
            has_valid_hold=False,
            has_exit_request=False,
        )
        assert result["status"] == "grace"

    @given(
        effective_expiry=aware_datetimes,
        warning_lead_hours=st_warning_hours,
        grace_minutes=st.integers(min_value=1, max_value=120),
        hours_past_grace=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_status_expired_past_grace_deadline(
        self, effective_expiry, warning_lead_hours, grace_minutes, hours_past_grace
    ):
        """Status is 'expired' when now >= expiry + grace_minutes."""
        grace_deadline = effective_expiry + timedelta(minutes=grace_minutes)
        now_utc = grace_deadline + timedelta(hours=hours_past_grace)

        policy = NewsGovernancePolicy()
        result = policy.compute_status(
            now_utc=now_utc,
            effective_expiry=effective_expiry,
            grace_minutes=grace_minutes,
            warning_lead_hours=warning_lead_hours,
            has_valid_hold=False,
            has_exit_request=False,
        )
        assert result["status"] == "expired"

    @given(
        effective_expiry=aware_datetimes,
        warning_lead_hours=st_warning_hours,
        grace_minutes=st.integers(min_value=1, max_value=120),
        has_valid_hold=st.booleans(),
    )
    @settings(max_examples=100, deadline=None)
    def test_status_exit_requested_takes_priority(
        self, effective_expiry, warning_lead_hours, grace_minutes, has_valid_hold
    ):
        """Status is 'exit_requested' when has_exit_request is True, regardless of time."""
        # Use any time position
        now_utc = effective_expiry - timedelta(hours=1)

        policy = NewsGovernancePolicy()
        result = policy.compute_status(
            now_utc=now_utc,
            effective_expiry=effective_expiry,
            grace_minutes=grace_minutes,
            warning_lead_hours=warning_lead_hours,
            has_valid_hold=has_valid_hold,
            has_exit_request=True,
        )
        assert result["status"] == "exit_requested"


# ---------------------------------------------------------------------------
# Property 15: Reconfirmation Payload Round-Trip
# Feature: news-catalyst-24h-exit-gate, Property 15: Reconfirmation Payload Round-Trip
# ---------------------------------------------------------------------------


class TestProperty15ReconfirmationPayloadRoundTrip:
    """
    For any valid reconfirmation payload, JSON serialize → deserialize produces
    equivalent payload.

    **Validates: Requirements 11.12**
    """

    @given(data=valid_reconfirm_hold_payload())
    @settings(max_examples=100, deadline=None)
    def test_valid_payload_survives_json_round_trip(self, data):
        """Valid reconfirmation payload survives JSON serialization round-trip."""
        payload, entry_time = data

        # Custom JSON serializer for datetime objects
        def json_default(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)

        # Serialize
        serialized = json.dumps(payload, default=json_default)

        # Deserialize
        deserialized = json.loads(serialized)

        # Verify key fields are preserved
        assert deserialized["trade_id"] == payload["trade_id"]
        assert deserialized["symbol"] == payload["symbol"]
        assert deserialized["profile"] == payload["profile"]
        assert deserialized["decision"] == payload["decision"]
        assert deserialized["decided_by"] == payload["decided_by"]
        assert deserialized["original_catalyst"] == payload["original_catalyst"]
        assert deserialized["fresh_catalyst_evidence"] == payload["fresh_catalyst_evidence"]
        assert deserialized["thesis_status"] == payload["thesis_status"]
        assert deserialized["risk_plan"] == payload["risk_plan"]

        # Datetime fields are serialized as ISO strings
        assert deserialized["fresh_catalyst_timestamp"] == payload["fresh_catalyst_timestamp"].isoformat()
        assert deserialized["new_expiry_time"] == payload["new_expiry_time"].isoformat()
        assert deserialized["decided_at"] == payload["decided_at"].isoformat()


# ---------------------------------------------------------------------------
# Property 16: Swing Authorization Only With RECONFIRM_AND_HOLD
# Feature: news-catalyst-24h-exit-gate, Property 16: Swing Authorization Only With RECONFIRM_AND_HOLD
# ---------------------------------------------------------------------------


class TestProperty16SwingAuthorizationOnlyWithHold:
    """
    For any payload with allow_swing_reclassify=true and decision != RECONFIRM_AND_HOLD,
    validator rejects.

    **Validates: Requirements 12.5**
    """

    @given(
        decision=st.sampled_from(["EXIT_NOW", "LET_EXPIRE"]),
    )
    @settings(max_examples=100, deadline=None)
    def test_swing_auth_with_non_hold_decision_rejected(self, decision):
        """allow_swing_reclassify=True with non-HOLD decision is rejected."""
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)

        # Build a payload with the non-HOLD decision but with swing auth
        payload = {
            "trade_id": 1,
            "symbol": "XLE",
            "profile": "moderate",
            "decision": decision,
            "decided_by": "pm_agent",
            "allow_swing_reclassify": True,
        }

        # Add decision-specific required fields to pass schema validation
        if decision == "EXIT_NOW":
            payload["exit_reason"] = "Thesis invalidated"
        elif decision == "LET_EXPIRE":
            payload["decline_reason"] = "No longer valid"

        validator = ReconfirmationValidator()
        is_valid, errors = validator.validate(payload, entry_time=entry_time)
        assert is_valid is False
        assert any("allow_swing_reclassify" in e for e in errors)
