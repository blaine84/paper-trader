"""Tests for the pending-limit-order dashboard endpoints in web/app.py.

The requirement these exist to satisfy: a resting order must be presentable as
something distinct from a rejected decision, and its pre-trade lifecycle events
must be reachable at all — which /api/trade-events cannot do, because it requires
a trade_id that pre-fill events do not have.

Requirements: 8.8, 10.5, 11.1-11.10, 12.1-12.6
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import web.app as web_app
from db.schema import get_session, init_pending_order_schema
from utils.pending_order_registry import (
    OrderState,
    PendingOrder,
    PendingOrderRegistry,
)
from utils.trade_events import log_trade_event

LIMIT = 593.87
STOP = 585.00
TARGET = 620.00


@pytest.fixture
def app_engine(tmp_path):
    """Point web.app at a throwaway database with the pending-order schema."""
    from db.schema import init_db

    db_path = tmp_path / "dash.db"
    engine = init_db(str(db_path))
    init_pending_order_schema(engine)

    original = web_app.engine
    web_app.engine = engine
    try:
        yield engine
    finally:
        web_app.engine = original


@pytest.fixture
def client(app_engine):
    web_app.app.config["TESTING"] = True
    with web_app.app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def no_quotes():
    """Never hit a real provider from a dashboard test."""
    quote = MagicMock()
    quote.get_quote.return_value = {"price": 601.24}
    with patch("web.app.FinnhubClient", return_value=quote):
        yield quote


def make_order(engine, **overrides) -> PendingOrder:
    now = datetime.now(timezone.utc)
    defaults = dict(
        order_id=str(uuid.uuid4()),
        profile_id="moderate",
        symbol="META",
        side="BUY",
        setup_type="technical_breakout",
        limit_price=LIMIT,
        stop_price=STOP,
        target_price=TARGET,
        risk_reward=2.9,
        fresh_price_at_creation=601.24,
        runaway_pct_at_creation=0.0124,
        intended_quantity=10,
        pm_rationale="waiting for the pullback",
        created_at=now - timedelta(minutes=10),
        expires_at=now + timedelta(hours=1),
    )
    defaults.update(overrides)
    order = PendingOrder(**defaults)
    registry = PendingOrderRegistry(engine)
    registry.create_order(order)
    return registry.get_order(order.order_id)


# ---------------------------------------------------------------------------
# /api/pending_orders
# ---------------------------------------------------------------------------


def test_endpoint_returns_an_active_order(client, app_engine):
    order = make_order(app_engine)

    payload = client.get("/api/pending_orders").get_json()

    assert payload["available"] is True
    assert len(payload["orders"]) == 1
    row = payload["orders"][0]
    assert row["order_id"] == order.order_id
    assert row["state"] == "pending"
    assert row["is_active"] is True


def test_response_carries_every_required_field(client, app_engine):
    """Requirement 11.2 spells out the columns the view must show."""
    make_order(app_engine)
    row = client.get("/api/pending_orders").get_json()["orders"][0]

    for field in (
        "symbol", "profile", "side", "limit_price", "current_price",
        "distance_to_limit", "expires_at", "seconds_remaining", "state",
        "reason", "rationale",
    ):
        assert field in row, f"missing {field}"


def test_distance_to_limit_is_signed_toward_the_limit(client, app_engine):
    """Positive means the market has not come back to the limit yet."""
    make_order(app_engine)
    row = client.get("/api/pending_orders").get_json()["orders"][0]

    # Quote is 601.24, limit is 593.87 -> BUY still needs a 7.37 pullback.
    assert row["current_price"] == pytest.approx(601.24)
    assert row["distance_to_limit"] == pytest.approx(7.37, abs=0.01)
    assert row["distance_to_limit_pct"] > 0


def test_short_distance_is_inverted(client, app_engine):
    make_order(
        app_engine, side="SHORT", symbol="AMD",
        limit_price=610.0, stop_price=620.0, target_price=580.0,
    )
    row = client.get("/api/pending_orders").get_json()["orders"][0]

    # Quote 601.24 against a SHORT limit of 610 -> needs an 8.76 rally.
    assert row["distance_to_limit"] == pytest.approx(8.76, abs=0.01)


def test_seconds_remaining_is_never_negative(client, app_engine):
    make_order(
        app_engine,
        created_at=datetime.now(timezone.utc) - timedelta(hours=3),
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    row = client.get("/api/pending_orders").get_json()["orders"][0]
    assert row["seconds_remaining"] == 0


def test_all_six_outcome_categories_are_distinguishable(client, app_engine):
    """Requirement 11.3 — pending, filling, filled, expired, canceled, rejected."""
    registry = PendingOrderRegistry(app_engine)

    make_order(app_engine, symbol="AAA")

    filling = make_order(app_engine, symbol="BBB")
    registry.claim_for_fill(filling.order_id)

    filled = make_order(app_engine, symbol="CCC")
    registry.claim_for_fill(filled.order_id)
    registry.mark_filled(
        filled.order_id, fill_price=LIMIT, fill_policy="limit_price",
        fill_bar_ts=datetime.now(timezone.utc), trade_id=42,
    )

    expired = make_order(app_engine, symbol="DDD")
    registry.mark_expired(expired.order_id)

    canceled = make_order(app_engine, symbol="EEE")
    registry.mark_canceled(canceled.order_id, "signal_flipped")

    rejected = make_order(app_engine, symbol="FFF")
    registry.claim_for_fill(rejected.order_id)
    registry.mark_rejected(rejected.order_id, "setup_quality_gate")

    payload = client.get("/api/pending_orders").get_json()
    counts = payload["counts"]

    assert counts["pending"] == 1
    assert counts["filling"] == 1
    assert counts["filled"] == 1
    assert counts["expired"] == 1
    assert counts["canceled"] == 1
    assert counts["rejected"] == 1
    assert payload["active_count"] == 2  # pending + filling


def test_include_terminal_false_returns_only_active(client, app_engine):
    registry = PendingOrderRegistry(app_engine)
    active = make_order(app_engine, symbol="AAA")
    done = make_order(app_engine, symbol="BBB")
    registry.mark_canceled(done.order_id, "signal_flipped")

    payload = client.get(
        "/api/pending_orders?include_terminal=false"
    ).get_json()

    ids = {o["order_id"] for o in payload["orders"]}
    assert ids == {active.order_id}


def test_profile_filter_narrows_results(client, app_engine):
    make_order(app_engine, symbol="AAA", profile_id="moderate")
    make_order(app_engine, symbol="BBB", profile_id="aggressive")

    payload = client.get("/api/pending_orders?profile=aggressive").get_json()

    assert len(payload["orders"]) == 1
    assert payload["orders"][0]["profile"] == "aggressive"


def test_terminal_order_exposes_its_reason(client, app_engine):
    registry = PendingOrderRegistry(app_engine)
    order = make_order(app_engine)
    registry.mark_canceled(order.order_id, "gap_through")

    row = client.get("/api/pending_orders").get_json()["orders"][0]
    assert row["state"] == "canceled"
    assert row["reason"] == "gap_through"
    assert row["is_active"] is False


def test_filled_order_exposes_fill_detail_and_trade_linkage(client, app_engine):
    registry = PendingOrderRegistry(app_engine)
    order = make_order(app_engine)
    registry.claim_for_fill(order.order_id)
    bar_ts = datetime.now(timezone.utc)
    registry.mark_filled(
        order.order_id, fill_price=LIMIT, fill_policy="limit_price",
        fill_bar_ts=bar_ts, trade_id=99,
    )

    row = client.get("/api/pending_orders").get_json()["orders"][0]
    assert row["fill_price"] == pytest.approx(LIMIT)
    assert row["fill_policy"] == "limit_price"
    assert row["trade_id"] == 99
    assert row["filled_at"] is not None


def test_orders_are_newest_first(client, app_engine):
    now = datetime.now(timezone.utc)
    make_order(app_engine, symbol="OLD", created_at=now - timedelta(hours=2))
    make_order(app_engine, symbol="NEW", created_at=now - timedelta(minutes=5))

    symbols = [o["symbol"] for o in client.get("/api/pending_orders").get_json()["orders"]]
    assert symbols == ["NEW", "OLD"]


def test_events_are_embedded_inline(client, app_engine):
    """Requirement 11.6 — pre-trade events must be reachable here."""
    registry = PendingOrderRegistry(app_engine)
    order = make_order(app_engine)
    registry.mark_canceled(order.order_id, "signal_flipped")

    row = client.get("/api/pending_orders").get_json()["orders"][0]
    types = [e["event_type"] for e in row["recent_events"]]
    assert "state_pending" in types
    assert "state_canceled" in types


def test_empty_registry_returns_an_empty_list(client, app_engine):
    payload = client.get("/api/pending_orders").get_json()
    assert payload["available"] is True
    assert payload["orders"] == []
    assert payload["counts"] == {}


def test_quote_failure_degrades_to_null_price(client, app_engine):
    make_order(app_engine)

    with patch("web.app.FinnhubClient", side_effect=RuntimeError("no key")):
        row = client.get("/api/pending_orders").get_json()["orders"][0]

    assert row["current_price"] is None
    assert row["distance_to_limit"] is None
    assert row["state"] == "pending"


def test_quotes_are_fetched_once_per_symbol(client, app_engine, no_quotes):
    for setup in ("technical_breakout", "vwap_reclaim", "news_breakout"):
        make_order(app_engine, setup_type=setup)

    client.get("/api/pending_orders")

    assert no_quotes.get_quote.call_count == 1


def test_missing_table_degrades_gracefully(client):
    """Requirement 11.10 — the front end needs no conditional."""
    with patch("web.app._pending_orders_available", return_value=False):
        payload = client.get("/api/pending_orders").get_json()

    assert payload["available"] is False
    assert payload["orders"] == []


def test_registry_failure_degrades_gracefully(client, app_engine):
    make_order(app_engine)

    with patch.object(
        PendingOrderRegistry, "get_orders_for_profile",
        side_effect=RuntimeError("locked"),
    ):
        response = client.get("/api/pending_orders")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["available"] is False
    assert payload["orders"] == []


# ---------------------------------------------------------------------------
# /api/pending_orders/<order_id>/events
# ---------------------------------------------------------------------------


def test_order_scoped_events_endpoint_returns_full_history(client, app_engine):
    registry = PendingOrderRegistry(app_engine)
    order = make_order(app_engine)
    registry.claim_for_fill(order.order_id)
    registry.release_claim(order.order_id, reason="stale_fill_bar")
    registry.mark_canceled(order.order_id, "signal_flipped")

    payload = client.get(
        f"/api/pending_orders/{order.order_id}/events"
    ).get_json()

    types = [e["event_type"] for e in payload["events"]]
    assert types == [
        "state_pending", "state_filling", "state_pending", "state_canceled"
    ]
    assert payload["order_id"] == order.order_id
    assert payload["state"] == "canceled"


def test_events_carry_state_transitions(client, app_engine):
    registry = PendingOrderRegistry(app_engine)
    order = make_order(app_engine)
    registry.claim_for_fill(order.order_id)

    events = client.get(
        f"/api/pending_orders/{order.order_id}/events"
    ).get_json()["events"]

    claim = events[1]
    assert claim["from_state"] == "pending"
    assert claim["to_state"] == "filling"


def test_unknown_order_id_returns_404(client, app_engine):
    response = client.get("/api/pending_orders/does-not-exist/events")
    assert response.status_code == 404


def test_events_endpoint_404s_when_uninitialized(client):
    with patch("web.app._pending_orders_available", return_value=False):
        response = client.get("/api/pending_orders/whatever/events")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# /api/trade-events coherence
# ---------------------------------------------------------------------------


def test_trade_events_now_includes_pending_order_types(client, app_engine):
    """A filled order's event carries a trade_id, so it stays visible per-trade."""
    session = get_session(app_engine)
    try:
        log_trade_event(
            session,
            "pending_order_filled",
            trade_id=77,
            agent="pending_order_filler",
            symbol="META",
            profile="moderate",
            price=LIMIT,
            message="filled at the limit",
            payload={"order_id": "abc", "fill_price": LIMIT},
        )
        session.commit()
    finally:
        session.close()

    payload = client.get("/api/trade-events?trade_id=77").get_json()

    assert len(payload) == 1
    assert payload[0]["event_type"] == "pending_order_filled"


def test_trade_events_still_requires_a_trade_id(client, app_engine):
    """Which is exactly why pre-trade events need the order-scoped endpoint."""
    response = client.get("/api/trade-events")
    assert response.status_code == 400


def test_pre_trade_events_are_unreachable_via_trade_events(client, app_engine):
    """Documents the constraint that drove the separate endpoint.

    pending_order_created has trade_id=None, so no trade_id query can ever
    return it.
    """
    session = get_session(app_engine)
    try:
        log_trade_event(
            session,
            "pending_order_created",
            agent="pending_order_creation",
            symbol="META",
            profile="moderate",
            price=LIMIT,
            message="resting",
            payload={"order_id": "abc"},
        )
        session.commit()
    finally:
        session.close()

    for trade_id in (1, 77, 999):
        payload = client.get(f"/api/trade-events?trade_id={trade_id}").get_json()
        assert payload == []


# ---------------------------------------------------------------------------
# /api/pending_orders/summary
# ---------------------------------------------------------------------------


def test_summary_reports_outcomes_by_state(client, app_engine):
    registry = PendingOrderRegistry(app_engine)
    make_order(app_engine, symbol="AAA")
    expired = make_order(app_engine, symbol="BBB")
    registry.mark_expired(expired.order_id)

    payload = client.get("/api/pending_orders/summary").get_json()

    assert payload["available"] is True
    assert payload["outcomes"]["pending"] == 1
    assert payload["outcomes"]["expired"] == 1


def test_summary_computes_fill_rate_per_setup_and_profile(client, app_engine):
    registry = PendingOrderRegistry(app_engine)

    filled = make_order(app_engine, symbol="AAA")
    registry.claim_for_fill(filled.order_id)
    registry.mark_filled(
        filled.order_id, fill_price=LIMIT, fill_policy="limit_price",
        fill_bar_ts=datetime.now(timezone.utc),
    )
    expired = make_order(app_engine, symbol="BBB")
    registry.mark_expired(expired.order_id)

    payload = client.get("/api/pending_orders/summary").get_json()
    setup = payload["fill_rate_by_setup"]["technical_breakout"]

    assert setup["filled"] == 1
    assert setup["resolved"] == 2
    assert setup["fill_rate"] == pytest.approx(0.5)

    profile = payload["fill_rate_by_profile"]["moderate"]
    assert profile["fill_rate"] == pytest.approx(0.5)


def test_summary_fill_rate_is_none_with_nothing_resolved(client, app_engine):
    make_order(app_engine)
    payload = client.get("/api/pending_orders/summary").get_json()
    assert payload["fill_rate_by_setup"]["technical_breakout"]["fill_rate"] is None


def test_summary_counts_decline_reasons(client, app_engine):
    """Both gating measurements must be countable from the event stream alone."""
    session = get_session(app_engine)
    try:
        for reason in (
            "target_already_exceeded",
            "target_already_exceeded",
            "repaired_before_check",
        ):
            log_trade_event(
                session,
                "pending_order_declined",
                agent="pending_order_creation",
                symbol="META",
                profile="moderate",
                message=f"declined - {reason}",
                payload={"reason": reason},
            )
        session.commit()
    finally:
        session.close()

    payload = client.get("/api/pending_orders/summary").get_json()

    assert payload["declines"]["target_already_exceeded"] == 2
    assert payload["declines"]["repaired_before_check"] == 1


def test_summary_ranks_near_misses_by_distance(client, app_engine):
    """Requirement 10.5 / 12.2 — a one-cent miss must be distinguishable."""
    session = get_session(app_engine)
    try:
        for order_id, distance in (("far", 12.5), ("close", 0.01), ("mid", 3.0)):
            log_trade_event(
                session,
                "pending_order_expired",
                agent="pending_order_monitor",
                symbol="META",
                profile="moderate",
                message="expired",
                payload={
                    "order_id": order_id,
                    "reason": "window_elapsed",
                    "side": "BUY",
                    "setup_type": "technical_breakout",
                    "limit_price": LIMIT,
                    "closest_approach_price": LIMIT + distance,
                    "closest_approach_distance": distance,
                },
            )
        session.commit()
    finally:
        session.close()

    payload = client.get("/api/pending_orders/summary").get_json()
    ids = [n["order_id"] for n in payload["near_misses"]]

    assert ids == ["close", "mid", "far"]
    assert payload["near_misses"][0]["closest_approach_distance"] == pytest.approx(0.01)


def test_summary_degrades_when_uninitialized(client):
    with patch("web.app._pending_orders_available", return_value=False):
        payload = client.get("/api/pending_orders/summary").get_json()

    assert payload["available"] is False
    assert payload["outcomes"] == {}


def test_summary_window_is_configurable(client, app_engine):
    payload = client.get("/api/pending_orders/summary?days=1").get_json()
    assert payload["window_days"] == 1


# ---------------------------------------------------------------------------
# Isolation from position and equity reporting (Requirement 8.8)
# ---------------------------------------------------------------------------


def test_pending_orders_do_not_appear_in_positions(client, app_engine):
    """A resting order is committed intent, not exposure."""
    make_order(app_engine)

    payload = client.get("/api/positions").get_json()
    rows = payload if isinstance(payload, list) else payload.get("positions", [])

    assert rows == [] or all(r.get("symbol") != "META" for r in rows)


def test_api_decisions_is_untouched_by_pending_orders(client, app_engine):
    """Requirement 11.4 — the decision log must not describe a resting order."""
    make_order(app_engine)

    payload = client.get("/api/decisions").get_json()
    blob = json.dumps(payload)

    assert "pending_order" not in blob
    assert "rejected" not in blob.lower() or "pending" not in blob.lower()


# ---------------------------------------------------------------------------
# would_have_filled_after_expiry (Requirement 12.2)
# ---------------------------------------------------------------------------


def _expire_order(engine, order, minutes_ago: int = 20):
    """Put an order into EXPIRED with its window closed `minutes_ago`."""
    from sqlalchemy import text

    registry = PendingOrderRegistry(engine)
    now = datetime.now(timezone.utc)
    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE pending_orders SET created_at = :c, expires_at = :e "
                "WHERE order_id = :oid"
            ),
            {
                "c": (now - timedelta(hours=2)).isoformat(),
                "e": (now - timedelta(minutes=minutes_ago)).isoformat(),
                "oid": order.order_id,
            },
        )
        conn.commit()
    registry.mark_expired(order.order_id)


def _post_expiry_bars(*, lows, minutes_ago_start):
    base = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago_start)
    return {
        "symbol": "META",
        "resolution": "1",
        "timestamps": [
            int((base + timedelta(minutes=i)).timestamp()) for i in range(len(lows))
        ],
        "open": [low + 3.0 for low in lows],
        "high": [low + 4.0 for low in lows],
        "low": list(lows),
        "close": [low + 1.0 for low in lows],
    }


def test_post_expiry_report_is_off_by_default(client, app_engine):
    """It costs a provider call per symbol, so the common poll stays cheap."""
    order = make_order(app_engine)
    _expire_order(app_engine, order)

    payload = client.get("/api/pending_orders/summary").get_json()
    assert "post_expiry" not in payload


def test_post_expiry_detects_a_limit_hit_just_after_expiry(client, app_engine):
    """The signal that says the active window was too short."""
    order = make_order(app_engine)
    _expire_order(app_engine, order, minutes_ago=20)

    # Bars from 15 and 14 minutes ago — AFTER the 20-minutes-ago expiry — with
    # the second dipping through the 593.87 limit.
    client_mock = MagicMock()
    client_mock.get_candles.return_value = _post_expiry_bars(
        lows=[599.0, 590.0], minutes_ago_start=15
    )

    with patch("utils.finnhub_client.FinnhubClient", return_value=client_mock):
        payload = client.get(
            "/api/pending_orders/summary?check_post_expiry=true"
        ).get_json()

    report = payload["post_expiry"]
    assert report["checked"] == 1
    assert report["would_have_filled"] == 1
    assert report["would_have_filled_rate"] == pytest.approx(1.0)

    hit = report["orders"][0]
    assert hit["order_id"] == order.order_id
    assert hit["minutes_after_expiry"] > 0


def test_post_expiry_ignores_crossings_before_expiry(client, app_engine):
    """Only bars strictly after expires_at count.

    A crossing inside the active window would have filled the order normally, so
    counting it here would report a phantom missed opportunity.
    """
    order = make_order(app_engine)
    _expire_order(app_engine, order, minutes_ago=10)

    # Bars from 40 and 39 minutes ago — inside the original window, not after it.
    client_mock = MagicMock()
    client_mock.get_candles.return_value = _post_expiry_bars(
        lows=[599.0, 590.0], minutes_ago_start=40
    )

    with patch("utils.finnhub_client.FinnhubClient", return_value=client_mock):
        payload = client.get(
            "/api/pending_orders/summary?check_post_expiry=true"
        ).get_json()

    assert payload["post_expiry"]["would_have_filled"] == 0


def test_post_expiry_reports_zero_when_price_never_returned(client, app_engine):
    order = make_order(app_engine)
    _expire_order(app_engine, order, minutes_ago=20)

    client_mock = MagicMock()
    client_mock.get_candles.return_value = _post_expiry_bars(
        lows=[610.0, 615.0], minutes_ago_start=15
    )

    with patch("utils.finnhub_client.FinnhubClient", return_value=client_mock):
        payload = client.get(
            "/api/pending_orders/summary?check_post_expiry=true"
        ).get_json()

    report = payload["post_expiry"]
    assert report["checked"] == 1
    assert report["would_have_filled"] == 0
    assert report["orders"] == []


def test_post_expiry_only_considers_expired_orders(client, app_engine):
    """A resting order has not missed anything yet."""
    make_order(app_engine, symbol="STILLRESTING")

    client_mock = MagicMock()
    client_mock.get_candles.return_value = _post_expiry_bars(
        lows=[500.0], minutes_ago_start=5
    )

    with patch("utils.finnhub_client.FinnhubClient", return_value=client_mock):
        payload = client.get(
            "/api/pending_orders/summary?check_post_expiry=true"
        ).get_json()

    assert payload["post_expiry"]["checked"] == 0


def test_post_expiry_survives_a_provider_failure(client, app_engine):
    order = make_order(app_engine)
    _expire_order(app_engine, order)

    with patch(
        "utils.finnhub_client.FinnhubClient", side_effect=RuntimeError("no key")
    ):
        response = client.get(
            "/api/pending_orders/summary?check_post_expiry=true"
        )

    assert response.status_code == 200
    assert response.get_json()["post_expiry"]["would_have_filled"] == 0
