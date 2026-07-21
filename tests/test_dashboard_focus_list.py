import json
from datetime import datetime, timedelta

from sqlalchemy import create_engine

from db.schema import AgentMemory, Base, get_session
from web.app import get_active_focus


def _make_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _add_focus_record(engine, *, key_date: str, payload_date: str, selected: list[str], context: str = "analyst_refresh"):
    db = get_session(engine)
    db.add(AgentMemory(
        agent="scout",
        symbol=None,
        key=f"focus_list:{key_date}:{context}",
        timestamp=datetime.utcnow(),
        value=json.dumps({
            "date": payload_date,
            "context": context,
            "max_symbols": 3,
            "selected": selected,
            "generated_at": datetime.utcnow().isoformat(),
        }),
    ))
    db.commit()
    db.close()


def test_get_active_focus_returns_todays_focus_symbols():
    engine = _make_engine()
    today = datetime.utcnow().date().isoformat()
    _add_focus_record(engine, key_date=today, payload_date=today, selected=["mstr", "AMD", "MU"])

    db = get_session(engine)
    try:
        result = get_active_focus(db)
    finally:
        db.close()

    assert result["symbols"] == ["MSTR", "AMD", "MU"]
    assert result["context"] == "analyst_refresh"
    assert result["max_symbols"] == 3


def test_get_active_focus_ignores_stale_focus_rows():
    engine = _make_engine()
    yesterday = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
    _add_focus_record(engine, key_date=yesterday, payload_date=yesterday, selected=["TSLA"])

    db = get_session(engine)
    try:
        result = get_active_focus(db)
    finally:
        db.close()

    assert result["symbols"] == []
