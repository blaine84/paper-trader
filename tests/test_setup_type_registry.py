import json

from sqlalchemy import create_engine

from db.schema import Base, DynamicStrategy, get_session
from utils.strategy_store import get_all_setup_types


def _make_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _add_strategy(engine, key, status):
    db = get_session(engine)
    db.add(DynamicStrategy(
        key=key,
        name=f"Test {key}",
        description=f"Description for {key}",
        status=status,
        pipeline_stage=status,
        ideal_conditions=json.dumps({}),
        failure_conditions=json.dumps([]),
        execution_notes=json.dumps([]),
    ))
    db.commit()
    db.close()


def test_setup_registry_includes_strategy_aliases():
    engine = _make_engine()

    setup_types = get_all_setup_types(engine)

    assert "technical_breakout" in setup_types
    assert "news_breakout" in setup_types


def test_setup_registry_includes_live_dynamic_strategies_only():
    engine = _make_engine()
    _add_strategy(engine, "live_50_setup", "live_50")
    _add_strategy(engine, "live_100_setup", "live_100")
    _add_strategy(engine, "active_setup", "active")
    _add_strategy(engine, "backtest_setup", "backtest")

    setup_types = get_all_setup_types(engine)

    assert "live_50_setup" in setup_types
    assert "live_100_setup" in setup_types
    assert "active_setup" in setup_types
    assert "backtest_setup" not in setup_types
