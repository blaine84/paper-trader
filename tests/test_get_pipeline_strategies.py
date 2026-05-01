"""
Tests for get_pipeline_strategies() in utils/strategy_store.py.
Validates Requirements 9.1, 9.2 — querying strategies by pipeline stage.
"""

import json
from sqlalchemy import create_engine
from db.schema import Base, DynamicStrategy, get_session
from utils.strategy_store import get_pipeline_strategies


def _make_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _add_strategy(engine, key, status, pipeline_stage=None):
    db = get_session(engine)
    strat = DynamicStrategy(
        key=key,
        name=f"Test {key}",
        description=f"Description for {key}",
        status=status,
        pipeline_stage=pipeline_stage or status,
        ideal_conditions=json.dumps({}),
        failure_conditions=json.dumps([]),
        execution_notes=json.dumps([]),
    )
    db.add(strat)
    db.commit()
    db.close()
    return strat


class TestGetPipelineStrategies:
    """Tests for get_pipeline_strategies()."""

    def test_returns_empty_when_no_strategies(self):
        engine = _make_engine()
        result = get_pipeline_strategies(engine)
        assert result == []

    def test_returns_all_pipeline_stages_when_stage_is_none(self):
        engine = _make_engine()
        _add_strategy(engine, "s1", "backtest")
        _add_strategy(engine, "s2", "paper_trade")
        _add_strategy(engine, "s3", "live_50")
        _add_strategy(engine, "s4", "live_100")

        result = get_pipeline_strategies(engine)
        keys = {s.key for s in result}
        assert keys == {"s1", "s2", "s3", "s4"}

    def test_excludes_non_pipeline_statuses(self):
        engine = _make_engine()
        _add_strategy(engine, "s_active", "active")
        _add_strategy(engine, "s_retired", "retired")
        _add_strategy(engine, "s_probation", "probation")
        _add_strategy(engine, "s_failed", "backtest_failed")
        _add_strategy(engine, "s_backtest", "backtest")

        result = get_pipeline_strategies(engine)
        keys = {s.key for s in result}
        assert keys == {"s_backtest"}

    def test_filter_by_specific_stage(self):
        engine = _make_engine()
        _add_strategy(engine, "s1", "backtest")
        _add_strategy(engine, "s2", "paper_trade")
        _add_strategy(engine, "s3", "live_50")

        result = get_pipeline_strategies(engine, stage="paper_trade")
        keys = {s.key for s in result}
        assert keys == {"s2"}

    def test_filter_by_stage_returns_empty_when_no_match(self):
        engine = _make_engine()
        _add_strategy(engine, "s1", "backtest")

        result = get_pipeline_strategies(engine, stage="live_100")
        assert result == []

    def test_returns_dynamicstrategy_objects(self):
        engine = _make_engine()
        _add_strategy(engine, "s1", "backtest")

        result = get_pipeline_strategies(engine)
        assert len(result) == 1
        assert isinstance(result[0], DynamicStrategy)
        assert result[0].key == "s1"
        assert result[0].status == "backtest"

    def test_mixed_statuses_only_returns_pipeline(self):
        engine = _make_engine()
        _add_strategy(engine, "s_bt", "backtest")
        _add_strategy(engine, "s_pt", "paper_trade")
        _add_strategy(engine, "s_active", "active")
        _add_strategy(engine, "s_retired", "retired")
        _add_strategy(engine, "s_l50", "live_50")
        _add_strategy(engine, "s_l100", "live_100")
        _add_strategy(engine, "s_failed", "backtest_failed")

        result = get_pipeline_strategies(engine)
        keys = {s.key for s in result}
        assert keys == {"s_bt", "s_pt", "s_l50", "s_l100"}
