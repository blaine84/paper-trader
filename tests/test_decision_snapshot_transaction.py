from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from db.replay_schema import init_replay_db
from db.schema import Base
from utils.decision_snapshot import build_and_persist_snapshot
from utils.trade_events import log_trade_event


def _make_engine(tmp_path):
    db_path = tmp_path / "snapshot_txn.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"timeout": 1},
    )
    Base.metadata.create_all(engine)
    init_replay_db(engine)
    return engine


def test_pm_snapshot_persists_inside_active_session_transaction(tmp_path):
    engine = _make_engine(tmp_path)
    Session = sessionmaker(bind=engine)
    db = Session()

    log_trade_event(
        db,
        "entry_requested",
        agent="pm_moderate",
        symbol="QQQ",
        profile="moderate",
        price=733.30,
        message="candidate selected",
    )
    db.flush()

    build_and_persist_snapshot(
        engine,
        session=db,
        candidate_lineage_id="lineage-001",
        decision={
            "symbol": "QQQ",
            "action": "BUY",
            "quantity": 3,
            "entry_price": 733.30,
            "stop": 726.58,
            "target": 736.23,
            "setup_type": "sector_rotation",
        },
        signal={
            "symbol": "QQQ",
            "setup_type": "sector_rotation",
            "signal_strength": 8.0,
            "confidence": "high",
        },
        profile_id="moderate",
        account_equity=100000,
        available_cash=50000,
        open_positions=[],
        gate_config={"min_reward_to_risk": 2.0},
        feature_flags={"PM_CANDIDATE_MODE": True},
        policy_version_id="test-policy",
    )

    # The snapshot and the event are in the same uncommitted transaction.
    assert db.execute(text("SELECT COUNT(*) FROM decision_snapshots")).scalar_one() == 1
    assert db.execute(text("SELECT COUNT(*) FROM trade_events")).scalar_one() == 1

    db.rollback()
    db.close()

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM decision_snapshots")).scalar_one() == 0
        assert conn.execute(text("SELECT COUNT(*) FROM trade_events")).scalar_one() == 0


def test_legacy_snapshot_persistence_still_commits_without_session(tmp_path):
    engine = _make_engine(tmp_path)

    build_and_persist_snapshot(
        engine,
        candidate_lineage_id="lineage-legacy",
        decision={
            "symbol": "XLF",
            "action": "BUY",
            "quantity": 10,
            "entry_price": 54.68,
            "stop": 54.16,
            "target": 54.90,
        },
        signal={"symbol": "XLF", "setup_type": "sector_rotation"},
        profile_id="aggressive",
        account_equity=100000,
        available_cash=50000,
        policy_version_id="legacy-test",
    )

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM decision_snapshots")).scalar_one() == 1
