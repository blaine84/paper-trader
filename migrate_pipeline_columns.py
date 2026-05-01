"""
One-time migration: add pipeline tracking columns to the dynamic_strategies table.
SQLAlchemy create_all doesn't ALTER existing tables, so we do it manually.

Run once: python migrate_pipeline_columns.py
"""
import sqlite3
import sys

DB_PATH = "db/paper_trader.db"


def migrate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check which columns already exist
    cursor.execute("PRAGMA table_info(dynamic_strategies)")
    existing = {row[1] for row in cursor.fetchall()}

    added = []
    for col, col_type in [
        ("pipeline_stage", "VARCHAR(16)"),
        ("backtest_report_id", "VARCHAR(128)"),
        ("paper_trade_start_date", "DATETIME"),
        ("live_50_start_date", "DATETIME"),
        ("live_100_start_date", "DATETIME"),
        ("failure_stage", "VARCHAR(16)"),
        ("failure_reason", "TEXT"),
    ]:
        if col not in existing:
            cursor.execute(f"ALTER TABLE dynamic_strategies ADD COLUMN {col} {col_type}")
            added.append(col)

    conn.commit()
    conn.close()

    if added:
        print(f"Added columns: {', '.join(added)}")
    else:
        print("All columns already exist. Nothing to do.")


if __name__ == "__main__":
    migrate()
