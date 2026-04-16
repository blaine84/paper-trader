"""
One-time migration: add thesis-anchored exits columns to the trades table.
SQLAlchemy create_all doesn't ALTER existing tables, so we do it manually.

Run once: python migrate_thesis_columns.py
"""
import sqlite3
import sys

DB_PATH = "db/paper_trader.db"


def migrate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check which columns already exist
    cursor.execute("PRAGMA table_info(trades)")
    existing = {row[1] for row in cursor.fetchall()}

    added = []
    for col, col_type in [
        ("thesis", "TEXT"),
        ("setup_type", "VARCHAR(64)"),
        ("invalidators", "TEXT"),
    ]:
        if col not in existing:
            cursor.execute(f"ALTER TABLE trades ADD COLUMN {col} {col_type}")
            added.append(col)

    conn.commit()
    conn.close()

    if added:
        print(f"Added columns: {', '.join(added)}")
    else:
        print("All columns already exist. Nothing to do.")


if __name__ == "__main__":
    migrate()
