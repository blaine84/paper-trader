"""
One-time migration: add Tier 1 columns to the existing trades table.
Run once: python migrate_tier1.py
"""
import sqlite3
import sys

DB_PATH = "db/paper_trader.db"

COLUMNS = [
    ("edge_score", "REAL"),
    ("similarity_winrate", "REAL"),
    ("similarity_sample_size", "INTEGER"),
    ("similarity_confidence", "REAL"),
]


def migrate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get existing column names
    cursor.execute("PRAGMA table_info(trades)")
    existing = {row[1] for row in cursor.fetchall()}

    added = []
    for col_name, col_type in COLUMNS:
        if col_name not in existing:
            cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
            added.append(col_name)

    conn.commit()
    conn.close()

    if added:
        print(f"Added columns: {', '.join(added)}")
    else:
        print("All columns already exist, nothing to do.")


if __name__ == "__main__":
    migrate()
