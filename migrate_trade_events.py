"""
One-time migration: create normalized trade_events audit table.

Run once: python migrate_trade_events.py
"""
import sqlite3

DB_PATH = "db/paper_trader.db"


def migrate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_events (
            id INTEGER PRIMARY KEY,
            trade_id INTEGER NULL,
            timestamp DATETIME NOT NULL,
            event_type VARCHAR(64) NOT NULL,
            agent VARCHAR(64),
            symbol VARCHAR(10),
            profile VARCHAR(16),
            price REAL,
            message TEXT,
            payload_json TEXT,
            FOREIGN KEY(trade_id) REFERENCES trades(id)
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_trade_events_trade_id ON trade_events(trade_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_trade_events_symbol_time ON trade_events(symbol, timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_trade_events_type_time ON trade_events(event_type, timestamp)")

    conn.commit()
    conn.close()
    print("trade_events table ready")


if __name__ == "__main__":
    migrate()
