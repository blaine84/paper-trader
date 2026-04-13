"""Fix the two SPY trades that entered at a hallucinated ~$260 price."""
import sqlite3

DB_PATH = "db/paper_trader.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Void the bad trades
c.execute("""
    UPDATE trades
    SET status='closed', exit_price=0, pnl=0, pnl_pct=0,
        reason_exit='voided - bad entry price (LLM hallucinated $260)'
    WHERE symbol='SPY' AND status='open' AND entry_price < 300
""")
print(f"Trades voided: {c.rowcount}")

# Remove the bad positions
c.execute("DELETE FROM positions WHERE symbol='SPY' AND avg_cost < 300")
print(f"Positions removed: {c.rowcount}")

# Restore the margin that was deducted for the short trades
# For each profile, find the latest balance and add back the margin
for profile in ('moderate', 'aggressive'):
    c.execute("""
        SELECT cash FROM balance
        WHERE profile=? ORDER BY timestamp DESC LIMIT 1
    """, (profile,))
    row = c.fetchone()
    if row:
        # Find what was deducted (quantity * entry_price for the voided trades)
        c.execute("""
            SELECT SUM(quantity * entry_price) FROM trades
            WHERE symbol='SPY' AND profile=?
            AND reason_exit LIKE 'voided%' AND entry_price < 300
        """, (profile,))
        margin = c.fetchone()[0] or 0
        if margin > 0:
            new_cash = row[0] + margin
            c.execute("""
                INSERT INTO balance (profile, cash, timestamp)
                VALUES (?, ?, datetime('now'))
            """, (profile, new_cash))
            print(f"{profile}: restored ${margin:,.2f} margin, cash now ${new_cash:,.2f}")

conn.commit()
conn.close()
print("Done.")
