# Running Paper Trader on a Raspberry Pi

The Pi only orchestrates — all heavy compute goes to Finnhub + OpenAI/Anthropic APIs.
A Pi 3B+ or better is sufficient.

## Requirements

- Raspberry Pi 3B+ or newer
- Raspberry Pi OS (Bookworm recommended)
- Python 3.10+
- Internet connection (always-on)

## Quick Setup

```bash
# Copy project files to Pi (from your PC):
scp -r paper-trader/ pi@<pi-ip>:/home/pi/paper-trader

# SSH into Pi:
ssh pi@<pi-ip>

# Run setup script:
cd /home/pi/paper-trader
bash deploy/setup_pi.sh

# Add your API keys:
nano .env

# Test a single cycle:
source venv/bin/activate
python orchestrator.py once

# Start the service:
sudo systemctl start paper-trader
```

## Service Management

```bash
# Start / stop / restart
sudo systemctl start paper-trader
sudo systemctl stop paper-trader
sudo systemctl restart paper-trader

# Check status
sudo systemctl status paper-trader

# Live logs
sudo journalctl -u paper-trader -f
tail -f /home/pi/paper-trader/logs/service.log
```

## The service:
- Starts automatically on boot
- Restarts automatically if it crashes (30s delay)
- Logs to both journalctl and logs/service.log
- APScheduler handles the market-hours schedule — no cron needed

## Checking in from your PC

SSH in anytime to see what's happened:

```bash
ssh pi@<pi-ip>
cd /home/pi/paper-trader
source venv/bin/activate

# Quick status
python -c "
from db.schema import init_db, get_session
from agents.bookkeeper import print_dashboard
engine = init_db()
print_dashboard(engine)
"

# Or just tail the log
tail -100 logs/service.log
```

## Notes

- The Pi's clock must be accurate — `timedatectl` should show NTP synced
- APScheduler uses `America/New_York` timezone — Pi local time doesn't matter
- SQLite DB lives at `db/paper_trader.db` — back it up periodically
- If you update the code, `sudo systemctl restart paper-trader`
