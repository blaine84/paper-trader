# Paper Trader 📈

A multi-agent paper trading system for day trading SPY, QQQ, IWM, TSLA, NVDA, AMD.

## Architecture

| Agent | Role |
|---|---|
| 📰 Researcher | News, sentiment, market context via Finnhub |
| 📊 Analyst | Technical analysis (RSI, MACD, EMA, BB, VWAP) |
| 🧠 Portfolio Manager | Trade decisions, position sizing, execution |
| 📋 Bookkeeper | Tracks positions, P&L, stop losses, daily summaries |
| 🔍 Reviewer | Scores closed trades, extracts lessons, feeds back |
| 🎯 Orchestrator | Runs the market-hours loop via APScheduler |

## Feedback Loop

Reviewer → lessons/feedback → AgentMemory DB → Analyst + PM read before deciding

## Setup

### 1. Install dependencies
```bash
cd paper-trader
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your API keys
```

Get a free Finnhub key at: https://finnhub.io/register  
Get OpenAI key at: https://platform.openai.com  
Or set LLM_PROVIDER=anthropic for Claude.

### 3. Run

**Full scheduler (market hours, Mon-Fri):**
```bash
python orchestrator.py
```

**Single test cycle:**
```bash
python orchestrator.py once
```

## Schedule (ET)
- **8:30 AM** — Pre-market: Scout scans, Researcher + Analyst prep
- **9:30–4:00 PM** — Intraday: every 15 min (configurable)
- **4:15 PM** — EOD: Reviewer scores, daily log saved

## Database

SQLite at `db/paper_trader.db`

Tables:
- `trades` — all paper trades with entry/exit/P&L/scores
- `positions` — current open positions
- `balance` — cash balance history
- `agent_memory` — shared notes between agents (signals, lessons, feedback)
- `daily_log` — end-of-day summaries

## Config (.env)

| Key | Default | Description |
|---|---|---|
| FINNHUB_API_KEY | required | Free at finnhub.io |
| OPENAI_API_KEY | required | Or use Anthropic |
| LLM_PROVIDER | openai | openai or anthropic |
| LLM_MODEL | gpt-4o-mini | Any compatible model |
| STARTING_BALANCE | 100000 | Paper trading balance |
| WATCHLIST | SPY,QQQ,IWM,TSLA,NVDA,AMD | Comma-separated tickers |
| LOOP_INTERVAL_MINUTES | 15 | Intraday loop frequency |

## Tips

- `gpt-4o-mini` is cheap and fast for intraday loops
- For better reasoning on PM decisions, try `gpt-4o` or `claude-3-5-sonnet`
- Check `logs/orchestrator.log` for full agent activity
- The Reviewer needs at least 1 closed trade to run
