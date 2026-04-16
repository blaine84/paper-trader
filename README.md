# Paper Trader 📈

A multi-agent paper trading system for day trading SPY, QQQ, IWM, TSLA, NVDA, AMD.

## Architecture

| Agent | Role |
|---|---|
| 📰 Researcher | News, sentiment, market context via Finnhub |
| 📊 Analyst | Technical analysis (RSI, MACD, EMA, BB, VWAP) |
| 🧠 Portfolio Manager | Trade decisions, position sizing, thesis-anchored exits |
| 📋 Bookkeeper | Tracks positions, P&L, stop losses, daily summaries |
| 🔍 Reviewer | Scores closed trades, extracts lessons, feeds back |
| 🎯 Orchestrator | Runs the market-hours loop via APScheduler |

### Core Modules (Tier 1)

Three deterministic, LLM-free modules in `core/` gate every trade:

| Module | File | Purpose |
|---|---|---|
| Edge Score | `core/edge_score.py` | 6-component trade quality score (0.0–1.0) |
| Similarity Engine | `core/similarity.py` | Historical pattern matching via weighted scoring |
| Portfolio Risk | `core/portfolio_risk.py` | Cross-position exposure control with adaptive throttling |

Every BUY/SHORT runs through: similarity lookup → edge score → portfolio risk → existing validation.

### Thesis-Anchored Exits

Exit decisions are anchored to the original trade thesis, not signal freshness:

| Concept | Description |
|---|---|
| Entry Contract | Thesis, setup type, and structured invalidators recorded at trade open |
| Maintenance Review | Default review for open positions — can hold, tighten stop, raise target, or trim. Cannot close. |
| Reversal/Close Review | Only triggered by thesis invalidation, strong opposing signal, or explicit CLOSE. The only path that can close a position. |
| DRIFTING state | Positions without recent analyst signals. Explicitly does NOT trigger exits. |
| Thesis Invalidation Engine | Price Monitor evaluates structured invalidator conditions every 60s. |

PM profiles have an `opposing_evidence_threshold` (conservative: moderate, moderate/aggressive: strong) that gates when opposing signals trigger a Reversal/Close Review.

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
- `trades` — all paper trades with entry/exit/P&L/scores/edge_score/similarity/entry_contract data
- `positions` — current open positions
- `balance` — cash balance history
- `agent_memory` — shared notes between agents (signals, lessons, feedback)
- `daily_log` — end-of-day summaries

## Config (.env)

| Key | Default | Description |
|---|---|---|
| FINNHUB_API_KEY | required | Free at finnhub.io |
| OPENAI_API_KEY | — | Required if using OpenAI |
| ANTHROPIC_API_KEY | — | Required if using Anthropic |
| LLM_PROVIDER | openai | `openai`, `anthropic`, `mistral`, `ollama` |
| LLM_MODEL | gpt-4o-mini | Primary model |
| LLM_LOW_PROVIDER | — | Provider for low-effort tasks |
| LLM_LOW_MODEL | — | Model for low-effort tasks |
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama endpoint |
| OLLAMA_FALLBACK_PROVIDER | anthropic | Fallback if Ollama hangs |
| OLLAMA_FALLBACK_MODEL | claude-haiku-4-5 | Fallback model |
| STARTING_BALANCE | 100000 | Paper trading balance |
| WATCHLIST | SPY,QQQ,IWM,TSLA,NVDA,AMD | Comma-separated tickers |
| LOOP_INTERVAL_MINUTES | 15 | Intraday loop frequency |

## Tips

- `gpt-4o-mini` is cheap and fast for intraday loops
- For better reasoning on PM decisions, try `gpt-4o` or `claude-3-5-sonnet`
- Check `logs/orchestrator.log` for full agent activity
- The Reviewer needs at least 1 closed trade to run
