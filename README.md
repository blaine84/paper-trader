# Paper Trader 📈

A multi-agent paper trading system for day trading SPY, QQQ, IWM, TSLA, NVDA, AMD.

## Architecture

| Agent | Role |
|---|---|
| 📰 Researcher | News, sentiment, market context via Finnhub (8:30 AM pre-market) |
| 📡 News Monitor | Breaking news detection at 10 AM, 12 PM, 2 PM + event-driven checks |
| 📊 Analyst | Technical analysis + freshness-aware signal generation |
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

### Catalyst Freshness

The system tracks how current each symbol's catalyst data is and surfaces that information across the web dashboard, analyst agent, and terminal display.

**The problem:** The Researcher runs once at 8:30 AM. The News Monitor catches breaking news at 10/12/2 PM. But the Analyst and the analysis tab only ever read the Researcher's pre-market sentiment — so catalysts appear stale all day, and breaking news is invisible.

**The solution:** A shared freshness module (`utils/catalyst_freshness.py`) computes per-symbol freshness metadata. All consumers import from this single module, so the logic is consistent everywhere.

#### Data flow

```
Researcher (8:30 AM)  ──sentiment──▶  AgentMemory
News Monitor (10/12/2 PM)  ──breaking_news──▶  AgentMemory
Price-Spike Check (every 15 min)  ──breaking_news──▶  AgentMemory
Position News Poll (every 30 min)  ──breaking_news──▶  AgentMemory
                                          │
                                          ▼
                              utils/catalyst_freshness.py
                              (compute freshness state,
                               confidence, labels)
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                    web/app.py      agents/analyst.py   display.py
                    (/api/data)     (LLM prompt)        (terminal)
```

#### Freshness states

The freshness state is based on the age of the most recent catalyst data (whichever is newer — researcher sentiment or breaking news):

| State | Age | Color | Meaning |
|---|---|---|---|
| `fresh` | < 60 min | 🟢 green | Catalyst data is current |
| `aging` | 60–180 min | 🟡 yellow | Data may not reflect current conditions |
| `stale` | > 180 min | 🔴 red | Data is outdated, confidence reduced |

#### How agents interact with freshness

- **Analyst** queries breaking news for each symbol before generating signals. The freshness state and any breaking alerts are injected into the LLM prompt. Stale data gets a warning; aging data gets a note. If the breaking news query fails, the Analyst proceeds with researcher sentiment only.
- **Web API** (`/api/data`) returns three new fields per symbol: `breaking_news` (alert list), `catalyst_freshness` (metadata object), and `freshness_label` (human-readable string). Each data source is wrapped in its own try/except — a failure in breaking news doesn't affect sentiment, and vice versa.
- **Terminal display** shows a color-coded "Fresh" column in the analysis table and appends the most recent breaking news headline (truncated to 40 chars) to the catalysts column.

#### Event-driven news checks

Two lightweight orchestrator jobs supplement the scheduled News Monitor:

| Job | Trigger | What it does |
|---|---|---|
| Price-Spike Check | Every 15 min | Compares current price to price ~15 min ago. If any symbol moved ≥ 2%, fetches news for those symbols. |
| Position News Poll | Every 30 min | Fetches news for all symbols with open positions. |

Both jobs call `fetch_and_store_news()` which merges new alerts with existing market-day alerts (deduplicating by headline) and stores them in AgentMemory. They do not trigger full researcher reanalysis — the Analyst interprets the raw news on its next run.

#### Confidence mapping

Confidence is a function of the researcher's confidence level and the freshness state. It decreases monotonically as freshness degrades:

| Researcher Level | Fresh | Aging | Stale |
|---|---|---|---|
| high | 0.9 | 0.6 | 0.3 |
| medium | 0.7 | 0.4 | 0.2 |
| low | 0.4 | 0.2 | 0.0 |

#### Market day boundary

The market day starts at 4:00 AM ET. Activity between midnight and 4 AM belongs to the previous market day. This matches the existing orchestrator timezone conventions.

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
- **Every 15 min** — Price-spike news check (fetches news for symbols with unusual moves)
- **Every 30 min** — Position news poll (fetches news for symbols with open positions)
- **10 AM, 12 PM, 2 PM** — Scheduled News Monitor (full breaking news scan)
- **4:15 PM** — EOD: Reviewer scores, daily log saved
- **4:30 PM** — Daily Review journal generation

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
- Freshness thresholds, confidence mappings, spike detection parameters, and polling intervals are all configurable in `utils/catalyst_freshness.py`
