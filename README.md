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
| 📝 Narrator | Bloomberg-style desk commentary throughout the day (read-only) |
| 🔬 Quant Researcher | Proposes dynamic strategies, runs backtester, retires underperformers |
| 🔭 Sector Scout | Multi-sector deterministic screening + Chief Scout LLM curation |
| 🎯 Orchestrator | Runs the market-hours loop via APScheduler |

### Core Modules (Tier 1)

Three deterministic, LLM-free modules in `core/` gate every trade:

| Module | File | Purpose |
|---|---|---|
| Edge Score | `core/edge_score.py` | 6-component trade quality score (0.0–1.0) |
| Similarity Engine | `core/similarity.py` | Historical pattern matching via weighted scoring |
| Portfolio Risk | `core/portfolio_risk.py` | Cross-position exposure control with adaptive throttling |

Every BUY/SHORT runs through: safety gates → similarity lookup → edge score → portfolio risk → existing validation.

### Trade Safety Gates (Phase 1)

Composable pre-trade validators in `utils/` that run before the Tier-1 edge/risk
modules. Each gate returns a structured decision (`allow`, `warn`, `downgrade`,
`reject`, `reduce_size`, or `override_required`) and logs its own audit event to
`trade_events`. The PM orchestrates the pipeline with short-circuit-on-reject
semantics — a rejected gate stops evaluation and no subsequent gates run.

| Gate | File | Purpose |
|---|---|---|
| Gate Config | `utils/gate_config.py` | Shared constants and thresholds for all gates |
| Setup Quality Gate | `utils/setup_quality_gate.py` | Blocks/downgrades setup types with poor win rates |
| Pre-Trade Quality Gate | `utils/pre_trade_quality_gate.py` | Rejects trades with low Reviewer quality scores |
| Risk Geometry Gate | `utils/risk_geometry_gate.py` | Validates stop distance, position size, dollar risk, R:R ratio, and target feasibility |

The setup quality gate evaluates case-library performance using a deterministic
rule chain: insufficient data → consecutive losses → historical underperformance
(with recovery override) → rolling underperformance → weak but allowed → allow.
Per-setup-type thresholds are configurable via `MIN_WIN_RATE_BY_SETUP`.

The pre-trade quality gate evaluates Reviewer selection and execution scores with
override support for high-confidence PM decisions.

The risk geometry gate validates full trade geometry before order execution. It
computes a minimum stop distance from both a percentage-based floor and an
ATR-based volatility floor (per symbol class), then either passes the trade
unchanged, reconstructs it with valid geometry (adjusted stop, quantity, dollar
risk), or rejects it outright. Symbol class rules are configurable for high-beta
mega-caps, ETFs, and small-cap momentum setups.

Phase 2 will add catalyst timing and concentration limit gates. See
`.kiro/specs/trade-safety-gates/` for the full spec.

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

### Sector Scout Expansion

A multi-sector deterministic screening pipeline that widens daily opportunity
discovery without weakening trade execution gates. Design principle: **wider
funnel, same strict bouncer**.

| Component | File | Purpose |
|---|---|---|
| Config | `config/sector_scout_config.yaml` | Sector buckets, scoring weights, budget ceilings |
| Models | `utils/sector_scout_models.py` | CandidateRow dataclass, RunSummary, ChiefScoutPick |
| Screener | `utils/sector_scout.py` | Data collection, hard gates, scoring, ranking |
| Chief Scout | `utils/sector_scout_chief.py` | LLM curation + deterministic fallback |
| Watchlist | `utils/expanded_watchlist.py` | Daily expanded watchlist management |
| Persistence | `utils/sector_scout_persistence.py` | Run summaries and candidate row storage |
| Outcomes | `utils/sector_scout_outcomes.py` | Lifecycle tracking (analyst → PM → trade) |
| Metrics | `utils/sector_scout_metrics.py` | Daily success metrics and reporting |
| Logging | `utils/scout_logging.py` | Structured event logging |

**Pipeline flow:** Config → Sector Screeners (7 buckets) → Hard Gates → Scout Score
→ Ranking → Chief Scout LLM (0–8 picks) → Expanded Watchlist → Analyst/PM loops.

**Schedule:** Premarket (existing), 10:00 ET confirmation scan, 12:30 ET midday scan.
Intraday scans use reanalysis cooldown to avoid redundant work.

**Budget ceilings:** Max 7 sectors/run, 20 candidates/sector, 5 finalists/sector,
12 total expanded watchlist symbols, 90s pipeline timeout.

### Catalyst Freshness

The system tracks how current each symbol's catalyst data is. A shared module
(`utils/catalyst_freshness.py`) computes per-symbol freshness state (fresh / aging / stale),
confidence scores, and human-readable labels. This data flows to the web dashboard,
analyst agent (injected into the LLM prompt), and terminal display.

Two event-driven news checks supplement the scheduled News Monitor:
- **Price-Spike Check** (every 15 min) — fetches news when a symbol moves ≥ 2%
- **Position News Poll** (every 30 min) — fetches news for symbols with open positions

See the [User Guide](USER_GUIDE.md#catalyst-freshness-utilscatalyst_freshnesspy) for
the full data flow diagram, freshness thresholds, confidence mapping, and error isolation details.

### Setup-Aware Exit Governance

Replaces the generic ~90-minute force-close timer with setup-specific lifecycle
logic. Each setup type has its own alert/revalidate/force-close/extension timing
defined in a single-source-of-truth registry.

| Component | File | Purpose |
|---|---|---|
| Policy Registry | `utils/setup_time_policy.py` | Per-setup timing (alert, revalidate, force-close, extension) |
| Lifecycle Evaluator | `utils/setup_aware_evaluator.py` | Pure-function deterministic evaluator (no LLM) |
| Entry Validator | `utils/entry_contract_validator.py` | Validates entry metadata for exit governance eligibility |
| Case Classifier | `utils/case_memory_classifier.py` | Classifies closed trades into exit categories |

**Key behaviors:**
- **Thesis-development setups** (news_breakout, news_catalyst, trend_pullback) get
  revalidation windows and extension eligibility up to 180 minutes
- **Fast tactical setups** (momentum_fade, orb, short_squeeze) use shorter timers
  with no extension path
- Extensions require explicit invalidation criteria (numeric stop or structural level)
- Revalidation is deterministic: price vs stop/entry/VWAP/target progress
- Fail-closed on missing/stale market data
- All existing hard controls (stop-loss, EOD hard wall, overnight auth, 24h news governance)
  remain supreme overrides

**Shadow mode:** Set `SETUP_AWARE_SHADOW_MODE=true` to log setup-aware decisions
alongside legacy behavior without altering execution (recommended for 3+ sessions
before enforcement).

**Env vars:**
- `SETUP_AWARE_SHADOW_MODE` — `true`/`false` (default: false)
- `SETUP_AWARE_MAX_MARKET_DATA_STALENESS_SECONDS` — max staleness for revalidation (default: 30)

### News Catalyst 24h Exit Gate

A hard governance layer (`utils/news_trade_governance.py`) that enforces a 24-hour
maximum hold duration for news-driven trades. Prevents silent swing reclassification
that previously allowed stale-catalyst positions to persist indefinitely.

- **Deterministic classification** — no LLM calls, pure field matching
- **Durable persistence** — once classified, governance survives field drift
- **Multi-window reconfirmation** — each `RECONFIRM_AND_HOLD` extends the expiry with fresh evidence
- **Force-close at expiry** — regardless of `target_price` being set
- **Swing reclassification blocked** — requires explicit authorization

See the [User Guide](USER_GUIDE.md#news-catalyst-24h-exit-gate) for full details.

### Strategy Lifecycle Pipeline

Dynamic strategies proposed by the Quant Researcher go through a staged deployment
pipeline before reaching live trading:

```
propose_strategy() → Backtest → Paper Trade (7d) → Live 50% (7d) → Live 100%
```

Each gate requires win_rate > 55% to advance. Failures revert to `backtest_failed`
and the Quant Researcher is notified for iteration. The pipeline runs automatically
during the pre-market orchestrator cycle.

| Stage | Status | Position Size | Gate |
|---|---|---|---|
| Backtest | `backtest` | — | ≥50 trades, >55% win rate |
| Paper Trade | `paper_trade` | — | 7 days, >55% win rate |
| Live 50% | `live_50` | 0.5× | 7 days, >55% win rate |
| Live 100% | `live_100` | 1.0× | Terminal stage |

Key modules: `strategy_backtester.py`, `deployment_pipeline.py`

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
- **8:30 AM** — Pre-market: Scout scans (incl. sector screening), Researcher + Analyst prep, Pipeline evaluation
- **9:30–4:00 PM** — Intraday: every 15 min (configurable)
- **10:00 AM** — Sector Scout confirmation scan (post-open volatility settle)
- **Every 15 min** — Price-spike news check (fetches news for symbols with unusual moves)
- **Every 30 min** — Position news poll (fetches news for symbols with open positions)
- **10 AM, 12 PM, 2 PM** — Scheduled News Monitor (full breaking news scan)
- **12:30 PM** — Sector Scout midday unusual-mover scan
- **4:15 PM** — EOD: Reviewer scores, daily log saved
- **4:30 PM** — Daily Review journal generation

### Desk Narrator Schedule
- **After pre-market** — Morning briefing (triggered by `run_pre_market()`)
- **10, 11, 12 PM** — Hourly recaps
- **2 PM** — Afternoon recap with midday P&L summary
- **4:15 PM Mon–Thu** — Daily wrap
- **4:15 PM Friday** — Weekly wrap (replaces daily wrap)
- **5:15 PM Sunday** — Sunday prep narrative
- **Event-driven** — 🚨 Flash updates on ATR spikes, force exits, catalyst shocks

## Database

SQLite at `db/paper_trader.db`

Tables:
- `trades` — all paper trades with entry/exit/P&L/scores/edge_score/similarity/entry_contract data
- `positions` — current open positions
- `balance` — cash balance history
- `agent_memory` — shared notes between agents (signals, lessons, feedback)
- `dynamic_strategies` — agent-proposed strategies with pipeline lifecycle tracking
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
| CEO_SLACK_BOT_TOKEN | — | Slack bot token (optional) |
| CEO_SLACK_CHANNEL_ID | — | Slack channel ID (optional) |
| SETUP_AWARE_SHADOW_MODE | false | Log setup-aware decisions without executing (shadow mode) |
| SETUP_AWARE_MAX_MARKET_DATA_STALENESS_SECONDS | 30 | Max staleness for revalidation data |
| BLOGGER_BLOG_ID | — | Google Blogger blog ID (optional, for narrator) |
| GOOGLE_CLIENT_ID | — | OAuth2 client ID (optional, for narrator) |
| GOOGLE_CLIENT_SECRET | — | OAuth2 client secret (optional, for narrator) |
| GOOGLE_REFRESH_TOKEN | — | OAuth2 refresh token (optional, for narrator) |

## Tips

- `gpt-4o-mini` is cheap and fast for intraday loops
- For better reasoning on PM decisions, try `gpt-4o` or `claude-3-5-sonnet`
- Check `logs/orchestrator.log` for full agent activity
- The Reviewer needs at least 1 closed trade to run
- Freshness thresholds, confidence mappings, spike detection parameters, and polling intervals are all configurable in `utils/catalyst_freshness.py`
