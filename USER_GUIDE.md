# Paper Trader — User Guide

A multi-agent paper trading system that runs autonomously during market hours,
learns from every trade, and improves over time through structured feedback loops.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [How It Works](#how-it-works)
4. [The Agents](#the-agents)
5. [Risk Profiles](#risk-profiles)
6. [The Case Library](#the-case-library)
7. [Feedback Loops](#feedback-loops)
8. [Daily Schedule](#daily-schedule)
9. [Running the System](#running-the-system)
10. [Inspect CLI](#inspect-cli)
11. [Running on a Raspberry Pi](#running-on-a-raspberry-pi)
12. [Watchlist](#watchlist)
13. [Understanding the Scores](#understanding-the-scores)
14. [Edge Score & Risk Engine](#edge-score--risk-engine)
15. [Database](#database)
16. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies
cd paper-trader
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
nano .env   # add your API keys

# 3. Test a single cycle (no scheduler)
python orchestrator.py once

# 4. Run live (market hours, Mon–Fri)
python orchestrator.py
```

---

## Configuration

All config lives in `.env`. Copy `.env.example` to get started.

| Key | Default | Description |
|---|---|---|
| `FINNHUB_API_KEY` | required | Free at [finnhub.io](https://finnhub.io/register) |
| `OPENAI_API_KEY` | — | Required if using OpenAI |
| `ANTHROPIC_API_KEY` | — | Required if using Anthropic |
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `mistral`, or `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Primary model (high tier) |
| `LLM_MED_PROVIDER` | — | Provider for medium-effort tasks (Analyst, Quant Researcher) |
| `LLM_MED_MODEL` | — | Model for medium tier (e.g. `llama3.1:8b`) |
| `LLM_LOW_PROVIDER` | — | Provider for low-effort tasks (Scout, Researcher, Weekly Prep) |
| `LLM_LOW_MODEL` | — | Model for low tier (e.g. `mistral:latest`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (can point to another machine) |
| `OLLAMA_TIMEOUT` | `60` | Seconds before Ollama fallback triggers |
| `OLLAMA_FALLBACK_PROVIDER` | `anthropic` | Fallback if Ollama fails |
| `OLLAMA_FALLBACK_MODEL` | `claude-haiku-4-5` | Fallback model |
| `STARTING_BALANCE` | `100000` | Paper balance per profile (×3) |
| `WATCHLIST` | `SPY,QQQ,IWM,TSLA,NVDA,AMD` | Core tickers, comma-separated |
| `LOOP_INTERVAL_MINUTES` | `15` | How often intraday loop runs |

### LLM Tiers

| Tier | Used by | Recommended model |
|---|---|---|
| High | PM decisions, Meta Reviewer | `claude-sonnet-4-6` (Anthropic) |
| Medium | Analyst, Quant Researcher | `llama3.1:8b` (Ollama, local) |
| Low | Scout, Researcher, Weekly Prep | `mistral:latest` (Ollama, local) |

Only PM decisions and the Meta Reviewer hit the cloud API. Everything else runs locally via Ollama.

---

## How It Works

The system runs 9 agents on a market-hours schedule. Each agent has a single job.
No agent does another agent's job.

```
8:30 AM ──► Scout ──► Researcher ──► Quant Researcher ──► Analyst
                                                              │
9:30 AM ──► Price Monitor (every 60s) ◄──────────────────────┘
            │                                                  │
            ├─ Stop/target hit? → close immediately            │
            └─ Key level breach? → trigger PM                  │
                                                               │
9:30-12  ──► PM decisions every 15 min ◄───────────────────────┘
12-4 PM  ──► PM decisions every 30 min
            Analyst refreshes every 15 min all day (free, local LLM)

4:15 PM ──► Reviewer (score trades, build cases)
        ──► Bookkeeper (daily log)

Sunday  ──► Weekly Prep
        ──► Meta Reviewer (grades agents, suggests improvements)
```

All three Portfolio Manager profiles run in **parallel** with **isolated portfolios**.
They all see the same signals. They make different decisions based on their risk rules.

---

## The Agents

### 🔭 Scout
Runs at 8:30 AM. Scans ~50 liquid stocks for unusual activity (moves ≥3%).
Surfaces 1–3 additional symbols worth watching today based on catalysts and
historical case win rates. Returns **symbols and context only** — no trade opinions.

### 📰 Researcher
Covers the full watchlist (core tickers + Scout picks). Pulls news, earnings,
analyst ratings, and market context from Finnhub. Assesses sentiment per symbol:
bullish / bearish / neutral, with key catalysts and risks.

### 📊 Analyst
Runs technical analysis on every symbol. Computes RSI, MACD, EMA (9/21/50),
Bollinger Bands, ATR, and VWAP. Produces a **signal** — not a trade recommendation.

**Analyst output:**
```
signal:       LONG | SHORT | HOLD
strength:     weak | moderate | strong
confidence:   low | medium | high
setup_type:   gap_and_go | vwap_reclaim | news_breakout | etc.
key_levels:   support, resistance, vwap, prior_high, prior_low
invalidation: the condition that would make this setup wrong
indicators:   rsi, macd_bias, ema_trend, above_vwap, bb_position
```

The Analyst does **not** suggest entry price, stop, or target. That's the PM's job.

### 🧠 Portfolio Manager (×3)
Three profiles run in parallel — Conservative, Moderate, Aggressive.
Each reads the same Analyst signals and Researcher context, then decides:

- **Whether** to act (maybe the signal is right but timing is wrong for their profile)
- **Action**: BUY / SHORT / CLOSE / pass
- **Entry price** (based on key levels, not just current price)
- **Stop** (placed at Analyst's invalidation level + profile risk tolerance)
- **Target** (based on key levels and required R:R)
- **Position size** (profile's max % / stop distance)

### 📋 Bookkeeper
Tracks all positions, cash, and P&L across all three portfolios. Monitors stop
losses every cycle and force-closes positions that breach them. Prints the
terminal dashboard. Saves end-of-day summaries.

### 🔍 Reviewer
Runs at 4:15 PM after market close. Reviews every closed trade (in batches of 3),
extracts a **structured case** for the case library, and routes feedback to the
right agents. Scores are split — see [Understanding the Scores](#understanding-the-scores).

### 📐 Quant Researcher
Evaluates which strategies have edge in current market conditions. Cross-references
the strategy library against the case library and backtest results. Runs the
backtester automatically every 3 days. Can propose new dynamic strategies based
on patterns in the case data — strategies that underperform after 10+ trades are
automatically retired.

### ⚡ Price Monitor
Runs every 60 seconds during market hours using yfinance (free, no rate limit).
Checks open positions against stop/target levels and analyst signals against key
levels. Triggers immediate PM action when conditions are met — no waiting for the
next scheduled cycle.

### 🔬 Meta Reviewer
Runs weekly after Sunday prep. Grades each agent (A–F), tracks trends
(improving/stable/degrading), and writes specific recommendations that agents
read as context. Also suggests code refactors and feature additions visible in
the web dashboard's System Review tab.

---

## Risk Profiles

Three portfolios run simultaneously, each with $100k paper balance.

### 🛡️ Conservative
| Setting | Value |
|---|---|
| Max positions | 2 |
| Max position size | 15% of portfolio |
| Minimum R:R | 3:1 |
| Min signal strength | strong |
| Avoid first/last | 30 min |
| Daily loss limit | 2% |

Prefers ETFs (SPY, QQQ, IWM). Only acts on high-conviction setups.
Stops trading for the day if down 2%.

### ⚖️ Moderate
| Setting | Value |
|---|---|
| Max positions | 3 |
| Max position size | 25% |
| Minimum R:R | 2:1 |
| Min signal strength | moderate |
| Avoid first/last | 15 min |
| Daily loss limit | 3% |

Balanced. Trades across the full watchlist. Trusts the analyst but applies judgment.

### 🔥 Aggressive
| Setting | Value |
|---|---|
| Max positions | 4 |
| Max position size | 35% |
| Minimum R:R | 1.5:1 |
| Min signal strength | weak |
| Avoid first/last | 5 min |
| Daily loss limit | 5% |

Chases momentum. Prefers individual stocks (TSLA, NVDA, AMD). Wider stops,
bigger targets. Will trade Scout picks early. May pyramid into winners.

---

## The Case Library

Every closed trade and every Scout pick becomes a **structured, queryable case record** — not a prose summary.

```
symbol:                 NVDA
date:                   2026-03-22
setup_type:             gap_and_go
catalyst_type:          analyst_upgrade
float_profile:          mega_cap
sector:                 tech
market_regime:          risk_on
premarket_gap_pct:      4.2
premarket_volume_rank:  high
entry_timing:           first_15min
bias:                   LONG
signal_strength:        strong
signal_confidence:      high
rsi_at_entry:           58.3
above_vwap:             true
above_daily_resistance: true
ema_trend:              bullish
bb_position:            upper
invalidation:           loses VWAP or closes below 118.50
entry_vs_level:         above_vwap
outcome:                success
pnl_pct:                +2.1
holding_minutes:        23
lesson:                 gap_and_go strongest in first 15 min when regime risk_on and above daily resistance
conditions_for_success: ["market_regime=risk_on", "above_daily_resistance=true", "premarket_gap_pct>3"]
conditions_to_avoid:    ["entry_timing=open", "rsi_at_entry>75"]
selection_score:        8.5
execution_score:        6.5
review_score:           7.5
```

This is the system's institutional memory. Over time, agents query it to find
relevant precedents before making decisions.

---

## Feedback Loops

```
Reviewer
  ├── selection_score + selection_feedback
  │     ↓
  │   Scout + Analyst
  │   "Are we finding and reading the right setups?"
  │
  ├── execution_score + execution_feedback (per profile)
  │     ↓
  │   Conservative PM  (gets its own history)
  │   Moderate PM      (gets its own history)
  │   Aggressive PM    (gets its own history)
  │   "Are we entering, sizing, and exiting correctly?"
  │
  └── behavioral parameters (auto-extracted from feedback)
        ↓
      Prose feedback → LLM → executable parameters:
        entry_offset_pct, size_multiplier, stop_buffer_pct,
        avoid_setups, favor_setups, min_r_override, etc.
      Applied automatically to every PM decision before execution.

Confidence Adjustment (pre-trade)
  ├── Query case library for setup_type + regime win rate
  ├── Win rate < 35% (5+ cases) → BLOCK trade
  ├── Win rate 35-50% → downgrade confidence modifier
  └── Win rate ≥ 50% → no adjustment

Meta Reviewer (weekly)
  ├── grades each agent A–F with trend
  ├── writes per-agent recommendations → agents read as context
  ├── suggests code refactors and features
  └── compares week-over-week performance

Quant Researcher
  ├── proposes new strategies from case patterns
  ├── tracks dynamic strategy win rates
  └── retires strategies that underperform
```

**Selection score** — was the setup correctly identified? Did the Analyst's read
match what actually happened? Scored independently of how PM traded it.

**Execution score** — did PM make good decisions given the signal? Entry level,
stop logic, sizing, exit discipline. Scored independently of whether the setup was good.

A great read on a poorly executed trade scores high on selection, low on execution.
Clean execution on a bad setup scores low on selection, high on execution.

This prevents bad signal reads from corrupting PM feedback and vice versa.

---

## Decision Logic Flow

The full lifecycle of a trade from signal to exit:

```
1. SIGNAL GENERATION
   Analyst detects setup → outputs signal (LONG/SHORT/HOLD)
   with setup_type, setup_reasoning, key_levels, invalidation, strength, confidence

2. CONFIDENCE ADJUSTMENT (automatic, pre-trade)
   Query case library for setup_type + market_regime win rate:
   ├─ Win rate < 35% (5+ cases) → BLOCK trade entirely
   ├─ Win rate 35-50% → downgrade confidence modifier (0.7-1.0)
   └─ Win rate ≥ 50% or <5 cases → no adjustment

3. TRADE DECISION (PM)
   PM reads: signal + portfolio state + feedback + news + position health
           + behavioral parameters + meta-reviewer recommendations
   PM decides: action, entry price, stop, target, quantity, rationale

4. EDGE SCORE & RISK GATING (automatic, pre-validation)
   Three deterministic modules run before existing validation:

   a. SIMILARITY ENGINE
      Query case library for historically similar trades (weighted scoring).
      Returns similarity_winrate, similarity_avg_r, similarity_confidence.
      If no matches → skip similarity weighting (no penalty).

   b. EDGE SCORE
      6-component weighted formula (0.0–1.0):
        0.25 × setup win rate
        0.20 × similarity win rate
        0.15 × signal strength
        0.10 × signal confidence
        0.15 × indicator confluence
        0.15 × similarity quality (sample-size confidence)

      Hard rejection: if setup has 10+ cases and win rate < 35% → block outright.
      Soft rejection: if edge score < 0.4 → block trade.
      Position sizing: quantity × edge_score, capped at 1.2× base size.

   c. PORTFOLIO RISK ENGINE
      Checks exposure across correlated buckets:
        index (SPY, QQQ, IWM, DIA)
        semis (NVDA, AMD, INTC, TSM)
        ev (TSLA, LCID, RIVN)
        mega_growth (NVDA, TSLA, META, AMZN)
      Symbols can belong to multiple buckets.
      Rules: max 50% per bucket, configurable total exposure (1.2–1.5×).
      Adaptive throttling: 3+ consecutive losses → reduce size 25–50%.

   d. All three modules are deterministic (no LLM calls), fast (<10ms).

5. BEHAVIORAL ADJUSTMENT (automatic, post-decision)
   Behavioral parameters (extracted from reviewer feedback) applied:
   ├─ Entry offset (earlier/later entries)
   ├─ Size multiplier (scale up/down)
   ├─ Reduce size on low confidence
   ├─ Block avoided setup types
   ├─ Boost favored setup types
   ├─ Stop buffer (widen/tighten)
   └─ Min R:R override

6. TRADE VALIDATION (automatic)
   Before any trade hits the database:
   ✓ Stop price is valid and on correct side of entry
   ✓ Target price is valid and on correct side of entry
   ✓ R:R ratio ≥ 1:1
   ✓ Position size within profile max allocation
   ✓ No correlated pair exposure (SPY+IWM, SPY+QQQ, QQQ+IWM)
   ✓ Quantity is positive
   → If any check fails, trade is REJECTED with reason logged

7. STOP DERIVATION (if LLM omits stop/target)
   Priority: ATR-based (1.5× ATR) → key level (support/resistance) → 1.5% fallback

8. EXECUTION
   Trade written to DB with stop_price, target_price, edge_score,
   similarity_winrate, similarity_sample_size, and similarity_confidence persisted
   Position created, cash deducted
   Trade queued for review automatically

9. PROFIT MANAGEMENT (every 60 seconds)
   ├─ +1R: take partial profit (25-50% by profile), move stop to breakeven
   ├─ +2R: take more profit (if profile allows), trail stop to +1R
   ├─ +3R: trail stop to +2R
   └─ Each action fires once per trade, tracked in memory

   Partial profit rules by profile:
   ├─ Conservative: 50% at +1R, 25% at +2R
   ├─ Moderate: 33% at +1R, 25% at +2R
   └─ Aggressive: 25% at +1R, let rest ride

10. MONITORING (continuous, every 60 seconds)
   Price Monitor checks:
   ├─ Stop hit? → close immediately (0.1% buffer, direction-validated)
   ├─ Target hit? → close immediately (direction-validated)
   ├─ Key level breach? → filter through local LLM → trigger PM if actionable
   ├─ Rapid move (>1.5% in 5 min)? → filter → trigger PM
   ├─ Approaching key level (<0.3%)? → log for awareness
   └─ 15-min cooldown per symbol to prevent alert spam

   Position Timer checks (every 5 min):
   ├─ Setup-specific time limits (see below)
   ├─ Stale trade detection (momentum_fade: <0.5R after 35 min)
   ├─ Thesis revalidation at 60 min (LLM checks VWAP, volume, structure)
   ├─ Force close at setup max time
   └─ Hard wall: ALL intraday positions closed at 3:45 PM ET

   Position Health (every hour, local LLM):
   └─ Reviews all positions against current indicators
       Flags deteriorating positions even if stops haven't hit

   News Monitor (every 2 hours, local LLM):
   └─ Checks for breaking catalysts that could affect open positions

10. EXIT
    Trade closed → P&L calculated → cash returned
    Trade auto-queued for review

11. REVIEW (every 15 min during market hours)
    Reviewer pulls from queue → scores trade → creates case
    Selection feedback → Scout + Analyst
    Execution feedback → PM (per profile)
    Behavioral parameters extracted from feedback → applied next cycle
    Stale review alert if pending >24 hours

12. LEARNING (weekly)
    Meta Reviewer grades all agents A–F
    Quant Researcher proposes/retires dynamic strategies
    Backtester validates strategy edge on historical data
    All feedback written to agent_memory → agents read next cycle
```

### Position Time Limits

| Setup Type | Stale | Alert | Revalidate | Force Close |
|---|---|---|---|---|
| momentum_fade | 35 min (<0.5R) | 45 min | 60 min (LLM) | 75 min |
| gap_and_go | — | 60 min | — | 90 min |
| vwap_reclaim | — | 60 min | — | 90 min |
| orb | — | 45 min | — | 75 min |
| trend_pullback | — | 90 min | — | 120 min |
| news_catalyst | — | 60 min | — | 90 min |
| short_squeeze | — | 30 min | — | 60 min |
| **All intraday** | — | — | — | **3:45 PM ET hard wall** |

### Trade Validation Rules

Every BUY/SHORT is validated before execution:
1. Entry price must be a valid positive number
2. Stop must be non-null (fallback: ATR-based → key level → 1.5%)
3. Target must be non-null
4. Stop on correct side (LONG: below entry, SHORT: above entry)
5. Target on correct side (LONG: above entry, SHORT: below entry)
6. R:R ratio ≥ 1:1
7. Position size ≤ profile max allocation %
8. No correlated pair in same direction (SPY+IWM, SPY+QQQ, QQQ+IWM)
9. Case library win rate ≥ 35% for this setup_type + regime (blocks if below)

### Stop Derivation Priority

When the LLM doesn't provide a stop price:
1. **ATR-based** — 1.5× ATR from entry (adapts to current volatility)
2. **Key level** — just below support (long) or above resistance (short) from analyst signal
3. **Last resort** — 1.5% from entry (only if ATR and levels both unavailable)

### Partial Profit Taking

| R Multiple | Conservative | Moderate | Aggressive |
|---|---|---|---|
| +1R | Take 50%, stop → breakeven | Take 33%, stop → breakeven | Take 25%, stop → breakeven |
| +2R | Take 25% more, trail to +1R | Take 25% more, trail to +1R | Trail to +1R |
| +3R | Trail to +2R | Trail to +2R | Trail to +2R |

### Behavioral Parameters

The Reviewer automatically converts prose feedback into executable parameters:

| Parameter | Effect |
|---|---|
| `entry_offset_pct` | Shift entry price (negative = enter earlier) |
| `size_multiplier` | Scale position size (0.5 = half, 1.5 = 150%) |
| `reduce_size_on_low_confidence` | Halve size when confidence is "low" |
| `avoid_setups` | Setup types to skip entirely |
| `favor_setups` | Setup types to boost size 20% |
| `stop_buffer_pct` | Widen/tighten stops |
| `min_r_override` | Require higher R:R than profile default |

### Momentum Fade Lifecycle

```
0 min   → Trade opened
35 min  → If <0.5R achieved → mark STALE
45 min  → If still stale → ALERT PM
60 min  → REVALIDATE thesis via LLM:
           - Still below VWAP?
           - Volume fading?
           - Lower highs/lower lows intact?
           → If invalid → EXIT IMMEDIATELY
75 min  → FORCE EXIT regardless
3:45 PM → HARD WALL close
```

---

## Daily Schedule

All times Eastern (ET), Monday–Friday.

| Time | Event |
|---|---|
| **Sunday 5:00 PM** | Weekly prep + Meta Reviewer (grades agents, suggests improvements) |
| 8:30 AM | Scout → Researcher → Quant Researcher → Analyst |
| 9:30 AM | Market opens, price monitor starts (every 60s), position timer starts (every 5 min) |
| 9:30–12:00 | PM decisions every 15 min, Analyst refresh every 15 min |
| 10:00, 12:00, 2:00 | News monitor checks for breaking catalysts |
| 10:30–3:30 | Position health check every hour |
| 10:00–4:00 | Reviewer queue processes pending reviews every 15 min |
| 12:00–4:00 | PM decisions every 30 min, Analyst refresh every 15 min |
| 3:45 PM | Hard wall: all intraday positions force-closed |
| 4:00 PM | Market closes, price monitor stops |
| 4:15 PM | Reviewer scores remaining trades, Bookkeeper saves daily log |

---

## Running the System

### Single test cycle
```bash
python orchestrator.py once
```
Runs pre-market + one intraday cycle immediately. Good for testing your API keys
and checking everything works before letting it run live.

### Test weekly prep
```bash
python orchestrator.py weekly
```
Runs the Sunday weekly prep immediately. Good for testing on any day.

### Live market-hours scheduler
```bash
python orchestrator.py
```
Starts APScheduler. Waits for the scheduled times (8:30, 9:30–4:00, 4:15).
Logs to `logs/orchestrator.log` and stdout.

### Web dashboard
```bash
python web/app.py
```
Opens a dashboard at `http://localhost:5000` (or `http://<pi-ip>:5000` from another machine).
Shows live positions with unrealized P&L, watchlist signals, performance analytics, agent feedback, and case library.

### Backtester
```bash
python backtest.py                          # all symbols, all strategies, 1 year
python backtest.py --symbols SPY QQQ TSLA  # specific symbols
python backtest.py --days 180              # last 180 days
python backtest.py --strategy gap_and_go   # single strategy
python backtest.py --export results.csv    # export to CSV
```
Rule-based backtest using the same technical indicators as the Analyst. The Quant Researcher also runs this automatically every 3 days and feeds results into its strategy recommendations.

### On-demand signal refresh (Linux/Pi only)
```bash
# Trigger a full pre-market refresh (scout + researcher + analyst)
kill -USR1 $(pgrep -f orchestrator.py)

# Trigger an intraday cycle (analyst refresh + PM decisions)
kill -USR2 $(pgrep -f orchestrator.py)
```

### Logs
```bash
tail -f logs/orchestrator.log
tail -f logs/service.log   # if running as systemd service on Pi
```

---

## Inspect CLI

Query the case library, scores, feedback, and trade history from the command line.
Works locally or over SSH on the Pi.

```bash
# Browse the case library
python portfolio_inspect.py cases
python portfolio_inspect.py cases --setup gap_and_go
python portfolio_inspect.py cases --outcome failure
python portfolio_inspect.py cases --symbol TSLA
python portfolio_inspect.py cases --regime risk_off
python portfolio_inspect.py cases --bias SHORT
python portfolio_inspect.py cases --setup vwap_reclaim -v   # verbose: shows lessons + conditions

# Score trends over time
python portfolio_inspect.py scores
python portfolio_inspect.py scores --limit 50

# Agent feedback
python portfolio_inspect.py feedback                        # all profiles
python portfolio_inspect.py feedback --profile aggressive   # one profile

# What setup types are working?
python portfolio_inspect.py winrates

# Trade history
python portfolio_inspect.py trades
python portfolio_inspect.py trades --profile conservative
python portfolio_inspect.py trades --limit 50

# Current open positions (all profiles)
python portfolio_inspect.py positions

# Daily P&L log
python portfolio_inspect.py summary
python portfolio_inspect.py summary --limit 60
```

---

## Running on a Raspberry Pi

The Pi orchestrates and the Mac Mini runs local LLMs via Ollama. Cloud API calls
(Anthropic) are only used for PM decisions. A Pi 3B+ or better is sufficient.

### Setup
```bash
# Copy project to Pi (from your PC)
scp -r paper-trader/ blaine@<pi-ip>:/home/blaine/

# SSH into Pi
ssh blaine@<pi-ip>

# Run setup script
cd /home/blaine/paper-trader
bash deploy/setup_pi.sh

# Add your API keys
nano .env

# Test
source venv/bin/activate
python orchestrator.py once

# Start the service
sudo systemctl start paper-trader
```

### Service management
```bash
sudo systemctl start paper-trader
sudo systemctl stop paper-trader
sudo systemctl restart paper-trader
sudo systemctl status paper-trader

# Live logs
sudo journalctl -u paper-trader -f
tail -f /home/blaine/paper-trader/logs/service.log
```

The service starts automatically on boot and restarts on crash.

### Checking in remotely
```bash
ssh blaine@<pi-ip>
cd /home/blaine/paper-trader
source venv/bin/activate

python portfolio_inspect.py positions    # what's open right now
python portfolio_inspect.py trades       # what traded today
python portfolio_inspect.py summary      # daily P&L
```

---

## Watchlist

The **core watchlist** is set in `.env` and never changes:
```
WATCHLIST=SPY,QQQ,IWM,TSLA,NVDA,AMD
```

Every morning the **Scout** adds 1–3 additional symbols based on unusual activity,
news catalysts, and historical case win rates. These are active for that day only.

To permanently add a ticker, add it to `WATCHLIST` in `.env`.

---

## Understanding the Scores

All scores are 1–10. Color coding in the terminal:

| Range | Color | Meaning |
|---|---|---|
| 7–10 | 🟢 Green | Strong |
| 5–6.9 | 🟡 Yellow | Acceptable |
| 1–4.9 | 🔴 Red | Poor |

### Selection Score (→ Scout + Analyst)
*"Was the setup correctly identified?"*

High when:
- Analyst called the right direction
- Setup type matched what actually played out
- Key levels were accurate
- Invalidation condition was meaningful

Low when:
- Wrong direction called
- Setup type misidentified
- Market regime misread
- Invalidation was arbitrary

### Execution Score (→ PM, per profile)
*"Did PM make good decisions given the signal?"*

High when:
- Entered at a logical level (key level, VWAP, breakout)
- Stop placed at the invalidation level
- Size appropriate for profile rules
- Exit was disciplined

Low when:
- Chased entry above key levels
- Stop was arbitrary or too tight/wide
- Oversized relative to profile rules
- Held past target or exited too early

### Review Score
Average of selection + execution. Used for overall tracking.

---

## Edge Score & Risk Engine

Three deterministic modules in `core/` gate every BUY/SHORT trade before it reaches
the existing validation pipeline. No LLM calls — pure Python, fast, deterministic.

### Edge Score (`core/edge_score.py`)

Computes a continuous 0.0–1.0 score for each proposed trade using six weighted components:

| Component | Weight | Source |
|---|---|---|
| Setup win rate | 0.25 | Case library (setup_type + regime) |
| Similarity win rate | 0.20 | Similarity engine matches |
| Signal strength | 0.15 | Analyst signal (weak/moderate/strong) |
| Signal confidence | 0.10 | Analyst signal (low/medium/high) |
| Indicator confluence | 0.15 | VWAP, EMA, RSI, MACD, BB alignment |
| Similarity quality | 0.15 | Sample-size confidence: min(1.0, n/10) |

Gating rules:
- Hard reject: setup has 10+ cases and win rate < 35% → blocked outright
- Soft reject: edge score < 0.4 → blocked
- Position sizing: `quantity × edge_score`, capped at `base_size × 1.2`

### Similarity Engine (`core/similarity.py`)

Finds historically similar trades using weighted scoring (not strict filtering):

| Criterion | Weight |
|---|---|
| Setup type match | 0.30 |
| Market regime match | 0.25 |
| RSI distance (continuous) | 0.15 |
| VWAP alignment | 0.15 |
| EMA trend alignment | 0.15 |

Returns top 10 matches by similarity score. Computes aggregate stats:
`similarity_winrate`, `similarity_avg_r`, `similarity_confidence`.
When no matches are found, similarity weighting is skipped entirely (no penalty).

### Portfolio Risk Engine (`core/portfolio_risk.py`)

Prevents overexposure across correlated positions.

Exposure buckets (symbols can belong to multiple):

| Bucket | Symbols |
|---|---|
| index | SPY, QQQ, IWM, DIA |
| semis | NVDA, AMD, INTC, TSM |
| ev | TSLA, LCID, RIVN |
| mega_growth | NVDA, TSLA, META, AMZN |

Rules:
- Max 50% of equity per bucket
- Configurable total exposure threshold (default 1.5×)
- Adaptive throttling: 3+ consecutive losses → reduce position size 25–50%

Outputs a composite `risk_score` (0.0–1.0) summarizing overall portfolio risk.

### Error Handling

| Module | On failure | Behavior |
|---|---|---|
| Edge Score | Exception | Reject trade (fail-closed) |
| Similarity | Exception | Proceed with zero stats (fail-open) |
| Portfolio Risk | Exception | Proceed with existing validation (fail-open) |

### Structured Logging

Every BUY/SHORT trade logs three blocks:
```
EDGE SCORE: 0.64 | setup_winrate=0.58 (n=12) | similarity_winrate=0.60 (n=8) | similarity_confidence=0.80 | confluence=0.80 | similarity_quality=0.80
PORTFOLIO RISK: total_exposure=0.42 | index=0.17, semis=0.25, ev=0.00, mega_growth=0.15, other=0.00
DECISION: size_scaled=64 status=EXECUTED edge=0.640
```

---

## Database

SQLite at `db/paper_trader.db`. Back it up periodically.

| Table | Contents |
|---|---|
| `trades` | All paper trades with entry/exit/P&L/scores/stop/target/edge_score/similarity data |
| `positions` | Current open positions (per profile + side) |
| `balance` | Cash balance history (per profile) |
| `agent_memory` | Shared notes between agents (signals, feedback, meta reviews) |
| `daily_log` | End-of-day summaries |
| `cases` | The case library — structured trade lessons |
| `dynamic_strategies` | Agent-proposed strategies with win rate tracking |
| `review_queue` | Trades pending review (auto-queued on close) |

### Quick backup
```bash
cp db/paper_trader.db db/paper_trader.db.bak
```

---

## Troubleshooting

### "No module named finnhub"
```bash
pip install -r requirements.txt
```

### Finnhub returns empty candles
Free tier has rate limits (~60 calls/min). If the loop interval is too short
or the watchlist is too large, you may hit limits. Try increasing
`LOOP_INTERVAL_MINUTES` to 30.

### LLM returns invalid JSON
Usually a model fluke. The system retries automatically. If persistent,
try a more capable model (`gpt-4o` instead of `gpt-4o-mini`).

### Orchestrator ran but no trades were taken
Normal — the PM profiles have strict rules. On low-conviction days they will
often pass entirely. Check `python portfolio_inspect.py feedback` for PM notes on why.

### Pi service not starting
```bash
sudo journalctl -u paper-trader -n 50
```
Most common causes:
- `.env` missing API keys
- Python version < 3.10 (`python3 --version`)
- Wrong working directory in service file (check `WorkingDirectory` in `deploy/paper-trader.service`)

### Daily P&L tab shows zeros
The daily log is written at 4:15 PM EOD. If the Reviewer errors on a given day, the bookkeeper still runs and saves the log. If you see zeros for past dates, those EOD runs failed before the fix — they can't be backfilled. Going forward the log will populate correctly.

### Check what happened today
```bash
python portfolio_inspect.py trades
python portfolio_inspect.py feedback
tail -100 logs/orchestrator.log
```
