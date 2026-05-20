# Agent Data Quality Audit — 2026-05-19

## Why this audit happened

The decision log showed several signs that the PM agents were not reasoning from clean current state:

- Conservative profile appeared absent from decision logs.
- PM notes referenced missing or vague trade information.
- Shadow ledger existed but had no live rows yet.
- PM notes referenced non-existent XLE positions in critical condition.
- Analyst/PM behavior looked overly cautious, possibly because upstream context was bad.

The conclusion: the agents were not simply “being cautious.” We found real data/context plumbing issues that could make them look crippled.

## Fixes completed today

### 1. Confirmed shadow ledger machinery exists, but live capture is not yet proven

Status: partially verified.

Findings:

- Pi had the shadow ledger commits deployed:
  - `e0d7981 feat: add shadow blocked trade ledger (Phase 0 capture layer)`
  - `4ced087 Blocker 1: Replaced db.commit() with db.flush() in record_blocked_candidate()`
- DB table exists: `blocked_trade_candidates`.
- Temp DB smoke test inserted a blocked candidate successfully.
- Live table had `0` rows because no PM reject/block path had fired yet during the checked window.

Interpretation:

- Code/schema work.
- Live blocked-candidate capture still needs proof from an actual blocked entry cycle.

### 2. Identified conservative profile was silently skipped, not necessarily broken

Findings from today’s live signals:

- 182 analyst signals checked.
- 179 were `HOLD`.
- Only 3 were `LONG`.
- 0 were `strong`.
- Conservative requires stronger eligible signals, so it skipped entry LLM by design.

Problem:

- The skip path did not create a useful conservative decision note.
- Result: conservative looked absent rather than explicitly cautious.

Needed follow-up:

- Add explicit “skipped because no eligible signals” decision log entries per profile.

### 3. Fixed intraday session contamination / wrong-day technical context

Commit: `5b0f2f8 fix: reset intraday context to current trading session`

Root cause:

- `get_candles(days=2)` intentionally returns yesterday + today for indicator warmup.
- `compute_indicators()` was calculating VWAP cumulatively across both sessions.
- This polluted today’s intraday context with yesterday’s candles.

Example before fix:

- AMD mixed 2-day VWAP: ~`413.84`
- AMD true current-session VWAP: ~`406.08`

That is a large enough error to distort analyst/PM reasoning.

Fix:

- VWAP now resets by candle session date.
- Technical indicators now expose:
  - `session_date`
  - `session_open`
  - `session_high`
  - `session_low`
  - `prior_session_high`
  - `prior_session_low`
  - `prior_session_close`
- Analyst signal enrichment now adds deterministic:
  - `session_date`
  - `prior_day_high`
  - `prior_day_low`
  - `prior_day_close`
- Analyst `key_levels.prior_high/prior_low` are overwritten with candle-derived prior-session levels instead of LLM guesses.

Verification:

- Deployed to Pi.
- `py_compile` passed.
- `paper-trader.service` restarted and active.
- Live sanity check showed AMD session VWAP around `406.08`, with prior-day levels separated correctly.

### 4. Fixed stale position-health contamination in PM entry context

Commit: `2165602 fix: prevent stale position health in PM entry context`

Root cause:

- PM entry prompt injected the latest global `position_health` memory unconditionally.
- Live `positions` table was empty.
- Live open trades were empty.
- But PM notes referred to critical XLE positions because the latest health memory was stale and global.

Impact:

- PM thought non-existent XLE positions were in critical condition.
- This contaminated new-entry reasoning and caused false caution.

Fix:

- If a profile has no open positions, PM now receives explicit no-position health context.
- If positions exist, PM only receives health assessments that are:
  - fresh within 2 hours,
  - matching that PM profile,
  - matching currently open symbols.

Verification:

- Deployed to Pi.
- `py_compile` passed.
- `paper-trader.service` restarted and active.

## Remaining known data gaps

### A. Relative volume is unreliable

Observed:

- 47 / 182 signals had `relative_volume = 0`.

Risk:

- Volume confirmation gates and analyst confidence are weakened.
- Agents may downgrade valid setups or fail to distinguish real moves from noise.

Next step:

- Compute relative volume against same-time historical intraday volume, or at least against completed current-session 5-minute bars excluding the current partial bar.

### B. Symbol classification is incomplete

Observed:

- 56 / 182 signals had `symbol_class = unknown`.
- This mostly affects normal single stocks like AMD/MSFT/NVDA.

Risk:

- “Unknown” makes downstream validation look uncertain even when the ticker is just a regular stock.

Next step:

- Change classifier fallback from `unknown` to `single_stock` for symbols not in known ETF/index sets.
- Reserve `unknown` for malformed/unsupported symbols.

### C. Premarket and opening range fields are missing

Missing fields:

- `premarket_high`
- `premarket_low`
- `premarket_gap_pct`
- `premarket_volume`
- `opening_range_high`
- `opening_range_low`
- `opening_range_break_direction`

Risk:

- The agents cannot reason cleanly about gap-and-go, ORB, failed breakouts, or opening range reclaim/rejection.

Next step:

- Add deterministic premarket/opening-range computation from intraday candles and inject it into analyst signals.

### D. Analyst does not provide executable trade geometry

Missing by design:

- `entry_price`
- `stop_loss`
- `target_price`
- `risk_reward`

Current design says PM should decide this, not Analyst. That is still reasonable, but PM needs deterministic candidate levels.

Next step:

- Add a deterministic “trade geometry scaffold” for PM:
  - candidate entry zones,
  - invalidation levels,
  - nearest resistance/support targets,
  - computed R:R ranges.
- PM can still decide whether to trade, but should not invent geometry from prose.

### E. Market and sector context is too thin

Missing fields:

- market breadth,
- SPY/QQQ regime score,
- sector relative strength,
- symbol vs sector/index relative strength,
- risk-on/risk-off score.

Risk:

- PM may overfit single-symbol technicals without knowing whether the broader tape supports the trade.

Next step:

- Add a market context packet each cycle and include it in Analyst + PM prompts.

### F. Liquidity / microstructure context is missing

Missing fields:

- bid,
- ask,
- spread,
- avg volume,
- slippage estimate,
- liquidity guard.

Risk:

- PM can size or enter low-quality setups without knowing whether execution is realistic.

Next step:

- Add a lightweight liquidity gate before PM execution.

## Recommended next implementation order

### Step 1 — Observability first

Add decision-log entries for every profile every cycle, including skip reasons.

Target output examples:

- Conservative skipped: no non-HOLD signals above `strong` threshold.
- Moderate skipped: XLE eligible but rejected because no valid geometry.
- Aggressive skipped: stale catalyst / volume confirmation failed.

Why first:

- We need the system to explain silence.
- No more “nothing happened, maybe it’s broken” ambiguity.

### Step 2 — Verify new session fields in next live analyst refresh

Check next `signal_seen` payloads for:

- `session_date`
- `prior_day_high`
- `prior_day_low`
- `prior_day_close`
- corrected `key_levels.prior_high/prior_low`

Why:

- Confirms today’s fix is flowing into live memory/events, not just callable code.

### Step 3 — Fix symbol classification

Change default classification:

- known ETFs/indexes stay explicit,
- normal listed equities become `single_stock`,
- malformed/unsupported tickers become `unknown`.

Why:

- Removes unnecessary uncertainty from single-stock setup validation.

### Step 4 — Fix relative volume

Replace naive current-bar volume ratio with a more stable calculation.

Minimum acceptable version:

- use completed bars only,
- compare current session cumulative volume pace against prior session or recent average.

Better version:

- same-time-of-day historical relative volume.

Why:

- Volume confirmation is a core input for confidence and gates.

### Step 5 — Add premarket/opening-range context

Add deterministic fields:

- premarket high/low,
- regular-session open,
- first 5/15/30 minute range,
- gap percentage,
- reclaim/break status.

Why:

- Needed for ORB, gap-and-go, vwap reclaim, and failed-breakout logic.

### Step 6 — Add PM geometry scaffold

Before PM calls the LLM, generate structured candidate geometry from deterministic levels:

- current price,
- VWAP,
- day high/low,
- prior high/low,
- ATR,
- nearest support/resistance,
- candidate stop/target levels,
- possible R:R.

Why:

- PM should choose and justify trades, not hallucinate price geometry.

### Step 7 — Expand shadow ledger

Current shadow ledger only captures certain blocked PM candidates.

Expand it to capture:

- analyst signals filtered before PM,
- PM skipped cycles,
- gate rejections,
- hypothetical outcome tracking.

Why:

- This lets us quantify whether the system is avoiding bad trades or missing winners.

## Current confidence assessment

The system is not hopeless. The core architecture can work, but today exposed that the agents are only as good as their context packet.

The two biggest bugs found today were not model-quality problems:

1. Wrong intraday session context from mixed-day VWAP.
2. Stale position-health memory contaminating flat portfolio decisions.

Those are plumbing bugs, and they are fixable. The next phase should focus less on adding more “intelligence” and more on making the data contract deterministic, auditable, and impossible for stale ghosts to leak into live decisions.
