# Spec: Market State and Conditional Trigger Contract

Date: 2026-07-21
Status: Proposed
Owner: Paper Trader

## Problem

The paper trader currently treats mixed technical evidence as `unclear_direction`.
That is safe, but it is too vague for both operator review and downstream PM
decision-making. A lack of signal can be high-value information, but the system
does not yet name the environment, explain the hierarchy of conflicting
timeframes, or express the conditional boundaries that would convert a flat
posture into an actionable setup.

Recent focused cycles showed the gap clearly:

- A symbol can show `breakout_confirmed` while the Analyst still outputs
  `HOLD / unclear_direction`.
- Higher-timeframe resistance or bearish daily context can suppress an
  intraday bullish move, but the current signal shape does not name that as a
  specific state.
- PM receives useful key levels and trigger state, but not a complete
  if-then contract describing what would activate, invalidate, or reclassify
  the setup.
- The dashboard can display `HOLD`, `FOCUS`, and trigger status, but not the
  broader market state that explains why flatness is intentional.

The result is operationally safe but semantically weak: the bot avoids many bad
trades, but the operator and PM have to infer whether the system is waiting for
confirmation, rejecting a chase, or seeing genuine indecision.

## Proposed Fix

Introduce a deterministic-plus-analyst contract that names market state,
enforces timeframe authority, reclassifies conflicting intraday moves, and
emits explicit conditional trigger boundaries.

This contract should clarify:

- What environment the symbol is currently in.
- Which timeframe has authority over the setup.
- Whether an intraday move is trend-aligned or counter-trend.
- What exact price/volume conditions would activate a long, activate a short,
  keep the system flat, or invalidate the current thesis.
- How PM should treat those fields without bypassing existing trade gates.

The goal is not to make PM trade more often. The goal is to make flatness and
conditional readiness explicit.

## Requirements

### 1. Market State Classification

The system shall assign each analyzed symbol a structured `market_state`.

At minimum, the supported states shall include:

- `trend_aligned_breakout`
- `breakout_extended`
- `breakout_retest_watch`
- `compression_under_resistance`
- `counter_trend_retracement_under_resistance`
- `range_bound_churn`
- `confounded`
- `risk_off_suppression`
- `pullback_validating`
- `pullback_failed`

The state shall be based on deterministic technical inputs where possible:

- current price
- VWAP
- support
- resistance
- day high / day low
- prior high / prior low
- higher-timeframe trend
- intraday trend
- RSI / MACD or equivalent momentum fields
- volume or relative-volume confirmation when available
- market regime / macro-risk context when available

The state shall be stored in the Analyst signal payload and returned by the
dashboard API.

### 2. Timeframe Authority

The system shall add a structured `timeframe_authority` field to the Analyst
signal payload.

The field shall identify:

- `higher_timeframe_trend`
- `intraday_trend`
- `authority`
- `conflict`
- `reason`

The `authority` value shall describe which timeframe currently controls trade
interpretation. Allowed values shall include:

- `higher_timeframe`
- `intraday`
- `aligned`
- `confounded`

If higher-timeframe trend is bearish while intraday trend is bullish, the
system shall not treat the intraday move as a standalone long thesis. It shall
classify the move as a counter-trend retracement or compression state unless
explicit breakout/reclaim conditions are met.

Example:

```json
{
  "higher_timeframe_trend": "bearish",
  "intraday_trend": "bullish",
  "authority": "higher_timeframe",
  "conflict": true,
  "reason": "Intraday strength is occurring below higher-timeframe resistance."
}
```

### 3. Setup Reclassification

The system shall add a `setup_reclassification` field when raw technical inputs
would otherwise produce an ambiguous or misleading setup label.

The field shall include:

- `original_setup_type`
- `reclassified_setup_type`
- `reason`
- `trade_posture`

Allowed `trade_posture` values shall include:

- `flat`
- `watch_long_trigger`
- `watch_short_trigger`
- `watch_retest`
- `eligible_for_pm_review`
- `veto_long`
- `veto_short`

Example:

```json
{
  "original_setup_type": "technical_breakout",
  "reclassified_setup_type": "counter_trend_retracement_under_resistance",
  "reason": "Price is above intraday VWAP but below higher-timeframe resistance with bearish daily trend.",
  "trade_posture": "watch_retest"
}
```

### 4. Conditional If-Then Triggers

The system shall emit an `if_then_triggers` array for each analyzed symbol.

Each trigger shall include:

- `id`
- `condition`
- `threshold`
- `confirmation`
- `then`
- `trade_posture`
- `invalidates`

Triggers shall be quantitative wherever data exists.

Example:

```json
[
  {
    "id": "long_breakout_activation",
    "condition": "price_breaks_above_resistance",
    "threshold": 104.58,
    "confirmation": "volume >= 1.5 standard deviations above baseline or relative_volume >= configured threshold",
    "then": "higher_timeframe_resistance_invalidated; long breakout watch activates",
    "trade_posture": "watch_long_trigger",
    "invalidates": "flat_due_to_compression_under_resistance"
  },
  {
    "id": "long_veto",
    "condition": "price_cracks_below_vwap",
    "threshold": 102.88,
    "confirmation": "sustained below VWAP or close below VWAP on configured intraday bar",
    "then": "long thesis remains vetoed; possible short watch toward support",
    "trade_posture": "veto_long",
    "invalidates": "breakout_retest_watch"
  }
]
```

### 5. PM Interpretation

PM shall consume `market_state`, `timeframe_authority`,
`setup_reclassification`, `trigger_status`, and `if_then_triggers` as
decision context.

These fields shall not override existing PM entry gates.

PM shall not place a new trade solely because:

- `trigger_status.entry_trigger == breakout_confirmed`
- `market_state == trend_aligned_breakout`
- an if-then trigger has a `watch_*` posture

PM may only consider a new entry when existing requirements are also satisfied:

- Analyst signal is directional, not `HOLD`.
- Setup type is executable or explicitly mapped to an executable setup.
- Strength meets profile threshold.
- Candidate geometry exists.
- Risk, quality, and portfolio gates pass.

When `trade_posture` is `watch_retest`, PM shall remain flat unless a later
cycle emits an executable directional signal and a valid candidate geometry.

### 6. Dashboard Behavior

The dashboard shall display the current market state in a compact badge or
label.

The label shall make intentional flatness legible to the operator.

Examples:

- `Market State: Compression Under Resistance`
- `Market State: Counter-Trend Bounce`
- `Market State: Range-Bound Churn`
- `Market State: Breakout Extended`
- `Market State: Risk-Off Suppression`

When the system is flat, the dashboard shall prefer explicit explanatory state
over vague `unclear_direction` language where possible.

The dashboard shall expose conditional trigger summaries in a compact format.

Example:

`IF > 104.58 with volume confirmation -> long watch; IF < VWAP -> long veto`

### 7. Analyst Output Contract

The Analyst prompt/schema shall require:

- `market_state`
- `timeframe_authority`
- `setup_reclassification`
- `if_then_triggers`
- `veto_reason`
- `activation_conditions`
- `invalidation_conditions`

Analyst prose may explain these fields, but the fields themselves shall be
structured and parseable.

If the Analyst emits `HOLD` while deterministic trigger state is bullish, the
Analyst shall provide a structured veto reason.

Allowed structured veto reason categories shall include:

- `higher_timeframe_resistance`
- `counter_trend_move`
- `extended_from_vwap`
- `thin_volume`
- `bearish_market_regime`
- `mixed_momentum`
- `stale_or_missing_catalyst`
- `risk_reward_unfavorable`

### 8. Safety Requirements

The implementation shall be fail-closed for trading authority.

Missing, malformed, or contradictory market-state fields shall not create trade
eligibility.

When fields conflict, PM shall prefer the more conservative interpretation.

The system shall preserve existing behavior that filters out `HOLD` signals
before PM new-entry decisioning.

The system shall log market-state and reclassification decisions for later CEO
and reviewer analysis.

### 9. Observability Requirements

The system shall make market-state decisions inspectable through at least one
operator-facing surface.

Acceptable surfaces include:

- dashboard row data
- decision log metadata
- trade event audit payloads
- AgentMemory signal payloads

For each focused symbol, the operator shall be able to answer:

- Why is the system flat?
- What would activate a long?
- What would activate a short or veto the long?
- Which timeframe currently has authority?
- Is the current move trend-aligned or counter-trend?

## Non-Goals

This spec does not require PM to trade more often.

This spec does not authorize deterministic trigger state to override Analyst
direction or PM gates.

This spec does not require a new LLM call.

This spec does not require a full strategy rewrite.

This spec does not require changing broker/execution behavior.

## Acceptance Criteria

The work is acceptable when:

1. Analyst signal payloads include structured market-state and if-then trigger
   fields for each analyzed symbol.
2. PM prompt context includes those fields for any eligible entry signal.
3. PM continues to reject or ignore `HOLD` signals for new entries.
4. Dashboard shows a human-readable market-state label for focused symbols.
5. A counter-trend intraday move under higher-timeframe resistance is
   reclassified away from standalone long logic.
6. A breakout that is confirmed but extended from VWAP is represented as a
   watch/retest state, not an automatic buy.
7. A compression/range-bound/confounded state explains flatness with clear
   activation and invalidation levels.
8. Tests cover at least:
   - higher-timeframe bearish + intraday bullish reclassification
   - breakout confirmed but extended from VWAP
   - breakout approaching resistance
   - pullback failed below VWAP/support
   - malformed market-state payload fails closed
9. Live dashboard/API verification can show, for a focused symbol, current
   `market_state`, `timeframe_authority`, and conditional trigger summary.

## Amendment: Setup Lifecycle and Watch Candidate Contract

### Problem

Focused-symbol trading exposed a second gap beyond vague `unclear_direction`
labels: the system does not distinguish where a symbol is in the setup
lifecycle.

For example, a focused symbol may show strong intraday action, sector
confirmation, and relative strength while the higher-timeframe chart remains
bearish or capped by resistance. The correct response may be neither "buy" nor
"ignore"; it may be "watch for reclaim", "wait for retest", or "track a
conditional long activation."

Today, those cases often collapse into `HOLD / unclear_direction`. That is safe
for execution, but it loses valuable timing information:

- The system cannot explain whether a setup is early, late, invalidated, or
  waiting for confirmation.
- PM does not receive a non-trading watch object to monitor.
- Reviewer/CEO cannot later score missed AMD/MU-style opportunities because no
  structured watch candidate was recorded.
- The dashboard cannot show that the system identified the right symbol but was
  waiting for a specific activation boundary.

### Proposed Fix

Introduce a structured setup lifecycle layer between Analyst signal generation
and PM trade candidate creation.

The lifecycle layer shall convert mixed timeframe and trigger evidence into
explicit non-trading watch states before any trade authority is granted.

The intended flow is:

```text
market data -> Analyst/trigger state -> setup lifecycle -> watch candidate
             -> activation event -> PM trade candidate -> PM gates -> trade
```

The goal is to make conditional opportunity tracking observable and reviewable
without making PM more aggressive.

### Lifecycle States

Each focused analyzed symbol shall receive a `setup_lifecycle_state`.

Allowed initial states shall include:

- `no_setup`
- `early_watch`
- `compression_watch`
- `breakout_watch`
- `breakout_confirmed_wait_retest`
- `pullback_watch`
- `pullback_validating`
- `activation_pending`
- `activated_for_pm_review`
- `invalidated`
- `expired`

The lifecycle state shall be derived from structured fields including:

- `market_state`
- `timeframe_authority`
- `setup_reclassification`
- `trigger_status`
- `if_then_triggers`
- volume/relative-volume confirmation when available
- VWAP/support/resistance distance
- higher-timeframe trend and resistance context
- sector/breadth/relative-strength context

`setup_lifecycle_state` shall not replace Analyst `signal`. It shall describe
setup maturity and timing.

### Timeframe Role Assignment

The system shall treat timeframes as distinct decision roles:

- Higher timeframe controls regime and trade permission.
- Intermediate timeframe controls setup structure.
- Intraday timeframe controls execution timing.
- Sector, breadth, and relative strength control tailwind/headwind context.

When these roles disagree, the lifecycle layer shall not classify the symbol as
generically unclear if a more precise conditional state applies.

Examples:

- Higher timeframe bearish + intraday bullish + below resistance:
  `counter_trend_retracement_under_resistance` and `breakout_watch` or
  `compression_watch`.
- Higher timeframe bearish + intraday bullish + confirmed reclaim over
  resistance: `activation_pending` or `activated_for_pm_review`, subject to
  Analyst direction and PM gates.
- Higher timeframe bullish + intraday pullback to VWAP/support:
  `pullback_validating`.
- Breakout confirmed while extended from VWAP: `breakout_confirmed_wait_retest`.

### Watch Candidates

The system shall create a non-trading `watch_candidate` record when a focused
symbol has a meaningful conditional opportunity but does not yet satisfy trade
entry requirements.

A watch candidate shall include:

- `watch_id`
- `symbol`
- `created_at`
- `expires_at`
- `source_cycle_id`
- `market_state`
- `setup_lifecycle_state`
- `timeframe_authority`
- `direction_watch`
- `trade_posture`
- `activation_conditions`
- `invalidation_conditions`
- `key_levels`
- `trigger_status`
- `reason`
- `source_signal_snapshot`

Allowed `direction_watch` values shall include:

- `long`
- `short`
- `two_sided`
- `none`

Allowed `trade_posture` values shall include:

- `watch_only`
- `watch_long_breakout`
- `watch_short_breakdown`
- `watch_retest`
- `watch_pullback_hold`
- `invalidated`

Watch candidates shall be explicitly non-executable. They shall not be inserted
into `pm_candidates` unless activation and existing PM eligibility requirements
are satisfied.

### Activation and Invalidation

Each watch candidate shall define quantitative activation and invalidation
conditions.

Activation examples:

```json
{
  "activation_conditions": [
    {
      "id": "long_reclaim_resistance",
      "condition": "price_above",
      "threshold": 544.65,
      "confirmation": "configured close/hold rule and volume confirmation",
      "then": "activated_for_pm_review"
    }
  ],
  "invalidation_conditions": [
    {
      "id": "long_vwap_failure",
      "condition": "price_below",
      "threshold": 533.82,
      "confirmation": "sustained below VWAP/support",
      "then": "invalidated"
    }
  ]
}
```

The implementation shall support activation in observe-only mode before any
live PM promotion is enabled.

### PM Promotion Rules

Watch candidates may be promoted to PM review only when all required conditions
are true:

1. The watch candidate is active and unexpired.
2. Activation conditions have been met.
3. Analyst signal is directional, not `HOLD`, or a future explicitly approved
   observe-to-PM promotion mode is enabled.
4. Setup type is executable or mapped to an executable setup.
5. Candidate geometry exists with valid entry, stop, target, and risk/reward.
6. Existing PM quality, risk, profile, and portfolio gates pass.

This amendment does not authorize watch candidates to bypass PM gates.

When activation occurs while Analyst remains `HOLD`, the system shall record an
activation event and keep the candidate in observe-only state unless a separate
future requirement explicitly changes that behavior.

### Shadow Outcome Tracking

The system shall record shadow outcomes for watch candidates.

At minimum, the system shall capture:

- activation time, if any
- hypothetical entry level
- maximum favorable excursion
- maximum adverse excursion
- whether invalidation occurred before activation
- whether the Analyst remained `HOLD`
- whether PM would have accepted or rejected the activated candidate
- reason actual trading did or did not occur

Shadow outcomes shall be reviewable by the Reviewer and CEO agents.

The purpose is to learn whether missed opportunities were:

- correct avoids
- late/chase setups
- Analyst underconfidence
- PM gate overrestriction
- missing volume/timing confirmation
- data freshness or provider failures

### Dashboard Requirements

For focused symbols, the dashboard shall show:

- market state
- setup lifecycle state
- watch posture, when present
- activation level
- invalidation level
- whether the setup is watch-only, activated, invalidated, or expired

Dashboard language shall make flatness intentional.

Examples:

- `Watch: Long breakout over 544.65`
- `State: Counter-trend rally under daily resistance`
- `Posture: Wait for retest`
- `Invalidates below VWAP 533.82`

### Safety Requirements

Watch candidates are not trades.

Missing or malformed lifecycle fields shall fail closed.

Expired watch candidates shall not activate.

If activation and invalidation both appear true in the same evaluation window,
the system shall prefer the conservative interpretation and require a fresh
cycle before PM promotion.

No lifecycle state shall create broker/execution authority by itself.

### Additional Acceptance Criteria

The amendment is acceptable when:

1. Focused symbols can produce watch candidates while Analyst remains `HOLD`.
2. Watch candidates are stored separately from executable PM candidates.
3. PM does not trade from watch candidates unless promotion rules are satisfied.
4. A higher-timeframe bearish / intraday bullish symbol under resistance is
   classified as a counter-trend or compression watch instead of generic
   `unclear_direction`.
5. A confirmed breakout that is extended from VWAP becomes a retest watch, not
   an automatic entry.
6. The dashboard can show the current watch posture and activation/invalidation
   levels for a focused symbol.
7. Shadow outcome tracking can later answer whether an AMD/MU-style watch would
   have produced a favorable trade.
8. Observe-only mode can run without changing live trading behavior.
