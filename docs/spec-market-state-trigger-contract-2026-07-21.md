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

