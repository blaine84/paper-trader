# PM Entry Geometry Source-of-Truth Fix — Spec

Date: 2026-05-19
Status: proposed
Scope: one fix only — resolve the Analyst/PM conflict over who supplies entry, stop, and target prices.

## Problem statement

The system currently has a contract conflict around executable trade geometry.

The Analyst prompt explicitly says the Analyst must not decide:

- entry price or timing,
- stop placement,
- target,
- position size.

It also forbids Analyst output fields such as `entry_price`, `stop_loss`, `target`, and `quantity`.

However, the PM entry prompt and compact signal layer imply that Analyst signals include executable geometry. `compact_signal_for_pm()` currently emits:

```text
entry: ? / stop: ? / target: ?
```

because it looks for fields the Analyst is forbidden to provide. This causes PM decision logs such as:

> Analyst must provide concrete price levels before entry.

That diagnosis is wrong. The Analyst has provided market levels, but not executable trade geometry — by design.

This creates downstream problems:

- PM blames the Analyst for missing fields the Analyst contract forbids.
- PM sometimes passes valid setups because `entry/stop/target` appear unknown.
- PM sometimes invents geometry from prose and gets rejected by gates.
- Shadow ledger captures plumbing failures instead of meaningful trade-quality decisions.

## Goal

Create a single, explicit source of truth for candidate entry/stop/target geometry before PM decision-making.

The correct ownership model is:

1. Analyst owns market read and levels.
2. Deterministic geometry scaffold proposes executable candidate plans.
3. PM chooses, rejects, or modifies within profile rules.
4. Gates / StopAuthority validate legality and risk constraints.
5. Execution/monitoring systems own fills, stop triggers, target triggers, and lifecycle management.

## Non-goals

This fix does not:

- make the Analyst output final trades,
- allow PM to place trades without valid geometry,
- change StopAuthority rules,
- change profile risk thresholds,
- tune R:R requirements,
- alter shadow ledger schema except optional richer payload capture,
- decide whether XLE/AMD/etc. are good trades.

## Required behavior

### Requirement 1 — Analyst remains non-executing

The Analyst must continue to output only:

- `signal`: `LONG`, `SHORT`, or `HOLD`,
- `strength`,
- `confidence`,
- `setup_type`,
- `setup_reasoning`,
- `reasoning`,
- `key_levels`,
- `invalidation`,
- `indicators`,
- deterministic quote/session context added by code.

The Analyst must not output:

- `entry_price`,
- `stop_loss`,
- `target`,
- `quantity`,
- `portfolio_notes`,
- `decisions[]`.

### Requirement 2 — Remove false missing-geometry display from Analyst signal compaction

`compact_signal_for_pm()` must stop printing:

```text
entry: ? / stop: ? / target: ?
```

for raw Analyst signals.

Instead, compact Analyst context should show the fields the Analyst actually owns, such as:

```text
XLE: LONG (strong) | setup: trend_pullback | current_price: 61.32 | VWAP: 60.76 | support: 60.295 | resistance: 61.33 | invalidation: below VWAP/support
```

If no geometry scaffold is available, the prompt should explicitly say:

```text
No geometry scaffold available — PM must not trade this signal.
```

It should not imply that the Analyst failed to provide forbidden fields.

### Requirement 3 — Add deterministic geometry scaffold

Add a deterministic component that generates candidate executable trade plans from Analyst signal + quote/session context.

Suggested module:

```text
utils/entry_geometry.py
```

Suggested public function:

```python
def build_entry_geometry_scaffold(signal: dict, profile_id: str | None = None) -> dict:
    ...
```

The function must be deterministic and side-effect free.

It should return a structure like:

```json
{
  "symbol": "XLE",
  "direction": "LONG",
  "source": "deterministic_geometry_scaffold",
  "status": "ok|insufficient_data",
  "reason": null,
  "levels_used": {
    "current_price": 61.32,
    "vwap": 60.76,
    "support": 60.295,
    "resistance": 61.33,
    "day_high": 61.33,
    "day_low": 60.295,
    "prior_day_high": 60.70,
    "prior_day_low": 58.72,
    "atr": 0.42
  },
  "candidates": [
    {
      "name": "pullback_to_vwap",
      "entry_price": 60.80,
      "stop_loss": 60.25,
      "target": 61.90,
      "risk_reward": 2.0,
      "trigger": "price pulls back near VWAP and holds",
      "invalidation_basis": "below session support/VWAP buffer",
      "target_basis": "2R from stop distance"
    },
    {
      "name": "breakout_continuation",
      "entry_price": 61.35,
      "stop_loss": 60.75,
      "target": 62.55,
      "risk_reward": 2.0,
      "trigger": "break and hold above day high/resistance",
      "invalidation_basis": "failed breakout below VWAP/day-high reclaim",
      "target_basis": "2R from stop distance"
    }
  ]
}
```

### Requirement 4 — Candidate generation rules must be explicit

For `LONG` setups:

- stop must be below entry,
- target must be above entry,
- candidate must include computed R:R,
- candidate must identify whether it is a pullback, breakout, reclaim, or support-bounce plan.

For `SHORT` setups:

- stop must be above entry,
- target must be below entry,
- candidate must include computed R:R,
- candidate must identify whether it is a rejection, breakdown, fade, or failed-reclaim plan.

For `HOLD` setups:

- no executable candidates should be generated,
- status should be `insufficient_data` or `not_tradeable_signal`.

### Requirement 5 — Minimum data needed for scaffold

The scaffold may use fields including:

- `current_price`,
- `key_levels.support`,
- `key_levels.resistance`,
- `key_levels.vwap`,
- `key_levels.prior_high`,
- `key_levels.prior_low`,
- `day_high`,
- `day_low`,
- `prior_day_high`,
- `prior_day_low`,
- `indicators.atr` or top-level `atr` when available,
- `setup_type`,
- `signal`,
- `invalidation` text only as secondary context.

If required numeric fields are missing, the scaffold must fail closed:

```json
{
  "status": "insufficient_data",
  "reason": "missing_support_or_vwap",
  "candidates": []
}
```

### Requirement 6 — PM prompt must receive scaffold candidates, not raw question marks

PM entry prompt should include both:

1. Analyst read.
2. Geometry scaffold candidates.

Example compact prompt section:

```text
ANALYST SIGNAL:
XLE LONG strong trend_pullback, current 61.32, VWAP 60.76, support 60.295, resistance 61.33, invalid below VWAP/support.

GEOMETRY SCAFFOLD:
Candidate A pullback_to_vwap: entry 60.80, stop 60.25, target 61.90, R:R 2.0, trigger pullback holds VWAP.
Candidate B breakout_continuation: entry 61.35, stop 60.75, target 62.55, R:R 2.0, trigger break/hold above day high.
```

PM can then output:

- selected candidate name,
- final `entry_price`,
- final `stop_loss`,
- final `target`,
- `quantity`,
- rationale.

### Requirement 7 — PM remains decision owner

PM is not required to accept scaffold candidates.

PM may:

- choose a candidate,
- reject all candidates,
- adjust a candidate within profile rules,
- pass because profile threshold, market context, timing, volume, or catalyst quality is insufficient.

But if PM outputs a trade, it must output valid executable fields:

- `entry_price`,
- `stop_loss`,
- `target`,
- `quantity`,
- `setup_type`,
- `rationale`,
- optional `geometry_candidate`: candidate name/id.

### Requirement 8 — Gates remain final validators

Existing validation remains mandatory:

- PM entry normalizer validates required fields.
- RiskGeometryGate validates risk geometry.
- StopAuthority validates stop legality.
- Profile gates validate R:R, sizing, cooldowns, daily loss, etc.

The scaffold is not permission to trade. It is a deterministic proposal layer.

### Requirement 9 — Decision logs must name the real blocker

Decision logs must not say:

```text
Analyst must provide concrete price levels.
```

Instead, logs should distinguish:

- scaffold unavailable,
- scaffold generated but PM rejected candidates,
- PM selected candidate but gate rejected,
- PM passed due to profile/timing/market context.

Examples:

```text
No trade: geometry scaffold unavailable — missing numeric support/VWAP.
```

```text
No trade: scaffold candidates exist, but all R:R below conservative threshold.
```

```text
No trade: breakout candidate valid, but price extended into resistance and volume confirmation weak.
```

### Requirement 10 — Shadow ledger should capture scaffold-aware rejects

When a PM or gate rejects a scaffold candidate, shadow ledger should be able to record:

- `geometry_candidate_name`,
- `entry_price`,
- `stop_price`,
- `target_price`,
- `risk_reward`,
- rejection reason,
- gate/profile responsible,
- signal snapshot,
- scaffold snapshot.

This makes shadow review meaningful: it can evaluate real hypothetical trades instead of malformed or missing geometry.

## Proposed implementation plan

### Step 1 — Add `utils/entry_geometry.py`

Implement:

- numeric extraction helpers,
- R:R calculation,
- long candidate generation,
- short candidate generation,
- fail-closed missing data output.

Candidate types for v1:

- `pullback_to_vwap`,
- `support_bounce`,
- `breakout_continuation`,
- `resistance_rejection` for shorts,
- `breakdown_continuation` for shorts.

### Step 2 — Add tests

Suggested tests:

```text
tests/test_entry_geometry.py
```

Coverage:

- LONG scaffold produces only stop < entry < target.
- SHORT scaffold produces only target < entry < stop.
- HOLD produces no candidates.
- Missing support/VWAP/resistance fails closed.
- Candidate R:R is calculated correctly.
- Candidate names and basis fields are present.

### Step 3 — Update prompt compaction

Change `compact_signal_for_pm()` or add a new function:

```python
def compact_signal_and_geometry_for_pm(symbol: str, signal: dict, scaffold: dict) -> str:
    ...
```

Remove raw `entry: ? / stop: ? / target: ?` from Analyst-only signal display.

### Step 4 — Wire scaffold into PM entry cycle

In PM entry logic:

- after `entry_signals` filtering,
- build scaffold per signal,
- include scaffold in PM prompt,
- optionally attach scaffold to normalization/rejection context.

### Step 5 — Update PM prompt wording

Replace wording that says PM receives Analyst signals “with entry, stop, target.”

Use:

```text
You receive Analyst signals with market direction, setup type, confidence, key levels, and invalidation.
You also receive deterministic geometry scaffold candidates with possible entry/stop/target plans.
You own the final trade decision and must output executable geometry if trading.
```

### Step 6 — Improve decision log language

Update PM notes or post-processing so logs name the correct source of missing data.

Forbidden wording:

```text
Analyst must provide entry/stop/target.
```

Preferred wording:

```text
No executable geometry candidate met profile/gate requirements.
```

or

```text
Geometry scaffold unavailable due to missing numeric support/resistance/VWAP.
```

### Step 7 — Optional shadow ledger enrichment

If quick and safe, include scaffold payload in blocked candidate records.

If not, leave schema unchanged and store scaffold snapshot in existing JSON fields.

## Acceptance criteria

This fix is complete when:

1. Analyst still does not output final entry/stop/target fields.
2. PM prompt no longer displays `entry: ? / stop: ? / target: ?` for Analyst signals.
3. Every eligible non-HOLD signal has either:
   - at least one scaffold candidate, or
   - an explicit scaffold failure reason.
4. PM decision logs no longer blame Analyst for not providing forbidden geometry.
5. PM trade decisions include executable `entry_price`, `stop_loss`, and `target` selected from or justified against scaffold candidates.
6. Gate rejections and shadow ledger records contain meaningful geometry when available.
7. Tests cover long, short, hold, missing-data, and R:R calculations.

## Example: XLE current issue after fix

Given Analyst signal:

```json
{
  "symbol": "XLE",
  "signal": "LONG",
  "strength": "strong",
  "confidence": "high",
  "setup_type": "trend_pullback",
  "current_price": 61.32,
  "key_levels": {
    "support": 60.295,
    "resistance": 61.33,
    "vwap": 60.76,
    "prior_high": 60.70,
    "prior_low": 58.72,
    "day_high": 61.33,
    "day_low": 60.295
  },
  "invalidation": "Price closes below VWAP, loses support at 60.295"
}
```

The system should not say:

```text
Analyst must provide concrete price levels.
```

It should generate candidates like:

```text
A pullback_to_vwap: entry 60.80, stop 60.25, target 61.90, R:R 2.0
B breakout_continuation: entry 61.35, stop 60.75, target 62.55, R:R 2.0
```

Then PM should say one of:

```text
Selected pullback_to_vwap because it fits moderate profile and provides valid R:R.
```

or:

```text
Rejected breakout_continuation because price is extended into resistance and volume confirmation is weak.
```

That is the intended audit trail.

## Why this unlocks shadow ledger

Before this fix, shadow ledger mostly risks recording malformed candidates or missing geometry. That measures handoff failure, not trading judgment.

After this fix, shadow ledger can record real hypothetical trades:

- candidate entry,
- stop,
- target,
- R:R,
- rejection reason,
- later hypothetical outcome.

This lets us answer the question shadow mode was designed for:

> Are the gates preventing bad trades, or are they blocking winners?
