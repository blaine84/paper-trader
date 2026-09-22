# Design Document: Wait-for-Entry Candidate Routing

## Overview

The candidate pipeline needs a clear boundary between:

1. a trade idea,
2. an entry trigger,
3. an executable trade.

Today, deterministic geometry can turn a conditional setup into an immediate PM
candidate. That makes the system ask PM to accept trades before the entry price
or structure exists. When price has already moved beyond target, the eventual
failure looks like stale execution or compute latency even though the real issue
was lifecycle routing.

The design adds an execution-mode classifier before PM offering:

```text
Analyst signal
-> deterministic geometry scaffold
-> execution-mode classification
   -> immediate candidate path
   -> wait-for-entry watch/pending-order path
-> trigger monitoring
-> fresh geometry/preflight on trigger
-> PM/risk/execution path
```

Recent commit `e46b9ba` added target-crossed guards. Keep those guards as a
lower-level safety net; this spec moves the decision earlier and gives
conditional setups their own lifecycle.

## Existing Systems To Reuse

Kiro should inspect and prefer these existing modules before adding new schema:

- `utils/candidate_builder.py`
- `utils/entry_geometry.py`
- `utils/setup_watch_manager.py`
- `utils/setup_watch_registry.py`
- `utils/watch_candidates.py`
- `utils/pending_order_creation.py`
- `utils/pending_order_registry.py`
- `utils/pending_order_filler.py`
- `agents/portfolio_manager.py`
- `utils/shadow_outcomes.py`
- `web/app.py`
- tests around `setup_watches`, `pending_orders`, `watch_candidates`, and
  `candidate_builder`

There are two related watch systems:

- `watch_candidates`: market-state watch promotions into PM candidates.
- `setup_watches`: setup lifecycle watches with maturation, invalidation,
  promotion, and outcomes.

The preferred implementation is to route conditional setup geometry through
`setup_watches` and only use `pending_orders` when the order semantics are
already active and compatible with the setup. Avoid expanding the older
`watch_candidates` table unless it is the only practical hook.

## Execution Mode Classification

Add a small deterministic classifier close to geometry construction. Suggested
interface:

```python
class ExecutionMode(str, Enum):
    IMMEDIATE = "immediate"
    WAIT_FOR_ENTRY = "wait_for_entry"
    NO_OFFER = "no_offer"

def classify_execution_mode(candidate, signal_snapshot, current_price) -> dict:
    return {
        "execution_mode": "immediate" | "wait_for_entry" | "no_offer",
        "reason_code": "...",
        "entry_trigger_active": True | False | None,
        "target_already_crossed": True | False,
        "current_price": current_price,
    }
```

Initial mapping:

- `breakout_continuation`: immediate only when BUY price is at/above trigger
  and target is still above current price.
- `breakdown_continuation`: immediate only when SHORT price is at/below trigger
  and target is still below current price.
- `support_bounce`: wait for entry unless bounce trigger is active now; if
  target already crossed, no-offer/missed-move.
- `pullback_to_vwap`: wait for entry unless pullback trigger is active now.
- `resistance_rejection`: wait for entry unless rejection trigger is active now.
- `fade`: wait for entry unless fade trigger is active now.
- Unknown trigger state: wait-for-entry or no-offer, never immediate.

The classifier must use the freshest price available at classification time and
must persist the price source/timestamp in telemetry.

## Immediate Candidate Path

If `execution_mode == immediate`:

1. Register as `pm_candidates` as today.
2. Persist `execution_mode`, classifier reason, current price, and trigger
   status into candidate metadata/event telemetry.
3. Continue through deterministic preflight and PM candidate-id selection.
4. Keep final stale-price and target-crossed guards in execution.

## Wait-for-Entry Path

If `execution_mode == wait_for_entry`:

1. Do not register an immediate `pm_candidates` row.
2. Create or update a setup-watch record with:
   - source signal/candidate lineage,
   - symbol, profile, direction, setup type, geometry name,
   - entry trigger condition,
   - entry price or entry zone,
   - stop/invalidation condition,
   - target and missed-move condition,
   - expiry,
   - state and reason telemetry.
3. If PM approval is required to arm the watch, present it separately as a
   watch proposal. PM approval moves it to armed/watching; PM rejection records
   a bounded watch rejection.
4. Monitor the watch during normal cycles.
5. When trigger becomes active, rebuild/refresh geometry from current market
   data and rerun all gates before execution.

Do not let a watch become an immediate trade because the old scaffold had a
plausible entry/stop/target. The trigger must be true now.

## Terminal Outcomes

Use explicit terminal states:

- `missed_move`: target crossed before entry trigger.
- `invalidated`: thesis/invalidation breached before entry.
- `expired`: TTL/session expiry before trigger.
- `gate_rejected`: trigger occurred but current gates failed.
- `executed`: trigger occurred and trade was created.
- `execution_failed`: trigger occurred, gates passed, execution was attempted,
  and the execution layer failed.

Dashboard and Shadow Ledger should show these as operator-facing outcomes:

- `Waiting for Entry`
- `Triggered`
- `Missed Move`
- `Invalidated`
- `Expired`
- `Gate Rejected`
- `Execution Failed`

## META Regression

The fixture should model the 2026-09-21 META miss:

- profile: `aggressive`
- setup: `gap_and_go` / `support_bounce`
- candidate created around `2026-09-21T19:51:17Z`
- PM accepted around `2026-09-21T19:52:01Z`
- signal snapshot current price around `744.99`
- MTF price around `746.25`
- scaffold entry around `740.01`
- stop around `738.53`
- target around `742.23` to `744.45` depending adjustment layer
- fresh execution price around `747.17`

Expected behavior after this spec:

- no immediate PM candidate for the support-bounce geometry,
- no execution attempt,
- watch/no-offer telemetry records target already crossed or missed move,
- Shadow Ledger does not call it an execution failure,
- immediate breakout/continuation geometry may still be offered if independently
  valid and target still has room.

## Rollout

Use a feature flag, for example `WAIT_FOR_ENTRY_CANDIDATE_ROUTING`, with three
stages:

1. `disabled`: current behavior.
2. `observe`: classify and log what would have been routed to watch, but do not
   alter execution.
3. `enabled`: route conditional candidates to watches and remove them from
   immediate PM offering.

Metrics to monitor:

- immediate candidate count by setup type/profile,
- watch proposal count,
- armed watch count,
- trigger rate,
- missed-move rate,
- invalidation/expiry rate,
- stale-entry execution failures,
- PM accept/reject rate for immediate candidates and watches separately,
- time from signal to trigger to execution.

## Failure Behavior

- Classifier exceptions fail closed: do not offer immediate execution.
- Watch creation failures must write diagnostic telemetry and must not fall back
  to immediate execution.
- Trigger evaluation can fail open for observability, but execution must fail
  closed unless fresh geometry and all gates are available.
- Existing target-crossed execution guards remain mandatory backstops.
