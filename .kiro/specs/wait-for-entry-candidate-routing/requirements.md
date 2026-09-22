# Requirements Document

## Introduction

The PM candidate pipeline currently treats some non-immediate trade ideas as
immediate execution candidates. That is correct for a true breakout candidate
whose entry condition is already active and whose target still has reward left.
It is wrong for conditional setups such as support bounces, pullbacks to VWAP,
resistance rejections, and fades. Those are plans that should wait for price to
come to the entry condition.

The 2026-09-21 META missed move exposed the gap. The system built an aggressive
`gap_and_go` / `support_bounce` BUY candidate with entry below market. By the
time execution refreshed price, META was already beyond the candidate target,
so execution failed with stale-entry / target-crossed messaging. Recent safety
guards now block candidates whose target is already crossed, but that is only a
backstop. The structural fix is to route conditional setup geometry into a
wait-for-entry watch or pending-order lifecycle instead of offering it to PM as
an immediate executable trade.

## Non-Goals

- Do not loosen risk, sizing, exposure, or market-regime gates.
- Do not let PM-authored prices override deterministic geometry.
- Do not create broker/live-order behavior beyond the existing paper-trader
  pending-order and setup-watch abstractions.
- Do not replace the existing `pm_candidates`, `setup_watches`,
  `watch_candidates`, or `pending_orders` tables unless reuse proves unsafe.
- Do not classify a never-triggered setup as an execution failure.

## Glossary

- **Immediate candidate**: A candidate whose entry trigger is active now, whose
  target has not already been crossed, and whose geometry can be executed after
  normal PM/risk/preflight approval.
- **Conditional setup candidate**: A valid trade idea whose entry condition is
  not active yet. It needs a setup watch or pending order before execution.
- **Entry trigger**: The price/structure condition that makes a conditional
  setup executable, such as touching a pullback level or reclaiming VWAP.
- **Missed move**: Price reaches or crosses the target before the entry trigger
  becomes executable.
- **Invalidation**: Price or structure violates the thesis before entry.
- **Triggered watch**: A conditional setup whose entry trigger became active and
  is ready for fresh geometry, preflight, and PM or deterministic promotion.

## Requirements

### Requirement 1: Classify candidate execution mode before PM offering

**User Story:** As the system operator, I want every deterministic scaffold to
declare whether it is immediately executable or must wait for entry, so PM is
not asked to accept trades that are not currently enterable.

#### Acceptance Criteria

1. WHEN deterministic geometry is built, THE system SHALL assign
   `execution_mode` as either `immediate` or `wait_for_entry`.
2. IF a BUY candidate's current price has already crossed or equaled its target,
   THEN the system SHALL NOT offer it as an immediate PM candidate.
3. IF a SHORT candidate's current price has already crossed or equaled its
   target, THEN the system SHALL NOT offer it as an immediate PM candidate.
4. IF a candidate's setup type is `support_bounce`, `pullback_to_vwap`,
   `resistance_rejection`, or `fade`, THEN the default execution mode SHALL be
   `wait_for_entry` unless an explicit setup-specific trigger says the entry is
   active now.
5. IF a candidate is `breakout_continuation` or `breakdown_continuation`, THEN
   it MAY be `immediate` only when the breakout/breakdown trigger is active now
   and the target still has reward left.
6. IF the system cannot determine whether the trigger is active, THEN it SHALL
   fail closed into `wait_for_entry` or no-offer, not immediate execution.
7. Candidate telemetry SHALL include the classification reason, current price
   used for classification, entry price, target price, and trigger status.

### Requirement 2: Route conditional setups into setup-watch or pending-order lifecycle

**User Story:** As the system operator, I want valid conditional ideas to remain
observable while waiting for the right entry price instead of disappearing or
being executed immediately.

#### Acceptance Criteria

1. WHEN `execution_mode == wait_for_entry`, THE candidate builder SHALL NOT
   register the scaffold as an immediate `pm_candidates` row.
2. WHEN `execution_mode == wait_for_entry`, THE system SHALL create or update a
   setup watch or pending-order-style record with explicit trigger,
   invalidation, target, expiry, profile, and source signal lineage.
3. THE implementation SHOULD reuse the existing `setup_watches` and
   `pending_orders` infrastructure before introducing a new table.
4. IF existing infrastructure cannot safely represent this lifecycle, THEN the
   design SHALL document why and add the smallest compatible schema extension.
5. Conditional setup records SHALL be idempotent by symbol, profile, direction,
   setup type, source signal, and active cycle where appropriate.
6. Conditional setup records SHALL expire using existing TTL/session rules or a
   clearly configured setup-specific expiry.
7. The system SHALL prevent a conditional setup from being both watched and
   offered for immediate execution in the same cycle.

### Requirement 3: PM prompt must distinguish immediate candidates from watches

**User Story:** As the system operator, I want PM judgment to approve or reject
the plan without pretending the plan is executable before entry conditions exist.

#### Acceptance Criteria

1. PM prompt content SHALL separate immediate executable candidates from
   conditional watches.
2. PM MAY accept, reject, or decline to arm a conditional watch.
3. PM acceptance of a conditional watch SHALL NOT create an immediate trade.
4. PM acceptance of a conditional watch SHALL move the setup to an armed watch
   state, or SHALL leave an existing armed watch unchanged with audit telemetry.
5. PM response schema SHALL not allow PM to invent entry, stop, target, or
   quantity for either immediate candidates or watches.
6. PM rejection telemetry for watches SHALL use bounded reason codes and remain
   distinguishable from immediate-candidate rejection.

### Requirement 4: Triggered watches must rebuild and revalidate geometry

**User Story:** As the system operator, I want a watched setup to execute only
after current market data confirms the entry and all gates still pass.

#### Acceptance Criteria

1. WHEN a watch trigger becomes active, THE system SHALL rebuild or refresh
   geometry from current market data before execution.
2. The triggered path SHALL rerun deterministic preflight, reward/risk checks,
   market-regime/context checks, sizing, exposure, and stale-price validation.
3. A triggered watch SHALL NOT execute using stale scaffold prices without a
   fresh quote check.
4. IF refreshed geometry no longer meets profile gates, THEN the watch SHALL
   terminate with a gate-specific reason, not `execution_failed`.
5. IF refreshed geometry passes all gates, THEN the system MAY promote to a PM
   candidate or create a pending order according to the existing profile mode.
6. The triggered path SHALL preserve lineage from original signal, watch, PM
   watch decision if any, triggered geometry, and final trade or rejection.

### Requirement 5: Missed moves and invalidations are terminal watch outcomes

**User Story:** As the system operator, I want the ledger to say when a move was
missed because price ran away before entry, rather than calling it a failed
execution.

#### Acceptance Criteria

1. IF price crosses the target before the entry trigger becomes active, THEN the
   watch SHALL terminate as `missed_move` or `target_already_crossed`.
2. IF price violates the setup invalidation before entry, THEN the watch SHALL
   terminate as `invalidated`.
3. IF the watch expires before trigger, THEN it SHALL terminate as `expired`.
4. `execution_failed` SHALL be reserved for failures after an actual executable
   trigger and execution attempt.
5. Shadow Ledger and dashboard APIs SHALL display `Missed Move`, `Invalidated`,
   `Expired`, `Waiting for Entry`, and `Triggered` distinctly.
6. Shadow/outcome scoring SHALL not create incomplete-geometry rows for
   pre-entry watches that never reached an executable trigger.

### Requirement 6: Preserve immediate breakout behavior

**User Story:** As the system operator, I want true immediate continuation trades
to remain available, so the fix does not make the system permanently passive.

#### Acceptance Criteria

1. Existing immediate PM candidate behavior SHALL continue for valid active
   breakout/breakdown continuation setups.
2. Immediate candidates SHALL still pass target-crossed guards, stale-price
   guards, preflight, risk, sizing, and portfolio gates.
3. The change SHALL not reduce immediate candidate count merely because the
   setup-watch feature flags are enabled.
4. Tests SHALL prove at least one active breakout candidate still registers as
   an immediate PM candidate.

### Requirement 7: Regression coverage for the META missed move

**User Story:** As the system operator, I want the exact failure mode from
2026-09-21 covered so it cannot return under a different label.

#### Acceptance Criteria

1. A regression fixture SHALL represent META aggressive `gap_and_go` /
   `support_bounce` with current price above entry and target already crossed.
2. The fixture SHALL NOT create an immediate PM candidate.
3. The fixture SHALL produce either a terminal `missed_move` / target-crossed
   watch outcome or a deterministic no-offer event with equivalent reason.
4. The fixture SHALL NOT produce `execution_failed` unless an executable trigger
   and execution attempt actually occurred.
5. The fixture SHALL assert that PM compute was not the root cause; stale or
   conditional geometry routing was.
