# Implementation Plan

- [ ] 1. Inspect existing setup-watch and pending-order contracts
  - Read `utils/setup_watch_manager.py`, `utils/setup_watch_registry.py`,
    `utils/pending_order_creation.py`, `utils/pending_order_registry.py`,
    `utils/candidate_builder.py`, and their tests.
  - Decide whether conditional setup routing can use `setup_watches` directly
    or needs a small schema/metadata extension.
  - Document the selected integration point before code changes.
  - _Requirements: 2.2, 2.3, 2.4_

- [ ] 2. Add deterministic execution-mode classification
  - Implement a classifier for `immediate`, `wait_for_entry`, and `no_offer`.
  - Include target-crossed, trigger-active, current-price, and reason-code
    telemetry.
  - Fail closed to non-immediate behavior on unknown trigger status.
  - _Requirements: 1.1-1.7_

- [ ] 3. Route wait-for-entry candidates away from immediate PM offering
  - Prevent `wait_for_entry` scaffolds from registering as immediate
    `pm_candidates`.
  - Create/update setup-watch or pending-order-compatible records instead.
  - Enforce idempotency and prevent double booking in the same cycle.
  - _Requirements: 2.1, 2.2, 2.5, 2.7_

- [ ] 4. Split PM prompt and response handling for watches
  - Show immediate executable candidates separately from conditional watches.
  - Allow PM to accept/reject/decline-to-arm watches without creating trades.
  - Persist bounded watch rejection/approval telemetry.
  - _Requirements: 3.1-3.6_

- [ ] 5. Rebuild and revalidate geometry on watch trigger
  - On trigger, refresh current price and rebuild geometry.
  - Rerun preflight, risk/reward, market-regime/context, sizing, exposure, and
    stale-price checks.
  - Preserve lineage from original signal/watch through final outcome.
  - _Requirements: 4.1-4.6_

- [ ] 6. Implement terminal watch outcomes and dashboard language
  - Add or normalize outcomes for `Waiting for Entry`, `Triggered`,
    `Missed Move`, `Invalidated`, `Expired`, `Gate Rejected`, and
    `Execution Failed`.
  - Ensure never-triggered watches do not create incomplete-geometry shadow
    outcome rows.
  - _Requirements: 5.1-5.6_

- [ ] 7. Preserve immediate breakout behavior
  - Verify valid active breakout/breakdown continuation candidates still reach
    immediate PM offering.
  - Ensure setup-watch feature flags do not suppress immediate candidates.
  - _Requirements: 6.1-6.4_

- [ ] 8. Add regression tests for the META missed move
  - Build the 2026-09-21 META support-bounce fixture.
  - Assert no immediate PM candidate is created.
  - Assert terminal missed-move/no-offer telemetry, not execution failure.
  - Assert PM compute/latency is not identified as the root cause.
  - _Requirements: 7.1-7.5_

- [ ] 9. Add rollout flag and observe-mode metrics
  - Add `WAIT_FOR_ENTRY_CANDIDATE_ROUTING=disabled|observe|enabled`.
  - In observe mode, log would-route decisions without changing execution.
  - Add metrics for immediate candidates, watch proposals, armed watches,
    triggers, missed moves, invalidations, stale-entry failures, and PM watch
    decisions.
  - _Requirements: 1.7, 2.2, 5.5_

- [ ] 10. Checkpoint
  - Run focused tests for candidate builder, setup watches, pending orders,
    PM prompt contract, shadow outcomes, and the META regression.
  - Run the smallest practical integration smoke test against the local
    paper-trader dashboard/API before enabling.
