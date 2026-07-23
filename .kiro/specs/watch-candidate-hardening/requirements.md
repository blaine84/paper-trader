# Requirements Document

## Introduction

Hardening patch for the watch candidate lifecycle in the market-state-trigger-contract feature. Addresses four logic gaps discovered during review that must be resolved before enabling `MARKET_STATE_MODE=enforcing` in production: promoted-watch re-query duplication, promoted-watch eligibility bypass, missing structural invalidation checks, and same-cycle instant-promotion without observation.

All changes are feature-flagged behind `MARKET_STATE_MODE != "disabled"` and follow existing fail-open (observability) / fail-closed (safety) conventions.

## Non-Goals

- **Short-side market-state parity** (trigger model, VWAP distance) is a follow-up design and not addressed by this hardening patch.
- **Same-cycle activation fast-path** (evaluating newly-created watches in the same pass they are created) is explicitly deferred to a separate feature. In this patch, newly created watches are only evaluated starting the next cycle. The evaluation-order change (evaluate existing watches before creating new ones) means same-cycle promotion is now an edge case (limited to watches whose `source_cycle_id` equals the current `cycle_id`, governed by `WATCH_SAME_CYCLE_PROMOTION_POLICY`) rather than the norm.

## Glossary

- **Watch_Candidate_Manager**: The module (`utils/watch_candidates.py`) responsible for creating, evaluating, promoting, and expiring watch candidates.
- **Candidate_Builder**: The module (`utils/candidate_builder.py`) that integrates watch candidate promotions into the PM candidate pipeline.
- **Promoted_Watch**: A watch candidate whose activation threshold has been crossed and whose state has transitioned from `active` to `promoted`.
- **Registered_Watch**: A terminal state indicating a promoted watch has been successfully consumed (converted into a PM candidate) and must never be re-queried.
- **Structural_Invalidation**: The invalidation of a watch candidate due to degradation of market structure (lifecycle state regression or key-level drift) rather than price threshold breach.
- **Same_Cycle_Promotion**: The scenario where a watch candidate is created and promoted within the same evaluation cycle, meaning the watch never actually observed a subsequent market state.
- **Eligibility_Checks**: The set of pre-promotion filters (held_symbols exclusion, min_signal_strength threshold) that normal PM candidates pass through.
- **WATCHABLE_LIFECYCLE_STATES**: The frozenset of setup lifecycle states eligible for watch-candidate creation (`compression_watch`, `breakout_watch`, `breakout_confirmed_wait_retest`, `pullback_watch`, `pullback_validating`, `activation_pending`).
- **CAS_Transition**: Compare-And-Swap state transition pattern where UPDATE succeeds only if current state matches expected state.
- **Cycle_ID**: A unique identifier for each evaluation cycle, used to scope queries and prevent cross-cycle contamination.
- **Evaluation_Order**: The mandated sequence within a cycle: expire stale promoted watches → evaluate existing active watches → consume promoted watches → create new watches. New watches are NOT evaluated in the same pass in which they are created; they are first evaluated in the next cycle.

## Requirements

### Requirement 1: Promoted-Watch Lifecycle Scoping

**User Story:** As the system operator, I want promoted watch candidates scoped to their originating cycle and transitioned to a terminal state after consumption, so that stale promoted watches are never re-queried in subsequent cycles.

#### Acceptance Criteria

1. WHEN `get_promotable_candidates()` queries promoted watch candidates, THE Watch_Candidate_Manager SHALL filter results by the cycle_id in which the watch was promoted (the current cycle_id passed to `evaluate_active_watch_candidates()`), in addition to `state='promoted'` and `profile_id`.
2. WHEN the Candidate_Builder completes PM candidate creation from a promoted watch without raising an exception, THE Candidate_Builder SHALL transition the promoted watch to `registered` state via CAS_Transition.
3. IF the CAS_Transition to `registered` fails (rowcount == 0), THEN THE Candidate_Builder SHALL log the failure at WARNING level and continue without blocking the PM candidate creation.
4. THE Watch_Candidate_Manager SHALL treat `registered` as a terminal state that is never returned by any active or promoted query.
5. WHEN `_transition_watch_state()` is called for a promoted-to-registered transition, THE Watch_Candidate_Manager SHALL accept `promoted` as a valid source state (in addition to `active`).
6. WHEN `evaluate_active_watch_candidates()` transitions a watch to `promoted` state, THE Watch_Candidate_Manager SHALL record the current cycle_id on the watch candidate record so that `get_promotable_candidates()` can filter by promotion cycle.
7. WHEN the Candidate_Builder promotes a watch to a PM candidate, THE Candidate_Builder SHALL verify that no PM candidate already exists for the combination of `source_signal_id = watch_id`, `profile_id`, and `cycle_id` before registering. IF such a PM candidate already exists, THEN THE Candidate_Builder SHALL skip registration, transition the watch to `registered` state, and log at DEBUG level (idempotent promotion consumption).
8. WHILE `MARKET_STATE_MODE` is `"enforcing"`, IF `cycle_id` is `None` when `get_promotable_candidates()` is called, THEN THE Watch_Candidate_Manager SHALL return an empty list and log a warning indicating that cycle_id is required in enforcing mode.
9. WHILE `MARKET_STATE_MODE` is NOT `"enforcing"`, IF `cycle_id` is `None` when `get_promotable_candidates()` is called, THEN THE Watch_Candidate_Manager SHALL fall back to returning all promoted candidates for the profile (backward-compatible behavior for tests and observe mode).
10. THE orchestration layer SHALL enforce the following evaluation order within a cycle: (a) expire stale promoted watches, (b) evaluate existing active watches, (c) consume promoted watches via the Candidate_Builder, (d) create new watch candidates. New watches SHALL NOT be evaluated in the same pass in which they are created; they are first evaluated in the next cycle.
11. WHEN a cycle begins, before evaluating any active watch, THE Watch_Candidate_Manager SHALL expire every `promoted` watch whose `promoted_cycle_id` does not equal the current cycle_id, transitioning it to `expired` with outcome_json containing `"terminal_reason": "promotion_expired_stale_cycle"`. This is a decisive expire (not a re-consume), producing a clean audit trail and preventing cross-cycle promotion of a watch that crashed before consumption.
12. WHEN transitioning a watch to `promoted` state, THE Watch_Candidate_Manager SHALL persist `promoted_cycle_id` atomically with the state change (via an optional parameter to `_transition_watch_state()` or a dedicated helper), so that a promoted row is never left with a NULL or stale `promoted_cycle_id` relative to the state transition.
13. WHEN `expire_session_watch_candidates()` runs its TTL sweep, THE Watch_Candidate_Manager SHALL expire both stale `active` rows AND stale `promoted` rows whose `expires_at` is earlier than the current time, so that terminal-state guarantees do not rely solely on the happy-path consumption of promoted watches.

### Requirement 2: Promoted-Watch Eligibility Enforcement

**User Story:** As the system operator, I want promoted watch candidates to pass the same eligibility checks as normal PM candidates, so that held-symbol duplication and weak-signal promotions are blocked.

#### Acceptance Criteria

1. WHEN the Candidate_Builder processes a promotable watch candidate, THE Candidate_Builder SHALL check the symbol against the current held_symbols set (symbols with open positions for the profile) before creating a PM candidate.
2. WHEN the Candidate_Builder processes a promotable watch candidate, THE Candidate_Builder SHALL verify that the current cycle signal's strength meets or exceeds the profile min_signal_strength threshold using the same STRENGTH_ORDER comparison as normal candidate filtering.
3. WHEN the Candidate_Builder applies eligibility checks to a promotable watch candidate, THE Candidate_Builder SHALL evaluate the held_symbols check before the min_signal_strength check and short-circuit on first failure (recording only the first blocking reason).
4. IF the held_symbols check fails for a promoted watch, THEN THE Candidate_Builder SHALL transition the watch to `expired` state with outcome_json containing `"terminal_reason": "promotion_blocked_held_symbol"`.
5. IF the min_signal_strength check fails for a promoted watch, THEN THE Candidate_Builder SHALL transition the watch to `expired` state with outcome_json containing `"terminal_reason": "promotion_blocked_weak_signal"`.
6. WHEN an eligibility check blocks a promotion, THE Candidate_Builder SHALL log the blocking reason at INFO level and skip PM candidate creation for that watch.
7. IF the state transition to `expired` fails for a blocked promotion (CAS rowcount == 0), THEN THE Candidate_Builder SHALL log the failure at WARNING level and continue without blocking processing of remaining promotable candidates.
8. IF the geometry scaffold fails or raises an exception during promotion processing, THEN THE Candidate_Builder SHALL transition the watch to `expired` state with outcome_json containing `"terminal_reason": "promotion_blocked_geometry_failed"`.
9. IF the geometry scaffold returns zero candidates (empty result), THEN THE Candidate_Builder SHALL transition the watch to `expired` state with outcome_json containing `"terminal_reason": "promotion_blocked_no_geometry_candidates"`.
10. IF the CandidateRegistry raises an error during PM candidate registration for a promoted watch, THEN THE Candidate_Builder SHALL transition the watch to `expired` state with outcome_json containing `"terminal_reason": "promotion_blocked_registry_error"` and log the error at WARNING level.

### Requirement 3: Structural Invalidation of Active Watches

**User Story:** As the system operator, I want active watch candidates invalidated when their underlying market structure degrades, so that watches do not promote on stale structural context.

#### Acceptance Criteria

1. WHEN `evaluate_active_watch_candidates()` evaluates an active watch, THE Watch_Candidate_Manager SHALL check the current signal's `setup_lifecycle_state` for the watched symbol.
2. IF the current `setup_lifecycle_state` is not in WATCHABLE_LIFECYCLE_STATES, THEN THE Watch_Candidate_Manager SHALL invalidate the watch with outcome_json containing `"terminal_reason": "structural_degradation"`.
3. WHEN `evaluate_active_watch_candidates()` evaluates an active watch, THE Watch_Candidate_Manager SHALL compare the current key levels (support and resistance) against the stored key levels from watch creation by computing drift as `abs(current_value - stored_value) / stored_value * 100` for each level that has a non-null numeric value greater than zero in both the stored snapshot and the current signal.
4. IF any key level (support or resistance) has drifted more than `WATCH_KEY_LEVEL_DRIFT_PCT` percent from the stored value, THEN THE Watch_Candidate_Manager SHALL invalidate the watch with outcome_json containing `"terminal_reason": "key_level_drift"`.
5. THE Watch_Candidate_Manager SHALL evaluate an active watch in the following order: (a) structural invalidation checks, (b) price-threshold invalidation, (c) price-threshold activation detection, (d) same-cycle promotion policy, (e) the actual promote/block transition. Structural invalidation SHALL take priority over all price-threshold evaluation, and the same-cycle promotion policy SHALL only be applied after price activation has been detected but before the watch is transitioned to `promoted`.
6. IF the current signal is missing, contains no lifecycle data for the watched symbol, or contains no non-null key levels for comparison, THEN THE Watch_Candidate_Manager SHALL skip structural checks and proceed to price-threshold evaluation (fail-open).
7. THE Watch_Candidate_Manager SHALL expose the key-level drift percentage threshold as a configurable constant `WATCH_KEY_LEVEL_DRIFT_PCT` in `gate_config.py` with a default value of `2.0` (percent).
8. WHEN computing key-level drift, THE Watch_Candidate_Manager SHALL skip any stored or current level value that is non-numeric or less than or equal to zero, and SHALL log the skipped level at DEBUG level with the reason (non-numeric or non-positive value).
9. WHEN an active watch has no current signal data for its symbol for repeated evaluation cycles, THE Watch_Candidate_Manager SHALL rely on TTL expiration as the backstop cleanup mechanism. TTL expiration is the defined behavior for symbols with no current signal data.

### Requirement 4: Same-Cycle Promotion Policy

**User Story:** As the system operator, I want to control whether watch candidates can promote in the same cycle they are created, so that watches actually observe market state before triggering promotion.

#### Acceptance Criteria

1. THE Watch_Candidate_Manager SHALL read a `WATCH_SAME_CYCLE_PROMOTION_POLICY` configuration from `gate_config.py` with allowed values `"never"`, `"activation_pending_only"`, `"always"`.
2. THE Watch_Candidate_Manager SHALL default `WATCH_SAME_CYCLE_PROMOTION_POLICY` to `"activation_pending_only"` when the environment variable is not set or contains an unrecognized value.
3. WHEN `evaluate_active_watch_candidates()` identifies a watch as promotion-eligible, THE Watch_Candidate_Manager SHALL check whether `source_cycle_id` matches the current cycle_id.
4. WHILE `WATCH_SAME_CYCLE_PROMOTION_POLICY` is `"never"`, THE Watch_Candidate_Manager SHALL skip promotion for any watch where `source_cycle_id` equals the current cycle_id regardless of lifecycle state.
5. WHILE `WATCH_SAME_CYCLE_PROMOTION_POLICY` is `"activation_pending_only"`, IF the current signal for the watched symbol contains a `setup_lifecycle_state` of `"activation_pending"`, THEN THE Watch_Candidate_Manager SHALL allow same-cycle promotion; otherwise THE Watch_Candidate_Manager SHALL skip promotion for that watch.
6. WHILE `WATCH_SAME_CYCLE_PROMOTION_POLICY` is `"always"`, THE Watch_Candidate_Manager SHALL allow same-cycle promotion without restriction.
7. WHEN same-cycle promotion is blocked by policy, THE Watch_Candidate_Manager SHALL leave the watch in `active` state (not expire it) so it can be evaluated in the next cycle.
8. WHEN same-cycle promotion is blocked by policy, THE Watch_Candidate_Manager SHALL log the block at DEBUG level with the policy name and watch_id.
9. WHEN `evaluate_active_watch_candidates()` is called, THE Watch_Candidate_Manager SHALL accept a `cycle_id` parameter with a default value of `None` to maintain backward compatibility with existing callers.
10. WHILE `MARKET_STATE_MODE` is `"enforcing"`, IF `cycle_id` is `None` when `evaluate_active_watch_candidates()` is called, THEN THE Watch_Candidate_Manager SHALL still perform invalidation checks (both structural and price-threshold) but SHALL NOT transition any watch to `promoted` (promotion is blocked entirely), and SHALL log a warning indicating that cycle_id is required for promotion in enforcing mode. This prevents leaving a watch in `promoted` state with a NULL `promoted_cycle_id`, which `get_promotable_candidates()` could never consume.
11. WHILE `MARKET_STATE_MODE` is NOT `"enforcing"`, IF `cycle_id` is `None` when `evaluate_active_watch_candidates()` is called, THEN THE Watch_Candidate_Manager SHALL skip same-cycle policy evaluation entirely and allow promotion as normal (backward-compatible behavior for tests and observe mode).
12. IF the current signal is missing or contains no `setup_lifecycle_state` for the watched symbol while `WATCH_SAME_CYCLE_PROMOTION_POLICY` is `"activation_pending_only"`, THEN THE Watch_Candidate_Manager SHALL treat the watch as not meeting the `"activation_pending"` condition and skip promotion for that cycle.
13. THE Watch_Candidate_Manager SHALL evaluate the same-cycle promotion policy AFTER structural invalidation checks (Requirement 3) AND AFTER price-threshold activation has been detected, but BEFORE the watch is actually transitioned to `promoted`. The same-cycle policy cannot be applied before the activation threshold is known to be crossed, so it governs only whether a watch that has already met its activation threshold is allowed to promote in the same cycle it was created.

### Requirement 5: Feature Flag Guarding and Backward Compatibility

**User Story:** As the system operator, I want all hardening patches guarded by the existing `MARKET_STATE_MODE` feature flag, so that production behavior is unchanged when the flag is set to `"disabled"`.

#### Acceptance Criteria

1. WHILE `MARKET_STATE_MODE` is `"disabled"`, THE Watch_Candidate_Manager SHALL not execute any of the hardening logic (structural invalidation, same-cycle policy, eligibility enforcement) and SHALL return early from hardening code paths without modifying watch state.
2. WHILE `MARKET_STATE_MODE` is `"observe"`, THE Watch_Candidate_Manager SHALL execute hardening logic (structural invalidation, same-cycle policy) — logging outcomes and transitioning state — but SHALL NOT execute promotion to PM candidates (promotions are expired with shadow outcome as per the existing observe-mode behavior).
3. THE Candidate_Builder SHALL maintain backward-compatible function signatures by adding new parameters with default values only.
4. IF any hardening check (structural invalidation or same-cycle policy) raises an unexpected exception during evaluation, THEN THE Watch_Candidate_Manager SHALL log the exception at WARNING level, leave the affected watch in its current state unchanged, and continue evaluation of remaining watches (fail-open).
5. IF an eligibility check raises an unexpected exception during promotion, THEN THE Candidate_Builder SHALL block the promotion (fail-closed for safety) and transition the watch to `expired` with outcome_json containing `"terminal_reason": "promotion_blocked_eligibility_error"`.
6. THE Watch_Candidate_Manager SHALL not cause any previously passing tests to fail when all new configuration defaults are applied (i.e., `MARKET_STATE_MODE` defaults to `"disabled"`).
