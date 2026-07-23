# Implementation Plan: Watch Candidate Hardening

## Overview

Hardening patch for the watch candidate lifecycle addressing four logic gaps: promoted-watch re-query duplication, promoted-watch eligibility bypass, missing structural invalidation, and same-cycle instant-promotion. All changes are guarded by `MARKET_STATE_MODE` and follow fail-open/fail-closed conventions.

Key structural change: the candidate_builder evaluation order is restructured into a mandated 4-step sequence — expire stale promoted watches (Step 0) → evaluate existing active watches → consume promoted watches → create new watches (never evaluated same pass). There is no same-cycle fast-path; newly created watches are first evaluated in the next cycle.

## Tasks

- [x] 1. Add configuration constants and schema migration
  - [x] 1.1 Add `WATCH_KEY_LEVEL_DRIFT_PCT` and `WATCH_SAME_CYCLE_PROMOTION_POLICY` constants to `utils/gate_config.py`
    - Add `WATCH_KEY_LEVEL_DRIFT_PCT` with default 2.0, read from env var
    - Add `WATCH_SAME_CYCLE_PROMOTION_POLICY` with default "activation_pending_only", validate against allowed values ("never", "activation_pending_only", "always"), fallback to default on unrecognized
    - _Requirements: 3.7, 4.1, 4.2_

  - [x] 1.2 Add `promoted_cycle_id` column and partial index to schema migration in `orchestrator.py`
    - `ALTER TABLE watch_candidates ADD COLUMN promoted_cycle_id TEXT`
    - `CREATE INDEX IF NOT EXISTS idx_watch_candidates_promoted_cycle ON watch_candidates (profile_id, promoted_cycle_id) WHERE state = 'promoted'`
    - Use non-destructive `IF NOT EXISTS` / `ADD COLUMN` pattern (fail-open on pre-existing column)
    - _Requirements: 1.6_

- [x] 2. Implement watch_candidates.py hardening (structural invalidation, same-cycle policy, cycle scoping, stale cleanup, TTL sweep)
  - [x] 2.1 Modify `_transition_watch_state()` to accept `expected_state` and `promoted_cycle_id` parameters
    - Add keyword argument `expected_state: str = "active"` for backward compatibility
    - Change CAS WHERE clause from hardcoded `state = 'active'` to `state = :expected_state`
    - Add keyword argument `promoted_cycle_id: str | None = None`
    - Write `promoted_cycle_id` atomically in the same UPDATE via `COALESCE(:promoted_cycle_id, promoted_cycle_id)` so non-promotion transitions never clear an existing value and a promoted row is never observable with a NULL/stale `promoted_cycle_id`
    - _Requirements: 1.5, 1.12_

  - [x] 2.2 Implement `_check_structural_invalidation()` function in `utils/watch_candidates.py`
    - Accept `watch: WatchCandidate` and `current_signal: dict`
    - Check 1: current `setup_lifecycle_state` not in `WATCHABLE_LIFECYCLE_STATES` → return `"structural_degradation"`
    - Check 2: compute key-level drift for support/resistance — `abs(current - stored) / stored * 100`
    - Skip levels where stored or current value is non-numeric or <= 0, log skipped at DEBUG with reason
    - If any level drift > `WATCH_KEY_LEVEL_DRIFT_PCT` → return `"key_level_drift"`
    - Fail-open: return None if signal missing lifecycle data or no comparable key levels
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8_

  - [x] 2.3 Implement `_check_same_cycle_policy()` function in `utils/watch_candidates.py`
    - Accept `watch: WatchCandidate`, `current_signal: dict`, `cycle_id: str | None`
    - Return True (blocked) / False (allowed)
    - If `cycle_id is None` → return False (skip policy, backward compat)
    - If `source_cycle_id != cycle_id` → return False (not same-cycle)
    - Read `WATCH_SAME_CYCLE_PROMOTION_POLICY` from gate_config
    - "never" → return True (block); "always" → return False (allow)
    - "activation_pending_only" → allow only if current signal's `setup_lifecycle_state == "activation_pending"`, else block
    - If signal missing lifecycle data under "activation_pending_only" → block
    - Log blocked promotions at DEBUG with policy name and watch_id
    - _Requirements: 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.12, 4.13_

  - [x] 2.4 Modify `evaluate_active_watch_candidates()` for corrected per-watch order, cycle_id, and enforcing-mode promotion blocking
    - Add `cycle_id: str | None = None` keyword parameter (backward-compatible)
    - Corrected per-watch evaluation order: structural invalidation → price-threshold invalidation → price-threshold activation detection → same-cycle policy → promote/block (same-cycle policy now runs AFTER activation detection, gating only the final `active → promoted` transition — not before price-threshold evaluation)
    - On structural invalidation: transition to `invalidated` with appropriate terminal_reason (before any price logic runs)
    - On same-cycle blocked: leave in `active` state, increment `still_active`
    - Enforcing mode + `cycle_id is None`: still run all invalidation checks (structural + price-threshold) but SHALL NOT transition any watch to `promoted` (promotion blocked entirely); log WARNING that cycle_id is required for promotion in enforcing mode
    - Non-enforcing mode + `cycle_id is None`: skip same-cycle policy entirely and allow promotion as normal (backward-compatible)
    - On promotion: transition `active → promoted` recording `promoted_cycle_id = cycle_id` via the `promoted_cycle_id` parameter of `_transition_watch_state()` (atomic write)
    - Wrap structural/same-cycle checks in try/except: fail-open (log WARNING, continue)
    - Fetch `source_cycle_id` and `key_levels_json` in the SELECT query
    - _Requirements: 1.6, 1.12, 3.1, 3.2, 3.3, 3.4, 3.5, 4.3, 4.9, 4.10, 4.11, 4.13, 5.1, 5.4_

  - [x] 2.5 Modify `get_promotable_candidates()` to accept and filter by `cycle_id`
    - Add `cycle_id: str | None = None` keyword parameter
    - Enforcing mode + cycle_id=None: return empty list + log WARNING
    - Non-enforcing mode + cycle_id=None: fall back to existing behavior (all promoted for profile)
    - When cycle_id provided: add `AND promoted_cycle_id = :cycle_id` to WHERE clause
    - _Requirements: 1.1, 1.8, 1.9_

  - [x] 2.6 Implement `expire_stale_promoted_watches()` function in `utils/watch_candidates.py`
    - Accept `engine`, `profile_id: str`, `cycle_id: str`
    - Query promoted rows where `promoted_cycle_id IS NULL OR promoted_cycle_id != :cycle_id` for the profile
    - For each matched row, decisively expire via `_transition_watch_state(..., expected_state="promoted")` with outcome_json `"terminal_reason": "promotion_expired_stale_cycle"` (decisive expire, NOT a re-consume)
    - Per-row CAS failure (already consumed concurrently): log WARNING and skip harmlessly
    - Fail-open: catch OperationalError on missing `promoted_cycle_id` column (pre-migration), log WARNING, return 0
    - Return count of expired stale promoted rows
    - _Requirements: 1.11_

  - [x] 2.7 Modify `expire_session_watch_candidates()` TTL sweep to include stale promoted rows
    - Add `profile_id: str` parameter consistent with cycle-scoped usage
    - Widen the sweep predicate from `state = 'active'` to `state IN ('active', 'promoted')` for rows whose `expires_at < now`
    - Transition matched rows to `expired` with `"terminal_reason": "ttl_expired"` (preserve existing outcome_json via COALESCE)
    - Guarantees terminal state for a promoted watch that was never consumed (crash between promotion and the promotion loop)
    - _Requirements: 1.13_

  - [x] 2.8 Write unit tests for `_check_structural_invalidation()` in `tests/test_watch_candidates.py`
    - Test lifecycle state regression (not in WATCHABLE_LIFECYCLE_STATES)
    - Test key-level drift above threshold; test drift within threshold (no invalidation)
    - Test non-numeric / <= 0 stored levels are skipped
    - Test missing signal data returns None (fail-open)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.8_

  - [x] 2.9 Write unit tests for `_check_same_cycle_policy()` in `tests/test_watch_candidates.py`
    - Test "never" blocks all same-cycle; "always" allows all same-cycle
    - Test "activation_pending_only" with matching and non-matching lifecycle
    - Test cycle_id=None skips policy; different source_cycle_id always allowed
    - Test missing signal lifecycle data under "activation_pending_only" blocks
    - _Requirements: 4.4, 4.5, 4.6, 4.7, 4.12_

  - [x] 2.10 Write unit tests for stale cleanup, TTL sweep, atomic transition, and enforcing cycle_id=None in `tests/test_watch_candidates.py`
    - `expire_stale_promoted_watches()`: promoted row with `promoted_cycle_id != cycle_id` (and NULL) expired with `promotion_expired_stale_cycle`; column-missing returns 0 (fail-open)
    - `expire_session_watch_candidates()`: expires both stale `active` and stale `promoted` rows past `expires_at`
    - `_transition_watch_state()`: writes `promoted_cycle_id` atomically on active → promoted; COALESCE preserves existing value on non-promotion transitions
    - `evaluate_active_watch_candidates()` enforcing + cycle_id=None: runs invalidation but transitions nothing to `promoted` (WARNING logged)
    - _Requirements: 1.11, 1.12, 1.13, 4.10_

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Restructure candidate_builder evaluation order and implement promotion helper
  - [x] 4.1 Restructure the watch candidate management block in `utils/candidate_builder.py` to the mandated 4-step order
    - Step 0: call `expire_stale_promoted_watches(engine=db, profile_id=profile_id, cycle_id=cycle_id)` FIRST (only when `cycle_id is not None`) to decisively retire promoted rows from crashed prior cycles
    - Step 1: `evaluate_active_watch_candidates()` passing `cycle_id=cycle_id`
    - Step 2 (enforcing only): `get_promotable_candidates()` passing `cycle_id=cycle_id` + promotion loop delegating to `_process_promoted_watch()`
    - Step 3: `evaluate_and_create_watch_candidates()` LAST — new watches are NOT evaluated this pass (no same-cycle fast-path); first evaluated next cycle
    - Import `expire_stale_promoted_watches` and `_transition_watch_state` from watch_candidates
    - Wrap the block in try/except (fail-open, log WARNING)
    - _Requirements: 1.10, 1.11_

  - [x] 4.2 Implement `_process_promoted_watch()` helper function in `utils/candidate_builder.py`
    - Accept: engine, promo dict, registry, held_symbols, min_signal_strength, profile_id, cycle_id, cycle_expires_at
    - Eligibility check 1 (short-circuit): symbol in held_symbols → expire with `promotion_blocked_held_symbol`
    - Eligibility check 2: signal strength below threshold (STRENGTH_ORDER) → expire with `promotion_blocked_weak_signal`
    - Exception during eligibility → expire with `promotion_blocked_eligibility_error` (fail-closed)
    - Idempotent dedup: query pm_candidates for existing (source_signal_id=watch_id, profile_id, cycle_id) → if exists, skip register, transition to registered, log DEBUG
    - Geometry scaffold: failure/exception → expire with `promotion_blocked_geometry_failed`
    - Geometry scaffold: 0 candidates → expire with `promotion_blocked_no_geometry_candidates`
    - Registry.register() error → expire with `promotion_blocked_registry_error`, log WARNING
    - Success path: register PM candidate, transition watch promoted → registered via `_transition_watch_state(expected_state="promoted")`
    - CAS promoted→registered failure: log WARNING, continue (fail-open)
    - All transitions to expired use `_transition_watch_state(expected_state="promoted")`
    - Log eligibility blocks at INFO level
    - _Requirements: 1.2, 1.3, 1.7, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 5.5_

  - [x] 4.3 Write integration tests for 4-step order, stale-cycle recovery, and idempotent promotion in `tests/test_market_state_integration.py`
    - Verify candidate_builder executes Step 0 → 1 → 2 → 3 (expire stale promoted → evaluate → consume → create) in correct order (mock-based ordering verification)
    - Stale-cycle crash recovery: promote a watch in cycle A, skip consumption, start cycle B → watch expired with `promotion_expired_stale_cycle`, never enters cycle B's candidate set
    - Verify idempotent promotion: pre-insert PM candidate, run promotion loop → no duplicate, watch → registered
    - Verify geometry failure produces `promotion_blocked_geometry_failed`; registry error produces `promotion_blocked_registry_error`
    - Verify eligibility short-circuit ordering (held_symbols checked before strength)
    - _Requirements: 1.2, 1.7, 1.10, 1.11, 2.3, 2.8, 2.9, 2.10_

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Property-based tests for correctness properties (in `tests/test_prop_watch_hardening.py`)
  - [x] 6.1 Write property test for promoted-cycle scoping
    - **Property 1: Promoted-cycle scoping excludes cross-cycle watches**
    - **Validates: Requirements 1.1, 1.6**

  - [x] 6.2 Write property test for registered state query-invisibility
    - **Property 2: Registered state is query-invisible**
    - **Validates: Requirements 1.4**

  - [x] 6.3 Write property test for successful promotion terminal state
    - **Property 3: Successful promotion produces registered terminal state**
    - **Validates: Requirements 1.2**

  - [x] 6.4 Write property test for idempotent promotion consumption
    - **Property 4: Idempotent promotion consumption**
    - **Validates: Requirements 1.7**

  - [x] 6.5 Write property test for enforcing mode cycle_id requirement
    - **Property 5: Enforcing mode requires cycle_id for get_promotable_candidates**
    - **Validates: Requirements 1.8**

  - [x] 6.6 Write property test for held-symbol exclusion with short-circuit
    - **Property 6: Held-symbol exclusion blocks promotion with short-circuit**
    - **Validates: Requirements 2.1, 2.3, 2.4**

  - [x] 6.7 Write property test for signal strength threshold blocking
    - **Property 7: Signal strength threshold blocks weak promotions**
    - **Validates: Requirements 2.2, 2.5**

  - [x] 6.8 Write property test for promotion loop terminal failures
    - **Property 8: Promotion loop terminal failures produce correct terminal reasons**
    - **Validates: Requirements 2.8, 2.9, 2.10**

  - [x] 6.9 Write property test for structural degradation invalidation
    - **Property 9: Structural degradation invalidates active watches**
    - **Validates: Requirements 3.1, 3.2, 3.5**

  - [x] 6.10 Write property test for key-level drift invalidation
    - **Property 10: Key-level drift invalidates active watches**
    - **Validates: Requirements 3.3, 3.4**

  - [x] 6.11 Write property test for drift computation skipping non-numeric levels
    - **Property 11: Drift computation skips non-numeric and non-positive levels**
    - **Validates: Requirements 3.8**

  - [x] 6.12 Write property test for same-cycle "never" policy
    - **Property 12: Same-cycle "never" policy blocks all same-cycle promotions**
    - **Validates: Requirements 4.4, 4.7**

  - [x] 6.13 Write property test for same-cycle "activation_pending_only" policy
    - **Property 13: Same-cycle "activation_pending_only" policy is lifecycle-gated**
    - **Validates: Requirements 4.5, 4.7, 4.12**

  - [x] 6.14 Write property test for evaluation order (structural precedes price, same-cycle gates only after activation)
    - **Property 14: Evaluation order — structural precedes price, same-cycle policy gates only after activation**
    - **Validates: Requirements 3.5, 4.13**

  - [x] 6.15 Write property test for disabled mode no-op
    - **Property 15: Disabled mode executes no hardening logic**
    - **Validates: Requirements 5.1**

  - [x] 6.16 Write property test for hardening exceptions fail-open
    - **Property 16: Hardening exceptions are fail-open (evaluation)**
    - **Validates: Requirements 5.4**

  - [x] 6.17 Write property test for eligibility exceptions fail-closed
    - **Property 17: Eligibility exceptions are fail-closed (promotion)**
    - **Validates: Requirements 5.5**

  - [x] 6.18 Write property test for stale promoted expiration at cycle start
    - **Property 18: Stale promoted watches are expired at cycle start**
    - **Validates: Requirements 1.11**

  - [x] 6.19 Write property test for enforcing mode + cycle_id=None blocking all promotion
    - **Property 19: Enforcing mode with cycle_id=None blocks all promotion**
    - **Validates: Requirements 4.10**

  - [x] 6.20 Write property test for TTL sweep expiring stale promoted rows
    - **Property 20: TTL sweep expires stale promoted rows**
    - **Validates: Requirements 1.13**

- [x] 7. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from design §Correctness Properties (20 properties total)
- The candidate_builder now uses a mandated 4-step order with Step 0 stale cleanup (task 4.1) — there is no same-cycle fast-path
- Per-watch evaluation order is corrected (task 2.4): same-cycle policy runs AFTER activation detection, gating only the final promote transition
- `expire_stale_promoted_watches()` (task 2.6) and the widened TTL sweep (task 2.7) together guarantee promoted watches always reach a terminal state
- `_transition_watch_state()` writes `promoted_cycle_id` atomically via COALESCE (task 2.1)
- `WATCH_KEY_LEVEL_DRIFT_PCT` default is 2.0 (not 5.0) per design §10
- All new function parameters have None/default values for backward compatibility
- Property tests go in `tests/test_prop_watch_hardening.py`, unit tests in `tests/test_watch_candidates.py`, integration tests in `tests/test_market_state_integration.py`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.6", "2.7"] },
    { "id": 3, "tasks": ["2.4", "2.5"] },
    { "id": 4, "tasks": ["2.8", "2.9", "2.10"] },
    { "id": 5, "tasks": ["4.1", "4.2"] },
    { "id": 6, "tasks": ["4.3"] },
    { "id": 7, "tasks": ["6.1", "6.2", "6.3", "6.4", "6.5", "6.6", "6.7", "6.8", "6.9", "6.10", "6.11", "6.12", "6.13", "6.14", "6.15", "6.16", "6.17", "6.18", "6.19", "6.20"] }
  ]
}
```
