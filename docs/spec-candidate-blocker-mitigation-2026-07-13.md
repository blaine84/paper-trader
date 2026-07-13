# Candidate Blocker Mitigation Spec

Date: 2026-07-13
Target implementer: Kiro or Cirrus
Related specs:
- `specs/pm-candidate-id-selection-contract-2026-06-05.md`
- `specs/pm-candidate-backed-entry-execution-2026-06-23.md`
- `docs/spec-swing-candidate-pipeline-2026-07-06.md`
- `docs/spec-swing-observability-normalization-2026-07-07.md`

## Problem

The paper trader can now build executable PM candidates, including swing
candidates with deterministic entry, stop, target, and risk/reward geometry.
On 2026-07-13, the system built two XLE `sector_rotation_swing` candidates,
but both PM profiles rejected them with vague "missing trade details" rationales
even though the candidate registry held complete geometry.

Commit `c44b9e1` fixed the immediate prompt/audit handoff by repeating full
candidate specs in the PM prompt and enriching the `offered` candidate event.
The next expected blockers are no longer missing geometry, but ambiguous PM
rejections, accepted candidates failing downstream gates, accepted candidates
dying in execution, weak post-entry lifecycle checks, and incomplete daily
traceability.

The goal of this spec is to make each candidate loss point explicit and
machine-readable without forcing unsafe trades.

## Proposed Fix

Add a candidate blocker mitigation layer around candidate-ID PM selection:

1. Require PM rejections to use bounded reason codes.
2. Surface deterministic candidate preflight status before the PM call.
3. Avoid offering candidates that are known to be non-executable before PM
   judgment.
4. Record preflight, PM decision, gate/sizing, execution, and post-entry
   lifecycle events with consistent candidate lineage.
5. Provide daily integrity output that explains where candidates were lost.

This should preserve PM discretion while replacing vague prose with structured
diagnostics.

## Goals

1. Keep deterministic candidate geometry as the only executable source of
   entry, stop, target, direction, and symbol.
2. Keep PM discretion over whether an otherwise valid candidate is worth taking.
3. Make every candidate outcome classifiable as:
   - not offered because deterministic preflight failed,
   - offered and PM rejected,
   - PM accepted but sizing/gate rejected,
   - PM accepted but execution failed,
   - executed and lifecycle armed,
   - executed but lifecycle incomplete.
4. Make daily CEO/review output able to identify the dominant blocker without
   scraping free-form PM prose.

## Non-Goals

1. Do not relax risk gates just to create trades.
2. Do not force PM acceptance of candidates that fail profile or market-quality
   judgment.
3. Do not allow the PM to provide executable prices, quantities, or alternate
   symbols.
4. Do not replace the existing candidate registry, candidate pipeline, shadow
   ledger, or provenance framework.
5. Do not create live orders from preflight-only or diagnostic-only candidates.

## Requirements

### Requirement 1 - Bounded PM Rejection Reasons

Every PM candidate rejection in candidate-ID mode must include a bounded
`rejection_reason_code`.

Allowed initial codes:

- `low_confidence`
- `mixed_timeframes`
- `late_day`
- `hostile_breadth`
- `thin_volume`
- `risk_off_conflict`
- `profile_rule_failed`
- `exposure_conflict`
- `liquidity_or_spread`
- `event_risk`
- `other`

Acceptance criteria:

1. The PM prompt must instruct the model to provide one reason code per rejected
   candidate.
2. The structured PM response schema must accept and validate the reason code.
3. Unknown reason codes must be normalized to `other` and logged as contract
   violations.
4. PM rejection events must persist both `rationale` and
   `rejection_reason_code`.
5. Existing candidate-ID decisions without a reason code must remain parseable
   during rollout, but must emit an observability warning.

### Requirement 2 - Deterministic Candidate Preflight

Before a candidate is offered to the PM, trusted code must compute a deterministic
preflight summary.

The preflight summary must include:

- `has_entry_stop_target`
- `min_risk_reward_met`
- `direction_valid`
- `profile_allowed`
- `candidate_not_expired`
- `cash_available`
- `sizing_possible`
- `max_positions_available`
- `same_symbol_allowed`
- `blocking_reason_codes`

Acceptance criteria:

1. Candidates with missing geometry, invalid direction, expired state, or
   profile-disallowed setup must not be offered to the PM.
2. Candidates that fail deterministic preflight must write a structured
   candidate event with `event_type='preflight_failed'`.
3. Candidates that pass deterministic preflight must write
   `event_type='preflight_passed'` before the PM call.
4. The PM prompt must show the preflight summary for each offered candidate.
5. Preflight must not use LLM judgment.

### Requirement 3 - Do Not Offer Known Non-Executable Candidates

Candidates that deterministic code already knows cannot execute must be excluded
from the PM prompt.

Acceptance criteria:

1. A candidate that cannot be sized with current profile/account state must not
   be offered unless a feature flag enables diagnostic observe-only offering.
2. A candidate excluded by preflight must retain enough audit data to explain
   why it was excluded.
3. Excluded candidates must remain eligible for shadow analysis only if they
   have complete trusted geometry.
4. Exclusion must not remove analyst signal telemetry.

### Requirement 4 - PM Prompt Must Distinguish Valid Geometry From PM Judgment

The PM prompt must explicitly distinguish deterministic validity from PM
judgment.

Acceptance criteria:

1. For each offered candidate, the prompt must state that entry, stop, target,
   R:R, and candidate identity passed deterministic preflight.
2. The PM may reject for market quality, profile fit, timing, exposure, or risk,
   but must not reject because executable geometry is missing.
3. Swing candidates must include clear text explaining whether the candidate is
   intraday or swing and what holding horizon was assumed.
4. Rejections that claim missing geometry for a preflight-passed candidate must
   be classified as `contract_violation_missing_geometry_claim` or equivalent
   observability telemetry.

### Requirement 5 - Accepted Candidate Gate/Sizing Telemetry

When the PM accepts a candidate, every downstream gate and sizing step must
persist a structured result.

Acceptance criteria:

1. Accepted candidates must write a `pm_accept` event with candidate ID, profile,
   reason code if available, and risk multiplier.
2. Sizing must write a pass/fail event with quantity, dollar risk, risk percent,
   and reason codes for failure.
3. Pre-trade gates must write pass/fail events with gate names and reason codes.
4. If a candidate is accepted but not executed, the final candidate state must
   identify the first blocking stage.
5. CEO/daily review must be able to count accepted-but-blocked candidates by
   blocking stage.

### Requirement 6 - Execution Failure Classification

If a candidate passes PM selection and deterministic gates but no trade row is
created, the system must classify the execution failure.

Acceptance criteria:

1. Execution failures must be distinct from PM rejection and gate rejection.
2. Failures must include candidate ID, profile, symbol, intended action,
   attempted quantity, and reason code.
3. Execution failures must not silently fall back to legacy free-form order
   construction.
4. A failed execution must not leave the candidate in a registered or reserved
   state.

### Requirement 7 - Post-Entry Lifecycle Checklist

Every executed candidate-backed trade must write a post-entry lifecycle checklist.

The checklist must include:

- trade row created,
- position row created or updated,
- stop registered,
- target registered,
- thesis/invalidation recorded,
- position timer/monitor armed,
- review lineage linked.

Acceptance criteria:

1. The lifecycle checklist must be stored as structured telemetry linked by
   candidate ID and trade ID.
2. Missing lifecycle components must produce a warning event.
3. Missing lifecycle components must not automatically close a position unless
   an existing safety rule requires it.
4. Daily review must surface any executed trade with incomplete lifecycle state.

### Requirement 8 - Daily Candidate Loss Summary

The daily review/CEO context must include a candidate loss summary.

Acceptance criteria:

1. The summary must report counts for:
   - signals seen,
   - candidates built,
   - preflight failed,
   - offered to PM,
   - PM rejected by reason code,
   - PM accepted,
   - gate/sizing rejected,
   - execution failed,
   - executed,
   - lifecycle incomplete.
2. The summary must identify the top 3 blocker reason codes for the day.
3. The summary must distinguish zero trades caused by no candidates from zero
   trades caused by PM/gate/execution blockers.
4. The summary must be queryable from persisted telemetry, not reconstructed
   from logs alone.

### Requirement 9 - Compatibility And Rollout

The mitigation layer must be compatible with existing candidate-ID mode.

Acceptance criteria:

1. Existing tests for candidate registry, candidate prompt builder, swing
   candidate generation, candidate pipeline, and decision contract must pass.
2. The new rejection reason code field may be optional for a short rollout
   window, but missing codes must be observable.
3. Feature flags may be used for strict enforcement, but default behavior should
   improve observability without disabling live candidate flow.
4. No schema migration may drop or rewrite existing candidate/trade history.

## Acceptance Criteria

The implementation is acceptable when:

1. A PM rejection for an offered candidate includes a bounded reason code and
   full rationale.
2. A preflight-passed candidate cannot be rejected as "missing entry/stop/target"
   without producing explicit contract-violation telemetry.
3. A candidate that cannot be sized or cannot pass deterministic execution
   prerequisites is either excluded before PM with a structured reason, or
   offered only under explicit observe-mode configuration.
4. An accepted but unexecuted candidate can be traced to the first blocking
   stage without reading raw logs.
5. An executed candidate-backed trade records a lifecycle checklist.
6. Daily review/CEO output can say whether the day produced zero trades because
   of signal scarcity, candidate construction, PM rejection, gate/sizing, or
   execution/lifecycle failure.
7. Existing candidate and swing tests do not regress.

## Open Questions

1. Should strict reason-code enforcement be enabled immediately, or should
   missing codes warn for one trading day first?
2. Should known non-executable candidates be excluded from PM prompts entirely,
   or shown in observe mode for calibration?
3. Should preflight summary reuse existing gate code directly, or run a lighter
   deterministic subset to avoid side effects before PM selection?
4. Should the daily candidate loss summary live in `daily_review`, `ceo`, or a
   separate agent memory key consumed by both?
