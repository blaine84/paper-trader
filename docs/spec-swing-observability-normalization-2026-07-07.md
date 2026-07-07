# Spec: Swing Observability And Normalization Dev Cycle

Date: 2026-07-07
Status: Draft
Owner: Blaine / Cirrus
Target implementer: Kiro or Cirrus
Related spec: `docs/spec-swing-candidate-pipeline-2026-07-06.md`

## Problem

The swing candidate pipeline is deployed and mechanically active in observe
mode, but it is not yet ready for live paper swing trading.

The immediate Postgres telemetry/schema problems were repaired in commit
`8280c7f`:

- `pm_candidate_events.id` now has generated IDs on Postgres.
- Swing no-candidate events can be written.
- `compute_confidence_regime()` no longer compares a timestamp to a string.

The remaining blocker is behavioral, not mechanical.

Recent observe-mode logs show that the swing bridge sees analyst signals, but
rejects nearly all of them for reasons such as:

- `diagnostic_only`
- `unmapped_label`
- `insufficient_normalization_evidence`
- `context_mismatch`
- `failed_risk_gates`

That means the analyst layer is producing useful market descriptions, but the
system cannot yet reliably determine when those descriptions should become
executable swing candidates.

The next development cycle must improve the visibility and contract around
those rejections before loosening any gates.

## Proposed Fix

Run a focused swing-observability and normalization dev cycle.

The goal is to make every rejected swing signal explain itself in structured,
reviewable form, then use those observed rejection patterns to tune analyst
output and setup normalization.

This is not a hot patch. The system should remain in
`SWING_CANDIDATE_MODE=observe` until the new rejection telemetry proves that
candidate generation is understandable, repeatable, and appropriately cautious.

## Goals

1. Persist structured per-symbol swing rejection details for every PM cycle.
2. Distinguish diagnostic labels from executable swing setup candidates.
3. Show whether rejection happened because of label mapping, missing evidence,
   geometry construction, freshness, risk gates, profile policy, or exposure.
4. Preserve analyst raw labels for review while separately storing normalized
   executable labels when available.
5. Give the CEO/reviewer loop enough information to decide which normalization
   rules should be changed.
6. Tune normalization only after seeing concrete observe-mode examples.
7. Keep all trading behavior in observe mode until acceptance criteria are met.

## Non-Goals

1. Do not enable live swing trading.
2. Do not force swing candidates through by broadly relaxing gates.
3. Do not make `directional_confusion_breakout` executable by default.
4. Do not replace the existing swing candidate pipeline.
5. Do not redesign the dashboard.
6. Do not require a new backtester before improving observability.
7. Do not change real-money or broker behavior.

## Requirements

### Rejection Event Detail

1. Every PM cycle that evaluates swing candidates must produce a structured
   swing evaluation summary, even when zero candidates are built.
2. The summary must include cycle ID, profile ID, timestamp, candidate mode,
   and total signals evaluated.
3. The summary must include one per-symbol entry for each evaluated signal.
4. Each per-symbol entry must include:
   - symbol
   - raw signal direction
   - raw setup label
   - normalized setup label, if any
   - confidence
   - strength
   - construction attempted flag
   - construction succeeded flag
   - final rejection reason, if rejected
5. The final rejection reason must use a stable enum-like value rather than
   free-form prose.
6. Supported top-level rejection categories must include:
   - `diagnostic_only`
   - `unmapped_label`
   - `insufficient_normalization_evidence`
   - `context_mismatch`
   - `missing_geometry`
   - `stale_signal`
   - `stale_catalyst`
   - `failed_risk_gates`
   - `profile_policy`
   - `same_symbol_exposure`
   - `correlation_exposure`
   - `data_provider_error`
   - `unknown_error`
7. The cycle-level no-candidates event must summarize counts by rejection
   category.
8. The cycle-level event must preserve enough detail to answer, "Why did this
   specific symbol not become a swing candidate?"
9. The system must continue to fail open if rejection telemetry cannot be
   written, but the failure must be logged clearly.

### Normalization Evidence Contract

10. Setup normalization must distinguish between:
    - raw observed setup label,
    - diagnostic label,
    - normalized executable setup label, and
    - normalization rejection reason.
11. A raw analyst label must not become executable unless required evidence is
    present.
12. `sector_rotation` may normalize to `sector_rotation_swing` only when the
    signal has sufficient direction, confidence, relative context, and usable
    levels.
13. `risk_off_macro_short` may normalize to an executable short swing setup
    only when broader risk context and symbol-specific weakness are present.
14. `directional_confusion_breakout` must remain diagnostic unless deterministic
    evidence resolves it into a cleaner executable setup.
15. Unknown labels must be preserved for review and rejected as
    `unmapped_label`.
16. Signals with data-provider error markers must be rejected as
    `data_provider_error`.
17. Normalization must explain which evidence was missing when it rejects a
    setup as `insufficient_normalization_evidence`.

### Freshness And Catalyst Checks

18. Swing evaluation must check whether the source analyst signal is fresh
    enough for swing consideration.
19. Swing evaluation must check whether catalyst/news context is fresh enough
    for an overnight or multi-day hold.
20. Stale analyst signals must be rejected as `stale_signal`.
21. Stale or missing catalyst support must be rejected or flagged as
    `stale_catalyst`, depending on profile policy.
22. Freshness thresholds must be explicit and reviewable.

### Geometry And Risk Gates

23. If normalization succeeds but entry/stop/target construction fails, the
    rejection must be `missing_geometry`.
24. If geometry succeeds but risk/reward, stop distance, or sizing gates fail,
    the rejection must be `failed_risk_gates`.
25. If profile-specific policy blocks a candidate, the rejection must be
    `profile_policy`.
26. Same-symbol and correlation checks must have separate rejection categories.
27. A swing candidate must not be offered unless it has entry, stop, target,
    risk/reward, holding horizon, and invalidation basis.

### Reviewability

28. Rejection telemetry must be stored in `pm_candidate_events` or an existing
    compatible audit surface.
29. Stored event payloads must be JSON and safe to inspect from reviewer/CEO
    workflows.
30. The CEO/reviewer loop must be able to summarize:
    - top rejection categories,
    - most frequently rejected symbols,
    - labels that almost normalized,
    - missing evidence patterns, and
    - any candidates that would have been offered.
31. The daily review should distinguish "no swing opportunity" from "pipeline
    could not understand the analyst output."

### Mode And Rollout

32. `SWING_CANDIDATE_MODE=observe` must remain the default during this dev
    cycle.
33. Observe mode may write telemetry and shadow candidates, but must not open
    swing trades.
34. Moderate/aggressive profile behavior may be evaluated first; conservative
    profile should remain stricter.
35. No mode may trade a swing candidate unless the candidate passed
    normalization, freshness, geometry, and risk checks.

## Acceptance Criteria

This dev cycle is acceptable when:

1. A PM cycle with zero swing candidates writes a structured no-candidates
   event successfully.
2. The event contains per-symbol rejection detail, not just a single generic
   reason.
3. Rejection categories are stable and test-covered.
4. `sector_rotation`, `risk_off_macro_short`, and
   `directional_confusion_breakout` each have tests covering accepted and/or
   rejected normalization paths.
5. Stale analyst signals are rejected distinctly from stale catalyst context.
6. Missing geometry and failed risk gates are distinguishable in telemetry.
7. Observe-mode production logs show no `pm_candidate_events.id` errors and no
   confidence-regime timestamp errors.
8. After at least one trading session in observe mode, the rejection summary can
   identify the top 3 reasons swing candidates are not being offered.
9. No existing intraday candidate tests regress.
10. No live swing trading is enabled as part of this spec.

## Out Of Scope

This spec does not require:

1. Enabling swing trading.
2. Adding new broker integrations.
3. Replacing analyst prompts wholesale.
4. Building a dashboard redesign.
5. Reclassifying all historical setup labels.
6. Rewriting PM decision logic.
7. Backtesting every swing setup before observe-mode telemetry improves.

## Open Questions

1. Should swing rejection summaries be stored only as `pm_candidate_events`, or
   should a dedicated swing evaluation table be added later?
2. Should stale catalyst context be a hard reject for all profiles, or a warning
   for aggressive profile in observe mode?
3. Which deterministic evidence fields are mandatory for
   `sector_rotation_swing`?
4. Should `risk_off_macro_short` require market-index confirmation, sector ETF
   confirmation, or both?
5. What is the minimum useful observe window before tuning normalization: one
   full trading day, three PM cycles, or a fixed number of rejected signals?
