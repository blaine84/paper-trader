# Spec: Swing Candidate Pipeline for Paper Trader

Date: 2026-07-06
Status: Draft
Owner: Blaine / Cirrus
Target implementer: Kiro or Cirrus

## Problem

The paper-trader runtime is now mechanically capable of executing paper trades
through Postgres-backed PM candidate mode, but the live pipeline is not producing
tradable candidates often enough.

Recent production behavior shows:

- Analyst refresh produces fresh LONG/SHORT/HOLD signals.
- PM candidate mode is enabled.
- PM cycles run successfully for all profiles.
- Price monitor and alert dispatch write live telemetry.
- PM still reports `Candidate-ID mode: no eligible candidates for this cycle.`

The main cause is a mismatch between the analyst's current setup vocabulary and
the candidate pipeline's executable setup vocabulary.

Current executable setup types are narrowly intraday-oriented:

- `momentum_fade`
- `news_breakout`
- `gap_and_go`
- `technical_breakout`
- `vwap_reclaim`

Recent analyst outputs use labels such as:

- `sector_rotation`
- `risk_off_macro_short`
- `directional_confusion_breakout`

Those labels may represent real swing ideas, but they are not consistently
converted into candidate geometry. The system therefore sees ideas but does not
offer the PMs executable swing candidates.

## Proposed Fix

Add a controlled swing candidate bridge that converts eligible analyst signals
into executable swing candidates with swing-specific setup labels, geometry,
risk rules, and observability.

The goal is not to make every analyst label executable. The goal is to define
which signal patterns may become swing trades, which labels remain diagnostic,
and what safety gates are required before PMs can accept them.

Intraday candidate mode should remain available, but the next active trading
push should focus on swing setups because swing trading better fits the current
local-model cadence, Postgres-backed audit trail, and CEO/reviewer learning
loop.

## Goals

1. Produce a small number of legitimate paper swing candidates from fresh
   analyst signals.
2. Avoid turning ambiguous diagnostic labels into automatic trade candidates.
3. Give PMs concrete swing geometry: entry, stop, target, risk/reward, and
   expected holding horizon.
4. Keep conservative profile cautious while allowing moderate/aggressive
   profiles to collect paper trading reps.
5. Preserve observability for why a signal did or did not become a candidate.
6. Keep intraday telemetry running without requiring intraday perfection before
   collecting swing results.

## Non-Goals

1. Do not replace the existing intraday price monitor.
2. Do not disable scheduled PM cycles.
3. Do not allow every arbitrary analyst setup label to execute.
4. Do not make real-money trading decisions.
5. Do not remove existing stop, target, thesis-invalidation, review, or CEO
   workflows.
6. Do not require a new database migration beyond any additive fields needed for
   swing metadata.

## Requirements

### Swing Setup Taxonomy

1. The system must define a closed set of executable swing setup types.
2. Initial executable swing setup types should include:
   - `sector_rotation_swing`
   - `risk_off_macro_short`
   - `breakout_retest`
   - `pullback_continuation`
   - `relative_strength_swing`
   - `support_bounce_swing`
   - `failed_breakdown_reclaim`
3. Existing intraday executable setup types must remain valid.
4. Diagnostic setup labels must remain allowed as analyst output but must not
   become executable candidates by default.
5. `directional_confusion_breakout` must be treated as diagnostic by default.
6. A diagnostic label may be converted into an executable setup only when
   deterministic evidence identifies a cleaner setup, such as a true
   `technical_breakout`, `breakout_retest`, or `failed_breakdown_reclaim`.

### Setup Normalization

7. Analyst setup labels must pass through a normalization layer before candidate
   construction.
8. `sector_rotation` may normalize to `sector_rotation_swing` when:
   - signal is LONG or SHORT,
   - confidence is at least medium,
   - strength is at least moderate, and
   - the symbol has usable support/resistance or trend context.
9. `risk_off_macro_short` may remain executable for SHORT swing candidates when
   bearish trend and risk context are present.
10. `directional_confusion_breakout` must remain non-executable unless
    deterministic technical context can resolve direction and trade shape.
11. Signals with setup type `error` must never become executable candidates.
12. Signals created from data-provider errors, including Finnhub 429 fallbacks,
    must never become executable candidates.
13. Unknown setup labels may be preserved for review, but must carry a warning
    and must not be executable unless explicitly mapped.

### Swing Candidate Geometry

14. Swing candidates must include:
    - symbol
    - direction
    - normalized setup type
    - entry price or entry condition
    - stop price
    - target price
    - risk/reward
    - expected holding horizon
    - invalidation basis
    - source signal ID or source signal timestamp
15. Swing stops must be wider than fast intraday stops and should be based on
    technical levels, ATR, recent swing low/high, or setup-specific invalidation.
16. Swing targets must be based on realistic multi-day levels, measured moves,
    prior highs/lows, or sector-relative continuation targets.
17. Default expected holding horizon should be 2-10 trading days unless a setup
    type defines a narrower range.
18. Swing candidates must not expire only because an intraday entry window has
    closed.
19. Swing candidates may expire when:
    - the source signal is stale,
    - price moves too far from entry,
    - setup invalidation occurs,
    - broader regime contradicts the thesis, or
    - a maximum candidate age is reached.

### Profile Rollout

20. Conservative profile should initially observe or reject swing candidates
    unless confidence and risk/reward are unusually strong.
21. Moderate profile may accept small paper swing trades.
22. Aggressive profile may accept normal paper swing trades within existing
    risk limits.
23. Moderate and aggressive profiles must not both create overlapping
    same-symbol exposure unless correlation/same-symbol policy explicitly
    permits it.
24. Candidate construction should support profile-specific sizing assumptions.

### Risk And Exposure Controls

25. Swing position sizing must be smaller than or equal to existing profile risk
    budgets and must account for wider stops.
26. The system must enforce same-symbol exposure checks before opening a swing
    trade.
27. The system should enforce sector/correlation warnings for concentrated
    swing exposure.
28. Mega-cap high-volatility symbols such as AMD, NVDA, TSLA, MSTR, and MU must
    keep existing cooldown and stop-buffer protections.
29. Swing trades must always have an explicit stop and target before execution.
30. A PM may reject an otherwise valid swing candidate, but the rejection reason
    must be persisted.

### Candidate Mode Integration

31. Swing candidates must use the existing candidate registry and PM candidate
    audit flow where practical.
32. Candidate records must distinguish swing candidates from intraday
    candidates.
33. PM prompts must clearly show expected holding horizon and setup type.
34. PM prompts must not present diagnostic-only labels as executable candidates.
35. Candidate lifecycle states must remain auditable through candidate events,
    trade events, and PM notes.
36. If no swing candidates are built, PM notes should explain whether the reason
    was:
    - no fresh analyst signals,
    - no executable setup mapping,
    - missing geometry,
    - failed risk gates,
    - stale data,
    - same-symbol/correlation exposure, or
    - profile policy.

### Analyst Prompt And Output Contract

37. Analyst prompts should prefer canonical executable setup labels when the
    analyst believes a trade is actually actionable.
38. Analyst prompts should reserve diagnostic labels for ambiguous, conflicting,
    or non-actionable conditions.
39. Analyst output may include both the raw observed setup and normalized
    suggested setup when useful.
40. Analyst output must continue to support `llm_veto_reason` and
    `veto_evidence` when model direction conflicts with deterministic sanity.
41. Veto repair must remain local by default through `ANALYST_VETO_REPAIR_TIER`.

### Observability

42. Logs and/or database events must show:
    - raw setup label,
    - normalized setup label,
    - whether candidate construction was attempted,
    - whether candidate construction succeeded,
    - candidate rejection or block reason,
    - PM accepted/rejected outcome, and
    - final trade or no-trade result.
43. Dashboard/API behavior must remain stable when no swing candidates exist.
44. Existing CEO/reviewer flows should be able to inspect swing trades and their
    thesis/invalidation trail.

## Acceptance Criteria

The swing candidate bridge is acceptable when:

1. Fresh analyst signals with eligible swing setup labels can produce PM
   candidates.
2. Signals with `error` setup type never produce candidates.
3. `directional_confusion_breakout` does not produce candidates unless resolved
   into a cleaner executable setup by deterministic evidence.
4. At least one test covers `sector_rotation` normalization into
   `sector_rotation_swing`.
5. At least one test covers `risk_off_macro_short` becoming a SHORT swing
   candidate.
6. At least one test verifies diagnostic setup labels are preserved but
   non-executable.
7. Candidate records include swing-specific holding horizon and geometry.
8. PM notes explain why no candidate was produced when analyst signals exist but
   are not executable.
9. Moderate/aggressive profiles can receive swing candidates without enabling
   broad conservative-profile trading.
10. No existing intraday candidate tests regress.
11. Production logs after deployment show candidate construction attempts for
    fresh swing-eligible signals.
12. If a PM accepts a swing candidate, the resulting paper trade has explicit
    entry, stop, target, setup type, profile, and thesis/invalidation metadata.

## Out of Scope

This spec does not require:

1. A full strategy rewrite.
2. A new PM agent.
3. Real-money broker integration.
4. Disabling intraday alerts.
5. A dashboard redesign.
6. A historical swing-trading backtester before first paper deployment.
7. Perfect local model output before swing candidates can be attempted.

## Open Questions

1. Should conservative profile be observe-only for the first full trading day of
   swing candidate rollout?
2. What maximum concurrent swing positions should be allowed per profile?
3. Should sector-level exposure be warning-only or enforcing for paper swing
   trades?
4. Should swing candidates be generated only at scheduled PM cycles, or also
   from alert-dispatch events near major levels?
5. What default maximum swing candidate age should be used: 4 hours, end of day,
   or next market open?
6. Should failed analyst data-provider rows be excluded from PM candidate input
   entirely or shown as diagnostic context?

