# PM Candidate-Backed Entry Execution

Date: 2026-06-23
Priority: P0 upstream decision integrity
Status: proposed
Scope: new-entry Portfolio Manager decisions only

## Problem statement

The paper trader is still allowing the Portfolio Manager to act as both:

- trade selector,
- order geometry author.

That lets the PM model emit executable-looking fields such as `entry_price`,
`stop_loss`, `target`, and `quantity` even when deterministic candidate geometry
already exists. On 2026-06-23, live blocked-candidate records showed the PM
structured output and PM rationale disagreeing materially:

- TSLA moderate SHORT stated "R:R 2.0:1 meets threshold" while its structured
  fields computed to about `0.42:1`, then `0.27:1` after minimum stop adjustment.
- MU moderate BUY stated risk around `2.17` points and quantity `97`, while its
  structured fields implied risk around `10.30` points and quantity `38`,
  computing to about `0.42:1`.

The risk geometry gate correctly rejected these trades, but that is downstream
damage control. Adding more gates does not fix the upstream ownership problem.

The intended architecture is:

```text
code builds valid trades; LLM makes judgment calls
```

## Goal

Make deterministic candidate geometry the only source of truth for live new-entry
execution.

The PM should select, reject, or request bounded adjustments to trusted candidates.
It should not create executable symbols, prices, stops, targets, or absolute
quantities.

The desired flow is:

```text
Analyst signal
-> deterministic geometry candidates
-> candidate registry
-> PM candidate-id judgment
-> deterministic order materialization
-> existing execution/risk validation
-> paper trade
```

## Non-goals

This spec does not:

- loosen reward-to-risk thresholds,
- weaken existing risk gates,
- add another LLM arithmetic checker,
- make model prose authoritative over computed geometry,
- redesign exits, maintenance reviews, or position lifecycle governance,
- require the PM to discover symbols,
- remove PM judgment over whether a candidate is worth taking,
- fully specify implementation tasks or deployment sequencing.

## Ownership model

### Analyst

The Analyst owns setup interpretation:

- direction,
- catalyst and thesis,
- confidence,
- relevant technical and market context,
- key levels and invalidation concepts.

The Analyst does not own executable order construction.

### Deterministic geometry engine

Trusted code owns executable candidate construction:

- symbol,
- direction/action,
- setup type,
- entry price,
- stop price,
- target price,
- computed reward-to-risk,
- trigger,
- invalidation basis,
- target basis,
- candidate expiration,
- candidate lineage.

### Portfolio Manager

The PM owns judgment:

- accept a candidate,
- reject a candidate,
- reduce risk with a bounded multiplier,
- explain profile-specific rationale,
- optionally request a supported deterministic adjustment.

The PM does not own executable price geometry.

### Execution and risk layer

Trusted code owns order materialization and final invariant checks:

- candidate lookup,
- stale/unknown candidate rejection,
- quantity calculation,
- risk multiplier application,
- cash and exposure checks,
- existing gate evaluation,
- trade creation and audit logging.

## Requirements

### Requirement 1 - Live entries must be backed by a candidate ID

Every new live entry order must resolve from a valid `candidate_id`.

A new-entry order must not execute if it originated only from PM-returned
`symbol`, `entry_price`, `stop_loss`, `target`, or `quantity` fields.

### Requirement 2 - Candidate geometry is authoritative

For accepted candidates, trusted code must derive the following from the candidate
registry or equivalent deterministic candidate source:

- symbol,
- direction/action,
- setup type,
- entry price,
- stop price,
- target price,
- base reward-to-risk,
- trigger,
- invalidation basis,
- target basis.

PM-returned copies of these fields must be ignored, rejected as a contract
violation, or retained only as non-authoritative audit data.

### Requirement 3 - PM response contract is bounded

For new-entry decisions, the PM response must be limited to:

- `candidate_id`,
- `decision`: `accept` or `reject`,
- optional `risk_multiplier`,
- optional `rationale`,
- optional supported `adjustment_request`.

The PM response must not require or permit executable authority over:

- `symbol`,
- `entry_price`,
- `stop`,
- `stop_loss`,
- `target`,
- `target_price`,
- `quantity`,
- sector labels as executable instruments.

### Requirement 4 - Risk multiplier can reduce, not invent

If enabled, `risk_multiplier` may reduce deterministic position size.

It must not:

- increase risk above the profile maximum,
- alter entry price,
- alter stop price,
- alter target price,
- change direction,
- change setup type,
- convert one candidate into another.

### Requirement 5 - Adjustments are requests, not executable edits

If PM adjustments are supported, the PM may only request bounded adjustment types,
such as:

- `wider_stop_lower_size`,
- `smaller_size`,
- `wait_for_retest`,
- `reject_until_next_cycle`.

Trusted code must decide whether an adjustment request can produce a new valid
candidate. The PM must not directly emit adjusted executable geometry.

Unsupported, incomplete, or unsafe adjustment requests must fail closed.

### Requirement 6 - Legacy freeform PM entries are non-executable

The legacy PM path that emits freeform entry tickets may remain available for
shadow comparison or telemetry, but it must not create live new-entry trades once
candidate-backed execution is enabled.

If the legacy path produces a plausible order, it must be recorded as shadow or
diagnostic output unless it can be resolved to a valid current candidate ID.

### Requirement 7 - Existing gates remain invariant checks

Risk and execution gates remain active as invariant checks.

They should verify candidate-backed orders, not routinely discover that the PM
invented bad arithmetic. A high rate of gate rejections due to PM-authored
geometry must be treated as an upstream contract failure, not as evidence that
more gates are needed.

### Requirement 8 - Candidate IDs must be current and profile-scoped

A PM decision is non-executable when its `candidate_id`:

- is unknown,
- belongs to another profile,
- belongs to another cycle,
- has expired,
- was already consumed,
- does not match a current eligible signal,
- has missing or invalid deterministic geometry.

No fallback symbol or price inference may occur for invalid candidate IDs.

### Requirement 9 - Telemetry must distinguish upstream contract failure

Telemetry must distinguish:

- PM rejected candidate,
- PM accepted candidate and code executed/rejected it,
- PM output contract violation,
- invalid/stale candidate ID,
- legacy freeform output ignored,
- deterministic gate rejected candidate-backed geometry.

This separation is required so operator review does not confuse bad PM contracts
with bad gate policy.

### Requirement 10 - Tests must enforce the ownership boundary

Test coverage must prove:

- accepted candidate IDs preserve registry entry/stop/target,
- PM-returned price fields cannot override candidate geometry,
- `risk_multiplier` changes sizing only,
- unknown/stale candidate IDs cannot execute,
- legacy freeform PM output cannot create live entries,
- candidate-backed accepted orders reach execution with `normalized=True` or an
  equivalent invariant-safe contract,
- current focused tests can run with `pytest` under Python 3.11.

### Requirement 11 - Checkpoint funnel logging must expose candidate flow

The system must emit structured checkpoint events that show where every entry
opportunity moved or died.

Required checkpoints:

- `analyst_signal_seen`,
- `candidate_registered`,
- `candidate_offered_to_pm`,
- `pm_candidate_accepted`,
- `pm_candidate_rejected`,
- `pm_contract_violation`,
- `order_materialized`,
- `gate_evaluated`,
- `order_fired`,
- `order_rejected`.

Each checkpoint should carry, when available:

- cycle ID,
- candidate ID,
- lineage ID,
- profile,
- symbol,
- setup type,
- stage,
- decision,
- reason code.

The daily review and CEO memo should be able to summarize:

```text
signals -> candidates -> PM accepts -> materialized orders -> gate passes -> trades
```

### Requirement 12 - SQLite candidate-state writes must be lock-hardened

Candidate and PM state transitions must not be silently lost to transient SQLite
lock contention.

The runtime must:

- ensure WAL mode is enabled,
- configure a nonzero SQLite busy timeout,
- retry lock/timeout operational errors around candidate and PM state writes,
- log retry attempts and final failures as structured events,
- avoid retrying unrelated database errors as if they were lock contention.

The retry policy should be small and bounded, such as three attempts with
short backoff.

## Acceptance criteria

The spec is satisfied when:

1. Live new-entry trades can only be created from candidate-backed decisions.
2. The PM no longer has executable authority over entry, stop, target, or
   absolute quantity.
3. Legacy freeform PM entry output is disabled for live execution or limited to
   shadow/diagnostic use.
4. Candidate-backed accepted decisions are materialized by trusted code.
5. Existing gates continue to run but primarily validate invariants.
6. Telemetry shows whether a failure came from PM contract, candidate resolution,
   deterministic materialization, or gate rejection.
7. Focused pytest coverage passes in the repo Python 3.11 environment.
8. CEO daily review can identify the stage where candidates were lost without
   manual log spelunking.
9. Transient SQLite lock errors around candidate/PM state transitions are retried
   and surfaced instead of silently dropping candidate flow.

## CEO memo alignment

The 2026-06-23 CEO Daily Operating Memo identified:

- database contention and missing execution visibility,
- zero trades across June 22-23 despite many analyst signals,
- possible candidate-ID pipeline regression after June 8 hardening work.

This spec addresses those issues by:

- removing PM-authored live geometry from the execution path,
- making candidate-backed geometry the only live entry source,
- adding checkpoint funnel visibility,
- hardening candidate-state writes against SQLite locks,
- using replay and stage counts to distinguish PM selection, candidate creation,
  gate rejection, executor failure, and true lack of valid setups.

## Recommended build order

### Phase 0 - Preflight and preservation

Before logic changes:

- preserve current dirty work,
- keep `requirements.txt` with `pytest`,
- force-add this ignored spec file only if it should be committed,
- run tests through `.venv/bin/python -m pytest`,
- deploy live only after focused tests pass.

### Phase 1 - Candidate-backed execution only

Implement candidate-backed live entry execution first.

Required behavior:

- enable candidate-ID mode for live new entries,
- disable legacy freeform PM entry execution,
- resolve accepted decisions only through current candidate IDs,
- materialize symbol, direction, entry, stop, target, setup type, and base
  reward-to-risk from trusted candidate state,
- allow PM risk multipliers to reduce size only,
- reject or ignore PM-returned executable price fields.

Success condition:

```text
No new live entry can exist without candidate ID lineage.
```

### Phase 2 - Checkpoint funnel logging

Add the structured checkpoints in Requirement 11 immediately after or alongside
Phase 1.

Success condition:

```text
CEO memo can say exactly where candidates died.
```

### Phase 3 - SQLite lock hardening

Add WAL verification, busy timeout, and bounded retry/backoff around candidate
and PM state writes.

Success condition:

```text
No silent candidate loss from transient SQLite locks.
```

### Phase 4 - Replay and diagnosis pass

Replay or summarize June 22-23 candidate flow through the new visibility model.

Compare:

- candidates generated,
- candidates offered,
- PM accepts and rejects,
- contract violations,
- materialized orders,
- gate rejections,
- executed trades,
- shadow outcomes.

Success condition:

```text
The team knows whether zero-trade days are caused by candidate generation,
PM selection, materialization, gates, executor failure, or true lack of valid setups.
```

## Related specs

- `specs/pm-candidate-id-selection-contract-2026-06-05.md`
- `specs/pm-decision-geometry-integrity-and-provenance-2026-06-11.md`
- `specs/pm-entry-geometry-source-of-truth-2026-05-19.md`
