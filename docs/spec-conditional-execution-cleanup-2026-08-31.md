# Conditional Execution Cleanup Spec

Date: 2026-08-31

## Problem

The paper trader has overlapping deferred-entry execution systems in source and
runtime configuration:

- `PENDING_ORDER_MODE`: the current live mechanism for deferred paper entry.
- `TRIGGERED_PLAN_MODE`: an older mechanism for creating trade plans and later
  executing them when trigger conditions are met.
- Setup-watch and fast-path monitoring: adjacent watch/alert systems that can
  make execution behavior harder to reason about when their boundary is unclear.

On 2026-08-31, this overlap caused a production PM decisioning failure. The live
runtime had `PENDING_ORDER_MODE=enabled`, but `agents/portfolio_manager.py`
still reached a legacy triggered-plan import path:

```text
cannot import name 'TRIGGERED_PLAN_MODE' from 'utils.gate_config'
```

The service was still running, analyst refreshes continued, and price monitoring
continued, but the PM decisioning phase failed mid-cycle. This created the worst
kind of paper-trading failure: the runtime looked alive while an execution-critical
phase could silently stop making complete decisions.

The earlier AMD stale-entry miss from the same morning is related but distinct.
That failure correctly rejected a stale entry after price had already crossed the
short target. The correct long-term behavior is not to recalculate and chase the
trade, but to drop or expire the stale opportunity and rely on the conditional
execution system to recognize a fresh setup if it appears again.

## Proposed Fix

Make pending orders the only active conditional-entry execution path.

Triggered-plan code should be retired from active runtime behavior. The system
may keep historical tables, documentation, or archived files for reference, but
no supported production path should create, monitor, trigger, or execute a trade
through `TRIGGERED_PLAN_MODE`, `plan_monitor`, `plan_executor`, or
`trade_plan_registry`.

The intended flow should be:

```text
PM candidate
-> immediate fill if executable and safe
-> pending order if valid but waiting for price
-> monitor cached live-ish data
-> fill, expire, cancel, reject, or revalidate
-> write audit events
```

## Goals

- Prevent retired triggered-plan imports from breaking PM decisioning.
- Standardize deferred paper entry around pending orders.
- Preserve the good idea behind triggered plans: conditional execution from fresh
  market data when stated conditions are met.
- Prevent stale favorable paper fills.
- Avoid recalculating and chasing stale opportunities after the original setup has
  already moved through its target or invalidation area.
- Keep execution state auditable for reviewer, CEO, dashboard, and future public
  stream surfaces.
- Make production deployment verification simple enough to run before market open.

## Non-Goals

- This cleanup does not remove analyst signal generation.
- This cleanup does not remove PM candidate construction.
- This cleanup does not require deleting historical triggered-plan database tables
  or audit rows.
- This cleanup does not require live brokerage integration.
- This cleanup does not require full bracket-order simulation for exits.
- This cleanup does not require a dashboard redesign beyond removing misleading
  active triggered-plan surfaces.

## Requirements

### Canonical Deferred Entry

- The system must treat pending orders as the canonical deferred-entry model.
- A PM-approved entry that is valid but not immediately executable at the desired
  price must route to pending-order creation when pending-order requirements pass.
- A PM-approved entry that is immediately executable may enter through the normal
  paper-entry path only after fresh-price and risk checks pass.
- A candidate whose fresh price has already crossed its intended target or
  invalidation area must be dropped, expired, or recorded as missed. It must not
  be recalculated into a chase trade in the same decision path.

### Triggered-Plan Retirement

- No active runtime path may import `TRIGGERED_PLAN_MODE` as part of PM decisioning.
- No active runtime path may create a new triggered trade plan.
- No active runtime path may execute a trade through `plan_executor`.
- The scheduler must not register `plan_monitor` in any supported production mode.
- Legacy triggered-plan modules must either be deleted, moved to an archive path,
  or made impossible to import from production code.
- Deprecated triggered-plan environment variables must be removed from production
  launch configuration or explicitly documented as ignored compatibility shims.

### Pending-Order Behavior

- `PENDING_ORDER_MODE=enabled` must remain the expected production setting.
- The pending-order monitor must run during regular market hours when pending
  order mode is enabled.
- Pending orders must have explicit states: pending, filled, expired, canceled,
  and rejected.
- Pending-order fills must be based on cached live-ish market data or fresh OHLC
  bars whose timestamps are inside the order's active window.
- Fill-time validation must re-check risk, buying power, exposure, same-symbol
  constraints, cooldowns, and market-data freshness.
- Duplicate active pending orders for the same profile, symbol, side, and setup
  type must be rejected, replaced, or explicitly superseded according to one
  documented rule.

### Setup Watch and Fast Path Boundaries

- Setup-watch code may remain as candidate observation, telemetry, or learning
  infrastructure.
- Fast-path code may remain as alerting or candidate acceleration infrastructure.
- Neither setup-watch nor fast-path code may bypass pending-order or immediate-fill
  risk checks.
- Any path that results in a paper trade must converge through the same final
  execution validation and audit event flow.

### Audit and Dashboard

- Trade events must distinguish immediate fills, pending orders, expired orders,
  canceled orders, rejected orders, and missed stale opportunities.
- Dashboard decision logs must not imply that a PM-approved plan became a trade
  unless an actual trade row was created.
- Dashboard/API surfaces must show pending orders as the active deferred-entry
  system.
- Retired triggered-plan state must not appear as live execution work.

### Deployment Safety

- Deployment must not leave source/runtime in a mixed state where a newer PM file
  depends on missing config or utility modules.
- Runtime verification must include an import smoke test for PM decisioning.
- Runtime verification must confirm that `PENDING_ORDER_MODE` is enabled.
- Runtime verification must confirm that `plan_monitor` is not registered.
- Runtime verification must confirm that the pending-order monitor is registered
  and executing during market hours.
- Runtime verification must check recent logs for PM decisioning failures,
  import errors, and tracebacks.

## Acceptance Criteria

- PM decisioning can complete with `PENDING_ORDER_MODE=enabled` and no
  `TRIGGERED_PLAN_MODE` constants present in active config.
- `agents/portfolio_manager.py` has no production dependency on triggered-plan
  modules or flags.
- `plan_monitor`, `plan_executor`, and `trade_plan_registry` are absent from active
  production imports.
- Existing pending-order tests pass after cleanup.
- Tests prove that no scheduler path registers `plan_monitor`.
- Tests prove that stale crossed-target candidates are dropped or recorded as
  missed, not recalculated into same-cycle chase entries.
- Tests prove that valid wait-for-price entries create pending orders when they
  pass risk and freshness gates.
- Live deployment shows normal service health, working API response, active price
  monitor, active pending-order monitor, and no recent PM import failures.

## Recommended First Patch

Start with source cleanup and tests, not database deletion.

1. Remove `_maybe_create_trade_plan` and triggered-plan imports from
   `agents/portfolio_manager.py`.
2. Remove scheduler registration paths and tests for `plan_monitor`.
3. Delete or archive `utils/plan_monitor.py`, `utils/plan_executor.py`,
   `utils/trade_plan_registry.py`, and `utils/entry_zone.py` if they are not used
   by pending orders.
4. Replace triggered-plan tests with pending-order assertions that cover stale
   entry, wait-for-price entry, fill, expiry, cancellation, and duplicate handling.
5. Keep historical DB tables and old docs for now, clearly marked retired.
6. Deploy to the Mac runtime and run a pre-market verification checklist.

## Operational Note

The temporary live compatibility shim added on 2026-08-31 is a guardrail, not the
desired final architecture. It prevents the current import crash by defaulting
triggered-plan mode to disabled, but it should be removed after active production
code no longer references triggered-plan symbols.
