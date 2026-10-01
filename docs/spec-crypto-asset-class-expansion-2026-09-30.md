# Crypto Asset-Class Expansion (Bitcoin First)

Date: 2026-09-30
Status: Draft
Owner: Blaine
Target implementer: Kiro

Related specs:
- `docs/market-data-reliability-layer-2026-07-15.md`
- `docs/spec-fast-path-deterministic-execution-2026-08-20.md`
- `specs/pm-entry-geometry-source-of-truth-2026-05-19.md`
- `.kiro/specs/symbol-class-validation/design.md`

## Problem

Paper trader is a mature, equities-only system. Its trading logic (position
sizing, risk geometry, candidate pipeline, PM/analyst/reviewer agents,
paper-fill mechanics, P&L and bookkeeping) is asset-agnostic and would transfer
to Bitcoin with modest work. But two load-bearing assumptions are wired through
the entire runtime and do NOT hold for crypto:

1. **The US regular-session clock gates everything.** Nearly every scheduled
   job in `orchestrator.py` is wrapped in `_skip_closed_market_job` or
   `_skip_outside_regular_market_job` (09:30–16:00 ET, Mon–Fri). The
   `position_lifecycle_governance` module is built on session concepts that do
   not exist for a 24/7 market: a 15:45 ET `HARD_WALL`, end-of-day force-close,
   "overnight authorization," and a hardcoded `US_MARKET_HOLIDAYS` set. Bitcoin
   trades continuously, with no open, no close, and no holidays.

2. **Symbol classification has no crypto class.** `utils/symbol_class.py`
   recognizes eight asset classes; a symbol like `BTC` or `BTC-USD` falls
   through to `"unknown"`, which trips low-confidence flags, `gap_and_go`
   setup-reclassification quirks, and review gates across the candidate
   pipeline.

A third, smaller gap: **all market-data providers are equity-only as wired.**
`FinnhubClient` and the reliability layer's `provider_adapter` fetch US stock
bars from Finnhub / Alpaca stocks / yfinance. Crypto needs a distinct feed
(Alpaca has a separate crypto endpoint; Coinbase/Kraken are alternatives).

Treating Bitcoin as "just another watchlist symbol" would therefore produce
silent, dangerous misbehavior: force-closing a 24/7 position at a meaningless
"end of day," skipping the price monitor overnight while a BTC position runs
unguarded, and routing every BTC candidate through `unknown`-class review gates.

## Proposed Fix

Introduce **`crypto` as a first-class asset class with a 24/7 session model**,
so that time-based governance branches on asset class rather than assuming a US
equity session. Add a **crypto market-data provider adapter** behind the
existing provider contract, so the reliability layer treats a BTC quote the
same way it treats a SPY quote once the data is normalized.

The expansion is **additive and gated**. A `CRYPTO_TRADING_MODE` flag
(`disabled` default) keeps the live equities system untouched until crypto is
explicitly enabled. No equity behavior changes when the flag is off.

Core principle: **session-based governance is replaced by session-aware
governance.** For equities, nothing changes. For crypto, "end of day,"
"overnight," and "market closed" are redefined in terms of a continuous clock
with crypto-appropriate risk windows (e.g. a rolling max-hold timer and
volatility/funding windows) rather than a 16:00 ET wall.

## Goals

1. Add a `crypto` symbol class recognized deterministically, with
   crypto-appropriate setup-type validation.
2. Define a 24/7 session model so scheduler gating, the price monitor, and
   lifecycle governance behave correctly for a market that never closes.
3. Add a crypto market-data provider adapter that returns the reliability
   layer's canonical snapshot shape, with the same freshness/trust semantics.
4. Keep position sizing, risk geometry, candidate pipeline, PM/analyst/reviewer
   agents, and bookkeeping working for crypto with minimal change.
5. Make the whole expansion gated and default-off, so the live equities system
   is unaffected until crypto is explicitly enabled.
6. Preserve all existing safety gates (risk, exposure, sizing, stop/target,
   market-data reliability, audit) for crypto.

## Non-Goals

1. Do NOT connect a live crypto brokerage or execute real crypto orders — this
   remains a paper trader.
2. Do NOT support every coin in the first pass; Bitcoin (`BTC-USD`) only.
3. Do NOT loosen any existing equity risk, sizing, or execution gate.
4. Do NOT redesign the dashboard; crypto rows reuse existing surfaces with a
   session/asset-class label added.
5. Do NOT change equity behavior at all when `CRYPTO_TRADING_MODE=disabled`.
6. Do NOT build crypto-specific alpha (perp funding arbitrage, on-chain signals)
   in this pass; the goal is correct 24/7 mechanics, not new strategies.
7. Do NOT remove or weaken the US-equity lifecycle governance; crypto branches
   around it, it is not replaced globally.

## Requirements

### Requirement 1 — Crypto Symbol Class

The system shall classify crypto symbols deterministically as a first-class
asset class.

Acceptance criteria:

1. `classify_symbol()` shall return `"crypto"` for canonical crypto symbols
   (at minimum `BTC`, `BTC-USD`, `BTCUSD`), normalizing format variants to one
   canonical internal symbol.
2. `"crypto"` shall be added to the non-single-stock class set so single-stock-
   only setup types (e.g. `gap_and_go`) are reclassified or flagged, not
   silently accepted.
3. A crypto symbol shall never resolve to `"unknown"` solely because it is
   crypto; `unknown`-class low-confidence review flags shall not fire for a
   recognized crypto symbol.
4. Setup-type validation shall define which setups are valid for crypto
   (e.g. technical_breakout, trend continuation, range reversion) and reclassify
   or reject equity-only setups with audit metadata, mirroring the existing
   `validate_setup_for_symbol` three-way logic.
5. Classification shall remain purely deterministic with no LLM involvement.

### Requirement 2 — 24/7 Session Model

The system shall provide a session model that correctly represents a market
that is always open.

Acceptance criteria:

1. A session abstraction shall report, for any symbol, whether its market is
   open, and the symbol's asset class, without assuming a US equity calendar.
2. For `crypto` symbols the market shall always be considered open; weekend and
   `US_MARKET_HOLIDAYS` checks shall not apply.
3. For equity symbols the session model shall return exactly today's behavior
   (regression: no change).
4. `market_session` on a crypto snapshot shall be reported as `open`
   (never `closed`, `pre_market`, or `after_hours`).
5. The session model shall be the single source of truth consulted by scheduler
   gating, the price monitor, and lifecycle governance — not scattered
   `datetime` checks.

### Requirement 3 — Scheduler Gating For 24/7 Symbols

Scheduled jobs shall run for crypto symbols on a continuous schedule, not a US
equity session schedule.

Acceptance criteria:

1. `_skip_closed_market_job` and `_skip_outside_regular_market_job` shall be
   asset-class aware: for a crypto job they shall not skip on weekends,
   holidays, or outside 09:30–16:00 ET.
2. Crypto analyst/PM cycles shall be schedulable on a continuous cadence
   (e.g. every N minutes, 24/7), independent of equity session windows.
3. The crypto price monitor shall run continuously, including overnight and
   weekends, so a crypto position is never left unmonitored.
4. Equity jobs shall retain their existing session gating unchanged.
5. Crypto and equity schedules shall be able to coexist without one blocking the
   other (respect existing PM-cycle serialization where registries are shared).

### Requirement 4 — Crypto Lifecycle Governance

Position lifecycle governance shall apply crypto-appropriate rules instead of
US-session rules for crypto positions.

Acceptance criteria:

1. For crypto positions the `HARD_WALL` (15:45 ET) force-close, end-of-day
   force-close, and `US_MARKET_HOLIDAYS` logic shall NOT apply.
2. "Overnight authorization" shall not be required for crypto; time-based risk
   shall instead use a configurable rolling max-hold duration and/or
   crypto-appropriate risk windows.
3. A crypto position shall still resolve to exactly one lifecycle state and one
   decision, using states meaningful for a 24/7 market (e.g. within-max-hold,
   approaching-max-hold, max-hold-expired, invalid-stop-geometry,
   close-required).
4. Equity positions shall produce byte-for-byte the same lifecycle decisions as
   today (regression).
5. All crypto lifecycle decisions shall be auditable with reason codes, matching
   the existing `LifecycleDecision` structure.

### Requirement 5 — Crypto Market-Data Provider Adapter

The system shall fetch crypto market data through an adapter that returns the
reliability layer's canonical provider shape.

Acceptance criteria:

1. A crypto provider adapter shall return quotes in the canonical
   `s/c/h/l/o/pc/t/v` dict shape and candles in the canonical list shape, so the
   existing normalizer and `Snapshot` contract consume them unchanged.
2. At least one crypto provider shall be integrated (Alpaca crypto endpoint
   preferred for consistency with existing Alpaca usage; Coinbase/Kraken public
   endpoints acceptable as a no-key fallback for the prototype).
3. The adapter shall support BTC quotes and sub-daily candles (1/5/15/60 min).
4. Freshness and trust semantics shall apply to crypto snapshots exactly as to
   equities; a stale or cross-symbol crypto response shall be degraded/untrusted
   with a stable reason code.
5. The adapter shall be selected by asset class: crypto symbols route to the
   crypto provider, equity symbols continue to route to Finnhub/Alpaca/yfinance.
6. Crypto freshness thresholds may differ from equities (configurable) because
   crypto has no "market closed" state and quotes are continuous.

### Requirement 6 — Risk, Sizing, And Exposure For Crypto

Existing risk, sizing, and exposure gates shall apply to crypto with
crypto-aware parameters.

Acceptance criteria:

1. Position sizing shall support fractional quantities (BTC is not whole-share),
   including minimum fractional increments.
2. Risk geometry (stop/target/R:R) shall validate for crypto using the same math
   with crypto-appropriate ATR/volatility inputs.
3. Per-asset-class exposure caps shall be configurable so crypto exposure can be
   limited independently of equity exposure.
4. Crypto decimal precision shall be handled consistently (price and quantity),
   using `Decimal` where the reliability layer already does.
5. No crypto path shall bypass stop-loss, target, exposure, or sizing gates.

### Requirement 7 — Configuration And Gating

The entire crypto capability shall be configurable and default-off.

Acceptance criteria:

1. `CRYPTO_TRADING_MODE` (`disabled` | `shadow` | `enabled`) shall default to
   `disabled`.
2. When `disabled`, no crypto job registers, no crypto symbol enters any
   watchlist, and equity behavior is identical to pre-change.
3. A crypto watchlist shall be configurable separately from the equity
   watchlist.
4. Crypto cadence, max-hold duration, exposure caps, and freshness thresholds
   shall be configurable with safe defaults.
5. `shadow` mode shall allow crypto analysis/candidate generation and outcome
   recording without opening crypto paper trades, mirroring existing shadow
   patterns.

### Requirement 8 — Observability And Audit

Crypto activity shall be fully observable and distinguishable from equity
activity.

Acceptance criteria:

1. Trades, candidates, and events shall carry asset class so crypto and equity
   activity can be separated in logs, dashboard, and review output.
2. The P&L / bookkeeping ledger shall track crypto positions and realized/
   unrealized P&L with fractional quantities.
3. Daily/weekly review and CEO output shall be able to summarize crypto activity
   on its own continuous clock.
4. Market-data degradation telemetry shall apply to crypto snapshots.
5. Observability writes shall be fail-open: a crypto telemetry failure shall not
   crash the trading loop.

### Requirement 9 — Testing

The crypto expansion shall have focused regression and property-style tests.

Acceptance criteria:

1. Tests shall cover `classify_symbol()` returning `"crypto"` for all canonical
   BTC variants and setup-type validation for crypto.
2. Tests shall cover the 24/7 session model: crypto always open, equities
   unchanged on weekends/holidays/after-hours.
3. Tests shall cover scheduler gating: a crypto job runs on a weekend; an equity
   job still skips.
4. Tests shall cover crypto lifecycle governance: no hard-wall force-close for
   crypto; max-hold expiry produces the correct decision; equities unchanged.
5. Tests shall cover the crypto provider adapter: valid quote/candle, stale
   response, cross-symbol response, empty payload.
6. Tests shall cover fractional sizing and crypto exposure caps.
7. Existing equity tests shall not regress with `CRYPTO_TRADING_MODE=disabled`
   AND with it `enabled` for equity-only watchlists.

## Acceptance Criteria

This spec is acceptable when:

1. `BTC-USD` is recognized as a `crypto` symbol end-to-end and never resolves to
   `unknown`.
2. A 24/7 session model exists and is the single source of truth for crypto
   open/closed, consulted by scheduler, price monitor, and lifecycle governance.
3. A crypto position is monitored continuously (including overnight/weekends) and
   is never force-closed by the equity hard-wall.
4. Crypto market data flows through the reliability layer with correct
   freshness/trust state via a crypto provider adapter.
5. Position sizing supports fractional BTC and crypto exposure caps apply.
6. With `CRYPTO_TRADING_MODE=disabled`, the equities system behaves identically
   to before this change (full regression pass).
7. Focused crypto tests pass locally, and deployment includes compile/schema
   checks before restart.

## Out Of Scope

1. Live crypto brokerage / real order execution.
2. Coins beyond Bitcoin in the first pass.
3. Crypto-specific alpha (funding, on-chain, perps).
4. Dashboard redesign.
5. A historical crypto tick warehouse.
6. Loosening any existing trading gate.

## Phased Implementation Plan

- **Phase 0 — Data layer prototype (de-risk plumbing).** Crypto provider adapter
  returning canonical snapshots for BTC. Standalone, no runtime wiring. *(This
  is the companion prototype delivered alongside this spec:
  `utils/crypto_data_feed.py`.)*
- **Phase 1 — Classification + session model.** `crypto` symbol class; 24/7
  session abstraction; unit tests. No scheduler/governance wiring yet.
- **Phase 2 — Scheduler + price-monitor gating.** Make session gating
  asset-class aware; continuous crypto price monitor behind the flag.
- **Phase 3 — Lifecycle governance branch.** Crypto rules (max-hold, no
  hard-wall) alongside unchanged equity governance.
- **Phase 4 — Sizing/exposure + bookkeeping.** Fractional quantities, crypto
  exposure caps, asset-class-tagged ledger.
- **Phase 5 — End-to-end shadow mode, then enable.** Run crypto in `shadow`,
  validate outcomes, then flip to `enabled`.
