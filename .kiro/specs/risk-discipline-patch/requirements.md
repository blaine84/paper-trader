# Requirements Document: Risk Discipline Patch

## Introduction

The 2026-05-07 decision log showed two high-leverage risk-control weaknesses that should be fixed before broader behavior tuning:

1. **Missing/blank setup types can pass Setup Quality Gate** because the gate treats `setup_type == ""` as an unseen setup with zero cases and allows the trade.
2. **Maintenance stop updates are too noisy**. QQQ and NVDA received dozens of accepted stop changes in one afternoon, including loosening after prior tightening. This creates stop-thrash, weakens auditability, and lets LLM maintenance reviews churn risk levels without enough state discipline.

This patch intentionally focuses on these two issues only. Data-sanity checks, rejected-setup cooldowns, and catalyst reconfirmation are important but should remain separate follow-up specs to avoid turning this into software soup.

## Glossary

- **Portfolio_Manager**: `agents/portfolio_manager.py`, responsible for PM entry/maintenance decisions and trade execution.
- **Setup_Quality_Gate**: `utils/setup_quality_gate.py`, deterministic gate evaluating setup track record before entry.
- **Setup_Type**: Setup classification such as `technical_breakout`, `vwap_reclaim`, `momentum_fade`, or `sector_rotation_into`.
- **Missing_Setup**: A setup value that is absent, null, empty string, whitespace-only, or non-string/non-normalizable.
- **StopAuthority**: `utils/stop_authority.py`, centralized stop-price validation and stop update application.
- **Maintenance_Stop_Update**: A stop update requested from Portfolio_Manager maintenance review or reversal/close hold-tighten logic.
- **Maintenance_Review_Cadence**: Current PM review cadence is every 15 minutes from 9:30 AM–12:00 PM ET and every 30 minutes from 12:00 PM–4:00 PM ET. Price_Monitor runs every 60 seconds, but routine PM maintenance decisions do not.
- **Stop_Thrash**: Repeated stop updates on the same open trade over short intervals, especially changes smaller than market noise or loosening after prior tightening.
- **Monotonic_Tightening**: For long trades, accepted maintenance stops may only move upward; for short trades, accepted maintenance stops may only move downward.
- **Intraday_ATR_For_Stop_Delta**: The 5-minute intraday ATR value computed by the existing ATR helper using ATR(14) over recent 5-minute candles, when available. If the helper uses a different period today, this spec requires pinning it to ATR(14) for this threshold or documenting the exact existing period before implementation.
- **Exceptional_Loosening**: Out of scope for this patch. Routine maintenance stop loosening SHALL be impossible until a follow-up spec explicitly defines an exceptional loosening workflow.

## Requirements

### Requirement 1: Fail Closed on Missing Setup for New Entries

**User Story:** As the operator, I want new entries without a valid setup type to be rejected before execution, so that no trade can bypass setup-quality controls through an empty setup field.

#### Acceptance Criteria

1. WHEN a BUY or SHORT decision reaches the gate pipeline and setup type cannot be resolved from the PM decision or latest analyst signal, THE Portfolio_Manager SHALL reject the trade.
2. WHEN rejecting a trade due to Missing_Setup, THE system SHALL log a `gate_rejected` trade event with a clear reason containing `missing setup_type`.
3. THE Setup_Quality_Gate SHALL NOT treat empty string setup types as valid unseen setups.
4. THE Missing_Setup rejection SHALL happen inside the existing gate pipeline before Risk_Geometry_Gate evaluation and before any Trade row can be created; this uses the current gate order of Setup_Quality_Gate → Pre_Trade_Quality_Gate → Risk_Geometry_Gate and SHALL NOT require a broad pipeline restructure.
5. THE Missing_Setup rule SHALL apply to all PM profiles.

### Requirement 2: Normalize Setup Type Before Gate Evaluation

**User Story:** As the operator, I want setup type normalization to be deterministic, so that whitespace, casing, and alternate decision keys do not produce inconsistent gate behavior.

#### Acceptance Criteria

1. THE Portfolio_Manager SHALL resolve setup type from, in order: `decision.setup_type`, `decision.setup`, `signal.setup_type`, `signal.setup`.
2. THE resolved setup type SHALL be stripped of leading/trailing whitespace.
3. IF the stripped setup type is empty, THE result SHALL be Missing_Setup.
4. THE normalized setup type SHALL be used consistently for Setup_Quality_Gate, Risk_Geometry_Gate, Entry_Contract construction, confidence adjustment, and strategy multiplier lookup where applicable.
5. Existing valid setup strings SHALL preserve their semantic value; this patch SHALL NOT rename setup taxonomies.

### Requirement 3: Enforce Monotonic Maintenance Stop Tightening

**User Story:** As the operator, I want routine maintenance stop updates to only reduce risk, so that PM maintenance cannot loosen risk after tightening unless explicitly escalated.

#### Acceptance Criteria

1. FOR a LONG trade, a Maintenance_Stop_Update SHALL be rejected if the new stop is below the current stop.
2. FOR a SHORT trade, a Maintenance_Stop_Update SHALL be rejected if the new stop is above the current stop.
3. Equal stop values SHALL be treated as no-op and rejected or skipped without writing a new `stop_update_accepted` event.
4. A Maintenance_Stop_Update SHALL pass both the Monotonic_Tightening requirement and the Minimum Stop Change requirement before it may be accepted; monotonic-but-tiny changes SHALL NOT be accepted.
5. Rejections SHALL log `stop_update_rejected` with reason containing `non-monotonic maintenance stop` or equivalent.
6. This rule SHALL apply to maintenance sources including `portfolio_manager` maintenance review and `profit_manager` stop updates unless a source is explicitly documented as exempt.

### Requirement 4: Require Minimum Stop Change Before Accepting Maintenance Updates

**User Story:** As the operator, I want tiny stop changes ignored, so that the decision log is not polluted by noise-level stop adjustments.

#### Acceptance Criteria

1. A Maintenance_Stop_Update SHALL be accepted only if it changes the stop by at least the larger of:
   - 0.25% of current price, or
   - 0.25 × Intraday_ATR_For_Stop_Delta.
2. Intraday_ATR_For_Stop_Delta SHALL mean ATR(14) computed from 5-minute candles by the existing ATR helper, or the implementation SHALL document any existing helper period and adjust this requirement before coding.
3. IF Intraday_ATR_For_Stop_Delta is unavailable, THE system SHALL use only the 0.25% current-price threshold.
4. IF current price is unavailable, THE system SHALL fall back to a conservative absolute threshold of 0.1% of the current stop.
5. Stop changes below the threshold SHALL be skipped or rejected without mutating the Trade row, even if the change is directionally monotonic.
6. Skipped tiny updates SHALL be observable via log or trade event, but SHALL NOT create `stop_update_accepted` events.

### Requirement 5: Add Stop Update Cooldown Per Open Trade

**User Story:** As the operator, I want each open trade to have a cooldown between accepted maintenance stop updates, so that one LLM review loop cannot repeatedly churn stops.

#### Acceptance Criteria

1. The system SHALL enforce a configurable cooldown between accepted Maintenance_Stop_Update events for the same `trade_id`.
2. Default cooldown SHALL be 15 minutes, matching the fastest current Maintenance_Review_Cadence so that routine PM maintenance can accept at most one stop update per trade per normal morning review cycle and at most one per afternoon cycle.
3. IF PM maintenance cadence is later changed below 15 minutes, THE cooldown default SHALL be reviewed before deployment.
4. IF a new maintenance stop request arrives before cooldown expiry, THE system SHALL reject or skip it without changing the Trade row.
5. The cooldown SHALL be based on the timestamp of the latest accepted stop update for the same trade.
6. Cooldown rejection SHALL be logged with `stop update cooldown active` or equivalent.

### Requirement 6: Preserve Emergency Exit and Initial Stop Behavior

**User Story:** As the operator, I want the new stop-thrash controls to avoid blocking genuine protective exits or initial stop setup.

#### Acceptance Criteria

1. Initial stop creation at trade entry SHALL NOT be subject to cooldown or minimum-change rules.
2. Price-monitor stop-triggered exits SHALL NOT be delayed by stop-update cooldowns.
3. Target-triggered exits SHALL NOT be delayed by stop-update cooldowns.
4. Reversal/Close Review full-close or partial-close decisions SHALL NOT be blocked by stop-update cooldowns.
5. Exceptional_Loosening SHALL NOT be implemented in this patch; routine maintenance stop loosening SHALL be rejected. A future follow-up spec may define an exceptional loosening workflow with explicit reason metadata and auditable events.

### Requirement 7: Verification and Observability

**User Story:** As the operator, I want tests and logs that prove the patch is working, so that future reviews can distinguish blocked risk churn from successful risk updates.

#### Acceptance Criteria

1. Unit tests SHALL cover Missing_Setup rejection.
2. Unit tests SHALL cover monotonic long and short maintenance stop updates.
3. Unit tests SHALL cover tiny stop update rejection.
4. Unit tests SHALL cover cooldown rejection.
5. Existing stop authority, PM integration, and trade event tests SHALL continue passing.
6. Decision logs SHALL make it possible to count accepted vs rejected/skipped maintenance stop updates by symbol/profile/trade.
