# Requirements Document

## Introduction

The Portfolio Manager currently treats the absence of fresh analyst signals as an implicit exit signal, causing premature exits on positions whose original trade thesis remains intact. This was observed when a high-confidence AMD gap-and-go with VWAP and bullish EMA support was exited at +1.08% — well short of its target — simply because the analyst stopped refreshing signals.

This feature anchors all exit decisions to the original trade thesis captured at entry. Analyst signals become advisory inputs for open positions rather than authoritative exit triggers. A two-tier review system (maintenance vs. reversal/close) replaces the current flat decision loop, ensuring the system holds winners to target and only exits on genuine thesis invalidation.

## Glossary

- **Portfolio_Manager**: The agent (`agents/portfolio_manager.py`) that makes entry, hold, and exit decisions for each PM profile.
- **Entry_Contract**: A persistent record of the trade thesis captured at the moment of entry, containing entry price, stop price, target price, thesis narrative, setup type, and invalidation conditions.
- **Thesis_Invalidator**: A structured, machine-readable condition defined at entry that, if met, means the original trade thesis is no longer valid. Each invalidator is stored as a JSON object with fields: `type` (the condition kind, e.g., `"price_below_level"`, `"price_above_level"`, `"structure_break"`), `reference` (the anchor level or indicator, e.g., `"VWAP"`, `"162.50"`), `confirmation` (how the breach is confirmed, e.g., `"5m_close"`, `"tick"`), and `lookback_bars` (number of bars for confirmation, e.g., `1`). Example: `{"type": "price_below_level", "reference": "VWAP", "confirmation": "5m_close", "lookback_bars": 1}`.
- **Trade**: The database model (`db/schema.py`) representing an open or closed paper trade.
- **Position**: The database model representing a currently held position.
- **Analyst**: The agent (`agents/analyst.py`) that generates technical signals per symbol.
- **Signal**: A JSON record written to AgentMemory by the Analyst containing direction, strength, confidence, setup type, key levels, and invalidation.
- **Price_Monitor**: The agent (`agents/price_monitor.py`) that checks real-time prices against stops and targets.
- **Position_Health_Monitor**: The agent (`agents/position_health.py`) that performs hourly health checks on open positions.
- **DRIFTING_State**: A position state indicating that no new analyst signals have been received since entry, but the original thesis remains intact and no exit condition has been met.
- **Maintenance_Review**: The default review mode for open positions. Produces one of: hold, tighten stop, raise target, or trim partial. Does not produce close decisions.
- **Reversal_Close_Review**: A review mode invoked only when the thesis is clearly broken, opposing evidence crosses a defined threshold, or an explicit CLOSE/REVERSE condition is met. This is the only review path that can produce a full close decision.
- **PM_Profile**: A risk profile configuration (conservative, moderate, aggressive) in `models/pm_profiles.py`.

## Requirements

### Requirement 1: Persist Entry Contract at Trade Open

**User Story:** As a portfolio manager, I want the full trade thesis recorded at entry, so that all subsequent hold/exit decisions reference the original rationale rather than relying on fresh signals.

#### Acceptance Criteria

1. WHEN a BUY or SHORT action is executed, THE Portfolio_Manager SHALL persist an Entry_Contract containing: entry price, stop price, target price, thesis narrative (from rationale), setup type, and a list of Thesis_Invalidators stored as structured JSON objects.
2. THE Trade model SHALL store the Entry_Contract fields so that they are queryable for the lifetime of the trade.
3. WHEN an Entry_Contract is persisted, THE Portfolio_Manager SHALL extract Thesis_Invalidators from the analyst Signal's invalidation field and store each invalidator as a structured JSON object with fields: `type`, `reference`, `confirmation`, and `lookback_bars` (e.g., `{"type": "price_below_level", "reference": "VWAP", "confirmation": "5m_close", "lookback_bars": 1}`).
4. IF the analyst Signal does not contain an invalidation field, THEN THE Portfolio_Manager SHALL derive a default Thesis_Invalidator from the stop price level using the structured schema (e.g., `{"type": "price_below_level", "reference": "<stop_price>", "confirmation": "5m_close", "lookback_bars": 1}`) and log a warning.

### Requirement 2: Prohibit Exit on Signal Absence

**User Story:** As a portfolio manager, I want the system to never exit a position solely because the analyst has not produced a new signal, so that profitable trades are not cut short by signal staleness.

#### Acceptance Criteria

1. THE Portfolio_Manager SHALL NOT use the absence of a fresh analyst Signal as a reason to close a position.
2. WHEN the Portfolio_Manager evaluates an open position and no new analyst Signal exists for that symbol, THE Portfolio_Manager SHALL continue holding the position and evaluate it against the Entry_Contract instead.
3. THE Portfolio_Manager SHALL only close a position WHEN one of the following conditions is met: stop price is hit, target price is hit, an explicit CLOSE signal is received, or a Thesis_Invalidator condition is met.

### Requirement 3: DRIFTING State for Signal-Stale Positions

**User Story:** As a portfolio manager, I want positions without recent analyst signals to be marked as DRIFTING, so that I can distinguish between actively-monitored and passively-held positions without triggering exits.

#### Acceptance Criteria

1. WHEN an open position has no new analyst Signal since the trade was opened, THE Portfolio_Manager SHALL assign the DRIFTING_State to that position.
2. WHILE a position is in DRIFTING_State, THE Portfolio_Manager SHALL continue to enforce the Entry_Contract stop price and target price via the Price_Monitor.
3. WHILE a position is in DRIFTING_State, THE Portfolio_Manager SHALL NOT generate a CLOSE decision based on the DRIFTING_State alone.
4. WHEN a new analyst Signal is received for a position in DRIFTING_State, THE Portfolio_Manager SHALL remove the DRIFTING_State and resume normal signal-informed evaluation.
5. THE Portfolio_Manager SHALL include the DRIFTING_State label in portfolio snapshots and position health reports so that operators can see which positions are drifting.

### Requirement 4: Analyst Signals Advisory After Entry

**User Story:** As a portfolio manager, I want analyst signals to serve as advisory context for open positions rather than authoritative exit commands, so that the PM retains decision authority anchored to the trade thesis.

#### Acceptance Criteria

1. WHILE a position is open, THE Portfolio_Manager SHALL treat new analyst Signals as advisory context that informs but does not override the Entry_Contract.
2. WHEN a new analyst Signal contradicts the Entry_Contract direction (e.g., LONG position receives a SHORT signal), THE Portfolio_Manager SHALL flag the contradiction for Reversal_Close_Review rather than immediately closing the position.
3. WHEN a new analyst Signal confirms the Entry_Contract direction, THE Portfolio_Manager SHALL use the Signal to inform Maintenance_Review actions such as tightening the stop or raising the target.
4. THE Portfolio_Manager SHALL log each cycle whether analyst Signals were used in an advisory or authoritative capacity for audit purposes.

### Requirement 5: Thesis Invalidation Exit Logic

**User Story:** As a portfolio manager, I want positions to be exited when the original thesis is provably broken, so that the system protects capital on genuine invalidation rather than on noise.

#### Acceptance Criteria

1. WHEN a Thesis_Invalidator condition is met for an open position, THE Portfolio_Manager SHALL initiate a Reversal_Close_Review for that position.
2. THE Price_Monitor SHALL evaluate each Thesis_Invalidator's structured fields (`type`, `reference`, `confirmation`, `lookback_bars`) programmatically against current market data on every price update cycle (currently every 60 seconds).
3. WHEN the Price_Monitor detects a Thesis_Invalidator breach confirmed per the invalidator's `confirmation` method and `lookback_bars` window, THE Price_Monitor SHALL emit a thesis_invalidation trigger with the specific invalidator object that was breached.
4. IF multiple Thesis_Invalidators are defined for a trade, THEN THE Portfolio_Manager SHALL treat any single invalidator breach as sufficient to trigger Reversal_Close_Review.

### Requirement 6: Maintenance Review System

**User Story:** As a portfolio manager, I want a default maintenance review mode for open positions, so that the system can adjust stops, targets, and size without prematurely closing trades.

#### Acceptance Criteria

1. THE Portfolio_Manager SHALL use Maintenance_Review as the default review mode for all open positions during each PM decision cycle (currently every 15–30 minutes depending on time of day).
2. WHEN performing a Maintenance_Review, THE Portfolio_Manager SHALL produce one of the following actions: hold, tighten stop, raise target, or trim partial.
3. THE Maintenance_Review SHALL NOT produce a CLOSE action for any position.
4. WHEN performing a Maintenance_Review, THE Portfolio_Manager SHALL reference the Entry_Contract, current price, current indicators, and any advisory analyst Signals.
5. THE Position_Health_Monitor SHALL feed its health assessments into the Maintenance_Review as additional context, running on an hourly cadence.

### Requirement 7: Reversal / Close Review System

**User Story:** As a portfolio manager, I want a separate, higher-threshold review mode for closing positions, so that the system only exits when the evidence clearly warrants it.

#### Acceptance Criteria

1. THE Portfolio_Manager SHALL invoke Reversal_Close_Review only WHEN one of the following event-driven triggers is met: a Thesis_Invalidator is breached, opposing analyst evidence crosses a configurable threshold, or an explicit CLOSE or REVERSE condition is received. Reversal_Close_Review does not run on a periodic cadence; it is triggered exclusively by these events.
2. WHEN performing a Reversal_Close_Review, THE Portfolio_Manager SHALL evaluate the Entry_Contract thesis against current market conditions and produce one of: close full, close partial, or escalate to hold with tightened stop.
3. THE Portfolio_Manager SHALL log the specific trigger that caused the Reversal_Close_Review for post-trade audit.
4. WHILE no Reversal_Close_Review trigger condition is met, THE Portfolio_Manager SHALL NOT produce a CLOSE decision for the position.
5. WHERE a PM_Profile defines a configurable opposing-evidence threshold, THE Portfolio_Manager SHALL use that threshold to determine when opposing signals warrant Reversal_Close_Review.

### Requirement 8: Entry Contract Survives Across Decision Cycles

**User Story:** As a portfolio manager, I want the entry contract to persist across all decision cycles for the lifetime of the trade, so that the thesis anchor is never lost due to agent restarts or memory rotation.

#### Acceptance Criteria

1. THE Entry_Contract SHALL be stored on the Trade database record, not in volatile AgentMemory.
2. WHEN the Portfolio_Manager loads an open position for evaluation, THE Portfolio_Manager SHALL retrieve the Entry_Contract from the Trade record.
3. IF the Entry_Contract fields are missing from a Trade record (e.g., legacy trades opened before this feature) and the Trade record contains stop_price and target_price, THEN THE Portfolio_Manager SHALL construct a best-effort Entry_Contract using stop_price as the stop anchor, target_price as the target anchor, and reason_entry as the thesis narrative.
4. IF a legacy Trade record contains stop_price but not target_price (or vice versa), THEN THE Portfolio_Manager SHALL construct a partial Entry_Contract from the available fields and derive a default Thesis_Invalidator from the stop_price using the structured schema.
5. IF a legacy Trade record contains neither stop_price nor target_price, THEN THE Portfolio_Manager SHALL fall back to the existing signal-based evaluation for that position.
6. WHEN the Portfolio_Manager constructs a best-effort Entry_Contract for a legacy trade, THE Portfolio_Manager SHALL log a warning identifying the trade and which fields were missing or inferred.
