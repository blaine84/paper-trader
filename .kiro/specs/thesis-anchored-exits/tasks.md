# Implementation Plan: Thesis-Anchored Exits

## Overview

Replace the Portfolio Manager's implicit "no signal = exit" behavior with a thesis-anchored decision framework. Every trade records an Entry Contract at open, and all subsequent hold/exit decisions reference this contract. A two-tier review system (Maintenance Review vs. Reversal/Close Review) replaces the current flat LLM decision loop. The Thesis Invalidation Engine in Price Monitor evaluates structured invalidator conditions against live market data every 60 seconds.

## Tasks

- [x] 1. Extend Trade model and PM profiles with Entry Contract fields
  - [x] 1.1 Add `thesis`, `setup_type`, and `invalidators` columns to the `Trade` model in `db/schema.py`
    - Add `thesis = Column(Text, nullable=True)` for the trade thesis narrative
    - Add `setup_type = Column(String(64), nullable=True)` for the analyst's setup classification
    - Add `invalidators = Column(Text, nullable=True)` for JSON array of structured invalidator objects
    - All columns nullable to preserve backward compatibility with existing trades
    - _Requirements: 1.2, 8.1_

  - [x] 1.2 Add `opposing_evidence_threshold` to each PM profile in `models/pm_profiles.py`
    - Add `"opposing_evidence_threshold": "moderate"` to conservative profile
    - Add `"opposing_evidence_threshold": "strong"` to moderate profile
    - Add `"opposing_evidence_threshold": "strong"` to aggressive profile
    - _Requirements: 7.5_

- [x] 2. Implement Entry Contract builder and persist at trade open
  - [x] 2.1 Create `build_entry_contract(decision, signal, stop, target)` function in `agents/portfolio_manager.py`
    - Extract thesis from `decision["rationale"]` combined with signal context
    - Extract setup_type from signal or decision fields
    - Parse invalidators from `signal["invalidation"]` into structured JSON objects with fields: `type`, `reference`, `confirmation`, `lookback_bars`
    - Fall back to stop-price-based default invalidator if signal lacks invalidation field, log a warning
    - Return `{"thesis": str, "setup_type": str, "invalidators": list[dict]}`
    - _Requirements: 1.1, 1.3, 1.4_

  - [ ]* 2.2 Write property test for Entry Contract completeness (Property 1)
    - **Property 1: Entry Contract completeness and invalidator structure**
    - Generate random decision dicts, signal dicts, stop prices, and target prices
    - Verify thesis is non-empty string, setup_type is non-empty string, invalidators is non-empty list with all required fields and valid values
    - **Validates: Requirements 1.1, 1.3**

  - [ ]* 2.3 Write property test for default invalidator fallback (Property 2)
    - **Property 2: Default invalidator fallback from stop price**
    - Generate random signals without invalidation field and random positive stop prices
    - Verify invalidators list contains at least one invalidator whose reference equals the string representation of the stop price
    - **Validates: Requirements 1.4**

  - [x] 2.4 Integrate `build_entry_contract` into `execute_trade()` for BUY/SHORT actions
    - Call `build_entry_contract` after existing validation, before DB commit
    - Set `trade.thesis`, `trade.setup_type`, `trade.invalidators` (JSON-serialized) on the Trade record
    - Preserve all existing entry logic (edge score, similarity, portfolio risk, trade validation)
    - _Requirements: 1.1, 1.2_

- [x] 3. Implement DRIFTING state detection
  - [x] 3.1 Create `detect_drifting(db, trade)` function in `agents/portfolio_manager.py`
    - Query `AgentMemory` for the latest analyst signal for `trade.symbol`
    - Compare signal timestamp against `trade.entry_time`
    - Return `True` if no signal exists after entry time
    - _Requirements: 3.1, 3.4_

  - [ ]* 3.2 Write property test for DRIFTING state detection (Property 4)
    - **Property 4: DRIFTING state detection round-trip**
    - Generate random entry times and sets of analyst signal timestamps
    - Verify `detect_drifting` returns True iff no signal timestamp is strictly after entry_time
    - **Validates: Requirements 3.1, 3.4**

  - [x] 3.3 Add DRIFTING label to `get_portfolio_for_profile()` output
    - For each open position, call `detect_drifting` and add `"drifting": True/False` to the position dict
    - _Requirements: 3.5_

- [x] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Thesis Invalidation Engine in Price Monitor
  - [x] 5.1 Create `evaluate_invalidators(trade, current_price, candle_data=None)` function in `agents/price_monitor.py`
    - Parse `trade.invalidators` from JSON
    - For each invalidator, resolve `reference` to a numeric value (literal price or live indicator like VWAP)
    - Check breach based on `type` (`price_below_level`: reference > current_price; `price_above_level`: reference < current_price)
    - Apply `confirmation` logic: `"tick"` = immediate, `"5m_close"` = candle close check using `lookback_bars`
    - Skip `structure_break` type (handled by LLM in Reversal Review)
    - Return list of breached invalidator objects
    - Handle errors gracefully: skip unresolvable references, log warnings, fall back to existing stop/target checks if JSON is malformed
    - _Requirements: 5.2, 5.3_

  - [ ]* 5.2 Write property test for invalidator evaluation engine (Property 5)
    - **Property 5: Invalidator evaluation engine**
    - Generate random lists of invalidator objects and random current prices
    - Verify correct breach detection for `price_below_level` and `price_above_level` with `tick` confirmation
    - Verify OR logic: if multiple invalidators defined and at least one breached, result is non-empty
    - **Validates: Requirements 5.2, 5.3, 5.4**

  - [x] 5.3 Integrate invalidator evaluation into `check_stops_and_targets()` in `agents/price_monitor.py`
    - After existing stop/target checks, evaluate invalidators for each open trade that has them
    - Emit `thesis_invalidation` triggers alongside existing `stop_loss` and `target_hit` triggers
    - Store thesis_invalidation triggers in AgentMemory for PM to read during Reversal/Close Review
    - _Requirements: 5.1, 5.3, 5.4_

- [x] 6. Implement Two-Tier Review System in Portfolio Manager
  - [x] 6.1 Create Maintenance Review prompt template and handler in `agents/portfolio_manager.py`
    - Define `MAINTENANCE_REVIEW_PROMPT` template that receives Entry Contract, current price, indicators, advisory signals, position health
    - Constrain output to `{"action": "hold|tighten_stop|raise_target|trim_partial", ...}`
    - Include Entry Contract thesis, setup_type, and invalidators in the prompt context
    - Include DRIFTING state label and position health assessments
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 6.2 Write property test for Maintenance Review output constraint (Property 6)
    - **Property 6: Maintenance Review output constraint**
    - Generate random review inputs (Entry Contracts, prices, indicators, signals)
    - Verify output action is always one of `{"hold", "tighten_stop", "raise_target", "trim_partial"}`
    - Verify output never contains close/close_full/close_partial/CLOSE
    - **Validates: Requirements 6.2, 6.3**

  - [x] 6.3 Create Reversal/Close Review prompt template and handler in `agents/portfolio_manager.py`
    - Define `REVERSAL_CLOSE_PROMPT` template that receives Entry Contract, breach details, current market conditions, opposing evidence
    - Output: `{"action": "close_full|close_partial|hold_tighten", ...}`
    - Only invoked when a trigger fires (thesis_invalidation, opposing signal, explicit CLOSE)
    - Log the specific trigger that caused the Reversal/Close Review
    - _Requirements: 7.1, 7.2, 7.3_

  - [ ]* 6.4 Write property test for Reversal Review trigger routing (Property 7)
    - **Property 7: Reversal Review trigger routing**
    - Generate random positions, signals, and profile thresholds
    - Verify Reversal/Close Review is invoked iff at least one trigger is present (invalidation breach, opposing signal meeting threshold, explicit CLOSE)
    - Verify opposing-evidence comparison correctly uses profile's `opposing_evidence_threshold`
    - **Validates: Requirements 4.2, 7.1, 7.5**

  - [ ]* 6.5 Write property test for Reversal/Close Review output constraint (Property 8)
    - **Property 8: Reversal/Close Review output constraint**
    - Generate random review inputs
    - Verify output action is always one of `{"close_full", "close_partial", "hold_tighten"}`
    - **Validates: Requirements 7.2**

- [x] 7. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Refactor PM decision loop to use two-tier review routing
  - [x] 8.1 Modify `run_profile()` in `agents/portfolio_manager.py` to route positions through two-tier review
    - Load all open positions with their Entry Contracts
    - Check for pending Reversal triggers (thesis_invalidation from AgentMemory, opposing signals)
    - For positions WITH a Reversal trigger → call Reversal/Close Review handler
    - For positions WITHOUT a Reversal trigger → call Maintenance Review handler
    - For NEW entries (no existing position) → use existing entry logic unchanged
    - Execute resulting decisions via `execute_trade()`
    - Log each cycle whether signals were used in advisory or authoritative capacity
    - _Requirements: 2.1, 2.2, 2.3, 3.3, 4.1, 4.2, 4.3, 4.4, 7.4_

  - [ ]* 8.2 Write property test for close-conditions invariant (Property 3)
    - **Property 3: Close-conditions invariant**
    - Generate random positions with valid Entry Contracts, random market conditions (prices, signals present/absent, DRIFTING or not)
    - Verify CLOSE decision produced only when: stop hit, target hit, explicit CLOSE signal, or Thesis_Invalidator met
    - Verify no CLOSE decision when none of these conditions hold
    - **Validates: Requirements 2.1, 2.2, 2.3, 3.3, 7.4**

- [x] 9. Implement legacy trade migration
  - [x] 9.1 Create `build_legacy_entry_contract(trade)` function in `agents/portfolio_manager.py`
    - If `trade.thesis` is already populated → return None
    - If `trade.stop_price` and `trade.target_price` exist → construct full Entry Contract with `reason_entry` as thesis, `"unknown"` as setup_type, stop-price default invalidator
    - If only `stop_price` exists → partial contract with stop-based invalidator
    - If neither exists → return None (fall back to signal-based evaluation)
    - Log a warning for each legacy trade migrated, identifying the trade and which fields were missing or inferred
    - _Requirements: 8.3, 8.4, 8.5, 8.6_

  - [ ]* 9.2 Write property test for legacy trade migration (Property 9)
    - **Property 9: Legacy trade migration**
    - Generate random Trade records with varying field presence (thesis, stop_price, target_price, reason_entry)
    - Verify correct contract or None based on available fields
    - Verify stop-based invalidator is present when stop_price exists
    - **Validates: Requirements 8.3, 8.4, 8.5**

  - [x] 9.3 Integrate legacy migration into `run_profile()` position loading
    - When loading open positions, check if Entry Contract fields are empty
    - If empty, call `build_legacy_entry_contract` and persist result to Trade record
    - _Requirements: 8.2, 8.3_

- [x] 10. Update Position Health Monitor with Entry Contract context
  - [x] 10.1 Modify `agents/position_health.py` to include Entry Contract data in health check
    - Include thesis, invalidators, and setup_type from the Trade record in position data sent to the health LLM
    - Include DRIFTING state label in position data
    - No changes needed to AgentMemory storage or PM reading pattern — health assessments already flow to PM via `health_text`
    - _Requirements: 3.5, 6.4, 6.5_

- [x] 11. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document using Hypothesis
- Unit tests validate specific examples and edge cases
- The design uses Python throughout — all implementation tasks use Python
- All existing entry logic (edge score, similarity, portfolio risk, trade validation) is preserved unchanged
- New columns are nullable to avoid breaking existing data; SQLAlchemy's `create_all` handles schema migration
