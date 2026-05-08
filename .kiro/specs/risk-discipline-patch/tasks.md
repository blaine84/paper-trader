# Implementation Plan: Risk Discipline Patch

## Overview

Implement the smallest high-impact risk discipline patch:

1. Fail closed when new entries lack a resolved setup type.
2. Add anti-thrash controls to routine maintenance stop updates: monotonic tightening, minimum meaningful delta, and per-trade cooldown.

## Tasks

- [ ] 1. Add setup type resolver and missing-setup rejection
  - [ ] 1.1 Add `_resolve_setup_type(decision, signal)` helper to `agents/portfolio_manager.py`
    - Resolve from decision setup fields first, then analyst signal fallback
    - Strip whitespace
    - Return `None` for missing/empty setup
    - _Requirements: 1.1, 2.1, 2.2, 2.3_

  - [ ] 1.2 Use resolver in `_run_gate_pipeline()`
    - Replace ad hoc setup extraction
    - If setup is missing, append setup-quality gate rejection note and return `False`
    - Ensure existing caller logs `gate_rejected` with `missing setup_type`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ] 1.3 Reuse normalized setup in related entry paths where practical
    - Entry timing gate
    - LessonRegistry input
    - Confidence adjustment
    - Strategy multiplier lookup
    - _Requirements: 2.4, 2.5_

- [ ] 2. Test missing setup behavior
  - [ ] 2.1 Add tests for missing setup fail-closed
    - Decision setup missing and signal setup missing rejects
    - Whitespace-only setup rejects
    - Rejection note contains `missing_setup_type`
    - _Requirements: 1.1, 1.2, 7.1_

  - [ ] 2.2 Add fallback test for valid analyst setup
    - Decision setup missing but signal setup valid proceeds to setup-quality evaluation
    - _Requirements: 2.1, 2.4_

- [ ] 3. Add StopAuthority anti-thrash helpers
  - [ ] 3.1 Add maintenance classification helper
    - Identify routine maintenance updates by source agent / stop role / reason
    - Ensure initial stop creation is excluded
    - _Requirements: 3.5, 6.1_

  - [ ] 3.2 Add monotonic tightening helper
    - LONG: new stop must be greater than current stop
    - SHORT: new stop must be less than current stop
    - Equal values are no-op/reject
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ] 3.3 Add minimum stop delta helper
    - Use max of 0.25% current price and 0.25× ATR when available
    - Fall back to 0.1% of current stop if current price unavailable
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ] 3.4 Add stop update cooldown lookup
    - Query latest `stop_update_accepted` for same trade_id
    - Default cooldown 15 minutes
    - _Requirements: 5.1, 5.2, 5.4_

- [ ] 4. Enforce anti-thrash rules in `apply_stop_update()`
  - [ ] 4.1 Reject non-monotonic routine maintenance updates
    - Do not mutate Trade row
    - Log `stop_update_rejected` with reason payload
    - _Requirements: 3.1, 3.2, 3.4, 7.6_

  - [ ] 4.2 Reject or skip tiny routine maintenance updates
    - Do not mutate Trade row
    - Do not emit `stop_update_accepted`
    - Log observable reason
    - _Requirements: 4.4, 4.5, 7.6_

  - [ ] 4.3 Reject routine maintenance updates inside cooldown
    - Do not mutate Trade row
    - Log cooldown reason
    - _Requirements: 5.3, 5.5, 7.6_

  - [ ] 4.4 Preserve emergency/exit behavior
    - Initial stops unaffected
    - Price-monitor stop exits unaffected
    - Target exits unaffected
    - Reversal/Close full/partial close unaffected
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 5. Test StopAuthority anti-thrash behavior
  - [ ] 5.1 Add monotonic tests for long and short trades
    - Long lower stop rejects
    - Long higher stop accepts when threshold/cooldown pass
    - Short higher stop rejects
    - Short lower stop accepts when threshold/cooldown pass
    - _Requirements: 3.1, 3.2, 7.2_

  - [ ] 5.2 Add no-op/equal stop test
    - Equal stop does not create accepted event
    - _Requirements: 3.3_

  - [ ] 5.3 Add tiny-delta tests
    - Below threshold rejects/skips
    - Above threshold accepts when monotonic and cooldown pass
    - _Requirements: 4.1, 4.4, 7.3_

  - [ ] 5.4 Add cooldown tests
    - First accepted update succeeds
    - Second update inside 15 minutes rejects/skips
    - Update after cooldown succeeds
    - _Requirements: 5.1, 5.2, 5.3, 7.4_

  - [ ] 5.5 Add initial-stop / emergency-path regression tests
    - Initial stop set unaffected
    - Stop-triggered close path unaffected
    - _Requirements: 6.1, 6.2, 7.5_

- [ ] 6. Verification and deployment
  - [ ] 6.1 Run focused local verification
    - `python -m py_compile agents/portfolio_manager.py utils/stop_authority.py`
    - Focused pytest files for missing setup and stop anti-thrash
    - _Requirements: 7.5_

  - [ ] 6.2 Run existing related regression tests if available
    - Stop authority integration/property tests
    - PM integration tests touching gate pipeline
    - _Requirements: 7.5_

  - [ ] 6.3 Commit and deploy through Mac → GitHub → Pi workflow
    - Commit from `/Users/cirrusclaude/projects/paper-trader`
    - Pull on `/home/blaine/paper-trader`
    - Restart service after market/when safe
    - _Requirements: 7.6_

  - [ ] 6.4 Next-day audit query
    - Count accepted/rejected stop updates by trade/symbol/profile
    - Confirm missing setup rejections are visible
    - Confirm no high-frequency stop churn recurs
    - _Requirements: 7.6_
