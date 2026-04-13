# Implementation Plan: Daily Review Agent

## Overview

Implement a post-market Daily Review Agent that synthesizes the full trading day into a narrative journal entry. The agent follows a two-phase approach: deterministic summary (pure Python/SQL) then LLM narrative generation. Includes orchestrator scheduling, web API endpoint, and Journal tab UI. Tasks build incrementally, referencing existing patterns from `agents/reviewer.py`, `agents/bookkeeper.py`, `orchestrator.py`, `web/app.py`, and `utils/llm.py`.

## Tasks

- [x] 1. Create the Daily Review agent module with core data-gathering functions
  - [x] 1.1 Create `agents/daily_review.py` with `run(engine)` entry point and `gather_trade_performance(engine, today)` function
    - Query `trades` (closed today), `daily_log`, and `positions` tables
    - Compute total_trades, wins, losses, total_pnl, total_pnl_pct, realized_pnl, unrealized_pnl, net_daily_change
    - Compute per_profile breakdowns (conservative, moderate, aggressive)
    - Identify best_trade and worst_trade by pnl_pct
    - Compute setup_breakdown counts and outcomes
    - Handle no-trades case with `no_trades=true` flag
    - Follow the same `get_session(engine)` pattern as `agents/bookkeeper.py`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ]* 1.2 Write property test for trade performance aggregation (Property 1)
    - **Property 1: Trade performance aggregation correctness**
    - Generate arbitrary lists of trade dicts with varying profiles, P&L values, and setup types
    - Assert total_trades == len(input), wins == count(pnl > 0), losses == count(pnl <= 0)
    - Assert total_pnl == sum of all P&L, per_profile sums match totals, setup_breakdown counts match grouping
    - **Validates: Requirements 2.3, 2.5**

  - [ ]* 1.3 Write property test for best/worst trade identification (Property 2)
    - **Property 2: Best and worst trade identification**
    - Generate non-empty lists of trades with distinct pnl_pct values
    - Assert best_trade.pnl_pct == max(pnl_pct), worst_trade.pnl_pct == min(pnl_pct)
    - **Validates: Requirements 2.4**

  - [ ]* 1.4 Write property test for realized vs unrealized P&L separation (Property 3)
    - **Property 3: Realized vs unrealized P&L separation**
    - Generate combinations of closed trades and open positions
    - Assert realized_pnl == sum(closed P&L), unrealized_pnl == sum(open unrealized), net_daily_change == realized + unrealized
    - **Validates: Requirements 2.7**

- [x] 2. Implement git commit parser and categorizer
  - [x] 2.1 Implement `gather_git_commits(since_date)` and `categorize_commit(message, files)` in `agents/daily_review.py`
    - Use `subprocess.run(["git", "log", "--since=...", "--format=%H|%an|%aI|%s", "--name-only"])` to get commits
    - Parse output into list of commit dicts with hash, author, timestamp, message, files, category
    - Implement deterministic categorization based on file paths and message keywords per the design table
    - Handle git unavailable / not a repo gracefully (log warning, return empty with `no_commits=true`)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 2.2 Write property test for git log parsing round-trip (Property 4)
    - **Property 4: Git log parsing round-trip**
    - Generate valid commit data (hash, author, ISO timestamp, message, file list)
    - Format as expected git log output, parse with the parser, assert fields match original
    - **Validates: Requirements 3.2**

  - [ ]* 2.3 Write property test for commit categorization valid output (Property 5)
    - **Property 5: Commit categorization valid output**
    - Generate arbitrary commit message strings and file path lists
    - Assert `categorize_commit` returns exactly one of: agent_logic, risk_management, infrastructure, strategy, bugfix, other
    - **Validates: Requirements 3.3**

- [x] 3. Implement context gatherer and previous review loader
  - [x] 3.1 Implement `gather_agent_context(engine)` and `load_previous_review(engine, today)` in `agents/daily_review.py`
    - Read latest `market_context` from researcher, `selection_feedback` and `execution_feedback` from reviewer, analyst signals from agent_memory
    - Track missing sources in a list
    - Load previous day's review from agent_memory where agent="daily_review"
    - Query today's case library entries from `cases` table
    - Handle missing entries gracefully (proceed without, note in missing_sources)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.3_

- [x] 4. Implement deterministic summary builder
  - [x] 4.1 Implement `build_deterministic_summary(trade_perf, git_commits, agent_context, cases, previous_review)` in `agents/daily_review.py`
    - Assemble all gathered data into the structured summary dict per the design schema
    - Compute completeness section: boolean flags for each source, missing_sources list, confidence level (high/medium/low)
    - Pure data transformation — no LLM calls
    - Use empty defaults for any None component data, flag in completeness
    - _Requirements: 6.5, 7.4, 7.5_

  - [ ]* 4.2 Write property test for deterministic summary makes no LLM calls (Property 6)
    - **Property 6: Deterministic summary makes no LLM calls**
    - Generate valid combinations of trade performance, git commits, agent context, cases, previous review
    - Mock `call_llm` to raise if called, assert `build_deterministic_summary` completes without triggering it
    - **Validates: Requirements 6.5**

  - [ ]* 4.3 Write property test for completeness tracking accuracy (Property 8)
    - **Property 8: Completeness tracking accuracy**
    - Generate subsets of available agent context sources
    - Assert completeness booleans match availability, missing_sources lists unavailable ones
    - Assert confidence = "high" when all present, "medium" when some missing, "low" when most missing
    - **Validates: Requirements 7.4, 7.5**

- [x] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement LLM narrative generator and storage
  - [x] 6.1 Implement `generate_narrative(deterministic_summary, tier="medium")` in `agents/daily_review.py`
    - Define SYSTEM_PROMPT with rules for observational language, hedging on small samples, referencing specific trades, separating process quality from outcomes
    - Call `call_llm(system_prompt, user_prompt, tier="medium")` using the existing `utils/llm.py` abstraction
    - Parse response with `parse_json_response` from `utils/llm.py`
    - Merge narrative fields with deterministic summary to produce the final Daily_Review JSON
    - Handle LLM failure gracefully: log error, return review with deterministic data only
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.6, 9.1, 9.2_

  - [x] 6.2 Wire `run(engine)` to call all components and store the final review in agent_memory
    - Call gather_trade_performance, gather_git_commits, gather_agent_context, load_previous_review
    - Call build_deterministic_summary, then generate_narrative
    - Store in agent_memory with agent="daily_review", symbol=today's date, key="daily_review"
    - Check for existing review on same date — skip storage if exists (no-overwrite)
    - Follow the same try/except + log pattern as `agents/reviewer.py`
    - _Requirements: 6.4, 8.1, 8.2, 1.4_

  - [ ]* 6.3 Write property test for output structure completeness (Property 7)
    - **Property 7: Output structure completeness**
    - Generate valid deterministic summaries, mock LLM to return valid narrative JSON
    - Assert resulting Daily_Review contains all required top-level fields: market_summary, trade_performance, git_changes, correlations, lessons_learned, process_quality, outlook, watchouts, completeness
    - **Validates: Requirements 6.1, 6.3**

  - [ ]* 6.4 Write property test for no-overwrite on duplicate date (Property 9)
    - **Property 9: No-overwrite on duplicate date**
    - Seed agent_memory with an existing review for a date
    - Run the agent for the same date, assert the original review is unchanged
    - **Validates: Requirements 8.2**

- [x] 7. Integrate with orchestrator scheduling
  - [x] 7.1 Add `run_daily_review` function and 4:30 PM ET cron job to `orchestrator.py`
    - Import `agents.daily_review as daily_review`
    - Add `run_daily_review()` function following the same error-handling pattern as `run_post_market()`
    - Add scheduler job: `CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone="America/New_York")`
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ]* 7.2 Write unit tests for orchestrator integration
    - Test cron trigger configuration matches weekday 4:30 PM ET
    - Mock `daily_review.run` to raise, verify orchestrator logs error and continues
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 8. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement web app API endpoint and Journal tab UI
  - [x] 9.1 Add `/api/journal` endpoint to `web/app.py`
    - Query agent_memory where agent="daily_review" and key="daily_review"
    - Sort by date descending (using the symbol field which stores the date)
    - Support `?page=1&per_page=10` query parameters with defaults
    - Parse each stored JSON value, include date from symbol field
    - Return at most per_page entries at the correct offset
    - Handle invalid pagination params by defaulting to page=1, per_page=10
    - Handle DB errors by returning empty list with 200 status
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ]* 9.2 Write property test for journal API sort order (Property 10)
    - **Property 10: Journal API sort order**
    - Seed agent_memory with Daily_Review entries at varying dates
    - Assert `/api/journal` returns entries in strictly descending date order
    - **Validates: Requirements 10.2**

  - [ ]* 9.3 Write property test for journal API pagination (Property 11)
    - **Property 11: Journal API pagination**
    - Generate positive integer page and per_page values, seed journal entries
    - Assert response contains at most per_page entries at the correct offset slice
    - **Validates: Requirements 10.3**

  - [x] 9.4 Add Journal tab to `web/templates/index.html`
    - Add "Journal" tab to the main navigation tabs alongside existing tabs
    - Add `tab-journal` content div with expandable date-labeled cards
    - Render sections: market summary, trade performance, git changes, correlations, lessons learned, outlook, watchouts, process quality, completeness
    - Omit missing sections rather than showing empty placeholders
    - Add `loadJournal()` JS function that fetches from `/api/journal`
    - Add "Load More" button for pagination
    - Display empty state message when no journal entries exist
    - Wire into `switchTab()` function
    - _Requirements: 10.5, 10.6, 10.7, 10.8, 10.9, 10.10_

  - [ ]* 9.5 Write unit tests for journal API and UI
    - Test `/api/journal` returns correct JSON structure with seeded data
    - Test empty state returns empty list
    - Test pagination defaults on invalid params
    - Verify index.html contains Journal tab
    - _Requirements: 10.1, 10.4, 10.5, 10.10_

- [x] 10. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document (Properties 1–11)
- Unit tests validate specific examples and edge cases
- The project uses Hypothesis for property-based testing (see `tests/test_edge_score.py`)
- All LLM calls use `utils/llm.py` with `tier="medium"` for zero-cost local inference
- Storage follows the same `agent_memory` pattern as `meta_reviewer` weekly reviews
