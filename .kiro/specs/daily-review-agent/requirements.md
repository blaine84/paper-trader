# Requirements Document

## Introduction

The Daily Review Agent produces a holistic end-of-day narrative journal every weekday after market close. Unlike the Reviewer (which scores individual trades) and the Bookkeeper (which tracks P&L numbers), this agent synthesizes the full day into a cohesive story: what happened in the markets, how the system performed, what code changes were deployed, and what lessons emerged. It ties together trade performance with system changes to build an evolving institutional journal.

## Glossary

- **Daily_Review_Agent**: The new agent that generates end-of-day narrative reviews combining trade performance, git history, and lessons learned
- **Daily_Review**: The structured output document produced by the Daily_Review_Agent for a single trading day
- **Git_Change_Summary**: A structured summary of git commits from the previous trading day, including files changed, commit messages, and categorization of changes
- **Trade_Performance_Summary**: An aggregation of the day's closed trades, open position changes, P&L, win/loss counts, and per-profile breakdowns
- **Orchestrator**: The APScheduler-based system that schedules and runs all agents on market-hours cron triggers
- **Agent_Memory**: The shared persistent key-value store (agent_memory table) used by agents to read and write context for other agents
- **LLM**: The language model abstraction layer supporting OpenAI, Anthropic, Mistral, and Ollama via tiered routing (high/medium/low)
- **Web_App**: The Flask-based web dashboard (web/app.py) that serves the Paper Trader UI and exposes API endpoints for data retrieval

## Requirements

### Requirement 1: Scheduled Execution

**User Story:** As a trader, I want the Daily Review Agent to run automatically every weekday after market close, so that I have a fresh journal entry waiting for me each evening.

#### Acceptance Criteria

1. THE Orchestrator SHALL schedule the Daily_Review_Agent to run on weekdays (Monday through Friday) after the post-market phase completes
2. WHEN the post-market phase completes, THE Orchestrator SHALL invoke the Daily_Review_Agent after the Reviewer and Bookkeeper have finished
3. THE Daily_Review_Agent SHALL NOT run on weekends or market holidays
4. IF the Daily_Review_Agent encounters an error during execution, THEN THE Orchestrator SHALL log the error and continue normal operation without affecting other agents

### Requirement 2: Trade Performance Analysis

**User Story:** As a trader, I want the daily review to summarize the day's trading activity across all profiles, so that I can quickly understand how the system performed.

#### Acceptance Criteria

1. WHEN the Daily_Review_Agent runs, THE Daily_Review_Agent SHALL query all trades closed on the current trading day from the trades table
2. WHEN the Daily_Review_Agent runs, THE Daily_Review_Agent SHALL query the daily_log table for the current day's P&L summary
3. THE Daily_Review_Agent SHALL produce a Trade_Performance_Summary containing: total trades taken, wins, losses, total P&L, and per-profile breakdowns (conservative, moderate, aggressive)
4. WHEN closed trades exist for the day, THE Daily_Review_Agent SHALL identify the best and worst trades by P&L percentage
5. WHEN closed trades exist for the day, THE Daily_Review_Agent SHALL summarize which setup types were traded and their outcomes
6. WHEN no trades were closed on the current day, THE Daily_Review_Agent SHALL note the absence of trading activity and summarize open position changes instead
7. THE Daily_Review_Agent SHALL distinguish realized P&L from closed trades, unrealized P&L from open positions, and net daily change in the Trade_Performance_Summary

### Requirement 3: Git Commit Review

**User Story:** As a trader, I want the daily review to detect and summarize code changes from the previous day, so that I can correlate system changes with trading performance.

#### Acceptance Criteria

1. WHEN the Daily_Review_Agent runs, THE Daily_Review_Agent SHALL retrieve git commits from the repository since the last trading day's close
2. THE Daily_Review_Agent SHALL parse each commit to extract the commit message, author, timestamp, and list of changed files
3. THE Daily_Review_Agent SHALL categorize each commit into one of: agent_logic, risk_management, infrastructure, strategy, bugfix, or other
4. IF no git commits exist since the last trading day, THEN THE Daily_Review_Agent SHALL note that no code changes were detected
5. IF the git repository is unavailable or the git command fails, THEN THE Daily_Review_Agent SHALL log a warning and proceed with the review without git data
6. WHEN categorizing commits, THE Daily_Review_Agent SHALL distinguish committed-but-not-deployed changes from deployed or live system changes when deployment metadata is available

### Requirement 4: Correlation of Changes and Performance

**User Story:** As a trader, I want the review to connect code changes with trading outcomes, so that I can understand the impact of system modifications.

#### Acceptance Criteria

1. WHEN both git changes and closed trades exist for the day, THE Daily_Review_Agent SHALL prompt the LLM to identify potential correlations between code changes and trading outcomes
2. WHEN an agent logic or strategy commit is detected, THE Daily_Review_Agent SHALL highlight the affected agent and note any observable performance differences
3. THE Daily_Review_Agent SHALL clearly label correlations as observational rather than causal
4. WHEN the number of data points for a correlation is small (fewer than five occurrences), THE Daily_Review_Agent SHALL use hedging language and explicitly note the limited sample size

### Requirement 5: Lessons Learned Extraction

**User Story:** As a trader, I want the review to extract actionable lessons from the day, so that the system and I can improve over time.

#### Acceptance Criteria

1. WHEN the Daily_Review_Agent runs, THE Daily_Review_Agent SHALL prompt the LLM to extract up to five actionable lessons from the day's trading activity and case library entries
2. THE Daily_Review_Agent SHALL categorize each lesson as one of: signal_quality, execution, risk_management, strategy, or system
3. THE Daily_Review_Agent SHALL reference specific trades or cases when stating a lesson
4. WHEN the cases table contains new entries for the current day, THE Daily_Review_Agent SHALL incorporate case library lessons and conditions_for_success and conditions_to_avoid fields into the review
5. THE Daily_Review_Agent SHALL include supporting evidence and a suggested follow-up action for each extracted lesson

### Requirement 6: Narrative Journal Generation

**User Story:** As a trader, I want the review output to be a readable narrative journal entry, so that I can review it like a trading diary.

#### Acceptance Criteria

1. THE Daily_Review_Agent SHALL produce a Daily_Review containing these sections: market summary, trade performance, git changes, correlations, lessons learned, and outlook for the next trading day
2. THE Daily_Review_Agent SHALL use the LLM (via the call_llm function) at the configured tier to generate the narrative
3. THE Daily_Review_Agent SHALL return the Daily_Review as a structured JSON object with distinct fields for each section
4. THE Daily_Review_Agent SHALL store the Daily_Review in the Agent_Memory table with agent="daily_review" and key="daily_review" and the current date in the symbol field
5. THE Daily_Review_Agent SHALL generate a deterministic structured summary from database queries and git data before invoking the LLM for narrative generation
6. THE Daily_Review_Agent SHALL include a machine-readable watchouts field in the Daily_Review JSON containing flagged items for next-day preparation

### Requirement 7: Context Gathering from Existing Agents

**User Story:** As a trader, I want the daily review to incorporate context from other agents, so that the narrative reflects the full picture of the day.

#### Acceptance Criteria

1. WHEN the Daily_Review_Agent runs, THE Daily_Review_Agent SHALL read the latest market_context entry from the Agent_Memory table written by the Researcher
2. WHEN the Daily_Review_Agent runs, THE Daily_Review_Agent SHALL read the latest selection_feedback and execution_feedback entries from the Agent_Memory table written by the Reviewer
3. WHEN the Daily_Review_Agent runs, THE Daily_Review_Agent SHALL read the latest analyst signals from the Agent_Memory table to understand what signals were generated during the day
4. IF any Agent_Memory entry is missing, THEN THE Daily_Review_Agent SHALL proceed without that context and note its absence in the review
5. THE Daily_Review_Agent SHALL include a completeness section in the Daily_Review indicating which input sources were available and which were missing, along with a confidence indicator reflecting data completeness

### Requirement 8: Review Persistence and Retrieval

**User Story:** As a trader, I want daily reviews to be stored and retrievable, so that I can look back at past reviews and track patterns over time.

#### Acceptance Criteria

1. THE Daily_Review_Agent SHALL store each Daily_Review in the Agent_Memory table so that reviews are queryable by date
2. THE Daily_Review_Agent SHALL NOT overwrite a previously stored review for the same date
3. WHEN generating a new review, THE Daily_Review_Agent SHALL read the previous trading day's review from Agent_Memory to provide continuity and reference prior observations

### Requirement 9: Process Quality Assessment

**User Story:** As a trader, I want the daily review to evaluate process quality separately from financial outcomes, so that I can distinguish good decision-making from lucky results.

#### Acceptance Criteria

1. THE Daily_Review_Agent SHALL include a process quality section in the Daily_Review that evaluates adherence to strategy rules, risk limits, and execution discipline independently from P&L outcomes
2. THE Daily_Review_Agent SHALL present the process quality assessment in a separate section from the trade performance summary so that process evaluation is not conflated with financial results

### Requirement 10: Web App Journal Display

**User Story:** As a trader, I want to view daily review journal entries in the web dashboard organized by date, so that I can browse past reviews and track the evolution of trading insights over time.

#### Acceptance Criteria

1. THE Web_App SHALL expose an API endpoint at /api/journal that returns Daily_Review entries from the Agent_Memory table where agent="daily_review" and key="daily_review"
2. THE Web_App SHALL return journal entries sorted by date in descending order so that the most recent entry appears first
3. THE Web_App SHALL support a query parameter for pagination so that the client can request entries in pages of a configurable size
4. WHEN the /api/journal endpoint is called, THE Web_App SHALL parse each stored Daily_Review JSON value and include the date from the symbol field in the response
5. THE Web_App SHALL include a "Journal" tab in the main navigation alongside the existing tabs (Watchlist, Analysis, Positions, Trades, Performance, Agent Feedback, Decision Log, System Review, Daily P&L)
6. WHEN a user selects the Journal tab, THE Web_App SHALL display journal entries as a list of expandable date-labeled cards with the most recent entry visible first
7. THE Web_App SHALL render each journal entry with distinct sections for: market summary, trade performance, git changes, correlations, lessons learned, outlook, watchouts, process quality, and completeness
8. WHEN a journal entry is missing one or more narrative sections, THE Web_App SHALL omit the missing section from the rendered card rather than displaying empty placeholders
9. THE Web_App SHALL provide a mechanism for the user to load additional past journal entries beyond the initial page (infinite scroll or a "Load More" button)
10. IF no journal entries exist in the Agent_Memory table, THEN THE Web_App SHALL display an informative empty state message in the Journal tab
